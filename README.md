# reims-vgpu

[![License: GPL-2.0-or-later OR LGPL-3.0-or-later](https://img.shields.io/badge/License-GPL--2.0--or--later%20OR%20LGPL--3.0--or--later-blue.svg)](LICENSE) [![Discord](https://img.shields.io/badge/Discord-Join%20the%20community-5865F2?logo=discord&logoColor=white)](https://discord.gg/D2AM9mrDgs)

> **Beta.** reims-vgpu has largely stabilized. We're now working through differences between
> platforms and GPU drivers, so how well it runs can still depend on your host and its GPU.

reims-vgpu is a virtual GPU for macOS VMs. It gives x86-64 macOS running in QEMU on Linux
accelerated graphics, on your GPU's Vulkan driver. The guest drives it with macOS drivers of our
own, which `reims-ctl` puts in the VM's OpenCore image, so there's nothing to install in the guest
yourself.

Alongside the GPU, we ship a QEMU build that uses it, and `reims-ctl`, a small tool that creates
and runs macOS VMs for you.

![reims-vgpu running an x86_64 macOS 26 Tahoe guest desktop on a Linux host](assets/readme/reims-vgpu-macos-x86-desktop.png)

*x86_64 macOS 26 Tahoe guest on a Linux host.*

![reims-vgpu running an arm64 macOS 26 Tahoe guest desktop on an Apple Silicon host](assets/readme/reims-vgpu-macos-arm64-desktop.png)

*arm64 macOS 26 Tahoe guest on an Apple Silicon host.*

## Getting started

We don't ship disk images: macOS comes straight from Apple. `reims-ctl` downloads the installer
from Apple and sets the VM up for you, or uses a disk that already has macOS on it.

You need an x86-64 Linux host. Our drivers load on macOS 26.4 and later for now; we've tested
macOS 26 on an RTX 5080 laptop GPU and a Radeon 780M. arm64 macOS guests on Apple silicon Macs aren't built
right now: they come back with Ambrosia, our own arm64 firmware. [docs/qemu.md](docs/qemu.md) has
the details.

### Install reims-ctl

Clone the repository (without `--recursive`: QEMU's own submodules are not needed) and run:

```sh
scripts/install-ctl.sh
```

It installs what the build needs (apt on Ubuntu and Debian, pacman on Arch and CachyOS; Rust
through rustup.rs), fetches `vendor/qemu`, builds the library on Vulkan and QEMU on it, installs
QEMU into `~/.local/share/reims-ctl/qemu`, and `reims-ctl` into `~/.cargo/bin`, with
[reims-util](https://github.com/steelbrain/reims-util) beside it. `vendor/qemu` is
fetched from its public repository (`steelbrain/qemu-reims-vgpu`), at the commit this checkout
pins, and Cargo fetches metal2vulkan from its public repository, at the commit the workspace's
`Cargo.toml` names. `reims-ctl doctor` checks the host and the install.

### Set up a VM

1. **Host prep.** You need KVM (`/dev/kvm`) and a Vulkan driver; `reims-ctl gpus` lists the GPUs a
   VM can be given. The guest's memory reaches the GPU through `/dev/udmabuf`, whatever your GPU,
   so you need to be able to open it: without it the guest can crash at higher resolutions, at 4K
   even at its desktop. That means being in the group that owns it (usually `kvm`, the same group
   as `/dev/kvm`); a desktop session may let you open it anyway, but a VM started over ssh needs
   the group. `scripts/install-ctl.sh` offers to add you, and `reims-ctl` warns when you can't.

   On an NVIDIA GPU, raising `vm.max_map_count` makes things faster: past a point the GPU gets
   copies of the guest's memory instead of the memory itself. `reims-ctl doctor` and the
   installer tell you the value that avoids the copies, if yours is lower. It's optional: the
   guest works at your distribution's default.

2. **Create a VM.** Run `reims-ctl`, press `n`, pick a macOS version and press Ctrl-S. It downloads
   the installer from Apple and makes the VM's OpenCore image and disk; then press `s` to start it.
   Or from a shell:

   ```sh
   reims-ctl create mac --macos tahoe --start              # about 1 GB; installs the newest Tahoe
   reims-ctl create mac --macos 26.6.2 --offline --start   # one exact release, 13 to 20 GB
   ```

   The VM boots into macOS Recovery. Open Disk Utility, erase the VM's disk (QEMU HARDDISK Media)
   as APFS, and pick Reinstall macOS; with the offline installer, run
   `sh /Volumes/Installer/install-macos.sh` in Terminal instead. Once macOS is installed,
   `reims-ctl eject mac` takes the installer out. Downloads are kept for your next VM, in
   `~/.local/share/reims-ctl/installers` (delete them there to get the space back, once your VMs'
   installers are ejected), and an interrupted one picks up where it stopped.

   Already have macOS on a disk? `--disk` uses it, copied or in place. The copy is instant where the
   disk and the VM's folder are on one file system that makes reflinks, such as btrfs or XFS;
   otherwise reims-ctl copies it in full, and says so, with its size, before it starts. Where the
   OpenCore image it booted with sits beside it as `opencore.qcow2`, as in a VM's folder, the VM's
   new image presents that same Mac; `--opencore` copies an image of yours instead, with OVMF beside
   it, which keeps its Mac too. With neither, or when you copy the disk of one of your reims-ctl
   VMs, which can then run beside it, the VM is a new Mac to that macOS, so it opens Setup Assistant
   once at its first boot; finish it in the window, since it won't shut down until you do.
   `--installer` takes an installer of your own.

### Day to day

`reims-ctl` with no command opens its interactive interface: create a VM, start it (its display
opens in a window), stop it (the guest shuts down, or is powered off after a timeout), edit its
CPUs, memory, resolution, borderless window, GPU, audio and logs, change its installer, print the
exact QEMU command it runs, and zip its logs with the host's details into Downloads for a bug
report.

```sh
reims-ctl list                  # the VMs and whether they run
reims-ctl set mac --memory 12G --resolution 1920x1080   # applies at the next start
reims-ctl command mac           # the QEMU command it runs, as a shell script
reims-ctl report mac            # logs, settings and the host's details, zipped, for a bug report
reims-ctl update                # pull the checkout and install again
reims-ctl update --channel development   # switch to the development branch first (stable is master)
```

`reims-ctl help` lists every command; every option takes `--name value` or `--name=value`. A running
VM keeps the QEMU it started with across an update.

### The window takes your keyboard shortcuts and your mouse

The guest's display opens in a window the library owns (`crates/reims-vgpu-window`). **Click into the
guest's picture to capture the keyboard and the mouse:** shortcuts and system keys then go to the
guest instead of the host, and the pointer stays inside the picture until you release it, so you
can throw it at the guest's corners and edges without landing on another window. Keys reach the
guest by position, so the guest's own layout applies, and the Super key is the guest's ⌘. While
captured, the window grabs the whole keyboard (SDL3's window, on Wayland or X11), so the
compositor's shortcuts, Super and Alt+Tab among them, reach the guest. **Ctrl+Alt+Super** together
releases it.

The title says when input is captured and how to release it. Losing focus releases capture too,
and gives the host back its system keys and its pointer. Losing focus, releasing capture or closing the window sends a key-up for every key
still down, so nothing sticks. Closing a window powers the VM off.

Fn/Globe does not reach the guest (QEMU's USB keyboard cannot carry it), and there is no HDR output.
[`crates/reims-vgpu-window/features.md`](crates/reims-vgpu-window/features.md) lists everything the
window does: its sizing, HiDPI, pointer, scrolling and cursor, each entry with its guard.

### Environment overrides

Set in QEMU's environment (`reims-ctl set <vm> --env KEY=VALUE`). Every one is optional; the library
reads them once per process, and an unset or unparsable value is the default.

| Variable | Effect |
|---|---|
| `REIMS_VGPU_LOG=info` | the library's log lines to stderr, filtered by a level and `target=level` pairs (`warn,reims_vgpu_graphics=debug`); unset, nowhere. `reims-ctl set <vm> --log-level` sets it |
| `REIMS_VGPU_WINDOW=0` | QEMU stays headless under `-display none` (QEMU reads it, not the library) |
| `REIMS_VGPU_WINDOW_BORDERLESS=1` | a borderless window that fills its screen |
| `REIMS_VGPU_STRICT=1` | end the process at each difference from Metal's answer the Vulkan platform would otherwise log |
| `REIMS_VGPU_EXACT` | the runtime's exact forms in place of its fast ones, by field ([docs/accuracy-ledger.md](docs/accuracy-ledger.md)) |

The window opens at the size of QEMU's `reims-graphics-pci` property `resolution=` (1920x1080 by
default), which `reims-ctl`'s resolution setting sets, and the guest follows the window as you
resize it. [docs/api.md](docs/api.md) section 2 lists every setting and variable.

## What's here

- The library: `crates/`. Core holds the behavior, generic over its platform; beside it are the
  Vulkan platform, which runs on metal2vulkan, the reims-graphics device's host side (`crates/reims-vgpu-graphics`), a display's window, host video on VA-API and NVDEC/NVENC,
  the Rust API and the C API ([docs/development.md](docs/development.md#crates)).
- The guest's side: `guest/`, the kexts that drive the device and the Metal bundle that turns the
  guest's Metal calls into its command stream. `guest/prebuilt` holds the built set, which
  reims-ctl puts in each VM's OpenCore image.
- QEMU: `vendor/qemu` tracks
  [`steelbrain/qemu-reims-vgpu@steelbrain/reims-vgpu`](https://github.com/steelbrain/qemu-reims-vgpu/tree/steelbrain/reims-vgpu):
  `reims-graphics-pci` in C on the C API, which finds the library through pkg-config and links it
  statically.
- The option ROM: `firmware/`, the device's UEFI driver, in C, which the guest's firmware runs.
  `firmware/prebuilt` holds the built ROM, which reims-ctl gives the device. `firmware/ambrosia`
  is the device's driver module for Ambrosia, which gives an arm64 guest's firmware an early
  console on the device, with our configuration of Ambrosia and the check of a guest's boot on it.
- The Vulkan platform's runtime: metal2vulkan's public repository, at the commit the workspace's
  `Cargo.toml` names.
- VMs: `reims-ctl` (`crates/reims-ctl`) creates, boots and manages them.

## Developing

The reims-graphics device (PCI `1234:5247`) and everything on both sides of it are ours. In the
guest, our kexts drive the device, and our Metal bundle turns macOS's Metal calls into the
device's command stream. On the host, QEMU's `reims-graphics-pci` hands that stream to the
library, which runs it on Vulkan through [`metal2vulkan`](https://github.com/steelbrain/metal2vulkan)
and draws the guest's display into a window of its own. The library also builds on a Mac, over
MoltenVK, for development and tests, and the guest's side is built there; VMs run on Linux.

The library's C API (`crates/reims-vgpu-c/include/reims_vgpu.h`, specified in
[docs/api.md](docs/api.md)) is its interface to QEMU: `cargo xtask dist` writes the library, the
header and a pkg-config package into `target/dist-vulkan`, which QEMU links.
[AGENTS.md](AGENTS.md) has the rules for changing the code. `scripts/install-ctl.sh` installs what
it needs itself; working on the code needs the following.

### Requirements

- A Linux host on x86-64 for the library and for VMs, and an Apple silicon Mac for the metallibs
  and the guest's kexts. Every Rust binary built on the Mac is arm64
  (`aarch64-apple-darwin`): rustc's arm64e target has no pointer authentication, and its binaries
  crash before `main` (`rust-toolchain.toml`).
- The toolchain `rust-toolchain.toml` pins: nightly-2026-11-03 with rustfmt and clippy, for rustc's
  parallel front end (`-Zthreads`, in `.cargo/config.toml`). rustup installs it on first use.
- `cargo-nextest`, which runs the tests in `cargo xtask check`
  (`cargo install --locked cargo-nextest`).
- cbindgen 0.29.4, for `cargo xtask header` (`cargo install --locked cbindgen --version 0.29.4`).
- On the Mac, Xcode's clang, for objc2's exception helper (which the `cc` crate builds), Xcode's
  Metal toolchain, with which the Vulkan platform's crate compiles its metallibs as it builds
  (`xcodebuild -downloadComponent MetalToolchain`), and Homebrew's `vulkan-loader` and
  `molten-vk`, the Vulkan the library runs on there. The guest's kexts build with Xcode and a
  macOS 26 KDK (`guest/Makefile`).
- clang and lld for the option ROM and the Ambrosia module (`firmware/`; on a Mac, Homebrew's
  `lld`).
- For `make qemu`: the `vendor/qemu` submodule (`git submodule update --init vendor/qemu`) and
  QEMU's build dependencies (`meson`, `ninja`, `pkg-config`, `glib`, `pixman` and so on).

On x86-64 Linux the library builds on the Vulkan platform alone, with no Mac and no Xcode. It needs
the toolchain above (cargo fetches metal2vulkan at the commit the workspace's `Cargo.toml` names),
CMake and a C compiler for SDL3, which the build compiles from its pinned source, and `make` for
jemalloc, the allocator the library uses, which the build also compiles.

### Building

```sh
make              # the library (cargo xtask dist), in target/dist-vulkan
make qemu         # qemu-system-x86_64 with reims-graphics-pci on the library, in build/qemu
make clean        # remove build/
make -C guest test     # the guest side's tests, on a Mac
make -C firmware       # the option ROM, in firmware/build
make -C firmware/ambrosia bundle   # the Ambrosia module and a bundle that starts it, in firmware/ambrosia/build
make -C firmware/ambrosia config-bundle GRAFT=... REIMS_GRAPHICS=... RGFXGATE=...
                                   # our configuration: the module, Graft and our kexts (arm64e builds)
```

The Makefile's header describes every target and its variables. `make` runs `cargo`, which you can
drive directly. [docs/development.md](docs/development.md) has the repository's layout, its tests,
the crates, `cargo xtask` and the build profiles and features.

If you have [kache](https://github.com/kunobi-ninja/kache) installed, run
`. scripts/internal/dev-env.sh` before cargo and your builds go through it, so a new worktree reuses
what other worktrees on your machine already compiled. It's optional: without kache, builds are
cargo's own. See "Building through kache" in docs/development.md for the setup.

The Vulkan platform's shaders are metallibs, which only Apple's compiler makes. A Mac compiles its
own as it builds; any other host copies those a Mac compiled, committed in
`crates/reims-vgpu-vulkan/metallibs`, once their `metallibs.toml` names each with the sha256 of the
commands and sources this tree compiles it from (`crates/reims-vgpu-vulkan/build.rs`).
`RVG_METALLIB_DIR=<dir>` takes them from `<dir>` instead.

```sh
cargo xtask metallibs         # on a Mac: compile them into crates/reims-vgpu-vulkan/metallibs
cargo xtask metallibs <dir>   # the same into <dir>, for RVG_METALLIB_DIR
```

A change to a `.metal` source of the Vulkan crate, `shaders.metal` among them, commits
the metallibs `cargo xtask metallibs` writes with it: until then a test of `cargo xtask check` fails
on the Mac (`the_committed_metallibs_are_this_trees`), and the build fails on Linux.

## License

The repository is free software under the GNU General Public License, version 2 or later, or the
GNU Lesser General Public License, version 3 or later, at the recipient's choice:
`GPL-2.0-or-later OR LGPL-3.0-or-later`, the workspace's `license` in `Cargo.toml`.
[LICENSE](LICENSE) says so, and `LICENSES/` holds the texts, the GPL's version 3 among them, since
the LGPL's version 3 adds its permissions to it.

QEMU (`vendor/qemu`) keeps QEMU's licenses. The crates the build takes keep their own:
metal2vulkan's, which the Vulkan platform links, and SDL3's sources, which the window crate builds
and links statically, among them.

## Legal

Metal and macOS are trademarks of Apple Inc. reims-vgpu is an independent project and is not
affiliated with, sponsored by, or endorsed by Apple Inc.
