# reims-vgpu

[![License: GPL-2.0-or-later OR LGPL-3.0-or-later](https://img.shields.io/badge/License-GPL--2.0--or--later%20OR%20LGPL--3.0--or--later-blue.svg)](LICENSE) [![Discord](https://img.shields.io/badge/Discord-Join%20the%20community-5865F2?logo=discord&logoColor=white)](https://discord.gg/D2AM9mrDgs)

> **Beta.** reims-vgpu has largely stabilized. We're now working through differences between
> platforms and GPU drivers, so how well it runs can still depend on your host and its GPU.

reims-vgpu is a virtual GPU for macOS VMs. It gives macOS running in QEMU accelerated graphics,
through Metal on a Mac or Vulkan on Linux, using the GPU driver macOS already ships with. There's
nothing to install in the guest.

Alongside the GPU, we ship a QEMU build that uses it, and `reims-ctl`, a small tool that creates
and runs macOS VMs for you.

![reims-vgpu running an arm64 macOS 26 Tahoe guest desktop on an Apple Silicon host](assets/readme/reims-vgpu-macos-arm64-desktop.png)

*arm64 macOS 26 Tahoe guest on an Apple Silicon host.*

![reims-vgpu running an x86_64 macOS 26 Tahoe guest desktop on a Linux host](assets/readme/reims-vgpu-macos-x86-desktop.png)

*x86_64 macOS 26 Tahoe guest on a Linux host.*

## Getting started

We don't ship disk images: macOS comes straight from Apple. On an Apple silicon Mac, that's an
IPSW, or a VM you already made with macosvm or tart. On Linux, `reims-ctl` downloads the installer
from Apple and sets the VM up for you, or uses a disk that already has macOS on it.

We've tested arm64 macOS 13, 14, 15 and 26 on an M3 Max, and x86-64 macOS 26 on Linux with an RTX
5080 laptop GPU. [docs/qemu.md](docs/qemu.md) has the details.

### Install reims-ctl

Clone the repository (without `--recursive`: QEMU's own submodules are not needed) and run:

```sh
scripts/install-ctl.sh
```

It installs what the build needs (Homebrew on macOS; apt on Ubuntu and Debian, pacman on Arch and
CachyOS; Rust through rustup.rs), fetches `vendor/qemu`, builds the library and QEMU on it, installs
QEMU into `~/.local/share/reims-ctl/qemu`, and `reims-ctl` into `~/.cargo/bin`, with
[reims-util](https://github.com/steelbrain/reims-util) beside it on Linux. `vendor/qemu` is
fetched from its public repository (`steelbrain/qemu-reims-vgpu`), at the commit this checkout
pins, and Cargo fetches metal2vulkan from its public repository, at the commit the workspace's
`Cargo.toml` names. `reims-ctl doctor` checks the host and the install.

### x86-64 guest on Linux (KVM)

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

   Already have macOS on a disk? `--disk` uses it (copied, reflinked where the file system can, or
   in place), and `--opencore` the OpenCore image it booted with, with OVMF beside it, which keeps
   its Mac identity. `--installer` takes an installer of your own.

3. **On apple-gfx, set Metal's error mode in the guest**, once. A VM on reims-vgpu's own device,
   the default for macOS 26.4 and later, doesn't need it: skip this step. An older macOS runs on
   apple-gfx, which reims-ctl picks for it. On apple-gfx (`--graphics apple-gfx`), macOS's
   icon service crashes as soon as it starts without it, and a fresh install never gets a working
   desktop: either it stays on its wallpaper with no menu bar and no Dock, or it comes up gray,
   with a menu bar but no Dock. Since you can't count on a desktop yet, you set it from macOS
   Recovery:

   1. Turn on the VM's boot selection, so that OpenCore shows its picker: in reims-ctl, press
      `e` on the VM, switch **Boot selection** on and save (or run
      `reims-ctl update-opencore NAME --boot-selection`). It's off by default, and then the VM
      boots straight into macOS.
   2. Start the VM, and pick the Recovery entry in OpenCore's picker before it boots macOS (it
      waits 5 seconds; any arrow key stops the countdown, and Space shows the entry if it's
      hidden).
   3. In Recovery, open Utilities > Terminal and type this as one line:

      ```sh
      cd /Volumes/*Data/Library/Preferences && defaults write "$PWD/.GlobalPreferences" METAL_ERROR_MODE -int 5 && chmod 644 .GlobalPreferences.plist && reboot
      ```

      The VM restarts only if every step worked. If it doesn't, read the error: the setting
      wasn't saved, and macOS will keep crashing until it is. Don't drop the `chmod`: without it,
      macOS can't read the setting.
   4. Pick macOS in the picker. The desktop should come up. You can turn the boot selection off
      again the same way.

   If you already have a desktop, run this in its Terminal instead, then restart:

   ```sh
   sudo defaults write /Library/Preferences/.GlobalPreferences METAL_ERROR_MODE -int 5
   ```

   If a Dock icon stays blank afterwards, clear the Dock's icon cache:
   `rm -rf "$(getconf DARWIN_USER_CACHE_DIR)com.apple.dock.iconcache"; killall Dock`.
   [docs/qemu.md](docs/qemu.md) ("x86-64 macOS guests: Metal's error mode") has the details.

### arm64 guest on an Apple silicon Mac (HVF / vmapple)

QEMU's `vmapple` machine boots what Virtualization.framework makes, so a guest comes from an IPSW
or from an existing Virtualization.framework VM. There is no OpenCore step.

```sh
reims-ctl create mac --ipsw latest              # install the newest macOS this Mac supports (a 64G disk)
reims-ctl create mac --ipsw ~/Downloads/UniversalMac.ipsw   # or a given IPSW
reims-ctl create mac --bundle ~/VMs/macos13     # or import a macosvm or tart VM (an APFS clone)
reims-ctl start mac                             # a window opens for each guest display
```

`make vm VM=path/to/macosvm-vm` boots a macosvm VM on `make qemu`'s QEMU without `reims-ctl`
([docs/qemu.md](docs/qemu.md)).

### Day to day

`reims-ctl` with no command opens its interactive interface: create a VM, start it (a window for
each display), stop it (the guest shuts down, or is powered off after a timeout), edit its CPUs,
memory, displays, resolution, HiDPI, borderless windows, GPU, audio and logs, change its installer,
print the exact QEMU command it runs, and zip its logs with the host's details into Downloads for a
bug report.

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

Each guest display opens in a window the library owns (`crates/reims-vgpu-window`). **Click into the
guest's picture to capture the keyboard and the mouse:** shortcuts and system keys then go to the
guest instead of the host, and the pointer stays inside the picture until you release it, so you
can throw it at the guest's corners and edges without landing on another window. Keys reach the
guest by position, so the guest's own layout applies, and on Linux the Super key is the guest's ⌘.

| Host | While captured | Release |
|---|---|---|
| macOS (AppKit's window) | ⌘ shortcuts and system keys, ⌘Q, ⌘W, ⌘Tab and ⌘Space among them | **⌃⌥⌘** together |
| Linux (SDL3's window, on Wayland or X11) | the whole keyboard, which the window grabs, so the compositor's shortcuts (Super, Alt+Tab) reach the guest | **Ctrl+Alt+Super** together |

The title says when input is captured and how to release it. Losing focus releases capture too,
and gives the host back its system keys and its pointer. Uncaptured, ⌘ shortcuts drive the host's
menu on a Mac. Losing focus, releasing capture or closing the window sends a key-up for every key
still down, so nothing sticks. Closing a window powers the VM off.

Fn/Globe does not reach the guest (QEMU's USB keyboard cannot carry it), and there is no HDR output.
[`crates/reims-vgpu-window/features.md`](crates/reims-vgpu-window/features.md) lists everything the
window does: its sizing, HiDPI, pointer, scrolling and cursor, each entry with its guard.

### Environment overrides

Set in QEMU's environment (`reims-ctl set <vm> --env KEY=VALUE`). Every one is optional; the library
reads them once per process, and an unset or unparsable value is the default.

| Variable | Effect |
|---|---|
| `REIMS_VGPU_LOG=info` | the library's log lines to stderr, filtered by a level and `target=level` pairs (`warn,reims_vgpu_core::fifo=debug`); unset, nowhere. `reims-ctl set <vm> --log-level` sets it |
| `REIMS_VGPU_STATS=1` | a line of frame statistics a second per display |
| `REIMS_VGPU_WINDOW=0` | QEMU stays headless under `-display none` (QEMU reads it, not the library) |
| `REIMS_VGPU_WINDOW_BORDERLESS=1` | borderless windows that fill their screens |
| `REIMS_VGPU_WINDOW_SCREENS=1,0` | display i opens on the host screen the list's i-th entry names |
| `REIMS_VGPU_WINDOW_REFRESH=120` | the refresh rate of the modes offered to the guest |
| `REIMS_VGPU_WINDOW_IMPL=sdl` | macOS: SDL3's window in place of AppKit's |
| `REIMS_VGPU_STRICT=1` | the Vulkan platform: end the process at each difference from Metal's answer it would otherwise log |
| `REIMS_VGPU_EXACT` | the Vulkan platform: the runtime's exact forms in place of its fast ones, by field ([docs/accuracy-ledger.md](docs/accuracy-ledger.md)) |

The guest's mode is QEMU's `apple-gfx` property `resolution=` (`2560x1600`, `2560x1600@200` for a
density, or `host`; 1920x1080 at 1× by default), which `reims-ctl`'s resolution and HiDPI settings
set. [docs/api.md](docs/api.md) section 2 lists every setting and variable,
`REIMS_VGPU_CENSUS_DIR`'s diagnostic among them.

## Three pathways

| Pathway | Host | Guest | QEMU machine and device | Platform | Run with |
|---|---|---|---|---|---|
| **x86-64 macOS / Linux Vulkan** | x86-64 Linux (KVM) | x86-64 macOS | `q35` through OpenCore, PCI `reims-graphics-pci` with its option ROM and reims-vgpu's own macOS drivers (reims-ctl's default on macOS 26.4 and later), or `apple-gfx-pci` (`106b:eeee`) | **Vulkan** through metal2vulkan | `reims-ctl` |
| **arm64 macOS / macOS Metal** | Apple silicon Mac (HVF) | arm64 macOS | `vmapple`, MMIO `apple-gfx-mmio` | **Metal** | `reims-ctl`, or `make vm` |
| **arm64 macOS / macOS Vulkan** | Apple silicon Mac (HVF) | arm64 macOS | `vmapple`, MMIO `apple-gfx-mmio` | **Vulkan** through metal2vulkan, on MoltenVK | `reims-ctl`, installed with `GPU_ARM=vulkan scripts/install-ctl.sh` |

On a Mac, Metal is the product and the default, and `reims-ctl` runs on it. The Vulkan platform
builds there too, into `target/dist-vulkan`, so the smoke, the suite and a QEMU configured on that
tree can run on it beside the Metal one. `GPU_ARM=vulkan scripts/install-ctl.sh` installs
`reims-ctl`'s QEMU on it, and `GPU_ARM=metal` goes back.

- The library: `crates/`. Core holds all behavior, generic over its platform; beside it are the
  Metal and Vulkan platforms, a display's window, the Rust API and the C API
  ([docs/development.md](docs/development.md#crates)).
- QEMU: `vendor/qemu` tracks
  [`steelbrain/qemu-reims-vgpu@steelbrain/reims-vgpu`](https://github.com/steelbrain/qemu-reims-vgpu/tree/steelbrain/reims-vgpu):
  `apple-gfx` in C on the C API, which finds the library through pkg-config and links it
  statically.
- The option ROM: `firmware/`, the PCI device's UEFI driver, in C, which an x86-64 guest's
  firmware runs.
- The Vulkan platform's runtime: metal2vulkan's public repository, at the commit the workspace's
  `Cargo.toml` names. On macOS it runs on MoltenVK.
- VMs: `reims-ctl` (`crates/reims-ctl`) creates, boots and manages them.

## Developing

macOS already includes a paravirtual GPU driver named `AppleParavirtGPU.kext`, written against the
host side Apple ships as `ParavirtualizedGraphics.framework`. reims-vgpu replaces that framework.
It takes the guest's GPU command stream and runs it on Metal on a Mac, or on Vulkan on Linux
through [`metal2vulkan`](https://github.com/steelbrain/metal2vulkan), and can draw each guest
display into a window of its own. QEMU's `apple-gfx` devices link it, on a Mac as on Linux. There
is no guest driver to install. On x86-64, the OpenCore image reims-util makes carries one small
kext, ReimsVgpu.kext, which has macOS 26's driver use the GPU as its arm64 build does.

The guest drivers were written against Apple's framework, so what the guest and the host observe is
the contract: API results and refusals, host callbacks and their order, guest memory, MMIO reads,
interrupts, frames, GPU results, and where a run faults, dies or hangs. The codebase's own tests pin
it: the behaviour-lock suite's expected files record what the code does, and every change is
checked against them. [AGENTS.md](AGENTS.md) has the rules for changing it.

The library is the device `vmapple` and `apple-gfx-pci` guests draw through, its displays, its
IOSurface host device and its isolated GPU tasks. It draws frames with its own shaders, and under
process isolation it runs each guest task other than task 0 on threads of its own, in the host's
process. Its C API (`crates/reims-vgpu-c/include/reims_vgpu.h`, specified in
[docs/api.md](docs/api.md)) is its interface: `cargo xtask dist` writes the library, the header, a
pkg-config package and the device's EFI option ROM into `target/dist`, which QEMU and the suite's
runner both link. `scripts/install-ctl.sh` installs what it needs itself; working on the code needs
the following.

### Requirements

- An Apple silicon Mac. Every Rust binary is arm64 (`aarch64-apple-darwin`): rustc's arm64e target
  has no pointer authentication, and its binaries crash before `main` (`rust-toolchain.toml`).
- The toolchain `rust-toolchain.toml` pins: nightly-2026-11-03 with rustfmt and clippy, for rustc's
  parallel front end (`-Zthreads`, in `.cargo/config.toml`). rustup installs it on first use.
- `cargo-nextest`, which runs the tests in `cargo xtask check`
  (`cargo install --locked cargo-nextest`).
- cbindgen 0.29.4, for `cargo xtask header` (`cargo install --locked cbindgen --version 0.29.4`).
- `cargo-llvm-cov` and rustup's `llvm-tools` component, for `make coverage` only.
- Xcode's clang, for objc2's exception helper (which the `cc` crate builds), the C API's pins, the
  smoke program and the suite's runner. The library compiles its shaders from source at run time,
  so `make` needs no Metal toolchain; the Vulkan platform's crate, which `cargo build --workspace`
  and `cargo xtask check` build, does (`xcodebuild -downloadComponent MetalToolchain`).
- Homebrew's `lld` (`brew install lld`), whose `lld-link` and `ld.lld` link the EFI option ROM
  (`firmware/`) that `cargo xtask dist` builds with Xcode's clang.
- For the QEMU targets: the `vendor/qemu` submodule (`git submodule update --init vendor/qemu`) and
  QEMU's build dependencies (Homebrew's `meson`, `ninja`, `pkg-config`, `glib`, `pixman` and so
  on), with Homebrew's `gnutls`, which gives QEMU the AES that `vmapple`'s AES engine needs:
  `make qemu` configures QEMU with `--enable-gnutls`, and stops where it is missing.

The suite's host-dependent expected files are recorded on an Apple M3 Max with a 40-core GPU, macOS
27.0.1 (build 26A434).

On x86-64 Linux the library builds on the Vulkan platform alone, with no Mac and no Xcode. It needs
the toolchain above (cargo fetches metal2vulkan at the commit the workspace's `Cargo.toml` names),
CMake and a C compiler for SDL3, which the build compiles from its pinned source, `make` for
jemalloc, the allocator the library uses, which the build also compiles, and, for
`cargo xtask dist`, clang, lld and binutils.

### Building

```sh
make              # the library as a plain library (cargo xtask dist): target/dist
make smoke        # link it as QEMU does and run an isolated guest task
make test         # the behaviour-lock suite on the library
make bless        # record expected files for new or changed scenarios
make coverage     # the suite on an instrumented build; a line report per file
make qemu         # QEMU with apple-gfx on the library (cargo xtask dist), in build/qemu
make vm VM=path/to/macosvm-vm   # boot a macOS guest on it (docs/qemu.md)
make clean        # remove build/
```

The Makefile's header describes every target and its variables. `make` runs `cargo`, which you can
drive directly. [docs/development.md](docs/development.md) has the repository's layout, its tests,
the crates, `cargo xtask` and the build profiles and features.

If you have [kache](https://github.com/kunobi-ninja/kache) installed, run
`. scripts/internal/dev-env.sh` before cargo and your builds go through it, so a new worktree reuses
what other worktrees on your machine already compiled. It's optional: without kache, builds are
cargo's own. See "Building through kache" in docs/development.md for the setup.

On Linux, `cargo xtask dist` builds the library on the Vulkan platform into `target/dist-vulkan`.
The Vulkan platform's shaders are metallibs, which only Apple's compiler makes. A Mac compiles its
own as it builds; any other host copies those a Mac compiled, committed in
`crates/reims-vgpu-vulkan/metallibs`, once their `metallibs.toml` names each with the sha256 of the
commands and sources this tree compiles it from (`crates/reims-vgpu-vulkan/build.rs`).
`RVG_METALLIB_DIR=<dir>` takes them from `<dir>` instead.

```sh
cargo xtask metallibs         # on a Mac: compile them into crates/reims-vgpu-vulkan/metallibs
cargo xtask metallibs <dir>   # the same into <dir>, for RVG_METALLIB_DIR
```

A change to a `.metal` source of the Vulkan crate, or to the metal crate's `shaders.metal`, commits
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
