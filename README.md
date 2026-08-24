# reims-vgpu

[![License: LGPL-3.0-or-later](https://img.shields.io/badge/License-LGPL%203.0%20or%20later-blue.svg)](LICENSE) [![Discord](https://img.shields.io/badge/Discord-Join%20the%20community-5865F2?logo=discord&logoColor=white)](https://discord.gg/D2AM9mrDgs)

> **Alpha.** This project is early and under active development. The QEMU device ABI, boot scripts,
> crate layout, backend behavior, and supported host/guest pathways may change without a stable
> compatibility guarantee. Treat it as research-quality: useful for experimentation and bring-up,
> not a frozen virtualization product.

reims-vgpu is an experimental virtual GPU for macOS guests. It aims to let macOS running inside a
VM use accelerated graphics instead of a basic framebuffer, while keeping the guest operating system
unchanged.

macOS already includes a paravirtual GPU driver named `AppleParavirtGPU.kext`.
reims-vgpu provides the QEMU device that driver attaches to, then decodes the guest's GPU command
stream on the host and executes it through Vulkan, with shader translation handled
by [`metal2vulkan`](https://github.com/steelbrain/metal2vulkan). There is no custom macOS kext and
no guest driver to install.

Contributions are welcome. I am especially interested in collaborating with developers who want to
work on correctness, visual glitches, synchronization bugs, command-stream decoding, Vulkan
translation, and making more host/guest combinations reliable.

![reims-vgpu running an arm64 macOS 13 Ventura guest desktop on an Apple Silicon host](assets/readme/reims-vgpu-macos-arm64-desktop.png)

*arm64 macOS 13 Ventura guest on an Apple Silicon host.*

![reims-vgpu running an x86_64 macOS 13 Ventura guest desktop on a Linux host](assets/readme/reims-vgpu-macos-x86-desktop.png)

*x86_64 macOS 13 Ventura guest on a Linux host.*

## Supported pathways

The project targets the following host/guest/backend combinations. Agents pick the pathway their
unit of work is on.

| Pathway | Host | Guest | Device attach | Backend | Boot |
|---|---|---|---|---|---|
| **x86 macOS / Linux Vulkan** | Linux x86_64 (KVM) | x86_64 macOS Metal guest | PCI `reims-vgpu-pci` | host **Vulkan** via `metal2vulkan` | `vm/boot-x86.sh` |
| **arm64 macOS / macOS Vulkan** | Apple Silicon macOS (HVF) | arm64 macOS Metal guest (`vmapple`) | sysbus MMIO `reims-vgpu-mmio` | host **Vulkan** via `metal2vulkan` through MoltenVK | `vm/boot-arm64.sh` |

- QEMU device shims: `vendor/qemu` tracks
  [`steelbrain/qemu-reims-vgpu@host-reims-vgpu-vmapple`](https://github.com/steelbrain/qemu-reims-vgpu/tree/host-reims-vgpu-vmapple)
  (thin C — QOM/MMIO/IRQ/console/HostOps only)
- Composition/staticlib: `crates/reims-vgpu` (decode orchestration, device scheduling, QEMU ABI,
  and core-to-executor adapters)
- Semantic model: `crates/reims-vgpu-protocol` + `crates/reims-vgpu-core` (typed contract,
  generational resources, lifecycle/content authority, and immutable execution commands)
- Guest memory: `crates/reims-vgpu-paging` + `crates/reims-vgpu-memory`
- Vulkan executor: `crates/reims-vgpu-vulkan` (capabilities, topology policy, GPU sessions and
  execution)
- Wire layouts: `crates/reims-vgpu-wire` (derived serializer views/parsers; decode uses these as
  the layout authority for covered records)
- Vulkan translator dependency: public `steelbrain/metal2vulkan` Git crate. On macOS, the Vulkan
  host backend runs through MoltenVK.
- VM lifecycle: `vm/` (snapshot-revert; arm and x86 guest boot scripts)

## Getting started

This tree ships **boot scripts and the device**, not a ready-made macOS disk image. Guest disks,
firmware vars, and OpenCore blobs are private/gitignored under `vm/`. Pick a pathway, provision a
guest once, freeze a golden snapshot, then use the snapshot-revert boots for day-to-day work.
macOS 13 Ventura is the recommended guest release for bring-up.

### x86_64 guest on Linux (KVM)

1. **Host prep.** You need KVM (`/dev/kvm`), a working NVIDIA (or other) Vulkan stack for the product
   backend, and build deps for the in-tree QEMU (`scripts/qemu-build/qemu-build.sh --target x86_64`).

2. **Generate OpenCore, OVMF, and a guest disk with [OSX-KVM](https://github.com/kholia/OSX-KVM).**
   **macOS 13 is recommended**. Follow that project’s docs to fetch recovery media, build OpenCore,
   and install macOS under QEMU+KVM. The point of this step is only to produce a
   **working, post-Setup-Assistant guest** plus the usual OpenCore/OVMF pieces — not to stay on
   OSX-KVM’s long-term launcher.

3. **Drop the artifacts where this repo expects them** (paths are the defaults in `vm/boot-x86.sh`;
   override with env if you prefer):

   | Artifact | Default location |
   |---|---|
   | Guest system disk | `vm/disks/macos.img` |
   | OpenCore boot disk | `vm/disks/OpenCore.qcow2` |
   | OVMF code | `vm/ovmf/OVMF_CODE_4M.fd` |
   | OVMF vars template | `vm/ovmf/OVMF_VARS-1920x1080.fd` |

   Finish install in the guest: enable Remote Login, install your SSH key, turn off sleep/screensaver
   as you like. Host SSH is typically `localhost:2222` → guest `:22` (see `vm/boot-x86.sh`).

4. **Capture the first immutable snapshot.** Guests are organised into **rails** — one rail per guest
   OS line (`macos-11` … `macos-26`), each with a snapshot history of its own under
   `vm/disks/rails/<rail>/snapshots/`. Create the rail's directory, then from a clean guest state
   (logged in, network/SSH known-good) shut down cleanly while booting in capture mode:

   ```bash
   mkdir -p vm/disks/rails/macos-15
   vm/boot-x86.sh --rail macos-15 --capture --device vmware-svga
   # clean shutdown from inside the guest → new label under
   # vm/disks/rails/macos-15/snapshots/, and that rail's snapshots/current points at it
   ```

   Every later boot clones the selected rail's `snapshots/current` (COW when possible) and **throws
   the clone away** on exit, so wedges and hard kills never poison the golden image.

   Importing a guest built elsewhere is the same shape without the boot — drop
   `{macos.img,OpenCore.qcow2,OVMF_VARS.fd}` (plus `OVMF_CODE.fd` if that guest was installed under
   a different OVMF build) into `vm/disks/rails/<rail>/snapshots/base/`, `chmod 444` them, and
   `ln -sfn base vm/disks/rails/<rail>/snapshots/current`. Use `cp --reflink=auto` on btrfs and the
   import costs no disk.

5. **Day-to-day boots.**

   ```bash
   vm/boot-x86.sh --list-rails                  # what guest lines exist (* = default)
   vm/boot-x86.sh --rail macos-15 --list-snapshots

   # Console only (mainstream OSX-KVM-style VGA) while you debug the host stack
   vm/boot-x86.sh --testing --device vmware-svga

   # Product Reims VGPU device (needs in-tree QEMU + reims-vgpu Vulkan)
   scripts/qemu-build/qemu-build.sh --target x86_64
   vm/boot-x86.sh --testing --device reims-vgpu-pci --rail macos-15

   # Host-window screenshot on the Linux/Plasma host
   scripts/screenshot-when-kde-plasma-host/screenshot-when-kde-plasma-host.sh -o /tmp/screen.png
   ```

   Without `--rail` a boot follows `vm/disks/rails/current`; change it with
   `ln -sfn <rail> vm/disks/rails/current`. Neither `--rail` nor `--snapshot` repoints anything.

### arm64 guest on Apple Silicon (HVF / vmapple)

Arm bring-up is **in-tree**: Virtualization.framework via Homebrew **`macosvm`**, then QEMU’s
`vmapple` machine under HVF. There is no OSX-KVM step.

1. Install **`macosvm`**, and build the vendored QEMU:

   ```bash
   scripts/qemu-build/qemu-build.sh --target aarch64
   ```

2. Provision a guest from a UniversalMac IPSW with the project helpers in
   `scripts/vmapple-provision/`. The live bundle lives under `vm/guest/` (disk, aux, `vm.json` /
   ECID).

3. Configure the guest once: enable Remote Login, run `scripts/vmapple-guest-config/` for no-sleep
   settings, and optionally enable auto-login by hand in System Settings. Capture a golden under
   `vm/guest/rails/<rail>/snapshots/` with the snapshot helpers (`scripts/vmapple-snapshot/`, or
   `vm/boot-arm64.sh --rail <rail> --capture` once the disk is ready).

4. Boot:

   ```bash
   vm/boot-arm64.sh --testing --device reims-vgpu-mmio    # product
   vm/boot-arm64.sh --testing --device apple-gfx-mmio   # Apple ParavirtualizedGraphics A/B
   scripts/screenshot-when-macos-host/screenshot-when-macos-host.sh /tmp/screen.png
   ```

   Optional **performance ceiling** reference: the same guest under native VZ via `macosvm --gui`.

### After the first snapshot

- Prefer **`--testing`** for agent/measurement boots (time-bounded, always reverts).
- Use **`--interactive`** when you need an open-ended GUI session (still reverts unless you are in
  `--capture` mode).
- Say which rail a result came from. A number from `macos-11` and a number from `macos-26` are two
  measurements, not one — that separation is the whole reason snapshots are per-rail.
- Never commit disks, IPSWs, or OpenCore/OVMF runtime under `vm/`.
- Device work follows the ownership map in [`docs/architecture.md`](docs/architecture.md); the
  shipping staticlib is `crates/reims-vgpu` and the shims remain thin under `vendor/qemu`. Rebuild
  QEMU after product changes before claiming a live boot result.

### Environment overrides

Set on the boot command; every one is optional and every default is "let the device decide". The
full list and parser live in `crates/reims-vgpu-config`; `crates/reims-vgpu/src/env.rs` is a
compatibility re-export. Boolean switches accept the documented on/off spellings; numeric and mode
controls document their own domains beside their definitions.

| Variable | Effect |
|---|---|
| `REIMS_VGPU_GUEST_IMPORT=off` | Disable `VK_EXT_external_memory_host` guest-RAM imports and exercise the copying rails used when host-pointer import is unavailable. |
| `REIMS_VGPU_DRAW_LOG=on` | Verbose per-draw detail on top of the always-on failure log. |

An override can only **narrow** what the device does. There is no way to switch a rail *on* that the
host reported it cannot run: capability is measured from the device at startup, and asking a driver
for an extension it does not advertise fails device creation rather than degrading. On a host
without host-pointer import the copying rail is already selected, and the `vk_caps` line in
`/tmp/reims-vgpu-fail.log` names which check said so.

## Repo layout

```text
AGENTS.md           - concise operating constraints for agents
docs/architecture.md - crate ownership, semantic seam, and regression gates
crates/             - wire, protocol, paging/memory, semantic core, Vulkan executor, composition,
                      support crates, and the separate UEFI option ROM
scripts/            - host setup, VM lifecycle, screenshot, and diagnostic helpers
vendor/             - vendored QEMU submodule and patch record
vm/                 - VM launch/configuration glue; images are private/untracked
```

`crates/reims-vgpu-wire` holds zero-copy views and parsers for the Apple
paravirtualized GPU serializer format, derived from Apple's own encoder rather
than inferred from captures. `crates/reims-vgpu`'s `runtime::decode` maps those
views once into semantic values owned by `reims-vgpu-protocol` and
`reims-vgpu-core`. Resolved commands then cross the executor boundary without
raw object tags, unresolved task-local references, or Vulkan-native payloads.
See [`docs/architecture.md`](docs/architecture.md) for the maintained ownership
map and the invariants that keep memory-topology optimization separate from
resource lifetime and guest-visible behavior.


## License

Licensed under the [GNU Lesser General Public License v3.0 or later](LICENSE)
(`LGPL-3.0-or-later`).

Metal, macOS are trademarks of Apple Inc. reims-vgpu is an independent project and is not affiliated
with, sponsored by, or endorsed by Apple Inc.
