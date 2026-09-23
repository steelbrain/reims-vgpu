# vm/ — macOS guests for the reims-vgpu pathways

`crates/reims-vgpu` runs three backend pathways over two guest rails. Pick the boot script for the
guest you are on and the QEMU/reims-vgpu backend for the host GPU path.

| Pathway | Script | Host accel | Backend | Typical device |
|---|---|---|---|---|
| x86 macOS / Linux Vulkan | `vm/boot-x86.sh` | KVM / OpenCore+OVMF | Vulkan via `metal2vulkan` | `reims-vgpu-pci` |
| arm64 macOS / macOS Metal | `vm/boot-arm64.sh` | HVF / `vmapple` | Metal-direct | `reims-vgpu-mmio` (product) or `apple-gfx-mmio` (Apple reference A/B) |
| arm64 macOS / macOS Vulkan | `vm/boot-arm64.sh` | HVF / `vmapple` | Vulkan via `metal2vulkan` through MoltenVK | `reims-vgpu-mmio` |

```bash
# arm64 macOS guest on Mac host (Metal or Vulkan/MoltenVK backend)
vm/boot-arm64.sh --testing                 # agent boot: GUI + serial-to-file, hard kill, reverts
vm/boot-arm64.sh --interactive             # human boot: GUI, no time limit, reverts
vm/boot-arm64.sh --device reims-vgpu-mmio --testing
vm/boot-arm64.sh --device apple-gfx-mmio --testing   # Apple-paravirt reference (arm only)

# x86 macOS guest on Linux host
vm/boot-x86.sh --testing
vm/boot-x86.sh --device reims-vgpu-pci --testing
vm/boot-x86.sh --device reims-vgpu-pci --testing --rail macos-11   # a specific guest OS line
```

Both scripts use a snapshot-revert lifecycle (testing vs interactive classes; testing hard kill).

### Host window system

`REIMS_VGPU_WINDOW_SYSTEM` decides which window system the host-owned window
opens on (`vm/window-system-env.sh`, applied by `vm/boot-x86.sh`):

- `auto` (default) — leave the environment as the caller set it, with the
  historical development fallbacks that invent `WAYLAND_DISPLAY`/`DISPLAY`
  when neither is present. Works on a Wayland desktop and on an X11 session.
- `x11` — require a non-empty `DISPLAY` and remove `WAYLAND_DISPLAY` and
  `WAYLAND_SOCKET` before QEMU/reims-vgpu start. winit 0.30 has no
  `WINIT_UNIX_BACKEND` and prefers Wayland whenever both are set, so the
  variables are what selects X11. A missing `DISPLAY` is a refusal.
- `wayland` — require `WAYLAND_DISPLAY` or `WAYLAND_SOCKET`.

Within an X11 session, `REIMS_VGPU_X11_WMLESS` decides *how* the window becomes
full-screen (Rust side, `host_window::present`):

- `0` (default) — winit's `Fullscreen::Borderless`, which on X11 is an EWMH
  request (`_NET_WM_STATE_FULLSCREEN`) and needs a window manager to honour it.
- `1` — require `REIMS_VGPU_FULLSCREEN` as well, and create the window
  override-redirect at the monitor's own rectangle, so no window manager is
  involved. This is the Reims OS appliance's session; a host with a window
  manager is unaffected because the switch is off there.

Contract test: `vm/tests/window-system-env-test.sh` (no VM, no display needed).

## Driving a guest, and sweeping every rail

A boot that sits at an idle Finder produces no rates, and almost every reading
this project takes is a rate. Two scripts supply the workload and the sweep:

```bash
vm/drive-desktop.sh          # host-driven workload against a guest already up
vm/rail-sweep.sh             # all six provisioned x86 rails, boot + drive + read
vm/rail-sweep.sh macos-15 macos-26        # or just these
```

`drive-desktop.sh` is the "three rounds of launching and quitting five
applications over ssh" that every validation reading this project quotes was
taken behind. It is committed rather than described because a rate is only
comparable across boots if the workload is. `ROUNDS` and `APPS` override it.

`rail-sweep.sh` boots each rail with `--testing`, drives it, snapshots
`/tmp/reims-vgpu-fail.log` to `/tmp/sweep-<rail>-fail.log`, and prints the same
presence list and route sums for each. A defect found on one guest OS line is
not a defect found on the pathway — `AGENTS.md` says not to generalise between
rails — so a regression sweep means all six.

Two things it will not do. It **refuses to start** while another boot or its
QEMU is live, or while host port 2222 is bound: QEMU's ssh `hostfwd` is a single
port, a second boot dies instantly on it, and the sweep would then wait out its
whole first-frame timeout and report a rail that never presented. Note that
`boot-x86.sh` outlives its own QEMU, so killing QEMU alone does not free the
port. And it does **not** rebuild between rails, because `boot-x86.sh` does:
editing source during a sweep gives the rails different binaries and destroys
the comparison the sweep exists for.

Add a route to the script's `KEYS`/`SUMS` when it becomes part of what a sweep
has to answer, so that every later sweep answers it too.

QMP is a per-boot unix socket under the run dir for that pathway (`scripts/qmp/qmp.py`).

**Networking** is typically QEMU SLIRP with SSH hostfwd. Guest disks, IPSWs, OpenCore blobs, and
runtime clones are **gitignored** - private and large; never commit them.

## Rails

A **rail** is one guest OS line — `macos-11` … `macos-26` — and each rail owns its own snapshot
history. Two coordinates select what boots, each with its own `current`, and neither selection
repoints anything:

```
vm/disks/rails/<rail>/snapshots/<label>/     # x86:   macos.img, OpenCore.qcow2, OVMF_VARS.fd
vm/disks/rails/<rail>/snapshots/current      #        (+ OVMF_CODE.fd when the guest needs its own)
vm/disks/rails/current -> <rail>             #        the rail a bare boot gets
vm/guest/rails/<rail>/snapshots/<label>/     # arm64: disk.img, aux.img.trimmed
vm/guest/rails/<rail>/vm.json                #        optional per-rail ECID
```

```bash
vm/boot-x86.sh --list-rails                        # * marks the default
vm/boot-x86.sh --rail macos-15 --list-snapshots
vm/boot-x86.sh --rail macos-15 --testing           # boot that rail's snapshots/current
vm/boot-x86.sh --rail macos-15 --snapshot base --testing
vm/boot-x86.sh --rail macos-15 --capture           # clean shutdown → new snapshot IN that rail
ln -sfn macos-15 vm/disks/rails/current            # change the default
```

Snapshots are per-rail because they are not comparable across rails: a macOS 15 disk and a macOS 11
disk share no history, and one flat namespace makes `current` mean "whichever guest was captured
last" — which is how a measurement gets attributed to the wrong OS.

**Importing a guest built elsewhere** needs no boot: drop its disk and firmware into
`rails/<rail>/snapshots/base/` under the names above, `chmod 444`, and point `snapshots/current` at
`base`. Copy with `cp --reflink=auto` on btrfs (or `cp -c` on APFS) and the import costs no disk.
Carry the guest's own `OVMF_CODE.fd` when it was installed under a different OVMF build — the NVRAM
in `OVMF_VARS.fd` only means anything to the code half that wrote it, and a snapshot that ships one
overrides `vm/ovmf/OVMF_CODE_4M.fd` for that boot.

## Layout (gitignored runtime)

Pathway-specific trees hold provisioned disks, rails, and per-boot clones. The run dir
(`vm/disks/run`, `vm/guest/run`) is shared across rails: clones are stamped and thrown away, and
`run/qmp.sock` is the one path every driver script resolves.

Stop a wedged guest with the pathway's shutdown helper when available
(`scripts/vmapple-shutdown` on arm); otherwise use QMP quit plus a process kill.

## macOS Tahoe installer with Reims from first boot

The Reims OS appliance owns the canonical OSX-KVM checkout:

https://github.com/felipeab10/reims-os

This component repository does not vendor OSX-KVM. Provisioning and appliance
lifecycle belong to reims-os. For isolated low-level development, provide an
external OSX-KVM checkout explicitly; downloaded Apple recovery media and guest
disks remain local and are ignored.

Prepare a Tahoe recovery image outside Git using an external checkout:

```bash
: "${OSX_KVM:?set OSX_KVM to an external OSX-KVM checkout}"
TAHOE="${TAHOE:-$PWD/.local/tahoe-installer}"
mkdir -p "$TAHOE"
python3 "$OSX_KVM/fetch-macOS-v2.py" --action download --shortname tahoe --os-type latest   --outdir "$TAHOE" --basename Tahoe
dmg2img -i "$TAHOE/Tahoe.dmg" "$TAHOE/Tahoe.img"
qemu-img create -f qcow2 "$TAHOE/macos.img" 256G
```

Start the installer with the Reims PCI device already attached:

```bash
DISKS_DIR="$TAHOE" \
DISK_MASTER="$TAHOE/macos.img" \
OPENCORE_MASTER="$OSX_KVM/OpenCore/OpenCore.qcow2" \
OVMF_VARS_MASTER="$OSX_KVM/OVMF_VARS-1920x1080.fd" \
INSTALL_MEDIA="$TAHOE/Tahoe.img" \
REIMS_VGPU_BACKEND=vulkan \
./vm/boot-x86.sh --interactive --device reims-vgpu-pci --rail macos-26
```
