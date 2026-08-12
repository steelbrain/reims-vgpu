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
