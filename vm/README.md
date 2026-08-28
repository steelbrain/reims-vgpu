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

Guest disks, IPSWs, OpenCore blobs, and runtime clones are **gitignored** - private and large;
never commit them.

## A dashboard for all of the above

`scripts/vgpu-dashboard/vgpu-dashboard.py` puts the boot classes, rails, snapshots, network modes,
USB device picker, env knobs, probes and the device's own live censuses on one local page. It drives
the scripts documented here rather than reimplementing them, so it cannot disagree with them about
what a spec means or which snapshot is current.

```sh
scripts/vgpu-dashboard/vgpu-dashboard.py            # prints a URL carrying a one-off token
scripts/vgpu-dashboard/vgpu-dashboard.py --selftest # read-only checks, no VM needed
```

Loopback-only and token-gated, because it execs QEMU. See that directory's README for what it
parses out of the fail log and which reading traps it implements.

## Networking

`vm/boot-x86.sh` defaults to **`NET=bridge`**: the guest joins the host bridge named by `BRIDGE`
(default `virbr0`, libvirt's `default` network) through `qemu-bridge-helper` and takes a real DHCP
lease, so it is a peer on the bridge's subnet rather than a client behind SLIRP's NAT. host→guest,
guest→host and guest→guest all work, which SLIRP structurally cannot do.

| Mode | Guest address | Reach it at | Needs |
|---|---|---|---|
| `NET=bridge` (x86 default) | DHCP lease on `$BRIDGE` | `vm/guest-ip.sh` | `tun`, a bridge, a privileged `qemu-bridge-helper`, an `allow` line in `/etc/qemu/bridge.conf` |
| `NET=user` | none (behind NAT) | `localhost:$SSH_PORT` | nothing |
| `NET=none` | none | nothing | nothing |

**The SSH endpoint moves with the mode, and that is the part that breaks a harness silently.** A
bridge has no hostfwd, so `localhost:2222` reaches nothing while every probe under `scripts/` still
types `ssh macos-vm`. Two scripts close that gap and a probe run needs the second one:

```bash
vm/guest-ip.sh                      # the guest's address, from the lease for its pinned MAC
vm/guest-ip.sh --ssh-target         # user@ip, ready for ssh
vm/guest-authorize.sh               # installs the key AND points `macos-vm` at this boot
vm/guest-authorize.sh --write-ssh-config   # ...and adds the Include to ~/.ssh/config, once
```

`guest-authorize.sh` writes `vm/disks/run/ssh-config` (in-workspace, replaced per boot) holding the
`macos-vm` alias for whichever endpoint this boot actually has. `ssh macos-vm` finds it only once
`~/.ssh/config` includes that file, which is what `--write-ssh-config` does; without it the script
prints the one line to add and exits non-zero rather than leaving the probes to fail at their first
hop. Pass `NET=user` to `guest-authorize.sh` when the boot used SLIRP — its default matches
`boot-x86.sh`'s, so a plain boot needs nothing.

**`vm/boot-arm64.sh` has no bridge mode** and still defaults to `NET=user`. `virbr0` is a
libvirt-on-Linux object and `qemu-bridge-helper` speaks `SIOCBRADDIF` over `linux/if_bridge.h`;
neither exists on the Apple host that pathway requires. The macOS equivalent is QEMU's
`-netdev vmnet-bridged`, which needs its own entitlement and root, and is not wired up.

**A bridge shows `DOWN`/`NO-CARRIER` until a guest attaches.** That is normal and not the failure;
only "does not exist" is. Bring libvirt's up with
`sudo virsh --connect qemu:///system net-start default`.

**Two concurrent boots on one bridge collide on the pinned MAC.** `GUEST_MAC` is fixed so the lease
is stable across reverts, which also means two live guests claim one address. Under `NET=user` a
stale VM announced itself by holding `hostfwd`'s port and failing the next boot loudly; on a bridge
there is no such signal, so kill the previous QEMU before booting (see `## Verification` in
`AGENTS.md`) or give the second boot its own `GUEST_MAC`.

## USB passthrough

Both boot scripts take `--usb SPEC` (repeatable) and `USB_PASSTHROUGH="SPEC SPEC ..."`, resolved by
`vm/lib/usb-passthrough.sh` — one owner, so the two pathways cannot drift. `--list-usb` prints every
host device with all three spec forms:

```bash
vm/boot-x86.sh --list-usb
vm/boot-x86.sh --device reims-vgpu-pci --testing --usb 5-1.2          # by physical port
vm/boot-x86.sh --device reims-vgpu-pci --testing --usb 046d:c099      # by descriptor
USB_PASSTHROUGH="5-1.2 7-1.1" vm/boot-x86.sh --testing
```

| Spec | Example | Pins | Loses |
|---|---|---|---|
| `BUS-PORT` | `5-1.2` | the physical socket — survives replug, separates identical twins | nothing; **prefer this** |
| `VID:PID` | `046d:c099` | the descriptor — follows the device between ports | ambiguous between two identical devices |
| `BUS.ADDR` | `5.3` | one device exactly, right now | `ADDR` is reassigned on replug and on some hub resets |

Each spec is resolved against `/sys/bus/usb/devices` before QEMU starts, so a typo, an unplugged
device, or a `/dev/bus/usb` node this user cannot write is a **named refusal with the fix** rather
than a guest that silently never sees the device. The device is announced with its manufacturer and
product strings, because a passed-through device is detached from the host driver for the life of
the boot — pass the keyboard you are typing on and you do not get it back until the VM exits.

Passed-through devices share the `qemu-xhci` the boot already builds for `usb-kbd`/`usb-tablet`, so
the guest sees one USB topology. **arm64 cannot resolve specs**: the resolver reads `/sys`, that
pathway is always macOS, and QEMU's `hostbus`/`hostaddr` are libusb's numbering rather than IOKit's.
It refuses by name; the Apple-host resolver is an honest gap.

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
