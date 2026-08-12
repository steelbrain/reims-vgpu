# vmapple-provision.sh

Provisions a fresh arm macOS guest in this workspace at `vm/guest/`, using `macosvm` (the
Virtualization.framework CLI). Nothing is copied from any external tree; everything lands
in-workspace.

## Target macOS version

**Ventura 13.x** (13.6/22G120 is the last 13.x published as a full-restore IPSW; 13.7.x were
OTA-only). Upstream QEMU documents only 12.x vmapple guests, but our fork carries the RFC v7
gicv2m/MSI patches for newer guests; **Monterey 12.x is the fallback** if 13 regresses. We do
**not** apply RFC v7 patch 7 (the Tahoe private-ISA path requires host SIP disabled). The guest
boots under `-M vmapple` with HVF on a macOS 26 host.

## Step 1 — get a UniversalMac restore IPSW (~13 GB) into `vm/ipsw/`

The restore image is a **UniversalMac** IPSW (Apple-Silicon virtual-machine restore), e.g.
`UniversalMac_13.6_22G120_Restore.ipsw`. Ways to obtain it:

- **ipsw.me API** (serves `VirtualMac2,1`):
  `curl -s "https://api.ipsw.me/v4/device/VirtualMac2,1?type=ipsw"` and pick the version's
  `updates.cdn-apple.com` URL, or
- **`mist-cli`** (`brew install mist-cli`; `mist download firmware "13.6"`).

Put it at `vm/ipsw/UniversalMac_13.x_..._Restore.ipsw` (gitignored) and verify the API's `sha1sum`.

## Step 2 — restore into `vm/guest/`

```sh
scripts/vmapple-provision/vmapple-provision.sh vm/ipsw/UniversalMac_13.6_22G120_Restore.ipsw
```

This runs `macosvm --disk vm/guest/disk.img,size=64g --aux vm/guest/aux.img --restore <ipsw> --net
nat -c 4 -r 8g vm/guest/vm.json`, then trims `aux.img → aux.img.trimmed` for QEMU. The restore is
non-interactive (`VZMacOSInstaller`). Provision to a side directory with `GUEST_DIR=` when the
current bundle must survive until the new one proves bootable.

## Step 3 — finish the golden image (one bootstrap boot)

```sh
mkdir -p vm/guest/rails/<rail>              # a rail is one guest OS line
vm/boot-arm64.sh --rail <rail> --capture    # rail is empty → boots the provisioned disk write-through
```

Complete Setup Assistant (create the user), enable Remote Login
(`sudo systemsetup -setremotelogin on`), and run `scripts/vmapple-guest-config` for no-sleep and
SSH settings. Enable auto-login manually in System Settings if desired. A clean guest shutdown then
captures the first immutable snapshot;
`vm/boot-arm64.sh --testing` reverts to it — confirm `ssh -p 2222` reaches the guest.

Env knobs: `GUEST_DIR DISK_SIZE CPUS RAM IPSW GUEST_MAC`.
