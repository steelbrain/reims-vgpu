#!/usr/bin/env bash
#
# scripts/vmapple-provision/vmapple-provision.sh
#
# Provision a FRESH arm macOS guest in this workspace (vm/guest/) with `macosvm`
# (the Virtualization.framework CLI). macosvm restores macOS from an IPSW
# non-interactively (VZMacOSInstaller) into a raw disk + aux image that QEMU's
# vmapple boots directly.
#
# Target: macOS 13.x (Ventura; 13.6/22G120 is the last 13.x with a full-restore
# IPSW — 13.7.x were OTA-only). Upstream vmapple documents only 12.x guests, but
# our fork carries the RFC v7 gicv2m/MSI patches for newer guests; 12.x remains
# the fallback. We do NOT apply RFC v7 patch 7 (the Tahoe private-ISA path needs
# host SIP disabled).
#
# Everything lands in-workspace; nothing is copied from any external tree.
#
# Usage:
#   vmapple-provision.sh /path/to/UniversalMac_13.x_Restore.ipsw
#   IPSW=/path/to.ipsw GUEST_DIR=... vmapple-provision.sh
#
# After this finishes, one interactive boot completes Setup Assistant + enables
# Remote Login (see the printed next steps), then the bundle is the golden image.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GUEST_DIR="${GUEST_DIR:-$REPO_ROOT/vm/guest}"
DISK_SIZE="${DISK_SIZE:-64g}"
CPUS="${CPUS:-4}"
RAM="${RAM:-8g}"
# Pin the NIC MAC to the SAME value vm/boot-arm64.sh uses. Critical: a stable MAC from
# first boot means macOS configures networking for it once and that config stays
# valid across snapshot reverts (a changing MAC poisons the guest's network state
# and breaks DHCP on every reverted boot).
GUEST_MAC="${GUEST_MAC:-52:54:00:76:61:70}"
IPSW="${IPSW:-${1:-}}"

die() { echo "vmapple-provision: $*" >&2; exit 1; }

command -v macosvm >/dev/null || die "macosvm not found (brew install macosvm or build s-u/macosvm)"
[ -n "$IPSW" ] || die "no IPSW given. Pass the path to a UniversalMac restore IPSW (13.x target, 12.x fallback):
  vmapple-provision.sh vm/ipsw/UniversalMac_13.6_22G120_Restore.ipsw
Acquire one (~13 GB) into vm/ipsw/ first (see scripts/vmapple-provision/README.md)."
[ -f "$IPSW" ] || die "IPSW not found: $IPSW"

mkdir -p "$GUEST_DIR"
DISK="$GUEST_DIR/disk.img"
AUX="$GUEST_DIR/aux.img"
VMJSON="$GUEST_DIR/vm.json"

if [ -f "$DISK" ]; then
  die "refusing to overwrite an existing bundle at $GUEST_DIR (remove it first to re-provision)"
fi

echo "vmapple-provision: restoring $IPSW → $GUEST_DIR (this takes a while) ..."
# macosvm creates the disk + aux, restores macOS from the IPSW, and writes the VZ
# config (incl. machineId/ECID + hardwareModel) to the trailing config path.
macosvm \
  --disk "$DISK,size=$DISK_SIZE" \
  --aux "$AUX" \
  --restore "$IPSW" \
  --net nat \
  --mac "$GUEST_MAC" \
  -c "$CPUS" \
  -r "$RAM" \
  "$VMJSON"

[ -f "$VMJSON" ] || die "macosvm did not write $VMJSON (restore failed?)"

# Trim the aux metadata page for QEMU's vmapple (docs/system/arm/vmapple.rst).
echo "vmapple-provision: trimming aux.img → aux.img.trimmed for QEMU vmapple ..."
dd if="$AUX" of="$GUEST_DIR/aux.img.trimmed" bs=$((0x4000)) skip=1 status=none

UUID="$(plutil -extract machineId raw "$VMJSON" | base64 -d | plutil -extract ECID raw -)"
cat <<EOF

vmapple-provision: DONE. Golden-image bundle at $GUEST_DIR
  disk.img / aux.img / aux.img.trimmed / vm.json   (ECID/uuid=$UUID)

Next (one bootstrap boot to finish the golden image):
  1. mkdir -p vm/guest/rails/<rail>          # a rail is one guest OS line; name it for the OS
     vm/boot-arm64.sh --rail <rail> --capture # rail is empty → boots write-through; complete Setup Assistant
  2. In the guest: enable Remote Login (sudo systemsetup -setremotelogin on), then run
     scripts/vmapple-guest-config (sleep off, SSH kept on)
  3. Shut the guest down cleanly — that captures the first immutable snapshot
  4. vm/boot-arm64.sh --testing then reverts to it; confirm ssh reaches localhost:2222
EOF
