#!/bin/bash
# The lane gate's OpenCore image, made on the lane host: BASE, a SIP-on image of the lane's
# OpenCore set, with the candidate's kexts in place of BASE's (opencore-add-kexts.sh replaces each
# bundle of the same name in its own entry's place): Graft (guest/graft's committed build),
# ReimsGraphics as its build stages it for Graft to bind (no personalities), and RgfxGate, which
# carries the candidate's bundle as its payload. Checks that OUT carries exactly the given
# binaries and no other kext: a BASE made before Graft (Lilu, a loader and its shims) is refused,
# to be made again with `reims-ctl update-opencore`.
#
#   msi-image.sh OCADD BASE OUT KEXTS RGFXGATE
#     OCADD     scripts/internal/opencore-add-kexts.sh
#     KEXTS     the set's kexts but RgfxGate: Graft.kext and ReimsGraphics.kext
#     RGFXGATE  the candidate's RgfxGate.kext
set -euo pipefail
OCADD=$1 BASE=$2 OUT=$3 KEXTS=$4 GATEKEXT=$5
[[ -f $BASE ]] || { echo "msi-image: no base image $BASE" >&2; exit 2; }
rm -f "$OUT"
bash "$OCADD" "$BASE" "$OUT" "$KEXTS/Graft.kext" "$KEXTS/ReimsGraphics.kext" "$GATEKEXT" > "$OUT.entries"

# What the image carries, read back from its ESP (the GPT's first partition).
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
qemu-img convert -O raw "$OUT" "$tmp/disk.raw"
off=$(python3 -I - "$tmp/disk.raw" << 'EOF'
import struct, sys
with open(sys.argv[1], "rb") as f:
    f.seek(512)
    h = f.read(92)
    lba, = struct.unpack_from("<Q", h, 72)
    f.seek(lba * 512)
    first, = struct.unpack_from("<Q", f.read(128), 32)
print(first * 512)
EOF
)
export MTOOLS_SKIP_CHECK=1
mcopy -s -i "$tmp/disk.raw@@$off" ::/EFI/OC/Kexts "$tmp/"
mcopy -i "$tmp/disk.raw@@$off" ::/EFI/OC/config.plist "$tmp/config.plist"
others=$(python3 -I - "$tmp/config.plist" << 'EOF'
import plistlib, sys
with open(sys.argv[1], "rb") as f:
    add = plistlib.load(f)["Kernel"]["Add"]
ours = {"Graft.kext", "ReimsGraphics.kext", "RgfxGate.kext"}
print(" ".join(e.get("BundlePath", "?") for e in add if e.get("BundlePath") not in ours))
EOF
)
[[ -z $others ]] || { echo "msi-image: $BASE also carries $others: make it again with reims-ctl update-opencore" >&2; exit 1; }
for k in "$KEXTS"/*.kext "$GATEKEXT"; do
  n=$(basename "$k" .kext)
  want=$(sha256sum < "$k/Contents/MacOS/$n" | cut -d' ' -f1)
  got=$(sha256sum < "$tmp/Kexts/$n.kext/Contents/MacOS/$n" | cut -d' ' -f1)
  [[ $want == "$got" ]] || { echo "msi-image: $OUT carries another $n ($got, not $want)" >&2; exit 1; }
  echo "IMAGE $n ${want:0:16}"
done
echo MSI_IMAGE_OK
