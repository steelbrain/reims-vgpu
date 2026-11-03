#!/bin/bash
# The lane gate's OpenCore image, made on the lane host: BASE, a SIP-on image of the lane's
# OpenCore set, with the candidate's kexts in place of BASE's (opencore-add-kexts.sh replaces each
# bundle of the same name in its own entry's place): the loader that matches the candidate's
# ReimsGraphics, its IOGraphicsFamily and AppleGraphicsDeviceControl shims, ReimsGraphics itself
# (the loader builder's staged copy, whose Info.plist carries no personalities: the loader holds
# them), and RgfxGate, which carries the candidate's bundle as its payload. Checks that OUT
# carries exactly the given binaries.
#
#   msi-image.sh OCADD BASE OUT KEXTS RGFXGATE
#     OCADD     scripts/internal/opencore-add-kexts.sh
#     KEXTS     the loader builder's kexts: AppleParavirtGPULoader.kext, ReimsGraphics.kext and
#               the *Bridge.kext shims
#     RGFXGATE  the candidate's RgfxGate.kext
set -euo pipefail
OCADD=$1 BASE=$2 OUT=$3 KEXTS=$4 GATEKEXT=$5
[[ -f $BASE ]] || { echo "msi-image: no base image $BASE" >&2; exit 2; }
rm -f "$OUT"
shims=("$KEXTS"/*Bridge.kext)
bash "$OCADD" "$BASE" "$OUT" "${shims[@]}" "$KEXTS/ReimsGraphics.kext" \
  "$KEXTS/AppleParavirtGPULoader.kext" "$GATEKEXT" > "$OUT.entries"

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
for k in "$KEXTS"/*.kext "$GATEKEXT"; do
  n=$(basename "$k" .kext)
  want=$(sha256sum < "$k/Contents/MacOS/$n" | cut -d' ' -f1)
  got=$(sha256sum < "$tmp/Kexts/$n.kext/Contents/MacOS/$n" | cut -d' ' -f1)
  [[ $want == "$got" ]] || { echo "msi-image: $OUT carries another $n ($got, not $want)" >&2; exit 1; }
  echo "IMAGE $n ${want:0:16}"
done
echo MSI_IMAGE_OK
