#!/bin/bash
# The lane gate's publishing of a tip, on the lane host (lane-gate.sh --publish): the run's image
# and host build, read-only, where the lane's agents boot them from.
#   msi-publish.sh RUNDIR QEMU_BUILD DIST SHORT QEMU_COMMIT M2V RGK_RUN CDHASH
#     RUNDIR       the gate's run on the host: oc.qcow2, qemu-system-x86_64, ReimsGraphics.rom,
#                  reims-graphics.bundle.tgz (the signed bundle RgfxGate carries)
#     QEMU_BUILD   QEMU's build directory, whose qemu-bundle holds its firmware data
#     DIST         the release dist the run built (cargo xtask dist)
#     SHORT        the commit, as the lane names its tips (9 hex digits)
#     CDHASH       the bundle's code directory hash
# Writes RGK_RUN/loader-fix/oc-sipon-SHORT.qcow2 and RGK_RUN/tip-host/SHORT/: qemu-system-x86_64
# with its data in qemu-bundle/ (where it finds it by itself; share/ is the same directory, for
# -L), dist/, ReimsGraphics.rom, bundle/ (the signed reims-graphics.bundle and its package) and
# SOURCE (what each was built from, and their sha256s). What exists already is left as it is.
set -euo pipefail
MR=$1 QB=$2 DIST=$3 SHORT=$4 QC=$5 M2V=$6 RGK_RUN=$7 CDHASH=${8:-}

I=$RGK_RUN/loader-fix/oc-sipon-$SHORT.qcow2
if [[ -e $I ]]; then
  echo "PUBLISHED image $I (there already)"
else
  cp --reflink=auto "$MR/oc.qcow2" "$I.tmp"
  chmod a-w "$I.tmp"
  mv "$I.tmp" "$I"
  echo "PUBLISHED image $I"
fi

D=$RGK_RUN/tip-host/$SHORT
if [[ -e $D ]]; then
  echo "PUBLISHED host $D (there already)"
  exit 0
fi
rm -rf "$D.tmp"
mkdir -p "$D.tmp"
cp --reflink=auto "$MR/qemu-system-x86_64" "$MR/ReimsGraphics.rom" "$D.tmp/"
# QEMU's data, dereferenced, so the copy needs nothing of the gate's: only what `ninja
# qemu-system-x86_64` makes. Links to what a full build would generate (edk2's images, keymaps,
# other targets' device trees, trace-events-all) dangle, and the lane's boots take OVMF from
# reims-ctl's VM.
S=$QB/qemu-bundle/usr/local/share/qemu Q=$D.tmp/qemu-bundle/usr/local/share/qemu
mkdir -p "$Q"
(cd "$S" && find . \( -type f -o -type l \) -print) | while IFS= read -r f; do
  if [[ -e $S/$f ]]; then
    mkdir -p "$Q/$(dirname "$f")"
    cp -L --reflink=auto "$S/$f" "$Q/$f"
  else
    echo "${f#./}" >> "$D.tmp/qemu-data-not-built.txt"
  fi
done
ln -s qemu-bundle/usr/local/share/qemu "$D.tmp/share"
cp -a --reflink=auto "$DIST" "$D.tmp/dist"
mkdir -p "$D.tmp/bundle"
cp "$MR/reims-graphics.bundle.tgz" "$D.tmp/bundle/"
tar -C "$D.tmp/bundle" -xzf "$MR/reims-graphics.bundle.tgz"
[[ -d $D.tmp/bundle/reims-graphics.bundle ]] || { echo "msi-publish: the package holds no reims-graphics.bundle" >&2; exit 1; }
{
  echo "tree $SHORT: the release dist (cargo xtask dist; metal2vulkan $M2V)"
  echo "qemu: vendor/qemu $QC, x86_64-softmmu, linked against dist/ statically"
  echo "rom: firmware/ at $SHORT"
  echo "bundle: bundle/reims-graphics.bundle, signed, CDHash $CDHASH (bundle/reims-graphics.bundle.tgz its package)"
  echo "image: $I"
  echo "QEMU finds its firmware data in qemu-bundle/ beside the binary; share/ is the same directory, for -L."
  echo "qemu-data-not-built.txt: the data files of a full QEMU build that this one lacks"
  echo
  (cd "$D.tmp" && sha256sum qemu-system-x86_64 ReimsGraphics.rom dist/lib/libreims_vgpu.a dist/include/reims_vgpu.h \
    bundle/reims-graphics.bundle.tgz bundle/reims-graphics.bundle/Contents/MacOS/reims-graphics)
  sha256sum "$I"
} > "$D.tmp/SOURCE"
if ldd "$D.tmp/qemu-system-x86_64" | grep -q reims; then echo "msi-publish: QEMU links libreims_vgpu dynamically" >&2; exit 1; fi
"$D.tmp/qemu-system-x86_64" -L help | grep -q "$D.tmp/qemu-bundle" || { echo "msi-publish: QEMU does not find its data" >&2; exit 1; }
"$D.tmp/qemu-system-x86_64" -device help | grep -q '"reims-graphics-pci"' || { echo "msi-publish: QEMU has no reims-graphics-pci" >&2; exit 1; }
mv "$D.tmp" "$D"
chmod -R a-w "$D"
echo "PUBLISHED host $D"
