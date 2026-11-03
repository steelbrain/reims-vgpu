#!/bin/bash
# The lane gate's host build, run on the Linux lane host (lane-gate.sh sends it with the
# candidate's tree). Builds, outside any GPU lock:
#   - the library's Vulkan dist (cargo xtask dist, into $TARGET/dist-vulkan),
#   - the option ROMs (firmware/build/ReimsGraphics.rom among them),
#   - QEMU's qemu-system-x86_64 for the tree's vendor/qemu commit, out of tree, linked against
#     that dist.
#
#   msi-build.sh GATE TREE STAGE QEMU_COMMIT SUBPROJECTS
#     GATE         the gate's directory on this host ($GATE/target is CARGO_TARGET_DIR,
#                  $GATE/qemu/COMMIT holds QEMU's source and build)
#     TREE         the candidate's tree, kept from run to run: STAGE (a fresh `git archive` of
#                  the candidate) is synced into it by content, so a file that did not change keeps
#                  its time and cargo and ninja rebuild only what did
#     QEMU_COMMIT  vendor/qemu's commit; $GATE/qemu/COMMIT/src must hold its source
#     SUBPROJECTS  a QEMU tree's subprojects/ whose downloaded wraps (softfloat, testfloat,
#                  keycodemapdb) are copied into the source where it lacks them, so configure
#                  downloads nothing
# Prints the paths it built ("BUILT qemu PATH", "BUILT rom PATH") and "MSI_BUILD_OK".
set -euo pipefail
GATE=$1 TREE=$2 STAGE=$3 QC=$4 SUB=$5
export CARGO_TARGET_DIR=$GATE/target
mkdir -p "$TREE" "$CARGO_TARGET_DIR"

# By content (-c), without -t: a changed file gets this run's time, an unchanged one keeps its own.
rsync -rlc --delete --exclude=/target/ --exclude=/firmware/build/ "$STAGE/" "$TREE/"
echo "synced $(du -sh "$TREE" | cut -f1) into $TREE"

cd "$TREE"
. scripts/internal/dev-env.sh
t0=$(date +%s)
cargo xtask dist
echo "dist built in $(( $(date +%s) - t0 )) s"
make -C firmware > /dev/null
echo "BUILT rom $TREE/firmware/build/ReimsGraphics.rom $(sha256sum < firmware/build/ReimsGraphics.rom | cut -c1-16)"

Q=$GATE/qemu/$QC
[[ -f $Q/src/configure ]] || { echo "msi-build: no QEMU source in $Q/src" >&2; exit 2; }
for d in "$SUB"/*/; do
  n=$(basename "$d")
  [[ -e $Q/src/subprojects/$n ]] || cp -a --reflink=auto "$d" "$Q/src/subprojects/$n"
done
mkdir -p "$Q/build"
cd "$Q/build"
if [[ ! -f build.ninja ]]; then
  PKG_CONFIG_PATH=$CARGO_TARGET_DIR/dist-vulkan/lib/pkgconfig "$Q/src/configure" \
    --target-list=x86_64-softmmu --enable-kvm --disable-sdl --enable-gtk --enable-slirp \
    --enable-libusb --enable-pvg --disable-docs --disable-tests --disable-werror > configure.log 2>&1 ||
    { tail -30 configure.log >&2; exit 1; }
fi
# The library links in statically, and ninja does not see the archive change: relink each time.
rm -f qemu-system-x86_64
t0=$(date +%s)
ninja qemu-system-x86_64 > ninja.log 2>&1 || { tail -40 ninja.log >&2; exit 1; }
echo "qemu built in $(( $(date +%s) - t0 )) s"
echo "BUILT qemu $Q/build/qemu-system-x86_64"
echo MSI_BUILD_OK
