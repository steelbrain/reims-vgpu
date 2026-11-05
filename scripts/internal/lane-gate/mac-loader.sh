#!/bin/bash
# The loader that matches one ReimsGraphics build, built on the Mac (lane-gate.sh's loader step):
#
#   mac-loader.sh KEXT_DIR WORKDIR
#
# Writes WORKDIR/kexts/: AppleParavirtGPULoader.kext, ReimsGraphics.kext (the staged copy, whose
# Info.plist carries no personalities: the loader holds them) and the kext's *Bridge.kext shims,
# which msi-image.sh puts into the gate's OpenCore image on the lane host. It runs the build steps
# of ~/.cache/rgk-run-mac/rgk-loader-for.sh (gen-shims.py, stage-consumer.py, build-stage1.sh
# --bridge-set) on a private copy of the staged set in ~/.cache/rgk-run-mac/base, with the loader
# source exported from ~/.cache/rgk-stage3's LOADER_REV (default steelbrain/loader-no-gop), and
# none of its steps on the lane host: the kexts reach it in the gate's shipped commit.
#
# Why per build: the loader compiles in, per consumer, the offset of every site where the kext
# refers to IOGraphicsFamily and the kext's IOKitPersonalities; a kext build that moves them makes
# the loader refuse it. The shims are data-only placeholder kexts (gen-shims.py), our own output:
# the KDK's IOGraphicsFamily and AppleGraphicsDeviceControl are only read for their symbols.
set -euo pipefail
die() { echo "mac-loader: $*" >&2; exit 2; }
[[ $# -eq 2 ]] || die "usage: mac-loader.sh KEXT_DIR WORKDIR"
KEXT=$(cd "$1" && pwd) || die "no directory $1"
W=$2
[[ -f $KEXT/Contents/Info.plist ]] || die "$KEXT is not a kext bundle"
EXE=$(plutil -extract CFBundleExecutable raw "$KEXT/Contents/Info.plist")
[[ $EXE == ReimsGraphics && -f $KEXT/Contents/MacOS/$EXE ]] || die "$KEXT is not a ReimsGraphics kext (executable '$EXE')"

ST=$HOME/.cache/rgk-stage3
REV=${LOADER_REV:-steelbrain/loader-no-gop}
BASE=$HOME/.cache/rgk-run-mac/base
K=$ST/build/ds/artifacts/macos-26.7.1-25G241/kdk/Extensions
FAMILY=$K/IOGraphicsFamily.kext/IOGraphicsFamily
AGDC=$K/AppleGraphicsControl.kext/Contents/PlugIns/AppleGraphicsDeviceControl.kext/Contents/MacOS/AppleGraphicsDeviceControl
[[ -d $BASE/set && -d $BASE/deps ]] || die "no base set in $BASE"
[[ -f $FAMILY ]] || die "no KDK IOGraphicsFamily at $FAMILY"
LIBS=("$FAMILY")
PREFIX=$EXE
if plutil -extract OSBundleLibraries.com\\.apple\\.AppleGraphicsDeviceControl raw "$KEXT/Contents/Info.plist" > /dev/null 2>&1; then
  [[ -f $AGDC ]] || die "no KDK AppleGraphicsDeviceControl at $AGDC"
  LIBS+=("$AGDC")
  PREFIX=Rgfx
fi
rm -rf "$W"
mkdir -p "$W/tree/build"
git -C "$ST" archive "$REV" src/loader tools/loader | tar -x -m -C "$W/tree"
grep -q startOnGoplessBoot "$W/tree/src/loader/AppleParavirtGPULoader/Loader.cpp" ||
  die "$REV's Loader.cpp lacks the no-GOP start (startOnGoplessBoot)"
# bridge.py imports the decompilation's Mach-O tools from ROOT/build/ds/tools.
ln -s "$(readlink "$ST/build/ds" || echo "$ST/build/ds")" "$W/tree/build/ds"
cp -R "$BASE/set" "$W/set"
ln -s "$BASE/deps" "$W/deps"

step() {  # each step on its own, so the first failure stops the build
  "$@" >> "$W/build.log" 2>&1 || { tail -20 "$W/build.log" >&2; die "failed: ${*:1:3}...; log $W/build.log"; }
}
step python3 -I -B "$W/tree/tools/loader/gen-shims.py" "$KEXT/Contents/MacOS/$EXE" \
  "${LIBS[@]}" --prefix "$PREFIX" --out "$W/reims-shims"
step python3 -I -B "$W/tree/tools/loader/stage-consumer.py" "$KEXT" "$W/reims-shims" "$W/set"
step bash "$W/tree/tools/loader/build-stage1.sh" --bridge-set "$W/set" "$W"
grep -E '^fixups:|staged without' "$W/build.log" | sed 's/^/  /' || true

mkdir "$W/kexts"
cp -R "$W/stage1/AppleParavirtGPULoader.kext" "$W/set/ReimsGraphics.kext" "$W/set/$PREFIX"*Bridge.kext "$W/kexts/"
echo "  kext sha256 $(shasum -a 256 < "$KEXT/Contents/MacOS/$EXE" | awk '{print $1}')," \
  "loader sha256 $(shasum -a 256 < "$W/kexts/AppleParavirtGPULoader.kext/Contents/MacOS/AppleParavirtGPULoader" | awk '{print $1}')"
echo "MAC_LOADER_OK $W/kexts"
