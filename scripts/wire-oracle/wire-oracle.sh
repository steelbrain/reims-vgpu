#!/usr/bin/env bash
# Regenerate reims-vgpu-wire's ground-truth fixtures from Apple's own
# paravirt serializer.
#
# Apple-host only, and x86_64-under-Rosetta only: the serializer bundle ships
# x86_64 and arm64e slices with no plain-arm64 one, and third-party arm64e
# needs a preview-ABI boot-arg. Rosetta is the supported route.
#
# Output is gitignored on purpose. The bytes are Apple's serializer output, so
# they are regenerated rather than committed; what the repository keeps is the
# derived layout, the expectations, and the coverage manifest.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CRATE="$REPO_ROOT/crates/reims-vgpu-wire"
OUT_DIR="${REIMS_WIRE_FIXTURES_DIR:-$CRATE/fixtures}"
BUILD_DIR="$REPO_ROOT/target/wire-oracle"
BUNDLE=/System/Library/Extensions/AppleParavirtGPUMetal.bundle

usage() {
  cat <<'EOF'
usage: wire-oracle.sh [--fixtures] [--inventory] [--all]

  --fixtures   capture wire operations + expectations   (default)
  --inventory  dump every selector on the serializer classes
  --all        both

Writes into crates/reims-vgpu-wire/fixtures/ unless REIMS_WIRE_FIXTURES_DIR is set.
EOF
}

do_fixtures=0
do_inventory=0
case "${1:---fixtures}" in
  --fixtures) do_fixtures=1 ;;
  --inventory) do_inventory=1 ;;
  --all) do_fixtures=1; do_inventory=1 ;;
  -h|--help) usage; exit 0 ;;
  *) usage; exit 2 ;;
esac

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "wire-oracle: Apple host only (this is $(uname -s))." >&2
  echo "The serializer being measured only exists on macOS." >&2
  exit 1
fi

if [[ ! -d "$BUNDLE" ]]; then
  echo "wire-oracle: $BUNDLE not present." >&2
  echo "Without it there is nothing to measure against." >&2
  exit 1
fi

if ! arch -x86_64 /usr/bin/true 2>/dev/null; then
  echo "wire-oracle: cannot execute x86_64 binaries." >&2
  echo "Rosetta is required: softwareupdate --install-rosetta" >&2
  exit 1
fi

mkdir -p "$BUILD_DIR" "$OUT_DIR"

# Remove the targets before the build, not just before the run.
#
# Three things can end this script early: the compile can fail, the capture can
# abort (a case that makes Apple's serializer assert used to do exactly that),
# or the run can be interrupted. The JSON is only written at the very end, so
# any of them leaves the PREVIOUS file on disk — and every test that reads it
# afterwards then passes against stale ground truth while the run that produced
# it failed loudly and scrolled away. No file is a skip; a stale one is a lie.
#
# Written as `if` rather than `[[ … ]] && rm`, because under `set -e` a false
# `[[ … ]]` as the last statement of its own line ends the script.
if [[ $do_fixtures == 1 ]]; then
  rm -f "$OUT_DIR/fixtures.json"
fi
if [[ $do_inventory == 1 ]]; then
  rm -f "$OUT_DIR/inventory.json"
fi

echo "wire-oracle: building oracle (x86_64)"
clang -arch x86_64 -fobjc-arc -O1 -Wall \
  -Wno-deprecated-declarations \
  -framework Foundation -framework Metal \
  -o "$BUILD_DIR/oracle" "$CRATE/oracle/oracle.m"

if [[ $do_fixtures == 1 ]]; then
  echo "wire-oracle: capturing fixtures"
  arch -x86_64 "$BUILD_DIR/oracle" fixtures "$OUT_DIR/fixtures.json"
fi

if [[ $do_inventory == 1 ]]; then
  echo "wire-oracle: dumping selector inventory"
  arch -x86_64 "$BUILD_DIR/oracle" inventory "$OUT_DIR/inventory.json"
fi

cat <<EOF

Fixtures are in $OUT_DIR (gitignored).
Verify against them with:

  cargo test -p reims-vgpu-wire -- --test-threads=1

A run with no fixtures present reports the oracle tests as ignored, one line
each; set REIMS_WIRE_FIXTURES_REQUIRED=1 to make their absence fail the build.
EOF
