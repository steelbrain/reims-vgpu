#!/usr/bin/env bash
# Build and run the battery on a native macOS host -- the oracle arm.
#
#   run-native.sh [outdir]
#
# The oracle is what makes a guest failure a finding. A case that fails here is
# a wrong expectation in the suite, not a device defect, and the only way to
# tell those apart is to run the same source on both sides. `ORACLE` names the
# ssh host; it needs Xcode's command line tools and nothing else.
set -uo pipefail
export LC_ALL=C
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE/build}"
ORACLE="${ORACLE:-scaleway-m4-macos15}"
CONFORMANCE_MODE="${CONFORMANCE_MODE:-full}"
case "$CONFORMANCE_MODE" in
  full) CONFORMANCE_ARG="" ;;
  integer-clear) CONFORMANCE_ARG="--integer-clear-only" ;;
  topology) CONFORMANCE_ARG="--topology-only" ;;
  float-sampling) CONFORMANCE_ARG="--float-sampling-only" ;;
  indexed-draw) CONFORMANCE_ARG="--indexed-draw-only" ;;
  *) echo "conformance: CONFORMANCE_MODE must be full, integer-clear, topology, float-sampling, or indexed-draw" >&2; exit 2 ;;
esac
mkdir -p "$OUT"

{
  echo "kind=native-oracle"
  echo "oracle=$ORACLE"
  echo "conformance_mode=$CONFORMANCE_MODE"
  echo "started=$(date --iso-8601=seconds)"
  find "$HERE/suite" -type f -print0 | sort -z | xargs -0 sha256sum
} >"$OUT/manifest.txt"

REMOTE="$(ssh "$ORACLE" 'mktemp -d /tmp/reims-vgpu-conformance.XXXXXX')" || exit 1
case "$REMOTE" in
  /tmp/reims-vgpu-conformance.*) ;;
  *) echo "oracle returned an unsafe staging path: $REMOTE"; exit 1 ;;
esac
cleanup() { ssh "$ORACLE" "rm -rf -- '$REMOTE'" >/dev/null 2>&1 || true; }
trap cleanup EXIT
tar cf - -C "$HERE" suite | ssh "$ORACLE" "cd '$REMOTE' && tar xf -" || exit 1

# Also cross-compile for the x86 rails, whose guests may have no compiler. The
# deployment target is the oldest rail this project boots.
ssh "$ORACLE" "cd '$REMOTE' && \
  swiftc -O suite/*.swift suite/cases/*.swift -o conf-native && \
  swiftc -O -target x86_64-apple-macos11.0 suite/*.swift suite/cases/*.swift \
    -o conformance-x86_64" 2>&1 | tee "$OUT/build-native.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "native build failed"; exit 1; }

ssh "$ORACLE" "cd '$REMOTE' && sw_vers && uname -a && \
  shasum -a 256 conf-native conformance-x86_64" >>"$OUT/manifest.txt" || exit 1

ssh "$ORACLE" "cd '$REMOTE' && ./conf-native $CONFORMANCE_ARG" >"$OUT/native.txt" 2>&1
rc=$?
scp -q "$ORACLE:$REMOTE/conformance-x86_64" "$OUT/conformance-x86_64" || exit 1
chmod +x "$OUT/conformance-x86_64"
echo "native rc=$rc"
tail -2 "$OUT/native.txt"
echo "completed=$(date --iso-8601=seconds)" >>"$OUT/manifest.txt"
echo "result_rc=$rc" >>"$OUT/manifest.txt"
exit $rc
