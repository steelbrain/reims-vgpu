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
mkdir -p "$OUT"

ssh "$ORACLE" 'rm -rf ~/conformance && mkdir -p ~/conformance' || exit 1
tar cf - -C "$HERE" suite | ssh "$ORACLE" 'cd conformance && tar xf -' || exit 1

# Also cross-compile for the x86 rails, whose guests may have no compiler. The
# deployment target is the oldest rail this project boots.
ssh "$ORACLE" 'cd conformance && \
  swiftc -O suite/*.swift suite/cases/*.swift -o conf-native && \
  swiftc -O -target x86_64-apple-macos11.0 suite/*.swift suite/cases/*.swift \
    -o conformance-x86_64' 2>&1 | tee "$OUT/build-native.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "native build failed"; exit 1; }

ssh "$ORACLE" 'cd conformance && ./conf-native' >"$OUT/native.txt" 2>&1
rc=$?
scp -q "$ORACLE:conformance/conformance-x86_64" "$OUT/conformance-x86_64" || exit 1
chmod +x "$OUT/conformance-x86_64"
echo "native rc=$rc"
tail -2 "$OUT/native.txt"
exit $rc
