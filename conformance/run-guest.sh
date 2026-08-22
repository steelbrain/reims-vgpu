#!/usr/bin/env bash
# Boot one x86 rail and run the Metal conformance battery inside the guest.
#
#   run-guest.sh <outdir>
#
# The same source the oracle runs, on the paravirtual device. `run-native.sh`
# builds the x86_64 fallback into conformance/build for the rails with no
# developer tools -- `AGENTS.md` records that a guest-side build does not
# degrade gracefully, it simply fails, and reports a build error that reads like
# noise.
#
# Environment passes through, so an arm is `REIMS_VGPU_X=y run-guest.sh ...`.
set -uo pipefail
export LC_ALL=C
OUT="${1:?usage: run-guest.sh <outdir>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
RAIL="${RAIL:-macos-13}"
BIN="$HERE/build/conformance-x86_64"
mkdir -p "$OUT"
# Not fatal: a rail with `swiftc` builds the suite itself, and only a rail
# without one needs the cross-built fallback.
[ -x "$BIN" ] || echo "conformance: no cross-built fallback at $BIN (run-native.sh makes it)"

pkill -f 'qemu-system-x86_6[4].*reims-vgpu' 2>/dev/null
for _ in $(seq 1 30); do
  pgrep -f 'qemu-system-x86_6[4].*reims-vgpu' >/dev/null || break
  sleep 1
done
rm -f /tmp/reims-vgpu-fail.log

echo "conformance: booting rail=$RAIL"
TESTING_TIMEOUT="${TESTING_TIMEOUT:-1800}" \
  "$REPO/vm/boot-x86.sh" --device reims-vgpu-pci --rail "$RAIL" --testing \
  >"$OUT/boot-stdout.log" 2>&1 &
BOOT_PID=$!

# Only a live device writes the fail log, so this is what says the boot is ours
# and not a survivor holding 2222.
for _ in $(seq 1 240); do
  [ -f /tmp/reims-vgpu-fail.log ] && break
  kill -0 "$BOOT_PID" 2>/dev/null || break
  sleep 2
done
[ -f /tmp/reims-vgpu-fail.log ] || { echo "device never came up"; exit 1; }

"$REPO/vm/guest-authorize.sh" >"$OUT/authorize.log" 2>&1
for _ in $(seq 1 60); do
  timeout 20 ssh -o BatchMode=yes macos-vm true 2>/dev/null && break
  sleep 5
done

# Prefer building in the guest where the rail has `swiftc`: the battery is
# edited far more often than a rail is added, and a guest-side build removes the
# cross-build round trip from that loop. Where the rail has no compiler -- which
# `AGENTS.md` records is the normal case, not the exception -- the cross-built
# binary is what runs, and it is what makes the two hosts comparable.
timeout 120 ssh -o BatchMode=yes macos-vm 'rm -rf /tmp/conformance-src && mkdir -p /tmp/conformance-src'
tar cf - -C "$HERE" suite | timeout 120 ssh -o BatchMode=yes macos-vm \
  'cd /tmp/conformance-src && tar xf -'
if timeout 120 ssh -o BatchMode=yes macos-vm 'command -v swiftc >/dev/null' 2>/dev/null; then
  echo "conformance: building in the guest"
  timeout 900 ssh -o BatchMode=yes macos-vm \
    'cd /tmp/conformance-src && swiftc -O suite/*.swift suite/cases/*.swift -o /tmp/conformance' \
    >"$OUT/build.log" 2>&1 || { echo "guest build failed; see $OUT/build.log"; }
fi
timeout 120 ssh -o BatchMode=yes macos-vm 'test -x /tmp/conformance' 2>/dev/null || {
  timeout 60 scp -o BatchMode=yes -q "$BIN" macos-vm:/tmp/conformance || {
    echo "could not copy the battery into the guest"; exit 1; }
}
timeout 600 ssh -o BatchMode=yes macos-vm 'chmod +x /tmp/conformance && /tmp/conformance' \
  >"$OUT/conformance.txt" 2>&1
rc=$?
echo "conformance rc=$rc"
cp /tmp/reims-vgpu-fail.log "$OUT/device.log" 2>/dev/null

grep -q 'guest kernel panic' "$OUT/boot-stdout.log" && echo "PANIC" || echo "no panic"

# Score it against the oracle rather than printing 216 lines for a reader to
# diff by eye. The baseline is the native arm when no fresher one is given;
# `NATIVE=<file>` points at a run from the same session, which is what to use
# when the suite has cases the baseline predates.
NATIVE="${NATIVE:-$HERE/baselines/native-apple-m4-macos15.txt}"

# A narrowing switch can close the rail a case claims, so its claim not moving
# is then the switch working rather than a case that missed its rail. Report it
# without failing the run.
ADVISORY=()
env | grep -q '^REIMS_VGPU_' && ADVISORY=(--claims-advisory)

echo "--- verdict ---"
python3 "$HERE/verdict.py" --native "$NATIVE" --guest "$OUT/conformance.txt" \
  --expect "$HERE/expectations/known-failures.txt" --device "$OUT/device.log" \
  "${ADVISORY[@]}" --quiet | tee "$OUT/verdict.txt"
vrc="${PIPESTATUS[0]}"
pkill -f 'qemu-system-x86_6[4].*reims-vgpu' 2>/dev/null
# The battery's own exit code says a case failed; the verdict's says one failed
# that nobody had written down. The second is the one a sweep should gate on.
[ "$vrc" -eq 0 ] || exit "$vrc"
exit $rc
