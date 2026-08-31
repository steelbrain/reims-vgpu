#!/usr/bin/env bash
# Boot one x86 rail and run the Metal conformance battery inside the guest.
#
#   run-guest.sh <outdir>
#
# The same source the oracle runs, on the paravirtual device. `run-native.sh`
# builds the x86_64 artifact that this runner sends to the rail. The same source
# and build identity therefore feed both oracle and guest results.
#
# `RAIL` selects the guest rail and defaults to macos-13.
#
# Environment passes through, so an arm is `REIMS_VGPU_X=y run-guest.sh ...`.
set -uo pipefail
export LC_ALL=C
OUT="${1:?usage: run-guest.sh <outdir>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
# A guest rail is a guest driver, so the failures a rail owns are that rail's
# alone. Each rail therefore carries its own debt inventory, and a rail with no
# inventory is refused rather than silently scored against another rail's debt.
RAIL="${RAIL:-macos-13}"
EXPECT="$HERE/expectations/$RAIL"
if [ ! -d "$REPO/vm/disks/rails/$RAIL" ]; then
  echo "conformance: no such rail: $RAIL (see vm/boot-x86.sh --list-rails)" >&2
  exit 2
fi
if [ ! -r "$EXPECT/driver-errors.txt" ] || [ ! -r "$EXPECT/translation-errors.txt" ]; then
  echo "conformance: rail $RAIL has no expectation inventory at $EXPECT" >&2
  exit 2
fi
RUN_KIND="${RUN_KIND:-candidate}"
case "$RUN_KIND" in baseline|candidate) ;; *)
  echo "conformance: RUN_KIND must be baseline or candidate" >&2; exit 2 ;;
esac
CONFORMANCE_MODE="${CONFORMANCE_MODE:-full}"
case "$CONFORMANCE_MODE" in
  full) CONFORMANCE_ARG="" ;;
  integer-clear) CONFORMANCE_ARG="--integer-clear-only" ;;
  topology) CONFORMANCE_ARG="--topology-only" ;;
  float-sampling) CONFORMANCE_ARG="--float-sampling-only" ;;
  indexed-draw) CONFORMANCE_ARG="--indexed-draw-only" ;;
  *) echo "conformance: CONFORMANCE_MODE must be full, integer-clear, topology, float-sampling, or indexed-draw" >&2; exit 2 ;;
esac
if [ "$CONFORMANCE_MODE" = "full" ]; then
  PROBE_TIMEOUT="${CONFORMANCE_TIMEOUT:-600}"
else
  PROBE_TIMEOUT="${CONFORMANCE_TIMEOUT:-180}"
fi
case "$PROBE_TIMEOUT" in
  ''|*[!0-9]*|0) echo "conformance: CONFORMANCE_TIMEOUT must be a positive integer" >&2; exit 2 ;;
esac
if [ "$CONFORMANCE_MODE" != "full" ] && [ -z "${NATIVE:-}" ]; then
  echo "conformance: NATIVE=<matching oracle output> is required for focused mode" >&2
  exit 2
fi
if [ -n "${NATIVE:-}" ] && [ ! -r "$NATIVE" ]; then
  echo "conformance: oracle output is not readable: $NATIVE" >&2
  exit 2
fi
BIN="${CONFORMANCE_BIN:-$HERE/build/conformance-x86_64}"
mkdir -p "$OUT"
[ -x "$BIN" ] || {
  echo "conformance: no x86_64 battery at $BIN (run-native.sh makes it)" >&2
  exit 2
}

QEMU_ACTUAL="$(realpath -m "${QEMU_BIN:-$REPO/vendor/qemu/build/qemu-system-x86_64}")"
QEMU_SOURCE_REPO="$(cd "$(dirname "$QEMU_ACTUAL")/../../.." && pwd)"
MANIFEST="$OUT/manifest.txt"
{
  echo "kind=$RUN_KIND"
  echo "started=$(date --iso-8601=seconds)"
  echo "pathway=x86-macos-linux-vulkan"
  echo "rail=$RAIL"
  echo "snapshot=$(readlink -f "$REPO/vm/disks/rails/$RAIL/snapshots/current")"
  echo "backend=vulkan"
  echo "attach=reims-vgpu-pci"
  echo "page_shift=12"
  echo "host=$(uname -a)"
  echo "environment_overrides_begin"
  env | grep '^REIMS_VGPU_' | sort || true
  echo "environment_overrides_end"
  echo "harness_repo=$REPO"
  echo "harness_head=$(git -C "$REPO" rev-parse HEAD)"
  echo "harness_tracked_diff_sha256=$(git -C "$REPO" diff --binary HEAD | sha256sum | awk '{print $1}')"
  echo "harness_status_begin"
  git -C "$REPO" status --short
  echo "harness_status_end"
  echo "suite_inputs_begin"
  find "$HERE/suite" -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum "$HERE/verdict.py" "$EXPECT/translation-errors.txt" \
    "$EXPECT/driver-errors.txt"
  [ ! -x "$BIN" ] || sha256sum "$BIN"
  echo "suite_inputs_end"
  echo "qemu_source_repo=$QEMU_SOURCE_REPO"
  echo "qemu_source_head=$(git -C "$QEMU_SOURCE_REPO" rev-parse HEAD)"
  echo "qemu_source_tracked_diff_sha256=$(git -C "$QEMU_SOURCE_REPO" diff --binary HEAD | sha256sum | awk '{print $1}')"
  echo "qemu_source_status_begin"
  git -C "$QEMU_SOURCE_REPO" status --short
  echo "qemu_source_status_end"
  echo "qemu_path=$QEMU_ACTUAL"
  echo "serial_path=pending; see boot_stdout_path"
  echo "boot_stdout_path=$OUT/boot-stdout.log"
  echo "fail_log_path=/tmp/reims-vgpu-fail.log"
  echo "probe=Metal conformance binary over ssh"
  echo "conformance_mode=$CONFORMANCE_MODE"
  echo "conformance_timeout_seconds=$PROBE_TIMEOUT"
} >"$MANIFEST"

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

# Prove what the live process loaded. The hash is taken through /proc, not from
# a filename we hope the process used.
for _ in $(seq 1 30); do
  QEMU_PID="$(pgrep -o -f "^$QEMU_ACTUAL " 2>/dev/null || true)"
  [ -n "$QEMU_PID" ] && break
  sleep 1
done
[ -n "${QEMU_PID:-}" ] || { echo "could not identify the live QEMU process"; exit 1; }
{
  echo "boot_pid=$BOOT_PID"
  echo "qemu_pid=$QEMU_PID"
  echo "qemu_loaded_path=$(readlink -f "/proc/$QEMU_PID/exe")"
  echo "qemu_loaded_sha256=$(sha256sum "/proc/$QEMU_PID/exe" | awk '{print $1}')"
  STATICLIB="$QEMU_SOURCE_REPO/target/release/libreims_vgpu.a"
  if [ -f "$STATICLIB" ]; then
    echo "staticlib_path=$STATICLIB"
    echo "staticlib_sha256=$(sha256sum "$STATICLIB" | awk '{print $1}')"
  else
    echo "staticlib=not-found"
  fi
  grep -m1 'snapshot=' "$OUT/boot-stdout.log" || true
  grep -m1 'serial →' "$OUT/boot-stdout.log" || true
} >>"$MANIFEST"

"$REPO/vm/guest-authorize.sh" >"$OUT/authorize.log" 2>&1
for _ in $(seq 1 60); do
  timeout 20 ssh -o BatchMode=yes macos-vm true 2>/dev/null && break
  sleep 5
done

# Run the exact x86_64 artifact produced beside the oracle result. A rail guest
# has no developer tools, and probing its `swiftc` shim opens the installer
# rather than proving a compiler exists.
timeout 60 scp -o BatchMode=yes -q "$BIN" macos-vm:/tmp/conformance || {
  echo "could not copy the battery into the guest"; exit 1; }
ssh -o BatchMode=yes macos-vm 'shasum -a 256 /tmp/conformance' \
  | sed 's# /tmp/conformance$# guest:/tmp/conformance#' >>"$MANIFEST"
timeout "$PROBE_TIMEOUT" ssh -o BatchMode=yes macos-vm \
  "chmod +x /tmp/conformance && /tmp/conformance $CONFORMANCE_ARG" \
  >"$OUT/conformance.txt" 2>&1
rc=$?
echo "conformance rc=$rc"
cp /tmp/reims-vgpu-fail.log "$OUT/device.log" 2>/dev/null

PANIC=0
if grep -q 'guest kernel panic' "$OUT/boot-stdout.log"; then
  echo "PANIC"
  PANIC=1
else
  echo "no panic"
fi

CATASTROPHIC=0
if [ "$rc" -eq 124 ]; then
  echo "CATASTROPHIC conformance probe timed out after ${PROBE_TIMEOUT}s"
  CATASTROPHIC=1
fi
if grep -Eq '^(stamp_wait_timeout|.*device_lost[^=]|.*VK_ERROR_DEVICE_LOST)' "$OUT/device.log"; then
  echo "CATASTROPHIC device timeout or loss"
  CATASTROPHIC=1
fi

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
  --translation-errors "$EXPECT/translation-errors.txt" \
  --driver-errors "$EXPECT/driver-errors.txt" --device "$OUT/device.log" \
  "${ADVISORY[@]}" --quiet | tee "$OUT/verdict.txt"
vrc="${PIPESTATUS[0]}"
pkill -f 'qemu-system-x86_6[4].*reims-vgpu' 2>/dev/null
wait "$BOOT_PID" 2>/dev/null
boot_rc=$?
SERIAL_PATH="$(sed -n 's/.*serial → \([^ ]*\).*/\1/p' "$OUT/boot-stdout.log" | tail -1)"
if [ -n "$SERIAL_PATH" ] && [ -f "$SERIAL_PATH" ]; then
  cp "$SERIAL_PATH" "$OUT/serial.log"
fi
{
  echo "completed=$(date --iso-8601=seconds)"
  echo "conformance_rc=$rc"
  echo "verdict_rc=$vrc"
  echo "boot_rc=$boot_rc"
  echo "panic=$PANIC"
  echo "catastrophic=$CATASTROPHIC"
} >>"$MANIFEST"
# The battery's own exit code says a case failed; the verdict's says one failed
# that nobody had written down. The second is the one a sweep should gate on.
[ "$PANIC" -eq 0 ] || exit 1
[ "$CATASTROPHIC" -eq 0 ] || exit 1
[ "$vrc" -eq 0 ] || exit "$vrc"
exit 0
