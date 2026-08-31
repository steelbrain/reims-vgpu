#!/usr/bin/env bash
# input-lever.sh KIND -- issue exactly one input onto a settled guest and watch
# it live or die.
#
# The `s9` arm killed its guest three times out of three within ~2 s of the
# menu probe's right-click on a dock icon. That probe issues three different
# inputs -- a desktop-wide pointer sweep, a left-click dismiss, and the
# right-click -- inside one eight-second window, so which of them is the lever
# is not decidable from its logs. This probe issues one, names it, and then
# stops touching the guest entirely.
#
# The hold afterwards is the measurement, not a courtesy: a guest that panics
# 2 s after an input and a guest that panics 60 s after an unrelated background
# event are different defects, and only a silent hold separates them. Nothing
# is issued during the hold, so a panic inside it belongs to the one input that
# preceded it or to nothing at all.
#
# KIND is one of:
#   rclick  -- one right-click on the dock icon (the s9 probe's own action)
#   click   -- one left-click at the same point, which is the s9 probe's
#              dismiss and arrives ~3 s before its right-click
#   move    -- one pointer move to the same point, no button at all
#   none    -- no input; the control that prices the hold itself
#
# Env: QPID (required, the qemu pid), OUT (required, a directory).
#      LEVER_SETTLE  seconds to settle before acting   (default 8)
#      LEVER_HOLD    seconds to hold untouched after   (default 90)
#      LEVER_X/Y     guest-pixel target                (default the Finder icon)
set -u

KIND="${1:-rclick}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
QMP="$REPO/scripts/qmp/qmp.py"
X="${LEVER_X:-333}"; Y="${LEVER_Y:-1033}"
SETTLE="${LEVER_SETTLE:-8}"; HOLD="${LEVER_HOLD:-90}"

[ -n "${OUT:-}" ] || { echo "input-lever: OUT unset"; exit 2; }
mkdir -p "$OUT"
[ -n "${QPID:-}" ] || { echo "input-lever: no running guest"; exit 2; }

# The serial log is the panic's own witness and it is what this probe grades.
# Its length before the input is the baseline: a panic that was already in it
# is not this input's panic, and without the mark there is no way to say so.
SER="$(ls -t "$REPO"/vm/disks/run/serial-*.log 2>/dev/null | head -1)"
[ -n "$SER" ] || { echo "input-lever: no serial log to watch"; exit 2; }
BEFORE="$(wc -c < "$SER")"

alive() { kill -0 "$QPID" 2>/dev/null; }
panicked() {
  tail -c +"$((BEFORE + 1))" "$SER" 2>/dev/null \
    | grep -q 'Debugger called: <panic>\|panic(cpu '
}

if panicked || ! alive; then
  echo "input-lever: the guest was already gone before the input; this run says nothing"
  exit 2
fi

echo "input-lever: kind=$KIND target=$X,$Y settle=${SETTLE}s hold=${HOLD}s qemu=$QPID"
sleep "$SETTLE"
if panicked || ! alive; then
  echo "input-lever: VERDICT=PANIC_IN_SETTLE"; exit 1
fi

T0="$(date +%s)"
case "$KIND" in
  rclick|click|move) python3 "$QMP" "$KIND" "$X" "$Y" >> "$OUT/action.log" 2>&1 ;;
  none)              echo "no input issued" >> "$OUT/action.log" ;;
  *) echo "input-lever: unknown kind '$KIND'"; exit 2 ;;
esac
ACT_RC=$?
echo "input-lever: issued kind=$KIND rc=$ACT_RC"

# Polled rather than slept-then-checked, because the latency from the input to
# the panic is the number that distinguishes "this input killed it" from "it
# died later while idle", and a single sleep to the end of the hold cannot
# report one.
ELAPSED=0
while [ "$ELAPSED" -lt "$HOLD" ]; do
  if panicked; then
    echo "input-lever: VERDICT=PANIC latency=${ELAPSED}s after kind=$KIND"
    exit 1
  fi
  if ! alive; then
    echo "input-lever: VERDICT=GONE_NO_PANIC latency=${ELAPSED}s after kind=$KIND"
    exit 1
  fi
  sleep 2
  ELAPSED=$(( $(date +%s) - T0 ))
done

echo "input-lever: VERDICT=SURVIVED kind=$KIND held=${ELAPSED}s"
