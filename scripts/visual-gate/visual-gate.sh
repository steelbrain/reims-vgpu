#!/usr/bin/env bash
# visual-gate.sh — did this build lose any pixels?
#
# Performance work on this device has been verified against clippy, unit tests,
# the feature matrix and device-side counters. None of those can see a rendering
# regression, so a branch of performance commits was once reset off wholesale
# after the glitches showed up later: every commit had been verified, and the
# question was never asked.
#
# Three probes already answer it, each of them a two-observation instrument —
# the guest declares what it believes it drew, the host measures the frame at
# exactly those places. That is what separates "the guest declined to draw it"
# from "this device lost it", which is the distinction a screenshot cannot make.
# What was missing is that none of them gated anything.
#
# This is the one entry point. It runs all three, applies a counter budget over
# the eight silent-loss classes for its own window of the fail log, and exits
# non-zero if any part failed.
#
#   scripts/visual-gate/visual-gate.sh [--quick] [--keep DIR] [--host GUEST]
#
# Exits 0 when every probe passed and every silent-loss counter read zero, 1 on
# any regression, 2 on a setup failure — a guest that never settled, a probe
# that could not run, or a QEMU that died under it.
#
# A green `--quick` is not a phase sign-off. See the README.
set -euo pipefail
export LC_ALL=C

QUICK=0
KEEP=""
GUEST="${GUEST:-macos-vm}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FAILLOG="${REIMS_VGPU_FAIL_LOG:-/tmp/reims-vgpu-fail.log}"

# A probe on an unsettled guest measures the guest's own startup work. One such
# run read 12.2 fps with the device idle at `duty=0.001` — the desktop was not
# compositing yet. sshd answers well before that, so the port being open is not
# the signal; the 1-minute load average is.
SETTLE_LOAD="${SETTLE_LOAD:-1.0}"
SETTLE_TIMEOUT="${SETTLE_TIMEOUT:-420}"
# And the load average is not a signal either until the guest has been up longer
# than the average's own window. Measured on a fresh boot here: 40 s up reading
# 1.28, 65 s up reading 1.09 — a figure averaging over more time than the
# machine has existed, still climbing as the desktop loads. Guests that have
# settled read 0.93-0.99 at two to three minutes up.
SETTLE_MIN_UPTIME="${SETTLE_MIN_UPTIME:-180}"

while [ $# -gt 0 ]; do
  case "$1" in
    --quick) QUICK=1; shift ;;
    --keep) KEEP="$2"; shift 2 ;;
    --host) GUEST="$2"; shift 2 ;;
    -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "visual-gate: unknown argument $1" >&2; exit 2 ;;
  esac
done
export GUEST

WORK="${KEEP:-$(mktemp -d)}"
mkdir -p "$WORK"
[ -n "$KEEP" ] || trap 'rm -rf "$WORK"' EXIT
say() { echo "visual-gate: $*"; }

if [ "$QUICK" = 1 ]; then
  WEB_N=5; WALL_N=3; MODAL_N=3; MODE=quick
else
  WEB_N=20; WALL_N=10; MODAL_N=10; MODE=full
fi

# ---------------------------------------------------------------- settle wait

qemu_alive() { pgrep -f 'qemu-system-x86_6[4]' >/dev/null 2>&1; }

qemu_alive || { say "no QEMU running — boot the pathway first" >&2; exit 2; }

say "waiting for the guest to settle (up ${SETTLE_MIN_UPTIME}s, 1-minute load under $SETTLE_LOAD, giving up after ${SETTLE_TIMEOUT}s)"
settled=0
deadline=$((SECONDS + SETTLE_TIMEOUT))
while [ "$SECONDS" -lt "$deadline" ]; do
  qemu_alive || { say "QEMU exited while waiting for the guest" >&2; exit 2; }
  # Not from `uptime`, which spells its own age "40 secs", "1 min" and "1:23" by
  # turns. `kern.boottime` reports one integer in every state.
  up=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$GUEST" \
       'echo "$(date +%s) $(sysctl -n kern.boottime)" | awk "{print \$1 - (\$5+0)}"' 2>/dev/null)
  # macOS `uptime` ends with "load averages: 1.23 1.45 1.67"; the 1-minute
  # figure is the first of the three, which is $(NF-2).
  load=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$GUEST" uptime 2>/dev/null \
         | awk '{print $(NF-2)}' | tr -d ',')
  if [ -n "$up" ] && [ -n "$load" ] \
     && [ "$up" -ge "$SETTLE_MIN_UPTIME" ] \
     && awk -v l="$load" -v t="$SETTLE_LOAD" 'BEGIN{exit !(l+0 < t+0)}'; then
    say "guest settled: up ${up}s at load $load"
    settled=1
    break
  fi
  sleep 10
done
[ "$settled" = 1 ] || {
  say "guest never settled (up '${up:-unreachable}s', last load '${load:-unreachable}')" >&2
  exit 2
}

# The power state is not a verdict input — the gate answers correct-or-not, and
# frame rate is bimodal here regardless. It is recorded because a log read later
# beside a performance number needs to say which state it was taken in.
if command -v nvidia-smi >/dev/null 2>&1; then
  say "host GPU $(nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,utilization.gpu,pstate \
       --format=csv,noheader 2>/dev/null | head -1)"
fi

# ------------------------------------------------------------------- probes

# The fail log is bounded by byte offset rather than by timestamp: the device's
# own `t=` clock does not correlate with this shell's, and the accumulated log
# spans builds. Marked before the first probe so the window is exactly the
# frames this gate drove.
OFF=0
[ -f "$FAILLOG" ] && OFF=$(stat -c %s "$FAILLOG")

# Every probe runs even when an earlier one fails: which probe fails is the
# diagnostic. `set -e` is off for exactly these three lines.
run_probe() {
  local name="$1"; shift
  local log="$WORK/$name.log"
  say "running $name"
  set +e
  "$@" >"$log" 2>&1
  local rc=$?
  set -e
  # The probes' own verdict is their last non-empty line.
  local verdict
  verdict=$(grep -v '^[[:space:]]*$' "$log" | tail -1)
  printf '%s\t%s\t%s\n' "$name" "$rc" "$verdict" >>"$WORK/verdicts"
  say "  $name exit=$rc: $verdict"
  # A count of failures is not a diagnosis. The probes name the region, the
  # trial and the colour they measured on their own lines, and without those a
  # reader has to re-run with --keep to learn anything — by which time the boot
  # that produced the failure is gone. Bounded because a probe that fails every
  # trial would otherwise bury its own verdict.
  if [ "$rc" != 0 ]; then
    grep -v '^[[:space:]]*$' "$log" | head -n -1 | tail -"$FINDING_LINES" \
      | sed "s/^/visual-gate:     /"
  fi
}

# Findings echoed from a failing probe. Ten covers every trial of a `--quick`
# run and the first ten of a full one, which is enough to see whether one region
# is failing repeatedly or every region failed once — a distinction that decides
# where to look next.
FINDING_LINES="${FINDING_LINES:-10}"

: >"$WORK/verdicts"
# --keep is passed down only when this gate was asked to keep, so a failing run
# leaves every probe's frames and decode records beside its own log.
keep_arg() { [ -n "$KEEP" ] && printf -- '--keep\n%s\n' "$WORK/$1"; }

mapfile -t k < <(keep_arg web-content)
run_probe web-content "$REPO_ROOT/scripts/web-content-probe/web-content-probe.sh" \
  -n "$WEB_N" "${k[@]}"
mapfile -t k < <(keep_arg wallpaper)
run_probe wallpaper "$REPO_ROOT/scripts/wallpaper-probe/wallpaper-probe.sh" \
  -n "$WALL_N" "${k[@]}"
mapfile -t k < <(keep_arg modal-button)
run_probe modal-button "$REPO_ROOT/scripts/modal-button-probe/modal-button-probe.sh" \
  -n "$MODAL_N" --host "$GUEST" "${k[@]}"

qemu_alive || { say "QEMU exited during the probes — no verdict" >&2; exit 2; }

# ------------------------------------------------------------ counter budget

WINDOW="$WORK/window.log"
if [ -f "$FAILLOG" ]; then
  tail -c "+$((OFF + 1))" "$FAILLOG" >"$WINDOW"
else
  : >"$WINDOW"
fi

# The eight silent-loss classes, each held to the budget `baseline.tsv` records
# for it. Zero is the default and four of them keep it; the two that do not
# carry their measurement and their argument in that file, because a non-zero
# budget is an admission about the device rather than a setting.
#
# The parsing lives in its own script so `self-test.sh` can exercise it against
# synthetic log text without a boot: a parser that matches nothing prints the
# same eight zeros a clean run does.
set +e
"$SCRIPT_DIR/counter-budget.sh" "$WINDOW" >"$WORK/counters"
budget_rc=$?
set -e
[ "$budget_rc" -le 1 ] || { say "counter budget could not read its window" >&2; exit 2; }

# --------------------------------------------------------------- the verdict

probe_fails=$(awk -F'\t' '$2 == 1' "$WORK/verdicts" | wc -l)
probe_setup=$(awk -F'\t' '$2 != 0 && $2 != 1' "$WORK/verdicts" | wc -l)
# Every class that fired is named in the verdict whether or not it was inside
# its budget, so a `PASS` is never read as "nothing was lost".
seen=$(awk -F'\t' '$2 > 0 {printf "%s=%s/%s ", $1, $2, $3}' "$WORK/counters")
seen=${seen% }
over=$(awk -F'\t' '$2 > $3 {printf "%s=%s>%s ", $1, $2, $3}' "$WORK/counters")
over=${over% }

bytes=$(stat -c %s "$WINDOW")
summary="$MODE mode, $(awk -F'\t' '$2 == 0' "$WORK/verdicts" | wc -l)/3 probes green"
summary="$summary, counters over ${bytes} B of fail log: ${seen:-all zero}"

if [ "$probe_setup" -gt 0 ]; then
  say "SETUP-FAILED — $summary"
  awk -F'\t' '$2 != 0 && $2 != 1 {printf "visual-gate:   %s exit=%s: %s\n", $1, $2, $3}' "$WORK/verdicts"
  exit 2
fi

if [ "$probe_fails" -gt 0 ] || [ "$budget_rc" != 0 ]; then
  say "FAIL — $summary"
  awk -F'\t' '$2 == 1 {printf "visual-gate:   %s: %s\n", $1, $3}' "$WORK/verdicts"
  [ -z "$over" ] || say "  over budget: $over"
  exit 1
fi

say "PASS — $summary"
exit 0
