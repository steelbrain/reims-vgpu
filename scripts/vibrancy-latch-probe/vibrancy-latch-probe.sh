#!/usr/bin/env bash
# vibrancy-latch-probe.sh — does a load spike permanently break the guest's vibrancy rail?
#
# The bug this instruments: after the guest runs a heavy animated page and its
# window is dragged, macOS *vibrancy* stops working. Popups, tooltips and
# Settings panes render see-through to the wallpaper — the backdrop layer is
# missing and the blur pass is absent. It outlives the app that produced it and
# survives until the guest reboots. That last property is the whole point: this
# is a rail that crosses a threshold and never comes back, not an artifact that
# ages out. An instrument that samples only the degraded state cannot see it,
# because a broken rail and a rail that was never exercised read the same.
#
# So this probe samples the *same* idle UI twice, either side of the load, and
# reports the difference. Four phases:
#
#   settle   the guest at the desktop, nothing running
#   before   open a vibrancy-bearing pane, screenshot, collect a census window
#   load     Safari on an 8-row animated page, window repositioned throughout
#   after    close Safari, re-open the same pane, screenshot, collect again
#
# `before` and `after` do the same thing to the same guest, so every route whose
# rate moves between them is a candidate, and a route present in exactly one of
# them is a strong one. That is the reading to make first: a rail healthy in
# `before` and refusing in `after` names itself, where any single window from
# the degraded state would need a model of what the number should have been.
#
# Both screenshots are kept. The visual verdict is the user's — "you can see the
# wallpaper through it" is not a counter — so the probe never claims the bug
# reproduced or did not. It reports what the counters did and where the two PNGs
# are, and exits 0 whenever the phases ran.
#
# Usage:
#   scripts/vibrancy-latch-probe/vibrancy-latch-probe.sh [--load-seconds N]
#     [--census-seconds N] [--pane NAME] [--url URL] [--out DIR]
#
# Exits 2 on a setup failure, which includes the load phase never moving the
# window. A load that produced no motion leaves the counters idle, and an idle
# `after` window differs from `before` for reasons that have nothing to do with
# the bug.
set -euo pipefail
export LC_ALL=C

LOAD_SECONDS=30
CENSUS_SECONDS=12
# System Settings is the pane the report names, and it is present on every macOS
# version this guest runs. `open -b` by bundle id rather than by name because
# the app was renamed (System Preferences -> System Settings) and the id was not.
PANE_BUNDLE="com.apple.systempreferences"
PANE_PROC="System Settings"
# 8 rows of animation is what the report specifies; fewer does not reproduce it.
URL="https://testufo.com/framerates#count=8"
OUT=""
GUEST="${GUEST:-macos-vm}"
FAILLOG="${REIMS_FAIL_LOG:-/tmp/reims-vgpu-fail.log}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SHOT="$REPO_ROOT/scripts/screenshot-when-kde-plasma-host/screenshot-when-kde-plasma-host.sh"
# The x86 boot's stable per-boot symlink; the drag rides `input-send-event`
# through it. See scripts/qmp/README.md — the arm64 default does not apply here.
QMP_SOCK="${QMP_SOCK:-$REPO_ROOT/vm/disks/run/qmp.sock}"

while [ $# -gt 0 ]; do
  case "$1" in
    --load-seconds) LOAD_SECONDS="$2"; shift 2 ;;
    --census-seconds) CENSUS_SECONDS="$2"; shift 2 ;;
    --pane) PANE_PROC="$2"; shift 2 ;;
    --url) URL="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,44p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "vibrancy-latch-probe: unknown argument $1" >&2; exit 2 ;;
  esac
done

WORK="${OUT:-$(mktemp -d -t vibrancy-latch-XXXXXX)}"
mkdir -p "$WORK"
say() { echo "vibrancy-latch-probe: $*"; }
osa() { ssh -o BatchMode=yes "$GUEST" "osascript -e '$1'" 2>/dev/null; }
sh_guest() { ssh -o BatchMode=yes "$GUEST" "$1" 2>/dev/null; }

ssh -o ConnectTimeout=8 -o BatchMode=yes "$GUEST" true 2>/dev/null || {
  say "no guest at $GUEST" >&2; exit 2; }
[ -f "$FAILLOG" ] || { say "no fail log at $FAILLOG — is a boot running?" >&2; exit 2; }

# Collect one census stretch: mark the log by byte offset, wait, keep the tail.
# Byte offset rather than the log's `t=`, which is device time and not this
# shell's clock.
collect() {
  local label="$1" secs="$2" off
  off=$(stat -c %s "$FAILLOG")
  sleep "$secs"
  tail -c "+$((off + 1))" "$FAILLOG" >"$WORK/$label.log"
  say "$label: $(wc -l <"$WORK/$label.log") log lines over ${secs}s"
}

# Quit every app with a window, not only the ones this probe opened, and wait
# for them to actually go.
#
# Two reasons it has to be every app. A pane still tearing down while the next
# phase samples puts its own teardown traffic in the window. And an app left
# running from before the probe started — a snapshot that was captured with
# Notes open is enough — sits in front of the pane the next phase raises, so the
# screenshot photographs the wrong window and the visual verdict is lost. That
# happened on a 240 s run: both census phases named the pane, `after` measured a
# fully idle guest, and the PNG showed a Notes window.
#
# Finder is exempt because quitting it takes the desktop with it.
quiesce() {
  # `tell application p` where p is a repeat variable binds an item reference,
  # not the name, and the quit silently does nothing — which is how Safari
  # survived a quiesce and stood behind the pane for the whole `after` phase.
  # `(p as text)` is the fix, and the kill below is what makes the outcome
  # independent of whether any app decided to argue about it.
  osa 'tell application "System Events" to set doomed to name of every process whose background only is false and name is not "Finder"
      repeat with p in doomed
        try
          tell application (p as text) to quit
        end try
      end repeat' >/dev/null 2>&1 || true
  sleep 5
  sh_guest 'for a in $(osascript -e "tell application \"System Events\" to get name of every process whose background only is false and name is not \"Finder\"" | tr "," "\n" | sed "s/^ *//"); do killall "$a" 2>/dev/null; done' >/dev/null 2>&1 || true
  sleep 3
}

# Open the pane and return only once it is the frontmost process — the vibrancy
# pass runs on the window that is on screen and in front, so a phase that
# samples before the raise lands measures whatever was there instead. Returns
# nonzero when the raise never took, which the caller reports rather than
# silently photographing another app's window.
open_pane() {
  sh_guest "open -b '$PANE_BUNDLE'" >/dev/null || true
  local i front
  for i in 1 2 3 4 5 6 7 8 9 10; do
    sleep 2
    osa "tell application \"System Events\" to tell process \"$PANE_PROC\" to set frontmost to true" >/dev/null 2>&1 || true
    front=$(osa 'tell application "System Events" to get name of first process whose frontmost is true' || true)
    [ "$front" = "$PANE_PROC" ] && { sleep 3; return 0; }
  done
  say "the pane never came to the front (frontmost is '${front:-unknown}')" >&2
  return 1
}

say "work dir $WORK"
quiesce

# ---- before -------------------------------------------------------------
say "phase before: opening $PANE_PROC"
open_pane || { say "the baseline pane never composited; there is nothing for \
\`after\` to be compared against" >&2; exit 2; }
"$SHOT" -o "$WORK/before.png" >/dev/null 2>&1 || say "before screenshot failed" >&2
collect before "$CENSUS_SECONDS"
quiesce

# ---- load ---------------------------------------------------------------
say "phase load: $URL for ${LOAD_SECONDS}s with the window in motion"
sh_guest "open -a Safari '$URL'" >/dev/null || { say "could not open Safari" >&2; exit 2; }
sleep 10
# Parked at an x the motion loop never visits, so the mid-run sample below
# cannot read the start position back by coincidence.
osa 'tell application "System Events" to tell process "Safari" to set position of window 1 to {160, 180}' >/dev/null 2>&1 || true
osa 'tell application "System Events" to tell process "Safari" to set size of window 1 to {1000, 640}' >/dev/null 2>&1 || true
sleep 2

START_POS=$(osa 'tell application "System Events" to tell process "Safari" to get position of window 1' || true)
LOAD_OFF=$(stat -c %s "$FAILLOG")

# A real pointer drag, through QEMU's own usb-tablet.
#
# The report is "run testufo at count=8 **and drag the window around**", and the
# second half was the part this probe could not do. `set position` teleports the
# window, and a window the AX API moves does not take the window server's drag
# path. `window-drag-probe` records why a *guest-side* posted drag cannot be
# arranged — the posting process is not trusted for Accessibility and TCC.db is
# unwritable under SIP — and settles for repositioning for the same reason.
#
# Neither needed to. The machine already has a usb-tablet, and an
# `input-send-event` pointer stream reaches the window server as a real mouse
# with no trust to arrange. `drag-load.py` drives that, and keeps the resize and
# second-app churn alongside it because those are a different stressor: surface
# *reallocation*, which a window that only moves never does, and which a previous
# session measured as the thing that moves `type4_pages_stale` off zero.
"$REPO_ROOT/scripts/vibrancy-latch-probe/drag-load.py" \
  --seconds "$LOAD_SECONDS" --guest "$GUEST" --app Safari --qmp "$QMP_SOCK" \
  >"$WORK/load.count" 2>"$WORK/load.err" &
LOAD_PID=$!

# Sample the window's real position *during* the motion, not after it. The
# destinations cycle, so where the window stops is whichever one the last
# iteration reached — comparing that to the start reports "never moved" for a run
# that moved many times and came back. What the check is for is the case where
# the pointer stream went nowhere, and only a mid-run sample can see that.
sleep 12
MID_POS=$(osa 'tell application "System Events" to tell process "Safari" to get position of window 1' || true)
"$SHOT" -o "$WORK/load.png" >/dev/null 2>&1 || true
wait "$LOAD_PID" || {
  say "the load phase did not run — see $WORK/load.err" >&2
  sed 's/^/  /' "$WORK/load.err" >&2; exit 2; }

tail -c "+$((LOAD_OFF + 1))" "$FAILLOG" >"$WORK/load.log"
say "load: $(cat "$WORK/load.count"), window ($START_POS) mid ($MID_POS)"

# A load that never moved the window leaves the counters idle, and an idle
# `after` differs from `before` for reasons unrelated to the bug.
if [ "$MID_POS" = "$START_POS" ] || [ -z "$MID_POS" ]; then
  say "the window never moved — the load phase measured an idle guest and the \
diff below would not be about this bug" >&2
  exit 2
fi

# ---- after --------------------------------------------------------------
quiesce
say "phase after: re-opening $PANE_PROC"
# A pane that will not come to the front after the load is itself a finding —
# but it is a different one from the see-through class, and an `after` window
# sampled with the pane behind another app is an idle census, not a degraded
# one. Say which happened rather than diffing the two.
open_pane || say "the pane did not raise after the load; the \`after\` census \
below is not of the pane and the diff is not about this bug" >&2
"$SHOT" -o "$WORK/after.png" >/dev/null 2>&1 || say "after screenshot failed" >&2
collect after "$CENSUS_SECONDS"

# ---- diff ---------------------------------------------------------------
python3 - "$WORK/before.log" "$WORK/after.log" "$CENSUS_SECONDS" <<'PY'
import re, sys
from collections import defaultdict


def routes(path, secs):
    """Per-second rates for every `store_routes` key in one phase's log.

    The line is free-form `key=count` pairs drained once a second, so summing
    the windows and dividing by the phase length gives a rate that survives a
    phase being a different number of windows than the other.
    """
    total, windows = defaultdict(int), 0
    for line in open(path, errors="replace"):
        if " store_routes " not in f" {line}":
            continue
        windows += 1
        for k, v in re.findall(r"\b([a-z0-9_]+)=(\d+)\b", line):
            total[k] += int(v)
    span = max(windows, 1)
    return {k: v / span for k, v in total.items()}, windows


before, nb = routes(sys.argv[1], float(sys.argv[3]))
after, na = routes(sys.argv[2], float(sys.argv[3]))
print(f"\nstore_routes windows: before={nb} after={na}")

keys = sorted(set(before) | set(after))
appeared = [k for k in keys if k not in before and after.get(k, 0) > 0]
vanished = [k for k in keys if k not in after and before.get(k, 0) > 0]


def block(title, names, note):
    print(f"\n{title}  ({len(names)})")
    if note:
        print(f"  {note}")
    for k in names:
        b, a = before.get(k, 0.0), after.get(k, 0.0)
        print(f"  {k:<42} before={b:>12.2f}  after={a:>12.2f}")


block("ROUTES ONLY IN `after` — a rail that started refusing", appeared,
      "the strongest signal: absent while the pane composited correctly, "
      "present once it did not")
block("ROUTES ONLY IN `before` — a rail that stopped running", vanished,
      "a route that carried the pane's work and no longer does")

# Both phases open the same pane on the same guest, so a rate that moved by more
# than a small factor is a change in behaviour rather than in sampling. Counts
# below a floor are excluded: a route that went 1 -> 3 per window is noise at
# this sample size and would bury the rows that matter.
moved = []
for k in keys:
    b, a = before.get(k, 0.0), after.get(k, 0.0)
    if max(b, a) < 5:
        continue
    if k in appeared or k in vanished:
        continue
    ratio = (a + 1e-9) / (b + 1e-9)
    if ratio > 2.0 or ratio < 0.5:
        moved.append((ratio, k, b, a))
moved.sort(key=lambda r: -abs(r[0] - 1))
print(f"\nROUTES THAT MOVED BY MORE THAN 2x  ({len(moved)})")
for ratio, k, b, a in moved:
    print(f"  {k:<42} before={b:>12.2f}  after={a:>12.2f}  x{ratio:.2f}")

# The fail channel, ranked. Off-channel records carry `reason=` too, for
# ordering events that are not losses, so they are dropped before ranking —
# keeping them inverts the queue.
for label, path in (("before", sys.argv[1]), ("after", sys.argv[2])):
    reasons = defaultdict(int)
    for line in open(path, errors="replace"):
        if line.startswith("OFF "):
            continue
        m = re.search(r"reason=([a-z0-9_]+)", line)
        if m:
            reasons[m.group(1)] += 1
    top = sorted(reasons.items(), key=lambda kv: -kv[1])[:15]
    print(f"\nfail-channel reasons in `{label}`  ({sum(reasons.values())} records)")
    for r, n in top:
        print(f"  {r:<42} {n}")
PY

say "screenshots: $WORK/before.png  $WORK/load.png  $WORK/after.png"
say "logs kept in $WORK"
exit 0
