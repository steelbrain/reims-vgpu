#!/usr/bin/env bash
# dock-hover-probe — photograph the guest's dock hover effect, and record what
# the device did while it was on screen.
#
# The bug this exists for: on macos-26 a dock hover tooltip renders as a flat
# untextured polygon with no icon highlight, and the dock's own background comes
# out mottled rather than blurred. That was reported from a hand-taken
# screenshot, which is not a regression gate — `AGENTS.md` asks for a log- or
# test-level proxy for a bug class before a visual fix lands, and there was no
# way to ask for the effect on demand at all.
#
# # Why this drives the pointer from the host
#
# The first version of this probe compiled a Quartz pointer poster on the guest,
# the way `window-drag-probe`'s `drag.c` does. That cannot work on the rail with
# the bug: **macOS 26 has no command line developer tools**, so `clang` is absent
# and the build step requests an installer dialog instead. The other guest-side
# routes are no better — `screencapture` fails outright there ("could not create
# image from display", so `guest_display_size` cannot answer), and the
# `osascript` desktop-bounds queries need Apple Events consent a fresh ssh
# session does not have.
#
# QMP's `input-send-event` reaches the machine's usb-tablet from outside the
# guest, so it needs no guest tooling, no consent and no permission, and it works
# identically on all six rails. It is the same transport `vibrancy-latch-probe`
# already drives its gestures over. The consequence worth noting is that this
# probe **does not use ssh at all** — a guest whose sshd never came up can still
# be photographed.
#
# A hover is an *arrival followed by rest*, not a coordinate: the window server
# starts its tooltip timer when the pointer stops, so the probe glides in over
# several sub-moves and then stops sending events entirely. Re-asserting the
# position on a timer would restart that timer forever and the tooltip would
# never appear.
#
# It does not judge the picture. It produces a screenshot per slot and a fail-log
# slice per slot; comparing those against a known-good rail is the reading.
#
# # The crash census, and why a screenshot needed one
#
# On macos-26 the picture is only half the defect. Three guest processes —
# `com.apple.dock.extra`, `Spotlight` and `iconservicesagent` — abort in a loop
# every few seconds, all three on a byte-identical stack: Metal's
# `+[MTLLoader sliceIDForDevice:legacyDriverVersion:airntDriverVersion:]`
# asserting while RenderBox loads a precompiled binary archive for an icon.
# A process that dies mid-composite leaves exactly the reported picture, so
# "how many aborted, and on what" is the reading the screenshot cannot give.
#
# So this harvests the guest's own crash reports and ranks them by faulting
# symbol. That is a *count*, which survives host contention, unlike anything
# timed.
#
# **The path is the whole trick, and four earlier sessions concluded these
# reports do not exist because of it.** `scp macos-vm:~/Library/...` does not
# expand the `~` in a remote glob, so it matches nothing and fails exactly as a
# missing file does; `/Library/Logs/DiagnosticReports` is the *system*
# directory and holds only some of them. A remote path with no leading `/` is
# already relative to the home directory, so `Library/Logs/...` is the spelling
# that works, and both directories are polled.
#
# The census needs ssh, which the hover half deliberately does not — a guest
# whose sshd never came up can still be photographed. So it is best-effort and
# never fails the run: no ssh means the census says so and the screenshots still
# stand. `--no-crash-census` skips it outright.
#
# Usage:
#   scripts/dock-hover-probe/dock-hover-probe.sh [--slots N] [--rest SECONDS]
#                                                [--keep DIR] [--qmp SOCK]
#                                                [--no-crash-census] [--ssh HOST]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SLOTS=5
REST=2.5
KEEP=""
CRASH_CENSUS=1
SSH_HOST="${REIMS_GUEST_SSH:-macos-vm}"
FAILLOG="${REIMS_FAIL_LOG:-/tmp/reims-vgpu-fail.log}"
# The x86 boot's stable per-boot symlink. qmp.py defaults to the arm64 path, so
# this must be passed explicitly here — the same override vibrancy-latch-probe
# makes, for the same reason.
QMP_SOCK="${QMP_SOCK:-$REPO_ROOT/vm/disks/run/qmp.sock}"

while [ $# -gt 0 ]; do
  case "$1" in
    --slots) SLOTS="$2"; shift 2 ;;
    --rest) REST="$2"; shift 2 ;;
    --keep) KEEP="$2"; shift 2 ;;
    --qmp) QMP_SOCK="$2"; shift 2 ;;
    --no-crash-census) CRASH_CENSUS=0; shift ;;
    --ssh) SSH_HOST="$2"; shift 2 ;;
    -h|--help) sed -n '2,65p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "dock-hover-probe: unknown option $1" >&2; exit 2 ;;
  esac
done

say() { echo "dock-hover-probe: $*"; }

# The screenshot is taken partway into each hover's rest, so a shorter rest would
# photograph a pointer that has already left. Refused rather than clamped: a run
# that silently ignored `--rest` would produce dock shots with no hover in them
# and nothing would say why.
if ! awk -v r="$REST" 'BEGIN { exit !(r > 1.5) }'; then
  say "--rest must be greater than 1.5 s (the shot lands 1.2 s into the rest)" >&2
  exit 2
fi

WORK="${KEEP:-$(mktemp -d)}"
mkdir -p "$WORK"

QMP="$REPO_ROOT/scripts/qmp/qmp.py"
[ -S "$QMP_SOCK" ] || { say "no QMP socket at $QMP_SOCK — is a boot running?" >&2; exit 2; }
[ -f "$FAILLOG" ] || { say "no fail log at $FAILLOG — is a boot running?" >&2; exit 2; }

read -r SW SH < <(QMP_SOCK="$QMP_SOCK" "$QMP" size 2>"$WORK/size.err")
case "${SW:-}" in
  [0-9]*) ;;
  *) say "QMP did not report a display size — see $WORK/size.err" >&2; exit 2 ;;
esac
say "guest display ${SW}x${SH}"

# The dock's own geometry is not queryable without assistive access, which five
# of six rails do not have. It does not need to be: the probe hovers a band of
# slots across the bottom centre of the screen, which is where a default dock
# sits, and reports the coordinates it used. A rail whose dock is hidden or
# repositioned produces screenshots with no dock in them — visibly a miss rather
# than a silent one.
#
# 44 px above the bottom edge is the centre of a default 64 pt icon row plus its
# margin, in points; these guests are non-Retina so a point is a pixel.
HOVER_Y=$(( SH - 44 ))
APPROACH_Y=$(( SH - 260 ))
[ "$APPROACH_Y" -lt 0 ] && APPROACH_Y=0

SHOT="$REPO_ROOT/scripts/screenshot/screenshot.sh"

span=$(( SW / 2 ))
left=$(( SW / 4 ))
captured=0
for i in $(seq 1 "$SLOTS"); do
  if [ "$SLOTS" -gt 1 ]; then
    x=$(( left + (span * (i - 1)) / (SLOTS - 1) ))
  else
    x=$(( SW / 2 ))
  fi

  before=$(wc -c < "$FAILLOG")

  # The approach: a handful of sub-moves so the window server sees a pointer
  # entering the target rather than teleporting into it. Then nothing, so the
  # tooltip timer can run.
  for step in 1 2 3 4 5 6; do
    y=$(( APPROACH_Y + (HOVER_Y - APPROACH_Y) * step / 6 ))
    QMP_SOCK="$QMP_SOCK" "$QMP" move "$x" "$y" >/dev/null 2>>"$WORK/move-$i.err"
  done

  sleep 1.2
  "$SHOT" -o "$WORK/dock-slot-$i.png" > "$WORK/shot-$i.log" 2>&1 \
    && captured=$(( captured + 1 )) \
    || say "slot $i at x=$x: host screenshot failed — see $WORK/shot-$i.log"

  # Let the rest finish before cutting the log slice, so the slice covers the
  # whole time the effect was on screen rather than only up to the shot.
  sleep "$(awk -v r="$REST" 'BEGIN { d = r - 1.2; print (d > 0 ? d : 0) }')"

  after=$(wc -c < "$FAILLOG")
  if [ "$after" -gt "$before" ]; then
    tail -c "$(( after - before ))" "$FAILLOG" > "$WORK/faillog-slot-$i.txt"
  else
    : > "$WORK/faillog-slot-$i.txt"
  fi
  say "slot $i at x=$x,y=$HOVER_Y — $(wc -l < "$WORK/faillog-slot-$i.txt") device lines"
done

if [ "$captured" -eq 0 ]; then
  say "no screenshot was captured — this run is not a measurement" >&2
  say "artifacts in $WORK" >&2
  exit 1
fi

# The union of what the device said across every hover, ranked. Fail-channel
# records only: an `OFF ` record carries `reason=` too, for ordering and
# control-flow events that are not losses, so ranking without this filter
# inverts the queue.
cat "$WORK"/faillog-slot-*.txt 2>/dev/null \
  | grep -v '^OFF ' \
  | grep -o 'reason=[a-z_0-9]*' \
  | sort | uniq -c | sort -rn > "$WORK/reasons.txt"

say "captured $captured/$SLOTS screenshots in $WORK"
if [ -s "$WORK/reasons.txt" ]; then
  say "device refusals during the hovers:"
  sed 's/^/  /' "$WORK/reasons.txt"
else
  say "no fail-channel refusal was emitted during any hover"
fi

# --- guest crash census -----------------------------------------------------
# Best-effort by construction: this half needs ssh and the hover half does not,
# so every failure below reports and returns rather than failing the run.
if [ "$CRASH_CENSUS" -eq 1 ]; then
  IPS="$WORK/ips"
  mkdir -p "$IPS"
  if ! timeout 20 ssh -o BatchMode=yes "$SSH_HOST" true >/dev/null 2>&1; then
    say "crash census skipped: no ssh to $SSH_HOST (the screenshots above still stand)"
  else
    # Both directories: the per-user one holds the icon-renderer aborts, the
    # system one holds a different subset. Neither alone is the population.
    timeout 90 scp -o BatchMode=yes \
      "$SSH_HOST:Library/Logs/DiagnosticReports/*.ips" "$IPS/" >/dev/null 2>&1
    timeout 90 scp -o BatchMode=yes \
      "$SSH_HOST:/Library/Logs/DiagnosticReports/*.ips" "$IPS/" >/dev/null 2>&1
    n=$(find "$IPS" -name '*.ips' 2>/dev/null | wc -l)
    if [ "$n" -eq 0 ]; then
      say "crash census: no .ips on the guest — no process aborted this boot"
    else
      # An .ips is a one-line JSON header followed by a JSON body. Rank by
      # (process, faulting symbol): the symbol is what says whether several
      # crashing processes are one defect or several.
      python3 - "$IPS" > "$WORK/crashes.txt" 2>/dev/null <<'PY'
import json, pathlib, sys, collections
tally = collections.Counter()
for p in sorted(pathlib.Path(sys.argv[1]).glob("*.ips")):
    try:
        body = json.loads(p.read_text().split("\n", 1)[1])
        thread = body["threads"][body["faultingThread"]]
        images = body["usedImages"]
        top = "?"
        # The first frame below the *whole* raise path is the one that names the
        # defect. That path is two layers, and stopping after the first leaves
        # every abort in the tree reading as the same `MTLReportFailure`: libc
        # raises it (`abort`/`__assert_rtn`) and the framework that failed
        # reports it (`MTLReportFailure`). Both are identical for every assert
        # and neither says which check refused.
        raise_images = (
            "libsystem_kernel.dylib", "libsystem_pthread.dylib", "libsystem_c.dylib",
        )
        raise_symbols = ("MTLReportFailure", "_MTLMessageContextEnd", "__assert_rtn")
        for f in thread["frames"]:
            sym = f.get("symbol", "")
            image = images[f["imageIndex"]].get("name") or "?"
            if not sym or image in raise_images:
                continue
            if any(sym.startswith(p) for p in raise_symbols):
                continue
            top = f"{image} {sym}"
            break
        tally[(body.get("procName", "?"), body.get("termination", {}).get("indicator", "?"), top)] += 1
    except Exception:
        tally[(p.name, "unparsed", "?")] += 1
for (proc, how, top), n in tally.most_common():
    print(f"{n:4d}  {proc}  [{how}]  {top}")
PY
      say "crash census: $n report(s) on the guest, by (process, faulting symbol):"
      if [ -s "$WORK/crashes.txt" ]; then
        sed 's/^/  /' "$WORK/crashes.txt"
      else
        say "  (reports present but none parsed — see $IPS)"
      fi
      say "reports kept in $IPS — Apple's data, never commit them"
    fi
  fi
fi
