#!/usr/bin/env bash
# Drive Maps.app with a SUSTAINED pan-and-zoom, from the host.
#
#   maps-probe.sh <outdir> <seconds>
#
# Same (outdir, seconds) interface as `sustained-animation-probe.sh`, so it
# drops into `perf-ab.sh` as `--probe`.
#
# Why a third probe. The animation probe measures one Safari canvas repainting
# itself at whatever rate the compositor will take, and the window-drag probe
# measures the window server moving an opaque rectangle. Maps is neither: it is
# a full-screen client drawing a tiled vector scene through its own Metal
# pipelines, and pan and zoom are the two interactions that make it redraw the
# whole scanout every frame rather than damage a corner of it. A device change
# can help a repainting canvas and do nothing here, so this is a population of
# draws in its own right and must be named as such in any result.
#
# Sustained, not bursty. `AGENTS.md` records that a probe built from discrete
# interactions spends most of its wall clock inside guest animations with zero
# draws, and that such a probe ranks this device's costs in a different ORDER
# from a sustained one. So the drive loop here never sleeps: every phase is one
# QMP invocation that itself takes seconds, and the phases run back to back for
# the whole window.
#
# Unlike the animation probe, host input DOES land inside the measured window.
# That is not contamination, it is the workload -- there is no way to pan a map
# without moving the pointer. What it means is that the probe's own cost is part
# of every number it produces, so the phase mix below is fixed and must not be
# tuned per run, or two boots stop being comparable.
set -u
OUT="${1:?outdir}"; SECS="${2:-40}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export QMP_SOCK="${QMP_SOCK:-$REPO/vm/disks/run/qmp.sock}"
Q="$REPO/scripts/qmp/qmp.py"
SHOT="$REPO/scripts/screenshot/screenshot.sh"
FAILLOG=/tmp/reims-vgpu-fail.log
mkdir -p "$OUT"

# Maps needs Apple's tile servers to draw anything but the empty graticule, and
# the rails reach the internet only through QEMU's user-net NAT. Record the
# verdict rather than failing on it: a tileless Maps still exercises the same
# client pipelines and is still a valid perf population, but a correctness
# reading taken from one would be wrong, so the answer has to be in the outdir.
timeout 30 ssh -o BatchMode=yes macos-vm \
  "curl -s -o /dev/null -w '%{http_code}' --max-time 10 https://gspe21-ssl.ls.apple.com/ 2>&1" \
  >"$OUT/tiles-reachable.txt" 2>&1
echo "tile server probe: $(cat "$OUT/tiles-reachable.txt" 2>/dev/null)"

# `open -a` at setup, never at drive time, so a rail with flaky ssh still
# produces a measurable window.
timeout 60 ssh -o BatchMode=yes macos-vm "open -a Maps" 2>/dev/null \
  || { echo "could not open Maps on the guest"; exit 3; }

# The reverted macOS snapshot presents Maps' own first-run sheet on every boot,
# but it does so late: the process and even the map behind the sheet are visible
# several seconds before the sheet accepts input. Wait for the same cold-start
# interval the map itself needs, then follow the sheet's keyboard focus order:
# its privacy link owns focus first, Tab moves to Continue, and Space activates
# it. Return is not handled by this sheet. Do this before full-screen and before
# the measurement mark: leaving the sheet up produces compositor heartbeats but
# no map draws, which looks like a slow device instead of an unstarted workload.
sleep 20
"$Q" key tab spc >/dev/null 2>&1
sleep 2
# Continue advances to the snapshot's notification opt-in sheet. It has the
# same focus order: Tab selects Not Now and Space accepts it without raising the
# host notification permission prompt.
"$Q" key tab spc >/dev/null 2>&1

# Dismissing the sheets starts Maps' own first layout and tile fetch. Keep that
# work out of the scored window; a still-blank map draws nothing.
sleep 10

# Preserve the windowed state before the probe enters macOS full screen. The
# full-screen captures below cannot tell a missing system menu bar from the OS
# intentionally hiding it, while this one can. It is setup evidence only and
# remains outside the scored window.
timeout 30 ssh -o BatchMode=yes macos-vm '
  echo "processes:"
  ps ax -o pid=,state=,command= | egrep "(/ControlCenter|/SystemUIServer)" | grep -v egrep || true
  echo "controlcenter defaults:"
  defaults read com.apple.controlcenter 2>&1 || true
  echo "host controlcenter defaults:"
  defaults -currentHost read com.apple.controlcenter 2>&1 || true
' >"$OUT/status-items.txt" 2>&1 || echo "status item diagnostics failed"
"$SHOT" -o "$OUT/windowed.png" >/dev/null 2>&1 || echo "windowed screenshot failed"

read -r W H < <("$Q" size) || { echo "no display size"; exit 2; }
echo "display ${W}x${H}"

# Full-screen so the map drives the whole scanout rather than a window inset in
# a static desktop. ctrl+cmd+f is the system-wide Enter Full Screen binding, so
# no window chrome geometry has to be guessed; `meta_l` is Command in QEMU's
# qcode names, which is not free choice.
"$Q" key ctrl+meta_l+f >/dev/null 2>&1
sleep 6

CX=$((W / 2)); CY=$((H / 2))
# Keep the whole path well inside the map view. A drag that leaves the window
# ends the gesture, and one that reaches an edge hits Maps' own chrome.
R=$((H / 5))

# Park the pointer at the centre before the mark, so the first phase starts from
# a known position and the settle is not inside the measured window.
"$Q" move "$CX" "$CY" >/dev/null 2>&1
sleep 2

"$SHOT" -o "$OUT/before.png" >/dev/null 2>&1 || echo "pre-window screenshot failed"

# Everything above is setup; the scored window starts here.
OFFSET=$(stat -c %s "$FAILLOG")
END=$(( $(date +%s) + SECS ))

# One drag invocation walks the whole path in interpolated sub-moves, so a pan
# phase is one process and one QMP connection for several seconds of continuous
# motion. Repeating the invocation per segment instead would put a process
# spawn between every pair of points and turn a sustained pan into a bursty one.
export QMP_DRAG_STEPS=6 QMP_DRAG_HOLD_S=0.012
# Fixed, not tunable per run: two boots whose phase mix differs are not two
# measurements of the same workload. Change it deliberately, and re-baseline.
ZOOM_TICKS=12

pan_box() {  # a closed rectangular circuit, so the map returns to where it began
  "$Q" drag $((CX - R)) $((CY - R)) $((CX + R)) $((CY - R)) \
            $((CX + R)) $((CY + R)) $((CX - R)) $((CY + R)) \
            $((CX - R)) $((CY - R)) >/dev/null 2>&1
}
pan_diag() {  # a zigzag, which changes direction more often than the circuit
  "$Q" drag $((CX - R)) $((CY - R)) $((CX + R)) $((CY + R)) \
            $((CX + R)) $((CY - R)) $((CX - R)) $((CY + R)) \
            $((CX - R)) $((CY - R)) >/dev/null 2>&1
}

phase=0
while [ "$(date +%s)" -lt "$END" ]; do
  case $((phase % 4)) in
    0) pan_box ;;
    # `wheel` sends its ticks over one connection, dt apart, which is the only
    # sustained zoom available -- a keyboard zoom is one discrete step per
    # invocation and would spend the phase in process spawns.
    1) "$Q" wheel up "$ZOOM_TICKS" 0.05 >/dev/null 2>&1 ;;
    2) pan_diag ;;
    # Equal and opposite, which bounds the drift but does NOT cancel it: Maps
    # clamps at the world view long before it clamps at street level, so from a
    # regional start the out-ticks hit the clamp and are discarded while the
    # in-ticks all land, and every circuit nets a little further in. A 45 s run
    # of nine circuits at 40 ticks was measured walking the scale bar from
    # 37,5 km to 10 m -- i.e. to the far clamp, where the back half of the
    # window measured a map that could not zoom any further. `ZOOM_TICKS` is
    # sized to keep the excursion inside both clamps for a window of this
    # length; a much longer run still needs a mid-zoom start.
    3) "$Q" wheel down "$ZOOM_TICKS" 0.05 >/dev/null 2>&1 ;;
  esac
  phase=$((phase + 1))
done

tail -c "+$(( OFFSET + 1 ))" "$FAILLOG" >"$OUT/window.log"
echo "drove $phase phases over ${SECS}s"

"$SHOT" -o "$OUT/after.png" >/dev/null 2>&1 || echo "post-window screenshot failed"
# Keep an out-of-window settled image beside the immediate one. The immediate
# capture says what continuous interaction actually displayed; the settled one
# separates a tile fetch still in flight from texture content that never became
# visible at all. Neither delay nor capture contributes to the scored window.
sleep 10
"$SHOT" -o "$OUT/settled.png" >/dev/null 2>&1 || echo "settled screenshot failed"

# Reading a captured window is not specific to this probe, so the analysis is
# not carried here; `MAPS_ANALYZE` names it, and absent one the window is still
# written for the caller to analyse however it likes.
ANALYZE="${MAPS_ANALYZE:-}"
[ -n "$ANALYZE" ] && [ -f "$ANALYZE" ] && python3 "$ANALYZE" "$OUT/window.log"
exit 0
