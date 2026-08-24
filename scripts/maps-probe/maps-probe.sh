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
SHOT="$REPO/scripts/screenshot-when-kde-plasma-host/screenshot-when-kde-plasma-host.sh"
VISUAL_GATE="$REPO/scripts/maps-probe/maps-visual-gate.sh"
# Capture at the scanout's own resolution. The helper's default 720p cap exists
# to bound the token cost of an agent reading a screenshot, and Maps' map labels
# are the smallest thing in the frame: a 1920x1080 scanout downscaled to
# 1280x719 leaves them as text-shaped blobs. Every capture this probe keeps is
# there to be read by a human -- that is the only thing that scores a frame --
# so it is kept at the resolution the thing being judged exists at.
export REIMS_SHOT_NATIVE=1
FAILLOG=/tmp/reims-vgpu-fail.log
mkdir -p "$OUT"

# Maps needs its tile servers to draw the workload, and the rails reach them
# only through QEMU's user-net NAT. An HTTP response proves the route even when
# the root resource itself is absent; curl's 000 means there was no response.
# A tileless run is not a performance population because it did not render the
# scene whose cost the probe claims to measure.
timeout 30 ssh -o BatchMode=yes macos-vm \
  "curl -s -o /dev/null -w '%{http_code}' --max-time 10 https://gspe21-ssl.ls.apple.com/ 2>&1" \
  >"$OUT/tiles-reachable.txt" 2>&1
tile_status=$(tail -c 3 "$OUT/tiles-reachable.txt" 2>/dev/null)
echo "tile server probe: $tile_status"
[ -n "$tile_status" ] && [ "$tile_status" != 000 ] || {
  echo "Maps tile service did not respond"; exit 3; }

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

# Select a declared, label-dense geographic workload after first-run handling.
# A fixed map link is probe setup, not device policy: it makes separate boots
# render the same class of scene, and it gives whoever reads the captures a
# known expectation to read them against. Re-opening it also prevents
# snapshot-restored location history from choosing an ocean or a different zoom
# level for one arm.
#
# z=14 is a dense metropolitan viewport: roads, water, park and building fills
# at a scale where a whole layer going missing is obvious to a reader, and where
# the place and street labels are drawn large enough to be legible in the
# capture. Walking z12/z14/z16/z18 over one boot, z14 was the zoom whose type
# was readable at the captured resolution.
MAP_URL='http://maps.apple.com/?ll=40.7128,-74.0060&z=14'
timeout 60 ssh -o BatchMode=yes macos-vm \
  "open -a Maps '$MAP_URL'" 2>/dev/null \
  || { echo "could not select the Maps workload"; exit 3; }
sleep 15

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

# Presentation is a scored parameter, because the two are different device
# workloads and not two views of one. Full screen gives Maps the whole scanout,
# so the frame this device publishes is the application's own output. Windowed
# leaves it inset in a desktop the window server composites around it every
# frame, which is a different mix of draws, a different scanout owner, and the
# regime a user is in most of the time.
#
# Default is `fullscreen`, which is what this probe has always scored -- every
# number recorded before this parameter existed is a full-screen number, and the
# default must keep them comparable. The mode is written to the output directory
# so a captured run always says which regime produced it; two arms differing in
# presentation are not two boots of one measurement.
PRESENTATION="${MAPS_PRESENTATION:-fullscreen}"
case "$PRESENTATION" in
  fullscreen|windowed) ;;
  *) echo "MAPS_PRESENTATION must be 'fullscreen' or 'windowed', got '$PRESENTATION'"; exit 2 ;;
esac
echo "$PRESENTATION" >"$OUT/presentation.txt"
echo "presentation $PRESENTATION"

if [ "$PRESENTATION" = fullscreen ]; then
  # ctrl+cmd+f is the system-wide Enter Full Screen binding, so no window chrome
  # geometry has to be guessed; `meta_l` is Command in QEMU's qcode names, which
  # is not free choice.
  "$Q" key ctrl+meta_l+f >/dev/null 2>&1
  sleep 6
fi

CX=$((W / 2)); CY=$((H / 2))
# Keep the whole path well inside the map view. A drag that leaves the window
# ends the gesture, and one that reaches an edge hits Maps' own chrome.
R=$((H / 12))

# Park the pointer at the centre before the mark, so the first phase starts from
# a known position and the settle is not inside the measured window.
"$Q" move "$CX" "$CY" >/dev/null 2>&1
sleep 2

# Do not start the scored input stream until there is a scene to measure. Tiles
# arrive asynchronously, so a fixed sleep can start the window over a blank map
# and report the frame rate of a workload that never began. Each attempt is
# retained as evidence; the final one becomes `before.png` and is read by a
# human against the declared scene.
ready=no
for attempt in 1 2 3 4 5 6; do
  candidate="$OUT/before-$attempt.png"
  "$SHOT" -o "$candidate" >/dev/null 2>&1 || true
  if "$VISUAL_GATE" --before "$candidate" --after "$candidate" \
      --settled "$candidate" >"$OUT/before-$attempt-verdict.txt" 2>&1; then
    cp "$candidate" "$OUT/before.png"
    ready=yes
    break
  fi
  sleep 10
done
[ "$ready" = yes ] || {
  echo "Maps did not render the declared geographic scene before measurement"
  cp "$FAILLOG" "$OUT/setup.log" 2>/dev/null || true
  cat "$OUT/before-"*-verdict.txt
  exit 1
}
# Preserve the exact pre-interaction interval for A/B diagnosis. The device log
# is append-only, so a later full-boot copy cannot recover this boundary.
cp "$FAILLOG" "$OUT/setup-ready.log" 2>/dev/null || true

# Everything above is setup; the scored window starts here.
OFFSET=$(stat -c %s "$FAILLOG")
END=$(( $(date +%s) + SECS ))

# Each pan is one closed gesture whose pointer returns to its starting point
# before release. Two separately released opposite gestures are not inverses:
# Maps applies kinetic motion to each release independently, and driven runs
# eventually reached open water while the visual gate misreported the valid sea
# canvas as lost rendering. A closed gesture names zero net displacement in the
# input contract; the endpoint hold makes its one release zero-velocity.
export QMP_DRAG_STEPS=6 QMP_DRAG_HOLD_S=0.012 QMP_DRAG_RELEASE_HOLD_S=0.15
# Fixed, not tunable per run: two boots whose phase mix differs are not two
# measurements of the same workload. Change it deliberately, and re-baseline.
# One tick is one semantic zoom step. Twelve-tick phases could outrun Maps'
# zoom animation: on a driven 20 s run the nominally inverse phases moved the
# scale bar from 2.5 km to 375 km. Letting one step settle keeps input cadence
# behind the application rather than making the requested viewport depend on
# which direction happened to be coalesced.
ZOOM_TICKS=1
ZOOM_SETTLE_S=0.4

pan_box() {
  "$Q" drag "$CX" "$CY" $((CX - R)) "$CY" $((CX + R)) "$CY" "$CX" "$CY" \
    >/dev/null 2>&1
}
pan_diag() {
  "$Q" drag "$CX" "$CY" $((CX - R)) $((CY - R)) \
    $((CX + R)) $((CY + R)) "$CX" "$CY" >/dev/null 2>&1
}

phase=0
while [ "$(date +%s)" -lt "$END" ]; do
  case $((phase % 4)) in
    0) pan_box || break ;;
    # `wheel` sends its ticks over one connection, dt apart, which is the only
    # sustained zoom available -- a keyboard zoom is one discrete step per
    # invocation and would spend the phase in process spawns.
    1)
      "$Q" wheel up "$ZOOM_TICKS" 0.05 >/dev/null 2>&1 || break
      sleep "$ZOOM_SETTLE_S"
      ;;
    2) pan_diag || break ;;
    3)
      "$Q" wheel down "$ZOOM_TICKS" 0.05 >/dev/null 2>&1 || break
      sleep "$ZOOM_SETTLE_S"
      ;;
  esac
  phase=$((phase + 1))
done

"$Q" size >/dev/null 2>&1 || {
  echo "guest display disappeared during Maps interaction"
  exit 3
}

tail -c "+$(( OFFSET + 1 ))" "$FAILLOG" >"$OUT/window.log"
echo "drove $phase phases over ${SECS}s"

# Refuse a window in which the device drew nothing, rather than reporting its
# frame rate. A probe that drove input the application never received produces a
# well-formed capture full of compositor heartbeats and near-zero draws, and
# that reads as a slow device instead of an unstarted workload -- the failure
# `window-drag-probe` already refuses a verdict for.
#
# This is not analysis, which stays with the caller: it is the same validity
# class as the display check above, and it is the device's own record of whether
# the scored input produced any work. The bound is presence, not a rate. It
# matters most in the windowed regime, where the pointer path is only inside the
# map if the guest placed its window where the drags go, so an empty window is
# the expected shape of that failure rather than a device result.
scored_draws=$(grep -o 'draws=[0-9]*' "$OUT/window.log" 2>/dev/null \
  | cut -d= -f2 | awk '{s += $1} END {print s + 0}')
echo "scored window drew $scored_draws"
if [ "$scored_draws" -eq 0 ]; then
  echo "no draws in the scored window: the driven input did not reach the map"
  echo "presentation=$PRESENTATION"
  exit 3
fi

# Preserve the free-running endpoint, but do not use its content to judge the
# renderer: navigation is valid guest state, and an ocean legitimately has no
# roads or labels. Restore the declared location outside the scored window so
# the correctness gate compares a scene whose expected content is known.
"$SHOT" -o "$OUT/motion-end.png" >/dev/null 2>&1 || echo "motion-end screenshot failed"
timeout 60 ssh -o BatchMode=yes macos-vm "open -a Maps '$MAP_URL'" 2>/dev/null \
  || { echo "could not restore the Maps workload"; exit 3; }

# A successful `open` means Maps accepted the URL, not that its asynchronous
# tile layers have arrived. Apply the same canvas check as setup, so the frames
# kept for reading are of the restored scene rather than of a blank map that had
# not caught up yet.
restored=no
for attempt in 1 2 3 4 5 6; do
  candidate="$OUT/restored-$attempt.png"
  sleep 10
  "$SHOT" -o "$candidate" >/dev/null 2>&1 || true
  if "$VISUAL_GATE" --before "$candidate" --after "$candidate" \
      --settled "$candidate" >"$OUT/restored-$attempt-verdict.txt" 2>&1; then
    restored=yes
    break
  fi
done
[ "$restored" = yes ] || {
  echo "Maps did not restore the declared geographic scene after measurement"
  cp "$FAILLOG" "$OUT/full-boot.log" 2>/dev/null || true
  cat "$OUT/restored-"*-verdict.txt
  exit 1
}

# Exercise movement after the restore rather than judging a static reload. One
# bounded pan cannot leave the declared metropolitan scene; unlike the scored
# closed gestures it deliberately ends displaced, so the captured result proves
# a real viewport update rendered.
"$Q" move "$CX" "$CY" >/dev/null 2>&1
"$Q" drag "$CX" "$CY" $((CX + R)) "$CY" >/dev/null 2>&1 \
  || { echo "could not drive the bounded visual pan"; exit 3; }
sleep 5
"$SHOT" -o "$OUT/after.png" >/dev/null 2>&1 || echo "post-restore screenshot failed"
# Keep an out-of-window settled image beside the immediate one. Neither delay
# nor capture contributes to the scored performance window.
sleep 10
"$SHOT" -o "$OUT/settled.png" >/dev/null 2>&1 || echo "settled screenshot failed"

# A nonzero verdict makes the whole run invalid: an empty canvas is cheap to
# draw, so its frame rate outranks every real one and must not be reported.
# Whether the scene that IS present is correct is not decided here -- the three
# frames named below are the ones to open and look at.
if ! "$VISUAL_GATE" --before "$OUT/before.png" --after "$OUT/after.png" \
     --settled "$OUT/settled.png" >"$OUT/visual-verdict.txt" 2>&1; then
  cat "$OUT/visual-verdict.txt"
  exit 1
fi
cat "$OUT/visual-verdict.txt"
cp "$FAILLOG" "$OUT/full-boot.log" 2>/dev/null || true

# Reading a captured window is not specific to this probe, so the analysis is
# not carried here; `MAPS_ANALYZE` names it, and absent one the window is still
# written for the caller to analyse however it likes.
ANALYZE="${MAPS_ANALYZE:-}"
[ -n "$ANALYZE" ] && [ -f "$ANALYZE" ] && python3 "$ANALYZE" "$OUT/window.log"
exit 0
