#!/usr/bin/env bash
# Drive the guest with a load whose weight is a dial, served from the host.
# Fourth probe alongside `maps.sh`, `hammer.sh` and `sustained-animation-probe`;
# same (outdir, seconds) interface, so it drops into `ab.sh` as `AB_PROBE`.
#
#   gpu-load-probe.sh <outdir> <seconds>
#
# Why it exists — and read the correction below before believing it. The
# sustained-animation probe was once **saturated** on macos-13: the guest
# produced ~26 800 draws and ~76 presented frames a second whatever the device
# did, the drain worker sat at duty 0.83, and a measured 3.9 us/chain CPU saving
# moved `draws_s` by less than 0.1 % because the worker spent what it saved
# waiting for guest work. A probe the device cannot fall behind cannot rank a
# device change, in either direction — it reports "no effect" for a real win and
# for a real regression alike.
#
# So this probe makes each guest frame heavier instead of asking for more
# frames, which is also the only thing a page *can* do: the guest's frame rate
# is its own display cadence and no content raises it.
#
# # That saturation is gone, and the sustained probe ranks changes again
#
# The reading above was taken on a slower device and through a presenter that
# clamped at ~41 frames a second. Neither holds now: the drain worker sits at
# duty 0.55-0.58 and a fast-latching boot presents 109-119 Hz. Twenty-four
# interleaved boots with the device pushed 20.6 % the wrong way
# (`REIMS_VGPU_COMPUTE_GATHER=off`), scored over their fast boots only, separate
# **disjointly** on this probe — 113.2 Hz mean against 105.5, slowest shipping
# boot above fastest slowed one — while `us/draw` overlaps across the same
# fourteen boots.
#
# So the sustained probe is the sharpest instrument in this harness, not a
# saturated one, and `present_hz` over the fast population is what to rank by.
# See `runtime::drain::census`'s `VBL_REPORT_EARLY` for the run and the
# elasticity it puts on a candidate.
#
# This probe keeps its own job, which was always the second half of the sentence
# below: loading one *particular rail* harder than a compositing page does.
#
# The load is three independent dials, set through `GPU_LOAD_ARGS`, because they
# land on three different rails:
#
#   layers=N   composited surfaces  -> the writeback / store rail
#   verts=K    per-frame WebGL buffer upload -> the **guest buffer gather** rail
#   tex=W      per-frame 2D canvas repaint   -> the sampled-image rail
#
#   GPU_LOAD_ARGS='layers=16&boxes=8&verts=90000&tex=1024' \
#     kb/harness/ab.sh /tmp/out on macos-13 25
#
# Name the dials in any result. Two boots at different `GPU_LOAD_ARGS` are two
# workloads, not two readings, exactly as two rails are two guest drivers.
#
# # Turning a dial up can make the measurement worse, and the reading that says
# # so is `drain_duty`
#
# The failure this probe exists to fix has a mirror image: a load heavy enough
# to starve the *guest* measures Safari's JavaScript engine instead of the
# device, and looks like a very slow device while doing it. The first run of
# this probe at `layers=16&boxes=8&verts=90000&tex=1024` reported a worst second
# of 11.3 Hz — and produced **1 329 draws a second at drain duty 0.12**, against
# 26 800 at 0.83 on the far lighter CSS-only probe. The device was idle seven
# eighths of the time.
#
# So read `drain_duty`'s `duty` and `draws` before believing any frame rate off
# this probe, and choose dials that push `duty` toward 1.0 with `draws` at or
# above what the sustained-animation probe produces.
#
# # What each dial is worth, measured — and the one that does not work
#
# Four driven macos-13 boots, one binary, quiesced host. `verts` builds its
# array once and only re-uploads it, so none of this is the JavaScript fill:
#
#   dials                                     duty   draws/s  gather regions/s
#   (sustained-animation probe, for scale)    0.83    26 800           427 000
#   layers=24&boxes=6                         0.68    11 525           270 807
#   layers=16&boxes=6&verts=30000&tex=512     0.14     1 390            34 976
#   layers=8&boxes=6&verts=60000              0.11     1 017            26 592
#
# **The `verts` axis does not work on this rail.** Any WebGL at all collapses
# the guest to ~1 000 draws a second at duty 0.11 — the device is then idle
# seven eighths of the time and the boot measures Safari's WebGL path, not this
# device. That is a property of the guest, not of the dial: the array is static
# and the shader is two lines. Do not use `verts` to rank a device change until
# something explains the collapse; the dial stays because the collapse is itself
# worth reproducing in one command.
#
# `layers` is the axis that works, and it is worth using for what it does to the
# *shape* rather than the size of the load: at 24 layers it drives **23.5 gather
# regions per draw against the sustained probe's 15.8**, so it is the heavier arm
# for anything about the guest buffer gather even though its absolute draw rate
# is lower. `tex` has not been measured alone.
#
# Nothing here beats the sustained-animation probe for total device load, and
# that probe is also the one shown to separate arms disjointly, so it remains the
# arm for ranking a change; this one is for loading a *particular rail* harder
# than a compositing page does.
#
# The page is served by the host over QEMU's user-net gateway (10.0.2.2), not
# fetched from the internet: a probe whose workload can change under it cannot
# be A/B'd, and the rails have no reason to have working DNS.
set -u
OUT="${1:?outdir}"; SECS="${2:-40}"
REPO=/home/aneesiqbal/Projects/steelbrain/reims-vgpu
export QMP_SOCK="${QMP_SOCK:-$REPO/vm/disks/run/qmp.sock}"
Q="$REPO/scripts/qmp/qmp.py"
FAILLOG=/tmp/reims-vgpu-fail.log
# Fixed port: the guest is told the URL once and a random port would have to be
# plumbed through `open`. A different default from the sustained probe's 8123,
# so a stray server left behind by one cannot serve the other's page.
PORT="${GPU_LOAD_PORT:-8124}"
GATEWAY=10.0.2.2
ARGS="${GPU_LOAD_ARGS:-layers=8&boxes=6}"
mkdir -p "$OUT"

# Serve from the page's own directory so nothing else in the repo is exposed.
python3 -m http.server "$PORT" --directory "$REPO/scripts/gpu-load-probe" \
  >"$OUT/httpd.log" 2>&1 &
HTTPD=$!
trap 'kill $HTTPD 2>/dev/null' EXIT
sleep 1
kill -0 $HTTPD 2>/dev/null || { echo "http server failed to start:"; cat "$OUT/httpd.log"; exit 2; }

URL="http://$GATEWAY:$PORT/load.html?$ARGS"
echo "serving $URL (pid $HTTPD) secs=$SECS"
# The dials are the workload, so they go in the output directory rather than
# only into this script's stdout — a result read later has to be able to say
# which load produced it.
printf '%s\n' "$URL" >"$OUT/load-url.txt"

# `open -a` at setup and never at drive time, so a rail with flaky ssh still
# produces a measurable window. Safari because it is on every rail from macos-11
# up and its compositor path is the one the window server actually serves.
timeout 60 ssh -o BatchMode=yes macos-vm "open -a Safari '$URL'" 2>/dev/null \
  || { echo "could not open Safari on the guest"; exit 3; }

# Safari has to fetch, lay out, compile the WebGL program and get a first frame
# composited before the measurement means anything. Longer than the sustained
# probe's 12 s because a shader compile is on this page's warm-up path.
sleep 16

read -r W H < <("$Q" size) || { echo "no display size"; exit 2; }
echo "display ${W}x${H}"
# Full-screen it, so the load drives the whole scanout rather than a window
# inset in a static desktop. ctrl+cmd+f is Safari's binding and needs no chrome
# geometry to be guessed — `meta_l` is Command in QEMU's qcode names, which is
# not free choice. Then park the pointer in a corner: a cursor resting over a
# moving layer adds hover work that is not the workload under test.
"$Q" key ctrl+meta_l+f >/dev/null 2>&1
sleep 3
"$Q" move 4 4 >/dev/null 2>&1
sleep 2

# Mark the log so only the driven window is measured.
OFFSET=$(stat -c %s "$FAILLOG")
ANALYZE="${ANIM_ANALYZE:-}"

# Every counter this device emits measures this device, so a window where the
# drain worker is idle seven eighths of the time says nothing about who held the
# other seven eighths. QEMU names its threads, so a per-thread CPU census does —
# and it is the only reading here that can tell a computing guest from a guest
# waiting on a round trip.
"$REPO/scripts/qemu-thread-census/qemu-thread-census.sh" \
  "$OUT/threads.log" "$(( SECS + 5 ))" >"$OUT/threads.err" 2>&1 &
THREADS=$!

# The guest's display sleeps, and a probe that only animates does not stop it.
# This page drives itself, so a window longer than the guest's display-sleep
# timeout receives no input at all — and when the display sleeps, `WindowServer`
# stops compositing and disables the vblank class, so the device goes to zero
# while the page's `setTimeout` chain keeps running and the boot looks alive.
# That is not a slow device and it is not a slow guest: a 145 s window read
# `delivered=1607` frozen across two vblank censuses, every QEMU thread under
# 20 % of one core, and the leg boundaries arriving exactly on schedule.
#
# One pointer move a screen-sleep timeout cannot ignore, alternating between two
# adjacent corner pixels so the cursor never rests over a moving layer.
(
  while :; do
    sleep 20
    "$Q" move 4 4 >/dev/null 2>&1
    sleep 20
    "$Q" move 5 5 >/dev/null 2>&1
  done
) &
AWAKE=$!
trap 'kill $HTTPD $THREADS $AWAKE 2>/dev/null' EXIT

case "$ARGS" in
*scan=*)
  # A scanned load walks a list of dial sets in one boot, and each leg's window
  # has to be cut at the moment the page changed rather than at a time computed
  # from it. The page announces each boundary by fetching a path this server has
  # no file for, so the 404 landing in `httpd.log` is the signal — and cutting
  # the fail log by *byte offset* when it lands needs no mapping between the
  # host's clock and the device's `t=`, which is a different clock.
  #
  # A leg's file is written when the next boundary arrives, so the last leg is
  # cut by the probe's own deadline instead.
  leg=0
  deadline=$(( SECONDS + SECS ))
  seen=0
  while [ $SECONDS -lt $deadline ]; do
    n=$(grep -c 'leg-end-' "$OUT/httpd.log" 2>/dev/null); n=${n:-0}
    if [ "$n" -gt "$seen" ]; then
      tail -c "+$(( OFFSET + 1 ))" "$FAILLOG" >"$OUT/leg${leg}.log"
      OFFSET=$(stat -c %s "$FAILLOG")
      seen=$n
      leg=$(( leg + 1 ))
    fi
    sleep 1
  done
  tail -c "+$(( OFFSET + 1 ))" "$FAILLOG" >"$OUT/leg${leg}.log"
  cp -f "$OUT/leg${leg}.log" "$OUT/window.log"
  # The dial set of each leg is in the URL the page asked for when it entered
  # that leg, so the server's log is also the record of what was measured.
  echo "===== legs the page walked ====="
  grep -o 'GET /load.html?[^ ]*' "$OUT/httpd.log" || true
  for f in "$OUT"/leg*.log; do
    echo "===== $(basename "$f") ====="
    [ -n "$ANALYZE" ] && [ -f "$ANALYZE" ] && python3 "$ANALYZE" "$f"
  done
  ;;
*)
  # Nothing to drive: the page animates itself. That is the point — no host input
  # lands during the window, so nothing in the measurement is the probe's own cost.
  sleep "$SECS"
  tail -c "+$(( OFFSET + 1 ))" "$FAILLOG" >"$OUT/window.log"
  # Reading a captured window is not specific to this probe, so the analysis is
  # not carried here. `ANIM_ANALYZE` names it — the same variable the sustained
  # probe uses, because the window format is the device's and not the probe's.
  [ -n "$ANALYZE" ] && [ -f "$ANALYZE" ] && python3 "$ANALYZE" "$OUT/window.log"
  ;;
esac
exit 0
