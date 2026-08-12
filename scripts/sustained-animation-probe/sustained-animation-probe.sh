#!/usr/bin/env bash
# Drive the guest with a SUSTAINED full-rate animation, served from the host.
# Third probe alongside `maps.sh` (one app's canvas) and `hammer.sh` (the window
# server in bursts); same (outdir, seconds) interface, so `AB_PROBE=anim.sh`
# drops it into `ab.sh`.
#
#   anim.sh <outdir> <seconds>
#
# Why it exists. `hammer.sh` sleeps between its phases — Mission Control and
# Launchpad each cost it ~2 s of wall clock — so whole seconds of a hammer boot
# have literally zero draws, and its `present_hz` median reads 2.8 Hz on a device
# that was separately observed sustaining ~79 Hz under a frame-rate test page.
# Reading a hammer boot as this device's cadence is how the engine's CPU phases
# got ranked against a workload that was idle most of the time. The two probes
# rank the costs in a different ORDER, which is the part that matters:
#
#     chain_phase share    hammer      sustained animation
#     store                 10.3 %                  34.9 %
#     engine                49.0 %                  28.2 %
#     sampled               18.5 %                  20.9 %
#     drain worker duty   0.00 med             0.22 med, 0.88 peak
#
# Only the sustained arm ever makes the drain worker the bottleneck, so it is the
# only one on which a per-draw CPU saving can turn into frames. Run both before
# any "faster" claim; a change can help one and hurt the other.
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
# The guest reaches the host at the user-net gateway. Fixed port: the guest is
# told the URL once and a random port would have to be plumbed through `open`.
PORT="${ANIM_PORT:-8123}"
GATEWAY=10.0.2.2
mkdir -p "$OUT"

# Serve from the page's own directory so nothing else in the repo is exposed to
# the guest, and bind to all interfaces because the gateway is not loopback.
python3 -m http.server "$PORT" --directory "$REPO/scripts/sustained-animation-probe" \
  >"$OUT/httpd.log" 2>&1 &
HTTPD=$!
trap 'kill $HTTPD 2>/dev/null' EXIT
sleep 1
kill -0 $HTTPD 2>/dev/null || { echo "http server failed to start:"; cat "$OUT/httpd.log"; exit 2; }

URL="http://$GATEWAY:$PORT/anim.html"
echo "serving $URL (pid $HTTPD) secs=$SECS"

# `open -a` at setup and never at drive time, so a rail with flaky ssh still
# produces a measurable window. Safari because it is on every rail from macos-11
# up and its compositor path is the one the window server actually serves.
timeout 60 ssh -o BatchMode=yes macos-vm "open -a Safari '$URL'" 2>/dev/null \
  || { echo "could not open Safari on the guest"; exit 3; }

# Safari has to fetch, lay out, and get a first frame composited before the
# measurement means anything. A page that is still blank draws nothing.
sleep 12

read -r W H < <("$Q" size) || { echo "no display size"; exit 2; }
echo "display ${W}x${H}"
# Full-screen it, so the animation drives the whole scanout rather than a window
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
# Nothing to drive: the page animates itself. That is the point — no host input
# lands during the window, so nothing in the measurement is the probe's own cost.
sleep "$SECS"

tail -c "+$(( OFFSET + 1 ))" "$FAILLOG" >"$OUT/window.log"
# Reading a captured window is not specific to this probe, so the analysis is
# not carried here. `ANIM_ANALYZE` names it; absent, the window is still written
# and the caller analyses it however it likes.
ANALYZE="${ANIM_ANALYZE:-}"
[ -n "$ANALYZE" ] && [ -f "$ANALYZE" ] && python3 "$ANALYZE" "$OUT/window.log"
exit 0
