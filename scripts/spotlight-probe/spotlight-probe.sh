#!/usr/bin/env bash
# Open Spotlight, type into it, and record what the guest and this device did.
#
#   spotlight-probe.sh <outdir> <seconds>
#
# Same (outdir, seconds) interface as the other probes, so it drops into
# `kb/harness/ab.sh` as `AB_PROBE`.
#
# # Why Spotlight is worth its own probe
#
# Spotlight's window is the guest's heaviest *vibrancy* surface: a live blur of
# whatever is behind it, resized on every keystroke as results arrive, over a
# desktop that keeps compositing underneath. Nothing else a rail reaches by
# default combines a live backdrop blur with a window whose geometry changes
# several times a second, and this device has three rails that care —
# the sampled-image resolve, the render-target registry keyed on geometry, and
# the writeback.
#
# It is driven from the host over QMP and needs no guest tooling: `cmd+space`
# and typed ASCII, which is exactly what `AGENTS.md` says to reach for. That
# matters here because the failure under investigation is a *crash*, and a probe
# that needs ssh cannot survive the thing it is measuring.
#
# # What it collects, and why each is needed
#
# A crash inside the guest is invisible to every counter this device has: the
# device sees a window disappear, which is what a window closing looks like. So
# the probe takes the guest's own record afterwards — the crash reports macOS
# writes for a process that died — and it does that *after* the measured window,
# over ssh, bounded by `timeout`, so a wedged guest cannot hang the run.
set -u
OUT="${1:?outdir}"; SECS="${2:-30}"
REPO=/home/aneesiqbal/Projects/steelbrain/reims-vgpu
export QMP_SOCK="${QMP_SOCK:-$REPO/vm/disks/run/qmp.sock}"
Q="$REPO/scripts/qmp/qmp.py"
FAILLOG=/tmp/reims-vgpu-fail.log
# What to type. Several rounds of a query that matches a lot (so the results
# list grows and the window resizes) alternating with one that matches little.
QUERIES="${SPOTLIGHT_QUERIES:-safari system preferences terminal calculator}"
mkdir -p "$OUT"

# Park the pointer away from the Dock and any hover target.
"$Q" move 4 4 >/dev/null 2>&1
sleep 1

OFFSET=$(stat -c %s "$FAILLOG")
echo "spotlight probe: queries='$QUERIES' secs=$SECS"

# The pid before, so "it did not crash" is a comparison and not an observation
# that some process of that name exists. A relaunched Spotlight has the same
# name and a different pid, which is exactly the case a `pgrep` alone misreads.
timeout 30 ssh -o BatchMode=yes macos-vm 'pgrep -x Spotlight' \
  >"$OUT/pid-before.txt" 2>/dev/null

end=$(( SECONDS + SECS ))
round=0
while [ $SECONDS -lt $end ]; do
  for q in $QUERIES; do
    [ $SECONDS -lt $end ] || break
    round=$(( round + 1 ))
    # cmd+space *toggles*, so a round that starts with the window already open
    # closes it instead and types into whatever has focus. Escaping first makes
    # the starting state the same every round whatever the last one left behind
    # — without it, one round in three photographed a bare desktop.
    #
    # `meta_l` is Command in QEMU's qcode names, which is not free choice — see
    # `scripts/qmp/qmp.py`'s table.
    "$Q" key esc >/dev/null 2>&1
    sleep 1
    "$Q" key meta_l+spc >/dev/null 2>&1
    sleep 2
    # Typed a character at a time by the helper, which is what makes the window
    # re-layout repeatedly rather than once.
    "$Q" type "$q" >/dev/null 2>&1
    sleep 3
    # Photograph the *typed* state on the first round of each pass. A probe that
    # only photographs the end shows an empty search field, which is what a
    # closed Spotlight and a Spotlight that never received a keystroke both look
    # like — and the second is a probe bug reported as a clean result.
    if [ $round -le 3 ]; then
      "$REPO/scripts/screenshot-when-kde-plasma-host/screenshot-when-kde-plasma-host.sh" \
        -o "$OUT/typed-$round.png" >/dev/null 2>&1
    fi
    # Walk the results list and open the top hit: the list is what resizes the
    # window, and launching is what makes the guest tear the vibrancy surface
    # down while something else is being composited in front of it.
    "$Q" key down >/dev/null 2>&1
    sleep 1
    "$Q" key down >/dev/null 2>&1
    sleep 1
    # Escape closes it; the next round opens it again, so the blur surface is
    # created and destroyed once per round rather than living for the window.
    "$Q" key esc >/dev/null 2>&1
    sleep 1
    echo "round $round: $q" >>"$OUT/rounds.log"
  done
done

timeout 30 ssh -o BatchMode=yes macos-vm 'pgrep -x Spotlight' \
  >"$OUT/pid-after.txt" 2>/dev/null

"$REPO/scripts/screenshot-when-kde-plasma-host/screenshot-when-kde-plasma-host.sh" \
  -o "$OUT/screen.png" >/dev/null 2>&1 || echo "screenshot failed" >>"$OUT/rounds.log"

tail -c "+$(( OFFSET + 1 ))" "$FAILLOG" >"$OUT/window.log"

# The guest's own record. Bounded on the host side per `AGENTS.md`: a host-side
# `timeout` does not kill the remote process, so each of these is issued once and
# never retried.
{
  echo "===== crash reports (user) ====="
  timeout 60 ssh -o BatchMode=yes macos-vm \
    'ls -lt ~/Library/Logs/DiagnosticReports/ 2>/dev/null | head -40' 2>&1
  echo "===== crash reports (system) ====="
  timeout 60 ssh -o BatchMode=yes macos-vm \
    'ls -lt /Library/Logs/DiagnosticReports/ 2>/dev/null | head -40' 2>&1
  echo "===== is Spotlight alive ====="
  timeout 30 ssh -o BatchMode=yes macos-vm \
    'pgrep -lf Spotlight; pgrep -x Dock >/dev/null && echo dock=alive || echo dock=GONE' 2>&1
  echo "===== newest crash report body ====="
  timeout 60 ssh -o BatchMode=yes macos-vm \
    'f=$(ls -t ~/Library/Logs/DiagnosticReports/*.ips /Library/Logs/DiagnosticReports/*.ips 2>/dev/null | head -1); \
     echo "file=$f"; [ -n "$f" ] && head -120 "$f"' 2>&1
} >"$OUT/guest-reports.log" 2>&1

ANALYZE="${ANIM_ANALYZE:-}"
[ -n "$ANALYZE" ] && [ -f "$ANALYZE" ] && python3 "$ANALYZE" "$OUT/window.log"
exit 0
