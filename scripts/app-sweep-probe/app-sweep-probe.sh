#!/usr/bin/env bash
# app-sweep-probe.sh — does this device survive a guest's stock applications?
#
# The existing probes each drive one motion: `window-drag-probe` repositions a
# window, `sustained-animation-probe` runs a rAF loop, `spotlight-probe` opens
# one panel. All three measure throughput on a workload this repository chose.
# None of them answers the question a release has to answer, which is whether an
# ordinary user opening ordinary applications gets a freeze or a panic.
#
# Those applications are not interchangeable stressors. Each one reaches a
# different corner of the guest's own Metal use:
#
#   Safari      layer-backed compositing, video decode surfaces, tab churn
#   Maps        a continuously-redrawing MapKit view — the only stock app that
#               holds a high-frequency render loop open with no user input
#   App Store   an asynchronously-populated collection view over the network,
#               so its draws arrive in bursts separated by real idle
#   Contacts    a split view with a vibrancy sidebar (the material rail)
#   Reminders   list animation and sheet presentation
#   Launchpad   a full-screen blur of the whole desktop — the largest single
#               composite any of these guests performs — then a live re-filter
#               of the icon grid on every keystroke, and finally a real app
#               launch (Screenshot.app) out of that layer as it tears down
#
# # What counts as a failure
#
# Three verdicts, and they are not the same failure:
#
#   PANIC   the guest kernel died. Read from the boot's own serial log, not from
#           here — `vm/boot-x86.sh` exits 126 and prints the line. This probe
#           reports the guest going unreachable; the boot script names it.
#   FREEZE  the device stopped producing frames while the boot was still up.
#           AGENTS.md's rule is that a log which *stops* is a different failure
#           from one that complains, and every census here is written at the end
#           of a drain tranche — so a wedged drain thread emits nothing at all
#           while the boot reads healthy. The freeze test is therefore a gap in
#           `drain_duty` lines wider than FREEZE_GAP_S, measured per app.
#   NO-FRAMES
#           the leg ran and this device presented nothing at all. The FREEZE test
#           above cannot see it: a drain worker whose GPU has been reset goes on
#           writing one census a second with zero draws in it, so the gap stays
#           at 1.0 and the leg reads `ok` at 0.0 Hz. One driven macos-11 boot
#           reported five consecutive legs that way after Maps wedged the GPU,
#           and a sweep judged on the verdict column would have called it 6/7.
#   STALL   the guest is up and the device is drawing, but the screen did not
#           change between two different applications. That is a compositor
#           that latched, and no counter in this device can see it — only the
#           screenshots can, which is why they are hashed rather than kept for
#           a human to leaf through.
#
# A slow app is not a failure and this probe does not rank one. `present_hz` is
# reported per app because a collapse to single digits is worth seeing next to
# the verdict, but AGENTS.md is explicit that a bursty interaction probe
# measures the gaps between its bursts — so these rates are not comparable with
# `sustained-animation-probe`'s and must not be quoted against a code change.
#
# # Why the driving is host-side
#
# Everything except launching an app rides QMP to the machine's usb-tablet and
# usb-kbd. AGENTS.md's rule: a guest-side probe does not degrade gracefully on
# macOS 26, it fails to build, and a second path that works on five rails of six
# rots. Launching still needs ssh, because there is no host-side way to say
# "open Maps" — so that one step is bounded by `timeout` and its failure is
# reported as a launch failure rather than read as a device result.
#
# Usage:
#   scripts/app-sweep-probe/app-sweep-probe.sh [--apps "Safari,Maps,..."]
#     [--seconds N] [--torture-seconds N] [--shots DIR] [--rail NAME] [--keep DIR]
#
# Exits 0 when every app ran and none of the three verdicts fired, 1 when a
# verdict fired, 2 on a setup failure (no guest, no fail log, no QMP socket).
set -euo pipefail
export LC_ALL=C

APPS="Safari,Maps,App Store,Contacts,Reminders"
SECONDS_PER_APP=20
TORTURE_SECONDS=45
SHOTS=""
RAIL="${RAIL:-}"
KEEP=""
GUEST="${GUEST:-macos-vm}"
FAILLOG="${REIMS_FAIL_LOG:-/tmp/reims-vgpu-fail.log}"
# A drain census lands once per tranche and the interval is ~1 s, so a gap this
# wide is not a slow second — it is the drain thread not returning.
FREEZE_GAP_S="${FREEZE_GAP_S:-8}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && cd .. && pwd)"
QMP_SOCK="${QMP_SOCK:-$REPO/vm/disks/run/qmp.sock}"

while [ $# -gt 0 ]; do
  case "$1" in
    --apps) APPS="$2"; shift 2 ;;
    --seconds) SECONDS_PER_APP="$2"; shift 2 ;;
    --torture-seconds) TORTURE_SECONDS="$2"; shift 2 ;;
    --shots) SHOTS="$2"; shift 2 ;;
    --rail) RAIL="$2"; shift 2 ;;
    --keep) KEEP="$2"; shift 2 ;;
    -h|--help) sed -n '2,60p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "app-sweep-probe: unknown argument $1" >&2; exit 2 ;;
  esac
done

WORK="${KEEP:-$(mktemp -d)}"
mkdir -p "$WORK"
[ -n "$KEEP" ] || trap 'rm -rf "$WORK"' EXIT
[ -n "$SHOTS" ] && mkdir -p "$SHOTS"
say() { echo "app-sweep-probe: $*"; }

qmp() { QMP_SOCK="$QMP_SOCK" timeout 30 "$REPO/scripts/qmp/qmp.py" "$@" >/dev/null 2>&1 || true; }
gssh() { timeout "${1:-20}" ssh -o BatchMode=yes -o ConnectTimeout=8 "$GUEST" "$2" 2>/dev/null; }

# End the measured application's process before the next leg. Command-Q is the
# ordinary path, but a first-run sheet may consume or defer it while continuing
# to own the desktop. That made every later leg measure the sheet and report
# zero frames from applications that never became frontmost. Force termination
# is test cleanup only, and only runs when the process survived Command-Q.
close_app() {
  local app="$1"
  qmp key meta_l+q
  sleep 2
  if gssh 10 "pgrep -x '$app' >/dev/null"; then
    gssh 10 "killall '$app' >/dev/null 2>&1" || true
    sleep 2
  fi
}

[ -S "$QMP_SOCK" ] || { say "no QMP socket at $QMP_SOCK — is a boot running?" >&2; exit 2; }
[ -f "$FAILLOG" ] || { say "no fail log at $FAILLOG — is a boot running?" >&2; exit 2; }
gssh 10 true || { say "no guest at $GUEST" >&2; exit 2; }

# QMP sizes from QEMU's `DisplaySurface`, which under the host-owned window is
# not the surface the guest is drawing into — it only usually still carries the
# right dimensions. On macos-11 it has been seen reporting **1920x24** for a
# composited desktop, and a probe that believes that drives a geometry no result
# can be attributed to.
#
# So the answer is retried, and if it stays implausible the device's own
# published scanout size is used instead: `host_window_start id=1 WxH` is what
# this device told the window system, which is the strongest available statement
# about how many pixels the guest has. Whichever source won is named, because a
# run driven against a fallback geometry is a different measurement from one
# driven against the guest's own.
#
# 400 rows is the floor. It is far below any desktop these rails boot at and far
# above the degenerate values seen, so it separates the two without needing to
# know the rail's real size.
MIN_ROWS=400
SIZE=""
for _ in 1 2 3 4 5; do
  SIZE=$(QMP_SOCK="$QMP_SOCK" timeout 20 "$REPO/scripts/qmp/qmp.py" size 2>/dev/null || echo "")
  H=$(echo "$SIZE" | awk '{print $2}')
  case "${H:-}" in ''|*[!0-9]*) ;; *) [ "$H" -ge "$MIN_ROWS" ] && break ;; esac
  sleep 2
done
W=$(echo "$SIZE" | awk '{print $1}')
H=$(echo "$SIZE" | awk '{print $2}')
SIZE_FROM=qmp
case "${W:-}${H:-}" in
  ''|*[!0-9]*) W=""; H="" ;;
  *) [ "$H" -lt "$MIN_ROWS" ] && { W=""; H=""; } ;;
esac
if [ -z "${H:-}" ]; then
  DEV_SIZE=$(grep -m1 -oE 'host_window_start id=1 [0-9]+x[0-9]+' "$FAILLOG" 2>/dev/null \
    | grep -oE '[0-9]+x[0-9]+' || true)
  W=${DEV_SIZE%x*}
  H=${DEV_SIZE#*x}
  SIZE_FROM=device
fi
case "${W:-}${H:-}" in
  ''|*[!0-9]*)
    say "neither QMP nor the device would report a usable display size \
(qmp said '$SIZE') — a run driven at an unknown geometry is not a measurement" >&2
    exit 2 ;;
esac
say "rail=${RAIL:-unknown} display=${W}x${H} (from $SIZE_FROM) apps=[$APPS] ${SECONDS_PER_APP}s each"

# The device's own name for this boot, so a result can be attributed the way
# AGENTS.md requires. `vk_caps` lands once per device creation.
grep -m1 -o 'host_pointer_import=[a-z_]*' "$FAILLOG" 2>/dev/null | sed 's/^/app-sweep-probe: /' || true

VERDICTS="$WORK/verdicts.tsv"
: >"$VERDICTS"
PREV_HASH=""
PREV_APP=""

# One app: launch it, drive it, photograph it, and read the slice of the fail
# log it produced. Everything that can fail says which failure it was.
run_app() {
  local app="$1" secs="$2" slug off gap cad_n hz_min hz_med drain_n alarms live
  local shot="" hash=""
  slug=$(echo "$app" | tr 'A-Z ' 'a-z-')

  off=$(stat -c %s "$FAILLOG")
  # `open -a` returns as soon as LaunchServices accepts the request, so its
  # success is not the app being up; the wait below is what establishes that.
  gssh 30 "open -a '$app'" >/dev/null || {
    printf '%s\tLAUNCH-FAILED\t-\t-\t-\t-\n' "$app" >>"$VERDICTS"
    say "$app: could not be launched (ssh/open failed)"
    return 1
  }

  # A cold app on a cold rail can take a while to have a window. Wait for the
  # process, then give the first frame room to composite — sshd answering is
  # not the desktop having drawn, and the same is true of an app.
  local waited=0
  while [ "$waited" -lt 40 ]; do
    gssh 10 "pgrep -f '$app' >/dev/null" && break
    sleep 2; waited=$((waited + 2))
  done
  sleep 4

  # Drive it from the host. A pointer sweep across the window plus wheel ticks
  # is the most compositing per second obtainable without knowing the app's
  # layout, and it needs no consent, no assistive access and no guest tooling.
  local t_end=$((SECONDS + secs))
  while [ "$SECONDS" -lt "$t_end" ]; do
    qmp move $((W / 4)) $((H / 3))
    qmp wheel down 12 0.03
    qmp move $((W * 3 / 4)) $((H * 2 / 3))
    qmp wheel up 12 0.03
  done

  # Photographed unconditionally, because the frame is evidence and not a
  # keepsake: two consecutive apps hashing identically is this probe's only
  # detector for a host window that has stopped updating while the census keeps
  # ticking from another thread. Gating it on `--shots` disabled the detector on
  # exactly the run that exists to find hangs. `--shots` chooses whether the
  # frame is *kept*, not whether it is taken.
  shot="$WORK/$slug.png"
  if "$REPO/scripts/screenshot-when-kde-plasma-host/screenshot-when-kde-plasma-host.sh" \
    -o "$shot" >/dev/null 2>&1 && [ -f "$shot" ]; then
    hash=$(sha256sum "$shot" | cut -c1-16)
    [ -n "$SHOTS" ] && cp -f "$shot" "$SHOTS/${RAIL:-rail}-$slug.png"
  else
    shot=""
  fi

  # Is the guest still there? A panic is the boot script's verdict to report,
  # but an unreachable guest is this probe's evidence for one.
  live=ok; gssh 15 true || live=UNREACHABLE

  tail -c "+$((off + 1))" "$FAILLOG" >"$WORK/$slug.log"
  read -r gap cad_n hz_min hz_med drain_n alarms < <(
    python3 "$REPO/scripts/app-sweep-probe/read_window.py" "$WORK/$slug.log")

  local verdict=ok
  awk -v g="$gap" -v f="$FREEZE_GAP_S" 'BEGIN{exit !(g > f)}' && verdict=FREEZE
  [ "$drain_n" = 0 ] && verdict=FREEZE
  # A leg that presented **nothing** is not a slow leg. The census-gap test
  # cannot see this: the drain worker goes on writing one census a second with
  # zero draws in it, so `gap` stays at 1.0 and the leg reads `ok` at 0.0 Hz.
  # One driven macos-11 boot reported five consecutive legs that way after Maps
  # wedged the GPU — `ok  present_hz med=0.0 min=0.0  worst census gap=1.0s` —
  # and a sweep judged on the verdict column would have called it 6/7.
  awk -v m="$hz_med" 'BEGIN{exit !(m + 0 <= 0)}' && verdict=NO-FRAMES
  [ "$live" = UNREACHABLE ] && verdict=GUEST-GONE
  if [ -n "$hash" ] && [ "$hash" = "$PREV_HASH" ]; then
    verdict="STALL(screen identical to $PREV_APP)"
  fi
  [ -n "$hash" ] && { PREV_HASH="$hash"; PREV_APP="$app"; }

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$app" "$verdict" "$hz_med" "$hz_min" "$gap" "$alarms" \
    >>"$VERDICTS"
  say "$app: $verdict  present_hz med=$hz_med min=$hz_min  worst census gap=${gap}s  alarms=$alarms"
  [ "$verdict" = ok ]
}

FAILED=0
IFS=',' read -ra APP_LIST <<<"$APPS"
for app in "${APP_LIST[@]}"; do
  run_app "$app" "$SECONDS_PER_APP" || FAILED=1
  # Close it so the next app is not composited behind a growing stack, which
  # would make each later app's reading a different workload from the first's.
  close_app "$app"
done

# Safari again, harder. The sweep above opens an app and pushes a wheel at it;
# this is the tab churn and window resizing that the sweep does not reach, run
# for long enough that a leak or a growing cache has somewhere to show.
if [ "$TORTURE_SECONDS" -gt 0 ]; then
  say "Safari torture for ${TORTURE_SECONDS}s"
  off=$(stat -c %s "$FAILLOG")
  gssh 30 "open -a Safari" >/dev/null || true
  sleep 6
  t_end=$((SECONDS + TORTURE_SECONDS))
  while [ "$SECONDS" -lt "$t_end" ]; do
    qmp key meta_l+t                      # new tab
    qmp type 'apple.com'
    qmp key ret
    qmp wheel down 25 0.02
    qmp key meta_l+shift+bracket_right    # next tab
    qmp wheel up 25 0.02
    qmp drag $((W / 2)) 14 $((W / 3)) 200 $((W / 2)) 14
    qmp key meta_l+w                      # close tab
  done
  if [ -n "$SHOTS" ]; then
    "$REPO/scripts/screenshot-when-kde-plasma-host/screenshot-when-kde-plasma-host.sh" \
      -o "$SHOTS/${RAIL:-rail}-safari-torture.png" >/dev/null 2>&1 || true
  fi
  tail -c "+$((off + 1))" "$FAILLOG" >"$WORK/safari-torture.log"
  read -r gap cad_n hz_min hz_med drain_n alarms < <(
    python3 "$REPO/scripts/app-sweep-probe/read_window.py" "$WORK/safari-torture.log")
  verdict=ok
  awk -v g="$gap" -v f="$FREEZE_GAP_S" 'BEGIN{exit !(g > f)}' && { verdict=FREEZE; FAILED=1; }
  awk -v m="$hz_med" 'BEGIN{exit !(m + 0 <= 0)}' && { verdict=NO-FRAMES; FAILED=1; }
  gssh 15 true || { verdict=GUEST-GONE; FAILED=1; }
  printf 'Safari-torture\t%s\t%s\t%s\t%s\t%s\n' "$verdict" "$hz_med" "$hz_min" "$gap" "$alarms" \
    >>"$VERDICTS"
  say "Safari torture: $verdict  present_hz med=$hz_med min=$hz_min  worst census gap=${gap}s  alarms=$alarms"
  close_app Safari
fi

# Launchpad last, and it is two composites rather than one. Opening it blurs
# whatever is on the desktop — the largest single composite these guests
# perform, and larger here than against a bare wallpaper because the apps above
# have been opened and closed behind it. Then typing into it *re-filters the
# grid live on every keystroke*, which is a second and quite different load:
# icons fading out and the survivors re-laid-out, once per character.
#
# So the run does not stop at a photograph of Launchpad. It types `screenshot`
# and presses Return, which launches Screenshot.app — a real app launch out of
# the blurred full-screen layer, with Launchpad tearing down underneath it. That
# teardown-plus-launch is the step that has to survive, not the blur.
#
# It is opened with `open -a Launchpad` and not with F4. F4 is the glyph on an
# Apple keyboard, where the key sends a HID *consumer* usage that macOS routes to
# Launchpad; QMP's usb-kbd sends the plain F4 keycode, which is bound to nothing.
# Photographed on macos-13 and macos-14: the desktop did not change, the eight
# characters below went into whichever app was still frontmost, and the leg
# reported on a Maps dialog for as long as the comment here claimed F4 was "the
# stock binding on every rail". LaunchServices is the one route that behaves the
# same on all four rails, and it needs no Apple Events consent.
#
# The frame before and after says whether it opened. A full-screen blur is the
# largest composite these guests perform, so an identical hash across it means
# nothing happened — either Launchpad did not come up, or the window is stuck,
# and both are results this leg must not report as a pass.
say "Launchpad -> Screenshot"
off=$(stat -c %s "$FAILLOG")
BEFORE="$WORK/launchpad-before.png"
"$REPO/scripts/screenshot-when-kde-plasma-host/screenshot-when-kde-plasma-host.sh" \
  -o "$BEFORE" >/dev/null 2>&1 || true
gssh 20 'open -a Launchpad' || true
sleep 5
LP_SHOT="$WORK/launchpad.png"
LP_OPENED=unknown
if "$REPO/scripts/screenshot-when-kde-plasma-host/screenshot-when-kde-plasma-host.sh" \
  -o "$LP_SHOT" >/dev/null 2>&1 && [ -f "$LP_SHOT" ] && [ -f "$BEFORE" ]; then
  if [ "$(sha256sum <"$BEFORE" | cut -c1-16)" = "$(sha256sum <"$LP_SHOT" | cut -c1-16)" ]; then
    LP_OPENED=no
  else
    LP_OPENED=yes
  fi
  [ -n "$SHOTS" ] && cp -f "$LP_SHOT" "$SHOTS/${RAIL:-rail}-launchpad.png"
fi
# Launchpad's search field has focus the moment it opens, so the characters go
# to it without a click. One per keystroke re-filter, then Return launches the
# first hit.
qmp type 'screenshot'
sleep 3
qmp key ret
sleep 6
if [ -n "$SHOTS" ]; then
  "$REPO/scripts/screenshot-when-kde-plasma-host/screenshot-when-kde-plasma-host.sh" \
    -o "$SHOTS/${RAIL:-rail}-launchpad-screenshot-app.png" >/dev/null 2>&1 || true
fi
# Asked before the dismissal below, not after: Escape tears the capture UI down
# and this is the only evidence that the Return landed on Screenshot rather than
# on whatever else the grid had filtered to.
# The process is `screencaptureui`, not `Screenshot`. Screenshot.app is a stub
# that launches /System/Library/CoreServices/screencaptureui.app, so neither
# `pgrep -x Screenshot` nor `pgrep -f Screenshot\.app` ever matches — this leg
# reported NO-LAUNCH on macos-11, macos-13 and macos-14 in one sweep while the
# capture toolbar is plainly visible in its own `-launchpad-screenshot-app.png`.
# One pattern covering both spellings, so a rail that does name it `Screenshot`
# still counts.
LAUNCHED=no
gssh 15 "pgrep -x screencaptureui >/dev/null 2>&1 || pgrep -x Screenshot >/dev/null 2>&1" \
  && LAUNCHED=yes
# Screenshot.app comes up as a floating capture bar over the desktop rather than
# a window, so it is dismissed with Escape and not with Command-Q.
qmp key esc
sleep 2
tail -c "+$((off + 1))" "$FAILLOG" >"$WORK/launchpad.log"
read -r gap cad_n hz_min hz_med drain_n alarms < <(
  python3 "$REPO/scripts/app-sweep-probe/read_window.py" "$WORK/launchpad.log")
verdict=ok
# A Launchpad that opened, filtered and launched nothing is not a device
# failure, but it is not a pass either — it means the step this leg exists to
# exercise did not happen, and the counters below are a blurred desktop's. The
# two are reported apart because they are two different failures: NO-LAUNCHPAD
# is the trigger, NO-LAUNCH is the Return.
# NO-LAUNCH is asserted first so NO-LAUNCHPAD can overwrite it: a Launchpad that
# never opened cannot have launched anything, and the root cause is the one to
# report.
[ "$LAUNCHED" = yes ] || { verdict=NO-LAUNCH; FAILED=1; }
[ "$LP_OPENED" = no ] && { verdict=NO-LAUNCHPAD; FAILED=1; }
awk -v g="$gap" -v f="$FREEZE_GAP_S" 'BEGIN{exit !(g > f)}' && { verdict=FREEZE; FAILED=1; }
awk -v m="$hz_med" 'BEGIN{exit !(m + 0 <= 0)}' && { verdict=NO-FRAMES; FAILED=1; }
gssh 15 true || { verdict=GUEST-GONE; FAILED=1; }
printf 'Launchpad->Screenshot\t%s\t%s\t%s\t%s\t%s\n' \
  "$verdict" "$hz_med" "$hz_min" "$gap" "$alarms" >>"$VERDICTS"
say "Launchpad -> Screenshot: $verdict  screenshot_app_launched=$LAUNCHED  \
present_hz med=$hz_med min=$hz_min  worst census gap=${gap}s  alarms=$alarms"

echo
echo "app                   verdict                     hz_med  hz_min  gap_s  alarms"
awk -F'\t' '{printf "%-21s %-27s %-7s %-7s %-6s %s\n", $1, $2, $3, $4, $5, $6}' "$VERDICTS"
[ -n "$KEEP" ] && say "per-app log slices kept in $WORK"
exit "$FAILED"
