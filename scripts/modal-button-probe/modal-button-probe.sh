#!/usr/bin/env bash
# modal-button-probe.sh — does a guest modal's buttons actually reach the screen?
#
# Summons a loginwindow modal in the guest, asks the guest's accessibility API
# where it believes each button is, then measures the host-captured frame at
# exactly those rectangles. A button the guest declares and the frame does not
# show is a compositing loss in this device; a button the guest never declares
# is a guest-side absence and not ours.
#
# That separation is the whole point. The bug this exists for -- "the logout
# window sometimes shows no sleep, shutdown or restart buttons, more likely in
# dark mode" -- is a screenshot observation, and a screenshot alone cannot say
# which side dropped them. Two independent observations of the same frame can.
#
# Usage:
#   scripts/modal-button-probe/modal-button-probe.sh [-n TRIALS] [--appearance dark|light|alternate] [--keep DIR]
#
# Exits 0 when every declared button was drawn in every trial, 1 when any trial
# found a declared-but-undrawn button, and 2 on a setup failure (no guest, no
# host window, modal never appeared).
set -euo pipefail
# ImageMagick reports statistics with a '.', and awk/printf must agree. Under a
# comma-decimal locale every sigma read here is either rejected by printf or
# silently truncated to 0 by awk, which turns every button into a false MISSING.
export LC_ALL=C

TRIALS=10
APPEARANCE=alternate
KEEP=""
SSH_HOST=macos-vm
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHOT="$REPO_ROOT/scripts/screenshot/screenshot.sh"

usage() {
  sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
  exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    -n|--trials) TRIALS="$2"; shift 2 ;;
    --appearance) APPEARANCE="$2"; shift 2 ;;
    --keep) KEEP="$2"; shift 2 ;;
    --host) SSH_HOST="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "modal-button-probe: unknown argument $1" >&2; exit 2 ;;
  esac
done

case "$APPEARANCE" in dark|light|alternate) ;; *)
  echo "modal-button-probe: --appearance must be dark, light or alternate" >&2; exit 2 ;;
esac

WORK="${KEEP:-$(mktemp -d)}"
mkdir -p "$WORK"
[ -n "$KEEP" ] || trap 'rm -rf "$WORK"' EXIT

say() { echo "modal-button-probe: $*"; }

ssh -o ConnectTimeout=8 -o BatchMode=yes "$SSH_HOST" true 2>/dev/null || {
  say "no guest at $SSH_HOST" >&2; exit 2; }

# The guest's own idea of its display, so the point->pixel scale below is read
# rather than assumed. A guest whose display is not the size the host window
# captures would otherwise silently shift every rectangle. See
# scripts/lib/guest-display.sh for why the question is asked the way it is.
# shellcheck source=../lib/guest-display.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/guest-display.sh"
read -r GUEST_W GUEST_H < <(guest_display_size "$SSH_HOST") || {
  say "guest reported no display resolution" >&2; exit 2; }
[ -n "$GUEST_W" ] && [ -n "$GUEST_H" ] || { say "guest reported no display resolution" >&2; exit 2; }
say "guest display ${GUEST_W}x${GUEST_H}"

set_appearance() {
  guest_osa "$SSH_HOST" \
    "tell application \"System Events\" to tell appearance preferences to set dark mode to $1" \
    >/dev/null 2>&1 || true
}

# Summon the modal and report, one record per line:
#   WIN <x> <y> <w> <h>
#   BTN <x> <y> <w> <h> <name>
# Buttons are asked for by index so a name containing a space cannot split the
# record. An empty answer means the modal never appeared.
summon_and_describe() {
  ssh -o BatchMode=yes "$SSH_HOST" 'bash -s' <<'GUEST'
osascript -e 'tell application "System Events" to log out' >/dev/null 2>&1 &
for i in $(seq 1 25); do
  n=$(osascript -e 'tell application "System Events" to tell process "loginwindow" to count windows' 2>/dev/null || echo 0)
  [ "${n:-0}" -ge 1 ] && break
  sleep 0.4
done
[ "${n:-0}" -ge 1 ] || exit 0
osascript <<'AS'
tell application "System Events" to tell process "loginwindow"
  set w to window 1
  set p to position of w
  set s to size of w
  set out to "WIN " & (item 1 of p) & " " & (item 2 of p) & " " & (item 1 of s) & " " & (item 2 of s)
  repeat with i from 1 to (count buttons of w)
    set b to button i of w
    set bp to position of b
    set bs to size of b
    set out to out & linefeed & "BTN " & (item 1 of bp) & " " & (item 2 of bp) & " " & ¬
      (item 1 of bs) & " " & (item 2 of bs) & " " & (name of b)
  end repeat
  return out
end tell
AS
GUEST
}

dismiss() {
  ssh -o BatchMode=yes "$SSH_HOST" \
    "osascript -e 'tell application \"System Events\" to key code 53'" >/dev/null 2>&1 || true
}

# Standard deviation of one rectangle of a PNG, in ImageMagick's 0..1 scale.
patch_sigma() {
  magick "$1" -crop "${4}x${5}+${2}+${3}" +repage -colorspace Gray \
    -format '%[fx:standard_deviation]' info: 2>/dev/null || echo 0
}

fails=0
trials_run=0
for t in $(seq 1 "$TRIALS"); do
  case "$APPEARANCE" in
    dark) want=true ;;
    light) want=false ;;
    alternate) if [ $((t % 2)) -eq 1 ]; then want=true; else want=false; fi ;;
  esac
  set_appearance "$want"
  sleep 1

  desc="$WORK/trial-$t.txt"
  summon_and_describe >"$desc" 2>/dev/null || true
  if ! grep -q '^WIN ' "$desc"; then
    say "trial $t (dark=$want): modal never appeared — skipped"
    dismiss
    continue
  fi

  png="$WORK/trial-$t.png"
  "$SHOT" -o "$png" >/dev/null 2>&1 || { say "trial $t: capture failed" >&2; dismiss; continue; }
  dismiss

  read -r _ WX WY WW WH < <(grep -m1 '^WIN ' "$desc")
  IMG_W=$(identify -format '%w' "$png")
  IMG_H=$(identify -format '%h' "$png")
  # The capture is downscaled to fit 720p, so every rectangle scales with it.
  # Read the factor off the image rather than hardcoding the cap.
  SX=$(awk -v a="$IMG_W" -v b="$GUEST_W" 'BEGIN{print a/b}')
  SY=$(awk -v a="$IMG_H" -v b="$GUEST_H" 'BEGIN{print a/b}')

  # Background reference: tile the modal with patches and take the flattest.
  # The dialog is mostly its own fill, so the flattest patch is background —
  # whatever the appearance, whatever the layout, and without naming a corner
  # that some dialog puts an icon in. Judging every button against this rather
  # than against a fixed sigma is what lets one rule cover dark and light.
  PW=$(awk -v w="$WW" -v s="$SX" 'BEGIN{v=(w*s)/5; printf "%d", v<4?4:v}')
  PH=$(awk -v h="$WH" -v s="$SY" 'BEGIN{v=(h*s)/5; printf "%d", v<4?4:v}')
  ctrl=""
  for gx in 0 1 2 3 4; do
    for gy in 0 1 2 3 4; do
      cx=$(awk -v x="$WX" -v s="$SX" -v g="$gx" -v p="$PW" 'BEGIN{printf "%d", x*s+g*p}')
      cy=$(awk -v y="$WY" -v s="$SY" -v g="$gy" -v p="$PH" 'BEGIN{printf "%d", y*s+g*p}')
      s=$(patch_sigma "$png" "$cx" "$cy" "$PW" "$PH")
      ctrl=$(awk -v a="$ctrl" -v b="$s" 'BEGIN{print (a=="" || b+0 < a+0) ? b : a}')
    done
  done

  trials_run=$((trials_run + 1))
  missing=""
  while read -r _ BX BY BW BH NAME; do
    px=$(awk -v v="$BX" -v s="$SX" 'BEGIN{printf "%d", v*s}')
    py=$(awk -v v="$BY" -v s="$SY" 'BEGIN{printf "%d", v*s}')
    pw=$(awk -v v="$BW" -v s="$SX" 'BEGIN{printf "%d", (v*s)<1?1:(v*s)}')
    ph=$(awk -v v="$BH" -v s="$SY" 'BEGIN{printf "%d", (v*s)<1?1:(v*s)}')
    sig=$(patch_sigma "$png" "$px" "$py" "$pw" "$ph")
    # A drawn button has an edge and a label, so it varies more than the plain
    # background beside it. Equal-or-less variation means nothing was drawn
    # there -- the comparison is against this same frame, so it carries no
    # assumption about colour, theme or contrast.
    #
    # A bare '>' rather than a tuned factor because the separation is not close.
    # Measured on the guest's logout modal: buttons read sigma 0.055-0.104
    # against a flattest-patch background of 0.0008-0.0012, in both dark and
    # light -- 45x to 130x. Any factor between 2 and 40 would decide the same
    # cases, so picking one would be inventing precision the data does not need.
    verdict=$(awk -v s="$sig" -v c="$ctrl" 'BEGIN{print (s > c) ? "drawn" : "MISSING"}')
    printf '  trial %-3s dark=%-5s %-28s sigma=%.4f ctrl=%.4f %s\n' \
      "$t" "$want" "$NAME" "$sig" "$ctrl" "$verdict"
    [ "$verdict" = "MISSING" ] && missing="$missing $NAME"
  done < <(grep '^BTN ' "$desc")

  if [ -n "$missing" ]; then
    fails=$((fails + 1))
    say "trial $t (dark=$want): guest declared but frame does not show:$missing"
    [ -n "$KEEP" ] && say "  frame kept at $png"
  fi
  sleep 2
done

say "$trials_run trials, $fails with a declared-but-undrawn button"
[ "$fails" -eq 0 ] || exit 1
[ "$trials_run" -gt 0 ] || { say "no trial produced a modal" >&2; exit 2; }
exit 0
