# guest-display.sh — the one place a probe asks the guest how big its desktop is.
#
# Source it and call `guest_display_size <ssh-host>`; it prints `WIDTH HEIGHT`
# on stdout and returns non-zero if the guest could not answer.
#
# # Why not system_profiler
#
# `system_profiler SPDisplaysDataType | grep Resolution:` is the obvious
# spelling and it is not portable across the guest OS lines this device
# supports. A macOS 12 guest on this device prints the GPU and **no display at
# all** — no `Displays:` section, no `Resolution:` line — so every probe that
# read it exited 2 with "guest reported no display resolution", which reads like
# a wedged guest rather than like one report being unavailable. It has also been
# observed hanging indefinitely on a macOS 11 guest, which an unattended harness
# cannot tell from a wedged boot.
#
# Nor from Finder: `tell application "Finder" to get bounds of window of desktop`
# answers `AppleEvent timed out (-1712)` on these guests.
#
# `screencapture` writes the framebuffer and `sips` reads its header. Both ship
# with every macOS, neither needs root, and the answer is the strongest form of
# the question a two-observation probe is asking — not what the guest's
# *configuration* claims the display is, but how many pixels its compositor just
# produced. A probe that then draws at that size is drawing at the size the
# frame it will measure actually has.
#
# The capture is bounded host-side by `timeout`: an unattended harness cannot
# tell a wedged guest command from a wedged boot, and this one runs against
# guests that are sometimes wedged by construction.

# Run one AppleScript in the guest, bounded.
#
# `osascript -e 'tell application "System Events" to ...'` over ssh **hangs
# forever** on a guest that has not granted the ssh session Automation access:
# the request raises a consent prompt on the guest's own desktop and nobody is
# there to answer it. Measured on the macos-12 rail, where every System Events
# call in every probe hung until the probe's own harness gave up, and the probe
# then reported the empty answer as "the guest says the desktop picture is ''".
#
# So the bound is not defensive tidiness, it is the difference between a probe
# that reports "this rail cannot be scripted" in 15 seconds and one that stalls
# an unattended sweep. The host-side `timeout` does not kill the remote
# osascript; it only stops us waiting on it.
guest_osa() {
  local guest="$1" script="$2" secs="${GUEST_OSA_TIMEOUT:-15}"
  timeout "$secs" ssh -o BatchMode=yes "$guest" "osascript -e '$script'" 2>/dev/null
}

GUEST_DISPLAY_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Consent this ssh session to send Apple Events to one named guest application,
# answering the panel if it comes up. Returns 0 when the app answers.
#
# TCC scopes Apple Events per (client, target) pair, so consenting "System
# Events" says nothing about "Safari": every application a probe scripts needs
# its own answer. The panel is modal and its default button is Allow, so the
# answer is one Return through QMP's usb-kbd.
#
# The keystroke is sent only after a call has been *seen* to hang, because a
# hang is the only evidence available that a modal is up — nothing on the host
# can see the guest's screen from here. A Return pressed blind at a live desktop
# is not free, and a consented call answers in well under a second, so the
# timeout is generous enough that a merely busy guest is not mistaken for a
# blocked one and sent a keystroke it did not ask for.
#
# PREFER THIS OVER `System Events` for anything a scriptable application can do
# itself. Reading or setting `bounds of front window` through an application's
# own Standard Suite needs only this consent. The same thing through System
# Events' `process`/`window` objects needs **assistive access**, which has no
# panel to answer — it is granted by hand in System Preferences, and exactly one
# rail of six ever had it. A probe written against System Events therefore runs
# on that one rail and reports every other one as having no windows.
#
# The cycle runs `GUEST_CONSENT_ATTEMPTS` times rather than once, because one
# press is not reliably the press that lands. A cold application can take longer
# to accept its first event than the probe's timeout, so the first Return goes
# out before any panel exists; and the desktop raises its own transient furniture
# — a software-update notification arrived mid-run once — which a keystroke aimed
# at a modal can meet on the way. Each further round costs nothing on a guest
# that has already consented, because the call returns at once and no key is
# pressed at all.
guest_osa_consenting() {
  local guest="$1" what="$2" script="$3"
  local qmp_sock="${QMP_SOCK:-$GUEST_DISPLAY_REPO_ROOT/vm/disks/run/qmp.sock}"
  local secs="${GUEST_CONSENT_TIMEOUT:-15}"
  local attempts="${GUEST_CONSENT_ATTEMPTS:-3}"
  local i out

  for ((i = 1; i <= attempts; i++)); do
    if out=$(GUEST_OSA_TIMEOUT="$secs" guest_osa "$guest" "$script"); then
      printf '%s\n' "$out"
      return 0
    fi
    if [ ! -S "$qmp_sock" ]; then
      echo "guest-display: '$what' did not answer and there is no QMP socket at \
$qmp_sock to answer a consent panel with" >&2
      return 1
    fi
    echo "guest-display: '$what' did not answer — pressing the default button of \
whatever modal is up (try $i/$attempts) ..." >&2
    QMP_SOCK="$qmp_sock" timeout 30 "$GUEST_DISPLAY_REPO_ROOT/scripts/qmp/qmp.py" \
      key ret >/dev/null 2>&1 || true
  done
  echo "guest-display: '$what' still will not answer after $attempts tries — the \
panel may have been for something else, or there is no session to raise one in" >&2
  return 1
}

# The smallest call that raises the panel: it names the app and asks it
# something, which is the whole trigger. Its answer is discarded.
guest_apple_events_consent() {
  local guest="$1" app="$2"
  guest_osa_consenting "$guest" "$app" "tell application \"$app\" to get name" >/dev/null
}

# Print "WIDTH HEIGHT" for $1's desktop, or return 1 having said why on stderr.
guest_display_size() {
  local guest="$1" out w h
  out=$(timeout 90 ssh -o BatchMode=yes "$guest" 'bash -s' 2>/dev/null <<'GUEST_EOF'
set -e
out=/tmp/reims-guest-display-size.png
rm -f "$out"
/usr/sbin/screencapture -x -t png "$out" >/dev/null 2>&1
/usr/bin/sips -g pixelWidth -g pixelHeight "$out" 2>/dev/null
rm -f "$out"
GUEST_EOF
  ) || true
  w=$(printf '%s\n' "$out" | sed -n 's/.*pixelWidth: *\([0-9][0-9]*\).*/\1/p' | head -1)
  h=$(printf '%s\n' "$out" | sed -n 's/.*pixelHeight: *\([0-9][0-9]*\).*/\1/p' | head -1)
  if [ -z "$w" ] || [ -z "$h" ]; then
    echo "guest-display: $guest did not report a desktop size (got '$out')" >&2
    return 1
  fi
  printf '%s %s\n' "$w" "$h"
}
