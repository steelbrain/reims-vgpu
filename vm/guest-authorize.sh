#!/usr/bin/env bash
#
# vm/guest-authorize.sh — make the running x86 guest reachable as ssh host
# `macos-vm`, whichever rail it came from.
#
# WHY THIS EXISTS. Every probe under `scripts/` reaches the guest as
# `ssh -o BatchMode=yes macos-vm`, which is key auth and nothing else. Exactly
# one rail was provisioned with that key; the rest authenticate by password. A
# probe run against them fails at the first hop, and `BatchMode=yes` makes that
# failure look like "guest not up yet" rather than "guest has no key".
#
# Installing the key into each rail's snapshot would work and is the wrong
# shape: a `--testing` boot is a byte-identical COW clone of an immutable
# snapshot, so authorizing the *clone* costs one second, survives exactly as
# long as the boot it belongs to, and mutates nothing that outlives it. Run
# this once after a boot comes up and every existing probe works unchanged.
#
# THE SECOND AUTHORIZATION IS APPLE EVENTS, and it is the one that hangs. Every
# probe that drives the GUI does it through `osascript ... "System Events"`, and
# the first such call from an ssh session raises a TCC consent panel on the
# guest's own screen: "sshd-keygen-wrapper wants access to control System
# Events". Until somebody answers it the `osascript` never returns. Nothing on
# the host can see the panel, so the failure arrives as a probe that hangs until
# its host-side `timeout` fires and then reports whatever a missing answer looks
# like — `window-drag-probe` reports "could not read the window frame", which
# reads as a broken window rather than an unanswered dialog.
#
# The panel's default button is its Allow button, so the answer is one Return
# through QMP's usb-kbd. Sending it blind would be the wrong shape — a Return
# landing on a Finder desktop is not free — so it is sent only after a probe
# `osascript` has been seen to hang, which is what says a modal is up. That also
# makes the step self-verifying: the same call is retried afterwards and has to
# answer. This grant lives in the boot's COW clone and dies with it, exactly
# like the ssh key above.
#
# Idempotent. On a rail that already has the key it does not need the password
# at all, and on one that has already consented no key is pressed, so it is safe
# to run unconditionally in a harness.
#
#   vm/guest-authorize.sh                 # wait for sshd, authorize, verify
#   vm/guest-authorize.sh --timeout 300   # bound the wait differently
#   vm/guest-authorize.sh --no-automation # skip the Apple Events consent step
#
# Environment:
#   SSH_PORT              host port forwarded to the guest's 22 (default 2222)
#   REIMS_GUEST_USER      guest account (default aneesiqbal)
#   REIMS_GUEST_PASSWORD  its password (default aneesiqbal) — a throwaway
#                         credential for a local development VM, not a secret
#   REIMS_GUEST_KEY       private key whose .pub gets installed
#                         (default ~/.ssh/macos_x86_guest, the key `macos-vm`
#                         names in ~/.ssh/config)
#   QMP_SOCK              the running boot's QMP socket, used only to press
#                         Return at the consent panel (default the x86 rail's
#                         `vm/disks/run/qmp.sock`)
#
# Every guest-side step is bounded by `timeout` on the host side. A wedged
# guest — a stuck `sshd`, a login shell waiting on something — must cost this
# script its deadline and no more, because the harness that calls it is
# unattended and a hang here reads as a hung boot.
set -euo pipefail

SSH_PORT="${SSH_PORT:-2222}"
GUEST_USER="${REIMS_GUEST_USER:-aneesiqbal}"
GUEST_PASSWORD="${REIMS_GUEST_PASSWORD:-aneesiqbal}"
GUEST_KEY="${REIMS_GUEST_KEY:-$HOME/.ssh/macos_x86_guest}"
WAIT_SECONDS=420
WANT_AUTOMATION=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QMP_SOCK="${QMP_SOCK:-$REPO_ROOT/vm/disks/run/qmp.sock}"
QMP_PY="$REPO_ROOT/scripts/qmp/qmp.py"

# One ssh attempt must never outlast this, password or key.
STEP_TIMEOUT=30

# How long a probe `osascript` may take before the only remaining explanation is
# an unanswered modal. A granted call answers in well under a second even on the
# slowest rail; this is generous so that a merely busy guest is not mistaken for
# a blocked one and sent a keystroke it did not ask for.
CONSENT_PROBE_TIMEOUT=15

die() { echo "guest-authorize: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --timeout) shift; WAIT_SECONDS="${1:-}"; [ -n "$WAIT_SECONDS" ] || die "--timeout needs seconds"; shift ;;
    --timeout=*) WAIT_SECONDS="${1#--timeout=}"; shift ;;
    --no-automation) WANT_AUTOMATION=0; shift ;;
    -h|--help) sed -n '2,60p' "$0"; exit 0 ;;
    *) die "unknown arg: $1" ;;
  esac
done

[ -f "$GUEST_KEY" ] || die "no private key at $GUEST_KEY (set REIMS_GUEST_KEY)"
[ -f "$GUEST_KEY.pub" ] || die "no public key at $GUEST_KEY.pub"
PUBKEY="$(cat "$GUEST_KEY.pub")"

# The guest's host key changes with every rail and every reprovision, and this
# is a loopback port on a development box — pinning it would only produce a
# mismatch to click through. Keep that decision out of the user's known_hosts.
SSH_COMMON=(
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o LogLevel=ERROR
  -o ConnectTimeout=8
  -p "$SSH_PORT"
)

ssh_key() {
  timeout "$STEP_TIMEOUT" ssh "${SSH_COMMON[@]}" \
    -o BatchMode=yes -o IdentitiesOnly=yes -i "$GUEST_KEY" \
    "$GUEST_USER@127.0.0.1" "$@"
}

ssh_password() {
  command -v sshpass >/dev/null 2>&1 || die "sshpass not found (needed to authorize a rail that has no key yet)"
  timeout "$STEP_TIMEOUT" sshpass -p "$GUEST_PASSWORD" ssh "${SSH_COMMON[@]}" \
    -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 \
    "$GUEST_USER@127.0.0.1" "$@"
}

# --- Wait for sshd ---------------------------------------------------------
# Either credential answering means sshd is up; which one it was is the next
# question, not this one. A refused connection is "not yet", anything else is
# still "not yet" until the deadline, because a guest mid-login refuses in
# several different ways.
deadline=$(( $(date +%s) + WAIT_SECONDS ))
up=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  if ssh_key true 2>/dev/null || ssh_password true 2>/dev/null; then up=1; break; fi
  sleep 5
done
[ "$up" -eq 1 ] || die "guest sshd did not answer on port $SSH_PORT within ${WAIT_SECONDS}s"

# --- Authorize -------------------------------------------------------------
if ssh_key true 2>/dev/null; then
  echo "guest-authorize: key auth already works (port $SSH_PORT, user $GUEST_USER)"
else
  echo "guest-authorize: installing $GUEST_KEY.pub for $GUEST_USER ..."
  # Appending only when absent keeps a re-run from growing the file, and the
  # 0700/0600 modes are what macOS sshd requires before it will read it at all.
  ssh_password "set -e
    mkdir -p ~/.ssh && chmod 700 ~/.ssh
    touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
    grep -qxF '$PUBKEY' ~/.ssh/authorized_keys || printf '%s\n' '$PUBKEY' >> ~/.ssh/authorized_keys" \
    || die "could not install the key over password auth"

  ssh_key true 2>/dev/null \
    || die "key installed but key auth still fails — check the guest's sshd config"
  echo "guest-authorize: key auth now works"
fi

# --- Report what a probe will see ------------------------------------------
# `macos-vm` is what the probes actually type, and it authenticates against the
# user's own ~/.ssh/known_hosts. That file cannot hold a useful entry for this
# endpoint: `127.0.0.1:2222` is a different machine on every rail, so whichever
# rail booted first wins and every other rail then fails the host-key check —
# with `BatchMode=yes` turning the mismatch into a silent probe failure. Forget
# the pin before verifying; `accept-new` re-learns it for the rail that is
# actually running.
for host in "[127.0.0.1]:$SSH_PORT" "[localhost]:$SSH_PORT"; do
  timeout "$STEP_TIMEOUT" ssh-keygen -R "$host" >/dev/null 2>&1 || true
done

# If ~/.ssh/config points `macos-vm` somewhere else, say so here rather than
# letting a probe fail obscurely.
if timeout "$STEP_TIMEOUT" ssh -o BatchMode=yes -o ConnectTimeout=8 \
     -o StrictHostKeyChecking=accept-new macos-vm true 2>/dev/null; then
  echo "guest-authorize: ssh macos-vm ok — probes under scripts/ will reach this guest"
else
  echo "guest-authorize: WARNING ssh macos-vm failed even though direct key auth works." >&2
  echo "guest-authorize: probes default to GUEST=macos-vm; check ~/.ssh/config names port $SSH_PORT and $GUEST_KEY." >&2
  exit 1
fi

# --- Consent to Apple Events ------------------------------------------------
# The smallest call that raises the same panel every GUI probe raises: it names
# System Events and asks it something, which is the whole trigger. Its answer is
# discarded — this is a consent probe, not a query.
consent_probe() {
  timeout "$CONSENT_PROBE_TIMEOUT" ssh -o BatchMode=yes -o ConnectTimeout=8 \
    -o StrictHostKeyChecking=accept-new macos-vm \
    'osascript -e '\''tell application "System Events" to get name'\''' >/dev/null 2>&1
}

if [ "$WANT_AUTOMATION" -eq 1 ]; then
  if consent_probe; then
    echo "guest-authorize: Apple Events already consented — GUI probes will drive this guest"
  elif [ ! -S "$QMP_SOCK" ]; then
    # Not fatal: a rail whose GUI cannot be driven still boots and still reports
    # every device-side counter. Say which capability is missing and why.
    echo "guest-authorize: WARNING Apple Events consent is pending and there is no QMP socket at $QMP_SOCK." >&2
    echo "guest-authorize: GUI probes (window-drag, wallpaper, modal-button) will hang on the consent panel." >&2
  else
    echo "guest-authorize: Apple Events consent panel is up — answering it over QMP ..."
    # The panel is modal and its default button is Allow, so Return answers it.
    # It only exists because the probe above raised it, and that probe has
    # already returned (its timeout fired), so the panel is the frontmost thing
    # on the guest's screen and nothing else can take the keystroke.
    QMP_SOCK="$QMP_SOCK" timeout "$STEP_TIMEOUT" "$QMP_PY" key ret >/dev/null 2>&1 \
      || echo "guest-authorize: WARNING could not send Return over QMP at $QMP_SOCK" >&2
    if consent_probe; then
      echo "guest-authorize: Apple Events consented — GUI probes will drive this guest"
    else
      echo "guest-authorize: WARNING Apple Events consent did not take." >&2
      echo "guest-authorize: a panel may still be up, or WindowServer is not running to show one." >&2
      echo "guest-authorize: GUI probes will hang; device-side counters are unaffected." >&2
    fi
  fi
fi
