#!/usr/bin/env bash
# wait-for-desktop.sh — block until this rail has a composited desktop, and
# collect the guest's crash reports if it finds a login window instead.
#
# sshd answers well before the desktop composites, so every harness here waits on
# `pgrep -x Dock` rather than on port 2222. That wait has one failure it cannot
# distinguish from a slow boot, and macos-12 sits in it forever: **the guest
# reached the login window and stopped**. Its serial log says so —
# `IOConsoleUsers: ... lin 0, llk 1` and `gIOScreenLockState 3`, repeating — while
# ssh answers, `guest-authorize.sh` installs its key, and Apple Events report
# consented. Every signal a harness reads says the guest is healthy, because it
# is; nobody is logged in. A whole rail was reported NO-DESKTOP for it.
#
# The console user is the discriminator and it costs one ssh round trip:
# `stat -f%Su /dev/console` names the account once a session owns the console.
# Before that it is a *system* account, and which one is not stable — the usual
# answer is `root`, but a macos-11 guest whose WindowServer had aborted answered
# `_windowserver`, measured live. So the test is not "is it root": it is "is it
# an account a person could log in as", and every system answer — `root`, any
# leading-underscore daemon account, or an empty string from an ssh that did not
# land — counts as nobody being logged in.
#
# **A login window is evidence, and logging in destroys it.** The two states
# above are not equally innocent. `root` owning the console is a guest that never
# logged in; `_windowserver` owning it is a guest whose WindowServer *aborted*,
# and the guest wrote a crash report saying why. Typing the password restarts the
# session over the top of it, and the next boot's snapshot revert throws the
# report away — so a whole class of failure has been arriving as "the desktop
# took a while" for as long as this script has existed.
#
# So the login window is now a collection point. Every time one is seen this
# pulls `/Library/Logs/DiagnosticReports/*.ips` and the user's own copy to
# `--reports DIR` first, and **refuses to log in** when any of them was written
# by *this boot*. `--login-after-crash` is the override for a session that wants
# the desktop anyway, and it is off by default: an unattended sweep must not
# trade a crash report for a screenshot.
#
# "This boot" is the whole qualifier and it was missing. These rails boot from
# snapshots, so a report captured into the image is in every boot of it forever,
# and macos-12's carries fourteen four-day-old `bluetoothd` ones. The gate fired
# on them and skipped the entire rail — one of the two that carry the standing
# GPU hang — with `WINDOWSERVER-CRASH` in the verdict table and no WindowServer
# report anywhere in the directory it named. See `crashes_since_boot`.
#
# Usage:
#   scripts/app-sweep-probe/wait-for-desktop.sh [--timeout N] [--password P]
#     [--reports DIR] [--login-after-crash]
#
# Exit 0 once `pgrep -x Dock` succeeds. Exit 1 on timeout, having said which of
# the two states it timed out in, which is the part the old inline loop could not
# report. Exit 2 when report age cannot be established, and exit 3 when a fresh
# crash report was collected and the login was refused.
set -uo pipefail
export LC_ALL=C

TIMEOUT=400
REPORTS="${REIMS_GUEST_REPORTS:-/tmp/reims-guest-reports}"
LOGIN_AFTER_CRASH=no
# The same throwaway default `vm/guest-authorize.sh` documents; these rails share
# one account and it is not a secret this repository is keeping.
PASSWORD="${REIMS_GUEST_PASSWORD:-aneesiqbal}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

while [ $# -gt 0 ]; do
  case "$1" in
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --password) PASSWORD="$2"; shift 2 ;;
    --reports) REPORTS="$2"; shift 2 ;;
    --login-after-crash) LOGIN_AFTER_CRASH=yes; shift ;;
    -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "wait-for-desktop: unknown argument $1" >&2; exit 2 ;;
  esac
done

say() { echo "wait-for-desktop: $*"; }
gssh() { timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=5 macos-vm "$1" 2>/dev/null; }

# Copy every diagnostic report the guest holds into `$REPORTS`, and answer with
# the names of the ones that are crashes.
#
# Both directories, because they hold different halves of the same event: a
# WindowServer abort is a *system* process and lands in `/Library/Logs`, while
# anything the logged-in user ran lands under `~`. Read without `sudo` on
# purpose — `AGENTS.md` records a guest `sudo` wedging with its timestamp lock
# held, which queues every later `sudo` forever, and a report this account
# cannot read is worth less than a boot.
#
# `tar` over the pipe rather than `scp` per file: one round trip, no second
# authentication, and it needs no path quoting for names that carry spaces.
collect_reports() {
  mkdir -p "$REPORTS"
  local collection
  collection=$(mktemp -d "$REPORTS/collection.XXXXXX") || return 2
  timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=5 macos-vm \
    'tar -cf - -C / Library/Logs/DiagnosticReports 2>/dev/null; \
     tar -cf - -C "$HOME" Library/Logs/DiagnosticReports 2>/dev/null' \
    2>/dev/null | tar -xf - -C "$collection" 2>/dev/null
  find "$collection" \( -name '*.ips' -o -name '*.crash' -o -name '*.panic' \) \
    ! -name '._*' 2>/dev/null
}

# Of the collected reports, the ones this boot produced.
#
# **A report older than the guest's boot came with the snapshot.** Every rail
# here boots from a snapshot and reverts on exit, so whatever crash reports were
# in the image when it was captured are in every boot of it, forever. macos-12's
# snapshot carries fourteen `bluetoothd` reports from 2026-08-08 — nothing to do
# with graphics, four days older than any boot that finds them — and the first
# version of this gate treated the whole directory as evidence. The result was
# the worst possible one: **the rail was skipped as WINDOWSERVER-CRASH on every
# boot**, and it is one of the two rails that carry the standing GPU hang. A gate
# that cannot fire is better than one that always does, because an always-firing
# one reads as a finding.
#
# The cutoff is the guest's own `kern.boottime`, taken over the same ssh rather
# than from this host's clock — the two do not agree and it is the guest that
# timestamps the reports. A reference file with that mtime is portable where
# `find -newermt` is not.
#
# The AppleDouble `._` sidecars are excluded above for a smaller version of the
# same mistake: they match `*.ips`, they are resource forks and not reports, and
# counting them doubled every tally this script has ever printed.
crashes_since_boot() {
  local boot ref
  for _ in 1 2 3; do
    boot=$(gssh 'sysctl -n kern.boottime' | sed -n 's/.*sec *= *\([0-9][0-9]*\).*/\1/p')
    [ -n "$boot" ] && break
    sleep 2
  done
  # An unavailable timestamp is not evidence that every report is fresh. The
  # caller must refuse to classify the login window rather than guess across
  # the missing contract term.
  [ -n "$boot" ] || return 2
  ref=$(mktemp) || return 2
  # `date -d @epoch` on this Linux host; the epoch itself came from the guest, so
  # no clock comparison crosses the boundary.
  touch -d "@$boot" "$ref" 2>/dev/null || { rm -f "$ref"; return 2; }
  local f
  for f in "$@"; do
    [ -n "$f" ] && [ "$f" -nt "$ref" ] && printf '%s\n' "$f"
  done
  rm -f "$ref"
}
qmp() {
  QMP_SOCK="${QMP_SOCK:-$REPO/vm/disks/run/qmp.sock}" \
    timeout 30 "$REPO/scripts/qmp/qmp.py" "$@" >/dev/null 2>&1
}

# At most twice. A password typed at a window that is not the login window goes
# somewhere else, and retrying a wrong guess forever is how a harness turns one
# failure into a locked account.
attempts=0
collected=no
state=starting
deadline=$((SECONDS + TIMEOUT))

while [ "$SECONDS" -lt "$deadline" ]; do
  if gssh 'pgrep -x Dock >/dev/null'; then
    say "desktop up (console user $(gssh 'stat -f%Su /dev/console' || echo '?'))"
    # Collected on the way out too. A WindowServer that aborted and was restarted
    # by autologin never shows this loop a login window at all, so the console
    # check cannot see that class — the report is the only thing that can, and it
    # is sitting on the guest either way.
    crashes=$(collect_reports)
    fresh=$(crashes_since_boot $crashes)
    age_status=$?
    if [ "$age_status" -ne 0 ]; then
      say "could not establish report age; leaving the collected files unclassified in $REPORTS"
    elif [ -n "$fresh" ]; then
      say "crash reports from THIS boot on a guest that reached the desktop — $REPORTS:"
      printf '  %s\n' $fresh
    elif [ -n "$crashes" ]; then
      say "$(printf '%s\n' $crashes | grep -c .) crash reports collected, all older \
than this boot (they came with the snapshot) — $REPORTS"
    fi
    exit 0
  fi

  console=$(gssh 'stat -f%Su /dev/console' || true)
  # An empty answer is an ssh that did not land, not a verdict: say nothing and
  # come round again rather than typing a password at a guest we cannot see.
  [ -z "$console" ] && { sleep 10; continue; }
  case "$console" in
    root|_*)
      state="login-window (console $console)"
      # Collected before anything is typed, and only once: the reports are the
      # reason this state is interesting, and a login overwrites the session
      # that produced them.
      if [ "$collected" = no ]; then
        collected=yes
        crashes=$(collect_reports)
        fresh=$(crashes_since_boot $crashes)
        age_status=$?
        if [ "$age_status" -ne 0 ]; then
          say "cannot establish whether the collected reports predate this boot"
          say "refusing to log in over evidence of unknown age — files are in $REPORTS"
          exit 2
        elif [ -n "$fresh" ]; then
          say "CRASH REPORTS from this boot collected into $REPORTS:"
          printf '  %s\n' $fresh
          if [ "$LOGIN_AFTER_CRASH" != yes ]; then
            say "refusing to log in — a login restarts the session over the evidence."
            say "pass --login-after-crash if the desktop is wanted anyway."
            exit 3
          fi
          say "--login-after-crash given; logging in over the crash anyway"
        elif [ -n "$crashes" ]; then
          # Collected and kept — they cost nothing and a later session may want
          # them — but they are the snapshot's, not this boot's, so they do not
          # decide anything. Said out loud because a silent "no crash" beside a
          # populated reports directory reads like a bug in the collector.
          say "$(printf '%s\n' $crashes | grep -c .) crash reports on the guest, \
all predating this boot — snapshot baggage, not evidence; logging in"
        else
          say "no crash reports on the guest; this is a guest that never logged in"
        fi
      fi
      if [ "$attempts" -lt 2 ]; then
        attempts=$((attempts + 1))
        say "console is owned by '$console' — nobody is logged in; typing the password (attempt $attempts)"
        # A single-account login window comes up with the password field focused,
        # so attempt one just types. If that did not take, the window was showing
        # the user *list* instead and the characters went nowhere: a Return picks
        # the highlighted account and gives the field focus, so attempt two leads
        # with one.
        [ "$attempts" -ge 2 ] && { qmp key ret; sleep 2; }
        qmp type "$PASSWORD"
        sleep 1
        qmp key ret
        sleep 20
      fi
      ;;
    # A real account owns the console, so someone is logged in and the desktop is
    # merely still coming up. Nothing to do but wait.
    *) state="logged-in-as-$console" ;;
  esac
  sleep 10
done

say "no Dock within ${TIMEOUT}s, last state: $state"
exit 1
