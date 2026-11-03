#!/bin/bash
# The lane gate's probe batch, its guest side, on the lane host: msi-boot.sh runs it in a probe
# boot (boot.env PROBES=DIR) once the guest's desktop is up and our Metal device is in it, and
# probe-batch.sh (on the Mac) runs it with FAKE_GUEST=DIR for a dry run.
#   msi-probes.sh
# Its environment:
#   PROBES          the round's probes, one directory each (run.sh, budget, tip, and any files
#                   run.sh reads), run in the order PROBES/.order lists them (one name a line,
#                   the round's take order), else in name order
#   PROBE_RESULTS   where each probe's results go: PROBE_RESULTS/NAME/, replaced
#   PROBE_TIP       the published tip the boot runs (its image and host): a probe whose tip names
#                   another commit is skipped
#   DEADLINE        the epoch second the hold ends: a probe starts only where its budget and 30 s
#                   fit before it, else it stays queued for the next round
#   B, R            the boot's directory (QEMU's launch.log) and the run's (gpu-clocks.txt)
#   QEMU_PID        the boot's QEMU, whose Xid lines xid.txt marks as its own
#   PORT, GUEST_USER, GUEST_PASS   the guest's SSH; or FAKE_GUEST, a directory that stands for the
#                   guest's home: its commands run there, on this machine
#   SETTLE_LOAD, SETTLE_S   the host's 1-minute load average the probes wait for (default 4), and
#                   the longest wait (default 180 s)
# Each probe's results (PROBE_RESULTS/NAME/):
#   status          "ran", "skipped: WHY" or "not reached: WHY" (probe-batch.sh queues it again)
#   stdout, stderr  run.sh's, over SSH, in the guest's copy of the probe's directory (~/probes/NAME)
#   rc              its exit status: 137 where its budget ran out (the guest kills its process group)
#   wall_s          how long it ran; start, end: the host's epoch times around it
#   load            the host's load average before it
#   host.log        QEMU's log lines from start to end (the host's display updates, warnings,
#                   fatal lines and the rest)
#   gpu-clocks.txt  the GPU's clocks over the same span, where the host samples them
#   xid.txt         the NVIDIA driver's GPU errors (the kernel's NVRM Xid lines) over the span, as
#                   msi-xid.sh prints them, "own" those of the boot's QEMU (QEMU_PID); "unread"
#                   where the kernel log could not be read
#   out/            what run.sh left in its directory's out/, copied back
#   probe/          the probe as it was dropped
#   shots.txt       where it asked for screenshots (below): "EPOCH LABEL" each, or why none came
#   hang/           where it failed and the guest's SSH no longer answers: msi-hang.py's snapshot
#                   of the guest, taken before QEMU is stopped (QEMU's threads, every vCPU's
#                   registers and stack, the scanout; the scanout alone where the hang catcher
#                   below took the rest); unless the guest answers again within 3 minutes (it
#                   restarted), the probes after it are not reached (they go back to the queue)
# The hang catcher (HANG_CATCH, default msi-hang-catch.py beside this script) watches the boot
# while the probes run and writes R/hang/ (watch.txt, guest-info.txt, catch-N/ at a hang,
# reboot-N/ with the panic reports after a restart: its docstring says what); the round waits for
# a catch it started before QEMU stops.
# run.sh gets PROBE_NAME and PROBE_PASS (the guest user's password, for sudo) in its environment.
# Screenshots: a probe whose directory holds a file named shots may ask for the guest's own
# screenshot, which screencapture over SSH cannot take (no Screen Recording grant): run.sh runs
# `bash ~/probes/.shot.sh LABEL`, which waits up to 20 s for out/shot-LABEL.png (exit 1 where none
# came). The host polls the probe's out/ for the request once a second while it runs, sends
# Command-Shift-3 over QMP ($B/qmp.sock), and moves the guest's new Desktop screenshot there.
set -uo pipefail
: "${PROBES:?}" "${PROBE_RESULTS:?}" "${PROBE_TIP:?}" "${DEADLINE:?}"
log() { echo "$(date +%T) probes: $*"; }
# ServerAlive: a session to a guest whose network stalled ends after 60 s of silence (a stall can
# outlast the guest's TCP state, and the session then waits forever: the shots poll did, for 15
# minutes, after a 1-minute stall).
SOPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PreferredAuthentications=password
  -o PubkeyAuthentication=no -o ConnectTimeout=5 -o ServerAliveInterval=5 -o ServerAliveCountMax=12
  -o LogLevel=ERROR)
if [[ -n ${FAKE_GUEST:-} ]]; then
  G() { (cd "$FAKE_GUEST" && HOME=$FAKE_GUEST bash -c "$*"); }
  Gput() { cp -R "$1" "$FAKE_GUEST/$2"; }
  Gget() { cp -R "$FAKE_GUEST/$1" "$2"; }
else
  G() { sshpass -p "$GUEST_PASS" ssh -p "$PORT" "${SOPTS[@]}" "$GUEST_USER@127.0.0.1" "$@" < /dev/null; }
  Gput() { sshpass -p "$GUEST_PASS" scp -q -r -P "$PORT" "${SOPTS[@]}" "$1" "$GUEST_USER@127.0.0.1:$2"; }
  Gget() { sshpass -p "$GUEST_PASS" scp -q -r -P "$PORT" "${SOPTS[@]}" "$GUEST_USER@127.0.0.1:$1" "$2"; }
fi
# Whether the guest's SSH answers (twice ConnectTimeout, which bounds the banner's exchange too).
alive() {
  timeout 20 sshpass -p "$GUEST_PASS" ssh -p "$PORT" "${SOPTS[@]}" "$GUEST_USER@127.0.0.1" true < /dev/null 2> /dev/null
}
iso() { date -d "@$1" +%Y-%m-%dT%H:%M:%S.%3N 2> /dev/null || date -r "${1%.*}" +%Y-%m-%dT%H:%M:%S.000; }
now() { date +%s.%N 2> /dev/null | cut -c1-14; }
# The tips match where one names the other (a short sha and the full one).
same_tip() { [[ -n $1 && -n $2 && ( $1 == "$2"* || $2 == "$1"* ) ]]; }

# Waits for the host's load to settle before the first probe.
settle() {
  local t0 l
  t0=$(date +%s)
  while :; do
    l=$(cut -d' ' -f1 /proc/loadavg 2> /dev/null || echo 0)
    awk -v l="$l" -v m="${SETTLE_LOAD:-4}" 'BEGIN { exit !(l < m) }' && break
    (( $(date +%s) - t0 >= ${SETTLE_S:-180} )) && { log "load $l after ${SETTLE_S:-180} s: running anyway"; return; }
    sleep 5
  done
  log "load $l after $(( $(date +%s) - t0 )) s"
}

G 'rm -rf ~/probes && mkdir -p ~/probes'
# run.sh in a process group of its own, killed at its budget: "perl .run.pl BUDGET CMD ARG...".
runpl=$(mktemp)
cat > "$runpl" << 'EOF'
my $b = shift;
my $c = fork;
if (!$c) { setpgrp(0, 0); exec @ARGV; exit 127 }
$SIG{ALRM} = sub { kill "KILL", -$c };
alarm $b;
waitpid($c, 0);
exit($? & 127 ? 128 + ($? & 127) : $? >> 8);
EOF
Gput "$runpl" probes/.run.pl
rm -f "$runpl"
# The guest's side of a screenshot: a request, then the wait for its file.
shotsh=$(mktemp)
cat > "$shotsh" << 'EOF'
l=${1:-1}
mkdir -p out
rm -f "out/shot-$l.png"
: > "out/shot-request.$l"
for _ in $(seq 1 40); do [ -s "out/shot-$l.png" ] && exit 0; sleep 0.5; done
exit 1
EOF
Gput "$shotsh" probes/.shot.sh
rm -f "$shotsh"
[[ -n ${FAKE_GUEST:-} ]] || G 'defaults write com.apple.screencapture show-thumbnail -bool false' 2> /dev/null
# The screenshots a probe asked for (its out/shot-request.LABEL files): Command-Shift-3 over QMP,
# then the guest's newest Desktop screenshot moved to out/shot-LABEL.png.
shots() {
  local name=$1 l f
  # (no glob in the guest's command: its shell, zsh, fails one that matches nothing)
  for l in $(G "ls ~/probes/$name/out 2> /dev/null" | sed -n 's/^shot-request\.//p'); do
    G "rm -f ~/probes/$name/out/shot-request.$l; touch ~/probes/.before-shot"
    if [[ -n ${FAKE_GUEST:-} ]]; then
      echo fake > "$FAKE_GUEST/probes/$name/out/shot-$l.png"
      echo "$(now) $l fake" >> "$out/shots.txt"
      continue
    fi
    printf '%s\n' '{"execute":"qmp_capabilities"}' \
      '{"execute":"send-key","arguments":{"keys":[{"type":"qcode","data":"meta_l"},{"type":"qcode","data":"shift"},{"type":"qcode","data":"3"}],"hold-time":200}}' |
      timeout 10 socat - "UNIX-CONNECT:$B/qmp.sock" > /dev/null 2>&1
    f=
    for _ in $(seq 1 16); do
      sleep 1
      f=$(G 'find ~/Desktop -maxdepth 1 -name "*.png" -newer ~/probes/.before-shot | head -1' 2> /dev/null)
      [[ -n $f ]] && break
    done
    if [[ -n $f ]] && sleep 0.5 && G "mv \"$f\" ~/probes/$name/out/shot-$l.png"; then
      echo "$(now) $l" >> "$out/shots.txt"
    else
      echo "$(now) $l none: the guest saved no screenshot" >> "$out/shots.txt"
    fi
  done
}
settled=0 hung= HC=
HANG_CATCH=${HANG_CATCH:-$(dirname "$0")/msi-hang-catch.py}
if [[ -z ${FAKE_GUEST:-} && -f $HANG_CATCH && -n ${QEMU_PID:-} && -n ${R:-} ]]; then
  mkdir -p "$R/hang"
  python3 -I "$HANG_CATCH" --boot-dir "$B" --pid "$QEMU_PID" --port "$PORT" --out "$R/hang" \
    --user "$GUEST_USER" --password "$GUEST_PASS" > "$R/hang.log" 2>&1 &
  HC=$!
  log "the hang catcher: $HANG_CATCH ($(sha256sum "$HANG_CATCH" | cut -c1-12)), pid $HC, into $R/hang"
fi
# Waits while the catcher's catch is under way (a HANG line with no "catch N written" after it),
# 150 s at most; with an argument, first up to that many seconds for its HANG line.
catch_wait() {
  [[ -n $HC ]] || return 0
  local w=$R/hang/watch.txt i
  for (( i = 0; i < ${1:-0}; i++ )); do grep -q HANG "$w" 2> /dev/null && break; sleep 1; done
  for (( i = 0; i < 150; i++ )); do
    awk '/HANG/ { h = 1 } /catch [0-9]+ written/ { h = 0 } END { exit !h }' "$w" 2> /dev/null || return 0
    sleep 1
  done
}
in_order() {  # the probes' directories, in .order's order where it is there
  if [[ -s $PROBES/.order ]]; then
    while read -r n; do [[ -n $n && -d $PROBES/$n ]] && echo "$PROBES/$n/"; done < "$PROBES/.order"
  else
    for p in "$PROBES"/*/; do echo "$p"; done
  fi
}
for p in $(in_order); do
  [[ -d $p ]] || continue  # a round with no probe (a capture alone)
  p=${p%/}
  name=${p##*/}
  out=$PROBE_RESULTS/$name
  rm -rf "$out"
  mkdir -p "$out"
  budget=$(tr -dc 0-9 < "$p/budget" 2> /dev/null)
  tip=$(tr -dc 0-9a-f < "$p/tip" 2> /dev/null)
  if [[ ! -f $p/run.sh ]]; then
    echo "skipped: no run.sh" > "$out/status"
  elif [[ -z $budget ]] || (( budget < 1 || budget > 300 )); then
    echo "skipped: budget '${budget:-none}' is not 1 to 300 s" > "$out/status"
  elif ! same_tip "$tip" "$PROBE_TIP"; then
    echo "skipped: valid for ${tip:-no tip}, and the published tip is $PROBE_TIP" > "$out/status"
  elif [[ -n $hung ]]; then
    echo "not reached: the guest stopped answering SSH in $hung" > "$out/status"
  elif (( $(date +%s) + budget + 30 > DEADLINE )); then
    echo "not reached: the hold had $(( DEADLINE - $(date +%s) )) s left for a ${budget} s budget" > "$out/status"
  else
    ((settled)) || { settle; settled=1; }
    log "$name (budget $budget s)"
    cut -d' ' -f1-3 /proc/loadavg > "$out/load" 2> /dev/null || true
    Gput "$p" "probes/$name"
    start=$(now)
    # The guest runs run.sh in a process group of its own and kills the group at its budget
    # (~/probes/.run.pl); the host stops waiting 30 s after that.
    G "cd ~/probes/$name && PROBE_NAME='$name' PROBE_PASS='${GUEST_PASS:-}' perl ~/probes/.run.pl $budget bash run.sh" \
      > "$out/stdout" 2> "$out/stderr" &
    gp=$!
    ( sleep $(( budget + 30 )) & s=$!
      trap 'kill $s 2> /dev/null; exit 0' TERM
      wait $s && kill "$gp" 2> /dev/null ) &
    wp=$!
    if [[ -f $p/shots ]]; then
      while kill -0 "$gp" 2> /dev/null; do
        shots "$name"
        sleep 1
      done
    fi
    wait "$gp"
    echo $? > "$out/rc"
    kill "$wp" 2> /dev/null; wait "$wp" 2> /dev/null
    end=$(now)
    echo "$start" > "$out/start"
    echo "$end" > "$out/end"
    awk -v a="$start" -v b="$end" 'BEGIN { printf "%.1f\n", b - a }' > "$out/wall_s"
    # A probe that failed where the guest's SSH no longer answers: the guest's state, now.
    if [[ $(cat "$out/rc") != 0 && -z ${FAKE_GUEST:-} ]] && ! alive && ! alive; then
      log "$name: the guest's SSH no longer answers: the hang snapshot"
      catch_wait 45
      python3 -I "$(dirname "$0")/msi-hang.py" "${QEMU_PID:-0}" "$B" "$out/hang" "$([[ -n $HC ]] && echo scanout)" > /dev/null 2>&1
      sed 's/^/  hang: /' "$out/hang/hang.log" 2> /dev/null
      # A guest that panicked restarts and answers again (about 2 minutes): the probes go on in
      # it. One that does not stays hung, and the probes after it are not reached.
      back=0
      for (( k = 0; k < 18; k++ )); do alive && { back=1; break; }; sleep 10; done
      if ((back)); then
        log "$name: the guest answers again (restarted): the probes go on"
      else
        hung=$name
        log "$name: the guest still does not answer: the probes after it are not reached"
      fi
    else
      G "test -d ~/probes/$name/out" 2> /dev/null && Gget "probes/$name/out" "$out/out" 2> /dev/null
    fi
    if [[ -n ${B:-} && -f $B/launch.log ]]; then
      awk -v a="$(iso "$start")" -v b="$(iso "$end")" '
        /^20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]T/ { t = substr($1, 1, 23); keep = t >= a && t <= b }
        keep' "$B/launch.log" > "$out/host.log"
    fi
    [[ -n ${R:-} && -s $R/gpu-clocks.txt ]] &&
      awk -v a="$start" -v b="$end" '$1 >= a && $1 <= b' "$R/gpu-clocks.txt" > "$out/gpu-clocks.txt"
    if [[ -z ${FAKE_GUEST:-} ]]; then
      bash "$(dirname "$0")/msi-xid.sh" "${start%.*}" "$(( ${end%.*} + 1 ))" "${QEMU_PID:-}" > "$out/xid.txt" ||
        echo unread > "$out/xid.txt"
      [[ -s $out/xid.txt ]] && log "$name: Xid lines: $(tr '\n' ';' < "$out/xid.txt")"
    fi
    echo ran > "$out/status"
    log "$name: rc $(cat "$out/rc"), $(cat "$out/wall_s") s"
  fi
  cp -R "$p" "$out/probe"
  [[ $(cat "$out/status") == ran ]] || log "$name: $(cat "$out/status")"
done
if [[ -n $HC ]]; then
  catch_wait
  kill "$HC" 2> /dev/null
  grep -E 'HANG|catch [0-9]+ written|REBOOT|restarting|answers again|panic report' "$R/hang/watch.txt" 2> /dev/null |
    sed 's/^/  hang catcher: /'
fi
exit 0
