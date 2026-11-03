#!/bin/bash
# The lane gate's probe batch, its guest side, on the lane host: msi-boot.sh runs it in a probe
# boot (boot.env PROBES=DIR) once the guest's desktop is up and our Metal device is in it, and
# probe-batch.sh (on the Mac) runs it with FAKE_GUEST=DIR for a dry run.
#   msi-probes.sh
# Its environment:
#   PROBES          the round's probes, one directory each (run.sh, budget, tip, and any files
#                   run.sh reads), run in name order
#   PROBE_RESULTS   where each probe's results go: PROBE_RESULTS/NAME/, replaced
#   PROBE_TIP       the published tip the boot runs (its image and host): a probe whose tip names
#                   another commit is skipped
#   DEADLINE        the epoch second the hold ends: a probe starts only where its budget and 30 s
#                   fit before it, else it stays queued for the next round
#   B, R            the boot's directory (QEMU's launch.log) and the run's (gpu-clocks.txt)
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
#   out/            what run.sh left in its directory's out/, copied back
#   probe/          the probe as it was dropped
# run.sh gets PROBE_NAME and PROBE_PASS (the guest user's password, for sudo) in its environment.
set -uo pipefail
: "${PROBES:?}" "${PROBE_RESULTS:?}" "${PROBE_TIP:?}" "${DEADLINE:?}"
log() { echo "$(date +%T) probes: $*"; }
SOPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PreferredAuthentications=password
  -o PubkeyAuthentication=no -o ConnectTimeout=5 -o LogLevel=ERROR)
if [[ -n ${FAKE_GUEST:-} ]]; then
  G() { (cd "$FAKE_GUEST" && HOME=$FAKE_GUEST bash -c "$*"); }
  Gput() { cp -R "$1" "$FAKE_GUEST/$2"; }
  Gget() { cp -R "$FAKE_GUEST/$1" "$2"; }
else
  G() { sshpass -p "$GUEST_PASS" ssh -p "$PORT" "${SOPTS[@]}" "$GUEST_USER@127.0.0.1" "$@" < /dev/null; }
  Gput() { sshpass -p "$GUEST_PASS" scp -q -r -P "$PORT" "${SOPTS[@]}" "$1" "$GUEST_USER@127.0.0.1:$2"; }
  Gget() { sshpass -p "$GUEST_PASS" scp -q -r -P "$PORT" "${SOPTS[@]}" "$GUEST_USER@127.0.0.1:$1" "$2"; }
fi
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
settled=0
for p in "$PROBES"/*/; do
  p=${p%/}
  name=${p##*/}
  out=$PROBE_RESULTS/$name
  rm -rf "$out"
  mkdir -p "$out"
  budget=$(tr -dc 0-9 < "$p/budget" 2> /dev/null)
  tip=$(tr -dc 0-9a-f < "$p/tip" 2> /dev/null)
  if [[ ! -f $p/run.sh ]]; then
    echo "skipped: no run.sh" > "$out/status"
  elif [[ -z $budget ]] || (( budget < 1 || budget > 180 )); then
    echo "skipped: budget '${budget:-none}' is not 1 to 180 s" > "$out/status"
  elif ! same_tip "$tip" "$PROBE_TIP"; then
    echo "skipped: valid for ${tip:-no tip}, and the published tip is $PROBE_TIP" > "$out/status"
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
    wait "$gp"
    echo $? > "$out/rc"
    kill "$wp" 2> /dev/null; wait "$wp" 2> /dev/null
    end=$(now)
    echo "$start" > "$out/start"
    echo "$end" > "$out/end"
    awk -v a="$start" -v b="$end" 'BEGIN { printf "%.1f\n", b - a }' > "$out/wall_s"
    G "test -d ~/probes/$name/out" 2> /dev/null && Gget "probes/$name/out" "$out/out" 2> /dev/null
    if [[ -n ${B:-} && -f $B/launch.log ]]; then
      awk -v a="$(iso "$start")" -v b="$(iso "$end")" '
        /^20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]T/ { t = substr($1, 1, 23); keep = t >= a && t <= b }
        keep' "$B/launch.log" > "$out/host.log"
    fi
    [[ -n ${R:-} && -s $R/gpu-clocks.txt ]] &&
      awk -v a="$start" -v b="$end" '$1 >= a && $1 <= b' "$R/gpu-clocks.txt" > "$out/gpu-clocks.txt"
    echo ran > "$out/status"
    log "$name: rc $(cat "$out/rc"), $(cat "$out/wall_s") s"
  fi
  cp -R "$p" "$out/probe"
  [[ $(cat "$out/status") == ran ]] || log "$name: $(cat "$out/status")"
done
exit 0
