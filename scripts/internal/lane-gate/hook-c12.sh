#!/bin/bash
# A probe round's --qemu-hook for car 12's probes ('bash {DIR}/bin/hook-c12.sh {PID} {DIR}'): the
# host's side of three probes, each started by a MARK line in that probe's stdout (one newer than
# this hook: msi-probes.sh makes a probe's results afresh when it starts it).
#   c12-s1               (S1_PROBE, default c12-s1) from the probe's start (its stdout, or "MARK
#                        s1.start"): the guest's `log stream` of the kernel's and our kext's and
#                        bundle's lines, over SSH, into DIR/s1-guest-stream.log, until "MARK s1.end"
#                        or the probe's end (its rc) (a panic's lines reach the host up to the
#                        panic, whether or not logd wrote them). A session that ends before then is
#                        started again once SSH answers.
#   c12-shmem-pages      (SP_PROBE, default c12-shmem-pages) "MARK spmark.ready EPOCH pid=N" (once
#                        its marks are written): spscan.py (the probe's own copy, from the car's
#                        tree) on QEMU's RAM for guest process N, into DIR/spscan.txt.
#   c12-hreset1..3       (HRESET_PROBES) "MARK hardreset.go": a hard reset over QMP (system_reset,
#                        no shutdown in the guest), with QEMU's RssAnon before it and 20 s after it
#                        in DIR/hreset.tsv.
#   c12-scanout-restart  "MARK scanout.restart": msi-capture.py's scanout from that moment
#                        (CAPTURE_FROM_NOW) into DIR/restart/capture/, to the host's first present
#                        after the restart.
# Its own log: DIR/hook-c12.log. It ends with QEMU, or after 2 hours.
set -uo pipefail
qpid=$1 D=$2
here=$(cd "$(dirname "$0")" && pwd)
log() { echo "$(date +%T.%N) $*" >> "$D/hook-c12.log"; }
eval "$(grep -E '^(GUEST_USER|GUEST_PASS|PROBE_RESULTS|RGK_RUN|TAG)=' "$D/boot.env")"
B=$RGK_RUN/boots/$TAG
m=$D/.hook-c12-start; touch "$m"
SOPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PreferredAuthentications=password
  -o PubkeyAuthentication=no -o ConnectTimeout=5 -o ServerAliveInterval=5 -o ServerAliveCountMax=6
  -o LogLevel=ERROR)
port() { awk '$1 == "port" {print $2; exit}' "$D/host.txt" 2> /dev/null; }
# A MARK line in probe NAME's stdout, newer than this hook: prints the line.
mark() { local o=$PROBE_RESULTS/$1/stdout; [[ $o -nt $m ]] && grep -m1 "^MARK $2" "$o" 2> /dev/null; }
alive() { kill -0 "$qpid" 2> /dev/null; }
end=$(( $(date +%s) + 7200 ))
log "watching QEMU $qpid, round $D"
PRED='sender == "ReimsGraphics" OR sender == "reims-graphics" OR process == "kernel"'
s1=0 s1pid= sp=0 rs=0 S1=${S1_PROBE:-c12-s1} SP=${SP_PROBE:-c12-shmem-pages}
HR=${HRESET_PROBES:-c12-hreset1 c12-hreset2 c12-hreset3} hrdone=" "
rss() { awk '$1 == "RssAnon:" {print $2}' "/proc/$qpid/status" 2> /dev/null; }
started() { [[ $PROBE_RESULTS/$1/stdout -nt $m ]]; }
ended() { [[ -f $PROBE_RESULTS/$1/rc && $PROBE_RESULTS/$1/rc -nt $m ]]; }
while alive && (( $(date +%s) < end )); do
  # c12-s1: the guest's log stream from s1.start to s1.end.
  if ((s1 == 0)) && { started "$S1" || mark "$S1" 's1.start' > /dev/null; }; then
    s1=1; log "$S1 started: the guest's log stream into s1-guest-stream.log"
  fi
  if ((s1 == 1)); then
    if mark "$S1" 's1.end' > /dev/null || ended "$S1"; then
      [[ -n $s1pid ]] && kill "$s1pid" 2> /dev/null
      s1=2; log "$S1 ended: the stream stopped"
    elif [[ -z $s1pid ]] || ! kill -0 "$s1pid" 2> /dev/null; then
      [[ -n $s1pid ]] && log "the stream's session ended (rc $(wait "$s1pid" 2> /dev/null; echo $?))"
      p=$(port)
      if [[ -n $p ]] && timeout 10 sshpass -p "$GUEST_PASS" ssh -p "$p" "${SOPTS[@]}" "$GUEST_USER@127.0.0.1" true < /dev/null 2> /dev/null; then
        echo "=== $(date +%s.%N) stream starts" >> "$D/s1-guest-stream.log"
        sshpass -p "$GUEST_PASS" ssh -p "$p" "${SOPTS[@]}" "$GUEST_USER@127.0.0.1" \
          "log stream --style compact --level debug --predicate '$PRED'" < /dev/null >> "$D/s1-guest-stream.log" 2>&1 &
        s1pid=$!
        log "stream $s1pid started"
      fi
    fi
  fi
  # c12-shmem-pages: spscan.py on the marked guest process.
  if ((sp == 0)); then
    l=$(mark "$SP" 'spmark.ready') && [[ $l =~ pid=([0-9]+) ]] && {
      sp=1 gp=${BASH_REMATCH[1]}
      log "spmark.ready: spscan.py $qpid $gp"
      ( sudo -n python3 -I "$D/probes/$SP/spscan.py" "$qpid" "$gp" > "$D/spscan.txt" 2>&1
        log "spscan.py ended $?" ) &
    }
  fi
  # c12-scanout-restart: the capture from the restart on.
  if ((rs == 0)) && mark c12-scanout-restart 'scanout.restart' > /dev/null; then
    rs=1
    mkdir -p "$D/restart"; touch -r "$D/.start" "$D/restart/.start"
    ( exec sudo -n env CAPTURE_FROM_NOW=1 CAPTURE_MAX_S=600 CAPTURE_MODE="${CAPTURE_MODE:-3840x2400}" \
        CAPTURE_MS="${CAPTURE_MS:-100}" python3 -I "$here/msi-capture.py" scanout "$B" "$D/restart" ) &
    log "scanout.restart: the restart's scanout capture into restart/capture/ (the desktop after it: c12-after-restart's shot)"
  fi
  # c12-hreset*: a hard reset each, once.
  for h in $HR; do
    [[ $hrdone == *" $h "* ]] && continue
    mark "$h" 'hardreset.go' > /dev/null || continue
    hrdone+="$h "
    r0=$(rss)
    printf '%s\n' '{"execute":"qmp_capabilities"}' '{"execute":"system_reset"}' |
      timeout 10 socat - "UNIX-CONNECT:$B/qmp.sock" > /dev/null 2>&1
    log "$h: system_reset (rc $?), RssAnon ${r0:-?} kB before"
    ( sleep 20; echo "$(date +%s) $h before_kb ${r0:-?} after20s_kb $(rss)" >> "$D/hreset.tsv" ) &
  done
  sleep 0.5
done
[[ -n $s1pid ]] && kill "$s1pid" 2> /dev/null
log "QEMU gone or 2 hours passed (s1=$s1 spscan=$sp restart=$rs)"
