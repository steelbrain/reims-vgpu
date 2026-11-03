#!/bin/bash
# The lane gate's boot, on the lane host: bash msi-boot.sh RUNDIR, through the host's
# rgk-boot2.sh, which takes one of the host's VM slots, then the GPU lock, in its turn (the host's
# turnstile orders shared boots and exclusive runs by arrival), and the port's lock. The
# correctness boot (EXCLUSIVE=0) shares the GPU with other VMs; the bench boot (EXCLUSIVE=1,
# RGK_EXCLUSIVE) holds it alone for QEMU's life. Either is stopped after HOLD_S seconds.
# RUNDIR holds boot.env (below), and gets the results:
#   boot.log      what this script did, with times
#   guest.log     guest-run.sh's output, each line after the host's time (epoch seconds)
#   cpu.txt       QEMU's CPU time once a second: "EPOCH UTIME_TICKS STIME_TICKS HZ"
#   host-events.log  the host log's display updates ("the display shows IOSurface") and its
#                 fatal, error and warning lines, as the host logged them (local ISO times)
#   launch.log.gz, serial.log   QEMU's whole log and the guest's serial port
#   desktop.ppm   a screendump of the desktop after the guest's run
#   host.txt      "KEY VALUE" lines: the wait for the GPU, boot and desktop times (from QEMU's
#                 start), the fatal count, the screendump's colours, how long the hold was
# boot.env sets: QEMU, ROM, OC (the OpenCore image), DISK (the macOS disk, only read: the boot
# writes an overlay), PORT, TAG, PAYLOAD (the guest's directory: guest-run.sh and its tools),
# CORPUS, GUEST_USER, GUEST_PASS, RGK_RUN (the host's boot scripts: rgk-boot2.sh), EXCLUSIVE,
# HOLD_S, LOCK_WAIT_S (the longest wait for the turnstile, a VM slot and the GPU, each), PHASES
# and PRESENT_S (guest-run.sh's phases and GATE_PRESENT_S).
set -uo pipefail
R=$(cd "${1:?usage: msi-boot.sh RUNDIR}" && pwd)
. "$R/boot.env"
START=$(date +%s)
T0=$START
log() { echo "$(date +%T) $*" | tee -a "$R/boot.log"; }
kv() { echo "$1 $2" >> "$R/host.txt"; }
: > "$R/boot.log"
: > "$R/host.txt"

SOPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PreferredAuthentications=password
  -o PubkeyAuthentication=no -o ConnectTimeout=5 -o LogLevel=ERROR)
G() { sshpass -p "$GUEST_PASS" ssh -n -p "$PORT" "${SOPTS[@]}" "$GUEST_USER@127.0.0.1" "$@"; }
Gs() { sshpass -p "$GUEST_PASS" ssh -p "$PORT" "${SOPTS[@]}" "$GUEST_USER@127.0.0.1" "$@"; }
Gcp() { sshpass -p "$GUEST_PASS" scp -q -r -P "$PORT" "${SOPTS[@]}" "$@"; }
left() { echo $(( HOLD_S - ($(date +%s) - T0) )); }

P=
stop_qemu() {
  [[ -n $P ]] || return 0
  if [[ $(cat "/proc/$P/comm" 2>/dev/null) == qemu-system-x86* ]]; then
    log "stopping QEMU $P"
    kill "$P" 2> /dev/null
    for _ in $(seq 1 30); do kill -0 "$P" 2> /dev/null || break; sleep 1; done
    kill -0 "$P" 2> /dev/null && { log "QEMU $P still up: SIGKILL"; kill -9 "$P"; sleep 1; }
  fi
  P=
}
B=$RGK_RUN/boots/$TAG
finish() {
  stop_qemu
  [[ -n ${SAMPLER:-} ]] && kill "$SAMPLER" 2> /dev/null
  [[ -n ${WATCHDOG:-} ]] && kill "$WATCHDOG" 2> /dev/null
  if [[ -f $B/launch.log ]]; then
    kv fatal_lines "$(grep -c 'reims-vgpu: fatal' "$B/launch.log")"
    grep -E 'the display shows IOSurface|reims-vgpu: fatal|panic| ERROR | WARN ' "$B/launch.log" > "$R/host-events.log"
    gzip -c "$B/launch.log" > "$R/launch.log.gz"
    cp -f "$B/serial.log" "$R/serial.log" 2> /dev/null
    kv serial_fatal_lines "$(grep -c 'reims-vgpu: fatal' "$B/serial.log" 2> /dev/null || echo 0)"
  fi
  rm -f "$B"/*.qcow2
  kv hold_s "$(( $(date +%s) - T0 ))"
  log "BOOT_DONE"
}
trap finish EXIT

for f in "$QEMU" "$(dirname "$QEMU")/qemu-bundle/usr/local/share/qemu/efi-virtio.rom" "$ROM" "$OC" "$DISK" \
  "$PAYLOAD/guest-run.sh" "$CORPUS/MANIFEST" "$RGK_RUN/rgk-boot2.sh"; do
  [[ -s $f ]] || { log "BOOT_FAIL missing $f"; exit 2; }
done
log "boot $TAG ($([[ ${EXCLUSIVE:-1} == 1 ]] && echo exclusive || echo shared)): QEMU $QEMU, ROM $ROM, OC $OC"
# The bench boot is the gate's optional part: it joins the turnstile only once nobody waits for
# the GPU (no live marker in gpu-want-shared or gpu-want-exclusive), so it never holds up another
# boot or measurement, and gives up after YIELD_S seconds (default 4 h).
waiters() {
  local f n=0
  for f in "$RGK_RUN"/gpu-want-shared/* "$RGK_RUN"/gpu-want-exclusive/*; do
    [[ -e $f ]] && kill -0 "${f##*.}" 2> /dev/null && n=$((n + 1))
  done
  echo $n
}
if [[ ${EXCLUSIVE:-1} == 1 ]]; then
  y0=$(date +%s)
  until [[ $(waiters) == 0 ]]; do
    (( $(date +%s) - y0 < ${YIELD_S:-14400} )) || { log "BOOT_FAIL the GPU's queue was never empty in ${YIELD_S:-14400} s"; exit 3; }
    sleep 20
  done
  kv yield_wait_s "$(( $(date +%s) - y0 ))"
  log "the GPU's queue is empty after $(( $(date +%s) - y0 )) s: taking the turn"
fi
excl=()
[[ ${EXCLUSIVE:-1} == 1 ]] && excl=(RGK_EXCLUSIVE=1 "RGK_EXCLUSIVE_MAX=$HOLD_S")
out=$(env "${excl[@]}" RGK_SLOT_WAIT="${LOCK_WAIT_S:-3600}" QEMU="$QEMU" ROM="$ROM" PORT="$PORT" \
  REIMS_VGPU_LOG=info bash "$RGK_RUN/rgk-boot2.sh" "$TAG" "$OC" "$DISK" 2>&1)
echo "$out" >> "$R/boot.log"
P=$(cat "$B/qemu.pid" 2> /dev/null)
[[ -n $P && $(cat "/proc/$P/comm" 2> /dev/null) == qemu-system-x86* ]] || P=
# The hold began when QEMU started, holding the lock.
if [[ -n $P ]]; then
  T0=$(( $(date +%s) - $(ps -o etimes= -p "$P" | tr -d ' ') ))
  kv lock_wait_s "$(( T0 - START ))"
fi
case $out in
*SSH_OK*) ;;
*) log "BOOT_FAIL $(echo "$out" | tail -3 | tr '\n' ' ')"; exit 4 ;;
esac
[[ -n $P ]] || { log "BOOT_FAIL no QEMU pid in $B/qemu.pid"; exit 4; }
kv boot_ssh_s "$(( $(date +%s) - T0 ))"
log "QEMU $P up, SSH answers"

# This run's background helpers close every descriptor above 2 first: none may hold the GPU
# locks' descriptors, which gpu-locks.sh hands down, past this script's end.
closefds() { local f; for f in /proc/$BASHPID/fd/*; do f=${f##*/}; (( f > 2 )) && eval "exec $f>&-"; done; }
# The hold's end: QEMU stops there whatever the guest is doing.
( closefds; sleep "$(left)"; echo "$(date +%T) watchdog: the hold's $HOLD_S s are up" >> "$R/boot.log"; kill "$P" 2> /dev/null ) &
WATCHDOG=$!
( closefds; hz=$(getconf CLK_TCK); while [[ -r /proc/$P/stat ]]; do
    echo "$(date +%s.%N | cut -c1-14) $(sed 's/^.*) //' "/proc/$P/stat" | awk '{print $12, $13}') $hz"; sleep 1
  done ) > "$R/cpu.txt" 2> /dev/null &
SAMPLER=$!

desktop() {
  for _ in $(seq 1 36); do
    G 'pgrep -x WindowServer > /dev/null && pgrep -x Dock > /dev/null && pgrep -x Finder > /dev/null' 2> /dev/null && return 0
    sleep 5
  done
  return 1
}
desktop || { log "BOOT_FAIL no desktop (WindowServer, Dock, Finder) in 3 min"; exit 5; }
kv desktop_s "$(( $(date +%s) - T0 ))"
log "desktop up"

G "rm -rf ~/lane-gate" && Gcp "$PAYLOAD" "$GUEST_USER@127.0.0.1:lane-gate" && Gcp "$CORPUS" "$GUEST_USER@127.0.0.1:lane-gate/corpus" ||
  { log "BOOT_FAIL copying the payload into the guest"; exit 6; }
log "payload copied"

# RgfxGate installs the bundle at the first boot, and the accelerator registers for it the same
# boot: a Metal device that is not ours here is a failure (the run goes on after one reboot).
dev=$(G "perl -e 'alarm 30; exec @ARGV' ~/lane-gate/bench -l 2>/dev/null")
if [[ $dev == *"bench: device"* && $dev != *"Apple Paravirtual"* ]]; then
  kv first_boot_accel 1
else
  kv first_boot_accel 0
  log "no Metal device of ours on the first boot ($dev): rebooting the guest once"
  G "echo '$GUEST_PASS' | sudo -S -p '' reboot" > /dev/null 2>&1
  sleep 20
  for _ in $(seq 1 60); do G true 2> /dev/null && break; sleep 5; done
  desktop || { log "BOOT_FAIL no desktop after the reboot"; exit 5; }
fi
sleep 15

log "guest run: ${PHASES:-all phases}, $(left) s left"
ts() { perl -MTime::HiRes=time -ne '$| = 1; printf "%.3f %s", time, $_'; }
timeout "$(( $(left) - 40 ))" sshpass -p "$GUEST_PASS" ssh -n -p "$PORT" "${SOPTS[@]}" "$GUEST_USER@127.0.0.1" \
  "GATE_PASS='$GUEST_PASS' GATE_PRESENT_S=${PRESENT_S:-10} bash ~/lane-gate/guest-run.sh ~/lane-gate ${PHASES:-}" 2>&1 | ts > "$R/guest.log"
kv guest_rc "${PIPESTATUS[0]}"
log "guest run done"

printf '{"execute":"qmp_capabilities"}\n{"execute":"screendump","arguments":{"filename":"%s","device":"gfx"}}\n' \
  "$R/desktop.ppm" | timeout 15 socat - "UNIX-CONNECT:$B/qmp.sock" > /dev/null 2>&1
sleep 1
if [[ -s $R/desktop.ppm ]]; then
  kv desktop_colours "$(python3 -I - "$R/desktop.ppm" << 'EOF'
import sys
d = open(sys.argv[1], "rb").read()
# P6: magic, width, height, maxval, then the pixels.
parts, i = [], 0
while len(parts) < 4:
    while d[i:i + 1].isspace():
        i += 1
    j = i
    while not d[j:j + 1].isspace():
        j += 1
    parts.append(d[i:j])
    i = j
w, h = int(parts[1]), int(parts[2])
px = d[i + 1:]
seen = {px[o:o + 3] for y in range(0, h, 8) for o in range(y * w * 3, (y + 1) * w * 3, 24)}
print(len(seen))
EOF
)"
fi
exit 0
