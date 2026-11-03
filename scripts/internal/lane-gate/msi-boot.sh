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
#   desktop.png   the guest's screenshot of its screen after its run, at a quarter of its size
#   host.txt      "KEY VALUE" lines: the SSH port, the wait for the GPU, boot and desktop times
#                 (from QEMU's start), the fatal counts, the screenshot's colours and the share of
#                 its commonest colour (or desktop_capture none), how long the hold was
#   host-video.tsv  the host GPUs' hardware decode profiles, as Vulkan Video lists them
# boot.env sets: QEMU, ROM, OC (the OpenCore image), DISK (the macOS disk, only read: the boot
# writes an overlay), TAG, PAYLOAD (the guest's directory: guest-run.sh and its tools), CORPUS,
# GUEST_USER, GUEST_PASS, RGK_RUN (the host's boot scripts: rgk-boot2.sh), EXCLUSIVE, HOLD_S,
# LOCK_WAIT_S (the longest wait for the turnstile, a VM slot and the GPU, each), PHASES and
# PRESENT_S (guest-run.sh's phases and GATE_PRESENT_S).
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
  [[ -s $R/watchdog.pid ]] && kill "$(cat "$R/watchdog.pid")" 2> /dev/null
  [[ -n ${WATCHDOG:-} ]] && kill "$WATCHDOG" 2> /dev/null
  rm -f "$R/watchdog.pid"
  if [[ -f $B/launch.log ]]; then
    kv fatal_lines "$(grep -c 'reims-vgpu: fatal' "$B/launch.log")"
    grep -E 'the display shows IOSurface|reims-vgpu: fatal|panic| ERROR | WARN ' "$B/launch.log" > "$R/host-events.log"
    gzip -c "$B/launch.log" > "$R/launch.log.gz"
    cp -f "$B/serial.log" "$R/serial.log" 2> /dev/null
    n=$(grep -c 'reims-vgpu: fatal' "$B/serial.log" 2> /dev/null)
    kv serial_fatal_lines "${n:-0}"
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
# The host GPUs' hardware decoders, as Vulkan Video lists each GPU's decode profiles (a query: no
# GPU work): "GPU-NAME<TAB>gpu<TAB>-" for each GPU, then "GPU-NAME<TAB>CODEC<TAB>PROFILE" for each
# profile it decodes, h264 or hevc, the profile as the Std headers name it (MAIN, MAIN_10, HIGH).
# No file where vulkaninfo cannot answer.
timeout 30 vulkaninfo --show-video-props 2> /dev/null | awk -F' = ' '
  /^GPU[0-9]+:$/ { g = $0 }
  /^\tdeviceName / { name[g] = $2; print $2 "\tgpu\t-" }
  /videoCodecOperation = / { op = $2 }
  /stdProfileIdc = / && op ~ /_DECODE_H26[45]_/ {
    p = $2; sub(/^STD_VIDEO_H26[45]_PROFILE_IDC_/, "", p)
    print name[g] "\t" (op ~ /H265/ ? "hevc" : "h264") "\t" p }' | sort -u > "$R/host-video.tsv"
[[ -s $R/host-video.tsv ]] || rm -f "$R/host-video.tsv"
# The bench boot is an ordinary exclusive request: it waits its turn in the host's turnstile, in
# arrival order, and holds the GPU alone for QEMU's life.
excl=()
[[ ${EXCLUSIVE:-1} == 1 ]] && excl=(RGK_EXCLUSIVE=1 "RGK_EXCLUSIVE_MAX=$HOLD_S")
# rgk-boot2.sh picks the SSH port (PORT unset) and holds its lock for QEMU's life: "PORT n".
out=$(env -u PORT "${excl[@]}" RGK_SLOT_WAIT="${LOCK_WAIT_S:-3600}" QEMU="$QEMU" ROM="$ROM" \
  REIMS_VGPU_LOG=info bash "$RGK_RUN/rgk-boot2.sh" "$TAG" "$OC" "$DISK" 2>&1)
echo "$out" >> "$R/boot.log"
PORT=$(echo "$out" | awk '$1 == "PORT" && $2 ~ /^[0-9]+$/ {print $2; exit}')
[[ -n $PORT ]] || PORT=$(cat "$B/port" 2> /dev/null)
kv port "${PORT:-none}"
P=$(cat "$B/qemu.pid" 2> /dev/null)
[[ -n $P && $(cat "/proc/$P/comm" 2> /dev/null) == qemu-system-x86* ]] || P=
# The hold began when QEMU started, holding the lock.
if [[ -n $P ]]; then
  T0=$(( $(date +%s) - $(ps -o etimes= -p "$P" | tr -d ' ') ))
  kv lock_wait_s "$(( T0 - START ))"
fi
case $out in
*SSH_OK*) ;;
# The host's queue did not reach this boot in LOCK_WAIT_S: nothing of the candidate ran.
*GPU_WANTED_EXCLUSIVE* | *VM_SLOTS_BUSY* | *GPU_LOCKED*)
  log "BOOT_QUEUE_TIMEOUT $(echo "$out" | tail -2 | tr '\n' ' ')"; exit 7 ;;
*) log "BOOT_FAIL $(echo "$out" | tail -3 | tr '\n' ' ')"; exit 4 ;;
esac
[[ -n $P ]] || { log "BOOT_FAIL no QEMU pid in $B/qemu.pid"; exit 4; }
kv boot_ssh_s "$(( $(date +%s) - T0 ))"
log "QEMU $P up, SSH answers"

# This run's background helpers close every descriptor above 2 first: none may hold the GPU
# locks' descriptors, which gpu-locks.sh hands down, past this script's end.
closefds() { local f; for f in /proc/$BASHPID/fd/*; do f=${f##*/}; (( f > 2 )) && eval "exec $f>&-"; done; }
# Nor may they hold its stdout and stderr, the SSH session's, which would keep the session open
# after this script ends.
# The hold's end: QEMU stops there whatever the guest is doing. finish() stops the watchdog by its
# sleep's pid (watchdog.pid), and the watchdog then ends without stopping QEMU.
( closefds; sleep "$(left)" & s=$!; echo "$s" > "$R/watchdog.pid"; wait "$s" || exit 0
  echo "$(date +%T) watchdog: the hold's $HOLD_S s are up" >> "$R/boot.log"; kill "$P" 2> /dev/null ) > /dev/null 2>&1 &
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

# The window drags (msi-drag.py): a plain window and a glass one, each dragged DRAG_S seconds over
# QMP while the guest tracks the window, with the host's display updates and CPU in that span.
if [[ ${DRAG:-1} == 1 ]] && (( $(left) > 150 )); then
  log "window drags"
  PORT=$PORT GUEST_USER=$GUEST_USER GUEST_PASS=$GUEST_PASS DRAG_S=${DRAG_S:-10} \
    timeout "$(( $(left) - 60 ))" python3 -I "$(dirname "$0")/msi-drag.py" "$B" "$R" >> "$R/drag.log" 2>&1
  kv drag_rc $?
  log "window drags done"
fi

# The screen as WindowServer composed it, read back by the guest: the screenshot a person takes
# with Command-Shift-3, the keys sent over QMP to the guest's keyboard. The system's screenshot
# needs no screen-recording grant, which the lane's SIP-on guests cannot give a tool run over SSH,
# and QEMU's screendump shows nothing of reims-graphics-pci, whose frames go to the library's
# window.
rm -f "$R/desktop.png"
G 'defaults write com.apple.screencapture show-thumbnail -bool false; touch ~/lane-gate/.before-shot' 2> /dev/null
sleep 2
printf '%s\n' '{"execute":"qmp_capabilities"}' \
  '{"execute":"send-key","arguments":{"keys":[{"type":"qcode","data":"meta_l"},{"type":"qcode","data":"shift"},{"type":"qcode","data":"3"}],"hold-time":200}}' |
  timeout 10 socat - "UNIX-CONNECT:$B/qmp.sock" > /dev/null 2>&1
shot=
for _ in $(seq 1 15); do
  sleep 2
  shot=$(G 'find ~/Desktop -maxdepth 1 -name "*.png" -newer ~/lane-gate/.before-shot | head -1' 2> /dev/null)
  [[ -n $shot ]] && break
done
if [[ -n $shot ]] && sleep 1 && G "cp \"$shot\" ~/lane-gate/desktop.png" && Gcp "$GUEST_USER@127.0.0.1:lane-gate/desktop.png" "$R/desktop.png" &&
  [[ -s $R/desktop.png ]]; then
  kv desktop_colours "$(magick "$R/desktop.png" -format %k info: 2> /dev/null || echo 0)"
  # The share of the pixels the commonest colour covers: a white or blank capture is one colour.
  kv desktop_top_share "$(magick "$R/desktop.png" -format %c histogram:info:- 2> /dev/null |
    awk '{n = $1 + 0; t += n; if (n > m) m = n} END {printf "%.3f", t ? m / t : 1}')"
  magick "$R/desktop.png" -resize 25% "$R/desktop.png" 2> /dev/null || true
else
  kv desktop_capture none
fi
exit 0
