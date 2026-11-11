#!/bin/bash
# The lane gate's boot, on the lane host: bash msi-boot.sh RUNDIR, inside a potluck job that holds
# the 5080 and the whole host (--exclusive: potluck's lease is the lock), through msi-launch.sh.
# Each boot waits for running builds to end and the CPU package to cool below 80 C before QEMU
# starts (msi-launch.sh), and is stopped after HOLD_S seconds; the bench boot (EXCLUSIVE=1) is
# also powered off by msi-launch.sh's watchdog after HOLD_S.
# RUNDIR holds boot.env (below), and gets the results:
#   boot.log      what this script did, with times
#   guest.log     guest-run.sh's output, each line after the host's time (epoch seconds); where the
#                 bench phase left GPU-bound cases with no repetition at the GPU's high clock
#                 (reps.py), their runs again (benchretry) before the phases after it
#   cpu.txt       QEMU's CPU time once a second: "EPOCH UTIME_TICKS STIME_TICKS HZ"
#   thermal.txt   the host CPU's heat once a second while QEMU runs: "EPOCH PKG_C PKG_COUNT PKG_MS
#                 CORE_COUNT CORE_MS", the package's temperature (x86_pkg_temp) and the throttle
#                 counters (thermal_throttle's package_ and core_throttle_count and _total_time_ms),
#                 each the largest over the CPUs; msi's package reaches 103 C under builds and
#                 throttles for seconds at a time, which moves CPU-bound numbers
#   host-events.log  the host log's display updates ("the display shows IOSurface"), its
#                 fatal, error and warning lines, and its work counts (reims_vgpu::work, where
#                 HOST_LOG enables it), as the host logged them (local ISO times)
#   launch.log.gz, serial.log   QEMU's whole log and the guest's serial port
#   desktop.png   the guest's screenshot of its screen after its run, at a quarter of its size
#   host.txt      "KEY VALUE" lines: the SSH port, the wait for the GPU, boot and desktop times
#                 (from QEMU's start), the fatal counts, the screenshot's colours and the share of
#                 its commonest colour (or desktop_capture none), how long the hold was, the
#                 host's load average beside the measurements, and host_quiet (1 where no build
#                 ran beside QEMU, 0 where msi-launch.sh said HOST_NOT_QUIET);
#                 at QEMU's start the host's free memory and page cache (memfree_, memavailable_,
#                 cached_start_mib), uptime (uptime_start_s: a fresh boot's page cache is empty),
#                 and the package's temperature and throttle counters (pkg_temp_start_c,
#                 pkg_throttle_count_start, pkg_throttle_ms_start, core_throttle_ms_start); at
#                 the end the temperature and what the counters grew by during the boot
#                 (pkg_temp_end_c, pkg_throttle_count_during, pkg_throttle_ms_during,
#                 core_throttle_ms_during); the memory compaction counters at the start and end and
#                 their growth (compact_stall_, compact_fail_, compact_success_start, _end and
#                 _during, /proc/vmstat), and the free memory at the end (memfree_end_mib, ...)
#   host-video.tsv  the host GPUs' hardware decode profiles, as each GPU's video API lists them
#   kernel-rgfx.log, kring.raw, kring-clock.txt   the kext's lines from the guest kernel's ring
#                 (it logs to no log stream), the guest's boot time and clock against the host's;
#                 kext-stalls.txt its texture-create stalls on the host's clock (below)
#   xid.txt       the kernel's NVRM Xid lines (the NVIDIA driver's GPU errors) from this script's
#                 start to its end, as msi-xid.sh prints them; host.txt's xid_lines counts them
#                 (unread where the kernel log could not be read), xid_own those of this boot's QEMU
#   gpuload.log   where LOAD is set: the fixed GPU load's duty each second, and over its run
#   gpu-clocks.txt  the GPU's power state and clocks ten times a second over the guest's run and
#                 the drags: "EPOCH PSTATE CORE_MHZ MEM_MHZ POWER_W"
#   appload/, appload.txt, appload.log   where boot.env sets APPLOAD=1: the app load's files, its
#                 numbers (msi-appload.sh), and its output
#   capture/, capture.txt   where boot.env sets CAPTURE=1: the boot as the display showed it
#                 (msi-capture.py's scanout reads and the guest's screenshots, the kext's log lines
#                 of the boot), and a summary of it
# boot.env sets: QEMU, ROM, OC (the OpenCore image), DISK (the macOS disk, only read: the boot
# writes an overlay), TAG, PAYLOAD (the guest's directory: guest-run.sh and its tools), CORPUS,
# GUEST_USER, GUEST_PASS, RGK_RUN (the host's shared files: boots/, ts-lines, kwin-reaper.sh),
# EXCLUSIVE, HOLD_S, LOCK_WAIT_S (unused: potluck's lease is the lock), PHASES and
# PRESENT_S (guest-run.sh's phases and GATE_PRESENT_S), and LOAD (a duty, 0 to 1: a fixed GPU
# load, gpuload -d LOAD, from the guest's run to the drags' end; unset for none), and HOST_LOG
# (the host's REIMS_VGPU_LOG filter, default info), CAPTURE (1: the boot capture, below), and
# APPLOAD (1: the app load after the measurements, msi-appload.sh, with the tools lane-gate.sh
# copies into this script's appload-tools/), and POST_PROBES and POST_PROBE_TIP (a gate boot's
# probes after everything else, msi-probes.sh's PROBES and PROBE_TIP, into post-probes/; at the
# end). A probe boot (probe-batch.sh) sets PROBES,
# PROBE_RESULTS and PROBE_TIP in place of PHASES and CORPUS (msi-probes.sh says what they are).
# A boot of another device than ours (a stock control) sets GFX_B64 (base64 of QEMU's graphics
# arguments, a shell fragment msi-launch.sh pastes in place of its reims-graphics-pci), QEMU_ENV
# ("K=V ...", each exported for QEMU, {DIR} the boot's directory) and METAL_CHECK=0 (no check that
# the guest's Metal device is ours).
set -uo pipefail
R=$(cd "${1:?usage: msi-boot.sh RUNDIR}" && pwd)
. "$R/boot.env"
if [[ -n ${GFX_B64:-} ]]; then GFX=$(printf '%s' "$GFX_B64" | base64 -d); export GFX; fi
for kv_ in ${QEMU_ENV:-}; do export "${kv_//\{DIR\}/$R}"; done
START=$(date +%s)
T0=$START
log() { echo "$(date +%T) $*" | tee -a "$R/boot.log"; }
kv() { echo "$1 $2" >> "$R/host.txt"; }
# How many lines of the warning whose message starts with $1 (an ERE) the library wrote into log
# $2: the library writes the first of each kind at warn and the later ones at debug, counting them
# in its line of repeats at each reset and at the exit (docs/conventions.md), "<n> more of
# <file>:<line>[ [<kind>]] (<target>: <the first line's message>)".
warned() {
  local first more
  first=$(grep -cE " WARN [^ ]+: $1" "$2")
  more=$(grep -oE "[0-9]+ more of [^ ]+( \[[^]]*\])? \([^ ]+: $1" "$2" | awk '{ s += $1 } END { print s + 0 }')
  echo $((first + more))
}
# The host CPU's heat now: "PKG_C PKG_COUNT PKG_MS CORE_COUNT CORE_MS" (thermal.txt's fields, the
# counters the largest over the CPUs), read with the shell's own read, as a sampler does it once a
# second over 30-odd CPUs.
thermal_now() {
  local z t= pc=0 pm=0 cc=0 cm=0 v d
  for z in /sys/class/thermal/thermal_zone*; do
    read -r v < "$z/type" 2> /dev/null && [[ $v == x86_pkg_temp ]] && { read -r t < "$z/temp"; break; }
  done
  for d in /sys/devices/system/cpu/cpu[0-9]*/thermal_throttle; do
    read -r v < "$d/package_throttle_count" 2> /dev/null && (( v > pc )) && pc=$v
    read -r v < "$d/package_throttle_total_time_ms" 2> /dev/null && (( v > pm )) && pm=$v
    read -r v < "$d/core_throttle_count" 2> /dev/null && (( v > cc )) && cc=$v
    read -r v < "$d/core_throttle_total_time_ms" 2> /dev/null && (( v > cm )) && cm=$v
  done
  [[ -n $t ]] && t=$(( t / 1000 )) || t=na
  echo "$t $pc $pm $cc $cm"
}
# The lane host's display as the boot met it: the lid, and each output kwin drives (kscreen-doctor
# in the desktop session's Wayland display): "lid=closed eDP-1=3840x2400@120,scale=1.9", as KEY.
display_state() {
  local lid outs u w
  lid=$(awk '{print $2}' /proc/acpi/button/lid/*/state 2> /dev/null | head -1)
  u=$(id -u)
  for w in /run/user/$u/wayland-[0-9]*; do
    [[ -S $w ]] || continue
    outs=$(XDG_RUNTIME_DIR=/run/user/$u DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$u/bus WAYLAND_DISPLAY=${w##*/} \
      QT_QPA_PLATFORM=wayland timeout 5 kscreen-doctor -j 2> /dev/null | python3 -I -c '
import json, sys
for o in json.load(sys.stdin).get("outputs", []):
    if not o.get("connected"):
        continue
    m = {x["id"]: x for x in o.get("modes", [])}.get(o.get("currentModeId"), {})
    z = m.get("size", {})
    v = "%sx%s@%.0f,scale=%s" % (z.get("width"), z.get("height"), m.get("refreshRate", 0), o.get("scale"))
    print(o.get("name") + "=" + (v if o.get("enabled") else "off"))' 2> /dev/null | tr '\n' ' ')
    [[ -n $outs ]] && break
  done
  local reaper
  reaper=$(XDG_RUNTIME_DIR=/run/user/$u DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$u/bus \
    timeout 5 systemctl --user is-active kwin-reaper 2> /dev/null)
  kv "$1" "lid=${lid:-none} ${outs:-outputs=unread} reaper=${reaper:-unknown}"
}
# The kwin reaper's lines (rgk-run's kwin-reaper.sh: kwin's held windows freed while no output is
# on, with kwin's dma-buf bytes and the GPU's memory before and after) from this script's start to
# now, into kwin-reaper.txt.
reaper_lines() {
  local a
  a=$(date -d "@$START" '+%m-%d %H:%M:%S')
  awk -v a="$a" -v b="$(date '+%m-%d %H:%M:%S')" '{ t = $1 " " $2 } t >= a && t <= b' \
    "$RGK_RUN/kwin-reaper.log" > "$R/kwin-reaper.txt" 2> /dev/null
  kv reaper_lines "$(grep -c . "$R/kwin-reaper.txt" 2> /dev/null || echo 0)"
}
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
# The GPU's power state and clocks beside the measurements, ten times a second (nvidia-smi, where
# the host has it): "EPOCH PSTATE CORE_MHZ MEM_MHZ POWER_W". msi's 5080 holds its core at 1.2 to
# 1.5 GHz under its power cap for up to 12 s, which a GPU-bound case's numbers follow: reps.py
# keeps a number only from time the core ran high throughout.
CLOCKPID=
stop_clocks() {  # the sampler's pipeline, nvidia-smi among it, then the sampler
  [[ -n $CLOCKPID ]] || return 0
  kill $(ps -o pid= --ppid "$CLOCKPID") "$CLOCKPID" 2> /dev/null
  CLOCKPID=
}
LOADPID=
stop_load() {
  [[ -n $LOADPID ]] || return 0
  kill "$LOADPID" 2> /dev/null
  wait "$LOADPID" 2> /dev/null
  LOADPID=
  kv load_duty "$(sed -n 's/^gpuload: end duty=\([0-9.]*\).*/\1/p' "$R/gpuload.log")"
  log "fixed GPU load stopped: $(tail -1 "$R/gpuload.log")"
}
# The host's state at the boot's end, read before QEMU stops (its guest RAM still pinned).
end_readings() {
  # The host's free memory and its memory compaction at the end: the runtime pins all of the
  # guest's RAM at the start, and a host short of free pages compacts (compact_stall: a process
  # that waited for it), which moves a boot's times.
  while read -r k v _; do kv "$(tr 'A-Z' 'a-z' <<< "${k%:}")_end_mib" "$(( v / 1024 ))"
  done < <(grep -E '^(MemFree|MemAvailable|Cached):' /proc/meminfo)
  if [[ -s $R/vmstat-start.txt ]]; then
    while read -r k v; do
      v0=$(awk -v k="$k" '$1 == k {print $2}' "$R/vmstat-start.txt")
      kv "${k}_end" "$v"; [[ -n $v0 ]] && kv "${k}_during" "$(( v - v0 ))"
    done < <(grep -E '^compact_(stall|fail|success) ' /proc/vmstat)
  fi
  if [[ -s $R/thermal-start.txt ]]; then
    read -r _ pc0 pm0 _ cm0 < "$R/thermal-start.txt"
    read -r t1 pc1 pm1 _ cm1 <<< "$(thermal_now)"
    kv pkg_temp_end_c "$t1"
    kv pkg_throttle_count_during "$(( pc1 - pc0 ))"
    kv pkg_throttle_ms_during "$(( pm1 - pm0 ))"
    kv core_throttle_ms_during "$(( cm1 - cm0 ))"
  fi
}
finish() {
  local qp=$P
  [[ -n ${KRINGPID:-} ]] && kill "$KRINGPID" 2> /dev/null
  [[ -d $R/capture ]] && touch "$R/capture/stop"
  stop_clocks
  stop_load
  end_readings
  stop_qemu
  # The NVIDIA driver's GPU errors from this script's start to now (msi-xid.sh), each with its
  # process: xid.txt, and xid_own those from this boot's QEMU, which fail the gate.
  if bash "$(dirname "$0")/msi-xid.sh" "$START" "$(date +%s)" "$qp" > "$R/xid.txt"; then
    kv xid_lines "$(grep -c . "$R/xid.txt")"
    kv xid_own "$(grep -c ' own ' "$R/xid.txt")"
    [[ -s $R/xid.txt ]] && log "Xid lines: $(tr '\n' ';' < "$R/xid.txt")"
  else
    kv xid_lines unread
  fi
  [[ -n ${SAMPLER:-} ]] && kill "$SAMPLER" 2> /dev/null
  [[ -n ${THERMAL:-} ]] && kill "$THERMAL" 2> /dev/null
  [[ -s $R/watchdog.pid ]] && kill "$(cat "$R/watchdog.pid")" 2> /dev/null
  [[ -n ${WATCHDOG:-} ]] && kill "$WATCHDOG" 2> /dev/null
  rm -f "$R/watchdog.pid"
  if [[ -f $B/launch.log ]]; then
    kv fatal_lines "$(grep -c 'reims-vgpu: fatal' "$B/launch.log")"
    # The host's records of paths a tip adds: queues made unshared (RGFX_QUEUE_UNSHARED),
    # SUBMIT_ROOM records posted (reims_vgpu_graphics::room at debug, HOST_LOG), and ICB_SLOTS messages
    # whose runs were not all carried out.
    kv queues_unshared "$(grep -c 'made unshared' "$B/launch.log")"
    kv submit_room_posted "$(grep -c 'SUBMIT_ROOM posted' "$B/launch.log")"
    kv icb_slots_refused "$(warned 'ICB_SLOTS of' "$B/launch.log")"
    # The boot's hold of host quiet: HOST_QUIET where no build ran beside QEMU, HOST_NOT_QUIET where
    # the wait for running builds ran out (msi-launch.sh).
    case $(grep -m1 -o -E 'HOST_(NOT_)?QUIET' "$B/launch.log") in
    HOST_QUIET) kv host_quiet 1 ;;
    HOST_NOT_QUIET) kv host_quiet 0 ;;
    esac
    grep -E 'the display shows IOSurface|reims-vgpu: fatal|panic| ERROR | WARN | reims_vgpu::work: ' "$B/launch.log" > "$R/host-events.log"
    gzip -c "$B/launch.log" > "$R/launch.log.gz"
    cp -f "$B/serial.log" "$R/serial.log" 2> /dev/null
    n=$(grep -c 'reims-vgpu: fatal' "$B/serial.log" 2> /dev/null)
    kv serial_fatal_lines "${n:-0}"
  fi
  rm -f "$B"/*.qcow2
  kv hold_s "$(( $(date +%s) - T0 ))"
  log "BOOT_DONE"
}
# bash 3.2, macOS's, exits 0 from a set -u error where an EXIT trap is set, with $? 0 in the
# trap too, so an aborted run would read as a pass (docs/lessons.md): the trap fails a run
# that exits 0 short of its end, which sets `ended`.
ended=0
trap 'rc=$?; finish; [[ $rc != 0 || $ended == 1 ]] || rc=1; exit $rc' EXIT

# A probe boot (PROBES) runs msi-probes.sh in place of guest-run.sh, with no corpus.
needs=("$PAYLOAD/guest-run.sh" "$CORPUS/MANIFEST")
[[ -n ${PROBES:-} ]] && needs=("$PAYLOAD/bench" "$(dirname "$0")/msi-probes.sh")
for f in "$QEMU" "$(dirname "$QEMU")/qemu-bundle/usr/local/share/qemu/efi-virtio.rom" "$ROM" "$OC" "$DISK" \
  "${needs[@]}" "$(dirname "$0")/msi-launch.sh"; do
  [[ -s $f ]] || { log "BOOT_FAIL missing $f"; exit 2; }
done
log "boot $TAG ($([[ ${EXCLUSIVE:-1} == 1 ]] && echo exclusive || echo shared)): QEMU $QEMU, ROM $ROM, OC $OC"
# LOAD: a fixed GPU load beside the guest's run and the drags (gpuload, built here for this host).
GL=
if [[ -n ${LOAD:-} ]]; then
  GL=$R/gpuload
  # Built here, inside the boot's job, before QEMU starts.
  ( cd "$R" && glslc -mfmt=c "$(dirname "$0")/gpuload.comp" -o gpuload.spv.inc &&
    cc -O2 -Wall -Wextra -Werror -I"$R" "$(dirname "$0")/gpuload.c" -lvulkan -o gpuload ) \
    >> "$R/boot.log" 2>&1 || { log "BOOT_FAIL missing gpuload (its build failed)"; exit 2; }
fi
# The host GPUs' hardware decoders, as each GPU's own video API lists them (queries: no GPU
# work): "GPU-NAME<TAB>gpu<TAB>-" for each GPU, then "GPU-NAME<TAB>CODEC<TAB>PROFILE" for each
# profile it decodes, h264 or hevc, the profile as VA-API names it without its codec (HIGH, MAIN,
# MAIN10). A GPU's name is vulkaninfo's deviceName, which the guest's Metal device carries, found by
# its PCI address. Its decoders are vainfo's VLD profiles on its render node, or, on an NVIDIA GPU,
# NVDEC's, as the crate's nvidia-probe beside QEMU answers them (msi-build.sh builds it): 8-bit
# 4:2:0 H.264 as Baseline, Main and High, and HEVC as Main at 8 bits and Main 10 at 10, the
# profiles the reims-graphics device's NVDEC arm offers for them. No file where vulkaninfo
# cannot answer.
probe=$(dirname "$QEMU")/nvidia-probe
timeout 30 vulkaninfo 2> /dev/null | awk -F' = ' '
  /^GPU[0-9]+:$/ { g = $0 }
  /^\tdeviceName / { name[g] = $2 }
  /^\tpciDomain / { dom[g] = $2 }
  /^\tpciBus / { bus[g] = $2 }
  /^\tpciDevice / { dev[g] = $2 }
  /^\tpciFunction / { fn[g] = $2 }
  END { for (g in name) if (g in bus)
    printf "%s\t%04x:%02x:%02x.%x\n", name[g], strtonum(dom[g]), strtonum(bus[g]),
      strtonum(dev[g]), strtonum(fn[g]) }' | sort -u |
  while IFS=$'\t' read -r name pci; do
    printf '%s\tgpu\t-\n' "$name"
    driver=$(basename "$(readlink -f "/sys/bus/pci/devices/$pci/driver")")
    if [[ $driver == nvidia ]]; then
      [[ -x $probe ]] && timeout 30 "$probe" "$pci" 2> /dev/null | awk -v n="$name" '
        /^  H264 Yuv420 8-bit:/ { print n "\th264\tBASELINE"; print n "\th264\tMAIN"; print n "\th264\tHIGH" }
        /^  Hevc Yuv420 8-bit:/ { print n "\thevc\tMAIN" }
        /^  Hevc Yuv420 10-bit:/ { print n "\thevc\tMAIN10" }'
    else
      node=$(find "/sys/bus/pci/devices/$pci/drm" -maxdepth 1 -name 'renderD*' -printf '%f\n' 2> /dev/null | sort | head -n1)
      [[ -n $node ]] && timeout 30 vainfo --display drm --device "/dev/dri/$node" 2> /dev/null | awk -v n="$name" '
        $1 ~ /^VAProfile(H264|HEVC)/ && /VAEntrypointVLD/ {
          p = $1; sub(/:$/, "", p); sub(/^VAProfile/, "", p); c = (p ~ /^HEVC/) ? "hevc" : "h264"; sub(/^(H264|HEVC)/, "", p)
          print n "\t" c "\t" toupper(p) }'
    fi
  done | sort -u > "$R/host-video.tsv"
[[ -s $R/host-video.tsv ]] || rm -f "$R/host-video.tsv"
# Both boots run in a potluck job that holds the host alone (--exclusive), and both wait for running
# builds and the cool-down first (msi-launch.sh). The bench boot is also powered off after HOLD_S.
excl=(RGK_QUIET=1)
[[ ${EXCLUSIVE:-1} == 1 ]] && excl=(RGK_EXCLUSIVE=1 "RGK_EXCLUSIVE_MAX=$HOLD_S")
# Where the boot's time goes: msi-launch.sh writes run.sh, which waits for running builds until its
# HOST_QUIET or HOST_NOT_QUIET line in QEMU's log, then for the cool-down (COOL_WAIT), says
# GPU_HELD and execs QEMU, whose lines carry their times. A watch takes the times of those lines
# (boot-times.txt: quiet, held), and the hold counts from GPU_HELD.
sub() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.1f", a - b }'; }
stamp() { date -d "$1" +%s.%N 2> /dev/null | cut -c1-14; }
touch "$R/.start"
# The boot capture (CAPTURE=1, msi-capture.py): the guest's scanout read from QEMU's memory every
# CAPTURE_MS (default 100) ms from QEMU's start to the host's first present of an IOSurface (from
# there the display's content is in the host's device caches, and the scanout is not drawn), and
# the guest's screenshots once a second from the desktop (WindowServer, Dock and Finder up: Shift
# held during the login stops macOS's automatic login) until CAPTURE_SETTLE_S (default 60) seconds
# after it, where the Dock has settled (below). The scanout's reader runs as root, for
# /proc/PID/mem.
CAPPIDS=()
# A cold pipeline cache (COLD_CACHE=KEY) and a command at QEMU's start (QEMU_HOOK_B64): msi-launch.sh's
# run.sh execs a wrapper, which execs QEMU in turn, so QEMU keeps run.sh's pid (msi-launch.sh
# counts SSH only from that pid). The wrapper sets the host's cache
# entry KEY aside and leaves a process holding the lock that, once QEMU has stopped, removes the
# entry QEMU made and puts KEY back; it starts the hook with no lock held. A command before QEMU
# (PRE_QEMU_B64) runs in it under the lock, just before the exec, into R/pre-qemu.txt.
if [[ -n ${COLD_CACHE:-} || -n ${QEMU_HOOK_B64:-} || -n ${PRE_QEMU_B64:-} ]]; then
  hook=$(printf '%s' "${QEMU_HOOK_B64:-}" | base64 -d)
  pre=$(printf '%s' "${PRE_QEMU_B64:-}" | base64 -d)
  {
    echo '#!/bin/bash'
    echo "C=$HOME/.cache/reims-vgpu-vulkan/v1/${COLD_CACHE:-}"
    echo "A=$HOME/.cache/reims-vgpu-vulkan/v1/.lane-gate-aside-${COLD_CACHE:-}"
    echo "COLD=${COLD_CACHE:+1}"
    echo "REAL=$(printf '%q' "$QEMU")"
    echo "HOOK=$(printf '%q' "$hook")"
    echo "PRE=$(printf '%q' "$pre")"
    echo "R=$(printf '%q' "$R")"
    cat << 'WRAP'
say() { echo "$(date +%Y-%m-%dT%H:%M:%S.%3N%z) lane-gate qemu wrapper: $*"; }
if [[ -n $COLD ]]; then
  [[ -e $A && ! -e $C ]] && mv "$A" "$C" && say "put back an earlier boot's $A"
  if [[ -e $A ]]; then say "$A is there already: the cache stays warm"; COLD=
  elif [[ -e $C ]]; then mv "$C" "$A" && say "the pipeline cache entry set aside: $C"
  else say "no pipeline cache entry $C: cold already"; COLD=; fi
fi
q=$$  # QEMU's, once this script execs it
if [[ -n $COLD ]]; then
  ( while kill -0 "$q" 2> /dev/null && [[ $(cat "/proc/$q/stat" 2> /dev/null | cut -d' ' -f3) != Z ]]; do sleep 0.5; done
    rm -rf "$C"; mv "$A" "$C" && say "the pipeline cache entry put back: $C" ) &
fi
if [[ -n $PRE ]]; then
  say "before QEMU: $PRE"
  ( exec 6>&- 7>&- 8>&-; eval "$PRE" ) > "$R/pre-qemu.txt" 2>&1
  say "before QEMU: exit $?"
fi
if [[ -n $HOOK ]]; then
  ( exec 6>&- 7>&- 8>&- 9>&-; sleep 0.5; h=${HOOK//\{PID\}/$q}; h=${h//\{DIR\}/$R}; say "hook: $h"; eval "$h" ) &
fi
exec "$REAL" "$@"
WRAP
  } > "$R/qemu-wrap.sh"
  chmod +x "$R/qemu-wrap.sh"
  QEMU=$R/qemu-wrap.sh
fi
# The hang catcher from QEMU's start (CATCH_FROM_BOOT=1, msi-hang-catch.py --from-boot): a boot's
# own restart or silence before the desktop; it writes R/hang/, and the probes use it as theirs.
HCPID=
if [[ ${CATCH_FROM_BOOT:-0} == 1 ]]; then
  mkdir -p "$R/hang"
  ( for f in /proc/$BASHPID/fd/*; do f=${f##*/}; (( f > 2 )) && eval "exec $f>&-"; done
    qp=
    while sleep 0.2; do  # this boot's QEMU and its SSH port
      [[ $B/qemu.pid -nt $R/.start && -s $B/port ]] || continue
      qp=$(cat "$B/qemu.pid" 2> /dev/null)
      [[ $(cat "/proc/$qp/comm" 2> /dev/null) == qemu-system-x86* ]] && break
    done
    exec python3 -I "$(dirname "$0")/msi-hang-catch.py" --from-boot --boot-dir "$B" --pid "$qp" \
      --port "$(tr -dc 0-9 < "$B/port")" --out "$R/hang" --user "$GUEST_USER" --password "$GUEST_PASS"
  ) > "$R/hang.log" 2>&1 &
  HCPID=$!
  CAPPIDS+=($HCPID)
fi
if [[ ${CAPTURE:-0} == 1 ]]; then
  rm -rf "$R/capture" && mkdir -p "$R/capture"
  ( for f in /proc/$BASHPID/fd/*; do f=${f##*/}; (( f > 2 )) && eval "exec $f>&-"; done
    exec sudo -n env CAPTURE_MODE="${CAPTURE_MODE:-3840x2400}" CAPTURE_MS="${CAPTURE_MS:-100}" \
      python3 -I "$(dirname "$0")/msi-capture.py" scanout "$B" "$R" ) > /dev/null 2>&1 &
  CAPPIDS+=($!)
  ( for f in /proc/$BASHPID/fd/*; do f=${f##*/}; (( f > 2 )) && eval "exec $f>&-"; done
    exec python3 -I "$(dirname "$0")/msi-capture.py" shots "$B" "$R" ) > /dev/null 2>&1 &
  CAPPIDS+=($!)
fi
( for f in /proc/$BASHPID/fd/*; do f=${f##*/}; (( f > 2 )) && eval "exec $f>&-"; done
  q= g=
  while sleep 0.5; do
    [[ -f $B/launch.log && $B/launch.log -nt $R/.start ]] || continue
    [[ -z $q ]] && grep -q -x -E 'HOST_(NOT_)?QUIET' "$B/launch.log" && q=1 && echo "quiet $(date +%s.%N | cut -c1-14)"
    # The host's free memory as QEMU starts (it moves a boot's time by 30 s and more where it is
    # below the guest's RAM: the runtime imports all of it at the start).
    grep -q -x GPU_HELD "$B/launch.log" && { echo "held $(date +%s.%N | cut -c1-14)"
      grep -E '^(MemFree|MemAvailable|Cached):' /proc/meminfo > "$R/meminfo-start.txt"
      thermal_now > "$R/thermal-start.txt"; cut -d' ' -f1 /proc/uptime > "$R/uptime-start.txt"
      grep -E '^compact_(stall|fail|success) ' /proc/vmstat > "$R/vmstat-start.txt"; break; }
  done ) > "$R/boot-times.txt" 2> /dev/null &
TIMES=$!
# msi-launch.sh picks the SSH port (PORT unset): "PORT n".
out=$(env -u PORT "${excl[@]}" RGK_RUN="$RGK_RUN" QEMU="$QEMU" ROM="$ROM" \
  REIMS_VGPU_LOG="${HOST_LOG:-info}" bash "$(dirname "$0")/msi-launch.sh" "$TAG" "$OC" "$DISK" 2>&1)
T_SSH=$(date +%s.%N | cut -c1-14)
kill "$TIMES" 2> /dev/null
echo "$out" >> "$R/boot.log"
PORT=$(echo "$out" | awk '$1 == "PORT" && $2 ~ /^[0-9]+$/ {print $2; exit}')
[[ -n $PORT ]] || PORT=$(cat "$B/port" 2> /dev/null)
kv port "${PORT:-none}"
P=$(cat "$B/qemu.pid" 2> /dev/null)
[[ -n $P && $(cat "/proc/$P/comm" 2> /dev/null) == qemu-system-x86* ]] || P=
T_QEMU=
if [[ -f $B/launch.log && $B/launch.log -nt $R/.start ]]; then
  T_QEMU=$(stamp "$(grep -m1 -E '^20[0-9-]+T[0-9:.]+[+-][0-9]{4} ' "$B/launch.log" | cut -d' ' -f1)")
fi
T_RUN=$([[ $B/run.sh -nt $R/.start ]] && stat -c %Y "$B/run.sh")
T_GPU=$(awk '$1 == "gpu" {print $2}' "$R/boot-times.txt")
T_QUIET=$(awk '$1 == "quiet" {print $2}' "$R/boot-times.txt")
T_HELD=$(awk '$1 == "held" {print $2}' "$R/boot-times.txt")
# potluck's lease held the GPU before the job started: no GPU wait here. run.sh's wait for running
# builds ends at its quiet line, and its cool-down at GPU_HELD (cool_wait_s).
T_GPU=${T_RUN:-$START}
[[ -n $T_RUN ]] && kv turn_wait_s "$(sub "$T_RUN" "$START")"
[[ -n $T_GPU ]] && kv gpu_wait_s "$(sub "$T_GPU" "${T_RUN:-$START}")"
[[ -n $T_GPU && -n $T_QUIET ]] && kv quiet_wait_s "$(sub "$T_QUIET" "$T_GPU")"
[[ -n $T_QUIET && -n $T_HELD ]] && kv cool_wait_s "$(sub "$T_HELD" "$T_QUIET")"
# The hold began when run.sh said GPU_HELD, after the cool-down (else when QEMU started).
T0=${T_HELD:-$T_QEMU}
T0=${T0%.*}
[[ -n $T0 ]] && kv lock_wait_s "$(( T0 - START ))"
[[ -s $R/meminfo-start.txt ]] && while read -r k v _; do
  kv "$(tr 'A-Z' 'a-z' <<< "${k%:}")_start_mib" "$(( v / 1024 ))"
done < "$R/meminfo-start.txt"
[[ -s $R/uptime-start.txt ]] && kv uptime_start_s "$(cut -d. -f1 "$R/uptime-start.txt")"
[[ -s $R/vmstat-start.txt ]] && while read -r k v; do kv "${k}_start" "$v"; done < "$R/vmstat-start.txt"
if [[ -s $R/thermal-start.txt ]]; then
  read -r t pc pm cc cm < "$R/thermal-start.txt"
  kv pkg_temp_start_c "$t"; kv pkg_throttle_count_start "$pc"; kv pkg_throttle_ms_start "$pm"
  kv core_throttle_ms_start "$cm"
fi
[[ -n $T0 ]] || T0=$START
case $out in
*SSH_OK*) ;;
# Kept from the host's queue before potluck (rgk-boot2.sh's refusals): nothing of the candidate ran.
*GPU_WANTED_EXCLUSIVE* | *VM_SLOTS_BUSY* | *GPU_LOCKED*)
  log "BOOT_QUEUE_TIMEOUT $(echo "$out" | tail -2 | tr '\n' ' ')"; exit 7 ;;
*) log "BOOT_FAIL $(echo "$out" | tail -3 | tr '\n' ' ')"; exit 4 ;;
esac
[[ -n $P ]] || { log "BOOT_FAIL no QEMU pid in $B/qemu.pid"; exit 4; }
# The boot's own time: from QEMU's first line to SSH answering.
[[ -n $T_QEMU ]] && kv boot_ssh_s "$(sub "$T_SSH" "$T_QEMU")"
log "QEMU $P up, SSH answers"
display_state display_start

# This run's background helpers close every descriptor above 2 first: none may hold a
# descriptor this script was handed past its end.
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
( closefds; while [[ -r /proc/$P/stat ]]; do echo "$(date +%s.%N | cut -c1-14) $(thermal_now)"; sleep 1; done ) \
  > "$R/thermal.txt" 2> /dev/null &
THERMAL=$!

desktop() {
  for _ in $(seq 1 36); do
    G 'pgrep -x WindowServer > /dev/null && pgrep -x Dock > /dev/null && pgrep -x Finder > /dev/null' 2> /dev/null && return 0
    sleep 5
  done
  return 1
}
desktop || { log "BOOT_FAIL no desktop (WindowServer, Dock, Finder) in 3 min"; exit 5; }
[[ -n $T_QEMU ]] && kv desktop_s "$(sub "$(date +%s.%N | cut -c1-14)" "$T_QEMU")"
log "desktop up"
T_DESK=$(date +%s)
# The capture's screenshots start here (msi-capture.py shots), never during the login.
[[ ${CAPTURE:-0} == 1 ]] && touch "$R/capture/desktop-up"
# The kext's lines: its IOLog lines reach only the kernel's ring (dmesg, 128 KiB, which a busy
# desktop's other lines fill in a minute or two), no log stream. From the desktop on, the ring's
# Rgfx, RGFX and ReimsGraphics lines are taken every KRING_S (default 5) s into kring.raw ("[UPTIME]: LINE"), in the
# correctness boot and in a captured one (the bench boot's at the desktop and at its end only, so
# no read lands among its measurements); kernel-rgfx.log holds them once each, in order. The guest's boot
# time and clock against the host's (kring-clock.txt: "BOOTTIME HOST_MINUS_GUEST") put each line
# on the host's clock.
kring() {
  G "echo '$GUEST_PASS' | sudo -S -p '' dmesg" 2> /dev/null |
    perl -ne 'while (/\[ *(\d+\.\d+)\]: ((?:R[Gg][Ff][Xx]|ReimsGraphics).*?)(?=\[ *\d+\.\d+\]: |Sandbox: |$)/g) { print "[$1]: $2\n" }' >> "$R/kring.raw"
}
kring_log() {  # kring.raw's lines once each, by uptime
  sort -u "$R/kring.raw" 2> /dev/null | sort -t']' -k1.2,1n > "$R/kernel-rgfx.log"
}
h0=$(date +%s.%N | cut -c1-14)
g=$(G "perl -MTime::HiRes=time -e 'printf \"%.3f\\n\", time'; sysctl -n kern.boottime" 2> /dev/null)
h1=$(date +%s.%N | cut -c1-14)
gt=$(echo "$g" | sed -n 1p)
bt=$(echo "$g" | sed -n 's/.*sec = \([0-9]*\), usec = \([0-9]*\).*/\1 \2/p' | head -1 | awk '{ printf "%d.%06d", $1, $2 }')
[[ $gt =~ ^[0-9.]+$ && -n $bt ]] &&
  echo "$bt $(awk -v a="$h0" -v b="$h1" -v g="$gt" 'BEGIN { printf "%.3f", (a + b) / 2 - g }')" > "$R/kring-clock.txt"
: > "$R/kring.raw"
kring
KRINGPID=
if [[ ${EXCLUSIVE:-0} != 1 && -z ${PROBES:-} ]] || [[ ${CAPTURE:-0} == 1 ]]; then
  ( for f in /proc/$BASHPID/fd/*; do f=${f##*/}; (( f > 2 )) && eval "exec $f>&-"; done
    while kill -0 "$P" 2> /dev/null; do sleep "${KRING_S:-5}"; kring; done ) > /dev/null 2>&1 &
  KRINGPID=$!
fi
# The boot's timeline from QEMU's first line: the host's first present of a guest frame, and from
# the guest (its clock set against the host's over SSH, to a second) the kernel's boot and the
# start of WindowServer, loginwindow and Dock (the user's login).
if [[ -n $T_QEMU ]]; then
  fp=$(grep -m1 'the display shows IOSurface' "$B/launch.log" | cut -d' ' -f1)
  [[ -n $fp ]] && kv first_present_s "$(sub "$(stamp "$fp")" "$T_QEMU")"
  hnow=$(date +%s)
  g=$(G 'date +%s; sysctl -n kern.boottime; for p in WindowServer loginwindow Dock; do echo "$p $(ps -o etime= -p "$(pgrep -x $p | head -1)" 2>/dev/null)"; done' 2> /dev/null)
  gnow=$(echo "$g" | sed -n 1p)
  if [[ $gnow =~ ^[0-9]+$ ]]; then
    off=$(( hnow - gnow ))
    kb=$(echo "$g" | sed -n 's/.*sec = \([0-9]*\),.*/\1/p' | head -1)
    [[ -n $kb ]] && kv kernel_s "$(sub "$(( kb + off ))" "$T_QEMU")"
    for p in WindowServer loginwindow Dock; do
      e=$(echo "$g" | awk -v p=$p '$1 == p {print $2}')
      [[ -n $e ]] || continue
      el=$(echo "$e" | awk -F'[-:]' '{ n = NF; s = $n + 60 * $(n - 1); if (n >= 3) s += 3600 * $(n - 2); if (n >= 4) s += 86400 * $(n - 3); print s }')
      kv "$(echo "$p" | tr 'A-Z' 'a-z')_s" "$(sub "$(( gnow - el + off ))" "$T_QEMU")"
    done
  fi
fi

# The boot capture's end (CAPTURE=1): the readers stop CAPTURE_SETTLE_S seconds after the desktop
# came up. The guest's screenshots come back into capture/shots/ (shots-files.tsv: "FILE
# GUEST_MTIME NAME", the guest's clock; clock.txt its offset from the host's), and the kext's lines
# of the boot into capture/kext.log, whose scanout line says whether the reader's mode was the
# guest's. capture.txt sums it up.
if [[ ${CAPTURE:-0} == 1 ]]; then
  w=$(( T_DESK + ${CAPTURE_SETTLE_S:-60} - $(date +%s) ))
  (( w > 0 )) && sleep "$w"
  touch "$R/capture/stop"
  for p in "${CAPPIDS[@]}"; do
    for _ in $(seq 1 40); do kill -0 "$p" 2> /dev/null || break; sleep 0.5; done
  done
  h0=$(date +%s.%N | cut -c1-14)
  gt=$(G "perl -MTime::HiRes=time -e 'printf \"%.3f\\n\", time'" 2> /dev/null)
  h1=$(date +%s.%N | cut -c1-14)
  [[ $gt =~ ^[0-9.]+$ ]] &&
    echo "host_minus_guest_s $(awk -v a="$h0" -v b="$h1" -v g="$gt" 'BEGIN { printf "%.3f", (a + b) / 2 - g }')" > "$R/capture/clock.txt"
  Gs 'bash -s' > "$R/capture/shots-files.tsv" 2> /dev/null << 'SHOTS'
bt=$(sysctl -n kern.boottime | sed 's/.* sec = \([0-9]*\),.*/\1/')
rm -rf ~/lane-gate-shots && mkdir -p ~/lane-gate-shots
n=0
for f in ~/Desktop/*.png; do
  [ -f "$f" ] || continue
  [ "$(stat -f %m "$f")" -ge "$bt" ] || continue
  n=$((n + 1))
  b=$(printf '%03d.png' "$n")
  echo "$b $(stat -f %Fm "$f") ${f##*/}"
  mv "$f" ~/lane-gate-shots/"$b"
done
SHOTS
  Gcp "$GUEST_USER@127.0.0.1:lane-gate-shots" "$R/capture/shots" 2> /dev/null && G 'rm -rf ~/lane-gate-shots' 2> /dev/null
  kring
  kring_log
  cp "$R/kernel-rgfx.log" "$R/capture/kext.log"
  cp "$R/kring-clock.txt" "$R/capture/" 2> /dev/null
  line=$(grep -m1 -oE 'scanout [0-9]+x[0-9]+ at [0-9.]+ Hz, stride [0-9]+' "$R/capture/kext.log")
  mode=${CAPTURE_MODE:-3840x2400}
  stride=$(( (${mode%x*} * 4 + 255) / 256 * 256 ))
  if [[ -z $line ]]; then mok=unknown
  elif [[ $line == "scanout $mode at "*", stride $stride" ]]; then mok=1
  else mok=0
  fi
  {
    echo "dir $R/capture"
    echo "scanout $(cat "$R/capture/scanout.done" 2> /dev/null)"
    echo "the scanout's frames are the screen only up to WindowServer's first present (until_first_present); later the display shows IOSurfaces, which the guest's screenshots (shots/) show"
    echo "shots $(cat "$R/capture/shots.done" 2> /dev/null) files=$(grep -c . "$R/capture/shots-files.tsv")"
    echo "mode $mode stride $stride; the kext's first: ${line:-none}; as assumed: $mok"
    cat "$R/capture/clock.txt" 2> /dev/null
  } > "$R/capture.txt"
  kv capture_distinct "$(sed -n 's/.*distinct=\([0-9]*\).*/\1/p' "$R/capture/scanout.done" 2> /dev/null)"
  kv capture_shots "$(grep -c . "$R/capture/shots-files.tsv")"
  kv capture_mode_ok "$mok"
  log "boot capture: $(tr '\n' ';' < "$R/capture.txt")"
fi

# Right after login the guest's SSH can miss its banner timeout while the desktop's apps start: a
# copy is tried three times, 15 s apart. After a failed try, where it stopped: whether QEMU's main
# loop (which also runs its user network) answers QMP, and whether the guest's SSH sends its banner
# through the forwarded port.
ssh_state() {
  local q c
  q=$(printf '%s\n' '{"execute":"qmp_capabilities"}' '{"execute":"query-status"}' |
    timeout 5 socat - "UNIX-CONNECT:$B/qmp.sock" 2> /dev/null | grep -o '"status": *"[a-z-]*"' | head -1)
  # The kernel completes a connect to QEMU's forwarded port whatever QEMU does: a banner that does
  # not come is the guest's where QMP answers (the main loop, which runs the user network, is up).
  c=$(timeout 10 python3 -I -c '
import socket, sys
s = socket.create_connection(("127.0.0.1", int(sys.argv[1])), 3)
s.settimeout(5)
try:
    b = s.recv(40)
except OSError:
    b = b""
print("connected, " + ("banner " + b.decode(errors="replace").strip() if b else "no banner in 5 s"))' "$PORT" 2> /dev/null)
  echo "QMP ${q:+answers ($q)}${q:-silent for 5 s}; TCP to the SSH port: ${c:-connect failed}"
}
# Where a copy failed for good: the serial port's last lines, and the longest gaps between the
# guest's library lookups (its processes launching) in the last three minutes of QEMU's log.
copy_failed_state() {
  log "serial tail: $(tail -3 "$B/serial.log" 2> /dev/null | tr -dc '[:print:]\n' | tr '\n' '|')"
  local since; since=$(date -d '-3 min' +%Y-%m-%dT%H:%M:%S)
  log "library lookup gaps since $since: $(grep 'library lookup' "$B/launch.log" | cut -c1-23 | awk -v s="$since" '$1 >= s' |
    while read -r t; do date -d "$t" +%s.%N; done | awk 'NR > 1 { g = $1 - p; if (g > 5) printf "%.0f s at %s, ", g, strftime("%T", p) } { p = $1 }
      END { if (NR == 0) printf "no lookups"; else printf "last at %s", strftime("%T", p) }')"
}
copied=0
for try in 1 2 3; do
  G "rm -rf ~/lane-gate" && Gcp "$PAYLOAD" "$GUEST_USER@127.0.0.1:lane-gate" &&
    { [[ -n ${PROBES:-} ]] || Gcp "$CORPUS" "$GUEST_USER@127.0.0.1:lane-gate/corpus"; } && { copied=1; break; }
  log "copying the payload into the guest failed (try $try): $(ssh_state)"
  sleep 15
done
((copied)) || { copy_failed_state; log "BOOT_FAIL copying the payload into the guest"; exit 6; }
kv payload_tries "$try"
log "payload copied"
# The kext's load address, for a hang snapshot's registers and stacks (msi-hang.py) to be read
# against.
G "kmutil showloaded --list-only 2> /dev/null | grep -i -E 'reims|rgfx'; true" > "$R/kext-load.txt" 2>&1 || true
log "kext loaded: $(tr -s ' ' < "$R/kext-load.txt" | tr '\n' ';' | cut -c1-300)"

# RgfxGate installs the bundle at the first boot, and the accelerator registers for it the same
# boot: a Metal device that is not ours here is a failure (the run goes on after one reboot).
dev=$(G "perl -e 'alarm 30; exec @ARGV' ~/lane-gate/bench -l 2>/dev/null")
if [[ ${METAL_CHECK:-1} == 0 ]]; then
  log "the guest's Metal device, not checked (METAL_CHECK=0): $dev"
elif [[ $dev == *"bench: device"* && $dev != *"Apple Paravirtual"* ]]; then
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

if [[ -n $GL ]]; then
  ( closefds; exec "$GL" -d "$LOAD" ) > "$R/gpuload.log" 2>&1 &
  LOADPID=$!
  for _ in $(seq 1 40); do grep -q '^gpuload: .* of .* ms' "$R/gpuload.log" && break; sleep 0.5; done
  grep -q '^gpuload: .* of .* ms' "$R/gpuload.log" ||
    { log "BOOT_FAIL gpuload did not start: $(tail -2 "$R/gpuload.log" | tr '\n' ' ')"; exit 8; }
  log "fixed GPU load: $(head -1 "$R/gpuload.log")"
fi
if command -v nvidia-smi > /dev/null; then
  ( closefds; exec > "$R/gpu-clocks.txt" 2> /dev/null
    timeout "$(left)" nvidia-smi --query-gpu=pstate,clocks.gr,clocks.mem,power.draw --format=csv,noheader,nounits -lms 100 |
      while IFS=', ' read -r ps gr mem pw; do echo "$(date +%s.%N | cut -c1-14) $ps $gr $mem $pw"; done ) &
  CLOCKPID=$!
fi
# The host's load beside the measurements (other work on the host shares its CPU and memory).
kv load_start "$(cut -d' ' -f1 /proc/loadavg)"
# The kext's texture-create stalls ("RgfxAccelerator: IOSurface N: its previous block is cached
# still, its release unanswered: it is not cached", about 2 s each), from its lines in the ring
# (kernel-rgfx.log): kext-stalls.txt "HOST_EPOCH S_AFTER_DESKTOP" a hit, the uptime set on the
# host's clock (kring-clock.txt). block_stalls counts them, block_stalls_early those in the 60 s
# after the desktop came up; kring_first_s says from when the ring's lines reach (its first line's
# uptime: a boot's earlier lines had left the ring by its first read).
# A probe boot counts them too, at its end.
stalls() {
  [[ -n ${KRINGPID:-} ]] && { kill "$KRINGPID" 2> /dev/null; KRINGPID=; }
  kring
  kring_log
  if [[ -s $R/kring-clock.txt ]]; then
    read -r bt off < "$R/kring-clock.txt"
    sed -n 's/^\[\([0-9.]*\)\]: RgfxAccelerator: IOSurface [0-9]*: its previous block is cached still.*/\1/p' "$R/kernel-rgfx.log" |
      awk -v bt="$bt" -v off="$off" -v d="${T_DESK:-0}" '{ t = bt + $1 + off; printf "%.3f %.1f\n", t, t - d }' > "$R/kext-stalls.txt"
    kv kring_lines "$(grep -c . "$R/kernel-rgfx.log")"
    kv kring_first_s "$(sed -n '1s/^\[\([0-9.]*\)\].*/\1/p' "$R/kernel-rgfx.log")"
    kv block_stalls "$(grep -c . "$R/kext-stalls.txt")"
    kv block_stalls_early "$(awk '$2 >= 0 && $2 <= 60' "$R/kext-stalls.txt" | grep -c .)"
    [[ -d $R/capture ]] && cp "$R/kext-stalls.txt" "$R/capture/"
  else
    kv block_stalls unread
  fi
}
# The app load (APPLOAD=1, msi-appload.sh): rgk-ctx-slots' busy desktop on this boot, after the
# measurements (a probe boot's: before its probes, which then run on that desktop), as it leaves
# every app open; where the hold has room for it (about 6 minutes). appload.txt holds its numbers
# (msi-appload.sh), appload/ the rest.
app_load() {
  if (( $(left) > ${APPLOAD_NEED_S:-480} )) && [[ -f $(dirname "$0")/appload-tools/analyze-load.py ]]; then
    log "app load"
    PORT=$PORT GUEST_USER=$GUEST_USER GUEST_PASS=$GUEST_PASS B=$B TOOLS="$(dirname "$0")/appload-tools" \
      timeout "$(( $(left) - 60 ))" bash "$(dirname "$0")/msi-appload.sh" "$R/appload" > "$R/appload.log" 2>&1
    kv appload_rc $?
    cp -f "$R/appload/appload.txt" "$R/appload.txt" 2> /dev/null
    log "app load done: $(tr '\n' ' ' < "$R/appload.txt" 2> /dev/null)"
  else
    kv appload_rc "skipped: $(left) s of the hold left, or no tools"
    log "app load skipped: $(left) s of the hold left"
  fi
}
# A probe boot: the round's probes back to back (msi-probes.sh), and nothing of the gate's run.
if [[ -n ${PROBES:-} ]]; then
  [[ ${APPLOAD:-0} == 1 ]] && app_load
  log "probes: $(ls "$PROBES" | tr '\n' ' ')$(left) s left"
  PROBES=$PROBES PROBE_RESULTS=$PROBE_RESULTS PROBE_TIP=$PROBE_TIP QEMU_PID=$P DEADLINE=$(( T0 + HOLD_S - 20 )) B=$B R=$R \
    HANG_CATCH_PID=$HCPID \
    PORT=$PORT GUEST_USER=$GUEST_USER GUEST_PASS=$GUEST_PASS bash "$(dirname "$0")/msi-probes.sh" 2>&1 | tee -a "$R/boot.log"
  kv load_end "$(cut -d' ' -f1 /proc/loadavg)"
  log "probes done"
  [[ ${CAPTURE:-0} == 1 ]] && stalls
  ended=1
  exit 0
fi
log "guest run: ${PHASES:-all phases}, $(left) s left"
ts() { perl -MTime::HiRes=time -ne '$| = 1; printf "%.3f %s", time, $_'; }
guest_rc=0
guest_run() {  # PHASE...: guest-run.sh's phases, its lines after the host's time into guest.log
  timeout "$(( $(left) - 40 ))" sshpass -p "$GUEST_PASS" ssh -n -p "$PORT" "${SOPTS[@]}" "$GUEST_USER@127.0.0.1" \
    "GATE_PASS='$GUEST_PASS' GATE_PRESENT_S=${PRESENT_S:-10} GATE_RETRY='${RETRY:-}' GATE_RETRY_S=${RETRY_S:-0} GATE_FINAL_CMD_B64='${FINAL_CMD_B64:-}' bash ~/lane-gate/guest-run.sh ~/lane-gate $*" 2>&1 |
    ts >> "$R/guest.log"
  local rc=${PIPESTATUS[0]}
  (( guest_rc == 0 )) && guest_rc=$rc
}
# The phases up to bench, then the GPU-bound cases with no repetition at the GPU's high clock
# again (at most two rounds, each while the hold has 480 s left, and the round given what it
# leaves above 300 s: the phases after it, the drags and the screenshot), then the rest.
read -ra ph <<< "${PHASES:-info idle bench present sweep decode webgl final}"
first=() rest=() benched=0
for p in "${ph[@]}"; do
  if ((benched)); then rest+=("$p"); else first+=("$p"); [[ $p == bench ]] && benched=1; fi
done
: > "$R/guest.log"
guest_run "${first[@]}"
for round in 1 2; do
  ((benched)) && [[ -n $CLOCKPID ]] && (( $(left) > 480 )) || break
  RETRY=$(python3 -I "$(dirname "$0")/reps.py" "$R/guest.log" "$R/gpu-clocks.txt")
  [[ -n $RETRY ]] || break
  log "no repetition at the GPU's high clock: $RETRY; running them again (round $round)"
  kv bench_retry_$round "${RETRY// /,}"
  RETRY_S=$(( $(left) - 300 )) guest_run benchretry
done
((${#rest[@]})) && guest_run "${rest[@]}"
kv guest_rc "$guest_rc"
log "guest run done"

# The window drags (msi-drag.py): a plain window and a glass one dragged, a window resized live and
# Mission Control opened and closed, each for DRAG_S seconds over QMP while the guest tracks the
# window, with the host's display updates and CPU in that span.
if [[ ${DRAG:-1} == 1 ]] && (( $(left) > 150 )); then
  log "window drags"
  PORT=$PORT GUEST_USER=$GUEST_USER GUEST_PASS=$GUEST_PASS DRAG_S=${DRAG_S:-10} \
    timeout "$(( $(left) - 60 ))" python3 -I "$(dirname "$0")/msi-drag.py" "$B" "$R" >> "$R/drag.log" 2>&1
  kv drag_rc $?
  log "window drags done"
fi

# The app load (app_load, above), after the measurements.
[[ ${APPLOAD:-0} == 1 ]] && app_load
kv load_end "$(cut -d' ' -f1 /proc/loadavg)"
stop_load
stop_clocks
display_state display_end
reaper_lines

stalls

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

# The gate's own probes last (POST_PROBES, lane-gate.sh --post-probes): after everything above,
# on the app load's desktop, as a probe round runs them (msi-probes.sh), into post-probes/. They
# wait at most POST_SETTLE_S (default 60) for the host's load to settle.
if [[ -n ${POST_PROBES:-} && -d ${POST_PROBES:-} ]]; then
  log "post probes: $(ls "$POST_PROBES" | tr '\n' ' ')$(left) s left"
  PROBES=$POST_PROBES PROBE_RESULTS=$R/post-probes PROBE_TIP=$POST_PROBE_TIP QEMU_PID=$P \
    DEADLINE=$(( T0 + HOLD_S - 20 )) B=$B R=$R HANG_CATCH_PID=$HCPID SETTLE_S=${POST_SETTLE_S:-60} \
    PORT=$PORT GUEST_USER=$GUEST_USER GUEST_PASS=$GUEST_PASS bash "$(dirname "$0")/msi-probes.sh" 2>&1 | tee -a "$R/boot.log"
  kv post_probes_rc "${PIPESTATUS[0]}"
  log "post probes done"
fi
ended=1
exit 0
