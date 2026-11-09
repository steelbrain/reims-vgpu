#!/bin/bash
# Boots one arm64 macOS guest on Ambrosia with a configuration bundle, ours by default, and checks
# the boot as Ambrosia's done list has it: one line a check, then the verdict.
#
#   firmware/ambrosia/check-boot.sh -f AMBROSIA [-q QEMU] [-t] [-b BUNDLE] [-T TOOLS] [-p PORT]
#                                   [-o OUT] [-e ECID] VM_DIR
#
#   VM_DIR   the guest: disk.img, aux.img.trimmed (or aux.img, which a trimmed copy is made of),
#            and vm.toml, whose ecid is the guest's ECID unless -e gives it. It is never booted
#            or written: the boot runs on clones of its disks made in OUT, by cp -c on APFS or
#            cp --reflink=always on Linux, and removed at the end. Where neither works the script
#            stops.
#   -f FILE  Ambrosia's image, its built/ambrosia.bin
#   -q FILE  our QEMU's qemu-system-aarch64 with reims-graphics-pci (docs/qemu.md, "arm64 guests:
#            the device on Ambrosia"); default $QEMU
#   -t       TCG, for a Linux member, in place of HVF (docs/qemu.md, "arm64 guests: the boot
#            check of our configuration", says what such a run needs)
#   -b FILE  the configuration bundle; default build/Config.cpio (make config-bundle)
#   -T DIR   the guest probes, bench and present_probe; default build/guest-tools (make
#            guest-tools, on a Mac)
#   -p PORT  the host port forwarded to the guest's SSH; default 10090
#   -o DIR   where the logs go, a new directory; default one made under ${TMPDIR:-/tmp}
#   -e ECID  the guest's ECID, in decimal or 0x hex
#
# The guest runs with 8 GiB and 4 CPUs, as Ambrosia's tested boots do, with QEMU's
# REIMS_VGPU_WINDOW=0 (no window on the host) and REIMS_VGPU_LOG=info, as the lane gate runs it.
# Under TCG, for a Linux member, its RAM is a shared memfd, as reims-ctl's VMs on Linux have it. The
# device maps the memfd-backed main RAM in place and leaves out vmapple's firmware and configuration
# regions, which have no file behind them and hold no IOSurface, so every surface still maps in place
# (docs/qemu.md, "arm64 guests: the boot check of our configuration").
# Nothing is typed or clicked in the guest. The checks log in over SSH as GUEST_USER, password
# GUEST_PASS (default aneesiqbal, as every owner VM), and copy the probes into ~/ambrosia-check.
#
# The checks, each PASS or FAIL with what it found:
#   firmware   Ambrosia entered the kernel. Where it stops the boot instead ("bad configuration:
#              ...", "boot failed: ...", docs/running.md in Ambrosia), this fails with its line,
#              QEMU is stopped, and the guest's checks fail as not run.
#   console    Ambrosia logged the module's early console.
#   ssh        the guest took an SSH login within 90 s of QEMU's start under HVF, 300 s under TCG.
#   sip        `csrutil status` says enabled (the lane gate's sip).
#   boot-args  `sysctl kern.bootargs` is empty.
#   kexts      `kmutil showloaded` lists Graft and our two kexts (stage-config.py's identifiers),
#              as the lane gate's kexts lists ReimsGraphics.
#   windowserver-metal
#              WindowServer composites with Metal on our device, as the lane gate's guest checks
#              and boot read it: the desktop came up (WindowServer, Dock and Finder) within 60 s
#              of SSH under HVF, 180 s under TCG; `bench -l` lists a Metal device, not Apple's
#              paravirtual one; WindowServer kept its pid to the end; and the host's log says the
#              display shows an IOSurface the guest presented.
#   metal-app  present_probe, an on-screen Metal app, ran full-screen for 10 s in the user's
#              session, as the lane gate's present phase: OK, frames presented, and at most a
#              tenth dropped or unanswered (report.py's bar for the correctness boot).
#   no-fatal   no `reims-vgpu: fatal` line in QEMU's log or the serial log.
#
# OUT keeps serial.log, qemu.log, screen.png (QMP's screendump of the display at the end, before
# QEMU is stopped), guest.log (what the guest's commands printed) and verdict.txt.
# Exit 0 when every check passed, 1 when one failed, 2 when the boot could not be set up.
set -uo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
die() { echo "check-boot: $*" >&2; exit 2; }
usage() { sed -n '/^#   firmware\/ambrosia\/check-boot.sh/,/^#   -e ECID/p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }

firmware= qemu=${QEMU:-} tcg=0 bundle=$here/build/Config.cpio tools=$here/build/guest-tools
port=10090 out= ecid=
while getopts f:q:tb:T:p:o:e:h opt; do
  case $opt in
  f) firmware=$OPTARG ;;
  q) qemu=$OPTARG ;;
  t) tcg=1 ;;
  b) bundle=$OPTARG ;;
  T) tools=$OPTARG ;;
  p) port=$OPTARG ;;
  o) out=$OPTARG ;;
  e) ecid=$OPTARG ;;
  *) usage ;;
  esac
done
shift $((OPTIND - 1))
[[ $# -eq 1 ]] || usage
vm=$1
GUEST_USER=${GUEST_USER:-aneesiqbal} GUEST_PASS=${GUEST_PASS:-aneesiqbal}

ram=8G
if ((tcg)); then
  accel=(-accel tcg,thread=multi) limit=300 desktop_wait=180
  machine=",memory-backend=ram,dump-guest-core=off"
  mem=(-m "$ram" -object "memory-backend-memfd,id=ram,size=$ram,share=on")
else
  accel=(-accel hvf,kernel-irqchip=on) limit=90 desktop_wait=60
  machine= mem=(-m "$ram")
fi

[[ -n $firmware && -f $firmware ]] || die "no Ambrosia image: give its built/ambrosia.bin with -f"
[[ -n $qemu && -x $qemu ]] || die "no QEMU: give our qemu-system-aarch64 with -q, or set QEMU"
[[ -f $bundle ]] || die "no bundle $bundle: make config-bundle, or give one with -b"
[[ -x $tools/bench && -x $tools/present_probe ]] ||
  die "no guest probes in $tools: make guest-tools on a Mac (copy build/guest-tools to a Linux member), or give them with -T"
[[ -f $vm/disk.img ]] || die "no disk.img in $vm"
[[ -f $vm/aux.img.trimmed || -f $vm/aux.img ]] || die "no aux.img.trimmed or aux.img in $vm"
for t in sshpass python3 perl timeout; do command -v "$t" > /dev/null || die "needs $t"; done
if [[ -z $ecid ]]; then
  ecid=$(sed -n 's/^ecid *= *"\{0,1\}\([0-9a-fA-Fx]*\)"\{0,1\} *$/\1/p' "$vm/vm.toml" 2> /dev/null | head -1)
  [[ -n $ecid ]] || die "no ECID: $vm/vm.toml has no ecid; give it with -e"
fi
python3 -I -c 'import socket, sys; s = socket.socket(); s.bind(("127.0.0.1", int(sys.argv[1])))' "$port" 2> /dev/null ||
  die "port $port is in use: give another with -p"
if [[ -z $out ]]; then
  out=$(mktemp -d "${TMPDIR:-/tmp}/check-boot.XXXXXX") || die "can't make a directory for the logs"
else
  mkdir "$out" || die "can't make $out (it must be new)"
fi
out=$(cd "$out" && pwd)
qmp_sock=$out/qmp.sock
((${#qmp_sock} < 100)) || die "$qmp_sock is too long for a socket's path: give a shorter -o"

# --- QMP, SSH and cleaning up -------------------------------------------------------------------

now() { perl -MTime::HiRes=time -e 'printf "%.1f\n", time'; }
since() { perl -e 'printf "%.1f\n", $ARGV[0] - $ARGV[1]' "$(now)" "$1"; }
# A QMP command, given as JSON: prints the answer, and fails where QEMU answered an error.
qmp() {
  python3 -I - "$qmp_sock" "$1" << 'PY'
import json, socket, sys
s = socket.socket(socket.AF_UNIX)
s.settimeout(30)
s.connect(sys.argv[1])
f = s.makefile("rw")
f.readline()
def run(cmd):
    f.write(json.dumps(cmd) + "\n")
    f.flush()
    while True:
        line = f.readline()
        if not line:
            return {"return": "closed"}
        r = json.loads(line)
        if "return" in r or "error" in r:
            return r
run({"execute": "qmp_capabilities"})
r = run(json.loads(sys.argv[2]))
print(json.dumps(r))
sys.exit(0 if "return" in r else 1)
PY
}

# The guest's commands share one SSH connection, the first login's (ControlMaster), so the
# password is given once; the socket's name is a hash of the port, short enough for a socket.
SOPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PreferredAuthentications=password
  -o PubkeyAuthentication=no -o ConnectTimeout=5 -o LogLevel=ERROR
  -o ControlMaster=auto -o "ControlPath=/tmp/check-boot-%C" -o ControlPersist=yes)
# A command in the guest, under a limit of SECONDS: prints its output, also kept in guest.log, and
# returns its status. A login that fails (sshpass's 5, a refused password; ssh's 255) is tried
# twice more, 5 s apart, as the lane gate retries its copies.
G() {
  local s=$1 try rc
  shift
  for try in 1 2 3; do
    echo "\$ $*" >> "$out/guest.log"
    timeout "$s" sshpass -p "$GUEST_PASS" ssh -n -p "$port" "${SOPTS[@]}" "$GUEST_USER@127.0.0.1" "$@" > "$out/g.out" 2>&1
    rc=$?
    cat "$out/g.out" >> "$out/guest.log"
    [[ $rc == 5 || $rc == 255 ]] && ((try < 3)) || break
    echo "(the login failed, $rc: again in 5 s)" >> "$out/guest.log"
    sleep 5
  done
  cat "$out/g.out"
  rm -f "$out/g.out"
  return "$rc"
}

qpid=
stop_qemu() {
  [[ -n $qpid ]] || return 0
  if kill -0 "$qpid" 2> /dev/null; then
    qmp '{"execute": "quit"}' > /dev/null 2>&1
    for _ in $(seq 1 20); do kill -0 "$qpid" 2> /dev/null || break; sleep 0.5; done
    kill -TERM "$qpid" 2> /dev/null && sleep 2
    kill -KILL "$qpid" 2> /dev/null
  fi
  wait "$qpid" 2> /dev/null
  qpid=
}
cleanup() {
  ssh -p "$port" "${SOPTS[@]}" -O exit "$GUEST_USER@127.0.0.1" > /dev/null 2>&1
  stop_qemu
  rm -f "$clone/disk.img" "$clone/aux.img.trimmed" "$qmp_sock"
  rmdir "$clone" 2> /dev/null
}
# bash 3.2, macOS's, exits 0 from a set -u error where an EXIT trap is set, with $? 0 in the
# trap too, so an aborted run would read as a pass (docs/lessons.md): the trap fails a run
# that exits 0 short of its end, which sets `ended`.
ended=0
trap 'rc=$?; cleanup; [[ $rc != 0 || $ended == 1 ]] || rc=1; exit $rc' EXIT
trap 'exit 2' INT TERM

# --- the clones ---------------------------------------------------------------------------------

clone=$out/clone
mkdir "$clone" || die "can't make $clone"
case $(uname -s) in
Darwin) cpc=(cp -c) ;;
*) cpc=(cp --reflink=always) ;;
esac
"${cpc[@]}" "$vm/disk.img" "$clone/disk.img" ||
  die "can't clone $vm/disk.img into $clone with ${cpc[*]}: the boot runs only on a clone, so give -o a directory on the guest's volume"
if [[ -f $vm/aux.img.trimmed ]]; then
  "${cpc[@]}" "$vm/aux.img.trimmed" "$clone/aux.img.trimmed" || die "can't clone $vm/aux.img.trimmed into $clone"
else
  dd if="$vm/aux.img" of="$clone/aux.img.trimmed" bs=16384 skip=1 2> /dev/null || die "can't trim $vm/aux.img"
fi

# --- the boot -----------------------------------------------------------------------------------

t0=$(now)
REIMS_VGPU_WINDOW=0 REIMS_VGPU_LOG=info "$qemu" "${accel[@]}" \
  -M "vmapple,uuid=$ecid$machine" "${mem[@]}" -smp 4 -bios "$firmware" \
  -drive "file=$clone/aux.img.trimmed,if=pflash,format=raw" \
  -drive "file=$clone/disk.img,if=pflash,format=raw" \
  -drive "file=$clone/aux.img.trimmed,if=none,id=aux,format=raw" \
  -drive "file=$clone/disk.img,if=none,id=root,format=raw" \
  -device vmapple-virtio-blk-pci,variant=aux,drive=aux \
  -device vmapple-virtio-blk-pci,variant=root,drive=root \
  -netdev "user,id=net0,ipv6=off,hostfwd=tcp:127.0.0.1:$port-:22" \
  -device virtio-net-pci,netdev=net0 \
  -device reims-graphics-pci \
  -device "loader,file=$bundle,addr=0x120000000,force-raw=on" \
  -display none -serial "file:$out/serial.log" \
  -qmp "unix:$qmp_sock,server=on,wait=off" \
  > "$out/qemu.log" 2>&1 < /dev/null &
qpid=$!

# --- the checks ---------------------------------------------------------------------------------

names=(firmware console ssh sip boot-args kexts windowserver-metal metal-app no-fatal)
# A check's verdict, PASS or FAIL, and what it found; in plain variables, for macOS's bash 3.2.
verdict() { printf -v "st_${1//-/_}" '%s' "$2"; printf -v "dt_${1//-/_}" '%s' "$3"; }
state() { local v="st_${1//-/_}"; echo "${!v:-}"; }
detail() { local v="dt_${1//-/_}"; echo "${!v:-}"; }
serial() { [[ -f $out/serial.log ]] && tr -d '\r' < "$out/serial.log"; }

# The boot, up to the SSH limit: Ambrosia's stop line, QEMU's exit, or the guest's login.
stop= ssh_s=
while :; do
  stop=$(serial | grep -a -m1 -E "^ambrosia: (bad configuration|bad graphics driver|can't read the boot objects|boot failed|exception [0-9]|panic:)")
  [[ -n $stop ]] && break
  kill -0 "$qpid" 2> /dev/null || break
  if serial | grep -a -q '^ambrosia: entering the kernel' &&
    timeout 10 sshpass -p "$GUEST_PASS" ssh -n -p "$port" "${SOPTS[@]}" "$GUEST_USER@127.0.0.1" true 2> /dev/null; then
    ssh_s=$(since "$t0")
    break
  fi
  perl -e "exit !($(since "$t0") > $limit)" && break
  sleep 1
done
# A stop line may come just after the loop's last look.
[[ -n $stop ]] || stop=$(serial | grep -a -m1 -E "^ambrosia: (bad configuration|bad graphics driver|can't read the boot objects|boot failed|exception [0-9]|panic:)")

entered=$(serial | grep -a -m1 '^ambrosia: entering the kernel')
if [[ -n $stop ]]; then
  hint=
  [[ $stop == *KextsNotPlaced* ]] && hint=" (this Ambrosia reads Kernel > Add but places no kext yet)"
  verdict firmware FAIL "Ambrosia stopped the boot: \"${stop#ambrosia: }\"$hint"
elif ! kill -0 "$qpid" 2> /dev/null; then
  verdict firmware FAIL "QEMU exited: $(tail -3 "$out/qemu.log" | tr '\n' ' ')"
elif [[ -n $entered ]]; then
  verdict firmware PASS "${entered#ambrosia: } ($(serial | grep -a -m1 '^ambrosia: Ambrosia ' | sed 's/^ambrosia: //'))"
else
  verdict firmware FAIL "Ambrosia did not enter the kernel in $limit s; its last line: $(serial | grep -a '^ambrosia: ' | tail -1)"
fi

console=$(serial | grep -a -m1 -E '^ambrosia: .*: early console, ')
if [[ -n $console ]]; then
  verdict console PASS "${console#ambrosia: }; $(serial | grep -a -m1 '^ambrosia: reims-graphics at ' | sed 's/^ambrosia: //')"
else
  verdict console FAIL "no early console; the module's lines: $(serial | grep -a -E '^ambrosia: (graphics driver|reims-graphics)' | sed 's/^ambrosia: //' | tr '\n' ';')"
fi

if [[ -n $ssh_s ]]; then
  verdict ssh PASS "a login $ssh_s s after QEMU started (limit $limit s)"
  G 20 'sw_vers -productVersion; sw_vers -buildVersion' > /dev/null

  csr=$(G 30 csrutil status)
  case $csr in
  *"enabled."*) verdict sip PASS "$csr" ;;
  *) verdict sip FAIL "csrutil status: ${csr:-no answer}" ;;
  esac

  if args=$(G 20 sysctl -n kern.bootargs); then
    if [[ -z ${args//[[:space:]]/} ]]; then verdict boot-args PASS "kern.bootargs is empty"
    else verdict boot-args FAIL "kern.bootargs is \"$args\""; fi
  else
    verdict boot-args FAIL "sysctl failed: $args"
  fi

  if shown=$(G 120 kmutil showloaded); then
    loaded=$(echo "$shown" | grep -o -E 'com\.steelbrain\.[A-Za-z.]+' | sort -u | tr '\n' ' ')
    missing=
    for id in $(python3 -I -B "$here/tools/stage-config.py" --identifiers); do
      [[ " $loaded " == *" $id "* ]] || missing+="$id "
    done
    if [[ -z $missing ]]; then
      verdict kexts PASS "loaded: ${loaded% }"
    else
      added=$(serial | grep -a -m1 -E '^ambrosia: configuration: ' | grep -o -E '[0-9]+ kexts')
      verdict kexts FAIL "not loaded: ${missing% }; loaded of ours: ${loaded:-none}; the bundle adds ${added:-no kexts}"
    fi
  else
    verdict kexts FAIL "kmutil showloaded failed: $(echo "$shown" | tail -2 | tr '\n' ' ')"
  fi

  # The desktop, as the lane gate waits for it.
  desk= t_desk=$(now)
  while :; do
    d=$(G 20 'for p in WindowServer Dock Finder; do n=$(pgrep -x $p | head -1); echo ${n:-none}; done') &&
      desk=$(echo $d)
    set -- $desk
    [[ $# -eq 3 && " $desk" != *" none "* ]] && break
    perl -e "exit !($(since "$t_desk") > $desktop_wait)" && break
    sleep 5
  done
  set -- $desk none none none
  desk_say="WindowServer $1, Dock $2, Finder $3" ws_pid=${1#none}
  if [[ " $desk" != *" none "* ]]; then desk_ok=1
  else desk_ok=0 desk_say="no desktop in $desktop_wait s: $desk_say"; fi

  # The probes, copied over the shared connection.
  G 30 'mkdir -p ~/ambrosia-check' > /dev/null
  for t in bench present_probe; do
    timeout 60 sshpass -p "$GUEST_PASS" ssh -p "$port" "${SOPTS[@]}" "$GUEST_USER@127.0.0.1" \
      "cat > ~/ambrosia-check/$t && chmod +x ~/ambrosia-check/$t" < "$tools/$t" >> "$out/guest.log" 2>&1
  done
  if dev=$(G 40 "perl -e 'alarm 30; exec @ARGV' ~/ambrosia-check/bench -l"); then
    dev=$(echo "$dev" | grep -v '^[0-9-]* [0-9:.]* .*RGFX' | head -5 | tr '\n' ' ' | sed 's/ *$//')
    case $dev in
    *Apple\ Paravirtual*) dev_ok=0 dev_say="Apple's paravirtual device: $dev" ;;
    *bench:\ device*) dev_ok=1 dev_say=$dev ;;
    '') dev_ok=0 dev_say="no Metal device: bench -l lists none" ;;
    *) dev_ok=0 dev_say="no Metal device: bench -l printed \"$dev\"" ;;
    esac
  else
    dev_ok=0 dev_say="bench -l failed: $(echo "$dev" | tail -2 | tr '\n' ' ')"
  fi

  if ((dev_ok && desk_ok)); then
    probe=$(G 60 "echo '$GUEST_PASS' | sudo -S -p '' launchctl asuser \$(id -u) sudo -u \$(id -un) perl -e 'alarm shift @ARGV; exec @ARGV' 40 \$HOME/ambrosia-check/present_probe -t 10" |
      grep '^present_probe:' | head -1)
    # report.py's bar for the correctness boot: frames presented, and at most a tenth lost, a
    # loss being a drop or an answer still missing beyond the last 3.
    if [[ $probe == "present_probe: OK "* ]] && perl -e '
        my %k = map { /^(\w+)=([-0-9.]+)$/ ? ($1, $2) : () } split " ", $ARGV[0];
        my ($p, $l) = ($k{presented} // 0, ($k{dropped} // 0) + (($k{unanswered} // 0) > 3 ? $k{unanswered} - 3 : 0));
        exit !($p > 0 && $l <= 0.1 * ($p + $l))' "$probe"; then
      verdict metal-app PASS "${probe#present_probe: }"
    else
      verdict metal-app FAIL "${probe:-present_probe printed no result line}"
    fi
  elif ((!dev_ok)); then
    verdict metal-app FAIL "not run: $dev_say"
  else
    verdict metal-app FAIL "not run: $desk_say"
  fi
  ws_end=$(G 20 'pgrep -x WindowServer | head -1') || ws_end="unread: $(echo "$ws_end" | tail -1)"
elif [[ $(state firmware) == FAIL ]]; then
  for n in ssh sip boot-args kexts windowserver-metal metal-app; do verdict "$n" FAIL "not run: the firmware check failed"; done
else
  verdict ssh FAIL "no SSH login in $limit s"
  for n in sip boot-args kexts windowserver-metal metal-app; do verdict "$n" FAIL "not run: no SSH login"; done
fi
# What the display shows at the end: the desktop, or the console's log and the kernel's progress.
qmp "{\"execute\": \"screendump\", \"arguments\": {\"filename\": \"$out/screen.png\", \"format\": \"png\"}}" > /dev/null 2>&1 ||
  qmp "{\"execute\": \"screendump\", \"arguments\": {\"filename\": \"$out/screen.ppm\"}}" > /dev/null 2>&1
stop_qemu

# What the host's log says of the display and of deaths, once QEMU has ended.
presents=$(grep -c -E 'the display shows IOSurface [1-9]' "$out/qemu.log")
fatal_q=$(grep -c 'reims-vgpu: fatal' "$out/qemu.log")
fatal_s=$(grep -a -c 'reims-vgpu: fatal' "$out/serial.log" 2> /dev/null)
fatal_s=${fatal_s:-0}
if [[ -n $ssh_s ]]; then
  ws=() ok=1
  ws+=("$desk_say"); ((desk_ok)) || ok=0
  if ((dev_ok)); then ws+=("Metal: $dev_say"); else ws+=("$dev_say") ok=0; fi
  if [[ -n $ws_pid && $ws_end == "$ws_pid" ]]; then ws+=("WindowServer kept pid $ws_pid")
  elif [[ -n $ws_pid ]]; then ws+=("WindowServer's pid was $ws_pid, then ${ws_end:-none}") ok=0; fi
  if ((presents > 0)); then ws+=("the host's display switched to an IOSurface the guest presented $presents times")
  else ws+=("the host's log shows no IOSurface the guest presented") ok=0; fi
  verdict windowserver-metal "$( ((ok)) && echo PASS || echo FAIL)" "$(IFS=';'; echo "${ws[*]}" | sed 's/;/; /g')"
fi
if ((fatal_q + fatal_s == 0)); then
  verdict no-fatal PASS "no 'reims-vgpu: fatal' line in QEMU's log or the serial log"
else
  verdict no-fatal FAIL "$fatal_q 'reims-vgpu: fatal' lines in QEMU's log, $fatal_s in the serial log: $(grep -m1 'reims-vgpu: fatal' "$out/qemu.log")"
fi

# --- the verdict --------------------------------------------------------------------------------

{
  echo "check-boot: $(basename "$vm") on $( ((tcg)) && echo TCG || echo HVF), bundle $(basename "$bundle"), logs in $out"
  [[ -n $ssh_s ]] && echo "check-boot: guest macOS $(grep -A2 -m1 '^\$ sw_vers' "$out/guest.log" | tail -2 | tr '\n' ' ' | sed 's/ $//')"
  failed=()
  for n in "${names[@]}"; do
    printf 'check-boot: %-18s %s  %s\n' "$n" "$(state "$n")" "$(detail "$n")"
    [[ $(state "$n") == PASS ]] || failed+=("$n")
  done
  if ((${#failed[@]} == 0)); then
    echo "check-boot: PASS, ${#names[@]} of ${#names[@]} checks"
  else
    echo "check-boot: FAIL, ${#failed[@]} of ${#names[@]} checks failed: ${failed[*]}"
  fi
} | tee "$out/verdict.txt"
ended=1
grep -q '^check-boot: PASS' "$out/verdict.txt"
