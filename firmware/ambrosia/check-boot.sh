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
#   -f FILE  Ambrosia's image: target/ambrosia.bin after its `make`, or a release's ambrosia.bin
#   -q FILE  our QEMU's qemu-system-aarch64 with reims-graphics-pci (docs/qemu.md, "arm64 guests:
#            the device on Ambrosia"); default $QEMU
#   -t       TCG, for a Linux member, in place of HVF (docs/qemu.md, "arm64 guests: the boot
#            check of our configuration", says what such a run needs). The device runs on the GPU
#            the run is given: the one a potluck job holds (POTLUCK_GPUS, its first), else the one
#            MESA_VK_DEVICE_SELECT names; with neither the script stops, as on a host with several
#            GPUs the library would take its own pick (docs/lessons.md).
#   -b FILE  the configuration bundle; default build/Config.cpio (make config-bundle)
#   -T DIR   the guest probes, bench and present_probe; default build/guest-tools (make
#            guest-tools, on a Mac)
#   -p PORT  the host port forwarded to the guest's SSH; default 10090
#   -o DIR   where the logs go, a new directory; default one made under ${TMPDIR:-/tmp}
#   -e ECID  the guest's ECID, in decimal or 0x hex
#
# The guest runs with 8 GiB and 4 CPUs, as Ambrosia's tested boots do, with QEMU's
# REIMS_VGPU_WINDOW=0 (no window on the host) and REIMS_VGPU_LOG=info with the display's flips
# (reims_vgpu::graphics::shown at debug), as the lane gate runs it.
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
#              Under TCG every command the checks run in the guest, each SSH login the boot waits
#              on, and the wait for the desktop have 5 times their HVF limits.
#              A check whose reading in the guest printed nothing (its command failed or ran
#              out of time) fails as unread: no reading passes as an empty one.
#   sip        `csrutil status` says enabled (the lane gate's sip).
#   boot-args  `sysctl kern.bootargs` is empty.
#   kexts      `kmutil showloaded` lists Graft and our two kexts (stage-config.py's identifiers),
#              as the lane gate's kexts lists ReimsGraphics.
#   windowserver-metal
#              WindowServer composites with Metal on our device, as the lane gate's guest checks
#              and boot read it: the desktop came up (WindowServer, Dock and Finder) within 60 s
#              of SSH under HVF, 300 s under TCG, each of the three read as a pid; `bench -l`
#              lists a Metal device, not Apple's paravirtual one; WindowServer kept its pid to
#              the end; and the host's log says the display shows an IOSurface the guest
#              presented.
#   metal-app  present_probe, an on-screen Metal app, ran full-screen for 10 s in the user's
#              session, as the lane gate's present phase: OK, frames presented, and at most a
#              tenth dropped or unanswered (report.py's bar for the correctness boot). Under TCG
#              it passes on OK, frames presented and at most 3 unanswered; its drops and rates
#              are printed but not judged, as the emulated CPU sets the rate WindowServer
#              composites at, and pacing is judged under HVF only. Beside the probe's line, the
#              host's count of display flips during the probe's call, WindowServer's own present
#              rate, to read its drops against. A probe that falls short while Setup Assistant
#              runs fails as unread, naming it: its full-screen window covers the probe's, whose
#              frames are then never shown. On Linux the line also gives the host's pressure over
#              the probe's call, as /proc/pressure and /proc/vmstat count it: the time a second
#              that some task waited for a CPU, for memory and for IO, the time all did for
#              memory, and the direct compactions and reclaims; a probe that falls short beside
#              other work on the host shows it there. The probe is full-screen at the status
#              level, which needs no right to be in front: WindowServer denied a probe started
#              over SSH that right, and its titled window was never shown (docs/lessons.md).
#   bundle-install
#              RgfxGate installed the Metal bundle, as it records it and as the disk shows it:
#              its Graft client state (GraftClient.ai.aneesiqbal.RgfxGate) reads installed,
#              tagged or present; /Library/GPUBundles/reims-graphics.bundle is there with the
#              restricted flag; and its com.apple.rootless class is KernelExtensionManagement.
#   amfi-map   AMFI admitted the bundle where it counts: the WindowServer running when the
#              check runs (its pid read then, as a WindowServer that died since the desktop came
#              up has another) has the bundle's executable mapped (`sudo lsof` on its pid, which
#              reads the mappings without task_for_pid, so SIP allows it), and this boot's log
#              has no AMFI or library-validation denial naming reims-graphics.
#   page-count the kext's count of a surface's pages (mach_vm_region over kernel_map, and
#              inside IOKit's pageable maps by IOIteratePageableMaps) did not take the guest
#              down: read after the Metal work, its kern.bootsessionuuid is the one it had at
#              the login (each boot sets a new one; kern.boottime is not compared, as macOS
#              corrects it while its clock settles, more so under TCG), and no panic report was
#              written in this run. Its detail is the host's last count of IOSurface device
#              caches (made, deferred, carrying pages, refused) and the kext's own count lines
#              this boot (counted, not counted). Zero deferred while caches were made, no count
#              line, or no count that counted is printed as a finding under the verdict, not a
#              failure.
#   ax-matrix  WindowServer set the display's accessibility matrix: this boot's log has no "not able
#              to set accessibility matrix" line (ReimsGraphics' colour filter, its set_parameter).
#   amfi-refuse
#              AMFI refuses a signature it rejects, through RgfxGate's signature hook, with the
#              guest up: a copy of /usr/bin/true signed ad hoc with com.apple.rootless.install, an
#              entitlement only Apple's signatures may carry, is killed (exit 137), and 5 s later
#              kern.bootsessionuuid is the login's. It runs last of the guest's checks, after
#              every read of this boot's log, as a failure can take the guest down.
#   screen     the display at the end, as QMP's screendump has it (screen.png, else screen.ppm), is
#              not one colour or nearly so: its commonest colour covers at most 90% of it, and it
#              has at least 1024 distinct colours. A desktop composited empty is one colour while
#              WindowServer still presents, which metal-app, measuring presentation and not pixels,
#              passes. Measured on 2026-11-11 over the screendumps of 69 passing boots under HVF
#              (14.6.1, 15.6.1, 26.6.2, 27.0.1 and 27.2.1, white-widget and clean desktops among
#              them): the commonest colour on 3.7% to 21.0% of the screen and 240414 to 404446
#              distinct colours; the 48 of 27.0.1, 3.7% to 5.0% and 395116 to 404446. An
#              all-black frame has one colour on all of it.
#   no-fatal   no `reims-vgpu: fatal` line in QEMU's log or the serial log.
#
# OUT keeps serial.log, qemu.log, screen.png (QMP's screendump of the display at the end, before
# QEMU is stopped), guest.log (what the guest's commands printed, Graft's state on the registry
# root among them) and verdict.txt.
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
  # The GPU the run is given (the header's -t): a potluck job's lease names it as vendor:device@bus.
  gpu=${POTLUCK_GPUS:-}
  gpu=${gpu%%,*} gpu=${gpu%%@*}
  if [[ $gpu == *:* ]]; then
    export MESA_VK_DEVICE_SELECT="$gpu!"
  elif [[ -z ${MESA_VK_DEVICE_SELECT:-} ]]; then
    die "no GPU given: run in a job that holds one (POTLUCK_GPUS) or name it in MESA_VK_DEVICE_SELECT"
  fi
  accel=(-accel tcg,thread=multi) limit=300 desktop_wait=300 slow=5
  machine=",memory-backend=ram,dump-guest-core=off"
  mem=(-m "$ram" -object "memory-backend-memfd,id=ram,size=$ram,share=on")
else
  accel=(-accel hvf,kernel-irqchip=on) limit=90 desktop_wait=60 slow=1
  machine= mem=(-m "$ram")
fi

[[ -n $firmware && -f $firmware ]] || die "no Ambrosia image: give its target/ambrosia.bin with -f"
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
# The host's pressure counters (Linux): cpu some, memory some and full, io some and full, in µs,
# then direct compactions and direct reclaims, one line; nothing where /proc/pressure is not.
host_pressure() {
  [[ -r /proc/pressure/memory ]] || return 0
  local r
  for r in cpu memory io; do
    awk '{ for (i = 2; i <= NF; i++) if (sub(/^total=/, "", $i)) printf "%s ", $i }' "/proc/pressure/$r"
  done
  awk '$1 == "compact_stall" { c = $2 } $1 ~ /^allocstall_/ { a += $2 } END { printf "%d %d\n", c, a }' /proc/vmstat
}
# Two readings of host_pressure over SECS seconds, said: the waits as ms a second, and the counts.
host_pressure_say() {
  perl -e 'my @a = split " ", $ARGV[0]; my @b = split " ", $ARGV[1]; my $s = $ARGV[2] || 1;
    my @d = map { $b[$_] - $a[$_] } 0 .. $#a; my $ms = sub { sprintf "%.0f", $d[$_[0]] / 1000 / $s };
    # cpu some, then cpu full where the kernel counts it (skipped), memory some and full, io some and full.
    my $o = @d == 8 ? 1 : 0;
    printf "cpu some %s ms/s, memory some %s full %s ms/s, io some %s full %s ms/s, ",
      $ms->(0), $ms->(1 + $o), $ms->(2 + $o), $ms->(3 + $o), $ms->(4 + $o);
    printf "%d compaction stalls, %d direct reclaims", $d[-2], $d[-1]' "$1" "$2" "$3"
}
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
# The boot's login probe opens none (PROBE_SOPTS): a probe its timeout cuts off, as one can be
# while a slow boot (-v) brings sshd up, could leave a master behind that later probes would wait
# on. The first command after the login (G) opens the shared one.
# A connection's own limit is slowed with the rest: under TCG a guest busy after its login can take
# longer than 5 s to send sshd's banner, and longer than 10 s to finish the key exchange.
BASE_SOPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PreferredAuthentications=password
  -o PubkeyAuthentication=no -o ConnectTimeout=$((5 * slow)) -o LogLevel=ERROR)
SOPTS=("${BASE_SOPTS[@]}" -o ControlMaster=auto -o "ControlPath=/tmp/check-boot-%C" -o ControlPersist=yes)
PROBE_SOPTS=("${BASE_SOPTS[@]}" -o ControlMaster=no -o ControlPath=none)
# The guest's commands run in the C locale: macOS's ssh sends the host's LANG and LC_* (SendEnv in
# its ssh_config.d), and a locale the guest lacks (C.UTF-8 on macOS 14) has its perl print five
# warning lines before a command's own output.
GUEST_ENV=(env LANG=C LC_ALL=C)
# A command in the guest, under a limit of SECONDS (times `slow`, 5 under TCG): prints its output,
# also kept in guest.log, and returns its status. A login that fails (sshpass's 5, a refused
# password; ssh's 255) is tried twice more, 5 s apart, as the lane gate retries its copies.
G() {
  local s=$(($1 * slow)) try rc
  shift
  for try in 1 2 3; do
    echo "\$ $*" >> "$out/guest.log"
    timeout "$s" "${GUEST_ENV[@]}" sshpass -p "$GUEST_PASS" ssh -n -p "$port" "${SOPTS[@]}" "$GUEST_USER@127.0.0.1" "$@" > "$out/g.out" 2>&1
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
REIMS_VGPU_WINDOW=0 REIMS_VGPU_LOG=info,reims_vgpu::graphics::shown=debug "$qemu" "${accel[@]}" \
  -M "vmapple,uuid=$ecid$machine" "${mem[@]}" -smp 4 -bios "$firmware" \
  -drive "file=$clone/aux.img.trimmed,if=pflash,format=raw" \
  -drive "file=$clone/disk.img,if=pflash,format=raw" \
  -drive "file=$clone/aux.img.trimmed,if=none,id=aux,format=raw" \
  -drive "file=$clone/disk.img,if=none,id=root,format=raw" \
  -device vmapple-virtio-blk-pci,variant=aux,drive=aux \
  -device vmapple-virtio-blk-pci,variant=root,drive=root \
  -netdev "user,id=net0,ipv6=off,hostfwd=tcp:127.0.0.1:$port-:22" \
  -device virtio-net-pci,netdev=net0,romfile= \
  -device reims-graphics-pci \
  -device "loader,file=$bundle,addr=0x120000000,force-raw=on" \
  -display none -serial "file:$out/serial.log" \
  -qmp "unix:$qmp_sock,server=on,wait=off" \
  > "$out/qemu.log" 2>&1 < /dev/null &
qpid=$!

# --- the checks ---------------------------------------------------------------------------------

names=(firmware console ssh sip boot-args kexts windowserver-metal metal-app bundle-install amfi-map page-count ax-matrix
  amfi-refuse screen no-fatal)
# What a check reports beside its verdict, printed under it: neither a pass nor a failure.
findings=()
# The guest's kern.bootsessionuuid at the login and after the Metal work, the panic reports
# written in this run, and the kext's count lines this boot (page-count).
boot0= boot1= panics= kc=
# A check's verdict, PASS or FAIL, and what it found; in plain variables, for macOS's bash 3.2.
verdict() { printf -v "st_${1//-/_}" '%s' "$2"; printf -v "dt_${1//-/_}" '%s' "$3"; }
state() { local v="st_${1//-/_}"; echo "${!v:-}"; }
detail() { local v="dt_${1//-/_}"; echo "${!v:-}"; }
serial() { [[ -f $out/serial.log ]] && tr -d '\r' < "$out/serial.log"; }
# What the serial log gained since the last call, in slog_new: the line the last call ended part
# way through, if it did, and the bytes written since. The boot's wait reads the log this way,
# each byte about once, as a -v boot's log runs to hundreds of KB.
slog_off=0 slog_last= slog_new=
serial_read() {
  local size chunk
  slog_new=$slog_last
  [[ -f $out/serial.log ]] || return 0
  size=$(wc -c < "$out/serial.log" | tr -d ' ')
  ((size > slog_off)) || return 0
  # The x keeps a final newline, which command substitution would drop.
  chunk=$(tail -c +"$((slog_off + 1))" "$out/serial.log" | head -c "$((size - slog_off))" | tr -d '\r'; printf x)
  slog_new+=${chunk%x}
  slog_off=$size
  slog_last=${slog_new##*$'\n'}
}

# The boot, up to the SSH limit: Ambrosia's stop line, QEMU's exit, or the guest's login.
stop= ssh_s= in_kernel=
while :; do
  serial_read
  stop=$(grep -a -m1 -E "^ambrosia: (bad configuration|bad graphics driver|can't read the boot objects|boot failed|exception [0-9]|panic:)" <<< "$slog_new")
  [[ -n $stop ]] && break
  [[ -n $in_kernel ]] || ! grep -a -q '^ambrosia: entering the kernel' <<< "$slog_new" || in_kernel=1
  kill -0 "$qpid" 2> /dev/null || break
  if [[ -n $in_kernel ]] &&
    timeout $((10 * slow)) "${GUEST_ENV[@]}" sshpass -p "$GUEST_PASS" ssh -n -p "$port" "${PROBE_SOPTS[@]}" "$GUEST_USER@127.0.0.1" true 2> /dev/null; then
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
  # The guest's release, from a command that answered: the login probe's success can come before
  # sshd takes the password every time, and G's output is its last try's alone.
  gver=$(G 20 'sw_vers -productVersion; sw_vers -buildVersion') || gver=
  boot0=$(G 20 'sysctl -n kern.bootsessionuuid') || boot0=

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
    loaded=$(echo "$shown" | grep -o -E 'ai\.aneesiqbal\.[A-Za-z.]+' | sort -u | tr '\n' ' ')
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
  # It is up once all three read as pids; a poll that failed or printed nothing is no reading.
  desk= t_desk=$(now) up='^[0-9]+ [0-9]+ [0-9]+$'
  while :; do
    d=$(G 20 'for p in WindowServer Dock Finder; do n=$(pgrep -x $p | head -1); echo ${n:-none}; done') &&
      desk=$(echo $d)
    [[ $desk =~ $up ]] && break
    perl -e "exit !($(since "$t_desk") > $desktop_wait)" && break
    sleep 5
  done
  [[ $gver =~ ^[0-9]+\.[0-9] ]] || { gver=$(G 20 'sw_vers -productVersion; sw_vers -buildVersion') || gver=; }
  if [[ $desk =~ $up ]]; then
    set -- $desk
    desk_ok=1 desk_say="WindowServer $1, Dock $2, Finder $3" ws_pid=$1
  elif [[ -z $desk ]]; then
    desk_ok=0 desk_say="no desktop in $desktop_wait s: unread, no poll answered" ws_pid=
  else
    set -- $desk none none none
    desk_ok=0 desk_say="no desktop in $desktop_wait s: WindowServer $1, Dock $2, Finder $3"
    ws_pid=${1//[^0-9]/}
  fi

  # The probes, copied over the shared connection.
  G 30 'mkdir -p ~/ambrosia-check' > /dev/null
  for t in bench present_probe; do
    timeout $((60 * slow)) "${GUEST_ENV[@]}" sshpass -p "$GUEST_PASS" ssh -p "$port" "${SOPTS[@]}" "$GUEST_USER@127.0.0.1" \
      "cat > ~/ambrosia-check/$t && chmod +x ~/ambrosia-check/$t" < "$tools/$t" >> "$out/guest.log" 2>&1
  done
  # bench -l's device lines; anything else it prints (the bundle's log lines, a warning) is
  # quoted only where it lists no device.
  if bl=$(G 40 "perl -e 'alarm $((30 * slow)); exec @ARGV' ~/ambrosia-check/bench -l"); then
    dev=$(echo "$bl" | grep '^bench: device' | head -5 | tr '\n' ' ' | sed 's/ *$//')
    case $dev in
    *Apple\ Paravirtual*) dev_ok=0 dev_say="Apple's paravirtual device: $dev" ;;
    ?*) dev_ok=1 dev_say=$dev ;;
    *)
      other=$(echo "$bl" | grep -v '^[0-9-]* [0-9:.]* .*RGFX' | tail -3 | tr '\n' ' ' | sed 's/ *$//')
      dev_ok=0 dev_say="no Metal device: bench -l lists none${other:+; it printed \"$other\"}"
      ;;
    esac
  else
    dev_ok=0 dev_say="bench -l failed: $(echo "$bl" | tail -2 | tr '\n' ' ')"
  fi

  if ((dev_ok && desk_ok)); then
    # Setup Assistant after a login (a release's new pages) covers the session with a full-screen
    # window, so the probe's frames are never shown: read whether it runs before the probe.
    cover=$(G 20 "pgrep -x 'Setup Assistant' | tr '\\n' ' '; echo; echo cover-read")
    if [[ $cover == *cover-read* ]]; then
      cover=$(echo "$cover" | head -1) cover=${cover% }
      cover_say=${cover:+Setup Assistant (pid ${cover// /, }) covers the session}
    else
      cover_say=
    fi
    # The host logs each display flip (reims_vgpu::graphics::shown, which this run names at
    # debug); those during the probe's call are WindowServer's own
    # presents, the rate the probe's frames can reach the display at.
    p0=$(host_pressure) q0=$(wc -l < "$out/qemu.log") t0=$SECONDS
    probe=$(G 60 "echo '$GUEST_PASS' | sudo -S -p '' launchctl asuser \$(id -u) sudo -u \$(id -un) perl -e 'alarm shift @ARGV; exec @ARGV' $((40 * slow)) \$HOME/ambrosia-check/present_probe -t 10" |
      grep '^present_probe:' | head -1)
    flips=$(tail -n +"$((q0 + 1))" "$out/qemu.log" | grep -c -E 'the display shows IOSurface [1-9]') secs=$((SECONDS - t0))
    ws_say="WindowServer: $flips display flips in the probe's ${secs} s call ($(perl -e 'printf "%.1f", $ARGV[0] / ($ARGV[1] || 1)' "$flips" "$secs")/s)"
    [[ -n $p0 ]] && ws_say="$ws_say; host over it: $(host_pressure_say "$p0" "$(host_pressure)" "$secs")"
    # report.py's bar for the correctness boot: frames presented, and at most a tenth lost, a
    # loss being a drop or an answer still missing beyond the last 3. Under TCG: frames
    # presented and at most 3 answers missing; drops are not judged.
    if [[ $probe == "present_probe: OK "* ]] && perl -e '
        my ($line, $tcg) = @ARGV;
        my %k = map { /^(\w+)=([-0-9.]+)$/ ? ($1, $2) : () } split " ", $line;
        my ($p, $u) = ($k{presented} // 0, $k{unanswered} // 0);
        exit !($p > 0 && $u <= 3) if $tcg;
        my $l = ($k{dropped} // 0) + ($u > 3 ? $u - 3 : 0);
        exit !($p > 0 && $l <= 0.1 * ($p + $l))' "$probe" "$tcg"; then
      verdict metal-app PASS "${probe#present_probe: }; $ws_say"
    elif [[ -n $cover_say ]]; then
      verdict metal-app FAIL "unread: $cover_say; ${probe:-present_probe printed no result line}; $ws_say"
    else
      verdict metal-app FAIL "${probe:-present_probe printed no result line}; $ws_say"
    fi
  elif ((!dev_ok)); then
    verdict metal-app FAIL "not run: $dev_say"
  else
    verdict metal-app FAIL "not run: $desk_say"
  fi
  # Graft's state on the registry root, kept in guest.log alone, so that a boot that fails into
  # Graft leaves it behind: GraftState, GraftReason, GraftLog and each GraftClient.<bundle id>.
  G 30 "ioreg -l -w0 -d1 | grep -E '\"Graft(State|Reason|Log|Client\\.[A-Za-z.]+)\" = '" > /dev/null
  # RgfxGate's install: the state it records, then the bundle's flags and SIP class on disk.
  bpath=/Library/GPUBundles/reims-graphics.bundle
  gstate=$(G 30 "ioreg -l -w0 | grep -m1 -o '\"GraftClient.ai.aneesiqbal.RgfxGate\" = \"[^\"]*\"'")
  gstate=${gstate#*= }
  bflags=$(G 20 "ls -ldO $bpath 2>&1 | head -1")
  bclass=$(G 20 "xattr -p com.apple.rootless $bpath 2>&1 | head -1")
  inst=() iok=1
  case $gstate in
  \"installed:* | \"tagged:* | \"present:*) inst+=("RgfxGate's state $gstate") ;;
  '') inst+=("no GraftClient.ai.aneesiqbal.RgfxGate state") iok=0 ;;
  *) inst+=("RgfxGate's state $gstate") iok=0 ;;
  esac
  if [[ $bflags == *restricted* ]]; then inst+=("bundle restricted")
  else inst+=("bundle not restricted: ${bflags:-no answer}") iok=0; fi
  if [[ $bclass == KernelExtensionManagement ]]; then inst+=("rootless class $bclass")
  else inst+=("rootless class: ${bclass:-none}") iok=0; fi
  verdict bundle-install "$( ((iok)) && echo PASS || echo FAIL)" "$(IFS=';'; echo "${inst[*]}" | sed 's/;/; /g')"

  # AMFI's admission: the bundle mapped in the WindowServer running now, and no denial naming it
  # in this boot's log. The pid is read again here: the desktop's may have died since.
  ws_now=$(G 20 'pgrep -x WindowServer | head -1') || ws_now=
  ws_now=${ws_now//[^0-9]/}
  if [[ -n $ws_now ]]; then
    mapped=$(G 60 "echo '$GUEST_PASS' | sudo -S -p '' lsof -p $ws_now 2>/dev/null | grep -c 'GPUBundles/reims-graphics.bundle'")
    mapped=${mapped//[^0-9]/}
  else
    mapped=
  fi
  ws_was=
  [[ -n $ws_pid && -n $ws_now && $ws_now != "$ws_pid" ]] && ws_was=", the desktop's was $ws_pid"
  # The guest's log is read with /usr/bin/log by its path: the guest user's shell is zsh, whose
  # `log` builtin takes the name over ssh and fails on these arguments, so a bare `log show`
  # read no line and the checks below counted 0 of everything.
  # A denial, not AMFI's notes on the bundle's signature: every boot logs that the bundle is ad hoc
  # signed and amfid's "not valid" for it, and WindowServer maps it all the same (measured on
  # arm64 27.0.1). The lines themselves come back, into guest.log, then a line that says the
  # log was read.
  denied=$(G 120 "/usr/bin/log show --last boot --style compact --predicate 'eventMessage CONTAINS \"reims-graphics\"' 2>/dev/null | grep -i -E 'library validation failed|rejecting|denying|denied'; echo check-boot-read")
  denied_first=
  if [[ $denied == *check-boot-read ]]; then
    denied_first=$(echo "$denied" | grep -v '^check-boot-read$' | head -1)
    denied=$(echo "$denied" | grep -c -v '^check-boot-read$')
  else
    denied=
  fi
  if [[ -z $ws_now ]]; then
    verdict amfi-map FAIL "not run: no WindowServer"
  elif [[ -z $denied ]]; then
    verdict amfi-map FAIL "WindowServer (pid $ws_now$ws_was) maps the bundle ${mapped:-0} times; this boot's log unread, so AMFI's denials are not known"
  elif [[ ${mapped:-0} -gt 0 && $denied -eq 0 ]]; then
    verdict amfi-map PASS "WindowServer (pid $ws_now$ws_was) maps the bundle ($mapped mappings); no AMFI or library-validation denial naming it"
  else
    verdict amfi-map FAIL "WindowServer (pid $ws_now$ws_was) maps the bundle ${mapped:-0} times; ${denied:-unread} AMFI or library-validation denials name reims-graphics${denied_first:+, the first: ${denied_first:0:200}}"
  fi

  # The page count's path, after the Metal work: the guest's boot unchanged, the panic reports
  # written in this run (each marked where it names our kexts or the count's calls), and the
  # kext's count lines this boot. The host's counts are read from its log once QEMU has ended.
  boot1=$(G 20 'sysctl -n kern.bootsessionuuid') || boot1=
  run_min=$(perl -e 'printf "%d\n", $ARGV[0] / 60 + 2' "$(since "$t0")")
  panics_read=1
  panics=$(G 60 "echo '$GUEST_PASS' | sudo -S -p '' sh -c 'find /Library/Logs/DiagnosticReports -maxdepth 2 -type f \\( -name \"*.panic\" -o -name \"panic-*\" \\) -mmin -$run_min | while read -r f; do if grep -q -E \"ai\\.aneesiqbal|mach_vm_region|IOIteratePageableMaps\" \"\$f\"; then echo \"\$f (names our kexts)\"; else echo \"\$f\"; fi; done'") || panics_read=0
  # The count lines, then a line that says the log was read: grep's status is 1 for no lines.
  # The kext's IOLog lines reach the kernel's message buffer but not the unified log, so they are
  # read with dmesg (128 KiB: a long boot can wrap its first lines out).
  # The read counts as read only where the buffer holds some line of our kexts, its control.
  kc=$(G 120 "echo '$GUEST_PASS' | sudo -S -p '' dmesg 2>/dev/null > /tmp/check-boot-dmesg; grep -E 'RgfxAccelerator: .*(counting an IOSurface|not a buffer the kernel allocated|could not be mapped to count)' /tmp/check-boot-dmesg; echo \"check-boot-read \$(grep -c -E 'ReimsGraphics: |RgfxAccelerator: ' /tmp/check-boot-dmesg)\"")
  kc_read=0
  kc_ours=$(echo "$kc" | sed -n 's/^check-boot-read \([0-9]*\)$/\1/p' | tail -1)
  [[ ${kc_ours:-0} -gt 0 ]] && kc_read=1
  kc=$(echo "$kc" | grep -v '^check-boot-read')
  ws_end=$(G 20 'pgrep -x WindowServer | head -1') || ws_end="unread: $(echo "$ws_end" | tail -1)"

  # The accessibility matrix, read while this boot is still the last one: WindowServer logs each
  # time it cannot set it. Then a line that says the log was read: grep's status is 1 for none.
  ax=$(G 120 "/usr/bin/log show --last boot --style compact --predicate 'eventMessage CONTAINS \"accessibility matrix\"' 2>/dev/null | grep -c 'not able to set accessibility matrix'; echo check-boot-read")
  ax_n=$(echo "$ax" | head -1)
  if [[ $ax != *check-boot-read || ! $ax_n =~ ^[0-9]+$ ]]; then
    verdict ax-matrix FAIL "unread: this boot's log not read in time"
  elif ((ax_n == 0)); then
    verdict ax-matrix PASS "no \"not able to set accessibility matrix\" line this boot"
  else
    verdict ax-matrix FAIL "$ax_n \"not able to set accessibility matrix\" lines this boot"
  fi

  # AMFI's refusal, last: an ad hoc signature with an entitlement only Apple's may carry. AMFI
  # kills the copy, and the guest keeps its boot session.
  sig=$(G 60 "cp /usr/bin/true /tmp/check-boot-sigfail && printf '%s' '<?xml version=\"1.0\" encoding=\"UTF-8\"?><plist version=\"1.0\"><dict><key>com.apple.rootless.install</key><true/></dict></plist>' > /tmp/check-boot-sigfail.plist && /usr/bin/codesign -f -s - --entitlements /tmp/check-boot-sigfail.plist /tmp/check-boot-sigfail > /dev/null 2>&1; /tmp/check-boot-sigfail; echo \"check-boot-sigfail \$?\"")
  sleep 5
  boot2=$(G 20 'sysctl -n kern.bootsessionuuid') || boot2=
  sig_rc=$(echo "$sig" | grep -o 'check-boot-sigfail [0-9]*' | tail -1 | grep -o '[0-9]*$')
  if [[ -z $boot2 ]]; then
    verdict amfi-refuse FAIL "the guest did not answer 5 s after the refused run (exit ${sig_rc:-unread})"
  elif [[ -z $boot0 ]]; then
    verdict amfi-refuse FAIL "kern.bootsessionuuid unread at the login, so a boot again is not told"
  elif [[ $boot2 != "$boot0" ]]; then
    verdict amfi-refuse FAIL "the guest booted again: kern.bootsessionuuid was $boot0, then $boot2"
  elif [[ $sig_rc == 137 ]]; then
    verdict amfi-refuse PASS "AMFI killed the ad hoc signed copy (exit 137); kern.bootsessionuuid kept"
  else
    verdict amfi-refuse FAIL "the ad hoc signed copy exited ${sig_rc:-unread}, not 137 (killed)"
  fi
elif [[ $(state firmware) == FAIL ]]; then
  for n in ssh sip boot-args kexts windowserver-metal metal-app bundle-install amfi-map page-count ax-matrix amfi-refuse; do
    verdict "$n" FAIL "not run: the firmware check failed"
  done
else
  verdict ssh FAIL "no SSH login in $limit s"
  for n in sip boot-args kexts windowserver-metal metal-app bundle-install amfi-map page-count ax-matrix amfi-refuse; do
    verdict "$n" FAIL "not run: no SSH login"
  done
fi
# What the display shows at the end: the desktop, or the console's log and the kernel's progress.
qmp "{\"execute\": \"screendump\", \"arguments\": {\"filename\": \"$out/screen.png\", \"format\": \"png\"}}" > /dev/null 2>&1 ||
  qmp "{\"execute\": \"screendump\", \"arguments\": {\"filename\": \"$out/screen.ppm\"}}" > /dev/null 2>&1
stop_qemu

# The screendump's colours: "DISTINCT SHARE", how many distinct colours it has and the share of
# its pixels the commonest covers, read by the standard library alone (zlib for an 8-bit PNG, as
# QEMU writes it, or a binary PPM); else "unread: WHY".
screen_colours() {
  python3 -I - "$1" << 'PY' 2>&1 | tail -1
import collections, struct, sys, zlib
path = sys.argv[1]
try:
    d = open(path, 'rb').read()
    if d[:8] == b'\x89PNG\r\n\x1a\n':
        i, idat, hdr = 8, [], None
        while i + 8 <= len(d):
            n, t = struct.unpack('>I4s', d[i:i + 8])
            if t == b'IHDR':
                hdr = struct.unpack('>IIBBBBB', d[i + 8:i + 21])
            elif t == b'IDAT':
                idat.append(d[i + 8:i + 8 + n])
            elif t == b'IEND':
                break
            i += 12 + n
        w, h, depth, ctype, _, _, interlace = hdr
        if depth != 8 or interlace or ctype not in (0, 2, 4, 6):
            raise ValueError(f'a PNG of depth {depth}, colour type {ctype}, interlace {interlace}')
        bpp = {0: 1, 2: 3, 4: 2, 6: 4}[ctype]
        raw, stride, prev, px = zlib.decompress(b''.join(idat)), w * bpp, bytearray(w * bpp), []
        for y in range(h):
            at = y * (stride + 1)
            f, line = raw[at], bytearray(raw[at + 1:at + 1 + stride])
            for x in range(stride) if f else ():
                a = line[x - bpp] if x >= bpp else 0
                b = prev[x]
                c = prev[x - bpp] if x >= bpp else 0
                if f == 1:
                    line[x] = (line[x] + a) & 255
                elif f == 2:
                    line[x] = (line[x] + b) & 255
                elif f == 3:
                    line[x] = (line[x] + ((a + b) >> 1)) & 255
                elif f == 4:
                    q = a + b - c
                    pa, pb, pc = abs(q - a), abs(q - b), abs(q - c)
                    line[x] = (line[x] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
            px.extend(bytes(line[x:x + min(bpp, 3)]) for x in range(0, stride, bpp))
            prev = line
    else:
        parts = d.split(maxsplit=4)
        if len(parts) < 5 or parts[0] != b'P6' or parts[3] != b'255':
            raise ValueError('neither an 8-bit PNG nor an 8-bit binary PPM')
        w, h, body = int(parts[1]), int(parts[2]), parts[4]
        px = [body[i:i + 3] for i in range(0, w * h * 3, 3)]
    if not px:
        raise ValueError('no pixels')
    colours = collections.Counter(px)
    print(f'{len(colours)} {colours.most_common(1)[0][1] / len(px):.4f}')
except Exception as e:
    print(f'unread: {e}')
PY
}
shot=$out/screen.png
[[ -s $shot ]] || shot=$out/screen.ppm
if [[ ! -s $shot ]]; then
  verdict screen FAIL "no screendump: QMP's screendump wrote no screen.png or screen.ppm"
else
  sc=$(screen_colours "$shot")
  read -r sc_n sc_top <<< "$sc"
  if ! [[ $sc_n =~ ^[0-9]+$ && $sc_top =~ ^[0-9.]+$ ]]; then
    verdict screen FAIL "$(basename "$shot") ${sc:-unread}"
  else
    sc_say="$(basename "$shot"): $sc_n distinct colours, the commonest on $(awk -v t="$sc_top" 'BEGIN { printf "%.1f", t * 100 }')% of it"
    if awk -v t="$sc_top" -v n="$sc_n" 'BEGIN { exit !(t <= 0.9 && n >= 1024) }'; then
      verdict screen PASS "$sc_say"
    else
      verdict screen FAIL "$sc_say: one colour or nearly so (a pass has the commonest on at most 90% and 1024 colours or more)"
    fi
  fi
fi

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

  # The page count: the host's last count of its IOSurface device caches (logged only once one
  # is made, at most once a minute, so up to a minute behind the boot's end), and the kext's count
  # lines, read in the guest after its Metal work.
  caches=$(grep -o 'IOSurface device caches: .*' "$out/qemu.log" | tail -1)
  cc=($(echo "$caches" | perl -ne 'my ($m) = /device caches: (\d+) made;/;
    print "$m $1 $2 $3 $4\n" if defined $m && /; (\d+) deferred \((\d+) carrying pages\), (\d+) refused, (\d+) refused as they might lie in place/'))
  made=${cc[0]:-0} deferred=${cc[1]:-0}
  n_counted=$(echo "$kc" | grep -c ': counted,')
  n_not=$(echo "$kc" | grep -c ': not counted,')
  notes=$(echo "$kc" | grep -o -E 'an IOSurface.s memory (is a [A-Za-z0-9_]+, not a buffer the kernel allocated|could not be mapped to count its pages)' | sort -u | tr '\n' ';' | sed 's/;$//; s/;/; /g')
  pc=() pok=1
  if [[ -z $boot1 ]]; then pc+=("the guest did not answer after its Metal work") pok=0
  elif [[ -z $boot0 ]]; then pc+=("kern.bootsessionuuid unread at the login, so a boot again is not told") pok=0
  elif [[ $boot1 != "$boot0" ]]; then pc+=("the guest booted again during the run: kern.bootsessionuuid was $boot0, then $boot1") pok=0
  else pc+=("kern.bootsessionuuid kept"); fi
  if ((!panics_read)); then pc+=("the panic reports unread: ${panics:-no answer}") pok=0
  elif [[ -n $panics ]]; then pc+=("panic reports written in this run: $(echo "$panics" | tr '\n' ';' | sed 's/;$//; s/;/, /g')") pok=0
  else pc+=("no panic report in this run"); fi
  if [[ -n $caches && -n ${cc[0]:-} ]]; then
    pc+=("host: ${cc[0]} device caches made, ${cc[1]} deferred (${cc[2]} carrying pages), ${cc[3]} refused (${cc[4]} as they might lie in place)")
  elif [[ -n $caches ]]; then pc+=("host: a device caches line in an unknown form: ${caches:0:160}")
  else pc+=("host: no device cache made"); fi
  if ((kc_read)); then pc+=("kext: $n_counted counts counted, $n_not not counted${notes:+; $notes} ($kc_ours lines of our kexts in dmesg)")
  else pc+=("kext: its count lines unread, the kernel's message buffer gave no line of our kexts"); fi
  verdict page-count "$( ((pok)) && echo PASS || echo FAIL)" "$(IFS='|'; echo "${pc[*]}" | sed 's/|/; /g')"
  ((made > 0 && deferred == 0)) && findings+=("page-count: zero deferred while the host made $made device caches")
  ((kc_read && n_counted + n_not == 0)) && findings+=("page-count: the kext logged no count of an IOSurface's pages this boot: the count's path did not run")
  ((n_counted == 0 && n_not > 0)) && findings+=("page-count: every count the kext logged was refused ($n_not not counted)")
  [[ -n $notes ]] && findings+=("page-count: the kext refused counts: $notes")
fi
if ((fatal_q + fatal_s == 0)); then
  verdict no-fatal PASS "no 'reims-vgpu: fatal' line in QEMU's log or the serial log"
else
  verdict no-fatal FAIL "$fatal_q 'reims-vgpu: fatal' lines in QEMU's log, $fatal_s in the serial log: $(grep -m1 'reims-vgpu: fatal' "$out/qemu.log")"
fi

# --- the verdict --------------------------------------------------------------------------------

{
  echo "check-boot: $(basename "$vm") on $( ((tcg)) && echo TCG || echo HVF), bundle $(basename "$bundle"), logs in $out"
  if [[ -n $ssh_s ]]; then
    if [[ $gver =~ ^[0-9]+\.[0-9] ]]; then echo "check-boot: guest macOS $(echo $gver)"
    else echo "check-boot: guest macOS unread: $(echo "$gver" | tail -1)"; fi
  fi
  failed=()
  for n in "${names[@]}"; do
    printf 'check-boot: %-18s %s  %s\n' "$n" "$(state "$n")" "$(detail "$n")"
    [[ $(state "$n") == PASS ]] || failed+=("$n")
  done
  for ((i = 0; i < ${#findings[@]}; i++)); do echo "check-boot: finding: ${findings[i]}"; done
  if ((${#failed[@]} == 0)); then
    echo "check-boot: PASS, ${#names[@]} of ${#names[@]} checks"
  else
    echo "check-boot: FAIL, ${#failed[@]} of ${#names[@]} checks failed: ${failed[*]}"
  fi
} | tee "$out/verdict.txt"
ended=1
grep -q '^check-boot: PASS' "$out/verdict.txt"
