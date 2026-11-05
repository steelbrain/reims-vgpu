#!/bin/bash
# The lane gate's VM launch on the lane host, inside a potluck job (msi-boot.sh runs it):
#
#   msi-launch.sh TAG OCIMG DISKBASE [extra qemu args...]
#
# It does what the host's shared rgk-boot2.sh does, without its locks: potluck's lease is the lock
# (the boot's job holds the 5080 and, --exclusive, the whole host), so there is no turnstile, VM
# slot, GPU flock, host-quiet lock or port lock here.
# - Overlays: RGK_RUN/boots/TAG/{oc,disk}.qcow2, backed by OCIMG and DISKBASE (never written).
#   max-ram-below-4g=1G: q35 keeps 1 GiB of RAM below 4 GiB, so macOS can place the 1 GiB BAR 2 there.
# - The SSH port: the first in 2240-2299 that nothing listens on ("PORT n"; RGK_RUN/boots/TAG/port).
# - Before QEMU starts, run.sh waits up to RGK_QUIET_WAIT seconds (default 180) for running builds
#   (cargo or rustc processes, which no potluck job beside an exclusive one can start) to end, and
#   says HOST_QUIET, or HOST_NOT_QUIET where they did not; then for the host CPU package to cool
#   below COOL_C (default 80) degrees, at most COOL_WAIT_S seconds (default 300), saying
#   "COOL_WAIT pkg N C after S s". msi throttles at 98-105 C under builds, which moves CPU-bound
#   numbers. Then GPU_HELD, from which msi-boot.sh counts the boot's hold, and QEMU.
# - RGK_EXCLUSIVE=1 (the bench boot) powers QEMU off after RGK_EXCLUSIVE_MAX seconds (default
#   900), else after RGK_SHARED_MAX (default 1500).
# - QEMU runs detached (setsid) with SCHED_OTHER (chrt, for ananicy), its stdout and stderr
#   timestamped (RGK_RUN/ts-lines) into launch.log. Prints SSH_OK once the guest answers SSH on
#   the port, else NO_SSH or QEMU_EXITED.
set -euo pipefail
RUN=${RGK_RUN:-/home/aneesiqbal/.cache/rgk-run}
TAG=$1 OC=$(readlink -f "$2") DISK=$(readlink -f "$3"); shift 3
B=$RUN/boots/$TAG; rm -rf "$B"; mkdir -p "$B"
qemu-img create -q -f qcow2 -b "$OC" -F qcow2 "$B/oc.qcow2"
qemu-img create -q -f qcow2 -b "$DISK" -F qcow2 "$B/disk.qcow2"
cp -f /home/aneesiqbal/.local/share/reims-ctl/vms/macos-26/OVMF_VARS.fd "$B/OVMF_VARS.fd"
if [ -z "${PORT:-}" ]; then
  for p in $(seq 2240 2299); do
    if ! ss -Hltn "sport = :$p" | grep -q .; then PORT=$p; break; fi
  done
  [ -n "${PORT:-}" ] || { echo "PORT_BUSY none free in 2240-2299"; exit 4; }
fi
echo "PORT $PORT"
if ss -Hltn "sport = :$PORT" | grep -q .; then echo "PORT_BUSY $PORT"; exit 4; fi
QEMU=${QEMU:?set QEMU= to the qemu-system-x86_64 built from the tip being booted}
ROM=${ROM:?set ROM= to the option ROM built from the tip being booted}
GFX=${GFX:--device reims-graphics-pci,id=gfx,bus=pcie.0,addr=0x4,romfile=$ROM}
cat > "$B/run.sh" <<RUNEOF
#!/bin/bash
export WAYLAND_DISPLAY=wayland-1 DISPLAY=:0 XDG_RUNTIME_DIR=/run/user/1000 XDG_SESSION_TYPE=wayland DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
export REIMS_VGPU_LOG=\${REIMS_VGPU_LOG:-info} REIMS_VGPU_WINDOW_BORDERLESS=1
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
# Powers QEMU (this shell's pid, which execs it) off after \$1 s. It ends when QEMU does, and kills
# only the same process: the pid's start time (/proc stat field 22) must not have changed.
watchdog() {
  ( st=\$(awk '{print \$22}' /proc/\$\$/stat); end=\$(( \$(date +%s) + \$1 ))
    while [ "\$(awk '{print \$22}' /proc/\$\$/stat 2>/dev/null)" = "\$st" ]; do
      [ \$(date +%s) -ge \$end ] && { kill \$\$; break; }
      sleep 5
    done ) &
}
# Host quiet: no build beside the boot.
q=0
while pgrep -x cargo > /dev/null || pgrep -x rustc > /dev/null; do
  [ \$q -ge \${RGK_QUIET_WAIT:-180} ] && break
  sleep 5; q=\$((q + 5))
done
if pgrep -x cargo > /dev/null || pgrep -x rustc > /dev/null; then echo HOST_NOT_QUIET; else echo HOST_QUIET; fi
# The cool-down: the package below COOL_C before the timed part.
z=\$(grep -l x86_pkg_temp /sys/class/thermal/thermal_zone*/type 2>/dev/null | head -1); z=\${z%/type}
c=0
if [ -n "\$z" ]; then
  while [ \$(cat \$z/temp) -ge \$(( \${COOL_C:-80} * 1000 )) ] && [ \$c -lt \${COOL_WAIT_S:-300} ]; do sleep 5; c=\$((c + 5)); done
  echo "COOL_WAIT pkg \$(( \$(cat \$z/temp) / 1000 )) C after \$c s"
fi
if [ "\${RGK_EXCLUSIVE:-0}" = 1 ]; then watchdog \${RGK_EXCLUSIVE_MAX:-900}; else watchdog \${RGK_SHARED_MAX:-1500}; fi
echo GPU_HELD
sudo -n chrt -o -p 0 \$\$ 2>/dev/null || true
# QEMU's stdout and stderr, each line timestamped (ts-lines), into launch.log.
exec > >($RUN/ts-lines) 2>&1
exec $QEMU -name rgk-$TAG \\
  -machine q35,memory-backend=ram,dump-guest-core=off,max-ram-below-4g=1G \\
  -object memory-backend-memfd,id=ram,size=16384M,share=on \\
  -accel kvm -m 16384M \\
  -cpu Skylake-Client-noTSX-IBRS,kvm=on,vendor=GenuineIntel,+invtsc,vmware-cpuid-freq=on,vmx=on,+ssse3,+sse4.2,+popcnt,+avx,+aes,+xsave,+xsaveopt,check \\
  -smp 8,sockets=1,cores=8,threads=1 \\
  -global ICH9-LPC.acpi-pci-hotplug-with-bridge-support=off -global ICH9-LPC.disable_s3=1 \\
  -device pcie-root-port,bus=pcie.0,addr=0x1,slot=1,x-speed=16,x-width=32 \\
  -device 'isa-applesmc,osk=ourhardworkbythesewordsguardedpleasedontsteal(c)AppleComputerInc' \\
  -drive if=pflash,format=raw,readonly=on,file=/home/aneesiqbal/.local/share/reims-ctl/vms/macos-26/OVMF_CODE.fd \\
  -drive if=pflash,format=raw,file=$B/OVMF_VARS.fd \\
  -smbios type=2 \\
  -device ich9-ahci,id=sata,addr=0x2 \\
  -drive id=opencore,if=none,format=qcow2,file=$B/oc.qcow2 \\
  -device ide-hd,bus=sata.2,drive=opencore,bootindex=1 \\
  -drive id=disk,if=none,format=qcow2,file=$B/disk.qcow2,discard=unmap \\
  -device nvme,drive=disk,serial=reims-ctl,addr=0xa \\
  -netdev user,id=net0,ipv6=off,hostfwd=tcp:127.0.0.1:$PORT-:22 \\
  -device virtio-net-pci,netdev=net0,id=net0,addr=0x3 \\
  -vga none $GFX \\
  -device qemu-xhci,id=xhci,addr=0x5,conditional-intr-mapping=on \\
  -device usb-kbd,id=kbd,bus=xhci.0 -device usb-tablet,id=tablet,bus=xhci.0 \\
  -display none \\
  -serial file:$B/serial.log -msg timestamp=on \\
  -qmp unix:$B/qmp.sock,server=on,wait=off -monitor none \\
  -pidfile $B/qemu.pid $*
RUNEOF
chmod +x "$B/run.sh"
echo "$PORT" > "$B/port"
# kwin keeps each window closed while every output is off (kwin-reaper.sh): free what earlier
# boots left before this one starts.
"$RUN/kwin-reaper.sh" --once > /dev/null 2>&1 || true
setsid bash "$B/run.sh" > "$B/launch.log" 2>&1 < /dev/null &
L=$!  # run.sh execs QEMU, so this is QEMU's pid
# run.sh's quiet wait and cool-down come before GPU_HELD.
for _ in $(seq 0 5 $(( ${RGK_QUIET_WAIT:-180} + ${COOL_WAIT_S:-300} + 60 ))); do
  grep -q GPU_HELD "$B/launch.log" 2>/dev/null && break
  kill -0 $L 2>/dev/null || { echo QEMU_EXITED; tail -20 "$B/launch.log"; exit 1; }
  sleep 5
done
SOPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 -o PreferredAuthentications=password -o PubkeyAuthentication=no -o LogLevel=ERROR"
for i in $(seq 1 60); do
  kill -0 $L 2>/dev/null || { echo QEMU_EXITED; tail -20 "$B/launch.log"; exit 1; }
  # SSH counts only once this QEMU itself listens on the port.
  if ss -Hltnp "sport = :$PORT" | grep -q "pid=$L," &&
     sshpass -p aneesiqbal ssh -n -p $PORT $SOPTS aneesiqbal@127.0.0.1 true 2>/dev/null; then echo "SSH_OK after $((i*5))s"; exit 0; fi
  sleep 5
done
echo NO_SSH; tail -20 "$B/launch.log"; exit 1
