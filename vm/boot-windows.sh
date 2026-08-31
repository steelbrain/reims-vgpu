#!/usr/bin/env bash
# vm/boot-windows.sh — boot the macos-13 rail on a Windows host (WHPX + native Vulkan).
# Run from an MSYS2 MINGW64 shell. Expects the rail files under C:/hackintosh/vm/.
set -euo pipefail
VM_DIR="C:/hackintosh/vm"
QEMU="${QEMU_BIN:-$(dirname "$0")/../vendor/qemu/build/qemu-system-x86_64.exe}"
DEVICE="${DEVICE:-reims-vgpu-pci}"
ACCEL="${ACCEL:-whpx}"

ARGS=(
  -accel "$ACCEL"
  -m 16G
  -cpu "Skylake-Client,-hle,-rtm,vendor=GenuineIntel,+invtsc,vmware-cpuid-freq=on,+ssse3,+sse4.2,+popcnt,+avx,+avx2,+aes,+xsave,+xsaveopt,check"
  -machine q35
  -vga none
  -device pci-bridge,chassis_nr=5,id=pci.5,bus=pcie.0,addr=1e.0
  -device qemu-xhci,id=xhci
  -device usb-kbd,bus=xhci.0
  -device usb-tablet,bus=xhci.0
  -smp 16,cores=16,sockets=1
  -device 'isa-applesmc,osk=ourhardworkbythesewordsguardedpleasedontsteal(c)AppleComputerInc'
  -drive if=pflash,format=raw,readonly=on,file="$VM_DIR/OVMF_CODE_4M.fd"
  -drive if=pflash,format=raw,file="$VM_DIR/OVMF_VARS.fd"
  -smbios type=2
  -device ich9-ahci,id=sata
  -drive id=OpenCoreBoot,if=none,format=qcow2,file="$VM_DIR/OpenCore.qcow2"
  -device ide-hd,bus=sata.2,drive=OpenCoreBoot
  -drive id=MacHDD,if=none,format=qcow2,file="$VM_DIR/macos.img"
  -device ide-hd,bus=sata.4,drive=MacHDD
  -qmp tcp:127.0.0.1:4444,server=on,wait=off
  -action reboot=shutdown
  -netdev user,id=net0,ipv6=off,hostfwd=tcp::2222-:22
  -device virtio-net-pci,netdev=net0,id=net0,mac=52:54:00:c9:18:27
  -serial "file:$VM_DIR/serial-windows.log"
)

EXTRA_ARGS=()
if [ "$DEVICE" = "reims-vgpu-pci" ]; then
  EXTRA_ARGS=(-display none -device "reims-vgpu-pci,id=reimsvgpu,romfile=$VM_DIR/reims-vgpu-gop.rom,rombar=1,bus=pci.5,addr=00.0")
  export REIMS_VGPU_WINDOW="${REIMS_VGPU_WINDOW:-on}"
else
  EXTRA_ARGS=(-display sdl -device "$DEVICE")
fi

"$QEMU" "${ARGS[@]}" "${EXTRA_ARGS[@]}"
