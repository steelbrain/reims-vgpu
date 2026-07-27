#!/usr/bin/env bash
#
# vm/boot-x86.sh — boot an x86_64 macOS guest under QEMU+KVM on Linux.
#
# Display is selected by --device (primary VGA):
#   vmware-svga      default console (OSX-KVM mainstream)
#   reims-vgpu-pci   product Reims VGPU (thin C → reims-vgpu); -vga none + secondary bus
#
# RAILS. A rail is one guest OS line — `macos-11` … `macos-26` — with a history
# of its own. Rails are siblings under `vm/disks/rails/`, and `rails/current` is
# a symlink naming the one a boot gets when `--rail` is not given:
#
#   vm/disks/rails/<rail>/snapshots/<label>/{macos.img,OpenCore.qcow2,OVMF_VARS.fd}
#   vm/disks/rails/<rail>/snapshots/current -> <label>
#   vm/disks/rails/current -> <rail>
#
# Snapshots are per-rail because they are not comparable across rails: a macOS 15
# disk and a macOS 11 disk share no history, and a single flat namespace makes
# `current` mean "whichever guest was captured last", which is how a measurement
# ends up attributed to the wrong OS. Two coordinates, `--rail` and `--snapshot`,
# each with its own `current`, keep that from being expressible.
#
# SNAPSHOT-REVERT (same model as vm/boot-arm64.sh): within a rail, snapshots form
# an IMMUTABLE HISTORY (each file read-only, never overwritten). EVERY boot starts
# from a byte-identical COW clone of the selected snapshot (btrfs reflink when
# available) and discards that clone on exit, so a harsh kill or a wedge costs
# nothing and poisons nothing. A snapshot is never booted directly.
#
# Selection by either coordinate is per-boot and repoints no `current` symlink.
#
# A snapshot may carry its own `OVMF_CODE.fd`. macOS releases are installed under
# whatever OVMF build their installer ran on, and the NVRAM in `OVMF_VARS.fd`
# only means anything to the code half it was written by — a split OVMF pair is
# a matched set, not two independent files. When the snapshot ships one it wins
# over `$OVMF_DIR/OVMF_CODE_4M.fd`; when it does not, the tree default is used
# and the pair is whatever it has always been.
#
# Boot classes:
#   --testing      agent-driven measurement (default): GUI + serial-to-file,
#                  SSH-driven, 7-minute hard kill + capture-then-revert. Reverts.
#   --interactive  human/GUI boot, no time limit. Reverts (nothing persists).
#   --capture      boot writable to CAPTURE A NEW snapshot: on a clean guest
#                  shutdown the modified disk/OpenCore/OVMF_VARS are saved as a
#                  NEW immutable snapshot and `current` is repointed to it.
#                  Existing snapshots (incl. the base) are never touched.
#                  A bare `--snapshot` (no label) still means this.
#
# AUDIO is class-compliant USB, not HD Audio. An emulated ich9-intel-hda does
# reach the guest's PCI bus (pci8086,293e, pciclass,040300), but QEMU's codec
# advertises subsystem 1af4:1100 and AppleHDA binds only codecs it carries a
# profile for, so AppleHDAController stays loaded with zero references and
# CoreAudio enumerates no device. AppleUSBAudio ships in the guest and binds a
# class-compliant device with no kext work, which is why this rail is USB.
# virtio is not an option either: macOS has no virtio-sound driver.
#
# The audiodev is named on the device rather than left implicit — any `-audiodev`
# on the command line clears QEMU's default-backend list, and a sound device
# without `audiodev=` then refuses to realize ("no default audio driver
# available"), which kills the boot before OVMF.
#
# WHICH audiodev is chosen at run time, native-host-backend first, because the
# backends are not interchangeable under load. QEMU refills the backend from
# `audio_run_out` on its main loop at `timer-period` (10 ms), so every backend
# is exposed to main-loop jitter; what differs is what happens when the refill
# is late:
#
#   - `sdl` holds `4 * 11610 us` and its callback fills the shortfall with
#     zeroes. Zeroes in the middle of a waveform is what a click is, and it
#     neither logs nor counts them, so the symptom reaches the user and nothing
#     else. It is last in the preference order for that reason, not for quality.
#   - `pipewire` and `pa` each run their own thread with their own ring
#     (PipeWire's is a fixed 1 MiB) and default to 46440 us; PipeWire reports an
#     underrun when it does happen.
#   - `alsa` has an xrun trace (`alsa_xrun_out`).
#
# `AUDIODEV=` forces one. `AUDIO_BUFFER_US` and `AUDIO_USB_BUFFER` set the two
# buffers explicitly so the jitter budget is stated rather than inherited.
# `scripts/audio-crackle-probe` is what turns "it crackles" into a number.
#
# Launch configuration is CLI flags / env here (this is the boot script, not
# device/backend code) — never an env sniff inside the device.
#
# Credits for the historical OSX-KVM shape: Leoyzen/KVM-Opencore, thenickdude/KVM-Opencore.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Configuration (override via env or flags) ----------------------------------
DISKS_DIR="${DISKS_DIR:-$SCRIPT_DIR/disks}"
OVMF_DIR="${OVMF_DIR:-$SCRIPT_DIR/ovmf}"
RAILS_DIR="${RAILS_DIR:-$DISKS_DIR/rails}"
# Per-boot scratch is shared across rails on purpose: the clones are stamped and
# thrown away, and `run/qmp.sock` is the one path every driver script resolves.
# Splitting it per rail would give each rail its own qmp.sock and leave those
# scripts pointing at whichever rail booted last.
RUN_DIR="${RUN_DIR:-$DISKS_DIR/run}"

# In-tree QEMU is rebuilt every boot unless QEMU_BIN is overridden. The boot
# script still builds both Rust products first so stale host code and stale GOP
# ROMs do not survive a launch.
#
# Overriding QEMU_BIN is how a multi-boot batch pins one arm while the Rust tree
# is edited. Two constraints on what may be pinned, both learned by losing runs
# to them:
#
# - Keep the pinned copy inside `vendor/qemu/build/`. QEMU finds its firmware
#   through /proc/self/exe -> `<exedir>/qemu-bundle/usr/local/share/qemu`, so a
#   copy in /tmp prints this script's normal header and then dies on
#   `failed to find romfile "efi-virtio.rom"` — a boot that looks started.
# - Keep `qemu-system-x86_64` in the filename. Every VM sweep in this repo and
#   in the repro scripts matches that pattern. A pin named anything else
#   survives the sweep, keeps hostfwd 2222, and the next boot fails to bind and
#   exits — after which the scoring script talks to the *pinned* guest and
#   reports a clean run of the wrong binary.
QEMU_BIN_DEFAULT="$REPO_ROOT/vendor/qemu/build/qemu-system-x86_64"
QEMU_BIN="${QEMU_BIN:-$QEMU_BIN_DEFAULT}"
REIMS_VGPU_EFI_ROM_SCRIPT="$REPO_ROOT/crates/reims-vgpu-efi/scripts/reims-vgpu-efi-rom/reims-vgpu-efi-rom.sh"
# A snapshot's own OVMF_CODE.fd overrides this default (see the header), but an
# explicitly-exported OVMF_CODE outranks both — same DEFAULT-sentinel shape as
# QEMU_BIN above, because `${VAR:-fallback}` cannot tell "set to the default"
# from "not set" once it has run.
OVMF_CODE_DEFAULT="$OVMF_DIR/OVMF_CODE_4M.fd"
OVMF_CODE="${OVMF_CODE:-$OVMF_CODE_DEFAULT}"
OVMF_VARS_MASTER="${OVMF_VARS_MASTER:-$OVMF_DIR/OVMF_VARS-1920x1080.fd}"
OPENCORE_MASTER="${OPENCORE_MASTER:-$DISKS_DIR/OpenCore.qcow2}"
DISK_MASTER="${DISK_MASTER:-$DISKS_DIR/macos.img}"

RAM="${RAM:-16G}"
CPU_SOCKETS="${CPU_SOCKETS:-1}"
CPU_CORES="${CPU_CORES:-16}"
CPU_THREADS="${CPU_THREADS:-16}"
SSH_PORT="${SSH_PORT:-2222}"
TESTING_TIMEOUT="${TESTING_TIMEOUT:-420}" # 7-minute hard kill for testing boots
# PIN the guest NIC MAC across reverts (load-bearing for DHCP/sshd).
GUEST_MAC="${GUEST_MAC:-52:54:00:c9:18:27}"

CPU_MODEL="${CPU_MODEL:-Skylake-Client}"
CPU_OPTIONS="${CPU_OPTIONS:-+ssse3,+sse4.2,+popcnt,+avx,+avx2,+aes,+xsave,+xsaveopt,check}"

# Audio backend. `AUDIODEV` names a QEMU audiodev driver explicitly; unset picks
# the first one this build has that is native to the host — see the AUDIO note
# in the header for why SDL is the last resort rather than the default, and
# scripts/audio-crackle-probe for the measurement.
AUDIODEV="${AUDIODEV:-}"
# How much audio QEMU's backend holds, in microseconds. The mixer that refills
# it runs on QEMU's main loop at `timer-period`, so this is the jitter the host
# may show before the guest's sound has a hole in it. 46440 us is what the
# PipeWire and PulseAudio backends default to on their own; it is stated here so
# the number is the same on every backend rather than 11610 on one of them.
AUDIO_BUFFER_US="${AUDIO_BUFFER_US:-46440}"
# Bytes of guest-supplied audio the `usb-audio` device rings before the backend
# takes it. QEMU's default is 32 packets — 6144 bytes, about 32 ms at 48 kHz
# stereo — and `streambuf_put` drops a whole packet, silently, whenever the ring
# is full when an isochronous transfer lands. 64 KiB is ~340 ms, which is the
# other half of the same jitter budget and costs only latency nobody in a VM
# notices.
AUDIO_USB_BUFFER="${AUDIO_USB_BUFFER:-65536}"

BOOT_CLASS="testing"          # testing | interactive | capture
RAIL_LABEL="${RAIL:-}"        # empty = follow rails/current; else a rail name
SNAPSHOT_LABEL=""             # empty = follow the rail's snapshots/current
LIST_RAILS=0
LIST_SNAPSHOTS=0
# Default to the product device: it is the whole point of this tree, and its
# host-owned Vulkan window (REIMS_VGPU_WINDOW=1, present + input) is what a human
# expects to see. The legacy vmware-svga console is opt-in via --device — a bare
# `--capture`/`--interactive` under vmware-svga shows QEMU's gtk window with NO
# GPU output (the guest renders only through reims-vgpu-pci), which reads as a dead
# window. Agents pass --device explicitly anyway, so this only fixes the bare
# human invocation.
GFX_DEVICE="reims-vgpu-pci"  # reims-vgpu-pci | vmware-svga

usage() {
  cat <<EOF
usage: vm/boot-x86.sh [--device reims-vgpu-pci|vmware-svga] [--testing|--interactive|--capture]
                      [--rail NAME] [--snapshot LABEL]

  --device NAME          primary VGA (default: reims-vgpu-pci)
                         reims-vgpu-pci   product Reims VGPU (PCI thin shim → reims-vgpu),
                                            host-owned Vulkan window (present + input)
                         vmware-svga         legacy OSX-KVM console (QEMU gtk window;
                                            no GPU output once the guest uses Reims vGPU)
  --testing              agent boot (default): GUI, ${TESTING_TIMEOUT}s hard kill, reverts
  --interactive          human/GUI boot, no time limit, reverts
  --capture              boot writable; a clean guest shutdown CAPTURES a new snapshot
                         into the selected rail (also bootstraps an empty rail)
  --rail NAME            guest OS line to boot, e.g. --rail macos-11.
                         Default: whatever \`rails/current\` names.
  --snapshot LABEL       snapshot WITHIN that rail. Default: the rail's own
                         \`snapshots/current\`. A bare --snapshot (no label) is
                         the old spelling of --capture.
  --list-rails           print the rails and exit
  --list-snapshots       print the selected rail's snapshots and exit

Both selections are per-boot and repoint no \`current\`. Layout:
  $RAILS_DIR/<rail>/snapshots/<label>/{macos.img,OpenCore.qcow2,OVMF_VARS.fd[,OVMF_CODE.fd]}
Change the default rail with:  ln -sfn <rail> $RAILS_DIR/current
Always builds reims-vgpu-efi and reims-vgpu before boot. In-tree QEMU is rebuilt
unless QEMU_BIN is set to something other than the default path.
Env: DISKS_DIR OVMF_DIR RAILS_DIR RAIL RUN_DIR QEMU_BIN OVMF_CODE OVMF_VARS_MASTER
     OPENCORE_MASTER DISK_MASTER RAM CPU_SOCKETS CPU_CORES CPU_THREADS CPU_MODEL
     CPU_OPTIONS SSH_PORT TESTING_TIMEOUT QMP_DUMP_TIMEOUT GUEST_MAC REIMS_VGPU_BACKEND
     (metal|vulkan for qemu-build)
     NET=user (SLIRP, default) | NET=none (no NIC)
     REIMS_VGPU_PCI_ATTACH=pcibridge|bus0   (default pcibridge; product secondary bus)
     REIMS_VGPU_GOP_ROM=path | REIMS_VGPU_GOP_ROM= (option ROM on reims-vgpu-pci; auto if built)
     QEMU_REBOOT_ACTION=exit|pause|reset
       (default exit — guest reboot/KP-reset → QEMU quits; serial already on disk)
     TRACE=1 — QEMU trace events → \$RUN_DIR/trace-<stamp>.log
     TRACE_PATTERN=glob — override the default trace glob
     TRACE_EVENTS_FILE=path — QEMU trace event list file; overrides TRACE_PATTERN
EOF
}

set_gfx_device() {
  GFX_DEVICE="$1"
  case "$GFX_DEVICE" in
    vmware-svga|reims-vgpu-pci) ;;
    *)
      echo "boot-x86.sh: invalid --device '$GFX_DEVICE' (vmware-svga | reims-vgpu-pci)" >&2
      exit 64
      ;;
  esac
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --device)
      shift
      set_gfx_device "${1:-}"
      shift
      ;;
    --device=*)
      set_gfx_device "${1#--device=}"
      shift
      ;;
    --testing) BOOT_CLASS="testing"; shift ;;
    --interactive) BOOT_CLASS="interactive"; shift ;;
    --capture) BOOT_CLASS="capture"; shift ;;
    --rail) shift; RAIL_LABEL="${1:-}"; [ -n "$RAIL_LABEL" ] || { echo "boot-x86.sh: --rail needs a name" >&2; exit 64; }; shift ;;
    --rail=*) RAIL_LABEL="${1#--rail=}"; shift ;;
    # `--snapshot` carries two meanings, kept apart by whether a label follows.
    # With a label it SELECTS a snapshot within the rail; bare it is the old
    # capture class, which every existing invocation in this repo's docs and
    # helper scripts still spells that way. A following `--flag` is the next
    # option, not a label.
    --snapshot)
      case "${2:-}" in
        ""|-*) BOOT_CLASS="capture"; shift ;;
        *) SNAPSHOT_LABEL="$2"; shift 2 ;;
      esac
      ;;
    --snapshot=*) SNAPSHOT_LABEL="${1#--snapshot=}"; shift ;;
    --list-rails) LIST_RAILS=1; shift ;;
    --list-snapshots) LIST_SNAPSHOTS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "boot-x86.sh: unknown arg: $1" >&2; usage >&2; exit 64 ;;
  esac
done

# --- Preflight ------------------------------------------------------------------
die() { echo "boot-x86.sh: $*" >&2; exit 1; }

# Directory children of a dir; a `current` symlink is -type l, so it is skipped
# and never lists itself as one of the things it points at.
list_dir_labels() {
  find "$1" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort
}
list_rail_labels() { list_dir_labels "$RAILS_DIR"; }
list_snapshot_labels() { list_dir_labels "$RAILS_DIR/$RAIL_NAME/snapshots"; }

# A label names one directory directly under its parent. Refusing a path keeps
# both histories flat, and keeps `--rail ../../elsewhere` from putting a capture
# outside the tree.
require_plain_label() {
  case "$2" in
    */*|.|..|"") die "$1 takes a plain label, not a path: '$2'" ;;
  esac
}

# --- Resolve the rail ------------------------------------------------------------
# Answered before anything is built — these are questions about the disk tree.
if [ "$LIST_RAILS" -eq 1 ]; then
  echo "rails under $RAILS_DIR (current -> $(readlink "$RAILS_DIR/current" 2>/dev/null || echo '(unset)')):"
  list_rail_labels | while IFS= read -r label; do echo "  $label"; done
  exit 0
fi

if [ -n "$RAIL_LABEL" ]; then
  require_plain_label --rail "$RAIL_LABEL"
  RAIL_NAME="$RAIL_LABEL"
else
  RAIL_NAME="$(readlink "$RAILS_DIR/current" 2>/dev/null || true)"
  [ -n "$RAIL_NAME" ] || die \
    "no default rail: $RAILS_DIR/current is unset.
Pick one per boot with --rail NAME, or set the default:  ln -sfn <rail> $RAILS_DIR/current
available: $(list_rail_labels | tr '\n' ' ')"
  # `current` is allowed to be an absolute symlink; reduce it to a name so the
  # rail reads the same in the log line whichever way it was written.
  RAIL_NAME="$(basename "$RAIL_NAME")"
fi
RAIL_DIR="$RAILS_DIR/$RAIL_NAME"
SNAPSHOTS_DIR="$RAIL_DIR/snapshots"
# An unknown rail is always an error. Only an EMPTY one may bootstrap, and only
# under --capture: a typo'd name would otherwise create the rail on capture and
# the boot would quietly become a different guest line.
[ -d "$RAIL_DIR" ] || die \
  "no rail '$RAIL_NAME' at $RAIL_DIR
available: $(list_rail_labels | tr '\n' ' ')
(start a new guest line by creating the directory first:  mkdir -p $RAIL_DIR
 then bootstrap it with:  vm/boot-x86.sh --rail $RAIL_NAME --capture)"

# --- Resolve the snapshot within that rail ---------------------------------------
CURRENT="$SNAPSHOTS_DIR/current"
if [ "$LIST_SNAPSHOTS" -eq 1 ]; then
  echo "rail '$RAIL_NAME' snapshots under $SNAPSHOTS_DIR (current -> $(readlink "$CURRENT" 2>/dev/null || echo '(unset)')):"
  list_snapshot_labels | while IFS= read -r label; do echo "  $label"; done
  exit 0
fi

if [ -n "$SNAPSHOT_LABEL" ]; then
  require_plain_label --snapshot "$SNAPSHOT_LABEL"
  SNAPSHOT_SRC="$SNAPSHOTS_DIR/$SNAPSHOT_LABEL"
  SNAPSHOT_NAME="$SNAPSHOT_LABEL"
else
  SNAPSHOT_SRC="$CURRENT"
  SNAPSHOT_NAME="$(readlink "$CURRENT" 2>/dev/null || echo current)"
fi

HAVE_SNAPSHOT=0
if [ -e "$SNAPSHOT_SRC" ] && \
   [ -f "$SNAPSHOT_SRC/macos.img" ] && \
   [ -f "$SNAPSHOT_SRC/OpenCore.qcow2" ] && \
   [ -f "$SNAPSHOT_SRC/OVMF_VARS.fd" ]; then
  HAVE_SNAPSHOT=1
fi
if [ "$HAVE_SNAPSHOT" -eq 0 ]; then
  # A named snapshot that is missing or half-populated is an error in every
  # class. Falling through to the bootstrap path here would silently boot the
  # provisioned masters — a different guest than the one that was asked for,
  # and under --capture it would then repoint the rail's `current` at it.
  [ -z "$SNAPSHOT_LABEL" ] || die \
    "rail '$RAIL_NAME' has no usable snapshot '$SNAPSHOT_LABEL' at $SNAPSHOT_SRC
(needs macos.img, OpenCore.qcow2 and OVMF_VARS.fd; OVMF_CODE.fd is optional)
available: $(list_snapshot_labels | tr '\n' ' ')"
  [ "$BOOT_CLASS" = "capture" ] || die \
    "rail '$RAIL_NAME' has no snapshot yet — bootstrap it with:  vm/boot-x86.sh --rail $RAIL_NAME --capture
(boots the provisioned disk/OpenCore/OVMF_VARS writable for Setup Assistant + config; a clean
guest shutdown then captures the rail's first immutable snapshot. --testing/--interactive need a
snapshot to revert to.)"
  [ -f "$DISK_MASTER" ] || die "no provisioned disk at $DISK_MASTER"
  [ -f "$OPENCORE_MASTER" ] || die "no OpenCore image at $OPENCORE_MASTER"
  [ -f "$OVMF_VARS_MASTER" ] || die "no OVMF_VARS master at $OVMF_VARS_MASTER"
else
  # The code half of a split OVMF pair belongs to the snapshot that wrote the
  # vars half. Only adopt it when OVMF_CODE was left at the tree default: an
  # exported OVMF_CODE is an explicit override and outranks the snapshot.
  if [ "$OVMF_CODE" = "$OVMF_CODE_DEFAULT" ] && [ -f "$SNAPSHOT_SRC/OVMF_CODE.fd" ]; then
    OVMF_CODE="$SNAPSHOT_SRC/OVMF_CODE.fd"
  fi
fi
[ -f "$OVMF_CODE" ] || die "OVMF_CODE not found: $OVMF_CODE"

# metal2vulkan spawns `llvm-dis` and `spirv-val` on every uncached shader
# translate, and QEMU inherits this script's PATH — resolve them here so a
# missing toolchain fails now, not at the guest's first shader.
require_shader_toolchain() {
  command -v llvm-dis >/dev/null 2>&1 || die \
    "llvm-dis not found in PATH (install the LLVM tools, e.g. apt install llvm; versioned packages install llvm-dis-<N>, so symlink or add that bin dir to PATH)"
  command -v spirv-val >/dev/null 2>&1 || die \
    "spirv-val not found in PATH (ships in SPIRV-Tools, not LLVM: apt install spirv-tools)"
}

clone_file() {
  local src="$1" dst="$2"
  if cp --reflink=auto -f "$src" "$dst" 2>/dev/null; then
    return 0
  fi
  cp -f "$src" "$dst"
}

ensure_rust_tools() {
  if ! command -v cargo >/dev/null 2>&1 && [ -x "$HOME/.cargo/bin/cargo" ]; then
    export PATH="$HOME/.cargo/bin:$PATH"
  fi
  command -v cargo >/dev/null 2>&1 || die "cargo not found (needed to build reims-vgpu)"
}

build_reims_vgpu_efi() {
  [ -x "$REIMS_VGPU_EFI_ROM_SCRIPT" ] || die "EFI ROM builder not executable: $REIMS_VGPU_EFI_ROM_SCRIPT"
  echo "boot-x86.sh: building reims-vgpu-efi option ROM ..."
  "$REIMS_VGPU_EFI_ROM_SCRIPT" || die "reims-vgpu-efi build failed"
}

require_shader_toolchain
ensure_rust_tools
build_reims_vgpu_efi
# Product Linux x86 rail needs Vulkan. Override REIMS_VGPU_BACKEND only for an explicit
# alternate build.
if [ "$QEMU_BIN" = "$QEMU_BIN_DEFAULT" ]; then
  REIMS_VGPU_BACKEND="${REIMS_VGPU_BACKEND:-vulkan}"
  echo "boot-x86.sh: building in-tree QEMU (scripts/qemu-build --target x86_64 --backend $REIMS_VGPU_BACKEND) ..."
  "$REPO_ROOT/scripts/qemu-build/qemu-build.sh" --target x86_64 --backend "$REIMS_VGPU_BACKEND" \
    || die "qemu-build failed"
else
  # An overridden QEMU_BIN is a binary that already exists, and the reims-vgpu
  # crate is a *staticlib* linked into it (`ldd` on it names no reims object),
  # so rebuilding the crate here cannot change a single byte of what this boot
  # runs. This branch used to do that build anyway, via `(cd "" && cargo
  # build ...)` — a null `cd` that failed outright, which is why pinning
  # QEMU_BIN could never boot at all.
  #
  # Pinning is what makes it safe to edit the tree while a multi-boot harness
  # runs: the default branch above rebuilds QEMU every boot and would pick up a
  # half-finished edit mid-run (AGENTS.md records one run discarded for exactly
  # that). Keep this branch free of the tree.
  REIMS_VGPU_BACKEND="${REIMS_VGPU_BACKEND:-vulkan}"
  echo "boot-x86.sh: QEMU_BIN pinned ($QEMU_BIN) — not building; the staticlib is already linked in"
fi
[ -x "$QEMU_BIN" ] || die "QEMU not available: $QEMU_BIN"
[ -r /dev/kvm ] || die "KVM not available (/dev/kvm); is kvm loaded and are you in the kvm group?"

# --- Choose the boot disk: revert-clone, or bootstrap write-through -------------
mkdir -p "$RUN_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
SERIAL_LOG="$RUN_DIR/serial-$STAMP.log"
QMP_SOCK="$RUN_DIR/qmp-$STAMP.sock"
ln -sfn "qmp-$STAMP.sock" "$RUN_DIR/qmp.sock"

# --- Control-plane trace rail ---------------------------------------------------
TRACE="${TRACE:-0}"
TRACE_LOG=""
TRACE_SPEC=""
if [ "$TRACE" = "1" ]; then
  TRACE_LOG="$RUN_DIR/trace-$STAMP.log"
  if [ -n "${TRACE_EVENTS_FILE:-}" ]; then
    [ -f "$TRACE_EVENTS_FILE" ] || die "TRACE_EVENTS_FILE not found: $TRACE_EVENTS_FILE"
    TRACE_SPEC="events=$TRACE_EVENTS_FILE"
  else
    if [ -z "${TRACE_PATTERN:-}" ]; then
      case "$GFX_DEVICE" in
        reims-vgpu-pci) TRACE_PATTERN="reims_vgpu_pci_*" ;;
        *)                TRACE_PATTERN="vmware_vga*" ;;
      esac
    fi
    TRACE_SPEC="$TRACE_PATTERN"
  fi
fi

if [ "$HAVE_SNAPSHOT" -eq 0 ]; then
  DISK="$DISK_MASTER"
  OPENCORE="$OPENCORE_MASTER"
  OVMF_VARS="$OVMF_VARS_MASTER"
  IS_CLONE=0
  SNAPSHOT_NAME="(bootstrap)"
  echo "boot-x86.sh: rail '$RAIL_NAME' — bootstrap; booting provisioned masters write-through (rail is empty) ..."
else
  DISK="$RUN_DIR/macos-$STAMP.img"
  OPENCORE="$RUN_DIR/OpenCore-$STAMP.qcow2"
  OVMF_VARS="$RUN_DIR/OVMF_VARS-$STAMP.fd"
  IS_CLONE=1
  echo "boot-x86.sh: rail '$RAIL_NAME' — reverting to snapshot '$SNAPSHOT_NAME' ($SNAPSHOT_SRC) ..."
  clone_file "$SNAPSHOT_SRC/macos.img" "$DISK"
  clone_file "$SNAPSHOT_SRC/OpenCore.qcow2" "$OPENCORE"
  clone_file "$SNAPSHOT_SRC/OVMF_VARS.fd" "$OVMF_VARS"
  chmod u+w "$DISK" "$OPENCORE" "$OVMF_VARS"
fi

# --- Network -------------------------------------------------------------------
# SLIRP user-mode NAT; ipv6=off avoids a phantom IPv6 default that macOS prefers.
NET="${NET:-user}"
case "$NET" in
  user) NETDEV="user,id=net0,ipv6=off,hostfwd=tcp::${SSH_PORT}-:22" ;;
  none) NETDEV="" ;;
  *) die "unknown NET: $NET (user | none)" ;;
esac

# Product Reims VGPU: Tahoe x86 kext path is sensitive to high SMP (StorageNode::init).
if [ "$GFX_DEVICE" = "reims-vgpu-pci" ]; then
  if [ "${CPU_THREADS}" -gt 8 ] 2>/dev/null; then
    echo "boot-x86.sh: reims-vgpu-pci — capping SMP at 8 (was threads=$CPU_THREADS cores=$CPU_CORES)"
    CPU_THREADS=8
    CPU_CORES=8
    CPU_SOCKETS=1
  fi
fi

# --- Pick the audio backend -----------------------------------------------------
# Preference order, first that this build has. Native host backends come first
# because each runs its own thread with its own ring, so a late QEMU main loop
# costs latency there instead of silence; SDL is last because its callback fills
# the gap with zeroes and says nothing. See the AUDIO note in the header.
if [ -z "$AUDIODEV" ]; then
  case "$(uname -s)" in
    Darwin) AUDIO_PREFERENCE="coreaudio sdl none" ;;
    *)      AUDIO_PREFERENCE="pipewire pa alsa sdl none" ;;
  esac
  AUDIO_AVAILABLE="$("$QEMU_BIN" -audiodev help 2>/dev/null)"
  for candidate in $AUDIO_PREFERENCE; do
    if printf '%s\n' "$AUDIO_AVAILABLE" | grep -qx "  *$candidate" \
       || printf '%s\n' "$AUDIO_AVAILABLE" | grep -qx "$candidate"; then
      AUDIODEV="$candidate"
      break
    fi
  done
  # `none` is always compiled in, so this only fires if the query itself failed.
  AUDIODEV="${AUDIODEV:-none}"
fi

# --- Build the QEMU command line ------------------------------------------------
# q35 + OVMF + AppleSMC + SATA OpenCore/HDD. Display is attached below.
QEMU_ARGS=(
  -enable-kvm
  -m "$RAM"
  -cpu "${CPU_MODEL},-hle,-rtm,kvm=on,vendor=GenuineIntel,+invtsc,vmware-cpuid-freq=on,${CPU_OPTIONS}"
  -machine q35
  -smp "$CPU_THREADS",cores="$CPU_CORES",sockets="$CPU_SOCKETS"
  -device qemu-xhci,id=xhci
  -device usb-kbd,bus=xhci.0
  -device usb-tablet,bus=xhci.0
  -device usb-ehci,id=ehci
  -device isa-applesmc,osk="ourhardworkbythesewordsguardedpleasedontsteal(c)AppleComputerInc"
  -drive "if=pflash,format=raw,readonly=on,file=$OVMF_CODE"
  -drive "if=pflash,format=raw,file=$OVMF_VARS"
  -smbios type=2
  -audiodev "$AUDIODEV,id=audio0,out.buffer-length=$AUDIO_BUFFER_US"
  -device "usb-audio,bus=xhci.0,audiodev=audio0,buffer=$AUDIO_USB_BUFFER"
  -device ich9-ahci,id=sata
  -drive "id=OpenCoreBoot,if=none,format=qcow2,file=$OPENCORE"
  -device ide-hd,bus=sata.2,drive=OpenCoreBoot
  -drive "id=MacHDD,if=none,format=qcow2,file=$DISK"
  -device ide-hd,bus=sata.4,drive=MacHDD
  -qmp "unix:$QMP_SOCK,server=on,wait=off"
)

# Guest KP often reboots (even with OpenCore DB_HALT). Default exit so the GTK
# window disappears and serial stays under vm/disks/run/serial-*.log.
QEMU_REBOOT_ACTION="${QEMU_REBOOT_ACTION:-exit}"
case "$QEMU_REBOOT_ACTION" in
  exit)
    QEMU_ARGS+=(-action reboot=shutdown)
    ;;
  pause)
    QEMU_ARGS+=(-action reboot=shutdown,shutdown=pause)
    ;;
  reset)
    ;;
  *)
    die "unknown QEMU_REBOOT_ACTION: $QEMU_REBOOT_ACTION (exit|pause|reset)"
    ;;
esac

# Display attach.
case "$GFX_DEVICE" in
  reims-vgpu-pci)
    # Default: conventional pci-bridge secondary bus (IOFBIntegrated=No; OVMF maps BAR0).
    # Override: REIMS_VGPU_PCI_ATTACH=bus0 (root bus; IOFBIntegrated=Yes).
    REIMS_VGPU_PCI_ATTACH="${REIMS_VGPU_PCI_ATTACH:-pcibridge}"
    QEMU_ARGS+=(-vga none)
    # UEFI GOP on this same PCI device (BAR1 + option ROM) — never a second display.
    # Build: crates/reims-vgpu-efi/scripts/reims-vgpu-efi-rom/reims-vgpu-efi-rom.sh
    if [ -z "${REIMS_VGPU_GOP_ROM+x}" ]; then
      _reims_vgpu_gop_default="$REPO_ROOT/crates/reims-vgpu-efi/out/reims-vgpu-gop.rom"
      if [ -f "$_reims_vgpu_gop_default" ]; then
        REIMS_VGPU_GOP_ROM="$_reims_vgpu_gop_default"
      fi
    fi
    _reims_vgpu_dev="reims-vgpu-pci,id=reimsvgpu"
    if [ -n "${REIMS_VGPU_GOP_ROM:-}" ] && [ -f "$REIMS_VGPU_GOP_ROM" ]; then
      _reims_vgpu_dev="${_reims_vgpu_dev},romfile=${REIMS_VGPU_GOP_ROM},rombar=1"
      echo "boot-x86.sh: reims-vgpu-pci UEFI GOP romfile=$REIMS_VGPU_GOP_ROM"
    fi
    case "$REIMS_VGPU_PCI_ATTACH" in
      bus0)
        QEMU_ARGS+=(-device "${_reims_vgpu_dev},bus=pcie.0,addr=07.0")
        ;;
      pcibridge)
        QEMU_ARGS+=(
          -device pci-bridge,chassis_nr=5,id=pci.5,bus=pcie.0,addr=1e.0
          -device "${_reims_vgpu_dev},bus=pci.5,addr=00.0"
        )
        ;;
      *)
        die "unknown REIMS_VGPU_PCI_ATTACH: $REIMS_VGPU_PCI_ATTACH (pcibridge|bus0)"
        ;;
    esac
    ;;
  *)
    QEMU_ARGS+=(-device "$GFX_DEVICE")
    ;;
esac

if [ -n "$TRACE_LOG" ]; then
  QEMU_ARGS+=(-trace "$TRACE_SPEC" -D "$TRACE_LOG")
fi
if [ -n "$NETDEV" ]; then
  QEMU_ARGS+=(-netdev "$NETDEV" -device "virtio-net-pci,netdev=net0,id=net0,mac=$GUEST_MAC")
else
  QEMU_ARGS+=(-nic none)
fi

echo "boot-x86.sh: device=$GFX_DEVICE class=$BOOT_CLASS rail=$RAIL_NAME snapshot=$SNAPSHOT_NAME cpu=$CPU_MODEL smp=${CPU_THREADS},cores=${CPU_CORES} mem=$RAM reboot=${QEMU_REBOOT_ACTION}"
echo "boot-x86.sh: audiodev=$AUDIODEV out.buffer-length=${AUDIO_BUFFER_US}us usb-audio buffer=${AUDIO_USB_BUFFER}"
echo "boot-x86.sh: ssh → localhost:$SSH_PORT   serial → $SERIAL_LOG   qmp → $QMP_SOCK"
[ -n "$TRACE_LOG" ] && echo "boot-x86.sh: trace → $TRACE_LOG ($TRACE_SPEC)"

discard_clone() {
  if [ "${IS_CLONE:-1}" -eq 1 ]; then
    rm -f "$DISK" "$OPENCORE" "$OVMF_VARS"
  fi
  rm -f "$QMP_SOCK"
  # `qmp.sock` is the shared name every driver script resolves, and it is
  # re-pointed by whichever boot started last. A boot shutting down must only
  # remove it while it still names ITS socket: killing one VM and starting the
  # next immediately otherwise has the dying instance delete the live
  # instance's symlink, and the driver then fails with a bare ENOENT partway
  # through a run — which reads as a guest defect, not as a missing socket.
  if [ "$(readlink "$RUN_DIR/qmp.sock" 2>/dev/null)" = "qmp-$STAMP.sock" ]; then
    rm -f "$RUN_DIR/qmp.sock"
  fi
}

# Captures land in the SELECTED rail, next to the snapshot they descend from,
# and repoint only that rail's `current`. Nothing here touches `rails/current`:
# capturing on one guest line must not silently move what the next bare boot
# gets, which is the failure a flat snapshot namespace made easy.
promote_to_snapshot() {
  local label new_dir
  if [ "$HAVE_SNAPSHOT" -eq 0 ]; then
    label="$(date +%Y-%m-%d-%H%M%S)-base"
  else
    label="$(date +%Y-%m-%d-%H%M%S)-snap"
  fi
  new_dir="$SNAPSHOTS_DIR/$label"
  echo "boot-x86.sh: rail '$RAIL_NAME' — capturing new immutable snapshot '$label' ..."
  mkdir -p "$new_dir"
  clone_file "$DISK" "$new_dir/macos.img"
  clone_file "$OPENCORE" "$new_dir/OpenCore.qcow2"
  clone_file "$OVMF_VARS" "$new_dir/OVMF_VARS.fd"
  chmod 444 "$new_dir/macos.img" "$new_dir/OpenCore.qcow2" "$new_dir/OVMF_VARS.fd"
  # Carry the code half forward whenever it is not the tree default: the vars
  # just captured are only meaningful to the OVMF build that wrote them, and a
  # descendant that inherited only the vars would boot against the wrong pair.
  if [ "$OVMF_CODE" != "$OVMF_CODE_DEFAULT" ]; then
    clone_file "$OVMF_CODE" "$new_dir/OVMF_CODE.fd"
    chmod 444 "$new_dir/OVMF_CODE.fd"
  fi
  ln -sfn "$label" "$CURRENT"
  discard_clone
  echo "boot-x86.sh: snapshot '$label' captured; rail '$RAIL_NAME' current -> $label"
}

# --- Interactive / capture: foreground GUI, no time limit -----------------------
# Display backend: the supported reims-vgpu-pci path is the custom Rust host
# window (REIMS_VGPU_WINDOW=1), so QEMU defaults to `-display none` and owns no UI
# window. The older QEMU `gtk,gl=on` display path is deprecated for product work;
# keep it only for explicit archaeology / A/B debugging, not as an fps lever.
# Host-owned window is the default for reims-vgpu-pci: the
# staticlib opens its own winit + Vulkan window (present + input) and QEMU owns
# no window. Opt out with REIMS_VGPU_WINDOW=0 (falls back to QEMU's display). vmware-svga
# has no host window, so it is never defaulted on.
if [ "$GFX_DEVICE" = "reims-vgpu-pci" ]; then
  REIMS_VGPU_WINDOW="${REIMS_VGPU_WINDOW:-1}"
fi
# Normalize to presence semantics (the C shim reads getenv): unset = off,
# exported = on. 0/no/off/false disable; anything else enables.
case "${REIMS_VGPU_WINDOW:-}" in
  ''|0|no|off|false) unset REIMS_VGPU_WINDOW ;;
  *) export REIMS_VGPU_WINDOW=1 ;;
esac

if [ -n "${REIMS_VGPU_WINDOW:-}" ]; then
  REIMS_VGPU_DISPLAY="${REIMS_VGPU_DISPLAY:-none}"
  # winit (in the staticlib) needs a display connection to open the window, and
  # QEMU inherits these from this script's environment. When launched from a
  # minimal/agent shell they are often unset, so fall back to this host's known
  # session values (only when absent) and export them. XAUTHORITY carries a
  # per-login random suffix; override any of these in the environment if yours
  # differ (e.g. a different seat, DISPLAY, or Wayland socket).
  : "${XDG_RUNTIME_DIR:=/run/user/$(id -u)}"
  # winit's Linux backend picks Wayland whenever WAYLAND_DISPLAY is a non-empty
  # string, regardless of whether that socket actually exists (see winit
  # platform_impl/linux/mod.rs EventLoop::new). Defaulting it unconditionally
  # therefore forces Wayland (and a silent, unlogged EventLoopError) on
  # X11-only hosts. Only default it when a real Wayland socket is present.
  if [ -z "${WAYLAND_DISPLAY:-}" ] && [ -S "$XDG_RUNTIME_DIR/wayland-0" ]; then
    WAYLAND_DISPLAY="wayland-0"
  fi
  : "${DISPLAY:=:0}"
  # XAUTHORITY's suffix is a per-login random string, so it cannot be written
  # down: a hardcoded one goes stale at the next login and then points at a file
  # that does not exist. Discover the newest cookie in the runtime dir instead.
  if [ -z "${XAUTHORITY:-}" ]; then
    for _xauth in $(ls -t "$XDG_RUNTIME_DIR"/xauth_* 2>/dev/null); do
      XAUTHORITY="$_xauth"
      break
    done
  fi
  export XDG_RUNTIME_DIR WAYLAND_DISPLAY DISPLAY
  [ -n "${XAUTHORITY:-}" ] && export XAUTHORITY

  # A window with no display server still opens, still says "first frame
  # presented", and still fills the log with `host_window_cadence` — at a
  # present rate governed by nothing consuming the images. That reads exactly
  # like a pacing defect in this device. Measuring it once already produced a
  # peak of 33 Hz against a guest that believed it was drawing 119, so say so
  # here rather than letting a later reader believe the number.
  if command -v xdpyinfo >/dev/null 2>&1 && ! xdpyinfo >/dev/null 2>&1; then
    if [ ! -S "$XDG_RUNTIME_DIR/${WAYLAND_DISPLAY:-wayland-0}" ]; then
      echo "boot-x86.sh: WARNING — no usable display connection." >&2
      echo "boot-x86.sh:   DISPLAY=$DISPLAY XAUTHORITY=${XAUTHORITY:-unset}" >&2
      echo "boot-x86.sh:   The host window will open with nothing consuming it." >&2
      echo "boot-x86.sh:   Guest-side measurements stay valid; every host-window" >&2
      echo "boot-x86.sh:   number (host_window_cadence present_hz, busy_acquire," >&2
      echo "boot-x86.sh:   direct_frac) is MEANINGLESS on this boot." >&2
    fi
  fi
  # This once warned that host-window pacing was pinned at exactly 20.0 Hz
  # against `offered=51` with `busy_acquire=330`, and that FIFO->MAILBOX had not
  # moved it. That is fixed and the reasoning behind it was wrong, so the warning
  # is gone rather than softened.
  #
  # The swapchain was being created with a literal FIFO while the census line
  # printed the *chosen* MAILBOX, so "MAILBOX was granted and did not help" was
  # never a measurement of MAILBOX. One value now carries both the census field
  # and the call argument, and the ceiling went with it.
  #
  # Current: `present_hz == offered_hz` exactly, `busy_acquire=0`,
  # `direct_frac=1.00` — the window presents every frame it is offered, measured
  # up to 71.6 Hz. Whatever bounds the frame rate now is upstream of this window,
  # so do not go looking for it here.
  :
else
  REIMS_VGPU_DISPLAY="${REIMS_VGPU_DISPLAY:-gtk}"
fi

if [ "$BOOT_CLASS" = "interactive" ] || [ "$BOOT_CLASS" = "capture" ]; then
  # gtk display + serial multiplexed with the monitor on stdio (Apple EB logs on console).
  QEMU_ARGS+=(-display "$REIMS_VGPU_DISPLAY" -serial mon:stdio)
  rc=0
  "$QEMU_BIN" "${QEMU_ARGS[@]}" || rc=$?
  if [ "$BOOT_CLASS" = "capture" ] && [ "$rc" -eq 0 ]; then
    # mon:stdio does not fill SERIAL_LOG; promote on clean QEMU exit.
    promote_to_snapshot
  else
    [ "$BOOT_CLASS" = "capture" ] && echo "boot-x86.sh: qemu exited rc=$rc (not clean) — snapshot NOT updated"
    discard_clone
  fi
  exit "$rc"
fi

# --- Testing: background GUI + hard kill + capture-then-revert -------------------
QEMU_ARGS+=(-display "$REIMS_VGPU_DISPLAY" -serial "file:$SERIAL_LOG")

# Best-effort QMP register dump. Must never block hard-kill more than
# QMP_DUMP_TIMEOUT seconds (default 3). Unbounded `nc -U` hung testing boots
# after the 7-minute timer (kill was unreachable behind a wedged QMP).
QMP_DUMP_TIMEOUT="${QMP_DUMP_TIMEOUT:-3}"

qmp_dump_registers() {
  local out="$RUN_DIR/registers-$STAMP.txt"
  local watchdog_pid="" nc_pid=""
  if [ ! -S "$QMP_SOCK" ] || ! command -v nc >/dev/null 2>&1; then
    return 0
  fi
  if command -v timeout >/dev/null 2>&1; then
    # GNU coreutils timeout (Linux product rail).
    timeout --signal=KILL "${QMP_DUMP_TIMEOUT}s" sh -c "
      {
        printf '%s\\n' '{\"execute\":\"qmp_capabilities\"}'
        printf '%s\\n' '{\"execute\":\"human-monitor-command\",\"arguments\":{\"command-line\":\"info registers -a\"}}'
        sleep 0.3
      } | nc -U \"\$1\"
    " sh "$QMP_SOCK" >"$out" 2>/dev/null || true
    return 0
  fi
  # Portable fallback (macOS / no timeout): background nc + watchdog kill.
  {
    printf '{"execute":"qmp_capabilities"}\n'
    printf '{"execute":"human-monitor-command","arguments":{"command-line":"info registers -a"}}\n'
    sleep 0.3
  } | nc -U "$QMP_SOCK" >"$out" 2>/dev/null &
  nc_pid=$!
  (
    sleep "$QMP_DUMP_TIMEOUT"
    kill -9 "$nc_pid" 2>/dev/null || true
  ) &
  watchdog_pid=$!
  wait "$nc_pid" 2>/dev/null || true
  kill "$watchdog_pid" 2>/dev/null || true
  wait "$watchdog_pid" 2>/dev/null || true
}

kill_qemu() {
  if [ -z "${QEMU_PID:-}" ]; then
    return 0
  fi
  if ! kill -0 "$QEMU_PID" 2>/dev/null; then
    return 0
  fi
  echo "boot-x86.sh: killing qemu pid=$QEMU_PID"
  kill -TERM "$QEMU_PID" 2>/dev/null || true
  sleep 2
  if kill -0 "$QEMU_PID" 2>/dev/null; then
    kill -KILL "$QEMU_PID" 2>/dev/null || true
  fi
  # Reap so wait loops exit cleanly.
  wait "$QEMU_PID" 2>/dev/null || true
}

capture_then_revert() {
  local reason="$1"
  echo "boot-x86.sh: capture-then-revert ($reason)"
  # Dump first (bounded), then always kill — never gate kill on QMP success.
  qmp_dump_registers
  kill_qemu
  discard_clone
  echo "boot-x86.sh: reverted (clone discarded); evidence in $RUN_DIR (serial-$STAMP.log)"
}

"$QEMU_BIN" "${QEMU_ARGS[@]}" &
QEMU_PID=$!
trap 'capture_then_revert signal; exit 130' INT TERM

# A boot that boot.efi aborted is not a wedge, and it must not be scored as one.
#
# Observed shape, twice: QEMU stays alive, the guest never issues a single GPU
# command, the host window shows the macOS boot-failure graphic, and the fail log
# fills with nothing but `host_window_cadence`. That is indistinguishable from a
# device hang by every signal an agent normally reads, and one session lost 53
# minutes to it before anyone opened the serial log. The serial log says it
# plainly:
#
#   AAPL: #[EB.MM.AKMR|!] Err(0xE) <- EB.M.BAPr2 2 2 50271 0x700000
#   AAPL: #[EB.B.MN|!]    Err(0xE) <- EB.MM.AKMR
#   AAPL: #[EB|STOP] 0x15
#   OC: Boot failed - Aborted
#
# boot.efi could not get the contiguous kernel region it asks for at a fixed
# guest-physical address, so it stopped before loading the kernel. Nothing after
# that point is a reading about this device.
#
# Detect it and exit distinctly (125), so a batch can retry the boot instead of
# waiting out TESTING_TIMEOUT and then treating the sample as data.
BOOT_ABORT_RE='#\[EB\|STOP\]|Boot failed - Aborted'

# A guest KERNEL PANIC is the loudest result this rig can produce and it was, for
# months, the quietest. `-action reboot=shutdown` turns the panic reboot into a
# QEMU exit, so the boot script reported "qemu exited", the drive script reported
# "VM IS GONE", and nothing anywhere said the word panic. The evidence was on disk
# the whole time — `vm/disks/run/serial-*.log` is never cleaned, and a sweep of
# 547 of them found 11 panics (2.0 %) whose reasons are a corruption census:
#
#   [kalloc.type.var6.6144]: element modified after free (val:0xffffffffffffffff)
#   [kalloc.type.var3.256]:  element modified after free (val:0xffffffffffffffff)
#   pmap_page_protect() pmap=... pn=... vaddr=...            (x2)
#   Kernel trap at 0xffffffffffffffff  (an indirect call through a clobbered
#                                       ifnet function pointer)
#   "hitting assertion" @AppleParavirtPageTable.cpp:200
#
# The two `element modified after free` reports are the important ones: the
# kernel's own poison check found a whole freed element (256 B and 6144 B) filled
# with 0xFF from offset 0. That is not a stray pointer, it is a bulk write of
# opaque white pixels into memory the guest kernel had already freed — the
# kernel-side face of the write-after-fence class `311cb11` repaired.
#
# So say it at the moment it happens, and exit 126 so a batch can tell a panic
# from a firmware abort (125), a wedge (124) and a clean exit.
PANIC_RE='Debugger called: <panic>'
PANIC_KEEP_DIR="${PANIC_KEEP_DIR:-/tmp/reims-vgpu-panics}"

# A panic is reachable from two places — the poll loop below, and the path after
# QEMU exits on its own, because `-action reboot=shutdown` turns the panic reboot
# into an exit that otherwise reads as a clean guest shutdown. Both say the same
# thing about the same log, so both go through these two functions. They were
# written out twice, and a second copy of a report is exactly where the keep-dir,
# the exit code or the `-A2` drift apart without either arm looking wrong.
serial_has_panic() {
  [ -s "$SERIAL_LOG" ] && grep -qF "$PANIC_RE" "$SERIAL_LOG" 2>/dev/null
}

report_panic_and_revert() {
  local kept
  kept="$PANIC_KEEP_DIR/$(basename "$SERIAL_LOG")"
  echo "boot-x86.sh: $1"
  grep -A2 -F "$PANIC_RE" "$SERIAL_LOG" 2>/dev/null | head -6
  mkdir -p "$PANIC_KEEP_DIR"
  cp -f "$SERIAL_LOG" "$kept" 2>/dev/null || true
  echo "boot-x86.sh: serial log kept at $kept"
  capture_then_revert "guest kernel panic"
}

elapsed=0
while kill -0 "$QEMU_PID" 2>/dev/null; do
  if [ "$elapsed" -ge "$TESTING_TIMEOUT" ]; then
    capture_then_revert "timeout ${TESTING_TIMEOUT}s — wedge verdict"
    exit 124
  fi
  if [ -s "$SERIAL_LOG" ] && grep -qE "$BOOT_ABORT_RE" "$SERIAL_LOG" 2>/dev/null; then
    echo "boot-x86.sh: GUEST FIRMWARE ABORTED THE BOOT — the kernel never loaded."
    grep -E "$BOOT_ABORT_RE|EB\.MM\.AKMR|EB\.B\.MN" "$SERIAL_LOG" 2>/dev/null | tail -5
    echo "boot-x86.sh: no measurement from this boot is about the device. Retry it."
    capture_then_revert "boot.efi aborted — firmware boot failure, not a device wedge"
    exit 125
  fi
  if serial_has_panic; then
    report_panic_and_revert "GUEST KERNEL PANIC."
    exit 126
  fi
  sleep 5
  elapsed=$((elapsed + 5))
done

wait "$QEMU_PID" 2>/dev/null || true
# QEMU exiting on its own is normally a guest shutdown, but `-action
# reboot=shutdown` makes a panic reboot look identical. Check before saying
# nothing happened.
if serial_has_panic; then
  report_panic_and_revert "GUEST KERNEL PANIC (qemu exited on the panic reboot)."
  exit 126
fi
capture_then_revert "qemu exited"
