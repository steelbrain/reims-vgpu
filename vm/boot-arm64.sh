#!/usr/bin/env bash
#
# vm/boot-arm64.sh — boot an arm64 macOS guest under QEMU's vmapple machine (HVF) on the Mac.
#
# Display is selected by --device (machine property gfx-device):
#   apple-gfx-mmio     Apple ParavirtualizedGraphics.framework (reference, default)
#   reims-vgpu-mmio    product thin C → crates/reims-vgpu Rust staticlib
# The vmapple machine creates exactly one device at the fixed Reims vGPU GFX/IOSFC
# addresses — do not add a second display via -device.
#
# RAILS. A rail is one guest OS line with a history of its own. Rails are
# siblings under `vm/guest/rails/`, and `rails/current` is a symlink naming the
# one a boot gets when `--rail` is not given:
#
#   vm/guest/rails/<rail>/snapshots/<label>/{disk.img,aux.img.trimmed}
#   vm/guest/rails/<rail>/snapshots/current -> <label>
#   vm/guest/rails/<rail>/vm.json            (optional; else the bundle's)
#   vm/guest/rails/current -> <rail>
#
# Snapshots are per-rail because they are not comparable across rails: two guest
# OS lines share no history, and a single flat namespace makes `current` mean
# "whichever guest was captured last", which is how a measurement ends up
# attributed to the wrong OS. Two coordinates, `--rail` and `--snapshot`, each
# with its own `current`, keep that from being expressible.
#
# A rail may carry its own `vm.json`. The ECID in it identifies the machine the
# guest was personalized against, so a second provisioned guest is a second
# vm.json, not a second disk under the first one's identity. When the rail ships
# one it wins over `$GUEST_DIR/vm.json`; a single-guest tree ships none and uses
# the bundle's, exactly as before.
#
# SNAPSHOT-REVERT: within a rail, snapshots form an IMMUTABLE HISTORY (each file
# read-only, never overwritten). EVERY boot starts from a byte-identical APFS
# clone of the selected snapshot (clonefile: instant, COW) and discards that
# clone on exit, so a harsh kill or a wedge costs nothing and poisons nothing.
# A snapshot is never booted directly.
#
# Selection by either coordinate is per-boot and repoints no `current` symlink.
#
# Boot classes:
#   --testing      agent-driven measurement (default): GUI + serial-to-file,
#                  SSH-driven, 7-minute hard kill + capture-then-revert. Reverts.
#   --interactive  human/GUI boot, no time limit. Reverts (nothing persists).
#   --capture      boot writable to CAPTURE A NEW snapshot: on a clean guest
#                  shutdown the modified disk/aux are saved as a NEW immutable
#                  snapshot and `current` is repointed to it. Existing snapshots
#                  (incl. the base) are never touched. Roll back by repointing
#                  `current` (see scripts/vmapple-snapshot).
#                  A bare `--snapshot` (no label) still means this.
#
# Launch configuration is CLI flags / env here, not device/backend code.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Configuration (override via env or flags) ----------------------------------
# Guest bundle provisioned by scripts/vmapple-provision (large + private, gitignored).
GUEST_DIR="${GUEST_DIR:-$REPO_ROOT/vm/guest}"
# Guest OS lines, each with its own immutable snapshot history.
RAILS_DIR="${RAILS_DIR:-$GUEST_DIR/rails}"
# Per-boot scratch (clones + logs). Same APFS volume as GUEST_DIR for clonefile.
# Shared across rails on purpose: the clones are stamped and thrown away, and
# `run/qmp.sock` is the one path every driver script resolves. Splitting it per
# rail would leave those scripts pointing at whichever rail booted last.
RUN_DIR="${RUN_DIR:-$GUEST_DIR/run}"

QEMU_BIN_DEFAULT="$REPO_ROOT/vendor/qemu/build/qemu-system-aarch64"
QEMU_BIN="${QEMU_BIN:-$QEMU_BIN_DEFAULT}"
REIMS_VGPU_EFI_ROM_SCRIPT="$REPO_ROOT/crates/reims-vgpu-efi/scripts/reims-vgpu-efi-rom/reims-vgpu-efi-rom.sh"
AVPBOOTER="${AVPBOOTER:-/System/Library/Frameworks/Virtualization.framework/Resources/AVPBooter.vmapple2.bin}"

RAM="${RAM:-8G}"
CPUS="${CPUS:-4}"
SSH_PORT="${SSH_PORT:-2222}"
TESTING_TIMEOUT="${TESTING_TIMEOUT:-420}" # 7-minute hard kill for testing boots
# PIN the guest NIC MAC. Without a fixed MAC, QEMU assigns a random one each boot,
# so a reverted snapshot shows macOS a brand-new unconfigured interface → no DHCP
# lease → broken networking + unreachable sshd. A stable MAC keeps the guest's
# saved network service valid across reverts.
GUEST_MAC="${GUEST_MAC:-52:54:00:76:61:70}"

BOOT_CLASS="testing"     # testing | interactive | capture
RAIL_LABEL="${RAIL:-}"   # empty = follow rails/current; else a rail name
SNAPSHOT_LABEL=""        # empty = follow the rail's snapshots/current
LIST_RAILS=0
LIST_SNAPSHOTS=0
GFX_DEVICE="apple-gfx-mmio"  # apple-gfx-mmio | reims-vgpu-mmio

usage() {
  cat <<EOF
usage: vm/boot-arm64.sh [--device apple-gfx-mmio|reims-vgpu-mmio] [--testing|--interactive|--capture]
                        [--rail NAME] [--snapshot LABEL]

  --device NAME          Reims vGPU slot backend (default: apple-gfx-mmio)
                         apple-gfx-mmio  Apple PVG framework (reference)
                         reims-vgpu-mmio    product (reims-vgpu Rust path)
  --testing              agent boot (default): GUI, ${TESTING_TIMEOUT}s hard kill, reverts
  --interactive          human/GUI boot, no time limit, reverts
  --capture              boot writable; a clean guest shutdown CAPTURES a new snapshot
                         into the selected rail (also bootstraps an empty rail)
  --rail NAME            guest OS line to boot. Default: whatever \`rails/current\` names.
  --snapshot LABEL       snapshot WITHIN that rail. Default: the rail's own
                         \`snapshots/current\`. A bare --snapshot (no label) is
                         the old spelling of --capture.
  --list-rails           print the rails and exit
  --list-snapshots       print the selected rail's snapshots and exit

Both selections are per-boot and repoint no \`current\`. Layout:
  $RAILS_DIR/<rail>/snapshots/<label>/{disk.img,aux.img.trimmed}
Change the default rail with:  ln -sfn <rail> $RAILS_DIR/current
Always builds reims-vgpu-efi and reims-vgpu before boot. In-tree QEMU is rebuilt
unless QEMU_BIN is set to something other than the default path.
Env: GUEST_DIR RAILS_DIR RAIL RUN_DIR QEMU_BIN AVPBOOTER RAM CPUS SSH_PORT
     TESTING_TIMEOUT QMP_DUMP_TIMEOUT GUEST_MAC
     NET=user (SLIRP, default) | NET=none (no NIC — one-time offline Setup Assistant bootstrap)
     TRACE=1 — control-plane trace rail: the display device's QEMU trace events
     (MMIO order, ring records, map/unmap, IRQs, frames) → \$RUN_DIR/trace-<stamp>.log
     TRACE_PATTERN=glob — override the default display-device trace glob
     TRACE_EVENTS_FILE=path — QEMU trace event list file; overrides TRACE_PATTERN
EOF
}

# `--device NAME` and `--device=NAME` differ only in where the value comes from,
# so the accepted set and its error string live here once. Written out per arm,
# adding a device means editing two `case`s and two message strings, and the arm
# that gets missed is the spelling nobody on this pathway happens to type.
set_gfx_device() {
  GFX_DEVICE="$1"
  case "$GFX_DEVICE" in
    apple-gfx-mmio|reims-vgpu-mmio) ;;
    *)
      echo "boot-arm64.sh: invalid --device '$GFX_DEVICE' (apple-gfx-mmio | reims-vgpu-mmio)" >&2
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
    --rail) shift; RAIL_LABEL="${1:-}"; [ -n "$RAIL_LABEL" ] || { echo "boot-arm64.sh: --rail needs a name" >&2; exit 64; }; shift ;;
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
    *) echo "boot-arm64.sh: unknown arg: $1" >&2; usage >&2; exit 64 ;;
  esac
done

# --- Preflight ------------------------------------------------------------------
die() { echo "boot-arm64.sh: $*" >&2; exit 1; }

# Directory children of a dir; a `current` symlink is -type l, so it is skipped
# and never lists itself as one of the things it points at. BSD find has no
# -printf, so the label is taken with basename.
list_dir_labels() {
  find "$1" -mindepth 1 -maxdepth 1 -type d 2>/dev/null \
    | while IFS= read -r d; do basename "$d"; done | sort
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

# A tree from before rails keeps its history one level too high. Say so with the
# move that fixes it rather than migrating it here: this is the user's only copy
# of a provisioned guest, and a boot script is the wrong thing to have silently
# rearranged it when the next thing it does is fail for an unrelated reason.
if [ ! -d "$RAILS_DIR" ] && [ -d "$GUEST_DIR/snapshots" ]; then
  die "pre-rail layout found at $GUEST_DIR/snapshots.
Snapshots are now per-rail. Move that history into a rail — name it for the guest OS line it holds:
  mkdir -p $RAILS_DIR/<rail>
  mv $GUEST_DIR/snapshots $RAILS_DIR/<rail>/snapshots
  ln -sfn <rail> $RAILS_DIR/current"
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
 then bootstrap it with:  vm/boot-arm64.sh --rail $RAIL_NAME --capture)"

# The ECID identifies the machine this guest was personalized against, so it
# belongs to the rail when the rail has its own. A single-guest tree has none
# and uses the bundle's, which is the pre-rail behavior unchanged.
VM_JSON="$GUEST_DIR/vm.json"
[ -f "$RAIL_DIR/vm.json" ] && VM_JSON="$RAIL_DIR/vm.json"

# --- Resolve the snapshot within that rail ---------------------------------------
# When the rail has none, only --capture can bootstrap it: it boots the freshly
# provisioned disk WRITE-THROUGH so you can finish Setup Assistant + config, and
# a clean guest shutdown captures the rail's first immutable snapshot.
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
if [ -e "$SNAPSHOT_SRC" ] && [ -f "$SNAPSHOT_SRC/disk.img" ] && [ -f "$SNAPSHOT_SRC/aux.img.trimmed" ]; then
  HAVE_SNAPSHOT=1
fi
if [ "$HAVE_SNAPSHOT" -eq 0 ]; then
  # A named snapshot that is missing or half-populated is an error in every
  # class. Falling through to the bootstrap path here would silently boot the
  # provisioned bundle — a different guest than the one that was asked for, and
  # under --capture it would then repoint the rail's `current` at it.
  [ -z "$SNAPSHOT_LABEL" ] || die \
    "rail '$RAIL_NAME' has no usable snapshot '$SNAPSHOT_LABEL' at $SNAPSHOT_SRC
(needs disk.img and aux.img.trimmed)
available: $(list_snapshot_labels | tr '\n' ' ')"
  [ "$BOOT_CLASS" = "capture" ] || die \
    "rail '$RAIL_NAME' has no snapshot yet — bootstrap it with:  vm/boot-arm64.sh --rail $RAIL_NAME --capture
(boots the provisioned disk writable for Setup Assistant + config; a clean guest shutdown then
captures the rail's first immutable snapshot. --testing/--interactive need a snapshot to revert to.)"
  [ -f "$GUEST_DIR/disk.img" ] && [ -f "$GUEST_DIR/aux.img.trimmed" ] \
    || die "no provisioned bundle at $GUEST_DIR (run scripts/vmapple-provision first)"
fi

# metal2vulkan spawns `llvm-dis` and `spirv-val` on every uncached shader
# translate, and QEMU inherits this script's PATH — resolve them here so a
# missing toolchain fails now, not at the guest's first shader.
require_shader_toolchain() {
  # Homebrew keeps llvm keg-only, so a stock macOS box HAS llvm-dis and does
  # not have it on PATH — the gate below would refuse a host that is in fact
  # fully provisioned. Adopt it instead of refusing, and resolve it through
  # `brew --prefix` rather than a hardcoded path: the prefix is /opt/homebrew
  # on Apple Silicon and /usr/local on Intel. Prepending is the operative part
  # rather than merely locating it, because QEMU inherits this PATH and
  # metal2vulkan spawns the tool from it on every uncached translate.
  if ! command -v llvm-dis >/dev/null 2>&1 && command -v brew >/dev/null 2>&1; then
    local llvm_prefix
    llvm_prefix="$(brew --prefix llvm 2>/dev/null || true)"
    if [ -n "$llvm_prefix" ] && [ -x "$llvm_prefix/bin/llvm-dis" ]; then
      export PATH="$llvm_prefix/bin:$PATH"
    fi
  fi
  command -v llvm-dis >/dev/null 2>&1 || die \
    "llvm-dis not found in PATH (install the LLVM tools, e.g. brew install llvm, then put \"\$(brew --prefix llvm)/bin\" on PATH)"
  command -v spirv-val >/dev/null 2>&1 || die \
    "spirv-val not found in PATH (ships in SPIRV-Tools, not LLVM: brew install spirv-tools)"
}

# APFS clonefile: instant and COW, which is what makes a per-boot revert free.
# `cp -c` fails on a non-APFS volume (or a cross-volume copy), so fall back to a
# real copy rather than leaving the boot without a disk. Written out at each of
# the four call sites, one of them eventually loses the fallback and a guest
# bundle on an external volume stops booting for a reason nothing prints.
clone_file() {
  local src="$1" dst="$2"
  if cp -c "$src" "$dst" 2>/dev/null; then
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
  echo "boot-arm64.sh: building reims-vgpu-efi option ROM ..."
  "$REIMS_VGPU_EFI_ROM_SCRIPT" || die "reims-vgpu-efi build failed"
}

require_shader_toolchain
ensure_rust_tools
build_reims_vgpu_efi
if [ "$QEMU_BIN" = "$QEMU_BIN_DEFAULT" ]; then
  echo "boot-arm64.sh: building in-tree QEMU (scripts/qemu-build --target aarch64) ..."
  "$REPO_ROOT/scripts/qemu-build/qemu-build.sh" --target aarch64 \
    || die "qemu-build failed"
else
  # See the matching note in boot-x86.sh: an overridden QEMU_BIN already has the
  # reims-vgpu staticlib linked into it, so rebuilding the crate cannot affect
  # this boot. The build this replaced was `(cd "" && cargo build ...)`, a null
  # `cd` that failed outright and made a pinned QEMU_BIN unbootable.
  echo "boot-arm64.sh: QEMU_BIN pinned ($QEMU_BIN) — not building; the staticlib is already linked in"
fi

[ -x "$QEMU_BIN" ] || die "QEMU not available: $QEMU_BIN"
[ -f "$AVPBOOTER" ] || die "AVPBooter ROM not found: $AVPBOOTER"
[ -f "$VM_JSON" ] || die "guest vm.json not found: $VM_JSON (provision first)"

# ECID/UUID: vmapple's uuid= is the ECID from the bundle's machineId (== macosvm
# contrib/vmapple/uuid.sh). Extract it from vm.json.
UUID="$(plutil -extract machineId raw "$VM_JSON" | base64 -d | plutil -extract ECID raw -)"
[ -n "$UUID" ] || die "could not extract ECID/UUID from $VM_JSON"

# --- Choose the boot disk: revert-clone, or bootstrap write-through -------------
mkdir -p "$RUN_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
SERIAL_LOG="$RUN_DIR/serial-$STAMP.log"
QMP_SOCK="$RUN_DIR/qmp-$STAMP.sock"
# Stable alias to the live boot's QMP socket (scripts/qmp): QMP_SOCK=vm/guest/run/qmp.sock
ln -sfn "qmp-$STAMP.sock" "$RUN_DIR/qmp.sock"

# --- Control-plane trace rail ---------------------------------------------------
# TRACE=1 enables the display device's QEMU trace events — protocol records ONLY
# (MMIO access order, ring/FIFO records, map/unmap requests, IRQ raises, frames,
# mode changes), never referenced guest-memory content. This is the sanctioned
# boot-script switch for tracing.
TRACE="${TRACE:-0}"
TRACE_LOG=""
TRACE_SPEC=""
if [ "$TRACE" = "1" ]; then
  TRACE_LOG="$RUN_DIR/trace-$STAMP.log"
  if [ -n "${TRACE_EVENTS_FILE:-}" ]; then
    [ -f "$TRACE_EVENTS_FILE" ] || die "TRACE_EVENTS_FILE not found: $TRACE_EVENTS_FILE"
    TRACE_SPEC="events=$TRACE_EVENTS_FILE"
  else
    # TRACE_PATTERN may be overridden in env; default matches the selected device.
    if [ -z "${TRACE_PATTERN:-}" ]; then
      case "$GFX_DEVICE" in
        reims-vgpu-mmio) TRACE_PATTERN="reims_vgpu_mmio_*" ;;
        *)            TRACE_PATTERN="apple_gfx_*" ;;
      esac
    fi
    TRACE_SPEC="$TRACE_PATTERN"
  fi
fi

if [ "$HAVE_SNAPSHOT" -eq 0 ]; then
  # Bootstrap (--capture only): boot the provisioned master write-through so
  # Setup Assistant + config persist; a clean shutdown captures snapshot #1.
  DISK="$GUEST_DIR/disk.img"; AUX="$GUEST_DIR/aux.img.trimmed"; IS_CLONE=0
  SNAPSHOT_NAME="(bootstrap)"
  echo "boot-arm64.sh: rail '$RAIL_NAME' — bootstrap; booting provisioned disk write-through (rail is empty) ..."
else
  # Revert: clone the selected snapshot into a throwaway working copy.
  DISK="$RUN_DIR/disk-$STAMP.img"; AUX="$RUN_DIR/aux-$STAMP.img"; IS_CLONE=1
  echo "boot-arm64.sh: rail '$RAIL_NAME' — reverting to snapshot '$SNAPSHOT_NAME' ($SNAPSHOT_SRC) ..."
  clone_file "$SNAPSHOT_SRC/disk.img" "$DISK"
  clone_file "$SNAPSHOT_SRC/aux.img.trimmed" "$AUX"
  chmod u+w "$DISK" "$AUX"   # snapshots are read-only; the working clone must be writable
fi

# --- Network -------------------------------------------------------------------
# NET=user (default): QEMU SLIRP user-mode NAT — no privileges, real outbound
#   TCP/UDP+DNS (verified reaching Apple over HTTPS), SSH via hostfwd. ipv6=off
#   per QEMU's own vmapple.rst reference invocation: SLIRP's fec0::/64 RA
#   otherwise gives the guest a phantom IPv6 default that macOS prefers and that
#   goes nowhere, so outbound traffic (DNS first) stalls before falling back to v4.
# NET=none: no NIC at all (passes -nic none). Note QEMU adds a DEFAULT
#   user-mode NIC when no -netdev/-nic is given, so genuinely disabling the
#   network requires an explicit -nic none — omitting the netdev is not enough.
#   Rarely needed: Setup Assistant completes fine online (its Device-Enrollment
#   pane stall is non-deterministic and clears on its own).
NET="${NET:-user}"
case "$NET" in
  user) NETDEV="user,id=net0,ipv6=off,hostfwd=tcp::${SSH_PORT}-:22" ;;
  none) NETDEV="" ;;
  *) die "unknown NET: $NET (user | none)" ;;
esac

# --- Build the QEMU command line ------------------------------------------------
# Per docs/system/arm/vmapple.rst: aux + disk each as pflash (pre-boot env) AND as
# virtio-blk (runtime), plus the AVPBooter ROM as -bios and -M vmapple,uuid=ECID.
QEMU_ARGS=(
  -m "$RAM"
  -accel hvf
  -smp "$CPUS"
  -M "vmapple,uuid=$UUID,gfx-device=$GFX_DEVICE"
  -bios "$AVPBOOTER"
  -drive "file=$AUX,if=pflash,format=raw"
  -drive "file=$DISK,if=pflash,format=raw"
  -drive "file=$AUX,if=none,id=aux,format=raw"
  -drive "file=$DISK,if=none,id=root,format=raw"
  -device vmapple-virtio-blk-pci,variant=aux,drive=aux
  -device vmapple-virtio-blk-pci,variant=root,drive=root
  -qmp "unix:$QMP_SOCK,server=on,wait=off"
)
if [ -n "$TRACE_LOG" ]; then
  QEMU_ARGS+=(-trace "$TRACE_SPEC" -D "$TRACE_LOG")
fi
if [ -n "$NETDEV" ]; then
  QEMU_ARGS+=(-netdev "$NETDEV" -device "virtio-net-pci,netdev=net0,mac=$GUEST_MAC")
else
  QEMU_ARGS+=(-nic none)   # suppress QEMU's implicit default user-mode NIC
fi

# The product build owns its AppKit window in Rust and therefore disables QEMU's
# Cocoa display. The Apple reference build retains Cocoa.
DISPLAY_KIND="cocoa"
if [ "$GFX_DEVICE" = "reims-vgpu-mmio" ]; then
  DISPLAY_KIND="reims-host-window"
  VULKAN_LOADER_DIR="/opt/homebrew/opt/vulkan-loader/lib"
  MOLTENVK_ICD="/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json"
  [ -d "$VULKAN_LOADER_DIR" ] || die \
    "Vulkan loader not found: $VULKAN_LOADER_DIR (install Homebrew vulkan-loader)"
  [ -f "$MOLTENVK_ICD" ] || die \
    "MoltenVK ICD not found: $MOLTENVK_ICD (install Homebrew molten-vk)"
  export DYLD_FALLBACK_LIBRARY_PATH="$VULKAN_LOADER_DIR${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
  export VK_ICD_FILENAMES="$MOLTENVK_ICD"
fi

echo "boot-arm64.sh: device=$GFX_DEVICE class=$BOOT_CLASS rail=$RAIL_NAME snapshot=$SNAPSHOT_NAME uuid=$UUID"
echo "boot-arm64.sh: display=$DISPLAY_KIND"
echo "boot-arm64.sh: ssh → localhost:$SSH_PORT   serial → $SERIAL_LOG   qmp → $QMP_SOCK"
[ -n "$TRACE_LOG" ] && echo "boot-arm64.sh: trace → $TRACE_LOG ($TRACE_SPEC)"

# Discard the per-boot working clone. Never deletes the provisioned master (used
# write-through during bootstrap), only a RUN_DIR clone.
discard_clone() {
  [ "${IS_CLONE:-1}" -eq 1 ] && rm -f "$DISK" "$AUX"
  rm -f "$QMP_SOCK"
  # Only drop the shared alias while it still names THIS boot's socket — same
  # guard as boot-x86.sh. A dying instance that deletes the live instance's
  # symlink makes the next driver run fail on a missing socket partway through,
  # which is indistinguishable from a guest defect in the captures it leaves.
  if [ "$(readlink "$RUN_DIR/qmp.sock" 2>/dev/null)" = "qmp-$STAMP.sock" ]; then
    rm -f "$RUN_DIR/qmp.sock"
  fi
}

promote_to_snapshot() {
  # Save this boot's (modified) disk/aux as a NEW immutable snapshot in the
  # SELECTED rail, and repoint only that rail's `current`. Existing snapshots
  # (incl. the base) are never overwritten, and `rails/current` is not touched:
  # capturing on one guest line must not silently move what the next bare boot
  # gets, which is the failure a flat snapshot namespace made easy.
  # Called only after a clean guest shutdown in --capture mode.
  local label new_dir
  if [ "$HAVE_SNAPSHOT" -eq 0 ]; then
    label="$(date +%Y-%m-%d-%H%M%S)-base"
  else
    label="$(date +%Y-%m-%d-%H%M%S)-snap"
  fi
  new_dir="$SNAPSHOTS_DIR/$label"
  echo "boot-arm64.sh: rail '$RAIL_NAME' — capturing new immutable snapshot '$label' ..."
  mkdir -p "$new_dir"
  clone_file "$DISK" "$new_dir/disk.img"
  clone_file "$AUX" "$new_dir/aux.img.trimmed"
  chmod 444 "$new_dir/disk.img" "$new_dir/aux.img.trimmed"
  ln -sfn "$label" "$CURRENT"
  discard_clone
  echo "boot-arm64.sh: snapshot '$label' captured; rail '$RAIL_NAME' current -> $label"
}

# --- Interactive / capture: foreground GUI, no time limit -----------------------
if [ "$BOOT_CLASS" = "interactive" ] || [ "$BOOT_CLASS" = "capture" ]; then
  if [ "$DISPLAY_KIND" = "reims-host-window" ]; then
    QEMU_ARGS+=(-display none -serial mon:stdio)
  else
    QEMU_ARGS+=(-display cocoa -serial mon:stdio)
  fi
  rc=0
  "$QEMU_BIN" "${QEMU_ARGS[@]}" || rc=$?
  if [ "$BOOT_CLASS" = "capture" ] && [ "$rc" -eq 0 ]; then
    promote_to_snapshot
  else
    [ "$BOOT_CLASS" = "capture" ] && echo "boot-arm64.sh: qemu exited rc=$rc (not clean) — snapshot NOT updated"
    discard_clone
  fi
  exit "$rc"
fi

# --- Testing: background GUI + hard kill + capture-then-revert -------------------
if [ "$DISPLAY_KIND" = "reims-host-window" ]; then
  QEMU_ARGS+=(-display none -serial "file:$SERIAL_LOG")
else
  QEMU_ARGS+=(-display cocoa -serial "file:$SERIAL_LOG")
fi

# Best-effort QMP register dump. Must never block hard-kill more than
# QMP_DUMP_TIMEOUT seconds (default 3). Unbounded `nc -U` can hang testing
# boots after the timer (kill was unreachable behind a wedged QMP).
QMP_DUMP_TIMEOUT="${QMP_DUMP_TIMEOUT:-3}"

qmp_dump_registers() {
  local out="$RUN_DIR/registers-$STAMP.txt"
  local watchdog_pid="" nc_pid=""
  if [ ! -S "$QMP_SOCK" ] || ! command -v nc >/dev/null 2>&1; then
    return 0
  fi
  if command -v timeout >/dev/null 2>&1; then
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
  echo "boot-arm64.sh: killing qemu pid=$QEMU_PID"
  kill -TERM "$QEMU_PID" 2>/dev/null || true
  sleep 2
  if kill -0 "$QEMU_PID" 2>/dev/null; then
    kill -KILL "$QEMU_PID" 2>/dev/null || true
  fi
  wait "$QEMU_PID" 2>/dev/null || true
}

capture_then_revert() {
  local reason="$1"
  echo "boot-arm64.sh: capture-then-revert ($reason)"
  # Dump first (bounded), then always kill — never gate kill on QMP success.
  qmp_dump_registers
  kill_qemu
  discard_clone
  echo "boot-arm64.sh: reverted (clone discarded); evidence in $RUN_DIR (serial-$STAMP.log)"
}

"$QEMU_BIN" "${QEMU_ARGS[@]}" &
QEMU_PID=$!
trap 'capture_then_revert signal; exit 130' INT TERM

elapsed=0
while kill -0 "$QEMU_PID" 2>/dev/null; do
  if [ "$elapsed" -ge "$TESTING_TIMEOUT" ]; then
    capture_then_revert "timeout ${TESTING_TIMEOUT}s — wedge verdict"
    exit 124
  fi
  sleep 5
  elapsed=$((elapsed + 5))
done

wait "$QEMU_PID" 2>/dev/null || true
capture_then_revert "qemu exited"
