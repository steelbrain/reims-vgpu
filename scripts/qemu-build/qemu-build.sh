#!/usr/bin/env bash
#
# scripts/qemu-build/qemu-build.sh
#
# Configures and builds the vendored QEMU submodule (steelbrain/qemu-reims-vgpu,
# branch host-reims-vgpu-vmapple) into vendor/qemu/build/qemu-system-<arch>.
#
# Targets:
#   aarch64  — arm macOS guest rail using Vulkan through MoltenVK.
#   x86_64   — x86 macOS guest rail using the host's native Vulkan ICD.
#
# The vmapple enablement (RFC v7 patches 1-6; patch 7 excluded -- it needs SIP
# disabled) is committed IN the submodule, so this script does not clone or
# patch: future device work commits directly in vendor/qemu.
#
# Product device reims-vgpu-mmio links the Vulkan-only crates/reims-vgpu staticlib.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
QEMU_SRC="$REPO_ROOT/vendor/qemu"
# Vulkan paths: public metal2vulkan crate + in-crate ash engine. On macOS this
# uses MoltenVK through the Vulkan loader; on Linux it uses the native ICD.
# Empty → auto: aarch64 on Darwin, x86_64 on Linux.
QEMU_TARGET="${QEMU_TARGET:-}"

echo "[qemu-build] qemu submodule: $QEMU_SRC"

# --- Ensure the submodule is checked out ----------------------------------------
if [ ! -f "$QEMU_SRC/configure" ]; then
  echo "[qemu-build] submodule not populated; running git submodule update --init ..."
  git -C "$REPO_ROOT" submodule update --init vendor/qemu
fi

cd "$QEMU_SRC"

while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help)
      cat <<'EOF'
usage: scripts/qemu-build/qemu-build.sh [options]

Build vendor/qemu (reims-vgpu-mmio + reims-vgpu for the selected target).

  --target aarch64|x86_64  softmmu target (default: aarch64 on Darwin, x86_64
                           on Linux). Env: QEMU_TARGET.
Examples:
  scripts/qemu-build/qemu-build.sh
  scripts/qemu-build/qemu-build.sh --target aarch64
  scripts/qemu-build/qemu-build.sh --target x86_64
EOF
      exit 0
      ;;
    --target)
      shift
      QEMU_TARGET="${1:-}"
      if [ -z "$QEMU_TARGET" ]; then
        echo "[qemu-build] ERROR: --target needs aarch64|x86_64" >&2
        exit 1
      fi
      shift
      ;;
    --target=*)
      QEMU_TARGET="${1#--target=}"
      shift
      ;;
    *)
      echo "[qemu-build] ERROR: unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

# --- Resolve target -------------------------------------------------------------
if [ -z "$QEMU_TARGET" ]; then
  case "$(uname -s)" in
    Darwin) QEMU_TARGET=aarch64 ;;
    Linux)  QEMU_TARGET=x86_64 ;;
    *)
      echo "[qemu-build] ERROR: unknown host OS '$(uname -s)'; pass --target aarch64|x86_64" >&2
      exit 1
      ;;
  esac
fi

case "$QEMU_TARGET" in
  aarch64|x86_64) ;;
  *)
    echo "[qemu-build] ERROR: unknown target '$QEMU_TARGET' (aarch64 | x86_64)" >&2
    exit 1
    ;;
esac

echo "[qemu-build] target=$QEMU_TARGET"
echo "[qemu-build] backend=vulkan (reims-vgpu-mmio → reims-vgpu)"

# Product aarch64 path needs macOS (HVF + Cocoa). Fail early with a clear message.
if [ "$QEMU_TARGET" = aarch64 ] && [ "$(uname -s)" != Darwin ]; then
  echo "[qemu-build] ERROR: aarch64 product build (vmapple + HVF + Cocoa) requires macOS" >&2
  echo "[qemu-build]        On Linux use: scripts/qemu-build/qemu-build.sh --target x86_64" >&2
  exit 1
fi

# Prefer a known cargo if PATH is minimal (agent shells). Always needed: both
# targets link libreims_vgpu.a.
if ! command -v cargo >/dev/null 2>&1; then
  if [ -x "$HOME/.cargo/bin/cargo" ]; then
    export PATH="$HOME/.cargo/bin:$PATH"
  else
    echo "[qemu-build] ERROR: cargo not found (needed to build reims-vgpu)" >&2
    exit 1
  fi
fi

if [ "$QEMU_TARGET" = aarch64 ]; then
  # The vmapple enablement must be present in the checked-out submodule.
  grep -q '^CONFIG_VMAPPLE=y' configs/devices/aarch64-softmmu/default.mak \
    || { echo "[qemu-build] ERROR: submodule missing vmapple enablement (wrong commit checked out?)" >&2; exit 1; }
  # Patch 7 (private ISA) must NOT be present -- we do not disable SIP.
  if grep -q 'com.apple.private.hypervisor' accel/hvf/entitlements.plist; then
    echo "[qemu-build] ERROR: private-hypervisor entitlement present (patch 7 leaked in)" >&2
    exit 1
  fi
fi

# --- Configure ------------------------------------------------------------------
# Target + backend choice are read at configure. If an existing build used a
# different stamp, force reconfigure.
#
# `REIMS_VGPU_COVERAGE` belongs in the stamp for the same reason the backend
# does: `hw/display/meson.build` reads it with `run_command` and turns it into a
# `--whole-archive` link argument, which is a configure-time decision. Without
# it here, a tree already configured without coverage skipped configure and
# linked no profile writer, so `scripts/runtime-dead` booted an instrumented
# staticlib whose counters nothing ever wrote out — and reported every function
# in the crate as never having run. The path and not just a flag, because a
# different clang major is a different archive.
STAMP_WANTED="${QEMU_TARGET}:vulkan-only:reims-vgpu-crates:cov=${REIMS_VGPU_COVERAGE:-off}"
if [ -f "build/build.ninja" ]; then
  prev=""
  if [ -f "build/qemu-build.stamp" ]; then
    prev="$(cat build/qemu-build.stamp)"
  fi
  if [ "$prev" != "$STAMP_WANTED" ]; then
    echo "[qemu-build] build stamp changed ($prev -> $STAMP_WANTED); reconfiguring ..."
    rm -rf build
  fi
fi

if [ ! -f "build/build.ninja" ]; then
  case "$QEMU_TARGET" in
    aarch64)
      echo "[qemu-build] configuring (aarch64-softmmu, hvf, cocoa) ..."
      ./configure \
        --target-list=aarch64-softmmu \
        --enable-hvf \
        --enable-cocoa \
        --disable-docs \
        --disable-bsd-user \
        --disable-linux-user \
        --disable-tools
      ;;
    x86_64)
      echo "[qemu-build] configuring (x86_64-softmmu, no hvf/cocoa) ..."
      ./configure \
        --target-list=x86_64-softmmu \
        --disable-hvf \
        --disable-cocoa \
        --disable-docs \
        --disable-bsd-user \
        --disable-linux-user \
        --disable-tools
      ;;
  esac
else
  echo "[qemu-build] already configured; skipping configure."
fi
printf '%s\n' "$STAMP_WANTED" > build/qemu-build.stamp

# --- Build ----------------------------------------------------------------------
QEMU_BIN="$QEMU_SRC/build/qemu-system-${QEMU_TARGET}"
echo "[qemu-build] building qemu-system-${QEMU_TARGET} (links reims-vgpu / Vulkan) ..."
ninja -C build "qemu-system-${QEMU_TARGET}"

# --- Verify (target-specific) ---------------------------------------------------
case "$QEMU_TARGET" in
  aarch64)
    # QEMU's own build codesigns the binary with accel/hvf/entitlements.plist
    # (com.apple.security.hypervisor -- the public entitlement). Verify; do not
    # re-sign (re-signing trips on build xattrs).
    echo "[qemu-build] verifying HVF signature ..."
    codesign -v "$QEMU_BIN"
    if ! codesign -d --entitlements :- "$QEMU_BIN" 2>/dev/null | tr -d '\0' | \
         grep -q 'com.apple.security.hypervisor'; then
      echo "[qemu-build] ERROR: qemu-system-aarch64 lacks the hypervisor entitlement" >&2
      exit 1
    fi

    echo
    echo "[qemu-build] Success: $QEMU_BIN"
    "$QEMU_BIN" --version | head -1
    echo "[qemu-build] vmapple present:"
    "$QEMU_BIN" -M help | grep -i vmapple || { echo "  ERROR: vmapple not listed" >&2; exit 1; }
    echo "[qemu-build] backend=vulkan (gfx-device: apple-gfx-mmio | reims-vgpu-mmio)"
    ;;
  x86_64)
    echo
    echo "[qemu-build] Success: $QEMU_BIN"
    "$QEMU_BIN" --version | head -1
    # SysBus devices are not always listed in plain `-device help`; probe type.
    if "$QEMU_BIN" -device reims-vgpu-mmio,help >/dev/null 2>&1; then
      echo "[qemu-build] reims-vgpu-mmio device: present (reims-vgpu / Vulkan linked)"
    else
      echo "[qemu-build] WARNING: reims-vgpu-mmio type not registered" >&2
    fi
    if "$QEMU_BIN" -device reims-vgpu-pci,help >/dev/null 2>&1; then
      echo "[qemu-build] reims-vgpu-pci device: present (x86 Tahoe PCI thin shim)"
    else
      echo "[qemu-build] WARNING: reims-vgpu-pci type not registered" >&2
    fi
    if "$QEMU_BIN" -accel help 2>/dev/null | grep -qi kvm; then
      echo "[qemu-build] kvm accel: available"
    else
      echo "[qemu-build] kvm accel: not listed (TCG-only; install/enable KVM if you need it)"
    fi
    echo "[qemu-build] backend=vulkan (reims-vgpu staticlib linked)"
    echo "[qemu-build] for vm/boot-x86.sh: QEMU_BIN=$QEMU_BIN vm/boot-x86.sh ..."
    ;;
esac
