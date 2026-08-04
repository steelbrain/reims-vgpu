#!/usr/bin/env bash
# Compile every supported reims-vgpu arm, tests included.
#
# The project supports four arms, one per host GPU API actually available:
# Metal on Apple, Vulkan through MoltenVK on Apple, and Vulkan on a native ICD
# on Linux and on Windows. QEMU's meson picks one per build and day-to-day work compiles one,
# so a rename or a cfg change could break another arm silently for days. This
# script is the gate.
#
# It checks `--all-targets`, not the bare default, so arm-specific test code
# compiles too. Compiling is not enough on its own: a test that compiles on an
# arm but is cfg'd out still tests nothing, so the script also reports how many
# tests each arm actually runs.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_DIR="${REPO}"
CROSS_TARGET="${CROSS_TARGET:-x86_64-unknown-linux-gnu}"
WINDOWS_TARGET="${WINDOWS_TARGET:-x86_64-pc-windows-gnu}"
CARGO_CMD="check"
COUNT_TESTS=1

usage() {
  cat <<'EOF'
usage: scripts/feature-matrix/feature-matrix.sh [--build] [--no-counts]

Runs `cargo check --all-targets` (or `cargo build` with --build) over every
supported reims-vgpu arm and reports one PASS/FAIL line per arm. Exits non-zero
if any arm fails to compile.

It then reports, per natively-runnable arm, how many tests that arm actually
enumerates — a test that compiles but is cfg'd out still tests nothing, so a
cfg change that silently empties an arm shows up as a dropped count rather than
a green run. Counting links the test binaries, which is slower than checking;
pass --no-counts to skip it. The cross-compiled arm cannot be counted because
its binaries do not run on this host.

The four supported arms, one per host GPU API actually available:

  Metal              --features backend-metal                     Apple only
  Vulkan / MoltenVK  --no-default-features
                       --features backend-vulkan,host-window      Apple
  Vulkan / native    same feature set                             Linux
  Vulkan / native    same feature set                             Windows

The feature sets are exactly what vendor/qemu/hw/display/meson.build passes for
REIMS_VGPU_BACKEND=metal and REIMS_VGPU_BACKEND=vulkan. An Apple host builds
all three natively (the Linux one by cross check). A Linux host builds the
native Vulkan arm and cross-checks the other two.

The Metal arm cross-checks from any host. src/lib.rs rejects backend-metal on
`not(target_os = "macos")` — that is a condition on the *target*, not on the
host, so `--target *-apple-darwin` satisfies it and the real cfgs are
exercised. `cargo check` needs no Apple SDK. This script used to skip the arm
off Apple on the theory that Metal could not be cross-checked at all, and the
arm rotted to 11 errors unnoticed; every one of them was in first-party code
that this cell catches.

Checking is all that is claimed: it type-checks the arm, it does not link
against a real SDK and cannot run it.

Warnings do not fail an arm; the count is printed so drift stays visible.

env:
  CROSS_TARGET   Linux target to cross-check (default x86_64-unknown-linux-gnu)
  METAL_TARGET   Apple target to cross-check the Metal arm against off Apple
                 (default: aarch64-apple-darwin if installed, else
                 x86_64-apple-darwin)
  WINDOWS_TARGET Windows target to cross-check (default x86_64-pc-windows-gnu)
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --build)
      CARGO_CMD="build"
      shift
      ;;
    --no-counts)
      COUNT_TESTS=0
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "feature-matrix: unknown argument '$1'" >&2
      usage >&2
      exit 64
      ;;
  esac
done

if ! command -v cargo >/dev/null 2>&1; then
  if [ -x "$HOME/.cargo/bin/cargo" ]; then
    export PATH="$HOME/.cargo/bin:$PATH"
  else
    echo "feature-matrix: ERROR: cargo not found" >&2
    exit 1
  fi
fi

HOST_TRIPLE="$(rustc -vV | awk '/^host: /{print $2}')"

if ! rustc --print target-list | grep -qx "$CROSS_TARGET"; then
  echo "feature-matrix: ERROR: unknown target '$CROSS_TARGET'" >&2
  exit 1
fi
CROSS_TARGET_AVAILABLE=1
if [ "$CROSS_TARGET" != "$HOST_TRIPLE" ] &&
  ! rustup target list --installed 2>/dev/null | grep -qx "$CROSS_TARGET"; then
  CROSS_TARGET_AVAILABLE=0
fi

# Feature sets, verbatim from vendor/qemu/hw/display/meson.build.
FEATURES_METAL="--features backend-metal"
FEATURES_VULKAN="--no-default-features --features backend-vulkan,host-window"

FAILED=0
RESULTS=()

run_cell() {
  local label="$1" target="$2" features="$3"
  local target_args=()
  [ -n "$target" ] && target_args=(--target "$target")

  local log
  log="$(mktemp)"
  local status="PASS"
  # --all-targets is load-bearing: without it this compiles the product only,
  # and every arm's test code goes unchecked.
  # shellcheck disable=SC2086  # $features is an intentional argument list.
  if ! (cd "$WORKSPACE_DIR" && cargo "$CARGO_CMD" --all-targets -p reims-vgpu \
    ${target_args[@]+"${target_args[@]}"} \
    $features --message-format short) >"$log" 2>&1; then
    status="FAIL"
    FAILED=1
  fi
  # cargo replays nothing for an up-to-date unit, so a cached cell reports no
  # warnings at all. Say "cached" rather than "0" — a silent 0 reads as clean.
  local warns
  if grep -q '^ *Checking reims-vgpu\|^ *Compiling reims-vgpu' "$log"; then
    warns="$(grep -c ': warning' "$log" || true)"
  else
    warns="cached"
  fi
  RESULTS+=("$(printf '%-4s %-46s warnings=%s' "$status" "$label" "$warns")")
  if [ "$status" = "FAIL" ]; then
    echo "--- $label ---" >&2
    grep -E ': error|^error' "$log" >&2 || cat "$log" >&2
  fi
  rm -f "$log"
}

# Enumerate an arm's tests without running them. `--list` makes the libtest
# harness print one `path::name: test` line per test and exit, so the count is
# what that arm would actually execute — cfg'd-out tests are simply absent.
# Only natively-runnable arms can be counted; a cross-compiled binary does not
# run on this host.
COUNTS=()
count_cell() {
  local label="$1" features="$2"
  [ "$COUNT_TESTS" -eq 1 ] || return 0
  local out lib total
  out="$(mktemp)"
  # shellcheck disable=SC2086  # $features is an intentional argument list.
  if ! (cd "$WORKSPACE_DIR" && cargo test -p reims-vgpu $features --lib -- --list) \
    >"$out" 2>/dev/null; then
    COUNTS+=("$(printf '%-46s %s' "$label" "(could not enumerate)")")
    rm -f "$out"
    return 0
  fi
  lib="$(grep -c ': test$' "$out" || true)"
  # shellcheck disable=SC2086
  (cd "$WORKSPACE_DIR" && cargo test -p reims-vgpu $features -- --list) \
    >"$out" 2>/dev/null || true
  total="$(grep -c ': test$' "$out" || true)"
  COUNTS+=("$(printf '%-46s lib=%-5s all_targets=%s' "$label" "$lib" "$total")")
  rm -f "$out"
}

echo "[feature-matrix] host=$HOST_TRIPLE cross=$CROSS_TARGET cargo=$CARGO_CMD"

# Arm 1 — Metal. Native on Apple, cross-checked everywhere else: lib.rs gates
# backend-metal on target_os, so an Apple *target* is all the arm needs. Only
# the run half is Apple-only, which is why the off-Apple cell never counts
# tests.
case "$HOST_TRIPLE" in
  *-apple-*)
    run_cell "metal / $HOST_TRIPLE" "" "$FEATURES_METAL"
    count_cell "metal / $HOST_TRIPLE" "$FEATURES_METAL"
    ;;
  *)
    # arm64 macOS is the pathway this arm actually ships on, so prefer it and
    # fall back to the x86 Apple target; both carry target_os = "macos" and so
    # exercise the same cfgs.
    if [ -z "${METAL_TARGET:-}" ]; then
      for cand in aarch64-apple-darwin x86_64-apple-darwin; do
        if rustup target list --installed 2>/dev/null | grep -qx "$cand"; then
          METAL_TARGET="$cand"
          break
        fi
      done
    fi
    if [ -n "${METAL_TARGET:-}" ]; then
      run_cell "metal / $METAL_TARGET" "$METAL_TARGET" "$FEATURES_METAL"
      if [ "$COUNT_TESTS" -eq 1 ]; then
        COUNTS+=("$(printf '%-46s %s' "metal / $METAL_TARGET" \
          "(cross-compiled — cannot run here)")")
      fi
    else
      # Not a pass. Say which command restores the cell, because the last time
      # this arm went unchecked it accumulated 11 errors.
      RESULTS+=("$(printf '%-4s %-46s %s' "SKIP" "metal / $HOST_TRIPLE" \
        "(rustup target add aarch64-apple-darwin)")")
    fi
    ;;
esac

# Arms 2 and 3 — Vulkan through MoltenVK on Apple, native ICD on Linux. Same
# feature set; the host is what differs.
run_cell "vulkan,host-window / $HOST_TRIPLE" "" "$FEATURES_VULKAN"
count_cell "vulkan,host-window / $HOST_TRIPLE" "$FEATURES_VULKAN"
if [ "$CROSS_TARGET" = "$HOST_TRIPLE" ]; then
  : # already covered by the host cell above
elif [ "$CROSS_TARGET_AVAILABLE" -eq 1 ]; then
  run_cell "vulkan,host-window / $CROSS_TARGET" "$CROSS_TARGET" "$FEATURES_VULKAN"
  if [ "$COUNT_TESTS" -eq 1 ]; then
    COUNTS+=("$(printf '%-46s %s' "vulkan,host-window / $CROSS_TARGET" \
      "(cross-compiled — cannot run here)")")
  fi
else
  RESULTS+=("$(printf '%-4s %-46s %s' "SKIP" "vulkan,host-window / $CROSS_TARGET" "(rustup target add $CROSS_TARGET)")")
fi

# Arm 4 - Vulkan on a native Windows ICD. Same feature set again; only the host
# differs. Missing target support is a machine fact, so name it and keep the
# other cells useful.
if [ "$WINDOWS_TARGET" = "$HOST_TRIPLE" ]; then
  : # already covered by the host cell above
elif rustup target list --installed 2>/dev/null | grep -qx "$WINDOWS_TARGET"; then
  run_cell "vulkan,host-window / $WINDOWS_TARGET" "$WINDOWS_TARGET" "$FEATURES_VULKAN"
  if [ "$COUNT_TESTS" -eq 1 ]; then
    COUNTS+=("$(printf '%-46s %s' "vulkan,host-window / $WINDOWS_TARGET" \
      "(cross-compiled — cannot run here)")")
  fi
else
  RESULTS+=("$(printf '%-4s %-46s %s' "SKIP" "vulkan,host-window / $WINDOWS_TARGET" "(rustup target add $WINDOWS_TARGET)")")
fi

echo
for line in "${RESULTS[@]}"; do
  echo "[feature-matrix] $line"
done

if [ "${#COUNTS[@]}" -gt 0 ]; then
  echo
  echo "[feature-matrix] tests enumerated per arm:"
  for line in "${COUNTS[@]}"; do
    echo "[feature-matrix]   $line"
  done
  echo "[feature-matrix] A dropped count means a cfg change emptied an arm;"
  echo "[feature-matrix] compiling is not the same as testing."
fi

if [ "$FAILED" -ne 0 ]; then
  echo "[feature-matrix] FAILED: at least one supported arm does not compile" >&2
  exit 1
fi
echo "[feature-matrix] all supported arms compile"
