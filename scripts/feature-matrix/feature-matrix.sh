#!/usr/bin/env bash
# Compile both Vulkan host pathways, tests included, plus the option ROM.
#
# The same backend uses MoltenVK on macOS and a native ICD on Linux. A target or
# cfg change can still break one host while the other compiles, so this script
# checks both.
#
# It checks `--all-targets`, not the bare default, so arm-specific test code
# compiles too. Compiling is not enough on its own: a test that compiles on an
# arm but is cfg'd out still tests nothing, so the script also reports how many
# tests each arm actually runs.
#
# It also gates formatting, which is arm-independent and which nothing else in
# the toolchain can see — rustc and clippy are both silent on it.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_DIR="${REPO}"
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

The two supported pathways use one feature set:

  Vulkan / MoltenVK  --no-default-features --features host-window Apple
  Vulkan / native    same feature set                             Linux

A third cell checks crates/reims-vgpu-efi, the PCI option ROM every x86 boot
builds. It is a separate workspace targeting x86_64-unknown-uefi, so it is not
a backend arm — but it ships, and nothing else in the repository checks it.

Two cells before all of those run `cargo fmt --all -- --check`, once per
workspace. rustfmt.toml at the repo root is the format and both workspaces are
kept clean under it, so these are no-ops until a change leaves a diff.

The feature set is exactly what vendor/qemu/hw/display/meson.build passes.

Warnings do not fail an arm; the count is printed so drift stays visible.

env:
  CROSS_TARGET   Linux target to cross-check (default x86_64-unknown-linux-gnu)
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
case "$HOST_TRIPLE" in
  *-apple-*) DEFAULT_CROSS_TARGET="x86_64-unknown-linux-gnu" ;;
  *) DEFAULT_CROSS_TARGET="aarch64-apple-darwin" ;;
esac
CROSS_TARGET="${CROSS_TARGET:-$DEFAULT_CROSS_TARGET}"

if ! rustc --print target-list | grep -qx "$CROSS_TARGET"; then
  echo "feature-matrix: ERROR: unknown target '$CROSS_TARGET'" >&2
  exit 1
fi
if [ "$CROSS_TARGET" != "$HOST_TRIPLE" ] &&
  ! rustup target list --installed 2>/dev/null | grep -qx "$CROSS_TARGET"; then
  echo "feature-matrix: ERROR: target '$CROSS_TARGET' not installed" >&2
  echo "feature-matrix:        run: rustup target add $CROSS_TARGET" >&2
  exit 1
fi

# Feature sets, verbatim from vendor/qemu/hw/display/meson.build.
FEATURES_VULKAN="--no-default-features --features host-window"

FAILED=0
RESULTS=()

# label, target triple (empty for host), feature args, then three optionals for
# a package that is not `reims-vgpu` in the workspace: its directory, its
# `-p`/`--manifest-path` selector, and its target scope.
run_cell() {
  local label="$1" target="$2" features="$3"
  local dir="${4:-$WORKSPACE_DIR}" pkg="${5--p reims-vgpu}" scope="${6---all-targets}"
  local target_args=()
  [ -n "$target" ] && target_args=(--target "$target")

  local log
  log="$(mktemp)"
  local status="PASS"
  # --all-targets is load-bearing: without it this compiles the product only,
  # and every arm's test code goes unchecked. The option ROM is the one cell
  # that cannot use it — its bin is `#![no_main]` with its own panic handler,
  # so libtest's harness collides with `std`'s on any target that has one.
  # shellcheck disable=SC2086  # $features, $pkg and $scope are argument lists.
  if ! (cd "$dir" && cargo "$CARGO_CMD" $scope $pkg \
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

# Formatting is one question for the whole tree, not one per arm, so it gets its
# own cell shape rather than a feature set. `cargo fmt --all -- --check` exits
# non-zero and prints the offending hunks; on a clean tree it is silent and
# costs a second. A missing rustfmt is a FAIL and not a SKIP: a gate that
# quietly stands down on the machine that lacks the tool is not a gate.
fmt_cell() {
  local label="$1" dir="$2"
  local log status
  log="$(mktemp)"
  if (cd "$dir" && cargo fmt --all -- --check) >"$log" 2>&1; then
    status="PASS"
    RESULTS+=("$(printf '%-4s %-46s %s' "$status" "$label" "clean")")
  else
    status="FAIL"
    FAILED=1
    RESULTS+=("$(printf '%-4s %-46s %s' "$status" "$label" "run: cargo fmt --all")")
    echo "--- $label ---" >&2
    head -40 "$log" >&2
  fi
  rm -f "$log"
}

# Enumerate an arm's tests without running them. `--list` makes the libtest
# harness print one `path::name: test` line per test and exit, so the count is
# what that arm would actually execute — cfg'd-out tests are simply absent.
# Only natively-runnable arms can be counted; a cross-compiled binary does not
# run on this host.
COUNTS=()
# label, feature args, then the same three optionals `run_cell` takes. A cell
# whose `all_targets` enumeration cannot link — the option ROM's — passes
# `--lib` as its scope and reports why rather than a misleading count.
count_cell() {
  local label="$1" features="$2"
  local dir="${3:-$WORKSPACE_DIR}" pkg="${4--p reims-vgpu}" scope="${5---all-targets}"
  [ "$COUNT_TESTS" -eq 1 ] || return 0
  local out lib total
  out="$(mktemp)"
  # shellcheck disable=SC2086  # $features and $pkg are argument lists.
  if ! (cd "$dir" && cargo test $pkg $features --lib -- --list) \
    >"$out" 2>/dev/null; then
    COUNTS+=("$(printf '%-46s %s' "$label" "(could not enumerate)")")
    rm -f "$out"
    return 0
  fi
  lib="$(grep -c ': test$' "$out" || true)"
  if [ "$scope" = "--lib" ]; then
    COUNTS+=("$(printf '%-46s lib=%-5s all_targets=%s' "$label" "$lib" \
      "(bin is UEFI-only)")")
    rm -f "$out"
    return 0
  fi
  # shellcheck disable=SC2086
  (cd "$dir" && cargo test $pkg $features -- --list) \
    >"$out" 2>/dev/null || true
  total="$(grep -c ': test$' "$out" || true)"
  COUNTS+=("$(printf '%-46s lib=%-5s all_targets=%s' "$label" "$lib" "$total")")
  rm -f "$out"
}

echo "[feature-matrix] host=$HOST_TRIPLE cross=$CROSS_TARGET cargo=$CARGO_CMD"

# Cells 0a and 0b — formatting, one per workspace. These run first because they
# are the cheapest and need no target installed, and because a formatting diff
# is the one failure a reviewer should never have to read a compile log to find.
fmt_cell "rustfmt / workspace" "$WORKSPACE_DIR"
fmt_cell "rustfmt / reims-vgpu-efi" "$REPO/crates/reims-vgpu-efi"

# Vulkan through MoltenVK on Apple, native ICD on Linux. The same
# feature set; the host is what differs.
run_cell "vulkan,host-window / $HOST_TRIPLE" "" "$FEATURES_VULKAN"
count_cell "vulkan,host-window / $HOST_TRIPLE" "$FEATURES_VULKAN"
if [ "$CROSS_TARGET" != "$HOST_TRIPLE" ]; then
  run_cell "vulkan,host-window / $CROSS_TARGET" "$CROSS_TARGET" "$FEATURES_VULKAN"
  if [ "$COUNT_TESTS" -eq 1 ]; then
    COUNTS+=("$(printf '%-46s %s' "vulkan,host-window / $CROSS_TARGET" \
      "(cross-compiled — cannot run here)")")
  fi
fi

# The PCI option ROM is not a backend arm and not a workspace member: it
# is its own workspace targeting x86_64-unknown-uefi, and `vm/boot-x86.sh`
# rebuilds it before every x86 boot. It was invisible to this script and to
# every command in AGENTS.md: a live crate nothing checks.
#
# `--all-targets` cannot be used here. The bin is `#![no_main]` with the `uefi`
# crate's panic handler, so building its test harness collides with `std`'s —
# on the UEFI target and on the host alike. The lib is where the logic is
# (`paint`, the real Blt paths), and it does run on the host.
EFI_DIR="$REPO/crates/reims-vgpu-efi"
EFI_TARGET="x86_64-unknown-uefi"
if rustup target list --installed 2>/dev/null | grep -qx "$EFI_TARGET"; then
  run_cell "option-rom / $EFI_TARGET" "$EFI_TARGET" "" "$EFI_DIR" "" ""
else
  RESULTS+=("$(printf '%-4s %-46s %s' "SKIP" "option-rom / $EFI_TARGET" \
    "(rustup target add $EFI_TARGET)")")
fi
count_cell "option-rom / host lib" "" "$EFI_DIR" "" "--lib"

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
  echo "[feature-matrix] FAILED: an arm does not compile, or the tree is unformatted" >&2
  exit 1
fi
echo "[feature-matrix] all supported arms compile; both workspaces are rustfmt-clean"
