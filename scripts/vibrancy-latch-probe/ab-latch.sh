#!/usr/bin/env bash
# ab-latch.sh — score two binaries against each other, alternating boot by boot.
#
# `latch-rate.sh` scores one binary over N boots. That is the right shape when
# the only question is a rate, and the wrong shape for comparing two binaries,
# because it measures them at different times. This class is scored over hours,
# and anything that drifts across that window — host thermals, page-cache state,
# another process someone started, the agent's own `cargo` runs — lands entirely
# on whichever arm was running at the time and reads as an effect of the change.
#
# That is not hypothetical here. One 8-boot `latch-rate.sh` run on a single
# pinned binary produced clean, clean, degraded, clean, clean, degraded,
# degraded, degraded: the first five boots and the last three disagree sharply,
# on identical code, and builds were running on the host during the tail. A
# before-run compared against an after-run would have scored that drift as the
# change.
#
# Alternating removes it. Boot i uses arm A when i is odd and arm B when it is
# even, so any trend across the sweep is split evenly between the arms and the
# difference between them is what is left. Each boot is a full revert to the same
# immutable snapshot, so the boots stay independent by construction.
#
# ## Run it on a quiet machine anyway
#
# Interleaving defends against drift, not against noise: contention still costs
# boots, it just stops being *attributed* to one arm. Start nothing while this
# runs — no builds, no tests, no second VM.
#
# ## What it scores
#
#   verdict   pane-frost-gate on the pane before/after the load (the visible
#             defect, but only when the defect lands on the photographed pane)
#   declines  `vk_draw_exec_sampled_resident_missing` in that boot's fail log —
#             the mechanism itself, which fires whether or not the corruption
#             happens to be visible in the shot, and is the more sensitive of the
#             two. `prior=` on that line names which reclaim path took the
#             resident, or `no_record` if this device never held one.
#
# Usage:
#   ab-latch.sh --a /path/to/pin-A-qemu-system-x86_64 \
#               --b /path/to/pin-B-qemu-system-x86_64 \
#               [--boots N] [--load-seconds S] [--out DIR]
#
# Both paths must live inside `vendor/qemu/build/` and keep `qemu-system-x86_64`
# in the filename — QEMU finds its firmware relative to its own executable, and
# every VM sweep in this repo matches that name when killing stragglers. That
# second requirement is why `kill_vm` below cannot use the bare pattern the other
# sweeps use: this script's own arguments contain it. See the note there.
set -euo pipefail
export LC_ALL=C

BOOTS=8
LOAD_SECONDS=150
CENSUS_SECONDS=12
ARM_A=""
ARM_B=""
OUT=""
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROBE="$REPO_ROOT/scripts/vibrancy-latch-probe/vibrancy-latch-probe.sh"
GATE="$REPO_ROOT/scripts/vibrancy-latch-probe/pane-frost-gate.sh"
FAILLOG="${REIMS_FAIL_LOG:-/tmp/reims-vgpu-fail.log}"

while [ $# -gt 0 ]; do
  case "$1" in
    --a) ARM_A="$2"; shift 2 ;;
    --b) ARM_B="$2"; shift 2 ;;
    --boots) BOOTS="$2"; shift 2 ;;
    --load-seconds) LOAD_SECONDS="$2"; shift 2 ;;
    --census-seconds) CENSUS_SECONDS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,48p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "ab-latch: unknown argument $1" >&2; exit 2 ;;
  esac
done

[ -x "$ARM_A" ] || { echo "ab-latch: --a is not executable: $ARM_A" >&2; exit 2; }
[ -x "$ARM_B" ] || { echo "ab-latch: --b is not executable: $ARM_B" >&2; exit 2; }
if cmp -s "$ARM_A" "$ARM_B"; then
  echo "ab-latch: --a and --b are the same bytes; there is nothing to compare." >&2
  echo "(That is a useful run — it measures this harness's own noise floor — but" >&2
  echo "say so deliberately rather than discovering it in the summary.)" >&2
fi

WORK="${OUT:-$(mktemp -d -t ab-latch-XXXXXX)}"
mkdir -p "$WORK"
say() { echo "ab-latch: $*"; }

# Kill stragglers without killing the sweep.
#
# `pgrep -x` cannot see QEMU at all — the name is longer than 15 characters — so
# the pattern has to be a `-f` match on the command line. But this script takes
# two *paths to QEMU binaries* as arguments, so the bare `qemu-system-x86_64`
# pattern every other sweep here uses matches this script's own invocation, and
# `pkill -9` on it kills the sweep mid-run. That is not theoretical: it happened
# twice while building this, once to `latch-rate.sh` launched with `QEMU_BIN=…`
# on the command line and once to this script.
#
# Two independent guards, because either alone is thin:
#   - anchor on `-enable-kvm`, which the real process always carries directly
#     after the binary and no argument list here does;
#   - skip our own pid and our ancestors explicitly, so a future caller that
#     happens to reproduce the anchor still cannot be killed.
kill_vm() {
  for pid in $(pgrep -A -f 'qemu-system-x86_64 -enable-kvm' 2>/dev/null || true); do
    [ "$pid" = "$$" ] && continue
    kill -9 "$pid" 2>/dev/null || true
  done
  # Give the port forward time to come back, or the next boot dies on
  # "Could not set up host forwarding rule 'tcp::2222-:22'" and the ssh wait
  # then succeeds against the *previous* guest.
  sleep 6
}

budget=$(( LOAD_SECONDS + 4 * CENSUS_SECONDS + 420 ))
results="$WORK/results.tsv"
: >"$results"
say "work dir $WORK — $BOOTS boots, alternating, ${LOAD_SECONDS}s load each"
say "  arm A: $ARM_A"
say "  arm B: $ARM_B"

for i in $(seq 1 "$BOOTS"); do
  if [ $((i % 2)) -eq 1 ]; then arm="A"; bin="$ARM_A"; else arm="B"; bin="$ARM_B"; fi
  boot_dir="$WORK/boot-$i-$arm"
  mkdir -p "$boot_dir"
  kill_vm
  rm -f "$FAILLOG"
  say "boot $i/$BOOTS (arm $arm)"
  # Exported, never placed on a child's command line: `kill_vm`'s pattern would
  # otherwise match this script's own invocation and -9 it mid-sweep.
  export QEMU_BIN="$bin"
  TESTING_TIMEOUT="$budget" "$REPO_ROOT/vm/boot-x86.sh" --device reims-vgpu-pci --testing \
    >"$boot_dir/boot.log" 2>&1 &
  boot_pid=$!

  waited=0
  until ssh -o ConnectTimeout=5 -o BatchMode=yes macos-vm true 2>/dev/null; do
    sleep 5
    waited=$((waited + 5))
    [ "$waited" -ge 300 ] && break
    kill -0 "$boot_pid" 2>/dev/null || break
  done
  if ! ssh -o ConnectTimeout=5 -o BatchMode=yes macos-vm true 2>/dev/null; then
    say "boot $i never came up — see $boot_dir/boot.log" >&2
    printf '%s\t%s\t%s\t%s\t%s\n' "$i" "$arm" "no-boot" "-" "-" >>"$results"
    continue
  fi

  if ! "$PROBE" --load-seconds "$LOAD_SECONDS" --census-seconds "$CENSUS_SECONDS" \
      --out "$boot_dir" >"$boot_dir/probe.log" 2>&1; then
    say "boot $i (arm $arm): probe refused a verdict — see $boot_dir/probe.log" >&2
    cp "$FAILLOG" "$boot_dir/full-boot.log" 2>/dev/null || true
    printf '%s\t%s\t%s\t%s\t%s\n' "$i" "$arm" "probe-refused" "-" "-" >>"$results"
    continue
  fi

  gate_out=$("$GATE" --before "$boot_dir/before.png" --after "$boot_dir/after.png" 2>&1) \
    && verdict="clean" || verdict="degraded"
  echo "$gate_out" >"$boot_dir/gate.log"
  rmse=$(echo "$gate_out" | sed -n 's/.*pane=[^ ]* rmse=\([0-9.e-]*\).*/\1/p')
  echo "$gate_out" | grep -q "not of the same scene" && verdict="gate-refused"
  cp "$FAILLOG" "$boot_dir/full-boot.log" 2>/dev/null || true
  declines=$(grep -c 'vk_draw_exec_sampled_resident_missing' "$boot_dir/full-boot.log" 2>/dev/null || true)
  printf '%s\t%s\t%s\t%s\t%s\n' "$i" "$arm" "$verdict" "${rmse:--}" "${declines:-0}" >>"$results"
  say "boot $i (arm $arm): $verdict (pane rmse ${rmse:--}, declines ${declines:-0})"
done

kill_vm

say ""
say "boot	arm	verdict	pane_rmse	declines"
sed 's/^/ab-latch:   /' "$results"
say ""
for arm in A B; do
  n=$(awk -F'\t' -v a="$arm" '$2==a && ($3=="clean" || $3=="degraded")' "$results" | wc -l)
  d=$(awk -F'\t' -v a="$arm" '$2==a && $3=="degraded"' "$results" | wc -l)
  dec=$(awk -F'\t' -v a="$arm" '$2==a && $5 ~ /^[0-9]+$/ {s+=$5} END{print s+0}' "$results")
  say "arm $arm: $d/$n degraded, $dec declines total"
done
say ""
say "Neither number is a proportion test. Read them next to the per-boot rows —"
say "a class this rare needs the distribution, and a mean would hide it."
say "per-boot evidence in $WORK/boot-*/"
exit 0
