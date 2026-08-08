#!/usr/bin/env bash
# latch-rate.sh — how often does a boot come up with the vibrancy rail broken?
#
# `draw/vulkan.rs`'s `note_gva_resident_aliasing` records the property that
# makes every single-boot reading of this class useless: pooled over five
# 14-round boots on one binary the corruption was 0, 0, 0, 14, 0 — **all-or-
# nothing per boot, never mixed**. Something latches once per boot and then holds
# for every round after it. So a run that comes back clean has measured that
# boot's latch, not the change under test, and five clean runs in a row are
# exactly what a one-in-five rate looks like.
#
# Scoring this class needs boots. That was impractical while the verdict was a
# person looking at two PNGs; `pane-frost-gate.sh` makes it a number, and this
# script is the loop around it.
#
# Each iteration is a full boot from the same immutable snapshot, so the boots
# are independent by construction — nothing carries over but the binary.
#
#   settle   boot, wait for ssh
#   probe    vibrancy-latch-probe: pane / load / same pane again
#   gate     pane-frost-gate on the two panes
#   record   verdict + the fail log, then kill the VM
#
# Reports the rate and every per-boot RMSE, because the distribution is the
# finding: a class that latches per boot shows up as a bimodal column (a cluster
# at the noise floor and a cluster far above it), and a mean would hide exactly
# that.
#
# ## Read `gw_refused` before the verdict
#
# The pane gate only fires when the defect lands on the pane that was
# photographed, so it is the *less* sensitive of the two columns. The
# `gw_refused_guest_store` census counter is the mechanism itself: the guest-write
# witness telling a gather that the guest overwrote the pages it was about to
# skip. It separates the classes with no overlap at all, over twelve recorded
# boots on two binaries:
#
#   clean      155  157  164  168  174  174  186
#   degraded 20122 26072 27502 32604 34772
#
# Two orders of magnitude, and nothing in between. A boot in the thousands has
# latched whether or not the screenshot caught it. Score on this column and use
# the verdict to confirm.
#
# Once the cause was fixed in the guest-write witness itself, eight consecutive
# boots read 145 154 156 166 167 167 168 186 with `deferred_flush_clobber` 0 on
# every one. Read the two controls alongside it before believing such a run:
# `t11_gw_armed` and `gw_vouched` stayed at their pre-fix magnitudes, so the
# witness was still armed and still answering. A change that silenced the
# witness instead of fixing it would zero this column too, and those two are how
# the difference is told.
#
# Usage:
#   latch-rate.sh [--boots N] [--load-seconds S] [--census-seconds C] [--out DIR]
#
# Pin the binary with QEMU_BIN or every boot rebuilds QEMU, which both makes the
# boots differ and locks the tree for the length of the sweep. Export it rather
# than placing it on the command line — see kill_vm below.
#
# Exits 0 whenever the loop ran. It does not fail on a degraded boot — that is
# the measurement, not an error.
set -euo pipefail
export LC_ALL=C

BOOTS=6
LOAD_SECONDS=150
CENSUS_SECONDS=12
OUT=""
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROBE="$REPO_ROOT/scripts/vibrancy-latch-probe/vibrancy-latch-probe.sh"
GATE="$REPO_ROOT/scripts/vibrancy-latch-probe/pane-frost-gate.sh"
FAILLOG="${REIMS_FAIL_LOG:-/tmp/reims-vgpu-fail.log}"

while [ $# -gt 0 ]; do
  case "$1" in
    --boots) BOOTS="$2"; shift 2 ;;
    --load-seconds) LOAD_SECONDS="$2"; shift 2 ;;
    --census-seconds) CENSUS_SECONDS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,36p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "latch-rate: unknown argument $1" >&2; exit 2 ;;
  esac
done

WORK="${OUT:-$(mktemp -d -t latch-rate-XXXXXX)}"
mkdir -p "$WORK"
say() { echo "latch-rate: $*"; }

# Kill stragglers without killing the sweep.
#
# `pgrep -x` cannot see QEMU at all — the name is longer than 15 characters — so
# the pattern has to be a `-f` match on the command line. The bare
# `qemu-system-x86_64` this used matches any caller that named a QEMU binary on
# its own command line, including `QEMU_BIN=… latch-rate.sh`, and `pkill -9` then
# kills the sweep mid-run. Anchor on `-enable-kvm`, which the real process always
# carries directly after the binary and no invocation of this script does, and
# skip our own pid.
kill_vm() {
  for pid in $(pgrep -f 'qemu-system-x86_64 -enable-kvm' 2>/dev/null || true); do
    [ "$pid" = "$$" ] && continue
    kill -9 "$pid" 2>/dev/null || true
  done
  # Give the port forward time to come back, or the next boot dies on
  # "Could not set up host forwarding rule 'tcp::2222-:22'" — and the ssh wait
  # then succeeds against nothing.
  sleep 6
}

# Sum a per-second census counter over a whole boot's fail log. Absent reads 0,
# which for these counters is the same statement as a run of zero lines.
census_total() {
  grep -o "$2=[0-9]*" "$1" 2>/dev/null | cut -d= -f2 \
    | awk '{s += $1} END {print s + 0}'
}

# Count the lines matching a pattern. Not `grep -c ... || echo 0`: grep -c
# prints its count *and* exits 1 when that count is zero, so the fallback fires
# on exactly the case it was meant to cover and the substitution yields two
# lines. That put a bare "0" between every row of the TSV. Capture first, then
# default, so a missing file still reads 0 without a zero count printing twice.
line_count() {
  local n
  n=$(grep -c "$2" "$1" 2>/dev/null) || true
  printf '%s' "${n:-0}"
}

# The probe's phases plus the boot, with headroom; the boot's own hard kill is
# the backstop that keeps a wedged guest from stalling the sweep.
budget=$(( LOAD_SECONDS + 4 * CENSUS_SECONDS + 420 ))

say "work dir $WORK — $BOOTS boots, ${LOAD_SECONDS}s load each"
verdicts="$WORK/verdicts.tsv"
: >"$verdicts"

for i in $(seq 1 "$BOOTS"); do
  boot_dir="$WORK/boot-$i"
  mkdir -p "$boot_dir"
  kill_vm
  rm -f "$FAILLOG"
  say "boot $i/$BOOTS"
  TESTING_TIMEOUT="$budget" "$REPO_ROOT/vm/boot-x86.sh" --device reims-vgpu-pci --testing \
    >"$boot_dir/boot.log" 2>&1 &
  boot_pid=$!

  waited=0
  until ssh -o ConnectTimeout=5 -o BatchMode=yes macos-vm true 2>/dev/null; do
    sleep 5
    waited=$((waited + 5))
    if [ "$waited" -ge 300 ]; then break; fi
    # A boot that died takes its guest with it; do not wait out the whole budget
    # against a process that is gone.
    kill -0 "$boot_pid" 2>/dev/null || break
  done
  if ! ssh -o ConnectTimeout=5 -o BatchMode=yes macos-vm true 2>/dev/null; then
    say "boot $i never came up — see $boot_dir/boot.log" >&2
    printf '%s\t%s\t%s\t%s\t%s\n' "$i" "no-boot" "-" "-" "-" >>"$verdicts"
    continue
  fi

  if ! "$PROBE" --load-seconds "$LOAD_SECONDS" --census-seconds "$CENSUS_SECONDS" \
      --out "$boot_dir" >"$boot_dir/probe.log" 2>&1; then
    say "boot $i: the probe refused a verdict — see $boot_dir/probe.log" >&2
    cp "$FAILLOG" "$boot_dir/full-boot.log" 2>/dev/null || true
    # The mechanism counter still reads, and it does not need the screenshots
    # the gate refused on.
    printf '%s\t%s\t%s\t%s\t%s\n' "$i" "probe-refused" "-" \
      "$(census_total "$boot_dir/full-boot.log" gw_refused_guest_store)" \
      "$(line_count "$boot_dir/full-boot.log" 'deferred_flush_clobber')" \
      >>"$verdicts"
    continue
  fi

  gate_out=$("$GATE" --before "$boot_dir/before.png" --after "$boot_dir/after.png" 2>&1) \
    && verdict="clean" || verdict="degraded"
  echo "$gate_out" >"$boot_dir/gate.log"
  rmse=$(echo "$gate_out" | sed -n 's/.*pane=[^ ]* rmse=\([0-9.e-]*\).*/\1/p')
  # The gate's own refusal (control moved) is neither clean nor degraded.
  echo "$gate_out" | grep -q "not of the same scene" && verdict="gate-refused"
  cp "$FAILLOG" "$boot_dir/full-boot.log" 2>/dev/null || true
  gw=$(census_total "$boot_dir/full-boot.log" gw_refused_guest_store)
  clobber=$(line_count "$boot_dir/full-boot.log" 'deferred_flush_clobber')
  printf '%s\t%s\t%s\t%s\t%s\n' "$i" "$verdict" "${rmse:--}" "$gw" "$clobber" >>"$verdicts"
  say "boot $i: $verdict (pane rmse ${rmse:--}, gw_refused $gw, clobber $clobber)"
done

kill_vm

say ""
say "boot	verdict	pane_rmse	gw_refused	clobber"
sed 's/^/latch-rate:   /' "$verdicts"
clean=$(awk -F'\t' '$2 == "clean"' "$verdicts" | wc -l)
degraded=$(awk -F'\t' '$2 == "degraded"' "$verdicts" | wc -l)
scored=$((clean + degraded))
say ""
if [ "$scored" -gt 0 ]; then
  say "degraded $degraded of $scored scored boots"
else
  say "no boot produced a scoreable pair"
fi
# The mechanism column, which reads on every boot that produced a fail log —
# including the ones the gate could not score. The threshold is three orders of
# magnitude below the degraded cluster and two above the clean one, so no boot
# recorded so far lands near it.
latched=$(awk -F'\t' '$4 ~ /^[0-9]+$/ && $4 > 1000' "$verdicts" | wc -l)
withlog=$(awk -F'\t' '$4 ~ /^[0-9]+$/' "$verdicts" | wc -l)
say "gw_refused_guest_store over 1000 on $latched of $withlog boots with a fail log"
say "per-boot evidence in $WORK/boot-*/ (screenshots, census logs, full-boot.log)"
exit 0
