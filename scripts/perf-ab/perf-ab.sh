#!/usr/bin/env bash
# perf-ab.sh — interleaved A/B of driven boots, scored on the census fields that
# rank a graphical change on this device.
#
# `hang-bisect.sh` is the same boot loop scored on kernel `GPU HANG` lines; this
# is scored on frames and per-draw cost, and the differences between the two are
# all rules from `AGENTS.md`:
#
# - **Interleave the arms.** A macos-13 boot latches one of two compositing
#   regimes for its life and the *rate* at which it picks the slow one drifts
#   across a day, so an arm measured in a block is measured against a different
#   base rate from the other. Round-robin is the only ordering that cancels it.
# - **Quote `present_hz` and `offered_hz` together.** The presenter passes
#   everything it is offered, so `present_hz` alone cannot separate "the device
#   published more frames" from "the presenter stopped being a ceiling".
# - **Classify the boot before comparing it.** A slow-regime boot halves every
#   frame-rate field and reads ~10 % higher on `us/draw` for reasons that have
#   nothing to do with the change. The `regime` column here is the discriminator
#   and per-draw numbers must be scored within the fast population only.
# - **Say which probe a number came from.** A bursty interaction probe and a
#   sustained one are two populations of draws that rank the device's costs in a
#   different *order*, so the probe is a column and not a preamble.
# - **Bracket one character of every `pkill -f` pattern** and pass the arm as an
#   exported variable, never as argv: an ancestor command line naming the pin is
#   matched by the `pkill` the next boot issues.
#
# Usage:
#   scripts/perf-ab/perf-ab.sh [--rail NAME] [--arms "shipping FOO=off"]
#     [--rounds N] [--probe PATH] [--seconds N] [--out DIR]
#
# An arm is `shipping` or comma-joined `NAME=value` pairs exported as
# `REIMS_VGPU_NAME=value`. One boot per arm per round.
#
# A pair whose name already begins with `INTEL_` is exported verbatim instead,
# which is how a *driver* arm gets priced against a device arm in one
# interleaved run. That matters because the thing being priced may not be one of
# this crate's switches at all: the macos-11/12 GPU hang is Mesa's SIMD16/32
# shader codegen, and what it would cost to work around is a question about
# `INTEL_DEBUG=no16,no32` and nothing this device spells. Running the two arms as
# separate blocks would answer it against two different base rates of the
# slow-regime draw, which is exactly what the interleaving rule above exists to
# prevent.
#
# A pair whose name begins with `MAPS_` is likewise exported verbatim, which is
# how a *presentation regime* gets priced instead of a switch:
# `--probe scripts/maps-probe/maps-probe.sh --arms "MAPS_PRESENTATION=fullscreen
# MAPS_PRESENTATION=windowed"` interleaves the two regimes the Maps goal names.
#
# Values use `+` where the variable wants a comma, because the arm string is
# already comma-separated: `INTEL_DEBUG=no16+no32` exports `INTEL_DEBUG=no16,no32`.
set -uo pipefail
export LC_ALL=C

RAIL="macos-13"
ARMS="shipping"
ROUNDS=3
SECS=40
OUT="/tmp/reims-perf-ab"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROBE="$REPO/scripts/sustained-animation-probe/sustained-animation-probe.sh"

while [ $# -gt 0 ]; do
  case "$1" in
    --rail) RAIL="$2"; shift 2 ;;
    --arms) ARMS="$2"; shift 2 ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --probe) PROBE="$2"; shift 2 ;;
    --seconds) SECS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,35p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "perf-ab: unknown argument $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT"
say() { echo "perf-ab: $*"; }

# Build once and pin, so every boot in the run is the same binary.
#
# Without this each boot rebuilds from whatever the Rust tree says at that
# moment, so an edit landing mid-run silently splits the sweep into two arms
# nobody declared — and the split is invisible in the results, which is what
# makes it worth a dozen lines here. It also costs one build instead of
# `rounds x arms` of them.
#
# The pin lives in the build tree because QEMU resolves its BIOS datadir
# relative to its own path; a copy in /tmp fails to find `kvmvapic.bin` and the
# guest never starts. And it crosses to the boot as an exported variable rather
# than on a command line, because any argv naming `qemu-system-x86_64-...` is
# matched by the `pkill -f 'qemu-system-x86_6[4].*reims-vgpu'` that the next boot
# issues, which kills this runner instead of the previous VM.
if [ -z "${QEMU_BIN:-}" ]; then
  say "building QEMU once for the whole run ..."
  if ! "$REPO/scripts/qemu-build/qemu-build.sh" \
       >"$OUT/qemu-build.log" 2>&1; then
    say "qemu build failed; see $OUT/qemu-build.log"
    exit 2
  fi
  PIN="$REPO/vendor/qemu/build/qemu-system-x86_64-perfab-$$"
  cp "$REPO/vendor/qemu/build/qemu-system-x86_64" "$PIN" || exit 2
  export QEMU_BIN="$PIN"
fi
say "pinned QEMU_BIN=$QEMU_BIN"

RESULTS="$OUT/results.tsv"
PROBE_NAME="$(basename "$PROBE" .sh)"
printf 'round\tarm\tregime\tpresent_hz\toffered_hz\tdraws_s\tus_draw\tduty\tchain_us\tsampled_pct\tengine_pct\tstore_pct\tbinds_pct\tprobe\n' >"$RESULTS"

# Median of a numeric column read from the fail log, over the census windows the
# probe's own wall clock covers. Median rather than mean because one census
# second inside a stall is worth more than every healthy second put together and
# would drag a mean anywhere.
median() { sort -n | awk '{v[NR]=$1} END{ if(NR==0){print "-"} else if(NR%2){printf "%.2f", v[(NR+1)/2]} else {printf "%.2f",(v[NR/2]+v[NR/2+1])/2} }'; }

# Sum one field from one per-window census record. Field names are not globally
# unique: `draws` appears in both `drain_duty` and `draw_phase`, and `busy_us` in
# both `drain_duty` and `gpu_span`. Scoping at the record is part of the metric's
# contract; summing every spelling silently double-counts different instruments.
sum_tag_field() {
  awk -v tag="$1" -v field="$2" '
    $1 == "OFF" && $2 == tag {
      for (i = 3; i <= NF; i++) {
        split($i, pair, "=")
        if (pair[1] == field) sum += pair[2]
      }
    }
    END { print sum + 0 }
  ' "$3"
}

for round in $(seq 1 "$ROUNDS"); do
for arm in $ARMS; do
  tag="r${round}-${arm//[^A-Za-z0-9]/_}"
  say "=== round $round arm $arm ==="
  "$REPO/scripts/app-sweep-probe/stop-previous-vm.sh" || say "$tag: previous VM still holds :2222"
  rm -f /tmp/reims-vgpu-fail.log
  # Every namespace is swept, or an `INTEL_*` or `MAPS_*` arm would leak into
  # every later round and quietly make the whole run one arm.
  for stale in $(env | sed -n 's/^\(REIMS_VGPU_[A-Z0-9_]*\)=.*/\1/p'); do unset "$stale"; done
  for stale in $(env | sed -n 's/^\(INTEL_[A-Z0-9_]*\)=.*/\1/p'); do unset "$stale"; done
  for stale in $(env | sed -n 's/^\(MAPS_[A-Z0-9_]*\)=.*/\1/p'); do unset "$stale"; done
  if [ "$arm" != shipping ]; then
    old_ifs=$IFS; IFS=,
    for pair in $arm; do
      name="${pair%%=*}"; value="${pair##*=}"
      case "$name" in
        # A driver variable is exported under its own name; `+` becomes the
        # comma the arm string could not carry.
        INTEL_*) export "$name=${value//+/,}" ;;
        # A probe parameter is likewise its own name. This is how a *regime*
        # gets priced rather than a device switch: `MAPS_PRESENTATION=windowed`
        # against `MAPS_PRESENTATION=fullscreen` are two different workloads --
        # a composited window inset in a desktop, and an application owning the
        # scanout -- and the goal for this device names both. Running them as
        # separate blocks would score each against a different base rate of the
        # slow-regime draw, which is the thing the interleaving above exists to
        # cancel.
        MAPS_*)  export "$name=${value//+/,}" ;;
        *)       export "REIMS_VGPU_$name=$value" ;;
      esac
    done
    IFS=$old_ifs
  fi

  BOOTLOG="$OUT/$tag-boot.log"
  TESTING_TIMEOUT=900 nohup "$REPO/vm/boot-x86.sh" --device reims-vgpu-pci \
    --rail "$RAIL" --testing >"$BOOTLOG" 2>&1 &

  up=no
  for _ in $(seq 1 60); do
    [ -f /tmp/reims-vgpu-fail.log ] && { up=yes; break; }
    sleep 5
  done
  [ "$up" = yes ] || { say "$tag: no device"; printf '%s\t%s\tNO-BOOT\n' "$round" "$arm" >>"$RESULTS"; continue; }

  timeout 300 "$REPO/vm/guest-authorize.sh" >/dev/null 2>&1
  "$REPO/scripts/app-sweep-probe/wait-for-desktop.sh" --timeout 400 \
    --reports "$OUT/$tag-reports" \
    || { say "$tag: no desktop"; printf '%s\t%s\tNO-DESKTOP\n' "$round" "$arm" >>"$RESULTS"; \
         pkill -f 'qemu-system-x86_6[4].*reims-vgpu'; sleep 6; continue; }
  # A desktop is not a settled device. One boot scored here read 3.5 Hz and
  # 1 738 draws/s against a population of 50-58 Hz and 40-54 000 — and its slice
  # holds 69 `air_loading` records and 6 translates, so the probe window was
  # spent compiling the guest's shaders rather than drawing with them. The wait
  # is long enough to put that behind the mark, and the cost is one settle per
  # boot against an outlier that has to be found and excluded by hand.
  sleep 25

  # Everything before this point is boot noise; the scored window starts here.
  MARK=$(grep -c '' /tmp/reims-vgpu-fail.log)
  QMP_SOCK="$REPO/vm/disks/run/qmp.sock" timeout $((SECS + 240)) \
    "$PROBE" "$OUT/$tag-work" "$SECS" >"$OUT/$tag-probe.log" 2>&1
  probe_exit=$?
  SLICE="$OUT/$tag-slice.log"
  PROBE_SLICE="$OUT/$tag-probe-slice.log"
  tail -n +"$((MARK + 1))" /tmp/reims-vgpu-fail.log >"$PROBE_SLICE"
  # A probe may publish the exact interval in which it drove the guest. Prefer
  # that contract over the broad launch-to-exit slice, which can include prompt
  # dismissal, application setup, and post-interaction settling. Probes without
  # an exact window retain the broad slice.
  EXACT_SLICE="$OUT/$tag-work/window.log"
  if [ -s "$EXACT_SLICE" ]; then
    cp "$EXACT_SLICE" "$SLICE"
  else
    cp "$PROBE_SLICE" "$SLICE"
  fi
  pkill -f 'qemu-system-x86_6[4].*reims-vgpu'; sleep 6

  # Probe failure invalidates the population. In particular, Maps returns
  # nonzero when the captured frames do not contain the declared geographic
  # scene. Keeping its backend counters would reward an empty canvas for being
  # cheap, so publish an explicitly unrankable row instead.
  if [ "$probe_exit" -ne 0 ]; then
    printf '%s\t%s\tINVALID-PROBE\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t%s\n' \
      "$round" "$arm" "$PROBE_NAME" >>"$RESULTS"
    say "$tag: invalid probe (exit=$probe_exit); see $OUT/$tag-probe.log and $OUT/$tag-work"
    continue
  fi

  present=$(grep -ho 'present_hz=[0-9.]*' "$SLICE" | cut -d= -f2 | median)
  offered=$(grep -ho 'offered_hz=[0-9.]*' "$SLICE" | cut -d= -f2 | median)
  # A guest kernel panic outranks every number below it, and it can land after
  # the probe has already reported success.
  panic=no; grep -q 'guest kernel panic' "$BOOTLOG" && panic=yes

  draws=$(sum_tag_field drain_duty draws "$SLICE")
  draw_us=$(sum_tag_field drain_duty draw_us "$SLICE")
  busy_us=$(sum_tag_field drain_duty busy_us "$SLICE")
  windows=$(grep -c 'OFF drain_duty' "$SLICE")
  chain=$(sum_tag_field chain_phase chains "$SLICE")
  sampled=$(sum_tag_field chain_phase sampled_us "$SLICE")
  engine=$(sum_tag_field chain_phase engine_us "$SLICE")
  store=$(sum_tag_field chain_phase store_us "$SLICE")
  binds=$(sum_tag_field chain_phase binds_us "$SLICE")
  total_us=$((sampled + engine + store + binds))

  read -r draws_s us_draw duty chain_us s_pct e_pct st_pct b_pct <<EOF
$(awk -v d="$draws" -v du="$draw_us" -v b="$busy_us" -v w="$windows" -v c="$chain" \
      -v s="$sampled" -v e="$engine" -v st="$store" -v bi="$binds" -v t="$total_us" \
  'BEGIN{ printf "%.1f %.2f %.3f %.2f %.1f %.1f %.1f %.1f",
      (w? d/w : 0), (d? du/d : 0), (w? b/(w*1000000) : 0), (c? (s+e+st+bi)/c : 0),
      (t? 100*s/t : 0), (t? 100*e/t : 0), (t? 100*st/t : 0), (t? 100*bi/t : 0) }')
EOF

  # The regime is the discriminator, not a summary: on macos-13 the two
  # populations are empty between roughly 61 and 94 Hz, so anything in between
  # is `?` and is a boot to look at rather than to bin.
  regime=$(awk -v p="$present" 'BEGIN{ if(p=="-"){print "none"} else if(p<70){print "slow"} else if(p>90){print "fast"} else {print "?"} }')
  [ "$panic" = yes ] && regime="PANIC"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$round" "$arm" "$regime" "$present" "$offered" "$draws_s" "$us_draw" "$duty" \
    "$chain_us" "$s_pct" "$e_pct" "$st_pct" "$b_pct" "$PROBE_NAME" >>"$RESULTS"
  say "$tag: regime=$regime present=$present offered=$offered draws/s=$draws_s us/draw=$us_draw duty=$duty probe_exit=$probe_exit"
done
done

echo
echo "=== perf A/B on $RAIL, probe $PROBE_NAME ==="
column -t "$RESULTS" 2>/dev/null || cat "$RESULTS"
echo
echo "Score per-draw columns within the fast population only; quote present_hz"
echo "and offered_hz together. A PANIC row is not a measurement."
