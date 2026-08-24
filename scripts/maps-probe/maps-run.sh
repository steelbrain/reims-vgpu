#!/usr/bin/env bash
# maps-run.sh — one unattended macos-13 x86 boot driven by the Maps probe.
#
#   maps-run.sh <outdir> [seconds]
#
# Everything a Maps measurement needs to be trustworthy, in one job, so a
# harness never has to interleave a foreground poll with the boot it is timing.
# The steps are in the order `AGENTS.md` requires and each one is here because
# skipping it produces a result that reads clean and is not:
#
#   1. kill any surviving QEMU first. One that outlives its script still holds
#      `localhost:2222`, so the next boot dies on the hostfwd rule while ssh
#      answers immediately from the OLD VM -- a driven boot of the previous
#      binary, with self-consistent counters and a working screenshot.
#   2. truncate the fail log. The device appends and never truncates, and
#      `first_sight` latches per process, so a stale log turns one refusal per
#      boot into N identical lines that rank like a finding.
#   3. wait on the fail log, not on ssh. Only a live device creates it.
#   4. authorize the guest. Every probe reaches it as `ssh -o BatchMode=yes`,
#      and only macos-13 was provisioned with that key; the call is idempotent.
#   5. wait for the desktop. sshd answers well before anything composites, so a
#      probe started at port-open photographs the boot progress bar.
#   6. grep the boot's own stdout for a guest kernel panic. A panic can land
#      after the probe reports success, so `probe exit=0` is not a clean boot
#      and this verdict outranks it.
#
# This probe produces TWO workload populations and does not say which one a
# boot landed in. Classify before comparing any two boots, the same way a
# macos-13 boot must be classified by present rate first -- this is a second,
# unrelated split, by scene rather than by VBL latch:
#
#   dense   ~62 pages per visibility read set, ~4000-5200 draws/frame, ~6-7 fps
#   sparse  ~39 pages per visibility read set, ~1800 draws/frame,     ~15 fps
#
# Every per-draw number differs about twofold between them -- `rb_visibility`
# in particular tracks pages-per-set almost exactly, because that phase IS the
# page walk. Six interleaved boots of two arms put one sparse boot in the base
# arm, and reading it beside the dense ones makes the base arm look 2x better
# on a phase neither arm changed.
#
# The discriminator is pages-per-set, summed over the census windows (these
# counters are in the `windowed` group and reset each window, so they must be
# summed and not sampled):
#
#   guest_visibility_read_pages / guest_visibility_read_sets
#
# draws/frame works too and needs no counter knowledge. Do not use present_hz:
# it separates the populations here only because they draw different amounts,
# and AGENTS.md already forbids ranking frames without draws/frame beside them.
#
# Environment passes through, so an arm is `REIMS_VGPU_X=y maps-run.sh ...`.
set -uo pipefail
export LC_ALL=C

OUT="${1:?usage: maps-run.sh <outdir> [seconds]}"
SECS="${2:-45}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAIL="${RAIL:-macos-13}"
mkdir -p "$OUT"

say() { echo "maps-run: $*"; }

pkill -f 'qemu-system-x86_6[4].*reims-vgpu' 2>/dev/null
# A killed QEMU still has to release the tap, the snapshot overlay and 2222.
for _ in $(seq 1 30); do
  pgrep -f 'qemu-system-x86_6[4].*reims-vgpu' >/dev/null || break
  sleep 1
done
rm -f /tmp/reims-vgpu-fail.log

say "booting rail=$RAIL"
TESTING_TIMEOUT="${TESTING_TIMEOUT:-900}" \
  "$REPO/vm/boot-x86.sh" --device reims-vgpu-pci --rail "$RAIL" --testing \
  >"$OUT/boot-stdout.log" 2>&1 &
BOOT_PID=$!

# The device is live once it has written its first record. Bound the wait so a
# boot that never reaches the device fails as itself rather than as a probe.
ready=0
for _ in $(seq 1 240); do
  if [ -f /tmp/reims-vgpu-fail.log ]; then ready=1; break; fi
  kill -0 "$BOOT_PID" 2>/dev/null || break
  sleep 2
done
[ "$ready" = 1 ] || { say "device never came up"; sed -n '1,40p' "$OUT/boot-stdout.log"; exit 4; }
say "device live"

timeout 300 "$REPO/vm/guest-authorize.sh" >"$OUT/authorize.log" 2>&1 \
  || { say "guest-authorize failed"; tail -20 "$OUT/authorize.log"; }

# The desktop, not the port. `pgrep -x Dock` is the same signal the other
# probes wait on; the settle after it is the wallpaper and dock animating in.
desktop=0
for _ in $(seq 1 90); do
  if timeout 20 ssh -o BatchMode=yes macos-vm 'pgrep -x Dock >/dev/null' 2>/dev/null; then
    desktop=1; break
  fi
  sleep 4
done
[ "$desktop" = 1 ] || { say "desktop never composited"; exit 5; }
say "desktop up"
sleep 8

"$REPO/scripts/maps-probe/maps-probe.sh" "$OUT" "$SECS" >"$OUT/probe.log" 2>&1
PROBE_RC=$?
say "probe exit=$PROBE_RC"

# Outranks the probe's own verdict.
if grep -q 'guest kernel panic' "$OUT/boot-stdout.log" 2>/dev/null; then
  say "PANIC"; echo panic >"$OUT/verdict.txt"
else
  say "no panic"; echo "probe=$PROBE_RC" >"$OUT/verdict.txt"
fi

pkill -f 'qemu-system-x86_6[4].*reims-vgpu' 2>/dev/null
wait "$BOOT_PID" 2>/dev/null
say "done -> $OUT"
exit "$PROBE_RC"
