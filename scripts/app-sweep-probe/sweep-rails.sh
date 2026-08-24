#!/usr/bin/env bash
# sweep-rails.sh — run `app-sweep-probe` across several guest OS rails, one boot
# each, and print one verdict table.
#
# The probe itself judges one boot. This is the loop around it, and everything it
# does beyond looping is a rule from `AGENTS.md` that a hand-run sweep has got
# wrong before:
#
# - **The panic verdict outranks the probe's.** A guest kernel panic can land
#   *after* the probe has finished and reported success, so `probe exit=0` is not
#   a clean boot. `vm/boot-x86.sh` prints `guest kernel panic` on its own stdout
#   and exits 126; this greps that log and reports it per rail alongside the
#   probe's exit, because a rail that panics on a third of its boots is not a
#   rail that passes.
# - **Kill the previous QEMU and wait on the fail log, not on ssh.** The previous
#   boot's QEMU outlives its script for long enough to still hold
#   `localhost:2222`; a new boot that loses that race dies on the `hostfwd` rule
#   while `ssh macos-vm true` answers at once — from the *old* VM. The probe then
#   drives the previous rail and every number is attributed to the wrong guest.
#   Only a live device creates the fail log, so waiting on it catches the case.
# - **Bracket one character of the `pkill` pattern.** `pkill -f` matches whole
#   command lines including the shell issuing it, so an unbracketed pattern kills
#   this script.
# - **Authorize every rail.** Only `macos-13` was provisioned with the ssh key;
#   the others authenticate by password and `BatchMode=yes` turns that into a
#   silent failure that reads as "the guest is not up yet".
# - **Wait for the Dock.** sshd answers well before the desktop composites, and a
#   probe started at port-open photographs the boot progress bar.
#
# Usage:
#   scripts/app-sweep-probe/sweep-rails.sh [--rails "macos-11 macos-12 ..."]
#     [--seconds N] [--torture-seconds N] [--out DIR]
set -uo pipefail
export LC_ALL=C

RAILS="macos-11 macos-12 macos-13 macos-14"
SECONDS_PER_APP=15
TORTURE=40
OUT="/tmp/reims-rail-sweep"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

while [ $# -gt 0 ]; do
  case "$1" in
    --rails) RAILS="$2"; shift 2 ;;
    --seconds) SECONDS_PER_APP="$2"; shift 2 ;;
    --torture-seconds) TORTURE="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "sweep-rails: unknown argument $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT"
SUMMARY="$OUT/summary.tsv"
: >"$SUMMARY"
say() { echo "sweep-rails: $*"; }

# Build once, pin, and hand every rail the same binary.
#
# `boot-x86.sh` rebuilds in-tree QEMU on every boot unless `QEMU_BIN` names
# something else, so a four-rail sweep used to be four builds off whatever the
# tree held at each one. Nothing forced them to agree, and the way they stop
# agreeing is not exotic: a sweep takes the better part of an hour, and anyone
# editing Rust during it silently splits the run in two — producing a verdict
# table that looks exactly like one binary's. The rails are the populations this
# table compares, so they have to be one build or the comparison is not one.
#
# Exported, never passed as argv. `pkill -f 'qemu-system-x86_6[4].*reims-vgpu'`
# matches whole command lines, so a pin named on one would match the shell that
# named it; an environment variable never appears in `/proc/pid/cmdline`. The
# pin must live in `vendor/qemu/build` because QEMU finds `pc-bios` relative to
# its own path — a copy in /tmp dies on `kvmvapic.bin`.
if [ -z "${QEMU_BIN:-}" ]; then
  say "building once and pinning, so every rail runs one binary"
  "$REPO/scripts/qemu-build/qemu-build.sh" --target x86_64 || {
    echo "sweep-rails: qemu-build failed" >&2
    exit 1
  }
  sweep_pin="$REPO/vendor/qemu/build/qemu-system-x86_64-sweep-$$"
  cp -f "$REPO/vendor/qemu/build/qemu-system-x86_64" "$sweep_pin" || exit 1
  export QEMU_BIN="$sweep_pin"
  # The pin belongs to this run, so it goes when the run does. Without this
  # every sweep leaves ~110 MB behind and they accumulate unnoticed.
  trap 'rm -f "$sweep_pin"' EXIT
  say "pinned $QEMU_BIN"
else
  say "QEMU_BIN already set ($QEMU_BIN) — every rail uses it"
fi

for rail in $RAILS; do
  say "=== $rail ==="
  # Waits for the port to be free rather than for a clock: a five-second sleep
  # here left macos-11's just-GPU-reset QEMU holding :2222, and macos-12's boot
  # died on the hostfwd rule and was reported NO-BOOT.
  "$REPO/scripts/app-sweep-probe/stop-previous-vm.sh" || \
    say "$rail: the previous VM would not let go of :2222 — this boot may fail on hostfwd"
  rm -f /tmp/reims-vgpu-fail.log
  BOOTLOG="$OUT/$rail-boot.log"
  TESTING_TIMEOUT=1200 nohup "$REPO/vm/boot-x86.sh" --device reims-vgpu-pci \
    --rail "$rail" --testing >"$BOOTLOG" 2>&1 &

  # The fail log, not ssh: only a live device writes one, so this cannot be
  # answered by a surviving QEMU from the previous rail.
  up=no
  for _ in $(seq 1 60); do
    if [ -f /tmp/reims-vgpu-fail.log ]; then up=yes; break; fi
    sleep 5
  done
  if [ "$up" != yes ]; then
    say "$rail: no device came up in 300s"
    printf '%s\tNO-BOOT\t-\t-\n' "$rail" >>"$SUMMARY"
    continue
  fi

  # Password rails need the key installed into the running clone before any
  # BatchMode probe can reach them.
  timeout 300 "$REPO/vm/guest-authorize.sh" >"$OUT/$rail-authorize.log" 2>&1 \
    || say "$rail: guest-authorize did not finish cleanly (see $OUT/$rail-authorize.log)"

  # sshd answering is not the desktop compositing, and a rail stopped at the
  # login window is not a rail still booting. `wait-for-desktop.sh` owns that
  # distinction; this loop used to make neither and reported macos-12 as
  # NO-DESKTOP for a guest that had booted.
  #
  # Its exit 3 is its own verdict and outranks everything below: a login window
  # with crash reports behind it is a **WindowServer crash**, and it says so
  # rather than logging in over the evidence. Reported per rail, because a rail
  # that crashes its window server is not a rail whose app verdicts mean
  # anything.
  "$REPO/scripts/app-sweep-probe/wait-for-desktop.sh" --timeout 400 \
    --reports "$OUT/$rail-reports"
  case $? in
    0) ;;
    3)
      say "$rail: WINDOWSERVER CRASH — reports in $OUT/$rail-reports"
      printf '%s\tWINDOWSERVER-CRASH\t-\t%s\n' "$rail" \
        "$(grep -qF 'guest kernel panic' "$BOOTLOG" && echo PANIC || echo no-panic)" >>"$SUMMARY"
      continue
      ;;
    *)
      say "$rail: the Dock never appeared"
      printf '%s\tNO-DESKTOP\t-\t%s\n' "$rail" \
        "$(grep -qF 'guest kernel panic' "$BOOTLOG" && echo PANIC || echo no-panic)" >>"$SUMMARY"
      continue
      ;;
  esac
  sleep 8   # dock and wallpaper settle

  QMP_SOCK="$REPO/vm/disks/run/qmp.sock" timeout 900 \
    "$REPO/scripts/app-sweep-probe/app-sweep-probe.sh" \
    --rail "$rail" --seconds "$SECONDS_PER_APP" --torture-seconds "$TORTURE" \
    --shots "$OUT/shots" --keep "$OUT/$rail-work" >"$OUT/$rail-probe.log" 2>&1
  probe=$?
  tail -12 "$OUT/$rail-probe.log"

  # Read the panic *after* the probe: the whole point is that it can land later.
  pkill -f 'qemu-system-x86_6[4].*reims-vgpu' 2>/dev/null
  sleep 6
  panic=$(grep -qF 'guest kernel panic' "$BOOTLOG" && echo PANIC || echo no-panic)
  printf '%s\tprobe_exit=%s\t%s\t%s\n' "$rail" "$probe" \
    "$(grep -c '^' "$OUT/$rail-probe.log")lines" "$panic" >>"$SUMMARY"
  say "$rail: probe_exit=$probe $panic"
done

echo
echo "=== rail sweep ==="
cat "$SUMMARY"
echo
echo "per-app verdicts:"
for rail in $RAILS; do
  [ -f "$OUT/$rail-probe.log" ] || continue
  echo "--- $rail ---"
  sed -n '/^app  *verdict/,$p' "$OUT/$rail-probe.log"
done
