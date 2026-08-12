#!/usr/bin/env bash
# Per-thread CPU of the running QEMU, once a second, for the length of a probe.
#
#   qemu-thread-census.sh <outfile> <seconds> [pattern]
#
# # Why this exists
#
# Every counter this device emits measures *this device*. When a driven boot
# produces a tenth of the frames a lighter one does and the drain worker's duty
# reads 0.12, the census says only that the device was idle seven eighths of the
# time — it cannot say whether the guest was computing, whether a vCPU was
# spinning on a device register, or whether everyone was asleep waiting on the
# GPU. Those three have the same signature in the fail log and different fixes,
# and one of them is not this device's bug at all.
#
# A thread census separates them, because QEMU names its threads:
#
#   CPU 0/KVM ...   one per guest vCPU. Hot means the *guest* is computing —
#                   its own JavaScript, its own driver, its own compositor.
#                   Hot on one vCPU while the others idle is a serialized guest.
#   reims-drain     this device's drain worker. Its share should agree with
#                   `drain_duty`'s `duty`, and a disagreement is itself a finding.
#   reims-window    the host window presenter.
#   qemu-main       the main loop: MMIO the shim handles inline lands here, so a
#                   hot main loop with a cold drain is a guest blocked in a
#                   register write rather than on any work this device queued.
#
# Sum the four against the wall clock and what is left is nobody running, which
# is a latency problem — a round trip, a fence, a timer — and not a throughput
# one. That distinction is the whole reason to run this.
#
# Written next to a probe, not inside one: it costs two `/proc` reads a second
# and works for any workload, including one this repository does not own.
set -u
OUT="${1:?outfile}"; SECS="${2:-30}"; PAT="${3:-qemu-system-x86_6[4].*reims-vgpu}"

PID=$(pgrep -f "$PAT" | head -1)
[ -n "$PID" ] || { echo "no qemu matching $PAT" >&2; exit 1; }
HZ=$(getconf CLK_TCK)

declare -A prev
: >"$OUT"
echo "# pid=$PID hz=$HZ" >>"$OUT"
end=$(( SECONDS + SECS ))
while [ $SECONDS -lt $end ]; do
  now=$(date +%s.%N)
  line="t=$now"
  for d in /proc/"$PID"/task/*; do
    tid=${d##*/}
    # A thread that exits between the glob and the read is not an error.
    read -r comm <"$d/comm" 2>/dev/null || continue
    stat=$(cat "$d/stat" 2>/dev/null) || continue
    # utime and stime are fields 14 and 15, but `comm` in field 2 may itself
    # contain spaces, so count back from the end of the line rather than
    # forward from its start.
    set -- $stat
    u=${14}; s=${15}
    tot=$(( u + s ))
    key="$tid"
    p=${prev[$key]:-}
    if [ -n "$p" ]; then
      d_ticks=$(( tot - p ))
      if [ "$d_ticks" -gt 0 ]; then
        line="$line ${comm//[[:space:]]/_}:$tid=$(( d_ticks * 100 / HZ ))"
      fi
    fi
    prev[$key]=$tot
  done
  echo "$line" >>"$OUT"
  sleep 1
done
