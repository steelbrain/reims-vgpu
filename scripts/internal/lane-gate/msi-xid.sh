#!/bin/bash
# The NVIDIA driver's GPU errors (the kernel's "NVRM: Xid" lines) over a span, on the lane host:
#   msi-xid.sh FROM TO [PID]
# FROM and TO are epoch seconds. Prints one line an Xid: "TIME XID PID NAME OWN ERROR", TIME the
# kernel's (local ISO), OWN "own" where PID is the given one (a boot's QEMU), else "other", ERROR
# the driver's errorString (or "-"). Exits 3, printing nothing, where the kernel log cannot be
# read (sudo -n dmesg, then dmesg).
set -uo pipefail
from=$(date -d "@${1:?usage: msi-xid.sh FROM TO [PID]}" +%Y-%m-%dT%H:%M:%S)
to=$(date -d "@${2:?}" +%Y-%m-%dT%H:%M:%S)
own=${3:-}
k=$(sudo -n dmesg --time-format iso 2> /dev/null) || k=$(dmesg --time-format iso 2> /dev/null) || exit 3
[[ -n $k ]] || exit 3
# "2026-11-03T15:35:51,534368+02:00 NVRM: Xid (PCI:0000:01:00): 109, pid=4000599,
# name=qemu-system-x86, channel 0x0000001c, errorString CTX SWITCH TIMEOUT, Info 0x1fc022"
printf '%s\n' "$k" | awk -v a="$from" -v b="$to" -v own="$own" '
  / NVRM: Xid / {
    t = substr($1, 1, 19)
    if (t < a || t > b) next
    x = $0; sub(/.*NVRM: Xid \([^)]*\): */, "", x); sub(/[^0-9].*/, "", x)
    p = "-"; if (match($0, /pid=[0-9]+/)) p = substr($0, RSTART + 4, RLENGTH - 4)
    n = "-"; if (match($0, /name=[^,]+/)) n = substr($0, RSTART + 5, RLENGTH - 5)
    e = "-"; if (match($0, /errorString [^,]+/)) e = substr($0, RSTART + 12, RLENGTH - 12)
    gsub(/ /, "_", n)
    print t, x, p, n, (own != "" && p == own ? "own" : "other"), e
  }'
exit 0
