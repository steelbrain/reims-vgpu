#!/usr/bin/env python3
# spscan: finds the pages shmem's spmark case marked in a QEMU guest's RAM, from the host, and
# says how physically contiguous each 2 MiB range is in guest-physical memory.
#   spscan.py QEMU_PID PID [BELOW_4G_BYTES]
# QEMU_PID's RAM is its largest memfd (QEMU names it "memory-backend-memfd"), the first
# BELOW_4G_BYTES (default 1 GiB, max-ram-below-4g=1G) at guest-physical 0, the rest from 4 GiB.
# PID is the guest process that marked them. Reads the file through /proc/QEMU_PID/fd.
import mmap
import os
import struct
import sys


def main():
    qpid, gpid = int(sys.argv[1]), int(sys.argv[2])
    below = int(sys.argv[3]) if len(sys.argv) > 3 else 1 << 30
    path, best = None, 0
    for fd in os.listdir(f"/proc/{qpid}/fd"):
        p = f"/proc/{qpid}/fd/{fd}"
        if os.readlink(p).startswith("/memfd:"):
            n = os.stat(p).st_size
            if n > best:
                path, best = p, n
    if path is None:
        sys.exit("no memfd:ram")
    f = open(path, "rb")
    size = os.fstat(f.fileno()).st_size
    m = mmap.mmap(f.fileno(), size, prot=mmap.PROT_READ)
    pages = {}  # (kind, index) -> {page: gpa}
    off = m.find(b"RGSPMARK", 0)
    while off != -1:
        if off % 4096 == 0:
            pid, kind, idx, page = struct.unpack_from("<QQQQ", m, off + 8)
            if pid == gpid and page < 512:
                gpa = off if off < below else off - below + (4 << 30)
                pages.setdefault((kind, idx), {})[page] = gpa
            off = m.find(b"RGSPMARK", off + 4096)
        else:
            off = m.find(b"RGSPMARK", off + 1)
    for kind, name in ((1, "superpage"), (0, "plain")):
        ranges = [v for (k, _), v in pages.items() if k == kind]
        whole = [r for r in ranges if len(r) == 512]
        runs = []
        aligned = 0
        for r in whole:
            n = 1
            for p in range(1, 512):
                if r[p] != r[p - 1] + 4096:
                    n += 1
            runs.append(n)
            aligned += r[0] % (2 << 20) == 0 and n == 1
        one = sum(1 for n in runs if n == 1)
        mean = sum(runs) / len(runs) if runs else 0
        print(f"spscan: {name} ranges={len(ranges)} whole={len(whole)} one_run={one} "
              f"one_run_2MiB_aligned={aligned} mean_runs={mean:.1f} max_runs={max(runs, default=0)}")


main()
