#!/usr/bin/env python3
"""The lane gate's hang snapshot, on the lane host: what a guest that stopped answering was doing,
taken before its QEMU is stopped (msi-probes.sh runs it where a probe failed and the guest's SSH
no longer answers).

    python3 -I msi-hang.py QEMU_PID BOOTDIR OUTDIR [PART,...]

BOOTDIR is rgk-boot2.sh's (qmp.sock). It writes into OUTDIR:

  threads.txt     QEMU's threads, twice 1 s apart: "EPOCH TID COMM STATE UTIME STIME CPU" each
                  (/proc/PID/task/TID/{comm,stat}; times in clock ticks), so a vCPU thread
                  spinning in the guest (utime growing, stime still) reads apart from one blocked
  threads-delta.txt  each thread's utime and stime growth over that second
  registers-N.txt HMP `info registers -a` over QMP, 3 times 1 s apart (N = 1, 2, 3)
  stack-cpuK.txt  HMP `x/32gx RSP` in vCPU K's address space (QMP human-monitor-command's
                  cpu-index), RSP from registers-1.txt
  scanout.raw.zst the scanout (HMP `pmemsave` of the first CAPTURE_MODE rows of the
                  reims-graphics-pci allocations, where the kext sets it, at the guest-physical
                  address "info mtree" gives), BGRA8, zstd; scanout.png the same as an image, for
                  any panic text the guest drew there
  hang.log        what it did, and why a part is missing
PARTS (default all) picks among threads, registers, stacks and scanout.
"""
import json
import os
import re
import socket
import sys
import time
import zlib

if __name__ == "__main__":
    PID, B, O = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    os.makedirs(O, exist_ok=True)


def log(*a):
    with open(os.path.join(O, "hang.log"), "a") as f:
        print(time.strftime("%T"), *a, file=f, flush=True)


def qmp(*commands, wait_s=20):
    end = time.time() + wait_s
    while True:
        try:
            s = socket.socket(socket.AF_UNIX)
            s.settimeout(20)
            s.connect(os.path.join(B, "qmp.sock"))
            break
        except OSError:
            if time.time() > end:
                raise
            time.sleep(0.2)
    with s, s.makefile("rw") as f:
        f.readline()
        out = []
        for c in ({"execute": "qmp_capabilities"},) + commands:
            f.write(json.dumps(c) + "\n")
            f.flush()
            while True:
                r = json.loads(f.readline())
                if "event" not in r:
                    break
            out.append(r)
        return out[1:]


def hmp(line, cpu=None):
    args = {"command-line": line}
    if cpu is not None:
        args["cpu-index"] = cpu
    r = qmp({"execute": "human-monitor-command", "arguments": args})[0]
    if "error" in r:
        raise RuntimeError(r["error"].get("desc", r["error"]))
    return r.get("return", "")


def threads():
    rows = []
    t = time.time()
    for tid in sorted(os.listdir(f"/proc/{PID}/task"), key=int):
        try:
            comm = open(f"/proc/{PID}/task/{tid}/comm").read().strip()
            st = open(f"/proc/{PID}/task/{tid}/stat").read()
        except OSError:
            continue
        f = st[st.rindex(")") + 2:].split()  # fields from 3 (state) on
        rows.append((t, int(tid), comm.replace(" ", "_"), f[0], int(f[11]), int(f[12]), int(f[36])))
    return rows


def png(buf, w, h, stride):
    """BGRA8 rows of STRIDE bytes as an RGBA PNG."""
    rows = []
    for y in range(h):
        px = bytearray(buf[y * stride:y * stride + w * 4])
        px[0::4], px[2::4] = px[2::4], px[0::4]  # BGRA to RGBA
        rows.append(b"\0" + bytes(px))

    def chunk(kind, data):
        c = kind + data
        return len(data).to_bytes(4, "big") + c + zlib.crc32(c).to_bytes(4, "big")

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", w.to_bytes(4, "big") + h.to_bytes(4, "big") + bytes([8, 6, 0, 0, 0]))
            + chunk(b"IDAT", zlib.compress(b"".join(rows), 6)) + chunk(b"IEND", b""))


def part(name, fn):
    try:
        fn()
    except Exception as e:  # a part that fails leaves its reason, and the others still run
        log(f"{name}: {type(e).__name__}: {e}")


def take_threads():
    a = threads()
    time.sleep(1)
    b = threads()
    with open(os.path.join(O, "threads.txt"), "w") as f:
        for r in a + b:
            print(f"{r[0]:.3f} {r[1]} {r[2]} {r[3]} {r[4]} {r[5]} {r[6]}", file=f)
    before = {r[1]: r for r in a}
    with open(os.path.join(O, "threads-delta.txt"), "w") as f:
        print(f"# over {b[0][0] - a[0][0]:.2f} s: TID COMM STATE UTIME_GROWTH STIME_GROWTH CPU", file=f)
        for r in b:
            if r[1] in before:
                print(f"{r[1]} {r[2]} {r[3]} {r[4] - before[r[1]][4]} {r[5] - before[r[1]][5]} {r[6]}", file=f)
    spinning = [r for r in b if r[1] in before and r[4] - before[r[1]][4] > 50 and r[5] == before[r[1]][5]]
    log(f"threads: {len(b)}, of them {len(spinning)} with utime growing and stime still: "
        + " ".join(r[2] for r in spinning))


regs = {}


def take_registers():
    for n in (1, 2, 3):
        text = hmp("info registers -a")
        with open(os.path.join(O, f"registers-{n}.txt"), "w") as f:
            f.write(text)
        if n == 1:
            cpu = None
            for line in text.splitlines():
                m = re.match(r"CPU#(\d+)", line.strip())
                if m:
                    cpu = int(m.group(1))
                m = re.search(r"\b[RE]SP=([0-9a-f]+)", line)
                if m and cpu is not None:
                    regs[cpu] = int(m.group(1), 16)
            log(f"registers: {len(regs)} vCPUs with an RSP")
        if n < 3:
            time.sleep(1)


def take_stacks():
    for cpu, rsp in sorted(regs.items()):
        with open(os.path.join(O, f"stack-cpu{cpu}.txt"), "w") as f:
            f.write(hmp(f"x/32gx {rsp:#x}", cpu=cpu))


def take_scanout():
    tree = hmp("info mtree")
    base = None
    for line in tree.splitlines():
        m = re.search(r"([0-9a-f]+)-([0-9a-f]+) \(prio [-0-9]+, ram\): reims-graphics-pci\.allocations$", line.strip())
        if m:
            base = int(m.group(1), 16)
            break
    if base is None:
        raise RuntimeError("no reims-graphics-pci.allocations in info mtree")
    w, h = (int(x) for x in os.environ.get("CAPTURE_MODE", "3840x2400").split("x"))
    stride = (w * 4 + 255) & ~255
    raw = os.path.join(O, "scanout.raw")
    hmp(f"pmemsave {base:#x} {stride * h} {raw}")
    for _ in range(100):  # QEMU writes the file itself
        if os.path.exists(raw) and os.path.getsize(raw) == stride * h:
            break
        time.sleep(0.1)
    buf = open(raw, "rb").read()
    log(f"scanout: {len(buf)} bytes at guest-physical {base:#x}, {w}x{h} stride {stride}")
    with open(os.path.join(O, "scanout.png"), "wb") as f:
        f.write(png(buf, w, h, stride))
    try:
        from compression import zstd  # Python 3.14
        with open(raw + ".zst", "wb") as f:
            f.write(zstd.compress(buf, level=3))
        os.remove(raw)
    except ImportError:
        pass


if __name__ == "__main__":
    parts = (sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else "threads,registers,stacks,scanout").split(",")
    log(f"QEMU {PID}: {' '.join(parts)}")
    for name, fn in (("threads", take_threads), ("registers", take_registers), ("stacks", take_stacks),
                     ("scanout", take_scanout)):
        if name in parts:
            part(name, fn)
    log("done")
