#!/usr/bin/env python3
"""The lane gate's boot capture, on the lane host (msi-boot.sh where boot.env sets CAPTURE=1): what
the display showed while the guest booted, for a boot's flicker to be read frame by frame.

    sudo -n python3 -I msi-capture.py scanout BOOTDIR RUNDIR
    python3 -I msi-capture.py shots BOOTDIR RUNDIR

BOOTDIR is rgk-boot2.sh's (qemu.pid, qmp.sock, launch.log), RUNDIR msi-boot.sh's. Each waits for
this boot's QEMU (a qemu.pid newer than RUNDIR/.start), runs until RUNDIR/capture/stop exists or
CAPTURE_MAX_S seconds (default 600) after QEMU's start, and writes into RUNDIR/capture/:

  scanout   the guest's scanout buffer before WindowServer's IOSurfaces take the display: read
            from QEMU's memory every CAPTURE_MS milliseconds (default 100) until the host's first
            present of an IOSurface (launch.log's first "PRESENT_SURFACE of transaction" or "the
            display shows IOSurface" line), after which the display's content lives in the host's
            device caches and the scanout's pages are no longer drawn, at offset 0 of the
            reims-graphics-pci allocations RAM block (its host address from HMP's `info ramblock`,
            over QMP), through /proc/PID/mem, which needs root (Yama's ptrace scope). The kext
            sets its scanout there (ReimsGraphics::setScanout, RGFX_ALLOCATIONS_OFFSET), rows of
            stride bytes, BGRA8. Only the middle third of the rows is read (the progress bar's),
            about 12 MB a frame at 3840x2400. The mode is CAPTURE_MODE (default 3840x2400, the
            lane host's guest mode), the stride the kext's rule (4 bytes a pixel, rows aligned to
            256 bytes); msi-boot.sh checks it against the kext's "scanout WxH ... stride S" line.
              scanout/NNNNNN.zst   a frame whose bytes differ from the one before, zstd
              scanout.tsv          "EPOCH SEQ FRAME" a read: FRAME the file, or "same" where the
                                   bytes equal the frame before; its header line says the block,
                                   the offset, length, stride, rows and mode
  shots     the guest's own screenshots (Command-Shift-3 over QMP, the system's screenshot, which
            WindowServer composes as the screen shows) once a second from the host's first
            present on: shots.tsv "EPOCH SEQ" a key sent (msi-boot.sh copies the PNGs back).

Both write capture/NAME.done ("frames=N distinct=M ..."), and the scanout's files are given to the
user sudo ran it for.
"""
import json
import os
import re
import socket
import sys
import time

MODE, B, R = sys.argv[1], sys.argv[2], sys.argv[3]
C = os.path.join(R, "capture")
STOP = os.path.join(C, "stop")
MAX_S = float(os.environ.get("CAPTURE_MAX_S", "600"))
PERIOD = float(os.environ.get("CAPTURE_MS", "100")) / 1000
os.makedirs(C, exist_ok=True)


def log(*a):
    with open(os.path.join(C, MODE + ".log"), "a") as f:
        print(time.strftime("%T"), *a, file=f, flush=True)


def qemu():
    """This boot's QEMU pid, once its qemu.pid is newer than RUNDIR/.start (waits for it)."""
    start = os.path.getmtime(os.path.join(R, ".start"))
    p = os.path.join(B, "qemu.pid")
    while not os.path.exists(STOP):
        try:
            if os.path.getmtime(p) >= start:
                pid = int(open(p).read().strip())
                if open(f"/proc/{pid}/comm").read().startswith("qemu-system"):
                    return pid
        except (OSError, ValueError):
            pass
        time.sleep(0.05)
    return None


def qmp(*commands, wait_s=30):
    """The answers of COMMANDS over the boot's QMP socket, which serves one client at a time."""
    end = time.time() + wait_s
    while True:
        try:
            s = socket.socket(socket.AF_UNIX)
            s.settimeout(10)
            s.connect(os.path.join(B, "qmp.sock"))
            break
        except OSError:
            if time.time() > end:
                raise
            time.sleep(0.1)
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


def give_back(paths):
    uid, gid = int(os.environ.get("SUDO_UID", os.getuid())), int(os.environ.get("SUDO_GID", os.getgid()))
    for p in paths:
        try:
            os.chown(p, uid, gid)
        except OSError:
            pass


def scanout(pid, t_end):
    from compression import zstd  # Python 3.14

    # The allocations are a RAM pointer region (memory_region_init_ram_ptr), whose RAM block has
    # no name in "info ramblock": it is the nameless block of the region's size, which "info
    # mtree" states ("START-END (prio P, ram): reims-graphics-pci.allocations") once the guest's
    # firmware has mapped the device's BAR, a moment after QEMU's start: asked again until then.
    size, blocks = None, ""
    while size is None and time.time() < t_end and not os.path.exists(STOP):
        blocks, tree = (r.get("return", "") for r in qmp(
            {"execute": "human-monitor-command", "arguments": {"command-line": "info ramblock"}},
            {"execute": "human-monitor-command", "arguments": {"command-line": "info mtree"}}))
        for line in tree.splitlines():
            m = re.search(r"([0-9a-f]+)-([0-9a-f]+) \(prio [-0-9]+, ram\): reims-graphics-pci\.allocations$", line.strip())
            if m:
                size = int(m.group(2), 16) - int(m.group(1), 16) + 1
                break
        else:
            time.sleep(0.1)
    log(f"ramblocks (the allocations {size} bytes, {time.time() - os.path.getmtime(os.path.join(B, 'qemu.pid')):.1f} s "
        f"after QEMU's start):\n" + blocks)
    # "Block Name  PSize  Offset  Used  Total  HVA  RO": a nameless block's line starts with its
    # page size; its four hexadecimal fields are offset, used, total and the host's address.
    found = []
    for line in blocks.splitlines():
        f = line.split()
        hexes = [int(x, 16) for x in f if x.startswith("0x")]
        if len(hexes) >= 4 and f[0][:1].isdigit() and hexes[1] == size:
            found.append((hexes[2], hexes[3]))
    if len(found) != 1:
        log(f"the allocations region is {size} bytes (info mtree); nameless RAM blocks of that size: {len(found)}")
        return f"frames=0 distinct=0 why=allocations-block-{'unsized' if size is None else len(found)}"
    total, hva = found[0]
    name = f"reims-graphics-pci.allocations({size:#x})"
    w, h = (int(x) for x in os.environ.get("CAPTURE_MODE", "3840x2400").split("x"))
    stride = (w * 4 + 255) & ~255
    first, rows = h // 3, (2 * h) // 3 - h // 3
    off, length = first * stride, rows * stride
    if off + length > total:
        log(f"the rows {first}+{rows} at stride {stride} pass the block's {total} bytes")
        return "frames=0 distinct=0 why=outside-the-block"
    os.makedirs(os.path.join(C, "scanout"), exist_ok=True)
    made = [os.path.join(C, "scanout")]
    idx_path = os.path.join(C, "scanout.tsv")
    made.append(idx_path)
    idx = open(idx_path, "w")
    idx.write(f"# block {name} hva {hva:#x} offset {off} length {length} stride {stride} "
              f"rows {first}-{first + rows - 1} mode {w}x{h} pid {pid}\n")
    mem = os.open(f"/proc/{pid}/mem", os.O_RDONLY)
    prev, n, distinct = None, 0, 0
    nxt = time.time()
    # The host's log, read as it grows, for the first present (the reads end there).
    lp, seen, presented = os.path.join(B, "launch.log"), 0, None
    try:
        while time.time() < t_end and not os.path.exists(STOP):
            try:
                with open(lp, "rb") as f:
                    f.seek(seen)
                    new = f.read()
                cut = new.rfind(b"\n") + 1
                m = re.search(rb"^(\S+) .*(?:PRESENT_SURFACE of transaction|the display shows IOSurface)", new[:cut], re.M)
                seen += cut
                if m:
                    presented = m.group(1).decode(errors="replace")
                    log(f"the host's first present at {presented}: the scanout's reads end")
                    break
            except OSError:
                pass
            t = time.time()
            try:
                buf = os.pread(mem, length, hva + off)
            except OSError as e:  # QEMU gone
                log(f"read: {e}")
                break
            n += 1
            if buf == prev:
                idx.write(f"{t:.3f} {n} same\n")
            else:
                distinct += 1
                fn = f"{distinct:06d}.zst"
                p = os.path.join(C, "scanout", fn)
                with open(p, "wb") as f:
                    f.write(zstd.compress(buf, level=1))
                made.append(p)
                idx.write(f"{t:.3f} {n} scanout/{fn}\n")
                prev = buf
            idx.flush()
            nxt += PERIOD
            d = nxt - time.time()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.time()
    finally:
        os.close(mem)
        idx.close()
        give_back(made)
    return (f"frames={n} distinct={distinct} block={name} stride={stride} rows={first}-{first + rows - 1} "
            f"until_first_present={presented or 'none'}")


def shots(pid, t_end):
    # From the host's first present on: WindowServer draws the screen from then.
    lp = os.path.join(B, "launch.log")
    while time.time() < t_end and not os.path.exists(STOP):
        try:
            with open(lp, errors="replace") as f:
                if "the display shows IOSurface" in f.read():
                    break
        except OSError:
            pass
        time.sleep(0.2)
    idx = open(os.path.join(C, "shots.tsv"), "w")
    n, fails = 0, 0
    nxt = time.time()
    while time.time() < t_end and not os.path.exists(STOP):
        t = time.time()
        try:
            r = qmp({"execute": "send-key", "arguments": {"keys": [{"type": "qcode", "data": k} for k in
                                                                  ("meta_l", "shift", "3")], "hold-time": 100}},
                    wait_s=2)[0]
            ok = "error" not in r
        except (OSError, ValueError) as e:
            log(f"send-key: {e}")
            ok = False
        n += 1
        fails += not ok
        idx.write(f"{t:.3f} {n}{'' if ok else ' failed'}\n")
        idx.flush()
        nxt += 1.0
        d = nxt - time.time()
        if d > 0:
            time.sleep(d)
        else:
            nxt = time.time()
    idx.close()
    return f"sent={n} failed={fails}"


give_back([C])  # its stop file is the user's to write
pid = qemu()
if pid is None:
    result = "frames=0 why=no-qemu"
else:
    start = os.path.getmtime(os.path.join(B, "qemu.pid"))
    try:
        result = (scanout if MODE == "scanout" else shots)(pid, start + MAX_S)
    except Exception as e:  # a capture that fails leaves its reason
        result = f"why={type(e).__name__}:{str(e).replace(' ', '_')[:200]}"
log(result)
done = os.path.join(C, MODE + ".done")
with open(done, "w") as f:
    f.write(result + "\n")
give_back([done, os.path.join(C, MODE + ".log"), C])
