#!/usr/bin/env python3
"""The lane gate's window drags, on the lane host (msi-boot.sh, after the guest's run).

    msi-drag.py BOOTDIR RUNDIR      env: PORT GUEST_USER GUEST_PASS [DRAG_S] [DRAG_CASES]

BOOTDIR is rgk-boot2.sh's (qmp.sock, qemu.pid, launch.log), RUNDIR msi-boot.sh's. Each case opens
a window in the guest, finds a point that drags it, and drags it for DRAG_S seconds (default 10)
along a steady path, the pointer moved RATE times a second over QMP as a person's hand would:
  plain   a TextEdit document's window (no toolbar)
  glass   a Finder window (its toolbar is glass)
A point drags the window when a short probe drag from it moves the window by the pointer's
distance, as the window server states the window's bounds (winmon list). During the measured drag
the guest's winmon tracks the window's origin every 10 ms, and the drag counts only where the
window followed the pointer across the path's span; otherwise the case is not run (moved=0), not
a number. Writes RUNDIR/drag.txt, one line a case:
  drag CASE moved=1 composites_s= gap_p99_ms= gap_max_ms= qemu_cpu= gfx_cpu= window_moves_s= ...
the composites being the host log's display updates ("the display shows IOSurface") in the drag's
host-time span, the gaps those between them (and the span's ends), the CPU QEMU's and its
graphics thread's, in percent of a core.
"""
import datetime
import json
import math
import os
import re
import socket
import subprocess
import sys
import threading
import time

B, R = sys.argv[1], sys.argv[2]
PORT, USER, PASS = os.environ["PORT"], os.environ["GUEST_USER"], os.environ["GUEST_PASS"]
SECONDS = float(os.environ.get("DRAG_S", "10"))
CASES = os.environ.get("DRAG_CASES", "plain glass").split()
RATE = 120  # pointer moves a second
RADIUS = 150  # points: the path spans 2 RADIUS across and 1.2 RADIUS down
SOPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o",
         "PreferredAuthentications=password", "-o", "PubkeyAuthentication=no", "-o", "ConnectTimeout=5",
         "-o", "LogLevel=ERROR"]
OUT = open(os.path.join(R, "drag.txt"), "a")


def log(*a):
    print(time.strftime("%T"), *a, flush=True)


def guest(cmd, gui=False, timeout=60):
    """CMD in the guest over SSH, in the logged-in user's GUI session where gui."""
    if gui:
        cmd = f"echo '{PASS}' | sudo -S -p '' launchctl asuser $(id -u) sudo -u {USER} {cmd}"
    r = subprocess.run(["sshpass", "-p", PASS, "ssh", "-n", "-p", PORT, *SOPTS, f"{USER}@127.0.0.1", cmd],
                       capture_output=True, text=True, timeout=timeout)
    return r.stdout


class Qmp:
    def __init__(self, path):
        s = socket.socket(socket.AF_UNIX)
        s.connect(path)
        self.f = s.makefile("rw")
        self.f.readline()
        self.cmd({"execute": "qmp_capabilities"})

    def cmd(self, obj):
        self.f.write(json.dumps(obj) + "\n")
        self.f.flush()
        while True:
            r = json.loads(self.f.readline())
            if "event" not in r:
                return r

    def move(self, x, y):
        self.cmd({"execute": "input-send-event", "arguments": {"events": [
            {"type": "abs", "data": {"axis": "x", "value": int(max(0, min(32767, x / SW * 32767)))}},
            {"type": "abs", "data": {"axis": "y", "value": int(max(0, min(32767, y / SH * 32767)))}}]}})

    def button(self, down):
        self.cmd({"execute": "input-send-event",
                  "arguments": {"events": [{"type": "btn", "data": {"down": down, "button": "left"}}]}})

    def key(self, *names):
        self.cmd({"execute": "send-key", "arguments": {"keys": [{"type": "qcode", "data": n} for n in names]}})


def windows():
    """The guest's on-screen windows, front to back: (id, owner, x, y, w, h)."""
    out = []
    for line in guest("~/lane-gate/winmon list", gui=True).splitlines():
        f = line.split()
        if len(f) == 8 and f[0] == "window":
            out.append((int(f[1]), f[3], *map(float, f[4:8])))
    return out


def window(owner, wid=None):
    for w in windows():
        if (wid is None and w[1] == owner and w[4] >= 300 and w[5] >= 200) or w[0] == wid:
            return w
    return None


def pace(seconds, step):
    n = int(RATE * seconds)
    t0 = time.perf_counter()
    for i in range(n):
        step(i, n)
        d = t0 + (i + 1) / RATE - time.perf_counter()
        if d > 0:
            time.sleep(d)


def probe(q, w, x, y):
    """Whether pressing at (x, y) and moving 40 points right drags window w by those 40 points."""
    q.move(x, y)
    time.sleep(0.3)
    q.button(True)
    time.sleep(0.1)
    pace(0.25, lambda i, n: q.move(x + 40 * (i + 1) / n, y))
    time.sleep(0.3)
    after = window(None, w[0])
    pace(0.25, lambda i, n: q.move(x + 40 - 40 * (i + 1) / n, y))
    time.sleep(0.1)
    q.button(False)
    time.sleep(0.3)
    moved = after is not None and abs(after[2] - w[2] - 40) <= 12 and abs(after[3] - w[3]) <= 12
    if not moved:
        q.key("esc")  # whatever the press opened
        time.sleep(0.3)
    return moved


def threads(pid):
    t = {}
    for tid in os.listdir(f"/proc/{pid}/task"):
        try:
            comm = open(f"/proc/{pid}/task/{tid}/comm").read().strip()
            st = open(f"/proc/{pid}/task/{tid}/stat").read().rsplit(")", 1)[1].split()
            t[tid] = (comm, int(st[11]) + int(st[12]))
        except OSError:
            pass
    return t


def local_epoch(stamp):
    m = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)\.(\d{3})([+-])(\d\d)(\d\d)", stamp)
    if not m:
        return None
    t = datetime.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")
    off = (int(m.group(4)) * 60 + int(m.group(5))) * 60 * (1 if m.group(3) == "+" else -1)
    return (t - datetime.datetime(1970, 1, 1)).total_seconds() - off + int(m.group(2)) / 1000


def composites(t0, t1):
    out = []
    with open(os.path.join(B, "launch.log"), errors="replace") as f:
        for line in f:
            if "the display shows IOSurface" in line:
                t = local_epoch(line[:28])
                if t is not None and t0 <= t <= t1:
                    out.append(t)
    return out


def case(q, name):
    if name == "plain":
        owner = "TextEdit"
        guest("defaults write com.apple.TextEdit NSShowAppCentricOpenPanelInsteadOfUntitledFile -bool false")
        guest("open -a TextEdit", gui=True)
        ys = [12, 8, 18]
    else:
        owner = "Finder"
        guest("open ~/Documents", gui=True)
        ys = [12, 26, 8, 40]
    w = None
    for _ in range(15):
        time.sleep(1)
        w = window(owner)
        if w:
            break
    if not w:
        return {"moved": 0, "why": f"no {owner} window"}
    wid, _, wx, wy, ww, wh = w
    log(name, "window", w)
    press = None
    for fy in ys:
        for fx in (0.5, 0.62, 0.38, 0.75, 0.25, 0.85):
            x, y = wx + ww * fx, wy + fy
            if probe(q, window(None, wid) or w, x, y):
                press = (x, y)
                break
        if press:
            break
    if not press:
        return {"moved": 0, "why": "no point of the title bar dragged the window"}
    w = window(None, wid)
    x0, y0 = press[0], press[1]
    log(name, "press", press, "window", w)
    # The guest tracks the window's origin while the host drags it.
    tracker = {}
    th = threading.Thread(target=lambda: tracker.update(
        out=guest(f"~/lane-gate/winmon track {wid} {SECONDS + 3:.0f} 10", gui=True, timeout=SECONDS + 30)))
    th.start()
    time.sleep(1)
    pid = int(open(os.path.join(B, "qemu.pid")).read())
    q.move(x0, y0)
    time.sleep(0.3)
    q.button(True)
    time.sleep(0.1)
    a, t0 = threads(pid), time.time()
    pace(SECONDS, lambda i, n: q.move(x0 + RADIUS * math.sin(2 * math.pi * i / (RATE * 2)),
                                      y0 + 0.6 * RADIUS * (1 - math.cos(2 * math.pi * i / (RATE * 2)))))
    t1, b = time.time(), threads(pid)
    q.move(x0, y0)
    time.sleep(0.05)
    q.button(False)
    th.join()
    if name == "plain":
        guest("killall TextEdit", gui=True)
    else:
        q.key("meta_l", "w")  # the Finder window, frontmost
    # Did the window follow the pointer across the path?
    xs, ys_, moves = [], [], 0
    for line in tracker.get("out", "").splitlines():
        f = line.split()
        if len(f) == 4 and f[0] == "at":
            xs.append(float(f[2]))
            ys_.append(float(f[3]))
        m = re.match(r"track samples=(\d+) moves=(\d+)", line)
        if m:
            moves = int(m.group(2))
    xr = max(xs) - min(xs) if xs else 0
    yr = max(ys_) - min(ys_) if ys_ else 0
    res = {"x_range": round(xr), "y_range": round(yr), "window_moves_s": round(moves / SECONDS, 1),
           "press": f"{x0:.0f},{y0:.0f}", "seconds": round(t1 - t0, 2)}
    if xr < 0.7 * 2 * RADIUS or yr < 0.7 * 1.2 * RADIUS:
        res.update(moved=0, why=f"the window's origin spanned {xr:.0f} x {yr:.0f} points, the pointer's "
                                f"{2 * RADIUS} x {1.2 * RADIUS:.0f}")
        return res
    hz = os.sysconf("SC_CLK_TCK")
    span = t1 - t0
    cpu = {"qemu": 0, "gfx": 0}
    for tid, (comm, ticks) in b.items():
        d = ticks - a.get(tid, (comm, ticks))[1]
        cpu["qemu"] += d
        if comm.startswith("reims-vgpu.gra"):
            cpu["gfx"] += d
    ts = composites(t0, t1)
    gaps = sorted(y - x for x, y in zip([t0] + ts, ts + [t1]))
    p99 = gaps[min(len(gaps) - 1, int(math.ceil(0.99 * len(gaps))) - 1)] if gaps else span
    res.update(moved=1, composites_s=round(len(ts) / span, 2), gap_p99_ms=round(1000 * p99, 1),
               gap_max_ms=round(1000 * (gaps[-1] if gaps else span), 1),
               qemu_cpu=round(100 * cpu["qemu"] / hz / span, 1), gfx_cpu=round(100 * cpu["gfx"] / hz / span, 1))
    return res


SW, SH = 1920.0, 1200.0
m = re.search(r"screen (\d+) (\d+)", guest("~/lane-gate/winmon screen", gui=True))
if m:
    SW, SH = float(m.group(1)), float(m.group(2))
log("screen", SW, SH)
q = Qmp(os.path.join(B, "qmp.sock"))
for name in CASES:
    try:
        res = case(q, name)
    except Exception as e:  # a case that cannot run is not a number
        res = {"moved": 0, "why": f"{type(e).__name__}: {e}"}
    why = res.pop("why", None)
    line = f"drag {name} " + " ".join(f"{k}={v}" for k, v in res.items()) + (f" why={json.dumps(why)}" if why else "")
    log(line)
    OUT.write(line + "\n")
    OUT.flush()
