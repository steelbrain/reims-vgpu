#!/usr/bin/env python3
"""The lane gate's window drags, on the lane host (msi-boot.sh, after the guest's run).

    msi-drag.py BOOTDIR RUNDIR      env: PORT GUEST_USER GUEST_PASS [DRAG_S] [DRAG_CASES]

BOOTDIR is rgk-boot2.sh's (qmp.sock, qemu.pid, launch.log), RUNDIR msi-boot.sh's. Each case moves
the desktop for DRAG_S seconds (default 10) as a person would, over QMP, the pointer moved RATE
times a second along a steady path:
  plain   drags a TextEdit document's window (no toolbar) by its title bar
  glass   drags a Finder window (its toolbar is glass) by its title bar
  resize  resizes a TextEdit document's window live by its bottom right corner
  mission opens and closes Mission Control (Control-Up) once a second, TextEdit's and Finder's
          windows on the desktop
A point drags (or resizes) the window when a short probe from it moves the window's origin (or
its bottom right corner) by the pointer's distance, as the window server states the window's
bounds (winmon list). During the measured drag the guest's winmon tracks the window's origin (or
size) every 10 ms, and the case counts only where the window followed the pointer across the
path's span; otherwise the case is not run (moved=0), not a number. Mission Control counts only
where the display updated after a first Control-Up. Writes RUNDIR/drag.txt, one line a case:
  drag CASE moved=1 composites_s= gap_p99_ms= gap_max_ms= qemu_cpu= gfx_cpu= window_moves_s= ...
the composites being the host log's display updates ("the display shows IOSurface") in the
case's host-time span, the gaps those between them (and the span's ends), the CPU QEMU's (the
process's own, its exited threads' included) and its graphics threads' (each to its last reading,
Window), in percent of a core, and cpu_rule=2, that rule's number (report.py compares the CPU
columns only across runs of one rule: before it, rule 1, a thread born or gone inside the window
counted nothing). Mission Control's numbers are over its animations: each
from the first display update after a Control-Up to its last within 0.8 s (anim_ms their median
length, start_ms the median wait from the key to the first update, animated those with two
updates or more); its CPU over the whole case.
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
CASES = os.environ.get("DRAG_CASES", "plain glass resize mission").split()
RATE = 120  # pointer moves a second
# How qemu_cpu and gfx_cpu are counted (Window): 2 since QEMU's own utime+stime and each graphics
# thread's last reading; 1 before, each thread's ticks between the window's two looks.
CPU_RULE = 2
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


class Window:
    """QEMU's CPU over a measured window. The process's own utime+stime (/proc/PID/stat, which keeps
    what its exited threads used) at both ends gives QEMU's total. Each graphics thread's ticks are
    its last reading: the known graphics threads' every 20 ms, and every 100 ms a scan of the names
    finds new ones (a thread can name itself after it starts). A thread born in the window counts
    from 0, and one that exits in it counts to its last reading (at most 20 ms of its time lost),
    where a bare look at the window's ends counted the first as 0 and missed the second."""

    GFX = "reims-vgpu.gra"

    def __init__(self, pid):
        self.pid, self.seen, self.gfx, self.done, self.th = pid, {}, set(), threading.Event(), None

    def proc(self):
        st = open(f"/proc/{self.pid}/stat").read().rsplit(")", 1)[1].split()
        return ("_proc", int(st[11]) + int(st[12]))

    def scan(self):
        for tid in os.listdir(f"/proc/{self.pid}/task"):
            try:
                if open(f"/proc/{self.pid}/task/{tid}/comm").read().startswith(self.GFX):
                    self.gfx.add(tid)
            except OSError:
                pass

    def read(self):
        for tid in list(self.gfx):
            try:
                comm = open(f"/proc/{self.pid}/task/{tid}/comm").read().strip()
                st = open(f"/proc/{self.pid}/task/{tid}/stat").read().rsplit(")", 1)[1].split()
                self.seen[tid] = (comm, int(st[11]) + int(st[12]))
            except OSError:
                self.gfx.discard(tid)

    def run(self):
        k = 0
        while not self.done.wait(0.02):
            k += 1
            if k % 5 == 0:
                self.scan()
            self.read()

    def start(self):
        a = threads(self.pid)
        self.seen = dict(a)
        self.gfx = {tid for tid, (comm, _) in a.items() if comm.startswith(self.GFX)}
        a["_proc"] = self.proc()
        self.th = threading.Thread(target=self.run, daemon=True)
        self.th.start()
        return a

    def stop(self):
        self.done.set()
        self.th.join()
        self.scan()
        self.read()
        b = dict(self.seen)
        b["_proc"] = self.proc()
        return b


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
    if name == "resize":
        return resize(q)
    if name == "mission":
        return mission(q)
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
    win = Window(pid)
    a, t0 = win.start(), time.time()
    pace(SECONDS, lambda i, n: q.move(x0 + RADIUS * math.sin(2 * math.pi * i / (RATE * 2)),
                                      y0 + 0.6 * RADIUS * (1 - math.cos(2 * math.pi * i / (RATE * 2)))))
    t1, b = time.time(), win.stop()
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
    res.update(measured(a, b, t0, t1, [(t0, t1)]))
    return res


def measured(a, b, t0, t1, spans):
    """The display updates in SPANS (their rate over the spans' time, the gaps between them and
    each span's ends), and QEMU's CPU from t0 to t1 (Window's start() a at t0, stop() b at t1)."""
    hz = os.sysconf("SC_CLK_TCK")
    # QEMU's from the process's own ticks; the graphics threads' from their last samples, a thread
    # born in the window counted from 0 (Window).
    cpu = {"qemu": b["_proc"][1] - a["_proc"][1], "gfx": 0}
    for tid, (comm, ticks) in b.items():
        if tid != "_proc" and comm.startswith("reims-vgpu.gra"):
            cpu["gfx"] += ticks - a.get(tid, (comm, 0))[1]
    n, gaps, total = 0, [], 0.0
    for s0, s1 in spans:
        ts = composites(s0, s1)
        n += len(ts)
        total += s1 - s0
        gaps += [y - x for x, y in zip([s0] + ts, ts + [s1])]
    gaps.sort()
    p99 = gaps[min(len(gaps) - 1, int(math.ceil(0.99 * len(gaps))) - 1)] if gaps else total
    # t0 and t1, the window's ends on the host's clock, for the report's counts over it (the host's
    # work lines, report.py's work).
    return dict(moved=1, t0=round(t0, 3), t1=round(t1, 3), composites_s=round(n / total, 2), gap_p99_ms=round(1000 * p99, 1),
                gap_max_ms=round(1000 * (gaps[-1] if gaps else total), 1),
                qemu_cpu=round(100 * cpu["qemu"] / hz / (t1 - t0), 1),
                gfx_cpu=round(100 * cpu["gfx"] / hz / (t1 - t0), 1), cpu_rule=CPU_RULE)


def textedit():
    """A TextEdit document's window, opened: (id, owner, x, y, w, h), or None."""
    guest("defaults write com.apple.TextEdit NSShowAppCentricOpenPanelInsteadOfUntitledFile -bool false")
    guest("open -a TextEdit", gui=True)
    for _ in range(15):
        time.sleep(1)
        w = window("TextEdit")
        if w:
            return w
    return None


def probe_resize(q, w, x, y):
    """Whether pressing at (x, y) and moving 40 points right and 30 down resizes window w by those
    points, its origin kept."""
    q.move(x, y)
    time.sleep(0.3)
    q.button(True)
    time.sleep(0.1)
    pace(0.25, lambda i, n: q.move(x + 40 * (i + 1) / n, y + 30 * (i + 1) / n))
    time.sleep(0.3)
    after = window(None, w[0])
    pace(0.25, lambda i, n: q.move(x + 40 - 40 * (i + 1) / n, y + 30 - 30 * (i + 1) / n))
    time.sleep(0.1)
    q.button(False)
    time.sleep(0.3)
    return (after is not None and abs(after[4] - w[4] - 40) <= 12 and abs(after[5] - w[5] - 30) <= 12
            and abs(after[2] - w[2]) <= 4 and abs(after[3] - w[3]) <= 4)


def resize(q):
    w = textedit()
    if not w:
        return {"moved": 0, "why": "no TextEdit window"}
    wid, _, wx, wy, ww, wh = w
    log("resize", "window", w)
    if wx + ww + RADIUS > SW or wy + wh + 1.2 * RADIUS > SH or ww < RADIUS + 200 or wh < 200:
        guest("killall TextEdit", gui=True)
        return {"moved": 0, "why": f"the window {ww:.0f} x {wh:.0f} at {wx:.0f},{wy:.0f} has no room to grow by "
                                   f"{RADIUS} x {1.2 * RADIUS:.0f} on a {SW:.0f} x {SH:.0f} screen"}
    press = None
    for d in (-2, 1, -5, 3, -8):
        x, y = wx + ww + d, wy + wh + d
        if probe_resize(q, window(None, wid) or w, x, y):
            press = (x, y)
            break
    if not press:
        guest("killall TextEdit", gui=True)
        return {"moved": 0, "why": "no point at the window's bottom right corner resized it"}
    w = window(None, wid)
    x0, y0 = press
    log("resize", "press", press, "window", w)
    tracker = {}
    th = threading.Thread(target=lambda: tracker.update(
        out=guest(f"~/lane-gate/winmon track {wid} {SECONDS + 3:.0f} 10 size", gui=True, timeout=SECONDS + 30)))
    th.start()
    time.sleep(1)
    pid = int(open(os.path.join(B, "qemu.pid")).read())
    q.move(x0, y0)
    time.sleep(0.3)
    q.button(True)
    time.sleep(0.1)
    win = Window(pid)
    a, t0 = win.start(), time.time()
    pace(SECONDS, lambda i, n: q.move(x0 + RADIUS * math.sin(2 * math.pi * i / (RATE * 2)),
                                      y0 + 0.6 * RADIUS * (1 - math.cos(2 * math.pi * i / (RATE * 2)))))
    t1, b = time.time(), win.stop()
    q.move(x0, y0)
    time.sleep(0.05)
    q.button(False)
    th.join()
    guest("killall TextEdit", gui=True)
    # The guest's samples as winmon gave them ("at T W H"), for a size outside the pointer's path
    # to be read later.
    with open(os.path.join(R, "drag-resize-track.txt"), "w") as f:
        f.write(f"# press {x0:.0f},{y0:.0f} window {w} drag {t0:.3f}-{t1:.3f}\n" + tracker.get("out", ""))
    # The sizes during the drag only: at the button's release the window can grow on its own to
    # the screen's visible edges (seen: 603 x 505 to 1707 x 1023 in 250 ms, from the release on),
    # which w_after says. The guest's clock is within 0.2 s of the host's (clock.txt); the last
    # 0.1 s of the path is near its start, so leaving it out keeps the ranges.
    ws_, hs, after, moves = [], [], [], 0
    for line in tracker.get("out", "").splitlines():
        f = line.split()
        if len(f) == 4 and f[0] == "at":
            if float(f[1]) <= t1 - 0.1:
                ws_.append(float(f[2]))
                hs.append(float(f[3]))
            else:
                after.append(float(f[2]))
        m = re.match(r"track samples=(\d+) moves=(\d+)", line)
        if m:
            moves = int(m.group(2))
    wr = max(ws_) - min(ws_) if ws_ else 0
    hr = max(hs) - min(hs) if hs else 0
    res = {"w_range": round(wr), "h_range": round(hr), "w": f"{min(ws_, default=0):.0f}-{max(ws_, default=0):.0f}",
           "h": f"{min(hs, default=0):.0f}-{max(hs, default=0):.0f}", "w_after": f"{max(after, default=0):.0f}",
           "window_moves_s": round(moves / SECONDS, 1), "press": f"{x0:.0f},{y0:.0f}", "seconds": round(t1 - t0, 2)}
    if wr < 0.7 * 2 * RADIUS or hr < 0.7 * 1.2 * RADIUS:
        res.update(moved=0, why=f"the window's size spanned {wr:.0f} x {hr:.0f} points, the pointer's "
                                f"{2 * RADIUS} x {1.2 * RADIUS:.0f}")
        return res
    res.update(measured(a, b, t0, t1, [(t0, t1)]))
    return res


def all_windows():
    """Every on-screen window but the desktop's: {id: "OWNER LAYER X Y W H"}."""
    out = {}
    for line in guest("~/lane-gate/winmon all", gui=True).splitlines():
        f = line.split()
        if len(f) == 9 and f[0] == "window":
            out[f[1]] = " ".join(f[3:])
    return out


def mission(q):
    w = textedit()
    guest("open ~/Documents", gui=True)
    time.sleep(2)
    if not w or not window("Finder"):
        guest("killall TextEdit", gui=True)
        return {"moved": 0, "why": "no TextEdit or Finder window to lay out"}
    q.move(SW / 2, SH / 2)
    before = all_windows()
    # A first Control-Up, unmeasured: the display must update after it.
    t = time.time()
    q.key("ctrl", "up")
    time.sleep(1.0)
    during = all_windows()
    q.key("ctrl", "up")
    time.sleep(1.5)
    shown = len(composites(t, t + 0.8))
    new = [f"{k} {v}" for k, v in during.items() if before.get(k) != v]
    log("mission", f"{shown} display updates after the first Control-Up; windows new or moved in it: {new}")
    if shown < 3:
        guest("killall TextEdit", gui=True)
        q.key("meta_l", "w")
        return {"moved": 0, "why": f"{shown} display updates in the 0.8 s after Control-Up: Mission Control "
                                   f"did not open"}
    pid = int(open(os.path.join(B, "qemu.pid")).read())
    win = Window(pid)
    a, t0 = win.start(), time.time()
    keys = []
    while time.time() - t0 < SECONDS - 0.9 or len(keys) % 2:
        keys.append(time.time())
        q.key("ctrl", "up")
        time.sleep(max(0.0, keys[-1] + 1.0 - time.time()))
    t1, b = time.time(), win.stop()
    time.sleep(1.0)
    guest("killall TextEdit", gui=True)
    q.key("meta_l", "w")
    # Each animation from its first display update after the key to its last within 0.8 s: its
    # rate, the gaps inside it, its length, and the wait from the key to its first update.
    per = [(k, composites(k, k + 0.8)) for k in keys]
    anim = [(k, ts) for k, ts in per if len(ts) >= 2]
    res = {"toggles": len(keys), "animated": len(anim), "seconds": round(t1 - t0, 2)}
    res.update(measured(a, b, t0, t1, [(t0, t1)]))
    if not anim:
        res.update(moved=0, why=f"no Control-Up of {len(keys)} gave two display updates in 0.8 s")
        return res
    frames = sum(len(ts) - 1 for _, ts in anim)
    span = sum(ts[-1] - ts[0] for _, ts in anim)
    gaps = sorted(y - x for _, ts in anim for x, y in zip(ts, ts[1:]))
    lengths = sorted(ts[-1] - ts[0] for _, ts in anim)
    starts = sorted(ts[0] - k for k, ts in anim)
    res.update(composites_s=round(frames / span, 2) if span else 0,
               gap_p99_ms=round(1000 * gaps[min(len(gaps) - 1, int(math.ceil(0.99 * len(gaps))) - 1)], 1),
               gap_max_ms=round(1000 * gaps[-1], 1), anim_ms=round(1000 * lengths[len(lengths) // 2], 1),
               start_ms=round(1000 * starts[len(starts) // 2], 1))
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
