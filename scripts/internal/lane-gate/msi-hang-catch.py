#!/usr/bin/env python3
"""The lane gate's hang catcher, on the lane host: watches a probe boot (msi-probes.sh starts it
beside the round's probes) for a guest hang and, at one, records what the vCPUs run; a guest that
panics and restarts is told apart from one that hangs.

  msi-hang-catch.py --boot-dir B --pid QEMU_PID --port PORT --out DIR
                    [--user U] [--password P] [--catches 2] [--log B/launch.log] [--from-boot]

Runs beside the probes until QEMU exits. Every 2 s it reads QEMU's CPU time (/proc/PID/stat) and
the host log's size; every 10 s it tries the guest's SSH (`true`, 5 s connect, 15 s in all). A hang
is two SSH failures in a row, or the host log silent for 15 s while QEMU burns 5 or more CPUs. At
a hang (at most --catches, 60 s apart) it writes DIR/catch-N/:

  threads.txt    every QEMU thread: tid, name, state, CPU, and its user, system and guest ticks,
                 twice 2 s apart, with the rates (a vCPU spinning in the guest has guest == user)
  kstacks.txt    each thread's kernel stack and wchan (sudo -n)
  qmp.txt        the QMP transcript: query-status, query-cpus-fast (cpu-index -> thread id),
                 5 x `info registers -a` 0.5 s apart, then per vCPU `info lapic`, `x/16i RIP`,
                 the frame-pointer chain from RBP (`x/2gx` each frame, 48 frames at most) and
                 the stack from RSP to its 16 KiB-aligned top (`x/..gx`), then `info registers
                 -a` again 10 s and 30 s later. A command not answered in 10 s ends the transcript
                 (QEMU's main loop is blocked: the BQL is held).
  ustacks.txt    QEMU's user-space stacks, every thread (sudo -n eu-stack), after the QMP part
  hostlog.txt    the host log's last 200 lines
  ram-hits.txt   counts of the panic string and the kext's log prefixes in the guest's RAM (QEMU's
                 memfd, read through /proc/PID/mem as root), and ram-text.txt: 16 KiB around each
                 panic string and the 256 KiB windows holding 20 or more of the kext's lines (the
                 kernel's message buffer), unprintable bytes as dots
  summary.txt    the trigger, and per vCPU: CPL, IF, RIP, RSP and the return addresses found

Once SSH first answers it writes DIR/guest-info.txt: kern.version, kern.osversion, kern.bootargs,
sw_vers, and `kmutil showloaded` as root (the kexts' load addresses, for the symbols).

A restart: once SSH has answered, a "the device was reset" line in the host log or a boot
banner on the guest's serial console (B/serial.log: the firmware loading the boot entry, or the
hand-off to the kernel) is a restart (watch.txt "REBOOT"), a panic's as a rule. SSH failing then
is the restart, not a hang: no catch is taken for it. Once SSH answers again it writes
DIR/reboot-N/: guest-info.txt again (the kexts load at new addresses) and the panic reports the
guest wrote since (/Library/Logs/DiagnosticReports/*.panic changed in the last 30 minutes, as
root; asked again every 10 s for 3 minutes, as the guest writes them a while after its desktop).

--from-boot (msi-boot.sh's CATCH_FROM_BOOT, started with QEMU): the boot itself is watched too. A
second hand-off to the kernel on the serial console is a restart (REBOOT, before SSH has
answered), and once the kext has started (the host's first "the device was reset") and before
the host's first present, the host log silent 10 s is a hang ("before the desktop"), caught as
above.
"""
import argparse
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time

CLK = os.sysconf("SC_CLK_TCK")
SOPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
         "-o", "PreferredAuthentications=password", "-o", "PubkeyAuthentication=no",
         "-o", "ConnectTimeout=5", "-o", "LogLevel=ERROR"]


def now():
    return time.time()


def stamp(t=None):
    t = now() if t is None else t
    return time.strftime("%H:%M:%S", time.localtime(t)) + ".%03d" % int((t % 1) * 1000)


def proc_ticks(pid):
    """(utime, stime, guest_time) of the process, in ticks, or None once it is gone."""
    try:
        with open("/proc/%d/stat" % pid) as f:
            s = f.read()
    except OSError:
        return None
    f = s[s.rindex(")") + 2:].split()
    # Fields after the name, from field 3: utime 14, stime 15, guest_time 43.
    return int(f[11]), int(f[12]), int(f[40])


def threads(pid):
    out = {}
    try:
        tids = sorted(int(t) for t in os.listdir("/proc/%d/task" % pid))
    except OSError:
        return out
    for tid in tids:
        try:
            with open("/proc/%d/task/%d/stat" % (pid, tid)) as f:
                s = f.read()
        except OSError:
            continue
        name = s[s.index("(") + 1:s.rindex(")")]
        f = s[s.rindex(")") + 2:].split()
        out[tid] = {"name": name, "state": f[0], "utime": int(f[11]), "stime": int(f[12]),
                    "guest": int(f[40]), "cpu": int(f[36])}
    return out


def ssh(args, cmd, timeout):
    try:
        r = subprocess.run(["sshpass", "-p", args.password, "ssh", "-p", str(args.port)] + SOPTS +
                           ["%s@127.0.0.1" % args.user, cmd], stdin=subprocess.DEVNULL,
                           capture_output=True, timeout=timeout, text=True)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return 255, "timed out after %d s" % timeout


class Qmp:
    """One QMP connection; each command waits at most `timeout` s for its answer."""

    def __init__(self, path, log, timeout=10):
        self.log = log
        self.timeout = timeout
        self.buf = b""
        deadline = now() + 15
        while True:
            try:
                self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.s.settimeout(timeout)
                self.s.connect(path)
                self.read()  # the greeting
                break
            except (OSError, socket.timeout) as e:
                if now() > deadline:
                    raise RuntimeError("QMP: no greeting at %s (%s)" % (path, e))
                time.sleep(1)
        self.cmd("qmp_capabilities")

    def read(self):
        while b"\n" not in self.buf:
            chunk = self.s.recv(65536)
            if not chunk:
                raise RuntimeError("QMP: connection closed")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return json.loads(line)

    def cmd(self, name, **arguments):
        msg = {"execute": name}
        if arguments:
            msg["arguments"] = arguments
        self.s.sendall((json.dumps(msg) + "\n").encode())
        t0 = now()
        while True:
            try:
                r = self.read()
            except socket.timeout:
                self.log("%s %s: no answer in %d s" % (stamp(), json.dumps(msg), self.timeout))
                raise RuntimeError("QMP: %s not answered in %d s (the BQL held?)" %
                                   (name, self.timeout))
            if "event" in r:
                continue
            ms = (now() - t0) * 1000
            if "error" in r:
                self.log("%s %s -> error %s (%.0f ms)" % (stamp(), json.dumps(msg),
                                                          json.dumps(r["error"]), ms))
                return None
            self.log("%s %s (%.0f ms)" % (stamp(), json.dumps(msg), ms))
            return r.get("return")

    def hmp(self, line, cpu=None):
        kw = {"command-line": line}
        if cpu is not None:
            kw["cpu-index"] = cpu
        r = self.cmd("human-monitor-command", **kw)
        r = r if isinstance(r, str) else ""
        self.log(r.rstrip("\n"))
        return r

    def close(self):
        try:
            self.s.close()
        except OSError:
            pass


REG = re.compile(r"\b(R[A-Z0-9]{1,2}|EIP|EFL|RFL|RIP) ?=([0-9a-f]+)")


def parse_regs(text):
    """Per CPU index: the registers of one `info registers -a`."""
    cpus = {}
    cur = None
    for line in text.splitlines():
        m = re.match(r"CPU#(\d+)", line)
        if m:
            cur = cpus.setdefault(int(m.group(1)), {})
            continue
        if cur is None:
            continue
        for k, v in REG.findall(line):
            cur[k] = int(v, 16)
        m = re.search(r"CPL=(\d)", line)
        if m:
            cur["CPL"] = int(m.group(1))
        m = re.search(r"HLT=(\d)", line)
        if m:
            cur["HLT"] = int(m.group(1))
        m = re.match(r"(GS|FS|CS|SS) =([0-9a-f]+) ([0-9a-f]+)", line)
        if m:
            cur[m.group(1) + "base"] = int(m.group(3), 16)
        m = re.search(r"\bCR3=([0-9a-f]+)", line)
        if m:
            cur["CR3"] = int(m.group(1), 16)
    return cpus


def words(text):
    """The 64-bit words of an `x/Ngx` answer, with their addresses."""
    out = []
    for line in text.splitlines():
        m = re.match(r"([0-9a-f]+): (.*)", line)
        if not m:
            continue
        addr = int(m.group(1), 16)
        for i, w in enumerate(m.group(2).split()):
            try:
                out.append((addr + 8 * i, int(w, 16)))
            except ValueError:
                pass
    return out


def kernel_addr(v):
    return 0xffffff8000000000 <= v < 0xffffffffffffffff


def catch(args, n, why):
    d = os.path.join(args.out, "catch-%d" % n)
    os.makedirs(d, exist_ok=True)
    summary = open(os.path.join(d, "summary.txt"), "w")

    def say(s):
        summary.write(s + "\n")
        summary.flush()

    say("%s catch %d: %s" % (stamp(), n, why))
    # The threads twice, 2 s apart.
    t1 = threads(args.pid)
    time.sleep(2)
    t2 = threads(args.pid)
    with open(os.path.join(d, "threads.txt"), "w") as f:
        f.write("tid name state cpu | utime stime guest (ticks) | over 2 s: user sys guest "
                "(CPUs)\n")
        for tid, b in t2.items():
            a = t1.get(tid, b)
            rate = lambda k: (b[k] - a[k]) / CLK / 2.0
            f.write("%d %s %s %d | %d %d %d | %.2f %.2f %.2f\n" % (
                tid, b["name"].replace(" ", "_"), b["state"], b["cpu"], b["utime"], b["stime"],
                b["guest"], rate("utime"), rate("stime"), rate("guest")))
    with open(os.path.join(d, "kstacks.txt"), "w") as f:
        for tid in t2:
            r = subprocess.run(["sudo", "-n", "cat", "/proc/%d/task/%d/stack" % (args.pid, tid)],
                               capture_output=True, text=True)
            try:
                wchan = open("/proc/%d/task/%d/wchan" % (args.pid, tid)).read()
            except OSError:
                wchan = "?"
            f.write("== %d %s wchan=%s\n%s%s\n" % (tid, t2[tid]["name"], wchan, r.stdout,
                                                   r.stderr))
    # QMP.
    qf = open(os.path.join(d, "qmp.txt"), "w")

    def qlog(s):
        qf.write(s + "\n")
        qf.flush()

    regs = {}
    try:
        q = Qmp(os.path.join(args.boot_dir, "qmp.sock"), qlog)
        q.cmd("query-status")
        cpus = q.cmd("query-cpus-fast") or []
        qlog(json.dumps(cpus))
        for k in range(5):
            text = q.hmp("info registers -a")
            if k == 0:
                regs = parse_regs(text)
            time.sleep(0.5)
        for i in sorted(regs):
            r = regs[i]
            rip, rsp, rbp = r.get("RIP", r.get("EIP", 0)), r.get("RSP", 0), r.get("RBP", 0)
            rfl = r.get("RFL", r.get("EFL", 0))
            qlog("== CPU %d: CPL %s IF %d HLT %s RIP %016x RSP %016x RBP %016x" % (
                i, r.get("CPL"), (rfl >> 9) & 1, r.get("HLT"), rip, rsp, rbp))
            q.hmp("info lapic", cpu=i)
            if rip:
                q.hmp("x/16i 0x%x" % rip, cpu=i)
            # The frame-pointer chain.
            chain = []
            fp = rbp
            for _ in range(48):
                if not kernel_addr(fp) or fp % 8:
                    break
                w = words(q.hmp("x/2gx 0x%x" % fp, cpu=i))
                if len(w) < 2:
                    break
                chain.append((fp, w[1][1]))
                nxt = w[0][1]
                if nxt <= fp or nxt - fp > 0x100000:
                    break
                fp = nxt
            # The stack from RSP to the top of its 16 KiB (kernel stacks are 16 KiB, aligned).
            found = []
            if kernel_addr(rsp):
                top = (rsp | 0x3fff) + 1
                count = max(1, min((top - rsp) // 8, 2048))
                for a, v in words(q.hmp("x/%dgx 0x%x" % (count, rsp), cpu=i)):
                    if kernel_addr(v):
                        found.append((a, v))
            say("CPU %d: CPL %s IF %d HLT %s RIP %016x RSP %016x RBP %016x GSbase %016x" % (
                i, r.get("CPL"), (rfl >> 9) & 1, r.get("HLT"), rip, rsp, rbp,
                r.get("GSbase", 0)))
            say("  frames (fp -> return): " + " ".join("%x" % ret for _, ret in chain))
            say("  kernel words on the stack: %d (see qmp.txt)" % len(found))
        for wait in (10, 20):
            time.sleep(wait)
            again = parse_regs(q.hmp("info registers -a"))
            say("%s after %d s more: " % (stamp(), wait) + " ".join(
                "cpu%d=%016x" % (i, again[i].get("RIP", again[i].get("EIP", 0))) for i in sorted(again)))
        q.close()
    except Exception as e:  # noqa: BLE001 - a catch writes what it got
        say("QMP: %s" % e)
    qf.close()
    r = subprocess.run(["sudo", "-n", "timeout", "60", "eu-stack", "-p", str(args.pid)],
                       capture_output=True, text=True)
    with open(os.path.join(d, "ustacks.txt"), "w") as f:
        f.write(r.stdout + r.stderr)
    # The guest's RAM, for the panic string and the kernel's message buffer.
    r = subprocess.run(["sudo", "-n", "timeout", "120", sys.executable, "-I",
                        os.path.abspath(__file__), "--scan-ram", str(args.pid), d],
                       capture_output=True, text=True)
    say("RAM scan: rc %d %s" % (r.returncode, (r.stdout + r.stderr).strip()[-300:]))
    try:
        lines = open(args.log, errors="replace").read().splitlines()[-200:]
        with open(os.path.join(d, "hostlog.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
    except OSError:
        pass
    say("%s catch %d done" % (stamp(), n))
    summary.close()


PATTERNS = [b"panic(cpu", b"RgfxAccelerator: ", b"ReimsGraphics: ", b"RGFX"]
PRINTABLE = bytes(c if c in (9, 10) or 32 <= c < 127 else 46 for c in range(256))


def scan_ram(pid, d):
    """As root: reads QEMU's guest RAM (its memfd mappings, through /proc/PID/mem) and writes
    ram-hits.txt (each pattern's count) and ram-text.txt: 16 KiB around each panic string, and
    the 256 KiB windows holding 20 or more of the kext's lines (the kernel's message buffer)."""
    maps = []
    for line in open("/proc/%d/maps" % pid):
        f = line.split()
        if len(f) >= 6 and "memfd:ram" in f[5]:
            a, b = (int(x, 16) for x in f[0].split("-"))
            maps.append((a, b))
    hits = {p: [] for p in PATTERNS}
    mem = os.open("/proc/%d/mem" % pid, os.O_RDONLY)
    chunk = 64 << 20
    overlap = 64
    for a, b in maps:
        at = a
        while at < b:
            n = min(chunk + overlap, b - at)
            try:
                buf = os.pread(mem, n, at)
            except OSError:
                at += chunk
                continue
            for p in PATTERNS:
                i = buf.find(p)
                while i >= 0:
                    if i < chunk:
                        hits[p].append(at + i)
                    i = buf.find(p, i + 1)
            at += chunk
    with open(os.path.join(d, "ram-hits.txt"), "w") as f:
        for p in PATTERNS:
            f.write("%s %d\n" % (p.decode(), len(hits[p])))
    windows = []
    for h in hits[b"panic(cpu"][:16]:
        windows.append((max(h - 0x1000, 0), h + 0x3000))
    kext = sorted(hits[b"RgfxAccelerator: "] + hits[b"ReimsGraphics: "])
    j = 0
    for i in range(len(kext)):
        while kext[i] - kext[j] > 0x40000:
            j += 1
        if i - j + 1 >= 20:
            windows.append((kext[j] - 0x1000, kext[i] + 0x1000))
    windows.sort()
    merged = []
    for a, b in windows:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    total = 0
    with open(os.path.join(d, "ram-text.txt"), "wb") as f:
        for a, b in merged:
            if total > (8 << 20):
                break
            try:
                buf = os.pread(mem, b - a, a)
            except OSError:
                continue
            total += len(buf)
            f.write(b"\n==== host address %x, %d bytes ====\n" % (a, len(buf)))
            f.write(buf.translate(PRINTABLE))
    print("%s; %d windows, %d bytes" % (", ".join("%s %d" % (p.decode(), len(hits[p]))
                                                  for p in PATTERNS), len(merged), total))


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--scan-ram":
        return scan_ram(int(sys.argv[2]), sys.argv[3])
    p = argparse.ArgumentParser()
    p.add_argument("--boot-dir", required=True)
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--user", default="aneesiqbal")
    p.add_argument("--password", default="aneesiqbal")
    p.add_argument("--catches", type=int, default=2)
    p.add_argument("--log")
    p.add_argument("--from-boot", action="store_true")
    args = p.parse_args()
    args.log = args.log or os.path.join(args.boot_dir, "launch.log")
    os.makedirs(args.out, exist_ok=True)
    events = open(os.path.join(args.out, "watch.txt"), "a")

    def ev(s):
        events.write("%s %s\n" % (stamp(), s))
        events.flush()

    ev("watching QEMU %d, port %d, log %s" % (args.pid, args.port, args.log))
    state = {"ssh_fail": 0, "ssh_ok": False, "info": False, "last_ssh": 0.0, "busy": False,
             "reboots": 0, "reboot_at": 0.0, "fetch": False}
    info_cmd = ("sysctl -n kern.version kern.osversion kern.bootargs; sw_vers; "
                "echo '%s' | sudo -S -p '' kmutil showloaded 2>&1" % args.password)

    def after_reboot(n):
        """Once SSH answers after restart N: the guest's info and its new panic reports."""
        d = os.path.join(args.out, "reboot-%d" % n)
        os.makedirs(d, exist_ok=True)
        rc, info = ssh(args, info_cmd, 120)
        with open(os.path.join(d, "guest-info.txt"), "w") as f:
            f.write(info)
        got = []
        for _ in range(18):
            rc, names = ssh(args, "find /Library/Logs/DiagnosticReports -maxdepth 1 -name '*.panic' "
                            "-mmin -30 2> /dev/null", 20)
            got = [x for x in names.split() if x.endswith(".panic")]
            if got:
                break
            time.sleep(10)
        for path in got:
            rc, text = ssh(args, "echo '%s' | sudo -S -p '' cat '%s'" % (args.password, path), 60)
            with open(os.path.join(d, os.path.basename(path)), "w") as f:
                f.write(text)
        ev("reboot %d: %d panic report(s) into reboot-%d%s" % (
            n, len(got), n, ": " + " ".join(os.path.basename(x) for x in got) if got else ""))

    def ssh_check():
        rc, text = ssh(args, "true", 15)
        if rc == 0:
            if state["ssh_fail"]:
                ev("SSH answers again, after %d failures in a row" % state["ssh_fail"])
            state["ssh_fail"] = 0
            if not state["ssh_ok"]:
                ev("SSH answers")
            state["ssh_ok"] = True
            if not state["info"]:
                state["info"] = True
                rc2, info = ssh(args, info_cmd, 120)
                with open(os.path.join(args.out, "guest-info.txt"), "w") as f:
                    f.write(info)
                ev("guest-info.txt written (rc %d)" % rc2)
            if state["fetch"]:
                state["fetch"] = False
                after_reboot(state["reboots"])
        elif state["ssh_ok"]:
            state["ssh_fail"] += 1
            ev("SSH failed (%d in a row): %s" % (state["ssh_fail"], text.strip()[-160:]))
        state["busy"] = False

    hist = []
    log_size, log_at = -1, now()
    # The host log's and the serial console's new lines, read from where SSH first answered.
    serial = os.path.join(args.boot_dir, "serial.log")
    offsets = {}

    def new_text(path):
        try:
            size = os.path.getsize(path)
        except OSError:
            return b""
        if path not in offsets or size < offsets[path]:
            offsets[path] = size
            return b""
        with open(path, "rb") as f:
            f.seek(offsets[path])
            data = f.read(size - offsets[path])
        cut = data.rfind(b"\n") + 1
        offsets[path] += cut
        return data[:cut]
    caught, last_catch = 0, 0.0
    # --from-boot: the boot's own markers, read from the start.
    boot = {"off": 0, "kext": False, "present": False, "handoffs": 0}

    def boot_scan():
        try:
            with open(args.log, "rb") as f:
                f.seek(boot["off"])
                data = f.read()
        except OSError:
            data = b""
        cut = data.rfind(b"\n") + 1
        boot["off"] += cut
        data = data[:cut]
        if b"the device was reset" in data:
            boot["kext"] = True
        if b"PRESENT_SURFACE of transaction" in data or b"the display shows IOSurface" in data:
            if not boot["present"]:
                ev("the host's first present")
            boot["present"] = True
        try:
            n = open(serial, "rb").read().count(b"HANDOFF TO XNU")
        except OSError:
            n = 0
        if n > boot["handoffs"]:
            if boot["handoffs"] >= 1:
                return "hand-off to the kernel %d on the serial console" % n
            boot["handoffs"] = n
        return None
    while True:
        t = now()
        ticks = proc_ticks(args.pid)
        if ticks is None:
            ev("QEMU exited")
            break
        hist.append((t, ticks[0] + ticks[1]))
        hist = [h for h in hist if t - h[0] <= 10.5]
        rate = (hist[-1][1] - hist[0][1]) / CLK / max(hist[-1][0] - hist[0][0], 1e-3)
        try:
            size = os.path.getsize(args.log)
        except OSError:
            size = log_size
        if size != log_size:
            log_size, log_at = size, t
        if args.from_boot:
            again = boot_scan()
            if again:
                boot["handoffs"] += 1
                boot["kext"], boot["present"] = False, False
                state["reboots"] += 1
                state["reboot_at"] = t
                state["fetch"] = True
                ev("REBOOT %d: %s, before the desktop" % (state["reboots"], again))
        if state["ssh_ok"]:
            m = re.search(rb"^(\S+) .*the device was reset", new_text(args.log), re.M)
            b = re.search(rb"BdsDxe: loading|HANDOFF TO XNU", new_text(serial))
            if (m or b) and t - state["reboot_at"] > 120:  # one restart resets and banners again
                state["reboots"] += 1
                state["reboot_at"] = t
                state["fetch"] = True
                ev("REBOOT %d: %s" % (state["reboots"], "the host's device reset at %s" % m.group(1).decode(
                    errors="replace") if m else "a boot banner on the serial console"))
        if not state["busy"] and t - state["last_ssh"] >= 10:
            state["busy"] = True
            state["last_ssh"] = t
            threading.Thread(target=ssh_check, daemon=True).start()
        why = None
        if state["reboots"] and t - state["reboot_at"] < 300:
            if state["ssh_fail"] >= 2 and not state.get("said"):
                state["said"] = True
                ev("SSH down %d times in a row: the guest is restarting (REBOOT %d), no catch" % (
                    state["ssh_fail"], state["reboots"]))
        elif state["ssh_fail"] >= 2:
            why = "SSH failed %d times in a row" % state["ssh_fail"]
        elif args.from_boot and boot["kext"] and not boot["present"] and t - log_at >= 10:
            why = "before the desktop: the host log silent %.0f s after the kext started, QEMU at %.1f CPUs" % (
                t - log_at, rate)
        elif t - log_at >= 15 and rate >= 5.0 and len(hist) >= 5:
            why = "host log silent %.0f s, QEMU at %.1f CPUs" % (t - log_at, rate)
        if why and caught < args.catches and t - last_catch >= 60:
            caught += 1
            last_catch = t
            ev("HANG: %s; catch %d" % (why, caught))
            catch(args, caught, why)
            ev("catch %d written" % caught)
            state["ssh_fail"] = 0
        if state["ssh_fail"] == 0:
            state["said"] = False
        time.sleep(2)


if __name__ == "__main__":
    sys.exit(main())
