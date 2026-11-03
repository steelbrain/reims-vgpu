#!/usr/bin/env python3
"""The lane gate's report (scripts/internal/lane-gate.sh): reads one run's directory, compares it
with a baseline, and writes results.json and report.txt into the run.

    report.py RUN [--baseline DIR] [--noise FILE] [--history FILE]
    report.py --record RUN BASELINES          adds RUN as a sample of its commit's baseline
    report.py --noise-from BASELINES FILE     the noise band from every baseline with 2+ samples

A run's directory holds:
  steps/NAME.ok | NAME.fail     each step's result (the first line: its seconds), NAME.log
  results/guest.log             guest-run.sh's lines, each after the host's epoch time
  results/host.txt, cpu.txt, host-events.log   msi-boot.sh's (its header says what each holds)
  results/cargo-fails.txt       the cargo tests that failed, "BINARY TEST" a line
  meta.json                     the commit, the run's id, the tools' source hashes, what ran

An item is PASS, FAIL or SKIP, or KNOWN: a failure that an entry of the known-failures list
(meta.json's known_fails, from the gate's KNOWN_FAILS file) matches exactly, which decides nothing.
An entry names the item, a regular expression its failure's detail must match (pattern), and
optionally a relation between two of the pattern's named groups ("m = n - 1"), its owner and its
ledger row; an entry missing one of those is ignored. --history keeps, across runs, each entry's
status in each run, and the report counts its hits among the runs that ran the check. Against
the baseline, an item the baseline passed that is now
SKIP or missing fails. A metric is a number; the gated ones (GATED) fail where they are worse
than the baseline's median by more than the noise band: the larger of the metric's floor and
NOISE_K times the largest relative distance from the median that repeated runs of one commit
showed. Exit 0 when every item passes and no gated metric regressed, 1 when one failed, 2 when a
part did not run (an incomplete run passes nothing).
"""
import datetime
import json
import os
import re
import statistics
import sys

# Gated metrics: (regex, +1 higher is better / -1 lower is better, floor of the band).
GATED = [
    (r"bench\.frag\.alu\.(1080p|2160p)\.fps", 1, 0.05),
    (r"bench\.frag\.fill\.2160p\.gpix_s", 1, 0.05),
    (r"bench\.compute\.fma\.gflops_wall", 1, 0.05),
    (r"bench\.blit\.256m\.gbps_wall", 1, 0.05),
    (r"bench\.sample\.tex\.gsamples_s_wall", 1, 0.05),
    (r"bench\.draws\.10k\.draws_s", 1, 0.05),
    (r"bench\.(upload|readback)\.64m\.mbps", 1, 0.05),
    (r"bench\.mps\.sgemm\.(2048|4096)\.gflops_wall", 1, 0.05),
    (r"bench\.onscreen\.presented_fps", 1, 0.03),
    (r"present\.(fullscreen|windowed)\.presented_fps", 1, 0.03),
    (r"present\.(fullscreen|windowed)\.latency_ms_p50", -1, 0.10),
    (r"decode\.[a-z0-9]+\.[A-Za-z0-9._-]+\.fps", 1, 0.08),
    (r"webgl\.raf_fps", 1, 0.03),
    (r"composites_s\.present\.fullscreen", 1, 0.05),
    (r"qemu_cpu\.(idle|present\.fullscreen)", -1, 0.25),
    (r"drag\.(plain|glass)\.composites_s", 1, 0.10),
]
NOISE_K = 2.0
# The Mac's check steps, which --skip mac leaves out.
MAC_STEPS = {"kext-check", "kext-test", "bundle-test", "bundle-files", "rgfxgate-test", "fmt", "cargo"}
# The informational metrics a report shows beside the gated ones.
SHOWN = [r"present\..*\.(dropped|wait_ms_p50|gap_ms_max)", r"qemu_cpu\..*", r"composites_s\..*",
         r"(shared\.)?(drag\..*|gfx_cpu\.drag\..*|qemu_cpu\.drag\..*)",
         r"host\.(boot_ssh_s|desktop_s|hold_s|lock_wait_s)", r"bench\.draws\.10k\.encode_ms",
         r"shared\.(host\..*|webgl\.raf_fps|present\..*\.presented_fps|composites_s\.present\.fullscreen)"]


def gated(name):
    for rx, sign, floor in GATED:
        if re.fullmatch(rx, name):
            return sign, floor
    return None


def read(path, default=""):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return default


def kv_numbers(text):
    out = {}
    for k, v in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=(-?[0-9.]+(?:e-?[0-9]+)?)(?=\s|$)", text):
        try:
            out[k] = float(v)
        except ValueError:
            pass
    return out


def local_epoch(stamp):
    """2026-11-03T04:24:14.596+0200 -> epoch seconds."""
    m = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)\.(\d{3})([+-])(\d\d)(\d\d)", stamp)
    if not m:
        return None
    t = datetime.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")
    off = (int(m.group(4)) * 60 + int(m.group(5))) * 60 * (1 if m.group(3) == "+" else -1)
    return (t - datetime.datetime(1970, 1, 1)).total_seconds() - off + int(m.group(2)) / 1000


class Run:
    def __init__(self, path):
        self.path = path
        self.items = {}
        self.metrics = {}
        self.meta = json.loads(read(os.path.join(path, "meta.json"), "{}"))

    def item(self, name, status, detail=""):
        self.items[name] = {"status": status, "detail": detail.strip()[:300]}

    # A boot's items and metrics carry its leg's prefixes: the correctness boot's items none and its
    # metrics "shared." (numbers taken beside other VMs, shown and never gated), the bench boot's
    # items "perf." and its metrics none.
    ip, mp = "", ""

    def litem(self, name, status, detail=""):
        self.item(self.ip + name, status, detail)

    def lmetric(self, name, value):
        self.metrics[self.mp + name] = value

    def parse(self):
        self.steps()
        r = os.path.join(self.path, "results")
        # results/boot is the correctness boot's, results/bench the bench boot's; a run from before
        # the split has one exclusive boot's files in results/ itself.
        legs = [(os.path.join(r, "boot"), "", "shared."), (os.path.join(r, "bench"), "perf.", "")]
        if os.path.exists(os.path.join(r, "guest.log")):
            legs = [(r, "", "")]
        for d, self.ip, self.mp in legs:
            if os.path.exists(os.path.join(d, "guest.log")):
                self.guest(read(os.path.join(d, "guest.log")))
                self.host(d)
                self.drag(d)
                self.video(d)
        self.ip, self.mp = "", ""
        self.cargo(r)
        return self

    def steps(self):
        d = os.path.join(self.path, "steps")
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            name, ext = os.path.splitext(f)
            if ext not in (".ok", ".fail"):
                continue
            body = read(os.path.join(d, f)).splitlines()
            secs = body[0] if body else "?"
            self.metrics["step_s." + name] = float(secs) if re.fullmatch(r"[0-9.]+", secs) else 0
            if ext == ".ok":
                self.item("step." + name, "PASS", f"{secs} s")
            else:
                tail = read(os.path.join(d, name + ".log")).strip().splitlines()[-4:]
                self.item("step." + name, "FAIL", f"{secs} s: " + " | ".join(tail))

    def guest(self, text):
        self.phases = {}
        self.video_offer, self.metal_device = None, None
        crashes = None
        for line in text.splitlines():
            m = re.match(r"([0-9.]+) (.*)$", line)
            if not m:
                continue
            t, body = float(m.group(1)), m.group(2)
            if body.startswith("GATE phase "):
                _, _, name, edge, *_ = body.split()
                self.phases.setdefault(name, [None, None])[0 if edge == "start" else 1] = t
            elif body.startswith("GATE item "):
                parts = body.split(" ", 4)
                self.litem("guest." + parts[2], parts[3], parts[4] if len(parts) > 4 else "")
                if parts[2] == "metal-device" and len(parts) > 4:
                    m2 = re.search(r'device "([^"]+)"', parts[4])
                    self.metal_device = m2.group(1) if m2 else None
            elif body.startswith("GATE info video-offer "):
                m2 = re.match(r"GATE info video-offer hevc=(\d)", body)
                self.video_offer = {"hevc": m2 is not None and m2.group(1) == "1"}
            elif body.startswith("GATE crashes "):
                crashes = body[len("GATE crashes "):].strip()
            elif body.startswith("GATE webgl "):
                try:
                    w = json.loads(body[len("GATE webgl "):])
                except ValueError:
                    self.litem("webgl.safari", "FAIL", "unreadable result: " + body[:200])
                    continue
                bad = [k for k, c in w.get("checks", {}).items() if not c.get("ok")]
                ok = w.get("ok") and not bad and len(w.get("checks", {})) == 5
                self.litem("webgl.safari", "PASS" if ok else "FAIL",
                          f"{w.get('version')}, {w.get('renderer')}; "
                          + (f"failed {bad}" if bad else "5 readPixels checks exact") + (f"; {w['error']}" if "error" in w else ""))
                if "raf_fps" in w:
                    self.lmetric("webgl.raf_fps", round(float(w["raf_fps"]), 2))
            elif body.startswith("GATE item webgl "):
                pass
            elif body.startswith("sweep: "):
                m2 = re.match(r"sweep: (\S+)\s+(OK|FAIL|SKIP)\s+(.*)$", body)
                if m2:
                    self.litem("sweep." + m2.group(1), m2.group(2).replace("OK", "PASS"), m2.group(3))
                elif re.match(r"sweep: \d+ OK", body):
                    self.lmetric("sweep.cases", sum(int(x) for x in re.findall(r"(\d+) [A-Z]+", body)))
            elif body.startswith("bench: "):
                m2 = re.match(r"bench: (\S+) (OK|FAIL|SKIP|NOTRUN)\s*(.*)$", body)
                if m2:
                    self.litem("bench." + m2.group(1), {"OK": "PASS"}.get(m2.group(2), m2.group(2)), m2.group(3)[:200])
                    if m2.group(2) == "OK":
                        for k, v in kv_numbers(m2.group(3)).items():
                            self.lmetric(f"bench.{m2.group(1)}.{k}", v)
            elif body.startswith("present_probe: "):
                m2 = re.match(r"present_probe: (\S+) (OK|FAIL)\s*(.*)$", body)
                if m2:
                    kv = kv_numbers(m2.group(3))
                    st = "PASS" if m2.group(2) == "OK" else "FAIL"
                    # Presentation answers: frames presented, none (the bench boot, the GPU alone) or at
                    # most a tenth (the correctness boot, beside other VMs) dropped or left unanswered.
                    pres, lost = kv.get("presented", 0), kv.get("dropped", 0) + max(0, kv.get("unanswered", 0) - 3)
                    if st == "PASS" and (pres == 0 or (lost > 0 if self.ip else lost > 0.1 * (pres + lost))):
                        st = "FAIL"
                    self.litem("present." + m2.group(1), st, m2.group(3)[:200])
                    for k, v in kv.items():
                        self.lmetric(f"present.{m2.group(1)}.{k}", v)
            elif body.startswith("decode_md5: "):
                m2 = re.match(r"decode_md5: (\S+) (\S+) (OK|FAIL|SKIP)\s*(.*)$", body)
                if m2:
                    name = f"decode.{m2.group(1)}.{m2.group(2)}"
                    self.litem(name, m2.group(3).replace("OK", "PASS"), m2.group(4))
                    kv = kv_numbers(m2.group(4))
                    if "fps" in kv:
                        self.lmetric(name + ".fps", kv["fps"])
        if crashes is not None:
            names = sorted({re.sub(r"[-_]\d{4}-\d\d-\d\d.*$", "", c) for c in crashes.split() if c != "none"})
            self.crashes = names
            ws = [n for n in names if "WindowServer" in n]
            self.litem("guest.crash-reports", "FAIL" if ws else "PASS", ", ".join(names) or "none new")
        for p, (a, b) in self.phases.items():
            if a is None or b is None:
                self.litem("phase." + p, "FAIL", "started but did not end (a hang the alarm or the hold cut)")

    def video(self, d):
        """Each decode SKIP (decode_md5's: VideoToolbox finds no hardware decoder for the stream's
        codec) as what the gate measured: SKIP where the host's GPU has no hardware decoder for the
        codec (host-video.tsv, Vulkan Video's decode profiles) or the candidate's device does not
        offer the codec (meta.json's tip_codecs; for HEVC also the guest's accelerator, which
        carries IOGVAHEVCDecode where it offers it); FAIL where both hold and VideoToolbox still
        finds none."""
        labels = {"h264": "H.264", "hevc": "HEVC"}
        host = self.meta.get("lane_host") or "the lane host"
        tip = self.meta.get("tip_codecs")
        rows = [l.split("\t") for l in read(os.path.join(d, "host-video.tsv")).splitlines()]
        gpu = self.metal_device
        names = {r[0] for r in rows if len(r) == 3}
        decodes = {}
        for r in rows:
            if len(r) == 3 and r[0] == gpu and r[1] != "gpu":
                decodes.setdefault(r[1], []).append(r[2])
        offer = self.video_offer
        for name, it in self.items.items():
            if not name.startswith(self.ip + "decode.") or it["status"] != "SKIP":
                continue
            codec = name[len(self.ip + "decode."):].split(".")[0]
            label = labels.get(codec, codec)
            vt = f"VideoToolbox finds no hardware {label} decoder"
            accel = ""
            if codec == "hevc" and offer is not None:
                accel = "the accelerator carries " + ("IOGVAHEVCDecode" if offer["hevc"] else "no IOGVAHEVCDecode") + ", "
            if gpu in names and codec not in decodes:
                it["detail"] = f"{host}'s {gpu} has no hardware {label} decoder (Vulkan Video lists no {label} decode profile)"
            elif tip is not None and codec not in tip:
                it["detail"] = f"the device offers no {label} decoder on this tip ({accel}{vt})"
            elif tip is None:
                it["detail"] = f"the device offers no {label} decoder ({accel}{vt})"
            else:
                it["status"] = "FAIL"
                where = (f"{host}'s {gpu} decodes {label} ({', '.join(sorted(decodes[codec]))})" if codec in decodes
                         else f"{host}'s hardware decoders were not measured")
                it["detail"] = f"{vt}, where this tip offers {label} and {where}; {accel.rstrip(', ') or 'no accelerator answer'}"

    def host(self, r):
        kv = {}
        for line in read(os.path.join(r, "host.txt")).splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2:
                kv[parts[0]] = parts[1].strip()
        # The boot's waits (turn, host quiet, GPU lock), its timeline from QEMU's first line (the
        # host's first present, the guest's kernel, WindowServer, loginwindow, Dock, SSH, the
        # desktop), and the host's state beside the measurements.
        for k in ("turn_wait_s", "quiet_wait_s", "gpu_wait_s", "lock_wait_s", "first_present_s", "kernel_s",
                  "windowserver_s", "loginwindow_s", "dock_s", "boot_ssh_s", "desktop_s", "hold_s",
                  "load_start", "load_end", "load_duty", "host_quiet"):
            if re.fullmatch(r"[0-9]+(\.[0-9]*)?", kv.get(k, "")):
                self.lmetric("host." + k, float(kv[k]))
        fatal = int(kv.get("fatal_lines", "-1") or -1) + max(0, int(kv.get("serial_fatal_lines", "0") or 0))
        self.litem("host.no-fatal", "PASS" if fatal == 0 else "FAIL",
                  f"{fatal} 'reims-vgpu: fatal' lines" if fatal >= 0 else "no host log")
        if "first_boot_accel" in kv:
            self.litem("host.same-boot-accel", "PASS" if kv["first_boot_accel"] == "1" else "FAIL",
                      "our Metal device on the first boot" if kv["first_boot_accel"] == "1"
                      else "no Metal device of ours until a reboot")
        # The guest's screenshot of its screen at the end (msi-boot.sh).
        if kv.get("desktop_capture") == "none":
            self.litem("guest.desktop-capture", "FAIL", "the guest saved no screenshot for Command-Shift-3")
        elif "desktop_colours" in kv:
            colours = int(kv["desktop_colours"] or 0)
            top = float(kv.get("desktop_top_share", "1") or 1)
            self.litem("guest.desktop-capture", "PASS" if colours >= 64 and top <= 0.5 else "FAIL",
                       f"the guest's capture of its screen: {colours} colours, the commonest on {top:.1%} of it")
        # Display updates and QEMU's CPU in each phase.
        updates = [local_epoch(l[:28]) for l in read(os.path.join(r, "host-events.log")).splitlines()
                   if "the display shows IOSurface" in l]
        updates = [u for u in updates if u is not None]
        cpu = []
        for line in read(os.path.join(r, "cpu.txt")).splitlines():
            p = line.split()
            if len(p) == 4:
                cpu.append((float(p[0]), (int(p[1]) + int(p[2])) / int(p[3])))
        for name, (a, b) in getattr(self, "phases", {}).items():
            if a is None or b is None or b - a < 1:
                continue
            n = sum(1 for u in updates if a <= u <= b)
            self.lmetric("composites_s." + name, round(n / (b - a), 2))
            inside = [c for c in cpu if a <= c[0] <= b]
            if len(inside) >= 2:
                self.lmetric("qemu_cpu." + name, round(
                    100 * (inside[-1][1] - inside[0][1]) / (inside[-1][0] - inside[0][0]), 1))
        if getattr(self, "phases", {}).get("present.fullscreen"):
            n = self.metrics.get(self.mp + "composites_s.present.fullscreen", 0)
            self.litem("host.composites", "PASS" if n > 0 else "FAIL",
                      f"{n} display updates a second during the full-screen present")

    # The window drags (msi-drag.py): a case whose window did not follow the pointer is not run, not
    # a number; a gap of more than a second between display updates while it moved is a stall.
    def drag(self, r):
        for line in read(os.path.join(r, "drag.txt")).splitlines():
            m = re.match(r"drag (\S+) (.*)$", line)
            if not m:
                continue
            name, kv = m.group(1), kv_numbers(m.group(2))
            why = re.search(r'why=(".*")', line)
            if not kv.get("moved"):
                self.litem("drag." + name, "NOTRUN", "the window did not follow the pointer: "
                           + (json.loads(why.group(1)) if why else m.group(2)))
                continue
            stall = kv.get("gap_max_ms", 0) > 1000
            self.litem("drag." + name, "FAIL" if stall else "PASS",
                       f"{kv.get('composites_s')} display updates a second, gap p99 {kv.get('gap_p99_ms')} ms, "
                       f"max {kv.get('gap_max_ms')} ms" + (" (a stall over 1 s)" if stall else "")
                       + f"; the window moved {kv.get('window_moves_s')} times a second; QEMU {kv.get('qemu_cpu')}%, "
                       f"its graphics thread {kv.get('gfx_cpu')}%")
            for k in ("composites_s", "gap_p99_ms", "gap_max_ms", "window_moves_s"):
                if k in kv:
                    self.lmetric(f"drag.{name}.{k}", kv[k])
            for k in ("qemu_cpu", "gfx_cpu"):
                if k in kv:
                    self.lmetric(f"{k}.drag.{name}", kv[k])

    def cargo(self, r):
        p = os.path.join(r, "cargo-fails.txt")
        if not os.path.exists(p):
            return
        self.cargo_fails = sorted({l.strip() for l in read(p).splitlines() if l.strip()})
        self.metrics["cargo.failed"] = len(self.cargo_fails)
        summary = read(os.path.join(r, "cargo-summary.txt")).strip()
        # The tests that failed under the check's load and passed run again alone.
        alone = sorted({l.strip() for l in read(os.path.join(r, "cargo-alone.txt")).splitlines() if l.strip()})
        if alone:
            summary += f"; {len(alone)} passed run again alone (load-sensitive): " + ", ".join(alone)
        self.item("cargo.tests", "PASS", summary)

    def to_json(self):
        return {"meta": self.meta, "items": self.items, "metrics": self.metrics,
                "cargo_fails": getattr(self, "cargo_fails", None), "crashes": getattr(self, "crashes", None)}


def median_baseline(bdir):
    """A baseline's reference: its samples' items (the first sample's) and each metric's median."""
    runs = []
    rd = os.path.join(bdir, "runs")
    for f in sorted(os.listdir(rd)) if os.path.isdir(rd) else []:
        if f.endswith(".json"):
            runs.append(json.loads(read(os.path.join(rd, f))))
    if not runs:
        return None
    ref = {"meta": runs[0]["meta"], "items": runs[0]["items"], "samples": len(runs),
           "cargo_fails": runs[0].get("cargo_fails"), "crashes": runs[0].get("crashes"), "metrics": {}}
    names = set().union(*(r["metrics"] for r in runs))
    for n in names:
        vals = [r["metrics"][n] for r in runs if n in r["metrics"]]
        ref["metrics"][n] = statistics.median(vals)
    return ref


def bench_host(metrics):
    """The lane host's state beside the bench boot: whether it was quiet, and its load average."""
    if not any(k.startswith("host.") for k in metrics):
        return ""
    q = metrics.get("host.host_quiet")
    quiet = "HOST_QUIET" if q == 1 else "HOST_NOT_QUIET" if q == 0 else "host quiet not stated"
    load = (f", load average {metrics['host.load_start']:g} to {metrics['host.load_end']:g}"
            if "host.load_start" in metrics and "host.load_end" in metrics else "")
    return f"; bench host: {quiet}{load}"


def noise_from(bdirs_root):
    """Per gated metric: the largest relative distance from its commit's median seen in repeated runs,
    from the bench samples taken in host quiet only (host.host_quiet 1: no build on the lane host beside
    QEMU), as other work's CPU and memory use moves the Vulkan platform's numbers far past noise."""
    dev = {}
    for sha in sorted(os.listdir(bdirs_root)):
        rd = os.path.join(bdirs_root, sha, "runs")
        if sha == "LAST" or not os.path.isdir(rd):
            continue
        runs = [json.loads(read(os.path.join(rd, f))) for f in sorted(os.listdir(rd)) if f.endswith(".json")]
        runs = [r for r in runs if r["metrics"].get("host.host_quiet") == 1]
        if len(runs) < 2:
            continue
        for n in set().union(*(r["metrics"] for r in runs)):
            vals = [r["metrics"][n] for r in runs if n in r["metrics"]]
            if len(vals) < 2:
                continue
            med = statistics.median(vals)
            if med:
                d = max(abs(v - med) / abs(med) for v in vals)
                dev[n] = max(dev.get(n, 0), d)
    return dev


# The tool whose numbers a metric is: where its source differs from the baseline's, the metric is
# shown but not gated, as the measuring stick moved.
TOOL_OF = [(r"bench\..*", "sweep+bench"), (r"present\..*", "present_probe"), (r"decode\..*", "decode_md5"),
           (r"(drag\.|[a-z]+_cpu\.drag\.).*", "drag")]


def tool_changed(name, meta, base_meta):
    for rx, tool in TOOL_OF:
        if re.fullmatch(rx, name):
            a, b = meta.get("tools", {}).get(tool), base_meta.get("tools", {}).get(tool)
            return a is not None and b is not None and a != b
    return False


def band(name, noise):
    g = gated(name)
    floor = g[1] if g else 0.05
    return max(floor, NOISE_K * noise.get(name, 0))


def compare(run, base, noise):
    lines, failed, incomplete = [], [], False
    cur = run.to_json()
    items = dict(cur["items"])
    # Cargo: a failure no baseline knew of fails.
    if run.__dict__.get("cargo_fails") is not None:
        known = set(base.get("cargo_fails") or []) if base else set()
        known |= set(run.meta.get("known_cargo_fails", []))
        strip = lambda t: t.split(" ", 1)[-1]
        new = [t for t in run.cargo_fails if t not in known and strip(t) not in {strip(k) for k in known}]
        fixed = sorted(set(base.get("cargo_fails") or []) - set(run.cargo_fails)) if base else []
        if new:
            items["cargo.tests"] = {"status": "FAIL", "detail": f"{len(new)} failures not in the known set: " + ", ".join(new[:8])}
        elif run.cargo_fails:
            items["cargo.tests"]["detail"] += f" ({len(run.cargo_fails)} known failures)"
        if fixed:
            lines.append(f"  note: {len(fixed)} known cargo failures now pass: " + ", ".join(fixed[:5]))
    ran = set(run.meta.get("ran", []))
    for name, b in sorted((base or {}).get("items", {}).items()):
        # Publishing is the tips' alone (--publish): a run without it misses nothing.
        if name == "step.publish" and name not in items:
            continue
        if name in items:
            if b["status"] == "PASS" and items[name]["status"] == "SKIP":
                items[name] = {"status": "FAIL", "detail": "SKIP, where the baseline passed: " + items[name]["detail"]}
            # A drag the baseline made that does not move the window now may be the candidate's doing.
            if b["status"] == "PASS" and items[name]["status"] == "NOTRUN" and ".drag." in "." + name:
                items[name] = {"status": "FAIL", "detail": "where the baseline's window moved, "
                               + items[name]["detail"]}
            continue
        group = name.split(".", 1)[0]
        if group == "step":
            group = {"boot": "boot", "bench": "perf"}.get(name[5:], "mac" if name[5:] in MAC_STEPS else "steps")
        part = {"mac": "mac", "step": "steps", "cargo": "mac", "perf": "bench"}.get(group, "boot")
        if part != "steps" and part not in ran:
            items[name] = {"status": "NOTRUN", "detail": f"the {part} part did not run"}
            # The bench leg reports and never blocks: a run without it is still complete.
            incomplete = incomplete or part != "bench"
        elif b["status"] == "PASS":
            items[name] = {"status": "FAIL", "detail": "missing, where the baseline passed"}
    # A failure an entry of the known-failures list matches exactly is KNOWN and decides nothing.
    entries = []
    for e in run.meta.get("known_fails") or []:
        if not isinstance(e, dict) or not all(e.get(k) for k in ("item", "pattern", "owner", "ledger")):
            lines.append(f"  note: a known-failures entry without its item, pattern, owner and ledger row is ignored: {e}")
            continue
        entries.append(e)
        it = items.get(e["item"])
        if it and it["status"] == "FAIL" and known_match(e, it["detail"]):
            items[e["item"]] = {"status": "KNOWN", "detail": f"{it['detail']} (known: {e.get('why', '')}; "
                                f"owner {e['owner']}; {e['ledger']})"}
    cur["known"] = [dict(e, status=items.get(e["item"], {}).get("status", "absent")) for e in entries]
    for name, it in sorted(items.items()):
        if it["status"] == "FAIL":
            failed.append(name)
    # Metrics: a regression is a failed item of its own, metric.NAME.
    mlines = []
    bm = (base or {}).get("metrics", {})
    for name in sorted(cur["metrics"]):
        v = cur["metrics"][name]
        g = gated(name)
        shown = g or any(re.fullmatch(rx, name) for rx in SHOWN)
        if not shown:
            continue
        if name not in bm:
            mlines.append(f"  {name:48s} {v:>12.6g}   (no baseline)")
            continue
        bv = bm[name]
        rel = (v - bv) / abs(bv) if bv else 0.0
        verdict = ""
        if g and tool_changed(name, run.meta, base["meta"]):
            verdict = "not gated: its tool changed since the baseline"
        elif g:
            bnd = band(name, noise)
            worse = -rel * g[0]
            if worse > bnd:
                verdict = f"REGRESSION beyond {bnd:.1%}"
                failed.append("metric." + name)
                items["metric." + name] = {"status": "FAIL", "detail": f"{v:g} against {bv:g} ({rel:+.1%}), band {bnd:.1%}"}
            elif -worse > bnd:
                verdict = f"better beyond {bnd:.1%}"
            else:
                verdict = f"within {bnd:.1%}"
        mlines.append(f"  {name:48s} {v:>12.6g}   base {bv:>12.6g}  {rel:+7.1%}  {verdict}")
    benched = "bench" in ran or os.path.exists(os.path.join(run.path, "results", "guest.log"))
    for name in sorted(bm):
        if gated(name) and name not in cur["metrics"]:
            if benched and not tool_changed(name, run.meta, base["meta"]):
                mlines.append(f"  {name:48s} {'missing':>12s}   base {bm[name]:>12.6g}  REGRESSION (missing)")
                failed.append("metric." + name)
                items["metric." + name] = {"status": "FAIL", "detail": f"missing, where the baseline had {bm[name]:g}"}
    cur["items"] = items
    return cur, failed, incomplete, lines, mlines


def known_match(e, detail):
    """Whether a known-failures entry matches a failure's detail: its pattern, then its relation."""
    m = re.search(e["pattern"], detail)
    if not m:
        return False
    rel = e.get("relation")
    if not rel:
        return True
    # "A OP B", or "A OP B + K" / "A OP B - K", OP one of = < <= > >=, A and B named groups.
    r = re.fullmatch(r"\s*(\w+)\s*(<=|>=|=|<|>)\s*(\w+)\s*(?:([+-])\s*(\d+))?\s*", rel)
    if not r:
        return False
    a, op, b, sign, k = r.groups()
    try:
        va, vb = int(m.group(a)), int(m.group(b))
    except (IndexError, ValueError, TypeError):
        return False
    vb += (int(k) if sign == "+" else -int(k)) if k else 0
    return {"=": va == vb, "<": va < vb, "<=": va <= vb, ">": va > vb, ">=": va >= vb}[op]


def known_history(path, cur):
    """Records this run's status of each known entry in the history file (TSV: RUNID SHA ITEM
    STATUS), replacing the run's earlier lines, and returns each entry's report line."""
    runid, sha = cur["meta"].get("runid", "?"), cur["meta"].get("sha", "?")
    rows = [l.split("\t") for l in read(path).splitlines() if l.count("\t") == 3]
    rows = [r for r in rows if r[0] != runid]
    rows += [[runid, sha, e["item"], e["status"]] for e in cur["known"]]
    with open(path + ".new", "w") as f:
        f.write("".join("\t".join(r) + "\n" for r in rows))
    os.replace(path + ".new", path)
    out = []
    for e in cur["known"]:
        mine = [r for r in rows if r[2] == e["item"]]
        ran = [r for r in mine if r[3] in ("PASS", "FAIL", "KNOWN")]
        hit = [r for r in mine if r[3] == "KNOWN"]
        out.append(f"  {e['item']}: this run {e['status']}; known hits in {len(hit)} of the {len(ran)} runs "
                   f"that ran it (owner {e['owner']}; {e['ledger']})")
    return out


def write_report(run, base, noise, history=None):
    cur, failed, incomplete, notes, mlines = compare(run, base, noise)
    known_lines = known_history(history, cur) if history and cur.get("known") else []
    counts = {}
    for it in cur["items"].values():
        counts[it["status"]] = counts.get(it["status"], 0) + 1
    # Correctness decides; the bench leg's failures and regressions are reported beside it.
    bench_issues = [f for f in failed if f.startswith(("perf.", "metric.")) or f == "step.bench"]
    failed = [f for f in failed if f not in bench_issues]
    verdict = "FAIL" if failed else ("INCOMPLETE" if incomplete else "PASS")
    m = run.meta
    ran = set(m.get("ran", []))
    benched = "bench" in ran or os.path.exists(os.path.join(run.path, "results", "guest.log"))
    bench = (f"bench: {len(bench_issues)} failures or regressions (not gating)" if bench_issues
             else "bench: no regression beyond the noise band" if benched else "bench: not run")
    out = [f"lane-gate {verdict}: {m.get('sha', '?')} ({m.get('subject', '')}); {bench}",
           f"run {m.get('runid', '?')}, {m.get('wall_s', '?')} s wall; baseline "
           + (f"{base['meta'].get('sha', '?')[:12]} ({base['samples']} samples)" if base else "none")
           + (f"; a control: the host build {m['host_build']} against this commit's guest" if m.get("host_build") else "")
           + (f"; beside a fixed GPU load (gpuload, duty {m['load']})" if m.get("load") else "")
           + bench_host(cur["metrics"]),
           "items: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))]
    for title, names in (("failed:", failed), ("bench (not gating):", bench_issues)):
        if names:
            out.append(title)
        for f in names:
            it = cur["items"].get(f)
            out.append(f"  {f}: {it['detail'] if it else ''}")
    if cur.get("known"):
        out.append("known failures (not gating):")
        out += known_lines or [f"  {e['item']}: this run {e['status']}" for e in cur["known"]]
    out += notes
    out.append("")
    out.append("items:")
    groups = {}
    for name, it in sorted(cur["items"].items()):
        g = name.split(".", 1)[0]
        groups.setdefault(g, []).append((name, it))
    for g, its in groups.items():
        if g in ("sweep",) or (g in ("decode", "perf") and len(its) > 6):
            st = {}
            for _, it in its:
                st[it["status"]] = st.get(it["status"], 0) + 1
            out.append(f"  {g}: " + ", ".join(f"{v} {k}" for k, v in sorted(st.items())))
            # A part that did not run at all takes one line, not one per item.
            if len({(it["status"], it["detail"]) for _, it in its}) == 1 and its[0][1]["status"] != "PASS":
                out[-1] += f" ({its[0][1]['detail']})"
                continue
            for name, it in its:
                if it["status"] != "PASS":
                    out.append(f"    {it['status']:7s} {name}: {it['detail']}")
            continue
        for name, it in its:
            out.append(f"  {it['status']:7s} {name}: {it['detail']}")
    out.append("")
    out.append("metrics (gated ones show their verdict; the band is max(floor, "
               f"{NOISE_K:g} x the largest relative deviation repeated runs showed)):")
    out += mlines
    text = "\n".join(out) + "\n"
    with open(os.path.join(run.path, "report.txt"), "w") as f:
        f.write(text)
    res = dict(cur)
    res["verdict"] = verdict
    res["failed"] = failed
    res["bench_issues"] = bench_issues
    with open(os.path.join(run.path, "results.json"), "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    sys.stdout.write(text)
    return 1 if failed else (2 if incomplete else 0)


def main(argv):
    if argv[:1] == ["--record"]:
        run, root = argv[1], argv[2]
        res = json.loads(read(os.path.join(run, "results.json")))
        sha = res["meta"]["sha"]
        d = os.path.join(root, sha, "runs")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, res["meta"]["runid"] + ".json"), "w") as f:
            json.dump(res, f, indent=1, sort_keys=True)
        print(f"recorded {res['meta']['runid']} as a sample of baseline {sha[:12]}")
        return 0
    if argv[:1] == ["--noise-from"]:
        dev = noise_from(argv[1])
        with open(argv[2], "w") as f:
            json.dump(dev, f, indent=1, sort_keys=True)
        for n in sorted(dev):
            if gated(n):
                print(f"  {n:48s} max deviation {dev[n]:6.2%} -> band {band(n, dev):6.2%}")
        return 0
    run = Run(argv[0]).parse()
    base, noise, history = None, {}, None
    i = 1
    while i < len(argv):
        if argv[i] == "--baseline":
            base = median_baseline(argv[i + 1])
        elif argv[i] == "--noise":
            noise = json.loads(read(argv[i + 1], "{}") or "{}")
        elif argv[i] == "--history":
            history = argv[i + 1]
        i += 2
    return write_report(run, base, noise, history)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
