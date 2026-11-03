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
showed. A bench number counts only from time the GPU's core ran at its high clock (reps.py): a
frame case's from its repetitions that did, any other case's where its whole run did; one that
did not is KNOWN-clock, compared with nothing and in no band. Where the bench boot ran beside a
build on the lane host (HOST_NOT_QUIET), its other numbers (bound by the host's CPU and memory:
draws, upload, readback, present timing, drags, decode, WebGL) are tagged, compared with nothing
and in no band; a baseline's median leaves them out too. Exit 0 when every item passes and no
gated metric regressed, 1 when one failed, 2 when a part did not run (an incomplete run passes
nothing).

A boot's numbers are TAINTED where tainted/LEG in the run's directory (LEG boot or bench) holds
why, such as someone using the guest while it measured: they are shown, compared with nothing
and left out of every baseline and band; its items and its host's counts still decide. Record
the run again (--record) once the file is written.
"""
import datetime
import json
import os
import re
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reps  # noqa: E402

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
         r"host\..*", r"bench\.draws\.10k\.encode_ms", r"(shared\.)?icb\..*", r"(core_mhz|clock_reps_ok)\.bench\..*",
         r"shared\.(host\..*|webgl\.raf_fps|present\..*\.presented_fps|composites_s\.present\.fullscreen)"]


def gpu_bound(name):
    """Whether a metric is a GPU-bound bench case's (reps.clocked): judged by the GPU's clock, and
    usable beside a build on the lane host."""
    return name.startswith("bench.") and reps.clocked(name[len("bench."):].rsplit(".", 1)[0])


def host_tagged(metrics, name):
    """Whether a bench boot's metric is tagged: the boot ran beside a build (HOST_NOT_QUIET) and the
    metric is not GPU-bound. The correctness boot's (shared.) are never gated."""
    return metrics.get("host.host_quiet") == 0 and not name.startswith(("shared.", "host.", "step_s.", "pstate.", "core_mhz.", "mem_mhz.", "clock_")) \
        and not gpu_bound(name)


def tainted(metrics, name):
    """Whether a metric is from a boot whose numbers are tainted (tainted/LEG): any of its numbers
    but its host's counts and waits."""
    leg = "boot" if name.startswith("shared.") else "bench"
    return metrics.get("taint." + leg) == 1 and not name.startswith(
        ("host.", "shared.host.", "taint.", "step_s."))


def clock_ok_of(metrics, name):
    """Whether a bench metric's number is from the GPU's high core clock (reps.py): True, False, or
    None where the run sampled no clocks for its case."""
    if not name.startswith("bench."):
        return None
    v = metrics.get("clock_ok.bench." + name[len("bench."):].rsplit(".", 1)[0])
    return None if v is None else v == 1


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
        # A crash report's summary keeps its frames; any other detail its first 300 characters.
        self.items[name] = {"status": status, "detail": detail.strip()[:1500 if name.endswith("crash-reports") else 300]}

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
        self.taint = {}
        for d, self.ip, self.mp in legs:
            why = read(os.path.join(self.path, "tainted", os.path.basename(d))).strip()
            if why and os.path.exists(os.path.join(d, "guest.log")):
                leg = os.path.basename(d) if d != r else "bench"
                self.taint[leg] = why
                self.metrics["taint." + leg] = 1
            if os.path.exists(os.path.join(d, "guest.log")):
                self.guest(read(os.path.join(d, "guest.log")))
                self.host(d)
                self.drag(d)
                self.video(d)
                self.clocks(d)
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
        self.guest_text = text  # clocks() reads the bench cases' times from it
        crashes, crash_lines = None, []
        icb = {}  # oracle_icb's cases: their lines and exit status (the icb phase)
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
                if parts[2] == "features" and len(parts) > 4:
                    # The bits the kext took, by number (FEATURES_LO's: SUBMIT_ROOM is 16, ICB_SLOTS 17).
                    m2 = re.search(r"RgfxFeaturesTaken = (\d+)", parts[4])
                    if m2:
                        bits = [str(b) for b in range(64) if int(m2.group(1)) >> b & 1]
                        self.items[self.ip + "guest.features"]["detail"] += " (bits taken: " + ",".join(bits) + ")"
                if parts[2] == "metal-device" and len(parts) > 4:
                    m2 = re.search(r'device "([^"]+)"', parts[4])
                    self.metal_device = m2.group(1) if m2 else None
            elif body.startswith("GATE info video-offer "):
                m2 = re.match(r"GATE info video-offer hevc=(\d)", body)
                self.video_offer = {"hevc": m2 is not None and m2.group(1) == "1"}
            elif body.startswith("GATE crashes "):
                crashes = body[len("GATE crashes "):].strip()
            elif body.startswith("GATE crash "):
                crash_lines.append(body[len("GATE crash "):].strip())
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
            elif body.startswith("oracle_icb: "):
                m2 = re.match(r"oracle_icb: (\S+) \| (.*)$", body)
                if m2:
                    icb.setdefault(m2.group(1), {"lines": [], "rc": None})["lines"].append(m2.group(2))
                m2 = re.match(r"oracle_icb: (\S+) rc (\d+)$", body)
                if m2:
                    icb.setdefault(m2.group(1), {"lines": [], "rc": None})["rc"] = int(m2.group(2))
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
        self.icb(icb)
        if crashes is not None:
            names = sorted({re.sub(r"[-_]\d{4}-\d\d-\d\d.*$", "", c) for c in crashes.split() if c != "none"})
            self.crashes = names
            ws = [n for n in names if "WindowServer" in n]
            self.litem("guest.crash-reports", "FAIL" if ws else "PASS",
                       "; ".join([", ".join(names) or "none new"] + crash_lines))
        for p, (a, b) in self.phases.items():
            if a is None or b is None:
                self.litem("phase." + p, "FAIL", "started but did not end (a hang the alarm or the hold cut)")

    def icb(self, cases):
        """oracle_icb's cases (the icb phase): FAIL where a result is not what Metal draws or
        dispatches ("^ differs", "<- differs"), where the barriered dispatches came out wrong in
        any run, or where the case's child died or exited nonzero; its timing's CPU encode costs
        and execute times are metrics."""
        for case, c in sorted(cases.items()):
            lines = c["lines"]
            checks = sum(1 for l in lines if re.match(r"  \S.{59} ", l))
            bad = []
            for i, l in enumerate(lines):
                if l.strip() == "^ differs" and i:
                    bad.append(lines[i - 1].strip())
                elif l.rstrip().endswith("<- differs") or re.match(r"\s*=> .*(died on signal|exited) ", l):
                    bad.append(l.strip())
                else:
                    m = re.search(r"with barrier: (\d+) of (\d+) runs not all", l)
                    if m and int(m.group(1)):
                        bad.append(l.strip())
            if c["rc"] not in (0, None):
                bad.append(f"exit {c['rc']}")
            if c["rc"] is None:
                bad.append("no exit status (cut off)")
            self.litem(f"icb.{case}", "FAIL" if bad else "PASS",
                       (f"{len(bad)} of {checks} results wrong: " + "; ".join(re.sub(r"\s+", " ", b) for b in bad[:4]))
                       if bad else f"{checks} results as Metal's" if checks else "ran; it checks no results")
            if case != "timing":
                continue
            gpu = []
            for l in lines:
                m = re.match(r"\s*(\d+): CPU encode, (pipeline \+ 2 buffers \+ draw|draw alone \(inherited\)): (\d+) ns a command", l)
                if m:
                    key = "encode_ns" if m.group(2).startswith("pipeline") else "encode_inherited_ns"
                    self.lmetric(f"icb.{key}.{m.group(1)}", float(m.group(3)))
                m = re.match(r"\s*execute (\d+) draws: wall [0-9.]+ ms, GPU ([0-9.]+) ms", l)
                if m:
                    gpu.append(float(m.group(2)))
            if gpu:
                self.lmetric("icb.execute_gpu_ms.16384", statistics.median(gpu))

    def clocks(self, d):
        """The GPU's core clock over each bench case (gpu-clocks.txt, the host's nvidia-smi ten times
        a second; reps.py). A frame case's numbers are taken again from its repetitions at the high
        clock alone, over all its runs (the bench phase's, and benchretry's where msi-boot.sh ran it
        again): metrics clock_reps and clock_reps_ok count them, and clock_ok is 1 where any
        repetition qualified. So are compute.fma's, blit.256m's, sample.tex's and mps.sgemm's
        where the bench build prints their repetitions; where it does not, clock_ok is 1 where the
        case's whole run was at the high clock. Each case's item shows that, with its runs' median core and memory clocks and their
        commonest power state."""
        rows = reps.parse_clocks(read(os.path.join(d, "gpu-clocks.txt")))
        if not rows:
            return
        m = self.metrics
        for case, j in reps.judge(reps.parse_guest(self.guest_text), rows).items():
            p = f"bench.{case}."
            ps = max(set(j["states"]), key=j["states"].count)
            # The core clock the case's number ran at: around what qualified, else its whole run.
            core, mem = statistics.median(j["ok_core"] or j["core"]), statistics.median(j["mem"])
            self.lmetric(f"pstate.bench.{case}", ps)
            self.lmetric(f"core_mhz.bench.{case}", core)
            self.lmetric(f"mem_mhz.bench.{case}", mem)
            if j["reps"]:
                k, n = len(j["ok"]), j["reps"]
                self.lmetric(f"clock_reps.bench.{case}", n)
                self.lmetric(f"clock_reps_ok.bench.{case}", k)
                self.lmetric(f"clock_ok.bench.{case}", 1 if k else 0)
                note = f"{k} of {n} reps at {reps.CORE_MIN_MHZ} MHz or more"
                if k and case.startswith("frag."):
                    # The case's line gives the medians of all its repetitions: its numbers that
                    # follow the frame rate are scaled to the qualifying repetitions' median rate.
                    fps = statistics.median(r[2]["fps"] for r in j["ok"])
                    gpu = statistics.median(r[2]["gpu_ms"] for r in j["ok"])
                    line_fps, line_gpu = m.get(self.mp + p + "fps"), m.get(self.mp + p + "gpu_ms")
                    for key, nd in (("gpix_s", 2), ("mpix_s", 0)):
                        if line_fps and self.mp + p + key in m:
                            self.lmetric(p + key, round(m[self.mp + p + key] * fps / line_fps, nd))
                    if line_gpu and self.mp + p + "gpix_s_gpu" in m:
                        self.lmetric(p + "gpix_s_gpu", round(m[self.mp + p + "gpix_s_gpu"] * line_gpu / gpu, 2))
                    self.lmetric(p + "fps", round(fps, 1))
                    self.lmetric(p + "gpu_ms", round(gpu, 3))
                    self.lmetric(p + "wall_ms", round(statistics.median(1e3 / r[2]["fps"] for r in j["ok"]), 3))
                    note += f", judged on those: fps {fps:.1f}"
                    if self.mp + p + "gpix_s" in m:
                        note += f", gpix_s {m[self.mp + p + 'gpix_s']:g}"
                elif k:
                    # Any other case's repetition gives its own numbers: each is their median.
                    keys = sorted(set().union(*(r[2] for r in j["ok"])))
                    for key in keys:
                        vals = [r[2][key] for r in j["ok"] if key in r[2]]
                        self.lmetric(p + key, round(statistics.median(vals), 3))
                    note += ", judged on those: " + " ".join(f"{key}={m[self.mp + p + key]:g}" for key in keys)
                else:
                    note += ": not compared"
            else:
                self.lmetric(f"clock_ok.bench.{case}", 1 if j["window_ok"] else 0)
                note = (f"core at {reps.CORE_MIN_MHZ} MHz or more throughout" if j["window_ok"] else
                        f"core down to {min(j['core']):.0f} MHz: not compared")
            it = self.items.get(self.ip + "bench." + case)
            if it:
                it["detail"] = f"[{note}; P{ps} {core:.0f}/{mem:.0f} MHz] " + it["detail"]

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
            codec = name[len(self.ip + "decode."):].split(".")[0].split("-")[0]  # hevc-odd's is hevc
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
                  "load_start", "load_end", "load_duty", "host_quiet", "queues_unshared",
                  "submit_room_posted", "icb_slots_refused"):
            if re.fullmatch(r"[0-9]+(\.[0-9]*)?", kv.get(k, "")):
                self.lmetric("host." + k, float(kv[k]))
        fatal = int(kv.get("fatal_lines", "-1") or -1) + max(0, int(kv.get("serial_fatal_lines", "0") or 0))
        self.litem("host.no-fatal", "PASS" if fatal == 0 else "FAIL",
                  f"{fatal} 'reims-vgpu: fatal' lines" if fatal >= 0 else "no host log")
        # The NVIDIA driver's GPU errors over the boot (msi-boot.sh's xid.txt: "TIME XID PID NAME
        # OWN ERROR"): one from this boot's QEMU fails it; the other processes' are named.
        if kv.get("xid_lines") == "unread":
            self.litem("host.no-xid", "NOTRUN", "the lane host's kernel log could not be read")
        elif "xid_lines" in kv:
            xids = [x for x in (l.split(None, 5) for l in read(os.path.join(r, "xid.txt")).splitlines())
                    if len(x) >= 5]
            def said(x):
                return f"Xid {x[1]} ({x[5] if len(x) > 5 else '-'}) from {x[3]} pid {x[2]} at {x[0][11:]}"
            own = [said(x) for x in xids if x[4] == "own"]
            other = [said(x) for x in xids if x[4] != "own"]
            more = f"; other processes': {', '.join(other)}" if other else ""
            self.litem("host.no-xid", "FAIL" if own else "PASS",
                       (f"{len(own)} from this boot's QEMU: {', '.join(own)}" if own
                        else "no Xid line from this boot's QEMU") + more)
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
            # A drag's window moves, a live resize's size changes; Mission Control's numbers are over
            # its animations, and start_ms its first update's wait after the key.
            did = (f"; the window {'resized' if name == 'resize' else 'moved'} {kv['window_moves_s']:g} times a second"
                   if "window_moves_s" in kv else "")
            if "start_ms" in kv:
                did += f"; over {kv.get('toggles', 0):g} openings and closings, the first update {kv['start_ms']:g} ms after the key"
            self.litem("drag." + name, "FAIL" if stall else "PASS",
                       f"{kv.get('composites_s')} display updates a second, gap p99 {kv.get('gap_p99_ms')} ms, "
                       f"max {kv.get('gap_max_ms')} ms" + (" (a stall over 1 s)" if stall else "")
                       + f"{did}; QEMU {kv.get('qemu_cpu')}%, its graphics thread {kv.get('gfx_cpu')}%")
            for k in ("composites_s", "gap_p99_ms", "gap_max_ms", "window_moves_s", "start_ms"):
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
    ref["metrics_best"] = {}
    for n in names:
        # A HOST_NOT_QUIET sample's numbers other than the GPU-bound ones are left out.
        # A tainted boot's numbers are left out (tainted/LEG).
        vals = [r["metrics"][n] for r in runs if n in r["metrics"] and not host_tagged(r["metrics"], n)
                and not tainted(r["metrics"], n)]
        # A bench number is the median of the samples at the high clock, where any sample says its clock.
        clocked = [r["metrics"] for r in runs if n in r["metrics"] and clock_ok_of(r["metrics"], n) is not None
                   and not tainted(r["metrics"], n)]
        if clocked:
            vals = [x[n] for x in clocked if clock_ok_of(x, n)]
        if not vals:
            continue
        ref["metrics"][n] = statistics.median(vals)
        ref["metrics_best"][n] = max(vals)
    return ref


# Bimodal metrics, in samples without clock data (gpu-clocks.txt) alone: (regex, the slow mode's
# share of the fast one, low and high, its owner). There a value in the slow mode is KNOWN-bimodal
# and decides nothing; any other compares with the baseline's fast mode (its best sample), and
# such a metric takes no noise band from them. frag.fill.2160p ran 134 or 256 Gpix/s on msi's 5080
# from boot to boot (11-03): the two modes were the core clock under the power cap, which a
# sample with clock data judges per repetition instead.
BIMODAL = [(r"bench\.frag\.fill\.2160p\.gpix_s", 0.40, 0.65, "rgk-fill")]


def bimodal(name):
    return next((b for b in BIMODAL if re.fullmatch(b[0], name)), None)


def contains(sha, fix):
    """Whether commit sha has commit fix (git, in the repository this script is in)."""
    if not sha:
        return False
    r = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "merge-base", "--is-ancestor",
                        fix, sha], capture_output=True)
    return r.returncode == 0


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
    """Per gated metric: the largest relative distance from its commit's median seen in repeated runs.
    A GPU-bound bench number's band comes from the samples at the GPU's high clock (reps.py), any
    other's from the bench samples taken in host quiet only (host.host_quiet 1: no build on the lane
    host beside QEMU), as other work's CPU and memory use moves those numbers far past noise."""
    dev = {}
    for sha in sorted(os.listdir(bdirs_root)):
        rd = os.path.join(bdirs_root, sha, "runs")
        if sha == "LAST" or not os.path.isdir(rd):
            continue
        runs = [json.loads(read(os.path.join(rd, f))) for f in sorted(os.listdir(rd)) if f.endswith(".json")]
        if len(runs) < 2:
            continue
        for n in set().union(*(r["metrics"] for r in runs)):
            mine = [r for r in runs if n in r["metrics"] and not tainted(r["metrics"], n)]
            if gpu_bound(n):
                vals = [r["metrics"][n] for r in mine if clock_ok_of(r["metrics"], n)]
            else:
                vals = [r["metrics"][n] for r in mine if r["metrics"].get("host.host_quiet") == 1]
            if len(vals) < 2 or (bimodal(n) and not gpu_bound(n)):
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
        # Publishing is the tips' alone (--publish): a run without it misses nothing. Nor does one
        # without a stream's repeated checks (STREAM.r2, ...: decode-repeat.txt).
        if name not in items and (name == "step.publish" or re.search(r"\.r\d+$", name)):
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
        elif part in ("boot", "bench") and items.get("step." + part, {}).get("status") == "FAIL":
            # A boot that failed before its guest ran fails once, as step.boot (or step.bench).
            items[name] = {"status": "NOTRUN", "detail": f"step.{part} failed before it"}
            incomplete = incomplete or part != "bench"
        elif b["status"] == "PASS":
            items[name] = {"status": "FAIL", "detail": "missing, where the baseline passed"}
    # A failure an entry of the known-failures list matches exactly is KNOWN and decides nothing.
    entries = []
    for e in run.meta.get("known_fails") or []:
        if not isinstance(e, dict) or not all(e.get(k) for k in ("item", "pattern", "owner", "ledger")):
            lines.append(f"  note: a known-failures entry without its item, pattern, owner and ledger row is ignored: {e}")
            continue
        # An entry whose fix landed (fixed_by, a commit) holds only for commits without the fix.
        if e.get("fixed_by") and contains(run.meta.get("sha", ""), e["fixed_by"]):
            lines.append(f"  note: {e['item']}'s known failure is fixed by {e['fixed_by'][:12]}, which this commit has: it must pass")
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
        clock = cur["metrics"].get("core_mhz.bench." + name[len("bench."):].rsplit(".", 1)[0]) if gpu_bound(name) else None
        at = f"@{clock:.0f} MHz  " if clock is not None else ""
        if name not in bm:
            mark = "TAINTED, not compared" if tainted(cur["metrics"], name) else ""
            mlines.append(f"  {name:48s} {v:>12.6g}   (no baseline)  {at}{mark}".rstrip())
            continue
        bv = bm[name]
        bi = bimodal(name) if clock_ok_of(cur["metrics"], name) is None else None
        if bi:
            bv = base.get("metrics_best", {}).get(name, bv)
        rel = (v - bv) / abs(bv) if bv else 0.0
        verdict = ""
        if tainted(cur["metrics"], name):
            leg = "boot" if name.startswith("shared.") else "bench"
            verdict = f"TAINTED: {getattr(run, 'taint', {}).get(leg, 'tainted')}, not compared"
            if g:
                items.pop("metric." + name, None)
        elif host_tagged(cur["metrics"], name):
            verdict = "tagged HOST_NOT_QUIET: a build ran beside the boot, not compared"
        elif clock_ok_of(cur["metrics"], name) is False:
            case = name[len("bench."):].rsplit(".", 1)[0]
            n = cur["metrics"].get("clock_reps.bench." + case)
            why = (f"none of its {n:g} reps ran at {reps.CORE_MIN_MHZ} MHz or more" if n
                   else f"its core clock fell under {reps.CORE_MIN_MHZ} MHz")
            verdict = f"KNOWN-clock: {why}"
            if g:
                items["metric." + name] = {"status": "KNOWN", "detail": f"{v:g}: {why} (the GPU's power cap), not compared"}
        elif bi and bv and bi[1] <= v / bv <= bi[2]:
            verdict = f"KNOWN-bimodal: its slow mode ({bi[3]} owns it)"
            items["metric." + name] = {"status": "KNOWN", "detail": f"{v:g} against the fast mode's {bv:g}: the slow mode, which {bi[3]} owns"}
        elif g and tool_changed(name, run.meta, base["meta"]):
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
        mlines.append(f"  {name:48s} {v:>12.6g}   base {bv:>12.6g}  {rel:+7.1%}  {at}{verdict}")
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
