"""Which of a bench boot's numbers ran at the GPU's high core clock (the lane gate's report.py, and
msi-boot.sh on the lane host, which runs the frame cases again where none of their repetitions did).

msi's RTX 5080 Laptop holds its core at about 1.2 to 1.5 GHz under its power cap ("SW power cap",
115 W), 2 s into some starts from idle and about 10 s into sustained load, for up to 12 s; its
frame cases follow the core clock (docs/graphics-kext/benchmark.md). A GPU-bound case's number
(clocked) counts only from time the core ran at CORE_MIN_MHZ or more throughout: from its
repetitions that did, where bench -v prints them ("# rep" lines: the frame cases', and compute.fma's,
blit.256m's, sample.tex's and mps.sgemm's in bench builds that print theirs), else where its whole
run did. The other cases (draws, upload, readback, onscreen) are bound by the host's CPU, memory and
display, not by the GPU's clock: report.py judges them by the host's quiet instead.

Inputs: guest.log (guest-run.sh's lines, each after the host's time in epoch seconds) and
gpu-clocks.txt (the host's nvidia-smi ten times a second: "EPOCH PSTATE CORE_MHZ MEM_MHZ POWER_W").

    python3 -I reps.py GUEST_LOG GPU_CLOCKS
prints the cases that ran, printed repetitions, and have none at the high clock, one line, space
separated (nothing where every one has some, or where the host sampled no clocks).
"""
import re
import sys

CORE_MIN_MHZ = 2500


def clocked(case):
    """Whether a bench case is GPU-bound: its number follows the GPU's clocks."""
    return case.startswith(("frag.", "compute.", "blit.", "sample.", "mps."))

# A frame case's repetition, and any other case's ("# rep NAME R at S s: MS ms, key=value ...").
FRAME_REP = re.compile(r"# rep (\S+) (\d+) at [0-9.]+ s: (\d+) frames \(command buffers\) in ([0-9.]+) ms, "
                       r"([0-9.]+) a second, gpu span ([0-9.]+) ms a frame")
REP = re.compile(r"# rep (\S+) (\d+) at [0-9.]+ s: ([0-9.]+) ms, (.*)$")


def numbers(text):
    out = {}
    for k, v in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=(-?[0-9.]+(?:e-?[0-9]+)?)(?=[\s,]|$)", text):
        try:
            out[k] = float(v)
        except ValueError:
            pass
    return out


def parse_guest(text):
    """Each bench case's runs, in order: {case: [{"start": t, "end": t or None, "reps": [...]}]},
    a repetition (end, seconds, its numbers: a frame case's fps and gpu_ms, any other's as its line
    names them), its end the host's time of its line, which bench prints as it ends."""
    cases = {}
    for line in text.splitlines():
        m = re.match(r"([0-9.]+) (.*)$", line)
        if not m:
            continue
        t, body = float(m.group(1)), m.group(2)
        if body.startswith("# start "):
            cases.setdefault(body[len("# start "):].strip(), []).append({"start": t, "end": None, "reps": []})
        elif body.startswith("# rep "):
            r = FRAME_REP.match(body)
            if r and cases.get(r.group(1)):
                cases[r.group(1)][-1]["reps"].append(
                    (t, float(r.group(4)) / 1e3, {"fps": float(r.group(5)), "gpu_ms": float(r.group(6))}))
                continue
            r = REP.match(body)
            if r and cases.get(r.group(1)):
                cases[r.group(1)][-1]["reps"].append((t, float(r.group(3)) / 1e3, numbers(r.group(4))))
        elif body.startswith("bench: "):
            r = re.match(r"bench: (\S+) (OK|FAIL|SKIP|NOTRUN)\b", body)
            if r and cases.get(r.group(1)) and cases[r.group(1)][-1]["end"] is None:
                cases[r.group(1)][-1]["end"] = t
                cases[r.group(1)][-1]["ok"] = r.group(2) == "OK"
    return cases


def parse_clocks(text):
    """gpu-clocks.txt's samples: (epoch, P number, core MHz, memory MHz), in time order."""
    rows = []
    for line in text.splitlines():
        f = line.split()
        if len(f) >= 4 and re.fullmatch(r"P[0-9]+", f[1]):
            try:
                rows.append((float(f[0]), int(f[1][1:]), float(f[2]), float(f[3])))
            except ValueError:
                pass
    return sorted(rows)


def span(rows, a, b):
    """The samples over [a, b]: those inside, and the one before a (the clock a ran at), or None
    where the samples do not cover the span."""
    inside = [r for r in rows if a <= r[0] <= b]
    before = [r for r in rows if r[0] < a]
    after = [r for r in rows if r[0] > b]
    if not before or not after:
        return None
    return [before[-1]] + inside


def high(rows, a, b):
    s = span(rows, a, b)
    return s is not None and min(r[2] for r in s) >= CORE_MIN_MHZ


def judge(cases, rows):
    """Per case: {"reps": n, "ok": [the high-clock repetitions], "window_ok": a run of the case at
    the high clock throughout (cases without repetitions), "core": [the core clock over its runs]}."""
    out = {}
    for case, runs in cases.items():
        if not clocked(case):
            continue
        j = {"reps": 0, "ok": [], "window_ok": False, "core": [], "states": [], "mem": []}
        for run in runs:
            if run["end"] is None or not run.get("ok"):
                continue
            s = span(rows, run["start"], run["end"]) or []
            j["core"] += [r[2] for r in s]
            j["states"] += [r[1] for r in s]
            j["mem"] += [r[3] for r in s]
            for rep in run["reps"]:
                j["reps"] += 1
                if high(rows, rep[0] - rep[1], rep[0]):
                    j["ok"].append(rep)
            if not run["reps"] and high(rows, run["start"], run["end"]):
                j["window_ok"] = True
        if j["core"]:
            out[case] = j
    return out


def retry(cases, rows):
    """The cases that ran, printed repetitions, and have none at the high clock."""
    return [c for c, j in judge(cases, rows).items() if j["reps"] and not j["ok"]]


if __name__ == "__main__":
    def read(p):
        try:
            with open(p, errors="replace") as f:
                return f.read()
        except OSError:
            return ""
    print(" ".join(retry(parse_guest(read(sys.argv[1])), parse_clocks(read(sys.argv[2])))))
