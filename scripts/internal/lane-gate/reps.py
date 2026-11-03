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

# Between the modes msi's 5080 shows under load: 2392 to 2805 MHz high, 1.2 to 1.5 GHz under the
# power cap, and 1.75 to 2.38 GHz in P4 from idle.
CORE_MIN_MHZ = 2400
# nvidia-smi's clocks change only about every half second under load (the driver's update): a span
# is judged on the readings around it, from the one before it to those half a second after it.
LAG_S = 0.5


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
    """The readings around [a, b]: the one before a (the clock a ran at), those inside, and those
    up to LAG_S after b (a change during the span shows there), or None where the samples do not
    cover the span."""
    inside = [r for r in rows if a <= r[0] <= b + LAG_S]
    before = [r for r in rows if r[0] < a]
    after = [r for r in rows if r[0] > b + LAG_S]
    if not before or not after:
        return None
    return [before[-1]] + inside


def high(rows, a, b):
    """The core clocks around [a, b] where all are at CORE_MIN_MHZ or more, else None."""
    s = span(rows, a, b)
    return [r[2] for r in s] if s is not None and min(r[2] for r in s) >= CORE_MIN_MHZ else None


def judge(cases, rows):
    """Per case: {"reps": n, "ok": [the high-clock repetitions], "window_ok": a run of the case at
    the high clock throughout (cases without repetitions), "core": [the core clock over its runs],
    "ok_core": [the core clock around what qualified], "states", "mem"}."""
    out = {}
    for case, runs in cases.items():
        if not clocked(case):
            continue
        j = {"reps": 0, "ok": [], "window_ok": False, "core": [], "ok_core": [], "states": [], "mem": []}
        for run in runs:
            if run["end"] is None or not run.get("ok"):
                continue
            s = span(rows, run["start"], run["end"]) or []
            j["core"] += [r[2] for r in s]
            j["states"] += [r[1] for r in s]
            j["mem"] += [r[3] for r in s]
            for rep in run["reps"]:
                j["reps"] += 1
                c = high(rows, rep[0] - rep[1], rep[0])
                if c:
                    j["ok"].append(rep)
                    j["ok_core"] += c
            c = None if run["reps"] else high(rows, run["start"], run["end"])
            if c:
                j["window_ok"] = True
                j["ok_core"] += c
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
