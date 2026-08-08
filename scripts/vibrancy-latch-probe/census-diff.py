#!/usr/bin/env python3
"""census-diff.py — what separates a degraded boot's fail log from a clean one?

`latch-rate.sh` scores each boot `clean` or `degraded` and keeps that boot's
whole fail log beside the verdict. That turns the vibrancy latch into a
two-sample problem: the same binary, the same workload, the same snapshot, and
the only difference is which side of the latch the boot landed on. Whatever
decides it has to show up as a difference between the two groups.

Reading one degraded log alone cannot find it. Every interesting counter in this
device is nonzero on a healthy boot too, so "335 aliased spans" means nothing
without the clean boots' number beside it. This script is the comparison.

## Reading the log correctly

Three properties of the fail log make a naive `grep -c` wrong, and all three are
handled here:

- **Two channels share the file.** An off-channel record begins with the literal
  `OFF `; a fail-channel one begins with its own event name. Ranking `reason=`
  without splitting them first inverts the queue, because the highest-volume
  `reason=` values are off-channel ordering events. Fail events and `store_routes`
  counters are therefore collected separately.
- **Counters are per-window for some names and cumulative for others.** In one
  boot `blit_dest_bound` reads 6 then 4 (a per-window delta) while
  `guest_dmabuf_pinned_kb_sum` only ever grows (a running total). Summing a
  cumulative series overcounts it by roughly the window count; taking the max of
  a per-window series throws almost all of it away. So each series is classified
  by whether it is non-decreasing across that boot's windows, and reduced with
  `max` if it is and `sum` if it is not. A name that disagrees between boots is
  reported rather than silently reduced two different ways.
- **Many fail lines are `first_sight`-deduped.** Those count *distinct keys* for
  the life of the boot, not occurrences. That is still comparable across boots —
  both sides count the same way — but it means a count of 335 is 335 distinct
  spans, not 335 events, and the ratio should be read as such.

## Output

Per metric: the clean group's mean, the degraded group's mean, and the ratio.
Sorted by how strongly the metric separates the groups. A metric that is zero on
every clean boot and nonzero on every degraded one is listed first and flagged,
because that is the shape a latch has.

Usage:
  census-diff.py --clean A.log B.log --degraded C.log D.log
  census-diff.py --run-dir /tmp/latch8          # reads verdicts.tsv itself
"""

import argparse
import gzip
import io
import os
import re
import sys

# A fail-channel record starts with its event name; `Emit` renders it as a bare
# leading token followed by `key=value` fields.
EVENT_RE = re.compile(r"^([a-z][a-z0-9_]*)(?:\s|$)")
KV_RE = re.compile(r"([a-z][a-z0-9_]*)=(\d+)\b")
REASON_RE = re.compile(r"\breason=([a-z][a-z0-9_]*)")


def open_log(path):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), errors="replace")
    return open(path, errors="replace")


def parse_log(path):
    """Return (fail_event_counts, counter_totals) for one boot's fail log.

    `counter_totals` reduces each `store_routes` series with max or sum
    depending on whether it was non-decreasing across this boot's windows.
    """
    events = {}
    # name -> list of readings, in window order
    series = {}
    with open_log(path) as fh:
        for line in fh:
            if line.startswith("OFF "):
                body = line[4:]
                if body.startswith("store_routes "):
                    for name, val in KV_RE.findall(body):
                        series.setdefault(name, []).append(int(val))
                continue
            m = EVENT_RE.match(line)
            if m:
                events[m.group(1)] = events.get(m.group(1), 0) + 1
                # The event name alone merges unrelated defects: `linux_m2v_draw`
                # is every engine rejection there is, and the one that matters
                # here is a single `reason=`. Count the pair as its own metric so
                # a class can be tracked without the noise of its siblings.
                r = REASON_RE.search(line)
                if r:
                    key = f"{m.group(1)}/{r.group(1)}"
                    events[key] = events.get(key, 0) + 1

    counters = {}
    for name, vals in series.items():
        # A single reading cannot be classified: one number is both a running
        # total and a per-window delta. Calling it monotone (which `all()` over
        # an empty pairing does) made every short log agree with itself and
        # disagree with every real boot, which is noise, not a warning. `None`
        # means unknown and is excluded from the cross-boot consistency check;
        # max and sum coincide for one reading, so the value is unaffected.
        monotone = None if len(vals) < 2 else all(b >= a for a, b in zip(vals, vals[1:]))
        counters[name] = (max(vals) if monotone is not False else sum(vals), monotone)
    return events, counters


def collect(paths):
    """Merge per-boot readings into name -> list of one value per boot."""
    ev, ct, monot = {}, {}, {}
    for i, p in enumerate(paths):
        e, c = parse_log(p)
        for name, v in e.items():
            ev.setdefault(name, [0] * len(paths))[i] = v
        for name, (v, mono) in c.items():
            ct.setdefault(name, [0] * len(paths))[i] = v
            if mono is not None:
                monot.setdefault(name, set()).add(mono)
    # Pad names first seen on a later boot.
    for d in (ev, ct):
        for name, vals in d.items():
            if len(vals) < len(paths):
                vals.extend([0] * (len(paths) - len(vals)))
    return ev, ct, monot


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", nargs="*", default=[])
    ap.add_argument("--degraded", nargs="*", default=[])
    ap.add_argument("--run-dir", help="a latch-rate.sh --out dir; reads verdicts.tsv")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    clean, degraded = list(args.clean), list(args.degraded)
    if args.run_dir:
        vpath = os.path.join(args.run_dir, "verdicts.tsv")
        if not os.path.exists(vpath):
            sys.exit(f"census-diff: no verdicts.tsv in {args.run_dir}")
        with open(vpath) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                log = os.path.join(args.run_dir, f"boot-{parts[0]}", "full-boot.log")
                if not os.path.exists(log):
                    continue
                if parts[1] == "clean":
                    clean.append(log)
                elif parts[1] == "degraded":
                    degraded.append(log)

    if not clean or not degraded:
        print(
            f"census-diff: need both groups (clean={len(clean)} degraded={len(degraded)}).",
            file=sys.stderr,
        )
        print(
            "A one-sided run is not a finding: every counter here is nonzero on a "
            "healthy boot too, so a degraded log alone cannot say which name is the "
            "difference. Score more boots.",
            file=sys.stderr,
        )
        return 2

    print(f"census-diff: {len(clean)} clean vs {len(degraded)} degraded boots\n")
    for label, kind in (("FAIL EVENTS", 0), ("STORE_ROUTES COUNTERS", 1)):
        c_ev, c_ct, c_mono = collect(clean)
        d_ev, d_ct, d_mono = collect(degraded)
        cm, dm = (c_ev, d_ev) if kind == 0 else (c_ct, d_ct)
        rows = []
        for name in sorted(set(cm) | set(dm)):
            cv = cm.get(name, [0] * len(clean))
            dv = dm.get(name, [0] * len(degraded))
            c, d = mean(cv), mean(dv)
            if c == 0 and d == 0:
                continue
            # A metric absent from every clean boot and present on every degraded
            # one is the shape a latch has; rank those first, then by fold change.
            exclusive = all(x == 0 for x in cv) and all(x > 0 for x in dv)
            ratio = (d / c) if c else float("inf")
            rows.append((exclusive, abs(ratio - 1) if ratio != float("inf") else 1e9,
                         name, c, d, ratio))
        rows.sort(key=lambda r: (not r[0], -r[1]))
        print(f"== {label} ==")
        print(f"{'metric':<44}{'clean':>12}{'degraded':>12}{'ratio':>10}")
        for exclusive, _, name, c, d, ratio in rows[: args.top]:
            r = "inf" if ratio == float("inf") else f"{ratio:.2f}"
            flag = "  <== degraded-only" if exclusive else ""
            print(f"{name:<44}{c:>12.1f}{d:>12.1f}{r:>10}{flag}")
        if kind == 1:
            mixed = [n for n in set(c_mono) & set(d_mono)
                     if len(c_mono[n] | d_mono[n]) > 1]
            if mixed:
                print(
                    "\nreduced inconsistently across boots (cumulative on some, "
                    "per-window on others) — read these with suspicion:\n  "
                    + ", ".join(sorted(mixed))
                )
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
