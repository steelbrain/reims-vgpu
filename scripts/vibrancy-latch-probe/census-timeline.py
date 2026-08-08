#!/usr/bin/env python3
"""census-timeline.py — find the window where a rail changed and did not change back.

A latching degradation is a *step*, and a two-point before/after diff is the
wrong instrument for it: it cannot say whether the rail fell once and stayed
down or has been oscillating all along, and it charges every difference in
workload between the two points to the bug. This reads every `store_routes`
window in a fail log in order and looks for counters whose behaviour splits the
boot in two.

For each counter it finds the split point that best separates the boot into a
"before" and an "after" with different means, then scores it by how much of the
counter's total variance that single step explains. A counter that steps once
and holds scores near 1.0; one that oscillates, ramps, or simply tracks the
workload scores low. Counters are ranked by score, so the rails that latched
sort above the rails that were merely busy.

The ratio rows matter more than the raw ones. A rung ladder that demotes shows
up as a *share* moving even when every absolute count is dominated by how hard
the guest happened to be drawing, which is exactly the confound that makes raw
per-window rates unreadable across a load phase.

Usage:
  scripts/vibrancy-latch-probe/census-timeline.py FAILLOG [--top N] [--min-total N]
  scripts/vibrancy-latch-probe/census-timeline.py FAILLOG --track name1,name2,...

`--track` prints the raw per-window series for named counters instead of the
ranking, for reading a step you have already found.

It makes no claim about *why* a step happened. It says which counter stepped and
at which window, and the fail-log lines around that window are what say why.
"""

import argparse
import re
import sys

# Shares, not counts. Each entry is (label, numerator, denominator-terms): a
# ladder's rung is only interpretable against the ladder's own total, because
# every absolute count on it also moves with how much the guest drew.
RATIOS = [
    ("share:t11rung_resident", ["t11rung_resident"],
     ["t11rung_resident", "t11rung_host_cache", "t11rung_zero_copy",
      "t11rung_guest_memo", "t11rung_miss"]),
    ("share:t11rung_host_cache", ["t11rung_host_cache"],
     ["t11rung_resident", "t11rung_host_cache", "t11rung_zero_copy",
      "t11rung_guest_memo", "t11rung_miss"]),
    ("share:t11rung_guestpages", ["t11rung_zero_copy", "t11rung_guest_memo"],
     ["t11rung_resident", "t11rung_host_cache", "t11rung_zero_copy",
      "t11rung_guest_memo", "t11rung_miss"]),
    ("share:gw_vouched", ["gw_vouched"],
     ["gw_vouched", "gw_refused_guest_store", "gw_refused_host_write",
      "gw_unarmed"]),
    ("share:render_flush_gpu_direct", ["render_flush_gpu_direct"],
     ["render_flush_gpu_direct", "render_flush_leased"]),
    ("share:zc_buffer_dmabuf", ["zc_buffer_dmabuf"],
     ["zc_buffer_dmabuf", "zc_buffer_below_floor"]),
]


def windows(path):
    """Every `store_routes` window in order, as dicts, with its `t=` if present.

    `t=` is the device's own clock and is carried through only as a label — the
    ordering used everywhere below is the order the windows were emitted in.
    """
    out = []
    for line in open(path, errors="replace"):
        if " store_routes " not in f" {line}":
            continue
        got = {k: int(v) for k, v in re.findall(r"\b([a-z0-9_]+)=(\d+)\b", line)}
        if got:
            out.append(got)
    return out


def series_for(wins, name, num, den):
    """The share series for one ratio row, and the windows it is defined on.

    A window where the denominator is zero is not a zero share, it is a window
    that did not exercise the ladder at all. Scoring it as zero would invent a
    step wherever the guest went idle, which is the most common shape in any
    boot, so those windows are dropped rather than filled.
    """
    xs, idx = [], []
    for i, w in enumerate(wins):
        d = sum(w.get(k, 0) for k in den)
        if d <= 0:
            continue
        xs.append(sum(w.get(k, 0) for k in num) / d)
        idx.append(i)
    return xs, idx


def best_step(xs):
    """Best single split of `xs`, as (score, split, mean_before, mean_after).

    Score is the fraction of total squared deviation removed by replacing one
    mean with two — the same quantity a one-break changepoint test maximises.
    1.0 is a perfect step, 0.0 is a series the split does not describe at all.
    """
    n = len(xs)
    if n < 6:
        return (0.0, 0, 0.0, 0.0)
    total = sum(xs)
    grand = total / n
    sse_one = sum((v - grand) ** 2 for v in xs)
    if sse_one <= 1e-12:
        return (0.0, 0, grand, grand)
    # Running prefix sums so the scan is linear rather than quadratic: with a
    # window per second, a long boot is thousands of points and this runs once
    # per counter.
    pre = 0.0
    pre_sq = 0.0
    tot_sq = sum(v * v for v in xs)
    best = (0.0, 0, grand, grand)
    # Leave a margin at both ends: a "step" whose after-side is two windows long
    # is one noisy window, not a rail that changed state.
    margin = max(3, n // 20)
    for i in range(n):
        pre += xs[i]
        pre_sq += xs[i] * xs[i]
        k = i + 1
        if k < margin or n - k < margin:
            continue
        m1 = pre / k
        m2 = (total - pre) / (n - k)
        sse = (pre_sq - k * m1 * m1) + ((tot_sq - pre_sq) - (n - k) * m2 * m2)
        score = 1.0 - sse / sse_one
        if score > best[0]:
            best = (score, k, m1, m2)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("faillog")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--min-total", type=int, default=200,
                    help="ignore raw counters whose whole-boot total is below this")
    ap.add_argument("--track", default="")
    args = ap.parse_args()

    wins = windows(args.faillog)
    if not wins:
        print("no store_routes windows in that log", file=sys.stderr)
        return 2
    print(f"{len(wins)} store_routes windows")

    if args.track:
        names = [n.strip() for n in args.track.split(",") if n.strip()]
        for name in names:
            vals = [w.get(name, 0) for w in wins]
            print(f"\n{name}")
            print("  " + " ".join(str(v) for v in vals))
        return 0

    rows = []
    for label, num, den in RATIOS:
        xs, _ = series_for(wins, label, num, den)
        if len(xs) < 6:
            continue
        score, split, m1, m2 = best_step(xs)
        rows.append((score, label, split, len(xs), m1, m2, True))

    keys = set()
    for w in wins:
        keys.update(w)
    for k in sorted(keys):
        if k == "t":
            continue
        vals = [w.get(k, 0) for w in wins]
        if sum(vals) < args.min_total:
            continue
        score, split, m1, m2 = best_step([float(v) for v in vals])
        rows.append((score, k, split, len(vals), m1, m2, False))

    rows.sort(key=lambda r: -r[0])
    print(f"\n{'counter':<44} {'step':>6} {'of':>5} {'before':>13} {'after':>13}  score")
    print("-" * 96)
    shown = 0
    for score, label, split, n, m1, m2, is_ratio in rows:
        if shown >= args.top:
            break
        # A step that changes the mean by under a twentieth is a step the eye
        # would not find either; it is precision, not a finding.
        if m1 and abs(m2 - m1) / max(abs(m1), 1e-9) < 0.05:
            continue
        fmt = "{:>13.3f}" if is_ratio else "{:>13.1f}"
        print(f"{label:<44} {split:>6} {n:>5} "
              + fmt.format(m1) + " " + fmt.format(m2) + f"  {score:.3f}")
        shown += 1
    print("\nA score near 1.0 is a rail that changed once and held; a low one "
          "tracks the workload.\nThe fail-log lines around the step window are "
          "what say why it stepped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
