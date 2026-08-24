#!/usr/bin/env python3
"""Score a driven boot's fail log on CPU *and* GPU microseconds per draw.

Reads `/tmp/reims-vgpu-fail.log` (or the copies a harness keeps) and prints one
line per boot. It reads a log, never this repository's source; see `AGENTS.md`'s
ban on source-scanning gates for why that distinction is the whole licence for a
script like this to exist.

Two things it does that the obvious one-liner does not, and both were mistakes
made on real boots before they were fixed here.

**It joins the censuses by their own `t`, not by line ordinal.** `drain_duty`,
`gpu_span`, `window_publish` and `store_routes` are emitted from different places
and each skips windows the others do not — an idle second costs `gpu_span`
nothing and `drain_duty` a line. Pairing them by position therefore drifts within
a boot and pulls idle-desktop publishes into a driven band. The harness that had
been scoring driven Maps boots did exactly that and reported ~31 fps where the
same logs, joined by `t`, read 47-52.

**It reports both measured halves against their own populations.** CPU work is
the whole packet-processing span (`proc_us`), not only the nested draw encoder.
GPU time travels with `retired_draws` on the timestamped ring slot, so that is
its exact denominator. A CPU census draw and a retired GPU draw usually converge
over a sustained window, but assuming they are identical defeats the reason the
GPU probe carries its own count.

Columns:

    n         busy census windows scored (drain duty >= 0.5, draws > 0)
    cpu       drain_duty proc_us / guest draws
    gpu       gpu_span busy_us / retired_draws
    sum       the two per-draw prices added (interpret with overlap/occupancy)
    fps       host-window presents / cadence window time over the probe slice
    offered   host-window offers / cadence window time over the probe slice
    duty      mean drain_duty duty
    draws/s   guest draws / summed matched window time
    occ       measured CPU plus GPU busy time / summed matched window time
    d/frame   draws per frame, which is the workload and drifts between boots

`occ` is allowed to exceed 1.0 when CPU and GPU overlap. It is a measured-time
sum, not evidence that the two phases serialize.

`d/frame` is printed because frames are **not** comparable across boots without
the amount of guest work in each frame. Do not derive FPS from `sum` unless the
same workload has separately established how its CPU and GPU phases overlap.

Usage:  scripts/boot-score/boot-score.py FAIL_LOG [FAIL_LOG ...]
"""

import re
import statistics
import sys

FIELD = re.compile(r"(\w+)=(-?[\d.]+)")
# Census windows are ~1 s apart and the emitters stamp `t` within a couple of
# milliseconds of each other, so a small window is an exact match and cannot
# reach a neighbouring second.
SLOP = 3


def _fields(line):
    return dict(FIELD.findall(line))


def score(path):
    gpu, pub, duty, cadence = {}, {}, [], []
    legacy_gpu = False
    with open(path, errors="ignore") as handle:
        for line in handle:
            if line.startswith("OFF gpu_span "):
                f = _fields(line)
                if "t" in f and "busy_us" in f and "retired_draws" in f:
                    gpu[int(float(f["t"]))] = (
                        float(f["busy_us"]),
                        float(f["retired_draws"]),
                    )
                elif "t" in f and "busy_us" in f:
                    legacy_gpu = True
            elif line.startswith("OFF window_publish "):
                f = _fields(line)
                if "t" in f and "fresh" in f and "win_ms" in f:
                    pub[int(float(f["t"]))] = (float(f["fresh"]), float(f["win_ms"]))
            elif line.startswith("OFF host_window_cadence "):
                f = _fields(line)
                if "present_hz" in f and "offered_hz" in f:
                    cadence.append((float(f["present_hz"]), float(f["offered_hz"])))
            elif line.startswith("OFF drain_duty "):
                duty.append(_fields(line))

    def near(table, t):
        for k in range(-SLOP, SLOP + 1):
            if t + k in table:
                return table[t + k]
        return None

    n = draws = retired_draws = cpu_us = gpu_us = duty_sum = 0
    fresh = frame_draws = busy_win_ms = publish_win_ms = 0
    for f in duty:
        if "t" not in f or "proc_us" not in f or "win_ms" not in f:
            continue
        d, count = float(f["duty"]), float(f["draws"])
        if d < 0.5 or count <= 0:
            continue
        gpu_window = near(gpu, int(float(f["t"])))
        if gpu_window is None or gpu_window[1] == 0:
            continue
        n += 1
        draws += count
        cpu_us += float(f["proc_us"])
        gpu_us += gpu_window[0]
        retired_draws += gpu_window[1]
        duty_sum += d
        busy_win_ms += float(f["win_ms"])
        published = near(pub, int(float(f["t"])))
        if published:
            fresh += published[0]
            publish_win_ms += published[1]
            frame_draws += count

    if not n:
        if legacy_gpu:
            return f"{path}: gpu_span lacks retired_draws — log predates the exact GPU denominator"
        return f"{path}: no joined busy windows — undriven, or a log from the test suite"
    per_second = draws / (busy_win_ms / 1000)
    cpu_per_draw = cpu_us / draws
    gpu_per_draw = gpu_us / retired_draws
    total = cpu_per_draw + gpu_per_draw
    fps = statistics.median(sample[0] for sample in cadence) if cadence else 0.0
    offered_fps = statistics.median(sample[1] for sample in cadence) if cadence else 0.0
    occupancy = (cpu_us + gpu_us) / (busy_win_ms * 1000)
    return (
        f"{path:<40} n={n:<3} cpu={cpu_per_draw:5.2f} gpu={gpu_per_draw:5.2f} "
        f"sum={total:5.2f} fps={fps:5.1f} offered={offered_fps:5.1f} "
        f"duty={duty_sum / n:.2f} "
        f"draws/s={per_second:6.0f} occ={occupancy:.2f} "
        f"d/frame={frame_draws / fresh if fresh else 0:6.0f}"
    )


def main(argv):
    if not argv:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    for path in argv:
        print(score(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
