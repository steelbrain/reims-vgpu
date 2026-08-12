# sustained-animation-probe

Drives the guest with a **sustained, full-rate** animation and captures the
device census for exactly that window.

```sh
scripts/sustained-animation-probe/sustained-animation-probe.sh /tmp/out 40
```

Takes `(outdir, seconds)`, writes `<outdir>/window.log`, and ends by running the
analysis over it — the same interface the other driven-boot probes take, so it
is interchangeable with them in a multi-boot harness.

## Why it exists

An undriven boot measures an idle device; `AGENTS.md` already says so. This
probe exists because a **bursty** driven boot measures the bursts' *gaps*, which
is a different error and reads as a device result rather than as an idle one.

A window-server probe that opens Mission Control and Launchpad spends ~2 s of
wall clock per round waiting for their animations, so whole seconds of it have
literally zero draws. Its `present_hz` median came out at **2.8 Hz** on a device
observed sustaining **78.8 Hz** (peak 92.2) under a frame-rate test page in the
same VM, minutes apart. Nothing in the bursty capture said it was idle: the
counters were self-consistent and the log well-formed.

The consequence is not a scale factor, it is a different ranking. Same guest
rail (macos-13), same build, same quiesced host:

| `chain_phase` share | bursty window-server probe | sustained animation |
|---|---|---|
| `store` | 10.3 % | **34.9 %** |
| `engine` | 49.0 % | 28.2 % |
| `sampled` | 18.5 % | 20.9 % |
| `pipeline` | 6.2 % | 8.5 % |
| per-chain total | 129 µs | 87 µs |
| drain worker duty | 0.00 median, 0.39 peak | **0.22 median, 0.88 peak** |

The last row is the one that decides what is worth fixing. Only the sustained
arm ever makes the drain worker the bottleneck, so it is the only arm on which
a per-draw CPU saving can become frames — which is why several CPU-side wins on
the bursty probe (a bounded pipeline cache, a 39 % cut in submissions, a
twentyfold cut in `stage_us`) each bought real microseconds and **zero** frames.

Run both before any "faster" claim. A change can help one and hurt the other,
and neither is the whole workload.

## The page is served by the host, on purpose

`anim.html` is served over QEMU's user-net gateway (`10.0.2.2:8123`), not fetched
from the internet. A probe whose workload can change under it cannot be A/B'd,
and the guest rails have no reason to have working DNS. Override the port with
`ANIM_PORT`.

Everything the page draws steps per *frame*, never per wall-clock millisecond,
so a slow boot and a fast boot draw identical content per frame number and
differ only in how many frames they complete. It loads both rails that matter:
eight layers the window server composites separately, and a canvas repainted
every frame so texture content is uploaded rather than only re-composited.

## Layer promotion is forced, because a hint made the probe bimodal

`will-change: transform` is advisory. Ten boots of one pinned binary — same
snapshot, same probe, same quiesced host — split into two tight clusters with
nothing in between:

| | promoted | collapsed |
|---|---|---|
| draws per presented frame | 417.9 – 429.0 | 267.7 – 268.5 |
| `present_hz` median | 39.1 – 41.7 | 49.0 – 50.6 |

Eight of ten landed in the first. Which cluster a boot drew was uncorrelated
with anything under test. The **24 %** `present_hz` gap is larger than any
device effect yet measured against this probe, so a sweep that mixes clusters
cannot see a real 17 % change and will credit the cluster to whichever arm drew
it. Within a cluster the counters reproduce to a fraction of a percent, which is
what makes three boots per arm enough.

So the page no longer hints. `.band` carries `backface-visibility: hidden` and
`tick()` writes a `translate3d`, which together take the decision out of the
compositor's hands. **Both halves are load-bearing**: the CSS rule cannot carry
the 3D transform, because `tick()` overwrites the inline `transform` every
frame, and a CSS-only fix would be silently discarded.

**It did not work, and the first three boots said it had.** The first three
boots after the change read 424.8, 427.7 and 427.6 draws per presented frame,
all promoted, `present_hz` spanning 1.5 %. The seventh collapsed: **265.8**.
Running tally since the change is 6 promoted, 1 collapsed — about 14 %, against
2 in 10 before. There is no evidence the change reduced anything.

The promotion edit stays. It is correct, costs nothing, and rules out one real
possibility. But the split is still here, so:

**Classify every boot before comparing two.** `drain_duty draws` over
`window_publish fresh`; the clusters are ~420 and ~268 with nothing in between,
so no threshold tuning is needed. Expect to discard roughly one boot in seven,
and plan a sweep with that headroom rather than assuming three boots per arm
will all land together.

The methodological lesson is worth more than the fix: a three-boot green run
against a one-in-five failure is not evidence — it comes up about half the time.
Write the probability down beside the reading, not the verdict.

Boots taken before this change are not comparable to boots taken after it.

## What it does not do

No host input lands inside the measured window — the page animates itself — so
nothing in the capture is the probe's own cost. It also cannot report a verdict
the way a drag probe can: there is no "the window never moved" check, because
there is no host-driven motion to check. Confirm the page is live from the
screenshot the surrounding harness takes, and from `present_hz` being nowhere
near zero.
