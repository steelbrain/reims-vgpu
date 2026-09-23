# T009: Vulkan memory accounting and latency investigation

## Goal

Explain the VRAM growth and frame-time instability seen in the macOS guest, then
make evidence-backed improvements without regressing the already improved image
stability. Keep memory accounting and latency diagnosis as related but separate
tracks: high retained memory is not automatically a leak, and low FPS is not
automatically caused by VRAM pressure.

Do not add a global VRAM cap, change slab eviction policy, or tune rendering
behavior until measurements identify a specific cause and a test can validate
the change. Preserve guest disks and do not merge upstream changes as part of
this task.

## Existing instrumentation

The working tree already contains instrumentation for:

- `vk_memory_live`: successful Vulkan allocations and matching frees, grouped
  by allocation site, heap, memory type, and property flags; includes
  `unmatched_frees`.
- `vk_memory_budget`: Vulkan driver's heap size, dynamic budget, usage, and
  availability.
- `image_pool_levels` and `buffer_slab_levels`: image populations and slab
  backing versus carved bytes.
- `drain_duty`, `host_window_loop`, and `engine_delta`: worker duty, draw/present
  cadence, fresh versus stale frames, rendering work, and selected timing
  counters.
- Host-side NVML: total device memory/utilization/temperature; this is system
  wide and must not be confused with the Reims process ledger.

See [`../vulkan-memory-map.md`](../vulkan-memory-map.md) for source ownership,
counter semantics, and known instrumentation boundaries. Verify all direct
allocation/free paths when new allocator paths are introduced.

## Current evidence and caveats

The idle RTX 3050 baseline was approximately 503 MiB process device-local,
536 MiB Vulkan heap-0 driver usage, 526 MiB NVML, and 400 MiB image-slab backing.
An exploratory browser sequence raised local usage to about 1,160 MiB and slab
backing to 1,056 MiB; the ledger reported zero unmatched frees. The final idle
sample remained at that level for about 90 seconds. This is evidence of
retention, not proof of a leak or of unnecessary duplication.

The exploratory sequence is **not a valid per-stage A/B**: `open -a Safari URL`
created new tabs, and navigating only the selected tab to `about:blank` left
prior probe tabs active in the background. The browser probes were later closed
explicitly. Reuse one Safari tab in future runs, verify the exact URL before and
after every phase, and do not label a white screenshot as idle unless all load
pages are confirmed stopped.

Exploratory `16x6` and `24x6` canvas loads showed roughly 53–59 fresh host-window
draws/s, worker duty around 0.58–0.76, draw time reaching roughly 678 ms per
one-second sample, and IRQ wait around 183–256 ms/s. Flush time was generally
small. Images showed no visible artifacts and QMP remained running. These
counters suggest where to investigate, but they do not identify a bottleneck;
host-window draws/s are not an exact measurement of Safari FPS.

The first corrected single-tab moderate sample (`8x6`, `tex=512`) reused the
same Safari tab for the probe and then `about:blank`. During ~30 seconds of the
probe, the ledger held steady at 1,552 MiB device-local / 1,448 MiB of slab
backing, driver heap-0 usage 1,585 MiB, and NVML 1,643 MiB; no unmatched frees
were reported. `drain_duty` was ~0.62–0.64, `draw_us` ~565–583 ms/s, IRQ wait
~186–224 ms/s, and the host window recorded 51–58 fresh draws/s. After
navigating that same tab to `about:blank`, memory and slab levels did not change
over the next 30 seconds and duty fell to ~0.003. The guest image stayed clean
and QMP remained `running`.

This run did not have a true immediate pre-navigation idle ledger sample: the
first reading was captured shortly after opening the page. It therefore shows
stability during the moderate phase and no short-term trim on stop, but cannot
attribute the 1,552 MiB level to that phase alone. Compare future runs against
a stable idle sample captured before navigation.

### Controlled same-tab moderate run (23 Sep 2026, host 17:38–17:47)

The probe was run in the already-open Safari tab (`layers=8&boxes=6&tex=512`),
after a true 60-second idle baseline, then stopped by navigating that same tab
to `about:blank`. During the first idle interval a single 8 MiB slab allocation
occurred; the ledger then stabilized before the load phase. The comparable
baseline, all three load samples (10/30/60 s), and the 30/60/90 s recovery
samples stayed at 1,560 MiB device-local, 1,456 MiB image-slab backing, 228 MiB
carved from that image slab, 98/83 MiB upload-slab held/carved, 1,593 MiB
driver heap-0 usage, and 1,651 MiB NVML. No unmatched frees were reported.
During load, host-window fresh draws were 53–55/s, `drain_duty` 0.599–0.631,
`draw_us` 541–565 ms/s, and IRQ wait about 222–224 ms/s; flush remained
negligible. After stopping, duty returned to ~0.002 and fresh draws to zero.
QMP stayed `running`, the guest clock advanced, and screenshots through 90 s
recovery showed no black frame or visible artifact.

This controlled run shows no measurable memory change from the moderate phase
and no short-term recovery in allocated slab backing. However, held minus
carved leaves about 1,228 MiB free *inside slabs*; that is not equivalent to
1,228 MiB of fully empty blocks that the allocator can return. Existing logs
do not expose empty-block counts or free-space distribution by slab class, so
they cannot yet tell whether the capacity is a useful live/cache working set or
fragmentation pinned by a small number of live images. The current `slab_live`
value is a live-image count, not bytes.

To resolve this missing distinction, the working tree now adds a low-cost
`image_slab_inventory` diagnostic, grouped by small, large, dedicated, and
poisoned blocks. Each group reports block count, held/carved/free MiB, empty
block count/bytes, and free-range count. A focused unit test validates the
classification and accounting. This is diagnostic only; it changes no
allocation or reclaim policy. It requires a rebuilt host runtime to collect
real values.

## Test protocol

1. Start from a recorded idle state. Record the QEMU command/build identity,
   guest OS, backend, display/window path, environment flags, GPU identity, and
   whether any VRAM limit or slab-reclaim patch is enabled.
2. Confirm SSH readiness. Use one Safari tab and the local deterministic probe;
   verify the requested URL and `hidden`/focus state. Keep the VM window visible
   and windowed (not fullscreen) for screenshot-based freeze/glitch detection.
3. For each independent phase, collect a pre-sample, then synchronized samples
   at 1-second intervals during load and after returning that same tab to
   `about:blank`. Save screenshots at baseline, early load, peak/steady load,
   and recovery. Record guest clock progression, QMP status, and serial/panic
   evidence.
4. Capture at least `vk_memory_live`, `vk_memory_budget`,
   `image_pool_levels`, `buffer_slab_levels`, `drain_duty`, `host_window_loop`,
   relevant `engine_delta` counters, plus NVML and host CPU/QEMU utilization.
   Preserve raw output and exact probe parameters.
5. Use a ladder of light → moderate → heavy → one extra-heavy level, with idle
   recovery between stages. Increase only while the guest image remains valid,
   the guest clock advances, QMP is running, and frames remain fresh. Stop the
   load and preserve evidence immediately on a black/stale window, panic,
   device loss, or severe host/guest instability.
6. Repeat a candidate fix against the same baseline and workload. Change one
   factor at a time. Report medians and tail frame times, not only a single FPS
   sample.

## Questions to answer

- Which allocation sites/categories account for the growth in device-local
  bytes? How much is image backing, buffer/staging, guest import, and driver
  overhead?
- Do frees balance allocations? Which Vulkan objects remain live after the
  browser has stopped, and which slabs are empty, carved, pinned, or awaiting
  GPU completion?
- Does the observed memory plateau represent useful reusable cache, resources
  still referenced by guest surfaces, delayed fence retirement, or an
  avoidable duplicate/copy?
- At the point frame times worsen, are `draw_us`, IRQ/fence waits, flush,
  scheduler gaps, CPU use, or GPU utilization the dominant correlated cost?
- Does the same workload recover after it is truly stopped? Does memory, fresh
  present cadence, and frame-time distribution return to baseline, and on what
  timescale?

## Acceptance criteria

- Every successful engine-owned `VkDeviceMemory` allocation is attributed and
  has a matching free, or is explicitly documented as intentionally live.
- Memory totals reconcile within documented driver/allocator overhead; slab
  backing, carved occupancy, and live resource bytes are not conflated.
- A reproducible test distinguishes live resources from retained empty slab
  capacity and correlates memory with frame-time/present degradation.
- Any optimization has a focused test where possible and a repeatable runtime
  A/B showing its effect on memory, FPS/frame-time tails, and visual integrity.
- No global memory cap or reclaim-policy change is accepted solely because
  NVML usage is high; avoid trading correct rendering for an apparently better
  FPS score.

## Next action

Rebuild and run the new slab-inventory diagnostic, then repeat this same-tab
idle → moderate 60 s → recovery 90 s sequence with browser frame-time/FPS
capture. Correlate empty-block bytes versus free bytes pinned in partially
carved blocks with the sampled cache population and presentation latency.
Only then decide whether to change pool lifetime/reclaim behavior or profile a
specific high-cost rendering interval. Preserve the current user changes and
do not use `orca-ide`.
