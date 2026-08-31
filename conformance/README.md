# conformance

A self-verifying Metal battery that runs the same source on a native macOS host
and inside the guest, and names the seam when the two disagree.

Every result this project had about Maps' missing type layer was scored by
opening a screenshot, and a screenshot names no seam: "labels absent" is what a
wrong pitch, a wrong swizzle, a dropped dispatch and a lost mip level all look
like. Each case here computes a value the CPU can predict exactly, asks the GPU
for it, and compares — so a failure names the case, the bytes wanted and the
bytes returned.

```
CASE <name> PASS|FAIL|SKIP <detail>
SUMMARY cases=N failures=N skipped=N
```

## Reading a result

`verdict.py` applies the table below to a pair of runs and exits non-zero when
something in it is unexplained. `run-guest.sh` calls it, so a sweep reads a
verdict rather than diffing hundreds of lines by eye:

```sh
conformance/verdict.py --native baselines/native-apple-m4-macos15.txt \
  --guest /tmp/conf-out/conformance.txt \
  --translation-errors expectations/macos-13/translation-errors.txt \
  --driver-errors expectations/macos-13/driver-errors.txt
```

It is stricter than the eye in two directions the eye is bad at. A case that
**stopped being reported** reads as nothing at all in a summary line -- the
totals simply shrink -- and it says `NOT-RUN`. A classified case that now
**passes** is a failure too: an inventory nobody prunes stops being a list of
defects and becomes a list of cases nobody looks at.

The comparison between the two hosts is the whole instrument:

| native | guest | meaning |
|---|---|---|
| PASS | PASS | nothing to see |
| PASS | FAIL | a named device defect |
| FAIL | — | a wrong expectation in the suite, not a finding |
| — | SKIP | the device's own reported limits make the case inexpressible |

`baselines/` holds the native runs, one file per oracle host. Re-record whenever
a case is added, and never treat a guest failure as a finding until the same case
is green natively.

## Failure ownership is part of the verdict

`expectations/<rail>/translation-errors.txt` and
`expectations/<rail>/driver-errors.txt` are separate inventories. One case name
goes on each line; `#` starts the required diagnosis. A name in both files is
`DUPLICATE-CLASSIFICATION` and fails scoring.

**They are per rail, and a rail is a driver.** `macos-13` and `macos-15` are two
different guest drivers running the same battery, so a case one of them fails
says nothing about the other, and neither does a case one of them passes. An
entry established on one rail may not be copied to another to make a sweep
green; it has to be re-established there, against that rail's own control. A
rail with no inventory directory is refused by the runner rather than scored
against a neighbour's debt.

A translation entry names its gitignored `bugs/` handoff package, containing the
AIR reproducer and translator evidence. A driver entry names the violated Metal
relation and the owning device rail. An unexplained mismatch belongs in neither
inventory until the owner is proved. Both files are checked in both directions:
a listed case that starts passing is `FIXED-TRANSLATION` or `FIXED-DRIVER`.

It is not a place to put a case that fails on the oracle too. That is
`SUITE-BUG`, the expectation is wrong, and listing it hides a bug in the battery
rather than one in the device.

## The oracle says right; only the control says backwards

`verdict.py` asks the native oracle whether a guest result is *correct*.
`ratchet.py` asks an identified control run whether a candidate moved
*backwards*. They are different questions and a change has to answer both.

```sh
conformance/ratchet.py \
  --control  runs/control/conformance.txt  --control-device  runs/control/device.log \
  --candidate runs/cand/conformance.txt    --candidate-device runs/cand/device.log
```

The distinction is load-bearing on a rail whose debt is not yet classified. Such
a rail reads to the oracle scorer as scores of unexplained failures, and will
until every one has an established owner. That is honest, and it also means the
oracle scorer is red for *every* candidate on that rail and so cannot tell one
that broke something from one that broke nothing. The control can: it already
contains the debt, so a candidate reproducing it exactly has preserved the
device's behavior whether or not anyone has yet written down whose fault it is.

Classifying a failure and detecting a regression are therefore separate jobs,
and only the second one gates a commit.

**Give it more than one control.** The cases on this rail are reproducible and
the device log is not: two back-to-back macos-13 controls of the same build
agreed on all 290 case results exactly and still disagreed on their typed-reason
counts, `stamp_wait_timeout` among them. `--control` may be repeated, and a
repeated control measures the rail's variance instead of assuming it away -- a
candidate inside the envelope has not moved. With one control the tool says so
in its totals rather than leaving a reader to infer it. A case the controls
themselves disagree about is `CONTROL-UNSTABLE` and is dropped from scoring,
because charging a candidate for the rail's own noise is the same error in the
other direction.

It scores the workflow's transition table, and two things no case comparison
can see. A name in only one of the two runs is a regression rather than a
missing row -- coverage that moved is the failure this catches. And the device's
own typed reasons are counted on both sides, so a candidate that keeps every
pixel and doubles the fence timeouts is red on that alone.

One reading it hands back rather than decides: a case that fails in both runs
with a *different* detail is `CHANGED-DETAIL` and does not fail the run, because
nothing in a `CASE` line separates the typed reason from its payload. The same
defect described differently and a different defect wearing the same name look
identical here, and only a reader who knows the case can say which it is.

## A case name must mean the same thing on both hosts

`minimumLinearTextureAlignment` is 16 on an M-series device and 256 on Apple's
paravirtual one, so a pitch derived from it lands on a different integer per
host. A case named after that integer -- `linear_a8Unorm_54x16_pitch80` on one
host, `pitch512` on the other -- has no counterpart on the other side to be
scored against, and the whole native/guest instrument is gone for it. Sixty-two
of 216 cases were in that state, silently: they ran everywhere, tested the same
thing everywhere, and could not be paired.

So the label carries the **derivation** and the detail carries the number.
`Pitch` is `.tight`, `.padded(rows:)` or `.exact()`, and it computes the byte
count as well as naming it, which also took the alignment arithmetic out of the
call sites where it was written by hand five times. A case whose *width* is
derived gets the same treatment -- `offsetOracleCase(256, …)` and `(250, …)`
mean "tight" and "six bytes of padding" on a device aligning to 16 and on one
aligning to 256 alike, which `align` and `align - 6` did not.

When adding a parameterized case, ask what its name would be on the other host.

A `SKIP` is not a soft failure. `minimumLinearTextureAlignment` is 16 on an
M-series device and 256 on Apple's paravirtual one, so a literal pitch from a
guest census is not expressible on every host — Metal itself rejects the
descriptor. The battery also derives padded pitches from whatever the running
device reports, so every host runs padded-pitch cases whatever its limit.

## Where things live

```
conformance/
  suite/
    main.swift        every case invocation, and nothing else
    Harness.swift     report/counters, the device and queue
    Shaders.swift     the one Metal source string
    Support.swift     the library, pipelines, readback, helpers
    cases/*.swift     the case bodies, one file per rail or family
  baselines/          native runs, one per oracle host
  expectations/<rail>/  separate translation and driver inventories, per rail
  verdict.py          scores a guest run against the native one
  ratchet.py          scores a candidate guest run against a control guest run
  run-native.sh       build and run on the oracle
  run-guest.sh        boot a rail and run the same source in the guest
  build/              binaries; gitignored
```

**Swift permits top-level statements in `main.swift` alone**, which is what makes
the split worth having: a case body physically cannot invoke itself, so the
invocation list is one file a reader can count against the parameter grid. The
"a case nobody called reports nothing" trap below is closed by construction.
`dev`, `queue` and `library` are globals initialised by a closure for the same
reason — they were top-level `guard`s and a top-level `do`/`catch` when this was
one file, and neither form is legal outside `main.swift`.

Declaration order across files does not matter in Swift, so a case file may use
anything in `Support.swift` and vice versa. Adding a case is: a function in the
right `cases/` file, and a call in `main.swift`.

## Running it

Native, on the oracle — this also cross-builds the x86_64 fallback:

```sh
conformance/run-native.sh
```

`ORACLE=<ssh-host>` names the machine; it needs Xcode's command line tools and
nothing else.

In the guest — this boots a rail, runs the battery and collects the device's own
fail log beside the results:

```sh
conformance/run-guest.sh /tmp/conf-out
```

`RAIL` selects the guest rail and defaults to `macos-13`. Environment passes
through, so an ablation arm is
`REIMS_VGPU_GUEST_IMPORT=off conformance/run-guest.sh /tmp/conf-off`.

For a fast, shaderless gate of the integer-clear contract, use
`CONFORMANCE_MODE=integer-clear`. Use `CONFORMANCE_MODE=topology` for the
point, line, line-strip, and triangle raster contract, or
`CONFORMANCE_MODE=float-sampling` for the `RGBA32Float` compute/vertex sampling
contract. Use `CONFORMANCE_MODE=indexed-draw` for UInt16/UInt32 index offsets,
signed base vertex, and base instance. The default is the
complete battery. Focused guest runs require `NATIVE` to name the matching
focused oracle output; this prevents a partial run from being scored against
the full baseline. Every run writes `manifest.txt`, hashes the live QEMU through
`/proc`, hashes the linked static library and guest test binary, and preserves
the serial and fail logs. A stamp timeout, device loss, or guest panic fails the
runner even when all classified pixel comparisons are expected. Focused probes
are bounded at 180 seconds by default (the full battery at 600);
`CONFORMANCE_TIMEOUT` may narrow that bound for an investigation.

The runner ships `build/conformance-x86_64`, the Mach-O built beside the native
oracle result on an Apple host. This makes the oracle and guest source/build
identity explicit and avoids treating macOS's developer-tools installer shim
as a working compiler. Re-run `run-native.sh` whenever a case is added.

## A refusal is not a mismatch, and the kernel is what says which

This device refuses a dispatch on the *host* side, so the guest's command buffer
completes clean and the output buffer keeps whatever was in it. Left alone,
every case here then compares its sentinel fill against what it wanted and
reports a **content** failure — "the device returned the wrong bytes" for a
device that returned none.

The `offset_oracle` cases showed how far that misleads. Their fill is
`1 + (i % 251)` and zero means "a byte nothing in this buffer ever held", but
`readBack`'s `0xEE` sentinel is 238 — inside the fill's own range. A refused
dispatch inverted to a constant source offset of 237, every texel landed in a
different delta bucket, and four cases reported `absent=0 shifted=4080`: a
precise, plausible account of a defect that did not exist.

No sentinel value fixes this. A battery covering many formats has no byte that
is out of range for all of them. So each readback kernel writes `ran[0] = 1u`
into a buffer of its own, **before** its grid guard — so the witness says the
kernel was reached rather than that some thread was in range — and `readBack`
returns `nil` when that word stays zero. A case that gets `nil` calls `refused`,
which prints one wording for all of them.

Two consequences when adding a case:

- **A new readback kernel must take `device uint *ran [[buffer(4)]]` and write
  it first.** A kernel without the witness reports every refusal as a content
  failure, which is the state this section exists to describe.
- **A case that cannot run because another case did not must `skipDependent`,
  not vanish.** The case *count* is what a reader diffs two runs by, and a name
  that stops appearing reads as a deleted case. One refusal of
  `incremental_a8_first_read` silently took three other names out of the totals
  before this was added.

## Reading it beside the device log

`run-guest.sh` copies `/tmp/reims-vgpu-fail.log` to `device.log` in the
output directory. A case that fails with a refusal in that log is an
unimplemented rail refusing by name; a case that fails with *nothing* in the log
is silent loss, which is the worse of the two and the one worth chasing first.

## A `.private` render target does not reach the target-import rail

Every render target in sections A-H is `rd.storageMode = .private`, which means
the device allocates it and the guest never names the pages behind it. That is
half the render targets a compositing app uses. The other half are `.shared`:
layers the CPU also rasterizes into, or that another process composites, whose
bytes are guest memory the device may bind a Vulkan image *directly over*
instead of copying.

Those are two rails, and for a long time this battery had exactly one case on
the second of them — `cpu_write_after_render`, section F5. A whole rail behind
one case is not coverage. It is a single sample that happens to pass, and it is
what let a live defect sit underneath 173 green cases: a driven Maps boot lost
its entire type layer on the arm where guest-backed targets are imported, and
nothing in this file could see it, because nothing in this file created a
guest-backed target except that one case.

Section I is that coverage. When adding a render-target case, ask which rail it
is on and say so in the case name — `srt_` is the guest-backed prefix. A new
case that reaches for `.private` because that is what the case above it used has
tested the rail that already had coverage.

The widths there (60, 256, 1000) are not decoration. 60 and 1000 texels are 240
and 4000 bytes, neither a multiple of the 256-byte linear alignment this device
reports, so a rail that confuses the guest's stride with a padded one has
somewhere to show it. A case list of only round widths cannot see a shear.

## A case that is never called reports nothing, and the totals do not notice

`cpu_write_after_render_256x64` was absent from every run of this battery, on
both arms, because the invocation list called three of its four arms. Nothing
flagged it: `ran` counts cases that reported, so a case that was never invoked
is indistinguishable from one that does not exist.

That is the same failure mode as the `refused(); return` bug that `skipDependent`
exists to fix, one level up — there the case vanished from the totals mid-run,
here it never entered them. When adding a parameterized case, count the
invocations against the parameter grid before trusting a green summary.


## A case may claim the device rail it was written to move

`claims(label, "sl_gpu_landed")` in a case body prints a `ROUTE`
line, and `verdict.py` reads those against the device's own fail log. A case
that passes without its claimed counter ever moving is `NOT-COVERED`, which is
the worst reading this battery produces: it is indistinguishable from coverage,
and it is what three attempts at the case below did on the *broken* build.

Two limits, both deliberate:

- **The check is per run, not per case.** Nothing carries a case name across
  into the device, so the question it answers is "did anything in this run reach
  that rail". That is the question that was being got wrong.
- **A claim nobody watched is worse than no claim.** Claiming a counter because
  the name sounds right makes the table read as coverage while measuring
  spelling. Watch the counter move, then claim it. An unclaimed case is
  unclaimed, not covered.

A narrowing `REIMS_VGPU_*` switch can legitimately close a claimed rail, so
`run-guest.sh` passes `--claims-advisory` when any is set.

## A race is only a test if the broken arm loses it

`srt_blit_pipelined_*` is the regression gate for a host read of guest pages
that was not ordered against this device's own submitted GPU writes. Getting it
to reproduce took six guest boots and three wrong shapes, and each wrong shape
failed in a way that read as a pass.

Three things it turns out to need, none of which is negotiable:

- **The copy has to be one the guest driver actually emits as a whole-surface
  texture-to-texture copy.** A `.bgra8Unorm` pair at 512x512 is not: the driver
  stages it through a buffer, and the case then exercises a different device rail
  and passes. The shape that works was *measured* off a driven compositor with a
  temporary device probe rather than chosen — a linear source, an IOSurface
  destination, `BGRA8Unorm_sRGB`, one level and one slice, at window and screen
  size.
- **The render and the copy have to be in separate command buffers.** In one
  command buffer the driver resolves the read-after-write hazard itself and
  stages the copy, which again lands on another rail.
- **One frame in flight is not enough.** The single-shot `srt_blit_after_render_*`
  cases reach the rail — confirmed by probe — and still pass on the broken arm,
  because one render finishes before a host reader can decode a copy and memcpy
  three megabytes. Only the pipelined form fails: eight frames, each rendering
  and copying without waiting, so a copy is serviced while earlier renders are
  still executing. That is what a compositor does and it is why the defect was
  a compositor defect.

The single-shot cases are kept even though they gate nothing. They record the
shape that reaches the rail without losing the race, which is the thing that
made three earlier attempts look like they had disproved a real bug.

**Distinct colours per frame are load-bearing.** Each frame renders its own
colour into its own source and copies to its own destination, so a stale read is
identifiable as *which* earlier state was read rather than merely "wrong". The
report separates the two: a previous frame's colour names a stale copy, and zero
names a copy that ran before anything was written at all.
