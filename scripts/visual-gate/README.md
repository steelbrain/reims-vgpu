# visual-gate

**A performance commit does not land until this passes on a live boot of the
pathway it touches, and the commit body says which pathway and quotes the
verdict line.**

```sh
scripts/visual-gate/visual-gate.sh                    # full, for phase sign-off
scripts/visual-gate/visual-gate.sh --quick            # fewer trials, for iteration
scripts/visual-gate/visual-gate.sh --keep /tmp/vg     # keep every probe's frames
scripts/visual-gate/self-test.sh                      # the parser's own tests, no boot
```

Exits `0` when every probe passed and every silent-loss counter read zero, `1`
on any regression, `2` on a setup failure — a guest that never settled, a probe
that could not run, or a QEMU that died under it.

## Why it exists

A branch of 59 commits was once reset off wholesale because it had introduced
graphical glitches. Every one of the 21 code-changing commits on it had been
verified — against clippy, unit tests, the feature matrix, and device-side
performance counters. The work was performance work, so the instruments it was
checked with were performance instruments, and a rendering regression was
invisible **by construction**. Nothing lied; the right question was never asked.

Three probes existed the whole time that ask exactly that question. None of them
gated anything. This is the entry point that makes them gate.

## What it runs

Each probe is a two-observation instrument: the guest declares what it believes
it drew and where, and the host measures its own capture at exactly those
places. That is what separates *the guest declined to draw it* from *this device
lost it* — the distinction a screenshot cannot make, and the reason staring at
images never resolved it.

| probe | what the guest declares | trials, full | trials, `--quick` |
|---|---|---|---|
| `web-content-probe` | a palette and the screen rects that carry it | 20 captures | 5 |
| `wallpaper-probe` | a 64-bar aperiodic barcode wallpaper the probe itself supplied | 10 trials | 3 |
| `modal-button-probe` | the accessibility hierarchy's buttons and their rects | 10 trials | 3 |

All three run even when an earlier one fails. Which probe fails is the
diagnostic, and a gate that stops at the first one throws that away.

Read each probe's own README before quoting its result. In particular
`modal-button-probe` summons the **log-out** modal, not the Control-Power one a
bug report may name, and `web-content-probe` has twice produced a worthless
result — once from a stressor clipped away by `contain: strict`, once from a
macOS sheet dimming every region to exactly half.

## The counter budget

Probes see what they drive. The other half of a silent loss is the device saying
so in the fail log and nobody reading it, so the gate marks the log by byte
offset before the first probe and applies eight classes over its own window only —
never the accumulated log, which spans builds.

| counter | meaning |
|---|---|
| `deferred_flush_lost` | a guest render this device dropped. Always a real loss |
| `gw_audit_unsound` | the gather witness refuted itself; a stale image is being served |
| `render_flush_over_guest_write` | documented as expected-never; if it fires the writeback ordering repair has broken |
| `mapping_page_drift` | the page list changed under an armed window |
| `THRASH present_action_starvation` | zero across the whole accumulated log to date; a first occurrence is a real change |
| `tdc_overflow` | census target map overflowed and re-seeded — only meaningful when the census probe is on |

Each class is held to the budget `baseline.tsv` records for it, and every class
that fired is named in the verdict line whether or not it was inside budget — so
a `PASS` reading `deferred_flush_lost=2/4` is never mistaken for nothing lost.
Two of these are standing alarms that read zero across the whole accumulated
log; a zero is the working state, so do not delete them for being constant.

**Zero is the default, and six of the eight keep it. Two do not, and the first
full run on unmodified HEAD is what said so.** The plan that specified this gate
asserted all six read zero. `deferred_flush_lost` and `mapping_page_drift` read
1 and then 2 over two consecutive full windows of one x86/PCI boot, five over
the boot as a whole. Every event is the same pair — the guest re-points a
24-pixel-tall strip (1877x24 at 1920 wide: the menu bar) with no packet saying
so, the cached page list disagrees with a fresh walk, and the render is dropped
rather than landed at stale GPAs. Refusing that write is correct; not re-landing
it at the new translation is an open defect, and it predates this gate.

A non-zero budget is an admission about the device, not a setting. It does not
belong in that file without a measurement and a reason beside it, and it is
never raised to make a run pass — only when a boot of the *unmodified* tree
exceeds it, saying in the commit which boot. A class with no row at all is
budgeted zero, so a forgotten row fails strict rather than leaving the class
unwatched.

`counter-budget.sh` does the parsing and `self-test.sh` tests it against
synthetic log text with no guest, no QEMU and no GPU. That split is deliberate:
a parser that matches nothing prints the same eight zeros a clean run does, so
without those cases a green gate would mean nothing and there would be no way to
tell from its output. The cases are mutation-checked — dropping the `^` anchor
on the line families, or matching route fields by prefix instead of exactly,
each fails exactly one case.

## What it refuses to do

**It will not start on an unsettled guest.** sshd answers well before the
desktop composites, and a probe started in that window measures the guest's own
startup work. One such run read 12.2 fps with the device idle at `duty=0.001`.
The gate polls `ssh macos-vm uptime` for a 1-minute load average under
`SETTLE_LOAD` (1.0) for up to `SETTLE_TIMEOUT` (420 s) and exits 2 rather than
guess. It also aborts if QEMU stops matching, so a boot that died early is never
mistaken for a clean run.

## Two things this is not

**Not a frame-rate check.** Frame rate is bimodal here — ~59 and ~118 with
nothing between, both on one unchanged binary within one boot — so a rAF figure
cannot support a claim in either direction. This gate answers *correct or not*.
The device-side counters answer *fast or not*.

**Not proof of absence.** The probes bound the classes they drive. A green gate
means "no regression this instrument can see", and a green `--quick` is not a
phase sign-off. For a change whose failure mode needs a second frame to appear —
a stale tile the next frame declines to rewrite, say — run the full gate twice
on one boot.

If a change cannot be gated because it only manifests under a workload no probe
drives, say so explicitly in the commit body under "Not verified". An honest
admission is what the reset branch's suspect commit already did; the failure was
that nobody treated the admission as blocking.
