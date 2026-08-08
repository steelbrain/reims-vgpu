# runtime-dead

Which reims-vgpu functions compile, link, are reachable, and the guest protocol
still never takes.

`scripts/dead-state` answers a different question — what nothing *references* —
and its intersection report has been empty for a while. AGENTS.md names the
remaining lever as runtime-dead code, and says `/tmp/reims-vgpu-fail.log` was the
only ground truth for it. The log is a poor instrument for this: every decoder in
`decode/` is silent on success, so absence of a line proves nothing, and the only
usable never-fired signal was the 85-name `store_routes` counter set. That set
has been audited to exhaustion.

This measures the whole crate instead, at function granularity, from one boot.

## Running it

```sh
scripts/runtime-dead/runtime-dead.sh              # ~10 min
scripts/runtime-dead/runtime-dead.sh --seconds 40 --app Safari
scripts/runtime-dead/runtime-dead.sh --import-off # the copying rails; see below
```

Needs `llvm-profdata`, `llvm-cov`, and a `libclang_rt.profile-x86_64.a` whose
LLVM major matches `rustc --version --verbose`. Outputs land in
`/tmp/reims-vgpu-runtime-dead/` — or, under `--import-off`, in
`/tmp/reims-vgpu-runtime-dead-import-off/`, because the two reports exist to be
diffed and a run that overwrote the other would destroy the comparison:

| file | what |
|---|---|
| `by-file.txt` | per-file region/function/line coverage |
| `never-ran.txt` | every function whose counter stayed at zero, tab-separated `path<TAB>mangled` |
| `drive.log` | the drag probe's verdict, so you can see the boot was actually driven |

## How it works, and the three things that bite

**The profile runtime is linked by hand.** Building the staticlib with
`-C instrument-coverage` gets you `__llvm_prf_*` sections and nothing that writes
them: rustc bundles `profiler_builtins` into a *final artifact*, and this
staticlib is not one — QEMU links it. So `hw/display/meson.build` names
compiler-rt's `libclang_rt.profile-x86_64.a` when `REIMS_VGPU_COVERAGE` is set,
`--whole-archive` because no instrumented object references it and the linker
drops every member otherwise.

**`RUSTFLAGS` must be scoped to the host triple.** A bare `RUSTFLAGS` also
reaches the `x86_64-unknown-uefi` option ROM, which has no `profiler_builtins`
for its target; the ROM build fails and the boot never starts. Use
`CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS`.

**QEMU has to exit, not be killed.** The counters are written by an atexit hook,
so SIGKILL loses the run. The script sends SIGTERM and waits. Continuous mode
(`LLVM_PROFILE_FILE=...%c...`) would survive a kill but needs runtime counter
relocation, which this toolchain does not build.

**A 0-byte profile is not how you notice.** This README used to say it was, and
that was wrong in the direction that costs a session. The boot's own QEMU can
write a 0-byte `.profraw` while the run still produces a complete-looking
report, because it is not the only process writing to `$OUT_DIR`: the crate's
build scripts write one each, and the boot script's own short-lived
`qemu-system-x86_64` probe invocations each dump the *full* function table with
every counter at zero — 4.3 MB of records that are all misses. The script's
`-size +0c` filter drops the empty file that mattered, keeps the all-zero ones
that did not, `merge -sparse` discards their zero records, and the report is
then built from whatever a build script happened to touch. The observed result
was six functions of coverage, `TOTAL 0.00 %`, and a `never-ran.txt` naming all
3360 functions in the crate — produced by a boot that reached the desktop and
passed the drag probe's did-the-window-move verdict.

Nothing about that output says it is broken. It is a kill list for the entire
device, in the format a reader is invited to act on. So there are two guards,
and both refuse to write rather than warn:

- **The boot's own raw must exist and be non-empty**, checked by pid before the
  merge. This names the real failure exactly, because the script already knows
  which process it SIGTERMed.
- **Not every function of ours may read zero**, checked before `never-ran.txt`
  is opened. This is the backstop for the case where the pid check passes on a
  raw that is present but all misses.

Do not gate on `llvm-profdata show`'s `Instrumentation level:`. It reports
`Front-end` on this toolchain even for the genuinely-instrumented dumps, so a
check for `IR` refuses good runs.

Also note the boot script re-runs `qemu-build` itself, so the instrumentation
env has to be exported for the *boot*, not just for a build beforehand.
Otherwise the boot quietly relinks a clean QEMU over your instrumented one.

## Reading it: a zero is a question, not a verdict

This is the part that matters. The measurement is cheap and the conclusion is
not. Every reduction sweep in this repo that went looking for "code nothing
runs" came back with contract fidelity, and this instrument finds *more* of that
class, not less.

`runtime/icb/mod.rs` reads 0.00% — all 22 functions. That is the correct
reading, and `runtime/icb/` is explicitly not a deletion; `runtime/exec`
prices it and AGENTS.md records five boots behind that call. The instrument
agreeing with a conclusion already reached the expensive way is the reason to
trust it, not a reason to act on it.

Legitimate reasons a function reads zero:

1. **It is a decline, and nothing failed.** `backend/vulkan/engine/draw_preparation.rs`
   is 0% across all 9 functions because no draw ever failed preparation. A
   never-firing decline is the healthy state; deleting it is deleting the alarm.
2. **It is a real Apple opcode this workload does not issue.** An audit of all 85
   `store_routes` names found 39 that never fire and none reducible. Deleting a
   decoded-but-untaken arm loses guest work silently the first time a guest takes
   it.
3. **The workload was one 25-second Safari drag.** Window compositing is not
   resize, is not video, is not a mode change, is not sleep/wake, is not a second
   display. Cold here means "this run did not ask", full stop.
4. **It is an error or allocation-failure path.**
5. **It belongs to the other pathway.** This run is x86 / 12-bit page shift /
   Vulkan. Most of `backend/metal/` is `cfg`-ed out of the build and does not
   appear at all — but page-geometry and attach-specific paths do, and read cold.
6. **It is Metal-arm logic that was deliberately moved out of the gated tree,
   so it is compiled in and still only called from the other arm.** This is
   reason 5's exception and it did not exist when the first baseline was taken.
   `AGENTS.md` asks for pure logic under `backend/metal/` to be moved out, so
   its tests run on every arm instead of on none; `backend/hash.rs`,
   `backend/render_pso_key.rs` and `model/content_cache.rs` were moved for
   exactly that reason and all three read 0.00% here. Reason 5 as it used to be
   written — "`backend-metal` is cfg'd out entirely, so it does not appear at
   all" — turns that into an argument that they *cannot* be Metal-only, and so
   into a deletion. Read the module's own doc: each of these says in its first
   paragraph why it sits outside `metal`.
7. **It is a capability-gated fallback and this host is too capable to need
   it.** `runtime/spirv_vertex_input.rs` is 0% across all 10 functions because
   it is reached only from `VertexFormatSupport::resolve`'s widening fallback,
   which runs when a device declines a three-component vertex format. This
   RTX 5080 declines none. The same shape covers the copying guest-memory
   rails, which is why `REIMS_VGPU_GUEST_IMPORT=off` exists — a rail that is
   the *only* rail on a less capable host reads identically to a dead one here.

The test to apply before deleting: **name the guest action that would take this
path.** If you can name it, the path is contract fidelity and stays. If you
cannot, you have a candidate — and a candidate still needs the ordinary
justification, because "I could not think of one" is not a measurement.

One artifact to know about: generics and closures get one entry per
instantiation, so a mangled name reading zero may be one monomorphization of a
function that ran under another. Check the demangled name against
`by-file.txt`'s per-file numbers before concluding a whole function is cold.

## It was never the profile: the script could not name its own QEMU

Three runs refused to write a report at the guard `the boot's own profile is
missing or empty`, and the guard was wrong every time. The last of those runs
left `reims-996479.profraw` on disk, 4 310 352 bytes, `Maximum function count:
225055584`, `Total count: 4331733340` — a complete measurement of the boot it
had just declared unmeasured.

The cause is the pid, not the profile. The script found QEMU with

```sh
ps -eo pid,args | grep '[q]emu-system-x86_64' | awk '{print $1}' | head -1
```

which matches any process whose **argv** contains that string, not any process
that **is** it. Three things in this script's own critical path qualify, and all
three fall inside the loop's two-second polling window:

- the boot script re-runs `qemu-build`, whose ninja link step spawns
  `cc ... -o .../qemu-system-x86_64 ...` — a link of a 119 MB binary is not
  brief;
- `qemu-build` then makes four short-lived `qemu-system-x86_64 -device help`
  probes;
- anything else on the machine mentioning the name — a shell running a command
  that contains it matches too, which is how this was finally caught.

`head -1` returns whichever came first, and none of those writes a profile under
the pid the script then went looking for. So `$OUT_DIR/reims-$qemu_pid.profraw`
did not exist, and the guard reported the boot's own profile missing while it sat
in the same directory under its real name.

Three changes, and the shape of each matters more than the fix:

- **`qemu_pids()` reads `/proc/<pid>/exe`.** An argv match cannot tell a compiler
  writing that name from a process running it; the exe link can. Same helper now
  backs the stale-VM guard, which had the identical flaw.
- **The measurement pid is sampled after the guest answers SSH**, when the build
  is long finished and the probes are gone, and the run refuses unless exactly
  one QEMU is alive. The early loop survives only to notice a boot that never
  started and to have something to kill on the refusal paths.
- **The guard tests counts, not bytes.** A probe's dump is 4 310 352 bytes of
  records that are all misses — exactly as large as the measurement, and it
  passed `-s` every time. `llvm-profdata show | Maximum function count:` reads 0
  for every probe and 225 055 584 for a driven boot, so ask for the property that
  matters rather than for a file that exists.

Two things ruled out earlier remain ruled out, and were never the problem: the
QMP `quit` path (worth keeping — it is the correct shutdown — but the profile is
written under SIGTERM too, which is how the run that proved this was stopped),
and the toolchain. This host is rustc 1.97 / LLVM 22.1.6 with compiler-rt 22 and
`llvm-profdata` 22.1.8, and it writes complete profiles.

## The app has to have a window before the probe starts

SSH answering is not the guest being ready. The boot reverts to a snapshot, so
`sshd` is listening within seconds while the window server is still restoring
sessions and the drive app has no window yet. The probe's first act is to read
that window's frame, so it exited in about a second with

```
window-drag-probe: could not read Safari's window frame (pos '' size '')
```

which is **not** the "window never moved" refusal — it is a failure to start at
all. `|| true` swallowed it, the boot was stopped seconds after reaching the
desktop, and the only thing that reported a problem was the all-zero guard ten
minutes later. The script now waits for the app to present a window before
driving, and treats both that timeout and a probe failure as fatal where they
happen rather than as a report to refuse at the end.

## Baseline

First run, `7a7ffec` + the console-paint verdict, x86 / Vulkan / host-window,
driven with `--seconds 25 --app Safari` (496 draws/s median, 11 fresh, so the
device was compositing rather than idle):

```
crates/reims-vgpu/src   2826 functions   1066 never ran   (62.3% ran)
regions 56.77%   functions 64.78%   lines 56.42%
```

Second run, `006d2216` — the first since the host-pointer import landed, and the
first the pid fix above let through — same pathway, same drive:

```
crates/reims-vgpu/src   3450 functions   1413 never ran   (59.0% ran)
regions 53.65%   functions 61.13%   lines 53.33%
```

The two are not directly comparable: the crate grew by 624 counted functions
between them, and the second boot ran at a tenth of the first's present rate
because it was taken minutes after a full instrumented rebuild. Coverage is a
count and survives that; the drive log's `us=` fields do not, and nothing here
should be read as a timing.

Files at 0.00% function coverage: `runtime/icb/mod.rs`, `runtime/fence_exec.rs`,
`runtime/mipmap.rs`, `runtime/heap_query.rs`, `runtime/plan/event_sync.rs`,
`backend/vulkan/engine/draw_preparation.rs`.

## All six 0.00% files have been adjudicated. None is a deletion.

Do not re-run this. Each was traced upward to the decode dispatch arm that
reaches it, and each named the guest action that would take it:

| file | why it is zero | the guest action that takes it |
|---|---|---|
| `runtime/icb/mod.rs` | already priced, five boots behind it | indirect command buffers, type-7 tag `0x36` |
| `runtime/heap_query.rs` | Vulkan host has no Metal device, so it answers `NoMetalDevice` | `CmdHeapTextureSizeAndAlign`, child-FIFO config op `0x40` — `[MTLDevice heapTextureSizeAndAlignForDescriptor:]` |
| `runtime/mipmap.rs` | a window drag creates no multi-mip textures | blit opcode `0x133 generateMipmaps` |
| `runtime/plan/event_sync.rs` | pure planning behind `fence_exec`; same reason | Metal events and encoder fences, segment type 3 and the blit/compute/render fence fields |
| `runtime/fence_exec.rs` | the sole executor for the above; this workload issued no cross-encoder sync | same |
| `backend/vulkan/engine/draw_preparation.rs` | it is the decline type — no draw failed preparation | nothing; a firing here *is* the bug |

Two of the six (`icb`, `draw_preparation`) are healthy-zero alarms; the other
four are real opcodes this one workload does not issue. That ratio is the point:
the instrument's yield is a map of the contract, not a list of things to cut.

## The three caches the in-place plan reserved for measurement are all live

`.agents/guest-memory-in-place-plan.md` §5 names three modules as "needs
measurement before you touch it — do not delete on inspection", on the theory
that referencing guest RAM in place removes their reason to exist. The second
baseline above is that measurement, and it refuses all three:

| module | regions | functions | verdict |
|---|---|---|---|
| `runtime/gva_view.rs` | 80.35% | 26/28 | live — the highest-covered of the three |
| `runtime/surface_cache/mod.rs` | 72.13% | 49/62 | live |
| `runtime/m2v_cache.rs` | 62.97% | 40/65 | live |

Every one of them is executing on a driven boot *after* the host-pointer import
landed, so the import did not strand them. Their cold halves are worth reading
one day, but the modules are not deletions and the question of whether they are
should not be re-opened without a boot that says otherwise.

One reading next to them is the same shape and is *not* a finding:
`runtime/storage_flush/` is 21–98% across its modules on this host, which is a
discrete RTX 5080. The plan's §5 claim is about a **UMA** host, where the armed
window count should go to zero. This boot cannot measure that, and its non-zero
coverage is the dGPU rail working as designed rather than evidence against the
claim.

## The copying rails, measured: `--import-off`

`--import-off` exports `REIMS_VGPU_GUEST_IMPORT=off` and reports into
`/tmp/reims-vgpu-runtime-dead-import-off`. It is the other half of the ordinary
run, not a variant of it: where the host can import guest RAM every guest window
takes the import, so the copying rails run zero times and the ordinary report
says nothing about them. Reason 7 above is that whole class.

The run refuses unless `vk_caps` reports `host_pointer_import=disabled_by_env`.
That guard is the point of the flag — an override that silently did not take
produces a full report, self-consistent counters, and a directory named
`-import-off` whose coverage is the ordinary run's. Diffed, it would show no
copying rails at all, which reads as "the copying rails are dead".

First paired reading, `cda0f2ff`, both runs x86 / Vulkan / host-window, driven
`--seconds 25 --app Safari`:

```
import on    3450 functions   1413 never ran   regions 53.65%   functions 61.13%
import off   3450 functions   1457 never ran   regions 52.50%   functions 59.72%
```

**Diff at file granularity, not at mangled-name granularity.** The mangled-name
diff says 26 functions ran only with the import off and 70 only with it on, and
both numbers are inflated for the reason the generics note above gives: one
entry per instantiation, so a monomorphization that moved between runs counts as
two changes and a workload that differed slightly counts as many.
`model/lru_memo.rs` is the worked example — 4 of 10 functions executed in *both*
runs, with three mangled names in the "only off" set. Per file, 20 files moved
at all:

| direction | files | what |
|---|---|---|
| import on reaches more | 14 | `runtime/guest_ram.rs` **19 → 2**, `engine/mod.rs` 84 → 71, `engine/exec.rs` 41 → 37, `engine/context.rs` 29 → 26, `engine/host_ram.rs` 4 → 1, `runtime/guest_window_regions.rs` 2 → 0 |
| import off reaches more | 6 | `runtime/objects/mod.rs` +3, `runtime/draw/mod.rs` +3, `runtime/mapping_write/mod.rs` +1, `runtime/storage_flush/land.rs` +1, `runtime/decode/fifo.rs` +1, `engine/pools/mod.rs` +1 |

Two things to take from it. First, `runtime/guest_ram.rs` collapsing from 19
executed functions to 2 is an independent confirmation that the override took —
it does not depend on the `vk_caps` string the guard reads, and it is the shape
you should expect to see. Second, **the copying rails are about ten functions
across six files, not a subsystem.** They are still not a deletion — the plan
this instrument serves is explicit that they are the dGPU rail and the only rail
on a host without the extension, and now there is a boot behind that rather than
an argument. But do not quote the 26.

What is *not* established: the two runs are two different Safari drags, so a
one-function delta on a file that has nothing to do with importing is as likely
to be workload variance as a rail. The deltas worth trusting are the large ones
and the ones whose file is about the import decision. A third run would separate
them and has not been done.

## What is cold in both runs, and why `never-ran.txt` is the wrong file to count

With an import-on and an import-off report on disk, the intersection is
answerable for the first time: **1387 mangled names are cold in both.** Ranking
their files is the closest thing this instrument has to a candidate list, and
the top of it is a good illustration of why it is not one.

| file | cold in both | why |
|---|---|---|
| `runtime/blit_exec/mod.rs` | 89 | 28 of 117 functions ran — a window drag issues few blits (reason 3) |
| `runtime/mapper/mod.rs` | 67 | **arm64-only**, and the file says so at its own resolve rail: entered from `capture_at_producer`, so only an arm64 boot can move it. `AGENTS.md` names this rail as its standing example of a one-pathway rail. 42 of 79 functions still ran here, so it is half live even on x86 |
| `runtime/compute_exec/mod.rs` | 46 | 62 of 108 ran; the cold half is opcodes this workload does not issue |
| `observe/decline.rs` | 40 | the file has **three** functions and missed **two** |

That last row is the warning. `never-ran.txt` is one line per *monomorphization*,
and the generics note above says to check a name against `by-file.txt` before
concluding a function is cold. Here is the size of the effect:

```
by-file.txt     3175 functions   1234 missed
never-ran.txt   1413 lines                     — 1.15x, +179
```

In aggregate that is a 15 % overstatement and the headline number is roughly
honest. **Per file it is not**, and the error does not distribute evenly:
`observe/decline.rs` supplies 40 lines against 2 missed functions, a factor of
20 in one file, because a `Decline` impl is generic and every instantiation
counts separately. So rank files by `by-file.txt`'s missed-function column, and
treat a `never-ran.txt` line as a pointer to look rather than as a fact about a
function.

The same caution retires the 26-versus-70 figure in the `--import-off` section
above, and it is the reason that section diffs at file granularity.

### What a healthy `--import-off` fail channel looks like

Rank the fail channel of that boot and the top entry is alarming and fine. Filter
the `OFF` records first (`grep -v '^OFF '`, per `AGENTS.md` — off-channel lines
carry `reason=` too and invert the queue), and the first paired run gave:

| count | reason | reading |
|---|---|---|
| 30 | `gpuwb_guest_ref_refused` | every one `via=guest_ram_map_no_backend_import`. The GPU writeback declined because no backend published a granularity and the CPU rail ran. `mapping_write`'s own doc predicts this exact line being ranked to the top and says why it restates the inner check |
| 30 | `sync_exec_lock_hold` | 250 ms threshold, hit at 313–378 ms with 1–4 draws. A coverage-instrumented build with every guest window on the copying path is slow; treat as an artifact of this build, not a reading about the device |
| 12 | `sampled_resident_reclaimed` | |
| 8 | `lin_rung_blank_with_host_entry` | all `host_blank=1 backing=Same`, which the record's own legend defines as "the cache agrees and nothing was lost" over pages the entry still covers |

**No lost guest work anywhere in it.** That is the point of writing it down: a
reader who runs `--import-off` for the first time meets thirty records whose
name says the host cannot import guest RAM, on a run where they turned that off
themselves.

Two counting traps live in these lines, both cheap to fall into:

- `lin_rung_blank_with_host_entry` greps 13 times and is 8 records. Five are
  `OFF store_routes` census lines carrying it as a *counter name*. Summed as
  `AGENTS.md` requires — the samples are per window — the route fired 9 times
  across 5 windows, busiest window 5, against 8 deduped fail lines.
- `grep -c 'backing=Same'` returns 16 for those 8 records, because each record's
  explanatory parenthetical contains the string `backing=Same` while defining
  it. Match the field with its neighbour (`host_blank=[01] backing=`) or you are
  counting the legend.
