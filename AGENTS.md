# AGENTS.md

Operating guide for AI agents working in this repository.

## What Belongs In This File

Durable rules that change how an agent works: the principles below, the support matrix, the commands
that verify a change, and what a commit must say.

**Findings do not belong here.** A measurement, a counter reading, a sweep that came back empty, or
an account of how a past session was misled is not an instruction — and this file has repeatedly
grown to several times its useful length by collecting them. Put them where they will be read:

- **next to the code they explain**, as a module or function doc — that is where the next reader
  meets them, and it is the only place that stays true when the code moves;
- **in the commit body**, for what one change measured and did not verify;
- **in `kb/` and `journal/`** (both gitignored) for investigation notes, working hypotheses, and
  session logs. `kb/` entries carry frontmatter and `[[links]]`; follow the existing shape.

Before adding anything to this file, ask whether it changes what someone *does*. If it only records
what was once true, it goes in one of the three places above.

## What This Project Is

This research project emulates Apple's paravirtualized GPU on the host. An unmodified macOS guest
uses Apple's own GPU drivers; our QEMU device and Rust backend decode the command stream and
execute it through Metal or Vulkan. We ship no guest driver.

`crates/reims-vgpu` supports three first-class pathways:

| Pathway | Host | Guest | Attach | Page shift | Backend | Boot |
|---|---|---|---|---|---|---|
| x86 macOS / Linux Vulkan | Linux x86_64 (KVM) | x86_64 macOS Metal guest | PCI (`reims-vgpu-pci`) | 12 | Vulkan | `vm/boot-x86.sh` |
| arm64 macOS / macOS Metal | Apple Silicon macOS (HVF) | arm64 macOS Metal guest | sysbus MMIO (`reims-vgpu-mmio`) | 14 | Metal-direct | `vm/boot-arm64.sh` |
| arm64 macOS / macOS Vulkan | Apple Silicon macOS (HVF) | arm64 macOS Metal guest | sysbus MMIO (`reims-vgpu-mmio`) | 14 | Vulkan through MoltenVK | `vm/boot-arm64.sh` |

Pathway-specific facts must be verified on the pathway being changed. Do not generalize from arm64
to x86, from Metal to Vulkan, or from one host GPU class to another. Some rails run on exactly one
pathway — the arm64-only mapper rail is the standing example — and no boot on the other host can
measure them.

## Main Components

- `vendor/qemu` - QEMU fork with the thin device shim: QOM, MMIO/BAR, IRQ/MSI, console/display
  integration, and HostOps plumbing.
- `crates/reims-vgpu` - Rust staticlib that owns protocol decode, device model, memory mapping,
  command planning/execution, scheduling, and Metal/Vulkan backend behavior.
- `crates/reims-vgpu/src/observe/` - crate-wide observability: fail logs, typed decline reasons,
  emission helpers, and gates.
- `crates/reims-vgpu-wire` - derived wire-format views, with their own `AGENTS.md`. Where that file
  is stricter than this one, it wins.
- `vm/` - snapshot-revert boot scripts for arm64 and x86 guests.
- `bugs/` - gitignored. One directory per defect that belongs to `metal2vulkan` rather than here.

Start with the owning source modules and nearby tests when changing device, decode, present, or
backend behavior. Keep durable design facts in code comments close to the behavior they explain.

### A translator defect is packaged, not described

Some of what looks like a device bug is a `metal2vulkan` bug, and the only useful handoff is one the
translator's own agent can act on without this repository, without a VM, and without asking a
question. So when a defect is upstream, write it into `bugs/<defect-name>/` before doing anything
else with it:

| File | Contents |
|---|---|
| `README.md` | What the guest loses, the defect, where it lives, what has been ruled out |
| `input-*.air` | The AIR that reproduces, one file per distinct reproducing blob |
| `failure.txt` | Verbatim validator output **and the per-tier retry trace** |
| `repro.sh` | Runs every input and prints the verdict |

Name the directory after the defect, not the symptom and not the shader. Two inputs failing the same
way are one bug; two failing differently are two, however alike the device-level refusal looks.
`bugs/README.md` carries the rest — how to recover AIR from a boot's scratch directory, and why the
tier trace is the part worth capturing.

`bugs/` is gitignored because every payload in it is Apple's AIR, and third-party bytes stay local
under the rule at the top of `## Commit Guidelines`. Hand a directory over by copying it.

**Check the pin before diagnosing.** `git ls-remote` on the dependency costs a second. Two sessions
here diagnosed a translator defect down to a function and a line number without running it, and the
arm that made the repair reachable was already sitting one commit ahead of what we pinned.

## Operating Principles

### C Is A Thin Shim

C and Objective-C in the QEMU path exist to connect QEMU to Rust. Keep product logic in
`crates/reims-vgpu`: protocol interpretation, resource state, scheduling, GPU encode, present model,
backend policy, and performance behavior belong in Rust.

A shim that calls two queries and branches on the pair has reconstructed a rule, which is the same
violation as writing one. Export the answer, not the inputs — and delete the inputs, because a shim
that can still assemble its own answer eventually will.

Once a `reims_vgpu_qemu_*` entry point is wrapped in the shared `reims-vgpu-shim.c`, that wrapper is
the only caller; neither device shim may reach the raw entry point. This has cost twice, both times
on who owns the host console — `reims_vgpu_qemu_scanout_may_paint` assembled shim-side from two
other queries, and `reims_vgpu_qemu_console_feed` called raw by the arm64 shim, which read a failed
call as "not early". A test used to check this by parsing the C; it was a source grep and is gone.
Each shim is built for a different host, so a re-inlined call fails on whichever pathway is not
being booted — check both shims by hand when you touch one.

Anything crossing the boundary lives twice, once in Rust and once in
`crates/reims-vgpu/include/reims_vgpu_qemu_abi.h`, and nothing else in the toolchain compares the
two: Rust does not `#include` the header and the shims do not read Rust. Every constant that crosses
gets a test, using `qemu::abi::header_define` — see `the_abi_header_agrees_on_the_version`,
`..._on_the_entry_point_return_codes`, `..._on_the_guest_ram_codes` and
`..._on_the_host_action_layout`. Add one with any new shared constant; a drift here is a bug on
exactly one pathway.

### Never Fail Silently

If a decoded guest command is rejected, dropped, degraded, unsupported, or mis-executed, make the
reason visible. Use typed decline/refusal reasons and emit them through the always-on failure path
so `/tmp/reims-vgpu-fail.log` explains what happened.

Expected control flow should stay quiet. A resolver saying "not ready yet" or an intentionally
unbound `ref == 0` is not a failure. A real loss of guest work is.

### Measure Before Fixing

You cannot fix what you cannot measure. If we do not know what class of failure we are fixing, we
are operating blind and guessing.

Before landing a visual, protocol, performance, or translation fix, add or identify a log-level or
test-level proxy for the bug class. Screenshots are useful evidence, but they are not a regression
gate by themselves.

### Tests Define Done

If there is no test for it, it is not done. No test means a future agent can regress the changeset
without noticing.

Behavior changes need tests that fail without the change. Bug fixes need a focused synthetic case
or proxy test for the bug class. Run Rust tests serially with `-- --test-threads=1`; GPU-touching
tests are not safe to run in parallel.

### Do Not Validate By Grepping Source

**No test, script, or gate may read this repository's own Rust source as text and assert on what it
finds.** No `read_to_string` over `src/`, no regex for a call shape, no walking `src/` to build a
census of sites, no verdict table keyed by file and line that a new match must be added to.

This crate accumulated forty such scanners — 17,000 lines, more than the behavioural suite — and
retired all of them at once. They fail in three ways that no amount of care fixes:

- **They test spelling, not behavior.** A scanner proves a `push` sits near a `len()` comparison. It
  cannot prove the guest keeps its work, which is the only thing that matters. A rename satisfies
  one; a real regression walks past it.
- **They are wrong in the direction that reads as thorough.** One scan here was off by 40 % because
  `use ops::{texture_view as w_view}` never puts the family name after the token `ops::`. It
  reported a clean sweep of a population it could not see, and its output looked identical to a
  correct one.
- **Their verdict tables become the work.** A table demanding a written line per site turns every
  edit into a documentation exercise against the scanner rather than against the device, and the
  lines go stale the moment the code they describe moves.

What to do instead, in order of preference:

1. **Make the invariant unrepresentable.** A bound that must not be exceeded belongs in a type or a
   constructor, not in a scan looking for places that forgot to check it.
2. **Derive, don't duplicate and compare.** If two constants must agree, make one `= the other`. A
   scanner that checks two spellings match is solving a problem you created by having two spellings.
3. **Assert the relation where the constant is declared.** See below.
4. **Write a behavioural test** that fails when the guest loses work.

If none of those can reach it, the invariant goes in a doc comment next to the code and is enforced
by review. An unenforced rule that is honestly unenforced beats a scanner that reports success on a
population it cannot see.

One narrow thing is not covered by this ban: reading the **C header** at
`crates/reims-vgpu/include/reims_vgpu_qemu_abi.h` to compare its `#define`s against the Rust
constants, because Rust genuinely cannot `#include` it and the two spellings have no other
comparison. That is `qemu::abi::header_define` and its `#[cfg(test)]` tests — it parses one
`#define` by name out of an `include_str!`, and it is not licence to grep the C shims for call
shapes. Coverage-instrumented tools that observe a real boot (`scripts/runtime-dead`) are also
fine; they measure execution, not text.

### Inline `const` Assertions

A `const _: () = assert!(..)` is the right tool for a **relation between two independently-derived
values** — that a table is wide enough for the mask indexing it, that two binding bands do not
overlap, that five bit-masks tile a word, that an external crate's enum discriminant still matches
the ordinal this device decodes. `rustc` evaluates these on every arm that compiles the file,
including the cross-compiled `--target aarch64-apple-darwin` clippy run, which is why they reach
Metal code no Linux host can run tests for.

**Declaring a constant and then asserting it equals what you just declared it to be is not a
check.** `pub const X: u64 = u64::MAX;` followed by `assert!(X == u64::MAX)` proves nothing. Nor
does hand-copying a value and asserting the copy equals the original: write `pub const X: u32 =
other::X;` and there is nothing left to drift.

### Do Not Overfit Fixes

Never special-case behavior for a screenshot, boot stage, pixel dimension, resource size, object id,
function name, pipeline ref, or observed content pattern. Implement the decoded API contract.

Temporary probes are fine when they collect evidence. Remove probe-only behavior before claiming the
fix. Do not turn observations into product heuristics.

### No Magic Numbers

Do not guess numbers because they fit one observation. Derive constants from the contract: SDK
headers, `sizeof`/`offsetof`, decoded guest fields, documented serializer output, or controlled
empirical measurement. Record the basis in the code or the commit body when the value is not obvious.

Guest page geometry is always explicit. Portable code takes `page_shift` or `page_size`; arch-fixed
helpers must say so in their names.

### Write Comments For The Code

Inline comments should explain the code as it exists now. Do not mention the prompt, a temporary
plan, a phase number, a section of an implementation plan, or the reason a plan asked for the
change. Plans are consumed and deleted; code comments outlive them and must not point at things
that no longer exist.

### Keep Claims Narrow

State exactly what you verified. A single green boot does not prove an entire class is fixed. Broad
claims such as "zero-copy everywhere" or "no fallback remains" require an audit of every place that
could falsify them. One workload on one pathway proves one workload on one pathway.

### A Subagent Shares Your Working Tree

A delegated agent runs in this same checkout, so anything it does to git happens to you. Brief every
one of them read-only, by name: no `checkout`, `switch`, `stash`, `reset`, `restore` or `commit`.
The failure is quiet — an agent that runs `git checkout HEAD~1` to get a "clean build" and does not
return leaves HEAD detached, and the next commit lands off the branch where nothing but the reflog
can find it. Check `git status` after any delegated run before committing.

**The same rule binds you, and the likeliest way to break it is reverting a probe.** Stubbing a gate
out to prove a test really fails without it is the right habit; undoing it with `git checkout --
<file>` takes every uncommitted edit in that file with it, including the change you were probing.
Copy the file aside and copy it back, or edit the stub out the way you edited it in. Never reach for
git to undo a probe.

## Before A Broad Sweep

Deletion and audit sweeps over this crate have been run many times. What each concluded lives next
to the code it concluded it about — read the module doc before deciding a rail is dead, and do not
"discover" a comment recording a heuristic that was already measured and removed.

Four rules survive every sweep:

- **A never-firing branch is almost never a deletion.** A decoded-but-untaken arm is usually
  contract fidelity — a real Apple opcode this workload does not issue — or a healthy-zero alarm,
  where a firing *is* the bug. The test to apply: **name the guest action that would take this
  path.** If you can name it, it stays. Deleting one loses guest work silently the first time a
  guest takes it.
- **A zero can be an artifact of where it is sampled.** Find a census field's sampling point before
  cutting it. A zero hit rate on one pathway is not a dead cache either — page shift alone changes
  it. The same
  trap applies to a *check*, where it is easier to miss: a guard that tests two endpoints of a range
  reads as thorough and samples two bytes of it. Ask what granularity the underlying answer varies
  at, and walk the range at that granularity or ask the host for a span verdict.
- **A drop counter reading zero is not a measurement.** A record stopping at slot 4 and one stopping
  at slot 30 both read zero, and only one says the bound has headroom. Band the *requested* reach
  before widening or narrowing any table.
- **Two arms that consume one wire form must be diffed against each other, not read alone.** The
  comment that settles a divergence is more often on the callee than on the call site, and the arm
  that is easiest to read is often not the arm the boot takes. Check which one runs before calling a
  divergence theoretical. When you find one, check whether its failure line is shared or copied — a
  copied one is the next divergence, and so is a copied *check*. Where three arms consume one wire
  form, the arm holding the shortest version of the rule is the one to read first: the render-pass
  attachment prefix had a four-term admission rule written by hand three times, and two of the three
  copies were missing a different term. Count the terms before reading what they say.

**Bounds are the class most often swept, and sweeping is not how you find them.** A bound can cost
guest work in four ways — an entry evicted, an entry never recorded, a run read only partway, and a
**bitmask standing in for a set**, where `mask |= 1u32 << index` bounds the membership to 32 with
nothing declared anywhere and a shift past the width *wraps* in release rather than failing. A fifth
has no number at all: **a slot holding one decoded record**, where `acc.x = Some(rec)` in a decode
arm is a capacity of one and the second record the guest sends drops the first. A sixth is **an enum
narrowed to its ordinal at the producer**: write `selector as u32` and every consumer downstream must
match integers, which `rustc` cannot check for coverage, so a table that is silently one member short
compiles and reads as complete. That is not hypothetical — it cost the arm64 pathway every `R32Uint`
storage bind, against a selector the contract declared and the x86 pathway ran.

None of these is found reliably by looking for them, which is why the scanners that used to be
listed here are gone. Make them unrepresentable instead: put the bound in the type that carries the
collection, give the mask a width pinned by a `const` assertion against the table it indexes, and
where a latch must not be overwritten, make the second write a typed refusal rather than an
assignment. Then a new site cannot be added without meeting the rule, and nothing has to go hunting.

**Where a selector, opcode or class tag is this crate's own vocabulary rather than a guest value,
carry the type and not the integer.** A guest value is arbitrary and must be parsed once, at the
boundary, into something total; after that boundary the ordinal has no job left. Keeping it costs the
exhaustiveness check on every consumer, and buys nothing a `Display` impl or an `as u32` at the one
log site would not. This is the same rule as "derive, don't duplicate and compare", one step earlier:
the second spelling you avoid is the integer.

Prefer an instrument over a reading, where an instrument that is not a source grep exists. Reading
an audit against itself cannot see an opcode that is simply the wrong number, a length four bytes
off, or a field two bytes too wide:

| Question | Instrument |
|---|---|
| What is reachable but never runs? | `scripts/runtime-dead` — coverage-instrumented driven boot |
| Does a decoder refuse or drop a record Apple emits? | `crates/reims-vgpu/tests/wire_fixtures_reach_the_decoders.rs` |
| Does a `[`link`]` in a doc comment name a symbol that no longer exists? | `cargo doc`'s intra-doc link pass |
| Does a constant crossing the C boundary still match the header? | the `header_define` tests in `src/qemu/abi.rs` |

```sh
RUSTDOCFLAGS="-A rustdoc::private_intra_doc_links" cargo doc -p reims-vgpu \
  --no-deps --document-private-items \
  --no-default-features --features backend-vulkan,host-window
```

Triage its output before editing anything, because most of it is not rot. Three classes, and only
the first is:

- **The symbol exists nowhere.** Real rot, and the only class worth a commit. Confirm with a grep
  for the leaf name; run the doc build on the Metal arm too (`--target aarch64-apple-darwin
  --features backend-metal`) and take the intersection, or a `backend-metal`-gated target will read
  as missing on the Vulkan arm.
- **A bare name inside a `//!` module doc.** These never resolve here whatever they name — a
  `pub fn` in that same module fails exactly as a deleted one does, and `self::` does not help.
  Only a fully-qualified `crate::…` path resolves from a `//!` doc. Cosmetic; the reference is
  correct, it just does not become a hyperlink.
- **An accurate path to a private item**, or to anything under `engine::pools` (a private `mod`).
  `--document-private-items` does not make these linkable across modules. Correct as prose; do not
  "fix" one by deleting a true reference.

A wire module with no importer is either a real gap or a family still declared twice. Where a device
offset names a field a wire struct already declares, reach for `offset_of!` rather than a re-exported
number, so a rename fails the build.

Their output is a map, not a kill list. And one trap they teach: **an `Ok` from `render::decode` is
not a decode** — `Kind::OtherAccepted` is the catch-all for "no arm claimed this", and reading it as
success hides a whole family of lost records behind a green run.

### Reading the fail log

- **A log that stops is a different failure from a log that complains.** Every census here —
  `drain_duty`, `store_routes`, `engine_delta` — is written at the *end* of a drain tranche, so a
  drain thread that never returns produces no line at all while `display_vbl` and `host_window_loop`
  keep ticking from other threads. The boot then reads as healthy. Two lines say otherwise and both
  come from outside the drain: `driver_call reason=driver_call_outstanding` (a host driver call past
  its deadline, once at 10 s and then once a minute) and `driver_quarantine` (a call a previous
  process died inside, refused rather than made again). If the censuses stopped and neither line is
  there, the stall is somewhere they do not reach — attach a debugger, remembering that
  `yama/ptrace_scope=1` means gdb has to be an *ancestor* of the QEMU process.
- **Count the boots before ranking anything.** The device appends and never truncates, so a log you
  did not just create may hold several boots of several builds. `grep -c vk_caps` is the boot count:
  one line per device creation. This is not the check below for "a boot's log and not the test
  suite's", and it fails in the opposite direction — the log *is* a real boot's, it is also three
  other boots'. It inflates in a way that reads as a finding rather than as noise, because
  `first_sight` latches per process: one refusal seen once per boot arrives as N identical-keyed
  lines and looks like a decoder failing thousands of times. A stale log ranked this way named two
  documented healthy-zero decoders as firing ~96 000 times between them; a clean driven boot on the
  same tree put both at zero. **`rm -f /tmp/reims-vgpu-fail.log` before the boot** and the question
  never arises.
- **Volume is not alarm.** Most records are on the `OFF` channel, and the highest-volume tags are a
  1 Hz heartbeat. That is cadence working, not an over-eager emitter.
- **Absence of a decode line proves nothing.** Every decoder in `decode/` is silent on success and
  emits only on `Err*`. "Opcode X never appears in the log" is not evidence that arm never ran; the
  `store_routes` counter set is the only usable never-fired signal.
- **Filter the channel before ranking `reason=`.** `OFF` records carry `reason=` too, for ordering
  and control-flow events that are not losses, so the obvious
  `grep -o 'reason=[a-z_0-9]*' | sort | uniq -c | sort -rn` inverts the queue. A fail-channel record
  begins with its own event name and an off-channel one begins with the literal `OFF `, so
  `grep -v '^OFF '` first.
- **A named reason on the fail channel is not automatically lost work.** Some report a repair that
  *succeeded*, fail-visible so the reliance stays measurable. Read the emitting type's own doc
  before concluding a reading is a loss.
- **A counter and a fail line count different things.** Emitters dedupe; counters do not. Do not
  quote one as the other.
- **`store_routes` counters are per-window: sum the samples.** They reset each census interval, so
  `sort -n | tail -1` returns the busiest window and reads like a boot total — three to four times
  under the real one on a routine boot. The other census counters (`registry_pressure`'s `peak`,
  `peak_mib`, `resample_peak_ms`) are cumulative high-waters, where the last sample *is* the answer
  and summing is the error. Check which you have before quoting either: a per-window series
  descends across a boot, a high-water never does. Where a route splits into sub-counters, the split
  must add up (`no_state + texture + linear == unknown_object`), and that identity is the cheapest
  way to catch the mistake.

## Support Matrix

arm64 and x86 are both first-class. Metal and Vulkan are both first-class where the host supports
them.

The Vulkan backend must support all four memory/import cells:

| | Host-pointer import available | No import available |
|---|---|---|
| Unified memory | Apple M-series / MoltenVK, Intel/AMD iGPU on Mesa | Unified-memory hosts without the extension |
| Discrete memory | Discrete GPUs that import and then copy into VRAM | Discrete GPUs that stage every crossing |

On a unified host the import *is* the rail: a `GuestSlice` binds directly and there is no
device-local mirror. On a discrete host the device-local resource is the working memory and the
import is its backing store, so the copy between them is GPU-side and correct rather than a
fallback. `caps::memory_topology` decides which, from two structural signals — do not add a third
classifier and do not branch on vendor or driver name. A misclassification must stay a
**performance** bug: nothing may branch on topology in a way that changes what the guest observes.

Vulkan 1.2 is the baseline. Anything above Vulkan 1.2 must have a fallback or a capability-gated
path. Gate on capabilities, not vendor names, driver names, or API-version assumptions.

### Guest RAM reaches the GPU by importing a host pointer, sized to a RAMBlock

The GPU reads and writes guest memory through the mapping QEMU already holds over each RAMBlock,
imported once and held for the VM's lifetime. The primitive is per platform and the three converge:
`VK_EXT_external_memory_host` on Linux and Windows, the same extension through MoltenVK on macOS,
and `newBufferWithBytesNoCopy` on the Metal-direct arm — which is what MoltenVK implements the
extension over.

Portability is why. dma-buf is a Linux kernel object and there is no Windows equivalent —
`VK_KHR_external_memory_win32` moves NT handles for GPU-allocated or D3D resources, not arbitrary
host pointers. One primitive spans all three targets and it is the host-pointer import.

**The bound is a type, and there is no scanner behind it.** A host-pointer import carries no bound
of its own, so the whole remaining safety argument is `runtime/guest_ram.rs`: a `GuestRamImport` is
sized to one RAMBlock exactly, `GuestSlice` has that one constructor, and the constructor
bounds-checks with checked arithmetic. A slice's absolute position is obtainable only by presenting
it back to the import that made it, which is also where the cross-import check lives. The threat
this bounds is not the guest reaching its own RAM — it authored the shaders — it is a stray *past*
the RAMBlock into this process's own memory.

Nothing scans for violations and nothing should. Read that module's doc before adding an import
site; if you find yourself needing a raw offset or a per-slice host pointer, the answer is a new
method on the import, not a field on the slice.

**The import is never required.** A host without the extension reaches a negative rung, asks for
nothing at `vkCreateDevice`, and runs every guest-memory rail through the copying path. Those rails
are not a legacy arm: they are the only arm on such a host, and they are the arm a discrete GPU
takes regardless, because there the copy into VRAM is the point. Both halves are gated — the
capability at `caps::host_pointer`, and the reference at `runtime/guest_ram_map.rs`, which refuses
by name when no backend published a granularity.

**Page recycling is unchanged and still load-bearing.** The guest reassigning a GPA to a different
allocation while we hold a reference over it is the PTE-corruption class. It applied to the dma-buf
and it applies here. Three modules carry it now: `runtime/guest_ram.rs` for the surface
page-ownership guards, `runtime/render_writeback.rs`, whose doc walks the four hazards the retired
deferred-flush window carried and says why each cannot arise in the direct path, and
`runtime/node_guard.rs`, the alarm for the worst end of it — a host write landing on a page that
holds the guest's own page-table entries.

**The pinning difference is real; do not gloss it.** A udmabuf fd made the pages it named
unswappable and unmigratable, and closing it revoked the GPU's access. `VK_EXT_external_memory_host`
promises neither in its specification. amdgpu and the NVIDIA driver call `get_user_pages` at import
time in practice, but that is an observation about two drivers. We traded a kernel-enforced pin and
a revocation handle for a primitive that exists on all three hosts. If a host is ever observed
migrating a page under a live import, that is a real defect with a measurement — it belongs in
`kb/`, not in a retrofitted guarantee here.

Guest RAM is not **fd-backed**: the import is over an ordinary mapping, so `vm/boot-x86.sh` uses a
plain `-m` allocation. `memory-backend-memfd,share=on` outlived the dma-buf rail it was for, on the
grounds that a shared memfd is what makes uffd minor-fault mode applicable; it is gone, because uffd
needs a privilege QEMU does not have on the dev host anyway. Restoring the backing is a
prerequisite for ever wanting uffd here, but it buys nothing alone.

So the deferred-flush rail — the device's largest cost — is retired, by writing into guest pages
directly. **`runtime/storage_flush/` went with it and no longer exists**; do not go looking for it,
and do not read a reference to it in an older commit body or `kb/` entry as a live path. Its two
halves are now `runtime/render_writeback.rs`, whose module doc lists the four hazards the deferred
window carried and why each cannot arise in the direct path, and `runtime/host_writes.rs`, the
per-page record of which guest pages this device has written. Read `render_writeback`'s doc before
assuming a landing is safe to skip. Note that `runtime/gva_view.rs::ensure_gva_view` hands back a
host pointer but is not a window resolver — it requires the span to be one contiguous page run and
returns `None` otherwise.

### Environment overrides

Every variable the crate reads is named in `crates/reims-vgpu/src/env.rs`, which also owns the parse.
Read a variable through it or the second spelling of "off" is a divergence nothing can find.

**An override may only narrow what the device does; it may never widen it.** A switch can turn off a
rail the host could have run. It can never turn on one the host reported it cannot: capability is
measured from the device, and binding an extension a host does not advertise fails `vkCreateDevice`
while importing a handle type it declines is undefined behavior inside the driver. Add a switch as a
new refusal reason, never as a new permission.

`REIMS_VGPU_GUEST_IMPORT=off` is the one that matters for verification: it takes a capable host down
to the `disabled_by_env` rung, which is how the copying rails get exercised without hunting for
hardware that lacks the extension.

## Verification

Pick the pathway your change affects.

Then pick the **rail** — one guest OS line, `macos-11` … `macos-26`, each with its own snapshot
history under `vm/disks/rails/<rail>/` (x86) or `vm/guest/rails/<rail>/` (arm64). `--rail NAME`
selects it, `--snapshot LABEL` selects within it, and a boot with neither follows `rails/current`.
`--list-rails` says what exists. **Name the rail in any result you report**: a macOS 11 guest and a
macOS 26 guest are two measurements of two guest drivers, and the `rail=` field in the boot's own log
line is what says which one a reading came from. The same rule as the pathway table above applies —
do not generalize from one rail to another.

- Arm64: `vm/boot-arm64.sh --device reims-vgpu-mmio --testing`, then
  `scripts/screenshot-when-macos-host/screenshot-when-macos-host.sh /tmp/screen.png`
- x86: `vm/boot-x86.sh --device reims-vgpu-pci --testing`, then
  `scripts/screenshot-when-kde-plasma-host/screenshot-when-kde-plasma-host.sh -o /tmp/screen.png`

### Run `vm/guest-authorize.sh` after an x86 boot, before any probe

Every probe under `scripts/` reaches the guest as `ssh -o BatchMode=yes macos-vm`, which is key auth
and nothing else. Only `macos-13` was provisioned with that key; the other rails authenticate by
password, and `BatchMode=yes` turns that into a silent failure that reads as "the guest is not up
yet". `vm/guest-authorize.sh` waits for sshd, installs the key into the running clone, and verifies
`ssh macos-vm` before returning. It is idempotent and needs no password on a rail that already has
the key, so a harness may call it unconditionally.

It also forgets the host-key pin for `127.0.0.1:2222`, which is a different machine on every rail —
without that, whichever rail booted first wins and every later rail fails the host-key check.

**Bound every guest-side command with `timeout` on the host side.** An unattended harness cannot
tell a wedged guest command from a wedged boot. `system_profiler SPDisplaysDataType` has been
observed hanging indefinitely on a macos-11 guest, and a `sudo` that wedges after authenticating
holds its timestamp lock, so every later `sudo` — including `sudo true` — queues behind it forever.
A host-side `timeout` does not kill the remote process, so root steps in particular must be issued
once and never retried after one times out.

### Drive a probe from the host, not from inside the guest

**macOS 26 has no command line developer tools and no working `screencapture`.** A probe that
compiles C on the guest (`window-drag-probe`'s `drag.c`) cannot build there at all, and
`scripts/lib/guest-display.sh`'s `guest_display_size` returns nothing because `screencapture` fails
with "could not create image from display". Both `osascript` desktop-bounds routes are empty too — a
fresh ssh session holds no Apple Events consent. So a guest-side probe does not silently degrade on
that rail; it does not run, and it reports a *build* or *permission* failure that reads like noise.

Reach for the host side first. It works identically on all six rails and needs no guest tooling, no
consent and no permission — and no ssh, which matters on a rail that panics during a third of driven
boots:

| want | use |
|---|---|
| pointer / keyboard | `scripts/qmp/qmp.py` — `move`, `click`, `drag`, `key`, `type`, `wheel` (QMP `input-send-event` to the machine's usb-tablet/usb-kbd) |
| guest display size | `scripts/qmp/qmp.py size` |
| a screenshot | the host helper (`screenshot-when-kde-plasma-host`), **never** QMP `shot` |

QMP `shot`/`screendump` stays disabled on purpose: with the host-owned window and QEMU at
`-display none` the frame never crosses into QEMU's address space, so a screendump shows something
other than what the window shows. Sizing is exempt because the DisplaySurface still carries the right
dimensions, which is all the input helpers ever used it for.

Do not add a guest-side fallback "for the rails that have clang". A second path that works on five of
six rails rots, and the rail it skips is the one with the open defects. `dock-hover-probe`'s
`hover.c` was deleted rather than kept for exactly this reason. Where a probe genuinely needs guest
state — a process list, a log — ssh remains the only route, but bound it with `timeout` and never
read its absence as a device result.

**sshd answers well before the desktop composites.** A probe started when port 2222 opens
photographs the Apple logo and the boot progress bar. Wait for `pgrep -x Dock` over ssh, then give
the dock and wallpaper a few seconds to settle.

### `probe exit=0` is not a clean boot — grep the boot's own stdout for a panic

A guest kernel panic can land **after** the probe has finished and reported success, so the two
signals every sweep here has been judged on — `probe exit=0` and `dev=1` — both read green on a boot
whose guest died. That is not hypothetical: on macos-26 it is roughly one driven boot in three, and
it went unrecorded for as long as rails were judged that way.

`vm/boot-x86.sh` already prints `capture-then-revert (guest kernel panic)` and keeps the serial log,
so the check is one grep of the boot's own stdout and it is the verdict that outranks the probe's:

```sh
grep -q 'guest kernel panic' "$OUT/boot-stdout.log" && echo PANIC || echo ok
```

Report it per rail alongside `dev=` and the probe's exit. A rail that panics on a third of its boots
is not a rail that passes, and one clean boot of it is not evidence — **band a suspected panic over
at least six boots before believing a rate**, in both directions. A single green run says nothing,
and so does a single red one.

### A boot on a capable host does not exercise the copying rails

Where the import works, every guest window takes it, and the copying rails run zero times — so a
green boot says nothing about them, and they are the only rails on a host without the extension. A
change touching guest-memory upload, writeback or bind needs the boot a second time with
`REIMS_VGPU_GUEST_IMPORT=off`. Confirm it took: `vk_caps` reports
`host_pointer_import=disabled_by_env`, and `OFF guest_ram_map reason=guest_ram_map_no_backend_import`
appears once. Nothing may then report a bound import — a non-zero import count means a bind ran past
a closed gate.

### `present_hz` tracks `offered_hz` exactly, so read the pair and never one alone

The x86/Vulkan host window presenter used to clamp at **~41 frames a second
whatever it was offered**, because `WindowPresenter` allowed one present in
flight. It now runs `PRESENT_IN_FLIGHT` of them and the clamp is gone. Eight
driven macos-13 sustained-animation boots across the two arms of
`REIMS_VGPU_PRESENT_DEPTH`:

| arm | `presents`/`offered` per boot | `busy_fence` per boot |
|---|---|---|
| depth 3 (shipping) | **1.000** on all five | 0, 0, 0, 0, 0 |
| depth 1 (`=off`) | 0.828, 0.822, 0.853 | 411, 421, 372 |

Every shipping boot presented **exactly** what it was offered, including two
whose offered rate was ~56 Hz rather than ~49 — so this is not a new clamp
sitting a little higher, and it holds across both compositing regimes.

At n=3 vs n=3 after regime exclusion, `presents_s` rose **16.2 %** with the arms
disjoint at 3.6x their own spread, while `offered_hz`, `draws_s` and
`kib_per_draw` all *overlapped* — the device did identical work and only the
presenter changed. So `present_hz` is now a real reading, and the old advice to
ignore it is retired.

**But it is a reading of two things, and the pair is what says which.** The
presenter now passes everything, so `present_hz` equals `offered_hz`, and
`offered_hz` is the device's own publish rate. A device change that raises
frames raises *both*. Always quote them together:

- both rose — the device published more, which is the win.
- `offered` rose and `present` did not — the presenter has become a ceiling
  again. Check `busy_fence` and `busy_acquire`; on the shipping depth they are 0.
- neither moved — the change bought no frames, whatever else it bought.

**In the fast population `present_hz` is the score, and it is sharper than
`us/draw`.** A fast-latching macos-13 guest free-runs on a zero frame period, so
its rate is *work-limited*: whatever this device stops doing per draw, the guest
spends on more frames. Twenty-four interleaved boots pushed the device 20.6 %
the wrong way (`REIMS_VGPU_COMPUTE_GATHER=off`), scored over their fast boots
only:

| arm | n fast | `present_hz` mean (range) | `us/draw` mean (range) |
|---|---|---|---|
| shipping | 5 | **113.2** (109.8-116.3) | 13.21 (12.66-14.02) |
| 20.6 % more GPU work/draw | 9 | **105.5** (101.4-107.7) | 15.07 (13.92-16.16) |

The `present_hz` arms are **disjoint** — the slowest shipping boot beats the
fastest slowed one — while `us/draw` *overlaps* on the same fourteen boots. So
the frame rate is the more sensitive instrument of the two, not the noisier one,
and a per-draw saving that cannot be seen in `present_hz` over five fast boots an
arm is smaller than it looks. Elasticity for sizing a candidate: about **0.35
frames per unit of per-draw GPU work**, so a 10 % per-draw saving is worth
looking for and a 2 % one is not measurable here.

`drain_duty draws` and anything normalized per draw stay the right way to
*attribute* a change — which phase paid — but they are no longer the way to rank
one.

**Quote the presents, never the drop percentage.** This survives the fix and is
the trap that cost a session a wrong call. With a *clamped* presenter the drop
read 4.9 % on one boot and 17.3 % on another because all the variation was in
the denominator: a guest offering 44 loses 5 % and one offering 50 loses 17 %,
and both presented 41. That session quoted 4.9-5.7 % from three consecutive
low-offering boots and concluded the presenter was "worth at most ~5 %, not the
lever it looked like". It was worth 17.8 %.

This also retired a standing puzzle. Several CPU wins here "bought microseconds
and zero frames" — a bounded pipeline cache, −39 % submissions, −20x `stage_us`.
None of them could have bought frames through a presenter already at its
ceiling, and a per-draw saving measured before this fix is **owed a re-run**
rather than believed to have been worthless.

### An undriven boot measures an idle device

A `--testing` boot reaches the desktop and then sits there. Reading its counters as this device's
behavior is how a rail gets called dead when the workload simply never asked for it. If a change is
about throughput, caching, writeback or present cadence, the boot has to be driven.

Run the boot in the background and drive the guest **while it is up** — the `--testing` boot exposes
SSH on `localhost:2222` (`macos-vm` in `~/.ssh/config`) for its whole life, so a probe does not need
its own boot:

```sh
pkill -f 'qemu-system-x86_6[4].*reims-vgpu'; rm -f /tmp/reims-vgpu-fail.log
vm/boot-x86.sh --device reims-vgpu-pci --testing &     # ~7 min before its own hard kill
until [ -f /tmp/reims-vgpu-fail.log ] && ssh macos-vm true; do sleep 5; done
scripts/window-drag-probe/window-drag-probe.sh --seconds 25 --app Safari
```

That produces real window-server compositing, against 0 draws/s idle. The probe refuses a verdict if
the window never moved, so a run that produced no motion cannot be mistaken for a slow device.

### A bursty driven boot measures the gaps between its bursts

Driving is not enough. A probe built out of discrete interactions — a Mission Control round, a
Launchpad round, a drag — spends most of its wall clock waiting for guest animations, and whole
seconds of it have **zero** draws. Nothing in the capture says so: the counters are self-consistent
and the log is well-formed. One such probe put `present_hz` at a median of 2.8 Hz on a device
sustaining 78.8 Hz minutes later in the same VM.

The damage is not a scale factor, it is a different **ranking**. On one build, one rail and one
quiesced host, the bursty probe put `chain_phase`'s `engine` at 49 % and `store` at 10 %; the
sustained one puts `store` at 35 % and `engine` at 28 %. Decisively, the drain worker's duty is 0.00
median on the bursty probe and 0.91 on the sustained one — so **only the sustained arm can turn a
per-draw CPU saving into frames**, and three separate CPU wins measured against the bursty one each
bought real microseconds and no frames at all.

So a throughput, caching or cadence change needs
`scripts/sustained-animation-probe/sustained-animation-probe.sh` as well as an interaction probe.
Name which probe a number came from, the same way a rail is named: they are two populations of draws
and a change can help one and hurt the other.

**Classify the boot before comparing two of them.** A macos-13 boot presents either ~60 frames a
second or ~95-117 for its whole life, tightly, with nothing in between, and the guest picks per boot.
`present_hz`, `draws/s` and `fresh` are all halved on a slow boot, so none of them is comparable
across the two. Mixing boots from both populations is how a real 17 % effect ends up buried under a
2x artifact. `present_hz` alone is the discriminator — the gap between 61 and 94 is empty on every
boot on record.

**`us/draw` is not comparable across the populations either, and it reads like it should be.** A slow
boot asks the GPU for half the work, the governor clocks down, and every GPU microsecond gets more
expensive: over 16 boots with `nvidia-smi` sampled alongside, slow boots read **10.3 % higher
`us/draw`** than fast ones on one binary (15.68 against 14.22), and `us/draw` against the driven-window
clock correlates at r=-0.89. So an arm that happens to draw more slow boots reads slower per draw for
a reason that has nothing to do with the change. Score per-draw numbers within the fast population
only.

Within that population `us/draw` has a **coefficient of variation of 3.4 %** and a 12 % max-to-min
spread over twelve boots, so a per-draw change under ~5 % needs several boots an arm and a single
pair proves nothing. Do not try to correct for the clock arithmetically: dividing by the SM clock
makes the spread *worse* (12 % to 22 %), because the gather is bandwidth-bound and the memory clock
swings 405-14001 MHz independently.

**A run's slow rate is a Bernoulli draw, and its base rate drifts.** Over 40 interleaved boots it was
12 slow in 40; two runs later the same binary read 7 in 12, twice. So a slow rate is comparable only
**within one interleaved run**, never against a number from an earlier one, and a change claiming to
move it needs about twenty boots an arm — twelve cannot separate 0.4 from 0.7. This is a different
rule from classifying a boot: that one is about which population a reading came from, this one is
about the population *rate* being unstable across time on one host.

**The split is not about VBL delivery, and six hypotheses that assumed it was all came back null.**
The guest's compositor paces on a period the kernel hands it, which is initialised to a synthesised
1/60 s and only corrected from the `IOFBCurrentPixelClock`/`IOFBCurrentPixelCount` framebuffer
properties — and the paravirtual framebuffer driver suppresses those, on every boot, confirmed by
`ioreg` on eleven. A boot therefore latches either 16 666 666 ns (paced, **exactly** 60 Hz) or 0
(free-running, work-limited 95-117 Hz), once, for its life. That is why the slow population is a
constant and the fast one is a 22 Hz spread, and it means the *fast* boots are the ones where the
guest never learned a period. Do not "fix" it by forcing a second display-mode change: that would
set the period on every boot and take the good 70 % down to 60 Hz. Full chain and the live
confirmation are in `VBL_REPORT_EARLY` beside `runtime::drain::census::VblCensus`. Read `VBL_REPORT_EARLY` beside
`runtime::drain::census::VblCensus` before spending boots on a new theory: a run of eight holds one
or two slow boots and so cannot tell a cause from a coincidence, which has already produced one
confident wrong answer here.

**Bracket one character of every `pkill -f` pattern**, as `x86_6[4]` does above and as
`reims_vgp[u]-` does further down. `pkill -f` matches against whole command lines, and the shell
running the `pkill` has the pattern in *its* command line, so an unbracketed pattern matches the
process issuing it and the shell kills itself before `pkill` ever reaches QEMU. The bracket changes
the pattern text without changing what it matches. The failure is easy to misread as "the command
worked": the shell dies with status 144 and the surviving QEMU then holds `localhost:2222`, so the
*next* boot dies on the `hostfwd` rule for the reason described below. Two symptoms, one cause.

**The bracket protects only the shell that issues the `pkill`. Every ancestor is still a match**, so
a pinned QEMU path must cross as an **exported variable and never as argv**. Any command line that
names `.../reims-vgpu/…/qemu-system-x86_64-pin-whatever` matches `qemu-system-x86_6[4].*reims-vgpu`,
and a chain runner that takes two pins as arguments is killed by the first boot it starts. That
failure reads as success twice over: the runner prints its first "round 1 arm A" line and then
simply stops, and the harness it launched under reports the unit as finished. An environment
variable never appears in `/proc/pid/cmdline`, which is the whole fix.

**Wait on the fail log, not on SSH alone.** The previous boot's QEMU outlives its script by long
enough to still hold `localhost:2222`, and a new boot that loses that race dies on the `hostfwd` rule
alone — the script prints one line about it and every other line looks like a normal start. `ssh
macos-vm true` then answers at once, from the **old** VM, and the probe drives the guest running the
*previous build*. It fails in the direction that hides it: you get a driven boot, self-consistent
counters and a screenshot of a working desktop, all from the binary you were trying to replace.
Only a live device creates `/tmp/reims-vgpu-fail.log`, so waiting on it catches the case, and
killing any surviving QEMU first avoids the race. Confirm afterwards that the log is a boot's and
not the test suite's, by the presence of `store_routes`/`present_page_identity`.

### A boot measured next to your own subagents measures the contention

Every `us=` number this device reports is wall clock on a shared machine, so a driven boot taken
while a subagent greps, a `cargo` build runs, or a second VM lives is measuring your harness as much
as the device. This does not look like an error — the log is well-formed and the counters are
self-consistent — and it has been measured to halve throughput, triple per-draw cost, and invert the
ranking between the device's two largest costs.

**Run the boot with nothing else running, and check `uptime` before believing a timing.** Counts are
far more robust than timings: `store_routes`, refusal counters and the gate do not measure time and
survive contention. When a machine cannot be quiesced, reason from counts and treat every `_us`
field as an upper bound.

### Rust tests

Run the relevant native tests serially from the repo root:

```sh
cargo test -p reims-vgpu --no-default-features --features backend-vulkan,host-window -- --test-threads=1
cargo test -p reims-vgpu --no-default-features --features backend-metal -- --test-threads=1
```

`backend-metal` is Apple-only; run that arm only on Apple hosts. Run the feature matrix from the
repo root when cfgs, features, backend boundaries, or shared Rust code change:

**On a non-Apple host, the test functions under `backend/metal/` do not run, and nothing in the
output says so.** This is worse than the fixture gap below, which at least reports `ignored`: these
tests are `cfg`-ed out of the arm you can run, so a Linux session's green count simply does not
include them and reads exactly like a clean tree. The cross-compiled clippy and `cargo check`
commands above *compile* them, which is why a code warning there is still caught — but
`cargo test --target aarch64-apple-darwin … --no-run` fails at the **link** step (no Apple linker,
no macOS SDK), so no binary is ever produced. Do not read "compiles on the Metal arm" as "its tests
passed"; nobody on a Linux host has run them.

Nothing counts them any more — a source scan used to, and went with the rest. Do not try to restore
the count with a `grep` for `#[test]`: that is what the scan did, and it was wrong by four, because
the prose in those files says "a `const` assertion rather than a `#[test]`" and the grep counted the
sentences. The gap is real and unclosed: work on `backend/metal/` needs an Apple host to be tested
at all.

Where a file under `backend/metal/` is pure logic, **move it out of the gated tree** rather than
working around the gate — `backend::hash` is the worked example, and its two tests now run on every
arm instead of on none. The bar is that it names nothing from the `metal` crate; state in the
module's own doc why it sits outside `metal`, or the next reader moves it back.

Copying the file to `/tmp` and building with bare `rustc --test` still works for a one-off reading,
and needs the `//!` module doc stripped if it links outside the file. It is not a gate: nothing
re-runs it. A file that reaches `crate::` for more than constants needs its dependency closure
copied too, which is usually the point at which the logic belongs in `contract/` instead.

```sh
scripts/feature-matrix/feature-matrix.sh
```

Before and after long Rust test runs, sweep orphaned test binaries:

```sh
pkill -9 -f 'target/debug/deps/reims_vgp[u]-'
```

**The only layout-truth tests do not run on a checkout without Apple's captured fixtures**, which is
every non-Apple checkout — `crates/reims-vgpu-wire/fixtures/` is gitignored. They report `ignored`
rather than `ok`, so the run says so, and the ignored count is the one to read. Nothing else in
either suite covers what they cover, so a green run is not evidence about a wire layout. Regenerate
with `scripts/wire-oracle/wire-oracle.sh` on an Apple host, and set `REIMS_WIRE_FIXTURES_REQUIRED=1`
there so their absence fails the build.

**The `backend-metal` `--lib` arm is expected to be green.** It used to carry six standing failures
in a module that has since been deleted outright, and this file used to tell you to expect them;
they were Vulkan-rail tests compiled unconditionally, and they carried the gate before the module
went. A red there is a real
result again — do not restore the exception, and do not silence a new one by weakening what it
asserts.

## Commit Guidelines

Commit only work you wrote. Never commit third-party code or intellectual property, including Apple
software, firmware, disk images, `.mtlb`, AIR, or SPIR-V, or a disassembly listing of any of them.
Keep those artifacts ignored and local. Reports may include original analysis, metadata, hashes, and
reproduction steps, but no third-party bytes or excerpts.

Each commit should have a detailed message body that states:

- Which component or pathway it touches.
- What behavior changed and why.
- What tests, clippy runs, feature-matrix checks, or live-VM verification were performed.
- What was not verified, if anything.

Rust commits should be warning-free under clippy with `-D warnings` for every affected matrix arm.
**All three run on a Linux host** — the Metal arm needs its `--target`, and with it clippy analyses
the `backend-metal` code without an Apple machine:

```sh
cargo clippy -p reims-vgpu --target aarch64-apple-darwin --all-targets --no-default-features --features backend-metal -- -D warnings
cargo clippy -p reims-vgpu --all-targets --no-default-features --features backend-vulkan,host-window -- -D warnings
cargo clippy -p reims-vgpu --target x86_64-unknown-linux-gnu --all-targets --no-default-features --features backend-vulkan,host-window -- -D warnings
```

A commit touching `crates/reims-vgpu-efi` — its own workspace, so the commands above do not reach it
— adds two more, both from the crate directory. `--all-targets` cannot be used on either: the bin is
`#![no_main]` with the `uefi` crate's panic handler, so building a libtest harness for it collides
with `std`'s (E0152) on every target.

```sh
cargo clippy --target x86_64-unknown-uefi -- -D warnings   # the shipping artifact
cargo clippy --profile test --lib -- -D warnings           # the host-runnable lib and its tests
```

Expect zero from all three. **`scripts/feature-matrix` does not cover this**: it runs `cargo check`,
so its `warnings=0` is a rustc count and it cannot see a clippy lint on any arm. That gap plus a
"the Metal command is Apple-only" line that used to sit here is how a `clippy::question_mark` in
`runtime/draw/mod.rs` survived several commits that each said "clippy clean" — every one of
them was clean on the arms it ran, and nobody on a Linux host ran the Metal one.

Do not hide warnings, skip an affected arm, or commit a dropped test
count without calling it out — and **do not read "clippy clean" in a commit body as covering every
arm**; it means the arms that commit ran.

One standing exception, carried by `#[allow]`s at the module declaration that states the reason.
`backend::metal::error::Status` is large by design — the payload is what makes each refusal name the
check that refused, and it is `Copy` and compared by value at hundreds of sites — so
`result_large_err` and `large_enum_variant` are exempted there. **A new error type that is large for
no such reason should still be boxed**, not added to the exemption.

### Never transmute a guest ordinal into a Metal enum

The `MTL*` types are fieldless `#[repr(u64)]` enums, so producing one whose discriminant is not a
declared variant is **undefined behavior, not a decode error** — the same rule
`reims-vgpu-wire`'s invariant 4 states for wire structs. A decoded guest value is an arbitrary
`u32`, so `transmute` is never the conversion.

`backend::metal::mtl_enum` is the only way across: name every variant, get `None` for anything
else, turn that into a typed refusal. Add a table there rather than a cast, and read that module's
doc first — two of these enums have interior holes, so a `<= max` range check is not a substitute,
and `MTLStepFunction`'s names in `metal` 0.33 are not Apple's.
