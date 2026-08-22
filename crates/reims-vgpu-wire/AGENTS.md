# AGENTS.md — reims-vgpu-wire

Procedure for extending this crate until every operation Apple's serializer can emit has a view.
Read `README.md` first for what the crate is and why the oracle works; this file is the how.

The repository-root `AGENTS.md` applies, including its rule on what belongs in an `AGENTS.md`:
durable procedure here, findings in the view's own doc, the commit body, or `kb/`. Where this file
is stricter than the root, it wins, because here the contract is executable and there is no excuse
for a guess.

**The work queue is not in this file, and must not be written into it.** The manifest's coverage
line, `untriaged()`, the capability attribution table and the divergence instrument's gap map are
all generated, and each is current by construction. Read those.

## Prime directive

**Derive, never guess.** Every offset, width, shift and mask in this crate must trace to one of
exactly two sources:

1. **The Objective-C type encoding**, read at run time from the loaded serializer's own metadata
   (`method_getTypeEncoding`). This fixes widths and field order.
2. **Perturbation of a live serializer.** Change one input, serialize, diff the bytes. This fixes
   meaning.

Nothing else counts. Not "it looks like a stride", not "the value is plausible", not "the other
descriptor does it this way", and above all not a corpus: matching a captured byte pattern is how a
translation layer starts passing tests without implementing the contract.

Both produce *fixtures* — bytes the serializer actually emitted, replayable in CI. A structure no
serializer record carries cannot be derived this way at all; it belongs in Tier 2 and must say so.

If you cannot derive it, **say so in the type**. See "Naming the unknown".

## Invariants

These are load-bearing. A change that breaks one is wrong even if it compiles and the tests pass.

1. **`#![no_std]`, no allocation, no dependencies.** A view that must allocate is a view that copied,
   which defeats the crate. `serde_json` is a dev-dependency of the integration tests only, and
   `tests/` is a separate crate, so nothing reaches the staticlib QEMU links.
2. **Every wire struct is align-1.** Build them from `crate::le` scalars only. Never put a bare
   `u16`/`u32`/`u64` in a `#[repr(C)]` wire struct. `Wire::ASSERT_ALIGN_1` fails the build if you do
   — do not silence it, and never reach for `#[repr(packed)]`, which trades this problem for
   unaligned references to fields.
3. **Every constructor is fallible.** The bytes are guest-controlled. Never add an unchecked cast,
   and never `unwrap()` a view in library code.
4. **No enums, `bool`, `char`, `NonZero*` or references in wire structs.** An out-of-range guest
   value in any of those is undefined behaviour, not a decode error. Store the raw scalar and expose
   a fallible accessor. A field is safe only if it is a `le` scalar, `u8`/`i8`, an array of those,
   or another `Wire` type; anything else is unsafe until argued otherwise. A source scan used to
   enforce this and is gone, so know what the compiler does and does not do for you here:
   `ASSERT_ALIGN_1` already rejects a bare `u32`, a `char` and a `#[repr(u32)]` enum because those
   are over-aligned, so the cases that reach this rule are the **align-1** ones — `bool`,
   `NonZeroU8`, a `#[repr(u8)]` enum. All three compile, and nothing will stop you.
5. **`unsafe impl Wire` needs a comment** naming why both requirements hold. The existing ones are
   the template.
6. **Never widen a field to make a value fit.** If a value does not fit, the layout is wrong;
   re-derive it.
7. **A view is not ownership.** Everything borrows the caller's buffer. Do not add a method
   returning an owned value to "make it easier" — that is a copy, and the crate's whole claim is
   that it does not make them.

## Adding an operation

Work one selector at a time, end to end. A half-added operation is worse than an absent one because
the manifest will claim it.

### 1. Find its shape

Run the inventory and read the type encoding for the selector:

```sh
scripts/wire-oracle/wire-oracle.sh --inventory
jq -r '.classes[] | select(.class=="PGSerializerRenderCommandEncoder")
       | .selectors[] | "\(.selector)\t\(.type_encoding)"' \
  crates/reims-vgpu-wire/fixtures/inventory.json
```

Every selector carries its encoding, straight from `method_getTypeEncoding`, so this costs nothing
and settles argument widths before a byte is captured. It is also the cheapest check on a view you
have already written: `the_type_encodings_agree_with_the_widths_the_views_read` compares the two
derivations for the records that carry their arguments verbatim.

Do not read it as the *wire* layout. It is the **API's** widths, and the serializer is free to
narrow them — every draw argument is declared 64-bit and the compact draws put them on the wire at
16. What the encoding settles for those is signedness: `baseVertex` is `q` where every count beside
it is `Q`, which is why it is the one field read through a signed scalar.

Structure characters map as: `C`=u8, `S`=u16, `I`=u32, `Q`=u64, `bN`=an N-bit bitfield,
`^{...}`=pointer to struct, `@`=object, `*`=char*. Bitfields pack from the **least significant bit**
of the containing word, in declaration order.

A `^{...}` first argument means the *argument* does not go on the wire. It says nothing about
whether a record is emitted — a selector can stage bytes through the command stream and emit a
record naming the buffer and offset. Drive it anyway.

### 2. Capture it

Add a case function to `oracle/oracle.m` beside `textureCases`. Requirements:

- **Exactly one operation per case.** `caseFromCapture` refuses otherwise, so a selector that emits
  several needs `addEncoderCaseSplit`, which drives it once and records one fixture per operation
  with the count asserted rather than discovered.
- **Expectations come from the input object, not the buffer.** Read `(unsigned)d.someProperty` off
  the descriptor. Never transcribe an enum ordinal by hand and never read the value back out of the
  bytes you are checking; either makes the fixture agree with itself no matter what.
- **Distinctive values.** `0x1111`, `0x2222`, `0x3333` — a byte that lands somewhere unexpected
  should be recognisable on sight. Never `0` or `1` for a field you are trying to locate.

### 3. Perturb it

One property per case, everything else at baseline. You need enough cases that **every field you
intend to name has moved at least once**. A field that never moved is not derived, no matter how
obvious it looks — and a field that is *constant* across every fixture may be constant because the
input never varied.

Four failure modes to watch:

- **Coupled properties.** Metal rejects some combinations, so `sampleCount = 4` needs
  `textureType = 2DMultisample`. Change both and you have derived neither cleanly — add a third case
  that moves only the type, so the difference is attributable.
- **Silent clamping.** Metal may normalize a property before the serializer sees it. That is why
  expectations are read from the descriptor *after* setting it: the value Metal kept is the value
  that should appear.
- **Arguments with nothing to read back.** An `NSRange` is not a property, so an expectation for it
  can only be what was asked — and Apple's serializer *truncates* a plural bind's range at the
  stage's argument table, so what was asked is exactly what the record does not hold. A case whose
  answer is not what it asked for must say `requested` in its expectations rather than `count`, and
  the two generic `reads_back_what_metal_was_asked_for` tests skip on that key. Writing the wire's
  own answer into an expectation is what makes a fixture agree with itself.
- **A condition in the selector's name.** `maybeEmitSerialBarrier` emits or does not depending on
  the encoder's dispatch type. When a name states a condition, the second case is the one that finds
  the condition, and the state it puts the encoder in has to be set through the API rather than
  assumed. One case would record the opcode and get the contract wrong.

### 4. Write the view

In `src/ops/<family>.rs`, following `ops::texture`:

- `pub const OPCODE_*: u32` and `pub const *_TOTAL_LEN: u32`.
- A `#[repr(C)]` body struct of `le` scalars, plus `unsafe impl Wire` with its safety comment.
- Accessors for packed subfields, each documented with **the observations that derived it** — the
  actual value pairs, as `ops::texture` does ("BGRA8Unorm→80, RGBA8Unorm→70, R8Unorm→10").
- A `pub fn <family>(op: &Op) -> Result<&Body, WireError>` entry point.

Add a unit test asserting `size_of::<Body>() + OP_HEADER_LEN == TOTAL_LEN`. That one line catches
most layout slips immediately.

### 5. Wire it into the manifest

Change the row in `src/manifest.rs` from `Unimplemented` to `Covered { module: "ops::<family>" }`
and record the opcode. If the selector emits nothing, use `Excluded` with a reason that says *why*,
not just "n/a".

### 6. Extend the fixture test

Add the per-field assertions to `tests/oracle_fixtures.rs`, following
`every_texture_fixture_reads_back_what_metal_was_asked_for`. Assert opcode and total length against
the crate's own constants — that is what catches a contract change on a new macOS build.

### 7. Verify

```sh
scripts/wire-oracle/wire-oracle.sh --all
REIMS_WIRE_FIXTURES_REQUIRED=1 cargo test -p reims-vgpu-wire -- --test-threads=1
cargo clippy -p reims-vgpu-wire --all-targets -- -D warnings
```

Confirm the coverage line moved. If it did not, the manifest was not updated and the work is not
done.

Then run the device against the same fixtures, because a new fixture is a new test of `reims-vgpu`:

```sh
REIMS_WIRE_FIXTURES_REQUIRED=1 cargo test -p reims-vgpu --no-default-features \
  --features host-window --test wire_fixtures_reach_the_decoders \
  -- --nocapture
```

## Capabilities

The serializer carries sixteen `-setSupportsX:` / `-supportsX` capability pairs and **all sixteen
default off** (measured every capture under `capability_defaults`). Several selector families emit
nothing at all until their flag is forced, so **silence measured at the default state is a statement
about this harness, not about Apple.**

The capture handles this and you should read its output rather than guessing:

- A sweep pass forces all sixteen before the first case and publishes `silent_with_every_capability`.
  A selector silent at default but not under the sweep is capability-gated, and
  `every_silent_selector_is_silent_under_every_capability` fails if it claims `EMITS_NO_OPERATION`.
- Each flag also gets a pass alone, so `capability_attribution` names *which* flag unlocks a given
  selector. Read that table rather than trying flags in turn;
  `every_capability_gated_selector_names_the_flag_that_unlocks_it` prints it.
- `capability_content_deltas` diffs each pass's records against the default pass byte by byte, so a
  flag that changes what an existing record *contains* is caught too.

Rules that follow:

- **Before writing `Excluded { EMITS_NO_OPERATION }`, drive the selector with its flag forced.**
- **Restore the flag.** `withCapability` forces one and restores it after; a serializer left in a
  different state changes what every *later* case emits, and that failure looks like a layout error
  somewhere else entirely.
- **Drive the family, not the selector whose name matches the flag.** A flag's name is not a
  reliable guide to which selectors it switches.
- **A capability can change the *opcode*, not just the length.** Reading a capability delta as "the
  record grew" is wrong in a way that matters: a decoder dispatching on opcode is safe by
  construction, one dispatching on selector-and-length is not.
- The sweep's *fixtures* are discarded — capability state can change what a record contains, and the
  fixtures this crate pins must come from a serializer in its default state — so only the outcome
  lists are kept.
- Three capability pairs are read-only (`CorrectBaseVertex`, `OpenGL`, `SharedTextures`), so a
  family gated on one of those would still read silent. That caveat applies to every silent row.

## Five capture outcomes

`cases` is bytes. `unsupported` is Apple asserting (→ `Excluded`, `REFUSED_BY_SERIALIZER`). `silent`
is a selector that ran and wrote nothing (→ `Excluded`, `EMITS_NO_OPERATION`). `crashed` is *our
stub* faulting, which is evidence about the harness and none about Apple (→ stays `Unimplemented`).
`multi` is a case that produced a different number of records than it claimed; it should stay empty,
a selector on it may not claim `Covered`, and a test enforces that.

**These are outcomes of a *case*, not of a selector, and the difference is load-bearing.** A
selector driven twice can land in two of them at once. One that appears in both `cases` and `silent`
is a **conditional emitter** — a third thing from "silent always" and "silent because a capability
is off" — and its row is `Covered`.
`every_excluded_row_that_claims_silence_still_gets_it` knows the state by name and only allows it
where a case actually observed a record.

Two rules about `Excluded`, which is where a wrong claim hides because nothing about it looks like a
gap:

- **Every `Excluded` row must come from a case that ran.** A block of refusals next to a selector is
  not evidence about that selector — a family is not uniform, and two adjacent optional Metal
  features can be one absent and one complete.
- **An assertion inside Apple's serializer is a refusal of this harness's inputs, not a claim about
  Apple.** Driving with different capability state or a valid input can turn a `REFUSED_BY_SERIALIZER`
  row into a record.

Triage every selector. One with no manifest row is indistinguishable from one that does not exist,
which is what `untriaged()` counts.

## The divergence instrument

`crates/reims-vgpu/tests/wire_fixtures_reach_the_decoders.rs` runs every fixture through the matching
decoder in `reims_vgpu::runtime::decode`. It is the reason to keep adding fixtures even for records
this crate already understands, and it has found real guest-work losses that a person reading one
file beside the other did not.

Its verdict has three states and the distinction is the whole design: a record that decoded, an
opcode the decoder does not implement (a **gap**, reported not failed), and a well-formed record
refused **for its shape** (a **failure**, because that is a layout this project has wrong).

Its second half reaches a bug shape a reading cannot.
`no_decoder_reads_a_bit_apples_serializer_never_wrote` takes each fixture's measured `written_mask`,
repaints every **unwritten** bit to all-zero and then to all-one, decodes both, and requires the same
answer. A decoder whose output moves read a byte the serializer never wrote — which in a guest is
stale ring, not data. That is invisible to a test that decodes Apple's buffer once, because a
capture arena is zero-filled where a ring is not. Treat its zero the way this project treats every
healthy zero: a *non*-zero is the finding.

Four rules for reading it:

- **An `Ok` is not necessarily a decode.** `render::Kind::OtherAccepted` is the catch-all for "no arm
  claimed this", and counting it as decoded hides whole families behind a passing run. It is a
  `NotImplemented` verdict here.
- **Before adding a `gap(...)` arm, grep for a decoder that reads the record under another name.**
  This device does not always meet a record off the command stream: it may arrive as an object-list
  entry dispatched on `object_type`, or wrapped in a request behind a task/reply header. Both are
  "the same bytes arriving another way" and neither is visible from an opcode table. The cheap check
  is that the decoder's own constants equal this record's opcode and length, whatever it calls them.
- **An exempted class is not a covered one.** `UNCOVERED_CLASSES` is empty and should stay that way;
  dispatch per opcode. A class-level skip takes the records that *do* have readers down with it, and
  a summarised class reads as one with nothing in it worth reading.
- **An instrument that compares outcome *lists* is weaker than one that compares the records.** When
  a sweep reports a healthy zero, check whether it could have seen the other outcomes at all.

An opcode on its gap map can often be closed by **decoding it and naming the loss precisely, without
implementing it** — the counter is what decides whether the implementation is worth building. Do
that first, and do not make the counter uniform across a family: a record that is semantically safe
to skip, one whose loss is a dropped draw, and one whose loss is *stale commands executing* need
different slugs, or a driven boot's reading is useless. Count a state change only when the guest
asked for something other than the API default, so the counter is a healthy zero; count lost geometry
unconditionally, because there is no default it could be sitting at.

## Naming the unknown

A field or bit you could not make move gets:

- a name beginning `unidentified_`;
- a doc comment stating **what has already been tried** and what it read;
- a doc comment stating **the specific experiment** that would settle it.

`SamplerBody::unidentified_flag_bits` is the template. This is not a
placeholder to tidy up later — it is the honest encoding of what is known, and it is what stops the
next reader from re-running an experiment that already failed.

Never name a field after what it "probably" is. `reserved`, `padding` and `flags` are all claims: the
first two say nothing will ever be there, the third says the bits are independent. If you have not
shown it, do not say it.

**Refusing a derivation is a legitimate outcome.** Where the choice lives in Apple's host
implementation and no capture of the command stream can see it — `fillBuffer:range:pattern4:` says
what repeats but not the pattern's phase — execute the case where every reading agrees and refuse the
rest by name, with a counter. Do not "finish" it by picking a reading; the non-zero count is the
argument for deriving the rule.

## Two kinds of test — do not conflate them

Unit tests in `src/` synthesize buffers from the crate's own constants. They prove the views are
self-consistent and **cannot** detect a wrong layout.

Only `tests/oracle_fixtures.rs`, running against bytes Apple produced, can. When you add an operation
you must add to **both**: the unit test for the arithmetic and the error paths, the fixture test for
the truth.

Never move Apple bytes into a unit test to make it run without fixtures. That commits third-party
bytes and converts the one test that could find a layout error into one that cannot.

## Traps

- **`length` is guest-controlled.** Always route through `op::op` or `OpStream`, which check it.
  Never compute `length - 8` yourself; it underflows on a malformed operation.
- **The FIFO header is not this header.** `reims_vgpu::runtime::decode::fifo` frames a different
  level with fields at the same offsets 4 and 8 meaning different things. Do not port constants
  between them.
- **Zero is not absence.** A field reading 0 in every fixture may be unexercised rather than unused.
  Design a case that would make it non-zero before concluding anything.
- **Poison is signal — read the mask, never the byte.** Bytes the serializer never wrote are
  uninitialized on the real wire; in a guest they hold whatever the ring last contained. Every case
  is captured twice, under `0xAA` and `0x55`, and XORed into a per-bit `written_mask`. A bit that
  agreed was written; a bit that disagreed was not, and the two fills are complements so nothing can
  agree by accident. A single fill cannot tell a serializer that writes `0xAA` from one that writes
  nothing, and cannot see a bitfield written inside a byte the serializer otherwise leaves alone.
  `PARTIALLY_WRITTEN` in `tests/oracle_fixtures.rs` is the whole map; a record absent from it is
  written end to end and the test fails rather than assuming.
- **The thing that moves is often a *width*, not an offset.** A `Q` argument can arrive sixteen bits
  wide; one field can be four bytes wide in a record's wide form and two in its narrow one; a
  one-bit flag can sit in a `u32` slot. Do not take a width from the type encoding without the
  written-bit mask beside it, and when two arms of one wire form disagree on a width, neither
  comment settles it — the mask does.
- **A record's declared length is not its written extent**, and a creation record is not guaranteed
  to carry the new object's ref.
- **A fixture family that shares a leading field cannot locate the header boundary.** Every
  object-creation record begins its payload with the new object's ref, so reading the header as 12
  bytes and as 8 produce identical field offsets for every one of them. Only a record with no object
  ref can tell them apart. When deriving a *framing* rather than a field, the case that settles it is
  the one whose payload shape differs most, not the one you have most of.
- **A hole in an opcode run is a selector nobody has driven yet, not a number Apple skipped.**
- **One workload proves one workload.** A selector that never appears in a guest boot is still
  contract fidelity and still gets a view. Absence of traffic is not absence of the command.

## Scope: what belongs in this crate

The goal is that **all wire-format interpretation lives here**. That is not the same as "all of
`runtime/decode` moves", because some of what those modules do is not format interpretation. Three
tiers, by what evidence is available.

### Tier 1 — oracle-verified. Moves, with ground truth.

Everything the userspace serializer emits: the selectors in [`INVENTORY`], covering object creation
(`PGSerializer`) and every encoder record (render, compute, blit, info). This is the bulk of
`decode/{render,compute,blit,event,stream}.rs` plus the creation half of `decode/resource`.

`runtime::exec` decodes exactly these, so this tier is where the crate earns its place. Follow
"Adding an operation"; every entry gets a fixture.

### Tier 2 — moves, but with no oracle. Mark it.

Structures no serializer record carries, so no fixture can pin one. Their format comes from the
device contract instead, and nothing here can be replayed in CI:

- **The guest GPU page table** ([`crate::page_table`]) — the worked example.
- **FIFO packet framing** (`decode/fifo.rs`) — rings, doorbells, completion stamps.
- **Device-side descriptors reached by GVA** — the type-4/5/11 IOSurface descriptors and the
  116-byte texture descriptor `decode_texture_descriptor` reads. That 116-byte record is a
  *different structure* from the 36-byte creation payload in [`crate::ops::texture`]; they are not
  two readings of one thing.

These still belong here — the align-1 and bounds-check discipline is worth as much on unverified
formats as on verified ones — but the bar for what may be written is different, so:

- **Say in the module doc that no oracle backs it**, and what the layout does rest on.
- **Its manifest row must not claim `Covered`**, which means "a fixture pins it"; add a distinct
  state rather than stretching that one. The manifest is selector-indexed and mostly does not
  describe this tier at all — do not invent rows for structures with no selector.
- **Where the structure has write guards, mirror them in a test helper** (see
  [`crate::page_table::Builder`]) so tests walk trees assembled by the format's own rules rather
  than trees written to satisfy the walker. The two agree only if the walker is right.
- **A constant derived on one pathway is that pathway's.** The ten-bit index mask is x86's fan-out;
  arm64e's is twelve. Do not generalize one to the other — derive the relationship instead, as
  `Geometry::index_bits` does. What settles an arm64-only question is an arm64 **boot**.

### Tier 3 — resolution stays. Its formats still move.

The line is **not** drawn by subject matter, and not by module. It is one question: *can this be
answered from bytes the crate can reach?*

Needing guest memory is not the same as needing the device: [`crate::mem::GuestMemory`] is a
two-method trait the device implements, and with it the *algorithms* move while the memory stays
behind them. So:

- **A layout is always in scope**, including a layout that lives inside a function that stays.
- **A traversal is in scope when its only device input is "read this address."** The page-table walk
  is the worked example.
- **What stays is what needs device state**: a cache, a lifetime, a resource table, a residency
  decision, a refusal channel.

So several modules split rather than staying whole:

| Moves | Stays |
|---|---|
| x86 PTE format *and* the multi-level walk (`page_table`) | the translation cache, and mapping a walk failure to a typed refusal |
| object-list entry: 12 bytes, `[type\|desc_len]` + `desc_gva` | which task's list to read, and the lifetime of what it names |
| type-4 descriptor layout | the 256-task search for which list holds it |
| IOSurface page-table entry | `build_table_plan`'s `MappingInternal` field chase (arm64 only) |

The rule for `GuestMemory` implementations is in its module doc and is worth repeating because this
crate has no way to enforce it: **one implementation serves exactly one address space.** Do not write
one that inspects the address to decide whether it is virtual or physical. `reims-vgpu` has already
shipped one bug from mixing the two.

Applying the test honestly moves *more* than a module-level split would. Do not leave a layout behind
because the function around it stays.

### What gates a migration

The lifetime question is **settled**: `render::Command` is never stored in a struct field or a `Vec`
— `handle_render_record` decodes it, matches on `cmd.kind`, and drops it, and `handle_blit_record`
has the same shape. Borrowed views can replace owned decode without restructuring the call sites.
Variable tails (bind tables) become `view_slice`, which is what `apply_binds` already takes a slice
of.

Highest value first: drive the capability-gated families, then close what the resulting counters say
is actually being lost, then the remaining object-creation family, then Tier 2 marked as unverified.
Which selectors are still open is generated — read the attribution table and `untriaged()`, not a
list written here.
