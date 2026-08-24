# reims-vgpu-wire

Zero-copy views over the Apple paravirtualized GPU wire format, derived from
Apple's own serializer rather than inferred from captures.

## Why this exists

`reims-vgpu` reverse-engineers a command stream that an unmodified macOS guest
produces. Until now, the only way to check a layout was to boot a guest, drive
it, and see whether the picture looked right. That has two problems the project
has run into repeatedly:

- **Success is silent.** Every decoder in `runtime::decode` emits nothing when
  it succeeds, so "no failure line for opcode X" is not evidence that anything
  was decoded correctly — or at all.
- **There is no denominator.** "Are we handling every command?" had no answer,
  because nothing enumerated the commands that exist.

Both are fixed by the same discovery: **the serializer runs on the host.**

## The oracle

`AppleParavirtGPUMetal.bundle` — the guest-side userspace driver that turns
Metal calls into this wire format — ships on every macOS install, including
bare-metal Macs that have no paravirtualized GPU. Three facts make it usable:

1. It ships **x86_64 and arm64e** slices. There is no plain-arm64 slice and
   third-party arm64e needs a preview-ABI boot-arg, so the harness is built
   `-arch x86_64` and run under Rosetta.
2. `-[PGSerializer initWithDevice:objectRefAllocator:]` takes `id<MTLDevice>` —
   a **protocol** — so the host's real GPU satisfies it. `AppleParavirtDevice`
   does need an `AppleParavirtGPU` IOKit service that bare metal lacks, but
   `PGSerializer` sits below it and never asks for one.
3. `PGSerializerAllocator` requires exactly one method,
   `-allocateOperationBytes:(size_t)` → `char *`. Implement it and the
   serializer writes every operation into your buffer, and the size it asks for
   is the operation's true length.

So we can call a Metal API, get back the exact bytes a guest would put on the
wire, and check our views against them — in milliseconds, with no VM.

## What it does with that

**Layouts are derived twice over.** Field *widths* come from the Objective-C
type encoding, which Apple ships in the binary's metadata: the texture
descriptor is declared `^{?=b4b1b1b1b1b8b16IIISSSSQ}`, which fixes the bitfield
split and the scalar sequence without guessing. Field *meanings* come from
perturbation: change one Metal property, serialize, diff the bytes, and whatever
moved is that property's home.

**Anything not derived is not named.** The private texture fields are named only
after independent perturbations moved them: `framebufferOnly`, `isDrawable`,
`writeSwizzleEnabled`, and `protectionOptions`. Inventing a plausible name is
how a guess becomes folklore.

**Exhaustiveness is a number.** `class_copyMethodList` enumerates Apple's whole
surface — currently 364 selectors across the serializer and its four encoder
classes — and `manifest.rs` records what we have done about each. The tests
print the gap on every run and fail when the host's surface stops matching what
the manifest believes.

## Reading it in memory, not decoding it

A view is a length check and a pointer cast. Two design points make that safe:

- **Align-1 everywhere.** Operations are variable length and the texture one is
  44 bytes, which is not a multiple of 8 — so consecutive operations put `u64`
  fields on 4-aligned addresses. A `#[repr(C)]` struct with a real `u64` has
  align 8, and taking `&Struct` from a 4-aligned address is undefined behaviour.
  Every field is a `#[repr(transparent)]` wrapper over its bytes instead, so
  structs are align-1 and every offset is legal.
- **The bounds check stays.** These bytes come from a VM guest, so every length
  on the wire is attacker-controlled from this process's point of view, and this
  process holds QEMU's address space. One comparison per operation is not
  parsing.

## Two kinds of test, proving different things

| | Where | Proves | Needs |
|---|---|---|---|
| Unit | `src/**/tests` | The views read what the layout constants say | nothing |
| Oracle | `tests/oracle_fixtures.rs` | The layout matches **Apple** | Apple host + fixtures |

The unit tests synthesize their buffers from the same constants the views read,
so they are self-consistency checks and cannot find a layout error. Only the
oracle tests can. Keeping the distinction sharp matters: a green unit suite says
nothing about whether we understand the protocol.

## Running it

```sh
# Apple host: regenerate ground truth (needs Rosetta)
scripts/wire-oracle/wire-oracle.sh --all

# Anywhere: unit tests. On Apple, also verifies against the fixtures.
cargo test -p reims-vgpu-wire -- --test-threads=1
```

With no fixtures present the oracle tests report `ignored`, one line each, with
the command that regenerates them. That is a build-time decision (`build.rs`
sets a `wire_fixtures` cfg), so a fixture-less run cannot read as coverage — it
used to print `ok` for 34 tests that returned on their first line. Set
`REIMS_WIRE_FIXTURES_REQUIRED=1` on any Apple host to make their absence fail
the build instead.

## Fixtures are not committed

They are Apple's serializer output, and `AGENTS.md` at the repository root
forbids committing third-party bytes. `fixtures/` is gitignored and regenerated
on demand. What the repository keeps is ours: the derived layout, the
expectations, the coverage manifest, and the harness that produces the bytes.

The expectations are also not read back from the buffer — each case records what
was set on the `MTLTextureDescriptor`, read from the descriptor object, so enum
ordinals come from Metal itself. A fixture whose expectations came out of the
bytes it checks would pass regardless of what the layout did.

## Status

Texture creation (opcode 1) was the worked example. The render encoder's whole
**draw family** followed: six selectors, twelve opcodes, because each draw has a
compact 16-bit encoding and a wide 64-bit one and the serializer picks by the
magnitude of the arguments. `ops::render` carries the pairing and the boundary,
and every one of the twelve has a fixture. Beside those are six state records
(scissor, viewport, cull mode, winding, blend colour).

Beside those are the render state records: scissor, viewport, blend colour, and
the four one-`NSUInteger` and two one-`float` records that share a shape.

Seven render selectors are `Excluded` because Apple's serializer *refuses* them
— `setPointSize:`, `setPrimitiveRestartEnabled:` and five more each fail an
assertion inside the encoder rather than emitting anything. The oracle drives
them every run and records what refused, so those exclusions are measured on
each capture rather than remembered.

The current coverage line, printed by the test suite:

```
wire coverage: 207 covered, 0 unimplemented, 153 excluded, 4 untriaged of 364 selectors
```

Every selector known to emit a fixed operation now has a view. The four remaining
untriaged selectors are the explicit manifest gap, not measured opcodes waiting
beside an implemented decoder.

## Relationship to `reims-vgpu`

This crate is the **layout authority** for serializer records that
`reims_vgpu::runtime::decode` consumes. Decode maps wire views into the
product model (`Command`, `Kind`, decline slugs, exec routing); it does not
re-declare opcodes or restate field layouts when a wire parser or view already
exists.

Covered families used today:

| Decode module | Wire module(s) | How it is consumed |
|---|---|---|
| `decode::blit` | `ops::blit` | `op` framing + family parsers (`copy_*`, `fill_*`, …) |
| `decode::compute` | `ops::compute` | `op` framing + family parsers (dispatch, binds, fences, …) |
| `decode::render` | `ops::render`, `ops::render_pass`, `ops::tile` | `op` framing + draw/state parsers and views |
| `decode::resource` | `ops::{sampler,depth_stencil,texture_view,icb,heap_texture,backed_texture,…}` | parsers for covered create records (sampler, depth/stencil, texture views, ICB, heap/buffer textures) |
| `decode::stream` | `ops::segment`, `OP_HEADER_LEN` | segment header types and record framing constants |
| `reims-vgpu-paging` page walk | `page_table` | GVA walk layout |

What stays local in decode (wire has no export):

- **FIFO** packet framing (`decode::fifo`) — a different level of the protocol
- **Event** opcodes `0x190`–`0x192` (no wire event module yet)
- **Compute residency** `0x86`/`0x87` (no `useHeaps:` / `useResources:` on the
  compute serializer surface measured here)
- **Unobserved segment type** `SEGMENT_TYPE_EVENT` (wire only names what the
  oracle has driven)

`op::OpHeader` — `[opcode u32][length u32]` — is the serializer record head.
It is *not* the FIFO packet header in `decode::fifo`, which frames a different
level with fields at offsets 4 and 8 meaning something else entirely.

Wire views also surface layout bugs that a boot cannot prove. The draw-family
derivation is the standing example: compact vs wide forms, index-type width, and
opcodes that used to fall through to `Kind::OtherAccepted` are recorded next to
the views that settle them in `ops::render`.

## Caveats

- **Rosetta.** Apple has signalled its eventual removal. The fallback is an
  arm64e build, which needs a boot-arg; neither the crate nor the fixtures
  depend on which route produced them.
- **One serializer version.** Layouts here were derived against
  `AppleParavirtGPUMetal` 64.4.7 (macOS 26.5). `PGSerializer` also exposes
  `initWithDevice:objectRefAllocator:deserializerVersion:`, so the format is
  explicitly versioned — a guest on a different build may not match, and the
  fixture provenance records the version so a mismatch is visible rather than
  assumed away.
