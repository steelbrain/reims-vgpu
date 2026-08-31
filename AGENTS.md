# AGENTS.md

This project emulates Apple's paravirtualized GPU. The unmodified macOS guest supplies the driver;
QEMU and the Rust backend decode its command stream and execute it through Metal or Vulkan.

Nested `AGENTS.md` files apply within their directories.

## Build the contract into the structure

- Implement behavior from decoded fields, layouts, calling conventions, or host capabilities. If a
  decision-affecting contract term is unknown, recover it or return a typed refusal; do not guess.
- Put each invariant in the type, resolver, transaction, or state machine that owns it. Change the
  owner when it cannot express the invariant instead of adding call-site flags, duplicate lookups,
  fixup passes, or special cases around it.
- Parse guest ordinals once into total Rust types and carry those types. Keep page geometry explicit
  through `page_shift` or `page_size`, and derive constants from the contract.
- Prefer making invalid states unrepresentable, deriving duplicated values from one source, and
  testing behavior at the owner's boundary.
- Gate optional paths on structural capabilities, not device names. Environment overrides may
  narrow capability but never widen it.

Conformance records compatibility; it does not define an unknown API contract. Static inspection
of locally available binaries is acceptable for recovering missing contracts, but third-party
bytes, disassembly, extracted assets, and binary provenance must remain local and uncommitted.
Persist only the resulting field, layout, lifetime, ordering, or calling-convention contract.

## Ownership boundaries

- `vendor/qemu`: thin QEMU device shim for QOM, MMIO/BAR, IRQ/MSI, display integration, and
  `HostOps` plumbing.
- `crates/reims-vgpu`: device model, decode, mapping, scheduling, command execution, presentation,
  and backend policy.
- `crates/reims-vgpu-contract`: backend-neutral layouts, formats, geometry, page arithmetic, and
  contract refusals.
- `crates/reims-vgpu-wire`: derived wire views; its nested instructions also apply.
- `crates/reims-vgpu-env`: the single parse and declaration point for environment switches.
- `crates/reims-vgpu-observe`: typed observations and refusals. It describes decisions but does not
  select behavior.
- `conformance`: native-oracle and guest-visible compatibility cases.
- `vm`: rail-selected, snapshot-reverting boot harnesses.

Product logic belongs in Rust. C and Objective-C connect QEMU to Rust and must not reconstruct
policy from multiple Rust queries. Shared C/Rust constants need a `qemu::abi::header_define` test.
No panic may cross an FFI boundary.

Guest RAM bounds and provenance belong to `runtime/guest_ram.rs`; extend `GuestRamImport` or
`GuestSlice` instead of exposing raw pointers and offsets. Resource state that represents guest
work follows the contract-owned guest lifetime. Do not silently evict it with arbitrary cache
bounds; refuse excess work explicitly when the contract provides no lawful loss.

For asynchronous work, the owning transaction must retain inputs through host completion, make
results visible before the completion word or interrupt, and prevent callbacks or memory access
after the guest may release or reuse resources. Submission is not completion.

Unknown, dropped, rejected, degraded, or unsupported guest work must produce a typed reason on the
always-on failure channel. Expected not-ready control flow stays quiet. Read environment variables
only through the environment owner.

## Supported pathways

| Pathway | Host | Guest | Attach | Page shift | Backend |
|---|---|---|---|---|---|
| x86 macOS | Linux x86_64 KVM | x86_64 macOS | PCI | 12 | Vulkan |
| arm64 macOS | Apple Silicon HVF | arm64 macOS | sysbus | 14 | Metal |
| arm64 macOS | Apple Silicon HVF | arm64 macOS | sysbus | 14 | Vulkan/MoltenVK |

Do not generalize observations between architectures, backends, memory topologies, host GPU
classes, or guest rails. Vulkan 1.2 is the baseline; newer functionality requires a
capability-gated fallback. Host-pointer import is optional, and guest-visible semantics must be the
same on imported and copying paths.

## Working and verification

Use a workflow proportionate to the change:

1. Reproduce or otherwise identify the behavior being changed.
2. Establish the relevant contract and its owner. Use focused instrumentation when necessary.
3. Add an owner-level regression test when the invariant is testable there.
4. Implement the invariant in its owner.
5. Format and run the focused and affected tests. Run GPU-touching Rust tests serially.
6. For guest-visible translation or rendering changes, run the relevant conformance case and live
   pathway. Use the broader conformance suite when the change can affect unrelated cases.

Mechanical changes do not require VM or conformance work unless they can change behavior. A defect
seen only in an unmodified guest requires a live guest check. Intermittent failures require repeated
runs before claiming a fix. State what was actually verified without treating one pathway as proof
of another.

For VM work, select and report the rail explicitly. Ensure an older VM cannot answer probes, clear
`/tmp/reims-vgpu-fail.log` before a new evidence run, and preserve the useful crash or serial output
before rebooting. Use host-driven input and host-owned frame capture for visual interactions.

Follow `conformance/README.md` when changing or running conformance. Native results establish the
expected Metal behavior; existing classified guest failures are compatibility debt, not permission
to add or hide regressions.

Run formatting, tests, clippy, feature checks, documentation checks, and live validation only for
the workspaces, features, APIs, and pathways the change can affect. Do not weaken warnings or tests
to make a gate pass. Treat every commit as a release candidate: clear regressions exposed by the
change before committing rather than recording knowingly broken intermediate states.

## Repository safety

Existing dirty changes belong to the user. Preserve them and avoid unrelated edits. Do not use
checkout, switch, stash, reset, or restore to manufacture a baseline in the shared checkout; use an
isolated copy when a control build is needed. Create a new commit by default; amend an existing
commit only when the user specifically requests or authorizes it.

Do not commit guest images, firmware, captured shaders, AIR, SPIR-V, disassembly, extracted assets,
or other third-party binary material. Keep investigation artifacts in ignored locations. Commit
only task-related source and state validation results narrowly.
