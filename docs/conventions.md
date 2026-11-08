# Conventions

The conventions the code follows. [AGENTS.md](../AGENTS.md) holds the rules
of work around them, and [architecture.md](architecture.md) section 17
repeats the ones code cites.

- **Names say what things mean**: `Raise`, `Held`, `SerialQueue`,
  `HostRange`.
  - No `PG` prefixes, no Objective-C class names, no
    `dealloc`/`init_with_*` names, and no selector-keyword method names.
  - No `_locked`/public pairs unless both are called.
  - The threads, queues and Metal object labels the library makes are
    `reims-vgpu.<role>` (`reims-vgpu.graphics-control`,
    `reims-vgpu.display-present`).
    They are diagnostics: no host or guest reads them.
- **Modules** are named by responsibility (`graphics/decode/cmdbuf.rs`,
  `memory/mapping.rs`, `display/pipeline.rs`), never after `-init` or an
  Objective-C class. One module style: `foo.rs` beside `foo/` (D44).
- **Code for one OS** goes in files of its own (the window crate's
  `native/macos/` and `native/linux/`, the Vulkan crate's `os/macos.rs`
  and `os/linux.rs`, the C API's `ffi/catch/{macos,linux}.rs`), which the
  module above them picks with `#[cfg(target_os = ...)]`, rather than in
  `cfg`s spread through shared code. It sits in the crates both OSes share:
  no crate is one OS's. A file both OSes share is the exception, where
  splitting cannot be avoided, and keeps its `cfg`s to a few whole items;
  index files that pick modules are exempt.
- **The Vulkan platform** answers as Metal does through metal2vulkan's
  runtime, and these rules decide where its code goes and how it gets fast:
  - **Guest memory in place**, as on Metal. Resources are packed into
    contiguous host views of guest memory (a Mach remap on macOS, `memfd`
    layouts on Linux) that Vulkan imports as host pointers, with linear
    images and buffers made over them. A copy is taken only where a named
    hardware rule forbids use in place, and only of the area drawn or
    changed. The driver is asked, not assumed: NVIDIA's drivers lack the
    core colour-attachment bit for linear images, and give it through
    `VK_NV_linear_color_attachment`.
  - **Stalls are fixed on the Vulkan side**: in this crate (caches, their
    keys, when states are made) and in the runtime (translation speed,
    sharing, making ahead), never by making core or the device's lanes
    non-blocking, or by deferring or reordering their waits, since the guest
    sees where they wait. Compiles run concurrently only inside the synchronous wait
    that needs them, which joins every one before it returns: nothing
    compiles in the background, and nothing is deferred or released early.
    Nothing the guest sees (IDs, addresses, stamps, reflection order, shared
    state) is decided inside a concurrent section; it is assigned in order,
    before or after. Each concurrency change is a small commit of its own.
  - **Speed comes from what every run gets**: fewer and smaller pipelines,
    sharing, concurrency inside a wait. Nothing is loaded or ordered by what
    an earlier run drew, and no measurement starts from a cache seeded for
    it.
  - **A workaround for the OS or a driver** lives in the runtime's code that
    calls it, never in core or the library's threads: a compile the system
    runs at its caller's priority is raised by the runtime, not by raising
    the library's threads'.
  - **What Metal does is the runtime's to build**, its private API
    included; what only our device has is this crate's. A gap
    in the runtime goes to it with a repro (the call, its descriptor or
    arguments, the error or the difference from Metal, the device) rather
    than being patched around here.
  - **Dependencies are used as released.** MoltenVK, SPIRV-Cross and the
    others are never patched or rebuilt with changes: a fix one needs is
    proposed upstream, and the workaround lives here or in the runtime. A
    mark in the runtime that pays off only once upstream takes a fix stays
    if it changes nothing on the stock release.
- **Closed internal value sets are enums**, converted with `TryFrom<u32>` at
  the guest edge. A raw integer is kept only where the guest can write any
  value. There are no `pub mod x { pub const … }` integer-enum modules.
- **Refcounts**: a type whose drop has an observable effect (a Metal object,
  an IOSurface, a host block, a device or display handle) never implements
  `Clone`. Another reference is taken with `.clone_ref()`, everywhere, API
  types included (D6). A derive that would copy one does not compile.
- **Order-bearing types** have an explicit `Drop` that takes their parts in
  the order the API shows (architecture 15.3). Field declaration order is
  never relied on.
- **Never guess a name** for guest records, the wire and copies of Metal
  structs. A field whose meaning nothing states is `field_xx` (its byte
  offset in lowercase hex), and a name that is ours carries
  `// name is ours: <evidence>`.
- **An odd behaviour** the guest or host can see carries a doc comment at
  the code that says what it is, and a test named for the behaviour
  (`denormal_depths_store_and_compare_as_metals`). It is kept as it is:
  changing it is a deliberate change of behaviour, in a commit of its own
  that changes its test.
- **Comments** say why, in terms of what the guest or host observes ("the
  stamp is in guest memory before its interrupt"), and may name a test.
- Tests sit in `mod tests` beside the code and are named by behaviour.
- **Integers**: std spells wrapping, overflowing, widening and sign
  reinterpretation (`wrapping_add`, `overflowing_mul`, `u64::from`,
  `cast_signed`). `core::num` holds the rest: `div0`/`rem0` for a divisor
  the guest or host can make zero, `Truncate::low_u32/u16/u8`,
  `ToUsize`/`ToU64` (lossless on the 64-bit target), `const_u64` for layout
  constants, `align_up_wrapping`, `add_u64_into_u32` and `refresh_16_16`.
  An integer result the guest sees goes through the rule of `num` that
  names it: a mask bit through `bit32`, `bit64` or `bit64_set` (the shift
  mod the width), a narrow field's widening through `zero_extend_16`,
  `sign_extend_16` or `sign_extend_32`, and a copy between aliased mappings
  through `forward_copy`.
  Division by a constant takes a `NonZero` divisor built in a const block,
  `x / const { NonZeroU64::new(8).unwrap() }`, which
  `integer_division_remainder_used` accepts. A float conversion is an `as`
  at its site under `#[expect(clippy::as_conversions, reason = "…")]`.
- Default rustfmt. A lint is lowered with `#[expect(lint, reason = "…")]` on
  the narrowest item, in place of an `allow` with a comment.
- **The option ROM's C** (`firmware/`) follows EDK2's style and the UEFI
  specification's names instead (architecture 11.4). Its registers carry
  the wire header's names (`RGFX_REG_DOORBELL`).
- Branches are prefixed `steelbrain/`. Commits follow AGENTS.md (emoji
  prefix, imperative, 72-column subject, body wrapped at 80 explaining why).
