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
  `native/macos/`, `native/linux/` and `native/windows.rs`, the Vulkan
  crate's `os/macos.rs`, `os/linux.rs` and `os/windows.rs`, the graphics
  crate's `clock/`, `region/` and `wake/`, each `unix.rs` and `windows.rs`,
  the C API's `ffi/catch/{macos,linux}.rs`, Linux's of which Windows
  takes), which the module above them picks with `#[cfg(target_os = ...)]`
  (or `unix` and `windows`), rather than in `cfg`s spread through shared
  code. Code some OSes share and others do not
  goes in a module of its own that the index picks for those OSes (the
  window crate's `native/sdl_thread/`, SDL on the library's own thread, for
  Linux and Windows). It sits in the crates the OSes share: no crate is one
  OS's. A file the OSes share is the exception, where
  splitting cannot be avoided, and keeps its `cfg`s to a few whole items;
  index files that pick modules are exempt.
- **The Vulkan platform** (`RtPlatform`) answers as Metal does through
  metal2vulkan-runtime, and these rules decide where its code goes and how it
  gets fast:
  - **GPU memory is the runtime's.** rt allocates it, and the device maps
    its host-visible blocks into the guest: the pool's blocks and the
    guest's Private blocks are rt's (`pool::RtBlocks`), and the buffers and
    heaps the guest pins in them are ranges of them, at the guest's
    addresses. A buffer's `gpuAddress` is its Vulkan device address, and a
    `gpuResourceID` a descriptor index. Textures are device images, synced
    at the guest's events (an IOSurface's lock and unlock, `didModifyRange:`,
    `replaceRegion:`), with no page logs. An app's no-copy memory, which no
    block maps, is rt's import of its pages, by their pointer or as a
    dma-buf of their files, and only where rt takes neither a copy synced
    at the commits that use it (`rt_platform::copied`). One VkDevice per
    process serves everything, the window's presents included, and no path
    shares objects across devices. The guest's processes are therefore not
    isolated from each other on the device: every byte on it is
    guest-writable, and the host never parses or trusts host data it keeps
    there.
  - **Stalls are fixed on the Vulkan side**: in this crate (caches, their
    keys, when states are made) and in the runtime (translation speed,
    sharing, making ahead), never by making core or the device's lanes
    non-blocking, or by deferring or reordering their waits, since the guest
    sees where they wait. Compiles run concurrently only inside the synchronous wait
    that needs them, which joins every one before it returns, with two
    exceptions. A command buffer's dispatch pre-pass makes its dispatches'
    pipelines on the compile threads while its walk goes on, the walk's
    thread makes those no compile thread took at its first dispatch, and the
    walk's end drops those no thread took and joins those being made, before
    the message lets go of anything
    (`crates/reims-vgpu/src/graphics/decode/prepass.rs`). And a pipeline
    create's reply is posted once the platform's make has made every answer
    it carries, ahead of the driver's create of the state's main pipelines,
    which the state's finish then makes on the same compile thread before
    the pipeline is stored; until the store, the command buffers, indirect
    command buffer writes and function handle and table creates that name
    the pipeline wait for it (`crates/reims-vgpu/src/graphics/compile.rs`,
    "Order"). Otherwise nothing compiles in the background, and nothing is
    deferred or released early.
    Nothing the guest sees (IDs, addresses, stamps, reflection order, shared
    state) is decided inside a concurrent section; it is assigned in order,
    before or after. Each concurrency change is a small commit of its own.
  - **Speed comes from what every run gets**: fewer and smaller pipelines,
    sharing, concurrency inside a wait. Nothing is loaded or ordered by what
    an earlier run drew, and no measurement starts from a cache seeded for
    it. Three stores stand in for Metal's own caches across runs, as a Mac
    keeps its compiled pipelines on disk: rt's shader and pipeline caches,
    in `reims-vgpu-rt` under the user's cache directory
    (`crates/reims-vgpu-vulkan/src/rt_platform/device.rs`, `CACHE`), across
    a host's processes; the device's kept pipeline answers
    (`crates/reims-vgpu/src/graphics/answers.rs`), across boots; and, in
    the guest, the GL shader variants the bundle lowered, in each app's own
    cache directory beside Metal's (`guest/bundle/src/RgfxGLStore.hpp`),
    across the app's runs, as Metal's compiler service keeps what it
    compiled. Each holds only what the platform, the device or the
    generator gave before, under the identity of what decides it, so it
    changes no value the guest reads, only how soon the guest has it; none
    is ever seeded for a measurement.
  - **A workaround for the OS or a driver** lives in the runtime's code that
    calls it, never in core or the library's threads: a compile the system
    runs at its caller's priority is raised by the runtime, not by raising
    the library's threads'.
  - **What Metal does is the runtime's to build**, its private API
    included; what only our device has is this crate's. A gap
    in the runtime goes to it with a repro (the call, its descriptor or
    arguments, the error or the difference from Metal, the device) rather
    than being patched around here.
  - **Dependencies are used as released.** The Vulkan loader, the drivers
    (Mesa's KosmicKrisp on a Mac among them) and the others are never
    patched or rebuilt with changes: a fix one needs is
    proposed upstream, and the workaround lives here or in the runtime. A
    mark in the runtime that pays off only once upstream takes a fix stays
    if it changes nothing on the stock release.
- **Closed internal value sets are enums**, converted with `TryFrom<u32>` at
  the guest edge. A raw integer is kept only where the guest can write any
  value. There are no `pub mod x { pub const … }` integer-enum modules.
- **Refcounts**: a type whose drop has an observable effect (a Metal object,
  a host block, a device or display handle) never implements `Clone`. Another reference is taken with `.clone_ref()`, everywhere, API
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
- **Each degradation is logged once per boot, then counted.** A
  degradation, fallback, refusal or unsupported case writes its first line
  of each kind, and counts the rest:
  - The library's logger (`reims_vgpu_vulkan::core::logging`) writes a warn
    or error line, and an info line that names a `kind`, once per kind: the
    first at its level, each later one at debug, counted. The kind is the
    call site, and the class of the cause where the site names one,
    `log::warn!(target: "…", kind:% = error; "…: {error}")`: the cause's
    text up to its first digit, quote or `;`. A site with one cause names
    none.
  - What the platform has not built and what rt refuses are answered as
    Metal's call answers where it fails, each kind's first on a line of
    stderr naming the call, and counted: `rt-unbuilt-<piece>` and
    `rt-refused-<kind>` (`rt_platform::unbuilt`).
  - At each reset of the device, at the process's exit and when the host
    asks for them (`rvg_write_counts`, which QEMU's device calls when its
    `x-write-counts` property is set, as `reims-ctl report` does on a
    running VM), two lines give the counts. `reims-vgpu: degradation counts: <code> <n>, <code> <n>, …`,
    on stderr, sorted by code, is every kind relayed or counted since the
    process began: the platform's (`rt-unbuilt-<piece>`,
    `rt-refused-<kind>`, `rt-refused-texture-format-<n>` and the other
    `rt-…` codes), `out-of-host-memory`, and the reims-graphics device's completion
    rings' (`control-ring-full`, `control-ring-post-waited`,
    `vblank-dropped`, `vt-ring-full`, `queue-ring-full`,
    `queue-ring-post-waited`) and its
    decoder's (`two-ranges-kept-whole`, a command buffer dealt to two
    threads that the platform kept whole at a chunk's end, which fails at
    its stand-in; `decode-malformed`, `decode-unknown-command`,
    `decode-refused`, `decode-bad-reference`, `decode-bad-resource-id` and
    `decode-no-function`, the messages it did not carry out, by why
    (`DecodeError`; `decode-no-function` is a pipeline create naming no
    vertex function or kernel, as the bundle sends one where the device
    refused the library its functions were to come from, and no guest
    protocol error; one of a context closed before it came is owed
    nothing, and not counted), and
    `decode-faulted`, those a raise kept it from, each kind's first logged;
    `read-only-pages-buffers`, the no-copy buffers it made
    over pages their process maps read-only, and `read-only-pages-kib`,
    their bytes' sum in KiB, rounded up: a copy of each on every host,
    whose GPU writes are dropped (`rt_platform/buffer.rs`, "Read-only
    memory"); `no-copy-buffers`, the no-copy buffers it made, and
    `buffer-escaped-gpu-address`, `-argument-encoder`, `-resource-id` and
    `-residency-set`, those whose identity the guest noted escaped, by the
    act, with `buffer-escaped-read-only-pages`, those of them over a copy of
    their pages, and `escaped-copy-commits`, the command buffers begun of a
    process that held such an escaped copy: no degradation,
    docs/graphics-kext/design.md, "A buffer's escaped identity"), its VA-API
    video's on the rt platform on Windows (`rt-video-gpu-copy-declined`, a
    picture the CPU copied, the GPU's copy declined;
    `video-pictures-undrawn`, a picture whose draw into a cache's planes
    over Direct3D 11 textures failed, which the CPU's copy landed there
    instead; `video-pictures-drawn`, a picture drawn into a cache's planes
    over Direct3D 11 textures, and `video-drawn-kib`, the KiB of their
    crops: one GPU pass a picture that a Mac does not make;
    `video-drawn-spares-used` and
    `video-drawn-spares-missed`, such a cache's surfaces that used a spare
    made ahead and those made on the guest's ring-0 thread,
    `video-drawn-spares-waited`, those that waited for the spare being made
    for them, with `video-drawn-spares-wait-ns` their nanoseconds waited and
    `video-drawn-spares-wait-max-ns` the longest wait, and
    `video-drawn-spares-made` and `video-drawn-spares-dropped`, the spares
    made and those dropped untaken: no degradation), its VA-API video's
    on the rt platform on Linux (`video-pictures-gpu-copied`, a picture
    the GPU copied into its output's device caches from a scratch the
    driver's video processing wrote, and `video-gpu-copied-kib`, the KiB of
    their crops: one GPU pass a picture that a Mac does not make), its NVDEC
    video's on Linux (`video-pictures-gpu-gathered`, a VideoToolbox
    decoder's 4:4:4 picture the GPU wrote into its surface's two planes
    from the decoder's scratch: one GPU pass a picture that a Mac does not
    make) and its
    decodes' (`video-landing-failure-carried`, a picture whose copy or
    draw into its output's caches failed after its command buffer
    completed at its decode, the session's next decode failing in its
    stead), the rt
    platform's on Windows (`rt-d3d11-surfaces`, the surfaces over Direct3D
    11 textures it made, and `rt-refused-d3d11-surface`, those it refused),
    and the contexts' Private
    takes' (`private-take-queued`, a take that waited behind its context's
    earlier one; `private-take-sourced`, one its worker took from the
    source, and `private-take-sourced-us`, the source's time for them in
    microseconds; `private-take-span` and `private-take-span-us`, those of
    them longer than the guest's 16 MiB block, an object's own span, and the
    source's time for them; `private-take-lane-us-most`, the longest a take
    held the thread that carried it out), `managed-at-copy-unkept`, the
    `Managed` buffers' creates at their CPU copies' addresses that the
    platform made at others, kept there (docs/graphics-kext/design.md,
    "Managed buffers at their copies"),
    `destroys-at-rt-address-in-flight-<kind>`, the objects made at the
    platform's own address, a pin it did not keep or none, whose
    `OBJECT_DESTROY` came while a command buffer of their context was posted
    or running, by the kind of make (a diagnostic, no degradation:
    `graphics/decode/unkept.rs`, "Destroys in flight"), the trims of the
    heap after heavy pipeline makes on Linux with glibc (`heap-trims`,
    `heap-trim-us`, their time in microseconds, and `heap-trim-returned-kib`,
    the resident KiB they handed back: no degradation,
    `os/linux/heap_trim.rs`), the display's
    (`display-view-refused`, a draw of the display that kept the window's
    last frame, or a read of it that answered black, because the platform
    refused the display's view of the texture it shows: `graphics/sink.rs`,
    "The display's view"), and the guest's slow answers'
    (`slow-answer-over-make`, a ring-0 answer that went on its ring a frame,
    16.7 ms, or more after the device read its command, during which a make
    of rt's pool ran, and `slow-answer-over-make-max-ns` the longest;
    `slow-answer-first-use-waits`, the compute dispatches' first-use waits,
    `rt-first-use-variant-waits` and their share of
    `rt-first-use-unfinished-waits`, that met any such answer: no degradation,
    `graphics/slow_answers.rs`), and the guest's pipeline creates' waits for
    their replies (`pipeline-create-replies`, the replies posted;
    `pipeline-create-reply-ns` and `pipeline-create-reply-max-ns`, each
    create's nanoseconds from the host's take of it off ring 0 to its
    reply's post, summed, and the longest; `pipeline-create-replies-over-16ms`,
    those of 16.7 ms or more; `pipeline-create-waiting-ns`, the nanoseconds
    during which at least one create waited, the waits' union: no
    degradation, `graphics/create_waits.rs`); a boot's degradations are
    read from its last such line. `reims-vgpu INFO reims_vgpu_vulkan::core::logging: lines
    repeated since the last count: <n> more of <file>:<line>[ [<kind>]]
    (<target>: <the first line's message>); …` is how many lines of each
    kind the logger wrote at debug since its last such line.
  - No line at info or above fires per present, frame, draw, command
    buffer, ring message, interrupt or other per-event path: such a line is
    debug or trace, under a target of its own where a tool counts it
    (`reims_vgpu::graphics::shown`, `::library` and `::queues`, which the
    lane gate's boots log at debug). Info keeps state changes (modes, the
    display between the scanout and a surface, starts and stops, a new kind
    of degradation) and summaries, at most once a minute.
  - QEMU's `reims-graphics-pci` says a failure the guest can repeat with
    `error_report_once`.
  - The guest's kexts log a failure a caller can repeat once a boot per
    kind (`rgfxFirstCalls`, `guest/kext/src/RgfxTrace.hpp`), and the Metal
    bundle once a process per kind (`rgfxLogFirst`,
    `guest/bundle/src/RgfxLogOnce.hpp`); the kind is the site, and a status,
    selector or reason where the site has several. Their summaries (the
    rings' room, the releases, the dormant blocks, the swaps, the surface
    copies) come at most once a minute, and a bring-up trace logs its first
    calls alone.
    The bundle's os_log lines go to its own subsystem,
    `ai.aneesiqbal.reims-graphics`, category `bundle` (`RgfxLog.hpp`), so
    `log config --subsystem ai.aneesiqbal.reims-graphics` sets the level
    they are kept and persisted at. Its Objective-C sources log through
    `rgfxLogLine`, at the default level, which the log store keeps, with
    the whole line public: an NSLog argument that is an object or a string
    made at run time reads `<private>` in macOS 26's log. Its lines start
    `RGFX:`.
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
