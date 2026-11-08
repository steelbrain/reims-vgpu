# reims-vgpu: architecture

This is how `libreims_vgpu` is built: its crates, what each part owns, its threads and locks, how
it fails, where it meets Apple's frameworks and the Vulkan runtime, and the interfaces the host
reaches it through.

The library is the host side of the reims-graphics device (PCI 1234:5247), the GPU our own guest
drivers (`guest/`) drive in x86-64 macOS guests. The guest's Metal bundle serialises its Metal
calls onto the device's rings, and the library carries them out on the host's GPU: on Metal on a
Mac, and on metal2vulkan's runtime on Linux. The device's wire protocol and what the guest sees are
[graphics-kext/design.md](graphics-kext/design.md); its C API is [api.md](api.md).

What crosses the API, and the wire, is the contract: the C API's results and refusals; the host
callbacks with their arguments, their order and the threads the header names for them; the order
of operations on guest memory and on retained objects; guest memory and MMIO reads; interrupts, in
their order against the guest-visible writes; the display's frames, modes and cursor; GPU results;
every value that reaches the guest; and where a run dies or hangs. Everything else is free: which
library thread runs what where the header names no thread, internal locking, the retention
machinery where the order of releases the host sees is unchanged, Metal's internal call order where
no result or guest-visible value changes, and diagnostics (log lines, thread names, Metal labels,
crash messages, signals). Much of the structure below is there because the contract fixes it; each
such place says what fixes it, and section 16 collects them.

Where the rest is:
- `README.md`: what the project is; [development.md](development.md): how it is laid out;
- [api.md](api.md): the C API and the Rust API it wraps, and the thread each callback arrives on;
- [graphics-kext/design.md](graphics-kext/design.md): the reims-graphics device, its protocol and
  the guest's stack;
- [testing.md](testing.md): how the code is tested, and the tags a claim about Metal carries;
- [conventions.md](conventions.md): the coding conventions (17 repeats the ones code cites);
- [qemu.md](qemu.md): QEMU on the library, its build and VM boots;
- [lessons.md](lessons.md): traps that tests, reviews and boots found;
- [accuracy-ledger.md](accuracy-ledger.md): what the Vulkan platform's defaults trade against
  Metal's answers, and the switches that take Metal's;
- `crates/reims-vgpu-window/features.md`: what the device's window does for the person at the host;
- module `//!` docs: each area's ordering contracts and the odd behaviours it keeps, each naming its
  test, and the platform facts the area relies on.

Code comments that cite a decision (`D17`) or a section (`architecture 13.2`) cite this document.

## Contents

1. Decisions
2. Workspace and crates
3. Core's modules
4. The platform boundary
5. (retired)
6. Outcomes: expected, raise, death
7. Concurrency
8. Containers
9. The reims-graphics device
10. The Rust API (`reims-vgpu`)
11. The C API (`reims-vgpu-c`), the dist tree and the option ROM
12. QEMU
13. Tests
14. Diagnostics
15. Guest memory, lifetimes and drop order
16. What the contract keeps
17. Conventions

---

## 1. Decisions

The decisions code cites by number, each in force as written here. A number missing from the
table is retired and not reused.

| # | decision | why |
|---|---|---|
| D1 | One Cargo workspace at the repository root: the library's crates (`reims-vgpu-core`, `-vulkan`, `-window`, `reims-graphics`, `reims-vgpu`, `-c`), the host video crates (`reims-vgpu-video-*`), and `xtask`. `reims-vgpu-vulkan` is the library's `Platform`, over `metal2vulkan-runtime`, a Metal-shaped API on Vulkan: on Linux, and on macOS over MoltenVK, a build for development and tests (2.1, 2.2). | Each crate does one job the others cannot without breaking a rule below. |
| D2 | Modules by area and responsibility, not one file per Objective-C class. | The code is Rust; the contract binds behaviour, not file shape. |
| D3 | Core is generic over one `P: Platform` whose associated types name every Metal object kind. `dyn` only for the host's callbacks, the display executor and the OS services. | A handle flows into another Metal call (a texture into an encoder) with its kind checked at compile time. |
| D4 | One typed platform method per Metal method the library calls, and one enum variant per encoder and ICB command (`BlitCmd`, `ComputeCmd`, `RenderCmd`, `IcbRenderCmd`, `IcbComputeCmd`), with no table of ops between the caller and Metal. The platform's dispatch is one exhaustive `match` per command family, and it names each call by its selector and raise policy (4.3, 4.6). A call whose position the guest or host observes (a creation, release or retain, an aliasing setter, a command, a call with raw guest values) is sent where the caller makes it; a getter or device query is not ordered. | A send per guest operation, in order, with raw guest values: Metal's results, aborts and numbering follow the calls a process makes. |
| D5 | A descriptor the library only fills in and hands to a creator is a value (`platform/values.rs`), which the platform applies to a fresh Metal descriptor inside the creator, sending its setters in the order the value's type states. A descriptor Metal checks when it is set (the texture descriptor's pixel format, the render pass's sample positions, the rate map's layers), or whose attachments retain what they are given (the render pass), stays a Metal object behind a handle (4.3). | Metal folds some properties and aliases others, so the platform sends the raw values to a real descriptor in a fixed order rather than model them; an object stays where Metal's check at the setter, or a retain, is observable. |
| D6 | Handles are not `Clone`. Another reference is `.clone_ref()`; `Drop` is the release. | An IOSurface's retain count is visible, so no retain is taken that the count would show. |
| D7 | The raise channel is a value (`Raise`, `RaiseResult`), decided at the raise by its site: `Err` where a handler is always live, a death where none ever is, before any `Drop` runs, and the caller's `OnFault` where both are reached (6.2). There is no real `NSException` throw per raise. | The runtime's two-phase unwinder: with no handler, phase 1 fails and the process ends before any cleanup. Which frames handle a raise is fixed by the code's paths, so each site knows it. |
| D8 | Three outcome channels, never converted into one another: expected failures, raises, deaths (6). | The contract observes each separately: a refusal or error, a fault the guest is told of, a process death. |
| D9 | `panic = "unwind"` in every profile. The API's singleton installs a panic hook that aborts before unwinding. A death is `abort()`. | objc2's `exception::catch` needs unwind; an unintended panic still dies at its op with no side effects (6.5). |
| D14 | Containers are std's, each named by the order it gives (8). | Nothing depends on a hash table's order. |
| D16 | Queues, threads, locks, condition variables and signals are ours (7.1). A lock keeps the scope, and a race the outcome, that decides what crosses the API. No runtime lock-order checker. | A checker that panics would turn the hangs the API shows into deaths. |
| D17 | One platform instance per process (`VulkanPlatform`), a singleton in production (`reims-vgpu/src/singleton.rs`). | Process-wide where the platform holds one state per process (its devices by registry ID, its compiled shaders); every test hermetic. |
| D18 | The guest's RAM is reached only through the host's mapping of it, by copies and the platform's imports; there is never a Rust reference into guest memory (15.1). | The guest writes concurrently; a Rust reference into memory another writer changes is undefined behaviour. |
| D19 | Integer widths and wrap points are visible: std's `wrapping_*`, `overflowing_*`, `checked_*`, `From`/`TryFrom`, and `num::` for the few with odd semantics (17.4). | Widths and wrap points are behaviour. |
| D22 | Values drop where Rust's scoping puts them. Each unit of work (a command, an API entry) pushes a real pool (`rt::Pool`) for what Metal autoreleases, and a release that crosses the API is held to its point in the API's order by a holder (`rt::Held`) (7.4). | Metal's autoreleased returns must be drained (the 65th command buffer); the host's callbacks and IOSurface counts are observable, the lifetime of the library's own values is not. |
| D24 | The display's present shaders are the library's own, in Metal Shading Language (`vulkan/src/gpu/shaders.metal`), which the Vulkan platform loads as a metallib built from that source, once per device (4.7). | No shader of Apple's is read at run time or committed. |
| D27 | The C API: prefix `rvg_`; an opaque handle; one config struct per object, headed by `struct_size`; one callback table per object, with one `ctx` and one `destroy` (the window's from its own config); a status from every call that can fail; no Objective-C types, dispatch queues or blocks; a cbindgen header (11). | A C host calls it with nothing of Objective-C, as QEMU's `reims-graphics-pci` does. |
| D29 | QEMU's `reims-graphics-pci` is C on the C API, and links the library statically through pkg-config (12). | |
| D32 | arm64 only, for every Rust binary built for macOS (`aarch64-apple-darwin`); on Linux, `x86_64-unknown-linux-gnu`. | rustc's arm64e target has no pointer authentication, and its binaries crash before `main` (`rust-toolchain.toml` has the measurement). |
| D33 | A path no guest or host input reaches stays unreached: it is not brought to life. | |
| D34 | Nothing with a `Drop` is live in a frame an Objective-C exception can unwind: a Metal call runs in a closure that borrows its receiver and arguments (a `Retained` result exists only once the method has returned), and a foreign boundary's body runs in an inner function (4.6). | A foreign unwind runs landing pads; a value dropped there is a release at a point the code never chose (`vulkan/src/os/macos/msg.rs`). |
| D35 | Every Objective-C call catches its own exception where it starts: a call into AppKit, Foundation or IOSurface runs in the macOS message layer (`vulkan/src/os/macos/msg.rs`), and a call into host code under `host_call` (6.7). No Rust unwind handler sits below the library's Rust code that sends Objective-C messages either: the queues' workers and the device's threads are pthreads the platform starts (`Threads`) with an `extern "C-unwind"` start routine. Every `rvg_` function is `extern "C"` (6.5). | An exception out of a call ends the process at that call, before any frame of the library's unwinds (D34), whatever the caller holds. |
| D36 | objc2's debug checks are on in dev builds and off in `checked` (`[profile.checked.package.objc2] debug-assertions = false`); release has none. | The checks panic on a message to nil, a missing method and an encoding the runtime disagrees with; the tests that send to real Metal check every send they make. |
| D41 | Platform types follow the runtime's encodings: `ResourceAddressRange` is `{MTLAddressRange=QQ}`, the tile dispatch's render target array index is `u32`. | The runtime's `method_getTypeEncoding` gives these types. |
| D43 | Core denies `unsafe_code` at its root; the modules of 17.3 allow it. | `forbid` cannot be lowered per module. |
| D44 | One module style: `foo.rs` beside `foo/` (`clippy::mod_module_files`). `reims-vgpu-c` is a `cdylib`, `staticlib` and `rlib`. | The `rlib` lets tests link the C API. |
| D46 | A line goes through the `log` crate's macros under its module's path, on the caller's thread (14). | Diagnostics are outside the contract, so a line needs no fixed site or text. Formatting is paid only by a line a logger takes. |
| D47 | The library is a plain library: `cargo xtask dist` writes a static library, a shared library, the header and a pkg-config package, `reims-vgpu.pc` (11.3). | A host links it the way it links its other dependencies; QEMU links the archive, so its binary needs no rpath. |
| D49 | The device's cursor callbacks run on one display thread per process, a core `SerialQueue`, in the order the device queued them; every other callback runs synchronously on the thread the device calls it on (10; api.md section 5). | The API has no dispatch queues or blocks, and a host's console takes the cursor in one order. |
| D50 | `core::testkit` (the fake platform, its models and harnesses) is compiled only under `cfg(any(test, feature = "testkit"))`, and only dev-dependencies turn the `testkit` feature on, never a default; `cargo xtask testkit` checks both (13.1). | No product build can name the fake platform or a test harness. |
| D51 | The device's window is the library's: a crate of its own, `reims-vgpu-window`, between the platform and the API, which knows nothing of devices (2.2). A device created with a window config gets an AppKit window on macOS (or SDL3's with `REIMS_VGPU_WINDOW_IMPL=sdl`), into whose `CAMetalLayer` drawable the guest's scanout is drawn on the GPU, and whose input goes to the host's window callbacks; on Linux an SDL3 window, into whose Vulkan swapchain's image the scanout is drawn the same way. **Threads:** the window's AppKit side and callbacks run on the process's main thread, one at a time across every window, and must not block; the library reaches that thread only through blocks it posts to the main run loop, the application's start alone, and never waits for it. On Linux they run on a UI thread of the library's own, `reims-vgpu.ui`, and the main thread is never used (7.3). Each window draws on a render thread of its own, which nothing joins. **Size:** the window keeps the size the person gives it, and the device offers the guest one mode, the window's size in pixels, at the refresh rate of the host screen it opened on. What the window does for the person at the host is `features.md`'s register. | A host may create its device on the main thread before that thread runs its run loop, so a create or release that waited for the main thread would hang (the window crate's `native/macos/main_thread.rs`). |

Retired with Apple's paravirtualized device and its protocol: D10 to D13, D15, D20, D21, D23,
D26, D28, D40 and D48.

---

## 2. Workspace and crates

### 2.1 Layout

```
Cargo.toml Cargo.lock rust-toolchain.toml clippy.toml .cargo/config.toml
crates/
  reims-vgpu-core/      what the platforms and the device share, generic over P: Platform;
                        links no Apple framework
  reims-vgpu-vulkan/    Platform over metal2vulkan-runtime, the library's only one: on Linux, and
                        over MoltenVK on macOS; its OS services; the library's shaders
  reims-vgpu-window/    the device's own window: AppKit or SDL3, input, the host's screens,
                        and features.md, the register of what the window does
  reims-graphics/       the device's wire, its rings and control, and the C header the guest
                        compiles (include/reims_graphics_wire.h)
  reims-vgpu/           the Rust API: the graphics device over the platform, and its window
  reims-vgpu-c/         the C API (cdylib, staticlib, rlib) and include/reims_vgpu.h
  reims-vgpu-video-*/   the host's video decoders and encoders (docs/video)
  reims-ctl/            the VM manager, a workspace of its own
xtask/                  testkit, header, dist, metallibs, check
firmware/               the device's option ROM: its UEFI driver, in C, and the prebuilt ROM (11.4)
guest/                  the guest's kexts, loader and Metal bundle, and the prebuilt kext set
tests/shaders/          the tests' metallibs and their sources
scripts/                install-ctl.sh: installs reims-ctl and its QEMU
scripts/internal/       the developer's scripts (development.md)
vendor/qemu             QEMU, whose reims-graphics-pci is C on the C API
```

`cargo xtask dist` writes `target/dist-vulkan` (11.3, D47): the static library and the shared library,
the header and `reims-vgpu.pc`. The library compiles its own shaders (D24), and reads no
preferences domain and no entitlement: the device's options are its config's fields.

### 2.2 Crate graph

```
reims-vgpu-core, metal2vulkan-runtime  <-  reims-vgpu-vulkan  <-  reims-vgpu-window  <-  reims-vgpu
reims-graphics, reims-vgpu-video-*  <-  reims-vgpu  <-  reims-vgpu-c  <-  QEMU
```

- **The Vulkan platform** (`reims-vgpu-vulkan`) is the library's `Platform`, over
  `metal2vulkan-runtime`, the Metal-shaped API on Vulkan of metal2vulkan, at the commit of its
  public repository `Cargo.toml` names, on the host's Vulkan driver, or MoltenVK on a Mac (a build
  for development and tests). Its OS services are its own: on macOS over the system's (`os/macos/`:
  IOSurfaces, Mach VM, the clock, threads, the autorelease pool, and the message layer, 4.6); on
  Linux `os/linux/` (a host VM over `mmap` of the guest RAM's files, and an IOSurface stand-in over
  that memory). It compiles the library's shaders (4.7). Where its answer differs from Metal's it logs the difference and goes on, and dies
  where Metal would or where the runtime cannot run the call ([accuracy-ledger.md](accuracy-ledger.md)).
- **Core** depends on `parking_lot`, `zerocopy`, `bitflags` and `log` only. Its `testkit` feature
  exposes `core::testkit` outside core's own tests; only dev-dependencies enable it, and
  `cargo xtask testkit` checks that (D50).
- **The window** (`reims-vgpu-window`) is the device's own window (D51), and knows nothing of
  devices. Its pure logic is unit-tested with no AppKit or SDL: `geometry` (the image's rectangle
  in the window, the pointer's position in the guest's mode, the buttons), `keymap` (key codes to
  USB usages, by position), `keys` (modifiers and held keys), `capture` (who gets the keyboard, and
  where the pointer may go), `scroll` (wheel and trackpad deltas to notches, or smooth with their
  gestures' phases), `policy` (the modes and density offered for a host screen) and `env` (the
  `REIMS_VGPU_WINDOW_*` variables). `handle.rs` is the `Window` the device holds, `Send` and `Sync`
  with no AppKit object in it, and `FrameSink`, the trait it draws through; `present.rs` the
  presenter both OSes' render threads draw through.
  - **On macOS** (`native/macos/`) the AppKit side runs on the main thread (7.3), its state in
    statics only that thread reaches (`main_thread::MainOnly`): `main_thread.rs` posts work to the
    main run loop in order, and the application's start, alone; `app.rs` is the application, its
    `sendEvent:`, the menu, the activity against App Nap, Quit, and the screens' changes;
    `shell.rs` the `NSWindow` and its delegate; `view.rs` the layer-hosting content view over a
    `CAMetalLayer`; `input.rs` the events to the host's callbacks, deciding over plain values;
    `cursor.rs` the guest's glyph as an `NSCursor`; `hotkeys.rs` the system's hot keys, off while
    the window captures the keyboard; `render.rs` the render thread. Through SDL3
    (`REIMS_VGPU_WINDOW_IMPL=sdl`, in a build with the `macos-sdl` feature, on by default) SDL's
    shell and cursor (`sdl/`) run on the main thread inside the same application, with AppKit's
    events given to the same `InputState`.
  - **On Linux** the window is SDL3's (`native/linux/`, with SDL's shared shell and cursor in
    `sdl/`), on Wayland or X11, built from SDL's pinned source release with its video subsystem
    alone and linked statically (`sdl3-sys`), and drawn through Vulkan on the runtime's own device.
    SDL runs on the library's thread `reims-vgpu.ui` (7.3).
  - It holds two `#[link]`s of its own: Carbon's (`KBGetLayoutType` and `LMGetKbdType`, for the
    ISO keyboard check) and AppKit's (the `NSApp` global). SkyLight's hot-key functions are looked
    up with `dlsym`, since the package's link line cannot name a private framework.
  - **Exceptions and panics** (D35, 6.5). Every send to AppKit, Foundation, QuartzCore or Metal
    runs in the Vulkan crate's macOS message layer (`reims_vgpu_vulkan::os::msg`), nearly all
    through `msg::get`, so an exception ends
    the process at its send. An event passed on to AppKit and `setPresentationOptions:` catch
    theirs (`msg::catch`) and go on. Every `define_class!` method and every block posted to the
    main thread starts in `BlockEnv::foreign`, which marks the thread as the library's, so a panic
    there aborts before it unwinds into AppKit.
- **The device** (`reims-graphics`) is the wire and the device's transport (9), with no platform:
  `reims-vgpu` drives the platform from what it carries.
- **The API** (`reims-vgpu`) is where the device meets the platform: its types are generic over an
  `ApiPlatform`, the OS's default platform by default, so its unit tests run its forwarding on the
  fake platform (10).
- **The C API** (`reims-vgpu-c`) forwards to the API and adds no behaviour (11).

### 2.3 Profiles

| profile | what | notes |
|---|---|---|
| `release` | the product | `panic = "unwind"` (D9), no overflow checks (wrapping is explicit), thin LTO |
| `dev` | tests | overflow checks on |
| `checked` | a second pass of the tests | release with overflow checks and debug assertions |

objc2's debug assertions are on in `dev` and off in `checked` (D36). Every macOS binary is
`aarch64-apple-darwin`, and the Linux library `x86_64-unknown-linux-gnu` (D32). `cargo xtask dist`
builds the library for macOS 11.0, rustc's default, so the C that objc2's exception helper compiles
is built for it too.

---

## 3. Core's modules

`crates/reims-vgpu-core/src/`, one file per module beside its directory (D44). Each module's `//!`
doc holds its ordering contracts and the odd behaviours it keeps, each naming its test.

| module | what it owns |
|---|---|
| `raise`, `raise/{on_fault,death,kept,refusals}` | the raise channel, `OnFault`, deaths, values kept for good, refusal scopes (6) |
| `logging` | the library's `log` logger: silent by default, stderr with `REIMS_VGPU_LOG` (14) |
| `trace` | the Vulkan platform's timeline of the device's work, with `REIMS_VGPU_TRACE` |
| `num` | integer operations std has no single spelling for (17.4) |
| `collections` | the containers, named by the order they give (8) |
| `rt` | `SerialQueue`, `BlockEnv`, `Pool`, `Held` and `Hold`, `CoalescingSignal`, `Timer`, named threads (7) |
| `sync` | `parking_lot` re-exports, `WriterPrefRwLock` (7) |
| `platform`, `platform/*` | the `Platform` trait, handle traits, the command enums, descriptor values, device caps, `HostRange`, the OS services (4) |
| `host` | the executor the device's display jobs run on (`HostQueue`) and the host's opaque values |
| `memory` | the guest's RAM as the host maps it (`MemoryMap`), mappings of it, and its descriptor reads |
| `display` | the display's present pipelines, by pixel format (`DisplayPipelineCache`) |
| `testkit` | (tests only) the fake platform, harnesses and models (13.2) |

`pub(crate)` is the default; stateful types have no `pub` fields except where a sibling area
reaches them directly.

---

## 4. The platform boundary

### 4.1 Shape

`platform.rs` defines `Platform`, with one associated type per Metal object kind the library
touches (device, queues, command buffers, the three encoders, buffers, textures, heaps, state
objects, pipelines, libraries, events, ICBs and their commands, rate maps, descriptors) and the OS
services (4.5). Two implementations: `reims_vgpu_vulkan::VulkanPlatform` and the testkit's
`FakePlatform`. There is no table of ops
between them and their callers: each Metal method the library calls is a typed method (4.3).

- **`Handle`**: every object handle. `clone_ref()` is a retain, `Drop` a release, `id()` its
  `ObjId` (the object's pointer, for identity checks). No `Clone` (D6).
- **Nil** is `Option<H>`; a message to nil is written with std's combinators, or `nil_send!` where
  the answer is the type's zero (`NilDefault`).
- **Host memory**: the few methods that take a host pointer (`new_buffer_over_host_memory`,
  `replace_region`, `set_contents`, the texture fills) take a `HostRange<'_>`, whose one
  constructor is `unsafe` and is called where the caller proves the range stays mapped (17.3).

### 4.2 Conventions every method follows

- A method that can raise inside Metal returns `RaiseResult<T>`; zero-argument property getters and
  descriptor property setters do not (their raise is fatal at the send, 4.6).
- An object argument the caller can pass as nil is `Option<&P::X>`; one it never passes as nil is
  `&P::X`.
- Integer widths are the runtime's encodings (D41). Raw guest values reach Metal unvalidated.
- Creators keep one method each, because each differs in `+1` or `+0`, nil, `NSError` or raise.
- Only calls made straight-line with nothing fallible between are grouped into one (a sub-object
  chain such as `colorAttachments[i].loadAction`, a run of pure getters). Nothing is grouped across
  a lookup, a guest read, a log, a lock or a raise.

### 4.3 Commands, values and caps

Each Metal method the library calls is one typed method of a platform trait, sent where its caller
calls it.
- **Encoder and ICB commands** are `BlitCmd`, `ComputeCmd`, `RenderCmd`, `IcbRenderCmd` and
  `IcbComputeCmd`: `encoder.encode(on, cmd)` (the ICB commands' `encode(cmd)`) sends exactly one
  selector per variant, the one `cmd.selector()` names.
- **Device capabilities** are one `DeviceCaps` (`platform/caps.rs`): every query is sent once per
  device, and the platform keeps the answers by registry ID.
- **A descriptor the library only fills in** is a value (`platform/values.rs`: buffer, heap,
  sampler, depth-stencil, ICB, compute pass, the render and compute pipelines, and the texture and
  rate map values a made descriptor starts from). Each type's doc states the order its setters are
  sent in, which the Vulkan platform (`gpu/desc.rs`) and the fake follow. The
  storage properties that alias one another (`resourceOptions`, `storageMode`,
  `hazardTrackingMode`) are an ordered `OptionSet` list, applied in the order they were recorded.
- **A descriptor the library keeps** is a Metal object (`platform/descriptor.rs`, D5): the texture
  descriptor, the render pass descriptor, and the rate map descriptor and its layers.
- **An ICB command's arguments** are values (`DrawArgs`, `DispatchThreadsArgs`, …) read whole from
  Metal's private argument objects.

### 4.4 Families

| file | traits |
|---|---|
| `platform/device.rs` | `GpuDevice`: queries and creators |
| `platform/command.rs` | `CommandQueue`, `CommandBuffer` (commit, handlers, status and error) |
| `platform/blit.rs`, `compute.rs`, `render.rs` | the three encoders and `ResourceRef` |
| `platform/resource.rs` | `GpuResource`, `Buffer`, `Texture`, `Heap`, state objects, pipelines, `Library` |
| `platform/descriptor.rs` | the descriptors the library keeps as Metal objects |
| `platform/values.rs` | the descriptor values and the render pass's `*Set`s |
| `platform/caps.rs` | `DeviceCaps` |
| `platform/icb.rs` | indirect command buffers and their commands |
| `platform/rate_map.rs` | rasterization rate maps |
| `platform/event.rs` | events and shared events |
| `platform/types.rs` | value types (`MtlSize`, `MtlRegion`, pixel formats, enums) |

### 4.5 OS services

`platform/os/`: `clock` (monotonic and wall time, `wait_until`), `threads` (the threads the library
runs on and the timers' clock), `autorelease` (the real pool), `surface` (IOSurface creation,
getters and retain counts), `vm` (allocate, remap, protect, deallocate). The Vulkan crate implements
each over the system: macOS's on macOS, Linux's on Linux; the testkit models each (13.2).

### 4.6 The answer layer and the macOS message layer

Every runtime call the Vulkan platform makes for core runs in its answer layer (`answer.rs`),
named by the Metal selector it stands for, which a raise and a death name: the call's `Policy`
decides what the runtime's typed error becomes, an `Err(Raise)` for the caller to catch, a nil, or
a death at the call, where Metal raises, answers nil or ends the process
([accuracy-ledger.md](accuracy-ledger.md)).

On macOS the Objective-C sends the library makes itself (the window's AppKit side, IOSurface's
creation) run in the message layer, `os/macos/msg.rs` of the Vulkan crate, named by their
selector: `msg::get(sel, || call)` for a call whose raise dies at the call (a getter, a property
setter, a C function), `msg::catch(sel, || call)` for one whose exception is the caller's to
decide, and `fatal_send` and `value` for Foundation and IOSurface sends, which die at the send. The
call runs inside objc2's `exception::catch`, so an exception stops at the call before any Rust
frame unwinds, and the closure borrows its receiver and arguments and owns nothing with a `Drop`
(D34).

### 4.7 The library's shaders

`crates/reims-vgpu-vulkan/src/gpu/shaders.metal` holds the library's own shaders, in Metal Shading
Language: the display's present functions (`displayPresentVertex`, `displayPresentFragment`,
`displayPresentFragmentWithGammaTableAndColorMatrix`) and the display pipe's colour stages
(`displayPipeColor*`, `displayPipeLut*`). What each computes is written beside it in the source.

`cargo xtask metallibs` compiles it, and the Vulkan crate's other `.metal` sources, with Apple's
tools on a Mac into `crates/reims-vgpu-vulkan/metallibs`, which a build off a Mac copies
(development.md); a build on a Mac compiles its own. `gpu/shaders.rs` loads the metallib with the
runtime's `new_library_with_data` once per device, the first time it is asked for, by registry ID,
and every later ask answers the same library.

---

## 5. (retired)

Isolated guest tasks went with Apple's paravirtualized device.

---

## 6. Outcomes: expected, raise, death

### 6.1 Three channels

Never converted into one another, with no `From` between them (D8):
- **Expected failure**: `Option` or `Result` with an error enum per operation, where the API
  answers a refusal, an error or a status.
- **Raise**: `RaiseResult<T> = Result<T, Raise>`, an exception as a value, which a catch turns into
  a fault the guest is told of, a status or a reply.
- **Death**: `fatal(id)`, one stderr line and `abort()`.

### 6.2 Raises and the hold

`raise.rs`, `raise/on_fault.rs` and `rt/held.rs`:
- `Raise { id: RaiseId, detail, origin }`. `RaiseId` has one variant per area, each wrapping the
  area's `RaiseKind`, whose `Display` is a dotted id; `Platform` is a raise inside Metal, whose
  origin names the op. `Raise::new` is pure.
- **The hold** (`rt::Hold`) names the holder that what a unit's work holds deep (`hold.keep`), and
  what a raise it ends carries (`hold.land`), go to (7.4). A function takes it as a parameter only
  where it needs it. It is `Copy`, and `!Send`, so it never reaches a queue's block, a thread or a
  completion handler. It does not decide raises.
- **A raise is decided by its site's label.** A site no path reaches without a handler is `Err(r)`
  (`raise!(id)`); one no path reaches under a handler calls `fatal` there, before any `Drop` runs
  (`Raise::new(id).fatal()`); a function its callers reach both ways takes an `OnFault` from its
  caller, which decides (`raise!(on => id)`, `on.throw(r)`).
- **Where a hold starts.** A unit of work starts one with the unit's holder (`Hold::of`): each
  command of the device's decoder, each API entry. A queue's block, a timer handler, a once's
  block, a thread's body, a Metal completion handler and a drop start with `Hold::NONE`: their code
  is `Send` and the caller's hold is not. A raise there with no handler of its own dies at the
  raise (an exception never unwinds out of a callout, `rt/block_env.rs`).
- **A handler** is a frame that takes a raise's `Err`: it ends it (`hold.land(r)`), turns it into a
  fault, a status or a reply, or passes it on with `?`. The device's decoder runs under a live
  handler: a platform raise comes out of a command for the decoder to fault the guest's command
  buffer with (9).
- **The cost of a raise** is nothing the API shows: there is no real `NSException` per raise (D7).

### 6.3 Raises inside Metal

Decided at the call by the answer layer (4.6), by the op's policy; an Objective-C send of the
library's own is caught at the send by the macOS message layer. No Objective-C raise of the
library's crosses foreign frames (D34).

### 6.4 Deaths

A death is the one fatal path, `raise::death::fatal(id)`: one line on stderr,
`reims-vgpu: fatal: <what died>`, written in one write, then `abort()`. No `Drop` runs and nothing
unwinds, so the process ends at the guest op that caused it and nothing after it reaches the host
or the guest. `fatal!` and `fatal_unless!` are its macro forms; an uncaught raise (`Raise::fatal`)
adds its origin and detail to the line. Where the process dies is the contract; the message and the
signal are diagnostics.

`DeathId` (`raise/death.rs`) names what died: one variant per area, wrapping the area's
`DeathKind`; `Raised` (a raise nothing may catch); `Platform` (a platform call that ends the
process, among them the Vulkan platform's passing on of a refusal of the runtime's); and
`HostRaised` (6.7).

In tests (`cfg(any(test, feature = "testkit"))`) with capture on, `fatal` records the death's
`DeathSite` in the test's `DeathSlot` and panics with a `Death` payload (`testkit::capture`); the
subprocess harness (`testkit::death`) turns capture off to see what a real death leaves behind.
`DeathSite` and its `DeathKind` class are that test identity only: production never computes them.

### 6.5 Panics

A panic is a bug, never control flow, and it ends the process as a death does: one line
(`reims-vgpu: panic: …`), then `abort()`, before anything unwinds, so no drop runs first and no
panic of the library's reaches the host's frames. Inside a refusal scope (the device's ops,
`raise::refusals`) it is refused as a death there is: the line is written, the panic unwinds to the
scope, and the op is refused, logged as a bug; under the test capture it is the test's failure. The
API's singleton (`reims-vgpu/src/singleton.rs`) installs a chained panic hook that does this when
the panicking thread is running the library's code (a worker running one of the library's blocks,
one of the device's threads, a Metal completion handler or buffer deallocator, a public entry
point, or a C export, each marked by `rt::enter_ours`), and leaves any other panic, the host's own,
to the host's hook (D9). Every C export is `extern "C"` and runs its body in `ffi::export`
(`reims-vgpu-c/src/ffi.rs`), which installs the hook and marks the thread first. `panic = "unwind"`
stays because objc2's `exception::catch` needs it.

### 6.6 Values kept for good

`raise/kept.rs`: `keep_forever(x, Kept::…)` is `mem::forget` with the rule that names what the
guest or host sees because the value's release never comes. A value is kept only where that shows:
a release nothing sees is a plain drop. Each call has a doc comment and a test.

### 6.7 Host calls

Every call into the host's own code runs inside the OS's catch (`ffi::host_call`,
`reims-vgpu-c/src/ffi.rs`, over `ffi/catch/{macos,linux}.rs`): objc2's `exception::catch` on macOS,
std's `catch_unwind` on Linux. It surrounds each function of `RvgGraphicsCallbacks` and of
`RvgWindowCallbacks`, and each table's `destroy`. Anything the host throws ends the process at the
call: on macOS an Objective-C throw is `DeathId::HostRaised`; on Linux a Rust unwind that reaches
the catch is `HostRaised`, and a foreign one (a C++ throw) cannot be caught, so std ends the process
where it reaches the catch's frame. The callback types stay `extern "C-unwind"`, which both catches
need.

---

## 7. Concurrency

### 7.1 Primitives

| primitive | what it is | kept |
|---|---|---|
| serial queues (`reims-vgpu.display`, the display thread, among them) | `rt::SerialQueue`: a `VecDeque` of blocks and sync turns under a `Mutex`, run by a worker thread of the queue's own (`Threads::spawn_worker`), which ends after 5 s idle or once the last handle is gone | blocks run one at a time in submission order. `sync` runs the block directly on the caller's thread once the blocks ahead have run ([lessons.md](lessons.md)); no exception leaves it, since every call it makes catches its own (D35). The worker is woken only when it has work, and after the lock is released |
| add and or signals | `rt::CoalescingSignal` | merges before the handler runs give one call; a release without a cancel drops pending data |
| timers | `rt::Timer`: a thread of its own on the platform's clock merging fires into a `CoalescingSignal` | missed fires coalesce |
| mutexes and condition variables | `parking_lot::{Mutex, Condvar, ReentrantMutex}` | each mutex guards its own fields and nothing more; every timed wait goes through `Clock::wait_until` |
| reader-writer locks | `sync::WriterPrefRwLock` (`parking_lot::RwLock`) | writer preference: a thread's first read waits behind a queued writer; a nested read on a thread that already holds one goes through at once |
| weak references | `std::sync::Weak` | fails to upgrade once teardown starts |
| Metal completion handlers | `Box<dyn FnOnce + Send>` run on Metal's thread | a foreign thread: under `Hold::NONE`, owning no queue, holding none of the committer's locks |
| the display thread | `reims_vgpu::platform`'s one `SerialQueue` per process, behind core's `HostQueue` (D49) | the device's cursor jobs run on it one at a time, in the order the device queued them; the library never waits for it (api.md section 5) |

Which thread runs a block, and the threads' names, are not part of the contract: what crosses the
API, and its order, is, and the header names the thread each callback arrives on (api.md section 5).

### 7.2 Locks around the host's callbacks

The device calls `raise_interrupt` with no lock held that one of its functions waits for, directly
or through a thread of the library's, so a host may take its own lock (QEMU's BQL) in it; api.md
section 5, "The locks around `raise_interrupt`", lists the locks of the library's it may run under.
No `ReentrantMutex` is held across `SerialQueue::sync`: the sync block runs on the caller's thread,
so a block that takes a recursive lock its caller holds would re-enter it.

### 7.3 Threads

- **The device's threads** (9): its control thread (`reims-vgpu.graphics-control`), ring 0's lanes
  (`reims-vgpu.graphics-ring0-<i>`), its refresh thread (`reims-vgpu.graphics-refresh`), a service
  thread and lane threads per command queue (`reims-vgpu.graphics-queue-<id>`,
  `reims-vgpu.graphics-lane-<id>`), and its pools: pipeline compiles
  (`reims-vgpu.graphics-compile-<n>`), surface and pages-block views
  (`reims-vgpu.graphics-surfaces-<i>`), ICB creation (`reims-vgpu.icb-create-<i>`), a closed
  process's releases (`reims-vgpu.graphics-closes-<i>`), video (`reims-vgpu.graphics-video-<i>`)
  and event waits (`reims-vgpu.graphics-event-waits`). How long a pool keeps an idle thread is
  `reims-graphics`'s `keep`.
- A worker per serial queue while it has work (it ends after 5 s idle), the process's display
  thread among them.
- Metal's completion threads, and the host's own threads inside a call that may call back (both
  foreign).
- The process's main thread, the host's, where the window's AppKit side runs (D51). The library
  reaches it only by posting blocks to the main run loop
  (`reims-vgpu-window/src/native/macos/main_thread.rs`: `CFRunLoopPerformBlock` in the common
  modes, then `CFRunLoopWakeUp`), which run in the order posted. The first window's attach posts
  the application's start, alone; what is posted before the application has launched is held in
  the library, and runs in order once it has. The library never waits for the main thread, and
  never posts to the main dispatch queue (`main_thread.rs`'s doc has the measurement).
- A render thread per window (`reims-vgpu.window`, 512 KiB, started by the platform), from the
  window's attach. It waits on the window's surface for a new frame, a new layout or the window
  showing again, and draws only while the window is at least partly visible: into the swapchain's
  next image, through the window's sink (10). The image is its pacing. The thread ends at the
  window's end, and nothing joins it.
- On Linux, in place of the main thread, the library's UI thread (`reims-vgpu.ui`, 1 MiB), started
  with the first window or the first look at the host's screens, where SDL's video is initialized
  and every SDL call but a render thread's Vulkan surface is made: the window's opening, changes and
  end, its input and the host's window callbacks, one at a time. The process's main thread is never
  used or waited for. The render thread presents on a queue made for presenting, which the runtime
  keeps apart from the guest's queues where the device has one to spare.

No thread of the library's that runs Rust code sending Objective-C messages has a Rust unwind
handler at its base (D35): the platform starts them as pthreads (`vulkan/src/os/macos/threads.rs`).

### 7.4 Autorelease pools and release points

Values drop where Rust's scoping puts them. There is no pool emulation for the library's own
values. Two things remain, each for what crosses the API.

- **Real pools** (`rt/pool.rs`). A `Pool` pushes the platform's real pool around a unit of work and
  pops it at the unit's end, so that what Metal autoreleases inside it is freed there: a command of
  the device's decoder, an API entry (`reims_vgpu::entry::run`), a window's draw. Without one, an
  uncommitted command buffer lives on in a dev build, where Metal's autorelease of it is not elided
  (4.6), and the queue's 65th `-commandBuffer` blocks. The library's threads keep a last-resort
  pool for what is autoreleased on them outside any unit.
- **Holders** (`rt/held.rs`). A value whose last release crosses the API is held to its point in
  the API's order by a `Held`: each API entry's, each unit's, released most recent first, before
  its unit's pool pops. Under a hold with no holder (a queue's block, a thread's body, a completion
  handler, a drop), a value held deep goes to the thread's last-resort holder on a thread of the
  library's, and on any other thread drops at once.
- A value whose release the guest or host would see, and which is never released, is kept under a
  named rule (6.6).

### 7.5 Clocks

`Deadline::{Mono, Wall}` (`platform/os/clock.rs`), each wait on a fixed clock. `Clock::wait_until`
waits in awake time and reads the answer on the deadline's own clock. Nothing calls `Instant::now`
or `SystemTime::now` outside the clock (`clippy.toml` disallows them).

---

## 8. Containers

`collections/`, over std's containers, each named by the order it gives. `LookupMap<K, V>` holds
what is only looked up (the display's pipeline cache): it has no iteration API, and a fixed hasher,
so its drop order is the same in every run. Drain APIs return values, so callers choose their drop
points inside or outside their locks.

---

## 9. The reims-graphics device

[graphics-kext/design.md](graphics-kext/design.md) is the device's design and wire protocol. In the
code:
- **`reims-graphics`** is the wire and the transport, with no platform: `wire` (the registers, the
  region's layout, the rings, the control commands and events, which cbindgen writes into
  `include/reims_graphics_wire.h` for the guest's kexts and the option ROM), `ops` (the command
  buffer op stream the guest's bundle serialises), `icb` (the CPU's encodes into an indirect
  command buffer), `ring` (the host's two ends of a ring), `control` (the control commands),
  `display` (the display's modes, scanout and cursor), `region` (the region's memory, BAR 2), and
  `Device`: the registers (BAR 0), the control ring's thread and the display's refresh thread, over
  the host's callbacks.
- **`reims-vgpu`'s `graphics`** is the device as the host's process runs it, over the platform:
  `GraphicsDevice` (10); `decode/`, the Metal-call decoder, which reads the guest's serialised
  Metal calls off a command queue's ring and drives the platform, keeping an `obj_id` → handle
  table for each guest process (a create inserts the new handle, a destroy drops it); `present`,
  `sink` and `read`, the display pipe's presents, the window's draw of the scanout or a presented
  IOSurface, and the host's read-back; `color`, the display pipe's colour; `compile`, the pipeline
  compiles; `completion`, the completion ring's events and the interrupt; `device_info`, what the
  guest is told of the GPU; and the host video arms (`video_host`, `vaapi`, `nvidia`,
  [video/](video/)).
- **The decoder runs under a live raise handler**: a platform raise comes out of a command for the
  decoder to fault the guest's command buffer with, and an op-stream reference the host cannot
  honour faults it the same way, except one to an object whose create the host refused, whose uses
  are dropped (`decode/refused.rs`).
- **The window's draw** imports the region's pages under the scanout as a no-copy buffer on the
  platform and blits the scanout out of it into each drawable, on the GPU, with no copy on the CPU
  (`sink.rs`); the buffer's deallocator holds the region, so it stays mapped until the GPU has let
  the buffer go, after the device and the window.

---

## 10. The Rust API (`reims-vgpu`)

The Rust API is owned Rust types over the device, which the C API wraps one to one; api.md section
3 lists them.
- **Objects.** `GraphicsDevice`, made from one `GraphicsConfig` whose callbacks are closures. A
  create answers a `Result`, whose `Error` names what refused it. The last drop of a device is its
  end: its window closes and its threads end, waited for by nothing.
- **Generic over the platform.** The types are generic over an `ApiPlatform` (core's `Platform`,
  plus the process's instance), the OS's default platform by default, so the unit tests run the
  API's forwarding on the fake platform, which gets a fixed screen and no sink, so its windows
  never open.
- **The display thread.** The device's cursor callbacks run on the process's display thread
  (`platform.rs`, D49): core's `HostQueue` over one `SerialQueue` made at the process's first device
  whose host takes the cursor.
- **The process's platform.** `singleton.rs` makes the one platform on first use: it installs the
  panic hook (6.5), then the stderr logger if `REIMS_VGPU_LOG` asks for it (14). The platform is
  `VulkanPlatform`, an `ApiPlatform`.
- **Entries.** Every public method, and every release of a handle, runs inside `entry::run`: the
  thread is marked as the library's (6.5), and a real pool and a holder are open (7.4), so what the
  call held or Metal autoreleased on the thread is released before it returns.
- **The window** (`window.rs`, D51). A `WindowConfig` makes the device's `Window` (the window
  crate's), which keeps its size: the device offers the guest the window's size in pixels as its
  one mode, and tells the guest when the person resizes it. Once the device exists the window gets
  its sink. `graphics/vulkan`'s sink makes a
  presenter over a swapchain of the window's surface (on macOS one of its `CAMetalLayer`), which
  draws into the swapchain's next image, commits, and presents the image on the swapchain's
  presenting queue once the runtime has scheduled the command buffer. Neither waits for the GPU.
  The guest's cursor is the host's cursor over the image, shown while the guest shows its cursor.
- There is no host-object layer: no Metal device, dispatch queue or Objective-C type crosses the
  API as a config, handle or callback a host passes.

---

## 11. The C API (`reims-vgpu-c`), the dist tree and the option ROM

### 11.1 Conventions

docs/api.md is the C API's specification; the header repeats each rule where it applies.
- Prefix `rvg_`. Every function is `extern "C"`, and its body runs in `ffi::export` (6.5);
  `tests/exports.rs` checks both, and that the shared library exports every function of the header.
- The handle is opaque (`handle.rs`). A release of NULL does nothing; any other NULL handle is
  `RVG_ERR_INVALID_ARGUMENT` where the function answers a status, and a death where it does not.
- Every struct the host writes starts with `uint32_t struct_size` (`sized.rs`): a larger size is
  accepted only when its extra bytes are zero.
- Each object's callbacks are one table with one `ctx` and one `destroy` (`table.rs`). `destroy`
  runs once, when the last closure built from the table goes. The window's table (`window.rs`) is
  held beside the device's and let go of first: its `destroy` runs when the release closes the
  window.
- A call that can fail answers an `RvgStatus` (`status.rs`), a plain `u32`, and
  `rvg_status_message` describes each.
- Every call of a host function runs in `ffi::host_call` (6.7).
- The exports are written over `Plat`: `VulkanPlatform`, and the testkit's `FakePlatform` in the
  crate's unit tests, which then run the product's code. The crate's `vulkan` feature changes
  nothing, and stays for the builds that name it.

### 11.2 The header

`include/reims_vgpu.h` is cbindgen's output: `cargo xtask header` writes it, and `--check` fails
when the committed header differs. QEMU's `reims-graphics-pci` and the C harness are compiled
against it.

### 11.3 The dist tree

`cargo xtask dist` (D47) builds `reims-vgpu-c` on the Vulkan platform and writes
`target/dist-vulkan`:

```
include/reims_vgpu.h                   the committed header
lib/libreims_vgpu.a                    the staticlib
lib/libreims_vgpu.dylib                the cdylib, on macOS: @rpath/libreims_vgpu.dylib, signed ad hoc
lib/libreims_vgpu.so                   the cdylib, on Linux
lib/pkgconfig/reims-vgpu.pc            the package
share/doc/reims-vgpu/                  the license: LICENSE and LICENSES/, as in the repository
```

`reims-vgpu.pc` takes its prefix from its own path, so the tree can move. Its `Libs.private` lists
the frameworks and libraries a static link of the archive needs. A host that reads the package once,
as meson does at configure, must read it again when it changes (`make qemu` does; lessons.md).
`CARGO_TARGET_DIR` moves the tree with the rest of the build. `--platform vulkan` names the only
platform, for the builds that pass it; any other is refused.

On Linux `dist` writes `Libs.private`
as rustc prints it for the archive. The archive bundles SDL3, which draws the window there (2.2),
so it is installed as one object (`ld -r` of the built archive's members) whose only global symbols
are the C API's `rvg_*` functions, every other made local (`objcopy --keep-global-symbols`): a host
that links its own SDL beside it, as QEMU's `-display sdl` does, keeps its own. `dist` fails where
the archive installed defines any other global. The shared library exports the C API's functions
alone through rustc's version script.

On macOS the archive bundles SDL3 too (the window crate's `macos-sdl` feature), and is installed
the same way: one object (`ld -r -exported_symbols_list`) whose only globals are the C API's
functions. SDL's Objective-C classes are registered by name whatever their symbols, so the build
renames them through defines in the C flags SDL's CMake build takes (`xtask/src/dist/macos.rs`).

### 11.4 The option ROM

`firmware/` is the reims-graphics device's EFI option ROM: an x86-64 UEFI boot-service driver in C,
which an x86-64 guest's firmware runs. reims-ctl carries the prebuilt ROM,
`firmware/prebuilt/ReimsGraphics.rom`, and names it as the device's `romfile`; QEMU's device has no
default ROM.

```
Makefile                     clang and lld: the image, then the ROM; prebuilt and prebuilt-check
prebuilt.sh                  writes prebuilt/ReimsGraphics.toml, the record of the ROM's sources
prebuilt/                    ReimsGraphics.rom and its record
src/ReimsGraphics.c          the entry point, the driver binding and the control ring
src/ReimsGraphicsOutput.c    the Graphics Output Protocol
src/ReimsGraphicsName.c      the driver's and the controller's names
src/ReimsGraphics.h          the driver's private data, over the wire's header
src/Uefi.h                   the part of the UEFI specification the driver uses
src/ReimsGraphicsRom.S, src/OptionRom.S   the ROM around the image: the 0x55AA header and the PCIR structure
```

**What the driver does.** `Start` resets the device, adopts the control rings' counters, asks the
device for its displays and offers one mode: the preferred mode of the first connected display, or
1920 × 1080 when there is none. The framebuffer is the region's memory from
`RGFX_ALLOCATIONS_OFFSET`, and `SetMode` points the display at it with `SET_SCANOUT`. Firmware
takes no interrupts: a command polls the completion ring for its answer. The registers, rings and
commands are the wire's, from `crates/reims-graphics/include/reims_graphics_wire.h`.

**The build.** `firmware/Makefile` compiles the C freestanding for `x86_64-unknown-windows`, whose
default calling convention is the Microsoft x64 one UEFI uses, at `-Os` with `-Werror`; `lld-link`
links it as an EFI boot-service driver, and the assembly lays the ROM header and a PCIR structure
around the image. The bytes are the compiler's: Xcode's clang and a Linux host's build different
ones from the same sources. So `make -C firmware prebuilt` rewrites the committed ROM on a Mac with
its record, `prebuilt/ReimsGraphics.toml` (the sha256 of each file it was built from and the
compilers), and `make -C firmware prebuilt-check`, which `cargo xtask check` and reims-ctl's tests
run, fails where a file the record names has changed.

**Its conventions.** The C follows EDK2's style and the UEFI specification's names (`EFI_STATUS`,
`gBS`, `IN` and `OUT`), not the Rust conventions of section 17. `Uefi.h` declares only what the
driver calls, and keeps every table's offsets by declaring the services it does not call as opaque
pointers.

---

## 12. QEMU

`vendor/qemu`'s `reims-graphics-pci` (`hw/display/reims-graphics-pci.c`) is plain C on the C API,
the device's transport: BAR 0 is the library's registers with the MSI-X table, BAR 2 the library's
region, and the control ring's doorbell an `ioeventfd` the library's control thread polls, so
ringing it never runs the library on the vCPU. Meson finds the library as
`dependency('reims-vgpu', method: 'pkg-config', static: true)`, the package `cargo xtask dist`
writes (11.3), so `libreims_vgpu.a` is part of QEMU's binary. Under `-display none` the library
opens the device's window; the console shows what the display shows, read back with
`rvg_graphics_device_read_display` on a thread of the device's pool, and takes the guest's cursor
from the cursor callbacks. [qemu.md](qemu.md), "The reims-graphics device", describes it, its
configure lines (`make qemu`), the VM boot method and the results.

---

## 13. Tests

The codebase is its own reference: its unit tests record what it does, and a change passes
everything its predecessor passed. [testing.md](testing.md) says how the code is tested.

### 13.1 `cargo xtask check`

Before a branch merges: the testkit manifest check (D50), `header --check` (11.2), the prebuilt
records' checks (the option ROM's and the guest's kext set's), `cargo fmt --check`,
`clippy -D warnings` over every target, once with every feature and once with the tested features
without the default ones, `cargo nextest run --workspace --no-fail-fast` (each test in its own
process, in the order `.config/nextest.toml` sets on a Mac: the tests that act on the whole GPU
first and alone) with the tested features, and the doc-tests (`cargo test --workspace --doc`). The
tested features are every feature but the C API's `vulkan` (`TESTED_FEATURES` in
`xtask/src/main.rs`), which makes the C API's platform the Vulkan one (11.1); on Linux, where the
Vulkan platform is the only one, it is among them. On a Mac the check also runs clippy for
`x86_64-unknown-linux-gnu`, which type-checks every Linux file with SDL left unbuilt, and, where
Rosetta runs x86-64 code, core's tests as `x86_64-apple-darwin`.

### 13.2 The testkit

`core::testkit`, under `cfg(any(test, feature = "testkit"))`:
- **`FakePlatform`**: every platform trait over small object models, logging each call whose
  position is observable (4.3) and scripted by `Faults` (nil, error, raise or crash at the nth call
  of a selector on a class). A scripted raise is decided as the platform's answer layer decides
  the same call (4.6), by a `Policy` with the same labels. A free call answers from the model and is not logged.
  Its models keep what the contract sees: fresh numbering, command buffers retaining what they
  reference until completion, counted IOSurfaces.
- **Memory**: `GuestRam`, a VM model with remap aliasing and maximum protection, `VirtualClock`
  with both deadlines.
- **Threads**: `FakeThreads` (std threads, recorded; timers on the virtual clock).
- **Harnesses**: `capture` (`catch_death`, `expect_death`, `expect_hang`), and `death`, which re-runs
  one test in a child process to see what a real death leaves behind; `logs`, a capture of the `log`
  crate's lines.

Tests assert what the guest and host observe: guest bytes, MMIO reads, interrupts, host calls in
order, outcomes, IOSurface counts, object creation and release order, and the ordered Metal calls
in the fake's log. They never assert internal field values, and assert a log line only where the
line is the point of the test (14).

### 13.3 The real platform

The Vulkan crate's tests run its platform on the host's device, strict. On a Mac they compare its
answers with Metal's for the same calls, recorded from Metal on the M3 Max under macOS 27.0.1 into
text files beside each test (`metal_answers/`, read by core's `testkit::metal_answers`), never
rewritten from the library's own output.

### 13.5 The C API's harness

- **The C harness**: `tests/c_harness.rs` builds `tests/c/harness.c` as C11 against the committed
  header, links it to the crate's shared library and runs it, and checks that the header compiles
  alone as C++17 and, on macOS, as Objective-C.
- **The exports**: `tests/exports.rs` (11.1).

13.4, 13.6 and 13.7 are retired: the loom models, the behaviour-lock suite and the smoke went with
Apple's paravirtualized device.

---

## 14. Diagnostics

- **A line**: `log::error!`, `warn!`, `info!` or `debug!`, under its module's path as its target,
  synchronously on the caller's thread. Warn is guest or host input refused and handled, error a
  failure the host's operator should see, info lifecycle, and debug detail.
- **Its arguments** are evaluated only when a logger takes the line. A value a line shows whose read
  has an effect (guest memory, which can die; a question to Metal) is read before the line,
  unconditionally and in the same order, so the process does the same with logging on or off.
- **Where lines go** is `logging.rs`'s: the host's own `log` logger, or with `REIMS_VGPU_LOG` the
  library's, to stderr.
- **Tests** that are about a line read it through `testkit::logs::capture`, which relies on
  nextest's one process per test.

Kept: every read a line's arguments did, before the line. Free (diagnostics): the text, targets and
levels of lines.

---

## 15. Guest memory, lifetimes and drop order

### 15.1 Guest memory

The guest's RAM is the host's memory map (`memory::MemoryMap`, from the config's `RvgMemoryMap`),
with the files behind its ranges. The library reaches it only through the host's mapping, by copies
and through the platform's imports of it (on Linux, views of the guest's pages mapped from their
files, and dma-bufs of them, `reims-vgpu-vulkan/src/os/linux`), never through a Rust reference
(D18). The device's own memory is its region, BAR 2, which the library maps and the guest maps
write-back.

### 15.3 Drop order is explicit

A type whose fields' drops have effects in an order the API shows has an explicit `Drop` that takes
them in that order: the device's handle lets go of its window, and with it the window table, before
the device, so the window table's `destroy` runs first (`reims-vgpu-c/src/graphics.rs`). Field
declaration order is never relied on. The same holds for locals whose drop reaches the host or the
guest (7.4): each is a named `drop` or a binding that outlives the call, never a temporary's end.

15.2 and 15.4 are retired with the paravirtualized device's two-tier lifetimes and device state.

---

## 16. What the contract keeps

Where the contract leaves the code free, it is plain Rust. The mechanisms below exist because what
crosses the API, or the platform under it, forces them.

| mechanism | where | forced by |
|---|---|---|
| a real autorelease pool around each unit of work, and a last-resort one on the library's threads | `rt/pool.rs`, `rt/queue.rs`, `reims-vgpu/src/entry.rs`, `vulkan/src/os/macos/autorelease.rs` | Metal: what it autoreleases (an uncommitted command buffer in a dev build, its encoders) must go at the unit's end, or the queue's 65th `-commandBuffer` hangs |
| named release points for a drop that crosses the API (`Held`, `Hold::keep`) | `rt/held.rs` | the order of host callbacks and IOSurface retain counts against the rest of the API (7.4) |
| a label at each raise site: `Err`, a death at the raise, or the caller's `OnFault` | `raise.rs`, `raise/on_fault.rs` | where a run dies: a raise no frame handles ends the process at the raise, before any drop (6.2) |
| every Objective-C call inside a catch: the message layer around AppKit, Foundation and IOSurface, `host_call` around every host call | `vulkan/src/os/macos/msg.rs`, `reims-vgpu-c/src/ffi.rs` | an exception out of Metal, Foundation or host code ends the process at the call (D35, 6.7) |
| one typed call per Metal method, each ordered call where its caller makes it, with raw guest values; descriptor values applied to real descriptors in a stated order | `platform/`, `vulkan/src/answer.rs`, `vulkan/src/gpu/` | Metal's results, aborts and numbering (resource indices, GPU addresses) follow the calls a process makes (D4, D5) |
| one display thread per process for the cursor callbacks | `reims-vgpu/src/platform.rs` | the order of the cursor callbacks across devices (D49) |
| a window that closes at the device's release first, and waits for nothing, the main thread included | `reims-vgpu/src/window.rs`, `reims-vgpu-window/src/handle.rs` | where the tables' `destroy`s land, and a host whose main thread creates the device before it runs its run loop (D51) |
| `extern "C"` exports, each inside `ffi::export`, and the library's panic hook | `reims-vgpu-c/src/ffi.rs`, `singleton.rs` | the header: nothing unwinds into the caller, and a panic of the library's ends the process before any drop (6.5) |

---

## 17. Conventions

[conventions.md](conventions.md) has the full list; the ones code cites:

### 17.1 Naming

Names say what things mean (`Raise`, `Held`, `SerialQueue`). No Objective-C class or
selector-keyword names. A field of a guest record, the wire or a copy of a Metal struct whose
meaning nothing states is `field_xx`, its byte offset in hex; a name that is ours carries
`// name is ours: <evidence>`.

### 17.2 Odd behaviours and comments

An odd behaviour the guest or host can see carries a doc comment at the code that says what it is
and who sees it, and a test named for the behaviour. Changing it changes what the guest or host
sees, and is done only as a deliberate change in a commit of its own. Comments say why, in terms of
what the guest or host observes, and may name a test.

### 17.3 `unsafe`

Core denies `unsafe_code` at its root (D43). It is allowed, with a `// SAFETY:` comment per block
(`clippy::undocumented_unsafe_blocks`), only in:
- `platform/host_range.rs`: `HostRange`'s constructor;
- the testkit (`clock.rs`, `guest_ram.rs`, `death.rs`: `mmap` over live pages, async-signal-safe
  handlers and `_exit`).

The Vulkan crate, the window crate, the API and the C API use `unsafe` at their
FFI edges and where they make a no-copy buffer over memory they hold mapped.

### 17.4 Lints and integers

Core denies `indexing_slicing`, `unwrap_used`, `expect_used`, `panic`, `unreachable`,
`disallowed_types`, `disallowed_methods` and `mod_module_files` everywhere, and `as_conversions`,
the lossy casts, `integer_division_remainder_used` and `undocumented_unsafe_blocks` outside tests.
Arithmetic uses std's explicit forms; `num.rs` holds the rest: wrapping alignment, the lossless
`usize` conversions, truncation to the low bits, and the integer results the guest sees, each a
named rule that says why it is kept. It is the only production module of core that may use `as`.
`#[expect(lint, reason = "…")]` on the narrowest item replaces an `allow` with a comment.
