# reims-vgpu: architecture

This is how `libreims_vgpu` is built: its crates, what each part owns, its threads and locks, how
it fails, where it meets Apple's frameworks, and the interfaces the host and the suite reach it
through.

The library is the host side of the paravirtualized GPU macOS guests drive, in place of Apple's
`ParavirtualizedGraphics.framework`, and answers to the contract the guest and the host rely on:
what crosses the API, in its order. That is the C API's results and refusals; the host callbacks
(create and destroy task, map, unmap and read memory, raise interrupt, add and remove trace range,
the display callbacks, a display window's input callbacks, each callback table's `destroy`) with
their arguments, their order and the threads the header names for them; the order of operations on
guest memory and on retained objects (unmaps, releases, IOSurface use counts); guest memory and MMIO
reads; interrupts, in their order against the guest-visible writes; display frames, pixels, modes
and the cursor; GPU results; every value that reaches the guest (resource IDs, GPU addresses,
numbering, stamps); the suspend state; and whether a run completes, dies or hangs, at the same guest
op. The codebase's own unit tests, its pins and the behaviour-lock suite's expected files record
that contract, and a change passes everything its predecessor passed (13).

Everything else is free: which library thread runs what where the header names no thread,
internal locking, how isolated tasks are hosted and reach the device, the retention machinery
where the order of releases the host sees is unchanged, Metal's internal call order where no
result or guest-visible value changes, and diagnostics (log lines, thread and queue names, Metal
labels, crash messages, signals). Much of the structure below is there because the contract
fixes it; each such place says what fixes it, and section 16 collects them.

Where the rest is:
- `README.md`: what the project is and how it is laid out; `tests/suite/README.md`: the
  suite's tooling;
- `docs/api.md`: the C API and the Rust API it wraps, the thread each callback arrives on, and
  how QEMU's `apple-gfx` and the suite's runner use them;
- `docs/testing.md`: the behaviour-lock suite's specification, the guest ABI (section 6) and the
  odd behaviours the suite records (section 7);
- `docs/conventions.md`: the coding conventions (17 repeats the ones code cites);
- `docs/qemu.md`: QEMU on the library, its build and VM boots;
- `docs/lessons.md`: traps the suite and reviews found;
- `docs/accuracy-ledger.md`: what the Vulkan platform's defaults trade against Metal's answers,
  and the switches that take Metal's;
- `crates/reims-vgpu-window/features.md`: what a display's window does for the person at the
  host, each entry deliberate and guarded;
- module `//!` docs: each area's ordering contracts and the odd behaviours it keeps, each naming
  its test, and the platform facts the area relies on.

Code comments that cite a decision (`D17`) or a section (`architecture 13.2`) cite this
document.

## Contents

1. Decisions
2. Workspace and crates
3. Core's modules
4. The platform boundary
5. Isolated tasks
6. Outcomes: expected, raise, death
7. Concurrency
8. Containers
9. How the areas reach one another
10. The Rust API (`reims-vgpu`)
11. The C API (`reims-vgpu-c`), the dist tree and the option ROM
12. QEMU
13. Tests
14. Diagnostics
15. Guest memory, lifetimes and device state
16. What the contract keeps
17. Conventions

---

## 1. Decisions

The decisions code cites by number, each in force as written here. A number missing from the
table is retired and not reused.

| # | decision | why |
|---|---|---|
| D1 | One Cargo workspace at the repository root: six product crates (`reims-vgpu-core`, `-metal`, `-vulkan`, `-window`, `reims-vgpu`, `-c`), and `xtask`. `reims-vgpu-vulkan` is a second `Platform`, over `metal2vulkan-runtime`, a Metal-shaped API on Vulkan: the library's platform on Linux, and on macOS one the API's `vulkan` features select, the Metal platform staying the product there (2.1, 2.2). | Each crate does one job the others cannot without breaking a rule below. |
| D2 | Modules by area and responsibility, not one file per Objective-C class. | The code is Rust; the contract binds behaviour, not file shape. |
| D3 | Core is generic over one `P: Platform` whose associated types name every Metal object kind. `dyn` only for the host's callbacks and the display executor, the OS services, the transport and one internal seam (9). | A handle flows into another Metal call (a texture into an encoder) with its kind checked at compile time. |
| D4 | One typed platform method per Metal method core calls, and one enum variant per encoder and ICB command (`BlitCmd`, `ComputeCmd`, `RenderCmd`, `IcbRenderCmd`, `IcbComputeCmd`), with no table of ops between core and Metal. The metal crate's dispatch is one exhaustive `match` per command family, and it names each send by its selector and raise policy (4.3, 4.6). A call whose position the guest or host observes (a creation, release or retain, an aliasing setter, a command, a call with raw guest values) is sent where core makes it; a getter or device query is not ordered. | A send per guest operation, in order, with raw guest values: Metal's results, aborts and numbering follow the calls a process makes. |
| D5 | A descriptor core only fills in and hands to a creator is a value (`platform/values.rs`), which the platform applies to a fresh Metal descriptor inside the creator, sending its setters in the order the value's type states. A descriptor Metal checks when it is set (the texture descriptor's pixel format, the render pass's sample positions, the rate map's layers) or makes itself (the pipeline descriptors from serialized data), or whose attachments retain what they are given (the render pass), stays a Metal object behind a handle (4.3). | Metal folds some properties and aliases others, so the platform sends the raw values to a real descriptor in a fixed order rather than model them; an object stays where Metal's check at the setter, or a retain, is observable. |
| D6 | Handles are not `Clone`. Another reference is `.clone_ref()`; `Drop` is the release. | IOSurface retain counts are guest-visible (a mapper unmap is `Busy` unless the count is 1), so no retain is taken that the count would show. |
| D7 | The raise channel is a value (`Raise`, `RaiseResult`), decided at the raise by its site: `Err` where a handler is always live, a death where none ever is, before any `Drop` runs, and the caller's `OnFault` where both are reached (6.2). There is no real `NSException` throw per raise and no run-time record of handlers: a raise is a value, and a throw costs time for nothing the API shows. | The runtime's two-phase unwinder: with no handler, phase 1 fails and the process ends before any cleanup. Which frames handle a raise is fixed by the code's paths, so each site knows it, and a callout's code, which no handler of its caller reaches, raises as the site it is. |
| D8 | Three outcome channels, never converted into one another: expected failures, raises, deaths (6). | The contract observes each separately: a refusal or error, an exec fault, a process death. |
| D9 | `panic = "unwind"` in every profile. The API's singleton installs a panic hook that aborts before unwinding. A death is `abort()`, except one in an isolated task's own code, which ends the task (5.5). | objc2's `exception::catch` needs unwind; an unintended panic still dies at its op with no side effects (6.5). |
| D10 | Isolated guest tasks run in the host's process: one fresh service (`GpuTask`) per isolated task, on threads of its own (`tasks::isolation::TaskThreads`). No pooling, no reuse (5). | A death in the task's own code ends that task and nothing else, and the guest sees status 4; a crash inside Metal or the driver ends the process, as it does outside isolation. The task's objects come from the host's Metal device, so their resource indices and GPU addresses follow the process's numbering. |
| D11 | A remote task and its service speak our own messages (`tasks/wire.rs`) over an in-process link (`tasks/isolation/link.rs`): the host's messages on the task's connection queue, the service's messages and replies on the link's queue. The service reads guest RAM through the host's own mapping of it; a lent IOSurface reaches it as another object over the same surface. | Only what crosses the API is the contract; the transport between a remote task and its service is ours. |
| D12 | The service lives in core behind the transport traits (`tasks/launcher.rs`). The product's launcher, `TaskThreads`, is also the testkit's, over the fake platform (`testkit::launcher`). | Tests get a death at op K deterministically, on the product's own path. |
| D13 | In-process tasks (the blocks and map setups, task 0 everywhere) are `LocalTask`; an isolated task's service is `GpuTask`. Both share `TaskCore` (`tasks/shared.rs`). | The two differ in hundreds of places the guest and host see (testing.md 6.10, 7.2). |
| D14 | Containers are std's, each named by the order it gives (8). Where an order reaches the guest or the host it is one of ours, stated at the container: the task table and the outstanding completions go in ascending ID. | Those orders are recorded in the suite's expected files; nothing depends on a hash table's order. |
| D15 | The root task's VA allocator is pure Rust in core, never behind the platform, and never reset. | Its offsets reach the host as `map_memory`'s offset, in over a thousand expected files. |
| D16 | Queues, threads, locks, condition variables and signals are ours (7.1). A lock keeps the scope, and a race the outcome, that decides what crosses the API: what an unlocked MMIO or guest read sees, which call hangs, the handler calls a burst of merges gives. No runtime lock-order checker. | A checker that panics would turn the hangs the API shows into deaths. |
| D17 | Process-wide state lives in `ProcessState` (`process.rs`), one per platform instance, a singleton in production. | Process-wide where the API shows one state per process (a display's serial number, the latest IOSurface device, the report budget); every test hermetic. |
| D18 | Guest memory is reached only through `GuestPtr` and `GuestPage` accessors. There is never a Rust reference into guest RAM; a NULL dereference is a death (15.1). | The guest writes concurrently; a NULL dereference in Rust is undefined behaviour, not a crash. |
| D19 | Integer widths and wrap points are visible: std's `wrapping_*`, `overflowing_*`, `checked_*`, `From`/`TryFrom`, and `num::` for the few with odd semantics (17.4). | Widths and wrap points are behaviour. |
| D20 | Closed sets are enums: resource kinds, task kinds, decoders, mapping teardowns. An operation a kind does not answer raises `RaiseKind::Unrecognized` (`resources::resource::ResourceClass`). | Which kind answers which operation, and which raise the rest give, is behaviour the guest sees (`resource::responds_table`). |
| D21 | One `dyn` seam remains, `ResourceHost`, because it has two implementors; everything else is concrete (9). | |
| D22 | Values drop where Rust's scoping puts them. Each unit of work (a command, a work item, an API entry) pushes a real pool (`rt::Pool`) for what Metal autoreleases, and a release that crosses the API is held to its point in the API's order: by a holder (`rt::Held`), with the raise that carries it, or by name (7.4). | Metal's autoreleased returns must be drained (the 65th command buffer); host callbacks, unmaps and IOSurface counts are observable, the lifetime of core's own values is not. |
| D23 | Stamp and task completions are `Arc<dyn Fn>`, callable more than once. | A second completion is a death at that op; `FnOnce` would silently remove it. |
| D24 | The display's present shaders and the XR10 paging kernels are the library's own, in Metal Shading Language (`metal/src/gpu/shaders.metal`), compiled from source at run time once per Metal device (4.7); the Vulkan platform loads metallibs built from the same source. | No shader of Apple's is read at run time or committed, and the Metal platform's build needs no Metal toolchain. |
| D26 | The configs cover core's input space: every value core refuses or dies on stays representable, and there is no lifecycle typestate. | Refusals and odd call orders are part of the contract (testing.md 3.3, 6.24, 7.3). |
| D27 | The C API: prefix `rvg_`; opaque counted handles; one config struct per object, headed by `struct_size`; one callback table per object, with one `ctx` and one `destroy` (a display with a window has the window's too, from its own config); a status from every call that can fail; no Objective-C types, dispatch queues or blocks; frames copied into memory the host gives; a cbindgen header (11). | A C host calls it with nothing of Objective-C, as QEMU's `apple-gfx` does, and it grows only by addition (api.md section 1). |
| D28 | The suite drives the C API (`reims_vgpu.h`) as QEMU does, through its runner, `tests/suite/runner/pgrun.m`. | `pgrun.m` is hashed into every expected file's input. |
| D29 | QEMU's `apple-gfx` is C on the C API, and links the library statically through pkg-config (12). | |
| D32 | arm64 only, for every Rust binary built for macOS (`aarch64-apple-darwin`); on Linux, `x86_64-unknown-linux-gnu`. | rustc's arm64e target has no pointer authentication, and its binaries crash before `main` (`rust-toolchain.toml` has the measurement). |
| D33 | A path no guest or host input reaches stays unreached: it is not brought to life (the tiled EFI texture path among them). | |
| D34 | Nothing with a `Drop` is live in a frame an Objective-C exception can unwind: a Metal call runs in a closure that borrows its receiver and arguments (a `Retained` result exists only once the method has returned), and a foreign boundary's body runs in an inner function (4.6). | A foreign unwind runs landing pads; a value dropped there is a release at a point the code never chose (`metal/src/msg.rs`). |
| D35 | Every Objective-C call catches its own exception where it starts: a call into Metal or Foundation runs in the message layer (`metal/src/msg.rs`: `send`, `get`, `catch`, `fatal_send`, `value`; `metal/tests/message_layer.rs` keeps them there), and a call into host code under `host_call` (6.7). So a sync block runs directly on the caller's thread. No Rust unwind handler sits below the library's Rust code that sends Objective-C messages either: core makes no such thread itself, and the FIFO thread and the queues' workers are pthreads the platform starts (`Threads`) with an `extern "C-unwind"` start routine. Every `rvg_` function is `extern "C"` (6.5). | An exception out of a call ends the process at that call, before any frame of the library's unwinds (D34), whatever the caller holds; one no call can catch (out of a release's `dealloc`) ends it in its search phase, with no drops (`metal/src/os/threads.rs`). |
| D36 | objc2's debug checks are on in dev builds and off in `checked` (`[profile.checked.package.objc2] debug-assertions = false`); release has none. The methods `metal/src/spi/` declares are also verified against the runtime by `metal/tests/spi_encodings.rs`. | The checks panic on a message to nil, a missing method and an encoding the runtime disagrees with. The library sends none of these (nil receivers are answered in Rust, and the selectors the host does not implement raise without a send), so the tests that send to real Metal check every send they make. |
| D40 | An isolated task's service makes no Metal allocation the guest's commands do not ask for, and while its task runs keeps no Metal object alive past the guest's last use of it; a task that died leaks its objects (5.5). | Resource indices and GPU VAs depend only on the calls the process made and what is still alive: a released object's index and VA go to the objects made after it. |
| D41 | Platform types follow the runtime's encodings: `ResourceAddressRange` is `{MTLAddressRange=QQ}`, the tile dispatch's render target array index is `u32`. | The runtime's `method_getTypeEncoding` gives these types, and objc2's checks compare each send with it (`metal/tests/spi_encodings.rs`). |
| D43 | Core denies `unsafe_code` at its root; the modules of 17.3 allow it. | `forbid` cannot be lowered per module. |
| D44 | One module style: `foo.rs` beside `foo/` (`clippy::mod_module_files`). `reims-vgpu-c` is a `cdylib`, `staticlib` and `rlib`. | The `rlib` lets tests link the C API. |
| D46 | A line goes through the `log` crate's macros under its module's path, on the caller's thread (14). | Diagnostics are outside the contract, so a line needs no fixed site or text. Formatting is paid only by a line a logger takes. |
| D47 | The library is a plain library: `cargo xtask dist` writes a static library, a dylib, the header and a pkg-config package, `reims-vgpu.pc`, into `target/dist` (11.3). | A host links it the way it links its other dependencies; QEMU links the archive, so its binary needs no rpath. |
| D48 | The EFI option ROM is a data file of the dist tree, not the library's: `cargo xtask dist` builds it from `firmware/` (11.4) into `share/reims-vgpu/ReimsVgpu.rom`, `reims-vgpu.pc`'s `optionrom` variable names it, and QEMU installs it as `apple-gfx-pci`'s default `romfile`. The API names no ROM. | Only an x86-64 guest's firmware runs it, and the library never reads it. A host that wants another ROM passes `romfile=`, as for QEMU's own `efi-*.rom` files. |
| D49 | A display's callbacks run on one display thread per process, a core `SerialQueue`, in the order core queued them; every other callback runs synchronously on the thread core calls it on (10; api.md section 5). | The API has no dispatch queues or blocks, and the order in which the host saw its display callbacks on the one queue QEMU gave every display stays. |
| D50 | `core::testkit` (the fake platform, its models and harnesses) is compiled only under `cfg(any(test, feature = "testkit"))`, and only dev-dependencies turn the `testkit` feature on, never a default; `cargo xtask testkit` checks both (13.1). | No product build can name the fake platform or a test harness. |
| D51 | A display's window is the library's: a crate of its own, `reims-vgpu-window`, between the platform and the API, which knows nothing of devices or displays (2.2). A display created with a window config gets an AppKit window (on macOS, or SDL3's with `REIMS_VGPU_WINDOW_IMPL=sdl`, 2.2), into whose `CAMetalLayer` drawable the display's current frame is encoded on the GPU, and whose input goes to the host's window callbacks (10, 2.2); on Linux an SDL3 window, into whose Vulkan swapchain's image the frame is encoded the same way. **Lifetime:** nothing on the window's side holds the display. The tee's closures hold the window, and the sink holds the display weakly and upgrades it for one draw's encode and commit; the encode's completion handler holds it until the frame completes, as a frame's does. The display's last reference closes the window before it lets the display go, and waits for nothing: not the main thread, the drawing or the GPU. **Threads:** the window's AppKit side and callbacks run on the process's main thread, one at a time across every window, and must not block; the library reaches that thread only through blocks it posts to the main run loop, the application's start alone, and never waits for it. On Linux they run on a UI thread of the library's own, `reims-vgpu.ui`, and the main thread is never used (7.3). Each window draws on a render thread of its own, which nothing joins (7.3). **Density:** a display matching its host screen offers, for its first mode, 110 ppi with a mode the host chose (QEMU's `resolution`, 1920 × 1080 by default: a 1× guest the window scales on the GPU), or the density the host chose with it (`mode_pixels_per_inch`, QEMU's `resolution=2560x1600@200`), else the screen's own modes at ppi = 110 × the screen's backing scale; the size in millimetres = pixels / (ppi / 25.46). The scale factor is 2 above 143 ppi, else 1, for the screen's own modes, and for a chosen density the scale the window takes the guest to run. A window opens at a guest point per host point by the scale it takes the guest to run each mode at: 2 where the mode, at least 1920 pixels wide, comes to 146.667 ppi or more across the width as the guest reads it, in whole centimetres (rounded to whole millimetres, then up). What the window does for the person at the host is `features.md`'s register. | A window that held its display would move the display's end, and with it the releases the host sees and its table's `destroy`; a window frame keeping it as a frame does puts both where the pins `api.lifetime.display_through_*` and `api.destroy.display_*` already allow. QEMU under `-display none` creates its displays on the main thread, holding the BQL, before that thread runs its run loop, so a create or release that waited for the main thread would hang (`api.window.no_run_loop`; the window crate's `native/macos/main_thread.rs`). The density rule gives a Retina screen a HiDPI guest desktop and a 1× screen a 1× one: a macOS 13.6 guest picks its default from the width in whole centimetres alone (rounded to whole millimetres, then up), not from the scale factor, and runs 1× at up to 146.666 ppi across it and HiDPI from 146.667, for modes 1920 to 3811 pixels wide (measured, `policy`, which also lists the wider and narrower modes that do not follow it), so 72 ppi per unit of scale gave a Retina screen a 1× desktop; the divisor 25.46 is kept, since changing it changes the size every such guest is told. |

---

## 2. Workspace and crates

### 2.1 Layout

```
Cargo.toml Cargo.lock rust-toolchain.toml clippy.toml .cargo/config.toml
crates/
  reims-vgpu-core/   all behaviour, generic over P: Platform; links no Apple framework
  reims-vgpu-metal/  Platform over Metal, IOSurface, Mach VM and threads; the library's shaders
  reims-vgpu-window/ a display's own window: AppKit, a CAMetalLayer, input, the host's screens,
                     and features.md, the register of what the window does
  reims-vgpu/        the Rust API: core over MetalPlatform, and a display's window
  reims-vgpu-c/      the C API (cdylib, staticlib, rlib) and include/reims_vgpu.h
  reims-vgpu-vulkan/ Platform over metal2vulkan-runtime: the library on Linux, an option on macOS
xtask/               testkit, header, dist, metallibs, smoke, check
firmware/            the option ROM: the PCI device's UEFI driver, in C, and its Makefile (11.4)
tests/suite/         the behaviour-lock suite (pgtest, pgrun.m, scenarios, expected files)
scripts/             install-ctl.sh: installs reims-ctl and its QEMU
scripts/internal/    run_vmapple.sh, vm_drive.py, vm_ocr.swift: VM boots (docs/qemu.md);
                     walk_window_macos.sh: the window's manual walk (features.md)
vendor/qemu          QEMU, whose apple-gfx is C on the C API
```

`cargo xtask dist` writes `target/dist` (11.3, D47): the static library and the dylib, the header,
`reims-vgpu.pc`, and `ReimsVgpu.rom`, the option ROM QEMU hands an `apple-gfx-pci` guest, built from
`firmware/` (11.4, D48). The library compiles its own shaders (D24), and reads no preferences domain
and no entitlement: the device's options are its config's fields, and a display's nits and flags
keep their defaults unless a Rust host hands the display a `Preferences` reader (4.5).

### 2.2 Crate graph

```
reims-vgpu-core  <-  reims-vgpu-metal  <-  reims-vgpu-window  <-  reims-vgpu  <-  reims-vgpu-c
reims-vgpu-c  <-  QEMU, the suite's runner, the pins
reims-vgpu-core, reims-vgpu-metal (macOS), metal2vulkan-runtime  <-  reims-vgpu-vulkan
reims-vgpu-vulkan  <-  reims-vgpu, reims-vgpu-c  (their `vulkan` feature: off by default on macOS,
                                                  required on Linux)
```

- **The Vulkan platform** (`reims-vgpu-vulkan`) is a second `Platform` over
  `metal2vulkan-runtime`, the Metal-shaped API on Vulkan of metal2vulkan, at the commit of
  its public repository `Cargo.toml` names, on the host's Vulkan driver, or MoltenVK on a Mac. On
  macOS it takes its OS services (IOSurfaces, Mach VM, the clock, threads) from the metal crate;
  on Linux it has its own (`os/linux/`: a host VM over `mmap` of the guest RAM's files, and an
  IOSurface stand-in over that memory). The API's and the C API's `vulkan` features take it
  (10, 11.1): on Linux the library builds only with them, and on macOS `cargo xtask dist
  --platform vulkan` builds a tree on it (11.3), the Metal platform staying the product and the
  default. Where its answer differs from Metal's it logs the difference and goes on, and dies
  where Metal would or where the runtime cannot run the call (api.md section 2; the ledger,
  `docs/accuracy-ledger.md`). A method it has not built dies at its selector.

- **Core** depends on `parking_lot`, `zerocopy`, `bitflags` and `log` only (`loom` under
  `cfg(loom)`). Its `testkit` feature exposes `core::testkit` outside core's own tests; only
  dev-dependencies enable it, and `cargo xtask testkit` checks that (D50).
- **Metal** implements `Platform` and compiles the library's shaders (4.7). It holds every
  `#[link]` but the window crate's two, and every call into Metal, Foundation or IOSurface runs in
  its message layer (4.6), the API's own few included. It depends on core for its traits and value
  types only.
- **The window** (`reims-vgpu-window`) is a display's own window (D51), and knows nothing of
  devices or displays. It depends on core (its `sync::Mutex` and `Condvar`, `rt::BlockEnv` and
  `rt::spawn_named`) and metal (the message layer, and the platform's threads), on objc2's AppKit,
  QuartzCore, Metal, CoreGraphics and CoreFoundation crates, and on `block2` and `libc`.
  - `handle.rs`: the `Window` a display holds, `Send` and `Sync` with no AppKit object in it,
    and `FrameSink`, the trait it draws through;
  - `config.rs`: `WindowConfig` and the host's input callbacks;
  - the pure logic, each module unit-tested with no AppKit: `geometry` (the image's rectangle in
    the window, the pointer's position in the guest's mode, the buttons), `keymap` (macOS key
    codes to USB usages, by position), `keys` (modifiers and held keys), `capture` (who gets the
    keyboard, and where the pointer may go), `scroll` (wheel and trackpad deltas to notches, or
    smooth with their gestures' phases), `policy` (the modes and density offered for a host
    screen) and `env` (the `REIMS_VGPU_WINDOW_*` variables);
  - `screens.rs`: the host's screens from CoreGraphics, which answers on any thread;
  - the AppKit side (`native/macos/`, behind `native/macos.rs`), on the main thread (7.3), whose
    state lives in statics only that thread reaches (`main_thread::MainOnly`):
    - `main_thread.rs`: `post_main`, which posts work to the main run loop in order, and the
      application's start, posted alone, with the work posted before it has launched held back;
    - `app.rs`: the application, once per process: `ReimsApplication`, whose `sendEvent:` gives
      the key window's input each event before AppKit dispatches it (a local event monitor does,
      for key events, in a host's own application), its delegate, the menu, the activation
      policy, the activity against App Nap, Quit, and the screens' changes;
    - `shell.rs`: a display's `NSWindow` (`DisplayWindow`) and its delegate, windowed or
      borderless, on the screen matched by its CoreGraphics number, and the windows open, by
      the `Window`'s id;
    - `view.rs`: the content view, layer-hosting over a `CAMetalLayer`, which lays the image out,
      gives its events to the input and sets the cursor's rectangle;
    - `input.rs`: `InputState`, the view's and the application's events to the host's
      callbacks, deciding in an `InputLogic` over plain values, unit-tested with no AppKit;
    - `cursor.rs`: the guest's glyph as an `NSCursor`, and an invisible one;
    - `hotkeys.rs`: the system's hot keys, off while any window captures the keyboard;
    - `render.rs`: each window's render thread (7.3), and the `Surface` it shares with the window
      and its view;
  - `features.md`: the register of what the window does for the person at the host, each entry
    guarded by a named test (`features::features_md_guards_exist` fails when one is gone) or a
    manual check on a VM boot.

  **On Linux** the window is SDL3's (`native/linux/`, behind `native/linux.rs`, with SDL's
  shared shell and cursor in `sdl/`), on Wayland or X11, built from SDL's pinned source release
  with its video subsystem alone and linked statically (`sdl3-sys`), and drawn through Vulkan on
  the runtime's own device. SDL runs on a thread of the library's own, `reims-vgpu.ui` (7.3):
  `native/linux/ui.rs` (the thread, the work posted to it, and the host's screens as SDL lists
  them), `sdl/shell.rs` (each window's opening, size, title, layout, events and end),
  `native/linux/input.rs` (its events to the host's callbacks, over the shared pure logic),
  `sdl/cursor.rs` (the guest's glyph as SDL's cursor, with an image at the screen's own scale),
  `native/linux/keymap.rs` (SDL's scancodes, USB usages already) and `native/linux/render.rs`
  (the render thread, which draws through the `Presenter` the display's sink makes over the
  window's swapchain, `present.rs`, shared with macOS's render thread on the Vulkan platform).
  It depends on the Vulkan crate for the platform's threads, and on ash for the Vulkan types of
  a window's surface.
  What Linux does its own way (placing, the surface, SDL's input events, the end) is
  `native/linux/sdl.rs`, the shell's `os`.

  **On macOS through SDL3** (`REIMS_VGPU_WINDOW_IMPL=sdl`, in a build with reims-vgpu-window's
  `macos-sdl` feature, on by default; AppKit's window stays the default until SDL3's passes
  `features.md` on a VM boot) the same shell and cursor run on the main thread, inside the
  application `native/macos/app.rs` runs, and the shell's `os` is `native/macos/sdl.rs`: SDL
  starts as the application launches and never runs a loop of its own, an event watch posts a
  drain of SDL's queue to the main run loop (`native/macos/sdl/ui.rs`); where a window opens, how
  it follows the guest's mode and borderless windows are AppKit's window's frames over SDL's
  `NSWindow` (`shell.rs`'s); the render thread is `render.rs`'s, into the `CAMetalLayer` of SDL's
  Metal view (on the Vulkan platform through a Vulkan surface of that layer, as for AppKit's
  window, `native/macos/metal_surface.rs`, not SDL's own Vulkan surface, which would make another
  Metal view); and the application's `sendEvent:` gives AppKit's events to `input.rs`'s
  `InputState` (`native/macos/sdl/input.rs`), so keys, capture, the hot keys and scrolling are
  AppKit's window's.

  It holds two `#[link]`s of its own: Carbon's over `keymap.rs`'s `extern "C"` block
  (`KBGetLayoutType` and `LMGetKbdType`, for the ISO keyboard check), and AppKit's over
  `app.rs`'s, for the `NSApp` global, which says whether the host has made an application.
  SkyLight's `CGSMainConnectionID` and `CGSSetGlobalHotKeyOperatingMode` are looked up with
  `dlsym` (`hotkeys.rs`), since the package's link line cannot name a private framework.
  CoreGraphics, for the screens, comes through objc2-core-graphics, which links it.

  **Exceptions and panics** (D35, 6.5). Every send to AppKit, Foundation, QuartzCore or Metal
  runs in the metal crate's message layer, and nearly all through `msg::get`, so an exception
  ends the process at its send. Two kinds of send catch theirs (`msg::catch`) and go on: an event
  passed on to AppKit (`view::pass`, for a view's event methods' `super` and the application's
  `sendEvent:`), whose exception is logged and the event dropped, as `-[NSApplication run]`
  would; and `setPresentationOptions:`, whose exception leaves the Dock and the menu bar as they
  were. Every `define_class!` method and every block posted to the main thread starts in
  `BlockEnv::foreign`, which marks the thread as the library's, so a panic there aborts before it
  unwinds into AppKit. The host's window callbacks run under the C API's `host_call` (6.7).
- **The API** (`reims-vgpu`) is where core meets `MetalPlatform`: its types are generic over an
  `ApiPlatform`, `MetalPlatform` by default, so its unit tests run its forwarding on the fake
  platform (10). It joins a display to its window (`window.rs`): on the Vulkan platform, on
  either OS, through `window/vulkan/swapchain.rs`, whose presenter draws each frame into the
  window's swapchain (`reims-vgpu-vulkan/src/swapchain.rs`) with no copy.
- **The C API** (`reims-vgpu-c`) forwards to the API and adds no behaviour (11).

### 2.3 Profiles

| profile | what | notes |
|---|---|---|
| `release` | the product | `panic = "unwind"` (D9), no overflow checks (wrapping is explicit), thin LTO |
| `dev` | tests | overflow checks on |
| `checked` | the suite's second pass | release with overflow checks and debug assertions |

objc2's debug assertions are on in `dev` and off in `checked` (D36). Every macOS binary is
`aarch64-apple-darwin`, and the Linux library `x86_64-unknown-linux-gnu` (D32). `cargo xtask
dist` builds the library for macOS 11.0, rustc's default, so the C that objc2's exception helper
compiles is built for it too.

---

## 3. Core's modules

`crates/reims-vgpu-core/src/`, one file per module beside its directory (D44). Each module's `//!`
doc holds its ordering contracts and the odd behaviours it keeps, each naming its test.

| module | what it owns |
|---|---|
| `raise`, `raise/{on_fault,death,kept}` | the raise channel, `OnFault`, deaths, values kept for good (6) |
| `logging` | the library's `log` logger: silent by default, stderr with `REIMS_VGPU_LOG`, or the host's hook (14) |
| `num` | integer operations std has no single spelling for (17.4) |
| `collections` | the containers, named by the order they give (8) |
| `rt` | `SerialQueue`, `BlockEnv`, `Pool`, `Held` and `Hold`, `CoalescingSignal`, `Timer`, `ProcessOnce`, the FIFO thread (7) |
| `sync` | `parking_lot` re-exports, `WriterPrefRwLock`, `StagedMutex` (7) |
| `platform`, `platform/*` | the `Platform` trait, handle traits, the command enums, descriptor values, device caps, the OS services (4) |
| `guest` | `GuestPtr`, `GuestPage`, the guest's records and ABI codecs (15.1) |
| `host` | what the host hands over: its callbacks (`HostBlocks`, `HostCallbacks`), the executor a display's jobs run on (`HostQueue`) and its opaque values |
| `process` | `ProcessState`: serial numbers, the latest IOSurface host device, the report budget (D17) |
| `memory` | the memory map, mappings, the page-table walk, the range tracker, cursors |
| `device`, `device/*` | the device: config, MMIO, the root, FIFOs, stamp waits, tasks, displays, lifecycle |
| `fifo`, `fifo/*` | guest FIFOs: ring, stamp queue, thread, wire and each opcode's handler |
| `tasks`, `tasks/*` | `Task`, `LocalTask`, `RemoteTask` with its endpoint and completions, the transport traits, the wire, isolated tasks' threads, link and life |
| `gpu_task`, `gpu_task/*` | an isolated task's service: handshake, memory, paging, exec, surfaces, XR10 blits |
| `deserializer`, `deserializer/*` | the command-stream walk and its render, compute, blit and info decoders |
| `resources`, `resources/*` | buffers, heaps, ICBs, textures, and the resource manager |
| `display`, `display/*` | the display, its guest display (the guest's side of a port), the EFI display, frames, gamma, pipelines |
| `iosurface`, `iosurface/*` | the IOSurface host device: registers, ring, descriptor table, page walk |
| `seams`, `seams/links` | the one `dyn` seam and the cross-area link aliases (9) |
| `testkit` | (tests only) the fake platform, harnesses and models (13.2) |

`pub(crate)` is the default; stateful types have no `pub` fields except where a sibling area
reaches them directly.

---

## 4. The platform boundary

### 4.1 Shape

`platform.rs` defines `Platform`, with one associated type per Metal object kind core touches
(device, queues, command buffers, the three encoders, buffers, textures, heaps, state objects,
pipelines, libraries, events, ICBs and their commands, rate maps, descriptors) and the OS
services (4.5). Two implementations: `reims_vgpu_metal::MetalPlatform` and the testkit's
`FakePlatform`. There is no table of ops between them and core: each Metal method core calls is
a typed method (4.3).

- **`Handle`**: every object handle. `clone_ref()` is a retain, `Drop` a release, `id()` its
  `ObjId` (the object's pointer, for identity checks). No `Clone` (D6).
- **Nil** is `Option<H>`; a message to nil is written with std's combinators, or `nil_send!`
  where the answer is the type's zero (`NilDefault`).
- **Host memory**: the few methods that take a host pointer (`new_buffer_over_host_memory`,
  `replace_region`, `set_contents`, the texture fills) take a `HostRange<'_>`, whose one
  constructor is `unsafe` and is called where a lease proves the range stays mapped
  (`guest/ptr.rs`, `memory/mapping.rs`, the no-copy buffer sites of 17.3).

### 4.2 Conventions every method follows

- A method that can raise inside Metal returns `RaiseResult<T>`; zero-argument property getters
  and descriptor property setters do not (their raise is fatal at the send, 4.6). A method takes
  the caller's hold (6.2) only where its work holds deep: the pipeline-descriptor
  deserializations, whose function resolver runs core's lookups.
- An object argument core can pass as nil is `Option<&P::X>`; one it never passes as nil is
  `&P::X`.
- Integer widths are the runtime's encodings (D41). Raw guest values reach Metal unvalidated.
- Creators keep one method each (about 30), because each differs in `+1` or `+0`, nil, `NSError`
  or raise.
- Only calls made straight-line with nothing fallible between are grouped into one (a
  sub-object chain such as `colorAttachments[i].loadAction`, a run of pure getters). Nothing is
  grouped across a lookup, a guest read, a log, a lock or a raise.

### 4.3 Commands, values and caps

Each Metal method core calls is one typed method of a platform trait, sent where core calls it.
- **Encoder and ICB commands** are `BlitCmd`, `ComputeCmd`, `RenderCmd`, `IcbRenderCmd` and
  `IcbComputeCmd`: `encoder.encode(on, cmd)` (the ICB commands' `encode(cmd)`) sends exactly one
  selector per variant, the one `cmd.selector()` names.
- **Device capabilities** are one `DeviceCaps` (`platform/caps.rs`): every query is sent once per
  device, in no order core observes, and the metal crate keeps the answers by registry ID.
- **A descriptor core only fills in** is a value (`platform/values.rs`: buffer, heap, sampler,
  depth-stencil, ICB, compute pass, render pipeline, and the texture and rate map values a made
  descriptor starts from). Each type's doc states the order its setters are sent in, which the
  metal crate's `apply_*` functions (`gpu/desc.rs`) and the fake follow. The storage properties
  that alias one another (`resourceOptions`, `storageMode`, `hazardTrackingMode`) are an
  ordered `OptionSet` list, applied in the order core recorded them.
- **A descriptor core keeps** is a Metal object (`platform/descriptor.rs`, D5): the texture
  descriptor, made from a `TextureDesc` where core builds it and changed later by typed setters;
  the render pass descriptor, whose typed sets take a record's `PassSet` or attachment `*Set`
  whole; the pipeline descriptors Metal makes from serialized data; the rate map descriptor and
  its layers.
- **An ICB command's arguments** are values (`DrawArgs`, `DispatchThreadsArgs`, …) read whole
  from Metal's private argument objects.

The calls that keep their place in core's control flow, with their receiver and arguments, are
the creations, releases and retains, the aliasing setters, the commands, and the calls with raw
guest values (the texture descriptor's `setPixelFormat:`, `setSamplePositions:count:`, the ICB
getters at an index, the rate map's coordinate queries). A getter, a device query or a value's
other setters are free: only their answers are observable.

### 4.4 Families

| file | traits |
|---|---|
| `platform/device.rs` | `GpuDevice`: queries and creators |
| `platform/command.rs` | `CommandQueue`, `CommandBuffer` (commit, completion handler, status and error) |
| `platform/blit.rs`, `compute.rs`, `render.rs` | the three encoders and `ResourceRef` |
| `platform/resource.rs` | `GpuResource`, `Buffer`, `Texture`, `Heap`, state objects, pipelines, `Library` |
| `platform/descriptor.rs` | the descriptors core keeps as Metal objects |
| `platform/values.rs` | the descriptor values and the render pass's `*Set`s |
| `platform/caps.rs` | `DeviceCaps` |
| `platform/icb.rs` | indirect command buffers and their commands |
| `platform/rate_map.rs` | rasterization rate maps |
| `platform/event.rs` | events and shared events |
| `platform/types.rs` | value types (`MtlSize`, `MtlRegion`, pixel formats, enums) |

### 4.5 OS services

`platform/os/`: `clock` (monotonic and wall time, `wait_until`), `threads` (the threads core runs
on and the timers' clock), `autorelease` (the real pool), `surface` (IOSurface creation, getters
and retain counts), `vm` (Mach VM: allocate, remap, protect, deallocate). The metal crate
implements each over the system; the testkit models each (13.2). The task launcher (5.3) is
core's `TaskThreads` on every platform, the testkit's wrapped to record what it launched.
`prefs` holds no service: it is the `Preferences` interface the Rust API's
`DisplayConfig::apply_preferences` and `DisplayMode::with_preferences` take, which nothing in
the workspace implements over a system domain.

### 4.6 The metal crate's message layer

Metal is called the Rust way: objc2-metal's typed methods, with objc2's `Retained` ownership.
What objc2-metal does not declare, or declares in a way that cannot carry what core passes, is
declared in `metal/src/spi/`:
- `spi/methods.rs`: extension traits of the objects that answer them (`DeviceSpi`,
  `RenderEncoderSpi`, `TextureDescriptorSpi`, …), one `msg_send!` per method, typed from the
  runtime's encoding. They hold the private selectors, and the public ones whose objc2-metal
  signature would change a value: a nil the header marks nonnull (the guest's nil goes to Metal
  as it is), a nil or NULL result objc2-metal types as nonnull, and an `error:` out-parameter
  Metal may leave nil (objc2's `Result` typing would invent an error there, and core tells the
  two apart). A re-declared public selector keeps objc2-metal's name, so a call names its trait
  (`RenderEncoderSpi::setRenderPipelineState(encoder, state)`).
- `spi/classes.rs`: the private classes (`MTLBufferDescriptor`, `MTLResourceAddressRangeArray`,
  the `_MTLIndirect*Arguments`) with `extern_methods!`; `spi/functions.rs`: Metal's C functions.

`metal/src/msg.rs` runs every call core asks for, named by its selector (a Metal C function by
its name with `()`):
- `msg::send(sel, policy, || call)`, whose `Policy` labels the call where the platform
  method returns `RaiseResult`: `Caught` where no path reaches it without a handler, `Fatal`
  where its raise dies whatever handler is live, `On(on)` in a method reached both ways, whose
  caller passes its `OnFault`; `msg::get(sel, || call)` for a call whose raise always dies (a
  getter, a property setter, a C function, and every send of a platform method no path reaches
  under a handler, which returns its value with no `RaiseResult`); `msg::catch(sel, || call)`
  for the deserialization calls, whose raise may be their resolver's own, decided by
  `Caught::decide` with the same labels;
- the call runs inside objc2's `exception::catch`, so a raise inside Metal stops at the call
  before any Rust frame unwinds, and the label alone decides it: `Caught` (and `On(Return)`)
  answers `Err(Raise::platform(sel))`, the caller's to catch; `Fatal` (and `On(Die)`) dies at
  the call. The raise's origin (`Origin::Platform`) holds the selector, which a death names;
- the closure borrows its receiver and arguments and owns nothing with a `Drop` (D34); a raw
  guest value goes as objc2-metal's newtype over the same word (`MTLPixelFormat(v)`),
  unvalidated, and a C array as a pointer to its first element, nil entries as NULL;
- a `new…` result is adopted at `+1`, and objc2 retains a `+0` object result: either way the
  handle holds one reference of its own (`Obj::from_retained`) and puts none in the thread's
  pool. An optimized build elides Metal's own autorelease of a `+0` result where the runtime's
  return handshake succeeds, as it does for `-commandBuffer`, whose uncommitted buffer then goes
  at its handle's drop; a dev build cannot, and that reference stays in the thread's pool until
  the pool pops (7.4);
- a pointer result stays a raw pointer where NULL is an answer (`contents`, the sample
  storages), and an `iosurface` getter's surface takes exactly one retain, the handle's, because
  the surface's retain count is guest-visible (D6);
- a thread running the work of an isolated task that has ended sends nothing: it stops at the
  send (`stop_if_task_ended`, 5.5).

`tests/message_layer.rs` checks that `objc_msgSend` stands only in `msg/` and `msg_send!` only
in `spi/`; `tests/spi_encodings.rs` checks each method `spi/` declares against the runtime.
Foundation and IOSurface sends go through `fatal_send`, which catches a raise in one and dies at
the send, and a send that only makes or reads a value through `value`.

### 4.7 The library's shaders

`metal/src/gpu/shaders.metal` holds the library's own shaders, in Metal Shading Language: the
display's present functions (`displayPresentVertex`, `displayPresentFragment`,
`displayPresentFragmentWithGammaTableAndColorMatrix`) and the XR10 two-plane paging kernels
(`BlitInRGB_2P_XR10_A8`, `BlitOutRGB_2P_XR10_A8`). Core names them and binds their arguments
(`display/render.rs`, `resources/texture/compute.rs`); what each computes is written beside it
in the source, and `gpu/shaders.rs`'s tests check it on known pixels.

`gpu/shaders.rs` compiles the source with `newLibraryWithSource:options:error:` once per Metal
device, the first time core asks for it (a device's blit pipelines, a display's pipeline cache,
an isolated task's blit pipelines), and every later ask answers the same library. Devices are
told apart by registry ID. A compiled library stays for the life of the process; one that fails
to compile is not kept, and the next ask compiles again.

The Vulkan platform (2.2) runs the same source as a metallib: its `build.rs` compiles
`shaders.metal` with Apple's tools on a Mac (`metal -std=metal4.0`, fast math, then `metallib`:
what Metal's run-time compile uses), and a build host without them takes metallibs a
Mac compiled (`build.rs`'s doc says how). Its `gpu/shaders.rs` loads the metallib with the
runtime's `new_library_with_data` once per device, by registry ID, and keeps it as the metal
crate keeps its library.

---

## 5. Isolated tasks

### 5.1 One service per task

Under isolation, each guest task other than task 0 runs in a fresh service of its own, a
`GpuTask`, on threads of its own in the host's process (`tasks::isolation::TaskThreads`, D10):
the task's connection queue (the handshake and the host's messages, in order), its service
queue (its jobs) and Metal's completion threads. An exec's job runs on the FIFO thread that sent
it instead, in the same order (5.4). There is no pooling, no reuse and no second task in one
service; a define after a death makes a new service. The service's objects come from the host's
Metal device, and it reads guest RAM through the host's own mapping of it. The XR10 blit pipelines
and the resource-index rebuild are the task's own, made or done once per task.

### 5.2 Which task runs where

| setup | task 0 | tasks other than 0 |
|---|---|---|
| blocks | `LocalTask` (the host's task and memory callbacks) | `LocalTask` (the host's callbacks) |
| map | `LocalTask` (memory-map backend) | `LocalTask` (memory map) |
| isolate | `LocalTask` (memory map) | `RemoteTask` → link → `GpuTask` on the task's threads |

A `LocalTask` death kills the host. A death in a `GpuTask`'s own code ends its task, and the host
sees status 4 for that command and every later one of the task; a crash inside Metal ends the
process.

### 5.3 The transport traits

```text
  RemoteTask --ToService--> dyn ServiceLink --> (the task's connection queue) dyn ServiceSide
  Endpoint <--ToHost-- (on the link's queue) <-- dyn HostLink <-- GpuTask
       ^ dyn HostSide: on_message / on_interrupted / on_invalidated
```

- `TaskLauncher::launch` makes a new service per call. `ServiceLink::send` is ordered and
  asynchronous; replies come back on the link's one serial queue, which also delivers the
  service's messages, so a reply and a later message never reorder. `ServiceLink::send_here`
  is `send` delivered on the caller's thread, in the same order (5.4): the remote exec's, whose
  FIFO thread waits for the reply anyway.
- `tasks/wire.rs`: `ToService` (17 variants) and `ToHost` (4); a reply is a `Reply` of its
  values, callable more than once.
- `tasks/remote/completions.rs` is the one piece of status-4 code every launcher shares: a crash
  signals every outstanding token 4 in ascending id order, and every token issued after it is born
  4. The first token born 4 writes one warning, `task <n>: its service has ended: this command and
  every later one fault (4)`, the only line a dead task's later faults leave. Its table drops the
  tokens it still holds in ascending id order.
- `RemoteTask`'s drop waits, with no timeout, until every token is completed or crashed, then
  invalidates the link.

### 5.4 The link

`tasks/isolation/link.rs`:
- **The connection queue** delivers the host's messages to the service one at a time, in send
  order, each as the task's code (`TaskLife::run`). The handshake runs there, and so does the
  task's end: the service's state is released, then the host is told. A message sent with
  `send_here` (an exec) is delivered on the sender's thread, which takes the connection queue's
  turn and then the service queue's (`SerialQueue::sync`), so it runs after every message and job
  before it and holds both queues while the service runs its job there. What the delivery holds
  with no holder goes when it returns, as the queues' workers release theirs once they have left
  them (`rt::held::LastResortScope`). A death there unwinds to
  the delivery's `TaskLife::run` and ends only the task, as on the queues' workers. The reply,
  made there, runs there too when the link's queue is idle with nothing queued: its turn on that
  queue, taken under the state lock, is its place in the queue's order (`hold_if_idle`), and the
  link queue's worker runs it otherwise. Of the exec's four waits for a woken thread (the
  connection queue's worker, the service queue's, the link queue's and the FIFO thread's), the
  delivery saves the first two when both queues are idle, and the reply the last two when the
  link's queue is.
- **The link's queue** makes every `HostSide` call and runs every reply, one at a time, in the
  order the service sent them. A message or reply the service hands over while the task runs is
  always delivered, before the end the host is told of.
- **Pending replies**: the service gets a relay in each reply's place, so the host's `Reply` is
  released exactly when `ServiceLink::send` says: after its one run, right after the interruption
  or invalidation, or at once for a send the link drops.
- **Lent surfaces**: an IOSurface the host lends the service reaches it as another object over the
  same surface, looked up from its Mach port, as it reaches a process it is sent to.
- **The memory map** must be whole pages at non-NULL, page-aligned host addresses
  (`tasks/remote.rs`): any other map makes no task, nothing is launched, and the define gets no
  task, as when the handshake fails.
- **Round trips**: `RetainIoSurface` and `NewSharedTextureHandle` block the service queue until
  the host answers or the task ends (`TaskLife::waiting`).
- **Invalidation**: the remote task's drop invalidates the link after its drain. The invalidation
  ends the task and releases the service's state before it returns, so the task's Metal objects,
  their resource IDs and GPU addresses are free before the host's next allocation. The wait takes
  no lock the host may hold.

### 5.5 A task's life

`tasks/isolation/life.rs`. Every thread that runs the task's code does so inside `TaskLife::run`,
which sets the task's trap on the thread: the connection queue's messages, each job on the service
queue, and each Metal completion handler the task adds. A death there (`raise::death::fatal`, an
uncaught raise, a NULL guest access) writes its line, marks the task dead and unwinds to that
`run`.

The task's other threads stop at once, each at its next step with an effect
(`stop_if_task_ended`): a message to the host, a Metal send, a write to guest memory, a job or a
completion handler that has not begun. A thread waiting on the host for a reply is woken and
stops. The host's invalidation ends the task the same way, without a death.

A task that died releases no Metal object: while a thread unwinds the task's code, and while the
end releases the task's state, a platform handle dropped is leaked until the process exits
(`leaks_objects`). Its Metal objects stay as the death left them (an encoder never ended, a
command buffer half encoded), because releasing one in that state can end the process.

A crash inside Metal is not the task's: a Metal exception no handler takes, a platform call that
ends the process, and a real crash in Metal, IOGPU or the driver end the process, as they do
outside isolation.

### 5.6 The test launcher

`testkit::launcher::ThreadLauncher` is `TaskThreads` over the fake platform, recording each task
it launched and letting a test script the service. A launched task shares the launching platform,
as an isolated task shares its process: the same fake device, world and address space. A death in
the task's own code ends the task, never the test; a scripted crash inside Metal (`Step::crash`)
ends the test's process, as it ends the real one.

### 5.7 Death classes, end to end

| event | mechanism | the host sees |
|---|---|---|
| a death in the service's own code (an assert, a NULL dereference, a raise no job catches) | `fatal` unwinds to the task's `run`; the task ends and its other threads stop | interruption → status 4 now and for every later token |
| a crash inside Metal, IOGPU or the driver | the real crash | the process ends |
| a GPU or command-buffer error | the completion's status and error | status 3 for that command |
| a memory map that is not whole pages at non-NULL, page-aligned host addresses | no task is launched | the define gets no task |
| the remote task's drop | the invalidation ends the task and releases its state | nothing more |

---

## 6. Outcomes: expected, raise, death

### 6.1 Three channels

Never converted into one another, with no `From` between them (D8):
- **Expected failure**: `Option` or `Result` with an error enum per operation, where the API
  answers a refusal, an error or a status.
- **Raise**: `RaiseResult<T> = Result<T, Raise>`, an exception as a value, which a catch turns
  into an exec fault, a status or a reply.
- **Death**: `fatal(id)`, one stderr line and `abort()`.

### 6.2 Raises and the hold

`raise.rs`, `raise/on_fault.rs` and `rt/held.rs`:
- `Raise { id: RaiseId, detail, origin }`. `RaiseId` has one variant per area, each wrapping the
  area's `RaiseKind`, whose `Display` is a dotted id (`device.wait.circular`); `Platform` is a
  raise inside Metal, whose origin names the op. `Raise::new` is pure.
- **The hold** (`rt::Hold`) names the holder that what a unit's work holds deep (`hold.keep`),
  and what a raise it ends carries (`hold.land`), go to (7.4). A function takes it as a parameter
  only where it needs it: where it holds deep, ends a raise or runs an R frame's cleanup
  (`hold.cleanup_step`), or where it passes the hold to a function that does. A trait method
  takes it when one of its implementations needs it; the others name it `_hold`. It is `Copy`,
  and `!Send`, so it never reaches a queue's block, a thread or a completion handler. It does
  not decide raises.
- **A raise is decided by its site's label.** A site no path reaches without a handler is
  `Err(r)` (`raise!(id)`, `raise!(id, detail)`, or `Err(r)` written out); one no path reaches
  under a handler calls `fatal` there, before any `Drop` runs, so lock guards stay held and
  nothing more is sent or written to the guest (`Raise::new(id).fatal()`); a function its
  callers reach both ways takes an `OnFault` from its caller, which decides (`raise!(on => id)`,
  `on.throw(r)`).
- **Where a hold starts.** A unit of work starts one with the unit's holder (`Hold::of`): each
  FIFO command, each service job, each API entry. A queue's block, a source or timer handler, a
  once's block, a thread's body, a Metal completion handler and a drop start with `Hold::NONE`,
  no holder: their code is `Send` and the caller's hold is not. A raise there with no handler of
  its own dies at the raise, whatever handler the caller had (an exception never unwinds out of
  a callout, `rt/block_env.rs`), so every site in them is labelled as having none. A function
  that needs a hold passes on its caller's, except a landing frame, which gives its body a hold
  of its own holder (`Hold::of(held)`) and needs none from its caller.
- **A handler** is a frame that takes a raise's `Err`: it ends it (`hold.land(r)`, landing what
  the raise carries in the hold's holder), turns it into a fault, a status or a reply, or passes
  it on with `?`. A body whose raise a handler takes runs in a function or closure of its own
  (often an immediately invoked closure), so its locals drop at the same point whether it
  returns or raises. A frame with no cleanup after its body (a compute list's, the range
  tracker's add and remove) is reached only under a handler, so its body's raise goes on as
  `Err`.
- **The frames that pass on a raise or die**: an R frame (a paging or decoding catch that ends
  its encoders and commits first) passes one on after its cleanup, as `Err`, since every R frame
  is reached only under a handler. A frame reached only with no handler (the exec-indirect task
  calls, the GPU task's jobs and completions) never sees a raise, since every raise below it has
  died at its own site already; were one to reach it, it dies there (`.fatal()`, `no_catch`,
  `unraised`). A platform method no path reaches under a handler (a queue's or a shared event's
  creation, the display's pipelines) answers its value with no `RaiseResult` (4.6), so its caller
  has nothing to end. `Raise::fatal` dies even under a handler (the uncaught sites). A raise
  from an R frame's cleanup goes on in place of the caught one, which `hold.cleanup_step` ends
  first.
- **The cost of a raise** is nothing the API shows: there is no real `NSException` per raise
  (D7).

### 6.3 Raises inside Metal

Caught at the call by the message layer (4.6) and decided by the op's policy. One place keeps a
real Objective-C raise through foreign frames, from a frame that owns nothing (D34): the pipeline
resolver's callback through Metal. What the resolver's lookups hold deep goes to the holder of
the hold the pipeline creation is passed (4.2), and the resolver answers `RaiseResult`: its raise
comes back as `Err` (the handler Metal's call forces), goes back through Metal as an
`NSException`, and once Metal has unwound, the caller raises it again. Every caller is a pipeline
creation, reached only under a handler, so it is `Err`. A death or a stop inside the resolver of
an isolated task's code is a Rust unwind (5.5), which never crosses Metal's frames: the context
catches it in the callback, raises through Metal in its place, and the deserialization goes on
with the unwind once Metal has returned.

### 6.4 Deaths

A death is the one fatal path, `raise::death::fatal(id)`: one line on stderr,
`reims-vgpu: fatal: <what died>`, written in one write, then `abort()`. No `Drop` runs and
nothing unwinds, so the process ends at the guest op that caused it and nothing after it reaches
the host or the guest. `fatal!` and `fatal_unless!` are its macro forms; an uncaught raise
(`Raise::fatal`) adds its origin and detail to the line. Where the process dies is the contract;
the message and the signal are diagnostics.

`DeathId` (`raise/death.rs`) names what died: one variant per area, wrapping the area's
`DeathKind`, whose `Display` is a dotted id (`gpu_task.root.null`); `Raised` (a raise nothing may
catch); `Platform` (a platform call that ends the process); and `HostRaised` (6.7).

In tests (`cfg(any(test, feature = "testkit"))`) with capture on, `fatal` records the death's
`DeathSite` in the test's `DeathSlot` and panics with a `Death` payload (`testkit::capture`); the
subprocess harness (`testkit::death`) turns capture off to see what a real death leaves behind,
and reads the site a testkit build appends to the line. `DeathSite` and its `DeathKind` class are
that test identity only: production never computes them.

### 6.5 Panics

A panic is a bug, never control flow, and it ends the process as a death does: one line
(`reims-vgpu: panic: …`), then `abort()`, before anything unwinds, so no drop runs first and no
panic of the library's reaches the host's frames. Inside a refusal scope (the graphics lane's
ops, `raise::refusals`) it is refused as a death there is: the line is written, the panic unwinds
to the scope, and the op is refused, logged as a bug; under the test capture it is the test's
failure, and passes every scope. The API's singleton
(`reims-vgpu/src/singleton.rs`) installs a chained panic hook that does this when the panicking
thread is running the library's code (the FIFO thread, a worker running one of core's blocks, a
Metal completion handler or buffer deallocator, a public entry point, or a C export, each marked
by `rt::enter_ours`), and leaves any other panic, the host's own, to the host's hook (D9). Every C
export is `extern "C"` and runs its body in `ffi::export` (`reims-vgpu-c/src/ffi.rs`), which
installs the hook and marks the thread first: the `extern "C"` frame would end the process too,
but only after the frames below it had run their drops. `panic = "unwind"` stays because objc2's
`exception::catch` needs it.

### 6.6 Values kept for good

`raise/kept.rs`: `keep_forever(x, Kept::…)` is `mem::forget` with the rule that names what the
guest or host sees because the value's release never comes: a guest page's unmap, an IOSurface's
use count (a later unmap is Busy), Metal's numbering (a live texture shifts later resource
indices, a live buffer or heap later GPU VAs, both of which placements write back to the guest),
the host's releases, a timer that keeps firing, the address space, a reader of a replaced
mapping. A value is kept only where one of these shows: a release nothing sees, such as what an
initializer made before a raise where no path reaches it, is a plain drop. Each call has a doc
comment and a test.

### 6.7 Host calls

Every call into the host's own code runs inside the OS's catch (`ffi::host_call`,
`reims-vgpu-c/src/ffi.rs`, over `ffi/catch/{macos,linux}.rs`): objc2's `exception::catch` on
macOS, std's `catch_unwind` on Linux. It surrounds each function of `RvgHostCallbacks` (the task
and memory calls, `raise_interrupt`, the trace-range calls), of `RvgDisplayCallbacks`, of
`RvgWindowCallbacks` and of `RvgIOSurfaceCallbacks`, a table's `destroy`, and the log handler.
Anything the host throws ends the process at the call, so no foreign exception crosses the
library's frames: on macOS an Objective-C throw is `DeathId::HostRaised`; on Linux a Rust unwind
that reaches the catch is `HostRaised`, and a foreign one (a C++ throw) cannot be caught, so std
ends the process where it reaches the catch's frame. The callback types stay `extern "C-unwind"`,
which both catches need.

---

## 7. Concurrency

### 7.1 Primitives

Each row names a primitive the library uses and what of its behaviour is kept because the API
shows it.

| primitive | what it is | kept |
|---|---|---|
| serial queues (`reims-vgpu.fifo-control`, `reims-vgpu.reporting`, `reims-vgpu.efi-present`, `reims-vgpu.iosurface`, `reims-vgpu.display`, `reims-vgpu.release`, and an isolated task's `reims-vgpu.task-connection`, `reims-vgpu.task-service` and `reims-vgpu.task-link`) | `rt::SerialQueue`: a `VecDeque` of blocks and sync turns under a `Mutex`, run by a worker thread of the queue's own (`Threads::spawn_worker`), which ends after 5 s idle or once the last handle is gone | blocks run one at a time in submission order. `sync` runs the block directly on the caller's thread once the blocks ahead have run, and the caller goes on the moment it returns (`docs/lessons.md`); no exception leaves it, since every call it makes catches its own (D35). A `sync` onto a queue the thread already owns runs its block at once, as a nested call, since the blocks behind the one the thread runs cannot run first. The worker releases what its blocks deferred once it has left the queue (it ran dry, or a sync caller holds it); a `sync` from there runs the blocks ahead itself. The worker is woken only when it has work, and after the lock is released, so a submit or a `sync` never waits on a woken worker that has no CPU yet |
| add and or signals | `rt::CoalescingSignal`: a display's new-frame signal, the EFI display's present source and timer, the stamp-wait report, the IOSurface device's doorbell | merges before the handler runs give one call; a release without a cancel drops pending data; the cancel callout frees the handler on the target |
| the EFI timer | `rt::Timer`: a thread of its own on the platform's clock merging fires into a `CoalescingSignal` on the queue | starts at the wall time of its start, every 100 ms; missed fires coalesce; dropped uncancelled it keeps firing |
| once | `rt::ProcessOnce` (`OnceLock`), held by its owner: an isolated task's XR10 blit pipelines, once per task | concurrent callers wait for the first; the block runs under `Hold::NONE`, whatever the caller's hold (6.2) |
| the FIFO thread | a named thread (576 KiB: a queue worker's 512 KiB, which an isolated task's exec job it runs gets, and room for its own frames) the platform starts (`rt::thread::spawn_named`) | its body, then what was held on it under a hold with no holder (its last-resort holder), then the FIFO (its target) dropped, on the thread |
| mutexes and condition variables | `parking_lot::{Mutex, Condvar, ReentrantMutex}` | each mutex guards its own fields and nothing more; every timed wait goes through `Clock::wait_until` |
| reader-writer locks (a task's, the range tracker's) | `sync::WriterPrefRwLock` (`parking_lot::RwLock`) | writer preference: a thread's first read waits behind a queued writer, and `try_write` fails while a writer is queued; a read on a thread that already holds one goes through at once, so a nested read never hangs |
| fields read both with and without a lock | `sync::StagedMutex` | unlocked readers see a guard's stores at its drop or explicit `publish()`, never store by store |
| weak references | `std::sync::Weak` | fails to upgrade once teardown starts (15.2) |
| Metal completion handlers | `Box<dyn FnOnce + Send>` run on Metal's thread | a foreign thread: under `Hold::NONE`, owning no queue, holding none of the committer's locks |
| the display thread | `reims_vgpu::platform`'s one `SerialQueue` per process, behind core's `HostQueue` (D49) | every display's jobs run on it one at a time, in the order core queued them; the library never waits for it (api.md section 5) |

Which thread runs a block, and the threads' names, are not part of the contract: what crosses
the API, and its order, is, and the header names the thread each callback arrives on (api.md
section 5).

### 7.2 Lock-order register

```
device.root_node      -> device.root_task
device.root_fifo      -> device.root_task
device.root_fifo      -> fifo.lock
display               -> device.root_node | device.root_task
device.tasks          -> task teardown (TaskCore.parts)
device.signal_mutex   -> the host's raise_interrupt   (the host's function runs under the lock)
fifo_queue            -> device.signal_mutex
fifo_queue            -> reporting_queue (sync) -> device.root_fifo
fifo.lock             -> stamp queue state -> device.signal_mutex   (sinks run under the queue lock)
task.mutex (WriterPrefRwLock)  -> manager.creation (reentrant) -> object_lists.mutex
task.object_database_mutex     -> manager.creation (reentrant) -> object_lists.mutex
object_lists.state    (leaf: never held across a call into the task or a mapping's drop)
endpoint.lock         -> completions.state -> token.status -> stamp queue state -> device.signal_mutex
                      (a crash signals each token 4 under both locks; the token's handler runs the
                       FIFO's completion)
completions.state     -> token.status -> stamp queue state -> device.signal_mutex
                      (a token born 4 after a crash runs its handler under the table lock, on the
                       issuing thread)
endpoint.lock         -> iosurface queue (sync)
                      (a surface retain holds the endpoint's lock across the IOSurface device's
                       lookup, which syncs onto its queue)
device.displays       -> guest_display.guest_display_mutex (an attach or detach sets its client)
device.displays       -> efi.present_queue (sync)
                      (an EFI register write's mode rebuild stops presentation there)
guest_display.guest_display_mutex -> display.display_mutex (a present)
display.mode_list_update -> display.display_mutex -> display.guest_display
display.mode_list_update -> guest_display.guest_display_mutex (set_mode_list, after the display's
                                                               mutex is let go)
```

**A present and `set_mode_list`**: a present takes the guest display's mutex then the display's.
`DisplayCore::set_mode_list` stores the list and reads the guest display pointer under the
display's mutex, lets it go, and only then takes the guest display's mutex, so the two never wait
on each other. Its own mutex, `mode_list_update`, held across the whole call, keeps lists set at
once reaching the guest display in the order they were stored.

Rules beyond the chains:
- **No `ReentrantMutex` is held across `SerialQueue::sync`**: the sync block runs on the caller's
  thread, so a block that takes a recursive lock its caller holds would re-enter it. No path does.
- **The link's queue is one serial queue for replies and host messages.** If the last reference
  to a `RemoteTask` drops inside a reply while a token is outstanding, its drain waits for a
  completion queued behind it on the same queue: a hang the host sees (`tasks/remote.rs`,
  `the_last_reference_in_a_reply_with_a_token_outstanding_hangs`).
- **An EFI display that ends a display hangs, as Apple's does.** The display's end detaches it
  under `device.displays`. A present that holds the last reference after the host's release ends
  it on the present queue, while an EFI register write holds `device.displays` and syncs onto
  that queue; a mode-count or mode-value read ends it under the `device.displays` its own thread
  holds (`display/efi.rs`, testing.md 7.3,
  `device::displays::tests::efi_display_holding_the_last_reference`).
- **A FIFO delete's wait for the tasks' completions** counts them down under a lock of its own
  (`Countdown` in `device/fifos.rs`): the last completion notifies under that lock, and the wait
  checks the count and parks under it, so no completion is lost between the two.
- Races whose outcome the API shows are kept: a contended task delete's `try_lock` with the
  tearing-down fallback, the nil handler call when a display's client detaches before a queued
  job runs, the unlocked current-frame signal.

### 7.3 Threads

- One FIFO thread (`reims-vgpu.fifo`) for the root FIFO and one per running child.
- A worker per serial queue while it has work (it ends after 5 s idle): the device's FIFO-control
  and reporting queues, the EFI display's present queue, the IOSurface device's queue, the
  process's display thread, a release queue for each teardown moved off a queue it syncs onto
  (`held::release_off_thread`), and each isolated task's connection, service and link queues.
- A thread per running EFI timer (`reims-vgpu.timer`), which only waits and merges.
- Metal's completion threads, and the host's own threads inside a call that may call back (both
  foreign).
- The process's main thread, the host's, where every window's AppKit side runs: the application,
  the `NSWindow`s, their views and layers, input, the cursor and the host's window callbacks
  (D51). The library reaches it only by posting blocks to the main run loop
  (`reims-vgpu-window/src/native/macos/main_thread.rs`: `CFRunLoopPerformBlock` in the common
  modes, then `CFRunLoopWakeUp`), which run in the order posted, under the host's
  `CFRunLoopRun`, under `[NSApp run]`, and during a live resize or with a menu open. The first
  window's attach posts the application's start, alone, since the run loop runs its pending
  blocks as one batch and a start that runs the application never returns; what is posted before
  the application has launched is held in the library, and runs in order once it has. Where the
  host has made no `NSApplication`, the start runs the library's, for good: the host's run loop
  runs inside it from then on. The library never waits for the main thread, and never posts to
  the main dispatch queue, which `[NSApp run]` started from one of its blocks starves for good
  (`main_thread.rs`'s doc has the measurement).
- A render thread per window (`reims-vgpu.window`, 512 KiB, started by the platform as the FIFO
  thread is), from the window's attach. It waits on the window's surface (a `Mutex` and a
  `Condvar`) for a new frame, a new layout or the window showing again, and draws only while the
  window is at least partly visible: its layer's `drawableSize` when the layout changed, in a
  `CATransaction` of its own with actions off, then the next drawable, inside an autorelease pool
  of its own, into which the window's sink encodes the display's current frame, presents and
  commits. `nextDrawable` blocks for up to a refresh when frames come faster than the screen
  shows them, and is the only pacing. On the Vulkan platform the sink presents itself, as on
  Linux: the thread has the sink's presenter take the next image of a swapchain over the layer
  (`VK_EXT_metal_surface`), whose images are the layer's drawables, fetch its drawable on this
  thread (MoltenVK's export of its texture, `VK_EXT_metal_objects`), draw the frame into it and
  present it on the swapchain's presenting queue (10); the image's fence and the drawable's
  fetch are the pacing there. The thread ends at the window's end, and nothing joins it: a
  release that races a draw ends the display on this thread, its presenter goes before the
  layer, and its reference to the layer is posted to the main thread to be released there, as
  the view's is.
- On Linux, in place of the main thread, the library's UI thread (`reims-vgpu.ui`, 1 MiB),
  started with the first window or the first look at the host's screens, where SDL's video is
  initialized and every SDL call but a render thread's Vulkan surface is made: each window's
  opening, changes and end, its input and the host's window callbacks, one at a time. Work is
  posted to it (`reims-vgpu-window/src/native/linux/ui.rs`) and runs in order between its events,
  an event of its own waking it from its wait in SDL. The process's main thread is never used or
  waited for. On Linux the render thread takes the swapchain's next image with a fence it waits
  for up to 33 ms, which is its pacing; the frame's command buffer waits on nothing of the
  compositor's, so the guest's frames never wait for the window. It presents on a queue made for
  presenting, whose `VkQueue` the runtime keeps apart from the guest's queues where the device
  has one to spare, so a present that blocks holds none of the guest's work. At its end the
  render thread lets its swapchain and Vulkan surface go, then posts the SDL window's release to
  the UI thread.
  On macOS SDL3's window runs on the main thread as AppKit's does, through the same posted
  blocks, and never pumps AppKit's events: the application dispatches them.

No thread of the library's that runs Rust code sending Objective-C messages has a Rust unwind
handler at its base (D35): the platform starts them as pthreads (`metal/src/os/threads.rs`). The
timer's thread is a `std::thread`: it runs no Objective-C.

### 7.4 Autorelease pools and release points

Values drop where Rust's scoping puts them. There is no pool emulation for core's own values: no
value is parked only because an Objective-C convention would have autoreleased it. Two things
remain, each for what crosses the API.

- **Real pools** (`rt/pool.rs`). A `Pool` pushes the platform's real pool around a unit of work
  and pops it at the unit's end, so that what Metal autoreleases inside it is freed there: a FIFO
  command, a service job, an API entry (`reims_vgpu::entry::run`), and the units inside them
  that make Metal objects (a display's create and presents, a frame's encode, the suspend walk,
  the tasks' teardown, a paging walk, the EFI present). Without one, an uncommitted command
  buffer lives on in a dev build, where Metal's autorelease of it is not elided (4.6), and the
  queue's 65th `-commandBuffer` blocks.
  - `pool::around` leaves its pool pushed on a raise, as clang's `@autoreleasepool` does: the
    enclosing pop frees it, after the catcher's report.
  - Core's threads (the FIFO thread, each queue's worker) keep a last-resort pool for what is
    autoreleased on them outside any unit: a worker pops it once it has left its queue, and the
    FIFO thread at its end, before it drops its target.
- **Holders** (`rt/held.rs`). A value whose last release crosses the API (a task's `destroy_task`
  and root-page unmap, a FIFO's ring unmap, a guest display or display teardown, a mapping's
  unmap, an IOSurface's retain count) is held to its point in the API's order:
  - by a `Held`: each FIFO command's (`Command::held`), each service job's, each API entry's,
    and the landing frames (`Held::raising`) that end a step inside one where a deep hold must
    go at the step's end (a lookup's read, a paging walk, a reply, a present). Code with no
    handle to its holder (the kernel task a user task reaches through its device, a page-table
    read's load of the device) holds through its hold (`hold.keep`), which names the holder of
    its unit or of the landing frame it runs in (6.2). A holder releases what it
    holds most recent first, before its unit's pool pops;
  - by the raise that leaves the work: `Raise::defer` carries a value outward (a mapping a
    writability check refuses, what a discard deleted before its raise, a present's surface and
    texture), and a landing frame whose body raises puts what it holds in the raise. The catch
    that ends the raise (`hold.land`) lands them in its hold's holder, each at the point it was
    first held, so they go after the catcher's report (an S-class command's stamp;
    `did_resume`'s interrupt, for a failed restore's surface and texture). A raise nothing
    catches is a death, which releases nothing, and one dropped with values still in it is a
    bug, which a build with debug assertions reports;
  - by a value handed back for the caller to let go after its reply, completion or commit
    (`tasks::shared::Deleted`, `deserializer::walk::Decoders`), or by a guard that owns what a
    pointer points into (`resources::host::MappedAddress`).
- Under a hold with no holder (a queue's block, a thread's body, a completion handler, a
  drop), a value held deep, or what a raise ended there carries, goes to the thread's
  last-resort holder on a core thread, released just before its last-resort pool pops, and on
  any other thread drops at once.
- A reference loaded from a weak one in a queue's block, whose last release would run a teardown
  that syncs onto that queue (the reporting handler's device, the EFI source's display), goes
  through `held::release_off_thread`: when it is the last, the drop runs after the block, on a
  queue of its own.
- A resource manager's lookup and the guest display a doorbell loads are the caller's references,
  and go at the end of their use.
- A value whose release the guest or host would see, and which is never released, is kept under a
  named rule (6.6).

### 7.5 Clocks

`Deadline::{Mono, Wall}` (`platform/os/clock.rs`), each wait on a fixed clock: the device's 2 s
stamp-wait report is `Mono`, the display's 2 s in-flight wait and the EFI timer's schedule
`Wall`. `Clock::wait_until` waits in awake time and reads the answer on the deadline's own clock.
Nothing calls `Instant::now` or `SystemTime::now` (`clippy.toml` disallows them).

---

## 8. Containers

`collections/`, over std's containers, each named by the order it gives. Where a container's
order reaches the guest or the host, it is named below; the others give no order the guest or
host sees.

| container | used for | order |
|---|---|---|
| `BucketOrderMap<V>` | the device's task table | ascending task ID (a `BTreeMap`): every fan-out across tasks and the releases when the table empties |
| `ListOrderMap<V>` | the resource caches | none the guest or host sees: a `HashMap` with a fixed hasher, walked by a flush and drained by a teardown |
| `FirstFitVa` | the root task's VA | first fit over a sorted free list, never reset (D15) |
| `LookupMap<K, V>` | maps that are only looked up: the pipeline caches, an isolated task's surfaces and FIFOs, the endpoint's kept surfaces | none: no iteration API, and a fixed hasher so its drop order is the same in every run |

Outside `collections/`, the remote task's outstanding completions are a `BTreeMap`
(`tasks/remote/completions.rs`): a crash signals them, and the table drops them, in ascending
id. The memory map's ranges are sorted with std's stable sort, by start and then length
(`memory/map.rs`).

Drain APIs return values, so callers choose their drop points inside or outside their locks.

---

## 9. How the areas reach one another

Most areas call each other's concrete types (`seams.rs`):
- the FIFO, a task and a mapping call their device through `DeviceCore`'s methods, holding it
  unretained through the aliases of `seams/links.rs` (`FifoDeviceLink`, `TaskDeviceLink`,
  `MappingDeviceLink`, the last a `Weak<MappingSink>` that outlives the core's other fields);
- the device and the FIFO call a task through `tasks::Task`, an enum of `LocalTask` and
  `RemoteTask`;
- the deserializer and the resource manager share the manager's lookup (`ObjectLookupRef`).

One `dyn` seam remains: `resources::host::ResourceHost` (with `CallBorrow`, the loan of a task
part for one call), what resources, the manager and the object lists ask of their task. It has
two implementors, `LocalTask` and `GpuTask`, and is held as a `Weak` whose failed upgrade is a
torn-down task.

Conventions every cross-area call follows:
- handles cross by value, retained at the callee's return, never as borrows of the callee's
  fields; IOSurfaces cross only where one is handed over (their count is guest-visible);
- a method that can raise returns `RaiseResult<T>`;
- records the guest wrote cross as `fifo::wire::Records`, a bounds-checked unaligned view, never
  as `&[T]` into guest memory.

---

## 10. The Rust API (`reims-vgpu`)

The Rust API is owned Rust types over core, which the C API wraps one to one; docs/api.md
section 3 has its types and signatures.
- **Objects.** `Device`, `Display` and `IoSurfaceDevice`, each made from one config
  (`DeviceConfig`, `DisplayConfig`, `IoSurfaceDeviceConfig`) whose callbacks are closures, and
  `FrameTarget` and `Frame`. A create answers a `Result`, whose `Error` names the check that
  refused it (core's `DeviceRefusal`, `DisplayRefusal`, `IoSurfaceRefusal`), in core's order.
  `Device`, `Display`, `FrameTarget` and `IoSurfaceDevice` are counted references with
  `clone_ref()` and no `Clone`: the last drop of a device, display or IOSurface device is its
  end, run synchronously on the releasing thread, where it may block or die. A display keeps its
  device alive, and a frame its display and its target.
- **Generic over the platform.** The types are generic over an `ApiPlatform` (core's `Platform`,
  plus the process's instance, the Metal device a config names and a frame target's pixels),
  `MetalPlatform` by default, so the unit tests run the API's forwarding on the fake platform. A
  window display's host screen and sink are chosen by the platform's type ("Windows", below): the
  fake gets a fixed Retina screen and no sink, so its window displays never open a window.
- **Behaviour is core's.** The closures are called synchronously on the thread core calls them
  on, with the arguments core passes, in core's order. A display's callbacks run on the process's
  display thread (`platform.rs`, D49): core's `HostQueue` over one `SerialQueue` made at the
  process's first display, on which core's `CoalescingSignal` serves `new_frame`.
- **Frames.** A `FrameTarget` is a shared-storage 2D texture of the library's on the device's
  Metal device, and a `Frame` one encode of a display's current frame into it: a command buffer
  of the process's frame queue for that Metal device (one command queue per Metal device, made
  when the first device is created on it and kept for the process's life), encoded, committed
  and waited on as separate calls. `read_pixels` copies a target into host memory.
- **The process's platform.** `singleton.rs` makes the one `MetalPlatform` on first use: it
  installs the panic hook (6.5), then the stderr logger if `REIMS_VGPU_LOG` asks for it (14).
- **The Vulkan platform** (feature `vulkan`, 2.2). `VulkanPlatform` is an `ApiPlatform` too:
  `singleton.rs` makes its one instance the same way, its devices are the runtime's (the default
  one, or the one whose registry ID a config names: Metal's for the default device, the runtime's
  own for any other), and `read_pixels` reads the texture back through the runtime. On Linux it
  is the default platform. A window display on it gets the host's screen as the window crate
  lists them, and a sink that draws into the window's swapchain ("Windows", below).
- **Entries.** Every public method, and every release of a handle, runs inside `entry::run`:
  the thread is marked as the library's (6.5), and a real pool and a holder are open (7.4), so
  what the call held or Metal autoreleased on the thread is released before it returns.
  `retained.rs`'s `Ours` runs a handle's release that way, so a last release runs the teardown
  the same way, under `Hold::NONE`.
- **Preferences.** `DisplayConfig::apply_preferences` and `DisplayMode::with_preferences` take a
  `Preferences` reader a Rust host may give; the library itself reads no system domain.
- **Windows** (`window.rs`, D51). `DisplayConfig::window` makes the display's `Window` (the window
  crate's), and the display's callbacks become the tee: all five present, each telling the window
  and then calling the host's where it gave one, so the guest reads a cursor glyph size limit of
  256 and both cursor handler bits. With `match_host_screen`, the display is made with the scale
  factor, the size in millimetres and the modes of its host screen (`window::host_screen`), from
  the window crate's policy. Once the display exists the window gets its sink
  (`window::window_sink`; the testkit's platform gets a fixed screen and no sink). On
  `MetalPlatform`, `MetalSink` encodes the display's current frame into a drawable in a command
  buffer of the frame queue, presents it and commits. On `VulkanPlatform` (feature `vulkan`),
  `window::vulkan`'s `SwapchainSink` makes a presenter over a swapchain of the window's surface
  (on macOS one of its `CAMetalLayer`), which encodes the frame into the swapchain's next image,
  commits, and presents the image on the swapchain's presenting queue once the runtime has
  scheduled the command buffer, on either OS. Neither waits for the GPU. Where the guest's
  cursor position, which the window reads through the sink at each cursor doorbell, changes with
  no pointer event of the window's in the second before (a mouse passed through to the guest),
  the window hands each draw the guest's cursor at that position, and the sink draws its glyph
  over the frame in the same command buffer (`window::cursor`: the display's quad buffers and an
  overlay pipeline of its shaders blending by alpha, both the display's, and a glyph texture
  that on the Vulkan platform takes no resource ID or range of Metal's space, made before the
  frame's encode), while the host's cursor is invisible over the image. `Display`
  holds a `WindowLink` every `clone_ref` shares; its explicit `Drop` lets go of that share
  before the reference itself, so the last reference closes the window before it releases the
  display (15.3). A refused create closes the window before it returns.
- There is no host-object layer: no Metal device, dispatch queue or Objective-C type crosses the
  API as a config, handle or callback a host passes, nor in `ApiPlatform`: a window's screen and
  sink are chosen inside the API by the platform's type. Core's objects behind the handles are
  reached only with the `testkit` feature (D27).

---

## 11. The C API (`reims-vgpu-c`), the dist tree and the option ROM

### 11.1 Conventions

docs/api.md is the C API's specification; the header repeats each rule where it applies.
- Prefix `rvg_`. Every function is `extern "C"`, and its body runs in `ffi::export` (6.5);
  `tests/exports.rs` checks both, and that the dylib exports every function of the header.
- Handles are opaque and counted (`handle.rs`): each pointer a create or a `_retain` returns is
  one `Arc`. A release of NULL does nothing; any other NULL handle is `RVG_ERR_INVALID_ARGUMENT`
  where the function answers a status, and a death where it does not.
- Every struct the host writes starts with `uint32_t struct_size` (`sized.rs`): fields past it
  read as zero, and a larger size is accepted only when its extra bytes are zero.
- Each object's callbacks are one table with one `ctx` and one `destroy` (`table.rs`).
  `destroy` runs once, when the last closure built from the table goes, which for a table with
  functions is at the object's end; a table with none has its `destroy` at the host's release,
  and a refused create runs it before it returns. A display with a window has a second table, the
  window's (`window.rs`), which the display's handle holds beside its own and lets go of first:
  its `destroy` runs when the release closes the window, before the display's.
- A call that can fail answers an `RvgStatus` (`status.rs`), a plain `u32`; the codes are
  additive, and `rvg_status_message` describes each.
- Every call of a host function runs in `ffi::host_call` (6.7). The log hook
  (`rvg_log_set_handler`, `log.rs`) sends the library's lines to the host's function (14).
- The exports are written over `Plat`: `MetalPlatform` in the product, the testkit's
  `FakePlatform` in the crate's unit tests, which then run the product's code. The crate's
  `vulkan` feature (off by default) makes it `VulkanPlatform` (2.2, 10); nothing else of the C API
  changes, the header included.

### 11.2 The header

`include/reims_vgpu.h` is cbindgen's output: `cargo xtask header` writes it, and `--check` fails
when the committed header differs. QEMU's `apple-gfx`, the suite's runner, the pins, the C
harness and the smoke are compiled against it, and a change to it is additive: no function,
field, constant or status value is renamed, renumbered or removed (api.md section 1).

### 11.3 The dist tree

`cargo xtask dist` (D47) builds `reims-vgpu-c` for macOS 11.0 (2.3) and writes `target/dist`:

```
include/reims_vgpu.h                   the committed header
lib/libreims_vgpu.a                    the staticlib
lib/libreims_vgpu.dylib                the cdylib, @rpath/libreims_vgpu.dylib, signed ad hoc
lib/pkgconfig/reims-vgpu.pc            the package
share/reims-vgpu/ReimsVgpu.rom         the EFI option ROM, built from firmware/ (11.4, D48)
share/doc/reims-vgpu/                  the license: LICENSE and LICENSES/, as in the repository
```

`reims-vgpu.pc` takes its prefix from its own path, so the tree can move. Its `Libs.private` lists
the frameworks and libraries the dylib links (`otool -L`), which a static link needs, and its
`optionrom` variable names the ROM. The list follows what the code links: the window crate's ISO
keyboard check brought Carbon. A host that reads the package once, as meson does at configure, must
read it again when it changes (`make qemu` does; docs/lessons.md). `dist` builds the ROM first
(11.4), so a host without lld fails before the library builds. QEMU and the smoke link the archive;
the suite's runner links the dylib by its install name, with an `LC_RPATH` at the tree's `lib`.
`CARGO_TARGET_DIR` moves the tree with the rest of the build. `--platform vulkan` builds the crate
with its `vulkan` feature (11.1) into `target/dist-vulkan`, laid out the same, so the smoke and the
suite can run on the Vulkan platform's tree beside the Metal one.

On Linux `dist` builds on the Vulkan platform alone, into `target/dist-vulkan`, with
`lib/libreims_vgpu.so` (its soname its own name) in place of the dylib, and `Libs.private` as rustc
prints it for the archive. The archive bundles SDL3, which draws a display's window there (2.2),
so it is installed as one object (`ld -r` of the built archive's members) whose only global
symbols are the C API's `rvg_*` functions, every other made local (`objcopy
--keep-global-symbols`): a host that links its own SDL beside it, as QEMU's `-display sdl` does,
keeps its own. `dist` fails where the archive installed defines any other global. The shared
library exports the C API's functions alone through rustc's version script.

On macOS the archive bundles SDL3 too (the window crate's `macos-sdl` feature, 2.2), and is
installed the same way: one object (`ld -r -exported_symbols_list`) whose only globals are the C
API's functions. SDL's Objective-C classes are registered by name whatever their symbols, and a
host's SDL2 that is sdl2-compat (Homebrew's, which QEMU links for `-display sdl`) loads SDL3 with
the same class names: so the build renames them, and the class method SDL's `NSCursor` category
adds, through defines in the C flags SDL's CMake build takes (`xtask/src/dist/macos.rs`). `dist`
fails where the archive defines any other global, or a class under an SDL name.

### 11.4 The option ROM

`firmware/` is the PCI device's EFI option ROM: an x86-64 UEFI boot-service driver in C, which
QEMU's `apple-gfx-pci` hands an x86-64 guest's firmware as its default `romfile` (D48, 12).
Nothing else runs it: `vmapple`'s `apple-gfx-mmio` has no ROM, an arm64 guest's firmware does
not run x86-64 code, and the library never reads the file.

```
Makefile                 clang and lld: the image, then the ROM
src/ReimsVgpu.c          the entry point and the driver binding (Supported, Start, Stop)
src/GraphicsOutput.c     the Graphics Output Protocol (QueryMode, SetMode, Blt)
src/ComponentName.c      the driver's and the controller's names: "Reims vGPU Driver", "Reims vGPU"
src/ReimsVgpu.h          the PCI identity, the EFI display's registers, the driver's private data
src/Uefi.h               the part of the UEFI specification the driver uses
src/OptionRom.S          the ROM around the image: the 0x55AA header and the PCIR structure
```

**What the driver does.** The registers are the EFI display's (testing.md 6.2 and 6.22), and the
driver's macros carry the device's names for them (`REIMS_VGPU_REG_EFI_ENABLE` is `EfiEnable`).
- The entry point installs the driver binding and the component-name protocols on its image
  handle, and the supported-EFI-version protocol (2.10).
- `Supported` matches vendor 0x106b and device 0xeeee through `PciIo`.
- `Start` enables the PCI device, writes 0x1200 = 1, and reads the mode list: 0x1204's count,
  then for each mode 0x1208 = i and a read of 0x120c (width | height << 16), starting over when
  0x120c reads all ones. A count of 0, or over 127, gives three modes instead: 1920 × 1200,
  1920 × 1080 and 1600 × 1200. It allocates one buffer for the largest mode in guest RAM, as
  runtime services data, maps it for bus mastering, writes its start (0x1210) and length
  (0x1214), picks the framebuffer's address (below), and installs the Graphics Output Protocol
  on a child handle, whose device path is the controller's with an ACPI `_ADR` display node.
- `SetMode` writes the depth (0x1218 = 32) and the mode (0x121c). The framebuffer is linear and
  `PixelBlueGreenRedReserved8BitPerColor`, 4 × width bytes a row; the driver writes no stride,
  and the device takes a stride of 0 as 4 × width.
- `Blt` does its four operations on the buffer in guest RAM. The device presents that memory on
  its 100 ms timer, since QEMU gives it no trace callback (12).

**The framebuffer's address.** The protocol's `FrameBufferBase`, which boot.efi hands the
kernel, is BAR 2's address where BAR 2 shows the buffer, and else the buffer's own. QEMU's
`apple-gfx-pci` makes BAR 2 a window onto the RAM at the start the driver writes (0x1210;
docs/qemu.md, "apple-gfx on reims-vgpu"), so the registers, and what the library maps and
presents, are the same either way.
After its 0x1214 write the driver asks PCI I/O's `GetBarAttributes` for BAR 2: a memory BAR at
least as long as the buffer, below 4 GiB, with no address translation. It then writes a pattern
and its complement into the buffer's first and last dwords and reads each back through BAR 2
(`PciIo->Mem.Read`), restoring the dwords; all four must match. A macOS with no driver of the
device's own, an installer or recovery image, binds IOGraphics' generic `IONDRVFramebuffer`,
which takes the boot framebuffer over only inside one of the device's memory BARs (docs/qemu.md,
"x86-64 macOS guests: installers and recovery"). The buffer is runtime services data, which the
OS keeps past `ExitBootServices`. macOS's boot.efi may move runtime data before the kernel runs
(read from OpenCore's description of its `AvoidRuntimeDefrag` quirk), which would leave the
device and BAR 2 on the old pages: the pages stay the framebuffer's only where the loader keeps
runtime data in place. OpenCore's `AvoidRuntimeDefrag` does so: while boot.efi lays out the
runtime memory, it marks runtime code and data that carry the runtime attribute, which edk2 gives
this buffer, as MMIO (read from OpenCore's `OcAfterBootCompatLib` and edk2's `Page.c`).
reims-util's image, which reims-ctl makes for each new x86-64 VM, keeps it on from OpenCore
1.0.8's `Sample.plist`, and mac-temp's and macos-26's configurations set it (read from code and
from their `config.plist`). Before
BAR 2 the kernel drew to the same pages directly.

**Pages.** The driver writes 0x1210 in 4 KiB pages and rounds the length up to 4 KiB, the
x86-64 guest's pages. A library built for x86-64 counts 0x1210 in those pages and takes a length
of whole ones (`APV_PAGE_SHIFT`). One built for arm64 counts 16 KiB pages and refuses a length
that is not a 16 KiB multiple (testing.md 6.22), so there it would map a range other than the
driver's buffer, or none; that is read from the code. On a Linux host, x86-64 macOS guests boot
on `apple-gfx-pci` with this ROM, as `cargo xtask dist` builds it, and a library built for x86-64
(docs/qemu.md, "x86-64 macOS guests": a macOS 26.6.1 guest on an RTX 5080 Laptop GPU host,
2026-10-10 and 2026-10-11).

**The build.** `firmware/Makefile` compiles the C freestanding for `x86_64-unknown-windows`, whose
default calling convention is the Microsoft x64 one UEFI uses, at `-Os` with `-Werror`. `lld-link`
links it as an EFI boot-service driver (subsystem 11) with `/Brepro`, which puts a hash of the image
where the link time would go. `OptionRom.S` lays the ROM header and a PCIR structure (revision 3,
class 0x030200, code type 3, the last image) around the image with `.incbin`, padding with 0xff to a
512-byte multiple, and `ld.lld --oformat binary` flattens it. The PCI identity is defined once, in
`ReimsVgpu.h`, which the assembly includes too. `cargo xtask dist` runs `make -C firmware
BUILD=<target>/firmware rom` (11.3); `make -C firmware` alone writes `firmware/build/`. With one
compiler, the same sources build the same bytes, wherever `BUILD` points. The bytes are the
compiler's: on Linux, `dist` builds the ROM with the host's clang, and a Linux host's clang 23.1.1
builds a ROM of the same size as the Mac's, from Xcode's clang 21, whose bytes differ. So a check
that a change keeps the ROM's bytes compares two builds of one toolchain, and the Linux `dist`
prints the ROM's sha256 and the clang that built it.

**Its conventions.** The C follows EDK2's style and the UEFI specification's names (`EFI_STATUS`,
`gBS`, `IN` and `OUT`, `ReimsVgpu…` functions, `REIMS_VGPU_…` macros), not the Rust conventions
of section 17. `Uefi.h` declares only what the driver calls, and keeps every table's offsets by
declaring the services it does not call as opaque pointers.

**Checking it.** Neither `cargo xtask check` nor the suite builds or runs the ROM, and nothing
records what it does. A change is checked by building it. A change that is not meant to alter
what the guest's firmware sees keeps the ROM's bytes.

---

## 12. QEMU

`vendor/qemu`'s `apple-gfx` is plain C on the C API (`hw/display/apple-gfx.c`,
`apple-gfx-mmio.c` and `apple-gfx-pci.c`, `#include <reims_vgpu.h>`), with no Objective-C,
Metal, dispatch or blocks. Its one Objective-C file, `apple-gfx-activity.m`, calls nothing of the
C API: it holds a user-initiated activity against App Nap while a device is realized
(docs/qemu.md, "App Nap"). Meson finds the library as `dependency('reims-vgpu', method:
'pkg-config', static: true)`, the package `cargo xtask dist` writes (11.3), so `libreims_vgpu.a`
is part of QEMU's binary. Meson also installs the file the package's `optionrom` variable names
into QEMU's data directory as `ReimsVgpu.rom`, `apple-gfx-pci`'s default `romfile` (D48).

QEMU gives the device `raise_interrupt` and a memory map, and no task, memory or trace callback, and
makes one display a port for its `displays` property (1 to 8). `apple-gfx-pci` adds BAR 2, a window
onto the guest RAM at the EFI display's start (0x1210), which the library never sees (11.4). Its vCPU
threads make the MMIO accesses themselves, and the IOSurface device's values are truncated to QEMU's
32 bits. Each display's console has a pixman surface: a pool thread encodes the display's frame into
a frame target and reads it with `rvg_frame_target_read_pixels`, and a bottom half copies the pixels
into the surface. Without window mode the surface follows the guest's frames only while a display
change listener (VNC) or a screendump has asked for it within the last 6 s, and nothing is rendered
while none does. Under `-display none`, unless `REIMS_VGPU_WINDOW=0`, it passes each display a
window (D51), into which the library draws the guest's frames, draws the surface only when a
screendump or a listener asks, and sends the window's input to the guest's USB tablet and keyboard.
`docs/qemu.md` describes how it uses the API ("apple-gfx on reims-vgpu"), its configure lines
(`make qemu` builds it against `<target>/dist`), the VM boot method and the results.

---

## 13. Tests

The codebase is its own reference: its unit tests, its pins and the suite's expected files record
what it does. A change passes everything its predecessor passed. A change meant to alter
behaviour re-records the expected files or pins it alters from its own build, and their diff is
reviewed in the same commit.

### 13.1 `cargo xtask check`

Before every commit of code: the testkit manifest check (D50), `header --check` (the committed
header is cbindgen's output, 11.2), `cargo fmt --check`, `clippy -D warnings` over every target,
once with every feature and once with the tested features without the default ones, `cargo
nextest run --workspace --no-fail-fast` (each test in its own process, in the order
`.config/nextest.toml` sets on a Mac: the tests that act on the whole GPU first and alone) with
the tested features, and the doc-tests, which nextest does not run (`cargo test --workspace
--doc`). The tested features are every feature but the C API's `vulkan` (`TESTED_FEATURES` in
`xtask/src/main.rs`):
it makes the C API's platform the Vulkan one (11.1), and the pins and the C harness (13.5) drive
the product's. On Linux, where the Vulkan platform is the only one, the C API's `vulkan` is among
them. On a Mac the check also runs clippy for `x86_64-unknown-linux-gnu`, which type-checks every
Linux file with SDL left unbuilt, and, where Rosetta runs x86-64 code, core's tests as
`x86_64-apple-darwin`, for the x86-64 guest's 4 KiB pages and instructions.

### 13.2 The testkit

`core::testkit`, under `cfg(any(test, feature = "testkit"))`:
- **`FakePlatform`**: every platform trait over small object models, logging each call whose
  position is observable (4.3) and scripted by `Faults` (nil, error, raise or crash at the nth
  call of a selector on a class). A scripted raise is decided as the metal crate decides the
  same send (4.6), by a `Policy` of its own with the same labels: `Caught` is `Err`, `Fatal` dies
  at the send, and `On(on)` is as the caller's `OnFault` says. A free call answers from the
  model and is not logged. Its models keep what the contract sees: fresh numbering per launch,
  command buffers retaining what they reference until completion, counted IOSurfaces. Its
  results are owned values, released when core lets them go; the real pool's hold on Metal's
  `+0` results is tested on the metal crate.
- **Host and memory**: the host's callbacks (`FakeHost`) and a display executor
  (`FakeHostQueue`), `GuestRam`, a Mach VM model with remap aliasing and maximum protection,
  `VirtualClock` with both deadlines.
- **Threads**: `FakeThreads` (std threads, recorded; timers on the virtual clock), the thread
  recorder, `ThreadLauncher` (5.6).
- **Harnesses**: `capture` (`catch_death`, `expect_death`, `expect_hang`), and `death`, which
  re-runs one test in a child process to see what a real death leaves behind; `logs`, a capture
  of the `log` crate's lines.

Tests assert what the guest and host observe: guest bytes, MMIO reads, interrupts, host calls in
order, outcomes, IOSurface counts, object creation and release order, and the ordered Metal
calls in the fake's log. They never assert internal field values, and assert a log line only
where the line is the point of the test (14).

### 13.3 The real platform

`reims-vgpu-metal/tests/`: `spi_encodings` (each method `spi/` declares against the runtime),
`message_layer` (every send goes through the message layer), `retain_audit`, `numbering`
(fresh-process Metal numbering), `golden_coalescing` (`CoalescingSignal` against dispatch
sources), `golden_shared_mutex` (`WriterPrefRwLock` against libc++'s `std::shared_mutex`). The
metal crate's unit tests send each platform method to Metal, apply each descriptor value to a
Metal descriptor and read it back, and check the library's shaders on known pixels. The Vulkan
crate's tests run its platform on the host's device, and on a Mac compare its answers with
Metal's own for the same calls; they run strict (api.md section 2).

### 13.4 Loom

Loom models (`--cfg loom`) cover the FIFO's stamp queue and the remote task's completion table.

### 13.5 The C API's pins and harness

The host surface is pinned black-box where the suite does not reach it:
- **The pins**: `crates/reims-vgpu-c/tests/pins` builds `api_pins.c` against the committed
  header, runs it on the crate's dylib and compares its lines with `tests/pins/expected.tsv`
  (`RVG_PINS_RECORD=1` records them again). It is a cargo test, so part of 13.1.
- **The C harness**: `tests/c_harness.rs` builds `tests/c/harness.c` as C11 against the committed
  header, checks that it calls every function the header declares and that the header compiles
  alone as C++17 and as Objective-C, and runs it on the dylib.
- **The exports**: `tests/exports.rs` (11.1).

### 13.6 The suite

`tests/suite` drives the C API as QEMU does and compares what the guest and host observe
with its expected files (`make test`, and a second pass on the `checked` profile). Every case
must match, `died` and `watchdog` outcomes included. `make bless` records the expected files that
are missing or were recorded from other input; `make bless ARGS=--update` records the selected
cases again for a deliberate change of behaviour. `docs/testing.md` is its specification.

### 13.7 The smoke

`cargo xtask smoke` (`make smoke`) compiles a guest program (`xtask/src/smoke_guest.m`) against
the dist tree's header and links it with the tree's `libreims_vgpu.a` and the package's
`Libs.private`, as QEMU is linked. The guest makes a device over a memory map with process
isolation, brings it up through MMIO and the root FIFO, and runs an exec in an isolated task;
every fact it reports must equal the library's expected values (`GUEST_EXPECTED` in
`xtask/src/smoke.rs`).

---

## 14. Diagnostics

- **A line**: `log::error!`, `warn!`, `info!` or `debug!`, under its module's path as its target
  (`reims_vgpu_core::fifo::thread`), synchronously on the caller's thread. Warn is guest or host
  input refused and handled, error a failure the host's operator should see, info lifecycle, and
  debug detail.
- **Its arguments** are evaluated only when a logger takes the line. A value a line shows whose
  read has an effect (guest memory, which can die; a question to Metal) is read before the line,
  unconditionally and in the same order, so the process does the same with logging on or off.
- **Where lines go** is `logging.rs`'s: the host's own `log` logger, or with `REIMS_VGPU_LOG` or
  the C API's hook (`rvg_log_set_handler`), the library's, to stderr or the hook.
- **Tests** that are about a line read it through `testkit::logs::capture`, which relies on
  nextest's one process per test.

Kept: every read a line's arguments did, before the line. Free (diagnostics): the text, targets
and levels of lines.

---

## 15. Guest memory, lifetimes and device state

### 15.1 Guest memory

Reached only through `guest::ptr::GuestPtr` and `guest::page::GuestPage` accessors, each naming
its width (D18): one read where the guest may be writing; atomics with fixed orderings
(`SeqCst`: `ldar`, `stlr`, `ldsetal`; one acquire load, `ldapr`); the few single-instruction
unaligned accessors. Typed views sit on top (the shared page, the root node, the task root).
`zerocopy` is used only on private copies. A NULL dereference is a death at that access.

### 15.2 Two-tier lifetimes

- The host holds a `Device`, an `Arc<Handle>`; each `clone_ref` is a retain. When the last
  `Handle` goes, its drop runs `DeviceCore::shutdown` while the core is still alive.
- FIFOs, tasks, displays, mappings and the EFI display hold the core unretained (9), and those
  links still upgrade during the shutdown, so a late stamp still raises its interrupt.
- Weak loads (`DeviceCore::retain_self`, the report handler's) go through `Weak<Handle>` and
  answer nothing from the moment the shutdown starts.
- The display (`Display` over `DisplayCore`) and the IOSurface host device keep the same shape:
  the host's reference is the outer tier, and their end runs on the thread of the host's last
  release.

### 15.3 Drop order is explicit

A type whose fields' drops have effects in an order the API shows has an explicit `Drop` that takes
them in that order, with a test of the host-call order at shutdown: the device's copies of the
host's callbacks release `create_task`, `destroy_task`, `map_memory`, `unmap_memory`,
`read_memory`, `raise_interrupt`, `add_trace_range`, `remove_trace_range` in that order
(`host/callbacks.rs`); the device core's parts; a display's end; a remote task's reply captures;
a Rust API `Display`, whose share of its window goes before the reference itself
(`reims-vgpu/src/display.rs`); and a C `RvgDisplay`, whose window table goes before the display
(`reims-vgpu-c/src/display.rs`).
Field declaration order is never relied on.

The same holds for locals whose drop reaches the host or the guest (7.4). The resources a
discard deletes come back (`Deleted`) and go after the reply or completion, newest first
(`tasks/local.rs`, `gpu_task/service.rs`); a mapping-backed pointer carries its mapping
(`MappedAddress`) for as long as it is used; a walk's decoders stay with the exec past its commit
(`Decoders`); a rate map's layers outlive its descriptor; a task list taken for a walk is dropped
after the wait it bounds. Each is a named `drop` or a binding that outlives the call, never a
temporary's end.

### 15.4 Device state

The device's state is private to `device/`: `DeviceCore`'s fields, with `StagedMutex`es for the
registers and root objects that MMIO and the lifecycle read without the lock (7.1). Process-wide
state is `ProcessState` (D17).

---

## 16. What the contract keeps

Where the contract leaves the code free, it is plain Rust. The mechanisms below exist because
what crosses the API, or the platform under it, forces them; each row says what forces it. There
is no emulation of autorelease pools for core's own values, no preferences domain or
entitlement, no fixed log site or crash message, and no hand-rolled message layer.

| mechanism | where | forced by |
|---|---|---|
| a real autorelease pool around each unit of work (a command, a work item, an API entry), and a last-resort one on core's threads | `rt/pool.rs`, `rt/queue.rs`, `reims-vgpu/src/entry.rs`, `metal/src/os/autorelease.rs` | Metal: what it autoreleases (an uncommitted command buffer in a dev build, its encoders) must go at the unit's end, or the queue's 65th `-commandBuffer` hangs at a guest op |
| named release points for a drop that crosses the API: holders (`Held`, `Hold::keep`, a core thread's last-resort holder), the values a raise carries (`Raise::defer`), `MappedAddress`, `Deleted`, `Decoders`, `release_off_thread` | `rt/held.rs`, `resources/host.rs`, `tasks/shared.rs`, `deserializer/walk.rs` | the order of host callbacks (`destroy_task`, `unmap_memory`, the `destroy` at an object's end), IOSurface retain counts and open encoders against the rest of the API (7.4, 15.3) |
| a label at each raise site, fixed by whether a handler is live on the paths that reach it: `Err`, a death at the raise (`Raise::fatal`), or the caller's `OnFault` | `raise.rs`, `raise/on_fault.rs`, every raise site | where a run dies: a raise no frame handles ends the process at the raise, before any drop, and one a frame handles becomes that frame's fault, status or reply (6.2) |
| explicit `Drop` orders, and two-tier lifetimes (an unretained core behind the host's handle) | `device.rs`, `display.rs`, `iosurface/device.rs`, `host/callbacks.rs` | the host-call order at a teardown; a late stamp still raises its interrupt during the shutdown, and a weak load answers nothing once it starts (15.2, 15.3) |
| `StagedMutex`, the writer preference of `WriterPrefRwLock` (`parking_lot::RwLock`), `CoalescingSignal` | `sync/`, `rt/signal.rs` | what an unlocked MMIO or guest read sees, a queued writer's turn ahead of new readers, the handler calls a burst of merges gives |
| every Objective-C call inside a catch: the message layer around Metal and Foundation, `host_call` around every host call | `metal/src/msg.rs`, `metal/tests/message_layer.rs`, `reims-vgpu-c/src/ffi.rs` | an exception out of Metal, Foundation or host code ends the process at the call, whatever handler the caller has (D35, 6.7) |
| the containers whose order is named | `collections/`, `tasks/remote/completions.rs` | orders that reach guest memory and host calls (the task table's fan-out, the root VA, a crash's token order), the same in every process (8) |
| one typed call per Metal method, each ordered call where core makes it, with raw guest values; descriptor values applied to real descriptors in a stated order | `platform/`, `metal/src/msg.rs`, `metal/src/spi/`, `metal/src/gpu/` | Metal's results, aborts and numbering (resource indices, GPU addresses) follow the calls a process makes (D4, D5) |
| one fresh service per isolated task, on threads of its own, whose death ends only its task, and no Metal allocation the guest did not ask for | `tasks/isolation/`, `gpu_task/` | D10, D40: invalid guest input in an isolated task ends that task (status 4), not the device; an allocation of the service's own would move the numbering the guest sees |
| the guest-memory access shapes: widths, single reads, single-copy-atomic stores, the orderings | `guest/ptr.rs` | what a guest racing the host can see (15.1) |
| values kept for good, each under a named rule | `raise/kept.rs` | what a kept value holds shows in numbering, retain counts and address space, and an unmap or a host release never comes (6.6) |
| one display thread per process for every display's callbacks | `reims-vgpu/src/platform.rs` | the order of the display callbacks across displays and devices (D49) |
| a display's window that never holds its display but for a frame in flight, closes at the display's last reference first, and waits for nothing, the main thread included | `reims-vgpu/src/window.rs`, `reims-vgpu/src/display.rs`, `reims-vgpu-window/src/handle.rs` | where the display's end and its tables' `destroy`s land (`api.destroy.display_*`, `api.lifetime.display_through_*`), and a host whose main thread creates displays before it runs its run loop (`api.window.no_run_loop`; D51) |
| `extern "C"` exports, each inside `ffi::export`, and the library's panic hook | `reims-vgpu-c/src/ffi.rs`, `singleton.rs` | the header: nothing unwinds into the caller, and a panic of the library's ends the process before any drop (6.5) |

---

## 17. Conventions

`docs/conventions.md` has the full list; the ones code cites:

### 17.1 Naming

Names say what things mean (`Raise`, `Held`, `SerialQueue`, `define_task`). No `PG` prefixes,
no Objective-C class or selector-keyword names. A field of a guest record, the wire or a copy of
a Metal struct whose meaning nothing states is `field_xx`, its byte offset in hex; a name that is
ours carries `// name is ours: <evidence>`. Guest-ABI names follow `docs/testing.md` 6.

### 17.2 Odd behaviours and comments

An odd behaviour the guest or host can see carries a doc comment at the code that says what it is
and who sees it, and a test named for the behaviour. It is kept as it is: changing it changes what
the guest or host sees, and is done only as a deliberate change that re-records what it alters
(13). Comments say why, in terms of what the guest or host observes, and may name a test or
scenario.

### 17.3 `unsafe`

Core denies `unsafe_code` at its root (D43). It is allowed, with a `// SAFETY:` comment per block
(`clippy::undocumented_unsafe_blocks`), only in:
- `guest/ptr.rs`: the one raw access to guest RAM;
- `platform/host_range.rs`: `HostRange`'s constructor;
- the sites that make a no-copy buffer or upload over host memory they hold a lease on:
  `tasks/local/memory.rs`, `gpu_task/memory.rs`, `resources/buffer/detach.rs`, `display/efi.rs`,
  `deserializer/blit.rs`;
- the testkit (`clock.rs`, `guest_ram.rs`, `death.rs`: `mmap` over live pages, async-signal-safe
  handlers and `_exit`), and one test module (`tasks/object_lists.rs`'s `alias_tests`, whose Mach
  calls make real page aliases).

The metal crate, the window crate, the API and the C API use `unsafe` at their FFI edges.

### 17.4 Lints and integers

Core denies `indexing_slicing`, `unwrap_used`, `expect_used`, `panic`, `unreachable`,
`disallowed_types`, `disallowed_methods` and `mod_module_files` everywhere, and `as_conversions`,
the lossy casts, `integer_division_remainder_used` and `undocumented_unsafe_blocks` outside tests.
Arithmetic uses std's explicit forms; `num.rs` holds the rest: wrapping alignment, the lossless
`usize` conversions, truncation to the low bits, the display's one rounding rule, and the integer
results the guest sees, each a named rule that says why it is kept: `div0`/`rem0` (arm64's `udiv`
by zero gives 0, `urem` gives the dividend), the mask bits a shift makes mod its width
(`bit32`, `bit64`, `bit64_set`), the sign or zero extension of a narrow field (`zero_extend_16`,
`sign_extend_16`, `sign_extend_32`), and the forward copy through aliased mappings
(`forward_copy`). It is the only production module that may use `as`. `#[expect(lint, reason =
"…")]` on the narrowest item replaces an `allow` with a comment.
