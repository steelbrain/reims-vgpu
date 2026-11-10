# The API

This is the library's API, as a host uses it:
- the C API (`crates/reims-vgpu-c`), whose header cbindgen generates from the crate;
- the Rust API it wraps (`crates/reims-vgpu`).

Its one object is the reims-graphics device (PCI 1234:5247), the GPU our guest drivers (`guest/`)
drive. QEMU's `reims-graphics-pci` (`vendor/qemu/hw/display/reims-graphics-pci.c`) is its host, in
C ([qemu.md](qemu.md), "The reims-graphics device"). The device's wire protocol, what the guest
sees, is [graphics-kext/design.md](graphics-kext/design.md).

The header, `crates/reims-vgpu-c/include/reims_vgpu.h`, is the reference. Sections 1 and 5 state
the rules, and the header repeats each rule where it applies. The C API has no Objective-C types,
dispatch queues or blocks, and takes no Metal object from the host. Callbacks arrive on the
library's threads, and a window's on the window's thread: the process's main thread on macOS, the
library's `reims-vgpu.ui` on Linux.

## 1. Principles

### The layers

| layer | crate | what it is |
|---|---|---|
| core | `reims-vgpu-vulkan`'s `core` | the platform traits, the runtime (threads, queues, holds), raises and deaths, the display's present pipeline, guest memory and the fake platform the tests run on |
| platform | `reims-vgpu-vulkan` | core's `Platform`: `VulkanPlatform`, over metal2vulkan's runtime, on Linux, and on macOS over MoltenVK, a build for development and tests |
| window | `reims-vgpu-window` | a display's own window (AppKit's or SDL3's on macOS, SDL3's on Linux), its input and the host's screens; it knows nothing of devices |
| device | `reims-vgpu-graphics` | the wire: the records, rings and registers the guest and the host share, and the C header the guest's kexts and the option ROM compile |
| Rust API | `reims-vgpu` | the graphics device over the platform: its rings, decoder, display and window, and owned Rust types for its config and callbacks |
| C API | `reims-vgpu-c` | a thin FFI layer over the Rust API; cbindgen writes `include/reims_vgpu.h` from it |

The C layer converts and forwards; it decides nothing. Every callback it builds calls the host's
function synchronously, on the thread its caller calls it on: the device's threads for the
device's callbacks, and the window crate's thread for a window's. So the order and the threads of
every call to the host are the device's, and a window's are the window's.

### Objects

| object | made by | released by | its end |
|---|---|---|---|
| `RvgGraphicsDevice` | `rvg_graphics_device_create` | `rvg_graphics_device_release` | its window closes and its threads end at its release, waited for by nothing; its region stays mapped until both threads and the window's GPU work have let it go |

- **One reference.** The host owns exactly one reference to a device; there is no retain.
- **NULL.** A release of NULL does nothing. Elsewhere a NULL handle is `RVG_ERR_INVALID_ARGUMENT`
  where the function returns an `RvgStatus`, and ends the process where it does not.
- **Configs are borrowed** for the call, and everything they point to: the library copies what it
  keeps. There are no descriptor objects and no setters.

### Configs and `struct_size`

Every struct the host fills in, and every struct the library hands out, starts with
`uint32_t struct_size` (`crates/reims-vgpu-c/src/sized.rs`).

- **A struct the host writes:**
  - `struct_size` below the struct's size is `RVG_ERR_INVALID_ARGUMENT`;
  - a `struct_size` above the library's is accepted when every byte past the library's size is
    zero, and is `RVG_ERR_UNSUPPORTED` otherwise. A host built against a newer header then works
    with an older library as long as it leaves the new fields at zero.
  - `RvgGraphicsCallbacks`, `RvgWindowConfig` and `RvgWindowCallbacks` are at their first
    version. `RvgGraphicsDeviceConfig` is at its second, which added `guest_arch`: a host built
    before it, whose `struct_size` ends before that field, serves an x86-64 guest. A struct that
    grows keeps its first version's size as the least it accepts, and reads the later fields of a
    host built before them as zero.
- **A struct the library writes** (`RvgCursorGlyph`, `RvgWindowScroll`): the host reads a field
  only if `struct_size` covers it.
- **Zero is every field's default**, and a NULL table is a table with every callback absent.
- **Plain data needs no size**: `RvgMemoryMap`, `RvgMemoryRange`, `RvgMemoryRangeFile` and
  `RvgDisplayCoord` are fixed. `RvgMemoryRangeFile.flags`, 0 today, has room for more.

### Status codes

Functions that can fail return `RvgStatus` (a `uint32_t`); `rvg_status_message` gives a static
description, and "unknown status" for a value the library does not define. A host treats an
unknown status as a failure.

| value | name | when |
|---|---|---|
| 0 | `RVG_OK` | |
| 1 | `RVG_ERR_INVALID_ARGUMENT` | a NULL handle, config or out-pointer where one is needed; a `struct_size` too small; a refused memory file; a `guest_arch` that is none of `RVG_GUEST_ARCH_*`; a doorbell that is not open, or a second one; `read_display`'s destination missing or too short |
| 2 | `RVG_ERR_UNSUPPORTED` | a larger `struct_size` with non-zero bytes the library does not know |
| 3 | `RVG_ERR_METAL` | the platform refused the display's texture or its draw (`rvg_graphics_device_read_display`) |
| 4 | `RVG_ERR_GPU` | the call's GPU work ended with an error |
| 5 | `RVG_ERR_HOST_RESOURCES` | the host refused the device's region (its memory), a pipe or a thread |

`rvg_graphics_device_create` reads, in order, the config, its callbacks, its window config and the
window's callbacks, then the memory map and the files behind its ranges. The struct checks
(`INVALID_ARGUMENT`, `UNSUPPORTED`) come first and touch nothing. The files are refused when given
without a map, then, range by range from the first, for an `fd` below -1, for `flags` other than 0,
and, for an `fd` of 0 or more, for a file `fstat` cannot read, a file holding fewer than `offset`
plus the range's length bytes, and an `offset` that is not a multiple of the host's page size. Each
refusal logs a line saying which, and no file is duplicated until every entry has passed.

### Thread safety

- Every function may be called from any thread, on the same handle from several threads at once,
  the vCPU's included. A handle must not be used during or after its release.
- **Calling back.** Only `rvg_graphics_device_mmio_write_taking` calls back, and only
  `raise_interrupt`, on the calling thread, for a doorbell it answers there (section 5). A refused
  create runs its tables' destroys; no other function calls back.
- **Waiting.** No function waits for the device's threads, the display thread or the main thread.
  A STATUS write of 0 and `rvg_graphics_device_reset` wait only for the command the device's control
  thread is taking, and for a take of a queue's ring or a post onto one under way, none of which
  waits for anything of the host's. `rvg_graphics_device_read_display` waits for its own draw on the
  GPU, on a queue of its own, and one read runs at a time.

### What may be called inside a callback

| inside | may call |
|---|---|
| `raise_interrupt` | `rvg_status_message` and the `rvg_graphics_device_*` functions, except the release of its own device |
| `cursor_glyph`, `cursor_show`, `cursor_move` | every function: the library never waits for the display thread |
| a window callback (`pointer`, `wheel`, `key`, `close`, `scroll`) | every function; it must not block (section 5) |
| a `destroy` | nothing |

### Deaths

The process ends at the operation, with a line on stderr (`reims-vgpu: fatal: <what died>`) and
then `abort()`, for:
- a NULL handle to a function that returns no status;
- anything a host callback throws (`ffi::host_call` runs each call inside a catch);
- a panic under an export: the library's panic hook ends the process before anything unwinds;
- a guest command the platform cannot carry out as Metal does, where Metal ends the process: a
  raise of Metal's that nothing catches, and on the Vulkan platform a refusal of the runtime's it
  passes on ([accuracy-ledger.md](accuracy-ledger.md)).

Every export is `extern "C"`: nothing unwinds out of one into the host
(`crates/reims-vgpu-c/src/ffi.rs`).

### The process's exit

A device's create registers an exit handler of the library's, once the GPU's driver is loaded,
and the library registers it again as its work goes on (`reims_vgpu_vulkan::exit_gate`). As the
process exits (`exit`, or the end of `main`), the handler waits for the library's work in progress
on other threads than the exiting one, at most 2 s, or 60 s while a pipeline compile is among it,
and then the exit goes on. Work that would begin once it ran, on the library's threads or in a
call of the host's on another thread, parks that thread for good instead, so no work of the
library's runs beside the GPU driver's destructors. A call on the exiting thread, from a later exit
handler, runs as before. The host does nothing for it: QEMU's quit waits for it.

## 2. The C header

The header has the ten functions, every type, callback typedef and constant, with the thread and
ordering rules on each callback and each function. Its preamble states the rules for handles,
configs, callbacks and deaths. The C API's tests compile it as C11, as C++17 and, on macOS, as
Objective-C.

| group | items |
|---|---|
| constants | the status codes; the PCI facts (`RVG_GRAPHICS_PCI_VENDOR_ID` 0x1234, `_DEVICE_ID` 0x5247, `_CLASS` 0x380, `RVG_GRAPHICS_PCI_BAR_REGISTERS` 0, `RVG_GRAPHICS_REGISTERS_SIZE` 4096, `RVG_GRAPHICS_MSIX_OFFSET` 2048, `RVG_GRAPHICS_MSIX_VECTORS` 1, `RVG_GRAPHICS_PCI_BAR_REGION` 2); the doorbell (`RVG_GRAPHICS_REG_DOORBELL`, `RVG_GRAPHICS_DOORBELL_CONTROL`); `RVG_GRAPHICS_ALLOCATIONS_OFFSET`; `RVG_GUEST_ARCH_X86_64` 0 and `RVG_GUEST_ARCH_ARM64` 1; `RVG_WINDOW_BUTTON_*`, `RVG_WINDOW_SCROLL_PHASE_*`, `RVG_WINDOW_SCROLL_MOMENTUM_*` |
| status | `rvg_status_message` |
| the device | `RvgGraphicsDeviceConfig`, `RvgGuestArch`, `RvgGraphicsCallbacks` and its function types, `RvgCursorGlyph`, `RvgDisplayCoord`, `RvgMemoryMap`, `RvgMemoryRange`, `RvgMemoryRangeFile`; `rvg_graphics_device_create`, `_release`, `_region`, `_mmio_read`, `_mmio_write_taking`, `_set_doorbell`, `_reset`, `_read_display`, `_cursor_position` |
| its window | `RvgWindowConfig`, `RvgWindowCallbacks`, `RvgWindowPointerFn`, `RvgWindowWheelFn`, `RvgWindowKeyFn`, `RvgWindowCloseFn`, `RvgWindowScrollFn`, `RvgWindowScroll`; no function of its own: `RvgGraphicsDeviceConfig.window` carries it |

### Settings and the environment

Everything a host can set that shapes what the library does, beside the guest's own input:
- **The config**: the device's callbacks, its window (its callbacks and the mode it opens at),
  the guest's RAM with the files behind it, which let the GPU use an IOSurface's own pages in place,
  and the guest's architecture, which the device's `isFramebufferReadSupported` follows: the
  host GPU's answer for an arm64 guest, no for an x86-64 one. Zero is every field's default
  (section 1): an x86-64 guest.
- **Cargo features**, for a host that builds the library: `macos-sdl` (on by default) builds
  SDL3's window into the macOS library ([development.md](development.md)). `vulkan`
  (`reims-vgpu`, `reims-vgpu-c`) changes nothing: the Vulkan platform is the API's only one, and
  the feature stays for the builds that name it.
- **The environment.** The library reads these once per process; an unset or unparsable value is
  the default.

  | variable | what it does |
  |---|---|
  | `REIMS_VGPU_LOG` | the library's log lines to stderr, filtered by a level and `target=level` pairs (core's `logging`); unset, nowhere |
  | `REIMS_VGPU_WINDOW_SCREENS`, `_BORDERLESS`, `_REFRESH` | a window's screen, its style and the offered mode's refresh rate (`reims-vgpu-window`) |
  | `REIMS_VGPU_WINDOW_IMPL` | macOS: `sdl` opens SDL3's window in place of AppKit's, in a build with `macos-sdl` |
  | `REIMS_VGPU_STRICT` | the Vulkan platform: `1` ends the process at each difference from Metal's answer it would otherwise log |
  | `REIMS_VGPU_EXACT` | the Vulkan platform: the runtime's exact forms in place of its fast ones, by field ([accuracy-ledger.md](accuracy-ledger.md)) |
  | `REIMS_VGPU_CENSUS_DIR` | the Vulkan platform on Linux, a diagnostic for inventory runs: each difference that would end the process is recorded in that directory and logged instead, and the run says it is not faithful to Metal |
  | `REIMS_VGPU_TRACE` | the Vulkan platform, a diagnostic for profiling: a timeline of the device's work, a line per event, written to the file it names (core's `trace`) |
  | `REIMS_VGPU_TRACE_DEGRADE` | with `REIMS_VGPU_TRACE`, `1`: every occurrence of every difference from Metal the runtime announces is written to the trace |

  Of these, only `REIMS_VGPU_STRICT`, `REIMS_VGPU_EXACT` and `REIMS_VGPU_CENSUS_DIR` change what the
  guest or host observes, and only on the Vulkan platform; `REIMS_VGPU_WINDOW_SCREENS` and
  `_REFRESH` change the mode the window's display offers its guest.

### cbindgen

`crates/reims-vgpu-c/cbindgen.toml`: language C, `c99` doc style, `cpp_compat`,
`sort_by = "None"`, `parse_deps = false`; its `header` preamble states the rules; functions come in
the order cbindgen parses them, the order of `lib.rs`'s public modules. On the Rust side the codes
are `pub type RvgStatus = u32` and `pub const` items, so a host's out-of-range value is never a
Rust enum; the handle is a Rust struct cbindgen emits as opaque; the configs are `#[repr(C)]`.

## 3. The Rust API (`reims-vgpu`)

`reims-vgpu` is the device; the C API wraps it one to one. Its rustdoc is the reference:
- `GraphicsDevice<P = DefaultPlatform>`, made by `GraphicsDevice::new(GraphicsConfig)` on the
  default platform (`VulkanPlatform`) or `with_platform`, so its
  forwarding is unit-tested on core's fake platform;
- `GraphicsConfig`: `GraphicsCallbacks` (closures, `Arc<dyn Fn … + Send + Sync>`, each optional),
  an optional `WindowConfig`, an optional `MemoryMap` with its `MemoryFile`s, and the guest's
  `GuestArch` (`X86_64`, the default, or `Arm64`);
- `Error`: `InvalidArgument`, `Metal`, `Gpu`, `HostResources`, the C API's statuses but
  `UNSUPPORTED`, which only the C layer's struct checks give;
- the host video types (`GraphicsVideo`, `PlanePages`, `VideoDecode`, `VideoPlane`), through which
  the device's video decodes reach the host's decoders ([video/](video/)).

Every public method runs as an entry (`crates/reims-vgpu/src/entry.rs`): an autorelease pool and a
holder of its own, so what the call held or Metal autoreleased on the host's thread goes before it
returns. The library logs through the `log` crate; a Rust host installs its own logger.

## 4. The functions

| function | calls back | may block on |
|---|---|---|
| `rvg_status_message` | never | nothing |
| `rvg_graphics_device_create` | a refusal's destroys | nothing of the host's |
| `rvg_graphics_device_release` | never | nothing: the threads end and the destroy runs afterwards |
| `rvg_graphics_device_region` | never | nothing |
| `rvg_graphics_device_mmio_read` | never | nothing |
| `rvg_graphics_device_mmio_write_taking` | `raise_interrupt`, for a doorbell answered on the calling thread | a STATUS write of 0: the command the control thread is taking |
| `rvg_graphics_device_set_doorbell` | never | nothing |
| `rvg_graphics_device_reset` | never | the command the control thread is taking, and a ring's take or post under way |
| `rvg_graphics_device_read_display` | never | its draw on the GPU |
| `rvg_graphics_device_cursor_position` | never | nothing |

## 5. Callbacks and threads

### Where each callback arrives

| callback | thread | serialized with | order |
|---|---|---|---|
| `raise_interrupt` | the thread that posts the event it raises for: the device's control thread, `reims-vgpu.graphics-control`; one of ring 0's threads, `reims-vgpu.graphics-ring0-<i>`, which carry out a guest process's control messages, its presents and a block's pages and downloads, each on its lane; its refresh thread, `reims-vgpu.graphics-refresh`; a command queue's service thread, `reims-vgpu.graphics-queue-<id>`, or one of its lane's threads, which walk and commit its command buffers, `reims-vgpu.graphics-lane-<id>`; a pipeline compile's thread, `reims-vgpu.graphics-compile-<n>`; a thread that makes IOSurfaces' and pages blocks' views, `reims-vgpu.graphics-surfaces-<i>`; a thread that makes indirect command buffers, `reims-vgpu.icb-create-<i>`; a thread that lets go of a closed process's objects and answers its close, `reims-vgpu.graphics-closes-<i>`; or the platform's thread that runs a command buffer's scheduled or completed handler; or the host's own, inside `rvg_graphics_device_mmio_write_taking`. Never inside another call of the host's | nothing: they may call at once | each after the completion ring's message it raises for is published |
| `cursor_glyph`, `cursor_show`, `cursor_move` | the display thread: one serial worker for the whole process (after 5 s idle it exits, and the next callback starts a new one) | every display callback of the process: one at a time | the order the device's control thread took the guest's cursor commands, and its resets |
| a window's `pointer`, `wheel`, `key`, `close`, `scroll` | on macOS the process's main thread, where AppKit runs; on Linux the library's thread `reims-vgpu.ui`; never inside a call of the host's | every window callback of the process: one at a time | the order the events arrived |
| `destroy` (each table) | the device's: once its threads have ended, after their last callback, possibly on one of them after the release has returned; the window's: once the release has closed the window, on the releasing thread or, while a window callback runs, the window's thread once it returns; either, on the caller's thread, in a refused create | nothing | once, after every call of its table |

Threads are named `reims-vgpu.<role>`: those above, `reims-vgpu.display` (the display thread),
`reims-vgpu.window` (a window's drawing thread), `reims-vgpu.ui` (on Linux, the thread every
window's events and callbacks run on), and `reims-vgpu.graphics-video-<i>` and
`reims-vgpu.graphics-event-waits` (a graphics device's). The names are diagnostics, not contract.
A queue's worker, the display thread among them, has core's `WORKER_STACK`, 512 KiB.

### The locks around `raise_interrupt`

The device's callbacks may wait for a lock the host holds while it calls the device (QEMU's BQL):
no function of the device waits for its threads, and the library calls them with no lock held that
one of its functions waits for, directly or through a thread of the library's. `raise_interrupt`
may run under these:
- a command queue's claim on its next message, on the queue's service thread and inside
  `rvg_graphics_device_mmio_write_taking`;
- there, at a command buffer's `CB_SUBMITTED` answered on the calling thread, the decoder's
  contexts' lock, the buffer's context's table read, its queue's answer slot and the reset gate
  read;
- the decoder's lock, with the download's answer slot, at a `SURFACE_DOWNLOAD` answered at once;
- a command buffer's `CB_SUBMITTED` latch, wherever that is posted;
- the presents' queue, at a present answered at once with no window.

No function of the device takes any of them, and `_mmio_write_taking` only tries the claim and the
decoder's locks. The one exception is `raise_interrupt` inside
`rvg_graphics_device_mmio_write_taking`, on the calling thread: it must not wait for a lock the
host holds while it calls that function. A control-ring download answered there raises with no
lock of the library's held. `rvg_graphics_device_read_display` takes none of them either: the host
may call it holding its lock, though QEMU calls it on a thread of its pool, so the GPU time is not
spent under the BQL.

### The display thread

The cursor callbacks run on one serial executor per process (core's `SerialQueue`), made at the
first device whose host takes the cursor; a device drops its handle at the end of its
threads. Jobs run in the order they were queued, across devices. The library never waits for the
display thread, and its drop joins nothing, so a display callback may block, for example on QEMU's
BQL, and may call any function; the next display callback waits behind it.

### The main thread

A window's callbacks run one at a time across every window of the process, in the order the events
arrived, and never inside a call of the host's: on macOS on the process's main thread, where AppKit
runs, and on Linux on the library's own thread `reims-vgpu.ui`. On Linux nothing of the library
runs on or waits for the process's main thread, and what follows about the run loop and the
application is macOS's.
- **They must not block.** Until one returns, no window of the process takes input or redraws its
  frame. The main thread is the host's too: QEMU parks it in its run loop, and a callback that waits
  for something that thread would do never returns. QEMU's queue their event for its BQL and return.
- **The library never waits for the main thread.** A host may create its device on the main thread,
  before that thread runs its run loop, so a create or a release that waited for it would hang. A
  window opens once the main thread runs its run loop, and a release closes it without waiting.
- **What the library runs there.** Each window's opening, its mode and cursor changes, and its
  close are blocks the library posts to the main run loop, `CFRunLoopPerformBlock` in its common
  modes then `CFRunLoopWakeUp`, and they run in the order posted: under a plain `CFRunLoopRun`,
  under `[NSApp run]`, and while a window is resized live or a menu is open. The library never uses
  the main dispatch queue. The application's start is posted at the first window's attach, alone;
  when the host has made no `NSApplication`, it makes the library's own and calls `[NSApp run]`
  (`crates/reims-vgpu-window/src/native/macos/main_thread.rs`).
- **A main thread that never runs its run loop** opens no window. What was posted to it stays
  queued, never run and never released; it holds the window at most weakly, so the release still
  ends the window and runs its table's `destroy`.

### Callback tables and their destroys

Each table has one `destroy`, which runs when the last closure built from it goes
(`crates/reims-vgpu-c/src/table.rs`): for a table with functions, at the end of the object, where
the device drops the last of its closures; for one with none, at the host's release. The create
reads the config, its callbacks, the window config and the window's callbacks before either table
is the library's: a `struct_size` refusal of any of the four runs neither `destroy`, and every later
refusal runs both before it returns, the window's first. The release closes the window before it
lets the device go, so the window table's `destroy` runs first.

### Blocking and re-entry

- A device callback may block on the host's lock, as above; `raise_interrupt` inside
  `rvg_graphics_device_mmio_write_taking` must not wait for a lock the host holds around that call.
- A window callback must not block at all ("The main thread", above).
- No device callback may release its own device (the header's "May call").

## 6. Keeping the API whole

- **The header.** `cargo xtask header` regenerates `include/reims_vgpu.h` with cbindgen 0.29.4, and
  `--check`, part of `cargo xtask check`, fails when the committed header differs from cbindgen's
  output. A different cbindgen version is reported, not compared. The C API's
  `header_has_no_drift` test runs the same comparison when cbindgen 0.29.4 is installed.
- **The exports** (`tests/exports.rs`): every export is
  `#[unsafe(no_mangle)] pub [unsafe] extern "C" fn rvg_…` whose whole body is one call of
  `ffi::export`; no other function or function-pointer type of the product code has a C ABI, except
  the host's callback types, which are `extern "C-unwind"` for `ffi::host_call`'s catch; the header
  declares exactly the exports, and the dylib exports every one.
- **The C harness** (`tests/c_harness.rs`) builds `tests/c/harness.c` against the committed header
  with `-std=c11 -Wall -Wextra -Werror -pedantic`, links it to the crate's dylib and runs it, and
  checks that the header compiles alone as C++17 and, on macOS, as Objective-C.
- **QEMU** calls the API from `vendor/qemu/hw/display/reims-graphics-pci.c`, in the same repository:
  a change to the API lands with QEMU's side of it, and a boot checks them together ([qemu.md](qemu.md)).
