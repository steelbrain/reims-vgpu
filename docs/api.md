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
library's `reims-vgpu.ui` on Linux and Windows.

## 1. Principles

### The layers

| layer | crate | what it is |
|---|---|---|
| core | `reims-vgpu-vulkan`'s `core` | the platform traits, the runtime (threads, queues, holds), raises and deaths, the display's present pipeline, guest memory and the fake platform the tests run on |
| platform | `reims-vgpu-vulkan` | core's `Platform`: `RtPlatform`, over metal2vulkan-runtime, on the host's Vulkan driver on Linux and Windows, and on macOS on Mesa's KosmicKrisp |
| window | `reims-vgpu-window` | a display's own window (AppKit's or SDL3's on macOS, SDL3's on Linux and Windows), its input and the host's screens; it knows nothing of devices |
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
| `RvgGraphicsDevice` | `rvg_graphics_device_create` | `rvg_graphics_device_release` | its windows close and its threads end at its release, waited for by nothing but the release of the process's last device ("The process's exit"); its region stays mapped until both threads and the window's GPU work have let it go |

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
  - `RvgWindowConfig` and `RvgWindowCallbacks` are at their first version.
    `RvgGraphicsDeviceConfig` is at its fifth: the second added `guest_arch`, and a host built
    before it, whose `struct_size` ends before that field, serves an x86-64 guest; the third added
    `doorbell_pages`, and a host built before it maps no doorbell pages; the fourth added
    `display_count` and `display_windows`, and a host built before it makes a device of one
    display; the fifth added `pool_physical_address` and `pool_physical_length`, and a host built
    before it has no pool. `RvgGraphicsCallbacks` is at its fifth: the second added
    `route_doorbell`, and a host built before it routes no doorbell; the third added
    `route_doorbell_page`, and a host built before it routes none on a doorbell page; the fourth
    added `cursor_glyph_at`, `cursor_show_at` and `cursor_move_at`, and a host built before it is
    told the cursor of display 0 alone; the fifth added `pool_add` and `pool_remove`. A
    struct that
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
| 1 | `RVG_ERR_INVALID_ARGUMENT` | a NULL handle, config or out-pointer where one is needed; a `struct_size` too small; a refused memory file; a `guest_arch` that is none of `RVG_GUEST_ARCH_*`; a `doorbell_pages` other than 0 or 1; a `display_count` past `RVG_GRAPHICS_DISPLAYS_MAX`, or `display_windows` not given exactly where `window` is (one for each display from 1), or holding NULL; a pool range not in whole 2 MiB, past the address space, overlapping the memory map, or given without `pool_add` and `pool_remove`; a doorbell that is not open, or a second one; `read_display`'s destination missing or too short; `read_display_at`'s display at or past the device's count; a pool call on a device without a pool, an address the device asked no add at, a second `pool_added` of one piece, or a `pool_grow` of 0 bytes |
| 2 | `RVG_ERR_UNSUPPORTED` | a larger `struct_size` with non-zero bytes the library does not know |
| 3 | `RVG_ERR_METAL` | the platform refused the display's texture or its draw (`rvg_graphics_device_read_display`) |
| 4 | `RVG_ERR_GPU` | the call's GPU work ended with an error |
| 5 | `RVG_ERR_HOST_RESOURCES` | the host refused the device's region (its memory), a pipe or a thread, or has no GPU device the library can use: none, or one its Vulkan driver refused |

`rvg_graphics_device_create` reads, in order, the config, its callbacks, its window config and the
window's callbacks, each display window's config and callbacks from display 1 on (where the
window is given and the count is one the device takes), then the memory map and the files behind
its ranges. The struct checks
(`INVALID_ARGUMENT`, `UNSUPPORTED`) come first and touch nothing. The files are refused when given
without a map, then, range by range from the first, for an `fd` below -1, for `flags` other than 0,
and, for an `fd` of 0 or more, for a file `fstat` cannot read, a file holding fewer than `offset`
plus the range's length bytes, and an `offset` that is not a multiple of the host's page size. Each
refusal logs a line saying which, and no file is duplicated until every entry has passed.

### Thread safety

- Every function may be called from any thread, on the same handle from several threads at once,
  the vCPU's included. A handle must not be used during or after its release.
- **Calling back.** `rvg_graphics_device_mmio_write_taking` and `rvg_graphics_device_doorbell_write`
  call `raise_interrupt` on the calling thread, for a doorbell they answer there, and
  `rvg_graphics_device_mmio_write_taking` and `rvg_graphics_device_reset` call `route_doorbell`
  and `route_doorbell_page` there to drop the routes a reset ends, and
  `rvg_graphics_device_mmio_write_taking` calls `route_doorbell_page` there to drop the routes
  on a doorbell page of its old context's queues, where a `DOORBELL_PAGE_BIND` write binds it to
  another context (section 5). A refused create runs its
  tables' destroys; no other function calls back.
- **Waiting.** No function waits for the display thread or the main thread, and none but the
  release of the process's last graphics device and `rvg_quiesce_for_exit` waits for the device's
  threads: the release waits for them at most 5 s, and the quiesce at most its bound ("The
  process's exit"). A STATUS write of 0 and `rvg_graphics_device_reset` wait only for the command the device's control
  thread is taking, and for a take of a queue's ring or a post onto one under way, none of which
  waits for anything of the host's. `rvg_graphics_device_read_display` and `_read_display_at`
  wait for their own draw on the GPU, on a queue of its own, and one read of each display runs at
  a time.

### What may be called inside a callback

| inside | may call |
|---|---|
| `raise_interrupt` | `rvg_status_message` and the `rvg_graphics_device_*` functions, except the release of its own device |
| `route_doorbell`, `route_doorbell_page` | nothing |
| `cursor_glyph`, `cursor_show`, `cursor_move` and their `_at` forms | every function: the library never waits for the display thread |
| a window callback (`pointer`, `wheel`, `key`, `close`, `scroll`) | every function; it must not block (section 5) |
| a `destroy` | nothing |

### Deaths

The process ends at the operation, with a line on stderr (`reims-vgpu: fatal: <what died>`) and
then `abort()`, for:
- a NULL handle to a function that returns no status;
- anything a host callback throws (`ffi::host_call` runs each call inside a catch);
- a panic under an export: the library's panic hook ends the process before anything unwinds;
- a guest command the platform cannot carry out as Metal does, where Metal ends the process: a
  raise of Metal's that nothing catches, the platform's own raises where Metal's call raises
  among them, or a raw value the platform has no form for (the Vulkan crate's `answer`,
  [accuracy-ledger.md](accuracy-ledger.md)). What the platform has not built, or what rt refuses,
  is answered as Metal's call answers where it fails, and counted, never a death.

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

The library keeps the GPU device it opens for the process, with what it makes of it for itself,
while any graphics device lives, and a device created after another was released runs on the same
one. The release of the process's last graphics device lets go of it
(`RtPlatform::let_go_of_device`): it waits, at most 5 s, for that device's threads, its objects and
their GPU work to let go of the GPU device, so that nothing beyond the platform holds it; then runs
rt's `Device::quiesce`, with what is left of the 5 s, drops its last handle, which tears it down on
the releasing thread before the release returns, and waits for rt's gauges of it to read zero. A
device whose quiesce answered false is kept to the exit, since its drop would wait with no deadline.
So no GPU device is live as the process exits: on NVIDIA's proprietary driver an exit with one live
can end with `SIGSEGV` inside the driver's own exit handler, which tears its SPIR-V compiler down
while a worker thread of the driver's still runs in it (on an RTX 3090, 6 of 80 exits right after
the device's check of textures over host pages, none with the device let go of first). A device
callback still running holds that wait up: one that waits for a lock the host holds around the
release, for the whole 5 s. Past them the library keeps its GPU device for the process, as it does
while a device lives, and logs a warning saying what still held it. A device created after the
let-go runs on a new GPU device, whose numbering starts again. QEMU releases its device as it quits,
once the VM stopped for good (its `vm_shutdown`), with the BQL let go for the release alone; a VM
whose shutdown action pauses it keeps its device to the exit, and so does a quit while a read of its
display (`rvg_graphics_device_read_display`) still waits for its draw on the GPU 5 s into the
teardown.

A host quiesces the library as its last call before it exits: `rvg_quiesce_for_exit`, on the thread
that then exits. It takes no handle, since the GPU devices are the process's. Its answer is whether
the host may then call `exit`. It closes the exit gate ahead of the exit, as the handler would, so
that each thread of the library's parks for good at its next step, and waits for the steps in
progress on other threads. Then, where a GPU device is still live (a device kept to the exit, or one
a let-go could not tear down), the runtime's `Device::quiesce` stops its threads: what is ready is
submitted and the rest ends unrun, the GPU completes what was submitted, and the runtime's threads
end. Every wait ends within the call's bound, and nothing is freed. It answers true where the steps
ended in time and the library holds no GPU device the exit could meet: none was opened in the
process, or the release of the last graphics device let go of it. It answers false otherwise,
quiesced or not, and where a step outlasted the wait, since metal2vulkan-runtime supports an `exit`
with a device live on no device (its README, "Quiesce and exit"): on an RTX 3090 (620.70.06), 3 of 9
power-offs that called `exit` after the quiesce stopped everything in time ended with `SIGSEGV` in a
thread of the driver's own while its exit handler waited on a lock. On false the library writes its
degradation counts then, as the exit gate's handler would, and the host runs its own exit work and
ends with `_exit`, running no exit handler, the runtime's and the driver's included. A second call
on the same thread answers at once, with the first answer; after it, a call on any other thread
parks that thread for good, and rt refuses GPU work asked for on this one (`rt-refused-quiesced` in
the degradation counts). After a true answer the exit gate's handler still runs at the exit, and
writes the counts.

QEMU calls it, with a bound of 5 s, at the end of its cleanup (`qemu_cleanup`, through a notifier of
its own run there, `qemu_add_cleanup_notifier`), on every quit that reaches it: a released device,
a device kept by a shutdown action that pauses, a quit of a stopped VM, and a read of a display
that kept it. That runs before `exit` itself, so before every exit handler. It lets the BQL go
around the call, as around the release. On a false answer it runs what `exit` would run of QEMU's
own: its exit notifiers (usb-host's return of host devices to their drivers, a tap's down script,
the pidfile's removal), the terminal's restore where a chardev took stdio, and the flush of the C
library's streams; then it ends with `_exit` and the cleanup's status. It does not run the simple
trace backend's flush, record and replay's finish or the plugins' exit callbacks, which QEMU as
reims-ctl runs it does not use. An `exit` that skips the cleanup (a refused realize of a later
device, a failed incoming migration) reaches a handler registered as the device is made, after the
driver loaded. No release let go of the GPU device then, and the runtime supports no quiesce inside
an exit handler, so the handler calls none: it runs the same exit work and ends with
`_exit(EXIT_FAILURE)`, and the library writes no degradation counts. After the cleanup's true
answer it does nothing. QEMU's trace event `reims_graphics_end` names each end through `_exit`.

### The guest's Shared pool

A host names the guest physical addresses it can give the guest's Shared pool
(`pool_physical_address`, `pool_physical_length`), 2 MiB-aligned and with no RAM under them, 0 and 0
for none. The device takes from their start as many as the memory heaps of its GPU device hold,
every heap summed, rounded up to 2 MiB. Its Shared blocks come from the heap of the first memory
type with `HOST_VISIBLE`, `HOST_COHERENT` and `HOST_CACHED` that a buffer of the device's usage can
use, but a heap's size is how the driver accounts for memory, not the most it allocates: on an AMD
APU, RADV splits the memory the GPU can use into that type's heap, a third of it (5.52 of 16.57 GiB
on a 780M, measured 11-18 with `vulkaninfo`), and a device-local heap over the same memory, and the
runtime takes Shared blocks until the driver refuses one. The bound is the heaps' sizes, not
`VK_EXT_memory_budget`'s budgets: the range is address space the guest keeps for the VM's life, a
budget moves with what other processes hold, and RAM is bounded at each block's take. Where the
heaps hold more than the addresses given, the range stops at their end and the device logs a warning
once; where the GPU answers no heap, the guest has no pool, and the device logs that too. The device
logs the range it took, the addresses it was given and the heaps' total once, at its creation.
`rvg_graphics_device_pool_range` answers the range it took, 0 and 0 for none, and the host reserves
that before the guest's firmware runs. The host then maps
the device's blocks there as the pool grows
(`pool_add`, answered by `rvg_graphics_device_pool_added`) and unmaps them as the device gives them
back to its block source (`pool_remove`, answered by `rvg_graphics_device_pool_removed`). The device
gives a block back, and the source may then free it, only once no thread of its own holds it and the
host said its piece is out of the guest's address space and no thread of the host's touches it; a
block the source hands out again is mapped anew. Before the release the host takes out every piece
it added and says so for each, as QEMU's teardown does; the release then waits for the device's own
holds and frees every block before it returns, so none is freed by a driver's exit handler (on
NVIDIA's driver, which unmaps its memory inside `exit()`, a thread still reading a block then
faults). QEMU gives the device the addresses past the RAM and the PCI host bridge's default 64-bit
window, up to three quarters of the guest's physical address space and, under KVM and WHPX, at most
to the end of the host's, and at realize, once the device is made, raises fw_cfg
`etc/reserved-memory-end` past the range it took, rounded up to 1 GiB, so OVMF's 64-bit window goes
above it, maps each piece as a RAM region of its own (an add is a pure add), and under WHPX runs
the hypervisor's map outside its BQL (`whpx_map_ahead`).

On the arm64 guests' machine, `vmapple`, which has no fw_cfg and places nothing past its RAM, QEMU
gives the device the addresses from the end of the RAM, rounded up to 1 GiB, to the end of the
guest's physical address space (40 bits under HVF where the host gives 40), in whole GiB, and
reserves nothing.

## 2. The C header

The header has every function, type, callback typedef and constant, with the thread and
ordering rules on each callback and each function. Its preamble states the rules for handles,
configs, callbacks and deaths. The C API's tests compile it as C11, as C++17 and, on macOS, as
Objective-C.

| group | items |
|---|---|
| constants | the status codes; the PCI facts (`RVG_GRAPHICS_PCI_VENDOR_ID` 0x1234, `_DEVICE_ID` 0x5247, `_CLASS` 0x380, `RVG_GRAPHICS_PCI_BAR_REGISTERS` 0, `RVG_GRAPHICS_REGISTERS_SIZE` 4096, `RVG_GRAPHICS_MSIX_OFFSET` 2048, `RVG_GRAPHICS_MSIX_VECTORS` 1, `RVG_GRAPHICS_PCI_BAR_REGION` 2, `RVG_GRAPHICS_PCI_BAR_DOORBELLS` 4, `RVG_GRAPHICS_DOORBELLS_SIZE` 4 MiB, `RVG_GRAPHICS_DOORBELL_PAGE_SIZE` 16 KiB, `RVG_GRAPHICS_DOORBELL_PAGES` 256); `RVG_GRAPHICS_DISPLAYS_MAX` 4; the doorbell (`RVG_GRAPHICS_REG_DOORBELL`, `RVG_GRAPHICS_DOORBELL_CONTROL`); `RVG_GRAPHICS_ALLOCATIONS_OFFSET`; `RVG_GUEST_ARCH_X86_64` 0 and `RVG_GUEST_ARCH_ARM64` 1; `RVG_WINDOW_BUTTON_*`, `RVG_WINDOW_SCROLL_PHASE_*`, `RVG_WINDOW_SCROLL_MOMENTUM_*` |
| status | `rvg_status_message` |
| the process's exit | `rvg_quiesce_for_exit` |
| the device | `RvgGraphicsDeviceConfig`, `RvgGuestArch`, `RvgGraphicsCallbacks` and its function types, `RvgCursorGlyph`, `RvgDisplayCoord`, `RvgMemoryMap`, `RvgMemoryRange`, `RvgMemoryRangeFile`; `rvg_graphics_device_create`, `_release`, `_region`, `_mmio_read`, `_mmio_write_taking`, `_doorbell_write`, `_set_doorbell`, `_reset`, `_read_display`, `_cursor_position`, `_read_display_at`, `_cursor_position_at`, `_pool_added`, `_pool_removed`, `_pool_grow`, `_pool_range` |
| its windows | `RvgWindowConfig`, `RvgWindowCallbacks`, `RvgWindowPointerFn`, `RvgWindowWheelFn`, `RvgWindowKeyFn`, `RvgWindowCloseFn`, `RvgWindowScrollFn`, `RvgWindowScroll`; no function of their own: `RvgGraphicsDeviceConfig.window` carries display 0's, and `display_windows` those of displays 1 and up |

### Windows forms

Two items take the host's OS handles, and Windows has its own form of each, under
`#if defined(_WIN32)`. Linux and macOS keep theirs, under `#if !defined(_WIN32)`. cbindgen's
`[defines]` maps the crate's `cfg(windows)` to `_WIN32`.
- **`RvgMemoryRangeFile`** names a section where Linux and macOS name a file:
  `{void *handle; uint32_t flags; uint64_t offset;}`, 24 bytes, NULL for a range with none. The
  range's host memory is a view of the pagefile-backed section `handle` names, from `offset`. The
  create checks that a view the library may write maps the range's bytes from `offset`
  (`MapViewOfFile` with `FILE_MAP_WRITE`, unmapped at once), where Linux and macOS check the
  file's size with `fstat`. It then duplicates the handle (`DuplicateHandle`).
- **`rvg_graphics_device_set_doorbell(device, void *event)`** takes an event's handle where Linux
  and macOS take a file. On Windows, QEMU's `EventNotifier` is a manual-reset event, and its memory
  core sets it for a write that matches a `memory_region_add_eventfd`, since only KVM matches
  those in the kernel. The device duplicates the handle, waits for the event, and resets it when
  it wakes.
- **The route functions** (`route_doorbell`, `route_doorbell_page`) keep their form. A device on
  Windows calls neither: with no kernel `ioeventfd` on the host, it makes no bell to route a
  queue's doorbell into.

### Settings and the environment

Everything a host can set that shapes what the library does, beside the guest's own input:
- **The config**: the device's callbacks, how many displays it has (1 to 4), each display's window
  (its callbacks and the mode it opens at),
  the guest's RAM with the files behind it, through which the GPU reaches an app's no-copy memory
  in the guest's own pages and, on a host GPU that shares the CPU's memory, set the device's
  `recommendedMaxWorkingSetSize` (three quarters of that RAM, at most the host GPU's answer), and the guest's architecture,
  which the device's `isFramebufferReadSupported` follows: the host GPU's answer for an arm64
  guest, no for an x86-64 one. Zero is every field's default (section 1): an x86-64 guest.
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
  | `REIMS_VGPU_STRICT` | the Vulkan platform: `1` ends the process where the host runs out of memory, mappings, files or commit charge, where by default the platform answers as Metal's call answers when memory runs out (the Vulkan crate's `answer`) |
  | `REIMS_VGPU_TRACE` | the Vulkan platform, a diagnostic for profiling: a timeline of the device's work, a line per event, written to the file it names (core's `trace`) |
  | `REIMS_VGPU_RECORD` | the Vulkan platform, a diagnostic for replay: every call core makes of the platform, the guest's presents as the display shows them, each device's identity as the stream first names it and again at the start of each later segment (`m2v.deviceIdentity` on the device, answering its `name` and `hasUnifiedMemory`), the host memory each command buffer reaches as it changes, the bytes a no-copy `Managed` buffer's create copies into the device's own copy where the device keeps two (its `hasUnifiedMemory` NO), each no-copy buffer's escape as the guest first notes it (`m2v.bufferEscaped` on the buffer, `[kind]`, the note's `RGFX_BUFFER_ESCAPED_*` act, after the memory writes of each command buffer committed before the note and before those of each committed after it, where a player stops, typed, at a later write that holds an address in the buffer), and each acquire of what another API wrote into a buffer's exported memory (`m2v.externalWriteDone` on the buffer, no arguments, as the call returns: the host's video once a picture is in a surface's device cache, before the session's signal that lets a command buffer read it), written as a replay stream (metal2vulkan's format version 2) into the directory it names, the libraries and those bytes into a `blobs` directory beside it, and its counts every few seconds into `<pid>.stats.txt`, which names the recorder's build first: the commit the library was built from (`recorder_commit`, which `cargo xtask dist` stamps, else `unknown`), the device's own display library, which the stream records as `newLibraryWithSource:options:error:` on the device with no arguments, by the hex of its SHA-256 and its length (`display_library`), and a `recorder_fix` line for each change to what a stream holds that it carries (the crate's `recorder`). The stream holds the guest's data: it stays on the host that made it |
  | `REIMS_VGPU_REFUSED_AIR_DIR` | the Vulkan platform, a diagnostic: for each pipeline state the translator refused, the metallib of each of its stages' functions, once per library, a line naming the creator, the functions and the refusal, and a record of the state's function-constant values and descriptor fields, written into `air-<pid>` in the directory it names (`rt_platform::diag`). It holds the guest's data: it stays on the host that made it, and is never written to the log |

  Of these, only `REIMS_VGPU_STRICT` changes what the guest or host observes, beside
  `REIMS_VGPU_WINDOW_SCREENS` and `_REFRESH`, which change the mode the window's display offers its guest.

### cbindgen

`crates/reims-vgpu-c/cbindgen.toml`: language C, `c99` doc style, `cpp_compat`,
`sort_by = "None"`, `parse_deps = false`; its `header` preamble states the rules; functions come in
the order cbindgen parses them, the order of `lib.rs`'s public modules. On the Rust side the codes
are `pub type RvgStatus = u32` and `pub const` items, so a host's out-of-range value is never a
Rust enum; the handle is a Rust struct cbindgen emits as opaque; the configs are `#[repr(C)]`.

## 3. The Rust API (`reims-vgpu`)

`reims-vgpu` is the device; the C API wraps it one to one. Its rustdoc is the reference:
- `GraphicsDevice<P = DefaultPlatform>`, made by `GraphicsDevice::new(GraphicsConfig)` on the
  default platform (`RtPlatform`, over metal2vulkan-runtime) or `with_platform`, so its
  forwarding is unit-tested on core's fake platform;
- `GraphicsConfig`: `GraphicsCallbacks` (closures, `Arc<dyn Fn … + Send + Sync>`, each optional),
  an optional `WindowConfig`, the display count and the windows of displays 1 and up, an optional
  `MemoryMap` with its `MemoryFile`s, and the guest's `GuestArch` (`X86_64`, the default, or
  `Arm64`); `GraphicsDevice::read_display_at` and `cursor_position_at` name a display, and
  `display_count` answers how many there are;
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
| `rvg_quiesce_for_exit` | never | the library's calls in progress on other threads, and the GPU's work, until its bound |
| `rvg_graphics_device_region` | never | nothing |
| `rvg_graphics_device_mmio_read` | never | nothing |
| `rvg_graphics_device_mmio_write_taking` | `raise_interrupt`, for a doorbell answered on the calling thread; `route_doorbell` and `route_doorbell_page`, dropping routes, for a STATUS write of 0, and `route_doorbell_page`, dropping a page's routes, for a `DOORBELL_PAGE_BIND` write that binds the page to another context | a STATUS write of 0: the command the control thread is taking; a `DOORBELL_PAGE_BIND` write: the queues' table, held elsewhere |
| `rvg_graphics_device_doorbell_write` | `raise_interrupt`, for a doorbell answered on the calling thread | the queues' table, held elsewhere |
| `rvg_graphics_device_set_doorbell` | never | nothing |
| `rvg_graphics_device_reset` | `route_doorbell` and `route_doorbell_page`, dropping routes | the command the control thread is taking, and a ring's take or post under way |
| `rvg_graphics_device_read_display` | never | its draw on the GPU |
| `rvg_graphics_device_cursor_position` | never | nothing |
| `rvg_graphics_device_read_display_at` | never | its draw on the GPU |
| `rvg_graphics_device_cursor_position_at` | never | nothing |
| `rvg_graphics_device_pool_added` | never | nothing |
| `rvg_graphics_device_pool_removed` | never | nothing |
| `rvg_graphics_device_pool_grow` | never | nothing |
| `rvg_graphics_device_pool_range` | never | nothing |

## 5. Callbacks and threads

### Where each callback arrives

| callback | thread | serialized with | order |
|---|---|---|---|
| `raise_interrupt` | the thread that posts the event it raises for: the device's control thread, `reims-vgpu.graphics-control`; one of ring 0's threads, `reims-vgpu.graphics-ring0-<i>`, which carry out a guest process's control messages, its presents and a block's pages and downloads, each on its lane; a display's refresh thread, `reims-vgpu.graphics-refresh` (display 0's) and, on a device of several displays, `reims-vgpu.graphics-refresh-<i>`; a command queue's service thread, `reims-vgpu.graphics-queue-<id>`, or one of its lane's threads, which walk and commit its command buffers, `reims-vgpu.graphics-lane-<id>`; a pipeline compile's thread, `reims-vgpu.graphics-compile-<n>`; a thread that makes IOSurfaces' and pages blocks' views, `reims-vgpu.graphics-surfaces-<i>`; a thread that makes indirect command buffers, `reims-vgpu.icb-create-<i>`; a thread that lets go of a closed process's objects and answers its close, `reims-vgpu.graphics-closes-<i>`; a thread that moves a staged copy of a surface's device cache into its pages and answers what waited for it, `reims-vgpu.graphics-into-pages-<i>`; a video session's picture landing thread, which posts a decode's completion as the decode is done, `reims-vgpu.graphics-video-land`; or the platform's thread that runs a command buffer's scheduled or completed handler; or the host's own, inside `rvg_graphics_device_mmio_write_taking` or `rvg_graphics_device_doorbell_write`. Never inside another call of the host's | nothing: they may call at once | each after the completion ring's message it raises for is published |
| `route_doorbell`, `route_doorbell_page` (the route of a queue whose process has a doorbell page, at that page's doorbell in BAR 4) | one of ring 0's threads, `reims-vgpu.graphics-ring0-<i>`, or the control thread, `reims-vgpu.graphics-control`, as a command queue is made, destroyed or its process closed, with no lock of the library's held, where it may wait for a lock the host holds while it calls the device (QEMU's BQL), since no function the host calls under it waits for a thread inside the callback but the release of the process's last device, at most 5 s; dropping a route, also the host's own inside `rvg_graphics_device_mmio_write_taking` (a STATUS write of 0, or a `DOORBELL_PAGE_BIND` write that binds a page to another context) or `rvg_graphics_device_reset`, where it must not wait for a lock the host holds around that call | nothing: they may call at once | a queue's route before its create is answered, and its drop once its queue has ended and before its destroy, its process's close or the reset is answered; a dropped route takes no write once the call returns |
| `cursor_glyph`, `cursor_show`, `cursor_move` (display 0's), and `cursor_glyph_at`, `cursor_show_at`, `cursor_move_at` (each display's from 1, named) | the display thread: one serial worker for the whole process (after 5 s idle it exits, and the next callback starts a new one) | every display callback of the process: one at a time | the order the device's control thread took the guest's cursor and display power commands, of every display, and its resets |
| `pool_add` | the device's pool thread, `reims-vgpu.graphics-pool`, with no lock of the library's held; never inside a call of the host's; it must not wait for a lock the host holds while it calls the device (QEMU queues the add for its main loop) | one at a time | the order the device places the pieces, each before its `pool_remove`; the host answers each with `rvg_graphics_device_pool_added` once the guest's address space has it |
| `pool_remove` | the device's pool thread, `reims-vgpu.graphics-pool`, before it gives the piece's block back to its block source, with no lock of the library's held; never inside a call of the host's; it must not wait for a lock the host holds while it calls the device | nothing | once a piece, after its `pool_add`, never once the release has begun; the host answers with `rvg_graphics_device_pool_removed` once the piece is out of the guest's address space and no thread of its own touches it, and the device gives the block back, which the source may free, only then |
| a window's `pointer`, `wheel`, `key`, `close`, `scroll` | on macOS the process's main thread, where AppKit runs; on Linux and Windows the library's thread `reims-vgpu.ui`; never inside a call of the host's | every window callback of the process: one at a time | the order the events arrived |
| `destroy` (each table) | the device's: once its threads have ended, after their last callback, possibly on one of them after the release has returned; the window's: once the release has closed the window, on the releasing thread or, while a window callback runs, the window's thread once it returns; either, on the caller's thread, in a refused create | nothing | once, after every call of its table |

Threads are named `reims-vgpu.<role>`: those above, `reims-vgpu.display` (the display thread),
`reims-vgpu.window` (a window's drawing thread), `reims-vgpu.ui` (on Linux and Windows, the thread
every window's events and callbacks run on), and `reims-vgpu.graphics-video-<i>`,
`reims-vgpu.graphics-let-go-<i>` and `reims-vgpu.graphics-event-waits` (a graphics device's). The names are diagnostics, not contract.
A queue's worker, the display thread among them, has core's `WORKER_STACK`, 512 KiB.

### The locks around `raise_interrupt`

The device's callbacks may wait for a lock the host holds while it calls the device (QEMU's BQL):
no function of the device waits for its threads, but the release of the process's last device,
which waits for them at most 5 s (section 1, "The process's exit"), and the library calls them with
no lock held that one of its functions waits for, directly or through a thread of the library's. `raise_interrupt`
may run under these:
- a command queue's claim on its next message, on the queue's service thread and inside
  `rvg_graphics_device_mmio_write_taking`;
- there, at a command buffer's `CB_SUBMITTED` answered on the calling thread, the decoder's
  contexts' lock, the buffer's context's table read, its queue's answer slot and the reset gate
  read;
- the decoder's lock, with the download's answer slot, at a `SURFACE_DOWNLOAD` answered at once;
- a command buffer's `CB_SUBMITTED` latch, wherever that is posted;
- on a picture landing thread, the run of the video processing that copies a decoded picture into
  its output's device caches, at the decode's completion posted as the decode is done;
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
runs, and on Linux and Windows on the library's own thread `reims-vgpu.ui`. On Linux and Windows
nothing of the library runs on or waits for the process's main thread, and what follows about the
run loop and the application is macOS's.
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
  `rvg_graphics_device_mmio_write_taking` or `rvg_graphics_device_doorbell_write` must not wait for
  a lock the host holds around that call.
  One that waits for a lock the host holds around the release of the process's last device holds
  that release up for 5 s, and the GPU device then stays for the process (section 1).
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
