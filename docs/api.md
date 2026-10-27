# The API

This is the library's API, as a host uses it:
- the C API (`crates/reims-vgpu-c`), whose header cbindgen generates from the crate;
- the Rust API it wraps (`crates/reims-vgpu`);
- how QEMU's `apple-gfx` and the suite's runner use them.

The header, `crates/reims-vgpu-c/include/reims_vgpu.h`, is the reference. Sections 1 and 5 state the rules, and the header repeats each rule where it applies.

The C API has no Objective-C types, dispatch queues or blocks, and neither API takes a Metal object from the host: a config names a Metal device by its registry ID. Callbacks arrive on the library's threads, and a display window's on its window's thread: the process's main thread on macOS, the library's `reims-vgpu.ui` on Linux. Frames are copied into host memory, or, for a display created with a window, drawn into a window of the library's own. Suspend and resume are public, and the device config carries the feature level and the other settings that shape what the guest is offered (section 2, "Settings and the environment").

## 1. Principles and versioning

### The layers

| layer | crate | what it is |
|---|---|---|
| core | `reims-vgpu-core` | the device, generic over `Platform` |
| platform | `reims-vgpu-metal`, `reims-vgpu-vulkan` | core's `Platform`: `MetalPlatform`, over Metal, IOSurface and the OS, on macOS; `VulkanPlatform`, over metal2vulkan's runtime, on Linux, and on macOS with the `vulkan` features |
| window | `reims-vgpu-window` | a display's own window (AppKit's or SDL3's on macOS, SDL3's on Linux), its input and the host's screens; it knows nothing of devices or displays |
| Rust API | `reims-vgpu` | owned Rust types over core: configs, closures, `Result`; a display's window over the window crate |
| C API | `reims-vgpu-c` | a thin FFI layer over the Rust API; cbindgen writes `include/reims_vgpu.h` from it |

The C layer converts and forwards; it decides nothing. Every callback it builds calls the host's function synchronously, on the thread its caller calls it on: core, for the device's, the display's and the IOSurface device's callbacks, and the window crate, on the window's thread, for a window's. So the order and the threads of every call to the host are core's, and a window's are the window's.

### Objects

Each object is made from one config struct, which is borrowed for the call. The library copies what it keeps. There are no descriptor objects and no setters.

| object | made by | released by | keeps alive | its end |
|---|---|---|---|---|
| `RvgDevice` | `rvg_device_create` | `rvg_device_release` | — | teardown at the last release of the device and of what keeps it alive; it may block |
| `RvgDisplay` | `rvg_display_create` | `rvg_display_release` | its device | the unplug, at its last release |
| `RvgFrameTarget` | `rvg_frame_target_create` | `rvg_frame_target_release` (with `_retain`) | its Metal texture only | at its last reference |
| `RvgFrame` | `rvg_display_encode_frame` | `rvg_frame_release` | its display and its target | when it completes, or when released uncommitted |
| `RvgIOSurfaceDevice` | `rvg_iosurface_device_create` | `rvg_iosurface_device_release` | — | teardown at its last release; it waits for its own thread |
| `RvgVideoDevice` | `rvg_video_device_create` | `rvg_video_device_release` | — (its IOSurface device is borrowed: unretained) | a reset at its last release; it waits for its host session's threads |
| `RvgScalerDevice` | `rvg_scaler_device_create` | `rvg_scaler_device_release` | — (its IOSurface device is borrowed: unretained) | its accelerator goes at its last release |
| `RvgTraceRangeHandler` | the device, handed to `add_trace_range` | `rvg_trace_range_handler_release` (with `_retain`) | the EFI display's present source | at its last reference |
| `RvgBytes` | `rvg_device_finish_suspend` | `rvg_bytes_free` | — | — |

- **No retain on devices and displays.** The host owns exactly one reference to each, as QEMU and the runner use them. Frame targets and trace handlers have a retain, because hosts hand them between threads.
- **A display keeps its device alive.** A frame keeps its display and its target alive until the frame completes, or until it is released uncommitted (the pins `api.lifetime.display_through_*`). A frame target may outlive the device it was made on.
- **A display's window.** A display created with a window (`RvgDisplayConfig.window`) closes it at its last release, before it lets the display go, and waits neither for the main thread (on macOS) or the library's `reims-vgpu.ui` thread (on Linux) nor for the window's frames. The window holds the display only for a frame it has in flight, which keeps the display alive until it completes, as an `RvgFrame` does: a release that races one ends the display, and runs its table's `destroy`, on a library thread after the release has returned (the thread that drew the frame, or Metal's completion thread; the unit test `window::tests::a_window_frame_keeps_its_display_until_it_completes` of `reims-vgpu`).
- **NULL.** A release of NULL does nothing. Elsewhere a NULL handle is `RVG_ERR_INVALID_ARGUMENT` where the function returns an `RvgStatus`, and ends the process where it does not. A host answers for a handle it has released itself, as the runner does for its released displays (section 7).

### Configs and `struct_size`

Every struct the host fills in, and every struct the library hands out, starts with `uint32_t struct_size`.

- **A struct the host writes:**
  - `struct_size` below the first version's size is `RVG_ERR_INVALID_ARGUMENT`;
  - fields past `struct_size` read as zero;
  - a `struct_size` above the library's is accepted when every byte past the library's size is zero, and is `RVG_ERR_UNSUPPORTED` otherwise. A host built against a newer header then works with an older library as long as it leaves the new fields at zero.
- **A struct that has grown** keeps its first version's size as the least it accepts, and a host built against an earlier version passes that version's size and is read as it, the later fields zero. Four have grown:
  - `RvgDeviceConfig`, at its third version, 64 bytes. Its first ends at `callbacks`: 48 bytes, the offset of `memory_files`, which came after. A host built before `memory_files` passes 48 and is read with no files, whatever follows its config in memory (the unit tests `device::tests::a_first_version_device_config_is_48_bytes` and `a_48_byte_device_config_is_accepted`). Its second ends at `memory_files`, 56 bytes, the offset of `process_isolation_follows_guest`; a host built before that field passes 56 and is read with it false (`device::tests::isolation_following_the_guest_is_accepted`).
  - `RvgDisplayConfig`, at its second version. Its first ends at `callbacks`: 56 bytes, the offset of `window`, which came after. A host built before `window` passes 56 and is read, with no window (the unit tests `window::tests::a_first_version_display_config_is_56_bytes` and `a_56_byte_display_config_is_accepted`).
  - `RvgWindowConfig`, at its third version, 32 bytes. Its first ends at `callbacks`, 16 bytes, and a host built against it gets no mode; its second ends at `mode_height`, 24 bytes with its padding, and a host built against it gets no density (the unit tests `window::tests::a_first_version_window_config_is_16_bytes_and_has_no_mode` and `a_second_version_window_config_is_24_bytes_and_has_no_density`).
  - `RvgWindowCallbacks`, at its second version, 64 bytes. Its first ends at `close`: 56 bytes, the offset of `scroll`, which came after. A host built against it passes 56 and is read with no `scroll`, whatever follows its table in memory (the unit test `window::tests::a_first_version_window_callbacks_is_56_bytes_and_has_no_scroll`), so its scrolling comes as notches through `wheel`.
  - Every other struct is still at its first version.
- **A struct the library writes** (`RvgCursorGlyph`, `RvgWindowScroll`): the host reads a field only if `struct_size` covers it.
- **Zero is every field's default.** A setting that is on by default is named for turning it off: `disable_argument_buffers`. The exception is `display_port_count`, where 0 is refused.
- **Plain data needs no size**: `RvgMemoryMap`, `RvgMemoryRange`, `RvgMemoryRangeFile`, `RvgDisplayMode`, `RvgRegion`, `RvgPhysicalMemoryRange`, `RvgBytes` and `RvgDisplayCoord` are fixed. `RvgDisplayMode.flags` has room for more bits, and `RvgMemoryRangeFile.flags`, 0 today, room for more.
- **Versioning.** Changes are additive:
  - new functions;
  - new fields at the end of a sized struct;
  - new constants and status codes.

  No function, field, constant or status value is renamed, renumbered or removed. A host must treat an unknown status as a failure.

### Status codes

Functions that can fail return `RvgStatus` (a `uint32_t`). `rvg_status_message` gives a static description.

| value | name | when |
|---|---|---|
| 0 | `RVG_OK` | |
| 1 | `RVG_ERR_INVALID_ARGUMENT` | a NULL handle, config or out-pointer; a `struct_size` too small; a value out of range (frame target size or format, `bytes_per_row`, the destination's length); a refused memory file (`RvgDeviceConfig.memory_files`); `add_trace_range` without `remove_trace_range`; a frame target on another Metal device |
| 2 | `RVG_ERR_UNSUPPORTED` | a larger `struct_size` with non-zero bytes the library does not know |
| 3 | `RVG_ERR_DISPLAY_PORT_COUNT` | `display_port_count` 0, or above 8 |
| 4 | `RVG_ERR_NO_METAL_DEVICE` | no Metal device with that registry ID, or none at all |
| 5 | `RVG_ERR_MEMORY_CONFLICT` | a memory map together with a task or memory callback |
| 6 | `RVG_ERR_MEMORY_MAP` | the memory map refused (overlap, overflow) |
| 7 | `RVG_ERR_ISOLATION_WITHOUT_MAP` | `process_isolation` without a memory map |
| 8 | `RVG_ERR_METAL` | Metal refused an object: the device's blit pipelines, the display's pipeline cache, a frame target's texture, or a frame's command buffer or the frame queue it comes from |
| 9 | `RVG_ERR_PORT_OUT_OF_RANGE` | the display's port is not below the port count |
| 10 | `RVG_ERR_PORT_IN_USE` | another display holds the port |
| 11 | `RVG_ERR_SERIAL_IN_USE` | a display in the process holds the serial number |
| 12 | `RVG_ERR_NO_MEMORY_MAP` | an IOSurface device without a memory map |
| 13 | `RVG_ERR_SUSPEND` | never returned: `finish_suspend` always has a state to give |
| 14 | `RVG_ERR_RESUME` | `will_resume` refused the state |
| 15 | `RVG_ERR_GPU` | the frame's GPU work ended with an error |
| 16 | `RVG_ERR_FRAME_STATE` | a frame committed twice, or waited on before its commit |

Each create checks in this order, which is core's:

- **Device:** port count, then the Metal device, then the memory conflict, then the memory map, then the trace pair, then the blit pipelines, then isolation.
- **Display:** port range, then the pipeline cache, then the serial number, then the port.
- **IOSurface device:** memory map present, then the memory map valid.

The host-struct checks (`INVALID_ARGUMENT`, `UNSUPPORTED`) come first and touch nothing. For a device they read, in order, the config, its callbacks, the out-pointer, the memory map, and the files behind the map's ranges. The files are refused when given without a map, then, range by range from the first, for an `fd` below -1, for `flags` other than 0, and, for an `fd` of 0 or more, for a file `fstat` cannot read, a file holding fewer than `offset` plus the range's length bytes, and an `offset` that is not a multiple of the host's page size. Each refusal logs a line saying which (the pin `api.device.memory_files`), and no file is duplicated until every entry has passed. A file of huge pages (hugetlbfs) passes them on Linux, and no view of a page of the host's size can be made of it, so its range is as one with no file: every guest use that needs a view of it fails as a failed remap does (QEMU's `apple-gfx` gives such RAM no file). For a display they read, in order, the config, its callbacks, its window config and the window's callbacks.

The resume reasons are the codes 0 to 5 (`RVG_RESUME_ERROR_*`), with `RVG_RESUME_ERROR_NONE` (`UINT32_MAX`) for "none stored". A refused resume need not store one.

### Thread safety

- Every function may be called from any thread, on the same handle from several threads at once. A handle must not be used during or after its release.
- **"May call back"**: a function marked so in the header may run the device's callbacks on the calling thread before it returns (section 5). The queries and every frame function never call back. Section 4 lists the marks.
- **Waiting.** A function that may call back may also wait for the library's FIFO and worker threads, where the device's callbacks run. The header marks each that may block, and says what it waits for:
  - the releases, for the teardown, and the MMIO accesses: a root FIFO write waits for the FIFO's thread;
  - `stop`, for each FIFO's thread to stop and for the commands it started to complete;
  - `reset`, as `stop` does, and for the EFI display's present thread;
  - `pause`, as `stop` does, and it hangs while a command never completes;
  - `unpause`, while the FIFOs start on the library's FIFO control thread;
  - `finish_suspend`, as `stop` does, then it flushes and ends the guest's tasks;
  - `will_resume`, for the EFI display's present thread, when the state's binary version has no EFI display;
  - `did_resume`, for the EFI display's present thread, and as `stop` does for a root the device still has.

  `will_suspend` quiesces the FIFOs without waiting for them. `rvg_frame_wait` waits for the GPU. The IOSurface device's MMIO write may wait for its thread, and its release does. The video device's `deliver` may wait for the IOSurface device's thread, a JPEG decode and a session's end, 10 s at most each; its `invalidate` waits for its sessions' threads to take the invalidation, 10 s at most in all; and its `reset`, its last release and a `start` that replaces a session wait as `invalidate` does, then for those threads to end. The scaler device's `process` may wait for the IOSurface device's thread and the GPU. No function waits for the display thread, and none waits for the main thread: a display with a window is created and released in a process whose main thread never runs its run loop (the pin `api.window.no_run_loop`).

### What may be called inside a callback

| inside | may call |
|---|---|
| a task or memory callback (`create_task` … `read_memory`) | `rvg_status_message` |
| `raise_interrupt` | `rvg_status_message` |
| `add_trace_range` | `rvg_trace_range_handler_retain`, `rvg_status_message` |
| `remove_trace_range` | `rvg_trace_range_handler_release`, `rvg_status_message` |
| a display callback | every function: the library never waits for the display thread |
| a window callback (`pointer`, `wheel`, `key`, `close`, `scroll`) | every function; it must not block (section 5) |
| the IOSurface `raise_interrupt` | every function except releasing its own IOSurface device, which ends the process |
| the video `send_to_guest` | `rvg_status_message` |
| the scaler's `is_ram`, `read`, `write` | nothing |
| the log handler | nothing |
| a `destroy` | nothing |

A call into the same device from a device callback may deadlock, or end the process at an assertion. The library does not check for it.

### Deaths

The process ends at the operation, with a line on stderr and then `abort()`, for:
- invalid guest input;
- the device's first interrupt without `raise_interrupt`, and the IOSurface device's first interrupt without its own;
- the EFI display's first present to a display without `new_frame` (a guest's present completes with no call);
- a second `rvg_device_pause` before `rvg_device_unpause`, and an unpause while the guest's displays are not paused, where the guest has given a port a display page (`pause` and `finish_suspend` pause the displays, `will_suspend` does not);
- `rvg_device_finish_suspend` without `rvg_device_will_suspend` before it;
- a new mapping of guest memory, the guest's root page write for one, from `rvg_device_will_resume` until `rvg_device_did_resume`, whether or not the state was accepted;
- a feature level above 4: at the guest's first binary version write, at `rvg_device_reset`, and at a `rvg_device_will_resume` that applies the state's version;
- `rvg_display_cursor_position` while the device is paused (by `pause`, or a finished suspend), for a display the guest has linked;
- a NULL `modes` with a `count` above 0 to `rvg_display_set_modes`;
- a display released while one of its callback jobs is queued (section 5);
- a raise nothing catches;
- anything a host callback throws (`ffi::host_call` runs each call inside a catch);
- a NULL handle to a function that returns no status;
- an IOSurface device released inside its own `raise_interrupt`.

Every export is `extern "C"`: nothing unwinds out of one into the host, and a panic under one ends the process before anything unwinds (`crates/reims-vgpu-c/src/ffi.rs`).

## 2. The C header

The header has 48 functions and every type, callback typedef and constant, with the thread and ordering rules on each callback and each function. It has no Objective-C types, no `id<…>`, no `dispatch_queue_t` and no `__OBJC__` block. The C API's tests compile it as C11, as C++17 and as Objective-C.

| group | items |
|---|---|
| constants | `RVG_MAX_DISPLAY_PORTS` 8, `RVG_DEVICE_MMIO_SIZE` 0x4000, the PCI facts (`RVG_PCI_VENDOR_ID` 0x106b, `RVG_PCI_DEVICE_ID` 0xeeee, `RVG_PCI_BAR_MMIO` 0, `RVG_PCI_MSI_VECTORS` 1), the status codes, `RVG_RESUME_ERROR_*`, `RVG_DISPLAY_MODE_*` flags, `RVG_PIXEL_FORMAT_*` (Metal's numbers: 70, 71, 80, 81), `RVG_WINDOW_MATCH_HOST_SCREEN` 1, `RVG_WINDOW_BUTTON_*` (1, 2, 4, 8, 16), `RVG_LOG_*` |
| status and log | `rvg_status_message`, `rvg_log_set_handler` |
| device | `RvgDeviceConfig`, `RvgHostCallbacks`, `RvgMemoryMap`, `RvgMemoryRange`, `RvgMemoryRangeFile`; `rvg_device_create`, `_release`, `_mmio_read`, `_mmio_write`, `_reset`, `_pause`, `_unpause`, `_stop` |
| suspend and resume | `rvg_device_will_suspend`, `_finish_suspend`, `_will_resume`, `_did_resume`, `rvg_bytes_free` |
| displays | `RvgDisplayConfig`, `RvgDisplayCallbacks`, `RvgDisplayMode`, `RvgCursorGlyph`; `rvg_display_create`, `_release`, `_set_modes`, `_cursor_position`, `_present_counts` |
| a display's window | `RvgWindowConfig`, `RvgWindowCallbacks`, `RvgWindowPointerFn`, `RvgWindowWheelFn`, `RvgWindowKeyFn`, `RvgWindowCloseFn`, `RvgWindowScrollFn`, `RvgWindowScroll`; no function of its own: `RvgDisplayConfig.window` carries it |
| frames | `RvgFrameTargetConfig`, `RvgRegion`; `rvg_frame_target_create`, `_retain`, `_release`, `_read_pixels`; `rvg_display_encode_frame`; `rvg_frame_commit`, `_wait`, `_release` |
| IOSurface device | `RvgIOSurfaceDeviceConfig`, `RvgIOSurfaceCallbacks`; `rvg_iosurface_device_create`, `_release`, `_mmio_read`, `_mmio_write` |
| video device | `RvgVideoDeviceConfig`, `RvgVideoCallbacks`, `RvgVideoSendToGuestFn`, the `RVG_VIDEO_*` constants (Apple's PCI IDs 0x106b and 0x1a03, two queues, the guest's 0x4000-byte buffers, `RVG_VIDEO_INVALID_SESSION` -12903, `RVG_VIDEO_MAX_MESSAGE_SIZE` 16 MiB, `RVG_VIDEO_ALLOCATION_FAILED` -12904); `rvg_video_device_create`, `_release`, `_start`, `_deliver`, `_invalidate`, `_reset` |
| scaler device | `RvgScalerDeviceConfig`, `RvgScalerGuestMemory`, `RvgScalerIsRamFn`, `RvgScalerReadFn`, `RvgScalerWriteFn`, `RvgScalerOutcome` and its `RVG_SCALER_REPLY`, `_NEEDS_RESET`, `_END_VM`, the `RVG_SCALER_*` constants (Apple's PCI IDs 0x106b and 0x1a0b, one queue); `rvg_scaler_device_create`, `_release`, `_scales`, `_start`, `_process`, `_reset` |
| trace ranges | `rvg_trace_range_handler_invoke`, `_retain`, `_release` |

### What the header fixes

- **The feature level** is `RvgDeviceConfig.feature_level` (`uint64_t`, as core's).
  - 0 and 4 offer every feature, and the guest reads the same device info at both.
  - 1, 2 and 3 cap the binary version the guest may choose at 62, 113 and 209, and trim the device info it reads. A capped version reads the registers as that version does: below 245 the port count reads 1 and the connection types their low 2 bits.
  - A level above 4 is accepted. The guest's first binary version write then ends the process, as do `rvg_device_reset` and a `rvg_device_will_resume` that applies the state's version.

  The pins `api.device.feature_levels` and `api.death.feature_level_above_4_*` record the registers and the deaths, and the unit test `device::tests::the_feature_level_reaches_the_guest` the device info.
- **The Metal device** is `metal_registry_id`. 0 means `MTLCreateSystemDefaultDevice()`. Any other value is matched against the `registryID` of `MTLCopyAllDevices()`. No match means `RVG_ERR_NO_METAL_DEVICE`, after the port count.
- **Host callbacks.** `map_memory` is batched: one call maps a run of guest ranges. The task, memory and trace functions are called where and in the order core calls them.
- **`display_port_connection_types`** (core's field, default 0) and the mode flags `RVG_DISPLAY_MODE_EDR` and `_10BPC` are settable, as the feature level is. The display's nits are reachable through the Rust API's `Preferences` only.
  - The guest reads the connection types from MMIO 0x1234: their low 2 bits below binary version 245, all of them from 245. A resume of a state from binary version 238 or later refuses one taken with other connection types, with `RVG_RESUME_ERROR_INTERNAL_FAULT` (the pin `api.device.connection_types`).
  - The mode flags reach each mode in the guest's display page, and a bit the header does not define is dropped (the unit test `display::tests::mode_flags_reach_the_guest_s_display_page`). The EFI mode registers read only the modes' sizes (`api.display.efi_modes`).
- **Argument buffers** are offered only with process isolation, on a Metal device that can detach backing and is not itself paravirtual; `disable_argument_buffers` withholds them. With `process_isolation_follows_guest`, device info offers them only while the IOSurface device has a descriptor table, read when the guest asks (below).
- **The cursor glyph** is `RvgCursorGlyph { struct_size, width, height, hot_spot, rgba, rgba_length }`: the pixels core makes, `width * 4` bytes a row, red and blue swapped from the guest's BGRA.
- **The show byte** is raw: the guest's `uint8_t`, where 2 stays 2.
- **Frames:**
  - The host makes a frame target: a shared-storage 2D texture, not mipmapped, with usage render-target and shader-read, 1 to 16384 pixels each way, in one of four formats.
  - The host encodes the display's current frame into the target, as an uncommitted frame, commits it, waits, and copies the pixels into its own memory in the target's format.
  - The encode, the commit and the wait are separate calls, so a host can hold a frame in flight, as the suite's `hold` does.
  - The command buffer comes from the process's frame queue: one `MTLCommandQueue` per Metal device. `rvg_device_create` makes it the first time it is given that Metal device, before core builds the device and even when the device is then refused, and keeps it for the life of the process. A window's frames come from the same queue.
- **A display's window** is `RvgDisplayConfig.window`, NULL for none: `RvgWindowConfig { struct_size, flags, callbacks, mode_width, mode_height, mode_pixels_per_inch }` (its first version, 16 bytes, ends at `callbacks`, and a host built against it gets no mode; its second, 24 bytes, ends at `mode_height`, and a host built against it gets no density), with `callbacks` an `RvgWindowCallbacks { struct_size, ctx, destroy, pointer, wheel, key, close, scroll }`, NULL being a table with every callback absent.
  - **What it is.** The library opens a window for the display on one of the host's screens and draws each frame the guest presents into it, in one pass on the GPU; the host copies nothing. Its pointer, wheel and keys go to the window's callbacks. It opens once the process's main thread runs its run loop (`CFRunLoopRun`, or `NSApplication`'s), and the library never waits for that.
  - **What the guest sees.** A display with a window has all five display callbacks, whichever of `RvgDisplayCallbacks` the host gives: the guest reads a cursor glyph size limit of 256 and both cursor handler bits, as with all five (the unit test `window::tests::a_window_display_has_every_callback_the_guest_reads`). Each callback tells the window, then calls the host's where it gave one; `cursor_move` is the host's alone.
  - **`RVG_WINDOW_MATCH_HOST_SCREEN`** (flag 1; the other bits are ignored). The display offers the guest modes, a size in millimetres and a scale factor for the host screen its window opens on, in place of the config's size.
    - *With a mode* (`mode_width` and `mode_height` both above 0): that mode first, then the standard sizes of 16:10, 16:9 and 4:3 within it (neither side larger), at the screen's refresh rate or `REIMS_VGPU_WINDOW_REFRESH`'s, and the mode's size at 110 pixels per inch, scale factor 1: the guest runs a 1× desktop at that mode, which the window scales to its size on the GPU. That was measured for modes up to 3811 pixels wide; a mode 5832 pixels wide ran HiDPI at 128.8 ppi, and where between the two that starts was not measured. QEMU passes 1920 × 1080 unless told otherwise, since a guest at the screen's own 4K HiDPI mode runs much slower. `mode_pixels_per_inch` tells the mode's size at that density instead (QEMU's `resolution=2560x1600@200` is the display `macosvm`, over Virtualization.framework, gave the macOS 13.6 guest: its `vm.json`), and the scale factor the scale the window takes the guest to run. A macOS 13.6 guest reads the width in whole centimetres, rounded to whole millimetres and then up, and runs a HiDPI desktop at the mode's pixels where that comes to 2021 pixels per 35 cm (146.667 ppi) or more, for a mode 1920 to 3811 pixels wide: `2560x1600@148` reads as 44 cm, 147.8 ppi, and runs HiDPI, and `@146` as 45 cm, 144.5 ppi, and runs 1×. It ran 1× at up to 3811 pixels per 66 cm (146.666 ppi); a 1280 × 800 mode stayed 1× at 216.7 ppi, and HiDPI was measured from 1920 pixels wide and from 29 cm. The window takes a mode narrower than 1920 pixels, the densities between 146.666 and 146.667, and a mode wider than 3811 pixels below them as 1×; QEMU refuses a `resolution` density that does not tell the guest's scale for its mode (`reims-vgpu-window`'s `policy` lists the boots); it checks that mode alone, and the smaller modes offered with it are told the same size. 0, or a value that is not a positive finite number, is 110 ppi.
    - *Without one*, the host screen's own: the modes are the largest 16:10, 16:9 and 4:3 rectangles within the screen's pixels, each side rounded down to a multiple of 8, 16:10 first, then the standard smaller sizes of each ratio, at the screen's refresh rate or `REIMS_VGPU_WINDOW_REFRESH`'s. The screen's rate is the one it is set to, its current mode's, even where it offers a higher one at its size; where the current mode gives none, the highest of the modes of its size, and 60 Hz where none does. The density is that of the first mode: 110 pixels per inch per unit of the screen's backing scale, the size in millimetres the mode's pixels divided by `ppi / 25.46`, and the scale factor 2 above 143 ppi, else 1. A macOS 13.6 guest picks its HiDPI default from the size alone, so a Retina screen, told 220 ppi, gets a 2× desktop and a 1× screen a 1× one (`reims-vgpu-window`'s `policy` says what was measured).
    - *Without a screen* (as from an ssh shell with no Wayland compositor or X server named, where the window does not open): the config's mode and the standard sizes within it at its density, as with a screen, or, without a mode, 1920 × 1080 and the sizes within it at 110 pixels per inch, at `REIMS_VGPU_WINDOW_REFRESH`'s rate or else 60 Hz. Offering none would leave the guest no mode to set, and a macOS guest stops its boot at the Apple logo.
    - In each case, a later `rvg_display_set_modes` replaces the modes.
  - **The callbacks.** `pointer(ctx, x, y, width, height, buttons)`: the pointer at (`x`, `y`) in the guest's pixels, in a mode `width` by `height`, with `x` below `width` and `y` below `height` (off the image, and outside the window during a drag, it holds at the image's nearest edge), and `buttons` the `RVG_WINDOW_BUTTON_*` bits held: `LEFT` 1, `RIGHT` 2, `MIDDLE` 4 (the host's third button), `SIDE` 8 (its fourth, "back") and `EXTRA` 16 (its fifth, "forward"). `wheel(ctx, dy, dx)`: wheel notches, `dy` above 0 turned away from the person (up) and `dx` above 0 to the left, in the device's own direction whatever the host's scrolling setting; not called where the table has `scroll`. `key(ctx, usage, down)`: a USB HID keyboard usage (page 0x07) down or up; a held key is one down, which the guest repeats, and every key still down gets its up when the window loses focus, releases capture or closes. `close(ctx)`: the person closed the window or chose Quit; the window stays open, the host ends the VM, and its release of the display closes the window. `scroll(ctx, scroll)`: a scroll event, `scroll` an `RvgWindowScroll { struct_size, dy, dx, phase, momentum, notches_dy, notches_dx }` the library writes, which lives for the call. `dy` and `dx` are its distance in 1/256 of the host's point, in the same directions as `wheel`'s, a notched wheel's event being its lines at 10 points each; `phase` is where the event is in its gesture, an `RVG_WINDOW_SCROLL_PHASE_*`: `NONE` 0 (no gesture: a notched wheel, or momentum), `MAY_BEGIN` 1, `BEGAN` 2, `CHANGED` 3 (AppKit's Stationary too), `ENDED` 4, `CANCELLED` 5; `momentum` where it is in the momentum scrolling after the gesture, an `RVG_WINDOW_SCROLL_MOMENTUM_*`: `NONE` 0, `BEGAN` 1, `CHANGED` 2, `ENDED` 3 (AppKit's Cancelled too); `notches_dy` and `notches_dx` the notches `wheel` would have had, for a guest that takes no smooth scrolling. A gesture that begins over the image goes to `scroll` to its end, momentum included, wherever the pointer goes; one that begins elsewhere does not, and an event of no phase goes where the pointer is over the image. `scroll` came after the table's first version, which ends at `close`: 56 bytes. A host built against it passes 56, and its scrolling goes to `wheel` as notches (the unit test `window::tests::a_first_version_window_callbacks_is_56_bytes_and_has_no_scroll`). A callback the host leaves NULL sends that input nowhere, and without `close`, closing the window does nothing and Quit is ignored.
  - **The environment** picks the window's screen and style, and the offered modes' rate: `REIMS_VGPU_WINDOW_SCREENS` (a comma-separated list of host screen indices, the i-th for the display on port i; without one, or where it names no screen, display i opens on screen i, or on the main screen where there is none), `REIMS_VGPU_WINDOW_BORDERLESS` (`1`, `true`, `yes` or `on`: borderless windows that fill their screens) and `REIMS_VGPU_WINDOW_REFRESH` (the offered modes' refresh rate, above 0 Hz and up to 1000). The library reads them once per process; an unset or unparsable value is the default. Screen i is the i-th display CoreGraphics lists as active, the main one first, or, while every display sleeps, as online. What the window does for the person at the host is listed in `crates/reims-vgpu-window/features.md`.
  - **The window.** Once the display exists, its window opens on macOS on the main thread, and on Linux on the library's thread `reims-vgpu.ui` (section 5, "The main thread"). Its title is "Reims vGPU", followed by " (port N)" for a port N other than 0; the display's name is not in it. It opens on the screen the environment picks, found among AppKit's screens by its CoreGraphics display number, else on the main screen.
    - *Windowed*, the default: titled, closable, miniaturizable, resizable and able to go full screen. It opens showing the guest's mode, else the first mode the display offers (`RVG_WINDOW_MATCH_HOST_SCREEN`'s, or those of an `rvg_display_set_modes` made before the window opens), else 1920 × 1080, at one guest point per host point, shrunk with its ratio kept to fit the screen's visible area, and centred there: the guest runs a mode at scale 2 where the mode's width comes to 146.667 ppi or more across the size in millimetres the display told it, as a macOS 13.6 guest reads it (whole millimetres to nearest, then whole centimetres up) and for a mode at least 1920 pixels wide, else at 1, so on a 2× screen with room for it a HiDPI guest's pixel is a screen pixel (a 1× screen halves it, and a mode too big for the screen is shrunk), and a 1× guest's pixel is a point, scaled on the GPU. A window that would open with its top left on another's opens 24 points right and down, made smaller from there, its ratio kept, where the visible area has no room for it, though never below 320 × 200 points of content (it then stays where it was); a window following the guest's mode moves off another's top left the same way. Until the person resizes it (an edge, tiling, zoom, full screen: any resize the library did not make), it follows the guest's mode: each mode change resizes it as it would open on its current screen, keeping its top left corner, moved inside the visible area with its title bar on the screen. After that, a mode change re-fits the image and the window keeps its size.
    - *Borderless*: it covers its screen's frame, with the image inside the screen's safe area (below a camera housing), and the Dock and the menu bar hide while the application is frontmost. When the screens change, it fills its screen again, or the main screen while its own is gone.
  - **Its drawing.** A thread of the window's own draws the display's current frame into the next drawable of the window's `CAMetalLayer` whenever the guest presents a frame, the layout changes or the window shows again, and only while some of the window is on screen: nothing is drawn while it is hidden (covered, or the host's screen locked). The layer is `BGRA8Unorm`, tagged sRGB, opaque on black, at the window's backing scale, so a drawable pixel is a screen pixel. The image keeps the guest mode's aspect ratio, centred on whole pixels between black bars. The draw is one encode of the display's current frame, as into a frame target ("Frames", above), in a command buffer of the frame queue, which presents the drawable and commits without waiting. `nextDrawable` paces the draws at the screen's refresh rate, and nothing else throttles them.
  - **Its input.** Until a click captures the keyboard and the pointer, and again once they are released, the title ends " — click to capture input". A click in the key window's image captures both, never one without the other: the title then ends " — input captured (press ⌃⌥⌘ to release)", every key goes to the guest, ⌘ shortcuts included, the system's hot keys (⌘Tab, ⌘Space, Mission Control) are off while any window of the process holds the keyboard, and the host's pointer is held over the image (the window's `mouseConfinementRect`; on Linux, SDL's mouse rect). Pressing ⌃⌥⌘ together, Window ▸ Release Input, or the window losing key status releases both, and every key still down gets its up. Uncaptured, a key with ⌘ held goes to the host's menu, and every other key to the guest. The click that makes the window key is AppKit's, never the guest's. Keys go by position: a key's USB usage follows its macOS key code, with the two keys an ISO keyboard has left of 1 and right of left Shift swapped when the keyboard in use is ISO, asked at each key event. With `scroll`, each scroll event goes to the guest as it came, with its phases. Without it, a notched wheel gives a notch per notch; a trackpad or Magic Mouse one per 40 points of movement, momentum included.
  - **The cursor.** Over the image, the host's cursor is the guest's glyph with its hot spot, at the size the image shows it, its alpha straight as the glyph gives it; while the guest hides its cursor, an invisible one. Before the guest sets a glyph, and over the bars and the title bar, it is the host's arrow.
  - **The application.** When the host has made no `NSApplication`, the library makes one and runs it on the main thread, for good (section 5): a regular application, with a Dock icon and a ⌘Tab entry; a menu named after the process with Hide, Hide Others, Show All and Quit, and a Window menu with Minimize, Zoom, Enter Full Screen and Release Input; and an activity that keeps App Nap from slowing the process while its windows are hidden, and still lets the host sleep when idle. When the host has made one, the windows open in it, and a local event monitor gives the key window the key events it takes from AppKit (⌘ keys while captured, key-ups under ⌘).
  - **Closing.** Closing a window with its close button calls that window's `close`, and Quit (⌘Q uncaptured, or the Dock's) calls every window's, each after the ups of the keys and buttons still down; Quit then cancels AppKit's own termination, which would `exit()`. Neither closes a window: the release of its display does, without waiting. From the release's return no window callback runs but one already running on the main thread, the window closes there when that thread next runs its run loop, the system's hot keys come back if it held them, and its drawing thread ends by itself.
- **The log hook** is process-wide: `rvg_log_set_handler(fn, ctx, max_level)`.
  - Lines arrive on any thread, possibly under the library's locks, and the handler must not call the library.
  - A `max_level` above `RVG_LOG_TRACE` counts as `RVG_LOG_TRACE`. When `REIMS_VGPU_LOG` is set, its filter picks the lines in place of `max_level`.
  - NULL sends lines where `REIMS_VGPU_LOG` says: by default, nowhere.
  - A process that installed a `log` logger of its own keeps it: the lines go there, and the hook is not kept.
  - Whether logging is on never changes what the library does.
  - The pin `api.log.handler` records, with `REIMS_VGPU_LOG` unset, the levels a handler gets at each `max_level`, and that NULL stops them.
- **Frame statistics** (`stats.rs`), a diagnostic, off by default: with `REIMS_VGPU_STATS` set to `1`, `true`, `yes` or `on` (read once per process), the thread `reims-vgpu.stats` writes one line to stderr every second for each live display, by port.
  - The guest's presents per second, as the display's new-frame callback reports them, and the longest time between two.
  - For a window display, the frames its window's sink drew per second and the longest time between two, and the mean and longest GPU time of those draws, as the sink reports them (`-` while it has reported none).
  - A rate is over the time since the display's last line; a gap still open when the line is written counts, so a guest that stops presenting shows a gap that grows each second.
  - The lines go to stderr directly, not through the log hook. Whether the statistics are on never changes what the library does: the new-frame callback is wrapped only where the display has one, so the guest reads the callbacks it reads without them.
- **The Vulkan platform's differences from Metal** (feature `vulkan`, `reims-vgpu-vulkan`'s `answer`; [accuracy-ledger.md](accuracy-ledger.md) lists them): a difference from Metal's answer the platform can go on past, a degradation the runtime announces or a value the platform computes otherwise than Metal, is logged and the process goes on. The first occurrence of each kind writes a line to stderr, naming it, and every occurrence is counted.
  - With `REIMS_VGPU_STRICT` set to `1` (read once per process), each ends the process at its operation instead, naming the kind: for tests. The crate's own tests run strict.
  - Where Metal itself ends the process on the input, or the runtime cannot perform the operation at all (nothing drawn, made or copied), the process ends either way. So does a difference that breaks what the guest or host relies on: a released state an indirect command names, a sampler ID read as another sampler's, the GPU's writes reaching guest pages Metal keeps them from, and a deallocator run after its release (`answer::DIES_BY_DEFAULT`).
- **The Vulkan platform's exact forms** (feature `vulkan`): by default every device takes the runtime's fast native forms, which off Apple's GPUs answer within the bounds the runtime records and announces. `REIMS_VGPU_EXACT`, a comma-separated list read once per process, names the fields of the runtime's `DevicePolicy` that take its exact form instead, Metal's answer at the cost of larger modules and longer pipeline creation, of `ycbcr_filter`'s pipeline made at a draw, or of the GPU time, memory or kernel submissions of the forms that change no pipeline: `sampling` (Metal's sampler emulated), `ycbcr_filter` (Metal's filter of a two-plane YCbCr texture under native sampling, in a pipeline the draw binding one makes), `float_maths` (the translation's routines for the maths functions, divisions and fused multiply-adds Metal answers beyond Vulkan's precision table), `texel_conversion` (Metal's conversion of each texel a read takes, and of a sample's or gather's with `sampling`), `vertex_normalized` (Metal's conversion of a normalized vertex attribute, its codes divided exactly in the shader, in place of the device's fixed-function fetch), `half_rounding` (Metal's rounding of a `half` result), `nan_and_min_max` (Metal's NaN bits, and the operand its minimum and maximum functions pick at a NaN or a signed zero), `rounding_order` (Metal's rounding order of `dot`, `mix` and `fast::sqrt`), `half_blend` (Metal's `half` blend of a `half4` output of a module without fast math into a 16-bit float attachment, which the function computes), `half_texel_bits` (a 16-bit float texel's own bits at a read, through the texture's unsigned-integer alias, where the device's decode would answer its own NaN), `multisample_depth_twin` (a multisampled `Depth32Float` texture the program only reads held in a colour image, which keeps every `float` a fill or a sample copy writes, on a device whose depth images may not), `decoded_texels` (Metal's values in the decoded twin of an ASTC LDR or EAC 11-bit texture on a device without the format), `tile_store` (Metal's tile store of a loaded signed normalized or `RGB9E5Float` colour attachment), `resolve` (Metal's resolve rule for a multisampled colour attachment, in place of the device's average) and `batch_order` (each batch of a queue ordered after the earlier ones by the accesses it depends on, in place of their ends). `all` names the runtime's `DevicePolicy::exact_answers()`: every one of them but `sampling`, which may be named beside it. Every device takes the runtime's exact completion (`CompletionPolicy::Exact`: each command buffer seen complete at its own end, as Metal's) and its uncompressed heap textures (`HeapCompressionPolicy::Uncompressed`: a heap's colour texture Metal keeps losslessly compressed in an image the driver keeps uncompressed in the heap's bytes, whose size and alignment the size query answers) whatever the variable names, and no name selects the grouped or the compressed form ([accuracy-ledger.md](accuracy-ledger.md)). A device that runs on Metal (MoltenVK) answers as Metal under either, but for the NaN a `half` read, sample or gather hands back for a NaN texel, Metal's only under `nan_and_min_max` and, for a read of a 16-bit float texture, whose NaN the device's decode answers as `0x7e00`, only under `half_texel_bits`, and a linear sample of a two-plane YCbCr texture, which Metal's own sampler filters by default, within one step of the format's grid, and Metal's filter of the format only under `ycbcr_filter`; a name not among these changes nothing, and a line on stderr says so.
- **No option ROM.** The API names no ROM: the dist tree carries it, and `reims-vgpu.pc` names it (docs/qemu.md, "The QEMU build").
- **The IOSurface device** has 64-bit MMIO. QEMU reads and writes it as 32 bits, and the truncation is QEMU's (docs/testing.md 3.1).

### Settings and the environment

Everything a host can set that shapes what the guest is offered or what the library does, beside the guest's own input. Each stays configurable.

- **The device config** (`RvgDeviceConfig`, the Rust API's `DeviceConfig`): `feature_level`, `display_port_connection_types` and `disable_argument_buffers` (above), with `metal_registry_id`, the memory map, `use_iosurface_mapper`, `process_isolation` and `process_isolation_follows_guest`. Zero is every field's default (section 1). With `process_isolation_follows_guest`, a task other than task 0 is isolated only when, as the guest defines it, the IOSurface device the device maps through has a descriptor table base other than 0; otherwise it runs in process, as without isolation, and each such task logs one info line saying which and why. Device info's argument-buffer answer (key 33 bit 0x20, key 42) follows the same rule, read each time the guest asks, since only isolated tasks serve argument buffers; it logs one info line each time the answer changes, with the reason (`device info: argument buffers off: the IOSurface device has no descriptor table`, or `on: the IOSurface device's descriptor table is at 0x…`). An isolated task's handshake carries the device's offer as it was, since such a task exists only when the table was there. The guest's mapper driver writes the base at its start, and Apple's arm64 GPU driver waits for the mapper before it makes a task. Apple's x86-64 driver never touches the IOSurface device and references IOSurfaces through task 0's backings (ref textures, docs/testing.md 6.20), which an isolated task does not serve; it does so at binary version 6 and, patched, at 245, so the version cannot tell the two drivers apart (docs/qemu.md, "The IOSurface device over PCI").
- **A mode's flags**: `RVG_DISPLAY_MODE_EDR` and `RVG_DISPLAY_MODE_10BPC` (the Rust API's `supports_edr`, `supports_10bpc`).
- **Preferences**, in the Rust API only: `DisplayConfig::apply_preferences` takes the display's nits, and `DisplayMode::with_preferences` a mode's EDR and 10 bpc, from a `Preferences` reader the host gives. The library reads no preferences domain itself: `singleton::preferences()` answers every key absent.
- **A display's window**: `RvgDisplayConfig.window` (above).
- **Cargo features**, for a host that builds the library: `vulkan` (`reims-vgpu`, `reims-vgpu-c`) puts the API on the Vulkan platform, and `macos-sdl` (on by default) builds SDL3's window into the macOS library ([development.md](development.md), "Features").
- **The environment.** The library reads these once per process; an unset or unparsable value is the default.

  | variable | what it does |
  |---|---|
  | `REIMS_VGPU_LOG` | the library's log lines to stderr, or to the log hook, filtered by a level and `target=level` pairs (`warn,reims_vgpu_core::fifo=debug`); unset, nowhere ("The log hook", above) |
  | `REIMS_VGPU_STATS` | `1`, `true`, `yes` or `on`: a line of frame statistics a second per display ("Frame statistics", above) |
  | `REIMS_VGPU_WINDOW_SCREENS`, `_BORDERLESS`, `_REFRESH` | a window's screen, its style and the offered modes' refresh rate ("The environment", under "A display's window", above) |
  | `REIMS_VGPU_WINDOW_IMPL` | macOS: `sdl` opens SDL3's window in place of AppKit's, in a build with `macos-sdl` |
  | `REIMS_VGPU_STRICT` | the Vulkan platform: `1` ends the process at each difference from Metal's answer it would otherwise log |
  | `REIMS_VGPU_EXACT` | the Vulkan platform: the runtime's exact forms in place of its fast ones, by field ("The Vulkan platform's exact forms", above) |
  | `REIMS_VGPU_CENSUS_DIR` | the Vulkan platform on Linux, a diagnostic for inventory runs: each difference that would end the process is recorded in that directory and logged instead, and the run says it is not faithful to Metal; ignored on macOS |
  | `REIMS_VGPU_TRACE` | the Vulkan platform, a diagnostic for profiling: a timeline of the device's work, a line per event, written to the file it names (core's `trace` module); the Metal platform writes nothing |
  | `REIMS_VGPU_TRACE_DEGRADE` | with `REIMS_VGPU_TRACE`, `1`: every occurrence of every difference from Metal the runtime announces is written to the trace; the log is as without it |

  `REIMS_VGPU_WINDOW` is QEMU's, not the library's: `0` keeps `apple-gfx` headless under `-display none` (docs/qemu.md, "Window mode"). Of these, only `REIMS_VGPU_STRICT`, `REIMS_VGPU_EXACT` and `REIMS_VGPU_CENSUS_DIR` change what the guest or host observes, and only on the Vulkan platform (the trace's two change only the trace and the log lines); `REIMS_VGPU_WINDOW_SCREENS` and `_REFRESH` change the modes a window display with `RVG_WINDOW_MATCH_HOST_SCREEN` offers its guest.

### cbindgen

`crates/reims-vgpu-c/cbindgen.toml`:
- language C, `c99` doc style, `cpp_compat`, `sort_by = "None"`, `parse_deps = false`;
- its `header` preamble states the rules for handles, lifetimes, configs, callbacks and deaths;
- functions come in the order cbindgen parses them: the order of `lib.rs`'s public modules.

On the Rust side:
- the codes and formats are `pub type RvgStatus = u32` and `pub const` items, so a host's out-of-range value is never a Rust enum;
- the handles are Rust structs cbindgen emits as opaque;
- the configs are `#[repr(C)]`.

## 3. The Rust API (`reims-vgpu`)

The C API is a thin layer over this; a Rust host uses it directly. Every handle type is generic over `P: Platform = DefaultPlatform` (`MetalPlatform` on macOS, `VulkanPlatform` on Linux), so the C layer's unit tests run over core's `FakePlatform`. `Device::with_platform`, `Device::new_display`, `IoSurfaceDevice::with_platform` and `FrameTarget::read_pixels` need `P: ApiPlatform`, which adds to core's `Platform` what the API asks of it: the process's instance, the Metal device a config names, and a frame target's pixels. A window display's host screen and the sink that draws its frames are chosen inside the API, by the platform's type (3.1).

### 3.1 Types and signatures

The crate's rustdoc is the reference. Without the bodies, its public surface is:

```rust
// Re-exported from core and the platform.
pub use reims_vgpu_core::host::{
    AddTraceRange, CreateTask, DestroyTask, HostPtr, HostTask, HostTraceRange, MapMemory,
    PhysicalMemoryRange, RaiseInterrupt, ReadMemory, RemoveTraceRange, TraceRangeHandler,
    UnmapMemory,
}; // the callback types (CreateTask … UnmapMemory, TraceRangeHandler): Arc<dyn Fn(..) + Send + Sync>
pub use reims_vgpu_core::platform::os::Preferences;
pub use reims_vgpu_metal::MetalPlatform;    // macOS
pub use reims_vgpu_vulkan::VulkanPlatform;  // feature `vulkan`
pub type DefaultPlatform = MetalPlatform;   // VulkanPlatform on Linux
pub use reims_vgpu_window::{
    BUTTON_EXTRA, BUTTON_LEFT, BUTTON_MIDDLE, BUTTON_RIGHT, BUTTON_SIDE, CloseFn, KeyFn,
    MomentumPhase, PointerFn, ScrollEvent, ScrollFn, ScrollPhase, WheelFn, WindowCallbacks,
    WindowConfig,
}; // a display's window, below

pub const MAX_DISPLAY_PORTS: u32 = 8;
pub const DEVICE_MMIO_SIZE: u64 = 0x4000;
pub const PCI_VENDOR_ID: u16 = 0x106b;
pub const PCI_DEVICE_ID: u16 = 0xeeee;
pub const MAX_FRAME_TARGET_SIZE: u32 = 16384;

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    InvalidArgument, DisplayPortCount, NoMetalDevice, MemoryConflict, MemoryMap,
    IsolationWithoutMap, Metal, PortOutOfRange, PortInUse, SerialInUse, NoMemoryMap,
    Suspend, Resume(Option<ResumeError>), Gpu, FrameState,
}
impl std::fmt::Display for Error {}
impl std::error::Error for Error {}

#[repr(u32)]
pub enum ResumeError {
    InternalFault = 0, InvalidSuspendStateVersion = 1, InvalidContent = 2,
    InvalidGuestVersion = 3, IncompatibleDevice = 4, InvalidDisplayPortCount = 5,
}
impl ResumeError {
    pub fn from_code(code: u64) -> Option<ResumeError>;
    pub fn code(self) -> u32;
}

/// Guest RAM and the host memory behind it, in order, with the file behind each range where the
/// host names one.
#[derive(Clone, Default, Debug)]
pub struct MemoryMap { /* the ranges and their files */ }
impl MemoryMap {
    pub fn new() -> MemoryMap;
    /// # Safety
    /// `host .. host + length` stays mapped while any device made over the map lives.
    pub unsafe fn push(&mut self, guest_physical_address: u64, length: u64, host: *mut u8);
    /// # Safety
    /// As `push`; with `file`, that memory is a shared mapping of `file.fd` from `file.offset`.
    pub unsafe fn push_with_file(
        &mut self, guest_physical_address: u64, length: u64, host: *mut u8,
        file: Option<MemoryFile>,
    );
    pub fn ranges(&self) -> &[GuestPhysicalRange]; // core's
    pub fn files(&self) -> &[Option<MemoryFile>];  // one per range
}

/// The file behind one range: its host memory is a shared mapping of `fd` from `offset`.
#[derive(Clone, Debug)]
pub struct MemoryFile {
    pub fd: Arc<OwnedFd>,
    pub offset: u64, // a multiple of the host's page size
}

/// Absent is `None`; the device behaves as for a NULL C function pointer.
#[derive(Clone, Default)]
pub struct HostCallbacks {
    pub create_task: Option<CreateTask>,         // (vm_size, &mut base) -> task
    pub destroy_task: Option<DestroyTask>,       // (task)
    pub map_memory: Option<MapMemory>,           // (task, count, offset, read_only, ranges) -> ok
    pub unmap_memory: Option<UnmapMemory>,       // (task, offset, length) -> ok
    pub read_memory: Option<ReadMemory>,         // (address, length, dst) -> ok
    pub raise_interrupt: Option<RaiseInterrupt>, // (vector)
    pub add_trace_range: Option<AddTraceRange>,  // (range, handler) -> trace range
    pub remove_trace_range: Option<RemoveTraceRange>, // (trace range)
}

#[derive(Clone, Debug)]
pub struct DeviceConfig {
    pub metal_registry_id: Option<u64>,       // None: the system default
    pub display_port_count: u32,              // Default: 1
    pub feature_level: u64,
    pub display_port_connection_types: u32,
    pub memory_map: Option<MemoryMap>,
    pub use_iosurface_mapper: bool,
    pub process_isolation: bool,
    pub process_isolation_follows_guest: bool, // With process_isolation: a mapper's table
    pub argument_buffers: bool,               // Default: true
    pub callbacks: HostCallbacks,
}
impl Default for DeviceConfig { /* the values above */ }

pub struct Device<P: Platform = DefaultPlatform> { /* core's device */ }
impl Device {
    pub fn new(config: DeviceConfig) -> Result<Device, Error>;
}
impl<P: ApiPlatform> Device<P> {
    pub fn with_platform(platform: Arc<P>, config: DeviceConfig) -> Result<Device<P>, Error>;
    pub fn new_display(&self, config: DisplayConfig) -> Result<Display<P>, Error>;
}
impl<P: Platform> Device<P> {
    pub fn clone_ref(&self) -> Device<P>;
    pub fn ptr_eq(&self, other: &Device<P>) -> bool;
    pub fn mmio_read(&self, offset: u64) -> u32;
    pub fn mmio_write(&self, offset: u64, value: u32);
    pub fn reset(&self);
    pub fn pause(&self);
    pub fn unpause(&self);
    pub fn stop(&self);
    pub fn will_suspend(&self);
    pub fn finish_suspend(&self) -> Result<Vec<u8>, Error>;          // Error::Suspend
    pub fn will_resume(&self, state: Option<&[u8]>) -> Result<(), Error>; // Error::Resume(_)
    pub fn did_resume(&self);
    pub fn new_frame_target(&self, width: u32, height: u32, format: PixelFormat)
        -> Result<FrameTarget<P>, Error>;
}
// The last drop of a Device, Display or IoSurfaceDevice is its end, on the dropping thread.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DisplayMode {
    pub width: u16,
    pub height: u16,
    pub refresh_rate_hz: f64,
    pub supports_edr: bool,
    pub supports_10bpc: bool,
}
impl DisplayMode {
    pub fn new(width: u16, height: u16, refresh_rate_hz: f64) -> DisplayMode;
    /// The flags `prefs` answers (PGDisplaySupportsEDR, PGDisplaySupports10bpc).
    pub fn with_preferences(prefs: &dyn Preferences, width: u16, height: u16, refresh_rate_hz: f64)
        -> DisplayMode;
}

#[derive(Clone, Copy, Debug)]
pub struct CursorGlyph<'a> {
    pub width: u32,
    pub height: u32,
    pub hot_spot: (u16, u16),
    pub rgba: &'a [u8],
}

pub type ModeChangeFn = Arc<dyn Fn(u16, u16, u32) + Send + Sync>;
pub type NewFrameFn = Arc<dyn Fn() + Send + Sync>;
pub type CursorGlyphFn = Arc<dyn Fn(&CursorGlyph<'_>) + Send + Sync>;
pub type CursorShowFn = Arc<dyn Fn(u8) + Send + Sync>;
pub type CursorMoveFn = Arc<dyn Fn() + Send + Sync>;

#[derive(Clone, Default)]
pub struct DisplayCallbacks {
    pub mode_change: Option<ModeChangeFn>,
    pub new_frame: Option<NewFrameFn>,
    pub cursor_glyph: Option<CursorGlyphFn>,
    pub cursor_show: Option<CursorShowFn>,
    pub cursor_move: Option<CursorMoveFn>,
}

#[derive(Clone)]
pub struct DisplayConfig {
    pub port: u64,
    pub serial_number: u32,
    pub name: Option<String>,                 // None: "Apple Virtual"
    pub size_in_millimeters: (f64, f64),
    pub callbacks: DisplayCallbacks,
    pub window: Option<WindowConfig>,         // None: no window
    // private: the rest of core's display descriptor (colour space, nits, …) at its defaults
}
impl DisplayConfig {
    pub fn new(port: u64, serial_number: u32) -> DisplayConfig; // no callbacks, no window
    /// The nits settings `prefs` answers.
    pub fn apply_preferences(&mut self, prefs: &dyn Preferences);
}

// The window crate's (reims_vgpu_window::config), re-exported. Every callback runs on the
// process's main thread on macOS, and on the library's reims-vgpu.ui thread on Linux (section 5).
pub const BUTTON_LEFT: u32 = 1 << 0;   // BUTTON_RIGHT 1 << 1, BUTTON_MIDDLE 1 << 2,
                                       // BUTTON_SIDE 1 << 3, BUTTON_EXTRA 1 << 4
pub type PointerFn = Arc<dyn Fn(u16, u16, u16, u16, u32) + Send + Sync>; // (x, y, width, height, buttons)
pub type WheelFn = Arc<dyn Fn(i32, i32) + Send + Sync>;                  // (dy, dx)
pub type KeyFn = Arc<dyn Fn(u16, bool) + Send + Sync>;                   // (usage, down)
pub type CloseFn = Arc<dyn Fn() + Send + Sync>;
pub type ScrollFn = Arc<dyn Fn(ScrollEvent) + Send + Sync>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScrollEvent {                                                     // RvgWindowScroll
    pub dy: i32,
    pub dx: i32,
    pub phase: ScrollPhase,
    pub momentum: MomentumPhase,
    pub notches: (i32, i32),                                                 // (notches_dy, notches_dx)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ScrollPhase { #[default] None, MayBegin, Began, Changed, Ended, Cancelled } // RVG_WINDOW_SCROLL_PHASE_*
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MomentumPhase { #[default] None, Began, Changed, Ended }                   // RVG_WINDOW_SCROLL_MOMENTUM_*

#[derive(Clone, Default)]
pub struct WindowCallbacks {
    pub pointer: Option<PointerFn>,
    pub wheel: Option<WheelFn>,
    pub key: Option<KeyFn>,
    pub close: Option<CloseFn>,
    pub scroll: Option<ScrollFn>,
}

#[derive(Clone, Debug, Default)]
pub struct WindowConfig {
    pub match_host_screen: bool,              // RVG_WINDOW_MATCH_HOST_SCREEN
    pub mode: Option<(u16, u16)>,             // mode_width, mode_height; None for 0
    pub mode_ppi: Option<f64>,                // mode_pixels_per_inch; None unless positive and finite
    pub callbacks: WindowCallbacks,
}

pub struct Display<P: Platform = DefaultPlatform> { /* core's display */ }
impl<P: Platform> Display<P> {
    pub fn clone_ref(&self) -> Display<P>;
    pub fn ptr_eq(&self, other: &Display<P>) -> bool;
    pub fn set_modes(&self, modes: &[DisplayMode]);
    pub fn cursor_position(&self) -> (u16, u16);
    pub fn guest_present_count(&self) -> u64;
    pub fn host_present_count(&self) -> u64;
    pub fn encode_frame(&self, target: &FrameTarget<P>, region: Option<Region>)
        -> Result<Frame<P>, Error>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PixelFormat { Rgba8Unorm, Rgba8UnormSrgb, Bgra8Unorm, Bgra8UnormSrgb }
impl PixelFormat {
    pub fn from_metal(format: u32) -> Option<PixelFormat>;
    pub fn metal(self) -> u32;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Region { pub x: u64, pub y: u64, pub width: u64, pub height: u64 }

pub struct FrameTarget<P: Platform = DefaultPlatform> { /* a counted reference to the texture */ }
impl<P: Platform> FrameTarget<P> {
    pub fn clone_ref(&self) -> FrameTarget<P>;
    pub fn width(&self) -> u32;
    pub fn height(&self) -> u32;
    pub fn format(&self) -> PixelFormat;
    pub fn ptr_eq(&self, other: &FrameTarget<P>) -> bool;
}
impl<P: ApiPlatform> FrameTarget<P> {
    pub fn read_pixels(&self, dst: &mut [u8], bytes_per_row: usize) -> Result<(), Error>;
}

pub struct Frame<P: Platform = DefaultPlatform> { /* command buffer, target */ }
impl<P: Platform> Frame<P> {
    pub fn commit(&mut self) -> Result<(), Error>;   // Error::FrameState if committed
    pub fn is_committed(&self) -> bool;
    pub fn wait(&self) -> Result<(), Error>;         // Error::Gpu, Error::FrameState
}

#[derive(Clone, Default)]
pub struct IoSurfaceDeviceConfig {
    pub memory_map: Option<MemoryMap>,
    pub raise_interrupt: Option<RaiseInterrupt>,
}
pub struct IoSurfaceDevice<P: Platform = DefaultPlatform> { /* core's */ }
impl IoSurfaceDevice {
    pub fn new(config: IoSurfaceDeviceConfig) -> Result<IoSurfaceDevice, Error>;
}
impl<P: ApiPlatform> IoSurfaceDevice<P> {
    pub fn with_platform(platform: Arc<P>, config: IoSurfaceDeviceConfig)
        -> Result<IoSurfaceDevice<P>, Error>;
}
impl<P: Platform> IoSurfaceDevice<P> {
    pub fn clone_ref(&self) -> IoSurfaceDevice<P>;
    pub fn ptr_eq(&self, other: &IoSurfaceDevice<P>) -> bool;
    pub fn mmio_read(&self, offset: u64) -> u64;
    pub fn mmio_write(&self, offset: u64, value: u64);
}

// The video device (106b:1a03).
pub const INVALID_SESSION: i32 = -12903;
pub const MAX_MESSAGE_SIZE: usize = 16 << 20; // a longer message is answered ALLOCATION_FAILED
pub const ALLOCATION_FAILED: i32 = -12904;
pub type SendToGuest = Arc<dyn Fn(&[u8]) + Send + Sync>;
pub struct VideoDeviceConfig<P: Platform = DefaultPlatform> {
    pub iosurface_device: Option<IoSurfaceDevice<P>>, // held unretained; None: the process's latest
    pub send_to_guest: Option<SendToGuest>,
}
pub struct VideoDevice<P: ApiPlatform = DefaultPlatform> { /* the host session and its backend */ }
impl VideoDevice {
    pub fn new(config: VideoDeviceConfig) -> VideoDevice;
}
impl<P: ApiPlatform> VideoDevice<P> {
    pub fn with_platform(platform: Arc<P>, config: VideoDeviceConfig<P>) -> VideoDevice<P>;
    pub fn start(&self);
    pub fn deliver(&self, message: &[u8]) -> i32;
    pub fn invalidate(&self);
    pub fn reset(&self);
}

// The M2 scaler device (106b:1a0b).
pub use reims_vgpu_video_core::scaler::{GuestMemory, Outcome as ScalerOutcome};
pub struct ScalerDeviceConfig<P: ApiPlatform = DefaultPlatform> {
    pub iosurface_device: Option<IoSurfaceDevice<P>>, // as the video device's
}
pub struct ScalerDevice<P: ApiPlatform = DefaultPlatform> { /* core's scaler device and its backend */ }
impl ScalerDevice {
    pub fn new(config: ScalerDeviceConfig) -> ScalerDevice;
}
impl<P: ApiPlatform> ScalerDevice<P> {
    pub fn with_platform(platform: Arc<P>, config: ScalerDeviceConfig<P>) -> ScalerDevice<P>;
    pub fn with_backend(platform: Arc<P>, config: ScalerDeviceConfig<P>,
        backend: Arc<dyn ScalerBackend>) -> ScalerDevice<P>;      // a host's own backend, the tests'
    pub fn without_gpu(platform: Arc<P>, config: ScalerDeviceConfig<P>,
        why: String) -> ScalerDevice<P>;                          // one that does not scale
    pub fn scales(&self) -> bool;
    pub fn start(&self) -> Result<(), &'static str>;
    pub fn process(&self, request: &[u8], writable: usize, memory: &dyn GuestMemory)
        -> ScalerOutcome;
    pub fn reset(&self);
}

// One plane of a surface, as a platform gives it to the video devices.
pub struct SurfacePlane {
    pub offset: u64,
    pub width: u64,
    pub height: u64,
    pub bytes_per_element: u64,
    pub bytes_per_row: u64,
}

pub mod platform {
    pub trait ApiPlatform: Platform {
        fn process() -> Arc<Self>;
        fn metal_device(&self, registry_id: Option<u64>) -> Option<Self::Device>;
        fn read_pixels(&self, texture: &Self::Texture, dst: &mut [u8], bytes_per_row: usize);
        fn guest_memory(&self, map: &MemoryMap) -> Box<dyn Any + Send + Sync>; // held while a device over `map` lives; nothing by default
        // What the video devices reach of a guest surface; each refuses by default.
        fn surface_planes(&self, surface: &Self::Surface) -> Vec<SurfacePlane>;
        fn lend_surface_bytes(&self, surface: &Self::Surface, write: &mut dyn FnMut(&mut [u8]))
            -> Result<(), &'static str>;
        fn lend_surface_bytes_for_reading(&self, surface: &Self::Surface, read: &mut dyn FnMut(&[u8]))
            -> Result<(), &'static str>;
        fn surface_device_memory(&self, surface: &Self::Surface)
            -> Result<Box<dyn DeviceMemory>, &'static str>; // reims_vgpu_video_core::device_memory
    }
    pub const DISPLAY_THREAD_LABEL: &str = "reims-vgpu.display";
    pub const FAKE_HOST_SCREEN: HostScreen; // testkit: 3024 × 1964 pixels, scale 2, 120 Hz
}

// The window crate's (reims_vgpu_window), which FAKE_HOST_SCREEN is.
pub struct HostScreen { pub id: u32, pub pixels: (u32, u32), pub scale: f64, pub max_refresh_hz: f64 }
pub mod singleton {
    pub fn platform() -> Arc<MetalPlatform>;               // the process's, made on first use (macOS)
    pub fn platform_ref() -> &'static Arc<MetalPlatform>;
    pub fn vulkan_platform() -> Arc<VulkanPlatform>;       // the same, for feature `vulkan`
    pub fn preferences() -> &'static dyn Preferences;      // every key absent
    pub fn install_panic_hook();
}
```

With the `testkit` feature, `Device`, `Display` and `IoSurfaceDevice` also give their core object (`core()`), and the crate re-exports `reims_vgpu_video_core`, whose scaler stand-in the C API's tests make devices on.

- **The Rust API has no log hook.** It logs through the `log` crate, and a Rust host installs its own logger. The C layer's `rvg_log_set_handler` installs one that forwards to the host's function.
- **Refusals leave the closures dropped.** A refused create drops the closures it was given before it returns, except a refused display's new-frame closure, which its source's cancel frees later on the display thread. The C layer's one `destroy` per table rides on that: every closure it builds holds one `Arc` of the table, whose last drop runs `destroy(ctx)` (section 5). A refused window display closes its window before it returns, which drops the window's closures even while that new-frame closure still holds the window (`window::tests::a_refused_window_display_closes_its_window`).
- **A display's window.** With `DisplayConfig::window`, `new_display` makes the window, gives the display the window's five callbacks in place of the config's (each tells the window, then calls the host's where it gave one), and, with `match_host_screen`, sets the display's scale factor and size in millimetres from the window's host screen, or, without one, from the window's mode or 1920 × 1080, and makes it with the offered modes, as `set_modes` sets them. Once the display exists, the window gets what draws its frames. Both are chosen by the platform's type (`window::host_screen`, `window::window_sink`): `MetalPlatform` answers from the host's screens (`reims_vgpu_window::screens`) and with `window::metal::MetalSink`, and the Vulkan platform (feature `vulkan`) from the same screens and with the sink of `window::vulkan`, `SwapchainSink`, which draws each frame into the window's swapchain on the runtime's device, on either OS (on macOS a swapchain of the window's `CAMetalLayer`). So `ApiPlatform` names no window type. Every `clone_ref` shares the window, and the last reference's drop closes it before that reference lets the display go (`window::tests::the_last_reference_closes_the_window_before_the_display_ends`). The testkit's platform answers `FAKE_HOST_SCREEN` and no sink, so its window displays never open a window.

### 3.2 How it sits on core

- **Refusal reasons.** Core's constructors for the device, the display and the IOSurface device return the reason for a refusal (`DeviceRefusal`, `DisplayRefusal`, `IoSurfaceRefusal`), and `Error` converts each. The trace pair is one of them: `add_trace_range` without `remove_trace_range` is refused at its point in the order, releasing what was built, as the refusals around it do.
- **The process's platform.** `singleton::platform` makes one `MetalPlatform` per process on first use (`vulkan_platform` one `VulkanPlatform`), after installing the aborting panic hook and, when `REIMS_VGPU_LOG` asks for it, the stderr logger. It runs each isolated task's service on threads of the process (`tasks::isolation::TaskThreads`). The process-wide state (the serial numbers, the latest IOSurface device) lives in it.
- **The display executor.** `platform.rs` implements core's `HostQueue` over one process-wide `SerialQueue`, and every display is given it (section 5):
  - `dispatch_retain` is another handle to the same queue;
  - core's `CoalescingSignal` on it serves `new_frame`.
- **The frame queue.** `Device::with_platform` makes it before it calls core's constructor (section 2).
- **Windows** (`window.rs`). The tee (`window::tee`) builds a window display's five callbacks, which hold the window and the host's closures, never the display. `WindowLink`, which every reference to the display shares, closes the window when it drops, inside an entry, waiting for nothing. `MetalSink` (`window/metal.rs`) holds core's display weakly: a draw upgrades it for its encode, `presentDrawable:` and commit alone, and core's encode registers a completion handler that holds the display until the frame completes, as a `Frame`'s does. The Vulkan platform's sink (`window/vulkan/`), `SwapchainSink`, holds it the same way, and its presenter holds the frame queue until the window's render thread ends. `host_matched` asks the window crate's policy for the modes and density of the host screen.
- **Entries.** Every public method, and every release of a handle, runs inside `entry::run`: the thread is marked as the library's, so a panic aborts at its operation, and a pool and a holder are open, so what the call held or Metal autoreleased on the thread is released before it returns. `retained.rs`'s `Ours` runs a handle's release as such an entry, so a last release runs the teardown the same way.

## 4. The functions

The header's 48 functions, in its order. "May" is the header's mark "may call back": the device's callbacks may run on the calling thread before the function returns. A function without the mark never runs one there. A release may also run its table's `destroy` on the calling thread (section 5).

| # | function | calls back | what it does |
|---|---|---|---|
| 1 | `rvg_status_message` | | a static description of a status; any thread, inside any callback |
| 2 | `rvg_log_set_handler` | | sends the library's log lines to the host's function (section 2) |
| 3 | `rvg_device_create` | | a device, from an `RvgDeviceConfig` |
| 4 | `rvg_device_release` | may | drops the host's reference; the last reference is the teardown |
| 5 | `rvg_device_mmio_read` | may | a 32-bit read of the device's MMIO region |
| 6 | `rvg_device_mmio_write` | may | a 32-bit write; a root FIFO write waits for the FIFO's thread |
| 7 | `rvg_device_reset` | may | the guest's bus reset; returns once the device is quiesced |
| 8 | `rvg_device_pause` | may | each FIFO finishes the commands it started and stops, then the guest's displays pause |
| 9 | `rvg_device_unpause` | may | unpauses the guest's displays, starts every FIFO and raises one interrupt |
| 10 | `rvg_device_stop` | may | stops the device as a root page of 0 does: its FIFOs, the guest's tasks and display pages are torn down |
| 11 | `rvg_device_will_suspend` | may | starts a suspend: no interrupt is raised from here; it quiesces the FIFOs without waiting |
| 12 | `rvg_device_finish_suspend` | may | ends it: the state as `RvgBytes`; the guest's tasks end and its displays pause |
| 13 | `rvg_bytes_free` | | frees what `finish_suspend` wrote |
| 14 | `rvg_device_will_resume` | may | offers a state: `RVG_OK`, or `RVG_ERR_RESUME` with the reason |
| 15 | `rvg_device_did_resume` | may | ends an accepted resume |
| 16 | `rvg_display_create` | may | a display on a port, from an `RvgDisplayConfig` |
| 17 | `rvg_display_release` | may | drops the host's reference; the last one unplugs the display |
| 18 | `rvg_display_set_modes` | may | offers the guest a list of modes, copied, of which its display page takes the first 128; 0 modes changes nothing |
| 19 | `rvg_display_cursor_position` | | the cursor's position; {65535, 65535} before the guest links the display |
| 20 | `rvg_display_present_counts` | | the guest's presents and the host's new-frame encodes |
| 21 | `rvg_frame_target_create` | | a frame target on the device's Metal device |
| 22 | `rvg_frame_target_retain` | | one more reference |
| 23 | `rvg_frame_target_release` | | drops one reference |
| 24 | `rvg_frame_target_read_pixels` | | copies the whole target into host memory |
| 25 | `rvg_frame_target_update_pixels` | | brings a host copy up to the target, writing and flagging only the rows that changed |
| 26 | `rvg_display_encode_frame` | | encodes the current frame into a target: an uncommitted frame |
| 27 | `rvg_frame_commit` | | submits it to the GPU |
| 28 | `rvg_frame_wait` | | waits for it: `RVG_OK` or `RVG_ERR_GPU` |
| 29 | `rvg_frame_release` | | drops it; an uncommitted frame stays counted in flight for the display's life |
| 30 | `rvg_iosurface_device_create` | | the IOSurface device, the process's latest |
| 31 | `rvg_iosurface_device_release` | | its teardown, which waits for its thread |
| 32 | `rvg_iosurface_device_mmio_read` | | a 64-bit read of its registers (0x1000 to 0x1030); other offsets read all ones |
| 33 | `rvg_iosurface_device_mmio_write` | | a 64-bit write; other offsets are ignored |
| 34 | `rvg_trace_range_handler_invoke` | | tells the device the guest wrote a traced range |
| 35 | `rvg_trace_range_handler_retain` | | one more reference |
| 36 | `rvg_trace_range_handler_release` | | drops one reference |
| 37 | `rvg_video_device_create` | | the video device (106b:1a03), from an `RvgVideoDeviceConfig`, with no host session yet; its codecs open on the backend the library picks and logs: VideoToolbox on macOS, on Linux NVIDIA's or VA-API on the GPU the GPU device runs on, else on the host's first hardware GPU that offers video (docs/video/backends.md, "Picking a backend"); none where no backend opens, and the guest is offered no codec |
| 38 | `rvg_video_device_release` | | drops the host's reference; the last one resets the device |
| 39 | `rvg_video_device_start` | | the guest's DRIVER_OK: a new host session |
| 40 | `rvg_video_device_deliver` | may | one message of queue 0: its `subs` surfaces held while the session needs them, the session's status answered; `RVG_VIDEO_INVALID_SESSION` with no session; `RVG_VIDEO_ALLOCATION_FAILED` for a message over `RVG_VIDEO_MAX_MESSAGE_SIZE`, which the session never sees |
| 41 | `rvg_video_device_invalidate` | | a reset's first phase: the session sends the guest nothing more |
| 42 | `rvg_video_device_reset` | | both phases: the session's work drains and it ends |
| 43 | `rvg_scaler_device_create` | | the M2 scaler device (106b:1a0b), from an `RvgScalerDeviceConfig`, made on every host: transforming with the library's own kernel on the GPU it draws with (Metal's on macOS, the Vulkan platform's on Linux, any GPU, a software rasterizer included), and with none where that opens no device or the device refuses the kernel |
| 44 | `rvg_scaler_device_release` | | drops the host's reference; the last one drops its accelerator and ends its guest-memory table |
| 45 | `rvg_scaler_device_scales` | | whether the device has a GPU to transform on; where it has not, the host refuses the guest's features (FEATURES_OK), so the guest's driver does without the device |
| 46 | `rvg_scaler_device_start` | | the guest's DRIVER_OK: a new accelerator; false where none can be made, as on a device that does not scale, a needs-reset |
| 47 | `rvg_scaler_device_process` | may | one element of the queue, answered in place: `RVG_SCALER_REPLY`, `_NEEDS_RESET` or `_END_VM`, with Apple's reason for the last two |
| 48 | `rvg_scaler_device_reset` | | a reset or a stop: the accelerator goes |

QEMU's `apple-gfx` calls 21 of them (docs/qemu.md), the suite's runner 33 (all but `rvg_log_set_handler`, `rvg_frame_target_retain` and `rvg_frame_target_update_pixels`), and the C harness (`tests/c/harness.c`) all 48, the scaler's process and reset after a start only where the device scales (elsewhere it checks that the start fails). What the runner leaves out is covered in the C API's tests: `api.log.handler` pins the log hook, `frame::tests::a_retained_target_lives_until_its_last_release` checks that a retained target keeps its texture until its last release, and the harness that an update rewrites and flags only a changed row.

## 5. Callbacks and threads

### Where each callback arrives

| callback | thread | serialized with | order |
|---|---|---|---|
| `create_task`, `destroy_task`, `map_memory`, `unmap_memory` | a library thread (a FIFO's, a worker of the device's); or the host's thread inside a call that may call back (a release, `reset`, `finish_suspend`, `will_resume`) | nothing: calls from different FIFOs may overlap | core's; at teardown, the task table's order, ascending task ID |
| `read_memory` | as above | nothing | core's |
| `raise_interrupt` | the FIFO threads and the reporting thread; or inside a call that may call back | the device's signal lock: never concurrent for one device | core's |
| `add_trace_range`, `remove_trace_range` | the EFI display's present queue: its worker, or the host's thread inside a call that may call back (the MMIO write that starts or stops the EFI display, a release), where the queue's `sync` runs its block | the EFI display's present queue | add before the handler can run; remove once for each non-NULL range add returned, when presents from it stop, and never for NULL |
| `mode_change`, `new_frame`, `cursor_glyph`, `cursor_show`, `cursor_move` | the display thread: one serial worker for the whole process (after 5 s idle it exits, and the next job starts a new one) | every display callback of the process: one at a time | the order the displays queued them |
| a window's `pointer`, `wheel`, `key`, `close`, `scroll` | on macOS the process's main thread, where AppKit runs; on Linux the library's thread `reims-vgpu.ui`; never inside a call of the host's | every window callback of the process: one at a time | the order the events arrived |
| IOSurface `raise_interrupt` | the IOSurface device's own thread | itself | the ring's order |
| video `send_to_guest` | the host session's threads, or the host's own inside `rvg_video_device_deliver` | nothing: the host copies each message and keeps the calls' order for the guest | the session's |
| scaler `is_ram`, `read`, `write` | the host's own, inside `rvg_scaler_device_process` | the host's calls of `rvg_scaler_device_process` | the request's: Apple's checks, then its gathers, then its scatters |
| `destroy` (each table) | whichever thread drops the last reference: the host's in a release or a refused create, or a library thread; a window table's, the releasing thread, or the window's thread (the main thread on macOS, `reims-vgpu.ui` on Linux) | nothing | once, after every call of its table |
| the log handler | any | nothing | none |

Threads and queues are named `reims-vgpu.<role>`: `reims-vgpu.fifo` (a FIFO's thread), `reims-vgpu.fifo-control`, `reims-vgpu.reporting`, `reims-vgpu.efi-present`, `reims-vgpu.display`, `reims-vgpu.iosurface`, `reims-vgpu.release`, `reims-vgpu.timer`, `reims-vgpu.window` (a window's drawing thread), `reims-vgpu.ui` (on Linux, the thread every window's events and callbacks run on), `reims-vgpu.stats` (the frame statistics' reporter, with `REIMS_VGPU_STATS`), an isolated task's `reims-vgpu.task-service`, `reims-vgpu.task-connection` and `reims-vgpu.task-link`, and the video device's host session's, which its crate names `reims-vgpu-video.<role>`: `decoder`, `decoder-copy`, `encoder`, `motion`, `jpeg` and `imports` (Linux keeps 15 bytes of a name, `reims-vgpu-vide`). The names are diagnostics, not contract. A queue's worker, the display thread among them, has core's `WORKER_STACK`, 512 KiB.

### The order of VM-memory operations is core's

The C layer's closures call the host's function synchronously, on the thread core calls them on, with the arguments core passes. Core decides every:
- create, map, unmap and destroy, and every read;
- IOSurface use count;
- Metal object release.

So the order the host sees is core's. The suite's `blk` records check it, as do the pins' call orders (`api.calls.*`).

### The display thread

Every display is given the same executor in place of a host's queue:
1. **One serial executor per process** (core's `SerialQueue`, "the display thread"), made at the process's first display. Another handle is the same queue, and a display drops its handle at its unplug.
2. **Jobs run in the order they were queued.** Core queues a display's job at each point where it hands work to the display's queue. A job runs after every job queued before it, across displays and devices, a `recreate` included.
3. **`new_frame` coalesces** through core's `CoalescingSignal`, since the display thread is no dispatch queue: calls merge, and their count depends on timing (docs/testing.md 4.6).
4. **A job reads its handler when it runs.** A display released with a job queued dies at that job (`ModeChangeHandlerGone` and the others); `QueueGone` is reachable in the same race. Only a guest's cursor doorbell queues a job that does not hold its display, so a host cannot cause this without a guest (the unit test `display::tests::a_display_released_with_a_job_queued_dies`).
5. **The library never waits for the display thread.** No core path `sync`s onto a display's queue, and `SerialQueue`'s drop joins nothing. A display callback may therefore block, for example on QEMU's BQL, and may call any function. The next display callback waits behind it.
6. **A display with a window** has all five callbacks. Each runs on the display thread as above: it tells the window what the guest did, without waiting, then calls the host's where the host gave one.

Since every display has a queue, core's no-queue paths (`ModeChangeWithoutQueue`, a present with no queue) never run, and the guest's glyph size limit and cursor-handler bits depend only on the callbacks the display has: the ones the host gives, or all five with a window.

### The main thread

A window's callbacks run one at a time across every window of the process, in the order the events arrived, and never inside a call of the host's: on macOS on the process's main thread, where AppKit runs, and on Linux on the library's own thread `reims-vgpu.ui`. On Linux nothing of the library runs on or waits for the process's main thread: the windows open, change and close on `reims-vgpu.ui`, and what follows about the run loop and the application is macOS's.
- **They must not block.** Until one returns, no window of the process takes input or redraws its frame. The main thread is the host's too: QEMU parks it in `CFRunLoopRun()` under `-display none`, and a callback that waits for something that thread would do never returns. QEMU's queue their event for its BQL and return.
- **The library never waits for the main thread.** QEMU creates its displays on the main thread, holding the BQL, before that thread runs its run loop, so a create or a release that waited for it would hang. A window opens once the main thread runs its run loop, and a release closes it without waiting (the pin `api.window.no_run_loop`: a window display created and released in a process whose main thread runs no run loop, each table's `destroy` once).
- **What the library runs there.** Each window's opening, its mode and cursor changes, and its close are blocks the library posts to the main run loop, `CFRunLoopPerformBlock` in its common modes then `CFRunLoopWakeUp`, and they run in the order posted: under a plain `CFRunLoopRun`, under `[NSApp run]`, and while a window is resized live or a menu is open. The library never uses the main dispatch queue.
- **The application's start** is posted at the first window's attach, alone, and what is posted until the application has launched waits in the library, in order: the run loop runs the blocks queued on it as one batch, and a start that runs the application never returns (`crates/reims-vgpu-window/src/native/macos/main_thread.rs`). When the host has made no `NSApplication`, the start makes the library's own and calls `[NSApp run]`: from then on the run loop the host parked the main thread in runs inside it, the host's own blocks and sources with it, until the process exits. When the host has made one, the windows open in it and the start returns. What was held runs, in the order posted, once the library's application has finished launching, or at once in the host's.
- **A main thread that never runs its run loop** opens no window. What was posted to it stays queued, never run and never released; it holds the window at most weakly, so the release still ends the window and runs its table's `destroy` (`api.window.no_run_loop`).

### Callback tables and their destroys

Each table has one `destroy`, which runs when the last closure built from the table goes: at the end of the object for a table with functions, where core drops the last of its closures (`crates/reims-vgpu-c/src/table.rs`). The pins `api.destroy.*` record when, and on which thread.

Two cases have no closure to carry the table:
- a table with no function: its `destroy` runs at the host's release of the object;
- a create given no config, or refusing a config or table for its `struct_size`: it has not read the table, and runs no `destroy`.

Every other refused create runs the table's `destroy` on the caller's thread before it returns (`api.destroy.refused`). That includes a display refused after its new-frame source was made: the source's cancel frees that closure later on the display thread, and it is never called.

The object's handle holds the table too, so that a table with no function still has its `destroy`, and it lets go of the table before the object: with functions, the last reference is a closure's. A display's `destroy` so runs at the display's end, before the device's teardown where the display held the device's last reference, on the thread of the display's last release (`api.destroy.display_last_release_main`, `_thread`). A display with `new_frame` has its source's cancel drop that closure on the display thread, so its `destroy` runs on whichever of the two threads drops the last closure, possibly after the release has returned (`api.destroy.display_with_new_frame`).

A display with a window has two tables, the display's and the window's, and `rvg_display_create` reads both before either is the library's:
- a `struct_size` refusal of any of the four structs (the config, its callbacks, the window config, the window's callbacks) runs neither `destroy` (`window::tests::a_window_refused_for_its_size_runs_no_destroy`);
- every later refusal runs both before it returns, the window's first (`window::tests::refusals_end_both_tables`);
- the release closes the window before it lets the display go, so the window table's `destroy` runs first, once, on the releasing thread before the release returns, with functions or without, or, while a window callback runs on the main thread, once that callback returns; the display table's follows at the display's end, as above (`window::tests::a_release_ends_the_window_table_first`, `a_release_ends_the_window_table_once`, and the pin `api.window.no_run_loop`). A window display always has `new_frame`, so its display table's `destroy` may run on the display thread after the release has returned.

### Blocking and re-entry

- A task, memory or trace callback may block, but must not wait for a host thread that is itself inside a call on the same device that may call back: that call may be waiting for the callback's thread. A host whose callback waits for its main loop keeps that loop running while its other threads wait in the library. QEMU gives none of these callbacks, so its vCPU threads call the library directly (docs/qemu.md, "apple-gfx on reims-vgpu").
- `raise_interrupt` runs under the device's signal lock and must not wait for a host thread that is in a call on the device.
- A window callback must not block at all ("The main thread", above).
- The IOSurface raise must not release its own IOSurface device: the teardown's wait for the device's thread would run on that thread, and ends the process.

## 6. Where the API differs from Apple's SDK

The guest drivers were written against Apple's framework, and upstream QEMU's `apple-gfx` against its `ParavirtualizedGraphics` SDK (`PGDevice`, `PGDisplay`, `PGIOSurfaceHostDevice` and their descriptors). A host written against that SDK finds these differences:

| | Apple's SDK | this API |
|---|---|---|
| objects | descriptor objects, set property by property and read back | one config struct per object, borrowed for the create |
| callbacks | one block per handler | one table per object, with one `ctx` and one `destroy` |
| display callbacks | on the host's dispatch queue | on the library's display thread, in the order the displays queued them |
| NULL handles | nil messaging: 0, NULL, nothing done | refused, or fatal (section 1) |
| refusals | nil, with the reason only in a log line | an `RvgStatus` |
| errors | an `NSError` domain and code | a status, and an `RvgResumeError` code for a resume |
| the Metal device | the host's `MTLDevice` | chosen by registry ID |
| frames | the host's command buffer and texture, any format, with the display's minimum texture usage | the library's frame target and command buffer, four formats |
| cursor glyph | an `NSBitmapImageRep` | pixels and a hot spot (`RvgCursorGlyph`) |
| suspend state | `NSData` | `RvgBytes` |
| resume state | `NSData` | a pointer and a length |
| MMIO length | an input of the device and IOSurface descriptors | the constant `RVG_DEVICE_MMIO_SIZE` |
| protected content | a descriptor flag nothing reads | absent: displays always store the guest's protection options |
| the IOSurface device's map and unmap handlers | die for any argument | absent |
| `add_trace_range` without `remove_trace_range` | a death at creation | `RVG_ERR_INVALID_ARGUMENT` (question 1) |
| the option ROM | the framework's own, found through the API | not in the API: the dist tree carries it (docs/qemu.md, "The QEMU build") |
| inputs | — | `display_port_connection_types`, mode flags, log hook; zero keeps the SDK's behaviour |
| showing a display | the host draws the frames it encodes | also a window of the library's own for a display (`RvgDisplayConfig.window`), with its input as callbacks on the main thread (on Linux, the library's `reims-vgpu.ui` thread); NULL keeps the SDK's behaviour |

A host of this API cannot observe, and the suite does not record:
- the descriptor read-backs;
- an `NSError` domain;
- the glyph's `NSBitmapImageRep` properties;
- the option ROM's name and digest;
- the identity of the handlers and the queue a display was given;
- a Metal device of the host's own: a proxy, or one lacking capabilities (question 7);
- encodes into a texture of the host's choosing: another device, other usage, integer formats.

## 7. The suite's runner

`tests/suite/runner/pgrun.m` plays the hypervisor to the C API; docs/testing.md is its specification. It is Objective-C for its JSON, its hashing and its `metal` queries, and includes `reims_vgpu.h` and nothing else of the library's.

| op or piece | what it calls |
|---|---|
| `config` | Allocates guest RAM with `mach_vm_allocate`, and loads a handoff when there is one. `MTLCreateSystemDefaultDevice()` serves the `metal` queries and `d_gpu`; the library makes its own frame queue. It writes `binding` (`library`, `setup`, `d_gpu`, `d_pid`): `library` is `<build>` for the build under test (a library under the `--build` directory, or one linked into the runner), else its real path. With `iosfc` it makes the IOSurface device, before the device or, with `iosfc_after`, after it; then the device and its displays. |
| the device | Fills an `RvgDeviceConfig`:<br>- `port_count` goes to `display_port_count`, with 1 when absent;<br>- the `blocks` setup, or a `blocks` key, sets the task and memory functions, which refuse as the scenario's `refuse` says;<br>- otherwise, or with `map_too`, `map_ranges` or the RAM goes to an `RvgMemoryMap`, and `no_map` leaves it out;<br>- the `isolate` setup sets `process_isolation`;<br>- `iosfc` sets `use_iosurface_mapper`;<br>- `enableArgumentBuffers: false` sets `disable_argument_buffers`;<br>- `trace` sets the add and remove functions;<br>- `raise_interrupt` is set unless `no_raise`;<br>- `metal_registry_id` is 0, or `UINT64_MAX` with `no_mtl_device`.<br>The keys `create`, `device`, `readback`, `mmioLength` and `enableProtectedContent` are a `scenario_error`. A refused create's record carries `status`. |
| trace ranges | `add_trace_range` keeps a reference to the handler (`rvg_trace_range_handler_retain`), and `remove_trace_range` releases it. `trace_fire` invokes each handler whose range the write touches, through a reference of its own. |
| `new_display` | An `RvgDisplayConfig`: `port`, `serial` and `name` (UTF-8; absent is the library's default), `mm` (default 400 × 300), and a callbacks table whose `ctx` is the display's index and whose functions are NULL where `handlers` leaves them out. `window` stays NULL: the suite opens no window. `queues`, `no_queue` and a `nil` name are a `scenario_error`. `set_modes` follows with the modes. |
| display callbacks | Recorded under the SDK handler's name. `cursorGlyph` records `w`, `h`, `hot`, `sha256` and `hex` from `RvgCursorGlyph`. `cursorShow`'s function takes a `BOOL` and is registered through a cast to `RvgCursorShowFn`, so it boxes the byte the library passes. |
| `set_modes` | An `RvgDisplayMode` array (`flags` 0); `nil` passes none. |
| `display_props`, `cursorPosition`, `present_counts` | `rvg_display_cursor_position` and `rvg_display_present_counts`. For a display it released or that was refused, the runner answers itself: `ok: false` for `display_props`, zeros for the others. |
| `encode` | On a pool thread:<br>1. `rvg_frame_target_create` (`w` and `h`, default 64 × 32; `format`, default 80);<br>2. `rvg_display_encode_frame` over `region`, default the whole target;<br>3. with `hold`, keeps the frame, the target and the op; with `no_commit`, releases the frame uncommitted;<br>4. otherwise `rvg_frame_commit`, `rvg_frame_wait`, then `rvg_frame_target_read_pixels` at `w * 4` bytes a row, and releases the frame and then the target before the record.<br>`status` is the status's name. `usage` and `other_device` are a `scenario_error`. |
| `encode_commit` | Commits, waits for and reads a held frame, as step 4. |
| `api` lifecycle | One call each. `finishSuspend` keeps the `RvgBytes` in its slot, freed with the slot. `willResume` passes the bytes, or NULL for `nil`, or a one-byte static buffer with length 0 for an empty state, and passes no error pointer for `nil_error`. |
| `release`, `recreate`, teardown | Release the displays in index order, then the device; the teardown then releases the IOSurface device. `recreate` makes the device again over the same RAM. |
| `iosfc`, `iosfc_again` | An `RvgIOSurfaceDeviceConfig` with the RAM map and the raise function; `iosfc_no_map`, or `no_map` for `iosfc_again`, leaves the map out. `iosfc_again` makes a second IOSurface device and releases it at once. |
| MMIO | A pool thread makes the call while the scenario thread waits, and `g_callThread` names it. IOSurface writes are zero-extended and reads truncated to 32 bits, as QEMU declares them. |
| thread rules | Each callback checks its thread, and breaks the scenario with a `scenario_error` naming the rule when it is wrong:<br>- a display callback runs on none of the runner's threads, and never while another runs;<br>- a task, memory, trace or raise callback runs on a thread the runner did not create, or on one of the runner's inside a call into the library;<br>- an IOSurface raise runs on none of the runner's threads, and never while another runs. |

## 8. QEMU's `apple-gfx`

`vendor/qemu`'s `apple-gfx` is a host of this API, in C. [docs/qemu.md](qemu.md), "apple-gfx on reims-vgpu" and "Window mode", describes how it uses it: what it gives the device, its realize, its callbacks and the BQL, its MMIO, its frames and its windows.

## 9. Keeping the API whole

- **The header.** `cargo xtask header` regenerates `include/reims_vgpu.h` with cbindgen 0.29.4, and `--check`, part of `cargo xtask check`, fails when the committed header differs from cbindgen's output. A different cbindgen version is reported, not compared. The C API's `header_has_no_drift` test runs the same comparison when cbindgen 0.29.4 is installed.
- **The exports** (`tests/exports.rs`): every export is `#[unsafe(no_mangle)] pub [unsafe] extern "C" fn rvg_…` whose whole body is one call of `ffi::export`; no other function or function-pointer type of the product code has a C ABI, except the host's callback types, which are `extern "C-unwind"` for `ffi::host_call`'s catch; the header declares exactly the exports, and the dylib exports every one.
- **The C harness** (`tests/c_harness.rs`) builds `tests/c/harness.c` against the committed header with `-std=c11 -Wall -Wextra -Werror -pedantic`, links it to the crate's dylib and runs it. It checks that the header declares 48 functions and that the harness calls each, and that the header compiles alone as C++17 and as Objective-C.
- **The pins** (`tests/pins`): `api_pins.c`, a C program a host could be, pins what the suite's records do not show: the order, arguments and thread of the device's callbacks; each table's `destroy`; the display thread; a display kept alive through a frame; a display with a window created and released in a process whose main thread runs no run loop (`api.window.no_run_loop`); `add_trace_range` alone as a refusal; the registers a fresh device reads, a resume's refusals and a display's first cursor position; the EFI display's traced range; resume-child writes (0x1028) of IDs past the 64 child FIFOs; the feature level and the connection types as the guest reads them; the log hook; and the deaths that remain. `tests/pins/main.rs` compares its output with `expected.tsv`, and `RVG_PINS_RECORD=1` records it again.
- **The suite** drives the API through the runner (section 7), against its expected files (docs/testing.md).
- **Versioning.** A change to the API is additive (section 1).

## 10. Decisions

The API rests on these decisions.

| # | question | decision |
|---|---|---|
| 1 | Is `add_trace_range` without `remove_trace_range` a death at creation, or a refusal at the same point? It is host input, not guest input. | **A refusal**, `RVG_ERR_INVALID_ARGUMENT`: a status-returning create does not abort on its own config. The trace ID is still taken, so numbering is unchanged. |
| 2 | One display thread per process, or one per device? | **Per process.** Every display callback of the process runs in one order, across a `recreate` too, as it did under Apple's SDK for a host that gave every display its main queue, as QEMU did. Per device would let an old device's late callbacks interleave with a new device's. |
| 3 | Are `display_port_connection_types` and the mode flags in the first version? | **Yes.** They are core's inputs, zero is the default behaviour, and the features stay configurable. |
| 4 | Does QEMU make an MMIO access on a thread of its own (QEMU's `ThreadPool`, or libdispatch's plain-C `dispatch_async_f`), while the vCPU waits, or on the vCPU's thread? | **The vCPU's thread** (docs/qemu.md, "apple-gfx on reims-vgpu"): an access waits on nothing of QEMU's, so a hop buys nothing, and the vCPU's wait for a pool thread's completion hung a vmapple boot once (docs/lessons.md). |
| 5 | Where does the PCI device's option ROM come from? | **The dist tree.** The API names no ROM; `cargo xtask dist` builds it from `firmware/`, `reims-vgpu.pc` names it, and `apple-gfx-pci`'s default `romfile` is it (docs/qemu.md, "The QEMU build"). |
| 6 | Does QEMU keep the paths below macOS 15.4 (tasks, `read_memory`, the IOSurface map callbacks)? | **No.** They are dead at its deployment target, 26.0. The API still has the task and memory callbacks, and the suite's `blocks` setup covers them. |
| 7 | How are a Metal device's capabilities covered, when a host cannot hand the library a device lacking them? | **Core's unit tests**: `device::tests::a_device_without_the_capabilities_leaves_them_out_of_device_info` turns off detach backing, the address-range heaps and buffers, IOSurface buffers, shared texture handles, dynamic stride, image blocks and command buffer jump, and `a_paravirtual_device_keeps_argument_buffers_off` names the device as a paravirtual GPU. |
| 8 | Does an uncommitted frame that is released stay counted in flight, so that every later present waits its 2 s, as when a host of Apple's SDK drops an uncommitted command buffer? | **Yes**, and the header says so. Committing on release would change what a present waits for. |
| 9 | Do the runner's `encode` records keep Metal's numeric command buffer status, or the API's status name? | **The name.** The runner records what the API answers. |
