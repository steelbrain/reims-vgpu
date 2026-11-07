# Testing: locking the behaviour

This document specifies the behaviour-lock suite, the codebase's own record
of what it does. The suite writes down every nook and cranny of that
behaviour in its expected files, so that no change can alter it silently: a
tree reproduces every expected file its predecessor passed, and a change
meant to alter behaviour records the affected files again and commits their
diff for review (decision 3, section 5.7).

Besides the suite's design, this document holds what the suite is built
on:
- the guest ABI as the code implements it (section 6);
- every quirk the suite locks (section 7);
- what varies between runs and hosts (section 8).

The suite is [tests/suite/](../tests/suite/README.md), described in section
5.10. It drives the library through its C API, as QEMU does.

**Tags.** Claims are tagged:
- **measured**: run, and the text says how.
- **read**: from the code.
- **cited**: from an external source listed in [Sources](#sources).
- **unknown**: not established.

An honest unknown is worth more than a plausible guess here, for the same
reason AGENTS.md gives for names.

**Paths** are relative to the repository root, except that `runner/`,
`pgtest/`, `scenarios/`, `generated/`, `expected/` and `fixtures/` are the
suite's, under `tests/suite/`.

**Hosts.** A measurement that names a scenario was made by the suite, on
its host: an Apple M3 Max with a 40-core GPU, on macOS 27.0.1 (26A434),
whose files are the only host-dependent expected files
(`AppleM3Max-40c-26A434`, section 5.7). One that names no scenario was made
with a probe on an Apple M4 Pro (a 14-core CPU, a 20-core GPU,
`AGXG16SDevice`, Apple9 family, 16 KiB `vm_page_size`) on macOS 26.6.2
(25G83), built with Xcode 26.6 (17F113) and the macOS 26.5 SDK: Apple clang
21.0.0 (clang-2100.1.1.101), ld-1267 and Apple metal 32023.883.

## Contents

1. [The decisions](#1-the-decisions)
2. [The convention, and who else follows it](#2-the-convention-and-who-else-follows-it)
3. [The input surface](#3-the-input-surface)
4. [What to assert: the observation surface](#4-what-to-assert-the-observation-surface)
5. [The suite's design](#5-the-suites-design)
6. [Guest ABI reference](#6-guest-abi-reference)
7. [Quirks: odd behaviour the suite pins](#7-quirks-odd-behaviour-the-suite-pins)
8. [Variance and host dependence](#8-variance-and-host-dependence)
9. [Unknowns](#9-unknowns)
10. [Sources](#sources)

## 1. The decisions

These decisions are the specification; everything else in the suite
follows from them.

1. **Drive it as a hypervisor does, through the C API, as QEMU does.** A
   hypervisor is the library's audience, so the suite uses what a
   hypervisor can use, the way QEMU uses it, and nothing else.
   - The API is:
     - the library's C header, `reims_vgpu.h`
       ([api.md](api.md)): every declaration, with any argument value the
       header makes legal, whether or not QEMU calls it. That includes a
       `NULL` resume-error pointer, a `NULL` callback the header allows, no
       resume state and zero modes.
     - through it, the guest ABI: MMIO register writes and reads, and bytes
       laid out in guest memory
   - Nothing outside that is called, set or observed: no symbol the header
     does not declare, and no preferences written (section 3.2).
   - The calls follow a hypervisor's contract, as QEMU keeps it:
     - MMIO reads and writes run on a pool thread while the scenario
       thread waits, where QEMU makes them on the vCPU's own thread
       (`apple_gfx_read` and `apple_gfx_write`,
       `vendor/qemu/hw/display/apple-gfx.c`; [qemu.md](qemu.md),
       "apple-gfx on reims-vgpu"): either way the call is made by a host
       thread in no other call on the device, and it may block on the
       library's threads and call back on its own;
     - guest RAM comes from `mach_vm_allocate` and is described to the
       library before the device is created;
     - display callbacks arrive on the library's display thread
       ([api.md](api.md) section 5), and the runner checks each callback's
       thread (section 5.10);
     - a frame is encoded on a pool thread after the new-frame callback,
       into a frame target the library makes, and the host commits it and
       waits for it. QEMU encodes one when its display system asks for the
       console's surface, and, without window mode, follows the new
       frames, one at a time, while a display change listener or a
       screendump has asked within the last 6 s (`apple_gfx_watched`,
       `apple_gfx_render_new_frame` and `apple_gfx_do_render`,
       `apple-gfx.c`).
   - Test code includes nothing from the implementation (the crates). It
     includes the C API's header, as QEMU does.
2. **Assert machine behaviour**, meaning what the guest or the host can
   observe:
   - register values
   - guest memory contents
   - interrupts
   - calls out through the host's callbacks
   - display callbacks and pixels
   - persisted state
   - whether a run completes, faults, crashes or hangs

   No interaction tests ("call X, assert X was called"), no mocks of
   internals, and no golden logs of which Metal calls were made. Metal's part
   is asserted through what the GPU produced (section 4.10).
3. **The codebase is its own reference.**
   - Expected values are recorded from the codebase's own build by a person,
     reviewed, and committed (section 5.7). A tree reproduces every expected
     file its predecessor passed.
   - A change meant to alter behaviour records its cases again with
     `pgtest bless --update`, and commits the diff of the expected files
     with the change, where it is reviewed.
   - The suite describes this codebase, not Apple's framework, and never
     runs against Apple's framework: its runner drives the C API, which
     Apple's framework does not have. That framework appears in this
     document as the origin of the behaviour guests rely on (section 7).
4. **Lock everything, exactly, with no exceptions.** What the codebase does
   is what every later commit does, unless a change alters it on purpose
   (decision 3):
   - A happy path never fails.
   - A refused, faulting or asserting input never becomes accepted. A guest
     command that crashes the host (QEMU) crashes it under every later
     commit, and at the same op.
   - A hang stays a hang.
   - Things happen in the same order (decision 8).

   Nothing changes silently: there is no list of accepted divergences and
   no expected-different mark. Any difference from an expected file is a
   regression, unless the change that makes it re-records the file on
   purpose and its diff is reviewed. A quirk (section 7) is behaviour like
   any other: changing one is a change of behaviour, never a fix slipped
   in. An expected file also changes when its scenario's input changes.

   Diagnostics may change: log text, assert and exception messages, and
   which signal ends a process that dies (section 4.9). The outcome may not:
   that the process died, which process, and at which op.

   Nothing is dropped because QEMU does not use it: every capability of the
   public API is kept and tested.
5. **GPU output and the host.**
   - The suite takes GPU output to be portable across Apple GPU
     generations ("2 + 2 is always 4"), its quirks and rounding included.
     Whether it is stays **unknown** until another generation runs the
     expected files (section 9).
   - Expected values are keyed to the GPU generation all the same: the
     scenarios marked `host` are keyed by GPU model, core count and macOS
     build (section 5.7).
   - Values that *report the host* rather than compute something are keyed to
     the exact chip configuration, computed from Metal inside the test, or
     asserted as relations (section 5.8). Examples:
     - device info (the GPU core count is one of its keys)
     - heap size and alignment
     - GPU addresses and resource IDs
6. **Prove adequacy.** Coverage (`make coverage`, section 5.10) says what
   ran; a mutation pass, when one is run, says what the suite would
   notice. A surviving mutant that changes something a hypervisor can
   observe gets a scenario that kills it (the `mutgap*` modules, section
   5.10). No mutation tool is part of the tree.
7. **Deterministic by construction.** A case waits for what the guest can
   see, never for a fixed time, and records only what its waits fixed.
   Runs of a case that differ are a scenario bug, never a reason to loosen
   a comparison (sections 5.4 and 5.7).
8. **Lock order.** Every order the implementation establishes is behaviour:
   - stamps publish in command order;
   - a stamp is in guest memory before its interrupt is raised;
   - one interrupt source is raised before another, as with a display's
     interrupt before the stamp of the command that raised it;
   - host callbacks happen in order: `create_task` before that task's
     `map_memory`, `unmap_memory` before `destroy_task`;
   - display callbacks arrive in order on the library's display thread;
   - GPU work runs in the order the guest encoded it, which the results in
     memory show.

   What timing alone decides is not an order, and is never asserted: which
   runner op an interrupt arrived during, how many `new_frame` calls
   coalesced into one, or whether a stamp was visible before the host
   had finished raising its interrupt. Section 5.4 has the measurements and
   the rules.
9. **No VM boots.** The suite never boots a guest OS, not even to record
   traffic. The harness plays the guest and the host, and every scenario is
   built from the guest ABI (section 6).

## 2. The convention, and who else follows it

The pattern has a name: **device-level characterization testing**.
- The test plays the guest driver, and the harness plays the VMM.
- Expected values are the current implementation's, recorded, reviewed and
  locked.
- Characterization tests "document your system's actual behavior, not check
  for the behavior you wish your system had" (Feathers, **cited**).
- Approval testing adds the rule that a person approves each new expectation,
  and that CI never does.

The closest analogues found (**cited**; URLs in [Sources](#sources)):

- **QEMU qtest and libqos** are the model for the mechanics.
  - A test writes registers (`writel`) and guest memory (`memwrite`), and
    reads back registers, memory and interrupt lines (`irq_intercept_in`).
  - Timer devices step a virtual clock. Work on other threads is awaited by
    polling what the guest can see, with a deadline: `qvirtio_wait_used_elem`
    polls the used ring.
  - Commit 92cb8f8b (2025) removed clock stepping from those waits for exactly
    that reason.
  - `vmapple` cannot run under qtest, because it needs the `host` CPU (HVF
    or KVM). Upstream has no test harness for apple-gfx.
- **rust-vmm's `MockSplitQueue` and Firecracker's device tests** put the
  rings in real guest memory and assert used-ring entries, status bytes and
  pending interrupts.
- **crosvm's `backcompat.rs`** is a set of golden-file "change detector"
  tests: "make sure the change is intended".
- **virglrenderer** is the closest in architecture: a host GPU library that
  VMMs embed, driven with no VM over `vtest`. It keeps checksums of trace
  frames per device.
- **gfxstream** runs the guest driver against its `kumquat` server with no
  VM, and compares against golden PNGs with a tolerance of 2 per channel.
- **Dolphin's FifoCI is the closest analogue overall.**
  - It locks a GPU emulator against its own previous commit.
  - Its input is exactly the guest-to-GPU interface: register state, FIFO
    bytes and memory updates, in a `.dff` file.
  - It asserts SHA-1s of the decoded frames, per backend, OS and driver,
    including Metal on macOS.
  - Its gaps are what this suite adds:
    - it checks only frames, not registers or memory;
    - it is advisory, not gating.
- **Mesa CI.**
  - Per-device MD5s of trace frames. Trace files are immutable, and a
    changed trace is renamed `_v2`.
  - A developer blesses locally with `update_traces_checksum.py`; CI never
    does.
  - `deqp-runner` fails the job on an unexpected pass.
- **Chromium's Skia Gold** keys images by GPU and driver. It prefers several
  approved images to fuzzy matching, which risks "causing a test to
  erroneously pass".
- **typescript-go** records every difference from the original compiler as
  a committed `.diff`, filed as accepted (with a reason) or triaged (with an
  issue). This suite does not adopt that: it keeps no list of accepted
  divergences (decision 4).
- **Ports that had to keep behaviour exactly:**
  - rav1d ran dav1d's test data and the Argon conformance streams by MD5 on
    every commit. Its successor added reference MD5s from AOM, because
    "matching dav2d cannot catch a bug faithfully ported from the C".
  - zlib-rs compares against zlib-ng in one binary and fuzzes the difference.
  - sudo-rs runs every test against both implementations
    (`SUDO_UNDER_TEST=ours|theirs`) and fails CI if a test marked as
    failing starts passing.
  - fish 4.0 ported one component at a time.
  - uutils runs GNU's suite, and still shipped a `date -r` regression that
    GNU's suite did not cover either.
  - librsvg's reference-image tests left 889 of 5,243 mutants uncaught when
    mutation testing was run.
- **SQLite** has 100% branch and MC/DC coverage, plus mutation checks, and
  calls coverage "a test of the test". Google shows surviving mutants in code
  review (Petrović et al.).

**Conventions this suite adopts from them:**
1. The test plays the guest, and asserts only what a guest or host sees.
2. Inputs are immutable, versioned data. Expected values are committed data,
   reviewed like code.
3. Exact comparison, keyed by configuration where the host decides a value
   (decision 5). No tolerance.
4. A person blesses expectations, locally, recorded from the codebase's own
   build and reviewed; CI never writes one.
5. No divergence is accepted silently: a change of behaviour is made on
   purpose, re-recorded and reviewed (decisions 3 and 4).
6. Asynchrony is handled by waiting for what the guest can see, with a
   deadline, never by sleeping.
7. The same suite runs against any implementation of the API: the runner
   loads whichever build of the library it is given (section 5.1).
8. Adequacy is measured by mutation testing, not coverage alone.

**Anti-patterns the same projects warn against:**
- blessing in CI
- snapshots too large for anyone to read
- hash-only goldens that hide what changed
- fuzzy thresholds
- bare sleeps
- treating the old output as correct without marking the suspect parts
- asserting that something failed without asserting how

## 3. The input surface

### 3.1 The host API

**The C API**, `reims_vgpu.h` ([api.md](api.md)): 35 functions. The last
column says what QEMU's `apple-gfx` (`vendor/qemu/hw/display/apple-gfx*.c`)
does with each item (**read**).

| item | QEMU |
|---|---|
| `rvg_device_create` with `display_port_count`, `metal_registry_id` 0, a memory map and `raise_interrupt` | used |
| `RvgDeviceConfig.use_iosurface_mapper`, `.process_isolation` | set by `apple-gfx-mmio` and `apple-gfx-pci` always |
| `RvgDeviceConfig.process_isolation_follows_guest` | set by `apple-gfx-pci` always: it has no option for isolation or the mapper, and adding an `apple-gfx-iosurface-pci` is the switch |
| `RvgDeviceConfig.feature_level`, `.display_port_connection_types`, `.disable_argument_buffers` | unused |
| `RvgHostCallbacks`: `create_task`, `destroy_task`, `map_memory`, `unmap_memory`, `read_memory` | unused: QEMU gives a memory map |
| `add_trace_range`, `remove_trace_range`, `rvg_trace_range_handler_invoke`, `_retain`, `_release` | unused |
| `rvg_device_mmio_read`, `_mmio_write`, `_reset`, `_release` | used |
| `rvg_device_pause`, `_unpause`, `_stop`, `_will_suspend`, `_finish_suspend`, `_will_resume`, `_did_resume`, `rvg_bytes_free` | unused (QEMU sets a migration blocker) |
| `rvg_display_create`, `_set_modes`, `_cursor_position`, and the five display callbacks | used, one display a port (the `displays` property) |
| `RvgDisplayConfig.window`: `RvgWindowConfig` with `RVG_WINDOW_MATCH_HOST_SCREEN`, and the four `RvgWindowCallbacks` | used in window mode (under `-display none`); the runner passes no window |
| `rvg_display_release` | on a failed realize only |
| `rvg_display_present_counts` | unused |
| `rvg_frame_target_create`, `_retain`, `_release`, `_read_pixels`; `rvg_display_encode_frame`; `rvg_frame_commit`, `_wait`, `_release` | used, with BGRA8Unorm targets only |
| `rvg_iosurface_device_create`, `_mmio_read`, `_mmio_write` | `apple-gfx-mmio`, and `apple-gfx-iosurface-pci` |
| `rvg_iosurface_device_release`, `rvg_log_set_handler` | unused |
| `rvg_status_message` | used |
| `RVG_PCI_*` (vendor 0x106b, device 0xeeee, BAR 0, one MSI vector), `RVG_DEVICE_MMIO_SIZE` (0x4000) | used |

**The IOSurface host device's registers** are 64 bits wide:
`rvg_iosurface_device_mmio_read` returns a `uint64_t` and `_write` takes one.
QEMU's `apple-gfx-mmio` reads and writes them as 32 bits: writes are
truncated, and reads truncated, then zero-extended (`iosfc_read` and
`iosfc_write`, `apple-gfx-mmio.c`, **read**). `apple-gfx-iosurface-pci`
passes an 8-byte access whole, and Virtualization.framework, below, uses
the full 64 bits. The runner truncates as `apple-gfx-mmio` does, so the
upper halves are out of its reach: the memory map's refusals of a ring too
large to reserve and of a base plus length that overflows are one 64-bit
write each, which core's unit tests make (`iosurface/regs.rs`).

**What the suite covers.** Every function of the header, with any argument
value it makes legal (decision 1), whether QEMU calls it or not: suspend,
resume, pause, stop, the trace ranges, the task and memory callbacks,
several displays, and the IOSurface host device. The runner never sets
`feature_level`, `display_port_connection_types` or a mode's flags, and never
calls `rvg_frame_target_retain` or `rvg_log_set_handler` (**read**,
`runner/pgrun.m`): every case runs with their zero defaults, and without a
log handler. The C API's pins and unit tests cover those
([api.md](api.md) sections 2 and 4).

**Virtualization.framework (VZ)**, the other host of Apple's framework
(**measured**: the imports and `objc_msgSend` stubs of
`Virtualization.framework/Versions/A/XPCServices/com.apple.Virtualization.VirtualMachine.xpc`,
read function by function, macOS 26.6, 25G229), is the only client besides
QEMU that drives a device; ParavirtualizedANE only calls
`+[PGIOSurfaceHostDevice ioSurfaceHostDevice]` and
`-retainSurfaceWithMappingID:`. It shows what a host other than QEMU asks
of the device:
- **What the C API carries of it:** the feature level, which VZ passes as
  `(n - 1 < 4) ? n : 0` for a configured level `n`, so 0-4 and never the
  fatal 5 or more (`feature_level`); and the port connection types
  (`display_port_connection_types`), with the modes' EDR and 10 bpc flags.
- **What it has no counterpart for:**
  - on the device, `saveState` and `resetWithMemoryMap:`;
  - on the display, `unplug`, `guestColorSpace`, `protectionRequirements`,
    `protectionOptions` and `setProtectionOptions:`, and on its descriptor
    `connectionType`, `scaleFactor`, `optIntoUsingUnplugMethod` (always
    YES), `displayUUID`, and the handlers `cursorGlyphHandler2`,
    `sleepHandler` and `modeListResponsivenessChangeHandler`;
  - on the IOSurface host device, `reset`, `resetWithMemoryMap:`,
    `saveState`, `restoreWithSavedState:error:`, and, from VZ's other
    devices and ParavirtualizedANE, `+ioSurfaceHostDevice` and
    `retainSurfaceWithMappingID:`.
- **Suspend without the will-suspend and finish-suspend pair.** VZ saves
  with `pause` then `saveState`, which returns the state and tears nothing
  down, so the VM can `unpause` and run on. It restores in a new process:
  the device, every display, then the will-resume call with no error
  pointer, and did-resume when the VM runs.
- **Calling contract.** GPU MMIO writes are posted (`dispatch_async_f` to
  one serial queue) and reads are `dispatch_sync_f` on it, where QEMU blocks
  the vCPU on each access; the device sees the same order. Each display
  has its own serial queue and no `name`. A frame is encoded on the
  display's queue, inside the new-frame handler, into BGRA8Unorm,
  BGR10A2Unorm or RGBA16Float, and committed and waited for at once. A
  host resize unplugs the display and creates a new one on the same port.
- **One port.** Shipping VZ refuses a display port count above 1, unless
  the OS variant has internal content.

The suite calls only what the C API declares, so none of this is locked.

### 3.2 Hidden inputs

The C API is not the only input. The suite has to pin these, or key its
expected values by them.

- **Preferences.** The library reads no preferences domain
  (`platform/os/prefs.rs`), and behaves as with every key absent. Apple's
  framework read CFPreferences, domain
  `com.apple.gpusw.ParavirtualizedGraphics`: `DisableTaskService`,
  `DisableSharedMemoryHeap`, `EnableArgumentBuffers`,
  `EnableResourceDetachment` and `ForceUnrestrictedDeviceFeatureLevel` at
  device creation, the display's nits (`PGDisplayMinNits`,
  `PGDisplayMaxNits`, `PGDisplayMaxSDRNits`) and a mode's
  `PGDisplaySupportsEDR` and `PGDisplaySupports10bpc`, all of which reached
  the guest; on a Mac without the domain (**measured**: `defaults read`) it
  read every key absent too. The display's settings stay reachable through
  the Rust API's `Preferences` (`DisplayConfig::apply_preferences`,
  `DisplayMode::with_preferences`), and EDR and 10 bpc through the C API's
  mode flags. The runner refuses to run while the domain exists (exit 67).
- **The host GPU.** Metal capability queries, some of them SPI, reach the
  guest:
  - `supportsSharedTextureHandles`, `supportsBufferWithIOSurface`,
    `supportsBufferWithAddressRanges`, `supportsHeapWithAddressRanges`,
    `supportsResourceDetachBacking`, at device creation
  - device limits
  - pixel-format info
  - the device name: "Apple Paravirtual device", which is what a macOS VM's
    GPU is called, turns argument buffers off
- **The shaders.** The display's present shaders and the XR10 paging
  kernels are the library's own, in Metal Shading Language
  (`crates/reims-vgpu-metal/src/gpu/shaders.metal`), compiled from source
  once per Metal device at run time. The library reads no Metal library from
  a file.
  The functions keep the names the display and paging code bind:
  - `displayPresentVertex`
  - `displayPresentFragment`
  - `displayPresentFragmentWithGammaTableAndColorMatrix`
  - `BlitInRGB_2P_XR10_A8`, `BlitOutRGB_2P_XR10_A8`

  Every displayed pixel is computed by them and by the host's Metal
  compiler and GPU, so the frames the suite locks stay keyed by the host
  (section 5.7).
- **The option ROM** is built from `firmware/` by `cargo xtask dist` into
  the dist tree ([architecture.md](architecture.md) 11.4), for QEMU's
  `apple-gfx-pci`. The library never reads it, and the suite never uses
  it.
- **Process-global state.**
  - Display serial numbers are reserved process-wide.
  - The last IOSurface host device created in the process is the one every
    device uses.
  - The frame queue: one `MTLCommandQueue` per Metal device, made at the
    first `rvg_device_create` given that device and kept for the life of
    the process ([api.md](api.md) section 2).
  - The display thread: one serial worker runs every display callback of
    the process.

  So: one device per process, and unique serial numbers.
- **What the code reads from its environment** (**read**; [api.md](api.md)
  section 2, "Settings and the environment", lists each variable):
  `REIMS_VGPU_LOG`, `REIMS_VGPU_STATS` and `REIMS_VGPU_TRACE` (with
  `REIMS_VGPU_TRACE_DEGRADE`), diagnostics that change nothing the guest
  or host observes; the `REIMS_VGPU_WINDOW_*` variables, which
  only a display with a window reads, and the runner makes none; and the
  Vulkan platform's `REIMS_VGPU_STRICT`, `REIMS_VGPU_EXACT` and
  `REIMS_VGPU_CENSUS_DIR`, which the Metal platform never reads. Nothing
  reads `NSUserDefaults` or `sysctl`, and there are no `@available` or
  OS-version branches. Whether a host gives a
  memory map or the task callbacks is the host's choice: QEMU's
  `apple-gfx` always gives a map.
- **The host's boot.** Some of Metal's output changes with each boot of
  the host, which the suite can neither pin nor key its files by; a
  scenario keeps it out of what it observes (section 8).

### 3.3 Host setups

Three configurations reach three different task implementations. Each is a
test dimension: several inputs behave differently in each (section 7).

| setup | how the host describes guest memory | tasks other than 0 | task 0 | QEMU |
|---|---|---|---|---|
| blocks | the task and memory callbacks: `create_task`, `map_memory`, `unmap_memory`, `read_memory`, `destroy_task` | in process, over host mappings | same | none: QEMU always gives a memory map |
| map | `RvgDeviceConfig.memory_map` | in process, over a `mach_vm_remap` of the ranges | same | `apple-gfx-pci`, for a task the guest defines with no descriptor table (the isolate row) |
| isolate | `memory_map` and `process_isolation` | one service per task, on threads of its own in the host's process (`tasks::isolation`) | in process, as in map | `apple-gfx-mmio`; `apple-gfx-pci`, for a task the guest defines while its `apple-gfx-iosurface-pci` has a descriptor table (`process_isolation_follows_guest`; otherwise, and with no such device, as in map) |

The runner records each task or memory callback as a `blk` record named
`createTask`, `mapMemory`, `unmapMemory`, `readMemory` or `destroyTask`.

**Device creation.** `rvg_device_create` refuses, with the status named,
when (**read**; the memory-map cases **measured**; the order is api.md
section 1's):
- the config or its callback table is NULL where one is needed, too small,
  or larger with non-zero bytes the library does not know
  (`RVG_ERR_INVALID_ARGUMENT`, `RVG_ERR_UNSUPPORTED`), or gives
  `add_trace_range` without `remove_trace_range`
  (`RVG_ERR_INVALID_ARGUMENT`; `device_create_trace_add_only`);
- the display port count is 0 or above 8 (`RVG_ERR_DISPLAY_PORT_COUNT`);
- no Metal device has `metal_registry_id` (`RVG_ERR_NO_METAL_DEVICE`);
- a memory map is given together with any task or memory callback
  (`RVG_ERR_MEMORY_CONFLICT`);
- the memory-map ranges overlap, or a range's address plus length
  overflows (`RVG_ERR_MEMORY_MAP`);
- the blit pipelines fail (`RVG_ERR_METAL`);
- isolation is on without a memory map (`RVG_ERR_ISOLATION_WITHOUT_MAP`).

It succeeds with no callbacks and no map at all, and the first nonzero
root-page write then calls the absent `create_task`: a call to an absent
callback the device needs ends the process there (`host/callbacks.rs`;
**measured**, `device_create_nothing_dies_on_root_page`). Without
`raise_interrupt` the device's first interrupt ends the process
(**measured**, `device_create_no_raise_interrupt`). The suite locks the
refusals a host can cause through the API in `device_create`: all but the
blit pipelines'.

**What isolation needs.**
- The memory map's ranges must be whole pages at non-NULL, page-aligned
  host addresses: the service reads guest RAM through the host's own
  mapping of those ranges. With any other map, a define of a task other
  than 0 makes no task.
- One service runs per guest task, on threads of its own in the host's
  process. A task's service ends when the task is deleted.
- A task's objects come from the host's Metal device, so their GPU
  addresses, resource IDs and resource indices follow everything the
  process made before them, the host's and the other tasks' objects
  included.
- A death in a task's own code ends that task only, and writes one line,
  `reims-vgpu: isolated task died: ...`, on stderr. A crash inside Metal
  ends the host process, as it does with isolation off, and leaves a `.ips`
  report in `~/Library/Logs/DiagnosticReports`.

## 4. What to assert: the observation surface

Everything below can be observed from outside the library, and all of it
is to be asserted. Section 8 says which parts vary between runs or hosts.

### 4.1 Return values

- **`rvg_device_create`**: a status for the reasons in section 3.3.
- **`rvg_display_create`** refuses, in this order: a NULL or ill-sized
  config (`RVG_ERR_INVALID_ARGUMENT`, `RVG_ERR_UNSUPPORTED`); a port not below
  the device's count (`RVG_ERR_PORT_OUT_OF_RANGE`); a failed pipeline cache
  (`RVG_ERR_METAL`); a serial number already reserved *anywhere in the
  process* (`RVG_ERR_SERIAL_IN_USE`); a port another display holds
  (`RVG_ERR_PORT_IN_USE`).
- **Frames.**
  - `rvg_frame_target_create` refuses a size outside 1-16384 and any format
    but RGBA8Unorm (70), its sRGB form (71), BGRA8Unorm (80) and its sRGB
    form (81), with `RVG_ERR_INVALID_ARGUMENT`
    (`covgap_dev_encode_integer_target_dies`: RGBA8Uint, 73, before and
    after a present).
  - `rvg_display_encode_frame` refuses a target on another Metal device
    (**read**: the runner has one). Otherwise it succeeds, even when there
    is no frame yet: it then clears the target to (0,0,0,1)
    (`display_encode_refusals`).
  - `rvg_frame_commit` of a committed frame, and `rvg_frame_wait` of an
    uncommitted one, are `RVG_ERR_FRAME_STATE`; a wait whose GPU work ended
    with an error is `RVG_ERR_GPU`.
- **The display's reads:**
  - `rvg_display_cursor_position` is an atomic load of the shared page's
    u32 at +0xe00, split into {x, y}; with no shared page it is
    {0xffff, 0xffff}.
  - `rvg_display_present_counts`: the guest count goes up by 1 per guest
    present that reaches the host, and the host count by 1 on the first
    encode after each new frame.
  - A NULL name is "Apple Virtual".
  - `rvg_display_set_modes` with no modes changes nothing.
- **Suspend and resume** (section 6.24):
  - `rvg_device_finish_suspend` writes the state blob, or returns
    `RVG_ERR_SUSPEND`.
  - `rvg_device_will_resume` returns `RVG_ERR_RESUME` and writes the
    reason, an `RvgResumeError` from 0 to 5, or `RVG_RESUME_ERROR_NONE`
    where the device stored none.

### 4.2 MMIO reads

The register map is section 6.2. A fresh device reads 1 at 0x122c and 0
everywhere else in 0x0-0x3ffc (**measured**, isolation off and on).

### 4.3 Writes into guest memory

This is the complete list found in the code (**read**, and **measured**
where noted). The last column marks the kind of value: D is deterministic
given the inputs, T depends on timing, H on the host GPU.

| # | what | trigger | kind |
|---|---|---|---|
| W1 | stamp: u32 at the FIFO base page + 4 × stampIndex. Only the last contiguously completed stamp is published | command completion | value D, timing T; **measured** |
| W2 | child FIFO node: readCount +4, faultOffset +8 (UINT32_MAX when healthy), running (u16 at +0xe) | commands on that FIFO, faults, start and stop | T |
| W3 | DeleteChildFIFO zeroes the u32 at root + 0x410 + id × 0x14, which is +0x10 (pageListPage) of node **id+1** | opcode 49 | D; **measured** |
| W4 | root node word[taskID] = task root page, or 0 on delete | opcodes 56, 32 | D; **measured** |
| W5 | root node word[0x240 + port] = shared state page, or 0. The clear on detach happens only while the device is running | opcode 1, detach, stop | D; **measured** |
| W6 | task root +0x14 = task length, in process and in the service | task setup | D; **measured** |
| W7 | object and placement list offset and length in the task root; zeroed when validation fails; the old placement list copied on relocation | opcodes 51, 68 | D |
| W8 | placement entries: resourceIndex and gpuResourceID for textures, pipelines, samplers and ICBs; gpuAddress for buffers and heaps when the guest left it 0; zeroed on delete. Service tasks only | object creation and deletion | H |
| W9 | device info: u32 key/value pairs, then a {0,0} terminator | opcode 58 | H; **measured** |
| W10 | compute info: keys 1, 3, 4, in process and in the service | opcode 59 | H; **measured** |
| W11 | heap texture size and alignment: 16 bytes, written only on success, in process and in the service | opcode 64 | H; **measured** |
| W12 | info-encoder replies, written by the CPU at decode time (section 6.17); also render 0xa4 and `copyParameterDataToBuffer` | segments of encoder type 4 | H; **measured** under isolation |
| W13 | ICB write-back: command types, pipelines, buffers, draw arguments. Service tasks only: in-process the writable cursor aborts the host (section 7.4) | synchronize, eviction | H; **measured** (`icb_sync`) |
| W14 | copied resources written back: private buffers, legacy heaps, normal textures (formats 550 and 551 through compute kernels), backings | the synchronize flag on exec resources, opcodes 53 and 62, and the flush in `rvg_device_finish_suspend` | H, T |
| W15 | direct GPU writes into memory wrapped without a copy: shared buffers, textures over them, visibility results (section 4.10) | draws, dispatches, blits | GPU output; **measured** |
| W16 | display shared state page (section 6.21): init fields, the ack state machine, raisedInterrupts, mode list and settings, present count, presented surface or mapping, gamma and colour matrix indices, sleep, glyph record, cursor show | display commands, `rvg_display_set_modes`, presents, the doorbell | D, except present-complete (T); **measured** |
| W17 | IOSurface host device: ring entry status (+8); the descriptor's `mapped` bit; the test command writes i × 8 into each 8-byte word of the surface | ring commands | D |

**Not in guest memory.** The root FIFO's read counter and fault offset are
host atomics, read through MMIO 0x100c and 0x103c (**read**). No host virtual
address was found written into guest memory (**read**, not exhaustive).

### 4.4 Interrupts

Every interrupt the device raises goes through one path:
- The vector is always 0. The suite's runner reports vectors only when one
  is not 0 (`irq_vectors`), so every expected file locks this without
  carrying it.
- It is raised only while the device is running (run state 0). While
  paused or suspending, the status bits accumulate silently.
- `pause` quiesces, signals and drains before it sets the paused state, so
  stamps the drain completes, faulted heads included, are published with
  their interrupts during the `pause` call itself (**measured**,
  `device_lifecycle_paused_interrupts`: two interrupts in the call, none for
  resumes while paused, one at `unpause`).

The sources:
- **A stamp**:
  - the stamp value is stored, then `1U << stampIndex` is ORed into the
    read-to-clear mask at 0x1018, then the interrupt is raised, all under
    one lock;
  - the shift is 32-bit (**measured**), so stamp indices 32-63 alias bits
    0-31;
  - the index check is `stamp > maxFIFOCount`, not `>=`.
- **A FIFO fault**: asynchronous, on the device's FIFO control queue. It sets
  bit stampIndex of the fault mask (a u64; 0x102c reads it as a boolean),
  then raises the interrupt.
- **A display**: `1 << port` into the read-to-clear mask at 0x1014, raised
  only if the guest enabled the matching bit in its shared page.
- **Resume**: writing 0x1024 or 0x1028 raises again if other faults remain.
- **Lifecycle**: `unpause` and `rvg_device_did_resume` always raise one.
- **The IOSurface host device**: one interrupt per ring command, through
  its own `raise_interrupt`.

The interrupt count for a given sequence is deterministic only at checkpoints
that waited for it (section 5.4).

### 4.5 Calls out through the host's callbacks

The task and memory callbacks are called only in the blocks setup; with a
memory map, only `raise_interrupt` and, when the host gives them, the trace
range callbacks are (**read**). The runner records them as `blk` records
(section 3.3).

- **`create_task`**:
  - for the root task (0x1000000 bytes), on a nonzero write to 0x101c;
  - for each guest task, on opcode 56;
  - for the EFI framebuffer.
- **`destroy_task`**: the matching teardowns.
- **`map_memory`**:
  - the root task's device mappings, with `read_only` false;
  - guest pages, lazily;
  - EFI, with `read_only` true.

  Runs are coalesced wrongly (section 7): at most 2 pages per call, and a run
  flushed mid-walk takes the writability of the *next* page (**measured**).
- **`unmap_memory`**: a false return raises, and the command faults.
- **`read_memory`**:
  - at most 16 KiB, and never across a 16 KiB page;
  - a false return writes nothing: an in-process copy keeps the zeros of
    its buffer, and a page-table walk the entries it held (**read**,
    section 7.5).
- **`add_trace_range` and `remove_trace_range`**:
  - EFI uses them to present on guest writes, and the host calls the
    handler it was given (`rvg_trace_range_handler_invoke`, the runner's
    `trace_fire`);
  - without them it falls back to a 100 ms timer;
  - `add_trace_range` without `remove_trace_range` is refused at creation
    (`RVG_ERR_INVALID_ARGUMENT`), and `remove_trace_range` alone is never
    called.

### 4.6 Display

Every display callback runs on the library's display thread: one serial
worker for the whole process, which runs the jobs the displays queue in the
order they queued them ([api.md](api.md) section 5). The runner checks
each callback's thread (section 5.10), and records it as `newFrame`,
`modeChange`, `cursorGlyph`, `cursorShow` or `cursorMove`.

- **`new_frame`**:
  - it goes through a coalescing signal (`rt::signal::CoalescingSignal`),
    so calls coalesce, and their count depends on timing;
  - triggers: present opcodes 6, 7 and 8, wake from sleep, and every EFI
    present;
  - without it, a guest present completes with no call, as to a client that
    takes no call (section 6.21), and an EFI present ends the process.
- **`mode_change(width, height, pixel_format)`**:
  - called by the present when the size, the IOSurface pixel format or
    the protection changes, or when the guest colour space is dirty;
  - `rvg_display_set_modes` does not call it.
- **`cursor_glyph(glyph)`**:
  - on opcode 4;
  - the glyph is at most 256×256 RGBA8, swizzled from the guest's BGRA,
    with its hot spot (`RvgCursorGlyph`);
  - a checksum mismatch faults the command and skips the call, and a checksum
    of 0 skips the check;
  - without it the guest reads a glyph size limit of 0.
- **`cursor_show(show)`**: on opcode 5, and on a doorbell when the guest's
  flag at +0xe04 differs from the host's copy at +0xf20. `show` is the
  guest's byte as written.
- **`cursor_move`**: on every doorbell (0x1220).
- **The private handlers Apple's framework had** (`sleepHandler`,
  `modeListResponsivenessChangeHandler`, `compositorParametersHandler`)
  have no counterpart in the C API; the code behaves as the framework did
  with them absent.
- **`rvg_display_encode_frame`**:
  - It encodes one render pass that clears the whole target to (0,0,0,1),
    then draws a full-screen quad limited by the viewport = region, on the
    process's frame queue.
  - The shaders are the library's own (section 3.2).
  - Once the guest sends a gamma table, the gamma shader is used from then
    on: the pipeline key's `gamma` never goes back (`display/gamma.rs`).
  - The frame counts as in flight until it completes. A present waits up to
    2 s for the frames in flight, then logs a fault and proceeds. A host
    that encodes and never commits therefore makes every later present take
    2 s.
  - An EFI frame is a zero-copy view of guest memory (section 6.22).

### 4.7 Persisted state

- **The suspend blob.** `rvg_device_finish_suspend` returns it, in the
  library's own format (section 6.24).
  - Every byte of it is written. The normalizer reports it by parsed
    section, and whole as hex where the parse would not give its bytes
    back.
  - It flushes dirty GPU resources into guest memory first (W14), and tears
    tasks down.
- **Resume.** `rvg_device_will_resume` either accepts the state or refuses
  it with a reason (section 4.1). After `rvg_device_did_resume` the device
  rebuilds FIFOs, tasks and displays from guest memory and raises one
  interrupt.
- **Round trips** are the test that matters:
  - suspend, resume on a fresh device, and continue the scenario: every later
    observation must equal an unsuspended run's;
  - feeding a blob one build saved into another build is the check across
    builds.
- **The IOSurface host device's save state.** Apple's framework had one,
  reached only by private calls. This library has none, and the C API no
  call for it (section 6.23).

### 4.8 Outcomes: fault, crash, hang

**Faults are guest-visible:**
- 0x102c reads 1;
- 0x103c reads the faulting command's ring offset (for the root FIFO), or
  node +8 holds it (for a child FIFO);
- a fault interrupt is raised;
- the faulted stamp is not published until the guest resumes with 0x1024 or
  0x1028.

**Raises** never escape a C API function: the MMIO paths and
`rvg_device_did_resume` catch them, and FIFO commands turn them into faults
(**read**).

**Crashes.** Where the guest or the host breaks an invariant the library checks,
it dies (`raise::death`): one line on stderr, `reims-vgpu: fatal: ...`, then
`abort()`, so the process ends at the same op every run. A guest can reach
several of these deaths. With isolation off, a guest crash kills the host
process (QEMU). With isolation on, a crash in a task's own code ends only that
task, and every later command on that task faults (status 4); a crash inside
Metal kills the host, isolated task or not. The guest-reachable ones found are
in section 7.4.

**Hangs.** `CmdDelay` (61) never signals its stamp. After it, device teardown
and `rvg_device_reset` hang forever (**measured**, section 7.4).

The suite records crash and hang as outcomes. One process per scenario, a
signal handler and a watchdog (section 5.5) make a crash the last observation
of a run, not the end of the suite.

### 4.9 Not behaviour

These are diagnostics. The suite may record them for a person to read, but
never asserts them:
- log lines: the library writes them through the `log` crate, with a
  target per area (`reims_vgpu::fifo` and so on), to nowhere by default,
  to stderr with `REIMS_VGPU_LOG` set, or to a host's handler
  (`rvg_log_set_handler`);
- the stderr line of a death, and of an isolated task's death;
- thread and queue names (`reims-vgpu.<role>`);
- Metal object labels (`reims-vgpu.<role>`);
- `.ips` reports.

Mutation testing counts a mutant that only deletes a log call as "arid":
no test is expected to kill it.

### 4.10 Dark behaviour, and how the GPU's work becomes visible

Much of what the library does is Metal state that no observation shows
until something draws or computes with it. The suite makes every such piece
visible by authoring a workload whose output lands where the test can read
it. This is how "assert the machine, not the calls" covers Metal.

**How GPU results reach guest memory** (**read**, **measured** where noted):
- **No copy.**
  - Shared buffers (type 1, flags 0) are `newBufferWithBytesNoCopy` over
    guest pages:
    - in the blocks setup, over the task's host mapping;
    - in the map setup, over a `mach_vm_remap(copy=false)` of the host's RAM;
    - in an isolated task's service, over the host's own mapping of the RAM.

    GPU writes land in guest memory, with no copy-back step
    (`Resource::synchronize_with_cpu` does nothing for them). **Measured**: a
    blit fill, a compute dispatch and a render into a texture over such a
    buffer, read straight out of guest RAM.
  - The service's detach buffers and heaps use `replaceBackingWithRanges:`
    over guest ranges, on GPUs with `supportsResourceDetachBacking`, which
    the M4 Pro and the suite's M3 Max have.
  - Textures over buffers (serializer textures, commandType 9 or 55), and
    visibility-result buffers.
  - The IOSurface mapper's surfaces over guest pages. Whether IOSurface wraps
    those pages or copies them is **unknown**.
- **Copied.** These are visible only after a synchronize: the exec resource
  flag bit 2, or opcodes 53 and 62:
  - private buffers
  - legacy heaps
  - normal textures (types 2, 3, 12)
  - IOSurface backings (type 4)

  A synchronize, and `rvg_device_finish_suspend`'s flush, copy back only a
  resource that needs it (`Resource::needs_sync`): guest-invalid and host-valid.
  The exec flag invalidateGuest (bit 1) is what marks the guest copy stale. So
  53 and 62 copy back the same way, and a GPU write under an exec without that
  flag stays on the host (**measured**,
  `resources_private_buffer_sync_commands`, `resources_flush_on_suspend`,
  `suspend_writeback`). In the service, private buffers are Shared detach
  buffers, so their GPU writes land at once (section 6.10).
- **Never visible**: memoryless textures, and protected content.
- **On the host side**, `rvg_display_encode_frame` into a frame target, read
  back with `rvg_frame_target_read_pixels`, is the one pixel path.

**The workload that surfaces each kind of dark behaviour:**

| dark behaviour | the workload that surfaces it |
|---|---|
| state objects: functions, pipelines, samplers, depth-stencil, vertex descriptors, rate maps | draw or dispatch into a shared buffer, or a texture over one; read guest RAM |
| render encoder state: load and store actions, blending, depth and stencil, viewport and scissor, MSAA resolve, tile shaders, sample positions, amplification | render into a buffer-backed texture; or present it and read it back through `rvg_display_encode_frame` |
| private and copied resources | a GPU write, then a synchronize, or a blit into a shared buffer; `rvg_device_finish_suspend` also writes them back |
| mip levels above 0 (type 3 pages one level and generates the rest) | sample lower levels in a kernel that writes a shared buffer |
| ICB encoding (service tasks only) | synchronize the ICB, or execute it into a shared buffer |
| ordering: fences, barriers, `useResource`/`useHeap`, events | producer-consumer passes whose result depends on the order; a weak check, because a race may pass by luck |
| gamma, colour matrix, ICC profile, EDR, 10 bpc | encoded frames' pixels only |
| protection options | only with protected content and a host read-back; the suite covers only the non-pixel effects |
| purgeable state and discard | effectively dark: purging is driven by the OS |
| 2P_XR10_A8 blit kernels (the library's own, section 3.2) | page a type-12 texture with format 550 or 551 |
| EFI's tiled texture: no class implements the selector | dead code (**measured**); nothing to assert |

## 5. The suite's design

### 5.1 Shape

- **A runner** interprets a scenario (`runner/pgrun.m`):
  - It is a dumb Objective-C program. It knows no opcodes and no layouts:
    it performs memory writes, MMIO, waits and API calls, and it writes
    observations.
  - It includes nothing from the implementation but the C API's header, as
    QEMU does.
  - It builds for arm64, as QEMU does, with `-mmacosx-version-min=26.6`
    (`runner/build.sh`). It links the dist tree's `libreims_vgpu.dylib`
    with an `LC_RPATH` at the tree's `lib`, runs with `DYLD_*` stripped
    from its environment, and confirms with `dladdr` which image answered
    `rvg_device_create` (the `binding` record's `library`).
- **Scenarios**:
  - Scenarios are data: JSON Lines, generated by code, with the output
    committed so a diff is reviewable.
  - The generator is the one place the guest ABI is defined: from byte
    offsets, each citing the code it was read from, and independent of the
    implementation's structs. If a refactor changes a struct, the generator
    stays unchanged and catches it.
- **An orchestrator** (`python3 -m pgtest`):
  - It runs each case in its own process with a timeout, and records the
    outcome: exit code, signal, watchdog or timeout, plus the last op index.
  - It normalizes the observations, compares them with the expected file,
    and reports differences.
  - It runs cases in parallel (**measured** safe: 400 runs 8 at a time).
    A probe that looks at the whole host must stay per process.
- **Expected files**: one per scenario and configuration key (section 5.7),
  committed, and blessed only by a person.

This is language-neutral by construction: the runner tests whatever build
of the library it is given to load. The codebase is tested through its C
API, the way QEMU drives it (section 5.12).

### 5.2 The runner's vocabulary

The ops a scenario is made of (**read**, `runner/pgrun.m`,
`pgtest/scenario.py`). [api.md](api.md) section 7 says which C API calls
each makes.

| op or option | does |
|---|---|
| `config` | Allocates guest RAM (`mach_vm_allocate`), loads a handoff when there is one (section 5.10's phases), and creates the IOSurface host device, the device and its displays from the scenario's config, recording the refusal of each that refuses. Writes the `binding` record: which image answered, the setup, and the GPU and pid as diagnostics. |
| `mem`, `fill`, `mem_file` | Writes hex bytes, a repeated byte, or a committed fixture (checked by its SHA-256) at a guest address, and records them in a shadow copy of guest RAM. |
| `mem_seq` | `count` copies of a hex template at `gpa + k × stride`; each field `[off, size, start, step]` stores `start + k × step`, little-endian, 4 or 8 bytes, at `off` in copy k. Written into the shadow copy like `mem`. For structures too large to spell out: thousands of commands, or page tables of thousands of entries. |
| `mmio_w`, `mmio_r`, `mmio_sweep` | One register write or read of the device, from a pool thread while the scenario thread waits (decision 1); a sweep reads every register in a range and reports the nonzero ones. |
| `iosfc_w`, `iosfc_r`, `iosfc_sweep` | The same for the IOSurface host device, each value truncated to 32 bits, as `apple-gfx-mmio` declares them (section 3.1). |
| `wait` | Polls until a guest u32 at `gpa` is `eq`, `ge` (serial-number comparison) or `ne` a value, a register that reading does not clear matches, and/or the interrupt, callback, block-call or IOSurface-interrupt counts reach theirs, or a callback kind arrives, with a deadline. Reports ok or timeout. |
| `wait` with `any` | A list of conditions, satisfied when one holds; the observation's `which` names the first in list order that does (`Scenario.wait_any`). |
| `quiesce` | Waits until no event has arrived for a quiet window, within a deadline: the settle a checkpoint takes. |
| `diff` | Reports every run of guest bytes the device changed since the last diff: the shadow copy against guest RAM. Up to 512 bytes inline, otherwise a SHA-256. |
| `dump` | Hex of a guest range. |
| `cmp` | Two guest ranges: `equal`, and if not, `first_diff`. The differential scenarios (`interplay_diff_*`) compare two encodings of one workload inside one run. |
| `events` | Settles, then reports what arrived since the last report: interrupts as a count, callbacks with their contents; with `order`, every event with its sequence number and the snapshot taken inside it (section 5.4). |
| `events` with `sort_blocks` | The host's block calls sorted by what the expected file keeps of them (every field but `t_*` and `d_*`), and left out of `order`. FIFOs running at once in the blocks setup interleave `read_memory` calls by timing, which nothing may lock (decision 8); sorted, the set of calls is locked and their order is not (`newops_session_concurrent_blocks`, `newops_fan_out_fan_in_blocks`). |
| `events` with `blocks_digest` | Block calls as a count per kind and a SHA-256 of their records (each without `t_*` and `d_*`), for tens of thousands of them (`newops_range_tracker_limit`: 30,720 `map_memory` and 61,440 `read_memory` calls in one checkpoint). |
| `events` with `collapse` | Consecutive callbacks of the named kinds for one display count once: the EFI timer's `new_frame` calls (records `newFrame`), whose count before a checkpoint is timing (`efi_gap_timer_collapsed`). |
| `api` | One host call: `reset`, `pause`, `unpause`, `stop`, `willSuspend`, `finishSuspend`, `willResume` (with `nil`, no state at all), `didResume`, `cursorPosition`, `present_counts`, `display_props`, `set_modes`, `new_display`, `release_display`, `release`, `recreate` (the device again over the same RAM), `iosfc_again`. |
| `encode` | A frame of a display, encoded on a pool thread into a frame target the library makes, committed, waited for and read back. With `hold` the frame is kept uncommitted, numbered `held`, until an `encode_commit` names it and commits it on a pool thread: a host holds a frame in flight on purpose this way, rather than by timing. `no_commit` releases the frame uncommitted. |
| `metal` | A value computed from Metal in the test (section 5.8). |
| `trace_fire` | Invokes each trace handler whose range a guest write touches. |
| `handoff_save` | Ends a phase: guest RAM and the blob slots saved for the next process (section 5.10's phases). |
| `mark` | A label. |

Suspend and resume either recreate the device in the same process (`api
recreate`), over the same guest RAM, or split the scenario into phases,
each a process of its own (a migration); `willResume` can also take a blob
given as hex (**read**).

### 5.3 Formats

- **Scenario**: one op per line.
  - Integers that can exceed 2^53 are hex strings: `"0x40004000"`.
  - Bytes are hex.
- **Output**: one observation per line, keys sorted, flushed as it is
  written, so that a crash leaves everything observed before it. Every
  observation carries the op index in progress.
- **Normalization.** Only fields named `t_*` (timing) and `d_*`
  (diagnostics: signals, messages, pointers) are dropped
  (`pgtest/normalize.py`). The runner reports the library that answered
  as `<build>` when it lies under the build under test (`runner/pgrun.m`),
  so logs do not depend on the checkout. **Nothing guest-visible is ever
  masked.** A value that varies is either:
  - not being controlled by the scenario, which is to be fixed; or
  - host-assigned (section 5.8); or
  - memory nothing wrote, which a scenario must not observe. Where the
    library's own reads leave bytes unwritten or run past their data, it
    reads zeros (section 7.5).

### 5.4 Determinism

**Measured** with probes on the M4 Pro, run one at a time and 8 at a time,
isolation off and on: every scenario gave one normalized log, its stamps,
memory, registers, replies, display pages, callbacks and their order, and
crash op alike. `bless` checks the same of every case it records (section
5.7). Only timing varied, in the unnormalized view: which op an interrupt
or a display callback arrived during (the doorbell write, or the wait after
it), and whether the stamp was visible in guest memory before
`raise_interrupt` had run. A read of 0x1018 just after the stamp became
visible was never 0 in 60,000 samples, but the stamp is stored before the
bit is set (section 4.4), so the race exists.

**Rules:**
- Wait for the guest-visible condition *and* the interrupt count, then read
  registers.
- For an expected fault, wait for the fault interrupt, then read 0x102c and
  0x103c.
- Record interrupts and callbacks only at checkpoints that waited for them,
  and compare counts and contents there, never the op they arrived during.
- Assert every order the implementation establishes (decision 8). The runner
  records each host-side event (an interrupt, a task or memory callback, a
  display callback) with a sequence number, and a snapshot of the guest state
  the order is about, taken inside the event: the stamp slots when an interrupt
  is raised, for example. The expected file holds the order of those events and
  what each snapshot saw.
- Never sleep for a fixed time, and never assert an order that only timing
  decides.
- A wait that is expected to time out burns its whole deadline, so keep
  those deadlines short.
- The EFI timer (100 ms) and the present wait (2 s) are real time. Prefer
  `add_trace_range` for deterministic EFI presents.

### 5.5 Crashes and hangs as outcomes

- **One process per scenario and configuration.**
- **In the runner** (`runner/pgrun.m`, **read**):
  - handlers for SIGABRT, SIGSEGV, SIGBUS, SIGILL, SIGTRAP and SIGFPE write
    `{"ev":"crash","i":K,"d_signal":N}` with an async-signal-safe `write`,
    then re-raise the signal;
  - an `alarm()` watchdog, `watchdog_s` (default 30 s), writes
    `{"ev":"watchdog","i":K}` and exits 3. It covers teardown too, because
    the `CmdDelay` hang is in teardown.
- **In the orchestrator**: a timeout of its own over the whole process,
  30 s past the watchdog.
- **The outcome.** The orchestrator turns the end of a run into one
  `outcome` record (`pgtest/normalize.py`):
  - `completed`;
  - `died`, with `after`: the op of the last observation before the death.
    A death on another thread (a FIFO's, a service reply's) races with the
    runner's next op, so the op in progress is not locked; what was
    observed before it is;
  - `watchdog`, with `i`: the op that never returned;
  - `timeout` (the orchestrator killed the process) or `error` (a scenario
    bug). `bless` refuses both.

  The signal and the message are recorded for a person to read, and are
  not compared (decision 4): a later commit may die by `abort()` where the
  code died by SIGSEGV before, but it must die.

The end of an isolated task is not an outcome: the host lives, and the guest
sees the exec fault and every later command on that task fault (section 4.8). A
host death must be a scenario's last command (tests/suite/README.md, rule 3).

**Measured** (`smoke_crash`). A page-table root of 0 followed by a walk:
- isolation off: the host dies with SIGABRT at the walk's first table read
  (`read_range`, `memory/page_table.rs`), with a line on stderr (section
  4.9);
- isolation on: the task ends and the host survives. The command faults,
  with 0x102c = 1 and 0x103c = 0x1c.

### 5.6 The configuration matrix

- **Host setup**: blocks, map, isolate (section 3.3). Every scenario runs in
  each setup where it means something. The divergences between setups are
  behaviour to lock (section 7.2).
- **Binary version**: the guest's choice. Scenarios default to 245 (0xf5),
  the highest. A version sweep covers negotiation (section 6.3), and each
  feature gate is tested on both sides of its version.
- **Display ports**: 1 to 8 (the host's count), and the guest-visible count
  at 0x122c (section 6.21).
- **Pinned**: the preferences domain absent and `MTL_DEBUG_LAYER` unset
  (the runner refuses to run otherwise, and the orchestrator drops
  `MTL_DEBUG_LAYER` and every `DYLD_*` variable from its environment), the
  feature level never set, one device per process, unique serial numbers.
- **Metal API validation** is not a configuration: it is a debugging
  environment, not something a hypervisor sets (decision 1).
  `MTL_DEBUG_LAYER=1` changes behaviour (section 6.21, "Presents").

### 5.7 Expected values and blessing

- **Keying.** An expected file is keyed by:
  - the scenario;
  - the host setup;
  - the GPU generation;
  - the macOS build, for anything that reaches the Metal compiler (every
    pixel the library's shaders draw), Metal's private formats or a value
    Metal reports.

  A scenario whose output does not depend on the host has a single key.
  A scenario marked `host` has files named
  `<setup>@<model>-<cores>c-<build>.jsonl`, from `ioreg`'s IOAccelerator
  model and `gpu-core-count` and `sw_vers -buildVersion`
  (`pgtest/cli.py`, `host_key`), such as `AppleM3Max-40c-26A434`. Every
  other file is `<setup>.jsonl`, keyed by nothing about the host. So, by
  decision 5, a GPU generation that computes differently shows up as a
  failure, not as a missing file.
- **Blessing.** A person runs `make bless` (`pgtest bless`) locally, reviews
  the result and commits it with the change that caused it. CI never writes
  an expected file. A missing expected file fails in CI.
  - `bless` records from the checkout's dist tree, `target/dist`, by
    default (`$PGTEST_DIST` or `--dist` names another build); `make bless`
    builds it first, under `CARGO_TARGET_DIR` when that is set.
  - It records the selected cases whose expected file is missing or was
    recorded from other input, and leaves every other file alone.
  - It runs a case three times (`--runs`), one process each, and records it
    only if all three agree. A run that ends in `timeout` or `error`
    refuses the case, and runs that differ report it nondeterministic
    (decision 7).
- **The input.** Each expected file's first line records its input: a
  SHA-256 of the generated scenario without its `#` doc line, plus the
  runner's source (`runner/pgrun.m`) and `pgtest/normalize.py`. A
  scenario's docstring therefore changes no input. `run` fails a file
  recorded from other input.
- **Changing behaviour on purpose.** Nothing marks a test as expected to
  differ (decision 4). A change meant to alter behaviour records the cases
  it changes again with `pgtest bless --update` (`make bless
  TEST=<patterns> ARGS=--update`), and the diff of their expected files is
  reviewed in the same commit (decision 3). `pgtest diffsum` summarizes
  that diff by record kind and by scenario.
- **A new OS build.** After an OS update the host-keyed files for the new
  key are missing, and `make bless` records them; the old key's files stay
  beside them, and comparing each new file with the old key's file for the
  same case is the review.

### 5.8 Values the host assigns

These are asserted without pinning the value:
- **Computed in the test from Metal.** The heap texture size and alignment
  reply equals `-[MTLDevice heapTextureSizeAndAlignWithDescriptor:]` for
  the same descriptor (section 6.10). The test computes the expected value
  the same way, and must build the same descriptor,
  `allowGPUOptimizedContents` included: the runner's `metal` op takes
  `allow_gpu_optimized` for this, and `textures_gap_heap_gpu_optimized`
  asks Metal with the bit clear and set, and the replies to opcode 64 equal
  Metal's both ways (64×64 RGBA8Unorm: 0x4000/0x80 clear, 0x4080/0x80 set,
  so a one-page heap takes the texture only with the bit clear;
  **measured**).
- **As a relation.** For example:
  - a placement entry's gpuResourceID equals the info reply 466 or 467 for
    the same object;
  - a gpuAddress resolves to the buffer it names;
  - two objects' unique identifiers differ.
- **Keyed to the chip**, when the value reports the host: device info
  (section 6.9), compute info, pipeline limits, 0x1224.

Whether GPU addresses and resource IDs are stable from run to run is
**unknown** (section 9).

### 5.9 Rules for test code

- Never include the implementation's source (the crates). Guest layouts
  are defined in the generator, and the API comes from the C header, the
  one QEMU includes.
- Call only what the C header declares, from the threads QEMU uses
  (decision 1).
- Cite the code for every layout (`pgtest/abi.py` does, constant by
  constant). The citation is documentation, not a dependency.
- Name only what something outside the implementation also names: the C
  header, QEMU's declarations, guest-ABI offsets and opcodes, Metal, and
  the persisted formats. The crates' own names (types, fields, modules) are
  internal.
- Commit shader fixtures as source plus a build script
  (`fixtures/shaders/build.sh`), and commit the compiled metallibs beside
  them, so the input bytes cannot drift with the toolchain. They are 3-7 KB
  each, compiled from the suite's own source.
- Test code is not built by the product's Cargo build, and never changes
  the product.

### 5.10 The suite

- **The smoke**, `make smoke` (`cargo xtask smoke`, `xtask/src/smoke.rs`),
  arm64: a guest program (`xtask/src/smoke_guest.m`) linked statically with
  the dist tree's `libreims_vgpu.a`, as QEMU links it, and driving its C
  API: a device over a memory map with process isolation, as
  `apple-gfx-mmio` makes it, driven through MMIO and the root FIFO to one
  exec in an isolated task, whose fill must land in guest RAM. Its facts
  must equal the library's own expected values (`GUEST_EXPECTED`), and
  `rvg_device_create` must be part of the guest's own image. It is a quick
  check; the suite is the definition of correct behaviour.
- **The suite**, `tests/suite/` (its README is the user's guide):
  - `runner/pgrun.m`, built by `runner/build.sh` into `bin/pgrun-<hash>`
    against a dist tree: it includes the tree's `reims_vgpu.h` and links
    its `libreims_vgpu.dylib` by an `LC_RPATH` at the tree's `lib`, so it
    runs whichever library that tree holds. It is the host: it plays the
    hypervisor and guest RAM, drives the C API ([api.md](api.md) section
    7), and writes observations (sections 5.1-5.5). Each callback checks
    the thread it arrives on: a display callback on a thread the runner did
    not create and never overlapping another, a device callback there too
    or inside one of the runner's own library calls. A broken rule is a
    `scenario_error`.
  - `pgtest/`, the Python library and orchestrator. `abi.py` holds the
    guest ABI's constants, each citing its source; `scenario.py`, `fifo.py`
    and `task.py` the machinery (memory, registers, commands, waits,
    checkpoints, page tables, object lists, segments); one builder module
    per area (`compute_objects.py`, `render_lib.py`, `display_guest.py`,
    and so on), and one generator per seeded walk.
  - `scenarios/<area>.py`: the scenarios, each a function decorated
    `@scenario` whose docstring says what it locks, and cites the code.
  - `generated/`: the scenarios as JSON Lines, committed.
  - `expected/<module>/<scenario>/<setup>[@host].jsonl`: the locked
    observations.
  - `fixtures/shaders/`: shader sources and their compiled metallibs, both
    committed (section 5.9), and `build.sh`, which reproduces them byte for
    byte; `render_blobs.m` makes the render pipeline blobs.
- **The orchestrator**, `python3 -m pgtest` from `tests/suite`:
  - `gen [--check]` writes the generated files, or reports stale ones;
  - `list` names every scenario, its module, setups and whether it is
    keyed by the host;
  - `run [-k PATTERNS] [--setup S] [-j N] [--dist DIR]` runs each case in
    its own process and compares it with its expected file; the build
    defaults to the checkout's dist tree, `target/dist` (`$PGTEST_DIST`);
  - `bless [--update]` records missing expected files and those recorded
    from other input, or with `--update` every selected case, from the
    same build (section 5.7);
  - `cov` runs the selected cases on the library built instrumented and
    reports the lines each crate file's code reached (`--tag`, `--files`,
    `--show FILE` for one file's missed lines, `--per-case` for the cases
    that reach each line);
  - `diffsum [A [B]]` summarizes a change to expected files by record kind
    and by scenario.

  `make test`, `make bless` and `make coverage` wrap `run`, `bless` and
  `cov`, with `TEST=<patterns>` selecting scenarios or modules and `ARGS`
  adding options.
- **Setups and phases.** Every case is a scenario in one setup: `blocks`,
  `map` or `isolate` (section 3.3). A scenario runs in all three unless
  its docstring gives the reason it cannot mean anything in one: most such
  need a memory map (device creation, the IOSurface host device, the
  mapper), or split where the setups diverge (section 6.10), in-process in
  one scenario and the service in another. A scenario in phases is a
  migration: each phase runs in a runner process of its own, handed the
  last one's guest RAM and blobs, and its setup is a chain such as
  `map-isolate` (tests/suite/README.md, "Phases").
- **The areas** (`pgtest list` names each scenario's module):

  | area | modules |
  |---|---|
  | device: registers, versions, creation, lifecycle, barriers, tasks | `device_*` |
  | FIFOs: framing, stamps, barriers, child FIFOs, opcodes | `fifo_*` |
  | tasks, memory, object lists, the service's view of task memory | `tasks_*`, `memory_gaps`, `service_gaps` |
  | blit and compute decoders, functions, compute pipelines | `blit*`, `compute*` |
  | render decoder, render and tile pipelines, state objects, rate maps | `render_*` |
  | execution, segments, the info decoder, ICBs | `exec`, `info`, `icb`, `decoder_gaps` |
  | resources, heaps, textures and their data paths | `resources*`, `textures*` |
  | displays, cursor, EFI, IOSurface device, suspend and resume | `display_*`, `efi*`, `iosfc*`, `suspend*` |
  | the host as the C API allows it and no other scenario plays it: callbacks that refuse or are left out, the public lifecycle orders, migrations between processes | `fresh_*` |
  | whole sessions and subsystems across each other (below) | `interplay_*` |
  | seeded programs: resources aliasing one another's memory, large memory maps, teardowns with work pending | `native_*` |
  | the range tracker at its limit, frames in flight, concurrent FIFOs in the blocks setup | `newops` |
  | paths a coverage report showed no case reaching | each area's `*_gaps`, `covgap_*` |
  | mutants a mutation pass left alive, one scenario or more each | `mutgap_*`, `mutgap2_*` to `mutgap4_*` |
  | examples of each mechanism | `smoke`, `smoke_host` |
- **The interplay modules** drive several subsystems at once, each with its
  builder in `pgtest/`:
  - whole guest sessions and a guest's whole life (`interplay_session`,
    `interplay_boot`), FIFOs across each other (`interplay_fifo`), host
    lifecycle calls mid-workload (`interplay_lifecycle`), differential
    checks (`interplay_diff`), isolation across tasks
    (`interplay_isolation`), and several subsystems at their limits
    (`interplay_limits`);
  - `interplay_rand` (`randgen.py`): seeded random programs of legal work
    over every subsystem, each item compared with an uninterrupted control
    run;
  - `interplay_faultwalk` (`randfault.py`): `interplay_rand`'s walk with
    guest errors injected at random points (faulting execs, out-of-range
    replies, unknown tasks and opcodes, circular waits below 32, and
    service deaths in the isolate setup) and recovered from as a driver
    would. A model predicts which FIFO faults, what it holds back and what
    survives; every item that runs is compared with its control, and at
    every fault, resume and checkpoint 0x102c is waited for at the model's
    value and 0x103c and every live child's fault offset are recorded. It
    locks, together and in random orders: the fault a pause leaves
    latched, so the next is silent (section 7.1), on the root FIFO and on
    children; a fault that survives a suspend as registers only (section
    6.24); a circular wait below 32 faulting the second waiter; W3 at an
    unpause and at a resume (section 6.7); and a faulted child's delete
    clearing its fault;
  - `interplay_scanout` (`scanout_util.py`): the display pipeline end to
    end. The GPU renders into IOSurface mapper surfaces (mapper textures
    and buffers, types 11 and 15) and into task 0's backings (ref
    textures, type 5), which opcodes 8, 7 and 6 present on up to four
    heads from any FIFO, ordered by barriers, while modes, hot plug, the
    cursor, gamma, sleep, pause, suspend, reset, stop, EFI and the
    IOSurface host device change around them (sections 6.21-6.23);
  - `interplay_lifetime` (`lifetime_util.py`): objects deleted, made again
    or retyped under the same reference while work that names them is
    queued, running or held on another FIFO; heaps whose buffers and
    textures alias; resources paged across many execs while the guest
    rewrites their pages; and what a pause, a suspend and a DeleteTask
    keep (sections 6.10, 6.11, 6.19, 7.1);
  - `interplay_dag` (`dagwalk.py`): seeded dependency graphs of execs over
    the root FIFO, up to 16 children and several tasks, whose edges are
    barriers, and hand-written shapes (diamonds, a fan-in of 16, a 51-node
    chain, chains across a pause past the 2 s report and across a suspend,
    children deleted and defined again, a full pending queue, W3 left
    unrepaired, an abandoned suspend over blocked work). Every node leaves
    an order witness that the graph alone fixes, compared with a control
    run in topological order; this locks what orders GPU work and what
    does not (section 6.7);
  - `interplay_versions` (`versions_util.py`, which holds the gate table
    with its citations): a whole guest at each version where a gate flips
    and at the requests that are not versions, exercising every gated
    opcode, object, display feature, child FIFO limit and a migration at
    once; one device walked through every valid version; versions changed
    under live tasks, across migrations, and across `stop` and `reset`
    with the guest setting up again; EFI against the driver's version;
    and what a service task copies at its creation (section 6.3);
  - `interplay_mix` (`randmix.py`): seeded programs that mix every
    dimension the other generators walk apart (barrier graphs over up to
    eight children and one or two tasks, renders into mapper surfaces and
    their presents, work held while the guest deletes, retypes or remakes
    what it names, injected faults and their recoveries, and the
    lifecycle), with one model predicting every observation: the
    differential control for GPU work, the bytes of every surface and
    scratch page, one `new_frame` per present, the fault registers, and
    which waits time out;
  - `interplay_memlive` (`memlive_util.py`, which restates the page-table
    layout and what the scenarios lean on): guest memory changing under
    live GPU work. The guest rewrites leaves and intermediate tables,
    punches holes, flips the read-only bit, aliases pages between tasks (a
    task's own page-table pages among them), moves pages between RAM
    ranges, maps and unmaps, and grows a task, while the work that uses the
    memory is queued, held behind a barrier on another FIFO, or running; a
    migration onto a smaller memory map ends it (sections 6.10-6.12);
  - `interplay_displaywalk` (`randdisplay.py`): seeded walks over one or
    two display ports and EFI (connects, acks, hot plug, mode lists,
    presents with gamma tables and colour matrices, cursors, sleep, pause,
    suspend, reset and stop, the guest overwriting fields the host owns),
    with a model restated from the guest display's and the display's code
    predicting every callback, every shared page byte for byte (compared
    with a mirror page), 0x1014, 0x1018, the root page's display words and
    the encoded pixels (section 6.21);
  - the other seeded walks, each named for what it walks: decode streams
    (`interplay_decodewalk`), the IOSurface host device and its consumers
    (`interplay_iosfcwalk`), lifecycle events landing in live mixed work
    (`interplay_lifewalk`), every subsystem stepped at once
    (`interplay_marathon`), and many GPU tasks with faults injected into
    one at a time (`interplay_servicewalk`).
- **Exclusive scenarios.** A command buffer that faults the GPU makes
  Metal discard other processes' work too ("Discarded (victim of GPU
  error)" in the system log), so the cases beside it stall with every wait
  timing out (**measured**). A scenario that faults the GPU, or crashes
  Metal's compiler service
  ("Compilation failed with XPC_ERROR", whose retries and crash reports
  under load outlast the watchdog), passes `exclusive=True`, and the
  orchestrator runs it alone, after the others (**read**, `pgtest/cli.py`,
  `run_all`). Among them are `render_draw_type9`, `render_draw_icb`,
  `render_pass_no_target`, `render_objects_pipeline`, and the two ICB uses
  after delete (section 7.1), `interplay_lifetime_icb_bound_buffer_deleted`
  and `interplay_lifetime_icb_pipeline_deleted`, whose GPU writes would
  fault a GPU if they did not land. Alone means within one orchestrator:
  nothing keeps two orchestrators, or an orchestrator and other GPU work,
  apart (the suite's README, rule 10).
- **Load-sensitive cases.** Some cases, among them the synchronize
  deaths, the kernel-held pair, two exclusive `render_objects` cases and a
  set of `interplay_*` walks, can fail under concurrent GPU load and pass
  alone; tests/suite/README.md lists them with their setups. Before calling
  one a failure, run it again alone.
- **Where a death is recorded.** A host death on another thread races with
  the runner's next op (**measured**: `info_reply_aborts_host[map]` died at
  op 396 in one run and 397 in another), so the outcome records the op of
  the last observation before the death (section 5.5). A crash inside Metal
  on an isolated task's thread is such a death: a scenario puts its command
  behind every other, rings its doorbell last and makes its fault the first
  wait, so the process dies in that wait every run (tests/suite/README.md,
  rule 3).
- **The coverage build** (`make coverage`, `pgtest/coverage.py`) is
  `build/cov`: the library, built in its own target directory under
  `cargo llvm-cov show-env`, which instruments the workspace's crates and
  no dependency. It is never a build under test. `cov` runs the selected
  cases, merges their profiles, and reports per crate file the lines the
  library reached; isolated tasks run in each case's own process, so its
  profile holds them. `--per-case` gives each case its own profiles and
  writes `map.json`: each line and the cases that reach it.

### 5.11 Tooling

- **Runner**: Objective-C, arm64 (section 5.1).
- **Generator and orchestrator**: Python.
- **No fuzzing.** Apple clang 21 ships ASan, TSan, UBSan and the profile
  runtime, but no `libclang_rt.fuzzer_osx.a` (**measured**); Homebrew
  LLVM ships a fuzzer runtime for arm64 only.

### 5.12 Running against the checkout

The build under test is the checkout's library, in the dist tree `cargo
xtask dist` writes (`target/dist`, or under `CARGO_TARGET_DIR`): the arm64
`libreims_vgpu.dylib`, with the install name `@rpath/libreims_vgpu.dylib`,
and its C header (development.md, "Dist"). It runs isolated tasks on threads of
its own, in the runner's process (section 3.3). The runner is built against
the tree `--dist` names, which `run` and `bless` default to:

```sh
make test                                     # cargo xtask dist, then the suite
cd tests/suite && python3 -m pgtest run -j 10 # target/dist ($PGTEST_DIST)
```

- The library is arm64 only, as the runner is. rustc's arm64e target has no
  pointer authentication ([architecture.md](architecture.md) D32).
- `--dist` may name the Vulkan platform's tree, `target/dist-vulkan`
  (development.md, "Dist"). The expected files are recorded from the Metal
  platform's.
- The pass bar: every case matches its expected file, `died` and
  `watchdog` outcomes included, at the same op. `cargo xtask dist --profile
  checked --out DIR`, run with `--dist DIR`, is a second pass, with
  overflow checks ([architecture.md](architecture.md) 13.6).

## 6. Guest ABI reference

This is the guest ABI as the code implements it, and what the suite's
scenarios are built from. Everything here is **read** from the code unless it
is marked **measured**; most of the core is also measured end to end. The
suite's builders restate each layout from offsets, and never import the code's
record definitions. Rust source paths below are under
`crates/reims-vgpu-core/src/`.

### 6.1 Conventions

- Pages are 16 KiB (`APV_PAGE_SIZE`, 0x4000, `guest/records.rs`). A "page" in a
  register or payload is a guest-physical page number, GPA >> 14.
- Little-endian. FIFO payload structs are packed to 4 bytes (`#pragma pack(4)`,
  `fifo/wire.rs`); segment records are packed or 4-aligned as noted.
- "Task VA" is an offset in a task's virtual space, translated by its page
  table (section 6.10).
- A payload or record longer than the code reads is accepted, and the extra
  bytes are ignored; a shorter one throws.

### 6.2 MMIO registers (main BAR)

`index = (uint16_t)((offset - 0x1000) / 4)` for both reads and writes
(`device/mmio.rs`). So offsets below 0x1000 wrap, 0x41000 aliases 0x1000
(**measured**), and unaligned offsets round down. Unhandled reads return 0;
unhandled writes are ignored.

| offset | read | write |
|---|---|---|
| 0x1000 | root FIFO enabled | nonzero: enable (asserts not already enabled: writing a value other than the current one while enabled aborts); 0: tear down (every child FIFO is deleted, and the work the root FIFO and the children held is abandoned; tasks, display pages and guest displays survive, and the guest defines its children and rings the work again: **measured**, `interplay_lifewalk_*`); equal to the current value: nothing |
| 0x1004 | | FIFO length (ignored, with a log, while enabled) |
| 0x1008 | | FIFO written: a free-running u32 byte count; the doorbell |
| 0x100c | root FIFO read count (bytes consumed; advanced before a command runs) | |
| 0x1010 | | FIFO start offset (ignored while enabled) |
| 0x1014 | display interrupt bits, read-to-clear | |
| 0x1018 | stamp interrupt bits, read-to-clear | |
| 0x101c | root page | root page: tears down the previous root (FIFOs, tasks, displays); 0 leaves it unset; in the blocks setup creates the root task |
| 0x1020 | | wake child FIFO n (0 or >= 64: log only) |
| 0x1024 | | resume the root FIFO |
| 0x1028 | | resume child FIFO n; n >= maxFIFOCount logs but still shifts `1ULL << n`. **Measured**: `1ULL << 64` shifts as `1 << 0` on arm64, so n = 64 (and n = 0) clears the root's fault bit without resuming the root: 0x102c reads 0 while the root's stamp stays held until 0x1024 (`fifo_child_resume_64`, `device_lifecycle_resume_child_order`). Past the 64 child slots there is no FIFO: n >= 64 clears bit n mod 64 and resumes nothing, as a wake ignores such an id, and the host lives (`fifo_child_resume_65_dies`, `device_gap_resume_child_huge_dies`, which complete). The mask clears by n, the FIFO's id, while a fault sets bit stampIndex. On a FIFO that is not faulted it logs and, for the process's first 9 such writes, reports every FIFO's waits synchronously (section 7.4) |
| 0x102c | 1 if any FIFO is faulted (`interrupt_fault_mask != 0`), else 0; not read-to-clear | |
| 0x1030 | | FIFO base page (ignored while enabled) |
| 0x1034 | negotiated binary version | requested binary version (section 6.3) |
| 0x1038 | | trace marker (kdebug only) |
| 0x103c | root FIFO fault offset: 0 with no FIFO, 0xffffffff when healthy, else the faulting command's ring offset | |
| 0x1200 | EFI enabled | EFI enable / disable |
| 0x1204 | EFI mode count (the display's mode list; 0 without EFI or display) | |
| 0x1208 | | EFI mode select |
| 0x120c | selected mode: w \| h << 16; 0xffffffff if out of range; 0 without EFI | |
| 0x1210-0x121c | EFI start page, length, depth, mode (w \| h << 16) | same |
| 0x1220 | | display doorbell: the port number |
| 0x1224 | EFI stride alignment: `minimumLinearTextureAlignmentForPixelFormat:BGRA8Unorm`, 16 on the M4 Pro and on the suite's M3 Max (**measured**, `efi_registers`); 0 without EFI | |
| 0x1228 | EFI stride | EFI stride |
| 0x122c | guest display port count: the host's count at binary version 245, else `count != 0`, with a log on each read | |
| 0x1234 | port connection types, `& 3` below version 245 | |

A fresh device's reads are section 4.2's.

### 6.3 Binary version negotiation

`DeviceCore::set_binary_version` (`device/mmio.rs`), from a write to 0x1034:
- negative (bit 31 set): log, unchanged;
- 1 or 236: unsupported, stored with the high bit set;
- 0: stored as 0;
- above 0xf5: clamped to 0xf5; an unknown value steps down to the nearest
  valid one;
- then clamped by the device's feature level (1: 62, 2: 113, 3: 209, else
  245; a level of 5 or more ends the process);
- features recomputed from the result (`features_for_version`,
  `guest/abi/features.rs`); EFI is torn down if the version lacks `efiDisplay`.

The 51 valid versions: 2-6, 10, 11, 20-22, 30, 31, 40-43, 60-62, 100-113,
200-209, 230-233, 235, 237, 238, 245. The switch has no case 234, so 234
steps down to 233 (**measured**, `device_gap_version_fingerprint`); 1 and
236 are cases that mark the version unsupported. Each enables a cumulative
set of the 51 `APVFeatures` flags, with three overrides after the switch (v == 40,
7 <= v < 100, 7 <= v < 104).

**Measured** (feature level 0, isolation off and on alike), write → read:

| write | read |
|---|---|
| 0xf5 | 0xf5 |
| 0x10000 | 0xf5 |
| 0xf4 | 0xee |
| 0xec (236) | 0x800000ec |
| 0x7f | 0x71 |
| 0x3e | 0x3e |
| 0x2b | 0x2b |
| 1 | 0x80000001 |
| 0 | 0 |
| 0x80000000 | 0 (unchanged from before) |
| `rvg_device_reset` | 0 |

**Write the version first.** Before any version is written the device's features
are zero: `maxFIFOCount` is 0, so child FIFOs and barriers on index > 0 fail,
and every gated opcode faults (**read**, **measured**). Any write that is not
ignored sets `maxFIFOCount` to 32 before the switch, and version 61 and up set
64: so 0, 1 and 236, which the switch does not enable, still give 32
(**measured**, `device_version_fifo_count`: ids up to 31 define after writes of
0, 1, 236, 2 and 60; 32 needs 61).

Feature gates, by the lowest version whose case sets the flag, and what reads
each (**read**; the switch in `features_for_version`, and `GATES` in the
suite's `pgtest/versions_util.py`, which keeps this table; a bare number is an
opcode):

| version | flag | read by |
|---|---|---|
| 3 | `efiDisplay` | `DeviceCore::set_binary_version`, which tears EFI down without it |
| 5, off for 7-99 | `supportsSharedTextures` | type 16, device info key 19 |
| 10 | `displayMapperSurface`, `synchronizeAndDiscardResources`, `mapperIOSurfaces` | opcodes 8 and 62; types 11 and 15 in-process (an assert: section 7.4) |
| 11 | `discardResources` | 63 |
| 21 | `displaySleep` | 9 |
| 22 | `supportsRGhAPresents` | 'RGhA' presents, which it logs and allows (section 6.21) |
| 31, off at 40 | `dualPlaneTextures` | types 12 and 18, key 12's value |
| 40 | `displaySetProperties` | 10 |
| 43 | `metalHeaps` | 64, keys 14-16 |
| 60 | `bufferFromIOSurface` | key 17, IOSurface buffers in-process |
| 61 | maxFIFOCount 64 | child ids and stamp indices 32-63 |
| 62 | `s8ByteCountForD32S8` 1, else 4 | the stencil plane's texel in Depth32Float_Stencil8 |
| 105 | `supportsProgrammableSamplePositions` | key 21 |
| 106 | `supportsVariableRasterizationRateMaps` | 66 |
| 107 | `supportsTileShadersImageBlocks` | keys 23-27 |
| 108 | `supportsDisplayCompositorParameters` | 11 |
| 109 | `supportsCmdExecIndirect3` | 67 |
| 111 | `supportsDynamicAttributeStride` | key 28 bit 1 |
| 113 | `supportsMultiplaneMippedBlitTextures` | mipped two-plane blit textures and shared backings |
| 202 | `supportsCommandBufferJump` | key 36 |
| 204 | `supportsHeapHazardTracking` | tracked heaps |
| 207 | `supportsArgumentBuffers` | 68; in the service, the task's placement list and read-only buffers and heaps |
| 208 | `supportsDisplaySetGuestICCProfile` | 12 |
| 231 | `supportsIndirectCommandBuffers` | the service's GPU-address registration |
| 235 | `supportsInfoIndirect` | 69 |
| 238 | `supportsDisplayPortConnectionType` | the resume's connection-type check |
| 245 | `supportsMultipleDisplays` | 0x122c and 0x1234 |

The other flags (`objectTables` 2, `displayDoorbell` 4, the sunburst
display, the 101-103 and 110-112 render flags, and twelve more from 30 to
237, which `versions_util.py` names) are set and never read, so a guest
below them sees nothing refused.

**Measured** by the version scenarios (`interplay_versions_*`: a whole guest at
each version where a gate flips and at 0, 1, 234 and 236; one device walked
through every version; versions changed under live tasks, across migrations and
across `stop` and `reset`):
- **When the version is read.** Every FIFO gate and device info read the
  device's current features, and so does an in-process task. A service task
  keeps the copy it was given at its creation. So a version changed while tasks
  live splits the setups: 67 follows the new version at once, and an object made
  after the change is made by the new version's features in-process and by the
  task's creation version in the service (`interplay_versions_live_245_to_30`,
  `_30_to_245`, `interplay_versions_walk`). A task defined at 230 and one
  defined after a raise to 231 run the same compute ICB, and 53 writes its bind
  back as reference ~0 and offset 0 for the first, the buffer's reference for
  the second (`interplay_versions_icb_registration`); a task defined at 206
  records no placement list at 207's 68 and refuses a read-only buffer, where
  one defined after the raise does neither
  (`interplay_versions_argbuf_snapshot`).
- **The stencil texel.** A guest lays out a Depth32Float_Stencil8
  texture's stencil plane with its own version's texel (4 bytes below 62,
  1 from it). A task that reads the other version's features refuses the
  texture, though both versions allow two-plane textures: task 1 in the
  service after 30 to 245, task 2 in-process once back at 245
  (`interplay_versions_live_*`).
- **Usable below the nominal gate.** Port 1 connects and presents below 245:
  only 0x122c and 0x1234 are gated, and the port checks use the host's count.
  The cursor doorbell 0x1220 works at 2 and 3 (`displayDoorbell` is never read).
  EFI enabled after a version write of 2 keeps presenting, since the flag is
  checked only at a version write, and a migration restores it on the fresh
  device (its `rvg_device_did_resume` replays the EFI registers after
  `rvg_device_will_resume` set version 2) until the guest writes its version
  again (`interplay_versions_efi_*`). In-process, read-only buffers and heaps
  are made at every version.
- **Migration over a lowered version.** Children defined above the lowered
  version's maxFIFOCount are lost: `rvg_device_did_resume` walks ids below the
  fresh device's count, so children 40 and 63 defined at 245, with the version
  lowered to 60, are not FIFOs on the resumed device though their nodes still
  say running. Raised to 245 again, the guest defines them anew and they run
  (`interplay_versions_suspend_lowered`). The blob's device info is written at
  the version of the suspend (`interplay_versions_suspend_info_*`), and a guest
  that raises its version on the resumed device gets the higher gates at once
  (`interplay_versions_resume_raise`).
- **Versions 1 and 236 cannot migrate.** The blob carries the stored version
  with bit 31 set, which `rvg_device_will_resume`'s first check reads as not
  below 0xf6: InvalidGuestVersion (3). The later check, for a version the device
  does not support, is not what refuses them (`interplay_versions_gate_1`,
  `_236`). The check for version 1 in `features::clamp_binary_version` is dead:
  1 has bit 31 set by then.
- **`stop` and `reset`.** After `stop` the device keeps its version and the
  device-info max key the guest last queried (45); after `reset` it has version
  0 and max key 1, so a guest that sets up again without writing a version gets
  version 0's features (67 and 69 refused, 32 FIFOs). Tasks do not come back
  from the root page's task words: an exec faults until the guest defines the
  task again (`interplay_versions_resetup_*`).

### 6.4 The root node page

The page written to 0x101c (**read**, **measured**):

| offset | content | written by |
|---|---|---|
| 0x000 + 4 × taskID (taskID < 256) | u32 task root page, 0 when none | host on define and delete (W4) |
| 0x400 + 0x14 × (fifoID − 1) | child FIFO node (section 6.7) | guest, and host fields |
| 0x900 + 4 × port | u32 shared display state page | host (W5) |

Resume re-creates every task whose word is nonzero, reading its length from task
root +0x14, and every display whose word is nonzero, so both host writes are
part of the suspend contract.

### 6.5 The root FIFO

`DeviceCore::create_root_fifo` (`device/root.rs`) on a write of 1 to 0x1000
needs, else it logs and the register stays 0: a root page; length (0x1004) a
multiple of 0x4000 and at most 0x20000; start (0x1010) nonzero and
0x400-aligned; the mapping to succeed; and start < length. Six refusals, each
silent to the guest. The largest ring, 0x20000, is the one
`newops_range_tracker_limit` writes 2,032 commands at a time into.

- The mapping is guest-physical [base page × 0x4000, + length), contiguous.
- A length of 0 passes the length checks. In the blocks setup the mapping then
  fails: the root task's allocator (`FirstFitVa`, `collections/first_fit.rs`)
  refuses a size of 0, so `DeviceCore::new_contiguous_mapping` makes no
  mapping and the FIFO is refused (logged). No root-task mapping is made, so
  `Mapping::new`'s death on a length of 0 is never reached (**read**).
- The stamp slots are u32s at the mapping's base: stamp i at base + 4i.
- The command ring is [base + start, base + length), size length − start.
  `written` (0x1008) and `read` (0x100c) are free-running u32 byte counts;
  the ring offset is count % size, and a command may straddle the end
  (**measured**).

### 6.6 Command framing, stamps and faults

The header, `FifoCommand` (`fifo/wire.rs`), **12 bytes**:

| offset | size | field |
|---|---|---|
| 0 | u16 | opcode (commandID) |
| 2 | u16 | barrierCount |
| 4 | u32 | length: the whole command, header included |
| 8 | u32 | stamp value |

Then barrierCount × `{u32 fifoIndex, u32 value}`, then the payload. The FIFO's
thread (`Fifo::run`, `fifo/thread.rs`), per command:

1. length < 12 or > the ring size: log and `continue`. The header's bytes are
   consumed locally but the published read count is not advanced (the pending
   advance accumulates), so `available` stays positive and the thread goes on
   parsing past `written`, into stale ring bytes, until something parses
   (**measured**: stale bytes parsed as opcode 0, the read count then jumped).
   No stamp, no fault. The suite makes this deterministic by putting a command
   that blocks on a barrier after the stale bytes (`fifo_frame_length_*_stale`).
2. payload underflow (length − 12 − 8 × barriers): push the stamp, fault.
3. barrierCount > 64: push, fault.
4. barrier bytes beyond length: push, fault. **Dead**: the underflow check of
   step 2 (a checked subtraction) fails exactly when this one would, so this
   check never runs (**read**).
5. barrier wait (`device/waits.rs`): an index >= maxFIFOCount throws, and a
   circular wait throws: fault. The wait compares `(int32_t)(slot − value) >=
   0`; a 2 s timeout logs and reports, then waits again. An unsatisfied barrier
   blocks that FIFO (**measured**). A barrier on the FIFO's own slot for a value
   beyond its last pushed stamp is circular and faults; one for a value pushed
   but not yet signalled is not, and blocks (`fifo_barrier_root`,
   `fifo_barrier_self_wait_pending`).
6. dispatch on the opcode (section 6.8); unknown opcodes log and fault
   (**measured**: opcode 13 and 99).

Steps 2 and 3 push, fault and `continue` without `Fifo::advance`: only the 12
header bytes are consumed, into the pending advance, and the rest of the
rejected command is parsed as new commands. **Measured** with a valid command
after it in the same doorbell: an underflow's leftover bytes, and a rejected
command's barrier words, run as commands, and 0x100c catches up when that
command advances (`fifo_frame_underflow`, `fifo_frame_barrier_count`). When the
rejected command is the last the doorbell counts, the read count stays behind as
in step 1 (**read**; not locked).

**The root FIFO never waits inside a command.** `available` is `written − read`
with the published read count, so any positive difference makes it read a whole
command, taking the bytes past 0x1008 from the ring as they are. **Measured**:
0x1008 set to 0x18, which counts only the header of a 40-byte NOP: the NOP ran
and 0x100c read 0x34, past 0x1008 (`fifo_frame_partial_root`). **Child FIFOs
differ**: they count reads locally (`Fifo::read_bytes`, `fifo/ring.rs`), wait
for a partial command's remaining bytes, and do not parse stale bytes; the
node's readCount is stored only by `Fifo::advance` (**measured**,
`fifo_frame_partial_child`).

**Stamps** (`fifo/stamps.rs`): every dispatched command pushes (stamp value,
opcode) into a 128-slot ring; `Fifo::push_stamp` blocks when 128 are pending.
Completion marks the *first* pending slot with that value, and asserts that it
was not already signalled or faulted: **two identical stamp values in flight
abort the host** (**measured**, `fifo_stamp_duplicate_signal_dies`,
`fifo_stamp_duplicate_of_faulted_dies`, `fifo_stamp_duplicate_fault_dies`). The
pump publishes only the value of the last contiguous signalled slot from the
head: completion is published in order. A faulted head stops progress, records
the fault offset (root: 0x103c; child: node +8), sets the fault-mask bit, and
raises the interrupt. Resume (0x1024/0x1028) treats the faulted head as
signalled. A fault offset >= 2^31 asserts; ring offsets stay below the ring
size, so a guest cannot reach it.

A resume while another FIFO is still faulted raises one interrupt of its own
before the resumed FIFO publishes anything: `DeviceCore::resume_fifo` and
`DeviceCore::resume_child_fifo` raise one when the fault mask, with the resumed
bit cleared, is still nonzero. With the 128-slot queue full on four FIFOs at
once (a faulted head and 129 NOPs each), each resume but the last gives that
interrupt, then one that publishes 128 stamps, then one for each stamp the
blocked push lets through (**measured**, `interplay_limits_pending_full_many`).

**Measured**, one NOP: the stamp slot becomes 1, 0x100c = 0xc, 0x1018 = 1 then
0, one interrupt. Three NOPs, one doorbell: stamp 4, 0x100c = 0x30. An
invalid opcode between two NOPs: stamp held at 1, 0x102c = 1, 0x103c = 0xc;
after 0x1024 the stamp jumps to 3 and 0x103c reads 0xffffffff.

Two error behaviours for display commands: **fault** (the stamp is not
signalled): opcodes 0, 1, 2, 4, 5, 6, 9, 11; **signal anyway** (the exception is
logged, the stamp signalled, no fault): 7, 8, 10, 12.

### 6.7 Child FIFOs

Node, 0x14 bytes, at root + 0x400 + (id − 1) × 0x14 (the child FIFOs are
`device/fifos.rs`):

| offset | size | field | written by |
|---|---|---|---|
| 0x0 | u32 | written | guest |
| 0x4 | u32 | readCount | host (0 at create) |
| 0x8 | u32 | faultOffset | host (UINT32_MAX at create and resume) |
| 0xc | u16 | stampIndex | guest; not checked for 0 or for uniqueness |
| 0xe | u16 | running | host |
| 0x10 | u32 | pageListPage | guest |

The page list is a page of u32 guest page numbers ending at the first 0 (at
most 4096); the ring is those pages, virtually contiguous. The guest writes
`written`, then MMIO 0x1020 = id. Opcode 48 (define) and 49 (delete) are
accepted only on the root FIFO. id 0 or >= maxFIFOCount faults; stampIndex
>= maxFIFOCount faults. Delete stops and drains the FIFO, then zeroes
**the next node's pageListPage** (W3, **measured**: root + 0x424 went from
0xdeadbeef to 0 on deleting FIFO 1), then tells every task (`fifo_deleted`).

**Measured** by the suite (`fifo_child_*`, `device_*`):
- A define whose page list cannot be mapped (an empty list, an unmappable page,
  and in the blocks setup a full 4096-page list) still signals, and the FIFO
  never runs: `Fifo::start`'s result is ignored (`fifo_child_page_list_empty`,
  `_unmappable`, `_full`).
- A define refused for stampIndex >= maxFIFOCount has already written the node:
  readCount 0 and faultOffset 0xffffffff, written before the check
  (`fifo_child_define_errors`).
- The stamp index is checked when a stamp is signalled, against the current
  `maxFIFOCount` with `>`. Lowering the version after a define silently drops
  the stamps of a stampIndex above the new maximum, with no interrupt, while
  exactly the maximum still publishes (`fifo_child_version_lowered`,
  `device_version_stamp_index_above_count`).
- A fault sets bit stampIndex of the fault mask, and 0x1028 clears bit n, the
  FIFO's id. With a stampIndex that differs from the id, resuming the FIFO
  publishes its stamp and leaves 0x102c reading 1, until 0x1028 names the
  stampIndex (`fifo_child_stamp_index_mismatch`).

**Measured** by the interplay and gap scenarios (`interplay_fifo_*`,
`interplay_life_*`, `device_gap_*`):
- A barrier's wait is registered under the waiting FIFO's stampIndex
  (`DeviceCore::add_wait`), but a child's pushed stamp is recorded in the
  waiting-info entry of its id (`DeviceCore::create_fifo_internal`). Three
  consequences: a child whose stampIndex differs from its id makes the
  circular-wait walk read 0 for it, a false "Circular wait" (section 7.1;
  `interplay_fifo_stamp_index_false_circular`); two children sharing a
  stampIndex that both wait kill the host, and so does defining a child whose id
  is a blocked FIFO's stampIndex (section 7.4).
- A child on stampIndex 0 shares the root's stamp slot and interrupt bit
  and works; it keeps a FIFO event of its own. In the service the task
  registers one FIFO event per stampIndex, the first it saw (the root's),
  so the child's exec signals the root's event with its own, lower count,
  and the root's next exec still completes
  (`interplay_fifo_stamp_index_alias_exec`).
- Deleting a faulted child publishes its faulted stamp and every stamp
  held behind it together, in one interrupt, and clears its fault bit
  (`interplay_fifo_delete_faulted_releases`). Deleting a child blocked on a
  barrier wakes and abandons the wait: the blocked command is never
  consumed and the node's readCount stays before it. Defined again, the
  child starts from readCount 0 and parses its ring from the start, so
  what it ran before runs again (`interplay_fifo_delete_blocked`,
  `interplay_life_delete_blocked_child`). A delete drains an exec the child
  had already read, which lands before the delete's own stamp publishes
  (`interplay_fifo_delete_exec_pending`).
- Defining a child that exists faults (`interplay_life_resume_rich`).

**Measured** by the limits and random-walk scenarios (`interplay_limits_*`,
`interplay_rand_*`, `mutgap_dev_*`):
- **At most 32 FIFOs wait at once.** `DeviceCore::add_wait` marks a waiter with
  the 32-bit `1 << waiter`, so waiter k and waiter k + 32 share a bit (the root
  and child 32 share bit 0), and a second waiter on a set bit kills the host
  (section 7.4). The circular-wait walk tests bit `next` of the 64-bit waiting
  mask, which is never set for a waiter above 31: a circular wait between
  children 40 and 41 is not detected, and both wait for good with no fault,
  while the same wait between children 8 and 9 faults the second (section 7.4;
  `interplay_limits_high_children_circular`). A waiter above 31 also makes its
  alias look waiting: with child 33 waiting (bit 1), a wait on idle child 1
  walks into child 1's empty entry, logs that the chain runs past the waiters,
  and simply waits (the same scenario). So `interplay_limits_children_suspend`
  runs all 63 children in two groups, 1-31 and 32-63.
- **In the blocks setup the root task's range fills.** The allocator (0xffffff
  bytes in 16 KiB granules) holds 1024 pages for the root node, the root FIFO,
  each local task's root page, each child's page list (for a moment) and ring,
  each display's shared page and each device info page. With it full, a child
  that cannot be placed still signals its define and never runs (`Fifo::start`'s
  result is ignored), and a display connect (1) and GetDeviceInfo (58) fault;
  deleting tasks frees their pages, and a child deleted and defined again then
  runs. The map and isolate setups allocate nothing there
  (`interplay_limits_root_range_full`).
- **Every unpause starts every child FIFO again**, mapping its page list again
  (`DeviceCore::unpause`, then each FIFO's `Fifo::start`), as a resume does. So
  W3 stops the child after a deleted one at the next `pause` and `unpause` (or
  `rvg_device_will_suspend` and `unpause`), not only at a resume: `Fifo::start`
  fails on its zeroed page list (logged), its node still says running, a NOP
  rung on it never runs, and a barrier on its stamp blocks the FIFO that waits.
  The guest recovers it by writing its page list again, then 49 and 48. The
  random walks meet it, and the programs that do lock it (`interplay_rand_11`,
  `_29`, `_32`, `_33`, `_41`, `_42`, `_44`, `_46` after an unpause; `_26`,
  `_36`, `_39`, `_41`, `_42`, `_44`, `_46`, `_49`, `_51`, `_52` after a resume).
- Child 63, the highest at version 245, is restarted by `unpause` and recreated
  by a resume (`mutgap_dev_child63_pause`, `mutgap_dev_child63_resume`). After a
  circular wait faults, the waiter's wait is finished
  (`DeviceCore::finish_wait`), and the same FIFO can wait again
  (`mutgap_dev_circular_then_wait`). A blocked barrier posts one stamp-wait
  report per timeout, then waits without one until a stamp is signalled
  (`mutgap_dev_barrier_report_once`, whose margins around the 2 s timer are
  0.5-0.75 s).

**What orders GPU work** (**measured** by the dependency-graph scenarios,
`interplay_dag_*`; `pgtest/dagwalk.py` says what the generator leans on):
- A FIFO publishes its stamps in command order (the stamp queue's pump), an
  exec's stamp is signalled when its command buffer completed, and a barrier
  holds its FIFO before the command is consumed until the slot reaches the
  value, each barrier of a command in list order (`DeviceCore::wait_stamps`, one
  `wait_stamp` each, `device/waits.rs`). So an exec runs after every exec whose
  stamp it waits on, and after their ancestors. The 45 seeded graphs and the
  hand-written shapes lock this over the root FIFO, 16 children and three tasks,
  with joins of up to 16 barriers, work held across a pause, a suspend and the 2
  s report, and children deleted and defined again between phases (a barrier on
  the old incarnation's stamp passes at once, since the child keeps its stamp
  count).
- **Same-task program order.** Execs on one FIFO that share a task are ordered
  with no barrier between them: they are command buffers on one queue, one after
  the other (in-process the device's exec queue; in the service the task's
  queue), and Metal orders them on the buffer they share.
  `interplay_dag_program_order_same_task` runs 800 increments of a counter and
  then a copy of it (as 55), behind a faulted head so that no stamp can publish
  alone, and the copy reads 800.
- **Different tasks on one FIFO are not ordered.** Two consecutive execs of
  different tasks on one FIFO, with no barrier between them, run at once: the
  second, a copy of a counter the first increments 1,100 times, read 0 after the
  first round, in-process and in the service (**measured**, and deliberately not
  locked: the result is timing). In-process each task's buffers are different
  `MTLBuffer`s over the same guest pages on the one exec queue, so Metal tracks
  no hazard between them; in the service the second task's command buffer waits
  only for the FIFO event the first signals when it is *scheduled*. Their stamps
  still publish in order. A guest that needs the order puts a barrier on the
  FIFO's own previous stamp, which is not circular (`DeviceCore::add_wait`);
  every edge of `interplay_dag` is such a barrier.
- **Repairing W3.** Writing the zeroed word back (+0x10) is enough: the next
  unpause's or resume's `Fifo::start` maps the ring again and the child runs,
  and a barrier on a deleted child's old stamps passes at once
  (`interplay_dag_redefined_chain`). Left unrepaired (above), the FIFOs whose
  barriers wait on the child stay blocked while independent work runs. The
  recovery by page list, 49 and 48 also writes back the word the 49 zeroes in
  the following node, and writes the lost command again at the start of the
  ring (the new FIFO reads from 0) with the same stamp: the chain completes
  and equals the control (`interplay_dag_w3_unrepaired`).
- **Waits across the 2 s report.** FIFOs blocked past the timeout have logged
  and posted a stamp-wait report and wait without a timeout; a pause wakes and
  quiesces them, the unpause starts every FIFO again, and they block again:
  nothing the guest sees changes (`interplay_dag_chain_pause_past_report`,
  `interplay_dag_diamond`). With child 33 waiting (its bit aliases child 1's),
  the report walks stamp index 1, finds child 1 idle with a next waiting index
  of UINT32_MAX, logs "StampIndex %u waiting on invalid next waitingIndex" and
  stops walking; the guest sees no difference (`interplay_dag_alias_report`).
- **`rvg_device_will_suspend` then `unpause` over blocked FIFOs** leaves two
  threads on each (section 6.24, and a death in section 7.4).
- **Not locked**: a child thread restarted by an unpause keeps its old cached
  written and read counts, and a guest that lowers the node's written count
  makes the thread's unsigned count underflow and parse again at once (**read**,
  and seen by the dependency-graph scenarios).

### 6.8 Opcodes

53 case labels: 38 live handlers and 15 deprecated ids, plus the default
(`fifo/commands/`). 49 payload-size checks (`>=`: longer is fine). Class: F = an
exception faults the stamp; S = logged, stamp signalled anyway; A = the task
completes asynchronously, and a status other than 1 faults at the latched
offset.

| op | name | payload (bytes) | gate (version) | class | rule |
|---|---|---|---|---|---|
| 0 | Debug | >= 1; the last byte is NUL-terminated | | F | |
| 1 | DisplaySetSharedStatePage | {u32 port, u32 page} (8) | | F | |
| 2 | DisplayAck | {u32 port, u32 ack} (8) | | F | |
| 3 | deprecated | | | fault | |
| 4 | DisplayCursorGlyph | port@0 taskID@4 vo u64@8 mappedLen u64@16 stride u64@24 w,h,hotX,hotY u16@32-38 sum u32@40 (44) | | F | taskID 0 |
| 5 | DisplayCursorShow | {u32 port, BOOL show@4} (8) | | F | |
| 6 | DisplayTransaction2_DEPRECATED | {port, surfaceRef, taskID} (12) | | F | taskID 0 |
| 7 | DisplayTransaction3 | port@0 taskID@4 surfaceRef@8 gammaVO u64@12 gammaLen u64@20 count@28 sum@32 (36) | | S | taskID 0 |
| 8 | DisplaySwapMapping | port@0 taskID@4 mappingID u64@8 gammaVO@16 gammaLen@24 count@32 sum@36 (40) | 10 | S | taskID 0 |
| 9 | DisplaySleepState | {u32 port, BOOL asleep} (8) | 21 | F | |
| 10 | DisplaySetProperties | {port, count} (8) + count × {u32 property, u32 wordCount, words} | 40 | S | |
| 11 | DisplayCompositorParameters | port@0 origin u16×2@4 epoch u64@8 scale u8@16 (20) | 108 | F | |
| 12 | DisplaySetGuestICCProfile | port@0 length@4 vo u64@8 mappedLen u64@16 (24) | 208 | S | length <= mappedLen |
| 30 | NOP | none | | always signals | |
| 31, 33, 35, 36, 38, 39, 41-47, 50 | deprecated (15) | | | fault | |
| 32 | DeleteTask | {u32 taskID} (4) | | F | < 256 |
| 34 | UnmapMemory | taskID@0 offset u64@4 length u64@12 (20) | | A | |
| 37 | DeleteResource | {taskID, ref} (8) | | A | |
| 40 | DeleteObject | {taskID, operationID, length} (12), payload >= length + 4 | | A | |
| 48 | DefineChildFIFO | {fifoID} (4) | | F | root FIFO only |
| 49 | DeleteChildFIFO | {fifoID} (4) | | F | root FIFO only |
| 51 | SetObjectList | {taskID, listPage (task VA page), listLength} (12) | | A | |
| 52 | InvalidateResources | {taskID, count} (8) + count × 8 | | A | |
| 53 | SynchronizeResources | {taskID, count} (8) + count × 4 | | A | |
| 54 | DeleteIOSurfaceBacking2 | {backingRef, taskID} (8) | | A | taskID 0 |
| 55 | ExecIndirect2 | {taskID, rc, cbc} (12) + rc × 24 + cbc × 16 | | A (setup errors F) | taskID != 0 |
| 56 | DefineTask2 | u32 (bit 0 unknown, bits 1-31 taskID)@0, length u64@4, taskRoot page u32@12 (16) | | F | < 256, new; local: length <= 0x800000000 |
| 57 | MapMemory2 | taskID@0 offset u64@4 length u64@12 (20) | | A | |
| 58 | GetDeviceInfo | {maxKey, maxCount, destinationPage} (12) | | F | |
| 59 | GetComputeInfo | {taskID, psoRef, maxKey, count, offset u64} (24) | | A | taskID != 0 |
| 60 | ReplacePhysical | {taskID, ref} (8) | | A | |
| 61 | Delay | u64 (8) | | **never signals** | |
| 62 | SynchronizeAndDiscardResources | as 53 | 10 | A | |
| 63 | DiscardResources | as 53 | 11 | A | |
| 64 | HeapTextureSizeAndAlign | taskID@0 replyVO u64@4 replyLen u64@12 payloadLen u32@20 (24) + payload | 43 | A | taskID != 0 |
| 65 | DeleteSharedTextureBacking | {ref} (4) | | A | kernel task |
| 66 | ResetRasterizationRateMap | {taskID, ref} (8) | 106 | A | taskID != 0 |
| 67 | ExecIndirect3 | {taskID, rc, cbc} (12) + rc × 8 + cbc × 16 | 109 | A (setup errors F) | taskID != 0 |
| 68 | SetObjectAndPlacementList | taskID@0 objVO u64@4 placeVO u64@12 objLen@20 placeLen@24 (28) | 207 | A | |
| 69 | InfoIndirect | as 67 | 235 | A | taskID != 0 |
| 13-29, 70-65535 | invalid | | | log, fault | |

Opcodes 9, 11, 8, 10 and 12 check their feature before the payload size.

### 6.9 Device info (58)

`write_device_info` (`guest/abi/device_info.rs`) writes `{u32 key, u32 value}`
pairs at the destination page while key < maxKey and count < maxCount, then {0,
0} if it still fits (maxCount 0 writes nothing; maxKey 5, count 2 gives two
pairs and no terminator, **measured**). Key order: 1-13, [14, 15, 16 with
metalHeaps], [17 with IOSurface buffers], 18, [level > 1: 19?, 20, 21?, 22?,
23-27?, 28], [level > 2: 33, 36?, 29-32, 34?, 41, 35, 37], [level > 3: 42, 44],
40, then 0. The last key written sets the suspend blob's device-info max key.

**Measured** on the M4 Pro, isolation off (34 pairs): 1=4, 2=0, 3=1024, 4=1024,
5=1024, 6=32768, 7=1, 8=1, 9=1, 10=8, 11=511, 12=1, 13=16, 14=1, 15=256, 16=1,
17=1, 18=0x20007, 19=1, 20=2, 21=1, 22=2, 23=1, 24=1, 25=1, 26=0, 27=1, 28=7,
33=0xbdf, 36=0xffffffff, 29=1024, 30=31, 31=65536, 32=16, 34=20 (the GPU core
count), 41=7, 35=2048, 37=1009, 42=0, 44=63, 40=16. With isolation on, key 33 =
0xbff (bit 0x20, argument buffers) and key 42 = 3, as the code predicts. Every
value comes from the host GPU and the host setup: key it by chip (section 5.8).
The suite's files, from the M3 Max, carry key 34 = 40
(`device_info_*@AppleM3Max-40c-26A434`).

`check_capabilities` (resume only) checks a saved blob's pairs against this
host: 43 defined checks (39 values, 4 flags, 3 alignments, 2 explicit), and
asserts on an unknown key. A resume blob cannot carry one: its parse refuses key
43 (InvalidContent), so the assert is not reached
(`suspend_capabilities_key_43`).

### 6.10 Tasks and memory

**Task root** (`APVTaskRoot2`, packed, 40 bytes):

| offset | size | field | written by |
|---|---|---|---|
| 0x00 | u32 | page-table root page | guest |
| 0x04 | u32 | page-table depth, 1-3 | guest |
| 0x08 | u64 | object list offset (task VA) | guest; host rewrites on opcode 51/68 |
| 0x10 | u32 | object list length in bytes (12 per entry) | guest; host rewrites |
| 0x14 | u64 | task length | **host** (W6) |
| 0x1c | u64 | placement list offset | host, service with argument buffers only |
| 0x24 | u32 | placement list length | same |

**Page table** (`memory/page_table.rs`): each level is one guest page of 4096
u32 entries; the VA page number splits top-down into 12-bit indices; VA width =
14 + 12 × depth bits (64 MiB, 256 GiB, 1 PiB). An entry's bits 0-30 are the
guest page of the next table or of the data; bit 31 marks a leaf writable; 0 is
a hole, which throws ("Found a hole in the page table"; holes are never
allowed). Task length must be at most 4096^depth pages ("Invalid Page Table
Depth").

- A page-table root of 0 is not checked at define time. The first walk dies
  in `read_range` (`memory/page_table.rs`) on a table page of 0: the host with
  isolation off; with it on, the task ends (**measured**).
- `read_range` ignores the read's result and the walker ignores the visitor's
  return: a table the host cannot read leaves the level's entries as they
  were, zeros at the start of each walk, so an unread root table is a hole
  and the command faults (`fresh_host_read_memory_refused`, section 7.5).

**Define** (56): `DefineTask2` payload in section 6.8; `word0 = taskID << 1`.
Effects: root word[taskID] = task root page (W4), task root +0x14 = length (W6)
(**measured**, all three setups). Refusals: taskID >= 256 ("Invalid task ID"),
existing ("Host task already exists"), length 0 ("Task Length Zero"), local and
length > 0x800000000 ("Task size too large"), depth 0 or > 3, length beyond the
table's coverage, mapping failure. **A task root page of 0 dereferences NULL in
the in-process task's creation: SIGSEGV of the host in the blocks and map
setups; in isolate the handshake fails and the command faults** (**measured**).

**What define reads, and what every walk reads.** The page-table root and depth
are read from the task root once, at define (`setup_task_root`, the local task's
in `tasks/local/new.rs` and the service's in `gpu_task/handshake.rs`), and the
length at +0x14 is the host's. The tables below the root are read again at every
walk. **Measured** by the guest-memory scenarios (`interplay_memlive_*`, every
setup unless the table below says otherwise):
- A guest that rewrites a live task's root in place (another depth, table root
  and length) changes nothing: objects first made afterwards are made over the
  old table's leaves, and a VA past the old length still faults. A second
  DefineTask2 (56) for a live task is refused ("Host task already exists"), so a
  task grows only by a delete (32) and a define over the rewritten root
  (`interplay_memlive_grow_refused`). An exec held on another FIFO across that
  delete and define names only a task ID, VAs and references, and runs on the
  grown task (`interplay_memlive_grow_held`).
- An intermediate entry pointed at a copy of its table, between execs or
  under a held one, is walked at the next use: a command buffer below it
  runs from the copy's leaf, and an object never made is made over it;
  with the entry cleared, an object never made faults ("Found a hole in
  the page table") (`interplay_memlive_intermediate_depth2`, `_depth3`,
  `interplay_memlive_held_intermediate_depth2`, `_depth3`). A walk reads
  whatever page an entry names, whichever task's table it came from: two
  tasks sharing a table or a subtree see each other's rewrites
  (`_shared_table_held`, `_shared_subtree`), and a task that maps a page
  table as a buffer and writes leaves into it with the GPU makes its next
  objects, and runs its next command buffers, over the leaves the GPU
  wrote (`_alias_page_table`, `_self_mapped_table`; the blocks setup's
  walks read guest RAM through the host's `read_memory`, which the GPU's
  writes reached).

**The memory map cannot change at runtime.** The device copies its config's
memory map once, at `rvg_device_create`, and the C API has no call that changes
it; QEMU builds it once, from the flat view of guest memory
(`apple_gfx_new_memory_ranges`, `vendor/qemu/hw/display/apple-gfx.c`). A range
can go only by a migration to a device created with a smaller map:
`rvg_device_did_resume` rebuilds each task from its root word, and a buffer
whose guest page is in no range of the new map faults at its next use, while one
in a remaining range works (**measured**, map and isolate,
`interplay_memlive_migration_drops_range`).

**Map and unmap** (57, 34): no-ops in the map and isolate setups (a misaligned
map succeeds there); in the blocks setup they walk the page table and call the
host's `map_memory`/`unmap_memory` (section 4.5). Unmap also unmaps overlapping
object and placement lists. The service's unmap does no range check and always
replies 1. A service task's map never contacts the service, so MapMemory2 on a
task whose service has died still signals (**measured**,
`interplay_iso_dead_task_commands`).

**A task whose service died** (isolate): every command that asks the service
anything gets a completion token that the dead connection signals with status 4
at once, and faults: 51, 68, 52, 53, 62, 63, 34, 37, 64, 55, 67, 69, 60, 40, 59
and 66. Only MapMemory2 (57) completes. Defining a child FIFO and deleting it
(48, 49) complete: the dead task's `fifo_deleted` completes at once. 56 on its
ID is refused ("Host task already exists"); 32 deletes it, and a task defined
again gets a new service and works. Tasks beside it, on any FIFO, never fault,
and their results equal a run with no death (**measured**,
`interplay_iso_dead_task_commands`, `interplay_iso_death_beside_healthy`,
`interplay_iso_death_on_child`, `interplay_iso_faults_stay_apart`,
`interplay_iso_four_deaths`). A suspend with a dead task completes (its flush
returns at once), and the resume rebuilds it from its root word with a new
service (`interplay_iso_suspend_with_dead_task`).

**Delete** (32): root word[taskID] = 0; an unknown task faults ("Failed to
delete task").

**Reads of task memory**: in-process tasks *copy* (`tasks/local/memory.rs`, a
page-table walk plus `DeviceCore::read_from_physical_address`); the service
reads *in place*, through the host's mapping of guest RAM. The copy rejects
`end >= taskLength`, off by one against the `>` everywhere else, the service's
in-place reads included. **Measured**: a record ending exactly at the task's end
faults in-process and is accepted by the service
(`tasks_objlist_record_at_end`); a command buffer ending there gets a null
cursor in-process and kills the host (section 6.12), and runs in the service
(`tasks_objlist_cmdbuf_at_end`, `exec_range_at_task_end`). An object list ending
there is refused in every setup, because the list check
(`objectListEnd >= taskLength`) is shared (`tasks_objlist_set`). Function data
is copied in the service too, and the service's copy rejects `end >= taskLength`
as the in-process one does: a function whose metallib ends exactly at the
task's end is refused in every setup ("Failed to get function data"), and the
nil function is a death (section 7.4; `mutgap_svc_function_at_task_end`). A
metallib that crosses a page is copied whole in every setup
(`mutgap_svc_function_across_pages`).

**Heap texture size** (64): replyLength in [16, 0x3fff] else "Invalid reply
length"; the serializer payload must be >= 0x28 bytes with u32 22 at +0 and a v1
serialized texture descriptor (section 6.19) at +8; the reply is
`MTLSizeAndAlign {u64 size, u64 align}` at replyVO, written only on success.
**Measured**: 256×256 RGBA8Unorm, ShaderRead, private → size 0x40000, align
0x80, equal to `heapTextureSizeAndAlignWithDescriptor:` in the test process. The
descriptor's `allowGPUOptimizedContents` bit (bit 6) reaches Metal, and Metal's
default is YES: on the suite's host the reply for a guest descriptor with the
bit clear is 0x40000/0x80, while the runner's query with Metal's default gives
0x40800/0x80 (**measured**, `tasks_heap_success`). A test that computes the
reply from Metal must set the bit as the guest did.

**Compute info** (59): the object list must map; psoRef must be a compute
pipeline; the reply, count × 8 bytes at the task VA, is {1,
maxTotalThreadsPerThreadgroup} if maxKey > 1, {3, threadExecutionWidth} if > 3,
{4, staticThreadgroupMemoryLength} if > 4, then {0, 0} if room. **Measured** on
the M4 Pro, both setups: {1, 1024} {3, 32} {4, 0} {0, 0}; the suite's M3 Max
gives the same (`compute_info_keys`). An unknown task, or task 0, faults
("Invalid task", "Invalid task ID for user task") before the handler's
`assert(task)`, which is unreachable (`compute_info_unknown_task`). Pipeline ref
0, count 0 and a read-only reply page differ by setup (the table below).

**Setup divergences** (each is behaviour to lock; section 7.2):

| input | blocks | map | isolate |
|---|---|---|---|
| MapMemory2 misaligned | fault | success | success (**measured**) |
| MapMemory2 valid | host `map_memory` calls, coalesced wrongly | no-op | no-op |
| UnmapMemory out of range | fault | fault | success (**measured**) |
| DefineTask2 with task root page 0 | host SIGSEGV | host SIGSEGV | fault (**measured**) |
| DefineTask2 length > 32 GiB | fault | fault | allowed up to coverage (**measured**, `tasks_define_too_large`, depth 3); 0x800000001, one byte past 32 GiB and not a page multiple, is refused in-process and defined by the service (**measured**, `mutgap_dev_task_length_one_past`) |
| command buffer ending exactly at task end | **host death** (null cursor) | **host death** | runs (**measured**) |
| record ending exactly at task end | fault | fault | accepted (**measured**) |
| info-encoder reply, render 0xa4, ICB CPU synchronize, ICB eviction at suspend | **host abort()** | **host abort()** | written (**measured**) |
| ResetRasterizationRateMap on an existing map | host abort() | host abort() | ok when mutable; an immutable map kills the service (**measured**); a record too short faults (**measured**), by the service's death (**read**) |
| ResetRasterizationRateMap on a map never created | fault | fault | success (**measured**); ref 0 faults in all three |
| exec with an empty resource list touching guest buffers | works | works | GPU "Insufficient Memory", fault (**measured**, detach-backing GPUs) |
| compute UseHeaps of a heap the exec does not list | not locked | not locked | fault (**measured**, `compute_err_unlisted_heap`) |
| placement list fields | never written | never written | written with argument buffers |
| GetComputeInfo (59), pipeline ref 0 | zero values written | zero values written | service asserts and dies (**measured**, `compute_info_ref_zero`) |
| GetComputeInfo, count 0, reply inside a page | fault ("zero length") | completes, nothing written | completes (**measured**, `compute_info_count_zero`; at a page boundary all three fault) |
| GetComputeInfo into a read-only page | reply written (in the blocks setup `LocalTask::mapped_address` ignores writability) | fault ("Mapping was not writable") | fault (**measured**, `compute_info_read_only_reply`) |
| buffer length not a page multiple (type 1) | mapped | refused, the exec faults | refused (**measured**, `resources_buffer_records_*`; a rate map's 0x6030-byte parameter buffer, which is why `mutgap2_res_info_453_exact_fit` runs in the blocks setup only) |
| private buffer (type 1, flag bit 0) | copied on synchronize | copied on synchronize | a Shared detach buffer: GPU writes land at once; FillTextureWithBytes (320) accepts it (**measured**, `blit_err_fill_texture_bytes_private`) |
| GPU fill of a read-only buffer | lands | lands | silently dropped (**measured**, `resources_readonly_buffer`) |
| GPU fill of a shared buffer over a read-only leaf (bit 31 clear) | lands in guest RAM: the host task's mapping is writable | silently dropped: the mapping of a range with a read-only page is read-only | silently dropped: the backing's ranges carry the leaves' read-only bit (**measured**, `mutgap_svc_readonly_leaf_buffer_write`) |
| a mapper-ref buffer (type 15) at a guest version below 10 | no IOSurface host device | the task asserts `mapperIOSurfaces`: **host death** | made through the host's delegate, and it works (**measured**, `mutgap_svc_mapper_below_v10`) |
| a memory map whose first range is zero-length and starts where the next starts | no memory map | the device and tasks work: the overlap check passes, since the zero-length range ends where it starts | the map is not whole pages, so a define of a task other than 0 makes no task and faults; the device and task 0 work (**measured**, `mutgap_mem_map_zero_length_first`, `_isolate`; the maps of `native_maps_tie_*` and `native_maps_both_accept_*`, whose zero-length ranges share starts, alike) |
| a buffer physically contiguous across two RAM ranges | the walk coalesces the pages into one physical range the host cannot map: the exec faults | lands | lands (**measured**, `memory_gap_ram_holes`, `memory_gap_many_ranges`) |
| a leaf rewritten after use, then a new buffer over it | lands in the old page (still mapped in the host task) | lands in the new page | lands in the new page (**measured**, `memory_gap_table_rewrite`; an existing object keeps its old page everywhere) |
| a memory map range ending 8 KiB into its last page | no memory map | used as given: a command buffer in the first half runs; a buffer over the page is refused, without a log: the path a buffer takes, `MemoryMap::new_range_mapping`, fails silently; the line for a range outside guest memory is `MemoryMap::map_memory_at_virtual_address`'s, a task mapping's path (**read**) | the map is not whole pages, so the define makes no task: the define and every later command on the task fault (**measured**, `memory_gap_partial_range`) |
| a memory map range the host gives no memory (NULL) | no memory map | objects over it fault, and a task root there faults its define; tasks elsewhere work | a define of a task other than 0 makes no task and faults, since the range is not at a non-NULL host address (**measured**, `memory_gap_null_host_map`, `memory_gap_null_host_define`) |
| a command buffer 2^62 bytes long | **host death** (longer than any local task: no cursor, then the missing cursor's death) | **host death** | the task ends: the exec and the next command fault (**measured**, `tasks_gap_cmdbuf_huge_length`) |
| a command buffer whose second page is outside RAM | the copy's failed read writes nothing: the pages after it move up, and the tail is zeros (section 7.5; not locked) | same | the task ends, reading in place through a NULL range (**measured**, `service_gap_cmdbuf_outside_ram`) |
| a first placement list (68) over a read-only leaf | accepted; the in-process list manager has no argument buffers and never records a placement list, and execs run | same | accepted, but the next exec cannot map it and faults, until a 68 moves it (**measured**, `service_gap_placement_unmappable`) |
| a private heap (13) and the buffers and textures placed in it | a host copy, a legacy placement heap paged by one blit, paged again only by invalidateHost: guest rewrites stay unseen, fills reach the guest pages only through synchronize, and 63 leaves it host-valid | same | a detach heap over the guest pages: every fill is in the guest pages at once and every read sees the guest's latest bytes (**measured**, `interplay_lifetime_heap_alias_buffers`, `_heap_discard_relist`, `_heap_deleted_under_buffer`) |
| GPU writes with flags 0, then a pause, guest rewrites of the pages, and a read after `unpause` | the private buffer and heap keep their host copies, not paged again: the GPU's bytes | same | the guest's bytes (**measured**, `interplay_lifetime_pause_resident`; invalidateHost pages the guest's bytes in every setup) |
| DeleteTask (32) after GPU writes listed with invalidateGuest | no flush: the private buffer's, heap's and texture's writes are lost, and the task defined again pages the guest's old bytes | same | the buffer's and heap's writes are in the guest pages; the texture, a host copy here too, loses its write (**measured**, `interplay_lifetime_task_delete_loses_dirty`) |
| DeleteTask (32) queued behind an exec whose kernel still runs (the guest holds it by a word of its memory) | the delete does not wait for the GPU: task 0's `unmap_memory` of the task's root page and `destroy_task` are called while the kernel runs, over RAM the host still has; the stamps still publish in order, the exec's first | nothing publishes while the kernel runs; after it, the exec's stamp, then the delete's | the same: releasing the service task drains the exec's outstanding completion token (**measured**, `fresh_res_delete_while_kernel_held`) |
| a second task defined over a live task's root page, sharing its object list | its private buffer is made again from the guest pages, which never saw the first task's GPU fill (the guest's 0x77) | same | the buffer is the guest pages, which hold the fill (0x5A) (**measured**, `interplay_lifetime_define_over_live_task`) |
| a resource discarded (63), then used by an exec that does not list it | a shared buffer's 63, and 60 of any kind, drop its Metal buffer: "Getting buffer from invalidated resource", a fault; a heap buffer's 63 does nothing; a heap's 60 makes it volatile and it stays host-valid | same | a detach resource whose discard detaches its backing: the GPU finds no memory under it and the exec faults (**measured**, `interplay_lifetime_discard_unlisted`; listed, every setup makes it or attaches it again) |
| a resource discarded (63), then a hole (leaf 0) punched under it, then listed | nothing faults: the host task keeps the old pages mapped | a shared buffer makes a new Metal buffer over the hole: "Failed to create buffer", a fault; a private buffer and a legacy heap run with flags 0, and fault with invalidateHost, paging through a new transfer buffer over the hole | each faults, attaching its backing again over the hole (**measured**, `interplay_lifetime_discard_then_hole`; with the leaves restored everything runs again) |
| a live object's leaf pointed at another page, then DiscardResources (63), then a read with flags 0 and one with invalidateHost | every read shows the old pages: the host task still maps them at those VAs; only an UnmapMemory (34) of the pages and another 63 bring the new pages in | flags 0 makes only the shared buffer again, over the new page (it has no Metal buffer); the private buffer, the heap and the texture read their host copies, the old bytes, until invalidateHost pages them from the new pages | the shared and private buffers and the heap are detach resources, attached again over the new pages at the next exec that lists them, whatever its flags; the texture, a host copy, only with invalidateHost (**measured**, `interplay_memlive_retarget_discard`, and decided at an exec's release, `interplay_memlive_held_discard_retarget`; across RAM ranges, `interplay_memlive_pages_move_ranges`) |
| two tasks' private buffers over one guest page, each filling its own part with invalidateGuest and synchronize (6) | a synchronize copies the whole host copy back over the page, so each write-back erases what the other task wrote since its copy was paged: the page ends with task 1's host copy, and task 2's bytes are lost | same | both buffers are the page itself: every fill survives (**measured**, `interplay_memlive_alias_writeback_race`; `interplay_memlive_alias_ordered`: a second task's private buffer, paged at its first read, misses the first task's later fill with flags 0 in-process, and sees it in the service) |
| InvalidateResources (52) clearing host-valid over GPU writes made with flags 0, after the objects' leaves moved | the next exec pages every host copy again from the pages bound at first use, not the new ones: the private buffer's, the heap's and the texture's unsynchronized fills are overwritten by the old guest bytes | same | only the texture's: its buffers and heap are the guest pages, which hold their fills (**measured**, `interplay_memlive_retarget_invalidate_52`; a 52 clearing guest-valid and a 53 then write each host copy back into the same old pages in every setup) |
| a reply ending one byte into a page (HeapTextureSizeAndAlign, 64, 16 bytes at 2 × PAGE − 15) | the lazy map asks the range tracker for the page run, which rounds with 0x3ffe (`RangeTracker::calculate_page_run_unaligned`): one page short, so only the first page is mapped and the 16th byte goes to the host task's unmapped page; the guest's byte there keeps its value (section 7.1) | written: the task maps with the end rounded up by 0x3fff | written, the same rounding (**measured**, `mutgap3_sys_heap_reply_one_past_page`) |
| the same reply with its last byte in a hole (the next page's leaf 0) | completes: the hole is never walked, and the last byte goes to the host task's unmapped page | faults | faults (**measured**, `mutgap4_svc_heap_reply_one_byte_into_hole`) |
| SynchronizeAndDiscardResources (62) on a shared buffer, then an exec that lists it | keeps the old pages | the next use walks the page table again | the next use walks the page table again (**measured**, `mutgap4_svc_sync_discard_rewalks`) |

### 6.11 Objects

No opcode creates an object. The guest writes records into task memory and lists
them in the **object list** (opcode 51 or 68, or the task root's fields at
define time); an object is created lazily the first time something references
it, and cached (`ResourceManager::get_object`, `resources/manager/`). A
**reference** is an index into the object list; **reference 0 is nil**.

**Object list entry** (`APVObjectEntry`, packed, 12 bytes, `guest/records.rs`):
`u32 {type: bits 0-7, record length: bits 8-31}`, `u64` task VA of the record.

**Placement entry** (`APVObjectPlacement`, 16 bytes): `u64` pinned GPU address
(buffers, heaps) or resourceIndex (textures, samplers, ICBs), `u64` resourceID.
One per object-list entry. Service tasks with argument buffers only; in-process
tasks always pass a null placement. A nonzero address pins the heap or buffer
there, and an address is written back only into an entry that holds zero: a
private heap and a private buffer of 2^36 bytes with pinned entries are not made
by Metal (the heap logs "Failed to create placed heap"), every exec that lists
one faults, and both entries keep the guest's pins (**measured**,
`mutgap2_res_heap_pinned_huge`). An isolated task shares the host's Metal
device, so a pin holds only where the process can still place an object at
that address: a buffer pinned before a suspend and made again after a resume
onto a fresh device in the same process is refused by Metal, and is not made
("Failed to create buffer host resource"; the exec faults,
`interplay_life_resume_rich`).

**Setting the list** (`tasks/object_lists.rs`) fails, logging "Error while
setting object and placement list" and zeroing both the manager's and the task
root's fields, when: an offset + length overflows; the object or placement
list's end is >= taskLength; the placement list is too short for count × 16; or
**the previous** object or placement offset is not 16 KiB-aligned (a call checks
the stored offsets before it stores the new ones, so the new values are tested
only on the next call). A zero-length list cannot be mapped, so an exec on a
task without a nonzero object list faults. **Measured**
(`tasks_objlist_placement`, every setup): a misaligned object list is accepted,
the next valid call is refused because of it, and the one after passes.
Placement offsets are stored only with argument buffers, so only in the isolate
setup does a misaligned placement list matter: it is refused, yet its offset was
stored first, so the next valid call is refused too; "Unable to do placement
list copy" zeroes the task root's fields, not the stored offsets. Opcode 68 is
accepted in-process. **Measured** in the isolate setup: a stored misaligned
placement offset is refused at the next valid call, the refusal zeroes the
stored offsets, and the call after it is accepted with nothing to copy, the old
list gone (`mutgap2_res_placement_list_stale_after_refusal`). A placement list
at offset 0 with a length is accepted, since offset 0 skips the size check and
the mapping; a 68 that then moves it finds an old list with a length and no
mapping, "Unable to do placement list copy": it faults and zeroes the task
root's four fields (`mutgap2_res_placement_list_at_zero`).

**Types** (the entry's type; creators under `resources/manager/`):

| type | object | record |
|---|---|---|
| 0 | invalid: "Attempting to fetch invalid object" | |
| 1 | buffer | `{u64 length, u32 task VA page, u8 flags}`, 16 bytes; flags bit 0 private, bit 1 read-only. Shared (0) wraps guest pages with no copy |
| 2, 3 | texture; 3 generates mipmaps | `{u64 length, u32 page}` (12) + dimension block 0x18 + mipLevels × 0x24 + operation header (commandType 1 or 52) + descriptor |
| 4 | IOSurface backing | `{u64 length, u32 page, u32 pixelFormat, u8 planeCount}` (pack 4, 0x14) + planeCount × `{u32 offset, u32 width, u32 height, u32 bytesPerRow:24 \| bytesPerElement:8}` |
| 5 | texture over a backing | `{u32 backingID, u32 taskID (0)}` + header (12/47/57) + IOSurface op |
| 6 | function | section 6.18 |
| 7 | serialized state: sampler (commandType 3), depth-stencil (4), compute pipeline (11), fence (13), render pipeline (14, tile 51), rate map (50) | header + op |
| 8, 9 | serializer texture: memoryless (1/52), view (7/8/27), on a buffer (9/55), in a heap (21/56); 8 and 9 are not told apart | header + op |
| 10 | none: "Unexpected object type" | |
| 11 | mapper-ref texture (IOSurface mapper) | `{u64 mappingID}` + header + op |
| 12 | dual-plane texture (feature dualPlaneTextures) | as 2, two dimension blocks |
| 13 | heap | `{u32 page @0, u64 length @4, u8 flags @0xc}`; bit 0 private, 1 tracked, 2 read-only. It is read as a 16-byte struct (`ObjectHeap`, two u64s, `resources/manager/records.rs`), so the record needs 16 bytes, not 13 (an 11-byte one faults, `resources_heap_records_*`; 13-15 are not locked). In-process heaps must be private (a shared one returns nil silently) |
| 14 | buffer in a heap | packed `{u32 heapRef, u64 offset, u64 length}` (0x14) |
| 15 | mapper-ref buffer | `{u64 mappingID}` |
| 16 | shared texture from a handle (supportsSharedTextures) | `{u32 backingID}` + header (49) |
| 17, 18 | shared texture backing, 1 or 2 planes | 12 + dimension(s) + header (48/53) + op |
| 19 | CPU buffer: not creatable; only an info reply target | `{u64 length, u32 page}` |
| 20 | indirect command buffer | `{u64 length, u32 page}` + header (54) + op 0x50 |

The operation header is `{u32 commandType, u32 unread}`. Every record is read
through a cursor bounded by the entry's length; short records throw.

**Measured by the suite** about objects:
- A serializer texture (8, 9) listed in an exec's resource list faults the exec:
  its prepare always throws "Attempting to prepare a view resource"
  (`textures_serializer_records`).
- A legacy heap's synchronize sets its host validity, not its guest validity, so
  a second 53 copies again (`resources_heap_paging`).
- Placements (service tasks with argument buffers): on the suite's host a
  texture's gpuResourceID equals its resourceIndex; mapper textures and heap
  buffers never write their entries back; a compute pipeline that does not
  support ICBs gets {0, 0}, which is what Metal reports, and one that does gets
  its index and ID (`resources_mapper_placement`, `textures_placement_*`,
  `compute_pipeline_placement`).
- Before `dualPlaneTextures`, D32S8 formats take Metal's pixel-format info as
  one plane: at version 40 Depth32Float_Stencil8 passes with 8 bytes a texel,
  and X32_Stencil8 is refused for every size tried, 8, 4 and 5 bytes
  (`textures_version_single_plane_depth_stencil`).

**Deletion** (40): the task receives the payload from operationID on; words[0]
selects: 1000/1001/1012 resource, 1002 depth-stencil, 1003 sampler, 1005
function, 1006 compute pipeline, 1007 render pipeline, 1009 fence, 1014 rate
map; others log "Invalid command for deletion"; a payload under 12 bytes returns
false silently; 1004, 1008, 1010, 1011, 1013 are unhandled. Every delete also
zeroes the placement entry (**measured** for a function and a sampler,
`mutgap_tex_placement_delete_function`, and for a buffer, whose entry 37 zeroes
and whose re-creation writes an address again,
`interplay_lifetime_placement_after_delete`). A delete forgets the object, not
its list entry: an exec that names a deleted buffer completes, and the task
makes the buffer again from its record (**measured**,
`interplay_limits_objects_heap`, after deleting 4096 buffers).

A reference of the wrong kind (a texture where a buffer is expected) asks the
object for an operation its kind does not have, which raises
(`Resource::unrecognized`); the decoder catches it, so a fault. An unknown or
invalid reference makes the delegate throw ("Failed to create <kind>") and
fault; the "nullable" decoder getters return nil only for ref 0.

**Measured** by the lifetime scenarios (`interplay_lifetime_*`, in every setup
unless the setup table of section 6.10 says otherwise):
- **A live object's entry and record are never read again.** The guest's rewrite
  of a cached buffer's entry or record changes nothing; after 37 the next exec
  makes it from the entry and record as they are then (another page, another
  length, private where it was shared,
  `interplay_lifetime_buffer_record_rewritten`). An entry retyped while its
  object lives changes nothing; retyped after a delete it makes the new kind: a
  private heap (13) where a buffer is expected has no `get_buffer` in-process
  (the kind's default raises: a fault) and is its cover buffer in the service; a
  texture is no buffer anywhere; type 0 is "Attempting to fetch invalid object"
  (`interplay_lifetime_buffer_retyped`, `_held_retyped_buffer_to_heap`). An
  object of the wrong kind made by an exec stays cached, so the entry made the
  right kind again still faults until a 37 forgets it
  (`interplay_lifetime_icb_recreated`).
- **One map per kind**: 40 with operation 1006 on a buffer's reference, and 37
  on a pipeline's, forget nothing, and one reference can hold two live objects
  at once: a cached buffer whose entry the guest rewrites into a compute
  pipeline's record is made a pipeline by the next dispatch while blits still
  find the buffer (`interplay_lifetime_one_reference_two_kinds`).
- **What holds storage, not a reference.** A heap buffer and a texture placed in
  a heap hold the heap's `MTLHeap`, a view holds its base's `MTLTexture`, and a
  texture on a buffer holds the buffer. Deleting the heap, base or buffer leaves
  them on the old storage, and the next exec that lists the deleted object makes
  a new one from its record: a new heap buffer placed at the same offset of a
  new heap does not alias the old one in-process, and the old one's fill never
  reaches the guest there (`interplay_lifetime_heap_deleted_under_buffer`,
  `_held_heap_deleted_heap`, `_held_heap_deleted_heap_buffer`,
  `_held_heap_texture`, `_view_outlives_base`). Two buffers, or a buffer and a
  texture, placed over the same bytes of one heap see each other's fills in the
  next exec (`interplay_lifetime_heap_alias_buffers`, `_heap_alias_texture`).
- **Work held across a delete.** An exec held on another FIFO's stamp
  while a worker FIFO deletes what it names makes it again from its
  record when released: a texture from its guest page, so its GPU fill is
  lost unless an exec had listed it with invalidateGuest and synchronize
  (`interplay_lifetime_held_texture_deleted`, `_synced`); a compute
  pipeline from its record, and a deleted function with its pipeline live
  changes nothing (`interplay_lifetime_held_pipeline_deleted`,
  `_function_deleted`); a pipeline whose entry was retyped 0 faults
  ("Failed to create ComputePipelineState",
  `interplay_lifetime_held_pipeline_retyped`), and one whose function's
  entry was retyped 0 is a death (section 7.4). A pipeline is made again
  over a function made again from a new entry only when both were
  deleted (`interplay_lifetime_function_swapped`). An exec names a task
  only by ID: held while the task is deleted and defined again over
  another root, it runs on the new task, its command buffer and objects
  (`interplay_lifetime_held_task_redefined`).
- **Paging flags across many execs** (`interplay_lifetime_paging_flags`,
  `_paging_flags_exec2`, `_texture_paging_flags`, `_private_across_fifos`).
  In-process a private buffer is paged at first use and again at each
  invalidateHost (1), from the pages it was bound to at first use (below), and
  otherwise keeps its host copy, whatever the guest writes; invalidateGuest (2)
  marks the guest's copy stale, and synchronize (4) writes the host copy back
  after the exec, over what the guest wrote since. A synchronize leaves the
  guest's copy valid, so a later exec's host write with flags 0 is never written
  back, even by a 53. InvalidateHost over a host copy that holds unsynchronized
  writes makes the guest's copy the valid one and discards them, even writes
  made from another FIFO. A normal texture behaves so in every setup; in the
  service buffers are the guest pages. ExecIndirect2 gives each flag a byte, cut
  to its low bit, so a byte of 2 is a clear flag.
- **Across a pause and a suspend.** A pause keeps every task and resource, and a
  host-valid texture and its view are not paged again
  (`interplay_lifetime_texture_pause`). A resumed device makes every object
  again from the guest: what `rvg_device_finish_suspend`'s flush did not write
  back is lost (`interplay_lifetime_texture_suspend`, `_heap_suspend`); an exec
  held at the suspend is parsed again and makes its objects from the records as
  they are after the guest's rewrites
  (`interplay_lifetime_suspend_held_record_rewritten`); and each task is rebuilt
  from the root its root word names, so a guest that points task 1's word at
  task 2's root gets task 2's page table and objects under ID 1, and task 2 is
  gone (`interplay_lifetime_suspend_root_word_rewritten`).

**Measured** by the guest-memory scenarios (`interplay_memlive_*`; the setups'
differences are rows of section 6.10's table):
- **An object is bound to its pages when it is made.** A shared buffer's Metal
  buffer, a private buffer's, a legacy heap's and a texture's transfer buffer,
  and the service's detach backing wrap the pages the table named at the
  object's first use. Nothing walks the table for a live object again: after the
  guest points its leaf at another page, or makes it a hole, reads and fills
  still use the old pages, and write-backs (invalidateGuest and synchronize)
  land there, in every setup, even for execs rung before the rewrite and running
  after it (`interplay_memlive_retarget_flags`, `_retarget_flags_exec2`,
  `_retarget_writeback`, `_hole_under_live`, `_inflight_rewrite`).
- **invalidateHost never rebinds.** It only marks the host copy stale, and the
  copy is paged again from the same old pages; a shared buffer and a detach
  resource are the old pages and page nothing. An object takes new pages only
  when it is made again: after a delete (37), or after a discard (63) drops what
  it was made over (section 6.10's table, where the blocks setup also needs an
  unmap). A guest that moves a live object's pages therefore copies it out,
  points the leaves at the new pages, unmaps the old VAs, discards it and lists
  it again, and ends equal to a task that never moved
  (`interplay_memlive_migrate_protocol`, `_unmap_retarget_map`).
- **Listing a reference makes the object.** An exec makes every object
  its resource list names when it runs, not when it is written: held
  behind a barrier while the guest points the leaves of objects not yet
  made at other pages, it makes them over the new pages, and live ones
  keep theirs (`interplay_memlive_held_first_use`); a hole there faults
  the exec at the release, and the next exec after the leaf is restored
  makes the object (`_held_first_use_hole_readonly`). Two tasks over the
  same pages alias whatever the tables say when each object is made
  (`_alias_held_retarget`, `_held_pages_move_ranges`).
- **Writability is fixed at creation.** A shared buffer made over a
  writable leaf takes fills after the leaf turns read-only; made again
  after a 63, or first made after the flip, it is read-only in the map
  setup and in the service, and the GPU's fill is dropped without a
  fault; the blocks setup maps it writable and the fill lands. Made
  read-only, it stays so when the leaf turns writable again, until
  another 63 (`interplay_memlive_readonly_flip`,
  `_held_first_use_hole_readonly`).
- **The object list is read through its mapping.** The list is mapped at its
  first use and the mapping is kept until an unmap (34) of its range or another
  51: a guest that points the list's leaf at another page changes nothing, and a
  reference resolves through the old list; after an unmap of the list's range
  the list is mapped again through the table
  (`interplay_memlive_held_list_page_moved`).

### 6.12 Execution (55, 67, 69)

Payload: `{u32 taskID (0 throws), u32 resourceCount, u32 cmdBufferCount}`, then
resourceCount resource records, then cmdBufferCount × `{u64 task VA offset, u64
length}` (`fifo/commands/exec.rs`). Overflow of the size arithmetic throws
"Overflow in CmdExecIndirect2/3" (69 says 3 too).
- Opcode 55's resource record is 24 bytes: `{u32 reference, u8 invalidateHost,
  u8 invalidateGuest, u8 synchronize, pad, u64, u64}`. Each flag byte is cut to
  its low bit when it is converted to the 8-byte form: a byte of 2 is 0, of 3 is
  1 (**measured**, `exec2_resource_flags`, every setup). In isolation each
  record goes to the service as its reference and one flag byte, and the service
  reads only that byte's bits 0-2 (**read**, `tasks/remote.rs`).
- Opcodes 67 and 69 use 8 bytes: `{u32 reference, u32 flags}`: bit 0
  invalidateHost, bit 1 invalidateGuest, bit 2 synchronize.
- The stamp is signalled when the Metal command buffer completes.
- **Each command-buffer range holds exactly one segment**: one 8-byte header is
  read, the cursor truncated to its length, and bytes beyond are ignored. A
  range under 8 bytes is skipped silently.
- **A missing cursor is a death**: an out-of-range range, a hole, an overflowing
  end, or a zero-length range makes no cursor, and
  `Deserializer::decode_segments` dies on it. **Measured**: the host dies
  in-process; under isolation the task ends, the exec faults and so does the
  task's next command (`exec_null_cursor_past_end`, `exec_null_cursor_hole`,
  `exec_null_cursor_overflow`). A zero-length range differs by setup: in-process
  one that is not page-aligned gets an empty cursor (the copy walks one page),
  which has no header and is skipped, so the exec completes; a page-aligned one
  kills the host; the service refuses every zero length ("Zero length cursor
  request") and the task ends (`exec_zero_length_range_aligned`,
  `exec_zero_length_range_unaligned` and their `_isolate` variants). Zero ranges
  run nothing and complete (`exec_ranges`).
- **InfoIndirect (69) never commits its command buffer**: its segments are
  decoded, so info replies land, and GPU work in them never runs (**measured**,
  `exec_opcodes`; in-process and in the service alike).
- In the isolate setup on a GPU with detach backing, every guest buffer the
  GPU touches must be in the resource list; otherwise the command buffer
  fails "Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)"
  and the exec faults. Flags 0, 1 or 4 all work; listing a pipeline faults
  "Unexpected object type (7) for resource" (**measured**). In-process an
  empty list works.
- After any decode error the command buffer is **never committed**: no
  segment of that exec runs on the GPU. Info replies written before the error
  stay in guest memory.
- **A held exec resolves at its release.** A FIFO copies an exec's payload
  before it waits on the command's barriers, but the command buffers are walked
  through the page table, and the objects made, only when the exec runs
  (in-process a copy; the service a cursor in place). **Measured**, in every
  setup: a command-buffer page the guest points at another page while the exec
  waits runs the new page's segment, and a segment across two pages is spliced
  from the old first page and the new second; a command buffer on a read-only
  page runs (it is only read) (`interplay_memlive_held_cmdbuf`); a table the GPU
  wrote while the exec waited is the one walked
  (`interplay_memlive_alias_page_table`).

### 6.13 Segments

**Segment header**, 8 bytes (**measured** offsets): `u32 length` (the whole
segment, header included), `u8 encoderType` (0 render, 1 compute, 2 blit, 4
info, 5 protection envelope), `u8 continuesPrevious`, `u8 continuedByNext`, `u8
pad`.

**Record header**, 8 bytes: `u32 commandID`, `u32 length` (record included),
then the payload through a sub-cursor. A longer payload is fine; a shorter
one throws "Failed to read" — and `sizeof` includes tail padding (e.g. 0x58
for `CopyFromTextureToBuffer`, not 0x56). A decoder loops while at least 8
bytes remain; 1-7 trailing bytes are ignored; a record length < 8 throws
"Underflow" (except in the render pass prefix, where it wraps and then
throws "Failed to read").

**Decoding a segment** (`deserializer/walk.rs`):
1. type 5: read a u64 of protection options and call the SPI
   `setProtectionOptions:` on the command buffer, then read the next 8-byte
   header and use it, **without** re-truncating to its length. If a read
   fails, it returns nil and drops a carried decoder without ending it.
2. continuesPrevious != 0: the type must match the carried decoder's, else
   "Encoder type doesn't match for continuation". With no carried decoder the
   type compared is 0, so a render segment (type 0) gets past
   that check and then throws "Previous stage didn't specify continuation"; any
   other type throws the mismatch first (**measured**; Apple's framework does
   the same).
3. continuesPrevious == 0 and no decoder: types 0, 1, 2, 4 create one; any
   other throws "Invalid encoder type".
4. continuesPrevious == 0 **with** a carried decoder: the carried decoder is
   reused whatever the type. Render and compute then open a second encoder over
   the open one, without ending it. **Measured**: Metal asserts ("A command
   encoder is already encoding to this command buffer"), so the host dies in
   every setup (`exec_carried_compute_reopened` and `_isolate`,
   `compute_death_second_encoder`, `render_segments_open_twice`). A carried blit
   decoder is reused for segments of any type.
5. continuedByNext keeps the decoder for the next segment. If the last segment
   of an exec says continuedByNext, the encoder is never ended before commit.
   **Measured**: Metal asserts in `encodeSignalEvent:value:` ("with uncommitted
   encoder") before the commit: a host death in every setup
   (`exec_open_encoder_at_commit` and `_isolate`, `render_segments_open_last`).
   A protection envelope while a carried encoder is open dies the same way in
   `setProtectionOptions:` (`exec_protection_envelope_open_encoder`).

Nothing checks a record's length against its command's size before the
command reads it: a record too short for its command fails at the read that
runs past its end (`next_record`, `deserializer/walk.rs`, gives each record a
cursor of its listed length).

### 6.14 Blit decoder (type 2)

`deserializer/blit.rs`: 24 commands, 300-323; anything else
"UnrecognizedCommand". The encoder is created at init and ended unless
continuedByNext.

| id | command | record | checks beyond reading and reference lookup |
|---|---|---|---|
| 300 | CopyFromBufferToTexture | 0x58 | |
| 301 | CopyFromBufferToBuffer | 0x20: src@0 dst@4 srcOff u64@8 dstOff@0x10 size@0x18 | |
| 302 | CopyFromTextureToBuffer | 0x58: tex@0 buf@4 origin@8 size@0x20 dstOff@0x38 bpr@0x40 bpi@0x48 slice u16@0x50 level@0x52 options@0x54 | |
| 303 | CopyFromTextureToTexture | 0x58 | |
| 304 | CopyFromTextureToTextureWithOptions | 0x5c | |
| 305 | CopyIndirectCommandBuffer | 0x20 | ICB refs non-null; range; destIndex < size |
| 306 | FillBuffer | 0x18: buf@0 location u64@4 length u64@0xc value u8@0x14 | range within the buffer ("Invalid buffer range") |
| 307 | GenerateMipmaps | 4 | |
| 308, 309 | OptimizeContentsForCPU/GPUAccess | 4 | |
| 310, 311 | the same, one image | 8 | |
| 312 | OptimizeIndirectCommandBuffer | 0x14 | ICB; range |
| 313 | ResetCommandsInBuffer | 0x14 | ICB; range |
| 314 | SynchronizeResource | 4 | nullable: an unknown ref is nil, not a throw |
| 315 | SynchronizeTexture (image) | 8 | |
| 316, 317 | UpdateFence, WaitForFence | 4 | fence non-null |
| 318 | CopyTextureToTexture (slices, levels) | 0x14 | |
| 319 | FillBufferWithPattern | 0x18, pattern u32@0x14 | range |
| 320 | FillTextureWithBytes | 0x4c | length >= the format's blockBytes (host info); buffer must be Shared; range. Reads `buffer.contents` at decode time |
| 321 | FillTextureWithColor | 0x5c: tex@0 level@4 slice@6 size@8 origin@0x20 color 4×f64@0x38 pixelFormat u16@0x58 | |
| 322, 323 | InvalidateCompressedTexture (image) | 4 / 8 | |

**Measured**: 306 filled bytes 0x10-0x2f of a shared buffer with 0xAB exactly;
306 + 301 in one segment gave 0x100 × 0xAB, zeros, then 0x00-0xff, in every
setup.

**Measured** by the mutation-gap scenarios (`mutgap_dec_blit_*`):
- A record whose length runs past the segment, after a record that left bytes
  unread (316 with 4 bytes of padding), fails `Cursor::sub_cursor`, and the
  decoder throws "Failed to read" before any command sees the failed cursor: the
  exec faults, nothing of it is committed, and the task keeps working
  (`mutgap_dec_blit_overlong_after_unread`).
- ICB ref 0 with an empty range, for 312, 313 and 305's source, is refused by
  the decoder's non-null getter ("Got invalid indirect command buffer") before
  the range check could accept it against a size of 0
  (`mutgap_dec_blit_icb_ref0_empty_range`).

### 6.15 Compute decoder (type 1)

`deserializer/compute.rs`: 33 commands. If the **first** record is 0xdb
(SetComputePassDispatchType) it is consumed as the pass descriptor; 0xdb
anywhere else is unrecognized. A nil encoder throws.

| id | command | record | checks |
|---|---|---|---|
| 0xc8 | DispatchThreadgroups | 0x30: tgPerGrid 3×u64, tptg 3×u64 | a pipeline set |
| 0xc9 | DispatchThreadgroupsIndirect | 0x24: tptg@0 off@0x18 buf@0x20 | pipeline |
| 0xca | DispatchThreads | 0x30: grid 3×u64, tptg 3×u64 | pipeline |
| 0xcb | SetBuffers | `{u32 location, u32 length}` + length × `{u32 ref, u64 offset}` (0xc each) | end <= maxComputeBuffers (host) |
| 0xcc, 0xcd | SetSamplers (with LOD clamps) | 8 + n × 4 / n × 0xc | <= maxComputeSamplers |
| 0xce | SetTextures | 8 + n × 4 | <= maxComputeTextures |
| 0xcf | SetBufferOffset | `{u32 index, u64 offset}` | index < max |
| 0xd0 | SetComputePipelineState | `{u32 ref}` | ref 0: "Invalid compute pipeline state" |
| 0xd1, 0xd2 | SetStageInRegion (indirect) | 0x30 / 0xc | indirect: buffer non-nil, offset 4-aligned |
| 0xd3 | SetThreadgroupMemoryLength | `{u64 length, u32 index}` | index < maxComputeLocalMemorySizes |
| 0xd4, 0xd5 | UpdateFence, WaitForFence | 4 | |
| 0xd6 | MemoryBarrierWithResources | count + count × u32 | "Too many resources" |
| 0xd7 | MemoryBarrierWithScope | 4 | |
| 0xd8 | SetImageblockWidth | 8 | supportsImageBlocks (host) |
| 0xd9, 0xda | SetBuffers / SetBufferOffset with stride | 8 + n × 0x14 / 0x14 | supportsDynamicAttributeStride, checked after reading |
| 0xdb | SetComputePassDispatchType | 4 | first record only |
| 0xdc-0xe2 | Encode Start/End DoWhile, While, If, Else | 0 or 0x14 | supportsCommandBufferJump; End* returning NO logs |
| 0xe3 | InsertCompressedTextureReinterpretationFlush | 0 | |
| 0xe4, 0xe5 | ExecuteCommandsInBuffer (ranged / indirect range) | 0x14 / 0x10 | ICB non-nil; range |
| 0xe6 | DispatchThreadsIndirect | `{u64 offset, u32 buffer}` | pipeline |
| 0x86 | UseHeaps | 4 + n × 4 | too many |
| 0x87 | UseResources | `{u32 count, u32 usage}` + n × 4 | too many |

**Measured**: `SetComputePipelineState(3)`, `SetBuffers(0, 1, {ref 4, 0})`,
`DispatchThreads(64,1,1 / 64,1,1)` with a host-compiled kernel `out[gid] =
gid * 2` wrote 0, 2, 4, ... 126 into guest memory.

**Measured by the suite** (`compute*`), locked as the GPU did it:
- The jump records pass their fields straight to Metal. EncodeStartIf on a guest
  word of 7 ran the else branch with Equal 7 and with Equal 8, and the if branch
  with NotEqual 7 (`compute_conditionals`): no pure inversion explains all
  three, and what Metal compares is **unknown**. Unbalanced records are all
  accepted, an End* that returns NO only logs (`compute_unbalanced_jumps`).
- An ICB's commands see its buffers only once UseResources makes them
  resident: ExecuteCommandsInBuffer without it writes nothing
  (`compute_indirect_command_buffer`).
- DispatchThreadgroupsIndirect and DispatchThreadsIndirect with indirect buffer
  ref 0 hand nil to Metal, which kills the host in every setup (section 7.4).
- 0xd6 with 256 refs and with 257 both succeed, and the dispatch after them
  writes its output (`mutgap_dec_compute_barrier_256_257`).
- 0xd8 sizes a kernel's explicit imageblock: 8×8, then 16×16, read back with
  `get_width()` and `get_height()` (`mutgap_dec_compute_imageblock`).

### 6.16 Render decoder (type 0)

`deserializer/render.rs`. When continuesPrevious is 0, a **pass prefix** comes
first: records until 0x1a, which ends it; any other id throws "Expected a render
pass descriptor command": 0x1e SetDefaultRasterSampleCount (4), 0x1f
SetDefaultColorSampleCount (4), 0x20 SetProgrammableSamplePositions (count ≤ 8,
then {f32 x, f32 y} each), 0x21 SetVariableRateRasterizationMap (ref), 0x22
ImageBlockSampleLength (4), 0x23 SetThreadGroupMemoryLength (4), 0x24
SetTileParameters ({u16 w, u16 h}).

**0x1a DescribeRenderPass**, 0x248 bytes (**measured** layout):

```
0x000 depth attachment, 0x28:  texture ref u32 @0, resolve texture ref @4,
      level u16 @8, slice @0xa, depthPlane @0xc, resolveLevel @0xe,
      resolveSlice @0x10, resolveDepthPlane @0x12, loadAction @0x14,
      storeAction @0x16, storeActionOptions @0x18, clearDepth f64 @0x1c,
      depthResolveFilter u16 @0x24
0x028 stencil attachment, 0x24: the same first 0x1c; clearStencil u32 @0x1c,
      stencilResolveFilter u16 @0x20
0x04c color attachments [8], stride 0x3c: the same first 0x1c;
      clear r, g, b, a as f64 @0x1c, 0x24, 0x2c, 0x34
0x22c u32 visibility result buffer ref
0x230 u64 renderTargetArrayLength
0x238 u64 renderTargetWidth
0x240 u64 renderTargetHeight
```

An attachment with texture ref 0 is skipped; the resolve texture only when
its ref is nonzero; `max_color_attachment_index` is the last colour attachment
with a texture. For memoryless textures, a store action outside {DontCare,
Unknown, MultisampleResolve} becomes DontCare and a load action outside
{DontCare, Clear} becomes Clear. A nil encoder throws.

**Main loop**: 95 commands: 0x00-0x19, 0x1b-0x1d, 0x65-0xa6; 0x1a and 0x1e-0x24
are invalid there. Highlights (the full table is the main loop's dispatch in
`deserializer/render.rs`):
- Draws 0x00-0x13 in 64-bit and 16-bit forms; e.g. **0x01 DrawPrimitives16**
  `{u32 primitive, u16 start, u16 count}` (8 bytes). All need a pipeline.
  Primitive type >= 10 throws, and **5-9 pass to Metal**; `indexType` is
  not validated. **Measured**: types 5 and 6 draw, 7 and 8 complete and
  draw nothing, and 9 makes the command buffer fail on the GPU, so the exec
  faults (`render_draw_odd_types`, `render_draw_type9`, which runs alone:
  section 5.10); index type 2 (and 0xffff) draws nothing. 0x0b's 16-bit
  base vertex is zero-extended, 0x1d's sign-extended
  (`render_draw_indexed`).
- 0x14, 0x15 ExecuteCommandsInBuffer; 0x16 RenderBarrierResources **always
  throws** "unsupported"; 0x17 barrier scope; 0x18, 0x19 fences; 0x1b
  UseHeapsWithStages.
- 0x65-0xa6: state: blend colour, store actions, depth-stencil (ref 0
  throws), cull, depth bias and clip, vertex/fragment/tile buffers, samplers
  and textures (bounded by the host's limits), winding, **0x74
  SetRenderPipelineState `{u32 ref}`** (ref 0 throws), scissors, stencil,
  tessellation, fill mode, viewports, visibility mode, texture barrier, use
  resources/heaps, line width, alpha test, point size, clip planes (index <
  6), provoking vertex, transform feedback, resolve textures, amplification,
  tile dispatches, 0xa4 GetTileDimensions (a reply into guest memory,
  section 6.17's rules).
- Wrong or odd checks, to lock (section 7): **0x6a** decodes as
  SetDepthStoreAction, not SetDepthStoreActionOptions (whose decoder is dead);
  **0x8e** bounds a fragment sampler index by maxVertexSamplers; **0x91** bounds
  a primitive-restart index by maxViewportCount; **0x96** requires index `<`
  maxColorAttachmentIndex, so with only attachment 0 bound index 0 always fails
  (0x66 and 0x67 use `>`). The 0x8e bound cannot show on the suite's host: both
  sampler limits are 16, and 0x8d and 0x8e fault at every index there anyway,
  because the GPU lacks the lodBias setter (`render_state_limits`).
- 0x87 UseResources looks refs up with the nullable getter: a ref that is not a
  resource (a pipeline) is nil, and the draw runs (`render_state_misc`).
  0x89 does the same, and Metal skips a nil entry of it too (**measured**: a
  ref past the list, then a triangle drawn as without it). Compute's 0x87 and
  0xd6 do the same, while UseHeaps (0x86) faults on a ref that is not a heap
  (`compute_err_resource_lists`). An empty entry (type 0) is nil to each of
  them too, except compute's 0x87 in a concurrent encoder (0xdb 1), where it
  raises ("resource lookup failed") and the exec faults, since Metal ends the
  process at a nil entry there (section 7.4;
  `interplay_lifetime_emptied_resource`,
  `interplay_lifetime_emptied_resource_render`).
- A pass with no attachment: with no target size, after 0x1e, Metal returns no
  encoder and the decoder throws "Failed to create render encoder"; with a
  target size and without 0x1e Metal aborts (section 7.4;
  `render_pass_no_target`).

**Measured** by the mutation-gap scenarios (`mutgap_dec_render_*`; host-keyed
where the GPU decides):
- Primitive type 10 in 0x02 and 0x04 throws "Invalid primitive type", a decode
  error: nothing of the exec is committed, not even a blit fill in an earlier
  segment (`_primitive_type_10`).
- 0x23 sets the pass's threadgroup memory, and 0x9c places a tile function's
  threadgroup memory in it; it persists across the tile dispatches of one pass:
  a read at offset 0 finds what an earlier dispatch stored there
  (`_tile_threadgroup_memory`). 0xa0 gives a tile function's sampler its LOD
  clamps (`_tile_sampler_lod`).
- The memoryless rewrite of load actions applies to stencil and depth and to a
  4-sample colour attachment: Load becomes Clear. A store action of Unknown is
  kept, and 0x66 (colour) or 0x78 (stencil) resolves it; without 0x78, Unknown
  does not store (`_memoryless_stencil_load`, `_msaa_memoryless`,
  `_stencil_store_action`). A 4-sample stencil resolves into a 1-sample texture
  named as its resolve texture (`_stencil_resolve`), and 0x96 with a second
  attachment bound moves a colour resolve (`_msaa_memoryless`).
- 0x1b with stages looks up every ref it counts: count 0 draws, a heap then a
  buffer faults (`_use_heaps_stages`). A record past the segment after one that
  left bytes unread throws "Failed to read" as in the blit decoder
  (`_overlong_after_unread`). In the service, 0xa4 with reply buffer 0 faults
  and the service survives it (`_tile_dims_after_null_reply`).
- Changing nothing on the suite's host, and locked as that: 0x94 and 0x95 (depth
  and stencil cleared hints), 0x17 between a fragment write and a vertex read in
  one pass (the vertex stages run first, and the barrier does not split the
  pass), 0x1f with counts that disagree with the pass, and 200 faulting execs,
  each releasing its encoder, after which a draw completes
  (`_depth_stencil_cleared`, `_fragment_vertex_barrier`,
  `_default_color_samples`, `_fault_releases_encoder`).

**Measured** by the `mutgap2_rdec_*` scenarios:
- **Residency.** An ICB's draw reads its vertex buffers only through the ICB, so
  it covers the target only after 0x89 UseResourcesWithStages names both, also
  as entries 1 and 2 after an unrelated buffer; a count of 0 is accepted. With
  the vertices in a private heap's buffer, the service's first draw from it
  needs 0x1b or 0x86 to name the heap, and a later exec draws without it (the
  residency outlasts the exec); in-process the heap's buffer is drawn from
  without either (`_use_resources_stages`).
- **Counts before lookups.** 0x86, 0x1b, 0x87 and 0x89 with a count of 2 and one
  reference throw "Too many resources specified" before any lookup, so nothing
  is made and a placement entry stays zero. 0x8e at index 16 (maxVertexSamplers)
  raises "Invalid sampler index" before its lookup; at index 0 the sampler is
  made and its placement entry written, and the exec then faults at the setter
  this GPU lacks (`_use_heaps_count_placement`, isolate).
- **Encoder state a draw shows.** 0x91 with {enabled, index} {0, 0} and {1, 0}
  draws alike, and unlike no 0x91; {0, 15} and {1, 15} cover the target; what
  the SPI's two arguments mean on this GPU is unknown. 0x8f with 0 leaves the
  clip-space triangle covering one pixel (`_restart_transform`). Without 0x90
  the first vertex provokes, with mode 1 the last (`_draw_state`). 0xa2 and 0xa3
  take their region in pixels: origin (0, 0) or (1, 0) at size 1×1 runs the
  first 16×16 tile, (16, 0) at 16×16 the second (`_tile_region_origin`).
- A memoryless depth attachment's Load becomes Clear too: its clear depth
  decides the depth test (`_memoryless_depth_load`).
- **Primitive type 9** passes the check, and the command buffer fails on the
  GPU: the exec faults, a blit fill in an earlier segment of it lands, and the
  pass's clear does not (`_primitive_type_9`). Type 10 throws in every draw
  form, and nothing of the exec is committed (`_primitive_type_10`). Both run
  alone (section 5.10). The two differ in what lands, not in the fault:
  `render_draw_type9`'s fault offset alone does not tell them apart.
- **Rate-map resets in the service** (66, `deserializer/rate_map.rs`): a
  two-layer mutable map resets, and so does a second 66
  (`_rate_map_reset_two_layers`). The reset's cursor runs from 12 bytes into the
  record to 4 bytes past its listed length: a 48-byte record listed as 44 bytes
  resets; listed as 43 it is one byte short, "Failed to read samples" kills the
  service, and the 66 and the next fault (section 7.4;
  `_rate_map_reset_short_entry`). An 8-byte record whose operation is not 0x32
  returns success at once (`_rate_map_reset_8_bytes`).

**Measured**: a 0x1a pass with colour attachment 0 = a type-8 texture over a
shared buffer (load Clear, store Store, 4×4), 0x74 with a host-compiled
vertex/fragment pipeline, 0x01 `{3, 0, 3}` → every pixel ff8040ff in guest
RAM, in-process and in the service.

### 6.17 Info decoder (type 4)

`deserializer/info.rs`: 20 commands, 450-469. There is no encoder. Replies are
**CPU writes into guest memory at decode time**, through a writable cursor over
the reply buffer object's own pages
(`ResourceManager::new_writable_memory_cursor_for_buffer`, which accepts object
types 1 and 19 only). The reply offset is applied with `advance`; an offset at
or past the end makes the write throw.

**In-process (blocks and map setups) every info reply aborts the host**: an
in-process task has no writable cursor over task memory
(`writable_cursor_from_virtual_offset`), and asking for one ends the process
(`tasks/local.rs`). **Measured**: 461 in the map setup exits 134. The same
reaches render 0xa4 and an ICB's CPU synchronize. In the isolate setup the
service implements it (`gpu_task/service.rs`), and 461 wrote 0x0000010000080000,
a GPU address (**measured**).

| id | command | record | reply |
|---|---|---|---|
| 450 | ComputePipelineStateInfo | `{pso, replyBuf, u64 offset}` 0x10 | 0x1c packed: u64 maxTotalThreadsPerThreadgroup (cut to 32 bits), u64 threadExecutionWidth, u64 staticThreadgroupMemoryLength, u8 supportICB, 3 bytes of zeros |
| 451 | HeapTextureSizeAndAlign | v1 descriptor 0x20, replyBuf@0x20, offset@0x24 (0x2c) | `{u64 size, u64 align}` |
| 452 | GetRasterizationRateMapInfo | 0x18 | 0x14 + layers × {u16, u16} |
| 453 | CopyRasterizationRateParameterBuffer | 0x10 | into a Metal buffer (Shared, in range, aligned) |
| 454, 455 | MapScreenToPhysical / PhysicalToScreen | 0x1c | {f32 x, f32 y} |
| 456 | MapPhysicalToScreenCoordinatesMultiple | 0x20 | rewrites count × 8 bytes of the reply buffer in place |
| 457 | RenderPipelineStateInfo | 0x10 | 0xc: u32 maxTotalThreads, u32 imageblockSampleLength, u8 threadgroupSizeMatchesTileSize, u8 supportICB, 2 padding bytes of zeros |
| 458, 459 | Render / Compute pipeline imageblock memory length | 0x28: pso@0, MTLSize@4, replyBuf@0x1c, offset@0x20 | u32 |
| 460 | ObjectUniqueIdentifier | `{ref, objectType, replyBuf, u64 offset}` 0x14; objectType 1 texture, 5 ICB, 9 render pipeline, 10 compute pipeline, others throw (the cursor is fetched first) | u64 |
| 461 | BufferHostResourceInfo | 0x10 | u64 gpuAddress |
| 462 | TextureHostResourceInfo | 0x10 | {gpuResourceID, resourceIndex} |
| 463 | HeapHostResourceInfo | 0x10 | u64 gpuAddress |
| 464-468 | Sampler, ICB, RenderPipeline, ComputePipeline, DepthStencil host info | 0x10 | {gpuResourceID, resourceIndex} |
| 469 | HeapTextureSizeAndAlign2 | v2 descriptor 0x27, replyBuf@0x28, offset@0x2c (0x34) | `{u64 size, u64 align}` |

Every reply value is host-assigned or host-derived (section 5.8). This is how
a guest builds argument buffers itself: no decoder has an argument-buffer
command; the host contributes GPU addresses, resource IDs and the placement
list.

**Measured** by the mutation-gap scenarios: 453 at an odd offset fails the
alignment mask ("Invalid buffer alignment") and leaves the buffer untouched, in
every setup (`mutgap_dec_info_parameter_offset_1`). In the service, a reply that
fills the end of a guest page whose next VA page is not adjacent in RAM goes on
at the start of that next page (`mutgap_mem_reply_split`), and a mutable rate
map made by a 452 query has physical size 0 and no mapping until 66 resets it,
while the record's mutability and minFactor do not change what 452-454 then
report (`mutgap_dec_rate_map_reset_min_factor`).

### 6.18 Functions and pipelines

**Shaders are metallib bytes.** `new_function_with_ir`
(`resources/manager/create/function.rs`) wraps the guest's bytes in
`dispatch_data` and calls `-[MTLDevice newLibraryWithData:]`, then requires
exactly one function. It never parses or patches the bytes. The device, and
each isolated task's service, call `allowLibrariesFromOtherPlatforms` on the
Metal device they share.
**Measured** (Apple metal 32023.883):

| input | result |
|---|---|
| a one-function metallib from `xcrun -sdk macosx metal -c` + `metallib` (kernel 3488 B, vertex 3582 B, fragment 3118 B, magic `MTLB`) | accepted; runs through the decoders (sections 6.14-6.16) |
| built for macOS 11 (`-std=macos-metal2.3`) or with `-sdk iphoneos` | accepted |
| raw `.air`, a truncated metallib, random bytes | "Invalid library file" → "Failed to create library" |
| two functions in one metallib | "Unexpected function count in library (2)" |
| an unspecialized `[[function_constant]]` | accepted as a function; building a pipeline from it **aborts in Metal** |

**Function record** (type 6, >= 20 bytes): `u64` task VA of the metallib, `u32`
its length, `u32` commandType 15, `u32` unread. The bytes are copied out of task
memory (`data_from_virtual_offset`).

**Pipeline records** (type 7): `u32 commandType` (11 compute, 14 render, 51 tile
render), `u32` unread, `u32` unread, `u32 descriptorLength`, then the descriptor
(`resources/manager/create/pipeline.rs`). The compute and render paths hand
Metal *entry length − 16* bytes, not descriptorLength, and the tail past
descriptorLength is zeros. Metal ignores trailing bytes (**measured**); the
suite's records are exactly 16 + descriptorLength all the same. The tile path
passes descriptorLength and leaks its buffer.

**The descriptor is Metal's private format**, passed to the SPI
`newComputePipelineDescriptorWithSerializedData:deserializationContext:` (and
render, tile). It appears to be `MTLSerializer::SerializedCompactPropertyList`
(the symbol in an abort backtrace). As decoded on this host (**measured**): a u8
count, then entries `{u8 key, u8 size = 4, u32 value}`; nested blocks and
strings follow the entries, and a value gives their offset from the blob's
start; an array is a u32 count then u32 offsets. A function is written as its
`functionRef` (u32), which Metal resolves through the context's
`functionWithFunctionRef:`; the device's context resolves it as an
object-list reference. A failed lookup resolves to nil, except three, which raise
the function lookup's failure: the exec faults ("function lookup failed")
rather than reaching Metal's abort on a nil function (`resources/manager/resolver.rs`).
- One whose entry is empty (type 0). A guest empties a task's entries when the
  process that owns it ends, without waiting for the execs that process
  queued, so an exec the host runs after that end can meet them
  (`interplay_lifetime_held_pipeline_function_retyped`).
- One whose IR sits over a hole in the task's page table. A guest commits a
  buffer's pages as it prepares a command buffer that lists the buffer, so a
  hole is IR the guest has not mapped, and nothing is cached: a lookup once the
  guest has mapped it makes the function (docs/qemu.md, "x86-64 macOS guests:
  WebGPU render bundles"; `mutgap_svc_function_over_hole`). IR the task cannot
  otherwise copy (past the task's end, or ending at it) is still nil to Metal.
- One whose library, or the function itself, the platform refuses for a gap
  of its own or of its device's, where Metal makes it: the Vulkan platform
  answers such a library its runtime's refusal as the `NSError`, and such a
  function nil (`inexact-refused-create`), and notes the relay for core
  (`platform::device::relaying_gaps`). The state that names the function is
  never made, so the exec or info that names the state faults, as where the
  platform refuses the state itself for a gap. Metal never refuses there, so
  the suite cannot reach it; core's unit tests lock it
  (`a_state_naming_a_function_refused_for_a_gap_is_never_made`). Metal's own
  refusal of a function's IR is still nil to Metal.

| blob | bytes (`<K>` = a function ref, u32 LE) |
|---|---|
| compute, function only (7 B) | `01  00 04 <K>` |
| + threadGroupSizeIsMultipleOfThreadExecutionWidth | `02  01 04 01000000  00 04 <K>` |
| + maxTotalThreadsPerThreadgroup 64 | `03  01 04 01000000  08 04 40000000  00 04 <K>` |
| render: vertex V, fragment F, colour[0] format P (41 B) | `03  08 04 14000000  01 04 <V>  02 04 <F>  00  01000000 08000000  02  00 04 00000000  01 04 <P>` |
| tile: function T, threadgroupSizeMatchesTileSize, colour[0] RGBA8 (41 B) | `03  04 04 01000000  03 04 14000000  01 04 <T>  00  01000000 08000000  02 00 04 00000000 01 04 46000000` |

Adding depthAttachmentPixelFormat adds key 9; rasterSampleCount adds key 4;
colour blending adds key 2 in the attachment's list. **To author richer
descriptors**: serialize with the host's `-[MTLDevice
serializeComputePipelineDescriptor:]` (and render, tile), passing an `NSProxy`
for each function whose `-functionRef` returns the object-list index wanted;
commit the blob as a fixture (round trip **measured** for compute, render and
tile). The format belongs to Metal and may change with macOS (**unknown**),
which is one reason expected files are keyed by macOS build.

**Metal aborts the process** (the host, in every setup) on (**measured**): 64
bytes of garbage; a valid blob truncated to 1 or 5 bytes; a compute descriptor
whose function ref resolves to nil; a render pipeline built from a descriptor
with a nil vertex function. An empty blob returns nil (no abort), and the device
then raises a pipeline-creation error carrying Metal's message. A kernel used as
a vertex function gives an error after seconds of retries. These are death
tests.

Pipelines are reached by render 0x74, compute 0xd0, info 450/457-460/466/467,
opcode 59 and ICB prepare. Deletion: 1005 function, 1006 compute, 1007 render.

### 6.19 Textures and other records

**Serialized texture descriptor v1** (32 bytes,
`deserializer/texture_descriptor.rs`): `u32 {textureType:4, framebufferOnly:1,
isDrawable:1, allowGPUOptimizedContents:1, :1, usage:8, pixelFormat:16}`, `u32
width`, `u32 height`, `u32 depth`, `u16 mipmapLevelCount`, `u16 sampleCount`,
`u16 arrayLength`, `u16 resourceOptions`, `u64 protectionOptions`. Inside the
pack(4) op structs it sits at +4. **v2** (0x27 bytes, packed): the 24 bits of
bit-fields (with writeSwizzleEnabled in bit 7) take 3 bytes, then `u32 usage`
@3, width @7, height @0xb, depth @0xf, `u16` mips @0x13, samples @0x15,
arrayLength @0x17, resourceOptions @0x19, `u64` protection @0x1b, swizzle RGBA
@0x23-0x26 (a channel >= 6 becomes One; **measured**,
`mutgap3_res_v2_swizzles_sampled`: a green or blue channel of 6 samples as 0xff,
and alpha in red's place samples alpha). The v2 offsets are derived from the
packing, not measured.

**Texture on a buffer** (type 8, commandType 9; **measured**): `{u32 9, u32
unread, u32 unread, u32 bufferRef, u64 offset, u64 bytesPerRow}` then a v1
descriptor (e.g. 2D, usage 5, RGBA8Unorm 70, 4×4×1, 1 mip, 1 sample, 1
layer). A 2DArray takes `newLinearTextureWithDescriptor`, else
`newTextureWithDescriptor:offset:bytesPerRow:`. Through commandType 55 (a
v2 descriptor) a 2D array of two slices at buffer offset 0x200, bytesPerRow
16, puts its slices 0x80 apart: the second copies out the buffer's bytes
at +0x280 (**measured**, `textures_gap_serializer_v2`).

**Normal textures** (2, 3, 12) are validated by `validate_texture_dimension`
(`resources/texture/dimension.rs`), 27 exits: plane count vs flag 0x800000,
allocation size (a u32: over 4 GiB fails), array length, mip count, cube,
compressed block math, per-mip row and size checks. Paging forces one mip level
for type 3 only (`generate_mipmaps = object_type == 3`,
`resources/texture/blit.rs`): type 3 pages level 0 and generates the rest, even
when its record describes more levels; type 2 pages every level (**measured**,
`textures_mip_levels`). Memoryless storage is rejected for normal textures and
required for serializer textures. Type 12 with formats 550/551 pages through the
`*_2P_XR10_A8` compute kernels.

**Samplers** (type 7, commandType 3), op 0x1c: `u32 unread; u32 {minFilter:2,
magFilter:2, lodAverage:1, compare:3, s:4, t:4, r:4, border:4, mipFilter:2,
maxAnisotropy:5, normalized:1}; u32 {supportArgumentBuffers:1, reductionMode:3};
f32 lodMin; f32 lodMax; 8 bytes unread` (bit packing inferred from the record's
asserted size, not dumped). **Depth-stencil** (4), op 0x20: `u32 unread; u32
{compare:3, write:1, hasFront:1, hasBack:1}; front {u32 {compare:3, fail:3,
depthFail:3, pass:3}, u32 readMask, u32 writeMask}; back {same}`. **Rate map**
(50), op 0x14: `u32 unread; u16 screenW; u16 screenH; u32 layerCount; u32
mutability (1 → mutable with minFactor); f32 minFactor`, then layerCount × {u16
h, u16 v}, then each layer's floats. **ICB** (20): descriptor 0x14, layout 0x34,
`u32 maxCommandCount`, `u16 options`; paged by the CPU both ways (W13).
**Measured** (`icb_*`; write-back only in the service, since in-process it
aborts, section 7.4):
- Once prepared, an ICB ignores its guest memory until it is invalidated (opcode
  52 or `invalidateHost`); DiscardResources (63) does nothing
  (`icb_bind_lookups`, `icb_prepare_invalid_guest`).
- A compute ICB's write-back sizes its stride cursor by the kernel bind count,
  as prepare does, whatever the descriptor's vertex bind count: with 31 vertex
  binds the ICB prepares, runs and is written back whole (`icb_sync_errors`),
  and with 2 or none both kernel binds and both strides are written back
  (`mutgap_tex_icb_sync_more`, which also locks an indexed-patches command's
  baseInstance and tessellation factor offset, and a render ICB written back
  with no strides cursor). Apple's GPU task service does the same (**read**,
  macOS 27.0.1, 26A434: `synchronizeIndirectComputeCommandBuffer` makes the
  strides sub-cursor at the layout's stride offset with the kernel bind count
  × 8 bytes, as it makes the kernel binds' with the kernel bind count × 20).
- Prepare resets a render command whose guest type is 0: a command that drew,
  rewritten to type 0 and invalidated, draws nothing
  (`mutgap_tex_icb_reset_again`).
- A compute slot of command type 0 writes back only its type; a render slot of
  type 0 still writes its pipeline and binds (`icb_sync`).
- A compute command's write-back writes each threadgroup memory length to its
  own word from the layout's lengths offset, where prepare reads them: a
  command prepared with lengths 0x40 and 0x80 reads 0x40 and 0x80 back, and
  one with none reads 0 in each word (`icb_sync`, `icb_evict`). The write-back
  takes one cursor over all of a command's lengths first: lengths past the
  command end it there, leaving that command's barrier and arguments and the
  later commands unwritten, the error dropped as below
  (`synchronize::tests::lengths_past_the_command_end_the_walk`). Apple's GPU
  task service does the same (**read**, macOS 27.0.1, 26A434:
  `synchronizeIndirectComputeCommandBuffer` makes one writable sub-cursor of
  the kernel threadgroup memory count × 8 bytes at the lengths offset, ends
  the walk when that fails, then writes each length with its own 8-byte
  write), and Metal keeps a length for each index of each command (**measured**
  with a probe, M3 Max, macOS 27.0.1).
- Write-back errors are dropped: the synchronize completes, and the guest copy
  is marked valid (`icb_sync_errors` locks the completion).
- A failed prepare only logs: the ICB stays host-invalid, the commands encoded
  before the failing one run, and every exec that lists it prepares it again
  from guest memory, so a guest that rewrites the refused command then gets both
  (`mutgap2_res_icb_prepare_fails_again`).
- A bind with reference ~0 is resolved by its GPU address, to the highest
  registered address at or below it. A service task registers a mapper-ref
  buffer (type 15) before its Metal buffer exists, so as address 0: a bind at
  address 0x40 writes the mapped surface's page at 0x40. In-process nothing is
  registered, prepare stops before the command, and the page keeps its bytes
  (**measured**, `mutgap2_res_icb_mapper_bind_by_address`).

**Measured** by the lifetime scenarios about ICBs (`interplay_lifetime_icb_*`):
- **A prepared ICB outlives what it names.** Its Metal command holds the bound
  buffer's GPU address and the pipeline. After the buffer is deleted (37) and
  made again by the next exec, the ICB run as prepared (flags 0) writes through
  the deleted buffer's address, and the writes land in the buffer's guest pages
  in every setup, since the old buffer wrapped the same pages; after its
  pipeline is deleted (40, 1006) it still dispatches with it. Nothing faults
  (checked in the system log while blessing; `_bound_buffer_deleted`,
  `_pipeline_deleted`, both exclusive, section 5.10). Listed with invalidateHost
  it is prepared again, and the bind's lookup through `get_object` makes a
  deleted buffer or pipeline again from its record: the same bytes (`_rebound`,
  where the ICB's prepare is what makes the buffer).
- A deleted ICB is made again from its guest memory, so the next exec runs the
  guest's rewritten command; retyped to a buffer after a delete it faults (a
  buffer has no `get_indirect_command_buffer`: the kind's default raises) until
  another delete (`_recreated`).
- **Across a suspend** the resumed device makes the ICB again from its guest
  memory and runs the guest's rewrite (`_suspend`); listed with invalidateGuest,
  `rvg_device_finish_suspend`'s flush evicts it through the CPU and writes the
  prepared command back over the rewrite, so the resumed device runs the old
  command (`_suspend_evicted`, isolate only: in-process the eviction aborts,
  section 7.4).
- **Write-back after deletes** (isolate): the service resolves a bind from its
  GPU address through the addresses buffers registered at creation, and a
  pipeline from its unique identifier, and deletes never unregister. A deleted
  bound buffer is "Resource not found for GPU address", written as reference ~0
  and offset 0; once the buffer is made again, the old address finds the old
  entry, whose reference names the new buffer, and the bind is written as
  before. A deleted pipeline is still named by its reference; a second pipeline
  built from an identical record registers the unique identifier Metal gives it,
  a new one on this host, so the command still names the first reference
  (`_writeback_after_delete`, host-keyed).

**Measured** by the mutation-gap scenarios about textures and paging
(`mutgap_tex_*`):
- The compressed path of `validate_texture_dimension` refuses a block height
  that does not match the format ("Compressed texture block size mismatch") and
  a mip whose height or depth is wrong ("Size mismatch at mip(0)")
  (`_compressed_dims`). A BC1 3D texture pages depth slice z from guest offset
  bytesPerRow × ceil(height / blockHeight) × z (`_bc3d_depth_slices`).
- A normal texture keeps the transfer buffer of its first prepare: after the
  guest moves its VA page to another RAM page, an invalidated prepare still
  reads the old page, in every setup (`_leaf_rewrite`).
- A backing plane may end exactly at the allocation length; one byte further is
  refused (`_backing_exact_fit`). A backing (type 4) and a shared texture
  backing (type 17) start guest-valid, so 53 before any GPU write leaves the
  guest's pages alone (`_backing_guest_valid`, `_shared_guest_valid`).
- Array and cube slices page at bytesPerSlice × slice: slice 6s + f for face f
  of cube s, for shared backings and for two-plane textures (`_shared_2d_array`,
  `_shared_cube_array`, `_dual_plane_cube_array`). A two-plane shared cube pages
  faces 0-5 only (`_shared_dual_plane_cube`).
- In the service a buffer whose page is in a RAM hole cannot have its backing
  replaced, and the exec faults in its prepare, before any of it runs
  (`_detach_buffer_hole`). A read-only heap in the service drops GPU writes to
  itself and to buffers in it (`_readonly_heap_write`). In-process, 63 on a
  legacy heap makes it volatile, and the next prepare finds it host-valid and
  does not copy the guest pages over it (`_legacy_heap_discard`).
- A mapper-ref buffer listed twice in one exec retains its surface once, so
  after 37 deletes it the surface unmaps (`_mapper_buffer_twice`).
- Placements: a texture's nonzero entry is reserved when the lists are first
  mapped, so a texture made before it with a zero entry gets another index; a
  sampler's reservation changes nothing on this host. Views placed by pixel
  format, type and swizzle are made at their indices; textures with no placement
  entry are made with no index and sample their own texels
  (`_placement_reservations`, `_placement_views_used`,
  `_unplaced_textures_sampled`; host-keyed).

**Measured** by the `mutgap3_res_*` scenarios:
- `validate_texture_dimension` counts six slices for a cube: a 4×4 RGBA8 cube
  fills 0x3c00 of a one-page allocation, and a mip's expected width is clamped
  to 1 after each halving, on the uncompressed and the compressed path
  (`_texture_record_clamps`).
- A shared texture backing that is a 3D texture pages depth slice z from
  bytesPerRow × ceil(height / blockHeight) × z (`_shared_backing_3d`).
- A sampler's lodMinClamp reaches Metal: 1.0 over a two-level texture samples
  level 1 where the kernel asks for level 0 (`_sampler_lod_min`).
- Info 469 (HeapTextureSizeAndAlignWithDescriptor2) passes the v2 descriptor's
  sampleCount to Metal: a 256×256 RGBA8 render target replies 0x40000 as 2D with
  one sample and 0x100000 as 2DMultisample with four, both aligned 0x80
  (`_heap_size_v2_samples`, isolate, host-keyed).
- Paging state: an IOSurface backing's synchronize marks its guest copy valid,
  so a second 53 writes nothing over the guest's later bytes
  (`_backing_sync_twice`, in-process). A compute-paged two-plane texture (format
  550, type 12) starts guest-valid: after a fill listed with flags 0, a 53 finds
  nothing to synchronize and the guest's bytes stay (`_xr10_host_only_fill`). A
  legacy heap keeps the transfer buffer of its first prepare, as a texture does
  (`_legacy_heap_leaf_rewrite`, and section 6.11).
- ICBs: prepare resets a compute command whose guest type is 0, as it does a
  render one (`_icb_compute_reset_again`). The compute write-back writes as many
  commands as the ICB has, not as the guest length has room for: the guest's
  bytes in a spare slot stay (`_icb_sync_spare_slot`). A render ICB's
  inheritViewports bit reaches Metal, and both ICBs, with the bit and without,
  draw only in the viewport the encoder set (`_icb_inherit_viewports`).
- Placements (isolate, host-keyed): a render pipeline whose entry has a nonzero
  resourceID is made at the entry's resourceIndex by force and writes its index
  and ID back: an entry {3, 1} reads {3, 3} (`_render_pipeline_forced_index`).
  Two render pipelines can hold one index: one made first with a zero entry is
  given index and ID 1, and one whose entry is {1, 1} is made at 1 too
  (`_render_pipeline_forced_taken`). An ICB's nonzero entry is reserved when the
  lists are first mapped, so on a fresh service an ICB made before it with a
  zero entry gets {2, 2} where the first ICB would get {1, 1}
  (`_icb_reservation`).
- **What Metal does with indices and ICB state on this host**
  (**measured**, `mutgap4_res_*`): it puts a v2 texture and a compute
  pipeline at the requested resource index whether or not the entry is
  forced, even at an index the service's rebuild reserved or a live
  texture holds (both textures then report {1, 1} and a copy works);
  it gives a freed index to the next resource, lowest first; an ICB's
  threadgroup-memory bind count is ignored (the ICB still runs with it
  0); and draws take the encoder's winding whether or not the ICB
  inherits it. A sampler with maxAnisotropy 16 keeps mip level 0 where
  one with 1 picks level 1 (`mutgap4_res_sampler_descriptor_fields`).

### 6.20 References across objects

Records reference other objects by object-list index: a heap buffer its heap, a
texture view its base, a buffer view its buffer, a heap texture its heap, a ref
texture its backing (through task 0's list: `backing_for_id`), an ICB command
its pipeline or buffer. Lookups recurse through `ResourceManager::get_object`.
In the service, a ref texture with a backing reaches `backing_for_id`, which the
service does not support: asking ends the task (`gpu_task.rs`, the
`BackingForId` death), and every later command on that task faults (status 4)
(**read**). `ResourceManager::reserve_objects` on task rebuild reserves texture,
sampler and ICB indices from placement resource IDs; pipelines are not reserved.

### 6.21 Displays

**The host side.** A display is created with `rvg_display_create` (section
4.1). Its callbacks run on the display thread. An EFI present needs its
`new_frame` callback:
the present checks for it, and an EFI present to a display without one ends
the process. Create the display before enabling EFI (section 6.22).

**The shared state page** (0x4000 bytes, set by opcode 1; layout
`display/page.rs`):

| offset | field | written by |
|---|---|---|
| 0x000 | serialNum | host |
| 0x004 | name[14]: UTF-8 and a NUL; a name of 14 bytes or more does not fit and becomes 14 zero bytes (**measured**); the default "Apple Virtual" is 13 | host |
| 0x012 | port | host |
| 0x014, 0x016 | size in mm, rounded | host |
| 0x018, 0x01a | cursor glyph max w, h: 256 with a `cursor_glyph` callback, else 0 | host |
| 0x01c | 1024 at init | host |
| 0x020 | cursor callbacks present: 1 `cursor_show`, 2 `cursor_move` | host |
| 0x024, 0x028 | size in mm, floats | host |
| 0x02c | colour space, 8 floats | host |
| 0x04c, 0x050 | origin, scale factor | host |
| 0x051 | compositorSupportsLiveResize: a one-way latch | guest |
| 0x058, 0x060 | configEpoch, connectionType | host |
| 0x068 | protection options | host |
| 0x070-0x078 | min, max, max SDR nits | host |
| 0x100 | raisedInterrupts: host ORs 2 present-complete, 4 connect/mode list, 8 disconnect, 12 update | host |
| 0x104 | enabledInterrupts | guest |
| 0x200 | expectedAck | host |
| 0x208 | mode count, u16; **a guest value above 128 aborts the host** at the next state machine run | host, and guest |
| 0x210 | modes: 128 × 16 bytes {u16 w, u16 h, u32 refresh 16.16, bit 0 EDR, bit 1 10 bpc} | host |
| 0xa10 | colour matrices [2][3][4] floats | guest |
| 0xa70-0xa73 | readIndex, forceRead, writeIndex, dirty | guest, host |
| 0xa74 | displayUUID[39] | host |
| 0xe00 | cursor position u32 {x, y} | guest |
| 0xe04 | guest cursor show | guest |
| 0xf00-0xf28 | cursor glyph record; 0xf20 host cursor show | host |
| 0xf80-0xf94 | gamma record | host |
| 0xf98 | present count, u64 | host |
| 0xfa0 | presented surface ref; 0xfa4 = 0; 0xfa9 ackPending | host |
| 0xfb0 | presented mapping ID; 0xfb8 asleep; 0xfbc lastAck | host |
| 0xfc0 | state | guest-initialised, host |
| 0xfc8, 0xfd0 | ICC profile VA, length | host |

**The state machine** (`display/guest_display.rs`), run on setting the client, setting the
mode list, an ack, unpause and restore; a no-op while paused. State 0: with a
client and a mode list, update settings and modes, expectedAck++, ackPending =
1, state 1, interrupt 4; else interrupt 0. State 1: an ack arrives through
opcode 2 (it must equal expectedAck with ackPending set, else "unexpected ack" /
"invalid ack" faults), then state 2. State 2 or any other value: if something
changed, expectedAck++, state 1, interrupt 12; with no client or a guest mode
count of 0, count = 0, state 0, interrupt 8. Raising sets raisedInterrupts and,
if the guest enabled the bit, sets the port's bit at 0x1014 and raises the
interrupt.

**Measured**: opcode 1 fills the page (serial, name, sizes, 3 modes, identity
colour matrix, expectedAck 1, state 1) and raises the display interrupt then the
stamp's; the ack moves state to 2; cursor show calls `cursor_show`; a doorbell
calls `cursor_show` (the flags differed) and `cursor_move`;
`rvg_display_cursor_position` returned (123, 456) from +0xe00.

**Presents.** Opcode 6/7 present a type-4 backing from task 0: the backing is
blitted from guest pages into its IOSurface once (it stays host-valid, so
re-presenting does not re-copy); IOSurface formats 'BGRA' and 'RGBA' only;
protection must be a subset of +0x68, read from the page at each present, so a
value the guest writes there after the connect counts (**measured**,
`mutgap2_dev_present_guest_protection`); then `present_frame`,
`signal_current_frame` unless asleep, present count +1, presented surface, and
on completion raisedInterrupts |= 2. Opcode 8 presents an IOSurface mapper
surface by mapping ID ('BGRA', 'RGBA', 'RGhA'), needs `use_iosurface_mapper` and
an IOSurface host device, and propagates a dirty guest colour space. 'RGhA' is
gated on `supportsRGhAPresents`, version 22, and the gate does not refuse: below
22 the host logs once ("Guest presenting RGhA without correct binary version,
allowing") and presents anyway (**measured**, `display_mapper_formats_v21`). The
texture view presenting from a backing creates lacks PixelFormatView usage: fine
normally, an assert under `MTL_DEBUG_LAYER=1` (**measured**).

**Gamma** (7, 8): entryCount <= 1024 and 4 × count <= mappedLength, else a
throw; the checksum is the sum of w[i] + w[n+i] + w[2n+i]; the table is read as
12 × count bytes although only 4 × count was checked, the bytes past the mapped
length as zeros (section 7.5); count 1 divides by zero in float (the slope step
is 1/0) and reads `gammaTable[entryCount - 2]` with a u32 index, 0xffffffff, not
−1: 16 GiB past the table, and the host dies (SIGBUS) at the present
(**measured**, `display_present_gamma_one`, in every setup, since presents run
in the host); once a gamma table arrives the gamma shader is used for good. The
green and blue slopes at zero take the red value. Components outside [0, 1] are
not clamped: the shader extends the table by those slopes, and on an 'RGhA'
surface with components from -0.5 to 1.75 every pixel of the row encodes
differently (**measured**, host-keyed, `mutgap_disp_gamma_out_of_range`). The
display copies the table's whole mapped length, which the guest display's checks
do not bound: a mapped length past the largest local task (32 GiB) raises before
anything is allocated or copied, after the present's `mode_change` and before
its frame is taken; opcodes 7 and 8 catch it, log, and signal the stamp, and the
display keeps showing its previous frame (**measured**, 2^62,
`interplay_displaywalk_*`, for example `_26` and `_39`).

**Cursor glyph** (4): width and height 1-256; `(h − 1) × stride + 4w` must not
overflow or exceed mappedLength; task 0; the checksum is the u32 wrapping sum
of the source words, and 0 skips the check. A mappedLength past the largest
local task (32 GiB) faults before anything is allocated or copied
(`display_cursor_glyph_sizes`). The host's `cursor_glyph` gets the glyph packed,
width × 4 bytes a row (`RvgCursorGlyph`).

**Colour matrix** (10, property 0, 12 words): written into
colorMatrices[writeIndex]; indices are guest bytes, so any index up to 255
selects 48 bytes somewhere in the page (always inside it).

**Sleep** (9): while asleep, presents update the current frame but do not
call `new_frame` or count a guest present (`rvg_display_present_counts`);
waking signals the current frame.

**Restore** replays the glyph and show, forces the colour matrix, and
re-presents through the mapped-surface path **only**: a guest that presented
with opcode 7 gets "Invalid mappingID" (caught and logged), and the state
machine call after it in the same catch is skipped. It also sets the
colour-matrix force-read byte at +0xa71, so a round trip's shared page differs
from an unsuspended run's there and nowhere else (**measured**,
`suspend_roundtrip`, `suspend_roundtrip_control`).

**Measured** by the gap and interplay scenarios (`display_gap_*`,
`interplay_session_*`, `interplay_life_display_*`):
- **A client that takes no call** (a port with a page and no display, or a
  display without a `new_frame` callback, such as one with only
  `mode_change`): a present completes at
  once (raisedInterrupts |= 2, the port poked), is counted at +0xf98 and records
  its surface or mapping ID; the glyph is checked and recorded, show records
  +0xf20, sleep and wake only +0xfb8; no handler is called and the display
  counts no guest present (`display_gap_clients_without_calls`,
  `display_gap_mapped_without_calls`).
- **The present bit masked** at +0x104: the completion still ORs 2 into
  raisedInterrupts but pokes no port, so 0x1014 stays 0; enabling the bit
  later does not raise the latched 2 (`display_gap_present_irq_masked`).
- **`mode_change`** is called when the width, the height, the IOSurface format
  or the protection differs from the last present's, once per change
  (`display_gap_mode_change_triggers`).
- **The gamma pipeline cache** keeps one pipeline per target pixel format and
  gamma path (`PipelineKey`, `display/pipeline.rs`), so a format and its sRGB
  counterpart (80 and 81, 70 and 71) each get a pipeline of their own
  (`display_gap_gamma_sizes`).
- **A bogus state** at +0xfc0 (7, or 0xffffffff) with ackPending set and the
  right ack: the machine resends (12) from the page's own mode count
  (`display_gap_bogus_resend_without_list`).
- **Unplug** (the host releases a presenting display): the guest display ORs 8
  into raisedInterrupts, zeroes the mode count and returns to state 0. Presents
  on the port then complete at once, are counted at +0xf98 and call nothing. A
  display plugged in again connects (4) from state 0
  (`display_gap_unplug_while_presenting`, `interplay_session_multihead`,
  `interplay_life_display_held_present`).
- **reset with displays**: a presenting port disconnects (8, one
  interrupt, mode count 0); a port whose connect ack is pending stays in
  state 1 with the ack pending and raises nothing, and after a new bring-up
  its page is left as it was, while the other reconnects
  (`display_gap_reset_with_displays`). A present held behind a barrier is
  abandoned (`interplay_life_display_reset_held`). The reset clears the
  root page's display words (0x240 + port) and leaves its task words as
  they were: `display_gap_reset_with_displays` reads the display words 0,
  and `interplay_limits_displays_reset`, with all eight ports busy, reads
  the display words 0 and the task words unchanged (section 6.24).
- **What a restore does**, per port (`display_gap_suspend_restore_variants`,
  `display_gap_release_before_suspend`, `interplay_session_workstation`,
  `interplay_session_iosurface`): it replays the cursor glyph and show; it
  re-presents the last opcode-8 surface (`mode_change` and a `new_frame`) and
  never an opcode-7 frame; a port acked and never presented comes back with no
  change; one with an ack pending stays pending; one whose display had no modes
  (its page restores with a mode count of 0) connects from the new display's
  list (4); one acked whose guest then zeroed the mode count (state 2) takes the
  new list as an update (12, expectedAck 2, state 1), not a disconnect; an
  asleep port is only put to sleep again. A port whose display was unplugged
  before the suspend keeps its page, and a display plugged into it on the target
  connects from the restored page. A present held behind a barrier at the
  suspend is parsed again by the resumed root FIFO and presents when released
  (`interplay_life_display_suspend_held_present`).
- **A restore after the guest deleted task 0**: `DeviceCore::get_kernel_task`
  raises (`DeviceCore::get_task_id` never answers none, so the guest display's
  own check for no kernel task is dead); `GuestDisplay::finish_restore` catches
  it and skips the state machine call after it
  (`display_gap_restore_without_task0`).
- **A backing deleted (54) after its present**: the display keeps the
  frame; a present of the same reference makes the backing again from
  task 0's list and guest memory (`interplay_life_display_delete_presented`).
- **`rvg_device_will_resume` with no state** on a device with displays: code 2,
  and the displays stay parked (`display_gap_resume_nil_with_displays`).
- **Mutation gaps** (`mutgap_disp_*`): in state 2 a guest overwrite of a
  settings field (a colour-space primary, the origin) is put back and counted as
  a change: one update (12) each (`_settings_overwritten`; the display walk
  below overwrites every field the host owns). A display presented with opcode 7
  and then put to sleep restores asleep, not presenting
  (`_restore_asleep_after_present`). SetProperties stops at the first property
  it does not know, and signals its stamp, so a colour matrix after it is never
  written (`_unknown_then_matrix`). A glyph 257 wide is refused and 256 accepted
  (`_cursor_width_257`). An EFI mode exactly as wide as the host's largest 2D
  texture is accepted (`_efi_width_max`).
- **Frames in flight** (**measured**, `newops_held_*`, with the runner's
  `encode` `hold` and `encode_commit`, section 5.2). Each encode counts a frame
  in flight until its command buffer completes, and
  `DisplayCore::wait_for_pending_frames` waits up to 2 s for none to be left,
  then logs a fault and goes on. A held frame so makes a guest present wait 2 s,
  and its stamp and `new_frame` follow; so does a guest disconnect
  (`DisplayCore::set_guest_display` with none), a guest connect, which waits in
  `DisplayCore::set_guest_display` before attaching the guest display, and
  `reset`, whose teardown disconnects the port. Committing the frame while a
  present waits wakes it early. With two frames held, one commit is not enough:
  its completion leaves one in flight and wakes nobody, and the second commit
  wakes the present. A display that has no guest display yet still counts its
  encode as in flight, though it draws no frame. Disconnect and reset call
  nothing; a held frame committed after them draws the frame it was encoded
  with, and a new encode draws none (`newops_held_present_timeout`,
  `_present_released`, `_two_frames`, `_connect`, `_disconnect`, `_reset`).

**Measured** by the display pipeline scenarios (`interplay_scanout_*`: the GPU
renders into mapper surfaces and task 0's backings, presented from any FIFO,
with barriers giving the order of writes and presents):
- **A frame is a view of its surface, not a copy.** A GPU write to a
  presented surface shows in the next encode with no present and no
  `new_frame` (`interplay_scanout_render_present_render`,
  `_mapper_buffer_frames`), and so does the IOSurface host device's test
  (3) of it (`_iosfc_test_under_frame`). An encode held uncommitted while
  the GPU draws draws the surface as it is at the commit
  (`_held_encode_sees_render`). A backing presented with opcode 7 stays
  host-valid: a GPU fill through a ref texture shows without a present,
  and the guest's rewrite of its pages, presented again, does not
  (`_ref_texture_frames`, in-process).
- **Close presents merge.** `new_frame` goes through a coalescing signal, so
  `interplay_scanout_frame_chain` encodes each of six frames before it releases
  the next render.
- **Presents beside faults.** A present written on a child behind its faulted
  command runs (`new_frame`, the page counts it) and its stamp is held until
  0x1028; a neighbour child presents meanwhile
  (`_present_from_faulted_neighbour`). Opcode 6 faults where opcode 7 signals: a
  reference that is not a backing faults the child (`_opcode6_children`).
- **While paused** the guest displays run no state machine: a display plugged in
  and a new mode list leave the pages as they were, and `unpause` runs each
  port's machine in port order, 12 on the port whose list changed, then 4 on the
  new one (`_hotplug_while_paused`); a display released while paused disconnects
  (8) at the unpause, and the queued presents then complete with nothing called
  (`_release_while_paused`). EFI's presents go to the display, not through a
  guest display, so a trace hit while paused still presents
  (`_efi_trace_while_paused`, section 6.22).
- **A present with no client or no page.** A present held behind a render
  while the host unplugs the display completes at once when released,
  calls nothing, and records its mapping (`_unplug_under_held_present`); a
  present after the guest disconnected the port's page (opcode 1, page 0)
  is refused, its stamp signalled, nothing called
  (`_disconnect_under_render`). A connect on a fresh page starts at
  expectedAck 1.
- **One surface on two heads**: with the task's mapper texture deleted,
  an unmap is Busy while either display shows the surface, and succeeds
  once both moved on; a present of the unmapped ID is "Invalid mappingID",
  and a map again over the same page presents what the GPU drew
  (`_two_heads_one_surface`; four heads fed by four FIFOs,
  `_four_heads`).
- **Mode updates while presenting**: frames reach the host while an ack is
  pending; a second change while the ack is pending only updates the list
  ("bogus ack"), and the ack raises the update it held (12)
  (`_update_pending_presents`). Each size or format change, 'RGhA' included,
  calls `mode_change` once (`_mode_switch_surfaces`, `_rgha_rendered`). A guest
  ICC profile (12) makes the next mapped present call `mode_change` though
  nothing else changed, once; an opcode-7 present between does not carry it;
  compositor parameters (11) reach nothing a host sees (`_icc_between_frames`).
  Gamma tables and colour matrices mix over opcodes 7 and 8 on one head
  (`_gamma_matrix_frames`); a present while asleep updates the frame without
  `new_frame`, and waking signals it (`_sleep_while_rendering`).
- **Restores**: each guest display re-presents its last opcode-8 mapping, with
  the page's gamma table, even when a later opcode-7 present replaced it
  (`_suspend_two_heads`); an asleep head is only put to sleep again, and waking
  it on the new device has no frame to signal (`_suspend_asleep_hidden`); a
  present held behind a render at the suspend is parsed again and follows the
  render (`_suspend_held_frame`); a mapping the IOSurface host device dropped
  before the suspend is "Invalid mappingID", caught by
  `GuestDisplay::finish_restore`, which skips its state machine call, so nothing
  is presented (`_suspend_after_teardown`). A mode list that differs from the
  one the guest acked is delivered by the restore only when the port never
  presented or last presented an opcode-8 mapping: the target display's list is
  written to the page and raises an update (12) that needs an ack, and a list
  set while that ack is pending waits for it. When the port presented only with
  opcode 7, the restore re-presents mapping 0, which throws and skips the state
  machine too, so the page keeps the stale list and nothing is pending
  (**measured**, `interplay_lifewalk_resume_page_modes`, `_after_present`).
- **reset and stop mid-composition**: `reset` tears down the guest displays (8
  on every page, the root words cleared) and the task with its mapper textures,
  so a surface nothing else holds unmaps; the guest redoes its set-up on fresh
  pages and presents again, and its GPU work gives the same bytes
  (`_reset_redo`, `_efi_reset_rebind`, where re-enabling EFI binds the display
  the reset parked). `stop` disconnects the guest displays and drops the task,
  and the guest brings up a fresh root page without negotiating again
  (`_stop_redo`).

**Measured** by the display walk (`interplay_displaywalk_*`, 44 seeds;
`pgtest/randdisplay.py` holds the model, and every seed agrees with it):
- **The guest display puts back what the host owns.** A guest that overwrites a
  field of the shared page the host owns has it put back, and counted as a
  change, at the next state machine run that updates the settings, or the mode
  list for a mode: the serial, the sizes, the glyph limits, the handler bits,
  the colour space, the origin, the scale factor, the nits, the configEpoch, the
  connectionType, the name and bytes after its NUL, the display UUID, a mode's
  width and its EDR and 10 bpc bits. So the comparisons against a descriptor's
  or a mode's private properties, which no host sets, are reached all the same:
  their values are the defaults, and the guest's bytes are what differ.
- **The display's shaders** (the library's own, section 3.2; each function
  says what it computes in `shaders.metal`, and
  `interplay_scanout_gamma_matrix_frames` agrees). Until a gamma table arrives
  the display copies the frame, alpha included, and ignores the colour matrix: a
  surface's alpha reaches the encoded frame unchanged, with the colour neither
  premultiplied nor blended, for opcodes 7 and 8 and EFI alike (an RGhA surface
  converted to unorm8). Once one has arrived each channel is looked up in its
  table, and then row c of the colour matrix gives output c = m[c][0] r +
  m[c][1] g + m[c][2] b + m[c][3], and every pixel's alpha is 0xff, whatever the
  surface holds (measured, `display_alpha_*`). The gamma path stays on for the
  rest of the display's life, through presents that carry no table and through a
  reset: after a reboot EFI's frames are drawn through the table and matrix the
  OS last presented with (`display_alpha_efi_after_gamma`, `interplay_boot_2`).
  The lookup is not interpolated between entries: a two-entry ramp is a
  threshold at 0.5. The result is stored as unorm8, rounded to nearest (0.5
  gives 0x80). The model predicts pixels only where the sampler's rounding
  cannot matter (tables constant per channel, 256- and 1024-entry identity ramps
  over 8-bit surfaces, the two-entry ramp away from 0.5), and the other frames
  are encoded and locked, not predicted.
- **A display with no queue** dispatches nothing when the guest disconnects
  it: `DisplayCore::set_guest_display` with none signals the responsiveness,
  0 with no guest display, and queues no job
  (`display::tests::signal_responsiveness::queueless_dispatches_nothing`).
  Through the C API every display has the display thread, so the case is out
  of a host's reach.

### 6.22 EFI framebuffer

Registers are section 6.2; the EFI display is `display/efi.rs`.
`DeviceCore::set_efi_display` on a nonzero write creates the EFI display and
binds **the display then parked at port 0**, only at creation; a zero write
tears it down and zeroes the six registers. The mode needs width and height at
most the device's 2D texture limit (16384 **measured**). The mapping (on start
or length writes) needs both nonzero, length a 16 KiB multiple, at most 64 MiB;
on every register write the frame is rebuilt if the task is ready, mode != 0,
depth == 32, stride >= 4 × width and a multiple of 0x1224's alignment, and
height × stride <= length — otherwise it silently does nothing. The texture is a
zero-copy linear BGRA8 view of guest memory. Presents come from
`add_trace_range`'s handler (one at once, then one per guest write) or, without
it, a 100 ms timer. `set_binary_version` and `reset` tear EFI down when the
version lacks `efiDisplay`.

**A refused mapping leaves a stale task.** The EFI display's task rebuild
(`EfiDisplay::rebuild_task`, `display/efi.rs`) destroys the task, then returns
early on the length and start checks with the task still marked ready, so the
next rebuild maps the new range with the stale task. **Measured**
(`efi_stale_task_*`), after a valid mapping: an unaligned length kills the host
in every setup (with a memory map, the memory map's assert); start 0, length 0
and a length over 64 MiB kill the host in the blocks setup, where the host's
`map_memory` is called with a NULL task, and complete in the others; a range
outside RAM completes everywhere.

**Measured** sequence: create the display (with `new_frame` and `mode_change`);
write 0x1200 = 1; 0x121c = 64 | 32 << 16; 0x1218 = 32; 0x1228 = 256; 0x1210 =
the frame's page; 0x1214 = 0x4000. Within 3 s: `mode_change(64, 32, 'BGRA')` and
one new frame. Encoding into a BGRA8 frame target succeeded and matched guest
memory bit for bit; a later guest write showed in the next encode. Bit for bit
only while the display has never had a gamma table: after the OS presents with
one, EFI frames go through the gamma table and colour matrix with alpha 0xff,
even after a reset (section 6.21, the display's shaders). 0x1204 read 0 (no mode
list set), 0x1224 read 16.

**Measured** by the gap and interplay scenarios (`efi_gap_*`,
`interplay_session_*`):
- **EFI and the guest's driver share the display.** Connecting port 0's page
  while EFI is live gives the guest display the display EFI bound, and presents
  nothing. EFI's rebuilds on its own register writes still reach the display:
  each calls `new_frame` and counts as a guest present, and raises no interrupt
  (`interplay_session_efi_live_handover`). The guest display's attach drops the
  current frame; after the guest's own presents, a trace hit still presents
  EFI's frame through its weak display reference (`mode_change` back to EFI's
  size); disabling EFI removes its trace range, and the guest's next present is
  its own again (`efi_gap_handoff_to_guest`).
- **The timer**: its first tick calls `mode_change` and `new_frame`, then only
  `new_frame`; depth 0 stops it, and the frame stays a zero-copy view (a guest
  write shows in the encode); a new mode and depth 32 restart it; `reset` tears
  EFI down and nothing is presented after (`efi_gap_timer_collapsed`). Detaching
  port 0's display unbinds EFI, and the next tick finds no display and stops; a
  display plugged in again is not bound, and a rebuild restarts the timer only
  for its first tick to stop it (`efi_gap_timer_unplug`).
- The texture is always the buffer-backed view: the tiled path is dead, since
  the device never responds to the tiled selector.
- **A mode over the limit** becomes 0 × 0. A mode exactly as tall as the limit
  presents (`mutgap2_dev_efi_height_max`, host-keyed). With the stride register
  0, a too-tall mode's stride is 4 × 0 = 0, which passes the stride checks: the
  framebuffer is mapped again and a trace range of length 0 is added, and
  nothing is presented until the mode is restored
  (`mutgap2_dev_efi_too_tall_stride_0`). A length of 0x4001 is refused like any
  unaligned length (`mutgap2_dev_efi_length_page_plus_1`).

### 6.23 The IOSurface host device

`iosurface/`. `rvg_iosurface_device_create` needs a memory map. Descriptors per
page = vm_page_size / 0x200 (32); page-table pages = vm_page_size / 4 (4096).
The last instance created in the process is global (section 3.2).

64-bit registers:

| offset | write | read |
|---|---|---|
| 0x1000 | ring base (physical address); unmaps the current ring | ring base |
| 0x1008 | ring size in entries; maps size × 16 bytes if size and base are set; overflow or map failure log and keep the old ring. The mapping must be whole 16 KiB pages: a size that is not a multiple of 1024 entries asserts in the memory map and kills the host (**measured**, `iosfc_ring_size_unaligned`) | ring size |
| 0x1010 | descriptor table base; tears down every surface; maps one page | descriptor base |
| 0x1018 | write counter; wakes the ring | write counter |
| 0x1020 | read counter, only while no ring is mapped | read counter |
| 0x1028 | | the features offered: 1 (v2 descriptors) where one page of table holds every mapping id in v2 descriptors, on 16 KiB pages; 0 on x86-64's 4 KiB pages, where it holds 4096 of the 8192 (v1 holds exactly 8192). The guest's mapper takes v2 whenever offered, and refuses to start when its table holds fewer than 8192 mappings |
| 0x1030 | descriptor enable: bit 0 selects 0x400-byte v2 descriptors, else 0x200-byte v1; clears the descriptor base | enable |
| other | ignored | UINT64_MAX (0xffffffff through QEMU) |

Ring entry, 16 bytes: `{u32 command, u32 mappingID, i32 status (written back),
u32}`. For each entry from read to write: 1 map, 2 unmap, 3 test, other →
kIOReturnUnsupported (0xe00002c7); status written; one interrupt per command.
Descriptor-table entry: `u32 {valid:1, isTable:1, pageNumber:30}`. Map: id 0
or >= 8192 → BadArgument (0xe00002c2); already mapped → Success; table
problems → NoMemory (0xe00002bd) or BadArgument; v1 descriptors are upgraded
(hostOnly 0, protection 0); host-only total over 1 GiB → NoMemory; planes >= 8
or components > 4 → NoMemory; success creates an IOSurface over the guest
pages and sets the descriptor's `mapped` bit. Unmap: still retained elsewhere
(e.g. by a display) → kIOReturnBusy (0xe00002d5). Test writes `i × 8` into
each 8-byte word of the surface — into guest pages for surfaces that are not
host-only.

**Measured** by the limits, mutation-gap and display pipeline scenarios:
- A task's mapper texture (type 11) holds its surface: an unmap of it is Busy,
  under isolation too (unless a retain was refused, below), while the surface
  stays mapped (a test still fills it), and a map of it again returns Success
  (`interplay_limits_iosurface_ring`, which also fills the 1024-entry ring in
  one doorbell and maps ID 8191). A mapper-ref buffer listed twice retains it
  once, and once the buffer is deleted the unmap succeeds
  (`mutgap_tex_mapper_buffer_twice`).
- **Under isolation, a refused retain stops the host holding.** A mapper buffer
  first listed while its ID is unmapped faults its exec, and the host's side
  keeps a nil entry for that ID, since it keeps the first answer for an ID
  (`tasks/remote/device_link.rs`). A buffer made for that ID later is then held
  only by the service: an unmap that is Busy in-process succeeds, a test after
  it is BadArgument, and the old buffer still writes the guest page. A 37 on the
  buffer erases the entry, and holding works again (**measured**,
  `interplay_iosfcwalk_refused_retain`; a divergence, section 7.2).
- **Measured** by the IOSurface walk (`interplay_iosfcwalk_*`):
  - opcode 1 with page 0 releases the head's surface: its unmap then
    succeeds; a reset drops every head's frame (an encode is black) and
    its surface unmaps;
  - a mapper buffer made before a teardown keeps writing the same guest
    page, in both setups, while the new surface for that ID is not held;
  - a resume re-presents the last accepted opcode 8, and only if its ID
    still has a surface; a refused opcode 8 leaves the page's mapping as
    it was;
  - the first opcode 8 on a newly connected page adds a `mode_change`: the guest
    display's guest colour space starts dirty at connect and only opcode 8
    hands it to the display, so a port that showed backings first calls
    `mode_change` again at its first surface; a guest display a resume restores
    starts clean;
  - 0x1020 takes effect only while no ring is mapped; mapping the ring
    (0x1008) does not wake it; a doorbell with the table unset (after
    0x1030) runs nothing; a zero entry in a new ring is Unsupported, with
    its status written back.
- **What IOSurface checks in a descriptor** (**measured**,
  `mutgap4_sys_iosfc_*`). It refuses (BadArgument, the test page
  untouched) a plane smaller than its rows (256 bytes where 64 × 8 needs
  512), extra left columns that do not fit (8 in a 512-byte plane of
  bytesPerRow 64; with 96 and 768 bytes it maps), and a v1 width that
  does not fit bytesPerRow. It ignores the address format (1, 2 and 3
  map and test like 0), a planeless surface's extended right and bottom
  pixels, and a fourth plane component's depth, name and range. An
  offset surface maps when offset + rows fits the allocation (256 or 512
  in 1024), and the test pattern then starts at the page's byte 0;
  `iosfc_surface_options`' offset case is refused because 256 with an
  allocation of 512 does not fit. In a v2 descriptor a plane's
  extendedPixelsOnLeft is at +0x40 and the base addressFormat at +0x3f.
- Seven planes, the most a descriptor holds, pass the plane-count check; the
  status is then IOSurface's. For seven 16×8 planes, v1 and v2, it is Success
  only when every plane has 1×1 elements and a base equal to its offset; planes
  with element size 0, or 1×1 elements based elsewhere, get BadArgument
  (`mutgap_disp_iosfc_seven_planes`). The plane maps of `iosfc_surface_options`
  (one plane, two planes) get BadArgument too. A v1 descriptor's seven planes
  are each upgraded, and component bit offsets are only metadata: 1, 3, 5 and
  200 map (`mutgap_disp_iosfc_v1_planes_upgraded`).
- Test (3) of an ID out of range, 0xffffffff or 8192, is BadArgument with one
  interrupt (`mutgap_disp_iosfc_test_id_max`,
  `mutgap_disp_iosfc_test_id_8192_page1`). ID 0 is BadArgument for map, test and
  unmap even over a valid descriptor
  (`mutgap2_dev_iosfc_id_0_valid_descriptor`).
- A guest-backed surface described read-only gets client direction 2, and Metal
  honours it: a GPU fill through a mapper-ref texture over it is silently
  dropped, with no fault, and the guest's page keeps its bytes; a host-only
  surface is created writable, and the fill lands
  (`mutgap2_dev_iosfc_host_only_texture`).
- A v2 surface of exactly 4096 × 4096 pages passes the page-count check, and one
  of 4096 × 4095 + 1 pages needs exactly the 4096 page-table pages allowed: both
  go on to walk the root table and are BadArgument there when its entry is
  invalid; one page more is NoMemory (`mutgap2_dev_iosfc_4096_tables`).
- A descriptor base write drops every descriptor page's cached mapping, not only
  page 0's (`HostDeviceCore::set_descriptor_base`, `iosurface/regs.rs`): a v2 ID
  on descriptor page 1, mapped from one table and unmapped, is read from the new
  table's page 1 at its next map (`mutgap3_res_iosfc_base_drops_every_page`).
- A v1 plane's extendedPixelsOnBottom is carried into the v2 form and reaches
  IOSurface, which decides: 8 extra rows in a 16×8 BGRA plane of 512 bytes are
  BadArgument for the map and the test, with the page untouched; the same plane
  of 1024 bytes maps, and the test writes 16 rows
  (`mutgap3_res_iosfc_v1_plane_extended_bottom`). A descriptor's surface-level
  compressionType reaches IOSurface as `kIOSurfacePlaneCompressionType` and
  changes nothing: types 1 and 2, v1 and v2, map, and the test writes what it
  writes for type 0 (`mutgap3_res_iosfc_compression_type_v1`, `_v2`).
- **Mapper surfaces under GPU work** (`interplay_scanout_iosfc_*`,
  `interplay_scanout_service_death_beside_frames`,
  `interplay_scanout_suspend_after_teardown`). A task's mapper texture is made
  by the first exec that lists it, not by 51: before one runs, an unmap of its
  surface succeeds (`_iosfc_ring_under_held_present`, whose one doorbell maps,
  tests and unmaps with an interrupt and status each, in order, while a present
  of the surface waits on a barrier). A descriptor base write drops every
  surface, but a display and a mapper texture keep their IOSurface over the same
  guest page: the GPU still draws through the old texture, a present of the
  dropped ID is "Invalid mappingID" until the guest maps it again, and the new
  IOSurface aliases the old one's page, so later renders through the old texture
  show in it (`_iosfc_teardown_under_holders`). After a resume the task's mapper
  textures are made again, and one over a mapping the device no longer has
  faults the exec ("Attempting to create mapper ref against invalid surface")
  until the guest maps it again (`_suspend_after_teardown`). A texture held by a
  task whose service died is no host reference: once no display shows the
  surface it unmaps (`_service_death_beside_frames`, isolate).

**No saved state.** The IOSurface host device has no save or restore, and
nothing in the C API carries its registers or its mappings: a migration's
target starts with a new one, and the guest starts its ring again and maps its
surfaces again (section 6.24, `fresh_resume_iosfc_live`).

### 6.24 Suspend and resume

**The blob** (`guest/abi/suspend_blob.rs`) is this library's own format, every
word a little-endian u32: u32 magic 'RVGS' 0x53475652, u32 version 1, then
sections to the end of the blob, each u32 tag, u32 payload length, payload;
each section exactly once, in either order (the device writes tag 1 first):
- tag 1, the registers, 20 u32s (80 bytes), reported under the device-state
  keys that name them: 1 root page, 2 binary version, 3 FIFO base page, 4 FIFO
  enabled, 5 length, 6 written, 7 read, 8 start, 9 display interrupts, 10
  stamp interrupts, 11 fault mask != 0, 12-17 EFI
  enabled/start/length/depth/mode/stride, 18 display port count, 19 root fault
  offset, 21 connection types;
- tag 2, the device info: an i32 max key (the last max key the guest's device
  info query used; default 1), then the pairs `write_device_info` wrote for
  it, up to their terminator and without it: at most 45, each key one a
  resume reads (1-42 or 44).

Size = 108 + 8 × pairs. Every byte is written, whatever max key the guest
stored (a negative one included). The normalizer reports the registers and
the device-info pairs (the runner's `blob_pgsd` record, `pgtest/normalize.py`),
and the whole blob as hex where the record would not give its bytes back.

**`rvg_device_will_resume`** (`device/lifecycle.rs`) → `RvgResumeError`
(`RVG_RESUME_ERROR_*`): no state (NULL), a state shorter than the header, a
magic other than 'RVGS' (Apple's PGSD blob included), a section that runs past
the end, is unknown, repeated or missing, a payload of the wrong length, more
than 45 device-info pairs, or a device-info key a resume does not read →
InvalidContent (2), with one log line saying which; a version other than 1 →
InvalidSuspendStateVersion (1). Nothing in a blob ends the process. Binary
version >= 0xf6, or one `set_binary_version` stores with bit 31 set →
InvalidGuestVersion (3); `check_capabilities` fails → IncompatibleDevice (4);
display port count differs → InvalidDisplayPortCount (5); connection types
differ at version >= 238 → InternalFault (0). It sets run state 3 first and
leaves it on failure, after which any mapping dies on the run state
(`DeviceDeath::MappingWhileLeavingSuspend`).

**`rvg_device_did_resume`**: replays the EFI registers, the root page, the root
FIFO (read, written, fault), child FIFOs from the root node, tasks from root
words 0-255 (task length from root +0x14), displays from root words 0x240 +
port, then restores the guest displays, starts the FIFOs and raises one
interrupt. **Resuming in place** (all four calls on the same device) loses the
root FIFO: `rvg_device_did_resume`'s root-page write tears down the FIFO still
marked enabled and clears the mark, so it is not recreated; 0x1000 reads 0 and a
later doorbell is never answered (**measured**, `suspend_in_place`).
**`rvg_device_will_suspend`**: run state 1 (interrupts suppressed), FIFOs
quiesced. **`rvg_device_finish_suspend`** asserts run state 1, drains, writes
back and tears down tasks, serializes, drops the root FIFO, pauses the guest
displays. **`pause`**: quiesce and drain, pause guest displays (a second pause
aborts, with a guest display), run state 5. **`unpause`**: run state 0, guest
displays unpaused (asserts they were paused), FIFOs restarted, one interrupt
always. Restarting a child maps its page list again, so the child after a
deleted one (W3) stops there, as at a resume (section 6.7; `interplay_rand_*`).
**`stop`**: root page 0. **`reset`**: root 0, binary version 0 (EFI torn down),
EFI off, device-info max key 1 (`mutgap_dev_reset_info_max_key`); the teardown
clears the root page's display words and leaves its task words
(`interplay_limits_displays_reset`, section 6.21). `rvg_device_will_resume`
takes the device-info max key from the blob's device-info section, and the next
blob carries it (`mutgap_dev_resume_info_max_key`).

**What a resume rebuilds** (**measured**, `suspend_gap_*`,
`interplay_life_*`, `interplay_session_*`):
- **Child FIFOs**: exactly those whose node's running word (+0xe) is nonzero,
  for ids below maxFIFOCount, with the node's counts and fault offset. Any
  nonzero u16 counts: a node whose word is 0x100 is rebuilt and runs, and the
  word then reads 1 (`mutgap2_dev_resume_running_high_byte`); children above a
  version the guest lowered are lost (section 6.3). Tasks are rebuilt for every
  root word that is set, task 255 included (`mutgap2_dev_resume_task_255`). A
  node with running 0 is not a FIFO, even with a valid ring; so a child deleted
  before the suspend does not come back. A node whose stampIndex is past the
  FIFOs is not created ("Failed to set up FIFO during resume"); one whose page
  list cannot be mapped is created and does not start ("Failed to start child
  fifo"), and waking it does nothing. That includes the child after a deleted
  one: the delete zeroed its pageListPage (W3). Nothing runs on it (a fault
  offset in its node still sets its fault bit), and a plain 48 on it faults,
  since it exists: the guest must write its page list again, then 49 and 48
  (`suspend_gap_resume_child_nodes`, `interplay_life_resume_rich`). Child nodes
  are read even when the blob's root FIFO is disabled: a restored child that
  signals a stamp then writes through a null stamp pointer, and the host dies
  (section 7.4).
- **A restored fault is only a register.** A root FIFO or child restored with a
  fault offset sets 0x103c (or node +8) and its fault-mask bit, but its stamp
  queue is new: the commands between its read and written counts run and publish
  while 0x102c reads 1, and 0x1024 then finds no faulted head. A blob whose
  fault offset (key 19) is 0, as a crafted blob that leaves it unset has it,
  restores the root FIFO faulted at offset 0; only 0xffffffff means no fault
  (`suspend_gap_restored_fault_*`). A suspend over a fault carries it (keys 11
  and 19), and the resumed FIFO's later faults are reported as usual, unlike
  after a pause (`interplay_life_suspend_over_fault`, and section 7.1).
- **The root FIFO** resumes from the blob's counts. A command rung between
  `rvg_device_will_suspend` and `rvg_device_finish_suspend` does not run there
  (the FIFOs are quiesced), but it is counted in written, so it runs on the
  resumed device without a new doorbell (`interplay_life_will_suspend_rung`). A
  command blocked on a barrier at the suspend is abandoned before its read count
  (key 7), parsed again on the resumed device, and waits again
  (`interplay_life_suspend_blocked`).
- **Every key** (`suspend_gap_every_key`): the pending display and stamp
  interrupt bits (keys 9, 10) read back once; key 11 is ignored (0x102c
  reads 0); the EFI registers take the blob's values though EFI stays off;
  the connection types (key 21) are not restored. With the root FIFO
  disabled, 0x1000 = 1 alone then creates it from the blob's base, length
  and start, with fresh counts.
- **The capability check depends on the setup**, because the host's own
  device info does: key 33 bit 0x20 and key 42 pass only in the isolate
  setup, and key 33 bits 1 and 2 and key 17 fail in the blocks setup, which
  has no memory map (`suspend_gap_capabilities_setups`,
  `suspend_gap_capabilities_versions`).
- **Private buffers**: in-process, `rvg_device_finish_suspend` writes back only
  a private buffer the guest listed with invalidateGuest; one listed without it
  is made again on the resumed device from its guest pages, with what they held
  before. The service keeps private buffers in guest pages, so isolate loses
  nothing (`interplay_life_suspend_between_parts`, `_dirty`).
- **Displays**: section 6.21.

**Suspend and resume across processes** (**measured**, `fresh_resume_*`,
run in phases: tests/suite/README.md, "Phases"). A migration restores the
VM in a new process, often on another host: the next phase gets the blob
and guest RAM by guest physical address, and a new device.
- **The rebuild is the one in process.** The new process rebuilds the root
  FIFO at the blob's counts, the child FIFOs, the tasks and the displays
  exactly as a resume in one process does (`fresh_resume_roundtrip`). The
  target's observations depend only on the target's setup, whichever setup
  the source ran. One image resumed by two processes in turn gives the same
  observations twice (`fresh_resume_twice`).
- **Setups change device info, so a migration can be refused.** Key 33 is
  0xbdc in blocks, 0xbdf in map and 0xbff in isolate, and key 42 is 3 in
  isolate only (M3 Max). A target lacking a bit the source reported refuses
  with IncompatibleDevice: map to blocks, isolate to blocks, isolate to
  map (`fresh_resume_caps`, keyed by host).
- **The target's RAM.** Larger RAM changes nothing, and so does the same RAM
  split into two ranges, with a task's pages straddling the seam. Smaller RAM
  that lacks a task's pages makes that task's exec fault the root FIFO, in every
  setup (`fresh_resume_ram_*`). RAM elsewhere, or RAM missing only the root
  page, leaves no root: 0x100c reads 0, no task or display comes back, and the
  cursor reads 0xffff; in the blocks setup task 0's `map_memory` returns false
  and the task is destroyed. `rvg_device_did_resume` still raises its interrupt
  (`fresh_resume_ram_moved`, `_ram_root_lost`). The blob over zeroed RAM
  recreates only the root FIFO (`fresh_resume_blob_only`).
- **Displays** attached between `rvg_device_will_resume` and
  `rvg_device_did_resume` restore as those attached before. A
  target with 2 ports refuses the blob's 1 with InvalidDisplayPortCount (5)
  (`fresh_resume_displays_*`).
- **Work at the suspend.** Commands written and not rung, and a command blocked
  on a barrier, are parsed again and run by the target (`fresh_resume_queued`,
  `_blocked`, `_mid_frame`). A NOP rung while paused does not run, and the
  target runs it at once: the pause is not in the blob (`fresh_resume_paused`).
  A pause with a display page, then `rvg_device_finish_suspend`, is a host
  death, the guest display's second pause (`fresh_resume_paused_display`). A
  faulted root FIFO resumes faulted (`fresh_resume_faulted`). A task whose
  service died is rebuilt with a new service (`fresh_resume_dead_task`).
- **The IOSurface host device is new** in the target, since nothing in the
  C API or QEMU carries its state: a display last presented
  with opcode 8 is not presented again, and the guest must start its ring
  again and map the surface (`fresh_resume_iosfc_live`).
- **Version 0xf5.** A `rvg_device_will_resume` refused at the capability check
  has already set the binary version: 0x1034 reads 0xf5 on a device the guest
  never versioned, and a later blob carries key 2 = 0xf5
  (`fresh_resume_version_f5`, `_control`).
- **Blob mutation** (`fresh_resume_mutate_*`, a device-written blob of
  device-info max key 45 and 41 pairs, 436 bytes; its header, registers,
  device-info header and first 13 pairs mutated; 584 offers per setup). In
  every setup: 301 accepted, InvalidContent 165, IncompatibleDevice 89,
  InvalidDisplayPortCount 9, InvalidSuspendStateVersion 9, InternalFault 9,
  InvalidGuestVersion 2. No truncation is accepted: every one cuts a section
  short. A blob whose binary version (key 2) is 1 is refused with
  InvalidGuestVersion (3), and 0x1034 then reads 0x80000001
  (`fresh_resume_mutate_state_02`).

**Abandoned suspends** (**measured**): `rvg_device_will_suspend` then `unpause`
(no guest display) restarts the quiesced FIFOs, and the cycle is transparent
(`interplay_life_will_suspend_unpause`); with a guest display, `unpause` asserts
(section 7.3). It is not transparent over a FIFO blocked on a barrier:
`rvg_device_will_suspend` quiesces every FIFO but, unlike `pause` and
`rvg_device_finish_suspend`, wakes no wait (it calls no
`DeviceCore::signal_fifos`), so the blocked thread stays in
`DeviceCore::wait_stamp` with its wait registered, and `unpause`'s `Fifo::start`
clears the quiesced flag and starts a second thread on the FIFO. The new thread
finds nothing to read, because a child counts what it read in the FIFO object
and the old thread had read the blocked command. When the barrier's stamp
arrives the old thread wakes, sees the flag clear and runs the held work, in
order (`interplay_dag_abandon_blocked`, which rings neither FIFO again: which
thread a doorbell wakes is timing). A doorbell that claims more bytes first
makes the new thread, which reads from the node's readCount before the blocked
command, parse that command again and register a second wait under the same
index: a host death (section 7.4, `interplay_dag_abandon_blocked_dies`). With
the root FIFO blocked instead, the new root thread reads on at once past the
blocked command, into ring bytes never written, which it skips (section 6.6)
until it wraps and replays the ring from its start; the define of the blocked
child then asserts in the child's creation: measured, but racing with
`unpause`'s return, so not locked.

`rvg_device_finish_suspend` then `unpause` leaves 0x1000 reading 1 with the root
FIFO, children and tasks gone: doorbells are never answered. The guest's 0x1000
= 0 then 1 makes a new FIFO at read 0, and the next doorbell replays the whole
ring from its start (`interplay_life_finish_suspend_unpause`).
`rvg_device_will_resume` with no state returns InvalidContent (2), and a valid
blob afterwards is accepted (`suspend_gap_nil_state`).

**Other orders of the host's calls** (**measured**, `fresh_dev_suspend_*`,
`fresh_dev_lifecycle_stop_reset_orders`; the deaths among them are in
section 7.4):
- `rvg_device_did_resume` on a device whose root is live or kept (twice, without
  `rvg_device_will_resume`, after `rvg_device_will_suspend` alone, on a running
  or paused device, after a refused `rvg_device_will_resume`) writes the root
  page, whose teardown finds the FIFOs still marked enabled, tears them down and
  clears the mark, as in place above: no FIFO is rebuilt, and 0x1000 reads 0.
  The display page is restored, and a paused guest display is unpaused. After a
  refused `rvg_device_will_resume` it sets run state 4 before the write, so no
  mapping asserts. `rvg_device_did_resume` on a device never given a root page
  builds nothing and raises its interrupt.
- Between a successful `rvg_device_will_resume` and `rvg_device_did_resume` the
  registers read the blob's values. A root-FIFO doorbell only stores the written
  offset; the rebuilt FIFO runs that command at the next doorbell, not at
  `rvg_device_did_resume`. A root-page write there dies (run state 3), and
  `reset` there zeroes the root page the blob set, so `rvg_device_did_resume`
  rebuilds nothing.
- `unpause` in place of `rvg_device_did_resume` rebuilds nothing, though 0x1000
  reads 1; the guest recovers with 0x1000 = 0 then 1.
- `rvg_device_will_suspend` twice is the same as once. `rvg_device_will_suspend`
  then `pause` and `unpause` abandons the suspend; `rvg_device_will_suspend`
  then `stop` or `reset`, then `rvg_device_finish_suspend`, writes a blob of the
  torn-down device.
- After `rvg_device_finish_suspend`, `stop` or `reset` tears down with the guest
  display still paused, and nothing asserts. The run state stays 2, so a root
  FIFO the guest restarts publishes its stamps and raises no interrupt.

### 6.25 Minimal sequences (measured)

Guest RAM at GPA 0x40000000 (16 or 64 MiB, `mach_vm_allocate`), registered
as one range of the device's `RvgMemoryMap`.

**(a) FIFO and a stamp.** 0x1034 ← 245; 0x101c ← root page; 0x1030 ← FIFO
page; 0x1004 ← 0x4000 (or 0x8000); 0x1010 ← 0x400; 0x1000 ← 1. Write `{30,
0, 12, 1}` at FIFO + 0x400; 0x1008 ← 12. Observe: the interrupt; u32 at FIFO
+0 = 1; 0x100c = 12; 0x1018 = 1 then 0.

**(b) A task.** Task root at page T: `{pt page, 1}`; page table at page P:
entry i = data page i | 1 << 31. Commands `{56, 0, 28, s}{2, len u64, T}`,
`{57, 0, 32, s}{1, 0 u64, len u64}`. Observe root word[1] = T, T + 0x14 = len.
Fault path: 0x102c = 1, 0x103c = the command's ring offset; 0x1024 ← 0
resumes. `{32, 0, 16, s}{1}` deletes: root word[1] = 0.

**(c)(d)(e) GPU into guest memory.** Task 1 with 32 writable pages; records at
VA page 1; object list at VA page 3 (11 entries, opcode 51 `{1, 3, 132}`):
buffers (type 1, `{0x4000, page, 0}`) at refs 1, 4, 8, 10; functions (type 6)
from `kernel.metallib`, `vert.metallib`, `frag.metallib` at refs 2, 5, 6; the
7-byte compute pipeline at ref 3; the 41-byte render pipeline (RGBA8Unorm) at
ref 7; a type-8 texture over buffer 8 (4×4 RGBA8, bytesPerRow 16) at ref 9. One
exec `{67}` `{1, 0, 3}` with three ranges: blit 306 on ref 1; compute
0xd0/0xcb/0xca on refs 3 and 4; render 0x1a/0x74/0x01 into ref 9. Results in
guest memory: 0xAB at 0x10-0x2f, gid × 2, every pixel ff8040ff. In the isolate
setup, list refs 1, 4 and 8 in the resource list.

**(f) A present read back**: section 6.22.

**(g) Queries**: opcode 58 (section 6.9), 59 and 64 (section 6.10).

## 7. Quirks: odd behaviour the suite pins

These are behaviours of Apple's framework that look odd, and that the
codebase keeps: guests and hosts were written against that framework, and
may rely on them. Each entry needs at least one scenario whose expected
file pins it. The section is a catalogue of where the behaviour is
surprising, so that a change keeps it knowingly. It is not a list of things
to fix: changing one is a change of behaviour, made on purpose and reviewed
(decisions 3 and 4).

### 7.1 Guest-visible quirks

| quirk | evidence | reached by |
|---|---|---|
| A header with length < 12 or past the ring is skipped without publishing the read count; the FIFO then parses stale ring bytes past `written` | **measured**, `fifo_frame_length_*_stale` | a bad header |
| `CmdDelay` (61) never signals its stamp; every later stamp on that FIFO stalls, with no fault | **measured**, also on Apple's framework, `fifo_op_delay` | opcode 61 |
| DeleteChildFIFO zeroes the *next* node's pageListPage, for id 63 too | **measured**, `fifo_child_*` | opcode 49 |
| Stamp interrupt bits alias for stamp indices 32-63 (`1U << stamp`); the index check is `>`, not `>=` | **measured** (section 4.4) | child FIFOs with high stamp indices |
| 0x102c is a boolean; 0x1014 and 0x1018 are read-to-clear, 0x102c is not | **measured** | faults |
| Version negotiation: clamping, stepping down, 1 and 236 unsupported, 0 accepted, negative ignored | **measured** (section 6.3) | 0x1034 |
| MMIO indices are `(uint16_t)((offset − 0x1000) / 4)`: aliasing and wrap | **measured** | any offset |
| 0x122c reads `count != 0` below version 245, logging each read; 0x1234 is masked `& 3` | **measured**, `device_version_*` | registers |
| Display commands 7, 8, 10, 12 signal their stamp even when they fail | **measured** for 10, `display_color_matrix`, `fifo_op_display_nub`; **read** for the others | bad display payloads |
| A continuation segment with no open decoder: "Encoder type doesn't match" for every type but render, "Previous stage didn't specify continuation" for render | **measured**, `exec_continuation` | segment flags |
| Nothing checks a record's length against its command's | **measured**, `exec_reset_rate_map_under_command` (the read that runs past the record raises instead) | short commands |
| A carried decoder is reused whatever the next segment's type; a second render or compute encoder over an open one is a death (section 7.4) | **measured**, `exec_continuation` | segment flags |
| The protection envelope's inner header length is not applied; a failed read skips the segment silently | **measured**, `exec_protection_envelope` | encoder type 5 |
| `ObjectLists::set_object_list` validates the *previous* offsets' alignment | **measured**, `tasks_objlist_placement` | opcodes 51, 68 |
| The in-process copy (`tasks/local/memory.rs`) rejects `end >= taskLength`; the service accepts `end == taskLength` | **measured**, `tasks_objlist_record_at_end`, `exec_range_at_task_end` (in-process a command buffer there kills the host) | ranges and records ending at the task's end |
| Render 0x6a decodes as SetDepthStoreAction; 0x8e bounds by maxVertexSamplers; 0x91 by maxViewportCount; 0x96 uses `>=` against the max colour index | **read**; 0x8e cannot show on the suite's host (section 6.16) | render commands |
| Render primitive types 5-9 pass validation (5, 6 draw; 7, 8 draw nothing; 9 faults on the GPU); index type is not validated | **measured**, `render_draw_odd_types`, `render_draw_type9` | draws |
| Render 0x0b zero-extends its 16-bit base vertex; 0x1d sign-extends | **measured**, `render_draw_indexed` | draws |
| Render 0x16 always throws "unsupported" | **measured**, `render_state_misc` | render |
| Compute 0xe5's error names the ICB ref where it means the buffer | **read** | compute |
| Info 460 fetches the reply cursor before checking the object type: in-process a bad type still aborts the host | **measured**, `info_460_bad_type_aborts_host` | info |
| An absent rate map on opcode 66: in-process faults, the service replies success; ref 0 faults in both | **measured**, `exec_reset_rate_map`, `exec_reset_rate_map_in_process`, `render_ratemap_reset_checks` | opcode 66 |
| `map_memory` runs are at most 2 pages, and a run flushed mid-walk takes the next page's writability | **measured** (blocks), `tasks_map_coalesce` | opcode 57, lazy mapping |
| The range tracker's `remove_from_range` stops after the first whole block, leaving later blocks' runs recorded, so an object list there is not mapped again and reads the host's stale pages | **measured** (blocks), `tasks_map_tracker_blocks`, `tasks_map_tracker_multiblock_unmap` | unmaps spanning blocks |
| "Range too complex" past 30,720 discontiguous runs (128 run blocks of 240); "Unmapped range too complex" past 16 holes in one lazy range | the first **measured** (blocks), `newops_range_tracker_limit`; the second **measured** (blocks), `tasks_map_unmapped_holes`, `tasks_map_tracker_multiblock_unmap` | mapping patterns |
| A map refused "Range too complex" has already been mapped by the host: the host's `map_memory` is called, then the command faults | **measured** (blocks), `newops_range_tracker_limit` | opcode 57 with every run block full |
| The service's unmap does no range check and always succeeds | **measured**, `tasks_map_ranges` | opcode 34 |
| In-process heaps must be private; a shared one fails silently | **read** | type 13 |
| A display name of 14 UTF-8 bytes or more becomes 14 zero bytes in the shared page | **measured**, `display_properties` | `RvgDisplayConfig.name` |
| The gamma shader stays in use once any gamma table arrived; green and blue slopes at zero take red's | **read** | opcodes 7, 8 |
| EFI binds the display parked at port 0 only when EFI is created | **read** | display and EFI order |
| Restore re-presents only through the mapped-surface path: the last opcode-8 mapping, even when a later opcode-7 present replaced it on that head | **measured**, `display_gap_suspend_restore_variants`, `interplay_scanout_suspend_two_heads` | suspend after opcode 7 presents |
| The device answers that protected content is enabled whatever its config says: core's `enable_protected_content` is ignored, and the C API has no field for it | **read**, `device.rs` (`tests::protected_content_ignored`) | device creation |
| Argument buffers (device info key 33 bit 0x20, key 42) require isolation, detach-backing support, `disable_argument_buffers` unset, and a device not named "Apple Paravirtual device"; with `process_isolation_follows_guest`, also an IOSurface device with a descriptor table when the guest asks | **measured** (key 33/42); **read** for the last, `device.rs` (`tests::argument_buffers_follow_the_iosurface_devices_descriptor_table`) | device info |
| The root FIFO never waits inside a command: it reads a partial command's missing bytes from the ring past 0x1008; child FIFOs wait | **measured**, `fifo_frame_partial_root`, `fifo_frame_partial_child` | a doorbell mid-command |
| An underflow or barrierCount > 64 consumes only the header: the rest of the command runs as commands; the "barrier bytes beyond length" check is dead | **measured**, `fifo_frame_underflow`, `fifo_frame_barrier_count`; **read** for the dead check | bad headers |
| A child define whose page list cannot be mapped signals, and the FIFO never runs; a define refused for its stampIndex has already written the node | **measured**, `fifo_child_page_list_*`, `fifo_child_define_errors` | opcode 48 |
| Stamps of a stampIndex above maxFIFOCount after the version is lowered are dropped silently; exactly the maximum publishes | **measured**, `fifo_child_version_lowered` | 0x1034 after opcode 48 |
| 0x1028 clears fault bit n (the id) while a fault sets bit stampIndex; `1ULL << 64` clears the root's bit without resuming it, and an id of 64 or more resumes nothing | **measured**, `fifo_child_stamp_index_mismatch`, `fifo_child_resume_64` | 0x1028 |
| Any version write gives maxFIFOCount 32, even 0, 1 and 236 | **measured**, `device_version_fifo_count` | 0x1034 |
| `pause` publishes what its drain completes, with interrupts, before it sets the paused state | **measured**, `device_lifecycle_paused_interrupts` | `pause` |
| GetComputeInfo (59) diverges by setup on pipeline ref 0, count 0 and a read-only reply page | **measured**, `compute_info_*` (section 6.10) | opcode 59 |
| A heap record is read as 16 bytes | **read**; an 11-byte record faults (`resources_heap_records_*`) | type 13 |
| A serializer texture in an exec's resource list always faults | **measured**, `textures_serializer_records` | resource lists |
| A legacy heap's synchronize leaves it guest-invalid, so a second 53 copies again | **measured**, `resources_heap_paging` | opcode 53 |
| A prepared ICB ignores guest writes until invalidated; 63 does nothing; write-back errors are dropped | **measured**, `icb_*` (section 6.19) | ICBs in the service |
| 'RGhA' presents below version 22 log and present anyway | **measured**, `display_mapper_formats_v21` | opcode 8 |
| Restore sets the colour-matrix force-read byte (+0xa71) | **measured**, `suspend_roundtrip` | resume |
| Resuming in place loses the root FIFO | **measured**, `suspend_in_place` | `rvg_device_will_resume`/`rvg_device_did_resume` on the same device |
| A refused EFI mapping leaves a stale task that the next rebuild maps | **measured**, `efi_stale_task_*` | EFI registers |
| A pause publishes a faulted head without clearing the stamp queue's fault latch: after unpause the resume finds no faulted head, and the next fault is never reported (no offset, 0x102c 0, the stamps behind it held) until a second resume | **measured**, `fifo_gap_fault_latch_after_pause`, `_child`, `interplay_life_pause_over_fault`, `interplay_life_pause_over_child_fault` | `pause` over a fault |
| A child whose stampIndex differs from its id breaks the circular-wait walk: a wait on its slot for a stamp it has pushed faults "Circular wait" | **measured**, `interplay_fifo_stamp_index_false_circular` | barriers on such a child |
| Barriers compare `(int32_t)(slot − value) >= 0` against the slot as it is now: a value 2^31 or more ahead passes at once, and a value the slot has passed can block again after the slot moves back | **measured**, `fifo_gap_barrier_signed_distance`, `interplay_fifo_serial_stamps` | stamp values far apart |
| 0x1000 = 0 over a faulted head publishes it and everything behind it, and clears the fault mask | **measured**, `fifo_gap_teardown_faulted_head` | disabling the root FIFO |
| A restored fault is only a register: the commands behind it run; a fault offset of 0 restores a fault at offset 0 | **measured**, `suspend_gap_restored_fault_*`, `suspend_gap_resume_child_nodes` (section 6.24) | resume |
| The range tracker keeps a run block its unmaps empty, and an empty block compares as after everything | **measured** (blocks), `tasks_gap_tracker_empty_block` | unmaps inside a later block |
| `RunBlock::remove`'s binary search can stop on a run after the one that holds the range, and the scan from there misses it: after an unmap of page 10 from runs [0,1), [4,5), [8,12), [16,17) the tracker still counts page 10 mapped, so a probe of it maps nothing | **measured** (blocks), `mutgap_mem_tracker_remove_search` | unmaps inside a run |
| A new run that ends exactly where a later block ends empties that block (InsideEdge), and the empty block then stops lookups: a page mapped since the start is mapped again at every probe | **measured** (blocks), `mutgap_mem_tracker_block_end_edge` | maps next to a full block |
| A run past a full last block opens a new block; an unmap of the whole task then clears only the first block, so pages there are mapped again and the later block's are not | **measured** (blocks), `mutgap_mem_tracker_full_last_block` | unmaps spanning blocks |
| The range tracker's page run rounds with 0x3ffe, not 0x3fff: a lazily mapped range that ends one byte into a page is one page short, so a reply's last byte there is not written to the guest; map and isolate round with 0x3fff and write it | **measured** (blocks), `mutgap3_sys_heap_reply_one_past_page` (section 6.10) | replies ending one byte into a page |
| SynchronizeResource (314) takes a nullable resource: on a texture whose placed creation failed, the exec completes with no fault | **measured**, `textures_gap_misplaced_sync` | blit 314 |
| A 32×16 RGBA16Float mapper texture over a 'BGRA' surface of the same row size passes `validate_iosurface_descriptor`, and Metal makes it | **measured**, `textures_gap_rgba16f_mapper` | type 11 |
| A ref texture's evict does nothing: after its backing is deleted from task 0, a fill through it with invalidateGuest never reaches the backing's pages, across a suspend | **measured** (in-process), `resources_gap_ref_texture_view` | type 5 after 54 |
| An info reply whose buffer is an object-list entry of type 0 or 10 faults before any cursor is asked for, so in-process it does not abort | **measured**, `decoder_gap_info_reply_type0` | info replies |
| A pixel-format view of a mapper or ref texture cannot be discarded: 60 on it throws and faults | **measured**, `resources_gap_mapper_views`, `resources_gap_ref_texture_view` | opcode 60 on a view |
| A prepared compute ICB runs with what it was prepared with: after its bound buffer is deleted (37) and made again, run with flags 0 it writes through the deleted buffer's GPU address (the writes land in the buffer's guest pages); after its pipeline is deleted (40) it still dispatches with it. Nothing faults | **measured**, `interplay_lifetime_icb_bound_buffer_deleted`, `interplay_lifetime_icb_pipeline_deleted` (section 6.19) | 37 or 40 of what a prepared ICB names |
| DeleteTask (32) does not flush, unlike `rvg_device_finish_suspend`: host writes the guest asked for with invalidateGuest are lost (in the service, a texture's) | **measured**, `interplay_lifetime_task_delete_loses_dirty` | 32 after GPU writes |
| A synchronize leaves the guest's copy valid, so a later exec's host write with flags 0 is never written back, not even by 53; invalidateHost over unsynchronized host writes discards them, whichever FIFO made them | **measured** (in-process; textures in every setup), `interplay_lifetime_paging_flags`, `_texture_paging_flags`, `_private_across_fifos` (section 6.11) | paging flags |
| Heap buffers, heap textures, views and textures on a buffer hold their base's Metal object: after the base is deleted they keep the old storage while the next listing makes a new base | **measured**, `interplay_lifetime_heap_deleted_under_buffer`, `_held_heap_*`, `_view_outlives_base` (section 6.11) | 37 of a heap, base or buffer |
| The object manager keeps one map per kind: a delete of the wrong kind forgets nothing, one reference can hold a buffer and a pipeline at once, and an object of the wrong kind stays cached, faulting, until a delete of its own kind | **measured**, `interplay_lifetime_one_reference_two_kinds`, `interplay_lifetime_icb_recreated` | retyped entries |
| A resume rebuilds each task from the root its root word names now, not the one it had | **measured**, `interplay_lifetime_suspend_root_word_rewritten` | root words rewritten while suspended |
| `rvg_device_will_suspend` then `unpause` over a FIFO blocked on a barrier leaves two threads on it: the old one runs the held work when the barrier's stamp arrives, and the new one waits for bytes the old one already read | **measured**, `interplay_dag_abandon_blocked` (section 6.24; a doorbell that claims more bytes kills the host, section 7.4) | an abandoned suspend over blocked work |
| The 2 s stamp-wait report over a waiter above 31 walks its alias's entry: with child 33 waiting it finds child 1 idle, logs "StampIndex %u waiting on invalid next waitingIndex" and stops, and the guest sees nothing | **measured**, `interplay_dag_alias_report` | children above 31 blocked past 2 s |

- **An ICB command's write needs its buffer referenced in the same
  command buffer** (Metal residency, not the library; **measured**, the
  decode walks): it lands only if the buffer is named there, by an 0x87
  in any encoder or by a blit or dispatch writing it; a 0xd6 barrier on
  it or only binding it does not count, and without such a reference
  whether the write lands depends on what earlier command buffers left
  mapped. Within one command buffer it is unordered against other work
  on the same buffer. `interplay_decodewalk_*` keeps to referenced,
  single-writer ICB outputs, so neither is locked.

### 7.2 Divergences between host setups

The table at the end of section 6.10. They are real behaviour: a guest sees
different results in the map setup and the isolate setup, under QEMU's
`apple-gfx-pci` with and without a descriptor table and under
`apple-gfx-mmio` (section 3.3). The codebase keeps them: its local tasks
and the service that runs each isolated task are separate code, each with
its setup's behaviour.

- **Only the blocks setup reports its unmaps in timing order under a
  pause.** A pause while GPU tasks exist gets the host's `unmap_memory`
  calls for task pages in an order that varies between runs (**measured**,
  `interplay_marathon_11[blocks]`, whose walker sorts the host's block
  calls during lifecycle calls). They are root-task mappings freed as the
  FIFOs drain: releasing a root-task mapping unmaps it through the device,
  which calls the host's `unmap_memory` on the root task (**read**).
  In the expected file every `unmap_memory` a pause gets names the root task,
  the 16 MiB host task the device created last (**measured**). The set of
  calls is locked; their order is not.

### 7.3 Host-API quirks

- `pause` twice and `unpause` without `pause` assert only when a guest display
  exists: the asserts are `GuestDisplay::pause`'s and `GuestDisplay::unpause`'s,
  and the device calls them only then. Without one, a second pause is harmless,
  and `unpause` while running raises an interrupt (**measured**,
  `device_lifecycle_pause_quirks`; with a guest display,
  `device_lifecycle_pause_twice_with_display_dies`,
  `device_lifecycle_unpause_without_pause_with_display_dies`,
  `suspend_host_asserts_pause_twice`, `suspend_host_asserts_unpause_unpaused`).
  So does `rvg_device_will_suspend` then `unpause` with a guest display:
  `rvg_device_will_suspend` never pauses the guest display, and its `unpause`
  asserts it was paused (**measured**,
  `interplay_life_will_suspend_unpause_display_dies`). Without a guest display
  that sequence is transparent (section 6.24).
- `rvg_device_finish_suspend` without `rvg_device_will_suspend` asserts; so do
  `rvg_display_cursor_position` and the display doorbell while paused, which
  read the guest display's page (the doorbell has an assert of its own;
  **measured**, `suspend_host_asserts_finish_without_will`,
  `suspend_host_asserts_cursor_while_paused`,
  `suspend_host_asserts_doorbell_while_paused`, and after GPU frames, cursor
  moves and presents from child FIFOs,
  `interplay_scanout_cursor_position_paused_dies`,
  `interplay_scanout_doorbell_paused_dies`).
- A failed resume leaves run state 3; any later mapping asserts
  (**measured**, `suspend_failed_resume_then_map`: a root-page write kills
  the host). So does teardown in the blocks setup, whose unmap asserts the
  same (`smoke_suspend_resume`, where a version-2 blob is refused last).
- A resume blob that passes the parse can still kill the host: a blob with the
  root FIFO enabled that `DeviceCore::create_root_fifo` refuses dies in
  `rvg_device_did_resume` (`DeviceCore::resume_root_fifo`;
  `suspend_resume_bad_root_fifo`).
- No `raise_interrupt`: the process ends at the first interrupt
  (**measured**, `device_create_no_raise_interrupt`).
- No task callbacks and no map: the device is created; the first
  root-page write ends the process (**measured**,
  `device_create_nothing_dies_on_root_page`).
- `add_trace_range` without `remove_trace_range` is refused at creation,
  `RVG_ERR_INVALID_ARGUMENT` (**measured**, `device_create_trace_add_only`,
  `efi_trace_add_only`); `remove_trace_range` alone is accepted, and EFI
  falls back to the timer.
- Display serial numbers are unique per process, not per device.
- `new_frame` calls coalesce.
- A frame encoded and held uncommitted makes the next present, guest
  connect, guest disconnect or `reset` wait up to 2 s for it (section 6.21;
  **measured**, `newops_held_*`). A frame released uncommitted
  (`rvg_frame_release`, the runner's `no_commit`) is never drawn and stays
  counted in flight for the display's life, so every later present waits
  its 2 s (**read**, the header and `display/frame.rs`; no scenario locks
  it).
- A frame target takes four formats only: RGBA8Unorm, BGRA8Unorm and their
  sRGB forms. Any other, such as RGBA8Uint, is refused at
  `rvg_frame_target_create` with `RVG_ERR_INVALID_ARGUMENT`, before a
  present and after one, and the host lives (**measured**,
  `covgap_dev_encode_integer_target_dies`, which completes).
- The IOSurface host device's registers are 64-bit
  (`rvg_iosurface_device_mmio_read` and `_write` take a `uint64_t`);
  QEMU's `apple-gfx-mmio` reads and writes them as 32 bits, and so does the
  runner, while `apple-gfx-iosurface-pci` passes an 8-byte access whole
  (section 3.1).
- The host's release of port 0's display while the EFI display holds it
  hangs, where Apple's framework hangs. A present loads the bound display and
  holds it to its end, so after the host's release the present ends the
  display, and the display's detach takes the device's display lock. An EFI
  register write (start, length, depth, mode or stride) holds that lock while
  its rebuild waits for the present queue, so neither returns. A mode-count or
  mode-value read holds the lock around its own load, so the display's end
  there waits for a lock its own thread holds. **Read** in Apple's framework,
  macOS 27.0.1 (26A434): `-[_PGDevice setEFIFramebufferMode:]` and its
  siblings hold the device's display `std::mutex` across the EFI display's
  call, and `getEFIModeCount` and `getEFIModeValue` hold it around theirs;
  `-[PGEFIDisplay updateFramebufferMode]` `dispatch_sync`s onto the present
  queue; `-[PGEFIDisplay present]` and `getModeCount` load the display with
  `objc_loadWeak` inside an autorelease pool they pop at their end;
  `-[_PGDisplay dealloc]` unplugs, and `detachDisplay:port:` locks the same
  mutex. A Darwin `std::mutex` locked again by the thread that holds it
  hangs (**measured**, M3 Max, macOS 27.0.1). No scenario locks it,
  since the runner cannot place a host release inside a present; the unit
  tests `device::displays::tests::efi_display_holding_the_last_reference`
  reproduce both hangs with bounded waits.

### 7.4 Guest-triggerable crashes and hangs

Each becomes a death test: one process, the outcome and the op it happened
at are the expected result, in every setup where it is reachable. A host
death in-process kills QEMU; `apple-gfx-pci` runs a task in-process unless
its guest's `apple-gfx-iosurface-pci` has a descriptor table (section 3.3).
Under isolation, a death in an isolated task's own code (the service's
checks, a NULL guest access, a raise no job catches) ends that task only, a
fault the guest sees: the exec faults, and every later command on that task
faults (status 4, section 4.8).

**A crash inside Metal ends the host in every setup.** Isolated tasks run in
the host's process, so a crash inside Metal (an abort, an assert or a fault
in Metal, IOGPU or the driver) on an isolated task's thread ends the host at
that op, as it does in-process.

"**measured**" with a scenario name means the suite locks it: the scenario's
expected files end `died` or `watchdog` (section 5.5), or, for the end of an
isolated task, record the faults. "Not locked" marks an outcome the suite
found or the code shows, that no expected file holds.

| input | outcome | evidence |
|---|---|---|
| any info reply (450-469), render 0xa4, or ICB CPU synchronize in an in-process task | host `abort()`: an in-process task has no writable cursor, and asking for one aborts | **measured** (461, exit 134); `info_reply_aborts_host`, `info_reply_aborts_host_cpu_buffer`, `info_460_bad_type_aborts_host` (460 fetches its cursor before it checks the type), `render_tile_dimensions_host`, `icb_sync_in_process` |
| in-process, `rvg_device_will_suspend` and `rvg_device_finish_suspend` with a guest-invalid ICB: the flush evicts it through the same cursor | host `abort()` | **measured**, `icb_evict_in_process` |
| opcode 66 on an existing rate map in-process | host `abort()`: an in-process task does not read a map's object-list entry (`LocalTask`'s `reset_rate_map`, `tasks/local/exec.rs`) | **measured**, `exec_reset_rate_map_in_process`, `render_ratemap_reset_death` |
| opcode 66 in the service on an immutable map, or on a map whose record no longer holds its header, its 0x14-byte command or its samples | the isolated task ends: the raises are uncaught in the service's job (`GpuTask`'s `reset_rasterization_rate_map`, `gpu_task/exec.rs`); a too-short command is not refused, since nothing checks a record's length against its command's (section 6.13) | **measured**, `exec_reset_rate_map_immutable` (the next command faults: the task has ended); `exec_reset_rate_map_tiny`, `_short` and `_under_command` lock only the fault; `mutgap_dec_rate_map_reset_7_bytes` locks the death for a 7-byte record, refused as shorter than its 8-byte header (`Deserializer::reset_rasterization_rate_map_with_serialized_data`) before its operation is looked at: the 66 faults, and so does the next; `mutgap2_rdec_rate_map_reset_short_entry` locks it for a record listed one byte short of its samples (section 6.16) |
| a page-table root of 0, then a walk | assert: the host without isolation; under isolation the task ends and the command faults | **measured**, `smoke_crash`, `tasks_service_crash` |
| a task root page of 0 in opcode 56 | host SIGSEGV (blocks, map); fault (isolate), the ID stays free | **measured**, `tasks_define_root_zero` |
| two in-flight commands with the same stamp value, signalled or faulted in either order | assert in the stamp queue | **measured**, `fifo_stamp_duplicate_signal_dies`, `fifo_stamp_duplicate_of_faulted_dies`, `fifo_stamp_duplicate_fault_dies` |
| two in-flight commands whose stamps are both faulted, the first still pending | the stamp queue's assert, in `StampQueue::fault_stamp`, that the pending stamp is not already faulted: host death at the doorbell. The fourth pair of the duplicate-stamp family above | **measured**, `fresh_dev_stamp_duplicate_fault_of_faulted_dies` |
| a blocks host that gives no `destroy_task` | host death at `reset` or `stop` with no user task, and at DeleteTask (32) or teardown with a user task, where the device calls the absent callback | **measured**, `fresh_dev_blocks_no_destroy_task_reset_dies`, `fresh_dev_blocks_no_destroy_task_delete_dies` |
| a blocks host that gives no `map_memory` | host death at the first root-page write | **measured**, `fresh_dev_blocks_no_map_memory_dies` |
| a blocks host that gives no `unmap_memory` | host death at `stop`; with a user task mapped, at a moment timing decides | **measured**, `fresh_dev_blocks_no_unmap_memory_dies` |
| child 33 deleted (49) while child 1 waits on a barrier, then child 2, sharing stamp index 1, waits on one too | `DeviceCore::finish_wait` clears the deleted child's bit with a 32-bit shift, so bit 1, and leaves child 1's waiting info set; child 2's wait passes the mask test and dies in `DeviceCore::add_wait`, whose entry for index 1 is not idle | **measured**, `fresh_dev_wait_alias_delete_dies` |
| the host paused (or after `rvg_device_finish_suspend`), the guest writes 0x1000 = 0 then 1, and a display command (opcodes 2, 4, 5, 7, 8, 9, 10, 11) | the restarted root FIFO runs the command into a paused guest display, whose methods die while it is paused (`GuestDisplay`'s `assert_running`). The dying doorbell's record may be missing: the guest display can die before it is written | **measured**, `fresh_dev_paused_nub_*`, `fresh_dev_suspended_nub_ack_dies` |
| `rvg_device_will_suspend`, `rvg_device_finish_suspend`, then `rvg_device_will_suspend` and `rvg_device_finish_suspend` again, with no resume between | with a display, the second `rvg_device_finish_suspend` dies at the guest display's pause assert. Without one, the second blob carries written, read, start and fault offset 0 with enable 1, and resuming it dies in `rvg_device_did_resume`, which cannot make the root FIFO (`DeviceCore::resume_root_fifo`) | **measured**, `fresh_dev_suspend_second_suspend_dies`, `fresh_dev_suspend_second_suspend_resume_dies` |
| `rvg_device_finish_suspend`, then `rvg_device_did_resume` on the same device, then a NOP on a child FIFO | the child signals through null stamp memory: host SIGSEGV | **measured**, `fresh_dev_suspend_finish_then_did_child_dies` |
| a fault offset >= 2^31 | assert in the stamp queue | **read**; unreachable, ring offsets stay below the ring size |
| writing a nonzero value other than 1 to 0x1000 while enabled | assert | **measured**, `device_mmio_enable_other_value_dies_2`, `device_mmio_enable_other_value_dies_80000000` |
| a stamp-wait report while a child FIFO waits whose stampIndex names another FIFO, whose own stampIndex differs | assert in `DeviceCore::fifo_for_stamp_index`, called from `DeviceCore::report_stamp_wait_timeout` | **measured** for the report 0x1028 asks for on a FIFO that is not faulted: `fifo_child_report_mismatch_dies`, `device_barriers_report_stamp_index_mismatch_dies`; and for the report a barrier blocked 2 s posts by itself, with no register write: `device_gap_barrier_timeout_report_mismatch_dies`, `interplay_fifo_timeout_report_mismatch_dies` (the host dies inside the guest's wait; `device_gap_barrier_timeout_report_matching` is the control, which survives); and for stamp index 1, which the lookup takes as child 1 like every index but 0: child 2 waits under index 1 while child 1 has index 4, `mutgap2_dev_report_stamp_index_1_dies` |
| defining a child FIFO (48) whose id is the stampIndex a blocked FIFO's wait is registered under | assert in `DeviceCore::create_fifo` that the id's waiting info is idle | **measured**, `interplay_fifo_define_over_waiting_stamp_index_dies`, `interplay_life_define_over_waiting_index_dies`, `mutgap2_dev_create_fifo_while_index_waits_dies` (child 1 waits under index 2, then 48 defines child 2). ID 0 is refused before the check, so a wait under index 0 only faults the root FIFO (`mutgap2_dev_create_fifo_0_while_index_0_waits`) |
| two child FIFOs sharing a stampIndex (nothing checks uniqueness), both waiting on barriers | assert in `DeviceCore::add_wait` | **measured**, `interplay_life_two_waiters_one_index_dies` |
| `rvg_device_will_suspend` then `unpause` while a child FIFO is blocked on a barrier, then a doorbell on that child that claims more bytes | `unpause` starts a second thread on the child while the first still waits (section 6.24); the new thread reads from the node's readCount, before the blocked command, parses it again and registers a second wait under the child's index: the same assert in `DeviceCore::add_wait`, at that doorbell | **measured**, `interplay_dag_abandon_blocked_dies`, in every setup (the blocked command carries a 4 MiB payload, so the death lands after the doorbell's record). With the root FIFO blocked instead, the replayed ring's define of the blocked child asserts in `DeviceCore::create_fifo`: measured, racing with `unpause`'s return, not locked |
| two waiters whose indices differ by 32 (child 1 and child 33, or the root and child 32) waiting at once | the same assert in `DeviceCore::add_wait`: the waiter mask is `1 << waiter` in 32 bits, so the second marks a bit already set | **measured**, `interplay_limits_child_wait_alias_dies`, `interplay_limits_root_wait_alias_dies` (each second waiter reaches its wait only after an 8 MiB NOP, so the death lands in the wait after its doorbell) |
| a resume whose blob has a root page and no root FIFO, over a child node that is running and holds a written command | the restored child runs it and signals its stamp through the null stamp memory the root FIFO would have provided: host SIGSEGV | **measured**, `interplay_life_resume_child_without_root_fifo_dies` |
| after a failed `rvg_device_will_resume` (run state left LeavingSuspendEarly, section 7.3): the EFI framebuffer's length write, which creates the EFI task | `DeviceCore::new_task_mapping` asserts the run state: host death at the write, in every setup | **measured**, `mutgap2_dev_failed_resume_then_efi` |
| after a failed `rvg_device_will_resume`, MapMemory2 (57) or UnmapMemory (34) of a task defined before it | blocks setup: `DeviceCore::map_memory_in_task` and `DeviceCore::unmap_memory_in_task` assert the run state: host death. With a memory map they never reach those calls, and the stamps land | **measured**, `mutgap2_dev_failed_resume_then_map_task`, `mutgap2_dev_failed_resume_then_unmap_task` |
| `rvg_device_finish_suspend` without `rvg_device_will_suspend`, on a fresh device with only a version (no FIFO, task or display) | assert on the run state: host death at the call. A host sequence, not the guest's (section 7.3) | **measured**, `mutgap2_dev_finish_suspend_fresh_dies`, and with a running FIFO `suspend_host_asserts_finish_without_will` |
| the cursor doorbell (0x1220) while the host has the device paused | assert in `GuestDisplay::doorbell`: host death at the register write (section 7.3) | **measured**, `suspend_host_asserts_doorbell_while_paused`, `interplay_scanout_doorbell_paused_dies` (after GPU frames and cursor moves; map, isolate); `mutgap3_res_display_doorbell_paused_dies` (one register write, on a port whose guest display has no display: nothing but the pause is needed); the display walks that end in it, `interplay_displaywalk_39`, `_40` and `_42` |
| a guest mode count above 128 at shared page +0x208 | assert at the next state machine run that updates the modes; presents do not look at the count | **measured**, `display_guest_mode_count_over`; `interplay_scanout_mode_count_over_dies` (while GPU frames flow: the host dies at its `set_modes` call). Exactly 128 is not a death: the update rewrites the count to 1 and raises the update (`mutgap2_dev_mode_count_128`) |
| a gamma table of one entry (opcodes 7, 8) | index 0xffffffff read: host dies (SIGBUS), in every setup | **measured** for opcode 7, `display_present_gamma_one`; for opcode 8 (a mapped surface, after GPU frames presented with 256- and 16-entry tables), `interplay_scanout_mapped_gamma_one_dies` (map, isolate) |
| an EFI mapping refused after a valid one, then a rebuild | the stale task: an unaligned length kills the host in every setup (with a memory map, the memory map's alignment assert); in the blocks setup the host's `map_memory` is called with a NULL task, and start 0, length 0 and a length over 64 MiB kill the host there | **measured**, `efi_stale_task_unaligned_length`, `efi_stale_task_start_0`, `efi_stale_task_length_0`, `efi_stale_task_length_over_64m` |
| EFI enabled with a display the host created without `new_frame` | `present_frame` ends the process | **measured**, `efi_display_without_newframe` |
| an IOSurface host device ring whose size × 16 is not a multiple of 16 KiB, written with a base set | the memory map's alignment assert | **measured**, `iosfc_ring_size_unaligned` |
| blocks setup: a lazy map over run blocks the range tracker has fragmented (481 single-page runs, a map of [0, 1000), a lazy probe of [0, 1010)) | host `abort()` in the tracker's add, which finds a block ordered before the run (`memory.tracker.add_unordered` on the case's stderr; the unit test `memory::tracker::tests::tasks_map_tracker_multiblock` reaches the same abort from a model of the runs) | **measured**, `tasks_map_tracker_multiblock` |
| blocks setup: a run block of 256 runs (adds stop at 240, but an unmap inside a run splits it in place up to the array's 256), then an unmap of a whole run in it | assert `run_count < RUNS_PER_BLOCK` in `RunBlock::remove`: host death | **measured**, `mutgap_mem_tracker_block_256_runs` |
| blocks setup: a map that finds its run block full among 128 blocks, while an emptied block stays after the others (runs added before an empty block all go to the block before it) and the runs' total is not a multiple of 240 | `compact` packs the blocks from the front, leaves the last non-empty one short with the empty one after it, and asserts `to_steal != 0`: host death at the map | **measured**, `mutgap_mem_tracker_compact_empty_block` |
| blocks setup: a run block of 240 runs (239 single-page runs and one of three pages), an unmap inside the three-page run, which splits it in place to 241 runs, then a map into the block | `RunBlock::split` keeps `old_count - half` runs, 121 and 121 of 241, and asserts that they add up: host death at the map | **measured**, `mutgap2_res_tracker_split_odd_death` (`mutgap2_res_tracker_split_adjacent` is the even control: 240 runs split into two of 120) |
| a guest version below 10 using object type 11 or 15 in-process | assert `mapperIOSurfaces`: host death. The service checks no feature, and a type 15 buffer works there | **measured** for type 15 in the map setup, `mutgap_svc_mapper_below_v10`; type 11 **read** |
| a malformed serialized pipeline descriptor (garbage, truncated to 1 or 5 bytes), a function ref resolving to nil (ref 0, a wrong type, a short record or header, data outside the task, commandType 14, an empty, truncated or two-function library), an unspecialized function constant, a render descriptor with a nil vertex function or a buffer as its vertex function | abort inside Metal: host death in every setup | **measured**, `compute_death_blob_*`, `compute_death_function_*`, `render_objects_malformed_*`. A function ref whose entry is empty (type 0), or whose IR sits over a hole in the task's page table, is not among them: it raises, and the exec faults (section 6.18, `interplay_lifetime_held_pipeline_function_retyped`, `mutgap_svc_function_over_hole`) |
| DispatchThreadgroupsIndirect or DispatchThreadsIndirect with indirect buffer ref 0 | nil reaches Metal: host death in every setup | **measured**, `compute_death_threadgroups_indirect_nil`, `compute_death_threads_indirect_nil` |
| a protection envelope with options 1 around a compute pass | Metal aborts opening the encoder: host death in every setup | **measured**, `compute_death_protection_options`; options 2 and all ones only fault (`compute_err_protected_encoder`, whose docstring says 3 dies too: not locked) |
| a second encoder over an open one (a carried decoder reused), an encoder left open at commit, a protection envelope while a carried encoder is open | Metal asserts: host death in every setup | **measured**, `compute_death_second_encoder`, `exec_carried_compute_reopened` and `_isolate`, `render_segments_open_twice`, `exec_open_encoder_at_commit` and `_isolate`, `render_segments_open_last`, `exec_protection_envelope_open_encoder` (section 6.13) |
| a render pass with no attachment but a target size, without 0x1e | Metal aborts on its compiler's callback thread once `renderCommandEncoderWithDescriptor:` has returned: host death in every setup, in the wait after the exec's doorbell. With no target size either, Metal returns no encoder and "Failed to create render encoder" faults, with or without 0x1e | **measured**, `render_pass_no_target`, `interplay_decodewalk_24` |
| a concurrent compute encoder (0xdb 1) whose 0x87 names a nil ref (ref 0, a non-resource ref, or one past the list); a serial encoder accepts it | host SIGSEGV in every setup | **measured**, `interplay_decodewalk_26`. A ref whose entry is empty (type 0) is not among them: it raises, and the exec faults (`interplay_lifetime_emptied_resource`) |
| consecutive compute encoders mixing dispatch types, with an indirect dispatch (0xc9 or 0xe6) | the host dies; in one program it killed isolate and not map, so it may depend on timing | **measured**, `interplay_decodewalk_27` (every setup) |
| a memoryless depth or stencil attachment with store MultisampleResolve, one sample and no resolve texture (the memoryless rule keeps that action) | Metal crashes: host death in every setup | **measured**, `render_pass_memoryless`, `render_pass_memoryless_stencil_resolve` |
| 8 programmable sample positions on a 4-sample pass | Metal aborts: host death in every setup | **measured**, `render_pass_msaa` |
| a rate map record of three layers | Metal crashes building the descriptor, before the library's layer-count check: host death in every setup | **measured**, `render_ratemap_three_layers` |
| a normal texture of pixel format 0; a format-550 texture 63 texels wide or multisampled; a 2D-array mapper texture that passes `validate_iosurface_descriptor`; in-process, a texture over a backing plane of 3 bytes per element | Metal aborts making it: host death in every setup | **measured**, `textures_invalid_pixel_format`, `textures_dual_plane_xr10_bad_row`, `textures_dual_plane_xr10_multisample`, `resources_mapper_texture_array`, `textures_backing_bad_element_size` |
| opcode 8 of a mapped surface Metal will not make a texture on: wider than the largest 2D texture (16400 BGRA pixels), 16400 high, or with a bytesPerRow not aligned to 16 (the IOSurface host device maps any of them) | Metal's descriptor validation aborts inside `newTextureWithDescriptor:iosurface:plane:`, before the library's check of the texture: host death (the display is the host's), with `MTL_DEBUG_LAYER` unset | **measured** (map, isolate), `mutgap_disp_mapper_texture_refused_dies` |
| an out-of-range, hole-covering or overflowing command-buffer range, a zero-length range (service), a page-aligned zero-length range (in-process), and in-process a range ending exactly at the task's end | `Deserializer::decode_segments` dies on the missing segment cursor: the host without isolation; under isolation the task ends, and the exec and the next command fault. A range 2^62 bytes long gets there in-process as longer than any local task (no cursor is made) and in the service as out of range | **measured**, `exec_null_cursor_past_end`, `exec_null_cursor_hole`, `exec_null_cursor_overflow`, `exec_zero_length_range_aligned`, `exec_range_at_task_end`, `tasks_objlist_cmdbuf_at_end`, `tasks_gap_cmdbuf_huge_length` |
| a function whose metallib ends exactly at the task's end | the copy of its data refuses `end >= taskLength` in-process and in the service, the function is nil, and Metal aborts building the pipeline: host death in every setup | **measured**, `mutgap_svc_function_at_task_end` |
| isolate: a command buffer whose second page's leaf names a page outside guest RAM | the service reads it in place: its cursor's second range starts at the NULL the memory map gives, and decoding into it ends the task: the exec and the next command fault | **measured**, `service_gap_cmdbuf_outside_ram` (in-process the copy's tail is zeros, section 7.5, and is not locked) |
| a ref texture (type 5) in a service task | the service does not support `backing_for_id`: the task ends, and every later command on it faults | **measured**, `textures_backing_ref_service` |
| the service releasing an IOSurface it does not hold for that mapping ID, or not the one it holds | the task ends | **read**, `gpu_task/surfaces.rs`; not locked |
| `CmdDelay` (61), then teardown, `reset`, `stop`, `release` (the device's end stops it), `recreate` (which releases first), `pause` or `rvg_device_finish_suspend`; with the Delay on a child FIFO, `pause` or `rvg_device_finish_suspend` | the call hangs forever, draining the Delay (teardown also on Apple's framework). `rvg_device_will_suspend` alone returns (it only quiesces), and a NOP rung after it is not run | **measured**, `fifo_op_delay`, `fifo_op_delay_long` (a 16-byte payload), `smoke_delay` (teardown); `interplay_life_delay_reset`, `_stop`, `_release`, `_recreate`, `_pause`, `_finish_suspend` (which locks `rvg_device_will_suspend` returning first), `interplay_life_delay_child_pause`, `interplay_life_delay_child_finish_suspend` |
| `CmdDelay`, then 0x1000 = 0 | the MMIO write itself hangs, draining the Delay | **measured**, `fifo_op_delay_disable` |
| `CmdDelay` on a child FIFO, then DeleteChildFIFO (49) | 49 hangs the root FIFO draining it; teardown then hangs | **measured**, `fifo_child_delete_delay_hangs` |
| a barrier on a FIFO's own slot for a stamp pushed behind a `CmdDelay` | not circular: the FIFO blocks for good, no fault; teardown quiesces the wait and hangs on the Delay | **measured**, `fifo_barrier_self_wait_pending` |
| a circular wait between two children above 31 (40 and 41 waiting on each other's first stamp) | not detected: the walk tests the waiting mask's bit 40, which the 32-bit waiter mask never sets, so both wait for good, with no fault; deleting one (49) abandons its wait. The same wait between children 8 and 9 faults the second ("Circular wait") | **measured**, `interplay_limits_high_children_circular` |
| an unsatisfied barrier | that FIFO blocks; after 2 s it logs, reports and waits again; a pause's quiesce wakes it and unpause waits again | **measured**, `fifo_barrier_blocked_wakes_at_timeout`, `fifo_barrier_blocked_wakes_on_signal`, `device_barriers_pause_while_blocked` |
| in process, `InfoIndirect` (69) behind an exec that runs longer than 5 s | 69 waits at most 5 s for the FIFO event the exec's completion signals, then reports status 3: a fault at 69's offset, published only after the exec's stamp. No GPU hang is needed: a kernel the guest holds by polling a word of its memory, through a shared buffer, is enough (`fresh_res_spin`). In the service the exec signals the event when its command buffer is scheduled (`encodeSignalEventScheduled`), so 69 completes, after the exec's stamp. A GPU error does not cause the fault either: a failed command buffer still signals the FIFO event, and a following 69 completes at once | **measured**, `fresh_res_info_while_kernel_held` (blocks, map: the fault; isolate: completes); `covgap_svc_gpu_fault_sync_info` |
| in-process, an exec whose synchronize step throws after both of its command buffers are committed (an ICB listed with synchronize flags, which is sorted into the blit list, where an ICB has no `synchronize_in_encoder`: the kind's default raises; a texture whose synchronize page has become a hole; or a view op 52 left host-valid and guest-invalid, listed with synchronize, which in the service faults (0x103c 0x88) with its fill landed, `mutgap4_res_view_synchronize_isolate`) | the stamp completes twice, from the exec's catch (status 2) and from the synchronize buffer's completion handler: host death on a stamp-queue assert. A race: with another stamp pending, the second completion is swallowed, so the scenarios make the exec the last command. In the service the exec faults and the late completion finds its token gone | **measured**, `covgap_svc_icb_exec_synchronize`, `covgap_svc_texture_hole_synchronize`, `covgap_svc_xr10_hole_synchronize`, `mutgap4_res_view_synchronize_dies` |
| a barrier wait on a child FIFO whose stamp index is at or above a FIFO count the guest lowered after creating it (children 31 and 32 made at version 245, then version 60) | assert `waiter < max_fifo_count` (`device/waits.rs`): host death at the wait | **measured**, `covgap_dev_barrier_waiter_above_count_dies` |
| isolate: an info reply into a buffer whose page is in a RAM hole | the service builds its writable cursor from the NULL host address and dies writing near 0; the exec faults | **measured**, `covgap_mem_reply_in_hole` |
| a texture backing plane 0 texels wide | Metal's assert "MTLTextureDescriptor has width of zero": host death (SIGABRT), blocks and map setups | **measured**, `covgap_mem_backing_zero_width` |
| in the service, a render exec that uses a heap its resource list does not name | the exec never completes, and its stamp holds every later one | found by the suite (the comment in `render_state_misc`, `scenarios/render_state.py`), **not locked**: that scenario lists its heap in every exec; compute's UseHeaps on an unlisted heap faults instead (`compute_err_unlisted_heap`) |
| a normal texture with a swizzle and a usage that allows shader writes; a rotated mapper texture that is not framebufferOnly | Metal asserts making it | found by the suite, **not locked**: `textures_gap_serializer_v2` and `textures_gap_mapper_rotation` avoid both |

### 7.5 Unwritten and over-read memory

Where a read leaves part of its destination unwritten, or runs past the data
it was given, the code gives those bytes fixed content, so what the guest
sees is deterministic. Most of these are not locked: the scenarios that
reach them keep away from the bytes, and those that are locked name their
case.

- **A guest-memory read that fails in-process** (the host's `read_memory`
  returns false, or the page is in no memory-map range) writes nothing
  (`DeviceCore::read_from_physical_address`, `device/root.rs`):
  - the local task's copy (`tasks/local/memory.rs`) reads the next page
    into the failed page's place, so the pages after it move up, and the
    rest of its buffer, zeroed before the copy, stays zero. A command
    buffer over a page outside guest RAM is decoded from those bytes (not
    locked: `service_gap_cmdbuf_outside_ram` runs in the isolate setup
    only, where the service reads in place);
  - a page-table walk (`memory/page_table.rs`) keeps a level's entries as
    they were: zeros in a level nothing was read into during the walk, so
    an unread root table is a hole and the command faults
    (`fresh_host_read_memory_refused`, blocks), and a table unread after
    another of its level keeps that table's entries;
  - `rvg_device_did_resume` reads a task root it cannot read as length 0
    (`device/lifecycle.rs`), and a task of length 0 is refused before
    `create_task` is called (`suspend_gap_task_root_unreadable` locks the
    map and isolate setups).
- **The gamma table** is read as 12 × count bytes, though only 4 × count
  are checked against its mapped length: the display copies the mapped
  length, and the bytes past it read as zeros (`display/gamma.rs`). Count 1
  is a host death (section 7.4).
- **Pipeline records longer than 16 + descriptorLength**: Metal gets the
  record's remaining bytes, zeros past descriptorLength (section 6.18).
- **The rate-map reset** reads 4 bytes past the record, as zeros
  (`deserializer/rate_map.rs`).
- **ExecIndirect2's flag bytes** under isolation: each record's flag byte
  carries bits 3-7 from the record before, and they are always 0, since the
  first record's are 0 and no record sets them (`tasks/remote.rs`); the
  service reads bits 0-2 only.
- **A display with no name**: the page's name is zeros
  (`display/guest_display/settings.rs`); the C API gives a NULL name "Apple
  Virtual", so no host reaches it.
- **Info replies' padding**: 457's two padding bytes, and 450's three bytes
  of tail padding (0x19 bytes of fields, packed and 4-aligned to 0x1c), are
  zeros (`deserializer/info.rs`). Neither is locked: `info_replies` dumps
  only the fields, and `interplay_diff_info_opcodes` never compares the
  tail.

## 8. Variance and host dependence

**Timing** (assert eventual state only): when stamps publish and how they
coalesce; when node readCounts are written; interrupt counts between
checkpoints; how many new-frame, cursor and glyph calls arrive; present
completion (up to 2 s); info-indirect's 5 s wait; the EFI 100 ms timer;
service handshake latency.

**Per run**: GPU addresses, resource IDs and unique identifiers (W8, W12).
No host virtual address is known to reach the guest (section 4.3).

**Host GPU and OS**: device info, compute info, pipeline replies, heap size
and alignment, rate-map sizes, 0x1224, argument-buffer support, detach vs
shared resource classes in the service, the version clamp by feature level,
every pixel the library's shaders draw (the host's Metal compiler and GPU
run them), Metal's pipeline blob format.

**Host boot**: output that changes from one boot of a host to the next and
is the same in every run within one. The host key (section 5.7) cannot tell
two boots apart, and `bless` makes its runs within one boot, so such output
is recorded and then fails after the next reboot, on every build alike. The
measured example is Apple's private fill (321) of a texture of pixel format
550 ([lessons.md](lessons.md)). A scenario keeps such output out of what it
observes: `textures_flush_xr10_on_suspend` and `mutgap_svc_xr10_synchronize`
fill texels their dumps do not cover, and write the dumped ones with a copy.

**Deterministic given inputs and order**: MMIO values apart from the above,
W3-W7, W16 apart from present completion, W17, suspend keys 1-19 and 21, every
nil, error and fault condition, and GPU compute and render output (decision
5).

## 9. Unknowns

- Whether GPU addresses, resource IDs and unique identifiers are stable
  between runs.
- Whether GPU output matches bit for bit across Apple GPU generations, as
  decision 5 takes it to. The host-keyed files come from one M3 Max;
  running them on another generation is the measurement.
- What EncodeStartIf compares on the GPU (section 6.15).
- Whether IOSurface wraps or copies the mapper's guest pages.
- Whether Metal's serialized pipeline format changes between macOS
  releases.
- Whether the library's shaders (section 3.2) compute what Apple's do on
  inputs no case reaches. On every frame and XR10 byte the suite records
  they agree with Apple's bit for bit (**measured**), and `gpu::shaders`'
  tests check each function on known pixels.

## Sources

Read in September 2026.

**Device and VMM testing**
- QEMU qtest: https://www.qemu.org/docs/master/devel/testing/qtest.html;
  a timer test: https://gitlab.com/qemu-project/qemu/-/blob/master/tests/qtest/cmsdk-apb-timer-test.c;
  libqos virtio: https://gitlab.com/qemu-project/qemu/-/blob/master/tests/qtest/libqos/virtio.c;
  commit 92cb8f8b: https://gitlab.com/qemu-project/qemu/-/commit/92cb8f8bf67e89254b0e4167521675eb6cd84981;
  vmapple's `host` CPU: https://gitlab.com/qemu-project/qemu/-/commit/51514a34b3dbc2cccf0c70aee07001ebbb804d20
- rust-vmm: https://github.com/rust-vmm/vm-virtio/blob/main/virtio-queue/src/mock.rs
- Firecracker: https://github.com/firecracker-microvm/firecracker/blob/main/src/vmm/src/devices/virtio/test_utils.rs
- crosvm: https://crosvm.dev/book/testing/index.html,
  https://github.com/google/crosvm/blob/main/e2e_tests/tests/backcompat.rs
- virglrenderer: https://gitlab.freedesktop.org/virgl/virglrenderer/-/blob/main/.gitlab-ci.yml,
  https://gitlab.freedesktop.org/virgl/virglrenderer/-/tree/main/.gitlab-ci/expectations
- gfxstream: https://github.com/google/gfxstream/blob/main/tests/end2end/gfxstream_end2end_tests.cpp

**GPU emulators and translation layers**
- Dolphin FifoCI: https://github.com/dolphin-emu/dolphin/blob/master/Source/Core/Core/FifoPlayer/FifoDataFile.h,
  https://github.com/dolphin-emu/fifoci/blob/master/fifoci/runner/__main__.py,
  https://github.com/dolphin-emu/fifoci/blob/master/fifoci/frontend/importer.py,
  https://dolphin-emu.org/blog/2015/01/25/making-developers-more-productive-dolphin-development-infrastructure/
- Mesa: https://docs.mesa3d.org/ci/index.html,
  https://gitlab.freedesktop.org/mesa/deqp-runner/-/blob/main/README.md,
  https://gitlab.freedesktop.org/mesa/piglit/-/blob/main/replayer/README.md,
  https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/bin/ci/update_traces_checksum.py
- Skia Gold: https://github.com/chromium/chromium/blob/main/docs/gpu/gpu_pixel_testing_with_gold.md

**Ports**
- rav1d: https://github.com/memorysafety/rav1d/blob/main/.github/workflows/test.sh,
  https://www.memorysafety.org/blog/porting-c-to-rust-for-av1/; rav2d:
  https://github.com/stukenov/rav2d
- zlib-rs: https://github.com/trifectatechfoundation/zlib-rs/tree/main/test-libz-rs-sys
- fish: https://fishshell.com/blog/rustport/
- uutils: https://github.com/uutils/coreutils/blob/main/DEVELOPMENT.md,
  https://bugs.launchpad.net/ubuntu/+source/rust-coreutils/+bug/2127970
- sudo-rs: https://ferrous-systems.com/blog/testing-sudo-rs/
- librsvg: https://github.com/GNOME/librsvg/blob/main/rsvg/tests/README.md,
  https://viruta.org/mutation-testing-librsvg.html
- typescript-go: https://github.com/microsoft/typescript-go/blob/main/internal/testutil/baseline/baseline.go,
  https://github.com/microsoft/typescript-go/blob/main/testdata/submoduleAccepted.txt

**Characterization, approval and snapshot testing**
- Feathers: https://michaelfeathers.silvrback.com/characterization-testing
- ApprovalTests: https://approvaltestscpp.readthedocs.io/en/latest/generated_docs/Reporters.html
- Jest: https://jestjs.io/docs/snapshot-testing; insta: https://insta.rs/docs/quickstart/
- Dodds: https://kentcdodds.com/blog/effective-snapshot-testing
- Fowler on non-determinism: https://martinfowler.com/articles/nonDeterminism.html

**Adequacy**
- SQLite: https://www.sqlite.org/testing.html
- Petrović et al.: https://arxiv.org/abs/2102.11378,
  https://research.google.com/pubs/archive/46584.pdf
