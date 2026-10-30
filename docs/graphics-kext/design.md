# reims-graphics: design

Status: stage 1 done. "Decided" holds what the owner and the research
settled, "Stage 2 wire protocol" is what stage 2 builds, and "Open" holds what
waits on the owner or on a later stage. The research's notes, with sources and
measured, read or inferred tags, are kept outside the repository.

## Goal

A second lane beside the Apple paravirt lane, with our own guest kext and
Metal GPU bundle, our own protocol, and our own QEMU device and host side.
Resources are stored and rendered in host GPU memory. The guest names them by
handle, and the guest CPU reaches GPU memory only where Metal gives it a
pointer.

It is engineered for maximum performance from the first version, without
cheating: every result a guest observes is Metal's, and no fast path trades
correctness.

Stages:
1. Research and probes, ending in this document.
2. The kext, unaccelerated: the device, the transport and the display.
3. Acceleration through the library's platforms (Metal on a Mac,
   metal2vulkan on Vulkan).

## Decided

1. **Device identity.** The device has its own PCI vendor:device ID, so
   Apple's paravirt kext never matches it and nothing of the guest OS is
   patched. The two lanes live side by side, and QEMU picks one by the device
   a VM is given. The lane serves x86-64 guests: Apple says third-party kexts
   are not supported in macOS VMs on Apple silicon, and vmapple guests boot
   the same way, so arm64 guests stay on the Apple lane.
2. **The cut: Metal-level calls.** The guest bundle streams Metal's calls,
   and the host decodes them into the library's `Platform`, which has a Metal
   and a metal2vulkan implementation, so both arms come with it. A
   Vulkan-level cut would put the Metal-to-Vulkan runtime, its shader
   translator and its caches in every guest process, translate on the
   guest's vCPUs, and translate twice on a Mac host.
   - Measured on one AMD host (XDC 2023): API forwarding (virgl, Venus,
     gfxstream) runs at 37-73% of bare metal, and DRM native context at
     64-98.5%. Native context wins by its mechanics, which this protocol
     keeps: the guest picks GPU addresses and sequence numbers itself,
     replies sit in shared memory, and mappings are made once.
   - In the Apple lane, copies are the cost and decoding is not: a 4K Finder
     drag moves 182-938 MB a frame to the GPU, while the decode threads use
     about 4% of a CPU.
3. **Queues and rings.** Each `MTLCommandQueue` gets one submission ring and
   one completion ring, created when the guest makes the queue. Thousands are
   allowed.
   - Encoders stream as they end, and at a size threshold inside a long
     encoder, so the host lowers encoder N while the guest encodes N+1.
     Commit is a small message.
   - No VM exit per submit. The host polls a busy ring, and the guest rings
     the doorbell (an `ioeventfd`) only after the host has marked the ring
     idle.
   - Each ring is decoded on its own host worker, so one queue's work never
     blocks another's. Single consumers are what limited the measured cases:
     virgl decodes every GL context on one host thread, and a global lock left
     gfxstream's GPU 8% busy (143 fps against 800 on Venus).
   - Native GPUs have one graphics engine, a few compute and copy engines,
     and far more queues than engines. RADV and ANV expose one graphics
     queue and NVIDIA 16, so on AMD and Intel the graphics work of every guest
     queue meets on one host queue, as native apps' work does.
   - WindowServer's queue runs at high priority where the host driver offers
     queue priorities.
   - Completion records (seqno, status, Metal's error and the failing
     encoders' info, GPU times) go in the queue's completion ring.
   - An interrupt fires only when the guest armed one, on several MSI-X
     vectors spread over the guest's CPUs.
   - Presents are never batched. Every command buffer completes exactly, with
     no grouping.
4. **Memory model.** VRAM is the primary allocation. `hasUnifiedMemory`
   follows the host GPU.
   - On a discrete GPU it is false:
     - Private buffers and all textures live in VRAM and are never mapped
       into the guest.
     - Managed resources are two copies: the VRAM copy, which is primary, and
       a host-cached CPU copy the guest maps.
       - `didModifyRange:` copies exactly the named range to VRAM, before any
         command of the next submission.
       - `synchronizeResource:` copies VRAM back to the CPU copy inside the
         blit encoder.
       - A range message is ordered before every later commit of its
         context, on any queue, as Metal promises.
       - Managed textures get two copies too: a linear CPU copy the guest
         maps, beside the VRAM image. `replaceRegion:` and `getBytes:` are
         then guest memcpys plus a region message, with no round trip.
     - Shared buffers are host-cached system memory mapped into the guest,
       which the GPU reads over PCIe (measured on a 3090: CPU reads 45 GB/s
       there, against 0.01 GB/s from resizable-BAR VRAM), whatever their
       CPU cache mode.
     - A texture made over a Shared buffer shares the buffer's system memory,
       and the GPU samples it there, as Apple documents for discrete GPUs:
       no VRAM copy and no dirty log (owner, 2026-10-30).
   - On an integrated GPU (Apple silicon, AMD and Intel iGPUs) it is true,
     with one copy of each resource.
   - The host allocates every block, and the guest sub-allocates buffers
     inside blocks, so creating a block is the only memory call that waits
     for the host. A block mapped into the guest is a dedicated, page-aligned
     host allocation.
5. **The guest maps host memory only.** VRAM is reached by the host alone.
   - What crosses the API (draws, uploads, `didModifyRange:` ranges and
     readbacks) reaches VRAM by the host's own GPU copies.
   - The guest maps memory only where Metal gives the app a pointer:
     `contents` of Shared and Managed buffers, and IOSurfaces the CPU locks.
     That memory is plain anonymous host memory, zeroed with stores, imported
     into Vulkan by pointer and mapped into the BAR. KVM maps it like guest
     RAM on any kernel, and the host GPU reads it in place. The staging rings
     live there too.
   - Real discrete Macs keep Shared in system memory, so this is Metal's
     discrete model.
   - VRAM mapped into the guest was considered and dropped. It needed kernel
     6.13 or later and resizable BAR, and was good for guest writes only:
     measured on amdgpu, CPU reads from VRAM run at 15 MiB/s with memcpy,
     against about 10 GiB/s for writes. It returns only if a measured frame
     shows the GPU's reads of Shared memory across PCIe costing.
   - The BAR is one host memory region that the device reserves at start and
     registers with KVM once. Blocks are ranges inside it, so making or
     freeing a block never changes the VM's memory map. It is ordinary
     cached memory, and the guest maps it write-back. Uncached loads cost
     79-138 ns against 2-4 ns cached, and AMD CPUs apply the guest's choice.
   - The region is private anonymous memory of QEMU's process, and the host
     side runs in that process: RADV refuses to import a memfd's or any shared
     mapping (measured on the 780M and the RX 7900 XTX).
   - Blocks are few and large, each imported into Vulkan once, and the guest
     sub-allocates buffers inside them. RADV lists every import in every
     submission, and an import costs 17-43 us plus about 16 us per MiB of
     fresh pages.
6. **Addresses and IDs.** The guest bundle picks every `gpuAddress` and
   `gpuResourceID`, and the host pins them in metal2vulkan's own Metal
   address space. Creating a buffer then needs no round trip, and argument
   buffers need no patching. Host device addresses never reach the guest.
7. **Contexts.** Every guest process that opens the device is a context, with
   its own resources, queues, and slices of the address and ID spaces.
   Sharing across contexts (IOSurfaces, shared events, shared textures) is
   explicit, by a global name.
8. **Shaders.** AIR crosses once, named by its SHA-256, with each pipeline's
   descriptor and function constants. The host takes no keys or reflection
   from the guest, and answers once translation finishes, without waiting
   for the driver's compile.
9. **Synchronization.** The bundle sends Metal's own semantics exactly:
   tracking modes, resource usage, `useResource`/`useHeap`, memory barriers,
   fences and events. The host derives the Vulkan barriers. Shared-event
   values sit on a page the guest reads, so `signaledValue` needs no round
   trip.
10. **Device answers.** One versioned blob at device open holds every answer,
   taken from the host GPU, and the bundle caches it.
11. **Stage 2 lays the final transport.** The rings, doorbells, interrupts,
    contexts and the display path are the final ones from stage 2 on, and
    stage 3 adds commands to them. The stage 2 display scans out host memory
    with no copy on its way to the host window.
12. **The stage 2 guest stack.** The kext subclasses `IOFramebuffer` on our
    PCI device. It implements the required methods (modes, pixel format,
    aperture, cursor, vblank and connect interrupts, display info), and
    matches with a probe score above Apple's generic IONDRV driver, which
    matches any display-class device. WindowServer composites in software
    with no Metal device: x86-64 macOS 26.5.2, 26.6.1 and 26.7.1 reach the
    desktop that way ("Software compositor activated"; measured).
13. **Loading: through OpenCore, with a bridge for our kext only.** OpenCore
    injects kexts into the Boot kernel collection and links them only against
    Boot-collection symbols, but a display kext uses IOGraphicsFamily, which
    is in the System collection. So our kext needs a loader bridge (a Lilu
    plugin plus a generated shim) to reach it, exactly as the x86-64 port's
    kexts do. That bridge is infrastructure, separate from the port's GPU
    driver kexts: our lane replaces those driver kexts with our own and keeps
    none of them. The clean image carries only Lilu, the loader, our shim and
    ReimsGraphics.kext. Nothing of the guest OS is patched or hooked beyond
    what this bridge already does for the existing lane.
    - Iterating: patch a copy of a VM's OpenCore image directly
      (scripts/internal/opencore-add-kexts.sh). No rebuild inside the guest.
    - Shipping (later): reims-util injects the set when it builds an image.
      Not needed to prototype.
14. **IOSurfaces follow the discrete model.** IOSurface.kext allocates
    surfaces in guest RAM, CAMetalLayer drawables and window backing stores
    included. The accelerator gets a per-surface device cache with a sync
    callback when the CPU locks the surface. So a surface's guest pages are
    its CPU copy, and its GPU copy lives in VRAM as the device cache, synced
    at the points IOSurface and the bundle define: CPU lock and unlock, and
    the GPU's use of a CPU-written surface. Surfaces the CPU never touches
    are never copied. A surface the GPU renders and the CPU locks is two
    copies synced at lock and unlock, as Managed textures are. A texture over
    host memory (over a Shared buffer, or a CPU-locked surface) follows
    metal2vulkan's host-page rules: in place where only sampled, otherwise a
    shadow copy that is announced. Stage 2 needs none of this, since its
    compositor is software.

15. **The stage 3 stack is IOGPUFamily.** x86-64 macOS 26 carries IOGPUFamily
    and IOGPU.framework, and the x86-64 port runs Metal compositing on them:
    WindowServer composites with Metal through an IOSurface display pipe on
    IOGPU that the driver provides. Our accelerator does the same.

## What metal2vulkan adds in stage 3

- No-copy Managed buffers and heaps over our host blocks, whose VRAM copy
  syncs at `didModifyRange:` and `synchronizeResource:`. Today a no-copy
  buffer has no device copy.
- Two-copy Managed textures, whose host copy is one of our blocks.
- Per-context address and texture-ID tables. Today one device has one
  address space and one texture table of up to 2^16 entries.
- A completion hook on its waiter thread, saving a thread hop.

## Stage 2 wire protocol

Stage 2 builds the device, the transport and an unaccelerated display. The
transport is final: stage 3 adds rings per command queue and commands, in the
same formats. All values are little-endian.

**PCI.** Class 0x0380 (display controller, other). The vendor:device ID is
the owner's call (Open); development uses the placeholder `1234:5247`.
Apple's paravirt kext matches only `106b:eeee`.

**BAR0: registers**, 4 KiB of MMIO, 32-bit accesses. The MSI-X table and PBA
sit in BAR0 from 0x800.

| offset | name | access | meaning |
|---|---|---|---|
| 0x000 | `MAGIC` | RO | 0x58464752 ("RGFX") |
| 0x004 | `VERSION` | RO | 1 |
| 0x008 | `FEATURES_LO` | RO | feature bits 0-31; 0 in stage 2 |
| 0x00c | `FEATURES_HI` | RO | feature bits 32-63 |
| 0x010 | `STATUS` | RW | writing 0 resets the device; the guest writes 1 (driver ready) once it has read the rings' headers |
| 0x014 | `REGION_SIZE_LO` | RO | BAR2's size, low 32 bits |
| 0x018 | `REGION_SIZE_HI` | RO | BAR2's size, high 32 bits |
| 0x020 | `DOORBELL` | WO | writing a ring's id wakes its consumer (an `ioeventfd`) |

**BAR2: the region.** A 64-bit prefetchable BAR over host memory (256 MiB),
mapped write-back by the guest. The device places the control rings at fixed
offsets; the guest places everything else.

| offset | size | what |
|---|---|---|
| 0x00000 | 64 KiB | control submission ring, id 0 (the kexts to host) |
| 0x10000 | 64 KiB | control completion ring (host to kext), MSI-X vector 0; the device info at 0x18100 |
| 0x400000 | 128 MiB | the framebuffer's scanout (`RGFX_SCANOUT_SPAN`) |
| 0x8400000 | to the cursor | the accelerator kext's blocks: every process's queue rings, Shared buffers and heaps, surfaces |
| end − 256 KiB | 256 KiB | the framebuffer's cursor image |

The control ring is the kexts' alone: the framebuffer's display commands, and
the object lifecycle the accelerator posts for the Metal processes, under the
framebuffer's ring lock. Every other ring is one command queue's own ring pair,
in a block the accelerator kext handed to the queue's process, so each ring
keeps a single producer and a single consumer however many processes run
Metal.

**Contexts.** Each process that opens the accelerator's user client gets a
context from the kext: a number `c`, handed out once from 1 (its object ids are
`c << 40` and up), and one of 32 slots, held until its close, for its GPU
addresses (`0x1_0000_0000 + s << 36`) and texture resource ids
(`1 + s * 2048`). The kext hands out its region blocks (first fit, page
granular). The process hands each create and destroy of its objects (ids in its
range) to the kext by a user-client call; the kext posts it on the control ring
and, for a pipeline create, hands the reflection back once the host answers
it. A queue's `QUEUE_CREATE` goes the same way, its rings checked to lie in the
process's blocks, and is answered once the host has the rings. At the
process's close or death the kext destroys its queues, then closes its context
(`CONTEXT_CLOSE`: the host drops its objects, then answers), then frees its
blocks. A command buffer is decoded once the host has carried out every
control-ring message published before its queue's thread took it.

**Rings.** A 256-byte header, then a byte ring whose size is a power of two
(64 KiB minus the header for the control rings). Each header field sits in its
own 64-byte line:

| offset | field | written by |
|---|---|---|
| 0x00 | `producer`: u64, free-running byte count | producer |
| 0x40 | `consumer`: u64, free-running byte count | consumer |
| 0x80 | `consumer_idle`: u32, 1 while the consumer sleeps | consumer |
| 0xc0 | `armed`: u32, 1 when the guest wants an interrupt | guest (completion rings) |

- A message starts with `type: u32`, `size: u32` (bytes including this
  16-byte header, a multiple of 8) and `seqno: u64`. The guest numbers its
  commands from 1, per ring. A message never wraps: type 0 pads to the
  ring's end.
- Doorbell. The producer writes the message, publishes `producer` (release),
  then issues a full fence and reads `consumer_idle`. If it reads 1, it writes
  the ring's id to `DOORBELL`. Before sleeping, the consumer sets
  `consumer_idle`, issues a full fence and reads `producer` again, so no wake
  is lost. It clears `consumer_idle` when it wakes.
- Interrupts. The host publishes a completion, issues a full fence and reads
  `armed`. If it reads 1, it clears it and raises the ring's vector. The guest
  sets `armed` after draining, issues a full fence and checks `producer`
  again.

**Control commands** (guest to host, ring 0). Each is answered by a
`COMPLETE` with its `seqno` and a status (0 for success), or by the reply
named.

| type | name | payload | answer |
|---|---|---|---|
| 1 | `GET_DISPLAYS` | none | `DISPLAYS` |
| 2 | `SET_SCANOUT` | display u32, width u32, height u32, stride u32, offset u64 into BAR2; width 0 blanks the display | `COMPLETE` |
| 3 | `SET_CURSOR` | display u32, width u32, height u32, hot_x u32, hot_y u32, offset u64 of a BGRA8 premultiplied image; width 0 removes the image | `COMPLETE` |
| 4 | `MOVE_CURSOR` | display u32, x i32, y i32, visible u32 | none |
| 5 | `SET_VBLANK` | display u32, enabled u32 | `COMPLETE` |

Scanout pixels are 32-bit BGRA8 (`kIO32BGRAPixelFormat`), the one format in
stage 2.

**Events** (host to guest, the control completion ring):

| type | name | payload |
|---|---|---|
| 0x100 | `COMPLETE` | status u32 |
| 0x101 | `DISPLAYS` | count u32, then per display: id u32, connected u32, preferred mode index u32, mode count u32, then per mode: width u32, height u32, refresh in mHz u32 |
| 0x102 | `DISPLAY_CHANGED` | display u32; the guest asks `GET_DISPLAYS` again |
| 0x103 | `VBLANK` | display u32, count u64, host time in ns u64 |

**Display.** The modes offered are the host window's size and its screen's
refresh rate (the guest runs at the host screen's rate). The host presents the
scanout buffer to the display's window at the host's refresh rate, only when
the guest wrote it, which it learns from KVM's dirty log over the scanout range
(QEMU's VGA devices do the same). The GPU reads the buffer in place: it is
imported once, so no copy is made on the way to the window. While vblank is
enabled, the host sends `VBLANK` at each refresh of the window's screen.

**Boot display.** The device's option ROM gives firmware a GOP framebuffer in
BAR2, and the kext takes the display over when it starts.

## Stage 3 wire protocol: the Metal-call stream

Stage 3 keeps the stage-2 transport unchanged — the registers, the region, the
256-byte ring header, the doorbell and interrupt handshake, and the 16-byte
message header (`type`, `size`, `seqno`). It adds rings per command queue, a
family of device-level object-lifecycle messages, command-buffer encodings on
the per-queue submission rings, and completion records on the per-queue
completion rings.

Our guest Metal GPU bundle is the host of Apple's Metal SPI: every Metal call
an app makes lands in one of the bundle's objects, and the bundle serialises it
into this stream. The host reads the stream and drives the library's `Platform`
(Metal on the Mac, metal2vulkan on Linux), the same trait reims-vgpu-core
already drives from the Apple-lane command stream. So the host decoder is: wire
message → handle-table lookup/insert → `Platform` call → completion. The
bundle's serialiser is the mirror of that decoder.

**Handles and addresses.** Every guest Metal object has a 64-bit `ObjId` the
guest picks, unique per device; the host keeps an `ObjId` → `Platform` handle
table (handles are refcounted in core, `clone_ref`/`Drop`). The guest also
picks buffer/texture/heap GPU addresses and resource IDs, and the host pins
them — the host never hands a host device address to the guest (m2v's rule). A
destroy carries the `ObjId` alone; the host drops its handle.

**Rings.** Ring 0 stays the control ring: the framebuffer kext's stage-2
display commands, and the device-level object lifecycle below, which the
accelerator kext posts there for each process. Every other ring is one
`MTLCommandQueue`'s ring pair (one submission and one completion ring per
queue, created on demand), which the bundle places in its kext-given blocks and
creates through the accelerator kext. Each ring keeps a single producer and a
single consumer, so their `consumer_idle` handshakes never race. A create carries a priority, and the host gives that ring's VkQueue a
matching priority, so WindowServer's queue runs high.

**Object lifecycle** (ring 0, through the accelerator kext, request/reply). Each `*_CREATE` carries our
schema for the Metal descriptor (plain little-endian fields, decoded straight
into the `Platform` descriptor struct — core keeps descriptors as public-field
structs), the guest-picked `ObjId`, and, where it applies, the address, id or
offset. Resource creates with a guest-picked id/address are fire-and-forget
(the host builds the mirror and pins lazily on first use); pipeline creates
block for the reply, which carries the reflection the guest needs
(`threadExecutionWidth`, max total threads per threadgroup, static threadgroup
memory) — m2v's "take state answers from the created pipeline".

| group | messages | `Platform` call |
|---|---|---|
| memory | `HEAP_OVER_HOST_MEMORY`, `HEAP_CREATE`, `BUFFER_CREATE`, `BUFFER_OVER_HOST_MEMORY`, `BUFFER_AT_OFFSET`, `TEXTURE_CREATE`, `TEXTURE_FROM_BUFFER`, `TEXTURE_FROM_IOSURFACE`, `*_DESTROY` | `new_heap_over_host_memory`, `new_heap`, `new_buffer`, `Heap::new_buffer_at_offset`, `new_texture`, `Buffer::new_texture`, drop |
| state | `SAMPLER_CREATE`, `DEPTH_STENCIL_CREATE`, `*_DESTROY` | `new_sampler_state`, `new_depth_stencil_state` |
| shaders | `LIBRARY_CREATE` (`.metallib`/AIR bytes forwarded unchanged), `RENDER_PIPELINE_CREATE`, `COMPUTE_PIPELINE_CREATE`, `*_DESTROY` | `new_library_with_data`, `new_render_pipeline_state`, `new_compute_pipeline_state` |
| sync | `EVENT_CREATE`, `SHARED_EVENT_CREATE`, `FENCE_CREATE`, `*_DESTROY` | `new_event`, `new_shared_event`, `new_fence` |
| queue | `QUEUE_CREATE` (priority, region offset, vector), `QUEUE_DESTROY` | `new_command_queue` |

`*_OVER_HOST_MEMORY` map guest region blocks (Shared) and the host builds a
placement heap/buffer over them. Managed is backed as Shared tonight (one
coherent copy, allowed by Metal's contract), behind one function so the
two-copy VRAM form drops in when m2v ships it.

**Command buffers** (per-queue submission ring). A `COMMAND_BUFFER` message is
one command buffer: its `ObjId`, then a byte stream of encoder ops. The decoder
opens the `Platform` command buffer and walks the ops:
- `ENCODER_BEGIN_RENDER` (our render-pass schema: colour/depth/stencil
  attachments by texture `ObjId`, load/store actions, clears),
  `ENCODER_BEGIN_BLIT`, `ENCODER_BEGIN_COMPUTE`; `ENCODER_END`.
- one op per `Platform` encoder method — set pipeline/buffer/texture/sampler/
  bytes, viewport, scissor, draw / draw-indexed / dispatch, copy/blit, update/
  wait fence, `useResource`/`useHeap`. The op set is the trait's encoder
  methods; the first slice is the subset WindowServer and simple Metal apps
  issue.
- `CB_SIGNAL` / `CB_WAIT` (event `ObjId`, value) → `encode_signal` /
  `encode_wait`; `CB_PRESENT` ties a drawable to the scanout (the present path,
  outside the trait); `CB_COMMIT` closes the message — the host commits and
  registers an `add_completed_handler` that writes the completion record.

Metal splits one command buffer into several kernel submissions (a render pass;
a run of blits or of computes with no event between) — m2v's measured
granularity. The host preserves it so IOGPU error reporting lines up.

**Completion** (per-queue completion ring). A `CB_COMPLETE` record: the
command-buffer `ObjId`, status, GPU start/end time, and on error the
`CommandBufferError` the host maps 1:1 from the IOGPU code the runtime returns
(page fault, hang/timeout, recovery victim, access revoked, invalid resource —
the map m2v gave). A shared event posts `EVENT_SIGNALLED` (`ObjId`, value) as it
fires. The ring's armed/interrupt handshake is the stage-2 one.

**Left for later** (extends the same op families, not the transport): enqueue
ordering held in the decoder; command-buffer/queue descriptors and
`errorOptions`; labels and debug groups answered guest-side; residency sets
(`useResource`/`useHeap` are in); `MTLIOCommandQueue`; Metal 4.

**Where the decoder lives.** In reims-vgpu-core, generic over `P: Platform`,
beside the Apple-lane decode it mirrors (same handle machinery, descriptor
structs and `Platform` calls). The reims-graphics crate stays the transport
(rings, region, registers); the decoder consumes ring bytes and drives
`Platform`.

## Stage 3 guest stack (from the prior-art study)

How a Metal bundle plugs into macOS, in Apple's own terms (notes:
`stage3-apple-metal-iface.md`):

- **Discovery/load.** The bundle installs under `/Library/GPUBundles/`. Metal
  does not scan that directory; it loads the bundle from two properties on the
  accelerator service's IOKit personality — `MetalPluginName` (the path) and
  `MetalPluginClassName` (the Obj-C class Metal instantiates as the
  `MTLDevice`) — under `IOMatchCategory IOAccelerator`. So stage 3 needs a
  guest **accelerator kext** that matches our PCI device and publishes those
  two keys; the bundle ships beside it.
- **Device class.** The bundle's device subclasses Apple's private IOAccel
  Metal base class and conforms to `<MTLDevice, MTLDeviceSPI>`; entry
  `-initWithAcceleratorPort:` (may return `nil` to decline). The base fills the
  limit tables (zero limits make CoreImage refuse), and the bundle implements
  the whole `MTLDevice` object graph plus enough of the GL/OpenCL private
  selectors. AIR arrives as `.metallib` through `newLibraryWith*`; the bundle
  forwards those bytes over the wire (`LIBRARY_CREATE`), the host compiles them.
- **Kernel family — the fork.** The prior art builds its accelerator on the
  older `IOAcceleratorFamily2`, but our guest already runs the **ported
  IOGPUFamily stack**, and IOAccel2 cannot start in a kernel where IOGPU runs.
  So our accelerator aligns with **IOGPU** (owner's correction, 2026-10-30),
  and the IOAccel2 user-client selectors do not transfer; our accelerator
  exposes the alloc/map + submit + completion-stamp + IOSurface-sharing
  boundary over our own protocol, not Apple's IOGPU user client.
- **Compositor/present.** WindowServer composites into IOSurfaces and presents
  through a display pipe. Our ported IOGPU stack already runs Metal compositing
  through its IOSurface display pipe (type-4 adapter), and we already have the
  framebuffer/display kext; stage 3's display work is making the accelerator and
  the framebuffer agree on the IOSurface scanout (flip, else copy).
- **Load policy.** Loading an unsigned bundle into WindowServer, the Metal
  compiler helpers and sandboxed apps needs relaxed library validation — a
  guest boot-policy step we control (OpenCore/boot-args), not a code concern.

## Stage 3 accelerator kext (scaffold)

`RgfxAccelerator` (`guest/kext/src/RgfxAccelerator.{hpp,cpp}`) is a second
IOService in the one kext, on the same PCI device as the Stage-2 framebuffer:
the framebuffer matches in category `IOFramebuffer`, the accelerator in category
`IOAccelerator`, both children of the one `IOPCIDevice`. Its `Info.plist`
personality carries `MetalPluginName
= /Library/GPUBundles/reims-graphics.bundle` and `MetalPluginClassName
= RgfxMetalDevice`, which is how Metal (`IOServiceMatching("IOAccelerator")`)
finds and loads the bundle. The accelerator maps BAR 0 for the doorbell and
holds BAR 2 to hand to the bundle; it touches no pixels and no control ring (the
framebuffer owns those).

`RgfxAcceleratorUserClient`
(`guest/kext/src/RgfxAcceleratorUserClient.{hpp,cpp}`) is **our own** transport,
not Apple's IOGPU user-client ABI: `clientMemoryForType(kRgfxAccelMemoryRegion)`
maps the BAR-2 region into the Metal client's task (so the bundle's `RgfxRegion`
/ `RgfxRingWriter` write into real device memory and read the caps page), and
`externalMethod(kRgfxAccelMethodDoorbell)` writes a ring id to
`RGFX_REG_DOORBELL`. These ids/selectors are defined in `RgfxAccelerator.hpp`;
the bundle's wire handshake (a boot-gated stub in `RgfxMetalDevice` today) opens
them with `IOServiceOpen` / `IOConnectMapMemory` / `IOConnectCallScalarMethod`.

- **Boot-gated — base class / kernel family (notes UNKNOWN #1).** The scaffold
  subclasses `IOAccelerator`, the base the macOS SDK and KDK expose (the IOGPU
  kernel family — `IOGPU`, its device/queue classes and its submission/
  allocation user-client ABI — is Apple-private and in neither, so it cannot be
  subclassed or linked off the guest). On the guest the base very likely becomes
  the IOGPU family class per Decided 15; the Metal discovery keys, the PCI match
  and our region-map + doorbell user client are expected to carry over. The
  IOGPU user-client ABI is deliberately not invented here.

## Stage 3 device control

`RgfxDeviceControl` (`guest/kext/src/RgfxDeviceControl.{hpp,cpp}`) is the
third service on the PCI device, in a category of its own: an
`AppleGraphicsDeviceControl` (AGDC), the System-KC library class
AppleGraphicsControl.kext's plug-in defines. WindowServer's IOPresentment
builds a display pipe only for a framebuffer whose PCI device has an AGDC,
and AppleGPUWrangler pairs the AGDC with the GPU's record and sets its
attribute 3 to its `InitFlags`. Without one, CoreDisplay logs "Unable to find
display pipe for IOFB service" and WindowServer picks the software
compositor. The kext names the library in `OSBundleLibraries`; the loader
binds its base-class sites as it binds IOGraphicsFamily's.
`AppleGraphicsDeviceControl.hpp` is our declaration of the base class, from
the names and vtable order the KDK's x86_64 binary exports. The control
answers attributes 1 (its description), 3 (what it makes), 0x301 (the
wrangler's GPU index), 0x921 and 0x980 (which IOPresentment reads before it
builds a pipe), and refuses the rest.

## Open

- **Stage 3 bundle placement — RESOLVED in principle:** the bundle sits in
  `/Library/GPUBundles` and our accelerator kext's personality carries
  `MetalPluginName`/`MetalPluginClassName`. Open: the exact IOGPU accelerator
  user-client ABI our kext exposes, and the library-validation boot state —
  both settled by a guest experiment, not desk work.
- **IOSurface details:** whether the device-cache sync runs on every CPU
  access path and on unlock (window buffers wrapped no-copy are written
  without a lock); whether IOGPU accepts a device cache that holds a host
  handle and no guest pages; and which surfaces the guest CPU touches
  (probe on the Apple lane: KVM's dirty bits per mapped surface).
- **The PCI vendor:device ID** the device ships with (owner call).
- **Owner calls**, each with a recommendation once the evidence is in:
  - the GPU name and families an x86-64 guest sees;
  - whether host device addresses are ever exposed.
