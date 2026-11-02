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
   - The bundle resolves a function's constant values as Metal does, against
     the constants the library's own bytes declare (each function's `CNST`
     table): a value set at a constant's index (by index or by range) wins,
     else the one set under its name; values for constants the function does
     not declare are ignored; a value of another type than its constant's
     fails the function with Metal's error; a required constant with no value
     reads 0, and an optional one stays undefined. Each pipeline create
     carries one record per constant given a value, by index, with up to 32
     bytes (a `long4`). A function made with no values builds no pipeline:
     Metal asserts there, and the bundle refuses it.
9. **Synchronization.** The bundle sends Metal's own semantics exactly:
   tracking modes, resource usage, `useResource`/`useHeap`, memory barriers,
   fences and events. The host derives the Vulkan barriers. Shared-event
   values sit on a page the guest reads, so `signaledValue` needs no round
   trip.
10. **Device answers.** One versioned blob at device open holds every answer,
   taken from the host GPU, and the bundle caches it.
   - The device shows the host GPU's real name, and claims the Apple GPU
     families the host platform claims, with every family and feature set
     those contain, as an Apple-silicon Mac of that generation answers. It
     invents no family of its own (owner, 2026-10-31).
   - An Apple family is claimed only up to the highest one whose rows in
     Apple's Metal Feature Set Tables the lane fully carries, and only where
     the platform claims it too; the iOS feature sets of a family not
     claimed answer no with it (2026-10-31). Apple1 answers as Apple2, as
     Metal does: the tables' first Apple column is Apple2. Today the lane
     carries Apple1 and Apple2: Apple2's rows need argument encoders,
     placement in `Automatic` heaps with the heap texture size, programmable
     sample positions, texture views and Metal IO queues, and it carries
     them all; the IO queues are Metal's own over the bundle's device, with
     nothing on the wire. Apple3 waits on indirect command buffers. Each
     family's pieces are listed in `device_info.rs` (`APPLE_PIECES`), and the
     claim rises as each piece lands.
   - Mac1, Mac2, Common1 to Common3 and Metal3 keep the platform's answer for
     now, though their rows need some of the same pieces: argument encoders
     for Mac1, Mac2 and Metal3, and the heaps, sample positions and IO
     queues for each of them.
   - `architecture.name` is the host platform's: Metal's own on a Mac, and
     on the Vulkan platform `vulkan_` with the GPU's PCI vendor and device
     IDs and its driver's `pipelineCacheUUID`. The name keys binary archives,
     so it stays one GPU's from boot to boot and changes where compiled
     pipelines would (another GPU, driver or driver build). It names no Apple
     GPU.
   - Binary archives are the bundle's own (`RgfxBinaryArchive`): no compiled
     code crosses the wire, so an archive keeps the keys Metal looks each
     function's compile up by, per architecture, and writes them to a file of
     its own format. Its answers and errors are Metal's, measured on the M3
     Max (guest/bundle/tools/archive_probe.m), including the miss of
     `MTLPipelineOptionFailOnBinaryArchiveMiss`. Apple frameworks that load
     one at start (IconRendering, RenderBox) get a working archive, not the
     base class's, which needs a target architecture the device has none of.
   - Where the lane's own device can do less than the host GPU, it answers
     no (Metal4, ray tracing, function pointers, dynamic libraries and the
     rest). One table in `crates/reims-vgpu/src/graphics/device_info.rs`
     lists each such answer with what it waits on.
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
    - **Today:** a surface's GPU copy is one region block, holding every
      plane in plane order, that every process holding the surface shares
      (the kext's, found by plane 0's pages; "Processes name only their
      own"), so one process's GPU writes reach another's GPU with no copy, and
      a texture over one plane that reads another (a two-plane YCbCr texture
      over plane 0 reads plane 1's chroma) finds both in one buffer. Its
      `TEXTURE_FROM_IOSURFACE` carries the surface's facts
      (`RgfxSurfaceFacts`: OSType, planes, size, plane 1's place, YCbCr
      matrix), which the host's surface creator reads as Metal reads the
      surface. The block syncs from the pages before
      a commit where the surface's seed moved since any holder synced it, and
      back to the pages at the completion of each command buffer that writes
      it, since no lock callback reaches the bundle yet. The block's header
      holds the seed every holder agrees on (`guest/bundle/src/RgfxIOSurface.h`).
      Moving the write-back to the CPU's lock, through IOSurface's device
      caches, is what is left of this item.

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

**BAR2: the region.** A 64-bit prefetchable BAR over host memory (1 GiB),
mapped write-back by the guest. The device places the control rings at fixed
offsets; the guest places everything else. Its pages are committed on the host
as they are first touched, and the host imports for the GPU only what it reads:
the scanout's pages, and each block a `*_OVER_HOST_MEMORY` names.

1 GiB is the largest size an x86-64 macOS guest takes (measured on macOS 26,
q35, OVMF):
- macOS's PCI configurator lays the bus out again at boot. It places every
  64-bit prefetchable BAR below 4 GiB, never in the host bridge's 64-bit window,
  and it misreads a 4 GiB BAR as an I/O BAR.
- Below 4 GiB the window ends at 0xE000_0000, so a 2 GiB BAR cannot fit.
- A 1 GiB BAR needs the window to start at or below 0x4000_0000. q35 keeps
  2 GiB of the guest's RAM below 4 GiB when the guest has more than that, and
  then the BAR gets no address: the kext finds no BAR2 and IONDRVFramebuffer
  keeps the display.

So the host starts QEMU with `-machine q35,...,max-ram-below-4g=1G` wherever
the device is present. QEMU then keeps 1 GiB of RAM below 4 GiB and puts the
rest above, which leaves the guest's memory size unchanged, and the BAR lands
at 0x4000_0000. The device stays on `pcie.0`: a root port does not help.

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
addresses (`0x1_0000_0000 + s << 36`). The kext hands out its region blocks
(first fit, page granular). The process hands each create and destroy of its
objects (ids in its range) to the kext by a user-client call; the kext posts it
on the control ring and, for a pipeline create, hands the reflection back once
the host answers it. A queue's `QUEUE_CREATE` goes the same way, its rings
checked to lie in the process's blocks, and is answered once the host has the
rings. At the process's close or death the kext destroys its queues, then
closes its context (`CONTEXT_CLOSE`: the host drops its objects, then
answers once the context's command buffers have finished), then frees its
chunks of resource ids and, on that answer only, its blocks. A command
buffer is decoded once the host has carried out every control-ring message
published before its queue's thread took it.

**Texture resource ids.** Every process's textures share the host's one
device, whose texture ids end below 16381: Metal ends the process for a texture
placed at 16381 or more (measured on an M4 and an M3 Max), and metal2vulkan
imitates that. So ids are not split by slot. The kext keeps one pool for the
guest, 255 chunks of 64 ids (chunk `k` holds `1 + 64k` to `64 + 64k`, all below
16321), and hands a process a chunk when it asks (user-client selector 0x106),
posting the grant (`RESOURCE_ID_CHUNK`: context, chunk) on the control ring
before it returns, so the host has it before any create naming its ids.
The host reserves the pool's ids on its device as it makes the graphics device,
its first call on the device (`reserve_guest_texture_ids`,
`crates/reims-vgpu/src/graphics.rs`). Metal hands the id of a texture placed at
one out again to plain allocation, so without it the guest's textures would share
ids with the host's own (the window's drawables, the cursor), and a texture placed
past the table a fresh device has corrupts Metal's heap (placements from 10000 up
crashed some runs). The reservation keeps the ids from plain allocation and grows
the table to hold them (measured on an M3 Max, macOS 27; metal2vulkan imitates
both). Metal takes one only before the device's first texture, and once the device
has a queue it hands the pool's first range out twice, so the host checks it with
a few plain textures and logs an error where one takes a guest id. Were the
paravirt lane's device in the same process, on the same GPU, the graphics lane's
reservation would win: it is made as the device is, before any guest runs, while
the paravirt lane reserves its placed objects as each task starts
(`crates/reims-vgpu-core/src/resources/manager/reserve.rs`), and Metal takes only
a type's first reservation. That would not make the two safe together: both lanes'
guests place textures in the device's one table, at ids each picks alone. So a
process runs one lane's device on a GPU, and today a VM has one lane or the other.
The bundle keeps each chunk's free ids in one 64-bit mask, takes the lowest free id
of the first chunk taken that has one, asks for a chunk only when all of its are
in use, and gives an id back as its texture goes, after the texture's destroy
(below), so a later create with the id follows the destroy on ring 0.
The host names an IOSurface by every live texture over its plane 0, and shows
the latest. A present holds its own reference to the texture it shows, so a
destroy after the present leaves the window's frame drawable. A process keeps its chunks
until its close. With every chunk held, a texture create answers nil, as Metal
answers a texture it cannot make. The host carries out a texture create only
where its id lies in a chunk of its own context (by the object id's context);
any other id, 0 and the limit's included, is refused before any platform call,
as a refused create is: unanswered, and a command buffer naming the texture
fails. Sampler states and pipeline states take no guest-picked ids (the host
numbers them), and the bundle makes no indirect command buffers, so textures
are the only kind with chunks.

**Objects go with their wrappers.** Every host object the bundle makes goes
from the host as the app lets its wrapper go. A buffer, heap, texture, sampler,
depth-stencil state, render or compute pipeline, library, event, fence or shared
event posts its `OBJECT_DESTROY` from its dealloc. A queue ends through the kext
(user-client selector 0x104), which stops watching its completion ring and
posts the destroy. Command buffers ride their queues' own rings, so a destroy
must not reach the host before a buffer that names the object. A command buffer
keeps every object its encoders name, from the encode until its completion
handlers have run, or until it goes uncommitted, as Metal's with retained
references keeps what it encodes. One made with unretained references keeps them
too, though it answers `retainedReferences` NO, since the host must not drop an
object before a buffer that names it runs. A buffer whose completion never comes
holds back only what it names. Objects are released in the order they go. What an object lay over goes back only after its destroy: a Shared buffer's
or heap's region block, an IOSurface plane's block, a queue's ring blocks (its
completion ring read no more first), and a texture's resource id. The block's
`BLOCK_RELEASE` then finds nothing of the object on the host. A resource placed
in a heap keeps the heap alive (its `heap`), a command buffer keeps its queue,
and a function keeps its library. So a pipeline made from a function whose
library the app let go names a library the host still holds. The host makes the
pipeline from the library before the library's destroy, and the pipeline needs
it no more.

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
| 2 | `SET_SCANOUT` | display u32, width u32, height u32, stride u32, offset u64 into BAR2, the scanout inside `RGFX_SCANOUT_SPAN`'s room; width 0 blanks the display | `COMPLETE` |
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
(QEMU's VGA devices do the same). The GPU reads the buffer in place: the host
pages under the scanout are imported at its first draw, and again only when
the guest moves the scanout or changes its size, so no copy is made on the way
to the window, and no other page of the region is pinned for it. While vblank is
enabled, the host sends `VBLANK` at each refresh of the window's screen.

**Boot display.** The device's option ROM gives firmware a GOP framebuffer in
BAR2, and the kext takes the display over when it starts.

**QEMU's console.** `reims-graphics-pci`'s QEMU console carries input only: its
display callbacks are empty, so QEMU's own displays show nothing of the device,
and a QMP `screendump` of it is always an empty 640x480 image, whatever the
guest shows. The display shows only in the library's window. A check of what
the guest shows reads the scanout where `SET_SCANOUT` put it in the region.

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

**Guest input the host refuses.** Everything on the wire comes from the guest,
which the host does not trust, and no guest input ends QEMU. A message the host
cannot carry out (one shorter than its record, a span it does not cover, an
object it does not hold or of the wrong kind, the platform refusing the create)
is refused at its op, and nothing of it is made. A create the guest waits on (a
pipeline's reflection, a library create or lookup, a queue's create, a
context's close) is answered at that op: a refused `PIPELINE_REFLECTION`, or a
`COMPLETE` with the refusal's status, so no guest thread waits out the kext's
timeout. A refused command buffer completes with an error status, as a buffer
Metal could not run does: a stand-in on its queue, holding the buffer's event
waits and signals alone, finishes in the queue's order, and its completion
posts the values the buffer signalled its shared events to, then the buffer's
internal-error `CB_COMPLETE`. So a buffer that names a pipeline the host
refused, or anything else it does not hold, fails after the buffers committed
before it, and nothing waiting for one of its events, a later buffer on the
GPU or a guest thread, waits for good. Where there is no stand-in (a message
too short to name its queue, a queue the process does not hold, a context
whose work is refused), the error status is posted at once. Nothing in
this lane ends the offending context or its process on malformed input. The
host runs each op in a refusal scope (`reims_vgpu_core::raise::refusals`): a
death at a platform call, where the platform refuses a create or an encode and
the paravirt device ends the process, unwinds to the op's scope instead, and the
op is refused there, logged once at WARN with the platform's reason. A command
buffer refused mid-stream has its open encoder ended and is never committed.
Work an op registers to run on another thread takes the op's scope with it
(`refusals::carry`): a command buffer's completed handler runs the platform's
read of the buffer, and itself, in the scope of the decode that added it. A
death in that read (a failure no Metal answer matches, which the Vulkan
platform reads on the runtime's thread) fails the command buffer, whose
`CB_COMPLETE` is an internal error, and fails its context: the host refuses
the context's later command buffers, creates (its queues' and samplers'
among them), library lookups and heap-size and argument-layout queries at
once, carries out its destroys, and forgets the failure at its close. A release the
platform dies at (at a destroy, a context's close, or the display's swap of
the texture it shows) is refused alone, and the destroy or the close is
carried out whole. So is each of the Vulkan platform's releases of what a
command buffer's commands used, and each change it makes, as the command
buffer ends on the runtime's thread. What dies elsewhere off the op's thread
still ends the process: a notice the runtime's scheduler thread announces at
a host step, which names no command buffer, the runtime's own releases after
the platform's handlers ran, and the window's draw. This
is the lane's own rule: the paravirt device's rule of ending the process at the
op (AGENTS.md, the contract) is Apple's framework's behaviour, which that
device keeps.

**Processes name only their own.** A process names only its own objects and
its own memory. The accelerator kext checks, before it posts a lifecycle
message, that the object the message creates or destroys is in the process's
id range, and that the region the message names lies in the process's own
blocks: a queue's rings, a heap's or a buffer's block over host memory, and a
library chunk's bytes and ack slot. The host checks these spans too, and also
the span a `COMMAND_BUFFER_IN_REGION` on a queue's ring names, which never
passes the kext. Each span must lie in one block the sender holds. The kext
posts a block's `BLOCK_GRANT` (context, offset, size) on ring 0 before its
allocation returns, keeping the block pinned until then so no free overtakes
it. The host keeps the grant until the block's `BLOCK_RELEASE` or the context's
`CONTEXT_CLOSE`, and copies a buffer out of a block only while the grant
stands. An IOSurface's shared block (§14) is held by every process that
holds the surface: the kext finds it by the physical pages of the surface's
plane 0, which a process presents through its own CPU mapping of the plane
(user-client selector 0x109), so only a holder of the surface can reach the
block, whatever id it claims. The kext wires those pages while the process holds the block,
posts `BLOCK_SHARE` (context, offset, size) for each holder before the process
learns of the block, and `BLOCK_UNSHARE` as a holder other than the last lets
it go (selector 0x10a, or its close). The last holder's let-go frees the block
as a free does, and its `BLOCK_RELEASE` ends every hold.

**Processes map only their own.** No process maps the region whole: it holds
every process's blocks, the control rings and the device page. Each
connection has a window, a range of the process the region's size that the
kext makes at the context's open (selector 0x100 answers its base and size).
The window is memory of the connection's own, mapped read-only, and holds no
page until one is read, which then reads zeros. Region offset `o` of what the
process is given lies at the base plus `o`, so the bundle's pointers stay
base plus offset. The kext maps in, over the window at the span's own offset,
only the process's blocks, the shared blocks of the surface planes it holds,
and the scanout it is granted. It maps a span in before the process learns of
it. Before the span can go to anyone else, it maps the window's own memory
back over the span, read-only: before a block's free marks it freed, and when
a hold ends, a scanout grant goes or a mode change moves the scanout. At the
close the whole window goes, every span in it, before the context's blocks
are freed. Each span is whole pages past the region's head, so no mapped page
holds a byte of anything else. The device info shares its page with the
control completion ring's tail, so the kext reads it for the process (selector
0x10b), and the region's own map (`kRgfxAccelMemoryRegion`) is refused. One
difference from Apple's: a process that touches a block after releasing it
reads zeros, and faults only on a write, where Apple's faults on both. Only a
guest that uses memory it released sees it. A mapping that fails fails its
op; a span that cannot be taken back closes the window, so the process
reaches nothing of the region from then on. A placed mapping is IOKit's
overwrite mapping, which IOKit never takes down itself (read from the public
xnu source, `IOMemoryDescriptor::doUnmap`), so only another mapping over it
takes a span back; vm_protect is not a KPI a third-party kext may link.

**The scanout is its owner's.** The scanout holds the display's pixels, so
only the process that drives the display reaches it: the task that holds the
framebuffer's server connection (`kIOFBServerConnectType`), WindowServer. The
kext grants the scanout (selector 0x108), locks an IOSurface with no backing
of its own over it, and gives out the framebuffer's VRAM range (which
IOFramebuffer maps on its server connection, and on its shared connection into
any process of the console's user) to that task and the kernel alone. Any
other process gets `kIOReturnNotPermitted`, or no memory for the VRAM range,
and nothing is mapped. The server connection is the test because IOFramebuffer
opens it for one task at a time, and the task that holds it maps the scanout
through it already (the system aperture), so the rule gives no process what
IOFramebuffer does not. An open in flight counts too, since IOFramebuffer sets
the display up inside it. A new connection's task replaces the last one, and
every other task's hold of the scanout is taken back then: each grant, its
span mapped back out of the context's window, and each lock's mapping,
unmapped from the task. What IOFramebuffer itself mapped for a task (the VRAM
range on its connections) is IOFramebuffer's, and goes only with its
connection.

Command buffers do not pass through the kext, so the host checks each
object a message names: it must be
of the message's context, which is the context of the object a ring-0 message
creates or destroys, or the context of the queue whose ring carried the
command buffer. A command buffer runs only on the queue whose ring carried it,
and a queue's ring carries nothing but command buffers: a message of any other
type there is refused, since it would pass the kext's checks by.
A create the kext posted but whose answer did not come in time is destroyed by
the kext right behind it, so a retry under a new id leaves nothing behind.

**A block outlives what lies in it.** A region block is given out again only
once the host has said that nothing of it is in use. A process's free marks the
block freed: from then on no message of the process can name it, and no one can
take it. The free returns at once, as Metal's releases never block. A thread
call of the kext's own then posts the block's `BLOCK_RELEASE` on ring 0, behind
every message the process posted before the free, and waits for the answer.
The host refuses the release at once where an object it holds lies in the
block: a buffer or heap over host memory, or a buffer, texture or heap placed
in one, which keeps its parent's span after the parent's destroy. It also
refuses one where a live queue's ring lies in the block. Otherwise it answers
OK once every command buffer the context committed before the release has run
its completion handler. Command buffers committed later do not hold it up. On
OK the block goes back to the pool; on a refusal, or no answer in time, it
stays out of use until its process's close. A block a create in flight names
is pinned from the kext's check until the create is answered, and a free of it
meanwhile fails busy, so no block changes hands between the check and the
host's carrying out of the message. A context's close is answered under the
same rule, after its objects are dropped and its queues' threads are past
their rings, and the kext frees the process's blocks only on that answer. A
close that is not answered keeps them from use for good.

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
single consumer, so their `consumer_idle` handshakes never race. A create carries a priority, which the bundle sends as 0 for every
queue and the host does not read: each queue is the platform's plain `newCommandQueue`.

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
| memory | `HEAP_OVER_HOST_MEMORY`, `HEAP_CREATE`, `BUFFER_CREATE`, `BUFFER_OVER_HOST_MEMORY`, `BUFFER_AT_OFFSET`, `TEXTURE_CREATE`, `TEXTURE_FROM_BUFFER`, `TEXTURE_FROM_IOSURFACE`, `MANAGED_TEXTURE_CREATE`, `*_DESTROY` | `new_heap_over_host_memory`, `new_heap`, `new_buffer`, `Heap::new_buffer_at_offset`, `new_texture`, `Buffer::new_texture`, drop |
| Managed ranges | `BUFFER_DID_MODIFY_RANGE`, `TEXTURE_REPLACE_REGION` | `Buffer::did_modify_range`, `Texture::replace_region` |
| heap sizes | `HEAP_TEXTURE_SIZE_AND_ALIGN` (answered `SIZE_AND_ALIGN` on the control completion ring; makes nothing) | `heap_texture_size_and_align` |
| state | `SAMPLER_CREATE`, `DEPTH_STENCIL_CREATE`, `*_DESTROY` | `new_sampler_state`, `new_depth_stencil_state` |
| shaders | `LIBRARY_LOOKUP` (the content key of the `.metallib`'s AIR slice: a hit binds the library the device made for any process, and no bytes cross), `LIBRARY_CREATE` and `LIBRARY_FROM_REGION` (the slice's bytes forwarded unchanged on a miss, answered once made), `RENDER_PIPELINE_CREATE`, `COMPUTE_PIPELINE_CREATE`, `*_DESTROY` | the device's library cache, `new_library_with_data`, `new_render_pipeline_state_full` and `new_compute_pipeline_state_full` (the whole descriptor the guest set: vertex or stage-input descriptor, every colour attachment, each stage's linked functions, those made with options or a specialized name by `Library::new_function_with_options`; the Vulkan platform refuses a specialized name, which its runtime's function descriptor lacks) |
| sync | `EVENT_CREATE`, `SHARED_EVENT_CREATE`, `FENCE_CREATE`, `*_DESTROY` | `new_event`, `new_shared_event`, `new_fence` |
| queue | `QUEUE_CREATE` (priority, region offset, vector), `QUEUE_DESTROY` | `new_command_queue` |

`*_OVER_HOST_MEMORY` map guest region blocks (Shared) and the host builds a
placement heap/buffer over them.

The host makes an `Automatic` heap a `Placement` heap: the guest places what
the heap makes itself, as Metal's Automatic heap places it (lowest free offset
at the resource's alignment; `guest/bundle/src/RgfxHeapRanges.hpp`), a buffer
at its length and a texture at the size `HEAP_TEXTURE_SIZE_AND_ALIGN` answers,
and sends each as a placement at that offset. The heap's `usedSize` and
`maxAvailableSizeWithAlignment:` answer from those placements.

**No-copy buffers.** `newBufferWithBytesNoCopy:` over bytes in a block of the
process's lies over them in place (`BUFFER_OVER_HOST_MEMORY` at their offset).
Metal makes one over any range the process maps readable, at any alignment:
malloc'd, `vm_allocate`d, read-only, or a file mapping (measured on an M3 Max,
16 KiB and 4 KiB pages). Over such memory the buffer lies over a block of its
own, and `contents` stays the app's pointer, since the host never reaches guest
RAM (Decided 5). The block syncs with the memory as the IOSurface copies do
(Decided 14): from the memory before the commit of each command buffer that uses
the buffer, and back to it as one that may write it completes, before its
completion is reported. Each side's bytes that moved since the two last agreed go
to the other, byte by byte, so CPU and GPU writes beside each other both stand
(`guest/bundle/src/RgfxMemoryShadow.hpp`). A GPU write shows in the memory at
its command buffer's completion, not as it runs, and memory mapped read-only is
never written back.

**Managed storage** (Decided 4). A Managed buffer's or texture's CPU copy is a
region block of its process's, which the guest maps, and the host decides the
form from the platform's answer of whether the GPU shares the CPU's memory:
- **Shared memory:** one copy. The buffer is a no-copy buffer over its block,
  and a one-level 2D texture of a linear format a texture over its block's
  Shared buffer; the range messages are checked and do nothing more.
- **Otherwise:** two copies. The object the command buffers name is the
  platform's own Managed buffer (at the guest's address) or texture, and the
  block sits beside it as a Shared buffer.
  - `BUFFER_DID_MODIFY_RANGE` copies the named range from the block to the
    platform buffer's host copy, and the platform's `didModifyRange:` copies it
    to the GPU's before any command of its next submission.
  - `TEXTURE_REPLACE_REGION` is the platform's `replaceRegion:` from the block,
    done before the call returns.
  - Ring 0 is carried out before any later command buffer of the process, on
    any queue, so a range is ordered before every command buffer committed after
    it, as Metal promises.
  - A blit encoder's `synchronizeResource:` and `synchronizeTexture:` become
    copies of the GPU's copy into the block at that point of the encoder.
- A buffer comes from `BUFFER_OVER_HOST_MEMORY` with Managed options; a texture
  from `MANAGED_TEXTURE_CREATE`, which names the block's Shared buffer and the
  texel block of its format (bytes, width, height). Both sides compute the
  copy's linear layout from it (`crates/reims-graphics/src/managed.rs`,
  `guest/bundle/src/RgfxManagedLayout.hpp`): levels in order, then slices,
  images and rows of blocks, each row rounded up to 256 bytes.
- In the guest, `replaceRegion:` and `getBytes:` are memcpys of the block, and
  `replaceRegion:` posts its region; neither waits for the host. A format the
  layout has no block for (depth, stencil, PVRTC, three-component) keeps no CPU
  copy.
- Until the device's identity answers carry `hasUnifiedMemory`, the host takes
  the two copies on every GPU, which hold Metal's contract on any of them.

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
- `SIGNAL_EVENT` / `WAIT_EVENT` (event `ObjId`, value), between encoders, as
  Metal's `encodeSignalEvent:value:` and `encodeWaitForEvent:value:` are →
  `encode_signal` / `encode_wait`, for an `MTLEvent` and an `MTLSharedEvent`
  alike (an event op with an encoder open, or naming no event, faults the
  buffer). The end of the message closes it: the host commits and registers an
  `add_completed_handler` that writes the completion record.

A command buffer too large for half its queue's submission ring goes out of
line: the bundle writes the whole `COMMAND_BUFFER` message into a block of the
process's own and posts a `COMMAND_BUFFER_IN_REGION` (region offset and
length) on the ring. The queue's service thread copies the message out of the
block once, as it does a ring's, with the reference's seqno, and the bundle
gives the block back when the buffer completes. Metal sets no maximum size, so
nothing is refused for size. A queue holds `maxCommandBufferCount` buffers
(64 by default) alive and not completed. `-commandBuffer` blocks past that, as
Metal's does, which keeps the ring from filling in the common case. A commit
that still finds the ring full rings the doorbell and sleeps 20 µs, doubling
to 1 ms, between looks. Each buffer's completion is expected before the host
can see it, under the ring's lock.

Metal splits one command buffer into several kernel submissions (a render pass;
a run of blits or of computes with no event between) — m2v's measured
granularity. The host preserves it so IOGPU error reporting lines up.

**Completion** (per-queue completion ring). A `CB_COMPLETE` record: the
command-buffer `ObjId`, status, GPU start/end time, and on error the
`CommandBufferError` the host maps 1:1 from the IOGPU code the runtime returns
(page fault, hang/timeout, recovery victim, access revoked, invalid resource —
the map m2v gave). The ring's armed/interrupt handshake is the stage-2 one.

**Shared events.** The host's event is the one command buffers wait on and
signal; the bundle keeps the value as the process knows it. It only rises, as
Metal's does (measured on the M3 Max: a CPU set or a GPU signal to a lower value
leaves it). A CPU `setSignaledValue:` posts `SHARED_EVENT_SET_VALUE` on the
control ring, which sets the host's (so its GPU waits are released), and raises
the bundle's value. A command buffer's signal of a shared event reaches the
process as `EVENT_SIGNALLED` (`ObjId`, value), which the host posts on the
buffer's queue completion ring once the buffer finishes, ahead of its
`CB_COMPLETE`, whatever its outcome. The queue's reader hands it to the
device's, where the process's shared events listen. `notifyListener:` blocks
run on their listener's queue with the value they were registered at, at once
where the value has reached it, as Metal's do (measured). A shared event made at
another value is set to it on the host after its create.

The difference from Metal: Metal's signal lands where the GPU reaches it in the
buffer, and its CPU value and listeners move then; here the process sees it only
once the whole buffer has finished. GPU waits are not affected: another buffer's
`WAIT_EVENT` waits on the host's event, which the host's GPU signals at the
point Metal's would. A guest thread that waits on the CPU for a signal from the
middle of a command buffer (`waitUntilSignaledValue:`, a listener) sees it as
late as the buffer's end. That is never a hang, since the buffer's end comes,
but it is later where the buffer runs on after its signal: work it would have
overlapped with the rest of the buffer starts after it instead. A wait for a
signal that only comes after the waiting thread does something else is the case
it changes: a thread that waits for a mid-buffer signal before it sets a value
the same buffer waits for later deadlocks here, where Metal's would not.

A command buffer that waits on an event is watched: one not finished 5 s after
its commit is named in the log at WARN, with each event and value it waits for,
and again at INFO once it finishes (at most 32 per device). It is diagnostics
only: Metal waits without end for a value nothing reaches (measured: the buffer
stays `Committed`), and so does the host.

**Who takes the completions.** Every queue raises the device's one vector,
the control ring's. The accelerator kext maps each queue's completion-ring
header into the kernel at its `QUEUE_CREATE`. The framebuffer's work loop
services the vector, and then the accelerator scans the watched rings: a ring
whose `producer` moved since the last scan, and still holds records its process
has not taken, wakes that process. The scan takes only the watch table's lock,
which no submit path takes. Each process has one completion thread in the bundle.
It sleeps in the kext (user-client selector `0x107`, an interruptible wait on a
per-client wake count it passes back), and each pass it drains every queue's
ring, arms it, fences and looks again. A completion posted while it works shows
in the count or in its last look, so none is missed and nothing polls. The
thread runs each command buffer's scheduled then completed handlers in ring
order, whether or not anything waits. `waitUntilCompleted` waits for the
thread to have run them. It takes the ring itself only inside a handler, or
once the thread has ended (its context closed).

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
  takes the AIR slice, asks the host for it by its content key
  (`LIBRARY_LOOKUP`), and forwards the slice's bytes over the wire only where
  the host holds none (`LIBRARY_CREATE`), which the host compiles.
- **Core Image's link.** Core Image makes its kernel libraries (file type 1,
  whose kernels are extern functions of `functionType` 4, left out of
  `functionNames`) through `newLibraryWith*`, takes each kernel and its own
  helpers with the private `newExternFunctionWithName:`, and links them on the
  CPU in Metal's base device class (`newLibraryWithImageFilterFunctionsSPI:…`,
  which calls `newLibraryWithCIFilters:…` on the device). The link reads
  Metal-internal library data from each function, which no function of ours
  holds. So each library of ours keeps the bytes it was made from and makes, on
  first use, a shadow: Metal's own library over them (the base class's
  `newLibraryWithData:`), whose extern functions ours hold. The device's link
  overrides hand Metal the shadows, and make the library Metal links (the bytes
  it answers as `libraryDataContents`: one kernel, `ciKernelMain`) the device's
  own through `newLibraryWithData:`, so the kernel's pipeline is made on the
  host like any other (measured: macOS 27, and 26.6.1 in the guest;
  `tools/ci_probe.m` renders through it).
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
holds BAR 2 to map each process's part of it in; it touches no pixels and no
control ring (the framebuffer owns those).

`RgfxAcceleratorUserClient`
(`guest/kext/src/RgfxAcceleratorUserClient.{hpp,cpp}`) is **our own** transport,
not Apple's IOGPU user-client ABI: the context's window ("Processes map only
their own"), where the bundle's `RgfxRegion` / `RgfxRingWriter` write into real
device memory, the device info (selector 0x10b), and
`externalMethod(kRgfxAccelMethodDoorbell)`, which writes a ring id to
`RGFX_REG_DOORBELL`. These ids/selectors are defined in `RgfxAccelerator.hpp`;
the bundle's wire handshake opens them with `IOServiceOpen` /
`IOConnectCallStructMethod` / `IOConnectCallScalarMethod`.

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
- **QEMU's console shows nothing of the display** (above, "QEMU's console"):
  it could show the scanout, so QEMU's displays and `screendump` see what the
  guest shows.
- **Owner calls**, each with a recommendation once the evidence is in:
  - whether host device addresses are ever exposed.
