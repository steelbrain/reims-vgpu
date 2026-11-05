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
   - IOSurfaces are the one place the host reaches guest RAM: where the
     device has `RGFX_FEATURE_SURFACE_PAGES`, the host maps a surface's own
     guest pages for its GPU in place (Decided 14, "Today"), from the guest
     RAM's map QEMU gives it, as the apple-gfx lane maps guest memory.
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
     Metal does: the tables' first Apple column is Apple2.
   - No Apple family is claimed until the lane carries every piece of
     Apple7's rows and of the families' before it (2026-11-01). A Mac's GPU
     answers no Apple family (Intel, AMD) or Apple1 to Apple7 at least
     (Apple silicon), and the guest's frameworks choose their paths from
     those answers together. With Apple1 and Apple2 answered yes and Apple3
     no, RenderBox made its passes' extra attachments memoryless and used
     them in a way Metal fails: every command buffer it drew an app icon in
     failed with `MTLCommandBufferErrorInvalidResource`, and the Dock's icons
     were blank (an x86-64 macOS 26 guest). With no Apple family, as an
     Intel or AMD Mac answers, they draw.
   - Today the lane carries Apple1's and Apple2's pieces (argument encoders,
     placement in `Automatic` heaps with the heap texture size, programmable
     sample positions, texture views and Metal IO queues; the IO queues are
     Metal's own over the bundle's device, with nothing on the wire) and
     binary archives, so it claims no Apple family. Apple3 waits on indirect
     command buffers. Each family's pieces are listed in `device_info.rs`
     (`APPLE_PIECES`), and the claim rises to Apple7 when the last of its
     pieces lands, then family by family.
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
    - The loader binds our kext from a table generated from one build of
      it: each bridge site (a GOT or vtable pointer, a call into
      IOGraphicsFamily, IOAcceleratorFamily or AppleGraphicsDeviceControl) by
      its offset in the binary. Swapping a new ReimsGraphics binary into an
      image's set therefore binds only while those sites stay put, and no
      script here regenerates the table. A change that grows the kext's
      `__const` or `__data` (a new initialized global, or a member of one)
      moves every pointer after it. A change in an object linked before the
      provider calls (`ReimsGraphics.o`, `RgfxAccelerator.o`) moves those
      calls. Either needs the table generated again; a swap fails to bind. To
      spot it, the loader logs `apvload: ... ReimsGraphics: got +0x...
      <symbol> failed the placeholder check (old 0x...)`, then
      `publishIndependent: ReimsGraphics not ready (bind failed)`, and the
      guest boots with no framebuffer or Metal device. Before a swap, check
      that `otool -rv` lists every relocation at or past `__DATA`, and every
      provider call, at the same offset and symbol as the binary the table
      was made from. The swap also keeps the set's own `Info.plist`, whose
      `OSBundleLibraries` name the bridge shims and which carries no
      personalities. The build's own `Info.plist` makes OpenCore refuse the
      kext ("Prelinked injection ReimsGraphics.kext ... Invalid Parameter"
      on the serial log).
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
    - **Today:** the GPU works on a surface's own guest pages in place, and
      copies them only where the host cannot use them. Where the device has
      `RGFX_FEATURE_SURFACE_PAGES` and the host's linear textures can lie
      over every plane (the base is a page's, each plane's offset from it and
      its rows are multiples of the host's linear texture alignment, every
      plane lies in the allocation), the bundle asks the kext for the surface's shared
      block in place. The kext wires the surface's whole allocation, writes
      its page runs into the block past the part the processes map, and
      hands them to the host (`SURFACE_PAGES`, answered), which maps them as
      one view. Each process's buffer lies over the view
      (`BUFFER_OVER_SURFACE_PAGES`), and each texture over its plane's own
      offset and rows. The block holds only the header and the writers, and
      nothing is copied or synced: the GPU and every CPU use the same
      pages, so such a surface has no GPU copy to keep in step. A surface
      whose layout misses the host's alignment, a device
      without the feature, or pages the host refuses (not whole pages of
      guest RAM, or a view it cannot map) keep the copy below; so does every
      other holder of that surface, since the kext gives a surface's blocks
      in one mode.
    - **The copy:** a surface's GPU copy is one region block, holding every
      plane in plane order, that every process holding the surface shares
      (the kext's, found by plane 0's pages; "Processes name only their
      own"), so one process's GPU writes reach another's GPU with no copy, and
      a texture over one plane that reads another (a two-plane YCbCr texture
      over plane 0 reads plane 1's chroma) finds both in one buffer. In
      either mode, its `TEXTURE_FROM_IOSURFACE` carries the surface's facts
      (`RgfxSurfaceFacts`: OSType, planes, size, plane 1's place, YCbCr
      matrix), which the host's surface creator reads as Metal reads the
      surface. The copy syncs from the pages before
      a commit where the surface's seed moved since any holder synced it, and
      back to the pages, with no lock of the surface, at the completion of
      each command buffer that writes it. The block's header holds the seed
      every holder agrees on (`guest/bundle/src/RgfxIOSurface.h`). Moving the
      copy's write-back to the CPU's lock (the device cache's op 0) is what is
      left of this item.
    - **CPU locks** wait for the GPU's pending writes, as Metal's do on the
      x86-64 Macs (measured on a MacBook Pro's AMD Radeon Pro 5300M and Intel
      UHD 630 under macOS 26.7, `guest/bundle/tools/lock_probe.m`). Where the
      Macs measured agree, their answer is the contract and is kept exactly;
      where they differ, the docs promise neither answer and the cheapest one
      within them is taken:
      - a read-only or read-write lock of a surface, in any process, returns
        once every buffer that writes the surface has finished and its
        results are in the pages. The AMD, the Intel and the M3 Max agree:
        contract, kept exactly;
      - a pending buffer that only reads the surface holds no lock. The AMD
        and the Intel agree; the M3 Max's read-write lock waits for it.
        Divergent, so the cheaper rule, which tracks writes only. A buffer
        writes a surface it draws to, a blit writes to, it uses with Write
        usage, or binds where a draw's or dispatch's function declares a
        `write` or `read_write` texture: the bundle reads each entry point's
        arguments from the AIR metadata of its bitcode in the `.metallib`
        (`guest/bundle/src/RgfxAirArguments.hpp`), since the host gives no
        binding reflection. Those declared under function constants are the
        function's whatever the constants' values. Where the arguments do
        not read (a binding of a kind the reader does not know among them),
        or a written texture's binding index is an expression of function
        constants, a bound texture counts as written where its usage has
        ShaderWrite or is unknown. A surface used without Write usage counts
        as written at a draw or dispatch whose function's arguments do not
        read or whose argument buffer may write a texture: Metal takes
        useResource's usage for the app's promise, but a discrete Mac's
        driver lands a write the promise left out. Narrowing may only drop a
        write the function cannot make. The same rule decides the copy's
        write-back: a buffer writes a copy back to the pages only where it
        writes the surface, never for a surface it only reads;
      - the committer's own lock waits from the buffer's submission on. At
        the commit itself the AMD's returns at once, a race. Loose: ours
        waits from the commit on;
      - a read of the pages with no lock after the buffer completes sees the
        GPU's pixels on ours and stale ones on the AMD and the Intel. The
        docs ask for a lock before the CPU reads, so either holds.

      The kext drives IOSurface's
      device caches as those Macs' drivers do (measured with fbt on the AMD's
      IOAcceleratorFamily2 and the guest's IOGPU; the kext's
      `RgfxIOSurfaceKPI.hpp`). At the first write a process names of a
      surface's shared block, the kext looks the surface up by the
      IOSurfaceID the process gives and takes it only where its memory holds
      the block's first page, makes its device cache for the accelerator with
      the kext's control function, and keeps both with the block. Each write
      named marks the cache current, which moves the surface's seed once, as
      Metal's submission does, and has IOSurface call the control function in
      every CPU lock of the surface: op 2 waits while a write is pending, and
      op 3 releases the reference IOSurface took on the cache for the lock,
      or for a purgeable change (ops 6 and 7), as the AMD's driver does. A
      write retires at the bundle's retire call after the buffer's completion
      and write-back, or, for a surface drawn in place, once the queue's
      completion ring's `completed_seqno` reaches the buffer, whichever comes
      first. A lock still waiting after 2 s goes on with an "RGFX: WARN" line
      (`guest/kext/src/RgfxAcceleratorLocks.cpp`).
    - **The device cache:** where the host's GPU does not share the CPU's
      memory, the device has `RGFX_FEATURE_SURFACE_CACHE`, and a block the
      kext marks cached (`RGFX_SURFACE_PAGES_CACHED`) gets a device cache on
      the host: a Private buffer of the surface's bytes in the pages' layout,
      which every buffer over the block is, so every texture over the
      surface, of any shape, plane or format, lies over the same bytes. Each
      command buffer names the cached surfaces it uses
      (`RGFX_OP_SURFACE_USE`) with the kext's count of the surface's
      read-write locks when it was committed; the host fills the cache from
      the pages before the first buffer of a newer count, on its own transfer
      queue, after every buffer that used the cache before. The pages are
      written from the cache only at `SURFACE_DOWNLOAD`, which the kext asks
      at a CPU lock that follows a GPU write, and at the block's
      `BLOCK_RELEASE`, and only where a buffer that may write the surface was
      committed since the last copy (`reims-vgpu`'s
      `graphics/decode/surface_cache.rs`). The one exception is a decoder's
      output the CPU reads back: once a download copied a write that was the
      only one since the cache's last copy, the next command buffer that
      holds a video encoder and is again the surface's only writer is
      followed on its own queue by a copy of the host's into the pages, which
      starts as the decode ends, and the lock's download is answered as that
      copy lands; a write no download followed stops it. A write the host's
      own devices make into the pages (a video device's frame, in a CPU lend
      or by a device of the host) moves no count: the host watches each
      cache's pages, and such a write has the cache's next use fill it
      whatever the count, and keeps a download or a release before that from
      copying the cache over the write (`reims-vgpu`'s `device_writes.rs`).
      Each choice, by what real GPUs show (M3, AMD and Intel Macs):
      - *Contract, where all agree:* a CPU lock after a GPU write sees it,
        and a GPU use after a read-write lock sees the CPU's writes; the
        seed moves at a read-write lock (not at its unlock), not at a
        read-only one, and by one at a GPU write's commit (the kext's).
      - *Loose, where they diverge, the cheapest chosen:* a CPU write without
        a read-write lock reaches the GPU only at the next fill, as on a GPU
        of its own memory (the M3 sees it at once); a command buffer
        committed before a CPU write and still running reads what it read,
        since a fill waits for it; a buffer of an older count taken after a
        fill of a newer one (two queues' buffers committed apart and taken in
        the other order) reads the newer bytes; a download of a cache nothing
        wrote since its last copy copies nothing; and a host whose GPU shares
        the CPU's memory keeps no cache, its surfaces in place as above.
      - *Which surfaces are cached:* the guest's choice, and the host's
        invalidation. The bundle asks every block in place cached, of any
        planes and format, as a GPU of its own memory keeps a copy of each
        surface it uses: the windows, layers and drawables the guest's
        compositors and apps draw, and video frames, which the GPU writes as
        a decode's output or a conversion's destination. Measured on the RTX
        5080: VideoToolbox's 10-to-8-bit conversion of a 1080p frame into a
        420v surface not cached takes 4.8 ms of the GPU, its writes crossing
        the bus into the pages. The outputs of the bundle's own video driver
        ask no cache where the host lands no picture in one (no
        `RGFX_VIDEO_CAPS_OUTPUT_CACHED`, item 16), its decode writing their
        pages: their textures ask under `RgfxAskUncachedScope`. Correctness
        does not rest on that choice:
        a surface a host-side device writes (the M2 scaler, the VideoToolbox
        decoder) may be cached all the same, and the host invalidates the
        cache of a block whose pages such a device wrote, so the next
        `SURFACE_USE` fills it from the pages and a download copies nothing.
        The device says `RGFX_FEATURE_SURFACE_CACHE` only with that
        invalidation in place.
      - *The guest's side:* the kext makes a cached block's IOSurface device
        cache with the block, and its control function, at op 2 of a CPU
        lock once the GPU's writes are done, asks `SURFACE_DOWNLOAD` where
        the GPU wrote the surface since the pages last took the cache, and
        waits for the answer, one lock of a surface asking at a time. A
        read-write lock then moves the surface's CPU generation, the count
        the kext writes into the block's header for the processes. IOSurface
        calls op 2 at every read-write lock of a surface with a device cache,
        current or not, and at a read-only lock only where a GPU write made
        the cache current (`RgfxIOSurfaceKPI.hpp`), so the count misses no
        read-write lock and a read-only lock after a GPU write downloads.
        Once the block's last holder lets it go, the block sleeps rather
        than going: dormant, it keeps the host's cache, its generation and
        its watch of device writes, the kext keeping the surface and its
        device cache retained and the surface's memory wired, so its
        IOSurfaceID and its cache are no other surface's meanwhile. The next
        ask of the surface, which IOSurface answers with the same device
        cache, holds the block again (`BLOCK_SHARE` with no `SURFACE_PAGES`),
        and the cache is used again rather than made and filled anew, as a
        GPU of its own memory keeps a surface's copy past the textures over
        it: in a window drag 98% of the blocks asked were such a surface's,
        most back within 100 ms. A dormant block is let go on the release
        thread, never in IOSurface's call: at op 5 (a client of the surface
        going, or the surface made nonvolatile after empty); at op 7 to
        empty, and a surface made empty while held does not sleep; at its
        sleep where the surface has no client left, its last gone while the
        block was held (IOSurface's client count, read once there); past
        1024 dormant blocks or an eighth of the guest's memory in their
        surfaces' wired bytes, and where the table or the region has no room
        for a new block, those of surfaces made volatile first (IOSurfaceRef.h
        lets a volatile surface's contents go), then the oldest; and where an
        ask of the pages names another cache or the other mode. Made
        nonvolatile again (op 6, or op 7), a surface is neither volatile nor
        empty for this. A let-go block's `BLOCK_RELEASE` copies the cache
        into the pages where a buffer wrote it since the last copy, and a
        lock waits for the answer; the kext releases the device cache, then
        the surface, then unwires the memory. Where the surface has no
        client left as the release is posted, the release says so
        (`RGFX_BLOCK_RELEASE_NO_CLIENT`), and the host copies all the same,
        counting it: a mach port or an XPC object holds a surface with no
        client, and a process that looks the surface up from one reads its
        pages (a Dock icon showed another app's icon, the bytes its pages
        held before the GPU drew it, while the host dropped such a cache
        uncopied). The kext logs the dormant
        blocks' count and the let-gos by cause. Op 6, which IOSurface calls
        as a surface turns nonvolatile again, is answered as the AMD's driver
        does (a byte set to 0xff), so IOSurface answers the old state
        volatile, after volatile and after empty alike, the contents kept,
        which IOSurfaceRef.h's old states allow. A present of a cached
        surface carries its generation. In the
        bundle, a commit names each cached surface its buffer binds,
        attaches, blits or declares it uses, with the generation the header
        holds and whether the buffer writes it, before the first encoder;
        and a texture's own CPU access (`getBytes`, `replaceRegion`) locks
        the surface as a CPU access does (`guest/kext/src/RgfxSurfaceCaches.hpp`,
        `guest/bundle/src/RgfxIOSurface.h`).
      - *Deferred pages:* a fresh surface only the GPU touches has no guest
        pages until a CPU lock needs them, as on a GPU of its own memory (the
        AMD 5300M's footprint grew 0 KiB from a 3840x2054 surface's first GPU
        use). Where the kext took `RGFX_FEATURE_SURFACE_DEFERRED`, and no
        IOSurface mapper device (`IOSurfaceParavirtMapperService`) is matched,
        a cacheable ask in place of a surface that reads fresh (its seed 0,
        none of its pages resident) defers its block: the kext makes the
        device cache and sets its control function first, then reads the seed
        and the resident pages under a read-only `lockSurface`, which waits
        out any lock calling the control functions, so a read-write lock
        either moved the seed before the read or calls the control function
        after it (`RgfxIOSurfaceKPI.hpp`). Found fresh, its `SURFACE_PAGES`
        carries `DEFERRED` and no runs, the block keeping room for one run a
        page of the whole allocation, every plane's; the host makes the cache
        as zeros. Not fresh, or refused, the kext names the pages at once, as
        for any block. A deferred block is found by its surface, never by
        pages, and its holders wire none. Its first read-write lock, or a
        read-only one after a GPU write, wires the pages and names them
        (`ATTACH`) before it goes on, one lock at a time, under no lock of the
        contexts'; a read-only lock of a surface the GPU never wrote names
        none, a fresh page reading as the zeros the cache holds. A dormant
        deferred block stays at a client's going (op 5) while others are
        left. Its release names its pages first where the GPU wrote it and
        the surface is retained past the kext's own three references (the
        table's, the device cache's, and the release thread's), so the
        release copies the cache into them; else nothing reads them. Deferred
        blocks fit the host's budget (`DEFERRED_BUDGET_LO` and `_HI`), held
        and dormant, and dormant ones are let go past half of it, the oldest
        first (`guest/kext/src/RgfxSurfaceBlocks.hpp`).

15. **The stage 3 stack is IOGPUFamily.** x86-64 macOS 26 carries IOGPUFamily
    and IOGPU.framework, and the x86-64 port runs Metal compositing on them:
    WindowServer composites with Metal through an IOSurface display pipe on
    IOGPU that the driver provides. Our accelerator does the same.

16. **Hardware video is AppleGVA's vendor driver, in our bundle.** On an
    x86-64 Mac, VideoToolbox's hardware decoders and encoders run in
    VTDecoderXPCService and VTEncoderXPCService, one of each per client
    process, through AppleGVA. AppleGVA reads the accelerator's
    `IOVARendererID` and `IODVDBundleName`, loads
    `/Library/GPUBundles/<name>.bundle/Contents/MacOS/<name>`, and calls its
    `AVD_CreateAVDAccelerator` for the driver's identity, its capability block
    and a table of sixteen entry points (sessions, bitstream and pixel buffer
    pools, command scheduling and status, parameters). Both services allow
    the bundle: their library validation is off, and their sandboxes read
    `/Library/GPUBundles` and open IOAccelerator user clients. A video
    plug-in of our own in `/Library/Video/Plug-Ins` is no way in: CoreMedia
    lists only Apple-signed plug-ins.
    - **Identity:** the accelerator's personality names the bundle,
      `reims-graphics`, as the video driver, with `IOVARendererID` 0x1020004,
      the AMD Navi family's, and `IOGVACodec`; the bundle exports the two
      entry points beside its Metal device
      (`guest/bundle/src/RgfxVideoDriver.cpp`). AppleGVA drives an AMD driver
      through high-level commands (picture parameters and slice data;
      rate-control settings), which map onto Vulkan Video. The driver decodes
      H.264 and HEVC, each where the host's caps list it; HEVC in the profiles
      they list (Main into NV12, Main 10 into P010).
    - **Routing:** AppleGVA hands a session to the AMD driver only on the
      Macs without an Intel GPU, or where `com.apple.AppleGVA`'s
      `gvaForceAMDKE` is set; elsewhere, an iMac20,1 among them, it looks for
      an Intel GPU and gives up without one. RgfxGate's install thread keeps
      the key set in `/Library/Preferences/com.apple.AppleGVA.plist` at every
      boot, after the bundle step (`guest/rgfxgate/src/RgfxGatePrefs.cpp`): an
      absent file gets a fresh XML plist, an XML plist gets the key with every
      other byte kept, and a binary or unreadable one is left and logged.
      AppleGVA reads it from the next service start (measured on the guest,
      macOS 26.6.1).
    - **The GPU is named GFX0.** AppleGVA's video routing finds the GPU by
      the name real AMD Macs give its PCI device, `GFX0`. HEVC needs it: the
      accelerator names its IOPCIDevice provider `GFX0` in `start`, before
      `registerService`, and without that name `VTDecompressionSessionCreate`
      refuses every HEVC stream with −12913 and the driver is never called.
      H.264 decodes either way, and the `IOGVACodec` string makes no
      difference (measured on the msi 5080, 2026-11-03: three boots with the
      provider's default name refused HEVC with −12913; a fourth that changed
      only the provider's name to `GFX0` decoded every frame, H.264 still
      working). Where a device already carries the name (a renamer such as
      WhateverGreen, or a second GPU), the name is left as it was and that is
      logged, so two are never published
      (`guest/kext/src/RgfxAccelerator.cpp`, `nameProviderForVideo`).
    - **Capabilities are the host's:** the block lists the engines and, per
      codec and direction, a function record (its ID, sub-ID, input format
      `rawb` for a bitstream or `NV12` for frames, and output formats), from
      what the host's Vulkan Video answers (`RgfxVideoCaps` in the device
      info). A codec the host lacks is not listed, so VideoToolbox falls back
      to software, as on a Mac without it.
    - **The wire:** a video session is an object of the device's
      (`VIDEO_SESSION_CREATE`), with a command queue of the driver's; each
      picture is a command buffer of that queue holding one video encoder,
      whose op (`RGFX_OP_VIDEO_DECODE_H264` or `RGFX_OP_VIDEO_DECODE_H265`,
      the 0x4xxx group) names the bitstream buffer and range, the parameter
      sets where they changed (H.264's SPS and PPS; HEVC's SPS and PPS, the
      host making the VPS from the SPS), and textures over the output
      surface's two planes. The host decodes on its Vulkan Video decode queue
      and has the command buffer wait for the decode on the session's shared
      event, so the buffer completes, and AppleGVA's token is answered done,
      once the picture is in the surface. For HEVC the driver writes the
      parameter sets from AppleGVA's parsed form, as for H.264
      (`guest/bundle/src/RgfxVideoH265.cpp`): each short-term reference
      picture set by its pictures, each scaling list and, where the stream's
      random access skips a leading picture, no output for it. AppleGVA
      holds a set the SPS predicts from another by its pictures as well,
      derived, so the driver writes every set by its pictures, a predicted
      one too (measured on the guest with a stream of 25 sets in its SPS, 22
      of them predicted). Encode is to take the same path.
    - **The HEVC output surface is named in the parameters, not on the
      command.** An H.264 decode arrives as a pipeline whose transfer command
      carries the output surface; an HEVC decode is a bare decode command with
      no output on it and no transfer command. AppleGVA names the output frame
      in its own decoded-picture record instead: the record pointer at the
      parameters' `+0x10`, its current frame at the record's `+0x08`, and that
      frame's global IOSurface id at the frame's `+0x88`. The driver looks the
      surface up there (`guest/bundle/src/RgfxVideoDriver.cpp`,
      `hevcOutputSurface`; measured on the guest with the RTX 5080: the id is
      that of the IOSurface VideoToolbox hands back for the picture, 7 for
      picture 0 and 27 for picture 1, and an IDR with no references resolves
      there too).
    - **A decode the driver cannot complete is a decode error, never a silent
      success.** AppleGVA reports a decode error only when scheduling the
      command fails: a schedule that answers 5 reaches the client as
      `kVTVideoDecoderMalfunctionErr` (-12911). It ignores every failure a
      status query can answer, a detail word, a failed query call, and the
      statuses 0 and 3 alike, and takes the picture for decoded (measured on
      the guest with the RTX 5080). So a command the driver cannot resolve or
      send (no output surface, refused parameter sets, no command buffer)
      fails its own schedule, and logs one line. A failure the host reports
      for a picture the driver sent is known only when the picture's status
      is queried: the driver marks its session, and the session's next
      schedule fails, one picture late (`guest/bundle/src/RgfxVideoStatus.hpp`,
      `queryReport` and `scheduleFails`).
    - **Bitstreams:** AppleGVA asks for one bitstream buffer and writes the
      next access unit into it as soon as a picture is scheduled, without
      waiting for the decode that reads it. The driver copies each access
      unit, from its first 3-byte start code and padded with zeros to the
      host's `bitstream_alignment`, into the next of its session's eight
      Shared buffers, which the host's decode reads in place. That copy of
      the compressed data is the only one.
    - **Pictures:** AppleGVA's commands name no references, and AppleGVA
      reuses an output surface whatever the stream still references, so the
      host keeps the decoded picture buffer in images of its own, from the
      stream's slice headers. Where the device decodes only into its DPB
      pictures (`DPB_AND_OUTPUT_COINCIDE`, as the RTX 5080's decoder), each
      picture then takes one GPU copy from its DPB image into the output
      surface's planes, and nothing else copies a frame on the GPU.
      AppleGVA's surfaces lie in place (§14, measured on the guest with the
      RTX 5080). On a GPU of its own memory, the host says
      `RGFX_VIDEO_CAPS_OUTPUT_CACHED` in the video caps, each output
      surface's block is cached as every surface's is (item 14, *Which
      surfaces are cached*; without the flag the driver asks them uncached),
      and the copy writes the surface's device cache, within the GPU's
      memory, on the queue the runtime reserves for the decoders (family 0
      on the RTX 5080), which no runtime work runs on, so the copy waits for
      its decode on the GPU and holds no guest work behind it.
      The guest's GPU work then reads the picture where it landed, and it
      crosses the bus only at a CPU lock, as the cache's download. Measured
      on the RTX 5080: a 4K picture lands in 100 us of the GPU, where the
      copy into the pages took 3.5 ms across a link idled down by a load
      of decodes alone; a CPU lock downloads it in 1 to 5 ms. Where the GPU
      shares the CPU's memory, the copy writes the surface's pages, on the
      decode queue.

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
| 0x004 | `VERSION` | RO | the wire's version, `RGFX_VERSION` (below) |
| 0x008 | `FEATURES_LO` | RO | feature bits 0-31; 0 in stage 2; the device info's `features_lo` holds its low byte for the processes |
| 0x00c | `FEATURES_HI` | RO | feature bits 32-63 |
| 0x010 | `STATUS` | RW | writing 0 resets the device; the guest writes 1 (driver ready) once it has read the rings' headers |
| 0x014 | `REGION_SIZE_LO` | RO | BAR2's size, low 32 bits |
| 0x018 | `REGION_SIZE_HI` | RO | BAR2's size, high 32 bits |
| 0x020 | `DOORBELL` | WO | writing a ring's id wakes its consumer (an `ioeventfd`) |
| 0x024 | `DRIVER_FEATURES_LO` | RW | feature bits 0-31 the guest's driver takes; reads back the ones the host took ("Negotiated features", below) |
| 0x028 | `DRIVER_FEATURES_HI` | RW | feature bits 32-63, likewise |
| 0x02c | `CONSUMED` | WO | writing a completion ring's offset wakes its producer waiting for room, under `CONSUMED_DOORBELL`; no host handles it yet, so it reads 0 and a write is ignored |
| 0x030 | `DEFERRED_BUDGET_LO` | RO | under `SURFACE_DEFERRED`, the bytes of the host GPU's memory deferred blocks' device caches may hold, low 32 bits: a quarter of the device-local heap's budget, the same for the device's life; 0 without the feature |
| 0x034 | `DEFERRED_BUDGET_HI` | RO | its high 32 bits |

**The wire's version.** `VERSION` reads `RGFX_VERSION`
(`crates/reims-graphics/src/wire.rs`), the version of the wire the host
speaks. The kext's framebuffer and accelerator, and the option ROM, each
accept only the version they were built with. They read it right after
`MAGIC`, before anything else of the device, and on any other version they
leave the device and say why, naming both versions: the kext in the kernel
log, the ROM on the firmware's console where it has one.

The bundle maps the region and speaks the messages too, so it must be of the
kext's version, and it says which it is: its package carries an empty
`Contents/Resources/wire-version-N`, N the `RGFX_VERSION` it was built with,
under its signature. The kext registers its accelerator only for a bundle
carrying the marker of its own version ("Registration waits for the bundle",
below); for any other it says so in the kernel log and stays unregistered,
so WindowServer composites in software, as with no bundle. The bundle does
not check the version itself: a bundle that declined to make a device under
a registered accelerator would leave WindowServer aborting, as with no
bundle at all.

Leaving the device means writing nothing to its registers or its region.
The framebuffer's start resets the device (`STATUS` 0) only once every check
that needs no command has passed, and resets it again on the way out only if
it did so. With the kext refused, or failed after its reset, the OS drives
the display through the firmware's framebuffer, which the host shows through
the scanout the firmware set up: a reset keeps the scanout (Reset, below).

Every change a guest built before it would misread moves `RGFX_VERSION` on by
one, in the same commit:
- a register's offset, width or meaning;
- an offset or a size in the region: the control rings, the device info, the
  scanout's room, the blocks' room, the cursor;
- a message's or an op's number, layout or meaning, a ring's format, or a
  status the guest acts on.

A change that no guest built before it can misread leaves it: a new feature
bit (`FEATURES_LO`, `FEATURES_HI`), or a new message or op that older guests
never send. A version number is never used twice. The host speaks one version,
and a host and a guest of that version from different builds meet through the
features they negotiate (below).

**Negotiated features.** A change lands as a feature bit, and moves `VERSION`
only where no negotiation can carry it.
- The host offers its bits in `FEATURES_LO` and `FEATURES_HI`. Bits 0-2
  (`SURFACE_PAGES`, `SURFACE_CACHE`, `CB_SUBMITTED`) need no answer: the host
  acts on them as it offers them.
- From bit 3, the host acts on a bit only where the guest's driver took it.
  The framebuffer kext writes `DRIVER_FEATURES_LO` and `_HI` after `STATUS` 0
  and before `STATUS` 1, in `ReimsGraphics::start`, with the bits it and its
  bundle carry out, masked by the offer (`RgfxFeatures.hpp`). It reads back the
  bits the host took: those written that the host offers and knows. It logs
  the three words, and sets them on its service as `RgfxFeaturesOffered`,
  `RgfxFeaturesWanted` and `RgfxFeaturesTaken`, which `ioreg` reads at any
  time after the kernel's message buffer has let go of the line.
- The host ignores a write while `STATUS` is 1, and a reset (the guest's
  `STATUS` 0 or the host's) clears the set. A guest that never writes it (the
  option ROM, a kext built before the registers) gets the wire as it was, and
  a host built before them reads them as 0, so its guest takes nothing.
- A feature with a mode per queue rides `QUEUE_CREATE`'s tail,
  `RgfxQueueCreateFlags` (`flags`, `RGFX_QUEUE_*`) after its 56 bytes. The kext
  sends the tail only with a flag set. A flag whose feature the guest did not
  take, or that the host does not carry out, refuses the create
  (`INVALID_ARGUMENT`).
- New ops and messages are decoded wherever the host offers their bit, as
  `RGFX_OP_SUBMITTED` is.
- The accelerator answers its processes' `Features` call (selector 0x10f) with
  two words: the device's, which they work by (bits 0-2 as offered, and from
  bit 3 the bits taken), and the kext's own (below, "The accelerator's user
  client"). `QueueCreate` (0x103) takes a seventh input, the flags the process
  wants, and answers a second output, the flags granted: those whose feature
  the device's word holds. A kext older than the call answers
  `kIOReturnUnsupported`, and the bundle then works by the device info's
  `features_lo` bits 0-2, with no kext feature.
- Bits 3-8 are named (`QUIET_SCHEDULED`, `CONSUMED_DOORBELL`, `QUEUE_BARRIER`,
  `CB_PARTS`, `PAGES_BUFFERS`, `QUEUE_VECTORS`; `wire.rs`), and bits 9-12 are
  reserved. Of them the host offers bit 7, `PAGES_BUFFERS`, where it offers
  `SURFACE_PAGES`, and the kext takes it: a process's no-copy buffers lie over
  its own pages, wired by the kext in a pages block (selector 0x10e, "The
  accelerator's user client"; `BUFFER_OVER_PAGES`). No host offers the others
  yet, the kext takes none of them, and the host carries out none of their
  queue flags: each lands with its feature, which says what it changes.
- Bit 10 is reserved for a pipeline's warm-up, and bit 11 is `SURFACE_ORDER`:
  a command buffer that uses IOSurfaces runs after the buffers of other queues
  it names (`RGFX_OP_SURFACE_ORDER`; "Surface order", below).
- Bit 13, `UNSHARED_QUEUES`, every host offers and the kext takes: the queue
  flag `RGFX_QUEUE_UNSHARED` (`1 << 3`; `1 << 2` is reserved), which the kext
  sets on WindowServer's queues ("WindowServer's queues are unshared", below).
- Bit 14 is `MANAGED_AT_OFFSET`: ring 0 carries `BUFFER_MANAGED_AT_OFFSET`, a
  Managed buffer whose CPU copy the host places in a heap over host memory of
  the sender's, so no buffer imports host memory of its own (the slabs,
  below). A host offers it where a Managed buffer is two copies, a GPU that does
  not share the CPU's memory.
- Bit 15 is `HEAP_OVER_PAGES`: ring 0 carries `HEAP_OVER_PAGES`, a heap over
  the one run of guest RAM a pages block of the sender's holds, made over the
  host's mapping of the guest's RAM rather than the block's view, so the
  buffers placed in it cost what those of a heap over host memory do. A host
  offers it with `PAGES_BUFFERS`.
- Bit 16 is `SUBMIT_ROOM`: a guest producer that finds a submission ring full
  stores 1 in the ring's `producer_waiting`, fences, and looks once more before
  it waits. After each take that moves the ring's `consumer`, the host fences
  and reads the flag, and where it reads 1 it stores 0 and posts a header-only
  `RGFX_EVT_SUBMIT_ROOM` (0x408) on the ring's completion ring, ring 0's or the
  queue's. The waiting producer looks again at that record: one interrupt after
  the host's take, not at a sleep's end. Every host offers it.
- Bit 18 is `COMMIT_ORDER`: a command buffer is committed after the buffers of
  its process's other queues it names (`RGFX_OP_COMMIT_ORDER`; "Commit order",
  below).
- Bit 19 is `SURFACE_DEFERRED`, which means nothing without `SURFACE_CACHE`:
  a fresh cached surface only the GPU touches gets no guest pages until a CPU
  lock needs them (`RGFX_SURFACE_PAGES_DEFERRED` and `_ATTACH`; `wire.rs`). A
  host offers it where it offers `SURFACE_CACHE`, with its budget in
  `DEFERRED_BUDGET_LO` and `_HI`.
- Bit 21 is `MANAGED_TEXTURE_SLABS`: a `MANAGED_TEXTURE_CREATE` may name any
  Shared buffer over the sender's memory as its CPU copy, and the host reads
  and writes the copy where the buffer's contents lie, so the bundle takes a
  Managed texture's CPU copy from its slabs ("Managed storage", below). Every
  host offers it, and the kext takes it.
- Bit 22 is `MANAGED_BUFFER_SLABS`: ring 0 carries `MANAGED_BUFFER_CREATE`, a
  Managed buffer whose CPU copy is the first bytes of any Shared buffer over
  the sender's memory, read and written where that buffer's contents lie, so
  the bundle takes a Managed buffer's CPU copy from its Shared slabs, its own
  superpages among them, on every host ("Managed storage", below). Every host
  offers it, and the kext takes it.
- Bit 24 is `DISPLAY_COLOR`: the guest sends the display pipe's gamma tables
  and colour matrix as `DISPLAY_COLOR` parts, which the next `PRESENT_SURFACE`
  shows, and a present may keep the surface shown and change only the colour
  (`RGFX_PRESENT_KEEP`; "The display's colour", below). Every host offers it;
  one with no window answers its presents at once.

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
`c << 40` and up), and a range of GPU addresses, held until its close, from
which its bundle picks every buffer's, heap's and texture's address
(`guest/kext/src/RgfxContextSlots.hpp`). The guest's addresses are
`[2^46, 2^47)` (`RGFX_CONTEXT_VA_BASE` to `RGFX_CONTEXT_VA_END`), which the
host's own ranges never reach, in 1024 units of 64 GiB
(`RGFX_CONTEXT_VA_UNIT`); a context holds an aligned run of units: one, unless
the device's `maxBufferLength` is more than 32 GiB, and then the least power of
two that holds twice it (at most 32), so a buffer of `maxBufferLength` always
fits. The kext reads `maxBufferLength` from the device info at the first open.
`OpenContext` answers the range's span sixth (kext feature `ContextSpan`); a
bundle that asks for five values picks through `RGFX_CONTEXT_VA_SPAN`, 2 TiB,
and gets an aligned run of 32 units, so an older bundle keeps working on a
newer kext, and a newer bundle on an older kext takes the 2 TiB it hands
out. Runs go round-robin, so a run given back is the last one the next
contexts take, and the last 32 units free go to WindowServer alone, as the
blocks' and queues' reserves do. The host knows none of this: it pins each
address as given. Sized from what a desktop uses, measured on the RTX 5080 host
(a6e722f47's host and guest, macOS 26.6.1): 17 processes held a context on an
idle desktop and 74 with 42 apps and 7 WebGL and WebGPU pages open, where the
32 slots of 2 TiB the kext had before refused the 33rd process and every later
one. The most GPU addresses one process held at once was WindowServer's 204 MiB
(in 1525 picks of `RgfxIdSpace::addressSpan`), the largest single pick 35 MiB,
and the furthest any process's picks went into its range 1.5 GiB (they go
forward and come round only at its end), so a 64 GiB unit holds the busiest
process's addresses over 300 times. Past the last run, `OpenContext` answers
`kIOReturnNoResources`, and the bundle answers no device in that process
(`initWithAcceleratorPort:` answers nil), which every Metal client handles,
where a device whose every create answered nil stopped the frameworks that make
their first objects at its open (Maps, Journal and Photos aborted, Games
faulted). The kext hands out its region blocks
(first fit, page granular). The process hands each create and destroy of its
objects (ids in its range) to the kext by a user-client call; the kext posts it
on the control ring and, for a pipeline create, hands the reflection back once
the host answers it. A queue's `QUEUE_CREATE` goes the same way, its rings
checked to lie in the process's blocks, and is answered once the host has the
rings. At the process's close or death the kext destroys its queues, then
closes its context (`CONTEXT_CLOSE`: the host takes its objects out of its
tables, lets go of them on a worker of its own, `reims-vgpu.graphics-closes-<n>`,
and answers once it has and the context's command buffers have finished),
then, on that answer only, frees its chunks of resource ids, its blocks and its
slot of GPU addresses. A command
buffer is decoded once the host has carried out every control-ring message of
its context's published before its queue's thread took it, and has taken every
message before those.

**A buffer's wait on ring 0.** A buffer names only its own context's objects
and spans in blocks its context holds, so what it can depend on on ring 0 is
its context's: a message about an object (every type from `QUEUE_CREATE` below
`CONTEXT_CLOSE`, which the kext takes from a process only with an object id of
the process's), whose id carries the context's bits, or one the kext posts for
a context (`CONTEXT_CLOSE` to `SURFACE_PAGES`), which names it.
Another process's surface reaches a buffer only through the share of its block
the kext posts for the buffer's context before the process learns of the block,
after the surface's `SURFACE_PAGES` was answered: what the buffer reads of the
surface the host made there, and the buffer names none of the other process's
objects, so it waits for its own context's share and for none of the other
process's messages. Shared events and textures do not cross processes. A
display command (`GET_DISPLAYS` to `DISPLAY_COLOR`) or a `SURFACE_DOWNLOAD`
holds up no buffer: none reads what they make. A present waits for the writers
the guest's record of its surface names, whatever the host has taken. A
download waits for the buffers that used the surface's device cache before it
was taken, so a buffer posted after it that writes the surface, but taken
before it, is among them, as one posted before it is: the guest writes a
surface a CPU lock holds only racing the lock. Every buffer waits for a message
of any other type, which the host refuses. The queues' threads read
ring 0 ahead of the control thread, each message once, and keep the count
through each context's last message (`reims_graphics`'s `device::backlog`), so
a buffer behind another process's backlog on ring 0 is decoded at once.
Tested by `a_command_buffer_waits_for_no_other_process_s_creates`,
`a_command_buffer_waits_for_no_other_process_s_later_object_messages`,
`a_process_s_messages_go_on_while_another_s_create_is_carried_out`,
`a_command_buffer_waits_for_its_share_of_another_process_s_surface_not_for_that_process`
and `a_doorbell_write_answers_a_command_buffer_behind_another_process_s_create`.

**Ring 0's lanes.** The control thread takes ring 0 in ring order and carries
out only the display commands itself. It hands each context's messages, in ring
order, to that context's lane, a thread of its own
(`reims-vgpu.graphics-context-<n>`, made as a message comes to a lane with
none, gone after 1 s with none), every `PRESENT_SURFACE` and
`DISPLAY_COLOR` to the presents' lane (`reims-vgpu.graphics-presents`), in
ring order, and every `SURFACE_PAGES` and
`SURFACE_DOWNLOAD` to the lane of the block it names
(`reims-vgpu.graphics-block-<offset>`), whichever process posted it and for
whichever context, so a block's pages handed over from another process's CPU
lock, or after its maker closed, and the download or release posted right
after them, keep their order without waiting for the maker's lane. A lane
carries its messages out one at a time and posts their answers, so one
process's slow message (a heap over pages took 469 ms at the median and 4.25 s
at most in a boot on an RTX 5080 host) holds up only that process's later
messages and command buffers. A message
waits for another context's lane only where it reads what that lane's earlier
messages change, and only for messages published before it
(`reims_graphics`'s `device::lanes`):
- a present waits for each lane's `TEXTURE_FROM_IOSURFACE` of its surface's
  plane 0, its destroy of a texture the host names the surface by, and the
  close of a context that holds one: the host shows the newest live texture
  over the surface;
- a shared block's `BLOCK_RELEASE` waits for each other context that held it:
  the host refuses a release while an object of any context lies in the block;
- a `BLOCK_GRANT` in the scanout's span waits for every lane: a grant takes the
  span from whoever held it, and WindowServer's contexts are granted it in turn;
- a `RESOURCE_ID_CHUNK` another context held waits for that context's close,
  and a context's first message for every close before it, whatever the
  context's number (the kext numbers contexts at their opens, and a process
  may post its first message after one opened later): the kext frees a closed
  context's chunks and its slot of GPU addresses once the close is answered;
- a context's message waits for each block's lane through the context's last
  `SURFACE_PAGES` on it, and a command buffer for its context's messages on
  the blocks' lanes too, as when they shared the context's lane;
- a `BUFFER_OVER_PAGES`, `HEAP_OVER_PAGES`, `BUFFER_OVER_SURFACE_PAGES` or
  `BLOCK_RELEASE` waits for its block's lane: each reads the view or device
  cache the block's lane makes and copies;
- a `SURFACE_PAGES` waits for its context's lane only through the last message
  of those four kinds that names its block; a `SURFACE_DOWNLOAD` waits for no
  other lane, as on the control thread before.

`reims_graphics`'s `device::lanes` lists every message that names a block or
its pages by offset, and what orders each against the block's lane.

A close's job on its lane is its table work: it lets go of the context's
objects on the closes' worker (`reims-vgpu.graphics-closes-<n>`), which the
close's answer waits for, so a message that waits for a close waits for none
of the platform's releases (an event's took 0.4 ms on an RTX 5080 Laptop GPU,
measured by the decoder's `close_cost`). A reset waits for the closes before it
to have let go of theirs.

Each wait is for a message published earlier, and every lane with a message
has a thread, so the waits end. What the guest can tell apart from one thread:
answers to different processes come in another order than their messages,
which the kext matches by seqno; and two processes' messages that no answer
orders are carried out in either order, as the processes' own posts race. A
process that waited for an answer finds its message carried out. A reset drops
the messages a lane has not begun, and is told to the host once those begun
have returned. A lane's thread is woken only for its own messages and for the
lanes its next message waits for, so a lane with nothing to do costs nothing
and goes after a second with none, whatever other processes do. Tested by
`a_present_waits_for_another_context_s_texture_over_its_surface`,
`a_present_waits_for_a_destroy_or_a_close_of_a_texture_over_its_surface`,
`a_present_waits_for_no_message_that_leaves_its_surface_s_textures`,
`a_shared_block_s_release_waits_for_its_other_holders`,
`an_unshared_block_s_release_waits_for_no_other_context`,
`a_scanout_grant_waits_for_every_context_s_messages_before_it`,
`a_chunk_or_a_new_context_waits_for_the_close_before_it`,
`a_shared_block_s_release_behind_a_holder_s_close_waits_for_none_of_its_releases`,
`a_reset_waits_for_a_close_s_slow_release`,
`a_context_s_first_message_waits_for_the_close_before_it_whatever_its_number`,
`a_sync_create_is_answered_before_its_context_s_later_messages_are_carried_out`,
`a_context_s_answer_comes_before_one_published_ahead_of_it_by_another_context`,
`a_reset_drops_what_the_lanes_have_not_begun`,
`an_idle_lane_goes_while_another_context_s_is_busy` and
`lanes_keep_every_order_over_random_interleavings`, which checks random
interleavings of several processes' messages against an order worked out from
the messages alone. A synchronous create's answer takes 7.2 us on average from
its doorbell, against 4.1 us from the control thread alone, and 68 us against
23 us where the lane's thread is made again after a pause
(`a_sync_create_s_round_trip`, release build, M3 Max). A context's burst of 200
messages and a synchronous create take as long beside 75 idle lanes as beside
none: 0.1 to 1.4 ms at the median either way, on the M3 Max under other builds'
load (`a_burst_beside_idle_lanes`, release build).

A queue's thread waiting for ring 0 before it hands a command buffer on is
woken once the control thread has taken the messages the buffer waits for, and
once its context's lane has carried out the context's among them, and by
nothing else but a reset, its queue's end and the device's
(`reims_graphics`'s `device::gate`): no other process's message wakes it.
Tested by `the_gate_wakes_each_waiter_once_what_it_waits_for_holds`, which
moves the count and many contexts' lanes at random beside many waiting
threads. Beside 75 other processes' queues each waiting on a held create, a
context's burst of 200 messages and a synchronous create took 11 to 16 ms at
the median when every lane's message and control step woke every waiting
queue, and take 49 to 123 us now, as beside none
(`a_burst_beside_waiting_queues`, release build, M3 Max under other builds'
load).

**The surface and ICB workers.** A lane hands what it would wait long for, and
the guest learns of only through an answer or a later message, to a worker of
the host's: a `SURFACE_PAGES`, whose view of a 3840x2400 layer took 9000
mappings, made in 16 to 478 ms (msi 5080 host), a pages block's
`BUFFER_OVER_PAGES` that makes or follows such a view, the let-go of a released
block's views and device cache and of a buffer over a view at its destroy, and
an `ICB_CREATE`. A worker runs the jobs of one
key one at a time, in the order they were handed over, and different keys' at
once, on at most four threads (`reims-vgpu.graphics-surfaces-<i>`,
`reims-vgpu.icb-create-<i>`), each made as a job finds none waiting and gone
after a second with none. The surface worker's key is the block, the ICB
worker's the process. So a block's slow view holds up only that block's later
jobs, and a key that finds every thread busy waits for the next free one. No
job waits for a job of another key, so jobs waiting for a thread never wait
for each other: a `BUFFER_OVER_PAGES` waits only for its own block's
`SURFACE_PAGES`, done before it, and a `SURFACE_PAGES` for nothing, as a debug
build checks. What the guest can tell apart from one thread: two
`BUFFER_OVER_PAGES` of one process on different blocks are made in either
order, which no answer orders, and the process's later messages wait for
both. Tested by `a_key_whose_job_hangs_holds_no_other_keys_jobs`,
`a_key_waits_for_a_free_thread_once_every_thread_is_busy`,
`each_keys_jobs_run_one_at_a_time_in_handed_order` and
`an_idle_thread_ends_after_the_idle_time`. Beside K blocks that each run 25 ms
jobs back to back, another block's job starts after its hand-off in 0.015 ms at
the median for K of 1 or 2, 5 to 6 ms for 4, 22 to 23 ms for 6 and 35 to 38 ms
for 8, where one thread took 24, 54 to 56, 115 to 117, 179 to 182 and 241 ms
(`another_blocks_job_beside_busy_blocks`, two runs of each, release build, M3
Max at load averages of 28 to 52).

**WindowServer's reserve.** The blocks have no per-process cap: Metal on a Mac
refuses an allocation only when memory runs out, never because another process
holds much, so neither does the kext. The one process whose refusal would stall
every other is the compositor. So the last 160 MiB of the blocks' room and the
last 1024 of the table's 16384 entries go only to WindowServer's blocks
(`RgfxBlockTable`'s reserve). WindowServer takes the first gap from the start
like any process, and reaches into the reserve only once the rest is full; any
other process is refused at the reserve's edge, as at the region's end. Of the
watched rings, the last 64 are WindowServer's alone in the same way, past room
for one process's 2048 queues beside as many of the others'
(`RgfxQueueLimits.hpp`). The kext logs the most
WindowServer's blocks and queues came to, which is what the reserve is sized
from: on an RTX 5080 host at the display's 3840x2400 mode, WindowServer held
134 MiB at most, at login, and 118 to 126 MiB in 110 to 366 blocks as five apps
opened, with 2 queues (measured). A larger display, or a second one,
needs the reserve measured again.

The kext knows WindowServer's client by two tests at its open: the opening
process is named `WindowServer`, and it holds the entitlement
`com.apple.private.graphics-restart-no-kill`
(`IOUserClient::copyClientEntitlement`). WindowServer holds that entitlement on
macOS 26.7.1 and 27, read from its signature (`com.apple.WindowServer`,
a platform binary). With SIP on, AMFI refuses any `com.apple.private`
entitlement to a binary Apple did not sign. So a process that only takes
WindowServer's name fails the second test. The direct tests, the process's
signing identifier and its platform-binary status (`cs_identity_get`,
`csproc_get_platform_binary`), are in `com.apple.kpi.private`, which a kext
outside Apple's namespace cannot link. Resolving them at run time through the
loader is not worth its risk.

**WindowServer's queues are unshared.** The host's runtime may let command
queues share a GPU queue, which raises the rate of small command buffers on a
GPU whose switches between queues idle it. A shared queue can then hold
another process's long command buffer ahead of the compositor's frame. So
the accelerator flags every queue WindowServer's process creates
`RGFX_QUEUE_UNSHARED` in its `QUEUE_CREATE`'s tail, where the host offered
`UNSHARED_QUEUES` and the framebuffer took it (`rgfxQueueFlagsSent`,
`RgfxFeatures.hpp`). It decides from its own test of the client's task at
the open (above), never from what the process asks: `QueueCreate`'s seventh
input never grants the flag. The host makes such a queue with the platform's
`new_command_queue_unshared` (`GpuDevice`): on the Vulkan platform the
runtime's queue `unshared_queue` makes, and on Metal the queue any create
makes. The host logs each unshared queue
at info level (`reims_vgpu::graphics`). Nothing the guest's processes see
changes.

**Reset.** A reset leaves the device as one just made, as a device's
function reset does: `STATUS` 0, which the framebuffer writes at its start,
and QEMU's reset of the device, which a reboot inside one QEMU run makes.
Every command queue ends, no block is any process's, and the host drops every
object of every context, as each one's close does, with no answer owed. The
blocks' surface views and device caches go too, uncopied, since their pages
are the rebooted guest's again, and so do the library cache, the presents
waiting and the pipelines compiling. Nothing begun before the reset reaches
the guest after it. The guest then finds what a new device gives it: the ids
it picks free from 1 again, every texture resource id and GPU address free
(a range a buffer of the guest's before the reset still held would refuse the
rebooted WindowServer's pin of it), a library lookup that misses, and VBLANK
counts from 0. What the guest wrote in the region stays.

The display keeps its scanout across a reset, as a GPU's display engine goes
on scanning out across a reset of its driver: the host's window shows what
the guest writes there until the guest sets another scanout or blanks it.
The kext's start resets the device to take it, and where the start fails
after that (the device lists no usable display, or `IOFramebuffer`'s start
fails) it resets it again on the way out; the OS then draws into the
firmware's framebuffer, which the firmware's scanout, kept through both
resets, still shows. Tested by
`a_reset_keeps_the_scanout_and_what_the_guest_writes_there_shows`
(`reims-graphics`) and
`a_failure_after_the_take_keeps_the_firmware_s_scanout_in_the_window`
(`graphics/tests.rs`).

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
as a refused create is: unanswered, and an id of its own context then stands
for a refused object, whose uses are dropped ("Guest input the host refuses").
Sampler states and pipeline states take no guest-picked ids (the host numbers
them), and the bundle makes no indirect command buffers, so textures are the
only kind with chunks.

**Objects go with their wrappers.** Every host object the bundle makes goes
from the host as the app lets its wrapper go. A buffer, heap, texture, sampler,
depth-stencil state, render or compute pipeline, library, event, fence or shared
event posts its `OBJECT_DESTROY` from its dealloc. Samplers and depth-stencil
states are made once a descriptor: the device answers an equal descriptor,
label included, with the object it made, as the M3 Max's Metal does, and keeps
the last 64 of each kind it made after the app lets them go, so an app that
makes one every frame pays one create (`guest/bundle/src/RgfxStateCache.h`).
A queue ends through the kext
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
| 0x08 | `completed_seqno`: u64, on a queue's completion ring the seqno of the last command buffer that finished OK, 0 before one has | host (queue completion rings) |
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
- The re-arm. A consumer already awake (the completion thread after a
  buffer's `CB_SCHEDULED`) can take a record between the host's publish and
  its clear, and arm again: the clear then takes an arm made after the take,
  and its vector comes for a record already taken, so the thread sleeps with
  `armed` 0 and the host raises nothing for the next record. So where the
  clear reads 1, the host reads `consumer` too, and where the guest has taken
  every record published, sets `armed` back to 1 and raises nothing
  (`ring.rs`, `rearm_or_raise`). The guest stores `consumer` before it arms,
  so a clear that read its arm sees its take, and only the host writes 0, so
  the 1 undoes nothing of the guest's. The fix is the host's alone: no kext or
  bundle changes, and ring 0's completion ring takes it too.
  `ring/interleavings.rs` walks every interleaving of the host's post, the
  kext's look and the completion thread's take, arm and sleep (up to 3
  records and 2 vectors for something else): none leaves a record untaken or
  the thread asleep unarmed. A wake index the host only reads (virtio's
  `used_event`) was weighed and declined: it would change a ring's format in
  the host, the kext and the bundle, and the re-arm closes the race without
  one.

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

**Refresh times.** The host answers a `PRESENT_SURFACE` in the refresh that
shows its surface, right after that refresh's `VBLANK`. The kext maps each
`VBLANK`'s host time into the guest's `mach_absolute_time` by the clocks'
offset with the shortest delivery (`guest/kext/src/RgfxVblankClock.hpp`). A
display pipe transaction completes with the time of the refresh it was
answered in (or the answer's own time where no `VBLANK` came within a period),
and a display pipe VBL request is answered once, at the next refresh, with that
refresh's time and the next one's.

**The display's colour.** A Mac's display pipe applies colour stages to what it
shows: plane 0's gamma table (selector 5), the pipe's gamma table before its
colour matrix (6), the matrix, which rides the transaction at 0xb0 with dirty
bit 0x40, and the pipe's gamma table after the matrix (7). CGDisplayFade, Night
Shift and ColorSync's calibration set them; WindowServer's boot fade is a run
of transactions about 16 to 30 ms apart that each set only selector 7's table,
with no surface. The kext decodes the tables as macOS 26's IOAcceleratorFamily2
does and stages each for the transaction begun (read;
`guest/kext/src/RgfxPipeColor.hpp`, `tests/pipe_color_test.cpp`); the submit
sends them, and the matrix where it is dirty, as `DISPLAY_COLOR` parts of at
most 256 entries before the transaction's `PRESENT_SURFACE`, and a transaction
with no surface that sets a colour as a present that keeps the surface shown
(`RGFX_PRESENT_KEEP`), which completes the transaction when the host answers
it, at the refresh that shows the colour, as a surface's present does. The host
stages the parts on the presents' lane and the present it accepts next applies
them, with the frame it shows; each stage keeps its table until a transaction
sets it again, and the device's reset turns them all off, though it keeps the
scanout: kept, a reset in the middle of a fade would leave the window black,
with no transaction left to undo it. While any stage is on, the window draws
the display through them (`crates/reims-vgpu/src/graphics/color.rs`,
`sink.rs`), each table interpolated linearly between its entries (how a Mac's
display engine samples between them is not measured). Selectors 10 and 11 (the
pipe's linearization before the matrix and its gamma after it, in
IOPresentment's configuration format), 12 (a surface copy) and 13 (triage) stay
refused: no WindowServer seen calls them.

**The display pipe's notification words.** The kext answers a pipe client with
asynchronous results whose words IOAccelerator's notification callbacks read
(`guest/kext/src/RgfxDisplayPipeUserClient.cpp`, `completeLocked` and `sendVbl`,
which build them with `RgfxPipeNotifications.hpp`; `tests/user_client_words_test.cpp`
pins them).
A transaction's completion is exactly 11 words: the transaction's cookie in word
0, its id in the high half of word 1 (the low half 0), the time it was shown in
word 3, and the action 1 in word 8, the rest 0. The callback reads it as follows:
- it refuses any other word count;
- it takes word 0 as the completion block the client registered with the
  transaction, so a completion with no cookie is not sent (a null word 0 ends the
  client: WindowServer aborted there when the kext notified its cookie-less
  shutdown frames, `RgfxPipeTransactions.hpp`);
- it records word 1's high half as the last transaction completed;
- in the extended mode IOPresentment uses (it registers its handler with
  `IOAccelDisplayPipeTransactionSetCompletionHandlerExt`, the one completion setter
  it imports, on an Intel Mac under macOS 26.7.1 and on the M3 under 27), it accepts
  action 4 or one with a bit of 0xb set, and hands the block word 1's low half,
  words 2 to 7 and words 9 and 10.

A VBL request's answer is 6 words: the request's callback and refcon, then the
refresh's time and the next refresh's, each split over two words, low half first.
These are the reader's layouts (measured 11-03 from the callbacks' code, read as
the guest's frameworks run it, for diagnosis only), and the completion's words
are the ones Apple's own paravirtual display kext sends. WindowServer completes
every transaction through them on the lane.

The bundle presents a drawable when its
command buffer is scheduled, as Metal does (`guest/bundle/src/RgfxCommandObjects.mm`);
what an app then sees of it, its presented handlers, `presentedTime` and
`nextDrawable`'s pacing, `guest/bundle/tools/present_probe.m` measures on a
guest and on a Mac alike.

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
internal-error `CB_COMPLETE`. So a buffer that names anything the host does
not hold fails after the buffers committed before it, and nothing waiting for
one of its events, a later buffer on the GPU or a guest thread, waits for good.
An object whose create the bundle answered at the call, without the host's
answer (a buffer, heap, texture, view, depth/stencil state, event, fence, video
session, or a pipeline equal to an accepted one), is one the guest goes on
naming where the host refused it, as Metal would have made it. Its id stands
for a refused object until its destroy (the decoder's `refused`), and what
names it is dropped, never the command buffer that names it: an op that uses
it (a copy, a fill, an indexed or indirect draw from it, a fence update or
wait) is dropped; a binding of it binds nil and marks its slot, and the
encoder's draws and dispatches are dropped while a slot is marked, until the
slot is bound again; a use list's entry of it is nil; a render pass with it as
an attachment, and a video encoder of its session, are skipped whole; a wait
for it as an event is dropped, and a signal of it as a shared event still
reaches the guest's value; a create over it stands refused in turn. The
buffer then completes as it would have. Where there is no stand-in (a message
too short to name its queue, a queue the process does not hold), the error
status is posted at once. Nothing in
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
`CB_COMPLETE` is an internal error (`MTLCommandBufferErrorInternal`), and
nothing else: the context's later command buffers, creates and queries are
carried out as any context's. On a Mac a GPU error fails the command buffers
it strikes, WindowServer's among them, and their processes' later buffers run
(docs/lessons.md, the test that faults the GPU, **measured** on macOS 27.0.1),
so a failed buffer of WindowServer's leaves its later frames to be carried
out, and the display goes on. A release the
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
as a free does, and its `BLOCK_RELEASE` ends every hold; a cached block in
place instead sleeps, its last holder's `BLOCK_UNSHARE` posted, until an ask of
its surface holds it again with a `BLOCK_SHARE` alone or the kext lets it go
with its `BLOCK_RELEASE` (§14, "The device cache"). A block in place
(0x109's fourth input) is found by the pages of the surface's whole
allocation, which every holder's ask wires. Before its first `BLOCK_SHARE`,
the kext writes their runs into the block, past the part the processes map
and are granted, and posts `SURFACE_PAGES`; other asks for the surface wait
for its answer. A refusal frees the block as a share not posted does, and the
ask answers `kIOReturnUnsupported`, so the process copies the surface. Where
the device has `RGFX_FEATURE_SURFACE_CACHE`, 0x109's fifth and sixth inputs
(the surface's IOSurfaceID, and 1 where the process asks it cached) have the
kext mark a block it makes cached in its `SURFACE_PAGES`, and a fifth output
answers whether the block is, whichever ask made it. The kext then writes the
surface's CPU generation at offset 16 of the block, and asks `SURFACE_DOWNLOAD`
(block offset) at a CPU lock that follows a GPU write, answered once the
surface's pages hold the host's device cache (Decided 14, "The device cache").
A command buffer that writes surfaces' shared blocks is named to the kext
after it is posted and before its commit returns (selector 0x10c: the queue,
the buffer's seqno, and a pair of the block's offset and the surface's
IOSurfaceID for each surface), and its completion retires the queue's writes
up to it once any copy is written back (selector 0x10d). Only a holder of a
block names it, only for one of its own queues, and the kext keeps a block's
surface only where the surface's memory holds the block's pages, so no
process makes another's locks wait (§14, "CPU locks").

**Processes map only their own.** No process maps the region whole: it holds
every process's blocks, the control rings and the device page. Each
connection has a window, a range of the process the region's size that the
kext makes at the context's open (selector 0x100 answers its base and size).
The window is memory of the connection's own, mapped read-only, and holds no
page until one is read, which then reads zeros. It is purgeable, never made
volatile, so the process's footprint counts only the pages it reads there: xnu
charges an IOKit mapping of anonymous memory to the task's footprint at its
whole size as it is made, but a purgeable object's resident pages to its
owner, the process that opened the context (read from the public xnu source:
`vm_map_enter`, `mach_make_memory_entry`). A window of plain pageable memory
charged every Metal process the region's size at its open, past many
processes' memory high watermarks. The spans are device memory, which no
footprint counts. Region offset `o` of what the process is given lies at the
base plus `o`, so the bundle's pointers stay base plus offset. The kext maps
in, over the window at the span's own offset, only the process's blocks, the
shared blocks of the surface planes it holds, and the scanout it is granted.
It maps a span in before the process learns of it. Before the span can go to
anyone else, it maps the window's own memory back over the span, read-only:
before a block's free marks it freed, and when a hold ends, a scanout grant
goes or a mode change moves the scanout. At the close the whole window goes,
every span in it, before the context's blocks are freed. Each span is whole
pages past the region's head, so no mapped page holds a byte of anything else.
The device info shares its page with the control completion ring's tail, so
the kext reads it for the process (selector 0x10b), and the region's own map
(`kRgfxAccelMemoryRegion`) is refused. One difference from Apple's: a process
that touches a block after releasing it reads zeros, and faults only on a
write, where Apple's faults on both. Only a guest that uses memory it released
sees it. A mapping that fails fails its op; a span that cannot be taken back
closes the window, so the process reaches nothing of the region from then on.
A placed mapping is IOKit's overwrite mapping, which IOKit never takes down
itself (read from the public xnu source, `IOMemoryDescriptor::doUnmap`), so
only another mapping over it takes a span back; vm_protect is not a KPI a
third-party kext may link.

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
take it. The free returns at once, as Metal's releases never block. The kext's
one release worker, a thread call that runs one at a time, then posts the
block's `BLOCK_RELEASE` on ring 0, behind every message the process posted
before the free. It posts every freed block's release, 16 under one hold of the
ring's lock with one doorbell at most, up to 256 in flight, and returns without
waiting. Each release has an entry of its own, which the work loop finds by
seqno when its answer comes and then runs the worker to retire it. Answers come
in whatever order the contexts' work finishes, so a slow context holds back
only its own blocks, and a storm of frees waits for each context's work once,
not once a release. A full table leaves the rest freed until entries retire.
The host refuses the release at once where an object it holds lies in the
block: a buffer or heap over host memory, or a buffer, texture or heap placed
in one, which keeps its parent's span after the parent's destroy. It also
refuses one where a live queue's ring lies in the block. Otherwise it answers
OK once every command buffer the context committed before the release has run
its completion handler, and, for a shared block, every one each other holder
committed before its `BLOCK_UNSHARE`, before its close, or before the release
itself where neither came first: the kext unwired that holder's pages at its
unshare, or at a close it took for refused, and gives the surface's memory back
at the answer. The kext posts an unshare out of its lock, so another holder's
release can come first, and the host then takes the release as that holder's
unshare too; the unshare that follows is not counted again. It also waits
while the host's display holds a surface's texture over the block, as a
compositor holds the IOSurface it shows: from the surface's present until the
display pipe's next present (or the device's reset), and each window draw of
it until the draw has finished. Command buffers committed later do not hold it
up. On OK the block goes back to the pool; on a refusal it stays out of use
until its process's close posts its release again. A release not answered in
10 s is logged and stays in flight: the host may still run a buffer over the
block or copy a cached surface's GPU copy into the surface's pages, so the
block, its wired pages, and a cached surface with its wire are held, never
handed back to macOS, until the answer comes. A block a create in flight names
is pinned from the kext's check until the create is answered, and a free of it
meanwhile fails busy, so no block changes hands between the check and the
host's carrying out of the message. A context's close is answered under the
same rule, after its objects are dropped and its queues' threads are past
their rings, and the kext frees the process's blocks only on that answer. None
of them is released from the close's post on, and a release the kext took just
before and posts behind the close waits for the close's buffers on the host
too. A close not answered in 10 s keeps everything the process held from every
other process until its answer comes, however late: its blocks with their wired
pages, its holds of shared blocks, its GPU address units and its resource-id
chunks, which the host may still use or map to the context's objects. None of
its blocks is released meanwhile. The late answer lets go of all of it as an
answer in time does. A close never answered, or refused, keeps it for good.

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

**Known pipeline creates.** A pipeline create whose message equals, in every
byte but its header and object id, one the device accepted for the same
process does not wait for its reply (the bundle's `RgfxKnownPipelines.hpp`).
It is posted with Submit's `kRgfxSubmitDeferred` flag, and its state takes the
earlier create's reflection at once. Its `gpuResourceID`, the one value the
host gives each create alone, waits at its first read for the create's own
reply, which the kext keeps until the process takes it (`TakeAnswers`). The
process also takes the replies no one asked for every 32 such posts. Each
create is still a new object with its own resource ID, as Metal's equal
creates are. The first create of each message waits as before, so a refusal is
still known at the create. A refusal is never kept, so its equal create waits
again. Where the kext has no slot left (64 per process, 256 in all), the create
waits as any. The bundle posts this way only where the kext's feature word
has `kRgfxKextFeatureDeferredAnswers`.

Every reason the host refuses a pipeline create is either impossible for an
equal create or known to the process when it posts:
- its object id is new;
- its libraries are held by its functions;
- its bytes are the accepted create's;
- the platform's pipeline resource IDs are not refused (measured to
  1,000,000);
- the ring and the context are checked by the kext before it posts;
- a command buffer's failure fails no other work of its context ("Guest input
  the host refuses", above).

One reason stays: the platform failing to make again what an equal create
needs (read from metal2vulkan's code at the pinned commit). The Vulkan runtime
gives a state of an equal descriptor the live variant of an earlier state (its
shader modules, layouts and `VkPipeline`s) with no driver call, or the objects
of one among the last 16 it released, and the `VkPipeline` it begins at a
create never refuses the create. Where no such variant is left, the create
makes the variant's objects again, which can fail for want of memory; where
the runtime's memo of compiled pipelines has let the pipeline go and its shader
cache misses, it compiles it again, under a compile attempt's 20-second
deadline. That refusal shows at the state's first `gpuResourceID` (as 0) and
in the command buffers that name it, whose draws with it are dropped on the
host ("Guest input the host refuses"), instead of at the create. Only a create
that waits for its answer would close it, at the cost the known creates remove.

| group | messages | `Platform` call |
|---|---|---|
| memory | `HEAP_OVER_HOST_MEMORY`, `HEAP_CREATE`, `BUFFER_CREATE`, `BUFFER_OVER_HOST_MEMORY`, `BUFFER_AT_OFFSET`, `BUFFER_MANAGED_AT_OFFSET`, `HEAP_OVER_PAGES`, `TEXTURE_CREATE`, `TEXTURE_FROM_BUFFER`, `TEXTURE_FROM_IOSURFACE`, `MANAGED_TEXTURE_CREATE`, `MANAGED_BUFFER_CREATE`, `*_DESTROY` | `new_heap_over_host_memory`, `new_heap`, `new_buffer`, `Heap::new_buffer_at_offset`, `new_texture`, `Buffer::new_texture`, drop |
| Managed ranges | `BUFFER_DID_MODIFY_RANGE`, `TEXTURE_REPLACE_REGION` | `Buffer::did_modify_range`, `Texture::replace_region` |
| heap sizes | `HEAP_TEXTURE_SIZE_AND_ALIGN` (answered `SIZE_AND_ALIGN` on the control completion ring; makes nothing) | `heap_texture_size_and_align` |
| state | `SAMPLER_CREATE`, `DEPTH_STENCIL_CREATE`, `*_DESTROY` | `new_sampler_state`, `new_depth_stencil_state` |
| shaders | `LIBRARY_LOOKUP` (the content key of the `.metallib`'s AIR slice: a hit binds the library the device made for any process, and no bytes cross), `LIBRARY_CREATE` and `LIBRARY_FROM_REGION` (the slice's bytes forwarded unchanged on a miss, answered once made), `RENDER_PIPELINE_CREATE`, `COMPUTE_PIPELINE_CREATE`, `*_DESTROY` | the device's library cache, `new_library_with_data`, `new_render_pipeline_state_full` and `new_compute_pipeline_state_full` (the whole descriptor the guest set: vertex or stage-input descriptor, every colour attachment, each stage's linked functions, those made with options or a specialized name by `Library::new_function_with_options`; the Vulkan platform refuses a specialized name, which its runtime's function descriptor lacks) |
| sync | `EVENT_CREATE`, `SHARED_EVENT_CREATE`, `FENCE_CREATE`, `*_DESTROY` | `new_event`, `new_shared_event`, `new_fence` |
| queue | `QUEUE_CREATE` (priority, region offset, vector), `QUEUE_DESTROY` | `new_command_queue` |

`*_OVER_HOST_MEMORY` map guest region blocks (Shared) and the host builds a
placement heap/buffer over them.

**Shared and Managed buffers lie in slabs** (`guest/bundle/src/RgfxSlabs.hpp`).
A device buffer of either mode takes whole 4 KiB pages of a slab, a region
block the bundle holds, so its create and its free call neither the kext nor
the host's import, and the kext's block table holds one entry per slab:
- A Shared slab is one Placement `HEAP_OVER_HOST_MEMORY` of the buffers'
  options, hazard tracking resolved, and each buffer a `BUFFER_AT_OFFSET`
  there, at the heap's GPU address plus its offset. Where the device took
  `MANAGED_BUFFER_SLABS`, a Managed buffer lies in the Shared class of its
  cache and hazard tracking modes: its CPU copy is a Shared buffer of that
  class, made as any (a `BUFFER_AT_OFFSET`, or the `BUFFER_OVER_PAGES` of a
  larger slab of superpages, below), and the buffer a `MANAGED_BUFFER_CREATE`
  over it at an address of its own, so a Managed buffer of any size lies where
  a Shared one of its size does, on every host; its range goes back once the
  copy's destroy follows the buffer's. Without that feature, where the device
  took `MANAGED_AT_OFFSET`, a Managed slab is a Shared Placement heap too, and
  each buffer a `BUFFER_MANAGED_AT_OFFSET` at an address of its own: the host's
  GPU copy, beside a CPU copy placed in the heap at the buffer's range.
  Elsewhere a Managed slab is a plain block, and each buffer a
  `BUFFER_OVER_HOST_MEMORY` over its own range, its CPU copy. The buffer's
  `heap` is nil: the slab is no object of the app's.
- `allocatedSize` is the pages taken, 4096 for a 256-byte buffer, and contents
  and GPU addresses start a page, as both x86-64 Macs' GPUs answer (AMD 5300M,
  Intel 630, measured with `tools/shmem`). Every new buffer reads zeros: a
  create zeroes what of its range is dirty, and a background thread zeroes
  freed ranges first.
- A buffer up to 1 MiB shares a slab of its class, the slabs growing from
  128 KiB to 2 MiB; a larger one takes a slab of its own, which is cached for
  reuse at offset 0 once freed (a sixteenth of the region at most, 2 s idle).
- A class keeps two empty slabs, each for 2 s at most, so a process at rest
  holds only the slabs its buffers lie in. A slab let go destroys its heap,
  then frees its block on the background thread, so no buffer's free waits
  for a `BLOCK_RELEASE`.
- Where the host maps a process's own pages (`PAGES_BUFFERS`, and the kext's
  `PagesBlock`) and makes heaps over them (`HEAP_OVER_PAGES`), a heaped slab
  of 2 MiB is the process's own memory instead: a 2 MiB superpage
  (`VM_FLAGS_SUPERPAGE_SIZE_2MB`, tagged `VM_MEMORY_IOACCELERATOR`), one run
  of guest RAM, wired into a pages block and made one `HEAP_OVER_PAGES`, its
  buffers placed in it as in a region slab's heap. A larger dedicated Shared
  slab is superpages too, more than one run, with no heap: its one buffer is a
  `BUFFER_OVER_PAGES` over the block's view, or a Managed buffer's CPU copy.
  A larger Managed slab of a class of its own (no `MANAGED_BUFFER_SLABS`) is
  the region's, as is a slab where the kernel has no superpage to give (no
  contiguous 2 MiB free, or the wire budget spent) or the kext gives no pages
  block. Each superpage is one kernel call: a slab of several reserves its
  range (no access, at a 2 MiB boundary), then gives back each piece in turn
  and makes a superpage in its hole (`VM_FLAGS_FIXED`), and one refusal gives
  back the superpages before it and the reservation after it, so the slab is
  the region's. A
  call for several superpages that runs out partway takes down past what it
  made on macOS 26.4 and later (xnu's `vm_map_enter` deletes up to the
  request's end after shrinking its size), a deallocation gap whose
  `EXC_GUARD` (`DEALLOC_GAP`) is fatal in every process whose VM guard is on:
  Apple's platform binaries and the hardened browser processes, WebKit's GPU
  process among them (`src/RgfxSuperpages.hpp`). A slab let go
  destroys its heap, releases its pages block (the kext answers once the host
  let go of it, and unwires the pages), then deallocates its pages. On the
  RTX 5080 a heap over a run of guest RAM took 13.4 us and a 4 KiB buffer
  placed in it 1.93 us, where a no-copy buffer over a block's view took
  65-75 us, an import of its own (`slab_heap_costs`, `slab_nocopy_costs`).
- Where no source has memory for a new slab, the process's own empty and
  cached slabs go first, then smaller region slabs are tried down to the
  buffer's size; nil only past that. Any other block of the process's the
  kext has no room for (a texture's CPU copy, a heap, an upload chunk, an
  IOSurface's block) is asked again once those slabs went
  (`RgfxRegion::setReclaim`).

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
16 KiB and 4 KiB pages). Writable Shared memory the kext wires lies in place
(`PagesBlock`, `BUFFER_OVER_PAGES`). Over any other memory (read-only, a Managed
buffer's, or memory the kext or host makes no pages block of) the buffer lies
over a range of a slab of its own options, as a device buffer of them does
("Shared and Managed buffers lie in slabs"): the process's own superpages where
the device takes them, so no app-sized copy lies in the region. `contents`
stays the app's pointer. The range syncs with the memory as the IOSurface copies do
(Decided 14): from the memory before the commit of each command buffer that uses
the buffer, and back to it as one that may write it completes, before its
completion is reported. Each side's bytes that moved since the two last agreed go
to the other, byte by byte, so CPU and GPU writes beside each other both stand
(`guest/bundle/src/RgfxMemoryShadow.hpp`). A GPU write shows in the memory at
its command buffer's completion, not as it runs, and memory mapped read-only is
never written back.

**Managed storage** (Decided 4). A Managed buffer's or texture's CPU copy is
memory of its process's, which the guest maps: a buffer's a Shared buffer of
the slabs where the device takes `RGFX_FEATURE_MANAGED_BUFFER_SLABS`, else a
range of a slab ("Shared and Managed buffers lie in slabs") or a region block;
a texture's a Shared buffer of the slabs where the device takes
`RGFX_FEATURE_MANAGED_TEXTURE_SLABS`, else a region block. The slabs are the
process's own superpages where the device takes them, so no app-sized copy
lies in the region. The host decides the form from the platform's answer of whether the GPU
shares the CPU's memory:
- **Shared memory:** one copy. The buffer is a no-copy buffer over its block,
  and a one-level 2D texture of a linear format a texture over its block's
  Shared buffer; the range messages are checked and do nothing more. A texture
  of any other shape is two copies, as below, and the guest's `getBytes:` of
  it reads its GPU copy, copied on the device: on a GPU that shares the CPU's
  memory Metal's CPU reads of a Managed texture see the GPU's writes with no
  synchronize (the Intel UHD 630 and the M3, dgpu-oracle.md Q2).
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
  - Where the platform refuses the GPU's copy (its memory is full), the object
    takes the one-copy form over its block instead, a texture only where it
    has that form's shape: Metal makes Managed resources past a discrete GPU's
    memory (measured on the AMD 5300M: 12 GiB of Managed buffers, three times
    its 4 GiB, made, used by one command buffer and read back as written).
- A buffer comes from `BUFFER_OVER_HOST_MEMORY` with Managed options; a texture
  from `MANAGED_TEXTURE_CREATE`, which names the copy's Shared buffer and the
  texel block of its format (bytes, width, height). The host reads the copy
  where the buffer's contents lie: its block, its heap's bytes at its offset
  (over a block or over a pages block's guest RAM), or a pages block's view. Both sides compute the
  copy's linear layout from it (`crates/reims-graphics/src/managed.rs`,
  `guest/bundle/src/RgfxManagedLayout.hpp`): levels in order, then slices,
  images and rows of blocks, each row rounded up to 256 bytes.
- In the guest, `replaceRegion:` and `getBytes:` are memcpys of the block, and
  `replaceRegion:` posts its region; neither waits for the host, but for the
  `getBytes:` of a two-copy texture on shared memory, above. A format the
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
- `SURFACE_USE` (block offset, the kext's count of the surface's read-write
  locks, whether the buffer may write it), one for each IOSurface with a device
  cache the buffer uses, before its first encoder: the host fills the cache
  before the buffer's work where it holds an older count (Decided 14, "The
  device cache"). One naming a block with no cache does nothing.
- `SUBMITTED` (no body), where the device has `CB_SUBMITTED`: the buffer is
  committed by `commitAndWaitUntilSubmitted`, and asks for its `CB_SUBMITTED`
  ("Submitted", below). The bundle puts it last, as it commits; the host reads
  it as it takes the buffer, before its other ops.

A command buffer too large for half its queue's submission ring goes out of
line: the whole `COMMAND_BUFFER` message lies in a block of the process's own,
from its queue's pool, and the ring carries a `COMMAND_BUFFER_IN_REGION`
(region offset and length).
- The bundle builds the message on its own heap while it fits half the ring.
- When it outgrows that, the message moves once into a block as large as the
  queue's recent largest message, and doubles from block to block after that.
- The header and the commit's `SURFACE_USE` ops go into room kept before the
  ops, so the commit posts the block with no copy.

The queue's service thread copies the message out of the block once, as it
does a ring's, with the reference's seqno, and the host walks and commits the
buffer from that copy, on its lane, copying it no more. It does not read the
block in place: the process may write its block at any time, and what the host
checked must stay what it carries out (the region's rule, `region.rs`). The
bundle gives the block back
to the queue's pool when the buffer completes. The pool keeps every block given
back and reuses the smallest that holds the next message, so of each size a
queue makes blocks only up to the most of its buffers in flight at once, and a
stream of large buffers asks the kext for none past them, where handing a block
back at each completion beyond a few kept cost a later buffer a block
allocation, a mapping into the process and a grant and release on the control
ring. Measured on an RTX 5080 host, an out-of-line buffer costs what an inline
one does, about 2.1 to 2.5 us a dispatch, up to 2000 in flight either way.
Encoders do not stream as they end yet (Decided 3): that needs a command buffer
in several messages, which the wire does not have. The whole buffer goes at its commit. Metal sets no maximum size, so
nothing is refused for size. A queue holds `maxCommandBufferCount` buffers
(64 by default) alive and not completed. `-commandBuffer` blocks past that, as
Metal's does, which keeps the ring from filling in the common case. A commit
that still finds the ring full rings the doorbell and, where the device's word
holds `SUBMIT_ROOM`, sets the ring's `producer_waiting`, looks once more, and
waits for the host's `SUBMIT_ROOM` record on the queue's completion ring, which
the queue's reader hands to the ring's writer, with a 20 ms backstop. Without
the bit it sleeps 20 µs, doubling to 1 ms, between looks. Each buffer's
completion is expected before the host can see it, under the ring's lock.

Metal splits one command buffer into several kernel submissions (a render pass;
a run of blits or of computes with no event between) — m2v's measured
granularity. The host preserves it so IOGPU error reporting lines up.

**Completion** (per-queue completion ring). A `CB_COMPLETE` record: the
command-buffer `ObjId`, status, GPU start/end time, and on error the
`CommandBufferError` the host maps 1:1 from the IOGPU code the runtime returns
(page fault, hang/timeout, recovery victim, access revoked, invalid resource —
the map m2v gave). The ring's armed/interrupt handshake is the stage-2 one.
The buffer's completion-seen handler posts it, with the values of the shared
events it signalled ahead of it: on metal2vulkan the runtime's thread that saw
the buffer complete runs it, in its queue's commit order, and with no
completed handler on the buffer no handler thread is woken; on Metal, which
tells of a completion only on its completed handlers' thread, a completed
handler.

**Scheduled** (per-queue completion ring). A `CB_SCHEDULED` record (the command-buffer
`ObjId`, at the buffer's seqno; 24 bytes, and the guest reads a record of at least that size)
goes on the queue's completion ring once the host's platform has taken the buffer's work for
the GPU, ahead of its `CB_COMPLETE`: the buffer's submission handler posts it. On
metal2vulkan it runs on the thread that took the buffer's last batch for its `VkQueue`, once
every wait of the buffer is satisfied, before the buffer reads `Scheduled`, and nothing
committed later on the queue runs before the buffer; on Metal, which tells of no point before
scheduling, it is a scheduled handler, run once every wait of the buffer is satisfied and its
work is submitted. A buffer the platform fails before it takes its work runs no submission
handler, and its completion handler posts its `CB_SCHEDULED` first
(`a_buffer_never_submitted_is_scheduled_ahead_of_its_completion`). The guest's
buffer is then submitted (`commitAndWaitUntilSubmitted` returns, before the scheduled handlers
run), becomes `Scheduled`, runs its scheduled handlers, then lets `waitUntilScheduled` return,
as Metal's does (measured on the M3 Max, macOS 27: on an idle queue
`commitAndWaitUntilSubmitted` returns in 0.05-0.1 ms with the buffer `Committed` and its
scheduled handler about 10 us from running; behind a 120 ms buffer it and `waitUntilScheduled`
return in 0.1 ms, `Scheduled`; a buffer that waits for an event nothing has signalled is
neither submitted nor scheduled until the event is signalled, or Metal's timeout fails it after
5 s). A CPU lock of a surface the buffer writes waits for the GPU's writes from the commit on
(§14, "CPU locks"), so a reader handed the surface at submission reads the GPU's results. A
buffer the host could not
carry out has no `CB_SCHEDULED`, and its `CB_COMPLETE` stands for both; so does every
`CB_COMPLETE` from a host that posts none.

**Submitted** (per-queue completion ring), where the device has `CB_SUBMITTED` (`FEATURES_LO`
bit 2; every host sets it). A command buffer committed by `commitAndWaitUntilSubmitted` carries a
`SUBMITTED` op, and the host posts its `CB_SUBMITTED` record (the command-buffer `ObjId`, at the
buffer's seqno; 24 bytes) once, ahead of its `CB_SCHEDULED` and `CB_COMPLETE`: as soon as the
host takes it, before it walks the buffer's ops, where the buffer waits for no event (its
`WAIT_EVENT` ops), the host fills no device cache it uses first (decided at the take from its
`SURFACE_USE` ops, which lead the stream; a fill not yet done counts) and no buffer before it on
its queue waits for one and is not yet scheduled; else at its scheduling. The call returns
there. A guest whose bundle sends no such op never sees the event,
and a bundle sends it only where the bit is set, which it reads in the device info's
`features_lo`. What Metal does, on the M3, the 5300M and the 630
(docs/graphics-kext/dgpu-oracle.md, Q6 and Q7), and the class of each choice:
- Contract (all three agree): the call returns once the buffer is submitted, about 0.1 to 0.4 ms
  behind a long buffer, and not before the signal where the buffer, or one before it on its
  queue, waits for an event. A buffer committed on any queue once the call returned is ordered
  after it for a tracked resource. The host answers as it takes the buffer, once its ticket in
  the commit order is issued, and commits every command buffer the buffer's process sends after
  the answer, on any of its queues, after that buffer's commit; a range message the process
  sends after it waits for that commit too ("What a walk sees", below). Another process's
  buffers do not wait for it: a tracked resource is one process's, and what processes share, an
  IOSurface, is ordered on the GPU by the order of IOSurface accesses ("Surface order", below),
  as a Mac orders it (Q16). metal2vulkan orders by commit across
  queues, and Metal orders behind what it has submitted, which a held queue's buffers are not,
  hence the rule for a queue held by a wait. Tested through the decoder on both platforms
  (`a_later_commit_on_another_queue_sees_a_submitted_buffer_s_writes`).
- Loose (they race): whether the buffer is `Scheduled`, and its scheduled handlers have run, by
  the time the call returns. Ours is the cheapest: still `Committed`, the handlers at its
  `CB_SCHEDULED`, as on an idle queue on all three.
- Loose (later than Metal, inside the envelope): a buffer that waits for an event already
  reached is answered at its scheduling, not at its take.
- Loose (what "submitted" means on the host): Metal answers once it has submitted the buffer;
  ours once the host has taken the message, before it encodes the buffer on its platform, which
  the guest cannot tell apart, as nothing the guest sends after the answer is carried out
  before that commit. Where nothing need be waited for, the answer comes earlier still: on the
  guest's vCPU, inside its doorbell write, before the queue's thread takes the buffer
  ("The answer at the doorbell", below).
- Inside the envelope (a fault): a buffer whose ops the host cannot carry out after it was
  answered fails at its `CB_COMPLETE`, as Metal reports a command buffer's errors at its
  completion, not at its commit.

**Shared events.** The host's event is the one command buffers wait on and
signal; the bundle keeps the value as the process knows it. It only rises, as
Metal's does (measured on the M3 Max: a CPU set or a GPU signal to a lower value
leaves it). A CPU `setSignaledValue:` posts `SHARED_EVENT_SET_VALUE` on the
control ring, which sets the host's (so its GPU waits are released), and raises
the bundle's value. A command buffer's signal of a shared event reaches the
process as `EVENT_SIGNALLED` (`ObjId`, value), which the host posts on the
buffer's queue completion ring, once for each signal, ahead of the buffer's
`CB_COMPLETE` and whatever its outcome. Where the buffer has more work after the
signal (an encoder, or a wait for an event), the host posts it as its own event
reaches the value: it asks its platform for a notification at that value, on one
listener of the device's (`notifyListener:atValue:block:` on Metal; on the
Vulkan platform the runtime sees the value rise as the batch that ends at the
signal finishes), which may come while the buffer runs, or while it waits to be
scheduled. A signal at the buffer's end, and any no notification posted, go
with the buffer's completion; so does a signal of a shared event whose create
the host refused, wherever it lies ("Guest input the host refuses"): the host
holds no event to watch for it, and the walk runs ahead of the GPU, so a post
there could come before the buffer's earlier work is done. So a thread that
waits on the CPU for such a mid-buffer signal before it sets a value the buffer
waits for later waits for good, the one case where ours deadlocks where Metal's
goes on (below). The queue's reader hands each to the device's, where the
process's shared events listen. `notifyListener:` blocks run on their listener's queue with the value
they were registered at, at once where the value has reached it, as Metal's do
(measured). A shared event made at another value is set to it on the host after
its create.

Metal's CPU sees a signal from the middle of a buffer while the buffer runs
(docs/graphics-kext/dgpu-oracle.md, Q18): on the 5300M and the M3 within a few
milliseconds of the GPU reaching it, on the 630 about 100 ms after the commit,
which is later where the buffer runs on. On all three, a thread that waits on the
CPU for a mid-buffer signal before it sets a value the same buffer waits for
later goes on, and the buffer completes. So does ours: the host's notification
comes as its GPU reaches the signal, as the 5300M's and the M3's do. A signal at
the buffer's end reaches the CPU with the buffer's completion on all three, its
listener's block within 0.3 ms of the completed handler, either one first, and
ours comes ahead of the completion record. GPU waits are not affected either
way: another buffer's `WAIT_EVENT` waits on the host's event, which the host's
GPU signals at the point Metal's would.

A command buffer that waits on an event is watched: one not finished 5 s after
its commit is named in the log at WARN, with each event and value it waits for,
and again at INFO once it finishes (at most 32 per device). It is diagnostics
only: Metal waits without end for a value nothing reaches (measured: the buffer
stays `Committed`), and so does the host while the process lives. Once its
context is gone, at its close or a reset of the device, nothing is left to
set or signal its events, which cross no process: the host sets each shared
event past every value and signals each plain event past every value from a
queue of its own, so a buffer still waiting on one runs and finishes. It then
lets go of what it holds (its buffers' GPU addresses among them, which the
rebooted guest picks again after a reset), and the close's answer, which
waits for the context's buffers, comes. A Mac drops such a buffer instead
(measured, the 5300M and the 630 under macOS 26.7.1: a buffer that waits on
its process's own shared or plain event when the process exits or is killed
never runs; its writes to a shared-memory buffer and to an IOSurface never
land, 2 s and 5 s after). The host has committed the buffer to its platform
behind a GPU wait by then, and a platform takes back no committed buffer:
running it is the only end that holds nothing for good, so its writes land
where a Mac's never do.

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
thread takes each command buffer's records whether or not anything waits, and
runs no handler. At a `CB_SUBMITTED` or `CB_SCHEDULED` the buffer is submitted
there, and at its `CB_COMPLETE` what the host is done with goes back: the
queue's slot, the message's block and the inline bytes' chunks. The rest runs
on the queue's completion queue, in ring order: the dispatch queue
`setCompletionQueue:` gave (WindowServer sets one), or a serial queue of the
queue's own, labelled `com.Metal.CompletionQueueDispatch` at user-interactive
QoS. That is where Metal runs a queue's handlers: one queue's one at a time,
two queues' at once (measured on the M3 Max, an AMD 5300M and an Intel 630).
The scheduling makes the buffer `Scheduled` and runs its scheduled handlers,
and the completion runs its write-backs, its status, its completed handlers
and its waiters. `waitUntilScheduled` and `waitUntilCompleted` return once
the handlers have run, as Metal's do. A buffer with no scheduled or completed
handler and nothing to write back (every surface it writes in place, no copy of
app memory) behind none of its queue's records still to run has its records run
by the completion thread itself, which wakes its waiters: one wake where the
queue's would be two, as Metal's waiter for a buffer with no completed handler
is woken by the thread that saw it complete. A wait for a buffer of a queue
from that queue's completion queue, a handler's for its own buffer or a later
buffer of its queue, waits for good, as Metal's does on all three GPUs; a wait
for another queue's buffer is released. `commitAndWaitUntilSubmitted` returns at
the submission, whatever handlers run. Waiters take the ring themselves only
once the thread has ended (its context closed), and then run the records too.

**Enqueue order.** A queue's command buffers go on its submission ring in the
order they were enqueued, whatever order they are committed in, as Metal runs
them (measured on the M3 Max, an AMD 5300M and an Intel 630). The queue keeps
its enqueued buffers that are not yet posted, oldest first; a commit with no
`enqueue` before it enqueues the buffer there. A buffer committed behind one
not yet committed stays `Committed` and unposted, and the commit that readies
the front posts every ready buffer behind it, in order, on its own thread. An
enqueued buffer let go uncommitted stays, holding back the buffers after it,
as Metal's does.

**Surface order**, where the guest took `SURFACE_ORDER` (`FEATURES_LO` bit 11).
On a Mac, a GPU access to an IOSurface committed after another process's
pending GPU access to it runs after that access, where either writes:
read after write, write after read and write after write. Accesses to other
surfaces are not ordered (measured on the MBP16's AMD 5300M, macOS 26.7.1: a
blit of the surface committed just after another process committed a 212 ms
pass over it waited for the pass in all three cases, and read or left what
the order says; a blit of another surface took 2.4-4 ms; the Intel 630
runs one process's work at a time, so it orders all of it). Queues and
processes here reach the host on rings of their own, which the host takes
apart, so the order the host takes buffers in is not the order they were
committed in. The bundle names the order instead, as it posts each buffer:
- Each surface's shared block holds its writers (`RgfxSurfaceWriters`, one
  slot a queue: its latest buffer that writes the surface) and, after them, its
  readers (the same record, the latest buffer of each queue that only reads
  it), which only the bundles read. Every block has both: a copy's planes start
  and end on 256 bytes, so its header, its writers and its readers (152 bytes)
  fit in the page its planes end in, and a block in place holds the header and
  the slots alone, from its start. Every holder asks a block's size from the
  writers' end, as a bundle without readers did.
- A slot names a buffer only while it runs: once the host has finished it, and
  its post has named it, its process empties the slot where no later buffer of
  the queue raised it, as a buffer that named it would wait for nothing. Four
  slots of each record are shared by every process's queues, so a fifth queue
  finds one only once a buffer another holds has finished; one that finds none
  is not named (its process logs it once), so a later write of the surface by
  another queue waits for none of that queue's reads, nor a present or a read
  for its writes.
- The post of a buffer that uses surfaces, just before it goes on its queue's
  ring, reads, for each, the writers of other queues, and where the buffer
  writes the surface the readers too, and names each queue once, at its highest
  seqno, in one `SURFACE_ORDER` op before its first encoder. Once the buffer is
  on its queue's ring it raises its own queue's slot: a writer's for a surface
  it writes, else a reader's. The post, not the commit, is where Metal submits
  the buffer: a buffer committed behind one enqueued before it and not yet
  committed is submitted once that one is, as Metal's enqueue order has it, and
  goes on its ring here then, so it runs after the accesses of other queues
  submitted meanwhile (read from Metal's enqueue order; no oracle has measured
  a held buffer's surface order).
- So each buffer named was posted before the post that names it read the
  slots: the buffers' waits run from later posts to earlier ones, and cannot
  meet in a cycle.
- The host keeps each queue's surface timeline, a shared event. A buffer that
  holds `SURFACE_ORDER` signals its queue's to its seqno as it ends, and waits
  first, on the GPU, for each named queue's to reach the seqno named, except
  where the host has seen that buffer finish, or the queue is the buffer's own,
  whose order is its ring's. A queue the host does not hold, or a seqno past
  every one the queue's ring has published, names no buffer and waits for
  nothing.
- A Mac orders no access after a buffer it has not submitted: one that waits on
  an event not yet at its value, or is behind one on its queue (measured, the
  5300M and the 630 under macOS 26.7.1: another process's blit of the surface,
  and its read-only CPU lock, went on at once while the write waited on its
  event, the blit reading the surface as it was). So the bundle holds such a
  buffer's accesses: one that waits on an event below its value as the process
  knows it (a shared event's value as the host posted it and the CPU set it, a
  plain event's as the host finished the buffers of the process that signalled
  it), or is posted behind a held buffer of its queue, raises its slots and
  names its writes to the kext at its `CB_SCHEDULED`, which the host posts once
  its platform has scheduled it, past its waits. A read posted meanwhile names
  none of it and waits for none of it, and nor does a CPU lock.
- The wait is an order on the GPU and nothing more, as a Mac's is: a Mac answers
  a command buffer it runs after another queue's or process's pending access to
  a surface submitted and scheduled within a millisecond of its commit (the 630,
  and the M3 in a run's first round, within 1 to 5 ms), and completes it after
  that access (dgpu-oracle.md, Q16). So the wait holds back none of the buffer's
  records, nor its queue: it gets its `CB_SUBMITTED` at its take, as a buffer
  that waits for nothing does, and its `CB_SCHEDULED` at its commit, though the
  platform may hold it until the buffer it waits for finishes (the Vulkan
  platform holds a wait for a shared event's value on the host until it is
  reached). So does every buffer committed behind it on its queue while the
  platform holds it, and each buffer's records still go in its queue's order. A
  buffer that waits for an event, or for the host's fill of a device cache, or
  is behind one, keeps its records at its scheduling.
- Work the host starts off a buffer's own GPU timeline waits for the same
  buffers on the CPU first: a video session's decode runs on a queue of its own
  as the host walks the buffer, so its output lands only after the reads and
  writes of other queues the buffer names.
- What will not signal is signalled for it: a buffer stood in for signals from
  its stand-in, and one with no stand-in from a command buffer of its own on its
  queue. A queue's destroy, or its context's close, signals the last value a
  buffer waits for of it. A buffer the host has taken that holds no
  `SURFACE_ORDER` signals where another queue's buffer waits past what the
  queue has signalled.

Measured on the lane's RTX 5080 laptop host with a macOS 26.6.1 guest, one boot
each of the effort's tip before the order (18fd065e5) and with it: the sweep's
`video.h264.reader`, an AVAssetReader of frames VTDecoderXPCService decodes, failed
35 of 40 runs, then none of 40; vlprobe's AVAssetReader of 420v frames read
another frame than its sample's in 96 of 100 runs (39 an earlier one, 56 a
later one), then in none of 100. A CPU lock needs nothing more: on the MBP16's
AMD 5300M and Intel 630 (macOS 26.7.1), once a command buffer that writes a
surface is scheduled, every lock of the surface waits for it, read-only or
read-write, in its process or another, and then reads what it wrote; a lock
taken before the buffer is scheduled may not wait. The kext's locks wait for the
writes named from the commit on, which is inside that.

**Commit order**, where the guest took `COMMIT_ORDER` (`FEATURES_LO` bit 18).
Metal orders a tracked resource across a process's queues by the order the
command buffers were committed in: a copy committed on a second queue right
after the first queue's `commit` of a 30 ms kernel that writes its source read
what the kernel wrote in every run on the AMD 5300M, the Intel 630 and the M3
(dgpu-oracle.md, Q6). The platform orders a tracked resource across its queues
by the order the host commits to it, so the host has to commit in the guest's
order. A plain `commit` returns once the buffer is on its queue's ring, and
each queue's ring is taken on a thread of its own, so a buffer the guest
committed later on another queue could reach the platform first (before this
op, the oracle's two-queue row raced in 4 of 15 runs on the lane, each with
every word stale). The bundle names the order as it posts each buffer:
- The process keeps its command queues in a list, each with the seqno of its
  latest buffer on its ring (`posted`), raised once the buffer is there.
- Before a buffer goes on its ring, it names each other queue of the process
  whose `posted` rose since this queue last named it, at that `posted`, in one
  `COMMIT_ORDER` op before its first encoder; then it goes on its ring, and its
  queue's `posted` rises.
- So each buffer named was on its ring before the post that names it read
  `posted`: the names run from later posts to earlier ones and cannot meet in a
  cycle. A buffer committed but held behind an uncommitted one on its queue is
  not on its ring, and is not named, as Metal does not submit it either.
- The host keeps two marks for each queue: the highest seqno whose buffer took
  its place among the commit order's tickets (its begin), and the highest whose
  buffer was committed, stood in for or failed, raised in the queue's order as
  each buffer's job ends. A buffer that holds the op takes its own place among
  the tickets only once each buffer it names has begun, and is committed only
  once each named queue's commit mark reaches the seqno named, after its turn
  in its queue and the tickets. The first wait keeps the tickets from crossing
  the names. Without it, a queue's service thread could take a buffer that
  names another queue's buffer, then a later buffer that asks for its
  `CB_SUBMITTED` and takes a ticket, all before the named buffer's begin. The
  named buffer would then wait for that ticket, the ticket's buffer for its
  turn, and the naming buffer for the named one. With it, every wait of a
  commit is for a buffer begun before it. An answer on a doorbell's thread,
  which waits for nothing, is left to the service thread where a named buffer
  has not begun.
- A name of its own queue, of another process's queue, of a queue the host
  does not hold, or past every seqno the queue's ring has published, waits for
  nothing. The waits hold no lock. A queue's destroy, its context's close and
  a reset of the device end them, and while a reset is under way none waits,
  so a process that names a cycle holds up only its own queues, until it
  closes.
- Nothing else changes: no GPU wait is added, the buffer's records go as they
  did, and the platform orders what the buffers share by the commit order.

**Left for later** (extends the same op families, not the transport):
command-buffer/queue descriptors and `errorOptions`; labels and debug groups
answered guest-side; residency sets (`useResource`/`useHeap` are in);
`MTLIOCommandQueue`; Metal 4.

**Where the decoder lives.** In reims-vgpu-core, generic over `P: Platform`,
beside the Apple-lane decode it mirrors (same handle machinery, descriptor
structs and `Platform` calls). The reims-graphics crate stays the transport
(rings, region, registers); the decoder consumes ring bytes and drives
`Platform`.

**What a walk sees.** No lock of the device's is held across a command buffer's
walk and commit. Each context's objects are in a table of their own, each
message is carried out on a decoder of its own, and every lock is held for a
short section (the list is in `crates/reims-vgpu/src/graphics/decode/mod.rs`).
Ring 0 goes on while a queue walks, and so does every other queue, and the
queue's own later buffers are walked beside it. Under the single lock this
replaces, the decoder's lock was held from a command buffer's take to its
commit, so each other message came wholly before the take or wholly after the
commit. Each outcome the new design allows is one of those two:
- **Objects.** A command buffer names objects as its context's table was when
  its first lookup (its queue, at its begin) took its view of the table. A
  create, destroy or close carried out on another thread after that is not
  seen, and each such change began after the buffer's take, so under the lock
  it would have waited for the commit. An object destroyed during the walk is
  named whole, and let go of once the walk's view closes, after the commit, as
  one destroyed after the buffer. An object made during the walk is not named:
  the op naming it faults, as it would before the create. An id destroyed, made
  again and destroyed again during the walk is named as the object it was at
  the begin. A close during the walk empties the table at once and the walk
  keeps naming its objects; the buffer took its context's drain before its
  view, so the close is answered after the buffer's completion handler. The
  begin comes after ring 0 has carried out every message of its context's
  published before the buffer was taken (`reims_graphics`'s queue gate, "A
  buffer's wait on ring 0"), so a buffer finds every create its process sent
  before it. Tested by
  `a_walk_names_what_its_table_held_at_its_begin`,
  `a_walk_names_nothing_made_after_its_begin` and
  `a_close_during_a_walk_leaves_it_its_objects` (the decoder's `cmdbuf`
  tests), and through the device by `ring_0_is_carried_out_while_a_walk_is_in_flight`,
  `a_close_during_a_walk_is_answered_after_the_buffer` and
  `a_queue_s_buffer_runs_beside_another_queue_s_walk` (`graphics/tests.rs`).
- **The commit order.** A buffer answered `CB_SUBMITTED` at its take holds a
  ticket, issued before the answer is posted, until it is committed (or its
  stand-in is); one answered at its doorbell holds the answer's. Every command
  buffer notes the tickets issued at its begin and is committed only once each
  of its own context's is given back. A buffer its process sends after an
  answer is begun after its ticket, so it is committed after the answered
  buffer, as under the lock the answered buffer held through its commit. A
  buffer begun before the answer does not wait: the guest sent it with no
  answer in hand, so under the lock it could have been taken first. Another
  context's buffer waits for no ticket of this one, though the lock held it
  too: it names none of the context's objects, and what it shares with it, an
  IOSurface, is ordered on the GPU (`SURFACE_ORDER`) and by the device cache's
  users, not by commit, as a Mac orders tracked resources by commit only
  within a process (dgpu-oracle.md, Q6 and Q16). So one process's answered
  buffer whose walk is long (a cold make in a draw) holds up only its own
  process's commits, where under a device-wide order every process's buffer
  begun meanwhile, the compositor's among them, waited for it. A ticket
  is issued after its buffer's begin, and a buffer waits only for older
  tickets, so the waits end. Tested by
  `a_buffer_sent_after_an_answer_is_committed_after_it`,
  `a_held_walk_answered_its_submitted_holds_only_its_process_s_later_commits`
  (`graphics/tests.rs`), and the sequencer's
  `a_commit_waits_only_for_the_tickets_issued_before_its_begin`,
  `a_commit_waits_for_its_own_context_s_tickets_alone` and
  `tickets_given_back_out_of_order_hold_through_the_oldest`.
- **Range messages.** `BUFFER_DID_MODIFY_RANGE` and `TEXTURE_REPLACE_REGION`
  reach the platform's next submission. One is carried out once every ticket
  its context held when it came is given back, so a buffer answered before the
  message was sent does not see its bytes, as under the lock, where the message
  waited for that buffer's commit. Another context's tickets are not waited
  for: its buffers cannot name the object, which is its own process's. A buffer
  of the same context not answered at its take is not waited for either: the
  guest sent the message with no answer from it, so under the lock the message
  could have come before its take, and the buffer may see the bytes, as it may
  on Metal. Ring 0 does not wait for those tickets: the message is checked
  there (a bad reference or a range past its object is found at once), and its
  copy and its platform call are made where the last ticket is given back,
  after that buffer's commit, a context's range messages one at a time in the
  order they came. Every command buffer of the context begun after it came is
  committed after it, as one begun after ring 0 carried it out was, and it
  holds its context's drain, so the context's close and its blocks' releases
  are answered after it. It reads the block when it runs, not when it came: a
  CPU write made after the message and before the next commit may reach the
  GPU with it, as Metal's copy of a `Managed` resource is made at the next
  submission. Tested by `a_buffer_sent_after_an_answer_is_committed_after_it`
  (its range messages) and the sequencer's
  `a_range_message_is_carried_out_once_its_context_s_tickets_are_given_back`.
- **Device caches.** A command buffer counts among the users of each cached
  block it uses from its begin to its commit. A fill waits for every user, and
  a copy out of a cache (a download, a release, the block named again) for every
  user that may write it, so each copy comes before a user's begin or after its
  commit, as under the lock. A download and a release do not wait on ring 0's
  thread: each is made in the finish of the last buffer that may write the
  cache begun before it was asked, after that buffer's commit, in the order
  they were asked, and answered once its copy is done, as before. A buffer
  begun after one was asked waits at its begin until it is made, as under the
  lock it would have been begun after it,
  and nothing of ring 0's names the block meanwhile that a copy changes: a
  download leaves the block's buffers as they are, and the kext names a
  released block again only once its release is answered. The block named
  again still waits there, as the buffers ring 0 makes over the block after it
  lie over what it leaves. A buffer counts its blocks in block order, a copy
  waits only for buffers begun before it, and one past its begin waits for
  nothing a fill or a copy holds, so the waits end. Tested by
  `a_fill_waits_for_the_buffers_using_the_cache`,
  `two_queues_take_one_fill_and_a_download_waits_for_both` and
  `a_copy_out_of_a_cache_is_made_as_its_writers_finish` (`surface_cache`).
- **The device's reset.** A reset ends every queue at once and is passed to
  the host on the control thread, before anything the guest publishes after
  it. Under the lock, the host's reset waited for the message holding the lock,
  and each message after it found its queue's poster stale and was not carried
  out. Here each message carried out off the control thread (a queue's command
  buffer, create or destroy) holds a gate shared, from before it asks whether
  its poster is stale to its end, and so does each compile's and library
  make's store. The host's reset holds the gate alone. So the reset waits for
  every message that passed its check, each of which could have run whole
  before the reset under the lock. A walk in flight names its objects and
  commits, and then every object goes, none outliving the reset; a message
  after the gate finds its poster stale. Only the reset makes ring 0's thread
  wait for a walk, as the lock did. Tested by `a_reset_during_a_walk_comes_after_it`
  and `a_reset_drops_every_object_the_guest_made` (`graphics/tests.rs`), and
  by the decoder's fuzz (`decode/fuzz.rs`).
- **A queue's own order.** A queue's service thread begins each of its
  command buffers in the order the guest published them and hands
  the walk and the commit to the queue's lane (`decode/lane.rs`), whose threads
  walk several at once, so one process's busy queue is not bound to one
  thread. Under the lock a queue's next buffer was taken after the commit of
  the one before it. Here each buffer gets its turn in its queue's lane at its
  begin, before anything of it can fail, and its commit, its stand-in's commit
  and the failure posted at once for it wait for every turn before its own, so
  they come in the order the buffers were taken, as under the lock. A walk
  encodes into its own command buffer, which the platform runs in commit order:
  Apple's Metal guide lets a queue's buffers be encoded on several threads and
  run in the order they are committed, and metal2vulkan resolves barriers,
  layouts and fences at submit, in commit order. Four things a buffer does
  before its commit are seen beyond it, and each keeps the lock's order:
  - its hold of its queue: a buffer that waits for an event, or for the
    host's fill of a device cache, holds its queue from its begin, not its
    commit, so a buffer after it that asks for its `CB_SUBMITTED` is answered
    at its scheduling, as behind one already committed;
  - a video encoder's work, which its session starts as it is walked: a buffer
    that holds one is walked only in its turn, and one whose turn is next at
    its begin is walked and committed on the service thread itself, where it
    waits for no buffer of its queue, rather than handed to a lane thread that
    would first be woken or made for it (so is a short buffer, its message of
    2 KiB at most, whose commit would wait for nothing at its begin: its turn
    next, and no buffer of its process it is committed after still to commit;
    on the lane, each buffer of a drain paid a thread's wake for its walk and
    another for the turn its commit waited on);
  - its signal of its queue's surface timeline ("Surface order"), and the
    release that stands in for one of a buffer that does not run: each is
    decided in the buffer's turn, after its walk, since the order reads a
    queue's buffer taken past a seqno as every buffer up to it taken, and a
    waiter that saw a later buffer taken while an earlier one still walked
    would wait for nothing;
  - its answer at its take, posted at its begin, before the buffers before it
    on its queue commit: everything its process sends after the answer is
    committed after the buffer (its ticket), which commits after them, so
    nothing sent later sees the difference, and those buffers are `Committed`
    or `Scheduled` at the answer by a race, as Metal's are.

  A turn passes as its buffer's work ends, and a buffer waits only for turns
  and tickets of buffers begun before it. A lane's threads take its jobs in
  the order they came, and every lane with work has a thread of its own, so
  the oldest unfinished buffer of each lane is on a thread, and so is the
  oldest of the device: the waits end, by induction on the begin order. A
  buffer carried out on its service thread is its lane's oldest unfinished
  one, and the lane gets no job while it runs. A
  lane's own thread goes after a second with no job; a job that comes while
  every thread of the lane is busy gets another, up to four a lane, only where
  the device-wide budget of twice the host's cores has one free, which a lane
  never waits for. So 2048 busy queues hold one thread each and the budget's
  at most. Tested through the device by
  `a_queue_s_buffers_are_walked_at_once_and_committed_in_order`,
  `a_buffer_behind_one_that_waits_and_is_still_walked_is_answered_at_its_scheduling`
  `a_video_buffer_is_walked_in_its_turn` and
  `a_lone_video_buffer_is_walked_on_its_queue_s_service_thread`
  (`graphics/tests.rs`), and by the lane's
  `a_turn_waits_for_every_turn_before_it`,
  `a_turn_is_next_once_every_turn_before_it_has_passed`,
  `a_lane_runs_its_jobs_at_once_on_the_budget_s_threads`,
  `a_lane_with_no_budget_left_runs_its_jobs_on_its_own_thread_in_order` and
  `busy_lanes_hold_one_thread_each_and_the_budget_s`.
- **A command buffer in parts.** Where the platform starts a command buffer's
  GPU work only at its commit (the Vulkan platform's runtime records it and
  submits it then), a long buffer's walk and its GPU work run in series:
  WindowServer's composite of a Liquid Glass desktop walked 2.6 to 3.5 ms and
  then ran 1.6 to 3.3 ms on an RTX 5080 Laptop GPU, each frame, against a
  refresh of 8.33 ms. So the walk commits such a buffer in parts
  (`decode/split.rs`): at an encoder's end, once the part walked since the last
  took 250 µs, it hands the platform that part (`CommandBuffer::split`), which
  commits it on the buffer's queue and answers a command buffer of the queue
  that goes on from it, and the walk goes on in that one. On the M3 Max (MoltenVK)
  a frame of 60 passes then completed in 6.0 ms from its message's take, where
  it took 9.5 ms whole (`decode/split.rs`, "What it gains"). A part is committed
  only where the whole buffer could be committed then, waiting for nothing:
  its turn is next, every ticket and range message before its begin is done,
  and the buffers its `COMMIT_ORDER` names are committed. Every wait of the
  last commit is met at the first part, so the walk never waits between
  parts. The parts run in their order on the queue, as one buffer's encoders
  do. The buffer's handlers, records, ticket and commit marks are its last
  part's, and its outcome is all of them: the first failed part's error, the
  first part's GPU start, the last part's GPU end. While a buffer's parts are
  out, from its first part's commit to its last, no other command buffer of
  its context is committed (each waits at its commit, never at its walk), and
  the context's range messages that come are carried out after its last
  commit, so each other message comes wholly before the buffer's commit or
  wholly after it, as under the lock. A buffer splits only while no other
  commit or range message of its context is under way, and only where it
  waits for no event, is behind no buffer of its queue that does, is not a
  video buffer, and its queue holds room for another command buffer. A buffer
  whose walk faults after a part has its failure at its stand-in's
  completion, as one whose walk faults whole; the parts committed before the
  fault ran, as the encoders before a fault of Metal's GPU run. The Vulkan
  platform keeps a buffer whole where a part would answer otherwise than the
  whole (`gpu/command.rs`, "A command buffer in parts"); the Metal platform
  keeps every buffer whole, as Metal's commit starts the GPU at once. Tested
  by `a_buffer_walked_long_is_committed_in_parts_with_one_outcome`,
  `a_buffer_is_committed_whole_where_its_parts_would_wait`,
  `another_queue_s_commit_waits_for_a_buffer_s_last_part` and
  `a_walk_that_faults_after_a_part_fails_at_its_stand_in`,
  `a_split_after_a_mid_buffer_signal_keeps_the_signal_before_the_cut` and
  `a_split_after_a_skipped_encoder_answers_as_the_whole` (the decoder's
  `cmdbuf` tests), and the gate's own (`decode/split.rs`).
- **What still waits on ring 0's thread:** a block named again, for that
  block's writers; a present's fill, for that block's users begun and not
  finished (not for the fill itself: the fill is committed there, and the
  present waits for it in the present queue, shown at the fill's completion);
  and the device's reset, for every message in flight. A context's close waits for no
  walk: `reims_graphics` fences each of its queues, which waits only for a take
  of the queue's submission ring or a post onto its completion ring under way,
  and the close is answered once its buffers have finished
  (`a_close_during_a_walk_holds_up_nothing_on_ring_0`). A doorbell's take tries
  every lock it takes, and takes nothing it would have to let go of.
- **The answer at the doorbell.** A queue's doorbell write, on the guest's
  vCPU, may answer `CB_SUBMITTED` to the buffer at the head of the queue's ring
  before the queue's thread wakes to take it (`reims_graphics`'s
  `Device::mmio_write_taking`), where the buffer would get it at its take and
  nothing need be waited for or allocated: the vCPU copies the message onto its
  stack, reads its ops where they lie, and tries every lock it takes. It reads
  the ops only where the stream's last op is the buffer's `SUBMITTED`, where the
  bundle appends it at the commit, so a buffer that does not ask costs the vCPU
  no scan of its ops. A buffer that does not ask, or whose ask is not its last
  op, or that uses a device cache, waits for an event or another queue's
  buffer, sits behind a held buffer, or finds a lock busy is left to the
  queue's thread, as before. The answer takes the buffer's turn in its queue's lane and its ticket
  in the commit order together, with its mark (every ticket issued before its
  own), keeps them on the queue (`cmdbuf`'s `Answered`), and posts the record.
  The queue's thread then takes the buffer without waiting for ring 0 again
  (the vCPU found ring 0 carried out as far as the buffer waits for), and its
  begin takes the turn, the mark and the ticket in place of its own, so its
  commit and every commit it orders come where they would had it been answered
  at its begin. Its later messages on ring 0 wait for nothing more of it than
  before. A buffer never begun after its answer (its queue gone, the device
  reset) lets them go with its queue's entry, and a buffer the platform refuses
  at its begin fails in its turn. The vCPU neither allocates nor frees, as
  the library's allocator may purge pages in either call: the platform's
  command buffer and every other allocation are made on the queue's thread.
  The answer is on the queue's completion ring before the doorbell write
  returns, so the bundle's post looks for it there as the write returns
  (`RgfxReplyReader::holdsSubmitted`, which takes no record): a buffer
  answered at its doorbell is submitted there, and
  `commitAndWaitUntilSubmitted` returns with no interrupt and no wake of the
  completion thread on its way. That thread takes the record later, in ring
  order, and finds the buffer submitted already. Tested by the bundle's
  `test_submitted_at_the_doorbell_wakes_no_thread` and
  `test_reply_reader_holds_submitted` (`tests/shell_tests.mm`).
  A buffer the doorbell leaves to the queue's thread because ring 0 is behind
  (a create of its process's not yet carried out) is offered the same way on
  the lane's thread whose carrying out ends the queue's thread's wait, before
  that thread is woken (`reims_graphics`'s queue gate, its offers), so its
  `CB_SUBMITTED` no longer waits for the queue's thread's wake
  (`a_buffer_behind_its_process_s_create_is_answered_on_the_lane_s_thread`).
  Tested by `a_doorbell_s_answer_allocates_nothing` and
  `a_doorbell_write_answers_a_command_buffer_on_its_thread`
  (`graphics/tests.rs`), the `cmdbuf` tests
  `a_buffer_is_answered_at_the_doorbell_only_where_it_need_wait_for_nothing`,
  `a_buffer_answered_at_the_doorbell_is_carried_out_as_one_decoded`,
  `a_buffer_begun_after_an_answer_is_committed_after_the_answered_buffer`,
  `a_reset_between_an_answer_and_its_begin_lets_the_answer_go` and
  `a_refused_make_after_an_answer_fails_in_turn`, the decoder's fuzz
  (`an_answer_at_the_doorbell_keeps_the_order_of_a_take`), and the device's
  `an_answered_command_buffer_waits_for_no_later_ring_0_message`.

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
- **Trusted-bundle gate (with SIP on).** Before it loads a GPU bundle from
  `/Library/GPUBundles`, Metal asks the system whether that path is trusted,
  and the answer is yes only for a path carrying the particular SIP "rootless"
  storage class the OS gives the bundles its own kext-staging writes there (or
  for a path sealed on the system volume). A bundle merely copied onto the data
  volume does not carry it, so with SIP on Metal declines to load it and
  WindowServer, finding the accelerator registered, crashes and restarts in a
  loop.
  Our install kext (`guest/rgfxgate`, a Lilu plug-in) therefore writes the
  bundle to the data volume and gives the whole tree that exact storage class
  and the matching restricted flag from kernel context, below the file-system
  MAC layer, so only our bundle's own paths are tagged and SIP is relaxed for
  nothing. The unsigned bundle's executable still has to pass library
  validation when WindowServer and the Metal helpers map it; the same kext
  answers the library-validation, mmap and signature MAC checks for our bundle
  by its code-directory hash, leaving every other binary's signing rules
  untouched. The signature check, the wall that gates the other two, also
  requires the executable's path to be under `/Library/GPUBundles`, so the
  exemption follows the installed bundle and a byte-identical copy of it placed
  elsewhere is not trusted — the same pinning the install writes carry. No
  boot-args and no SIP relaxation are involved — the guest boots
  with SIP fully enabled (measured on msi, x86-64 macOS 26.6.1: desktop
  composites on our Metal device and Safari WebGL renders, `csrutil` enabled).
  `/Library/GPUBundles` itself comes with every install: the data-volume
  template on the sealed system volume holds it, already restricted and of
  that storage class (measured: the x86-64 26.6.1 guest and a 27.0.1 host).
  The install kext waits for the data volume to mount (`/Library/Preferences`
  appears) and creates the container, with the same class, only where it is
  missing.
  - *Checklist (per supported OS).* The three load hooks and the three install
    hooks are Lilu routes of six XNU MAC dispatchers
    (`_mac_file_check_library_validation`, `_mac_file_check_mmap`,
    `_mac_vnode_check_signature`, `_mac_vnode_check_{create,open,truncate}`),
    whose argument lists `RgfxGateHooks.cpp` hard-codes. Routing is by symbol,
    not by raw bytes, so a moved patch site is not the risk; a changed
    *prototype* is. When a new `KernelVersion` is added to the supported range
    (Info.plist, BigSur..Tahoe), re-verify each of the six prototypes against
    that OS's KDK before shipping it, and confirm `_csfg_get_cdhash` and
    `_csblob_get_cdhash` still resolve (a miss on either now logs loudly and
    leaves the gate inert rather than bricking silently).
- **Bundle updates.** The marker beside the executable (below, "Registration
  waits for the bundle") carries the kext's wire version, and the bundle is the
  kext's own payload, so a newer kext whose executable or wire version differs
  replaces the installed bundle. The replacement is atomic: the whole new tree
  is built and tagged in a sibling staging directory (`NAME.bundle.new`), then
  put in place with a single rename — an existing bundle is exchanged with it
  (`VNOP_RENAMEX`, `VFS_RENAME_SWAP`) and the old tree, swapped aside, dropped;
  where there is none the staged tree is moved in. A reader (WindowServer, a
  Metal helper) therefore sees either the whole old bundle or the whole new
  one, never a half-written one, so an upgrade that keeps the wire version
  cannot leave a torn executable mid-boot (the "RgfxGate rewrite race"). The
  container is never unlocked, so SIP is not widened around it even mid-upgrade.
  A bundle that is byte-current but untagged — staged by a tool under relaxed
  SIP and then first seen on a SIP-on boot — is detected and tagged in place
  without a rewrite. The test is the bundle directory's storage class and
  restricted flag together: a bundle copied into the restricted container is
  given the flag but not the class, and the class is what Metal reads.
  Userspace cannot remove the class even with SIP off, so only the kext writes
  or clears it. A current, tagged bundle is left as it is, and the boot writes
  nothing. The accelerator, in turn, registers only once RgfxGate has signalled
  that its install pass has ended, so it never registers against a tree
  mid-swap.
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
- **Load policy.** Loading our unsigned bundle into WindowServer, the Metal
  compiler helpers and sandboxed apps is done with SIP and library validation
  left on, by the install kext's two gates above — the rootless storage-class
  tag that makes the path trusted, and the code-directory-hash answer to the
  library-validation and signature MAC checks. It is a concern of our kext, not
  a boot-args relaxation.

## Stage 3 accelerator kext (scaffold)

`RgfxAccelerator` (`guest/kext/src/RgfxAccelerator.{hpp,cpp}`) is a second
IOService in the one kext, on the same PCI device as the Stage-2 framebuffer:
the framebuffer matches in category `IOFramebuffer`, the accelerator in category
`IOAccelerator`, both children of the one `IOPCIDevice`. Its `Info.plist`
personality carries `MetalPluginName = reims-graphics` (the bundle
`/Library/GPUBundles/reims-graphics.bundle`) and `MetalPluginClassName
= RgfxMetalDevice`, which is how Metal (`IOServiceMatching("IOAccelerator")`)
finds and loads the bundle. The accelerator maps BAR 0 for the doorbell and
holds BAR 2 to map each process's part of it in; it touches no pixels and no
control ring (the framebuffer owns those).

**Registration waits for the bundle.** Where the accelerator is registered and
its bundle is not on disk, every WindowServer aborts in CoreDisplay as it
brings the display up, and launchd starts another until it gives up: the
screen stays black. That is every boot before the bundle is installed: the
install boot, Recovery, the installer. Where no accelerator is registered,
WindowServer composites in software on the framebuffer. Measured on msi
(x86-64 macOS 26, no bundle): an accelerator that does not start leaves
WindowServer up with a console session and the desktop drawn, the framebuffer
still naming it in `IOAccelTypes`; a registered accelerator still aborts it,
the framebuffer naming it or not. So the accelerator starts as before but
registers only once its bundle is on disk:
- From its start, a thread call checks every 100 ms for the bundle's
  executable, `/Library/GPUBundles/NAME.bundle/Contents/MacOS/NAME`, NAME its
  `MetalPluginName`, and for the marker of the kext's wire version beside it,
  `Contents/Resources/wire-version-N` ("The wire's version"): nothing before
  the root file system is mounted (`vfs_rootvnode`), then a `vnode_lookup` of
  each. An executable without that marker is a bundle of another version, and
  counts as none. `/Library/GPUBundles` is on the
  data volume, which launchd mounts early in boot, well before it starts
  WindowServer, so with the bundle installed the accelerator is registered
  before WindowServer looks for a GPU.
- WindowServer's first server connection to the framebuffer settles it.
  `ReimsGraphics::newUserClient` has the accelerator check once more before
  IOFramebuffer makes the connection, with no lock held, and register at once
  where the bundle is there; WindowServer opens that connection before any of
  the accelerator's user clients. No check runs after it.
- **RgfxGate's install pass comes first.** Where RgfxGate installs the
  bundle, nothing on disk counts while its install pass is under way: an update
  that keeps the wire version would find the old bundle and its marker before
  the pass drops the marker and rewrites the files, and WindowServer could
  then load a bundle mid-rewrite. RgfxGate publishes `RgfxGateInstallDone` on
  IOResources, false from the pass's start and true once the pass has ended,
  on every path (the installer and Recovery, a data volume that never mounts,
  a failed write). The check looks again while it is false. The settling
  connection waits up to 5 s for it, and leaves the accelerator unregistered
  for the boot where the pass is still under way. Without RgfxGate there is no
  property, and the check counts what it finds.
- **A bundle installed later takes effect at the next boot**: a running
  WindowServer never sees a GPU appear under it. The guest's install puts the
  bundle in place and reboots.
- Unregistered, the accelerator refuses every user client
  (`kIOReturnNotReady`): a task can still reach it by the framebuffer's
  `IOAccelTypes` path.
- Neither the check nor its wait runs on the kext's start or the
  framebuffer's console path.

`RgfxAcceleratorUserClient`
(`guest/kext/src/RgfxAcceleratorUserClient.{hpp,cpp}`) is **our own** transport,
not Apple's IOGPU user-client ABI: the context's window ("Processes map only
their own"), where the bundle's `RgfxRegion` / `RgfxRingWriter` write into real
device memory, the device info (selector 0x10b), and
`externalMethod(kRgfxAccelMethodDoorbell)`, which writes a ring id to
`RGFX_REG_DOORBELL`. These ids/selectors are defined in `RgfxUserClientABI.hpp`,
which the kext and the bundle both include;
the bundle's wire handshake opens them with `IOServiceOpen` /
`IOConnectCallStructMethod` / `IOConnectCallScalarMethod`.

**The accelerator's user client: its numbers.** This table gives out every
number of the 'RGFX' user client. A new selector takes the next free number
here first. A selector or a flag that a kext may lack gets a kext feature bit,
which the kext sets in the second word `Features` answers once it carries it
out, and the bundle uses it only where that word holds the bit. The numbers
and the kext feature bits are in `guest/kext/src/RgfxUserClientABI.hpp`, which
the kext and the bundle include.

| number | name | kext feature bit | what |
|---|---|---|---|
| doorbell 0 | `Doorbell` | | writes a ring's id to `DOORBELL` |
| doorbell 1 | `Consumed` | 1 << 3 `ConsumedDoorbell` | reserved: a completion ring's room doorbell (`CONSUMED`) |
| 0x100 | `OpenContext` | 1 << 5 `ContextSpan` (its sixth output) | the context and its window; five outputs, or six with its GPU addresses' span |
| 0x101 | `Alloc` | | a region block |
| 0x102 | `Free` | | frees a block |
| 0x103 | `QueueCreate` | | a command queue; 6 inputs, or 7 with the flags wanted |
| 0x104 | `QueueDestroy` | | closes a queue |
| 0x105 | `Submit` | 1 << 4 `DeferredAnswers` (its deferred flag) | one lifecycle message onto ring 0 |
| 0x106 | `ResourceIds` | | a chunk of texture resource ids |
| 0x107 | `WaitCompletions` | | waits for the client's completions |
| 0x108 | `SurfaceScanout` | | the scanout an IOAccel surface is drawn into |
| 0x109 | `SurfaceBlock` | | an IOSurface plane's shared block |
| 0x10a | `SurfaceBlockRelease` | | lets go of a shared block |
| 0x10b | `DeviceInfo` | | the device info and video caps |
| 0x10c | `SurfaceWrites` | | the surfaces a command buffer writes |
| 0x10d | `SurfaceRetire` | | a queue's writes up to a seqno retire |
| 0x10e | `PagesBlock` | 1 << 0 `PagesBlock` | a block of the runs of a process's own pages, wired, for its no-copy buffers |
| 0x10f | `Features` | | the device's feature word and the kext's |
| 0x110 | `SubmitBatch` | 1 << 1 `SubmitBatch` | reserved: lifecycle messages in one call |
| 0x111 | `WakeCompletions` | 1 << 2 `WakeCompletions` | reserved: wakes the client's completion waiters |
| 0x112 | `TakeAnswers` | 1 << 4 `DeferredAnswers` | hands a process its held pipeline-create answers ("Known pipeline creates") |

The kext sets 1 << 0 `PagesBlock`, 1 << 4 `DeferredAnswers` and 1 << 5 `ContextSpan`; no other bit yet.

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

- **Stage 3 bundle placement — RESOLVED:** the bundle sits in
  `/Library/GPUBundles` and our accelerator kext's personality carries
  `MetalPluginName`/`MetalPluginClassName`. The library-validation boot state
  is settled: the install kext gives the bundle the rootless storage class so
  Metal trusts its path, and answers the library-validation, mmap and signature
  MAC checks for it by its code-directory hash, so WindowServer composites on
  our Metal device and Safari WebGL renders with SIP fully on and no boot-args
  ("Trusted-bundle gate", above; measured on msi, x86-64 macOS 26.6.1). Open:
  the exact IOGPU accelerator user-client ABI our kext exposes, settled by a
  guest experiment.
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
