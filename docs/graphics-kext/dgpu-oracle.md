# The dGPU oracle: what an x86 Mac's Metal leaves loose

Every Metal rule the reims-graphics lane imitated was first measured on an Apple M3 Max, a
unified-memory tile-based GPU. The lane's guests are x86-64 macOS, whose apps and frameworks
were written against Intel Macs: an AMD discrete GPU, an Intel iGPU, or both. This report
measures those GPUs as a black box, with probes of our own, and compares each answer with what
our stack enforces. Where the x86 contract is looser, the host may do less work; where it is
stricter than the M3's, guests may rely on it.

The probes are in `guest/bundle/tools/oracle` ([README](../../guest/bundle/tools/oracle/README.md)):
`oracle_iosurface`, `oracle_storage`, `oracle_passes`, `oracle_cmdbuf`, `oracle_caps`,
`oracle_errors`, `oracle_source`, `oracle_ycbcr`, `oracle_ca_ycbcr`, `oracle_sampler`,
`oracle_icb`, `oracle_order`, `oracle_names`, `oracle_surftex`, `oracle_reflection`,
`oracle_mesh`, `oracle_events`, `oracle_texdesc` and `oracle_nilbind`; the upload costs (Q17)
are `guest/bundle/tools/upload_probe.m`'s.

**Machines.** Every claim below is measured, on:
- **AMD**: AMD Radeon Pro 5300M (Navi 14, `amdgpu_gfx1012`, 4 GB; AMDRadeonX6000 on
  IOAcceleratorFamily2), in a MacBookPro16,1 (i7-9750H), macOS 26.7.1 (25G241), x86_64.
- **Intel**: Intel UHD Graphics 630 in the same Mac. It drives the Mac's display.
- **M3**: Apple M3 Max, Mac15,9, macOS 27.0.1 (26A434), arm64. It ran only the probes where it
  shows the gap.

The two x86 GPUs agree on every coherence question except the ones named, so "x86" below
means both.

## Reading the oracles

Each answer below is read in one of four classes, and its class decides what the lane does with
it:

1. **Documented.** Metal's documentation promises the behaviour. The lane keeps it exactly,
   whatever a Mac measured here does.
2. **Divergent.** The documentation leaves it open, and the three Macs disagree. Any of their
   answers is inside the envelope a guest can meet, so the lane takes the one that costs the
   host least.
3. **Agreed.** The documentation leaves it open, and all three Macs do the same. Guests may rely
   on it, so the lane treats it as contract.
4. **Outside the envelope.** No Mac does what the lane does, or the documentation forbids it.
   The lane fixes it.

A boot overrides all four: what a guest's frameworks do on the lane decides, whatever the class.
The tags below, *(class N)*, say which class each answer is read in.

## Ranked relaxations

| # | relaxation | evidence | what it removes | expected win |
|---|---|---|---|---|
| 1 | *(class 2)* **An IOSurface's GPU copy is a device cache.** Copy the surface's pages in only at the first GPU use after a CPU read-write unlock, and write the GPU copy back only when a CPU lock follows a GPU write. | Q1: on x86 an unlocked CPU write is never seen by the GPU, and a GPU write is never seen in the pages without a lock, the committing process included. | m2v's whole-surface shadow copy-in before each command buffer, and its merge-back after each writing encoder, of every render-target surface. The copy path's per-command-buffer write-back on the guest CPU. | In WindowServer's steady state nothing CPU-locks the 3840x2400 display surface, so both copies of 36.9 MB per composite go to zero. That is most of the pl1 gap: CLTD 28.2 ms against stock's 0.53. The copies x86 does make cost 5.6 to 7.6 ms each at 4K, paid only at those locks. |
| 2 | *(class 3; landed, c95fca750)* **`commitAndWaitUntilSubmitted` and `waitUntilScheduled` return at submission.** | Q7: x86 returns 0.29 to 0.41 ms after commit, behind a 200 ms command buffer, as the M3 does. | The wait for completion the bundle restored (bbb153d9e), once rgk-lock's lock wait lands. | rgk-perf's fix 1 measured CLTD 19.6 to 6.1 ms. This is the M3's rule too; the oracle confirms x86 has no stricter one. |
| 3 | *(class 3; the handler's queue class 2)* **Asynchronous pipeline creators return before the compile.** | Q7: the handler runs about 60 ms later, on Metal's `Device serial queue`. | Our bundle compiles on the caller's thread, inside the call. | Compile time moves off apps' calling threads (launch, first frames). Nothing in WindowServer's steady state. |
| 4 | *(withdrawn)* **Shared textures' CPU access on a discrete host may use a CPU copy.** | Q3: on AMD, `getBytes:` of a Shared texture never sees GPU writes, and `replaceRegion:` uploads the CPU copy whole. | The waited staging blit of each `getBytes:`/`replaceRegion:` of a Shared texture. | Small: the calls are rare, and Intel and the M3 keep such textures coherent. |

Relaxation 4 is withdrawn: the calls are rare, and Intel and the M3 keep such textures
coherent, as ours does.

No relaxation was found for Managed storage (Q2: our discrete path already keeps two copies
synced by `didModifyRange:` and `synchronizeResource:`, as AMD does), Shared buffers (Q3:
coherent at command-buffer boundaries on x86, and ours are in place), load and store actions
(Q4), memoryless, tile and fetch features (Q5), hazard tracking (Q6), errors (Q8) or limits
(Q9). In Q4 and Q6 the x86 GPUs are stricter than the M3 (below).

## Where x86 is stricter than the M3 rules we copy

Every x86 Mac behaved this way. Where the M3 does otherwise, the answer is divergent, not
agreed:
- *(class 2: the lane keeps the M3's, which copies less)* **DontCare stores keep color.** A
  pass with store action DontCare writes its draws and clears to memory on both x86 GPUs. On the
  M3 it drops them, and m2v copies nothing back for a DontCare store (`render/shadows.rs`), as
  the M3 does (Q4).
- *(class 2: the lane keeps the M3's)* **Untracked resources are ordered across command
  buffers.** An untracked buffer written by one command buffer and read by the next on the same
  queue, with no event, is ordered on both x86 GPUs and races on the M3 (Q6).
- *(class 2)* **A timeout's error is IOAccelerator's.** x86 answers code 2 with "Caused GPU
  Timeout Error (00000002:kIOAccelCommandBufferCallbackErrorTimeout)" and no underlying error.
  Our bundle answers code 2 with the M3's "Caused GPU Hang Error
  (00000003:kIOGPUCommandBufferCallbackErrorHang)" (Q8).
- *(class 3: the M3's are real too)* **GPU times are real.** `GPUStartTime`/`GPUEndTime` are on
  the host clock on x86. The host sends the platform's GPU times in each completion, where it
  sent 0 for both (Q7).

## Q1: IOSurface-backed textures

Probe: `oracle_iosurface DEVICE single|cross|timing [W H [rt|ro [managed|shared|private]]]`. A
BGRA8 surface; each step writes a letter to every pixel on the CPU (with or without a lock) or
the GPU (a blit from a buffer, or a render-pass clear), and reads the GPU's view (a blit into a
Shared buffer) or the CPU's (the surface's base address, with or without a lock). The steps
run in one process, and across two with a texture each.

### What x86 does (AMD and Intel alike)

| step | AMD and Intel 630 | M3 |
|---|---|---|
| CPU write without a lock, then a GPU read: at the next command buffer, after another one, 200 ms later | old pixels, every time | new pixels |
| the same through a texture made after the write, in either process | old pixels | new pixels |
| a read-only lock and unlock | seed unchanged, GPU still old | (in place) |
| a read-write lock and unlock, even with no write | seed +1; the next GPU use uploads the whole surface from the pages | (in place) |
| an unlocked write of the top half, then a read-write lock and unlock | the GPU sees new top, old bottom: whatever the pages hold | (in place) |
| a write under lock and unlock | seen at the next command buffer | seen |
| GPU write, then the committing process reads without a lock: at completion, 200 ms later | old pixels | new pixels |
| GPU write, then a lock (read-only or read-write, any process) | the lock downloads the whole surface; the GPU's pixels | the GPU's pixels |
| GPU write, an unlocked CPU write of half, then a lock | the GPU's pixels everywhere: the download overwrites the unlocked CPU write | the CPU's half |
| GPU write, then a locked CPU write of half | CPU's half, GPU's other half | the same |
| another process's texture over the same surface, after a GPU write | sees it at once | sees it |

So each surface has one GPU copy per GPU, shared by every texture and every process on that
GPU. On AMD it is in VRAM: rgk-lock's traces of the driver's lock path show a written surface
marked current at submit and paged off at a lock (traced on this Mac, not by these probes). The GPU copy is refreshed from the pages at the first
GPU use after the seed moved past the GPU copy's, and the pages from the GPU copy at the next
lock after a GPU write. The same held for texture storage Managed, Shared and Private, and for
usage ShaderRead alone.

A GPU write moves the seed by one, but the GPU's own move does not make the next GPU use upload.
Measured at 3840 x 2400 (36.9 MB), in ms of a GPU read of the whole surface:

| GPU read | AMD | Intel 630 |
|---|---|---|
| nothing changed | 7.8 to 7.9 | 13.9 to 14.0 |
| after a read-write unlock (upload) | 13.5 | 21.5 |
| after a read-only unlock, or an unlocked CPU write | 7.8 to 7.9 | 13.9 to 14.0 |
| after a GPU write (seed +1) | 5.4 to 5.8 | 13.7 to 13.8 |

| lock | AMD | Intel 630 |
|---|---|---|
| after a GPU write (download) | 5.5 to 5.6 (8.1 read-only once) | 7.5 to 7.7 |
| again, with no GPU write between | 0.00 | 0.00 |

A read-write lock after a GPU write therefore costs a download, then an upload at the next GPU
use (AMD 5.6 + 5.6 ms at 4K), even when the CPU wrote nothing.

### What our stack does

- **In place** (`RGFX_FEATURE_SURFACE_PAGES`): the host maps the surface's guest pages and
  makes each texture over them. A texture the GPU only samples is a linear image over the
  pages. One the GPU renders to or writes is a shadow image: m2v copies the pages in before the
  first command of each command buffer that uses it, and merges the changed words back at the
  end of each encoder that wrote it. The copy-in is the whole texture: the graphics device
  takes no dirty-page source, whose resolution of the surface's pages cost the host more per
  command buffer than the copies it saved (measured on the 5080).
  Nothing consults the seed or a lock.
- **The copy:** at each commit the bundle copies every surface of the process whose seed moved
  into its block, and at each completion writes every written surface back under a lock.

### The relaxation

Hold one GPU copy per surface (m2v's shadow image is that copy), keyed by the surface, not the
texture:
- **In:** copy the pages to the GPU copy only at the first GPU use after a CPU read-write unlock
  took the seed past the GPU copy's. Record the seed after each GPU write as the GPU copy's own.
- **Out:** copy the GPU copy back to the pages only when a CPU lock comes after a GPU write.
  That is rgk-lock's page-off (op 0) at the kext's lock callback, which op 2 then waits for.
- Unlocked CPU readers and writers then see what they see on x86.

What it needs:
- the surface's seed at the host at each use (the copy path's block header holds it; the
  in-place path would report it);
- the lock callback (rgk-lock);
- every texture over a surface reading the one GPU copy, where a sampled-only texture today
  reads the pages in place;
- the host's scanout or present reading the GPU copy. AMD scans out of VRAM; a present that
  read the pages would need the copy back once a present.

**Expected win:** WindowServer composites into the 3840 x 2400 display surface every frame and
nothing locks it on the CPU, so the steady state copies nothing. Today each composite copies
36.9 MB in and merges back the changed words; pl1 measured CLTD 28.2 ms against stock's 0.53.
Surfaces apps draw on the CPU (backing stores) are copied in once per read-write unlock, as on
x86.

## Q2: Managed storage

Probe: `oracle_storage DEVICE mbuf 4096|67108864` and `mtex`.

| | AMD | Intel 630 and M3 |
|---|---|---|
| buffer: CPU write without `didModifyRange:` (next command buffer, 200 ms later) | never seen | seen at once |
| buffer: `didModifyRange:` of [0,256) after an unnamed whole-buffer write | the whole buffer uploaded (4 KiB and 64 MiB) | (one copy) |
| buffer: GPU write pending (no sync), then CPU [0,256) + `didModifyRange:` of it | only [0,256) uploaded; the GPU's bytes stay elsewhere | (one copy) |
| buffer: GPU write without `synchronizeResource:` (completion, 200 ms) | never seen by the CPU | seen at once |
| buffer: `didModifyRange:` after the reading command buffer was encoded, before its commit | seen: the upload happens at commit | seen |
| texture: GPU write without `synchronizeTexture:`, then `getBytes:` | old bytes, also 200 ms later | new |
| texture: unsynchronized GPU write, then `replaceRegion:` of a 16 x 1 corner | the whole CPU copy is uploaded: the GPU's write is lost outside the corner | corner new, GPU's write kept |

Our stack: on a discrete host Managed resources are two copies. `didModifyRange:` copies exactly
the named range to the GPU copy, ordered before the next commit; `synchronizeResource:` copies
back inside the blit encoder; `replaceRegion:` uploads its region (decode/managed.rs). On a
unified host they are one copy, as on Intel, a one-level 2D array by the linear SPI among them,
but for a texture of a shape no texture over a buffer has (more than one level, a cube or 3D
texture, multisampled, or of a block-compressed format): it keeps two copies there too, and the
bundle's `getBytes:` of it reads its GPU copy, copied on the device, so the GPU's writes show
with no synchronize, as on Intel. **Relaxation: none.** AMD is
looser than Metal in one way that costs us nothing: it uploads more than the named range when the
GPU copy is clean, so an app with a short `didModifyRange:` works on AMD, where ours uploads the
range alone.

## Q3: Shared storage

Probes: `oracle_storage DEVICE shtex`, `lintex shared|managed`, `argenc shared|managed`,
`nocopy`, `didmodify length|nocopy`, `linalign`, `spin`.

- **Shared textures** (`newTextureWithDescriptor:` of storage Shared) are made on AMD, Intel and
  the M3. On AMD such a texture behaves like an unsynchronized Managed one: `getBytes:` never
  sees a render clear or blit write (also 200 ms later), and `replaceRegion:` of a corner
  uploads the CPU copy whole over the GPU's write. Intel and the M3 are coherent. Ours: CPU
  access goes through a staging blit the call waits for (`RgfxTexture.mm`), which is coherent.
- **A linear texture over a Shared buffer** is in place on both x86 GPUs: a CPU write to the
  buffer is seen by the next GPU read of the texture, and a render clear of the texture lands
  in the buffer at completion. Over a Managed buffer it follows Managed's rules. That is our
  design (Decided 4).
- **Where a texture over a buffer may lie** (`linalign`, 11-03): the public form
  (`newTextureWithDescriptor:offset:bytesPerRow:`) and the linear SPI
  (`newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:`) take offsets and pitches
  that are multiples of `minimumLinearTextureAlignmentForPixelFormat:` (256 on both), over
  Shared and Managed buffers alike. Of 468 x 36 BGRA8 at offsets 0, 0x40 and 0x100 and pitches
  1872, 1920 and 2048, AMD makes those at 0 and 0x100 with 2048 and answers nil for the rest;
  Intel ends the process at `_mtlValidateStrideTextureParameters`' assertion. The tiled SPI
  (`newTiledTextureWithDescriptor:offset:bytesPerRow:`, a Managed descriptor over a Shared
  buffer) takes every one. Ours: the two forms answer nil off the alignment, as AMD, and the
  tiled SPI takes any place (`RgfxBuffer.mm`).
- **replaceRegion: of a linear texture over a Managed buffer** (`lintex managed`, 11-03) reaches
  the GPU's next read of the texture with no `didModifyRange:`, on AMD as on Intel, and the
  buffer's bytes hold the rows. A replaceRegion: of a 16 x 1 corner after a CPU write of the
  whole buffer with no call shows the GPU the corner and the CPU's write: AMD uploads the whole
  buffer, Intel is one copy. Ours: the bundle tells the host the rows' span, as the buffer's
  `didModifyRange:` of it (`RgfxTexture.mm`, rgfxCopyRegion:), and only the span, as Q2's
  `didModifyRange:` uploads its range alone.
- **didModifyData (SPI) of a linear texture over a Managed buffer** (`didmodify`, 11-03): on AMD
  the GPU sees a CPU write with no call only after the texture's `didModifyData`, which uploads
  the whole buffer, made by `newBufferWithLength:` or over the app's pages alike; Intel is one
  copy. CoreAnimation calls it: a dtrace (the objc provider) of an in-process CARenderer on AMD
  showed `-[MTLIOAccelTexture didModifyData]` on a backing texture it reused once the CPU redrew
  it, and no `didModifyRange:` of the texture's buffer; it made those textures with
  `newTiledTextureWithBytesNoCopy:` (a scratch probe, not kept). Ours: the buffer's
  `didModifyRange:` of all of it (`RgfxTexture.mm`).
- **A no-copy Managed buffer** (`nocopy`, 11-03) is two copies on AMD: the GPU's copy holds zeros,
  not the pages' bytes, until a `didModifyRange:`, and a CPU write with no call is never seen
  (also 200 ms later); Intel's is one copy. Ours is two copies on a discrete host, as AMD's, and
  one copy on a host whose GPU shares the CPU's memory, the GPU's copy filled from the pages at
  the create wherever it is one of its own (`RgfxBuffer.mm`, bufferOverPagesOf; over read-only
  memory and where the kext wires no pages, a shadow).
- **An argument encoder writing a Managed argument buffer** (`argenc managed`, 11-03):
  `setBuffer:` and `setTexture:` reach the GPU's copy on AMD only at the app's
  `didModifyRange:`; Intel is one copy. A Shared argument buffer's writes are seen at once on
  both. Ours: the encoder writes the buffer's bytes and tells the host nothing, as Metal leaves
  the call to the app.
- **A Shared buffer while the GPU runs:** a 1.3 s kernel polling a word (atomic and plain loads)
  never saw a CPU write made 20 ms after scheduling, on either x86 GPU; a kernel storing its
  progress was never seen by a CPU polling every millisecond before completion. Shared memory is
  coherent at command-buffer boundaries only. (Over a 6 s kernel the 5300M saw the CPU's write
  at 5.8 s, once.) Ours is in place and at least as coherent.

**Relaxation:** only the Shared texture's CPU access (ranked 4), and only on a discrete host.

## Q4: render-pass load and store actions

Probe: `oracle_passes DEVICE actions private|managed|iosurface`, `depth`, `stencil`, `msaa`. A
256 x 256 RGBA8 attachment, cleared to A and stored, then one pass of the row's actions with or
without a draw of B over its left half (Clear rows clear to C).

| load | store | draw | AMD and Intel: left, right | M3: left, right |
|---|---|---|---|---|
| Load | Store | B | B, A | B, A |
| DontCare | Store | B | B, A | B, 25% zeros + A |
| Load | DontCare | B | B, A | A, A |
| DontCare | DontCare | B | B, A | A, A |
| Clear | DontCare | B | B, C | A, A |
| Clear | DontCare | - | C, C | A, A |

The same for Private, Managed and IOSurface attachments. Depth32Float cleared with store
DontCare keeps the clear on AMD and drops it on Intel and the M3. The stencil of a
Depth32Float_Stencil8 attachment cleared with store DontCare keeps the clear on AMD and Intel,
and drops it on the M3, which keeps the stencil stored before. A 4x MSAA attachment resolved with
its samples DontCare keeps its samples on all three. *(class 2: what a DontCare store leaves is
divergent for color, depth and stencil.)*

Our stack: m2v copies a shadow in for a DontCare load, as for Load, and copies nothing back for
a DontCare store; a whole-texture Clear copies nothing in. **Relaxation: none**: x86 keeps every
color write a DontCare store discards on the M3, so skipping a copy-in on a DontCare load would
lose pixels x86 keeps. The DontCare store rule is a difference from x86 (above). Under Q1's
device cache it no longer decides anything for IOSurfaces, whose merge-back moves to the lock.

## Q5: memoryless, tile and imageblock, framebuffer fetch

Probe: `oracle_passes DEVICE features`.

| | AMD | Intel 630 | M3 |
|---|---|---|---|
| memoryless texture | Metal asserts: "MTLResourceStorageModeMemoryless is not a supported storage mode on this device." (the process ends) | the same | made |
| tile pipeline | nil, error nil | nil, error nil | made |
| framebuffer-fetch pipeline (`[[color(0)]]` input) | made; the fetch reads 0 | made; reads the attachment | made; reads it |
| `isFramebufferReadSupported` (SPI) | 0 | 1 | 1 |
| raster-order-group pipeline | made (`areRasterOrderGroupsSupported` 1) | made (1) | made (1) |
| `supportsTileShaders`, `supportsImageBlocks` (SPI) | 0, 0 | 0, 0 | 1, 1 |

Our stack: memoryless textures are made (m2v). The lane answers `isFramebufferReadSupported`
no to an x86-64 guest, as AMD does *(class 2: the Vulkan platform's fetch costs most)*, and the
host GPU's answer to an arm64 guest, as the M3 does (`device_info.rs`, `lane_flags`). It answers
`supportsTileShaders` and `supportsImageBlocks` no, as both x86 GPUs do (4ee4bcaf0). **Relaxation: none.** With no
Apple family claimed, guests should ask for none of these; a guest that makes a memoryless
texture ends on x86 and goes on with ours.

## Q6: hazard tracking

Probe: `oracle_cmdbuf DEVICE hazards`. A ~30 ms kernel writes 7 to 1M words (each thread spins
first); a second dispatch copies them. Runs where any copied word is not 7, of 5:

| ordering | AMD | Intel 630 | M3 |
|---|---|---|---|
| tracked, two compute encoders | 0 | 0 | 0 |
| untracked, two compute encoders, no fence | 0 | 0 | 0 |
| untracked, one serial encoder | 0 | 0 | 0 |
| untracked, one concurrent encoder, no barrier | 0 | 5 | 5 |
| untracked, concurrent encoder + `memoryBarrierWithScope:` | 0 | 0 | 0 |
| untracked, compute encoder then a blit copy, no fence | 5 | 0 | 5 |
| untracked, two command buffers, one queue | 0 | 0 | 5 |
| from an untracked heap, two compute encoders | 0 | 0 | 0 |
| tracked, two command buffers, two queues, the second committed once the first's `commitAndWaitUntilSubmitted` returns | 0 | 0 | 0 |
| untracked, the same | 0 | 0 | 5 |
| tracked, two queues, the second committed right after the first's `commit` | 0 | 0 | 0 |

On AMD a blit runs beside compute work: a tiny blit command buffer committed behind a 200 ms
compute one ran at once (GPU time 0.38 to 0.39 ms after commit), though its completion was
reported after the compute one's.

Our stack: tracked resources are ordered; untracked ones are left to the program, serial
encoders run in order, and the bundle has its own fence across every 8 encoders. **Relaxation:
none.** Untracked compute across encoders and command buffers is ordered on every x86 GPU, so
guests may rely on it; whether the Vulkan platform orders successive command buffers' untracked
work is worth a check. A tracked resource is ordered across two queues by commit on all three,
the later buffer committed as soon as the earlier one's `commitAndWaitUntilSubmitted` returns,
or right after its `commit` *(class 3)*; an untracked one is ordered so on x86 and races on the
M3 *(class 2)*.

## Q7: command-buffer timing and handlers

Probe: `oracle_cmdbuf DEVICE timing`. Times are from commit.

| | AMD | Intel 630 |
|---|---|---|
| a tiny buffer behind a 200 ms one, same queue: scheduled handler | 0.36 ms | 0.30 ms |
| its `waitUntilScheduled` returns | 0.38 ms | 0.34 ms |
| its `commitAndWaitUntilSubmitted` returns | 0.29 ms | 0.41 ms |
| its completed handlers | after the long one's (202.5 ms), in the order added | the same (197.4 ms) |
| a tiny buffer on a second queue | completed at 0.46 ms | 0.72 ms |
| a buffer that first waits on an unsignalled shared event, signalled at 300 ms | scheduled at the signal (302.2 ms) | the same (309.1 ms) |
| idle GPU, tiny buffer, median: scheduled / completed / `waitUntilCompleted` | 0.20 / 0.42 / 0.49 ms | 0.22 / 0.50 / 0.55 ms |
| handler threads | Metal's `com.Metal.CompletionQueueDispatch` queue, never the main thread | the same |
| `GPUStartTime`/`GPUEndTime` | real, on the host clock (the long one +0.59 to +202.28 ms) | real |
| `kernelStartTime`/`kernelEndTime` | +0.08 / +0.16 ms | +0.06 / +0.09 ms |
| async compute and render pipeline creators | return with the handler pending; it runs about 60 ms later on `Device serial queue` | the same |

`oracle_cmdbuf DEVICE gputimes`: what `GPUStartTime` and `GPUEndTime` answer for one long
buffer, K encoders of about 0.5 ms each in one buffer ("whole") against the same encoders
committed back to back one buffer each ("apart"), compute (`slow_write`) and render (K passes
over a 1920 x 1080 BGRA8 target, each clearing it and blending 6 full-target quads), medians of
7 runs, in ms, on the 5300M and the 630 (macOS 26.7.1):

| | AMD whole span | AMD apart, summed | Intel whole span | Intel apart, summed |
|---|---|---|---|---|
| compute, K 1 | 0.555 | 0.555 | 0.585 | 0.586 |
| compute, K 8 | 1.599 | 1.844 | 4.089 | 4.372 |
| compute, K 32 | 6.313 | 7.368 | 13.047 | 12.407 |
| render, K 1 | 0.336 | 0.335 | 3.175 | 3.063 |
| render, K 8 | 2.690 | 2.715 | 19.020 | 18.199 |
| render, K 32 | 11.044 | 10.887 | 59.561 | 59.990 |
| whole: `GPUStartTime` after the commit | 0.08 to 0.15 | | 0.12 to 0.40 | |
| whole: the completed handler after `GPUEndTime` | 0.015 to 0.14 | | 0.012 to 0.14 | |

So one buffer spans its encoders' GPU time summed, as back-to-back buffers do, within the GPUs'
clock ramp (both clock up under a sustained load: the AMD's 8 compute encoders took 0.2 ms each
against 0.56 ms alone) *(class 3)*. The CPU's time encoding a buffer never counts: a buffer of 8
compute encoders whose CPU slept 1 ms between them spanned 4.52 ms on the AMD and 1.50 ms on the
Intel, and the AMD, idle through the sleeps, started it 2.58 ms after its commit. Our stack: a
buffer the host commits in parts as it walks it answers its first part's start, and its parts'
GPU time summed as its span, never past its last part's end (`decode/split.rs`, "Its GPU times"),
where the span from its first part's start to its last part's end held the host's walk between
parts.

`oracle_cmdbuf DEVICE submitted`: a tiny buffer's `commitAndWaitUntilSubmitted`, 20 runs each,
times from the first commit (medians), on the 5300M, the 630 (both macOS 26.7.1) and the M3 Max
(macOS 27.0.1):

| | AMD | Intel 630 | M3 |
|---|---|---|---|
| behind a ~100 ms buffer, same queue: the call returns | 0.35 ms | 0.27 ms | 0.15 ms |
| its scheduled handler | 0.29 ms | 0.26 ms | 0.15 ms |
| status at the return | Scheduled 18, Committed 2 | Scheduled 20 | Scheduled 19, Committed 1 |
| scheduled handler run by the return | 18 of 20 | 20 of 20 | 17 of 20 |
| idle queue: the call returns / scheduled handler | 0.13 / 0.20 ms | 0.15 / 0.20 ms | 0.05 / 0.06 ms |
| idle queue: status at the return | Committed 20 | Committed 20 | Committed 20 |
| `waitUntilScheduled` right after the return, idle queue | blocks 0.12 ms | 0.09 ms | 0.01 ms |
| waiting on a shared event a second thread signals at 100 ms: the call returns | at the signal (107.1 ms) | the same (105.5 ms) | the same (106.0 ms) |
| the same: status at the return | Committed 20 | Committed 20 | Committed 20 |
| behind a buffer of its queue that waits on such an event: the call returns | at the signal (105.2 ms) | the same (109.5 ms) | the same (110.1 ms) |
| the same: status at the return | Scheduled 18, Committed 2 | Scheduled 18, Committed 2 | Scheduled 20 |

So the call returns once the buffer is submitted, which for a buffer waiting on an event, or
behind one on its queue, is at the signal *(class 3)*; whether the buffer is `Scheduled` by then,
and its scheduled handlers have run, is a race on all three *(class 2)*, and an idle queue
returns while it is still `Committed` *(class 3)*.

The M3 also runs a creator's handler after the creator returns, on the global default-QoS queue
(`com.apple.root.default-qos`) *(class 3 that it runs after the return; class 2 which queue)*.

Our stack: `commitAndWaitUntilSubmitted` and `waitUntilScheduled` return at submission
(c95fca750), and handlers run on the process's completion thread in Metal's order. A buffer
`commitAndWaitUntilSubmitted` commits asks for its submission (`CB_SUBMITTED`, design.md,
"Submitted"), and the call returns at the host's answer: as the host takes it, or at the signal
for one that waits on an event or is behind one that does, so an idle queue's buffer returns while it
is still `Committed`, as on all three. The host
sends the platform's GPU times, from which the bundle answers `GPUStartTime` and `GPUEndTime`,
and the async creators run their create and handler on the global default-QoS queue, after the
creator returns, as the M3 does.
**Relaxations:** ranked 2 and 3.

## Q8: errors and refusals

Probe: `oracle_errors DEVICE [purgeable|creates|residency|timeout]`.

| failure | AMD | Intel 630 |
|---|---|---|
| a buffer and a texture set purgeable Empty, then used | completes, no error | the same |
| 6 x 1 GiB Private buffers filled in one command buffer (AMD has 4 GB of VRAM) | completes in 836 ms, no error | completes, no error |
| a buffer of `maxBufferLength` + 4096 | made | nil |
| a Private heap of 4x `recommendedMaxWorkingSetSize` | made (17 GB) | nil |
| a 32768 x 32768 texture | Metal asserts (width and height over 16384); the process ends | the same |
| one kernel past the watchdog | status Error after 20.2 s | status Error after 16.1 s |

The timeout's error, on both: domain `MTLCommandBufferErrorDomain`, code 2,
`NSLocalizedDescription` "Caused GPU Timeout Error (00000002:kIOAccelCommandBufferCallbackErrorTimeout)",
and, with `MTLCommandBufferErrorOptionEncoderExecutionStatus`, `MTLCommandBufferEncoderInfoErrorKey`
naming the encoder with status 2. No `NSUnderlyingErrorKey`: x86 has no nested IOGPU error for
it. `GPUStartTime` and `GPUEndTime` are 0. The Intel kernel's writes were lost; the AMD kernel's
were kept. An 8 s kernel completes on both.

Our stack: the bundle's code 2 carries the M3's "Caused GPU Hang Error
(00000003:kIOGPUCommandBufferCallbackErrorHang)", and code 9 an underlying
`IOGPUCommandQueueErrorDomain` error (measured on the M3). **Relaxation: none.**

## Q9: limits and formats

Probe: `oracle_caps DEVICE`. Where the x86 GPUs' answers differ from the M3's or from our device's
(an x86-64 guest on the RTX 5080 host, metal_enum):

| answer | AMD | Intel 630 | M3 | ours (5080) |
|---|---|---|---|---|
| `maxBufferLength` | 3208642560 | 2147483648 | 41747087360 | 12821200896 (4ee4bcaf0; 1099511627776 before) |
| `maxThreadgroupMemoryLength` | 65536 | 65536 | 32768 | 32768 |
| `maxArgumentBufferSamplerCount` | 2048 | 2048 | 500000 | 0 |
| `readWriteTextureSupport` | 2 | 1 | 2 | 2 |
| `minimumLinearTextureAlignmentForPixelFormat:`, `linearTextureAlignmentBytes` | 256 | 256 | 16 | 256 (4a329e614) |
| `iosurfaceTextureAlignmentBytes` | 16 | 16 | 16 | |
| `isDepth24Stencil8PixelFormatSupported` | 1 | 0 | 0 | 0 |
| `supports32BitMSAA`, `supportsPullModelInterpolation` | 1, 1 | 1, 1 | 1, 1 | the runtime's: 1, 1 on the 5080 (metal2vulkan 7da81ee3d8; 0, 0 before) |
| `supportsTextureSampleCount:(8)` | 1 | 1 | 0 | 0 |
| barycentrics | 1 | 0 | 1 | 1 |
| `supportsRaytracing`, `supportsFunctionPointers`, `supportsDynamicLibraries` | 1 | 1 | 1 | 0 (lane) |
| `supportsPlacementSparse` | raises (not implemented) | raises | 1 | 0 |
| sparse tile sizes | 0x0x0 | n/r | 64x64 and up | 64x64 and up |
| families | Mac2, Common1-3, Metal3; no Apple | the same | Apple1-9, Metal4 too | as AMD |
| `supportsFeatureSet:(10002)` | 1 | 0 | 0 | 0 |
| `featureProfile` (SPI) | 10002 | 10002 | 10 | 10002, the x86 Macs' (Q11); 10001 where the guest's Metal lacks the MTLArchitecture class (macOS 11 and 12) |
| `heapTextureSizeAndAlign` alignment | 65536 | 4096 | 128 | |

Formats (`F` lines): both x86 GPUs make every uncompressed color format but B5G6R5, A1BGR5,
ABGR4 and R8/RG8 sRGB (BGR5A1 is made), and every BC format, the 4:2:2 formats and the depth formats (D24S8 and
X24S8 on AMD only). Metal asserts, ending the process, at the create of every ASTC, ETC2, EAC
and PVRTC format, B5G6R5, A1BGR5, ABGR4, R8Unorm_sRGB, RG8Unorm_sRGB and the XR10 formats
(552-555), and at a 4x MSAA BGR5A1 or RGB9E5 texture. `minimumTextureBufferAlignmentForPixelFormat:`
answers on x86 for every format (the format's bytes per pixel, or a compressed block's), where
the M3 asserts for compressed and depth ones. `minimumLinearTextureAlignmentForPixelFormat:`
(`oracle_bundle linalign`, every raw value from 0 to 799 and 1000, 1023, 4096 and 65535) answers 256
for the same 64 values on both, 46 of the SDK's formats and 18 it has no name for, and ends the
process at every other (compressed, depth and stencil, else MTLPixelFormatInvalid; D24S8 and X24S8
are depth on AMD, Invalid on Intel); ours does the same, D24S8 as Intel. Our device makes ASTC and the packed formats (m2v),
which no x86 Mac does; with no Apple family claimed, frameworks should not ask for them.

**Relaxation: none** for performance. Two answers help the host, and the lane gives both, as
choices inside the envelope *(class 2)*: `minimumLinearTextureAlignmentForPixelFormat:` and the
SPI linear alignments 256 for every format they answer, as both x86 GPUs answer, which suits the Vulkan
hosts' linear images (4ee4bcaf0, 4a329e614); and `maxBufferLength` at most three quarters of
`recommendedMaxWorkingSetSize`, the rule the 5300M and the M3 answer by (4ee4bcaf0).

## Q10: runtime source compile

Probe: `oracle_source DEVICE [OUT]`. A source of a vertex, a fragment and a kernel function,
compiled by `newLibraryWithSource:options:error:`, by the base class's implementation called on
the device, and by `_MTLDevice`'s compiler-task form called on the device with no task.

| | AMD | Intel 630 | M3 |
|---|---|---|---|
| library | `_MTLLibrary` | the same | the same |
| its bytes (`libraryDataContents`) | a .metallib (`MTLB`) of AIR, 12324 bytes | the same bytes as AMD | 12324 bytes (macOS 27) |
| the three routes | the same bytes | the same | the same |
| a source that does not compile | nil; `MTLLibraryErrorDomain` 3 with the compiler's log | the same | the same |
| a source that compiles, also with an unused variable | the library; `*error` nil | the same | the same |

How the base class compiles (dtrace on the AMD device's classes, and method swaps on all three):
its form without a task calls the compiler-task form on the device, which reaches Metal's
compiler service through `_MTLDevice`'s own `getCompilerConnectionManager:`, on the caller's
thread. It never asks the device for its `compiler`: a compiler answered nil, or one made with a
plugin that does not exist, changed nothing. Of the device it reads `featureProfile` (AMD and
Intel answer 10002, the M3 10), `supportsFamily:withFamilyArray:count:`, and Metal's own settings
for the compiler service and for shader capture. Six sources (a kernel, framebuffer fetch, an
imageblock, simdgroup matrices, bfloat and ray tracing) compiled under `featureProfile` 10 and
10002 made the same bytes, in either order.

Our stack: the device's source forms run that compiler-task form on super, over the device, and
make the device's library of its bytes through `newLibraryWithData:`, as an app's .metallib is
made; Metal's library is kept as the shadow, and Metal's error and nil pass through unchanged
*(class 3)*. Before this, the device refused every source, which lost Chrome's WebGL contexts
at their first link.

## Q11: mipmapped textures over a buffer, and the feature profile

Probe: a scratch probe, not kept. A 256 x 256 BGRA8 texture over a buffer, rows 1024 bytes apart,
with 1, 2 and 7 mipmap levels, by each form: the public `newTextureWithDescriptor:offset:bytesPerRow:`
of a Shared buffer, the buffer's SPI `newTiledTextureWithDescriptor:offset:bytesPerRow:` (descriptor
Managed, buffer Shared), and the device's SPI `newTiledTextureWithBytesNoCopy:length:descriptor:
offset:bytesPerRow:`, which asks the buffer it makes for the second.

| form, 2 or 7 levels | AMD | Intel 630 | M3 |
|---|---|---|---|
| public | Metal asserts: "Linear texture: cannot be mipmapped"; the process ends | the same | the same |
| the buffer's tiled SPI | made, with the levels asked, storage Shared; level 0 over the buffer's bytes; levels 1 and up rendered to and read by a blit and a kernel, their pixels nowhere in the buffer | the same | the buffer does not answer the selector |
| the device's tiled SPI | as the buffer's | the same | an unrecognized-selector exception (the device asks its buffer for the buffer's SPI) |

What such a texture holds was measured later (11-11, a scratch probe written from scratch, not
kept). The probe made a 256 x 256 texture of 9 levels over rows 1024 bytes apart, RGBA8 and
BGRA8, with Shared and Managed descriptors, by both tiled forms. Each level was read by a blit:

| step | AMD and Intel 630 |
|---|---|
| the texture made over the rows (x, y, 0x80, 0xff), then read | level 0 the rows, (128, 128, 128, 255) on average; levels 1 to 8 generated from them, (127, 127, 128, 255) |
| made over zero bytes, the rows written by the CPU right after the make or 100 ms after, then the first read | the same: the bytes are taken at the first GPU use, not at the make |
| after a read, the CPU rewrites every byte to 0x40 | no level changes |
| then `didModifyData` (SPI) | every level 0x40: level 0 copied again and every level generated again |
| after a read, the CPU writes the rows, calls `didModifyData`, then writes 0x20 with no call, then a read | every level 0x20: `didModifyData` takes the bytes at the next GPU use, not at the call |
| a blit writes level 0 | level 0 changes; levels 1 and up and the buffer's bytes do not |
| `storageMode` | 0 (Shared), from a Managed descriptor too |

So the texture is a copy of the buffer's bytes with every level generated from level 0. The copy
is taken at the first GPU use after the make, and again after each `didModifyData`, and a GPU
write stays in the texture *(class 3: both x86 GPUs agree)*. Safari 26's CoreGraphics makes such
textures for the pictures it draws: 1600 x 1600 of 11 levels and 600 x 400 of 10, RGBA8Unorm,
packed rows at offset 0. It never writes levels 1 and up, and it samples a picture drawn below
its size with linear mipmap filtering. It writes the bytes before it makes the texture, and calls
no `didModifyData` on it (26.7.2 walks 15718 and 15720, 11-11, a probe of the bundle's). The
host's platform makes the texture as the x86 Macs do (metal2vulkan's
`Buffer::new_tiled_texture`): a copy of the bytes with every level made from level 0, taken at
its first use in a command buffer, which no later CPU write reaches and from which no GPU write
reaches the buffer. A platform that kept level 0 in step with the bytes and left levels 1 and up
zero drew such a picture black or dark (26.7.2 walks 15511, 15613 and 15614 on a Radeon 780M
and an RTX 3090, 11-11). The bundle tells the host a mipmapped texture's `didModifyData`
(`TEXTURE_DID_MODIFY_DATA`), and the copy is taken again at its next use, as on the x86 Macs.

The CPU's calls on such a texture were measured next (11-11, the same probe), 256 x 256 RGBA8 of
9 levels by both forms, Shared and Managed descriptors. The texture was made over 0x40, read by a
blit, then each step ran in turn:

| step | AMD | Intel 630 |
|---|---|---|
| `getBytes:` of level 0 before any GPU use | the bytes | the process crashes in Metal (a segmentation fault) |
| `getBytes:` of level 0 after the CPU rewrote the bytes to 0x20 with no call, after a `replaceRegion:` of level 0, and after a blit wrote level 0 | the buffer's bytes as they lie each time, not the copy | the same |
| `getBytes:` of level 1 at the same points | made from the bytes as they lie: 32, then 96 | what the copy held then, 64 and 64, with a Shared descriptor; 58 and 58 with a Managed one |
| `replaceRegion:` of level 0 with 0x60 | the buffer's bytes become 0x60, and the next GPU use reads every level 0x60: the copy is taken again with no `didModifyData` | the same |
| `replaceRegion:` of level 1 with 0x70 | `getBytes:` of level 1 reads 0x70 back; no GPU use sees it, before or after a `didModifyData` (level 1 stays made from level 0) | the same with a Shared descriptor; with a Managed one the process crashes in Metal (a bus error) at it |

So the CPU reaches level 0 in the buffer's bytes, and a `replaceRegion:` of level 0 takes the
copy again as `didModifyData` does *(class 3: both agree)*. The bundle does both: its `getBytes:`
and `replaceRegion:` of level 0 address the buffer, and a `replaceRegion:` of level 0 of the
tiled form also sends `TEXTURE_DID_MODIFY_DATA`. The CPU's level 1 and up differ between the two
GPUs *(class 2)*; the bundle reads and writes them in the device's copy by a blit, so a
`replaceRegion:` of level 1 or up reaches the GPU's use until the copy is taken again, which
neither x86 Mac's does. No app is known to write those levels by the CPU.

*(class 2: the tiled forms are x86's alone.)* WindowServer's tiled backings use the buffer's SPI
(RgfxBuffer.mm). Our stack: the public form ends the process at more than one level with
Metal's words *(class 3)*, sending the host nothing; the tiled forms make one *(class 3)*:
`TEXTURE_FROM_BUFFER` carries the levels, and the host's platform makes one image of every
level, the copy above (metal2vulkan's `Buffer::new_tiled_texture`), the bundle logging the first
of each process with its levels. Before the host made them, with the lane answering `featureProfile`
10002, an x86 Mac's, WindowServer asked the lane for a 7-level texture over a buffer, which the
host's platform refused, and every later command buffer naming it faulted, so its composition
stopped (boot ro1-p2, 11-02); answering 10 it asked for none (ro1-p1 and its control). Core Image made no texture over a buffer by any of the three forms, under 10 or 10002,
for blurs, scales and renders to a texture on the AMD and the Intel GPU. WindowServer on the
oracle cannot be traced, being protected, so which framework reads the profile for it is not
measured. The lane answers the x86 Macs' 10002 on every host *(class 2)*, since the host makes
the levels; the frontend makes the same bytes under either (Q10).

Rechecked on 26.7.2 over the Vulkan platform (a Radeon 780M, boots 7395 and its control 7407,
11-07), one variable apart: answering 10002, WindowServer asked for a tiled texture over a buffer
of 6 mipmap levels, which the device answered nil, and the desktop still composed; answering 10,
it asked for none. With the host making the levels (boot 7666, 11-08, a 780M), WindowServer's
6-level texture was made and 26.7.2 composed its desktop under 10002, so the device answers 10002
where the guest's Metal has the MTLArchitecture class. Where it lacks it (macOS 11 and 12, whose
Metal ends WindowServer at device creation under 10), the device answers 10001, the profile both
of 12's x86 GPU drivers answer (RgfxFeatureProfile.hpp).

**Types and arrays over a buffer.** Probes: `oracle_storage DEVICE lintypes FORM:TYPE`, a texture
of one `MTLTextureType` over a Shared buffer at offset 0, rows 256 bytes apart, 64 x 4 BGRA8 (64
x 1 for a 1D type, 64 x 64 for a cube), 2 deep for 3D, 2 long for an array, 4 samples for a
multisample type, by the public form, the linear SPI (`newLinearTextureWithDescriptor:offset:
bytesPerRow:bytesPerImage:`) or the tiled SPI (descriptor Managed); and `oracle_storage DEVICE
linarray MODE`, a 2D array of 3 slices by the public form or the linear SPI, written and read
slice by slice by the CPU and the GPU. "Asserts" means Metal's validation ends the process with
these words in `_mtlValidateStrideTextureParameters`, each a line after "Texture Descriptor
Validation".

| form, type | AMD | Intel 630 | M3 |
|---|---|---|---|
| public: 1D, 1D array, 2D array, cube, cube array | asserts "Linear texture: must be of type MTLTextureType2D, textureType (T) disallowed" | the same | the same |
| public: 3D, 2 deep | asserts "Linear texture: must have depth == 1, depth (2) disallowed", then the type's line | asserts "Buffer texture: cannot support depth > 1." (`validateTextureDescriptor`) | as the AMD |
| public: 2D multisample, 2D multisample array | the descriptor's validation asserts first: no multisample texture in Shared or Managed storage on this device | the same | asserts the type's line |
| linear SPI: 1D, 1D array, cube, cube array | asserts "Linear texture: must be of type MTLTextureType2D or linear MTLTextureType2DArray, textureType (T) disallowed" | the same | the same |
| linear SPI: 2D array | asserts "2D linear texture array is not supported." | the same | made: slice s lies s times the rows' bytes rounded up to 128 from the first, whatever `bytesPerImage` (0, the rows' bytes or 1024), for the CPU's copies and the GPU's reads alike |
| linear SPI: 3D, 2 deep; multisample | as by the public form, the SPI's type line | the same | the same |
| tiled SPI: 1D, 1D array, 2D array, cube, cube array, 3D | made | the same | the buffer does not answer the selector |
| a texture buffer, 4 rows high, by any form | the descriptor's validation asserts first: 4 rows past its 1 | the same | the same, by the forms it has |
| public: a texture buffer, 4 RGBA32Float or 16 RGBA8Unorm texels | made over a Shared or Managed buffer at offsets 0, 4, 16, 64, 100 and 256, rows 64 or 4096 bytes apart; a kernel reads the buffer's bytes from the offset | the same | not measured |
| public: a texture buffer, its texels past the buffer's 4096 bytes (offset 4064) | made | the same | not measured |
| public: a texture buffer, rows fewer than its texels' 64 bytes | nil at 1; asserts at 0 "Linear texture: bytesPerRow (0) must be greater or equal to (64) bytes" | asserts at 1 and at 0, at 0 with the AMD's words | not measured |

*(class 3 for the type lines the three share; class 2 for the rest.)* Our stack: the public form
and the linear SPI end the process at each type above they do not make, with the words the three
share or, at a 3D or multisample type, the M3's, sending nothing, since the host's platform ends
the host's process at a texture over a buffer it does not make. A texture buffer by the public
form is made, at any offset, which the host's platform makes by that form; rows of 0 bytes end the
process with the words the two say, and fewer than its texels' bytes answer nil, the AMD's answer
*(class 2)*. One whose texels lie past the buffer's bytes answers nil where the two make it: the
host's platform refuses it, which would end the host's process (the texture buffer rows and ends:
a scratch probe, `tbprobe`, not kept, macOS 26.7.1). The tiled SPI answers nil for every type but 2D,
logged once, as it does for more than one level: the host does not carry the others. The linear
SPI makes a 2D array as the M3 does *(class 2)*, its slices where the M3 lays them
(`rgfxBufferSlicePitch`), and a view of more than one of its slices answers nil, which Metal
asserts at on the M3 Max and an M4 (`sliceRange.length == 1`, measured by metal2vulkan).

## Q12: texture descriptors Metal's validation ends the process at

Probe: a scratch probe, not kept, one case a process, with `MTL_DEBUG_LAYER` unset: a 64 x 64
Private texture by the device's `newTextureWithDescriptor:` and by an Automatic heap's, of the
three-channel formats 45 and 100, and 2DMultisample ones (4x, 2x and 1x) of BC1, ASTC (186, 222),
GBGR422, BGRG422 and Unspecialized (263). The guest's x86 macOS 26.6 makes these descriptors
and sent them on (boot pm29, oracle_caps); the oracles' answers, alike at the device and the heap:

| descriptor | AMD | Intel 630 | M3 |
|---|---|---|---|
| a three-channel format (45 to 48, 95 to 100, 120 to 122) | the process ends at the descriptor's setter: "pixelFormat (45) is not a valid MTLPixelFormat." | the same | the process ends at the create: "MTLTextureDescriptor has invalid pixelFormat (45)." |
| multisample, of a compressed or 4:2:2 format | the process ends at the create, Metal listing each rule broken: the format "does not support multisample textures", "a compressed pixel format set with a sample count greater than one" (at 1x: "sampleCount must be > 1"), compressed formats' types, and the 4:2:2 formats' 2D | the same | the same |
| multisample, of Unspecialized (263) | the process ends: "invalid pixelFormat (263)", and Invalid "does not support multisample textures" | the same | the process ends: Unspecialized "does not support multisample textures" |

The x86 GPUs name a format they lack (ASTC) as an invalid one, where the M3 names it. Every
message is `-[MTLTextureDescriptorInternal validateWithDevice:]`'s "Texture Descriptor
Validation", but the x86 setter's. Our stack: every texture creator ends the process at each of
these, sending the host nothing *(class 3)*, rather than sending a create the host refuses
(`rgfxEndAtInvalidTextureDescriptor`, RgfxTexture.h): with the x86 Macs' words, and at a
three-channel format, which a guest's x86 setter ends the process at first, with the M3's
create's. Q24 has the rest of the validation.

## Q13: two-plane Y'CbCr textures

Probe: `oracle_ycbcr DEVICE`. A 64 x 64 CVPixelBuffer over an IOSurface, of OSType 420v (video
range) or 420f (full range), its `kIOSurfaceYCbCrMatrix` set to ITU_R_709_2 or ITU_R_601_4, by
CoreVideo's propagated `kCVImageBufferYCbCrMatrixKey` or on the surface itself, or left unset;
each quadrant holds one patch of known codes. A texture of `MTLPixelFormat` 500 (YCBCR8_420_2P)
over plane 0, by `newTextureWithDescriptor:iosurface:plane:`, read by `read()` at each
quadrant's middle. Each case runs in a child process.

| | AMD | Intel 630 | M3 |
|---|---|---|---|
| the texture | the process ends at the create: "MTLTextureDescriptor has invalid pixelFormat (500)." | the same | made |
| what it reads | | | the codes converted by the surface's range and matrix, within one level on every channel: BT.709 under ITU_R_709_2, BT.601 under ITU_R_601_4 and with no matrix |

The surface reads back the matrix as set, by either route, on all three.

*(class 3 for this lane; reversed 11-05.)* Both x86 Macs end the process at every two-plane
texture: oracle_surftex (Q20) measured "MTLTextureDescriptor has invalid pixelFormat (N)." at each
two-plane Y'CbCr format (500, 502, 503, 505 to 510, 520, 522, 523, 525 to 530, 540 to 545) and at
the extended-range RGB10A8_2P_XR10 and its sRGB twin (550, 551), over a 420v surface's planes, a
one-plane surface and a surface without planes. Until 11-05 the lane read the formats as the M3's
alone *(class 2)* and made them as the M3 does. The lane imitates the x86 dGPU Macs, so where both
agree their answer is the contract, and the M3's making is outside the lane's envelope: the
device's creator now ends the process at each with the x86 Macs' words, before the host is sent
anything (`rgfxEndAtInvalidSurfaceTexture`, RgfxTexture.h). The surface facts a texture's create
carries (`RgfxSurfaceFacts`) and the host's conversion by them stay; no texture on the lane
reaches the conversion. No code of the bundle makes such a texture (its video decoder lands
pictures through R8, RG8, R16 and RG16 textures over the planes), and boot ycc4 below traced no
guest process making one; a boot is the check of what the guest's frameworks choose.

Where the guest meets Y'CbCr on the desktop (boot ycc4, 11-02, with the bundle's texture creates
traced): the default wallpaper is an aerial, a 3840 x 2160 frame of OSType `x420` (10-bit 4:2:0,
video range) tagged ITU_R_709_2. No process made a two-plane Y'CbCr texture of it. VideoToolbox's
decoder service made textures of its planes, R16Unorm (20) over plane 0 and RG16Unorm (60) over
plane 1, and a RGBA16Float (115) one over an `RGhA` surface it writes; WindowServer made the same
two plane textures and composites the frame from them. The matrix is a constant of Apple's own
shaders there, read from the surface's tag, and the frame shows by BT.709 on the lane. On the stock
lane the same frame shows by BT.601: the lane's desktop is the stock's colours re-encoded by
BT.601 and decoded by BT.709, within 1.1 levels on average (the guests' captures), with no
parameter fitted.

What the frameworks convert such a frame by on a Mac (probe: `oracle_ca_ycbcr DEVICE`, 11-02):
64 x 64 CVPixelBuffers over IOSurfaces, of OSType `x420` and `420v` (both video range), tagged
ITU_R_709_2 or ITU_R_601_4 by CoreVideo's propagated `kCVImageBufferYCbCrMatrixKey`, or untagged;
each quadrant holds one patch of known codes. CoreAnimation draws each as a layer's contents, by
a CARenderer into a BGRA8 texture of the device on a queue of the device; VideoToolbox converts
each to a 32BGRA pixel buffer by a VTPixelTransferSession, on a device it picks itself. Every
read is within 0.0044 of one matrix's conversion of the codes and at least 0.085 from the other's.

| | AMD | Intel 630 | M3 |
|---|---|---|---|
| CoreAnimation, `x420` and `420v` | by the tag: BT.709 under ITU_R_709_2, BT.601 under ITU_R_601_4 and untagged | the same | the same |
| VideoToolbox, `420v` | the same as CoreAnimation | the same | the same |
| VideoToolbox, `x420` | by the tag; untagged, BT.709 | the same | by the tag; untagged, BT.601 |

With the surface's primaries and transfer function set to ITU_R_709_2 and the renderer's colour
space sRGB, CoreAnimation's colour-matched reads still move with the tag, by 0.13 between the two
matrices, on all three. *(class 3 for a tagged frame: all three convert ITU_R_709_2 by BT.709,
as our lane shows the aerial; class 2 for VideoToolbox's untagged `x420`.)* On the lane the
guest's own frameworks convert, by the tag our surfaces keep as set, so the lane has no answer of
its own to change.

## Q14: sampler `lodBias` and `reductionMode`

Probe: `oracle_sampler DEVICE` (11-03). A 256 x 256 RGBA16Float texture of 9 levels, level i
holding i everywhere, so a linear-mip sample's R is the LOD the sampler used. It is sampled with
`lodBias` 0, +1, -0.5 and +2.5 at LOD 0, 2 and 3.25, from an explicit `level()`, a `gradient2d()`
and a fragment shader's implicit LOD. Then a 2 x 2 RGBA16Float texture of 2 levels (level 0 R =
[0.1 0.9; 0.3 0.7] and G = [0.8 0.2; 0.6 0.4], level 1 R 0.05 and G 0.95), sampled with each
`reductionMode` under linear min and mag filters. Both properties are new in macOS 26, and the
lane's wire 2 does not carry them.

| | AMD | Intel 630 | M3 |
|---|---|---|---|
| `supportsSamplerReductionMode`, `samplerReductionModeSupport` (SPI) | 1, 2 | 1, 2 | 0, 0 |
| `supportsExtendedSamplerLODBiasRange` (SPI) | 0 | 0 | 0 |
| `lodBias` | the LOD used is clamp(LOD + bias, 0, 8), from `level()`, `gradient2d()` and a fragment's LOD alike | the same | no effect; Metal's validation layer accepts it |
| min and max at the small texture's centre, LOD 0 | (0.1, 0.2) and (0.9, 0.8) | the same | the weighted average, (0.5, 0.5); under the validation layer the sampler's create ends the process: "Use of minimum or maximum reductionMode requires device support" |
| min and max at LOD 0.5, linear mip filter | across both levels: (0.05, 0.2) and (0.9, 0.95) | the same | the weighted average |
| min and max with a nearest mip filter | reduced within the level read | the same | the weighted average |
| at one texel's centre (weight 1) | that texel in every mode: texels of weight zero do not count | the same | the same |
| a nearest-mip LOD of exactly .5 | rounds up | up in compute, down in a fragment | down |
| the fraction of a linear-mip LOD of 3.25 | 3.1875 | 3.0 in compute, 3.2461 in a fragment | 3.25 from `level()`, 3.2422 otherwise |

- **`lodBias`** *(class 1; class 3 for an explicit LOD)*. The header documents the bias "when
  sampling from a texture". It says only the LOD clamps are ignored for an explicit LOD. Both x86
  GPUs add the bias to `level()` too. The lane applies it to every LOD source.
  - The M3 lacks the feature (metal2vulkan's family table: Apple10 only). A host on an Apple9 GPU
    cannot carry it, which is a ledger gap.
- **`reductionMode`** *(class 1, though both x86 GPUs diverge from the docs)*. The header says the
  sampler ignores the mode when the mip filter is nearest or none, or the min or mag filter is
  nearest. Both x86 GPUs reduce with a nearest mip filter anyway.
  - Decided: the documented rule, kept until a boot shows a guest relying on the x86 GPUs'
    behaviour. The host passes the mode only where all three filters are linear.
- **The answers** *(class 4)*. A yes that samples the weighted average is outside the envelope.
  So the guest answers the x86 GPUs' 2 and yes only where the host realizes min and max, and the
  M3's 0 and no elsewhere.
- **The rounding at exactly .5 and the LOD's fraction** *(class 2)*. The three GPUs differ, so the
  host takes its own GPU's.

Wire 2 drops both properties, so the lane samples with Metal's defaults (bias 0, the weighted
average). The next wire change carries them in a tail of the sampler's create.

## Q15: indirect command buffers after a reset, and after a release

Probe: `oracle_icb DEVICE resets` and `oracle_icb DEVICE freed` (11-03).

`resets` encodes two render commands and two compute commands with a pipeline state, buffers and
a draw or dispatch. It resets command 0 with the command's `reset` and command 1 with the
buffer's `resetWithRange:`, then encodes their buffers and draws again without the state. The
compute commands add 5 into an output.

| | AMD | Intel 630 | M3 |
|---|---|---|---|
| render, after either reset | both draw: the state is kept | neither draws | neither draws |
| compute, after either reset | the command buffer times out | neither adds | both add 5: the state is kept |

- **What a reset keeps of the state** *(class 2: divergent. The M3 keeps a compute command's
  state, the Intel 630 drops it and the AMD times out; the AMD alone keeps a render command's.
  The lane takes the cheapest answer)*. The lane's Vulkan arm keeps a compute command's state
  after its `reset`, as the M3 does. A range reset, one platform call for the range however many
  commands it spans, leaves every command fresh, as the Intel 630 does. A render command's reset
  leaves it fresh on both arms.

`freed` executes commands whose buffers the app released before the execute:
- a render command's fragment buffer, then also its vertex buffer, then buffers made since at
  the same size;
- a heap's buffer, its heap resident;
- a dispatch's 16-byte input, released before the execute, then behind a spin while the execute
  waits, and at 1, 8 and 64 MiB;
- the dispatch's output.

| | AMD | Intel 630 | M3 |
|---|---|---|---|
| the command buffer | completes with no error, every case | the same | the same |
| a released 16-byte input | the dispatch runs, reading the buffer's last bytes | the same | the same |
| a released input of 1 to 64 MiB | the dispatch runs, reading zero | it runs, reading other bytes | it runs, reading zero |
| a released fragment or vertex buffer | nothing draws | draws with the last bytes | draws with the last bytes |
| a buffer of the same size made since | the command reaches it | the same | the same |

- **No error, and the command runs** *(class 3)*. Every case completes, and every dispatch runs
  and writes its live buffers.
- **What a released buffer reads** *(class 2)*. The lane's Metal arm reads what the host's Metal
  reads. Its Vulkan arm reads zero, the runtime resolving the address to no live buffer, except
  in an indirect command buffer resident for writing, which runs nothing there yet (below).
- **A buffer made since at the same address** *(class 2 for the lane)*. All three Macs reuse the
  address. The guest's next-fit picks do not reuse a range soon after its release, so the lane's
  commands do not reach a buffer made soon after.
- **The command a miss skips** *(class 4)*. On the Vulkan arm over MoltenVK, a dispatch whose
  input was released runs nothing, with no error and no notice, where an encoder made its
  indirect command buffer resident for writing: the runtime then executes the buffer from its
  records on the host. Made resident for reading alone, the dispatch runs, reading zero there,
  as it does through a nil buffer at the same index. Skipping the command is no Mac's answer.
  It is handed to metal2vulkan, whose host replay is to run the command with the slot reading
  zero; the lane's test of it,
  `a_command_naming_a_released_buffer_reads_zero_in_a_buffer_resident_for_writing`, waits
  ignored until it does.

## Q16: the order of IOSurface accesses across queues and processes, and its records

Probe: `oracle_order DEVICE xproc|xqueue|xqueue1 raw|war|waw` (11-03). A compute pass of about
210 ms on the 5300M (610 ms on the 630, 55 ms on the M3) writes or reads a 1024 x 1024 R32Uint
surface S; right after its commit, a blit over S is committed with `commitAndWaitUntilSubmitted`
by another process (`xproc`), or by another queue of the same process through a texture of its
own (`xqueue`) or the pass's texture (`xqueue1`). Every other round the pass runs over an unrelated
surface instead, as a control. Times are from the blit's commit, over 3 rounds of each of the
cases each GPU ran:

| | AMD | Intel 630 | M3 |
|---|---|---|---|
| `commitAndWaitUntilSubmitted` returns | 0.05 to 0.81 ms | 0.57 to 4.61 ms | 0.04 to 0.12 ms; 1.7 to 4.7 ms in a run's first round |
| the blit's scheduled handler | 0.07 to 0.83 ms | 0.60 to 4.65 ms | 0.05 to 0.14 ms; 1.8 to 4.7 ms in the first |
| status at the return | Committed; Scheduled in 2 rounds | Committed; Scheduled in 1 round | Committed |
| another process's blit, read or write; another queue's write | completes after the pass (207 to 213 ms) | after the pass (609 to 760 ms) | writes after the pass (52 to 60 ms); another process's read as another queue's (next row) |
| another queue's read of S after the pass's pending write | after the pass in 3 of 6 rounds; at 2.7 to 2.9 ms, reading S before the write, in the others | after it in 1 of 3, at 5.2 ms reading S before it in 2 | after it in each run's first round only, at 0.7 to 1.7 ms reading S before it in the others, from another process too |
| the control rounds | complete at 0.8 to 2.9 ms | at 2.2 to 6.4 ms, or after the pass (the 630 runs other work in turn) | at 0.7 to 1.7 ms from another process; at 36 to 56 ms from another queue, after the pass |

So a Mac orders such an access on the GPU alone *(class 3)*: the later command buffer is submitted
and scheduled within a millisecond of its commit on AMD and the M3 (in 1.7 to 4.7 ms in the M3's
first round of a run), and within the 630's usual 1 to 5 ms, whatever it is ordered behind, and then
completes after the access it is ordered after. Our stack: the bundle names the order
(`SURFACE_ORDER`, design.md "Surface order"), the host makes the later buffer wait on the GPU, and
the wait holds back neither the buffer's `CB_SUBMITTED` nor its `CB_SCHEDULED`, which it posts at
the buffer's take where it asks and no buffer before it on its queue owes its own, else at its
commit, nor
those of the buffers committed behind it on its queue (`cmdbuf.rs`, `Line`).

**A divergence the lane does not use.** A read on another queue of the same process (on the M3, of
another process too) is not always ordered after a pending write: on all three GPUs some rounds read
S before the write, the first round of each run excepted *(class 2: how often differs)*. The probe
commits the read right after the write's `commit`, which returns before Metal submits the write, so
a read submitted first on its own queue may simply run first. Writes after a pending access were
ordered in every round on all three, and so were reads across processes on the x86 GPUs, not on the
M3. The lane orders all three cases on every queue pair: the read's wait is on the GPU and holds
back no record, so the order costs the guest no time, and a guest that relies on it sees it kept.

## Q17: what the CPU's uploads into textures and Managed buffers cost

Probe: `upload_probe -d DEVICE` (`guest/bundle/tools/upload_probe.m`, its header says what each case
times), 11-03, at the shapes of the linear textures WindowServer made over its buffers before the
bundle refused the ones off the 256-byte alignment (Q3): BGRA8 128 x 32, 336 x 64 and 480 x 36, A8
and RG8 64 x 64. The lane's column is a guest on the RTX 5080 (tip 392ce3daf), the same binary.
Microseconds, medians:

| | AMD | Intel 630 | M3 | lane (5080) |
|---|---|---|---|---|
| Managed `replaceRegion:` of a whole texture, the call | 1.0 to 8.5 | 0.6 to 4.7 | 1.0 to 7.5 | 2.1 to 3.5 |
| the same, calls 2 ms apart | 10 to 33 | 7 to 17 | 6 to 19 | 13 to 20 |
| 64 such calls on 64 textures, then one buffer that reads them, per texture over that buffer alone | +39 to +68 | +0.6 to +7.5 | +0.6 to +22 | +85 to +98 |
| a new Managed texture: `newTextureWithDescriptor:` | 12 to 13 | 7 to 9 | 9 to 19 | 13 to 25 |
| a new texture, filled and read by a blit, over the blit of an old one | +90 to +170 | +7.5 to +11 | +38 to +144 | +264 to +375 |
| Shared `replaceRegion:`, the call | 0.9 to 8.5 | 0.5 to 4.7 | 0.4 to 3.2 | 340 to 420 |
| `didModifyRange:` of 256 B to 64 KiB, the call | 0.1 | 0.1 to 0.2 | 0.0 | 1.5 to 2.1 |
| 64 such calls, then one buffer that reads the ranges, per range (256 B and 4 KiB; 64 KiB) | +1.2; +15 | +0.1 to +0.3 | +0.1; +0.3 | +7.5 to +8.7; +33 |
| `didModifyData` of a linear texture over a Managed buffer, the call | 0.1 | 0.1 | 0.0 | 1.4 to 2.2 |
| `replaceRegion:` of that texture, the call | 0.6 to 4.6 | 1.1 to 5.6 | 0.3 to 1.2 | 1.7 to 3.4 |

At 1024 x 1024 BGRA8 (4 MiB) a Managed `replaceRegion:` call takes 402 us on the AMD and 320 on the
lane, a Shared one 404 and 3,541, and a new Managed texture's make 12.7 and 685: the lane's make
zeroes the texture's CPU copy (`RgfxTexture.mm`), where the AMD's takes its pages as it uses them.

So on every Mac *(class 3)* `replaceRegion:` of a Managed or a Shared texture and
`didModifyRange:` return without waiting for the GPU: the call is a copy into memory the CPU
holds, and the GPU's copy is made at the next commit (AMD) or the call writes the GPU's memory
itself (Intel, M3). The AMD pays 39 to 68 us a texture for that upload inside the next buffer.
Ours:
- **Managed:** the call is a copy into the CPU copy and a post of `TEXTURE_REPLACE_REGION` or
  `BUFFER_DID_MODIFY_RANGE` (about 1.5 us of it the kext's call), on par with the Macs. The host
  carries each out on its own before the context's next buffer (`decode/managed.rs`), at 85 to
  98 us a texture on the 5080: twice the AMD's, in metal2vulkan's `replace_region`, which submits
  and waits for each upload (read from its code at the 392ce3daf pin).
- **Shared:** the call is a staging blit the call waits for (`RgfxTexture.mm`, rgfxCopyRegion:),
  hundreds of times a Mac's. The AMD's Shared texture is an unsynchronized Managed one (Q3).

## Q18: when the CPU sees a signal from the middle of a command buffer

Probe: `oracle_cmdbuf DEVICE events`, 11-05. Times are ms from the commit, medians of 10 runs
(20 for the last row), on the 5300M and the 630 (macOS 26.7.1) and the M3 Max (macOS 27.0.1).
A buffer runs a kernel calibrated near 100 ms alone, signals a shared event to 1 and runs a
second such kernel (the 630's buffer of two ended at 100 ms):

| | AMD | Intel 630 | M3 |
|---|---|---|---|
| `signaledValue` first reads 1 (a thread polling it) | 101.3 | 100.5 | 90.3 |
| `waitUntilSignaledValue:1` returns | 101.3 | 100.5 | 90.3 |
| a listener's block for 1 runs | 101.3 | 100.9 | 90.3 |
| `GPUEndTime` | 198.6 | 100.4 | 179.8 |
| the completed handler runs | 198.7 | 101.0 | 179.9 |

A buffer runs a kernel of half that, signals 1, waits for 2 and runs another; the CPU waits for 1
and only then sets 2, from the waiting thread or from a listener's block for 1:

| | AMD | Intel 630 | M3 |
|---|---|---|---|
| the CPU sees 1 (each way) | 55.9, 56.0 | 102.1, 102.2 | 45.7, 46.7 |
| the buffer completed, 10 of 10 each way | at 105.5, 105.5 | at 131.0, 131.1 | at 90.8, 92.1 |

A buffer whose last command is the signal, after a half kernel: its listener's block runs
within 0.03 ms of its completed handler on both x86 GPUs, before it in 16 of 20 runs on the AMD
and 5 of 20 on the 630, and within 0.28 ms on the M3 (median 0.04), before it in 19 of 20.

So Metal's CPU value, waits and listeners see a mid-buffer signal before the buffer ends, and a
thread that waits on the CPU for it before it sets the value the same buffer waits for next goes
on, on all three *(class 3)*. How soon is a race *(class 2)*: the AMD and the M3 see it within a
few milliseconds of the GPU reaching it (the M3's listener 89.7 ms after `GPUStartTime`, one
kernel's time), the 630 about 100 ms after the commit in both shapes: at the end of a buffer that
ends by then, and some 70 ms after the GPU reached it in one that waits (its half kernel ran 25
to 29 ms on its own). A signal at the buffer's end reaches the listener and the completed handler
together, either first *(class 2)*.

Our stack: the host posted a buffer's signals only at its end, which a buffer that waits after its
signal for a value the CPU sets only after it saw the signal never reaches: a deadlock no Mac has
*(class 4)*. The host now posts a signal the buffer has work after as its own event reaches the
value, from a notification on a listener of its device's, as the AMD and the M3 do, and one at the
buffer's end with the buffer's completion, ahead of its record (design.md, "Shared events").

## The display's colour against the stock lane: deliberate divergences

Measured on a Linux host, in boots of 11-02: the stock lane's host shot of the desktop beside
ours, 20 minutes apart, and each lane's guest capture (`screencapture`) beside its host shot.

- **The profile.** The guest builds each display's ColorSync profile from its EDID. Ours
  (`makeEdid`, ReimsGraphics.cpp) sets the base block's "sRGB is the default colour space" bit
  (byte 0x18, bit 2), and macOS builds an sRGB profile: its tone curve is the sRGB curve (within
  1e-5), its colourants and white sRGB's and D65's, whatever the chromaticity bytes say. The stock
  framebuffer's EDID, made from the host's display descriptor (its default colour space: sRGB's
  primaries and white 0.312727, 0.329023, each truncated to 10 bits), clears the bit and gives the
  same gamma byte, 2.2, and macOS builds a profile of gamma 2.2 and those chromaticities. Both
  profiles' video-card gamma (`vcgt`) is the identity.
- **The pixels.** Both WindowServers render the same sRGB-encoded frame: text in the desktop's
  widgets matches within 1.5 levels (a mean of 0.3) once the stock host's curve is undone. Stock's
  display pipe then gives the host a gamma table and a colour matrix that re-encode the frame for
  its profile's gamma 2.2 (`displayPresentFragmentWithGammaTableAndColorMatrix`, shaders.metal):
  about enc2.2(dec_sRGB(v)) a channel, 16 to 23, 32 to 38, 64 to 67, 128 to 127 and 200 to 199.
  Our display pipe took no gamma table then (request 2 of RgfxDeviceControl answered none), and
  the host presented our frame as it is: the guest capture and the host shot agree within 2 levels
  for 98.5% of pixels. Request 2 now answers the AMD's gamma (design.md, "The display's colour"),
  so CoreDisplay sends our pipe its profile's tables; the pixels have not been compared since. On the host, the stock lane's dark tones are lighter than ours, by up to about
  7 levels.
- **Decided: the lane keeps its own.** Colour management depends on the display, and each lane
  is consistent with what it declares: ours declares sRGB and shows sRGB pixels, which is right
  for the usual sRGB monitor showing the host's window without colour management, and stock
  declares gamma 2.2 and re-encodes for it. Real Macs differ by panel too. Matching stock would
  take its EDID's colour bytes, a profile change that may move no pixel, or a gamma table and a
  colour matrix on our display pipe, through the kext, the wire and the host, and would make our
  pixels less faithful on an sRGB monitor.
- **Y'CbCr images: decided, the lane keeps the tag's matrix.** The default aerial wallpaper, a
  frame tagged ITU_R_709_2, shows by BT.709 on our lane and by BT.601 on the stock lane: that is
  the shift between the two lanes' guest captures, which the display pipe's gamma does not reach
  (Q13). CoreAnimation and VideoToolbox convert such a frame by BT.709 on the AMD, the Intel 630
  and the M3 *(class 3)*: ours is the Macs' answer, and the stock lane's BT.601 is its own.

## Q19: the words of two refusals, and the framebuffer's memory size

Probe: `oracle_names DEVICE [constants|archive]`, 11-05, each step in a child process of its own,
without and with `MTL_DEBUG_LAYER=1`; the framebuffer's properties are `ioreg`'s.

**A function constant given a value of another type.** A kernel with `constant int k
[[function_constant(0)]]` is made with a value of each of the 95 `MTLDataType` values at index 0.
On all three, only `MTLDataTypeInt` makes the function, and every other value is refused with
`MTLLibraryErrorDomain` code 3, whose description names the value's type by the SDK's name
(`MTLDataTypeFloat2x2`, `MTLDataTypeTexture`, `MTLDataTypeBFloat4`) *(class 3)*. The sentence around
the names is each OS's own *(class 2)*:

| | the description |
|---|---|
| AMD, Intel 630 (macOS 26.7.1) | "Constant k (0) is of type MTLDataTypeInt but value found has type MTLDataTypeFloat" |
| M3 (macOS 27.0.1) | "Function f: constant k (0) is of type MTLDataTypeInt but value found has type MTLDataTypeFloat" |

A `bfloat` function constant does not compile on any of them ("type 'bfloat' is not valid for
attribute 'function_constant'"). Ours gives the M3's sentence (`RgfxLibrary.mm`), and names each
type as the three Macs do (`RgfxFunctionConstants.cpp`, whose test holds the measured names).

**A mesh pipeline added to a binary archive** (`addMeshRenderPipelineFunctionsWithDescriptor:error:`):

| the descriptor | AMD, Intel 630 | M3 |
|---|---|---|
| no mesh function: no function, a fragment function or an object function | NO, `MTLBinaryArchiveDomain` code 3 (`MTLBinaryArchiveErrorCompilationFailure`), "Compiling mesh pipelines for binary archives is not supported!" | Metal's descriptor validation asserts ("meshFunction must not be nil.", and "fragmentFunction must not be nil." where it has none), and the process ends |
| a mesh function and a fragment function, the pipeline an app adds | the same | YES, no error: the archive records it |

The answers are the same with and without the validation layer. All three make that pipeline on
the device (`newRenderPipelineStateWithMeshDescriptor:`); the x86 GPUs refuse only its archive
*(class 2)*. Ours refuses every descriptor with the x86 Macs' error (`RgfxBinaryArchive.mm`). Its
device makes mesh pipelines where the host does (Q22).

**The framebuffer's `IOFBMemorySize`.** The AMD's framebuffer publishes 268435456 (256 MiB, the
size of its aperture), and the Intel 630's publishes none. Neither does the stock lane's paravirtual
framebuffer, in a macOS 26 x86 guest. Ours publishes none, as the Intel's does: IOFramebuffer's own
setup, for a new mode too, gets no VRAM range (`ReimsGraphics.cpp`, getVRAMRange). Both answers are
inside the envelope *(class 2)*.

## Q20: which textures over an IOSurface Metal makes, answers nil for, or ends the process at

Probe: `oracle_surftex DEVICE`, 11-05, each of its 178 cases in a child process of its own, with
`MTL_DEBUG_LAYER` unset. `newTextureWithDescriptor:iosurface:plane:` over a 64 x 64 surface:
`'BGRA'`, `'RGhA'` and `'L008'` made without planes (`IOSurfaceCreate`, as WindowServer's
surfaces are), `'BGRA'` of one plane, `'BGRA'` without planes of 512-byte rows, and a `420v`
CVPixelBuffer's two planes. Each case changes one thing of a 2D texture of the plane's own size
and format: the plane, the size, the type, depth, array length, levels and samples, the format,
the usage, the storage, CPU cache and hazard tracking modes, and the swizzle.

The AMD and the Intel 630 gave the same answer at every case. Where the process ends, Metal's
assertion lists every rule of one step a descriptor breaks, a line each. There are three steps:

| step | what ends the process there (AMD and Intel 630) | M3 |
|---|---|---|
| the descriptor's validation, `-[MTLTextureDescriptorInternal validateWithDevice:]`, over every surface | a width or height of 0; a format the device lacks, named by number ("has invalid pixelFormat (40)": B5G6R5, BGRA10_XR, BGR10_XR, Invalid, and every two-plane format, Y'CbCr or extended-range, over any plane); a depth, array length or sample count above 1 that the type does not allow, the type named ("requests a depth of 2 but the texture type MTLTextureType2DArray requires that depth is 1"); a multisample texture of one sample, or in Shared or Managed storage; Memoryless storage; a depth or stencil format in Shared or Managed storage; a 4:2:2 format of an odd width ("MTLPixelFormat422 requires even width.") | the same, but it makes B5G6R5, BGR10_XR, every two-plane format (Q13) and Memoryless textures, and passes multisample ones in Managed storage on to the next step |
| the surface's rows, `_mtlValidateStrideTextureParameters`, over a surface without planes | a type other than 2D or TextureBuffer; a depth other than 1; more than one level; rows of a format of one texel a block wider than the surface's `bytesPerRow` ("bytesPerRow (256) must be greater or equal to (260) bytes"); a depth or stencil format ("cannot create compressed, depth, or stencil textures", though BC1 and BC7 pass) | the same, in the same words |
| the texture's place, `-[MTLIOAccelTexture initWithDevice:descriptor:iosurface:plane:...]`, over a surface without planes | a plane other than 0 ("IOSurface is non-planar, but plane 2 specified"); a compressed format's rows, in blocks, wider than the surface's ("Descriptor rowBytes (264) > IOSurface rowBytes (256)" at BC1 132 wide); more rows than the surface's, a compressed format's in texels ("Descriptor height*arrayLength > IOSurface height") | the same, `IOGPUMetalTexture`'s |

Over a plane of a surface with planes only the first step applies on the x86 Macs: a texture
wider or taller than its plane, of another format's bytes, of any type (1D, 2DArray, Cube, 3D of
depth 2, TextureBuffer, 2DMultisample in Private storage), mipmapped, of Depth32Float in Private
storage or of BC1 is made. The M3 makes the same but the Cube, at which its own assertion ends the
process ("carriesComplexTexture || numFaces == 1"), and the BC1, for which it answers nil. A plane
past a planar surface's planes answers nil on all three. All three make the formats of one texel a
block whose rows fit, a surface of 512-byte rows taking a texture 128 wide, the 4:2:2 formats of
an even width (an odd height too, as a render target as well), every usage (usage 0 answers 0x17),
every storage but Memoryless, and every CPU cache and hazard tracking mode and swizzle; and a
TextureBuffer of one row. The x86 Macs make BC1 and BC7 textures wherever their rows fit, where the
M3 answers nil, over a surface with planes or without, in Managed or Private storage. The M3
answers nil for a render target of the 4:2:0 Y'CbCr format 500 of an odd width or height.

The two x86 GPUs differ only in the words at `Depth24Unorm_Stencil8` in Managed storage: the
AMD's are the depth formats' storage rule, the Intel's "invalid pixelFormat (255)", as the Intel
lacks the format.

What our stack does:
- *(class 3)* The device's creator ends the process at every rule the x86 Macs end it at, with
  their words, before it sends the host anything (`rgfxEndAtInvalidSurfaceTexture`,
  RgfxTexture.h). The bundle had sent each of these to the host, posted, and answered a texture;
  the host refused most, and the guest's command buffers naming the texture then failed. That
  includes every two-plane format, which the lane made as the M3 does until 11-05 (Q13).
- *(class 3)* A BC1 to BC7 texture over a surface is made and posted, as both x86 Macs make it
  wherever its rows fit (`RgfxIOSurfaceTexture`). The Vulkan runtime makes it in their form since
  metal2vulkan's stack 34, row j of its blocks at the plane's row j, over a surface without planes
  and over a plane at an offset and rows that are multiples of 16 bytes
  (`bc_textures_over_a_surface_s_rows_are_made`). The bundle's own CPU access takes the same rows
  (`test_iosurface_block_compressed_textures_are_made`). Until then the bundle answered nil for
  every block-compressed texture over a surface, as the runtime made none.
- *(outside the envelope, the host's limit)* Metal's buffer creator answers nil for a BC1 or BC7
  texture over the buffer that stands for a surface's plane, as the M3's own IOSurface creator
  does (measured on the M3 under macOS 27).
- What the x86 Macs make, the Vulkan runtime still refuses in places, as its
  `NotImplemented(SurfacePlaneShape)` at stack 34, where the three GPUs lay the texels out no one
  way: over a planar surface's plane, a texture wider than its rows, a mipmapped one, a 2DArray of
  more than one slice, a 3D, Cube or 2DMultisample one, a 1D one wider than a row, and a BC
  texture or a TextureBuffer at an offset or rows off 16 bytes; and, over any surface, a texture
  past the surface's end (`TexelsPastBufferLength`). The bundle sends these, posted, and the
  host's refusal fails the command buffers that name them (`refused.rs`). They are the host
  runtime's gaps. Stack 34 makes a TextureBuffer over any surface, and over a planar surface's
  plane a 1D texture whose row fits and a 2DArray of one slice, each laid out as the 2D texture of
  its rows.
- *(outside the envelope, the runtime's M3 form)* A texture taller than its plane, inside the
  surface, is made as the M3 lays it out, its rows past the plane at the plane's pitch. On both
  x86 GPUs the bytes past the plane are left unwritten (metal2vulkan's byte probe), which the
  runtime cannot follow, as it is given no plane height.

## Q21: pipelines' reflection

Probe: `oracle_reflection DEVICE [CASE]`, 11-05, with `ORACLE_IVARS=1`; and the bundle's
`tools/pipeline_reflection_check.mm`, which makes the same pipelines on the device and compares
the device's reflection with the bundle's, getter by getter and instance variable by instance
variable. Both run over `oracle_reflection.metal`'s kernels, vertex, fragment, object and mesh
functions: buffers of every scalar, vector, matrix and struct type, argument buffers, threadgroup
memory, every texture type and access, arrays of textures and samplers, arguments the function
never references, arguments under function constants, stage inputs, built-ins and payloads.

**What the creators answer.** A pipeline made with options of neither
`MTLPipelineOptionBindingInfo` nor `MTLPipelineOptionBufferTypeInfo` answers no reflection (nil);
either, or both, answers the same whole reflection, an `MTLComputePipelineReflectionInternal` or
`MTLRenderPipelineReflectionInternal`, in the synchronous creators and in the completion handlers
alike. AMD and Intel agree on every value but one (below) *(class 3)*:
- each stage's bindings are its function's resource arguments, in the order the function declares
  them, after the buffers its vertex or stage input descriptor reads; `-bindings` and `-arguments`
  are two arrays of the same objects. Its built-in arguments are listed apart, a fragment
  function's nowhere. `builtInType` numbers them: `vertex_id` 1, `instance_id` 2, `base_vertex` 3,
  `base_instance` 4, `thread_position_in_grid` 5, `threads_per_grid` 6,
  `thread_position_in_threadgroup` 7, `thread_index_in_threadgroup` 8, `threads_per_threadgroup` 9,
  `threadgroup_position_in_grid` 10, `threadgroups_per_grid` 11, `thread_index_in_quadgroup` 17,
  `thread_index_in_simdgroup` 18, `threads_per_simdgroup` 20, `mesh_grid_properties` 23, a mesh
  24; `simdgroup_index_in_threadgroup`, `simdgroups_per_threadgroup`,
  `quadgroup_index_in_threadgroup`, `quadgroups_per_threadgroup` and
  `dispatch_threads_per_threadgroup` are listed nowhere. An empty list is a nil instance
  variable, which `-...Bindings` answers as an empty array and `-...Arguments` as nil;
- a texture's type (`dataTypeDescription`) is an `MTLTextureReferenceTypeInternal` of data type
  texture and of the texture's sample type (`textureDataType`), as the binding answers; an
  array member's type is of its element's data type and of the element's size's stride;
- an argument a false function constant gates is left out; one the AIR marks `air.arg_unused` is
  unused (`isUsed` and `isActive` 0), built-ins among them. An argument the function reads only in
  code its constants remove is unused too;
- a constant buffer's alignment is 32, where the M3 answers its pointee's (16 for a `Params` of
  `float4x4`) *(class 2)*. The M3's answers differ only there and in the private `_flags` of
  its pipelines' reflection (register spills and atomics, which it sets and the x86 GPUs leave
  0);
- a descriptor's buffer is `vertexBuffer.N` (`stageInBuffer.N`), N its place among the
  descriptor's buffers in buffer index order (buffers 1, 4 and 30 are `vertexBuffer.0`, `.1` and
  `.2`, at indices 1, 4 and 30): a struct buffer of no struct type, its layout's stride,
  alignment 4, used where the function reads an attribute it holds, and the only binding that is
  no argument (`isArgument` 0);
- the private `postVertexDumpStride` is the vertex function's outputs' size and 12 on AMD and on
  the M3, and its outputs' size and 8 on Intel *(class 2)*; a mesh pipeline's is 0.

**Ours** (`RgfxPipelineReflection.mm`): the bundle builds the same objects, of Metal's own classes,
from each function's AIR nodes, and answers them in every creator that takes options, with AMD's
stride. It answers nil, as before, for a pipeline of a function whose binding index function
constants compute (`[[buffer(kIndex)]]`), whose module the reader does not follow, or with a type,
built-in or binding no x86 Mac's answer was measured for. An argument read only in code its
constants remove is answered used *(class 4: Metal reflects the specialised code)*.
`pipeline_reflection_check` (11-05, 80 cases of 433490 values) finds no difference on the AMD
GPU, and on the Intel GPU only `postVertexDumpStride`, 4 less than ours on each vertex pipeline;
on both, the one case ours does not make is `c_constants` (`k_constants`), whose binding index
function constants compute.

## Q22: mesh render pipelines and their draws

Probe: `oracle_mesh DEVICE`, 11-05, each step in a child process of its own, without and with
`MTL_DEBUG_LAYER=1`. The SPI answers are `tools/base_answers_probe.m`'s. Plain Metal unless said.

**The device.** The AMD and the Intel 630 answer Metal3 and Mac2 yes and Apple7 to Apple9 no; the
M3 answers all five yes. All three answer the SPI's `supportsMeshShaders` YES. The x86 Macs answer
`supportsMeshShadersInICB` NO and `requiresMeshShaderEmulation` YES, and the M3 answers YES and NO.
Ours answers the Apple families no and Metal3 and Mac2 as the host's platform does (`device_info.rs`).
It offers `MESH_PIPELINES` where the platform answers Metal3, and then answers `supportsMeshShaders`
YES. It leaves `supportsMeshShadersInICB` and `requiresMeshShaderEmulation` to the IOGPU base's NO
*(class 2)*.

**What a made pipeline answers.** The descriptor is oracle_mesh's, with and without an object
function:

| query | AMD | Intel 630 | M3 |
|---|---|---|---|
| `maxTotalThreadsPerThreadgroup`, `gpuResourceID` | 0 | 0 | 0 |
| `maxTotalThreadsPerMeshThreadgroup` | the descriptor's, else 1024 | 1024 | the descriptor's, else 1024 |
| `maxTotalThreadsPerObjectThreadgroup` | as the mesh stage's; 0 with no object function | 1024; 0 with no object function | as the mesh stage's; 3 with no object function |
| `objectThreadExecutionWidth` | 32; 0 with no object function | the same | 32 |
| `meshThreadExecutionWidth` | 32 | 32 | 32 |
| `maxTotalThreadgroupsPerMeshGrid` | `NSUIntegerMax` | `NSUIntegerMax` | 2^20, else the descriptor's |
| `imageblockSampleLength` | 0 | 0 | 4, and 0 without rasterization |
| `requiredThreadsPerObjectThreadgroup`, `requiredThreadsPerMeshThreadgroup` | the descriptor's | the descriptor's | the descriptor's |

All three keep the label, and make the pipeline asynchronously, off the main thread, with a nil
reflection. With reflection options, the error form answers a reflection with one mesh binding,
and three object bindings where the pipeline has an object function. Ours answers each limit as
the host's platform answers it (`RgfxRenderPipelineState.mm`; on the M3, its numbers), the
required sizes as the descriptor's *(class 3)*, and the grid's `NSUIntegerMax` where the platform's
answer passes 32 bits *(class 2)*.

**What ends the process, and what is refused.** The same with and without the validation layer,
except where the table says so:

| the descriptor | AMD | Intel 630 | M3 |
|---|---|---|---|
| no mesh function | `validateWithDevice:1884: failed assertion` "Mesh Render Pipeline Descriptor Validation\nmeshFunction must not be nil.\n" | the same | the same, at line 1956 |
| no fragment function, rasterizing | the same, "fragmentFunction must not be nil.\n" | the same | the same |
| `supportIndirectCommandBuffers` YES | the same, "Use of mesh shader pipelines in indirect command buffers is not supported by this device.\n" | the same | made, `supportIndirectCommandBuffers` YES and a `gpuResourceID` of 1 |
| a stage's `maxTotalThreads...` of 2048 | `CompilerError` 2, "maxThreadsPerThreadgroup pipeline descriptor value (2048) exceeds maximum supported (1024)\n" | made; it answers 1024 | refused, `AGXMetalG15X_M1` 3 |
| `payloadMemoryLength` 16385 | `CompilerError` 2, "payloadMemoryLength property of MTLMeshRenderPipelineDescriptor (16385 bytes) exceeds maximum supported by this device (16384 bytes)" | the same | refused, `AGXMetalG15X_M1` 3 |
| a vertex function as the mesh function, rasterizing or not | `CompilerError` 2, "RasterizationEnabled is false but the vertex shader's return type is not void" | the same | SIGSEGV |
| an object function as the mesh function | `CompilerError` 1, "Compilation failed due to an interrupted connection: XPC_ERROR_CONNECTION_INTERRUPTED. This error occurred after multiple retries." | the same | refused, `AGXMetalG15X_M1` 2, the same words |
| a mesh function as the fragment function | the same as the row above | the same | SIGSEGV |
| a fragment function as the object function | SIGSEGV | `CompilerError` 1, the XPC words | SIGSEGV |
| a mesh function as the object function | made | made | refused, `AGXMetalG15X_M1` 2, the XPC words |
| `FailOnBinaryArchiveMiss`, an empty archive named | `CompilerError` 2, `Unable to find function "f_col" in binary archives. \n Expected <MTLBinaryKey: 0x…> Legacy key:\n` and 64 zeros, the fragment function named, else the mesh function | the same | refused, `AGXMetalG15X_M1` 3, its own words |
| `FailOnBinaryArchiveMiss`, no archive named | SIGSEGV | SIGSEGV | made |

The validation layer asserts at the setters for a function of another stage
(`-[MTLMeshRenderPipelineDescriptor setMeshFunction:]:1624: failed assertion` "meshFunction
functionType is not MTLFunctionTypeMesh.", and the same at `setObjectFunction:` and
`setFragmentFunction:`). An x86 Mac's archive holds no mesh pipeline (Q19), so on x86 every mesh
pipeline made with `FailOnBinaryArchiveMiss` and an archive misses.

Ours keeps the three deaths, in the x86 Macs' words: they are where Metal's validation ends the
process on x86 *(class 3 for the first two, class 2 for indirect command buffers)*
(`RgfxRenderPipelineState.mm`, `rgfxEndAtInvalidMeshDescriptor:`). It refuses the rest before the
host is asked, in the x86 Macs' domain and codes *(class 2)*:
- the payload and the 5300M's maximum, in their words;
- the archive miss in their words, without the key;
- every function of another stage, with the validation layer's words in place of the XPC ones.

The host refuses a function of another stage too, before its GPU's Metal sees it (the Metal
platform's `wrong_mesh_stage`), since the M3's Metal ends the host's process at three of them.
With no archive named it makes the pipeline, as the M3 does *(class 2)*.

**The draws**, into a 16 x 4 target, each mesh threadgroup covering its column of the grid in
the colour its draw names:

| draw | AMD | Intel 630 | M3 |
|---|---|---|---|
| `drawMeshThreadgroups:` | red | red | red |
| `drawMeshThreads:` | nothing, in every form | nothing | drawn |
| `drawMeshThreadgroupsWithIndirectBuffer:` | red | wrong colours (a gradient) | red |
| `drawMeshThreads:` with an object stage | nothing | drawn | drawn |
| an object stage's `drawMeshThreadgroups:` | drawn | drawn | drawn |

The documentation promises each draw *(class 1)*. Ours sends each draw to the host, and the
host's GPU draws it.

**Indirect command buffers.** All three make one of the mesh command types. On x86 no pipeline
can be set in it, since a mesh pipeline that supports indirect command buffers ends the process
(above). The validation layer says "MTLValidateFeatureSupport:8086: Mesh shader draws can be
encoded into ICBs is not supported on this device". The M3 draws from one. Ours keeps the x86
Macs' *(class 2)*: no mesh pipeline it makes supports them, so its indirect commands carry no
mesh draw (`RgfxIndirectCommandBuffer.mm`).

## Q23: shared events across processes

Probe: `oracle_events DEVICE`, 11-05, on the 5300M and the 630 (macOS 26.7.1), identical on both
but for timings. Every case runs in a child process of its own; `spawn` and `kill` run a second
process.

- **The classes.** A shared event is `_MTLSharedEvent`, a subclass of `IOSurfaceSharedEvent`,
  whose `_eventPort` (`I`) and `_signaledValue` (`r^Q`, a pointer: the value is a page the kernel
  shares) are its own; its handle is `MTLSharedEventHandle`, of one private struct (`{I port, @
  label, Q labelTraceID}`), made by `initWithSharedEvent:`. Both adopt `NSSecureCoding`. The device
  answers `newSharedEventWithHandle:` (`@24@0:8@16`) and the SPI `newSharedEventWithMachPort:`
  (`@20@0:8I16`).
- **The port and its references.** `eventPort` is the same name at each ask, a send right the
  event holds one reference of. A handle's `initWithSharedEvent:` asks the event its `eventPort`,
  `label` and `labelTraceID` (an object of the probe's own, answering each, made a handle of its
  port, which NSXPC carried to the far side) and takes a reference of its own; an event made from
  the port or a handle takes one more, is a new object (`sameObject=0`) of the same value and the
  same `eventPort`; each lets its reference go at its dealloc. With a reference kept, an event made
  from the port after the first event went has its value.
- **What it refuses.** `newSharedEventWithMachPort:` of `MACH_PORT_NULL`, `MACH_PORT_DEAD`, a send
  right of a port the probe receives, and `mach_task_self()` answer nil, the port's references as
  they were; `newSharedEventWithHandle:` of nil answers nil.
- **Labels.** A handle keeps the label the event had at its make (a relabel after leaves it); an
  event made from the handle has the handle's; one made from the port has none.
- **Coding.** The event and the handle each code only through an NSXPCCoder:
  `encodeWithCoder:` of a keyed archiver raises `NSInvalidArgumentException`, "This object may
  only be encoded by an NSXPCCoder.", `initWithCoder:` of a keyed unarchiver "This object may only
  be decoded by an NSXPCCoder.", and the convenience archiver answers no data and
  `NSCocoaErrorDomain` 4866, secure or not. Through an anonymous NSXPC listener, a handle arrives
  as a handle of the same port (made of the probe's own object, too), and the event itself
  arrives as an event, not the same object, of its value.
- **Values.** Two events of one port in one process read each other's CPU sets at once; a
  listener of one fires on the other's set and on its GPU signal; a GPU wait on one is released
  by a set of the other. Across two processes (`spawn`): the peer's event made from the port reads
  the probe's value; a CPU set in the peer is read in the probe at once; a GPU signal in the probe
  fires the peer's listener before the probe tells it; a GPU wait in the peer is released by a
  CPU set in the probe, and its signal after is seen in the probe within 0.2 ms; after the probe
  let go of its event, the peer's sets and waits go on.
- **The port outlives its maker** (`kill`). A peer makes an event, sends its port and is killed
  with SIGKILL: the probe's send right stays (2 references, a send right), and an event made from
  the port after the kill has the value and the same `eventPort`. A CPU set of it is read at once
  in the event made before the kill, and a GPU signal of that one releases a CPU wait on the other.
- **IOKit's calls on ports that are no shared event's** (`connect`). `IOConnectAddClient` of a
  port to an IOSurfaceRoot connection, which takes the port as its argument, never as its
  destination, answers `kIOReturnBadArgument` (0xe00002c2) for `mach_task_self()` and for a send
  right of a port the probe receives, and `kIOReturnUnsupported` (0xe00002c7) for the connection
  itself. `IOObjectGetClass` and `IOConnectCallMethod` answer `kIOReturnBadArgument` for the
  connection and the task port, and `IOObjectConformsTo` false. The first call made on the
  received port as its destination (`IOObjectGetClass`) kills the process (SIGKILL) before it
  answers.

**Ours** (design.md, "Shared events across processes"): the port is a connection of the kext's
(the kext can link no call that gives a process a port of its own objects), the handle Metal's own
over it, and both processes' events one host event, so GPU signals and waits, CPU sets, waits and
listeners work across them, and the references, labels, refusals and coding are as measured. The
port outlives a maker killed with SIGKILL, as measured: the host event lives as long as the port.
Before it calls on a port it was handed, the bundle asks `IOConnectAddClient` of it to its own
connection, which the kext answers success only for its own shared event ports, so a port of
another process's own is answered nil, where a call on it would kill the process. Two
events of one port in one process are one object's, so they read each other at once. Across
processes a value reaches the other process as the host posts it there, so a read right after the
other's CPU set may see the value before it, where Metal's shared page is read at once *(class
4)*.

## Q24: which texture descriptors each creator makes or ends the process at

Probe: `oracle_texdesc DEVICE CREATOR`, 11-05, each of its 108 cases in a child process of its
own, with `MTL_DEBUG_LAYER` unset, at the device's `newTextureWithDescriptor:`, a Private
Automatic heap's of 256 MiB, and a Shared buffer's `newTextureWithDescriptor:offset:bytesPerRow:`.
Each case changes one or more things of a 64 x 64 2D BGRA8Unorm texture of one level and sample,
usage ShaderRead, in Private storage (Shared over the buffer): its size, type, depth, array
length, levels, samples, format, usage or storage. Its last 13 cases (11-07) are RGBA32Float
textures of 2 GiB to 8 TiB, at the device's creator.

Every creator runs the descriptor's validation first,
`-[MTLTextureDescriptorInternal validateWithDevice:]`. Where it fails, one assertion, "Texture
Descriptor Validation", lists every rule the descriptor breaks, a line each, in this order on
both x86 Macs *(class 3)*:
1. A width, height, depth or array length of 0, or past what the type takes ("has height (2)
   greater than the maximum allowed size of 1."): 16384 a side, 2048 for a 3D texture's sides,
   one row for the 1D types and a TextureBuffer, one slice for every type but 3D, one element
   for every type but the arrays, and 2048 for an array.
2. A format the device lacks, by number ("has invalid pixelFormat (40)."): Invalid, B5G6R5,
   A1BGR5, ABGR4, sRGB R8 and RG8, the XR10 formats, ASTC, ETC2, EAC, PVRTC, Unspecialized
   (263), the two-plane ones (500), and, on the Intel only, Depth24Unorm_Stencil8 and
   X24_Stencil8.
3. What the type takes: a 1D type's height, a depth for every type but 3D and an array length
   for every type but the arrays ("requests a depth of 2 but the texture type MTLTextureType2D
   requires that depth is 1"), and an array of 0 elements or more than 2048, or 341 for a cube
   array.
4. Levels: a 1D type's other than 1, a 4:2:2 format's above 1, and 0 or more than the size
   holds ("requests 8 mipmap levels, but the dimensions (64, 64, 1) can only support a maxiumum
   of 7 levels"). Then a cube that is not square.
5. Multisample textures: levels other than 1; a format that has none (BGR5A1Unorm, RGB9E5Float,
   the compressed and 4:2:2 formats, and one the device lacks, named Invalid); a sample count
   other than 1 on another type; a count of 1, or one the device does not make (3 and 16; it
   makes 2, 4 and 8); and a compressed format with samples.
6. A compressed format of a type other than 2D, 2DArray, 3D, Cube or CubeArray; a depth or
   stencil format of 1D, 1DArray, 3D or TextureBuffer; a 4:2:2 format of a type other than 2D,
   or of an odd width.
7. Storage: a depth, stencil or multisample texture in Shared or Managed storage; Memoryless
   storage; and a mode Metal has no name for (5), whose line comes twice. Then usage bit 0x80.

The M3 Max under macOS 27.0.1 (11-09, every case but the 2 GiB and larger ones and the
16384 x 16384 one, at all three creators) lists the rules it shares with the x86 Macs in their
words and order, and differs in these: it has every format the x86 Macs lack but Invalid,
Depth24Unorm_Stencil8 (255) and X24_Stencil8 (262), making B5G6R5, A1BGR5, ABGR4, sRGB R8 and RG8,
BGRA10_XR, BGR10_XR, ASTC, ETC2, EAC and PVRTC, and answering nil at the device's and the heap's
creators for the two-plane format 500 and Unspecialized; it makes multisample BGR5A1Unorm and
RGB9E5Float textures, depth, stencil and multisample textures in Shared and Managed storage, and
Memoryless ones, a Private heap's among them; and it makes 2 and 4 samples but not 8 ("sampleCount
(8) is not supported by device"). Its heap names its class `IOGPUMetalHeap` in the storage
assertion. Over a buffer it answers nil for every compressed format, BC1 included. The bundle
takes the M3's rules where the device answers an Apple family, as it does in an arm64 guest
(`rgfxAppleGPU`, RgfxTexture.h).

A texture type past TextureBuffer (10) ends the process at every creator with no words. A pixel
format value no format has (9999) ends it at the descriptor's setter, which is Metal's own in a
guest too. After the validation, a heap's creator ends the process at a descriptor of another
storage mode than the heap's ("The requested storage mode (MTLStorageModeShared) is not
compatible with the heap's mode (MTLStorageModePrivate)", in
`-[MTLIOAccelHeap newSubResourceWithLength:alignment:options:offset:]`), and otherwise answers as
the device's. A buffer's creator then checks its own rules (Q11). The two GPUs differ at
Depth24Unorm_Stencil8 and X24_Stencil8, which the AMD makes, and over a buffer: there the Intel
ends the process at a depth other than 1, 0 included, before the validation, with "Buffer
texture: cannot support depth > 1." (`validateTextureDescriptor`), where the AMD lists the
validation's depth lines.

The device's creator answers nil, past the validation, for a texture of more bytes than the
device makes, on all three; creating one that is made commits no memory (the process's footprint
stays under 3 MB). By the bytes `heapTextureSizeAndAlignWithDescriptor:` answers, which here are
the texture's own:

| texture | AMD (`maxBufferLength` 3208642560, `recommendedMaxWorkingSetSize` 4278190080) | Intel 630 (2147483648, 1610612736) |
|---|---|---|
| 16384 x 8192 (2 GiB) | made | made |
| 16384 x 8193 (2 GiB and a row) | made | nil |
| 3 and 4 GiB (2D), 8 and 16 GiB (2DArrays of 16384 x 16384) | made | nil |
| 20, 24, 28, 32 and 64 GiB; a 2DArray of 2048 slices (8 TiB); a 3D 2048 a side (128 GiB) | nil | nil |

The Intel answers nil past its `maxBufferLength`, as it does a buffer and a heap past it (Q8);
the AMD makes one of 16 GiB, just over 4 times its working set, and none of 20 GiB. The M3 Max
under macOS 27.0.1 answers nil for the 8 TiB and the 128 GiB textures (the smaller ones were not
asked of it, as its memory is the desktop's).

What our stack does:
- *(class 2)* The device's creator answers nil for a texture of more bytes than the device's
  `maxBufferLength`, every level of every slice and sample, sending nothing
  (`rgfxTexturePastLength`, RgfxTexture.h), as the Intel does and as the bundle answers a buffer
  and a heap past it. Until 11-07 the bundle sent it, posted, and answered a texture whose
  create the host's platform could refuse for its size: a texture the app held, whose command
  buffers' draws the host would then drop (`refused.rs`).
- *(class 3)* Every texture creator runs one copy of the validation before it sends the host
  anything (`rgfxEndAtInvalidTextureDescriptor`, RgfxTexture.h): the device's, both of a heap's,
  a buffer's public form and its linear and tiled SPI, and the creator over an IOSurface (Q20).
  It takes the Intel's answer at the two depth-stencil formats, as the lane lacks them, and
  checks a sample count against the device's `supportsTextureSampleCount:`. A heap's creators
  then end the process at a storage mode other than the heap's (`rgfxEndAtHeapStorageMismatch`).
  Until 11-05 the bundle copied a few of these rules and sent the rest to the host, posted: a
  texture the app held, whose command buffers' draws the host then dropped (`refused.rs`).
- *(class 2)* Over a buffer, at a depth other than 1, the AMD's words.
- *(class 2)* Over a Shared buffer by the public form, a block-compressed format answers nil, as
  the AMD answers it 64 x 64 and 62 x 64, where the Intel makes it; the host's platform makes no
  such texture over a buffer's bytes either. Both make a single block's (2 x 2), which ours
  answers nil too *(class 4)*. So it does a TextureBuffer, which all three make by that form
  *(class 4)*: the host carries none by it.
- The shell tests check every case the device answered (`kDescriptorCases`) and each creator's
  order (`kCreatorCases`).

## Q25: what a command buffer does with nil where an object belongs

Probe: `oracle_nilbind DEVICE`, 11-05, each of its 27 cases in a child process of its own, with
`MTL_DEBUG_LAYER` unset. The target is a 4 x 4 RGBA8Unorm texture cleared to 64s. A draw covers
it and writes red one half plus half of what its fragment function read, so a read of zero
draws (128, 0, 0, 255) and a draw that did not run leaves 64s; a dispatch writes what it read and
a marker into a buffer. The texture bound where one is bound is 1 x 1 of 255s, the buffer a
float4 of 1s.

| case | AMD | Intel 630 |
|---|---|---|
| a fragment or kernel function reads a texture bound nil, or never bound | the draw or dispatch runs, reading zero | the same |
| a fragment function samples a bound texture with a sampler bound nil | it runs, reading the texture | it runs, reading zero |
| a fragment or kernel function reads a buffer bound nil, or never bound | it runs, reading zero | it runs, reading other values ((255, 0, 255) drawn, -2.8e38 dispatched) |
| a kernel writes a texture bound nil | it runs, and the write goes nowhere | the same |
| a render pass of color attachment 0 nil (its size set or not, a pipeline of the target's format or of none, color 1 the target), or of a depth attachment nil beside color 0 | the encoder is made, its draws run against what is attached, and the command buffer completes | the same |
| `setRenderPipelineState:nil`, then a draw | the process ends (SIGSEGV) | the same |
| `useResource:nil` | the process ends (SIGSEGV) | it goes on, and the draw runs |
| a blit from or to nil: a copy from a texture into a buffer and from the target into a buffer, a fill, `generateMipmapsForTexture:` | the process ends (SIGSEGV) | the same |
| a copy from a nil texture into the target | the process ends (SIGSEGV) | the command buffer completes, and nothing is copied |

Every command buffer that ran completed with no error. So a draw or dispatch with nil bound runs
on both *(class 3)*, a nil texture reading zero *(class 3)*, a nil buffer zero on the AMD *(class
2)*, and a nil sampler the default one's sampling on the AMD and zero on the Intel *(class 2)*.

What our stack does with an object whose posted create the host refused (`decode::refused`):
- *(class 3, and class 2 the AMD's)* A binding of a buffer, texture or sampler of one binds nil,
  and the encoder's draws and dispatches run, as with nil bound. Nil is the platform's: on the
  Vulkan platform metal2vulkan's neutral image, sampler and zero buffer, which read zero and
  sample by the default sampler, as the AMD. Until 11-05 they
  were dropped while such a slot was bound.
- *(class 4)* A render pass with an attachment of one is skipped whole, where both Macs run it
  without it: an assertion in the host's Metal at a pass with no attachment would end the host's
  process, and the host's platform was not measured there. A pipeline state of one is not set, and
  the draws are dropped until another is, and a blit from or to one is dropped, where both Macs
  end the process at nil: the bundle does not know the create was refused, so cannot end it.
- A use list's entry of one is nil, as the Intel goes on.

## Not measured

- Cross-process lock waits for pending GPU work: rgk-lock's `lock_probe` (x86 matches the M3,
  except that a read-write lock does not wait for a buffer that only reads the surface).
- Page faults: no probe made a GPU address fault, on a Mac whose Intel GPU drives the display.
- Surfaces whose rows miss the host's alignment.
