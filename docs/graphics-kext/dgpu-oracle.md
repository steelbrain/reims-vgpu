# The dGPU oracle: what an x86 Mac's Metal leaves loose

Every Metal rule the reims-graphics lane imitated was first measured on an Apple M3 Max, a
unified-memory tile-based GPU. The lane's guests are x86-64 macOS, whose apps and frameworks
were written against Intel Macs: an AMD discrete GPU, an Intel iGPU, or both. This report
measures those GPUs as a black box, with probes of our own, and compares each answer with what
our stack enforces. Where the x86 contract is looser, the host may do less work; where it is
stricter than the M3's, guests may rely on it.

The probes are in `guest/bundle/tools/oracle` ([README](../../guest/bundle/tools/oracle/README.md)):
`oracle_iosurface`, `oracle_storage`, `oracle_passes`, `oracle_cmdbuf`, `oracle_caps` and
`oracle_errors`.

**Machines.** Every claim below is measured, on:
- **AMD**: AMD Radeon Pro 5300M (Navi 14, `amdgpu_gfx1012`, 4 GB; AMDRadeonX6000 on
  IOAcceleratorFamily2), in a MacBookPro16,1 (i7-9750H), macOS 26.7.1 (25G241), x86_64.
- **Intel**: Intel UHD Graphics 630 in the same Mac. It drives the Mac's display.
- **M3**: Apple M3 Max, Mac15,9, macOS 27.0.1 (26A434), arm64. It ran only the probes where it
  shows the gap.

The two x86 GPUs agree on every coherence question except the ones named, so "x86" below
means both.

## Ranked relaxations

| # | relaxation | evidence | what it removes | expected win |
|---|---|---|---|---|
| 1 | **An IOSurface's GPU copy is a device cache.** Copy the surface's pages in only at the first GPU use after a CPU read-write unlock, and write the GPU copy back only when a CPU lock follows a GPU write. | Q1: on x86 an unlocked CPU write is never seen by the GPU, and a GPU write is never seen in the pages without a lock, the committing process included. | m2v's whole-surface shadow copy-in before each command buffer, and its merge-back after each writing encoder, of every render-target surface. The copy path's per-command-buffer write-back on the guest CPU. | In WindowServer's steady state nothing CPU-locks the 3840x2400 display surface, so both copies of 36.9 MB per composite go to zero. That is most of the pl1 gap: CLTD 28.2 ms against stock's 0.53. The copies x86 does make cost 5.6 to 7.6 ms each at 4K, paid only at those locks. |
| 2 | **`commitAndWaitUntilSubmitted` and `waitUntilScheduled` return at submission.** | Q7: x86 returns 0.29 to 0.41 ms after commit, behind a 200 ms command buffer, as the M3 does. | The wait for completion the bundle restored (bbb153d9e), once rgk-lock's lock wait lands. | rgk-perf's fix 1 measured CLTD 19.6 to 6.1 ms. This is the M3's rule too; the oracle confirms x86 has no stricter one. |
| 3 | **Asynchronous pipeline creators return before the compile.** | Q7: the handler runs about 60 ms later, on Metal's `Device serial queue`. | Our bundle compiles on the caller's thread, inside the call. | Compile time moves off apps' calling threads (launch, first frames). Nothing in WindowServer's steady state. |
| 4 | **Shared textures' CPU access on a discrete host may use a CPU copy.** | Q3: on AMD, `getBytes:` of a Shared texture never sees GPU writes, and `replaceRegion:` uploads the CPU copy whole. | The waited staging blit of each `getBytes:`/`replaceRegion:` of a Shared texture. | Small: the calls are rare, and Intel and the M3 keep such textures coherent. |

No relaxation was found for Managed storage (Q2: our discrete path already keeps two copies
synced by `didModifyRange:` and `synchronizeResource:`, as AMD does), Shared buffers (Q3:
coherent at command-buffer boundaries on x86, and ours are in place), load and store actions
(Q4), memoryless, tile and fetch features (Q5), hazard tracking (Q6), errors (Q8) or limits
(Q9). In Q4 and Q6 the x86 GPUs are stricter than the M3 (below).

## Where x86 is stricter than the M3 rules we copy

Guests may rely on these, since every x86 Mac behaved this way:
- **DontCare stores keep color.** A pass with store action DontCare writes its draws and
  clears to memory on both x86 GPUs. On the M3 it drops them, and m2v copies nothing back for a
  DontCare store (`render/shadows.rs`), as the M3 does (Q4).
- **Untracked resources are ordered across command buffers.** An untracked buffer written by
  one command buffer and read by the next on the same queue, with no event, is ordered on both
  x86 GPUs and races on the M3 (Q6).
- **A timeout's error is IOAccelerator's.** x86 answers code 2 with "Caused GPU Timeout Error
  (00000002:kIOAccelCommandBufferCallbackErrorTimeout)" and no underlying error. Our bundle
  answers code 2 with the M3's "Caused GPU Hang Error (00000003:kIOGPUCommandBufferCallbackErrorHang)"
  (Q8).
- **GPU times are real.** `GPUStartTime`/`GPUEndTime` are on the host clock on x86; our host
  sends 0 for both (Q7).

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
  end of each encoder that wrote it. With no dirty-page log the copy-in is the whole texture.
  The dirty-page source (525ea4f03, on `steelbrain/rgk-perf`) shrinks it to the changed rows.
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
unified host they are one copy, as on Intel. **Relaxation: none.** AMD is looser than Metal in
one way that costs us nothing: it uploads more than the named range when the GPU copy is clean,
so an app with a short `didModifyRange:` works on AMD, where ours uploads the range alone.

## Q3: Shared storage

Probes: `oracle_storage DEVICE shtex`, `lintex shared|managed`, `spin`.

- **Shared textures** (`newTextureWithDescriptor:` of storage Shared) are made on AMD, Intel and
  the M3. On AMD such a texture behaves like an unsynchronized Managed one: `getBytes:` never
  sees a render clear or blit write (also 200 ms later), and `replaceRegion:` of a corner
  uploads the CPU copy whole over the GPU's write. Intel and the M3 are coherent. Ours: CPU
  access goes through a staging blit the call waits for (`RgfxTexture.mm`), which is coherent.
- **A linear texture over a Shared buffer** is in place on both x86 GPUs: a CPU write to the
  buffer is seen by the next GPU read of the texture, and a render clear of the texture lands
  in the buffer at completion. Over a Managed buffer it follows Managed's rules. That is our
  design (Decided 4).
- **A Shared buffer while the GPU runs:** a 1.3 s kernel polling a word (atomic and plain loads)
  never saw a CPU write made 20 ms after scheduling, on either x86 GPU; a kernel storing its
  progress was never seen by a CPU polling every millisecond before completion. Shared memory is
  coherent at command-buffer boundaries only. (Over a 6 s kernel the 5300M saw the CPU's write
  at 5.8 s, once.) Ours is in place and at least as coherent.

**Relaxation:** only the Shared texture's CPU access (ranked 4), and only on a discrete host.

## Q4: render-pass load and store actions

Probe: `oracle_passes DEVICE actions private|managed|iosurface`, `depth`, `msaa`. A 256 x 256
RGBA8 attachment, cleared to A and stored, then one pass of the row's actions with or without a
draw of B over its left half (Clear rows clear to C).

| load | store | draw | AMD and Intel: left, right | M3: left, right |
|---|---|---|---|---|
| Load | Store | B | B, A | B, A |
| DontCare | Store | B | B, A | B, 25% zeros + A |
| Load | DontCare | B | B, A | A, A |
| DontCare | DontCare | B | B, A | A, A |
| Clear | DontCare | B | B, C | A, A |
| Clear | DontCare | - | C, C | A, A |

The same for Private, Managed and IOSurface attachments. Depth32Float cleared with store
DontCare keeps the clear on AMD and drops it on Intel and the M3. A 4x MSAA attachment resolved
with its samples DontCare keeps its samples on all three.

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

Our stack: memoryless textures are made (m2v), and `isFramebufferReadSupported`,
`supportsTileShaders` and `supportsImageBlocks` are not overridden by the bundle, so the base
class answers them. **Relaxation: none.** With no Apple family claimed, guests should ask for
none of these; a guest that makes a memoryless texture ends on x86 and goes on with ours.

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

On AMD a blit runs beside compute work: a tiny blit command buffer committed behind a 200 ms
compute one ran at once (GPU time 0.38 to 0.39 ms after commit), though its completion was
reported after the compute one's.

Our stack: tracked resources are ordered; untracked ones are left to the program, serial
encoders run in order, and the bundle has its own fence across every 8 encoders. **Relaxation:
none.** Untracked compute across encoders and command buffers is ordered on every x86 GPU, so
guests may rely on it; whether the Vulkan platform orders successive command buffers' untracked
work is worth a check.

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

Our stack (HEAD): `commitAndWaitUntilSubmitted` and `waitUntilScheduled` wait for completion
(bbb153d9e restored it; the host still posts `CB_SCHEDULED`), handlers run on the process's
completion thread in Metal's order, `GPUStartTime == GPUEndTime` is the delivery time (the host
sends 0), and the async creators call their handler on the caller's thread before returning.
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
| `maxBufferLength` | 3208642560 | 2147483648 | 41747087360 | 1099511627776 |
| `maxThreadgroupMemoryLength` | 65536 | 65536 | 32768 | 32768 |
| `maxArgumentBufferSamplerCount` | 2048 | 2048 | 500000 | 0 |
| `readWriteTextureSupport` | 2 | 1 | 2 | 2 |
| `minimumLinearTextureAlignmentForPixelFormat:`, `linearTextureAlignmentBytes` | 256 | 256 | 16 | not overridden |
| `iosurfaceTextureAlignmentBytes` | 16 | 16 | 16 | |
| `isDepth24Stencil8PixelFormatSupported` | 1 | 0 | 0 | 0 |
| `supports32BitMSAA`, `supportsPullModelInterpolation` | 1, 1 | 1, 1 | 1, 1 | 0, 0 |
| `supportsTextureSampleCount:(8)` | 1 | 1 | 0 | 0 |
| barycentrics | 1 | 0 | 1 | 1 |
| `supportsRaytracing`, `supportsFunctionPointers`, `supportsDynamicLibraries` | 1 | 1 | 1 | 0 (lane) |
| `supportsPlacementSparse` | raises (not implemented) | raises | 1 | 0 |
| sparse tile sizes | 0x0x0 | n/r | 64x64 and up | 64x64 and up |
| families | Mac2, Common1-3, Metal3; no Apple | the same | Apple1-9, Metal4 too | as AMD |
| `supportsFeatureSet:(10002)` | 1 | 0 | 0 | 0 |
| `heapTextureSizeAndAlign` alignment | 65536 | 4096 | 128 | |

Formats (`F` lines): both x86 GPUs make every uncompressed color format but B5G6R5, A1BGR5,
ABGR4 and R8/RG8 sRGB (BGR5A1 is made), and every BC format, the 4:2:2 formats and the depth formats (D24S8 and
X24S8 on AMD only). Metal asserts, ending the process, at the create of every ASTC, ETC2, EAC
and PVRTC format, B5G6R5, A1BGR5, ABGR4, R8Unorm_sRGB, RG8Unorm_sRGB and the XR10 formats
(552-555), and at a 4x MSAA BGR5A1 or RGB9E5 texture. `minimumTextureBufferAlignmentForPixelFormat:`
answers on x86 for every format (the format's bytes per pixel, or a compressed block's), where
the M3 asserts for compressed and depth ones. Our device makes ASTC and the packed formats (m2v),
which no x86 Mac does; with no Apple family claimed, frameworks should not ask for them.

**Relaxation: none** for performance. One answer may help placement: both x86 GPUs answer
`minimumLinearTextureAlignmentForPixelFormat:` 256 where our bundle leaves the base class's
answer, and a 256-byte row alignment suits the Vulkan hosts' linear images.

## Not measured

- Cross-process lock waits for pending GPU work: rgk-lock's `lock_probe` (x86 matches the M3,
  except that a read-write lock does not wait for a buffer that only reads the surface).
- Page faults: no probe made a GPU address fault, on a Mac whose Intel GPU drives the display.
- Two-plane (YCbCr) surfaces, and surfaces whose rows miss the host's alignment.
