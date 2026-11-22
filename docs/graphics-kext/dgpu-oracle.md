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
`oracle_mesh`, `oracle_events`, `oracle_texdesc`, `oracle_nilbind`, `oracle_complete`,
`oracle_residency`, `oracle_threads`, `oracle_overlap`, `oracle_readonly`, `oracle_failways`
and `oracle_twinwrites`; the upload costs (Q17) are `guest/bundle/tools/upload_probe.m`'s.

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

Probe: `oracle_storage DEVICE mbuf 4096|67108864`, `mtex` and `capture` (11-12, m3-max-22553 on
the AMD, 22554 on the Intel; not run on the M3); `capture 4096|67108864` with steps 5 to 7
(11-17: m3-max-44786 and 44795 on the AMD, 44789 and 44796 on the Intel; 44788 and 44790, at
64 MiB, ended at their 1 GiB cap in step 7 and agree on steps 0 to 6). A 64 MiB `capture`
peaks at 1.0 GiB, so it runs with `--ram 2`. `mtexcap full|part` (11-18: m3-max-45008 and 45010
on the AMD, 45011 and 45012 on the Intel, macOS 26.7.1 (25G241), API validation off; not run on
the M3).

| | AMD | Intel 630 and M3 |
|---|---|---|
| buffer: CPU write without `didModifyRange:` (next command buffer, 200 ms later) | never seen | seen at once |
| buffer: `didModifyRange:` of [0,256) after an unnamed whole-buffer write | the whole buffer uploaded (4 KiB and 64 MiB) | (one copy) |
| buffer: GPU write pending (no sync), then CPU [0,256) + `didModifyRange:` of it | only [0,256) uploaded; the GPU's bytes stay elsewhere | (one copy) |
| buffer: GPU write without `synchronizeResource:` (completion, 200 ms) | never seen by the CPU | seen at once |
| buffer: `didModifyRange:` after the reading command buffer was encoded, before its commit | seen: the upload happens at commit | seen |
| buffer: CPU bytes written again after `didModifyRange:`, before the commit (`capture` steps 1 and 2) | seen, the command buffer encoded before the write or after it (4 KiB and 64 MiB) | seen |
| buffer: CPU bytes written again after the commit, 20 ms after it is scheduled, while the command buffer's first kernel holds the GPU (1.27 s on AMD, 400 to 640 ms on Intel; step 3) | not seen (4 KiB and 64 MiB): the kernel reads the bytes taken at the submission | seen |
| buffer: CPU bytes written again at once after the commit returns, before the command buffer is scheduled (step 5: on AMD the write was done 0.03 ms after the commit at 4 KiB and 2.1 ms at 64 MiB, and the buffer scheduled at 0.25 and 2.9 ms) | seen (4 KiB and 64 MiB): the bytes are taken at the submission, after the commit returns | seen |
| buffer: as step 3, then a second `didModifyRange:` of the range while the first kernel still holds the GPU (step 6) | not seen (4 KiB and 64 MiB): the call does not reach a command buffer already running | seen |
| buffer: as step 5, then a second `didModifyRange:` of the range at once (step 7) | seen (4 KiB and 64 MiB), as without it | seen |
| texture: GPU write without `synchronizeTexture:`, then `getBytes:` | old bytes, also 200 ms later | new |
| texture: unsynchronized GPU write, then `replaceRegion:` of a 16 x 1 corner | the whole CPU copy is uploaded: the GPU's write is lost outside the corner | corner new, GPU's write kept |
| texture: `replaceRegion:` of the whole texture (B over A) after the commit of a buffer that waits on an unsignalled shared event, then blits the texture, made at once (0.01 ms after the commit) or 1 s later, the CPU signalling after it (`mtexcap full`) | seen, both: the buffer is scheduled only at the signal (Q7), 0.05 to 0.27 ms after it | seen |
| texture: as above, with a first kernel that spins in place of the event wait, the replace 24 ms after the GPU started the buffer (`mtexcap full`, last step) | not seen: the blit reads A | seen |
| texture: the three above with only the rect (64,64)-(192,128) replaced (`mtexcap part`) | B in the rect, A elsewhere, while the wait holds the buffer; A where the replace came 29 ms after the GPU's start | B in the rect, A elsewhere, in all three |

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

When the 5300M takes the bytes (`capture`, 11-17): after the commit returns and by the time the
command buffer starts on the GPU. A write in that window (step 5) is seen whole at 64 MiB, so
nothing copied the bytes before the write was done; a write 20 ms after the buffer is scheduled
(step 3) is not, nor is one followed by `didModifyRange:` (step 6): the call does not reach a
command buffer already running. At 64 MiB the buffer was scheduled 1.7 to 2.9 ms after the
commit and started on the GPU about 11 ms later (at 4 KiB: 0.2 to 0.4 ms, then at most
0.08 ms), consistent with an upload of the 64 MiB (6 GB/s) before its first kernel.

When the 5300M takes a Managed texture's bytes (`mtexcap`, 11-18): at the submission, as a
buffer's, and not at the `replaceRegion:` call. A replace made after the commit, of the whole
texture or a part of it, is seen for as long as the buffer is not yet scheduled, which a wait
on an unsignalled shared event stretches to the signal (1 s here); one made after the GPU started
the buffer is not. The spin step cannot tell the submission from the GPU's start (the buffer was
scheduled 1.98 ms after the commit and started 0.03 ms later). No read was mixed: each replace
was seen whole or not at all.

Ours, by `capture` in an x86-64 macOS 26.7.2 guest on two-copy hosts, 11-12: on the RX 7900 XTX
(m3-max-22650, metal2vulkan 38ab829481) the five steps read A, B, B, A, B, the 5300M's answers;
on the RX 550 (m3-max-22791, metal2vulkan 86d01646ae, which takes the range's bytes at the commit)
the first four read A, B, B, A, and the fifth has no answer: the fourth step's spin held the GPU
for 2 s there, until the driver recovered the context (`radv/amdgpu: ... guilty of a soft
recovery`), and the fifth step's command buffer was discarded as its victim. A CPU write
after the commit, while the command buffer's first kernel runs, is not seen on either, as on the
5300M. Steps 5 to 7 have not run on ours. Step 5, a write after the commit returns and before
the command buffer is scheduled, is where taking the bytes at the commit and at the submission
differ: the 5300M and the Intel 630 both read B there. On the XTX the spin saw the CPU's write
to its Shared flag 140 ms in; the 5300M's saw it once in the 11-17 runs (44786, step 7, about
1.06 s after the write) and in no other held step (Q3: ours is in place).

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

`oracle_waits DEVICE`: waits against handlers that block (each handler sleeps 20 ms), 20 runs
each, tiny blit buffers, on the 5300M, the 630 (both macOS 26.7.1) and the M3 Max (macOS 27.0.1).
Every row was the same in all 20 runs on all three:

| | all three |
|---|---|
| `waitUntilCompleted` against a completed handler that blocks | returns after it ends (0.01 to 0.15 ms after) |
| the same with two completed handlers, the second blocking | returns after the second ends; the first ran before the second |
| the status read in a completed handler, and once the wait returns | `Completed` |
| `waitUntilScheduled` against a scheduled handler that blocks | returns after it ends |
| the buffer's completed handler, its `Completed` status and `waitUntilCompleted` against that scheduled handler | all after it ends |
| a second buffer of the queue, committed right after one whose completed handler blocks, with no handler or a quick one: its `Completed` status, its handler and its `waitUntilCompleted` | all after the first's handler ends |
| a second buffer with no handler, committed while the first's completed handler blocks: its `Scheduled` and `Completed` status, `waitUntilScheduled` and `waitUntilCompleted` | all after the first's handler ends (the polling thread never read `Scheduled` in 5 runs on the 630 and 2 on the 5300M, the buffer passing it between two reads) |
| idle GPU, tiny buffer, commit to `waitUntilCompleted`'s return, no handler / an empty completed handler (median) | AMD 0.50 / 0.50 ms, Intel 0.39 / 0.39 ms, M3 0.72 / 0.89 ms |

So a queue's buffers become `Scheduled` and `Completed`, run their handlers and release their
waiters in one order, on its completion queue: a buffer's status changes before its handlers of
the stage run, its waits return only after them, and a handler that blocks holds everything of its
queue after it, other buffers' statuses and waits included *(class 3)*. Our stack does the same:
each queue's records run in ring order on its completion queue (design.md, "Who takes the
completions"), a record's status changes before its handlers and its waiters go after them, and
a buffer's records run on the thread that took them only while none of its queue's are still to
run.

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

Probe: `oracle_errors DEVICE [purgeable|creates|residency|timeout|bounds]`.

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

The creates Metal bounds by a length (`bounds`, each step in a child process of its own,
`MTL_DEBUG_LAYER` unset). L is the device's `maxBufferLength`, 3208642560 on the AMD and
2147483648 on the Intel, and P is a 4 KiB page. No step ended the process.

| create | AMD | Intel 630 |
|---|---|---|
| `newBufferWithLength:` of L | made | made |
| `newBufferWithLength:` of L + 1 and of L + P | made | nil |
| `newBufferWithBytes:` of L + P | not measured: the job's 2 GiB memory cap ended it in the create | nil |
| `newBufferWithBytesNoCopy:` of 1 MiB, of P + 1 and of L | made | made |
| `newBufferWithBytesNoCopy:` of L + 1 and of L + P | made | nil |
| a Private heap of L | made | made |
| a Private heap of L + 1 and of L + P | made, of 3208904704 bytes | nil |
| in a 1 MiB Private Placement heap, a buffer of 64 KiB at its last 64 KiB | made | made |
| a buffer of 128 KiB there, and of 1 MiB + 64 KiB at 0 | nil | nil |
| a 128 x 128 RGBA8Unorm texture at the last offset of its alignment that holds it | made | made |
| the same texture one alignment on (AMD: 65536, at the heap's end; Intel: 4096, partly past it) | nil | nil |
| the tiled texture SPI, `newTiledTextureWithBytesNoCopy:length:deallocator:descriptor:offset:bytesPerRow:`, over 1 MiB and over L | made | made |
| the tiled texture SPI over L + P | made | nil |

The lengths past `maxBufferLength` diverge *(class 2)*; the placements past a heap's size agree
on the two x86 Macs *(class 3; the M3 not measured)*.

Our stack: the device's `maxBufferLength` is at most 2 GiB, the Intel's, and every create here
answers nil where the Intel does, posting nothing to the host and logging the bundle's note of a
nil: a buffer of a length, of bytes or over the caller's bytes, a heap, the tiled texture (over the
device's no-copy buffer), and a Placement heap's buffer or texture past the heap's size, the
texture's bytes the host's answer to `heapTextureSizeAndAlignWithDescriptor:`. Nil costs the host
least, and is what the app's resource then shows. **Relaxation: none.**

## Q9: limits and formats

Probe: `oracle_caps DEVICE`. Where the x86 GPUs' answers differ from the M3's or from our device's
(an x86-64 guest on the RTX 5080 host, metal_enum):

| answer | AMD | Intel 630 | M3 | ours (5080) |
|---|---|---|---|---|
| `maxBufferLength` | 3208642560 | 2147483648 | 41747087360 | 2147483648 (c795bf3d5; 12821200896 from 4ee4bcaf0, 1099511627776 before) |
| `maxThreadgroupMemoryLength` | 65536 | 65536 | 32768 | 32768 |
| `maxArgumentBufferSamplerCount` | 2048 | 2048 | 500000 | 0 |
| `readWriteTextureSupport` | 2 | 1 | 2 | 2 |
| `minimumLinearTextureAlignmentForPixelFormat:`, `linearTextureAlignmentBytes` | 256 | 256 | 16 | 128, the platform's on the 5080's driver (256 from 4a329e614 to 2026-11-12) |
| `iosurfaceTextureAlignmentBytes` | 16 | 16 | 16 | |
| `isDepth24Stencil8PixelFormatSupported` | 1 | 0 | 0 | 0 |
| `supports32BitMSAA`, `supportsPullModelInterpolation` | 1, 1 | 1, 1 | 1, 1 | the runtime's: 1, 1 on the 5080 (metal2vulkan 7da81ee3d8; 0, 0 before) |
| `supportsTextureSampleCount:(8)` | 1 | 1 | 0 | 0 |
| barycentrics | 1 | 0 | 1 | 1 |
| `supportsRaytracing`, `supportsFunctionPointers`, `supportsDynamicLibraries` | 1 | 1 | 1 | 0 (lane) |
| `gpuCoreCount` (SPI; oracle_caps 11-16, m3-max-37201 and 37202) | does not respond | does not respond | 40 | an x86-64 guest's device does not respond, and a call raises; an arm64 guest's answers the host's count, 0 where the driver has none (Intel's) |
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
are depth on AMD, Invalid on Intel); ours does the same, D24S8 as Intel. The M3 Max answers 175
values and ends at others; an arm64 guest's device answers as it does (Q34). Our device makes ASTC and the packed formats (m2v),
which no x86 Mac does; with no Apple family claimed, frameworks should not ask for them.

**Relaxation: none** for performance. One answer helps the host, and the lane gives it, as a
choice inside the envelope *(class 2)*: `maxBufferLength` at most three quarters of
`recommendedMaxWorkingSetSize`, the rule the 5300M and the M3 answer by (4ee4bcaf0), and at most
2147483648, the Intel's own answer, which the Vulkan platform's shaders address whole in their
direct form (c795bf3d5): today's runtime lands an access 4 GiB or more past a buffer's bound
address 4 GiB lower in it.
`minimumLinearTextureAlignmentForPixelFormat:` and the SPI linear alignments describe the GPU,
each Mac answering its own, so the device answers the host GPU's: the platform's, 256 on RADV
and 128 on NVIDIA's driver (`device_info.rs`). Both x86 GPUs answer one value for every format;
a host GPU that asks more of larger texels answers each format its own, and the SPI ones the
largest (a Radeon RX 550: 256 bytes for texels of 1 to 4 bytes, 512 for 8, 1024 for 16). The
lane answered 256 for every format from 4ee4bcaf0 and 4a329e614, as both x86 GPUs answer, while
the platform answered Metal's 16 on every host; on NVIDIA's driver, which takes a linear image
in place only at its own pitch, rows at 256 bytes were shadows wherever that pitch is not one.

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
| public: a texture buffer, its texels past the buffer's 4096 bytes (offset 4064) | made | the same | made (metal2vulkan's `probes/texture_buffer_past_length.m`: 64 RGBA8Unorm texels from 3968, 3976 and 4096) |
| public: a texture buffer of 4 RGBA32Float texels, rows 64, past a buffer of 16 KiB or more: from 16352 and 16576 of 16384 bytes (ending at 16416 and 16640), from 32736 of 20000 bytes (ending at 32800) | made | the same | nil; made ending at the length (from 16320 of 16384), from 19968 of 20000 and from 12288 of 4096 |
| public: a 2D texture 64 RGBA8Unorm texels wide, rows 256, from 0, past a buffer of 16 KiB or more: 64 x 65 over 16384 bytes (ending at 16640), 64 x 129 over 20000 (ending at 33024); a buffer of 20000 bytes answers `allocatedSize` 20480 | made | the same | nil, a buffer of 20000 bytes answering `allocatedSize` 32768; made 64 x 64 over 16384, 64 x 80 over 20000 and 64 x 17 over 4096 |
| public: a texture buffer, rows fewer than its texels' 64 bytes | nil at 1; asserts at 0 "Linear texture: bytesPerRow (0) must be greater or equal to (64) bytes" | asserts at 1 and at 0, at 0 with the AMD's words | not measured |

*(class 3 for the type lines the three share; class 2 for the rest.)* Our stack: the public form
and the linear SPI end the process at each type above they do not make, with the words the three
share or, at a 3D or multisample type, the M3's, sending nothing, since the host's platform ends
the host's process at a texture over a buffer it does not make. A texture buffer by the public
form is made, at any offset, which the host's platform makes by that form; rows of 0 bytes end the
process with the words the two say, and fewer than its texels' bytes answer nil, the AMD's answer
*(class 2)*. One whose texels lie past the buffer's bytes is made, as the three make it *(class
3)*, and the host's platform makes it, its texels past the length in a tail of the texture's own.
One past the M3's allocation of a buffer of 16 KiB or more, its length rounded up to 16 KiB,
answers nil, as the M3's and as the host's platform answers it on every device, where the two x86
GPUs make it *(class 2)*; so does a 2D texture by the public form or the linear SPI and a 2D array
by the linear SPI, whose slices metal2vulkan measured the M3's nil for (`rgfxPastAllocation:`)
(the texture buffer rows and ends: a scratch probe, `tbprobe`, not kept, macOS 26.7.1; past the
allocation `tbpa`, a scratch probe, jobs m3-max-30900 to 30902, the M3 under macOS 27.0.1). The
tiled SPI answers nil for every type but 2D,
logged once, as it does for more than one level: the host does not carry the others. The linear
SPI makes a 2D array as the M3 does *(class 2)*, its slices where the M3 lays them
(`rgfxBufferSlicePitch`), and a view of more than one of its slices answers nil, which Metal
asserts at on the M3 Max and an M4 (`sliceRange.length == 1`, measured by metal2vulkan).


**Texels past the buffer's length.** Probe: `oracle_pastlen DEVICE paths` and `oracle_pastlen
DEVICE limits`, 11-15: the M3 Max under macOS 27.0.1 as arm64 (m3-max-32055, and its bounds
m3-max-32081, 32093 and 32098) and as x86_64 under Rosetta (m3-max-32056), the AMD 5300M
(m3-max-32058) and the Intel 630 (m3-max-32057) under macOS 26.7.1. Each `paths` texture is
R32Uint over a buffer whose words name their offsets, made beside 128 neighbour buffers of 4096
bytes whose words name theirs, so a texel read names the byte it came from. Each path runs in a
process of its own: a kernel's `read()`, a blit to a buffer and `getBytes:`, and the writes a
kernel's `write()`, a blit from a buffer and `replaceRegion:`, each write read back by the three
reads.

| texels past the length | AMD 5300M | Intel 630 | M3 Max |
|---|---|---|---|
| inside the buffer's allocation (the x86 GPUs' `allocatedSize`, the length rounded up to 4 KiB; the M3's `resourceSize`, the bounds below) | one memory: zero at first, and each read sees each write, a Managed buffer's two copies meeting at `synchronizeResource:` | the same | the same; past a 4096-byte buffer, whose `allocatedSize` is 4096, the next buffer's bytes in its pool, 4096 bytes on, which every path reads and writes alike; under Rosetta, where a buffer of 4096 bytes or more lies alone, zero |
| past the allocation | the CPU's paths reach the next allocation of the process's memory (`contents` + 4096 in a 4096-byte buffer); the GPU's reads see zero, its own writes or a word an earlier process wrote, never the CPU's; a blit of a 2D texture there fails (`kIOAccelCommandBufferCallbackErrorTimeout`), and the queue's later command buffers are ignored, while a texture buffer's blits read and write as its kernels do | the CPU's as the AMD's; the GPU's reach a page every process shares, holding words earlier processes wrote, so a kernel reads a blit's or a kernel's write back, but not `replaceRegion:`'s, and `getBytes:` reads the next allocation, not the GPU's writes | no such texture is made (the bounds below) |
| a no-copy buffer whose bytes end at the end of the process's mapping (the next page `PROT_NONE`) | the CPU's paths die on SIGBUS; the GPU's as past the allocation | the same | 4096 bytes at a 16 KiB page's end: the CPU's paths die on SIGBUS, and the GPU's read zero and lose their writes; under Rosetta, 4096 bytes at a 4 KiB page's end, the GPU's keep their own writes |

Inside the allocation every GPU keeps one memory *(class 3)*. Past it the x86 Macs keep two that
disagree, the CPU's another allocation's or a fault and the GPU's a page of their own, and the
M3 makes no texture *(class 2)*. Past the process's mapping, inside a no-copy buffer's
allocation, every GPU keeps two, the CPU's a fault *(class 3)*.

| which such textures Metal makes (`limits`) | AMD 5300M | Intel 630 | M3 Max, arm64 | M3 Max, Rosetta |
|---|---|---|---|---|
| a texture buffer, a 2D texture by the public form and by the linear SPI, over a buffer of 16 bytes to 256 KiB by `newBufferWithLength:` (Shared, Managed, Private) or `newBufferWithBytes:`, at each position in its pool | made at every end bisected: 16 MiB from the buffer's start for a texture buffer, 4 MiB for a 2D texture | the same | nil past the buffer's `resourceSize` (SPI) from its start: 128 KiB for a buffer under 16 KiB, the length rounded up to 16 KiB for the rest | the same, but 128 KiB for a buffer under 4096 bytes alone, 16 KiB for one of 4096 to 16384 |
| the same over a Shared heap's buffer, a heap of 4 MiB | no Shared heap ("Shared storage mode disallowed") | the same | nil past the heap's size from the buffer's start, wherever the buffer lies in it | the same |
| the same over a no-copy buffer, at a page's start or 4096 bytes short of a page's end | made at every end bisected | the same | nil past `resourceSize` from the buffer's start, the size of the 16 KiB pages its bytes meet: 16 KiB for 4096 bytes at a page's end, so a texture over them is made 12 KiB into the next page, and 32 KiB for 4097 bytes there | the same |
| what the end is | | | a 2D texture's and a texture buffer's last texel, whatever the rows' bytes past it (rows 4096 bytes past a texture buffer's texels; a 2D texture's rows 512 bytes apart from 256); a linear 2D array's last slice's end, each slice its rows' bytes rounded up to 128 apart, its last texel inside the bound or not (2 slices of 64 x 127 R32Uint rows 512 apart from 256 made in a 4096-byte buffer, of 64 x 128 nil) | the same |

The AMD also answers nil for a 2D R32Uint texture 64 wide, rows 512 bytes apart, of 2 rows or more,
over every buffer, inside its length too, which the Intel makes; that is not about the length,
and the device makes it, as the Intel does *(class 2)*.

Our stack: the host's platform keeps a texture's bytes past its buffer's length in a tail of the
texture's own, zero when made, which every GPU path reads and writes (metal2vulkan's
`linear_plan.rs`, "Length"), while the guest's memory past the buffer is another allocation's or
none. The bundle's `getBytes:` and `replaceRegion:` copy the bytes inside the length in place,
and those past it by a blit on the device's copy queue that the call waits for
(`rgfxCopyRegion`), so every path sees one memory past the length: the M3's answer inside its
allocation, and inside the envelope past the x86 Macs' allocations, where they differ *(class
2)*. It costs one round trip to the host for each call whose region has a byte past the length,
and nothing else. Before it the CPU's copies read and wrote the guest's memory past the buffer,
another allocation's bytes, which no GPU path saw. A no-copy buffer's bytes past its length are
the app's own memory, which Metal's CPU paths reach on all three, dying where the app's mapping
ends *(class 3)*: the bundle copies them in place, as Metal does, and the GPU's paths there use
the tail, as each GPU's read memory of its own past the app's mapping (the table's last row). Past
the length inside the app's last page is not measured. Under an Apple GPU's rules (`rgfxAppleGPU`:
an arm64 guest's device) the bundle answers nil past the M3's bound from arm64, the table's
`resourceSize` (`rgfxAppleAllocation`), sending nothing; the x86 guest's device makes them, as
the x86 Macs do, short of the host's platform's own bound above *(class 3 for each guest)*. The
shell tests check both: `test_buffer_texture_copies_past_the_length_reach_the_tail` and
`test_apple_textures_past_a_buffers_allocation_answer_nil`. Over a buffer of 16 KiB or more the
host's platform counts as the M3 does (metal2vulkan's `linear_plan.rs`, "Length"): a 2D
texture's and a texture buffer's last texel, a 2D array's slices' end, against the length
rounded up to 16 KiB, or a heap's buffer's heap's size. A buffer over pages, a no-copy buffer's
or one over the device's region, it counts from the host pointer's offset in its 16 KiB page,
which the guest does not know; the bundle counts the length rounded up to 16 KiB for those on
either guest (`rgfxHostAllocation`), never past the host's bound, and so answers nil for a
no-copy buffer's texture the M3 makes where its bytes start into a page and cross one more page
than its length alone *(class 2)*.

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
an even width (an odd height too, as a render target as well), every usage the SDK names (usage 0
answers 0x17; Q36 has which bits each answers and ends the process at), every storage but
Memoryless, and every CPU cache and hazard tracking mode and swizzle; and a TextureBuffer of one
row. The x86 Macs make BC1 and BC7 textures wherever their rows fit, where the
M3 answers nil, over a surface with planes or without, in Managed or Private storage. The M3
answers nil for a render target of the 4:2:0 Y'CbCr format 500 of an odd width or height.

The two x86 GPUs differ only in the words at `Depth24Unorm_Stencil8` in Managed storage: the
AMD's are the depth formats' storage rule, the Intel's "invalid pixelFormat (255)", as the Intel
lacks the format.

The M3's answer at each case, measured again on 11-13 with the cases added since, is in Q34, with
how an arm64 guest's device answers them.

What our stack does:
- *(class 3)* The device's creator ends the process at every rule the x86 Macs end it at, with
  their words, before it sends the host anything (`rgfxEndAtInvalidSurfaceTexture`,
  RgfxTexture.h). An arm64 guest's device ends it and answers nil where the M3 does (Q34). The bundle had sent each of these to the host, posted, and answered a texture;
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

**Post-tessellation vertex functions** (11-12, `pipeline_reflection_check`'s r_patch,
r_patch_points, r_patch_buffers, r_patch_plain and r_patch_unused, of a descriptor of the control
points' attributes in buffer 0, stepping per control point, and the patch's attribute in buffer 1,
stepping per patch; m3-max-22312 on the AMD, 22313 on the Intel, 22314 on the M3): AMD, Intel and
the M3 agree *(class 3)* that the control points (`patch_control_point<T>`) and the patch's
attributes are no binding, that the descriptor's buffers are unused (`isUsed` and `isActive` 0)
though the function reads them, and that `position_in_patch` and `patch_id` are built-ins 14 and
13 (unused where the function never reads them). `postVertexDumpStride` adds the patch's position
to the outputs and the GPU's 12 (Intel 8): 12 bytes for a triangle patch and 8 for a quad, whether
or not the function takes its position. The AMD compiler dies making r_patch_buffers (a quad
patch of no control points, reading its points from a buffer), which the Intel GPU and the M3 make.

**Ours** (`RgfxPipelineReflection.mm`): the bundle builds the same objects, of Metal's own classes,
from each function's AIR nodes, and answers them in every creator that takes options, with AMD's
stride. It answers nil, as before, for a pipeline of a function whose binding index function
constants compute (`[[buffer(kIndex)]]`), whose module the reader does not follow, or with a type,
built-in or binding no x86 Mac's answer was measured for. An argument read only in code its
constants remove is answered used *(class 4: Metal reflects the specialised code)*.
`pipeline_reflection_check` (11-05, 80 cases of 433490 values) finds no difference on the AMD
GPU, and on the Intel GPU only `postVertexDumpStride`, 4 less than ours on each vertex pipeline;
on both, the one case ours does not make is `c_constants` (`k_constants`), whose binding index
function constants compute. Since 11-12 the reader follows a post-tessellation vertex function's
control points (`air.patch_control_point_input`), and the patch cases answer as above: no
difference on the AMD GPU, the Intel GPU's stride 4 less, and on the M3 only its constant buffer
alignment and `_flags`, as everywhere.

**Deep argument buffers, colour inputs, imageblocks** (11-14, `pipeline_reflection_check`'s
c_deep_ab, an argument buffer of structs nested 6 deep; r_fetch_group and r_fetch_at, colour
attachment inputs in a raster order group and at an index a function constant gives; and
c_imageblock_explicit, c_imageblock_late, c_imageblock_on, c_imageblock_off, r_imageblock_data,
r_imageblock_master and r_rog_texture: m3-max-31129 and 31135 on the M3, 31132 and 31210 on the
AMD, 31133 and 31211 on the Intel). The deep argument buffer and the colour inputs answer as the
rest on all three (the colour inputs listed nowhere). A kernel's explicit imageblock
(`imageblock<T, imageblock_layout_explicit>`) is a binding in its place on the M3, an
`MTLImageBlockArgument` of binding type 17 and `imageBlockKind` 1, read-only, at index 0 wherever
the argument is, of its data's size and its struct's type, of no master struct, aliasing no
implicit imageblock (`aliasImplicitImageBlockRenderTarget` 0xFFFFFFFF); a false function constant
leaves it out. Neither x86 GPU makes a pipeline of one (AMD: its compiler's connection drops at
each try; Intel: "AIR builtin function was called but no definition was found"), and the M3 makes
none of an implicit one in a kernel (c_imageblock: "Encountered unlowered function call to
air.load.implicit_imageblock.v4f32"). A fragment function's imageblock data (`[[imageblock_data]]`)
is a fragment binding on the M3 and the Intel GPU, an `MTLImageBlockDataArgument` of binding type
16, of the same fields; one that views a master struct (`[[imageblock_data(Master)]]`, its members
named by their `user(...)` names, as the AIR names them) holds the master as its
`imageBlockMasterStructMembers`, and the AMD GPU and the Intel GPU make it, where the AMD GPU makes
no pipeline of plain imageblock data. A texture in a raster order group
(`[[texture(n), raster_order_group(g)]]`) answers as one in none on the AMD and Intel GPUs. Ours
answers each so *(class 3)*: AMD no difference, Intel only its stride.

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
answer passes 32 bits *(class 2)*. `imageblockSampleLength` is the platform's answer on an Apple
GPU's device, the arm64 guest's, and 0 on the x86 guest's, as the AMD and the Intel 630 answer for
a vertex and fragment pipeline too.

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
   268435456 (2^28) elements for a TextureBuffer's width (below), one row for the 1D types and a
   TextureBuffer, one slice for every type but 3D, one element for every type but the arrays, and
   2048 for an array.
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

**Texture buffers** (`oracle_texdesc DEVICE CREATOR --texbuf` and `--tbread`, 11-14, the AMD and
the Intel under macOS 26.7.1 and the M3 Max under 27.0.1, each step in a child process of its
own; jobs m3-max-31563, 31564 and 31565). The descriptor was
`textureBufferDescriptorWithPixelFormat:width:resourceOptions:usage:`, as MoltenVK makes a texel
buffer view's texture, of R8Sint, RG8Unorm, RGBA8Unorm, RG32Float and RGBA32Float (1 to 16 bytes),
usage ShaderRead. It was Private at the device's creator and a Private heap's (the heap of the
texture's bytes, at most `maxBufferLength`), and Shared over a Shared buffer by the public form, at
offset 0. The rows there were the texture's bytes rounded up to 256, over a buffer of those bytes
(at most `maxBufferLength`) and over one of 4096 bytes. The widths ran from 16384 to 2^33, and
each format's largest width made and largest width that does not end the process were bisected
from 1 to 2^36:

| | AMD | Intel 630 | M3 Max |
|---|---|---|---|
| `maxTextureBufferWidth` (SPI) | 268435456 | 268435456 | 268435456 |
| `minimumTextureBufferAlignmentForPixelFormat:` | the format's bytes | the format's bytes | 16 |
| up to 2^28 elements, every format and creator | made | made, but nil past 2 GiB at the device's and the heap's (16 bytes past 2^27) | made |
| 2^28 + 1 elements and more, to 2^33, every format and creator | ends the process at the validation | the same | the same |
| over a 4096-byte buffer, its texels past the buffer's end | made, to 2^28 R8Sint texels | made | nil past 128 KiB (46626 R8Sint made, 46626 RGBA8Unorm nil) |
| over a buffer, 16-byte texels, rows of 2^32 bytes (2^28 wide) | made | ends the process (SIGFPE), over either buffer | made |
| a kernel's reads of elements 0, 1, W / 2, W - 2 and W - 1, R8Sint and RGBA32Sint, 46626 and 2^24 wide, every creator | right | right | right |

So the limit counts elements, not bytes, and it is the device's `maxTextureBufferWidth`. One
element past it, every creator ends the process with the width line every type gives:
"MTLTextureDescriptor has width (268435457) greater than the maximum allowed size of
268435456." *(class 3)*. Below it, the bytes decide as they do for any texture: the Intel's
device's and heap's creators answer nil past its `maxBufferLength`. A texture buffer over a buffer
is made with its texels past the buffer's end on the x86 Macs, and is nil past its allocation on
the M3 *(class 2)*. The Intel's death at rows of 2^32 bytes is a third answer there *(class 2)*.

In a guest (macOS 26.7.2 on our device over the Iris Xe of Raptor Lake-P, which answers
`maxTextureBufferWidth` 268435456, `maxBufferLength` 4294967296 and
`minimumTextureBufferAlignmentForPixelFormat:` 16 for each of the five formats), the same cases of
R8Sint and RGBA8Unorm at each creator, with heaps and buffers of at most 256 MiB and 64 KiB
(`ORACLE_TB_CAP`): development's set before 11-14 (52caf9a57) ended the process one element past
16384 at every creator, 46626 included, in the 2D words ("MTLTextureDescriptor has width (16385)
greater than the maximum allowed size of 16384.", job m3-max-31642). The fixed set (5e75bfa39) makes
them up to 268435456 elements at every creator, the largest width R8Sint makes and the largest it
lives at, bisected, and ends the process at every width past it, to 2^35, in the measured words, as
the x86 Macs do. A kernel reads elements 0, 1, W / 2, W - 2 and W - 1 right, R8Sint 46626 and 2^24
wide and RGBA32Sint 46626 wide, at every creator (job m3-max-31664, no fatal line). RGBA8Unorm 2^27
and 2^28 wide answer nil at the heap's and the buffer's creators, past the cap: the heap has no
room, and the bundle answers nil past a buffer's allocation.

What our stack does:
- *(class 2)* The device's creator answers nil for a texture of more bytes than the device's
  `maxBufferLength`, every level of every slice and sample, sending nothing
  (`rgfxTexturePastLength`, RgfxTexture.h), as the Intel does and as the bundle answers a buffer
  and a heap past it. Until 11-07 the bundle sent it, posted, and answered a texture whose
  create the host's platform could refuse for its size: a texture the app held, whose command
  buffers' draws the host would then drop (`refused.rs`).
- *(class 3)* A texture buffer takes up to `kRgfxMaxTextureBufferWidth` (2^28) elements under the
  x86 Macs' rules and an Apple GPU's alike, ending the process past it in the measured words. Until
  11-14 the bundle gave it the 2D limit, 16384, and a guest's MoltenVK died at a texel buffer view
  46626 elements wide. The host's Vulkan devices take a texel buffer view of at most
  `maxTexelBufferElements`: 2^29 on RADV (the RX 7900 XTX and both 780Ms, Mesa 26.2.6), 2^28 on
  NVIDIA's 620.70.06 (the RTX 3090 and 2080) and on the RTX 5080 (metal2vulkan's profile), and 2^27
  on ANV (the Iris Xe of Raptor Lake-P, Mesa 26.2.6), below Metal's 2^28. On ANV, metal2vulkan
  refuses a binding of one past 2^27 elements (`TexelBufferTooLarge`), which is metal2vulkan's to
  carry. Over a buffer, its rules past the buffer's end stay as they were (`rgfxPastAllocation:`,
  RgfxBuffer.mm), and at rows of 2^32 bytes it takes the AMD's answer, sending the create.
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

## Q26: when a resource is complete, and what orders a read-back across queues

Probe: `oracle_complete DEVICE answers 300` and `... answers 300 sched`, and `oracle_complete
DEVICE xqueue 200`, 11-11 (m3-max-17599, 17600 and 17601), the M3 Max on macOS 27 among them.
Each command buffer spins about 300 ms (200 ms for `xqueue`) before it uses the resource.

| asked | AMD | Intel 630 | M3 |
|---|---|---|---|
| right after the commit (status Committed), any use | `isComplete` and `isWriteComplete` YES, `waitUntilComplete` returns at once | the same | the same |
| once `waitUntilScheduled` returns, a render pass, a kernel's write or a blit into the texture, a kernel's write of a buffer | YES, at once | NO and NO; the wait returns as the buffer's GPU work ends (298 to 428 ms), its status then still Scheduled in 7 of 12 cases | NO and NO; the wait returns at 293 to 299 ms, its status then still Scheduled in 8 of 12 cases |
| the same, a kernel that only reads the texture or the buffer | YES, at once | NO and NO; waits | `isComplete` NO, `isWriteComplete` YES; waits |
| a buffer not committed | YES, at once | the same | the same |
| a resource only a completed buffer named, while a buffer of a second queue runs | YES, at once | the same | the same |

Managed and Private storage answer alike. Once the buffer completed, every answer is YES.

`xqueue`: whether queue B's buffer, committed once queue A's is scheduled, completes before A's
(5 rounds each):

| B's buffer | AMD | Intel 630 | M3 |
|---|---|---|---|
| empty | before A | before A | before A |
| a blit encoder with no command | before A | after A | before A |
| `synchronizeTexture:` of the texture A clears, or of another | before A | after A | before A |
| a compute dispatch of one thread | after A | after A | before A |
| that synchronize after the texture's `waitUntilComplete` | before A (the wait does not wait) | after A | after A |

So `isComplete` is *(class 2)*: the AMD answers YES while the 630 and the M3 answer NO, and
the M3 tells reads from writes for `isWriteComplete`. Before the buffer is scheduled, and for
one not committed, all three answer YES at once *(class 3)*. On the M3 only the wait orders a
second queue's synchronize behind the first queue's buffer; on the AMD nothing does, and the
630 runs the two queues' buffers in order, an empty one aside.

What our stack does *(class 2, the 630's answer)*: a resource answers NO to both, for any use,
and `waitUntilComplete` waits, from the buffer's submission (`commitAndWaitUntilSubmitted`'s
return, or its scheduling) until the bundle takes its completion and has written back what it
wrote of copies (`RGFX_RESOURCE_SPI`, `RgfxKept`); a view of a texture, and a texture over a
buffer, count with the memory they lie over. Until 11-11 it answered YES at once, the M3's
answer right after the commit: Apple's GL renderer on macOS 11.7.11 reads a drawable back with
the texture's `waitUntilComplete`, then a `synchronizeTexture:` on a queue of its own and
`getBytes:`, and on a unified-memory host, where the synchronize is a no-op that finished
before the draw, the read saw the texture as it was before the draw (Safari's WebGL reads,
about 2.5% of them on a 780M). The M3's `isWriteComplete` YES for a read-only use is not kept:
arm64 guests see NO, which only waits for readers too.

## Q27: the rows IOSurface works out where none are named

Probe: `oracle_rows DEVICE`, 11-12 (m3-max-21363, 21364 and 21365), the M3 Max on macOS 27.0.1
among them. Surfaces made by `IOSurfaceCreate` with a width, a height, bytes an element and an
OSType alone (`kIOSurfacePlaneInfo` with each plane's width, height and bytes an element for
`420v`), and CVPixelBuffers made with an empty `kCVPixelBufferIOSurfacePropertiesKey`. Rows are
`IOSurfaceGetBytesPerRow` (of each plane for `420v`), offsets each plane's base from the
surface's.

| asked | AMD | Intel 630 | M3 |
|---|---|---|---|
| `IOSurfaceGetPropertyAlignment` of `kIOSurfaceBytesPerRow` and `kIOSurfacePlaneBytesPerRow` | 64 | 64 | 128 |
| the same of `kIOSurfaceOffset` and `kIOSurfacePlaneOffset`; of `kIOSurfaceAllocSize` and `kIOSurfacePlaneSize` | 64; 1 | 64; 1 | 16384; 1 |
| `IOSurfaceAlignProperty(kIOSurfaceBytesPerRow, N)` for N of 1, 8, 64, 144, 192, 384, 1336, 3840, 4004 | 64, 64, 64, 192, 192, 384, 1344, 3840, 4032 | the same | 128, 128, 128, 256, 256, 384, 1408, 3840, 4096 |
| `'RGhA'` (8 bytes) 1 x 1, 18 x 28, 48 x 48, 85 x 162, 167 x 161, 170 x 134: rows | 64, 192, 384, 704, 1344, 1408 | the same | 128, 256, 384, 768, 1408, 1408 |
| `'BGRA'` 96, 463, 1001 and 3662 wide: rows | 384, 1856, 4032, 14656 | the same | 384, 1920, 4096, 14720 |
| `'L008'` 1920 x 1080 and 1001 x 1: rows; size | 1920, 1024; 2073600, 1024 | the same | 1920, 1024; 2080768, 16384 |
| `420v` 1920 x 1080: each plane's rows; plane 1's offset | 1920, 1920; 2073600 | the same | 1920, 1920; 2080768 |
| `420v` 18 x 28 | 64, 64; 1792 | the same | 128, 128; 16384 |
| CVPixelBuffer `420v` 1920 x 1080: rows; the planes' offsets | 1920; 32, 2073632 | the same | 1920; 512, 2074112 |
| CVPixelBuffer `420v` 18 x 28 | 32; 32, 928 | the same | 64; 512, 2560 |
| CVPixelBuffer `32BGRA` 96 x 96, `64RGBAHalf` 18 x 28: rows | 384, 160 | the same | 384, 256 |
| `minimumLinearTextureAlignmentForPixelFormat:` of RGBA16Float, BGRA8Unorm and R8Unorm; `linearTextureAlignmentBytes`, `deviceLinearTextureAlignmentBytes` | 256 | 256 | 16 |
| `iosurfaceTextureAlignmentBytes`, `iosurfaceReadOnlyTextureAlignmentBytes` | 16 | 16 | 16 |

The rows of a surface with none named are IOSurface's own, byte for byte the same in the
AMD's process and the 630's: 64-byte rows and offsets, on a MacBook Pro whose AMD and Intel
accelerators are both loaded and whose GPUs both align a linear texture's rows to 256. Neither
accelerator raises IOSurface's alignment to its GPU's linear one *(class 2: 64 on x86, 128 and
16 KiB offsets on the M3)*. CoreVideo names its rows and plane offsets itself (on x86 rows at a
multiple of 32 bytes and the first plane 32 bytes in, on the M3 the first plane 512 bytes in),
and IOSurface keeps them. Our kext raises IOSurface's alignment to the host GPU's linear one,
256 on RADV (`RgfxSurfaceRows.hpp`); the rows a client names, CoreAnimation's and CoreVideo's
among them, stay as it names them, as on these Macs.

## Q28: rasterization rate maps

Probe: `oracle_ratemap DEVICE`, 11-12 (m3-max-20796), the AMD alone: a map of a 256 x 128 screen
of one layer (horizontal samples 1, 0.25, 1, vertical 1, 0.5) and of two (a second layer of
0.5, 1 and 0.25, 1, 0.75), their answers, their parameter data, a full-target triangle drawn
through each, and the calls that may end the process, each in a child process.

| asked | AMD |
|---|---|
| `supportsRasterizationRateMapWithLayerCount:` 0, 1, 2, 3 | YES, YES, YES, NO |
| a map's class | `GFX10_MtlRasterizationRateMap`, below `AMD_MtlRasterizationRateMap` and Metal's `_MTLRasterizationRateMap` |
| `physicalGranularity`, `parameterBufferSizeAndAlign` | 32 x 32 x 1; 16,472 bytes aligned to 256, of which the copy changes 144, at offset 0 or 4 |
| `physicalSizeForLayer:` of each layer, and both coordinate maps | the screen's 256 x 128 x 1, and every coordinate as it was: the map is the identity, whatever its samples |
| `physicalSizeForLayer:` past the layers | 0 x 0 x 1 at layer 1 of a one-layer map, 256 x 128 x 1 at layer 2 |
| `mapScreenToPhysicalCoordinates:forLayer:` past the layers | (-1, -1) |
| `mapPhysicalToScreenCoordinates:forLayer:` past the layers | ends the process (SIGSEGV) |
| the pass, with and without a viewport of the screen's size | 32,768 fragments, each at its pixel, decoded to the same pixel |
| a fresh descriptor's `mutability` and `minFactor` (SPI); a map's | 2 (Immutable) and 0; the descriptor's |
| a `Mutable` map not reset (`setMutability:` on its descriptor) | compiled at its creation: 256 x 128, and its coordinate maps answer |
| a map of no layers | its creator ends the process (SIGSEGV) |
| `copyParameterDataToBuffer:offset:` into a Private buffer | ends the process (SIGSEGV) |
| `formattedDescription:` (SPI) | Metal's base map's: the description, then the label, the screen's width and height, the layer count, the granularity's width and height and the parameter data's size and alignment, a line each |

So the AMD's maps do not vary the rate: a pass over one draws every pixel. The M3 Max and the M4
compile Apple's map (metal2vulkan's runtime, `rate_map.rs`, measured there): a physical size
below the screen's where the samples are below 1, 24,624 bytes of parameter data aligned to 4,
the coordinate maps past the layers reading the last layer, and supporting 1 or 2 layers, not
0. Both answer what a correct app reads back, so the two families' answers bound the contract
*(class 1)*.

What our stack does: the host's map, the platform's compile (`RATE_MAP_CREATE`, design.md); the
layer counts the platform supports, 1 and 2 (the M3's answer, as before, for 0), and on rt,
which makes no rate map, none: the device then answers no for every count and offers no
`RATE_MAPS`; a map of no layers made, as the M3 makes it. The bundle ends the process where
every Mac does, at a copy into a Private buffer, and where the M3's map does and the AMD's
answers nothing the host can give: a coordinate map of a map of no layers or of an empty
screen, and the physical-to-screen map of a `Mutable` map not yet reset.

## Q29: function pointers

Probe: `oracle_fnptr DEVICE`, 11-12 (m3-max-21084 the AMD, m3-max-21086 the Intel, with
`ORACLE_FNPTR_NO_RENDER`): three compute pipelines linking `fp_add1`, `fp_mul2` and `fp_neg` of
one library, their handles, visible function tables of 0 to 4096 entries, dispatches calling
through a table bound at a buffer index, a range, and an argument buffer, and a render
pipeline's handles and tables.

| asked | AMD | Intel |
|---|---|---|
| `supportsFunctionPointers`, `supportsFunctionPointersFromRender` | YES, NO | YES, NO |
| a handle of a linked function | its name and type 5; `gpuResourceID` raises an unrecognized selector | the same |
| a handle of an unlinked function, of the kernel | nil, nil | nil, nil |
| `functionHandleWithName:` | raises an unrecognized selector | the same |
| a table's `gpuResourceID` | 1 for each, 3 for another pipeline's | 1 to 9, one each |
| a table's `allocatedSize` for 0, 1, 3, 2016, 2017 and 4096 entries | 4096, 4096, 4096, 16384, 16384, 36864 | 4096, 4096, 4096, 16384, 20480, 36864 |
| a table's `resourceOptions`, storage mode, `isAliasable` | 528, Managed, NO | 512, Shared, YES |
| `setPurgeableState:` Volatile, then asked | 2, then 3 | 2, then 2 |
| dispatches through a table set before and after encoding, bound alone or in a range | each entry's function: 11 22 13 26 15 16, 20 12 13 26 | the same |
| an argument encoder of a table, its length, alignment and bytes | 8, 8, the table's id | 8, 32, the table's id |
| a dispatch reading a table through an argument buffer without `useResource:` | the command buffer times out (status 5) | not run |
| the same with `useResource:` | 20 12 24 14 28 30 | the same |
| a render pipeline's handle | not measured (the AMD's compiler service failed the pipeline) | ends the process: Metal's base asserts "Driver needs to override this" |

What our stack does (m3-max-21123, a macOS 26.7.2 guest on an Iris Xe, and m3-max-21327 on a
Radeon 780M): every compute answer above as the two Macs give it, but the tables'
`allocatedSize` and `resourceOptions`, which are the host's table's (256 bytes and 8 an entry,
past 16 KiB rounded up to a multiple of it: 256, 264, 280, 16384, 32768 and 49152 for the counts
above; and 512, Shared), and their `isAliasable`, NO. A render pipeline's handles and tables are
nil, with Metal's unsupported error, as `supportsFunctionPointersFromRender` answers NO. A table
read through an argument buffer without `useResource:` is read, where the AMD's command buffer
times out: inside the envelope, as no Mac's correct use depends on it. A table answers
`setPurgeableState:` NonVolatile whatever it is told, as the Intel's and the M3 Max's do (Q35).

## Q30: dynamic libraries

Probe: `oracle_fnptr DEVICE`, 11-12 (m3-max-21413 the AMD, m3-max-21414 the Intel, with
`ORACLE_FNPTR_NO_RENDER`): a dynamic library (`metal -dynamiclib`, install name
`@executable_path/liboracle_fnptr_dyn.metallib`, exporting `fpd_triple`) made of its library and
read from its file; a kernel linked against it (`-l`) whose library is read from its file, or made
of the same bytes with `newLibraryWithData:`, its pipeline preloaded with the dynamic library or
bare; a kernel linked against a library whose install name names no file; a compile of source with
the dynamic library in `MTLCompileOptions.libraries`, through a proxy that keeps the messages sent
to it; and the refusals.

| asked | AMD | Intel |
|---|---|---|
| `supportsDynamicLibraries`, `supportsRenderDynamicLibraries` | YES, NO | YES, NO |
| a dynamic library of the library, of its file | `_MTLDynamicLibrary`, its install name, label nil until set | the same |
| the linked kernel of the file: preloaded, by install name alone | made, 30 33 36 39; made, 30 33 36 39 | the same |
| the same bytes of no file: bare, preloaded | CompilerError 2, "Error creating compute variant"; the same | CompilerError 2; made, 30 33 36 39 |
| a kernel whose install name names no file: bare, preloaded | CompilerError 2; CompilerError 2 | CompilerError 2; made, 30 33 36 39 |
| the compile of source | made; it sends the dynamic library `airData` alone | the same |
| that kernel: bare, preloaded | CompilerError 2; CompilerError 2 | CompilerError 2; made, 60 63 66 69 |
| `airData`, asked twice | the file's 3,364 bytes, a new `dispatch_data_t` at each call that the caller owns (one reference past its pool) | the same |
| `newDynamicLibrary:` of an executable library, `newDynamicLibraryWithURL:` of its file | MTLBinaryArchiveDomain 5, "Expected a library with type MTLLibraryTypeDynamic, found invalid type: null" | the same |
| `newDynamicLibraryWithURL:` of a missing file | NSCocoaErrorDomain 260 | the same |
| `serializeToURL:` a new file; a file in a missing directory | YES, 10,080 bytes, read back as the same library; NO, MTLDynamicLibraryDomain 1, "Invalid URL", no file | YES, 24,480 bytes; the same |

So both Macs load the install names of a library read from a file, and of no other: the same
bytes made into a library, or a compile of source whose library records the same name, link
nothing by it *(class 3, of the two x86 Macs; the M3 is not measured)*. A preloaded library links
a kernel on the Intel, and on the AMD only where its install name loads it from a file anyway
*(class 2)*.

What our stack does (m3-max-21422 on a Radeon 780M and m3-max-21423 on an Iris Xe, macOS 26.7.2
guests): every answer above as the AMD gives it, but `serializeToURL:`, which writes the library's
.metallib (3,364 bytes, the offline `metal -dynamiclib` output's bytes, which both Macs read back),
and `airData`, which answers the same object at each call, with a reference for the caller. A
preloaded library links only where an install name loads it, as on the AMD, so a kernel links
only what its install names load from files: the bundle reads each file an install name names in
the guest's process and sends its library to the host under that name.

## Q31: residency sets

Probe: `oracle_residency DEVICE`, 11-12, the M3 Max alone (m3-max-21721, m3-max-21768 with the
internal use, and m3-max-21882 with when a queue lets a removed set go; m3-max-21701 stopped at the
huge capacity): sets of a default descriptor, of one with
a label and an initial capacity of 16, and of one of 1,048,576; a 4 KiB Shared buffer, a 64 x 64
texture, a 1 MiB heap, a buffer of the heap and a view of the texture added, removed and committed
in turn; the sizes of more shapes; residency requests; what holds what; the residency calls of a
queue, a command buffer and its encoders; a `CAMetalLayer`'s set, with the members a set is sent by
caller; and the bad inputs, each case in a child process. The x86 Macs make no set:
`newResidencySetWithDescriptor:error:` answers nil and sets the error out to nil (m3-max-21028 and
m3-max-21029, the AMD and the Intel, macOS 26.7.1).

| asked | M3 |
|---|---|
| a set of each descriptor, and of nil | `AGXG15XFamilyResidencySet`, below `AGXResidencySet`, `IOGPUMetalResidencySet` and `_MTLObjectWithLabel`; the error out set to nil |
| a query that names the support | none: `supportsFamily:` answers Apple7 to 9, Mac2, Common3, Metal3 and Metal4 YES, and the device's members that name residency sets are its creators (`newResidencySetWithDescriptor:error:`, `allocResidencySet`) and its shader log buffer's set |
| a fresh set's `device`, `label`, `allocatedSize`, `allocationCount`, `allAllocations` | the device; nil, whatever the descriptor's label; 0; 0; an empty array, the same object at each ask |
| `setLabel:` | answered: the label is then the one set |
| an add | in `allocationCount`, `containsAllocation:` and `allAllocations` at once, before any commit; a second add of one held does nothing |
| a remove | out of `containsAllocation:` and `allAllocations` at once, but counted in `allocationCount` until the commit drops it; an add before that commit keeps it; a remove of an uncommitted add takes the add back; a remove of one never added does nothing |
| `removeAllAllocations` | a remove of each committed allocation and of each add |
| the order of `allAllocations` | another in each run |
| `allocatedSize` | 0 until a commit; then the memory the committed allocations lie in, each block once: a heap's buffer or texture, with the heap or without, the heap's 1,048,576; a view, its texture's 32,768; a texture over a 64 KiB buffer, the buffer's 65,536; a 4 KiB buffer, Shared or Private, 131,072, and two Shared ones together the same (a Shared and a Private one, 262,144); a buffer of 64 KiB, 1 MiB or 1 MiB + 4 KiB, its own size |
| `requestResidency`, `endResidency` | nothing the set answers changes, in any order, twice, or an end before a request |
| what holds what | the set holds each allocation it is given until the commit after its remove, and lets it go at that commit; a queue holds a set added until a moment after its last remove (held right after it, gone 100 ms later with no buffer made, as on a thread of the queue's own); a command buffer holds a set it uses until it completes |
| a queue's add and remove of sets, twice, of one never added, of none, then its command buffers | taken; they complete |
| a command buffer's use of sets, twice, of an empty set, of a set of uncommitted adds, after its commit | taken; it completes |
| `useResidencySet:`, `useResidencySets:count:` on a render, compute and blit encoder | answered; a set and nil taken on the render and compute encoders, which complete |
| a `CAMetalLayer`'s `residencySet` | a set of the device, empty, then holding the texture of each drawable `nextDrawable` answers; QuartzCore sends it `addAllocation:`, `removeAllocation:` and `commit` alone (its commits send `allocatedSize`, `setCurrentGeneration:` and `setExpiredGeneration:`, from IOGPU). QuartzCore makes it when the layer's device is set, labelled "CAMetalLayer.residencySet", and only for a device that answers `supportsFamily:` of Apple6 YES *(read from code: QuartzCore, macOS 27.0.1)* |
| a nil allocation added or removed, alone or in a list | ends the process (SIGSEGV) |
| `containsAllocation:` of nil | NO |
| an allocation of no device (an NSObject adopting `MTLAllocation`) | taken, counted and listed, 0 bytes; the commit ends the process (SIGSEGV) in one run of two |
| nil to a queue's add or remove, or a command buffer's use (`useInternalResidencySet:` among them) | raises NSInvalidArgumentException, the counted set's "*** -[NSCountedSet addObject:]: attempt to insert nil" (a remove: "removeObject:]: attempt to remove nil") |
| an object of no device as a set: a queue's add; a command buffer's use | taken, and its buffers complete; the buffer fails, status 5, MTLCommandBufferErrorDomain 1, "Invalid Input (0000000a:kIOGPUCommandBufferCallbackErrorInvalidInput)" |
| an initial capacity of NSUIntegerMax | the creator spins without end (stopped at 30 s) |

So Apple silicon's set is bookkeeping a commit applies, its size counts memory, not allocations, and
its queue and command buffer hold sets as counted sets *(class 3, of the one Apple GPU measured;
the x86 Macs make none)*.

What our stack does: the arm64 device makes a set of its own (`RgfxResidencySet`), which sends
nothing to the host, as the device keeps every allocation resident, and answers every answer above,
but: the label is the descriptor's, as Metal's documentation gives it ("The label specified at
creation", class 1); `allAllocations` lists the committed allocations in the order they were
committed, then the adds; a small buffer counts its own pages, its range of a slab, not a block it
shares; an allocation of no device is held and counts nothing, and a command buffer that uses an
object of no device as a set completes, inside the envelope, as no correct use depends on either;
and a capacity of NSUIntegerMax makes a set, as the capacity is a hint the set does not need. Its
queue and command buffer hold sets as the M3's do, but that the queue lets a set go at its last
remove, not a moment after, as no correct use can depend on what a moment decides; its render and
compute encoders give theirs to their command buffer, and its blit encoder answers no residency
member, as no Metal protocol gives a blit encoder one. A layer's `residencySet` stays nil on the
arm64 guest, as its device answers `supportsFamily:` of Apple6 NO: the device's family answers
decide it, not the set. The x86 device answers nil, as before.
The guest run, `oracle_residency` in the arm64 macOS 27.0.1 guest on the M3 Max under HVF
(m3-max-21942 at b15191aed, 11-12), answers what the M3 answered, case by case: every
count, `containsAllocation:` and hold, every queue and command buffer run, and every death. It
differs only where the paragraph above says it does, and in what only diagnostics show: the
classes are ours; `retainCount` counts other holds (the set's count of the memory it lies in
holds a plain buffer once more, and the queue holds a set once, however often it was added); and
a nil allocation ends the process with SIGABRT and a log line, not SIGSEGV. A first run
(m3-max-21874) showed a commit keeping the allocations it dropped alive until the caller's pool
drained, as it listed them in an autoreleased array; the set now lets them go at the commit.

## Q32: compute pipelines' threadgroup fields

Probe: `oracle_threads DEVICE`, 11-12 (m3-max-22216 on the AMD, 22217 on the Intel, 22214 on the
M3; the same with `MTL_DEBUG_LAYER=1` answers the same), each step a first compile of a salted
kernel in a child process: `plain`, `max64` of `[[max_total_threads_per_threadgroup(64)]]` and
`max256`, made of required sizes (`requiredThreadsPerThreadgroup`) and descriptor maxima
(`maxTotalThreadsPerThreadgroup`) within, at and past the kernel's and the device's 1024, through
the synchronous creators, the completion handlers and with reflection.

| asked | AMD | Intel | M3 |
|---|---|---|---|
| a required size with an axis of 0, of more threads than 1024, or more than the descriptor's maximum | ends the process: `validateWithDevice:1530: failed assertion` "Compute Pipeline Descriptor Validation", a line a failed check | as AMD | as AMD, at `validateWithDevice:1623` |
| a required size of more threads than the kernel's attribute | refused, CompilerError 2, "error: requested required threads per threadgroup W,H,D but 'NAME' expects max threads per threadgroup N" | as AMD | as AMD, AGXMetalG15X_M1 3 |
| a descriptor maximum other than what the kernel specifies (the required size's threads, else its attribute) | refused, "max_total_threads_per_threadgroup shader attribute(Metadata) value (S) does not match maxThreadsPerThreadgroup pipeline descriptor value (M)" | refused, "[[max_total_threads_per_threadgroup]] declared in Metal doesn't match [MTLComputePipelineDescriptor maxTotalThreadsPerThreadgroup]" | refused, ": Kernel specified max total threads per threadgroup (S) must match compute pipeline specified max total threads per threadgroup (M)" |
| a descriptor maximum past 1024 | refused before the kernel's check, "maxThreadsPerThreadgroup pipeline descriptor value (M) exceeds maximum supported (1024)" | made | refused after the kernel's check, ": Specified total max threads per threadgroup (M) exceeds the maximum total threads per threadgroup supported (1024)" |
| a pipeline made of a required size | answers its threads as `maxTotalThreadsPerThreadgroup` | answers 1024 (448 for one thread) | as AMD |
| the kernel compiled once before, then a required size past its attribute | made, answering the attribute | made, answering 1024 | as AMD |
| `forwardProgressUsage` 0, 1, 2; `contentionRelief` 0, 1; `optimizeForPersistentKernel` | no such fields (macOS 26) | as AMD | made; Automatic (0) answers Weak (1), as a pipeline of a function alone does |
| `forwardProgressUsage` 3 or -1; `contentionRelief` 2 or 3 | none | none | refused, AGXMetalG15X_M1 2, "Compilation failed due to an interrupted connection: XPC_ERROR_CONNECTION_INTERRUPTED. This error occurred after multiple retries." |

The assertions agree *(class 3)*; the refusals' words and order differ *(class 2)*.

What our stack does: the bundle checks the fields before it posts the create
(`RgfxComputeThreadsVerdictOf`, `RgfxComputeProgressError`), against the kernel's attribute, which
its AIR reader reads (`air.max_work_group_size`), and answers each case as above, the x86 guest as
the AMD GPU and the arm64 guest as the M3, the host sent none of it. A kernel compiled before is
refused all the same, as a first compile is and as the runtime refuses it. A pipeline of a required
size answers its threads as its maximum, and the arm64 guest's pipelines answer
`forwardProgressUsage` Automatic as Weak.

## Q33: function constants that put two arguments at one location

Probe: `oracle_overlap DEVICE`, 11-13 (m3-max-25546 on the AMD, 25547 on the Intel, 25548 on the
M3), then 11-14 with eight more functions (k_halfpred, k_halfidx, k_fidx, k_sidx, k_vft, k_ift,
k_pas and k_ias; m3-max-28663 on the AMD, 28662 on the Intel, 28659 on the M3). It covers 22
functions of `oracle_overlap.metal`: kernels, a vertex function and a fragment function. Each
has arguments of one argument table that may take overlapping locations, each present under
function constants. Every value of the two bools that gate them was tried, by
every way of making a specialised function: values set by index, by name and by range, a copy
of the values, a descriptor, a copy of it, a descriptor that names a `specializedName`, and each
completion-handler form.

The three GPUs make and refuse the same functions, line for line, with the same errors. One
pipeline differs: the 630 refuses the compute pipeline of k_vft wherever its visible function
table is present and called (useB), with `CompilerError` 1, "Compilation failed due to an
interrupted connection: XPC_ERROR_CONNECTION_INTERRUPTED. This error occurred after multiple
retries.", at every way and in each of its 11-14 runs, though it answers
`supportsFunctionPointers` true; the 5300M and the M3 make it. That is a pipeline's answer, not
the overlap's, and making it stays inside what the two x86 Macs answer. The answers:

- Where the values put two present arguments of one table at overlapping locations, Metal
  refuses the function where it makes it. It answers nil with MTLLibraryErrorDomain 3
  (CompileFailure) and one line per such argument, for example "error: buffer argument b has
  invalid location (<0,1>) in k_overlap\n".
- A line names each argument, in parameter order, whose location overlaps the location of an
  earlier present argument. The line gives that argument's own index and count, and the
  entry point's name, not the specialized name.
- The word is `buffer` for buffers, constant buffers, argument buffers and threadgroup memory,
  `texture` for textures and arrays of them, and `sampler` for samplers.
- Function tables (visible and intersection) and acceleration structures (primitive and
  instance) take locations of the buffers' table, and their word is `buffer` too.
- Threadgroup memory's table is not the buffers' table.
- An argument counts as present wherever its predicate is true. It counts whether the function
  uses it or not, and whether its code is dead under the values or not.
- An index set by a function constant (`[[buffer(kIdx)]]`) counts like a literal one, and so
  do a predicate and an index computed from a half constant (`kH > 1.0h`, `uint(kH * 0.5h)`)
  and an index through conversions to a float and back, truncated toward zero
  (`uint(float(kIdx + 1u) * 0.75f)`, `uint(int(float(kW) * -0.5f) + 2 - int(kIdx))`, where -1.5
  gives -1).
- With no overlap, the function and its pipeline are made.

Where this showed: the guest's pipelines tool (`mps.fn`) gives
`cnnConvArray_32x32_128` every constant 0. Metal refuses that at once with "texture argument
src has invalid location (<0,1>)". The bundle accepted it, and the host's compile refused the
pipeline later (`OverlappingArgument`).

What our stack does: the bundle reads each argument's predicate, index and count. It reads them
from the function's AIR, under the module's initialisers run with the values
(`rgfxAirConstantGlobals`). It does not follow the entry point's body, which costs many times the
initialisers at every make of the function. It refuses as above in
`newFunctionWithName:constantValues:error:`, which every other way reaches, with no call to the
host (`RgfxOverlappingArgumentError`). For this check, the initialisers' evaluator also folds a
half constant's value, each half result rounded to the nearest half as LLVM folds one, and the
compiler's scalar conversions between floats and integers, the calls `air.convert.u.i32.f.f16`
and the like that its initialisers make (read: the probe's module). Where the AIR does not tell,
the host's compile refuses the pipeline, as before (read from code):

- a predicate, index or count whose initialiser takes an operation the evaluator does not fold:
  it folds integer and float arithmetic, comparisons, casts, `air.convert` of scalars, selects,
  branches and `is_function_constant_defined`, not a vector constant's elements, an array's
  elements, a conversion of two integers or of vectors, or other calls;
- an `air.buffer` with no `air.address_space`, whose table (buffers or threadgroup memory) is not
  known: none of the 22 functions of the 11-14 probe, built with Xcode 27.2 for Metal 3.0, writes
  one;
- a function of a library Metal's link made (`newLibraryWithDAG:functions:error:`), which the
  bundle makes over the linked library's bytes but keeps no source for, so no AIR is read.

A stitched library holds stitched visible functions, which have no argument locations, and a
library of a function array answers the function given whatever the values (oracle_bundle
libforms2 and libforms3), as Metal's does: neither has a function to refuse here.

## Q34: what an Apple GPU answers where the x86 Macs' rules differ, and the arm64 guest

An arm64 guest's device takes an Apple GPU's rules (`rgfxAppleGPU`, RgfxTexture.h), as every GPU
an arm64 Mac has is Apple's: its Metal reference is the M3 Max's, where the x86 guest's is the
x86 Macs'. This section measures the M3 where the bundle had taken the x86 Macs' answer.

Probes, 11-13, on the M3 Max under macOS 27.0.1 (26A434), each case in a child process of its
own, with `MTL_DEBUG_LAYER`, `METAL_DEVICE_WRAPPER_TYPE`, `MTL_SHADER_VALIDATION` and
`METAL_ERROR_MODE` unset (each probe prints them): `oracle_surftex Apple`, 297 cases with its
fourth and fifth rounds (m3-max-26136, 26149 and 26152); `oracle_bundle Apple` view.*,
buftex.*, linear.*, linalign, heapbuffers, heap.shared, bit17 to bit19, tiled.responds,
iosurf.buffer and devtex.* (m3-max-26136 and 26149), and argbuf.ds.* (m3-max-26237 and 26239);
`oracle_mesh Apple icb` (m3-max-26237); and `oracle_spi_members` (m3-max-26149). Then
`oracle_bundle Apple` ds.queries, ds.resourceIndex, ds.uniqueIdentifier, tiled.bytes, tiled.buffer,
functypes.dyn, libforms3, texmem and texaddr, `oracle_icb Apple spi`, `oracle_fnptr Apple`,
`tools/pipeline_reflection_check Apple` and `tools/reflection_check` (m3-max-26347),
`oracle_bundle Apple ds.grid` with `pipeline_reflection_check Apple` again (m3-max-26373), and
`oracle_profile 10 10002` (m3-max-26482).
Every answer is measured with no validation layer, as a guest's app meets it: every process end
below is the release Metal's, not the validation layer's.

| call | x86 Macs (Q20, Q24, Q9) | M3 Max, measured | arm64 guest |
|---|---|---|---|
| `newTextureWithDescriptor:iosurface:plane:`, a compressed format (BC1, BC7, ASTC 4x4, ETC2 RGB8, EAC R11, PVRTC RGBA 4 bpp) whose rows and height fit, over a surface without planes, a one-plane BGRA surface's plane, or a 420v surface's luma plane, Managed or Private | BC made; the others end the process at the descriptor's validation (formats they lack) | nil; rows or height past the surface's end the process at the third step, in the x86 Macs' words | the M3's (`rgfxSurfaceMakesNoTextureOf`) |
| the same, a Cube or CubeArray type, 64 or 32 a side, over a one-plane BGRA surface's plane or a 420v surface's luma plane | made | ends the process: `Assertion failed: (carriesComplexTexture \|\| numFaces == 1), function -[IOGPUMetalTexture initWithDevice:descriptor:iosurface:plane:field:args:argsSize:], file IOGPUMetalTexture.m, line 687.`, a C assertion of the driver's | the M3's (`rgfxEndAtInvalidSurfaceTexture`) |
| the same, a two-plane format of 4:2:0 or 4:2:2 chroma (500, 502, 505, 506, 508, 509, 520, 522, 525, 526, 528, 529, 540, 541, 543, 544) of an odd width or height (63 x 63, 64 x 63, 63 x 64 over the luma plane; 31 x 31 over the chroma plane; 63 x 63 over a one-plane surface and one without planes), usage RenderTarget or ShaderWrite | ends the process (formats they lack) | nil; of usage ShaderRead alone, or of even sides, made; the 4:4:4 formats (503, 507, 510, 523, 527, 530, 542, 545) and the XR10 ones (550, 551) made at every size and usage asked | the M3's (`rgfxSurfaceMakesNoTextureOf`) |
| a view in `MTLPixelFormatInvalid` (`view.noformat`) | ends the process | nil | the M3's (`rgfxEndAtViewMetalEnds`) |
| a view in a format of another class, of levels or slices past the texture's, of a type of another family, or a cube of slices not a multiple of six (`view.class`, `.levels`, `.slices`, `.type`, `.slicecount`) | ends the process | ends the process, in the x86 Macs' words | unchanged |
| `newHeapWithDescriptor:`, Shared, 4 MiB | ends the process ("Shared storage mode disallowed") | made (`AGXG15XFamilyHeap`); Shared buffers of 1 to 65537 bytes in it, each with contents; a Shared BGRA8Unorm texture in it | the M3's: a heap over a block of the Shared pool where the device has one, else of the device region (`heapForDeviceWithDescriptor:`) |
| `minimumLinearTextureAlignmentForPixelFormat:`, every raw value from 0 to 799 and 1000, 1023, 4096 and 65535 | 64 values answered; ends at the rest | 175 values answered, 16 each: those the x86 Macs answer, those they lack but Invalid, D24S8 and X24S8 (B5G6R5, A1BGR5, ABGR4, sRGB R8 and RG8, Unspecialized, the two-plane and XR10 formats), and 26, 44, 66, 116, 501, 504, 521, 524, 546 to 548, 560, 562 to 613 and 615 to 629; ends as at a compressed format at 130 to 135, 140 to 143, 150 to 153, 160 to 167, 170, 172, 174, 176, 178 to 183, 185 to 200, 203 to 218, 221 to 236, 240 and 241; as at a depth or stencil one at 250, 252, 253, 260, 261, 651 and 652; as Invalid at every other | the M3's (`rgfxLinearAlignmentAnswer`); the alignment answered stays the host GPU's |
| a texture buffer, a 2D texture by the public form or the linear SPI, or a linear 2D array, over a buffer, whose texels end past the buffer's length (`oracle_pastlen Apple limits`, m3-max-32081, 32093 and 32098, 11-15) | made at every end bisected, up to 16 MiB from the buffer's start (Q11) | nil past the buffer's `resourceSize` from its start: 128 KiB for a buffer under 16 KiB, a heap's size for a heap's buffer, the 16 KiB pages a no-copy buffer's bytes meet, and the length rounded up to 16 KiB for the rest (Q11) | the M3's (`rgfxAppleAllocation`, `rgfxPastAllocation:`) |
| a depth or stencil texture (Depth16Unorm, Depth32Float, Stencil8, Depth32Float_Stencil8, X32_Stencil8) in Shared storage over a Shared buffer, by `newTextureWithDescriptor:offset:bytesPerRow:`, and Depth32Float by the linear SPI | ends the process at the descriptor's validation (the storage) | ends the process after the validation, which takes it: `_mtlValidateStrideTextureParameters`, "Linear texture: cannot create compressed, depth, or stencil textures, pixelFormat (...) disallowed" | the M3's (`rgfxEndAtLinearDepthStencil`) |
| a depth/stencil state's `gpuResourceID`; an argument encoder's `setDepthStencilState:atIndex:` of a state or nil, at a depth/stencil descriptor's index, beside a texture and a pointer, on an encoder of one texture, and on one of a function's argument buffer (`argbuf.ds.*`) | raise an unrecognized selector, though each answers that it responds | `gpuResourceID` 0 for every state; the encoders write nothing and raise nothing | the M3's (`RgfxDepthStencilState`, `RgfxArgumentEncoder`) |
| a depth/stencil state's `resourceIndex` and `uniqueIdentifier` (SPI; `ds.resourceIndex`, `ds.uniqueIdentifier`) | answer that they respond, and raise an unrecognized selector | the same | unchanged |
| a depth/stencil state's `readsDepth`, `writesDepth`, `readsStencil` and `writesStencil` (SPI, which Apple's GL renderer asks of each state it binds; `ds.queries`, `ds.grid`: 717 descriptors, every depth compare with and without writes, every stencil compare of either face with each set of operations and read and write masks of 0, 1 and 0xff, under the depth tests Always, Never and Less) | YES to each for every descriptor | what the descriptor does: `writesDepth` the depth write; `readsDepth` that, or a depth compare other than Always and Never; a face writes the stencil where its write mask is not 0 and one of its three operations is not Keep, and reads it where it writes it, or its read mask is not 0 and its compare is neither Always nor Never; each query answers either face's | the M3's (`RgfxDepthStencilState`) |
| a pipeline's reflection of a constant buffer: its `bufferAlignment` and its pointer type's `alignment` (`pipeline_reflection_check Apple`, 144 cases) | 32 | its pointee's, but at least 4: 16 for a struct of `float4x4` members, 8 for a `half3`, 4 for a `float`, a `half` and a `char`; every other getter of the 144 cases answers as the bundle's reflection does, and only two private flag fields (a function's register spills and memory atomics, which no getter answers) differ | the M3's (`RgfxPipelineReflection`) |
| a library's reflection of its functions (`reflection_check`, 19 functions) | as the bundle's | the same | unchanged |
| a function handle's `gpuResourceID` (`oracle_fnptr Apple`), of a compute pipeline's linked functions and a render pipeline's fragment ones | raises an unrecognized selector (Q29) | 0 | the M3's (`RgfxFunctionHandle`) |
| a compute pipeline's `functionHandleWithName:` of a function it links | raises an unrecognized selector (Q29) | that function's handle | unchanged: an unlinked name's answer is not measured |
| a kernel linked against a dynamic library whose own library was made of the same bytes as its file (`newLibraryWithData:`), or compiled from source with the dynamic library in its options, bare and preloaded | CompilerError 2, "Error creating compute variant", but the Intel's preloaded (Q30) | made, and they run as the file's (30 33 36 39, 60 63 66 69) | the M3's for an `@executable_path` install name (`pipelineLibraries`); an `@loader_path` one, which no file is the loader of, still refused (not measured) |
| a kernel whose install name names no file, bare and preloaded | CompilerError 2, "Error creating compute variant" | AGXMetalG15X_M1 2, "Undefined symbols:\n  SYMBOL, referenced from: KERNEL\n" | unchanged: the bundle does not know the symbol |
| `newDynamicLibrary:` of an executable library, `newDynamicLibraryWithURL:` of its file or of a missing file; `serializeToURL:` into a missing directory | MTLBinaryArchiveDomain 5 (Q30); NSCocoaErrorDomain 260; MTLDynamicLibraryDomain 1, "Invalid URL" | the same, in the same words | unchanged |
| a function of a dynamic library's exported function (`functypes.dyn`) | ends the process at `validateMTLFunctionType`'s assertion | the same | unchanged |
| a texture's SPI `virtualAddress` and `gpuAddress`, of each storage mode and shape, in heaps, over buffers, views and over a surface (`texaddr`, `texmem`) | a Shared or Managed texture's CPU address of its memory, in its GPU's layout; NULL for a Private one and one over a surface; a heap's offset in a Placement heap | the same pattern, in the M3's twiddled and compressed layouts | unchanged: as on an x86 guest, a Shared texture answers NULL (`RgfxTexture`) |
| `featureProfile` (SPI), which Metal reads at device creation; what Metal's base device (`_MTLDevice`) answers under it (`oracle_profile 10 10002`, a subclass of the base that answers the profile alone, m3-max-26482) | 10002 | 10; under 10 the base answers 143 of its getters otherwise than under 10002, 20 of which the device answers itself (families, feature sets, function pointers, image blocks, tile shaders, programmable blending, raster order groups, barycentrics, the linear texture alignment), and 123 it leaves to the base: among them `supportsMemorylessRenderTargets`, `supportsSparseTextures`, `supportsSparseHeaps` and `sparseTexturesSupport` (3), the `supportsMTL4*` queues, encoders, compiler and counters, `supportsMeshShadersInICB`, `supportsBufferWithIOSurface`, `supportsTensors`, `supportsMachineLearningCommandEncoders`, `supportsLossyCompression`, `supportsRasterOrderGroupsColorAttachment`, `supportsVariableRateRasterization`, `supportsPVRTC` and the ASTC and YCbCr format queries, YES under 10 and NO under 10002; `maxTileBuffers`, `maxTileTextures`, `maxTileSamplers`, `maxTileInlineDataSize` and `minTilePixels` (31, 128, 16, 4096 and 256, against 0), `maxInterpolants` (124 against 32), `maxIndirectBuffers` and `maxIndirectTextures` (96 against 64 and 128), `maxFramebufferStorageBits` (512 against 1024) and `minConstantBufferAlignmentBytes` (4 against 32); the base's feature queries (`_MTLDeviceFeatureQueries`) follow the same split | unchanged, 10002: under 10 the base would answer that the device has an Apple GPU's memoryless targets, tile memory, sparse textures, Metal 4 queues and mesh commands in indirect command buffers, which the host's GPU does not; the move waits for the device to answer those members itself. The tiled texture SPI stays as it is with it: the M3's device's `newTiledTextureWithBytesNoCopy:` in both forms raises an unrecognized selector of its buffer's `newTiledTextureWithDescriptor:offset:bytesPerRow:`, which its buffers do not answer, in every storage mode and at 1 and 6 levels (`tiled.bytes`, `tiled.buffer`), while under 10002 WindowServer asks for those textures (Q11) |
| an indirect render or compute command's read-back SPI (`oracle_icb Apple spi`) | what each command holds | the same; a fresh command's draw arguments are 0x20000000 | unchanged: the device keeps commands where these cannot read them, on either guest (`RgfxIndirectCommandBuffer`) |
| an indirect command buffer of the mesh command types, and a mesh draw through it (`oracle_mesh Apple icb`) | no mesh pipeline supports indirect command buffers (Q22) | made, 2 commands, `allocatedSize` 16384; both mesh draws through it draw | unchanged: mesh pipelines that support indirect command buffers are a feature the device lacks |
| a buffer by length (each storage mode) or by bytes (Shared, Managed), or the device's texture, of options holding bit 18 | nil | nil | unchanged (`RgfxMakesNoResourceOf`) |
| the same of options holding bit 17 or 19 | made | bit 17: buffers made, then a Shared texture of the same process ends it in `IOGPUMetalSuballocatorAllocate` ("Different MTLResourceOptions used for the same heap index, this is not supported"); bit 19: buffers nil, the texture ends it the same | unchanged; not measured one call a process |
| `newBufferWithIOSurface:` (SPI) over 64 x 64 and 100 x 7 BGRA surfaces and a 420v one | nil | a buffer (`AGXG15XFamilyBuffer`) of the surface's allocation size, options 0x200, contents NULL, `iosurface` the surface, a GPU write through it reaching the surface's pages | unchanged: nil; a buffer over a surface is a feature the device lacks |
| a buffer's tiled texture SPI, `newTiledTextureWithDescriptor:offset:bytesPerRow:` | answered | not answered by the M3's buffers in any storage mode (the linear SPI is); the device answers `newTiledTextureWithBytesNoCopy:` in both forms | unchanged: the bundle's buffers answer it |
| the members `tests/protocol_members.txt` gives the x86 Macs' verdicts for (`%spi-both`, `%spi-neither`, `%spi-differ`), by `respondsToSelector:` | the bundle answers each as they do | of the 67 the probe asks, the M3's objects answer 50 (44 selectors) the bundle's do not: 49 of the `%spi-neither` ones (a buffer's backing SPI such as `detachBacking` and `parentGPUAddress`, a command buffer's resource groups and `encodeSignalEventScheduled:value:` (which an arm64 guest's buffers answer since 11-14, Q45), a compute encoder's virtual substreams, a render encoder's resolve textures and tile dispatches, a texture's `compressionMetadata` and `generateMipmapLevel:slice:`), and `copyFromTexture:...options:`, a `%spi-differ` one | unchanged: SPI members, not rules |

The M3 answers the rest of what this section asked as the x86 Macs do (the surface cases of Q20
it shares with them, the views' rules, a BC1 texture over a buffer's bytes answered nil, rows
short of a texture's ending the process). Its words name its own classes (`IOGPUMetalTexture`,
`IOGPUMetalHeap`) where the x86 Macs name `MTLIOAccelTexture` and `MTLIOAccelHeap`; the bundle
keeps the x86 Macs' names in its words, which are diagnostics.

What our stack does *(class 3 for each guest: each answer is its reference GPU's)*: where the
device answers an Apple family, and in every arm64 guest, the bundle answers the M3's column
above, sending the host nothing for what ends the process or answers nil; an x86 guest's answers
are unchanged. The shell tests check both: `test_surface_textures_as_an_apple_gpu`,
`test_views_as_an_apple_gpu`, `test_shared_heaps_on_an_apple_gpu`,
`test_linear_alignment_as_an_apple_gpu`, `test_depth_buffer_textures_as_an_apple_gpu`,
`test_apple_textures_past_a_buffers_allocation_answer_nil`, `test_gl_renderer_spi`,
`test_pipeline_reflection_from_air`, `test_function_pointers` and
`test_compute_pipeline_libraries`, and the protocol tests' `test_argument_encoder_members`, beside
the x86 Macs' tests they keep.

## Q35: what each kind of resource answers to `setPurgeableState:`

Probe: `oracle_bundle DEVICE purgeable.kinds`, 11-13 (m3-max-26711 the AMD and m3-max-26712 the
Intel under macOS 26.7.1, m3-max-26713 the M3 Max under macOS 27.0.1): each resource is told
Volatile, KeepCurrent, Empty, KeepCurrent, NonVolatile and KeepCurrent in turn, and each answer
is printed (2 NonVolatile, 3 Volatile, 4 Empty).

| resource | AMD, Intel and M3 Max alike |
|---|---|
| a Shared or Private buffer by length, a Private texture, a texture over an IOSurface, a heap, an indirect command buffer | the state told before: 2 3 3 4 4 2 |
| a buffer by `newBufferWithBytesNoCopy:` over the process's pages, a texture view, a texture over a buffer, a heap's buffer or texture | NonVolatile to each: 2 2 2 2 2 2 |

A visible function table (Q29) answers NonVolatile to each on the Intel and the M3 Max, and keeps
the state told on the AMD.

What our stack does: each answer above, on every guest. A no-copy buffer stays NonVolatile
(`RgfxBuffer`), as a view, a texture over a buffer and a heap's resource do, and so does a
visible function table, the Intel's and the M3's answer (`RgfxVisibleFunctionTable`); the rest
keep the state told (`RgfxResource`). The device purges nothing, whatever the state.

## Q36: which usage bits a texture answers, and which end the process

Probe: `oracle_surftex DEVICE`, its sixth and seventh rounds, 11-13 (m3-max-27149 and 27157 the
M3 Max under macOS 27.0.1, 27152 and 27156 the AMD 5300M and 27151 and 27155 the Intel UHD 630
under 26.7.1), each case in a child process of its own with `MTL_DEBUG_LAYER`,
`METAL_DEVICE_WRAPPER_TYPE`, `MTL_SHADER_VALIDATION` and `METAL_ERROR_MODE` unset. The atomic bit
with each format over each surface kind and over none (`nosurface.` cases, by
`newTextureWithDescriptor:`), in Managed, Shared and Private storage, with and without
PixelFormatView, a view of the texture over the surface, and every usage bit the SDK names none
for, alone and three at once.

All three Macs answer alike, but where the table says:

| usage asked | AMD, Intel and M3 Max |
|---|---|
| ShaderAtomic (0x20) on R32Uint or R32Sint, with any other bits | made; the answer leaves 0x20 out (0x23 answers 0x3, 0x20 answers 0, 0x37 answers 0x17), over an IOSurface or over none, in each storage mode; a view of the texture answers the same |
| ShaderAtomic on RG32Uint | made, 0x20 left out, on the M3; ends the process on the x86 Macs: `This device does not support ulong texture atomics, so MTLPixelFormatRG32Uint cannot be used with texture atomics` |
| ShaderAtomic on any other format (BGRA8Unorm, R32Float, RGBA8Uint, R8Uint, R16Uint, RG32Sint, RGBA32Uint, 500) | ends the process at `-[MTLTextureDescriptorInternal validateWithDevice:]`: `MTLPixelFormatBGRA8Unorm is not a valid format to be used with texture atomics.`, before a lacked format's line (500 on the x86 Macs) |
| 0x8 | made; the answer leaves it out (0x9 answers 0x1, 0x8 answers 0) |
| 0x4000, 0x8000, 0x10000 | made and answered as asked |
| 0x40, 0x80, 0x100, 0x200 to 0x2000, 0x20000 to 0x100000, 0x1000000, 0x80000000 | ends the process at the same validation: `MTLTextureUsage has unknown bits 0x40.`, the bits in one line (`0x1c0` for three), before the atomic bit's line |

What our stack does *(class 3, and the M3's column for RG32Uint on an arm64 guest, class 2)*: a
texture answers the usage it was made with but 0x20 and 0x8 (`rgfxAnsweredUsage`, RgfxTexture.mm),
on every guest, while its create carries the usage given, so the host's texture keeps the atomic
bit; and every creator ends the process at the unknown bits and the atomic formats above, in
Metal's words (`rgfxEndAtInvalidTextureDescriptor`). Until 11-13 a texture answered the bits as
given, and the bundle ended the process at 0x80 alone. The shell tests'
`test_texture_usage_leaves_out_atomic` and the `usage.*` and `atomic.*` descriptor cases hold it.

## Q37: what one command buffer's encoders see of each other's writes through two textures over one plane

Probe: `oracle_surftex DEVICE --alias N`, its aliasing cases, 11-13 (m3-max-27359 the M3 Max
under macOS 27.0.1; m3-max-27356 the AMD 5300M and the Intel UHD 630 under 26.7.1), each case in
a child process of its own with `MTL_DEBUG_LAYER` unset. An `'L008'` surface 64 x 64 in rows of
128 bytes; `wide`, an `RG8Unorm` 64 x 64 texture over its plane 0, its rows the surface's whole
128 bytes, and `narrow`, an `R8Unorm` 64 x 64 one, the plane's own texels. One command buffer's
first encoder writes through one of them (a blit from a buffer, or a render pass that clears and
stores) and its second blits the other into a Shared buffer; both textures keep the device's
hazard tracking, or are Untracked with the writer updating a fence the reader waits for. 16 runs a
case, the CPU setting every byte of the surface to 0x11 under a lock before each.

| what | M3 Max | AMD 5300M and Intel UHD 630 |
|---|---|---|
| the second encoder reads the first's writes through the other texture, either way, tracked or behind a fence | every byte, 16 of 16 runs, every case | the same |
| the surface after the command buffer, under a read-only lock: the plane's texels | the writes, 16 of 16 | the same |
| the bytes past the plane's texels after the wide texture wrote them | the writes, 16 of 16 | the CPU's 0x11, 16 of 16: the GPU's writes there never reach the CPU |
| what the wide texture reads past the plane's texels, the CPU having written 0x11 there | 0x11, 16 of 16 | never 0x11 (0 of 16) |

So Metal orders a write through one texture over a plane before the next encoder's read through
another over it, on all three; the bytes past a plane's texels are the surface's own on the M3
and the GPU's own on the x86 Macs, which sync only the texels with the CPU. Neither answer is a
promise of Metal's.

What our stack does *(class 3 for the order; class 2 for the bytes past the texels)*: on a host
whose device cache lays a plane out at the GPU's pitch, a texture past the plane's texels lies over
the block's wide home, and each encoder that writes through either it or a texture over the cache is
followed in its command buffer by a copy of its texel rows into the other side, before the next
encoder (`decode::surface_cache`, "Wider than the plane"). The home over the block's view of its
pages gives the M3's bytes past the texels; the stand-in, where the block has no view, gives the x86
Macs'. Until 11-13 such a texture raised `RowPitch` where it was wider than the cache's pitch
(`l008.fmt.rg8` on an arm64 macOS 27 guest, MoltenVK), and read the cache's own bytes past the plane
where it fitted the pitch. On an arm64 macOS 27.0.1 guest on the M3 the eight cases now answer as
the M3 does natively (before: the wide texture refused, so its writes never landed and the narrow
texture's never reached it).

## Q38: writes into read-only memory

Probe: `oracle_readonly.sh DEVICE` (`oracle_readonly`), 11-14, every shape in a process of its
own, against a writable control (`plain`): the M3 Max under macOS 27.0.1 (26A434), m3-max-29292
(2 runs a shape) and m3-max-29309 (4 runs of the shapes it asked again); the AMD 5300M and the
Intel UHD 630 under macOS 26.7.1 (25G241), m3-max-29293 and m3-max-29294, then m3-max-29310 and
m3-max-29311 (2 runs a shape each). Read-only memory is a no-copy buffer over the process's pages
or a heap made with `MTLResourceOptions` bit 24, Metal's private read-only option, which the
guest's no-copy buffers over memory its process maps read-only carry (design.md, "No-copy
buffers"), or a buffer over pages the process maps read-only (`ro-pages-`). Each shape's command
buffer is sent twice on one queue, with a fill on the same queue and on a new one after each.

**On the x86 Macs bit 24 does nothing.** Every GPU write below lands in the pages (or the heap's
texture) and completes, as without the bit, on both GPUs. Over pages the process maps read-only,
every GPU write measured (a pass's store, colour resolve and `StoreAndMultisampleResolve`, a
fragment or vertex function's and a kernel's `write()`, a blit's copy) completes, the pages
unchanged. The M3 Max keeps the GPU's writes out under bit 24 and over read-only pages alike.

| what, into read-only memory | M3 Max | AMD 5300M and Intel UHD 630 | class |
|---|---|---|---|
| a pass's store-out: a colour `Store`; a colour `MultisampleResolve` or `StoreAndMultisampleResolve` into it, the descriptor's or late, into the descriptor's resolve texture or one the setter's SPI named under a known action; a depth or stencil `Store`; a depth `CustomSampleDepthStore`, the descriptor's or late; a depth `MultisampleResolve` and a stencil `MultisampleResolve` or `StoreAndMultisampleResolve`, the descriptor's or late, a depth one into a late texture too | `PageFault` (3), every first command buffer, the event signalled and the memory unchanged; the next command buffer of the queue and of a new queue run; the second command buffer faults too and revokes the queue (`AccessRevoked`, 4), but for a stencil `StoreAndMultisampleResolve`'s 4 runs of 6, whose second fails with code 1 (`kIOGPUCommandBufferCallbackErrorInnocentVictim`), the queue running on | completes, the memory written | 2 |
| a depth attachment's `StoreAndMultisampleResolve` into it | completes, 4 runs of 6 (2 failed with code 1, `kIOGPUCommandBufferCallbackErrorInnocentVictim`), the second command buffer 2 of 6 (4 the same victims), never `PageFault`, the queue never revoked; the resolve texture unchanged | completes, 4 of 4 each, the resolve texture written | 3 |
| a pass that does not store it: `DontCare` of colour, depth or stencil; a colour or stencil `CustomSampleDepthStore` set late; a resolve the pass makes into a texture of a format pair Metal leaves untouched (BGRA8 into RGBA16Float) | completes, the memory unchanged | completes; the untouched pair's resolve written | 3 (the untouched pair: 2) |
| a colour or stencil attachment the descriptor keeps by `CustomSampleDepthStore` | ends the process in the descriptor's validation (`abort`), with the bit or without | the same | 3 |
| a fragment or vertex function's `write()` of a texture over it | `PageFault`, every first command buffer; the second faults and revokes the queue, but 2 runs of 12 whose second fails with code 1, the queue running on | completes, the pages written | 2 |
| a texture over it made resident for writing that no function writes | completes | completes | 3 |
| a kernel's `write()`; a blit's copy into a texture over it; mipmaps of a heap's texture | completes, the writes dropped (`mipmaps-shared`: level 1 of a Shared heap's texture whose level 0 the CPU wrote stays zeros, written without the bit) | completes, written | 2 |
| a blit's colour or bytes fill and an invalidation of such a texture (the `fillTexture:` and `invalidateCompressedTexture:` SPI) | completes, the writes dropped | ends the process, with the bit or without: the SPI is not the x86 drivers' | 2 |
| the CPU's `replaceRegion:` of a texture over a bit-24 buffer of writable pages | the pages take the bytes, which the GPU reads | the same | 3 |
| the CPU's `replaceRegion:` of a texture in a Shared heap of bit 24 | the texture takes the bytes, which a blit reads back | no Shared heap is made | 3 (the M3 alone) |
| the CPU's `replaceRegion:` of a texture over pages the process maps read-only | ends the process (`SIGBUS`), with the bit or without | the same | 3 |
| a late colour resolve texture where the pass named none, under a late resolving action | completes, nothing resolved, with the bit or without | the 5300M ends the process at `endEncoding` (`SIGSEGV`); the 630 fails the command buffer (code 9), with the bit or without | 2 |

What our stack does, on every host: the rt platform makes read-only memory (a no-copy buffer or
heap of bit 24, and pages the process maps read-only) over a copy of its pages whose GPU writes
it drops, the pages written whole into the copy at each commit that uses it, so the CPU and the
command buffers committed after a GPU write read the pages' bytes (accuracy-ledger.md,
`read-only-pages-written`; `crates/reims-vgpu-vulkan/src/rt_platform/copied.rs`, "Writes
dropped"). Every GPU write into it completes, dropped: a kernel's and a blit's, as the M3 Max's
*(class 2)*; a draw's `write()`, which the M3 Max faults, and a render pass's store-out, its
stores and resolves there, which the M3 Max faults with `PageFault`, as the 5300M and the UHD
630 complete them over pages the process maps read-only, the pages unchanged *(class 2: the x86
Macs' completion, the cheaper)*. A pass with an attachment over such a copy is counted at the
pass, `rt-copied-store-outs`, the process's first with a line (`copied.rs`, `store_outs`). Within
a command buffer, and between command buffers in flight at once, a read after such a write sees
it until the next commit that uses the copy. A CPU write writes the pages where the process can,
and ends the process over pages it maps read-only *(class 3)*.

## Q39: a command buffer failed two ways: a read-only store-out beside protection options

Probe: `oracle_failways.sh DEVICE` (`oracle_failways`), 11-14, every shape in a process of its
own, against a writable control (`plain`): the M3 Max under macOS 27.0.1 (26A434), m3-max-29535
(2 runs a shape) and m3-max-29547 (4 runs); the AMD 5300M and the Intel UHD 630 under macOS
26.7.1 (25G241), m3-max-29536 and m3-max-29537 (2 runs; a probe error lost the shapes of four
encoders there) and m3-max-29549 and m3-max-29550 (4 runs). A command buffer of compute encoders
that each write a mark and a render pass that clears and stores a 4 x 4 RGBA8Unorm texture over a
no-copy buffer made with bit 24 (Q38's store-out), each encoder followed by the signal of an
event of its own, with `setProtectionOptions:` (the command buffer's SPI) of 2 or 0x10 set before
the pass, after it, or after every encoder. Type-9 draws, which hang the GPU, are left out on
every Mac: what one does beside another failure is not measured.

| what | M3 Max | AMD 5300M | Intel UHD 630 | class |
|---|---|---|---|---|
| encoders made under options 2 | code 16 ("Protection Violation"); the first encoder made under them and every later one run nothing; the signals encoded before it fire, none after it | the same | code 16; every encoder runs and every signal fires (the error's encoder infos say Pending) | 2 |
| encoders made under options 0x10 | code 1 ("Invalid Input"); encoders and signals as under 2 | code 16; as under 2 | code 16; as under 2 | 2 (the code: 4 for the x86 lane, below) |
| options set after every encoder | 2: completes; 0x10: code 1, every encoder running, every signal firing | completes, under both | the same | 2 (0x10: 4 for the x86 lane) |
| a store-out alone | `PageFault` (3), as Q38; the compute encoder after the pass ran in 11 runs of 18 (here and beside options 2 or 0x10 set after every encoder); every signal fires | completes, the store written | the same | 2 |
| a store-out, then encoders made under options 2 or 0x10 | the options' failure alone, code 16 or 1, no `PageFault`; the encoders made under them run nothing; the signals encoded before the first of them fire, the pass's among them; the pages unchanged | the same, code 16, the store written | code 16, every encoder running and every signal firing, the store written | 2 |
| encoders made under the options, the pass among them | the options' failure; the pass runs nothing, so nothing faults | the same, code 16 | code 16, every encoder running | 2 |
| a store-out, then options 0x10 after every encoder | code 1, no `PageFault`; every signal fires; the encoder after the pass ran in 3 runs of 6 | completes, the store written | the same | 2 |
| a store-out, then options 2 after every encoder | `PageFault`, as the store-out alone | completes, the store written | the same | 2 |

The next command buffer on the same queue, and one on a new queue, ran after every shape on all
three Macs.

What our stack does: the rt platform builds no protection options. A command buffer with options
other than 0 runs as one without them, every encoder running and every signal firing, counted
`rt-unbuilt-protection` (accuracy-ledger.md;
`crates/reims-vgpu-vulkan/src/rt_platform/command.rs`, "Protection options"): where encoders
are made under options 2 or 0x10 it completes, which no Mac answers *(class 4)*; options set
after every encoder complete it, as the x86 Macs answer under both and the M3 Max under 2
*(class 2)*. A store-out into
read-only memory completes, its store dropped (Q38), alone and beside options, where the M3 Max
faults it alone *(class 2: the x86 Macs' completion)*. A type-9 draw, of a primitive type Metal
documents none of, is refused, counted and dropped, beside another way or not
(`crates/reims-vgpu-vulkan/src/rt_platform/render.rs`, `primitive`).

Outside the envelope for x86 guests *(class 4, not yet answered)*: the M3 Max's code 1 under
options from 0x10, and its failure where such options come after every encoder; both x86 Macs
answer code 16 and complete the latter. The platform fails no command buffer for its options.

## Q40: what Metal stores through the 4:4:4 `_sRGB` two-plane formats

Probe: `oracle_twinwrites.sh DEVICE` (`oracle_twinwrites`), 11-14, one process a format and a
write: the M3 Max under macOS 27.0.1 (26A434), m3-max-29535; the AMD 5300M and the Intel UHD
630 under macOS 26.7.1 (25G241), m3-max-29536 and m3-max-29537, which end the process at every
texture of the four formats, at its creation ("MTLTextureDescriptor has invalid pixelFormat
(N)", as Q13 and Q20 found): the x86 Macs make none. An 8 x 2 texture of 503, 507 or their
`_sRGB` twins 523 (YCBCR8_444_2P_sRGB) and 527 (YCBCR10_444_2P_sRGB) over plane 0 of a two-plane
surface ('444v' for the 8-bit ones, 'x444' for the 10-bit ones; rows of 64 bytes, plane 1 at
4096, no matrix), each column x written colour x, alpha 1, by a kernel's `write()`, by a fragment
function's `write()` drawn over a BGRA8Unorm target of its size, and by a render pass a colour
that clears the texture and stores it.

Every command buffer completed. Both rows hold the same bytes, and a 10-bit sample's code is its
16-bit word's top 10 bits, its low 6 zero. The codes, as Y, Cb and Cr; the kernel's and the
fragment function's alike in every case, and the pass's but where named:

| colour | 503 | 523 | 507 | 527 |
|---|---|---|---|---|
| black | 16, 128, 128 | 16, 128, 128 | 64, 512, 512 | 64, 512, 512 |
| white | 235, 128, 128 | 235, 128, 128 | 940, 512, 512 | 940, 512, 512 |
| grey 0.5 | 126, 128, 128 | 177, 128, 128 | 502, 512, 512 | 708, 512, 512 |
| red | 81, 90, 240 | 81, 90, 240 | 326, 361, 960 | 326, 361, 960 |
| green | 145, 54, 34 | 145, 54, 34 | 578, 215, 137 | 578, 215, 137 |
| blue | 41, 240, 110 | 41, 240, 110 | 164, 960, 439 | 164, 960, 439 |
| (0.2, 0.6, 0.9) | 129, 177, 78 | 174, 157, 90 | 515, 707, 311 | 696, 630, 360; the pass 697, 630, 360 |
| (0.9, 0.3, 0.1) | 116, 83, 199 | 162, 88, 174 | 464, 332, 795 | 649, 351, 695 |

Where every channel is 0 or 1 a twin stores its format's codes. Elsewhere its codes are those of
the colour encoded by the sRGB curve first: grey 0.5 encodes to 0.7354, and 16 + 219 x 0.7354 is
177.05 and 64 + 876 x 0.7354 is 708.2; (0.2, 0.6, 0.9) encodes to (0.4845, 0.7977, 0.9547),
whose BT.601 codes at the video range are 174.1, 157.4 and 90.05, and 696.4 for the 10-bit luma
(arithmetic on these eight colours alone; the 10-bit pass rounds that luma up).

What our stack does: the runtime (metal2vulkan 20f7e82bbe) stores neither twin's writes. It
refuses a kernel's or a fragment function's write through one at the slot's bind, and a pass
into one at its creation, so the dispatch, the draw or the pass ends the process, naming the
refusal; a blit into one from a buffer completes *(class 4: Metal stores them; these bytes went
to metal2vulkan)*. Through an argument buffer the runtime refuses a write of every two-plane
YCbCr texture, of 503, 507, 542 and 545, whose slot writes it stores, and of 500, whose slot
writes it drops as Metal does, among them, ending the process at the dispatch *(class 4: the M3
Max stores them as through a slot, below; metal2vulkan takes them on)*. Measured on ms01's Iris
Xe (ANV), m3-max-29517 and m3-max-29593, over
textures made as the device makes them over a guest's surface plane
(`gpu::ycbcr::plane_writes`). Only an arm64 guest reaches it: the x86 guests' bundle ends the
process at a texture of a two-plane format's creation, as the x86 Macs do (Q13; read from code,
`guest/bundle/src/RgfxTexture.mm`'s descriptor check).

**Through an argument buffer.** `oracle_twinwrites.sh Apple` with `FORMATS="500 503 507 523
527" OPS="kernel argbuf"`, the M3 Max under macOS 27.0.1 (26A434), m3-max-29643: the kernel
writes the eight colours through an argument buffer that holds the texture
(`MTLArgumentEncoder`), made resident with `useResource:usage:` of `MTLResourceUsageWrite`,
beside the same kernel's write through a slot, one process each. 500 is an 8 x 2 texture over
plane 0 of a '420v' surface (plane 1 of 4 x 1 Cb, Cr pairs at 4096). Every command buffer
completed, and both planes hold the same bytes after the write through the argument buffer as
after the one through the slot, format by format: 503, 507, 523 and 527 store the codes of the
table above, and 500 stores nothing, both planes keeping their 0xAB bytes. So a write through
an argument buffer stores what a write through a slot stores *(measured on the M3 Max alone, as
the x86 Macs make none of these textures)*.

## Q41: a colour resolve into a texture over an IOSurface, in a pair the M3 leaves untouched

Probe: `oracle_surfresolve.sh DEVICE` (`oracle_surfresolve`), 11-14, one process a case: the M3
Max under macOS 27.0.1 (26A434), m3-max-29682, and m3-max-29690 for plane 1; the AMD 5300M and
the Intel UHD 630 under macOS 26.7.1 (25G241), m3-max-29684 and m3-max-29683, and m3-max-29691
and m3-max-29692 for plane 1. A 16 x 16 4-sample `BGRA8Unorm` or `BGRA8Unorm_sRGB` attachment
(Private, of its own) is cleared and stored, then loaded, drawn over every texel and stored by
`MultisampleResolve` or `StoreAndMultisampleResolve` into a 16 x 16 texture of usage
`ShaderRead` and `RenderTarget`, filled with 0xA5 bytes first, of `RGBA16Float`, `BGR10_XR`,
`BGR10_XR_sRGB`, or the attachment's own format (the control): a texture of its own, one over a
surface without planes of its size ('RGhA', 'w30r' or 'BGRA'), one over plane 0 of a 128 x 16
'420v' surface, or one over plane 1 of a 128 x 32 '420v' surface, each Shared, Managed and
Private. A blit reads the texture back, and a second pass resolves the samples into a texture of
their format.

| what | M3 Max | AMD 5300M and Intel UHD 630 |
|---|---|---|
| the resolve into `RGBA16Float`, every texture and storage mode | completes, the texture keeps its bytes | completes, the drawn colour written |
| the resolve into `BGR10_XR` or `BGR10_XR_sRGB` | completes, the texture keeps its bytes | none: the process ends at the texture's descriptor ("MTLTextureDescriptor has invalid pixelFormat (554)", `abort`) |
| the resolve into the attachment's own format | the drawn colour written | the same |
| the samples after `MultisampleResolve` | as before the pass (the earlier clear) | the drawn colour |
| the samples after `StoreAndMultisampleResolve` | the drawn colour | the same |

Over a surface, without planes or over either plane of one with planes, in every storage mode,
each GPU answers as for a texture of its own: what decides the resolve is the format pair, not
the texture's memory *(class 2: the M3 Max leaves the texture untouched where the x86 Macs write
it)*.

What our stack does: a resolve into a texture of another format that the M3 Max leaves untouched
is dropped (`gpu::encode_render`'s `untouched_resolves`): the pass is made without it, its
samples kept as the M3 Max keeps them, and the texture keeps its texels, the M3 Max's answer,
which costs nothing, as the runtime resolves into no other format. A texture the guest makes
over a surface's plane, which the device makes over the buffer that stands for the surface
(`new_texture_over_surface_plane`), Private over a device cache whatever the guest asked, is
dropped into alike, over a surface without planes or a plane of one with planes, in any storage
mode, and so is one over read-only memory, whose pass completes with the memory unchanged as the
M3 Max completes it (Q38). Until 11-14 such a pass ended the process at its making, as a
resolve into a texture over a buffer, a shape not measured.

## Q42: the device's compiler target, and Metal's archive readers

Probe: `oracle_archtype DEVICE FILE...`, 11-14: the AMD 5300M and the Intel UHD 630 under macOS
26.7.1 (25G241), m3-max-30048 and m3-max-30049, and the M3 Max under macOS 27.0.1 (26A434),
m3-max-30050. The files: a `.metallib` of AIR (`oracle.metallib`), a dynamic library of AIR
(`oracle_bundle_dyn.metallib`), and a binary archive and a dynamic library the device serializes
itself.

| what | AMD 5300M | Intel UHD 630 | M3 Max |
|---|---|---|---|
| `targetDeviceArchitecture` | an `MTLTargetDeviceArchitecture` (Metal's class, of `NSObject`) | the same | the same |
| its `cpuType`, `subType` | 0x1000014, 0x1773 | 0x1000015, 0x950200 | 0x1000013, 0x1a3 |
| its `driverVersion`, `versionCombined` | 0x7d4, 0xe0002 | 0x1a0701, 0x1a0701 | 0x30000, 0x30000 |
| the archive reader, AIR `.metallib`; AIR dynamic library | 1, 1 | 1, 1 | 1, 1 |
| the dynamic library reader, the same files | 0, 1 | 0, 1 | 0, 1 |
| the archive reader, the device's own archive; its dynamic library | 2, 1 | 3, 1 | 2, 1 |
| the dynamic library reader, the device's own archive; its dynamic library | 0, 2 | 1, 1 | 0, 2 |

The class has `cpuType`, `subType`, `driverVersion`, `version` (`{MTLTargetDeviceArchVersion=SCC}`)
and `versionCombined`, and setters of all but the last. The readers' numbers are the names
MTLAssetUpgraderD prints for them: the archive reader's 0 Invalid, 1 Legacy, 2 Compatible,
3 Rebuild, 4 NoVendorSlice, and the dynamic library reader's 0 Invalid, 1 AIROnly, 2 Compatible,
3 Rebuild, 4 NoVendorSlice.

Every Mac answers an object, and no Mac answers nil *(class 3)*. Its values are each GPU's own
compiler target and versions: an answer that describes the device, which is the device's own and
not a pick between the Macs'. What it is read for (read from macOS 27's Metal and MTLAssetUpgraderD):
- Metal's loader (`+[MTLLoader sliceIDForDevice:legacyDriverVersion:airntDriverVersion:]`) makes
  the slice it looks for in a binary archive or a dynamic library of `cpuType` and `subType`, and
  ends the process where the answer is nil. MTLAssetUpgraderD, told of an app installed, asks
  Metal's archive and dynamic library readers about each `.metallib` in it, for each device, and
  queues a rebuild of the file for the device where the archive reader answers 3 or the dynamic
  library reader 1 or 3.
- MPS keeps `cpuType << 32 | subType` and gives it to its NDArray kernels' prefix and postfix
  queries, and Metal's `_MTLUseAIRNTBinaryArchive` answers YES for a `cpuType` of 0x1000013 to
  0x1000015 alone, where no setting of Metal's decides it.

What our stack does: until 11-14 the device left `targetDeviceArchitecture` to the IOGPU base,
which answers nil *(class 4)*, and MTLAssetUpgraderD ended with SIGABRT at each app installed in
the guest. The device now answers an `MTLTargetDeviceArchitecture` of its own target: `cpuType`
0x01005247, `CPU_ARCH_ABI64` with its PCI device id, which no Metal slice carries, `subType` 0, and
version 1.0.0 (`versionCombined` 0x10000) with `driverVersion` 0x10000. No Mac's archive holds a
slice of it, and the device's own archives are of its own format.

## Q43: a command a kernel encodes naming a released state

Probe: `oracle_icb DEVICE freedstate`, 11-14, 2 runs on each Mac: the M3 Max under macOS 27.0.1
(26A434), m3-max-30466; the AMD 5300M and the Intel UHD 630 under macOS 26.7.1 (25G241),
m3-max-30464 and m3-max-30465. A kernel writes compute command 0 of an indirect command buffer
from an argument buffer holding the ID of an `icb_add` state x of indirect command buffer support,
which the program released (its last reference dropped in an autorelease pool), and the command is
executed. `out` reads 5 where `icb_add` ran, 6 where an `icb_mul` state y ran, and 2 where nothing
did.

| shape | M3 Max | AMD 5300M | Intel UHD 630 | class |
|---|---|---|---|---|
| x released before the write, no state made since | x runs (5); completes | the command buffer fails with code 2 (`kIOAccelCommandBufferCallbackErrorTimeout`), nothing run (2) | x runs (5); completes | 2 |
| x released after the write's command buffer completed, before the execute | x runs | the same | the same | 3 |
| x released before the write, y made since | y took x's ID (7) and runs (6) | the same | y took a new ID (10), and x runs (5) | 3: the ID's last holder runs |
| the CPU set x, x released, y made, the kernel writes the state | y runs (6) | the same | x runs (5): y took another ID | 3, as above |
| the CPU set x, x released, y made, the kernel writes the buffers alone | x runs (5) | the same | the same | 3 |

- **Which state runs** *(class 3)*. A record the GPU wrote runs the state that last took the ID it
  names, released or not, and a command whose state the GPU did not write runs the state the CPU
  set, whoever took its ID since. Every command buffer but the 5300M's first completes with no
  error. The M3 Max and the 5300M hand a new state the lowest free ID, the 630 a new one.
- **A state released before the write, its ID taken by none** *(class 2)*. The M3 Max and the 630
  run it; the 5300M times the command buffer out.

What our stack does (metal2vulkan d9236a8a11, v0.2.0): a record the GPU wrote runs the program of
the state that held the ID as the GPU last could write the buffer's records, released since or not:
the device keeps the program of each freed ID's last holder while an indirect command buffer of it
lives, and a buffer the GPU may write takes those at its first such write, so the second shape is
every Mac's. In the first shape x runs and the command buffer completes, the M3 Max's and the 630's
answer *(class 2)*, unannounced. The record tells a pipeline a kernel wrote from one the CPU set, so
the fourth shape runs y, the ID's last holder, and the fifth x, every Mac's answer, both
unannounced. Still relayed (`released-command-state`), the command running nothing, no Mac's answer
*(class 4)*: a state released while no indirect command buffer of the device lived, or past its
ID's last four holders. Until 11-14 the process ended there.

## Q44: an indirect command buffer made with `MTLResourceOptions` bit 24

Probe: `oracle_readonly.sh DEVICE` with `SHAPES="icb-cpu icb-gpu icb-gpu-over-cpu"`, 11-14, 2 runs
a shape and backing, beside Q43's jobs (m3-max-30466, m3-max-30464, m3-max-30465). An indirect
command buffer of one compute command, made with bit 24 or without it (the control), Shared where
the CPU encodes and Private where a kernel alone does; its command a dispatch that adds 5 to a word
of 16, the CPU's (`icb-cpu`), a kernel's (`icb-gpu`), or a kernel's over the CPU's add of 1
(`icb-gpu-over-cpu`); executed in one command buffer, and again in another.

| what | M3 Max | AMD 5300M and Intel UHD 630 | class |
|---|---|---|---|
| the buffer | made; `resourceOptions` without the bit (0x200 Shared, 0x220 Private), as without it | the same | 3 |
| a command the CPU encoded | runs (21), in both command buffers | the same | 3 |
| a command a kernel encoded | nothing runs (16): the GPU's write of the record is dropped | runs (21) | 2 |
| a kernel's command over the CPU's | the CPU's runs (17) | the kernel's runs (21) | 2 |

Every command buffer completes with no error, and the queue runs on, as without the bit.

What our stack does (metal2vulkan ce54ad8b25): the runtime makes such a buffer as without the bit,
relayed (`private-resource-option`), dying in strict mode: the CPU's commands run, every Mac's
answer, and a kernel's records land, the x86 Macs' *(class 2)*. The M3 Max's drop of the GPU's
writes is the runtime's to build, for an arm64 guest's frameworks.

## Q45: a command buffer's scheduled signal, `encodeSignalEventScheduled:value:`

Probe: `oracle_cmdbuf Apple schedsig` and its child cases `schedsig_open`, `schedsig_late`,
`schedsig_nil`, `schedsig_nil_unretained`, `schedsig_plain` and `schedsig_plain_unretained`, 11-14,
on the M3 Max under macOS 27.0.1 (26A434): m3-max-30621 and m3-max-30629. The x86 Macs' buffers
do not answer the member (`%spi-neither`, Q34), so it is the M3's alone. Its command buffer
(`AGXG15XFamilyCommandBuffer`) answers it from `IOGPUMetalCommandBuffer`, retaining its
references or not. Times are ms from the commit, medians of 5 runs; a kernel ran 92 ms.

| what | M3 Max |
|---|---|
| on an idle queue, the signal of 1 encoded before a kernel: `signaledValue` reads 1 (a polling thread); a listener's block for 1; the scheduled handler; `GPUStartTime` | 0.044; 0.061; 0.114; 0.638. The scheduled handler reads 1 in 5 of 5 |
| the same, the signal encoded after the kernel | 0.090; 0.104; 0.109; 0.606. The handler reads 1 in 5 of 5, the listener's block before it in 2 of 5: the signal waits for none of the buffer's GPU work |
| behind a 185 ms buffer of its queue: the value read; the buffer's scheduled handler; its `GPUStartTime`; a buffer of a second queue that waits for the value completes | 0.092; 0.138; 19.97; 20.45, before the first buffer's `GPUEndTime` (184.98) in 5 of 5 |
| an event at 5: a scheduled signal of 3; of 7; of 6 then 8; of 8 then 6; against a GPU signal of 3 and a CPU set of 3 | 5; 7; 8; 8; 5 and 5: the value only rises, as a GPU signal and a CPU set leave it |
| a kernel, a GPU signal of 2, a scheduled signal of 1 | 1 at 0.117 (the scheduled handler reads 1), 2 at 46.78, as the kernel ends (`GPUEndTime` 46.66) |
| a scheduled signal of 2, a kernel, a GPU signal of 1 | 2 at 0.062, a listener for 1 at 0.059; 2 at the end |
| a wait for another shared event W at 1, then the signal; the CPU sets W at 200 ms (3 runs) | the buffer Committed at 200 ms; the value reads 1 0.04 ms after the set, the scheduled handler 0.1 ms later |
| the signal, then the wait for W | the value reads 1 at 0.03 to 0.07 ms, while the buffer stays Committed until W is set |
| a kernel, the wait for W, the signal | the value reads 1 0.04 ms after the set |
| the signal, a kernel, the wait for W | at 0.03 to 0.04 ms |
| behind an earlier buffer of its queue that waits for W | 0.06 to 0.17 ms after the set |
| a wait for 1 on its own event, then the signal of 1, nothing else setting it | the value stays 0; the buffer fails at 5.0 s with Metal's timeout (`MTLCommandBufferErrorDomain` 2, "Caused GPU Timeout Error (00000002:kIOGPUCommandBufferCallbackErrorTimeout)") |
| the signal of 1, then a wait for 1 on its own event | the buffer completes (10.7 ms) |
| a listener's block, a default listener and one of a queue of its own | on the listener's queue (`MTLSharedEventListener`, and its own) |
| with an encoder open | ends the process: "-[IOGPUMetalCommandBuffer encodeSignalEventScheduled:value:]:468: failed assertion `encodeSignalEventScheduled:value: with uncommitted encoder'" |
| after the commit | returns; the event keeps its value, and the buffer completes |
| a nil event, on a buffer that retains its references | raises `NSInvalidArgumentException`, "*** -[__NSArrayM insertObject:atIndex:]: object cannot be nil" |
| a nil event, on one that retains none | returns; the buffer fails: `MTLCommandBufferErrorDomain` 1, "Invalid Input (0000000a:kIOGPUCommandBufferCallbackErrorInvalidInput)" |
| a plain event (`newEvent`), either buffer | logs "MTLEvent passed to API expecting MTLSharedEvent" and ends the process (SIGABRT) |

So the call writes a command into the buffer's stream for the kernel, which sets the shared event
as its queue's submission reaches it: in the stream's order, behind every wait before it on the
queue, the buffer's own and an earlier buffer's, and ahead of the buffer's GPU work, its
scheduling and any wait after it (read from code: macOS 27.0.1's IOGPU, which reserves a kernel
command for it in the buffer's storage, where a wait is one too).

Who sends it (read from code, macOS 27.0.1's shared cache): MPSGraph, from one place,
`GPU::ANERegionCallOpHandler::encodeAsynchronousWithSharedEvents`, the op that runs a graph's
region on the Neural Engine, with no `respondsToSelector:` check. An arm64 guest has no Neural
Engine (docs/qemu.md, "arm64 guests: the Neural Engine"), so no recorded run reached it. CMImaging
and PhotosensitivityProcessing carry the member only in their copies of the `MTLCommandBufferSPI`
protocol: neither has a call site of it.

What our stack does *(class 3, the arm64 guest's reference GPU's)*: an x86 guest's buffers answer
no such member, as the x86 Macs' do. An arm64 guest's buffers answer it, and the bundle sets the
event as the CPU's set does (the host first, `SHARED_EVENT_SET_VALUE`, then the process's
listeners and waits): at the buffer's post, where no buffer of its queue is held at a wait and
every wait the buffer encoded before the call is met as the process knows it; else at the
buffer's scheduling, before its scheduled handlers. Later than Metal's *(class 2)*: a signal
behind a wait met on the host before the process knows it, and one held that a wait after it on
its buffer holds too, as Metal's is not; a buffer that holds a signal behind an earlier buffer's
wait and then waits for that signal's value never runs, where Metal's does. The plain event's
end, the nil event's exception and its failed buffer, and the call after the commit are the
M3's. The open encoder's end is not carried: the bundle keeps no record of an open encoder for any
command-buffer call. The tests are `test_a_scheduled_signal_is_set_as_its_queue_submits_it`,
`test_a_scheduled_signal_behind_a_wait_is_set_at_its_scheduling` and
`test_a_scheduled_signal_takes_only_a_shared_event` (tests/queue_tests.mm, arm64).

## Q46: an argument encoder's reflection, and a function array's names

Probes, 11-14: `oracle_bundle DEVICE` argbuf.reflection, argbuf.binding, argbuf.binding.lib,
libforms3 and libforms.order, on the M3 Max under macOS 27.0.1 (m3-max-31079) and on the AMD
Radeon Pro 5300M and the Intel UHD Graphics 630 under macOS 26.7.1 (m3-max-31082 and 31083), as
on 11-05 for argbuf.* (the x86 Macs) and 11-13 for libforms3 (the M3).

**An argument encoder's reflection.** `newArgumentEncoderWithBufferIndex:reflection:` of a
kernel's argument buffer answers, in its out-parameter, the buffer's binding as the function's
pipelines' reflection answers it: of the argument buffer's members, a `metal::array` of textures
of stride 0. A library's reflection of the same function (`reflectionForFunctionWithName:`)
answers that array's stride as its element's size, 8. AMD, Intel and the M3 agree *(class 3)*.
Ours (`RgfxFunction.mm`, `rgfxArgumentOfBufferAtIndex:`) answers the pipelines' binding
(`RgfxStageReflectionOf`), and the library's reflection where that makes none
(`test_pipeline_reflection_from_air`'s encoder binding, tests/shell_tests.mm).

**A function array's names.** A library made of functions (`newLibraryWithFunctions:` and the
stitched forms, libforms3) answers `functionNames` in an order of its own, which follows the order
the functions were given without reversing it, alike on AMD, Intel and the M3 *(class 3)*:
libforms.order's 6 orders of div_kernel, div_visible and div_stitchable and 4 orders of five
(adding div_vertex and div_fragment), and a probe's 60 orders of two, three and four of
div_visible, div_stitchable, div_vertex and div_fragment (m3-max-31122 on the M3, 31123 and 31124
on AMD and Intel, the same 60 answers). Given div_visible first, the three answer div_kernel,
div_stitchable, div_visible whatever follows; given div_stitchable first, div_kernel, div_visible,
div_stitchable; given div_kernel first, div_stitchable, div_visible, div_kernel. The given order
reversed is the answer for 3 of libforms.order's 6 orders of three and 2 of its 4 of five, and for
28 of the probe's 60: all 12 of two, 12 of the 24 of three and 4 of the 24 of four. No container
of the names gives it: `NSMutableDictionary` (with and without a capacity), `NSDictionary`,
`NSMutableSet`, `NSMapTable` and `CFMutableDictionary`, filled in the given order or reversed and
read by their keys, matched at best 10 of the 60 (m3-max-31104 and the probe above), and libc++'s
`std::unordered_map` and `std::unordered_set` of the names 44 of 70 orders (m3-max-31128): it is
a hash structure of Metal's own. Ours (`RgfxLibrary.mm`, `functionNames`) answers the given order
reversed *(class 2: Metal's for every array of two and for some longer ones)*.

## Q47: binary functions, and the device's compile of a visible function

Probe: `oracle_binfn DEVICE`, 11-16, on the AMD Radeon Pro 5300M and the Intel UHD Graphics 630
under macOS 26.7.1 (m3-max-37213 and 37214; the device's own handles and tables m3-max-37220 and
37221). It wraps the device class's three forms of `compileVisibleFunction:withDescriptor:`, so
each call Metal makes of them is printed. macOS 12 and 13 guests' iconservicesagent and
`com.apple.dock.extra` send the form with an error out at start-up, from Metal's own link of
Core Image's kernels, which the bundle answered unsupported until then (17 boots of 11-15/16).

| asked | AMD | Intel |
|---|---|---|
| the device responds to the error, archive and handler forms | YES, YES, YES | the same |
| `newFunctionWithDescriptor:` of a visible function with `MTLFunctionOptionCompileToBinary` | the function, options 1; Metal's library sends the error form once, with the function (options still 0) and the descriptor | the same |
| the same of a kernel | the function, options 1; no compile sent | the same |
| the error form, sent with a visible function and a descriptor of options 0 or 1, a nil descriptor, or a kernel | returns, the error untouched | the same |
| the archive form, into an archive of the device | returns, the error untouched; the archive then has one key, of the function | the same |
| the handler form | calls its handler, of one argument, with nil, on the caller's thread before it returns | the same |
| a compute pipeline linking `binaryFunctions` (binary ones, or one made without the option) | made; the handles of its functions, dispatches through them right | the same |
| `newComputePipelineStateWithAdditionalBinaryFunctions:error:` (a binary function, a plain one, none) | the same pipeline object each time, no error; no handle of an added function | another pipeline each time, linking the added function: its handle made, a dispatch through it right |
| the device's `functionHandleWithFunction:` (macOS 26) and `functionHandleWithFunction:resourceIndex:` | raise an unrecognized selector, though the device responds to both | the same |
| the device's `newVisibleFunctionTableWithDescriptor:` | a table (`GFXAAMD_MtlVisibleFunctionTable`) | a table (`MTLIGAccelVisibleFunctionTable`) |

Ours (`RgfxMetalDevice.mm`, `RgfxBinaryArchive`'s `rgfxCompileVisibleFunction:descriptor:archive:`,
`RgfxComputePipelineState.mm`), where the host makes function pointers: each compile form as the
two Macs answer it, compiling nothing, as the host compiles a binary function where a pipeline links
it (`test_visible_compiles`, tests/archive_tests.mm); more binary functions answer the pipeline
itself, as the AMD's does *(class 2: the Macs diverge, and the AMD's answer makes nothing)*
(`test_function_pointers`, tests/shell_tests.mm); the device's handles raise as the two Macs' do. The
device's table stays unsupported: a table of the wire belongs to a pipeline
(`FUNCTION_TABLE_CREATE`).

## Q48: another process's read of an IOSurface behind the writing process's backlog

Probe: `oracle_backlog DEVICE ARM 6 20`, 11-21, on the AMD Radeon Pro 5300M and the Intel UHD
Graphics 630 under macOS 26.7.1 (25G241), off screen: m3-max-64081. It takes Q16's `xproc raw`
from one long pass to a queue of work. Process A commits 8 busy compute buffers on one queue, each
calibrated to about 20 ms of GPU time, then a ninth that writes 1000 + round into a 1024 x 1024
R32Uint surface S: by `commitAndWaitUntilSubmitted` in arms 1 and 2, by a plain `commit` in arm 3.
In arm 2 A then commits 4 more busy buffers. Then process B commits a blit that copies S into a
buffer, by `commitAndWaitUntilSubmitted`. On odd rounds the ninth writes an unrelated surface
instead, as a control. Each GPU ran 3 rounds and 3 controls of each arm. Times are ms from B's
commit, on one clock for both processes, and B's buffer completed in every round. A's busy buffers
ran 20.9 to 21.1 ms each on the 5300M; on the 630, 19.7 to 20.2 ms in arms 2 and 3, and 10.8 to
16.5 ms in arm 1, whose calibration came out short (16.6 ms).

| what | AMD 5300M | Intel 630 |
|---|---|---|
| B's `commitAndWaitUntilSubmitted` returns; its scheduled handler | 0.08 to 0.37; 0.12 to 0.41 (1.03 to 1.32; 1.05 to 1.34 in a run's first round) | 0.05 to 0.08, and 1.06 in one round; 0.08 to 0.13, and 1.09 in that one (3.25 to 4.81; 3.27 to 4.81 in a run's first round) |
| A's call for the ninth returns: `commitAndWaitUntilSubmitted` (arms 1 and 2); `commit` (arm 3) | 0.28 to 1.46; 0.002 to 0.012 | 0.09 to 1.00; 0.002 to 0.009 |
| arm 1: B, the ninth writing S | completes 0.67 to 0.83 ms after A's ninth (at 163.6 to 170.1), reading A's write, 3 of 3 | 0.52 to 0.95 ms after A's ninth (at 90.5 to 125.6), reading A's write, 3 of 3 |
| arm 1: the controls | at 0.99 to 1.97, 166.9 to 167.0 ms before A's ninth | 0.50 to 0.68 ms after A's ninth (at 86.9 to 116.5) |
| arm 2: B, the ninth writing S | 0.80 to 0.83 ms after A's ninth (at 165.1 to 170.2), reading A's write; 19.06 to 19.26 ms before the first of A's 4 later buffers completes, 79.06 to 79.11 before the last, 3 of 3 | 79.24 to 79.48 ms after A's ninth (at 227.9 to 236.9), reading A's write; 0.46 to 0.66 ms after the last of the 4, 3 of 3 |
| arm 2: the controls | at 1.90 to 1.99, before A's ninth and all 4 | 0.51 to 0.52 ms after the last of the 4 (at 236.1 to 236.9) |
| arm 3: B, the ninth writing S | 0.82 and 0.83 ms after A's ninth (at 165.8 and 170.4), reading A's write, in 2 rounds; in 1, at 2.76, 166.68 ms before A's ninth, reading S as it was before the write | 0.49 to 0.65 ms after A's ninth (at 152.6 to 161.9), reading A's write, 3 of 3 |
| arm 3: the controls | at 2.55 to 2.73, before A's ninth | 0.50 to 0.53 ms after A's ninth (at 161.7 to 162.1) |

So on both GPUs a read committed after the writer's `commitAndWaitUntilSubmitted` returned waits
out the writing process's whole queue ahead of the write, here 8 buffers of about 20 ms, and
completes after the write *(class 3: both x86 GPUs; the M3 did not run it)*. Neither GPU holds
back B's submission or scheduling for it: outside a run's first round both come within 0.41 ms of
the commit (one 630 round 1.09 ms), as in Q16. What else the read waits for differs *(class 2)*:
- **The 5300M orders the read by the surface alone.** A control, which reads S with nothing
  pending on it, completes in 1 to 3 ms, ahead of A's queue. In arm 2 the read completes 19 ms
  before the first of the 4 buffers A committed after the write, though A committed them before
  B committed the read.
- **The 630 runs the two processes' work first in, first out.** In each of its 18 rounds,
  controls included, B completed after every buffer A committed before it, 0.46 to 0.95 ms after
  the last.
- **A writer committed by a plain `commit` is not always ordered on the 5300M.** `commit` returns
  before Metal submits the buffer (Q16), and a read committed after it ran ahead of the write in 1
  of 3 rounds. The 630's order put all 3 after it.

What our stack does: a buffer that reads S names, at its post, the latest buffer of each other
queue that writes S, and the host makes it wait on the GPU for that queue's surface timeline to
reach the writer's seqno, which it reaches once the writer and every buffer before it on its
queue have finished (`surface_order.rs`; design.md, "Surface order"). So the read waits out the
writer's queue ahead of the write, as arm 1 does on both GPUs, and its `CB_SUBMITTED` and
`CB_SCHEDULED` are not held back for it, as B's were not. The wait names no buffer the writer's
queue committed after the writer, but the Vulkan platform runs every queue's work on one
`VkQueue` in submission order (design.md, "A command buffer in parts"), so a read the host submits
after them runs after them, as the 630 runs arm 2 *(class 2: the 630's answer)*. The 5300M's
overtaking is the GPU's own order across processes, which a change of the runtime's queues may
adopt. A writer committed by a plain `commit` is named before its `commit` returns
(`RgfxIOSurface.h`), so a read committed after that waits for it every time, as on the 630:
stricter than the 5300M, which left 1 of 3 such reads unordered, and kept as it is, since the wait
is on the GPU and holds back no record (Q16, "A divergence the lane does not use").

## Q49: how many command buffers a queue keeps in flight, and how many Apple's GL renderer does

Probes, 11-21, off screen:
- `oracle_depth DEVICE 10 48` on the AMD Radeon Pro 5300M and the Intel UHD Graphics 630 under
  macOS 26.7.1 (25G241), run twice on the 5300M and once on the 630 (m3-max-65538), and twice on
  the M3 (m3-max-65551). It makes a queue and makes and commits busy compute buffers back to back
  from one thread: 48 on a queue made with `newCommandQueueWithMaxCommandBufferCount:32`, by
  `commitAndWaitUntilSubmitted` and by `commit`, and 80 on a queue of `newCommandQueue`'s default
  count, each way. Each buffer's spin is calibrated to about 10 ms of GPU time, and a fifth arm
  repeats the first with buffers of about 2 ms. Each arm's span, first commit to last
  completion, came within 2.1% of the buffers' GPU times added up, so the GPU ran them one after
  another, with the queue full. Counts are of buffers made and not completed when a creation
  starts.
- `oracle_gldepth 5 120` and `oracle_gldepth 10 80` on the M3 (m3-max-65545). It hooks the Metal
  device's queue creators, and each queue's command-buffer creators, before it makes a CGL context.
  Then it draws into an FBO, calibrated to about 5 or 10 ms of the GPU's wall time a flush:
  - one draw a `glFlush`;
  - eight draws a `glFlush`, each after a `glBufferSubData` of the quad's vertex buffer.

`oracle_depth`, times in ms from each buffer's commit:

| what | AMD 5300M | Intel 630 | M3 |
|---|---|---|---|
| `maxCommandBufferCount` a queue answers (SPI), asked 32; default | 32; 64 | 34; 64 | 32; 64 |
| the first `-commandBuffer` that blocks over 1 ms, asked 32 | the 33rd buffer's (the 34th in one arm), with 32 uncompleted | the 35th to 38th buffer's, with 34 uncompleted | the 33rd buffer's, with 32 uncompleted |
| the same on the default queue | the 65th buffer's, with 64 uncompleted | the 65th to 69th buffer's, with 64 uncompleted | by `commit`, the 65th buffer's, with 64 uncompleted; by `commitAndWaitUntilSubmitted`, none (the call blocks first, below) |
| `commit` or `commitAndWaitUntilSubmitted` blocking over 1 ms | never, 64 deep included | never, 64 deep included | never at 32; on the default queue `commitAndWaitUntilSubmitted` blocks from the 37th buffer on, with 36 uncompleted, for about one buffer's GPU time each (p50 5.2 and 9.6 ms in the two runs) |
| commit to its scheduled handler, asked 32 | p50 0.05 to 0.22, max 0.35 | p50 0.04 to 0.24, max 0.79 | p50 0.02 to 0.08, max 0.26 |
| the same on the default queue | p50 0.13 to 0.22, max 0.39 | p50 0.06 to 0.12, max 0.81 | by `commit`, p50 32 and 54 and max 159 and 281: a buffer past the 36th uncompleted is scheduled only once one ahead of it completes |
| `commitAndWaitUntilSubmitted`'s call, asked 32 | p50 0.04 to 0.15, max 0.24 | p50 0.03 to 0.23, max 0.66 | p50 0.01 to 0.07, max 0.33 |

The GL renderer over Metal on the M3 (`GL_RENDERER` "Apple M3 Max", `GL_VERSION` "2.1 Metal -
91.7"):
- **Its queues.** It makes two queues at context creation:
  - one by `newCommandQueueWithDescriptor:` with `maxCommandBufferCount` 32, which answers 32. It
    draws on this one.
  - one by `newCommandQueue`, of the default 64. No arm made a buffer on it.
- **One draw a flush.** It makes one buffer per `glFlush`, and after each `glFlush` 32 buffers
  are uncompleted (p50 32, max 32, in both runs).
  - Its command-buffer creation blocks at the cap, and returns with 31 others uncompleted: 88 of
    120 creations at 5 ms, 48 of 80 at 10 ms.
  - `glFlush` itself returns in p50 0.05 to 0.07 ms, max 0.17.
  - The arm's wall time is the GPU's: 5.02 and 9.87 ms a flush.
- **Eight draws a flush, each after an upload.** It makes 16 buffers per `glFlush`, two for each
  draw. After each `glFlush` 31 are uncompleted (p50 and max). At 10 ms, 618 of its 1280 creations
  blocked, each returning with 31 others uncompleted.

So on all three GPUs, a queue keeps its `maxCommandBufferCount` of buffers uncompleted. Its
`-commandBuffer` blocks there until the oldest completes, and at 32 neither `commit` nor
`commitAndWaitUntilSubmitted` blocks before it. Each buffer is scheduled within 0.8 ms of its
commit *(class 3)*. What differs *(class 2)*:
- **The 630 holds 34 for an ask of 32.**
- **The M3 schedules at most 36 buffers of a queue ahead of the GPU.** Past that, its
  `commitAndWaitUntilSubmitted` waits, and a plain `commit`'s scheduled handler waits, for one to
  complete. 36 is above the 32 the GL renderer asks for.

Apple's GL renderer over Metal asks for 32. When the GPU is the slower side, it keeps 32 buffers
in flight, so its depth in frames is 32 over the buffers it makes a frame. Each upload in a frame
adds one.

What our stack does:
- **The queue's slots.** The bundle gives a queue the slots it asks for (64 for 0), takes one at
  each buffer's creation and gives it back at `CB_COMPLETE`, which the host posts once the
  buffer's work is done (`RgfxCommandObjects.mm`, `completion.rs`). The bundle keeps the 5300M's
  answer, the count asked, not the 630's.
- **Submitted and scheduled.** The host posts `CB_SUBMITTED` and `CB_SCHEDULED` at the buffer's
  take. The backlog does not hold them back, as on the x86 GPUs. The bundle does not take the M3's
  36-buffer scheduling limit: no x86 GPU has it, and the GL renderer's 32 stays under it.
- **The guest's GL renderer.** The x86-64 guest's GL is the same renderer, and a guest's GL
  thread blocks in its command-buffer getters, in the bundle's slot take. Lane glblock's walk of
  the GPU benchmark scene on the Iris Xe (m3-max-63902) measured the renderer's queue at 32
  uncompleted, at most, for a third of the scene. That is the M3's depth, about 3.8 frames at that
  scene's 8.4 buffers a frame.
- **What a Mac's own GL drivers show.** The x86 Macs' own GL drivers do not run over Metal. The
  frames a GL app keeps in flight on them (about 1.7 on the 5300M's driver, lane mbpscene) belong
  to that driver, and are no reference for the guest's GL.

## Q50: a background service's queue priority marks, and another process's short buffer

Probe: `oracle_prio DEVICE ARM 8 60 20`, 11-22, on the Intel UHD Graphics 630 and the AMD Radeon
Pro 5300M under macOS 26.7.1 (25G241), off screen: on the 630 arms 1, 2, 3 and 1 in m3-max-68129
and arms 1, 4 and 1 in m3-max-68139, on the 5300M arms 1, 2, 3, 4 and 1 in m3-max-68137 (built
in m3-max-68121 and m3-max-68136). It asks what an Intel Mac's queues do with the priority marks a
guest's processes set. The guest bundle's log of the setters (lane priolog's set, in a walk on the
Iris Xe) shows the guest's background media-analysis service marking each of its five heavy queues
`setBackgroundGPUPriority: 2` and then `setGPUPriority: 2`, and WindowServer marking two of its
queues `setGPUPriority: 0`. No process used an offset form. Process A makes one queue, marks it as
the arm says before its first buffer, and each round commits 4 compute buffers (L0 to L3) of about
60 ms of GPU time each. 17.7 to 20.1 ms into L0's GPU run, process B commits one short compute
buffer, 0.04 to 0.09 ms of GPU time alone, on a queue of its own over a buffer of its own. So L1 to
L3 are committed and waiting when B commits. The arms:
1. no marks;
2. A marked as the service marks its queues, B unmarked;
3. A as in 2, and B marked `setGPUPriority: 0`, as WindowServer marks its own;
4. A unmarked, and B marked `setGPUPriority: 0`.

A run of an arm is 8 rounds. Times are ms from B's commit, on one clock for both processes. The
buffers' `GPUStartTime` and `GPUEndTime` are on it: each run's calibration buffer started 0.20 to
0.56 ms after its commit, and its completed handler ran 0.09 to 0.16 ms after its GPU end.
m3-max-68129 ran the probe before its arm 4 and its 500 ms warm-up were added. That job's first run
calibrated short, and its buffers ran 46.7 ms. In every other run, L1 to L3 ran 59.84 to 60.27 ms.

What a queue answers, the same on both GPUs and on the M3 *(class 3)*:
- The queue is `MTLIGAccelCommandQueue` on the 630 and `GFXAAMD_MtlCmdQueue` on the 5300M, each a
  `MTLIOAccelCommandQueue`. It answers all four setters: `setGPUPriority:` and
  `setBackgroundGPUPriority:` as `c24@0:8Q16`, and their offset forms as `c28@0:8Q16S24`. Each
  returned 1 for every value set here. `MTLIOAccelCommandQueue` also has `getGPUPriority` and
  `getBackgroundGPUPriority` (`Q16@0:8`), and `_setGPUPriority:backgroundPriority:`
  (`c32@0:8Q16Q24`).
- A new queue reads GPU priority 1 and background priority 2.
  - `setBackgroundGPUPriority: 2` leaves both as they were.
  - After it, `setGPUPriority: 2` reads 2.
  - `setGPUPriority: 0` reads 0.
  - The offset forms set the same values (`setGPUPriority: 2 offset: 1` reads 2), and no getter
    reads an offset back.
- The M3 Max under macOS 27.0.1 (26A434) answers the same, every setter returning 1 and every
  readback above (m3-max-68256: the probe built for arm64 and stopped after these lines by a flag
  not committed). Its queue is `AGXG15XFamilyCommandQueue`, an `IOGPUMetalCommandQueue`, with the
  setters as `B24@0:8Q16` and `B28@0:8Q16S24`.
- So the service's marks change one value: its queues' GPU priority, from 1 to 2. WindowServer's
  change its own GPU priority from 1 to 0.

| what | Intel 630 | AMD 5300M |
|---|---|---|
| arm 1, no marks: B's GPU start | +221.74 to +223.47 in 3 runs, and +166.99 to +169.23 in the run of 46.7 ms buffers; 180.01 to 180.33 (140.25 to 140.29) ms after L0's end, after all 3 waiting buffers, 32 of 32 | +0.21 to +0.44; 49.18 to 49.95 ms before L0's end, ahead of all 3 waiting buffers, 16 of 16 |
| arm 2, A marked | +0.38 to +0.44; 41.54 to 42.06 ms before L0's end, ahead of all 3, 8 of 8 | +0.20 to +0.37; 49.19 to 49.77 ms before L0's end, ahead of all 3, 8 of 8 |
| arm 3, A and B marked | +0.38 to +0.43; 41.45 to 41.78 ms before L0's end, ahead of all 3, 8 of 8 | +0.20 to +0.41; 49.38 to 49.91 ms before L0's end, ahead of all 3, 8 of 8 |
| arm 4, B marked | +0.37 to +0.42; 41.51 to 41.93 ms before L0's end, ahead of all 3, 8 of 8 | +0.28 to +0.39; 49.28 to 50.02 ms before L0's end, ahead of all 3, 8 of 8 |
| B completes, where it starts beside L0 | +0.47 to +0.59 | +0.61 to +1.44 |
| B's scheduled handler, every arm | +0.08 to +0.14 | +0.10 to +0.31 |
| B's GPU run | 0.04 in every round (0.036 to 0.069 alone, in its warm-up) | 0.36 to 0.89 (0.050 to 0.094 alone) |
| L0's GPU run past the median of L1 to L3, a run's median | 1.80 to 1.83 in arm 1's runs of 60 ms buffers; 2.00, 1.96 and 1.96 in arms 2, 3 and 4 | 7.65 to 7.87 in every arm, arm 1's included |

Neither GPU holds back B's scheduling: its scheduled handler runs within 0.31 ms of its commit in
every arm. How soon B runs differs *(class 2)*:
- **The 630 orders by the queues' GPU priority, and preempts for it.**
  - Between queues of one GPU priority it runs the two processes' work first in, first out (arm 1,
    as in Q48).
  - A buffer whose queue's GPU priority value is below the running buffer's starts within 0.44 ms
    of its commit, about 41.5 ms before the running buffer ends, and ahead of every buffer waiting
    behind it. That holds for the service's 2 against the default 1 (arm 2) and for WindowServer's
    0 against the default (arm 4). The background priority kept its default of 2 in every arm, so
    no effect of it shows apart from the GPU priority.
  - B's GPU run beside L0 is what it is alone, so B did not share the GPU with A's kernel. L0's run
    grows about 0.15 ms over the controls', B's 0.04 ms and about 0.1 ms more: A is switched out
    for B and back.
- **The 5300M runs B beside A's long buffer, marked or not.** In every arm B starts within 0.44
  ms, about 49.5 ms before L0 ends, so the marks change nothing measured here.
  - B's GPU run grows from 0.05 to 0.09 ms alone to 0.36 to 0.89 ms. So B shares the GPU with A's
    running kernel: concurrent execution, not a switch.
  - L0 runs 7.5 to 7.95 ms past L1 to L3 in every arm alike. No round left B out, so B's share of
    that is not measured. On the 630, the first buffer after the rounds' 300 ms idle runs 1.8 ms
    long with no B beside it.

What our stack does:
- **The marks.** The bundle keeps each setter's value and answers it back from the getters
  (`RgfxCommandObjects.mm`). A new queue there reads GPU priority 1 and background priority 2, as
  all three Macs' do (`test_queue_priorities_read_back_as_metals`). The bundle sends the host no
  priority: a `QUEUE_CREATE` carries 0 for every queue (design.md, "Rings"). The wire's comment on
  that field, `RgfxQueueCreate.priority` in `reims_graphics_wire.h`, read "0 normal and higher
  higher", the opposite of the order the 630 keeps, where a lower GPU priority value runs first and
  a new queue's value is 1. It now says that no side acts on the field, and names this order.
- **The order.** The runtime runs every queue's work on one `VkQueue` in submission order
  (design.md, "A command buffer in parts"). So B would start only once L0 to L3 had all run, in
  every arm: the 630's answer for queues of one GPU priority. With the guest's marks, a
  compositor's buffer committed behind the service's long buffers starts at once on both GPUs. On
  the 630 that is because its queue's priority value is below theirs; on the 5300M it happens
  whatever the marks.

## Not measured

- Cross-process lock waits for pending GPU work: rgk-lock's `lock_probe` (x86 matches the M3,
  except that a read-write lock does not wait for a buffer that only reads the surface).
- Page faults: no probe made a GPU address fault, on a Mac whose Intel GPU drives the display.
- Surfaces whose rows miss the host's alignment.
- A no-copy buffer past the M3's `maxBufferLength`, 41.7 GB there: the M3 is used little, and
  the x86 Macs' answers bound the lane's.
