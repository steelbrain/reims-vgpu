# Accuracy ledger

A ledger, not a journal: rows are edited in place. It lists where the Vulkan platform's answers
can differ from Metal's. The platform (`RtPlatform`, over metal2vulkan-runtime) has no switches: no
fast or exact form to choose, and no policy by device. What differs is of four kinds:
- **the device's own answers**: values the host's GPU computes (its maths, sampling, conversions,
  blends and resolves), which the contract makes the device's own (AGENTS.md, "The contract"),
  each with its bound where one was measured;
- **what the platform has not built, and what rt refuses**: answered as Metal's call answers
  where it fails, and counted;
- **memory whose GPU writes Metal drops**, which the platform makes over a copy;
- **the loss of the device**, which rt never recovers from.

The process ends only where Metal ends it, or where the platform has no form for a raw value the
guest sends (the Vulkan crate's `answer`, `metal_reads_near_null` and `unrepresentable`).
`REIMS_VGPU_STRICT=1` ends it where the host runs out of memory, mappings, files or commit charge,
where by default the platform answers as Metal's call answers when memory runs out; the Vulkan
crate's tests run strict.

## The device's own answers

Where the device computes a value, the value is the device's. The bounds below were measured
before the platform ran on rt, on the drivers each names, against Metal's answers; none has been
measured again on rt, except a row that names its rt run.

| The device's answer | Bound against Metal's |
|---|---|
| The device's sampler (filter, footprint, level, anisotropy, sRGB, comparisons) | linear filter 1 code, sRGB 1.83; the rest unmeasured |
| The device's conversion of a texel's codes | 1 f32 ulp (8/16-bit), 0.18 code (5/6-bit packed) |
| The device's fixed-function conversion of a normalized vertex attribute's codes (`UCharNormalized` through `Short4Normalized`, `UChar4Normalized_BGRA`, `Int1010102Normalized`, `UInt1010102Normalized`) | 1 f32 ulp at some codes on a driver whose conversion multiplies by the reciprocal; Metal's and Vulkan's rule is the correctly rounded quotient |
| Device maths library, division, fma | Vulkan's precision table (e.g. Atan 4096 ulp) against Metal's 4-6 |
| Device NaN bits; device min/max/clamp | NaN bits, ±0 and denormal picks; on ANV and RADV a `half` read from its bits widens by the device's conversion, every value but a NaN exact, a NaN's bits the device's (ANV quiets a signalling one) |
| Device `dot`, `mix`, `fast::sqrt` | dot (3n−1)·2^-24·Σ, mix 2^-21, sqrt 4.5 ulp |
| Device half rounding, widening and folding; a half texel's NaN | AMD and Intel (RADV, ANV) round a computed half through an f32→f16→f32 conversion pair under Vulkan's 16-bit float controls (round to nearest even, denormals and signed zeros kept), which the spec makes exact (0 mismatches outside NaN inputs over every half and every half +, −, × pair on Iris Xe, the 780M and the 7900 XTX). The RTX 5080 rounds exactly (8192 of 8192 narrowed sums are Metal's). The RTX 3090, on the same 615.71.09 driver, and lavapipe drop the rounding: a half result keeps float precision, its differences compounding across operations. A value past half's range stays finite until a half store saturates it to infinity. Elsewhere 1 half ulp. NaN bits; half denormals may flush |
| The device's blend of a safe-math half output into a blended 16-bit float attachment | at the largest operand: 16-bit float 2.5 half ulps on NVIDIA (blend-colour factors), 2 on AMD, Intel and lavapipe; 32-bit float 1 ulp on NVIDIA, 2 elsewhere |
| The device's decode of a 16-bit float texel (`R16Float`, `RG16Float`, `RGBA16Float`, `RGBX16Float`) that a `half` or `float` read answers | a NaN texel reads as the decode's NaN, where Metal hands back the texel's own code; RADV's reads keep the texel's bits; every other value the texel's |
| The device's lines at width 1, which cover the pixel the driver chooses where a line runs through a vertex two pixels' diamonds share, as Vulkan's open diamond leaves it | such a pixel, and the rest of a line that runs along such vertices, a row or a column from Metal's lower pixel: NVIDIA covers the other pixel, lavapipe leaves out a strip's first pixel where it starts on one; RADV and ANV not measured |
| The device's `AVERAGE` resolve of a multisampled colour attachment whose format has Metal's resolve rule | at 2 and 4 samples a unorm, snorm or packed 16-bit channel within one code of Metal's and an sRGB one within two (NVIDIA at 4 samples; one elsewhere) on RADV, ANV, NVIDIA and lavapipe, each rounding an 8-bit tie down; a float channel in its last bits, unbounded; wholly at a NaN or infinite sample |
| The device's copy of 32-bit float depth values into a `Depth32Float` texture or `Depth32Float_Stencil8`'s depth (`replaceRegion:`, a Managed texture's upload, a blit copy from a buffer), rt's static note `PrecisionNote::RestrictedDepthUpload` | within [0, 1] exact, but a device may store a denormal or -0.0 as +0.0; outside [0, 1] the device's: RADV and ANV (Mesa 26.2.6, the 780M and Iris Xe) clamp to [0, 1] and store a NaN as +0.0, NVIDIA (620.70.06, the RTX 3090) keeps every bit, and a comparison reads what was stored; Metal keeps all 16 probe values bit for bit and compares them unclamped (M3 Max, Radeon Pro 5300M, UHD 630); measured on rt by metal2vulkan at 64be1d50ae |

## What the platform has not built, and what rt refuses

The platform's pieces are built one by one. A call that reaches a piece not built yet answers as
Metal's call answers where it fails (a nil object, a failed creation with an `NSError`, zeros
from a getter, a command that encodes nothing), counted under `rt-unbuilt-<piece>`, the process's
first of each piece with a line on stderr naming the call (`crates/reims-vgpu-vulkan/src/rt_platform/unbuilt.rs`,
`Unbuilt::ALL`). A boot's degradation counts line lists every unbuilt piece it reached, with how
many times. The pieces, by code:

| Code | What is not built |
|---|---|
| `rt-unbuilt-buffer` | a Private buffer made with contents (`newBufferWithDescriptor:` with bytes): nil |
| `rt-unbuilt-iosurface-texture` | a texture over a plane of an IOSurface of a format that reads beside its plane (a two-plane YCbCr or XR10 format over a surface of two planes or more): nil |
| `rt-unbuilt-shared-texture` | shared textures and their handles (`newSharedTextureWithDescriptor:`): nil |
| `rt-unbuilt-heap` | sparse heaps: nil |
| `rt-unbuilt-mesh-pipeline` | mesh render pipeline states: nil and an `NSError` |
| `rt-unbuilt-render-command` | indexed patch draws (direct and indirect), mesh draws, tile dispatches, resolve textures, and private settings other than Metal's default: nothing encoded, a line width other than 1 drawing later lines 1 pixel wide |
| `rt-unbuilt-compute-command` | an indirect `dispatchThreads`: nothing encoded |
| `rt-unbuilt-predication` | command predication: a start encodes nothing, an end answers NO |
| `rt-unbuilt-protection` | protection options other than 0: the command buffer runs as one with none |
| `rt-unbuilt-fill-texture` | `fillTexture:level:slice:region:bytes:length:`: nothing encoded |
| `rt-unbuilt-blit` | a texture fill with a colour, a buffer fill with a 4-byte pattern of more than one byte value, copies between textures with blit options, and copies between a buffer and a texture with blit options other than one plane of depth and stencil: nothing encoded |
| `rt-unbuilt-render-pass` | imageblock and threadgroup memory lengths, tile sizes and sample positions of a render pass: the pass renders without them. rt's pass renders its samples at Vulkan's standard locations, which are Metal's default positions of 1, 2 and 4 samples (measured on the M3 Max, macOS 27, `guest/bundle/tools/samplepos_probe.m`), so a set of positions equal to Metal's defaults of their count renders as Metal's does and counts nothing; a set that differs from them, and a set of any other count, whose defaults are not measured, is counted, the process's first line of the code, where such a set writes it, naming the set's count and each position (`rt_platform/descriptor.rs`) |
| `rt-unbuilt-rate-map` | rasterization rate maps: a layer's creation answers nil. The device answers no layer count on rt and offers no `RATE_MAPS` (`rt_platform::answers`, `device_info::makes_rate_maps`), so a guest's map is nil in the guest and reaches no create |
| `rt-unbuilt-icb` | what of indirect command buffers rt does not build (`rt_platform/icb.rs`): commands the CPU encodes (`indirectRenderCommandAtIndex:`, `indirectComputeCommandAtIndex:`), nil; a CPU `resetWithRange:`, which resets nothing; a blit's copy between buffers (`copyIndirectCommandBuffer:...`), nothing encoded; a descriptor of a resource index, nil; and the reservation of their resource indices (type 5), nothing reserved. The buffers themselves, their GPU-encoded commands, executions and blit resets are rt's |
| `rt-unbuilt-dynamic-library` | dynamic libraries (`newDynamicLibrary:error:`): nil |
| `rt-unbuilt-resource-type` | a reservation of resource indices of a type other than 0, 1 and 5: nothing reserved |
| `rt-unbuilt-tessellation-factor-scale` | a tessellation factor scale other than 1: patch draws read the factors unscaled |

A call of a built piece that rt refuses (a limit, Metal's own refusal, a gap of rt's, a failed
Vulkan call) answers the same way, counted under the error's kind, `rt-refused-<kind>`
(`RtRefusal::ALL`: `loader`, `below-floor`, `vulkan`, `metal`, `ours`, `limit`, `no-such-block`,
`thread`, `translation`, `gap`, `quiesced`, `no-device`, `other`), its first line naming the call
and rt's error. A texture rt makes none of for its pixel format is counted under its format's
code instead, `rt-refused-texture-format-<n>`; a pipeline state's imageblock length rt refuses,
`rt-refused-imageblock-length`; and a heap texture rt refuses on its place in its block,
`rt-refused-heap-texture-place`. rt's refusals (`Error::Refused`) are counted by the basis rt
declares for each (`Error::basis`, metal2vulkan-runtime's `error/basis.rs`;
`RtRefusal::of_basis`): `rt-refused-metal` counts those of Metal's own rules, documented or
measured, with the platform's own refusals of Metal's, and a heap texture's place that Metal's
own rule refuses too. No refusal of rt's declares the device's basis, and rt's limits and gaps
keep their own codes whatever their basis. `rt-refused-ours` counts those rt declares its own
(a gap, cap or deadline of rt's, or the misuse of an extension of rt's): among them an indirect
argument offset without the arguments' bytes after it (`Refusal::IndirectArgumentsPastEnd`; an
offset off a multiple of 4 is Metal's documented refusal, `Refusal::IndirectOffset`, counted as
Metal's), a pass attachment's slice past its texture's
(`Refusal::AttachmentPlace`), a buffer offset set on an empty slot (`Refusal::NoBufferAtIndex`),
a Managed buffer's modified range past its end (`Refusal::ModifiedRange`), a wait in a
scheduled handler or on rt's own thread (`Refusal::InScheduledHandler`,
`Refusal::WaitOnOwnThread`), a sampler state placed at an index the guest did not reserve
(`Refusal::IndicesNotGranted`), a visible function table's set of entries past its size
(`Refusal::FunctionTableIndex`), which sets none of them, and a texture buffer not over a buffer
its consumer places (`Refusal::TextureBufferNotOverBuffer`): rt makes a texture buffer only over
a buffer. The
device's (`newTextureWithDescriptor:` of
`MTLTextureTypeTextureBuffer`) is made over a buffer of the host's own, of the texture's bytes
(its row rounded up to the device's `minimumTextureBufferAlignmentForPixelFormat:`) and the
descriptor's storage mode, which the texture holds and which the guest names by no object ID,
GPU address or resource index (`rt_platform/texture.rs`, `make_texture_buffer`), so it no longer
reaches the code. A heap's (`MTLHeap`'s `newTextureWithDescriptor:` of that type) still does:
nil. Among rt's gaps (`rt-refused-gap`),
nothing encoded: a blit copy to or from a texture buffer (`Gap::TextureCopy`), and a copy
between regions of two textures over one buffer whose bytes meet where rt cannot cut the copy
into ordered slabs, or between two textures whose memory meets where one is placed in a heap
(`Gap::OverlappingCopy`). rt makes no texture over a buffer but a 2D texture of one level,
sample and slice, or a texture buffer (`Gap::TextureOverBuffer`), so a linear texture
(`newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:`) of any other shape, a 2D
array among them, answers nil without asking rt, counted nowhere (`rt_platform/buffer.rs`,
"Linear textures"): a `Managed` 2D array on unified memory is two copies there, logged once
(`graphics/decode/managed.rs`). Three forms of a kernel's `[[stage_in]]` are gaps too, each
named by its form in the boot diagnostics' gaps line (`reims-vgpu: rt-diag gaps:`, written
beside every counts line) and in its first refusal's line: a stage-in region read from a buffer
(`setStageInRegionWithIndirectBuffer:indirectBufferOffset:`, `StageIn-IndirectRegion`), which
encodes nothing and leaves the encoder's region the one set before it, or none; and, as the
compute pipeline state is made, nil and an `NSError`, a `[[stage_in]]` attribute the kernel reads
by thread position through a layout of `MTLBufferLayoutStrideDynamic` (`StageIn-DynamicStride`),
and one it reads through an indexed step at a step rate past 1 (`StageIn-IndexedStepRate`),
whose Metal answer is not measured (`rt_platform/compute.rs`, `rt_platform/diag.rs`,
`refusal_name`). A render pipeline of a maximum vertex amplification count above one
is a gap too: nil and an `NSError`. The device answers `maxVertexAmplificationCount` 1
(`rt_platform::answers::VERTEX_AMPLIFICATION_COUNT`; Metal answers 2 on the M3 Max), so
`supportsVertexAmplificationCount:` answers no above one, and a guest that asks first reaches no
create. A compute pipeline whose kernel reads its imageblock's size (an explicit imageblock in a
kernel, rt's `KernelInput` fact `ImageblockSize`) is a gap too: nil and an `NSError` at the
create (`rt-refused-gap`), where Metal makes the state. arm64 guests make one such pipeline
once at login (measured on macOS 15, 26 and 27 boots), with no failure the guest shows. A render pass that resolves into a texture made without
`RenderTarget` usage that rt can resolve into neither in place nor through a resolve image is
refused (`Gap::RenderPass`, `ResolveUsage`) and skipped whole (below), where Metal resolves into a
texture of any usage: a depth or stencil texture, a colour texture of a type other than 2D, 2D
array, cube and cube array or of a format the device's optimal image does not render, a texture
in place over a dma-buf's import, and a view of another pixel format of a texture in place over a
buffer whose linear image takes no attachment usage. A resolve into a texture in place over a
buffer whose linear image takes no attachment usage (every format on the RTX 3090 and on the
Mac's KosmicKrisp, and on every device a buffer that ends at the texture's last texel with its
rows padded, measured by metal2vulkan), or into a view of the texture's own format, is made: the
pass resolves into the texture's resolve image, a one-sample image rt makes at the first such
pass and frees with the texture, and the pass's end copies the render area's texels into the
buffer at the texture's pitch, leaving the bytes past each row's texels as they were. That is a
copy Metal does not make, one a pass (1,440,000 bytes for a 600 x 600 `BGRA8Unorm_sRGB` pass),
counted `rt-resolve-copies` and `rt-resolve-copy-bytes`, with the images at their highest,
`rt-resolve-images-most` and `rt-resolve-image-bytes-most` (`rt_platform/device.rs`, "Its
resolve copies"). It is a copy, not a value difference: the buffer's bytes after the pass are the
resolve's, as in place. The boot diagnostics (`rt_platform/diag.rs`, lines that start
`reims-vgpu: rt-diag`) add what the call asked of rt where its site names it. Their refusals
line counts the translator's refusals by the feature each refused, `Translation-<feature>`
(metal2vulkan's `Feature`'s variant, as `Translation-DivergentInterpolation`), or by its
error's variant where it refused none (`Translation-Cancelled`), all of them one code,
`rt-refused-translation`; the process's first of each has a line naming the feature, what it
needs and the attachment it is at, and no function's or symbol's name (`rt_platform/diag.rs`,
`refusal_name`; `rt_platform/unbuilt.rs`, `refusal_detail`).

A render, compute or blit pass whose encoder the platform makes none of (on rt, a pass it
refuses, as a resolve it builds none of, counted as above) is skipped whole: the decoder drops
the pass's ops up to its end, and the command buffer runs the rest and completes as it would
have, as a Metal app's sends to a nil encoder do nothing and fail no command buffer
(`crates/reims-vgpu/src/graphics/decode/cmdbuf.rs`, `unmade`). The process's first such pass of
each kind has a line. The host's video landing does the same with a picture's copy into its
output's device caches that rt makes no blit encoder for: the copy is skipped, the caches keep
their bytes and the decode's command buffer completes as for any landed picture; an encode's copy
of its source out of the caches fails the frame instead
(`crates/reims-vgpu/src/graphics/video_host.rs`, `Stage::copy_with_caches`).

Among the translator's refusals (`rt-refused-translation`), a fragment function that reads an
interpolant at a sample or an offset (`interpolate_at_sample`, `interpolate_at_offset`) outside
its first block, or after a `discard_fragment` in it, is refused on every device, where Metal
makes it (metal2vulkan's `Feature::DivergentInterpolation`): the render pipeline state's creator
answers its `NSError`. A device's sample and offset reads may read the quad's other invocations,
and RADV's read garbage where those have left the control flow (measured by metal2vulkan in a
1x1 framebuffer, whose quad holds three helper invocations). The same reads in the first block,
before any discard, are made.

Among them too, a render pipeline state that fetches a colour attachment of one of Apple's
extended-range formats (`BGR10_XR`, `BGRA10_XR`, `RGB10A8_2P_XR10` and their `_sRGB` twins), or
blends onto one, is refused on every device, where Metal makes it (`Translation-Format`): Metal's
tile holds such an attachment as `half` through the pass, so a fetch, and a blend after an
earlier draw of the pass wrote it, read that draw's `half`, which the device's attachment, holding
the format's codes, does not keep. A write with blending off is made. rt makes no texture of
those formats either (`rt-refused-texture-format-<n>`). A fragment function whose SIMD-group
operations read, combine or count other lanes is made where the device reports
`shaderQuadControl` (`VK_KHR_shader_quad_control`), its quads run whole, helpers included, as
Metal runs them, and is refused elsewhere (`Translation-SubgroupOperations`): metal2vulkan
measured KosmicKrisp on the M3 Max and the UHD 630's Windows driver without it, and RADV, ANV,
NVIDIA and lavapipe with it.

Among them too, a compute pipeline state whose kernel reads a visible function table through an
argument buffer is refused on every device, where Metal makes it: nil and an `NSError` at the
create. A kernel that takes its table at a buffer index is made, and its handles and tables are
rt's (`rt_platform/function_table.rs`); a table's `allocatedSize` is its words' bytes, 4 for
each of its two header words and each entry, in pages of 4 KiB (4096 up to 1022 entries), where
the AMD Radeon Pro 5300M and the Intel UHD Graphics 630 count about 8 bytes an entry
(`docs/graphics-kext/dgpu-oracle.md` Q29). Render pipelines take no tables: the device
answers `supportsFunctionPointersFromRender` NO, as the two Macs do.

A pipeline create whose driver's create fails after its reply is answered made, where Metal's
synchronous creator answers the failure as the create's error. rt's creators answer before the
driver's create of the state's main pipelines, which the state's finish (rt's `finish_pipeline`)
then makes on the device's compile thread, before the pipeline is stored
(`crates/reims-vgpu/src/graphics/compile.rs`, "Order"); Vulkan lets that create fail only for
exhausted host or device memory, or with `VK_ERROR_UNKNOWN` or `VK_ERROR_VALIDATION_FAILED`. The
failure is counted, `pipeline-finish-failed`, the process's first with a line at warn, and no
answer of the create's key is kept for the next runs; the pipeline is stored as made, a draw or
dispatch that needs the failed pipeline makes it again at its encode, and where that make fails
too the platform answers the encode as one it refuses.

## Memory whose GPU writes Metal drops

| Entry | What runs instead of Metal's answer |
|---|---|
| `read-only-pages-written` | Memory whose GPU writes Metal drops (pages a guest process maps read-only, `RGFX_SURFACE_PAGES_READ_ONLY`, and a no-copy buffer or heap of `MTLResourceOptions` bit 24) is made over a copy of its pages, and the pages are never written, so the CPU and the command buffers committed after a GPU write read the pages' bytes, as Metal's GPU drops a kernel's and a blit's writes there (the M3 Max, the 5300M and the UHD 630, dgpu-oracle.md Q38). Where Metal drops a write at the write, the copy keeps it for a while. A copy of pages that may change is written whole from the pages at each commit that uses it, so within a command buffer, and between command buffers in flight at once, a read after the GPU's write sees the write until the next commit that uses the copy. A copy of unchanging pages (`RGFX_SURFACE_PAGES_UNCHANGING`, a file's own on a volume mounted read-only, anonymous memory with a copy pending, or anonymous memory its process's mapping alone holds) is never copied in, and is written whole again from the pages at each completion of a command buffer that can write it, so a read after the write sees it until that completion (`crates/reims-vgpu-vulkan/src/rt_platform/copied.rs`, "Writes dropped" and "Unchanging pages"). Anonymous memory its mapping alone holds can still take a write another driver started before the app removed write for good, landing after the block was made: the GPU then reads the bytes the pages had as the copy was made, or at the last completion that wrote it again, where Metal's GPU reads the write. The bundle's reading of such memory was checked against the published kernel source through macOS 26.5; later releases are checked by test (`guest/bundle/tools/sole_holder_probe.c`, which the guests' boot checks run). A render pass with an attachment over such a copy completes, its stores and resolves there dropped with the copy's other writes, as over read-only pages on the 5300M and the UHD 630, where the M3 Max faults the command buffer of a pass that stores or resolves into it (`PageFault`): counted at the pass, `rt-copied-store-outs`. A no-copy buffer or heap of bit 24 pinned in a known block that is no such copy stays at the guest's address, where the GPU's writes reach its bytes, counted `rt-no-copy-read-only`, the process's first with a line (`crates/reims-vgpu-vulkan/src/rt_platform/buffer.rs`, "Read-only memory") |

## The loss of the device

| What runs instead of Metal's answer | Where |
|---|---|
| The loss of rt's `VkDevice` is never recovered from, where Metal's device survives its GPU's recovery: rt makes its device once (`crates/reims-vgpu-vulkan/src/rt_platform/swapchain.rs`, "No device recovery"). Of the work the loss ends, the earliest submitted command buffer rt ends in flight fails as the hang, `Timeout` (2), once for the device, and every other as a victim, `Internal` (1), never `DeviceRemoved` (11) (`crates/reims-vgpu-vulkan/src/rt_platform/command.rs`). Every later commit ends at once, `Internal` (1), its handlers run (metal2vulkan-runtime's `CommandBuffer::commit`), and a wait for a shared event's value that lost work owed answers false (its `SharedEvent::wait_until_signaled_value`). rt checks for no loss where it makes an object: a new one is made where the driver's Vulkan calls make it, and any call rt refuses after the loss answers as Metal's call answers where it fails, a nil object or nothing, counted under `rt-refused-<kind>` (`crates/reims-vgpu-vulkan/src/rt_platform/unbuilt.rs`); none of its GPU work runs. The window presents no frame after the loss: a frame whose command buffer did not run is never presented (`RtSwapchain::present`), and a take the lost device fails ends its presenter's draws, a new presenter made at most 3 times in a row before the window stays as it is (`crates/reims-vgpu/src/graphics/rt_window.rs`, `crates/reims-vgpu-window/src/present.rs`). The guest's presents are still answered at the display's refreshes, a failed command buffer finishing as a completed one does (`crates/reims-vgpu/src/graphics.rs`, `Presented`; `crates/reims-vgpu/src/graphics/present.rs`). Nothing on the platform ends the process at the loss or after it: the guest runs on, every command buffer failing, until the VM ends. The first command buffer that fails with the device lost writes one line on stderr, once for the device, naming the GPU (its name, PCI vendor and device IDs) and how many command buffers were committed and not complete then, and each command buffer that fails so is counted, `rt-device-lost` (`crates/reims-vgpu-vulkan/src/rt_platform/command.rs`, "A lost device's line"); rt itself writes nothing at a loss. Lines of the driver's report of the fault follow it, where rt read one (`VK_EXT_device_fault`): each address it names, placed among the decoders' live objects and the blocks the host sees, with the nearest below and above where none holds it (`crates/reims-vgpu-vulkan/src/rt_platform/fault.rs`); the guest sees none of it. Each command buffer rt ends as in flight then writes a line of its own, once however many of its parts it ends so, naming the guest's command buffer and the first compute and render pipeline states its encoders set, by their object ids, or that it is the host's own: the first 8, then one line that the rest are counted only, each counted, `rt-device-lost-in-flight`. What the guest's apps and compositor do with these codes is not measured. Two rt walks lost their device while the platform answered 11 (`DeviceRemoved`) for every command buffer the loss ended: m3-max-53777 on an RTX 3090 and m3-max-54078 on an RX 7900 XTX. In each, the guest's running app logged nothing for its last ~3 minutes until the walk stopped it | device loss |
