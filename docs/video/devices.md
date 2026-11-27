# The graphics device's video

The reims-graphics device (PCI 1234:5247) decodes and encodes the guest's H.264 and HEVC on the
host's hardware. The guest's video driver, in our Metal bundle, is AppleGVA's vendor driver
([docs/graphics-kext/design.md](../graphics-kext/design.md), "Decided", 16); it decodes, and
encodes H.264 (`ENCODE_H264`), sending no `ENCODE_H265` yet. This file is the host's side: what
it answers the guest, and what a decode and an encode do. The wire's records and their
fields are documented where they are defined: `crates/reims-vgpu-graphics/src/wire.rs` (the caps and
the session) and `crates/reims-vgpu-graphics/src/ops/video.rs` (the ops).

## Which hardware

On Linux, the device's video runs on the first of the host's two video arms that takes the
runtime device (reims-vgpu's `graphics/video_choice.rs`):

1. NVIDIA's, decoding on NVDEC and encoding on NVENC, where the device is an NVIDIA GPU that
   CUDA opens (`graphics/nvidia.rs`, [nvidia.md](nvidia.md)).
2. VA-API, on the device's render node, whatever its vendor (`graphics/vaapi.rs`,
   [vaapi.md](vaapi.md)).

On Windows the device's video runs on VA-API alone, through libva-win32 on the DXGI adapter of
the runtime device's GPU (the LUID Vulkan answers for it) and Mesa's VA-API driver on Direct3D 12,
`vaon12`, which reims-ctl's install takes from MSYS2's CLANG64 packages. Windows makes no
dma-buf: its pictures decode into shared Direct3D 12 textures, which the runtime device's GPU
copies into the guest's planes, or their device caches; where the device imports none, they come
back through `vaGetImage` and the CPU's copy ([vaapi.md](vaapi.md), "Loading libva"). On Intel's UHD 630 (driver 31.0.101.2145)
`vaon12` decodes H.264 and HEVC Main and Main 10 and encodes nothing. The Windows arm answers no
encoder on any GPU, whatever `vaon12` encodes (`open_on_rt` in
`crates/reims-vgpu/src/graphics/vaapi.rs` opens with `encodes: false`): rt imports no shared
Direct3D 12 texture, which an encoder's input would be written through. So on a Windows host the
device lists no encoder, and the guest encodes in software, on every GPU.

Each arm that declines says why in one line at info. A Mac has no video arm, on either platform.
Where no arm takes the device, the host writes no video caps, and the guest is offered no
codec: the kext names AppleGVA no video driver (`publishVideoCodecs`,
`guest/kext/src/RgfxAccelerator.cpp`), so VideoToolbox answers that no codec decodes in hardware
(`VTIsHardwareDecodeSupported`), lists no hardware encoder, refuses sessions that require
hardware (`kVTCouldNotFindVideoDecoderErr`, `kVTCouldNotFindVideoEncoderErr`), and decodes and
encodes every other session in software, as on a Mac without hardware video.

An arm64 guest's hardware video takes another route: VideoToolbox's paravirtual path, over the
arm64e kext ([design.md](../graphics-kext/design.md), "Decided", 17). The arm64e kext names
AppleGVA no video driver; where the guest took `RGFX_FEATURE_VT_MESSAGES` and the caps list a
decoder, it publishes VideoToolbox's paravirtualization service, and the guest's VideoToolbox
registers each decoder and encoder the host lists, running each session as `VT_MESSAGE`s on
ring 0. The host lists a decoder for each H.264 or HEVC decode entry of the caps (`vt_features`
and the decoder list beside it in `graphics.rs`), and an encoder for each codec its arm encodes a
profile of (`guest_encoders` in `graphics/vaapi.rs` and `graphics/nvidia.rs`). So on a Linux host
an arm64 guest decodes H.264 and HEVC and encodes H.264 and HEVC Main and Main 10 in hardware; on
a Windows host it decodes alone, as the Windows arm answers no encoder. Its decoders land 4:2:0
pictures, and HEVC's 4:4:4 where the arm's decoder decodes it (`guest_declines`): NVDEC's, and
VA-API's Main 4:4:4 at 8 bits where the driver has one.

Where the host lists no decoder, the kext publishes no service, and the guest's VideoToolbox
answers as above for decoders. Its encoders are then the OS's own: arm64 macOS 27.0.1 lists no
hardware encoder and refuses a session that requires one (−12908), while arm64 macOS 26 lists
Apple's AVE encoders (`com.apple.videotoolbox.videoencoder.ave.avc` and `.ave.hevc`) as
hardware, which have no encoder behind them in a VM, and fails such a session −12903, with the
kexts' video keys or without them (measured under TCG on a 780M host, 2026-11-16). The route
AppleGVA's vendor driver serves is x86-64's: with its keys an arm64 macOS 14.6.1 guest on a
VA-API host answered `VTIsHardwareDecodeSupported` true for HEVC, while a session requiring a
hardware HEVC decoder failed −17692, one requiring H.264 −12906, and no session reached the
host (measured 2026-11-16).

## The caps

At the device's init, the host writes an `RgfxVideoCaps` at `RGFX_VIDEO_CAPS_OFFSET`, after
the device info (`GraphicsDevice::with_platform`, in `graphics.rs`):

- **The entries**: one `RgfxVideoCodecCaps` for each codec and direction the arm takes: its
  profiles, its highest level, its smallest and largest coded pictures, its DPB and reference
  bounds, its output formats, and the alignment at which it reads an access unit in place.
  Each answer is the arm's video API's own (the arms' docs), but for VA-API's largest H.264
  size, kept under the 65,535 macroblocks its drivers decode: radeonsi's 4096x4096 and iHD's
  8192x8192 are answered as 4096x4080 ([bugs.md](bugs.md), U28).
- **The flags**: `RGFX_VIDEO_CAPS_OUTPUT_CACHED` where the kext makes device caches, of every
  surface it asks one for (`RGFX_FEATURE_SURFACE_CACHE`, every host now, within a budget) or of
  those whose textures would not lie in place in their pages
  (`RGFX_FEATURE_SURFACE_CACHE_REFUSED`, which no host offers now), and the arm lands pictures in
  them: NVDEC by its own GPU
  work, where CUDA imports the device's exportable buffers ([nvidia.md](nvidia.md)), the
  device's caches of surfaces of two planes (the outputs' NV12 and P010, as the layout a cached
  block's `SURFACE_PAGES` carries gives them) then made exportable as their memory is made (in
  the per-shape form at its first use, such as a decode that lands in the cache's buffer, so that
  a cache whose pictures all land in its surface may hold none), and every other cache made as
  any buffer; VA-API by the runtime device's copy
  of the decode surface, or the CPU's through the session's staging ([vaapi.md](vaapi.md)).
  The guest's video driver then asks its output surfaces cached
  (`guest/bundle/src/RgfxMetalDevice+Video.mm`), with their layout, as for any other surface,
  and on a GPU that shares the CPU's memory the device keeps a cache of those it answers so;
  elsewhere it asks them uncached. A cache made for a surface whose first plane starts past its
  first byte, as CoreVideo's 32 bytes into their surface, has the platform place that byte where
  its textures over the cache's memory may start in place (on the Vulkan platform a multiple of
  256 bytes, `BufferDescriptor::with_aligned_byte`), so textures over the planes lie in the cache
  rather than in the runtime's shadows (`surface_cache::aligned_byte`): from the layout the
  kext's ask gives, or on a GPU of its own memory the layout a cached or deferred block's
  `SURFACE_PAGES` carries, the exportable caches NVDEC lands its pictures in among them. That
  holds where the runtime binds images in the cache's memory: on the RTX 5080 metal2vulkan makes
  each exportable buffer its own dedicated allocation and binds no image there, so the planes'
  textures over NVDEC's caches stay its shadows wherever the byte lies (measured 2026-11-11).
- **The layouts**: an `RgfxVideoOutputLayout` per entry, which says how a surface is laid out
  to take a picture's whole coded rows and columns in place. An arm that lands every picture by
  a copy of its crop answers zero, and every entry's is zero where the flag is set, since a
  picture lands in place only in a surface's pages. For an encode entry it would say how a
  source is laid out to be read in place; both arms read every source by a copy, and answer
  zero. The guest's video driver answers it in AppleGVA's GetPixelBufferInfo, which does not
  lay out the surfaces AppleGVA decodes into: their rows and plane offsets are CoreVideo's own
  (a 1080p output's rows are 1920 bytes, its first plane 32 bytes in, whatever the answer), and
  AppleGVA's HEVC decoders do not ask (measured on an x86-64 macOS 26.7.2 guest, 2026-11-21:
  `guest/bundle/src/RgfxVideoDriver.cpp`, `PixelRequirement`).
- **What an encode entry takes**: an `RgfxVideoEncodeCaps` per entry (`encode`): the rate
  controls (a bit each of CQP, CBR and VBR, as the arm's API lists them for the codec's
  profiles) and the most consecutive B pictures, 0 on both arms, whose encoders keep frames in
  display order.
- **What a decode entry bounds**: an `RgfxVideoDecodeCaps` per entry (`decode`): the narrowest
  HEVC tile column the arm decodes, in luma samples, 128 on both arms
  (`reims_vgpu_video::codec::hevc::MIN_TILE_COLUMN_WIDTH`); zero, no bound, in every other
  entry. An HEVC decode entry's `max_references` is 8 on both arms: the most current
  references with which every picture decodes ([vaapi.md](vaapi.md), "What it refuses";
  [nvidia.md](nvidia.md)).

The guest's video driver checks an HEVC picture against these before it schedules it
(`hevcTaken`, `guest/bundle/src/RgfxVideoDriver.cpp`): a picture whose coded size is outside its
entry's, whose PPS has a tile column narrower than `decode`'s, or whose first slice segment's
header names more current references than `max_references` fails its schedule, and VideoToolbox
reports `kVTVideoDecoderMalfunctionErr` for it and every picture after it. Sent, the host would
refuse it after it took it, and the schedule would fail only once the host's verdict came back.
AppleGVA's H.264 decoder delivers a picture whose schedule failed, so an H.264 decode is checked
at its context's create (`sizeTaken`, `CreateContexts`): a coded size outside its entry's fails
the create, and VideoToolbox decodes the stream in software. What the host refuses only from a
picture's parameter sets or slices still reaches the guest undecoded ([bugs.md](bugs.md), D25).

## Sessions

`VIDEO_SESSION_CREATE` names a codec and a direction. The arm makes a session where it has an
entry for them. Elsewhere the create is refused, with a warning that names the codec and the
direction. Each session has a shared event, which the host signals.

## A decode

A decode is an op (`DECODE_H264` 0x4000, `DECODE_H265` 0x4001) in a video encoder of a
command buffer of the session's process (`graphics/decode/video.rs`):

- **The op's objects** are resolved as the command buffer is taken: the bitstream buffer, and
  the buffers the output surface's planes lie in. An op faults the command buffer where it is
  of another codec than its session's, where its parameter sets run past its end or its access
  unit past its buffer, and where it names no texture over a plane.
- **The work starts at once**, after the session's earlier work, on the host's thread that
  takes the command buffer. It does not wait for the buffer's earlier encoders, since the
  guest's video driver encodes none before it. It does wait, on the CPU, for the buffers of
  other queues the command buffer runs after (its `SURFACE_ORDER`), so a decode whose output
  another process still reads lands once that read is done. And it waits, on the CPU, for the
  host's own work on an output's device cache, its last fill and a copy of its last picture into
  the pages still in flight, as long as that takes, as a Mac's video work waits for the GPU's
  earlier work on its surface: a GPU slowed by other work slows the picture and fails nothing.
  Each ends as the platform sees its command buffer complete, failed where the device was lost;
  only a reset of the device ends the wait sooner, faulting the op. A wait past 2 s is named in
  the log.
- **The access unit is read in place**, in the guest's buffer, where it starts and ends at the
  entry's `bitstream_alignment`; elsewhere the arm copies it. The parameter sets the op carries
  are handed to the decoder before the access unit.
- **The picture lands** in the output surface's planes, cropped: in the surface's pages, or in
  its device cache where the kext keeps the surface in one. The arms' ways differ (below).
  A picture short of the planes (a stream smaller than the guest's surfaces) lands with zeros in
  the rest of each plane's crop, on every way: the CPU's copy zeros past what it copies, the
  session zeros the rest of a plane in host bytes before the hardware's write, and clears a plane
  in a device cache whole first on its queue (a render pass of the plane's image in a cache's
  surface, a fill of its bytes in a cache's buffer), and on Windows the draw into a cache's
  surface over Direct3D 11 textures clears them first. No texel past the picture keeps the
  plane's earlier bytes, another picture's or a staging buffer's. A VA-API decode straight into
  the output's cache takes only a coded picture that covers the planes; a shorter one is decoded
  elsewhere and lands as above (reims-vgpu's `graphics/video_host.rs`, "A short picture").
  A new output in a device cache takes no fill before its first decode, which so waits for
  none: its planes read zeros wherever no picture lands in them, the session clearing them
  where a picture fails to land, and the device filling the cache from its pages where the
  decode fails or answers no picture.
- **The command buffer's later work, and its completion, wait** for the session's event to
  reach the value the decode signals once its picture has landed. So a CPU lock of the output
  waits for the picture, as for any GPU write.
- **A command buffer of a decode alone into device caches completes at the decode**, where the
  VA-API arm copies the picture into the caches by its video processing on the GPU, or on Windows
  draws it into a cache's surface over Direct3D 11 textures: its `CB_COMPLETE` goes as the decode
  is done and that copy is issued, or before that draw, in its queue's order, with GPU times of 0,
  so the guest's decoder hands the picture on while the copy or the draw runs. The rest stays at
  the buffer's end, after the copy or the draw: its later work, its surface timeline and the
  caches' copy into the pages. Every reader of the output that the guest's writer slots, emptied
  at the `CB_COMPLETE`, no longer point at the decode waits for that end all the same: a command
  buffer of another queue that uses the output (a blit or copy from it among them), on the GPU,
  or on the CPU for a decode; a CPU lock's download; and a present (reims-vgpu's
  `graphics/decode/video.rs`, "Completed at the decode").
- **A picture that fails to land** still signals its value, and the command buffer completes
  failed, with Metal's internal error. The guest's video driver answers that as a decode error:
  an HEVC picture's schedule waits for its verdict and fails (`kVTVideoDecoderMalfunctionErr`,
  design.md, "A decode the driver cannot complete is a decode error"); an H.264 picture reaches
  the guest as decoded, as AppleGVA's H.264 decoder reports no failure, so the driver refuses what
  it can tell before a picture, its size, at the create ([bugs.md](bugs.md), D25). A picture
  whose copy or draw into its output's caches fails after its command buffer completed at the
  decode (above) fails the session's next decode in its stead, whose command buffer completes
  failed at its end: the guest is told of the failure one picture late, counted as a degradation
  (`video-landing-failure-carried`).
- **A picture whose copy into its output's caches the runtime makes no blit encoder for**
  (metal2vulkan-runtime's refusal, which it counts under `rt-refused-<kind>`) lands with the
  caches as they were: the copy is skipped, as the decoder skips whole a pass the platform makes
  no encoder of, and the command buffer completes as for any landed picture. The process's first
  is logged. An encode whose source's copy out of its caches the runtime makes no blit encoder
  for fails its frame instead (`RGFX_VIDEO_ENCODE_FAILED`), since the encoder would read the
  staging buffer's earlier bytes.
- **A failed picture's output holds what the driver wrote** where the driver writes the output
  itself: a picture decoded into its output's cache, and one the driver's video processing
  copies there or into the pages, which on VA-API is issued right behind its decode and copies
  the decode surface whatever the decode answers (the VA crate's `zero_copy.rs`), the rt
  platform's references among them. So does a picture NVDEC answers failed (below). Where the
  runtime device's GPU or the CPU copies a picture, VA-API's last two ways, each waits for the
  decode first, and a failed picture is not copied: its output holds what it held before.
- **A picture NVDEC answers failed**, H.264's or HEVC's, lands as NVDEC left it, its command
  buffer completing as any, and the pictures after it decode from it, as a Mac's hardware
  decoders hand such a picture on with no error ([nvidia.md](nvidia.md), "A picture NVDEC
  failed lands as NVDEC left it").
- **A picture the stream's random access skips, or one of the stream's that does not decode,**
  lands nothing, and its command buffer completes as any other, its output keeping its bytes.
  One that does not decode is one whose bytes do not read as its standard reads them, one that
  predicts from a picture the decoder does not hold (a stream's first pictures, where it starts
  at a P picture), or, in an HEVC stream, one after a picture that did not decode, before the
  next IRAP picture, from which the session decodes again (reims-vgpu's
  `graphics/video_host.rs`, "A picture of the stream that does not decode"). An H.264 picture
  after one that did not decode decodes and lands as any, as a Mac's hardware decoder answers
  it with no error and an image, concealed, until the next IDR picture (**measured** with
  VideoToolbox on a MacBook Pro of 2019, macOS 26.7.1, potluck job m3-max-60715). A Mac's
  hardware decoders answer a picture that does not decode alone and decode on too, H.264's with
  no error and HEVC's with `kVTVideoDecoderBadDataErr` or `kVTVideoDecoderReferenceMissingErr`
  and no image (**measured** with VideoToolbox on a MacBook Pro of 2019, macOS 26.7.1,
  2026-11-20). In the guest, AppleGVA hands the device every H.264 picture, and the client gets
  each one that does not decode with no error and the output's earlier bytes; it answers an
  HEVC stream started at a P picture itself, as a Mac does, `kVTVideoDecoderBadDataErr` and no
  image for each picture until the IRAP picture, and none of them reaches the device
  (**measured** in the guest on a Radeon 780M, 2026-11-20).
  A stream the decoder does not take, an `avcC` or `hvcC` record's error, and the hardware's or
  the host API's errors still fault the command buffer. The range extensions' streams, HEVC's
  4:4:4 at 8 bits among them, are streams this decoder does not take: the caps offer 4:2:0
  alone, and its decoders land 4:2:0 pictures alone, where VideoToolbox's paravirtual decoders
  on arm64 guests land HEVC 4:4:4 at 8 bits on hosts that decode it
  ([design.md](../graphics-kext/design.md), "The answers are the host's arms'").
- **A Mac's picture of an H.264 stream started at a P picture** is a decode against a
  reference of zeros, the picture's residuals added, which shows green. On VideoToolbox on a
  MacBook Pro of 2019 (macOS 26.7.1, its hardware decoder, 2026-11-25, potluck job
  m3-max-74665; **measured**), a 1080p stream of our own with IDR pictures at 0 and 30, started
  at its P picture 5, gave picture 5 a mean Y of 0.78 (min 0, max 89), Cb 3.96 and Cr 5.51,
  picture 6 Y 1.20, Cb 6.62 and Cr 8.70, and picture 29, the last before the IDR picture,
  Y 13.13 (max 253), Cb 48.05 and Cr 43.76, each with no error; the whole stream's picture 5
  has Y 120.30, Cb 130.81 and Cr 130.08. From the IDR picture on, the pictures are the whole
  stream's (picture 30: Y 122.23, Cb 132.28, Cr 129.05). Ours lands nothing in such a
  picture's output (above), so a new output reads zeros, green as well, and an output the
  client reused reads its older bytes, an earlier picture's.

## An encode

An encode is an op (`ENCODE_H264` 0x4002, `ENCODE_H265` 0x4003) in a video encoder of a command
buffer of an encode session's process (reims-vgpu's `graphics/video_host/encode.rs`):

- **The op is checked** as the command buffer is taken: the settings it carries (a session's
  first encode carries them, and each change after), the source's size against them and the
  entry's sizes, and the buffers the access unit and its result go in. A refusal faults the
  command buffer.
- **A change of settings** that makes a new sequence (the size, format, profile, level,
  entropy coder, B pictures, keyframe interval or colour) makes the arm a new encoder, whose
  first access unit is a keyframe;
  a change of the bitrates, QPs or frame rate goes to the encoder from the next frame.
  `min_qp`, `max_qp` and a change of `quality` alone are not taken.
- **The frame is encoded** on the session's encode thread (`reims-vgpu.graphics-video-encode`),
  in the order the encodes were taken, once the command buffer's waits are done. The runtime
  device's GPU copies its two planes, where they lie, in the surface's pages or its device
  caches, into the encoder's own input: on VA-API a buffer of the runtime device's, a dma-buf
  the encoder's display imports as an input surface (on Linux; on Windows the arm answers no
  encoder, as the runtime imports no shared Direct3D 12 texture), or on NVENC an exportable
  buffer CUDA imported, which NVENC registers and maps for the frame. The margin past the
  picture to the coded size is written as the upload writes it. Where the arm
  declines that copy (said once a session), the planes are read as host bytes, in the surface's
  pages, or, for a plane in a device cache, in a staging buffer the GPU copies it into first, and
  the encoder's upload copies them. The encoder codes the same bytes either way.
- **The access unit is written** in Annex-B form into the bitstream buffer, a keyframe's with
  its parameter sets first, and the frame's `RgfxVideoEncodeResult` after it. One past the
  capacity is not written (`RGFX_VIDEO_ENCODE_OVERFLOW`, with the bytes it needed), and the
  next frame is made a keyframe; a frame the hardware fails gets `RGFX_VIDEO_ENCODE_FAILED`.
  Either way the command buffer completes as any other.
- **The command buffer's later work, and its completion, wait** for the session's event to
  reach the value the encode signals once its result is written.

| arm | encoder | its access unit |
|---|---|---|
| NVENC | `NvencEncoder` ([nvidia.md](nvidia.md)); `quality` picks the preset at the high-quality tuning: P1 up to the middle, 50, then P2 to P7, and the keyframe interval is the GOP (`graphics/nvidia/encode.rs`) | Annex B as NVENC writes it, its parameter sets before each IDR picture |
| VA-API | `VideoEncoder` ([vaapi.md](vaapi.md)), on `EncSlice`, else `EncSliceLP`; `quality` maps onto the driver's quality levels (`graphics/vaapi/encode.rs`) | the encoder's length-prefixed sample, written as Annex B, a keyframe's parameter sets first from its configuration record |

## How each arm lands a picture

| arm | access unit in place | a picture lands | layout asked of the guest |
|---|---|---|---|
| NVDEC | at any place and length | by one CUDA copy a plane: into the output's device caches, through CUDA's import of their memory, where the flag is set; into the pages, where CUDA registers their view; else read back and copied by the CPU | none |
| VA-API | at any place and length | decoded into the output's device cache, where the driver's check of decodes into targets passed the codec and format and the cache's memory is a dma-buf (on the rt platform, only a picture no later picture predicts from: a reference is decoded into a surface of the decoder's own); else by the driver's video processing into the pages, or into such a cache through its memory, where the import check passed for the format and the planes meet its alignment (iHD's way into every cache: its decodes fail the check); else by the runtime device's GPU, where the open made a copier: on Linux, for planes in a device cache outside its surface, the driver's video processing writes the picture into a scratch of the session's that lies as the driver's imports must, and the GPU copies the crop from it into the cache (a cache in CoreVideo's layout, which radeonsi's imports refuse, among them); else read back and copied by the CPU. The caches are made dma-bufs where either check passed anything | granularity [2, 2], the import check's alignment |

Both run in reims-vgpu's host video session (`graphics/video_host.rs`, [backends.md](backends.md),
"The traits"): a plane in a device cache the arm does not write itself takes its picture staged
in host bytes, and the session copies it into the cache on a runtime queue. Each session counts
how its pictures landed, and logs the counts at its end ([vaapi.md](vaapi.md), "The
reims-graphics device").
