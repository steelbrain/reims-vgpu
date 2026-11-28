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
ring 0. The host lists a decoder for each H.264, HEVC or VP9 decode entry of the caps
(`vt_features` and the decoder list beside it in `graphics.rs`), an AV1 decoder where the arm
decodes AV1 ("AV1 on arm64 guests", below), and an encoder for each codec its arm encodes a
profile of (`guest_encoders` in `graphics/vaapi.rs` and `graphics/nvidia.rs`). So on a Linux host
an arm64 guest decodes H.264, HEVC and VP9 and encodes H.264 and HEVC Main and Main 10 in
hardware; on a Windows host it decodes alone, as the Windows arm answers no encoder. Its decoders
land 4:2:0 pictures, and HEVC's range extensions, 4:2:0, 4:2:2 and 4:4:4 at 8, 10 and 12 bits,
where the arm's own query lists a decoder of the stream at its size (`guest_declines`; "A stream
outside the host decoder's sizes" and "HEVC's range extensions on arm64 guests", below).

VP9 reaches the host by this route alone. Its caps entry (codec 3, `RGFX_VIDEO_CODEC_VP9`) lists
profile 0, into NV12, where the arm decodes VP9 at 8 bits, and profile 2, into P010, where it
decodes VP9 at 10 (`vp9_caps` in `graphics/vaapi.rs` and `graphics/nvidia.rs`), at the arm's sizes
up to 16384 a side, and VP9's top level as `vpcC` numbers it (62, level 6.2). An x86-64 guest's
kexts read no entry past HEVC's, and the device takes no session of codec 3 ("A decode"). The
guest's VideoToolbox lists the VP9 decoder only once a process calls
`VTRegisterSupplementalVideoDecoderIfAvailable` for `vp09`, as on a Mac, and as browsers do: before
that every VP9 session's create fails −12906 (`kVTCouldNotFindVideoDecoderErr`) in the guest and
none reaches the host; after it a session decodes on the host. arm64 macOS 26.6.2 has no VP9
decoder of its own, and the M3's macOS 27.0.1 none in software: a session that allows no hardware
fails −12906 before and after (**measured** under TCG on an RX 7900 XTX host, and on the M3,
2026-11-27, potluck jobs m3-max-77263 and m3-max-77262). So a VP9 stream the host does not decode,
or any VP9 on a host whose driver lists none, is not decoded in the guest at all. The guest's
VideoToolbox hands a stream of every profile to the host's decoder, though the caps' profiles
name 0 and 2 alone: before the host took profiles 1 and 3, an 8-bit 4:4:4 Profile 1 stream's
session reached the host, whose `dsta` refused it (−12910,
`kVTVideoDecoderUnsupportedDataFormatErr`, whether the session required hardware or allowed it),
and a session that allowed no hardware failed −12906 (**measured** on the Iris Xe host, 2026-11-27,
potluck job m3-max-77382). The host takes profiles 1 and 3, and profile 2 at 12 bits, where its
driver decodes them, and refuses the rest at `dsta` so ("VP9's profiles 1 and 3 on arm64 guests",
below). Each sample, a frame or a superframe whole, is split into its frames on the host,
which decodes them all and lands the one it shows. The guest's pictures of our two libvpx streams
equal vpxdec's, frame by frame, by the IVF route (the probe's own `vpcC`) and the MP4 route
(AVFoundation's), in sessions that require hardware: on an RX 7900 XTX (radeonsi), an Iris Xe
(iHD) and an RTX 3090 (NVDEC) under arm64 macOS 26.6.2, and on the RTX 3090 under macOS 14.6.1,
all under TCG (**measured** 2026-11-27, potluck jobs m3-max-77354, m3-max-77382, m3-max-77355
and m3-max-77373). A VP9 decode runs as fast as HEVC's on the same route, measured with the guest's
CPU emulated (TCG, 4 CPUs): the rates are the emulated guest's, not the GPU's, and at 1080p the
three proof cards' VP9 rates lie within 5% of each other (one run each: 89.1, 90.7 and 93.2 frames
a second on the RX 7900 XTX, the Iris Xe and the RTX 3090, against HEVC's 79.9, 90.9 and 94.2, in
jobs m3-max-77354, m3-max-77382 and m3-max-77355). Each measure took
sessions that require hardware, VP9's and HEVC's in turns, four runs each, at 1920x1080 (60
frames) and at 8192x4352 (8 frames): the largest picture both codecs' entries take, as the drivers
answer up to 8192x4352 on the AMD cards and 16384x16384 on the Iris Xe for both, and both
entries' level, 6.2, bounds a picture at 35,651,584 luma samples, which 8192x4352 fills. As
medians, VP9's frames a second and seconds of guest CPU against HEVC's: on the RX 7900 XTX, with
the node to itself (potluck job m3-max-77411), 84.4 and 2.85 s against 78.6 and 3.14 s at 1080p,
and 21.2 and 1.55 s against 20.7 and 1.56 s at 8192x4352, the guest CPU's medians there leaving
out one run of each codec that read 0.00 s, as the guest's tick count did not move in its 0.3 s;
and on the Radeon 780M, beside a job of another card's (m3-max-77433), 92.7 and 2.58 s against 94.4
and 2.55 s at 1080p, and 22.6 and 1.43 s against 20.6 and 1.57 s at 8192x4352, leaving out one
HEVC run there that read 0.00 s likewise (**measured** 2026-11-27).

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

### AV1 on arm64 guests

An arm64 guest's VideoToolbox gets an AV1 decoder where the arm's own capability query lists
AV1's Main profile: VA-API's `VAProfileAV1Profile0` decoder (`av1_limits` in the video crate's
`vaapi::va::av1`), or NVDEC's `cudaVideoCodec_AV1` at 4:2:0 (`graphics/nvidia.rs`). Each arm
logs it at info (`hardware video: ...: AV1 for VideoToolbox's decoders, Main at 8 and 10 bits
up to 8192x4352`, radeonsi's on the RX 7900 XTX). The
decoder list names `av01` with profile 0 at level 19, AV1's 6.3, the top level Apple's own AV1
decoder lists (`VTCopyAV1DecoderCapabilitiesDictionary` on an M3 Max, macOS 27.0.1, measured
2026-11-27: `VTMaxDecodeLevel` 19 for profiles 0, 1 and 2). The device's caps list no AV1, as
the x86-64 guest's video driver, AppleGVA, decodes none (its strings name no AV1), so AV1's
number among an arm's codecs, `HOST_VIDEO_CODEC_AV1` (`graphics/av1.rs`), is the host's own
and reaches no guest: the wire, the caps and the option ROM are unchanged, and the host offers
an x86-64 guest no AV1 decoder. The route serves arm64 macOS 14.6.1 as it does 26.6.2: on an
RTX 3090 (NVDEC) under 14.6.1, the 17 pictures of our `svt-main8` stream, decoded in a session
that requires hardware, equal its reference frame by frame (**measured** 2026-11-27, under
TCG, potluck job m3-max-77531).

A session's `av1C` record is read as H.264's and HEVC's records are
(`paravirt/decoder/av1.rs`): the stream is told by the sequence header among its `configOBUs`,
or by the record's own fields where it carries none, and lands in `420v` or `420f` at 8 bits
and in `x420` or `xf20` at 10, by its range. Each sample, a temporal unit, goes to the arm
whole, and the video crate's AV1 decoder (`codec::av1`) reads its OBUs, the sequence and frame
headers and the tile groups, keeps the eight reference slots, and hands the hardware each
frame's tile data.

Profile 0 (Main) alone is listed. High (profile 1) is 4:4:4, which neither arm's AV1 decoder
lands. Professional (profile 2) is 4:2:2 below 12 bits (the AV1 specification, 5.5.2) and
4:2:0 only at 12 bits, which the guest's output formats, two planes of 10 bits at most, would
cut by two bits; of the lab's cards only the RX 7900 XTX's radeonsi lists Professional at all.
A session of either profile is refused at its start, and so is a Main stream past the depths
or the largest frame the arm lists, or below the smallest: NVDEC's caps list one (128x128 on
the RTX 3090), and VA-API's where the driver gives the AV1 decoder's surfaces a minimum size
(16x16 on radeonsi, none on iHD; m3-max-77432 and m3-max-77435) (`av1::declines`). The codec module refuses the same at a sequence header
(`codec::av1::check_supported`), which covers a record that carries none, whose stream is
known only at its first sample. The guest has nothing to fall back on:
arm64 macOS decodes no AV1 in software, and with no decoder from the host a session that asks
for AV1 fails −12906 (measured on an arm64 macOS 26.6.2 guest, 2026-11-27).

A picture the hardware reports failed fails to the guest as a decode error, as an H.264 or
HEVC picture does on this route: VideoToolbox's decoders copy no failed picture. NVDEC on the
RTX 3090 decodes a superres stream's inter frames wrongly and reports each failed, so the guest
gets the key frame and an error for each of those: of our `aom-superres`, 1 picture and 5
errors, −12909 (an arm64 macOS 26.6.2 guest, measured 2026-11-27, m3-max-77410;
[nvidia.md](nvidia.md), "The decoders"; [bugs.md](bugs.md), U36).

Film grain is applied by the hardware, and the frame with grain is what the guest gets, as the
specification's output is: VA-API decodes into `current_frame`, which later frames predict
from, and the driver writes the frame with grain into `current_display_picture`; NVDEC decodes
into `decodePicIdx` and writes the frame with grain into `CurrPicIdx`. Each driver draws the
grain itself, and the guest gets that driver's:
- **AMD (radeonsi, the RX 7900 XTX) and NVIDIA (NVDEC, the RTX 3090)** add the grain the
  specification describes: each frame with grain equals libdav1d's.
- **Intel (iHD, the Iris Xe)** draws grain of its own: noise of the specification's strength
  and shape in a different pattern, so no frame with grain equals libdav1d's, while each frame
  before grain does, and ffmpeg's own VA-API decode on iHD gives the same frames as ours
  ([bugs.md](bugs.md), U35). The card test judges it by its statistics against the
  specification's grain ([backends.md](backends.md), "Tests").

(Measured 2026-11-27: m3-max-77429, m3-max-77421 and m3-max-77438.)

How fast the guest decodes AV1 beside HEVC on the same route, the same frames of `testsrc2`
coded by SVT-AV1 and by x265, each decode a session that requires hardware, AV1 and HEVC in
turns, frames per second and the guest's CPU seconds for the run (measured with the guest's CPU
emulated, not on an arm64 host: an arm64 macOS 26.6.2 guest on 4 emulated CPUs, 2026-11-27):

| card (job) | size | AV1: fps, guest CPU s | HEVC: fps, guest CPU s |
|---|---|---|---|
| Radeon 780M (m3-max-77412) | 1920x1080, 60 frames | 70.8, 3.40; 16.7, 14.33; 14.7, 16.32; 89.7, 2.68; 98.6, 2.44; 97.2, 2.48 | 92.0, 2.59; 100.6, 2.38; 95.4, 2.52; 92.4, 2.61; 74.2, 3.24; 92.3, 2.61 |
| Radeon 780M (m3-max-77412) | 8192x4352, 8 frames | 19.3, 1.67; 23.5, 1.36; 23.8, 1.35; 21.0, 1.52; 23.2, 1.37; 23.4, 1.36 | 22.7, 1.44; 23.4, 1.36; 23.2, 1.45; 22.5, 1.42; 22.3, 1.45; 23.2, 1.38 |
| RX 7900 XTX (m3-max-77361) | 1920x1080, 60 frames | 89.6, 2.67; 95.5, 2.53 | 83.2, 2.90; 87.8, 2.72 |
| Iris Xe (m3-max-77386) | 1920x1080, 60 frames | 88.8, 2.72; 97.0, 2.48 | 93.4, 2.57; 92.9, 2.58 |
| RTX 3090 (m3-max-77410) | 1920x1080, 60 frames | 96.6, 2.49; 114.6, 2.09 | 105.0, 2.27; 115.6, 2.06 |

On the 780M, the medians of the six runs are AV1 80.2 and HEVC 92.4 fps at 1920x1080, a ratio
of 0.87, and 23.3 and 22.9 fps at 8192x4352, 1.02. Two of the first AV1 runs at 1920x1080 were
slow while the guest's CPU was busy with other work after its login: the decoding process's own
CPU stayed at 0.44 s, as in every run, and the guest's came to 14.3 and 16.3 s. At 1920x1080 the
other cards' ratios, of their runs' means, are 1.08 on the RX 7900 XTX, 1.00 on the Iris Xe and
0.96 on the RTX 3090. At that size the guest's CPU seconds come to about its 4 CPUs' whole time
over each run, so the emulated CPUs, not the card, set the rate.

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
command buffer of the session's process (`graphics/decode/video.rs`). VP9 has no op: its caps
entry serves VideoToolbox's paravirtual decoders alone ("Which hardware"), and a
`VIDEO_SESSION_CREATE` of codec 3 is refused as any codec without a session.

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
  the host API's errors still fault the command buffer. The range extensions' streams are
  streams this decoder does not take: the caps offer 4:2:0 alone, and its decoders land 4:2:0
  pictures at 8 and 10 bits alone, where VideoToolbox's paravirtual decoders on arm64 guests
  land them on hosts that decode them ([design.md](../graphics-kext/design.md), "The answers
  are the host's arms'"). With a kext that published the range extensions' profiles too (a
  scratch build, never landed), an x86-64 macOS 26.7.2 guest on the Iris Xe called the
  device's decode for every one of them, but its decoders asked for 4:2:0 destinations alone,
  `x420` for 10 and 12 bits and `420v` for 8, whatever the stream's chroma format, so no
  picture could match the guest's software decode of a 4:2:2 or 4:4:4 stream, or of a 12-bit
  one (**measured**, 2026-11-27, potluck jobs m3-max-77246 and m3-max-77257).
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

## A stream outside the host decoder's sizes

A host decoder takes pictures from its least size up to its largest, as the arm's own query of it
lists them:

- **NVDEC**: its caps of the stream's codec at its chroma format and depth (`cuvidGetDecoderCaps`),
  from `nMinWidth` by `nMinHeight` up to `nMaxWidth` by `nMaxHeight` and `nMaxMBCount` blocks of
  16x16, which the decoder's make checks too: on the RTX 5080 Laptop GPU, H.264 from 48x64 and
  HEVC from 144x144 ([bugs.md](bugs.md), D14).
- **VA-API**: the driver's decoder the VA crate's decoder picks for the stream, of the first
  profile that decodes it the driver lists with its render targets (for H.264, of Constrained
  Baseline, Main and High; for HEVC, Main, Main 10 or the range extensions' profile of the
  stream's chroma format and depth), from the least size its surfaces list (none, where they list
  none) up to the least of the largest its config and its surfaces list (iHD's H.264 config lists
  8192x8192, its surfaces 4096x4096), and for H.264 at most the 65,535 blocks the VA crate's
  decoder takes.

An x86-64 guest's caps carry the decoder's sizes, and the guest's video driver refuses a session
of a coded size outside them at its create, so VideoToolbox decodes such a stream in software
(`guest/bundle/src/RgfxVideoDriver.cpp`). On an arm64 guest, a VideoToolbox session of an H.264
or HEVC stream whose coded size lies outside them is refused at its `dsta`, its warning naming the
stream and the sizes (`guest_declines` in `graphics/nvidia.rs` and `graphics/vaapi.rs`, through
`graphics/decode_sizes.rs`, as `NVDEC decodes HEVC 4:2:0 from 144x144, not 128x128`), and the
guest's VideoToolbox decodes the stream in software where its client allows it. Before, the
session was made and its pictures failed in the host's decoder. AV1's sessions are checked
against the arm's AV1 decoder the same way ("AV1 on arm64 guests", above).

Measured on an arm64 macOS 26.6.2 guest under TCG on an RTX 3090 host (potluck job m3-max-77509,
2026-11-27), each stream decoded by a session that allows a hardware decoder and requires none:
H.264 at 32x32, HEVC Main at 128x128 and Main 10 at 128x96, each under the 3090's least size (H.264
48x16, HEVC 144x144), were refused at their `dsta` with `NVDEC decodes H.264 from 48x16, not 32x32`
and `NVDEC decodes HEVC 4:2:0 from 144x144, not ...`, and the guest decoded each stream's 30
pictures in software, every one the same as a decode that allows no hardware decoder; H.264 and
HEVC Main at 640x360 decoded in hardware, every picture the same.

## HEVC's range extensions on arm64 guests

VideoToolbox's paravirtual decoders, an arm64 guest's alone, take HEVC's range extensions where
the arm's own query lists a decoder of the stream: VA-API's decoder of the stream's range
extensions profile with the render targets and the surfaces it needs ([vaapi.md](vaapi.md),
"Surfaces, formats and readback"), NVDEC's caps for its chroma format, depth and size
([nvidia.md](nvidia.md)). Where the query lists none, the session's `dsta` is refused, its warning
naming the stream and the arm's reason (`guest_declines` in `graphics/vaapi.rs` and
`graphics/nvidia.rs`), and the guest decodes the stream in software. Monochrome, depths past 12
bits and screen content coding are the codec's own refusals on every host.

Each picture lands in the format the guest's own software decoder gives the stream, which `Dpba`
names (`PixelFormat::of`): `420v`, `x420` and `tv20` for 4:2:0 at 8, 10 and 12 bits, `422v`,
`x422` and `tv22` for 4:2:2, `444v`, `x444` and `tv44` for 4:4:4, and their full-range forms
(`420f`, `xf20`, `tf20` and the rest). Measured on arm64 macOS 27.0.1 on an M3 Max (potluck jobs
m3-max-77228 and m3-max-77231): the software decoder answers those formats, `tv20`, `tv22` and
`tv44` being two planes of 16-bit samples with the 12 bits at the top, as `x420` holds 10; the
M3's hardware decoder answers `422v` and `444v` at 8 bits and its own packed 10-bit formats,
`p420`, `p422` and `p444` (three samples in 32 bits), at 10, and refuses 12-bit streams (−12911).
CoreVideo's own descriptions of its packed formats give none the layout of VA-API's Y210, Y212,
Y410 or Y412 (**measured** on macOS 27.0.1, potluck job m3-max-77310): `v216` orders a 4:2:2
pair's 16-bit samples Cb, Y0, Cr, Y1, where Y210 and Y212 put Y0, Cb, Y1, Cr; `v410` holds a
pixel's 10-bit Cb, Y and Cr from bit 2 of its 32-bit word, Y410 from bit 0; `y416` orders A, Y,
Cb, Cr, where Y412 puts Cb, Y, Cr, A; and the M3's own 10-bit formats pack three samples in 32
bits. `yuvs` has YUY2's layout, but neither of the Mac's decoders gives it: both answer `422v`
for a 4:2:2 stream at 8 bits. So the device writes the guest's formats, and converts where the
host's surface has another layout, each picture it converts on the CPU counted
(`video-pictures-cpu-converted`):

| host | stream | host surface | guest's format | written by |
|---|---|---|---|---|
| Iris Xe, iHD 26.3.5 | 4:2:0, 12 bits | P012 | `tv20` | a copy, as Main 10's |
| | 4:2:2, 8 bits | YUY2 | `422v` | the CPU's split |
| | 4:2:2, 10 bits | Y210 | `x422` | the CPU's split |
| | 4:2:2, 12 bits | Y212 | `tv22` | the CPU's split |
| | 4:4:4, 8 bits | XYUV | `444v` | the GPU's split where the import check passed XYUV, else the CPU's |
| | 4:4:4, 10 bits | Y410 | `x444` | the CPU's split, each 10-bit field moved to the top of 16 bits |
| | 4:4:4, 12 bits | Y412 | `tv44` | the CPU's split |
| | 4:2:0, 8 and 10 bits | none: Main 12 lists P012 and P016 alone | | refused |
| RTX 3090, 620.70.06 | 4:2:0, 8, 10 and 12 bits | NV12, P016 | `420v`, `x420`, `tv20` | a copy, as Main and Main 10's |
| | 4:4:4, 8 bits | three planes | `444v` | the GPU's interleave where CUDA imports the device's buffers, else the CPU's |
| | 4:4:4, 10 and 12 bits | three planes of 16-bit samples | `x444`, `tv44` | the CPU's interleave |
| | 4:2:2 | none: NVDEC lists no HEVC 4:2:2 | | refused |
| RTX 5080 Laptop GPU, 615.71.09 | 4:2:2, 8 bits | NV16 | `422v` | a copy, as Main's |
| | 4:2:2, 10 and 12 bits | P216 | `x422`, `tv22` | a copy, as Main 10's |
| | 4:2:0 and 4:4:4 | as the RTX 3090's | | |
| Radeon 780M, radeonsi | any | none: no decoder listed | | refused |

The full-range forms land as the video-range ones. Our streams of the 4:2:0 range extensions
are 12-bit alone, so the 3090's 4:2:0 rows at 8 and 10 bits rest on its caps and the unit tests,
not on a boot.

Measured with VideoToolbox in arm64 macOS 26.6.2 guests under TCG, each stream one of our x265
streams at 640x360 of 10 pictures, decoded once by a session that requires hardware and once by
one that allows none (2026-11-27):
- On the Iris Xe (potluck job m3-max-77302), the hardware sessions decoded Main 10 and the
  range extensions at 4:2:0 12 bits and at 4:2:2 and 4:4:4 of 8, 10 and 12 bits, and the
  full-range forms of the 12-bit and the 4:2:2 and 4:4:4 ones past 8 bits, each into the format
  the software session gave, every picture byte for byte the same. The run counted 100
  pictures converted by the CPU and 50 split by the GPU. An arm64 macOS 14.6.1 guest there
  gave the same on the same streams: the same formats, every picture the same (m3-max-77338).
- On the RTX 3090 (m3-max-77315), the hardware sessions decoded Main 10, Main 12, and 4:4:4 at
  8, 10 and 12 bits, and the full-range forms of the 12-bit and the 4:4:4 ones past 8 bits, each
  into the software session's format, every picture byte for byte the same; the run counted 40
  pictures converted by the CPU and 50 interleaved by the GPU. Its sessions of 4:2:2 streams
  were refused at their start. Once the arm took 4:2:2 where NVDEC's caps list it, a boot there
  (m3-max-77572) decoded Main 10, Main 12 and 4:4:4 at 8, 10 and 12 bits in hardware again, every
  picture byte for byte the same, and refused each 4:2:2 session, at 8, 10 and 12 bits, naming
  the caps, as `NVDEC does not decode HEVC 4:2:2 at 10 bits (HEVC general_profile_idc 4,
  chroma_format_idc 2, luma 10 bits, chroma 10 bits, 640x360)`; the guest decoded those in
  software.
- On the RTX 5080 Laptop GPU (615.71.09), no arm64 guest has run; its 4:2:2 is checked on the
  host alone. The arm's VideoToolbox decoder, its sessions taken on the card's own caps (the card
  test `an_hevc_picture_lands_in_its_guest_format_as_the_software_decode_has_it` in
  `graphics/nvidia/tests.rs`, potluck job m3-max-77563), decoded our x265 streams of 4:2:2 at 8,
  10 and 12 bits, at 640x360 and 1920x1080 and at 10 bits at 1366x766, into `422v`, `x422` and
  `tv22` surfaces laid out as a pool's IOSurface, every picture byte for byte as ffmpeg's software
  decode of the stream, with none copied or converted by the CPU; and 4:2:0 and 4:4:4 at 8, 10
  and 12 bits, every picture the same too. It decoded and landed the 1920x1080 4:2:2 pictures at
  344 to 384 a second, each landed before the next was sent (a debug build). How fast a guest
  decodes them there is not measured.
- On the Radeon 780M (m3-max-77300), each hardware session of a range extensions stream was
  refused at its start, the warning naming the decoder radeonsi lacks, as `VA-API: the driver
  has no HEVCMain422_10 decoder (HEVC general_profile_idc 4, chroma_format_idc 2, luma 10
  bits, chroma 10 bits, 640x360)`, and the software sessions decoded them.

The conversions cost less than the guest's own decode. On the Iris Xe, the one host here that
lists the 4:2:2 decoders, and slower than the RTX 3090 at each stream both decode in the runs
above, a hardware session and a software one took turns three times on each of
our 1920x1080 streams of 30 pictures, in one arm64 macOS 26.6.2 guest under TCG (potluck job
m3-max-77327, 2026-11-27; measured with the guest's CPU emulated, not on an arm64 host). The
medians of the three, in seconds for the 30 pictures:

| stream | host surface, written by | hardware | software |
|---|---|---|---|
| 4:2:0, 12 bits | P012, a copy | 0.413 | 2.218 |
| 4:2:2, 8 bits | YUY2, the CPU's split | 0.379 | 3.306 |
| 4:2:2, 10 bits | Y210, the CPU's split | 0.615 | 2.320 |
| 4:2:2, 12 bits | Y212, the CPU's split | 0.587 | 2.313 |
| 4:4:4, 8 bits | XYUV, the GPU's split | 0.375 | 1.950 |
| 4:4:4, 10 bits | Y410, the CPU's split | 0.499 | 3.415 |
| 4:4:4, 12 bits | Y412, the CPU's split | 0.593 | 3.585 |

Every hardware run was faster than every software run of its stream (hardware 0.361 to 0.660 s,
software 1.883 to 5.315 s), and the run counted 450 pictures converted by the CPU, the five CPU
rows' 90 each.

## VP9's profiles 1 and 3 on arm64 guests

VP9's profile 1 (4:2:2 and 4:4:4 at 8 bits), profile 3 (the same at 10 and 12 bits) and profile 2
at 12 bits reach the host by the paravirtual route as profiles 0 and 2 do. The caps entry still
lists profiles 0 and 2 alone, and the device's wire is unchanged: the host's `dsta` decides from
the arm's own answers (`declines_vp9` in `graphics/vaapi.rs` and `graphics/nvidia.rs`), naming
the stream and the reason in its warning where it refuses. VA-API takes a stream where the
driver's decoder of its profile (`VAProfileVP9Profile1` to `3`) makes the render targets and
lists the surfaces of its chroma format and depth, the ones HEVC's range extensions decode into:
P012 for 4:2:0 at 12 bits, YUY2, Y210 and Y212 for 4:2:2, XYUV, Y410 and Y412 for 4:4:4
([vaapi.md](vaapi.md), "The VP9 decoder"). Its pictures land as the range extensions' pictures
of the same chroma format and depth do (the table in "HEVC's range extensions on arm64 guests",
above): `tv20`, `422v`, `x422`, `tv22`, `444v`, `x444` and `tv44`. NVDEC takes profile 0 and
profile 2 at 10 bits alone ([nvidia.md](nvidia.md)). 4:4:0 and VP9's RGB colour space are the
codec's own refusals on every host.

| host | taken | refused |
|---|---|---|
| Iris Xe, iHD 26.3.5 | profile 1 at 4:4:4; profile 2 at 12 bits; profile 3 at 4:4:4, 10 and 12 bits | 4:2:2 at every depth: the decoders of profiles 1 and 3 make 4:2:2 render targets but list no 4:2:2 surfaces, as `VA-API's VP9Profile1 decoder writes [AYUV, XYUV], not YUY2`, and fail a 4:2:2 frame's decode into any surface (vaapi.md) |
| RX 7900 XTX and Radeon 780M, radeonsi | none | profiles 1 and 3: `VA-API: the driver has no VP9Profile1 decoder`; profile 2 at 12 bits: no YUV420_12 render targets |
| RTX 3090, NVDEC | none | `NVDEC does not decode VP9 profile 1 at 8 bits, chroma_format_idc 3`, and so on |

The guest's pictures of our libvpx streams of the formats past profile 0 and profile 2 at 10 bits
(`src/codec/vp9/tests` of the video crate: 352x288, 20 frames, two superframes each) equal
vpxdec's frame by frame, by the IVF route and the MP4 route, in sessions that require hardware, on
the Iris Xe under arm64 macOS 26.6.2 and 14.6.1 under TCG: profile 1 at 4:4:4 into `444v`,
profile 2 at 12 bits into `tv20`, and profile 3 at 4:4:4 into `x444` and `tv44`. The session of
each 4:2:2 stream failed −12910 at its create, and the profile 0 and 2 streams decoded as before
(**measured** 2026-11-27, potluck jobs m3-max-77567 and m3-max-77606). A profile 1 4:4:4 stream at
1920x1080 (60 frames, profile 0's stream's frames) decodes at 85% and 87% of profile 0's rate on
the same card, its pictures split by the GPU from XYUV into `444v`'s two planes (3.4 ms a picture
on average, from the driver's write to the pass's end). Measured with the guest's CPU emulated
(TCG, 4 CPUs), in sessions that require hardware taking turns with profile 0's, in frames a
second: under macOS 26.6.2 (m3-max-77567), 77.9 and 82.1 against profile 0's 99.5, 82.0 and
94.6, and HEVC's 92.1; under 14.6.1 (m3-max-77606), 81.2 and 77.9 against 91.0, 93.5 and 88.8,
and HEVC's 84.9. The percentages are the means of profile 1's runs against the medians of
profile 0's.
