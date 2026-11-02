# The Vulkan Video decoder

`crates/reims-vgpu-video-vulkan`: H.264 and H.265 decode through Vulkan
Video (`VK_KHR_video_queue`, `VK_KHR_video_decode_queue`,
`VK_KHR_video_decode_h264`, `VK_KHR_video_decode_h265`): H.264 8-bit 4:2:0
progressive, and H.265 Main and Main 10 (4:2:0 at 8 and 10 bits), as the
codec crate's decoders take them. It is not a backend yet: nothing in core or the glue
opens it. It is built to run on the device the library draws with on
Linux (metal2vulkan's), and to land each picture in the GPU copy of a
guest IOSurface, which the guest's Metal then samples per plane, or, until
the IOSurfaces are images, in the IOSurface's own bytes in host memory. Every fact names
its run; runs are on an NVIDIA GeForce RTX 5080 Laptop GPU (driver
615.71.09, Vulkan loader and validation layers 1.4.357, on CachyOS) unless
they say otherwise.

## What is here

| what | where |
|---|---|
| The device: another's (`VideoDevice::from_raw`) or its own (`VideoDevice::create`) | `src/device.rs` |
| Copies out on a queue another submitter lends (`LentQueue`), each submitted from a thread once the host sees its decode done | `src/lent.rs` |
| The profiles, and what the device answers for each (`h264_capabilities`, `h265_capabilities`) | `src/profile.rs` |
| The outputs a decoder lands its pictures in: images (`SurfaceSpec`) or buffer planes (`BufferPlanes`), and sources of its own (`DecodeImages`, `HostBuffers`) | `src/surface.rs` |
| A decoded picture: its output, its DPB slot, its timeline value; the read back of tests | `src/picture.rs` |
| The timeline semaphore, and what waits on it before it goes | `src/timeline.rs` |
| The codec crate's H.264 accelerator: slots, references, pictures | `src/h264.rs` |
| The H.264 parameter sets as std structures | `src/h264/params.rs` |
| The codec crate's H.265 accelerator, and the plan of each decode: its slots, its current sets as slots, its std information | `src/h265.rs` |
| The H.265 parameter sets as std structures: the VPS made from the SPS, the SPS's sets coded outright, the scaling lists, the tiles | `src/h265/params.rs` |
| The GPU-less tests of the H.265 plans over the codec crate's decoding process, and of the std structures | `src/h265/tests.rs` |
| The session, of either codec: its memory, its DPB pictures, and the command and bitstream buffers of each decode and copy | `src/session.rs` |
| A session parameters object, of either codec | `src/params.rs` |
| Each frame's NV12 MD5, as the VA-API backend's `h264-framemd5`, from images or buffers (`--buffer`), with copies on the decode or the graphics queue (`--copy-on-graphics`) or on a lent graphics queue (`--copy-lent`); the device's answers (`--capabilities`) | `examples/vulkan-h264-framemd5.rs` |
| Each frame's NV12 or P010 MD5, as the VA-API backend's `hevc-framemd5`, with the same options, and access units read in place (`--in-place`) | `examples/vulkan-h265-framemd5.rs` |
| The VA-API H.264 check's streams, decoded here against their references | `tools/h264-check.sh` |
| The VA-API HEVC check's streams, decoded here against their references | `tools/h265-check.sh` |

## The device

A decoder takes the device it runs on; it never makes one. The device is
another's, handed over as raw handles (`RawDevice`: the loader, instance,
physical device, device, a video-decode queue family and a queue of it, a
queue that copies and its family, and the graphics family whose work
samples the outputs), or one `VideoDevice::create` makes for the tests and
tools: the first device, discrete GPUs first, with Vulkan 1.3 and a queue
family that decodes H.264, with `VK_KHR_video_decode_h265` too where the
device lists it and the family decodes H.265; one graphics queue and one video-decode queue,
the copy queue being the decode queue where its family copies, else the
graphics queue; and `VK_KHR_video_maintenance1` and
`VK_KHR_unified_image_layouts` (`unifiedImageLayoutsVideo`) where it has
them. A device handed over decodes H.264 alone (`VideoDevice::from_raw`),
or the codecs its owner enabled (`VideoDevice::from_raw_with_codecs`),
copying out on the copy queue it names, or on a queue of the owner's that
the owner lends for each submission (`VideoDevice::from_raw_copying_on`,
a `LentQueue`): reims-vgpu lends a metal2vulkan command queue's, whose
family the runtime's buffers, an IOSurface's device cache among them, are
shared with;
`h265_capabilities` answers nothing for a device without H.265. Where no
device decodes H.264 it says why, device by device (on a Mac: the loader is not found,
or MoltenVK has no video queue).

The 5080 answers (`vulkan-h264-framemd5 --capabilities`, 11-02): decode
family 3 (video decode and transfer), graphics family 0,
`unifiedImageLayoutsVideo` on; Baseline, Main and High each to level 5.2,
48x16 to 4096x4096 at a 16x16 granularity, 17 DPB slots, 16 active
references, `DPB_AND_OUTPUT_COINCIDE` only, bitstream offsets and sizes
aligned to 256. Its decode family copies, so each picture's copy into its
output runs in its decode's own command buffer. For H.265 (11-03): Main
and Main 10 each to level 6.1, 144x144 to 8192x8192 at a 16x16
granularity, 16 DPB slots, 16 active references, `DPB_AND_OUTPUT_COINCIDE`
only, bitstream offsets and sizes aligned to 256.

## Pictures

A reference is never an output. The guest's pixel buffer pool recycles an
output once it has been shown, whether or not its picture is still a
reference, so each session owns its DPB pictures: one a slot, the SPS's
`max_num_ref_frames` and one more for the picture being decoded (for
H.265, the DPB's `sps_max_dec_pic_buffering_minus1 + 1`, the current
picture's among them), NV12 (`VK_FORMAT_G8_B8R8_2PLANE_420_UNORM`), or
P010 (`VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16`) for Main 10,
optimally tiled, their extent the coded size rounded up to the profile's
`pictureAccessGranularity`, with the stream's
profile in a `VkVideoProfileListInfoKHR`. Each decode sets up the least
slot no reference holds, its DPB picture taken from `UNDEFINED`; with
unified layouts every DPB picture stays in `GENERAL`, else in
`VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR`.

Each picture lands in an output its `SurfaceSource` gives
(`Surface::landing`): an image, or two planes in a buffer.

An output image is made as `SurfaceSpec` says: NV12 or P010
(`SurfaceSpec::format`), at the DPB pictures' extent, optimally tiled, with `MUTABLE_FORMAT` and
`EXTENDED_USAGE` so that its planes can be viewed as R8 and RG8, sampled,
transfer source and destination, and shared between the graphics, decode
and copy families. Where the device decodes into an output beside the
DPB picture (`DPB_AND_OUTPUT_DISTINCT`), the decode writes the output,
which then also has the decode-destination usage and the profile list.
Where the output and the DPB picture coincide (`DPB_AND_OUTPUT_COINCIDE`,
all the 5080 offers), the decode writes the DPB picture and one GPU copy
of both planes takes the picture into the output: in the decode's own
command buffer where the decode family copies (the 5080's does), else on
the copy queue. Either way the output is taken from `UNDEFINED` and left
in `GENERAL`. The profile is the stream's own where the device decodes
it, else Main, else High: a Baseline stream without FMO, ASO or redundant
pictures is Main's.

Buffer planes (`BufferPlanes`) are the luma plane and the interleaved
chroma plane at the caller's offsets and row pitches in a buffer of its
own, which the host binds over the guest surface's memory. One copy
writes them, cropped to the SPS's rectangle: out of the DPB picture where
the two coincide, else out of a scratch image of the session's that the
decode writes, in the decode's command buffer or on the copy queue as for
an image. It makes its writes available to the host before the picture's
value is signalled. Each offset is a multiple of 4 and each pitch at least
the crop's width, the chroma pitch even, as Vulkan copies to a buffer on a
queue that neither draws nor computes; others are refused. P010 rows are
two bytes a sample: each pitch twice the crop's width at least (chroma's
of whole Cb/Cr pairs), luma's a multiple of 2 and chroma's of 4.

The DPB's references go to the decode with their slots, each picture
keeping its slot while it is a reference. A stand-in for a lost frame
shares its source's output and slot (only a decode sets a slot up): the
decode names it where its source is no longer a reference, and names the
source for both where both are.

## Bitstream and parameters

Each slice goes into a host-visible bitstream buffer after a three-byte
start code, its offset that of the start code. The Vulkan specification
has `pSliceOffsets` locate each slice and says nothing of start codes;
ffmpeg's Vulkan decoder writes them so, and the 5080 decodes every stream
below so.

The session parameters object holds the stream's SPSs and PPSs as std
structures, made again, with the others it held, when one arrives that it
does not hold as it is. Scaling lists go as the codec crate resolves them,
every list a set's syntax carries marked present, in zig-zag order; the
check's asymmetric matrix (`high-720p-cqm`) catches a list in the wrong
order or slot. The VUI is left out: decoding needs none of it.

## Frame numbers

Each reference goes with its `FrameNum` (or `LongTermFrameIdx`), and the
picture with its `frame_num`, as the stream has them, except past a wrap.
NVIDIA's driver (615.71.09, 11-02) builds a P slice's initial reference
list from `FrameNum` without wrapping it, so from the first P picture after
`frame_num` wraps whose list holds frames from both sides of the wrap, it
predicts from the wrong frames: every Baseline stream of the check with
two or more references broke at frame 17 (`MaxFrameNum` 16), and so does
ffmpeg's Vulkan decoder, while NVDEC through CUDA decodes them right. Its
list modifications do wrap: passing `FrameNumWrap` as a signed number
mended the initial lists and broke `main-720p`'s modified ones. So where
a short-term reference's `FrameNum` exceeds `frame_num`, the decoder hands
the driver numbers shifted alike: the picture `MaxFrameNum - 1`, each
reference its `FrameNumWrap` plus the same shift. No reference wraps then,
and every `PicNum` and `CurrPicNum` moves by the same amount, so the
initial lists and the modifications name the same frames on any driver
that numbers from the picture's information, as the Vulkan structures
carry it.

## H.265

The codec crate's HEVC decoder runs the decoding process above the slice
data: order counts (8.3.1), the reference picture set's search and
marking (8.3.2), random access (IDR, CRA and BLA pictures,
`NoRaslOutputFlag` and the RASL pictures it skips, 8.1.3) and output
order (C.5.2). A RASL picture it skips answers
`Error::H265Stream(hevc::Error::SkippedLeadingPicture)`: no picture, and
no failure. The hardware reads each slice segment header itself.

Each decode's plan (`h265::plan`):
- Every picture of the reference picture set the DPB holds goes as a
  reference with its slot, order count and long-term marking: a current
  one, or one kept for later pictures (`RefPicSetStFoll`,
  `RefPicSetLtFoll`), which so keeps its slot. The picture is set up in the
  least slot none of them holds. A picture held only for output holds no
  slot: its output was written when it was decoded.
- `RefPicSetStCurrBefore`, `RefPicSetStCurrAfter` and `RefPicSetLtCurr`
  name slots in 8-5's order, the rest of each `0xFF`. The std structure
  holds 8 a set, so a picture whose set names more (kvazaar's `--ref 15`
  names 9 from its tenth picture on) is refused before it reaches the GPU,
  after the pictures before it.
- `IsReference` is 1 for every picture: 8.1.3 marks each decoded picture
  "used for short-term reference", and a sub-layer non-reference picture
  may still be predicted from by a higher sub-layer.
- A slice header's own set goes as its size in bits and, where it is
  predicted from an SPS set, that set's `NumDeltaPocs`, for the hardware
  to skip.

The parameters object holds VPSs, SPSs and PPSs as std structures, made
again, with the others it held, when a set arrives that it does not hold
as it is:
- The VPS is made from the SPS: the codec crate drops VPS units, as a
  base-layer decode reads none. It has the SPS's sub-layers, temporal
  nesting, profile, tier, level and DPB limits, as the one-layer VPS of a
  conforming stream has them.
- The SPS's short-term sets go each coded outright, of the pictures the
  syntax derives, whether the stream predicts one from another or not.
- Scaling lists go as the codec crate resolves them (the defaults where
  the SPS enables lists and codes none), marked present, in coding order,
  the two 32x32 lists those of `matrixId` 0 and 3.
- Tiles go with every column's and row's size, uniform spacing too.
- A `general_profile_idc` other than Main, Main 10 or Main Still Picture,
  of a stream the codec crate takes as compatible with one, goes as the
  profile the session decodes it with. Main and Main Still Picture decode
  with Main, 10-bit streams with Main 10.

`H265Decoder::decode_in_place` reads an Annex B access unit where it lies
in device memory, as H.264's does: where the first slice segment's start
code lies at the device's bitstream offset alignment and the sample is
padded to its size alignment, else the slices are copied.

## Synchronization

Each decode signals the decoder's timeline semaphore with the next value.
A copy on the copy queue waits for its decode's value and signals the
next, and the decode after it waits for that copy, since it may set the
same DPB picture up again. A picture names the value after which its
output holds it (`Picture::value`): its decode's where the decode or a
copy in its own command buffer writes the output, its copy's where the
copy queue does. On a lent queue, which the owner's own work also runs
on, a copy that waited on the GPU would hold that work behind its decode:
a thread of the device's instead waits on the host for the decode's value,
then submits the copy with no wait. Graphics work waits for it before it
samples the output, and `Picture::wait` blocks on it. An output, a view, a bitstream buffer, a
parameters object, a DPB picture or a session is let go only once the
timeline has passed the last work that used it. `Picture::read_back`, for
tests, copies both planes out on the copy queue after the picture's
value.

## Checks

- The unit tests (`cargo nextest run -p reims-vgpu-video-vulkan`) need no
  GPU: the std structures, the levels, the slices' layout, the choice of
  slots and references. For H.265 they write streams from the standard's
  syntax (hierarchical GOPs, long-term pictures across the order count's
  wrap, CRA pictures leading and mid-stream, slice headers' own sets),
  run them through the codec crate's decoding process, and check each
  decode's plan against a model of the slots: every slot a decode names
  holds the picture the codec crate names there.
- `tools/h264-check.sh` decodes the VA-API H.264 check's streams
  (`crates/reims-vgpu-video-vaapi/tools/h264-check.sh`, whose recipes it
  copies, into the same directory) from their MP4 and Annex B forms, in
  display order and decode order, against ffmpeg's frames: the crafted
  streams, the SPS switch, the lost reference and the interlaced
  refusals included. One stream is decoded again with its copies on the
  graphics queue, and again with them on a lent graphics queue, into
  images and into buffers, and three landed in buffers, one of them
  through the graphics queue. It makes the streams first, then runs itself
  again holding every GPU's lock.
- On the 5080 (11-03) all 68 of the check's verdicts pass: every stream
  from MP4 and Annex B in display order and from MP4 in decode order, the
  crafted streams, 2- and 3-byte lengths, frames held by the caller, the
  copy on the graphics queue, the two copies on a lent queue, the four
  buffer landings, the SPS switch, the lost reference and both interlaced
  refusals. Before the frame
  numbers were shifted, the Baseline streams with two or more
  references, `baseline-720p-poc1`, the long-term streams and
  `gaps2-ref4` and `gaps9-ref4` failed from the first wrap on.
- Under the validation layer, `baseline-720p` and `high-1080p` decode
  bit-exact with no message into images and into buffers, with copies on
  the decode queue and on the graphics queue, with unified layouts and
  with them turned off by a local patch (11-02).
- `tools/h265-check.sh` decodes the VA-API HEVC check's streams
  (`crates/reims-vgpu-video-vaapi/tools/hevc-check.sh`, whose recipes it
  copies, into the same directory) as the H.264 check does: x265's,
  kvazaar's (found as `RVGV_KVAZAAR` names it, else on the PATH; without
  it each of its streams is a FAIL) and hevc-craft's, against ffmpeg's
  frames or the encoder's reconstruction where the two differ. Besides,
  it cuts the open-GOP streams at their second CRA picture and splices a
  BLA picture there, lands pictures in buffers at 8 and 10 bits, reads
  access units in place, switches from Main to Main 10 mid-stream, checks
  the refusals, and decodes eight runs again under the validation layer,
  whose log must be empty.
- On the 5080 (11-03) all 191 of the H.265 check's verdicts pass, in
  about 2 minutes of GPU time: 45 streams (19 of x265's, 12 of kvazaar's
  and 14 crafted) each from MP4 and Annex B in display order and from MP4
  in decode order; the three cuts at a CRA picture, from Annex B and by
  `--skip`, and the three BLA splices; 2- and 3-byte lengths, frames held
  by the caller, two runs copied out on the graphics queue, six buffer
  landings; six in-place runs, every picture read in place; the SPS
  switch; the 4:2:2, 4:4:4, 12-bit and narrow-tile refusals; the 9 frames
  of `kvz-720p-ref15` before its picture of nine current references, which
  is refused; and the eight runs under the validation layer, which says
  nothing. No case needed a workaround of the driver.
- The validation layer runs with `VK_LOADER_LAYERS_ENABLE='*validation'`
  and, on this driver, `VK_KHRONOS_VALIDATION_UNIQUE_HANDLES=false`: with
  the layer's handle wrapping on, the driver's `vkQueueSubmit2` of a
  video decode segfaults, ffmpeg's Vulkan decoder's too (11-02).

## Not done

- No backend: core's `DecoderBackend` over it, and the glue's choice of
  it, come with the integration on metal2vulkan's device.
- Only H.264 and H.265. AV1 and VP9 decode on the same session and
  picture machinery with their own std structures.
- The distinct path, decoding straight into the output image or into the
  scratch image for a buffer, is built but unmeasured: the 5080 offers
  coincident pictures alone.
- Only NVIDIA measured. RADV and ANV, which expose Vulkan Video on the
  AMD and Intel hosts, are unchecked.
