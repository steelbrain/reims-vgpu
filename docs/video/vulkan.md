# The Vulkan Video decoder

`crates/reims-vgpu-video-vulkan`: H.264 decode through Vulkan Video
(`VK_KHR_video_queue`, `VK_KHR_video_decode_queue`,
`VK_KHR_video_decode_h264`), 8-bit 4:2:0 progressive, as the codec crate's
decoder takes it. It is not a backend yet: nothing in core or the glue
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
| The profiles, and what the device answers for each (`h264_capabilities`) | `src/profile.rs` |
| The outputs a decoder lands its pictures in: images (`SurfaceSpec`) or buffer planes (`BufferPlanes`), and sources of its own (`DecodeImages`, `HostBuffers`) | `src/surface.rs` |
| A decoded picture: its output, its DPB slot, its timeline value; the read back of tests | `src/picture.rs` |
| The timeline semaphore, and what waits on it before it goes | `src/timeline.rs` |
| The codec crate's accelerator: slots, references, pictures | `src/h264.rs` |
| The parameter sets as std structures, and the parameters object | `src/h264/params.rs` |
| The session, its memory, its DPB pictures, and the command and bitstream buffers of each decode and copy | `src/h264/session.rs` |
| Each frame's NV12 MD5, as the VA-API backend's `h264-framemd5`, from images or buffers (`--buffer`), with copies on the decode or the graphics queue (`--copy-on-graphics`); the device's answers (`--capabilities`) | `examples/vulkan-h264-framemd5.rs` |
| The VA-API H.264 check's streams, decoded here against their references | `tools/h264-check.sh` |

## The device

A decoder takes the device it runs on; it never makes one. The device is
another's, handed over as raw handles (`RawDevice`: the loader, instance,
physical device, device, a video-decode queue family and a queue of it, a
queue that copies and its family, and the graphics family whose work
samples the outputs), or one `VideoDevice::create` makes for the tests and
tools: the first device, discrete GPUs first, with Vulkan 1.3 and a queue
family that decodes H.264; one graphics queue and one video-decode queue,
the copy queue being the decode queue where its family copies, else the
graphics queue; and `VK_KHR_video_maintenance1` and
`VK_KHR_unified_image_layouts` (`unifiedImageLayoutsVideo`) where it has
them. Where no device decodes
H.264 it says why, device by device (on a Mac: the loader is not found,
or MoltenVK has no video queue).

The 5080 answers (`vulkan-h264-framemd5 --capabilities`, 11-02): decode
family 3 (video decode and transfer), graphics family 0,
`unifiedImageLayoutsVideo` on; Baseline, Main and High each to level 5.2,
48x16 to 4096x4096 at a 16x16 granularity, 17 DPB slots, 16 active
references, `DPB_AND_OUTPUT_COINCIDE` only, bitstream offsets and sizes
aligned to 256. Its decode family copies, so each picture's copy into its
output runs in its decode's own command buffer.

## Pictures

A reference is never an output. The guest's pixel buffer pool recycles an
output once it has been shown, whether or not its picture is still a
reference, so each session owns its DPB pictures: one a slot, the SPS's
`max_num_ref_frames` and one more for the picture being decoded, NV12
(`VK_FORMAT_G8_B8R8_2PLANE_420_UNORM`), optimally tiled, with the stream's
profile in a `VkVideoProfileListInfoKHR`. Each decode sets up the least
slot no reference holds, its DPB picture taken from `UNDEFINED`; with
unified layouts every DPB picture stays in `GENERAL`, else in
`VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR`.

Each picture lands in an output its `SurfaceSource` gives
(`Surface::landing`): an image, or two planes in a buffer.

An output image is made as `SurfaceSpec` says: NV12, optimally tiled, with `MUTABLE_FORMAT` and
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
queue that neither draws nor computes; others are refused.

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

## Synchronization

Each decode signals the decoder's timeline semaphore with the next value.
A copy on the copy queue waits for its decode's value and signals the
next, and the decode after it waits for that copy, since it may set the
same DPB picture up again. A picture names the value after which its
output holds it (`Picture::value`): its decode's where the decode or a
copy in its own command buffer writes the output, its copy's where the
copy queue does. Graphics work waits for it before it samples the output,
and `Picture::wait` blocks on it. An output, a view, a bitstream buffer, a
parameters object, a DPB picture or a session is let go only once the
timeline has passed the last work that used it. `Picture::read_back`, for
tests, copies both planes out on the copy queue after the picture's
value.

## Checks

- The unit tests (`cargo nextest run -p reims-vgpu-video-vulkan`) need no
  GPU: the std structures, the levels, the slices' layout, the choice of
  slots and references.
- `tools/h264-check.sh` decodes the VA-API H.264 check's streams
  (`crates/reims-vgpu-video-vaapi/tools/h264-check.sh`, whose recipes it
  copies, into the same directory) from their MP4 and Annex B forms, in
  display order and decode order, against ffmpeg's frames: the crafted
  streams, the SPS switch, the lost reference and the interlaced
  refusals included. One stream is decoded again with its copies on the
  graphics queue, and three landed in buffers, one of them through the
  graphics queue. It makes the streams first, then runs itself again
  holding every GPU's lock.
- On the 5080 (11-02) all 66 of the check's verdicts pass: every stream
  from MP4 and Annex B in display order and from MP4 in decode order, the
  crafted streams, 2- and 3-byte lengths, frames held by the caller, the
  copy on the graphics queue, the four buffer landings, the SPS switch,
  the lost reference and both interlaced refusals. Before the frame
  numbers were shifted, the Baseline streams with two or more
  references, `baseline-720p-poc1`, the long-term streams and
  `gaps2-ref4` and `gaps9-ref4` failed from the first wrap on.
- Under the validation layer, `baseline-720p` and `high-1080p` decode
  bit-exact with no message into images and into buffers, with copies on
  the decode queue and on the graphics queue, with unified layouts and
  with them turned off by a local patch (11-02).
- The validation layer runs with `VK_LOADER_LAYERS_ENABLE='*validation'`
  and, on this driver, `VK_KHRONOS_VALIDATION_UNIQUE_HANDLES=false`: with
  the layer's handle wrapping on, the driver's `vkQueueSubmit2` of a
  video decode segfaults, ffmpeg's Vulkan decoder's too (11-02).

## Not done

- No backend: core's `DecoderBackend` over it, and the glue's choice of
  it, come with the integration on metal2vulkan's device.
- Only H.264. HEVC, AV1 and VP9 decode on the same session and picture
  machinery with their own std structures.
- The distinct path, decoding straight into the output image or into the
  scratch image for a buffer, is built but unmeasured: the 5080 offers
  coincident pictures alone.
- Only NVIDIA measured. RADV and ANV, which expose Vulkan Video on the
  AMD and Intel hosts, are unchecked.
