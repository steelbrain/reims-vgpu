# The graphics device's video

The reims-graphics device (PCI 1234:5247) decodes and encodes the guest's H.264 and HEVC on the
host's hardware. The guest's video driver, in our Metal bundle, is AppleGVA's vendor driver
([docs/graphics-kext/design.md](../graphics-kext/design.md), "Decided", 16); it decodes, and
sends no encode op yet. This file is the host's side: what it answers the guest, and what a
decode and an encode do. The wire's records and their
fields are documented where they are defined: `crates/reims-graphics/src/wire.rs` (the caps and
the session) and `crates/reims-graphics/src/ops/video.rs` (the ops).

## Which hardware

On Linux, the device's video runs on the first of the host's two video arms that takes the
runtime device (reims-vgpu's `graphics/video_choice.rs`):

1. NVIDIA's, decoding on NVDEC and encoding on NVENC, where the device is an NVIDIA GPU that
   CUDA opens (`graphics/nvidia.rs`, [nvidia.md](nvidia.md)).
2. VA-API, on the device's render node, whatever its vendor (`graphics/vaapi.rs`,
   [vaapi.md](vaapi.md)).

Each arm that declines says why in one line at info. A Mac has no video arm, on either platform.
Where no arm takes the device, the host writes no video caps, and the guest is offered no
codec.

## The caps

At the device's init, the host writes an `RgfxVideoCaps` at `RGFX_VIDEO_CAPS_OFFSET`, after
the device info (`GraphicsDevice::with_platform`, in `graphics.rs`):

- **The entries**: one `RgfxVideoCodecCaps` for each codec and direction the arm takes: its
  profiles, its highest level, its smallest and largest coded pictures, its DPB and reference
  bounds, its output formats, and the alignment at which it reads an access unit in place.
  Each answer is the arm's video API's own (the arms' docs).
- **The flags**: `RGFX_VIDEO_CAPS_OUTPUT_CACHED` where the kext makes device caches
  (`RGFX_FEATURE_SURFACE_CACHE`) and the arm lands pictures in them by its own GPU work: NVDEC,
  where CUDA imports the device's exportable buffers ([nvidia.md](nvidia.md)). The device's
  caches are then made exportable as they are made, and the guest's video driver asks its
  output surfaces cached (`guest/bundle/src/RgfxMetalDevice+Video.mm`); elsewhere it asks them
  uncached. VA-API never sets the flag.
- **The layouts**: an `RgfxVideoOutputLayout` per entry, which says how a surface is laid out
  to take a picture's whole coded rows and columns in place. An arm that lands every picture by
  a copy of its crop answers zero, and every entry's is zero where the flag is set, since a
  picture lands in place only in a surface's pages. For an encode entry it would say how a
  source is laid out to be read in place; both arms read every source by a copy, and answer
  zero.
- **What an encode entry takes**: an `RgfxVideoEncodeCaps` per entry (`encode`): the rate
  controls (a bit each of CQP, CBR and VBR, as the arm's API lists them for the codec's
  profiles) and the most consecutive B pictures, 0 on both arms, whose encoders keep frames in
  display order.

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
  another process still reads lands once that read is done.
- **The access unit is read in place**, in the guest's buffer, where it starts and ends at the
  entry's `bitstream_alignment`; elsewhere the arm copies it. The parameter sets the op carries
  are handed to the decoder before the access unit.
- **The picture lands** in the output surface's planes, cropped: in the surface's pages, or in
  its device cache where the kext keeps the surface in one. The arms' ways differ (below).
- **The command buffer's later work, and its completion, wait** for the session's event to
  reach the value the decode signals once its picture has landed. So a CPU lock of the output
  waits for the picture, as for any GPU write.
- **A picture that fails to land** still signals its value, and the command buffer completes
  failed, with Metal's internal error. The guest's video driver answers that as a decode error:
  the session's next schedule fails (`kVTVideoDecoderMalfunctionErr`, design.md, "A decode the
  driver cannot complete is a decode error").
- **A picture the stream's random access skips** lands nothing, and its command buffer
  completes as any other.

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
  in the order the encodes were taken, once the command buffer's waits are done: its two planes
  are read as host bytes, in the surface's pages, or, for a plane in a device cache, in a staging
  buffer the GPU copies it into first.
- **The access unit is written** in Annex-B form into the bitstream buffer, a keyframe's with
  its parameter sets first, and the frame's `RgfxVideoEncodeResult` after it. One past the
  capacity is not written (`RGFX_VIDEO_ENCODE_OVERFLOW`, with the bytes it needed), and the
  next frame is made a keyframe; a frame the hardware fails gets `RGFX_VIDEO_ENCODE_FAILED`.
  Either way the command buffer completes as any other.
- **The command buffer's later work, and its completion, wait** for the session's event to
  reach the value the encode signals once its result is written.

| arm | encoder | its access unit |
|---|---|---|
| NVENC | `NvencEncoder` ([nvidia.md](nvidia.md)); `quality` picks the presets P1 to P7 at the high-quality tuning, and the keyframe interval is the GOP (`graphics/nvidia/encode.rs`) | Annex B as NVENC writes it, its parameter sets before each IDR picture |
| VA-API | `VideoEncoder` ([vaapi.md](vaapi.md)), on `EncSlice`, else `EncSliceLP`; `quality` maps onto the driver's quality levels (`graphics/vaapi/encode.rs`) | the encoder's length-prefixed sample, written as Annex B, a keyframe's parameter sets first from its configuration record |

## How each arm lands a picture

| arm | access unit in place | a picture lands | layout asked of the guest |
|---|---|---|---|
| NVDEC | at any place and length | by one CUDA copy a plane: into the output's device caches, through CUDA's import of their memory, where the flag is set; into the pages, where CUDA registers their view; else read back and copied by the CPU | none |
| VA-API | at any place and length | by the driver's video processing into the pages, where the import check passed for the format and the planes meet its alignment; else read back and copied by the CPU | granularity [2, 2], the import check's alignment |

Both run in reims-vgpu's host video session (`graphics/video_host.rs`, [backends.md](backends.md),
"The traits"): a plane in a device cache the arm does not write itself takes its picture staged
in host bytes, and the session copies it into the cache on a runtime queue. Each session counts
how its pictures landed, and logs the counts at its end ([vaapi.md](vaapi.md), "The
reims-graphics device").
