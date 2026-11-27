# The host's video arms

The reims-graphics device's video runs on one of two arms of the host's, NVDEC and VA-API
([devices.md](devices.md), "Which hardware"). This file is how they are put together: the
crate, the traits between them, and how each is tested.

## The crate

`crates/reims-vgpu-video` holds the video code, in five modules:

| module | what it is |
|---|---|
| `core` (`src/core.rs`) | What the arms share: the threads and clock the host gives them, their locks, a guest surface's memory as a device reaches it, and their bounded waits ([core.md](core.md)) |
| `codec` (`src/codec.rs`) | The codecs above the hardware, in plain Rust: H.264's and HEVC's syntax, order counts and DPB over an `Accelerator` that decodes the slice data, AV1's OBUs, headers, tile groups and reference slots over one that decodes the tile data, VP9's superframes, frame headers and reference slots over one that decodes a frame's compressed header and tiles, and the encoders' portable half: settings, GOP, parameter sets, slice headers and configuration records |
| `vaapi` (`src/vaapi.rs`, Linux and Windows) | VA-API through libva, loaded at run time: the decoders and encoders on the codec module, the import check and zero copy ([vaapi.md](vaapi.md)) |
| `nvidia` (`src/nvidia.rs`, Linux) | NVDEC and NVENC over the CUDA driver API, loaded at run time: the decoder of whole access units and the encoder ([nvidia.md](nvidia.md)) |
| `paravirt` (`src/paravirt.rs`) | VideoToolbox's paravirtual protocol, the host's side: the `vtpv` messages an arm64 guest's VideoToolbox sends through the device's `VT_MESSAGE`, the decoder list and capabilities answered from the arm's decoders, the decoder sessions, each on a thread of its own over a backend that lands its pictures in the buffers the guest lends, and the encoder sessions, over a backend that encodes the source frames the guest lends (on VA-API, written by the GPU from a buffer over their pages into an input of the encoder's, memory of the runtime device's that the encoder's display imports, on NVENC into memory CUDA imported, which NVENC registers as input, else read through the host's view of the pages and uploaded), whose samples the session emits ([design.md](../graphics-kext/design.md), "Decided", 17) |

Its subcrate `crates/reims-vgpu-video/conformance` (`reims-vgpu-video-conformance`) is the test
streams' readers and writers, which the examples use: an MP4 track's configuration record and
samples, an MP4 of what an encoder gave, and frames hashed as ffmpeg's `framemd5` does.

reims-vgpu's `graphics` module holds the arms themselves: `graphics/nvidia.rs` and
`graphics/vaapi.rs`, each over its module, and `graphics/video_choice.rs`, which picks one.

## The traits

- **`GraphicsVideo`** (`graphics/decode/video.rs`) is an arm as the device sees it: its caps,
  the layout each entry lands in place at, its sessions, and whether it lands pictures in
  device caches by importing their memory (`lands_in_device_caches`, `imports_device_caches`).
- **`GraphicsVideoSession`** is one session: its shared event, its codec, and a decode
  (`VideoDecode`: the op's access unit and parameter sets, and the output's two planes, each in
  pages or in a device cache) or an encode (`VideoEncode`: the op's settings, the source's two
  planes, and the buffers its access unit and result go in), each of which answers the event's
  value it signals.
- **`HostVideoSession`** (`graphics/video_host.rs`) is both arms' session, as no runtime queue
  carries their work. The arm gives a `HostDecoder`, which starts the
  hardware's work, and a `HostPicture` for each picture, which writes it into its output. The
  session lands the pictures in decode order on a thread of its own
  (`reims-vgpu.graphics-video-land`), copies a plane the arm staged in host bytes into its
  device cache on a runtime queue, and signals each decode's value once its picture is there. It holds no
  more pictures unlanded than the decoder spares (`HostDecoder::most_unlanded`).
- **`HostEncodeSession`** (`graphics/video_host/encode.rs`) is both arms' encode session. The
  arm gives a `HostEncoder`, which checks the settings and takes each frame, and a
  `HostEncodedFrame` for each, which encodes it from the source's host bytes into the
  bitstream buffer. The session finishes the frames in order on a thread of its own
  (`reims-vgpu.graphics-video-encode`), stages a source plane in a device cache in host bytes
  first, writes each frame's result, and signals its encode's value.

## Picking an arm

`graphics_video` (`graphics/video_choice.rs`) asks each arm in turn for the runtime device:
NVIDIA's (NVDEC and NVENC), then VA-API. Each that declines says why in one line at info, and costs no more than
asking: a GPU that is not NVIDIA's loads no CUDA. NVIDIA's is built on Linux alone, VA-API's on Linux
and Windows.

## Tests

- **Unit tests**, in each module, run on any host that builds it (`cargo nextest run -p
  reims-vgpu-video`). The `vaapi` and `nvidia` modules' tests that need their hardware or the
  vendor's headers are ignored; on their hosts,
  `cargo nextest run -p reims-vgpu-video --run-ignored only -E 'test(/^vaapi::/)'` (a VA-API
  host) and `cargo nextest run -p reims-vgpu-video --run-ignored only -E 'test(/^nvidia::/)'`
  (an NVIDIA host) run them: the bindings' layouts against the installed headers, and the
  hardware's own answers. The probes, which print what they measure over inputs a checkout
  does not hold and check nothing else, are ignored tests outside both sets (`src/probes.rs`),
  run by hand.
- **AV1's streams are committed.** `src/codec/av1/tests/streams` holds eleven small streams of
  ours (128 KiB with their references: aomenc 3.15.3, SVT-AV1 4.2.0 and rav1e 0.8.1 from ffmpeg's synthetic sources;
  tiles in several tile groups, film grain, superres, screen content, an odd size with frame ids,
  10 bits, an AVIF still's item with its reduced still picture header, and the 4:4:4 and 12-bit
  profiles the decoder refuses) and each shown frame's MD5,
  ffmpeg's `framemd5` of libdav1d's decode (NV12, P010), made once on a node and committed, so
  that no test needs those tools. The CPU tests read them through the codec module, and truncate
  and fuzz them (`codec::av1::tests`); the ignored `vaapi::va::av1::` and
  `nvidia::annex_b::av1_tests::` tests decode them on the hardware, each frame's MD5 against its
  reference, or reported failed by the hardware (NVDEC's status), which the decoder delivers as
  no frame: a wrong frame delivered as decoded fails the test, on any card. A stream with film
  grain is decoded twice (`codec::av1::tests::judge`): with the grain off, each frame against its
  reference without grain, which every card must match; and as the arm outputs it, each frame
  with the grain the driver applies. That frame is the specification's, its reference with
  grain, or the driver's own synthesis, whose residual (the frame less the frame before grain,
  in each of Y, U and V) has the statistics of the specification's own residual, which
  `tests/streams/aom-grain.grain` holds from libdav1d's decode with and without grain: its mean
  within `GRAIN_MEAN` code values, its standard deviation within `GRAIN_STD` of the
  specification's, and its 32x32 blocks' variances correlated with the specification's by
  `GRAIN_CORRELATION` at least, bounds set from what the cards measured, each with its jobs
  (`codec::av1::tests::card`). The check is shown to fail on grain of other parameters: the
  stream decoded with its grain half as strong, on luma alone, or strong where it is weak, each
  frame refused (`grain_of_other_parameters_is_refused_by_the_grain_check`). radeonsi's and
  NVDEC's grain is the specification's; iHD's is its own ([devices.md](devices.md), "AV1 on
  arm64 guests").
- **The arms' tests** are reims-vgpu's `graphics::` tests, on any host: the host sessions' on
  the fake platform (`video_host`'s tests), and the VA-API arm's on driver answers the tests
  make up (`graphics/vaapi/tests.rs`).

## The conformance suite

The decoders against software references, stream by stream, on the host's hardware:

`crates/reims-vgpu-video/tools/h264-check.sh [render node]` and
`tools/hevc-check.sh [render node]`: each stream's frames, decoded by `examples/h264-framemd5`
and `examples/hevc-framemd5` from its MP4 and its Annex B form, against ffmpeg's software
decode, MD5 by MD5, and the streams the decoder must refuse refused ([vaapi.md](vaapi.md),
"Testing"). The `nvidia` module has no such suite: its decoder is tested by its unit tests and its
ignored hardware tests ([nvidia.md](nvidia.md)).

x264, x265, kvazaar and ffmpeg are test oracles alone: they write the test streams and the
reference MD5s under `build/`, and nothing they write is committed. VP9 has no such suite: the
codec module's VP9 tests carry two streams of their own, committed (`src/codec/vp9/tests`:
ffmpeg's `testsrc2` encoded by libvpx at 640x360, profile 0 and profile 2, 60 and 56 KB), with
vpxdec's MD5 of each frame, which ffmpeg's own VP9 decoder matched frame for frame; each arm's
ignored VP9 tests decode them against those MD5s ([vaapi.md](vaapi.md), "The VP9 decoder";
[nvidia.md](nvidia.md), "Tests"). Each script runs in a potluck
job that holds the render node's GPU (`--gpu`).
