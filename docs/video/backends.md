# The host's video arms

The reims-graphics device's video runs on one of two arms of the host's, NVDEC and VA-API
([devices.md](devices.md), "Which hardware"). This file is how they are put together: the
crates, the traits between them, and how each is tested.

## The crates

| crate | what it is |
|---|---|
| `crates/reims-vgpu-video-core` | What the arms share: the threads and clock the host gives them, their locks, a guest surface's memory as a device reaches it, and their bounded waits ([core.md](core.md)) |
| `crates/reims-vgpu-video-codec` | The codecs above the hardware, in plain Rust: H.264's and HEVC's syntax, order counts and DPB over an `Accelerator` that decodes the slice data, and the encoders' portable half: settings, GOP, parameter sets, slice headers and configuration records |
| `crates/reims-vgpu-video-vaapi` | VA-API through libva, loaded at run time: the decoders and encoders on the codec crate, the import check and zero copy ([vaapi.md](vaapi.md)) |
| `crates/reims-vgpu-video-nvidia` | NVDEC and NVENC over the CUDA driver API, loaded at run time: the decoder of whole access units and the encoder ([nvidia.md](nvidia.md)) |
| `crates/reims-vgpu-video-conformance` | The test streams' readers and writers: an MP4 track's configuration record and samples, an MP4 of what an encoder gave, and frames hashed as ffmpeg's `framemd5` does |

reims-vgpu's `graphics` module holds the arms themselves: `graphics/nvidia.rs` and
`graphics/vaapi.rs`, each over its crate, and `graphics/video_choice.rs`, which picks one.

## The traits

- **`GraphicsVideo`** (`graphics/decode/video.rs`) is an arm as the device sees it: its caps,
  the layout each entry lands in place at, its sessions, and whether it lands pictures in
  device caches by importing their memory (`lands_in_device_caches`, `imports_device_caches`).
- **`GraphicsVideoSession`** is one session: its shared event, its codec, and a decode
  (`VideoDecode`: the op's access unit and parameter sets, and the output's two planes, each in
  pages or in a device cache), which answers the event's value the decode signals.
- **`HostVideoSession`** (`graphics/video_host.rs`) is both arms' session, as no runtime queue
  carries their work. The arm gives a `HostDecoder`, which starts the
  hardware's work, and a `HostPicture` for each picture, which writes it into its output. The
  session lands the pictures in decode order on a thread of its own
  (`reims-vgpu.graphics-video-land`), copies a plane the arm staged in host bytes into its
  device cache on a runtime queue, and signals each decode's value once its picture is there. It holds no
  more pictures unlanded than the decoder spares (`HostDecoder::most_unlanded`).

## Picking an arm

`graphics_video` (`graphics/video_choice.rs`) asks each arm in turn for the runtime device:
NVDEC, then VA-API. Each that declines says why in one line at info, and costs no more than
asking: a GPU that is not NVIDIA's loads no CUDA. Both are built on Linux alone.

## Tests

- **Unit tests**, in each crate, run on any host. The VA-API and NVIDIA crates' tests that need
  their hardware or the vendor's headers are ignored; on their hosts,
  `cargo nextest run -p reims-vgpu-video-vaapi --run-ignored only` (a VA-API host) and
  `cargo nextest run -p reims-vgpu-video-nvidia --run-ignored only` (an NVIDIA host) run them:
  the bindings' layouts against the installed headers, and the hardware's own answers.
- **The arms' tests** are reims-vgpu's `graphics::` tests, on any host: the host sessions' on
  the fake platform (`video_host`'s tests), and the VA-API arm's on driver answers the tests
  make up (`graphics/vaapi/tests.rs`).

## The conformance suite

The decoders against software references, stream by stream, on the host's hardware:

`crates/reims-vgpu-video-vaapi/tools/h264-check.sh [render node]` and
`tools/hevc-check.sh [render node]`: each stream's frames, decoded by `examples/h264-framemd5`
and `examples/hevc-framemd5` from its MP4 and its Annex B form, against ffmpeg's software
decode, MD5 by MD5, and the streams the decoder must refuse refused ([vaapi.md](vaapi.md),
"Testing"). The NVIDIA crate has no such suite: its decoder is tested by its unit tests and its
ignored hardware tests ([nvidia.md](nvidia.md)).

x264, x265, kvazaar and ffmpeg are test oracles alone: they write the test streams and the
reference MD5s under `build/`, and nothing they write is committed. Each script takes its GPU's
locks itself, and none where its caller holds them already (`RVG_GPU_LOCKS_HELD=1`).
