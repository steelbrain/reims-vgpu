# Backends

The devices are generic: core (`reims-vgpu-video-core`) holds everything
the guest sees, behind a few backend traits. Each backend is a crate of
its own that implements them:
- VA-API on Linux (`reims-vgpu-video-vaapi`);
- NVIDIA's NVDEC and NVENC on Linux (`reims-vgpu-video-nvidia`);
- VideoToolbox on macOS (`reims-vgpu-video-videotoolbox`).

There are three backends, and nothing above them: each is its driver's
own interface (libva; libnvcuvid and libnvidia-encode over libcuda;
VideoToolbox), with no FFmpeg or other layer between. Each asks its driver
what it does, and the host offers the guest that and nothing more. The
glue in reims-vgpu picks one for each device. One suite
(`reims-vgpu-video-conformance`) checks every backend the same way.

This file says:
- what a backend is, and which crate implements which part;
- how what a backend reports becomes what the guest is offered;
- how a backend's errors reach the guest;
- the interface each backend crate exports;
- the conformance suite, and how to run it;
- how the glue picks a backend, and what each one loads.

## What a backend is

A backend is core's `Backend` (`crates/reims-vgpu-video-core/src/backend.rs`):
a name for the log, and four parts, each optional.

```rust
pub struct Backend {
    pub name: String,
    pub decoders: Option<Arc<dyn DecoderBackend>>,
    pub encoders: Option<Arc<dyn EncoderBackend>>,
    pub motion: Option<Arc<dyn MotionEstimationBackend>>,
    pub jpeg: Option<Arc<dyn JpegDecoder>>,
}
```

A part a backend lacks is `None`, never an empty list. A host session made
from it (`HostSession::with_backend`) lists no decoder or encoder for a
missing part. The scaler device is no backend's part: it runs on the
library's own GPU kernel on every host ([scaler.md](scaler.md), "The
kernel").
`Backend::none(name)` is a backend with no part, which a host whose backend
does not open gets. `Backend::summary()` is the one line the glue logs:
`VA-API (Mesa Gallium driver ...): decodes H264 [66, 77, 100] up to
4096x4096, ...; encodes ...; decodes JPEG`, and `; estimates
motion up to WxH` where there is an estimator.

### The traits

| part | trait | what core asks of it |
|---|---|---|
| decoders | `DecoderBackend` | `capabilities()`: the codecs, each with its profiles, levels and largest size; `open(DecoderConfig)` |
| one decoder | `BackendDecoder` | `output_format()`, `decode(sample)`, `flush()`, `can_accept(DecoderConfig)` |
| one decoded frame | `DecodedFrame` | `frame_id()`, `pixel_format()`, `width()`, `height()`, `read_back()`: every driver call a copy needs, then its `FrameReadBack::copy_to(planes)`, the CPU's alone, inside the guest buffer's lend ([core.md](core.md), "The copy into the guest's buffer") |
| encoders | `EncoderBackend` (`encoder_backend.rs`) | `encoders()`: the codecs, each with its profiles, largest size, source formats and whether it reorders; `open(EncoderConfig)`; `check_property(codec, key, value)` |
| one encoder | `BackendEncoder` | `configuration()` (the avcC or hvcC record), `encode(SourceFrame)`, `flush()`, `reconfigure(EncoderSettings)` |
| scaler, the scaler device's, apart from `Backend` | `ScalerBackend` (`scaler/backend.rs`) | `create_accelerator()`, at the guest's DRIVER_OK |
| one accelerator | `Accelerator` | `capabilities()`, `transform(TransformJob)` |
| JPEG | `JpegDecoder` (`jpeg.rs`) | `capabilities()`, `decode(JpegDecode, planes)` |
| motion | `MotionEstimationBackend` (`motion.rs`) | `capabilities()`: the source formats and largest size; `open(MotionEstimatorConfig)`: the size, the block size the guest asked for, and whether it asked for true motion |
| one estimator | `MotionEstimator` | `estimate(reference, current)`: a `MotionField`, the backend's own blocks |

The table leaves out the methods with a default: `decodes_while_copying`,
`frames_decoded`, `wait`, `write_to_guest` (the device writes the frame
into the guest's buffer, [core.md](core.md), "Zero copy"),
`session_capabilities` and `prepare`.

The traits carry nothing of any backend's API. There are no VA surfaces,
CUDA contexts or CVPixelBuffers in them. Pixels cross as planes of the
guest's memory, lent by core's `Surfaces`:
- a decoder or JPEG decoder writes into `PlaneDestination`s;
- an encoder reads `PlaneSource`s;
- a scaler reads and writes its job's planes;
- an estimator reads two `MotionPicture`s, each a format and its planes.

A `MotionField` is the backend's: its block size (4, 8 or 16, whatever its
API gives), its columns and rows, and each block's vector in pixels, with
Apple's sign, the reference's position less the current block's. A
backend converts its API's precision and sign to that, and core makes
Apple's grid of it (`motion::resample`; [devices.md](devices.md), "The
motion-estimation session"), so no backend knows Apple's grid.

Each plane has its own bytes per row. `copy_biplanar` is the helper for a
decoded 4:2:0 frame. Bitstream syntax that more than one backend needs
lives in the codec crate (`reims-vgpu-video-codec`): H.264's, HEVC's,
AV1's and VP9's syntax and the part of each decoding process a stateless
accelerator is not given, JPEG's parser, and the encoders' portable half,
their parameter sets, slice headers, GOPs and decode times. It has no FFI.

### Who implements what

| part | VA-API | NVIDIA | VideoToolbox | stand-ins |
|---|---|---|---|---|
| decoders | `VaapiBackend`: H.264, HEVC, AV1, VP9 | `NvidiaDecoders` (NVDEC) | `VideoToolboxBackend` (VTDecompressionSession): H.264, HEVC, AV1 and VP9, where the hardware decodes them | core's `NoDecoders`; the conformance crate's pattern decoder |
| encoders | `VaapiEncoderBackend`: H.264, HEVC | `NvidiaEncoders` (NVENC) | `VideoToolboxEncoderBackend` (VTCompressionSession): H.264, HEVC | core's `NoEncoders` |
| JPEG | `VaapiJpegDecoder` | `NvidiaJpegDecoder` (nvJPEG's hardware decoder), where libnvjpeg is installed; else none | none: the guest is told there is no JPEG engine | |
| motion | none: radeonsi lists no motion-estimation entry point ([vaapi.md](vaapi.md)) | `NvidiaMotion` (NVENC's H.264 ME-only mode) | `VideoToolboxMotionBackend` (VTMotionEstimationSession, macOS 26 and later) | core's tests' and fuzzer's stand-ins; the conformance crate's CPU block search |

The scaler device's `ScalerBackend` is the library's own on every host:
`GpuScalerBackend`, a compute kernel on the GPU the library draws with
([scaler.md](scaler.md), "The kernel"). Core's `ReferenceBackend` stands
in for it on the CPU.

Each backend's own document says what it offers on which hardware:
[vaapi.md](vaapi.md), [nvidia.md](nvidia.md) and
[videotoolbox.md](videotoolbox.md).

## What the guest is offered

Core builds every answer from what the backend reports, in Apple's shapes
(`crates/reims-vgpu-video-core/src/host/codec_lists.rs`;
[devices.md](devices.md), "How the guest finds the host's codecs"). A
backend reports facts, and core alone decides what they look like to the
guest.

- **`decs`** lists one entry per codec in `DecoderBackend::capabilities()`,
  in its order:
  - ID `reims-vgpu-video.videodecoder.{avc,hevc,av1,vp9}`, rating 400 (as
    Apple's AVD decoders), and `VTCodecType`;
  - for H.264 and HEVC, `VTDecoderCapabilities`: the profiles, and each
    one's `VTMaxDecodeLevel`. That level is the backend's `max_level`, or
    Apple's own level where the backend knows no limit (80 for H.264, 186
    for HEVC), as VA-API knows none;
  - for AV1 and VP9, no capabilities, as Apple's AV1 and VP9 entries have
    none.
- **`dcap`** answers per profile: `VTIsHardwareAccelerated` true, and the
  decode and playback levels. AV1 has a shape of its own, with Apple's
  level 19 where the backend knows no limit, and VP9 and an unlisted codec
  answer NULL.
- **`+dec`** for a codec the backend does not list, or a `,cid` naming
  another decoder, gets a session whose reply says -12906, as Apple's
  selection finds none. The decoders' `max_width` and `max_height` are
  for the log and the backend's own refusals; core checks no decoder size
  itself.
- **`encs`** lists one entry per codec in `EncoderBackend::encoders()`: the
  codec type, rating 400, the codec's name, our name and our ID.
  - `+enc` takes our ID, or Apple's AVE ID as the guest's fixed encoders
    send it, as `,cid`. For an unlisted codec or another ID it gets -12908.
  - The profiles become `ProfileLevel`'s values.
  - `source_formats` become `Epba`'s `PixelFormatType`, in order.
  - `frame_reordering` says whether `AllowFrameReordering` gives B-frames.
  - `check_property` answers each `eset` before the session keeps it.
  - An `esta` past `max_width` or `max_height` is -19354, as Apple's host
    answers a size its encoder does not take.
- **`meps`** lists one processor, ours, where the backend has a motion
  part, and none where it has not.
  - `MotionEstimationCapabilities::source_formats` become `Mspa`'s
    `PixelFormatType`, in order.
  - A `mest` past `max_width` or `max_height`, or past Apple's sizes, is
    -12902, as Apple's processor answers a size it does not take.
- **The scaler's type 1 request** writes `Accelerator::capabilities()` as
  its 0x288 bytes (`scaler/capabilities.rs`). Where the host has no GPU
  for the kernel, the scaler device is made all the same, and refuses the
  guest's features, so the guest's kext gives it up before DRIVER_OK
  ([scaler.md](scaler.md), "A host without a GPU").
- **`jcap`** writes `JpegCapabilities` as `,jca`. With no JPEG part,
  `jcap`, `jdec` and `jenc` answer -16994, as CMPhoto's host does without
  the hardware.

## How errors reach the guest

A backend reports a failure as core's `BackendError`, and core gives the
guest the status Apple's host would:

| error | from a decoder | from an encoder | from an estimator |
|---|---|---|---|
| `Unsupported` | -12910 (`kVTVideoDecoderUnsupportedDataFormatErr`) | -12902 (`kVTParameterErr`); -12900 from `reconfigure`, for an `eset` | -12902 |
| `BadData` | -12909 (`kVTVideoDecoderBadDataErr`) | -12902 | -12902 |
| `Malfunction` | -12911 (`kVTVideoDecoderMalfunctionErr`) | -12912 (`kVTVideoEncoderMalfunctionErr`) | -17691, the status Apple's processor emits a frame with once it is invalidated |
| `SkippedLeadingPicture` | -17694 (`kVTVideoDecoderReferenceMissingErr`), as Apple's HEVC decoder drops a RASL picture it skips ([devices.md](devices.md), "HEVC") | a decoder's alone | a decoder's alone |
| `Native { status, .. }` | `status`, as it is | `status`, as it is | `status`, as it is |

- **`Native`** is the backend API's own status. Apple's host sends its
  codec's own status for `esta` and `efrm` (read from Apple's host). So a
  backend whose API's statuses the guest's VideoToolbox knows passes them
  through: VideoToolbox's does, for every VideoToolbox, CoreMedia and
  CoreVideo call. VA-API's and NVIDIA's statuses mean nothing to a guest,
  so those backends use the other kinds. A `Native` status of 0 or more is
  taken as a malfunction.
- **Where each one lands:**
  - a decoder that does not open fails `dsta`;
  - a frame that does not decode is emitted with the status and no image;
  - a frame that does not copy is emitted with the status and its buffer
    let go;
  - an encoder that does not open fails `esta`;
  - a frame the encoder does not take, an `Err` from its `encode`, fails
    `efrm`, and is never emitted;
  - a frame the encoder takes and then fails, given back with an error
    and no data, is emitted with the status, info flags 0 and no sample,
    after an `efrm` reply of 0, and the frames given back with it keep
    their own samples, as Apple's host forwards each emit with its own
    status (read from Apple's host);
  - an estimator that does not open fails `mest`;
  - a pair of frames that does not estimate is emitted, `Memt`, with the
    status and no vector buffer, after a `mepf` reply of 0;
  - a frame an encoder drops, given back with no data, is emitted with
    `kVTEncodeInfo_FrameDropped` and no sample, as Apple's host forwards
    one.
- **`check_property`** returns the status itself, which goes as it is.
- **The scaler.**
  - `create_accelerator` failing is the device's needs-reset at DRIVER_OK,
    as Apple's host sets it.
  - `transform` returns the request's IOReturn, which goes as it is.
- **JPEG.** core maps `JpegError` to Apple's IOReturns (`host/jpeg.rs`):

  | error | IOReturn |
  |---|---|
  | `BadData`, `UnsupportedStream`, `SizeMismatch` | kIOReturnBadMedia, 0xe00002d1 |
  | `SubsamplingMismatch`, `ScanEndsEarly`, `Malfunction` | kIOReturnDeviceError, 0xe00002e9 |
  | `UnsupportedFormat` | kIOReturnError, 0xe00002bc |
  | `UnsupportedScale` | kIOReturnUnsupported, 0xe00002c7 |
- **A backend that does not open** is `Backend::none`. The guest is
  offered nothing and decodes in software.

## The interface each backend crate exports

Each backend crate exports a name and an opener at its root, and the glue
and the crate's `conformance` example reach the backend through the opener
alone:

| crate | name | opener |
|---|---|---|
| `reims-vgpu-video-vaapi` (Linux) | `pub const NAME: &str = "VA-API"` | `pub fn open_backend(path: impl AsRef<Path>, runtime: Arc<dyn Runtime>) -> Result<Backend, Error>`: every part on one VA display of the render node (`src/va/whole.rs`) |
| `reims-vgpu-video-nvidia` (Linux) | `pub const NAME: &str = "NVIDIA"` | `pub fn open_backend(gpu: &str, runtime: Arc<dyn Runtime>) -> Result<Backend, Error>`: a CUDA ordinal, a PCI bus ID or a DRM node (`src/backend.rs`) |
| `reims-vgpu-video-videotoolbox` (macOS) | `pub const NAME: &str = "VideoToolbox"` | `pub fn backend(runtime: Arc<dyn Runtime>) -> Backend`, which always opens, with no part where the Mac lists none (`src/macos/whole.rs`) |

What each opener does:
- it asks the driver what each part offers, once, at the opening, but for
  VideoToolbox's encoders, which are listed whatever their state and read
  once VideoToolbox answers for them;
- it leaves out a part that offers nothing;
- it names the backend `NAME`, with the device where there is one:
  `VA-API (<driver>)`, `NVIDIA (<GPU>, <bus ID>)`;
- the threads its parts start and the clock they time and bound their
  waits by are `runtime`'s (core's `runtime`): the glue gives its
  platform, and the examples std's runtime, `StdRuntime::shared()`;
- a failure's error has `Display`, which the glue logs.

Every crate builds on every host and offers nothing off its own:
- the vaapi crate's opener is Linux's alone;
- the nvidia crate loads its libraries at run time, so it builds
  anywhere;
- the videotoolbox crate's items are macOS's.

Each crate also has `examples/conformance.rs`, which runs the suite on that
backend:

```rust
reims_vgpu_video_conformance::cli::run("vaapi-conformance", &args, &|device| {
    reims_vgpu_video_vaapi::open_backend(device, StdRuntime::shared()).map_err(|e| e.to_string())
})
```

It needs a dev-dependency on `reims-vgpu-video-conformance`. Each of the
three plugs in the same way: it implements the parts it has, exports `NAME`
and an opener, has that example, and has an arm in the glue's
`backend_choice.rs`.

## The conformance suite

`crates/reims-vgpu-video-conformance` holds a stand-in for the guest. It
does what Apple's guest frameworks do, as read from Apple's code, and
drives core's devices on any `Backend`, message by message, as a guest
would. It checks what comes back against references made elsewhere. What
the guest sees is core's, so a backend passes when it does what it lists.

| command (`conformance DEVICE ...`) | module | what it checks |
|---|---|---|
| `summary` | | the backend opens; its summary line |
| `framemd5 STREAM [--async] [--description WxH [--hash WxH]] [--sequence FILE]` | `decode` | each frame's MD5, as `ffmpeg -f framemd5` hashes NV12 or P010; with `--sequence`, the host's messages in the order they came, to FILE |
| `decode-check STREAM REFERENCE [STREAM REFERENCE]...` | `decode`, `suite` | every frame of each stream matches its reference, a software decode's hashes, compared as a set, synchronous and asynchronous; an empty reference fails |
| `encode INPUT WxH PREFIX [OPTION]...` | `encode` | nothing: raw frames encoded through the host session, and what the guest got written to PREFIX.mp4, PREFIX.264 or .265 and PREFIX.log |
| `encode-check [--case P]...` | `encode`, `encode_check` | for each listed codec: each source depth with reordering and without, 641x361, full range, 1080p, 2160p or its refusal, a size past the largest, forced key frames, the rate controls, changes by `eset` mid-stream, `Quality`, and AVE's answers to edge values (below) |
| `jpeg-check [OPTION]... FILE.jpg` | `jpeg` | a JPEG through the JPEG part, against libjpeg's and ffmpeg's planes; a decode with no `--reference` fails; a stream outside what the part answers it decodes (`decodes`) passes refused as unsupported, and fails decoded |
| `motion-check [--case P]...` | `motion::check` | eight moved textures, at 16x16 and 4x4 blocks and four sizes, through the motion session: Apple's grid, its padding cells (0, 0), and at least 70% of the picture's cells the way back, within 2 pixels at 16 (Apple's own steps are 4) and exactly at 4 |

The suite's scaler cases (`scaler`) take a `ScalerBackend`, not a
`Backend`: the library's kernel runs them on each GPU in its tests
([scaler.md](scaler.md), "The kernel").

`encode-check` makes its cases from what the backend lists, so that every
backend is held to the same ones where it lists the same. For each codec:

| case | what it adds to the checks below |
|---|---|
| `-420v`, `-x420`, each `-reordered` where the encoder reorders | the first 8-bit and 10-bit formats, 640x360, 60 frames at 3 Mbit/s, a key frame every 30 |
| `-641x361` | an odd size, encoded at the even size below |
| `-420f`, `-xf20` | full-range formats, on a picture that reaches 0 and 255: the SPS's `video_full_range_flag` is 1, and the decoded luma goes below 16 and above 235 |
| `-1920x1080`, `-3840x2160` | large sizes, reordered where the encoder reorders; 2160p only where listed, else `-3840x2160-refused` |
| `-WxH-refused` | `esta` past the largest size listed, and 2160p where it is not: -19354 and no `Epba`, as Apple's host refuses a size ([devices.md](devices.md), "The encoder session") |
| `-force` | `ForceKeyFrame` on frames 17 and 44, reordered where it can be: each holds an IDR or IRAP picture, by its NAL unit types |
| `-abr`, `-limits`, `-abr-limits` | 10 s of a panning texture at 720p, at `AverageBitRate`, `DataRateLimits` or both: the average within a quarter of its target, and every second's bits within 5% of the limit, a hard one in VideoToolbox's header. Each target is a share of what the backend's encoder makes of the frames with no target, measured first, so that it is one the content reaches: a rate check on a picture cheaper than its target checks the picture, not the encoder |
| `-limits-keys` | `DataRateLimits` alone on a ramp and a box that move, bars, and noise that scrolls by whole pixels, a key frame a second: the noise is dear within a picture and free to predict, so the key frames cost most, and every second holds one at its most. The limit's checks, and 20 dB PSNR, since the noise in a sixth of the picture costs more than the limit allows |
| `-rate-change` | `AverageBitRate` raised by `eset` at frame 150: the stream follows the new rate after a second, or keeps the first, as Apple's AVE keeps it when frames come faster than real time; the line says which |
| `-keyint-change` | `MaxKeyFrameInterval` from 60 to 10 by `eset` at frame 45: the new interval from the first key frame after the change, or the first kept throughout, as AVE keeps it; the line says which |
| `-quality` | `Quality` 0.25 then 0.75: the higher gives more bytes and a better PSNR, or `Quality` is refused with -12900, as VideoToolbox refuses a property its encoder lacks |
| `-property-answers` | 19 `eset`s of edge values, each answered as Apple's AVE answers it: -12900 for those out of its ranges ([devices.md](devices.md), "The encoder session") |

Each encode is checked for:
- every `eset` answered 0, but where the case expects a refusal;
- every `efrm` answered 0 with no info flags, and every frame emitted
  once by its ID, with status 0, encode info flags 0 and a sample of its
  own PTS;
- a configuration record whose SPS has the size, rounded down to even
  where it is odd ([devices.md](devices.md), "The encoder session"), a
  listed profile, and the source's range;
- PTSes that are the frames';
- each sample's DTS the PTS of the frame sent in its place, as Apple's
  encoders give them ([devices.md](devices.md), "The encoder session"), so
  a B picture's is after its PTS;
- the samples reordered no further than the SPS's
  `max_num_reorder_frames` or `sps_max_num_reorder_pics` says, where it
  says;
- sync exactly on IDR or IRAP pictures;
- key frames no further apart than asked.

It then decodes the samples back on the backend's decoders, and needs 30
dB PSNR of every frame against its source, the source's top left at an
odd size, 25 dB for the texture and 20 for the scrolling noise; a backend
with no decoder of the codec fails, its pictures unchecked. In
`-property-answers` the `eset`s every case makes before its own must be
answered 0 too. Each
case's lines give its sync frames, how far it reordered, its rates
against their targets, and its first samples' times in frames, `PTS/DTS`
in decode order, so that backends can be compared by them.

**Running it.** Each backend crate's conformance example
(`examples/conformance.rs`, named `vaapi-conformance`, `nvidia-conformance`
or `videotoolbox-conformance`) runs the suite's commands on that backend,
with DEVICE as its opener takes it: a render node for VA-API; a CUDA
ordinal, a PCI bus ID or a DRM node for NVIDIA; `hardware` for
VideoToolbox.

```sh
cargo run --release -p reims-vgpu-video-vaapi --example vaapi-conformance -- /dev/dri/renderD128 encode-check
cargo run --release -p reims-vgpu-video-nvidia --example nvidia-conformance -- 0 motion-check
cargo run --release -p reims-vgpu-video-videotoolbox --example videotoolbox-conformance -- hardware summary
```

- Each command prints a line a case or run, and exits 1 when any fails
  or when the backend lacks the part.
- `encode-check` and `motion-check` draw their own
  pictures. The streams `framemd5` and `decode-check` take are not
  committed: the vaapi crate's codec checks
  (`crates/reims-vgpu-video-vaapi/tools/{h264,hevc,av1,vp9}-check.sh`, on
  Linux) write them and their references, under `RVGV_H264_OUT`,
  `RVGV_HEVC_OUT`, `RVGV_AV1_OUT` and `RVGV_VP9_OUT`.
- No one script runs the whole suite. The vaapi crate's
  `tools/host-session-check.sh` (every stream the codec checks left with a
  reference) and `tools/jpeg-check.sh` (the JPEG cases; another backend's
  conformance example goes in `RVGV_JPEG_CHECK`) run its parts on VA-API:
  `vaapi-conformance`'s `framemd5` and `jpeg-check`. The suite's `encode`
  command is its encoder, which no script runs. The videotoolbox crate's
  `tools/videotoolbox-check.sh` runs its tests and `decode-check`
  ([videotoolbox.md](videotoolbox.md), "Checks"). NVIDIA has no script.
- The suite's own tests (`cargo nextest run -p
  reims-vgpu-video-conformance`, part of `cargo xtask check`) open no
  driver: they run it on the pattern decoder, core's CPU scaler reference
  and a CPU block search, and check `encode-check`'s counting of sync
  samples, decode times, reordering and rates.

**Runs.** Measured with the suite's commands, on an RX 7900 XTX (radeonsi,
Mesa 26.2.4) at renderD128 and a Radeon 780M at renderD129 in one host,
an RTX 5080 Laptop GPU (driver 615.71.09), and the M3 Max (macOS 27.0.1):

| backend | where | when | result |
|---|---|---|---|
| VA-API | the RX 7900 XTX and the Radeon 780M | 2026-10-24 | each GPU: decode 80 of 80 (8 H.264, 10 HEVC, 10 AV1, 12 VP9, each both ways); the backend's scaler, since removed, 106 of 106 cases, 19 refused; JPEG 124 of 124 lines over both GPUs |
| NVIDIA | the RTX 5080 Laptop GPU | 2026-10-24 and 25 | decode through the suite's stand-in guest, [nvidia.md](nvidia.md), "Tests"; JPEG 102 of 102 cases, with libnvjpeg 13.2.3 ([nvidia.md](nvidia.md), "The JPEG engine") |
| VideoToolbox | the M3 Max | 2026-10-24 and 25 | decode, and the backend's scaler, since removed, [videotoolbox.md](videotoolbox.md), "Checks"; no JPEG part to check. No guest has run on this backend ([bugs.md](bugs.md), T11) |

`encode-check`, with every case above, `-limits-keys` included
(2026-10-24 and 25):
- VideoToolbox, the M3 Max: 35 of 35;
- VA-API, the RX 7900 XTX and the Radeon 780M: 33 of 33 each;
- NVIDIA, the RTX 5080 Laptop GPU: 35 of 35.

`motion-check` passed 8 of 8 on VideoToolbox (the M3 Max) and on NVIDIA
(the RTX 5080 Laptop GPU), 2026-10-24 ([videotoolbox.md](videotoolbox.md)
and [nvidia.md](nvidia.md), "Motion estimation"). VA-API has no motion
part to check.

## Picking a backend

The glue picks the video device's backend in one place:
`crates/reims-vgpu/src/video/backend_choice.rs`. The scaler device picks
none: it transforms with the library's own kernel on the GPU the library
draws with ([scaler.md](scaler.md), "The kernel").
Nothing configures it: no QEMU property, no field of the devices' configs
and no environment variable names a backend or a GPU. The host's choice of
GPU is the one it makes for the GPU device.

**Which GPU.**
- **macOS:** VideoToolbox, the host's own. There is nothing to choose.
- **Linux:** the GPU the library's GPU device runs on. That is the Vulkan
  runtime's system default device's, among the devices the Vulkan loader
  shows, so a VM that reims-ctl pins to a GPU through the loader's
  environment (`VK_DRIVER_FILES`, and `VK_LOADER_DEVICE_ID_FILTER` where
  one driver drives several GPUs) decodes and encodes on that GPU too.
  - The runtime answers the identity of the physical device it makes its
    default device of, without making a device
    (`Device::system_default_identity_with`, the same selection over the
    same devices), with its PCI address (`VK_EXT_pci_bus_info`) and its DRM
    render node (`VK_EXT_physical_device_drm`), as the device made of it
    answers them. reims-vgpu-vulkan's `host_gpus` takes both from it, the
    render node from sysfs by the PCI address where the driver names none,
    so the answer does not depend on whether a GPU device was made first,
    or at all.
  - On that GPU: NVIDIA's NVDEC and NVENC where it is an NVIDIA GPU (PCI
    vendor 0x10de) and CUDA opens the GPU of its PCI bus ID; VA-API on its
    render node otherwise, an NVIDIA GPU's among them where NVIDIA's
    backend does not open.
- **What the device needs** (`Need`): a backend with any part.
- **Where that GPU will not do:** a CPU implementation (lavapipe or
  llvmpipe, `VK_PHYSICAL_DEVICE_TYPE_CPU`), or a GPU whose backend has no
  part. The device then picks a hardware GPU itself: the first of the
  host's hardware GPUs whose backend has one, discrete ones
  first, then integrated, virtual and other ones, each kind in the loader's
  order. The host's hardware GPUs are read on a loader instance that loads
  every driver the loader's own Linux search finds
  (`VK_LUNARG_direct_driver_loading`, exclusive), so a VM pinned to
  lavapipe still finds them; CPU implementations are left out. That
  instance is reims-vgpu-vulkan's own, as the runtime lists the GPUs of
  its own loader environment alone ([bugs.md](bugs.md), U18), and it is
  made only here, where the GPU device's GPU will not do.
  - Measured with the Vulkan loader 1.4.363 on the RX 7900 XTX and Radeon
    780M host (2026-10-25): such an instance lists both GPUs and lavapipe
    under `VK_DRIVER_FILES` naming lavapipe's manifest alone, and only the
    GPU `VK_LOADER_DEVICE_ID_FILTER` names where that is set.
- **Where no GPU will do,** the device has no backend, and says so in a
  warning: the video device offers the guest no codec, and the guest's
  VideoToolbox decodes and encodes in software.
  - The device's unit tests and the C API's make such a device whatever the
    host has (`VideoDevice::without_backend`, built for the tests alone).
    They run on core's testkit platform, whose clock is virtual, and a
    backend waits for its GPU on its device's clock, at its opening too:
    VA-API's JPEG decoder for its conversions' trial runs, NVIDIA's for its
    trial decode. No test advances the clock meanwhile, so a backend opened
    there would wait for good. The tests that open the host's backend, on
    macOS, open it on std's runtime, and the C harness makes its device
    through `rvg_video_device_create`, on the host's backend and clock.
- **One backend per place.** A backend is open once per render node
  (VA-API), once per GPU (NVIDIA's), VideoToolbox once per host, and every
  device that picks it there shares it. A device keeps its share until it
  is destroyed, so the backend is released with the last device that
  holds it, in either order, and a device made after that opens it again.
  A backend that did not open, or opened with no part, is not kept, and
  the next device tries again. `backend_choice.rs`'s tests hold each of
  these, and the choice itself on stand-ins for a host's GPUs.
  Measured on the RX 7900 XTX and Radeon 780M host (2026-10-24), when the
  scaler device took its backend here too, with both devices realized
  paused in one QEMU and no guest: both devices on renderD128 opened VA-API
  once. The second device's line names the one it shares:

  ```
  scaler device backend VA-API (the GPU device's GPU, ...): the one open for the video device, VA-API at /dev/dri/renderD128: ...
  ```

  A boot with both devices (a macOS 26.7.1 guest on the RX 7900 XTX,
  2026-10-24) logged the same one open, and the guest scaled and decoded
  through the shared backend: a scaler probe passed 31 of 31 and 4 of 4
  cases, and both H.264 streams decoded 150 of 150 frames matching.
- Each device logs its choice and the reason in one line, at info:

  ```
  video device backend VA-API (the GPU device's GPU, AMD Radeon RX 7900 XTX (RADV NAVI31) (discrete, 0000:03:00.0, /dev/dri/renderD128)): VA-API (Mesa Gallium driver ...): decodes ...
  video device backend VA-API (the GPU device's GPU, llvmpipe (LLVM ...) (CPU), is a software rasterizer: the first hardware GPU that offers video, discrete ones first, AMD Radeon RX 7900 XTX (RADV NAVI31) (discrete, ...)): ...
  ```

**Measured** (2026-10-25), both devices realized paused on q35 under KVM
beside `apple-gfx-pci` and `apple-gfx-iosurface-pci`, with no guest, when
the scaler device took its backend here too, before the kernel:
- On the RX 7900 XTX and Radeon 780M host, with `VK_DRIVER_FILES` naming
  RADV's manifest: `VK_LOADER_DEVICE_ID_FILTER=0x744c` put both
  devices on the RX 7900 XTX's renderD128 and `0x15bf` on the Radeon
  780M's renderD129, the scaler device sharing the video device's VA-API
  each time; without `apple-gfx-pci`, `0x15bf` still gave renderD129. With
  `VK_DRIVER_FILES` naming lavapipe's manifest alone, both logged llvmpipe
  as a software rasterizer and took the RX 7900 XTX.
- On the RTX 5080 Laptop GPU host (driver 615.71.09), the loader's
  environment as users have it: the video device took NVIDIA's backend on
  0000:01:00.0. The scaler device found no video processing in NVIDIA's
  backend or in VA-API's on the same GPU, whose driver refuses
  `vaQueryVideoProcPipelineCaps` ([bugs.md](bugs.md), U16), had no other
  GPU, and was made without a backend; with the kernel it transforms on
  that GPU ([scaler.md](scaler.md), "A host without a GPU"). With
  lavapipe's manifest alone, the video device took NVIDIA's backend on the
  same GPU as the first hardware GPU.

**What each device takes from it.**
- The video device takes the decoders and encoders, each wrapped in the
  glue's logging (`video/logged.rs`), which passes every method of
  core's traits on, defaults included, so that the session gets each
  backend's own answers, and the motion part as it is.
- It takes the JPEG part too, and logs its capabilities, or that there
  is none. VA-API's comes on the decoders' display, and the vaapi crate
  logs its samplings and largest size.

**What each backend loads.**
- VA-API loads `libva.so.2` and `libva-drm.so.2` with dlopen when its
  first display opens (the vaapi crate's `src/va/libva.rs`), and links
  neither, so a host without them runs: there VA-API does not open, and
  the device says why. Measured on the RX 7900 XTX and Radeon 780M host
  with both hidden in a mount namespace of the run's own (2026-10-24):
  QEMU starts, the video device is offered nothing ("cannot load
  libva.so.2: ..."), the scaler device, which then took its backend here,
  refused the guest's features, and a VM with neither device boots to its
  firmware.
- NVIDIA loads `libcuda.so.1`, `libnvcuvid.so.1` and
  `libnvidia-encode.so.1` with dlopen when it opens, so a host without
  them runs, and `libnvjpeg.so.13` (or `.so.12`), a CUDA toolkit library
  the driver does not install, for its JPEG engine alone: without it the
  backend opens with no JPEG part, and says so once at info
  ([nvidia.md](nvidia.md), "The JPEG engine").
- VideoToolbox links Apple's frameworks, on macOS alone.

**Evidence.** Boots of an x86_64 macOS 26.7.1 guest with both devices
and both kexts, the loader's environment as reims-ctl builds it
(2026-10-25), when the scaler device took its backend here too, before
the kernel:
- pinned to the Radeon 780M (`VK_LOADER_DEVICE_ID_FILTER=0x15bf`), both
  devices logged VA-API on renderD129, the GPU device's GPU, and to the RX
  7900 XTX (`0x744c`) on renderD128, the scaler device sharing the video
  device's backend each time. On each, H.264, HEVC Main and Main 10, VP9
  profiles 0 and 2 and AV1 at 8 and 10 bits decoded in hardware, every
  frame matching its reference, H.264 and HEVC encoded, the scaler probe
  passed 31 of 31 and 4 of 4 cases, and the JPEG engine decoded 15 files
  and refused a progressive one;
- on the RTX 5080 Laptop GPU, the video device logged NVIDIA's backend and
  the same streams decoded and encoded alike, while the scaler device
  refused the guest's features and the guest did without it.
