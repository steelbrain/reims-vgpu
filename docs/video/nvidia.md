# The NVIDIA backend

`crates/reims-vgpu-video-nvidia`: the host's codecs on an NVIDIA GPU,
through NVDEC and NVENC over the CUDA driver API, with nothing above them.
It is core's third backend beside VA-API's and VideoToolbox's: core's
decoder and encoder backends, a motion part and, where libnvjpeg is
installed, a JPEG engine on nvJPEG's hardware decoder; no scaler.
This doc says how it binds the driver, what an RTX 5080 Laptop GPU
answers, what the guest is offered, how each codec is fed, how it is
tested and what it does not do. Every fact names its run; runs are on an
NVIDIA GeForce RTX 5080 Laptop GPU (driver 615.71.09, CUDA driver 13.4,
on CachyOS) unless they say otherwise.

## What is here

| what | where |
|---|---|
| Loading the driver's libraries, and the function tables | `src/dl.rs` |
| The CUDA driver API: a GPU, its primary context, copies | `src/cuda.rs`, `src/cuda/ffi.rs` |
| NVDEC's binding: caps, decoder, pictures, mapping, status, reconfiguration | `src/nvdec.rs`, `src/nvdec/ffi.rs` |
| An NVDEC decoder and its pictures: surfaces, retargeting, the copy out | `src/nvdec/decoder.rs` |
| The codec crate's accelerators on NVDEC, one a codec | `src/nvdec/h264.rs`, `hevc.rs`, `vp9.rs`, `av1.rs` |
| Core's decoder backend: the listing and the sessions | `src/decode.rs` |
| NVENC's binding, and an encoder of raw frames | `src/nvenc.rs`, `src/nvenc/ffi.rs`, `src/nvenc/encoder.rs` |
| Core's encoder backend over the codec crate's encode session | `src/encode.rs` |
| Core's motion part on NVENC's ME-only mode ("Motion estimation") | `src/motion.rs`, `src/nvenc/motion.rs`, `src/nvenc/motion_ffi.rs` |
| Core's JPEG decoder on nvJPEG's hardware decoder ("The JPEG engine"): libnvjpeg's binding, the stream it is given, what each decode asks of it, the CPU's resampling and conversions | `src/jpeg.rs`, `src/jpeg/nvjpeg.rs`, `src/jpeg/stream.rs`, `src/jpeg/plan.rs`, `src/jpeg/convert.rs` |
| The backend core opens, by GPU | `src/backend.rs` |
| The binding checked against NVIDIA's headers | `src/layout.rs` (an ignored test) |
| What the GPU answers, as `vainfo` would | `src/probe.rs`, `examples/nvidia-probe.rs` |
| Raw NVENC, and NVENC through core's encoder session | `examples/nvidia-encode.rs`, `examples/nvidia-session-encode.rs` |
| A decode through the backend at each frame's own size; one with damaged samples | `examples/nvidia-framemd5.rs`, `examples/nvidia-damage.rs` |
| How long the backend takes to open a decoder and an encoder | `examples/nvidia-open-time.rs` |
| The conformance suite on this backend | `examples/conformance.rs` |
| Its decode check of every stream in a directory with a reference beside it ("Decode") | `tools/decode-check.sh` |
| Zero copy: frames copied by the GPU into the guest's pages ("Zero copy") | `src/zero_copy.rs` |
| The waits on the GPU, each bounded ("How a session waits for the GPU") | `src/wait.rs`, `src/nvenc/locks.rs`, `src/nvdec/decoder.rs` |
| The reims-graphics device's decode on NVDEC: whole access units in place, each picture copied into the output's planes by the GPU ("The reims-graphics device") | `src/annex_b.rs`; reims-vgpu's `src/graphics/nvidia.rs`, `src/graphics/video_host.rs` |

## The binding

libcuda.so.1, libnvcuvid.so.1 and libnvidia-encode.so.1 are opened with
`dlopen` when the backend is opened, so a host with no NVIDIA driver runs
without them: opening the backend there fails, and nothing else does. No
crate binds them; the declarations are ours, for the subset used, and no
NVIDIA parser is used: the codec crate parses every stream, as for VA-API.

- **Checked against NVIDIA's headers.** `src/layout.rs`'s test
  (`cargo test -p reims-vgpu-video-nvidia layout -- --ignored`) writes a C
  program from the binding's tables, builds it against ffnvcodec-headers
  13.1.15.0 in `/usr/include/ffnvcodec`, and compares what C sees with
  what Rust declares: every struct's size and offsets, every bit-field,
  constant, GUID and function signature. It last passed on 2026-10-24,
  with `CUVIDRECONFIGUREDECODERINFO` and `cuvidReconfigureDecoder` among
  its checks. The C bit-fields are words in Rust, with each field's shift
  and width (`nvdec::ffi::bits`), in GCC's x86-64 order, which the test
  checks too.
- **NVENC API 13.1.** `NvEncodeAPIGetMaxSupportedVersion` answers 13.1;
  every versioned struct is declared at 13.1, and NVENC refuses a wrong
  size or version.
- **Threads.** The decoder is made with `cudaVideoCreate_PreferCUVID` and
  no `vidLock`: each call on a decoder holds that decoder's mutex with the
  GPU's primary context pushed, so a guest's copy on one thread and its
  decodes on another do not race. Each push is popped by a guard on the
  thread that pushed it, which cannot leave that thread (`cuda::Current`).
- **Handles.** Each is released once, on every path, a panic's unwinding
  included: a mapped picture is unmapped by a guard (`MappedFrame`,
  `nvdec/decoder.rs`), and an NVENC buffer's lock undone by one (`Locked`,
  `nvenc.rs`). An NVENC session keeps the buffers it makes, which their
  owner names by index (`nvenc::Buffer`), and destroys them before itself;
  every call that passes one to the driver is the session's, from the
  buffer's own handle, size and format. Its initialisation and
  configuration go after it, the configuration kept at the one address
  NVENC was given, which a reconfiguration changes in place. An NVDEC
  decode is given its
  bitstream and slice offsets by borrow, each offset checked within the
  bitstream, so that no other memory reaches the driver as either. Memory
  the driver hands over
  uninitialised is zeroed before it is read (the staging memory), or only
  written (NVENC's input buffers). Nothing of the driver's calls back into
  Rust: no parser, no NVENC events.
- **libnvjpeg** is a CUDA toolkit library, not one of the driver's, and is
  loaded only for the JPEG engine ("The JPEG engine"). Its declarations
  are written from `nvjpeg.h` 13.2.3.58, which no package on the hosts
  installs, so the layout test does not check them: each function takes
  `int`-sized enums, sizes and pointers, and `nvjpegImage_t` is the one
  struct, whose size is asserted.
- **Not in the headers.** `cuMemHostRegister_v2` and `cuMemHostUnregister`,
  for zero copy, are not in ffnvcodec's `dynlink_cuda.h`, so the layout
  test does not check them; their signatures are the CUDA driver API's
  (`void *`, a size, flags; `void *`), and libcuda 615.71.09 exports both
  (`nm -D`, 2026-10-25). A driver without them is lent no guest buffer,
  and the log says so once (`src/cuda.rs`).

## What the RTX 5080 Laptop GPU answers

`nvidia-probe` (2026-10-24), the parts the backend uses:

| codec | chroma, depth | sizes | output |
|---|---|---|---|
| H.264 | 4:2:0 8-bit | 48x64 to 8192x8192 | NV12 |
| HEVC | 4:2:0 8, 10, 12-bit | 144x144 to 8192x8192 | NV12; P016 above 8 |
| VP9 | 4:2:0 8, 10, 12-bit | 128x128 to 8192x8192 | NV12; P016 above 8 |
| AV1 | 4:2:0 and monochrome, 8 and 10-bit | 128x128 to 8192x8192 | NV12; P016 at 10 |
| JPEG | 4:0:0 to 4:4:4, 8-bit | 64x64 to 32768x16384 | NV12 |

Each has 2 decode engines but JPEG (1). The JPEG engine is nvJPEG's, not
NVDEC's ("The JPEG engine"). NVDEC also lists H.264 at 10 bits
and 4:2:2, HEVC 4:2:2 and 4:4:4, MPEG-2 and VP8, which core has no format
or codec for.

NVENC (`NvEncGetEncodeCaps`): H.264 Baseline, Main, High and High 10 among
others, 145x49 to 4096x4096, 7 B-frames; HEVC Main, Main 10 and the range
extensions, 129x33 to 8192x8192, 5 B-frames; AV1 Main. Presets P1 to P7, 2
encode engines, CBR, VBR and constant QP, lookahead (`SUPPORT_LOOKAHEAD`).

## What the guest is offered

From the caps, asked once when the backend opens:
- decoders: H.264 profiles 66, 77 and 100 where NVDEC decodes 8-bit 4:2:0;
  HEVC Main and Main Still Picture at 8 bits, Main 10 at 10; VP9 Profile 0
  at 8 bits and Profile 2 at 10; AV1 Main where both 8 and 10 bits decode.
  Each up to the largest size NVDEC answers, with no level, so the host
  session lists Apple's (as VA-API's do);
- encoders: H.264 and HEVC, each profile NVENC lists that core has
  (`encode_profiles`), with NVENC's sizes and B-frames.

## The reims-graphics device

The reims-graphics device's video, on an NVIDIA GPU, runs on this backend's
NVDEC through `AnnexBDecoder` (`src/annex_b.rs`), in a host video session of
reims-vgpu's (`crates/reims-vgpu/src/graphics/video_host.rs`, `nvidia.rs`): the
guest's video driver hands each decode a whole Annex B access unit, one
picture, in decode order, and the planes of the output surface it lands in.
The device's arm chooses the runtime device's own GPU, by its PCI address,
where it is NVIDIA's and CUDA opens it (`graphics/video_choice.rs`).

- **The access unit, in place.** The codec crate's decoder parses it where it
  lies, in the host bytes of the guest's bitstream buffer, and hands NVDEC each
  picture's slices, units of the access unit's own bytes. Where they lie in it
  as one run (after start codes, only zeros between them), NVDEC is handed that
  run as it is, with no copy (`nvdec::decoder::one_run`); otherwise the slices
  are gathered into one buffer. NVDEC copies the bitstream within
  `cuvidDecodePicture`, so the guest's buffer is read there alone. A session
  logs at its end how many of its pictures went each way.
- **The picture, by one GPU copy.** Where the output's planes lie in pages a
  device reaches in place (the surface's view of the guest's RAM, or a copy
  block of the device's region), their view is registered with CUDA once a
  mapping, as "Zero copy" registers a guest buffer's, and each picture is one
  `cuMemcpy2D` a plane from the mapped surface into the plane at its pitch,
  its crop's rows alone, waited for before the session signals the decode's
  value. Any pitch is taken, so the device asks no layout of the guest.
  Elsewhere the picture is read back and copied by the CPU.
- **A picture that does not land is a decode error.** A picture whose planes
  the session refuses, or whose decode NVDEC failed once it ran, still
  signals the decode's value, so its command buffer completes, failed with
  Metal's internal error; the guest's video driver fails its session's next
  schedule for it (`kVTVideoDecoderMalfunctionErr`), one picture late. A
  stream's last such picture has no later schedule, and reaches the guest as
  a decoded frame (AppleGVA's queries carry no error).
- **Surfaces.** A picture holds its decode surface until it lands; the decoder
  has `AnnexBDecoder::SPARE_PICTURES` (2) beyond its references and the
  picture being decoded, and the session holds no more pictures unlanded than
  that. Pictures land on the session's landing thread, in decode order.

## The decoders

Each codec is the codec crate's decoder over an accelerator of ours that
fills `CUVIDPICPARAMS` from what the crate parsed, as the VA-API backend
fills libva's buffers. ffmpeg's `nvdec_*.c` are the model for the field
mapping, and NVIDIA's own parser the judge where the two differ (AV1,
below); every mapping is confirmed by the bit-exact runs below, not by
the model.

- **Output.** 8 bits decode into NV12, 10 into P016, which holds each
  sample in the high bits as P010 does, and goes to the guest as `x420` or
  `xf20`. Where the GPU does not write the guest's buffer itself ("Zero
  copy"), a picture is mapped only to be read back (`cuvidMapVideoFrame64`
  with one output surface) once its decode is done, copied with two
  `cuMemcpy2DAsync` on the decoder's own stream into the decoder's
  staging memory, page-locked by `cuMemHostAlloc` (pageable where CUDA
  gives none), and unmapped once the stream has run them ("How a session
  waits for the GPU"); then the CPU copies it into the guest's planes, the
  only step inside the guest buffer's lend ([core.md](core.md), "The copy
  into the guest's buffer"). A decode NVDEC reports failed
  (`cuvidGetDecodeStatus`) is bad data. Frames come out in decode order,
  as VideoToolbox's do.
- **A failed decode fails the frames after it.** NVDEC reports a slice cut
  short failed (status 9, `cuvidDecodeStatus_Error_Concealed`), as Apple's
  AVD fails it ([devices.md](devices.md), "H.264", "HEVC"). The copy that
  reads that status notes it in the H.264, HEVC or VP9 session's
  `Failures` (`src/decode.rs`), and the session's next decode sets the
  codec crate's damaged state (`mark_failed`), so that each frame after it
  is refused until an I picture (H.264), an IRAP picture (HEVC) or a key
  frame (VP9), as AVD refuses them. A VP9 frame the codec crate's
  decoder or the backend refuses, as one larger than the key frame (D16),
  does the same at once. Core copies each frame before the next decode, since these
  decoders do not decode while copying ([core.md](core.md), "Decoding
  ahead"). A frame core does not copy has its status read by no one: one
  the guest gave up on, or an HEVC picture with `pic_output_flag` 0.
- **H.264.** Slices as Annex B, each after a three-byte start code, with
  `pSliceDataOffsets`; the scaling lists in raster order; the DPB by
  surface index with `FrameIdx` and both fields' order counts.
- **HEVC.** The references by surface index with their order counts and
  long-term flags; `RefPicSetStCurrBefore`, `StCurrAfter` and `LtCurr` as
  indices into them, in the order 8.3.2 derives them, which the codec
  crate hands on, since NVDEC builds each slice's lists itself; the
  scaling lists in coding (diagonal) order, as dynlink_cuviddec.h says
  ("scaling lists (diag order)"), 32x32's from `matrixId` 0 and 3;
  `NumDeltaPocsOfRefRpsIdx` from the slice header's own predicted set.
- **VP9.** The frame from its uncompressed header on, with
  `frameTagSize` and `offsetToDctParts` the two headers' sizes: NVDEC
  reads the compressed header and keeps the probability contexts.
  `mcomp_filter_type` in libvpx's order; `segment_pred_probs` 255 without a
  temporal update.
- **AV1.** `width` is the upscaled width; references are named by surface
  index, `primary_ref_frame` too (255 for none); `lr_type` is
  `FrameRestorationType` and `lr_unit_size` counts from 32; each CDEF byte
  holds the primary strength low and the secondary, as coded, high; the
  tile groups go one after another with a start and an end offset a tile.
  With film grain applied, `CurrPicIdx` takes the frame with its grain and
  `decodePicIdx` the frame later frames predict from, so a sequence with
  film grain gets twice the surfaces. `SkipModeFrame0` and `1` come from
  the codec crate's frame header, which keeps them. Monochrome streams
  decode on a monochrome decoder, whose NV12 or P016 chroma NVDEC writes
  neutral (128, 512 at 10 bits): checked on every frame of aom-720p-mono
  and -mono-10bit.

  Three values follow NVIDIA's parser, not ffmpeg. A scratch program fed
  each stream to `cuvidParseVideoData` and printed the `CUVIDPICPARAMS`
  its decode callback got, field by field beside ours, for nine streams
  (aom-720p, aom-641x361, aom-720p-resize, aom-360p-resize-random,
  aom-720p-superres, aom-480p-lr-10bit, aom-1080p-tiles, svt-720p-grain,
  aom-360p-screen; 2026-10-24):
  - `PicWidthInMbs` and `FrameHeightInMbs` count whole macroblocks,
    rounding down (641x361 is 40x22), where ffmpeg rounds up. Rounded up,
    aom-641x361 decoded wrongly from its first frame, and aom-720p-resize's
    853-wide frames from their first inter frame;
  - each of the seven references carries its coded width, before
    superres's upscaling (1024 for a 1280 frame at denominator 10), where
    ffmpeg gives the upscaled width. Upscaled, libaom's superres streams
    failed with a decode error;
  - `ref_pic_flag` is set only for a frame some slot keeps
    (`refresh_frame_flags` not 0).

  What differs from the parser after that is unused: `lr_unit_size` with
  loop restoration off, `SkipModeFrame` with skip mode off, and the
  surface numbers.

  An AV1 decoder is made at the sequence's largest frame where NVDEC takes
  it, and at the first frame's otherwise; its size is the frame's own, odd
  too, as the parser's sequence callback gives it, and each frame is mapped
  at its own size. Mapped at a size rounded up to even, aom-641x361 and
  aom-641x361-10bit came out wrong in every frame; at its own size, every
  frame matched, and the resized streams too.

### What NVDEC does that its header does not say

All measured on 2026-10-24.

- **AV1 and VP9 want opposite sizes at an odd size.** p2-643x363 (VP9
  Profile 2) came out off by up to 561 from frame 0 on a 643x363 decoder;
  with the surfaces rounded up to even sides (644x364), each frame mapped
  at that size and the frame's own size in the picture parameters, every
  frame matched, as p0-641x361's did. AV1 is the other way round (above):
  its decoder and maps are the frame's own size. H.264's and HEVC's coded
  sizes are even already.
- **A side of 4n+3 decodes wrongly in AV1.** An AV1 frame whose width or
  height is 3 more than a multiple of 4 comes out with wrong blocks in
  every frame, 8 or 10 bits, libaom or rav1e: a sweep of widths 424 to
  1923 at height 360 and heights 240 to 287 at width 640 found every
  4n+3 side wrong and every other right. The errors are a decode's, blocks
  and streaks spreading from a few places, not a scaling. NVIDIA's own
  parser decodes the same streams the same wrong way, and through it VP9
  too, which ours decodes right; no decoder size, map size or
  reconfiguration measured avoided AV1's ([bugs.md](bugs.md), U7). So the
  backend refuses such frames ("What it refuses", below).
- **A frame smaller than its decoder is scaled up when it is mapped.**
  libvpx's dynamic resize codes inter frames of 960x540 after a 1280x720
  key frame. NVDEC decodes them right, predicting from the larger
  references, but `cuvidMapVideoFrame64` scales each to the decoder's
  target size: the 1280x720 surface held frame 60 stretched (a mean luma
  difference of 0.56 from the reference scaled by 4/3, against 64 for its
  top-left corner). So each picture keeps the size it was decoded at, and
  before it is mapped the decoder is retargeted to that size
  (`cuvidReconfigureDecoder`: coded and target size and display area), under
  the decoder's lock. Every resized frame of p0-720p-resize, p2-720p-resize
  and p0-720p-resize-early then matched libvpx. AV1's size changes take
  the same path, with references scaled at 569x320, 465x262 and 853x480
  matching libdav1d. The header says reconfiguration "must be called
  during CUVIDPARSERPARAMS::pfnSequenceCallback"; we use no parser, and
  calling it between decodes broke nothing measured. Waiting for a decode
  to finish before the reconfiguration changed nothing either.

- **VP9 reports no failed decode.** In the damaged-input runs ("Damaged
  input", below) NVDEC reported 124 to 795 failed copies a stream for
  H.264, HEVC and AV1 through `cuvidGetDecodeStatus`, and none for the 468
  VP9 frames copied, damaged as they were. So a VP9 frame NVDEC decodes
  wrongly, as in a boot of one of our VMs (B37), reaches the guest as good.
- **One of our VMs' boots fails valid decodes in other processes.** While
  one boots on the GPU, NVDEC reports valid frames failed in every other
  process, most in the seconds the guest's long command buffers run (B26):
  20 of 864 stress runs in two boots, and 10 of 223 in a third; none in
  4,968 without one, nor in 480 beside kernels of our own that load the GPU
  as those command buffers do ("Several processes at once", below; B37).
  A load with no guest that does the same (2026-10-25) is contended
  `atomicAdd` on a word of imported host memory ([bugs.md](bugs.md), U21).
  Under it and in the boots, most wrong frames came with status 2. The
  status catches only some of them, and decoding a failed picture again
  does not make its frames right, so the backend does not retry.
- **VP9's spatial layers decode wrongly.** With the surfaces made at the
  format description's size, the top layer's, libvpx's three spatial
  layers at 1280x720 (svc3-720p, `vp9-svc 3 1 0 0`, written by
  `crates/reims-vgpu-video-vaapi/tools/vp9-svc.c`) decoded with no error,
  their first five frames right, and then the second top-layer frame wrong
  in its two top-left 64x64 blocks; the error spread from frame to frame,
  to 22.4 dB against libvpx by the 150th top-layer frame. Reconfiguring
  the decoder to each frame's size before its decode, or to the previous
  frame's, or surfaces larger than every frame, changed nothing. NVIDIA's
  own parser gives every frame after the third the top layer's size, so
  it is no oracle here ([bugs.md](bugs.md), B40, D16, U11).

### Concealment

A frame lost from an H.264 stream whose SPS allows no gap in
`frame_num` is stood in for, so that the frames predicting from it
decode, as on Apple's AVD and VA-API ([devices.md](devices.md), "H.264").
NVDEC's surfaces are written by a decode alone, so `NvdecH264::conceal`
gives the last reference's own surface under a picture of its own
(`Picture::stand_in`, `nvdec/decoder.rs`), held until both go. How the
DPB names it follows NVIDIA's parser, measured on 2026-10-24:
- **NVIDIA's parser**, fed a stream through `cuvidParseVideoData` by a
  scratch program that printed each `CUVIDPICPARAMS`'s DPB and decoded
  nothing, was given `crates/reims-vgpu-video-vaapi/tools/h264-check.sh`'s
  drop20-ref1 and a four-reference drop20-ref4 (x264 `--ref 4`, frame 20
  of 60 left out, made by `h264-craft --drop 20`). With four references it
  named the lost frame `not_existing` 1, `PicIdx` -1, `FrameIdx` 4, beside
  the frames before it. With one, it kept the last reference's surface,
  `FrameIdx` 3, and named no lost frame.
- **One surface named twice fails.** With the stand-in beside its source
  in the DPB (drop20-ref4, frames 20 to 22), NVDEC reported each of those
  pictures failed, status 9 (`cuvidDecodeStatus_Error_Concealed`), and the
  first picture without the two (frame 23) decoded.
- **So a stand-in whose surface another entry names goes as
  `not_existing`, `PicIdx` -1** (`dpb_entries`, `nvdec/h264.rs`), and one
  whose surface no other entry names keeps it. Then drop20-ref1 decoded
  59 of 59 frames, the 50 the loss does not reach equal to the source's
  ffmpeg decode, and drop20-ref4 59 of 59 with no failure, through the
  backend (`nvidia-framemd5 --keep-going`) and through the host session
  (`conformance framemd5`, both modes). The concealed frames came to a
  luma PSNR of 21.3 to 26.3 dB against the source for drop20-ref1, which
  is a copy of the last reference as VA-API's, and 27.7 to 30.0 dB for
  drop20-ref4.

### What it refuses

- What the codec crate refuses (interlaced H.264, HEVC's range extensions,
  VP9 Profiles 1 and 3 and 12 bits, AV1's High and Professional profiles
  and 12 bits), as unsupported.
- What NVDEC's caps refuse: a stream past NVDEC's sizes, as unsupported
  (-12910), or under them, as too small (`Error::TooSmall`), which the
  guest gets at `dsta` as -12911, Apple's decoders' status for a size
  they do not decode, so that it decodes the stream itself. Core leaves
  the smallest size to the backend ([devices.md](devices.md), "Where we
  differ from Apple"), so a guest's H.264 from 48x64 decodes here, where
  AVD starts at 64, and VP9 and AV1 under 128x128 (p0-63x65, aom-63x65)
  and HEVC under 144x144 are refused, where Apple's host decodes them from
  64 ([bugs.md](bugs.md), D14).
- An HEVC picture with more than 8 pictures in one of its current
  reference sets, as unsupported: `CUVIDHEVCPICPARAMS` holds 8 a set
  (`current_set`, `nvdec/hevc.rs`, "NVDEC takes 8").
- AV1 of a size NVDEC decodes wrongly, a side of 4n+3 (`decodes_wrongly`,
  ours; [devices.md](devices.md), "Where we differ from Apple"): a
  description of such a size at `dsta`, with -12911 as Apple's decoder
  refuses a size; a frame of such a size mid-stream, and every inter frame
  predicted from one, decoded for its references and refused as bad data,
  which core emits with -12909 and no image. A key or intra-only frame
  decodes again.
- A VP9 frame larger than its key frame (D16), or an AV1 frame larger than
  its decoder's surfaces, which are the sequence's largest frame where
  NVDEC takes that and else the first frame's. VP9's spatial layers code
  such frames in every superframe, and with surfaces that hold them NVDEC
  decodes them wrongly (above, "What NVDEC does that its header does not
  say"), so they stay refused. A sequence that allows more than NVDEC
  takes is checked at open at the description's size: svt-720p-bigmax
  allows 16384x8704 for a 720p stream, and decodes.

## The encoders

`NvencEncoder` encodes raw NV12 or P010 frames in NVENC's synchronous mode,
and `NvidiaEncoders` puts it under the codec crate's encode session
(`SessionEncoder`), which plans the GOP, the reorder and the sample
buffers, so the guest sees what it sees on VA-API.

- **Pictures in and out.** Each frame is copied into an NVENC input
  buffer and encoded with `frameIdx` counting from 0 in submission order;
  `NV_ENC_ERR_NEED_MORE_INPUT` means nothing yet, and after a success the
  pending output buffers are locked in submission order, each its picture
  in coding order with `outputTimeStamp` naming the frame. A flush sends a
  picture with the end-of-stream flag alone, and the encoder flushes
  before it is destroyed, since destroying it with frames in flight
  crashed in the driver (in libnvcuvid, under `nvEncDestroyEncoder`, with
  lookahead on).
- **Configuration.** The preset NVENC gives for the quality asked (P1 to
  P7), then: lookahead off (below); `frameIntervalP` the B-frames plus 1,
  B-frames not used as references; IDR pictures forced where the codec
  crate's `KeyFrames` places them, as it places VA-API's
  ([vaapi.md](vaapi.md), "Shape"), with parameter sets on each: at the
  first frame, at a forced key frame, where the interval runs out, and
  2^30 frames after an IDR picture at the latest. NVENC writes the order
  counts itself. That it counts 2 a frame in H.264 and 1 in HEVC, as the
  VA-API encoder does, so that those 2^30 frames hold its counts within 32
  bits, is not measured. The VBV a second at the peak rate, or half a
  second under `DataRateLimits`, its initial delay three quarters of it,
  as VA-API sizes its HRD buffer; the VUI's colour and range from the
  session's settings. QVBR maps to VBR with a target quality.
- **Its waits are bounded.** A bitstream buffer's lock, which waits for
  the picture, is made on a thread of the session's own and waited for
  until the deadline ("How a session waits for the GPU").
- **Lookahead is off.** With P6 and P7 at high quality, which turn it on,
  after `NV_ENC_SUCCESS` only the oldest pending output locked; the next
  lock gave `NV_ENC_ERR_INVALID_PARAM`, and frames were lost even after the
  end-of-stream picture ([bugs.md](bugs.md), U6 and D12).
- **The parameter sets** are taken from the bitstream NVENC writes, not
  from what we asked, and the configuration record (`avcC`, `hvcC`) is
  built from them, as on VA-API.

### The bitrate floor

No rate under 32,000 bit/s reaches NVENC (`MIN_BITS_PER_SECOND` in
`src/encode.rs`, ours): an average, a peak or a constant rate under it is
raised to it, and with it the VBV buffer, a second or half a second at
the peak; a peak under the average is raised to the average. It is the
VA-API backend's floor ([vaapi.md](vaapi.md), "The bitrate floor"), so a
guest's `AverageBitRate` of 1 gets the same on either backend. The floor
is in the backend's mapping of the session's settings; `nvidia-encode`
drives NVENC below it.

A sweep of NVENC as VA-API's sweeps VCN (2026-10-24): 90 frames at 30
fps, VBR peaking at 1.5 times the average with a second of buffer, the
average halved from 1 Mbit/s to the first step whose output fell by less
than 2%. Rates are bit/s of the samples:

| profile | source | sweep's last step: asked, got | at 32,000 |
|---|---|---|---|
| H.264 High | `testsrc2` 640x480 | 62,500: 208,285 | 211,493 |
| H.264 High | `mandelbrot` 640x480 | 125,000: 321,800 | 333,944 |
| H.264 High | `testsrc2` 1280x720 | 125,000: 454,842 | 454,749 |
| H.264 High | `mandelbrot` 1280x720 | 250,000: 773,744 | 795,525 |
| H.264 High | `testsrc2` 1920x1080 | not swept | 727,040 |
| H.264 High | `testsrc2` 3840x2160 | not swept | 2,293,757 |
| HEVC Main | `testsrc2` 640x480 | 31,250: 86,165 | 86,165 |
| HEVC Main | `mandelbrot` 640x480 | 31,250: 49,429 | 50,205 |
| HEVC Main | `testsrc2` 1280x720 | 31,250: 172,074 | 172,074 |
| HEVC Main | `mandelbrot` 1280x720 | 15,625: 96,426 | 98,826 |
| HEVC Main | `testsrc2` 1920x1080 | not swept | 323,069 |
| HEVC Main | `testsrc2` 3840x2160 | not swept | 1,113,976 |

- **Where it bottoms out.** H.264's output stops at 208 to 774 kbit/s,
  HEVC's two to eight times lower, in every case above the floor. Under
  what the content can reach, H.264 `mandelbrot` grew again: 698,416 at
  500,000, 773,744 at 250,000 and 795,525 at the floor. HEVC `mandelbrot`
  at 720p is the one case that still fell under 31,250, by 2.4% at 15,625.
  `mandelbrot` at 640x480 gave 47,354 at 62,500, under its 50,205 at the
  floor.
- **At the floor.** Each source also ran at 32,000 peaking at 32,000, as
  for `DataRateLimits` alone under it: within 7% of the first in every
  case (HEVC `testsrc2` 1080p, 344,904 against 323,069). No run failed,
  and the kernel logged no Xid.
- **Unfloored.** Raw NVENC at VBR 1 bit/s with a VBV of 1 bit encoded 90
  of 90 frames of 640x480 `testsrc2`, at 211,816 bit/s for H.264 and
  86,565 for HEVC, QP 50: the floor changes the stream little, and NVENC
  takes no buffer of a bit.
- **Through the backend**, `nvidia-session-encode --bitrate 1` (the
  session's `AverageBitRate` 1): 90 of 90 samples, H.264 at 211,675 and
  333,984 bit/s, HEVC at 85,923 and 47,795, for `testsrc2` and
  `mandelbrot` at 640x480.

## Motion estimation

`NvidiaMotion` (`src/motion.rs`) is core's motion part on NVENC's
motion-estimation-only mode: an H.264 encoder initialised with
`enableMEOnlyMode`, whose `nvEncRunMotionEstimationOnly` fills an MV
buffer from an input and a reference frame, with no bitstream. The
structures it adds are in `src/nvenc/motion_ffi.rs`, checked with the rest
against ffnvcodec's header (`src/layout.rs`, 2026-10-24: passed).

What it offers, and how it converts (2026-10-24):
- **Where.** The GPU must list H.264, give
  `NV_ENC_CAPS_SUPPORT_MEONLY_MODE` 1 and take NV12; H.264's sizes are
  then the limits: 145x49 to 4096x4096 on the RTX 5080 Laptop GPU. HEVC's
  ME-only mode is not used: its records are per coding unit, of 8x8 to
  64x64 as NVENC chooses (the header's `NV_ENC_HEVC_MV_DATA`), where
  H.264's are one a macroblock, a fixed grid of 8x8 blocks.
- **Sources** are NV12 in either range, both given to NVENC as NV12.
- **A session** is opened at each estimator's open, at the picture's size
  rounded up to even and to NVENC's least: P7, true motion or not, the
  preset's configuration at the high-quality tuning, with an infinite GOP
  and no B pictures ("The search", below). `tuningInfo` itself stays
  undefined: with it set, `nvEncInitializeEncoder` refused, "tuningInfo
  is not supported with H264 Meonly mode".
- **A session's first pair** is estimated twice, the first answer
  dropped, since NVENC searches from its last answer too ("The search").
- **A frame pair** is copied into the session's two input buffers, a
  picture smaller than the frame padded by repeating its last column and
  row.
- **The vectors** come back one record per macroblock,
  `NV_ENC_H264_MV_DATA`: four vectors in quarter pixels, a type, a
  partition and a cost. Each of the four is its own 8x8 block's, in
  raster order, whatever the partition: a 16x16 macroblock held four
  equal vectors, a 16x8 one the top's twice then the bottom's twice, and
  an 8x16 one the left's, the right's, then both again
  (`motion::gpu::print_what_nvenc_answers`, an ignored test: textures
  whose halves moved apart across row 72 and column 72). So the backend
  gives core 8x8 blocks of the picture, each vector divided by 4. Their
  sign is Apple's: content moved by (+8, +4) gave (-32, -16), and
  (2.5, -1.25) gave (-10, 5) in all 512 vectors, a quarter of a pixel
  finer than Apple's processor ([devices.md](devices.md), "The
  motion-estimation session").
- **Errors.** A session NVENC will not open is Unsupported, which core
  answers `mest` as -12902; a run that fails is a malfunction, which core
  emits with the frame.

### The search

Measured with `motion::gpu::measure_presets`, an ignored test that runs
the backend's estimator at each preset on the conformance texture
(2026-10-24). Frames K are the texture moved by K times (20, -12), and
pairs (K, K + 1) go to one session in order, as a guest's session sends
them. A hit is a vector within 2 pixels of the way back each way, counted
over the picture's 8x8 blocks and over its 16x16 cells on Apple's grid,
as the guest gets them.

- **Three levels.** P1 and P2 found the same vectors, P3 and P4 the
  same, and P5 to P7 the same, at both sizes and in every order. A
  session's first pair, as 16x16 cells:

  | preset | 256x128 | 1280x720 |
  |---|---|---|
  | P1, P2 | 25.8% | 92.1% |
  | P3, P4 | 46.1% | 94.3% |
  | P5 to P7 | 62.5% | 95.3% |

  Apple's processor gave 82.8% of the same 256x128 case on the M3 Max
  (macOS 27.0.1). The 8x8 blocks' shares were the cells' or a little
  under: 58.4% of them at P7 for 256x128. Moving by an eighth of its
  width, about a sixth of that picture's content is not in its reference
  at all.
- **NVENC searches from its last answer too.** On one P7 session the
  same 256x128 pair gave 62.5%, then 92.2%, then 93.0%; the three pairs
  in order 62.5%, 92.2%, 93.8%; and the third pair sent first 66.4%, then
  the first pair 89.8%. So a session's first estimate is the weak one,
  and the backend makes it twice: as a guest's session opens it, the
  three pairs gave 92.2%, 93.8% and 93.8% at 256x128, and 98.7%, 98.8%
  and 98.7% at 1280x720.
- **The time** is the same from P4 to P7, from the call to the field
  with the copies in, 100 estimates after 10 on one session:

  | size | P1 | P4 | P7 |
  |---|---|---|---|
  | 1280x720 | 1.23 ms | 2.00 ms | 2.02 ms |
  | 1920x1080 | 2.76 ms | 4.28 ms | 4.40 ms |
  | 3840x2160 | 11.16 ms | 17.12 ms | 17.17 ms |

  These are medians; the 95th percentiles were within 0.7 ms of them,
  and the slowest of each 100 within 1.8 ms. A 30 fps frame has 33 ms,
  so P7 fits at 4K too, and the session's first pair costs one estimate
  more: 4.3 ms against 2.0 at 1280x720.

So every session searches at P7. Lookahead is left as the preset has it:
the session runs one pair at a time, and P5, which turns no lookahead on,
found P7's vectors.

### The check

`conformance 0 motion-check` passed 8 of 8 at P7, the first pair
estimated twice, each pair in 0.7 to 6.9 ms (2026-10-24):

| case | share of cells the way back | commonest |
|---|---|---|
| b16-256x128, (8, 4) | 97.7% within 2 | (-8, -4) x124 |
| b16-256x128, (-12, 7) | 100% within 2 | (12, -7) x125 |
| b4-256x128, (3, -2) | 100% | (-3, 2) |
| b4-256x128, (-5, 7) | 93.0% | (5, -7) |
| b16-641x361, (7, -9) | 93.1% within 2 | (-7, 9) |
| b4-641x361, (-4, 3) | 97.1% | (4, -3) |
| b16-1280x720, (20, -12) | 98.7% within 2 | (-20, 12) |
| b16-128x48, (4, 3) | 100% within 2 | (-4, -3) |

Not measured: an intra macroblock's record (every one seen was type 1,
P), and what NVENC gives past the picture inside its padding, which core
does not read.

**In the guest** (2026-10-24), the guest's own
`VTMotionEstimationSession` on this backend found the way back in 32 of
33 pairs, at 16 and 4, from 128x48 to 1280x720, multipass included. The
one that failed was the first pair of a (20, -12) shift on 256x128, 58.6%
of its cells, at P4 and estimated once, where the backend's sessions
search at P7 and estimate a first pair twice ("The search" above). No
guest has run P7 and the doubled first pair.

## The JPEG engine

`NvidiaJpegDecoder` (`src/jpeg.rs`) is core's JPEG decoder for `jcap` and
`jdec` ([devices.md](devices.md), "The JPEG engine") on nvJPEG's hardware
decoder, `NVJPEG_BACKEND_HARDWARE`: the GPU's JPEG engine.

**Where it comes from.** libnvjpeg is a CUDA toolkit library: NVIDIA's
driver does not install it (nvidia-utils 615.71.09 has no such file), a
CUDA toolkit or its redistributable `libnvjpeg` archive does. The backend
loads `libnvjpeg.so.13`, else `libnvjpeg.so.12`, from the loader's path
when it opens, makes a handle on the hardware decoder, and decodes an 8x8
grey image of 128s to check it. Where the library is not found, the
handle or the decoder does not open, nvJPEG lists no hardware decoder
engine, or the trial comes back wrong, the backend has no JPEG part, the
log says why once at info, and the guest is told there is no JPEG engine,
as before ([bugs.md](bugs.md), D23). libnvjpeg 13 links the CUDA runtime
in, so it needs nothing beyond the driver's libcuda (`readelf -d`). Nothing
of NVIDIA's is shipped with reims-vgpu.

**What nvJPEG does here**, measured with probes of our own on this GPU
(libnvjpeg 13.2.3 from CUDA 13.4.3's redistributables, 2026-10-25),
against libjpeg-turbo 3.2.0's islow decode of cjpeg's streams of ffmpeg's
`testsrc2`, as raw planes (`tools/jpeg-planes.c`):
- `nvjpegCreateEx` opens every backend: the hardware decoder lists 1
  engine of 1 core; the hardware encoder answers
  `NVJPEG_STATUS_ARCH_MISMATCH` (nvJPEG's documentation names Jetson Thor
  alone).
- The hardware decoder's planes equal libjpeg's, every sample, for 4:2:0,
  4:2:2, 4:4:4, 4:4:0 and grey; qualities 10 to 100; restart markers every
  MCU, every three and every row; 1x1, 17x9, 15x15, 65x33, 333x211,
  4095x7, 7x4095, 16384x64, 64x16384, 4000x3000 and 16384x16000;
  optimized tables; and crops at the image's edge. The hybrid decoders
  (`NVJPEG_BACKEND_HYBRID`, `GPU_HYBRID`) are within 1, as VA-API is.
- It refuses 4:1:1, progressive and multi-scan streams
  (`NVJPEG_STATUS_JPEG_NOT_SUPPORTED`), which the hybrid decoders decode;
  no backend takes arithmetic coding or 12-bit samples; and it refuses a
  stream that defines no Huffman table, as a Motion JPEG frame.
- A quantization value over 255 is read as its low 8 bits, with success
  and wrong pixels ([bugs.md](bugs.md), U20).
- A stream without its EOI, or with damaged entropy-coded data, decodes
  with success, as on VCN.
- Its scale, `nvjpegDecodeParamsSetScaleFactor`, works on the hardware
  decoder alone (the hybrids answer `NVJPEG_STATUS_INVALID_PARAMETER`):
  each plane at its own sampling, rounded up (333x211 4:2:0 at 1/2 is
  167x106 and 84x53), within 0.15 of a sample on average of a box filter
  of libjpeg's planes, and at most 1 off in luma at even sizes.
- A region of interest (`nvjpegDecodeParamsSetROI`) decodes exactly at
  scale 1, and fails with a scale (`NVJPEG_STATUS_EXECUTION_FAILED` or
  `NVJPEG_STATUS_BAD_JPEG`).
- It writes NV12 only from 4:2:0 and YUY2 only from 4:2:2, and BGR from
  every sampling: it does not resample chroma between samplings.
- Into device memory, the parse included, without the copy back: 0.15 ms
  for 640x480 4:2:0, 0.75 ms for 1920x1080, 4.5 ms for 3840x2160, 7.1 ms
  for 4000x3000 4:2:0 and 14.5 ms for 4:4:4, 107 to 140 ms for 16384x16000
  4:2:0 and 57 ms for grey, with about 1 µs of the CPU's; the same at 1/8.
  The hybrids took 2.6 to 128 ms of the CPU for the same images.

**A decode.** The codec crate's parser reads the stream, the request is
checked against it (`src/jpeg/plan.rs`), and its scan is walked on the
CPU, as on VA-API ([vaapi.md](vaapi.md), "What it refuses"). Then nvJPEG
is given the image written again from what the parser read
(`src/jpeg/stream.rs`): its quantization tables as 8-bit values, which
refuses a value over 255 before nvJPEG sees it; Annex K's Huffman tables
where the stream defines none; and nothing after the scan. nvJPEG decodes
into device memory:
- the components' planes at their own sampling (`NVJPEG_OUTPUT_YUV`), for
  every YCbCr layout, and for `BGRA` at a scale;
- interleaved BGR with its chroma interpolated
  (`NVJPEG_FLAGS_UPSAMPLING_WITH_INTERPOLATION`), for `BGRA` at scale 1,
  made BGRA on the CPU.

A scale of 1/2, 1/4 or 1/8 is nvJPEG's. A crop at scale 1 is its region
of interest; at another scale it is cut from the whole image's scaled
decode, the same thing, as each 8x8 block scales alone. The part of each
plane the output is made of is copied on the decoder's own stream into
page-locked memory, and from there into the host session's buffer by
core's `write_samples`, where nvJPEG's samples are the output's:
- at scale 1 into a layout of the stream's sampling: the image's own
  samples, as VA-API copies them; grey's luma into every layout, core
  writing its chroma 0x80;
- at a scale, where each side of the region is the scale times the
  output's (for the chroma, times its divisors too): nvJPEG's samples,
  each the mean of a square of the image's, are where core's reference
  puts the output's.

Elsewhere the CPU makes the output of nvJPEG's samples first
(`src/jpeg/convert.rs`), as core's reference (`reference::resample`,
which the conformance compares with) makes it of the image's:
- an odd size at a scale: the reference spreads `ceil(width / scale)`
  samples over the region, where nvJPEG's cover from the image's
  top-left corner, so 333 at 1/2 is 167 samples over 333, not 334;
- chroma into a layout of another sampling, at any scale;
- `BGRA` at a scale.

Each output sample is the reference's filter over nvJPEG's samples, each
taken as the mean of the image's samples it covers: coverage where the
output sample covers more than one of the image's samples, and linear
interpolation between centres where it covers less. Where the layout's
chroma is finer than the stream's along an axis, as 4:2:0's into `422f`,
`yuvs` or `2vuy`, nvJPEG decodes at twice the scale (1/1 for 1/2). The
reference takes those rows from every chroma row of the image, which a
decode at the scale halves (Cr 31.6 dB at 1/2). The luma is then
resampled from the finer decode. `BGRA` at a scale is the planes made
BGRA, BT.601 at full range, each pixel taking the chroma sample it lies
in. nvJPEG's own BGR at a scale scored 21.1 to 28.6 dB per channel
against core's reference, under the conformance's 30 (4:2:0 at 1/2,
4:2:2 at 1/4, a crop of 4:2:0 at 1/4), where its planes at those scales
scored 57 to 67.

One decode runs at a time, as `jcap` says one core. Device and
page-locked memory of up to 64 MB are kept for the next decode.

What it refuses, before the GPU, with core's `JpegError`: what VA-API's
decoder refuses ([vaapi.md](vaapi.md), "What it refuses"), but Huffman
tables 2 and 3 and VA-API's table sizes, which nvJPEG takes, and the
driver's largest size, which nvJPEG answers; and 4:1:1, as above. `jcap`'s
capabilities are VA-API's: one core, and none of the four features.

**Conformance.** `crates/reims-vgpu-video-vaapi/tools/jpeg-check.sh
--large`, with `RVGV_JPEG_CHECK` naming the `nvidia-conformance` example,
runs the cases VA-API passes ([vaapi.md](vaapi.md), "Testing on the AMD
GPUs") through the backend's `jpeg` part. On this GPU (2026-10-25,
libjpeg-turbo 3.2.0, ffmpeg n9.0.2) all 102 passed:
- Every plane of the decodes that copy samples was equal to libjpeg's,
  and within 1 of ffmpeg's.
- The scales and conversions, the worst plane of each against libjpeg's:
  - conversions at scale 1, 59.3 to 94.6 dB;
  - scales of even sizes, 53.1 to 70.3 dB, 1/8 at 57.7 to 65.4.
    `BGRA` at a scale scored 53.1 to 57.7 dB against the nearest chroma,
    and 19.2 to 23.2 against the centres;
  - odd outputs: 333x211 at 1/2, 30.6 dB into `420f` (its Cr; Y 45.0)
    and equal into `2vuy`; a 93x51 crop of 4:4:4 at its edge, 49.9;
    4000x3000 at 1/8, whose 500x375 is odd, 40.2.

  VA-API scored 61.5 to 75.0, 56.8 to 67.1, and 30.9 to 32.6 and 46.2 for
  the same.
- Each refusal was refused before the GPU, with its reason.

| decode, with the walk | RTX 5080 Laptop GPU | VA-API, RX 7900 XTX |
|---|---|---|
| 1920x1080 4:2:0, mean of 20 | 3.36 ms | 4.75 ms without the walk |
| 4000x3000 4:2:0, mean of 10 | 19.6 ms | 25.8 ms; 22.2 without |
| 4000x3000 4:4:4, mean of 10 | 38.9 ms | 50.8 ms without |
| 4000x3000 4:2:0 at 1/8, mean of 10 | 15.0 ms | 21.9 ms, not said which |
| 8192x8192 4:2:0 | 119 ms | 135 ms |
| 16384x16000 4:2:0 | 425 ms | 529 ms |
| 16384x16000 grey | 255 ms | 278 ms without |

Each time covers what VA-API's do ([vaapi.md](vaapi.md)): the parse, the
surface or memory, the decode and the copy into the destination, and the
walk here. The walk takes about a third: of 4000x3000 4:2:0's 18 to 20 ms
the walk took 6.2 to 6.8 and the GPU and the copy back 8.6 to 9.3; of
16384x16000's 425 ms, 72 and 250.

**Waiting.** The stream is asked after with `cuStreamQuery` until the
deadline ("How a session waits for the GPU"). Past it the decode fails as a
malfunction (kIOReturnDeviceError), and nvJPEG's state, buffers and parsed
stream, the device and page-locked memory, and the bytes nvJPEG reads are
kept until the stream is done; each later decode fails at once the same
way until then.

**NVDEC's JPEG** (`cudaVideoCodec_JPEG`) is not used. NVDEC lists it
(64x64 to 32768x16384, 4:0:0 to 4:4:4, 8-bit), but writes NV12 alone from
every sampling, so a 4:2:2 or 4:4:4 stream's chroma would come back
halved; its picture parameters for JPEG are opaque, so only NVIDIA's own
parser could drive it.

## Zero copy

A decoded frame is copied by the GPU straight into the guest's buffer, and
the CPU copies nothing (`src/zero_copy.rs`; [core.md](core.md), "Zero
copy"). reims-vgpu lends the buffer as its view of the guest's pages: one
contiguous mapping of the memfd's runs in the order the buffer's bytes lie
in, at the surface's base (`crates/reims-vgpu-vulkan/src/os/linux/device_lend.rs`).
The backend registers that view's whole pages with CUDA once per mapping,
`cuMemHostRegister_v2` with `CU_MEMHOSTREGISTER_PORTABLE`, which pins them
for every context, and keeps the registration by the mapping's identity,
process-wide, since a range is registered once in a process whichever GPU
registered it (at most 96, the one used longest ago dropped first). It is
unregistered, with its GPU's context pushed, when the mapping goes, before
the hold that keeps the view mapped is let go. Each frame is then the
staged read back's calls ("Output", in "The decoders"): the wait for the
decode, the picture's map, one `cuMemcpy2DAsync` a plane from the mapped
surface into the guest's plane at its own offset and pitch, the wait for
the stream, and the unmap, all under the decoder's lock, so the frame is
in the guest's pages when the write returns. A write past its deadline
fails, and the buffer stays held, its registration kept, until the GPU is
done ("How a session waits for the GPU"). Each copy writes the plane's
rows of the frame's samples alone, as the CPU's did, inside the rows core
checked against the buffer (`GuestTarget`).

So every layout the guest asks for is written by the device here, 1080p
8-bit's 1920-byte rows among them: CUDA takes any pitch. No dma-buf is
made, so udmabuf's limits do not bear on it: on the RTX 5080 Laptop GPU at
the stock limits (1,024 runs, 64 MiB; driver 615.71.09, 2026-10-25), the
stand-in guest's 21 streams, H.264, HEVC, VP9 and AV1 at 720p, 1080p and
4K, 8 and 10 bits, sync and async, had all 2,520 frames written by the
device, none declined, each matching its reference, and the host's
messages the same by `cmp` as with zero copy off; the session's threads
used 0.6 to 1.1 ms of the CPU a 4K frame, against 1.9 to 3.6 ms off. What declines,
each frame then copied by the CPU with the reason logged once a session:
a registration CUDA refuses, a copy that fails, a buffer of other samples
than the frame's, or a driver without `cuMemHostRegister_v2`. Zero copy is
on, with nothing to set; only a check turns it off ([core.md](core.md),
"Zero copy").

**Measured** (2026-10-25, nothing else on the GPU but KWin):
- The decode check ("Decode", below) with zero copy on, the default: 387
  checks passed, 1 note (U1, `main-1366x766-ctu16`), none failed.
- perf-check's five streams ("How fast a session decodes", below) through
  `conformance /dev/dri/renderD128 framemd5 --sequence`, sync and async,
  zero copy on and off: every frame matched the reference, and the host's
  messages to the stand-in guest were the same by `cmp`, in all ten.
- The same streams, zero copy on and off taking turns, medians of five
  runs, fps on / off, and the host's split of a frame:

  | stream | sync | async | device write | read back, copy |
  |---|---|---|---|---|
  | 4K HEVC Main | 224.6 / 163.7 | 269.5 / 207.0 | 2.72 to 3.14 ms | 2.71 to 3.10, 1.08 to 1.26 ms |
  | 4K HEVC Main 10 | 158.0 / 130.1 | 160.6 / 135.0 | 3.44 to 3.50 ms | 3.59 to 3.70, 2.26 to 2.30 ms |
  | 4K AV1 | 209.1 / 168.4 | 233.2 / 189.8 | 2.46 to 2.78 ms | 2.49 to 2.61, 1.19 to 1.35 ms |
  | 4K AV1 10-bit | 148.1 / 117.1 | 149.6 / 124.8 | 2.79 to 3.01 ms | 2.93 to 3.11, 2.26 ms |
  | 1080p H.264 High | 472.1 / 448.3 | 564.8 / 523.4 | 1.14 to 1.21 ms | 1.11 to 1.24, 0.21 to 0.22 ms |

  The GPU's copy into the guest's pages takes what its copy into the
  page-locked staging memory took, and the CPU's copy after it, 1.1 to
  2.3 ms of a 4K frame, is gone: 4K ran 19% to 37% faster, sync and
  async. The stand-in's CPU time a frame, its MD5 included, fell by 0.8
  to 2.6 ms at 4K.
- In a guest: every frame of every session written by the device, 1080p
  H.264 among them, every decode matching its reference; the guest's 4K
  decodes 27% to 63% faster, and the host's CPU time a 4K frame, the
  session's own threads', down by 1.2 to 2.7 ms, to 1.3 to 2.1 ms.

### How a session waits for the GPU

Every wait of the backend's on the GPU is bounded, by core's deadline,
5 s from a frame's first wait ([core.md](core.md), "Waiting for the
device"), asked after every 100 µs for its first 50 ms and every
millisecond after (`src/wait.rs`). A decode call's waits share one
deadline, from the first of them (`wait::Deadline`): a VP9 picture's
decode, and the read backs of the frames the call gives, which core makes
before it answers the call's sample. So a frame is answered within one
deadline of its first wait, though a VP9 frame waits for its decode in
the decode call and for its copy in core's copy after it.

| wait | asked with | in place of |
|---|---|---|
| a picture's decode | `cuvidGetDecodeStatus`, until it is no longer `cuvidDecodeStatus_InProgress` | `cuvidMapVideoFrame64`'s own wait |
| a VP9 picture's decode | its `cuvidDecodePicture`, which returns only once the picture is decoded, made on the decoder's call thread (`reims-vgpu-video.nvdec-calls`) with a copy of the bitstream, whose answer the session's thread waits for | the same call on the session's thread |
| its map's post-processing and the two copies | `cuStreamQuery` of the decoder's own stream (`CU_STREAM_NON_BLOCKING`), which the map is given as its `output_stream` and the `cuMemcpy2DAsync` run on | `cuMemcpy2D`, which returns once its copy is done |
| the two copies into pageable staging memory, where `cuMemHostAlloc` gives no page-locked memory | the two `cuMemcpy2DAsync`, which return only once done into pageable memory, made on the decoder's call thread, whose answer the session's thread waits for; then `cuStreamQuery` | the same copies on the session's thread |
| an encode, a motion estimate | the waiting `nvEncLockBitstream`, made on the session's call thread (`reims-vgpu-video.nvenc-locks`), whose answer the session's thread waits for | the same lock on the session's thread |
| a JPEG decode and its copies | `cuStreamQuery` of the JPEG engine's own stream, on which nvJPEG's decode and the `cuMemcpy2DAsync` run | `cuStreamSynchronize`, which waits with no deadline |

The call threads are one mechanism (`src/wait.rs`, `Calls`): a thread of
the decoder's or the NVENC session's own, started with its first call,
which makes one call at a time, the next only once the last has returned,
and ends once its owner has gone and its call has returned.

**Past the deadline** the frame fails as the backend's malfunction
(`Error::Stalled`): -12911 with no image for a decoded frame, -12912 for an
encoded one, -17691 for a motion estimate. The decoder, encoder or
estimator then fails each later call at once, and keeps what the GPU may
still write in core's list until the GPU is done:
- a picture's: its surface, which no later picture takes; its mapped
  frame, unmapped once the stream is done; the decoder's staging memory;
  for zero copy, the guest buffer's registration and its hold, so the
  buffer stays held; and the decoder, destroyed after (`nvdec::decoder`,
  `UnfinishedPicture`), each let go once NVDEC answers the decode done or
  the stream answers done, and, for copies into pageable memory, once
  they have returned on the call thread first;
- a VP9 decode's: the decoder, its handle and its stream, held by the
  call until it returns, then destroyed once nothing else holds them
  (`wait::Unreturned`);
- an NVENC session's: the session, its buffers and its configuration,
  which its call thread finishes once the lock returns: it unlocks the
  buffer, ends the stream and locks each buffer still submitted, as an
  encoder drains its session before it is destroyed, then destroys the
  session (`nvenc::locks`).

Nothing a call left on a call thread holds is used until it has
returned: its decoder or session fails every later call before it makes
one, and the kept work asks no question of it until the call has
answered. An answer that is an error keeps the work, and a GPU that never
finishes keeps it until the process ends, with the call thread.

**What the waits answer**, measured on the RTX 5080 Laptop GPU (driver
615.71.09, 2026-10-25, nothing else on the GPU but KWin), every frame of
streams of 150 frames read back as the backend reads them:
- H.264, HEVC and AV1: `cuvidGetDecodeStatus` answered
  `cuvidDecodeStatus_InProgress`, then `cuvidDecodeStatus_Success`; some
  AV1 frames were done at the first question. The decode was done after
  0.43 ms (median) of a 1080p H.264 frame, 2.4 to 2.7 ms of a 4K HEVC one
  and 1.7 to 3.2 ms of a 4K AV1 one, 15 ms at most.
- Asked first, the map then took 3 to 122 µs, its post-processing queued
  on the stream, which had work left after the map and after the copies;
  the copies were done 0.15 ms after they were queued at 1080p, about
  0.5 ms at 4K at 8 bits and 0.93 to 1.05 ms at 10. Not asked first, the
  map waited for the decode itself, 2.4 ms (median) of a 4K HEVC frame.
- VP9: `cuvidGetDecodeStatus` answered an error for every frame (its
  header keeps a status for HEVC, H.264 and JPEG), and `cuvidDecodePicture`
  itself returned only once the decode was done, 2.87 to 3.04 ms (median)
  of a 4K frame and 0.36 ms of a 720p one. So VP9's decode is waited for
  in its submission, made on the decoder's call thread, and its read
  back's question is the stream's alone.
- From a suspended GPU (`runtime_status` `suspended`), a new process's
  first 4K HEVC frame was decoded in 2.86 ms at most: the wake goes into
  the backend's opening, not into a frame.
- NVENC: why its lock is not asked with `doNotWait` (`nvenc::locks`, which
  gives the measure): in synchronous mode, the only one on Linux, such a
  lock waited for its picture as a waiting lock does, never answered
  `NV_ENC_ERR_LOCK_BUSY`, and gave an IDR picture's size as about 390
  million bytes where it was about 1 MB, or, with split-frame encoding,
  its size right and half its bytes wrong ([bugs.md](bugs.md), U19).

**What it costs**, measured the same day on 4K HEVC Main (perf-check's
`hevc-2160p-main`, 150 frames) through core's host session with the
stand-in guest, synchronous, zero copy on, three runs each taking turns
with the backend's synchronous copies and map: 248.4, 261.5 and 277.1 fps
with those, 259.2, 263.9 and 264.5 with the bounded waits; every frame
matched the reference as a set in all six runs, and the host's messages
were the same by `cmp`. The process's CPU for the 150 frames, the
stand-in guest's MD5s included, in three more runs each: 3.45 and 3.47 s
with the synchronous copies, after a first run of 4.19 s, and 3.35 to
3.41 s with the bounded waits, whose questions sleep between them where
the synchronous copies spun ("How CUDA's waits spend the CPU").

**Its tests.** On the NVIDIA host, ignored elsewhere:
- a picture whose copies the GPU holds back behind a kernel of ours that
  spins until the test lets it go (`cuda::spin`) fails at a deadline of
  100 ms with -12911, the next frame's read back at once, and the picture
  stays kept until the kernel ends, then is let go (`decode::gpu`); and the
  same through pageable staging memory, whose copies the decoder's call
  thread makes, once a read back through it matched one through
  page-locked memory;
- a decode still running at a deadline of none fails the same way, and is
  let go once decoded; and so does a VP9 decode, still in its submission;
- a VP9 decode held inside its submission on the call thread by the test
  fails at a deadline of 100 ms, each later call at once, and the decoder,
  dropped, stays kept until the call returns, then is destroyed. A
  spinning kernel does not hold NVDEC's VP9 decode, hence the test's own
  hold: with one on the legacy null stream or on a stream of its own, a 4K
  frame's `cuvidDecodePicture` returned in 0.37 and 0.49 ms, against
  0.23 ms with none;
- a VP9 decode the test holds 0.7 s inside its submission returns by its
  deadline of 1 s, and the frame's read back, whose copies the spinning
  kernel holds, fails at that same deadline: answered 1.0002 s after the
  decode began, where a read back with a deadline of its own answered it
  at 1.70 s (`a_vp9_frame_is_answered_by_its_decodes_deadline`, run on
  the code before the shared deadline too);
- a 4K HEVC encoder whose lock passes a deadline of 100 µs fails that frame
  and the two after it, each later frame at once, and is destroyed by its
  call thread once NVENC is done (`nvenc::encoder::tests`).

The pageable and VP9 tests also count each use of the decoder's handle or
stream while a call of its call thread is in flight, and find none. The
spinning kernel ends on its own, whatever the host does: after 10 s by the
GPU's clock, or 6 million reads of its flag, 5.6 s at the 0.93 µs a read
took. All passed on the RTX 5080 Laptop GPU (driver 615.71.09,
2026-10-25), with no Xid, reset or watchdog line in the kernel's log.
Core's stand-in tests the guest's side ([core.md](core.md), "Waiting for
the device").

**Through the host session**, the same day and GPU, the backend before
the call threads and after: 11 streams of our own (H.264 720p and 1080p,
HEVC 1080p and 4K Main 10, VP9 Profile 0 at 720p with hidden frames, 1080p
with tiles and 4K, Profile 2 at 720p and 4K, AV1 by libaom and SVT-AV1 at
10 bits) bit-exact against ffmpeg's decode, both ways, with zero copy and
with the CPU's copies, 44 of 44 each; `encode-check` 35 of 35 and
`motion-check` 8 of 8 each. With the shared deadline, the same 11 streams
44 of 44 again.

### How CUDA's waits spend the CPU

Whether CUDA's synchronous copies spin while they wait, measured with the
copies made by `cuMemcpy2D`, which the bounded waits replace
(2026-10-25, nothing else on the GPU but KWin, perf-check's five streams
of 150 frames, every frame of every timed run below matching the
reference as a set).

**The context waits by spinning.** The backend retains the GPU's primary
context with the driver's own flags, and `cuDevicePrimaryCtxGetState`
answers 0x0 for them: `CU_CTX_SCHED_AUTO`, which the CUDA driver API's
reference says spins where a process has fewer contexts than the host has
CPUs, as here (1 and 24).

**Where the session's CPU goes.** perf of `conformance /dev/dri/renderD128
framemd5`, task-clock at 4999 Hz with DWARF call chains, as root with
`kernel.kptr_restrict` 1 for the length of the runs, each sample of the
session's own threads (comm `reims-vgpu-vide`: the decoder's thread and
its frames' copy threads) put under the outermost of our calls or the
driver's on its stack; ms a frame, over the 150:

| stream, mode | in all | the two copies | NVDEC's other calls | the decoder's creation, NVDEC's caps | the rest |
|---|---|---|---|---|---|
| 4K HEVC Main, sync | 1.33 | 0.50 | 0.22 | 0.49 | 0.12 |
| 4K HEVC Main, async | 1.30 | 0.47 | 0.21 | 0.49 | 0.14 |
| 4K HEVC Main 10, sync | 1.83 | 0.95 | 0.20 | 0.44 | 0.24 |
| 4K AV1 10-bit, async | 2.16 | 0.95 | 0.33 | 0.58 | 0.31 |
| 1080p H.264 High, sync | 0.88 | 0.20 | 0.18 | 0.36 | 0.14 |
| 1080p H.264 High, async | 0.85 | 0.17 | 0.17 | 0.39 | 0.12 |

- The copies spin: every sample under `cuMemcpy2D` is in user space, in
  libcuda's own code (82% to 87%) and in `clock_gettime` (13% to 17%),
  none in the kernel. They spin for as long as the copy runs, 0.5 ms of
  4K NV12 and 0.95 of P016.
- NVDEC's wait for the decode does not: the map, the unmap and the
  decode's submit run in nvidia.ko and the kernel, and most of the device
  write's 2.7 to 3.5 ms is spent asleep in the map.
- The decoder's creation and NVDEC's caps, once a session and on the
  session's thread, are 54 to 87 ms, 24% to 41% of a 150-frame session's
  CPU. The host CPU a frame measured in the guest ("Zero copy") counts
  them too. The rest is the codec's parse, core, and the buffers'
  registrations.

**Waiting by blocking costs throughput.** A build that chose by
environment how a session waits measured three changes: the primary
context's scheduling set before its retain (`cuDevicePrimaryCtxSetFlags`,
`CU_CTX_SCHED_BLOCKING_SYNC`); the planes' two copies queued on the null
stream (`cuMemcpy2DAsync`) with one wait for both; and that one wait on
an event made with `CU_EVENT_BLOCKING_SYNC`, the context left as it is,
for copies of 8 MiB or more. Each ran against the backend as it is, in
rounds that took turns, the order rotated each round: one run for fps,
and one under perf at 3999 Hz for the session's threads' CPU. Medians of
7 rounds (9 for the event), fps sync / async; then the session's threads'
ms a frame, sync (async within 0.06):

| stream | as it is | blocking context | blocking, one wait | spinning, one wait | event, one wait (its own rounds: as it is → event) |
|---|---|---|---|---|---|
| 4K HEVC Main | 239.3 / 270.9; 1.23 | 242.5 / 256.2; 0.80 | 211.6 / 267.5; 0.82 | 246.0 / 270.4; 1.23 | 241.6 / 271.4; 1.26 → 216.5 / 259.6; 0.83 |
| 4K HEVC Main 10 | 156.6 / 160.0; 1.83 | 153.8 / 158.5; 0.89 | 155.3 / 157.8; 0.90 | 157.3 / 159.9; 1.81 | 157.1 / 160.6; 1.83 → 154.5 / 159.5; 0.90 |
| 4K AV1 | 213.8 / 251.5; 1.47 | 200.2 / 233.3; 1.01 | 210.3 / 243.7; 1.00 | 205.5 / 247.8; 1.44 | 207.6 / 244.7; 1.46 → 206.0 / 227.0; 0.99 |
| 4K AV1 10-bit | 148.1 / 149.2; 2.06 | 146.5 / 148.0; 1.13 | 147.2 / 148.7; 1.11 | 147.6 / 148.8; 2.05 | 148.4 / 149.1; 2.07 → 147.5 / 148.8; 1.10 |
| 1080p H.264 High | 408.8 / 471.7; 0.86 | 380.8 / 409.7; 0.70 | 400.3 / 427.4; 0.66 | 414.1 / 473.8; 0.88 | 393.3 / 451.1; 0.88 → 401.9 / 477.1; 0.84 (under 8 MiB: copied as it is) |

- Blocking saves what the copies spun, 0.16 to 0.20 ms of a 1080p frame,
  0.4 to 0.5 of 4K 8-bit's and 0.9 to 1.0 of 10-bit's, and the frame
  comes later: the device write took up to 0.5 ms more. One arm's rounds
  spread over up to a quarter of their rate at 8 bits and 1080p, and 3%
  at 10, but the medians fell the same way bench after bench: 4K 8-bit's
  in 11 of 12, by up to 12% (8 of them by 3% or more), 1080p H.264's by
  2% to 13%, and 10 bits' by 0.2% to 1.8%. A first run of the blocking
  context alone, 7 rounds of its own, agreed: up to 11% lost at 8 bits
  and 1080p (one of the six medians 2% faster), 0.6% to 1.5% at 10.
- One wait for both copies changes nothing while it spins. When it
  blocks it saves the same, and loses less at 1080p and more on 4K HEVC
  Main sync.
- So the backend keeps the driver's scheduling: each way to block costs
  throughput beyond the noise for less CPU. Two ways might keep the
  saving without the loss, neither built. A blocking wait for long copies
  alone, from 4K 10-bit's 25 MB, would save 0.9 ms of those frames for
  0.2% to 1.7% of their rate, as the event arm measured them, but its
  bound would rest on two sizes. A wait of our own that sleeps through
  most of a copy's expected time and spins for the rest would not wake
  late, at the cost of a guess at each copy's length.

## Tests

The crate's unit tests need no GPU. Thirteen ignored tests need the driver
and run on an NVIDIA host: `layout`'s ("The binding");
`jpeg::tests::the_hardware_decoder_opens_and_decodes_its_trial_image`,
which needs libnvjpeg too ("The JPEG engine");
`motion::gpu::print_what_nvenc_answers` and `motion::gpu::measure_presets`
("Motion estimation"); the waits' seven, `decode::gpu`'s six and
`nvenc::encoder::tests::a_lock_past_its_deadline_stalls_the_session_until_nvenc_is_done`,
and `nvenc::encoder::tests::what_each_lock_returns`, which prints what
NVENC's lock answers ("How a session waits for the GPU");
`decode::gpu::a_picture_nvdec_fails_fails_each_after_it_until_an_i_picture`
("The decoders"). The runs below
drive the crate's examples on that host. `tools/decode-check.sh` runs the
decode check over directories of streams ("Decode"); the rest are run by
hand, so each section says what was run and what it found.

### Decode

The streams are those the VA-API checks' recipes write
(`crates/reims-vgpu-video-vaapi/tools/h264-check.sh`, `hevc-check.sh`,
`vp9-check.sh`, `av1-check.sh`), written by x264, x265, kvazaar, vpxenc,
aomenc, SvtAv1EncApp and rav1e from ffmpeg's lavfi sources, and an 8K
stream of each codec but H.264. All are test oracles, never linked.
kvazaar's streams were written by `hevc-check.sh` on another host
([vaapi.md](vaapi.md), "Testing HEVC"), the rest on the NVIDIA host.
- Each stream goes through core's host session with the conformance
  suite's stand-in guest (`conformance DEVICE decode-check STREAM
  REFERENCE`), once synchronously and once not, and every frame's NV12 or
  P010 MD5 must be the reference's.
- References: ffmpeg's software decode; x265's own reconstruction for HEVC,
  where ffmpeg is wrong on CTB-16 streams ([bugs.md](bugs.md), U1: NVDEC
  agrees with x265 on main-1366x766-ctu16, as the RX 7900 XTX and the
  Radeon 780M do).
- VP9 and AV1 also go whole through `nvidia-framemd5 --planar`, at each
  frame's own size, against two software decoders' planar MD5s (ffmpeg's
  vp9 and libvpx; libdav1d and libaom), which convert nothing and so are
  exact at odd sizes. A stream of either that changes size goes through
  the session whole (`conformance DEVICE framemd5`), against the direct
  run's packed frames, each with its sample and size (`ID WxH MD5`), since
  the session lends each such frame a buffer of its own size and the
  stand-in guest hashes it so.
- Expected refusals must fail as unsupported, or as too small.
- Streams with frames NVDEC decodes wrongly (U7) pass when the backend
  refuses those, at the open or frame by frame (`nvidia-framemd5
  --keep-going`), and every frame it shows matches. A whole decode of one
  means a driver decodes them, and the refusal can go.

`tools/decode-check.sh [stream dir...]` runs the first of these over
directories of the VA-API checks' streams, copied here, `build/hevc-check`
by default: every MP4 with a reference beside it (`hevc-check.sh`'s
`NAME.expect`, `h264-check.sh`'s `NAME.ref`, the packed `NAME.vp9.guest`
and `NAME.libdav1d.guest`), and kvz-720p-ref15, whose first 9 frames must
come out, its tenth with -12910 and no image and the 20 after it with
-12909, both ways. The rest it leaves to the runs by hand, each in a NOTE
line: kvz-720p-tiles-narrow, the streams that change size, and those with
no reference, such as the profiles `hevc-check.sh` checks as refused.

What the streams cover:

| codec | streams | result |
|---|---|---|
| H.264 | 9 | all bit-exact, 18 session runs |
| HEVC | 20 of x265's, 13 of kvazaar's, 14 of hevc-craft's | x265's all bit-exact against x265, 40 session runs; ffmpeg differs on main-1366x766-ctu16 (U1). kvazaar's eleven bit-exact against hevc-check.sh's reference both ways, 22 session runs: tiles 3x3 and 4x3 explicit in one slice, 2x2, 2x3 and 4x4 in a slice each (1080p, 1366x766, 4K), WPP rows in dependent slice segments with and without an open GOP's CRA and RASL pictures, low-delay B, 8 current references, slice headers with their own short-term sets, and 60 field pictures of 1280x360. kvz-720p-ref15's first 9 frames matched, and its tenth picture, with 9 in `RefPicSetStCurrBefore`, was refused ("NVDEC takes 8"): -12910, then -12909 for the 20 after it, both ways. kvz-1080p-ref1, with one reference and no temporal prediction, bit-exact both ways. hevc-craft's 14 bit-exact against their sources' references both ways, 28 session runs: identity list modification in P and B slices and list 0 put back after long-term pictures, short-term sets predicted in the SPS and in slice headers, and long-term pictures from the SPS's candidates and the headers' own ([vaapi.md](vaapi.md), "Crafted HEVC streams") |
| VP9 | 32, and 4 refused | all bit-exact whole against both decoders, odd sizes (641x361, 643x363 at 10 bits), three resizes (p0-720p-resize and p2-720p-resize through the session whole, 240 of 240 frames, p0-720p-resize-early 150 of 150), 4K and 8K, and vp9-craft's intra-only, shown-existing and segment-feature streams; 64 session runs; p0-63x65 refused as too small |
| AV1 | 66, 6 refused for a 4n+3 side, 3 refused | all 66 bit-exact whole against libdav1d and libaom, and through the session both ways, 132 session runs: libaom, SVT-AV1 and rav1e, 8 and 10 bits, tiles and tile groups, loop restoration, superres fixed and random, odd sizes (641x361), resizes with scaled references (aom-720p-resize, and aom-360p-resize-569 and -465-10bit's 569x320 and 465x262 frames between 640x360 key frames, each through the session whole at its own size), switch frames, error resilience, screen content (palette, and intra block copy in an intra frame), delta q and lf, lossless, monochrome, film grain (SVT's, rav1e's, and libaom's 16 test vectors at 8 bits and 4 at 10), 4K and 8K at 10 bits, a sequence allowing 16384x8704, and 642x362 and 644x364 beside 643x363. Checked with `av1-info --frames` for what they hold: aom-720p-fwdkf, a forward key frame shown again from its slot; aom-720p-kfflt2, key frames hidden and shown through overlays; aom-720p-gm-zoom and aom-720p-gm-pan-10bit, 17 and 22 ROTZOOM models; svc-720p-l1t2 to -l3t3, OBU extension headers and up to nine operating points, the spatial layers' lower frames, 320x180 and 640x360, references to the 1280x720 frame emitted. Refused for a 4n+3 side (U7), every frame shown matching: five streams at their open, and aom-360p-resize-427's 27 frames of 427x240 with its 3 key frames shown. Three refused by NVDEC's caps, `aom-63x65` among them, as too small |

Results, each with no VM running:
- 2026-10-24, the AV1 streams alone (libaom 3.15.2): 207 PASS, 0 FAIL, 0
  NOTE.
- 2026-10-24, the HEVC streams alone, kvazaar's among them: 62 PASS, 0
  FAIL, the NOTE on main-1366x766-ctu16.
- 2026-10-24, with decoding ahead in core ([core.md](core.md), "Decoding
  ahead"), which NVIDIA does not opt into, over the same streams: 335
  PASS, 3 NOTE, 0 FAIL.
- 2026-10-25, with core leaving the smallest size to the backend and a
  VP9 frame of another size shown at its own size ([devices.md](devices.md),
  "Where we differ from Apple"): 363 PASS, 1 NOTE (U1), with kvazaar's 12
  streams not run. A check of the smallest sizes, on streams written on
  another host, passed 94 of 94: H.264 from 48x64 decoded, every frame
  matching, where AVD refuses under 64; HEVC under 144x144 and VP9 and
  AV1 under 128x128 refused at `dsta` with -12911.
- 2026-10-25, every stream, zero copy on ("Zero copy"): 387 checks passed,
  1 note (U1), none failed.
- 2026-10-25, with a frame NVDEC fails failing the frames after it ("The
  decoders"), the H.264 and HEVC streams alone, kvazaar's among them,
  through the session with zero copy and without: 68 of 68 runs each,
  the three refusals at `dsta`, and kvz-720p-ref15's 9 frames, -12910 and
  20 refusals, both ways.
- 2026-10-25, `tools/decode-check.sh` over the HEVC streams, with
  x265's main-720p-ref1, kvazaar's kvz-1080p-ref1 and hevc-craft's 14
  added: 92 PASS, 0 FAIL, and NOTE lines for kvz-720p-tiles-narrow and
  the three profiles `hevc-check.sh` checks as refused, which have no
  reference. The 60 checks of the streams before them passed as they did
  before the new ones were added.

### Damaged input

`examples/nvidia-damage` decodes a stream through the backend with every
sample damaged, then the whole stream again on a new decoder in the same
process. The damage, from a seeded generator: 4 or 200 bytes set at
random (`flip:4`, `flip:200`), 16 bytes set past each sample's first 8
(`tail:16`), each sample cut to 50 or 3 percent (`cut:50`, `cut:3`), or 20
percent of the samples dropped, references and all (`drop:20`). A run
passes when nothing crashes or takes over 5 s, the whole pass gives the
reference's frames, and the kernel logs no new NVIDIA Xid.

Result (2026-10-24): 144 of 144 runs passed, 8 streams (H.264 High and
Main with open GOPs, HEVC Main and Main 10 with slices and scaling lists,
VP9 Profile 0 with tiles and Profile 2, AV1 with tiles and with film
grain at 10 bits) by 6 damages by 3 seeds.
- Every refusal was the backend's own error: 8,589 bad data and 2
  unsupported, no malfunction. Most damaged frames decode with NVDEC
  reporting the decode failed, which the copy answers as bad data.
- The slowest decode call took 79 ms and the slowest copy 12 ms.
- The Xid count stayed at 1, an earlier line, from before both runs.
- A run while one of our VMs booted on the GPU failed 6 of 144: in the
  undamaged pass on a new decoder, an H.264 frame's decode was reported
  failed (B37). Each of those passed when run again.

Run again (2026-10-25) on four H.264 and HEVC streams, with each frame
after one NVDEC fails refused until a random access point ("The
decoders"): H.264 Main with an open GOP and High at 1080p, HEVC Main with
slices and Main 10 with slices and scaling lists, by the same 6 damages
by 3 seeds. 72 of 72 runs passed: 5,442 bad data and 6 unsupported, no
malfunction; the slowest decode call took 34 ms and the slowest copy
6.6 ms; the kernel logged no new Xid.

### Several processes at once

Six streams (H.264 High, and Main with an open GOP, at 1080p; HEVC Main
10 with slices and scaling lists; VP9 Profile 0 with tiles; AV1 at 720p
and 4K at 10 bits) decoded from one or more worker processes, round after
round. Each stream goes straight through the backend (`nvidia-framemd5`,
its frames' MD5s against the reference's) and through core's host
session (`conformance DEVICE decode-check`, synchronously and not). A
direct run's failure is NVDEC's or the backend's, and one in the session's
runs alone is the session's. Some runs had another load beside the
workers, and the GPU's compute processes were sampled every 2 s, so a run
says what else was on the GPU.

Results (2026-10-24), each 12 rounds unless said, the Xid count at 1
throughout:

| run | beside the workers | runs passed |
|---|---|---|
| 1 | nothing: two workers, 30 rounds | 1,080 of 1,080 |
| 2 | the damaged-input runs, three times, one worker | 648 of 648, and 432 of 432 damage runs |
| 3 | U7's repro decoding AV1 and VP9 at 643x363, then at 642x362 and 641x361, then 643x363 again; one worker | 648 of 648, beside 724 repro runs |
| 4 | 4K x265 encodes loading the CPU (load 19 of 24 threads), then U7's repro, its encodes and decodes; two workers | 864 of 864 |
| 5 | 8K decodes (HEVC Main 10, VP9, AV1 10-bit), twice; two workers | 864 of 864, beside 89 rounds of the three |
| 6, 7 | one of our VMs booting, with no video device, then idling, twice; two workers | 420 of 432, then 424 of 432; and 432 of 432 each right after |

So NVDEC's failures of valid frames are B37: they came only with one of
our VMs booting on the GPU, in the minutes reims-vgpu logs its 2 s stamp
waits (B26), as direct runs' status 8 and the session's frames without an
image, lost replies and, in VP9, wrong frames with no status.

Run again (2026-10-25), two workers decoding H.264 High and HEVC Main 10
at 1080p and AV1 10-bit at 2160p straight through the backend
(`nvidia-framemd5`). Beside a boot of the guest, 10 of 223 runs failed, in
three bursts, two in the seconds of the guest's long command buffers
(B26). Beside kernels of our own that load the GPU as those do, with the
same signature in `nvidia-smi dmon` (SMs 100% busy, device memory idle),
480 runs passed: scattered stores to guest-like memory in command buffers
of 1.2 s and of 24 s, and 2.5 s dispatches of dependent loads, with and
without eight other queues beside them. 348 runs beside nothing passed.
Those loads slowed the decodes up to nine times, and failed none. So B26's
load alone is not what fails them; what is, in a booting VM, is not known
(B37).

Found (2026-10-25, same GPU and driver), beside the two workers and a
decoder of our own on NVIDIA's parser:
- One load does fail them: 16 workgroups of 64 invocations, each adding to
  one 32-bit word with `atomicAdd`, in dispatches of 190 to 280 ms, the word
  in host memory imported with `VK_EXT_external_memory_host`.
- Beside it, every `nvidia-framemd5` run failed, 5 of 5 in each of two
  runs, and our decoder's passes had up to 54 pictures failed and 142 of 150
  frames wrong.
- The same load on a word in device-local memory failed none of 53 runs.
  Neither did the load spread over words, or run in one workgroup, nor about
  25 other loads (memory, pipelines, render and compute passes over host
  memory, arithmetic and threadgroup-memory kernels).
- In a boot, the failures fell in the seconds of the guest's long execs,
  and its libraries named no AIR atomic intrinsic. So the load found is not
  shown to be what the boots run.
- Most wrong frames had status 2 in the boot and beside the load. Two
  copies of each frame always agreed, so the samples are wrong in NVDEC's
  surface.
- It is NVIDIA's ([bugs.md](bugs.md), B37 and U21).

Run again (2026-10-26) at metal2vulkan 6c9cad047a, where B26's command
buffers of seconds take under a second: the 2026-10-24 run, two workers,
12 rounds, beside a boot as in runs 6 and 7, on a build with B26's probe.
432 of 432 runs passed in each of two boots, and 432 of 432 alone after
each. In the second boot the workers ran from QEMU's start through its
longest command buffers (the 794 dispatches' 523 ms), with no stamp wait
past 2 s. So the boots' failures went with B26's seconds; which of their
work failed NVDEC is not known (B37).

### Encode

37 cases, 25 of raw NVENC (`nvidia-encode`) and 12 through core's encoder
session (`nvidia-session-encode`), each decoded back by ffmpeg and
checked: PSNR and SSIM against the source, the bitrate against the
target, the sync samples against the bitstream's IDR pictures, the GOP,
the picture types and reorder, and the parameter sets and configuration
record parsing (2026-10-24). Covered: constant QP, CBR, VBR, QVBR,
B-frames, a forced IDR and GOP lengths, a rate change mid-stream, CAVLC,
low latency, P1 and P7, odd sizes (1366x768, 200x200), 4K, HEVC Main and
Main 10, a key frame interval in seconds, and colour and full-range VUI.
All passed. PSNR ran from 31.1 dB (CBR on heavy noise) to 58.7 dB by
case. H.264's CBR came within 4% of its target, over 3 s of 1080p and
over 2 s of heavy noise at 288p; HEVC's came 24% over on the same 2 s of
noise ([bugs.md](bugs.md), B87); VBR 9 to 17% over across 10 s, and QVBR
19% over across 3 s.

The conformance suite's own encoder cases (`conformance 0 encode-check`,
through core's host session and its stand-in guest) passed 6 of 6:
H.264 and HEVC, 8-bit and HEVC 10-bit sources, with and without
reordering, each 60 samples with 2 sync samples and one configuration
record, PSNR 52.2 to 58.9 dB (2026-10-24).

- **ffmpeg's PSNR converts colour.** ffmpeg 9's psnr and ssim filters read
  a stream's colour tags, so an untagged raw reference against a BT.709
  encode read 23.8 dB for a 48.1 dB encode; the check untags the decoded
  side, and reads a full-range reference as `yuvj420p`.
- **NVENC's VBR overshoots short clips.** Over 2 s, VBR came to 1.3 to 1.6
  times its target, as ffmpeg's `h264_nvenc` did on the same source; over
  10 s, within 1.2. The VBR cases use 10 s sources.

### The bitrate through the session

A guest's `AverageBitRate` alone reaches NVENC as a VBR peaking at 1.05
times it, with a VBV buffer of a second at the peak, and `DataRateLimits`
as one peaking and aiming at two thirds of the limit, with half a second
(the codec crate's `src/encode/session.rs`). NVENC's VBR runs its first
second near its peak, whatever its buffer, so the peak is held near the
average; at the average itself NVENC runs as a constant bitrate and fills
pictures cheaper than their share, which AVE does not.

Measured through the host session (`conformance 0 encode`), against
Apple's AVE on the same frames ([devices.md](devices.md), "The encoder
session"): heavy noise at 352x288 at 5 Mbit/s, `mandelbrot` at 720p at 3
Mbit/s (HEVC 2), `testsrc2` at 1080p at 4 Mbit/s, H.264 and HEVC, with a
key frame at the start only, every 2 s, and with reordering (2026-10-25,
driver 615.71.09). As a share of the average:

| `AverageBitRate` alone | first second | over 3 s | over 10 s |
|---|---|---|---|
| AVE | 0.95 to 1.27 | 0.97 to 1.16 | 0.98 to 1.04 |
| NVENC, peaking at 1.05 times the average | 1.15 to 1.44 | 1.05 to 1.22 | 1.01 to 1.09 |
| NVENC, peaking at 1.5 times | 1.42 to 1.65 | 1.16 to 1.41 | 1.05 to 1.17 |

The first second's excess, most on HEVC over noise, is NVENC's
([bugs.md](bugs.md), B87). Static bars at 5 Mbit/s came to 0.00 of it
on NVENC and 0.01 on AVE. Under `DataRateLimits` alone NVENC made 1.00 to 1.03
of two thirds of the limit over 10 s, its busiest second 0.71 to 0.94 of
the limit; AVE 1.00, and 0.67 to 0.82.

In the encode check (`conformance 0 encode-check`, 35 of 35 passed
either way), `-abr` came to 1.06 (H.264) and 1.07 (HEVC) of its average,
where it was 1.11 and 1.10 at 1.5 times and AVE's is 1.02. The waves at
640x360 and 3 Mbit/s came to 0.44 to 0.87 of the rate, AVE's to 0.30 to
0.94. The case that lost most is `hvc1-420v-reordered-3840x2160`, 1 s of
HEVC at 2160p and 3 Mbit/s: 4,257 kbit/s with its worst picture at 31.3
dB, where 1.5 times made 4,597 at 33.4 dB and AVE makes 4,169 at 43.5.

### The GPU at work

`nvidia-smi dmon -s u` (2026-10-24) while one process decoded 300 frames
of 4K through `nvidia-framemd5`, in about 8 s each: H.264 High's decode
engine at 4 to 17 percent, HEVC Main's at 6 to 8, the encoder at 0. The
rate, about 37 frames a second, is the copy and MD5 of each 12 MB frame
on the host, not NVDEC's: decoding 4K and 8K VP9, AV1 and HEVC in six
processes at once, each mostly in its start-up, gave the decode engine up
to 29 percent. The encoder showed up to 43 percent in the encode check's
4K cases.

### How fast a session decodes

perf-check's five streams (`crates/reims-vgpu-video-vaapi/tools/perf-check.sh`
writes them: 150 frames each of 4K HEVC Main and Main 10, 4K AV1 at 8
and 10 bits, and 1080p H.264 High) go through core's host session with
the stand-in guest, `conformance DEVICE framemd5 --sequence` with
`RVGV_LOG=info` for core's split of each frame's time, every frame
matching the reference as a set. "Zero copy" has the rates as the
backend runs: 148 to 270 fps at 4K and 472 to 565 at 1080p with zero
copy, 117 to 207 and 448 to 523 where the CPU copies.

Where the CPU copies, a frame is read back into the decoder's staging
memory and copied from there ("Output", in "The decoders"). Measured
(2026-10-24, with no other process on the GPU):
- Page-locked memory takes 25 MB in 3.6 ms, about 7 GB/s, where the
  driver's own copy into pageable memory took 4.5 ms; with pageable
  staging, 4K ran at 109 to 171 fps.
- Against one copy by the driver into the guest's pageable pages, which
  would put a driver call inside the lend, the CPU's copy out of staging
  costs 10% to 30% of each 4K stream's rate, the most where it decoded
  fastest; every stream stays above 4K120. The stand-in guest's messages
  were the same either way, by `cmp`.

The backend copies one frame at a time and does not decode ahead
([bugs.md](bugs.md), D17).

### In the guest

An x86_64 guest on the NVIDIA host ran with the video device on this
backend, as the device picks it on an NVIDIA GPU, so the guest's hardware
codecs were this backend's; QEMU's log named it. With VtWaitPatch ([devices.md](devices.md), "What
an x86_64 guest has"), everything passed: the guest's VideoToolbox
decoded in hardware and matched the references for H.264 Main and High
with B-frames, HEVC Main and Main 10, AV1 8 and 10 bits up to 4K, and VP9
profiles 0 and 2; it encoded H.264 and HEVC at 720p and at 1080p with
reordering, every sample reaching the guest, luma PSNR 43.85 to 48.55 at
the least, and the host's dump of each gave the same PSNR. `nvidia-smi
dmon` showed the decoder at 2 to 7% and the encoder at 4 to 17% while
they ran.

In other boots a session's open came 0.9 to 2.4 s after the guest's
`dsta` or `esta`, beside the guest's command buffers that run for seconds
on NVIDIA's Vulkan, and the guest had given up on it ([bugs.md](bugs.md),
B26). The backend's
own open is 24 to 35 ms on this GPU (`examples/nvidia-open-time`: an
H.264 decoder, H.264 and HEVC encoders at 1920x1080, back to back and 15
s apart, 2026-10-24); the process's first open, the backend's first
decoder, took 470 to 685 ms in the boots.

## Limits

- No scaler. The JPEG engine needs libnvjpeg, which the driver does not
  install, and a GPU with nvJPEG's hardware decoder ([bugs.md](bugs.md),
  D23); it refuses 4:1:1 and quantization values over 255, which that
  decoder does not decode right (U20).
- Lookahead is off (D12).
- VP9 and AV1 under 128x128 and HEVC under 144x144 are refused at
  `dsta` with -12911, and frames larger than the decoder's surfaces: so
  VP9's spatial layers, which NVDEC decodes wrongly (D16).
- NVENC's smallest frames, 145x49 for H.264 and 129x33 for HEVC, are
  refused at `esta`.
- NVENC opens 12 sessions at once on the RTX 5080 Laptop GPU, across every
  process: the 13th is refused with `NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY`
  (a probe, 2026-10-24; [bugs.md](bugs.md), B39).
- AV1 frames with a side of 4n+3 are refused, since NVDEC decodes them
  wrongly (U7).
- While one of our VMs boots on the GPU, NVDEC fails valid decodes in
  every process, and VP9's with no status (B37, B26).
- The decoders are synchronous, as core's trait is: each `dfrm` waits for
  its decode's submission, not its completion, which the copy waits for.
