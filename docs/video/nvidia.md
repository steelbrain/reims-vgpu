# NVDEC and NVENC

`crates/reims-vgpu-video`'s `nvidia` module: decode on NVDEC and encode on
NVENC, over the CUDA driver API, with nothing above them. The reims-graphics device's
video decodes on its decoder of whole access units on an NVIDIA GPU
(reims-vgpu's `graphics/nvidia.rs`, [devices.md](devices.md)). Its NVENC
encoder is driven by its example and by reims-vgpu's tests, which encode
the streams they decode. This doc says how the module binds the driver, what
an RTX 5080 Laptop GPU answers, how each part works, and how it is tested.
Runs are on an NVIDIA GeForce RTX 5080 Laptop GPU (driver 615.71.09, CUDA
driver 13.4, on CachyOS) unless they say otherwise.

## What is here

| what | where |
|---|---|
| Loading the driver's libraries, and the function tables | `src/nvidia/dl.rs` |
| The CUDA driver API: a GPU, its primary context, copies | `src/nvidia/cuda.rs`, `src/nvidia/cuda/ffi.rs` |
| NVDEC's binding: caps, decoder, pictures, mapping, status, reconfiguration | `src/nvidia/nvdec.rs`, `src/nvidia/nvdec/ffi.rs` |
| An NVDEC decoder and its pictures: surfaces, stand-ins, the copy out | `src/nvidia/nvdec/decoder.rs` |
| The codec module's accelerators on NVDEC, one a codec | `src/nvidia/nvdec/h264.rs`, `hevc.rs` |
| The decoder of whole Annex B access units, which the reims-graphics device's video drives | `src/nvidia/annex_b.rs` |
| The failures NVDEC reports, which fail the pictures after them | `src/nvidia/decode.rs` |
| Zero copy: guest memory registered with CUDA ("Zero copy") | `src/nvidia/zero_copy.rs` |
| Another API's memory imported into CUDA, for a copy within the GPU | `src/nvidia/external.rs` |
| The waits on the GPU, each bounded ("Waiting for the GPU") | `src/nvidia/wait.rs`, `src/nvidia/nvenc/locks.rs` |
| NVENC's binding, and an encoder of raw frames | `src/nvidia/nvenc.rs`, `src/nvidia/nvenc/ffi.rs`, `src/nvidia/nvenc/encoder.rs` |
| What the GPU answers, as `vainfo` would | `src/nvidia/probe.rs`, `examples/nvidia-probe.rs` |
| Raw NVENC from the command line | `examples/nvidia-encode.rs` |
| The binding checked against NVIDIA's headers | `src/nvidia/layout.rs` (an ignored test) |

## The binding

libcuda.so.1, libnvcuvid.so.1 and libnvidia-encode.so.1 are opened with
`dlopen` when a GPU is first opened, so a host with no NVIDIA driver runs
without them: opening a GPU there fails, and nothing else does. No
crate binds them; the declarations are ours, for the subset used, and no
NVIDIA parser is used: the codec module parses every stream, as for VA-API.

- **Checked against NVIDIA's headers.** `src/nvidia/layout.rs`'s test
  (`cargo test -p reims-vgpu-video nvidia::layout -- --ignored`) writes a C
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
- **Not in the headers.** `cuMemHostRegister_v2` and `cuMemHostUnregister`,
  for zero copy, are not in ffnvcodec's `dynlink_cuda.h`, so the layout
  test does not check them; their signatures are the CUDA driver API's
  (`void *`, a size, flags; `void *`), and libcuda 615.71.09 exports both
  (`nm -D`, 2026-10-25). A driver without them is lent no guest buffer,
  and the log says so once (`src/nvidia/cuda.rs`). Nor does it check
  `cuImportExternalMemory`, `cuExternalMemoryGetMappedBuffer` and
  `cuDestroyExternalMemory`, for imports of another API's memory
  (`src/nvidia/external.rs`).

## What the RTX 5080 Laptop GPU answers

`nvidia-probe` (2026-10-24), the parts the module uses:

| codec | chroma, depth | sizes | output |
|---|---|---|---|
| H.264 | 4:2:0 8-bit | 48x64 to 8192x8192 | NV12 |
| HEVC | 4:2:0 8, 10, 12-bit | 144x144 to 8192x8192 | NV12; P016 above 8 |

Each has 2 decode engines. NVDEC also lists H.264 at 10 bits and 4:2:2,
and HEVC 4:2:2 and 4:4:4, which the device does not offer.

NVENC (`NvEncGetEncodeCaps`): H.264 Baseline, Main, High and High 10 among
others, 145x49 to 4096x4096, 7 B-frames; HEVC Main, Main 10 and the range
extensions, 129x33 to 8192x8192, 5 B-frames. Presets P1 to P7, 2 encode
engines, CBR, VBR and constant QP, lookahead (`SUPPORT_LOOKAHEAD`).

## The reims-graphics device

The reims-graphics device's video, on an NVIDIA GPU, runs on this module's
NVDEC through `AnnexBDecoder` (`src/nvidia/annex_b.rs`), in a host video session of
reims-vgpu's (`crates/reims-vgpu/src/graphics/video_host.rs`, `nvidia.rs`): the
guest's video driver hands each decode a whole Annex B access unit, one
picture, in decode order, and the planes of the output surface it lands in.
The device's arm chooses the runtime device's own GPU, by its PCI address,
where it is NVIDIA's and CUDA opens it (`graphics/video_choice.rs`).

- **The access unit, in place.** The codec module's decoder parses it where it
  lies, in the host bytes of the guest's bitstream buffer, and hands NVDEC each
  picture's slices, units of the access unit's own bytes. Where they lie in it
  as one run (after start codes, only zeros between them), NVDEC is handed that
  run as it is, with no copy (`nvdec::decoder::one_run`); otherwise the slices
  are gathered into one buffer. NVDEC copies the bitstream within
  `cuvidDecodePicture`, so the guest's buffer is read there alone. A session
  logs at its end how many of its pictures went each way.
- **The picture, by one GPU copy**, `cuMemcpy2D` a plane from the mapped
  surface into the plane at its pitch, its crop's rows alone, waited for
  before the session signals the decode's value. Any pitch is taken, so the
  device asks no layout of the guest.
  - **Into the output's device caches**, where CUDA imports the runtime
    device's exportable buffers, which the arm learns once as the video opens
    (a buffer of a 4K picture's bytes made exportable and imported, logged
    with the make's time). The caps then ask the guest for cached outputs
    (`RGFX_VIDEO_CAPS_OUTPUT_CACHED`), the device makes each cache exportable
    as it makes it, and the picture is copied within the GPU into the cache's
    memory, imported once a cache (`src/nvidia/external.rs`) and let go of as the
    cache goes.
  - **Into the output's pages**, where its planes lie in pages a device
    reaches in place (the surface's view of the guest's RAM, or a copy block
    of the device's region): their view is registered with CUDA once a
    mapping, as "Zero copy" registers a guest buffer's, and written by DMA.

  Elsewhere the picture is read back and copied by the CPU.
- **A picture that does not land is a decode error.** A picture whose planes
  the session refuses, or whose decode NVDEC failed once it ran, still
  signals the decode's value, so its command buffer completes, failed with
  Metal's internal error. For HEVC the guest's video driver fails the
  session's next schedule for it (`kVTVideoDecoderMalfunctionErr`), which
  AppleGVA then reports for every frame after; it schedules a picture before
  it queries the one before, so the failed picture and the next reach the
  guest as decoded frames. AppleGVA's H.264 decoder reports no failure a
  driver can answer, so an H.264 picture that does not land reaches the guest
  as a decoded frame (`guest/bundle/src/RgfxVideoStatus.hpp`).
- **Surfaces.** A picture holds its decode surface until it lands; the decoder
  has `AnnexBDecoder::SPARE_PICTURES` (2) beyond its references and the
  picture being decoded, and the session holds no more pictures unlanded than
  that. Pictures land on the session's landing thread, in decode order, while
  the next is decoded: a copy out holds the decoder's handle only for its map,
  copies and unmap, and each ask after its decode's status, so NVDEC decodes
  the next picture meanwhile (`nvdec::decoder`).

## The decoders

Each codec is the codec module's decoder over an accelerator of ours that
fills `CUVIDPICPARAMS` from what the codec module parsed, as the `vaapi`
module fills libva's buffers (`src/nvidia/nvdec/h264.rs`, `hevc.rs`). ffmpeg's
`nvdec_*.c` are the model for the field mapping, and NVIDIA's own parser
the judge where the two differ; every mapping was confirmed by bit-exact
runs against ffmpeg's software decode ([bugs.md](bugs.md), T16), not by
the model.

- **Output.** 8 bits decode into NV12, 10 into P016, which holds each
  sample in the high bits as P010 does. A picture is mapped
  (`cuvidMapVideoFrame64`) once its decode is done, and copied out of the
  mapped surface by the GPU ("The reims-graphics device", above), or read
  back with two `cuMemcpy2DAsync` on the decoder's own stream into the
  decoder's staging memory, page-locked by `cuMemHostAlloc` (pageable where
  CUDA gives none), for the CPU to copy.
- **A failed decode fails the pictures after it.** NVDEC's status of a
  picture's decode (`cuvidGetDecodeStatus`) is read at its copy, and a
  failed one is bad data (`Error::DecodeFailed`). NVDEC reports a slice
  cut short failed (status 9, `cuvidDecodeStatus_Error_Concealed`). The
  failure is noted (`Failures`, `src/nvidia/decode.rs`), and the decoder's next
  decode sets the codec module's damaged state, so that each picture after
  it is refused until an I picture (H.264) or an IRAP picture (HEVC). A
  picture whose copy is never made has its status read by no one: an HEVC
  picture with `pic_output_flag` 0, for one ([bugs.md](bugs.md), B41).
- **H.264.** Slices as Annex B, each after a three-byte start code, with
  `pSliceDataOffsets`; the scaling lists in raster order; the DPB by
  surface index with `FrameIdx` and both fields' order counts.
- **HEVC.** The references by surface index with their order counts and
  long-term flags; `RefPicSetStCurrBefore`, `StCurrAfter` and `LtCurr` as
  indices into them, in the order 8.3.2 derives them, which the codec
  module hands on, since NVDEC builds each slice's lists itself; the
  scaling lists in coding (diagonal) order, as dynlink_cuviddec.h says
  ("scaling lists (diag order)"), 32x32's from `matrixId` 0 and 3;
  `NumDeltaPocsOfRefRpsIdx` from the slice header's own predicted set.

### Concealment

A frame lost from an H.264 stream whose SPS allows no gap in
`frame_num` is stood in for, so that the frames predicting from it
decode, as on VA-API.
NVDEC's surfaces are written by a decode alone, so `NvdecH264::conceal`
gives the last reference's own surface under a picture of its own
(`Picture::stand_in`, `nvdec/decoder.rs`), held until both go. How the
DPB names it follows NVIDIA's parser, measured on 2026-10-24:
- **NVIDIA's parser**, fed a stream through `cuvidParseVideoData` by a
  scratch program that printed each `CUVIDPICPARAMS`'s DPB and decoded
  nothing, was given `crates/reims-vgpu-video/tools/h264-check.sh`'s
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
  ffmpeg decode, and drop20-ref4 59 of 59 with no failure, (**measured**,
  2026-10-24). The concealed frames came to a
  luma PSNR of 21.3 to 26.3 dB against the source for drop20-ref1, which
  is a copy of the last reference as VA-API's, and 27.7 to 30.0 dB for
  drop20-ref4.

### What it refuses

- What the codec module refuses (interlaced H.264, HEVC's range
  extensions), as unsupported.
- A stream past NVDEC's sizes, as unsupported, or under them, as too small
  (`Error::TooSmall`). The device's caps give the guest NVDEC's own sizes
  ([bugs.md](bugs.md), D14).
- An HEVC picture with more than 8 pictures in one of its current
  reference sets, as unsupported: `CUVIDHEVCPICPARAMS` holds 8 a set
  (`current_set`, `src/nvidia/nvdec/hevc.rs`, "NVDEC takes 8").

## The encoder

`NvencEncoder` (`src/nvidia/nvenc/encoder.rs`) is an NVENC session in NVENC's
synchronous mode, the only one on Linux: raw 4:2:0 frames in, NV12 or P010,
H.264 or HEVC access units out, in decode order.

- **Pictures in and out.** Each frame is copied into an NVENC input buffer
  and encoded with `frameIdx` counting from 0 in submission order;
  `NV_ENC_ERR_NEED_MORE_INPUT` means nothing yet, and after a success the
  pending output buffers are locked in submission order, each its picture
  in coding order. A flush sends a picture with the end-of-stream flag
  alone, and the encoder flushes before it is destroyed, since destroying
  it with frames in flight crashed in the driver (in libnvcuvid, under
  `nvEncDestroyEncoder`, with lookahead on).
- **Configuration.** The preset asked (P1 to P7) and its tuning, then:
  lookahead off (below); `frameIntervalP` the B-frames plus 1, B-frames
  never references; NVENC decides each picture's type (`enablePTD`) by its
  GOP length, and an IDR picture is forced where the caller asks for one.
  NVENC writes its own parameter sets, before each IDR picture, and slice
  headers, and keeps its own references. The VBV holds the time asked at
  the peak rate; the VUI's colour and range are the caller's.
- **Its waits are bounded.** A bitstream buffer's lock, which waits for
  the picture, is made on a thread of the session's own and waited for
  until the deadline ("Waiting for the GPU"; [bugs.md](bugs.md), U19).
- **Lookahead is off.** With P6 and P7 at high quality, which turn it on,
  after `NV_ENC_SUCCESS` only the oldest pending output locked; the next
  lock gave `NV_ENC_ERR_INVALID_PARAM`, and frames were lost even after the
  end-of-stream picture ([bugs.md](bugs.md), U6 and D12).

## Zero copy

A picture is copied by the GPU straight into memory the host's device
reaches in place, and the CPU copies nothing (`src/nvidia/zero_copy.rs`). The host
lends the memory as a view of its pages, contiguous in its address space:
reims-vgpu's view of a surface over the guest's RAM. The view's whole pages
are registered with CUDA once per mapping, `cuMemHostRegister_v2` with
`CU_MEMHOSTREGISTER_PORTABLE`, which pins them for every context, and the
registration is kept by the mapping's identity in core's `DeviceImports`,
process-wide, since a range is registered once in a process whichever GPU
registered it: at most 96, the one used longest ago dropped first. It is
unregistered, with its GPU's context pushed, when the mapping goes, before
the hold that keeps the view mapped is let go.

CUDA takes any pitch, so every layout is written by the GPU, and no
dma-buf is made, so udmabuf's limits do not bear on it. What declines, the
picture then read back and copied by the CPU with the reason logged once a
session: a registration CUDA refuses, a copy that fails, or a driver
without `cuMemHostRegister_v2`.

A picture lands in the GPU's own memory through `src/nvidia/external.rs`: a Vulkan
buffer whose memory its device exported as an opaque file descriptor,
imported with `cuImportExternalMemory` on the CUDA device of the same GPU
(CUDA refuses another's) and mapped whole, so that the copy is within the
GPU. The import is freed as the `ImportedBuffer` drops, and the exporter's
buffer must outlive it.

## Waiting for the GPU

Every wait of the module's on the GPU is bounded by core's `DEVICE_WAIT`,
5 s ([core.md](core.md), "Bounded waits"), asked after every 100 µs for its
first 50 ms and every millisecond after (`src/nvidia/wait.rs`):

- a picture's decode, with `cuvidGetDecodeStatus` until it is no longer
  `cuvidDecodeStatus_InProgress`, in place of `cuvidMapVideoFrame64`'s own
  wait;
- its map and copies, with `cuStreamQuery` of the decoder's own stream, in
  place of `cuMemcpy2D`, which returns once its copy is done; copies into
  pageable staging memory, which return only once copied, are made on the
  decoder's call thread, then the stream is asked;
- an encode, with the waiting `nvEncLockBitstream` made on the session's
  call thread and its answer waited for (`src/nvidia/nvenc/locks.rs`, which says
  why `doNotWait` does not serve).

Past the deadline, that picture fails as a malfunction
(`Error::Stalled`), and each later call on the decoder or encoder fails at
once the same way. NVDEC's and NVENC's handles are kept until the GPU is
done with the work, and destroyed then; a call on a call thread that has
not returned is left to return there, and nothing it holds is used,
unmapped, freed or destroyed until it has.

## Tests

- **Unit tests**, on any host (`cargo nextest run -p
  reims-vgpu-video -E 'test(/^nvidia::/)'`), with no driver: the binding's bit-fields and
  statuses, the slices' runs and the copies' rows, the failures, the waits
  and their call threads, NVENC's configuration and its buffers' locks, and
  HEVC's current sets.
- **On the NVIDIA host**, the ignored tests (`cargo nextest run -p
  reims-vgpu-video --run-ignored only -E 'test(/^nvidia::/)'`):
  - `layouts_match_nvidia_headers` (`src/nvidia/layout.rs`): the binding against
    ffnvcodec-headers 13.1.15.0 in `/usr/include/ffnvcodec`, built with the
    host's C compiler ("The binding");
  - `access_units_decode_in_place_and_land_by_the_gpu_as_they_read_back`
    (`src/nvidia/annex_b/tests.rs`): whole access units decoded in place, each
    picture written by the GPU equal to its read back;
  - `what_each_lock_returns` and
    `a_lock_past_its_deadline_stalls_the_session_until_nvenc_is_done`
    (`src/nvidia/nvenc/encoder.rs`): NVENC's locks ([bugs.md](bugs.md), U19).
- **reims-vgpu's NVDEC arm**, on the NVIDIA host:
  `a_picture_lands_in_an_imported_device_cache_as_it_reads_back`
  (`crates/reims-vgpu/src/graphics/nvidia/tests.rs`, ignored), a picture
  copied within the GPU into a device cache CUDA imported, read through the
  runtime as NVDEC's own read back of it.

The module has no conformance suite of its own: the `vaapi` module's
([backends.md](backends.md), "The conformance suite").

## Limits

- NVDEC fails valid decodes in every process while one of our VMs boots on
  its GPU, under contended atomics in imported host memory
  ([bugs.md](bugs.md), B37, U21).
- A truncated slice is found by NVDEC's status, but not in a picture whose
  copy is never made ([bugs.md](bugs.md), B41).
- NVENC runs without lookahead ([bugs.md](bugs.md), D12).
- The smallest frames are NVDEC's and NVENC's own ([bugs.md](bugs.md),
  D14).
