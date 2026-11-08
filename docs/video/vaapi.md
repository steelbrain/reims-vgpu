# The VA-API backend

`crates/reims-vgpu-video-vaapi`: the host's codecs through libva, with no
FFmpeg, GStreamer or other codec layer in between. This doc says what the
crate does, what the drivers answer, and how the decoders and encoders are
tested. Every measured fact names its run: what, how, where and when.

The AMD runs were made on one Linux host with two GPUs on radeonsi (Mesa
26.2.4), "the AMD GPUs" in the section names: an RX 7900 XTX (Navi 31,
`renderD128`) and a Radeon 780M (Phoenix, `renderD129`). The runs on
NVIDIA's VA-API driver were made on an RTX 5080 Laptop GPU (driver
615.71.09). Paths are
the crate's unless they name another.

## What is here

| what | where |
|---|---|
| H.264 syntax and the decoding process above the slice data: NAL units, `avcC`, SPS/PPS/scaling lists, slice headers, order counts, the DPB's marking, lists and bumping. Plain Rust on every host, behind an `Accelerator` trait | the codec crate's `src/h264.rs`, `src/h264/` (`crates/reims-vgpu-video-codec`), re-exported as `h264`; the bumping process HEVC shares, `src/dpb.rs` |
| libva's binding, written by hand for the subset used | `src/va/ffi.rs`, checked by `src/va/ffi_layout.c` |
| libva and libva-drm, loaded at run time, and a wrapper of each function the binding calls | `src/va/libva.rs` |
| A display on a render node, its capabilities, configs and contexts | `src/va/display.rs`, `src/va/caps.rs` |
| HEVC syntax and the decoding process above the slice data: NAL units, `hvcC`, SPS/PPS, scaling lists, short- and long-term sets, slice segment headers, order counts, the DPB, lists and bumping. Plain Rust, behind its own `Accelerator` trait | the codec crate's `src/hevc.rs`, `src/hevc/`, re-exported as `hevc`; the bumping process H.264 shares, `src/dpb.rs` |
| NV12 and P010 surface pools, a frame on one, and the readback into caller planes | `src/va/surface.rs`, `src/va/frame.rs` |
| What the driver's video processing writes into an imported surface, tried at each display's open ("The import check") | `src/va/import_check.rs` |
| The reims-graphics device's decode on VA-API: each picture written in place by the driver's video processing, or copied by the CPU ("The reims-graphics device") | reims-vgpu's `src/graphics/vaapi.rs`, `src/graphics/video_host.rs` |
| VA buffers and a picture's begin, render and end, for every codec | `src/va/buffer.rs` |
| The H.264 accelerator, `H264Decoder` and `Frame` | `src/va/h264.rs` |
| The HEVC accelerator and `HevcDecoder` | `src/va/hevc.rs` |
| AV1 syntax and the decoding process above the tile data: OBUs, `av1C`, the sequence and uncompressed frame headers, tile groups, the eight reference slots. Plain Rust on every host, behind an `Accelerator` trait | the codec crate's `src/av1.rs`, `src/av1/` (`crates/reims-vgpu-video-codec`), re-exported as `av1` |
| The AV1 accelerator and `Av1Decoder`; libva's AV1 buffers | `src/va/av1.rs`, `src/va/av1/ffi.rs`, checked by `src/va/av1/ffi_layout.c` |
| The backend's AV1 half | `src/va/backend/av1_session.rs` |
| `vaapi-probe`, which stands in for `vainfo` | `examples/vaapi-probe.rs` |
| The hardware check and its tools | `tools/h264-check.sh`, `examples/h264-framemd5/`, `examples/h264-info.rs`, `examples/h264-poc1.rs`, `examples/h264-craft.rs` |
| HEVC's hardware check and its tools | `tools/hevc-check.sh`, `examples/hevc-framemd5.rs`, `examples/hevc-info.rs`, `examples/hevc-cut.rs` |
| The AV1 check and its tools | `tools/av1-check.sh`, `examples/av1-framemd5/`, `examples/av1-info/` |
| VP9 syntax and the decoding process above the compressed header: superframes, `vpcC`, the uncompressed header with what each frame inherits, the eight reference slots with their sizes, per-segment quantizers and loop filter levels. Plain Rust on every host, behind an `Accelerator` trait | the codec crate's `src/vp9.rs`, `src/vp9/`, re-exported as `vp9` |
| The VP9 accelerator and `Vp9Decoder`; libva's VP9 buffers | `src/va/vp9.rs`, `src/va/vp9/ffi.rs`, checked by `src/va/vp9/ffi_layout.c` |
| The backend's VP9 half | `src/va/backend/vp9_session.rs` |
| The VP9 check and its tools | `tools/vp9-check.sh`, `examples/vp9-framemd5/`, `examples/vp9-info.rs`, `examples/vp9-craft.rs` |
| The JPEG engine's decoder, `VaapiJpegDecoder`, over the codec crate's `jpeg` parser; libva's JPEG buffers | `src/va/jpeg.rs`, `src/va/ffi/jpeg.rs`, checked by `src/va/ffi/jpeg_layout.c` |
| The JPEG check and its tools | `tools/jpeg-check.sh`, `examples/conformance.rs` (its `jpeg-check`), `tools/jpeg-planes.c` (libjpeg's raw planes, the reference) |

`src/va` is Linux's alone (lib.rs picks it with `cfg`). Every `unsafe`
block of the crate is there, each with its reason. On Linux the library
links neither `libva` nor `libva-drm`: it loads them when its first
display opens ("Loading libva").

### Loading libva

The binding's 37 functions come from `dlopen` and `dlsym`
(`src/va/libva.rs`), as the nvidia crate loads NVIDIA's libraries, so a
QEMU built with this crate starts on a host without libva.
- Each function has a wrapper of the same name and signature, which
  `ffi` and `ffi::encode` re-export, so no call site names the loaded
  table.
- `Display::open` loads `libva.so.2` and `libva-drm.so.2` first, once for
  the process, and keeps the answer. A host without them gets
  `Error::Load` from every open, with the loader's reason: "cannot load
  libva.so.2: ...".
- Every other libva call needs a display that `open` made, so no wrapper
  runs before a load that succeeded. The libraries are never unloaded,
  since libva loads the driver, whose threads and state outlive a display.

Measured on the RX 7900 XTX (2026-10-24): the crate's tests and its layout
checks pass, the `conformance` example's ELF needs no libva (`readelf
-d`), and the conformance suite passed, decode 80 of 80, encode 4 of 4 and
the backend's scaler, since removed, 106 of 106 with 19 refused. With
libva hidden, a QEMU with the video device starts and offers nothing
([backends.md](backends.md), "What each backend loads").

### Asking the driver once

A display keeps the driver's first answer to every query a session makes:
the profiles, each one's entrypoints, every pair's config attributes and
surface attributes, and the image formats (`src/va/caps.rs`, `Answers`).
`Display::open` asks them all on the thread that opens it
(`Display::prime`), for every pair the driver lists and for the union of
the attributes the decoders, the JPEG decoder and the encoders ask
(`ASKED_ATTRIBUTES`). A session, on a thread of its own, then reads the
opening thread's answers and asks the driver nothing. A query that failed
is not kept, and is asked again where it is next needed.

That is for NVIDIA's VA-API driver, which answers some of them on the
opening thread alone ([bugs.md](bugs.md), U5). A probe on libva (not in
this repository) asked everything the crate asks on the opening thread and
on a second one, and made an H.264 session's config, surfaces and context
on the second:
- libva-nvidia-driver 0.0.18 on the RTX 5080 (2026-10-24): 38 answers
  differ, all of them `VAConfigAttribMaxPictureWidth` and `Height` (8192,
  4096, 4080, 2048, and 32768x16384 for JPEG, on the opening thread; 0 on
  the second), on each of the 19 pairs that have them. The profiles, the
  entrypoints, the other 16 attributes, the surface attributes and the
  image formats are the same on both threads, and the session's objects
  are made on the second;
- radeonsi on both AMD GPUs (2026-10-24): no answer differs.

With the answers kept (2026-10-24), on the RTX 5080 core's host session
decoded both H.264 test streams on that driver, and a guest there decoded
them in hardware, every frame equal to the software decode of the M3 Max
(macOS 27.0.1). On the RX 7900 XTX the codec checks and the conformance
suite passed.

## The AMD drivers

Run 2026-10-24 on the AMD host (CachyOS, kernel 7.2.9-1-cachyos), with
`cargo run --example vaapi-probe -- /dev/dri/renderD128
/dev/dri/renderD129`. libva 2.24.1 (`pacman -Q`), VA-API 1.24 (the
driver's answer to `vaInitialize`).

| node | PCI | vendor string |
|---|---|---|
| `renderD128` | 1002:744c | Mesa Gallium driver 26.2.4-arch3.1 for AMD Radeon RX 7900 XTX (radeonsi, navi31, ACO, DRM 3.64, 7.2.9-1-cachyos) |
| `renderD129` | 1002:15bf | Mesa Gallium driver 26.2.4-arch3.1 for AMD Radeon 780M Graphics (radeonsi, phoenix, ACO, DRM 3.64, 7.2.9-1-cachyos) |

Both report the same, but for AV1 decode:

| codec | decode (VLD) | encode (EncSlice) | max size |
|---|---|---|---|
| H.264 | ConstrainedBaseline, Main, High: YUV420, NV12 | the same three | 4096x4096 |
| HEVC | Main, Main10: YUV420, YUV420_10, NV12/P010 | Main, Main10 | 8192x4352 |
| VP9 | Profile0, Profile2: YUV420, YUV420_10 | none | 8192x4352 |
| AV1 | Navi 31: Profile0 and Profile2, YUV420, _10, _12 (NV12, P010, P012). Phoenix: Profile0 only, YUV420 and _10 | Profile0 (YUV420, _10) | 8192x4352 |
| JPEG | Baseline: YUV420/422/444/400 | none | 16384x16384 |

- Not offered: H.264 High10, 4:2:2 or 4:4:4; HEVC 4:2:2, 4:4:4, 12-bit or
  SCC; VP9 Profiles 1 and 3; VP8, MPEG-2, VC-1.
- Encode attributes, the same for every encoder: rate control 0x416 (CBR,
  VBR, CQP, QVBR), 128 slices at most. H.264 takes packed headers 0x1f and
  references 0x10002 (two in list 0, one in list 1). HEVC and AV1 take
  references 0x1, one in list 0 and none in list 1: no B-frames. AV1's
  packed headers are 0x1b.
- Encode surfaces start at 128x128 (HEVC 130x128), decode surfaces at
  64x64 (VP9 and AV1 16x16).
- **Video processing**: `VAProfileNone` / `VAEntrypointVideoProc` on both,
  RT formats YUV420, 422, 444, 400, 420_10, 420_12, RGB32, RGB32_10;
  surfaces NV12, YV12, I420, P010, P016, P012, Y800, UYVY, YUY2, 422V,
  444P, RGBA, ABGR, BGRA, RGBX, BGRX, ARGB, AB30, AR30, XB30, XR30 up to
  16384x16384. `vaQueryVideoProcFilters` lists one filter: Deinterlacing.
  Scaling and colour conversion are the pipeline's own, not filters, and
  are what the JPEG decoder's conversions use ("The JPEG decoder").
- **Motion estimation: none.** Every pair either GPU lists is VLD,
  EncSlice or VideoProc. Neither lists `VAEntrypointFEI` (11), whose
  encode writes its vectors to a `VAEncFEIMVBufferType` buffer, nor
  `VAEntrypointStats` (12), whose `VAStatsMVBufferType` holds them
  (libva 2.24.1's `va.h`). So no motion search is reachable through VA-API
  here, and the backend offers the guest no motion-estimation processor:
  `meps` lists none ([devices.md](devices.md), "This host's motion
  estimation").
- **Image formats** (`vaQueryImageFormats`): NV12, P010, P012, P016, I420,
  YV12, YUYV, YUY2, UYVY, Y800, 444P, 422V, RGBP, BGRA, RGBA, ABGR, ARGB,
  BGRX, RGBX, AR30, AB30, XR30, XB30.
- **Surface memory types**: 0x68000001, that is VA, DRM_PRIME, DRM_PRIME_2
  and DRM_PRIME_3, for every config. **No `USER_PTR`**: radeonsi cannot
  wrap guest memory as a surface directly. So a guest's buffer reaches the
  driver only as a dma-buf import ("Zero copy"), and is otherwise copied
  into ("The copy back").

## The import check

Both ways the backend has a driver's video processing write memory
imported as a linear surface, the staged copy back and zero copy, rest on
what the driver does with such an import, and a driver may take the
import and its run, answer success, and write the wrong bytes or none
([bugs.md](bugs.md), U16, U26). So each display learns it as it opens, on
the opening thread, from a trial run of each of the decoders' formats,
NV12 and P010 (`src/va/import_check.rs`, `learn`):
- a picture holding every sample value of its format, 328x64, is put
  into a surface of the decoders' kind by `vaPutImage`;
- one video-processing run copies it into memory of the check's own,
  imported as zero copy imports a guest's buffer (one dma-buf of both
  planes);
- every byte of that memory is compared: each plane's samples, and every
  byte around them, the rows' padding included, left as it was.

It tries rows and plane starts 64, 128 and 256 bytes apart, in turn. Each
pitch is an odd multiple of its alignment and each plane starts an odd
multiple of it into its page, so a driver that takes only a larger
alignment refuses the import or misplaces the picture. Where a run
answers success and writes the picture wrongly, a second run at that
alignment tries rows of whole alignments, the luma plane at the dma-buf's
first byte and the chroma plane whole rows after it. The first that
passes is the format's `ImportWrite`: its alignment, and whether it
writes whole rows alone. Where none passes, the driver is lent no import
of that format, and its frames come back through `vaGetImage` ("The copy
back"). The display's open logs both answers, as `in place: NV12
Ok(ImportWrite { alignment: 256, whole_rows: false }), P010 ...`.

What the drivers answered (**measured**, libva 2.24.1, 2026-11-07):

| driver, GPU | NV12 and P010 |
|---|---|
| radeonsi (Mesa 26.2.6), Radeon 780M (Phoenix, 1002:15bf and 1002:1900) and RX 7900 XTX | 256 bytes |
| iHD 26.3.5, Iris Xe of an i9-13900H | 64 bytes, whole rows ([bugs.md](bugs.md), U26) |

The check has not run on NVIDIA's VA-API driver, which wrote nothing
into an import and answered success (U16); its byte comparison fails
such a driver.

**What it costs**, at each open (`examples/vaapi-probe`, three runs a
GPU): on the Radeon 780M 12.6 to 18 ms at the process's first open, of an
open of 28.6 to 128.9 ms, and 10.3 to 12.7 ms at a second; on the RX 7900
XTX 17.3 to 38.4 ms; on the Iris Xe 9.9 to 36.4 ms, of opens of 11.8 to
104.4 ms.

**The sweep**, `the_check_lends_no_way_a_decoder_shape_comes_out_wrong_through`
(ignored: it needs a VA-API GPU), puts each of the decoders' shapes, at
every pitch a guest's buffer may have, through each layout the check can
answer, and fails where a layout the answer lends comes out wrong. It
passed on the three radeonsi GPUs and the Iris Xe (2026-11-07).

## The copy back

Each decoded frame is copied out of the driver's surface into the guest's
buffer. A decode surface is in video memory and tiled on both AMD GPUs,
so the CPU cannot read it as it is laid out, and reads it uncached
over PCIe. The pool's surfaces come back one of four ways
(`src/va/surface.rs`, `Readback`; `src/va/readback.rs`):

- **Staged**, the default. The pool makes a memfd of its own, sealed
  against shrinking, makes it a dma-buf with udmabuf, maps it
  (`src/va/host_memory.rs`), and imports the dma-buf as a linear VA
  surface of the pool's format (`VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2`,
  one object, rows rounded up to 256 bytes, each plane at a 4096 boundary
  that is also whole rows of the luma's pitch from the first byte,
  `HostFormat::layout`), with a video-processing context (`VAProfileNone`,
  `VAEntrypointVideoProc`) on it. Each frame: the decode surface's sync
  (the *wait*), one processing run into the import, a `poll` on the
  dma-buf for its writes and its sync (the *transfer*: the GPU writes
  system memory; "Zero copy", "The driver's lock", "Waiting for the
  GPU"), then a copy of the cropped rectangle out of cached memory into
  the guest's planes (the *copy*). Every frame of every check is bit-exact
  through it, on both GPUs (below). A driver copies back so only where
  the import check passed on it at its display's open ("The import
  check", `Display::writes_imports` in `src/va/display.rs`), since a
  driver says nothing of what it gets wrong here: NVIDIA's VA-API driver
  (libva-nvidia-driver 0.0.18 on the RTX 5080, 2026-10-25) took the import
  and the run, answered success, and every frame that came back was wrong,
  the 120 of main-720p one MD5 ([bugs.md](bugs.md), U16). Where the check
  passed on neither of a driver's formats, its frames come back through
  `vaGetImage`, with a line in the log saying why. The planes' layout
  serves both ways the check passes, planes at an alignment or in whole
  rows, so iHD, which writes whole rows alone, copies back so too.
- **`vaGetImage`** into one image per pool, mapped, then the same copy:
  the fallback where the import or the context cannot be made, with a
  line in the log saying why, as with no `/dev/udmabuf` the process may
  open (on the AMD host it is `root:kvm`, mode 660, and the user is in
  `kvm`) or a frame past udmabuf's `size_limit_mb`: at its stock 64 MiB,
  an 8192x4352 P010 frame (107 MB staged) is past it, and an 8-bit one
  (53 MB) or a 4K P010 one (25 MB) is not. A staged run that fails later
  logs a warning and moves the pool to it.
- **`vaDeriveImage`**, where the driver makes no image.
- **Mapped**: `vaExportSurfaceHandle`, the dma-buf mapped, and
  `DMA_BUF_IOCTL_SYNC` around the copy. It is refused with the export's
  layout unless the surface is linear, which no decode surface on either
  AMD GPU is.

**Read back, then copied.** Each way runs in two steps (`Frame::read_back`,
then `ReadBack`'s copies): every driver call first, the wait, the
processing run and its sync, `vaGetImage` and the map, or the export's
`DMA_BUF_IOCTL_SYNC` start; then the CPU's copy of the rectangle alone,
with the pool's way back held locked between the two; then the unmap, the
derived image's destroy or the sync's end as the read back drops. Core
lends the guest's buffer around the copy alone, so no VA call runs inside
a lend ([core.md](core.md), "The copy into the guest's buffer").

A comparison picks one for every pool of the process (`read_back_by`, a
`ReadbackWay`); the library reads no environment, and the framemd5
examples take it from `RVGV_VAAPI_READBACK` (`staged`, `getimage`,
`derive`, `mapped`). H.264's concealment copy (`Surface::copy_to`) has an
image of its own, whatever the readback. The framemd5 examples print each
part's mean, as `readback a frame: wait X ms, transfer Y ms, copy Z ms`.

**What the ways gave** (a probe on libva, not in this repository: a
surface of the decoders' kind filled through `vaPutImage`, each way checked
byte for byte; both AMD GPUs, Mesa 26.2.4, 2026-10-24):
- `vaDeriveImage` gave wrong bytes on both GPUs: its mapping is the tiled
  surface. Unmapping it wrote the mapping back into the surface, which then
  read wrong every other way too, so the probe fills the surface again
  before each way.
- The export is not linear: modifier `0x200000028a01f04` on Navi 31,
  `0x200000010401b04` on Phoenix.
- Mesa has no `vaCopy`: it answers `VA_STATUS_ERROR_UNIMPLEMENTED` (20).
- `vaGetImage` with its copy out on 2, 4 or 8 threads gained nothing.
- An import's rows must be a multiple of 256 bytes: 1920 bytes was
  refused with `VA_STATUS_ERROR_ALLOCATION_FAILED` on both GPUs, 1280 and
  3840 were taken (a 1080p, 720p and 4K NV12 import whose rows are the
  frame's).
- A processing run into an import laid out as the destination costs the
  GPU what the staged one does, and leaves no copy for the CPU: 4K NV12,
  2.69 against 2.63 ms on Navi 31, 1.71 against 1.70 on Phoenix.

**Per frame, direct** (`tools/perf-check.sh`, the codec's framemd5
example, in display order, 150 frames each, 2026-10-24):

| stream, GPU | `vaGetImage`: transfer, copy | staged: transfer, copy |
|---|---|---|
| 4K HEVC Main, Navi 31 | 14.08, 1.39 ms | 2.89, 1.58 ms |
| 4K HEVC Main 10, Navi 31 | 14.81, 2.96 ms | 5.41, 3.20 ms |
| 4K AV1 10-bit, Navi 31 | 11.97, 2.99 ms | 5.74, 3.40 ms |
| 1080p H.264 High, Navi 31 | 2.01, 0.10 ms | 1.06, 0.18 ms |
| 4K HEVC Main, Phoenix | 9.49, 1.32 ms | 1.99, 1.53 ms |
| 4K HEVC Main 10, Phoenix | 8.91, 2.80 ms | 3.93, 3.15 ms |

Navi 31 is on a PCIe 4.0 x4 link (downgraded): its staged transfer of a 4K
P010 frame, 24.9 MB in 5.4 ms, is about 4.6 GB/s, near that link's
practical limit.

**Through the host session**, the suite's `framemd5` (`vaapi-conformance
NODE framemd5`) with the stand-in guest, every frame checked as a set, fps sync / async (measured with the
session decoding one frame at a time; [core.md](core.md), "Decoding
ahead", has the staged rates with decoding ahead):

| stream | Navi 31, `vaGetImage` | Navi 31, staged | Phoenix, `vaGetImage` | Phoenix, staged |
|---|---|---|---|---|
| 4K HEVC Main | 40.1 / 40.0 | 118.4 / 118.6 | 52.4 / 51.6 | 129.1 / 118.8 |
| 4K HEVC Main 10 | 37.4 / 39.7 | 53.6 / 53.5 | 52.7 / 53.2 | 88.5 / 86.9 |
| 4K AV1 | 41.0 / 41.5 | 126.5 / 135.0 | 55.2 / 55.3 | 144.0 / 146.3 |
| 4K AV1 10-bit | 38.6 / 39.0 | 59.5 / 56.1 | 55.0 / 59.6 | 101.9 / 101.5 |
| 1080p H.264 High | 232.6 / 229.1 | 294.0 / 252.1 | 271.7 / 273.9 | 358.4 / 366.0 |

The streams are perf-check's own: 150 frames, x265 `--preset faster
--bframes 3` at 20 Mbit/s and x264 High at 8 Mbit/s from `testsrc2` with
noise, SvtAv1EncApp preset 10 from `mandelbrot`. In the session, a frame's
copy includes the wait for its whole decode, as each frame is copied as
soon as it is decoded. These are single runs over SSH.

## Zero copy

Where the guest's buffer is one radeonsi can take, the frame is written
into it by the staged copy back's processing run, aimed at the guest's
pages instead of ours, and the CPU copies nothing (`src/va/zero_copy.rs`;
[core.md](core.md), "Zero copy"). reims-vgpu makes a dma-buf of ranges of
the buffer's bytes: a udmabuf of the memfd behind the guest's RAM, their
pages in the order the bytes lie in
(`crates/reims-vgpu-vulkan/src/os/linux/device_lend.rs`, `dma_buf_of`).
The backend imports the buffer once per mapping, frame size and format,
in strips of the frame's rows ("What udmabuf takes", below): each strip an
even number of rows, a linear VA surface over one dma-buf of its luma and
chroma rows, at the guest's own offsets and pitches (`DRM_PRIME_2`, one
object of two planes, `DRM_FORMAT_MOD_LINEAR`), as the import check
imports its memory. The
strips share one video-processing context, and the import is kept by the
mapping's identity until the mapping goes (at most 96 at once, the one
used longest ago dropped first). Each frame is then one run a strip, from
the decode surface's rows into the strip, BT.709 on both sides as the
staged run's, a wait for the runs' writes ("The driver's lock", below),
and their syncs, before the session's `Demt`.

**What radeonsi takes**, measured with a probe on libva (not in this
repository) on both AMD GPUs (Mesa 26.2.4, 2026-10-24), each import
checked byte for byte after a run, padding included:
- rows of a multiple of 256 bytes: 1920-byte rows were refused
  (`VA_STATUS_ERROR_ALLOCATION_FAILED`), 1280, 3840 and 7680 taken;
- planes 256 bytes apart from the dma-buf's first byte: a luma plane 64
  bytes into it was refused, a chroma plane 256 bytes past a page boundary
  taken, and so was Apple's `PlaneAlignment` of 256;
- an even size: at 641x361 every sample past the first chroma pair came
  out other than the probe expected, at 8 and 10 bits, though nothing was
  written outside the planes ([bugs.md](bugs.md), D22, has why that is not
  known);
- nothing else: in every case taken, every byte of the buffer that is not
  one of the frame's samples, the rows' padding included, stayed as it
  was.

So, with Apple's `Dpba` layout, which the device keeps as the guest sees it
(`BytesPerRowAlignment` 64), the guest makes each row its width rounded up
to 64 bytes (a guest's 4K NV12 buffer, measured in a boot, was 12,450,304
bytes with 3840-byte rows), and the device writes rows of 1280 (720p),
3840 (4K 8-bit, 1080p 10-bit) and 7680
bytes (4K 10-bit), and declines 1920 (1080p 8-bit), 640 (360p) and 1408
(1366x768), and every odd size: those take the CPU's copy, each reason
logged once a session, as `zero copy: the CPU copies frame 0x… into the
guest's buffer, and the session's like it: rows of 1920 bytes: radeonsi
imports rows of a multiple of 256 bytes alone`. The stand-in guest's pool
pads each row by 256 bytes more (`ROW_PADDING`, so that its checks see the
stride), which leaves a row a multiple of 256 where the guest's is and not
where it is not: its 1080p rows of 2176 bytes, 360p of 896 and 1366-wide
of 1664 were declined, its 4K and 720p written. A driver is lent guest
buffers only where the import check passed on it, and only buffers of the
layout it passed at (`Display::writes_imports`): rows and plane starts a
multiple of its alignment, and, where it writes whole rows alone as iHD
does, the luma plane at a page's first byte and the chroma plane whole
rows of the luma's pitch after it ([bugs.md](bugs.md), U26). On NVIDIA's
VA-API driver the import and the run answered success and the frames
were wrong (above).

**What udmabuf takes.** The kernel's `udmabuf` puts at most `list_limit`
runs of pages and `size_limit_mb` MiB in one dma-buf: 1,024 and 64 at its
stock limits, and nothing here asks a host to change them. Nearly every
page of a guest's buffer is a run of its own in the memfd (measured in a
guest's boot: 3,040 runs for a 4K NV12 buffer, 6,078 for P010), so one
dma-buf holds about 4 MiB of it. The host refuses a dma-buf past either
limit before it asks the kernel, with its count of runs and bytes and the
limits as `/sys/module/udmabuf/parameters` has them then (core's
`DmaBufError::TooLarge`, reims-vgpu's `LendRefusal::TooLarge`), and the
backend imports the buffer in as many strips as that count calls for: one,
the whole buffer, where each plane fits in a dma-buf, else more, until
every strip's planes fit, up to 64 strips (`MOST_STRIPS`) and none of
fewer than two rows. At the stock limits a guest's 720p and 1080p 10-bit
buffers take one strip, its 4K NV12 ones two and its 4K P010 ones four,
and an 8192x4352 P010 one, the largest the decoders make on both AMD GPUs,
17 or 18 by its count of pages; the stand-in guest's, whose rows are 256
bytes longer, take two at 1080p 10-bit, three at 4K NV12 and five at 4K
P010. A buffer past 64 strips at the host's limits, which no size the
decoders make is at the stock ones, is declined and its frames copied by
the CPU, the log saying so once a session: `zero copy: the CPU copies
frame 0x… into the guest's buffer, and the session's like it: no dma-buf
of the buffer in 64 strips or fewer: N pieces of 0x… bytes do not fit
udmabuf's list_limit 1024 and size_limit_mb 64`. Higher limits give fewer
strips and change nothing else ("What it saves" has what the strips cost).
The NVIDIA backend registers guest buffers with CUDA and needs no udmabuf
([nvidia.md](nvidia.md), "Zero copy"), and NVIDIA's VA-API driver is lent
none.

**The driver's lock.** Mesa's VA frontend waits for a processing run in
`vaSyncSurface` with its display's lock held: it waits for the output
surface's `pipe_fence` under `drv->mutex`, where it waits for a decode's
fence under the context's lock alone (**read** in Mesa 26.2.4,
`src/gallium/frontends/va/surface.c`, `_vlVaSyncSurface`;
[bugs.md](bugs.md), U17), and `vaBeginPicture`, `vaRenderPicture` and
`vaEndPicture` each take that lock too (`picture.c`). So while the copy's
thread waited there for its run, the session's next decode waited behind
it: with decoding ahead the backend's `decode` took 0.9 to 5.0 ms a 4K
frame, against 0.3 to 1.0 ms one frame at a time. The backend now waits
for a run's writes with no lock held first, by `poll` for writing on the
dma-bufs the run writes (`src/va/fence.rs`): the kernel puts the run's
fence on each of them, so the poll returns once the run is done, and
`vaSyncSurface` then finds the fence signalled. Zero copy's runs and the
staged copy back's both wait so. Measured on both AMD GPUs (2026-10-25):
`vaSyncSurface` took 0.0 to 0.3 ms after the poll, and with decoding ahead
`decode` took 0.3 to 0.9 ms a 4K frame, zero copy on and off, and the
stand-in guest's asynchronous 4K sessions with the CPU copying ran 8% to
59% faster than before. A driver that puts no fence on the dma-buf leaves
the poll nothing to wait for, and the sync's questions wait in its place.
The poll, and the sync after it, wait for all of one run's dma-bufs until
the frame's deadline ("Waiting for the GPU").

**The proof**, on both AMD GPUs (Mesa 26.2.4, 2026-10-25, udmabuf at its
stock limits), with zero copy on: the codec checks all passed, bit-exact
(`h264-check.sh` 62, `hevc-check.sh` 108, `av1-check.sh` 278,
`vp9-check.sh` 198); `host-session-check.sh` ran on every stream they leave
with zero copy on and off, and the stand-in guest's record of the host's
messages (`--sequence`), sync and async, was the same by `cmp` in all 144
runs a GPU, every frame matching. The device wrote 6,632 of their 9,374
frames on each GPU and declined none for udmabuf's limits; it declined the
rest for their rows or size, as it should: 1080p 8-bit, 360p, 1366x766,
odd sizes (`a frame of 465x262: radeonsi's video processing writes an odd
size wrongly`), and the resized VP9 streams' 960x540 frames (`rows of 1216
bytes`), whose 1280x720 ones the device wrote. perf-check's ten runs a GPU
were the same on and off in each of three rounds, and so were 21 streams
of the stand-in's, H.264, HEVC, VP9 and AV1 at 720p, 1080p and 4K, 8 and
10 bits, sync and async, against the tree before the strips too, every
1080p 10-bit and 4K frame written in strips. In an x86_64 macOS 26.7.1 guest
on each AMD GPU at the stock limits (2026-10-25), the device wrote 562 of
the 862 frames of H.264, HEVC, VP9 and AV1 the guest decoded, its 1080p
10-bit and 4K frames in strips, and the CPU copied the 300 of 1080p 8-bit
(`rows of 1920 bytes`), every frame matching its reference.

**What it saves**, measured with a bench script not in this repository:
perf-check's streams at udmabuf's stock limits, zero copy on and off
taking turns, medians of three runs, and one run of the tree before the
strips and the lock's wait (*before*: the CPU copied every 4K frame, and
the next decode waited behind the copy's run), both AMD GPUs, 2026-10-25,
the host's load at 1.5 to 2.3: the host's own split of a frame, the
device's write against the CPU path's read back and copy, and the stand-in
guest's fps.

| stream, GPU, sync / async | device write | read back, copy | fps on | fps off | fps before |
|---|---|---|---|---|---|
| HEVC Main, Navi 31, sync | 7.26 ms | 6.76, 1.21 ms | 123.0 | 111.7 | 102.8 |
| HEVC Main 10, Navi 31, sync | 18.55 ms | 17.04, 2.18 ms | 49.3 | 47.8 | 47.1 |
| AV1, Navi 31, sync | 8.73 ms | 5.47, 1.21 ms | 93.4 | 116.6 | 116.3 |
| AV1 10-bit, Navi 31, sync | 17.31 ms | 15.05, 2.30 ms | 48.2 | 47.7 | 48.0 |
| HEVC Main, Phoenix, sync | 6.79 ms | 5.84, 2.40 ms | 127.2 | 108.0 | 120.7 |
| HEVC Main 10, Phoenix, sync | 8.52 ms | 7.08, 2.39 ms | 88.3 | 85.7 | 66.2 |
| AV1, Phoenix, sync | 5.69 ms | 4.45, 1.37 ms | 131.9 | 128.2 | 122.2 |
| AV1 10-bit, Phoenix, sync | 8.10 ms | 5.63, 2.41 ms | 78.9 | 84.5 | 81.6 |
| HEVC Main, Navi 31, async | 3.91 ms | 2.74, 1.26 ms | 168.5 | 183.3 | 115.6 |
| HEVC Main 10, Navi 31, async | 7.39 ms | 5.15, 2.23 ms | 91.2 | 95.2 | 71.5 |
| AV1, Navi 31, async | 4.29 ms | 2.96, 1.23 ms | 132.8 | 150.3 | 138.0 |
| AV1 10-bit, Navi 31, async | 8.38 ms | 5.54, 2.21 ms | 70.6 | 77.3 | 67.2 |
| HEVC Main, Phoenix, async | 2.75 ms | 1.77, 1.89 ms | 165.9 | 178.4 | 148.0 |
| HEVC Main 10, Phoenix, async | 4.21 ms | 2.73, 3.06 ms | 94.8 | 95.0 | 86.7 |
| AV1, Phoenix, async | 3.18 ms | 1.78, 1.80 ms | 149.7 | 158.9 | 142.9 |
| AV1 10-bit, Phoenix, async | 5.49 ms | 2.43, 3.01 ms | 83.0 | 89.9 | 82.9 |

- The CPU's copy is gone: 1.2 to 2.3 ms of each 4K frame on Navi 31, 1.4
  to 3.1 on Phoenix. Through the stand-in's 60-frame runs of 4K, each
  buffer's import made in them, the session's own threads used 0.4 to
  1.4 ms of the CPU a frame on, against 1.3 to 3.4 ms off.
- One frame at a time, the device's write took less than the read back
  and the copy in six of the eight rows, 0.06 ms more in Phoenix's AV1
  10-bit, and 2.0 ms more in Navi 31's 8-bit AV1, which ran 20% slower on
  than off ([bugs.md](bugs.md), D22).
- Decoding ahead, the CPU path ran 0% to 12% faster than the device's
  writes: its copy runs on the CPU beside the GPU's next decode, and the
  device's write took the GPU 1.0 to 3.1 ms longer than the staged run
  ([bugs.md](bugs.md), D22). Both run past 4K60 in every asynchronous
  row.
- The lock's wait ("The driver's lock") is the asynchronous rows' gain over
  *before*, off as on: 8% to 59% with the CPU copying, the backend's
  `decode` down from 0.9 to 5.0 ms a frame to 0.3 to 0.9 ms. One frame at a
  time it changes nothing: there the rows' spread between *before* and off
  follows the read back's time, which moved between 11.4 and 17.2 ms for
  Navi 31's HEVC Main 10 from one run of the same build to the next.
- The strips' own cost, from the same pages joined in runs of eight and
  cut into one to five strips on Navi 31: 10-bit 4K one frame at a time
  went from 50.9 to 49.4 fps (HEVC Main 10) and 50.8 to 47.1 (AV1). Navi
  31's 8-bit device write took 1.3 ms less for AV1 and 0.4 ms less for
  HEVC in one strip over runs of eight pages than in three over single
  pages; Phoenix's moved within the runs' spread. A guest's buffers take a
  strip fewer than the stand-in's (above).

**Checks to run** (after a codec's own checks, which leave the streams):
the host session with zero copy on, the default, and off, each frame
checked against its reference, and every `<name>.<mode>.sequence` of the
two runs alike by `cmp`; then perf-check both ways.

```sh
t=crates/reims-vgpu-video-vaapi/tools
$t/host-session-check.sh /dev/dri/renderD128                # and renderD129
RVGV_ZERO_COPY=off RVGV_HOST_SESSION_OUT=build/host-session-check-off \
    $t/host-session-check.sh /dev/dri/renderD128
for f in build/host-session-check/*.sequence; do
    cmp "$f" "build/host-session-check-off/${f##*/}"
done
$t/perf-check.sh /dev/dri/renderD128                        # and with RVGV_ZERO_COPY=off
```

## The reims-graphics device

The reims-graphics device's H.264 and HEVC decode runs on this backend
where its GPU is not one NVDEC opens: the device's video chooser
(reims-vgpu's `src/graphics/video_choice.rs`) tries NVDEC, then VA-API on
the render node of the runtime device's own GPU, whatever its vendor
(`host_gpus::render_node_of`), then Vulkan Video. The arm
(`src/graphics/vaapi.rs`) runs the crate's `H264Decoder` and
`HevcDecoder` in a host video session of reims-vgpu's
(`src/graphics/video_host.rs`), whose event the host signals.

- **What it answers** is the driver's: the VLD profiles it lists, the
  formats its configs name, and the sizes its configs and surfaces give.
  VA has no level limit or DPB bound of its own, so the levels are each
  codec's top and the DPB its structures' bound. The access unit is copied
  into VA's slice buffers, so it is read at any place and length.
- **A decode** hands the guest's Annex B access unit to the decoder, in
  decode order. It starts the hardware's work and answers the access
  unit's picture: one, or none for parameter sets and for a picture that
  random access skips.
- **A picture lands** in the output's planes one of two ways, the second
  where the first refuses:
  1. by the driver's video processing, in place, into the planes' pages
     imported as zero copy imports a guest buffer ("Zero copy"): where both
     planes lie in one memory a dma-buf is made of, and the import check's
     answer for the frame's format takes their layout;
  2. by the CPU: the frame read back ("The copy back") and copied into the
     planes' host bytes, or into the session's staging where a plane lies
     in a device cache.

  A session logs the first reason its pictures take the second way, as
  `a picture copied by the CPU: rows of 1920 bytes: the driver imports
  rows of a multiple of 256 bytes alone`, and its counts at its end, as
  `a session ended after 300 decodes: landed 300 in the surface's pages,
  by a CPU copy; 0 skipped, 0 failed`. A picture that does not land fails
  its decode's command buffer, as on NVDEC ([nvidia.md](nvidia.md), "The
  reims-graphics device").
- **Surfaces.** A picture holds its decode surface until it lands, and
  the session holds no more pictures unlanded than the decoder's spare
  surfaces (`SPARE_FRAMES`, 2), so a decode always finds a free surface.
- **The display** opens once per runtime device and logs its driver, the
  open's time and thread, and the import check's answers.

**Measured** in cold boots of an x86-64 macOS guest (2026-11-07):
decode_md5 decoded H.264 and HEVC at 8 and 10 bits, up to 4K, at odd
sizes and through a soak, each picture checked against its reference.

| GPU, driver | decodes right, wrong | H.264 1080p | H.264 2160p | HEVC 1080p | HEVC Main 10 1080p | HEVC 4K | HEVC Main 10 4K |
|---|---|---|---|---|---|---|---|
| Radeon 780M, radeonsi (Mesa 26.2.6) | 440, 0 | 1163.7 fps | 265.6 | 983.9 | 521.1 | 249.5 | 116.7 |
| Iris Xe, iHD 26.3.5 | 688, 0 | 331.6 | 92.7 | 383.7 | 328.4 | 106.8 | 85.5 |

An earlier boot on the same 780M, of the arm before the staged copy
back's planes moved to whole rows, decoded 843 right and none wrong, at
1172.9 fps for H.264 1080p and 985.5 for HEVC 1080p. No boot logged a
`reims-vgpu: fatal` line, and in each every picture was copied by the
CPU. The frames VideoToolbox hands the guest's
decoder have CoreVideo's own layout, each plane 32 bytes into its page
and rows of the width rounded up a little, 1376 bytes at 1366. Neither
driver's video processing writes that: radeonsi wants rows and planes
256 bytes apart, iHD the luma plane at a page's first byte
([bugs.md](bugs.md), B93).

## Waiting for the GPU

Every wait of the backend's on the GPU is bounded, by core's deadline,
5 s from a frame's first wait ([core.md](core.md), "Waiting for the
device"), asked after every 100 µs for its first 50 ms and every
millisecond after (`src/va/wait.rs`). A frame's waits share that one
deadline (`wait::Deadline`): a decoder's call and the read backs of the
frames it gives, an encoder's call, or a JPEG decode, with every wait
among them for the display's calls ("The display's calls"), so that a
frame is answered within 5 s of its first wait, however many calls and
questions it makes.

| wait | asked with | in place of |
|---|---|---|
| a decode | `vaSyncSurface2` of its surface with a timeout of 0, until it answers other than `VA_STATUS_ERROR_TIMEDOUT` | `vaSyncSurface` |
| a processing run: the staged copy back's, zero copy's, the JPEG decoder's conversions and the trial runs at start | for the copy back, zero copy and a JPEG conversion's last run, into host memory, the `poll` of the dma-bufs the run writes until the deadline ("The driver's lock"); then `vaSyncSurface2` of its target, as a decode's | the poll for a second at most, then `vaSyncSurface` |
| a JPEG decode | into host memory ("The JPEG decoder"), the `poll` of its dma-buf until the deadline, then `vaSyncSurface2` of its surface; into the driver's surface, `vaSyncSurface2` alone | `vaSyncSurface` |
| an encode | `vaSyncBuffer` of its coded buffer with a timeout of 0 | `vaSyncBuffer` with `VA_TIMEOUT_INFINITE` |
| a driver call that may wait for the GPU itself: `vaGetImage`, `vaMapBuffer` of a derived image, and `vaSyncSurface` where the driver has no `vaSyncSurface2` | its answer, the call made on the display's call thread ("The display's calls") | the call on the caller's thread |
| every other driver call, while one of those is in the driver | the display's gate, which the call thread holds exclusively meanwhile | the driver's own lock, which the call that waits holds as it does |

A question with a timeout of 0 returned at once while the work ran
(below), so no question waits in the driver, where radeonsi holds the
display's lock while `vaSyncSurface` waits for a processing run
([bugs.md](bugs.md), U17). Each takes that lock only as it asks, so the
JPEG decoder's runs into the driver's surfaces, those a conversion passes
through, are waited for by their questions alone, with no `poll` of a
dma-buf first: **measured** on the RX 7900 XTX and the
Radeon 780M (2026-10-26), a 4K H.264 decode's calls (`decode`, a stream of
60 samples of our encoder's decoded 4 or 12 times over, each frame waited
for) took 0.046 to 0.050 ms at the median and 0.09 to 0.12 ms at the 99th
percentile alone, and 0.055 to 0.064 ms and 0.14 to 0.31 ms beside a
thread decoding 4K JPEGs at 1/2 into NV12 and into BGRA, each a processing
run asked after every 100 µs.

### The display's calls

Some of radeonsi's calls wait for the GPU inside the driver, and hold the
display's lock (`drv->mutex`) as they do: `vlVaGetImage` and
`vlVaMapBuffer2` map the surface for the CPU under it, which copies it on
the GPU and waits for the copy (**read** in Mesa 26.2.4, `image.c` and
`buffer.c`; [bugs.md](bugs.md), U22). While one waits, every other
thread's call on the display waits behind it in the driver, its questions
with a timeout of 0 among them. **Measured** on both GPUs (Mesa 26.2.4,
libva 2.24.1, 2026-10-26), each call timed on a thread of its own on 4K
NV12 surfaces, with the graphics ring held, or 16 compute queues, by a
compute shader of the tests' own that spins until let go
(`src/va/spin.rs`):

| call | the graphics ring held | the compute rings held |
|---|---|---|
| `vaGetImage` | waits | waits |
| `vaMapBuffer` of a derived image of a JPEG decode's surface, or of a processing run's NV12 target | waits | (its run stalls first) |
| `vaSyncSurface` of a processing run's target | returns, the run done | waits |
| a decode's or a run's `vaBeginPicture`, `vaRenderPicture`, `vaEndPicture` | return | return |
| a new decoder's first `vaEndPicture`, a buffer of the process destroyed while the ring was held | waits | not timed |
| `vaSyncSurface2` and `vaSyncBuffer` with a timeout of 0 | return | return |
| `vaPutImage`, `vaDeriveImage`, `vaMapBuffer` of an image `vaCreateImage` made, `vaExportSurfaceHandle`, creates and destroys | return | return |

A processing run and an encode were held back and waited for by their
bounded questions; a decode was not, VCN running beside both rings. Each
call that waited held the display's lock: another thread's
`vaSyncSurface2` of an idle surface, timeout 0, returned only once the
call had, 380 ms on. radeonsi makes the copies of `vaGetImage` and of a
map on a compute queue, so where one waited with the graphics ring held,
it waited on the kernel's page-table updates: a buffer of the process
destroyed while the ring was held had its unmap's update wait for the
spin, and the copy's job wait for that update (**measured** with the GPU
scheduler's tracepoints, RX 7900 XTX, 2026-10-26; [bugs.md](bugs.md),
B92). With nothing destroyed, a JPEG decode's derived map ran beside the
held graphics ring. A decoder's first `vaEndPicture` submits the job that
begins the decoder's session and waits for it (**read** in Mesa 26.2.4,
`si_video_dec.c`; [bugs.md](bugs.md), U24), so it waits as long as that
job does, behind a destroy's update as the copies are (B92).

So the backend makes three kinds of call (`src/va/calls.rs`):
- **A call that may wait for the GPU** (the table's above) is made on the
  display's call thread, `reims-vgpu-video.va-calls`, with the display's
  gate held exclusively, and its caller waits for its answer until the
  frame's deadline (`Display::wait_call`). Past it the caller goes on with
  `Error::Stalled`, and the call is left to return on the thread; what it
  gives once it does goes to the caller's late hook there (a map's
  unmap). An order not begun when its caller left is never made.
- **Every other call** is made on the caller's thread with the gate held
  shared (`Display::call`), once the call thread has nothing in hand or
  queued, so that no call overtakes one ordered before it, and once the
  gate is had, both by the frame's deadline, which a call made at once
  does not start. So while a call that waits is in the driver, no other
  call of ours is, and none waits for the driver's lock behind it: each
  waits for the gate, until its own deadline, and then fails as stalled.
- **A release** (a destroy, an unmap) is made at once where the call
  thread is idle and the gate free, and otherwise ordered on the call
  thread behind what it has (`Display::release`), never waited for and
  never left out: nothing a call left there uses is destroyed before it
  returns. The display's `vaTerminate` is its last.

The thread starts with the first order and ends after `vaTerminate`.
Where it cannot start, the log says so, and orders are made on their
callers' threads with no bound.

**What the gate costs** is the driver's lock's: a call that waits holds
the gate as long as the driver held its lock for it. On radeonsi the
JPEG decoder no longer makes such a call: it reads its outputs from host
memory of ours ("The JPEG decoder", "Surfaces and readback"), where it
read each through a derived map. **Measured** on the RX 7900 XTX and the
Radeon 780M (2026-10-26), a thread decoding a 4K 4:2:0 JPEG over and over
beside a 4K H.264 decode on one display (`decode` of our encoder's 60
samples of noise, 1.6 MB each, 12 times over, each frame waited for),
the tree before and after, the call thread's holds of the gate and the
H.264 thread's waits for it counted by instrumenting `src/va/calls.rs`
(not in the tree), three runs or more of each, before and after, on each
GPU:

| JPEG output, GPU | before: maps a run (holds of the gate), their mean and p99 | before: the H.264 thread's waits for the gate a run, and how many of its 720 `decode` calls waited | after: holds, waits |
|---|---|---|---|
| `420f` at full size, RX 7900 XTX | 372 to 379; 6.0 to 6.6 ms, 26 to 34 ms | 371 to 377; 2 to 17 | 0, 0 |
| the same, Radeon 780M | 371 to 372; 3.9 to 4.3 ms, 27 to 29 ms | 371 to 372; 0 | 0, 0 |
| `420f` at 1/2, RX 7900 XTX | 388 to 403; 7.4 to 8.7 ms, 25 to 31 ms | 378 to 393; 21 to 52 | 0, 0 |
| the same, Radeon 780M | 373 to 379; 7.4 to 8.2 ms, 27 to 29 ms | 364 to 371; 1 to 6 | 0, 0 |
| `BGRA`, RX 7900 XTX | 372 to 377; 5.0 to 5.1 ms, 5.2 to 7.9 ms | 372 to 377; 1 to 15 | 0, 0 |
| the same, Radeon 780M | 336 to 399; 1.1 to 1.2 ms, 1.3 to 1.7 ms | 330 to 379; 7 to 10 | 0, 0 |

The H.264 thread's waits not in a `decode` call were in its frames' waits,
at their questions: 323 to 371 of a run's 720 frame waits waited for the
gate, in the runs that counted them. After, none did. The JPEG thread
decoded more beside the H.264 decode than before, its copies out taking
13.3 to 16.8 ms each against 25.4 to 26.8, at 1/2 14.7 to 17.3 against
22.1 to 26.3, and into BGRA 21.6 to 27.4 against 24.0 to 29.5. The H.264
frames' times, decode call and wait together, moved within their spread,
being mostly the decode's own: 12.6 to 12.8 ms at the mean, 40 to 43 at
the 99th percentile and 43 to 44 at most alone; beside the JPEGs, before
and after alike, 11.7 to 15.1, 39 to 46 and 41 to 60 ms, but for one run
after on the Radeon 780M beside BGRA, under other processes' load on the
host's CPU, at 16.1, 64 and 77 ms, which three runs again came to 13.1 to
13.2, 42 to 44 and 45 to 46 ms. The `decode` calls' own 99th percentile
was 6.6 to 14.3 ms before and 6.6 to 14.1 after, and their longest 8.6 to
26.9 and 8.3 to 17.2 ms, against 6.6 to 13.9 and 8.3 to 17.2 alone, but
for that run, at 31 and 42 ms.

Past udmabuf's 64 MiB the engine reads its output in strips of host
memory ("The JPEG decoder", "Surfaces and readback"). **Measured** the
same way beside 8192x8192 4:2:0 JPEGs copied out at full size, the tree
that read them through derived maps and the tree that reads them in
strips taking turns, two runs of each on each GPU, then three or four
more of each with every display call timed (B92's), and the H.264 decode
alone once in each tree:

| GPU | maps: holds a run, mean, longest; the H.264 thread's waits | strips: holds, waits | `decode` calls' p99 and longest: maps, strips, alone |
|---|---|---|---|
| RX 7900 XTX | 85 to 91; 19.7 to 21.6 ms, 44 to 45 ms; 84 to 90 | 0, 0 | 22 to 25 and 46 to 56 ms; 38 to 49 (9.5 in one run) and 49 to 55; 10.6 to 10.7 and 11.7 to 11.8 |
| Radeon 780M | 77 to 78; 5.5 to 6.7 ms, 34 to 59 ms; 76 to 78 | 0, 0 | 9.0 to 12.5 and 47 to 155 ms; 8.6 to 10.4 and 133 to 138; 12.2 to 12.5 and 13.1 to 14.0 |

The bench made its H.264 decoder again for each of its 12 passes, and
the H.264 calls' longest are each a new decoder's first `vaEndPicture`,
which waits, with the driver's lock held, for the job that begins the
decoder's session, held behind the last decoder's unmaps and those behind
the JPEG decode on the GPU ([bugs.md](bugs.md), B92): 3 to 6 a run with
the maps and 8 to 10 in strips on the RX 7900 XTX, and 4 to 6 and 5 on
the Radeon 780M. A run's 99th percentile is its 8th longest call of 720,
so those set it in strips on the RX 7900 XTX; the strips made no running
session's call wait. With the decoder kept across the passes, as a
session keeps its decoder, no `decode` call took over 30 ms (**measured**,
2026-10-26, in strips, five runs on the RX 7900 XTX and two on the Radeon
780M): their 99th percentile was 6.7 to 8.8 ms and 8.9 ms, under the
decode alone's, and their longest 9.1 to 11.9 and 10.5 to 11.4 ms. In
the first two runs of each, the
JPEG thread decoded 92 to 93 images a run with the maps and 86 in strips
on the RX 7900 XTX, 126.6 ms at the mean against 118.7 to 119.0, and 79
to 80 against 82 on the Radeon 780M, 134 to 140 ms against 135 to 136.
A 4:2:2 copy past 64 MiB is read in strips too, through an alias of its
YUY2 surface as BGRA ("Surfaces and readback"). **Measured** the same way,
the H.264 decoder kept across its 12 passes, beside 4:2:2 JPEGs copied
out into `422f`, the tree that mapped them and the tree that reads them in
strips taking turns, three runs of each beside 8192x8192 and two beside
12288x8192 on each GPU, and the H.264 decode alone once in each tree
(2026-10-26):

| GPU, JPEG | maps: holds a run, mean, longest; the H.264 thread's waits | strips: holds, waits | `decode` calls' p99 and longest: maps, strips | the JPEG decodes' mean: maps, strips |
|---|---|---|---|---|
| RX 7900 XTX, 8192x8192 | 43; 20.5 to 20.6 ms, 20.7 to 20.8 ms; 43 | 0, 0 | 8.7 to 9.6 and 27.9 to 29.0 ms; 6.6 to 7.1 and 8.8 to 9.8 | 227.6 to 228.8 ms; 234.7 to 236.8 |
| RX 7900 XTX, 12288x8192 | 30; 30.2 ms, 30.3 to 30.4 ms; 30 | 0, 0 | 6.7 to 9.5 and 38.6 to 39.7 ms; 6.6 to 6.7 and 8.9 to 10.3 | 336.3 to 337.5 ms; 349.6 to 350.1 |
| Radeon 780M, 8192x8192 | 40 to 41; 5.1 to 5.2 ms, 5.8 to 7.3 ms; 40 to 41 | 0, 0 | 7.4 to 7.6 and 9.8 to 12.2 ms; 7.2 to 7.4 and 9.2 to 10.4 | 243.0 to 247.7 ms; 244.8 to 250.5 |
| Radeon 780M, 12288x8192 | 28 to 30; 7.3 to 7.4 ms, 8.9 to 12.6 ms; 28 to 30 | 0, 0 | 9.6 to 12.7 and 14.0 to 15.5 ms; 11.4 to 12.1 and 16.1 to 16.6 | 364.2 to 374.4 ms; 374.5 to 384.0 |

Alone, the H.264 `decode` calls' 99th percentile was 8.6 to 9.5 ms on
the RX 7900 XTX and 8.7 to 9.6 on the Radeon 780M, and their longest 9.8
to 10.3 and 8.9 to 10.4 ms. With the maps, 1 to 8 of a run's `decode`
calls waited for the gate and the rest of its waits were in frame waits;
each map's hold set the RX 7900 XTX's longest calls. On the Radeon 780M
beside 12288x8192, the `decode` calls' mean rose from 2.6 to 3.9 or 4.0
ms in both trees from the second pair of runs on, which sets their 99th
percentile there. The H.264 frames' longest waits for the GPU beside the
JPEGs, 90 to 92 ms on the Radeon 780M in both trees against 36 alone,
are not the gate's.

**Past the deadline** the frame fails as the backend's malfunction
(`Error::Stalled`): -12911 with no image for a decoded frame, -12912 for an
encoded one, and kIOReturnDeviceError for a JPEG decode. A frame whose zero copy runs
passed it fails so too, and is not declined to the CPU's copy, which would
write the guest's buffer beside the runs. What failed then fails each later
call at once, the same way: a decoder's pool and its host session, an
encoder, and the JPEG decoder until what it kept is let go. What the GPU may still write is kept in core's
list until the GPU is done, asked after with the same questions:
- a decoder's: its pool, every surface of it and the staged copy back's
  surface and memory, and its context once the decoder lets it go,
  destroyed first (`src/va/surface.rs`); for zero copy, the guest buffer's
  import and the session's hold on the buffer, so that the guest sees the
  buffer in use until each strip is done (`src/va/zero_copy.rs`);
- an encoder's: its session, with the coded buffer, the context and both
  pools of surfaces, until the coded buffer is done
  (`src/va/encoder/accelerator.rs`);
- a JPEG decode's: the surface the GPU may still write, with the decode's
  context or the conversions' (`src/va/jpeg.rs`). A surface a run only
  reads is let go.

An answer that is an error keeps the work, and a GPU that never finishes
keeps it until the process ends.

**What the waits answer**, measured on the RX 7900 XTX and the Radeon 780M
(radeonsi, Mesa 26.2.4, libva 2.24.1, 2026-10-25), the questions asked
again and again before each of ffmpeg n9.0.2's own syncs by a library of
ours preloaded into it, as ffmpeg decoded a 4K HEVC stream of 150 frames
(perf-check's `hevc-2160p-main`), scaled 60 of its frames to 1080p, and
encoded 60 as H.264 and as HEVC at 20 Mbit/s:
- `vaSyncSurface2` and `vaSyncBuffer` with a timeout of 0 answered
  `VA_STATUS_ERROR_TIMEDOUT` while the work ran, then success, and
  ffmpeg's own sync after them returned in 1 µs at most. ffmpeg's frames
  came out the same as without the questions.
- From ffmpeg's sync on, the first decode was done in 3.3 ms on the 7900
  XTX and 1.6 ms on the 780M, and the 147 after it at the first question,
  ffmpeg syncing each frame late; a scaling run in 0.32 and 0.69 ms
  (median), 0.48 and 2.0 ms at most; an encode's coded buffer in 5.1 to
  14.3 ms (median), 26 ms at most.

**What it costs**, measured the same day on perf-check's
`hevc-2160p-main` (4K HEVC Main, 150 frames, zero copy on), the tree
before and the tree with the bounded waits taking turns, nine runs each
on each GPU, every frame matching its reference in all 72 runs (mean fps,
before and after):

| run | RX 7900 XTX | Radeon 780M |
|---|---|---|
| direct, the codec's own decode and read back | 37.8, 37.7 | 39.5, 39.8 |
| the stand-in guest through core's host session, synchronous | 122.1, 120.6 | 126.5, 125.4 |
| the same, asynchronous | 168.0, 167.0 | 167.9, 166.4 |

The session runs' means are 0.6% to 1.2% lower, inside their spread from
run to run, 4 to 8 fps. A decode's wait now ends up to one pause, 100 µs,
after the decode does, about that share of a synchronous 4K frame. The
host session's records were the same by `cmp`, before and after, in all
of `host-session-check.sh`'s 144 runs a GPU, zero copy on and off.

**Its tests.** On the VA-API host, ignored elsewhere, on the render node
`RVG_VAAPI_NODE` names, one test at a time. In `src/va/wait/gpu.rs`, each
asked after with no time left, or 4 ms for an encode, so that the work is
still running at its deadline, on the GPU's own work: a 4K H.264 decode
fails as stalled, the frame's later wait and the decoder's next decode
fail at once, and its pool, then its context, are kept until the GPU is
done and then let go; a decoded frame's staged copy back fails the same
way, with no other way back tried; a 4K H.264 encode fails its picture
and the next frame at once, its session kept until the coded buffer is
done; and a JPEG decode into host memory fails, its surface and context
kept until the GPU is done, after which the engine decodes again. In
`src/va/wait/held.rs`, with a ring held by the spin, each frame's wait
50 ms: the JPEG decoder's conversion into host memory (`420f` at 1/2, the
compute rings held); its copy out through a derived map, and its
conversion into BGRA on a driver without `vaSyncSurface2`
(`vaSyncSurface2` answered as one answers), both as where it reads the
driver's images, the compute rings held, on one of which radeonsi copies
a surface for its map; the `vaGetImage` way back, the concealment copy,
and an encode whose upload maps its input surface each fail by the
frame's deadline while the GPU is still held, 50 to 55 ms, and the next
at once; a new decoder's first decode, another decoder let go with the
graphics ring held, fails so 57 to 58 ms on, its next decode at once and
the display's other calls each by its own deadline, and a decoder made
once the spin is let go decodes; beside the conversion into host
memory, while it waits and after, the display's other calls are made at
once; beside a call in hand on the call thread, another call on the
display fails by its own deadline, from the thread of the stalled frame
and from another, and two calls of one frame by one deadline, the second
at once; once the spin is let go, what was kept is let go, and the JPEG
decoder and the display's calls work again. With the graphics ring held,
the JPEG decoder's copies out into host memory decode and hold no other
call, and so do its conversions, then its copies out, on one engine,
each made once with the GPU free first, its surfaces kept ("Surfaces and
readback"). All passed on both AMD GPUs (2026-10-26). Core's stand-in
tests the guest's side, zero copy's held buffer among it ([core.md](core.md),
"Waiting for the device").

**Not bounded** ([bugs.md](bugs.md), B78): the calls measured not to wait
on radeonsi (the table's "return" rows) are made on the caller's thread,
with no bound of their own, so on a driver whose call among them waits for
the GPU, the call holds its caller's thread; and the `mapped` way back, for
comparison only, waits for the GPU in `DMA_BUF_IOCTL_SYNC`.

## The H.264 decoder

### Shape

`h264::Decoder<A: Accelerator>` does everything above the slice data:
- splitting samples into NAL units, and picture boundaries (7.4.1.2.4);
- SPS activation, which reconfigures the accelerator when size, format,
  profile or DPB size change (a new session at that IDR picture);
- order counts of types 0, 1 and 2 (8.2.1), the frame number gap process
  (8.2.5.2), marking by sliding window and by every MMCO (8.2.5);
- list initialisation and modification (8.2.4), for each slice;
- output in decode order, or in display order by the bumping process
  (C.4.5.3) with the VUI's `max_num_reorder_frames`.

Every slice's lists are built and checked before anything reaches the
accelerator, so a sample with an error submits nothing.

The VA accelerator (`src/va/h264.rs`) turns that into libva's buffers:
- one picture parameter buffer and one IQ matrix buffer;
- a slice parameter buffer and a slice data buffer per slice.

The conventions it follows:
- **Slice data** is the NAL unit from its header byte, still escaped.
  `slice_data_bit_offset` counts RBSP bits from the header byte, as va.h
  says ("relative to and includes the NAL unit byte ... after the removal
  of any emulation prevention bytes").
- **Scaling lists** go in raster order (va.h: "in raster scan order").
  The 8x8 lists are Intra Y and Inter Y.
- **Weights** that the stream infers (7.4.3.2) are filled in, as libva's
  users do: `1 << denominator`, offset 0.
- `CurrPic` is flagged short-term when it is a reference.
- `MinLumaBiPredSize8x8` is set from level 3.1 up (Table A-1).

**What radeonsi reads of the slice buffer is narrow.** In a run with
`RefPicList0`'s first two entries swapped in every slice that had two, and
another with the luma weights replaced by garbage, every frame of main-720p
and high-1080p still matched (Navi 31, 2026-10-24, with a patch of the
accelerator that is not kept). So VCN takes those from the slice header
itself. The decoder's lists and weights are checked by the unit tests, not
by radeonsi; a driver that reads them (Intel's, nvidia-vaapi-driver,
v4l2-request) is their first hardware test.

### Surfaces and readback

- Each session makes a pool of NV12 surfaces at the coded size, sized
  DPB + 1 + 2 spare. They are the context's render targets.
- A `Frame` holds its surface until it is dropped; a caller that holds
  frames makes the pool grow, up to 48 surfaces, then `OutOfSurfaces`.
  Surfaces made after the context are decoded into fine: main-720p
  matched with the caller holding 24 frames.
- Readback waits for the decode ("Waiting for the GPU"), has the GPU copy the
  frame into host memory of the pool's, and copies the cropped rectangle
  row by row into the caller's planes at the caller's strides; where that
  cannot be set up, through `vaGetImage`, then `vaDeriveImage` ("The copy
  back"). The speeds below were measured with `vaGetImage` as the way
  back.
- Speed at 1080p, decode, readback and the test tool's MD5 together, in
  the check's display-order runs: 128 to 135 frames a second on the Navi
  31, 110 to 131 on Phoenix. Decode order, where each frame is read back
  as soon as it is submitted, so that the readback waits for the whole
  decode, runs at about 110. Readback is 1.7 to 2.8 ms a 1080p frame.
  These are single runs over SSH, not benchmarks.

### What it refuses

Before anything reaches the GPU:
- with `h264::Error::Interlaced` ("interlaced: the SPS's
  frame_mbs_only_flag is 0"), any SPS with `frame_mbs_only_flag` 0,
  whether its pictures are fields, MBAFF frames or frames alone, as
  Apple's AVD refuses it ([devices.md](devices.md), "H.264"), x264's
  `--fake-interlaced` streams among them;
- with `Error::Bitstream(h264::Error::Unsupported(..))`, chroma other than
  4:2:0, bit depths other than 8, lossless macroblocks, and the Extended
  profile;
- the same, for FMO, data partitioning, and SP and SI slices.

Redundant pictures are dropped.

`H264Decoder::new` also checks the `avcC` record's first SPS, and the
driver's profiles and sizes, so that a session the driver cannot serve
fails when it is made.

The backend refuses a session whose SPS codes a size under the driver's
smallest surface, 64x64 on radeonsi, with `Malfunction`: the
guest gets -12911 at `dsta`, Apple's AVD's status for a size it does not
decode, and decodes the stream itself. Mesa's `vaCreateContext` would
refuse the size with `VA_STATUS_ERROR_RESOLUTION_NOT_SUPPORTED` anyway
(`src/va/backend.rs`, `refusal`; "The smallest sizes", below).

A P or B slice is refused with `MissingReference` when it has nothing to
predict from: after starting or `reset()` at a non-I picture, or when its
list names a frame the gap process inferred where the SPS allows gaps.
Where it allows none, the gap is frames lost. Each lost frame is then a
copy of the last reference, made by `vaGetImage` and `vaPutImage` through
an image the pool keeps for it (`Accelerator::conceal`,
`Surface::copy_to`), so the frames after it decode, as Apple's AVD decodes
them.

After a sample that does not decode, each picture is still decoded, so
that its references go on. Each is refused with `h264::Error::Unrecovered`,
until an I picture, IDR or not, decodes. Apple's AVD fails those frames so
([devices.md](devices.md), "H.264").

A slice cut short does not start that state: the parser stops at its
header, and radeonsi decodes what there is and reports nothing. After such
a slice, H.264's or HEVC's, `vaSyncSurface2` and `vaSyncSurface` answered
`VA_STATUS_SUCCESS`, `vaQuerySurfaceStatus` `VASurfaceReady`, and
`vaQuerySurfaceError` `VA_STATUS_ERROR_UNIMPLEMENTED`, on an RX 7900 XTX
and a Radeon 780M (Mesa 26.2.4, libva 2.24.1), 2026-10-25 ([bugs.md](bugs.md),
B41).

### Crafted streams

`examples/h264-craft.rs` (ours) rewrites an x264 Annex B stream into what no
encoder here writes, and writes it as Annex B and as an MP4 whose `avcC`
has the length size asked for:
- a smaller `max_num_ref_frames`, and gaps allowed;
- each slice's marking: the IDR long-term, or a list of MMCOs, with
  `frame_num` and the order count renumbered after operation 5;
- a gap in `frame_num` before a sample;
- damage: a sample left out, its slices cut short, or a garbage NAL unit
  after them (reserved type, a slice of noise, or one with the forbidden
  bit).

`--check` decodes the MP4 through this crate's decoder over a stand-in
accelerator. For each sample it prints what would reach the GPU, the
copies made for lost frames, and any error. A damaged stream goes to a
GPU only once this shows whole pictures of valid slices alone:
`host-session-check.sh` refuses to run a damage stream otherwise.
`h264-check.sh` builds the crafted streams from x264's one-reference
streams, so that each P slice keeps its one reference after the
markings. It checks each against ffmpeg, or against its source for a gap,
since ffmpeg leaves frames out after one ([bugs.md](bugs.md), U8). Run
2026-10-24 on both AMD GPUs:
- `h264-check.sh` passed 62 of 62. That covers the long-term and MMCO
  streams, CABAC and CAVLC, three gap streams, 2- and 3-byte lengths, and
  drop20-ref1 (frame 20 lost: 59 frames, the 50 the loss does not reach
  as the source's). The MBAFF and fake-interlaced streams were refused.
- `host-session-check.sh` passed its 35 H.264 checks. These are the
  streams above through the host session, synchronous and asynchronous;
  the refusals at `dsta` (-12911 interlaced, -12910 3-byte lengths); and
  drop20 and garbage after frame 20, which answer as Apple's host answered
  the same streams.

## The HEVC decoder

### Shape

`hevc::Decoder<A: Accelerator>` does everything above the slice data, as
H.264's does:
- splitting samples into NAL units and access units (7.4.2.4.4), with
  the `hvcC` record's parameter sets first and in-band ones taking their
  place, as `hev1` streams carry them;
- VPS, SPS and PPS up to their extension flags (7.3.2), with the VUI and
  its HRD read through, scaling lists with prediction and the defaults of
  Tables 7-5 and 7-6, and tiles (6.5.1);
- slice segment headers, dependent segments taking their independent
  segment's fields, short-term sets explicit and predicted (7.3.7), from
  the SPS or the header, long-term pictures with their MSB cycles, list
  modification, weight tables and entry points;
- order counts (8.3.1), the reference picture set's derivation and
  marking (8.3.2), with `MissingReference` when a set's `Curr` lists
  name a picture the DPB lacks;
- the lists (8.3.4), the output and removal of pictures (C.5.2.2) and
  bumping (C.5.2.3) with `sps_max_num_reorder_pics` and the latency
  count, in display order; or decode order;
- random access: `NoRaslOutputFlag` at the first picture, after an end of
  sequence and at every BLA or IDR, RASL pictures skipped after a CRA or
  BLA that has it, `NoOutputOfPriorPicsFlag` (inferred 1 for a CRA so
  flagged, as C.5.2.2 says), `pic_output_flag`.

Every picture's lists are built and checked before anything reaches the
accelerator.

The VA accelerator (`src/va/hevc.rs`) fills, per picture, one
`VAPictureParameterBufferHEVC`, a `VAIQMatrixBufferHEVC` when scaling
lists are on, and per slice segment a `VASliceParameterBufferHEVC` (the
long one) with its data. The conventions, each from va_dec_hevc.h unless
said:
- **Slice data** is the NAL unit from its two header bytes, escaped.
  `slice_data_byte_offset` is the header's length in bytes after the
  removal of emulation prevention bytes, and
  `slice_data_num_emu_prevn_bytes` the bytes removed.
- **ReferenceFrames** hold the set's pictures, at most 15, flagged
  `ST_CURR_BEFORE`, `ST_CURR_AFTER`, `LT_CURR` or none (the `Foll` ones),
  and `LONG_TERM`; unused slots are `VA_PICTURE_HEVC_INVALID`.
  `RefPicList` entries index them; 0xFF is no entry.
  `collocated_ref_idx` is 0xFF when TMVP is off.
- **Weights** go as coded deltas, with `ChromaOffsetLx` derived (7-56).
- **Scaling lists go in raster order**, each 8x8 list standing for its
  16x16 or 32x32 one, and the two 32x32 lists are `matrixId` 0 and 3.
  va_dec_hevc.h says both "raster scan order" and "corresponds to ...
  ScalingList[ i ][ MatrixID ][ j ]", which is in coding order; Mesa
  takes raster. Sent in coding order, a stream with the default lists
  decoded no picture right on radeonsi; in raster order it matched, and so
  did one of asymmetric lists, so raster is not mistaken for its
  transpose (2026-10-24, both AMD GPUs).
- `IntraPicFlag` is set when every slice is I, `NoPicReorderingFlag`
  when `sps_max_num_reorder_pics` is 0, `NoBiPredFlag` when no slice is
  B, and `RapPicFlag` for NAL types 16 to 21.

### Surfaces, formats and readback

- Main and Main Still Picture decode into NV12 surfaces
  (`VAProfileHEVCMain`), Main 10 into P010 (`VAProfileHEVCMain10`,
  `VA_RT_FORMAT_YUV420_10`). P010 is CoreVideo's `x420` and `xf20` byte
  for byte: two planes of little-endian 16-bit samples, the 10 bits in
  each sample's top bits.
- Pools, holds and readback are H.264's (above), in either format:
  `Frame::bit_depth` says which, and `copy_p010_to` copies two bytes a
  sample. The conformance window crops in chroma units, so a 4:2:0 frame
  is always of even size.
- `VaapiBackend` lists profiles 1, 2 and 3 where the driver has the VA
  profiles, and opens an `hvc1` session on `HevcDecoder`. Its output
  format follows the `hvcC` record's first SPS: `420v` or `420f` for 8
  bits, `x420` or `xf20` for 10, by `video_full_range_flag`, which is
  what Apple's host asks for ([devices.md](devices.md), "HEVC"). A new `hvcC`
  reopens the decoder.
- Speed, decode, readback and the tool's MD5 together in display order
  (2026-10-24, three runs on each AMD GPU over SSH, on a shared host): 720p
  at 165 to 230 fps, 1080p with four slices at 99 to 118, 4K at 24 to 29.
  Main 10 runs at 76 to 126 fps at 720p and 12 to 18 at 4K: its frames are
  twice the bytes to copy and to hash.

### What it refuses

With `Error::HevcBitstream(hevc::Error::Unsupported(..))`, before
anything reaches the GPU:
- profiles other than Main, Main 10 and Main Still Picture, unless the
  compatibility flags name one of them;
- chroma other than 4:2:0, bit depths above 10, and luma and chroma of
  different depths;
- the range and screen content extensions (`sps_range_extension_flag`,
  `sps_scc_extension_flag` and the PPS's).

Parameter sets, slice headers and pictures past the standard's ranges
are refused with `OutOfRange`, or the variant that names what they break
(`TooManyReferences`, `OverlappingSliceSegments` and the like), as bad
data ([hardening.md](hardening.md), F19 to F21 and F28 to F34).

A picture whose current reference sets hold more than 8 pictures together
is refused by the VA accelerator with `Error::Unsupported`, before any of
it reaches the driver: va.h says `NumPocTotalCurr` "should be equal to or
smaller than 8", and radeonsi decoded such a picture wrongly with no error,
as ffmpeg's own VA-API decode did (kvz-720p-ref15 from its tenth picture
on, on the Radeon 780M, 2026-10-24; "Testing HEVC"). The guest gets -12910
for it, and -12909 for each picture after it until an IRAP picture.

x265's streams of Main 4:2:2 10, Main 4:4:4 and Main 12 are refused
so. `HevcDecoder::new` checks the `hvcC` record's first SPS, the driver's
profile and its largest size, so that a session the driver cannot serve
fails when it is made. The backend refuses one whose SPS codes a size
under the driver's smallest surface, 64x64 on radeonsi, with -12911 at
`dsta`, as for H.264 above.

## The AV1 decoder

### Shape

The codec crate's `src/av1` parses a temporal unit into its OBUs, keeps the sequence header
in force and the eight reference slots' state (7.20, 7.21), parses each
frame header with everything it inherits (segmentation, loop filter
deltas, global motion and film grain from `primary_ref_frame` or
`load_grain_params`), and hands an `Accelerator` one frame at a time with
all its tile groups. It outputs the frames shown, `show_existing_frame`
ones included, one a temporal unit. Nothing in it names a backend, so it
is in `crates/reims-vgpu-video-codec` with the other codecs.

`src/va/av1.rs` is VA-API's accelerator. It follows libva's AV1 interface
as Mesa's frontend reads it (`src/gallium/frontends/va/picture_av1.c`,
read from Mesa's source; every convention below is confirmed only by the
bit-exact runs in "AV1 on the AMD GPUs"):
- `frame_width_minus1` is the upscaled width, the driver works out the
  coded width from `superres_scale_denominator`;
- inherited values go in as in force, not as coded;
- each tile group is one slice parameter buffer with an element per
  tile, offsets into the slice data buffer after it, all in one
  `vaRenderPicture` with the picture parameters;
- `ref_frame_map` names each slot's surface; a slot with nothing is
  `VA_INVALID_SURFACE`;
- CDEF strengths go in as `(primary << 2) | secondary`, as coded;
- with film grain, the render target (`current_frame`) is the frame as
  decoded, which later frames predict from, and `current_display_picture`
  is a surface of a second pool that the driver writes with the grain
  applied. The grained surface is what is shown.

The binding's layouts are checked against the installed headers by the
ignored test `av1_layouts_match_libva_headers`: it passed on 2026-10-24
against libva 2.24.1's `/usr/include/va` (VA-API 1.24), which marks
`tg_start`, `tg_end` and `chroma_sample_position` deprecated.

### Surfaces and output

- 8-bit streams decode to NV12, 10-bit to P010; `Frame::copy_p010_to`
  reads P010 back.
- One pool of 8 + 1 + 2 surfaces at the sequence's largest frame size,
  rounded to even, and a second as large when the sequence has
  `film_grain_params_present`.
- When the sequence allows frames larger than the driver's maximum, the
  pool is sized by the sequence's first frame instead, and a later frame
  larger than it is refused. SVT-AV1 v4.2.0 with `--pred-struct 1
  --sframe-dist 15` allows 16384x8704 in the sequence header of a 720p
  stream (svt-720p-bigmax below).
- A frame comes out at its upscaled width and frame height, as libdav1d's
  and libaom's do; the render size is not applied. A temporal unit of
  several spatial layers puts out its highest layer's frame alone, as
  Apple's host and ffmpeg's libdav1d do ([devices.md](devices.md), "AV1");
  the lower layers' frames are decoded as references. A monochrome frame's
  chroma is mid-grey: every sample 128 (512 at 10 bits) on both AMD GPUs.

### What it refuses

With `Unsupported` before anything reaches the GPU:
- profiles 1 and 2 (4:4:4, 4:2:2, 12 bits). Core's pixel formats are
  NV12 and P010 alone, and Phoenix has no Profile 2 ("The AMD drivers");
- tile list OBUs, which only large scale tile decoding has.

`Av1Decoder::new` checks the `av1C` record's sequence header, or failing
one its fields, and the driver's profile and formats, so that a session the
driver cannot serve fails when it is made.

The backend refuses, with -12911 at `dsta` ("The H.264 decoder", "What
it refuses"), a sequence header whose largest frame is under the
driver's smallest surface, 16x16 on radeonsi, and a 10-bit stream 32 or
fewer samples wide, whose inter frames radeonsi decodes wrongly
([bugs.md](bugs.md), U15). Sizes from 16 to 63, aom-63x65 among them ("AV1
on the AMD GPUs"), decode, where Apple's decoder refuses them
([devices.md](devices.md), "Where we differ from Apple").

## The VP9 decoder

### Shape

The codec crate's `src/vp9` splits a sample into its frames by the
superframe index (Annex B), parses each frame's uncompressed header (6.2)
with what it inherits (the loop filter deltas, the segmentation features,
the colour config of intra-only frames), keeps the eight reference slots'
sizes and formats, and hands an `Accelerator` one frame at a time with
its segments' quantizers and loop filter levels worked out (8.6.1, 8.8.1).
Hidden frames are decoded and not output; a `show_existing_frame` outputs
its slot's picture again with nothing decoded. It parses into a copy of
the persistent state and keeps it only when the frame parses. Nothing in
it names a backend.

`src/va/vp9.rs` is VA-API's accelerator:
- each frame goes in whole as the slice data, one slice parameter buffer
  at offset 0, in one `vaRenderPicture` with the picture parameters;
- `frame_header_length_in_bytes` is the uncompressed header's size and
  `first_partition_size` the compressed header's;
- `mcomp_filter_type` is libvpx's numbering (EIGHTTAP 0, SMOOTH 1, SHARP
  2, BILINEAR 3, SWITCHABLE 4), not the bitstream's;
- the reference fields, and the filter type, are filled for inter frames
  only, and an empty slot is `VA_INVALID_SURFACE`;
- `seg_param[8]` has each segment's reference and skip features, its
  `filter_level[ref][mode]` and its four quantizer scales; the segment
  tree and prediction probabilities are 255 where not coded.

Each convention is confirmed only by the bit-exact runs in "VP9 on the
AMD GPUs". Whether VCN reads `seg_param`'s quantizers and filter levels or
works them out from the header itself is not measured: the streams that
use them match either way.

The binding's layouts are checked against the installed headers by the
ignored test `vp9_layouts_match_libva_headers`: it passed on 2026-10-24
against libva 2.24.1's `/usr/include/va` (`pkg-config --modversion libva`
1.24.0).

### Surfaces and output

- Profile 0 decodes to NV12, Profile 2 to P010.
- One pool of 8 + 1 + 2 surfaces at the frame's size rounded to even.
  An inter frame of another size gets a new pool at its size, in the same
  context, while the references keep the surfaces they are in: libvpx's
  dynamic resize streams, whose frames predict from references of another
  size, decode bit-exact on both AMD GPUs that way (p0-720p-resize below).
- A frame comes out at its coded size; the render size is not applied, as
  ffmpeg's and libvpx's decodes do not apply it.

### What it refuses

With `Unsupported` before anything reaches the GPU: Profiles 1 and 3
(4:2:2, 4:4:0, 4:4:4), any chroma but 4:2:0, and 12 bits. radeonsi
offers neither Profile 1 nor 3 ("The AMD drivers"), and core's formats are
NV12 and P010. `Vp9Decoder::new` checks the `vpcC` record and the driver's
profile, render target format and largest size, so that a session the
driver cannot serve fails when it is made. A stream read without a
record (`Vp9Decoder::new_stream`, IVF) is checked at its first key frame.

The backend refuses, with -12911 at `dsta`, a description under the
driver's smallest surface, 16x16 on radeonsi, and a 10-bit one 32 or fewer
samples wide, as for AV1 above. The host session refuses 12 bits with
-12911, as Apple's decoder does. A frame of another size than the
session's is shown at its own size, after a `Dpba` at that size, where
Apple's decoder emits it with -12909 and no image ([devices.md](devices.md),
"Where we differ from Apple").

## The JPEG decoder

The JPEG engine's `jdec` ([devices.md](devices.md), "The JPEG engine")
decodes on `VaapiJpegDecoder`: core's `JpegDecoder` on the driver's
`VAProfileJPEGBaseline` / `VAEntrypointVLD`. `open_backend` gives it to
the glue as the backend's `jpeg` part, and leaves the part out where the
driver has no JPEG decoder.

### Shape

- The codec crate's `jpeg` parser reads the stream from SOI to EOI: the
  frame header (SOF0, or SOF1 at 8 bits), DQT, DHT, DRI, the one scan's
  header, and where its entropy-coded data ends. An image that defines no
  Huffman table takes Annex K's as tables 0 and 1, as a Motion JPEG frame
  needs. Our Tables K.3 to K.6 equal the four cjpeg writes without
  `-optimize` (`jpeg-check --annex-k`, libjpeg-turbo 3.2.0, 2026-10-24).
- `src/va/jpeg.rs` checks the stream against the request and the driver
  first ("What it refuses"). Then one picture gets, in one
  `vaRenderPicture`, the picture parameters (size, and each component's
  ID, factors and quantization table), the quantization tables (in zig-zag
  order, as the stream has them) and Huffman tables 0 and 1. A second
  `vaRenderPicture` gets the slice parameters (the scan's components and
  tables, the restart interval, the image's MCU count) and the slice data.
- The slice data is the entropy-coded data alone, from after the SOS
  header to the marker after the scan, with its restart markers. No EOI
  is passed, and every run below decoded without one. `color_space` is 0,
  `rotation` none and `crop_rectangle` zero.
- The config, with every RT format the driver lists, is made once. Each
  decode makes one surface and one context and destroys them after, and
  one decode runs at a time, as `jcap` says one core.

### Surfaces and readback

- Each sampling decodes into one surface format: 4:2:0 into NV12
  (`YUV420`), 4:2:2 into YUY2 (`YUV422`), 4:4:4 into 444P (`YUV444`) and
  grey into Y800 (`YUV400`). 4:1:1 would take 411P (`YUV411`), which
  radeonsi does not list.
- The surface is the image rounded up to whole MCUs, and at least the
  driver's smallest, 64x64: a 1x1 image decodes into a 64x64 surface.
  A guest's does too: core sends a decode under 24 blocks of area to the
  backend, which Apple's hardware refuses ([devices.md](devices.md),
  "Where we differ from Apple").
- **The output is read from host memory of ours** (`src/va/jpeg.rs`). At
  scale 1 into the stream's own layout, the decode's surface is an import
  of memory of the engine's (`HostSurface`, the staged copy back's, "The
  copy back"), so the JPEG decoder writes the samples there itself; a
  scale or another layout has its last processing run write there
  instead ("Scales and conversions"). The engine waits for the GPU's
  writes by a `poll` of the memory's dma-buf, holding no lock of the
  driver's, then by the surface's `vaSyncSurface2`, both by the decode's
  deadline ("Waiting for the GPU"), and the CPU reads the memory cached.
  No driver call maps or copies the output for the CPU: radeonsi makes
  those under its display's lock (`vaMapBuffer` of a derived image,
  `vaGetImage`; [bugs.md](bugs.md), U22), and each held every other
  session's calls on the display ("The display's calls", "What the gate
  costs").
- **The engine keeps its surfaces.** Once a decode succeeds, every
  surface it wrote or read, its host surfaces, the driver's surface it
  decoded into and those a conversion passed through, is kept for later
  decodes of the same format, size and corner written (`Shape` in
  `src/va/jpeg.rs`). Past 64 MiB the least recently used are destroyed,
  but for the last decode's host surfaces, which stay whatever their size;
  a 4000x3000 4:2:0 image's surfaces, decoded into `420f` whole and at
  1/2, 1/4 and 1/8, are 48 MB. Making a 4K host surface took 2.3 to 6 ms,
  about a decode's time, and a destroy costs more: the unmap has amdgpu
  make every later submission of the process, on any ring, wait for every
  job the process's GPU VM had in flight then, a guest's on the display
  among them, since RADV and radeonsi in one process are one VM
  ([bugs.md](bugs.md), B92, U23). **Measured** on both AMD GPUs (Linux
  7.2.9, Mesa 26.2.4, 2026-10-26), the GPU scheduler's tracepoints of a
  job's dependencies, the graphics ring held by the tests' spin: a 256x256
  host surface destroyed had the next decode's job wait for the spin's,
  through the unmap's page-table update, until its deadline; with nothing
  destroyed, every decode ran beside the spin. When the engine destroyed
  each conversion's surfaces after it, every decode after a conversion
  waited so. Kept, the JPEG gate's digests were the same before and after
  on both GPUs, at its default sizes and with `--large`. An output read in
  strips (below) is a surface of the driver's past 64 MiB by itself, so
  its decode leaves its strip's host surface kept, and destroys the rest.
- **An output past udmabuf's limit is read in strips** (`Strips` in
  `src/va/jpeg.rs`). Past `size_limit_mb`, 64 MiB at its stock value (an
  8192x8192 4:2:0 image's 101 MB is past it, a 4000x3000 4:4:4 one's
  37 MB not), no host surface holds the output, so the decode, or the
  conversion's last run, writes it into a surface of the driver's, and
  for each band of its rows a 1:1 processing run copies the band into the
  top of one host surface of the output's width, 32 MiB at most
  (`STRIP_BYTES`): 2720 rows of 8192 for 4:2:0, so 4 strips for
  8192x8192, 12 for 16384x16000 4:2:0 and 8 for grey. The CPU reads each
  strip as above, once its run is done, into the guest's rows for the band
  (`write_band`). The run is the staged copy back's (`readback::copy`,
  "The copy back"), aimed at a band: the same colour standard both sides
  and no scale, so each sample is read at a sample's centre and nothing is
  computed; each band starts on an even row, so 4:2:0's chroma rows are
  whole. The strip's host surface is kept with the others, so a run of
  decodes frees nothing for its strips ([bugs.md](bugs.md), B92).
  **Measured** on both AMD GPUs (Mesa 26.2.4, 2026-10-26): radeonsi's
  video processing takes the band's rectangle as its source region
  (`surface_region`) and its window (`output_region`); with every output
  of the JPEG gate read in strips, bands of 16 rows and more (a build of
  ours that made no host surface whole, not in the tree), all 99 cases
  wrote the same bytes as read whole, by their digests, on both GPUs:
  copies of 4:2:0, 4:4:4 and grey, conversions into NV12, `BGRA` and,
  through NV12, the 4:2:2 and 4:4:4 layouts, crops and odd sizes among
  them ("Testing on the AMD GPUs").
- **A 4:2:2 copy past the limit is read through an alias** of its YUY2
  surface (`Alias` in `src/va/readback.rs`). radeonsi's video processing
  writes no 4:2:2 surface: Mesa's compositor refuses each as a target
  (`vl_compositor_proc.c`, **read** in Mesa 26.2.4), and a 1:1 run into
  YUY2 or UYVY failed at `vaEndPicture` with
  `VA_STATUS_ERROR_OPERATION_FAILED` on both GPUs. So the engine exports
  the YUY2 surface (`vaExportSurfaceHandle`) and imports its dma-buf again
  as BGRA of half its width, at its offset and pitch, and each band's run
  copies from that alias into a BGRA host surface of the same width, whose
  rows are the output's bytes as YUY2 lays them out (`strip_format`):
  2048 rows of 8192, so 4 strips for 8192x8192. That holds because
  radeonsi lays out every YUY2 surface linearly, in one plane (its tiling
  takes no 4:2:2 format, `ac_surface.c`, **read**; the export's modifier
  was linear on both GPUs): each BGRA texel is one Y0 Cb Y1 Cr group, and
  a 1:1 run from RGB to RGB reads each texel at its centre and writes it
  unchanged, its alpha among it. The samples are never taken as YCbCr, so
  their siting does not arise, and 4:2:2's chroma is whole on every row.
  The YUY2 surface is made with the hint that it will be exported (the
  memory type `VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2` with no
  descriptor), so that radeonsi makes it shareable then, where it would
  move a surface made for the process alone into new memory at its first
  export and free the old (`si_texture_get_handle`, **read**), an unmap
  ([bugs.md](bugs.md), B92). The alias holds the same memory, and
  destroying it frees nothing. A trial run of an alias at start says
  whether the driver copies one; where it does not, a 4:2:2 copy past the
  limit is read through the driver's image. **Measured** on both AMD GPUs
  (Mesa 26.2.4, 2026-10-26): with a program on libva alone, the alias's
  bands of up to 32 MiB gave every byte put into a YUY2 surface of noise,
  at 336x216, 8192x8192 and 16384x16000; with every output of the JPEG
  gate read in strips, its 10 copies of 4:2:2 gave the digests they gave
  read whole, and with `--large`, its 8192x8192 4:2:2 copy in strips gave
  the digest its derived map gave ("Testing on the AMD GPUs"). Its runs
  hold no lock of the driver's past their submission, where the map held
  every other session's calls 20.5 ms on the RX 7900 XTX and 5.2 ms on
  the Radeon 780M ("What the gate costs"). `vaGetImage` of each band,
  also exact, held the lock for the band's copy on the CPU: 13.1 to 14.8
  ms at the mean for 32 MiB on the RX 7900 XTX and 9.5 to 10.2 on the
  Radeon 780M ([bugs.md](bugs.md), B90).
- **Measured** on both AMD GPUs (Mesa 26.2.4, 2026-10-26), against the
  derived map of a decode into the driver's own surface: a decode into
  host memory wrote the same samples, every one inside the image, for
  4:2:0, 4:2:2, 4:4:4 and grey at 333x211, 640x480 and 3840x2160 (the
  MCUs' padding past the image, which nothing reads, differed), in a
  decode's time (4K 4:2:0: 7.2 against 7.4 ms on the RX 7900 XTX, 20.7
  against 19.9 on the Radeon 780M). A 1:1 processing run from the
  driver's surface into host memory wrote the same bytes for NV12, 444P
  and Y800, 1.5 to 5.6 ms more for 4K, and radeonsi refused it into YUY2
  (`VA_STATUS_ERROR_OPERATION_FAILED` at `vaEndPicture`: Mesa's
  compositor writes no 4:2:2 surface). So the decode writes host memory
  itself.
- **Where the driver's images are read instead**: by `vaDeriveImage` and
  its map, or `vaCreateImage` and `vaGetImage` where the driver cannot
  derive, made on the display's call thread by the decode's deadline
  ("The display's calls"); past it the decode fails, with no other way
  tried. So on a driver whose writes into imports the import check did not
  pass (`Display::writes_imports`, "The import check");
  where no host surface can be made, as with no `/dev/udmabuf` the
  process may open; for an output past udmabuf's limit with no strips, on
  a driver with no video processing, or a 4:2:2 copy where the trial at
  start copied no alias, the log saying why; where a check asks for
  another way back (`RVGV_VAAPI_READBACK`, "The copy back"); and from the
  first decode on which a driver call writing host memory failed and the
  same decode into the driver's own surfaces did not, with a warning in
  the log. On radeonsi each such read derived its image. A map holds the
  display's calls as before ([bugs.md](bugs.md), B90).
- At scale 1, core's `write_samples` copies the region the
  guest asked for out of the output into the guest's layout, in the
  host's buffer the session gave the decode, which the session then
  copies into the guest's surface ([core.md](core.md), "The JPEG engine's
  lends"): biplanar
  4:2:0, 4:2:2 or 4:4:4 from the same sampling, `yuvs` and `2vuy` from
  4:2:2, and every YCbCr layout from grey, with chroma 0x80. It moves
  samples, and never computes one. A scale and every other layout go
  through video processing first ("Scales and conversions").
- fdinfo shows the work on the VCN's JPEG engine, `drm-engine-jpeg`, and
  a conversion's runs on `compute`. For a 640x480 4:2:0 decode copied out
  on the RX 7900 XTX, `jpeg` was busy 0.82 ms and nothing else; the
  derived map that read it was 1.03 ms of `compute` (jpeg-check's
  `engines` lines, 2026-10-26).

### Scales and conversions

A decode at 1/2, 1/4 or 1/8, or into a layout of another sampling than
the stream's, or into `BGRA`, decodes as above, and then the region of
the decoded surface goes through the driver's video processing
(`src/va/vpp.rs`). `Route` in `src/va/jpeg.rs` picks how:

| output | runs |
|---|---|
| `420f`, `420v` | one into NV12, its luma and chroma |
| any YCbCr layout, from grey | one into NV12, its luma alone; core writes the chroma 0x80 |
| `422f`, `422v`, `yuvs`, `2vuy` | one into NV12 for the luma, and one into NV12 twice the output's height, whose chroma is the layout's 4:2:2 |
| `444f`, `444v`, from 4:4:4 | one into NV12 for the luma, and one into NV12 twice the output's size both ways, whose chroma is the layout's 4:4:4 |
| `BGRA` | one into BGRA |

Then core's `write_samples`, or `write_bgra`, copies the result into the
guest's layout. The NV12 detours are ours: radeonsi's video processing
writes no 4:2:2 or 4:4:4 surface (measured through the backend's scaler,
since removed, which refused every 4:2:2 output on both AMD GPUs,
2026-10-24), and an NV12 surface's chroma at those sizes is the
layout's.

- **Which conversions the driver does** is tried at start: each surface
  the decoder writes (NV12, YUY2, 444P, Y800) is halved, 64x64 into
  32x32, into NV12 and into BGRA, and kept where the run succeeds. Every
  case below ran on both AMD GPUs, so
  all eight trials passed on each.
- **Matrix, range and siting.** Both sides name BT.601 at full range, a
  JFIF image's, and the source's chroma is named centred both ways
  (`VA_CHROMA_SITING_VERTICAL_CENTER | ..._HORIZONTAL_CENTER`), as JFIF
  sites it. Unnamed, Mesa 26.2.4 takes 4:2:0 chroma as sited on the left,
  as MPEG-2's: a 640x480 4:2:0 stream into `BGRA` scored 69.2 to 70.8 dB
  per channel against a conversion of chroma interpolated so, and 29.9
  to 35.3 dB against one from the centres. Named, it scored 69.0 to
  70.6 dB against the centres. From YUY2 (4:2:2), Mesa takes the nearest
  chroma sample whatever is named: 67.8 to 73.0 dB against that, 29.6 to
  35.0 dB against the centres (both AMD GPUs, 2026-10-24). The
  conformance suite's `siting` lines, run through the backend's scaler,
  since removed, found otherwise for YUY2: a 1:1 conversion of `yuvs` or
  `2vuy` into `BGRA` weighed the chroma 0.25 0.50 0.25, interpolated from
  the left, named or not (both AMD GPUs, 2026-10-24). Why the two runs
  differ was not read.
- **Each run shrinks by half a side at most.** One run of 1/4 or 1/8
  scored 22.1 to 42.6 dB against core's reference (the worst planes Cr
  29.5 at 1/4 and 22.1 at 1/8, from 640x480 4:2:0), where 1/2 scored 61
  to 64. That fits a filter that weighs two samples a side, which
  averages all of a halving's and skips half of a quarter's (a
  hypothesis). So a larger shrink passes through
  NV12 surfaces, each half the last (`steps`); then the same streams
  scored 59.2 and 57.8 dB. A source's chroma also shrinks by half at
  most in one run, so 4:4:4 goes into NV12 at its own size first.
- **Into BGRA**, every pixel takes chroma, so the NV12 surfaces stop at
  twice the output's size, whose chroma is the output's, and one run
  goes from there (`chain`). Halved all the way, 4:2:2 at 1/4 into
  `BGRA` scored 22.4 to 27.8 dB; so, 57.8.
- The surfaces a conversion passes through are the driver's, and its
  target is host memory of ours ("Surfaces and readback"); both have
  sides rounded up to 16, and only their top-left corner is read. The
  source region is a plain crop, which Mesa does right (measured through
  the backend's scaler, since removed, on both AMD GPUs, 2026-10-24).

### What it refuses

Each before anything reaches the GPU, with core's `JpegError`, which the
session turns into the guest's status ([devices.md](devices.md), "The JPEG
engine"):

| what | error | the guest gets |
|---|---|---|
| progressive, lossless, hierarchical or arithmetic coding, JPEG-LS, 12-bit samples, a height left to a DNL, an image in more than one scan | `UnsupportedStream` | kIOReturnBadMedia |
| a quantization value over 255, as 16-bit tables can hold: VA-API's tables are 8-bit | `UnsupportedStream` | kIOReturnBadMedia |
| Huffman tables 2 and 3, a DC table of over 12 values or an AC table of over 162: VA-API's buffer has two of each, of those sizes | `UnsupportedStream` | kIOReturnBadMedia |
| RGB components (an Adobe segment with transform 0) | `UnsupportedStream` | kIOReturnBadMedia |
| a sampling the driver does not list (4:1:1 on radeonsi), or a size over its largest (16384x16384) | `UnsupportedStream` | kIOReturnBadMedia |
| a stream cut short, without its EOI, or malformed | `BadData` | kIOReturnBadMedia |
| damaged entropy-coded data: a bad Huffman code, a coefficient past the 63rd, a restart marker missing or out of turn, data past the last MCU that Apple's hardware refuses | `BadData` | kIOReturnBadMedia |
| a scan cut short and closed with its EOI | `ScanEndsEarly` | kIOReturnDeviceError |
| a size other than the options' | `SizeMismatch` | kIOReturnBadMedia |
| a sampling other than the options' | `SubsamplingMismatch` | kIOReturnDeviceError |
| `444f` from 4:2:0 or 4:2:2, which Apple's hardware refuses too | `UnsupportedFormat` | kIOReturnError |
| a scale other than 1, where the driver has no video processing or the conversion's trial run failed | `UnsupportedScale` | kIOReturnUnsupported |
| at scale 1, a layout of another sampling or `BGRA`, likewise | `UnsupportedFormat` | kIOReturnError |

**Damaged entropy-coded data is refused before the driver.** The driver
decodes it to wrong pixels and reports success: four copies of a 640x480
4:2:0 stream, 8 bytes of their scans overwritten at offsets 5000 to
30180, each decoded with status 0, up to 255 off the original, on both
AMD GPUs, with no ring timeout or reset in the kernel log (2026-10-24,
with nothing in front of the driver). So after the checks above, the codec
crate's `jpeg::walk` reads the scan on the CPU as a decoder would, every
Huffman code, each coefficient's place, the restart markers and what
follows the last MCU, with no dequantisation or IDCT. It refuses what
Apple's hardware refuses, as Apple's does ([devices.md](devices.md), "The
JPEG engine"). The decode's log line gives the walk's time.

**What the walk costs**, the codec crate's `examples/jpeg-walk.rs`
(`--repeat N`) in a release build on the AMD host's Ryzen 7 7840HS, one
thread (2026-10-24), at a load average near 3, and in brackets near 6, as
other work ran; at 11 to 12 the times were near those at 6:

| stream | bytes of scan | walk | per megapixel |
|---|---|---|---|
| 640x480 4:2:0, quality 10 | 7,813 | 0.08 ms | 0.25 ms (0.44) |
| 640x480 4:2:0, quality 90 | 29,570 | 0.19 ms | 0.62 ms (1.26) |
| 640x480 4:2:0, quality 100 | 58,894 | 0.38 ms | 1.24 ms (2.16) |
| 640x480 4:4:4, quality 90 | 44,149 | 0.36 ms | 1.16 ms (2.01) |
| 1920x1080 4:2:0 | 116,446 | 1.57 ms | 0.76 ms (0.93) |
| 4000x3000 4:2:0 | 709,310 | 7.42 ms | 0.62 ms (0.87) |
| 4000x3000 4:4:4 | 1,121,050 | 12.1 ms | 1.01 ms (1.46) |
| 8192x8192 4:2:0 | 2,609,419 | 26.0 ms | 0.39 ms (0.50) |
| 16384x16000 4:2:0 | 7,512,515 | 77.9 ms | 0.30 ms (0.36) |
| 16384x16000 grey | 5,464,703 | 56.7 ms | 0.22 ms (0.25) |

So 0.2 to 1.3 ms a megapixel, about 6 to 11 ns a byte of scan: the
cost follows the bytes, not the pixels. Against the decode it guards, on
the RX 7900 XTX: 7.4 ms of 25.8 for 4000x3000 4:2:0, 78 ms of 529 for
16384x16000 ("Testing on the AMD GPUs", whose times with the walk include
it).

### Testing on the AMD GPUs

```sh
crates/reims-vgpu-video-vaapi/tools/jpeg-check.sh [--large] [NODE]...
```

The script makes its streams with cjpeg and jpegtran from ffmpeg's
`testsrc2`, cropped to exact sizes. It writes two references for each
stream that should decode:
- libjpeg's raw output with its islow IDCT (`tools/jpeg-planes.c`): the
  components' samples, with no upsampling and no colour conversion;
- ffmpeg's mjpeg decode, in the decoder's own planar format.

`vaapi-conformance NODE jpeg-check` then decodes the stream through
`open_backend`'s `jpeg` part on each render node, and compares every plane
with both references. A reference libjpeg warns about, or decodes short,
stops the script (`jpeg-planes.c` exits 1), and a case that should decode
fails with no reference to compare with.

The cases:
- qualities 10 to 100;
- 4:2:0, 4:2:2, 4:4:4 and grey;
- restart markers every MCU, every third and every MCU row, for each
  sampling;
- odd and tiny sizes (1x1, 17x9, 333x211, 4095x7, 7x4095);
- the driver's largest width and height (16384x64, 64x16384);
- with `--large`, 8192x8192 4:2:0 and 4:2:2, and 16384x16000. ffmpeg
  makes no 16384x16384 or 16384x16382 frame;
- optimized Huffman tables, and Annex K's;
- streams with their DHT segments stripped;
- crops in MCUs, at the image's edge among them;
- each of the other layouts, at scale 1 the stream's own and grey's
  copied, the rest converted;
- scales of 1/2, 1/4 and 1/8, into each kind of layout, of crops and odd
  sizes among them;
- each refusal above, three streams with 8 bytes of their scans
  overwritten and one cut short and closed with an EOI among them.

A decode that copies samples is compared sample for sample, within the
tolerance. One that scales or converts is compared, by PSNR per plane,
with core's scaler reference (`reference::resample`) of the reference
decoder's planes, resampled to the output's pixels and, for `BGRA`,
converted with BT.601 at full range: it passes at 34 dB for a conversion,
30 dB for a scale, and 28 dB where the output's size is odd, as the
conformance suite's scaler cases do ([scaler.md](scaler.md)). `BGRA`
passes against the chroma interpolated from its centres, or taken from the
nearest sample, as Apple's looks to be.

A stream outside what the backend answers it decodes (`JpegDecoder::decodes`:
the samplings, and the largest picture) passes refused as
`UnsupportedStream`, "refused as answered", and fails decoded; inside the
answer every refusal fails. So a GPU whose driver lists fewer samplings,
or a smaller picture, passes the cases past them, by its own answer: the
RX 550's lists no 4:4:4, and 4096x4096 at most. The script's last line
counts the passes, those refused as answered among them, and the failures.

**Results**, 2026-10-24, on both AMD GPUs with `--large`: all 102 cases
passed on each GPU, the same on both. Of the decodes that copy samples:
- Every plane was within 1 of both libjpeg-turbo 3.2.0 and ffmpeg
  n9.0.2: 44 planes equal, 480 off by 1 at most.
- The mean difference was under 0.03 everywhere but Cb at qualities 10
  to 50. There 14% of Cb samples differ by 1 from libjpeg (mean 0.14),
  and 27% from ffmpeg (0.27).
- **The IDCT tolerance is 1**, and the script's default.

| decode, without the walk | RX 7900 XTX | 780M |
|---|---|---|
| 1920x1080 4:2:0, mean of 20 | 4.75 ms | 6.99 ms |
| 4000x3000 4:2:0, mean of 10 | 22.2 ms | 35.2 ms |
| 4000x3000 4:4:4, mean of 10 | 50.8 ms | 78.2 ms |
| 8192x8192 4:2:0 | 116 ms | 204 ms |
| 16384x16000 4:2:0 | 405 ms | 773 ms |
| 16384x16000 grey | 278 ms | 486 ms |

Each time covers the parse, the surface and context, the decode and the
copy into the destination. The walk adds its own ("What the walk
costs"): with it, 8192x8192 4:2:0 took 135 ms on the RX 7900 XTX and
16384x16000 4:2:0 529 ms, with other work running.

**Again**, 2026-10-26, on both AMD GPUs, the tree that read every output
through the driver's images and the tree that reads them from host memory
("Surfaces and readback"): all 99 cases passed on each GPU, and all 102
with `--large`, each decode's `digest` (the MD5 of every byte it wrote
into the destination's planes, their padding among them) the same before
and after, and on both GPUs. The outputs `--large` makes past udmabuf's
64 MiB, 8192x8192 and 16384x16000, were read through the driver's images
in both trees.

**In strips**, 2026-10-26, on both AMD GPUs: the tree before the strips
and the tree after passed all 102 cases with `--large`, each digest the
same, the outputs past 64 MiB read in strips after. A build of the tree
after that made no output's host surface whole, so that every output
was read in strips, of 64 KiB and of 1 MiB (bands of 16 rows and more),
passed all 99 cases on each GPU, each digest the same as before: on each
GPU 118 decodes read in strips, every route among them, and the 10
copies of 4:2:2 through the driver's image. The large decodes' own
times, means of 5 with the walk, two runs of each, before and after:

| decode | RX 7900 XTX, maps | strips | 780M, maps | strips |
|---|---|---|---|---|
| 8192x8192 4:2:0 | 124.2 to 124.6 ms | 121.2 to 123.0 ms | 211.7 to 215.6 ms | 216.2 to 217.4 ms |
| 16384x16000 4:2:0 | 432.6 to 440.8 ms | 422.4 to 431.3 ms | 765.4 to 773.1 ms | 777.7 to 779.3 ms |
| 16384x16000 grey | 274.9 to 277.3 ms | 292.2 to 293.3 ms | 500.3 to 502.5 ms | 535.2 to 537.2 ms |

**4:2:2 in strips**, 2026-10-26, on both AMD GPUs: the tree that mapped a
4:2:2 copy past 64 MiB and the tree that reads it through an alias of its
surface ("Surfaces and readback") passed all 103 cases with `--large`,
its 8192x8192 4:2:2 copy among them, each digest the same. The tree
after, built to read every output in strips of 64 KiB and of 1 MiB,
passed all 99 cases on each GPU, each digest the same as before: on each
GPU all 128 decodes read in strips, every route among them, the 10
copies of 4:2:2 too, and none through the driver's image. The large
4:2:2 decodes' own times, alone, means of 5 with the walk, two runs of
each:

| decode, into `422f` | RX 7900 XTX, maps | strips | 780M, maps | strips |
|---|---|---|---|---|
| 8192x8192 4:2:2 | 229.8 to 230.7 ms | 240.2 to 240.5 ms | 353.7 to 353.9 ms | 358.8 to 359.1 ms |
| 12288x8192 4:2:2 | 337.3 to 338.1 ms | 357.6 to 358.6 ms | 521.3 to 521.8 ms | 528.9 to 529.3 ms |

The strips took 1 to 6% longer. Each band's run is waited for before the
CPU reads the band, and the next band's starts after; where the time
goes was not measured.

**The damaged streams** were each refused by the walk before the driver,
with a reason of its own: a coefficient past the 63rd (offset 5000), data
after the last MCU (15000), a bad Huffman code (29000), and
`ScanEndsEarly` for the one cut with an EOI.

**Scales and conversions** (34 cases, and two refusals of `444f`), the
worst plane of each, the same on both GPUs:
- conversions at scale 1, 61.5 to 75.0 dB;
- scales of even sizes, 56.8 to 67.1 dB, 1/8 at 56.8 to 59.6;
- odd outputs, 30.9 to 32.6 dB (333x211 at 1/2, a 93x51 crop of 4:4:4
  at its edge), and 46.2 dB for 4000x3000 at 1/8, whose 500x375 is odd.

A 640x480 decode took 1.7 to 5.5 ms converted, against 2.3 ms copied,
and 4000x3000 4:2:0 at 1/8 took 21.9 ms on the RX 7900 XTX and 41.0 ms on
the 780M. The backend's scaler, since removed, passed its 106 cases on
each GPU with the same sitings.

## The encoders

H.264 (Constrained Baseline, Main, High) and HEVC (Main, Main 10) on
`VAEntrypointEncSlice`, with the parameter sets and slice headers ours,
handed to the driver as packed headers. The driver writes the slice data,
and on radeonsi the parameter sets as well (below).

The portable half is the codec crate's (`crates/reims-vgpu-video-codec`),
re-exported as `encode`; the rest is this crate's.

| what | where |
|---|---|
| Settings, input frames, samples, profiles and rate control. Plain Rust | the codec crate's `src/encode.rs` |
| SPS/VUI/PPS/slice headers, `avcC`, levels (Table A-1) | the codec crate's `src/encode/h264.rs` |
| VPS/SPS/VUI/PPS/slice segment headers, `hvcC`, levels (Tables A.8, A.9) | the codec crate's `src/encode/hevc.rs` |
| Picture types and order, references, decode times | the codec crate's `src/encode/gop.rs` |
| Our rate control under `DataRateLimits` (below) | the codec crate's `src/encode/limit_rate.rs` |
| The core over an `Accelerator` trait, as `h264::Decoder` is | the codec crate's `src/encode/encoder.rs` |
| Core's `BackendEncoder` over the core: the host session's encoder | the codec crate's `src/encode/session.rs` |
| libva's encode binding, checked against the headers | `src/va/ffi/encode.rs`, `src/va/ffi/encode_layout.c` |
| The driver's encoder attributes, and `vaapi-encode-probe` | `src/va/encoder/caps.rs`, `examples/vaapi-encode-probe.rs` |
| The VA accelerator: config, context, surfaces, buffers, coded buffer | `src/va/encoder/accelerator.rs`, `params.rs`, `surfaces.rs` |
| `VideoEncoder`, and `VaapiEncoderBackend` (core's `EncoderBackend`) | `src/va/encoder.rs`, `src/va/encoder/backend.rs` |
| The checks' tools | `examples/vaapi-encode/`, `examples/conformance.rs` (its `encode`) |

The codec crate's half names no libva type, so another backend reuses the
headers, the levels and the GOP planning: the NVIDIA backend's encoders
run on its `SessionEncoder`, `KeyFrames` and `DecodeTimes`.

### What the driver offers

`vaapi-encode-probe`, run 2026-10-24 on both AMD GPUs, Mesa 26.2.4. The
answers are the same on both nodes, and for every profile but where shown.

| attribute | H.264 (CB, Main, High) | HEVC (Main, Main 10) |
|---|---|---|
| RT formats | 0x1 (YUV420) | 0x101 (YUV420, YUV420_10) |
| rate control | 0x416: CBR, VBR, CQP, QVBR | the same |
| packed headers | 0x1f | 0x1f |
| references (L0, L1) | 0x10002: 2 and 1 | 0x1: 1 and 0, so no B pictures |
| slices, structure, quality range | 0x80, 0x26, 0x20 | the same |
| largest picture | 4096x4096 | 8192x4352 |
| HEVC features | | 0x10541050: AMP, SAO, strong intra smoothing, constrained intra, transform skip, cu_qp_delta, deblocking disable supported; none required |
| HEVC block sizes | | 0xcf: CTB 64, min CB 8, TB 4 to 32, transform depth 0 |

### Packed headers: radeonsi writes its own

The parameter sets that come out are not the ones packed. Measured
2026-10-24 on the RX 7900 XTX by writing the packed sets ahead of the
hardware's in one stream and reading both with ffmpeg's `trace_headers`
(`vaapi-encode` writes `PREFIX.packed.record` and `PREFIX.record`):

- H.264 High, 1080p: `transform_8x8_mode_flag` 1 became 0, and the VUI's
  bitstream restriction changed (`max_bytes_per_pic_denom` 2 to 0,
  `max_bits_per_mb_denom` 1 to 0, both `log2_max_mv_length` 15 to 16).
- HEVC, 1080p: `pic_height_in_luma_samples` 1080 became 1088, cropped by a
  conformance window; both `max_transform_hierarchy_depth`s 0 became 4, though
  the driver reports a depth of 0; `bitstream_restriction_flag` became 0.
- HEVC at 96x64: our 136x128 became 192x128, with a wider window.

The colour, range and timing that the VUI carries came through: ffprobe
reads them back (the colour case below). Whether the driver parses our
packed sets or builds its own from the parameter buffers is not measured.
So the configuration record is made from the parameter sets the hardware
emits with each IDR picture: the first sample carries the record it
belongs with, and `configuration()` before the first frame is only ours.

### Shape

- **Frames go in display order**, NV12 or P010 in host memory, with any
  stride. Samples come out in decode order: length-prefixed NAL units with
  4-byte lengths, no parameter sets and no AUD, with PTS, DTS, duration,
  picture type, sync and dependency flags. The first sample after the
  parameter sets change carries the new record.
- **Pictures.** IDR at the first frame, at a forced key frame, where the
  interval in frames or in time stamp ticks runs out, and 2^30 frames
  after an IDR picture at the latest. So a picture's place in display
  order from its IDR picture is at most 2^30 - 1, and its order count,
  twice that in H.264 and that in HEVC, within the 32 bits H.264 8.2.1 and
  HEVC 8.3.1 allow (`MAX_ORDER` in the codec crate's `src/encode/gop.rs`).
  A guest meets that key frame only in a stream with no IDR picture for
  2^30 frames, 207 days at 60 frames a second. H.264 Main and High put B
  pictures between references, one by default and at most 4, L0 two
  references and L1 one. A run of B pictures is closed with a P before an
  IDR. HEVC and Constrained Baseline are I and P only.
- **Time stamps.** Each sample's DTS is the PTS of the frame sent in its
  place, as Apple's encoders give them ([devices.md](devices.md), "Decode
  times"): without reordering DTS equals PTS, and with it
  `I0 P2 B1` gets DTSes 0, 1 and 2 frames, so a B picture's is after its
  PTS. The DTSes increase. The NVIDIA backend's come from the same
  `DecodeTimes`.
- **Levels** are the lowest that fit the coded size, frame rate and peak
  bitrate, or the one asked for.
- **Rate control**: CQP, CBR, VBR (average and peak) and QVBR, with an HRD
  buffer of one second at the peak. The bitrate, the frame rate and the
  key-frame interval change mid-stream from the next picture; the mode
  does not.
- **Small and odd sizes.** Below the driver's minimum (128x128, HEVC
  130x128) the picture is padded up to it. Every size is padded to the
  coded size (16 for H.264, the minimum CB for HEVC) by repeating the last
  column and row, and the SPS crops back to the picture (frame cropping, or
  the conformance window). Odd widths and heights are refused, since 4:2:0
  cannot crop by one.
- **Upload.** The first frame tries `vaDeriveImage`, then falls back to
  `vaPutImage` through an image. radeonsi derives NV12 and P010 encode
  surfaces on both nodes. On 1080p H.264 CQP, Derive ran 231.5 fps on Navi
  31 and 191.9 on Phoenix, against 188.4 and 188.8 for PutImage (single
  runs, 2026-10-24).

### API

```rust
let mut settings = EncoderSettings::new(encode::Profile::H264High, 1920, 1080);
settings.frame_rate = FrameRate::new(30, 1);
settings.timescale = 600;                                  // ticks of pts and duration
settings.rate_control = RateControl::VariableBitRate {
    average_bits_per_second: 6_000_000, peak_bits_per_second: 9_000_000 };
settings.key_frame_interval = Some(60);                    // and/or _duration in ticks
let mut encoder = VideoEncoder::new(&display, settings)?;  // Unsupported: what the driver lacks

let coded: Vec<Coded<Error>> = encoder.encode(&InputFrame {
    planes: [Plane { data: y, bytes_per_row }, Plane { data: uv, bytes_per_row }],
    pts, duration, force_key_frame: false, id })?;         // none yet, or several
// each Ok(EncodedSample), or Err(FailedSample { id, error }) for its frame alone
let rest: Vec<Coded<Error>> = encoder.flush();             // more frames may follow
encoder.configuration();                                   // avcC or hvcC in use
encoder.set_rate_control(..)?; encoder.set_frame_rate(..)?; encoder.set_key_frame_interval(..);
encoder_support(&display);                                 // the profiles the driver encodes
```

`EncoderSettings` also takes the colour, the range, the sample aspect
ratio, reordering, the maximum frame delay, the B pictures, the level, the
entropy coder and VA-API's quality level.

### The host session's encoders

`VaapiEncoderBackend` is core's `EncoderBackend` on a render node,
opened by `VaapiEncoderBackend::open(node)`. It lists, for each codec, the
profiles the driver encodes (66, 77, 100; HEVC 1 and 2) and the largest
picture any of them takes. Its source formats are `420v` and `420f`, with
`x420` and `xf20` where HEVC Main 10 is there. Its encoders reorder where
the profile has B pictures and the driver a list 1: H.264's alone on
radeonsi. Each `esta` opens a `SessionEncoder` (the codec crate's
`src/encode/session.rs`):

- **The stream is made twice at the start.** It is made at open, which
  refuses what the driver cannot encode, and again at the first frame
  when the settings the guest set after `esta`, or the frame's format,
  range and clock, call for another.
- **Later changes.** The bitrate within its mode, the frame rate and the
  key-frame interval reach a running stream. Any other change (profile,
  level, entropy, colour, reordering, bitrate against quality) starts a
  new stream at the next frame, from an IDR with new parameter sets. The
  old stream's last frames come back in a batch of their own, under its
  record, so the session's `Efds` comes between them and the new ones.
- **Time stamps** come back as the guest sent them. DTS is left invalid
  where it equals PTS.
- **Bit depth.** A source of the other depth is converted (8-bit to the
  high byte of 16, 16-bit to its high byte).

Where the guest sets nothing, the choices are ours:

| setting | ours |
|---|---|
| `ProfileLevel` | the best profile the driver has: H.264 High, then Main, then Constrained Baseline; HEVC Main 10 for a 10-bit source, else Main; the lowest level that fits |
| bitrate | with `AverageBitRate` alone, VBR peaking at 1.05 times it, the HRD buffer a second at the peak, so that the stream meets the average within seconds, as Apple's AVE does, and needs no more where the pictures need less (below); with `DataRateLimits`, VBR peaking at two thirds of the limit with a buffer of half a second, at the average where it is under that peak, else at the peak. The limit is a hard one over any window, and VideoToolbox's header has the encoder pick an average "at a safe margin below" it (Apple's AVE made 0.66 of it, [devices.md](devices.md), "The encoder session"). A stream that keeps to the buffer then puts at most the buffer and a second at the peak, the limit, in any second. radeonsi does not keep it, so such a stream's QPs are ours (below, "Rate control under `DataRateLimits`") |
| neither | constant QP from `Quality` (default 0.5): P at 40 - 22 x quality, I 2 below, B 2 above |
| `MaxKeyFrameInterval` | none: key frames at the start and where forced, as an unset 0 reads |
| `ExpectedFrameRate` | the first frame's duration, else 30 |
| `AllowFrameReordering` | off, as core reads an unset one |
| `RealTime` | read, and not used |

Whether Apple's AVE makes the same choices is not measured
([devices.md](devices.md), "The encoder session"), but for the bitrate.
Through the host session (`vaapi-conformance NODE encode`), on the frames
AVE and NVENC were measured on ([nvidia.md](nvidia.md), "The bitrate
through the session"), radeonsi made 0.94 to 0.99 of an `AverageBitRate`
alone over 10 s, 0.88 to 1.05 over 3 s and 0.96 to 1.20 in the first
second; AVE 0.98 to 1.04, 0.97 to 1.16 and 0.95 to 1.27. Peaking at 1.5
times the average it made 0.93 to 0.99, 0.86 to 1.02 and 0.94 to 1.14.
Reordered H.264 made the least, 0.94 over 10 s. radeonsi's VBR does not
fill pictures cheaper than their share even at a peak of the average:
the encode check's waves came to 0.33 to 0.88 of 3 Mbit/s at the average
and at 1.05 times it (both GPUs, Mesa 26.2.4, 2026-10-25, the same
bytes).

`encode-check` passed 33 of 33 on each GPU either way; `-abr` came to
0.98 of its average for H.264 and HEVC (0.97 at 1.5 times, AVE 1.02).

### Testing

On any host, the codec crate's unit tests (`cargo nextest run -p
reims-vgpu-video-codec`; `cargo xtask check` runs them with the rest) test
the portable half: headers parsed back with the crate's own H.264 parsers
and bit readers, GOP planning, and the core and session encoder on a fake
accelerator. The conformance suite's `encode-check` holds every backend's
encoders to the same cases through the host session
([backends.md](backends.md), "The conformance suite").

On both AMD GPUs, a script not in this repository encoded ffmpeg's
synthetic sources (`testsrc2`, `mandelbrot`, `smptehdbars`, noise) through
`examples/vaapi-encode` (`VideoEncoder`) and `vaapi-conformance NODE
encode` (core's session). ffmpeg checked the results, and nothing more: a decode
under `-xerror` and every `-err_detect` check, of the MP4 and the Annex B
stream; PSNR and SSIM against the source; the bitrate against the target;
the GOP, time stamps and sync flags against ffprobe's packets and frames;
and the stream's profile, format and size. H.264 was also decoded by our
own decoder, and its frames' MD5s compared with ffmpeg's.

Run 2026-10-24 on both AMD GPUs, Mesa 26.2.4, 40 cases on each node:

| GPU | result |
|---|---|
| Navi 31, `renderD128` | 40 of 40 pass |
| Phoenix, `renderD129` | 40 of 40 pass |

- **Odd sizes, through the session** (2026-10-24, the same GPUs and Mesa,
  42 cases on each node, 84 of 84 pass): `sess-odd641` and
  `sess-hevc-odd641` encode a 641x361 source (testsrc2 at 642x362, scaled)
  as core does, at 640x360 (ffprobe), and measure 46.3 dB (H.264,
  bit-exact on our decoder) and 47.4 dB (HEVC) against the source's top
  left, alike on both GPUs ([devices.md](devices.md), "The encoder
  session").
- **Both GPUs wrote the same bitstreams** in every case but QVBR (the same
  sizes, PSNR and SSIM); QVBR differed by 0.3% in size.
- **Decoding.** Every MP4 and Annex B stream decoded without an error. Every
  H.264 case decoded bit-exact on our decoder, the record-changing one
  included.
- **The cases.** H.264 CB, Main and High, and HEVC Main and Main 10, at
  1080p (P010 for Main 10), 720p, 480p, CIF noise, 4K (H.264 level 5.1),
  96x64 and 200x120 below the minimum, and 1366x768. Modes CQP, CBR, VBR
  and QVBR; B pictures off and 1 to 3; forced key frames; an interval in
  frames and in ticks; a mid-stream bitrate change, direct and through
  `eset`; full range and BT.709 colour; CAVLC; PutImage. Through the
  session: properties before and after `esta`, Baseline with
  `MaxKeyFrameInterval`, Quality, HEVC 10-bit from `x420`, and a
  `ProfileLevel` change at frame 30, which gave two records.
- **Quality.** 1080p `testsrc2` at CQP 24/26/28: PSNR 46.1 dB, SSIM
  0.997, 7.9 Mbit/s. HEVC at 26/28/30: 46.5 dB at 8.0 Mbit/s. Main 10:
  46.5 dB.
- **Bitrate**, as a share of the target over 2 to 3 s: CBR 0.92 (H.264
  4 Mbit/s), 0.93 and 0.96 (5 Mbit/s noise); VBR 0.87 to 1.06; QVBR 0.93.
  A change from 2 to 6 Mbit/s at frame 45 gave 2.0 then 5.1 Mbit/s.
  In another run that day, static bars at VBR 1 Mbit/s came to 0.02 of
  the target: VBR does not stuff.
- **Keys and sync.** ffprobe's key flags matched our sync flags in every
  case. A 30-frame interval with key frames forced at 10 and 47 gave keys
  at 0, 10, 40, 47 and 77.
- **Speed and hardware.** 1080p H.264 ran 177 to 276 fps through
  `VideoEncoder` and 152 to 223 through the session (single runs over SSH,
  upload and the guest stand-in included); 4K ran 66 to 72 fps. Every
  run's fdinfo showed `drm-engine-enc` time (VCN): 199 to 366 ms for 90
  frames of 1080p H.264.
- **B pictures** (Navi 31, 1080p CQP 26/28/30): none gave 7.50 Mbit/s at
  45.4 dB. One gave 6.61 at 44.4, two 6.71 at 44.0, and three 6.55 at
  43.8; B pictures are coded at QP 30. At VBR 3 Mbit/s (mandelbrot 720p),
  none gave 3.11 Mbit/s at 38.4 dB, one 2.62 at 38.1, and two 2.76 at
  38.7. So the default is one (`DEFAULT_B_FRAMES`): it costs one frame of
  delay, and two were not clearly better in these short runs.

### The bitrate floor

No rate under 32,000 bit/s reaches the driver (`MIN_BITS_PER_SECOND` in
`src/va/encoder/params.rs`, ours). An average, a peak or a constant rate
under it is raised to it, and with it the HRD buffer, which is a second at
the peak, or half a second under a limit. Without it, a guest's
`AverageBitRate` of 1 would reach radeonsi as a VBR of 1 bit/s, with a
buffer of 1 bit, three quarters full at 0. Apple's AVE takes any average
and makes what its quantiser can ([devices.md](devices.md), "This host's
encoder"), so the guest is never refused here either.

The floor is where VCN's output has stopped falling, measured with a sweep
script not in this repository on both AMD GPUs (Mesa 26.2.4, 2026-10-24).
Each run was ordinary rate control: 90 frames at 30 fps, VBR peaking at
1.5 times the average, with the buffer at one second of the peak. The
sweep halved the average from 1 Mbit/s and stopped at the first step
whose output fell by less than 2%. Both GPUs gave the same bytes in every
run. Rates are bit/s of the samples:

| profile | source | sweep's last step: asked, got | at 32,000 |
|---|---|---|---|
| H.264 High | `testsrc2` 640x480 | 31,250: 294,290 | 294,376 |
| H.264 High | `mandelbrot` 640x480 | 62,500: 336,106 | 336,938 |
| H.264 High | `testsrc2` 1280x720 | 125,000: 597,864 | 603,480 |
| H.264 High | `mandelbrot` 1280x720 | 250,000: 732,024 | 740,042 |
| H.264 High | `testsrc2` 1920x1080 | not swept | 1,047,296 |
| H.264 High | `testsrc2` 3840x2160 | not swept | 3,161,930 |
| HEVC Main | `testsrc2` 640x480 | 31,250: 95,426 | 95,426 |
| HEVC Main | `mandelbrot` 640x480 | 15,625: 68,829 | 69,429 |
| HEVC Main | `testsrc2` 1280x720 | 31,250: 169,624 | 169,624 |
| HEVC Main | `mandelbrot` 1280x720 | 31,250: 141,957 | 142,493 |
| HEVC Main | `testsrc2` 1920x1080 | not swept | 339,960 |
| HEVC Main | `testsrc2` 3840x2160 | not swept | 857,213 |

- **Where it bottoms out.** H.264's output stops at 291 to 337 kbit/s at
  640x480, and at 598 to 740 at 720p, near AVE's 265,688 and 642,872 at
  those sizes. HEVC's stops three to five times lower. HEVC `mandelbrot` at 640x480 is the
  one case still falling under 62,500 (92,765 there, 69,258 at 31,250);
  every case had stopped by 31,250.
- **At the floor.** Each source ran at 32,000 peaking at 48,000, and at
  32,000 peaking at 32,000, as an `AverageBitRate` under the floor reaches
  the driver. The second was within 0.3% of the first in every case. No run printed an
  error, and the kernel log showed no reset or ring timeout on either GPU.
  A stream under `DataRateLimits` takes no floor: our rate control
  (below) raises the QP as far as 51.
- **Through the host session**, `vaapi-conformance NODE encode` with
  `--set AverageBitRate=1` on the RX 7900 XTX, libva's trace (`LIBVA_TRACE`) showed
  the driver given 32,000 bit/s at 100%, and a buffer of 32,000 bits,
  24,000 full, for H.264 and for HEVC. The guest's stand-in got 90 of 90
  samples of 640x480 `testsrc2`, at 315,373 and 95,560 bit/s.
- **Not measured:** content harder than `mandelbrot` near the floor, and
  sizes under 640x480, whose output at the floor may be over what the
  encoder could make at a lower rate. That is a stream over the guest's
  target, as AVE's is at every rate under its floor.

### Rate control under `DataRateLimits`

`DataRateLimits` is a hard limit over "any contiguous segment of that
duration (in decode time)" (VideoToolbox's `VTCompressionProperties.h`).
radeonsi's VBR and CBR pass the HRD buffer they are given, and no
parameter the driver reads holds them to it but a minimum QP picked for
the content (below; [bugs.md](bugs.md), U14), so a stream held to a
limit (`EncoderSettings::hard_limit`) is not given to them. Ours (the
codec crate's `src/encode/limit_rate.rs`, driven by `src/encode/encoder.rs`'s
`encode_planned`):

- **The driver runs a constant QP.** The config asks for `VA_RC_CQP`, and
  no rate control or HRD buffer is sent (`StreamParams::picture_qp`). Each
  picture's QP is its slice header's `slice_qp_delta` over the PPS's 26,
  which Mesa takes as the picture's QP (`picture_h264_enc.c:259` and
  `picture_hevc_enc.c:263` in Mesa 26.2.4's `src/gallium/frontends/va/`).
- **The limit is held over the pictures.** No second of decode time holds
  more than the buffer and a second at the peak, which the session makes
  the limit; the pictures to the next key frame share the session's
  average, two thirds of a limit set alone, as AVE's 0.66 to 0.67.
- **A picture that passes its second is coded again**, at a QP that fits
  as predicted, four tries at most, in place on the same context. Mesa
  keeps its DPB by reconstructed surface, so the try takes the same slot
  (`picture_h264_enc.c:79-126`). The VCN takes its references as DPB slots
  (`:236-254`; `src/gallium/drivers/radeonsi/mm/radeon_vcn_enc.c:546`) and
  `frame_num` from the slice (`radeon_vcn_enc_1_2.c:329`), not from the
  counters Mesa moves at each `vaEndPicture` (`va/picture.c:449-455`). A
  try moves Mesa's two-step eviction (`picture_h264_enc.c:55-77`) only of
  DPB entries the picture's `ReferenceFrames` do not list, which are ones
  we have let go.

It needs a driver with `VA_RC_CQP`. Where a driver keeps its buffer
(`DriverLimits::keeps_buffer`), its own rate control is used; VA-API has
no way to ask, and no driver here does.

Measured on both AMD GPUs (both gave the same bytes in every case; Mesa
26.2.4, 2026-10-24 and 25), the conformance suite's limit cases on
radeonsi's VBR, as the session sends it, and on ours: the mean rate, the
busiest second against the limit, and the PSNR of the worst and mean
frame:

| case | radeonsi's VBR | ours |
|---|---|---|
| `avc1-420v-limits` | 1488 kbit/s, 0.97, 33.7 / 35.4 dB | 1385, 0.71, 33.4 / 34.9 |
| `avc1-420v-abr-limits` | 1695 (1.02 of the average), 0.82, 34.6 / 36.1 | 1662 (1.00), 0.66, 34.7 / 36.0 |
| `avc1-420v-limits-keys` | 628, 1.37, 24.0 / 29.0 | 516, 0.74, 24.0 / 26.3 |
| `hvc1-420v-limits` | 1106, 0.74, 35.8 / 36.9 | 1113, 0.71, 35.5 / 36.9 |
| `hvc1-420v-abr-limits` | 1323 (0.99), 0.68, 37.0 / 38.2 | 1337 (1.00), 0.66, 37.1 / 38.3 |
| `hvc1-420v-limits-keys` | 489, 2.10, 25.8 / 29.8 | 375, 0.71, 26.7 / 27.3 |

Ours gives the session's averages, two thirds of a limit set alone, where
radeonsi's went over it by up to 0.2 of the limit on vtencode's frames;
that is the mean PSNR the keys cases give up. At limits of 0.5, 1 and 1.5
Mbit/s on vtencode's frames through the suite's `encode`, radeonsi's VBR
made 1.56, 1.27 and 1.15 of the limit (H.264) and 2.08, 1.74 and 1.19
(HEVC), and ours 0.84, 0.70 and 0.71, and 0.72, 0.70 and 0.74, every mean
at two thirds of the limit; at 1.5 Mbit/s the luma's worst frame was 32.6
and 34.2 dB and its mean 34.2 and 39.6.

**What radeonsi does with the rate parameters**, measured with the suite's
`-limits-keys` cases on the RX 7900 XTX (2026-10-24), each a parameter
changed alone from the VBR the session sends:

- the HRD buffer's size and initial fullness changed no picture's size:
  the first IDR picture took 600 kbit of a 258 kbit buffer. Mesa sends the
  firmware the buffer and `enforce_hrd` 1 (`va/context.c:344`, `:374`;
  `radeon_vcn_enc.c:346`, `:377`, read in Mesa 26.2.4's source). A probe
  on libva written from scratch (not in this repository; [bugs.md](bugs.md),
  U14) shows it: VBR or CBR at 1 Mbit/s with a 0.5 Mbit buffer made key
  frames 2.09 to 2.44 times the buffer, on both GPUs;
- `VAEncMiscParameterTypeMaxFrameSize` was not applied to the first
  picture and loosely after it: the second IDR took 374 kbit of a 258
  kbit cap, P pictures 110 kbit of 17. In the probe, a cap at the
  buffer left the first key frame at 2.09 times it;
- `initial_qp` and `window_size` are not read (`picture_h264_enc.c:360-404`
  reads neither), and a slice's QP is not used under VBR;
- `min_qp` is applied, the first picture's too. A floor of 46 on the key
  frames held `avc1-420v-limits-keys` (0.66 of the limit, a mean of 25.6
  dB, where ours makes 26.3), and 42 did not (0.86), but the floor that
  fits is the content's: one that holds vtencode's key frames costs a
  cheap key frame the quality it could have had;
- a cap or a QP per picture, sent with the picture, made the firmware's
  rate control erratic: P pictures filled their caps and key frames went to
  QP 51; tries in place under VBR fed it their sizes;
- CBR passed the buffer as VBR did.

A try cannot run on a new context over the same surfaces, so tries run in
place: Mesa marks the surfaces `is_dpb` with the old context's DPB
buffers, and the first picture on a new context failed `vaRenderPicture`
with `VA_STATUS_ERROR_INVALID_PARAMETER`, then crashed the process in
`vaEndPicture`, with no GPU reset (measured on radeonsi, Mesa 26.2.4).

### Limits

- HEVC has no B pictures: the driver offers no list 1.
- Odd sizes are refused (4:2:0 crops by two).
- The mode of rate control is fixed for a stream. Through the session, a
  change of mode makes a new stream.
- The slice headers that come out were not compared with the packed ones,
  as the parameter sets were.
- One slice a picture; no intra refresh, ROI or multipass, though the
  driver lists the first two.

## API

The decoders, as a caller drives them (`VaapiBackend` below is the host
session's caller):

```rust
let display = Display::open("/dev/dri/renderD128", runtime)?; // Display: Send + Sync
let caps: Capabilities = display.capabilities()?;         // every profile/entrypoint pair
display.supports(Profile::H264_HIGH, Entrypoint::VLD);    // a quick check

let mut decoder = H264Decoder::new(&display, avcc)?;      // avcC record; Send
// or H264Decoder::new_annex_b(&display)
decoder.set_output_order(OutputOrder::Display);           // crate root or h264::; default Decode
let frames: Vec<Frame> = decoder.decode(sample, id)?;     // one access unit, any id
let rest: Vec<Frame> = decoder.flush()?;                  // emits what waits; references kept
decoder.reset();                                          // forgets references, as for a seek
decoder.profile();                                        // the VA profile in use

frame.id(); frame.poc(); frame.width(); frame.height();   // cropped size
frame.coded_size(); frame.is_idr(); frame.is_reference();
frame.is_full_range();                                    // the SPS's video_full_range_flag
frame.sync()?;                                            // wait for the decode
frame.copy_nv12_to(y, y_stride, uv, uv_stride)?;          // waits, then copies; Frame: Send + Sync

let mut decoder = HevcDecoder::new(&display, hvcc)?;      // hvcC record; or new_annex_b
frame.bit_depth();                                        // 8 (NV12) or 10 (P010)
frame.copy_p010_to(y, y_stride, uv, uv_stride)?;          // 2 bytes a sample; the other refuses
```

`HevcDecoder` has H.264's methods. Its errors are
`Error::HevcBitstream(hevc::Error)`, and the portable layer is public as
`hevc::Decoder`, `hevc::check_supported(&Sps)` and the parsers.

- `decode` returns frames in decode order, each with its POC: one per
  picture, at once. In display order it returns the frames the DPB bumps.
- `VaapiBackend` (Linux) is the host session's backend over all this:
  core's `DecoderBackend`, listing H.264 profiles 66, 77 and 100 where the
  driver decodes them, `420v` or `420f` by the SPS
  (`src/va/backend.rs`; [core.md](core.md)).
- `Frame::poc` is `PicOrderCnt()` after operation 5's subtraction.
- `Error` is `Load(LoadError)`, `Open(io::Error)`, `NoDisplay`,
  `Va { call, status, message }`, `Bitstream(h264::Error)`,
  `HevcBitstream`, `Av1Bitstream`, `Vp9Bitstream`, `Unsupported(String)`,
  `OutOfSurfaces`, or `PlaneTooSmall(&'static str)`.
- The portable layer is public too: `h264::Decoder` over any
  `Accelerator`, `h264::check_supported(&Sps)`, and the parsers.

## Testing

### On any host

The codec crate's unit tests (`cargo nextest run -p
reims-vgpu-video-codec`) test the portable layer. The streams for them are
written in the tests from the standard's syntax. They cover what x264 does
not write:
- order count type 1 with non-reference pictures;
- type 2 across a `frame_num` wrap;
- long-term references and MMCO 1 to 6;
- operation 5's reset and bump;
- list modification, and the gap process;
- references refused when missing.

### On the AMD GPUs

On a Linux host with libva and the oracles below, from the checkout's
root:

```sh
crates/reims-vgpu-video-vaapi/tools/h264-check.sh /dev/dri/renderD128

# the binding's layouts against the installed headers
cargo test -p reims-vgpu-video-vaapi --lib -- --ignored
```

Streams, references and logs go to `build/h264-check/` (or
`RVGV_H264_OUT`), which is reused between runs and never committed.

**x264 and ffmpeg are test oracles.** x264 0.165.3222 writes the streams,
from ffmpeg n9.0.2's `testsrc2` with noise. ffmpeg decodes each in
software (`-hwaccel none -pix_fmt nv12 -f framemd5`) for the reference MD5
of every frame. Neither is linked into anything. None of their output is
committed, and the product never calls them.

Each stream is decoded three times, and every frame's NV12 MD5 must
equal ffmpeg's:
- from its MP4, as VideoToolbox hands a stream over (the `avcC` record,
  length-prefixed samples), in display order;
- as Annex B, in display order;
- in decode order, compared as a set.

Then:
- 2- and 3-byte NAL lengths;
- destination strides padded by 192 bytes;
- a caller holding 24 frames;
- a mid-stream switch from Baseline 720p to High 1080p;
- an interlaced (MBAFF) stream and a fake-interlaced one, which must be
  refused;
- the crafted streams ("Crafted streams", above).

What each stream holds, as `h264-info` reads it:

| stream | frames | profile, POC type | slices a picture | I pictures | B, refs, MMCO | other |
|---|---|---|---|---|---|---|
| baseline-720p | 90 | 66, type 2 | 4 | 2 IDR, 1 non-IDR | none, 3 refs | CAVLC |
| baseline-720p-poc1 | 90 | 66, **type 1** | 4 | 2 IDR, 1 non-IDR | none | baseline-720p's SPS rewritten by `h264-poc1` |
| baseline-1080p | 45 | 66, type 2 | 2 | 1 IDR, 1 non-IDR | none, 2 refs | |
| main-720p | 120 | 77, type 0 | 2 | 2 IDR, 1 non-IDR | 176 B slices, B-pyramid, 4 refs, MMCO 1 x56 | CABAC, weighted P (58 tables), 216 list modifications |
| main-1080p-opengop | 90 | 77, type 0 | 3 | 1 IDR, 3 non-IDR (open GOP) | 57 B slices, MMCO 1 x15 | 765 list modifications |
| high-720p-cqm | 60 | 100, type 0 | 3 | 1 IDR, 1 non-IDR | strict B-pyramid, MMCO 1 x26 | 8x8 transform, asymmetric PPS scaling matrix |
| high-1080p | 120 | 100, type 0 | 4 | 2 IDR, 1 non-IDR | 348 B slices, MMCO 1 x52 | JVT default matrices, implicit weighted B, direct auto |
| high-1080p-deep | 60 | 100, type 0 | 1 | 1 IDR | 8 B-frames, 16 refs (DPB 16) | |
| high-1080p-fakeint | 30 | 100, type 0 | 1 | 1 IDR | 19 B slices | `frame_mbs_only_flag` 0, all frames: refused |

The results are those of "Crafted streams": 62 of 62 on both AMD GPUs
(2026-10-24), every frame of every stream matching in every form, and the
interlaced streams refused with "interlaced: the SPS's
frame_mbs_only_flag is 0". high-1080p holds a PPS's lists to fall back by
rule A where the SPS has no matrix (7.4.2.2): by rule B, JVT's defaults
become flat lists and none of its frames matches.

### AV1 on the AMD GPUs

`tools/av1-check.sh [render node] [stream...]`, from the checkout, as
`h264-check.sh` is run. Streams, references and logs go to
`build/av1-check/` (or `RVGV_AV1_OUT`), reused between runs and never
committed.

**The oracles**: aomenc (libaom 3.15.1), SvtAv1EncApp 4.2.0 and rav1e
0.8.1 write the streams from ffmpeg n9.0.2's `testsrc2` (with noise, and
moved under the frame for global motion), `mandelbrot` and nothing else,
and `tools/av1-svc.c`, built by the script against libaom's headers,
writes the scalable ones through
libaom's SVC controls, which no aomenc option reaches; ffmpeg decodes
each with libdav1d and with libaom (`-c:v libdav1d|libaom-av1
-noautoscale -pix_fmt yuv420p|yuv420p10le|gray|gray10le -f framemd5`).
`av1-framemd5 --planar` hashes our frames in that planar layout, so that
ffmpeg converts nothing. Two of ffmpeg's conversions are no reference,
measured 2026-10-24:
- to `p010le`, it gets an odd width's last Cb/Cr pair wrong: our frames
  of aom-641x361-10bit equalled libdav1d's `yuv420p10le` output sample for
  sample, and ffmpeg's P010 of that output differed in chroma column 320
  of every row;
- from `gray` to `nv12`, it rescales the luma as full range (0..229
  became 16..213).

So the NV12 and P010 references `host-session-check.sh` takes, as the
guest's buffers hold its frames, are libdav1d's own frames packed by the
conformance suite's `planar-packed`
(`crates/reims-vgpu-video-conformance/examples/planar-packed.rs`), each at
the size `ffprobe` gives it, with no conversion of ffmpeg's: `lib.sh`'s
`guest_frames` writes them to `NAME.libdav1d.guest` for every colour
stream, once the planar frames' MD5s equal the `framemd5`'s, so that each
frame was cut where ffmpeg's was.

Each stream is decoded from its IVF file (each frame a temporal unit with
its own sequence headers) and from its MP4 (the `av1C` record and the
samples, as VideoToolbox hands them over), and each frame's MD5 must
equal libdav1d's and libaom's. Then: strides padded by 192 bytes, a caller
holding 8 frames, references reset at a key frame mid-stream, three
mid-stream sequence changes in one IVF file (720p to 1080p, 8 to 10 bits,
to film grain), and two streams that must be refused (4:4:4, 12 bits).

What the streams hold, as `av1-info` reads them (frames decoded; hidden
frames are decoded but not shown):

| streams | bits | size | what they cover |
|---|---|---|---|
| aom-720p, aom-720p-10bit | 8, 10 | 1280x720 | 60 shown; 26 and 27 hidden altrefs shown later by `show_existing_frame`; CDEF, delta q |
| aom-1080p-tiles, -10bit | 8, 10 | 1920x1080 | 4x2 tiles in 3 tile groups; 2x4 in 2 |
| aom-720p-nofilters | 8 | 1280x720 | CDEF and loop restoration off |
| aom-480p-lr, -10bit | 8, 10 | 854x480 | loop restoration on (3 and 1 frames) |
| aom-720p-superres, -random | 8, 10 | 853 to 1280 wide, upscaled to 1280 | superres, fixed and random denominators |
| aom-641x361, -10bit, aom-63x65, rav1e-643x363 | 8, 10 | odd | odd widths and heights; rav1e-643x363's render size 644x363 differs |
| aom-720p-sframe, rav1e-720p-sframe | 8 | 1280x720 | 2 and 5 switch frames |
| svt-720p-opengop | 8 | 1280x720 | an intra-only frame |
| aom-720p-resilient | 8 | 1280x720 | every frame error resilient |
| aom-360p-screen | 8 | 640x360 | screen content tools, intrabc |
| aom-360p-resize-random | 8 | 320x180 to 640x360 | frames of nine sizes, 29 predicting from references of another size |
| aom-720p-resize | 8 | 853x480 | every frame coded smaller than its render size, 1280x720 |
| aom-720p-deltaq | 8 | 1280x720 | segmentation, delta q, delta lf |
| aom-360p-lossless | 8 | 640x360 | coded lossless |
| aom-720p-mono, -10bit | 8, 10 | 1280x720 | monochrome (`--monochrome`: luma hashed, chroma checked mid-grey) |
| aom-grain-1 to -16, four at 10 bits | 8, 10 | 352x288 | each of libaom's 16 film grain test vectors |
| svt-720p-grain, -10bit, rav1e-720p-grain | 8, 10 | 1280x720 | film grain the encoders estimated |
| svt-1080p, -10bit, svt-720p-superres, svt-720p-bigmax | 8, 10 | 1920x1080, 1280x720 | `--enable-overlays`, 2x2 tiles, superres; a 16384x8704 largest frame size |
| svt-2160p, -10bit | 8, 10 | 3840x2160 | 4K |
| rav1e-720p, -10bit | 8, 10 | 1280x720 | rav1e's segmentation, 2x2 tiles |
| aom-720p-pad-under, -over | 8 | 1280x720 | aom-720p with its third sample padded to 2,093,056 and 2,101,248 bytes, either side of the guest's 0x200000 bytes a message |
| aom-720p-fwdkf | 8 | 1280x720 | forward key frames: 3 key frames, one hidden and shown later from its slot by `show_existing_frame` (key frame filtering off) |
| aom-360p-fwdkf-filtered | 8 | 640x360 | the same with key frame filtering on: forward key frames coded `showable_frame` 0 and shown, which libaom's decoder refuses ([bugs.md](bugs.md), U12); checked against libdav1d alone |
| aom-720p-kfflt2 | 8 | 1280x720 | 3 key frames all hidden (`decode Key show 0`), shown through inter overlay frames (key frame filtering 2); its MP4 is `av1-mp4`'s, since ffmpeg's stream copy wrote no sample |
| aom-720p-gm-zoom, aom-720p-gm-pan-10bit | 8, 10 | 1280x720 | global motion: 15 and 22 ROTZOOM models in 12 and 14 frames |
| svc-720p-l1t2, -l1t3 | 8 | 1280x720 | two and three temporal layers: operating points `0x103`, `0x101`; `0x107`, `0x103`, `0x101`; OBU extension headers |
| svc-720p-l2t1 | 8 | 640x360 under 1280x720 | two spatial layers, 120 frames in 60 units, the upper predicting from the lower at twice its size (60 scaled references) |
| svc-720p-l2t1q | 8 | 1280x720 | two spatial layers of one size |
| svc-720p-l3t3 | 8 | 320x180, 640x360, 1280x720 | three spatial and three temporal layers, nine operating points, 180 frames in 60 units |
| aom-360p-resize-569, -465-10bit | 8, 10 | 640x360 key frames, 569x320 and 465x262 between | frames smaller than the session's, each predicting from references of another size |

The padded pair is ours: `av1-pad` (`crates/reims-vgpu-video-conformance/examples/av1-pad.rs`)
ends the sample with an `OBU_PADDING` (AV1 5.3.1, `obu_type` 15), which a
decoder ignores, so aom-720p's references hold for both. On 2026-10-24
both matched on each AMD GPU, 60 of 60 frames, from the MP4.

`--enable-fwd-kf` with `--kf-max-dist` alone wrote no forward key frame;
with `--fwd-kf-dist=16` it did. A stream that must show something is held
to it by `covers` in the script, a pattern on `av1-info --frames`, so a
stream that stops covering its feature fails before any decode.

Of a unit of several spatial layers the decoder outputs the top layer's
frame, as Apple's host and ffmpeg's libdav1d and libaom do; the lower
layers are decoded as references ([devices.md](devices.md), "AV1"). No
stream here has a unit missing its top layer: av1-svc encodes with drops
off.

Results, on both AMD GPUs, Mesa 26.2.4:

| GPU | result |
|---|---|
| Navi 31, `renderD128` | 278 of 278 pass |
| Phoenix, `renderD129` | 278 of 278 pass |

The run is of 2026-10-25, with zero copy on ("Zero copy", "The proof").
Every frame of each stream was equal to libdav1d's and libaom's, from the
IVF file and from the MP4, but aom-360p-fwdkf-filtered's, equal to
libdav1d's alone, 60 of 60 from both forms. libdav1d and libaom agreed on
every stream, film grain included, so VCN's film grain is bit-exact too.
The checks of reference scale and shown frames ([hardening.md](hardening.md),
F35) refuse no stream the three encoders write, the superres and
hidden-altref streams included (2026-10-24). Through the host session
(`host-session-check.sh`), the 22 AV1 streams it takes passed 44 of 44
runs on each GPU (2026-10-24, and 2026-10-25 against libdav1d's frames
packed by `planar-packed`), the resized ones every frame at its own size.
`planar-packed`'s P010 differed from ffmpeg's in aom-641x361-10bit,
rav1e-643x363 and aom-360p-resize-465-10bit alone, and for the two resized
streams equalled the decoder's own packed frames (2026-10-25). The two streams to
refuse failed with "the driver cannot: AV1 High profile (4:4:4)" and
"... AV1 Professional profile". svt-2160p-10bit's 20 frames took 1.52 s
on Navi 31 and 1.53 s on Phoenix, 8-bit svt-2160p's 0.97 s and 0.92 s:
the whole run, the session's setup with it, one frame at a time, each
read back (0.31 to 0.39 s in all) before the next is decoded (2026-10-24).

### VP9 on the AMD GPUs

`tools/vp9-check.sh [render node] [stream...]`, from the checkout.
Streams, references and logs go to `build/vp9-check/` (or
`RVGV_VP9_OUT`), reused between runs and never committed.

**The oracles**: vpxenc (libvpx 1.17.0) writes the streams from ffmpeg
n9.0.2's `testsrc2` (with noise), `mandelbrot` and nothing else; ffmpeg
decodes each with its own `vp9` and with `libvpx-vp9` (`-noautoscale
-fps_mode passthrough -pix_fmt yuv420p|yuv420p10le -f framemd5`).
`vp9-framemd5` hashes our frames in that planar layout, and with
`--packed` as NV12 or P010, compared at every size with ffmpeg's `vp9`
decode packed by the conformance suite's `planar-packed`
(`NAME.vp9.guest`, as for AV1: "AV1 on the AMD GPUs"), since ffmpeg's own
P010 of an odd width is wrong in its last chroma pair. Apple's decoder on
the M3 Max (macOS 27.0.1) agreed with ours there, 30 of 30 frames of
p2-643x363 (2026-10-24).

Two things of vpxenc's, measured 2026-10-24:
- its frames are frame-parallel unless told `--frame-parallel=0`, so no
  probability adapts backward; the streams turn it off, but for the
  `-parallel` and `-resilient` ones;
- `--resize-width` and `--resize-height` changed no frame's size in good
  mode, two-pass VBR included. Dynamic resize under a starved CBR does.

**What libvpx does not write**, `examples/vp9-craft` writes into its
streams, rewriting only uncompressed headers so that the tiles parse as
they did: a key frame made a hidden intra-only frame (refreshing every
slot, resetting every context) shown at once from slot 0, in a
superframe; hidden frames shown again later with `show_existing_frame`;
and segment 0's quantizer and loop filter features, absolute or delta,
with loop filter deltas, levels and sharpness that change every frame;
and a render size of its own on every frame (`render-size:WxH`), which
changes no decoded sample: p0-720p with render sizes of 1920x1080,
1280x704 and 640x360 decoded to p0-720p's 60 frames, at 1280x720, in
ffmpeg's `vp9` and in libvpx, with no error (2026-10-25).
The intra-only mode needs the next frame not to adapt backward, since
libvpx and ffmpeg adapt faster after a key frame, and refuses a stream
that does.

**Segments held to a reference or skipped** libvpx writes only from an
ROI map (`VP9E_SET_ROI_MAP`), which no vpxenc option sets, so
`tools/vp9-segments.c`, our own encoder on libvpx's API, writes them; the
check builds it with `cc` against the host's libvpx headers, and it is
linked into nothing. Each frame gets a map of moving diagonal bands, and its
mode says what each band's segment does. Measured with `vp9-info` on
libvpx 1.17.0 (2026-10-24):
- libvpx applies an ROI map to inter frames only, in realtime mode;
- some tables it applied only between the second golden refresh and the
  next key frame (at 640x360, one skipped segment with no reference held
  anywhere, for one), so the modes are tables it applied to every inter
  frame, and the check fails a stream whose inter frames lost a feature
  its mode asks for;
- a skipped segment held to INTRA or GOLDEN made streams that libvpx's own
  decoder and ffmpeg's refuse ("Failed to decode tile data", "Invalid
  usage of segment feature on small blocks"), so no mode asks for one
  ([bugs.md](bugs.md), U4);
- GOLDEN is held on 52 of the 58 inter frames: libvpx drops it on those
  right after a golden refresh.

What the streams hold, as `vp9-info` reads them:

| streams | bits | size | what they cover |
|---|---|---|---|
| p0-720p, p2-720p | 8, 10 | 1280x720 | 60 shown, 4 hidden altrefs in superframes, 2 key frames, backward adaptation, 4 tile columns, switchable and fixed filters |
| p0-720p-arf6 | 8 | 1280x720 | libvpx's multi-layer altrefs: 28 hidden frames, 28 shown by `show_existing_frame` |
| p0-1080p-tiles, p2-1080p-tiles | 8, 10 | 1920x1080 | 4x4 and 2x2 tiles, rows and columns |
| p0-720p-resilient, p2-720p-resilient | 8, 10 | 1280x720 | every frame error resilient, no context refresh |
| p0-720p-parallel, p2-720p-parallel | 8, 10 | 1280x720 | frame-parallel, a key frame every 20 |
| p0-720p-aq1, -aq2, p0-720p-aq3, p2-720p-aq3 | 8, 10 | 1280x720 | segmentation with maps, temporal map updates (aq3), sharpness 4 (aq2) |
| p0-360p-lossless, p2-360p-lossless | 8, 10 | 640x360 | lossless |
| p0-720p-resize, p2-720p-resize | 8, 10 | 1280x720 then 960x540 | an inter frame resized, 68 frames with a reference of the other size, render size 1280x720 |
| p0-720p-resize-early | 8 | 1280x720 then 640x360 | the same from frame 1, 127 frames with a reference of the other size |
| p0-360p-screen | 8 | 640x360 | `--tune-content=screen` |
| p0-641x361, p2-643x363, p0-63x65 | 8, 10 | odd | odd widths and heights |
| p0-2160p, p2-2160p | 8, 10 | 3840x2160 | 4K, 8 and 4 tile columns |
| p0-720p-intra-only, p2-720p-intra-only | 8, 10 | 1280x720 | vp9-craft: an intra-only frame with `reset_frame_context` 3, its colour config coded in Profile 2 |
| p0-720p-show-existing, p2-720p-show-existing | 8, 10 | 1280x720 | vp9-craft: 4 hidden frames shown again |
| p0-720p-features, p2-720p-features, p0-720p-resilient-features, p0-641x361-features | 8, 10 | 1280x720, 641x361 | vp9-craft: segment 0's ALT_Q and ALT_L, absolute and delta, varying loop filter deltas, levels and sharpness |
| p0-720p-seg-ref | 8 | 1280x720 | vp9-segments: segments held to LAST, INTRA and GOLDEN, a map changing every frame, on all 58 inter frames |
| p0-720p-seg-skip | 8 | 1280x720 | vp9-segments: a segment skipped, one skipped and held to LAST, one with a quantizer delta |
| p0-720p-seg-mixed, p2-720p-seg-mixed | 8, 10 | 1280x720 | vp9-segments: eight segments, each reference alone and with a quantizer or loop filter delta |
| p0-720p-seg-mixed-skip, p2-720p-seg-mixed-skip | 8, 10 | 1280x720 | vp9-segments: eight segments, skip alone and with LAST, quantizer and loop filter deltas both ways |

Then: strides padded by 192 bytes, a caller holding 8 frames, references
reset at a key frame, three mid-stream switches in one IVF file (720p to
1080p, 8 to 10 bits, 10 to 8), and the three to refuse (Profile 1,
Profile 3, 12 bits), from IVF and MP4.

Not covered by any stream: a skipped segment held to INTRA or GOLDEN,
which libvpx cannot write decodably (above), and the bilinear filter. The
unit tests cover their parsing.

Results, 2026-10-24, on both AMD GPUs, Mesa 26.2.4 (and again on
2026-10-25, with zero copy on: "Zero copy", "The proof"):

| GPU | result |
|---|---|
| Navi 31, `renderD128` | 198 of 198 pass |
| Phoenix, `renderD129` | 198 of 198 pass |

Every frame of the 43 streams matched ffmpeg's vp9 and libvpx, from IVF
and MP4, and NV12 or P010 where even; the two references agreed on every
stream. On 2026-10-25, with the NV12 and P010 references packed by
`planar-packed`, 202 of 202 passed on each GPU: the NV12 or P010 of
p0-641x361, p2-643x363, p0-63x65 and p0-641x361-features, the odd sizes,
are checked too. The refusals said "unsupported: VP9 Profile 1 (8-bit 4:2:2, 4:4:0,
4:4:4)" (IVF, at the first frame) and "the driver cannot: VP9 Profile 1
(4:2:2, 4:4:0, 4:4:4)" (MP4, from the record), and the like for Profile 3
and 12 bits. p2-2160p's 20 frames took 1.65 s on Navi 31, read back one by
one; 720p ran at 70 to 300 fps.

Through the host session (`tools/host-session-check.sh`, "Through the
host session" below), twelve of the streams, each synchronously and
asynchronously: 24 of 24 on both GPUs; and p0-720p-seg-mixed and
p2-720p-seg-mixed-skip, 4 of 4 on both (2026-10-24). The resized streams
come out each frame at its own size, where Apple's decoder emits a frame of
another size with -12909 and no image ("The smallest sizes").

### Through the host session

`tools/host-session-check.sh [render node]`, after `h264-check.sh` has
made the streams: the suite's `framemd5` (`vaapi-conformance NODE
framemd5`) plays the guest's decoder against a `HostSession` on
`VaapiBackend`, message by message, and hashes each frame out of the
buffer it lent (rows padded 256 bytes, the conformance crate's
`ROW_PADDING`, then rounded up to `Dpba`'s `BytesPerRowAlignment`, 64).
Frames come out in decode order and are
compared with ffmpeg's as a set, decoded synchronously and with the
asynchronous flag. A frame the host emits with no image fails the stream,
whatever its status, but in the cases that expect such frames (the damaged
H.264 and the HEVC of 9 references), which count them by status.

Results, 2026-10-24, on both AMD GPUs, the eight streams above but
baseline-720p-poc1 and the interlaced one, each both ways:

| GPU | result | speed |
|---|---|---|
| Navi 31, `renderD128` | 16 of 16 pass, twice | 720p 143 to 171 fps, 1080p 94 to 102 |
| Phoenix, `renderD129` | 16 of 16 pass | 720p 161 to 164 fps, 1080p 103 to 111 |

Every frame matched, every buffer was released with a `rels` after its
frame (one held at a time), and none was held after `-dec`. The speeds
are the whole round trip, guest stand-in included, in single runs over
SSH.

### The smallest sizes

A script not in this repository decoded 47 streams of 10 frames through
core's host session, synchronously and asynchronously, on a backend's
`conformance` example: H.264 and HEVC from 16x16, VP9 and AV1
from 2x2, odd sizes, strips (16x128, 128x16) and 10-bit streams up to
144x144, made by x264, x265 (from raw frames, as it reads no Y4M under 64
a side), vpxenc and aomenc. Each passes decoded with every frame equal to
ffmpeg's software decode in NV12 or P010, or refused at `dsta` with
-12911; a session that starts and then fails its frames fails the check
([devices.md](devices.md), "Where we differ from Apple").

On both AMD GPUs (2026-10-25), 94 of 94 passed:
- H.264 and HEVC under 64 a side refused, the driver's smallest surface
  being 64x64; 64x64, 128x128 and 144x144 decoded;
- VP9 and AV1 from 16x16 decoded, 31x17, 32x32, 48x64 and 16x128 among
  them, and 2x2 and 8x8 refused, the driver's smallest being 16x16;
- 10-bit VP9 and AV1 of 34x64 and 64x64 decoded, and 16x16 and 32x64
  refused, 32 or fewer samples wide ([bugs.md](bugs.md), U15).

The same streams passed on NVDEC (the RTX 5080) and on the M3 Max's
VideoToolbox (macOS 27.0.1), each with its own smallest
([nvidia.md](nvidia.md), [videotoolbox.md](videotoolbox.md)).

`host-session-check.sh` and the conformance suite's `decode-check` compare
a VP9 stream that resizes whole, each frame at its own size:
p0-720p-resize and p2-720p-resize 240 of 240 frames, p0-720p-resize-early
150 of 150, on both GPUs, both ways (2026-10-25).

### Testing HEVC

**On any host**, the codec crate's unit tests (`cargo nextest run -p
reims-vgpu-video-codec`) test the HEVC layer, whose streams the tests
write from the standard's syntax (a 64x48 picture, CTB 16, cropped to
64x46). They cover what x265 does not write, the first three of which no
encoder the checks use writes (kvazaar's streams, below, reach dependent
segments and tiles on hardware, and hevc-craft's, "Crafted HEVC streams",
the first three):
- long-term pictures found by LSB and by full order count, with MSB
  cycles;
- list modification;
- a slice header's own short-term set predicted from the SPS's;
- dependent slice segments;
- order counts across the LSB wrap, `pic_output_flag` 0;
- a CRA's RASL pictures skipped and RADL ones decoded, and pictures
  refused after a reset until a random access point;
- tile sizes with uniform and explicit spacing;
- the refusals, and the `hvcC` record.

**On the hardware**, `tools/hevc-check.sh [render node]`, run as h264-check.sh
is (above), with x265 4.3 (8, 10 and 12 bits) and kvazaar 2.3.2 (8 bits)
for the streams and ffmpeg n9.0.2 for the references. It makes every
stream first, then decodes them:

```sh
crates/reims-vgpu-video-vaapi/tools/hevc-check.sh /dev/dri/renderD128
```

Each stream is decoded from its MP4 (`hvcC` and length-prefixed samples)
and as Annex B in display order, and from its MP4 in decode order as a
set, and every frame's MD5 over packed NV12 or P010 must match. Then:
- each open-GOP stream cut at its second CRA picture (`hevc-cut from`),
  as Annex B and as the MP4 from that sample (`--skip`), whose RASL
  pictures neither decoder outputs;
- the same CRA picture made a BLA one mid-stream (`hevc-cut bla`);
- 2- and 3-byte NAL lengths, strides padded 192 bytes, a caller holding
  24 frames;
- a switch from Main 720p to Main 10 1080p at an IDR picture, mid-stream;
- Main 4:2:2 10, Main 4:4:4 and Main 12, which must be refused;
- kvazaar's kvz-720p-tiles-narrow, whose tile column of 192 samples Main
  does not allow (H.265 A.3.2: 256 at least), which the PPS's bounds must
  refuse;
- kvazaar's kvz-720p-ref15, whose tenth picture names 9 current
  references ("The HEVC decoder", "What it refuses"): its first 9 frames
  must match, and that picture must be refused before it reaches the GPU.

What x265's streams hold, as `hevc-info` reads them (every stream has
`sign_data_hiding` on, and none has long-term pictures, list
modification, tiles, dependent segments or PCM: x265 writes none):

| stream | frames | profile, size, CTB | pictures | slices, WPP | other |
|---|---|---|---|---|---|
| main-720p | 120 | 1, 1280x720, 64 | 1 IDR, 2 CRA, 8 RASL | 1, WPP | B-pyramid, 4 refs, weighted P (23 tables) |
| main-1080p-slices | 60 | 1, 1920x1080, 64 | 1 IDR | 4, WPP | x265 needs WPP with slices |
| main-720p-closed | 90 | 1, mandelbrot, 64 | 3 IDR | 1, **no WPP** | closed GOP |
| main-720p-radl | 90 | 1, 64 | 3 IDR, 4 RADL | 1, WPP | |
| main-720p-cqm-default | 60 | 1, 64 | 1 IDR | 1, WPP | scaling lists on, **the defaults** |
| main-720p-cqm-file | 60 | 1, 32 | 1 IDR | 1, WPP | **asymmetric SPS lists**, some predicted |
| main-720p-fade | 120 | 1, 64 | 1 IDR | 1, WPP | brightness and contrast change; weighted P and B (119 tables) |
| main-1366x766-ctu16 | 60 | 1, 1366x766 (coded 1368x768), **16** | 1 IDR | 1, WPP | AMP, rect; not a CTB multiple |
| main-720p-deep | 90 | 1, 64 | 1 IDR | 1, WPP | 8 B-frames, 6 refs, DPB 7 |
| main-720p-temporal | 60 | 1, 64 | 1 IDR | 1, WPP | 43 pictures above sub-layer 0 |
| main-360p-lossless | 20 | 1, 640x360, 64 | 1 IDR | 1, WPP | `transquant_bypass` |
| main-4k | 30 | 1, 3840x2160, 64 | 1 IDR | 1, WPP | level 5 |
| mainstill-1080p | 1 | **3**, 1920x1080, 64 | 1 IDR | 1, WPP | Main Still Picture |
| main10-720p | 120 | 2, 64 | 1 IDR, 2 CRA, 7 RASL | 1, WPP | as main-720p |
| main10-1080p-slices-cqm | 60 | 2, mandelbrot, 64 | 1 IDR | 3, WPP | default lists, weighted B (177 tables) |
| main10-720p-fade | 90 | 2, 64 | 1 IDR | 1, WPP | weighted P and B |
| main10-1366x766-ctu16 | 45 | 2, **16** | 1 IDR | 1, WPP | not a CTB multiple |
| main10-4k | 30 | 2, 3840x2160, 64 | 1 IDR | 1, WPP | |

kvazaar's streams, for what x265 does not write. Each is Main at 8 bits,
CTB 64, from `testsrc2` with noise unless said, at the level its options
name, with an MD5 of every picture in its SEI (`--hash md5`). Every slice
header of a picture after the IDR one codes its own short-term set: the
SPS carries none (`hevc-info`'s `st-sets 0`), so none is predicted.

| stream | frames | size, level | tiles | segments a picture | pictures, references | other |
|---|---|---|---|---|---|---|
| kvz-720p-tiles | 60 | 1280x720, 4.1 | **3x3 uniform**, one slice, 480 entry points | 1 | 1 IDR, GOP 8 | `--preset medium` |
| kvz-720p-tiles-explicit | 60 | 1280x720, 4.1 | **4x3 explicit** (columns 256, 384, 320, 320; rows 128, 384, 208) | 1 | 1 IDR, GOP 8 | |
| kvz-1080p-tiles-slices | 30 | 1920x1080, 5.1 | 2x2, **a slice each** | 4 | 1 IDR, GOP 8 | |
| kvz-1366x766-tiles | 30 | 1366x766 (coded 1368x768), 4.1 | 2x3, a slice each | 6 | 1 IDR, GOP 8 | not a CTB multiple |
| kvz-4k-tiles-slices | 20 | 3840x2160, 5.1 | 4x4, a slice each | 16 | 1 IDR, GOP 8 | `mandelbrot`, `--preset ultrafast` |
| kvz-720p-dependent | 60 | 1280x720, 4.1 | none | 12: WPP rows, **11 dependent** | 1 IDR, GOP 8 | |
| kvz-1080p-dependent-cra | 70 | 1920x1080, 5.1 | none | 17, 16 dependent | 1 IDR, **2 CRA, 30 RASL**, GOP 16, DPB 16 | cut at its second CRA picture and made BLA there, as x265's open GOPs are |
| kvz-720p-lowdelay | 60 | 1280x720, 4.1 | none | 1 | 1 IDR, low-delay B (`lp-g8d4t2`), 4 references | |
| kvz-720p-ref8 | 30 | 1280x720, 4.1 | none | 1 | 1 IDR, **8 current references**, DPB 9 | the most va.h allows |
| kvz-720p-fields | 60 fields of 30 frames | **1280x360 fields** (`field_seq_flag`), 4.1 | none | 1 | 1 IDR, low-delay B (`lp-g4d3t1`) | top field first |
| kvz-720p-tools | 30 | 1280x720, 4.1 | none | 1 | 1 IDR, GOP 8 | `fade`; sign hiding, transform skip, AMP and SMP, variance-adaptive QP, deblocking offsets -3 and 2, band SAO, access unit delimiters |

kvazaar writes no long-term picture, list modification, PCM block or
predicted short-term set: it has no option for any, and no stream it wrote
held one ([bugs.md](bugs.md), T1); hevc-craft writes the first three into
copies of its streams and x265's ("Crafted HEVC streams", below). Its reconstruction (`--debug`) is the
second reference, as x265's is, and equalled ffmpeg's decode on every
stream but the fields, for which `--debug` writes other pictures
([bugs.md](bugs.md), U13);
there ffmpeg's check of each picture against kvazaar's MD5 SEI
(`-err_detect crccheck`) is the second, and found every picture's MD5 as
kvazaar wrote it. kvazaar took two things Main does not allow, with no
warning: the tile column of 192 samples (kvz-720p-tiles-narrow) and, with
`--ref 15`, pictures of up to 15 current references (kvz-720p-ref15).

Results, 2026-10-24, on both AMD GPUs, Mesa 26.2.4, 108 checks on each
GPU: x265's 68, and kvazaar's 40, its eleven streams each three ways, the
cut and the BLA splice, and the two refusals each as MP4 and Annex B (and
again on 2026-10-25, with zero copy on: "Zero copy", "The proof"):

| GPU | result |
|---|---|
| Navi 31, `renderD128` | 108 of 108 pass |
| Phoenix, `renderD129` | 108 of 108 pass |

Both scaling-list streams fail on every frame unless the lists go in
raster order ("The HEVC decoder", "Shape"). The parameter set ranges of
[hardening.md](hardening.md), F28 to F34, refuse no stream x265 writes,
the open-GOP CRAs, the cut and the mid-stream BLA among them (2026-10-24).

**ffmpeg's software decode is wrong on one stream.** On
main-1366x766-ctu16, ffmpeg n9.0.2's `-hwaccel none` decode differs from
frame 3 on, where x265's own reconstruction (`--recon`), our decode and
ffmpeg's `-hwaccel vaapi` decode on both GPUs all agree. Found by
narrowing, 2026-10-24, with 30-frame 1366x766 streams:
- it is the same with `-threads 1`, frame or slice threads, and
  `-cpuflags 0`, so it is ffmpeg's C code;
- x265 with `--ctu 16` makes it at 1366x766 and at 1280x720, and not
  with `--ctu 32` or 64;
- it goes away with `--no-sao`, with `--no-deblock`, or with
  `--max-tu-size 8`, and stays with `--no-tmvp`, `--no-weightp`,
  `--no-wpp` or `--min-cu-size 16`.

So the check takes x265's reconstruction as a second reference, says
when ffmpeg's differs from it, and checks against the reconstruction
then. It did so for this stream alone.

**Crafted HEVC streams.** `examples/hevc-craft.rs` (ours) rewrites an x265
or kvazaar Annex B stream with what neither writes, every picture decoding
as the source's. A slice segment header ends byte-aligned before the slice
data (7.3.6.1), so each header is written again at its new length, its
slice data after it byte for byte, entry points and all. The edits:
- `--lists`: `lists_modification_present_flag` in every PPS, and in each P
  or B header whose picture predicts from more than one picture a
  `ref_pic_lists_modification()` that leaves the lists as they were:
  `list_entry_lX[i]` is `i % NumPicTotalCurr`, since `RefPicListTempX`
  repeats its pictures to its length (8-8, 8-10). The flags turn from
  header to header: both lists, list 0 alone, list 1 alone.
- `--sets sps`: the SPS carries the short-term sets the headers use, each
  after the first predicted from the one before where a `deltaRps` gives
  it exactly (7.4.8), and the headers name them. `--sets slice`: the same
  SPS, and each header codes its own set, predicted from the nearest SPS
  set that gives it (`delta_idx_minus1`).
- `--long-term`: a reference becomes long-term from the first picture
  after which that changes no decode, to the end of its life. Long-term
  marking changes motion vector prediction (8.5.3.2.7 to 8.5.3.2.9) and is
  never undone (8.3.2), so a picture is made long-term only where it is
  in none of the lists, or is the one picture of every list with temporal
  motion vector prediction off in the SPS. It goes last in
  `RefPicListTempX` then, and a list modification puts the lists back.
  Each entry names one of the SPS's `lt_ref_pic_poc_lsb_sps`
  (`lt_idx_sps`) or codes its LSBs, every other one with its MSB cycle,
  and each where another picture before it in the coded video sequence
  shares its LSBs (7.4.7.1).

x265's and kvazaar's B-frame streams drop a picture from their sets soon
after the last list that holds it, so they give a long-term picture or
two. With
one reference, no B-frames and no temporal prediction, every reference
is long-term: x265's main-720p-ref1 (`--ref 1 --bframes 0
--no-temporal-mvp`) keeps the two pictures before each P picture, its one
list the nearer, and kvazaar's kvz-1080p-ref1 (`--gop 0 --ref 1
--no-tmvp`) the one before each B picture, in both its lists.

`--check` decodes the source and the output through the codec crate's
decoder over a stand-in accelerator. It fails unless every picture has the
source's order count, reference picture set (each picture's order count,
set and marking; one made long-term may move from `StCurrBefore` or
`StCurrAfter` to `LtCurr`, nothing else) and lists, as order counts, and
they come out in the source's order. `hevc-check.sh` crafts each stream
again every run, from its source as made, and after `--check` the decodes
of ffmpeg and of libde265 (dec265 1.1.3) must equal their decodes of the
source; then it is checked as every stream is, three ways, against its
source's reference. What each holds, as hevc-craft wrote it:

| stream | source | edits | what its headers hold |
|---|---|---|---|
| main-720p-lists | main-720p | `--lists` | list 0 modified in 76 headers, list 1 in 63 |
| main-720p-sets-sps | main-720p | `--sets sps` | 25 SPS sets, 22 predicted; 119 headers name one |
| main-720p-sets-slice | main-720p | `--sets slice` | the same SPS; 119 headers predict their own from it |
| main-720p-closed-all | main-720p-closed | `--lists --sets slice --long-term` | 13 SPS sets, 12 predicted; 87 headers predict their own; lists modified in 56 and 42; 1 long-term picture |
| main-1080p-slices-lists-sps | main-1080p-slices | `--lists --sets sps` | 14 SPS sets, 13 predicted; 236 headers, 4 a picture, name one; lists modified in 154 and 117 |
| main-1080p-slices-all | main-1080p-slices | `--lists --sets slice --long-term` | as above, each header predicting its own; no long-term picture |
| main10-720p-all | main10-720p | `--lists --sets slice --long-term` | 32 SPS sets, 29 predicted; 119 headers predict their own; lists modified in 75 and 64; 2 long-term pictures |
| main-720p-ref1-long-term | main-720p-ref1 | `--long-term` | 117 long-term pictures in 59, all predicted from, 58 with an MSB cycle, 16 naming the SPS's; list 0 put back in 22 |
| main-720p-ref1-all | main-720p-ref1 | `--lists --sets sps --long-term` | as above, list 0 modified in 46 headers; the SPS's one set, empty, named |
| kvz-1080p-dependent-cra-lists | kvz-1080p-dependent-cra | `--lists` | lists modified in 43 and 42 headers; open GOP, 16 dependent segments a picture |
| kvz-1080p-dependent-cra-all | kvz-1080p-dependent-cra | `--lists --sets slice --long-term` | 26 SPS sets, 25 predicted; 69 headers predict their own; 2 long-term pictures |
| kvz-720p-lowdelay-lists-sps | kvz-720p-lowdelay | `--lists --sets sps` | 27 SPS sets, 26 predicted; 59 headers name one; lists modified in 38 and 38 |
| kvz-720p-lowdelay-all | kvz-720p-lowdelay | `--lists --sets slice --long-term` | as above, each header predicting its own; no long-term picture |
| kvz-1080p-ref1-long-term | kvz-1080p-ref1 | `--long-term --sets slice` | 29 long-term pictures, each the one picture of a B picture's lists, 21 with an MSB cycle, 15 naming the SPS's |

Every MSB cycle written is 0, each stream being shorter than its
`MaxPicOrderCntLsb` of 256, and the only list put back in another order is
list 0 of P slices.

Results, 2026-10-25, on both AMD GPUs, Mesa 26.2.4, with the streams of a
run of the tree before hevc-craft on the same day kept: `hevc-check.sh`
passed 156 of 156 on each GPU, the 108 checks above with the verdicts that
run gave them (108 of 108 each), and 48 more, main-720p-ref1's and
kvz-1080p-ref1's three each and the fourteen crafted streams' three each.
Every crafted stream's frames were its source's on both GPUs, and so were
ffmpeg n9.0.2's and libde265's decodes of it. libde265 decodes
main-1080p-slices itself otherwise than x265, ffmpeg and both GPUs, every
frame, with threads and without, so its two crafted streams carry a NOTE
line, and libde265 decodes them as it decodes their source. On an RTX
5080 Laptop GPU's NVDEC and the M3 Max's VideoToolbox, through the host
session both ways, every crafted stream's frames were its source's too
(2026-10-25; [nvidia.md](nvidia.md), "Decode";
[videotoolbox.md](videotoolbox.md), "Checks").

**Through the host session**, `tools/host-session-check.sh` runs ten of
the streams above (Main and Main 10 at 720p, 1080p and 4K, the asymmetric
lists, the fades and CTB 16) as well as H.264's, synchronously and
asynchronously: 36 of 36 pass on each GPU (2026-10-24). The Main 10
sessions' `Dpba` asks for `x420`, and their buffers are hashed as P010.
It runs kvazaar's eleven streams too, and kvz-720p-ref15, whose first 9
frames must come out, its tenth with -12910 and no image and the 20 after
it with -12909: with x265's ten, 44 of 44 passed on each GPU
(2026-10-24). The fields' `Dpba` asks for 1280x360, as Apple's host
does ([devices.md](devices.md), "HEVC"). With hevc-craft's fourteen
streams, 72 of 72 passed on each GPU (2026-10-25), the 44 as before.

**AV1 through the host session**: `host-session-check.sh` runs ten
of av1-check.sh's even-sized colour streams (aom-720p, -10bit,
aom-1080p-tiles, aom-720p-superres, aom-720p-sframe, svt-720p-grain,
-10bit, svt-720p-bigmax, svt-2160p-10bit, rav1e-720p-sframe), each both
ways, against libdav1d's NV12 or P010 decode. On 2026-10-24 all 20 passed
on each GPU, beside the H.264 and HEVC streams (56 of 56 each), and
`Dpba` asked for `x420` for the 10-bit streams and `420v` for the 8-bit
ones. It runs the padded pair too, whose larger sample its stand-in guest
cuts at 0x200000 bytes as the guest does, so -over's arrives in two
`dfrm`s: 4 of 4 passed on each GPU (2026-10-24).

## Limits and what is next

- Interlaced video (field pictures, MBAFF) is refused. Supporting it
  means field-pair DPB entries, field lists (8.2.4.2.2, 8.2.4.2.5) and
  field POCs, and VA's field flags.
- Long-term references, MMCO 2 to 6, operation 5 and frame number gaps,
  which x264 does not write, reach the hardware only in `h264-craft`'s
  rewrites of x264's streams ("Crafted streams").
- A sample must be one access unit. Annex B streams are split with
  `h264::nal::access_units`, which does not follow arbitrary slice order.
- Through `H264Decoder` and the other decoders, a frame's readback waits
  for its own decode. Core's host session decodes ahead
  ([core.md](core.md), "Decoding ahead"), the copy waiting for its
  processing run outside the driver's lock ("Zero copy", "The driver's
  lock"), so the next decode does not wait behind it.
  Where zero copy declines a buffer, readback is a copy, since radeonsi
  offers no `USER_PTR` surfaces ("The AMD drivers").
- No concealment of damage: a sample with an error decodes nothing, and
  the pictures after it are refused until an I picture decodes ("The H.264
  decoder", "What it refuses"). Only a frame lost to a `frame_num` gap is
  concealed, with a copy of the last reference.
- `H264Decoder` takes an `avcC` with three-byte NAL lengths, where
  ISO/IEC 14496-15 allows 1, 2 and 4; the host session refuses such a
  session at `dsta` with -12910, as Apple's host does ("Crafted streams").
- HEVC long-term pictures, list modification, a slice header's set
  predicted from the SPS's, and PCM are checked by unit tests only: neither
  x265 nor kvazaar writes them ([bugs.md](bugs.md), T1). Like H.264's
  lists and weights, they may also be what VCN reads from the slice header
  itself.
- HEVC's field signalling (`field_seq_flag`) is not looked at: each
  field is a picture of its own, and VA-API's HEVC has no field flags.
  kvazaar's kvz-720p-fields decodes so, every field matching ("Testing
  HEVC").
- The encoders are H.264's and HEVC's alone ("The encoders"): radeonsi's
  AV1 encoder (Profile 0, "The AMD drivers") has no backend here, and
  radeonsi has no VP9 encoder.
