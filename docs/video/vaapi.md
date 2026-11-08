# VA-API

`crates/reims-vgpu-video`'s `vaapi` module: the host's H.264 and HEVC
decoders and encoders through libva, with no FFmpeg, GStreamer or other codec layer in
between, and the reims-graphics device's video arm over them
([devices.md](devices.md)). This doc says what the module does, what the
drivers answer, and how the decoders and encoders are tested. Every
measured fact names its run: what, how, where and when.

The AMD runs were made on one Linux host with two GPUs on radeonsi (Mesa
26.2.4 unless said), "the AMD GPUs" in the section names: an RX 7900 XTX
(Navi 31, `renderD128`) and a Radeon 780M (Phoenix, `renderD129`). The
Intel runs were made on the Iris Xe of an i9-13900H on iHD. Paths are
`crates/reims-vgpu-video`'s unless they name another.

## What is here

| what | where |
|---|---|
| H.264 syntax and the decoding process above the slice data: NAL units, `avcC`, SPS/PPS/scaling lists, slice headers, order counts, the DPB's marking, lists and bumping. Plain Rust on every host, behind an `Accelerator` trait | `src/codec/h264.rs`, `src/codec/h264/`; the bumping process HEVC shares, `src/codec/dpb.rs` |
| HEVC syntax and the decoding process above the slice data: NAL units, `hvcC`, VPS/SPS/PPS, scaling lists, short- and long-term sets, slice segment headers, order counts, the DPB, lists and bumping. Plain Rust, behind its own `Accelerator` trait | `src/codec/hevc.rs`, `src/codec/hevc/` |
| The encoders' portable half: settings, GOP, parameter sets, slice headers, configuration records and rate control | `src/codec/encode.rs`, `src/codec/encode/` ("The encoders") |
| libva's binding, written by hand for the subset used | `src/vaapi/va/ffi.rs`, checked by `src/vaapi/va/ffi_layout.c`; video processing's `src/vaapi/va/vpp_ffi.rs`, checked by `src/vaapi/va/vpp_ffi_layout.c` |
| libva and libva-drm, loaded at run time, and a wrapper of each function the binding calls | `src/vaapi/va/libva.rs` |
| A display on a render node, its capabilities, configs and contexts | `src/vaapi/va/display.rs`, `src/vaapi/va/caps.rs` |
| The display's call thread and gate ("The display's calls") | `src/vaapi/va/calls.rs` |
| VA buffers and a picture's begin, render and end | `src/vaapi/va/buffer.rs` |
| The H.264 accelerator and `H264Decoder` | `src/vaapi/va/h264.rs` |
| The HEVC accelerator and `HevcDecoder` | `src/vaapi/va/hevc.rs` |
| NV12 and P010 surface pools, a frame on one, and the ways back into caller planes | `src/vaapi/va/surface.rs`, `src/vaapi/va/frame.rs`, `src/vaapi/va/readback.rs`, `src/vaapi/va/host_memory.rs` |
| What the driver's video processing writes into an imported surface, tried at each display's open ("The import check") | `src/vaapi/va/import_check.rs` |
| A frame written in place into a caller's pages ("Zero copy"), and the wait for a run's writes outside the driver's lock | `src/vaapi/va/zero_copy.rs`, `src/vaapi/va/fence.rs` |
| The bounded waits ("Waiting for the GPU"), and their tests on a held GPU | `src/vaapi/va/wait.rs`, `src/vaapi/va/wait/`, `src/vaapi/va/spin.rs` |
| The encoders: `VideoEncoder` and its accelerator | `src/vaapi/va/encoder.rs`, `src/vaapi/va/encoder/`, `src/vaapi/va/ffi/encode.rs` |
| The reims-graphics device's decode on VA-API: each picture written in place by the driver's video processing, copied by the runtime device's GPU, or copied by the CPU ("The reims-graphics device") | reims-vgpu's `src/graphics/vaapi.rs`, `src/graphics/vaapi/gpu_copy.rs`, `src/graphics/video_host.rs`; the decode surface's export, `src/vaapi/va/export.rs`; the Vulkan platform's copy, `os/linux/plane_copy.rs` |
| `vaapi-probe`, which stands in for `vainfo`; `vaapi-encode-probe`, the encoders' attributes | `examples/vaapi-probe.rs`, `examples/vaapi-encode-probe.rs` |
| H.264's hardware check and its tools | `tools/h264-check.sh`, `examples/h264-framemd5/`, `examples/h264-info.rs`, `examples/h264-poc1.rs`, `examples/h264-craft.rs` |
| HEVC's hardware check and its tools | `tools/hevc-check.sh`, `examples/hevc-framemd5.rs`, `examples/hevc-info.rs`, `examples/hevc-cut.rs`, `examples/hevc-craft.rs` |
| The encoders' tool | `examples/vaapi-encode/` |

`src/vaapi.rs` and `src/vaapi/` are Linux's alone (lib.rs picks them with `cfg`). Every
`unsafe` block of the module is there, each with its reason. On Linux the library
links neither `libva` nor `libva-drm`: it loads them when its first
display opens ("Loading libva").

### Loading libva

The binding's 37 functions come from `dlopen` and `dlsym`
(`src/vaapi/va/libva.rs`), as the nvidia crate loads NVIDIA's libraries, so a
QEMU built with the library starts on a host without libva.
- Each function has a wrapper of the same name and signature, which
  `ffi` and `ffi::encode` re-export, so no call site names the loaded
  table.
- `Display::open` loads `libva.so.2` and `libva-drm.so.2` first, once for
  the process, and keeps the answer. A host without them gets
  `Error::Load` from every open, with the loader's reason: "cannot load
  libva.so.2: ...". The graphics device's VA-API arm then declines, its
  line at info saying why, and the device's video goes to the next arm
  ([devices.md](devices.md), "Which hardware").
- Every other libva call needs a display that `open` made, so no wrapper
  runs before a load that succeeded. The libraries are never unloaded,
  since libva loads the driver, whose threads and state outlive a display.

### Asking the driver once

A display keeps the driver's first answer to every query a session makes:
the profiles, each one's entrypoints, every pair's config attributes and
surface attributes, and the image formats (`src/vaapi/va/caps.rs`, `Answers`).
`Display::open` asks them all on the thread that opens it
(`Display::prime`), for every pair the driver lists and for the union of
the attributes the decoders and the encoders ask (`ASKED_ATTRIBUTES`). A
session, on a thread of its own, then reads the opening thread's answers
and asks the driver nothing. A query that failed is not kept, and is asked
again where it is next needed.

That is for NVIDIA's VA-API driver, which answers some of them on the
opening thread alone ([bugs.md](bugs.md), U5). A probe on libva (not in
this repository) asked everything the crate asks on the opening thread and
on a second one, and made an H.264 session's config, surfaces and context
on the second:
- libva-nvidia-driver 0.0.18 on an RTX 5080 Laptop GPU (2026-10-24): 38
  answers differ, all of them `VAConfigAttribMaxPictureWidth` and `Height`
  (8192, 4096, 4080, 2048 and 32768x16384 on the opening thread; 0 on the
  second), on each of the 19 pairs that have them. The profiles, the
  entrypoints, the other 16 attributes, the surface attributes and the
  image formats are the same on both threads, and the session's objects
  are made on the second;
- radeonsi on both AMD GPUs (2026-10-24): no answer differs.

## The AMD drivers

Run 2026-10-24 on the AMD host (CachyOS, kernel 7.2.9-1-cachyos), with
`cargo run --example vaapi-probe -- /dev/dri/renderD128
/dev/dri/renderD129`. libva 2.24.1 (`pacman -Q`), VA-API 1.24 (the
driver's answer to `vaInitialize`).

| node | PCI | vendor string |
|---|---|---|
| `renderD128` | 1002:744c | Mesa Gallium driver 26.2.4-arch3.1 for AMD Radeon RX 7900 XTX (radeonsi, navi31, ACO, DRM 3.64, 7.2.9-1-cachyos) |
| `renderD129` | 1002:15bf | Mesa Gallium driver 26.2.4-arch3.1 for AMD Radeon 780M Graphics (radeonsi, phoenix, ACO, DRM 3.64, 7.2.9-1-cachyos) |

Both report the same for the codecs the crate drives:

| codec | decode (VLD) | encode (EncSlice) | max size |
|---|---|---|---|
| H.264 | ConstrainedBaseline, Main, High: YUV420, NV12 | the same three | 4096x4096 |
| HEVC | Main, Main10: YUV420, YUV420_10, NV12/P010 | Main, Main10 | 8192x4352 |

- Not offered: H.264 High10, 4:2:2 or 4:4:4; HEVC 4:2:2, 4:4:4, 12-bit or
  SCC.
- Encode attributes, the same for both encoders: rate control 0x416 (CBR,
  VBR, CQP, QVBR), 128 slices at most. H.264 takes packed headers 0x1f and
  references 0x10002 (two in list 0, one in list 1). HEVC takes references
  0x1, one in list 0 and none in list 1: no B-frames.
- Encode surfaces start at 128x128 (HEVC 130x128), decode surfaces at
  64x64.
- **Video processing**: `VAProfileNone` / `VAEntrypointVideoProc` on both,
  RT formats YUV420, 422, 444, 400, 420_10, 420_12, RGB32, RGB32_10;
  surfaces NV12, YV12, I420, P010, P016, P012, Y800, UYVY, YUY2, 422V,
  444P, RGBA, ABGR, BGRA, RGBX, BGRX, ARGB, AB30, AR30, XB30, XR30 up to
  16384x16384. `vaQueryVideoProcFilters` lists one filter: Deinterlacing.
  The module's runs (the staged copy back's, zero copy's) use the
  pipeline's own copy, no filter.
- **Image formats** (`vaQueryImageFormats`): NV12, P010, P012, P016, I420,
  YV12, YUYV, YUY2, UYVY, Y800, 444P, 422V, RGBP, BGRA, RGBA, ABGR, ARGB,
  BGRX, RGBX, AR30, AB30, XR30, XB30.
- **Surface memory types**: 0x68000001, that is VA, DRM_PRIME, DRM_PRIME_2
  and DRM_PRIME_3, for every config. **No `USER_PTR`**: radeonsi cannot
  wrap host memory as a surface directly. So a caller's pages reach the
  driver only as a dma-buf import ("Zero copy"), and are otherwise copied
  into ("The copy back").

## The import check

Both ways the crate has a driver's video processing write memory imported
as a linear surface, the staged copy back and zero copy, rest on what the
driver does with such an import, and a driver may take the import and its
run, answer success, and write the wrong bytes or none ([bugs.md](bugs.md),
U16, U26). So each display learns it as it opens, on the opening thread,
from a trial run of each of the decoders' formats, NV12 and P010
(`src/vaapi/va/import_check.rs`, `learn`):
- a picture holding every sample value of its format, 328x64, is put
  into a surface of the decoders' kind by `vaPutImage`;
- one video-processing run copies it into memory of the check's own,
  imported as zero copy imports a caller's pages (one dma-buf of both
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
back"). The graphics arm logs both answers as its display opens, as `in
place: NV12 Ok(ImportWrite { alignment: 256, whole_rows: false }), P010
...`.

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
every pitch a caller's planes may have, through each layout the check can
answer, and fails where a layout the answer lends comes out wrong. It
passed on the three radeonsi GPUs and the Iris Xe (2026-11-07).

## The copy back

Each decoded frame the device does not write in place is copied out of
the driver's surface into the caller's planes. A decode surface is in
video memory and tiled on both AMD GPUs, so the CPU cannot read it as it
is laid out, and reads it uncached over PCIe. The pool's surfaces come
back one of four ways (`src/vaapi/va/surface.rs`, `Readback`;
`src/vaapi/va/readback.rs`):

- **Staged**, the default. The pool makes a memfd of its own, sealed
  against shrinking, makes it a dma-buf with udmabuf, maps it
  (`src/vaapi/va/host_memory.rs`), and imports the dma-buf as a linear VA
  surface of the pool's format (`VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2`,
  one object, rows rounded up to 256 bytes, each plane at a 4096 boundary
  that is also whole rows of the luma's pitch from the first byte,
  `HostFormat`'s layout), with a video-processing context (`VAProfileNone`,
  `VAEntrypointVideoProc`) on it. Each frame: the decode surface's sync
  (the *wait*), one processing run into the import, a `poll` on the
  dma-buf for its writes and its sync (the *transfer*: the GPU writes
  system memory; "The driver's lock", "Waiting for the GPU"), then a copy
  of the cropped rectangle out of cached memory into the caller's planes
  (the *copy*). A driver copies back so only where the import check passed
  on it at its display's open ("The import check", `Display::writes_imports`
  in `src/vaapi/va/display.rs`), since a driver says nothing of what it gets
  wrong here: NVIDIA's VA-API driver (libva-nvidia-driver 0.0.18 on the
  RTX 5080, 2026-10-25) took the import and the run, answered success,
  and every frame that came back was wrong ([bugs.md](bugs.md), U16).
  Where the check passed on neither of a driver's formats, its frames come
  back through `vaGetImage`, with a line in the log saying why. The
  planes' layout serves both ways the check passes, planes at an alignment
  or in whole rows, so iHD, which writes whole rows alone, copies back so
  too.
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
derived image's destroy or the sync's end as the read back drops. So no VA
call runs inside the caller's copy.

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

**Per frame** (the framemd5 examples, in display order, 150 frames each,
2026-10-24):

| stream, GPU | `vaGetImage`: transfer, copy | staged: transfer, copy |
|---|---|---|
| 4K HEVC Main, Navi 31 | 14.08, 1.39 ms | 2.89, 1.58 ms |
| 4K HEVC Main 10, Navi 31 | 14.81, 2.96 ms | 5.41, 3.20 ms |
| 1080p H.264 High, Navi 31 | 2.01, 0.10 ms | 1.06, 0.18 ms |
| 4K HEVC Main, Phoenix | 9.49, 1.32 ms | 1.99, 1.53 ms |
| 4K HEVC Main 10, Phoenix | 8.91, 2.80 ms | 3.93, 3.15 ms |

The streams: 150 frames, x265 `--preset faster --bframes 3` at 20 Mbit/s
and x264 High at 8 Mbit/s, from `testsrc2` with noise. Navi 31 is on a
PCIe 4.0 x4 link (downgraded): its staged transfer of a 4K P010 frame,
24.9 MB in 5.4 ms, is about 4.6 GB/s, near that link's practical limit.

## Zero copy

Where the driver can take the caller's pages, a decoded frame is written
into them by the staged copy back's processing run, aimed at those pages
instead of ours, and the CPU copies nothing (`src/vaapi/va/zero_copy.rs`,
`Frame::write_to_guest`). The caller hands the pages as a `GuestTarget`
over core's `DeviceMemory` ([core.md](core.md), "A guest surface's
memory"). The graphics device's are its output planes' pages in the
guest's RAM (reims-vgpu's `PlanePages`), whose dma-buf reims-vgpu makes of
ranges of their bytes: a udmabuf of the memfd behind the guest's RAM, the
pages in the order the bytes lie in (reims-vgpu's
`platform/vulkan/dma_buf.rs`, over
`crates/reims-vgpu-vulkan/src/os/linux/device_lend.rs`,
`dma_buf_of_host`).

The crate imports the pages once per mapping, frame size and format, in
strips of the frame's rows ("What udmabuf takes", below): each strip an
even number of rows, a linear VA surface over one dma-buf of its luma and
chroma rows, at the caller's own offsets and pitches (`DRM_PRIME_2`, one
object of two planes, `DRM_FORMAT_MOD_LINEAR`), as the import check
imports its memory. The strips share one video-processing context, and
the import is kept by the mapping's identity until the mapping goes (at
most 96 at once, `MOST_IMPORTS`, the one used longest ago dropped first).
Each frame is then one run a strip, from the decode surface's rows into
the strip, BT.709 on both sides as the staged run's, a wait for the runs'
writes ("The driver's lock", below), and their syncs.

**What radeonsi takes**, measured with a probe on libva (not in this
repository) on both AMD GPUs (Mesa 26.2.4, 2026-10-24), each import
checked byte for byte after a run, padding included:
- rows of a multiple of 256 bytes: 1920-byte rows were refused
  (`VA_STATUS_ERROR_ALLOCATION_FAILED`), 1280, 3840 and 7680 taken;
- planes 256 bytes apart from the dma-buf's first byte: a luma plane 64
  bytes into it was refused, and a chroma plane 256 bytes past a page
  boundary taken;
- an even size: at 641x361 every sample past the first chroma pair came
  out other than the probe expected, at 8 and 10 bits, though nothing was
  written outside the planes;
- nothing else: in every case taken, every byte of the buffer that is not
  one of the frame's samples, the rows' padding included, stayed as it
  was.

A driver is lent pages only where the import check passed on it, and only
at the layout it passed at (`Display::writes_imports`): rows and plane
starts a multiple of its alignment, and, where it writes whole rows alone
as iHD does, the luma plane at a page's first byte and the chroma plane
whole rows of the luma's pitch after it ([bugs.md](bugs.md), U26). Frames
of an odd width or height are declined on every driver. A declined frame
takes the runtime device's GPU copy, and the CPU's where that declines too
("The reims-graphics device"). The graphics device asks the guest for
planes at the check's alignment ([devices.md](devices.md), "How each arm
lands a picture"), but the frames VideoToolbox hands the guest's decoder
have CoreVideo's own layout, each plane 32 bytes into its page, which
neither driver's processing writes and the GPU's copy does.

**What udmabuf takes.** The kernel's `udmabuf` puts at most `list_limit`
runs of pages and `size_limit_mb` MiB in one dma-buf: 1,024 and 64 at its
stock limits, and nothing here asks a host to change them. Nearly every
page of the guest's RAM is a run of its own in the memfd (measured in a
guest's boot: 3,040 runs for a 4K NV12 buffer, 6,078 for P010), so one
dma-buf holds about 4 MiB of it. The host refuses a dma-buf past either
limit before it asks the kernel, with its count of runs and bytes and the
limits as `/sys/module/udmabuf/parameters` has them then (core's
`DmaBufError::TooLarge`, from the Vulkan platform's
`LendRefusal::TooLarge`), and the crate imports the pages in as many
strips as that count calls for: one, the whole frame, where each plane
fits in a dma-buf, else more, until every strip's planes fit, up to 64
strips (`MOST_STRIPS`) and none of fewer than two rows. At the stock
limits a 720p frame and a 1080p 10-bit one take one strip, a 4K NV12 one
two or three and a 4K P010 one four or five, by the planes' pitches. A
frame past 64 strips at the host's limits, which no size the decoders
make is at the stock ones, is declined and copied by the CPU, the reason
logged: `no dma-buf of the buffer in 64 strips or fewer: N pieces of 0x…
bytes do not fit udmabuf's list_limit 1024 and size_limit_mb 64`. Higher
limits give fewer strips and change nothing else.

**The driver's lock.** Mesa's VA frontend waits for a processing run in
`vaSyncSurface` with its display's lock held: it waits for the output
surface's `pipe_fence` under `drv->mutex`, where it waits for a decode's
fence under the context's lock alone (**read** in Mesa 26.2.4,
`src/gallium/frontends/va/surface.c`, `_vlVaSyncSurface`;
[bugs.md](bugs.md), U17), and `vaBeginPicture`, `vaRenderPicture` and
`vaEndPicture` each take that lock too (`picture.c`). So while a copy's
thread waited there for its run, the next decode on the display waited
behind it: a 4K frame's `decode` took 0.9 to 5.0 ms with decoding ahead,
against 0.3 to 1.0 ms one frame at a time. The crate waits for a run's
writes with no lock held first, by `poll` for writing on the dma-bufs the
run writes (`src/vaapi/va/fence.rs`): the kernel puts the run's fence on each
of them, so the poll returns once the run is done, and `vaSyncSurface`
then finds the fence signalled. Zero copy's runs and the staged copy
back's both wait so. Measured on both AMD GPUs (2026-10-25):
`vaSyncSurface` took 0.0 to 0.3 ms after the poll, and with decoding
ahead `decode` took 0.3 to 0.9 ms a 4K frame, zero copy on and off. A
driver that puts no fence on the dma-buf leaves the poll nothing to wait
for, and the sync's questions wait in its place. The poll, and the sync
after it, wait for all of one run's dma-bufs until the frame's deadline
("Waiting for the GPU").

**What it saves**, measured on both AMD GPUs (2026-10-25, udmabuf at its
stock limits, medians of three runs) through a decoding harness since
removed, which decoded 4K HEVC and AV1 at 8 and 10 bits into pages of its
own with zero copy on and off:
- the CPU's copy is gone: 1.2 to 2.3 ms of each 4K frame on Navi 31, 1.4
  to 3.1 on Phoenix, and the decoding threads used 0.4 to 1.4 ms of the
  CPU a frame on, against 1.3 to 3.4 ms off;
- one frame at a time, the device's write took less than the read back
  and the copy in six of eight streams, and 2.0 ms more in the worst;
- decoding ahead, the CPU path ran 0% to 12% faster than the device's
  writes: its copy runs on the CPU beside the GPU's next decode, and the
  device's write took the GPU 1.0 to 3.1 ms longer than the staged run.
  Both ran past 4K60;
- a strip has a cost of its own: 10-bit 4K one frame at a time went from
  50.9 to 49.4 fps (HEVC Main 10) cut into one to five strips, so the
  strips are as few as the limits allow.

## The reims-graphics device

The reims-graphics device's H.264 and HEVC decode runs on this crate
where its GPU is not one NVDEC opens: the device's video chooser
(reims-vgpu's `src/graphics/video_choice.rs`) tries NVDEC, then VA-API on
the render node of the runtime device's own GPU, whatever its vendor
(`host_gpus::render_node_of`). The arm
(`src/graphics/vaapi.rs`) runs the module's `H264Decoder` and
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
- **A picture lands** in the output's planes one of three ways, each
  where those before it refuse:
  1. by the driver's video processing, in place, into the planes' pages
     ("Zero copy"): where both planes lie in one memory a dma-buf is made
     of, and the import check's answer for the frame's format takes their
     layout;
  2. by the runtime device's GPU, into the buffer each plane lies in, the
     surface's pages or its device cache, at the plane's offset and rows:
     the decode surface's export as dma-bufs (`Frame::export`,
     `vaExportSurfaceHandle` with each plane a layer of its own, kept with
     the surface's pool), each plane imported once as an image of its
     format and DRM format modifier, and copied out with
     `vkCmdCopyImageToBuffer` on a queue of the session's copier
     (reims-vgpu's `src/graphics/vaapi/gpu_copy.rs`, the Vulkan
     platform's `PlaneCopier`). It writes the crop the CPU's copy writes,
     and nothing past it (`crop_extent`), once the decode is done, and the
     session signals the decode once the copy is done on the GPU. A device
     that lists none of the dma-buf image extensions, an export not of a
     layer a plane, or a plane in a buffer whose GPU writes do not reach
     its memory, declines it;
  3. by the CPU: the frame read back ("The copy back") and copied into the
     planes' host bytes, or into the session's staging where a plane lies
     in a device cache.

  Its outputs lie in device caches wherever the kext makes them: the arm
  lands pictures in them (`lands_in_device_caches`), so the device answers
  `RGFX_VIDEO_CAPS_OUTPUT_CACHED` on every host that makes caches, a GPU of
  its own memory or one that shares the CPU's ([devices.md](devices.md),
  "The flags"), and the guest's decoder asks its outputs cached. On a GPU
  that shares the CPU's memory, an output whose textures would not lie in
  place in its pages, or whose view has no room for its mappings, takes a
  cache, which the GPU's copy writes, its first plane on a multiple of 256
  bytes of the cache's memory; one whose textures lie in place stays in its
  pages.

  A session logs each reason its pictures take the third way once, as
  `a picture copied by the CPU: rows of 1920 bytes: the driver imports
  rows of a multiple of 256 bytes alone; and the GPU's copy: …` (`; and
  no GPU copy: …` where the device made the session no copier), its GPU
  copies at its end, as `a session's pictures copied by the GPU into their
  outputs: 300, 0.812 ms of GPU time each, 1.204 ms at most (300 timed)`,
  and its counts, as `a session ended after 300 decodes: landed 300 in the
  surface's pages, by the hardware; 0 skipped, 0 failed`. A picture that
  does not land fails its decode's command buffer ([devices.md](devices.md),
  "A decode").
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
CPU. The frames VideoToolbox hands the guest's decoder have CoreVideo's
own layout, each plane 32 bytes into its page and rows of the width
rounded up a little, 1376 bytes at 1366. Neither driver's video
processing writes that: radeonsi wants rows and planes 256 bytes apart,
iHD the luma plane at a page's first byte. The runtime device's GPU
copies them there instead (above).

**Measured** in cold boots of an x86-64 macOS 26.7.2 guest at 3648x2038
(2026-11-08), Safari playing 4K 60 fps HEVC clips, 8-bit then 10-bit, then
two and three at once, 20 s each, the arm with the GPU's copy and the arm
before it back to back on the same host and GPU:

| GPU, drivers | pictures copied by the CPU | by the GPU, GPU time each | presented fps, one clip | two at once | three at once (frames dropped) |
|---|---|---|---|---|---|
| Radeon 780M, radeonsi and RADV (Mesa 26.2.6), before | 9979 | none | 55.9, 55.4 | 53.0, 55.0 | 50.0, 50.3, 52.0 (10, 11, 5) |
| the same, with the GPU's copy | 0 | 10000, 1.01 ms (2.97 at most) | 56.2, 57.2 | 57.3, 53.2 | 53.2, 53.7, 52.5 (3, 2, 0) |
| Iris Xe, iHD 26.3.5 and ANV (Mesa 26.2.6), before | 9899 | none | 56.0, 54.3 | 52.5, 54.6 | 52.3, 51.7, 53.5 (71, 74, 1) |
| the same, with the GPU's copy | 0 | 10040, 0.52 ms (1.13 at most) | 56.8, 56.8 | 56.5, 56.3 | 53.8, 54.2, 56.4 (4, 3, 0) |

No boot logged a `reims-vgpu: fatal` line. The runtime's shadows of the
guest's textures over the planes are as they were either way: about 11 M
texels copied in a frame of one clip on the 780M, 0.3 M on the Iris.

**The GPU's copy against the CPU's**: the ignored test
`a_picture_copied_by_the_gpu_is_the_cpus_copy_byte_for_byte` (reims-vgpu's
`src/graphics/vaapi/tests.rs`) decodes the conformance scripts' Annex B
streams (`RVG_VA_STREAMS`, their `build/h264-check` and `build/hevc-check`,
`:`-separated), lands each picture both ways into CoreVideo's layout, 32
bytes into host pages a no-copy buffer lies over, at two pitches, and
compares every byte, the padding included. `RVG_VAAPI_NODE` names the GPU
the runtime's default device must be on. **Measured** (2026-11-08): 4849
pictures of 79 streams equal on the Radeon 780M, 4951 on the Iris Xe.

## Waiting for the GPU

Every wait of the module's on the GPU is bounded, by core's deadline,
`DEVICE_WAIT`, 5 s from a frame's first wait ([core.md](core.md), "Bounded
waits"), asked after every 100 µs for its first 50 ms and every
millisecond after (core's `device_wait`, `src/vaapi/va/wait.rs`). A frame's
waits share that one deadline (`wait::Deadline`): a decoder's call and
the read backs of the frames it gives, or an encoder's call, with every
wait among them for the display's calls ("The display's calls"), so that
a frame is answered within 5 s of its first wait, however many calls and
questions it makes.

| wait | asked with | in place of |
|---|---|---|
| a decode | `vaSyncSurface2` of its surface with a timeout of 0, until it answers other than `VA_STATUS_ERROR_TIMEDOUT` | `vaSyncSurface` |
| a processing run: the staged copy back's, zero copy's and the trial runs at open | for the copy back and zero copy, into host memory, the `poll` of the dma-bufs the run writes until the deadline ("The driver's lock"); then `vaSyncSurface2` of its target, as a decode's | the poll for a second at most, then `vaSyncSurface` |
| an encode | `vaSyncBuffer` of its coded buffer with a timeout of 0 | `vaSyncBuffer` with `VA_TIMEOUT_INFINITE` |
| a driver call that may wait for the GPU itself: `vaGetImage`, `vaMapBuffer` of a derived image, `vaSyncSurface` where the driver has no `vaSyncSurface2`, and a decoder's first `vaEndPicture` | its answer, the call made on the display's call thread ("The display's calls") | the call on the caller's thread |
| every other driver call, while one of those is in the driver | the display's gate, which the call thread holds exclusively meanwhile | the driver's own lock, which the call that waits holds as it does |

A question with a timeout of 0 returns at once while the work runs
(below), so no question waits in the driver, where radeonsi holds the
display's lock while `vaSyncSurface` waits for a processing run
([bugs.md](bugs.md), U17).

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
(`src/vaapi/va/spin.rs`):

| call | the graphics ring held | the compute rings held |
|---|---|---|
| `vaGetImage` | waits | waits |
| `vaMapBuffer` of a derived image of a processing run's NV12 target | waits | (its run stalls first) |
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
B92, U23). A decoder's first `vaEndPicture` submits the job that begins
the decoder's session and waits for it (**read** in Mesa 26.2.4,
`si_video_dec.c`; [bugs.md](bugs.md), U24), so it waits as long as that
job does, behind a destroy's update as the copies are (B92).

So the crate makes three kinds of call (`src/vaapi/va/calls.rs`):
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
the gate as long as the driver held its lock for it, and every other call
on the display waits for the gate meanwhile. So the crate makes such a
call on radeonsi only where it cannot help it: the decoders' frames are
written into host memory of ours and read there ("The copy back"), or
into the caller's pages ("Zero copy"), and `vaGetImage` is the fallback
alone. **Measured** on the RX 7900 XTX and the Radeon 780M (2026-10-26),
a 4K H.264 decode on a display where another thread read 4K pictures
back through derived maps waited for the gate 330 to 393 times a run of
720 `decode` calls, behind as many maps of 1.1 to 8.7 ms at the mean and
1.3 to 34 ms at the 99th percentile; with those reads moved to host memory
of ours, the call thread made no call and the decode never waited.

**Past the deadline** the frame fails as the module's malfunction
(`Error::Stalled`), which the graphics device answers as a decode that did
not land ([devices.md](devices.md), "A decode"). A frame whose zero copy
runs passed it fails so too, and is not declined to the CPU's copy, which
would write the caller's pages beside the runs. What failed then fails
each later call at once, the same way: a decoder's pool, and an encoder.
What the GPU may still write is kept in core's list until the GPU is done,
asked after with the same questions ([core.md](core.md), "Bounded
waits"):
- a decoder's: its pool, every surface of it and the staged copy back's
  surface and memory, and its context once the decoder lets it go,
  destroyed first (`src/vaapi/va/surface.rs`); for zero copy, the pages' import
  and the caller's hold on them, so that the pages stay in use until each
  strip is done (`src/vaapi/va/zero_copy.rs`);
- an encoder's: its session, with the coded buffer, the context and both
  pools of surfaces, until the coded buffer is done
  (`src/vaapi/va/encoder/accelerator.rs`).

An answer that is an error keeps the work, and a GPU that never finishes
keeps it until the process ends.

**What the waits answer**, measured on the RX 7900 XTX and the Radeon 780M
(radeonsi, Mesa 26.2.4, libva 2.24.1, 2026-10-25), the questions asked
again and again before each of ffmpeg n9.0.2's own syncs by a library of
ours preloaded into it, as ffmpeg decoded a 4K HEVC stream of 150 frames,
scaled 60 of its frames to 1080p, and encoded 60 as H.264 and as HEVC at
20 Mbit/s:
- `vaSyncSurface2` and `vaSyncBuffer` with a timeout of 0 answered
  `VA_STATUS_ERROR_TIMEDOUT` while the work ran, then success, and
  ffmpeg's own sync after them returned in 1 µs at most. ffmpeg's frames
  came out the same as without the questions.
- From ffmpeg's sync on, the first decode was done in 3.3 ms on the 7900
  XTX and 1.6 ms on the 780M, and the 147 after it at the first question,
  ffmpeg syncing each frame late; a scaling run in 0.32 and 0.69 ms
  (median), 0.48 and 2.0 ms at most; an encode's coded buffer in 5.1 to
  14.3 ms (median), 26 ms at most.

On the same day, the bounded waits cost the framemd5 examples' 4K HEVC
decode nothing measurable: 37.8 against 37.7 fps on the RX 7900 XTX and
39.5 against 39.8 on the Radeon 780M, the tree before and after, nine runs
each, every frame matching its reference. A decode's wait ends up to one
pause, 100 µs, after the decode does.

**Its tests.** On the VA-API host, ignored elsewhere, on the render node
`RVG_VAAPI_NODE` names, one test at a time. In `src/vaapi/va/wait/gpu.rs`, each
asked after with no time left, or 4 ms for an encode, so that the work is
still running at its deadline, on the GPU's own work: a 4K H.264 decode
fails as stalled, the frame's later wait and the decoder's next decode
fail at once, and its pool, then its context, are kept until the GPU is
done and then let go; a decoded frame's staged copy back fails the same
way, with no other way back tried; and a 4K H.264 encode fails its
picture and the next frame at once, its session kept until the coded
buffer is done. In `src/vaapi/va/wait/held.rs`, with a ring held by the spin,
each frame's wait 50 ms: the `vaGetImage` way back, the concealment copy,
a wait in `vaSyncSurface` on a driver without `vaSyncSurface2`
(`vaSyncSurface2` answered as one answers), and an encode whose upload
maps its input surface each fail by the frame's deadline while the GPU is
still held, and the next at once; a new decoder's first decode, another
decoder let go with the graphics ring held, fails so, its next decode at
once and the display's other calls each by its own deadline, and a
decoder made once the spin is let go decodes. Once the spin is let go,
what was kept is let go and the display's calls work again. All passed on
both AMD GPUs (2026-10-26).

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

The VA accelerator (`src/vaapi/va/h264.rs`) turns that into libva's buffers:
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
by radeonsi; a driver that reads them is their hardware test.

### Surfaces and readback

- Each session makes a pool of NV12 surfaces at the coded size, sized
  DPB + 1 + 2 spare. They are the context's render targets.
- A `Frame` holds its surface until it is dropped; a caller that holds
  frames makes the pool grow, up to 48 surfaces, then `OutOfSurfaces`.
  Surfaces made after the context are decoded into fine: main-720p
  matched with the caller holding 24 frames.
- Readback waits for the decode ("Waiting for the GPU"), has the GPU copy
  the frame into host memory of the pool's, and copies the cropped
  rectangle row by row into the caller's planes at the caller's strides;
  where that cannot be set up, through `vaGetImage`, then `vaDeriveImage`
  ("The copy back").

### What it refuses

Before anything reaches the GPU:
- with `h264::Error::Interlaced` ("interlaced: the SPS's
  frame_mbs_only_flag is 0"), any SPS with `frame_mbs_only_flag` 0,
  whether its pictures are fields, MBAFF frames or frames alone, x264's
  `--fake-interlaced` streams among them;
- with `Error::Bitstream(h264::Error::Unsupported(..))`, chroma other than
  4:2:0, bit depths other than 8, lossless macroblocks, and the Extended
  profile;
- the same, for FMO, data partitioning, and SP and SI slices.

Redundant pictures are dropped.

`H264Decoder::new` also checks the `avcC` record's first SPS, and the
driver's profiles and sizes, so that a session the driver cannot serve
fails when it is made. Through the graphics device, every refusal fails
the decode's command buffer ([devices.md](devices.md), "A decode").

A P or B slice is refused with `MissingReference` when it has nothing to
predict from: after starting or `reset()` at a non-I picture, or when its
list names a frame the gap process inferred where the SPS allows gaps.
Where it allows none, the gap is frames lost. Each lost frame is then a
copy of the last reference, made by `vaGetImage` and `vaPutImage` through
an image the pool keeps for it (`Accelerator::conceal`,
`Surface::copy_to`), so the frames after it decode.

After a sample that does not decode, each picture is still decoded, so
that its references go on. Each is refused with `h264::Error::Unrecovered`,
until an I picture, IDR or not, decodes.

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

`--check` decodes the MP4 through this module's decoder over a stand-in
accelerator. For each sample it prints what would reach the GPU, the
copies made for lost frames, and any error. A damaged stream goes to a
GPU only once this shows whole pictures of valid slices alone.
`h264-check.sh` builds the crafted streams from x264's one-reference
streams, so that each P slice keeps its one reference after the
markings. It checks each against ffmpeg, or against its source for a gap,
since ffmpeg leaves frames out after one ([bugs.md](bugs.md), U8). Run
2026-10-24 on both AMD GPUs, `h264-check.sh` passed 62 of 62. That covers
the long-term and MMCO streams, CABAC and CAVLC, three gap streams, 2- and
3-byte lengths, and drop20-ref1 (frame 20 lost: 59 frames, the 50 the loss
does not reach as the source's). The MBAFF and fake-interlaced streams
were refused.

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

The VA accelerator (`src/vaapi/va/hevc.rs`) fills, per picture, one
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
  `VA_RT_FORMAT_YUV420_10`): two planes of little-endian 16-bit samples,
  the 10 bits in each sample's top bits.
- Pools, holds and readback are H.264's (above), in either format:
  `Frame::bit_depth` says which, and `copy_p010_to` copies two bytes a
  sample. The conformance window crops in chroma units, so a 4:2:0 frame
  is always of even size.

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
data.

A picture whose current reference sets hold more than 8 pictures together
is refused by the VA accelerator with `Error::Unsupported`, before any of
it reaches the driver: va.h says `NumPocTotalCurr` "should be equal to or
smaller than 8", and radeonsi decoded such a picture wrongly with no error,
as ffmpeg's own VA-API decode did (kvz-720p-ref15 from its tenth picture
on, on the Radeon 780M, 2026-10-24; "Testing HEVC"). Each picture after
it is refused until an IRAP picture.

x265's streams of Main 4:2:2 10, Main 4:4:4 and Main 12 are refused
so. `HevcDecoder::new` checks the `hvcC` record's first SPS, the driver's
profile and its largest size, so that a session the driver cannot serve
fails when it is made.

## The encoders

H.264 (Constrained Baseline, Main, High) and HEVC (Main, Main 10) on
`VAEntrypointEncSlice`, with the parameter sets and slice headers ours,
handed to the driver as packed headers. The driver writes the slice data,
and on radeonsi the parameter sets as well (below).

The portable half is the codec module's (`src/codec/encode.rs`); the rest
is the `vaapi` module's.

| what | where |
|---|---|
| Settings, input frames, samples, profiles and rate control. Plain Rust | `src/codec/encode.rs` |
| SPS/VUI/PPS/slice headers, `avcC`, levels (Table A-1) | `src/codec/encode/h264.rs` |
| VPS/SPS/VUI/PPS/slice segment headers, `hvcC`, levels (Tables A.8, A.9) | `src/codec/encode/hevc.rs` |
| Picture types and order, references, decode times | `src/codec/encode/gop.rs` |
| Our rate control under a hard limit (below) | `src/codec/encode/limit_rate.rs` |
| The core over an `Accelerator` trait, as `h264::Decoder` is | `src/codec/encode/encoder.rs` |
| libva's encode binding, checked against the headers | `src/vaapi/va/ffi/encode.rs`, `src/vaapi/va/ffi/encode_layout.c` |
| The driver's encoder attributes, and `vaapi-encode-probe` | `src/vaapi/va/encoder/caps.rs`, `examples/vaapi-encode-probe.rs` |
| The VA accelerator: config, context, surfaces, buffers, coded buffer | `src/vaapi/va/encoder/accelerator.rs`, `params.rs`, `surfaces.rs` |
| `VideoEncoder` | `src/vaapi/va/encoder.rs` |
| The checks' tool | `examples/vaapi-encode/` |

The codec module's half names no libva type: the headers, the levels, the
GOP planning (`KeyFrames`, `DecodeTimes`) and the rate control are plain
Rust, tested on any host.

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
reads them back. Whether the driver parses our packed sets or builds its
own from the parameter buffers is not measured. So the configuration
record is made from the parameter sets the hardware emits with each IDR
picture: the first sample carries the record it belongs with, and
`configuration()` before the first frame is only ours.

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
  HEVC 8.3.1 allow (`MAX_ORDER` in `src/codec/encode/gop.rs`).
  H.264 Main and High put B pictures between references, one by default
  (`DEFAULT_B_FRAMES`) and at most 4, L0 two references and L1 one. A run
  of B pictures is closed with a P before an IDR. HEVC and Constrained
  Baseline are I and P only.
- **Time stamps.** Each sample's DTS is the PTS of the frame sent in its
  place: without reordering DTS equals PTS, and with it `I0 P2 B1` gets
  DTSes 0, 1 and 2 frames, so a B picture's is after its PTS. The DTSes
  increase.
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

### Testing

On any host, the codec module's unit tests (`cargo nextest run -p
reims-vgpu-video -E 'test(/^codec::/)'`; `cargo xtask check` runs them with the rest) test
the portable half: headers parsed back with the module's own H.264 parsers
and bit readers, GOP planning, and the core on a fake accelerator.

On both AMD GPUs, a script not in this repository encoded ffmpeg's
synthetic sources (`testsrc2`, `mandelbrot`, `smptehdbars`, noise) through
`examples/vaapi-encode` (`VideoEncoder`). ffmpeg checked the results, and
nothing more: a decode under `-xerror` and every `-err_detect` check, of
the MP4 and the Annex B stream; PSNR and SSIM against the source; the
bitrate against the target; the GOP, time stamps and sync flags against
ffprobe's packets and frames; and the stream's profile, format and size.
H.264 was also decoded by our own decoder, and its frames' MD5s compared
with ffmpeg's.

Run 2026-10-24 on both AMD GPUs, Mesa 26.2.4, 40 cases on each node:

| GPU | result |
|---|---|
| Navi 31, `renderD128` | 40 of 40 pass |
| Phoenix, `renderD129` | 40 of 40 pass |

- **Both GPUs wrote the same bitstreams** in every case but QVBR (the same
  sizes, PSNR and SSIM); QVBR differed by 0.3% in size.
- **Decoding.** Every MP4 and Annex B stream decoded without an error. Every
  H.264 case decoded bit-exact on our decoder, the record-changing one
  included.
- **The cases.** H.264 CB, Main and High, and HEVC Main and Main 10, at
  1080p (P010 for Main 10), 720p, 480p, CIF noise, 4K (H.264 level 5.1),
  96x64 and 200x120 below the minimum, and 1366x768. Modes CQP, CBR, VBR
  and QVBR; B pictures off and 1 to 3; forced key frames; an interval in
  frames and in ticks; a mid-stream bitrate change; full range and BT.709
  colour; CAVLC; PutImage.
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
  `VideoEncoder` (single runs over SSH, upload included); 4K ran 66 to 72
  fps. Every run's fdinfo showed `drm-engine-enc` time (VCN): 199 to 366
  ms for 90 frames of 1080p H.264.
- **B pictures** (Navi 31, 1080p CQP 26/28/30): none gave 7.50 Mbit/s at
  45.4 dB. One gave 6.61 at 44.4, two 6.71 at 44.0, and three 6.55 at
  43.8; B pictures are coded at QP 30. At VBR 3 Mbit/s (mandelbrot 720p),
  none gave 3.11 Mbit/s at 38.4 dB, one 2.62 at 38.1, and two 2.76 at
  38.7. So the default is one (`DEFAULT_B_FRAMES`): it costs one frame of
  delay, and two were not clearly better in these short runs.

### The bitrate floor

No rate under 32,000 bit/s reaches the driver (`MIN_BITS_PER_SECOND` in
`src/vaapi/va/encoder/params.rs`, ours). An average, a peak or a constant rate
under it is raised to it, and with it the HRD buffer, which is a second at
the peak, or half a second under a limit. Without it, an average of 1
would reach radeonsi as a VBR of 1 bit/s, with a buffer of 1 bit, three
quarters full at 0.

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
  640x480, and at 598 to 740 at 720p. HEVC's stops three to five times
  lower. HEVC `mandelbrot` at 640x480 is the one case still falling under
  62,500 (92,765 there, 69,258 at 31,250); every case had stopped by
  31,250.
- **At the floor.** Each source ran at 32,000 peaking at 48,000, and at
  32,000 peaking at 32,000, as an average under the floor reaches the
  driver. The second was within 0.3% of the first in every case. No run
  printed an error, and the kernel log showed no reset or ring timeout on
  either GPU. A stream under a hard limit takes no floor: our rate control
  (below) raises the QP as far as 51.
- **Not measured:** content harder than `mandelbrot` near the floor, and
  sizes under 640x480, whose output at the floor may be over what the
  encoder could make at a lower rate.

### Rate control under a hard limit

A hard limit holds a stream to a number of bytes over any window of decode
time (`EncoderSettings::hard_limit`). radeonsi's VBR and CBR pass the HRD
buffer they are given, and no parameter the driver reads holds them to it
but a minimum QP picked for the content (below; [bugs.md](bugs.md), U14),
so a stream held to a limit is not given to them. Ours (the codec module's
`src/codec/encode/limit_rate.rs`, driven by `src/codec/encode/encoder.rs`'s
`encode_planned`):

- **The driver runs a constant QP.** The config asks for `VA_RC_CQP`, and
  no rate control or HRD buffer is sent (`StreamParams::picture_qp`). Each
  picture's QP is its slice header's `slice_qp_delta` over the PPS's 26,
  which Mesa takes as the picture's QP (`picture_h264_enc.c:259` and
  `picture_hevc_enc.c:263` in Mesa 26.2.4's `src/gallium/frontends/va/`).
- **The limit is held over the pictures.** No second of decode time holds
  more than the buffer and a second at the peak; the pictures to the next
  key frame share the stream's average.
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

**What radeonsi does with the rate parameters**, measured on the RX 7900
XTX (2026-10-24), each a parameter changed alone from a VBR with a buffer
the key frames pass:

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
- `min_qp` is applied, the first picture's too, but the floor that holds a
  stream is the content's: one that holds expensive key frames costs a
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
- The mode of rate control is fixed for a stream.
- The slice headers that come out were not compared with the packed ones,
  as the parameter sets were.
- One slice a picture; no intra refresh, ROI or multipass, though the
  driver lists the first two.

## API

The decoders, as a caller drives them:

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
- `Frame::poc` is `PicOrderCnt()` after operation 5's subtraction.
- `Frame::read_back` and `ReadBack`'s copies are the two steps of the copy
  back ("The copy back"); `Frame::write_to_guest` is zero copy's write.
- `Error` is `Load(LoadError)`, `Open(io::Error)`, `NoDisplay`,
  `Va { call, status, message }`, `Bitstream(h264::Error)`,
  `HevcBitstream`, `Unsupported(String)`, `OutOfSurfaces`,
  `PlaneTooSmall(&'static str)`, or `Stalled(String)`.
- The portable layer is public too: `h264::Decoder` over any
  `Accelerator`, `h264::check_supported(&Sps)`, and the parsers.

## Testing

### On any host

The codec module's unit tests (`cargo nextest run -p
reims-vgpu-video -E 'test(/^codec::/)'`) test the portable layer. The streams for them are
written in the tests from the standard's syntax. They cover what x264 does
not write:
- order count type 1 with non-reference pictures;
- type 2 across a `frame_num` wrap;
- long-term references and MMCO 1 to 6;
- operation 5's reset and bump;
- list modification, and the gap process;
- references refused when missing.

The graphics arm's tests are reims-vgpu's `graphics::vaapi` tests, on
driver answers the tests make up (`src/graphics/vaapi/tests.rs`), and the
host session's on the fake platform (`video_host`'s tests).

### On a VA-API GPU

The module's ignored tests need a VA-API GPU and the installed headers:
the binding's layouts against libva's headers, the import check's sweep,
and the bounded waits ("Waiting for the GPU", "Its tests"):

```sh
cargo nextest run -p reims-vgpu-video --run-ignored only -E 'test(/^vaapi::/)'
```

The hardware checks, from the checkout's root on a Linux host with libva
and the oracles below:

```sh
crates/reims-vgpu-video/tools/h264-check.sh /dev/dri/renderD128
crates/reims-vgpu-video/tools/hevc-check.sh /dev/dri/renderD128
```

Streams, references and logs go to `build/h264-check/` and
`build/hevc-check/` (or `RVGV_H264_OUT` and `RVGV_HEVC_OUT`), which are
reused between runs and never committed. Each script takes its GPU's locks
(`scripts/internal/gpu-locks.sh`), and none where the caller holds them
already (`RVG_GPU_LOCKS_HELD=1`).

**x264, x265, kvazaar and ffmpeg are test oracles.** x264 0.165.3222
writes the H.264 streams, from ffmpeg n9.0.2's `testsrc2` with noise.
ffmpeg decodes each in software (`-hwaccel none -pix_fmt nv12 -f
framemd5`) for the reference MD5 of every frame. None is linked into
anything. None of their output is committed, and the product never calls
them.

Each H.264 stream is decoded three times, and every frame's NV12 MD5 must
equal ffmpeg's:
- from its MP4 (the `avcC` record, length-prefixed samples), in display
  order;
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
(2026-10-24), and on the Iris Xe and a third Radeon 780M (2026-11-07,
"Testing HEVC"), every frame of every stream matching in every form, and the
interlaced streams refused with "interlaced: the SPS's
frame_mbs_only_flag is 0". high-1080p holds a PPS's lists to fall back by
rule A where the SPS has no matrix (7.4.2.2): by rule B, JVT's defaults
become flat lists and none of its frames matches.

### Testing HEVC

**On any host**, the codec module's unit tests (`cargo nextest run -p
reims-vgpu-video -E 'test(/^codec::/)'`) test the HEVC layer, whose streams the tests
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

**On the hardware**, `tools/hevc-check.sh [render node]`, run as
h264-check.sh is (above), with x265 4.3 (8, 10 and 12 bits) and kvazaar
2.3.2 (8 bits) for the streams and ffmpeg n9.0.2 for the references. It
makes every stream first, then decodes them.

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
copies of its streams and x265's ("Crafted HEVC streams", below). Its
reconstruction (`--debug`) is the second reference, as x265's is, and
equalled ffmpeg's decode on every stream but the fields, for which
`--debug` writes other pictures ([bugs.md](bugs.md), U13); there ffmpeg's
check of each picture against kvazaar's MD5 SEI (`-err_detect crccheck`)
is the second, and found every picture's MD5 as kvazaar wrote it. kvazaar
took two things Main does not allow, with no warning: the tile column of
192 samples (kvz-720p-tiles-narrow) and, with `--ref 15`, pictures of up
to 15 current references (kvz-720p-ref15).

Results, 2026-10-24, on both AMD GPUs, Mesa 26.2.4, 108 checks on each
GPU: x265's 68, and kvazaar's 40, its eleven streams each three ways, the
cut and the BLA splice, and the two refusals each as MP4 and Annex B:

| GPU | result |
|---|---|
| Navi 31, `renderD128` | 108 of 108 pass |
| Phoenix, `renderD129` | 108 of 108 pass |

Both scaling-list streams fail on every frame unless the lists go in
raster order ("The HEVC decoder", "Shape"). The parameter sets' range
checks refuse no stream x265 writes, the open-GOP CRAs, the cut and the
mid-stream BLA among them (2026-10-24).

**ffmpeg's software decode is wrong on one stream.** On
main-1366x766-ctu16, ffmpeg n9.0.2's `-hwaccel none` decode differs from
frame 3 on, where x265's own reconstruction (`--recon`), our decode and
ffmpeg's `-hwaccel vaapi` decode on both GPUs all agree ([bugs.md](bugs.md),
U1). Found by narrowing, 2026-10-24, with 30-frame 1366x766 streams:
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
two. With one reference, no B-frames and no temporal prediction, every
reference is long-term: x265's main-720p-ref1 (`--ref 1 --bframes 0
--no-temporal-mvp`) keeps the two pictures before each P picture, its one
list the nearer, and kvazaar's kvz-1080p-ref1 (`--gop 0 --ref 1
--no-tmvp`) the one before each B picture, in both its lists.

`--check` decodes the source and the output through the codec module's
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

Results, 2026-10-25, on both AMD GPUs, Mesa 26.2.4: `hevc-check.sh`
passed 156 of 156 on each GPU, the 108 checks above and 48 more,
main-720p-ref1's and kvz-1080p-ref1's three each and the fourteen crafted
streams' three each. Every crafted stream's frames were its source's on
both GPUs, and so were ffmpeg n9.0.2's and libde265's decodes of it.
libde265 decodes main-1080p-slices itself otherwise than x265, ffmpeg and
both GPUs, every frame, with threads and without, so its two crafted
streams carry a NOTE line, and libde265 decodes them as it decodes their
source.

Both scripts again, 2026-11-07, on the Iris Xe (iHD) and on a Radeon
780M (an 8745H's, radeonsi): `h264-check.sh` passed 62 of 62 and
`hevc-check.sh` 156 of 156 on each, kvz-720p-ref15's 9 frames before its
refusal included.

## Limits and what is next

- Interlaced video (field pictures, MBAFF) is refused. Supporting it
  means field-pair DPB entries, field lists (8.2.4.2.2, 8.2.4.2.5) and
  field POCs, and VA's field flags.
- Long-term references, MMCO 2 to 6, operation 5 and frame number gaps,
  which x264 does not write, reach the hardware only in `h264-craft`'s
  rewrites of x264's streams ("Crafted streams").
- A sample must be one access unit. Annex B streams are split with
  `h264::nal::access_units`, which does not follow arbitrary slice order.
- Where zero copy declines a frame, readback is a copy, since radeonsi
  offers no `USER_PTR` surfaces ("The AMD drivers").
- No concealment of damage: a sample with an error decodes nothing, and
  the pictures after it are refused until an I picture decodes ("The H.264
  decoder", "What it refuses"). Only a frame lost to a `frame_num` gap is
  concealed, with a copy of the last reference.
- HEVC long-term pictures, list modification, a slice header's set
  predicted from the SPS's, and PCM are checked by unit tests only: neither
  x265 nor kvazaar writes them ([bugs.md](bugs.md), T1). Like H.264's
  lists and weights, they may also be what VCN reads from the slice header
  itself.
- HEVC's field signalling (`field_seq_flag`) is not looked at: each
  field is a picture of its own, and VA-API's HEVC has no field flags.
  kvazaar's kvz-720p-fields decodes so, every field matching ("Testing
  HEVC").
- The encoders are H.264's and HEVC's alone ("The encoders"): the device
  offers no other codec.
