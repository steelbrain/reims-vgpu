//! The JPEG engine's decodes on VA-API's baseline JPEG decoder
//! (`VAProfileJPEGBaseline`, `VAEntrypointVLD`): core's `JpegDecoder`.
//!
//! Each decode reads the stream with the codec crate's parser, hands the
//! driver its frame, tables and scan, decodes into a surface of the
//! stream's own sampling made for it, and copies the guest's region of
//! that surface into the guest's surface with core's `write_samples`. The
//! driver writes each sampling into one surface format: 4:2:0 into NV12,
//! 4:2:2 into YUY2, 4:4:4 into 444P and grey into Y800 ([`SurfaceKind`]).
//!
//! A scale, and a layout of another sampling or BGRA, go through the
//! driver's video processing instead (`super::vpp`):
//! the region of the decoded surface, scaled, into NV12 or BGRA
//! ([`Route`]). Which conversions the driver does is tried once, at start;
//! the rest is refused.
//!
//! **Where the output is read.** The GPU writes the output into host
//! memory of ours, imported as a linear surface ([`HostSurface`], the
//! staged copy back's, `super::readback`), and the CPU reads it there once
//! a `poll` of its dma-buf and the surface's sync say the GPU is done: at
//! scale 1 into the stream's own layout the decode itself writes it, and
//! otherwise the conversion's last run does. No driver call maps or copies
//! it for the CPU: radeonsi makes those (`vaMapBuffer` of a derived image,
//! `vaGetImage`) under its display's lock, copying the surface on the GPU
//! and waiting for the copy, which held every other session's calls on the
//! display for 1.1 to 8.7 ms at the mean a 4K decode, and 5.5 to 21.6 ms an
//! 8192x8192 one (docs/video/vaapi.md, "What the gate costs"; U22 in
//! docs/video/bugs.md).
//!
//! **An output past what one host surface takes**, udmabuf's
//! `size_limit_mb` (64 MiB at its stock value, which an 8192x8192 4:2:0
//! output passes), is read in strips ([`Strips`]): the decode, or the
//! conversion's last run, writes it into a surface of the driver's, and for
//! each band of its rows a 1:1 processing run copies the band into the top
//! of one host surface of its width, as many rows as [`STRIP_BYTES`] holds,
//! which the CPU reads once the run is done, as above, into the guest's
//! rows for the band ([`write_band`]). Each run copies, and computes
//! nothing: the same colour standard both sides and no scale, so that each
//! sample is read at a sample's centre (`super::readback::copy`), and each
//! band starts on an even row, so that 4:2:0's chroma rows are whole. The
//! host surface is kept as the others are, so a run of decodes frees
//! nothing for its strips. **Measured** on the RX 7900 XTX and the Radeon
//! 780M (radeonsi, Mesa 26.2.4, 2026-10-26): radeonsi's video processing
//! takes a band's rectangle as its source region and as its window, and
//! with every output read in strips, bands of 16 rows and more, the JPEG
//! gate's 99 cases wrote the same bytes into the destination as read
//! whole, by their digests, on both GPUs: copies of 4:2:0, 4:4:4 and grey,
//! and conversions into NV12, BGRA and, through NV12, the 4:2:2 and 4:4:4
//! layouts, crops and odd sizes among them. So did its 8192x8192 and
//! 16384x16000 outputs at udmabuf's stock limits, in 4 to 12 strips, in
//! the time the derived map took: 8192x8192 4:2:0 in 121 to 123 ms against
//! 124 on the RX 7900 XTX, and 216 to 217 against 212 to 216 on the Radeon
//! 780M, but for 16384x16000 grey, 292 to 293 ms against 275 to 277, and
//! 535 to 537 against 500 to 502 (means of 5).
//!
//! **A 4:2:2 copy past the limit** is read from its surface's memory as
//! BGRA of half its width: radeonsi's video processing writes no 4:2:2
//! surface (Mesa's compositor refuses each as a target,
//! `vl_compositor_proc.c`, **read** in Mesa 26.2.4, and a 1:1 run into YUY2
//! or UYVY failed at `vaEndPicture` with `VA_STATUS_ERROR_OPERATION_FAILED`
//! on both GPUs), so each band's run copies from an [`Alias`] of the YUY2
//! surface, BGRA of half its width at its pitch, into a BGRA host surface of
//! that width ([`strip_format`]), whose rows are the output's bytes as YUY2
//! lays them out. radeonsi lays out every YUY2 surface linearly, in one
//! plane (its tiling takes no 4:2:2 format, `ac_surface.c`, **read** in
//! Mesa 26.2.4; the export's modifier was linear on both GPUs), so each
//! BGRA texel is one Y0 Cb Y1 Cr group of the surface's, and a 1:1 run
//! from RGB to RGB reads each texel at its centre and writes it unchanged,
//! its alpha among it: the samples are never taken as YCbCr, so their
//! siting does not arise, and 4:2:2's chroma is whole on every row. The
//! YUY2 surface is made to be exported ([`OwnedSurface::new`]), and the
//! alias holds the driver's memory with it, so that destroying the alias
//! frees nothing. A trial run of an alias at start ([`Conversions`]) says
//! whether the driver copies one; where it does not, a 4:2:2 copy past the
//! limit is read through the driver's image. **Measured** on the RX 7900
//! XTX and the Radeon 780M (radeonsi, Mesa 26.2.4, 2026-10-26): with a
//! program on libva alone, an alias's bands of up to 32 MiB gave every
//! byte put into a YUY2 surface of noise, at 336x216, 8192x8192 and
//! 16384x16000; with every output read in strips, the JPEG gate's 10
//! copies of 4:2:2 gave the digests they gave read whole, and its
//! 8192x8192 4:2:2 copy at udmabuf's stock limits, in 4 strips, the digest
//! its derived map gave. That decode took 240.2 to 240.5 ms against the
//! map's 229.8 to 230.7 on the RX 7900 XTX, and 358.8 to 359.1 against
//! 353.7 to 353.9 on the Radeon 780M (means of 5); its runs hold no lock of
//! the driver's past their submission, where each map held the display's
//! calls 20.5 ms on the RX 7900 XTX and 5.2 ms on the Radeon 780M
//! (docs/video/vaapi.md, "What the gate costs").
//!
//! **What the engine keeps.** Once a decode succeeds, every surface it
//! wrote or read is kept for later decodes of the same shape ([`Shape`]):
//! its host surfaces, the driver's surface it decoded into, and those a
//! conversion passed through. Past [`KEPT_BYTES`] the least recently used
//! are destroyed, but for the host surfaces of the last decode, which stay
//! whatever their size: making a 4K one took 2.3 to 6 ms on the RX 7900
//! XTX and the Radeon 780M (radeonsi, Mesa 26.2.4, 2026-10-26), about a
//! decode's time. A surface destroyed costs more than its making: the
//! unmap of its memory has amdgpu make every later submission of the
//! process's GPU VM, on any ring, wait for every job the VM had in flight
//! then, and RADV's jobs in the process are of that VM, a guest's on the
//! display among them (docs/video/bugs.md, B92, U23). **Measured** on the
//! RX 7900 XTX and the Radeon 780M (Linux 7.2.9, Mesa 26.2.4, 2026-10-26)
//! by the GPU scheduler's tracepoints of a job's dependencies, with the
//! graphics ring held by a Vulkan spin of the tests' own (`super::spin`):
//! destroying a 256x256 host surface the GPU was done with queued a
//! page-table update that waited for the spin's job, and the next decode's
//! job waited for that update until the spin ended, the decode failing by
//! its deadline; with nothing destroyed, each decode ran beside the spin.
//! Where the engine destroyed the surfaces of each conversion after it,
//! the decode after a conversion waited so. An output read in strips is a
//! surface of the driver's past [`KEPT_BYTES`] by itself (8192x8192 4:2:0
//! is 96 MiB), so its decode leaves its strip's host surface kept, and
//! destroys the rest.
//!
//! **Measured** on both GPUs the same day, against the derived image's
//! map of a decode into the driver's own surface: a decode into a host
//! surface wrote the same samples, every one inside the image, for 4:2:0,
//! 4:2:2, 4:4:4 and grey at 333x211, 640x480 and 3840x2160 (the MCUs'
//! padding past the image differed, which nothing reads), and took what a
//! decode into the driver's surface takes (4K 4:2:0, 7.2 against 7.4 ms on
//! the RX 7900 XTX, 20.7 against 19.9 on the Radeon 780M). A 1:1 run from
//! the driver's surface into a host one wrote the same bytes for NV12,
//! 444P and Y800, at 1.5 to 5.6 ms more for 4K, and radeonsi refused it
//! into YUY2 (`vaEndPicture`, `VA_STATUS_ERROR_OPERATION_FAILED`: Mesa's
//! compositor writes no 4:2:2 surface); so at scale 1 the decode writes
//! host memory itself.
//!
//! The output is read through the driver's images instead, by
//! `vaDeriveImage` and the map, or `vaGetImage` where the driver cannot
//! derive, made on the display's call thread (`super::calls`): on a driver
//! other than radeonsi, whose writes into imports are not measured right
//! (`Display::writes_imports`); where no host surface can be made, as with
//! no `/dev/udmabuf` the process may open; for an output past udmabuf's
//! limit that has no strips, on a driver with no video processing, or a
//! 4:2:2 copy where the trial at start copied no alias, the log saying
//! why; where a check asks for another way back
//! (`super::readback::read_back_by`); and from the first decode on which a
//! driver call writing host memory failed and the same decode through the
//! driver's images did not, which the log says. Those maps hold the
//! display's calls as before (docs/video/bugs.md, B90).
//!
//! A decode's driver calls, its waits and its conversion's runs are
//! bounded by one deadline (`super::wait`). Past it the engine is stalled:
//! that decode fails as kIOReturnDeviceError, and the surface the GPU may
//! still write is kept until it is done, with the context it is written
//! on, and until the display's call thread is done with what it has; the
//! surfaces a run only reads are let go. Each later decode fails at once
//! the same way until the GPU is done and the kept surface is let go, and
//! then decodes go on: the engine is the backend's, for as long as the
//! process runs. A failed decode's surfaces are destroyed, not kept, or
//! kept until the GPU is done where it stalled.

use std::cell::OnceCell;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use reims_vgpu_video_codec::jpeg::{self as syntax, Image};
use reims_vgpu_video_core::jpeg::{
    DecodedSamples, JpegCapabilities, JpegDecode, JpegDecoder, JpegDecodes, JpegError,
    JpegOutputFormat, JpegSubsampling, PixelRect, SampleView, write_bgra, write_samples,
};
use reims_vgpu_video_core::runtime::MonoTime;
use reims_vgpu_video_core::surfaces::PlaneDestination;
use reims_vgpu_video_core::sync::Mutex;

use super::buffer::{Buffer, render_picture};
use super::caps::{Config, Context, Entrypoint, Fourcc, Profile};
use super::display::{Display, Error, check};
use super::ffi::{self, jpeg as va};
use super::image::Image as VaImage;
use super::readback::{self, Alias, Band, HostFormat, HostSurface, ReadbackWay};
use super::vpp::{BUFFER_ALIGNMENT, CONTEXT_SIZE, Pass, render, submit};
use super::vpp_ffi as proc;
use super::wait::{self, LetGo, Work};

/// `VAProfileJPEGBaseline`.
const JPEG_BASELINE: Profile = Profile(va::VAProfileJPEGBaseline);

/// The surface a sampling decodes into, as VA-API names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SurfaceKind {
    rt_format: u32,
    fourcc: u32,
}

/// What a conversion writes for a YCbCr layout.
const NV12: SurfaceKind = SurfaceKind {
    rt_format: ffi::VA_RT_FORMAT_YUV420,
    fourcc: ffi::VA_FOURCC_NV12,
};

/// What a conversion writes for `BGRA`: va.h's `VA_FOURCC_BGRA` is "blue,
/// green, red, alpha", CoreVideo's order.
const BGRA: SurfaceKind = SurfaceKind {
    rt_format: proc::VA_RT_FORMAT_RGB32,
    fourcc: proc::VA_FOURCC_BGRA,
};

/// What a 4:2:2 stream decodes into.
const YUY2: SurfaceKind = SurfaceKind {
    rt_format: va::VA_RT_FORMAT_YUV422,
    fourcc: va::VA_FOURCC_YUY2,
};

impl SurfaceKind {
    fn of(subsampling: JpegSubsampling) -> Self {
        let (rt_format, fourcc) = match subsampling {
            JpegSubsampling::Yuv420 => (ffi::VA_RT_FORMAT_YUV420, ffi::VA_FOURCC_NV12),
            JpegSubsampling::Yuv422 => (YUY2.rt_format, YUY2.fourcc),
            JpegSubsampling::Yuv444 => (va::VA_RT_FORMAT_YUV444, va::VA_FOURCC_444P),
            JpegSubsampling::Yuv411 => (va::VA_RT_FORMAT_YUV411, va::VA_FOURCC_411P),
            JpegSubsampling::Grey => (va::VA_RT_FORMAT_YUV400, va::VA_FOURCC_Y800),
        };
        Self { rt_format, fourcc }
    }

    /// The layout of a host surface of it, where there is one.
    fn host_format(self) -> Option<HostFormat> {
        Some(match self.fourcc {
            ffi::VA_FOURCC_NV12 => HostFormat::Nv12,
            va::VA_FOURCC_YUY2 => HostFormat::Yuy2,
            va::VA_FOURCC_444P => HostFormat::Yuv444,
            va::VA_FOURCC_Y800 => HostFormat::Y800,
            proc::VA_FOURCC_BGRA => HostFormat::Bgra,
            _ => return None,
        })
    }
}

/// VA-API's JPEG decoder on one display, as the JPEG engine's backend.
pub struct VaapiJpegDecoder {
    display: Display,
    /// The config each decode's context is made on, and holds.
    config: Arc<Config>,
    /// The render-target formats the driver decodes JPEG into.
    rt_formats: u32,
    /// The smallest surface the driver makes for JPEG.
    min_size: (u32, u32),
    /// The largest picture it decodes.
    max_size: (u32, u32),
    /// The driver's image formats, for reading a surface it cannot derive.
    image_formats: Vec<ffi::VAImageFormat>,
    /// Its video processing, where it has some: shared with a run kept
    /// past its deadline.
    conversions: Option<Arc<Conversions>>,
    /// Held for each decode, and for the conversions' context: the engine
    /// says it has one core.
    decoding: Mutex<()>,
    /// The decode or run the GPU did not finish by its deadline, and
    /// whether what it kept is let go (module docs).
    stalled: Mutex<Option<(String, Arc<AtomicBool>)>>,
    /// Why the outputs are read through the driver's images, where they
    /// are for every decode (module docs).
    host_refused: Mutex<Option<String>>,
    /// The surfaces the engine's decodes wrote and read, least recently
    /// used first, for later decodes of the same shape (module docs).
    kept: Mutex<Vec<Out>>,
}

impl VaapiJpegDecoder {
    /// The JPEG decoder of `display`'s driver, or [`Error::Unsupported`]
    /// where it has none. What it takes is read from the driver now, once.
    pub fn new(display: Display) -> Result<Self, Error> {
        if !display.supports(JPEG_BASELINE, Entrypoint::VLD) {
            return Err(Error::Unsupported("no baseline JPEG decoder".into()));
        }
        let attributes = display.config_attributes(
            JPEG_BASELINE,
            Entrypoint::VLD,
            &[
                ffi::VAConfigAttribRTFormat,
                ffi::VAConfigAttribMaxPictureWidth,
                ffi::VAConfigAttribMaxPictureHeight,
            ],
        )?;
        let [rt_formats, max_width, max_height] = attributes[..] else {
            return Err(Error::Unsupported("no JPEG config attributes".into()));
        };
        let rt_formats = rt_formats.unwrap_or(ffi::VA_RT_FORMAT_YUV420);
        let surfaces = display.surface_capabilities(JPEG_BASELINE, Entrypoint::VLD)?;
        let mut attribs = [ffi::VAConfigAttrib {
            type_: ffi::VAConfigAttribRTFormat,
            value: rt_formats,
        }];
        let config = Arc::new(Config::new(
            &display,
            JPEG_BASELINE,
            Entrypoint::VLD,
            &mut attribs,
        )?);
        let max = |attribute: Option<u32>, surface: Option<u32>| {
            attribute.or(surface).unwrap_or(u32::from(u16::MAX))
        };
        let decoded: Vec<SurfaceKind> = SUBSAMPLINGS
            .into_iter()
            .map(SurfaceKind::of)
            .filter(|kind| rt_formats & kind.rt_format != 0)
            .collect();
        let conversions = Conversions::probe(&display, &decoded).map(Arc::new);
        let host_refused = display.writes_imports().err();
        if let Some(why) = &host_refused {
            log::info!(
                "the JPEG engine reads its outputs through the VA driver's images, not host \
                 memory of ours: {why}"
            );
        }
        Ok(Self {
            min_size: (
                surfaces.min_width.unwrap_or(1),
                surfaces.min_height.unwrap_or(1),
            ),
            max_size: (
                max(max_width, surfaces.max_width),
                max(max_height, surfaces.max_height),
            ),
            image_formats: display.image_format_list().unwrap_or_default(),
            conversions,
            display,
            config,
            rt_formats,
            decoding: Mutex::new(()),
            stalled: Mutex::new(None),
            host_refused: Mutex::new(host_refused),
            kept: Mutex::new(Vec::new()),
        })
    }

    /// Stalls the engine, the GPU not having finished `what` by its
    /// deadline, and keeps `held`, with what the work may still write, until
    /// the GPU is done with `surface` and the display's calls in hand
    /// (module docs).
    fn stall(&self, what: &str, surface: ffi::VASurfaceID, held: impl Send + 'static) {
        self.stall_on(what, vec![Work::Surface(surface), Work::Calls], held);
    }

    /// [`Self::stall`], until each of `works` is done.
    fn stall_on(&self, what: &str, works: Vec<Work>, held: impl Send + 'static) {
        let (let_go, setter) = LetGo::new();
        *self.stalled.lock() = Some((what.to_owned(), let_go));
        wait::keep(
            &self.display,
            "a stalled JPEG decode's surface",
            works,
            (held, setter),
        );
    }

    /// `error` as the guest hears of it, the engine stalled first where it
    /// is [`Error::Stalled`]: a call the display's call thread has in hand
    /// did not return by the deadline, and the engine waits for it to (what
    /// it uses is destroyed behind it there).
    fn failed(&self, error: Error) -> JpegError {
        if let Error::Stalled(what) = &error {
            self.stall_on(what, vec![Work::Calls], ());
        }
        JpegError::Malfunction(error.to_string())
    }

    /// The work the GPU did not finish by its deadline, where what it kept
    /// is not let go yet; once it is, `None`, and the stall is over.
    fn still_stalled(&self) -> Option<String> {
        let mut stalled = self.stalled.lock();
        let (what, let_go) = stalled.as_ref()?;
        if !let_go.load(Ordering::Acquire) {
            return Some(what.clone());
        }
        log::info!("the GPU is done with the stalled JPEG decode's surface, and decodes go on");
        *stalled = None;
        None
    }

    /// The samplings the driver decodes.
    pub fn subsamplings(&self) -> Vec<JpegSubsampling> {
        SUBSAMPLINGS
            .into_iter()
            .filter(|&s| self.rt_formats & SurfaceKind::of(s).rt_format != 0)
            .collect()
    }

    /// The conversions its video processing did at start, as the decoded
    /// surface's fourcc and the target's.
    pub fn conversions(&self) -> Vec<(Fourcc, Fourcc)> {
        self.conversions
            .iter()
            .flat_map(|c| &c.works)
            .map(|&(from, to)| (Fourcc(from), Fourcc(to)))
            .collect()
    }

    /// The largest picture it decodes.
    pub fn max_size(&self) -> (u32, u32) {
        self.max_size
    }

    /// Checks `image` against the request and the driver, before any
    /// work on the GPU, and says how its output is made.
    fn check(&self, image: &Image<'_>, request: &JpegDecode<'_>) -> Result<Route, JpegError> {
        let frame = &image.frame;
        let (width, height) = (u32::from(frame.width), u32::from(frame.height));
        if (width, height) != (request.width, request.height) {
            return Err(JpegError::SizeMismatch { width, height });
        }
        if frame.subsampling().and_then(subsampling) != Some(request.subsampling) {
            return Err(JpegError::SubsamplingMismatch);
        }
        if frame.components.len() == 3 && image.adobe_transform == Some(0) {
            return Err(JpegError::UnsupportedStream(
                "RGB components (Adobe transform 0)".into(),
            ));
        }
        // What the driver cannot decode at all, before what it cannot write.
        if self.rt_formats & SurfaceKind::of(request.subsampling).rt_format == 0 {
            return Err(JpegError::UnsupportedStream(format!(
                "{:?}, which the driver does not decode",
                request.subsampling
            )));
        }
        if width > self.max_size.0 || height > self.max_size.1 {
            return Err(JpegError::UnsupportedStream(format!(
                "{width}x{height}, over the driver's {}x{}",
                self.max_size.0, self.max_size.1
            )));
        }
        let route = Route::of(request)?;
        let decoded = SurfaceKind::of(request.subsampling);
        let converts = |to: SurfaceKind| {
            self.conversions
                .as_ref()
                .is_some_and(|c| c.works.contains(&(decoded.fourcc, to.fourcc)))
        };
        let done = match route {
            Route::Copy => true,
            Route::Nv12 | Route::Nv12WithChroma { .. } => converts(NV12),
            Route::Bgra => converts(BGRA),
        };
        if done {
            Ok(route)
        } else if request.scale != 1 {
            Err(JpegError::UnsupportedScale(request.scale))
        } else {
            Err(JpegError::UnsupportedFormat(format!(
                "{:?} from a {:?} stream, which the driver's video processing does not convert",
                request.format, request.subsampling
            )))
        }
    }

    /// Decodes `image`, checked, and writes the output into `planes`, read
    /// from host memory of ours where it can be (module docs), and made
    /// again through the driver's images where a driver call on that memory
    /// failed.
    fn decode_checked(
        &self,
        image: &Image<'_>,
        request: &JpegDecode<'_>,
        route: Route,
        planes: &mut [PlaneDestination<'_>],
    ) -> Result<Readback, JpegError> {
        let buffers = Parameters::of(image)?;
        let kind = SurfaceKind::of(request.subsampling);
        let (mcu_width, mcu_height) = image.frame.mcu_size();
        let written = (
            request.width.next_multiple_of(mcu_width),
            request.height.next_multiple_of(mcu_height),
        );
        let size = (
            written.0.max(self.min_size.0),
            written.1.max(self.min_size.1),
        );
        let decode = Decode {
            buffers: &buffers,
            request,
            route,
            kind,
            size,
            written,
        };
        let Some(hosts) = self.host_outputs(&decode)? else {
            return self.decode_into(&decode, planes);
        };
        match self.decode_into_hosts(&decode, hosts, planes) {
            Ok(readback) => Ok(readback),
            Err(failed @ (Failed::Jpeg(_) | Failed::Running(..))) => Err(failed.into_jpeg()),
            Err(Failed::Host(error)) => {
                let read = self.decode_into(&decode, planes);
                if read.is_ok() {
                    log::warn!(
                        "the VA driver failed a JPEG decode's write into host memory of ours, and \
                         made it into its own surface, so the engine's outputs are read through \
                         its images from now on: {error}"
                    );
                    *self.host_refused.lock() = Some(error.to_string());
                    // No later decode writes host memory.
                    self.kept.lock().retain(|out| !matches!(out, Out::Host(_)));
                }
                read
            }
        }
    }

    /// The host surfaces `decode`'s outputs are written into, or read
    /// through in strips, in the order its route writes them: kept ones of
    /// the same format and size, else new ones. `None` where the outputs are
    /// read through the driver's images (module docs), the log saying why
    /// where a surface cannot be made.
    fn host_outputs(&self, decode: &Decode<'_, '_>) -> Result<Option<VecDeque<Host>>, JpegError> {
        let asked = matches!(readback::asked(), None | Some(ReadbackWay::Staged));
        if !asked || self.host_refused.lock().is_some() {
            return Ok(None);
        }
        let output = decode.request.output_size();
        // Each output's format, its size, and its rows for each row of the
        // output's.
        let wanted = match decode.route {
            Route::Copy => vec![(decode.kind.host_format(), decode.size, 1)],
            Route::Nv12 => vec![(Some(HostFormat::Nv12), padded(output), 1)],
            Route::Bgra => vec![(Some(HostFormat::Bgra), padded(output), 1)],
            Route::Nv12WithChroma { stretch, .. } => vec![
                (Some(HostFormat::Nv12), padded(output), 1),
                (
                    Some(HostFormat::Nv12),
                    padded((output.0 * stretch.0, output.1 * stretch.1)),
                    stretch.1,
                ),
            ],
        };
        let Some(wanted) = (wanted.into_iter())
            .map(|(format, size, stretch)| format.map(|format| (format, size, stretch)))
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        // Each output's whole host surface, or none where none is made, and
        // the output is to be read in strips.
        let mut wholes: Vec<Option<HostSurface>> = Vec::with_capacity(wanted.len());
        let mut refused = None;
        for &(format, (width, height), _) in &wanted {
            let made = match self.take_host(format, (width, height)) {
                Some(host) => Ok(host),
                None => HostSurface::new(&self.display, format, width, height),
            };
            match made {
                Ok(host) => wholes.push(Some(host)),
                Err(error @ Error::Stalled(_)) => {
                    self.keep(wholes.into_iter().flatten().map(Out::Host).collect());
                    return Err(self.failed(error));
                }
                Err(error) => {
                    refused.get_or_insert(format!("{format:?} {width}x{height}: {error}"));
                    wholes.push(None);
                }
            }
        }
        let Some(why) = refused else {
            let hosts = wholes.into_iter().flatten().map(Host::Whole);
            return Ok(Some(hosts.collect()));
        };
        // The rest in strips, a decode's in bands of as many of the
        // output's rows.
        let striped = || (wanted.iter().zip(&wholes)).filter(|(_, whole)| whole.is_none());
        let mut failed = match self.conversions.as_deref() {
            None => Some("the driver has no video processing to copy it out in strips".to_owned()),
            Some(conversions) => (striped().find(|((format, ..), _)| !conversions.strips(*format)))
                .map(|((format, ..), _)| {
                    format!("the driver's video processing copies no {format:?} in strips")
                }),
        };
        let rows = (striped()
            .map(|(&(format, (width, _), stretch), _)| band_rows(format, width, stretch)))
        .min()
        .unwrap_or(BUFFER_ALIGNMENT);
        let mut hosts = VecDeque::with_capacity(wanted.len());
        let mut stalled = None;
        for ((format, (width, _), stretch), whole) in wanted.into_iter().zip(wholes) {
            if let Some(whole) = whole {
                hosts.push_back(Host::Whole(whole));
                continue;
            }
            if failed.is_some() || stalled.is_some() {
                continue;
            }
            let (format, width) = strip_format(format, width);
            let size = (width, rows * stretch);
            let made = match self.take_host(format, size) {
                Some(band) => Ok(band),
                None => HostSurface::new(&self.display, format, size.0, size.1),
            };
            match made {
                Ok(band) => hosts.push_back(Host::Strips(band)),
                Err(error @ Error::Stalled(_)) => stalled = Some(error),
                Err(error) => failed = Some(format!("no strip of {format:?} {size:?}: {error}")),
            }
        }
        if let Some(error) = stalled {
            self.keep(hosts.into_iter().map(Host::into_out).collect());
            return Err(self.failed(error));
        }
        if let Some(refusal) = failed {
            self.keep(hosts.into_iter().map(Host::into_out).collect());
            log::info!(
                "a JPEG decode's output is read through the VA driver's image: no host surface of \
                 {why}, and {refusal}"
            );
            return Ok(None);
        }
        Ok(Some(hosts))
    }

    /// A kept host surface of `format` and `size`, taken from those kept,
    /// if there is one.
    fn take_host(&self, format: HostFormat, size: (u32, u32)) -> Option<HostSurface> {
        match self.take_kept(Shape::Host(format, size))? {
            Out::Host(host) => Some(host),
            Out::Driver(_) | Out::Strips(_) => None,
        }
    }

    /// The most recently used kept surface of `shape`, taken from those
    /// kept, if there is one.
    fn take_kept(&self, shape: Shape) -> Option<Out> {
        let mut kept = self.kept.lock();
        let n = kept.iter().rposition(|out| out.shape() == shape)?;
        Some(kept.remove(n))
    }

    /// A surface of the driver's of `kind` and `size`, of which a decode or
    /// a run writes the `written` top-left corner: a kept one of that
    /// shape, else a new one.
    fn driver_surface(
        &self,
        kind: SurfaceKind,
        size: (u32, u32),
        written: (u32, u32),
    ) -> Result<OwnedSurface, Error> {
        let shape = Shape::Driver {
            kind,
            size,
            written,
        };
        match self.take_kept(shape) {
            // Of that shape, a kept surface is the driver's.
            Some(Out::Driver(surface)) => Ok(surface),
            _ => OwnedSurface::new(&self.display, kind, size, written),
        }
    }

    /// Keeps `done`, the surfaces a decode is done with, as the most
    /// recently used, and destroys the least recently used of the others
    /// while what is kept is past [`KEPT_BYTES`]. The host surfaces among
    /// `done` stay, whatever their size: making one again takes about a
    /// decode's time (module docs).
    fn keep(&self, done: Vec<Out>) {
        let mut kept = self.kept.lock();
        let older = kept.len();
        kept.extend(done);
        let sizes: Vec<(u64, bool)> = (kept.iter().enumerate())
            .map(|(n, out)| (out.bytes(), n >= older && matches!(out, Out::Host(_))))
            .collect();
        let mut destroyed = past_budget(&sizes, KEPT_BYTES).into_iter();
        kept.retain(|_| !destroyed.next().unwrap_or(false));
    }

    /// [`Self::decode_with`], its outputs written into `hosts`; once it
    /// succeeds, every surface it wrote and read is kept for later decodes
    /// ([`Self::keep`]), and where it fails, they are destroyed, or kept
    /// until the GPU is done where it stalled.
    fn decode_into_hosts(
        &self,
        decode: &Decode<'_, '_>,
        hosts: VecDeque<Host>,
        planes: &mut [PlaneDestination<'_>],
    ) -> Result<Readback, Failed> {
        let mut outputs = Outputs {
            hosts,
            done: Vec::new(),
        };
        let readback = self.decode_with(decode, &mut outputs, planes)?;
        self.keep(outputs.done);
        Ok(readback)
    }

    /// Decodes into surfaces of the driver's own, and writes the output
    /// into `planes`, read through the driver's images.
    fn decode_into(
        &self,
        decode: &Decode<'_, '_>,
        planes: &mut [PlaneDestination<'_>],
    ) -> Result<Readback, JpegError> {
        self.decode_into_hosts(decode, VecDeque::new(), planes)
            .map_err(Failed::into_jpeg)
    }

    /// Decodes, makes the output as the route says, each output written
    /// into the next of `outputs`' host surfaces, or into a surface of the
    /// driver's where it has none left, and writes it into `planes`. Each
    /// surface it is done with goes into `outputs`.
    fn decode_with(
        &self,
        decode: &Decode<'_, '_>,
        outputs: &mut Outputs,
        planes: &mut [PlaneDestination<'_>],
    ) -> Result<Readback, Failed> {
        let Decode {
            buffers,
            request,
            route,
            kind,
            size,
            written,
        } = *decode;
        let malfunction = |error: Error| Failed::Jpeg(self.failed(error));
        // The decode writes the output itself at scale 1 into the stream's
        // own layout (module docs).
        let decoded = if route == Route::Copy {
            outputs.next(self, kind, size, written)
        } else {
            self.driver_surface(kind, size, written).map(Out::Driver)
        }
        .map_err(malfunction)?;
        let mut targets = [decoded.id()];
        let config = Arc::clone(&self.config);
        let context = match Context::new(&self.display, config, size.0, size.1, &mut targets) {
            Ok(context) => context,
            Err(error @ Error::Stalled(_)) => return Err(malfunction(error)),
            Err(error) => return Err(decoded.write_failed(error)),
        };
        if let Err(error) = buffers.render(&self.display, context.id, decoded.id()) {
            if let Error::Stalled(what) = &error {
                // A picture may be submitted: the context goes first.
                let id = decoded.id();
                self.stall(what, id, (context, decoded));
                return Err(Failed::Jpeg(JpegError::Malfunction(error.to_string())));
            }
            return Err(decoded.write_failed(error));
        }
        let deadline = wait::current(&self.display);
        if let Err(error) = decoded.wait(deadline, "a JPEG decode in time") {
            if let Error::Stalled(what) = &error {
                // The context goes first.
                let id = decoded.id();
                self.stall(what, id, (context, decoded));
                return Err(Failed::Jpeg(JpegError::Malfunction(error.to_string())));
            }
            return Err(decoded.write_failed(error));
        }
        let read = if route == Route::Copy {
            let reading = self.reading(deadline, &malfunction);
            let region = request.region;
            let rows = Rows {
                top: region.y,
                count: region.height,
            };
            in_bands(rows, decoded.band_rows(1), |band| {
                decoded.view(band, kind, &reading, |layout, bytes| {
                    let samples = samples_of(layout, bytes, request.subsampling)?;
                    write_band(&samples, request, band, planes)
                })
            })
        } else {
            self.convert(&decoded, request, route, outputs, planes, deadline)
        };
        // The context first, then what it decoded into.
        drop(context);
        self.done_reading(read, vec![decoded], outputs)
    }

    /// What reads of a decode's outputs take (`Out::view`).
    fn reading<'a>(
        &'a self,
        deadline: MonoTime,
        malfunction: &'a dyn Fn(Error) -> Failed,
    ) -> Reading<'a> {
        Reading {
            formats: &self.image_formats,
            context: self.conversions.as_deref().map(|c| &c.context),
            deadline,
            malfunction,
        }
    }

    /// `read`, a read of `outs`, which go back to `outputs`; where a run
    /// that copies a strip of one did not finish by the deadline
    /// ([`Failed::Running`]), the engine stalls instead, keeping `outs` with
    /// the conversions' context until the GPU is done with the strip.
    fn done_reading(
        &self,
        read: Result<Readback, Failed>,
        outs: Vec<Out>,
        outputs: &mut Outputs,
    ) -> Result<Readback, Failed> {
        match read {
            Err(Failed::Running(error, surface)) => {
                let what = match &error {
                    Error::Stalled(what) => what.clone(),
                    other => other.to_string(),
                };
                self.stall(&what, surface, (self.conversions.clone(), outs));
                Err(Failed::Jpeg(JpegError::Malfunction(error.to_string())))
            }
            read => {
                outs.into_iter().for_each(|out| outputs.done_with(out));
                read
            }
        }
    }

    /// Makes the output from the decoded surface on the driver's video
    /// processing, as `route` says, each run's target the next of
    /// `outputs`' host surfaces where it has one, and writes it into
    /// `planes`.
    fn convert(
        &self,
        decoded: &Out,
        request: &JpegDecode<'_>,
        route: Route,
        outputs: &mut Outputs,
        planes: &mut [PlaneDestination<'_>],
        deadline: MonoTime,
    ) -> Result<Readback, Failed> {
        let conversions = self
            .conversions
            .as_ref()
            .ok_or_else(|| Failed::Jpeg(JpegError::Malfunction("no video processing".into())))?;
        let malfunction = |error: Error| Failed::Jpeg(self.failed(error));
        let (width, height) = request.output_size();
        let whole_of = |(w, h): (u32, u32)| {
            rectangle(PixelRect {
                x: 0,
                y: 0,
                width: w,
                height: h,
            })
        };
        // A run into `to`, which it gives back once the run is done.
        let run = |from: (ffi::VASurfaceID, proc::VARectangle), to: Out, window| {
            let pass = conversion(from.0, from.1, to.id(), window);
            let ran = submit(&self.display, &conversions.context, &pass)
                .and_then(|()| to.wait(deadline, "a video-processing run in time"));
            match ran {
                Ok(()) => Ok(to),
                Err(error) => {
                    if let Error::Stalled(what) = &error {
                        let id = to.id();
                        self.stall(what, id, (Arc::clone(conversions), to));
                        return Err(Failed::Jpeg(JpegError::Malfunction(error.to_string())));
                    }
                    Err(to.write_failed(error))
                }
            }
        };
        let sampled = request.subsampling.chroma_divisors().unwrap_or((2, 2));
        // The region, scaled to `size`, into an output of `kind`, through
        // NV12 surfaces of the driver's where one run would shrink by more
        // than half. Their sides, and the output's, are rounded up to 16,
        // as Mesa rounds its buffers'; only the top-left corner is written
        // and read.
        let mut converted = |kind: SurfaceKind, size: (u32, u32)| {
            let mut from = (decoded.id(), rectangle(request.region));
            let mut passed = None;
            let region = (request.region.width, request.region.height);
            for step in chain(region, sampled, size, kind == BGRA) {
                let surface = self
                    .driver_surface(NV12, padded(step), step)
                    .map(Out::Driver);
                let between = run(from, surface.map_err(malfunction)?, whole_of(step))?;
                from = (between.id(), whole_of(step));
                // The one before is done with: each run waits for its target.
                if let Some(before) = passed.replace(between) {
                    outputs.done_with(before);
                }
            }
            let target = outputs
                .next(self, kind, padded(size), size)
                .map_err(malfunction)?;
            let target = run(from, target, whole_of(size))?;
            if let Some(before) = passed {
                outputs.done_with(before);
            }
            Ok::<_, Failed>(target)
        };
        // What the conversion wrote is the output, whole, at scale 1.
        let whole = |subsampling: JpegSubsampling| JpegDecode {
            subsampling,
            region: PixelRect {
                x: 0,
                y: 0,
                width,
                height,
            },
            scale: 1,
            ..*request
        };
        let reading = self.reading(deadline, &malfunction);
        let all = Rows {
            top: 0,
            count: height,
        };
        match route {
            Route::Copy => Err(Failed::Jpeg(JpegError::Malfunction(
                "a copy is not converted".into(),
            ))),
            Route::Bgra => {
                let rgb = converted(BGRA, (width, height))?;
                let read = in_bands(all, rgb.band_rows(1), |band| {
                    rgb.view(band, BGRA, &reading, |layout, bytes| {
                        let mut planes = planes_from(planes, request.format, band.top)?;
                        write_bgra(&bgra_of(layout, bytes)?, (width, band.count), &mut planes)
                            .map_err(Failed::Jpeg)
                    })
                });
                self.done_reading(read, vec![rgb], outputs)
            }
            Route::Nv12 => {
                let nv12 = converted(NV12, (width, height))?;
                let read = in_bands(all, nv12.band_rows(1), |band| {
                    nv12.view(band, NV12, &reading, |layout, bytes| {
                        let mut samples = samples_of(layout, bytes, JpegSubsampling::Yuv420)?;
                        if request.subsampling == JpegSubsampling::Grey {
                            // Its luma alone: core writes grey's chroma 0x80.
                            samples.subsampling = JpegSubsampling::Grey;
                            samples.chroma = None;
                        }
                        write_band(&samples, &whole(samples.subsampling), band, planes)
                    })
                });
                self.done_reading(read, vec![nv12], outputs)
            }
            Route::Nv12WithChroma { stretch, layout } => {
                let luma = converted(NV12, (width, height))?;
                let chroma = converted(NV12, (width * stretch.0, height * stretch.1))?;
                let each = [luma.band_rows(1), chroma.band_rows(stretch.1)]
                    .into_iter()
                    .flatten()
                    .min();
                let read = in_bands(all, each, |band| {
                    let chroma_rows = Rows {
                        top: band.top * stretch.1,
                        count: band.count * stretch.1,
                    };
                    luma.view(band, NV12, &reading, |luma_layout, luma_bytes| {
                        let luma =
                            samples_of(luma_layout, luma_bytes, JpegSubsampling::Yuv420)?.luma;
                        chroma
                            .view(
                                chroma_rows,
                                NV12,
                                &reading,
                                |chroma_layout, chroma_bytes| {
                                    let samples = DecodedSamples {
                                        subsampling: layout,
                                        luma,
                                        chroma: samples_of(
                                            chroma_layout,
                                            chroma_bytes,
                                            JpegSubsampling::Yuv420,
                                        )?
                                        .chroma,
                                    };
                                    write_band(&samples, &whole(layout), band, planes)
                                },
                            )
                            .map(|_| ())
                    })
                });
                self.done_reading(read, vec![luma, chroma], outputs)
            }
        }
    }
}

/// A decode as [`VaapiJpegDecoder::decode_checked`] makes it: its buffers,
/// the request, how its output is made, the surface kind it decodes into,
/// that surface's size and the top-left corner of it the decode writes,
/// its whole MCUs.
#[derive(Clone, Copy)]
struct Decode<'a, 'b> {
    buffers: &'a Parameters<'b>,
    request: &'a JpegDecode<'b>,
    route: Route,
    kind: SurfaceKind,
    size: (u32, u32),
    written: (u32, u32),
}

/// Why a decode failed.
enum Failed {
    /// As the guest hears of it: the stream's error, or a malfunction, a
    /// stall's among them (the engine stalled already).
    Jpeg(JpegError),
    /// A driver call that writes host memory of ours failed, not past its
    /// deadline: the decode is made again through the driver's images.
    Host(Error),
    /// A run that copies a strip of an output into host memory of ours
    /// (`Strips`) did not finish by the deadline: the GPU may still write
    /// that surface, which its owner keeps (`done_reading`).
    Running(Error, ffi::VASurfaceID),
}

impl From<JpegError> for Failed {
    fn from(error: JpegError) -> Self {
        Self::Jpeg(error)
    }
}

impl Failed {
    /// As the guest hears of it.
    fn into_jpeg(self) -> JpegError {
        match self {
            Self::Jpeg(error) => error,
            Self::Host(error) | Self::Running(error, _) => {
                JpegError::Malfunction(error.to_string())
            }
        }
    }
}

/// The outputs of one decode: the host surfaces still to write or read
/// through, in the order its route writes them, and the surfaces it is
/// done with, which the engine keeps once it succeeds.
struct Outputs {
    hosts: VecDeque<Host>,
    done: Vec<Out>,
}

/// A host surface an output is read from: written whole, or, for an output
/// past what one host surface takes, a strip at a time (`Strips`).
enum Host {
    Whole(HostSurface),
    Strips(HostSurface),
}

impl Host {
    fn into_out(self) -> Out {
        match self {
            Self::Whole(host) | Self::Strips(host) => Out::Host(host),
        }
    }
}

impl Outputs {
    /// The next output, of `kind` and `size`, of which the decode or the
    /// run writes the `written` top-left corner: the next host surface, or
    /// a surface of the driver's where none is left.
    fn next(
        &mut self,
        engine: &VaapiJpegDecoder,
        kind: SurfaceKind,
        size: (u32, u32),
        written: (u32, u32),
    ) -> Result<Out, Error> {
        // Made in this order for this route (`host_outputs`).
        match self.hosts.pop_front() {
            Some(Host::Whole(host)) => {
                debug_assert_eq!(
                    Some(host.shape()),
                    kind.host_format().map(|format| (format, size))
                );
                Ok(Out::Host(host))
            }
            Some(Host::Strips(band)) => {
                debug_assert_eq!(
                    Some((band.shape().0, band.shape().1.0)),
                    kind.host_format()
                        .map(|format| strip_format(format, size.0))
                );
                match engine.driver_surface(kind, size, written) {
                    Ok(surface) => Ok(Out::Strips(Strips {
                        alias: OnceCell::new(),
                        surface,
                        band,
                    })),
                    Err(error) => {
                        self.done.push(Out::Host(band));
                        Err(error)
                    }
                }
            }
            None => engine.driver_surface(kind, size, written).map(Out::Driver),
        }
    }

    /// Takes `out` back once the GPU and the CPU are done with it, a read
    /// in strips as its two surfaces.
    fn done_with(&mut self, out: Out) {
        match out {
            Out::Strips(Strips {
                alias,
                surface,
                band,
            }) => {
                // Destroyed before the surface whose memory it is.
                drop(alias);
                self.done.push(Out::Driver(surface));
                self.done.push(Out::Host(band));
            }
            out => self.done.push(out),
        }
    }
}

/// The bytes of the surfaces the engine keeps for later decodes, at most,
/// past the host surfaces of the last decode (`VaapiJpegDecoder::keep`):
/// those of a 4000x3000 4:2:0 image decoded into `420f` whole and at 1/2,
/// 1/4 and 1/8, 48 MB in all, fit.
const KEPT_BYTES: u64 = 64 << 20;

/// Which of the surfaces kept, least recently used first, to destroy so
/// that what is left is within `budget` bytes: the oldest first. Each is
/// its bytes and whether it stays whatever its size, as a host surface of
/// the last decode does.
fn past_budget(kept: &[(u64, bool)], budget: u64) -> Vec<bool> {
    let mut bytes = kept
        .iter()
        .map(|&(size, _)| size)
        .fold(0, u64::saturating_add);
    kept.iter()
        .map(|&(size, stays)| {
            if bytes <= budget || stays {
                return false;
            }
            bytes = bytes.saturating_sub(size);
            true
        })
        .collect()
}

/// What a kept surface may be used again for: a host surface of a format
/// and size, or a surface of the driver's of a kind and size, of which a
/// decode or a run writes the same top-left corner. Past that corner, a
/// surface of the driver's holds what it was made with, so a run that
/// samples past its region reads what it would in a new surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    Host(HostFormat, (u32, u32)),
    Driver {
        kind: SurfaceKind,
        size: (u32, u32),
        written: (u32, u32),
    },
}

/// A surface a decode or a run writes and the engine reads: host memory of
/// ours, a surface of the driver's read through its images, or one read a
/// strip at a time through host memory of ours.
enum Out {
    Host(HostSurface),
    Driver(OwnedSurface),
    Strips(Strips),
}

/// An output past what one host surface takes (udmabuf's `size_limit_mb`,
/// 64 MiB at its stock value), written into a surface of the driver's and
/// read a band of rows at a time: a 1:1 processing run copies the band into
/// the top of `band`, a host surface of its width, and the CPU reads it
/// there once the run is done (module docs). A YUY2 output's runs copy from
/// `alias`, the surface's memory as BGRA of half its width, into a BGRA
/// `band` of that width ([`strip_format`]).
struct Strips {
    /// Made at the first band's copy, and destroyed before `surface`.
    alias: OnceCell<Alias>,
    surface: OwnedSurface,
    band: HostSurface,
}

impl Strips {
    /// Copies `rows` of the surface into the band, and waits for the copy
    /// until `deadline`; the copy's rows are rounded up to even, within the
    /// surface, so that a 4:2:0 strip copies whole chroma rows.
    fn read(&self, rows: Rows, context: &Context, deadline: MonoTime) -> Result<(), Failed> {
        let (band_format, (width, capacity)) = self.band.shape();
        let left = self.surface.size.1.saturating_sub(rows.top);
        let count = rows.count.next_multiple_of(2).min(left);
        if count < rows.count || count > capacity || !rows.top.is_multiple_of(2) {
            return Err(Failed::Jpeg(JpegError::Malfunction(format!(
                "rows {rows:?} of a {}x{} output, in strips of {capacity}",
                self.surface.size.0, self.surface.size.1
            ))));
        }
        let band = Band {
            width,
            top: rows.top,
            rows: count,
        };
        let display = &self.surface.display;
        let source = if Some(band_format) == self.surface.kind.host_format() {
            Ok(self.surface.id)
        } else {
            self.alias(band_format, width)
        };
        let copied = source
            .and_then(|source| {
                readback::copy(display, context, source, self.band.surface(), Some(band))
            })
            .and_then(|()| {
                (self.band).wait_for_writes(deadline, "a strip of a JPEG decode's output in time")
            });
        match copied {
            Ok(()) => Ok(()),
            Err(error @ Error::Stalled(_)) => Err(Failed::Running(error, self.band.surface())),
            Err(error) => Err(Failed::Host(error)),
        }
    }

    /// The surface's memory as `format`, `width` wide: its alias, made at
    /// the first call.
    fn alias(&self, format: HostFormat, width: u32) -> Result<ffi::VASurfaceID, Error> {
        if let Some(alias) = self.alias.get() {
            return Ok(alias.surface());
        }
        let surface = &self.surface;
        let alias = Alias::new(&surface.display, surface.id, format, width, surface.size.1)?;
        Ok(self.alias.get_or_init(|| alias).surface())
    }

    /// The layout of the output's rows in the band: the output's own
    /// format's, whose bytes a band of BGRA for YUY2 holds as they are.
    fn layout(&self, planes: &[(usize, usize)]) -> Layout {
        let format = (self.surface.kind.host_format()).unwrap_or(self.band.shape().0);
        Layout::of_host(format, planes)
    }
}

/// Rows of an output: `count` of them from `top`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rows {
    top: u32,
    count: u32,
}

/// `rows` of an output in bands of `each` rows at most, top to bottom;
/// with no `each`, one band from the output's first row to the last of
/// `rows`.
fn bands(rows: Rows, each: Option<u32>) -> Vec<Rows> {
    let end = rows.top.saturating_add(rows.count);
    let Some(each) = each.filter(|&each| each > 0) else {
        return vec![Rows { top: 0, count: end }];
    };
    let step = usize::try_from(each).unwrap_or(usize::MAX);
    (rows.top..end)
        .step_by(step)
        .map(|top| Rows {
            top,
            count: each.min(end - top),
        })
        .collect()
}

/// `read` for each band of `rows` ([`bands`]): how the last was read.
fn in_bands(
    rows: Rows,
    each: Option<u32>,
    mut read: impl FnMut(Rows) -> Result<Readback, Failed>,
) -> Result<Readback, Failed> {
    let mut last = Err(Failed::Jpeg(JpegError::Malfunction(
        "an output of no rows".into(),
    )));
    for band in bands(rows, each) {
        last = Ok(read(band)?);
    }
    last
}

/// Writes the rows of `request.region` that `band` holds into `planes`:
/// `samples` are the band's, their first row the band's top, and each
/// plane's rows from the region's top are skipped down to the band's.
/// Where the band is the output from its first row, this is
/// [`write_samples`] itself.
fn write_band(
    samples: &DecodedSamples<'_>,
    request: &JpegDecode<'_>,
    band: Rows,
    planes: &mut [PlaneDestination<'_>],
) -> Result<(), Failed> {
    let region = request.region;
    let top = band.top.max(region.y);
    let bottom = (band.top.saturating_add(band.count)).min(region.y.saturating_add(region.height));
    if top >= bottom {
        return Ok(());
    }
    let within = JpegDecode {
        region: PixelRect {
            y: top - band.top,
            height: bottom - top,
            ..region
        },
        ..*request
    };
    let mut planes = planes_from(planes, request.format, top - region.y)?;
    write_samples(samples, &within, &mut planes).map_err(Failed::Jpeg)
}

/// `planes`, a destination in `format`, from its `row`th row of samples
/// down: a biplanar layout's chroma plane from its row for that row.
fn planes_from<'p>(
    planes: &'p mut [PlaneDestination<'_>],
    format: JpegOutputFormat,
    row: u32,
) -> Result<Vec<PlaneDestination<'p>>, Failed> {
    let biplanar = matches!(
        format,
        JpegOutputFormat::Biplanar420
            | JpegOutputFormat::Biplanar422
            | JpegOutputFormat::Biplanar444
    );
    let down = format.chroma_divisors().map_or(1, |(_, down)| down);
    let short = || {
        Failed::Jpeg(JpegError::Malfunction(
            "a destination plane smaller than the output".into(),
        ))
    };
    (planes.iter_mut().enumerate())
        .map(|(n, plane)| {
            let divisor = if biplanar && n == 1 { down } else { 1 };
            if !row.is_multiple_of(divisor) {
                return Err(short());
            }
            let skipped = row / divisor;
            let start = usize::try_from(skipped)
                .ok()
                .and_then(|rows| rows.checked_mul(plane.bytes_per_row))
                .filter(|&start| start <= plane.bytes.len())
                .ok_or_else(short)?;
            Ok(PlaneDestination {
                bytes: &mut plane.bytes[start..],
                bytes_per_row: plane.bytes_per_row,
                rows: plane.rows.checked_sub(skipped).ok_or_else(short)?,
            })
        })
        .collect()
}

/// What reads of a decode's outputs take ([`Out::view`]): the driver's
/// image formats, for one through its images; the conversions' context and
/// the decode's deadline, for one in strips; and what a driver call's error
/// fails the decode with.
struct Reading<'a> {
    formats: &'a [ffi::VAImageFormat],
    context: Option<&'a Context>,
    deadline: MonoTime,
    malfunction: &'a dyn Fn(Error) -> Failed,
}

/// The most bytes of a host surface an output past udmabuf's limit is read
/// through, a strip at a time: half udmabuf's 64 MiB at its stock
/// `size_limit_mb`.
const STRIP_BYTES: usize = 32 << 20;

/// The rows of a band of an output read in strips: as many as a host
/// surface of `format`, `width` wide, holds within [`STRIP_BYTES`] at
/// `stretch` of its rows a row of the output's, in whole 16s, and 16 at
/// the least.
fn band_rows(format: HostFormat, width: u32, stretch: u32) -> u32 {
    let sixteen = BUFFER_ALIGNMENT.saturating_mul(stretch.max(1));
    let Some(each) = format.bytes(width, sixteen).filter(|&bytes| bytes > 0) else {
        return BUFFER_ALIGNMENT;
    };
    // A plane's start rounds up to a page once a surface, not once in each
    // 16 rows, so the band's surface is within the bytes of its 16s.
    let sixteens = u32::try_from(STRIP_BYTES / each).unwrap_or(u32::MAX);
    sixteens.max(1).saturating_mul(BUFFER_ALIGNMENT)
}

/// The format and width of the host surface a strip of an output of
/// `format`, `width` wide, is copied into: its own, but for YUY2, which
/// radeonsi's video processing does not write, BGRA of half the width,
/// whose rows hold the same bytes at the same pitch (module docs).
fn strip_format(format: HostFormat, width: u32) -> (HostFormat, u32) {
    match format {
        HostFormat::Yuy2 => (HostFormat::Bgra, width.div_ceil(2)),
        format => (format, width),
    }
}

impl Out {
    /// The surface a decode or a run writes.
    fn id(&self) -> ffi::VASurfaceID {
        match self {
            Self::Host(host) => host.surface(),
            Self::Driver(surface) | Self::Strips(Strips { surface, .. }) => surface.id,
        }
    }

    fn shape(&self) -> Shape {
        match self {
            Self::Host(host) => {
                let (format, size) = host.shape();
                Shape::Host(format, size)
            }
            Self::Driver(surface) | Self::Strips(Strips { surface, .. }) => Shape::Driver {
                kind: surface.kind,
                size: surface.size,
                written: surface.written,
            },
        }
    }

    /// The bytes it holds: a host surface's memory, or a surface of the
    /// driver's at its format's samples, padding not counted.
    fn bytes(&self) -> u64 {
        let host = |host: &HostSurface| u64::try_from(host.bytes().0.len()).unwrap_or(u64::MAX);
        match self {
            Self::Host(band) => host(band),
            Self::Driver(surface) => surface.bytes(),
            Self::Strips(Strips { surface, band, .. }) => {
                surface.bytes().saturating_add(host(band))
            }
        }
    }

    /// Waits for the GPU's writes into it until `deadline`: into host
    /// memory, on its dma-buf first (`HostSurface::wait_for_writes`).
    fn wait(&self, deadline: MonoTime, what: &str) -> Result<(), Error> {
        match self {
            Self::Host(host) => host.wait_for_writes(deadline, what),
            Self::Driver(surface) | Self::Strips(Strips { surface, .. }) => {
                wait::wait_for(&surface.display, Work::Surface(surface.id), deadline, what)
            }
        }
    }

    /// The failure of a call that writes it, not past its deadline.
    fn write_failed(&self, error: Error) -> Failed {
        match self {
            Self::Host(_) | Self::Strips(_) => Failed::Host(error),
            Self::Driver(_) => Failed::Jpeg(JpegError::Malfunction(error.to_string())),
        }
    }

    /// The rows of the bands it is read in, at `stretch` of its rows a row
    /// of the output's: those its strip holds, or none where it is read
    /// whole.
    fn band_rows(&self, stretch: u32) -> Option<u32> {
        match self {
            Self::Strips(Strips { band, .. }) => Some(band.shape().1.1 / stretch.max(1)),
            Self::Host(_) | Self::Driver(_) => None,
        }
    }

    /// Lends `use_samples` its `rows`' samples, of `kind`, and where they
    /// are in the bytes, from the first of `rows`: host memory as the GPU
    /// wrote it, the driver's image ([`OwnedSurface::view`]), or the strip
    /// a 1:1 run copied `rows` into ([`Strips::read`]).
    fn view(
        &self,
        rows: Rows,
        kind: SurfaceKind,
        reading: &Reading<'_>,
        use_samples: impl FnOnce(&Layout, &[u8]) -> Result<(), Failed>,
    ) -> Result<Readback, Failed> {
        let from = |layout: Layout| {
            layout.at_row(rows.top).ok_or_else(|| {
                Failed::Jpeg(JpegError::Malfunction(format!(
                    "row {} of a {} surface",
                    rows.top,
                    Fourcc(layout.fourcc)
                )))
            })
        };
        match self {
            Self::Host(host) => {
                let (bytes, planes) = host.bytes();
                use_samples(&from(Layout::of_host(host.shape().0, planes))?, bytes)?;
                Ok(Readback::Host)
            }
            Self::Driver(surface) => {
                let formats = reading.formats;
                surface.view(kind, formats, reading.malfunction, |layout, bytes| {
                    use_samples(&from(*layout)?, bytes)
                })
            }
            Self::Strips(strips) => {
                let Some(context) = reading.context else {
                    return Err(Failed::Jpeg(JpegError::Malfunction(
                        "strips with no video processing".into(),
                    )));
                };
                strips.read(rows, context, reading.deadline)?;
                let (bytes, planes) = strips.band.bytes();
                use_samples(&strips.layout(planes), bytes)?;
                Ok(Readback::Strips)
            }
        }
    }
}

/// `size`, its sides rounded up to 16, as Mesa rounds its buffers'.
fn padded((width, height): (u32, u32)) -> (u32, u32) {
    (
        width.next_multiple_of(BUFFER_ALIGNMENT),
        height.next_multiple_of(BUFFER_ALIGNMENT),
    )
}

/// The sizes of the NV12 surfaces a conversion of a `from`-sized region,
/// whose chroma is `divisors` coarser than its luma, passes through on its
/// way to `to`: each run shrinks the luma, and the chroma, by at most half
/// a side. One run of 1/4 or 1/8 scored 22 to 43 dB against core's
/// reference, where 1/2 scored 61 to 64 (radeonsi, Mesa 26.2.4,
/// 2026-10-24). That fits a filter of two samples a side: a halving
/// averages them all and a quarter skips half. Ours.
fn steps(from: (u32, u32), divisors: (u32, u32), to: (u32, u32)) -> Vec<(u32, u32)> {
    let mut sizes = Vec::new();
    let (mut current, mut divisors) = (from, divisors);
    loop {
        // An NV12 surface's chroma is half its size, so a source's chroma
        // shrinks by half at most into one at least `current / divisors`.
        let next = |side: u32, target: u32, divisor: u32| {
            target.max(side.div_ceil(2)).max(side.div_ceil(divisor))
        };
        let size = (
            next(current.0, to.0, divisors.0),
            next(current.1, to.1, divisors.1),
        );
        // A run that keeps the size still takes the chroma to NV12's.
        let stuck = size == current && divisors == (2, 2);
        if size == to || stuck || sizes.len() >= 8 {
            return sizes;
        }
        sizes.push(size);
        (current, divisors) = (size, (2, 2));
    }
}

/// The NV12 surfaces a conversion to `size` passes through ([`steps`]).
/// Into RGB, whose every pixel takes chroma, they stop at twice the size,
/// whose chroma is the size's, and one run goes from there.
fn chain(from: (u32, u32), divisors: (u32, u32), size: (u32, u32), rgb: bool) -> Vec<(u32, u32)> {
    if !rgb {
        return steps(from, divisors, size);
    }
    let bound = (from.0.min(2 * size.0), from.1.min(2 * size.1));
    let mut sizes = steps(from, divisors, bound);
    if bound != from {
        sizes.push(bound);
    }
    sizes
}

/// The samplings a JPEG decoder may decode, in the order they are listed.
const SUBSAMPLINGS: [JpegSubsampling; 5] = [
    JpegSubsampling::Yuv420,
    JpegSubsampling::Yuv422,
    JpegSubsampling::Yuv444,
    JpegSubsampling::Yuv411,
    JpegSubsampling::Grey,
];

/// How a decode's output is made from the decoded surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    /// The decoded samples, as they are, at scale 1 into a layout of the
    /// stream's sampling, or grey into any YCbCr layout.
    Copy,
    /// One run into NV12: its luma and chroma for `420f`, its luma alone
    /// for a grey stream's YCbCr layouts.
    Nv12,
    /// One run into NV12 for the luma, and another into NV12 at `stretch`
    /// times the output's width and height, whose chroma is the output's
    /// at `layout`'s sampling: twice as tall for 4:2:2, twice both ways for
    /// 4:4:4. Ours: radeonsi's video processing writes no 4:2:2 or 4:4:4
    /// surface, and an NV12 surface's chroma at those sizes is that of the
    /// layouts.
    Nv12WithChroma {
        /// The second run's size over the output's.
        stretch: (u32, u32),
        /// The sampling the layout's chroma is at.
        layout: JpegSubsampling,
    },
    /// One run into BGRA, with BT.601 at full range named on both sides,
    /// as the JPEG's samples are JFIF's.
    Bgra,
}

impl Route {
    /// How `request`'s output is made, whatever the driver does: what
    /// Apple's hardware refuses is refused here, as
    /// [`JpegError::UnsupportedFormat`] (kIOReturnError).
    fn of(request: &JpegDecode<'_>) -> Result<Route, JpegError> {
        let Some(layout) = request.format.chroma_divisors() else {
            return Ok(Route::Bgra);
        };
        let Some(sampled) = request.subsampling.chroma_divisors() else {
            return Ok(if request.scale == 1 {
                Route::Copy
            } else {
                Route::Nv12
            });
        };
        if layout == (1, 1) && sampled != (1, 1) {
            // Measured: `444f` from 4:2:0 and 4:2:2 is kIOReturnError on
            // Apple's host.
            return Err(JpegError::UnsupportedFormat(format!(
                "{:?} from a {:?} stream, which Apple's hardware refuses too",
                request.format, request.subsampling
            )));
        }
        Ok(match layout {
            _ if layout == sampled && request.scale == 1 => Route::Copy,
            (2, 2) => Route::Nv12,
            (2, 1) => Route::Nv12WithChroma {
                stretch: (1, 2),
                layout: JpegSubsampling::Yuv422,
            },
            _ => Route::Nv12WithChroma {
                stretch: (2, 2),
                layout: JpegSubsampling::Yuv444,
            },
        })
    }
}

/// The driver's video processing for the engine: a context made once, the
/// conversions a trial run did at start, as (decoded fourcc, target
/// fourcc), and whether a trial run copied a YUY2 surface's memory as BGRA
/// (`Strips`).
struct Conversions {
    context: Context,
    works: Vec<(u32, u32)>,
    aliases: bool,
}

impl Conversions {
    /// None where the driver has no video processing. Each surface the
    /// decoder writes is tried into NV12 and into BGRA, halved, since a
    /// driver lists surfaces it cannot write (radeonsi's 4:2:2, measured
    /// 2026-10-24 on the RX 7900 XTX and the Radeon 780M); and where it
    /// decodes 4:2:2, a YUY2 surface's [`Alias`] as BGRA of half its width
    /// is copied 1:1, as a 4:2:2 output's strips are.
    fn probe(display: &Display, decoded: &[SurfaceKind]) -> Option<Conversions> {
        if !display.supports(Profile::NONE, Entrypoint::VIDEO_PROC) {
            return None;
        }
        let config = Config::new(display, Profile::NONE, Entrypoint::VIDEO_PROC, &mut []).ok()?;
        let context =
            Context::new(display, config, CONTEXT_SIZE.0, CONTEXT_SIZE.1, &mut []).ok()?;
        let tried = |from: SurfaceKind, to: SurfaceKind| {
            let (Ok(source), Ok(target)) = (
                OwnedSurface::new(display, from, (64, 64), (64, 64)),
                OwnedSurface::new(display, to, (32, 32), (32, 32)),
            ) else {
                return false;
            };
            let all = |side| {
                rectangle(PixelRect {
                    x: 0,
                    y: 0,
                    width: side,
                    height: side,
                })
            };
            let pass = conversion(source.id, all(64), target.id, all(32));
            match render(display, &context, &pass, wait::current(display)) {
                Ok(()) => true,
                Err(Error::Stalled(_)) => {
                    let works = vec![Work::Surface(target.id)];
                    wait::keep(display, "a stalled trial run", works, (source, target));
                    false
                }
                Err(_) => false,
            }
        };
        let mut works = Vec::new();
        for &from in decoded {
            for to in [NV12, BGRA] {
                if tried(from, to) {
                    works.push((from.fourcc, to.fourcc));
                }
            }
        }
        let aliases = decoded.contains(&YUY2) && Self::tried_alias(display, &context);
        Some(Conversions {
            context,
            works,
            aliases,
        })
    }

    /// Whether a 1:1 run on `context` copies a 64x64 YUY2 surface's alias
    /// as 32x64 BGRA into a BGRA surface.
    fn tried_alias(display: &Display, context: &Context) -> bool {
        let (Ok(source), Ok(target)) = (
            OwnedSurface::new(display, YUY2, (64, 64), (64, 64)),
            OwnedSurface::new(display, BGRA, (32, 64), (32, 64)),
        ) else {
            return false;
        };
        let Ok(alias) = Alias::new(display, source.id, HostFormat::Bgra, 32, 64) else {
            return false;
        };
        let ran =
            readback::copy(display, context, alias.surface(), target.id, None).and_then(|()| {
                let deadline = wait::current(display);
                wait::wait_for(display, Work::Surface(target.id), deadline, "a trial run")
            });
        match ran {
            Ok(()) => true,
            Err(Error::Stalled(_)) => {
                let works = vec![Work::Surface(target.id)];
                // The alias goes before the surface it aliases.
                wait::keep(
                    display,
                    "a stalled trial run",
                    works,
                    (alias, source, target),
                );
                false
            }
            Err(_) => false,
        }
    }

    /// Whether an output of `format` past what one host surface takes is
    /// read in strips: but for YUY2, where the trial at start copied its
    /// alias.
    fn strips(&self, format: HostFormat) -> bool {
        format != HostFormat::Yuy2 || self.aliases
    }
}

/// The colour standard and range both sides of a conversion are named
/// with: a JPEG's samples are BT.601's at full range (JFIF).
const JFIF: (proc::VAProcColorStandardType, u8) =
    (proc::VAProcColorStandardBT601, proc::VA_SOURCE_RANGE_FULL);

/// One run of `region` of `surface` into `window` of `target`, scaled with
/// the driver's smooth filter and not turned.
fn conversion(
    surface: ffi::VASurfaceID,
    region: proc::VARectangle,
    target: ffi::VASurfaceID,
    window: proc::VARectangle,
) -> Pass {
    Pass {
        surface,
        region: Some(region),
        target,
        window: Some(window),
        from: JFIF,
        to: JFIF,
        filter: proc::VA_FILTER_SCALING_HQ,
        turn: (proc::VA_ROTATION_NONE, proc::VA_MIRROR_NONE),
        siting: JFIF_SITING,
        out_siting: 0,
    }
}

/// Where a JPEG's chroma samples sit: between the luma samples they
/// cover, both ways (JFIF). Unnamed, Mesa 26.2 takes 4:2:0's chroma as
/// sited on the left, as MPEG-2's: its BGRA matched a conversion of
/// chroma interpolated so at 69 to 71 dB, and one from the centres at 30
/// to 35 dB (radeonsi, Mesa 26.2.4, 2026-10-24).
const JFIF_SITING: u8 =
    proc::VA_CHROMA_SITING_VERTICAL_CENTER | proc::VA_CHROMA_SITING_HORIZONTAL_CENTER;

/// A rectangle as video processing takes it. The decoder's surfaces are
/// at most 16384 a side, which fits.
fn rectangle(rect: PixelRect) -> proc::VARectangle {
    proc::VARectangle {
        x: i16::try_from(rect.x).unwrap_or(i16::MAX),
        y: i16::try_from(rect.y).unwrap_or(i16::MAX),
        width: u16::try_from(rect.width).unwrap_or(u16::MAX),
        height: u16::try_from(rect.height).unwrap_or(u16::MAX),
    }
}

impl JpegDecoder for VaapiJpegDecoder {
    /// The samplings the driver lists, up to its largest picture.
    fn decodes(&self) -> JpegDecodes {
        JpegDecodes {
            subsamplings: self.subsamplings(),
            max_size: self.max_size,
        }
    }

    fn capabilities(&self) -> JpegCapabilities {
        JpegCapabilities {
            cores: 1,
            // Only those whose values fit 8 bits decode: VA-API's tables
            // are 8-bit.
            sixteen_bit_quantization_tables: false,
            // The parser refuses a stream without its EOI.
            missing_eoi: false,
            restart_logging: false,
            interchange_formats: false,
        }
    }

    fn decode(
        &self,
        request: &JpegDecode<'_>,
        planes: &mut [PlaneDestination<'_>],
    ) -> Result<(), JpegError> {
        let started = self.display.runtime().now();
        let image = syntax::parse(request.data).map_err(stream_error)?;
        let route = self.check(&image, request)?;
        // VCN decodes damaged data without an error, so the scan is walked
        // on the CPU first, as Apple's hardware would refuse it.
        let walked = self.display.runtime().now();
        syntax::walk(&image).map_err(stream_error)?;
        let walk_time = self.display.runtime().elapsed_since(walked);
        let _decoding = self.decoding.lock();
        if let Some(what) = self.still_stalled() {
            let error = Error::Stalled(format!("{what}, of an earlier decode"));
            return Err(JpegError::Malfunction(error.to_string()));
        }
        // One deadline for the decode's calls and waits, from the first.
        let deadline = wait::Deadline::default();
        let readback = wait::by(&deadline, || {
            self.decode_checked(&image, request, route, planes)
        })?;
        log::info!(
            "VA-API JPEG decode: {}x{} {:?} into {:?}, region {}x{} at ({}, {}) at 1/{}, \
             by {route:?}, read back by {readback:?}, in {:.1} ms, the scan walked in {:.2} ms",
            request.width,
            request.height,
            request.subsampling,
            request.format,
            request.region.width,
            request.region.height,
            request.region.x,
            request.region.y,
            request.scale,
            self.display.runtime().elapsed_since(started).as_secs_f64() * 1000.0,
            walk_time.as_secs_f64() * 1000.0
        );
        Ok(())
    }
}

/// Core's error for the parser's or the walk's.
fn stream_error(error: syntax::Error) -> JpegError {
    match error {
        syntax::Error::Unsupported(what) => JpegError::UnsupportedStream(what.into()),
        syntax::Error::ScanEndsEarly => JpegError::ScanEndsEarly,
        other => JpegError::BadData(other.to_string()),
    }
}

/// The subsampling core numbers as CMPhoto does, of one the parser read.
fn subsampling(read: syntax::Subsampling) -> Option<JpegSubsampling> {
    Some(match read {
        syntax::Subsampling::Yuv420 => JpegSubsampling::Yuv420,
        syntax::Subsampling::Yuv422 => JpegSubsampling::Yuv422,
        syntax::Subsampling::Yuv444 => JpegSubsampling::Yuv444,
        syntax::Subsampling::Yuv411 => JpegSubsampling::Yuv411,
        syntax::Subsampling::Grey => JpegSubsampling::Grey,
        syntax::Subsampling::Yuv440 => return None,
    })
}

/// The four parameter buffers' contents and the slice's data, for one
/// image.
struct Parameters<'a> {
    picture: va::VAPictureParameterBufferJPEGBaseline,
    quantization: va::VAIQMatrixBufferJPEGBaseline,
    huffman: va::VAHuffmanTableBufferJPEGBaseline,
    slice: va::VASliceParameterBufferJPEGBaseline,
    data: &'a [u8],
}

impl<'a> Parameters<'a> {
    fn of(image: &Image<'a>) -> Result<Self, JpegError> {
        let unsupported = |what: &str| Err(JpegError::UnsupportedStream(what.into()));
        let frame = &image.frame;
        let mut picture = va::VAPictureParameterBufferJPEGBaseline {
            picture_width: frame.width,
            picture_height: frame.height,
            num_components: u8::try_from(frame.components.len()).unwrap_or(0),
            ..Default::default()
        };
        for (to, from) in picture.components.iter_mut().zip(&frame.components) {
            *to = va::VAPictureParameterBufferJPEGBaselineComponent {
                component_id: from.id,
                h_sampling_factor: from.horizontal,
                v_sampling_factor: from.vertical,
                quantiser_table_selector: from.quantization_table,
            };
        }
        let mut quantization = va::VAIQMatrixBufferJPEGBaseline::default();
        for (index, table) in image.quantization_tables.iter().enumerate() {
            let Some(table) = table else { continue };
            for (to, &from) in quantization.quantiser_table[index]
                .iter_mut()
                .zip(&table.values)
            {
                let Ok(value) = u8::try_from(from) else {
                    return unsupported("a quantization value over 255, which VA-API cannot take");
                };
                *to = value;
            }
            quantization.load_quantiser_table[index] = 1;
        }
        let mut huffman = va::VAHuffmanTableBufferJPEGBaseline::default();
        if image
            .scan
            .components
            .iter()
            .any(|c| c.dc_table > 1 || c.ac_table > 1)
        {
            return unsupported("Huffman tables 2 and 3, which VA-API has no room for");
        }
        // Only the tables the scan names: the parser checks their values
        // (F.1.2.1, F.1.2.2), and a table defined but unused is not checked.
        let scan = &image.scan.components;
        for index in 0..2 {
            let to = &mut huffman.huffman_table[index];
            let named = |table: u8| usize::from(table) == index;
            let dc_table =
                image.dc_tables[index].filter(|_| scan.iter().any(|c| named(c.dc_table)));
            let ac_table =
                image.ac_tables[index].filter(|_| scan.iter().any(|c| named(c.ac_table)));
            if let Some(dc) = dc_table {
                let Some(values) = to.dc_values.get_mut(..dc.values.len()) else {
                    return unsupported("a DC table of over 12 values");
                };
                values.copy_from_slice(dc.values);
                to.num_dc_codes = dc.counts;
                huffman.load_huffman_table[index] = 1;
            }
            if let Some(ac) = ac_table {
                let Some(values) = to.ac_values.get_mut(..ac.values.len()) else {
                    return unsupported("an AC table of over 162 values");
                };
                values.copy_from_slice(ac.values);
                to.num_ac_codes = ac.counts;
                huffman.load_huffman_table[index] = 1;
            }
        }
        let mut slice = va::VASliceParameterBufferJPEGBaseline {
            slice_data_size: u32::try_from(image.entropy_coded_data.len())
                .map_err(|_| JpegError::UnsupportedStream("over 4 GiB of data".into()))?,
            slice_data_flag: ffi::VA_SLICE_DATA_FLAG_ALL,
            num_components: u8::try_from(image.scan.components.len()).unwrap_or(0),
            restart_interval: image.restart_interval,
            num_mcus: frame.mcu_count(),
            ..Default::default()
        };
        for (to, from) in slice.components.iter_mut().zip(&image.scan.components) {
            *to = va::VASliceParameterBufferJPEGBaselineComponent {
                component_selector: from.selector,
                dc_table_selector: from.dc_table,
                ac_table_selector: from.ac_table,
            };
        }
        Ok(Self {
            picture,
            quantization,
            huffman,
            slice,
            data: image.entropy_coded_data,
        })
    }

    /// Decodes the picture into `target`.
    fn render(
        &self,
        display: &Display,
        context: ffi::VAContextID,
        target: ffi::VASurfaceID,
    ) -> Result<(), Error> {
        let picture = Buffer::with_value(
            display,
            context,
            ffi::VAPictureParameterBufferType,
            &self.picture,
        )?;
        let quantization = Buffer::with_value(
            display,
            context,
            ffi::VAIQMatrixBufferType,
            &self.quantization,
        )?;
        let huffman = Buffer::with_value(
            display,
            context,
            va::VAHuffmanTableBufferType,
            &self.huffman,
        )?;
        let slice = Buffer::with_value(
            display,
            context,
            ffi::VASliceParameterBufferType,
            &self.slice,
        )?;
        let data = Buffer::with_bytes(display, context, ffi::VASliceDataBufferType, self.data)?;
        render_picture(
            display,
            context,
            target,
            &[&[&picture, &quantization, &huffman], &[&slice, &data]],
        )
    }
}

/// How a surface's samples were read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Readback {
    /// From host memory of ours, which the GPU wrote (module docs).
    Host,
    /// From host memory of ours, into which 1:1 runs copied the driver's
    /// surface a strip at a time (`Strips`).
    Strips,
    /// `vaDeriveImage`: mapped in place.
    Derived,
    /// `vaCreateImage` and `vaGetImage`: copied out by the driver.
    Copied,
}

/// One surface of the driver's, of `kind` and `size`, of which each decode
/// or run into it writes the `written` top-left corner: kept for later
/// decodes, destroyed where a decode fails, or once the GPU is done with it
/// where a run into it passed its deadline.
struct OwnedSurface {
    display: Display,
    id: ffi::VASurfaceID,
    kind: SurfaceKind,
    size: (u32, u32),
    written: (u32, u32),
}

impl OwnedSurface {
    /// Its bytes at its format's samples, padding not counted.
    fn bytes(&self) -> u64 {
        let pixels = u64::from(self.size.0) * u64::from(self.size.1);
        // Bytes per pixel, in halves.
        let halves = match self.kind.fourcc {
            ffi::VA_FOURCC_NV12 => 3,
            va::VA_FOURCC_YUY2 => 4,
            va::VA_FOURCC_Y800 => 2,
            proc::VA_FOURCC_BGRA => 8,
            _ => 6,
        };
        pixels * halves / 2
    }

    /// A surface of `kind` and `size`. A YUY2 one is made to be exported,
    /// as a strip's [`Alias`] of it is: Mesa takes the memory type
    /// `VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2` with no descriptor as that
    /// hint, and makes the surface shareable (`vlVaCreateSurfaces2`), where
    /// radeonsi would move a surface made for the process alone into new
    /// memory at its first export and free the old (`si_texture_get_handle`;
    /// **read** in Mesa 26.2.4), an unmap after which the process's next
    /// work waits for every job then on the GPU (docs/video/bugs.md, B92).
    fn new(
        display: &Display,
        kind: SurfaceKind,
        size: (u32, u32),
        written: (u32, u32),
    ) -> Result<Self, Error> {
        let mut attributes = [
            ffi::VASurfaceAttrib::zeroed(),
            ffi::VASurfaceAttrib::zeroed(),
        ];
        attributes[0].type_ = ffi::VASurfaceAttribPixelFormat;
        attributes[0].flags = ffi::VA_SURFACE_ATTRIB_SETTABLE;
        attributes[0].value.type_ = ffi::VAGenericValueTypeInteger;
        attributes[0].value.value.i = i32::from_ne_bytes(kind.fourcc.to_ne_bytes());
        attributes[1].type_ = ffi::VASurfaceAttribMemoryType;
        attributes[1].flags = ffi::VA_SURFACE_ATTRIB_SETTABLE;
        attributes[1].value.type_ = ffi::VAGenericValueTypeInteger;
        attributes[1].value.value.i =
            i32::from_ne_bytes(ffi::VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2.to_ne_bytes());
        let count = if kind == YUY2 { 2 } else { 1 };
        let mut id = ffi::VA_INVALID_SURFACE;
        check(
            "vaCreateSurfaces",
            // SAFETY: one surface into `id`, with `count` valid attributes.
            display.call(|raw| unsafe {
                ffi::vaCreateSurfaces(
                    raw,
                    kind.rt_format,
                    size.0,
                    size.1,
                    &mut id,
                    1,
                    attributes.as_mut_ptr(),
                    count,
                )
            })?,
        )?;
        Ok(Self {
            display: display.clone_ref(),
            id,
            kind,
            size,
            written,
        })
    }

    /// Lends `use_samples` the surface's image's layout and its bytes,
    /// mapped by `vaDeriveImage` where the driver can, else copied out by
    /// `vaGetImage` into an image of the driver's own format for `kind`;
    /// a driver call's error as `malfunction` makes it. Both the map of a
    /// derived image and `vaGetImage` may wait for the GPU, and are made on
    /// the display's call thread (`super::calls`).
    fn view(
        &self,
        kind: SurfaceKind,
        image_formats: &[ffi::VAImageFormat],
        malfunction: &dyn Fn(Error) -> Failed,
        use_samples: impl FnOnce(&Layout, &[u8]) -> Result<(), Failed>,
    ) -> Result<Readback, Failed> {
        let (mut image, readback) = match VaImage::derive(&self.display, self.id) {
            Ok(image) => (image, Readback::Derived),
            Err(error @ Error::Stalled(_)) => return Err(malfunction(error)),
            Err(_) => {
                let format = image_formats
                    .iter()
                    .copied()
                    .find(|f| f.fourcc == kind.fourcc)
                    .unwrap_or(ffi::VAImageFormat {
                        fourcc: kind.fourcc,
                        ..ffi::VAImageFormat::default()
                    });
                let image = VaImage::create(&self.display, format, self.size.0, self.size.1)
                    .map_err(malfunction)?;
                image
                    .get_from(self.id, self.size.0, self.size.1)
                    .map_err(malfunction)?;
                (image, Readback::Copied)
            }
        };
        let mapping = image.map().map_err(malfunction)?;
        use_samples(&Layout::of_image(mapping.image().raw()), mapping.bytes())?;
        Ok(readback)
    }
}

impl Drop for OwnedSurface {
    fn drop(&mut self) {
        // Its context is destroyed first, since the context is made after
        // it, and drops before it.
        super::readback::destroy_surface(&self.display, self.id);
    }
}

/// Where a surface's samples are in the bytes it was read into: its
/// fourcc, and each plane's offset and pitch.
#[derive(Clone, Copy, Debug)]
struct Layout {
    fourcc: u32,
    planes: usize,
    offsets: [usize; 3],
    pitches: [usize; 3],
}

impl Layout {
    /// A driver's image's.
    fn of_image(image: &ffi::VAImage) -> Self {
        Self {
            fourcc: image.format.fourcc,
            planes: usize::try_from(image.num_planes).unwrap_or(0).min(3),
            offsets: image
                .offsets
                .map(|offset| usize::try_from(offset).unwrap_or(usize::MAX)),
            pitches: image
                .pitches
                .map(|pitch| usize::try_from(pitch).unwrap_or(0)),
        }
    }

    /// The same layout from its `top`th row down: each plane's offset moved
    /// down to its row for that row, NV12's chroma a row each two. `None`
    /// where `top` splits a chroma row, or past the address space.
    fn at_row(mut self, top: u32) -> Option<Self> {
        for n in 0..self.planes {
            let down = if self.fourcc == ffi::VA_FOURCC_NV12 && n == 1 {
                2
            } else {
                1
            };
            if !top.is_multiple_of(down) {
                return None;
            }
            let rows = usize::try_from(top / down).ok()?;
            let skipped = rows.checked_mul(self.pitches[n])?;
            self.offsets[n] = self.offsets[n].checked_add(skipped)?;
        }
        Some(self)
    }

    /// A host surface's, of `format`, its planes at `planes`' offsets and
    /// pitches.
    fn of_host(format: HostFormat, planes: &[(usize, usize)]) -> Self {
        let mut layout = Self {
            fourcc: format.fourcc(),
            planes: planes.len().min(3),
            offsets: [usize::MAX; 3],
            pitches: [0; 3],
        };
        for (n, &(offset, pitch)) in planes.iter().take(3).enumerate() {
            layout.offsets[n] = offset;
            layout.pitches[n] = pitch;
        }
        layout
    }
}

/// A BGRA image's pixels.
fn bgra_of<'a>(layout: &Layout, bytes: &'a [u8]) -> Result<SampleView<'a>, JpegError> {
    if layout.fourcc != proc::VA_FOURCC_BGRA || layout.planes < 1 {
        return Err(JpegError::Malfunction(format!(
            "a converted image of {}",
            Fourcc(layout.fourcc)
        )));
    }
    Ok(SampleView {
        bytes,
        offset: layout.offsets[0],
        pitch: layout.pitches[0],
        step: 4,
    })
}

/// Views of an image's samples, by its format.
fn samples_of<'a>(
    layout: &Layout,
    bytes: &'a [u8],
    subsampling: JpegSubsampling,
) -> Result<DecodedSamples<'a>, JpegError> {
    let plane = |index: usize, extra: usize, step: usize| SampleView {
        bytes,
        offset: layout.offsets[index].saturating_add(extra),
        pitch: layout.pitches[index],
        step,
    };
    let planes = layout.planes;
    let (luma, chroma) = match (layout.fourcc, planes) {
        (ffi::VA_FOURCC_NV12, 2..) => (plane(0, 0, 1), Some([plane(1, 0, 2), plane(1, 1, 2)])),
        (va::VA_FOURCC_YUY2, 1..) => (plane(0, 0, 2), Some([plane(0, 1, 4), plane(0, 3, 4)])),
        (va::VA_FOURCC_444P, 3..) => (plane(0, 0, 1), Some([plane(1, 0, 1), plane(2, 0, 1)])),
        (va::VA_FOURCC_Y800, 1..) => (plane(0, 0, 1), None),
        (fourcc, _) => {
            return Err(JpegError::Malfunction(format!(
                "a decoded image of {} in {planes} planes",
                Fourcc(fourcc)
            )));
        }
    };
    if chroma.is_some() != subsampling.chroma_divisors().is_some() {
        return Err(JpegError::Malfunction(format!(
            "a {:?} image decoded as {}",
            subsampling,
            Fourcc(layout.fourcc)
        )));
    }
    Ok(DecodedSamples {
        subsampling,
        luma,
        chroma,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_video_codec::jpeg::{DEFAULT_AC_LUMINANCE, DEFAULT_DC_LUMINANCE, HuffmanTable};

    fn segment(out: &mut Vec<u8>, marker: u8, payload: &[u8]) {
        let length = u16::try_from(payload.len() + 2).unwrap_or(u16::MAX);
        out.extend_from_slice(&[0xff, marker]);
        out.extend_from_slice(&length.to_be_bytes());
        out.extend_from_slice(payload);
    }

    fn dht(out: &mut Vec<u8>, class: u8, id: u8, table: &HuffmanTable<'_>) {
        let mut payload = vec![(class << 4) | id];
        payload.extend_from_slice(&table.counts);
        payload.extend_from_slice(table.values);
        segment(out, 0xc4, &payload);
    }

    #[test]
    fn a_decode_is_copied_or_converted_as_its_layout_and_scale_say() {
        use reims_vgpu_video_core::jpeg::JpegOutputFormat as F;
        use reims_vgpu_video_core::jpeg::JpegSubsampling as S;
        let route = |subsampling, format, scale| {
            Route::of(&JpegDecode {
                data: &[],
                width: 640,
                height: 480,
                subsampling,
                region: PixelRect {
                    x: 0,
                    y: 0,
                    width: 640,
                    height: 480,
                },
                scale,
                format,
            })
        };
        let yuv422 = Route::Nv12WithChroma {
            stretch: (1, 2),
            layout: S::Yuv422,
        };
        let yuv444 = Route::Nv12WithChroma {
            stretch: (2, 2),
            layout: S::Yuv444,
        };
        // The stream's own sampling, and grey into any YCbCr layout, at 1.
        assert_eq!(route(S::Yuv420, F::Biplanar420, 1), Ok(Route::Copy));
        assert_eq!(route(S::Yuv422, F::Uyvy, 1), Ok(Route::Copy));
        assert_eq!(route(S::Grey, F::Biplanar444, 1), Ok(Route::Copy));
        // Scaled, or of another sampling.
        assert_eq!(route(S::Yuv420, F::Biplanar420, 2), Ok(Route::Nv12));
        assert_eq!(route(S::Yuv444, F::Biplanar420, 1), Ok(Route::Nv12));
        assert_eq!(route(S::Grey, F::Yuyv, 8), Ok(Route::Nv12));
        assert_eq!(route(S::Yuv420, F::Biplanar422, 1), Ok(yuv422));
        assert_eq!(route(S::Yuv422, F::Yuyv, 4), Ok(yuv422));
        assert_eq!(route(S::Yuv444, F::Biplanar444, 2), Ok(yuv444));
        assert_eq!(route(S::Grey, F::Bgra, 1), Ok(Route::Bgra));
        assert_eq!(route(S::Yuv422, F::Bgra, 2), Ok(Route::Bgra));
        // What Apple's hardware refuses.
        for subsampling in [S::Yuv420, S::Yuv422] {
            assert!(matches!(
                route(subsampling, F::Biplanar444, 1),
                Err(JpegError::UnsupportedFormat(_))
            ));
        }
    }

    #[test]
    fn bands_cover_the_rows_once_from_their_top() {
        let rows = |top, count| Rows { top, count };
        assert_eq!(
            bands(rows(8, 100), Some(32)),
            [rows(8, 32), rows(40, 32), rows(72, 32), rows(104, 4)]
        );
        // Read whole, one band from the output's first row.
        assert_eq!(bands(rows(8, 100), None), [rows(0, 108)]);
        assert_eq!(bands(rows(0, 16), Some(16)), [rows(0, 16)]);
    }

    /// Bytes 0, 1, 2, ... of a pseudo-random sequence.
    fn noise(len: usize) -> Vec<u8> {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state.to_le_bytes()[0]
            })
            .collect()
    }

    /// Rows `rows` of a surface of `layout` in `bytes`, its planes' rows
    /// copied to the top of a surface of their own, rounded up to even
    /// within `surface_rows`, as a strip's run copies them; and that
    /// strip's layout.
    fn strip_of(layout: &Layout, bytes: &[u8], surface_rows: u32, rows: Rows) -> (Layout, Vec<u8>) {
        let count = rows.count.next_multiple_of(2).min(surface_rows - rows.top) as usize;
        let mut strip = *layout;
        let mut out = Vec::new();
        for n in 0..layout.planes {
            let down = if layout.fourcc == ffi::VA_FOURCC_NV12 && n == 1 {
                2
            } else {
                1
            };
            let pitch = layout.pitches[n];
            let first = rows.top as usize / down;
            strip.offsets[n] = out.len();
            for row in first..first + count.div_ceil(down) {
                let at = layout.offsets[n] + row * pitch;
                out.extend_from_slice(&bytes[at..at + pitch]);
            }
        }
        (strip, out)
    }

    /// A decode's region written into a padded destination of `format` from
    /// a surface of `fourcc`, whole and in bands of `each` rows, through
    /// strips and through the whole surface's rows: the same bytes,
    /// padding left as it was. A 4:2:2 strip is read as the BGRA band a run
    /// copies it into ([`strip_format`]) holds it: its rows at the top, at the
    /// surface's pitch, laid out as YUY2.
    #[test]
    fn an_output_written_in_bands_is_the_output_written_whole() {
        use reims_vgpu_video_core::jpeg::JpegOutputFormat as F;
        use reims_vgpu_video_core::jpeg::JpegSubsampling as S;
        // An 80x61 image in a 96x64 surface, 128 bytes a row (256 for
        // YUY2's two a pixel), its region two MCUs in and down and to its
        // edges: 64x45, its last row odd.
        let (height, rows) = (61u32, 64u32);
        let cases = [
            (ffi::VA_FOURCC_NV12, S::Yuv420, F::Biplanar420),
            (va::VA_FOURCC_Y800, S::Grey, F::Yuyv),
            (va::VA_FOURCC_Y800, S::Grey, F::Biplanar420),
            (va::VA_FOURCC_444P, S::Yuv444, F::Biplanar444),
            (va::VA_FOURCC_YUY2, S::Yuv422, F::Biplanar422),
            (va::VA_FOURCC_YUY2, S::Yuv422, F::Yuyv),
            (va::VA_FOURCC_YUY2, S::Yuv422, F::Uyvy),
        ];
        for (fourcc, subsampling, format) in cases {
            let planes = match fourcc {
                ffi::VA_FOURCC_NV12 => 2,
                va::VA_FOURCC_444P => 3,
                _ => 1,
            };
            let pitch = if fourcc == va::VA_FOURCC_YUY2 {
                256
            } else {
                128
            };
            let plane = pitch * rows as usize;
            let layout = Layout {
                fourcc,
                planes,
                offsets: [0, plane, 2 * plane],
                pitches: [pitch; 3],
            };
            let bytes = noise(plane * planes);
            let request = JpegDecode {
                data: &[],
                width: 80,
                height,
                subsampling,
                region: PixelRect {
                    x: 16,
                    y: 16,
                    width: 64,
                    height: 45,
                },
                scale: 1,
                format,
            };
            let destination = || match format {
                F::Yuyv | F::Uyvy => vec![vec![0x5a; 140 * 45]],
                F::Biplanar420 => vec![vec![0x5a; 72 * 45], vec![0x5a; 72 * 23]],
                F::Biplanar422 => vec![vec![0x5a; 72 * 45], vec![0x5a; 72 * 45]],
                _ => vec![vec![0x5a; 72 * 45], vec![0x5a; 136 * 45]],
            };
            let strides = match format {
                F::Yuyv | F::Uyvy => vec![140],
                F::Biplanar420 | F::Biplanar422 => vec![72, 72],
                _ => vec![72, 136],
            };
            fn lend<'a>(
                buffers: &'a mut [Vec<u8>],
                strides: &[usize],
            ) -> Vec<PlaneDestination<'a>> {
                (buffers.iter_mut().zip(strides))
                    .map(|(bytes, &bytes_per_row)| PlaneDestination {
                        rows: u32::try_from(bytes.len() / bytes_per_row).unwrap_or(0),
                        bytes,
                        bytes_per_row,
                    })
                    .collect()
            }
            let mut whole = destination();
            let samples = samples_of(&layout, &bytes, subsampling).expect("samples");
            write_samples(&samples, &request, &mut lend(&mut whole, &strides))
                .expect("written whole");
            let band = Rows { top: 16, count: 45 };
            for each in [16, 32, 48] {
                let (mut striped, mut offset) = (destination(), destination());
                for band in bands(band, Some(each)) {
                    let (strip, strip_bytes) = strip_of(&layout, &bytes, rows, band);
                    let samples = samples_of(&strip, &strip_bytes, subsampling).expect("samples");
                    write_band(&samples, &request, band, &mut lend(&mut striped, &strides))
                        .unwrap_or_else(|_| panic!("a strip of {band:?}"));
                    let from = layout.at_row(band.top).expect("a layout from the row");
                    let samples = samples_of(&from, &bytes, subsampling).expect("samples");
                    write_band(&samples, &request, band, &mut lend(&mut offset, &strides))
                        .unwrap_or_else(|_| panic!("rows {band:?}"));
                }
                assert!(striped == whole, "{format:?} in strips of {each}");
                assert!(offset == whole, "{format:?} in bands of {each}");
            }
        }
    }

    /// The BGRA band a YUY2 output's strip is copied into holds its rows'
    /// bytes at YUY2's offsets and pitches, at every width, odd ones among
    /// them: so does the alias the runs copy from, at the surface's pitch.
    #[test]
    fn a_yuy2_strip_is_bgra_of_half_the_width_in_the_same_bytes() {
        for width in [1, 2, 17, 64, 333, 336, 8192, 16384] {
            let (format, half) = strip_format(HostFormat::Yuy2, width);
            assert_eq!((format, half), (HostFormat::Bgra, width.div_ceil(2)));
            // One plane at the start of each: the same bytes at two heights
            // are the same pitch.
            for rows in [1, 2720] {
                assert_eq!(
                    HostFormat::Bgra.bytes(half, rows),
                    HostFormat::Yuy2.bytes(width, rows),
                    "{width}x{rows}"
                );
            }
        }
        // Every other output's strips are of its own format and width.
        for format in [
            HostFormat::Nv12,
            HostFormat::Yuv444,
            HostFormat::Y800,
            HostFormat::Bgra,
        ] {
            assert_eq!(strip_format(format, 333), (format, 333));
        }
    }

    #[test]
    fn a_band_holds_its_rows_within_the_strips_bytes() {
        // 8192x8192 4:2:0 in four strips; 16384x16000 grey in eight.
        assert_eq!(band_rows(HostFormat::Nv12, 8192, 1), 2720);
        assert_eq!(band_rows(HostFormat::Y800, 16384, 1), 2048);
        // 8192x8192 4:2:2 in four.
        assert_eq!(band_rows(HostFormat::Yuy2, 8192, 1), 2048);
        for (format, width, stretch) in [
            (HostFormat::Nv12, 16384, 1),
            (HostFormat::Nv12, 8192, 2),
            (HostFormat::Yuv444, 16384, 1),
            (HostFormat::Bgra, 16384, 1),
            (HostFormat::Yuy2, 16384, 1),
            (HostFormat::Y800, 333, 1),
        ] {
            let rows = band_rows(format, width, stretch);
            assert_eq!(rows % 16, 0);
            let bytes = format.bytes(width, rows * stretch).expect("bytes");
            assert!(
                bytes <= STRIP_BYTES,
                "{format:?} {width} x{stretch}: {rows} rows"
            );
        }
    }

    #[test]
    fn the_least_recently_used_surfaces_past_the_budget_are_destroyed() {
        // Within it, none.
        assert_eq!(past_budget(&[(10, false), (20, true)], 30), [false, false]);
        // Past it, the oldest first, until what is left is within it.
        assert_eq!(
            past_budget(&[(10, false), (10, false), (20, false)], 30),
            [true, false, false]
        );
        // The last decode's host surfaces stay, whatever their size, and
        // its others do not.
        assert_eq!(
            past_budget(&[(10, false), (50, true), (20, false)], 30),
            [true, false, true]
        );
    }

    #[test]
    fn a_conversion_shrinks_by_half_a_side_a_run_at_most() {
        // 4:2:0 at 1/2 is one run, at 1/8 three.
        assert_eq!(steps((640, 480), (2, 2), (320, 240)), []);
        assert_eq!(
            steps((640, 480), (2, 2), (80, 60)),
            [(320, 240), (160, 120)]
        );
        // Odd sides round up, as the output's do.
        assert_eq!(steps((333, 211), (2, 2), (42, 27)), [(167, 106), (84, 53)]);
        // 4:4:4's chroma goes to NV12's at its own size first.
        assert_eq!(steps((640, 480), (1, 1), (320, 240)), [(640, 480)]);
        assert_eq!(steps((640, 480), (1, 1), (640, 480)), []);
        // 4:2:2's chroma is full height.
        assert_eq!(
            steps((640, 480), (2, 1), (160, 120)),
            [(320, 480), (160, 240)]
        );
        // Growing is one run.
        assert_eq!(steps((640, 480), (2, 2), (640, 960)), []);
        // Into RGB, the chroma stays at the output's.
        assert_eq!(chain((640, 480), (2, 2), (320, 240), true), []);
        assert_eq!(chain((640, 480), (2, 2), (160, 120), true), [(320, 240)]);
        assert_eq!(
            chain((640, 480), (2, 1), (160, 120), true),
            [(320, 480), (320, 240)]
        );
        assert_eq!(
            chain((640, 480), (1, 1), (80, 60), true),
            [(640, 480), (320, 240), (160, 120)]
        );
        assert_eq!(chain((640, 480), (1, 1), (640, 480), true), []);
    }

    /// A grey 8x8 baseline image, written here from T.81's syntax,
    /// whose scan names tables 0 and which also defines a DC table 1 of
    /// category 255 and an AC table 1 of size 15, neither of which the
    /// parser checks, as no scan names them. Its entropy-coded data is
    /// arbitrary: nothing here decodes it.
    #[test]
    fn only_the_huffman_tables_the_scan_names_are_loaded() {
        let mut image = vec![0xff, 0xd8];
        let mut dqt = vec![0];
        dqt.extend_from_slice(&[1; 64]);
        segment(&mut image, 0xdb, &dqt);
        // SOF0: 8 bits, 8x8, one component (id 1, 1x1, table 0).
        segment(&mut image, 0xc0, &[8, 0, 8, 0, 8, 1, 1, 0x11, 0]);
        dht(&mut image, 0, 0, &DEFAULT_DC_LUMINANCE);
        dht(&mut image, 1, 0, &DEFAULT_AC_LUMINANCE);
        let mut counts = [0; 16];
        counts[0] = 1;
        let category_255 = HuffmanTable {
            counts,
            values: &[255],
        };
        let size_15 = HuffmanTable {
            counts,
            values: &[0x0f],
        };
        dht(&mut image, 0, 1, &category_255);
        dht(&mut image, 1, 1, &size_15);
        // SOS: component 1 on tables 0 and 0, spectral selection 0..63.
        segment(&mut image, 0xda, &[1, 1, 0x00, 0, 63, 0]);
        image.extend_from_slice(&[0x12, 0x34, 0x56, 0x78]);
        image.extend_from_slice(&[0xff, 0xd9]);

        let parsed = match syntax::parse(&image) {
            Ok(parsed) => parsed,
            Err(e) => unreachable!("the image parses: {e:?}"),
        };
        assert!(parsed.dc_tables[1].is_some() && parsed.ac_tables[1].is_some());
        let parameters = match Parameters::of(&parsed) {
            Ok(parameters) => parameters,
            Err(e) => unreachable!("parameters are made: {e:?}"),
        };
        assert_eq!(parameters.huffman.load_huffman_table, [1, 0]);
        let unused = &parameters.huffman.huffman_table[1];
        assert_eq!(unused.dc_values, [0; 12], "table 1 stays unloaded");
        assert!(unused.ac_values.iter().all(|&v| v == 0));
        assert_eq!(
            parameters.huffman.huffman_table[0].num_dc_codes,
            DEFAULT_DC_LUMINANCE.counts
        );
    }
}
