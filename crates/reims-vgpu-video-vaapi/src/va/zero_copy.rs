//! Zero copy on VA-API: a decoded frame written into the guest's buffer by
//! video-processing runs, with nothing for the CPU to copy
//! (`Frame::write_to_guest`).
//!
//! The buffer is imported once per mapping and size of frame, in strips:
//! each strip an even number of the frame's rows, a dma-buf the host makes
//! of their luma rows' pages and their chroma rows' pages (core's
//! `DeviceMemory::dma_buf_of`; reims-vgpu's `os/linux/device_lend.rs`, a
//! udmabuf of the memfd behind the guest's RAM, the pages in the order the
//! bytes lie in), and a linear VA surface over it at the buffer's own
//! pitches (`DRM_PRIME_2`, one object of both planes: iHD takes no import
//! of an object a plane, `super::import_check`). The strips are as few as
//! udmabuf's limits allow: the whole buffer where it fits in one dma-buf,
//! else as many as the host's count of the pages' runs calls for against
//! its `list_limit` and `size_limit_mb`. A guest's buffer lies nearly one
//! run a page, so at the stock limits (1,024 runs, 64 MiB) a 1080p NV12
//! buffer takes one strip and a 4K one three. Where the driver writes rows
//! of whole alignments alone, its luma plane at the dma-buf's first byte and
//! its chroma plane whole rows after it ([`ImportWrite::whole_rows`]), each
//! strip but the last is a multiple of the rows after which a luma row
//! starts a page again, so that every strip's luma plane starts its dma-buf
//! as the buffer's first does. They share one video-processing context, and
//! are kept by the mapping's identity in core's [`DeviceImports`] until the
//! mapping goes.
//! Each frame is then one run a strip, from the decode surface's rows into
//! it, and their syncs: the staged copy back's run, into the guest's pages
//! instead of ours.
//!
//! What the driver takes is learned as its display opens (`super::import_check`): the least
//! alignment of rows and of planes from a dma-buf's first byte at which its imports of each
//! format are taken and written right, and whether only in rows of whole alignments. Buffers
//! whose rows or planes are not at it, or with such a driver whose rows of samples are not whole
//! alignments or whose planes are not a page's start and whole rows after it, are declined, each
//! frame then copied by the CPU, and so is every buffer of a format the driver writes no import
//! of. Measured on radeonsi, on the RX 7900 XTX and the Radeon 780M (Mesa 26.2.4, 2026-10-24),
//! with a probe of our own that checked each import byte for byte:
//! - rows of a multiple of 256 bytes, and planes that start 256 bytes apart from the dma-buf's
//!   first: rows of 1920 bytes, and a luma plane 64 bytes into a page, were refused
//!   (`VA_STATUS_ERROR_ALLOCATION_FAILED`); a chroma plane 256 bytes past a page took;
//! - an even size: at 641x361 every plane's samples past the first chroma pair came out wrong,
//!   at 8 and 10 bits, though nothing was written past the planes. Frames of an odd width or
//!   height are declined on every driver, as the strips here hold whole chroma rows;
//! - nothing else: in every case taken, every byte of the buffer that is not one of the frame's
//!   samples, row padding included, stayed as it was.
//!
//! Strips write what one run does, measured on the same GPUs (2026-10-25,
//! udmabuf at its stock limits): every frame of 21 test streams at 720p,
//! 1080p and 4K, 8 and 10 bits, matched its reference, written in up to
//! five strips, and the decoding threads used 0.4 to 1.4 ms of the CPU a 4K
//! frame, against 1.3 to 3.4 ms with the CPU copying. Each strip's run has
//! a cost of its own, so the strips are as few as the limits allow
//! (docs/video/vaapi.md, "What it saves").
//!
//! A driver whose runs answer success and write nothing, as NVIDIA's VA-API
//! driver's do (docs/video/bugs.md, U16), fails the display's check, and is
//! lent no guest buffer.
//!
//! A frame's decode and its runs are waited for until one deadline
//! (`super::wait`). Runs past it may still write the guest's pages: the
//! frame fails as a malfunction, never declined to the CPU's copy, which
//! would write the buffer too, and the import is kept until the GPU is done
//! with each strip. The frame's pool is stalled with it
//! (`super::surface`).

use std::os::fd::{AsFd, AsRawFd, OwnedFd};
use std::sync::Arc;

use reims_vgpu_video_core::backend::DeviceWrite;
use reims_vgpu_video_core::device_memory::{
    DeviceImports, DmaBufError, GuestTarget, SurfaceDmaBuf, TargetPlane,
};
use reims_vgpu_video_core::runtime::MonoTime;
use reims_vgpu_video_core::sync::Mutex;

use super::caps::{Config, Context, Entrypoint, Profile};
use super::display::{Display, Error, check};
use super::ffi;
use super::frame::Frame;
use super::import_check::ImportWrite;
use super::surface::SurfaceFormat;
use super::vpp_ffi as vpp;
use super::wait::{self, Work};

/// The most imports a backend keeps at once: a guest's decoders use a few
/// pools of 10 to 30 buffers each, and each import is a VA surface a strip
/// and a context (core's [`DeviceImports`] drops the one used longest ago
/// past it).
const MOST_IMPORTS: usize = 96;

/// The most strips one buffer is imported in: an 8192x4352 P010 buffer, the
/// largest the decoders make, takes 17 or 18 at udmabuf's stock limits.
const MOST_STRIPS: usize = 64;

/// A backend's imports of guest buffers, shared by its sessions.
pub(crate) struct GuestImports {
    imports: DeviceImports<GuestImport>,
}

/// No imports yet, for a new backend on `display`.
pub(crate) fn imports(display: &Display) -> Arc<GuestImports> {
    Arc::new(GuestImports {
        imports: DeviceImports::new(display.runtime(), "vaapi", MOST_IMPORTS),
    })
}

/// A guest buffer imported for frames of one size and format: its rows in
/// strips, each a linear VA surface over a dma-buf of them, and a
/// video-processing context on them.
pub(crate) struct GuestImport {
    display: Display,
    /// Destroyed before the surfaces.
    context: Option<Context>,
    /// Top to bottom.
    strips: Vec<Strip>,
    /// One run at a time into it: a guest may lend one buffer to two
    /// sessions at once.
    running: Mutex<()>,
}

/// Some of a buffer's rows, imported: a linear VA surface over a dma-buf of
/// its luma rows and their chroma rows.
struct Strip {
    surface: ffi::VASurfaceID,
    /// Its first row, of the frame's, and its rows: both even.
    top: u32,
    rows: u32,
    /// The dma-buf of its rows, which the surface's buffer object holds too.
    dma_buf: OwnedFd,
}

impl Drop for GuestImport {
    fn drop(&mut self) {
        self.context.take();
        // Their context is gone, and no run into them is left, as every user
        // holds this import.
        destroy(&self.display, &self.strips);
    }
}

/// Why a buffer is copied by the CPU instead: core logs each reason once a
/// session, so each is the same for every buffer of a layout.
fn declined(why: impl Into<String>) -> DeviceWrite {
    DeviceWrite::Declined(why.into())
}

/// Writes `frame` into `target` by one video-processing run a strip (module
/// docs), the `size` of it from the frame's crop's first sample: the
/// buffer's whole size within the frame's coded picture, as its buffers hold
/// the coded rows and columns a decode writes in place
/// (`Frame::write_to_guest`). [`DeviceWrite::Declined`] where the driver
/// takes no such buffer, or the host lends no dma-buf of it; an `Err` where
/// the frame did not decode, or the GPU did not finish the frame by its
/// deadline.
pub(crate) fn write(
    frame: &Frame,
    target: &GuestTarget<'_>,
    imports: &GuestImports,
    (width, height): (u32, u32),
) -> Result<DeviceWrite, Error> {
    let format = frame.surface.format();
    let display = frame.surface.display();
    let write = match display.writes_imports(format) {
        Ok(write) => write,
        Err(why) => {
            return Ok(declined(format!(
                "the driver's video processing writes no {} import right: {why}",
                format.name()
            )));
        }
    };
    if width % 2 != 0 || height % 2 != 0 {
        return Ok(declined(format!(
            "a frame of {width}x{height}: the strips hold whole chroma rows, and radeonsi's \
             video processing wrote an odd size wrongly"
        )));
    }
    if width == 0 || height == 0 {
        return Ok(declined("an empty frame"));
    }
    if target.pixel_format().bytes_per_sample() != format.bytes_per_sample() {
        return Ok(declined(format!(
            "a {} buffer for {} frames",
            target.pixel_format().fourcc(),
            format.name()
        )));
    }
    let alignment = write.alignment;
    let planes = target.planes();
    if let Some(plane) = planes
        .iter()
        .find(|plane| plane.bytes_per_row % alignment != 0)
    {
        return Ok(declined(format!(
            "rows of {} bytes: the driver imports rows of a multiple of {alignment} bytes alone",
            plane.bytes_per_row
        )));
    }
    let samples = usize::try_from(width).unwrap_or(usize::MAX) * format.bytes_per_sample();
    if write.whole_rows && !samples.is_multiple_of(alignment) {
        return Ok(declined(format!(
            "rows of {samples} bytes of samples: the driver writes rows of whole {alignment} \
             bytes alone"
        )));
    }
    let rect = frame.rect;
    let (Ok(x), Ok(y), Ok(w), Ok(_)) = (
        i16::try_from(rect.x),
        i16::try_from(rect.y),
        u16::try_from(width),
        u16::try_from(height),
    ) else {
        return Ok(declined("a frame too large for a video-processing region"));
    };
    // The import is of the frame's size and the buffer's format: one buffer
    // given frames of another size gets another.
    let variant =
        u64::from(width) << 32 | u64::from(height) << 1 | u64::from(format == SurfaceFormat::P010);
    let memory = target.memory();
    let import = match imports.imports.get_or_make(
        &memory.identity(),
        variant,
        || GuestImport::new(display, target, format, (width, height), write, None),
        || "the buffer's mapping went while it was held".to_owned(),
    ) {
        Ok(import) => import,
        Err(why) => return Ok(declined(why)),
    };
    // The decode, waited for: a failure here is the frame's, as a read
    // back's would be.
    let deadline = wait::current(display);
    frame.surface.sync_by(deadline)?;
    match import.run(frame.surface.id(), x, y, w, deadline) {
        Ok(()) => Ok(DeviceWrite::Written),
        Err(Error::Stalled(what)) => {
            let works = import.works();
            let held = Arc::clone(&import);
            wait::keep(display, "a stalled write into a guest buffer", works, held);
            frame.surface.stalled_if(Err(Error::Stalled(what)))
        }
        Err(error) => Ok(declined(format!(
            "the video processing run failed: {error}"
        ))),
    }
}

impl GuestImport {
    /// `target`'s buffer imported for `format` frames of `size`, as `write`
    /// says the driver takes it, in as few strips as udmabuf's limits allow
    /// (or in `forced`, a test's), with a context on them; `Err` says why
    /// not, for the log.
    fn new(
        display: &Display,
        target: &GuestTarget<'_>,
        format: SurfaceFormat,
        (width, height): (u32, u32),
        write: ImportWrite,
        forced: Option<usize>,
    ) -> Result<Self, String> {
        let mut count = forced.unwrap_or(1);
        let strips = loop {
            match strips(display, target, format, (width, height), count, write) {
                Ok(strips) => break strips,
                Err(Cut::Refused(why)) => return Err(why),
                Err(Cut::TooLarge { why, .. }) if forced.is_some() => {
                    return Err(format!("not in {count} strips: {why}"));
                }
                Err(Cut::TooLarge {
                    pieces,
                    bytes,
                    most_pieces,
                    most_bytes,
                    why,
                }) => {
                    // Each strip as many times smaller as its pages were past
                    // the limits, one more strip at the least.
                    let times = |had: u64, most: u64| {
                        (u64::try_from(count).unwrap_or(u64::MAX).saturating_mul(had))
                            .div_ceil(most.max(1))
                            .try_into()
                            .unwrap_or(usize::MAX)
                    };
                    let next = (count + 1)
                        .max(times(pieces, most_pieces))
                        .max(times(bytes, most_bytes));
                    let most = MOST_STRIPS.min(usize::try_from(height / 2).unwrap_or(usize::MAX));
                    if next > most {
                        return Err(format!(
                            "no dma-buf of the buffer in {MOST_STRIPS} strips or fewer: {why}"
                        ));
                    }
                    count = next;
                }
            }
        };
        let mut surfaces: Vec<ffi::VASurfaceID> = strips.iter().map(|s| s.surface).collect();
        let made = Config::new(display, Profile::NONE, Entrypoint::VIDEO_PROC, &mut [])
            .and_then(|config| Context::new(display, config, width, height, &mut surfaces));
        let context = match made {
            Ok(made) => made,
            Err(error) => {
                destroy(display, &strips);
                return Err(format!("no video processing into the buffer: {error}"));
            }
        };
        let [luma, chroma] = target.planes();
        log::debug!(
            "a guest buffer imported for {} frames of {width}x{height} in {} strips, rows of {} \
             and {}",
            format.name(),
            strips.len(),
            luma.bytes_per_row,
            chroma.bytes_per_row
        );
        Ok(Self {
            display: display.clone_ref(),
            context: Some(context),
            strips,
            running: Mutex::new(()),
        })
    }

    /// The work on its strips the GPU may be doing.
    fn works(&self) -> Vec<Work> {
        (self.strips.iter())
            .map(|strip| Work::Surface(strip.surface))
            .collect()
    }

    /// One run from `source`'s region of `width` columns and the frame's
    /// rows at (`x`, `y`) into each strip, a wait for their dma-bufs' fences,
    /// and their syncs, until `deadline`: then the frame is in the guest's
    /// pages.
    fn run(
        &self,
        source: ffi::VASurfaceID,
        x: i16,
        y: i16,
        width: u16,
        deadline: MonoTime,
    ) -> Result<(), Error> {
        let Some(context) = &self.context else {
            return Err(Error::Unsupported("an import with no context".into()));
        };
        let _running = self.running.lock();
        for strip in &self.strips {
            let (Some(top), Ok(rows)) = (
                i16::try_from(strip.top)
                    .ok()
                    .and_then(|top| y.checked_add(top)),
                u16::try_from(strip.rows),
            ) else {
                return Err(Error::Unsupported(
                    "a strip past a video-processing region".into(),
                ));
            };
            let from = vpp::VARectangle {
                x,
                y: top,
                width,
                height: rows,
            };
            let to = vpp::VARectangle {
                x: 0,
                y: 0,
                width,
                height: rows,
            };
            let mut params = vpp::VAProcPipelineParameterBuffer::zeroed();
            params.surface = source;
            params.surface_region = &from;
            params.output_region = &to;
            // The same standard and range both sides: a plain copy.
            params.surface_color_standard = vpp::VAProcColorStandardBT709;
            params.output_color_standard = vpp::VAProcColorStandardBT709;
            let buffer = super::buffer::Buffer::with_value(
                &self.display,
                context.id,
                vpp::VAProcPipelineParameterBufferType,
                &params,
            )?;
            super::buffer::render_picture(&self.display, context.id, strip.surface, &[&[&buffer]])?;
        }
        // The runs' fences, waited for with no lock of the driver's held,
        // so that the display's other calls go on meanwhile (`fence.rs`).
        super::fence::wait_for_writes(
            &**self.display.runtime(),
            deadline,
            (self.strips.iter()).map(|strip| strip.dma_buf.as_fd()),
        );
        for strip in &self.strips {
            wait::wait_for(
                &self.display,
                Work::Surface(strip.surface),
                deadline,
                "a write into a guest buffer in time",
            )?;
        }
        Ok(())
    }
}

/// The display's import check's run (`super::import_check`): `source`, from its first sample,
/// written into `target` as a frame of `source`'s format and `target`'s size is, imported as
/// `write` says (in `strips` strips where given), and waited for. `Err` says why not. A run past
/// its deadline keeps its import, and the pages under it, until the GPU is done with it.
pub(crate) fn trial(
    display: &Display,
    source: &super::surface::Surface,
    target: &GuestTarget<'_>,
    write: ImportWrite,
    strips: Option<usize>,
) -> Result<(), String> {
    let (width, height) = (target.width(), target.height());
    let format = source.format();
    let import = GuestImport::new(display, target, format, (width, height), write, strips)?;
    let Ok(columns) = u16::try_from(width) else {
        return Err("a trial too wide for a video-processing region".to_owned());
    };
    match import.run(source.id(), 0, 0, columns, wait::deadline(display)) {
        Ok(()) => Ok(()),
        Err(Error::Stalled(what)) => {
            let works = import.works();
            wait::keep(display, "the import check's run", works, import);
            Err(format!("the run did not finish: {what}"))
        }
        Err(error) => Err(format!("the run failed: {error}")),
    }
}

/// Why a buffer was not cut into strips: its strips' pages are more than
/// one dma-buf holds, so that more strips may fit, or it is not to be had.
enum Cut {
    TooLarge {
        pieces: u64,
        bytes: u64,
        most_pieces: u64,
        most_bytes: u64,
        why: String,
    },
    Refused(String),
}

/// `target`'s rows imported in `count` strips for `format` frames of
/// `size` (module docs): each an even number of the frame's rows, a dma-buf
/// of their luma rows and their chroma rows, and a linear VA surface over it
/// at the guest's own pitches, as `write` says the driver takes it.
fn strips(
    display: &Display,
    target: &GuestTarget<'_>,
    format: SurfaceFormat,
    (width, height): (u32, u32),
    count: usize,
    write: ImportWrite,
) -> Result<Vec<Strip>, Cut> {
    let refused = |why: &str| Cut::Refused(why.to_owned());
    let memory = target.memory();
    let [luma, chroma] = target.planes();
    let count = u32::try_from(count).map_err(|_| refused("too many strips"))?;
    // With rows of whole alignments, each strip's luma plane at a page's start: its rows a
    // multiple of those after which a luma row starts a page again.
    let unit = if write.whole_rows {
        u32::try_from(PAGE / gcd(luma.bytes_per_row, PAGE)).unwrap_or(u32::MAX)
    } else {
        1
    };
    let mut made: Vec<Strip> = Vec::new();
    for (top, rows) in strip_rows(height, count, unit) {
        let range = |plane: &TargetPlane, top: u32, rows: u32| {
            let at = (plane.bytes_per_row)
                .checked_mul(usize::try_from(top).ok()?)?
                .checked_add(plane.offset)?;
            Some((
                at,
                plane
                    .bytes_per_row
                    .checked_mul(usize::try_from(rows).ok()?)?,
            ))
        };
        let ranges = range(&luma, top, rows).zip(range(&chroma, top / 2, rows / 2));
        let Some((luma_rows, chroma_rows)) = ranges else {
            destroy(display, &made);
            return Err(refused("a buffer past the address space"));
        };
        let strip = match memory.dma_buf_of(&[luma_rows, chroma_rows]) {
            Ok(dma_buf) => Strip::new(
                display,
                format,
                (width, rows),
                top,
                dma_buf,
                [luma, chroma],
                write,
            )
            .map_err(Cut::Refused),
            Err(DmaBufError::TooLarge {
                pieces,
                bytes,
                most_pieces,
                most_bytes,
                why,
            }) => Err(Cut::TooLarge {
                pieces,
                bytes,
                most_pieces,
                most_bytes,
                why: format!("no dma-buf of the buffer: {why}"),
            }),
            Err(DmaBufError::Refused(why)) => {
                Err(Cut::Refused(format!("no dma-buf of the buffer: {why}")))
            }
        };
        match strip {
            Ok(strip) => made.push(strip),
            Err(cut) => {
                destroy(display, &made);
                return Err(cut);
            }
        }
    }
    Ok(made)
}

/// The `height` rows of an even frame in `count` strips at most, top to
/// bottom, each its first row and its rows, both even, so that every strip
/// starts on a chroma row, and a multiple of `unit`; all but the last as
/// tall.
fn strip_rows(height: u32, count: u32, unit: u32) -> impl Iterator<Item = (u32, u32)> {
    let unit = unit.max(1).next_multiple_of(2);
    let each = height
        .div_ceil(count.max(1))
        .next_multiple_of(unit)
        .max(unit);
    (0..height)
        .step_by(usize::try_from(each).unwrap_or(usize::MAX))
        .map(move |top| (top, each.min(height - top)))
}

/// The greatest common divisor of `a` and `b`.
pub(super) fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

/// The bytes of a page, which a dma-buf's first byte starts.
pub(super) const PAGE: usize = 4096;

/// Destroys the strips' surfaces, made on `display` and used by nothing.
fn destroy(display: &Display, strips: &[Strip]) {
    for strip in strips {
        super::readback::destroy_surface(display, strip.surface);
    }
}

impl Strip {
    /// A linear VA surface of `size` over `dma_buf`, the strip's luma rows
    /// and its chroma rows, at `planes`' pitches, as `write` says the driver
    /// takes it: each plane at its alignment from the dma-buf's first byte,
    /// and the luma plane at that byte where it says whole rows; `Err` says
    /// why not.
    #[allow(clippy::too_many_arguments)]
    fn new(
        display: &Display,
        format: SurfaceFormat,
        (width, rows): (u32, u32),
        top: u32,
        dma_buf: SurfaceDmaBuf,
        [luma, chroma]: [TargetPlane; 2],
        write: ImportWrite,
    ) -> Result<Self, String> {
        let past = || "a buffer past 4 GiB".to_owned();
        let alignment = write.alignment;
        let (Some(&luma_at), Some(&chroma_at)) = (dma_buf.offsets.first(), dma_buf.offsets.get(1))
        else {
            return Err("the host's dma-buf holds no planes".to_owned());
        };
        if [luma_at, chroma_at]
            .iter()
            .any(|at| at % alignment as u64 != 0)
        {
            return Err(format!(
                "planes {} and {} bytes into their first page: the driver imports planes \
                 {alignment} bytes apart alone",
                luma_at % 4096,
                chroma_at % 4096
            ));
        }
        if write.whole_rows && luma_at != 0 {
            return Err(format!(
                "a luma plane {} bytes into its first page: the driver writes a luma plane at a \
                 page's start alone, its chroma plane whole rows after it",
                luma_at % 4096
            ));
        }
        let luma_pitch = luma.bytes_per_row as u64;
        if write.whole_rows
            && (luma_pitch == 0
                || chroma.bytes_per_row != luma.bytes_per_row
                || !chroma_at.is_multiple_of(luma_pitch))
        {
            return Err(format!(
                "a chroma plane {chroma_at} bytes past the luma's first, at rows of {} bytes: the \
                 driver writes a luma plane at a page's start alone, its chroma plane whole rows \
                 after it",
                chroma.bytes_per_row
            ));
        }
        let (Ok(luma_at), Ok(chroma_at)) = (u32::try_from(luma_at), u32::try_from(chroma_at))
        else {
            return Err(past());
        };
        let pitch = |bytes: usize| u32::try_from(bytes).unwrap_or(u32::MAX);
        let mut descriptor = ffi::VADRMPRIMESurfaceDescriptor {
            fourcc: format.fourcc(),
            width,
            height: rows,
            num_objects: 1,
            num_layers: 1,
            ..Default::default()
        };
        descriptor.objects[0] = ffi::VADRMPRIMESurfaceObject {
            fd: dma_buf.fd.as_raw_fd(),
            size: u32::try_from(dma_buf.size).map_err(|_| past())?,
            drm_format_modifier: ffi::DRM_FORMAT_MOD_LINEAR,
        };
        descriptor.layers[0] = ffi::VADRMPRIMESurfaceLayer {
            drm_format: match format {
                SurfaceFormat::Nv12 => ffi::DRM_FORMAT_NV12,
                SurfaceFormat::P010 => ffi::DRM_FORMAT_P010,
            },
            num_planes: 2,
            object_index: [0, 0, 0, 0],
            offset: [luma_at, chroma_at, 0, 0],
            pitch: [pitch(luma.bytes_per_row), pitch(chroma.bytes_per_row), 0, 0],
        };
        let mut attributes = [
            ffi::VASurfaceAttrib::zeroed(),
            ffi::VASurfaceAttrib::zeroed(),
        ];
        attributes[0].type_ = ffi::VASurfaceAttribMemoryType;
        attributes[0].flags = ffi::VA_SURFACE_ATTRIB_SETTABLE;
        attributes[0].value.type_ = ffi::VAGenericValueTypeInteger;
        attributes[0].value.value.i =
            i32::from_ne_bytes(ffi::VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2.to_ne_bytes());
        attributes[1].type_ = ffi::VASurfaceAttribExternalBufferDescriptor;
        attributes[1].flags = ffi::VA_SURFACE_ATTRIB_SETTABLE;
        attributes[1].value.type_ = ffi::VAGenericValueTypePointer;
        attributes[1].value.value.p = (&raw mut descriptor).cast();
        let mut surface = ffi::VA_INVALID_SURFACE;
        let created = display
            .call(|raw| {
                // SAFETY: one surface out; the attributes and the
                // descriptor they point at live for the call, and the
                // dma-bufs for the surface's life (the driver holds what it
                // imports).
                unsafe {
                    ffi::vaCreateSurfaces(
                        raw,
                        format.rt_format(),
                        width,
                        rows,
                        &mut surface,
                        1,
                        attributes.as_mut_ptr(),
                        2,
                    )
                }
            })
            .map_err(|error| format!("no import of the buffer: {error}"))?;
        check("vaCreateSurfaces (a guest buffer)", created)
            .map_err(|error| format!("the driver refused the buffer: {error}"))?;
        Ok(Self {
            surface,
            top,
            rows,
            dma_buf: dma_buf.fd,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Strips cover the frame's rows once, top to bottom, each starting and
    /// ending on a chroma row, no more of them than asked for
    #[test]
    fn strips_cover_the_rows_once_on_chroma_rows() {
        for height in [2, 16, 360, 720, 1080, 2160, 4352] {
            for count in 1..=40 {
                for unit in [1, 2, 16, 32, 64] {
                    let strips: Vec<_> = strip_rows(height, count, unit).collect();
                    assert!(strips.len() <= count as usize, "{height} in {count}");
                    let mut next = 0;
                    for (top, rows) in strips {
                        assert_eq!(top, next);
                        assert!(
                            top % 2 == 0 && rows % 2 == 0 && rows > 0,
                            "{height} in {count}"
                        );
                        assert_eq!(top % unit, 0, "{height} in {count} of {unit}");
                        next = top + rows;
                    }
                    assert_eq!(next, height);
                }
            }
        }
        assert_eq!(
            strip_rows(2160, 7, 1).collect::<Vec<_>>(),
            [
                (0, 310),
                (310, 310),
                (620, 310),
                (930, 310),
                (1240, 310),
                (1550, 310),
                (1860, 300)
            ]
        );
    }

    /// With rows of whole alignments, a strip's luma rows start a page wherever the buffer's
    /// first does: its rows a multiple of those after which a row of `pitch` bytes starts one.
    #[test]
    fn strips_of_whole_rows_start_their_luma_rows_at_a_page() {
        for pitch in [1920, 2048, 2176, 3840, 3904, 7680, 1344] {
            let unit = u32::try_from(PAGE / gcd(pitch, PAGE)).unwrap();
            for count in 1..=8 {
                for (top, _) in strip_rows(2160, count, unit) {
                    assert_eq!(top as usize * pitch % PAGE, 0, "rows of {pitch} in {count}");
                }
            }
        }
        assert_eq!(gcd(3840, PAGE), 256);
        assert_eq!(gcd(1344, PAGE), 64);
    }
}
