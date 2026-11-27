//! Zero copy on VA-API: a decoded frame written into the guest's buffer by
//! video-processing runs, with nothing for the CPU to copy
//! (`Frame::write_to_guest`).
//!
//! The buffer is imported once per mapping, place of its planes in it and
//! size of frame ([`Variant`]), in strips:
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
//! as the buffer's first does. The import is kept by the mapping's identity
//! in core's [`DeviceImports`] until the mapping goes, and runs on the one
//! video-processing context of its frame's size ([`Contexts`]), which every
//! import of the size shares: made with the size's first import, and
//! destroyed with its last. A driver's first run on a context costs it a few
//! milliseconds its later runs do not, and a guest's decoder makes a new pool
//! of buffers for each stream: measured on the Radeon 780M (radeonsi, Mesa
//! 26.2.6, 2026-11-19, the start of three streams in Safari), a new import's
//! first run was issued in 5.3 ms at the median on a context of its own, 43
//! of them, against 94 us on its size's shared one; later runs took 81 and
//! 88 us.
//! Each frame is then one run a strip, from the decode surface's rows into
//! it, and their syncs: the staged copy back's run, into the guest's pages
//! instead of ours. The runs are issued right behind the frame's decode,
//! which is waited for after their issue and before theirs.
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
//! A 4:4:4 frame, an XYUV surface's, is written the same way into a buffer of
//! one plane of its packed pixels ([`PackedTarget`], `Frame::write_4_4_4_to`),
//! where the display's check passed XYUV: each strip a dma-buf of its rows, and
//! a linear one-plane surface over it (`DRM_FORMAT_XYUV8888`). reims-vgpu's
//! VA-API arm writes so into a scratch of its runtime device's, whose GPU then
//! splits the pixels into the two planes of the guest's 4:4:4 surface.
//!
//! A frame's decode and its runs are waited for until one deadline
//! (`super::wait`). Runs past it may still write the guest's pages: the
//! frame fails as a malfunction, never declined to the CPU's copy, which
//! would write the buffer too, and the import is kept until the GPU is done
//! with each strip. The frame's pool is stalled with it
//! (`super::surface`).

#[cfg(unix)]
use std::os::fd::{AsFd, AsRawFd, OwnedFd};
use std::sync::{Arc, Weak};

use crate::core::backend::{Chroma, DeviceWrite};
use crate::core::device_memory::{DeviceImports, DeviceMemory, GuestTarget, TargetPlane};
#[cfg(unix)]
use crate::core::device_memory::{DmaBufError, SurfaceDmaBuf};
use crate::core::runtime::MonoTime;
use crate::core::sync::Mutex;

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
/// (core's [`DeviceImports`] drops the one used longest ago past it).
const MOST_IMPORTS: usize = 96;

/// The most strips one buffer is imported in: an 8192x4352 P010 buffer, the
/// largest the decoders make, takes 17 or 18 at udmabuf's stock limits.
const MOST_STRIPS: usize = 64;

/// A backend's imports of guest buffers, shared by its sessions.
pub(crate) struct GuestImports {
    imports: DeviceImports<GuestImport, Variant>,
    contexts: Contexts,
}

/// The video-processing contexts a backend's imports run on, one a frame size
/// (module docs): each import holds its size's, and this table none, so a
/// context goes with the last import of its size, and holds nothing past
/// them.
#[derive(Default)]
struct Contexts(Mutex<Vec<SizeContext>>);

/// A frame size, and the context its imports hold.
type SizeContext = ((u32, u32), Weak<SharedContext>);

impl Contexts {
    /// The context of frames of `size`: the one its imports hold, or a new one
    /// where none holds one.
    fn of(&self, display: &Display, size: (u32, u32)) -> Result<Arc<SharedContext>, Error> {
        let mut contexts = self.0.lock();
        contexts.retain(|(_, context)| context.strong_count() > 0);
        let held = (contexts.iter())
            .find(|(of, _)| *of == size)
            .and_then(|(_, context)| context.upgrade());
        if let Some(context) = held {
            return Ok(context);
        }
        let context = Arc::new(SharedContext::new(display, size)?);
        contexts.push((size, Arc::downgrade(&context)));
        Ok(context)
    }

    /// How many contexts imports hold.
    #[cfg(test)]
    fn held(&self) -> usize {
        (self.0.lock().iter())
            .filter(|(_, context)| context.strong_count() > 0)
            .count()
    }
}

/// A video-processing context on no surface, which runs into any import of its
/// frame size.
struct SharedContext {
    context: Context,
    /// One import's runs issued at a time: a context takes one picture at a
    /// time, and the imports sharing it may run at once.
    issue: Mutex<()>,
}

impl SharedContext {
    /// A context for frames of `size`.
    fn new(display: &Display, (width, height): (u32, u32)) -> Result<Self, Error> {
        let context = Config::new(display, Profile::NONE, Entrypoint::VIDEO_PROC, &mut [])
            .and_then(|config| Context::new(display, config, width, height, &mut []))?;
        Ok(Self {
            context,
            issue: Mutex::new(()),
        })
    }
}

/// What a buffer's import is made for besides its mapping: the frame's size
/// and format, and where the buffer's planes lie in the mapping, with their
/// pitches. One mapping may hold more than one buffer of a size, each with an
/// import of its own: macOS 26.7.2's HEIC decodes in Preview and Freeform
/// write the tiles of a grid image side by side into one surface, a buffer a
/// tile at the tile's offset (measured 2026-11-11, on the Iris Xe).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Variant {
    size: (u32, u32),
    format: SurfaceFormat,
    /// Each plane's offset in the mapping and its bytes from one row to the
    /// next: luma's, then chroma's, or a packed plane's and (0, 0).
    planes: [(usize, usize); 2],
}

impl Variant {
    /// The import `planes` take for `format` frames of `size`.
    fn of(planes: &ImportPlanes<'_>, format: SurfaceFormat, size: (u32, u32)) -> Self {
        let place = |plane: &TargetPlane| (plane.offset, plane.bytes_per_row);
        Self {
            size,
            format,
            planes: [
                place(&planes.first),
                planes.chroma.as_ref().map_or((0, 0), place),
            ],
        }
    }
}

/// A buffer a frame of a 4:4:4 stream is written into as its XYUV surface holds it
/// (`Frame::write_4_4_4_to`): one plane of packed pixels, four bytes each, Cr, Cb, Y and a byte
/// that holds nothing, in that order in memory, at its offset in its memory and its rows its
/// bytes a row apart, checked against the memory as [`GuestTarget`]'s planes are: each row holds
/// its pixels, and the last ends inside the memory.
pub struct PackedTarget<'a> {
    memory: &'a dyn DeviceMemory,
    plane: TargetPlane,
    width: u32,
    height: u32,
}

impl std::fmt::Debug for PackedTarget<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedTarget")
            .field("identity", &self.memory.identity())
            .field("length", &self.memory.length())
            .field("width", &self.width)
            .field("height", &self.height)
            .field("plane", &self.plane)
            .finish()
    }
}

impl<'a> PackedTarget<'a> {
    /// The bytes of a pixel.
    pub const PIXEL_BYTES: usize = 4;

    /// `width` by `height` pixels at `offset` of `memory`, rows `bytes_per_row` apart, checked
    /// (type docs); `Err` says why not.
    pub fn new(
        memory: &'a dyn DeviceMemory,
        offset: usize,
        bytes_per_row: usize,
        (width, height): (u32, u32),
    ) -> Result<Self, String> {
        let row_bytes = usize::try_from(width)
            .ok()
            .and_then(|width| width.checked_mul(Self::PIXEL_BYTES))
            .ok_or("rows too long")?;
        if bytes_per_row < row_bytes {
            return Err(format!(
                "{bytes_per_row} bytes a row hold no {row_bytes} bytes of pixels"
            ));
        }
        let end = usize::try_from(height)
            .ok()
            .and_then(|height| bytes_per_row.checked_mul(height))
            .and_then(|bytes| bytes.checked_add(offset))
            .ok_or("past the address space")?;
        let length = memory.length();
        if end > length {
            return Err(format!("ends at {end}, past the memory's {length} bytes"));
        }
        Ok(Self {
            memory,
            plane: TargetPlane {
                offset,
                bytes_per_row,
                rows: height,
                row_bytes,
            },
            width,
            height,
        })
    }

    /// Its width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Its height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Its plane.
    pub fn plane(&self) -> TargetPlane {
        self.plane
    }
}

/// What an import is made over: a buffer's memory, its size, and the planes of its frames'
/// format in it, a [`GuestTarget`]'s luma and interleaved chroma, or a [`PackedTarget`]'s one.
#[derive(Clone, Copy)]
pub(crate) struct ImportPlanes<'a> {
    memory: &'a dyn DeviceMemory,
    size: (u32, u32),
    /// Luma, or the packed pixels.
    first: TargetPlane,
    chroma: Option<TargetPlane>,
}

impl<'a> From<&GuestTarget<'a>> for ImportPlanes<'a> {
    fn from(target: &GuestTarget<'a>) -> Self {
        let [luma, chroma] = target.planes();
        Self {
            memory: target.memory(),
            size: (target.width(), target.height()),
            first: luma,
            chroma: Some(chroma),
        }
    }
}

impl<'a> From<&PackedTarget<'a>> for ImportPlanes<'a> {
    fn from(target: &PackedTarget<'a>) -> Self {
        Self {
            memory: target.memory,
            size: (target.width, target.height),
            first: target.plane,
            chroma: None,
        }
    }
}

impl ImportPlanes<'_> {
    /// Its planes, first to last.
    fn each(&self) -> impl Iterator<Item = &TargetPlane> {
        std::iter::once(&self.first).chain(self.chroma.as_ref())
    }
}

/// No imports yet, for a new backend on `display`.
pub(crate) fn imports(display: &Display) -> Arc<GuestImports> {
    Arc::new(GuestImports {
        imports: DeviceImports::new(display.runtime(), "vaapi", MOST_IMPORTS),
        contexts: Contexts::default(),
    })
}

/// A guest buffer imported for frames of one size and format: its rows in
/// strips, each a linear VA surface over a dma-buf of them, and the
/// video-processing context it runs on.
pub(crate) struct GuestImport {
    display: Display,
    /// Its size's ([`Contexts`]), or its own (the import check's); let go
    /// before the surfaces.
    context: Option<Arc<SharedContext>>,
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
    #[cfg(unix)]
    dma_buf: OwnedFd,
}

impl Drop for GuestImport {
    fn drop(&mut self) {
        self.context.take();
        // No run into them is left, as every user holds this import. The
        // context, where other imports still hold it, may outlive them.
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
/// deadline. The runs are issued before the decode is waited for, so a frame
/// that did not decode leaves the buffer holding what they copied from its
/// decode surface. `decoded` is called once the decode is done, the runs
/// issued and not yet waited for (`Frame::write_to_guest_telling`).
pub(crate) fn write(
    frame: &Frame,
    target: &GuestTarget<'_>,
    imports: &GuestImports,
    size: (u32, u32),
    decoded_now: &mut dyn FnMut(),
) -> Result<DeviceWrite, Error> {
    let format = frame.surface.format();
    // The driver's 4:2:0 frames, NV12 and P010, into two planes; a 4:4:4 one, XYUV, is
    // written whole into a packed buffer alone (`write_packed`), and the range extensions'
    // others are read back.
    if !matches!(format, SurfaceFormat::Nv12 | SurfaceFormat::P010)
        || target.pixel_format().bytes_per_sample() != format.bytes_per_sample()
        || target.pixel_format().chroma() != Chroma::Yuv420
    {
        return Ok(declined(format!(
            "a {} buffer for {} frames",
            target.pixel_format().fourcc(),
            format.name()
        )));
    }
    write_planes(
        frame,
        ImportPlanes::from(target),
        imports,
        size,
        decoded_now,
    )
}

/// [`write()`] for a 4:4:4 frame, an XYUV surface's, into `target`, a buffer of its packed
/// pixels (`Frame::write_4_4_4_to`).
pub(crate) fn write_packed(
    frame: &Frame,
    target: &PackedTarget<'_>,
    imports: &GuestImports,
    size: (u32, u32),
    decoded_now: &mut dyn FnMut(),
) -> Result<DeviceWrite, Error> {
    let format = frame.surface.format();
    if format != SurfaceFormat::Xyuv {
        return Ok(declined(format!(
            "a buffer of packed 4:4:4 pixels for {} frames",
            format.name()
        )));
    }
    write_planes(
        frame,
        ImportPlanes::from(target),
        imports,
        size,
        decoded_now,
    )
}

/// [`write()`] and [`write_packed`], once the buffer's format is the frame's.
fn write_planes(
    frame: &Frame,
    target: ImportPlanes<'_>,
    imports: &GuestImports,
    (width, height): (u32, u32),
    decoded_now: &mut dyn FnMut(),
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
    let alignment = write.alignment;
    if let Some(plane) = target
        .each()
        .find(|plane| plane.bytes_per_row % alignment != 0)
    {
        return Ok(declined(format!(
            "rows of {} bytes: the driver imports rows of a multiple of {alignment} bytes alone",
            plane.bytes_per_row
        )));
    }
    let samples = (usize::try_from(width).unwrap_or(usize::MAX))
        .saturating_mul(format.first_plane_bytes_per_pixel());
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
    // The import is of the frame's size and format and the buffer's planes:
    // one buffer given frames of another size gets another, and so does
    // another buffer in the same mapping.
    let memory = target.memory;
    let import = match imports.imports.get_or_make(
        &memory.identity(),
        Variant::of(&target, format, (width, height)),
        || {
            let contexts = Some(&imports.contexts);
            GuestImport::new(
                display,
                target,
                format,
                (width, height),
                write,
                None,
                contexts,
            )
        },
        || "the buffer's mapping went while it was held".to_owned(),
    ) {
        Ok(import) => import,
        Err(why) => return Ok(declined(why)),
    };
    // The runs go to the driver right behind the decode, before it is waited
    // for, so that the GPU starts them as it finishes the decode rather than
    // once the CPU has heard of it: the driver orders a run that reads the
    // decode's surface after the decode
    // (`runs_issued_right_behind_a_decode_write_what_it_decoded`). The decode
    // is waited for between their issue and their wait.
    let deadline = wait::current(display);
    let mut decoded = Ok(());
    let ran = import.run(frame.surface.id(), x, y, w, deadline, || {
        decoded = frame.surface.sync_by(deadline);
        if decoded.is_ok() {
            decoded_now();
        }
    });
    if let Err(Error::Stalled(_)) = &ran {
        let works = import.works();
        let held = Arc::clone(&import);
        wait::keep(display, "a stalled write into a guest buffer", works, held);
    }
    // The decode: a failure here is the frame's, as a read back's would be,
    // whatever the runs behind it wrote.
    decoded?;
    match ran {
        Ok(()) => Ok(DeviceWrite::Written),
        Err(Error::Stalled(what)) => frame.surface.stalled_if(Err(Error::Stalled(what))),
        Err(error) => Ok(declined(format!(
            "the video processing run failed: {error}"
        ))),
    }
}

impl GuestImport {
    /// `target`'s buffer imported for `format` frames of `size`, as `write`
    /// says the driver takes it, in as few strips as udmabuf's limits allow
    /// (or in `forced`, a test's), running on its size's context of
    /// `contexts`, or, without them, on a context of its own made as those
    /// are (the import check's); `Err` says why not, for the log.
    fn new(
        display: &Display,
        target: ImportPlanes<'_>,
        format: SurfaceFormat,
        (width, height): (u32, u32),
        write: ImportWrite,
        forced: Option<usize>,
        contexts: Option<&Contexts>,
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
        let made = match contexts {
            Some(contexts) => contexts.of(display, (width, height)),
            None => SharedContext::new(display, (width, height)).map(Arc::new),
        };
        let context = match made {
            Ok(made) => made,
            Err(error) => {
                destroy(display, &strips);
                return Err(format!("no video processing into the buffer: {error}"));
            }
        };
        let places: Vec<String> = (target.each())
            .map(|plane| format!("at {} in rows of {}", plane.offset, plane.bytes_per_row))
            .collect();
        log::debug!(
            "a guest buffer imported for {} frames of {width}x{height} in {} strips, its planes \
             {} of its mapping",
            format.name(),
            strips.len(),
            places.join(" and "),
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
    /// rows at (`x`, `y`) into each strip, `between` once they are all
    /// issued, then a wait for their dma-bufs' fences and their syncs, until
    /// `deadline`: then the frame is in the guest's pages. `between` is not
    /// called where a run's issue fails.
    fn run(
        &self,
        source: ffi::VASurfaceID,
        x: i16,
        y: i16,
        width: u16,
        deadline: MonoTime,
        between: impl FnOnce(),
    ) -> Result<(), Error> {
        let Some(shared) = &self.context else {
            return Err(Error::Unsupported("an import with no context".into()));
        };
        let context = &shared.context;
        let _running = self.running.lock();
        let issue = shared.issue.lock();
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
        drop(issue);
        between();
        // The runs' fences, waited for with no lock of the driver's held,
        // so that the display's other calls go on meanwhile (`fence.rs`).
        #[cfg(unix)]
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
    target: ImportPlanes<'_>,
    write: ImportWrite,
    strips: Option<usize>,
) -> Result<(), String> {
    let (width, height) = target.size;
    let format = source.format();
    let import = GuestImport::new(
        display,
        target,
        format,
        (width, height),
        write,
        strips,
        None,
    )?;
    let Ok(columns) = u16::try_from(width) else {
        return Err("a trial too wide for a video-processing region".to_owned());
    };
    match import.run(source.id(), 0, 0, columns, wait::deadline(display), || {}) {
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
/// of their rows of each plane (luma and chroma, or packed pixels), and a
/// linear VA surface over it at the buffer's own pitches, as `write` says the
/// driver takes it.
#[cfg(unix)]
fn strips(
    display: &Display,
    target: ImportPlanes<'_>,
    format: SurfaceFormat,
    (width, height): (u32, u32),
    count: usize,
    write: ImportWrite,
) -> Result<Vec<Strip>, Cut> {
    let refused = |why: &str| Cut::Refused(why.to_owned());
    let memory = target.memory;
    let first = target.first;
    let count = u32::try_from(count).map_err(|_| refused("too many strips"))?;
    // With rows of whole alignments, each strip's first plane at a page's start: its rows a
    // multiple of those after which a row of it starts a page again.
    let unit = if write.whole_rows {
        u32::try_from(PAGE / gcd(first.bytes_per_row, PAGE)).unwrap_or(u32::MAX)
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
        // A chroma plane's rows are half the luma's.
        let ranges: Option<Vec<(usize, usize)>> = std::iter::once(range(&first, top, rows))
            .chain((target.chroma.as_ref()).map(|chroma| range(chroma, top / 2, rows / 2)))
            .collect();
        let Some(ranges) = ranges else {
            destroy(display, &made);
            return Err(refused("a buffer past the address space"));
        };
        let strip = match memory.dma_buf_of(&ranges) {
            Ok(dma_buf) => Strip::new(
                display,
                format,
                (width, rows),
                top,
                dma_buf,
                (first, target.chroma),
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

/// On Windows, which makes no dma-buf: no strip is had (the display lends no import there, so
/// [`write()`] declines before it asks).
#[cfg(windows)]
fn strips(
    _display: &Display,
    _target: ImportPlanes<'_>,
    _format: SurfaceFormat,
    _size: (u32, u32),
    _count: usize,
    _write: ImportWrite,
) -> Result<Vec<Strip>, Cut> {
    Err(Cut::Refused(
        "this host makes no dma-buf of a buffer".to_owned(),
    ))
}

#[cfg(unix)]
impl Strip {
    /// A linear VA surface of `size` over `dma_buf`, the strip's rows of its
    /// planes, luma then chroma or packed pixels alone, at their pitches, as
    /// `write` says the driver takes it: each plane at its alignment from the
    /// dma-buf's first byte, and the first plane at that byte where it says
    /// whole rows; `Err` says why not.
    #[allow(clippy::too_many_arguments)]
    fn new(
        display: &Display,
        format: SurfaceFormat,
        (width, rows): (u32, u32),
        top: u32,
        dma_buf: SurfaceDmaBuf,
        (luma, chroma): (TargetPlane, Option<TargetPlane>),
        write: ImportWrite,
    ) -> Result<Self, String> {
        let past = || "a buffer past 4 GiB".to_owned();
        let drm_format = match (format, chroma) {
            (SurfaceFormat::Nv12, Some(_)) => ffi::DRM_FORMAT_NV12,
            (SurfaceFormat::P010, Some(_)) => ffi::DRM_FORMAT_P010,
            (SurfaceFormat::Xyuv, None) => ffi::DRM_FORMAT_XYUV8888,
            (format, chroma) => {
                return Err(format!(
                    "no import of {} in {} planes",
                    format.name(),
                    1 + usize::from(chroma.is_some())
                ));
            }
        };
        let alignment = write.alignment;
        let planes = 1 + usize::from(chroma.is_some());
        let Some(offsets) = dma_buf.offsets.get(..planes) else {
            return Err("the host's dma-buf holds no planes".to_owned());
        };
        if offsets.iter().any(|at| at % alignment as u64 != 0) {
            let into: Vec<String> = (offsets.iter()).map(|at| (at % 4096).to_string()).collect();
            return Err(format!(
                "planes {} bytes into their first page: the driver imports planes {alignment} \
                 bytes apart alone",
                into.join(" and ")
            ));
        }
        let luma_at = offsets[0];
        if write.whole_rows && luma_at != 0 {
            return Err(format!(
                "a first plane {} bytes into its first page: the driver writes a first plane at \
                 a page's start alone, a chroma plane whole rows after it",
                luma_at % 4096
            ));
        }
        let chroma_at = offsets.get(1).copied().unwrap_or(0);
        let luma_pitch = luma.bytes_per_row as u64;
        if let Some(chroma) = chroma
            && write.whole_rows
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
        let chroma_pitch = chroma.map_or(0, |chroma| pitch(chroma.bytes_per_row));
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
            drm_format,
            num_planes: u32::try_from(planes).unwrap_or(2),
            object_index: [0, 0, 0, 0],
            offset: [luma_at, chroma_at, 0, 0],
            pitch: [pitch(luma.bytes_per_row), chroma_pitch, 0, 0],
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

    /// A frame's runs, issued right behind its decode with no wait between them
    /// ([`write`]), write what the read back of the same frame copies once its
    /// decode is done: the driver orders a run that reads a surface after the
    /// decode that writes it. Each picture of a 4K H.264 stream with no
    /// reordering, written as soon as its decode is issued, into one target the
    /// earlier pictures' import is kept for, so that nothing comes between the
    /// two but the run's issue; at least one decode must still be running as its
    /// runs are issued, or the order went untested.
    #[test]
    #[cfg(unix)]
    #[ignore = "needs a VA-API driver that decodes H.264 at 4K, on its GPU: run on a VA-API host"]
    fn runs_issued_right_behind_a_decode_write_what_it_decoded() {
        use super::super::frame::{GuestWrites, PlaneRows};
        use super::super::import_check::TrialMemory;
        use super::super::teardown::{SIZE, Stream, display};
        use crate::core::fourcc::FourCc;
        use crate::core::surfaces::{PlaneLayout, SurfaceLayout};

        let display = display();
        let stream = Stream::new(&display);
        let mut decoder = stream.decoder(&display);
        let writes = GuestWrites::new(&display);
        let (width, height) = (SIZE.0 as usize, SIZE.1 as usize);
        // Rows of 3840 bytes, a multiple of every alignment a driver's imports
        // take, and the chroma plane at a page, whole rows after the luma.
        let pitch = width;
        let chroma_at = pitch * height;
        let size = (chroma_at + pitch * height / 2).next_multiple_of(PAGE);
        let plane = |offset: usize, w: u32, h: u32| PlaneLayout {
            offset,
            bytes_per_row: pitch,
            width: w,
            height: h,
        };
        let layout = SurfaceLayout {
            pixel_format: FourCc::new(*b"420v"),
            width: SIZE.0,
            height: SIZE.1,
            planes: vec![
                plane(0, SIZE.0, SIZE.1),
                plane(chroma_at, SIZE.0 / 2, SIZE.1 / 2),
            ],
            bytes_per_row: 0,
        };
        let memory = TrialMemory::new(size).expect("the target's memory");
        let target = GuestTarget::new(&layout, &memory).expect("a target");
        let (mut written, mut running) = (0, 0);
        let mut check = |id: usize, frame: &Frame| {
            if Work::Surface(frame.surface.id())
                .ask_now(&display)
                .is_none()
            {
                running += 1;
            }
            match frame.write_to_guest(&target, &writes).expect("the write") {
                DeviceWrite::Written => {}
                DeviceWrite::Declined(why) => panic!("picture {id} declined: {why}"),
            }
            let mut luma = vec![0; pitch * height];
            let mut chroma = vec![0; pitch * height / 2];
            (frame.read_back().expect("a read back"))
                .copy_into(
                    PlaneRows {
                        bytes: &mut luma,
                        bytes_per_row: pitch,
                        rows: height,
                        row_bytes: width,
                    },
                    PlaneRows {
                        bytes: &mut chroma,
                        bytes_per_row: pitch,
                        rows: height / 2,
                        row_bytes: width,
                    },
                )
                .expect("the CPU's copy");
            let held = memory.memory.lock();
            let bytes = held.bytes();
            let first = ((0..luma.len()).find(|&i| bytes[i] != luma[i]))
                .map(|i| ("luma", i))
                .or_else(|| {
                    ((0..chroma.len()).find(|&i| bytes[chroma_at + i] != chroma[i]))
                        .map(|i| ("chroma", i))
                });
            assert_eq!(
                first, None,
                "picture {id}: the run's byte differs from the read back's"
            );
            written += 1;
        };
        for (id, sample) in stream.samples.iter().enumerate() {
            memory.memory.lock().bytes_mut().fill(0xa5);
            let frames = decoder
                .decode(sample, u64::try_from(id).expect("an id"))
                .expect("a decode");
            for frame in &frames {
                check(id, frame);
            }
        }
        for frame in &decoder.flush().expect("a flush") {
            check(stream.samples.len(), frame);
        }
        assert_eq!(written, stream.samples.len(), "every picture written");
        assert!(
            running > 0,
            "no decode was still running as its runs were issued"
        );
        eprintln!("{written} pictures written, {running} of them behind a decode still running");
    }

    /// The imports of one frame size run on one context, made on no surface,
    /// which goes with the last of them, and another size's on another: each
    /// import's run writes what its frame decoded, a new context's after the
    /// old one went as well.
    #[test]
    #[cfg(unix)]
    #[ignore = "needs a VA-API driver that decodes H.264 at 4K, on its GPU: run on a VA-API host"]
    fn the_imports_of_a_size_share_a_context_that_goes_with_the_last() {
        use super::super::frame::PlaneRows;
        use super::super::import_check::TrialMemory;
        use super::super::teardown::{SIZE, Stream, display};
        use crate::core::fourcc::FourCc;
        use crate::core::surfaces::{PlaneLayout, SurfaceLayout};

        let display = display();
        let stream = Stream::new(&display);
        let mut decoder = stream.decoder(&display);
        let mut frames = Vec::new();
        for (id, sample) in stream.samples.iter().enumerate() {
            frames = decoder
                .decode(sample, u64::try_from(id).expect("an id"))
                .expect("a decode");
            if !frames.is_empty() {
                break;
            }
        }
        let frame = frames.first().expect("a decoded frame");
        let format = frame.surface.format();
        let write = display.writes_imports(format).expect("imports written");
        // The frame's samples, read back by the CPU: rows of the frame's width.
        let (width, height) = (SIZE.0 as usize, SIZE.1 as usize);
        let mut luma = vec![0; width * height];
        let mut chroma = vec![0; width * height / 2];
        (frame.read_back().expect("a read back"))
            .copy_into(
                PlaneRows {
                    bytes: &mut luma,
                    bytes_per_row: width,
                    rows: height,
                    row_bytes: width,
                },
                PlaneRows {
                    bytes: &mut chroma,
                    bytes_per_row: width,
                    rows: height / 2,
                    row_bytes: width,
                },
            )
            .expect("the CPU's copy");
        // Buffers of rows of 3840 bytes, a multiple of every alignment a
        // driver's imports take, their chroma plane at a page, whole rows after
        // the luma: of the frame's size, or of 2048x1152 from its first sample.
        let pitch = width;
        let buffer = |(w, h): (u32, u32)| {
            let chroma_at = pitch * h as usize;
            let plane = |offset: usize, w: u32, h: u32| PlaneLayout {
                offset,
                bytes_per_row: pitch,
                width: w,
                height: h,
            };
            let layout = SurfaceLayout {
                pixel_format: FourCc::new(*b"420v"),
                width: w,
                height: h,
                planes: vec![plane(0, w, h), plane(chroma_at, w / 2, h / 2)],
                bytes_per_row: 0,
            };
            let size = (chroma_at + pitch * h as usize / 2).next_multiple_of(PAGE);
            (layout, TrialMemory::new(size).expect("a buffer's memory"))
        };
        let contexts = Contexts::default();
        let import = |(layout, memory): &(SurfaceLayout, TrialMemory)| {
            let target = GuestTarget::new(layout, memory).expect("a target");
            let size = (layout.width, layout.height);
            GuestImport::new(
                &display,
                ImportPlanes::from(&target),
                format,
                size,
                write,
                None,
                Some(&contexts),
            )
            .expect("an import")
        };
        // Its run from the frame's first sample, and its bytes against the
        // read back's.
        let check = |what: &str,
                     import: &GuestImport,
                     (layout, memory): &(SurfaceLayout, TrialMemory)| {
            memory.memory.lock().bytes_mut().fill(0xa5);
            let (x, y) = (
                i16::try_from(frame.rect.x).expect("x"),
                i16::try_from(frame.rect.y).expect("y"),
            );
            let columns = u16::try_from(layout.width).expect("columns");
            let deadline = wait::deadline(&display);
            (import.run(frame.surface.id(), x, y, columns, deadline, || {}))
                .unwrap_or_else(|error| panic!("{what}: the run: {error}"));
            let (w, h) = (layout.width as usize, layout.height as usize);
            let chroma_at = pitch * h;
            let held = memory.memory.lock();
            let bytes = held.bytes();
            let first = (0..h)
                .flat_map(|row| (0..w).map(move |column| (row, column)))
                .find(|&(row, column)| bytes[row * pitch + column] != luma[row * width + column])
                .map(|at| ("luma", at))
                .or_else(|| {
                    (0..h / 2)
                        .flat_map(|row| (0..w).map(move |column| (row, column)))
                        .find(|&(row, column)| {
                            bytes[chroma_at + row * pitch + column] != chroma[row * width + column]
                        })
                        .map(|at| ("chroma", at))
                });
            assert_eq!(
                first, None,
                "{what}: the run's byte differs from the read back's"
            );
        };
        let (a, b, c) = (buffer(SIZE), buffer(SIZE), buffer((2048, 1152)));
        let (ia, ib) = (import(&a), import(&b));
        let shared = |one: &GuestImport, other: &GuestImport| {
            Arc::ptr_eq(
                one.context.as_ref().expect("a context"),
                other.context.as_ref().expect("a context"),
            )
        };
        assert!(shared(&ia, &ib), "two imports of a size share its context");
        assert_eq!(contexts.held(), 1);
        let ic = import(&c);
        assert!(!shared(&ia, &ic), "another size runs on another context");
        assert_eq!(contexts.held(), 2);
        check("the first import", &ia, &a);
        check("the second import of its size", &ib, &b);
        check("the other size's import", &ic, &c);
        drop(ia);
        assert_eq!(contexts.held(), 2, "an import of the size still holds it");
        drop(ib);
        assert_eq!(
            contexts.held(),
            1,
            "the size's context went with its last import"
        );
        let again = import(&a);
        assert_eq!(contexts.held(), 2);
        check("an import on the size's new context", &again, &a);
    }
}
