//! What the driver's video processing writes into memory imported as a linear surface, learned
//! by a trial run as the display opens ([`learn`]), so that what the backend lends a driver
//! follows what the driver does, not its name.
//!
//! Two paths have the driver's video processing write frames into memory of ours or the
//! guest's, imported over a dma-buf as a linear VA surface:
//! - the staged copy back (`super::readback`);
//! - the writes into a guest's decode outputs in place (`super::zero_copy`).
//!
//! A driver may refuse such an import. Or it may take the import and its runs, answer success,
//! and write the wrong bytes or none: NVIDIA's VA-API driver (libva-nvidia-driver 0.0.18) wrote
//! nothing and answered success (docs/video/bugs.md, U16).
//!
//! So for each of the decoders' formats, NV12 and P010, the display runs one copy the way zero
//! copy runs it, through zero copy's own import ([`super::zero_copy::trial`]: one dma-buf of both
//! planes), into memory of its own laid out as a guest's buffer is:
//! - A picture holding every sample value is put into a surface of the decoders' kind by
//!   `vaPutImage`.
//! - One video-processing run copies it into the import.
//! - Every byte of the memory is checked: each plane's samples, and every byte around them, row
//!   padding included, left as it was.
//!
//! This is tried at each alignment of rows and planes, 64, 128 and 256 bytes, in turn. Each pitch
//! is an odd multiple of its alignment, each row's samples a part of one, and each plane starts
//! an odd multiple of the alignment into its page, so a driver that takes only a larger
//! alignment refuses the import or misplaces the picture. Where the run at an alignment answers
//! success and writes the picture wrong, a second run there tries rows of whole alignments, the
//! luma plane at the dma-buf's first byte and the chroma plane whole rows after it, a row past
//! the luma's last. The first that passes is the format's
//! ([`ImportWrites::of`], an [`ImportWrite`]). Where none passes, the driver is lent no import of
//! that format, and its frames come back through `vaGetImage`, with why in the log.
//!
//! **Measured** on the Iris Xe of an i9-13900H (iHD 26.3.5, 2026-11-07, the decoders' shapes in
//! the test below, and a libva program of our own): iHD writes an import's luma plane at the
//! dma-buf's first byte whatever its offset, and its chroma plane `(chroma - luma) / pitch` whole
//! rows after it, the rest of the offset dropped. A chroma plane given at the page past the luma's
//! rows landed at the row before it (640-byte rows: at 232960, 364 rows, given 233472; 1344-byte
//! rows: at 970368, given 970752); a luma plane given 256 bytes in, or two whole rows in, landed
//! at the first byte, its chroma plane as many rows after it as it was given after the luma.
//! Planes given that way, the luma at the first byte and the chroma whole rows after it, were
//! written right at every pitch of a multiple of 64 bytes. NV12 rows of samples not a multiple of
//! 64 bytes (160, 352 and 416 wide) were written out to the next 64 bytes, the padding zeroed. An
//! import of a dma-buf a plane was refused (`VA_STATUS_ERROR_ALLOCATION_FAILED`) or written wrong
//! at every shape, and a luma plane 64 or 128 bytes into its page failed its run
//! (`VA_STATUS_ERROR_OPERATION_FAILED`). docs/video/bugs.md, U26.
//!
//! **What it costs**, at each display's open, on the thread that opens it. **Measured** on the
//! Radeon 780M (radeonsi, Mesa 26.2.6, 2026-11-07, `examples/vaapi-probe`, three runs): 12.6 to
//! 18 ms at the process's first open, of an open of 28.6 to 128.9 ms, and 10.3 to 12.7 ms at a
//! second, 5 to 7 ms a format. On the RX 7900 XTX (radeonsi, Mesa 26.2.6): 17.3 to 38.4 ms,
//! 38.4 at a process's first open. On the Iris Xe (iHD 26.3.5): 9.9 to 36.4 ms, of opens of
//! 11.8 to 104.4 ms, the trials of a dma-buf a plane then failing at every alignment.

use std::time::Duration;

use crate::core::device_memory::{
    DeviceMemory, DmaBufError, GuestTarget, MemoryIdentity, MemoryLife, SurfaceDmaBuf,
};
use crate::core::fourcc::FourCc;
use crate::core::surfaces::{PlaneLayout, SurfaceLayout};
use crate::core::sync::Mutex;

use super::display::{Display, Error};
use super::ffi;
use super::host_memory::HostMemory;
use super::image::Image;
use super::surface::{SurfaceFormat, SurfacePool};

/// The alignments tried, least first: of each row, and of each plane's first byte from its
/// dma-buf's.
const ALIGNMENTS: [usize; 3] = [64, 128, 256];

/// The trial picture's size: wide enough that a row of either format spans more than one
/// alignment, its row of neither format a multiple of any, and as many rows as hold every sample
/// value at least once.
const WIDTH: u32 = 328;
const HEIGHT: u32 = 64;

/// What fills the trial memory before the run: a byte the run must leave wherever it writes no
/// sample.
const UNTOUCHED: u8 = 0xa5;

const PAGE: usize = 4096;

/// What of an import of a format the driver's video processing writes right, as [`learn`] found
/// it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImportWrite {
    /// The least alignment of rows and of planes from the dma-buf's first byte.
    pub alignment: usize,
    /// Whether that holds only for rows of samples of whole alignments, the luma plane at the
    /// dma-buf's first byte and the chroma plane whole rows of its pitch after it (module docs:
    /// iHD).
    pub whole_rows: bool,
}

impl ImportWrite {
    /// Any rows at `alignment`.
    pub fn at(alignment: usize) -> Self {
        Self {
            alignment,
            whole_rows: false,
        }
    }
}

/// What the driver's video processing writes into imports of each format, as [`learn`] found it.
#[derive(Clone, Debug)]
pub struct ImportWrites {
    nv12: Result<ImportWrite, String>,
    p010: Result<ImportWrite, String>,
    /// How long each format's trials took: NV12's, then P010's.
    took: [Duration; 2],
}

impl ImportWrites {
    /// No import of either format, for `why`.
    pub(crate) fn none(why: &str) -> Self {
        Self {
            nv12: Err(why.to_owned()),
            p010: Err(why.to_owned()),
            took: [Duration::ZERO; 2],
        }
    }

    /// The answers `nv12` and `p010`, as a stand-in driver's check would give them: for a host's
    /// tests of what it makes of them.
    pub fn of_answers(
        nv12: Result<ImportWrite, String>,
        p010: Result<ImportWrite, String>,
    ) -> Self {
        Self {
            nv12,
            p010,
            took: [Duration::ZERO; 2],
        }
    }

    /// What of an import of `format` the driver takes and its video processing writes right;
    /// `Err` says why it is lent none.
    pub(crate) fn of(&self, format: SurfaceFormat) -> Result<ImportWrite, String> {
        match format {
            SurfaceFormat::Nv12 => self.nv12.clone(),
            SurfaceFormat::P010 => self.p010.clone(),
        }
    }

    /// [`ImportWrites::of`] NV12, the 8-bit decoders' format.
    pub fn nv12(&self) -> Result<ImportWrite, String> {
        self.nv12.clone()
    }

    /// [`ImportWrites::of`] P010, the 10-bit decoders' format.
    pub fn p010(&self) -> Result<ImportWrite, String> {
        self.p010.clone()
    }

    /// How long the trial runs took, both formats'.
    pub fn took(&self) -> Duration {
        self.took[0] + self.took[1]
    }

    /// How long each format's trial runs took: NV12's, then P010's.
    pub fn took_each(&self) -> [Duration; 2] {
        self.took
    }
}

/// Runs the trial for both formats on `display`, on this thread (module docs).
pub(crate) fn learn(display: &Display) -> ImportWrites {
    let clock = display.runtime();
    let start = clock.now();
    let nv12 = learn_format(display, SurfaceFormat::Nv12);
    let between = clock.now();
    let p010 = learn_format(display, SurfaceFormat::P010);
    let writes = ImportWrites {
        nv12,
        p010,
        took: [
            between.saturating_duration_since(start),
            clock.now().saturating_duration_since(between),
        ],
    };
    let said = |format: &str, answer: &Result<ImportWrite, String>| match answer {
        Ok(ImportWrite {
            alignment,
            whole_rows: false,
        }) => format!("{format} at a {alignment}-byte alignment"),
        Ok(ImportWrite {
            alignment,
            whole_rows: true,
        }) => format!("{format} at a {alignment}-byte alignment, in rows of whole {alignment}s"),
        Err(why) => format!("no {format}: {why}"),
    };
    log::info!(
        "VA-API at {}: video processing writes imports of {}, and of {}, as checked in {:.1} ms",
        display.path(),
        said("NV12", &writes.nv12),
        said("P010", &writes.p010),
        writes.took().as_secs_f64() * 1e3
    );
    writes
}

/// The first of [`ALIGNMENTS`] at which an import of `format` is written right, in any rows or
/// in rows of whole alignments (module docs), or why none is.
fn learn_format(display: &Display, format: SurfaceFormat) -> Result<ImportWrite, String> {
    let picture = Picture {
        format,
        width: WIDTH,
        height: HEIGHT,
    };
    let source = Source::new(display, picture)?;
    let mut refusals = Vec::new();
    for alignment in ALIGNMENTS {
        let any = ImportWrite::at(alignment);
        match trial(display, &source, picture, picture.layout(alignment), any) {
            Ok(None) => return Ok(any),
            Ok(Some(wrong)) => refusals.push(format!(
                "at {alignment} bytes, the run answered success and {wrong}"
            )),
            Err(why) => {
                refusals.push(format!("at {alignment} bytes, {why}"));
                continue;
            }
        }
        let whole = picture.whole(alignment);
        let whole_rows = ImportWrite {
            alignment,
            whole_rows: true,
        };
        match trial(
            display,
            &source,
            whole,
            whole.whole_layout(alignment),
            whole_rows,
        ) {
            Ok(None) => return Ok(whole_rows),
            Ok(Some(wrong)) => refusals.push(format!(
                "in rows of whole {alignment}s, the run answered success and {wrong}"
            )),
            Err(why) => refusals.push(format!("in rows of whole {alignment}s, {why}")),
        }
    }
    Err(refusals.join("; "))
}

/// A trial picture's format and size.
#[derive(Clone, Copy, Debug)]
struct Picture {
    format: SurfaceFormat,
    width: u32,
    height: u32,
}

impl Picture {
    /// The rows of `plane`: luma's, then chroma's half.
    fn rows(self, plane: usize) -> usize {
        to_usize(if plane == 0 {
            self.height
        } else {
            self.height.div_ceil(2)
        })
    }

    /// The bytes of a row of samples, the same in both planes of 4:2:0.
    fn row_bytes(self) -> usize {
        to_usize(self.width.next_multiple_of(2)) * self.format.bytes_per_sample()
    }

    /// Where the opening's trial at `alignment` lays its planes out in its memory: each plane's
    /// offset and its pitch, and the memory's bytes. The pitch is the least odd multiple of
    /// `alignment` that holds a row, and the luma plane starts `alignment` into the first page,
    /// so neither is a multiple of twice the alignment.
    fn layout(self, alignment: usize) -> ([(usize, usize); 2], usize) {
        let mut times = self.row_bytes().div_ceil(alignment);
        if times % 2 == 0 {
            times += 1;
        }
        let pitch = times * alignment;
        let luma = alignment;
        let chroma = luma + pitch * self.rows(0);
        let end = chroma + pitch * self.rows(1) + alignment;
        ([(luma, pitch), (chroma, pitch)], end.next_multiple_of(PAGE))
    }

    /// The picture cut to rows of whole `alignment`s: as wide as that leaves of it, from its
    /// first column.
    fn whole(self, alignment: usize) -> Self {
        let bytes = self.format.bytes_per_sample();
        let row = self.row_bytes() / alignment * alignment;
        Self {
            width: u32::try_from(row / bytes).unwrap_or(0),
            ..self
        }
    }

    /// Where the second trial at `alignment` lays its planes out ([`Picture::layout`]), its rows
    /// whole alignments and its planes whole rows from the memory's first byte: the luma plane
    /// there, the pitch the least odd multiple of `alignment` past a row, and the chroma plane a
    /// row past the luma's last, so a driver that puts the chroma plane right after the surface's
    /// rows misplaces it.
    fn whole_layout(self, alignment: usize) -> ([(usize, usize); 2], usize) {
        let mut times = self.row_bytes() / alignment + 1;
        if times % 2 == 0 {
            times += 1;
        }
        let pitch = times * alignment;
        let chroma = pitch * (self.rows(0) + 1);
        let end = chroma + pitch * self.rows(1) + alignment;
        ([(0, pitch), (chroma, pitch)], end.next_multiple_of(PAGE))
    }

    /// The picture as a guest's buffer of it lies, at rows of `pitch` bytes: the luma plane at
    /// the memory's first byte and the chroma plane right after its rows, and the memory's bytes.
    #[cfg(test)]
    fn guest_layout(self, pitch: usize) -> ([(usize, usize); 2], usize) {
        let chroma = pitch * self.rows(0);
        let end = chroma + pitch * self.rows(1);
        ([(0, pitch), (chroma, pitch)], end.next_multiple_of(PAGE))
    }
}

/// The trial picture of one format, in a surface of the decoders' kind.
struct Source {
    /// Holds the surface's pool.
    surface: super::surface::Surface,
}

impl Source {
    /// A surface of `picture`'s format and size holding [`sample`]'s picture, put there by
    /// `vaPutImage`.
    fn new(display: &Display, picture: Picture) -> Result<Self, String> {
        let Picture {
            format,
            width,
            height,
        } = picture;
        let pool = SurfacePool::new(display, format, width, height, 1)
            .map_err(|e| format!("no {} surface: {e}", format.name()))?;
        let surface = pool
            .get()
            .map_err(|e| format!("no {} surface: {e}", format.name()))?;
        let image_format = ffi::VAImageFormat {
            fourcc: format.fourcc(),
            bits_per_pixel: format.bits_per_pixel(),
            ..ffi::VAImageFormat::default()
        };
        let mut image = Image::create(display, image_format, width, height)
            .map_err(|e| format!("no {} image: {e}", format.name()))?;
        let raw = *image.raw();
        let planes = [
            (to_usize(raw.offsets[0]), to_usize(raw.pitches[0])),
            (to_usize(raw.offsets[1]), to_usize(raw.pitches[1])),
        ];
        {
            let mut mapping = image
                .map()
                .map_err(|e| format!("the {} image's map: {e}", format.name()))?;
            fill(picture, mapping.bytes_mut(), planes)?;
        }
        image
            .put_into(surface.id(), width, height)
            .map_err(|e| format!("the picture's vaPutImage: {e}"))?;
        Ok(Self { surface })
    }
}

/// Writes `picture` into `bytes`, its planes at `planes`' offsets and pitches.
fn fill(picture: Picture, bytes: &mut [u8], planes: [(usize, usize); 2]) -> Result<(), String> {
    let sample = picture.format.bytes_per_sample();
    for (plane, (offset, pitch)) in planes.into_iter().enumerate() {
        for row in 0..picture.rows(plane) {
            let at = offset + row * pitch;
            let line = bytes
                .get_mut(at..at + picture.row_bytes())
                .ok_or("an image smaller than its planes")?;
            for (column, out) in line.chunks_exact_mut(sample).enumerate() {
                let value = self::sample(picture.format, plane, column, row);
                out.copy_from_slice(&value.to_le_bytes()[..sample]);
            }
        }
    }
    Ok(())
}

/// The trial picture's sample at `column` (in samples: a chroma row's Cb and Cr alternate) of
/// `row` of `plane`, as its bytes hold it: 8-bit values for NV12, 10-bit ones in a sample's high
/// bits for P010. Across the opening's trial picture every value of the format turns up.
fn sample(format: SurfaceFormat, plane: usize, column: usize, row: usize) -> u16 {
    let (a, b, c) = if plane == 0 { (3, 5, 7) } else { (7, 11, 101) };
    let value = column * a + row * b + c;
    match format {
        SurfaceFormat::Nv12 => (value % 256) as u16,
        SurfaceFormat::P010 => ((value * 13 % 1024) as u16) << 6,
    }
}

/// Bytes a run left other than it should have ([`wrong`]).
#[derive(Clone, Debug, PartialEq, Eq)]
struct Wrong {
    /// The first: its offset, and what it holds.
    at: usize,
    got: u8,
    /// Where it lies: its plane, row and byte in the row, where it lies in one.
    place: Option<(usize, usize, usize)>,
    /// How many are wrong in all, and how many of those lie outside the planes' samples.
    count: usize,
    outside: usize,
}

impl std::fmt::Display for Wrong {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "left {} bytes other than the picture's", self.count)?;
        if self.outside > 0 {
            write!(f, " ({} of them outside its planes)", self.outside)?;
        }
        write!(f, ", the first byte {} at {:#04x}", self.at, self.got)?;
        if let Some((plane, row, byte)) = self.place {
            write!(f, ", plane {plane} row {row} byte {byte}")?;
        }
        Ok(())
    }
}

/// What in `bytes`, `picture` laid out at `planes` over memory first all [`UNTOUCHED`], is not
/// what a run that writes it right leaves: `None` where every byte is.
fn wrong(picture: Picture, bytes: &[u8], planes: [(usize, usize); 2]) -> Option<Wrong> {
    let mut expected = vec![UNTOUCHED; bytes.len()];
    if fill(picture, &mut expected, planes).is_err() {
        return Some(Wrong {
            at: bytes.len(),
            got: 0,
            place: None,
            count: 1,
            outside: 0,
        });
    }
    compare(picture, bytes, &expected, planes)
}

/// What of `picture`'s samples in `bytes`, at `planes`, is not the picture's, whatever lies
/// around them: `None` where every sample is.
#[cfg(test)]
fn samples_wrong(picture: Picture, bytes: &[u8], planes: [(usize, usize); 2]) -> Option<Wrong> {
    let mut expected = bytes.to_vec();
    if fill(picture, &mut expected, planes).is_err() {
        return Some(Wrong {
            at: bytes.len(),
            got: 0,
            place: None,
            count: 1,
            outside: 0,
        });
    }
    compare(picture, bytes, &expected, planes)
}

/// Where `bytes` differs from `expected`, `picture` laid out in both at `planes`.
fn compare(
    picture: Picture,
    bytes: &[u8],
    expected: &[u8],
    planes: [(usize, usize); 2],
) -> Option<Wrong> {
    let mut found: Option<Wrong> = None;
    for (at, (got, want)) in bytes.iter().zip(expected).enumerate() {
        if got == want {
            continue;
        }
        let place = place_of(picture, planes, at);
        let wrong = found.get_or_insert(Wrong {
            at,
            got: *got,
            place,
            count: 0,
            outside: 0,
        });
        wrong.count += 1;
        wrong.outside += usize::from(place.is_none());
    }
    found
}

/// The plane, row and byte in the row of the sample byte at `at`, `picture` laid out at
/// `planes`; `None` where no sample lies there.
fn place_of(
    picture: Picture,
    planes: [(usize, usize); 2],
    at: usize,
) -> Option<(usize, usize, usize)> {
    planes
        .into_iter()
        .enumerate()
        .find_map(|(plane, (offset, pitch))| {
            let into = at.checked_sub(offset)?;
            let (row, byte) = (into / pitch, into % pitch);
            (row < picture.rows(plane) && byte < picture.row_bytes()).then_some((plane, row, byte))
        })
}

/// One run of `source`, cropped to `picture`, into memory of `size` bytes laid out at `planes`,
/// imported as `write` says: `Ok(None)` where the import is taken and the run writes it right,
/// `Ok(Some)` with what it wrote wrong, and `Err` with why where the import or the run is
/// refused.
fn trial(
    display: &Display,
    source: &Source,
    picture: Picture,
    laid: ([(usize, usize); 2], usize),
    write: ImportWrite,
) -> Result<Option<Wrong>, String> {
    trial_in(display, source, picture, laid, write, None)
}

/// [`trial`], in `strips` strips where given (zero copy's least that fit where not).
fn trial_in(
    display: &Display,
    source: &Source,
    picture: Picture,
    (planes, size): ([(usize, usize); 2], usize),
    write: ImportWrite,
    strips: Option<usize>,
) -> Result<Option<Wrong>, String> {
    let memory = TrialMemory::new(size).map_err(|e| format!("no memory for it: {e}"))?;
    let fourcc = match picture.format {
        SurfaceFormat::Nv12 => FourCc::new(*b"420v"),
        SurfaceFormat::P010 => FourCc::new(*b"x420"),
    };
    let plane = |n: usize| PlaneLayout {
        offset: planes[n].0,
        bytes_per_row: planes[n].1,
        width: if n == 0 {
            picture.width
        } else {
            picture.width.div_ceil(2)
        },
        height: u32::try_from(picture.rows(n)).unwrap_or(u32::MAX),
    };
    let layout = SurfaceLayout {
        pixel_format: fourcc,
        width: picture.width,
        height: picture.height,
        planes: vec![plane(0), plane(1)],
        bytes_per_row: 0,
    };
    let target = GuestTarget::new(&layout, &memory)?;
    super::zero_copy::trial(display, &source.surface, &target, write, strips)?;
    let held = memory.memory.lock();
    Ok(wrong(picture, held.bytes(), planes))
}

/// Memory of the trial's own, as a guest buffer's is lent to a device: its pages as dma-bufs
/// of ranges of them.
struct TrialMemory {
    memory: Mutex<HostMemory>,
    life: MemoryLife,
}

impl TrialMemory {
    /// `size` bytes, every one [`UNTOUCHED`].
    fn new(size: usize) -> Result<Self, Error> {
        let mut memory = HostMemory::new(size)?;
        memory.bytes_mut().fill(UNTOUCHED);
        Ok(Self {
            memory: Mutex::new(memory),
            life: MemoryLife::new(),
        })
    }
}

impl DeviceMemory for TrialMemory {
    fn identity(&self) -> MemoryIdentity {
        self.life.identity()
    }

    fn length(&self) -> usize {
        self.memory.lock().len()
    }

    /// The whole pages from the first range's to the last range's end, as one dma-buf: the
    /// trial's ranges are one a call.
    fn dma_buf_of(&self, ranges: &[(usize, usize)]) -> Result<SurfaceDmaBuf, DmaBufError> {
        let (Some(&(first, _)), Some(&(last, length))) = (ranges.first(), ranges.last()) else {
            return Err(DmaBufError::Refused("no ranges".to_owned()));
        };
        let start = first / PAGE * PAGE;
        let end = (last + length).next_multiple_of(PAGE);
        let fd = (self.memory.lock())
            .dma_buf_of_pages(start, end - start)
            .map_err(|e| DmaBufError::Refused(e.to_string()))?;
        Ok(SurfaceDmaBuf {
            fd,
            size: (end - start) as u64,
            offsets: ranges.iter().map(|&(at, _)| (at - start) as u64).collect(),
        })
    }
}

fn to_usize(value: u32) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FORMATS: [SurfaceFormat; 2] = [SurfaceFormat::Nv12, SurfaceFormat::P010];

    fn opening(format: SurfaceFormat) -> Picture {
        Picture {
            format,
            width: WIDTH,
            height: HEIGHT,
        }
    }

    #[test]
    fn each_pitch_and_luma_start_is_an_odd_multiple_of_its_alignment() {
        for format in FORMATS {
            let picture = opening(format);
            for alignment in ALIGNMENTS {
                let ([(luma, pitch), (chroma, chroma_pitch)], size) = picture.layout(alignment);
                assert_eq!(pitch, chroma_pitch);
                assert!(pitch >= picture.row_bytes(), "{format:?} {alignment}");
                assert_eq!(pitch % alignment, 0);
                assert_ne!(pitch % (2 * alignment), 0, "{format:?} {alignment}");
                assert_eq!(luma % alignment, 0);
                assert_ne!(luma % (2 * alignment), 0);
                assert_eq!(chroma % alignment, 0);
                assert!(chroma >= luma + pitch * picture.rows(0));
                assert!(size >= chroma + pitch * picture.rows(1) + alignment);
                assert_eq!(size % PAGE, 0);
            }
        }
    }

    #[test]
    fn the_whole_rows_trial_has_whole_rows_and_planes_whole_rows_apart() {
        for format in FORMATS {
            for alignment in ALIGNMENTS {
                let picture = opening(format).whole(alignment);
                assert_eq!(picture.row_bytes() % alignment, 0, "{format:?} {alignment}");
                assert!(picture.width > 0 && picture.width <= WIDTH);
                assert_eq!(picture.width % 2, 0);
                let ([(luma, pitch), (chroma, _)], size) = picture.whole_layout(alignment);
                assert_eq!(luma, 0);
                assert!(
                    pitch > picture.row_bytes(),
                    "{format:?} {alignment}: padding"
                );
                assert_eq!(pitch % alignment, 0);
                assert_ne!(pitch % (2 * alignment), 0);
                assert_eq!(chroma % pitch, 0, "{format:?} {alignment}: whole rows");
                assert!(
                    chroma > pitch * picture.rows(0),
                    "{format:?} {alignment}: a row past the luma's"
                );
                assert!(size >= chroma + pitch * picture.rows(1) + alignment);
                assert_eq!(size % PAGE, 0);
            }
        }
    }

    #[test]
    fn no_row_of_the_first_trial_is_whole_alignments() {
        for format in FORMATS {
            for alignment in ALIGNMENTS {
                assert_ne!(
                    opening(format).row_bytes() % alignment,
                    0,
                    "{format:?} {alignment}"
                );
                assert!(opening(format).row_bytes() > alignment);
            }
        }
    }

    #[test]
    fn the_picture_holds_every_value_of_its_format() {
        let mut nv12 = std::collections::HashSet::new();
        let mut p010 = std::collections::HashSet::new();
        for plane in 0..2 {
            for row in 0..opening(SurfaceFormat::Nv12).rows(plane) {
                for column in 0..to_usize(WIDTH) {
                    nv12.insert(sample(SurfaceFormat::Nv12, plane, column, row));
                    p010.insert(sample(SurfaceFormat::P010, plane, column, row));
                }
            }
        }
        assert_eq!(nv12.len(), 256);
        assert_eq!(p010.len(), 1024);
        assert!(p010.iter().all(|v| v & 0x3f == 0));
    }

    /// The bytes a driver that writes `picture` right at `alignment` leaves: the picture at the
    /// planes, and [`UNTOUCHED`] elsewhere.
    fn written(picture: Picture, alignment: usize) -> (Vec<u8>, [(usize, usize); 2]) {
        let (planes, size) = picture.layout(alignment);
        let mut bytes = vec![UNTOUCHED; size];
        fill(picture, &mut bytes, planes).unwrap();
        (bytes, planes)
    }

    #[test]
    fn a_right_write_passes_and_any_other_byte_fails() {
        for format in FORMATS {
            let picture = opening(format);
            for alignment in ALIGNMENTS {
                let (mut bytes, planes) = written(picture, alignment);
                assert_eq!(wrong(picture, &bytes, planes), None);
                // A sample wrong, a padding byte written where rows have padding, and a byte
                // before the luma plane and the last after the chroma plane written.
                let mut cases = vec![
                    (planes[1].0 + 3, Some((1, 0, 3))),
                    (0, None),
                    (bytes.len() - 1, None),
                ];
                if planes[0].1 > picture.row_bytes() {
                    cases.push((planes[0].0 + picture.row_bytes(), None));
                }
                for (at, place) in cases {
                    let was = bytes[at];
                    bytes[at] ^= 0x40;
                    let found = wrong(picture, &bytes, planes).unwrap();
                    assert_eq!((found.at, found.got, found.place), (at, was ^ 0x40, place));
                    assert_eq!(found.count, 1);
                    assert_eq!(found.outside, usize::from(place.is_none()));
                    bytes[at] = was;
                }
            }
        }
    }

    #[test]
    fn a_driver_that_writes_nothing_fails() {
        for format in FORMATS {
            let picture = opening(format);
            let (planes, size) = picture.layout(64);
            let bytes = vec![UNTOUCHED; size];
            let found = wrong(picture, &bytes, planes).unwrap();
            assert_eq!(found.outside, 0);
            assert!(found.count > picture.row_bytes() * picture.rows(0));
        }
    }

    #[test]
    fn a_picture_at_a_pitch_of_twice_the_alignment_fails() {
        // A driver that lays rows out at its own larger pitch misplaces every row after the first.
        let picture = opening(SurfaceFormat::Nv12);
        let (planes, size) = picture.layout(64);
        let mut bytes = vec![UNTOUCHED; size];
        let misplaced = [(planes[0].0, planes[0].1 + 64), (planes[1].0, planes[1].1)];
        let mut scratch = vec![UNTOUCHED; size * 2];
        fill(picture, &mut scratch, misplaced).unwrap();
        bytes.copy_from_slice(&scratch[..size]);
        let found = wrong(picture, &bytes, planes).unwrap();
        // Row 0 lies right; row 1's first sample is where the driver left row 0's padding.
        assert_eq!(
            (found.at, found.place),
            (planes[0].0 + planes[0].1, Some((0, 1, 0)))
        );
    }

    #[test]
    fn a_guest_layout_follows_its_planes_on() {
        let picture = Picture {
            format: SurfaceFormat::P010,
            width: 1920,
            height: 1080,
        };
        let ([(luma, pitch), (chroma, _)], size) = picture.guest_layout(4096);
        assert_eq!((luma, pitch, chroma), (0, 4096, 4096 * 1080));
        assert_eq!(size, 4096 * (1080 + 540));
    }

    /// The decoders' shapes: a surface's coded size, and the picture cropped from it.
    const SHAPES: [((u32, u32), (u32, u32)); 9] = [
        ((160, 64), (160, 64)),
        ((352, 288), (352, 288)),
        ((416, 240), (416, 240)),
        ((640, 368), (640, 360)),
        ((1280, 368), (1280, 360)),
        ((1280, 720), (1280, 720)),
        ((1920, 1080), (1920, 1080)),
        ((1920, 1088), (1920, 1080)),
        ((3840, 2160), (3840, 2160)),
    ];

    /// The pitches a guest's buffer of `picture` may have: the least multiple of each alignment
    /// that holds a row, the next one, and a row and 256 bytes.
    fn pitches(picture: Picture) -> Vec<usize> {
        let row = picture.row_bytes();
        let mut pitches: Vec<usize> = ALIGNMENTS
            .iter()
            .flat_map(|&a| [row.next_multiple_of(a), row.next_multiple_of(a) + a])
            .chain([row + 256])
            .collect();
        pitches.sort_unstable();
        pitches.dedup();
        pitches
    }

    /// `source` copied back the staged way, whole, and `picture`'s samples checked in it.
    fn staged(
        display: &Display,
        source: &Source,
        coded: (u32, u32),
        picture: Picture,
    ) -> Result<Option<Wrong>, String> {
        let staging =
            super::super::readback::Staging::new(display, picture.format, coded.0, coded.1)
                .map_err(|e| format!("no staging: {e}"))?;
        staging
            .stage(source.surface.id(), super::super::wait::deadline(display))
            .map_err(|e| format!("the copy failed: {e}"))?;
        let (bytes, planes) = staging.bytes();
        Ok(samples_wrong(picture, bytes, planes))
    }

    fn said(answer: &Result<Option<Wrong>, String>) -> String {
        match answer {
            Ok(None) => "right".to_owned(),
            Ok(Some(wrong)) => format!("WRONG: {wrong}"),
            Err(why) => format!("refused: {why}"),
        }
    }

    /// The picture as a guest's buffer of it may lie with its chroma plane at the page after the
    /// luma plane's rows, as an IOSurface's planes may be.
    fn paged_layout(picture: Picture, pitch: usize) -> ([(usize, usize); 2], usize) {
        let chroma = (pitch * picture.rows(0)).next_multiple_of(PAGE);
        let end = chroma + pitch * picture.rows(1);
        ([(0, pitch), (chroma, pitch)], end.next_multiple_of(PAGE))
    }

    /// The picture as [`Picture::guest_layout`] lays it out, moved 256 bytes into its first
    /// page, as a buffer whose luma plane starts past a page's start lies.
    fn moved_layout(picture: Picture, pitch: usize) -> ([(usize, usize); 2], usize) {
        let ([(luma, _), (chroma, _)], _) = picture.guest_layout(pitch);
        let (luma, chroma) = (luma + 256, chroma + 256);
        let end = chroma + pitch * picture.rows(1);
        ([(luma, pitch), (chroma, pitch)], end.next_multiple_of(PAGE))
    }

    /// Whether zero copy uses `write` for `picture` laid out at `planes` (`zero_copy::write`,
    /// `Strip::new`): the pitch and the planes at its alignment, and with whole rows, the rows'
    /// samples whole alignments, the luma plane at the memory's first byte and the chroma plane
    /// whole rows from it.
    fn used(
        write: &Result<ImportWrite, String>,
        picture: Picture,
        planes: [(usize, usize); 2],
    ) -> bool {
        let Ok(write) = write else { return false };
        let a = write.alignment;
        let [(luma, pitch), (chroma, _)] = planes;
        pitch.is_multiple_of(a)
            && luma.is_multiple_of(a)
            && chroma.is_multiple_of(a)
            && (!write.whole_rows
                || (picture.row_bytes().is_multiple_of(a)
                    && luma.is_multiple_of(PAGE)
                    && (chroma - luma).is_multiple_of(pitch)))
    }

    /// Every decoder shape copied back the staged way, and written through zero copy's import at
    /// the pitches a guest's buffer may have, its chroma plane right after the luma's rows and at
    /// the next page, the two 256 bytes into the page, and the tall ones in two and in four
    /// strips: a way the display's check lends
    /// the driver writes every shape right where it is used (surface.rs's staged copy back, where
    /// any import is lent; zero_copy.rs's imports, by [`used`]). Each answer is printed, so a run
    /// on a driver shows what it does.
    #[test]
    #[ignore = "needs a VA-API GPU: run on the VA-API host"]
    fn the_check_lends_no_way_a_decoder_shape_comes_out_wrong_through() {
        use crate::vaapi::va::gated::GatedRuntime;
        let node = std::env::var("RVG_VAAPI_NODE").unwrap_or_else(|_| "/dev/dri/renderD128".into());
        let display = Display::open(&node, std::sync::Arc::new(GatedRuntime)).expect("a display");
        let writes = display.import_writes();
        let mut lent_wrong = Vec::new();
        for format in FORMATS {
            let lent = writes.of(format);
            eprintln!("{node} {}: the check answered {lent:?}", format.name());
            for ((coded_width, coded_height), (width, height)) in SHAPES {
                let coded = Picture {
                    format,
                    width: coded_width,
                    height: coded_height,
                };
                let source = match Source::new(&display, coded) {
                    Ok(source) => source,
                    Err(why) => {
                        eprintln!("  {coded_width}x{coded_height}: no source: {why}");
                        continue;
                    }
                };
                let picture = Picture {
                    format,
                    width,
                    height,
                };
                let shape = format!(
                    "{} {coded_width}x{coded_height} as {width}x{height}",
                    format.name()
                );
                let mut check =
                    |what: String, used: bool, answer: Result<Option<Wrong>, String>| {
                        eprintln!(
                            "  {shape} {what}: {}{}",
                            said(&answer),
                            if used { " (lent)" } else { "" }
                        );
                        if used && !matches!(answer, Ok(None)) {
                            lent_wrong.push(format!("{shape} {what}: {}", said(&answer)));
                        }
                    };
                let staged_used = lent.is_ok();
                let answer = staged(&display, &source, (coded_width, coded_height), picture);
                check("staged".to_owned(), staged_used, answer);
                let probe = ImportWrite::at(ALIGNMENTS[0]);
                for pitch in pitches(picture) {
                    for (name, laid) in [
                        ("", picture.guest_layout(pitch)),
                        (", chroma at a page", paged_layout(picture, pitch)),
                        (", 256 bytes into the page", moved_layout(picture, pitch)),
                    ] {
                        let answer = trial(&display, &source, picture, laid, probe);
                        check(
                            format!("rows of {pitch}{name}"),
                            used(&lent, picture, laid.0),
                            answer,
                        );
                    }
                }
                let Ok(write) = lent.clone() else { continue };
                if height < 1080 {
                    continue;
                }
                let pitch = picture.row_bytes().next_multiple_of(write.alignment);
                for strips in [2, 4] {
                    let laid = picture.guest_layout(pitch);
                    let answer = trial_in(&display, &source, picture, laid, write, Some(strips));
                    check(
                        format!("rows of {pitch} in {strips} strips"),
                        used(&lent, picture, laid.0),
                        answer,
                    );
                }
            }
        }
        assert!(
            lent_wrong.is_empty(),
            "the check lent ways that write shapes wrong:\n{}",
            lent_wrong.join("\n")
        );
    }

    /// Two buffers of one size in one memory, each written through zero copy with the same
    /// imports, as a HEIC grid image's tiles are (zero_copy.rs's `Variant`): stacked, each its
    /// planes from a page's start, and side by side in rows of both, the second's planes a row
    /// of samples into the memory's rows. Each buffer's frame lands at its own planes or is
    /// declined to the CPU's copy, and nothing else of the memory is written; where the display
    /// lends the format's imports, the stacked buffers are both written.
    #[test]
    #[ignore = "needs a VA-API GPU: run on the VA-API host"]
    fn buffers_in_one_memory_are_each_written_at_their_own_planes() {
        use std::sync::Arc;

        use super::super::frame::{Frame, GuestWrites};
        use super::super::surface::Rect;
        use super::super::wait::Deadline;
        use crate::core::backend::DeviceWrite;
        use crate::vaapi::va::gated::GatedRuntime;

        let node = std::env::var("RVG_VAAPI_NODE").unwrap_or_else(|_| "/dev/dri/renderD128".into());
        let display = Display::open(&node, Arc::new(GatedRuntime)).expect("a display");
        let mut failed = Vec::new();
        for format in FORMATS {
            let lent = display.writes_imports(format);
            eprintln!("{node} {}: the check answered {lent:?}", format.name());
            let alignment = lent.as_ref().map_or(ALIGNMENTS[0], |write| write.alignment);
            let picture = Picture {
                format,
                width: 512,
                height: 512,
            };
            let Source { surface } = Source::new(&display, picture).expect("a source");
            let frame = Frame {
                surface: Arc::new(surface),
                id: 0,
                poc: 0,
                idr: true,
                reference: false,
                rect: Rect {
                    x: 0,
                    y: 0,
                    width: picture.width,
                    height: picture.height,
                },
                coded_size: (picture.width, picture.height),
                full_range: false,
                deadline: Deadline::default(),
            };
            let pitch = picture.row_bytes().next_multiple_of(alignment);
            let (stacked, tile) = picture.guest_layout(pitch);
            let wide = 2 * pitch;
            let chroma = wide * picture.rows(0);
            let side_by_side = [0, pitch].map(|at| [(at, wide), (chroma + at, wide)]);
            // A target's planes span whole rows: the second's last chroma row ends a row of
            // samples past the rows of both.
            let size = (chroma + wide * picture.rows(1) + pitch).next_multiple_of(PAGE);
            for (name, buffers, size) in [
                (
                    "stacked",
                    [stacked, stacked.map(|(at, pitch)| (at + tile, pitch))],
                    2 * tile,
                ),
                ("side by side", side_by_side, size),
            ] {
                let memory = TrialMemory::new(size).expect("memory");
                let writes = GuestWrites::new(&display);
                let mut expected = vec![UNTOUCHED; size];
                let mut answers = Vec::new();
                for planes in buffers {
                    let plane = |n: usize| PlaneLayout {
                        offset: planes[n].0,
                        bytes_per_row: planes[n].1,
                        width: if n == 0 { 512 } else { 256 },
                        height: u32::try_from(picture.rows(n)).unwrap_or(u32::MAX),
                    };
                    let layout = SurfaceLayout {
                        pixel_format: match format {
                            SurfaceFormat::Nv12 => FourCc::new(*b"420v"),
                            SurfaceFormat::P010 => FourCc::new(*b"x420"),
                        },
                        width: picture.width,
                        height: picture.height,
                        planes: vec![plane(0), plane(1)],
                        bytes_per_row: 0,
                    };
                    let target = GuestTarget::new(&layout, &memory).expect("a target");
                    let answer = frame.write_to_guest(&target, &writes).expect("a write");
                    if matches!(answer, DeviceWrite::Written) {
                        fill(picture, &mut expected, planes).expect("the planes in the memory");
                    }
                    answers.push(answer);
                }
                let shape = format!("{} {name}", format.name());
                eprintln!("  {shape}: {answers:?}");
                let held = memory.memory.lock();
                if let Some(wrong) = compare(picture, held.bytes(), &expected, buffers[1]) {
                    failed.push(format!("{shape}: {answers:?}: {wrong}"));
                }
                let both = answers.iter().all(|a| matches!(a, DeviceWrite::Written));
                if name == "stacked" && lent.is_ok() && !both {
                    failed.push(format!("{shape}: lent, yet {answers:?}"));
                }
            }
        }
        assert!(failed.is_empty(), "{}", failed.join("\n"));
    }
}
