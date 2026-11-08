//! The host's own video sessions on the reims-graphics device ([`HostVideoSession`]): hardware
//! video on a host API whose work no runtime queue carries (NVDEC, VA-API), whose pictures the
//! host lands in the decodes' output surfaces, and whose session event the host signals. A
//! backend gives the decoder ([`HostDecoder`]) and the way each of its pictures is written
//! ([`HostPicture`]); the rest, here, is the same for every backend:
//!
//! - **A decode** ([`GraphicsVideoSession::decode`]) hands the decoder the op's parameter sets,
//!   then its access unit, one call each, on the thread that takes the command buffer. The
//!   decoder starts the hardware's work and returns. A refusal of either faults the command
//!   buffer and spends no value of the event. The access unit is read in place, in the host bytes
//!   of the guest's bitstream buffer. The decode holds that buffer, and the output's, until its
//!   value is signalled, and the guest writes neither again before the command buffer that waits
//!   for the value completes. So a decoder may hand the hardware the access unit's bytes in
//!   place, to be read after the call returns.
//! - **Landing.** The picture the decoder answers goes to the session's landing thread
//!   (`reims-vgpu.graphics-video-land`). It lands the pictures in decode order, each in its own
//!   decode's output, and signals each decode's value once its picture is there. A decode that
//!   answers no picture (one the stream's random access skips) is signalled in its turn, and its
//!   output keeps its bytes. Each plane of an output lies in one of two places:
//!   - **The surface's pages**, the host bytes of the plane's buffer ([`OutputPlane::host`]).
//!     The backend's hardware writes them, or the CPU does ([`Landing::copy_rows`]).
//!   - **The surface's device cache** ([`VideoPlane::cached`]), the runtime's buffer in the GPU's
//!     own memory. The backend writes it on the GPU where it reaches the buffer (a copy on a
//!     runtime queue, or an import of the buffer's memory). Otherwise it writes host bytes the
//!     session stages the plane in ([`Landing::host`]), as it writes pages, and the session then
//!     copies them into the cache on a runtime queue of its own, waiting for the copy before the
//!     signal.
//!
//!   A picture that fails to land is logged and counted, and its value is signalled all the
//!   same: the command buffer goes on with its output as the landing left it, as a GPU decode's
//!   error leaves one.
//!
//!   A picture holds the hardware's surface until it has landed, so the session holds no more
//!   pictures unlanded than the decoder spares ([`HostDecoder::most_unlanded`]): a decode past
//!   them waits, once its picture is decoded, for the oldest to land. The landing waits only
//!   for the hardware, so the wait ends.
//! - **The event** is a shared event, as the host signals it: the runtime holds a command
//!   buffer's wait for it on the host until the value is reached.
//! - **An end** ([`Drop`]) closes the landing thread's queue and waits up to [`TEARDOWN_WAIT`]
//!   for the last value: the thread lands every picture it was given first. A picture keeps
//!   alive what it lands from, so a decoder may go before its last pictures land.
//!
//! The session logs the first picture of each way it lands at `info` ("hardware video: ..."),
//! and its counts at its end.

#![allow(dead_code, reason = "each of the host's video arms uses a part of it")]

use std::sync::{Arc, mpsc};
use std::time::Duration;

use reims_vgpu_core::host::HostPtr;
use reims_vgpu_core::platform::Handle as _;
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
use reims_vgpu_core::platform::device::GpuDevice as _;
use reims_vgpu_core::platform::event::{EventRef, SharedEvent as _};
use reims_vgpu_core::platform::resource::Buffer as _;
use reims_vgpu_core::platform::types::CommandBufferStatus;
use reims_vgpu_core::raise::{OnFault, Raise, RaiseResult};
use reims_vgpu_core::sync::Mutex;

use super::decode::{GraphicsVideoSession, PlanePages, VideoDecode, VideoPlane, op_fault};

/// How long a session's end waits for its last picture to land.
pub(crate) const TEARDOWN_WAIT: Duration = Duration::from_secs(10);

/// `MTLResourceStorageModeShared`: the staging buffers' storage, host bytes the GPU reads.
const STORAGE_SHARED: u64 = 0;

/// Where a plane's staged bytes start in the staging buffer: the chroma plane's start, past the
/// luma plane's bytes, rounded up to this.
const STAGING_ALIGNMENT: u64 = 256;

/// The failures to land a session logs, one line each; the rest are counted.
const FAILURES_LOGGED: u64 = 8;

/// A backend's error: what it could not do, which the session logs or faults the command buffer
/// with.
pub(crate) type BackendError = Box<dyn std::error::Error + Send + Sync>;

/// A backend's decoder of one codec's Annex-B access units, in decode order, whose session is a
/// [`HostVideoSession`].
pub(crate) trait HostDecoder<P: Platform>: Send + 'static {
    /// Takes parameter sets (H.264's SPS and PPS, HEVC's VPS, SPS and PPS), NAL units in Annex-B
    /// form, which make no picture. The bytes are the op's, gone once this returns.
    fn parameter_sets(&mut self, sets: &[u8]) -> Result<(), BackendError>;

    /// Starts the decode of the access unit `au`, NAL units in Annex-B form, after the parameter
    /// sets the session handed it, and answers its picture, which the session lands in the
    /// decode's output; `None` for a picture the stream's random access skips (a RASL picture
    /// after a CRA picture at the stream's start), whose output gets nothing.
    ///
    /// `au` lies in the guest's bitstream buffer, in place. Its bytes stay mapped and as they are
    /// until the picture answered has landed ([`HostPicture::land`] returned), or, where none is
    /// answered, until the session signals the decode's value: the hardware may read them there
    /// after this returns.
    fn decode(&mut self, au: &[u8]) -> Result<Option<Box<dyn HostPicture<P>>>, BackendError>;

    /// The most pictures it lets its session hold unlanded, queued for the landing thread or
    /// landing, beside its references and the picture being decoded: a picture holds the
    /// hardware's surface until it has landed, and the decoder has this many to spare. A decode
    /// that would hold one more waits, once its picture is decoded, for the oldest to land. One
    /// at the least.
    fn most_unlanded(&self) -> usize;
}

/// A decoded picture, which the session's landing thread lands in its decode's output.
pub(crate) trait HostPicture<P: Platform>: Send {
    /// Writes the picture into `to`'s planes, cropped to them (each plane's `width` by `height`
    /// from its first byte), once its decode is done, and returns once it is there: the session
    /// signals its value right after (or copies its staged planes first). It writes each plane
    /// where it lies: by its hardware, into the planes' buffers or their host bytes
    /// ([`Landing::host`]), or by the CPU ([`Landing::copy_rows`]). It keeps alive what it reads.
    fn land(self: Box<Self>, to: &mut Landing<'_, P>) -> Result<(), BackendError>;
}

/// One plane of a decode's output, as a picture lands in it: the op's plane
/// ([`VideoPlane`]), its buffer held until the decode's signal.
pub(crate) struct OutputPlane<P: Platform> {
    /// The buffer the plane lies in: over the surface's pages, or its device cache (`cached`).
    pub(crate) buffer: P::Buffer,
    /// The plane's first byte in the buffer.
    pub(crate) offset: u64,
    /// Bytes from one of its rows to the next.
    pub(crate) bytes_per_row: u64,
    /// The bytes of one of its rows.
    pub(crate) row_bytes: u64,
    /// Its width in pixels.
    pub(crate) width: u64,
    /// Its height in pixels: the rows of the picture's crop it holds.
    pub(crate) height: u64,
    /// The rows the surface holds from the plane's first, `bytes_per_row` apart, each
    /// `row_bytes`: `height`, or more where the op's `output_rows` says the surface holds the
    /// picture's coded rows (half as many chroma rows), and they lie in the buffer. A backend
    /// that writes whole coded rows in place may write these.
    pub(crate) rows_held: u64,
    /// Whether the buffer is the surface's device cache, in the GPU's own memory, with no host
    /// bytes.
    pub(crate) cached: bool,
    /// The host address of the plane's first byte, in the surface's pages; `None` in a device
    /// cache.
    pub(crate) host: Option<HostPtr>,
    /// The memory the buffer lies in, as a device reaches it in place ([`VideoPlane::pages`]);
    /// `None` in a device cache, and in memory of another kind.
    pub(crate) pages: Option<PlanePages>,
}

/// One plane's host bytes, which a picture lands in ([`Landing::host`]).
pub(crate) struct HostPlane<P: Platform> {
    /// The buffer whose host bytes they are: the plane's own, over the surface's pages, or the
    /// session's staging buffer, which outlives the landing while it is not made again for a
    /// larger picture.
    pub(crate) buffer: P::Buffer,
    /// The host address of the plane's first byte.
    pub(crate) at: HostPtr,
    /// Bytes from one of its rows to the next.
    pub(crate) bytes_per_row: u64,
    /// The bytes of one of its rows.
    pub(crate) row_bytes: u64,
    /// The rows of the picture's crop.
    pub(crate) rows: u64,
    /// The rows that may be written, `rows` or more ([`OutputPlane::rows_held`]).
    pub(crate) rows_held: u64,
    /// Whether the bytes are staged for a copy into the plane's device cache.
    pub(crate) staged: bool,
}

impl<P: Platform> HostPlane<P> {
    /// The bytes from the plane's first to the end of its last row that may be written.
    pub(crate) fn len(&self) -> u64 {
        plane_end(0, self.bytes_per_row, self.rows_held, self.row_bytes).unwrap_or(0)
    }
}

/// Both planes' host bytes ([`Landing::host`]).
pub(crate) struct HostPlanes<P: Platform> {
    pub(crate) luma: HostPlane<P>,
    pub(crate) chroma: HostPlane<P>,
}

/// The rows of one plane of a picture in host memory, which [`Landing::copy_rows`] copies.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SourcePlane {
    /// The plane's first byte.
    pub(crate) at: *const u8,
    /// Bytes from one of its rows to the next.
    pub(crate) bytes_per_row: usize,
    /// The bytes of one of its rows that hold the picture.
    pub(crate) row_bytes: usize,
    /// Its rows.
    pub(crate) rows: usize,
}

/// A picture's landing in its decode's output ([`HostPicture::land`]).
pub(crate) struct Landing<'a, P: Platform> {
    /// The output's luma plane.
    pub(crate) luma: &'a OutputPlane<P>,
    /// The output's chroma plane.
    pub(crate) chroma: &'a OutputPlane<P>,
    stage: &'a mut Stage<P>,
    /// Where each plane in a device cache is staged in `stage`'s buffer, once asked.
    staged: Option<Staged>,
    /// Whether the CPU copied the picture.
    by_cpu: bool,
}

/// Where the planes in a device cache are staged in the staging buffer: each one's first byte.
#[derive(Clone, Copy, Debug, Default)]
struct Staged {
    luma: Option<u64>,
    chroma: Option<u64>,
}

impl<P: Platform> Landing<'_, P> {
    /// The host bytes the picture's planes land in: each plane's own in the surface's pages, and,
    /// for a plane in a device cache, the session's staging buffer at the plane's pitch, which the
    /// session copies into the cache once the landing returns.
    pub(crate) fn host(&mut self) -> Result<HostPlanes<P>, BackendError> {
        let luma_stage = self.luma.cached.then(|| staged_len(self.luma));
        let chroma_stage = self.chroma.cached.then(|| staged_len(self.chroma));
        let staged = match self.staged {
            Some(staged) => staged,
            None => {
                let luma = luma_stage.map(|_| 0);
                let chroma_at = luma_stage
                    .unwrap_or(0)
                    .checked_next_multiple_of(STAGING_ALIGNMENT)
                    .ok_or("a staged plane past 2^64 bytes")?;
                let chroma = chroma_stage.map(|_| chroma_at);
                let total = match chroma_stage {
                    Some(len) => chroma_at.checked_add(len),
                    None => luma_stage,
                };
                if let Some(total) = total {
                    self.stage.buffer_of(total)?;
                }
                let staged = Staged { luma, chroma };
                self.staged = Some(staged);
                staged
            }
        };
        let plane = |p: &OutputPlane<P>, at: Option<u64>| -> Result<HostPlane<P>, BackendError> {
            match (at, p.host) {
                (Some(at), _) => {
                    let buffer = self.stage.buffer.as_ref().ok_or("no staging buffer")?;
                    Ok(HostPlane {
                        buffer: buffer.clone_ref(),
                        at: buffer.contents().wrapping_add(at),
                        bytes_per_row: p.bytes_per_row,
                        row_bytes: p.row_bytes,
                        rows: p.height,
                        rows_held: p.height,
                        staged: true,
                    })
                }
                (None, Some(host)) => Ok(HostPlane {
                    buffer: p.buffer.clone_ref(),
                    at: host,
                    bytes_per_row: p.bytes_per_row,
                    row_bytes: p.row_bytes,
                    rows: p.height,
                    rows_held: p.rows_held,
                    staged: false,
                }),
                (None, None) => Err("a plane with no host bytes".into()),
            }
        };
        Ok(HostPlanes {
            luma: plane(self.luma, staged.luma)?,
            chroma: plane(self.chroma, staged.chroma)?,
        })
    }

    /// [`Self::host`], for a backend whose CPU copies the picture into them itself: the landing is
    /// counted as the CPU's.
    pub(crate) fn host_for_cpu(&mut self) -> Result<HostPlanes<P>, BackendError> {
        let planes = self.host()?;
        self.by_cpu = true;
        Ok(planes)
    }

    /// Copies the picture's planes, `luma` and `chroma`, into the planes' host bytes
    /// ([`Self::host`]) by the CPU: each plane's rows of the picture's crop, the fewer of its
    /// source's and its own, each the fewer of their bytes, leaving the bytes past each row as
    /// they were.
    ///
    /// # Safety
    ///
    /// Each source's `rows` rows of `row_bytes`, `bytes_per_row` apart from `at`, are readable
    /// for the call, and lie apart from the output's planes.
    pub(crate) unsafe fn copy_rows(
        &mut self,
        luma: SourcePlane,
        chroma: SourcePlane,
    ) -> Result<(), BackendError> {
        let planes = self.host()?;
        for (from, to) in [(luma, &planes.luma), (chroma, &planes.chroma)] {
            let rows = u64::try_from(from.rows).map_or(to.rows, |rows| rows.min(to.rows));
            let bytes = u64::try_from(from.row_bytes).map_or(to.row_bytes, |b| b.min(to.row_bytes));
            let (Ok(rows), Ok(bytes), Ok(pitch)) = (
                usize::try_from(rows),
                usize::try_from(bytes),
                usize::try_from(to.bytes_per_row),
            ) else {
                return Err("a plane past the host's address space".into());
            };
            for row in 0..rows {
                // SAFETY: the source's rows are readable (the caller's word); the destination's
                // `rows` rows of `row_bytes` lie in its buffer's host bytes from `at` (the
                // output's check at the decode, or the staging buffer's make), which the landing
                // holds, and the source lies apart from them.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        from.at.add(row * from.bytes_per_row),
                        to.at.as_ptr::<u8>().add(row * pitch),
                        bytes,
                    );
                }
            }
        }
        self.by_cpu = true;
        Ok(())
    }
}

/// The bytes a plane in a device cache takes staged: its crop's rows at its pitch.
fn staged_len<P: Platform>(p: &OutputPlane<P>) -> u64 {
    plane_end(0, p.bytes_per_row, p.height, p.row_bytes).unwrap_or(0)
}

/// What the landing thread keeps from one picture to the next: the staging buffer and the queue
/// its copies into device caches go on, each made at the first picture that needs it.
struct Stage<P: Platform> {
    device: P::Device,
    queue: Option<P::CommandQueue>,
    buffer: Option<P::Buffer>,
}

impl<P: Platform> Stage<P> {
    /// The staging buffer, of at least `len` bytes: the last one where it is that long, else a
    /// new one.
    fn buffer_of(&mut self, len: u64) -> Result<&P::Buffer, BackendError> {
        let kept = self.buffer.take().filter(|b| b.length() >= len);
        let buffer = match kept {
            Some(buffer) => buffer,
            None => (self.device)
                .new_buffer_of_length(OnFault::Return, len, STORAGE_SHARED)
                .map_err(|raise| format!("no staging buffer of {len} bytes: {raise}"))?
                .filter(|b| !b.contents().is_null())
                .ok_or_else(|| format!("no staging buffer of {len} bytes"))?,
        };
        Ok(self.buffer.insert(buffer))
    }

    /// Copies the planes `staged` says are staged into their device caches, on the session's
    /// queue, and waits for the copy.
    fn copy_into_caches(
        &mut self,
        luma: &OutputPlane<P>,
        chroma: &OutputPlane<P>,
        staged: Staged,
    ) -> Result<(), BackendError> {
        let fault =
            |what: &str, raise: Raise| format!("the copy into a device cache: {what}: {raise}");
        let Some(buffer) = self.buffer.as_ref() else {
            return Err("the copy into a device cache: no staging buffer".into());
        };
        if self.queue.is_none() {
            let queue = (self.device.new_command_queue())
                .ok_or("the copy into a device cache: no command queue")?;
            queue.set_label("reims-vgpu.graphics-video-copy");
            self.queue = Some(queue);
        }
        let Some(queue) = self.queue.as_ref() else {
            return Err("the copy into a device cache: no command queue".into());
        };
        let cb = (queue.command_buffer(OnFault::Return))
            .map_err(|r| fault("its command buffer", r))?
            .ok_or("the copy into a device cache: no command buffer")?;
        let blit = (cb.blit_encoder(OnFault::Return))
            .map_err(|r| fault("its blit encoder", r))?
            .ok_or("the copy into a device cache: no blit encoder")?;
        for (plane, at) in [(luma, staged.luma), (chroma, staged.chroma)] {
            let Some(at) = at else { continue };
            for (from, to, size) in plane_copies(at, plane) {
                let copy = BlitCmd::CopyBufferToBuffer {
                    source_buffer: Some(buffer),
                    source_offset: from,
                    destination_buffer: Some(&plane.buffer),
                    destination_offset: to,
                    size,
                };
                blit.encode(OnFault::Return, copy)
                    .map_err(|r| fault("a copy", r))?;
            }
        }
        (blit.encode(OnFault::Return, BlitCmd::EndEncoding)).map_err(|r| fault("its end", r))?;
        cb.commit(OnFault::Return)
            .map_err(|r| fault("its commit", r))?;
        (cb.wait_until_completed(OnFault::Return)).map_err(|r| fault("its wait", r))?;
        if cb.status() != CommandBufferStatus::COMPLETED {
            let error = cb.error().map(|e| format!("{e:?}")).unwrap_or_default();
            return Err(format!("the copy into a device cache failed {error}").into());
        }
        Ok(())
    }
}

/// The copies of a plane staged at `at` into its buffer, each (staged offset, buffer offset,
/// size): one of its crop's rows from first to last where its rows fill its pitch, else one a
/// row, so the bytes past each row (another texture's, inside a larger plane) stay.
fn plane_copies<P: Platform>(at: u64, plane: &OutputPlane<P>) -> Vec<(u64, u64, u64)> {
    let (pitch, bytes) = (plane.bytes_per_row, plane.row_bytes);
    if bytes >= pitch {
        let size = plane_end(0, pitch, plane.height, bytes).unwrap_or(0);
        return if size == 0 {
            Vec::new()
        } else {
            vec![(at, plane.offset, size)]
        };
    }
    (0..plane.height)
        .map(|row| (at + row * pitch, plane.offset + row * pitch, bytes))
        .collect()
}

/// The end of a plane's bytes from `at`, its first byte: its last row's last byte, `row_bytes`
/// past that row's start, past which a texture inside a plane whose rows end at its buffer's end
/// has nothing; `None` for no rows, or past 2^64.
fn plane_end(at: u64, bytes_per_row: u64, height: u64, row_bytes: u64) -> Option<u64> {
    let before_last = bytes_per_row.checked_mul(height.checked_sub(1)?)?;
    at.checked_add(before_last)?.checked_add(row_bytes)
}

/// A decode's output: its planes.
struct Output<P: Platform> {
    luma: OutputPlane<P>,
    chroma: OutputPlane<P>,
}

impl<P: Platform> Output<P> {
    /// The output of `decode`'s planes, which it holds; a fault for a plane past its buffer, or
    /// in pages with no host bytes.
    fn of(decode: &VideoDecode<'_, P>) -> RaiseResult<Self> {
        let output_rows = u64::from(decode.output_rows);
        let luma = output_plane(&decode.luma, output_rows)?;
        let chroma = output_plane(&decode.chroma, output_rows.div_ceil(2))?;
        Ok(Self { luma, chroma })
    }
}

/// The output plane of the op's `plane`, holding `rows` rows where they lie in its buffer
/// ([`OutputPlane::rows_held`]).
fn output_plane<P: Platform>(plane: &VideoPlane<'_, P>, rows: u64) -> RaiseResult<OutputPlane<P>> {
    let length = plane.buffer.length();
    let end = |rows| plane_end(plane.offset, plane.bytes_per_row, rows, plane.row_bytes);
    if !end(plane.height).is_some_and(|end| end <= length) {
        return Err(decode_fault("an output plane past its buffer"));
    }
    let rows_held = if rows > plane.height && end(rows).is_some_and(|end| end <= length) {
        rows
    } else {
        plane.height
    };
    let host = if plane.cached {
        None
    } else {
        let contents = plane.buffer.contents();
        if contents.is_null() {
            return Err(decode_fault("an output plane in pages with no host bytes"));
        }
        Some(contents.wrapping_add(plane.offset))
    };
    Ok(OutputPlane {
        buffer: plane.buffer.clone_ref(),
        offset: plane.offset,
        bytes_per_row: plane.bytes_per_row,
        row_bytes: plane.row_bytes,
        width: plane.width,
        height: plane.height,
        rows_held,
        cached: plane.cached,
        host,
        pages: plane.pages.clone(),
    })
}

/// The bytes of `decode`'s access unit, in place in its bitstream buffer.
fn access_unit<'a, P: Platform>(decode: &VideoDecode<'a, P>) -> RaiseResult<&'a [u8]> {
    let contents = decode.bitstream.contents();
    let (Ok(start), Ok(len)) = (usize::try_from(decode.offset), usize::try_from(decode.size))
    else {
        return Err(decode_fault("an access unit past the host's address space"));
    };
    if contents.is_null() {
        return Err(decode_fault(
            "an access unit in a buffer with no host bytes",
        ));
    }
    // SAFETY: the buffer's `length` bytes from `contents` hold `offset..offset + size` (the op's
    // check), and stay mapped while the buffer lives, which `decode` holds for `'a`.
    Ok(unsafe { std::slice::from_raw_parts(contents.as_ptr::<u8>().add(start), len) })
}

/// A command-buffer fault for a decode the session could not take.
fn decode_fault(what: impl std::fmt::Display) -> Raise {
    op_fault(format_args!("a video decode: {what}"))
}

/// One decode's work for the landing thread: its picture, its output and its bitstream's buffer,
/// held until its value is signalled.
struct Job<P: Platform> {
    value: u64,
    picture: Option<Box<dyn HostPicture<P>>>,
    output: Output<P>,
    _bitstream: P::Buffer,
}

/// How a picture landed, which the session counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Way {
    /// In the surface's pages, by the backend's hardware.
    Pages,
    /// In the surface's pages, by the CPU's copy.
    PagesByCpu,
    /// In a device cache, by the backend's hardware there.
    Cache,
    /// In a device cache, through the staging buffer the backend's hardware wrote, and a copy.
    Staged,
    /// In a device cache, through the staging buffer the CPU wrote, and a copy.
    StagedByCpu,
}

impl Way {
    const ALL: [Way; 5] = [
        Way::Pages,
        Way::PagesByCpu,
        Way::Cache,
        Way::Staged,
        Way::StagedByCpu,
    ];

    fn of(staged: bool, cached: bool, by_cpu: bool) -> Self {
        match (staged, cached, by_cpu) {
            (true, _, false) => Way::Staged,
            (true, _, true) => Way::StagedByCpu,
            (false, true, _) => Way::Cache,
            (false, false, false) => Way::Pages,
            (false, false, true) => Way::PagesByCpu,
        }
    }

    fn says(self) -> &'static str {
        match self {
            Way::Pages => "in the surface's pages, by the hardware",
            Way::PagesByCpu => "in the surface's pages, by a CPU copy",
            Way::Cache => "in a device cache, by the hardware",
            Way::Staged => "in a device cache, staged by the hardware and copied",
            Way::StagedByCpu => "in a device cache, staged by a CPU copy and copied",
        }
    }
}

/// A session's counts of its pictures, logged at its end.
#[derive(Debug, Default)]
struct Counts {
    /// Pictures landed, by [`Way`] (its index in [`Way::ALL`]).
    landed: [u64; 5],
    /// Decodes that answered no picture.
    skipped: u64,
    /// Pictures that failed to land.
    failed: u64,
}

/// A video session on a host API ([`HostDecoder`]): its decodes' pictures landed by its landing
/// thread, which signals its shared event from the host.
pub(crate) struct HostVideoSession<P: Platform> {
    codec: u32,
    /// The backend's name, which the log lines say.
    name: &'static str,
    state: Mutex<State<P>>,
    event: P::SharedEvent,
    counts: Arc<Mutex<Counts>>,
    landing: Option<std::thread::JoinHandle<()>>,
}

/// A session's decoder, the landing thread's queue, and the last value a decode answered.
struct State<P: Platform> {
    decoder: Box<dyn HostDecoder<P>>,
    jobs: Option<mpsc::SyncSender<Job<P>>>,
    value: u64,
}

impl<P: Platform> HostVideoSession<P> {
    /// A session of `codec` (`RGFX_VIDEO_CODEC_*`) on `decoder`, the backend `name`'s, its event
    /// and its copies into device caches on `device`; `None`, logged, where the device makes no
    /// shared event or no thread starts.
    pub(crate) fn new(
        device: &P::Device,
        codec: u32,
        name: &'static str,
        decoder: Box<dyn HostDecoder<P>>,
    ) -> Option<Self> {
        let Some(event) = device.new_shared_event() else {
            log::warn!("hardware video: {name}: no shared event for a session");
            return None;
        };
        // The landing thread holds the picture it lands, and the queue the rest.
        let (jobs, taken) = mpsc::sync_channel(decoder.most_unlanded().max(1) - 1);
        let counts = Arc::new(Mutex::new(Counts::default()));
        let stage = Stage {
            device: device.clone_ref(),
            queue: None,
            buffer: None,
        };
        let landing = {
            let (event, counts) = (event.clone_ref(), Arc::clone(&counts));
            std::thread::Builder::new()
                .name("reims-vgpu.graphics-video-land".into())
                .spawn(move || land_pictures(&taken, stage, &event, &counts, name))
        };
        let landing = match landing {
            Ok(landing) => landing,
            Err(error) => {
                log::warn!("hardware video: {name}: no landing thread for a session ({error})");
                return None;
            }
        };
        Some(Self {
            codec,
            name,
            state: Mutex::new(State {
                decoder,
                jobs: Some(jobs),
                value: 0,
            }),
            event,
            counts,
            landing: Some(landing),
        })
    }
}

impl<P: Platform> GraphicsVideoSession<P> for HostVideoSession<P> {
    fn event(&self) -> EventRef<'_, P> {
        EventRef::Shared(&self.event)
    }

    fn codec(&self) -> u32 {
        self.codec
    }

    fn decode(&self, decode: &VideoDecode<'_, P>) -> RaiseResult<u64> {
        let output = Output::of(decode)?;
        let au = access_unit(decode)?;
        let mut state = self.state.lock();
        if !decode.parameter_sets.is_empty() {
            (state.decoder.parameter_sets(decode.parameter_sets))
                .map_err(|error| decode_fault(format_args!("its parameter sets: {error}")))?;
        }
        let picture = state.decoder.decode(au).map_err(decode_fault)?;
        let value = state.value + 1;
        let job = Job {
            value,
            picture,
            output,
            _bitstream: decode.bitstream.clone_ref(),
        };
        let sent = (state.jobs.as_ref()).is_some_and(|jobs| jobs.send(job).is_ok());
        if !sent {
            return Err(decode_fault("the session's landing thread is gone"));
        }
        state.value = value;
        Ok(value)
    }
}

impl<P: Platform> Drop for HostVideoSession<P> {
    fn drop(&mut self) {
        let last = {
            let mut state = self.state.lock();
            drop(state.jobs.take());
            state.value
        };
        let millis = u64::try_from(TEARDOWN_WAIT.as_millis()).unwrap_or(u64::MAX);
        let landed = last == 0 || self.event.wait_until_signaled_value(last, millis);
        let name = self.name;
        match self.landing.take() {
            Some(thread) if landed => {
                if thread.join().is_err() {
                    log::error!("hardware video: {name}: a session's landing thread panicked");
                }
            }
            Some(_) => log::error!(
                "hardware video: {name}: a session's last picture did not land within \
                 {TEARDOWN_WAIT:?}"
            ),
            None => {}
        }
        let counts = self.counts.lock();
        let landed = (Way::ALL.iter().zip(counts.landed))
            .filter(|(_, n)| *n > 0)
            .map(|(way, n)| format!("{n} {}", way.says()))
            .collect::<Vec<_>>();
        log::info!(
            "hardware video: {name}: a session ended after {last} decodes: landed {}; {} skipped, \
             {} failed",
            if landed.is_empty() {
                "none".to_owned()
            } else {
                landed.join(", ")
            },
            counts.skipped,
            counts.failed
        );
    }
}

/// The landing thread: lands each decode's picture in its output, in the order the decodes were
/// taken, and signals each one's value.
fn land_pictures<P: Platform>(
    jobs: &mpsc::Receiver<Job<P>>,
    mut stage: Stage<P>,
    event: &P::SharedEvent,
    counts: &Mutex<Counts>,
    name: &'static str,
) {
    for job in jobs {
        let Job {
            value,
            picture,
            output,
            _bitstream,
        } = job;
        match picture.map(|picture| land(picture, &output, &mut stage)) {
            None => counts.lock().skipped += 1,
            Some(Ok(way)) => {
                let mut counts = counts.lock();
                if let Some(i) = Way::ALL.iter().position(|w| *w == way) {
                    if counts.landed[i] == 0 {
                        log::info!("hardware video: {name}: a picture landed {}", way.says());
                    }
                    counts.landed[i] += 1;
                }
            }
            Some(Err(error)) => {
                let mut counts = counts.lock();
                counts.failed += 1;
                if counts.failed <= FAILURES_LOGGED {
                    log::error!("hardware video: {name}: a picture did not land: {error}");
                }
            }
        }
        // The output's holds go, and its bracket ends, before the guest may read it.
        drop((output, _bitstream));
        if let Err(raise) = event.set_signaled_value(value) {
            log::error!("hardware video: {name}: the session's signal of {value}: {raise}");
        }
    }
}

/// Lands `picture` in `output`, copying its staged planes into their device caches, and answers
/// how.
fn land<P: Platform>(
    picture: Box<dyn HostPicture<P>>,
    output: &Output<P>,
    stage: &mut Stage<P>,
) -> Result<Way, BackendError> {
    let mut landing = Landing {
        luma: &output.luma,
        chroma: &output.chroma,
        stage,
        staged: None,
        by_cpu: false,
    };
    picture.land(&mut landing)?;
    let Landing {
        staged,
        by_cpu,
        stage,
        ..
    } = landing;
    let is_staged = staged.is_some_and(|s| s.luma.is_some() || s.chroma.is_some());
    if let Some(staged) = staged.filter(|_| is_staged) {
        stage.copy_into_caches(&output.luma, &output.chroma, staged)?;
    }
    let cached = output.luma.cached || output.chroma.cached;
    Ok(Way::of(is_staged, cached, by_cpu))
}

#[cfg(test)]
mod tests;
