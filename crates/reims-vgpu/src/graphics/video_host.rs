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
//!   answers no picture (one the stream's random access skips, or one of the stream's pictures
//!   that does not decode: below) is signalled in its turn, and its output keeps its bytes. Each
//!   plane of an output lies in one of two places:
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
//!   error leaves one. Where the hardware wrote the output's device caches, the caches are said
//!   written before the signal (`Landing::failed_in_caches`), so the device reads what it left
//!   there. Where its work ran past its deadline and may still write (a stall), the landing
//!   first waits up to `STALL_HOLD` for the work the backend kept to be let go, its device done
//!   with it, and the process dies past it: no read of the caches comes before their acquire,
//!   and none waits for them unbounded. A picture the hardware answered failed that the backend
//!   lands all the same, as the hardware left it ([`Landing::hardware_failed`]), is logged and
//!   counted apart, and lands as any other: a Mac's hardware H.264 decoder hands such a picture
//!   on with no error (`super::nvidia`). A copy into the caches that the platform makes no blit
//!   encoder for is skipped, as the decoder skips whole a pass the platform makes no encoder of:
//!   the picture lands with the caches as they were, and is not counted as a failure to land
//!   (`Stage::copy_with_caches`).
//!
//!   **A short picture**, one whose write covers less of a plane than its crop (each plane's
//!   `width` by `height` from its first byte: a stream smaller than the guest's surfaces), lands
//!   with zeros in the rest of the crop, on every way it lands: no texel past it keeps what the
//!   plane held, another picture's or the staging buffer's earlier bytes. The CPU's copy zeros
//!   past what it copies ([`Landing::copy_rows`]); a backend whose hardware writes the picture
//!   says first what it covers ([`Landing::writes_only`]), and the session zeros the rest of each
//!   plane in host bytes by the CPU, and clears a plane in a device cache whole on its queue,
//!   before the write. The rows past the crop that a plane holds for a coded picture
//!   ([`OutputPlane::rows_held`]) are no texels of it, and keep what they held. A picture that
//!   fails to land in a new output's planes, whose fill its decode took as met
//!   (`SurfaceTarget::zeros`), has them cleared whole the same way, so they read the zeros the
//!   fill would have left ([`land_pictures`]).
//!
//!   A picture holds the hardware's surface until it has landed, so the session holds no more
//!   pictures unlanded than the decoder spares ([`HostDecoder::most_unlanded`]): a decode past
//!   them waits, once its picture is decoded, for the oldest to land. The landing waits only
//!   for the hardware, so the wait ends.
//! - **A picture of the stream that does not decode** is answered with no picture
//!   ([`picture_error`], which the backend answers as its error, [`NotDecoded`], and the session
//!   takes as no picture), its command buffer completing as any other: one whose bytes do not
//!   read as its standard reads them, one that predicts from a picture the decoder does not
//!   hold (as a stream's first pictures do where it starts at a P picture), and, in an HEVC
//!   stream, one after a picture that did not decode, before the decoding starts afresh at an
//!   IRAP picture: the codec crate's `is_picture_error`. The session goes on: the H.264 pictures
//!   after one that did not decode decode and land as any (the codec crate's H.264 decoder),
//!   and HEVC's from the IRAP picture. A Mac's hardware decoders answer such a picture alone and
//!   go on (measured with VideoToolbox on a UHD 630, macOS 26.7.1): H.264's with no error, and
//!   for most of them an image it conceals; HEVC's with `kVTVideoDecoderBadDataErr` or
//!   `kVTVideoDecoderReferenceMissingErr` and no image. The output's earlier bytes, which the
//!   guest's decoder hands on here, are the device's own, not Apple's concealment. What is no
//!   picture's (a stream the decoder does not take, an `avcC` or `hvcC` record's error) and
//!   the hardware's or the host API's errors fault the command buffer. A session's first such
//!   picture is logged at `warn`, so a report at the default level shows that a guest was handed
//!   an output the picture did not land in (a new output's zeros, which show green), and its end
//!   counts them all.
//! - **The event** is a shared event, as the host signals it: the runtime holds a command
//!   buffer's wait for it on the host until the value is reached.
//! - **An end** ([`Drop`]) closes the landing thread's queue and waits up to [`TEARDOWN_WAIT`]
//!   for the last value: the thread lands every picture it was given first. A picture keeps
//!   alive what it lands from, so a decoder may go before its last pictures land.
//!
//! The session logs at `info` ("hardware video: ...") its start, with its codec; its first
//! decode, with its output's size, the picture its access unit started
//! ([`HostDecoder::last_picture`]: the first slice's type, `frame_num` and order count) and
//! whether it decoded; the first picture of each way it lands; and its counts at its end: its
//! pictures by the way each landed, the decodes that answered none and how many of them were
//! pictures of the stream that do not decode, the outputs they landed in ([`Outputs`]: how many
//! surfaces' memory, and the most of it live at once), and the time the CPU's copies of them
//! took.

#![allow(dead_code, reason = "each of the host's video arms uses a part of it")]

use std::collections::HashMap;
use std::sync::{Arc, mpsc};
use std::time::Duration;

use reims_vgpu_graphics::wire::{RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_CODEC_HEVC};
use reims_vgpu_video::codec::PictureStart;
use reims_vgpu_video::core::device_memory::MemoryIdentity;
#[cfg(unix)]
use reims_vgpu_video::core::device_wait::{KeptMark, Lingering, poll_until};
use reims_vgpu_video::core::runtime::{MonoTime, Runtime};
use reims_vgpu_vulkan::core::host::HostPtr;
use reims_vgpu_vulkan::core::platform::Handle as _;
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::blit::{BlitCmd, BlitEncoder as _};
use reims_vgpu_vulkan::core::platform::command::{CommandBuffer as _, CommandQueue as _};
use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
use reims_vgpu_vulkan::core::platform::event::{EventRef, SharedEvent as _};
#[cfg(unix)]
use reims_vgpu_vulkan::core::platform::render::{RenderCmd, RenderEncoder as _};
use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
#[cfg(unix)]
use reims_vgpu_vulkan::core::platform::resource::ExportedMemory;
use reims_vgpu_vulkan::core::platform::types::CommandBufferStatus;
use reims_vgpu_vulkan::core::platform::types::{MtlOrigin, MtlSize, NsRange};
use reims_vgpu_vulkan::core::raise::{OnFault, Raise, RaiseResult};
use reims_vgpu_vulkan::core::sync::Mutex;

#[cfg(unix)]
use super::decode::SurfaceTarget;
use super::decode::{
    Decoded, GraphicsVideoSession, Landed, PlanePages, VideoDecode, VideoPlane, op_fault,
};
#[cfg(windows)]
use super::drawn::DrawnTarget;

/// How long a session's end waits for its last picture to land.
pub(crate) const TEARDOWN_WAIT: Duration = Duration::from_secs(10);

/// How long a picture's landing that failed in device caches when the hardware's work ran past
/// its deadline waits for that work to be let go, before it acquires the caches
/// ([`Landing::failed_in_caches`]): 10 s on the session's clock, twice the hardware's own deadline
/// (`reims_vgpu_video::core::device_wait::DEVICE_WAIT`), so that the work has 15 s from its first
/// wait to finish. Ours. Past it, the process dies.
#[cfg(unix)]
pub(crate) const STALL_HOLD: Duration = Duration::from_secs(10);

/// `MTLResourceStorageModeShared`: the staging buffers' storage, host bytes the GPU reads.
const STORAGE_SHARED: u64 = 0;

/// Where a plane's staged bytes start in the staging buffer: the chroma plane's start, past the
/// luma plane's bytes, rounded up to this.
const STAGING_ALIGNMENT: u64 = 256;

/// The failures to land a session logs, one line each; the rest are counted.
const FAILURES_LOGGED: u64 = 8;

/// The code of a picture that did not land after its command buffer completed at its decode
/// (`super::decode`'s video module docs, "Completed at the decode"): the session's next decode
/// fails in its stead, its command buffer completing as one whose picture did not land, so that
/// the guest is told of the failure one decode late (docs/conventions.md, "Each degradation is
/// logged once per boot, then counted").
pub(crate) const CARRIED_CODE: &str = "video-landing-failure-carried";

/// The code of a VideoToolbox decoder's picture the CPU wrote into its surface's two planes from
/// a host surface of another layout: the range extensions' packed 4:2:2 and 4:4:4 samples a
/// VA-API driver decodes into (YUY2, Y210, Y212, Y410, Y412, and XYUV where the GPU's gather does
/// not take it) split into luma and Cb/Cr pairs, and NVDEC's three 4:4:4 planes interleaved into
/// two where the GPU's gather does not take them: one CPU pass a picture that a Mac, whose
/// decoder writes the guest's format, does not make (docs/conventions.md, "Each degradation is
/// logged once per boot, then counted").
pub(crate) const CONVERTED_CODE: &str = "video-pictures-cpu-converted";

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
    /// after a CRA picture at the stream's start), and for one of the stream's pictures that does
    /// not decode, which it answers as the error [`picture_error`] makes ([`NotDecoded`]), whose
    /// output gets nothing. Any other error faults the command buffer.
    ///
    /// `au` lies in the guest's bitstream buffer, in place. Its bytes stay mapped and as they are
    /// until the picture answered has landed ([`HostPicture::land`] returned), or, where none is
    /// answered, until the session signals the decode's value: the hardware may read them there
    /// after this returns.
    ///
    /// `output` is the decode's output, where the picture lands; a decoder whose hardware writes
    /// the picture straight into it reads it here (`super::vaapi`).
    fn decode(
        &mut self,
        au: &[u8],
        output: &DecodeOutput<P>,
    ) -> Result<Option<Box<dyn HostPicture<P>>>, BackendError>;

    /// The most pictures it lets its session hold unlanded, queued for the landing thread or
    /// landing, beside its references and the picture being decoded: a picture holds the
    /// hardware's surface until it has landed, and the decoder has this many to spare. A decode
    /// that would hold one more waits, once its picture is decoded, for the oldest to land. One
    /// at the least.
    fn most_unlanded(&self) -> usize;

    /// The picture the access unit last handed to [`Self::decode`] started, whether it decoded
    /// or not, which the session logs of its first decode; `None` where it started none (no
    /// slice, or a first slice header that did not parse), or the decoder reads no headers.
    fn last_picture(&self) -> Option<PictureStart> {
        None
    }
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
    /// The identity of the device cache the buffer is, where `cached` and the cache's memory was
    /// made for the host's video ([`VideoPlane::cache`]); of the cache's surface, where the plane
    /// lands there (`surface`).
    pub(crate) cache: Option<MemoryIdentity>,
    /// Where the plane lands in its cache's surface ([`VideoPlane::surface`]): the plane there,
    /// which `offset`, `bytes_per_row`, `rows_held` and `cache` then say of, in the surface's
    /// memory rather than the buffer.
    #[cfg(unix)]
    pub(crate) surface: Option<SurfaceTarget<P>>,
    /// Where the plane lands in its cache's surface over Direct3D 11 textures
    /// ([`VideoPlane::drawn`]): the plane there, whose picture is drawn into the texture under it,
    /// or else copied into its image by the CPU's copy.
    #[cfg(windows)]
    pub(crate) drawn: Option<DrawnTarget<P>>,
}

impl<P: Platform> OutputPlane<P> {
    /// Says that the host's video wrote the plane's memory, its cache's buffer or its surface,
    /// and that the caller waited for those writes (`Buffer::external_write_done`); `Err` says
    /// why the device will not read them.
    #[cfg(unix)]
    pub(crate) fn external_write_done(&self) -> Result<(), String> {
        match &self.surface {
            Some(target) => target.memory.external_writes_done(),
            None => self.buffer.external_write_done(),
        }
    }

    /// The plane's memory, its cache's buffer's (`Buffer::export_memory`, `dma_buf` for
    /// `Buffer::export_dma_buf`) or its surface's (`SurfaceMemory::export`, whose handle its
    /// cache's make chose), with what an import names; `Err` says why not.
    #[cfg(unix)]
    pub(crate) fn export(&self, dma_buf: bool) -> Result<ExportedMemory, String> {
        match (&self.surface, dma_buf) {
            (Some(target), _) => target.memory.export(),
            (None, true) => self.buffer.export_dma_buf(),
            (None, false) => self.buffer.export_memory(),
        }
    }

    /// Whether the plane lands in its cache's surface (`surface`, on Windows `drawn`).
    pub(crate) fn in_surface(&self) -> bool {
        #[cfg(unix)]
        return self.surface.is_some();
        #[cfg(windows)]
        self.drawn.is_some()
    }
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

/// What a picture's write covers of one plane, from its first byte ([`Landing::writes_only`]):
/// the bytes of each row from its first, and the rows from its first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Extent {
    pub(crate) row_bytes: u64,
    pub(crate) rows: u64,
}

impl Extent {
    /// Nothing of the plane: a write that covers none of it, whose plane is zeros whole.
    pub(crate) const NONE: Extent = Extent {
        row_bytes: 0,
        rows: 0,
    };

    /// What a 4:2:0 picture of `width` by `height` luma samples of `sample` bytes covers of its
    /// planes from their first byte, the luma's then the chroma's: the chroma's rows half as
    /// many, rounded up, and its rows a pair of samples for every two columns, the last one too
    /// where the width is odd.
    pub(crate) fn of_420(width: u64, height: u64, sample: u64) -> [Extent; 2] {
        [
            Extent {
                row_bytes: width.saturating_mul(sample),
                rows: height,
            },
            Extent {
                row_bytes: width.div_ceil(2).saturating_mul(sample.saturating_mul(2)),
                rows: height.div_ceil(2),
            },
        ]
    }

    /// Whether it covers a plane's crop of `rows` rows of `row_bytes` each.
    fn covers(self, row_bytes: u64, rows: u64) -> bool {
        self.row_bytes >= row_bytes && self.rows >= rows
    }
}

/// A picture's landing in its decode's output ([`HostPicture::land`]).
pub(crate) struct Landing<'a, P: Platform> {
    /// The output's luma plane.
    pub(crate) luma: &'a OutputPlane<P>,
    /// The output's chroma plane.
    pub(crate) chroma: &'a OutputPlane<P>,
    stage: &'a mut Stage<P>,
    /// The clock the CPU's copy is timed on: the video crates' runtime's.
    clock: &'a dyn Runtime,
    /// Where each plane in a device cache is staged in `stage`'s buffer, once asked.
    staged: Option<Staged>,
    /// What the backend's write covers of each plane, luma's then chroma's, once it said so
    /// ([`Landing::writes_only`]): the planes staged after are zeroed past it.
    only: Option<[Extent; 2]>,
    /// Whether the CPU copied the picture.
    by_cpu: bool,
    /// Whether the GPU copied the picture into the output's caches from a scratch the hardware
    /// wrote ([`Landing::copied_by_gpu`]).
    by_gpu_copy: bool,
    /// Whether the hardware decoded the picture straight into the output's planes
    /// ([`Landing::decoded_there`]).
    decoded: bool,
    /// What the hardware answered of a decode it failed whose picture lands all the same
    /// ([`Landing::hardware_failed`]).
    hardware_failed: Option<String>,
    /// When the CPU's copy of the picture started, once [`Landing::host_for_cpu`] answered.
    cpu_from: Option<MonoTime>,
    /// How long [`Landing::copy_rows`]'s copy took.
    cpu_took: Option<Duration>,
    /// Whether the picture landed, for the command buffer that waits for it, which its decode's
    /// end may complete ([`Landing::decode_done`]).
    landed: &'a Arc<Landed>,
}

/// Where the planes in a device cache are staged in the staging buffer: each one's first byte.
#[derive(Clone, Copy, Debug, Default)]
struct Staged {
    luma: Option<u64>,
    chroma: Option<u64>,
}

impl<P: Platform> Landing<'_, P> {
    /// Says that the picture's decode is done, and what is left of its landing is GPU work into
    /// the output's device caches, which every reader of the output waits for: where both planes
    /// lie in caches, the command buffer that waits for the picture may post its completion now
    /// ([`Landed::decoded`]; `super::decode`'s video module docs, "Completed at the decode"). Said
    /// by the VA-API arm's video processing into the caches, and on Windows by its draw into a
    /// cache's surface over Direct3D 11 textures (`super::vaapi`, `super::drawn`).
    pub(crate) fn decode_done(&self) {
        if self.luma.cached && self.chroma.cached {
            self.landed.decoded();
        }
    }

    /// Says that the hardware decoded the picture straight into the output's planes, so that
    /// nothing was copied (`super::vaapi`).
    pub(crate) fn decoded_there(&mut self) {
        self.decoded = true;
    }

    /// Says that the GPU copied the picture into the output's device caches from a scratch the
    /// hardware wrote (`super::vaapi`'s GPU copier): it lands staged by the hardware and copied.
    pub(crate) fn copied_by_gpu(&mut self) {
        self.by_gpu_copy = true;
    }

    /// Says that the hardware answered the picture's decode failed, `why`, and that the picture
    /// lands all the same, as the hardware left it: the session counts and logs it, and signals
    /// its value as any landed picture's.
    pub(crate) fn hardware_failed(&mut self, why: String) {
        self.hardware_failed = Some(why);
    }

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
                if let Some(only) = self.only {
                    self.zero_staged(staged, only)?;
                }
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
    /// counted as the CPU's, and its copy timed from here to the landing's return.
    pub(crate) fn host_for_cpu(&mut self) -> Result<HostPlanes<P>, BackendError> {
        let planes = self.host()?;
        self.by_cpu = true;
        self.cpu_from = Some(self.clock.now());
        Ok(planes)
    }

    /// Copies the picture's planes, `luma` and `chroma`, into the planes' host bytes
    /// ([`Self::host`]) by the CPU: each plane's rows of the picture's crop, the fewer of its
    /// source's and its own, each the fewer of their bytes, and zeros in the rest of the plane's
    /// crop where the source is the shorter or the narrower (the module docs, "A short
    /// picture"), leaving the bytes past each row's `row_bytes`, and past its rows, as they were.
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
        let started = self.clock.now();
        for (from, to) in [(luma, &planes.luma), (chroma, &planes.chroma)] {
            let rows = u64::try_from(from.rows).map_or(to.rows, |rows| rows.min(to.rows));
            let bytes = u64::try_from(from.row_bytes).map_or(to.row_bytes, |b| b.min(to.row_bytes));
            let copied = Extent {
                row_bytes: bytes,
                rows,
            };
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
            // SAFETY: as the copy's: the plane's crop lies in the host bytes the landing holds.
            unsafe { zero_past(to.at, to.bytes_per_row, to.row_bytes, to.rows, copied)? };
        }
        self.cpu_took = Some(self.clock.elapsed_since(started));
        self.by_cpu = true;
        Ok(())
    }
}

impl<P: Platform> Landing<'_, P> {
    /// Says that the backend's write of the picture covers `only` of the planes, the luma's then
    /// the chroma's, from each one's first byte, and no more of them, and zeros the rest of each
    /// plane's crop the write leaves, before the write (the module docs, "A short picture"): in
    /// its host bytes by the CPU (the surface's pages, or the staging buffer, now or as
    /// [`Self::host`] stages the plane), and in a device cache the hardware writes by a clear of
    /// the whole plane on the session's queue, waited for. A plane `only` covers is left as it
    /// is. Said before the write, which the clear would overwrite; a backend that writes the
    /// planes whole need not say it.
    pub(crate) fn writes_only(&mut self, only: [Extent; 2]) -> Result<(), BackendError> {
        self.only = Some(only);
        if let Some(staged) = self.staged {
            self.zero_staged(staged, only)?;
        }
        let staged = self.staged.unwrap_or_default();
        let mut clears = Vec::new();
        for (plane, at, extent) in [
            (self.luma, staged.luma, only[0]),
            (self.chroma, staged.chroma, only[1]),
        ] {
            if at.is_some() || extent.covers(plane.row_bytes, plane.height) {
                continue;
            }
            match plane.host {
                // SAFETY: the plane's crop lies in its pages' host bytes (the output's check at
                // the decode), which the decode holds through the landing.
                Some(host) => unsafe {
                    zero_past(
                        host,
                        plane.bytes_per_row,
                        plane.row_bytes,
                        plane.height,
                        extent,
                    )?
                },
                None => clears.push(plane),
            }
        }
        if clears.is_empty() {
            return Ok(());
        }
        self.stage.clear_planes(&clears)
    }

    /// Zeros past `only` each plane staged at `staged` ([`Self::writes_only`]).
    fn zero_staged(&self, staged: Staged, only: [Extent; 2]) -> Result<(), BackendError> {
        for (plane, at, extent) in [
            (self.luma, staged.luma, only[0]),
            (self.chroma, staged.chroma, only[1]),
        ] {
            let Some(at) = at else { continue };
            let buffer = self.stage.buffer.as_ref().ok_or("no staging buffer")?;
            // SAFETY: the staging buffer holds the plane's crop rows at the plane's pitch from
            // `at` (`staged_len`, at its make), and the landing thread alone writes it.
            unsafe {
                zero_past(
                    buffer.contents().wrapping_add(at),
                    plane.bytes_per_row,
                    plane.row_bytes,
                    plane.height,
                    extent,
                )?
            };
        }
        Ok(())
    }
}

/// Zeros the bytes of a plane's crop, `rows` rows of `row_bytes`, `bytes_per_row` apart from
/// `at`, that `extent` leaves: those past its bytes in each of its rows, and its rows past its
/// own. The bytes past each row's `row_bytes` stay as they were.
///
/// # Safety
///
/// The crop's rows lie in host bytes from `at` that nothing else reads or writes meanwhile.
unsafe fn zero_past(
    at: HostPtr,
    bytes_per_row: u64,
    row_bytes: u64,
    rows: u64,
    extent: Extent,
) -> Result<(), BackendError> {
    if extent.covers(row_bytes, rows) {
        return Ok(());
    }
    let (Ok(pitch), Ok(row_bytes), Ok(rows)) = (
        usize::try_from(bytes_per_row),
        usize::try_from(row_bytes),
        usize::try_from(rows),
    ) else {
        return Err("a plane past the host's address space".into());
    };
    let kept = usize::try_from(extent.row_bytes).map_or(row_bytes, |b| b.min(row_bytes));
    let full = usize::try_from(extent.rows).map_or(rows, |r| r.min(rows));
    for row in 0..rows {
        let from = if row < full { kept } else { 0 };
        // SAFETY: the row's `row_bytes` lie in the crop, in the host bytes the caller holds.
        unsafe {
            std::ptr::write_bytes(
                at.as_ptr::<u8>().add(row * pitch + from),
                0,
                row_bytes - from,
            );
        }
    }
    Ok(())
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
    /// The session's queue for its GPU work into device caches, made at its first.
    fn queue(&mut self) -> Result<&P::CommandQueue, &'static str> {
        if self.queue.is_none() {
            let queue = (self.device.new_command_queue()).ok_or("no command queue")?;
            queue.set_label("reims-vgpu.graphics-video-copy");
            self.queue = Some(queue);
        }
        self.queue.as_ref().ok_or("no command queue")
    }

    /// Clears each of `planes`, each in a device cache, to zeros whole, its crop's rows, on the
    /// session's queue, and waits for the clear ([`Landing::writes_only`]): a plane in its
    /// cache's surface by a render pass that clears its image (`SurfaceTarget::clear`), else
    /// by a fill of its bytes in the cache's buffer, from its first row's first byte to its last
    /// row's last. A plane drawn over Direct3D 11 textures takes no clear here: its draw clears
    /// the textures where the picture is short (`super::drawn`).
    fn clear_planes(&mut self, planes: &[&OutputPlane<P>]) -> Result<(), BackendError> {
        let fault = |what: &str, raise: Raise| format!("a plane's clear: {what}: {raise}");
        let queue = (self.queue())
            .map_err(|why| format!("a plane's clear: {why}"))?
            .clone_ref();
        let cb = (queue.command_buffer(OnFault::Return))
            .map_err(|r| fault("its command buffer", r))?
            .ok_or("a plane's clear: no command buffer")?;
        let mut fills = Vec::new();
        for plane in planes {
            #[cfg(unix)]
            if let Some(target) = &plane.surface {
                let clear = target
                    .clear
                    .as_ref()
                    .ok_or("a plane's clear: no clear pass")?;
                let encoder = (cb.render_encoder(OnFault::Return, clear))
                    .map_err(|r| fault("its render encoder", r))?
                    .ok_or("a plane's clear: no render encoder")?;
                (encoder.encode(OnFault::Return, RenderCmd::EndEncoding))
                    .map_err(|r| fault("its end", r))?;
                continue;
            }
            #[cfg(windows)]
            if plane.drawn.is_some() {
                return Err("a plane's clear: a plane drawn over Direct3D 11 textures".into());
            }
            if let Some(len) = plane_end(0, plane.bytes_per_row, plane.height, plane.row_bytes) {
                fills.push((&plane.buffer, NsRange::new(plane.offset, len)));
            }
        }
        if !fills.is_empty() {
            let blit = (cb.blit_encoder(OnFault::Return))
                .map_err(|r| fault("its blit encoder", r))?
                .ok_or("a plane's clear: no blit encoder")?;
            for (buffer, range) in fills {
                let fill = BlitCmd::FillBuffer {
                    buffer,
                    range,
                    value: 0,
                };
                blit.encode(OnFault::Return, fill)
                    .map_err(|r| fault("a fill", r))?;
            }
            (blit.encode(OnFault::Return, BlitCmd::EndEncoding))
                .map_err(|r| fault("its end", r))?;
        }
        cb.commit(OnFault::Return)
            .map_err(|r| fault("its commit", r))?;
        (cb.wait_until_completed(OnFault::Return)).map_err(|r| fault("its wait", r))?;
        if cb.status() != CommandBufferStatus::COMPLETED {
            let error = cb.error().map(|e| format!("{e:?}")).unwrap_or_default();
            return Err(format!("a plane's clear failed {error}").into());
        }
        Ok(())
    }

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
        self.copy_with_caches(luma, chroma, staged, true)
    }

    /// Copies each plane `staged` says is staged between the staging buffer and its device
    /// cache, into the cache where `into`, else out of it, on the session's queue, and waits for
    /// the copy.
    ///
    /// **A copy into the caches the platform makes no blit encoder for** (on the rt platform, an
    /// encoder metal2vulkan-runtime refuses, which it counts under `rt-refused-<kind>`) is
    /// skipped, as the decoder skips whole a pass the platform makes no encoder of
    /// (`decode::cmdbuf`'s `unmade`): the caches keep their bytes, and the picture lands with
    /// them, its value signalled and its command buffer completing as any landed picture's. The
    /// process's first is logged. A copy out of the caches, an encode's source, answers the
    /// refusal as an error, which fails the frame: skipped, it would hand the encoder the staging
    /// buffer's earlier bytes, another frame's or what the buffer held when it was made.
    fn copy_with_caches(
        &mut self,
        luma: &OutputPlane<P>,
        chroma: &OutputPlane<P>,
        staged: Staged,
        into: bool,
    ) -> Result<(), BackendError> {
        let way = if into { "into" } else { "out of" };
        let fault =
            |what: &str, raise: Raise| format!("the copy {way} a device cache: {what}: {raise}");
        let queue = (self.queue())
            .map_err(|why| format!("the copy {way} a device cache: {why}"))?
            .clone_ref();
        let Some(buffer) = self.buffer.as_ref() else {
            return Err(format!("the copy {way} a device cache: no staging buffer").into());
        };
        let cb = (queue.command_buffer(OnFault::Return))
            .map_err(|r| fault("its command buffer", r))?
            .ok_or_else(|| format!("the copy {way} a device cache: no command buffer"))?;
        let Some(blit) =
            (cb.blit_encoder(OnFault::Return)).map_err(|r| fault("its blit encoder", r))?
        else {
            if into {
                log::warn!(
                    "hardware video: the platform made no blit encoder for a picture's copy into \
                     its device caches: the copy is skipped, and the caches keep their bytes"
                );
                return Ok(());
            }
            return Err(format!("the copy {way} a device cache: no blit encoder").into());
        };
        for (plane, at) in [(luma, staged.luma), (chroma, staged.chroma)] {
            let Some(at) = at else { continue };
            // A plane that lands in its cache's surface is copied into its image, staged at the
            // surface's pitch (on Windows, at the plane's own).
            #[cfg(unix)]
            let image = plane.surface.as_ref().map(|target| &target.texture);
            #[cfg(windows)]
            let image = plane.drawn.as_ref().map(|target| &target.texture);
            if let Some(texture) = image.filter(|_| into) {
                let copy = BlitCmd::CopyBufferToTexture {
                    source_buffer: buffer,
                    source_offset: at,
                    source_bytes_per_row: plane.bytes_per_row,
                    source_bytes_per_image: plane.bytes_per_row.saturating_mul(plane.height),
                    source_size: MtlSize::new(plane.width, plane.height, 1),
                    destination_texture: texture,
                    destination_slice: 0,
                    destination_level: 0,
                    destination_origin: MtlOrigin::default(),
                };
                blit.encode(OnFault::Return, copy)
                    .map_err(|r| fault("a copy into a surface's plane", r))?;
                continue;
            }
            for (staged_at, cache_at, size) in plane_copies(at, plane) {
                let copy = if into {
                    BlitCmd::CopyBufferToBuffer {
                        source_buffer: Some(buffer),
                        source_offset: staged_at,
                        destination_buffer: Some(&plane.buffer),
                        destination_offset: cache_at,
                        size,
                    }
                } else {
                    BlitCmd::CopyBufferToBuffer {
                        source_buffer: Some(&plane.buffer),
                        source_offset: cache_at,
                        destination_buffer: Some(buffer),
                        destination_offset: staged_at,
                        size,
                    }
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
            return Err(format!("the copy {way} a device cache failed {error}").into());
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
pub(crate) struct DecodeOutput<P: Platform> {
    pub(crate) luma: OutputPlane<P>,
    pub(crate) chroma: OutputPlane<P>,
}

impl<P: Platform> DecodeOutput<P> {
    /// The output of `decode`'s planes, which it holds; a fault for a plane past its buffer, or
    /// in pages with no host bytes.
    fn of(decode: &VideoDecode<'_, P>) -> RaiseResult<Self> {
        let output_rows = u64::from(decode.output_rows);
        let luma = output_plane(&decode.luma, output_rows)?;
        let chroma = output_plane(&decode.chroma, output_rows.div_ceil(2))?;
        Ok(Self { luma, chroma })
    }
}

/// The output plane of the op's `plane`, holding `rows` rows where they lie in its buffer, and
/// in the rows the buffer holds for the plane where the host laid it out
/// ([`VideoPlane::rows_room`]), as many of them as do there ([`OutputPlane::rows_held`]).
fn output_plane<P: Platform>(plane: &VideoPlane<'_, P>, rows: u64) -> RaiseResult<OutputPlane<P>> {
    let length = plane.buffer.length();
    let end = |rows| plane_end(plane.offset, plane.bytes_per_row, rows, plane.row_bytes);
    if !end(plane.height).is_some_and(|end| end <= length) {
        return Err(decode_fault("an output plane past its buffer"));
    }
    let rows = rows.min(plane.rows_room.unwrap_or(u64::MAX));
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
    // In its cache's surface, the plane lies where the surface's memory holds it, with the rows
    // of a coded picture below its own.
    #[cfg(unix)]
    if let Some(target) = &plane.surface {
        return Ok(OutputPlane {
            buffer: plane.buffer.clone_ref(),
            offset: target.offset,
            bytes_per_row: target.bytes_per_row,
            row_bytes: plane.row_bytes,
            width: plane.width,
            height: plane.height,
            rows_held: target.rows.max(plane.height),
            cached: true,
            host: None,
            pages: None,
            cache: Some(target.identity.clone()),
            surface: Some(target.clone_ref()),
        });
    }
    // In its cache's surface over Direct3D 11 textures, the plane lies in its image at the
    // plane's own pitch, staged for the CPU's copy where its draw is not made.
    #[cfg(windows)]
    if let Some(target) = &plane.drawn {
        return Ok(OutputPlane {
            buffer: plane.buffer.clone_ref(),
            offset: plane.offset,
            bytes_per_row: plane.bytes_per_row,
            row_bytes: plane.row_bytes,
            width: plane.width,
            height: plane.height,
            rows_held: plane.height,
            cached: true,
            host: None,
            pages: None,
            cache: Some(target.identity.clone()),
            drawn: Some(target.clone_ref()),
        });
    }
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
        cache: plane.cache.clone(),
        #[cfg(unix)]
        surface: None,
        #[cfg(windows)]
        drawn: None,
    })
}

/// The plane of an encode's source `plane`, its first `rows` rows of `row_bytes` each, which it
/// holds; `Err` with why where the plane holds fewer rows or narrower ones, or lies past its
/// buffer, or in pages with no host bytes.
fn output_plane_of<P: Platform>(
    plane: &VideoPlane<'_, P>,
    rows: u64,
    row_bytes: u64,
) -> Result<OutputPlane<P>, String> {
    if plane.height < rows || plane.row_bytes < row_bytes {
        return Err(format!(
            "a source plane of {} rows of {} bytes, for {rows} rows of {row_bytes}",
            plane.height, plane.row_bytes
        ));
    }
    let end = plane_end(plane.offset, plane.bytes_per_row, rows, row_bytes);
    if !end.is_some_and(|end| end <= plane.buffer.length()) {
        return Err("a source plane past its buffer".to_owned());
    }
    let host = if plane.cached {
        None
    } else {
        let contents = plane.buffer.contents();
        if contents.is_null() {
            return Err("a source plane in pages with no host bytes".to_owned());
        }
        Some(contents.wrapping_add(plane.offset))
    };
    Ok(OutputPlane {
        buffer: plane.buffer.clone_ref(),
        offset: plane.offset,
        bytes_per_row: plane.bytes_per_row,
        row_bytes,
        width: plane.width,
        height: rows,
        rows_held: rows,
        cached: plane.cached,
        host,
        pages: plane.pages.clone(),
        cache: plane.cache.clone(),
        #[cfg(unix)]
        surface: None,
        #[cfg(windows)]
        drawn: None,
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

/// A backend's answer to a decode whose access unit is a picture of the stream that does not
/// decode, `error` (the module docs, "A picture of the stream that does not decode"): the error
/// [`NotDecoded`], which the session answers as no picture, faulting nothing, and logs and counts
/// ([`HostVideoSession::not_decoded`]).
pub(crate) fn picture_error(error: impl std::fmt::Display) -> BackendError {
    Box::new(NotDecoded(error.to_string()))
}

/// A picture of the stream that does not decode, as its decoder said it ([`picture_error`]).
#[derive(Debug)]
pub(crate) struct NotDecoded(String);

impl std::fmt::Display for NotDecoded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NotDecoded {}

/// One decode's work for the landing thread: its picture, its output and its bitstream's buffer,
/// held until its value is signalled.
struct Job<P: Platform> {
    value: u64,
    picture: Option<Box<dyn HostPicture<P>>>,
    output: DecodeOutput<P>,
    _bitstream: P::Buffer,
    /// Whether its picture landed, for the command buffer that waits for `value` ([`Landed`]).
    landed: Arc<Landed>,
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
    /// In a device cache, decoded there by the backend's hardware, with no copy.
    Decoded,
    /// In a device cache, through the staging buffer or scratch the backend's hardware wrote,
    /// and a copy.
    Staged,
    /// In a device cache, through the staging buffer the CPU wrote, and a copy.
    StagedByCpu,
}

impl Way {
    const ALL: [Way; 6] = [
        Way::Pages,
        Way::PagesByCpu,
        Way::Cache,
        Way::Decoded,
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
            Way::Decoded => "in a device cache, decoded there by the hardware",
            Way::Staged => "in a device cache, staged by the hardware and copied",
            Way::StagedByCpu => "in a device cache, staged by a CPU copy and copied",
        }
    }
}

/// `codec`, an `RGFX_VIDEO_CODEC_*`, as the log lines name it.
fn codec_says(codec: u32) -> String {
    match codec {
        RGFX_VIDEO_CODEC_H264 => "H.264".to_owned(),
        RGFX_VIDEO_CODEC_HEVC => "HEVC".to_owned(),
        other => format!("codec {other}"),
    }
}

/// A session's counts of its pictures, logged at its end.
#[derive(Debug, Default)]
struct Counts {
    /// Pictures landed, by [`Way`] (its index in [`Way::ALL`]).
    landed: [u64; 6],
    /// Decodes that answered no picture.
    skipped: u64,
    /// Of them, pictures of the stream that do not decode ([`NotDecoded`]).
    not_decoded: u64,
    /// Pictures that failed to land.
    failed: u64,
    /// Pictures the hardware answered failed that landed as it left them.
    landed_failed: u64,
    /// Pictures the CPU copied ([`Landing::copy_rows`], [`Landing::host_for_cpu`]) that landed.
    cpu_copies: u64,
    /// The time their copies took, all told.
    cpu_total: Duration,
    /// The longest of them.
    cpu_most: Duration,
}

/// The outputs a session's decodes landed in, told apart by their memory: the luma plane's
/// device cache's identity ([`OutputPlane::cache`]), else the identity of the mapping its pages
/// lie in ([`OutputPlane::pages`]). An output's memory is live until its mapping goes (the
/// surface's view of its pages, its block of the region until its `BLOCK_RELEASE`, or the device
/// cache), and an identity is never another mapping's, so the outputs live at a decode are the
/// surfaces whose memory the session's decodes have landed in and that the guest still holds.
#[derive(Debug, Default)]
struct Outputs {
    /// Those whose memory lived at the last decode, by their identity's key.
    live: HashMap<u64, MemoryIdentity>,
    /// The distinct outputs the decodes landed in.
    distinct: u64,
    /// The most live at once, at a decode.
    most_live: usize,
    /// Decodes into memory of no identity: a buffer over memory of another kind.
    unknown: u64,
}

impl Outputs {
    /// Counts `output`, a decode's, and the outputs live at it.
    fn saw<P: Platform>(&mut self, output: &DecodeOutput<P>) {
        let plane = &output.luma;
        let identity = (plane.cache.clone())
            .or_else(|| plane.pages.as_ref().map(|pages| pages.memory.identity()));
        let Some(identity) = identity else {
            self.unknown += 1;
            return;
        };
        self.live.retain(|_, live| live.is_live());
        if self.live.insert(identity.key(), identity).is_none() {
            self.distinct += 1;
        }
        self.most_live = self.most_live.max(self.live.len());
    }
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
    /// The outputs its decodes landed in, logged at its end.
    outputs: Outputs,
    /// Whether its first decode was logged.
    first_said: bool,
}

impl<P: Platform> HostVideoSession<P> {
    /// A session of `codec` (`RGFX_VIDEO_CODEC_*`) on `decoder`, the backend `name`'s, its event
    /// and its copies into device caches on `device`, the CPU's copies timed on `clock`'s clock;
    /// `None`, logged, where the device makes no shared event or no thread starts.
    pub(crate) fn new(
        device: &P::Device,
        clock: Arc<dyn Runtime>,
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
                .spawn(move || land_pictures(&taken, stage, &*clock, &event, &counts, name))
        };
        let landing = match landing {
            Ok(landing) => landing,
            Err(error) => {
                log::warn!("hardware video: {name}: no landing thread for a session ({error})");
                return None;
            }
        };
        log::info!(
            "hardware video: {name}: a session started, of {}",
            codec_says(codec)
        );
        Some(Self {
            codec,
            name,
            state: Mutex::new(State {
                decoder,
                jobs: Some(jobs),
                value: 0,
                outputs: Outputs::default(),
                first_said: false,
            }),
            event,
            counts,
            landing: Some(landing),
        })
    }

    /// Counts `not`, a decode's picture of the stream that does not decode, which lands nothing,
    /// its output keeping its bytes; the session's first is logged at `warn` (the module docs, "A
    /// picture of the stream that does not decode"), the rest counted at its end. Answers whether
    /// it was the first, said.
    fn not_decoded(&self, not: &NotDecoded) -> bool {
        let first = {
            let mut counts = self.counts.lock();
            counts.not_decoded += 1;
            counts.not_decoded == 1
        };
        if first {
            let name = self.name;
            log::warn!(
                kind:% = not;
                "hardware video: {name}: a picture of the stream that does not decode lands \
                 nothing, its output keeping its bytes (a new output's zeros): {not}; the \
                 session's others are counted at its end"
            );
        }
        first
    }
}

impl<P: Platform> GraphicsVideoSession<P> for HostVideoSession<P> {
    fn event(&self) -> EventRef<'_, P> {
        EventRef::Shared(&self.event)
    }

    fn codec(&self) -> u32 {
        self.codec
    }

    fn decode(&self, decode: &VideoDecode<'_, P>) -> RaiseResult<Decoded> {
        let output = DecodeOutput::of(decode)?;
        let au = access_unit(decode)?;
        let mut state = self.state.lock();
        if !decode.parameter_sets.is_empty() {
            (state.decoder.parameter_sets(decode.parameter_sets))
                .map_err(|error| decode_fault(format_args!("its parameter sets: {error}")))?;
        }
        let answer = state.decoder.decode(au, &output);
        if !state.first_said {
            state.first_said = true;
            let decoded = match &answer {
                Ok(Some(_)) => "decoded",
                Ok(None) => "no picture to land",
                Err(error) if error.is::<NotDecoded>() => {
                    "a picture of the stream that does not decode"
                }
                Err(_) => "an error, which faults its command buffer",
            };
            let picture = (state.decoder.last_picture())
                .map_or_else(|| "no picture started".to_owned(), |p| p.to_string());
            log::info!(
                "hardware video: {}: a session's first decode, into a {}x{} output: {picture}; \
                 {decoded}",
                self.name,
                output.luma.width,
                output.luma.height
            );
        }
        let picture = match answer {
            Ok(picture) => picture,
            Err(error) => match error.downcast::<NotDecoded>() {
                Ok(not) => {
                    self.not_decoded(&not);
                    None
                }
                Err(error) => return Err(decode_fault(error)),
            },
        };
        state.outputs.saw(&output);
        let skipped = picture.is_none();
        let value = state.value + 1;
        let landed = Arc::new(Landed::default());
        let job = Job {
            value,
            picture,
            output,
            _bitstream: decode.bitstream.clone_ref(),
            landed: Arc::clone(&landed),
        };
        let sent = (state.jobs.as_ref()).is_some_and(|jobs| jobs.send(job).is_ok());
        if !sent {
            return Err(decode_fault("the session's landing thread is gone"));
        }
        state.value = value;
        Ok(Decoded {
            value,
            skipped,
            landed: Some(landed),
        })
    }
}

impl<P: Platform> Drop for HostVideoSession<P> {
    fn drop(&mut self) {
        let (last, outputs) = {
            let mut state = self.state.lock();
            drop(state.jobs.take());
            (state.value, std::mem::take(&mut state.outputs))
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
        let mut outputs_said = format!(
            "; its outputs lay in {} surfaces' memory, {} live at once at most",
            outputs.distinct, outputs.most_live
        );
        if outputs.unknown > 0 {
            outputs_said += &format!(", and {} in memory of no identity", outputs.unknown);
        }
        let cpu_said = if counts.cpu_copies > 0 {
            let ms = |d: Duration| d.as_secs_f64() * 1e3;
            let mean = counts.cpu_total / u32::try_from(counts.cpu_copies).unwrap_or(u32::MAX);
            format!(
                "; the CPU copied {}, {:.3} ms each, {:.3} ms at most",
                counts.cpu_copies,
                ms(mean),
                ms(counts.cpu_most)
            )
        } else {
            String::new()
        };
        log::info!(
            "hardware video: {name}: a session ended after {last} decodes: landed {}; {} skipped \
             ({} of them pictures of the stream that do not decode), {} failed, {} landed as the \
             hardware failed them{outputs_said}{cpu_said}",
            if landed.is_empty() {
                "none".to_owned()
            } else {
                landed.join(", ")
            },
            counts.skipped,
            counts.not_decoded,
            counts.failed,
            counts.landed_failed
        );
    }
}

/// The landing thread: lands each decode's picture in its output, in the order the decodes were
/// taken, and signals each one's value.
fn land_pictures<P: Platform>(
    jobs: &mpsc::Receiver<Job<P>>,
    mut stage: Stage<P>,
    clock: &dyn Runtime,
    event: &P::SharedEvent,
    counts: &Mutex<Counts>,
    name: &'static str,
) {
    // A picture did not land after its command buffer completed at its decode: the next decode
    // fails in its stead ([`CARRIED_CODE`]).
    let mut carried = false;
    for job in jobs {
        // A step of the exit gate's: the process's exit waits for the picture landing, and one
        // after the exit began parks the thread before the hardware's calls.
        let Some(_step) = reims_vgpu_vulkan::exit_gate::step() else {
            reims_vgpu_vulkan::exit_gate::stay()
        };
        let Job {
            value,
            picture,
            output,
            _bitstream,
            landed,
        } = job;
        if std::mem::take(&mut carried) {
            // Before its landing, which then posts no completion at its decode: its command
            // buffer completes failed, at its end.
            landed.fail();
        }
        match picture.map(|picture| land(picture, &output, &mut stage, clock, &landed)) {
            None => counts.lock().skipped += 1,
            Some(Ok((way, hardware_failed, cpu))) => {
                let mut counts = counts.lock();
                if let Some(took) = cpu {
                    counts.cpu_copies += 1;
                    counts.cpu_total += took;
                    counts.cpu_most = counts.cpu_most.max(took);
                }
                if let Some(why) = hardware_failed {
                    counts.landed_failed += 1;
                    if counts.landed_failed <= FAILURES_LOGGED {
                        log::warn!(
                            kind:% = why;
                            "hardware video: {name}: a picture the hardware failed landed as it \
                             left it: {why}"
                        );
                    }
                }
                if let Some(i) = Way::ALL.iter().position(|w| *w == way) {
                    // A session's first of each way; once per process at info (`kind`).
                    if counts.landed[i] == 0 {
                        log::info!(
                            kind = way.says();
                            "hardware video: {name}: a picture landed {}", way.says()
                        );
                    }
                    counts.landed[i] += 1;
                }
            }
            Some(Err(error)) => {
                #[cfg(unix)]
                clear_unlanded(&output, &mut stage, name);
                // Before the signal: the command buffer waiting for it completes failed, or,
                // where it completed at the decode already, the next decode's does.
                carried = landed.fail();
                if carried {
                    reims_vgpu_vulkan::answer::count_kind(CARRIED_CODE);
                    // Once per process, at its level; the rest at debug, counted.
                    log::warn!(
                        "hardware video: {name}: a picture did not land after its command buffer \
                         completed at its decode: the session's next decode fails in its stead \
                         (counted as {CARRIED_CODE})"
                    );
                }
                let mut counts = counts.lock();
                counts.failed += 1;
                if counts.failed <= FAILURES_LOGGED {
                    log::error!(
                        kind:% = error;
                        "hardware video: {name}: a picture did not land: {error}"
                    );
                }
            }
        }
        // The output's holds go, and its bracket ends, before the guest may read it.
        drop((output, _bitstream));
        if let Err(raise) = event.set_signaled_value(value) {
            log::error!(
                kind:% = raise;
                "hardware video: {name}: the session's signal of {value}: {raise}"
            );
        }
    }
}

/// The planes of `output` whose fill the decode took as met (`SurfaceTarget::zeros`), as its
/// picture failed to land: cleared to zeros whole on the session's queue, as the fill would
/// have left them (the module docs, "A short picture"); a clear that fails is logged.
#[cfg(unix)]
fn clear_unlanded<P: Platform>(output: &DecodeOutput<P>, stage: &mut Stage<P>, name: &str) {
    let planes: Vec<&OutputPlane<P>> = [&output.luma, &output.chroma]
        .into_iter()
        .filter(|plane| (plane.surface.as_ref()).is_some_and(|target| target.zeros))
        .collect();
    if planes.is_empty() {
        return;
    }
    if let Err(error) = stage.clear_planes(&planes) {
        log::error!(
            kind:% = error;
            "hardware video: {name}: a new output's planes a picture did not land in were not \
             cleared: {error}"
        );
    }
}

/// Lands `picture` in `output`, copying its staged planes into their device caches, and answers
/// how, what the hardware answered where it failed the decode and the picture landed all the
/// same ([`Landing::hardware_failed`]), and how long the CPU's copy took where the CPU copied it
/// (from its start to the end of [`Landing::copy_rows`], or to the picture's landing's return).
/// `landed` is told where the decode is done before the landing ends ([`Landing::decode_done`]).
fn land<P: Platform>(
    picture: Box<dyn HostPicture<P>>,
    output: &DecodeOutput<P>,
    stage: &mut Stage<P>,
    clock: &dyn Runtime,
    landed: &Arc<Landed>,
) -> Result<(Way, Option<String>, Option<Duration>), BackendError> {
    let mut landing = Landing {
        luma: &output.luma,
        chroma: &output.chroma,
        stage,
        clock,
        staged: None,
        only: None,
        by_cpu: false,
        by_gpu_copy: false,
        decoded: false,
        hardware_failed: None,
        cpu_from: None,
        cpu_took: None,
        landed,
    };
    picture.land(&mut landing)?;
    let cpu = match (landing.cpu_took, landing.cpu_from) {
        (Some(took), _) => Some(took),
        (None, Some(from)) => Some(clock.elapsed_since(from)),
        (None, None) => None,
    };
    let Landing {
        staged,
        by_cpu,
        by_gpu_copy,
        decoded,
        stage,
        hardware_failed,
        ..
    } = landing;
    let is_staged = staged.is_some_and(|s| s.luma.is_some() || s.chroma.is_some());
    if let Some(staged) = staged.filter(|_| is_staged) {
        stage.copy_into_caches(&output.luma, &output.chroma, staged)?;
    }
    let cached = output.luma.cached || output.chroma.cached;
    let is_staged = is_staged || by_gpu_copy;
    if decoded && cached && !is_staged && !by_cpu {
        return Ok((Way::Decoded, hardware_failed, cpu));
    }
    Ok((Way::of(is_staged, cached, by_cpu), hardware_failed, cpu))
}

/// How the hardware's work ended where a picture failed to land in device caches
/// ([`Landing::failed_in_caches`]).
#[cfg(unix)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum Failed {
    /// It has stopped: nothing it wrote is still being written.
    Stopped,
    /// It ran past its deadline (a stall), and what the landing's thread kept of it from the
    /// mark on (`reims_vgpu_video::core::device_wait::Lingering`) may still write.
    Stalled(KeptMark),
}

impl<P: Platform> Landing<'_, P> {
    /// `error`, the picture's failed landing into `planes`' device caches, which the hardware
    /// wrote: each plane is said written ([`OutputPlane::external_write_done`]), as the landing
    /// would have said it, so the device's reads of the caches see what the hardware left there,
    /// as reads of an IOSurface that a failed decode wrote do, before the session's signal of the
    /// failure. Where the work stalled, its writes may land after an acquire, so the acquire
    /// waits first for the work kept since the mark to be let go, its device done with it, up to
    /// [`STALL_HOLD`] on the session's clock ([`hold_while_kept`]); past it, the process dies. A
    /// refused acquire is logged, and the landing fails with `error` either way, logged and
    /// counted as any failure.
    #[cfg(unix)]
    pub(crate) fn failed_in_caches(
        &self,
        planes: &[&OutputPlane<P>],
        failed: Failed,
        error: BackendError,
    ) -> BackendError {
        if let Failed::Stalled(since) = failed {
            hold_while_kept(self.clock, since, STALL_HOLD, &error);
        }
        for plane in planes {
            if let Err(why) = plane.external_write_done() {
                log::warn!(
                    "hardware video: a picture that failed to land in its device cache, {error}: \
                     the cache's acquire was refused: {why}"
                );
            }
        }
        error
    }
}

/// Waits on `clock` until no work the thread of `since` kept after the mark is held, its device
/// not done with it ([`Lingering::holds_since`]), asking after it as a backend asks after its
/// device ([`poll_until`]): at once where none was kept. Past `within`, the process dies with a
/// fault of the decode (`error`'s): the device's memory the work may still write is never read
/// before it is acquired, and no read of it waits unbounded.
#[cfg(unix)]
fn hold_while_kept(clock: &dyn Runtime, since: KeptMark, within: Duration, error: &BackendError) {
    let held = || Lingering::made().is_some_and(|kept| kept.holds_since(since));
    let let_go = poll_until(clock, clock.deadline_in(within), || (!held()).then_some(()));
    if let_go.is_none() {
        decode_fault(format_args!(
            "a picture's landing in its device caches, {error}: the hardware's work, which may \
             still write them, was not done {within:?} past its deadline"
        ))
        .fatal()
    }
}

mod encode;

// The arms' side of an encode session, which the VA-API and NVENC arms drive.
#[cfg(any(target_os = "linux", windows))]
pub(crate) use encode::{
    EncodeFrame, EncodeSource, Encoded, FrameSource, HostEncodeSession, HostEncodedFrame,
    HostEncoder,
};

#[cfg(test)]
mod tests;
