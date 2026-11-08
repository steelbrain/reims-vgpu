//! The video encoders of the guest's command buffers (`tag >> 12 ==`
//! [`RGFX_OP_GROUP_VIDEO`](reims_graphics::ops::RGFX_OP_GROUP_VIDEO)): the hardware video work of
//! the guest's video driver, a picture a command buffer, on a session
//! ([`RGFX_CMD_VIDEO_SESSION_CREATE`](reims_graphics::wire::RGFX_CMD_VIDEO_SESSION_CREATE)) the
//! platform's [`GraphicsVideo`] makes. The decoder resolves each op's objects (the bitstream's
//! buffer, the buffers the output surface's planes lie in) and hands the work to the session,
//! which starts it at once, as the host takes the committed buffer, after the session's earlier
//! work; the buffer's later work, and its completion, wait for the session's shared event to
//! reach the value the work signals once done. The work does not wait for the buffer's earlier
//! encoders: the guest's video driver encodes none before it. It does wait for the buffers of
//! other queues the command buffer runs after (its `SURFACE_ORDER`), on the CPU, as the session's
//! queue is not the command buffer's ([`super::surface_order`]): a decode whose output another
//! process still reads lands once the read is done.

use std::mem::ManuallyDrop;
use std::sync::Arc;

use reims_graphics::ops::video::{
    RGFX_OP_VIDEO_DECODE_H264, RGFX_OP_VIDEO_DECODE_H265, RgfxVideoOpDecodeH264,
    RgfxVideoOpDecodeH265,
};
use reims_graphics::wire::{
    RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_CODEC_HEVC, RgfxVideoCodecCaps, RgfxVideoOutputLayout,
};
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::command::CommandBuffer as _;
use reims_vgpu_core::platform::event::EventRef;
use reims_vgpu_core::platform::resource::Buffer as _;
use reims_vgpu_core::platform::types::PixelFormatInfo;
use reims_vgpu_core::raise::RaiseResult;
use zerocopy::FromBytes;

use super::worker::Worker;
use super::{GraphicsDecoder, ObjHandle, bad_reference, op_fault};

/// How long a decode waits for its output's device cache to be filled: a copy of the surface's
/// pages, a few milliseconds even across a slow bus.
const FILL_WAIT: std::time::Duration = std::time::Duration::from_secs(2);

/// The host's hardware video for a graphics device: the codecs it takes, which the device's video
/// caps answer, and its sessions.
pub trait GraphicsVideo<P: Platform>: Send + Sync {
    /// One answer for each codec and direction the hardware takes.
    fn caps(&self) -> Vec<RgfxVideoCodecCaps>;
    /// How a decode of `codec`, one of [`Self::caps`], lands its pictures in place, in the
    /// output surfaces' planes; all zero where it lands them by a copy alone.
    fn layout(&self, _codec: &RgfxVideoCodecCaps) -> RgfxVideoOutputLayout {
        RgfxVideoOutputLayout::default()
    }
    /// A session of `codec` (`RGFX_VIDEO_CODEC_*`) in `direction` (`RGFX_VIDEO_DECODE` or
    /// `RGFX_VIDEO_ENCODE`); `None` where the hardware has no such codec.
    fn session(&self, codec: u32, direction: u32) -> Option<Arc<dyn GraphicsVideoSession<P>>>;
    /// Whether a decode lands its picture in an output surface's device cache, where the
    /// surface's block has one, as well as in its pages
    /// ([`RGFX_VIDEO_CAPS_OUTPUT_CACHED`](reims_graphics::wire::RGFX_VIDEO_CAPS_OUTPUT_CACHED)).
    fn lands_in_device_caches(&self) -> bool {
        false
    }
}

/// A hold of a video session, the object table's or an open video encoder's. Its drop lets go of
/// the session on the decoder's video worker (`reims-vgpu.graphics-video-<i>`), under the
/// session's id, so one session's holds are let go of in the order they drop and different
/// sessions' beside each other, and the last hold's drop, which ends the session, runs on neither
/// ring 0's threads nor a queue's walk, whichever lets go last: the session's `OBJECT_DESTROY` or
/// the end of the walk of a command buffer that still had it open. A session's end waits for its last decode's signal and lets go of its
/// pictures' memory and the hardware's session: the device's own memory, none of it the guest's.
/// Measured on the RTX 5080 Laptop GPU host: in a guest's boot, a session whose pictures each had
/// a bitstream buffer of their own ended in 7.0 to 10.1 ms, its last signal landed or 1.6 ms from
/// landing; the video crate's session alone, outside a guest, ends in 1.1 ms at 1080p and 1.6 to
/// 4.2 ms at 4K where its pictures are read in place, and in 2.1 to 4.2 ms where they are copied
/// into bitstream buffers.
pub(crate) struct SessionHold<P: Platform> {
    /// The session's object id, which a refused release is logged under.
    id: u64,
    /// The session, which the drop hands to the worker.
    session: ManuallyDrop<Arc<dyn GraphicsVideoSession<P>>>,
    worker: Arc<Worker>,
}

impl<P: Platform> SessionHold<P> {
    /// A hold of `session`, object `id`, let go of on `worker`.
    pub(super) fn new(
        id: u64,
        session: Arc<dyn GraphicsVideoSession<P>>,
        worker: &Arc<Worker>,
    ) -> Self {
        SessionHold {
            id,
            session: ManuallyDrop::new(session),
            worker: Arc::clone(worker),
        }
    }

    /// The session.
    pub(super) fn session(&self) -> &Arc<dyn GraphicsVideoSession<P>> {
        &self.session
    }

    /// Another hold of the same session.
    pub(super) fn another(&self) -> Self {
        Self::new(self.id, Arc::clone(&self.session), &self.worker)
    }
}

impl<P: Platform> Drop for SessionHold<P> {
    fn drop(&mut self) {
        // SAFETY: the drop runs once, and nothing reads the session after it.
        let session = unsafe { ManuallyDrop::take(&mut self.session) };
        self.worker.drop_later(self.id, self.id, session);
    }
}

/// A video session: each picture's work, started in order, and the event it signals.
pub trait GraphicsVideoSession<P: Platform>: Send + Sync {
    /// The event the session's work signals, which command buffers wait for: plain or shared.
    fn event(&self) -> EventRef<'_, P>;

    /// The codec of its access units (`RGFX_VIDEO_CODEC_*`).
    fn codec(&self) -> u32;

    /// Starts the decode of `decode`'s access unit, after the session's earlier work, and answers
    /// the value [`Self::event`] reaches once the output holds the picture, or, for a picture the
    /// stream's random access skips, once the session's earlier work is done. An `Err` faults the
    /// command buffer, and starts nothing.
    fn decode(&self, decode: &VideoDecode<'_, P>) -> RaiseResult<u64>;
}

/// Where a texture over a plane of an IOSurface lies in its buffer, as its
/// `TEXTURE_FROM_IOSURFACE` placed it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlanePlace {
    /// The buffer the texture lies in.
    pub(crate) buffer_obj_id: u64,
    /// The texture's first byte in the buffer.
    pub(crate) offset: u64,
    /// Bytes from one of its rows to the next.
    pub(crate) bytes_per_row: u64,
    /// The bytes of one of its rows ([`row_bytes`]): its width in its format's texels, or in its
    /// blocks for a format of blocks, at most `bytes_per_row`, which it is where the format's size
    /// is unknown.
    pub(crate) row_bytes: u64,
    /// Its width in pixels.
    pub(crate) width: u64,
    /// Its height in pixels.
    pub(crate) height: u64,
}

/// `MTLPixelFormatInfo`'s flags bit of a format of blocks (the compressed formats, `GBGR422` and
/// `BGRG422`), whose words from +0x18 are its block's bytes, width, height and depth. Any other
/// format's are its texel's bytes, a packed word and two more (R8Unorm `[1, 257, 1, 1]`,
/// RG16Unorm `[4, 514, 4, 4]`, metal2vulkan's table of Metal's answers on the M3 Max).
const BLOCKS: u64 = 0x400;

/// The bytes of a row `width` pixels wide of a texture of the format `info` describes: its texels
/// across, or its blocks across for a format of blocks ([`BLOCKS`]); `bytes_per_row` where the
/// format's size is unknown (no size, or a block of no width), and at most that.
pub(crate) fn row_bytes(info: &PixelFormatInfo, width: u32, bytes_per_row: u64) -> u64 {
    let across = match (info.flags & BLOCKS != 0, info.block_width) {
        (true, 0) => return bytes_per_row,
        (true, block_width) => u64::from(width).div_ceil(block_width),
        (false, _) => u64::from(width),
    };
    if info.block_bytes == 0 {
        return bytes_per_row;
    }
    across.saturating_mul(info.block_bytes).min(bytes_per_row)
}

/// One plane of a picture's output: `height` rows of `width` pixels, `row_bytes` each,
/// `bytes_per_row` apart, from `offset` in `buffer`: its last byte is the last row's last, which
/// may lie less than `bytes_per_row` from the buffer's end (a texture at a place inside a plane
/// whose rows end where the buffer does).
pub struct VideoPlane<'a, P: Platform> {
    /// The buffer the plane lies in.
    pub buffer: &'a P::Buffer,
    /// The plane's first byte in the buffer.
    pub offset: u64,
    /// Bytes from one of its rows to the next.
    pub bytes_per_row: u64,
    /// The bytes of one of its rows.
    pub row_bytes: u64,
    /// Its width in pixels.
    pub width: u64,
    /// Its height in pixels.
    pub height: u64,
    /// Whether the buffer is the surface's device cache, in the GPU's own memory
    /// ([`RGFX_FEATURE_SURFACE_CACHE`](reims_graphics::wire::RGFX_FEATURE_SURFACE_CACHE)), rather
    /// than a buffer over its pages.
    pub cached: bool,
}

/// One access unit to decode, of the session's codec: H.264's ([`RGFX_OP_VIDEO_DECODE_H264`]) or
/// HEVC's ([`RGFX_OP_VIDEO_DECODE_H265`]).
pub struct VideoDecode<'a, P: Platform> {
    /// The buffer that holds the access unit's slices, in Annex-B form.
    pub bitstream: &'a P::Buffer,
    /// The access unit's first byte in the buffer.
    pub offset: u64,
    /// The access unit's bytes.
    pub size: u64,
    /// The parameter sets that changed since the session's last decode (H.264's SPS and PPS,
    /// HEVC's VPS, SPS and PPS), NAL units in Annex-B form; empty for none.
    pub parameter_sets: &'a [u8],
    /// The output's luma plane (`R8`, or `R16` at 10 bits).
    pub luma: VideoPlane<'a, P>,
    /// The output's chroma plane (`RG8`, a pair of samples a pixel, or `RG16`).
    pub chroma: VideoPlane<'a, P>,
    /// The luma rows the output surface holds from the luma plane's first, `bytes_per_row`
    /// apart, and half as many chroma rows from the chroma plane's (the op's `output_rows`): 0
    /// where the planes are the picture's crop alone.
    pub output_rows: u32,
}

/// The fields the decode ops' bodies share ([`RgfxVideoOpDecodeH264`],
/// [`RgfxVideoOpDecodeH265`]), and the session codec each op is of.
struct DecodeOp {
    codec: u32,
    bitstream_buffer_obj_id: u64,
    bitstream_offset: u64,
    bitstream_size: u64,
    luma_texture_obj_id: u64,
    chroma_texture_obj_id: u64,
    parameter_sets_size: u32,
    output_rows: u32,
}

impl DecodeOp {
    /// The op `tag`'s body at the front of `payload`, and the bytes after it, its parameter sets
    /// first (each body's `*_PARAMETER_SETS_OFFSET` is its size); `None` for a tag that is no
    /// decode.
    fn read(tag: u16, payload: &[u8]) -> Option<RaiseResult<(Self, &[u8])>> {
        macro_rules! body {
            ($body:ty, $codec:expr) => {
                <$body>::read_from_prefix(payload)
                    .map(|(m, rest)| {
                        let op = Self {
                            codec: $codec,
                            bitstream_buffer_obj_id: m.bitstream_buffer_obj_id,
                            bitstream_offset: m.bitstream_offset,
                            bitstream_size: m.bitstream_size,
                            luma_texture_obj_id: m.luma_texture_obj_id,
                            chroma_texture_obj_id: m.chroma_texture_obj_id,
                            parameter_sets_size: m.parameter_sets_size,
                            output_rows: m.output_rows,
                        };
                        (op, rest)
                    })
                    .map_err(|_| op_fault(format_args!("video op {tag:#06x}: body too short")))
            };
        }
        Some(match tag {
            RGFX_OP_VIDEO_DECODE_H264 => body!(RgfxVideoOpDecodeH264, RGFX_VIDEO_CODEC_H264),
            RGFX_OP_VIDEO_DECODE_H265 => body!(RgfxVideoOpDecodeH265, RGFX_VIDEO_CODEC_HEVC),
            _ => return None,
        })
    }
}

/// Decodes one video-encoder op: `tag` its op, `payload` its body (the bytes after its
/// [`RgfxOpHeader`](reims_graphics::ops::RgfxOpHeader)), on `session`, and has `cb`'s later work
/// wait for it. `Ok(false)` for a tag the host does not know.
pub(super) fn decode_op<P: Platform>(
    tag: u16,
    payload: &[u8],
    objs: &GraphicsDecoder<P>,
    session: &Arc<dyn GraphicsVideoSession<P>>,
    cb: &P::CommandBuffer,
) -> RaiseResult<bool> {
    let Some(read) = DecodeOp::read(tag, payload) else {
        return Ok(false);
    };
    let (m, rest) = read?;
    if m.codec != session.codec() {
        return Err(op_fault(format_args!(
            "video op {tag:#06x}: a decode of codec {} on a session of codec {}",
            m.codec,
            session.codec()
        )));
    }
    let sets = usize::try_from(m.parameter_sets_size)
        .ok()
        .and_then(|n| rest.get(..n))
        .ok_or_else(|| {
            op_fault(format_args!(
                "video op {tag:#06x}: {} bytes of parameter sets past its end",
                m.parameter_sets_size
            ))
        })?;
    let bitstream = objs.require_buffer(m.bitstream_buffer_obj_id)?;
    let in_bitstream = m
        .bitstream_offset
        .checked_add(m.bitstream_size)
        .is_some_and(|end| m.bitstream_size > 0 && end <= bitstream.length());
    if !in_bitstream {
        return Err(op_fault(format_args!(
            "video op {tag:#06x}: an access unit of {} bytes at {} past its buffer",
            m.bitstream_size, m.bitstream_offset
        )));
    }
    let (luma_place, luma) = objs.plane_of(m.luma_texture_obj_id)?;
    let (chroma_place, chroma) = objs.plane_of(m.chroma_texture_obj_id)?;
    // The block of a plane in its surface's device cache.
    let cache_of = |buffer_obj_id| {
        (objs.span_of(buffer_obj_id))
            .map(|(block, _)| block)
            .filter(|&block| objs.surface_settled(block) && objs.surface_caches.has(block))
    };
    let (luma_cache, chroma_cache) = (
        cache_of(luma_place.buffer_obj_id),
        cache_of(chroma_place.buffer_obj_id),
    );
    // An output in a device cache the host is filling for this buffer (a CPU wrote the
    // surface since): the picture lands once the fill is done, so the fill's older bytes
    // never land over it.
    // So too an output whose last picture the host is still copying into the surface's pages
    // (an early copy, its surface read back): the picture lands once the copy is done, so the
    // copy never reads it.
    for block in luma_cache.into_iter().chain(chroma_cache) {
        if !objs.surface_caches.wait_filled(block, FILL_WAIT) {
            return Err(op_fault(format_args!(
                "video op {tag:#06x}: its output's device cache was not filled within \
                 {FILL_WAIT:?}"
            )));
        }
        if !objs.surface_caches.wait_copied_out(block, FILL_WAIT) {
            return Err(op_fault(format_args!(
                "video op {tag:#06x}: its output's last picture was not copied into its pages \
                 within {FILL_WAIT:?}"
            )));
        }
    }
    let decode = VideoDecode {
        bitstream,
        offset: m.bitstream_offset,
        size: m.bitstream_size,
        parameter_sets: sets,
        luma: VideoPlane {
            buffer: luma,
            offset: luma_place.offset,
            bytes_per_row: luma_place.bytes_per_row,
            row_bytes: luma_place.row_bytes,
            width: luma_place.width,
            height: luma_place.height,
            cached: luma_cache.is_some(),
        },
        chroma: VideoPlane {
            buffer: chroma,
            offset: chroma_place.offset,
            bytes_per_row: chroma_place.bytes_per_row,
            row_bytes: chroma_place.row_bytes,
            width: chroma_place.width,
            height: chroma_place.height,
            cached: chroma_cache.is_some(),
        },
        output_rows: m.output_rows,
    };
    // After the buffers of other queues its command buffer runs after: the reads of the
    // output it would land over among them.
    objs.wait_for_order_on_cpu();
    let done = session.decode(&decode)?;
    cb.encode_wait(Some(session.event()), done)?;
    Ok(true)
}

impl<P: Platform> GraphicsDecoder<P> {
    /// The host's hardware video, which `VIDEO_SESSION_CREATE` makes its sessions on.
    pub(crate) fn set_video(&mut self, video: Arc<dyn GraphicsVideo<P>>) {
        // Set once, at the device's make.
        let _ = self.shared.video.set(video);
    }

    /// A hold of the session `id` names; an exec fault where it names none, and a refused
    /// reference where its create was refused ([`super::refused`]).
    pub(super) fn require_video_session(&self, id: u64) -> RaiseResult<SessionHold<P>> {
        match self.named(id) {
            Some(ObjHandle::VideoSession(session)) => Ok(session.another()),
            Some(ObjHandle::Refused(r)) => Err(r.reference(id)),
            _ => Err(bad_reference()),
        }
    }

    /// Where texture `id`, one over an IOSurface plane, lies, and the buffer it lies in; an exec
    /// fault where `id` names no such texture, and a refused reference where its create was
    /// refused ([`super::refused`]).
    fn plane_of(&self, id: u64) -> RaiseResult<(PlanePlace, &P::Buffer)> {
        let place = match self.entry(id).map(|e| (&e.handle, e.place)) {
            Some((ObjHandle::Texture(_), place)) => place,
            Some((ObjHandle::Refused(r), _)) => return Err(r.reference(id)),
            _ => None,
        }
        .ok_or_else(bad_reference)?;
        Ok((place, self.require_buffer(place.buffer_obj_id)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Metal's answer for a format, its flags and its four words from +0x18, as metal2vulkan's
    /// table of the M3 Max's answers holds them.
    fn info(
        flags: u64,
        [block_bytes, block_width, block_height, block_depth]: [u64; 4],
    ) -> PixelFormatInfo {
        PixelFormatInfo {
            name: None,
            flags,
            cast_class: 0,
            block_bytes,
            block_width,
            block_height,
            block_depth,
        }
    }

    /// R8Unorm, R16Unorm, RG8Unorm and RG16Unorm: a video plane's formats.
    const R8: (u64, [u64; 4]) = (0x0703_019f, [1, 257, 1, 1]);
    const R16: (u64, [u64; 4]) = (0x0703_019f, [2, 513, 2, 2]);
    const RG8: (u64, [u64; 4]) = (0x0503_019f, [2, 258, 2, 2]);
    const RG16: (u64, [u64; 4]) = (0x0703_019f, [4, 514, 4, 4]);
    /// GBGR422, a format of 2-by-1 blocks; BC1_RGBA, of 4-by-4 blocks of 8 bytes (blocks' flag
    /// alone, which is what the row reads).
    const GBGR422: (u64, [u64; 4]) = (0x0003_4405, [4, 2, 1, 1]);
    const BC1: (u64, [u64; 4]) = (BLOCKS, [8, 4, 4, 1]);

    /// A plain format's row is its texels across, whatever its second word (R8Unorm's 257, which a
    /// read of it as a block width made 8 bytes of a 1920-pixel luma row): the luma and chroma of
    /// 8-bit and 10-bit pictures at 1080p and 4K, and at most the pitch.
    #[test]
    fn a_plain_formats_row_is_its_texels_across() {
        let row = |(flags, words), width, pitch| row_bytes(&info(flags, words), width, pitch);
        assert_eq!(row(R8, 1920, 2048), 1920);
        assert_eq!(row(RG8, 960, 2048), 1920);
        assert_eq!(row(R16, 3840, 8192), 7680);
        assert_eq!(row(RG16, 1920, 8192), 7680);
        assert_eq!(row(R8, 3840, 2048), 2048);
    }

    /// A format of blocks has its whole blocks across, at most its pitch, and its pitch where its
    /// size is unknown: a 2-by-1 format, a compressed one of 4-pixel blocks over a partial block,
    /// a block of no width, and a format of no size.
    #[test]
    fn a_block_formats_row_is_its_blocks_across() {
        let row = |(flags, words), width, pitch| row_bytes(&info(flags, words), width, pitch);
        assert_eq!(row(GBGR422, 511, 2048), 1024);
        assert_eq!(row(BC1, 6, 2048), 16);
        assert_eq!(row(BC1, 1024, 2048), 2048);
        assert_eq!(row((BLOCKS, [8, 0, 4, 1]), 6, 2048), 2048);
        assert_eq!(row((0, [0; 4]), 512, 2048), 2048);
    }
}
