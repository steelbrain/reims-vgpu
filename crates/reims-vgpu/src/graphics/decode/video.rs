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

use std::sync::Arc;

use reims_graphics::ops::video::{
    RGFX_OP_VIDEO_DECODE_H264, RGFX_OP_VIDEO_DECODE_H265, RgfxVideoOpDecodeH264,
    RgfxVideoOpDecodeH265,
};
use reims_graphics::wire::{RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_CODEC_HEVC, RgfxVideoCodecCaps};
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::command::CommandBuffer as _;
use reims_vgpu_core::platform::event::EventRef;
use reims_vgpu_core::platform::resource::Buffer as _;
use reims_vgpu_core::raise::RaiseResult;
use zerocopy::FromBytes;

use super::{GraphicsDecoder, ObjHandle, bad_reference, op_fault};

/// How long a decode waits for its output's device cache to be filled: a copy of the surface's
/// pages, a few milliseconds even across a slow bus.
const FILL_WAIT: std::time::Duration = std::time::Duration::from_secs(2);

/// The host's hardware video for a graphics device: the codecs it takes, which the device's video
/// caps answer, and its sessions.
pub trait GraphicsVideo<P: Platform>: Send + Sync {
    /// One answer for each codec and direction the hardware takes.
    fn caps(&self) -> Vec<RgfxVideoCodecCaps>;
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

/// A video session: each picture's work, started in order, and the shared event it signals.
pub trait GraphicsVideoSession<P: Platform>: Send + Sync {
    /// The shared event the session's work signals, which command buffers wait for.
    fn event(&self) -> &P::SharedEvent;

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
    /// Its width in pixels.
    pub(crate) width: u64,
    /// Its height in pixels.
    pub(crate) height: u64,
}

/// One plane of a picture's output: `height` rows of `width` pixels, `bytes_per_row` apart, from
/// `offset` in `buffer`.
pub struct VideoPlane<'a, P: Platform> {
    /// The buffer the plane lies in.
    pub buffer: &'a P::Buffer,
    /// The plane's first byte in the buffer.
    pub offset: u64,
    /// Bytes from one of its rows to the next.
    pub bytes_per_row: u64,
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
    for block in luma_cache.into_iter().chain(chroma_cache) {
        if !objs.surface_caches.wait_filled(block, FILL_WAIT) {
            return Err(op_fault(format_args!(
                "video op {tag:#06x}: its output's device cache was not filled within \
                 {FILL_WAIT:?}"
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
            width: luma_place.width,
            height: luma_place.height,
            cached: luma_cache.is_some(),
        },
        chroma: VideoPlane {
            buffer: chroma,
            offset: chroma_place.offset,
            bytes_per_row: chroma_place.bytes_per_row,
            width: chroma_place.width,
            height: chroma_place.height,
            cached: chroma_cache.is_some(),
        },
    };
    // After the buffers of other queues its command buffer runs after: the reads of the
    // output it would land over among them.
    objs.wait_for_order_on_cpu();
    let done = session.decode(&decode)?;
    cb.encode_wait(Some(EventRef::Shared(session.event())), done)?;
    Ok(true)
}

impl<P: Platform> GraphicsDecoder<P> {
    /// The host's hardware video, which `VIDEO_SESSION_CREATE` makes its sessions on.
    pub(crate) fn set_video(&mut self, video: Arc<dyn GraphicsVideo<P>>) {
        // Set once, at the device's make.
        let _ = self.shared.video.set(video);
    }

    /// The session `id` names; an exec fault where it names none.
    pub(super) fn require_video_session(
        &self,
        id: u64,
    ) -> RaiseResult<Arc<dyn GraphicsVideoSession<P>>> {
        match self.named(id) {
            Some(ObjHandle::VideoSession(session)) => Ok(Arc::clone(session)),
            _ => Err(bad_reference()),
        }
    }

    /// Where texture `id`, one over an IOSurface plane, lies, and the buffer it lies in; an exec
    /// fault where `id` names no such texture.
    fn plane_of(&self, id: u64) -> RaiseResult<(PlanePlace, &P::Buffer)> {
        let place = match self.entry(id) {
            Some(e) if matches!(e.handle, ObjHandle::Texture(_)) => e.place,
            _ => None,
        }
        .ok_or_else(bad_reference)?;
        Ok((place, self.require_buffer(place.buffer_obj_id)?))
    }
}
