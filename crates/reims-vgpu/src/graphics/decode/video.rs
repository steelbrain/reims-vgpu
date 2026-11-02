//! The video encoders of the guest's command buffers (`tag >> 12 ==`
//! [`RGFX_OP_GROUP_VIDEO`](reims_graphics::ops::RGFX_OP_GROUP_VIDEO)): the hardware video work of
//! the guest's video driver, a picture a command buffer, on a session
//! ([`RGFX_CMD_VIDEO_SESSION_CREATE`](reims_graphics::wire::RGFX_CMD_VIDEO_SESSION_CREATE)) the
//! platform's [`GraphicsVideo`] makes. The decoder resolves each op's objects (the bitstream's
//! buffer, the buffers the output surface's planes lie in) and hands the work to the session,
//! which starts it at once, as the host takes the committed buffer, after the session's earlier
//! work; the buffer's later work, and its completion, wait for the session's shared event to
//! reach the value the work signals once done. The work does not wait for the buffer's earlier
//! encoders: the guest's video driver encodes none before it.

use std::sync::Arc;

use reims_graphics::ops::video::{
    RGFX_OP_VIDEO_DECODE_H264, RGFX_VIDEO_DECODE_H264_PARAMETER_SETS_OFFSET, RgfxVideoOpDecodeH264,
};
use reims_graphics::wire::RgfxVideoCodecCaps;
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::command::CommandBuffer as _;
use reims_vgpu_core::platform::event::EventRef;
use reims_vgpu_core::platform::resource::Buffer as _;
use reims_vgpu_core::raise::RaiseResult;
use zerocopy::FromBytes;

use super::{GraphicsDecoder, ObjHandle, bad_reference, op_fault};

/// The host's hardware video for a graphics device: the codecs it takes, which the device's video
/// caps answer, and its sessions.
pub trait GraphicsVideo<P: Platform>: Send + Sync {
    /// One answer for each codec and direction the hardware takes.
    fn caps(&self) -> Vec<RgfxVideoCodecCaps>;
    /// A session of `codec` (`RGFX_VIDEO_CODEC_*`) in `direction` (`RGFX_VIDEO_DECODE` or
    /// `RGFX_VIDEO_ENCODE`); `None` where the hardware has no such codec.
    fn session(&self, codec: u32, direction: u32) -> Option<Arc<dyn GraphicsVideoSession<P>>>;
}

/// A video session: each picture's work, started in order, and the shared event it signals.
pub trait GraphicsVideoSession<P: Platform>: Send + Sync {
    /// The shared event the session's work signals, which command buffers wait for.
    fn event(&self) -> &P::SharedEvent;

    /// Starts the decode of `decode`'s access unit, after the session's earlier work, and answers
    /// the value [`Self::event`] reaches once the output holds the picture. An `Err` faults the
    /// command buffer, and starts nothing.
    fn decode_h264(&self, decode: &H264Decode<'_, P>) -> RaiseResult<u64>;
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
}

/// One H.264 access unit to decode ([`RGFX_OP_VIDEO_DECODE_H264`]).
pub struct H264Decode<'a, P: Platform> {
    /// The buffer that holds the access unit's slices, in Annex-B form.
    pub bitstream: &'a P::Buffer,
    /// The access unit's first byte in the buffer.
    pub offset: u64,
    /// The access unit's bytes.
    pub size: u64,
    /// The SPS and PPS NAL units that changed since the session's last decode, in Annex-B form;
    /// empty for none.
    pub parameter_sets: &'a [u8],
    /// The output's luma plane (`R8`).
    pub luma: VideoPlane<'a, P>,
    /// The output's chroma plane (`RG8`, a pair of bytes a pixel).
    pub chroma: VideoPlane<'a, P>,
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
    match tag {
        RGFX_OP_VIDEO_DECODE_H264 => {
            let (m, rest) = RgfxVideoOpDecodeH264::read_from_prefix(payload)
                .map_err(|_| op_fault(format_args!("video op {tag:#06x}: body too short")))?;
            let sets = usize::try_from(m.parameter_sets_size)
                .ok()
                .and_then(|n| rest.get(..n))
                .ok_or_else(|| {
                    op_fault(format_args!(
                        "video op {tag:#06x}: {} bytes of parameter sets past its end",
                        m.parameter_sets_size
                    ))
                })?;
            debug_assert_eq!(
                payload.len() - rest.len(),
                RGFX_VIDEO_DECODE_H264_PARAMETER_SETS_OFFSET as usize
            );
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
            let decode = H264Decode {
                bitstream: &bitstream,
                offset: m.bitstream_offset,
                size: m.bitstream_size,
                parameter_sets: sets,
                luma: VideoPlane {
                    buffer: &luma,
                    offset: luma_place.offset,
                    bytes_per_row: luma_place.bytes_per_row,
                    width: luma_place.width,
                    height: luma_place.height,
                },
                chroma: VideoPlane {
                    buffer: &chroma,
                    offset: chroma_place.offset,
                    bytes_per_row: chroma_place.bytes_per_row,
                    width: chroma_place.width,
                    height: chroma_place.height,
                },
            };
            let done = session.decode_h264(&decode)?;
            cb.encode_wait(Some(EventRef::Shared(session.event())), done)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

impl<P: Platform> GraphicsDecoder<P> {
    /// The host's hardware video, which `VIDEO_SESSION_CREATE` makes its sessions on.
    pub(crate) fn set_video(&mut self, video: Arc<dyn GraphicsVideo<P>>) {
        self.video = Some(video);
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
    fn plane_of(&self, id: u64) -> RaiseResult<(PlanePlace, P::Buffer)> {
        let place = *self.surface_places.get(&id).ok_or_else(bad_reference)?;
        if !matches!(self.named(id), Some(ObjHandle::Texture(_))) {
            return Err(bad_reference());
        }
        Ok((place, self.require_buffer(place.buffer_obj_id)?))
    }
}
