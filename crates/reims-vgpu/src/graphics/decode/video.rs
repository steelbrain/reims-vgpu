//! The video encoders of the guest's command buffers (`tag >> 12 ==`
//! [`RGFX_OP_GROUP_VIDEO`](reims_vgpu_graphics::ops::RGFX_OP_GROUP_VIDEO)): the hardware video work of
//! the guest's video driver, a picture a command buffer, on a session
//! ([`RGFX_CMD_VIDEO_SESSION_CREATE`](reims_vgpu_graphics::wire::RGFX_CMD_VIDEO_SESSION_CREATE)) the
//! platform's [`GraphicsVideo`] makes. The decoder resolves each op's objects (the bitstream's
//! buffer, the buffers the output surface's planes lie in) and hands the work to the session,
//! which starts it at once, as the host takes the committed buffer, after the session's earlier
//! work; the buffer's later work, and its completion, wait for the session's shared event to
//! reach the value the work signals once done. The work does not wait for the buffer's earlier
//! encoders: the guest's video driver encodes none before it. It does wait for the buffers of
//! other queues the command buffer runs after (its `SURFACE_ORDER`), on the CPU, as the session's
//! queue is not the command buffer's ([`super::surface_order`]): a decode whose output another
//! process still reads lands once the read is done.
//!
//! A picture the session fails to land after its decode was taken (its planes refused, or a
//! decode the hardware failed once it ran) still signals its value, so the buffer completes; the
//! buffer completes failed, Metal's internal error ([`Landed`]), which the guest's video driver
//! answers as a decode error: an HEVC picture's schedule, waiting for the buffer, fails
//! (`kVTVideoDecoderMalfunctionErr`, docs/graphics-kext/design.md, "A decode the driver cannot
//! complete is a decode error").

use std::mem::ManuallyDrop;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use reims_vgpu_graphics::ops::video::{
    RGFX_OP_VIDEO_DECODE_H264, RGFX_OP_VIDEO_DECODE_H265, RGFX_OP_VIDEO_ENCODE_H264,
    RGFX_OP_VIDEO_ENCODE_H265, RGFX_VIDEO_ENCODE_FORCE_KEYFRAME, RgfxVideoEncodeResult,
    RgfxVideoEncodeSettings, RgfxVideoOpDecodeH264, RgfxVideoOpDecodeH265, RgfxVideoOpEncode,
};
use reims_vgpu_graphics::wire::{
    RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_CODEC_HEVC, RGFX_VIDEO_DECODE, RgfxSurfaceFacts,
    RgfxVideoCodecCaps, RgfxVideoDecodeCaps, RgfxVideoEncodeCaps, RgfxVideoOutputLayout,
};
use reims_vgpu_video::core::device_memory::{DeviceMemory, HostView, MemoryIdentity};
#[cfg(target_os = "linux")]
use reims_vgpu_video::core::device_memory::{DmaBufError, SurfaceDmaBuf};
use reims_vgpu_video::core::surfaces::SurfaceError;
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::command::CommandBuffer as _;
use reims_vgpu_vulkan::core::platform::event::EventRef;
use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
use reims_vgpu_vulkan::core::platform::types::PixelFormatInfo;
use reims_vgpu_vulkan::core::platform::values::PagesPlane;
use reims_vgpu_vulkan::core::raise::RaiseResult;
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
    /// What an encode of `codec`, one of [`Self::caps`] in the encode direction, takes beyond
    /// it; all zero for a decode entry, which is what a backend that encodes nothing answers.
    fn encode_caps(&self, _codec: &RgfxVideoCodecCaps) -> RgfxVideoEncodeCaps {
        RgfxVideoEncodeCaps::default()
    }
    /// What a decode of `codec`, one of [`Self::caps`] in the decode direction, refuses beyond
    /// it, for the guest's video driver to refuse first: the bounds of the video crate's codec
    /// decoders, which both arms decode through, HEVC's narrowest tile column
    /// ([`MIN_TILE_COLUMN_WIDTH`](reims_vgpu_video::codec::hevc::MIN_TILE_COLUMN_WIDTH)); all
    /// zero for an encode entry.
    fn decode_caps(&self, codec: &RgfxVideoCodecCaps) -> RgfxVideoDecodeCaps {
        codec_decode_caps(codec)
    }
    /// A session of `codec` (`RGFX_VIDEO_CODEC_*`) in `direction` (`RGFX_VIDEO_DECODE` or
    /// `RGFX_VIDEO_ENCODE`); `None` where the hardware has no such codec.
    fn session(&self, codec: u32, direction: u32) -> Option<Arc<dyn GraphicsVideoSession<P>>>;
    /// Whether a decode lands its picture in an output surface's device cache, where the
    /// surface's block has one, as well as in its pages
    /// ([`RGFX_VIDEO_CAPS_OUTPUT_CACHED`](reims_vgpu_graphics::wire::RGFX_VIDEO_CAPS_OUTPUT_CACHED)).
    fn lands_in_device_caches(&self) -> bool {
        false
    }
    /// Whether it writes a picture into a device cache by its own GPU work, through an import of
    /// the cache's memory, where the cache was made exportable: the device's caches are then made
    /// so ([`VideoPlane::cache`]).
    fn imports_device_caches(&self) -> bool {
        false
    }
    /// Whether its hardware decodes a picture straight into a device cache, through an import of
    /// the cache's memory as a dma-buf, where the cache was made so: the device's laid-out caches
    /// of two planes, the decodes' output, are then made with their memory a dma-buf
    /// ([`VideoPlane::cache`]).
    fn decodes_into_device_caches(&self) -> bool {
        false
    }
}

/// What the video crate's codec decoders refuse of `codec` beyond its entry
/// ([`GraphicsVideo::decode_caps`]): HEVC's tile columns under
/// [`MIN_TILE_COLUMN_WIDTH`](reims_vgpu_video::codec::hevc::MIN_TILE_COLUMN_WIDTH); nothing for
/// H.264 or an encode.
pub(crate) fn codec_decode_caps(codec: &RgfxVideoCodecCaps) -> RgfxVideoDecodeCaps {
    let hevc = codec.codec == RGFX_VIDEO_CODEC_HEVC && codec.direction == RGFX_VIDEO_DECODE;
    RgfxVideoDecodeCaps {
        min_tile_width: if hevc {
            reims_vgpu_video::codec::hevc::MIN_TILE_COLUMN_WIDTH
        } else {
            0
        },
        ..RgfxVideoDecodeCaps::default()
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
    /// stream's random access skips, once the session's earlier work is done, and where the
    /// picture may still fail to land after it, what it came to ([`Decoded`]). An `Err` faults the
    /// command buffer, and starts nothing.
    fn decode(&self, decode: &VideoDecode<'_, P>) -> RaiseResult<Decoded>;

    /// Starts the encode of `encode`'s frame, after the session's earlier work, and answers the
    /// value [`Self::event`] reaches once the frame's access unit and its result are in the
    /// guest's buffers, whatever the result says. An `Err` faults the command buffer, and starts
    /// nothing; a session that does not encode faults every encode.
    fn encode(&self, _encode: &VideoEncode<'_, P>) -> RaiseResult<u64> {
        Err(op_fault("a video encode on a session that does not encode"))
    }
}

/// A decode the session took ([`GraphicsVideoSession::decode`]).
pub struct Decoded {
    /// The value the session's event reaches once it is done.
    pub value: u64,
    /// The stream's random access skips its picture, and the output gets nothing.
    pub skipped: bool,
    /// Whether its picture landed, known once the event reaches `value`; `None` where the session
    /// lands every picture it takes, or faults the decode.
    pub landed: Option<Arc<Landed>>,
}

/// Whether a decode's picture landed: set by its session before it signals the decode's value,
/// and read by the command buffer waiting for it as it completes, which completes failed, Metal's
/// internal error, where the picture did not land (the module docs).
#[derive(Debug, Default)]
pub struct Landed(AtomicBool);

impl Landed {
    /// The picture did not land.
    pub fn fail(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    /// Whether the picture did not land.
    pub fn failed(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
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
    /// The plane of its surface it shows whole, where it does ([`whole_plane`]).
    pub(crate) whole: Option<u32>,
}

/// The plane `plane` of a surface whose facts are `facts` that a texture `width` by `height`
/// pixels over it shows whole: plane 0 or 1 of a surface of two planes whose second is half the
/// first's size each way, rounded up (a 4:2:0 picture's luma and chroma), the texture of its
/// plane's size; `None` for any other texture, and where the texture's message gave no facts.
pub(crate) fn whole_plane(
    facts: Option<&RgfxSurfaceFacts>,
    plane: u32,
    width: u32,
    height: u32,
) -> Option<u32> {
    let f = facts.filter(|f| f.plane_count == 2)?;
    let size = match plane {
        0 => (f.width, f.height),
        1 => (f.width.div_ceil(2), f.height.div_ceil(2)),
        _ => return None,
    };
    (size == (width, height)).then_some(plane)
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
    /// ([`RGFX_FEATURE_SURFACE_CACHE`](reims_vgpu_graphics::wire::RGFX_FEATURE_SURFACE_CACHE)), rather
    /// than a buffer over its pages.
    pub cached: bool,
    /// The memory the buffer lies in, as a device of the host's video reaches it in place: the
    /// surface's view of its pages, or its block of the region (a copy block); `None` for a device
    /// cache, and for a buffer over memory of another kind.
    pub pages: Option<PlanePages>,
    /// The identity of the device cache the buffer is, where `cached` and the cache's memory was
    /// made for the host's video (exportable, or a dma-buf), which lives as long as the cache:
    /// what a device of the host's video makes of the cache's memory (an import of it) is kept by
    /// it and let go of once it ends. `None` for a cache made as any other buffer, which no
    /// device of the host's video reaches in place.
    pub cache: Option<MemoryIdentity>,
}

/// The memory a plane's buffer lies in, as the host's video devices take it
/// ([`VideoPlane::pages`]): one mapping of it in the host's address space, whose identity lives
/// as long as the mapping is the surface's (its view, or its block until its `BLOCK_RELEASE`),
/// and whose host view is held mapped for as long as the device keeps it. What a device makes of
/// it (a CUDA registration, a dma-buf's import) is kept by the identity and let go of once it
/// ends.
#[derive(Clone)]
pub struct PlanePages {
    /// The mapping.
    pub memory: Arc<dyn DeviceMemory>,
    /// Where the buffer's first byte lies in it, from its first byte.
    pub buffer_offset: u64,
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

/// One frame to encode, of the session's codec: H.264's ([`RGFX_OP_VIDEO_ENCODE_H264`]) or
/// HEVC's ([`RGFX_OP_VIDEO_ENCODE_H265`]).
pub struct VideoEncode<'a, P: Platform> {
    /// The source's luma plane (`R8`, or `R16` at 10 bits).
    pub luma: VideoPlane<'a, P>,
    /// The source's chroma plane (`RG8`, or `RG16`).
    pub chroma: VideoPlane<'a, P>,
    /// The Shared buffer the access unit goes in.
    pub bitstream: &'a P::Buffer,
    /// Where its first byte goes in it.
    pub bitstream_offset: u64,
    /// The most bytes it may take there: at least one, inside the buffer.
    pub bitstream_capacity: u64,
    /// The Shared buffer the [`RgfxVideoEncodeResult`] goes in.
    pub result: &'a P::Buffer,
    /// Where it goes in it: a multiple of 8, the whole result inside the buffer.
    pub result_offset: u64,
    /// The frame's place in display order.
    pub frame_index: u64,
    /// Whether the frame must be a keyframe.
    pub force_keyframe: bool,
    /// The settings from this frame on, where they changed since the session's last encode.
    pub settings: Option<RgfxVideoEncodeSettings>,
}

/// A block of the region as the host's video devices take memory ([`PlanePages`]): its bytes in
/// the region's mapping, which stays mapped while the region lives, held by the host view, and
/// the block's life, which its `BLOCK_RELEASE` ends ([`GraphicsDecoder::block_released`]).
struct RegionMemory {
    identity: MemoryIdentity,
    /// The block's first byte's host address.
    first: u64,
    /// The block's bytes.
    length: usize,
    /// The host's page.
    page: usize,
    region: Arc<reims_vgpu_graphics::region::Region>,
}

impl RegionMemory {
    /// The block of `size` bytes at `block` in `region`, of life `identity`, in host pages of
    /// `page` bytes; `None` past it.
    fn new(
        region: &Arc<reims_vgpu_graphics::region::Region>,
        block: u64,
        size: u64,
        page: u64,
        identity: MemoryIdentity,
    ) -> Option<Self> {
        if !region.contains(block, size) {
            return None;
        }
        let base = u64::try_from(region.base().addr()).ok()?;
        Some(Self {
            identity,
            first: base.checked_add(block)?,
            length: usize::try_from(size).ok()?,
            page: usize::try_from(page).ok().filter(|&p| p > 0)?,
            region: Arc::clone(region),
        })
    }
}

impl DeviceMemory for RegionMemory {
    fn identity(&self) -> MemoryIdentity {
        self.identity.clone()
    }

    fn length(&self) -> usize {
        self.length
    }

    /// A dma-buf of the block's pages behind `ranges`, where the Vulkan platform's backing table
    /// knows the files behind the region's mapping (`platform/vulkan/lent_memory.rs`); refused
    /// where it does not. The region stays mapped while it lives.
    #[cfg(target_os = "linux")]
    fn dma_buf_of(&self, ranges: &[(usize, usize)]) -> Result<SurfaceDmaBuf, DmaBufError> {
        let first = usize::try_from(self.first)
            .ok()
            .and_then(|at| NonNull::new(std::ptr::with_exposed_provenance_mut::<u8>(at)))
            .ok_or_else(|| DmaBufError::Refused("a block at no address".to_owned()))?;
        crate::platform::host_dma_buf(first, self.length, ranges)
    }

    /// The block's whole host pages, the block's first byte at its offset in them, held mapped
    /// by the region.
    fn host_view(&self) -> Result<HostView, SurfaceError> {
        let page = self.page;
        let at = usize::try_from(self.first)
            .map_err(|_| SurfaceError::Unreachable("a block past the address space".to_owned()))?;
        let offset = at % page;
        let size = (offset + self.length).next_multiple_of(page);
        let first = NonNull::new((at - offset) as *mut u8)
            .ok_or_else(|| SurfaceError::Unreachable("a block at no address".to_owned()))?;
        Ok(HostView {
            first,
            size,
            offset,
            hold: Box::new(Arc::clone(&self.region)),
        })
    }
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

/// The luma and chroma textures a decode op `tag` of body `payload` writes; `None` for a tag
/// that is no decode, or a body too short.
pub(super) fn decode_textures(tag: u16, payload: &[u8]) -> Option<(u64, u64)> {
    let (m, _) = DecodeOp::read(tag, payload)?.ok()?;
    Some((m.luma_texture_obj_id, m.chroma_texture_obj_id))
}

/// The session codec of an encode op `tag`; `None` for a tag that is no encode.
fn encode_codec(tag: u16) -> Option<u32> {
    match tag {
        RGFX_OP_VIDEO_ENCODE_H264 => Some(RGFX_VIDEO_CODEC_H264),
        RGFX_OP_VIDEO_ENCODE_H265 => Some(RGFX_VIDEO_CODEC_HEVC),
        _ => None,
    }
}

/// Decodes one video-encoder op: `tag` its op, `payload` its body (the bytes after its
/// [`RgfxOpHeader`](reims_vgpu_graphics::ops::RgfxOpHeader)), on `session`, and has `cb`'s later work
/// wait for it, and its completion read whether its picture landed, from `landings`. `Ok(false)`
/// for a tag the host does not know.
pub(super) fn decode_op<P: Platform>(
    tag: u16,
    payload: &[u8],
    objs: &GraphicsDecoder<P>,
    session: &Arc<dyn GraphicsVideoSession<P>>,
    cb: &P::CommandBuffer,
    landings: &mut Vec<Arc<Landed>>,
) -> RaiseResult<bool> {
    if let Some(codec) = encode_codec(tag) {
        encode_op(tag, codec, payload, objs, session, cb)?;
        return Ok(true);
    }
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
    let (luma, chroma) = objs.video_planes(tag, m.luma_texture_obj_id, m.chroma_texture_obj_id)?;
    let decode = VideoDecode {
        bitstream,
        offset: m.bitstream_offset,
        size: m.bitstream_size,
        parameter_sets: sets,
        luma,
        chroma,
        output_rows: m.output_rows,
    };
    // After the buffers of other queues its command buffer runs after: the reads of the
    // output it would land over among them.
    objs.wait_for_order_on_cpu();
    let done = session.decode(&decode)?;
    // A decoder's new output whose first fill the buffer's begin took as met holds the cache's
    // zeros: a picture lands over them, and none makes the fill, before anything reads the cache
    // (`surface_cache`, "Fill").
    let device = done.skipped.then(|| cb.device()).flatten();
    for block in objs.decode_blocks(m.luma_texture_obj_id, m.chroma_texture_obj_id) {
        (objs.surface_caches).decoded(device.as_ref(), block, !done.skipped)?;
    }
    cb.encode_wait(Some(session.event()), done.value)?;
    landings.extend(done.landed);
    Ok(true)
}

/// Decodes one encode op, `tag` of `codec`, its body at the front of `payload` and its settings
/// after it where it has them, on `session`, and has `cb`'s later work wait for it.
fn encode_op<P: Platform>(
    tag: u16,
    codec: u32,
    payload: &[u8],
    objs: &GraphicsDecoder<P>,
    session: &Arc<dyn GraphicsVideoSession<P>>,
    cb: &P::CommandBuffer,
) -> RaiseResult<()> {
    let fault =
        |what: std::fmt::Arguments<'_>| op_fault(format_args!("video op {tag:#06x}: {what}"));
    let (m, rest) = RgfxVideoOpEncode::read_from_prefix(payload)
        .map_err(|_| fault(format_args!("body too short")))?;
    if codec != session.codec() {
        return Err(fault(format_args!(
            "an encode of codec {codec} on a session of codec {}",
            session.codec()
        )));
    }
    let settings = match m.settings_size {
        0 => None,
        size => {
            let bytes = usize::try_from(size).ok().and_then(|n| rest.get(..n));
            let settings = bytes.and_then(|b| RgfxVideoEncodeSettings::read_from_prefix(b).ok());
            match settings {
                Some((settings, _)) => Some(settings),
                None => {
                    return Err(fault(format_args!(
                        "settings of {size} bytes: past its end, or fewer than {}",
                        size_of::<RgfxVideoEncodeSettings>()
                    )));
                }
            }
        }
    };
    let bitstream = objs.require_buffer(m.bitstream_buffer_obj_id)?;
    let in_bitstream = (m.bitstream_offset.checked_add(m.bitstream_capacity))
        .is_some_and(|end| m.bitstream_capacity > 0 && end <= bitstream.length());
    if !in_bitstream {
        return Err(fault(format_args!(
            "a capacity of {} bytes at {} past its buffer, or none",
            m.bitstream_capacity, m.bitstream_offset
        )));
    }
    let result = objs.require_buffer(m.result_buffer_obj_id)?;
    let result_bytes = size_of::<RgfxVideoEncodeResult>() as u64;
    let in_result = m.result_offset.is_multiple_of(8)
        && (m.result_offset.checked_add(result_bytes)).is_some_and(|end| end <= result.length());
    if !in_result {
        return Err(fault(format_args!(
            "a result at {} past its buffer, or not at a multiple of 8",
            m.result_offset
        )));
    }
    let (luma, chroma) = objs.video_planes(
        tag,
        m.source_luma_texture_obj_id,
        m.source_chroma_texture_obj_id,
    )?;
    let encode = VideoEncode {
        luma,
        chroma,
        bitstream,
        bitstream_offset: m.bitstream_offset,
        bitstream_capacity: m.bitstream_capacity,
        result,
        result_offset: m.result_offset,
        frame_index: m.frame_index,
        force_keyframe: m.flags & RGFX_VIDEO_ENCODE_FORCE_KEYFRAME != 0,
        settings,
    };
    // After the buffers of other queues its command buffer runs after: the writes of the
    // source among them.
    objs.wait_for_order_on_cpu();
    let value = session.encode(&encode)?;
    cb.encode_wait(Some(session.event()), value)?;
    Ok(())
}

impl<P: Platform> GraphicsDecoder<P> {
    /// The host's hardware video, which `VIDEO_SESSION_CREATE` makes its sessions on.
    pub(crate) fn set_video(&mut self, video: Arc<dyn GraphicsVideo<P>>) {
        // Set once, at the device's make.
        let _ = self.shared.video.set(video);
    }

    /// Makes the device's caches exportable from now on, for the host's video to import
    /// ([`GraphicsVideo::imports_device_caches`]): once, at the device's make, before any cache.
    pub(crate) fn set_exportable_caches(&self) {
        self.shared.surface_caches.set_exportable();
    }

    /// Makes the device's laid-out caches of two planes dma-bufs from now on, for the host's video
    /// to decode into ([`GraphicsVideo::decodes_into_device_caches`]): once, at the device's make,
    /// before any cache.
    pub(crate) fn set_dma_buf_caches(&self) {
        self.shared.surface_caches.set_dma_buf();
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

    /// The planes of the IOSurface textures `luma_id` and `chroma_id` lie over, for a video op
    /// `tag`: each where it lies, in its surface's device cache where its block has one, else in
    /// its pages. A cache the host is filling, or copying a picture out of into the pages, is
    /// waited for first, so a picture never lands over a fill's older bytes and a copy never
    /// reads a picture landing, and a read finds the cache whole.
    fn video_planes(
        &self,
        tag: u16,
        luma_id: u64,
        chroma_id: u64,
    ) -> RaiseResult<(VideoPlane<'_, P>, VideoPlane<'_, P>)> {
        let (luma_place, luma) = self.plane_of(luma_id)?;
        let (chroma_place, chroma) = self.plane_of(chroma_id)?;
        // The block of a plane in its surface's device cache.
        let cache_of = |buffer_obj_id| {
            (self.span_of(buffer_obj_id))
                .map(|(block, _)| block)
                .filter(|&block| self.surface_settled(block) && self.surface_caches.has(block))
        };
        let (luma_cache, chroma_cache) = (
            cache_of(luma_place.buffer_obj_id),
            cache_of(chroma_place.buffer_obj_id),
        );
        for block in luma_cache.into_iter().chain(chroma_cache) {
            if !self.surface_caches.wait_filled(block, FILL_WAIT) {
                return Err(op_fault(format_args!(
                    "video op {tag:#06x}: its surface's device cache was not filled within \
                     {FILL_WAIT:?}"
                )));
            }
            if !self.surface_caches.wait_copied_out(block, FILL_WAIT) {
                return Err(op_fault(format_args!(
                    "video op {tag:#06x}: its surface's last picture was not copied into its \
                     pages within {FILL_WAIT:?}"
                )));
            }
        }
        let plane = |place: PlanePlace, buffer, cache: Option<u64>| VideoPlane {
            buffer,
            offset: place.offset,
            bytes_per_row: place.bytes_per_row,
            row_bytes: place.row_bytes,
            width: place.width,
            height: place.height,
            cached: cache.is_some(),
            pages: (cache.is_none())
                .then(|| self.plane_pages(place.buffer_obj_id, buffer))
                .flatten(),
            cache: cache.and_then(|block| self.surface_caches.identity(block)),
        };
        Ok((
            plane(luma_place, luma, luma_cache),
            plane(chroma_place, chroma, chroma_cache),
        ))
    }

    /// The memory the buffer `obj_id`, `buffer`, lies in, as the host's video devices take it
    /// ([`PlanePages`]): its surface's view where it lies over one, its block of the region where
    /// it lies in one; `None` for a buffer over neither, or whose bytes lie outside it.
    fn plane_pages(&self, obj_id: u64, buffer: &P::Buffer) -> Option<PlanePages> {
        let entry = self.entry(obj_id)?;
        let contents = u64::try_from(buffer.contents().addr()).ok()?;
        let length = buffer.length();
        let (memory, first) = if entry.over_view {
            let (block, _) = entry.span?;
            let view = self.shared.surface_views.lock().view(block)?;
            (view.memory()?, u64::try_from(view.base().addr()).ok()?)
        } else {
            let bytes = entry.bytes.filter(|b| b.ram.is_none())?;
            let (block, size) = self.region.block_of(bytes.at, bytes.length)?;
            let identity = (self.shared.block_lives.lock())
                .entry(block)
                .or_default()
                .identity();
            let page = self.platform.vm().page_size();
            let memory = RegionMemory::new(&self.region, block, size, page, identity)?;
            let first = memory.first;
            (Arc::new(memory) as Arc<dyn DeviceMemory>, first)
        };
        let buffer_offset = contents.checked_sub(first)?;
        let fits = (buffer_offset.checked_add(length))
            .zip(u64::try_from(memory.length()).ok())
            .is_some_and(|(end, bytes)| end <= bytes);
        fits.then_some(PlanePages {
            memory,
            buffer_offset,
        })
    }

    /// The blocks of the surfaces the textures `luma_id` and `chroma_id` of a decode lie over,
    /// each once, in that order.
    fn decode_blocks(&self, luma_id: u64, chroma_id: u64) -> Vec<u64> {
        let block = |id| {
            let place = self.entry(id)?.place?;
            self.span_of(place.buffer_obj_id).map(|(block, _)| block)
        };
        let mut blocks: Vec<u64> = [block(luma_id), block(chroma_id)]
            .into_iter()
            .flatten()
            .collect();
        blocks.dedup();
        blocks
    }

    /// The blocks whose surfaces the decodes `decodes` (each one's luma and chroma textures)
    /// write whole, each with the two planes, as laid out in the surface's bytes
    /// ([`Reach::decoded`](super::surface_cache::Reach::decoded)): those whose luma texture
    /// shows plane 0 whole and chroma texture plane 1 whole ([`PlanePlace::whole`]), both in
    /// the same buffer over the block.
    pub(super) fn decoded_whole(&self, decodes: &[(u64, u64)]) -> Vec<(u64, [PagesPlane; 2])> {
        let place = |id| match self.entry(id) {
            Some(e) if matches!(e.handle, ObjHandle::Texture(_)) => e.place,
            _ => None,
        };
        let plane = |p: PlanePlace| PagesPlane {
            offset: p.offset,
            bytes_per_row: p.bytes_per_row,
            row_bytes: p.row_bytes,
            rows: p.height,
        };
        (decodes.iter())
            .filter_map(|&(luma, chroma)| {
                let (luma, chroma) = (place(luma)?, place(chroma)?);
                let whole = luma.whole == Some(0) && chroma.whole == Some(1);
                if !whole || luma.buffer_obj_id != chroma.buffer_obj_id {
                    return None;
                }
                let (block, _) = self.span_of(luma.buffer_obj_id)?;
                Some((block, [plane(luma), plane(chroma)]))
            })
            .collect()
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

    /// The decode caps of HEVC decode name the codec crate's narrowest tile column, and those of
    /// H.264 decode and of an encode of either nothing.
    #[test]
    fn only_hevc_decode_bounds_its_tile_columns() {
        let entry = |codec, direction| RgfxVideoCodecCaps {
            codec,
            direction,
            ..RgfxVideoCodecCaps::default()
        };
        use reims_vgpu_graphics::wire::RGFX_VIDEO_ENCODE;
        assert_eq!(
            codec_decode_caps(&entry(RGFX_VIDEO_CODEC_HEVC, RGFX_VIDEO_DECODE)),
            RgfxVideoDecodeCaps {
                min_tile_width: 128,
                ..RgfxVideoDecodeCaps::default()
            }
        );
        for (codec, direction) in [
            (RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_DECODE),
            (RGFX_VIDEO_CODEC_HEVC, RGFX_VIDEO_ENCODE),
            (RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_ENCODE),
        ] {
            assert_eq!(
                codec_decode_caps(&entry(codec, direction)),
                RgfxVideoDecodeCaps::default()
            );
        }
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
