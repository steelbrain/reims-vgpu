//! The command-buffer op stream, as the guest and the host share it: the
//! [`RgfxCommandBuffer`] message that heads an
//! [`RGFX_CMD_COMMAND_BUFFER`](crate::wire::RGFX_CMD_COMMAND_BUFFER) and
//! the byte stream of encoder ops that follows it (docs/graphics-kext/design.md,
//! "Stage 3 wire protocol: the Metal-call stream"). The guest's Metal GPU bundle
//! serialises each `MTLCommandBuffer` into one of these; the host walks the op
//! stream and drives the library's `Platform`.
//!
//! **An op.** Every op starts with an [`RgfxOpHeader`] — its `tag`, `flags` and
//! `len` — followed by the op's body. `len` is the whole op's byte count, header
//! included, a multiple of 8, so the walker steps `len` bytes to the next op. The
//! op's body struct (the `RgfxOp*` records in [`render`], and later [`compute`]
//! and [`blit`]) is the bytes **after** the header, not the whole op: unlike a
//! ring message, whose struct is the whole message, an op's fields are the body
//! alone and the walker has already read the header. A body with no fields (an
//! `ENCODER_END`) is a bare header.
//!
//! **Tag ranges.** An op's `tag` names both its group and its op. The group is
//! `tag >> 12` ([`op_group`]):
//!
//! | group | `tag >> 12` | tags | module |
//! |---|---|---|---|
//! | framing / command buffer | 0 | `0x00xx` | this module |
//! | render encoder | 1 | `0x1xxx` | [`render`] |
//! | compute encoder | 2 | `0x2xxx` | [`compute`] |
//! | blit encoder | 3 | `0x3xxx` | [`blit`] |
//! | video encoder | 4 | `0x4xxx` | [`video`] |
//!
//! The host dispatches by group: framing ops (encoder begin/end) it handles
//! itself, and each encoder group's ops it hands to that group's decoder.
//!
//! **Object references.** Every reference is a `u64` `obj_id` the guest picked,
//! and `0` is nil. A reference that may be nil is resolved as an `Option`; a
//! reference that must name an object faults the stream where it is `0` or names
//! no object.
//!
//! Every value is little-endian, as [`crate::wire`]'s are, and the structs follow
//! its layout rules (a `u64` never at an offset that is not a multiple of 8, a
//! `size_of` a multiple of 8). cbindgen writes them into
//! `include/reims_graphics_wire.h`.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::wire::RgfxMsgHeader;

pub mod blit;
pub mod compute;
pub mod render;
pub mod video;

/// The group of ops a `tag` belongs to: `tag >> 12`.
pub const fn op_group(tag: u16) -> u16 {
    tag >> 12
}

/// The framing / command-buffer group (`tag >> 12`): encoder begin and end, the command
/// buffer's own event signals and waits, and the IOSurfaces it uses whose blocks have a device
/// cache.
pub const RGFX_OP_GROUP_FRAMING: u16 = 0x0;
/// The render-encoder group (`tag >> 12`): [`render`].
pub const RGFX_OP_GROUP_RENDER: u16 = 0x1;
/// The compute-encoder group (`tag >> 12`): [`compute`].
pub const RGFX_OP_GROUP_COMPUTE: u16 = 0x2;
/// The blit-encoder group (`tag >> 12`): [`blit`].
pub const RGFX_OP_GROUP_BLIT: u16 = 0x3;
/// The video-encoder group (`tag >> 12`): [`video`].
pub const RGFX_OP_GROUP_VIDEO: u16 = 0x4;

/// Op `ENCODER_BEGIN_RENDER`: open a render encoder for the pass the op's body
/// describes (an [`RgfxRenderPassDesc`] then its `color_count`
/// [`RgfxRenderPassColorAttachment`] records, then the [`RgfxRenderPassSamplePositions`] and
/// the [`RgfxRenderPassVisibility`] its flags name, in that order). Holds the encoder until
/// [`RGFX_OP_ENCODER_END`].
pub const RGFX_OP_ENCODER_BEGIN_RENDER: u16 = 0x0001;
/// Op `ENCODER_BEGIN_COMPUTE`: open a compute encoder for the pass the op's body
/// ([`RgfxComputePassDesc`]) describes.
pub const RGFX_OP_ENCODER_BEGIN_COMPUTE: u16 = 0x0002;
/// Op `ENCODER_BEGIN_BLIT`: open a blit encoder (no body).
pub const RGFX_OP_ENCODER_BEGIN_BLIT: u16 = 0x0003;
/// Op `ENCODER_END`: end the open encoder (`endEncoding`) and drop it (no body).
pub const RGFX_OP_ENCODER_END: u16 = 0x0004;
/// Op `SIGNAL_EVENT` ([`RgfxOpEvent`]): `encodeSignalEvent:value:`, between encoders. The event is
/// an `id<MTLEvent>` or an `id<MTLSharedEvent>`; a shared one's signal is posted to the guest as
/// [`RgfxEventSignalled`](crate::wire::RgfxEventSignalled) once the buffer finishes.
pub const RGFX_OP_SIGNAL_EVENT: u16 = 0x0005;
/// Op `WAIT_EVENT` ([`RgfxOpEvent`]): `encodeWaitForEvent:value:`, between encoders: the work
/// after it waits until the event reaches `value`.
pub const RGFX_OP_WAIT_EVENT: u16 = 0x0006;
/// Op `SURFACE_USE` ([`RgfxOpSurfaceUse`]): the command buffer uses the IOSurface whose block has a
/// device cache ([`RGFX_SURFACE_PAGES_CACHED`](crate::wire::RGFX_SURFACE_PAGES_CACHED)) through a
/// texture or buffer over it, as the CPU had left it at generation `cpu_gen` when the buffer was
/// committed. One for each such block the buffer uses (an attachment, a bound texture or buffer, a
/// resource it declares it uses, a blit's source or destination), before its first encoder. The
/// host fills the cache from the surface's pages before the buffer's work where the cache holds an
/// older generation, and orders a later copy of the cache after the buffer. An op naming a block
/// with no cache does nothing.
pub const RGFX_OP_SURFACE_USE: u16 = 0x0007;
/// Op `SUBMITTED` (no body), where the device has
/// [`RGFX_FEATURE_CB_SUBMITTED`](crate::wire::RGFX_FEATURE_CB_SUBMITTED): the command buffer is
/// committed by `commitAndWaitUntilSubmitted`, and asks for its
/// [`RGFX_EVT_CB_SUBMITTED`](crate::wire::RGFX_EVT_CB_SUBMITTED). Anywhere in the stream, once:
/// the host reads it as it takes the buffer, before its other ops. The bundle puts it last, as it
/// commits.
pub const RGFX_OP_SUBMITTED: u16 = 0x0008;
/// Op `ENCODER_BEGIN_VIDEO` ([`RgfxOpVideoPass`]): open a video encoder on the session the body
/// names, for the [`video`] ops, until [`RGFX_OP_ENCODER_END`]. A command buffer with a video
/// encoder has no other encoder.
pub const RGFX_OP_ENCODER_BEGIN_VIDEO: u16 = 0x0009;
/// Op `QUEUE_BARRIER` ([`RgfxOpQueueBarrier`]), where the host offers
/// [`RGFX_FEATURE_QUEUE_BARRIER`](crate::wire::RGFX_FEATURE_QUEUE_BARRIER):
/// `barrierAfterQueueStages:beforeStages:` on the open render, compute or blit encoder, at the
/// point the op stands. No host offers the feature yet, and a host without it faults the buffer at
/// the op, as at any tag it does not know.
pub const RGFX_OP_QUEUE_BARRIER: u16 = 0x000a;
/// Op `SCHEDULED_WANTED` (no body), where the guest took
/// [`RGFX_FEATURE_QUIET_SCHEDULED`](crate::wire::RGFX_FEATURE_QUIET_SCHEDULED): on a quiet queue,
/// the command buffer's `CB_SCHEDULED` raises the ring's vector. Anywhere in the stream, once, as
/// [`RGFX_OP_SUBMITTED`]. No host offers the feature yet.
pub const RGFX_OP_SCHEDULED_WANTED: u16 = 0x000b;
/// Op `SURFACE_ORDER` (as many [`RgfxSurfaceOrderWait`]s as its `len` holds, none among them),
/// where the guest took [`RGFX_FEATURE_SURFACE_ORDER`](crate::wire::RGFX_FEATURE_SURFACE_ORDER):
/// the command buffer uses IOSurfaces, and runs after each command buffer of another queue its body
/// names, as a Mac's GPU orders a process's access to an IOSurface after another's committed
/// before it where one of them writes. Before its first encoder, once. The bundle names, as it
/// commits, each surface's writers (`RgfxSurfaceWriters`) of other queues, and where the buffer
/// writes the surface its readers too; the buffer's own queue and seqno then go in the surface's
/// record for the buffers committed after it. The host makes the buffer wait on the GPU for each
/// named buffer that has not finished, and has the buffer signal its own queue's order as it ends,
/// for the buffers that name it.
pub const RGFX_OP_SURFACE_ORDER: u16 = 0x000c;

/// [`RgfxOpSurfaceUse::flags`]: the command buffer may write the surface.
pub const RGFX_SURFACE_USE_WRITES: u32 = 1 << 0;

/// The body of [`RGFX_OP_SURFACE_USE`].
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSurfaceUse {
    /// The surface's shared block's region offset.
    pub block_offset: u64,
    /// The kext's count of the surface's read-write CPU locks when the buffer was committed.
    pub cpu_gen: u64,
    /// `RGFX_SURFACE_USE_*`: [`RGFX_SURFACE_USE_WRITES`].
    pub flags: u32,
    /// Zero.
    pub reserved_14: u32,
}

/// The body of [`RGFX_OP_ENCODER_BEGIN_VIDEO`].
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpVideoPass {
    /// The video session
    /// ([`RGFX_CMD_VIDEO_SESSION_CREATE`](crate::wire::RGFX_CMD_VIDEO_SESSION_CREATE)'s object).
    pub session_obj_id: u64,
}

/// The body of [`RGFX_OP_QUEUE_BARRIER`]: raw `MTLStages`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpQueueBarrier {
    /// The stages of the work before the barrier that the work after it waits for.
    pub after_stages: u64,
    /// The stages of the work after the barrier that wait.
    pub before_stages: u64,
}

/// One command buffer an [`RGFX_OP_SURFACE_ORDER`]'s buffer runs after: the one at `seqno` on the
/// submission ring of queue `queue_obj_id`, and the queue's buffers before it.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxSurfaceOrderWait {
    /// The queue's id.
    pub queue_obj_id: u64,
    /// The buffer's `seqno` on the queue's submission ring.
    pub seqno: u64,
}

/// The body of [`RGFX_OP_SIGNAL_EVENT`] and [`RGFX_OP_WAIT_EVENT`].
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpEvent {
    /// The event (`EVENT_CREATE`'s or `SHARED_EVENT_CREATE`'s object).
    pub event_obj_id: u64,
    /// The value signalled, or waited for.
    pub value: u64,
}

/// Every op's header: its tag, its flags and its total byte count.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpHeader {
    /// The op's tag: its group ([`op_group`]) and its op within the group.
    pub tag: u16,
    /// Per-op flags; 0 where an op defines none.
    pub flags: u16,
    /// The op's bytes, this header included: a multiple of 8.
    pub len: u32,
}

/// Where an op's body starts in the op: right after its header.
pub const RGFX_OP_BODY_OFFSET: u32 = 8;

/// `COMMAND_BUFFER` ([`RGFX_CMD_COMMAND_BUFFER`](crate::wire::RGFX_CMD_COMMAND_BUFFER)):
/// one command buffer on a queue's submission ring. The op stream (`op_bytes`
/// bytes) follows this message at [`RGFX_COMMAND_BUFFER_OPS_OFFSET`]; the host
/// opens a command buffer on the queue `queue_obj_id`, walks the ops, commits,
/// and answers an [`RgfxCbComplete`](crate::wire::RgfxCbComplete) carrying
/// `cb_obj_id` when the work finishes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxCommandBuffer {
    /// [`RGFX_CMD_COMMAND_BUFFER`](crate::wire::RGFX_CMD_COMMAND_BUFFER).
    pub header: RgfxMsgHeader,
    /// The command queue this buffer belongs to.
    pub queue_obj_id: u64,
    /// The command buffer's guest-picked id, which its completion names.
    pub cb_obj_id: u64,
    /// The bytes of op stream that follow this message's fixed part: all of the message's
    /// bytes past it. A message whose `op_bytes` counts fewer or more is refused.
    pub op_bytes: u64,
}

/// Where a `COMMAND_BUFFER` message's op stream starts: right after its header.
pub const RGFX_COMMAND_BUFFER_OPS_OFFSET: u32 = 40;

// ---------------------------------------------------------------------------
// The render-pass descriptor wire schema, carried by an `ENCODER_BEGIN_RENDER`
// op: a fixed [`RgfxRenderPassDesc`] (the pass's own properties and the depth and
// stencil attachments) then `color_count` [`RgfxRenderPassColorAttachment`]
// records. An attachment whose `texture_obj_id` is 0 is absent (nil), and sets
// nothing; a non-zero `resolve_texture_obj_id` adds the resolve. Clear values
// travel as the bits of their `f64`/`u32`, so the schema stays all-integer and
// comparable, as the sampler's LOD clamps do in `wire`.
// ---------------------------------------------------------------------------

/// A pass's `render_target_array_length` is set (else Metal's default stands).
pub const RGFX_PASS_FLAG_RT_ARRAY_LENGTH: u32 = 1 << 0;
/// A pass's `render_target_width` and `render_target_height` are set.
pub const RGFX_PASS_FLAG_RT_SIZE: u32 = 1 << 1;
/// A pass's `default_raster_sample_count` is set.
pub const RGFX_PASS_FLAG_DEFAULT_RASTER_SAMPLE_COUNT: u32 = 1 << 2;
/// A pass's sample positions (`setSamplePositions:count:`) follow its colour attachment records,
/// as one [`RgfxRenderPassSamplePositions`].
pub const RGFX_PASS_FLAG_SAMPLE_POSITIONS: u32 = 1 << 3;
/// The most sample positions a pass holds: Metal's largest sample count.
pub const RGFX_PASS_MAX_SAMPLE_POSITIONS: u32 = 16;
/// A pass's visibility result buffer (`visibilityResultBuffer`) and its
/// `visibilityResultType` follow its colour attachment records and its sample positions, as one
/// [`RgfxRenderPassVisibility`]. A host that does not know the flag takes the pass without the
/// buffer, so its `setVisibilityResultMode:offset:` of a mode fails the command buffer, as
/// before the flag.
pub const RGFX_PASS_FLAG_VISIBILITY_RESULT: u32 = 1 << 4;

/// One colour attachment of a render pass (`colorAttachments[i]`). Absent (sets
/// nothing) when `texture_obj_id` is 0.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxRenderPassColorAttachment {
    /// The attachment's texture, or 0 for an absent attachment.
    pub texture_obj_id: u64,
    /// The resolve texture, or 0 for none.
    pub resolve_texture_obj_id: u64,
    /// `MTLClearColor` as the bits of its four `f64` (red, green, blue, alpha).
    pub clear_color_bits: [u64; 4],
    /// `MTLLoadAction`.
    pub load_action: u32,
    /// `MTLStoreAction`.
    pub store_action: u32,
    /// `MTLStoreActionOptions`.
    pub store_action_options: u32,
    /// The mip level.
    pub level: u32,
    /// The slice.
    pub slice: u32,
    /// The depth plane.
    pub depth_plane: u32,
    /// The resolve texture's mip level.
    pub resolve_level: u32,
    /// The resolve texture's slice.
    pub resolve_slice: u32,
    /// The resolve texture's depth plane.
    pub resolve_depth_plane: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_54: u32,
}

/// A render pass's depth attachment (`depthAttachment`). Absent when
/// `texture_obj_id` is 0.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxRenderPassDepthAttachment {
    /// The attachment's texture, or 0 for none.
    pub texture_obj_id: u64,
    /// The resolve texture, or 0 for none.
    pub resolve_texture_obj_id: u64,
    /// The clear depth as the bits of its `f64`.
    pub clear_depth_bits: u64,
    /// `MTLLoadAction`.
    pub load_action: u32,
    /// `MTLStoreAction`.
    pub store_action: u32,
    /// `MTLStoreActionOptions`.
    pub store_action_options: u32,
    /// The mip level.
    pub level: u32,
    /// The slice.
    pub slice: u32,
    /// The depth plane.
    pub depth_plane: u32,
    /// The resolve texture's mip level.
    pub resolve_level: u32,
    /// The resolve texture's slice.
    pub resolve_slice: u32,
    /// The resolve texture's depth plane.
    pub resolve_depth_plane: u32,
    /// `MTLMultisampleDepthResolveFilter`.
    pub depth_resolve_filter: u32,
}

/// A render pass's stencil attachment (`stencilAttachment`). Absent when
/// `texture_obj_id` is 0.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxRenderPassStencilAttachment {
    /// The attachment's texture, or 0 for none.
    pub texture_obj_id: u64,
    /// The resolve texture, or 0 for none.
    pub resolve_texture_obj_id: u64,
    /// `MTLLoadAction`.
    pub load_action: u32,
    /// `MTLStoreAction`.
    pub store_action: u32,
    /// `MTLStoreActionOptions`.
    pub store_action_options: u32,
    /// The mip level.
    pub level: u32,
    /// The slice.
    pub slice: u32,
    /// The depth plane.
    pub depth_plane: u32,
    /// The resolve texture's mip level.
    pub resolve_level: u32,
    /// The resolve texture's slice.
    pub resolve_slice: u32,
    /// The resolve texture's depth plane.
    pub resolve_depth_plane: u32,
    /// `MTLMultisampleStencilResolveFilter`.
    pub stencil_resolve_filter: u32,
    /// The clear stencil value.
    pub clear_stencil: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_3c: u32,
}

/// The fixed head of an `ENCODER_BEGIN_RENDER` op's body: the pass's own
/// properties and its depth and stencil attachments. `color_count`
/// [`RgfxRenderPassColorAttachment`] records follow it, at
/// [`RGFX_RENDER_PASS_COLOR_OFFSET`]. `flags` is an `RGFX_PASS_FLAG_*` set that
/// says which of the optional pass properties are set.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxRenderPassDesc {
    /// The colour attachment records that follow this struct.
    pub color_count: u32,
    /// An `RGFX_PASS_FLAG_*` set: which optional pass properties are set.
    pub flags: u32,
    /// `renderTargetArrayLength`, when [`RGFX_PASS_FLAG_RT_ARRAY_LENGTH`] is set.
    pub render_target_array_length: u32,
    /// `renderTargetWidth`, when [`RGFX_PASS_FLAG_RT_SIZE`] is set.
    pub render_target_width: u32,
    /// `renderTargetHeight`, when [`RGFX_PASS_FLAG_RT_SIZE`] is set.
    pub render_target_height: u32,
    /// `defaultRasterSampleCount`, when [`RGFX_PASS_FLAG_DEFAULT_RASTER_SAMPLE_COUNT`] is set.
    pub default_raster_sample_count: u32,
    /// The depth attachment (absent when its `texture_obj_id` is 0).
    pub depth: RgfxRenderPassDepthAttachment,
    /// The stencil attachment (absent when its `texture_obj_id` is 0).
    pub stencil: RgfxRenderPassStencilAttachment,
}

/// Where the colour attachment array starts in an `ENCODER_BEGIN_RENDER` op's
/// body: right after the fixed [`RgfxRenderPassDesc`].
pub const RGFX_RENDER_PASS_COLOR_OFFSET: u32 = 152;

/// A pass's sample positions, after its colour attachment records where the pass's flags hold
/// [`RGFX_PASS_FLAG_SAMPLE_POSITIONS`]: the `count` positions the pass descriptor holds
/// (`getSamplePositions:count:`), each coordinate as the bits of its `f32`; the rest 0.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxRenderPassSamplePositions {
    /// The positions set, at most [`RGFX_PASS_MAX_SAMPLE_POSITIONS`].
    pub count: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_04: u32,
    /// Each position's `x` then `y`, as `f32` bits.
    pub position_bits: [[u32; 2]; RGFX_PASS_MAX_SAMPLE_POSITIONS as usize],
}

/// A pass's visibility result buffer, after its colour attachment records and its sample
/// positions where the pass's flags hold [`RGFX_PASS_FLAG_VISIBILITY_RESULT`].
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxRenderPassVisibility {
    /// The buffer each `setVisibilityResultMode:offset:` of a mode writes its 64-bit result
    /// into, at its offset: a buffer of the sender's.
    pub buffer_obj_id: u64,
    /// `MTLVisibilityResultType` (`visibilityResultType`, macOS 26): 0 `Reset`, 1 `Accumulate`;
    /// the host refuses any other value.
    pub result_type: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_0c: u32,
}

/// The body of an `ENCODER_BEGIN_COMPUTE` op: the compute pass's properties.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputePassDesc {
    /// `MTLDispatchType`, when [`RGFX_COMPUTE_PASS_FLAG_DISPATCH_TYPE`] is set.
    pub dispatch_type: u32,
    /// An `RGFX_COMPUTE_PASS_FLAG_*` set.
    pub flags: u32,
}

/// A compute pass's `dispatch_type` is set (else Metal's default stands).
pub const RGFX_COMPUTE_PASS_FLAG_DISPATCH_TYPE: u32 = 1 << 0;

// Resource-kind tags for a resource reference inside an encoder op (`useResources:`,
// `memoryBarrierWithResources:`, `synchronizeResource:`): they pick the resolver the host runs
// `obj_id` through. Defined once here so the guest-facing ABI has a single numbering for every
// encoder group.
/// A resource reference that names nothing (a nil entry; `obj_id` is ignored).
pub const RGFX_RESOURCE_KIND_NONE: u32 = 0;
/// A resource reference whose `obj_id` names an `id<MTLBuffer>`.
pub const RGFX_RESOURCE_KIND_BUFFER: u32 = 1;
/// A resource reference whose `obj_id` names an `id<MTLTexture>`.
pub const RGFX_RESOURCE_KIND_TEXTURE: u32 = 2;
/// A resource reference whose `obj_id` names an `id<MTLIndirectCommandBuffer>`.
pub const RGFX_RESOURCE_KIND_ICB: u32 = 3;

// The op-stream layout.
const _: () = {
    use core::mem::size_of;
    assert!(size_of::<RgfxOpHeader>() == 8);
    assert!(RGFX_OP_BODY_OFFSET as usize == size_of::<RgfxOpHeader>());
    assert!(size_of::<RgfxCommandBuffer>() == 40);
    assert!(RGFX_COMMAND_BUFFER_OPS_OFFSET as usize == size_of::<RgfxCommandBuffer>());
    assert!(size_of::<RgfxRenderPassColorAttachment>() == 88);
    assert!(size_of::<RgfxRenderPassDepthAttachment>() == 64);
    assert!(size_of::<RgfxRenderPassStencilAttachment>() == 64);
    assert!(size_of::<RgfxRenderPassDesc>() == 152);
    assert!(RGFX_RENDER_PASS_COLOR_OFFSET as usize == size_of::<RgfxRenderPassDesc>());
    assert!(size_of::<RgfxRenderPassSamplePositions>() == 136);
    assert!(size_of::<RgfxRenderPassVisibility>() == 16);
    assert!(size_of::<RgfxComputePassDesc>() == 8);
    assert!(size_of::<RgfxOpEvent>() == 16);
    assert!(size_of::<RgfxOpSurfaceUse>() == 24);
    assert!(size_of::<RgfxOpVideoPass>() == 8);
    assert!(size_of::<RgfxOpHeader>() + size_of::<RgfxOpQueueBarrier>() == 24);
    assert!(op_group(RGFX_OP_QUEUE_BARRIER) == RGFX_OP_GROUP_FRAMING);
    assert!(op_group(RGFX_OP_SCHEDULED_WANTED) == RGFX_OP_GROUP_FRAMING);
    assert!(op_group(RGFX_OP_SURFACE_ORDER) == RGFX_OP_GROUP_FRAMING);
    assert!(size_of::<RgfxSurfaceOrderWait>() == 16);
    assert!(op_group(RGFX_OP_ENCODER_BEGIN_VIDEO) == RGFX_OP_GROUP_FRAMING);
    assert!(op_group(RGFX_OP_SURFACE_USE) == RGFX_OP_GROUP_FRAMING);
    // Each group's tag range maps to its own `op_group`.
    assert!(op_group(RGFX_OP_ENCODER_BEGIN_RENDER) == RGFX_OP_GROUP_FRAMING);
    assert!(op_group(RGFX_OP_ENCODER_END) == RGFX_OP_GROUP_FRAMING);
    assert!(op_group(RGFX_OP_WAIT_EVENT) == RGFX_OP_GROUP_FRAMING);
    assert!(op_group(render::RGFX_OP_RENDER_DRAW_PRIMITIVES) == RGFX_OP_GROUP_RENDER);
};

#[cfg(test)]
mod tests {
    use super::*;
    use zerocopy::{FromBytes, IntoBytes};

    /// The op header and the command-buffer message round-trip, and the group of
    /// each framing tag is the framing group.
    #[test]
    fn framing_round_trips() {
        let op = RgfxOpHeader {
            tag: RGFX_OP_ENCODER_BEGIN_RENDER,
            flags: 0,
            len: 8 + size_of::<RgfxRenderPassDesc>() as u32,
        };
        assert_eq!(op.as_bytes().len(), 8);
        assert_eq!(RgfxOpHeader::read_from_bytes(op.as_bytes()).unwrap(), op);

        let cb = RgfxCommandBuffer {
            header: RgfxMsgHeader {
                r#type: crate::wire::RGFX_CMD_COMMAND_BUFFER,
                size: 48,
                seqno: 9,
            },
            queue_obj_id: 3,
            cb_obj_id: 77,
            op_bytes: 8,
        };
        assert_eq!(cb.as_bytes().len(), 40);
        assert_eq!(
            RgfxCommandBuffer::read_from_bytes(cb.as_bytes()).unwrap(),
            cb
        );

        for tag in [
            RGFX_OP_ENCODER_BEGIN_RENDER,
            RGFX_OP_ENCODER_BEGIN_COMPUTE,
            RGFX_OP_ENCODER_BEGIN_BLIT,
            RGFX_OP_ENCODER_END,
        ] {
            assert_eq!(op_group(tag), RGFX_OP_GROUP_FRAMING);
        }
    }

    /// The render-pass schema round-trips, clear values travelling as bits.
    #[test]
    fn render_pass_schema_round_trips() {
        let desc = RgfxRenderPassDesc {
            color_count: 1,
            flags: RGFX_PASS_FLAG_RT_SIZE,
            render_target_width: 1920,
            render_target_height: 1080,
            depth: RgfxRenderPassDepthAttachment {
                texture_obj_id: 5,
                clear_depth_bits: 1.0f64.to_bits(),
                load_action: 2,
                store_action: 1,
                ..RgfxRenderPassDepthAttachment::default()
            },
            ..RgfxRenderPassDesc::default()
        };
        assert_eq!(
            RgfxRenderPassDesc::read_from_bytes(desc.as_bytes()).unwrap(),
            desc
        );
        assert_eq!(f64::from_bits(desc.depth.clear_depth_bits), 1.0);

        let color = RgfxRenderPassColorAttachment {
            texture_obj_id: 7,
            clear_color_bits: [
                0.0f64.to_bits(),
                0.25f64.to_bits(),
                0.5f64.to_bits(),
                1.0f64.to_bits(),
            ],
            load_action: 2,
            store_action: 1,
            ..RgfxRenderPassColorAttachment::default()
        };
        assert_eq!(
            RgfxRenderPassColorAttachment::read_from_bytes(color.as_bytes()).unwrap(),
            color
        );

        let visibility = RgfxRenderPassVisibility {
            buffer_obj_id: 9,
            result_type: 1,
            reserved_0c: 0,
        };
        assert_eq!(
            RgfxRenderPassVisibility::read_from_bytes(visibility.as_bytes()).unwrap(),
            visibility
        );

        let pass = RgfxComputePassDesc {
            dispatch_type: 1,
            flags: RGFX_COMPUTE_PASS_FLAG_DISPATCH_TYPE,
        };
        assert_eq!(
            RgfxComputePassDesc::read_from_bytes(pass.as_bytes()).unwrap(),
            pass
        );
    }
}
