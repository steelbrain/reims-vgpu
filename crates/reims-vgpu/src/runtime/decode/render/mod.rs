//! Render command decoder (port of `host/utils/reims-vgpu-render-decode`).
//!
//! Full opcode matrix is preserved for supported/rejected classification.
//! Per-opcode payload layouts for the highest-traffic families (set pipeline,
//! buffer/texture binds, draw, viewport/scissor, barriers, fences) are decoded;
//! remaining accepted opcodes are recognized and returned as typed kinds with
//! raw length validation where the contract specifies fixed sizes.

use reims_vgpu_wire::ops::render as wire;
use reims_vgpu_wire::ops::render_pass as wire_pass;
use reims_vgpu_wire::ops::tile as wire_tile;

use reims_vgpu_wire::OP_HEADER_LEN;

/// Map a `reims-vgpu-wire` view onto a payload, translating its refusal.
///
/// The draw layouts live in that crate, derived from Apple's own serializer and
/// pinned by fixtures, so this module reads them rather than restating them —
/// one declaration, and drift is impossible rather than merely detectable. What
/// stays here is everything the crate cannot know: which `Kind` a record is,
/// what the runtime's `Command` calls each field, and how a refusal is named.
#[inline]
fn wire_view<T: reims_vgpu_wire::Wire>(payload: &[u8]) -> Result<&T, DecodeStatus> {
    reims_vgpu_wire::view::<T>(payload).map_err(|_| DecodeStatus::ErrShort)
}

/// Narrow a wide draw's 64-bit count to the 32 bits `Command` carries.
///
/// The wide forms exist because the guest had a value above 16 bits, not above
/// 32: a vertex or index count of four billion is not a draw any GPU completes.
/// Truncating one would draw the wrong geometry in silence, so it is refused by
/// name instead.
#[inline]
fn narrow_count(value: u64) -> Result<u32, DecodeStatus> {
    u32::try_from(value).map_err(|_| DecodeStatus::ErrCountOutOfRange)
}

/// The instance count a selector that carries one on the wire asked for.
///
/// Eight decode arms wrote `(…get() as u32).max(1)` and none of them said why.
/// The clamp only does anything when the guest serialized `instanceCount:0`, and
/// there it draws one instance against whatever the instance buffers happen to
/// hold — the selector's own argument, overruled.
///
/// Three arms of this device disagree about that value and were never diffed
/// against each other. This one clamps it. `backend::metal::render` refuses it
/// by name as `metal_render_instance_count_zero` — a refusal this clamp makes
/// unreachable, so the two cannot both be describing the contract. And
/// `runtime::icb` decodes the same argument out of an ICB slot and hands it
/// straight to `drawPrimitives:…:instanceCount:` with no clamp at all, so the
/// device already ships a zero instance count to Metal on the path that happens
/// not to come through here.
///
/// A zero is counted here and nowhere else, as `draw_instance_count_zero` — a
/// census rather than a failure, because a guest that culled every instance
/// would be expected control flow rather than lost work.
///
/// **It reads zero.** Driven x86/Vulkan boot, Ventura desktop, Safari window
/// drag: never fired against `mrt_draw_single` in the thousands. So no guest
/// command on this workload reaches the clamp, and it is inert rather than
/// load-bearing — which is also why it has not been replaced. Both candidate
/// replacements (pass the zero through as the ICB path does, or refuse it as
/// `backend::metal::render` does) would be unobservable here, so the reading
/// cannot choose between them, and Metal's own validation rejects
/// `instanceCount:0` outright — which is the reason to doubt any guest emits it.
///
/// A firing is the signal. It would mean a real selector carries zero, and the
/// arm that receives it decides the pixels: this one draws one instance, the
/// Metal backend refuses the draw, the ICB path draws nothing.
///
/// Because this is the single site, the count it guarantees is what let the
/// four further `.max(1)`s downstream of it go: two in `runtime::exec`, one in
/// `runtime::draw` and one in `runtime::draw::vulkan`, each re-applying a rule
/// already applied here. The last of those outlived the sweep that claimed all
/// of them, which is the failure mode a restatement has: it changes nothing
/// until the rule it copies changes, and then it changes one arm.
#[inline]
fn wire_instance_count(value: u64) -> Result<u32, DecodeStatus> {
    let count = narrow_count(value)?;
    if count == 0 {
        crate::runtime::drain::note_store_route("draw_instance_count_zero");
        return Ok(1);
    }
    Ok(count)
}

// Layout lengths for fixed-size records and bind tables. Opcodes live in
// `reims_vgpu_wire::ops::{render,render_pass,tile}`; this module maps them into
// product `Kind`/`Command` and does not re-export wire opcode constants.
/// Compact `drawPrimitives:vertexStart:vertexCount:` payload length (`alloc(1, 8)`).
/// Checked exactly: a `0x1` record of any other size is not a form this contract knows.
pub const DRAW_COMPACT_PAYLOAD_LEN: usize = 8;
/// Compact draw total length including the shared op header.
pub const DRAW_COMPACT_CMD_LEN: usize = OP_HEADER_LEN + DRAW_COMPACT_PAYLOAD_LEN;
/// Fixed total lengths and field offsets for the two ICB execute records.
///
/// From the wire views, like the draw layouts above them and for the same
/// reason. These were eight literals with a note beside them saying to prefer
/// `wire::EXECUTE_COMMANDS_*_TOTAL_LEN` at new call sites — which leaves the old
/// sites reading a second transcription, and a note is not a mechanism.
#[cfg(test)]
pub(crate) const EXECUTE_INDIRECT_CMD_LEN: usize =
    wire::EXECUTE_COMMANDS_INDIRECT_TOTAL_LEN as usize;
#[cfg(test)]
pub(crate) const EXECUTE_RANGE_CMD_LEN: usize = wire::EXECUTE_COMMANDS_RANGE_TOTAL_LEN as usize;
// The two records' *field* offsets are not named here at all: both arms decode
// through `wire::execute_commands_indirect` / `_range` and read the fields off
// the view, so an offset beside them would be a name for something no code
// asks. The lengths above stay because the arms length-check before viewing.

/// Render-pass attachment layout, taken from the wire structs' own fields.
///
/// The three sections are contiguous, so each record's extent is the distance to
/// the one after it and is never written down separately: depth is
/// `[0x00, 0x28)`, stencil is `[0x28, 0x4c)`, and the color slots run from 0x4c
/// at `PASS_COLOR_ATTACH_STRIDE` each. A single "depth/stencil stride" constant
/// used to state both of the first two as 0x28, which is right for depth and
/// 4 bytes too long for stencil — that spare word is color slot 0's texture ref.
///
/// Offsets are `offset_of!` / `size_of!` on
/// [`reims_vgpu_wire::ops::render_pass`]. Attachment decode maps wire attachment
/// bodies rather than hand-loading fields; `level` is sixteen bits with `slice`
/// immediately above it (a former product colour-arm u32 load swallowed the
/// slice).
#[cfg(test)]
pub(crate) const PASS_DEPTH_ATTACH_OFF: usize = 0x00;
pub const PASS_STENCIL_ATTACH_OFF: usize = core::mem::size_of::<wire_pass::DepthAttachmentBody>();
pub const PASS_COLOR_ATTACH_OFF: usize =
    PASS_STENCIL_ATTACH_OFF + core::mem::size_of::<wire_pass::StencilAttachmentBody>();
pub const PASS_COLOR_ATTACH_STRIDE: usize = core::mem::size_of::<wire_pass::ColorAttachmentBody>();
#[cfg(test)]
pub(crate) const PASS_ATTACH_TEXREF: usize =
    core::mem::offset_of!(wire_pass::AttachmentPrefix, texture_ref);
#[cfg(test)]
pub(crate) const PASS_ATTACH_RESOLVEREF: usize =
    core::mem::offset_of!(wire_pass::AttachmentPrefix, resolve_texture_ref);
pub const PASS_ATTACH_LEVEL: usize = core::mem::offset_of!(wire_pass::AttachmentPrefix, level);
#[cfg(test)]
pub(crate) const PASS_ATTACH_SLICE: usize =
    core::mem::offset_of!(wire_pass::AttachmentPrefix, slice);
#[cfg(test)]
pub(crate) const PASS_ATTACH_DEPTH_PLANE: usize =
    core::mem::offset_of!(wire_pass::AttachmentPrefix, depth_plane);
#[cfg(test)]
pub(crate) const PASS_ATTACH_LOAD_ACTION: usize =
    core::mem::offset_of!(wire_pass::AttachmentPrefix, load_action);
#[cfg(test)]
pub(crate) const PASS_ATTACH_STORE_ACTION: usize =
    core::mem::offset_of!(wire_pass::AttachmentPrefix, store_action);
#[cfg(test)]
pub(crate) const PASS_ATTACH_CLEAR_COLOR: usize =
    core::mem::offset_of!(wire_pass::ColorAttachmentBody, clear_color_bits);
#[cfg(test)]
pub(crate) const PASS_DEPTH_ATTACH_CLEAR_DEPTH: usize =
    core::mem::offset_of!(wire_pass::DepthAttachmentBody, clear_depth_bits);
#[cfg(test)]
pub(crate) const PASS_STENCIL_ATTACH_CLEAR_STENCIL: usize =
    core::mem::offset_of!(wire_pass::StencilAttachmentBody, clear_stencil);
pub const PASS_MAX_COLOR_ATTACHMENTS: usize = wire_pass::RENDER_PASS_COLOR_ATTACHMENTS;

/// Offset of the pass-level tail, past the last colour slot.
///
/// Four fields this device decodes and does not apply. They are the guest's
/// explicit statement about the pass's extent and its occlusion query buffer,
/// and none of them can be recovered from the attachments: a guest may bind a
/// 4096-wide texture and ask for a 640-wide pass.
#[cfg(test)]
pub(crate) const PASS_TAIL_OFF: usize =
    PASS_COLOR_ATTACH_OFF + PASS_MAX_COLOR_ATTACHMENTS * PASS_COLOR_ATTACH_STRIDE;
#[cfg(test)]
pub(crate) const PASS_TAIL_VISIBILITY_BUFFER_REF: usize = 0x00;
#[cfg(test)]
pub(crate) const PASS_TAIL_ARRAY_LENGTH: usize = 0x04;
#[cfg(test)]
pub(crate) const PASS_TAIL_TARGET_WIDTH: usize = 0x0c;
#[cfg(test)]
pub(crate) const PASS_TAIL_TARGET_HEIGHT: usize = 0x14;
// The five load/store ordinals this record carries are declared in
// `contract::pass_action`, not here: both backends and the Metal C ABI mirror
// consume them, and while they lived in this decoder the mirror's copy was the
// only spelling the encode path could reach.
pub const PASS_MIN_PAYLOAD: usize = PASS_COLOR_ATTACH_OFF + PASS_COLOR_ATTACH_STRIDE;
/// Count width of `setScissorRects:count:` — eight bytes, not the four used by
/// `setViewports:count:`. The element is the singular scissor payload.
pub const SCISSOR_RECTS_COUNT_LEN: usize = core::mem::size_of::<wire::SetScissorRects>();
/// Bytes one LOD-bearing sampler entry occupies: ref, then two `f32` clamps.
pub const SAMPLER_LOD_BIND_ENTRY_SIZE: usize = core::mem::size_of::<wire::SamplerLodBind>();
/// Count width of `setViewports:count:` — four bytes (see [`SCISSOR_RECTS_COUNT_LEN`]).
#[cfg(test)]
pub(crate) const VIEWPORTS_COUNT_LEN: usize = core::mem::size_of::<wire::SetViewports>();

/// Residency record head: `count:u32` at `+0` on both forms.
///
/// Four wire opcodes, in two pairs. `wire::OPCODE_USE_HEAP` (`0x1b`) and
/// `wire::OPCODE_USE_RESOURCE` (`0x89`) are the `stages:`-qualified forms the
/// render encoder declares itself; `wire::OPCODE_USE_HEAPS_NO_STAGES` (`0x86`)
/// and `wire::OPCODE_USE_RESOURCES_NO_STAGES` (`0x87`) are the unqualified ones
/// it inherits. All four reach this rail and all four count as one hint.
#[cfg(test)]
pub(crate) const RESIDENCY_COUNT: usize = core::mem::offset_of!(wire::UseResource, count);
/// `useResource:` packs `usage` and `stages` into the word at `+4` as two
/// `u16`s, so its refs begin at `+8` — the size of the head the view declares.
#[cfg(test)]
pub(crate) const USE_RESOURCE_REFS: usize = core::mem::size_of::<wire::UseResource>();
/// `useHeap:` has no `usage` at all: `stages` sits alone at `+4` as a `u16` and
/// the refs begin at `+6`. That offset is deliberately not a multiple of four —
/// reading this record with the resource record's layout skips the first heap.
/// Two heads that differ by one field is exactly the pair that must not be two
/// numbers here, so both are the view's own size.
#[cfg(test)]
pub(crate) const USE_HEAP_REFS: usize = core::mem::size_of::<wire::UseHeap>();
/// The inherited forms take no `stages:`, so `useHeaps:count:` is a bare count
/// with its refs at `+4` and `useResources:count:usage:` puts them at `+8`.
/// Three head sizes across four opcodes, which is why each reads its own.
#[cfg(test)]
pub(crate) const USE_HEAPS_NO_STAGES_REFS: usize = core::mem::size_of::<wire::UseHeapsNoStages>();
#[cfg(test)]
pub(crate) const USE_RESOURCES_NO_STAGES_REFS: usize =
    core::mem::size_of::<wire::UseResourcesNoStages>();

/// Multi-entry bind header: `first:u32 @0`, `count:u32 @4`, entries after it.
#[cfg(test)]
pub(crate) const BIND_FIRST: usize = core::mem::offset_of!(wire::BindHeader, first);
#[cfg(test)]
pub(crate) const BIND_COUNT: usize = core::mem::offset_of!(wire::BindHeader, count);
pub const BIND_ENTRIES: usize = core::mem::size_of::<wire::BindHeader>();
pub const BUFFER_BIND_ENTRY_SIZE: usize = core::mem::size_of::<wire::BufferBind>();
/// The same entry with a `u64` attribute stride appended. See
/// [`wire::OPCODE_SET_VERTEX_BUFFER_STRIDE`].
pub const BUFFER_STRIDE_BIND_ENTRY_SIZE: usize = core::mem::size_of::<wire::BufferStrideBind>();
/// `setVertexAmplificationCount:viewMappings:`: a four-byte count, then one
/// `MTLVertexAmplificationViewMapping` (two `u32`) per view.
#[cfg(test)]
pub(crate) const AMPLIFICATION_COUNT_LEN: usize =
    core::mem::size_of::<wire::VertexAmplificationHeader>();
#[cfg(test)]
pub(crate) const AMPLIFICATION_MAPPING_SIZE: usize = core::mem::size_of::<wire::ViewMapping>();
pub const REF_BIND_ENTRY_SIZE: usize = core::mem::size_of::<wire::RefBind>();

/// Bytes a bind record needs for `count` entries of `entry_size`, or `None` if
/// no record could be that long.
///
/// **A bind record is bounded by its own length and by nothing else.** This
/// replaced a `MAX_BIND_ENTRIES = 32` cap that had no citation and was not
/// Apple's: `setVertexTextures:withRange:` over a range of 40 produces a
/// 176-byte record (fixture `render_set_vertex_textures_range_40`), which that
/// cap refused with `ErrBadLength` — dropping all forty binds rather than the
/// eight that would not fit a table. Metal's own limit is 128 textures per
/// stage, so 32 was not even the API's number.
///
/// The count stays guest-controlled and is never trusted before this check:
/// nothing is allocated or read until the entries are known to be inside the
/// record the guest itself sized.
#[inline]
pub fn bind_record_len(count: u32, entry_size: usize) -> Option<usize> {
    (count as usize)
        .checked_mul(entry_size)
        .and_then(|n| n.checked_add(BIND_ENTRIES))
}
/// set*BufferOffset: index:u32 @0, offset:u64 @4 (payload 12; full cmd 0x14).
#[cfg(test)]
pub(crate) const BUFFER_OFFSET_INDEX: usize = core::mem::offset_of!(wire::BufferOffset, index);
#[cfg(test)]
pub(crate) const BUFFER_OFFSET_VALUE: usize = core::mem::offset_of!(wire::BufferOffset, offset);
#[cfg(test)]
pub(crate) const BUFFER_OFFSET_PAYLOAD_LEN: usize = core::mem::size_of::<wire::BufferOffset>();
/// One scissor rectangle's extent. Its four fields are not named here: both
/// scissor arms read them off `wire::ScissorRect` through the view.
#[cfg(test)]
pub(crate) const SCISSOR_PAYLOAD_LEN: usize = core::mem::size_of::<wire::ScissorRect>();

// Supported window is the full C-accepted encoder range 0x00..=0x98 minus rejected.

/// Why the render decoder refused a command.
///
/// No `Ok` and no `ErrArgs`, for the reason recorded on `blit::DecodeStatus`:
/// success is the result's own `Ok`, and a bad argument here is a payload
/// shorter than the field, which `ErrShort` already names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeStatus {
    ErrShort,
    ErrUnknownOpcode,
    ErrUnsupportedOpcode,
    ErrBadLength,
    /// A wide draw's 64-bit count does not fit the 32 bits `Command` carries.
    /// See [`narrow_count`] for why that is refused rather than truncated.
    ErrCountOutOfRange,
}

impl crate::observe::Refusal for DecodeStatus {
    /// Slugs carry a `render_decode_` prefix: seven modules under
    /// `runtime/decode/` define a type called `DecodeStatus`, and five of them
    /// have an `ErrShort` that means a different read. Without the prefix the
    /// crate-wide uniqueness gate could not tell the render decoder's refusals
    /// from any other's.
    fn refusal(&self) -> Option<&'static str> {
        Some(match self {
            Self::ErrShort => "render_decode_short",
            Self::ErrUnknownOpcode => "render_decode_unknown_opcode",
            Self::ErrUnsupportedOpcode => "render_decode_unsupported_opcode",
            Self::ErrBadLength => "render_decode_bad_length",
            Self::ErrCountOutOfRange => "render_decode_count_out_of_range",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Kind {
    #[default]
    Unknown,
    SetPipeline,
    SetBuffer,
    /// setVertexBufferOffset / setFragmentBufferOffset (0x7e / 0x6f).
    SetBufferOffset,
    SetTexture,
    SetSampler,
    Draw,
    SetViewport,
    SetScissor,
    SetDepthStencil,
    SetBlendColor,
    SetCullMode,
    SetFrontFacing,
    SetDepthBias,
    SetStencilReference,
    Fence,
    Barrier,
    /// `setTriangleFillMode:` / `setDepthClipMode:`. The value is in
    /// [`Command::mode`]; which state it sets is [`Command::opcode`], as on the
    /// wire.
    SetRasterState,
    /// `setLineWidth:` / `setTessellationFactorScale:`, value in
    /// [`Command::float_value`].
    SetFloatState,
    /// `setColorStoreAction:atIndex:` and the depth and stencil forms. The
    /// colour form carries an index; the other two have one attachment each and
    /// carry none.
    SetStoreAction,
    UseResource,
    UseHeap,
    ExecuteCommands,
    RenderPass,
    /// `drawPrimitives:indirectBuffer:` and its indexed sibling. Which of the
    /// two is [`Command::opcode`], as on the wire. The buffer holding the
    /// counts is [`Command::indirect_buffer_ref`] at
    /// [`Command::indirect_buffer_offset`]; the indexed form also fills
    /// [`Command::index_type`], [`Command::index_buffer_ref`] and
    /// [`Command::index_buffer_offset`].
    DrawIndirect,
    /// `setVisibilityResultMode:offset:`. Mode in [`Command::mode`], offset in
    /// [`Command::visibility_result_offset`].
    SetVisibilityResultMode,
    /// `setVertexAmplificationMode:value:` fills [`Command::mode`] and
    /// [`Command::amplification_value`]; `setVertexAmplificationCount:viewMappings:`
    /// fills [`Command::count`]. Which of the two is [`Command::opcode`], as on
    /// the wire. The view mappings are not lifted — see
    /// [`wire::OPCODE_SET_VERTEX_AMPLIFICATION_MODE`].
    SetVertexAmplification,
    /// A bind against the **tile** argument tables: `0x9d`/`0x9e` (buffer and
    /// offset), `0x9f`/`0xa0` (sampler, plain and LOD-bearing), `0xa1`
    /// (texture). Which of the five is [`Command::opcode`], as on the wire;
    /// the slots are [`Command::first`] and [`Command::count`].
    ///
    /// **Deliberately not `SetBuffer`/`SetTexture`/`SetSampler` with a third
    /// [`Stage`].** The records are those records byte for byte, so folding
    /// them in is the tempting shape — but every existing executor arm reads
    /// `Stage` as vertex-or-fragment, and a tile texture routed through one
    /// would bind into the *fragment* table. That is worse than dropping it:
    /// the guest's fragment shader would sample a texture it never bound. A
    /// kind nothing else matches cannot be mis-applied by an arm that has not
    /// been taught about tiles.
    TileBind,
    /// A tile-shader dispatch: `0x9b`, or `0xa2`/`0xa3` bounded to a region.
    /// Threads per tile in [`Command::tile_threads`].
    TileDispatch,
    /// `getTileDimensions:` (`0xa4`), the guest asking the host to report tile
    /// geometry into [`Command::buffer_ref`] at [`Command::buffer_offset`].
    TileDimensionsQuery,
    /// `set{Color,Depth,Stencil}StoreActionOptions:` (`0x67`/`0x6a`/`0x79`).
    /// Options in [`Command::mode`]; the colour form also fills
    /// [`Command::first`] with the attachment index. Distinct from
    /// [`Kind::SetStoreAction`], which is the action rather than its options
    /// and is a different record at a different width.
    SetStoreActionOptions,
    /// `setTessellationFactorBuffer:offset:instanceStride:` (`0x7a`), in
    /// [`Command::buffer_ref`] and [`Command::buffer_offset`].
    SetTessellationFactorBuffer,
    /// A tessellated draw: [`wire::OPCODE_DRAW_PATCHES`] and its four siblings. Which
    /// form is [`Command::opcode`], as on the wire — except that `0x0c` is two
    /// records and [`Command::command_length`] is what separates them.
    ///
    /// No field is lifted, because nothing here tessellates and a `patch_count`
    /// with no consumer is worse than its absence. The record is still fully
    /// bounds-checked, so a truncated one is refused rather than reported as a
    /// smaller draw.
    DrawPatches,
    /// One pass property `writeDescriptor` emits as a record of its own rather
    /// than as a field of the pass descriptor: `0x1e` the default raster sample
    /// count, `0x20` the programmable sample positions, `0x21` the
    /// rasterization rate map, `0x22`/`0x23` the imageblock and threadgroup
    /// memory lengths, `0x24` the tile size. Which one is [`Command::opcode`],
    /// as on the wire; the scalar is [`Command::mode`] and the rate map's ref is
    /// [`Command::object_ref`].
    ///
    /// Decoded and counted rather than applied. Each of the six is behind a
    /// capability that defaults off, so a non-zero count is the first evidence
    /// this project would have that any guest negotiates one — which is a thing
    /// nothing in this device currently observes.
    RenderPassProperty,
    OtherAccepted,
}

/// One color attachment from a render-pass descriptor (0x1a).
///
/// # Whether a slot is bound is `texture_ref != 0`, and only that
///
/// All three attachment shapes carried a `present: bool` beside `texture_ref`
/// that every decode path set to `texture_ref != 0` — the derived copy of a
/// field sitting next to the field it is derived from. The two could disagree
/// only by construction, and did: a bound-but-textureless attachment is not a
/// thing this decoder can produce, yet a caller could build one and a consumer
/// reading `present` would honour it. One call site had already written both
/// halves in one expression (`!att.present || att.texture_ref == 0`), which is
/// the same test twice.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ColorAttachment {
    pub texture_ref: u32,
    pub resolve_texture_ref: u32,
    pub level: u32,
    /// Array slice, sixteen bits on the wire directly above `level`.
    pub slice: u32,
    /// Depth plane of a 3D attachment, sixteen bits above `slice`.
    pub depth_plane: u32,
    pub load_action: u16,
    pub store_action: u16,
    /// `MTLClearColor` as RGBA doubles. The attachment pixel format decides
    /// whether those components are continuous values or integer counts; an
    /// integer clear of `1.0` means `1`, not the format's normalized maximum.
    pub clear_color: [f64; 4],
}

/// Depth attachment from a render-pass descriptor (slot @0x00).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DepthAttachment {
    pub texture_ref: u32,
    pub resolve_texture_ref: u32,
    pub level: u32,
    /// Array slice and depth plane, the two sixteen-bit fields above `level`.
    ///
    /// All three attachment shapes share one 28-byte prefix — that is what
    /// `reims_vgpu_wire::ops::render_pass::AttachmentPrefix` is — so these are
    /// here for the same reason they are on [`ColorAttachment`]: a depth buffer
    /// bound at slice 5 is as real as a colour target bound there, and a field
    /// nothing decodes is a field nothing can report.
    pub slice: u32,
    pub depth_plane: u32,
    pub load_action: u16,
    pub store_action: u16,
    pub clear_depth: f64,
}

/// A scissor rectangle in target texels, as `MTLScissorRect` declares one.
///
/// A type because the four numbers used to travel as four loose fields here, as
/// an `Option<(u32, u32, u32, u32)>` through two request structs, and as four
/// and then six adjacent `u32` parameters into the coverage census — where the
/// rect sat next to the target extent and every permutation of the six
/// compiled. `ScissorResource` and `MTLScissorRect` are the two backends' own
/// ABI shapes and stay; this is the one the decode produces and the device
/// carries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScissorRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl ScissorRect {
    /// Whether this rect reaches every texel of a `target_w` x `target_h`
    /// attachment. A draw that does could have written anywhere in it, so
    /// nothing downstream can bound its writes by the scissor.
    pub fn covers(&self, target_w: u32, target_h: u32) -> bool {
        self.x == 0 && self.y == 0 && self.width >= target_w && self.height >= target_h
    }

    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }
}

/// One entry of a multi-slot buffer bind record.
///
/// A struct rather than the `(u32, u64)` tuple this used to be, because the
/// stride is a third value travelling with the same two. The pair had already
/// outgrown the tuple: the
/// twenty-byte `setVertexBuffers:offsets:attributeStrides:withRange:` entry
/// carries all three, and the decoder was reading the first two and stepping
/// over the third.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecodedBufferBind {
    pub buffer_ref: u32,
    pub offset: u64,
    /// The vertex fetch stride this bind declares, overriding whatever the
    /// pipeline's `MTLVertexBufferLayoutDescriptor` said for the same index.
    ///
    /// `None` is "this record carried no stride table", which is the plain
    /// `setVertexBuffers:offsets:withRange:` and every non-vertex stage. It is
    /// not the same as `Some(0)`: a zero stride is a legal Metal request that
    /// fetches every vertex from the same address.
    pub attribute_stride: Option<u64>,
}

/// Lift one wire rect. Shared by the singular and plural scissor opcodes, which
/// carry the identical element and differ only in how many of it follow.
fn scissor_from_wire(r: &wire::ScissorRect) -> ScissorRect {
    ScissorRect {
        x: r.x.get() as u32,
        y: r.y.get() as u32,
        width: r.width.get() as u32,
        height: r.height.get() as u32,
    }
}

/// Lift one wire viewport, in the `[originX, originY, width, height, znear,
/// zfar]` order both backends read it back in. Shared by the singular and
/// plural viewport opcodes for the same reason as [`scissor_from_wire`].
fn viewport_from_wire(v: &wire::Viewport) -> [f64; 6] {
    [
        v.origin_x.get(),
        v.origin_y.get(),
        v.width.get(),
        v.height.get(),
        v.znear.get(),
        v.zfar.get(),
    ]
}

/// The subresource coordinates and resolve target shared by all three
/// attachment shapes, lifted so the arms that read them cannot drift apart.
///
/// They are one 28-byte prefix on the wire
/// ([`reims_vgpu_wire::ops::render_pass::AttachmentPrefix`]), and this device
/// had two arms reading it with two copies of the same four-line check. A
/// third copy is what the colour arm would have needed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AttachSubresource {
    pub level: u32,
    pub slice: u32,
    pub depth_plane: u32,
    pub resolve_texture_ref: u32,
}

impl From<DepthAttachment> for AttachSubresource {
    fn from(a: DepthAttachment) -> Self {
        Self {
            level: a.level,
            slice: a.slice,
            depth_plane: a.depth_plane,
            resolve_texture_ref: a.resolve_texture_ref,
        }
    }
}

impl From<StencilAttachment> for AttachSubresource {
    fn from(a: StencilAttachment) -> Self {
        Self {
            level: a.level,
            slice: a.slice,
            depth_plane: a.depth_plane,
            resolve_texture_ref: a.resolve_texture_ref,
        }
    }
}

impl From<ColorAttachment> for AttachSubresource {
    fn from(a: ColorAttachment) -> Self {
        Self {
            level: a.level,
            slice: a.slice,
            depth_plane: a.depth_plane,
            resolve_texture_ref: a.resolve_texture_ref,
        }
    }
}

/// Whether the arm asking [`attachment_subresource_is_bindable`] can render into
/// a mip level other than zero.
///
/// The arms genuinely differ, which is why this is a parameter rather than a
/// second predicate. The colour rail materializes a level's own plane inside the
/// guest allocation — `TextureDescriptor::level_gva` gives its address, stride
/// and geometry, and `render_target`'s linear rung has rendered into one since
/// type-8 mip views existed. The depth/stencil rail has no such rung: it would
/// bind level 0 and the guest would read a level it never wrote.
///
/// Making it an enum rather than a `bool` is the point — an arm has to say which
/// it is, and it cannot say it by accident.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LevelSupport {
    /// This arm renders into level 0 and nothing else.
    LevelZeroOnly,
    /// This arm resolves the named level's own plane.
    AnyLevel,
}

/// Whether this device can honour an attachment's subresource as decoded.
///
/// Slice 0, plane 0, no multisample resolve for callers using this predicate,
/// and a level the caller's rail can reach. `slice` and `depth_plane` joined the test when they became decodable:
/// a depth buffer bound at slice 5 was previously read as slice 0 and silently
/// accepted.
///
/// It lives beside the structs it reads because four arms apply it — the stream
/// decode that admits an attachment into a pass, once per aspect, and the Metal
/// rail that builds a host-side buffer for one. **Every hand-written copy of it
/// that has existed was missing a term.** The rail's tested `level` and
/// `resolve_texture_ref` only, so the two `u16` fields above `level` were
/// checked in one place and not the other. The colour arm's tested `level`,
/// `slice` and `depth_plane` and not `resolve_texture_ref`, so a multisample
/// colour pass — the attachment multisampled, `storeAction =
/// MultisampleResolve`, `resolveTexture` naming where the single-sampled result
/// goes — was admitted, rendered at one sample into the attachment, and left
/// the resolve target the guest goes on to read holding whatever it held.
///
/// That is why it takes [`AttachSubresource`] rather than any one attachment
/// type: a fifth arm gets the whole rule or does not compile.
pub fn attachment_subresource_is_bindable(s: AttachSubresource, levels: LevelSupport) -> bool {
    let level_ok = match levels {
        LevelSupport::LevelZeroOnly => s.level == 0,
        LevelSupport::AnyLevel => true,
    };
    level_ok && s.slice == 0 && s.depth_plane == 0 && s.resolve_texture_ref == 0
}

/// Whether a colour attachment's directly-addressed coordinates are bindable.
///
/// A resolve texture is a second attachment and an end-of-pass operation, not
/// a coordinate of the multisample source. Keep it intact so the backend can
/// encode or precisely refuse that operation. Depth and stencil continue to
/// use [`attachment_subresource_is_bindable`] because their backend request
/// types do not yet carry resolve destinations.
pub fn color_attachment_subresource_is_bindable(s: AttachSubresource) -> bool {
    s.slice == 0 && s.depth_plane == 0
}

/// Stencil attachment from a render-pass descriptor (slot @0x28).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StencilAttachment {
    pub texture_ref: u32,
    pub resolve_texture_ref: u32,
    pub level: u32,
    /// See [`DepthAttachment::slice`]; the prefix is the same 28 bytes.
    pub slice: u32,
    pub depth_plane: u32,
    pub load_action: u16,
    pub store_action: u16,
    pub clear_stencil: u32,
}

/// Which encoder table a render bind record names.
///
/// Derived from the opcode, not from a wire field:
/// `wire::OPCODE_SET_VERTEX_*` versus `wire::OPCODE_SET_FRAGMENT_*`. The render
/// opcode set expresses no other stage, so there are no other variants — an
/// object/mesh/tile bind reaches the device through the indirect-command-buffer
/// path and carries [`crate::runtime::icb::IcbRenderBindStage`], which is a
/// different vocabulary with a different wire encoding.
///
/// Keeping this exhaustive is the point. With unreachable variants present,
/// every `match` over it needed a catch-all, and a catch-all is what would
/// swallow a genuinely new stage in silence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Stage {
    /// The record named no stage, or the opcode was not a stage-bearing one.
    #[default]
    Unknown,
    Vertex,
    Fragment,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Command {
    pub opcode: u32,
    pub command_length: u32,
    pub kind: Kind,
    pub stage: Stage,
    pub pipeline_ref: u32,
    pub first: u32,
    pub count: u32,
    pub buffer_ref: u32,
    pub buffer_offset: u64,
    /// Multi-entry buffer binds for slots `first..first+count`.
    pub buffer_binds: Vec<DecodedBufferBind>,
    pub texture_ref: u32,
    /// Multi-entry texture/sampler refs for slots first..first+count.
    pub ref_binds: Vec<u32>,
    /// `(lodMinClamp, lodMaxClamp)` as raw `f32` bits, one per entry of
    /// [`Self::ref_binds`], for the two sampler-bind opcodes that carry them.
    /// Empty for every other record, including the plain sampler bind — an
    /// empty list is "the record declared no clamps", which is not the same as
    /// a pair that happens to be `(0.0, 0.0)`.
    pub sampler_lod_binds: Vec<(u32, u32)>,
    pub sampler_ref: u32,
    pub primitive_type: u32,
    pub vertex_start: u32,
    pub vertex_count: u32,
    pub instance_count: u32,
    pub index_count: u32,
    pub index_type: u32,
    pub index_buffer_ref: u32,
    pub index_buffer_offset: u64,
    /// Metal `baseInstance` / Vulkan `firstInstance`. Zero on the draw forms
    /// whose selector has no such argument, which is what Metal defaults to.
    pub base_instance: u32,
    /// Metal `baseVertex` / Vulkan `vertexOffset`, on the indexed forms that
    /// carry one. Signed: Metal declares it `NSInteger`, Apple's serializer
    /// declares it `q` where every count beside it is `Q`, and a negative
    /// offset read as unsigned becomes a large index rather than an error.
    pub base_vertex: i64,
    /// Every viewport a [`Kind::SetViewport`] record carried, in the guest's
    /// order — the singular opcode's one, or all of `setViewports:count:`.
    ///
    /// A `Vec` rather than one entry plus a count, because the two forms are the
    /// same record with a different length and the count used to be kept only to
    /// say how many entries were being thrown away. Empty for every other kind.
    pub viewports: Vec<[f64; 6]>,
    /// Every scissor rect a [`Kind::SetScissor`] record carried, in the guest's
    /// order. See [`Self::viewports`].
    pub scissors: Vec<ScissorRect>,
    pub fence_ref: u32,
    /// Value of a [`Kind::SetRasterState`], [`Kind::SetStoreAction`] or
    /// [`Kind::SetVisibilityResultMode`] record.
    pub mode: u64,
    /// Buffer a [`Kind::DrawIndirect`] record reads its counts from, and the
    /// byte offset of the arguments structure within it. Distinct from
    /// [`Command::index_buffer_ref`], which the indexed form fills as well —
    /// an indexed indirect draw names two buffers and two offsets, and a
    /// decoder that crossed them would replay the wrong one.
    pub indirect_buffer_ref: u32,
    pub indirect_buffer_offset: u64,
    /// Byte offset a [`Kind::SetVisibilityResultMode`] record writes its
    /// occlusion counter to, within the pass's visibility result buffer.
    pub visibility_result_offset: u64,
    /// `value` of a [`Kind::SetVertexAmplification`] mode record. Thirty-two
    /// bits: the selector declares it `Q` and the serializer narrows it, which
    /// only the capture shows.
    pub amplification_value: u32,
    /// Threads per tile of a [`Kind::TileDispatch`], as width/height/depth.
    ///
    /// Unnarrowed `u64` — the serializer writes all three at full width, unlike
    /// almost every other count in this protocol. Read by `runtime::exec` both
    /// to tell an empty dispatch from a real one and to say in the fail line
    /// how much work was dropped.
    pub tile_threads: [u64; 3],
    /// Value of a [`Kind::SetFloatState`] record.
    pub float_value: f32,
    /// The sampler bind carried per-entry LOD clamps this decoder did not lift.
    /// Only ever true on [`Kind::SetSampler`]; see [`wire::OPCODE_SET_VERTEX_SAMPLER_LOD`].
    pub has_sampler_lod: bool,
    /// The vertex buffer bind carried a per-entry attribute stride this decoder
    /// did not lift. True on [`Kind::SetBuffer`] and [`Kind::SetBufferOffset`];
    /// see [`wire::OPCODE_SET_VERTEX_BUFFER_STRIDE`]. The buffer still binds — what is
    /// missing is the stride the guest wanted the vertex fetch to use.
    pub has_attribute_stride: bool,
    /// The stride of a single-slot [`Kind::SetBufferOffset`] record that
    /// carried one (`setVertexBufferOffset:attributeStride:atIndex:`). The
    /// multi-slot forms carry theirs per entry in
    /// [`DecodedBufferBind::attribute_stride`], because the record does.
    pub attribute_stride: Option<u64>,
    pub raw_payload_len: usize,
    /// Color attachment[0] when kind is RenderPass (boot clear path).
    pub color0: ColorAttachment,
    pub depth: DepthAttachment,
    pub stencil: StencilAttachment,
    /// The pass's own tail, on [`Kind::RenderPass`]. Decoded, not applied.
    ///
    /// `visibility_result_buffer_ref` is the buffer
    /// `setVisibilityResultMode:offset:` indexes — that record carries the mode
    /// and the offset and *only* the pass record names the buffer, so the two
    /// halves of an occlusion query arrive on different records. The three
    /// geometry fields are the guest's explicit statement about the pass extent
    /// and layer count, which cannot be recovered from the attachments: a guest
    /// may bind a 4096-wide texture and ask for a 640-wide pass.
    pub pass_visibility_result_buffer_ref: u32,
    pub pass_render_target_array_length: u64,
    pub pass_render_target_width: u64,
    pub pass_render_target_height: u64,
    /// setBlendColor RGBA floats (when kind is SetBlendColor).
    pub blend_color: [f32; 4],
    /// setCullMode
    pub cull_mode: u32,
    /// setFrontFacingWinding
    pub front_facing: u32,
    /// setDepthBias (depthBias, slopeScale, clamp) as f32.
    pub depth_bias: [f32; 3],
    /// setDepthStencilState object ref
    pub depth_stencil_ref: u32,
    /// setStencilReference front/back
    pub stencil_ref_front: u32,
    pub stencil_ref_back: u32,
    /// `0x14`/`0x15` executeCommandsInBuffer.
    pub indirect_command_buffer_ref: u32,
    /// `0x15` range form (unaligned after ICB ref).
    pub icb_range_location: u64,
    pub icb_range_length: u64,
    /// `0x14` indirect range buffer form.
    pub icb_args_buffer_ref: u32,
    pub icb_args_buffer_offset: u64,
    /// True when kind is ExecuteCommands with the range layout (`0x15`).
    pub icb_is_range: bool,
}

/// Whether an opcode is above every one Apple's serializer writes here.
///
/// Named for what it measures. Its predecessor was `opcode_is_apple_rejected`,
/// which asserted the serializer would never emit anything above the window --
/// and it emitted `0xa5` and `0xa6`, so this device refused four vertex binds as
/// records Apple does not produce. That is the same correction
/// `decode::blit::opcode_unimplemented_here` needed, and the same lesson: the
/// highest opcode *this project has driven* is not the highest Apple writes.
///
/// The bound comes from [`wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE`], which is now derived from
/// `reims_vgpu_wire`'s manifest rather than from observation.
/// An opcode inside the accepted window that no decode arm claims.
///
/// Two tests need one -- this module's catch-all test and `runtime::exec`'s
/// fail-visible test -- and both used to hardcode it. Both went stale, twice:
/// `wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE` stopped working when that bound was corrected to `0xa6`,
/// and its replacement `0x99` lasted one commit until
/// `setVertexAmplificationMode:value:` turned out to be exactly that number.
/// Searching keeps them honest as arms are added, because what they test is
/// that the catch-all exists and reports, not that any number is in it.
#[cfg(test)]
pub(crate) fn unclaimed_accepted_opcode() -> u32 {
    (0..=wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE)
        .find(|&op| {
            let mut v = vec![0u8; OP_HEADER_LEN];
            crate::contract::endian::st32(&mut v[0..4], op);
            crate::contract::endian::st32(&mut v[4..8], OP_HEADER_LEN as u32);
            matches!(decode(&v), Ok(c) if c.kind == Kind::OtherAccepted)
        })
        .expect("every opcode in the window is decoded; the catch-all is unreachable")
}

pub fn opcode_above_the_encoder_window(opcode: u32) -> bool {
    opcode > wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE
}

/// The full accepted window, from `reims_vgpu_render_decode.h`'s enum range.
///
/// One comparison, in [`opcode_above_the_encoder_window`]. This used to end with
/// `opcode <= wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE` after the early
/// return above it — the exact negation of the test just made, so it could not
/// be false, while reading like a second admission rule that a later edit would
/// have to keep in step.
pub fn opcode_supported(opcode: u32) -> bool {
    !opcode_above_the_encoder_window(opcode)
}

/// Decode color attachment slot `index` from a render-pass payload.
///
/// # `level` is sixteen bits and `slice` is the sixteen above it
///
/// This read was `ld32` at `PASS_ATTACH_LEVEL` for as long as the function has
/// existed, with a comment on [`decode_depth_attachment`] stating the rule as
/// "the archive uses u16 for depth/stencil level (color uses u32)". Nothing had
/// ever checked that; Apple's own bytes say all three shapes are identical here,
/// and the four bytes at `+0x08` are `level` then `slice`.
///
/// So a pass rendering into array slice 1 — a cube face, a texture-array layer,
/// a layered shadow map — reported mip level 65536 and lost its slice entirely.
/// Both are decoded now.
fn color_from_wire(c: &wire_pass::ColorAttachmentBody) -> ColorAttachment {
    let p = &c.prefix;
    ColorAttachment {
        texture_ref: p.texture_ref.get(),
        resolve_texture_ref: p.resolve_texture_ref.get(),
        level: u32::from(p.level.get()),
        slice: u32::from(p.slice.get()),
        depth_plane: u32::from(p.depth_plane.get()),
        load_action: p.load_action.get(),
        store_action: p.store_action.get(),
        clear_color: c.clear_color(),
    }
}

fn depth_from_wire(d: &wire_pass::DepthAttachmentBody) -> DepthAttachment {
    let p = &d.prefix;
    DepthAttachment {
        texture_ref: p.texture_ref.get(),
        resolve_texture_ref: p.resolve_texture_ref.get(),
        level: u32::from(p.level.get()),
        slice: u32::from(p.slice.get()),
        depth_plane: u32::from(p.depth_plane.get()),
        load_action: p.load_action.get(),
        store_action: p.store_action.get(),
        clear_depth: d.clear_depth(),
    }
}

fn stencil_from_wire(s: &wire_pass::StencilAttachmentBody) -> StencilAttachment {
    let p = &s.prefix;
    StencilAttachment {
        texture_ref: p.texture_ref.get(),
        resolve_texture_ref: p.resolve_texture_ref.get(),
        level: u32::from(p.level.get()),
        slice: u32::from(p.slice.get()),
        depth_plane: u32::from(p.depth_plane.get()),
        load_action: p.load_action.get(),
        store_action: p.store_action.get(),
        clear_stencil: s.clear_stencil.get(),
    }
}

pub fn decode_color_attachment(payload: &[u8], index: usize) -> ColorAttachment {
    let base = PASS_COLOR_ATTACH_OFF + index * PASS_COLOR_ATTACH_STRIDE;
    match reims_vgpu_wire::view_at::<wire_pass::ColorAttachmentBody>(payload, base) {
        Ok(c) => color_from_wire(c),
        Err(_) => ColorAttachment::default(),
    }
}

/// Decode the depth attachment (fixed slot @0).
pub fn decode_depth_attachment(payload: &[u8]) -> DepthAttachment {
    match reims_vgpu_wire::view::<wire_pass::DepthAttachmentBody>(payload) {
        Ok(d) if payload.len() >= PASS_STENCIL_ATTACH_OFF => depth_from_wire(d),
        _ => DepthAttachment::default(),
    }
}

/// Decode the stencil attachment (fixed slot after depth).
pub fn decode_stencil_attachment(payload: &[u8]) -> StencilAttachment {
    match reims_vgpu_wire::view_at::<wire_pass::StencilAttachmentBody>(
        payload,
        PASS_STENCIL_ATTACH_OFF,
    ) {
        Ok(s) => stencil_from_wire(s),
        Err(_) => StencilAttachment::default(),
    }
}

/// Transactional render command decode.
pub fn decode(command: &[u8]) -> Result<Command, DecodeStatus> {
    let op = reims_vgpu_wire::op(command, 0).map_err(|_| DecodeStatus::ErrShort)?;
    let opcode = op.opcode();
    let command_length = op.length() as usize;
    if opcode_above_the_encoder_window(opcode) {
        return Err(DecodeStatus::ErrUnsupportedOpcode);
    }
    if !opcode_supported(opcode) {
        return Err(DecodeStatus::ErrUnknownOpcode);
    }
    let payload = op.payload;
    let mut out = Command {
        opcode,
        command_length: command_length as u32,
        raw_payload_len: payload.len(),
        ..Default::default()
    };

    match opcode {
        wire::OPCODE_SET_RENDER_PIPELINE_STATE => {
            let r = wire::state_ref(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetPipeline;
            out.pipeline_ref = r.object_ref.get();
            Ok(out)
        }
        wire::OPCODE_SET_VERTEX_BUFFER | wire::OPCODE_SET_FRAGMENT_BUFFER => {
            let (head, entries) = wire::buffer_binds(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetBuffer;
            out.stage = if opcode == wire::OPCODE_SET_FRAGMENT_BUFFER {
                Stage::Fragment
            } else {
                Stage::Vertex
            };
            out.first = head.first.get();
            out.count = head.count.get();
            if out.count == 0 {
                return Err(DecodeStatus::ErrBadLength);
            }
            // Exact length: product refuses slack the guest did not size for.
            match bind_record_len(out.count, BUFFER_BIND_ENTRY_SIZE) {
                Some(need) if payload.len() >= need => {}
                _ => return Err(DecodeStatus::ErrShort),
            }
            out.buffer_binds.clear();
            for e in entries {
                out.buffer_binds.push(DecodedBufferBind {
                    buffer_ref: e.buffer_ref.get(),
                    offset: e.offset.get(),
                    attribute_stride: None,
                });
            }
            if let Some(b) = out.buffer_binds.first() {
                out.buffer_ref = b.buffer_ref;
                out.buffer_offset = b.offset;
            }
            Ok(out)
        }
        wire::OPCODE_SET_VERTEX_BUFFER_STRIDE => {
            // Attribute-stride form: twenty-byte entries, all three fields of
            // each lifted.
            let (head, entries) =
                wire::buffer_stride_binds(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetBuffer;
            out.stage = Stage::Vertex;
            out.has_attribute_stride = true;
            out.first = head.first.get();
            out.count = head.count.get();
            if out.count == 0 {
                return Err(DecodeStatus::ErrBadLength);
            }
            match bind_record_len(out.count, BUFFER_STRIDE_BIND_ENTRY_SIZE) {
                Some(need) if payload.len() >= need => {}
                _ => return Err(DecodeStatus::ErrShort),
            }
            out.buffer_binds.clear();
            for e in entries {
                out.buffer_binds.push(DecodedBufferBind {
                    buffer_ref: e.buffer_ref.get(),
                    offset: e.offset.get(),
                    attribute_stride: Some(e.attribute_stride.get()),
                });
            }
            if let Some(b) = out.buffer_binds.first() {
                out.buffer_ref = b.buffer_ref;
                out.buffer_offset = b.offset;
            }
            Ok(out)
        }
        wire::OPCODE_SET_VERTEX_TEXTURE
        | wire::OPCODE_SET_FRAGMENT_TEXTURE
        | wire::OPCODE_SET_VERTEX_SAMPLER
        | wire::OPCODE_SET_FRAGMENT_SAMPLER => {
            let (head, entries) = wire::ref_binds(&op).map_err(|_| DecodeStatus::ErrShort)?;
            let textures = opcode == wire::OPCODE_SET_VERTEX_TEXTURE
                || opcode == wire::OPCODE_SET_FRAGMENT_TEXTURE;
            out.kind = if textures {
                Kind::SetTexture
            } else {
                Kind::SetSampler
            };
            out.stage = if opcode == wire::OPCODE_SET_VERTEX_TEXTURE
                || opcode == wire::OPCODE_SET_VERTEX_SAMPLER
            {
                Stage::Vertex
            } else {
                Stage::Fragment
            };
            out.first = head.first.get();
            out.count = head.count.get();
            if out.count == 0 {
                return Err(DecodeStatus::ErrBadLength);
            }
            match bind_record_len(out.count, REF_BIND_ENTRY_SIZE) {
                Some(need) if payload.len() >= need => {}
                _ => return Err(DecodeStatus::ErrShort),
            }
            out.ref_binds.clear();
            for e in entries {
                out.ref_binds.push(e.object_ref.get());
            }
            if let Some(&r) = out.ref_binds.first() {
                if textures {
                    out.texture_ref = r;
                } else {
                    out.sampler_ref = r;
                }
            }
            Ok(out)
        }
        wire::OPCODE_SET_VERTEX_SAMPLER_LOD | wire::OPCODE_SET_FRAGMENT_SAMPLER_LOD => {
            // Both halves are lifted: the refs into `ref_binds` and the
            // per-entry clamps into `sampler_lod_binds` beside them.
            let (head, entries) =
                wire::sampler_lod_binds(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetSampler;
            out.stage = if opcode == wire::OPCODE_SET_VERTEX_SAMPLER_LOD {
                Stage::Vertex
            } else {
                Stage::Fragment
            };
            out.has_sampler_lod = true;
            out.first = head.first.get();
            out.count = head.count.get();
            if out.count == 0 {
                return Err(DecodeStatus::ErrBadLength);
            }
            match bind_record_len(out.count, SAMPLER_LOD_BIND_ENTRY_SIZE) {
                Some(need) if payload.len() >= need => {}
                _ => return Err(DecodeStatus::ErrShort),
            }
            out.ref_binds.clear();
            out.sampler_lod_binds.clear();
            for e in entries {
                out.ref_binds.push(e.sampler_ref.get());
                // Carried as bits. The clamps are per *entry* — the wire doc
                // says so and `render_set_vertex_samplers_lod_range` binds two
                // slots with four distinct values — so they are pushed beside
                // the refs rather than lifted to a per-record pair.
                out.sampler_lod_binds.push((
                    e.lod_min_clamp.get().to_bits(),
                    e.lod_max_clamp.get().to_bits(),
                ));
            }
            if let Some(&r) = out.ref_binds.first() {
                out.sampler_ref = r;
            }
            Ok(out)
        }
        wire::OPCODE_DRAW => {
            // Compact `drawPrimitives:vertexStart:vertexCount:`: an 8-byte
            // payload of `u32 primitiveType · u16 vertexStart · u16 vertexCount`.
            //
            // This used to read four u32s behind `payload.len() < 16`, which is
            // neither of the selector's two forms. The only test for it was a
            // synthetic 24-byte fixture built to match the code, so nothing
            // caught it — and every live compact draw was rejected `ErrShort`
            // and dropped. Silently, until the decode refusal was named: one
            // fired on the first arm64 boot that could report it. The layout is
            // now `reims_vgpu_wire::ops::render::Draw`, pinned by fixtures
            // `render_draw_primitives` and `render_draw_primitives_strip`.
            if command_length != DRAW_COMPACT_CMD_LEN {
                return Err(DecodeStatus::ErrBadLength);
            }
            let d = wire::draw(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::Draw;
            out.primitive_type = d.primitive_type.get();
            out.vertex_start = d.vertex_start.get() as u32;
            out.vertex_count = d.vertex_count.get() as u32;
            // Not on the wire: this selector is the non-instanced one, and
            // Metal draws it once.
            out.instance_count = 1;
            Ok(out)
        }
        // Wide `drawPrimitives:vertexStart:vertexCount:`, which the guest emits
        // instead of `0x01` when either count exceeds 16 bits.
        //
        // This arm used to decline by name, and it was right to: the layout it
        // would have guessed — `u64 · u64 · u32 primitiveType@0x10`, by analogy
        // with the wide instanced siblings — is wrong. `primitiveType` leads and
        // is 32-bit, exactly as in the compact form. Fixtures
        // `render_draw_primitives_wide`, `..._count_over_16bit` and
        // `..._start_over_16bit` settle it.
        wire::OPCODE_DRAW_WIDE => {
            if command_length != wire::DRAW_WIDE_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let d = wire::draw_wide(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::Draw;
            out.primitive_type = d.primitive_type.get();
            out.vertex_start = narrow_count(d.vertex_start.get())?;
            out.vertex_count = narrow_count(d.vertex_count.get())?;
            out.instance_count = 1;
            Ok(out)
        }
        // Compact `drawPrimitives:vertexStart:vertexCount:instanceCount:`.
        //
        // The layout is DISTINCT from the `0x01` form — the counts lead and
        // `primitiveType` is last and 16-bit. Derived here from live x86 WebKit
        // bytes (`00000400 0d000400` = vs0 vc4 inst13 primTriStrip) before the
        // oracle existed, and `render_draw_primitives_instanced` later agreed
        // field for field. This is WebKit's instanced glyph/rect batch; the
        // non-instanced `0x01` and indexed `0x07` forms render chrome text,
        // which is why chrome rendered while page content stayed blank.
        wire::OPCODE_DRAW_INSTANCED => {
            let d = wire::draw_instanced(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::Draw;
            out.vertex_start = d.vertex_start.get() as u32;
            out.vertex_count = d.vertex_count.get() as u32;
            out.instance_count = wire_instance_count(d.instance_count.get() as u64)?;
            out.primitive_type = d.primitive_type.get() as u32;
            Ok(out)
        }
        // Wide `drawPrimitives:…:instanceCount:` — the counts widen together
        // when any one of them passes 16 bits, so the whole record is 64-bit
        // even where two of the three would have fitted.
        wire::OPCODE_DRAW_INSTANCED_WIDE => {
            if command_length != wire::DRAW_INSTANCED_WIDE_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let d = wire::draw_instanced_wide(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::Draw;
            out.vertex_start = narrow_count(d.vertex_start.get())?;
            out.vertex_count = narrow_count(d.vertex_count.get())?;
            out.instance_count = wire_instance_count(d.instance_count.get())?;
            out.primitive_type = d.primitive_type.get() as u32;
            Ok(out)
        }
        // `drawPrimitives:…:instanceCount:baseInstance:`, both encodings.
        //
        // Neither was decoded at all until now: both fall inside the accepted
        // window, so they reached `Kind::OtherAccepted` and executed nothing —
        // an entire Metal draw selector dropped, wearing the shape of an
        // accepted state-set.
        wire::OPCODE_DRAW_INSTANCED_BASE => {
            if command_length != wire::DRAW_INSTANCED_BASE_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let d = wire::draw_instanced_base(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::Draw;
            out.vertex_start = d.vertex_start.get() as u32;
            out.vertex_count = d.vertex_count.get() as u32;
            out.instance_count = wire_instance_count(d.instance_count.get() as u64)?;
            out.base_instance = d.base_instance.get() as u32;
            out.primitive_type = d.primitive_type.get() as u32;
            Ok(out)
        }
        wire::OPCODE_DRAW_INSTANCED_BASE_WIDE => {
            if command_length != wire::DRAW_INSTANCED_BASE_WIDE_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let d = wire::draw_instanced_base_wide(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::Draw;
            out.vertex_start = narrow_count(d.vertex_start.get())?;
            out.vertex_count = narrow_count(d.vertex_count.get())?;
            out.instance_count = wire_instance_count(d.instance_count.get())?;
            out.base_instance = narrow_count(d.base_instance.get())?;
            out.primitive_type = d.primitive_type.get() as u32;
            Ok(out)
        }
        wire::OPCODE_DRAW_INDEXED_WIDE => {
            // The wide indexed form. This arm's head was already right; what it
            // called `u32 indexCount@8, u32 pad@0xc` and `u32
            // indexBufferOffset@0x10, u32 pad@0x14` are the two halves of two
            // 64-bit fields, which reads the same below 2³² and differently
            // above it. Fixtures `render_draw_indexed_count_over_16bit` and
            // `..._offset_over_16bit`.
            if command_length != wire::DRAW_INDEXED_WIDE_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let d = wire::draw_indexed_wide(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::Draw;
            out.primitive_type = d.primitive_type.get() as u32;
            out.index_type = d.index_type.get() as u32;
            out.index_buffer_ref = d.index_buffer_ref.get();
            out.index_count = narrow_count(d.index_count.get())?;
            out.index_buffer_offset = d.index_buffer_offset.get();
            out.instance_count = 1;
            Ok(out)
        }
        // Compact indexed draws: `0x07` is 20 bytes on the wire and `0x09` is
        // 24, the second appending a 16-bit instance count.
        //
        // Two things here were wrong and could not be seen from a boot. The
        // first four bytes were read as one `u32 primitiveType`, which absorbs
        // `indexType` at `+2`; and `index_type` was then hardcoded to
        // `MTLIndexTypeUInt16`. Both are right exactly while the guest uses
        // 16-bit indices, because that ordinal is 0. Fixture
        // `render_draw_indexed_uint32` is the case that separates them: with
        // `MTLIndexTypeUInt32` the word reads `04 00 01 00`, so the old arm
        // produced `primitiveType = 0x10004` — no such Metal primitive —
        // alongside a 32-bit index buffer drawn as 16-bit.
        //
        // The `payload.len() >= 28` branch that used to sit in front of this is
        // gone. These two records are 12 and 16 bytes of payload and never
        // anything else; the wide forms are separate opcodes (`0x06`, `0x08`).
        // It was unreachable, and the layout it carried was an invention.
        wire::OPCODE_DRAW_INDEXED | wire::OPCODE_DRAW_INDEXED_INSTANCED => {
            out.kind = Kind::Draw;
            // Shared compact body; the instanced form only adds instance_count.
            // Use the layout view (not draw_indexed's opcode assert) so both arms share it.
            let d = wire_view::<wire::DrawIndexed>(payload)?;
            out.primitive_type = d.primitive_type.get() as u32;
            out.index_type = d.index_type.get() as u32;
            out.index_buffer_ref = d.index_buffer_ref.get();
            out.index_count = d.index_count.get() as u32;
            out.index_buffer_offset = d.index_buffer_offset.get() as u64;
            out.instance_count = if opcode == wire::OPCODE_DRAW_INDEXED_INSTANCED {
                let i = wire::draw_indexed_instanced(&op).map_err(|_| DecodeStatus::ErrShort)?;
                wire_instance_count(i.instance_count.get() as u64)?
            } else {
                1
            };
            Ok(out)
        }
        wire::OPCODE_DRAW_INDEXED_INSTANCED_WIDE => {
            if command_length != wire::DRAW_INDEXED_INSTANCED_WIDE_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let d = wire::draw_indexed_instanced_wide(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::Draw;
            out.primitive_type = d.primitive_type.get() as u32;
            out.index_type = d.index_type.get() as u32;
            out.index_buffer_ref = d.index_buffer_ref.get();
            out.index_count = narrow_count(d.index_count.get())?;
            out.index_buffer_offset = d.index_buffer_offset.get();
            out.instance_count = wire_instance_count(d.instance_count.get())?;
            Ok(out)
        }
        // The full indexed draw, with a base vertex and a base instance.
        //
        // These two are the *only* records in the family that put the buffer
        // offset before the index count. Reading them with the siblings' order
        // swaps a guest's index count and its buffer offset, which draws from
        // the wrong place in the wrong amount — so the field order here is not
        // a copy of the arm above and must not be made one.
        wire::OPCODE_DRAW_INDEXED_INSTANCED_BASE => {
            if command_length != wire::DRAW_INDEXED_INSTANCED_BASE_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let d = wire::draw_indexed_instanced_base(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::Draw;
            out.primitive_type = d.primitive_type.get() as u32;
            out.index_type = d.index_type.get() as u32;
            out.index_buffer_ref = d.index_buffer_ref.get();
            out.index_count = d.index_count.get() as u32;
            out.index_buffer_offset = d.index_buffer_offset.get() as u64;
            out.instance_count = wire_instance_count(d.instance_count.get() as u64)?;
            out.base_instance = d.base_instance.get() as u32;
            out.base_vertex = d.base_vertex.get() as i64;
            Ok(out)
        }
        wire::OPCODE_DRAW_INDEXED_INSTANCED_BASE_WIDE => {
            if command_length != wire::DRAW_INDEXED_INSTANCED_BASE_WIDE_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let d =
                wire::draw_indexed_instanced_base_wide(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::Draw;
            out.primitive_type = d.primitive_type.get() as u32;
            out.index_type = d.index_type.get() as u32;
            out.index_buffer_ref = d.index_buffer_ref.get();
            out.index_count = narrow_count(d.index_count.get())?;
            out.index_buffer_offset = d.index_buffer_offset.get();
            out.instance_count = wire_instance_count(d.instance_count.get())?;
            out.base_instance = narrow_count(d.base_instance.get())?;
            out.base_vertex = d.base_vertex.get();
            Ok(out)
        }
        wire::OPCODE_SET_VERTEX_BUFFER_OFFSET | wire::OPCODE_SET_FRAGMENT_BUFFER_OFFSET => {
            let b = wire::buffer_offset(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetBufferOffset;
            out.stage = if opcode == wire::OPCODE_SET_FRAGMENT_BUFFER_OFFSET {
                Stage::Fragment
            } else {
                Stage::Vertex
            };
            out.first = b.index.get();
            out.buffer_offset = b.offset.get();
            Ok(out)
        }
        wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE => {
            let b = wire::buffer_offset_stride(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetBufferOffset;
            out.stage = Stage::Vertex;
            out.has_attribute_stride = true;
            out.first = b.index.get();
            out.buffer_offset = b.offset.get();
            out.attribute_stride = Some(b.attribute_stride.get());
            Ok(out)
        }
        wire::OPCODE_SET_VIEWPORT => {
            let v = wire::set_viewport(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetViewport;
            out.count = 1;
            out.viewports = vec![viewport_from_wire(v)];
            Ok(out)
        }
        wire::OPCODE_SET_VIEWPORTS => {
            // The whole array is lifted. `count` is the record's own, and the
            // slice the wire view returns is already that long, so the two
            // cannot disagree about how many the guest set.
            let (head, ports) = wire::set_viewports(&op).map_err(|_| DecodeStatus::ErrShort)?;
            let count = head.count.get();
            if count == 0 {
                return Err(DecodeStatus::ErrBadLength);
            }
            out.kind = Kind::SetViewport;
            out.count = count;
            out.viewports = ports.iter().map(viewport_from_wire).collect();
            Ok(out)
        }
        wire::OPCODE_SET_SCISSOR => {
            let r = wire::set_scissor(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetScissor;
            out.count = 1;
            out.scissors = vec![scissor_from_wire(r)];
            Ok(out)
        }
        wire::OPCODE_SET_SCISSOR_RECTS => {
            let (head, rects) = wire::set_scissor_rects(&op).map_err(|_| DecodeStatus::ErrShort)?;
            let count = head.count.get();
            if count == 0 {
                return Err(DecodeStatus::ErrBadLength);
            }
            let Ok(count) = u32::try_from(count) else {
                return Err(DecodeStatus::ErrCountOutOfRange);
            };
            out.kind = Kind::SetScissor;
            out.count = count;
            out.scissors = rects.iter().map(scissor_from_wire).collect();
            Ok(out)
        }
        wire::OPCODE_SET_BLEND_COLOR => {
            let b = wire::set_blend_color(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetBlendColor;
            out.blend_color = [b.red.get(), b.green.get(), b.blue.get(), b.alpha.get()];
            Ok(out)
        }
        wire::OPCODE_SET_CULL_MODE => {
            let m = wire::set_cull_mode(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetCullMode;
            out.cull_mode = m.mode.get() as u32;
            Ok(out)
        }
        wire::OPCODE_SET_FRONT_FACING => {
            let m = wire::set_front_facing(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetFrontFacing;
            out.front_facing = m.mode.get() as u32;
            Ok(out)
        }
        wire::OPCODE_SET_DEPTH_BIAS => {
            let d = wire::set_depth_bias(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetDepthBias;
            out.depth_bias = [d.bias.get(), d.slope_scale.get(), d.clamp.get()];
            Ok(out)
        }
        wire::OPCODE_SET_DEPTH_STENCIL_STATE => {
            let r = wire::state_ref(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetDepthStencil;
            out.depth_stencil_ref = r.object_ref.get();
            Ok(out)
        }
        wire::OPCODE_SET_STENCIL_REFERENCE => {
            let s = wire::set_stencil_reference(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetStencilReference;
            out.stencil_ref_front = s.front.get();
            out.stencil_ref_back = s.back.get();
            Ok(out)
        }
        wire::OPCODE_UPDATE_FENCE | wire::OPCODE_WAIT_FOR_FENCE => {
            let f = wire::fence(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::Fence;
            out.fence_ref = f.fence_ref.get();
            Ok(out)
        }
        wire::OPCODE_USE_RESOURCE => {
            // Refs are not lifted (exec no-ops residency); count bounds the
            // record via the wire layout (usage+stages pack to 4 bytes, refs
            // at +8).
            let (head, refs) = wire::use_resource(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::UseResource;
            out.count = head.count.get();
            if out.count as usize != refs.len() {
                return Err(DecodeStatus::ErrShort);
            }
            Ok(out)
        }
        wire::OPCODE_USE_HEAP => {
            // Heap form: no usage word, stages u16, refs at +6 (align-1).
            let (head, refs) = wire::use_heap(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::UseHeap;
            out.count = head.count.get();
            if out.count as usize != refs.len() {
                return Err(DecodeStatus::ErrShort);
            }
            Ok(out)
        }
        // The two residency forms that take no `stages:`. A render encoder
        // inherits them from the encoder base class, so they arrive on this rail
        // as readily as the two above; without these arms they reached the
        // `OtherAccepted` catch-all and were reported as unimplemented opcodes,
        // which is a wrong answer twice over — they are implemented, by doing
        // nothing, and `render_noop_residency_hint` was counting half its family.
        //
        // Separate arms rather than a shared one because the heads differ: four
        // bytes here against the qualified pair's six and eight. Reading either
        // with another's layout starts the refs in the wrong place, which is what
        // the `count == refs.len()` check catches.
        wire::OPCODE_USE_RESOURCES_NO_STAGES => {
            let (head, refs) =
                wire::use_resources_no_stages(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::UseResource;
            out.count = head.count.get();
            if out.count as usize != refs.len() {
                return Err(DecodeStatus::ErrShort);
            }
            Ok(out)
        }
        wire::OPCODE_USE_HEAPS_NO_STAGES => {
            let (head, refs) =
                wire::use_heaps_no_stages(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::UseHeap;
            out.count = head.count.get();
            if out.count as usize != refs.len() {
                return Err(DecodeStatus::ErrShort);
            }
            Ok(out)
        }
        wire::OPCODE_MEMORY_BARRIER_RESOURCES
        | wire::OPCODE_MEMORY_BARRIER_SCOPE
        | wire::OPCODE_TEXTURE_BARRIER => {
            out.kind = Kind::Barrier;
            Ok(out)
        }
        wire::OPCODE_SET_DEPTH_CLIP_MODE | wire::OPCODE_SET_TRIANGLE_FILL_MODE => {
            let m = wire::mode_state(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetRasterState;
            out.mode = m.mode.get();
            Ok(out)
        }
        wire::OPCODE_DRAW_INDIRECT => {
            if command_length != wire::DRAW_INDIRECT_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let d = wire::draw_indirect(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::DrawIndirect;
            // 16 bits here, where the direct draws give the same field 32. The
            // two bytes above it are never written by the serializer, so a
            // wider read takes the guest's stale ring.
            out.primitive_type = d.primitive_type.get() as u32;
            out.indirect_buffer_ref = d.indirect_buffer_ref.get();
            out.indirect_buffer_offset = d.indirect_buffer_offset.get();
            Ok(out)
        }
        wire::OPCODE_DRAW_INDEXED_INDIRECT => {
            if command_length != wire::DRAW_INDEXED_INDIRECT_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let d = wire::draw_indexed_indirect(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::DrawIndirect;
            out.primitive_type = d.primitive_type.get() as u32;
            // Its own 16-bit field beside `primitive_type`, not the upper half
            // of a 32-bit one — reading a `u32` at `+0` would absorb it, which
            // is the bug the compact indexed draw had.
            out.index_type = d.index_type.get() as u32;
            out.index_buffer_ref = d.index_buffer_ref.get();
            out.index_buffer_offset = d.index_buffer_offset.get();
            out.indirect_buffer_ref = d.indirect_buffer_ref.get();
            out.indirect_buffer_offset = d.indirect_buffer_offset.get();
            Ok(out)
        }
        wire_tile::OPCODE_SET_TILE_THREADGROUP_MEMORY => {
            if command_length != wire_tile::SET_TILE_THREADGROUP_MEMORY_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let m = wire_tile::tile_threadgroup_memory(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::TileBind;
            out.first = m.index.get();
            out.count = 1;
            // The length and the offset are read by the view and then not
            // lifted, exactly as the other tile binds' entries are not: nothing
            // downstream allocates imageblock memory, so a field carrying the
            // size would have no consumer. `tile_threads` in particular is a
            // dispatch's grid and must not be borrowed for it.
            Ok(out)
        }
        wire_tile::OPCODE_SET_TILE_BUFFER => {
            // Entries are not lifted (no tile table consumer); first/count and
            // length come from the wire bind walk.
            let (head, _entries) =
                wire_tile::tile_buffer_binds(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::TileBind;
            out.first = head.first.get();
            out.count = head.count.get();
            if out.count == 0 {
                return Err(DecodeStatus::ErrBadLength);
            }
            match bind_record_len(out.count, BUFFER_BIND_ENTRY_SIZE) {
                Some(need) if payload.len() >= need => Ok(out),
                _ => Err(DecodeStatus::ErrShort),
            }
        }
        wire_tile::OPCODE_SET_TILE_TEXTURE => {
            let (head, _entries) =
                wire_tile::tile_texture_binds(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::TileBind;
            out.first = head.first.get();
            out.count = head.count.get();
            if out.count == 0 {
                return Err(DecodeStatus::ErrBadLength);
            }
            match bind_record_len(out.count, REF_BIND_ENTRY_SIZE) {
                Some(need) if payload.len() >= need => Ok(out),
                _ => Err(DecodeStatus::ErrShort),
            }
        }
        wire_tile::OPCODE_SET_TILE_SAMPLER => {
            let (head, _entries) =
                wire_tile::tile_sampler_binds(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::TileBind;
            out.first = head.first.get();
            out.count = head.count.get();
            if out.count == 0 {
                return Err(DecodeStatus::ErrBadLength);
            }
            match bind_record_len(out.count, REF_BIND_ENTRY_SIZE) {
                Some(need) if payload.len() >= need => Ok(out),
                _ => Err(DecodeStatus::ErrShort),
            }
        }
        wire_tile::OPCODE_SET_TILE_SAMPLER_LOD => {
            let (head, _entries) =
                wire_tile::tile_sampler_lod_binds(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::TileBind;
            out.first = head.first.get();
            out.count = head.count.get();
            if out.count == 0 {
                return Err(DecodeStatus::ErrBadLength);
            }
            match bind_record_len(out.count, SAMPLER_LOD_BIND_ENTRY_SIZE) {
                Some(need) if payload.len() >= need => Ok(out),
                _ => Err(DecodeStatus::ErrShort),
            }
        }
        wire_tile::OPCODE_SET_TILE_BUFFER_OFFSET => {
            if command_length != wire_tile::SET_TILE_BUFFER_OFFSET_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let b = wire_tile::tile_buffer_offset(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::TileBind;
            out.first = b.index.get();
            out.count = 1;
            out.buffer_offset = b.offset.get();
            Ok(out)
        }
        wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE => {
            if command_length != wire_tile::DISPATCH_THREADS_PER_TILE_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let d =
                wire_tile::dispatch_threads_per_tile(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::TileDispatch;
            out.tile_threads = [d.width.get(), d.height.get(), d.depth.get()];
            Ok(out)
        }
        wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION
        | wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION_RT_INDEX => {
            if command_length != wire_tile::DISPATCH_THREADS_PER_TILE_IN_REGION_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let d = wire_tile::dispatch_threads_per_tile_in_region(&op)
                .map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::TileDispatch;
            out.tile_threads = [d.width.get(), d.height.get(), d.depth.get()];
            // Region / RT index not lifted — see wire tile module.
            Ok(out)
        }
        wire_tile::OPCODE_GET_TILE_DIMENSIONS => {
            if command_length != wire_tile::GET_TILE_DIMENSIONS_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let g = wire_tile::get_tile_dimensions(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::TileDimensionsQuery;
            out.buffer_ref = g.buffer_ref.get();
            out.buffer_offset = g.offset.get();
            Ok(out)
        }
        wire::OPCODE_SET_VERTEX_AMPLIFICATION_MODE => {
            if command_length != wire::SET_VERTEX_AMPLIFICATION_MODE_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let m = wire::vertex_amplification_mode(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetVertexAmplification;
            out.mode = m.mode.get() as u64;
            out.amplification_value = m.value.get();
            Ok(out)
        }
        wire::OPCODE_SET_VERTEX_AMPLIFICATION_COUNT => {
            // Four-byte count head (not BindHeader); mappings follow and are
            // not lifted — nothing downstream amplifies. Wire parser bounds
            // entries to the record length.
            let (head, mappings) =
                wire::vertex_amplification_count(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetVertexAmplification;
            out.count = head.count.get();
            if out.count as usize != mappings.len() {
                return Err(DecodeStatus::ErrShort);
            }
            let _ = mappings; // unlifted by design
            Ok(out)
        }
        wire::OPCODE_SET_VISIBILITY_RESULT_MODE => {
            if command_length != wire::SET_VISIBILITY_RESULT_MODE_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let v = wire::set_visibility_result_mode(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetVisibilityResultMode;
            // Offset first, mode second. See `wire::OPCODE_SET_VISIBILITY_RESULT_MODE`.
            out.visibility_result_offset = v.offset.get();
            out.mode = v.mode.get();
            Ok(out)
        }
        wire::OPCODE_SET_LINE_WIDTH | wire::OPCODE_SET_TESSELLATION_FACTOR_SCALE => {
            let f = wire::float_state(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetFloatState;
            out.float_value = f.value.get();
            Ok(out)
        }
        wire::OPCODE_SET_COLOR_STORE_ACTION => {
            let a = wire::set_color_store_action(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetStoreAction;
            out.mode = u64::from(a.store_action.get());
            out.first = a.index.get();
            Ok(out)
        }
        wire::OPCODE_SET_DEPTH_STORE_ACTION | wire::OPCODE_SET_STENCIL_STORE_ACTION => {
            // Depth/stencil store actions share the one-NSUInteger mode shape.
            let m = wire::mode_state(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetStoreAction;
            out.mode = m.mode.get();
            Ok(out)
        }
        wire::OPCODE_SET_COLOR_STORE_ACTION_OPTIONS
        | wire::OPCODE_SET_DEPTH_STORE_ACTION_OPTIONS
        | wire::OPCODE_SET_STENCIL_STORE_ACTION_OPTIONS => {
            // The same three-attachment split one opcode higher, and the widths
            // do *not* carry over: the options are a `u64` where the store
            // action is a `u32`, so the colour form's index sits at `+8` rather
            // than `+4` and the record is 20 bytes rather than 16.
            if opcode == wire::OPCODE_SET_COLOR_STORE_ACTION_OPTIONS {
                if command_length != wire::SET_COLOR_STORE_ACTION_OPTIONS_TOTAL_LEN as usize {
                    return Err(DecodeStatus::ErrBadLength);
                }
                let a = wire::set_color_store_action_options(&op)
                    .map_err(|_| DecodeStatus::ErrShort)?;
                out.mode = a.options.get();
                out.first = a.index.get();
            } else {
                if command_length != wire::SET_STORE_ACTION_OPTIONS_TOTAL_LEN as usize {
                    return Err(DecodeStatus::ErrBadLength);
                }
                let a = wire::set_store_action_options(&op).map_err(|_| DecodeStatus::ErrShort)?;
                out.mode = a.options.get();
            }
            out.kind = Kind::SetStoreActionOptions;
            Ok(out)
        }
        wire::OPCODE_DRAW_PATCHES
        | wire::OPCODE_DRAW_PATCHES_WIDE
        | wire::OPCODE_DRAW_INDEXED_PATCHES
        | wire::OPCODE_DRAW_PATCHES_INDIRECT
        | wire::OPCODE_DRAW_INDEXED_PATCHES_INDIRECT => {
            // Six records across five opcodes, because `0x0c` is two: the plain
            // wide draw at 56 bytes and the indexed wide draw at 68. Dispatched
            // on the length rather than guessed, and a `0x0c` at any other
            // length is refused rather than read as whichever is closer -- the
            // two bodies disagree from their tenth byte on.
            //
            // Every form is length-checked exactly, so a truncated patch draw
            // is an `ErrBadLength` rather than a draw with invented counts.
            //
            // None of the fields is lifted. Nothing here tessellates, so a
            // `patch_count` in `Command` would be a producer with no consumer;
            // what `runtime::exec` needs is that a patch draw happened and
            // which form, and the opcode carries both.
            let want = match opcode {
                wire::OPCODE_DRAW_PATCHES => wire::DRAW_PATCHES_TOTAL_LEN as usize,
                wire::OPCODE_DRAW_INDEXED_PATCHES => wire::DRAW_INDEXED_PATCHES_TOTAL_LEN as usize,
                wire::OPCODE_DRAW_PATCHES_INDIRECT => {
                    wire::DRAW_PATCHES_INDIRECT_TOTAL_LEN as usize
                }
                wire::OPCODE_DRAW_INDEXED_PATCHES_INDIRECT => {
                    wire::DRAW_INDEXED_PATCHES_INDIRECT_TOTAL_LEN as usize
                }
                // `0x0c`: the two wide forms, and nothing else.
                _ => match command_length {
                    n if n == wire::DRAW_PATCHES_WIDE_TOTAL_LEN as usize => n,
                    n if n == wire::DRAW_INDEXED_PATCHES_WIDE_TOTAL_LEN as usize => n,
                    _ => return Err(DecodeStatus::ErrBadLength),
                },
            };
            if command_length != want {
                return Err(DecodeStatus::ErrBadLength);
            }
            // Viewed, so a record whose declared length outran its bytes is
            // refused here rather than by whoever reads it next.
            match opcode {
                wire::OPCODE_DRAW_PATCHES => {
                    wire::draw_patches(&op).map_err(|_| DecodeStatus::ErrShort)?;
                }
                wire::OPCODE_DRAW_INDEXED_PATCHES => {
                    wire::draw_indexed_patches(&op).map_err(|_| DecodeStatus::ErrShort)?;
                }
                wire::OPCODE_DRAW_PATCHES_INDIRECT => {
                    wire::draw_patches_indirect(&op).map_err(|_| DecodeStatus::ErrShort)?;
                }
                wire::OPCODE_DRAW_INDEXED_PATCHES_INDIRECT => {
                    wire::draw_indexed_patches_indirect(&op).map_err(|_| DecodeStatus::ErrShort)?;
                }
                _ if command_length == wire::DRAW_PATCHES_WIDE_TOTAL_LEN as usize => {
                    wire::draw_patches_wide(&op).map_err(|_| DecodeStatus::ErrShort)?;
                }
                _ => {
                    wire::draw_indexed_patches_wide(&op).map_err(|_| DecodeStatus::ErrShort)?;
                }
            }
            out.kind = Kind::DrawPatches;
            Ok(out)
        }
        wire::OPCODE_SET_TESSELLATION_FACTOR_BUFFER => {
            // Not a bind: one buffer per encoder, so there is no slot and no
            // count -- the ref and its two `u64` sit directly in the payload.
            if command_length != wire::SET_TESSELLATION_FACTOR_BUFFER_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let t =
                wire::set_tessellation_factor_buffer(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::SetTessellationFactorBuffer;
            out.buffer_ref = t.buffer_ref.get();
            out.buffer_offset = t.offset.get();
            Ok(out)
        }
        wire::OPCODE_EXECUTE_COMMANDS_INDIRECT => {
            if command_length != wire::EXECUTE_COMMANDS_INDIRECT_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let e = wire::execute_commands_indirect(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::ExecuteCommands;
            out.icb_is_range = false;
            out.indirect_command_buffer_ref = e.icb_ref.get();
            out.icb_args_buffer_ref = e.indirect_buffer_ref.get();
            out.icb_args_buffer_offset = e.indirect_buffer_offset.get();
            Ok(out)
        }
        wire::OPCODE_EXECUTE_COMMANDS_RANGE => {
            if command_length != wire::EXECUTE_COMMANDS_RANGE_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrBadLength);
            }
            let e = wire::execute_commands_range(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::ExecuteCommands;
            out.icb_is_range = true;
            out.indirect_command_buffer_ref = e.icb_ref.get();
            out.icb_range_location = e.range_location.get();
            out.icb_range_length = e.range_length.get();
            Ok(out)
        }
        wire_pass::OPCODE_RENDER_PASS => {
            if payload.len() < PASS_MIN_PAYLOAD {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::RenderPass;
            // Full Apple record: one wire body. Shorter product fixtures still
            // decode through the attachment views (same wire layouts at offsets).
            if let Ok(body) = wire_pass::render_pass(&op) {
                out.depth = depth_from_wire(&body.depth);
                out.stencil = stencil_from_wire(&body.stencil);
                out.color0 = color_from_wire(&body.color[0]);
                out.pass_visibility_result_buffer_ref = body.visibility_result_buffer_ref.get();
                out.pass_render_target_array_length = body.render_target_array_length.get();
                out.pass_render_target_width = body.render_target_width.get();
                out.pass_render_target_height = body.render_target_height.get();
            } else {
                out.depth = decode_depth_attachment(payload);
                out.stencil = decode_stencil_attachment(payload);
                out.color0 = decode_color_attachment(payload, 0);
            }
            if out.color0.texture_ref != 0 {
                out.texture_ref = out.color0.texture_ref;
            }
            Ok(out)
        }
        wire_pass::OPCODE_RASTERIZATION_RATE_MAP => {
            let r = wire_pass::pass_rate_map(&op).map_err(|_| DecodeStatus::ErrBadLength)?;
            out.kind = Kind::RenderPassProperty;
            out.texture_ref = r.rate_map_ref.get();
            Ok(out)
        }
        wire_pass::OPCODE_DEFAULT_RASTER_SAMPLE_COUNT => {
            let c = wire_pass::default_raster_sample_count(&op)
                .map_err(|_| DecodeStatus::ErrBadLength)?;
            out.kind = Kind::RenderPassProperty;
            out.mode = u64::from(c.count.get());
            Ok(out)
        }
        wire_pass::OPCODE_IMAGEBLOCK_SAMPLE_LENGTH
        | wire_pass::OPCODE_THREADGROUP_MEMORY_LENGTH => {
            let m = wire_pass::tile_memory(&op).map_err(|_| DecodeStatus::ErrBadLength)?;
            out.kind = Kind::RenderPassProperty;
            out.mode = u64::from(m.length.get());
            Ok(out)
        }
        wire_pass::OPCODE_TILE_SIZE => {
            let s = wire_pass::tile_size(&op).map_err(|_| DecodeStatus::ErrBadLength)?;
            out.kind = Kind::RenderPassProperty;
            // width | height << 16 — product packing, not a second layout.
            out.mode = u64::from(s.width.get()) | (u64::from(s.height.get()) << 16);
            Ok(out)
        }
        wire_pass::OPCODE_SAMPLE_POSITIONS => {
            let (head, positions) =
                wire_pass::sample_positions(&op).map_err(|_| DecodeStatus::ErrBadLength)?;
            out.kind = Kind::RenderPassProperty;
            out.count = head.count.get();
            if out.count as usize != positions.len() {
                return Err(DecodeStatus::ErrBadLength);
            }
            Ok(out)
        }
        _ => {
            out.kind = Kind::OtherAccepted;
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests;
