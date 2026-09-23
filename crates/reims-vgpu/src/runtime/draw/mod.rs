//! Encode one decoded draw for a resolved pipeline + colour target, on whichever
//! backend this build has.
//!
//! Loads per-function MTLB containers from the object list, materializes stream
//! binds (vertex/fragment buffers, optional index buffer, viewport/scissor),
//! hands them to the backend, and writes the RGBA result into the mapper-ref-texture
//! mapping via [`mapping_write`]. The encode call is
//! [`crate::backend::metal::render::render_core_mrt`] on the Metal arm and
//! `try_metal2vulkan_draw` into `backend::vulkan::engine` on the Vulkan one.
//!
//! # This module is not Metal-only, and its name used to say it was
//!
//! It was `metal_draw` until the composition was counted. The gated Vulkan half
//! (`vulkan.rs`) is the largest file here by a factor of two, and the
//! backend-independent halves — `texture_view`, `render_target`, and this
//! file's own bind materialization — run on both arms on every draw. Only
//! `metal_icb` and `depth_stencil` are genuinely Metal-side, and both carry
//! their own gates.
//!
//! The fail-log event names emitted from this file (`metal_draw MissingPipeline`
//! and its siblings) and the counters that count them (`metal_draws_ok`,
//! `metal_draws_fail`) deliberately kept their spelling through that rename.
//! They are operator-facing vocabulary that appears in already-recorded
//! measurements, so respelling them would silently invalidate every reading
//! taken before it. The module path and the log vocabulary are two different
//! names, and only one of them was wrong.

use crate::backend::Backend as _;
use crate::model::DeviceState;
use crate::protocol::pixel_format::{
    self, solid_rgba8, SampledByteFormat, TexelLayout, MTL_FORMAT_BGRA8_UNORM, RGBA8_BPP,
};
// `Decline::slug` on typed draw, coverage, and translation reasons.
use crate::observe::Decline;
// The one downgrade site left in this tree is the secondary colour attachment,
// which is Vulkan-only. The CPU upload rails used to report here too and no
// longer downgrade at all: they carry the source format through to the bind.
// Only the tests read it here; the rail that acts on it imports its own. The
// Metal arm tests the band instead (`load_action_in_contract`).
use crate::runtime::decode::resource::{
    decode_buffer_texture_descriptor, decode_depth_stencil_descriptor,
    decode_render_pipeline_descriptor, decode_texture_descriptor, texture_view_opcode,
    BufferTextureDescriptor, DecodeStatus, RenderPipelineDescriptor,
    OBJECT_TYPE_MAPPER_REF_TEXTURE, OBJECT_TYPE_SERIALIZER_OBJECT, OBJECT_TYPE_TEXTURE,
    OBJECT_TYPE_TEXTURE_GENERATE_MIPMAPS, OBJECT_TYPE_TEXTURE_VIEW,
    TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE, TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE_WIDE,
};
use crate::runtime::gva_mem;
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::mapper;
use crate::runtime::mapping_write;
use crate::runtime::mtlb::{load_mtlb, AirLoadRail};
use crate::runtime::objects;
use crate::runtime::render_pass::{
    ColorAttachment, DepthAttachment, ScissorRect, StencilAttachment,
};
use reims_vgpu_protocol::pass_action::MTL_LOAD_ACTION_DONT_CARE;
use reims_vgpu_protocol::pass_action::{is_declared_load_action, is_declared_store_action};
use reims_vgpu_protocol::pass_action::{
    MTL_LOAD_ACTION_CLEAR, MTL_LOAD_ACTION_LOAD, MTL_STORE_ACTION_STORE,
};

// The Vulkan half of this path, named rather than re-exported flat — the
// sibling of `metal`, and gated once here rather than per item.
#[cfg(feature = "backend-vulkan")]
pub mod vulkan;

fn dontcare_seed_probe_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            crate::config::read(crate::config::DONTCARE_SEED_PROBE).0,
            crate::config::Switch::On
        )
    })
}
// Only for `exec`'s pass-extent census, which declares its own copy of these
// bands because it runs on every backend. See
// `the_two_coverage_censuses_use_the_same_bands`.
#[cfg(all(test, feature = "backend-vulkan"))]
pub(crate) use vulkan::coverage_band_for_test;

// The Metal half of this path, named rather than re-exported flat. Both rails
// own an `encode_draw_chain` with the same signature — that is the point — so
// flattening either one into this module would make the two collide and is why
// a build could previously carry only one of them.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
pub mod metal;

// Texture-view texture-view resolution and linear texture loads. Backend-independent,
// so the module carries no gate of its own; the two items inside it that are
// arm- or test-specific keep theirs.
mod texture_view;
pub(crate) use texture_view::*;

// The colour render-target resolve ladder: `texture_ref` → mapping id or linear
// guest VA. Backend-independent, so no gate, for the same reason as
// `texture_view`. Named individually rather than re-exported flat, because the
// ladder's two report helpers are its own working parts and only these two
// items have callers outside it.
mod render_target;
use render_target::{lookup_render_target, ResolvedRenderTarget};

/// Bind **index** cap for the buffer argument table.
///
/// Two independent derivations of one number, which is why it is the one bound
/// of the three that costs nothing: Metal's buffer argument table ends at 31
/// (`REIMS_VGPU_METAL_MAX_BUFFERS`, pinned equal to this by a `const` assertion
/// in `backend::metal::constants`), and Apple's own serializer truncates a
/// plural buffer bind there too (`reims_vgpu_wire::ops::bind_limit::BUFFER`).
/// A guest bind past it cannot come from an Apple stream, and no backend could
/// hold it if it did.
pub const MAX_BUFFER_BIND_SLOTS: u32 = 31;

/// Bind **index** cap for the texture argument table.
///
/// A slot count, not a byte budget. Resource byte sizes follow the guest
/// descriptor and page-table span; nothing here caps them.
///
/// # This is Apple's whole texture table, and nothing is refused below it
///
/// 128 is `reims_vgpu_wire::ops::bind_limit::TEXTURE` — the size of the argument
/// table Apple's serializer truncates a plural texture bind at — so no texture
/// bind an Apple guest can emit reaches this bound. A `const` assertion in
/// [`crate::runtime::exec`] pins the two equal, and
/// `render_texture_bind_slot_past_table` stays as the alarm for a stream that
/// somehow does.
///
/// It is also the width of the device's texture binding band, which is what used
/// to make it 31. The device names a bound resource by one `u32` descriptor
/// binding that packs class and index into bands, and `metal2vulkan` emits those
/// bands 32 apart — so texture 40 and sampler 8 were both binding 72, and the
/// *number* could not say which. Slots 32..127 were dropped for that reason
/// alone.
///
/// [`crate::runtime::spirv_bind::widen_sampled_bands`] removes it. The sampler
/// and ColorInput bands move up out of the way once per shader, keyed on each
/// variable's SPIR-V *type* rather than its number, leaving the texture band
/// exactly 128 wide with the translator's own texture decorations already
/// correct in it. So this constant is now the same fact twice — Apple's table
/// and the band's width — and the `const` assertions beside
/// `SAMPLER_BINDING_BASE` hold it to both.
pub const MAX_TEXTURE_BIND_SLOTS: u32 = 128;

/// Bind **index** cap for the sampler argument table.
///
/// The sampler band is `[160, 192)` — [`crate::runtime::spirv_bind::SAMPLER_BINDING_BASE`]
/// up to [`crate::runtime::spirv_bind::COLOR_INPUT_BINDING_BASE`] — so this is
/// the same encoding bound [`MAX_TEXTURE_BIND_SLOTS`] documents, applied to the
/// next band up.
///
/// The *table* that actually runs out first is Metal's, at 16
/// (`REIMS_VGPU_METAL_MAX_SAMPLERS`), and Apple's serializer truncates there too
/// (`bind_limit::SAMPLER`). That bound is not applied here on purpose: it
/// belongs to one backend, and the backend that owns it refuses at its own
/// encoder, fail-visibly, with `metal_render_sampler_binding_invalid` naming the
/// binding. Applying a Metal table size during stream accumulation would take
/// the slot away from the Vulkan arm as well, which is exactly the mistake the
/// single shared `MAX_BIND_SLOTS` made for two of its three classes.
pub const MAX_SAMPLER_BIND_SLOTS: u32 = 32;

/// The widest of the three bind bounds.
///
/// For sizing something one *descriptor type* draws from, where the type is
/// served by exactly one class and the caller does not know which — the Vulkan
/// descriptor arena's per-type block budget is the case. Declared beside the
/// three constants rather than at the site, so the three-way comparison is not
/// a fourth copy of the rule.
pub const MAX_ANY_BIND_SLOTS: u32 = {
    let widest = if MAX_TEXTURE_BIND_SLOTS > MAX_SAMPLER_BIND_SLOTS {
        MAX_TEXTURE_BIND_SLOTS
    } else {
        MAX_SAMPLER_BIND_SLOTS
    };
    if widest > MAX_BUFFER_BIND_SLOTS {
        widest
    } else {
        MAX_BUFFER_BIND_SLOTS
    }
};

/// Which of the three argument tables a bind record names.
///
/// The three constants above are compared against a guest slot index in exactly
/// one place — [`BindTableClass::table`] — and every consumer asks it rather
/// than spelling its own comparison. Before that, the same rule was written out
/// at twenty-two sites across four files in two spellings, one of them inverted,
/// and the three arms consuming one wire form had drifted into three different
/// behaviors for the identical input: the ICB arm refused with a typed reason,
/// the direct-Metal and Vulkan arms dropped the bind in silence.
///
/// [`crate::runtime::exec`] adds the census vocabulary — Apple's own table size
/// for the class, the reach bands, the drop slug — as its own `impl` on this
/// type, because those describe how a loss is *reported* rather than what the
/// table *is*.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindTableClass {
    Buffer,
    Texture,
    Sampler,
}

impl BindTableClass {
    /// This device's bind-index bound for the class.
    ///
    /// One constant per class, because the three have different bases: the
    /// buffer bound is Metal's argument table (and, independently, Apple's own),
    /// while the texture and sampler bounds are the width of a descriptor
    /// binding band. A single shared constant made two of the three the wrong
    /// number by construction — it was Metal's *buffer* table applied to all
    /// three — which is what [`MAX_TEXTURE_BIND_SLOTS`] records.
    pub fn table(self) -> u32 {
        match self {
            BindTableClass::Buffer => MAX_BUFFER_BIND_SLOTS,
            BindTableClass::Texture => MAX_TEXTURE_BIND_SLOTS,
            BindTableClass::Sampler => MAX_SAMPLER_BIND_SLOTS,
        }
    }

    /// The name this class carries on a fail line.
    pub fn name(self) -> &'static str {
        match self {
            BindTableClass::Buffer => "buffer",
            BindTableClass::Texture => "texture",
            BindTableClass::Sampler => "sampler",
        }
    }
}

/// A live bind in one draw request whose slot no argument table of its class can
/// name.
///
/// Carries the object ref as well as the slot, because the two say different
/// things: the slot names which table ran out, and the ref is what the guest
/// still believes is bound there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PastTableBind {
    pub class: BindTableClass,
    pub stage: reims_vgpu_protocol::render::ShaderStage,
    /// The guest's own slot index, so it reads against [`BindTableClass::table`].
    pub index: u32,
    /// The object bound there. Never zero — see [`first_bind_past_table`].
    pub resource_ref: u32,
}

impl PastTableBind {
    pub fn stage_name(&self) -> &'static str {
        // Two arms, because `ShaderStage` is the render opcode set's whole
        // stage vocabulary. The "unknown" this used to answer with named a
        // variant no decoder arm ever produced.
        self.stage.name()
    }
}

/// The first live bind in `req` that names a slot past its class's table, if any.
///
/// # Why every backend calls this once instead of checking at each consumer
///
/// A slot past the table is not a bind that can be degraded: no encoder of
/// either backend has an argument-table entry to put it in, and Metal answers an
/// out-of-range argument-table index with a process-aborting exception rather
/// than an error. So the only faithful answer is to refuse the whole draw and
/// say which slot did it — the same answer for all three classes, both stages
/// and both backends, which is why it is one function.
///
/// It is asked once, before any resource is resolved, so a refused draw does no
/// upload work first and the reported slot is the guest's own rather than
/// whichever consumer happened to notice.
///
/// **A zero ref is not reported.** Clearing a slot the device does not model
/// loses no guest work, and expected control flow stays quiet.
///
/// # This is a backstop, and it is meant to stay one
///
/// `runtime::exec::apply_binds` is the only writer of these six tables and
/// already stops a record's walk at the same bound, fail-visibly and with the
/// reach census beside it. So a `Some` here means that gate was bypassed, not
/// that a guest asked for something new. It is kept because the cost of being
/// wrong is a Metal exception that takes the process down, and because the check
/// that once stood at each consumer had already drifted three ways.
pub fn first_bind_past_table(req: &DrawEncodeRequest) -> Option<PastTableBind> {
    use reims_vgpu_protocol::render::ShaderStage as Stage;

    let buffers = [
        (Stage::Vertex, &req.vertex_buffers),
        (Stage::Fragment, &req.fragment_buffers),
    ];
    for (stage, binds) in buffers {
        for b in binds.iter() {
            if b.buffer_ref != 0 && b.index >= BindTableClass::Buffer.table() {
                return Some(PastTableBind {
                    class: BindTableClass::Buffer,
                    stage,
                    index: b.index,
                    resource_ref: b.buffer_ref,
                });
            }
        }
    }
    let textures = [
        (Stage::Vertex, &req.vertex_textures),
        (Stage::Fragment, &req.fragment_textures),
    ];
    for (stage, binds) in textures {
        for t in binds.iter() {
            if t.texture_ref != 0 && t.index >= BindTableClass::Texture.table() {
                return Some(PastTableBind {
                    class: BindTableClass::Texture,
                    stage,
                    index: t.index,
                    resource_ref: t.texture_ref,
                });
            }
        }
    }
    let samplers = [
        (Stage::Vertex, &req.vertex_samplers),
        (Stage::Fragment, &req.fragment_samplers),
    ];
    for (stage, binds) in samplers {
        for s in binds.iter() {
            if s.sampler_ref != 0 && s.index >= BindTableClass::Sampler.table() {
                return Some(PastTableBind {
                    class: BindTableClass::Sampler,
                    stage,
                    index: s.index,
                    resource_ref: s.sampler_ref,
                });
            }
        }
    }
    None
}

/// Convert a guest-declared byte length to a host allocation size.
///
/// Only fails when the length does not fit `usize` (process addressability) —
/// **not** an arbitrary product MiB budget.
#[inline]
pub fn host_alloc_len(bytes: u64) -> Option<usize> {
    usize::try_from(bytes)
        .ok()
        .filter(|&n| n <= isize::MAX as usize)
}

/// The vertex fetch stride in force for one buffer index.
///
/// `setVertexBuffer:offset:attributeStride:atIndex:` overrides whatever the
/// pipeline's `MTLVertexBufferLayoutDescriptor` declared for that index, so the
/// bind wins where it carried one and `pipeline_stride` stands where it did not.
///
/// One function rather than the rule spelled at each backend, because both arms
/// consume the same two inputs and a divergence between them would be a
/// difference in *geometry* — a mesh fetched at the wrong stride still
/// rasterizes, so nothing downstream reports it. The Metal arm reads this into
/// `ReimsVgpuBuffer::attribute_stride`; the Vulkan arm reads it into
/// `AttrKey::stride`, where it is already part of the pipeline key.
///
/// A stride wider than `u32` is left to the pipeline's own: it cannot reach
/// either backend, since Metal's ABI mirror and Vulkan's
/// `VkVertexInputBindingDescription::stride` are both 32-bit, and silently
/// truncating a guest `u64` would fetch at an unrelated stride rather than at
/// the one asked for.
pub fn bind_attribute_stride(
    vertex_buffers: &[BufferBind],
    buffer_index: u32,
    pipeline_stride: u32,
) -> u32 {
    vertex_buffers
        .iter()
        .find(|b| b.index == buffer_index)
        .and_then(|b| b.attribute_stride)
        .and_then(|s| u32::try_from(s).ok())
        .unwrap_or(pipeline_stride)
}

/// BGRA<->RGBA channel swap (swap byte 0 and 2 of each 4-byte pixel) producing a
/// fresh `Vec`, in a SINGLE read+write pass. Replaces the `src.to_vec()` +
/// in-place `chunks_exact_mut(4)` swizzle-loop idiom, which walked the pixel data
/// twice (a copy pass, then a read-modify-write pass). The swap is its own
/// inverse, so this serves both directions. Any trailing bytes that do not fill a
/// whole 4-byte pixel are copied through unchanged — byte-identical to the prior
/// `to_vec()` (which copied the tail) followed by `chunks_exact_mut` (which left
/// the tail untouched). This is the hottest per-bind byte-mover on the sampled
/// cache path (the `lin_guest` / `gva_copy` census branches).
#[inline]
fn swap_rb_channels(src: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; src.len()];
    let mut src_px = src.chunks_exact(4);
    let mut out_px = out.chunks_exact_mut(4);
    for (s, d) in (&mut src_px).zip(&mut out_px) {
        d[0] = s[2];
        d[1] = s[1];
        d[2] = s[0];
        d[3] = s[3];
    }
    let rem = src_px.remainder();
    if !rem.is_empty() {
        let start = out.len() - rem.len();
        out[start..].copy_from_slice(rem);
    }
    out
}

/// [`swap_rb_channels`] for a frame the caller already owns: one read-modify-write
/// pass, no allocation.
///
/// # Why the pair and not just the allocating one
///
/// Both directions of this exchange exist because both call shapes exist. A
/// reader handed an `Arc` or a borrowed cache slice cannot write through it and
/// needs the fresh `Vec`; a reader that has just *produced* the frame — a
/// resident readback, which is the shape both rails' capture and seed paths
/// take — owns it, and making it allocate a second full frame to reorder four
/// bytes at a time is a whole 8 MB of copy per present for nothing.
///
/// The exchange is its own inverse, so this serves both directions, and a
/// trailing partial pixel is left as it is — the same tail rule
/// [`swap_rb_channels`] follows, stated once so the two cannot drift.
#[inline]
pub(crate) fn swap_rb_channels_in_place(frame: &mut [u8]) {
    for px in frame.chunks_exact_mut(4) {
        px.swap(0, 2);
    }
}

/// One slot of a render encoder's vertex or fragment buffer table.
///
/// The stage is not a field. A bind lives in `vertex_buffers` or in
/// `fragment_buffers`, and which table holds it *is* the stage; carrying it
/// again inside the element made two encodings of one fact that had to agree,
/// and nothing ever read the copy.
#[derive(Clone, Debug, Default)]
pub struct BufferBind {
    pub index: u32,
    pub buffer_ref: u32,
    /// The resource this encoder slot named when it was bound.
    ///
    /// Object references may be deleted and reused after commands have been
    /// recorded. The binding is an object lifetime, not a deferred lookup of
    /// that integer, so draw preparation consumes this retained identity when
    /// it is available. `None` keeps synthetic requests and a construction
    /// that was not ready at setter time retryable through the numeric ref.
    pub resource: Option<std::sync::Arc<crate::model::TaskResource>>,
    pub offset: u64,
    /// The vertex fetch stride this bind declares, from
    /// `setVertexBuffer:offset:attributeStride:atIndex:` and its plural and
    /// offset-only siblings. `None` means the record carried no stride table,
    /// so whatever the pipeline's vertex layout declared for this index stands.
    ///
    /// Same shape as [`SamplerBind::lod_clamp`] — a value the bind record
    /// carries that overrides pipeline state — and it arrived the same way,
    /// which is that the opcodes carrying it were being decoded and their extra
    /// field stepped over. The compute rail has carried this field the whole
    /// time, on `ReimsVgpuBuffer::attribute_stride`, through
    /// `raw_metal::set_buffer_with_attribute_stride`.
    pub attribute_stride: Option<u64>,
}

/// One slot of a render encoder's vertex or fragment texture table. The stage
/// is the table it is in; see [`BufferBind`].
#[derive(Clone, Debug, Default)]
pub struct TextureBind {
    pub index: u32,
    pub texture_ref: u32,
    /// The object identity retained by this encoder slot. See
    /// [`BufferBind::resource`].
    pub resource: Option<std::sync::Arc<crate::model::TaskResource>>,
}

/// One slot of a render encoder's vertex or fragment sampler table. The stage
/// is the table it is in; see [`BufferBind`].
#[derive(Clone, Debug, Default)]
pub struct SamplerBind {
    pub index: u32,
    pub sampler_ref: u32,
    /// `(lodMinClamp, lodMaxClamp)` as raw `f32` bits, when the bind record
    /// carried its own pair — `setVertexSamplerStates:lodMinClamps:
    /// lodMaxClamps:withRange:` and its fragment sibling. `None` leaves the
    /// sampler object's own clamps in force, which is what
    /// `setVertexSamplerStates:` alone means.
    ///
    /// Bits rather than `f32` so the value crosses the two backends the way
    /// the compute rail's `ComputeSamplerBind` already sends it, and so a bind
    /// carrying a NaN clamp is the guest's NaN rather than one this device
    /// invented by rounding.
    pub lod_clamp: Option<(u32, u32)>,
}

#[derive(Clone, Debug, Default)]
pub struct IndexedDrawInfo {
    pub index_type: u32,
    pub index_count: u32,
    pub index_buffer_ref: u32,
    pub index_buffer_offset: u64,
    /// Metal `baseVertex` / Vulkan `vertexOffset`, added to every index before
    /// the vertex fetch. Signed, because Metal's is, and because a negative one
    /// read as unsigned becomes a huge index rather than an error.
    pub base_vertex: i64,
}

/// One color RT for MRT encode/writeback.
///
/// Archive `ApplePVGPURenderTarget`: either mapper-ref-texture IOSurface (`mapping_id`) or
/// normal-texture guest-VA linear (`target_gva` + `row_stride`). Wallpaper/background
/// layers are the GVA form.
#[derive(Clone, Debug, Default)]
pub struct ColorRtRequest {
    pub slot: u32,
    pub texture_ref: u32,
    pub mapping_id: u32,
    /// Non-zero ⇒ normal-texture linear GVA target (mapping_id must be 0).
    pub target_gva: u64,
    /// Bytes-per-row for GVA target (archive `bpr`).
    pub row_stride: u32,
    pub width: u32,
    pub height: u32,
    pub format: u16,
    /// Sample count of the attachment texture (the multisample source when a
    /// separate resolve texture is present).
    pub sample_count: u32,
    pub load_action: u16,
    pub store_action: u16,
    pub clear_color: [f64; 4],
    pub target_seed_rgba: Option<Vec<u8>>,
    /// Multisample attachment discarded into this request's single-sample
    /// target at pass end. Zero for an ordinary colour attachment.
    pub multisample_source_ref: u32,
}

/// One `setVisibilityResultMode:offset:`, as the encoder state it is.
///
/// The offset travels with the mode rather than beside it because they are one
/// record and mean nothing apart: the mode says what to count and the offset
/// says which 64-bit word of the pass's `visibilityResultBuffer` the count
/// lands in. Several offsets in one pass are legal Metal — that is how a guest
/// asks a pass several independent occlusion questions — so the writeback keys
/// results by offset rather than assuming one per pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisibilityArming {
    /// `MTLVisibilityResultMode`, carried raw and translated per backend, the
    /// way `cull_mode` and `fill_mode` beside it are: only the backend knows
    /// whether the host can spell the answer, so only the backend can refuse by
    /// name. Never `0` — `MTLVisibilityResultModeDisabled` is the `None` around
    /// this.
    pub mode: u32,
    /// Byte offset into the pass's `visibilityResultBuffer`.
    pub offset: u64,
}

/// One stage's retained bind table as a draw consumes it.
///
/// Encoder setters replace entries in sticky tables; draws retain the table
/// current at the point they were recorded. Sharing that immutable snapshot
/// through backend preparation preserves that lifecycle and avoids copying the
/// same entries again at the execution boundary. The accumulator mutates
/// through [`std::sync::Arc::make_mut`], which copies only when an earlier draw
/// still owns the previous snapshot: a stream that binds once and draws 400
/// times therefore allocates one table and retains 400 pointers, including
/// through backend preparation.
pub type BindTable<T> = std::sync::Arc<Vec<T>>;

#[derive(Clone, Debug, Default)]
pub struct DrawEncodeRequest {
    pub task_id: u32,
    pub pipeline_ref: u32,
    pub vertex_count: u32,
    pub instance_count: u32,
    pub primitive_type: u32,
    pub first_vertex: u32,
    /// Metal `baseInstance` / Vulkan `firstInstance`. Both backends already
    /// take it; until the draw forms that carry one were decoded, both were
    /// handed a hardcoded zero from here.
    pub base_instance: u32,
    /// Every color RT the pass declared, slot 0 first. The sole statement of
    /// what this draw renders into: geometry, format, target identity and Load
    /// seed all live here and nowhere else, so no two fields of one request can
    /// disagree about the attachment.
    pub colors: Vec<ColorRtRequest>,
    pub vertex_buffers: BindTable<BufferBind>,
    pub fragment_buffers: BindTable<BufferBind>,
    pub vertex_textures: BindTable<TextureBind>,
    pub fragment_textures: BindTable<TextureBind>,
    pub vertex_samplers: BindTable<SamplerBind>,
    pub fragment_samplers: BindTable<SamplerBind>,
    /// Every viewport the pass bound, in the guest's order, as
    /// `[originX, originY, width, height, znear, zfar]`. Empty means the guest
    /// bound none and the backend's full-target default stands.
    ///
    /// A list because `setViewports:count:` is one record with N entries, and
    /// this device used to keep entry 0 and count the rest as a named loss.
    /// Both backends take an array natively — `setViewports:count:` and
    /// `vkCmdSetViewport` — so the only thing bounded to one was this field.
    pub viewports: Vec<[f64; 6]>,
    /// Every scissor rect the pass bound, in the guest's order. Entry `i` clips
    /// viewport `i`; see [`Self::viewports`].
    pub scissors: Vec<ScissorRect>,
    /// The occlusion query this draw is armed with, or `None` where the guest
    /// disarmed it (`MTLVisibilityResultModeDisabled`) or never armed one.
    pub visibility: Option<VisibilityArming>,
    /// Samples the draw passed, filled in by the backend that ran the query.
    ///
    /// An **out** field on a request, which is the shape the encode chain
    /// already uses for what a draw produced rather than what it was asked to
    /// do. `None` where no query was armed *or* where this backend cannot run
    /// one; the two are told apart by whether [`Self::visibility`] is set, and
    /// the backend that cannot names its own refusal.
    pub visibility_samples: Option<u64>,
    pub indexed: Option<IndexedDrawInfo>,
    pub blend_color: Option<[f32; 4]>,
    pub cull_mode: Option<u32>,
    pub front_facing: Option<u32>,
    /// `MTLTriangleFillMode` from `setTriangleFillMode:`, raw. `None` means the
    /// stream bound none, so Metal's default (fill) stands.
    pub fill_mode: Option<u32>,
    /// `MTLDepthClipMode` from `setDepthClipMode:`, raw. `None` means Metal's
    /// default (clip).
    pub depth_clip_mode: Option<u32>,
    pub depth_bias: Option<[f32; 3]>,
    /// `setLineWidth:` — the width the stream last set, `None` where it set
    /// none and Metal's own default (1.0) stands.
    ///
    /// Raw, and not folded into a default here: `None` and `Some(1.0)` are the
    /// same rasterization but not the same stream, and the rail that cannot
    /// spell this command reports only the second.
    pub line_width: Option<f32>,
    pub depth_stencil_ref: u32,
    pub stencil_ref: Option<(u32, u32)>,
    pub depth_attach: Option<DepthAttachment>,
    pub stencil_attach: Option<StencilAttachment>,
    /// Records 2+ of a resident render-pass chain: load the prior record's
    /// content from the engine target instead of a CPU seed. Set by the exec
    /// chain loop (Vulkan rail only); default false.
    pub chain_from_resident: bool,
    /// This draw continues the Metal render encoder of the preceding draw in
    /// the same decoded stream. Vulkan may keep an identical render pass open
    /// when no command that is illegal inside it intervenes.
    pub continues_render_pass: bool,
    /// Another draw in this decoded Metal render encoder follows this one.
    /// Vulkan may defer `vkCmdEndRenderPass` until that draw, an outside-pass
    /// command, or the command-buffer flush closes it.
    pub render_pass_continues: bool,
    /// This pass's colour0 is a GVA target whose `MTLLoadActionLoad` was **not**
    /// seeded, because the engine still holds what the render Store published
    /// into its guest pages. Set by `mrt_draw_request` from
    /// `draw::vulkan::gva_resident_if_current`; Vulkan rail only.
    ///
    /// Distinct from [`Self::chain_from_resident`], which is about records 2+ of
    /// one pass and is read by two other rails besides the Load gate. This says
    /// only "the seed is deliberately absent, chain instead" and nothing else
    /// keys off it.
    ///
    /// **A `true` here obliges the encode side to produce content one way or the
    /// other.** `colors[0].target_seed_rgba` is `None` and the attachment still
    /// says LOAD, so an encode that neither chains nor re-seeds hands the pass an
    /// undefined attachment. The re-seed is not theoretical: the generation this
    /// was decided on is recomputed after the request is built, and a page set
    /// that moved in between names a different target.
    pub gva_load_from_resident: bool,
    /// Out-flag: this record kept chain content on the engine-resident
    /// target (no CPU pixels, no guest Store). The exec chain loop arms
    /// `chain_from_resident` for the next record when set.
    pub chain_resident_established: bool,
    /// Lifetime identity of the color0 GVA render resource.
    ///
    /// Resolved once per draw, before any GPU work, by
    /// `draw::vulkan::gva_alloc_generation`, and carried here so every
    /// `TargetIdentity::Gva` this draw builds agrees on one `generation`.
    /// Resource delete changes it; ordinary task map changes and transfer-
    /// backing discard do not.
    ///
    /// 0 means "no allocation named": color0 is not a GVA target, or the span
    /// does not fully walk. Vulkan rail only; the Metal arm never reads it.
    pub gva_alloc_gen: u64,
}

/// Fire `reason` once per `(pipeline_ref, slug)` so a recurring degradation
/// (e.g. a whole 3D scene requesting depth LOAD, or every draw of one pipeline
/// carrying the same out-of-contract raster value) logs once, not per draw.
/// Returns true the first time a given key is seen.
///
/// Backend-agnostic on purpose: both encode arms degrade, so both need the same
/// dedupe. While this was Vulkan-only the Metal arm had no way to report a
/// degradation without flooding per draw, and reported none.
#[cfg(any(
    feature = "backend-vulkan",
    all(feature = "backend-metal", target_os = "macos")
))]
fn degrade_log_first(pipeline_ref: u32, slug: &'static str) -> bool {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(u32, &'static str)>>> = Mutex::new(None);
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    seen.get_or_insert_with(HashSet::new)
        .insert((pipeline_ref, slug))
}

/// How a render-encode attempt ended.
///
/// Every refusal carries the registered slug of the check that produced it. The
/// variant is the *class* the caller acts on — `NoMetal` makes `exec` fall
/// back to the pass clear, `WritebackFailed` does not — and the payload is which
/// of the rail's checks refused. Before this, six payload-free variants spoke for
/// 27 checks: `BadArgs` alone covered eight, and `draw_encode_fail
/// reason=bad_args` could be a zero-size target, a vertexless draw or an ICB
/// range past the end of its buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncodeStatus {
    Ok,
    /// A rail refused with structure. See [`crate::runtime::compute_exec::ComputeStatus::RailRefused`]
    /// — its twin, and the same reason for being neutral and ungated.
    RailRefused(crate::backend::refusal::RailRefusal),
    MissingPipeline(&'static str),
    MissingMtlb(&'static str),
    MetalFailed(&'static str),
    WritebackFailed(&'static str),
    BadArgs(&'static str),
    /// Metal feature not built (vulkan boot), or nothing landed on the Vulkan
    /// rail — `exec` treats both as "honour the pass clear instead".
    NoMetal(&'static str),
    /// The record was well-formed and this device implements no answer for it on
    /// any pathway. Recovery is `NoMetal`'s — nothing was encoded, so honour the
    /// pass clear — but the class is not, and a reader triaging a black frame on
    /// a Metal host needs to know the difference between a stub and a gap.
    Unsupported(&'static str),
}

impl crate::observe::Refusal for EncodeStatus {
    fn refusal(&self) -> Option<&'static str> {
        match self {
            // The only non-refusal, and the reason this is a `Refusal` rather
            // than a `Decline`: `Emit::refusal` cannot render a line for it.
            Self::Ok => None,
            Self::RailRefused(refusal) => refusal.refusal(),
            Self::MissingPipeline(slug)
            | Self::MissingMtlb(slug)
            | Self::MetalFailed(slug)
            | Self::WritebackFailed(slug)
            | Self::BadArgs(slug)
            | Self::NoMetal(slug)
            | Self::Unsupported(slug) => Some(slug),
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        // The class beside the reason: which recovery path the caller took is
        // not derivable from the slug, and a reader correlating a dropped draw
        // with a black frame needs both.
        if let Self::RailRefused(refusal) = self {
            let mut fields = crate::observe::Refusal::fields(refusal);
            fields.push(("recovery", "metal_failed".to_string()));
            return fields;
        }
        vec![("class", self.class().to_string())]
    }
}

impl EncodeStatus {
    /// The variant name, for the `class=` field and for call sites that render
    /// their own line.
    pub fn class(&self) -> &'static str {
        match self {
            Self::Ok => "ok",
            // The two names the boot logs have carried since this was
            // `MetalBackend`; see `ComputeStatus::class`.
            Self::RailRefused(refusal) => {
                if refusal.is_args() {
                    "metal_args"
                } else {
                    "metal_execute"
                }
            }
            Self::MissingPipeline(_) => "missing_pipeline",
            Self::MissingMtlb(_) => "missing_mtlb",
            Self::MetalFailed(_) => "metal_failed",
            Self::WritebackFailed(_) => "writeback_failed",
            Self::BadArgs(_) => "bad_args",
            Self::NoMetal(_) => "no_metal",
            Self::Unsupported(_) => "unsupported",
        }
    }
}

/// Why an indexed draw's index bytes could not be resolved.
///
/// Eleven distinct checks, and until this type existed the Metal rail threw
/// every one of them away: `load_index_bytes` was an `Option` adapter over the
/// reasoned loader (`.ok()`), so a dropped indexed draw returned a bare
/// `MetalFailed` with **no log line at all** — the one fully silent refusal left
/// on the render rail. The Vulkan rail already consumed the reasons, as prose
/// inside a `String`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexLoadReason {
    TypeUnsupported,
    CountOverflow,
    CountZero,
    EntryMissing,
    ObjectType,
    DescRead,
    DescDecode,
    BackingMissing,
    OffsetOverflow,
    OutOfBounds,
    ReadFail,
    /// The guest's `baseVertex` does not fit Vulkan's signed 32-bit
    /// `vertexOffset`. Metal's is 64-bit, so this is a real narrowing rather
    /// than an impossible one — but no guest can currently produce it, because
    /// Apple's serializer truncates `baseVertex` to 16 bits in the compact
    /// records. A firing here means a wide record carried something enormous.
    BaseVertexOutOfRange,
}

impl crate::observe::Decline for IndexLoadReason {
    fn slug(&self) -> &'static str {
        match self {
            Self::TypeUnsupported => "draw_index_type_unsupported",
            Self::CountOverflow => "draw_index_count_overflow",
            Self::CountZero => "draw_index_count_zero",
            Self::EntryMissing => crate::observe::ladder_slug!("draw_index", no_list_entry),
            Self::ObjectType => crate::observe::ladder_slug!("draw_index", wrong_type),
            Self::DescRead => crate::observe::ladder_slug!("draw_index", desc_read),
            Self::DescDecode => crate::observe::ladder_slug!("draw_index", desc_decode),
            Self::BackingMissing => "draw_index_backing_missing",
            Self::OffsetOverflow => "draw_index_offset_overflow",
            Self::OutOfBounds => "draw_index_out_of_bounds",
            Self::ReadFail => "draw_index_read_fail",
            Self::BaseVertexOutOfRange => "draw_index_base_vertex_out_of_range",
        }
    }
}

/// Load the render pipeline a draw named, or say why it could not be loaded.
///
/// The sibling of `compute_exec::load_compute_pipeline`, and until now the half
/// of that pair that named none of its five failures: every caller collapses a
/// `None` into one coarse `MissingPipeline`, so a draw that lost its pipeline
/// said only that, on the rail that runs every frame.
///
/// `pipeline_ref == 0` is "no pipeline bound" and stays silent, matching the
/// compute sibling and the rest of the crate — `exec` filters it at both draw
/// call sites and `metal_icb` tests it directly, so nothing reaches here with a
/// zero today. The guard is what keeps that true if one ever does: ref 0 is a
/// valid object-list index, so without it an unbound ref would read entry 0 and
/// then report a rung for it.
///
/// The new lines were measured before being believed: a driven x86/Vulkan boot
/// of **177 746 draws** — one call here each — emitted zero `draw_load_pipeline`
/// lines, and the coarse `MissingPipeline` its callers raise was zero on that
/// same boot. So this is a rail that succeeds, not one that was failing quietly,
/// and a line from it is worth reading. See `runtime::mtlb` for the same
/// measurement on the loader one level down.
///
/// **That zero is per-guest-line, and macOS 26 is not on it.** Every driven
/// macos-26 boot measured so far emits 36-40 fail lines from here, all of them
/// `no_list_entry`, while a driven macos-15 boot of the same binary emits none.
/// The macOS 26 population has a measured mechanism and is not a regression to
/// re-derive: the guest clears the object-list slot it named while the packet
/// that named it is still undrained, so the slot reads zero when this device
/// gets to it. The deduped counters behind those lines run several times higher
/// than the lines themselves, so the two are not interchangeable.
///
/// Read a count here against the same rail's previous boot. Read it against the
/// paragraph above instead and macOS 26's standing behaviour arrives looking
/// like a fresh defect, which is a mistake this doc has already cost once.
pub(crate) fn load_render_pipeline<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    pipeline_ref: u32,
) -> Option<RenderPipelineDescriptor> {
    if pipeline_ref == 0 {
        return None;
    }
    let report = crate::observe::RungReport::new("draw_load_pipeline", "pipe_ref");
    // Live object-list: render pipeline is serializer-object with subtype 0x0e.
    let (_entry, desc) = match objects::resolve_descriptor(
        state,
        host,
        task_id,
        pipeline_ref,
        &[OBJECT_TYPE_SERIALIZER_OBJECT],
    ) {
        Ok(found) => found,
        Err(rung) => {
            report.rung(task_id, pipeline_ref, rung);
            return None;
        }
    };
    let p = match decode_render_pipeline_descriptor(&desc) {
        Ok(p) => p,
        Err(status) => {
            // The decoder's own name for what it refused, carried through rather
            // than collapsed into `desc_decode`. Without it this line said only
            // that a 292-byte descriptor did not decode, and finding out *why*
            // meant correlating its `t=` against an `OFF serializer_object_pipeline_shape`
            // line in the same millisecond — which is how the alpha-test and
            // logic-op tags were found and is not a step the next reader should
            // have to repeat.
            use crate::observe::Decline;
            report.reason(
                task_id,
                pipeline_ref,
                crate::observe::ladder_slug!("", desc_decode),
                &format!("desc_len={} decode={}", desc.len(), status.slug()),
            );
            return None;
        }
    };
    // Both stages are required to build a pipeline, and the two are reported
    // apart because they are different guest mistakes — the compute sibling
    // names its one stage the same way, as `kernel_func_zero`.
    if p.vertex_func_ref == 0 {
        report.reason(task_id, pipeline_ref, "vertex_func_zero", "");
        return None;
    }
    if p.fragment_func_ref == 0 {
        report.reason(task_id, pipeline_ref, "fragment_func_zero", "");
        return None;
    }
    // The guest has created this pipeline object, which is the semantic model's
    // `Declared` and nothing more — no host work has started here, and both
    // rails reach this same door before any of theirs does.
    //
    // After the two zero-stage checks rather than before them: a descriptor
    // naming no vertex or fragment function is not a pipeline the guest can
    // ever bind, and declaring one would put a name in the table that nothing
    // will ever advance or retire.
    if let Some(name) = objects::name_resource(state, host, task_id, pipeline_ref) {
        crate::runtime::drain::note_store_route(if state.declare_pipeline(name) {
            "pipeline_declared"
        } else {
            "pipeline_declared_already"
        });
    } else {
        // The model cannot name the slot, so it cannot hold a pipeline for it
        // either. Counted rather than silent: every one of these is an exec
        // transaction the ordering plane would refuse for a lease it has no
        // entry for.
        crate::runtime::drain::note_store_route("pipeline_declared_unnamed");
    }
    Some(p)
}

/// Name the depth-stencil state `ds_ref` names, from whichever rail just built
/// it.
///
/// One function for two call sites because the two rails load a depth-stencil
/// state separately — `draw::vulkan::load_depth_stencil_descriptor` and
/// `draw::metal::load_depth_stencil_state`, which differ in what they return
/// and in whether they retain it — and the *name* is neither rail's. A boot
/// measured four of these destroys a boot arriving with no name for the model
/// to hold; writing the route pair twice is how the second rail comes to spell
/// it differently and the count stops adding up.
pub(super) fn name_depth_stencil<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    ds_ref: u32,
) {
    objects::note_named_at_construction(
        state,
        host,
        task_id,
        ds_ref,
        "ds_state_model_named",
        "ds_state_model_unnamed",
    );
}

/// Tell the ordering plane what a rail's executor found its shaders do with
/// each bound slot.
///
/// Beside [`advance_pipeline`] and for the same reason: the naming is the
/// rail-neutral half and both rails must report through one function or the
/// counters read differently on each. Called **before** the `Ready` step, which
/// is the order [`DeviceState::publish_pipeline_usage`] requires — a lease
/// taken before the reflection lands gets a footprint that is conservative for
/// the life of that transaction.
pub(crate) fn publish_pipeline_usage<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    pipeline_ref: u32,
    usage: reims_vgpu_core::pipeline::PublishedUsage,
) {
    let Some(name) = objects::name_resource(state, host, task_id, pipeline_ref) else {
        crate::runtime::drain::note_store_route("pipeline_usage_unnamed");
        return;
    };
    crate::runtime::drain::note_store_route(if state.publish_pipeline_usage(name, usage) {
        "pipeline_usage_published"
    } else {
        // The same population `pipeline_advance_declined` describes: a build
        // finishing after the guest's delete, or a rail with no retained
        // pipeline state re-walking one the table has already retired.
        "pipeline_usage_declined"
    });
}

/// Step the pipeline `pipeline_ref` names along its build.
///
/// # Why the rails call this and not the model's door directly
///
/// `load_render_pipeline` tells the ordering plane *that* a pipeline exists —
/// `PipelineState::Declared`, the guest's own fact, and rail-neutral. Building
/// it is the running rail's, and until a rail reports the result an admitted
/// exec that leased the pipeline is parked on a compilation nothing finishes:
/// a hang, which is worse than the `Absent` refusal an empty table gave. So
/// every rail reports, and reports through one function so the counters read
/// the same on both.
///
/// The naming is free after the first sighting — `objects::name_resource`
/// answers from `DeviceState::object_name` and only walks the guest's list when
/// the ref has never been seen, which by the time a pipeline is being built it
/// has, in `load_render_pipeline`.
///
/// `Ready` goes through the model's own door rather than through `advance`,
/// because becoming ready is what releases the work parked on it.
pub(crate) fn advance_pipeline<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    pipeline_ref: u32,
    next: reims_vgpu_core::pipeline::PipelineState,
) {
    use reims_vgpu_core::pipeline::PipelineState;
    let Some(name) = objects::name_resource(state, host, task_id, pipeline_ref) else {
        crate::runtime::drain::note_store_route("pipeline_advance_unnamed");
        return;
    };
    let taken = if next == PipelineState::Ready {
        let taken = state.ready_pipeline(name);
        // The rail's own answer, beside the drain's two. See `ready_lease`.
        if taken && crate::observe::first_sight("pipeline_lease_ready_rail", u64::from(name.slot.0))
        {
            crate::observe::off(format!(
                "pipeline_lease_ready site=pipeline_lease_ready_rail slot={} gen={} task={task_id}",
                name.slot.0, name.generation.0
            ));
        }
        taken
    } else {
        state.advance_pipeline(name, next)
    };
    crate::runtime::drain::note_store_route(if taken {
        match next {
            PipelineState::Translating => "pipeline_translating",
            PipelineState::Compiling => "pipeline_compiling",
            PipelineState::Ready => "pipeline_ready",
            _ => "pipeline_advanced_other",
        }
    } else {
        // Not a defect on its own: a rail with no retained pipeline state
        // re-walks the same pipeline on every draw and finds it already
        // `Ready`. It is also what a build finishing after the guest's delete
        // answers, which is why it is counted rather than dropped.
        "pipeline_advance_declined"
    });
}

/// The rail cannot build this pipeline, and will not be asked to try again.
///
/// Terminal by contract — see `reims_vgpu_core::pipeline` — so a guest
/// re-binding a pipeline this device cannot build produces one refusal rather
/// than one per frame, and the reason survives to whoever reads the failure
/// channel.
pub(crate) fn refuse_pipeline<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    pipeline_ref: u32,
    reason: reims_vgpu_core::pipeline::RefusalReason,
) {
    let Some(name) = objects::name_resource(state, host, task_id, pipeline_ref) else {
        crate::runtime::drain::note_store_route("pipeline_refuse_unnamed");
        return;
    };
    let ended = state.refuse_pipeline(name, reason);
    // The two facts apart. A refusal the table did not take is one that had
    // already ended — refused before, or retired by the guest mid-build — and
    // it is not the same event as a refusal that took and had nobody parked on
    // it, which is every refusal until the cutover admits anything.
    crate::runtime::drain::note_store_route(if ended.took {
        "pipeline_refused"
    } else {
        "pipeline_refuse_already_ended"
    });
    if !ended.stranded.is_empty() {
        crate::runtime::drain::note_store_route_n(
            "pipeline_refuse_stranded",
            ended.stranded.len() as u64,
        );
    }
    if ended.took {
        crate::runtime::drain::note_store_route(reason.slug());
    }
}

/// Resolve buffer object → guest bytes starting at `offset`.
/// Where a buffer object's bytes live in the task GVA space. Both the
/// zero-copy gather and the CPU staging read need identical `(gva, size)`;
/// resolving it once ([`resolve_buffer_backing`]) avoids walking the task page
/// table twice for every sub-zero-copy-floor bind (the `buf_snap` population —
/// ~4.7 CPU snapshots/draw under Safari scroll, each of which previously paid
/// the object-list entry read + descriptor read + decode in the failed ZC
/// attempt *and* again in the CPU fallback).
pub(super) struct BufferBacking {
    pub(super) gva: u64,
    pub(super) size: u64,
}

/// The slug for each way a buffer ref fails to yield a span.
///
/// Five refusals, one per condition, in the vocabulary `observe::ladder`
/// declares — because the five lines this replaced carried **no `reason=` at
/// all**. `AGENTS.md` says the fail log is ranked by `reason=`; a line without
/// one is not in the ranking, and "load_buffer miss lookup" was not findable by
/// the grep that finds every other rail's first rung either.
fn buffer_refusal_slug(refusal: objects::BufferSpanRefusal) -> &'static str {
    match refusal {
        objects::BufferSpanRefusal::Rung(rung) => {
            crate::observe::ladder_slugs!("draw_buffer")(rung)
        }
        objects::BufferSpanRefusal::Decode => {
            crate::observe::ladder_slug!("draw_buffer", desc_decode)
        }
        // Not a rung: the descriptor decoded and names no allocation. The
        // resource exists and has nowhere to read from, which is a different
        // finding from a malformed record — see `observe::ladder`'s own note on
        // what does not belong in the ladder.
        objects::BufferSpanRefusal::NoBacking => "draw_buffer_no_backing",
    }
}

/// The one field each refusal is worth reporting beyond the ref.
///
/// Kept because the five lines this replaced each carried one and losing them
/// would make the consolidation a downgrade: a declared length says whether the
/// entry or the read is wrong, and the page shift says which geometry the
/// backing was computed against.
fn buffer_refusal_detail(refusal: objects::BufferSpanRefusal, page_shift: u32) -> String {
    match refusal {
        objects::BufferSpanRefusal::Rung(objects::LadderRung::WrongType { got }) => {
            format!("ty={got}")
        }
        objects::BufferSpanRefusal::Rung(objects::LadderRung::DescRead { declared_len }) => {
            format!("desc_len={declared_len}")
        }
        objects::BufferSpanRefusal::NoBacking => format!("shift={page_shift}"),
        _ => String::new(),
    }
}

/// Resolve a buffer `ref` to its backing `(gva, size)` (object-list
/// entry read + descriptor read + decode). Fail-visible per failing site —
/// this is the single owner of the `load_buffer *` reason slugs; the ZC and CPU
/// binds delegate to it so a failure logs exactly once, not once per attempt.
fn resolve_buffer_backing<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    buffer_ref: u32,
    resource: Option<&crate::model::TaskResource>,
) -> Option<BufferBacking> {
    if buffer_ref == 0 {
        return None;
    }
    let resolved = match resource {
        Some(resource) => objects::resolve_buffer_span_from_resource(state, resource),
        None => objects::resolve_buffer_span(state, host, task_id, buffer_ref),
    };
    match resolved {
        Ok((gva, size)) => Some(BufferBacking { gva, size }),
        Err(refusal) => {
            crate::observe::fail(format!(
                "load_buffer fail reason={} task={task_id} ref={buffer_ref} {}",
                buffer_refusal_slug(refusal),
                buffer_refusal_detail(refusal, state.page_shift),
            ));
            None
        }
    }
}

/// CPU staging read of a pre-resolved buffer backing at `offset`.
///
/// The one place a buffer's guest bytes are read with this thread, and so the
/// one place the settle belongs. It used to say "no host-store flush — the CPU
/// path has always read the pages as-is (the zero-copy rail owns the flush)",
/// and that stopped being true when the render Store began writing guest pages
/// through the GPU without waiting: a buffer-backed sampled texture
/// ([`load_buffer_texture_rgba`]) whose bytes a Store had just written read the
/// pre-Store frame. The rail above it settled at a fork two calls up
/// ([`seed_color_load`]) and the other three callers settled nowhere.
///
/// # Settling is half the obligation and this arm carried only that half
///
/// Four rails in this crate read a resource's raw guest bytes on the CPU, and
/// each owes the same three terms before it may believe them: the
/// `note_unnamed_reach` census, a payment of whatever the reference names, and
/// the disjointness-narrowed settle. Three of them — the linear sampled read,
/// its memoized twin, and the texture-view read — spell all three. This one
/// spelled the settle alone. All four go through
/// [`crate::runtime::writeback_debt::settle_for_texture`] now, so there is one
/// copy of the rule rather than four.
///
/// The difference is not academic. A settle waits for writes this device has
/// already **submitted**; a writeback debt is a frame it rendered and
/// deliberately did **not** submit, so there is nothing on any queue for the
/// settle to find and it returns immediately with the owed frame still sitting
/// in a host resident. The guest's own bytes are then one Store behind, and the
/// bind that reads them is a sampled texture — an icon, a glyph atlas, a blurred
/// backdrop — which is the shape this failure takes on screen.
///
/// `buffer_ref` is threaded down for exactly this: the payment is by name, and
/// the name is the buffer whose bytes are about to be read.
/// [`load_buffer_texture_rgba`] pays for its texture reference as well, because a
/// buffer-backed texture is two contract references over one allocation and
/// either may be the one a debt was armed under.
///
/// Narrowed on the buffer's own span, so the vertex and index reads that reach
/// here — none of which a render Store ever writes — do not start paying for a
/// wait they never owed.
///
/// `extent_cap` is the shader's proven reach, exactly as
/// `try_buffer_zero_copy_resolved` takes it, and it is not optional polish here.
/// This is where a narrowed bind *lands*: capping the span drops it under the
/// zero-copy floor, so the rail declines and the bind falls through to this
/// read. A cap applied only on the rail above therefore converts a whole-window
/// GPU gather into a whole-window CPU read, which a driven macos-13 boot
/// measured at 11x the bind cost — `binds_us/chain` 2.79 us -> 31.33 us — for a
/// rail whose point was to move fewer bytes. Both arms take the cap or neither
/// does.
fn read_buffer_bytes_resolved<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    buffer_ref: u32,
    backing: &BufferBacking,
    offset: u64,
    extent_cap: Option<u64>,
) -> Option<Vec<u8>> {
    let (gva, size) = (backing.gva, backing.size);
    if offset >= size {
        crate::observe::fail(format!(
            "load_buffer offset oob task={task_id} off={offset} size={size}"
        ));
        return None;
    }
    // The allocation still bounds the read when the two disagree: a declared
    // object larger than what is left of the allocation is the shader and the
    // guest contradicting each other, and only one of them owns these pages.
    let full = size - offset;
    let avail = match extent_cap {
        Some(cap) => full.min(cap),
        None => full,
    };
    if avail < full {
        crate::runtime::drain::note_store_route("cpu_buffer_extent_narrowed");
        crate::runtime::drain::note_store_route_n("cpu_buffer_extent_saved_bytes", full - avail);
    }
    let want = host_alloc_len(avail).filter(|&n| n > 0)?;
    let (read_gva, read_span) = (gva + offset, want as u64);
    // Census, pay, settle — the whole obligation of a CPU read of one named
    // resource's guest bytes. This site used to carry the settle alone, because
    // it held `DeviceState` shared and so *could* not pay; see
    // `writeback_debt::settle_for_texture`, whose doc is about that gap.
    crate::runtime::writeback_debt::settle_for_texture(
        state,
        host,
        task_id,
        buffer_ref,
        read_gva,
        read_span,
        crate::runtime::render_writeback::SettleSite::BufferGuestRead,
    );
    let mut buf = vec![0u8; want];
    // Use device page_shift (x86=12); unshifted helper defaults to arm14 and fails.
    if gva_mem::read_task_gva_by_id(
        host,
        &state.tasks,
        task_id,
        gva + offset,
        &mut buf,
        state.page_shift,
    )
    .is_err()
    {
        crate::observe::fail(format!(
            "load_buffer gva read fail task={task_id} gva={gva:#x}+{offset} want={want} shift={}",
            state.page_shift
        ));
        return None;
    }
    Some(buf)
}

/// Standalone CPU buffer read (non-draw-setup callers): resolve + read.
fn load_buffer_bytes<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    buffer_ref: u32,
    offset: u64,
) -> Option<Vec<u8>> {
    let backing = resolve_buffer_backing(state, host, task_id, buffer_ref, None)?;
    // No shader in scope here — these callers read a buffer outside a draw's
    // bind set, so there is no reflection to bound them and the whole span is
    // the only answer.
    read_buffer_bytes_resolved(state, host, task_id, buffer_ref, &backing, offset, None)
}

/// If `texture_ref` is a texture-view object whose descriptor is a buffer-backed
/// texture (view_opcode 9, `newTextureWithDescriptor:offset:bytesPerRow:`, or
/// its `TextureDescriptor2` form), return its decoded descriptor. `None` for a
/// non-texture-view object or a real texture VIEW (opcode 7/8/0x1b) — those stay on
/// the view path silently.
fn buffer_texture_descriptor<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    texture_ref: u32,
    resource: Option<&crate::model::TaskResource>,
) -> Option<BufferTextureDescriptor> {
    let owned;
    let resource = match resource {
        Some(resource) => resource,
        None => {
            owned = objects::resolve_resource(state, host, task_id, texture_ref).ok()?;
            &owned
        }
    };
    if resource.entry.object_type != OBJECT_TYPE_TEXTURE_VIEW {
        return None;
    }
    let desc_bytes = &resource.descriptor;
    if !matches!(
        texture_view_opcode(desc_bytes),
        Some(TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE) | Some(TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE_WIDE)
    ) {
        return None;
    }
    decode_buffer_texture_descriptor(desc_bytes).ok()
}

/// Say, once per (site, format), that a sampled texture reached the GPU
/// narrower than the guest stored it.
///
/// Every CPU-origin sampled loader here answers in RGBA8, which is exact for
/// the unorm8 and single/dual-channel-8 formats and **lossy** for the float
/// ones: `texel_to_rgba8`'s float arms clamp to `[0,1]` and quantise to 256
/// levels. That is a small visible error for a colour and unbounded data loss
/// for a texture whose texels are not colours — a colour-management LUT, a
/// coordinate pair, a table of offsets a shader walks.
///
/// It has been silent for the whole life of these loaders, because the
/// conversion *succeeds*: nothing downstream can tell a narrowed texel from a
/// native one, and no counter distinguishes a texture that lost precision from
/// one that never had any. So it goes on the fail channel — a degradation this
/// device chose, reported where it is chosen, which is the same rule
/// `frag_neutral_texture_substituted` follows.
///
/// Deduped per (site, format, **extent**) rather than per texture. Per (site,
/// format) was the first shape and it under-reports in the direction that reads
/// as reassuring: a boot narrowing a dozen different textures of one format
/// prints one line, which is indistinguishable from a boot narrowing one. The
/// extent separates those, and it is also the only field that ties a line to a
/// binding in the hang trail, which prints extents and no refs. Still not per
/// texture — a compositor binds thousands a second.
pub(crate) fn note_sampled_narrowing(
    site: &'static str,
    texture_ref: u32,
    fmt: u16,
    w: u32,
    h: u32,
) {
    if !pixel_format::narrows_to_unorm8(fmt) {
        return;
    }
    // Format in the low 16 bits, then the extent. Both dimensions in full:
    // 32x16 and 16x32 are different textures and a hash that folded them would
    // report one.
    let key = u64::from(fmt) | (u64::from(w) << 16) | (u64::from(h) << 40);
    if !crate::observe::first_sight(site, key) {
        return;
    }
    crate::observe::fail(format!(
        "sampled_texture_narrowed reason={site} ref={texture_ref} fmt={fmt:#x} {w}x{h} \
         to=rgba8 lost=clamp_to_unit_and_256_levels"
    ));
}

/// What rendering a Store's colour target at `RGBA8Unorm` does to the format
/// the guest declared for its destination.
///
/// # Why this exists at all
///
/// The Metal rail creates **every** colour render target as `RGBA8Unorm`. The
/// destination mapping's declared format is available at the same call site and
/// is not forwarded: `ColorRt::pixel_format` is a literal `0`, which
/// `backend::metal::render` reads as "the writeback format". That is a policy,
/// and until this type it was a constant with a comment.
///
/// It is not a harmless one. A driven macos-13 boot reports the window server's
/// main compositing surface as `MTLPixelFormatRGBA16Float`, so the guest's
/// half-float frame is rendered into an eight-bit attachment, read back as
/// unorm8, and expanded again to half-float on the way into the guest's pages —
/// where it arrives with 256 levels per channel and everything above 1.0 gone.
///
/// The Vulkan rail had exactly this defect and fixed it;
/// `backend::vulkan::present_identity::surface_identity`'s doc states the
/// finding in its own words: *"Ignoring the declaration renders a guest's
/// half-float compositing into an eight-bit image and quantizes it with nothing
/// to say so — the loss is invisible because every rail downstream still works,
/// which is how the same bug survived in the `Gva` namespace until a counter on
/// an unrelated gate exposed it."*
///
/// "With nothing to say so" is the part this type changes. `AGENTS.md` requires
/// degraded guest work to produce a typed reason on the always-on failure
/// channel, and the store direction had none — [`note_sampled_narrowing`] is
/// the load direction's and has no counterpart. Naming the decision does not
/// fix it; it makes the size of it countable, and it gives the fix one place to
/// happen.
#[cfg(any(test, all(feature = "backend-metal", target_os = "macos")))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ColorTargetNarrowing {
    /// The guest declared an eight-bit colour order, so an `RGBA8Unorm`
    /// attachment carries it exactly and the writeback conversion is a channel
    /// permutation.
    None,
    /// The guest declared a store layout this rail does not render in. Carries
    /// it, because "half-float" and "ten-bit packed" and "integer" are three
    /// different losses and a single slug would report them as one.
    Quantised(pixel_format::TexelLayout),
    /// The guest's declaration has no store layout in this contract at all, so
    /// what is lost cannot be named — only that the rail chose for it.
    Undeclared,
}

/// Read the declaration for a Store destination.
///
/// Asked of the **guest's** declared format, which is the only place the width
/// it wanted is still known: past the attachment every texel is four unorm8
/// bytes and nothing downstream can tell a narrowed target from a native one.
/// That is [`pixel_format::narrows_to_unorm8`]'s argument one stage earlier in
/// the pipeline.
#[cfg(any(test, all(feature = "backend-metal", target_os = "macos")))]
pub(crate) fn color_target_narrowing(declared_format: u16) -> ColorTargetNarrowing {
    use pixel_format::TexelLayout;
    match pixel_format::store_texel_order(declared_format) {
        // The two eight-bit colour orders. `RGBA8Unorm` holds either exactly;
        // which of the two the guest picked changes the byte order of the
        // writeback and not what survives it.
        Some(TexelLayout::Rgba8 | TexelLayout::Bgra8) => ColorTargetNarrowing::None,
        Some(layout) => ColorTargetNarrowing::Quantised(layout),
        None => ColorTargetNarrowing::Undeclared,
    }
}

/// Count, and describe once, a colour target this rail rendered narrower than
/// the guest declared.
///
/// The store-direction mirror of [`note_sampled_narrowing`], and deduped the
/// same way and for the same measured reason: per (format, extent), because a
/// boot narrowing one surface and a boot narrowing a dozen of one format print
/// the same single line otherwise.
///
/// The count is unconditional and the line is once, so a reader gets the size of
/// the loss from the census and its shape from the log.
#[cfg(any(test, all(feature = "backend-metal", target_os = "macos")))]
pub(crate) fn note_store_narrowing(declared_format: u16, w: u32, h: u32) {
    let (slug, detail) = match color_target_narrowing(declared_format) {
        ColorTargetNarrowing::None => return,
        ColorTargetNarrowing::Quantised(layout) => (
            "store_target_narrowed",
            format!("declared={layout:?} lost=clamp_to_unit_and_256_levels"),
        ),
        ColorTargetNarrowing::Undeclared => (
            "store_target_undeclared",
            "declared=none lost=unnameable".to_string(),
        ),
    };
    crate::runtime::drain::note_store_route(slug);
    let key = u64::from(declared_format) | (u64::from(w) << 16) | (u64::from(h) << 40);
    if !crate::observe::first_sight(slug, key) {
        return;
    }
    crate::observe::fail(format!(
        "store_target_narrowed reason={slug} fmt={declared_format:#x} {w}x{h} \
         rendered=rgba8unorm {detail}"
    ));
}

/// Load an opcode-9 buffer-backed texture as tight RGBA8 (width, height, bytes).
///
/// The sampled bytes are the source MTLBuffer's guest storage read at `offset`
/// with `bytes_per_row` stride and reinterpreted through the embedded texture
/// descriptor's pixel format. Only fires on a genuine buffer-texture object, so
/// every early-return here logs a fail-visible reason (the buffer is unresolved,
/// the format is unknown, or the span overruns the buffer) — those are real
/// dropped-draw causes, not speculative "not ready yet" polls.
fn load_buffer_texture_rgba<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    bt: &BufferTextureDescriptor,
) -> Option<(u32, u32, Vec<u8>)> {
    let (w, h) = (bt.desc.width, bt.desc.height);
    if w == 0 || h == 0 {
        crate::observe::fail(format!(
            "buftex zero_geom ref={texture_ref} buf={} {}x{}",
            bt.buffer_ref, w, h
        ));
        return None;
    }
    let fmt = if bt.desc.pixel_format != 0 {
        bt.desc.pixel_format
    } else {
        MTL_FORMAT_BGRA8_UNORM
    };
    let Some(tight) = pixel_format::tight_row_bytes(w, fmt) else {
        crate::observe::fail(format!(
            "buftex unknown_fmt ref={texture_ref} buf={} fmt={fmt:#x} {w}x{h}",
            bt.buffer_ref
        ));
        return None;
    };
    // A guest bytesPerRow of 0 means tight rows (single-row / API default).
    let bpr = if bt.bytes_per_row == 0 {
        tight as u64
    } else {
        bt.bytes_per_row
    };
    if bpr < tight as u64 {
        crate::observe::fail(format!(
            "buftex bpr_short ref={texture_ref} buf={} bpr={bpr} tight={tight} {w}x{h} fmt={fmt:#x}",
            bt.buffer_ref
        ));
        return None;
    }
    let span = bpr.checked_mul(h as u64)?;
    // A buffer-backed texture is two contract references over one allocation:
    // the texture-view texture object the guest binds and samples, and the buffer
    // buffer that owns the storage. A synchronize names the former and a debt
    // may be armed under either, so both are paid. `load_buffer_bytes` below
    // pays for `bt.buffer_ref`; this is the sibling call every other sampled
    // rail makes, and its absence here is what let a rendered frame stay in a
    // host resident while this read served the guest the frame before it.
    crate::runtime::writeback_debt::pay_for_texture(state, host, task_id, texture_ref);
    let raw = load_buffer_bytes(state, host, task_id, bt.buffer_ref, bt.offset)?;
    if (raw.len() as u64) < span {
        crate::observe::fail(format!(
            "buftex span_oob ref={texture_ref} buf={} off={} bpr={bpr} span={span} avail={} {w}x{h}",
            bt.buffer_ref,
            bt.offset,
            raw.len()
        ));
        return None;
    }
    note_sampled_narrowing("buftex_narrowed", texture_ref, fmt, w, h);
    let Some(row_rail) = pixel_format::RowToRgba8::for_format(fmt) else {
        crate::observe::fail(format!(
            "buftex convert_unsupported ref={texture_ref} buf={} fmt={fmt:#x} {w}x{h}",
            bt.buffer_ref
        ));
        return None;
    };
    let row_pixels = w as usize;
    let dst_row = row_pixels.checked_mul(RGBA8_BPP as usize)?;
    let mut rgba = vec![0u8; dst_row.checked_mul(h as usize)?];
    let tight = tight as usize;
    let bpr = bpr as usize;
    for y in 0..h as usize {
        let src = &raw[y * bpr..y * bpr + tight];
        let dst = &mut rgba[y * dst_row..(y + 1) * dst_row];
        if !row_rail.convert(src, w, dst) {
            crate::observe::fail(format!(
                "buftex convert_fail ref={texture_ref} buf={} fmt={fmt:#x} row={y} {w}x{h}",
                bt.buffer_ref
            ));
            return None;
        }
    }
    Some((w, h, rgba))
}

fn index_elem_size(index_type: u32) -> Option<usize> {
    match index_type {
        0 => Some(2), // MTLIndexTypeUInt16
        1 => Some(4), // MTLIndexTypeUInt32
        _ => None,
    }
}

/// Resolve an indexed draw to the guest allocation and exact byte window its
/// index count names. Reading those bytes is a backend choice: Metal's copied
/// upload path consumes them on the CPU, while Vulkan retains this resource
/// window and lets vertex input consume it when the command executes.
fn resolve_index_window_reason<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    info: &IndexedDrawInfo,
) -> Result<(BufferBacking, usize), IndexLoadReason> {
    use IndexLoadReason as R;
    let elem = index_elem_size(info.index_type).ok_or(R::TypeUnsupported)?;
    let need = (info.index_count as usize)
        .checked_mul(elem)
        .ok_or(R::CountOverflow)?;
    if need == 0 {
        return Err(R::CountZero);
    }
    let (gva, size) = objects::resolve_buffer_span(state, host, task_id, info.index_buffer_ref)
        .map_err(|refusal| match refusal {
            objects::BufferSpanRefusal::Rung(
                objects::LadderRung::NoListEntry | objects::LadderRung::NoTaskSpace,
            ) => R::EntryMissing,
            objects::BufferSpanRefusal::Rung(objects::LadderRung::WrongType { .. }) => {
                R::ObjectType
            }
            objects::BufferSpanRefusal::Rung(objects::LadderRung::DescRead { .. }) => R::DescRead,
            objects::BufferSpanRefusal::Decode => R::DescDecode,
            objects::BufferSpanRefusal::NoBacking => R::BackingMissing,
        })?;
    let end = info
        .index_buffer_offset
        .checked_add(need as u64)
        .ok_or(R::OffsetOverflow)?;
    if end > size {
        return Err(R::OutOfBounds);
    }
    Ok((BufferBacking { gva, size }, need))
}

/// Load the index bytes a bound indexed draw references, returning the **specific**
/// reason on failure. Metal emits it directly; Vulkan delegates it through
/// `DrawPreparationDecline::IndexLoad`, so both rails keep one reason vocabulary.
/// Runs on the drain worker (off main core); only reached when `req.indexed` is
/// set, so it cannot flood a 2D-UI boot.
#[cfg(any(test, all(feature = "backend-metal", target_os = "macos")))]
fn load_index_bytes_reason<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    info: &IndexedDrawInfo,
) -> Result<Vec<u8>, IndexLoadReason> {
    use IndexLoadReason as R;
    let (backing, need) = resolve_index_window_reason(state, host, task_id, info)?;
    let mut buf = vec![0u8; need];
    gva_mem::read_task_gva_by_id(
        host,
        &state.tasks,
        task_id,
        backing.gva + info.index_buffer_offset,
        &mut buf,
        state.page_shift,
    )
    .map_err(|_| R::ReadFail)?;
    Ok(buf)
}

/// Guest Store seed for mapper-ref-texture `image_changed` / GVA partial writeback.
///
/// Metal `storeAction=Store` writes the **whole** attachment after the pass.
/// Diff-only writeback is Store-equivalent only when `loadAction=Load` and
/// `load_seed` was the pre-pass guest content (unchanged texels match guest).
/// After Clear / DontCare the Metal RT holds clear (or undefined) + drawn
/// coverage: seed must be `None` so clear regions overwrite prior guest pixels
/// outside the scissor. `force_full_store` (multi-draw final) always full-writes.
///
/// Without this, Clear+partial scissor left boot-logo / wallpaper under window
/// chrome on the lagging dual-mid (seed=clear skipped outside-scissor rows).
#[cfg(any(test, all(feature = "backend-metal", target_os = "macos")))]
pub(crate) fn store_seed_policy(
    force_full_store: bool,
    load_action: u16,
    load_seed: Option<&[u8]>,
) -> Option<&[u8]> {
    if force_full_store || load_action != MTL_LOAD_ACTION_LOAD {
        None
    } else {
        load_seed
    }
}

/// Premultiplied `src` over `dst` with Metal factors **One / OneMinusSrcAlpha**,
/// in software.
///
/// When color0 blend is One/OneMinusSrcAlpha, the attachment Load composite is
/// `src + dst*(1 - src.a)`. A fully transparent fragment leaves the seed
/// untouched and an opaque one replaces it; only the partial alphas mix, which
/// is what `blended_texels` counts. Returns `(pixels, blended_texels)`.
///
/// **The product path does not call this** — the hardware does Load+blend — and
/// its two unit tests only check it against hand-written constants, so it reads
/// as dead on both of the obvious checks. It is not.
/// `premult_one_omsa_gpu_blend_matches_software_oracle` in
/// `tests/vk_engine_parity.rs` runs the real GPU blend and asserts it agrees
/// with this function to within 1 LSB, which makes this the only independent
/// statement of what that blend is supposed to compute. Deleting it deletes the
/// check, not the duplication.
pub fn load_composite_premult_one_omsa(draw_rgba: &[u8], seed_rgba: &[u8]) -> (Vec<u8>, usize) {
    if draw_rgba.len() != seed_rgba.len() || !draw_rgba.len().is_multiple_of(4) {
        return (draw_rgba.to_vec(), 0);
    }
    let mut out = vec![0u8; draw_rgba.len()];
    let mut blended = 0usize;
    for ((o, s), d) in out
        .chunks_exact_mut(4)
        .zip(draw_rgba.chunks_exact(4))
        .zip(seed_rgba.chunks_exact(4))
    {
        let sa = s[3] as u32;
        if sa == 0 {
            o.copy_from_slice(d);
            blended += 1;
        } else if sa >= 255 {
            o.copy_from_slice(s);
        } else {
            // out = src + dst * (1 - sa/255)  (integer, rounded)
            let inv = 255 - sa;
            for i in 0..4 {
                let v = s[i] as u32 + ((d[i] as u32 * inv) + 127) / 255;
                o[i] = v.min(255) as u8;
            }
            blended += 1;
        }
    }
    (out, blended)
}

/// Whether a decoded load action is one of the three `MTLLoadAction` values,
/// reporting the one case where it is not.
///
/// A fourth value is a corrupt or unsupported wire word, and both encode arms
/// treat it as DontCare — which discards whatever the attachment held, so a
/// pass the guest meant to composite onto goes blank. Only the Metal arm said
/// so; the Vulkan arm took the same value into a `_ => {}`.
#[cfg(any(
    feature = "backend-vulkan",
    all(feature = "backend-metal", target_os = "macos")
))]
pub(crate) fn load_action_in_contract(pipeline_ref: u32, load_action: u16) -> bool {
    if is_declared_load_action(load_action) {
        return true;
    }
    if degrade_log_first(pipeline_ref, "load_action_unmapped") {
        crate::observe::fail(format!(
            "pass_state_degraded reason=load_action_unmapped \
             pipe={pipeline_ref} load_action={load_action} \
             (not one of MTLLoadAction 0/1/2; attachment treated as DontCare)"
        ));
    }
    false
}

/// Whether a decoded store action is one of the named values this wire form
/// carries, reporting an unknown value.
///
/// The sibling of [`load_action_in_contract`], and it was missing while that one
/// existed — the two fields are decoded from adjacent words of the same
/// attachment prefix, so a decode that misreads one misreads the other, and only
/// half of that was visible.
///
/// Recognizing a value is not backend authorization. The Vulkan request builder
/// implements resolve-only for the supported shape and names every other
/// resolve action as a typed refusal; the direct-Metal path likewise refuses
/// before encoding until it carries the corresponding attachment lifecycle.
#[cfg(any(
    feature = "backend-vulkan",
    all(feature = "backend-metal", target_os = "macos")
))]
pub(crate) fn store_action_in_contract(pipeline_ref: u32, store_action: u16) -> bool {
    if is_declared_store_action(store_action) {
        return true;
    }
    if degrade_log_first(pipeline_ref, "store_action_unmapped") {
        crate::observe::fail(format!(
            "pass_state_degraded reason=store_action_unmapped \
             pipe={pipeline_ref} store_action={store_action} \
             (not one of the represented MTLStoreAction values 0/1/2/3; \
              attachment result may be dropped)"
        ));
    }
    false
}

/// Fail-visible diagnosis when a bound sample ref does not materialize.
///
/// Kept off the success path; only called after a sampled resolver
/// (`resolve_sampled_source` on the engine path, `load_sampled_rgba` on the
/// Metal path) returns None.
fn sample_miss_detail<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &M,
    task_id: u32,
    texture_ref: u32,
) -> String {
    if texture_ref == 0 {
        return "reason=zero_ref".into();
    }
    let Some(entry) = objects::lookup_list_entry(state, host, task_id, texture_ref) else {
        return "reason=no_list_entry".into();
    };
    let ot = entry.object_type;
    let desc_len = entry.descriptor_length;
    match ot {
        objects::OBJECT_TYPE_REF_TEXTURE => {
            match objects::read_descriptor(state, host, task_id, &entry) {
                None => format!("type=5 desc_len={desc_len} reason=no_desc"),
                Some(d) if reims_vgpu_wire::device_desc::ref_texture_header(&d).is_err() => {
                    format!("type=5 desc_len={desc_len} reason=short_desc")
                }
                Some(d) => {
                    let sid = reims_vgpu_wire::device_desc::ref_texture_header(&d)
                        .map(|h| h.surface_id.get())
                        .unwrap_or(0);
                    match objects::decode_ref_texture_view(&d) {
                        Some(view) => format!(
                            "type=5 desc_len={desc_len} surface_id={sid} view={}x{} fmt={:#x} reason=ref_texture_view",
                            view.width, view.height, view.pixel_format
                        ),
                        None => format!(
                            "type=5 desc_len={desc_len} surface_id={sid} reason=ref_texture_no_view"
                        )}
                }
            }
        }
        OBJECT_TYPE_MAPPER_REF_TEXTURE => {
            let Some(mid) = objects::resolve_mapper_ref_texture(state, host, task_id, texture_ref)
            else {
                return format!("type=11 desc_len={desc_len} reason=mapper_ref_texture_resolve");
            };
            match state.mappings.get(&mid) {
                None => format!("type=11 mid={mid} desc_len={desc_len} reason=no_mapping"),
                Some(m) => format!(
                    "type=11 mid={mid} desc_len={desc_len} geom={} {}x{} fmt={:#x} mapped={} pages={} reason=mapper_ref_texture_sample",
                    m.has_geom as u8,
                    m.width,
                    m.height,
                    m.format,
                    m.mapped as u8,
                    m.page_entries.len()
                )}
        }
        OBJECT_TYPE_TEXTURE_VIEW => {
            // Opcode-9 buffer-backed textures share the texture-view tag but are not views.
            if let Some(bt) = buffer_texture_descriptor(state, host, task_id, texture_ref, None) {
                return format!(
                    "type=8 desc_len={desc_len} buf={} off={} bpr={} {}x{} fmt={:#x} reason=buftex_load",
                    bt.buffer_ref,
                    bt.offset,
                    bt.bytes_per_row,
                    bt.desc.width,
                    bt.desc.height,
                    bt.desc.pixel_format
                );
            }
            match resolve_texture_view_reasoned(state, host, task_id, texture_ref) {
                Err(why) => {
                    crate::observe::Emit::decline("sample_view_resolve", &why)
                        .field("task", task_id)
                        .field("ref", texture_ref)
                        .fail_once(texture_ref as u64);
                    format!(
                        "type=8 desc_len={desc_len} reason=view_resolve view_reason={}",
                        why.slug()
                    )
                }
                Ok(view) => format!(
                    "type=8 desc_len={desc_len} base={} level={} fmt_ov={:?} reason=view_base_or_swizzle",
                    view.base_texture_ref,
                    view.level,
                    view.pixel_format
                )}
        }
        OBJECT_TYPE_TEXTURE | OBJECT_TYPE_TEXTURE_GENERATE_MIPMAPS => {
            let Some(desc_bytes) = objects::read_descriptor(state, host, task_id, &entry) else {
                return format!("type={ot} desc_len={desc_len} reason=desc_read");
            };
            match decode_texture_descriptor(&desc_bytes) {
                Err(_) => format!("type={ot} desc_len={desc_len} reason=desc_decode"),
                Ok(tex) => {
                    let l0 = tex.level(0);
                    format!(
                        "type={ot} desc_len={desc_len} has_fmt={} fmt={:#x} mips={} handle={:#x} alloc={} L0={}x{} bpr={} reason=linear_sample",
                        u8::from(tex.declared_pixel_format().is_some()),
                        tex.pixel_format,
                        tex.mipmap_level_count,
                        tex.handle,
                        tex.allocation_size,
                        l0.map(|l| l.width).unwrap_or(0),
                        l0.map(|l| l.height).unwrap_or(0),
                        l0.map(|l| l.row_stride).unwrap_or(0),
                    )
                }
            }
        }
        other => format!("type={other} desc_len={desc_len} reason=unsupported_object_type"),
    }
}

/// What the guest says a mapper-ref-texture mapping's texel **values** are, seen through an
/// optional texture-view format.
///
/// Distinct from the byte *order* its loaders hand back, and that distinction is
/// the whole point. `scanout::read_mapping_bgra8` normalises a mapping's channel
/// order to BGRA8 and touches no value, so those loaders key their convert on
/// BGRA8 — correct for order, and silent about the transfer function. This is
/// the answer that is not silent about it, and it is what a sampled bind pairs
/// with the layout in a [`SampledByteFormat`].
///
/// # Total on purpose
///
/// It answers a `u16` and cannot decline, because the only question asked of the
/// result is [`pixel_format::is_srgb`], which is total over `u16`. An earlier
/// draft ran the answer through [`effective_view_sample_format`] and inherited
/// its `Option`: a mapping declaring a format outside the bytes-per-pixel table
/// would then have failed the *bind*, losing guest work over a colour-space
/// question that has a correct answer for every value. Whether a view may
/// reinterpret a base at all is a different question with a different refusal,
/// and it belongs to the loaders that already ask it — asking it twice is how
/// two copies of one rule come to disagree.
///
/// A mapping this device holds no entry for has declared nothing, and
/// [`crate::runtime::mapping_write::mapping_store_format`] already owns what
/// "nothing declared" resolves to; a default entry is handed to it rather than
/// that answer being spelled a second time here.
fn mapping_declared_format(
    state: &DeviceState,
    mapping_id: u32,
    format_override: Option<u16>,
) -> u16 {
    use crate::runtime::mapping_write::mapping_store_format;
    if let Some(view) = format_override {
        return view;
    }
    match state.mappings.get(&mapping_id) {
        Some(entry) => mapping_store_format(entry),
        // Nothing declared. An entry that has latched no geometry is exactly
        // that case, so the owning rule answers it rather than a default being
        // named a second time here.
        None => mapping_store_format(&crate::model::MappingEntry::default()),
    }
}

/// Sample a mapper-ref-texture mapping as tight RGBA8 from guest pages.
///
/// Guest pages ARE the surface content: the CPU writeback lands Stores in them
/// and guest CPU writes are immediately visible. There is exactly one source;
/// no recovery ranking exists.
///
/// The resolve runs *before* the geometry read, not after. A mapping can be
/// mapped with a live `MappingInternal` and no latched W×H yet; resolving first
/// decodes the guest device-surface descriptor and latches the geometry, so the
/// sample succeeds instead of bailing out on `!has_geom` and dropping the bind.
fn load_mapper_ref_texture_mapping_rgba<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    format_override: Option<u16>,
) -> Option<(u32, u32, Vec<u8>)> {
    let _ = mapper::ensure_resolved_for_scanout(state, host, mapping_id);
    let (w, h) = {
        let m = state.mappings.get(&mapping_id)?;
        if !m.has_geom || m.width == 0 || m.height == 0 {
            return None;
        }
        (m.width, m.height)
    };
    let base_fmt = MTL_FORMAT_BGRA8_UNORM;
    let sample_fmt = effective_view_sample_format(base_fmt, format_override)?;
    let stride = w.saturating_mul(RGBA8_BPP);
    let mut raw = vec![0u8; (stride as usize).saturating_mul(h as usize)];
    if !crate::runtime::scanout::read_mapping_bgra8(state, host, mapping_id, &mut raw, stride, w, h)
    {
        return None;
    }
    let row_rail = pixel_format::RowToRgba8::for_format(sample_fmt)?;
    let mut rgba = vec![0u8; raw.len()];
    for y in 0..h as usize {
        let off = y * (stride as usize);
        let row = &raw[off..off + (w as usize) * 4];
        let dst = &mut rgba[off..off + (w as usize) * 4];
        if !row_rail.convert(row, w, dst) {
            return None;
        }
    }
    Some((w, h, rgba))
}

/// Store encode RGBA8 into **texture_ref** host cache as BGRA (not surface_id).
#[cfg(test)]
fn host_cache_store_rgba8(
    state: &mut DeviceState,
    task_id: u32,
    texture_ref: u32,
    width: u32,
    height: u32,
    rgba: &[u8],
) {
    if texture_ref == 0 || width == 0 || height == 0 {
        return;
    }
    let need = (width as usize)
        .saturating_mul(height as usize)
        .saturating_mul(4);
    if rgba.len() < need {
        return;
    }
    let bgra = swap_rb_channels(&rgba[..need]);
    crate::runtime::surface_cache::store_texture(
        state,
        task_id,
        texture_ref,
        width,
        height,
        bgra,
        0,
    );
}

/// Advance the guest-visible publish milestones for a mapper-ref-texture Store whose
/// pixels have landed in the mapping's guest pages.
///
/// Route-independent: the synchronous `cpu_portability` Store calls it inline,
/// and the resident render Store calls it from the writeback that performs the
/// same write. Both have just proved
/// the same thing — `write_rgba8_image_changed` verified geometry and landed a
/// complete frame — and without it the `present_unbacked` gate is structurally
/// dead on whichever route skips it, because no mapping's `dense_frame_seq`
/// would advance.
pub(crate) fn publish_surface_store<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    width: u32,
    height: u32,
    format: u16,
) {
    crate::backend::selected().note_plane_store_published(mapping_id);
    state.note_surface_composite(mapping_id);
    state.note_dense_frame_published(mapping_id, width, height);
    crate::runtime::scanout::note_front_buffer_writeback(
        state, host, mapping_id, width, height, format,
    );
}

/// Which of the three chain breaks sent a packet to the recovery rail.
///
/// `land_chain_before_abandon`'s doc has always named these three, and each one
/// emits its own line where it is decided. That was not enough to read a boot:
/// those lines dedupe per pipeline (`fail_once`) while the recovery does not, so
/// a driven macOS 26 boot shows 32 recoveries against 10 candidate causes and no
/// way to pair them. Carrying the cause into the recovery line makes the
/// expensive event name its own origin, which is the only form of it that
/// survives `first_sight`.
///
/// Ordinal-free on purpose: this is a label, never a wire value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChainAbandonCause {
    /// An intermediate record encoded `Ok` and returned no colour0, so every
    /// later draw in the packet would composite against a missing seed.
    NoColor0,
    /// The `NoMetal` carrier — this build has no host encode path for the
    /// record. On the Vulkan arm this is where `executeCommandsInBuffer:` and
    /// the other Metal-only records land.
    NoMetal,
    /// A typed terminal refusal from encode, already named by
    /// `note_draw_encode_fail`.
    TerminalRefusal,
}

impl ChainAbandonCause {
    /// The `cause=` token. Stable text: it is grepped out of boot logs.
    pub fn tag(self) -> &'static str {
        match self {
            Self::NoColor0 => "no_color0",
            Self::NoMetal => "no_metal",
            Self::TerminalRefusal => "terminal_refusal",
        }
    }
}

pub fn writeback_chain_rgba<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    color_slots: &[(u32, crate::runtime::render_pass::ColorAttachment)],
    rgba: &[u8],
    cause: ChainAbandonCause,
) -> bool {
    // This whole function is the recovery rail for an abandoned chain, so a
    // refusal here is the last frame being lost outright. Every arm names
    // itself: `let _ = writeback_chain_rgba(..)` is how both callers invoke it,
    // and `dirty_color_targets` advances the content generation on the next line
    // regardless — so a silent `false` leaves pages stale while the device
    // reports them fresh, which is the class `land_chain_before_abandon` exists
    // to prevent.
    let lost = |why: &'static str| -> bool {
        crate::runtime::drain::note_store_route("chain_land_refused");
        crate::observe::fail(format!(
            "writeback_chain_rgba fail reason={why} cause={} task={task_id} slots={} bytes={} \
             (the abandoned chain's last frame is not landing; guest pages keep stale bytes)",
            cause.tag(),
            color_slots.len(),
            rgba.len()
        ));
        false
    };
    if color_slots.is_empty() || rgba.is_empty() {
        return lost("no_source");
    }
    let Some((_, att)) = color_slots.first() else {
        return lost("no_color_slot");
    };
    if att.texture_ref == 0 {
        return lost("unbound_texture_ref");
    }
    let Some(ResolvedRenderTarget {
        mapping_id,
        target_gva: gva,
        width: w,
        height: h,
        row_stride: bpr,
        format: fmt,
        sample_count: _,
    }) = lookup_render_target(state, host, task_id, *att)
    else {
        return lost("render_target_unresolved");
    };
    let need = (w as usize).saturating_mul(h as usize).saturating_mul(4);
    if rgba.len() < need {
        return lost("readback_short");
    }
    if gva != 0 {
        // The refusal is carried out, not collapsed. `write_gva_rgba8`'s own doc
        // asks for exactly this — "a caller has to be able to tell 'the guest
        // tore this target down' from a write that genuinely lost content" — and
        // `MemError` already names all of its refusals, so `.is_ok()` was
        // throwing away the one word that distinguishes them.
        return match write_gva_rgba8(state, host, task_id, gva, w, h, bpr, fmt, rgba) {
            Ok(()) => true,
            Err(e) => {
                crate::runtime::drain::note_store_route("chain_land_refused");
                crate::observe::Emit::decline("writeback_chain_rgba", &e)
                    .field("task", task_id)
                    .field("gva", format!("{gva:#x}"))
                    .field("dims", format!("{w}x{h}"))
                    .field("bpr", bpr)
                    .field("fmt", format!("{fmt:#x}"))
                    .fail();
                false
            }
        };
    }
    if mapping_id == 0 {
        return lost("no_mapping_and_no_gva");
    }
    // An abandoned portability chain must still preserve the last successful
    // record. This is an error recovery rail, not normal product behavior: land
    // the resident readback into the mapper-ref-texture mapping, publish the Composite
    // Store, and keep the degradation fail-visible.
    crate::observe::fail(format!(
        "writeback_chain_rgba reason=resident_chain_abandoned_cpu_recovery \
         cause={} mid={mapping_id} {w}x{h} fmt={fmt:#x}",
        cause.tag()
    ));
    let wrote = mapping_write::write_rgba8_image_changed(
        state,
        host,
        mapping_id,
        rgba,
        None,
        w,
        h,
        mapping_write::FramePublication::HostCache,
    );
    if wrote {
        publish_surface_store(state, host, mapping_id, w, h, fmt);
    }
    wrote
}

/// The guest bytes one GVA render target occupies, as the rails that ask about
/// it name them.
///
/// One value rather than five parameters because the five only mean anything
/// together — a stride belongs to a height, and a format decides the channel
/// order the registry keys a resident on — and because two callers assembling
/// the same five by hand is how they come to disagree about one of them.
#[derive(Clone, Copy, Debug)]
pub struct GvaSpan {
    pub texture_ref: u32,
    pub gva: u64,
    pub row_stride: u32,
    pub width: u32,
    pub height: u32,
    /// The guest's declared pixel format, not a host one:
    /// the rail that keys a resident on it turns it into the `format` half of
    /// that key.
    pub format: u16,
}

// --- The record for a sample count taken from the pipeline -----------------
//
// Neutral, and here rather than in a rail, because every line of it is decoded
// guest state and `crate::observe`: the three counts, the resolve reference,
// the geometry. A rail that does not consult the pipeline hands `pipeline:
// None` and the emitter returns without a word, which is the same silence the
// `cfg` used to produce and one the compiler no longer has to be told about.

/// The three sample counts in play when a colour attachment is resolved, named
/// so the record below cannot transpose two of them.
pub struct AttachmentSampleCounts {
    /// `MTLRenderPipelineDescriptor.rasterSampleCount` of the bound pipeline,
    /// or `None` when the pipeline could not be resolved.
    pub pipeline: Option<u32>,
    /// What [`super::render_target::ResolvedRenderTarget`] carried.
    ///
    /// **Not the texture's creation sample count**, and reading it as one is a
    /// mistake this record has already caused once. That field is a hardcoded
    /// `1` at every one of its construction sites, because a linear texture
    /// resource's dimensions do not retain the creation descriptor's sample
    /// count — the field's own documentation says so, and says the Vulkan
    /// encode is expected to replace the provisional value with the pipeline's.
    ///
    /// So `pipeline != target` is *not* evidence that the guest's texture is
    /// single-sample. It is only evidence that the pipeline declared more than
    /// one sample, which is the case worth naming here for the reason below.
    pub target: u32,
    /// What this device gave the attachment. Today: the pipeline's, when it has
    /// one.
    pub resolved: u32,
}

/// Report a colour attachment whose sample count this device took from the
/// **pipeline** while the destination texture declared a different one.
///
/// # Why this needs a record
///
/// Metal requires a pipeline's `rasterSampleCount` to equal the sample count of
/// every colour attachment it renders into, and this device recovers that count
/// from the pipeline because the resolved target cannot carry it (see
/// [`AttachmentSampleCounts::target`]). So this record does not report a
/// disagreement between the guest's two declarations — it cannot see the
/// texture's declaration at all. What it reports is the passes that end up
/// multisampled, and where their samples are meant to go.
///
/// That matters because it has a downstream cost the site cannot see. The
/// engine creates the resident at the promoted count, the draw succeeds, and
/// then `resident_read_snapshot` refuses to read a `sample_count != 1` resident
/// back — so nothing is stored, `runtime::exec::finish_stream` applies the
/// pass's clear, and a rendered tile reaches the guest as a flat colour. On
/// the Vulkan rail that is measured at twice per boot on 300x300 targets, and until
/// the skipped-draw tail was corrected it was reported as an engine refusal
/// that never happened.
///
/// Measured on rail macos-15, boot s4: **two** records in a whole boot, both
/// `pipeline_samples=4 resolve_ref=0 store=0x1` on 300x300 linear GVA targets,
/// and they are the same two passes the corrected skipped-draw tail reports as
/// `engine_drew_store_lost_it`. Two out of a boot's several hundred pipelines
/// is also what says the pipeline's count is decoded correctly rather than
/// misread: `raster_sample_count` comes from a TLV tag, and a misread tag would
/// not be this rare.
///
/// So the guest really does run a 4x pass here, and the open question is no
/// longer "who invented the multisample" — it is **what the guest expects to
/// find in those guest pages afterwards**. Metal writes nothing to a linear
/// buffer for a multisample `MTLStoreActionStore`; this device writes the
/// pass's clear colour there. Neither this record nor any other establishes
/// which the guest reads, and until one does, no repair here is supportable.
///
/// Latched per `(pipeline, texture)`: a guest that means this means it every
/// frame, and the population's size belongs to a counter, not to this line.
/// The counter is beside it and is not conditioned on first sight.
pub fn note_attachment_sample_count_override(
    pipeline_ref: u32,
    att: ColorAttachment,
    counts: AttachmentSampleCounts,
    geom: (u32, u32),
    dest: (u32, u64),
) {
    let Some(pipeline) = counts.pipeline else {
        return;
    };
    if pipeline == counts.target {
        return;
    }
    // Split at the emitter, because the two halves have different owners. A
    // promotion with a resolve texture declared is a shape this device can
    // still land; one without has nowhere for the samples to go.
    crate::runtime::drain::note_store_route(if att.resolve_texture_ref != 0 {
        "attach_samples_from_pipeline_with_resolve"
    } else if pipeline > counts.target {
        "attach_samples_multisample_no_resolve"
    } else {
        "attach_samples_below_provisional"
    });
    if crate::observe::first_sight(
        "attachment_sample_count_override",
        (u64::from(pipeline_ref) << 32) | u64::from(att.texture_ref),
    ) {
        crate::observe::off(format!(
            "attachment_sample_count_override pipe={pipeline_ref} tex_ref={} \
             resolve_ref={} pipeline_samples={pipeline} target_samples={} \
             resolved_samples={} load={:#x} store={:#x} {}x{} mid={} gva={:#x} \
             (Metal requires these to agree; a promotion with no resolve \
              texture has nowhere to put the samples)",
            att.texture_ref,
            att.resolve_texture_ref,
            counts.target,
            counts.resolved,
            att.load_action,
            att.store_action,
            geom.0,
            geom.1,
            dest.0,
            dest.1,
        ));
    }
}

/// Resolve color texture ref → mapping geometry for a draw request.
#[allow(
    clippy::too_many_arguments,
    reason = "the request builder mirrors the decoded color attachment state"
)]
pub fn color_target_request<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &M,
    task_id: u32,
    color: crate::runtime::render_pass::ColorAttachment,
    pipeline_ref: u32,
    vertex_count: u32,
    instance_count: u32,
    primitive_type: u32,
    first_vertex: u32,
    base_instance: u32,
) -> Option<DrawEncodeRequest> {
    let color_texture_ref = color.texture_ref;
    let rt = lookup_render_target(state, host, task_id, color)?;
    let attachment_sample_count = crate::backend::selected()
        .pipeline_raster_sample_count(state, host, task_id, pipeline_ref)
        .unwrap_or(rt.sample_count);
    let c0 = ColorRtRequest {
        slot: 0,
        texture_ref: color_texture_ref,
        mapping_id: rt.mapping_id,
        target_gva: rt.target_gva,
        row_stride: rt.row_stride,
        width: rt.width,
        height: rt.height,
        format: rt.format,
        sample_count: attachment_sample_count,
        load_action: 0,
        store_action: MTL_STORE_ACTION_STORE,
        clear_color: [0.0; 4],
        target_seed_rgba: None,
        multisample_source_ref: 0,
    };
    Some(DrawEncodeRequest {
        task_id,
        pipeline_ref,
        vertex_count,
        instance_count,
        primitive_type,
        first_vertex,
        base_instance,
        colors: vec![c0],
        ..Default::default()
    })
}

/// Build an MRT draw request from pass color slots (same dimensions required).
#[allow(
    clippy::too_many_arguments,
    reason = "the MRT builder combines explicit pass, pipeline, and draw state"
)]
pub fn mrt_draw_request<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    pipeline_ref: u32,
    color_slots: &[(u32, crate::runtime::render_pass::ColorAttachment)],
    clears: &[crate::runtime::render_pass::ColorAttachment],
    draw: crate::protocol::draw::DrawArgs,
) -> Option<DrawEncodeRequest> {
    if color_slots.is_empty() {
        return None;
    }
    // Linear allocation dimensions expose mip and array geometry, but do not
    // repeat a texture's immutable creation sample count. At render time the
    // bound pipeline supplies the missing contract: every color attachment
    // must match its raster sample count. Resolve that before LOAD/CLEAR seed
    // policy and before this request is cloned by either encoder.
    let pipeline_sample_count =
        crate::backend::selected().pipeline_raster_sample_count(state, host, task_id, pipeline_ref);
    let mut colors = Vec::new();
    let mut base_w = 0u32;
    let mut base_h = 0u32;
    // Colour0's LOAD seed was skipped in favour of the engine resident. Declared
    // out here because it belongs to the request, not to the slot that set it.
    let mut gva_load_from_resident = false;
    for &(slot, att) in color_slots {
        if att.texture_ref == 0 {
            // An empty colour slot is the guest declining to attach one, not a
            // loss. Counted anyway, because it is the difference between the
            // slots the pass *has* and the slots it *uses*, and the census
            // below is unreadable without it.
            crate::runtime::drain::note_store_route("mrt_slot_empty");
            continue;
        }
        crate::runtime::drain::note_store_route("mrt_slot_attached");
        // Resolve both sides independently. The source proves the multisample
        // attachment's shape; the destination becomes the guest-visible target
        // that the backend stores and reads back.
        let Some(source_target) = lookup_render_target(state, host, task_id, att) else {
            crate::runtime::drain::note_store_route("mrt_slot_unresolved");
            return None;
        };
        let (target_ref, multisample_source_ref, target) = if att.resolve_texture_ref != 0 {
            let resolve_attachment = ColorAttachment {
                texture_ref: att.resolve_texture_ref,
                resolve_texture_ref: 0,
                level: 0,
                ..att
            };
            let Some(resolve_target) =
                lookup_render_target(state, host, task_id, resolve_attachment)
            else {
                crate::runtime::drain::note_store_route("mrt_resolve_target_unresolved");
                return None;
            };
            if crate::observe::first_sight(
                "render_resolve_contract",
                (u64::from(att.texture_ref) << 32) | u64::from(att.resolve_texture_ref),
            ) {
                crate::observe::off(format!(
                    "render_resolve_contract task={task_id} pipe={pipeline_ref} \
                     source_ref={} source_mid={} source_gva={:#x} source={}x{} \
                     source_fmt={:#x} resolve_ref={} resolve_mid={} resolve_gva={:#x} \
                     resolve={}x{} resolve_fmt={:#x} load={} store={} raster_samples={}",
                    att.texture_ref,
                    source_target.mapping_id,
                    source_target.target_gva,
                    source_target.width,
                    source_target.height,
                    source_target.format,
                    att.resolve_texture_ref,
                    resolve_target.mapping_id,
                    resolve_target.target_gva,
                    resolve_target.width,
                    resolve_target.height,
                    resolve_target.format,
                    att.load_action,
                    att.store_action,
                    // What the attachment will be encoded at, which is the
                    // pipeline's count where a rail consults it and the target's
                    // otherwise. Every construction site of the resolved target
                    // hardcodes `1` (see `AttachmentSampleCounts::target`), so
                    // this is the same number the `unwrap_or(1)` here used to
                    // spell — said in terms of where it comes from rather than
                    // as a literal that happens to match.
                    pipeline_sample_count.unwrap_or(source_target.sample_count),
                ));
            }
            if source_target.width != resolve_target.width
                || source_target.height != resolve_target.height
                || source_target.format != resolve_target.format
            {
                crate::observe::fail(format!(
                    "render_resolve_target_mismatch source={} resolve={} source_geom={}x{} \
                     resolve_geom={}x{} source_fmt={:#x} resolve_fmt={:#x}",
                    att.texture_ref,
                    att.resolve_texture_ref,
                    source_target.width,
                    source_target.height,
                    resolve_target.width,
                    resolve_target.height,
                    source_target.format,
                    resolve_target.format
                ));
                return None;
            }
            (att.resolve_texture_ref, att.texture_ref, resolve_target)
        } else {
            (att.texture_ref, 0, source_target)
        };
        let ResolvedRenderTarget {
            mapping_id,
            target_gva: gva,
            width: mw,
            height: mh,
            row_stride: bpr,
            format: mfmt,
            sample_count: target_sample_count,
        } = target;
        let attachment_sample_count = pipeline_sample_count.unwrap_or(target_sample_count);
        note_attachment_sample_count_override(
            pipeline_ref,
            att,
            AttachmentSampleCounts {
                pipeline: pipeline_sample_count,
                target: target_sample_count,
                resolved: attachment_sample_count,
            },
            (mw, mh),
            (mapping_id, gva),
        );
        if base_w == 0 {
            base_w = mw;
            base_h = mh;
        } else if mw != base_w || mh != base_h {
            // An attachment whose geometry differs from the first one is
            // dropped, and the draw goes on with the rest. **This is a loss the
            // guest is not told about**: the shader still writes that
            // `[[color(n)]]` output, the attachment it was aimed at never
            // receives it, and a later sample of that texture reads whatever was
            // there before. It is the same class `secondary_mrt_drop` reports
            // one stage further on, and it used to be a bare `continue` with a
            // comment — so a pass whose second attachment was skipped here
            // arrived at that census as a single-attachment draw and was
            // counted as `mrt_draw_single`, indistinguishable from a guest that
            // never asked for MRT at all.
            //
            // Reported rather than refused, and reported before it is fixed,
            // because the fix depends on which way the geometry differs and no
            // boot has yet produced one: a Metal attachment larger than the
            // render area is legal and should be rendered into at the pass's
            // size, while a smaller one is a guest error Metal itself would
            // reject.
            crate::runtime::drain::note_store_route("mrt_slot_geometry_dropped");
            if crate::observe::first_sight("mrt_slot_geometry_dropped", u64::from(slot)) {
                crate::observe::fail(format!(
                    "mrt_slot_geometry_dropped slot={slot} ref={} got={mw}x{mh} \
                     want={base_w}x{base_h} (the attachment is dropped and the \
                     draw runs without it, so the shader's output for this slot \
                     goes nowhere and a later sample reads stale content)",
                    att.texture_ref
                ));
            }
            continue;
        }
        let mut load_action = att.load_action;
        let mut clear_color = att.clear_color;
        let mut seed = None;
        if let Some(cl) = clears.iter().find(|a| a.texture_ref == att.texture_ref) {
            // Clear-only stream record for this attachment: real Metal Clear.
            load_action = MTL_LOAD_ACTION_CLEAR;
            clear_color = cl.clear_color;
            if mapping_id == 0 {
                seed = Some(solid_rgba8(mw, mh, &cl.clear_color));
            }
        } else if att.load_action == MTL_LOAD_ACTION_CLEAR {
            if mapping_id == 0 {
                seed = Some(solid_rgba8(mw, mh, &att.clear_color));
            }
        } else if mapping_id == 0
            && (att.load_action == MTL_LOAD_ACTION_LOAD
                || (dontcare_seed_probe_enabled()
                    && att.load_action == MTL_LOAD_ACTION_DONT_CARE))
        {
            if att.load_action == MTL_LOAD_ACTION_DONT_CARE
                && crate::observe::first_sight("dontcare_seed_probe", u64::from(att.texture_ref))
            {
                crate::observe::off(format!(
                    "dontcare_seed_probe texture_ref={} target={}x{}",
                    att.texture_ref,
                    mw,
                    mh
                ));
            }
            // # This arm compares an ordinal, and the contract term is wider
            //
            // `MTLLoadActionDontCare` also promises the prior contents --
            // undefined permits them, `backend::metal::render` hands the same
            // wire word to Metal which preserves them, and the guest declares
            // DontCare and then redraws only its damage rectangle.
            // `protocol::pass_action::LoadAction::preserves_prior_contents`
            // states that term and answers true for both ordinals. The seed
            // block in `draw::vulkan` was widened to it, and the
            // secondary-attachment path in the same file spells it directly as
            // `LoadAction::DontCare => resident_content_ready(&identity)`.
            // This arm was not.
            //
            // What that costs, measured on rail macos-15: a GVA attachment
            // declaring DontCare falls past here with no seed and with
            // `gva_load_from_resident` false, and every downstream door is then
            // shut to it -- `honour_gva_load_elision` returns on the flag, the
            // seed block's mapping door is guarded by `mapping_id != 0`, and
            // `target_seed_rgba` is `None`. `PassKey::single` reads "no seed",
            // `caches.rs` resolves that to `vk::AttachmentLoadOp::CLEAR`, and
            // every texel outside the draw's scissor becomes `target_clear` --
            // untouched at `[0.0; 4]`, transparent black, because that variable
            // is assigned only in the `Clear` arm. Boot s5: 461 partial draws
            // and 2 107 399 texels, over live guest content.
            //
            // # Why the one-line widening is not the repair
            //
            // Replacing this ordinal test with `preserves_prior_contents()` was
            // built and measured, and it does remove the whole defect: across
            // three candidate conformance batteries `dontcare_seed_empty`,
            // `draw_partial_preserving_unseeded` and its lost-texel total were
            // all **zero**, against 211 012 and 298 602 lost texels on two
            // control batteries, with `dontcare_seed_served` rising 16 -> 31.
            //
            // It also regressed the compatibility ratchet. Over five candidate
            // batteries against four control batteries on rail macos-15:
            //
            //   candidate  1 run hung 600 s at `srt_blit_after_render_1920x1080`
            //              (6 cases NOT-RUN, one of them previously classified)
            //   candidate  1 run `srt_blit_iosurface_source_1920x1080_x4`
            //              REGRESSION, stale_frames=2/4, 576 wrong texels
            //   control    4 runs clean, 293/293, 19/19 driver, 0 unexplained
            //
            // Both failures are in the deliberately racy heavy 1920x1080 blit
            // family, whose own source says its repeated whole-target draws
            // exist "so the GPU is still working when the copy behind them is
            // decoded". The mechanism is cost, not staleness: `gvaseed_chained`
            // is unchanged between the arms (195-347 control against 176-308
            // candidate), so the widening is not electing more resident
            // elisions -- it is paying ~15 extra full-frame `seed_color_load`
            // CPU reads per battery, and that latency is enough to flip cases
            // built to race.
            //
            // # A cost-negative variant was also built, and it hung too
            //
            // The obvious answer to "the CPU seed is what costs" is to preserve
            // from the resident instead: the price of preserving is the seed
            // *upload*, and when the pixels are already in the engine resident
            // there is nothing to upload -- `PassKey`'s seeded arm spells
            // `vk::AttachmentLoadOp::LOAD` against the attachment's existing
            // layout, which is what `chain_load_from_target` already does for a
            // render chain. That arm costs strictly *less* than the branch it
            // replaces: it removes a full-surface clear write and adds no read.
            // It is lawful for DontCare specifically, because undefined permits
            // any contents, so a resident that is stale against a guest CPU
            // write needs none of the currency reconciliation a LOAD would.
            //
            // It was implemented in `draw::vulkan`'s seedless-DontCare arm,
            // scoped to GVA targets, and it worked: one battery measured
            // `dontcare_resident_preserved` 47, `dontcare_resident_absent` 12,
            // and `draw_partial_preserving_unseeded` down from ~78 to 5.
            //
            // **And it hung in exactly the same place** --
            // `srt_blit_after_render_1920x1080`, rc=124 after the 600 s probe
            // timeout, immediately after `srt_blit_after_render_1024x768`
            // passed, the identical signature the seed variant produced.
            //
            // That is the reading to carry forward, because it retires the
            // latency explanation the seed variant suggested. Two variants with
            // opposite cost profiles -- one adding full-frame CPU reads, one
            // removing full-surface clear writes -- hang at the same case. So
            // either routing a DontCare GVA pass to preserve *by any door*
            // disturbs that case, or the case is flaky and both candidates were
            // unlucky. The counts do not separate those: 3 anomalies across 8
            // candidate batteries against 0 across 4 control batteries, which
            // is suggestive and not significant.
            //
            // Whoever takes this next should establish which, and the cheapest
            // way is to bound the control: run the control battery enough times
            // to give `srt_blit_after_render_1920x1080` a fair chance to hang on
            // its own. A control hang settles it as inherited raciness in a case
            // whose own source says its repeated whole-target draws exist "so
            // the GPU is still working when the copy behind them is decoded".
            // Absent that, the mechanism has to be understood before either
            // variant can land -- start from the hung run's device log, where
            // the device is idle (`drain_duty duty=0.002`, no submissions, no
            // typed failure) behind a `stamp_wait_timeout` on a 2.6 s
            // `gpu_span busy_max_us`, and a control run reached the same
            // escalated stamp pattern without hanging.
            //
            // # The shape that was taken
            //
            // `PassKey.load_seed` was a `bool` and the contract it represents
            // has three values: preserve, clear to the guest's colour, and
            // undefined. Two collapsed onto `false` and `caches.rs` resolved
            // `false` to `CLEAR`. It is now
            // `backend::vulkan::engine::caches::Color0Load`, and a seedless
            // preserving pass keys to `Undefined` and resolves to
            // `vk::AttachmentLoadOp::DONT_CARE` against the attachment's
            // resting layout -- lawful, writing none of the attachment, and
            // *cheaper* than the full-surface clear it replaces, so it removes
            // the invented colour without adding the latency that flipped those
            // cases.
            //
            // This arm is therefore left comparing an ordinal on purpose. It
            // decides whether to spend a CPU seed read, and that is the cost
            // the two withdrawn variants were withdrawn for; the colour the
            // guest never supplied is no longer downstream of it. What remains
            // open is the cost-negative preserving arm: when the engine
            // resident already holds this target's contents, the request could
            // elect `Color0Load::Preserve` and *guarantee* what `DONT_CARE`
            // only makes likely. Its witness is
            // `a_preserving_gva_attachment_reaches_the_encoder_able_to_preserve`,
            // still `#[ignore]`.
            //
            // GVA linear target: ephemeral host RT needs a CPU seed (archive
            // reims_vgpu_backend_metal; NULL seed → Metal Clear invent, still encode).
            // Mapper-ref-texture is seeded later instead, at the attachment site in
            // `encode_draw` — the same place the guest-backed alias used to be
            // built, and the same seed it already took whenever the alias was
            // refused. Seeding here would need the mapping read twice.
            //
            {
                // Before the read, not after it: the seed this is about to build
                // is the one a resident rung would replace, and a probe placed
                // downstream of here measures an empty population by
                // construction — see `note_gva_load_seed_probe`.
                // Before the read, not after it. The engine may still hold
                // exactly what the render Store published into these pages, in
                // which case reading them back costs a full-frame CPU walk and a
                // block on that same Store's writeback — the device's largest
                // remaining wait. See `draw::vulkan::gva_resident_if_current`;
                // the encode side honours the flag or re-seeds.
                let elided = crate::backend::selected().gva_load_seed_elidable(
                    state,
                    host,
                    task_id,
                    GvaSpan {
                        texture_ref: att.texture_ref,
                        gva,
                        row_stride: bpr,
                        width: mw,
                        height: mh,
                        format: mfmt,
                    },
                );
                // Only colour0. `gva_chain_identity` names the first attachment
                // and the chain rail carries that one, so a second slot whose
                // seed was skipped would reach the pass with nothing to load.
                // `colors.is_empty()` is "this push becomes `colors[0]`", taken
                // from the vector the identity will read rather than from the
                // slot number, which is the guest's and need not start at zero.
                let elided = elided && colors.is_empty();
                gva_load_from_resident = elided;
                if !elided {
                    seed = seed_color_load(state, host, task_id, att.texture_ref, gva, mw, mh);
                    if seed.is_none() {
                        crate::observe::fail(format!(
                            "color LOAD seed miss ref={} {}x{} fmt={:#x} gva={:#x} (archive: still encode)",
                            att.texture_ref, mw, mh, mfmt, gva
                        ));
                    }
                }
            }
        }
        colors.push(ColorRtRequest {
            slot,
            texture_ref: target_ref,
            mapping_id,
            target_gva: gva,
            row_stride: bpr,
            width: mw,
            height: mh,
            format: mfmt,
            sample_count: attachment_sample_count,
            load_action,
            store_action: att.store_action,
            clear_color,
            target_seed_rgba: seed,
            multisample_source_ref,
        });
    }
    if colors.is_empty() {
        return None;
    }
    Some(DrawEncodeRequest {
        task_id,
        pipeline_ref,
        vertex_count: draw.vertex_count,
        instance_count: draw.instance_count,
        primitive_type: draw.primitive_type,
        first_vertex: draw.first_vertex,
        base_instance: draw.base_instance,
        colors,
        gva_load_from_resident,
        ..Default::default()
    })
}

/// Archive `apple_pv_gpu_write_gva_rgba`: tight RGBA8 → native rows at GVA.
/// Packed contig HostOps view when possible; else multi-import per row
/// ([`crate::runtime::gva_view::write_span_within`]) — no `write_gpa` walk.
///
/// Carries the refusal out rather than collapsing to `false`: a caller has to be
/// able to tell "the guest tore this target down" (`MemError::is_guest_teardown`)
/// from a write that genuinely lost content.
///
/// # MapMemory2 does not bound this writer, and nothing else may pretend to
///
/// `MapMemory2` is a notification the guest sends *after* installing its own
/// PTEs and using the memory, so it cannot authorise anything — measured on the
/// x86/Vulkan rail at 0-29 ms after the write it would have had to precede, and
/// on one driven boot **44% of render-target Stores** (893 of 2048) sat outside
/// every span the writing task had filed. It does not describe render targets at
/// all: task 0 files a single 64 MiB span (`0x101000..0x4101000`) while the
/// Stores sit at GVAs like `0x4692000`, past all of it.
///
/// The tempting weaker rule — "allow when *some* task's span covers it" — is
/// worse, not better. A span filed by another task numerically containing this
/// range says only that two address spaces both have something there; across
/// 7 445 measured cases the two never once resolved to the same guest physical
/// page. A virtual-address coincidence is not evidence that a range is
/// legitimate.
///
/// What does bound these writes: every Store carries the page set its target
/// GVA resolved to *before* the GPU round trip and goes through
/// [`write_gva_rgba8_within`], so the walk that resolves its destination is also
/// the walk that authorises it. That includes the synchronous Store — see
/// [`sync_store_target_pages`] for why "synchronous" does not mean the guest
/// stood still. This unbounded form survives only for callers replaying a write
/// whose authorisation is the command being executed on this thread with no GPU
/// wait in between.
#[allow(
    clippy::too_many_arguments,
    reason = "the archive writer mirrors the target GVA and native row geometry"
)]
pub(crate) fn write_gva_rgba8<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    gva: u64,
    width: u32,
    height: u32,
    bpr: u32,
    format: u16,
    rgba: &[u8],
) -> Result<(), crate::runtime::host::MemError> {
    write_gva_rgba8_within(
        state, host, task_id, gva, width, height, bpr, format, rgba, None,
    )
}

/// Guest pages one color attachment's synchronous GVA Store may write, resolved
/// before the draw is submitted to the GPU.
///
/// The Store's write used to be unbounded on the argument that a synchronous
/// command's authorisation is the page table at the moment it runs. That holds
/// for the CLEAR store, which is a solid colour written on this thread with
/// nothing in between. It does not hold for the draw Store: both backends'
/// encode paths encode, submit, wait for the GPU and read the result back
/// before the Store resolves `target_gva`, and the guest runs on its own vCPUs
/// across that round trip. Resolving here makes the pages the command named and
/// the pages the bytes reach the same set.
///
/// The span is the attachment's whole image (`row_stride * height`) rather than
/// the scissor rect a partial store touches, because the packed rail maps the
/// whole image in one view and authorises every page it aliases.
///
/// The capture walk drops pages that do not resolve, while the writer's walk
/// fails the whole span on one. The set is therefore a subset of what the writer
/// will ask to write, never a superset, so the disagreement can only refuse and
/// never wrongly permit. The one case it refuses is a page that was unresolved
/// at capture and resolvable at write time, which is a re-point — the event this
/// bound exists to catch.
///
/// `None` — unbounded, the pre-existing behaviour — when there is no GVA target,
/// when the record does not store, or when the walk resolves no page at all.
/// The last arm is counted (`sync_store_unbounded`) rather than tightened on
/// suspicion: a span that resolves nothing here makes the writer's own walk fail
/// closed on its own terms, and refusing on an empty capture would drop live
/// Stores whenever the capture failed for an unrelated reason. If that counter
/// stays at zero it can be tightened with evidence.
pub(crate) fn sync_store_target_pages<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    c: &ColorRtRequest,
) -> Option<StoreTargetPages> {
    if c.target_gva == 0
        || !reims_vgpu_protocol::pass_action::store_action_publishes_single_sample(c.store_action)
        || c.width == 0
        || c.height == 0
    {
        return None;
    }
    let span = (c.row_stride as u64).checked_mul(c.height as u64)?;
    let ordered = crate::runtime::gva_mem::task_gva_page_gpas(
        host,
        &state.tasks,
        task_id,
        c.target_gva,
        span,
        state.page_shift,
    );
    if ordered.is_empty() {
        crate::runtime::drain::note_store_route("sync_store_unbounded");
        return None;
    }
    crate::runtime::drain::note_store_route("sync_store_bound");
    Some(StoreTargetPages {
        set: ordered.iter().copied().collect(),
        ordered,
        span,
    })
}

/// The guest pages a synchronous GVA render Store may write, from one walk
/// taken before the draw was submitted.
///
/// Two shapes of one answer, because the two writers ask it differently. The
/// row-by-row writer asks "is this page one of mine?" once per row, and the
/// GPU-direct writer needs the pages in GVA order so neighbours coalesce into
/// the contiguous runs a copy binds. Derived from a single walk rather than
/// taken twice, so the two rails cannot end up authorised differently — which
/// is the whole point of resolving before the submit.
/// Only the Vulkan backend has a GPU-direct GVA writeback, so on the Metal arm
/// the ordered form of the walk has no reader. Held rather than `cfg`-ed out of
/// the struct: both fields are produced by the one walk either way, and a
/// conditional shape would make the two arms disagree about what a Store's
/// authorisation *is*.
#[cfg_attr(not(feature = "backend-vulkan"), allow(dead_code))]
pub(crate) struct StoreTargetPages {
    ordered: Vec<u64>,
    set: std::collections::HashSet<u64>,
    span: u64,
}

impl StoreTargetPages {
    /// Reconstitute a transfer destination from a live resource's retained
    /// backing. The entries are physical page identities; bounded guest slices
    /// are created only when the backend submits the transfer.
    ///
    /// Not gated on the Vulkan backend: the compute rail builds one on every
    /// arm, because a page record present on only one of them would make the two
    /// arms disagree about what a staged window's authorisation is — the same
    /// reason the struct itself holds both fields unconditionally.
    pub(crate) fn from_ordered(ordered: &[u64], span: u64) -> Self {
        Self {
            ordered: ordered.to_vec(),
            set: ordered.iter().copied().collect(),
            span,
        }
    }

    /// The record a walk that resolved nothing leaves behind.
    ///
    /// Not the same as a complete record of zero pages, and no span can produce
    /// one: [`Self::ordered_complete`] asks for `pages_spanned(gva, span)`
    /// entries, which is at least one for every non-empty span, so a consumer
    /// meets a refusal here rather than a window that reads as having nothing
    /// in it.
    pub(crate) fn empty() -> Self {
        Self {
            ordered: Vec::new(),
            set: std::collections::HashSet::new(),
            span: 0,
        }
    }

    /// The same pages as a membership test, which is the bound
    /// [`write_gva_rgba8_within`] takes.
    pub(crate) fn membership(&self) -> &std::collections::HashSet<u64> {
        &self.set
    }

    /// Page GPAs in GVA order, **only** when the walk resolved every page of
    /// the destination span.
    ///
    /// `None` on a short walk, and that is not the same fail-closed the
    /// membership form has. A dropped page leaves the set a subset, which can
    /// only refuse a row; it leaves this vector *shifted*, because a consumer
    /// reads index `i` as page `i` of the window. A copy built from a shifted
    /// list would land the frame's bytes at the wrong guest addresses without
    /// anything noticing — the copy converts nothing and checks nothing.
    #[cfg_attr(not(feature = "backend-vulkan"), allow(dead_code))]
    pub(crate) fn ordered_complete(&self, gva: u64, page_size: u64) -> Option<&[u64]> {
        let want = reims_vgpu_paging::span::pages_spanned(gva, self.span, page_size);
        (self.ordered.len() as u64 == want).then_some(&self.ordered[..])
    }
}

/// [`write_gva_rgba8`] bounded to the guest pages a deferred window was armed
/// on.
///
/// A deferred window IS those pages: it was armed when they were the window's,
/// and it lands an unbounded time later. Re-walking is necessary — a cached view
/// goes stale silently — but a fresh walk answers *where this address points
/// now*, which is a different question from *is this still our memory*. Handing
/// the armed set into the walk makes them one question, so the bytes cannot
/// reach a page the window was not given, whatever the guest did in between.
///
/// This is what closes the gap a separate page-drift check leaves open. Such a
/// guard walks, decides, and returns; the writer then walks
/// again, and the guest runs on its own vCPUs between the two. The guard stays —
/// it names the event in the always-on log with the counts a reader needs — but
/// it is the report, and this is the bound.
#[allow(
    clippy::too_many_arguments,
    reason = "the archive writer mirrors the target GVA and native row geometry"
)]
pub(crate) fn write_gva_rgba8_within<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    gva: u64,
    width: u32,
    height: u32,
    bpr: u32,
    format: u16,
    rgba: &[u8],
    allowed: crate::runtime::gva_view::WindowPages<'_>,
) -> Result<(), crate::runtime::host::MemError> {
    write_gva_frame_within(
        state,
        host,
        task_id,
        gva,
        width,
        height,
        bpr,
        format,
        FrameRows::Rgba8(rgba),
        allowed,
    )
}

/// What a frame's source rows are, on their way into the guest's own pages.
///
/// A Store lands one of two things, and which one is not a property of the
/// guest's destination — it is a property of what the resident held and whether
/// the readback could narrow it. Naming both here is what lets the copying rail
/// serve a destination whose texel has no eight-bit form at all: the RGBA8 arm
/// converts per row, and the native arm is a memcpy because the bytes are
/// already the destination's.
///
/// The native arm is only ever reached when the frame's layout and the
/// destination's are the same layout — `store_texel_order`'s question, which the
/// GPU-direct rail has always asked and the copying rail could not.
pub(crate) enum FrameRows<'a> {
    /// Semantic RGBA8, converted into the destination's texel one row at a time.
    Rgba8(&'a [u8]),
    /// Already the destination's texel, copied verbatim.
    ///
    /// Produced only by the Vulkan Store's readback, the one rail that can hand
    /// back a resident's own texel. The Metal arm has no producer for it and the
    /// writer below still has to name it.
    #[cfg_attr(not(feature = "backend-vulkan"), allow(dead_code))]
    Native(&'a [u8]),
}

impl<'a> FrameRows<'a> {
    fn bytes(&self) -> &'a [u8] {
        match *self {
            Self::Rgba8(b) | Self::Native(b) => b,
        }
    }

    /// Bytes one source row occupies, which is the destination's tight row for
    /// the native arm and always four bytes a texel for the RGBA8 one.
    fn source_row_bytes(&self, width: u32, tight: u32) -> usize {
        match self {
            Self::Rgba8(_) => (width as usize).saturating_mul(RGBA8_BPP as usize),
            Self::Native(_) => tight as usize,
        }
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the target GVA and native row geometry"
)]
pub(crate) fn write_gva_frame_within<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    gva: u64,
    width: u32,
    height: u32,
    bpr: u32,
    format: u16,
    frame: FrameRows<'_>,
    allowed: crate::runtime::gva_view::WindowPages<'_>,
) -> Result<(), crate::runtime::host::MemError> {
    write_gva_frame_within_skipping(
        state,
        host,
        task_id,
        gva,
        width,
        height,
        bpr,
        format,
        frame,
        allowed,
        &[],
    )
}

/// [`write_gva_frame_within`], leaving `skip` untouched.
///
/// `skip` is in bytes from `gva`, ascending and disjoint — the same coordinate
/// system `bpr` and the row offsets below are in, and the GVA spelling of
/// [`crate::runtime::mapping_write::SkipRanges`]. It exists for the one caller
/// that has a third answer to give: a deferred writeback landing a frame the
/// device rendered into pages the guest CPU wrote part of in between. Writing
/// the whole frame loses the guest's stores and dropping it loses the device's;
/// `skip` names the bytes the guest's own memory keeps.
///
/// Every other caller passes `&[]` and lands the frame whole, which is what a
/// Store with no intervening guest write means.
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the target GVA and native row geometry, plus the bytes its owner may not overwrite"
)]
pub(crate) fn write_gva_frame_within_skipping<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    gva: u64,
    width: u32,
    height: u32,
    bpr: u32,
    format: u16,
    frame: FrameRows<'_>,
    allowed: crate::runtime::gva_view::WindowPages<'_>,
    skip: crate::runtime::mapping_write::SkipRanges<'_>,
) -> Result<(), crate::runtime::host::MemError> {
    use crate::runtime::host::MemError;
    if gva == 0 || width == 0 || height == 0 || bpr == 0 {
        return Err(MemError::BadArgs);
    }
    let Some(tight) = pixel_format::tight_row_bytes(width, format) else {
        return Err(MemError::BadArgs);
    };
    if bpr < tight {
        return Err(MemError::BadArgs);
    }
    let src = frame.bytes();
    let src_stride = frame.source_row_bytes(width, tight);
    let need = src_stride.saturating_mul(height as usize);
    if src.len() < need {
        return Err(MemError::BadArgs);
    }
    let span = (height as u64).saturating_mul(bpr as u64);
    // Only the RGBA8 arm converts into a scratch row; the native arm's bytes
    // are already the destination's texel and are copied straight out of the
    // frame, so it must not pay an allocation per Store for a buffer it never
    // reads.
    let mut row = match frame {
        FrameRows::Rgba8(_) => vec![0u8; tight as usize],
        FrameRows::Native(_) => Vec::new(),
    };
    // One parse for both arms below — the mapped-span walk and the fragmented
    // per-row one convert the same frame at the same format. A `Native` frame
    // converts nothing, so an unsupported format is not its problem and the
    // parse stays lazy. See `pixel_format::Rgba8ToRow`.
    let row_rail = pixel_format::Rgba8ToRow::for_format(format);
    // Guest writes resolve through a fresh PT walk at write time — never a
    // cached view (stale-view heap-corruption class; see
    // `gva_view::write_span_within`) —
    // and that walk carries `allowed`, so a deferred window cannot alias a page
    // outside itself even if the guest re-points the range mid-flush.
    if let Some(span_map) =
        crate::runtime::gva_view::map_fresh_span_within(state, host, task_id, gva, span, allowed)
    {
        let (base, avail) = (span_map.ptr, span_map.avail);
        let mut res = Ok(());
        for y in 0..height as usize {
            let at = y * src_stride;
            let out_row: &[u8] = match frame {
                FrameRows::Rgba8(rgba) => {
                    if !row_rail
                        .is_some_and(|r| r.convert(&rgba[at..at + src_stride], width, &mut row))
                    {
                        res = Err(MemError::BadArgs);
                        break;
                    }
                    &row
                }
                FrameRows::Native(native) => &native[at..at + src_stride],
            };
            let off = y.saturating_mul(bpr as usize);
            if off + out_row.len() > avail {
                res = Err(MemError::RunOutOfRange);
                break;
            }
            // One forward walk per row through the shared range subtraction, so
            // this rail and the mapping writer cannot come to disagree about
            // which bytes are excluded.
            for (from, to) in crate::runtime::mapping_write::unskipped(
                off as u64,
                (off + out_row.len()) as u64,
                skip,
            ) {
                let (at, len) = ((from as usize) - off, (to - from) as usize);
                // SAFETY: map_fresh_span covers `span`, and `from`/`to` are a
                // sub-range of this row, which the bound above put inside it.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        out_row[at..].as_ptr(),
                        base.add(from as usize),
                        len,
                    );
                }
            }
        }
        crate::runtime::gva_view::unmap_fresh_span(host, span_map);
        return res;
    }
    // Fragmented GVA: multi-import each row via `write_span_within`.
    for y in 0..height as usize {
        let at = y * src_stride;
        let out_row: &[u8] = match frame {
            FrameRows::Rgba8(rgba) => {
                if !row_rail.is_some_and(|r| r.convert(&rgba[at..at + src_stride], width, &mut row))
                {
                    return Err(MemError::BadArgs);
                }
                &row
            }
            FrameRows::Native(native) => &native[at..at + src_stride],
        };
        let row_off = (y as u64).saturating_mul(bpr as u64);
        for (from, to) in
            crate::runtime::mapping_write::unskipped(row_off, row_off + out_row.len() as u64, skip)
        {
            let run_gva = gva.saturating_add(from);
            let run = &out_row[(from - row_off) as usize..(to - row_off) as usize];
            if let Err(err) = crate::runtime::gva_view::write_span_within(
                state, host, task_id, run_gva, run, allowed,
            ) {
                let reason = crate::observe::Decline::slug(&err);
                crate::observe::fail(format!(
                    "gva_write fail reason={reason} task={task_id} gva={run_gva:#x} span={span:#x} \
                     row={y} rowlen={:#x} (multi)",
                    run.len()
                ));
                return Err(err);
            }
        }
    }
    Ok(())
}

/// Store only the Metal scissor rect of a full-size tight RGBA8 buffer to GVA,
/// bounded to the pages the Store's target resolved to before the GPU ran.
/// Packed contig view when possible; else multi-import each rect row.
///
/// Only the Metal encode path issues a scissored guest store today, but nothing
/// here is Metal-specific: it is plain page-table walking over
/// [`crate::runtime::host::HostMemory`],
/// so it stays compiled and tested on every arm. Gating it behind the backend
/// that happens to call it would put the guest-memory bound on the one matrix
/// arm that cannot be built or run from a Linux host.
#[cfg_attr(
    not(all(feature = "backend-metal", target_os = "macos")),
    allow(
        dead_code,
        reason = "only the Metal encode path scissors a guest store; the bound is tested everywhere"
    )
)]
#[allow(
    clippy::too_many_arguments,
    reason = "the archive writer mirrors the target GVA and its native row geometry"
)]
pub(crate) fn write_gva_rgba8_rect<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    gva: u64,
    full_w: u32,
    full_h: u32,
    bpr: u32,
    format: u16,
    rgba: &[u8],
    rect: mapping_write::Rect,
    allowed: crate::runtime::gva_view::WindowPages<'_>,
) -> bool {
    let mapping_write::Rect {
        origin_x,
        origin_y,
        width: rect_w,
        height: rect_h,
    } = rect;
    if gva == 0
        || full_w == 0
        || full_h == 0
        || rect_w == 0
        || rect_h == 0
        || bpr == 0
        || origin_x.saturating_add(rect_w) > full_w
        || origin_y.saturating_add(rect_h) > full_h
    {
        return false;
    }
    let Some(tight_full) = pixel_format::tight_row_bytes(full_w, format) else {
        return false;
    };
    let Some(tight_rect) = pixel_format::tight_row_bytes(rect_w, format) else {
        return false;
    };
    if bpr < tight_full {
        return false;
    }
    let Some(bpp) = pixel_format::bytes_per_pixel(format) else {
        return false;
    };
    let rgba_row = (full_w as usize).saturating_mul(RGBA8_BPP as usize);
    let need = rgba_row.saturating_mul(full_h as usize);
    if rgba.len() < need {
        return false;
    }
    // One parse for both arms below, as in the whole-frame writer above: this
    // rect lands in the guest's pages at one format however the span maps.
    let Some(row_rail) = pixel_format::Rgba8ToRow::for_format(format) else {
        return false;
    };
    let x_bytes = (origin_x as u64).saturating_mul(bpp as u64);
    let mut row = vec![0u8; tight_rect as usize];
    let mut src_rgba = vec![0u8; (rect_w as usize) * (RGBA8_BPP as usize)];
    let span = (full_h as u64).saturating_mul(bpr as u64);
    // Fresh PT walk at write time — never a cached view (stale-view class) —
    // and that walk carries `allowed`, so the rect cannot land on a page the
    // Store's own target did not resolve to before the GPU round trip.
    if let Some(span_map) =
        crate::runtime::gva_view::map_fresh_span_within(state, host, task_id, gva, span, allowed)
    {
        let (base, avail) = (span_map.ptr, span_map.avail);
        let mut ok = true;
        for dy in 0..rect_h as usize {
            let y = origin_y as usize + dy;
            let src_full = &rgba[y * rgba_row + (origin_x as usize) * 4
                ..y * rgba_row + (origin_x as usize) * 4 + (rect_w as usize) * 4];
            src_rgba.copy_from_slice(src_full);
            if !row_rail.convert(&src_rgba, rect_w, &mut row) {
                ok = false;
                break;
            }
            let off = (y as u64)
                .saturating_mul(bpr as u64)
                .saturating_add(x_bytes) as usize;
            if off + row.len() > avail {
                ok = false;
                break;
            }
            // SAFETY: map_fresh_span covers full image span.
            unsafe {
                std::ptr::copy_nonoverlapping(row.as_ptr(), base.add(off), row.len());
            }
        }
        crate::runtime::gva_view::unmap_fresh_span(host, span_map);
        return ok;
    }
    for dy in 0..rect_h as usize {
        let y = origin_y as usize + dy;
        let src_full = &rgba[y * rgba_row + (origin_x as usize) * 4
            ..y * rgba_row + (origin_x as usize) * 4 + (rect_w as usize) * 4];
        src_rgba.copy_from_slice(src_full);
        if !row_rail.convert(&src_rgba, rect_w, &mut row) {
            return false;
        }
        let row_gva = gva
            .saturating_add((y as u64).saturating_mul(bpr as u64))
            .saturating_add(x_bytes);
        if let Err(err) = crate::runtime::gva_view::write_span_within(
            state, host, task_id, row_gva, &row, allowed,
        ) {
            let reason = crate::observe::Decline::slug(&err);
            crate::observe::fail(format!(
                "gva_write fail reason={reason} task={task_id} gva={row_gva:#x} span={span:#x} \
                 row={y} rowlen={:#x} (rgba8 rect multi)",
                row.len()
            ));
            return false;
        }
    }
    true
}

/// Seed color RT LOAD from guest mapper-ref-texture (BGRA→RGBA) or normal-texture/view linear RGBA.
///
/// Every color RT is an ephemeral host RT now, so every `Load` needs this: the
/// mapper-ref-texture guest-memory alias that let Metal Load read the surface bytes in
/// place is deleted. This used to run only on the alias-reject fallback
/// (unaligned offset or row stride, span out of range, no device), which is why
/// it is already a complete path and not a new one.
fn seed_color_load<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    target_gva: u64,
    width: u32,
    height: u32,
) -> Option<Vec<u8>> {
    // Discrete GPU: exact target GVA is the strongest identity across object-ref
    // recycling. Fall back to the normal texture namespace, never the
    // unrelated backing record_id namespace. Guest memory is last.
    if width > 0 && height > 0 {
        if target_gva != 0 {
            // Recency for the encode cache's byte cap; a Load seed served from
            // here is a use, and this is the read path that keeps a
            // stored-once-sampled-forever entry warm.
            crate::runtime::surface_cache::touch_gva(state, target_gva, width, height);
        }
        // This is the reader that keeps `DeviceState::host_gva_surfaces` alive,
        // and the measurement is unambiguous. One driven x86/Vulkan boot (four
        // Safari pages, each scrolled six times then title-bar dragged;
        // `.agents/repros/gva-seed-serve-census.sh`) served **1 558 colour LOAD
        // seeds from this lookup and missed 0**. `load_seed_ok_color` was 1 558
        // in the same window, so every colour LOAD seed the device produced came
        // from here; the other 1 462 of `load_seed_ok` are mapper-ref-texture and take
        // `resolve_mapper_ref_texture_load_seed`.
        //
        // That is what a LOAD seed is worth: `MTLLoadActionLoad` says the guest
        // is drawing onto the content already in this attachment, so a seed that
        // is not found leaves every texel the pass does not itself draw
        // undefined — a rectangle of a compositing layer going blank until
        // something redraws the whole thing.
        //
        // So the map is load-bearing, and the deletion its cost invites is not
        // available. Two other readers of it were removed on measurements that
        // said the opposite (the sampled rung, 0 serves in 286 800 attempts);
        // this one is why the map, its byte cap and its eviction policy stay.
        // Whether the address this seed is about to be served from still names
        // the pages the pixels were produced over.
        //
        // # The level census and the serve census disagree, and the serve one wins
        //
        // `host_cache_levels` reports this for the map as a whole and reads
        // alarming: 25 moved and 149 unmapped of 176 entries on a driven boot
        // (31/105/138 on the one before). Read as a hazard that says most of
        // the cache would hand a LOAD seed some other allocation's pixels.
        //
        // It does not. A level is not a serve, and asked at the serve site the
        // same question answers differently — one driven x86/Vulkan boot
        // (Spotlight, Mission Control, Notification Center, Finder gallery/icon
        // + corner resize, apple.com scroll, Wikipedia, title-bar drag, window
        // closes, wallpaper drag):
        //
        //   gva_seed_backing_same        536      (load_seed_ok_color = 537)
        //   gva_seed_backing_moved         0
        //   gva_seed_backing_unmapped      0
        //   gva_seed_backing_unrecorded    0
        //
        // The two populations are disjoint: the entries the guest re-points or
        // unmaps are not the entries a LOAD seed reads. So there is no
        // wrong-content hazard on this reader, and the moved/unmapped bulk is
        // dead weight rather than a defect — which is a claim about what to
        // evict, not about what to serve.
        //
        // That reading is now taken by the serve's own admission verdict below
        // rather than by a census beside it. `gva_seed_verdict` answers the same
        // four states and one more — whether the entry is even this task's — and
        // two spellings of one question is how the two ends of a rail come to
        // disagree. Its `route()` names carry the counts; the old
        // `gva_seed_backing_*` keys are gone, so a boot log or a `kb/` entry
        // quoting them predates this.
        //
        // The zero above also has to be read with its blind spot: that census
        // walked the page table of the task that *stored* the entry, so it could
        // never have reported a second task asking at the same address. Its zero
        // was a measured zero for freshness and no evidence at all for
        // ownership.
        // Which door served, and — for the ref door — whether the pixels it
        // holds were produced over the address this seed is being served as.
        //
        // `load_seed_ok_color` counts both doors as one, so the ref door has
        // never had a reading of its own. It matters because the two doors carry
        // different guarantees: the GVA door's key *is* the allocation and the
        // block above asks whether that allocation still names the same pages,
        // while the ref door's key is an object-list slot the guest reuses and
        // its entry carries no page identity at all. A ref-door serve whose
        // `source_gva` differs from `target_gva` is this seed handing the pass
        // another allocation's picture as its prior content — and because the
        // matching Store writes the composite back, the next frame loads what
        // this one stored.
        //
        // The ref door only answers for the allocation its pixels came from.
        //
        // A LOAD seed is the attachment's *prior content*, and the matching Store
        // writes the composite back — so a door that hands the pass another
        // allocation's picture arms the next frame to load what this one stored.
        // The GVA door cannot do that: its key *is* the allocation, and the block
        // above asks whether that address still names the same pages. The ref
        // door's key is an object-list slot the guest reuses, and its entry
        // carries no page identity, so `source_gva` — the address the producing
        // Store rendered into — is the only thing that can separate "the GVA
        // entry aged out of its byte cap and this is the same allocation" from
        // "the guest re-pointed this texture and this is the previous one".
        //
        // An entry that cannot say where its pixels came from is refused for the
        // same reason, and so is a target with no address of its own: neither can
        // establish that these are this attachment's bytes. Refusing costs a
        // guest re-read below (`load_sampled_rgba_static`), never a lost seed —
        // the guest's own pages are the authoritative source and any deferred
        // window over this address was already landed at the top of this
        // function.
        //
        // Measured on a driven x86/Vulkan boot: the ref door was asked twice in
        // 3 066 seeds and held nothing both times, so the population this gates
        // is small on this pathway. It is not measurable on Metal from here —
        // `host_cache_store_gva_layer` is Vulkan-only, but the compute mirror in
        // `surface_cache::mirror_linear_color_cache` is not — which is why this
        // is a currency test rather than the removal the zero would otherwise
        // invite.
        // The GVA door's own admission test. `has_gva` answers "is there an
        // entry of this geometry"; it does not answer "is this entry this
        // task's, and does its address still name the pages it was stored
        // over". Those are the two the ref door has always been gated on in
        // spirit, and `gva_seed_verdict`'s doc records why the argument that the
        // GVA key *is* the allocation does not survive contact with a second
        // address space.
        // `has_gva` stays the existence gate — is there an entry of this
        // geometry — and the verdict only ever *removes* one from the door.
        //
        // It removes on **positive evidence** and nothing else: another task's
        // address space, or an address the guest has re-pointed. `Unmapped` and
        // `Unrecorded` keep serving, which is not laxity. `GvaBackingState`'s own
        // doc records that a failed walk is transient here (this device
        // routinely asks before the guest has finished mapping) and that an
        // entry stored with no backing at all is a question that *cannot be
        // asked*, not one answered "stale". Refusing those two as well regressed
        // `color_load_seed_uses_provenance_and_preserves_black`, which stores an
        // entry with `backing: None` and requires it served — so the two-state
        // rule is the measured one, not a cautious guess.
        let gva_present = target_gva != 0
            && crate::runtime::surface_cache::has_gva(state, target_gva, width, height);
        let gva_verdict = gva_present.then(|| {
            let verdict =
                crate::runtime::surface_cache::gva_seed_verdict(state, host, task_id, target_gva);
            crate::runtime::drain::note_store_route(verdict.route());
            verdict
        });
        let gva_served = matches!(
            gva_verdict,
            Some(
                crate::runtime::surface_cache::GvaSeedVerdict::Admit
                    | crate::runtime::surface_cache::GvaSeedVerdict::Unmapped
                    | crate::runtime::surface_cache::GvaSeedVerdict::Unrecorded
            )
        );
        // The ref door is the GVA door's fallback, so it may not be the more
        // permissive of the two. `GuestHolds` is a statement about *this
        // address*: the guest's own pages hold these bytes and track the guest
        // CPU, which no host-side copy does. The two caches are stored from one
        // call over one frame, so serving the ref entry here is serving the
        // refused entry under another key.
        //
        // The other refusals do not travel. `OtherTask` and `Moved` are
        // statements about the GVA *entry* — whose address space it was recorded
        // in, and whether the address still names those pages — and the ref
        // door's own `source_gva` test already answers the same question for
        // its own entry.
        let ref_blocked = matches!(
            gva_verdict,
            Some(crate::runtime::surface_cache::GvaSeedVerdict::GuestHolds)
        );
        let ref_served = !gva_served
            && !ref_blocked
            && texture_ref != 0
            && target_gva != 0
            && crate::runtime::surface_cache::texture_source_gva(
                state,
                task_id,
                texture_ref,
                width,
                height,
            ) == Some(target_gva);
        if gva_served {
            crate::runtime::drain::note_store_route("load_seed_color_from_gva");
        } else if texture_ref != 0 {
            // The denominator. A door that served nothing because the GVA door
            // always won and one that was asked and refused read identically at
            // zero, and only the second says what this gate costs.
            crate::runtime::drain::note_store_route("load_seed_color_ref_asked");
            crate::runtime::drain::note_store_route(if ref_served {
                "load_seed_color_from_ref"
            } else {
                "load_seed_color_ref_refused"
            });
        }
        let cached = if gva_served {
            crate::runtime::surface_cache::get_gva(state, target_gva, width, height)
        } else if ref_served {
            crate::runtime::surface_cache::get_texture(state, task_id, texture_ref, width, height)
        } else {
            None
        };
        if let Some(bgra) = cached {
            return Some(swap_rb_channels(bgra));
        }
        // The third door, and the only one a target with no address of its own
        // can reach. Both doors above are keyed on `target_gva`, so a rail whose
        // colour attachments are mapper-ref-texture surfaces rather than GVA
        // allocations — which is every colour target on the Metal rail — closed
        // them by construction: one driven macos-13 Metal boot asked the ref
        // door 201 times, was refused 201 times, and paid the full guest read
        // for every colour LOAD in the boot. That read is the largest single
        // cost on that rail's per-draw chain (`chain_phase`'s `seed_us`, 6.09 ms
        // a draw over 831 draws, of which 99 % is this).
        //
        // Keyed on the mapping, which is the identity a mapper-ref-texture
        // surface actually has. The entry is written by the neutral
        // `mapping_write`, which both rails' writebacks go through and which
        // stamps the guest-write witness in the same breath — so `Clean` here
        // means "no guest store since this device published these pixels", and
        // the copy is exactly the attachment's prior content.
        if let Some(seed) =
            seed_from_published_surface(state, host, task_id, texture_ref, width, height)
        {
            return Some(seed);
        }
    }
    // normal-texture (or texture-view base) linear GVA → convert to RGBA8.
    //
    // No settle at this fork. It used to sit at the head of this function, above
    // every host-cache lookup, and blocked 5 023 times for 2.63 s on a driven
    // Safari-drag boot serving seeds that never touched guest memory. Moving it
    // here narrowed it to the branch that reads, and then the branch turned out
    // to be three leaves that each know their own span while this fork knows
    // none: a settle here has to assume the whole of guest RAM.
    //
    // So each leaf under `load_sampled_rgba_static` owns it, narrowed on what it
    // actually reads — `read_buffer_bytes_resolved` on the buffer's span,
    // `scanout::paint_mapping` behind `load_mapper_ref_texture_mapping_rgba`, and
    // `draw::texture_view::load_linear_texture_impl` for the linear arm. The
    // buffer leaf had no settle at all before that, on any of its four callers.
    // The seed arm: this leaf is shared with the sampled resolve and the two
    // want opposite repairs, so it is charged separately.
    // A colour LOAD seed is copied into a render target through the RGBA8-shaped
    // seed path, so this arm takes no native layout — the bytes must be what
    // that path reads them as.
    let (rgba, _layout) = load_sampled_rgba_static(
        state,
        host,
        task_id,
        texture_ref,
        NativeUploads::NONE,
        crate::runtime::render_writeback::SettleSite::LinearTextureSeed,
    )?;
    Some(rgba)
}

/// This device's own last publication of a mapper-ref-texture surface, when the
/// hypervisor's witness says the guest has not repainted it since.
///
/// # Why this door takes the strict standard
///
/// A LOAD seed is the attachment's *prior content*: the pass composites onto it
/// and the matching Store publishes the composite back over the surface's guest
/// pages. So a stale serve is not a frame that the next rung corrects — it is a
/// frame that becomes the surface, and the frame after loads what this one
/// stored. There is no rung under this one that reads the entry again.
///
/// [`CurrencyStandard::WatchedAndUnwritten`] is what makes that safe on a
/// pathway whose dirty-tracking witness may never arm. Under the permissive
/// standard a rail that never stamps answers `NoStamp` to every ask, and this
/// door would then serve whatever the cache holds, unconditionally — a
/// compositing layer frozen on the last frame this device drew into it. Under
/// the strict standard the same rail simply never serves and pays the guest read
/// it pays today.
///
/// The miss is cheap and the wrong serve is not, which is the whole asymmetry.
///
/// # What it does not cover
///
/// Only a *direct* mapper-ref-texture reference. A texture view onto a
/// mapper-ref-texture base resolves through `load_sampled_rgba_static`'s view
/// rung and is not asked about here, because the view's format override can
/// reinterpret the storage and the cache entry records no such reinterpretation.
///
/// The order against `load_sampled_rgba_static`'s own first rung is safe by
/// construction rather than by care: that rung takes `OBJECT_TYPE_TEXTURE_VIEW`
/// and [`objects::resolve_mapper_ref_texture`] takes
/// `OBJECT_TYPE_MAPPER_REF_TEXTURE`, and an object has one type.
///
/// # No census here
///
/// Two callers ask this and they ask it for different reasons — one wants the
/// bytes, the other wants to know whether a render target it already holds is
/// still the frame — so each names the outcome under its own keys. Pooling them
/// into one counter would make the number unreadable for either, which is the
/// same rule [`crate::runtime::mapper::mapping_guest_write_verdict`]'s own doc
/// records for the verdict underneath it.
pub(crate) fn published_surface_frame<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    width: u32,
    height: u32,
) -> Result<PublishedFrame, NoPublishedFrame> {
    if texture_ref == 0 {
        return Err(NoPublishedFrame::NotMapped);
    }
    let Some(mapping_id) = objects::resolve_mapper_ref_texture(state, host, task_id, texture_ref)
    else {
        return Err(NoPublishedFrame::NotMapped);
    };
    published_mapping_frame(state, host, mapping_id, width, height)
}

/// [`published_surface_frame`] for a caller that has already resolved the
/// mapping.
///
/// The half that asks the question, split from the half that finds the surface,
/// because the sampled path resolves the mapping to read its *geometry* before
/// it can name the window to ask about — and resolving a second time is a second
/// lookup that could answer differently.
///
/// Takes `state` by shared reference, which is the statement that asking this
/// changes nothing.
pub(crate) fn published_mapping_frame<M: HostOps>(
    state: &DeviceState,
    host: &M,
    mapping_id: u32,
    width: u32,
    height: u32,
) -> Result<PublishedFrame, NoPublishedFrame> {
    if mapping_id == 0 || width == 0 || height == 0 {
        return Err(NoPublishedFrame::NotMapped);
    }
    let currency =
        crate::runtime::surface_currency::surface_currency(state, host, mapping_id, width, height);
    if !currency.serves(crate::runtime::surface_currency::CurrencyStandard::WatchedAndUnwritten) {
        return Err(NoPublishedFrame::Uncurrent(mapping_id, currency));
    }
    let Some(generation) =
        crate::runtime::surface_cache::frame_generation(state, mapping_id, width, height)
    else {
        return Err(NoPublishedFrame::Unpublished(mapping_id));
    };
    Ok(PublishedFrame {
        mapping_id,
        generation,
    })
}

/// A mapper-ref-texture surface whose host-side frame is current, and which
/// frame it is.
///
/// The generation is the point of carrying this as a struct rather than a bool:
/// a rail that keeps its own copy of these pixels compares generations to decide
/// whether its copy is still this one, and a caller that only wants the bytes
/// ignores it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PublishedFrame {
    pub mapping_id: u32,
    /// [`crate::runtime::surface_cache`]'s `host_gen` for this frame. Never 0.
    pub generation: u64,
}

/// Why [`published_surface_frame`] has nothing to offer.
///
/// The three are kept apart because they have different fixes and only the
/// second is about the guest: "this attachment is not one of these surfaces",
/// "the witness will not vouch for the copy", and "the witness is fine and this
/// device has published nothing at this geometry yet".
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum NoPublishedFrame {
    NotMapped,
    Uncurrent(u32, crate::runtime::surface_currency::SurfaceCurrency),
    Unpublished(u32),
}

/// A colour LOAD seed served from [`published_surface_frame`].
///
/// Both doors above this one in [`seed_color_load`] key on `target_gva`, and a
/// mapper-ref-texture attachment has no address of its own — so on the Metal
/// rail, whose colour targets are all mapper-ref-texture surfaces, they were
/// closed by construction. One driven macos-13 boot asked the ref door 201 times
/// and was refused 201 times.
fn seed_from_published_surface<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    width: u32,
    height: u32,
) -> Option<Vec<u8>> {
    use crate::runtime::surface_currency::SurfaceCurrency;

    let frame = match published_surface_frame(state, host, task_id, texture_ref, width, height) {
        Ok(frame) => {
            // The denominator. A door that served nothing because it was never
            // reached and one that was reached and refused read identically at
            // zero, and only the second says what the gate costs.
            crate::runtime::drain::note_store_route("load_seed_color_surface_asked");
            frame
        }
        Err(NoPublishedFrame::NotMapped) => return None,
        Err(decline) => {
            crate::runtime::drain::note_store_route("load_seed_color_surface_asked");
            crate::runtime::drain::note_store_route(match decline {
                // Split from the other refusals because it is the one that is
                // not about the guest: it says this device's witness was never
                // armed for these pages, and it is the counter that decides
                // whether widening this door is even a question.
                NoPublishedFrame::Uncurrent(_, SurfaceCurrency::Unwritten(_)) => {
                    "load_seed_color_surface_unwatched"
                }
                NoPublishedFrame::Uncurrent(_, SurfaceCurrency::WrotePixels(_)) => {
                    "load_seed_color_surface_repainted"
                }
                NoPublishedFrame::Uncurrent(_, SurfaceCurrency::WroteUnknown) => {
                    "load_seed_color_surface_unknown"
                }
                // `serves` admits this under both standards, so reaching it here
                // is a contradiction between the rule and this match rather than
                // a guest behaviour.
                NoPublishedFrame::Uncurrent(_, SurfaceCurrency::WroteElsewhere) => {
                    "load_seed_color_surface_impossible"
                }
                // Current, and this device has published nothing for the
                // surface at this geometry yet. Not a refusal of the witness,
                // and naming it apart is what keeps the `unwatched` count
                // readable. A frame ceded to a rail's resident no longer lands
                // here — `frame_generation` names it, and the two sources below
                // are what decide whether its bytes can be produced.
                NoPublishedFrame::Unpublished(_) => "load_seed_color_surface_empty",
                NoPublishedFrame::NotMapped => "load_seed_color_surface_impossible",
            });
            return None;
        }
    };
    // Two sources for one frame, and the door above has already said which frame
    // it is. The host cache holds the bytes unless the Store that published them
    // ceded to the running rail's resident
    // ([`crate::runtime::mapping_write::FramePublication`]), in which case the
    // rail is asked for the same generation.
    //
    // Falling through silently is what this must not do. A missing colour LOAD
    // seed leaves every texel the pass does not itself draw undefined — a
    // rectangle of a compositing layer going blank until something redraws the
    // whole of it — so both misses are named.
    if let Some(bgra) =
        crate::runtime::surface_cache::get_shared(state, frame.mapping_id, width, height)
    {
        crate::runtime::drain::note_store_route("load_seed_color_from_surface");
        return Some(swap_rb_channels(&bgra));
    }
    let Some(rgba) = crate::backend::selected().published_frame_rgba8(
        state,
        frame.mapping_id,
        width,
        height,
        frame.generation,
    ) else {
        crate::runtime::drain::note_store_route("load_seed_color_surface_unheld");
        return None;
    };
    crate::runtime::drain::note_store_route("load_seed_color_from_resident");
    Some(rgba)
}

/// Resolve sampled texture RGBA without requiring Metal feature (color LOAD seed path).
///
/// Texture-view views with a non-identity swizzle are rejected here: RT materialization does not
/// rematerialize through a remapped view (contract: swizzled views fail for RT/blit).
/// View `pixel_format` still overrides the base format when bpp-compatible.
fn load_sampled_rgba_static<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    native: NativeUploads,
    site: crate::runtime::render_writeback::SettleSite,
) -> Option<(Vec<u8>, SampledByteFormat)> {
    // Opcode-9 buffer-backed texture (texture-view): sample the source buffer directly.
    if let Some(bt) = buffer_texture_descriptor(state, host, task_id, texture_ref, None) {
        let source = bt.desc.pixel_format;
        return load_buffer_texture_rgba(state, host, task_id, texture_ref, &bt).map(
            |(_, _, r)| {
                (
                    r,
                    SampledByteFormat::from_source(TexelLayout::Rgba8, source),
                )
            },
        );
    }
    // Mapper-ref-texture path via resolve.
    if let Some(mid) = objects::resolve_mapper_ref_texture(state, host, task_id, texture_ref) {
        let source = mapping_declared_format(state, mid, None);
        return load_mapper_ref_texture_mapping_rgba(state, host, mid, None).map(|(_, _, r)| {
            (
                r,
                SampledByteFormat::from_source(TexelLayout::Rgba8, source),
            )
        });
    }
    // Texture-view view → base texture + mip + format. The view's SWIZZLE is
    // deliberately not consulted here: it is a property of the view, not of the
    // bytes, and the bind applies it as the image view's component mapping so
    // the GPU performs it at sample time. Refusing here (which this path used
    // to do, silently) dropped the texture from the draw entirely.
    let (tex_ref, level, fmt_override) =
        if let Some(view) = resolve_texture_view(state, host, task_id, texture_ref) {
            (view.base_texture_ref, view.level, view.pixel_format)
        } else {
            (texture_ref, 0, None)
        };
    // Mapper-ref-texture base through a view (format override may reinterpret BGRA storage).
    if let Some(mid) = objects::resolve_mapper_ref_texture(state, host, task_id, tex_ref) {
        if level != 0 {
            return None;
        }
        let source = mapping_declared_format(state, mid, fmt_override);
        return load_mapper_ref_texture_mapping_rgba(state, host, mid, fmt_override).map(
            |(_, _, r)| {
                (
                    r,
                    SampledByteFormat::from_source(TexelLayout::Rgba8, source),
                )
            },
        );
    }
    // The only rung here that can answer in anything but RGBA8. The three above
    // convert unconditionally — `load_buffer_texture_rgba` and
    // `load_mapper_ref_texture_mapping_rgba` have no native arm — so they state the layout
    // they always produced rather than being handed a choice they cannot make.
    // All four still name the guest format their values were read from, because
    // a convert to RGBA8 reorders channels and does not decode.
    load_linear_texture_host(
        state,
        host,
        task_id,
        tex_ref,
        level,
        fmt_override,
        native,
        site,
    )
}

#[cfg(test)]
mod tests;
