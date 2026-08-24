//! Encode one decoded draw for a resolved pipeline and colour target.
//!
//! Loads per-function MTLB containers from the object list, materializes stream
//! binds (vertex/fragment buffers, optional index buffer, viewport/scissor),
//! hands them to the backend, and writes the RGBA result into the IOSurface texture
//! mapping via [`mapping_write`]. [`execution`] resolves semantic execution
//! inputs and submits the draw through the injected executor.

use crate::runtime::Device;
use reims_vgpu_core::pixel_format::{
    self, solid_rgba8, SampledByteFormat, TexelLayout, MTL_FORMAT_BGRA8_UNORM, RGBA8_BPP,
};
// `Decline::slug` on typed draw, coverage, and translation reasons.

use crate::observe::Decline;
// The one downgrade site left in this tree is the secondary colour attachment,
// which is Vulkan-only. The CPU upload rails used to report here too and no
// longer downgrade at all: they carry the source format through to the bind.
use crate::runtime::census::srgb_census;
// Only `vulkan` and the tests read it, through this module's `use super::*`;
use crate::runtime::decode::render::{ColorAttachment, ScissorRect};
use crate::runtime::decode::resource::TextureDescriptor;
use crate::runtime::decode::resource::{
    decode_depth_stencil_descriptor, decode_render_pipeline_descriptor, decode_texture_descriptor,
    BufferTextureDescriptor, DecodeStatus, ObjectKind, RenderPipelineDescriptor,
};
use crate::runtime::gva_mem;
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::mapper;
use crate::runtime::mapping_write;
use crate::runtime::mtlb::{load_mtlb, AirLoadRail};
use crate::runtime::objects;
use reims_vgpu_protocol::pass_action::{is_declared_load_action, is_declared_store_action};
#[cfg(test)]
use reims_vgpu_protocol::pass_action::{
    MTL_LOAD_ACTION_CLEAR, MTL_LOAD_ACTION_DONT_CARE, MTL_LOAD_ACTION_LOAD, MTL_STORE_ACTION_STORE,
};

// The resolved execution half of this path. Re-exported flat so callers keep naming its items
// `crate::runtime::draw::<name>`.
mod execution;
// Only for `exec`'s pass-extent census, which declares its own copy of these
// bands because it runs on every backend. See
// `the_two_coverage_censuses_use_the_same_bands`.
#[cfg(test)]
pub(crate) use execution::coverage_band_for_test;
pub use execution::*;

// Immutable semantic request plus its pre-submission completion route. This is
// the seam between draw planning and Store/observation control flow.
mod prepared_draw;
use prepared_draw::*;

// Completion diagnostics are downstream of both planning and execution. They
// inspect output but own no execution or Store decisions.
mod observation;
use observation::*;

// Fixed-function guest ordinals normalize once into complete semantic state.
mod fixed_state;
use fixed_state::*;

// Semantic target identity, resident-content currency, and Store publication.
// Kept beside execution because these rules name guest-visible ownership and
// lifetime, not the native executor mechanism used to realize them.
mod resident;
pub(crate) use resident::*;

// Guest resource lookup, content-authority resolution, and construction of
// semantic sampled inputs. Native draw assembly consumes this boundary.
mod sampled_source;
pub(crate) use sampled_source::compute_iosurface_resident_sample;
pub(crate) use sampled_source::declared_guest_image_allocation;
use sampled_source::*;
pub(crate) use sampled_source::{task_gva_guest_run_source, WindowRefusal};

pub(crate) fn compute_gva_resident_sample<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    gva: u64,
    row_stride: u32,
    width: u32,
    height: u32,
    format: u16,
) -> Option<crate::model::TargetIdentity> {
    sampled_source::gva_resident_if_current(
        state,
        host,
        task_id,
        sampled_source::GvaSpan {
            texture_ref,
            gva,
            row_stride,
            width,
            height,
            format,
        },
    )
    .ok()
}

// Type-8 texture-view resolution and linear texture loads. Backend-independent,
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

// Decoded pass attachments become one complete semantic target set here. A
// malformed slot refuses the set instead of letting orchestration execute a
// reduced pass.
mod attachment_plan;
pub use attachment_plan::mrt_draw_request;

pub use reims_vgpu_core::{
    MAX_ANY_BIND_SLOTS, MAX_BUFFER_BIND_SLOTS, MAX_SAMPLER_BIND_SLOTS, MAX_TEXTURE_BIND_SLOTS,
};

pub type DrawPreparationDecline =
    reims_vgpu_core::DrawPreparationDecline<crate::runtime::executor::M2vCacheDecline>;
pub type RenderMtlbPair = (std::sync::Arc<[u8]>, std::sync::Arc<[u8]>);

/// Which of the three argument tables a bind record names.
///
/// The three constants above are compared against a guest slot index in exactly
/// one place — [`BindTableClass::table`] — and every consumer asks it rather
/// than spelling its own comparison. Before that, the same rule was written out
/// at twenty-two sites across four files in two spellings, one of them inverted,
/// and the consumers of one wire form had drifted into different behaviors for
/// identical input.
///
/// [`crate::runtime::exec`] adds the census vocabulary — Apple's own table size
/// for the class, the reach bands, the drop slug — as its own `impl` on this
/// type, because those describe how a loss is *reported* rather than what the
/// table *is*.
pub use reims_vgpu_core::BindTableClass;

/// A live bind in one draw request whose slot no argument table of its class can
/// name.
///
/// Carries the object ref as well as the slot, because the two say different
/// things: the slot names which table ran out, and the ref is what the guest
/// still believes is bound there.
pub use reims_vgpu_core::PastTableBind;

/// The first live bind in `req` that names a slot past its class's table, if any.
///
/// # Why this is checked once instead of at each consumer
///
/// A slot past the table is not a bind that can be degraded: the encoder has no
/// argument-table entry to put it in, and the guest API answers an
/// out-of-range argument-table index with a process-aborting exception rather
/// than an error. So the only faithful answer is to refuse the whole draw and
/// say which slot did it — the same answer for all three classes, both stages
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
/// wrong is a host failure that takes the process down, and because the check
/// that once stood at each consumer had already drifted.
pub fn first_bind_past_table(req: &DrawEncodeRequest) -> Option<PastTableBind> {
    use reims_vgpu_core::ShaderStage as Stage;

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
/// One function owns the rule because a divergence between consumers would be
/// a difference in *geometry* — a mesh fetched at the wrong stride still
/// `ReimsVgpuBuffer::attribute_stride`; the Vulkan arm reads it into
/// `AttrKey::stride`, where it is already part of the pipeline key.
///
/// A stride wider than `u32` is left to the pipeline's own: it cannot reach
/// Vulkan because `VkVertexInputBindingDescription::stride` is 32-bit, and silently
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

/// Bring a frame into the channel order a consumer wants, in place and only when
/// the two disagree.
///
/// The parameters name what the buffer *holds* and what is *wanted*, rather than
/// a direction, because that is what makes the call sites auditable: a readback's
/// order is a property of the attachment it came out of, so the caller states the
/// fact it was told and the ordering logic lives here. Spelled as a direction
/// ("swizzle if IOSurface texture") each site would re-derive the predicate, which is how
/// the two halves of a conversion end up disagreeing.
///
/// The exchange is an involution, so one routine serves both directions. Trailing
/// bytes that do not fill a whole pixel pass through untouched, matching
/// [`swap_rb_channels`].
#[inline]
fn reorder_rb_in_place(px: &mut [u8], have_bgra: bool, want_bgra: bool) {
    if have_bgra == want_bgra {
        return;
    }
    for p in px.chunks_exact_mut(4) {
        p.swap(0, 2);
    }
}

/// Whether a bound vertex buffer at Metal index `idx` must be exposed to the
/// engine as a StorageBuffer descriptor (binding `idx`).
///
/// A non-stage-in vertex buffer always is (its only consumer is a direct
/// `[[buffer(idx)]]` access). A stage-in buffer normally is NOT — its bytes flow
/// through the vertex-attribute path. The exception, and the reason this helper
/// exists: a buffer can be BOTH — the pipeline vertex descriptor declares
/// attributes on it while the vertex function *also* reads it directly as a
/// StorageBuffer. WebKit's glyph vertex shader is exactly that (a stride-48
/// stage-in is declared but never read; the function indexes the same buffer as
/// a per-glyph record array — `StorageBuffer` binding 1 — by `gl_InstanceIndex`).
/// Detection is purely structural: a buffer binding at `idx` in the adopted
/// reflection for the translated vertex module. Never keyed on a
/// shader/struct/variable name.
fn vertex_buffer_needs_storage_binding(
    reflection: &reims_vgpu_core::ShaderInterface,
    idx: u32,
    is_stage_in: bool,
) -> bool {
    !is_stage_in
        || reflection.bindings.iter().any(|binding| {
            binding.kind == reims_vgpu_core::ShaderResourceKind::Buffer
                && binding.metal_index == idx
        })
}

/// Which directly-bound Metal resource class a [`FragUnbound`] names.
///
/// Carried as a type rather than as the `buf`/`tex`/`smp` prefix this used to be
/// formatted into, because consumers need the class as a semantic value and
/// must not parse it back out of a string. `Display` is the only place the
/// prefix exists now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FragUnboundClass {
    Buffer,
    Texture,
    Sampler,
}

impl std::fmt::Display for FragUnboundClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Buffer => "buf",
            Self::Texture => "tex",
            Self::Sampler => "smp",
        })
    }
}

/// One directly-bound fragment resource the shader declares and the draw did not
/// provide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FragUnbound {
    pub class: FragUnboundClass,
    /// The Metal argument index used to resolve the reflected descriptor.
    pub metal_index: u32,
}

impl std::fmt::Display for FragUnbound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.class, self.metal_index)
    }
}

/// The fragment texture slots that must remain explicitly null: gaps the scan
/// flagged as textures that the module *statically uses*.
///
/// Vulkan requires the pipeline layout to contain a descriptor for every
/// statically-used resource, and `engine/exec.rs` builds that layout from
/// provided resources alone — so one of these left alone is not an unwritten
/// descriptor, it is a binding the layout does not mention. Mesa's Intel driver
/// scores each used binding as `(use_count << 7) / array_size` over an array it
/// sized to `max_binding + 1` and zero-filled, so the omission divides by zero
/// and kills the host process inside pipeline creation rather than returning an
/// error. That is why this population has to be filled rather than only counted.
///
/// Three narrowings, each load-bearing:
///
/// - **Textures only.** Samplers have their own null-descriptor rail, while a
///   storage buffer has no contract-defined replacement; the latter is still
///   reported by the caller.
/// - **[`DescriptorUse::Used`] only.** A declared-and-never-referenced variable
///   is legal to omit and must stay omitted, or the census that separated those
///   two populations cannot tell them apart any more.
/// - **Not `Ambiguous`.** Two variables on one binding is its own defect and is
///   not repaired by picking one of them; `is_violation` already excludes it.
fn frag_unbound_textures_to_bind_null(
    uses: &[(FragUnbound, reims_vgpu_core::DescriptorUse)],
) -> Vec<u32> {
    uses.iter()
        .filter(|(gap, use_)| use_.is_violation() && gap.class == FragUnboundClass::Texture)
        .map(|(gap, _)| gap.metal_index)
        .collect()
}

/// One pass over the fragment reflection finding standard directly-bound kinds
/// (`[[buffer(n)]]` / `[[texture(n)]]` / `[[sampler(n)]]`) the shader declares
/// but the draw never provided. Each names a descriptor the translated SPIR-V
/// references yet the Vulkan engine leaves unbound—an undefined read that
/// paints garbage. `ColorInput` / `ThreadgroupBuffer` are served elsewhere;
/// storage textures are declined before this scan because the render engine has
/// no storage-image descriptor path.
///
/// Membership is by caller-supplied predicates so the hot (all-bound) path
/// allocates nothing unless a genuine gap exists, which is near-never on a
/// healthy boot. Unsupported reflected resource families are refused before
/// this scan and therefore have no second classification here.
fn frag_unbound_scan(
    bindings: &[reims_vgpu_core::ShaderResourceBinding],
    has_buf: impl Fn(u32) -> bool,
    has_tex: impl Fn(u32) -> bool,
    has_smp: impl Fn(u32) -> bool,
    tex_declared_in_module: impl Fn(u32) -> bool,
) -> Vec<FragUnbound> {
    use reims_vgpu_core::ShaderResourceKind;
    let mut unbound: Vec<FragUnbound> = Vec::new();
    for rb in bindings {
        let (cls, provided) = match rb.kind {
            ShaderResourceKind::Buffer => (FragUnboundClass::Buffer, has_buf(rb.metal_index)),
            ShaderResourceKind::Texture | ShaderResourceKind::TextureArray => {
                (FragUnboundClass::Texture, has_tex(rb.metal_index))
            }
            ShaderResourceKind::Sampler => (FragUnboundClass::Sampler, has_smp(rb.metal_index)),
            _ => continue,
        };
        if provided {
            continue;
        }
        // A texture the reflection names but the module never declares is not a
        // gap. The reflection comes from the AIR entry point's signature, so a
        // Metal function that lists `[[texture(n)]]` and never samples it
        // produces an entry for a descriptor the translated SPIR-V does not
        // carry — nothing references the binding, so nothing is unbound.
        //
        // Asked of textures only, because that is the class observed firing and
        // its reflected binding is the one this caller can compute. A
        // buffer or sampler reported here is still worth reading as before.
        if matches!(
            rb.kind,
            ShaderResourceKind::Texture | ShaderResourceKind::TextureArray
        ) && !tex_declared_in_module(rb.metal_index)
        {
            continue;
        }
        unbound.push(FragUnbound {
            class: cls,
            metal_index: rb.metal_index,
        });
    }
    unbound
}

/// Ask the specification's own question of one reported gap: does the module
/// *statically use* the descriptor its layout does not contain?
///
/// The scan above stops at declaration because that is what it can answer per
/// draw for the price of a decoration walk. Declaration is not the bar: Vulkan
/// requires the pipeline layout to contain a descriptor for every resource the
/// shader references, and a declared-and-never-referenced variable is legal to
/// omit. So this is the difference between a specification violation and a
/// harmless reflection artefact, and until it is asked the fail line is naming a
/// population it cannot tell apart.
///
fn frag_unbound_static_use(
    gap: &FragUnbound,
    variant: &reims_vgpu_core::PreparedShaderVariant,
) -> reims_vgpu_core::DescriptorUse {
    match gap.class {
        FragUnboundClass::Buffer => variant.buffer_use(gap.metal_index),
        FragUnboundClass::Texture => variant.texture_use(gap.metal_index),
        FragUnboundClass::Sampler => variant.sampler_use(gap.metal_index),
    }
}

/// Whether an unprovided fragment resource still represents guest work loss.
///
/// A statically-used texture is repaired with an explicit null descriptor by
/// the caller. Declared-but-unused and absent executable variables require no
/// descriptor under Vulkan. Everything else remains visible: an unprovided
/// used buffer or sampler loses work, while an ambiguous binding cannot
/// authorize either omission or substitution.
fn frag_unbound_requires_report(gap: FragUnbound, use_: reims_vgpu_core::DescriptorUse) -> bool {
    match use_ {
        reims_vgpu_core::DescriptorUse::Used => gap.class != FragUnboundClass::Texture,
        reims_vgpu_core::DescriptorUse::Ambiguous => true,
        reims_vgpu_core::DescriptorUse::NotDeclared
        | reims_vgpu_core::DescriptorUse::DeclaredUnused => false,
    }
}

/// Decode the depth-stencil descriptor a draw bound, on the Linux path
/// `load_render_pipeline`: object-list lookup → descriptor read → decode (which
/// validates the depth-stencil construction opcode). Returns the specific reason slug on
/// failure so the caller — which only reaches this for a bound `ds_ref != 0`, i.e.
/// a guest that explicitly asked for a depth-stencil state — can fail-visibly
/// name why the state silently fell back to no-depth instead of dropping it into
/// the same silent hole every other depth/stencil sub-case is instrumented against.
fn load_depth_stencil_descriptor<M: HostMemory + HostOps>(
    state: &Device,
    host: &M,
    task_id: u32,
    ds_ref: u32,
) -> Result<crate::runtime::decode::resource::DepthStencilDescriptor, &'static str> {
    if let Some(state_) = state
        .task_objects
        .depth_stencil
        .get(task_id, reims_vgpu_protocol::SerializerRef::new(ds_ref))
    {
        crate::runtime::drain::note_store_route("ds_state_held");
        return Ok((*state_).clone());
    }
    let (_entry, desc) = objects::resolve_descriptor(
        state,
        host,
        task_id,
        ds_ref,
        &[ObjectKind::SerializerResource],
    )
    .map_err(crate::observe::ladder_slugs!("depth_stencil"))?;
    let decoded = decode_depth_stencil_descriptor(&desc)
        .map_err(|_| crate::observe::ladder_slug!("depth_stencil", desc_decode))?;
    // Registered only after a successful decode, on the same terms as
    // `resolve_sampler_state`: a descriptor still being published can succeed on
    // retry, and retaining a failure would make that retry impossible.
    crate::runtime::drain::note_store_route("ds_state_constructed");
    Ok((*state.task_objects.depth_stencil.register(
        task_id,
        reims_vgpu_protocol::SerializerRef::new(ds_ref),
        std::sync::Arc::new(decoded),
    ))
    .clone())
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
    /// time, on `ReimsVgpuBuffer::attribute_stride`.
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

#[derive(Clone, Debug)]
pub struct IndexedDrawInfo {
    pub index_type:
        Result<reims_vgpu_protocol::IndexType, reims_vgpu_protocol::IndexTypeDecodeError>,
    pub index_count: u32,
    pub index_buffer_ref: u32,
    pub index_buffer_offset: u64,
    /// Index offset within the buffer, still in elements. It is converted to
    /// bytes only after `index_type` has decoded successfully.
    pub index_start: u32,
    /// Metal `baseVertex` / Vulkan `vertexOffset`, added to every index before
    /// the vertex fetch. Signed, because Metal's is, and because a negative one
    /// read as unsigned becomes a huge index rather than an error.
    pub base_vertex: i64,
}

impl IndexedDrawInfo {
    fn raw_index_type(&self) -> u32 {
        match self.index_type {
            Ok(value) => value.guest_ordinal(),
            Err(error) => error.raw,
        }
    }

    fn resolved_byte_offset(&self) -> Result<u64, IndexLoadReason> {
        let width = self
            .index_type
            .map_err(|_| IndexLoadReason::TypeUnsupported)?
            .byte_size() as u64;
        let start_delta = u64::from(self.index_start)
            .checked_mul(width)
            .ok_or(IndexLoadReason::OffsetOverflow)?;
        self.index_buffer_offset
            .checked_add(start_delta)
            .ok_or(IndexLoadReason::OffsetOverflow)
    }
}

/// The allocation namespace and byte geometry of one colour attachment.
///
/// A type-2/3 target is a plane inside the allocation declared by its texture
/// descriptor. Keeping that relationship intact is what lets a backend retain
/// and import the allocation without reconstructing ownership from the plane's
/// address. The enum also makes the mapping and GVA forms mutually exclusive;
/// there is no `(mapping_id, target_gva)` pair whose invalid combinations every
/// consumer has to remember to reject.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ColorTargetStorage {
    #[default]
    None,
    Mapping(u32),
    Linear(LinearColorTarget),
}

impl ColorTargetStorage {
    pub fn mapping_id(&self) -> u32 {
        match self {
            Self::Mapping(mapping_id) => *mapping_id,
            Self::None | Self::Linear(_) => 0,
        }
    }

    pub fn linear(&self) -> Option<&LinearColorTarget> {
        match self {
            Self::Linear(linear) => Some(linear),
            Self::None | Self::Mapping(_) => None,
        }
    }

    pub fn target_gva(&self) -> u64 {
        self.linear().map_or(0, LinearColorTarget::target_gva)
    }

    pub fn row_stride(&self) -> u32 {
        self.linear().map_or(0, |linear| linear.row_stride)
    }
}

pub use reims_vgpu_core::LinearColorTarget;

/// One color RT for MRT encode/writeback.
///
/// Archive `ApplePVGPURenderTarget`: either IOSurface or a type-2/3
/// guest-allocation plane. Wallpaper/background layers are the latter form.
#[derive(Clone, Debug, Default)]
pub struct ColorRtRequest {
    pub slot: u32,
    pub texture_ref: u32,
    /// The attachment object retained when the render encoder recorded it.
    /// References may be deleted and reused before execution; attachment state
    /// belongs to this object lifetime, just like [`TextureBind::resource`].
    pub resource: Option<std::sync::Arc<crate::model::TaskResource>>,
    pub storage: ColorTargetStorage,
    pub width: u32,
    pub height: u32,
    pub format: u16,
    /// Sample count of the attachment texture (the multisample source when a
    /// separate resolve texture is present).
    pub sample_count: u32,
    pub load_action: reims_vgpu_protocol::pass_action::LoadAction,
    pub store_action: reims_vgpu_protocol::pass_action::StoreAction,
    pub clear_color: [f64; 4],
    pub target_seed_rgba: Option<Vec<u8>>,
    /// Multisample attachment discarded into this request's single-sample
    /// target at pass end. Zero for an ordinary colour attachment.
    pub multisample_source_ref: u32,
}

/// Authoritative source of colour0's prior contents for a GVA attachment LOAD.
///
/// Kept typed until backend preparation because the two backends consume the
/// copied render target, while Vulkan may retain/import them and issue the
/// buffer-to-image copy on the GPU. Collapsing `GuestPages` into
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GvaLoadSource {
    #[default]
    None,
    Resident,
    GuestPages,
}

impl ColorRtRequest {
    /// Whether pass completion makes one texel per pixel available in this
    /// request's target storage.
    ///
    /// A multisample `Store` preserves the attachment's samples in the
    /// attachment itself; it does not collapse them into the linear guest
    /// allocation. Only a resolve publishes a single-sample destination.
    pub const fn publishes_single_sample(&self) -> bool {
        use reims_vgpu_protocol::pass_action::StoreAction;

        match self.store_action {
            StoreAction::DontCare => false,
            StoreAction::Store => self.sample_count <= 1,
            StoreAction::MultisampleResolve | StoreAction::StoreAndMultisampleResolve => {
                self.multisample_source_ref != 0
            }
        }
    }

    /// Whether pass completion must retain the attachment image and all of
    /// its samples.
    pub const fn preserves_attachment_samples(&self) -> bool {
        matches!(
            self.store_action,
            reims_vgpu_protocol::pass_action::StoreAction::Store
                | reims_vgpu_protocol::pass_action::StoreAction::StoreAndMultisampleResolve
        )
    }

    pub fn mapping_id(&self) -> u32 {
        self.storage.mapping_id()
    }

    pub fn linear_target(&self) -> Option<&LinearColorTarget> {
        self.storage.linear()
    }

    pub fn target_gva(&self) -> u64 {
        self.storage.target_gva()
    }

    pub fn row_stride(&self) -> u32 {
        self.storage.row_stride()
    }
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
    /// Decoded `MTLVisibilityResultMode`. Disabled is the `None` around this;
    /// unknown ordinals never cross the stream-normalization boundary.
    pub mode: reims_vgpu_protocol::VisibilityResultMode,
    /// Byte offset into the pass's `visibilityResultBuffer`.
    pub offset: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DepthAttachmentState {
    pub texture_ref: u32,
    pub load_action: reims_vgpu_protocol::pass_action::LoadAction,
    pub store_action: reims_vgpu_protocol::pass_action::StoreAction,
    pub clear_depth: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StencilAttachmentState {
    pub texture_ref: u32,
    pub load_action: reims_vgpu_protocol::pass_action::LoadAction,
    pub store_action: reims_vgpu_protocol::pass_action::StoreAction,
    pub clear_stencil: u32,
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
    /// Exact setter classes changed since the preceding draw in this render
    /// encoder. Complete semantic state still travels beside it.
    pub encoder_delta: reims_vgpu_core::RenderEncoderDelta,
    pub vertex_count: u32,
    pub instance_count: u32,
    pub primitive_topology: reims_vgpu_protocol::PrimitiveTopology,
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
    pub indexed: Option<IndexedDrawInfo>,
    pub blend_color: Option<[f32; 4]>,
    /// Pass-owned raster bounds; unlike attachment geometry, these do not
    /// narrow load or store operations.
    pub render_target_extent: reims_vgpu_core::RenderTargetExtent,
    pub cull_mode: reims_vgpu_protocol::CullMode,
    pub front_face_ccw: bool,
    /// `MTLTriangleFillMode` from `setTriangleFillMode:`, initialized to the
    /// Metal default (`Fill`) until the stream replaces it.
    pub fill_mode: reims_vgpu_protocol::FillMode,
    /// Encoder line width, bit-preserving so backend capability projection can
    /// distinguish every guest value.
    pub line_width: reims_vgpu_core::LineWidth,
    /// `MTLDepthClipMode` from `setDepthClipMode:`, initialized to the Metal
    /// default (`Clip`) until the stream replaces it.
    pub depth_clip_mode: reims_vgpu_protocol::DepthClipMode,
    pub depth_bias: Option<[f32; 3]>,
    pub depth_stencil_ref: u32,
    pub stencil_ref: Option<(u32, u32)>,
    pub depth_attach: Option<DepthAttachmentState>,
    /// Canonical resource named by [`Self::depth_attach`].
    ///
    /// The pass descriptor carries a task-local serializer ref. Resolution
    /// replaces it before semantic request construction so backend resident
    /// identity cannot collide across tasks or object-slot reuse.
    pub depth_attachment_resource: Option<std::sync::Arc<crate::model::TaskResource>>,
    pub stencil_attach: Option<StencilAttachmentState>,
    /// Canonical resource named by [`Self::stencil_attach`].
    pub stencil_attachment_resource: Option<std::sync::Arc<crate::model::TaskResource>>,
    /// Records 2+ of a resident render-pass chain: load the prior record's
    /// content from the engine target instead of a CPU seed. Set by the exec
    /// chain loop (Vulkan rail only); default false.
    pub chain_from_resident: bool,
    /// Resolved render memory barriers immediately preceding this draw.
    pub render_barriers: Vec<reims_vgpu_core::RenderBarrier>,
    /// This draw continues the Metal render encoder of the preceding draw in
    /// the same decoded stream. Vulkan may keep an identical render pass open
    /// when no command that is illegal inside it intervenes.
    pub continues_render_pass: bool,
    /// Another draw in this decoded Metal render encoder follows this one.
    /// Vulkan may defer `vkCmdEndRenderPass` until that draw, an outside-pass
    /// command, or the command-buffer flush closes it.
    pub render_pass_continues: bool,
    /// Where colour0's GVA `MTLLoadActionLoad` obtains its prior contents.
    /// `Resident` is discharged by Vulkan against the recomputed allocation
    /// identity; `GuestPages` lets each backend choose its native transport.
    /// Either non-`None` value means `target_seed_rgba` is deliberately absent.
    pub gva_load_source: GvaLoadSource,
}

/// Compact command-level MRT census for the always-on draw proxy.
///
/// This records only decoded render-pass state. It deliberately does not rank
/// targets by dimensions, ids, or content; the point is to expose when the
/// shader/pass contract names more attachments than the backend executes.
fn color_target_diag(colors: &[ColorRtRequest]) -> String {
    colors
        .iter()
        .map(|c| {
            format!(
                "s{}:r{}:mid{}:gva={:#x}:{}x{}:fmt={:#x}:l{}:s{}",
                c.slot,
                c.texture_ref,
                c.mapping_id(),
                c.target_gva(),
                c.width,
                c.height,
                c.format,
                c.load_action,
                c.store_action
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn texture_bind_diag(textures: &[TextureBind]) -> String {
    textures
        .iter()
        .take(8)
        .map(|t| format!("i{}:r{}", t.index, t.texture_ref))
        .collect::<Vec<_>>()
        .join(",")
}

fn buffer_bind_diag(buffers: &[BufferBind]) -> String {
    buffers
        .iter()
        .take(8)
        .map(|b| format!("i{}:r{}+{:#x}", b.index, b.buffer_ref, b.offset))
        .collect::<Vec<_>>()
        .join(",")
}

fn linux_m2v_draw_failure(
    error: &impl crate::observe::Decline,
    req: &DrawEncodeRequest,
) -> crate::observe::Emit {
    let indexed = req
        .indexed
        .as_ref()
        .map(|idx| {
            format!(
                "1:ty{}:n{}:r{}+{:#x}",
                idx.raw_index_type(),
                idx.index_count,
                idx.index_buffer_ref,
                idx.index_buffer_offset
            )
        })
        .unwrap_or_else(|| "0".to_string());
    crate::observe::Emit::decline("linux_m2v_draw", error)
        .field("pipe", req.pipeline_ref)
        .field("task", req.task_id)
        // color0's declared extent, which *is* the pass extent — there is no
        // second geometry on the request for it to disagree with. A request
        // carrying no attachment keeps the `WxH` shape so the field stays
        // greppable.
        .field(
            "geom",
            req.colors
                .first()
                .map(|c| format!("{}x{}", c.width, c.height))
                .unwrap_or_else(|| "0x0".to_string()),
        )
        .field("vtx", req.vertex_count)
        .field("inst", req.instance_count)
        .field("prim", req.primitive_topology.guest_ordinal())
        .field("first", req.first_vertex)
        .field("idx", indexed)
        .field("colors", format!("[{}]", color_target_diag(&req.colors)))
        .field(
            "vbuf",
            format!("[{}]", buffer_bind_diag(&req.vertex_buffers)),
        )
        .field(
            "fbuf",
            format!("[{}]", buffer_bind_diag(&req.fragment_buffers)),
        )
        .field(
            "vtex",
            format!("[{}]", texture_bind_diag(&req.vertex_textures)),
        )
        .field(
            "ftex",
            format!("[{}]", texture_bind_diag(&req.fragment_textures)),
        )
        .field(
            "viewports",
            format!("{:?}", req.viewports).replace(char::is_whitespace, ""),
        )
        .field(
            "scissors",
            format!("{:?}", req.scissors).replace(char::is_whitespace, ""),
        )
}

/// Fire `reason` once per `(pipeline_ref, slug)` so a recurring degradation
/// (for example a whole 3D scene requesting depth LOAD) logs once, not per draw.
/// Returns true the first time a given key is seen.
///
/// Backend-agnostic on purpose: both encode arms degrade, so both need the same
/// degradation without flooding per draw, and reported none.
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
/// variant is the *class* the caller acts on — `BackendUnavailable` makes `exec` fall
/// back to the pass clear, `WritebackFailed` does not — and the payload is which
/// of the rail's checks refused. Before this, six payload-free variants spoke for
/// 27 checks: `BadArgs` alone covered eight, and `draw_encode_fail
/// reason=bad_args` could be a zero-size target, a vertexless draw or an ICB
/// range past the end of its buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncodeStatus {
    Ok,

    MissingPipeline(&'static str),
    MissingMtlb(&'static str),
    BackendFailed(&'static str),
    WritebackFailed(&'static str),
    BadArgs(&'static str),
    /// Nothing landed on the Vulkan rail, so `exec` honours the pass clear.
    BackendUnavailable(&'static str),
    /// The record was well-formed and this device implements no answer for it on
    /// this device does not implement. Recovery is `BackendUnavailable`'s —
    /// nothing was encoded, so honour the pass clear — but the class identifies
    /// an implementation gap rather than an unavailable host operation.
    Unsupported(&'static str),
}

impl crate::observe::Refusal for EncodeStatus {
    fn refusal(&self) -> Option<&'static str> {
        match self {
            // The only non-refusal, and the reason this is a `Refusal` rather
            // than a `Decline`: `Emit::refusal` cannot render a line for it.
            Self::Ok => None,

            Self::MissingPipeline(slug)
            | Self::MissingMtlb(slug)
            | Self::BackendFailed(slug)
            | Self::WritebackFailed(slug)
            | Self::BadArgs(slug)
            | Self::BackendUnavailable(slug)
            | Self::Unsupported(slug) => Some(slug),
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        // The class beside the reason: which recovery path the caller took is
        // not derivable from the slug, and a reader correlating a dropped draw
        // with a black frame needs both.

        vec![("class", self.class().to_string())]
    }
}

impl EncodeStatus {
    /// The variant name, for the `class=` field and for call sites that render
    /// their own line.
    pub fn class(&self) -> &'static str {
        match self {
            Self::Ok => "ok",

            Self::MissingPipeline(_) => "missing_pipeline",
            Self::MissingMtlb(_) => "missing_mtlb",
            Self::BackendFailed(_) => "backend_failed",
            Self::WritebackFailed(_) => "writeback_failed",
            Self::BadArgs(_) => "bad_args",
            Self::BackendUnavailable(_) => "backend_unavailable",
            Self::Unsupported(_) => "unsupported",
        }
    }
}

/// Why an indexed draw's index bytes could not be resolved.
///
/// The type keeps eleven distinct checks visible rather than collapsing an
/// indexed-draw failure into one uninformative backend error.
pub use reims_vgpu_core::IndexLoadReason;

/// Load the render pipeline a draw named, or say why it could not be loaded.
///
/// The sibling of `compute_exec::load_compute_pipeline`, and until now the half
/// of that pair that named none of its five failures: every caller collapses a
/// `None` into one coarse `MissingPipeline`, so a draw that lost its pipeline
/// said only that, on the rail that runs every frame.
///
/// `pipeline_ref == 0` is "no pipeline bound" and stays silent, matching the
/// compute sibling and the rest of the crate — `exec` filters it at both draw
/// call sites, so nothing reaches here with a zero today. The guard is what
/// keeps that true if one ever does: ref 0 is a
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
    state: &Device,
    host: &M,
    task_id: u32,
    pipeline_ref: u32,
) -> Option<RenderPipelineDescriptor> {
    if pipeline_ref == 0 {
        return None;
    }
    let report = crate::observe::RungReport::new("draw_load_pipeline", "pipe_ref");
    // Live object-list: this reference must resolve to a render pipeline.
    let (_entry, desc) = match objects::resolve_descriptor(
        state,
        host,
        task_id,
        pipeline_ref,
        &[ObjectKind::SerializerResource],
    ) {
        Ok(found) => found,
        Err(rung) => {
            report.rung(task_id, pipeline_ref, rung);
            return None;
        }
    };
    let Ok(p) = decode_render_pipeline_descriptor(&desc) else {
        report.reason(
            task_id,
            pipeline_ref,
            crate::observe::ladder_slug!("", desc_decode),
            &format!("desc_len={}", desc.len()),
        );
        return None;
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
    Some(p)
}

/// The two MTLB containers a render pipeline's stages live in, without
/// extracting or copying the AIR out of them.
///
/// Each function payload is retained by its guest object lifetime. The `Arc`s
/// returned here are therefore both the no-copy preflight input and ownership
/// that keeps the immutable containers alive while translation borrows them.
pub(crate) fn load_render_mtlb_pair<M: HostMemory + HostOps>(
    state: &Device,
    host: &M,
    task_id: u32,
    pipeline_ref: u32,
) -> Result<RenderMtlbPair, DrawPreparationDecline> {
    let pd = load_render_pipeline(state, host, task_id, pipeline_ref).ok_or(
        DrawPreparationDecline::PipelineMissing {
            task_id,
            pipeline_ref,
        },
    )?;
    let v_mtlb = load_mtlb(state, host, task_id, pd.vertex_func_ref, AirLoadRail::Draw).ok_or(
        DrawPreparationDecline::VertexMtlbMissing {
            task_id,
            function_ref: pd.vertex_func_ref,
        },
    )?;
    let f_mtlb = load_mtlb(
        state,
        host,
        task_id,
        pd.fragment_func_ref,
        AirLoadRail::Draw,
    )
    .ok_or(DrawPreparationDecline::FragmentMtlbMissing {
        task_id,
        function_ref: pd.fragment_func_ref,
    })?;
    Ok((v_mtlb, f_mtlb))
}

/// Resolve type-1 buffer object → guest bytes starting at `offset`.
/// Where a type-1 buffer object's bytes live in the task GVA space. Both the
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

/// The slug for each way a type-1 buffer ref fails to yield a span.
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

/// Resolve a type-1 buffer `ref` to its backing `(gva, size)` (object-list
/// entry read + descriptor read + decode). Fail-visible per failing site —
/// this is the single owner of the `load_buffer *` reason slugs; the ZC and CPU
/// binds delegate to it so a failure logs exactly once, not once per attempt.
fn resolve_buffer_backing<M: HostMemory>(
    state: &Device,
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
/// `try_buffer_zero_copy_resolved` takes it. This is the capability fallback
/// for the same decoded bind, so it must read the same bounded span; otherwise
/// host import support would change which guest bytes the command consumes.
fn read_buffer_bytes_resolved<M: HostMemory + HostOps>(
    state: &mut Device,
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
    // it held `Device` shared and so *could* not pay; see
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
    state: &mut Device,
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

/// If `texture_ref` is a type-8 object whose descriptor is a buffer-backed
/// texture (view_opcode 9, `newTextureWithDescriptor:offset:bytesPerRow:`, or
/// its `TextureDescriptor2` form), return its decoded descriptor. `None` for a
/// non-type-8 object or a real texture VIEW (opcode 7/8/0x1b) — those stay on
/// the view path silently.
fn buffer_texture_descriptor<M: HostMemory + HostOps>(
    state: &Device,
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
    if resource.entry().kind != ObjectKind::TextureView {
        return None;
    }
    match objects::decoded_resource(resource) {
        Ok(crate::runtime::decode::resource::Descriptor::BufferTexture(descriptor)) => {
            Some(*descriptor)
        }
        _ => None,
    }
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

/// Load an opcode-9 buffer-backed texture as tight RGBA8 (width, height, bytes).
///
/// The sampled bytes are the source MTLBuffer's guest storage read at `offset`
/// with `bytes_per_row` stride and reinterpreted through the embedded texture
/// descriptor's pixel format. Only fires on a genuine buffer-texture object, so
/// every early-return here logs a fail-visible reason (the buffer is unresolved,
/// the format is unknown, or the span overruns the buffer) — those are real
/// dropped-draw causes, not speculative "not ready yet" polls.
fn load_buffer_texture_rgba<M: HostMemory + HostOps>(
    state: &mut Device,
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
    // the type-8 texture object the guest binds and samples, and the type-1
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
    let row_pixels = w as usize;
    let dst_row = row_pixels.checked_mul(RGBA8_BPP as usize)?;
    let mut rgba = vec![0u8; dst_row.checked_mul(h as usize)?];
    let tight = tight as usize;
    let bpr = bpr as usize;
    for y in 0..h as usize {
        let src = &raw[y * bpr..y * bpr + tight];
        let dst = &mut rgba[y * dst_row..(y + 1) * dst_row];
        if !pixel_format::convert_row_to_rgba8(fmt, src, w, dst) {
            crate::observe::fail(format!(
                "buftex convert_fail ref={texture_ref} buf={} fmt={fmt:#x} row={y} {w}x{h}",
                bt.buffer_ref
            ));
            return None;
        }
    }
    Some((w, h, rgba))
}

struct ResolvedIndexWindow {
    backing: BufferBacking,
    byte_offset: u64,
    len: usize,
}

/// Resolve an indexed draw to the guest allocation and exact byte window its
/// index count names. Reading those bytes is a backend choice: Metal's copied
/// upload path consumes them on the CPU, while Vulkan retains this resource
/// window and lets vertex input consume it when the command executes.
fn resolve_index_window_reason<M: HostMemory>(
    state: &Device,
    host: &M,
    task_id: u32,
    info: &IndexedDrawInfo,
) -> Result<ResolvedIndexWindow, IndexLoadReason> {
    use IndexLoadReason as R;
    let elem = info.index_type.map_err(|_| R::TypeUnsupported)?.byte_size();
    let need = (info.index_count as usize)
        .checked_mul(elem)
        .ok_or(R::CountOverflow)?;
    if need == 0 {
        return Err(R::CountZero);
    }
    let (gva, size) = objects::resolve_buffer_span(state, host, task_id, info.index_buffer_ref)
        .map_err(|refusal| match refusal {
            objects::BufferSpanRefusal::Rung(objects::LadderRung::NoListEntry) => R::EntryMissing,
            objects::BufferSpanRefusal::Rung(objects::LadderRung::WrongType { .. }) => {
                R::ObjectType
            }
            objects::BufferSpanRefusal::Rung(objects::LadderRung::DescRead { .. }) => R::DescRead,
            objects::BufferSpanRefusal::Decode => R::DescDecode,
            objects::BufferSpanRefusal::NoBacking => R::BackingMissing,
        })?;
    let byte_offset = info.resolved_byte_offset()?;
    let end = byte_offset
        .checked_add(need as u64)
        .ok_or(R::OffsetOverflow)?;
    if end > size {
        return Err(R::OutOfBounds);
    }
    Ok(ResolvedIndexWindow {
        backing: BufferBacking { gva, size },
        byte_offset,
        len: need,
    })
}

/// Load the index bytes a bound indexed draw references, returning the **specific**
/// reason on failure. Metal emits it directly; Vulkan delegates it through
/// `DrawPreparationDecline::IndexLoad`, so both rails keep one reason vocabulary.
/// Runs on the drain worker (off main core); only reached when `req.indexed` is
/// set, so it cannot flood a 2D-UI boot.
#[cfg(test)]
fn load_index_bytes_reason<M: HostMemory + HostOps>(
    state: &Device,
    host: &M,
    task_id: u32,
    info: &IndexedDrawInfo,
) -> Result<Vec<u8>, IndexLoadReason> {
    use IndexLoadReason as R;
    let window = resolve_index_window_reason(state, host, task_id, info)?;
    let mut buf = vec![0u8; window.len];
    gva_mem::read_task_gva_by_id(
        host,
        &state.tasks,
        task_id,
        window.backing.gva + window.byte_offset,
        &mut buf,
        state.page_shift,
    )
    .map_err(|_| R::ReadFail)?;
    Ok(buf)
}

/// Guest Store seed for IOSurface texture `image_changed` / GVA partial writeback.
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
#[cfg(test)]
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
/// A fourth value is a corrupt or unsupported wire word. Request construction
/// refuses it before either executor can substitute a native load operation.
pub(crate) fn load_action_in_contract(pipeline_ref: u32, load_action: u16) -> bool {
    if is_declared_load_action(load_action) {
        return true;
    }
    if degrade_log_first(pipeline_ref, "load_action_unmapped") {
        crate::observe::fail(format!(
            "pass_state_degraded reason=load_action_unmapped \
             pipe={pipeline_ref} load_action={load_action} \
             (not one of MTLLoadAction 0/1/2; attachment refused)"
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
/// resolve action as a typed refusal; the encode path likewise refuses
/// before encoding until it carries the corresponding attachment lifecycle.
pub(crate) fn store_action_in_contract(pipeline_ref: u32, store_action: u16) -> bool {
    if is_declared_store_action(store_action) {
        return true;
    }
    if degrade_log_first(pipeline_ref, "store_action_unmapped") {
        crate::observe::fail(format!(
            "pass_state_degraded reason=store_action_unmapped \
             pipe={pipeline_ref} store_action={store_action} \
             (not one of the represented MTLStoreAction values 0/1/2/3; \
              attachment refused)"
        ));
    }
    false
}

/// Fail-visible diagnosis when a bound sample ref does not materialize.
///
/// Kept off the success path; only called after a sampled resolver
/// (`resolve_sampled_source` or `load_sampled_rgba`) returns `None`.
fn sample_miss_detail<M: HostMemory + HostOps>(
    state: &mut Device,
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
    let kind = entry.kind;
    let desc_len = entry.descriptor_length;
    match kind {
        ObjectKind::IOSurfacePlaneView => {
            match objects::read_descriptor(state, host, task_id, &entry) {
                None => format!("kind=iosurface_plane_view desc_len={desc_len} reason=no_desc"),
                Some(d)
                    if reims_vgpu_wire::device_desc::iosurface_plane_view_header(&d).is_err() =>
                {
                    format!("kind=iosurface_plane_view desc_len={desc_len} reason=short_desc")
                }
                Some(d) => {
                    let sid = reims_vgpu_wire::device_desc::iosurface_plane_view_header(&d)
                        .map(|h| h.surface_id.get())
                        .unwrap_or(0);
                    match objects::decode_iosurface_plane_view(&d) {
                        Some(view) => format!(
                            "kind=iosurface_plane_view desc_len={desc_len} surface_id={sid} view={}x{} fmt={:#x} reason=ref_texture_view",
                            view.width, view.height, view.pixel_format
                        ),
                        None => format!(
                            "kind=iosurface_plane_view desc_len={desc_len} surface_id={sid} reason=ref_texture_no_view"
                        )}
                }
            }
        }
        ObjectKind::IOSurfaceTexture => {
            let Some(mid) =
                objects::resolve_iosurface_texture_ref(state, host, task_id, texture_ref)
            else {
                return format!(
                    "kind=mapper_iosurface_texture_view desc_len={desc_len} \
                     reason=iosurface_texture_resolve"
                );
            };
            match state.surfaces.mappings.get(&mid) {
                None => format!(
                    "kind=mapper_iosurface_texture_view mid={mid} desc_len={desc_len} \
                     reason=no_mapping"
                ),
                Some(m) => format!(
                    "kind=mapper_iosurface_texture_view mid={mid} desc_len={desc_len} geom={} \
                     {}x{} fmt={:#x} mapped={} pages={} reason=iosurface_texture_sample",
                    m.has_geometry() as u8,
                    m.width_or_zero(),
                    m.height_or_zero(),
                    m.format_or_zero(),
                    m.lifecycle.active as u8,
                    m.pages.entries.len()
                ),
            }
        }
        ObjectKind::TextureView => {
            // Opcode-9 buffer-backed textures share the type-8 tag but are not views.
            if let Some(bt) = buffer_texture_descriptor(state, host, task_id, texture_ref, None) {
                return format!(
                    "kind=buffer_texture desc_len={desc_len} buf={} off={} bpr={} {}x{} fmt={:#x} reason=buftex_load",
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
                        "kind=texture_view desc_len={desc_len} reason=view_resolve view_reason={}",
                        why.slug()
                    )
                }
                Ok(view) => format!(
                    "kind=texture_view desc_len={desc_len} base={} range={:?} fmt_ov={:?} reason=view_base_or_swizzle",
                    view.base_texture_ref,
                    view.range,
                    view.pixel_format
                )}
        }
        ObjectKind::Texture => {
            let Some(desc_bytes) = objects::read_descriptor(state, host, task_id, &entry) else {
                return format!("kind={kind} desc_len={desc_len} reason=desc_read");
            };
            match decode_texture_descriptor(&desc_bytes) {
                Err(_) => format!("kind={kind} desc_len={desc_len} reason=desc_decode"),
                Ok(tex) => {
                    let l0 = tex.level(0);
                    format!(
                        "kind={kind} desc_len={desc_len} has_fmt={} fmt={:#x} mips={} handle={:#x} alloc={} L0={}x{} bpr={} reason=linear_sample",
                        u8::from(tex.declared_pixel_format().is_some()),
                        tex.declared_pixel_format().unwrap_or(0),
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

/// Describe the retained linear texture object which an encoder slot actually
/// binds.
///
/// The numeric reference may be deleted and reused after the bind while the
/// encoder still retains the original resource. A diagnostic that walks the
/// current object list in that case describes a different object and can send
/// the repair in exactly the wrong direction. Non-linear objects return
/// `None` so their existing, type-specific diagnostics remain in charge.
fn retained_linear_sample_miss_detail(resource: &crate::model::TaskResource) -> Option<String> {
    if resource.entry().kind != ObjectKind::Texture {
        return None;
    }
    let kind = resource.entry().kind;
    let desc_len = resource.entry().descriptor_length;
    match crate::runtime::objects::decoded_resource(resource) {
        Err(why) => Some(format!(
            "kind={kind} desc_len={desc_len} retained=1 reason={}",
            why.slug()
        )),
        Ok(crate::runtime::decode::resource::Descriptor::Texture(tex)) => {
            let l0 = tex.level(0);
            Some(format!(
                "kind={kind} desc_len={desc_len} retained=1 has_fmt={} fmt={:#x} \
                 mips={} handle={:#x} alloc={} base_off={} used={} L0={}x{} \
                 L0_off={} bpr={} reason=linear_sample",
                u8::from(tex.declared_pixel_format().is_some()),
                tex.declared_pixel_format().unwrap_or(0),
                tex.mipmap_level_count,
                tex.handle,
                tex.allocation_size,
                tex.base_offset,
                tex.used_size,
                l0.map(|level| level.width).unwrap_or(0),
                l0.map(|level| level.height).unwrap_or(0),
                l0.map(|level| level.offset).unwrap_or(0),
                l0.map(|level| level.row_stride).unwrap_or(0),
            ))
        }
        Ok(_) => Some(format!(
            "kind={kind} desc_len={desc_len} retained=1 reason=decoded_kind_mismatch"
        )),
    }
}

/// What the guest says an IOSurface texture mapping's texel **values** are, seen through an
/// optional type-8 view format.
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
fn mapping_declared_format(state: &Device, mapping_id: u32, format_override: Option<u16>) -> u16 {
    use crate::runtime::mapping_write::mapping_store_format;
    if let Some(view) = format_override {
        return view;
    }
    match state.surfaces.mappings.get(&mapping_id) {
        Some(entry) => mapping_store_format(entry),
        // Nothing declared. An entry that has latched no geometry is exactly
        // that case, so the owning rule answers it rather than a default being
        // named a second time here.
        None => mapping_store_format(&crate::model::SurfaceMappingEntry::default()),
    }
}

/// Sample an IOSurface texture mapping as tight RGBA8 from guest pages.
///
/// Guest pages ARE the surface content: the CPU writeback lands Stores in them
/// and guest CPU writes are immediately visible. There is exactly one source;
/// no recovery ranking exists.
///
/// The resolve runs *before* the geometry read, not after. A mapping can be
/// mapped with a live `MappingInternal` and no latched W×H yet; resolving first
/// decodes the guest device-surface descriptor and latches the geometry, so the
/// sample succeeds instead of bailing out on `!has_geom` and dropping the bind.
fn load_iosurface_mapping_rgba<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    mapping_id: u32,
    format_override: Option<u16>,
) -> Option<(u32, u32, Vec<u8>)> {
    let _ = mapper::ensure_resolved_for_scanout(state, host, mapping_id);
    let (w, h) = {
        let m = state.surfaces.mappings.get(&mapping_id)?;
        if !m.has_geometry() || m.width_or_zero() == 0 || m.height_or_zero() == 0 {
            return None;
        }
        (m.width_or_zero(), m.height_or_zero())
    };
    let base_fmt = MTL_FORMAT_BGRA8_UNORM;
    let sample_fmt = effective_view_sample_format(base_fmt, format_override)?;
    let stride = w.saturating_mul(RGBA8_BPP);
    let mut raw = vec![0u8; (stride as usize).saturating_mul(h as usize)];
    if !crate::runtime::scanout::read_mapping_bgra8(state, host, mapping_id, &mut raw, stride, w, h)
    {
        return None;
    }
    let mut rgba = vec![0u8; raw.len()];
    for y in 0..h as usize {
        let off = y * (stride as usize);
        let row = &raw[off..off + (w as usize) * 4];
        let dst = &mut rgba[off..off + (w as usize) * 4];
        if !pixel_format::convert_row_to_rgba8(sample_fmt, row, w, dst) {
            return None;
        }
    }
    Some((w, h, rgba))
}

fn vulkan_sampler_resource(
    sampler_ref: u32,
    binding: u32,
    sampler: &crate::runtime::decode::resource::SamplerDescriptor,
) -> Result<reims_vgpu_core::SamplerResource, DrawPreparationDecline> {
    use reims_vgpu_core::SamplerResource;

    Ok(SamplerResource {
        binding,
        source: reims_vgpu_core::SamplerSource::State,
        min_filter: reims_vgpu_protocol::sampler_filter(sampler.min_filter).map_err(|reason| {
            DrawPreparationDecline::SamplerMinFilterTranslation {
                sampler_ref,
                binding,
                reason,
            }
        })?,
        mag_filter: reims_vgpu_protocol::sampler_filter(sampler.mag_filter).map_err(|reason| {
            DrawPreparationDecline::SamplerMagFilterTranslation {
                sampler_ref,
                binding,
                reason,
            }
        })?,
        mip_filter: reims_vgpu_protocol::sampler_mip_filter(sampler.mip_filter).map_err(
            |reason| DrawPreparationDecline::SamplerMipFilterTranslation {
                sampler_ref,
                binding,
                reason,
            },
        )?,
        address_mode_u: reims_vgpu_protocol::sampler_address_mode(sampler.s_address).map_err(
            |reason| DrawPreparationDecline::SamplerAddressSTranslation {
                sampler_ref,
                binding,
                reason,
            },
        )?,
        address_mode_v: reims_vgpu_protocol::sampler_address_mode(sampler.t_address).map_err(
            |reason| DrawPreparationDecline::SamplerAddressTTranslation {
                sampler_ref,
                binding,
                reason,
            },
        )?,
        address_mode_w: reims_vgpu_protocol::sampler_address_mode(sampler.r_address).map_err(
            |reason| DrawPreparationDecline::SamplerAddressRTranslation {
                sampler_ref,
                binding,
                reason,
            },
        )?,
        border_color: reims_vgpu_protocol::sampler_border_color(sampler.border_color).map_err(
            |reason| DrawPreparationDecline::SamplerBorderColorTranslation {
                sampler_ref,
                binding,
                reason,
            },
        )?,
        // Metal reuses `MTLCompareFunction` for depth, stencil and sampler
        // compare, so this is `raster`'s table rather than `sampler`'s — one
        // Metal enum, one home.
        compare_function: reims_vgpu_protocol::compare_function(sampler.compare_function).map_err(
            |reason| DrawPreparationDecline::SamplerCompareFunctionTranslation {
                sampler_ref,
                binding,
                reason,
            },
        )?,
        lod_min: sampler.lod_min_clamp.to_bits(),
        lod_max: sampler.lod_max_clamp.to_bits(),
        max_anisotropy: sampler.max_anisotropy,
        unnormalized_coordinates: !sampler.normalized_coordinates,
    })
}

pub fn reflected_static_sampler_resource(
    stage: &'static str,
    binding: u32,
    sampler: reims_vgpu_core::ReflectedStaticSamplerState,
) -> Result<reims_vgpu_core::SamplerResource, DrawPreparationDecline> {
    use reims_vgpu_core::{
        ReflectedSamplerAddressMode as ReflectedAddress,
        ReflectedSamplerBorderColor as ReflectedBorder,
        ReflectedSamplerCompareFunction as ReflectedCompare,
        ReflectedSamplerCoordinates as SamplerCoordinates,
        ReflectedSamplerFilter as ReflectedFilter, ReflectedSamplerMipFilter as ReflectedMip,
        ReflectedSamplerReduction as SamplerReduction, SamplerAddressMode, SamplerBorderColor,
        SamplerCompareFunction, SamplerFilter, SamplerMipFilter, SamplerResource,
    };

    if sampler.reduction != SamplerReduction::WeightedAverage {
        return Err(DrawPreparationDecline::StaticSamplerReductionUnsupported {
            stage,
            binding,
            reduction: format!("{:?}", sampler.reduction),
            raw_words: sampler.raw_words,
        });
    }
    if sampler.lod_bias != 0.0 {
        return Err(DrawPreparationDecline::StaticSamplerLodBiasUnsupported {
            stage,
            binding,
            lod_bias_bits: sampler.lod_bias.to_bits(),
            raw_words: sampler.raw_words,
        });
    }
    let filter = |filter, min| match filter {
        ReflectedFilter::Nearest => Ok(SamplerFilter::Nearest),
        ReflectedFilter::Linear => Ok(SamplerFilter::Linear),
        ReflectedFilter::Bicubic if min => {
            Err(DrawPreparationDecline::StaticSamplerMinFilterUnsupported { stage, binding })
        }
        ReflectedFilter::Bicubic => {
            Err(DrawPreparationDecline::StaticSamplerMagFilterUnsupported { stage, binding })
        }
    };
    let mip_filter = match sampler.mip_filter {
        ReflectedMip::None => SamplerMipFilter::NotMipmapped,
        ReflectedMip::Nearest => SamplerMipFilter::Nearest,
        ReflectedMip::Linear => SamplerMipFilter::Linear,
    };
    let address = |address| match address {
        ReflectedAddress::ClampToZero => SamplerAddressMode::ClampToZero,
        ReflectedAddress::ClampToEdge => SamplerAddressMode::ClampToEdge,
        ReflectedAddress::Repeat => SamplerAddressMode::Repeat,
        ReflectedAddress::MirroredRepeat => SamplerAddressMode::MirrorRepeat,
        ReflectedAddress::ClampToBorder => SamplerAddressMode::ClampToBorderColor,
    };
    let border_color = match sampler.border_color {
        ReflectedBorder::TransparentBlack => SamplerBorderColor::TransparentBlack,
        ReflectedBorder::OpaqueBlack => SamplerBorderColor::OpaqueBlack,
        ReflectedBorder::OpaqueWhite => SamplerBorderColor::OpaqueWhite,
    };
    let compare_function = match sampler.compare_function {
        ReflectedCompare::None | ReflectedCompare::Never => SamplerCompareFunction::Never,
        ReflectedCompare::Less => SamplerCompareFunction::Less,
        ReflectedCompare::LessEqual => SamplerCompareFunction::LessEqual,
        ReflectedCompare::Greater => SamplerCompareFunction::Greater,
        ReflectedCompare::GreaterEqual => SamplerCompareFunction::GreaterEqual,
        ReflectedCompare::Equal => SamplerCompareFunction::Equal,
        ReflectedCompare::NotEqual => SamplerCompareFunction::NotEqual,
        ReflectedCompare::Always => SamplerCompareFunction::Always,
    };

    Ok(SamplerResource {
        binding,
        source: reims_vgpu_core::SamplerSource::State,
        min_filter: filter(sampler.min_filter, true)?,
        mag_filter: filter(sampler.mag_filter, false)?,
        mip_filter,
        address_mode_u: address(sampler.address_mode_s),
        address_mode_v: address(sampler.address_mode_t),
        address_mode_w: address(sampler.address_mode_r),
        border_color,
        compare_function,
        lod_min: sampler.lod_min_clamp.to_bits(),
        lod_max: sampler.lod_max_clamp.to_bits(),
        max_anisotropy: sampler.max_anisotropy,
        unnormalized_coordinates: sampler.coordinates == SamplerCoordinates::Pixel,
    })
}

pub(crate) fn load_vulkan_sampler<M: HostMemory + HostOps>(
    state: &Device,
    host: &M,
    task_id: u32,
    sampler_ref: u32,
    binding: u32,
) -> Result<reims_vgpu_core::SamplerResource, DrawPreparationDecline> {
    let sampler =
        objects::resolve_sampler_state(state, host, task_id, sampler_ref).map_err(|failure| {
            match failure {
                objects::SamplerResolveError::Rung(rung) => match rung {
                    objects::LadderRung::NoListEntry => {
                        DrawPreparationDecline::SamplerEntryMissing {
                            sampler_ref,
                            binding,
                        }
                    }
                    objects::LadderRung::WrongType { got } => {
                        DrawPreparationDecline::SamplerObjectType {
                            sampler_ref,
                            binding,
                            object_type: got,
                        }
                    }
                    objects::LadderRung::DescRead { .. } => {
                        DrawPreparationDecline::SamplerDescriptorMissing {
                            sampler_ref,
                            binding,
                        }
                    }
                },
                objects::SamplerResolveError::Decode {
                    status,
                    descriptor_len,
                    tag,
                    declared_len,
                } => match status {
                    DecodeStatus::ErrShort(_) => DrawPreparationDecline::SamplerDescriptorShort {
                        sampler_ref,
                        binding,
                        descriptor_len,
                    },
                    DecodeStatus::ErrUnknownType(_) => {
                        DrawPreparationDecline::SamplerDescriptorUnknownType {
                            sampler_ref,
                            binding,
                            descriptor_len,
                            tag,
                        }
                    }
                    DecodeStatus::ErrUnsupported(_) => {
                        DrawPreparationDecline::SamplerDescriptorUnsupported {
                            sampler_ref,
                            binding,
                            descriptor_len,
                            tag,
                            declared_len,
                        }
                    }
                },
            }
        })?;
    vulkan_sampler_resource(sampler_ref, binding, &sampler)
}

/// Store encode RGBA8 into **texture_ref** host cache as BGRA (not surface_id).
#[cfg(test)]
fn host_cache_store_rgba8(
    state: &mut Device,
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

/// Advance the guest-visible publish milestones for an IOSurface texture Store whose
/// pixels have landed in the mapping's guest pages.
///
/// Route-independent: the synchronous `cpu_portability` Store calls it inline,
/// and the resident render Store calls it from the writeback that performs the
/// same write. Both have just proved
/// the same thing — `write_rgba8_image_changed` verified geometry and landed a
/// complete frame — and without it the `present_unbacked` gate is structurally
/// dead on whichever route skips it, because no mapping's full-frame evidence
/// would advance.
pub(crate) fn publish_surface_store<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    mapping_id: u32,
    width: u32,
    height: u32,
    format: u16,
) {
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
    /// The `BackendUnavailable` carrier — this build has no host encode path for
    /// the record, such as an indirect-command-buffer execute.
    BackendUnavailable,
    /// A typed terminal refusal from encode, already named by
    /// `note_draw_encode_fail`.
    TerminalRefusal,
}

impl ChainAbandonCause {
    /// The `cause=` token. Stable text: it is grepped out of boot logs.
    pub fn tag(self) -> &'static str {
        match self {
            Self::NoColor0 => "no_color0",
            Self::BackendUnavailable => "backend_unavailable",
            Self::TerminalRefusal => "terminal_refusal",
        }
    }
}

pub fn writeback_chain_rgba<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    color_slots: &[(u32, crate::runtime::decode::render::ColorAttachment)],
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
        storage,
        width: w,
        height: h,
        format: fmt,
        sample_count: _,
    }) = lookup_render_target(state, host, task_id, *att)
    else {
        return lost("render_target_unresolved");
    };
    let mapping_id = storage.mapping_id();
    let gva = storage.target_gva();
    let bpr = storage.row_stride();
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
    // the resident readback into the IOSurface texture mapping, publish the Composite
    // Store, and keep the degradation fail-visible.
    crate::observe::fail(format!(
        "writeback_chain_rgba reason=resident_chain_abandoned_cpu_recovery \
         cause={} mid={mapping_id} {w}x{h} fmt={fmt:#x}",
        cause.tag()
    ));
    let wrote = mapping_write::write_rgba8_image_changed(state, host, mapping_id, rgba, None, w, h);
    if wrote {
        publish_surface_store(state, host, mapping_id, w, h, fmt);
    }
    wrote
}

/// Resolve color texture ref → mapping geometry for a draw request.
#[allow(
    clippy::too_many_arguments,
    reason = "the request builder mirrors the decoded color attachment state"
)]
pub fn color_target_request<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &M,
    task_id: u32,
    color: crate::runtime::decode::render::ColorAttachment,
    pipeline_ref: u32,
    vertex_count: u32,
    instance_count: u32,
    primitive_topology: reims_vgpu_protocol::PrimitiveTopology,
    first_vertex: u32,
    base_instance: u32,
) -> Option<DrawEncodeRequest> {
    let color_texture_ref = color.texture_ref;
    let rt = lookup_render_target(state, host, task_id, color)?;
    let attachment_sample_count = crate::runtime::pipeline_resolve::attachment_sample_count(
        state,
        host,
        task_id,
        pipeline_ref,
    )
    .unwrap_or(rt.sample_count);
    let c0 = ColorRtRequest {
        slot: 0,
        texture_ref: color_texture_ref,
        resource: objects::resolve_resource(state, host, task_id, color_texture_ref).ok(),
        storage: rt.storage,
        width: rt.width,
        height: rt.height,
        format: rt.format,
        sample_count: attachment_sample_count,
        load_action: reims_vgpu_protocol::pass_action::LoadAction::DontCare,
        store_action: reims_vgpu_protocol::pass_action::StoreAction::Store,
        clear_color: [0.0; 4],
        target_seed_rgba: None,
        multisample_source_ref: 0,
    };
    Some(DrawEncodeRequest {
        task_id,
        pipeline_ref,
        vertex_count,
        instance_count,
        primitive_topology,
        first_vertex,
        base_instance,
        colors: vec![c0],
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
    state: &mut Device,
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
    state: &Device,
    host: &M,
    task_id: u32,
    c: &ColorRtRequest,
) -> Option<StoreTargetPages> {
    if c.target_gva() == 0 || !c.publishes_single_sample() || c.width == 0 || c.height == 0 {
        return None;
    }
    let span = (c.row_stride() as u64).checked_mul(c.height as u64)?;
    let ordered = crate::runtime::gva_mem::task_gva_page_gpas(
        host,
        &state.tasks,
        task_id,
        c.target_gva(),
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
/// the ordered form of the walk has no reader. Held rather than `cfg`-ed out of
/// the struct: both fields are produced by the one walk either way, and a
/// conditional shape would make the two arms disagree about what a Store's
/// authorisation *is*.
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
    state: &mut Device,
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
    write_gva_rows_within(
        state,
        host,
        task_id,
        gva,
        width,
        height,
        bpr,
        format,
        rgba,
        SourceRows::Distinct,
        allowed,
    )
}

/// Land a solid colour into a GVA render target.
///
/// One tight RGBA8 row is built and handed to the same writer every full-image
/// landing uses, with [`SourceRows::Repeated`] — so the format conversion runs
/// once for the whole surface instead of once per row, and the caller never
/// materialises an image at all.
///
/// This is the CLEAR seed's whole path. Everything it lands is `w * h` copies of
/// one word; the previous route built that word into a full-surface buffer,
/// re-converted each of its identical rows into the destination format, and
/// copied them one at a time. `clear_seed_gva_us` measured **118 ms a second for
/// 175 MB** on the load probe's `blur=40` dial, which is 0.7 GB/s for a copy.
#[allow(
    clippy::too_many_arguments,
    reason = "the same target GVA and native row geometry every GVA writer takes"
)]
// The CLEAR seed at the head of a draw chain is the only caller.
pub(crate) fn write_gva_solid8<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    gva: u64,
    width: u32,
    height: u32,
    bpr: u32,
    format: u16,
    clear: &[f64; 4],
) -> Result<(), crate::runtime::host::MemError> {
    let row = pixel_format::solid_rgba8(width, 1, clear);
    write_gva_rows_within(
        state,
        host,
        task_id,
        gva,
        width,
        height,
        bpr,
        format,
        &row,
        SourceRows::Repeated,
        None,
    )
}

/// Whether the source buffer holds one row per destination row, or a single row
/// every destination row is a copy of.
///
/// The distinction is worth a type because it decides how many *format
/// conversions* the write performs, and that is the whole cost of a solid
/// landing: a CLEAR seed converts one 7 KiB row and then converts it again for
/// every one of a thousand identical rows. Measured on the load probe's
/// `blur=40` dial, `clear_seed_gva_us` was **118 ms a second for 175 MB** —
/// 0.7 GB/s, where the copy alone would be an order of magnitude faster.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SourceRows {
    /// `height` rows of `width` RGBA8 texels, one per destination row.
    Distinct,
    /// One row of `width` RGBA8 texels, written to every destination row.
    ///
    /// Constructed only by [`write_gva_solid8`].
    Repeated,
}

/// [`write_gva_rgba8_within`] and [`write_gva_solid8_within`] share this body;
/// [`SourceRows`] is the only difference between them.
#[allow(
    clippy::too_many_arguments,
    reason = "the archive writer mirrors the target GVA and native row geometry"
)]
fn write_gva_rows_within<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    gva: u64,
    width: u32,
    height: u32,
    bpr: u32,
    format: u16,
    rgba: &[u8],
    rows: SourceRows,
    allowed: crate::runtime::gva_view::WindowPages<'_>,
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
    let rgba_row = (width as usize).saturating_mul(RGBA8_BPP as usize);
    // A repeated source is one row however tall the destination is, which is
    // also what makes its conversion a one-off below.
    let src_stride = match rows {
        SourceRows::Distinct => rgba_row,
        SourceRows::Repeated => 0,
    };
    let need = match rows {
        SourceRows::Distinct => rgba_row.saturating_mul(height as usize),
        SourceRows::Repeated => rgba_row,
    };
    if rgba.len() < need {
        return Err(MemError::BadArgs);
    }
    let span = (height as u64).saturating_mul(bpr as u64);
    let mut row = vec![0u8; tight as usize];
    // A repeated source converts once and every destination row is that same
    // conversion; a distinct source converts per row. Tracked rather than
    // branched on inside the loop so the two forms cannot answer differently
    // about *which* row is in `row` at any point.
    let mut converted = false;
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
            if !converted || src_stride != 0 {
                let at = y * src_stride;
                let src = &rgba[at..at + rgba_row];
                if !pixel_format::convert_rgba8_to_row(format, src, width, &mut row) {
                    res = Err(MemError::BadArgs);
                    break;
                }
                converted = true;
            }
            let off = y.saturating_mul(bpr as usize);
            if off + row.len() > avail {
                res = Err(MemError::RunOutOfRange);
                break;
            }
            // SAFETY: map_fresh_span covers `span`.
            unsafe {
                std::ptr::copy_nonoverlapping(row.as_ptr(), base.add(off), row.len());
            }
        }
        crate::runtime::gva_view::unmap_fresh_span(host, span_map);
        return res;
    }
    // Fragmented GVA: multi-import each converted row via `write_span_within`.
    for y in 0..height as usize {
        if !converted || src_stride != 0 {
            let at = y * src_stride;
            let src = &rgba[at..at + rgba_row];
            if !pixel_format::convert_rgba8_to_row(format, src, width, &mut row) {
                return Err(MemError::BadArgs);
            }
            converted = true;
        }
        let row_gva = gva.saturating_add((y as u64).saturating_mul(bpr as u64));
        if let Err(err) = crate::runtime::gva_view::write_span_within(
            state, host, task_id, row_gva, &row, allowed,
        ) {
            let reason = crate::observe::Decline::slug(&err);
            crate::observe::fail(format!(
                "gva_write fail reason={reason} task={task_id} gva={row_gva:#x} span={span:#x} \
                 row={y} rowlen={:#x} (rgba8 multi)",
                row.len()
            ));
            return Err(err);
        }
    }
    Ok(())
}

/// Seed color RT LOAD from guest IOSurface texture (BGRA→RGBA) or type-2/3/view linear RGBA.
///
/// Every color RT is an ephemeral host RT now, so every `Load` needs this: the
/// IOSurface texture guest-memory alias that let Metal Load read the surface bytes in
/// place is deleted. This used to run only on the alias-reject fallback
/// (unaligned offset or row stride, span out of range, no device), which is why
/// it is already a complete path and not a new one.
pub(super) fn seed_color_load<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    target_gva: u64,
    width: u32,
    height: u32,
) -> Option<Vec<u8>> {
    // Discrete GPU: exact target GVA is the strongest identity across object-ref
    // recycling. Fall back to the type-2/3 texture namespace, never the
    // unrelated surface backing surface_id namespace. Guest memory is last.
    if width > 0 && height > 0 {
        if target_gva != 0 {
            // Recency for the encode cache's byte cap; a Load seed served from
            // here is a use, and this is the read path that keeps a
            // stored-once-sampled-forever entry warm.
            crate::runtime::surface_cache::touch_gva(state, target_gva, width, height);
        }
        // This is the reader that keeps `HostReplicaState::gva_surfaces` alive,
        // and the measurement is unambiguous. One driven x86/Vulkan boot (four
        // Safari pages, each scrolled six times then title-bar dragged;
        // `.agents/repros/gva-seed-serve-census.sh`) served **1 558 colour LOAD
        // seeds from this lookup and missed 0**. `load_seed_ok_color` was 1 558
        // in the same window, so every colour LOAD seed the device produced came
        // from here; the other 1 462 of `load_seed_ok` are IOSurface texture and take
        // `resolve_iosurface_texture_load_seed`.
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
        let gva_served = gva_present && {
            let verdict =
                crate::runtime::surface_cache::gva_seed_verdict(state, host, task_id, target_gva);
            crate::runtime::drain::note_store_route(verdict.route());
            !matches!(
                verdict,
                crate::runtime::surface_cache::GvaSeedVerdict::OtherTask
                    | crate::runtime::surface_cache::GvaSeedVerdict::Moved
            )
        };
        let ref_served = !gva_served
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
    }
    // Type-2/3 (or type-8 base) linear GVA → convert to RGBA8.
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
    // `scanout::paint_mapping` behind `load_iosurface_mapping_rgba`, and
    // `draw::texture_view::load_linear_texture_impl` for the linear arm. The
    // buffer leaf had no settle at all before that, on any of its four callers.
    // The seed arm: this leaf is shared with the sampled resolve and the two
    // want opposite repairs, so it is charged separately.
    // A colour LOAD seed is copied into a render target through the RGBA8-shaped
    // seed path, so this arm takes no native layout — the bytes must be what
    // that path reads them as.
    // A colour LOAD seed fills one 2D render target, so it names one
    // subresource. A layered source seeds from its first slice here and always
    // has; the target it seeds has no second layer to receive the rest.
    let (rgba, _layout) = load_sampled_rgba_static(
        state,
        host,
        task_id,
        texture_ref,
        0,
        NativeUploads::NONE,
        crate::runtime::render_writeback::SettleSite::LinearTextureSeed,
    )?;
    Some(rgba)
}

/// Resolve sampled texture RGBA for the color LOAD seed path.
///
/// Type-8 views with a non-identity swizzle are rejected here: RT materialization does not
/// rematerialize through a remapped view (contract: swizzled views fail for RT/blit).
/// View `pixel_format` still overrides the base format when bpp-compatible.
fn load_sampled_rgba_static<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    slice: u32,
    native: NativeUploads,
    site: crate::runtime::render_writeback::SettleSite,
) -> Option<(Vec<u8>, SampledByteFormat)> {
    // The three rungs below are single-subresource construction forms: a
    // buffer-backed texture views one buffer range and an IOSurface texture one
    // plane, so neither carries a second array layer or cube face to advance
    // to. Answering a later slice with slice 0's bytes would bind five copies
    // of one face and look like a working cube, so the request is refused
    // instead — only the linear rung below can honour it.
    if slice != 0
        && (buffer_texture_descriptor(state, host, task_id, texture_ref, None).is_some()
            || objects::resolve_iosurface_texture_ref(state, host, task_id, texture_ref).is_some())
    {
        return None;
    }
    // Opcode-9 buffer-backed texture (type-8): sample the source buffer directly.
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
    // IOSurface texture path via resolve.
    if let Some(mid) = objects::resolve_iosurface_texture_ref(state, host, task_id, texture_ref) {
        let source = mapping_declared_format(state, mid, None);
        return load_iosurface_mapping_rgba(state, host, mid, None).map(|(_, _, r)| {
            (
                r,
                SampledByteFormat::from_source(TexelLayout::Rgba8, source),
            )
        });
    }
    // Type-8 view → base texture + mip + format. The view's SWIZZLE is
    // deliberately not consulted here: it is a property of the view, not of the
    // bytes, and the bind applies it as the image view's component mapping so
    // the GPU performs it at sample time. Refusing here (which this path used
    // to do, silently) dropped the texture from the draw entirely.
    let (tex_ref, level, fmt_override) =
        if let Some(view) = resolve_texture_view(state, host, task_id, texture_ref) {
            (
                view.base_texture_ref,
                view.single_non_array_level()?,
                view.pixel_format,
            )
        } else {
            (texture_ref, 0, None)
        };
    // IOSurface texture base through a view (format override may reinterpret BGRA storage).
    if let Some(mid) = objects::resolve_iosurface_texture_ref(state, host, task_id, tex_ref) {
        if level != 0 {
            return None;
        }
        let source = mapping_declared_format(state, mid, fmt_override);
        return load_iosurface_mapping_rgba(state, host, mid, fmt_override).map(|(_, _, r)| {
            (
                r,
                SampledByteFormat::from_source(TexelLayout::Rgba8, source),
            )
        });
    }
    // The only rung here that can answer in anything but RGBA8. The three above
    // convert unconditionally — `load_buffer_texture_rgba` and
    // `load_iosurface_mapping_rgba` have no native arm — so they state the layout
    // they always produced rather than being handed a choice they cannot make.
    // All four still name the guest format their values were read from, because
    // a convert to RGBA8 reorders channels and does not decode.
    load_linear_texture_host(
        state,
        host,
        task_id,
        tex_ref,
        slice,
        level,
        fmt_override,
        native,
        site,
    )
}

/// Size a recycled scratch buffer to `span` for a `filled`-byte rect, without
/// re-zeroing the rect.
///
/// The caller's read overwrites all `filled` bytes and fails the whole bind if it
/// cannot, so zeroing them first buys nothing. It is not free either: at the
/// 1920x1080 that dominates this workload the rect is 8.3 MiB, and the memset was
/// paid on the memo *hit* path — the path whose entire purpose is to avoid
/// touching the surface.
///
/// The `host_alloc_len` padding past the rect is a different case: the read does
/// not write it, so it is zeroed here. A recycled buffer must not carry a
/// previous surface's tail into the memo comparison, where it would manufacture a
/// miss and cost a full conversion.
fn prepare_memo_scratch(scratch: &mut Vec<u8>, span: usize, filled: usize) {
    let filled = filled.min(span);
    scratch.resize(span, 0);
    scratch[filled..].fill(0);
}

/// Byte-exact revalidated memo for the IOSurface texture mapping-backed guest-page sampled
/// path. Same contract as [`load_linear_guest_memoized`] / the IOSurface plane view view memo:
/// re-read the native BGRA rect every bind (a guest CPU write is always
/// observed — neither `map_generation` nor `content_generation` tracks in-place
/// guest writes), memcmp against the memo, and on an unchanged hit return the
/// cached RGBA `Arc` + a namespaced content identity so BOTH the CPU convert/
/// alloc AND the engine's content hash + GPU upload are skipped. A dock-
/// magnification burst re-binds the same static icons ~1000×, so this collapses
/// the `iosurface_guest` CPU copies that saturate the serial drain worker (the
/// dock-hover whole-VM freeze). Returns `(rgba, identity)`.
fn load_iosurface_texture_rgba_memoized<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    mid: u32,
) -> Option<(std::sync::Arc<Vec<u8>>, LinearSampleIdentity)> {
    let (w, h) = {
        let m = state.surfaces.mappings.get(&mid)?;
        if !m.has_geometry() || m.width_or_zero() == 0 || m.height_or_zero() == 0 {
            return None;
        }
        (m.width_or_zero(), m.height_or_zero())
    };
    let sample_fmt = effective_view_sample_format(MTL_FORMAT_BGRA8_UNORM, None)?;
    let stride = w.saturating_mul(RGBA8_BPP);
    let span = host_alloc_len((stride as u64).checked_mul(h as u64)?)?;
    if span == 0 {
        return None;
    }
    // Coherence re-read: land any resident-authoritative writeback and read the
    // current native BGRA (read_mapping_bgra8 runs ensure_resolved_for_scanout +
    // flush internally). Reuse the scratch so a memo hit costs no allocation.
    let mut scratch = std::mem::take(&mut state.content.sampled.iosurface_texture_memo_scratch);
    prepare_memo_scratch(
        &mut scratch,
        span,
        (stride as usize).saturating_mul(h as usize),
    );
    if !{
        crate::runtime::scanout::read_mapping_bgra8(state, host, mid, &mut scratch, stride, w, h)
    } {
        state.content.sampled.iosurface_texture_memo_scratch = scratch;
        return None;
    }
    // Identity key namespace: bits 63+62 mark IOSurface texture memo content, distinct from
    // raw-GVA keys (bit 63 clear) and IOSurface plane view view keys (bit 63 set, bit 62 clear).
    // Every producer draws its generation from
    // `Device::next_sampled_content_generation`, so a (key, generation)
    // pair is unique device-wide and content can never alias on a collision.
    let identity_key = (1u64 << 63) | (1u64 << 62) | mid as u64;
    let key = (mid, w, h);
    if let Some(m) = state.content.sampled.iosurface_texture_memo.get_touch(&key) {
        if m.native == scratch {
            let rgba = m.rgba.clone();
            let generation = m.generation;
            state.content.sampled.iosurface_texture_memo_scratch = scratch;
            return Some((
                rgba,
                LinearSampleIdentity {
                    key: identity_key,
                    generation,
                },
            ));
        }
    }
    // First sight or the native bytes changed: convert BGRA→RGBA fresh.
    let mut rgba = vec![0u8; span];
    let converted = (0..h as usize).all(|y| {
        let off = y * (stride as usize);
        pixel_format::convert_row_to_rgba8(
            sample_fmt,
            &scratch[off..off + (w as usize) * 4],
            w,
            &mut rgba[off..off + (w as usize) * 4],
        )
    });
    if !converted {
        state.content.sampled.iosurface_texture_memo_scratch = scratch;
        return None;
    }
    let rgba = std::sync::Arc::new(rgba);
    let generation = state.next_sampled_content_generation();
    let entry_bytes = scratch.len() + rgba.len();
    state.content.sampled.iosurface_texture_memo.insert(
        key,
        crate::model::GuestLinearMemo {
            native: scratch,
            rgba: rgba.clone(),
            // This rail converts every format to RGBA8 unconditionally — the
            // loop above is `convert_row_to_rgba8` with no native arm — so the
            // layout is fixed rather than chosen.
            layout: reims_vgpu_core::pixel_format::TexelLayout::Rgba8,
            generation,
        },
        entry_bytes,
    );
    Some((
        rgba,
        LinearSampleIdentity {
            key: identity_key,
            generation,
        },
    ))
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod load_action_contract_tests {
    use super::load_action_in_contract;
    use reims_vgpu_protocol::pass_action::{
        MTL_LOAD_ACTION_CLEAR, MTL_LOAD_ACTION_DONT_CARE, MTL_LOAD_ACTION_LOAD,
    };

    /// `MTLLoadAction` has three values, and a fourth is named rather than
    /// swallowed.
    ///
    /// Request normalization refuses an out-of-contract value before either
    /// executor sees the pass. One helper names that boundary, so a third arm
    /// cannot reintroduce a fallback.
    #[test]
    fn a_load_action_outside_mtlloadaction_is_named_not_swallowed() {
        for (name, action) in [
            ("DontCare", MTL_LOAD_ACTION_DONT_CARE),
            ("Load", MTL_LOAD_ACTION_LOAD),
            ("Clear", MTL_LOAD_ACTION_CLEAR),
        ] {
            assert!(
                load_action_in_contract(0x10AD, action),
                "MTLLoadAction{name} is in contract"
            );
        }
        // Distinct pipeline refs, because the emitter latches per
        // (pipeline, slug) and a second call on one ref would report nothing.
        assert!(!load_action_in_contract(0xF001, MTL_LOAD_ACTION_CLEAR + 1));
        assert!(!load_action_in_contract(0xF002, u16::MAX));

        let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
        let line = log
            .lines()
            .rev()
            .find(|l| l.contains("reason=load_action_unmapped") && l.contains("pipe=61442"))
            .expect("an out-of-contract load action must name itself");
        assert!(
            line.starts_with("pass_state_degraded ") && line.contains("load_action=65535"),
            "the line must carry the value that was refused: {line}"
        );
    }
}

#[cfg(test)]
mod store_action_contract_tests {
    use super::store_action_in_contract;
    use reims_vgpu_protocol::pass_action::{
        MTL_STORE_ACTION_DONT_CARE, MTL_STORE_ACTION_MULTISAMPLE_RESOLVE, MTL_STORE_ACTION_STORE,
        MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE,
    };

    /// The sibling of `a_load_action_outside_mtlloadaction_is_named_not_swallowed`,
    /// and it did not exist while that one did.
    ///
    /// The two fields are adjacent words of one attachment prefix, so a decode
    /// that misreads the load action misreads the store action too — and only
    /// the load half said anything. An out-of-contract store action drops the
    /// frame the guest drew, which is the loss the ground rules say must be
    /// visible.
    #[test]
    fn a_store_action_outside_mtlstoreaction_is_named_not_swallowed() {
        for (name, action) in [
            ("DontCare", MTL_STORE_ACTION_DONT_CARE),
            ("Store", MTL_STORE_ACTION_STORE),
            ("MultisampleResolve", MTL_STORE_ACTION_MULTISAMPLE_RESOLVE),
            (
                "StoreAndMultisampleResolve",
                MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE,
            ),
        ] {
            assert!(
                store_action_in_contract(0x570E, action),
                "MTLStoreAction{name} is in contract"
            );
        }
        // Distinct pipeline refs, because the emitter latches per
        // (pipeline, slug) and a second call on one ref would report nothing.
        assert!(!store_action_in_contract(
            0xF101,
            MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE + 1
        ));
        assert!(!store_action_in_contract(0xF102, u16::MAX));

        let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
        let line = log
            .lines()
            .rev()
            .find(|l| l.contains("reason=store_action_unmapped") && l.contains("pipe=61698"))
            .expect("an out-of-contract store action must name itself");
        assert!(
            line.starts_with("pass_state_degraded ") && line.contains("store_action=65535"),
            "the line must carry the value that was refused: {line}"
        );
    }
}

#[cfg(test)]
mod memo_scratch_tests {
    use super::prepare_memo_scratch;

    /// The rect is left alone and only the padding is zeroed.
    ///
    /// The first assertion is the change: this buffer is recycled across binds and
    /// the caller's read overwrites the whole rect, so re-zeroing it was an
    /// 8.3 MiB memset per bind at 1920x1080 on the memo *hit* path. A `clear()`
    /// before the resize — which is what this replaced — fails it.
    ///
    /// The second is the hazard the first one introduces. `host_alloc_len` rounds
    /// the allocation up past the rect and the read does not write that tail, so a
    /// buffer still holding a previous surface's bytes there would differ from the
    /// memo and manufacture a miss.
    #[test]
    fn the_rect_is_not_rezeroed_and_the_padding_is() {
        let (span, filled) = (4096, 4000);
        let mut scratch = vec![0xAAu8; span];
        prepare_memo_scratch(&mut scratch, span, filled);
        assert_eq!(scratch.len(), span);
        assert!(
            scratch[..filled].iter().all(|&b| b == 0xAA),
            "the rect was re-zeroed; the caller's read overwrites it anyway"
        );
        assert!(
            scratch[filled..].iter().all(|&b| b == 0),
            "a recycled buffer carried its old tail into the memo comparison"
        );
    }

    /// Growing and shrinking both land on exactly `span`, with the tail zeroed.
    /// The scratch is shared across surfaces, so the geometry changes under it.
    #[test]
    fn a_recycled_buffer_resizes_either_way_with_a_clean_tail() {
        let mut scratch = vec![0xAAu8; 1024];
        prepare_memo_scratch(&mut scratch, 8192, 8000);
        assert_eq!(scratch.len(), 8192);
        assert!(scratch[8000..].iter().all(|&b| b == 0));

        prepare_memo_scratch(&mut scratch, 512, 500);
        assert_eq!(scratch.len(), 512);
        assert!(scratch[500..].iter().all(|&b| b == 0));

        // A rect that claims the whole span leaves no tail to zero, and a rect
        // larger than the span (impossible geometry) must not panic on the slice.
        prepare_memo_scratch(&mut scratch, 512, 512);
        assert_eq!(scratch.len(), 512);
        prepare_memo_scratch(&mut scratch, 512, 9999);
        assert_eq!(scratch.len(), 512);
    }

    /// The abandoned-chain recovery rail must name a lost frame.
    ///
    /// Both callers invoke this as `let _ = writeback_chain_rgba(..)` and then
    /// advance the content generation on the next line, so a bare `false` leaves
    /// the guest's pages stale while the device reports them fresh. That is the
    /// exact class the land-before-abandon rail exists to prevent, and it is
    /// also the last frame of a chain that already went wrong once.
    #[test]
    fn an_unlandable_chain_writeback_names_itself() {
        use crate::runtime::drain::store_route_count;
        let mut state =
            crate::runtime::Device::new(crate::model::DeviceId(1), crate::model::PAGE_SHIFT_X86);
        let mut host = crate::runtime::host::FakeHost::new();

        // No source at all: the commonest way this rail is reached with nothing
        // to land, and previously the quietest.
        let n = store_route_count("chain_land_refused");
        assert!(!super::writeback_chain_rgba(
            &mut state,
            &mut host,
            1,
            &[],
            &[1u8; 4],
            super::ChainAbandonCause::BackendUnavailable,
        ));
        assert_eq!(
            store_route_count("chain_land_refused"),
            n + 1,
            "an empty slot list is a lost frame, not a no-op"
        );

        // A slot whose texture_ref is unbound cannot resolve a target.
        let att = crate::runtime::decode::render::ColorAttachment {
            texture_ref: 0,
            ..Default::default()
        };
        let n = store_route_count("chain_land_refused");
        assert!(!super::writeback_chain_rgba(
            &mut state,
            &mut host,
            1,
            &[(0, att)],
            &[1u8; 4],
            super::ChainAbandonCause::TerminalRefusal,
        ));
        assert_eq!(store_route_count("chain_land_refused"), n + 1);
    }

    /// The refusal carries which break abandoned the chain.
    ///
    /// Three call sites reach this rail and each already emits its own line
    /// where it decides — but those dedupe per pipeline and this one does not,
    /// so a boot with 32 recoveries and 10 candidate causes cannot pair them.
    /// Reading the line back is the only way to assert the cause survives the
    /// hop: the tag is formatted into text, and a caller that passed a constant
    /// would still typecheck.
    #[test]
    fn the_chain_recovery_refusal_says_which_break_abandoned_it() {
        let mut state =
            crate::runtime::Device::new(crate::model::DeviceId(1), crate::model::PAGE_SHIFT_X86);
        let mut host = crate::runtime::host::FakeHost::new();

        let before = std::fs::read_to_string(crate::observe::fail_log_path())
            .unwrap_or_default()
            .len();
        for cause in [
            super::ChainAbandonCause::NoColor0,
            super::ChainAbandonCause::BackendUnavailable,
            super::ChainAbandonCause::TerminalRefusal,
        ] {
            assert!(!super::writeback_chain_rgba(
                &mut state,
                &mut host,
                1,
                &[],
                &[1u8; 4],
                cause,
            ));
        }
        let log = std::fs::read_to_string(crate::observe::fail_log_path()).unwrap_or_default();
        let added = &log[before.min(log.len())..];
        for cause in [
            super::ChainAbandonCause::NoColor0,
            super::ChainAbandonCause::BackendUnavailable,
            super::ChainAbandonCause::TerminalRefusal,
        ] {
            assert!(
                added.contains(&format!("cause={}", cause.tag())),
                "the {:?} refusal did not name its cause:\n{added}",
                cause
            );
        }
        // Three distinct tags, so a boot's recoveries can be banded by origin.
        let mut tags: Vec<&str> = [
            super::ChainAbandonCause::NoColor0,
            super::ChainAbandonCause::BackendUnavailable,
            super::ChainAbandonCause::TerminalRefusal,
        ]
        .iter()
        .map(|c| c.tag())
        .collect();
        tags.sort_unstable();
        tags.dedup();
        assert_eq!(
            tags.len(),
            3,
            "two causes share a tag and cannot be told apart"
        );
    }
}
