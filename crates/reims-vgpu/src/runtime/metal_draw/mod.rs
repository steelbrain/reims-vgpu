//! Attempt Metal draw encode for a resolved pipeline + color target.
//!
//! Loads per-function MTLB containers from the object list, materializes stream
//! binds (vertex/fragment buffers, optional index buffer, viewport/scissor),
//! calls [`crate::backend::metal::render::render_core_mrt`], and writes the RGBA
//! result into the type-11 mapping via [`mapping_write`].

#[cfg(feature = "backend-vulkan")]
use crate::backend::vulkan::engine::{DrawError, DrawPreparationDecline};
#[cfg(feature = "backend-vulkan")]
use crate::backend::vulkan::translate;
use crate::contract::endian::ld32;
use crate::contract::pixel_format::{self, TexelLayout, MTL_FORMAT_BGRA8_UNORM, RGBA8_BPP};
use crate::model::DeviceState;
// `Decline::slug` on typed draw, coverage, and translation reasons.
use crate::observe::Decline;
use crate::runtime::census::srgb_census;
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
use crate::runtime::decode::render::PASS_STORE_ACTION_DONT_CARE;
use crate::runtime::decode::render::{
    DepthAttachment, StencilAttachment, PASS_LOAD_ACTION_CLEAR, PASS_LOAD_ACTION_DONT_CARE,
    PASS_LOAD_ACTION_LOAD, PASS_STORE_ACTION_STORE,
};
use crate::runtime::decode::resource::ListObjectEntry;
#[cfg(feature = "backend-vulkan")]
use crate::runtime::decode::resource::TextureDescriptor;
use crate::runtime::decode::resource::{
    decode_buffer_descriptor, decode_buffer_texture_descriptor, decode_depth_stencil_descriptor,
    decode_function_descriptor, decode_render_pipeline_descriptor, decode_sampler_descriptor,
    decode_texture_descriptor, texture_type8_opcode, BufferTextureDescriptor, DecodeStatus,
    FunctionDescriptor, RenderPipelineDescriptor, OBJECT_TYPE_BUFFER, OBJECT_TYPE_FUNCTION,
    OBJECT_TYPE_IOSURFACE, OBJECT_TYPE_TEXTURE, OBJECT_TYPE_TEXTURE_VARIANT,
    OBJECT_TYPE_TEXTURE_VIEW, OBJECT_TYPE_TYPE7, TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE,
    TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE_WIDE,
};
use crate::runtime::gva_mem;
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::mapper;
#[cfg(feature = "backend-vulkan")]
use crate::runtime::mapper::{mapping_guest_write_verdict, GuestWriteVerdict};
use crate::runtime::mapping_write;
use crate::runtime::objects;

// The Vulkan half of this path. Gated once here rather than per item, and
// re-exported flat so callers keep naming its items
// `crate::runtime::metal_draw::<name>`.
#[cfg(feature = "backend-vulkan")]
mod vulkan;
// Only for `exec`'s pass-extent census, which declares its own copy of these
// bands because it runs on every backend. See
// `the_two_coverage_censuses_use_the_same_bands`.
#[cfg(all(test, feature = "backend-vulkan"))]
pub(crate) use vulkan::coverage_band_for_test;
#[cfg(feature = "backend-vulkan")]
pub use vulkan::*;

// The Metal ICB execute half of this path. Gated once here rather than per
// item, and re-exported flat for the same reason as `vulkan`. The
// `backend-vulkan` arm of `encode_icb_execute_and_writeback` is the one item
// the file carried that this gate does not describe, so it stays below.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
mod metal_icb;
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
pub use metal_icb::*;

// Type-8 texture-view resolution and linear texture loads. Backend-independent,
// so the module carries no gate of its own; the two items inside it that are
// arm- or test-specific keep theirs.
mod texture_view;
pub(crate) use texture_view::*;

/// Upper bound on a single buffer materialization (pathological pooled allocs).
/// Metal buffer/texture bind **index** cap (`REIMS_VGPU_METAL_MAX_BUFFERS`) — API slot
/// count, not a byte-size budget. Resource byte sizes follow the guest
/// descriptor / page-table span (zero-copy direction: no host MiB cap).
pub const MAX_BIND_SLOTS: u32 = 31;

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
/// ("swizzle if type-11") each site would re-derive the predicate, which is how
/// the two halves of a conversion end up disagreeing.
///
/// The exchange is an involution, so one routine serves both directions. Trailing
/// bytes that do not fill a whole pixel pass through untouched, matching
/// [`swap_rb_channels`].
#[cfg(feature = "backend-vulkan")]
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
/// Detection is purely structural: a `StorageBuffer` decoration at `idx` in the
/// translated vertex SPIR-V. Never keyed on a shader/struct/variable name.
#[cfg(feature = "backend-vulkan")]
fn vertex_buffer_needs_storage_binding(v_words: &[u32], idx: u32, is_stage_in: bool) -> bool {
    !is_stage_in || crate::runtime::spirv_bind::buffer_access(v_words, idx).is_some()
}

/// One pass over the fragment reflection classifying the two bind gaps the render
/// path cannot recover from. Returns `(unbound, embedded)`:
/// - **`unbound`** — standard directly-bound kinds (`[[buffer(n)]]` /
///   `[[texture(n)]]` / `[[sampler(n)]]`) the shader DECLARES but the draw never
///   provided (per the caller's membership predicates). Each names a descriptor the
///   translated SPIR-V references yet the Vulkan engine leaves unbound — an
///   undefined read that paints garbage. Entries are prefixed `buf`/`tex`/`smp` +
///   index. `ColorInput` / `ThreadgroupBuffer` / `StorageImage` reach the shader by
///   other paths (validated by `census_reflection_wellformed`) and are skipped.
/// - **`embedded`** — `EmbeddedArgBufferTexture` synthetic indices: textures
///   metal2vulkan flattened out of an `air.indirect_buffer` argument. The compute
///   path resolves these; the render (fragment) path has no code to source them, so
///   each is structurally unbindable here.
///
/// Membership is by caller-supplied predicates so the hot (all-bound) path
/// allocates nothing — both returned `Vec`s stay empty (no heap) unless a genuine
/// gap exists, which is near-never on a healthy boot.
#[cfg(feature = "backend-vulkan")]
fn frag_unbound_scan(
    bindings: &[metal2vulkan::reflect::ResourceBinding],
    has_buf: impl Fn(u32) -> bool,
    has_tex: impl Fn(u32) -> bool,
    has_smp: impl Fn(u32) -> bool,
) -> (Vec<String>, Vec<u32>) {
    use metal2vulkan::reflect::ResourceKind;
    let mut unbound: Vec<String> = Vec::new();
    let mut embedded: Vec<u32> = Vec::new();
    for rb in bindings {
        let (cls, provided) = match rb.kind {
            ResourceKind::Buffer => ("buf", has_buf(rb.metal_index)),
            ResourceKind::Texture | ResourceKind::TextureArray => ("tex", has_tex(rb.metal_index)),
            ResourceKind::Sampler => ("smp", has_smp(rb.metal_index)),
            ResourceKind::EmbeddedArgBufferTexture => {
                embedded.push(rb.metal_index);
                continue;
            }
            _ => continue,
        };
        if !provided {
            unbound.push(format!("{cls}{}", rb.metal_index));
        }
    }
    (unbound, embedded)
}

/// The fragment texture slots the shader declares, in Metal index space.
///
/// A Metal shader may declare a `[[texture(n)]]` the draw never binds, and
/// Metal defines that: sampling an unbound texture returns zero. Vulkan has no
/// such rule — a descriptor the pipeline uses must exist in the layout and be
/// valid, or the read is undefined. The Vulkan render path therefore has to
/// materialise a zero texture for every declared-but-unbound slot, and this is
/// the enumeration it fills from.
///
/// `EmbeddedArgBufferTexture` is deliberately excluded: it is not a directly
/// bindable slot and the render path declines it by its own name.
#[cfg(feature = "backend-vulkan")]
fn declared_fragment_texture_indices(
    bindings: &[metal2vulkan::reflect::ResourceBinding],
) -> Vec<u32> {
    use metal2vulkan::reflect::ResourceKind;
    let mut out: Vec<u32> = bindings
        .iter()
        .filter(|rb| matches!(rb.kind, ResourceKind::Texture | ResourceKind::TextureArray))
        .map(|rb| rb.metal_index)
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

#[cfg(feature = "backend-vulkan")]
fn reflected_sampled_binding_collision(
    vertex: &metal2vulkan::reflect::ShaderReflection,
    fragment: &metal2vulkan::reflect::ShaderReflection,
) -> bool {
    use crate::runtime::spirv_bind::{COLOR_INPUT_BINDING_BASE, TEXTURE_BINDING_BASE};

    let vertex_bindings = vertex
        .bindings
        .iter()
        .filter_map(|binding| binding.descriptor.map(|descriptor| descriptor.binding))
        .filter(|binding| (TEXTURE_BINDING_BASE..COLOR_INPUT_BINDING_BASE).contains(binding))
        .collect::<std::collections::BTreeSet<_>>();
    fragment
        .bindings
        .iter()
        .filter_map(|binding| binding.descriptor.map(|descriptor| descriptor.binding))
        .any(|binding| vertex_bindings.contains(&binding))
}

/// A depth-stencil state the Linux Vulkan engine can safely ignore because it is
/// functionally equivalent to no depth/stencil test: depth compare **Always**
/// (never occludes), depth writes off, and both stencil faces disabled. Anything
/// else — a real compare function, a depth write, or an enabled stencil face —
/// changes the rendered result if dropped, so ignoring it (the render path binds
/// no depth/stencil state) is a genuine mis-execution → wrong occlusion. macOS UI
/// compositing binds no depth-stencil at all (0 of 455k draws in a live boot); this
/// only bites 3D content (WebGL / 3D-CSS). `MTLCompareFunctionAlways = 7` is the
/// Metal API contract value (Never=0, Less=1, …, GreaterEqual=6, Always=7).
#[cfg(feature = "backend-vulkan")]
fn depth_stencil_descriptor_is_trivial(
    d: &crate::runtime::decode::resource::DepthStencilDescriptor,
) -> bool {
    const MTL_COMPARE_ALWAYS: u32 = 7;
    d.depth_compare_function == MTL_COMPARE_ALWAYS
        && !d.depth_write_enabled
        && !d.front_stencil_enabled
        && !d.back_stencil_enabled
}

/// Decode the type-7 depth-stencil descriptor a draw bound, on the Linux path
/// (the Metal `load_depth_stencil_state` is `backend-metal`-gated). Mirrors
/// `load_render_pipeline`: object-list lookup → descriptor read → decode (which
/// validates the type-7 depth-stencil tag). Returns the specific reason slug on
/// failure so the caller — which only reaches this for a bound `ds_ref != 0`, i.e.
/// a guest that explicitly asked for a depth-stencil state — can fail-visibly
/// name why the state silently fell back to no-depth instead of dropping it into
/// the same silent hole every other depth/stencil sub-case is instrumented against.
#[cfg(feature = "backend-vulkan")]
fn load_depth_stencil_descriptor<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    ds_ref: u32,
) -> Result<crate::runtime::decode::resource::DepthStencilDescriptor, &'static str> {
    let entry = objects::lookup_list_entry(state, host, task_id, ds_ref)
        .ok_or("depth_stencil_entry_missing")?;
    if entry.object_type != OBJECT_TYPE_TYPE7 {
        return Err("depth_stencil_object_type");
    }
    let desc =
        objects::read_descriptor(state, host, task_id, &entry).ok_or("depth_stencil_desc_read")?;
    decode_depth_stencil_descriptor(&desc).map_err(|_| "depth_stencil_desc_decode")
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
    pub offset: u64,
}

/// One slot of a render encoder's vertex or fragment texture table. The stage
/// is the table it is in; see [`BufferBind`].
#[derive(Clone, Debug, Default)]
pub struct TextureBind {
    pub index: u32,
    pub texture_ref: u32,
}

/// One slot of a render encoder's vertex or fragment sampler table. The stage
/// is the table it is in; see [`BufferBind`].
#[derive(Clone, Debug, Default)]
pub struct SamplerBind {
    pub index: u32,
    pub sampler_ref: u32,
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
/// Archive `ApplePVGPURenderTarget`: either type-11 IOSurface (`mapping_id`) or
/// type-2/3 guest-VA linear (`target_gva` + `row_stride`). Wallpaper/background
/// layers are the GVA form.
#[derive(Clone, Debug, Default)]
pub struct ColorRtRequest {
    pub slot: u32,
    pub texture_ref: u32,
    pub mapping_id: u32,
    /// Non-zero ⇒ type-2/3 linear GVA target (mapping_id must be 0).
    pub target_gva: u64,
    /// Bytes-per-row for GVA target (archive `bpr`).
    pub row_stride: u32,
    pub width: u32,
    pub height: u32,
    pub format: u16,
    pub load_action: u16,
    pub store_action: u16,
    pub clear_color: [f64; 4],
    pub target_seed_rgba: Option<Vec<u8>>,
}

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
    pub vertex_buffers: Vec<BufferBind>,
    pub fragment_buffers: Vec<BufferBind>,
    pub vertex_textures: Vec<TextureBind>,
    pub fragment_textures: Vec<TextureBind>,
    pub vertex_samplers: Vec<SamplerBind>,
    pub fragment_samplers: Vec<SamplerBind>,
    pub viewport: Option<[f64; 6]>,
    pub scissor: Option<(u32, u32, u32, u32)>,
    pub indexed: Option<IndexedDrawInfo>,
    pub blend_color: Option<[f32; 4]>,
    pub cull_mode: Option<u32>,
    pub front_facing: Option<u32>,
    pub depth_bias: Option<[f32; 3]>,
    pub depth_stencil_ref: u32,
    pub stencil_ref: Option<(u32, u32)>,
    pub depth_attach: Option<DepthAttachment>,
    pub stencil_attach: Option<StencilAttachment>,
    /// First draw of the Metal render pass that owns this stencil attachment.
    ///
    /// A pass clears its stencil once and then relies on it: one draw writes a
    /// mask, the next tests against it. Flattening the pass into per-draw
    /// requests loses that ordering, so the pass load action alone would clear
    /// before every draw and the mask would never survive to be tested. This
    /// says which draw the clear belongs to; the rest load.
    pub stencil_first_in_pass: bool,
    /// Records 2+ of a resident render-pass chain: load the prior record's
    /// content from the engine target instead of a CPU seed. Set by the exec
    /// chain loop (Vulkan rail only); default false.
    pub chain_from_resident: bool,
    /// Out-flag: this record kept chain content on the engine-resident
    /// target (no CPU pixels, no guest Store). The exec chain loop arms
    /// `chain_from_resident` for the next record when set.
    pub chain_resident_established: bool,
    /// Allocation identity of the color0 GVA render target: the
    /// order-independent hash of the guest physical pages backing
    /// `row_stride * height` at `colors[0].target_gva`.
    ///
    /// Resolved once per draw, before any GPU work, by
    /// `metal_draw::vulkan::gva_alloc_generation`, and carried here so every
    /// `TargetIdentity::Gva` this draw builds — the pinned Store identity, the
    /// cross-pass Load identity, the deferred window's stored copy — agrees on
    /// one `generation`. Two guest allocations that reuse one address at one
    /// geometry then get two registry slots instead of one shared GPU image
    /// whose pixels belong to whichever of them rendered last.
    ///
    /// 0 means "no allocation named": color0 is not a GVA target, or the span
    /// does not fully walk. Vulkan rail only; the Metal arm never reads it.
    pub gva_alloc_gen: u64,
}

/// Compact command-level MRT census for the always-on draw proxy.
///
/// This records only decoded render-pass state. It deliberately does not rank
/// targets by dimensions, ids, or content; the point is to expose when the
/// shader/pass contract names more attachments than the backend executes.
#[cfg(feature = "backend-vulkan")]
fn color_target_diag(colors: &[ColorRtRequest]) -> String {
    colors
        .iter()
        .map(|c| {
            format!(
                "s{}:r{}:mid{}:gva={:#x}:{}x{}:fmt={:#x}:l{}:s{}",
                c.slot,
                c.texture_ref,
                c.mapping_id,
                c.target_gva,
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

#[cfg(feature = "backend-vulkan")]
fn texture_bind_diag(textures: &[TextureBind]) -> String {
    textures
        .iter()
        .take(8)
        .map(|t| format!("i{}:r{}", t.index, t.texture_ref))
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(feature = "backend-vulkan")]
fn buffer_bind_diag(buffers: &[BufferBind]) -> String {
    buffers
        .iter()
        .take(8)
        .map(|b| format!("i{}:r{}+{:#x}", b.index, b.buffer_ref, b.offset))
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(feature = "backend-vulkan")]
fn linux_m2v_draw_failure(error: &DrawError, req: &DrawEncodeRequest) -> crate::observe::Emit {
    let indexed = req
        .indexed
        .as_ref()
        .map(|idx| {
            format!(
                "1:ty{}:n{}:r{}+{:#x}",
                idx.index_type, idx.index_count, idx.index_buffer_ref, idx.index_buffer_offset
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
        .field("prim", req.primitive_type)
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
            "viewport",
            format!("{:?}", req.viewport).replace(char::is_whitespace, ""),
        )
        .field(
            "scissor",
            format!("{:?}", req.scissor).replace(char::is_whitespace, ""),
        )
}

/// Fixed-function state decoded by the product request but not yet represented
/// by the Linux Vulkan engine request. This is an always-on diagnostic field;
/// it never changes draw execution.
#[cfg(feature = "backend-vulkan")]
fn vulkan_fixed_state_gap(req: &DrawEncodeRequest) -> String {
    let mut gaps = Vec::new();
    // Cull mode and front-facing winding ARE honored by the Vulkan raster state
    // (see the pipeline builder). Only an out-of-contract value is still a gap —
    // those stay fail-visible rather than being coerced to a face that silently
    // draws or drops geometry. What counts as out-of-contract is
    // `translate::raster`'s answer, not a local bound: a second copy of the
    // SDK's range here would silently disagree the moment one of them changed.
    if let Some(value) = req.cull_mode {
        if translate::raster::cull_mode(value).is_err() {
            gaps.push(format!("cull:{value}"));
        }
    }
    if let Some(value) = req.front_facing {
        if translate::raster::front_face_ccw(value).is_err() {
            gaps.push(format!("front:{value}"));
        }
    }
    // Depth test + attachment AND the stencil test are honored now (see
    // `resources.depth` wiring): a bound depth-stencil state attaches a transient
    // (combined) depth-stencil buffer, and a stencil-enabled state wires the
    // front/back op state + dynamic reference (`stencil_ref`) + stencil clear
    // (`stencil_attach`). The one still-unrepresented fixed-function field is
    // depth bias (Metal↔Vulkan constant-bias scale differs — unverifiable
    // without Apple ground truth). Depth LOAD and out-of-contract stencil ops
    // degrade with their own fail-visible slugs, not this census.
    if let Some([bias, slope, clamp]) = req.depth_bias {
        gaps.push(format!("bias:{bias:.3}/{slope:.3}/{clamp:.3}"));
    }
    gaps.join(",")
}

/// Resolve one decoded `DepthStencilFace` into the engine `StencilFaceOps`.
///
/// Declines by name if the compare function or any of the three ops is out of
/// contract, so the caller can log *which* field it was. The four fields carry
/// the same two Metal enums the depth path uses, and both live in
/// `translate::raster` — the one place that decides what an `MTLCompareFunction`
/// or an `MTLStencilOperation` means.
#[cfg(feature = "backend-vulkan")]
fn engine_stencil_face(
    f: &crate::runtime::decode::resource::DepthStencilFace,
) -> Result<crate::backend::vulkan::engine::StencilFaceOps, translate::TranslateReason> {
    Ok(crate::backend::vulkan::engine::StencilFaceOps {
        compare: translate::raster::compare_function(f.compare_function)?,
        fail_op: translate::raster::stencil_operation(f.stencil_failure_operation)?,
        depth_fail_op: translate::raster::stencil_operation(f.depth_failure_operation)?,
        pass_op: translate::raster::stencil_operation(f.depth_stencil_pass_operation)?,
        read_mask: f.read_mask,
        write_mask: f.write_mask,
    })
}

/// Translate one decoded raster field, falling back to Metal's default when the
/// guest bound nothing and naming the decline when it bound something this
/// contract does not cover.
///
/// The distinction is the whole point. `None` means "the guest never set this",
/// where Metal's documented default is the correct answer and there is nothing
/// to report — logging it would flood every draw. `Some(v)` that fails to
/// translate means the decode produced a value outside the SDK's range, which is
/// a real gap: the draw still runs (blocking it would lose a frame over a field
/// that may not matter) but it says so once per `(pipeline, slug)` first.
#[cfg(feature = "backend-vulkan")]
fn raster_or_default<T, E>(
    decoded: Option<u32>,
    translate_one: impl Fn(u32) -> Result<T, E>,
    metal_default: T,
    pipeline_ref: u32,
    slug: &'static str,
) -> T {
    let Some(value) = decoded else {
        return metal_default;
    };
    match translate_one(value) {
        Ok(mapped) => mapped,
        Err(_) => {
            if degrade_log_first(pipeline_ref, slug) {
                crate::observe::fail(format!(
                    "raster_state_degraded reason={slug} pipe={pipeline_ref} value={value} \
                     (out-of-contract Metal value; using Metal's default)"
                ));
            }
            metal_default
        }
    }
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
/// variant is the *class* the caller acts on — `NoMetal` makes `exec.rs` fall
/// back to the pass clear, `WritebackFailed` does not — and the payload is which
/// of the rail's checks refused. Before this, six payload-free variants spoke for
/// 27 checks: `BadArgs` alone covered eight, and `draw_encode_fail
/// reason=bad_args` could be a zero-size target, a vertexless draw or an ICB
/// range past the end of its buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncodeStatus {
    Ok,
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    MetalBackend(crate::backend::metal::error::Status),
    MissingPipeline(&'static str),
    MissingMtlb(&'static str),
    MetalFailed(&'static str),
    WritebackFailed(&'static str),
    BadArgs(&'static str),
    /// Metal feature not built (vulkan boot), or nothing landed on the Vulkan
    /// rail — `exec.rs` treats both as "honour the pass clear instead".
    NoMetal(&'static str),
}

impl crate::observe::Refusal for EncodeStatus {
    fn refusal(&self) -> Option<&'static str> {
        match self {
            // The only non-refusal, and the reason this is a `Refusal` rather
            // than a `Decline`: `Emit::refusal` cannot render a line for it.
            Self::Ok => None,
            #[cfg(all(feature = "backend-metal", target_os = "macos"))]
            Self::MetalBackend(status) => status.refusal(),
            Self::MissingPipeline(slug)
            | Self::MissingMtlb(slug)
            | Self::MetalFailed(slug)
            | Self::WritebackFailed(slug)
            | Self::BadArgs(slug)
            | Self::NoMetal(slug) => Some(slug),
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        // The class beside the reason: which recovery path the caller took is
        // not derivable from the slug, and a reader correlating a dropped draw
        // with a black frame needs both.
        #[cfg(all(feature = "backend-metal", target_os = "macos"))]
        if let Self::MetalBackend(status) = self {
            let mut fields = crate::observe::Refusal::fields(status);
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
            #[cfg(all(feature = "backend-metal", target_os = "macos"))]
            Self::MetalBackend(status) => {
                if status.is_args() {
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
            Self::EntryMissing => "draw_index_entry_missing",
            Self::ObjectType => "draw_index_object_type",
            Self::DescRead => "draw_index_desc_read",
            Self::DescDecode => "draw_index_desc_decode",
            Self::BackingMissing => "draw_index_backing_missing",
            Self::OffsetOverflow => "draw_index_offset_overflow",
            Self::OutOfBounds => "draw_index_out_of_bounds",
            Self::ReadFail => "draw_index_read_fail",
            Self::BaseVertexOutOfRange => "draw_index_base_vertex_out_of_range",
        }
    }
}

fn load_mtlb<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    func_ref: u32,
) -> Option<Vec<u8>> {
    if func_ref == 0 {
        return None;
    }
    let entry = objects::lookup_list_entry(state, host, task_id, func_ref)?;
    // Live object-list: function = type 6.
    if entry.object_type != OBJECT_TYPE_FUNCTION {
        return None;
    }
    let desc = objects::read_descriptor(state, host, task_id, &entry)?;
    let f: FunctionDescriptor = decode_function_descriptor(&desc).ok()?;
    if f.blob_gva == 0 || f.blob_size < 4 {
        return None;
    }
    // Guest blob_size is authoritative — no product 1 MiB MTLB ceiling.
    let len = host_alloc_len(f.blob_size as u64)?;
    let mut mtlb = vec![0u8; len];
    // Device page_shift (x86=12); unshifted helper defaults to arm14 and fails loads.
    gva_mem::read_task_gva_by_id(
        host,
        &state.tasks,
        task_id,
        f.blob_gva,
        &mut mtlb,
        state.page_shift,
    )
    .ok()?;
    Some(mtlb)
}

fn load_render_pipeline<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    pipeline_ref: u32,
) -> Option<RenderPipelineDescriptor> {
    // Five different losses used to share one `PipelineMissing` decline, which
    // named the pipeline but not what about it could not be resolved. The draw
    // is gone either way, so the reason is the only thing that separates "the
    // guest has not published this object yet" from "we cannot read the object
    // it published" — and those want opposite fixes.
    let say = |why: &str| {
        crate::observe::fail(format!(
            "render_pipeline_unresolved reason={why} task={task_id} ref={pipeline_ref}"
        ));
        None::<RenderPipelineDescriptor>
    };
    let Some(entry) = objects::lookup_list_entry(state, host, task_id, pipeline_ref) else {
        return say("object_list_entry_absent");
    };
    // Live object-list: render pipeline is type-7 with subtype 0x0e.
    if entry.object_type != OBJECT_TYPE_TYPE7 {
        return say("object_type_not_type7");
    }
    let Some(desc) = objects::read_descriptor(state, host, task_id, &entry) else {
        return say("descriptor_read_failed");
    };
    let Ok(p) = decode_render_pipeline_descriptor(&desc) else {
        return say("descriptor_decode_failed");
    };
    if p.vertex_func_ref == 0 || p.fragment_func_ref == 0 {
        crate::observe::fail(format!(
            "render_pipeline_unresolved reason=function_ref_unbound task={task_id} ref={pipeline_ref} vtx={} frag={} object={} mesh={}",
            p.vertex_func_ref, p.fragment_func_ref, p.object_func_ref, p.mesh_func_ref
        ));
        return None;
    }
    Some(p)
}

/// Is this pipeline object merely not readable *yet*?
///
/// The five ways `load_render_pipeline` fails split in two. An entry that is
/// absent from the task's object list, or a descriptor whose guest page is not
/// mapped at this instant, is asynchronous: the guest is still publishing it,
/// and the same read a moment later succeeds. An entry of the wrong type, a
/// descriptor that will not decode, or one that decodes with no function bound
/// is deterministic — waiting cannot change it.
///
/// The distinction decides whether the packet is retried or the draw is lost,
/// so it is drawn from the same reads `load_render_pipeline` makes rather than
/// from its collapsed `None`.
#[cfg(feature = "backend-vulkan")]
pub(crate) fn render_pipeline_unreadable_yet<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    pipeline_ref: u32,
) -> bool {
    let Some(entry) = objects::lookup_list_entry(state, host, task_id, pipeline_ref) else {
        return true;
    };
    if entry.object_type != OBJECT_TYPE_TYPE7 {
        return false;
    }
    objects::read_descriptor(state, host, task_id, &entry).is_none()
}

/// Resolve immutable AIR inputs for a render pipeline before executing its
/// packet. The Linux scheduler uses this read-only plan step to translate on a
/// background thread while unrelated child FIFOs continue draining.
#[cfg(feature = "backend-vulkan")]
pub(crate) fn load_render_air_pair<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    pipeline_ref: u32,
) -> Result<(Vec<u8>, Vec<u8>), DrawPreparationDecline> {
    let pd = load_render_pipeline(state, host, task_id, pipeline_ref).ok_or(
        DrawPreparationDecline::PipelineMissing {
            task_id,
            pipeline_ref,
        },
    )?;
    let v_mtlb = load_mtlb(state, host, task_id, pd.vertex_func_ref).ok_or(
        DrawPreparationDecline::VertexMtlbMissing {
            task_id,
            function_ref: pd.vertex_func_ref,
        },
    )?;
    let f_mtlb = load_mtlb(state, host, task_id, pd.fragment_func_ref).ok_or(
        DrawPreparationDecline::FragmentMtlbMissing {
            task_id,
            function_ref: pd.fragment_func_ref,
        },
    )?;
    let v_air = crate::runtime::mtlb::extract_air(&v_mtlb)
        .map_err(|reason| DrawPreparationDecline::VertexAirExtract {
            function_ref: pd.vertex_func_ref,
            reason,
        })?
        .to_vec();
    let f_air = crate::runtime::mtlb::extract_air(&f_mtlb)
        .map_err(|reason| DrawPreparationDecline::FragmentAirExtract {
            function_ref: pd.fragment_func_ref,
            reason,
        })?
        .to_vec();
    Ok((v_air, f_air))
}

/// Resolve type-1 buffer object → guest bytes starting at `offset`.
/// Where a type-1 buffer object's bytes live in the task GVA space. Both the
/// zero-copy gather and the CPU staging read need identical `(gva, size)`;
/// resolving it once ([`resolve_buffer_backing`]) avoids walking the task page
/// table twice for every sub-zero-copy-floor bind (the `buf_snap` population —
/// ~4.7 CPU snapshots/draw under Safari scroll, each of which previously paid
/// the object-list entry read + descriptor read + decode in the failed ZC
/// attempt *and* again in the CPU fallback).
struct BufferBacking {
    gva: u64,
    size: u64,
}

/// Resolve a type-1 buffer `ref` to its backing `(gva, size)` (object-list
/// entry read + descriptor read + decode). Fail-visible per failing site —
/// this is the single owner of the `load_buffer *` reason slugs; the ZC and CPU
/// binds delegate to it so a failure logs exactly once, not once per attempt.
fn resolve_buffer_backing<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    buffer_ref: u32,
) -> Option<BufferBacking> {
    if buffer_ref == 0 {
        return None;
    }
    let Some(entry) = objects::lookup_list_entry(state, host, task_id, buffer_ref) else {
        crate::observe::fail(format!(
            "load_buffer miss lookup task={task_id} ref={buffer_ref}"
        ));
        return None;
    };
    if entry.object_type != OBJECT_TYPE_BUFFER {
        crate::observe::fail(format!(
            "load_buffer bad type task={task_id} ref={buffer_ref} ty={}",
            entry.object_type
        ));
        return None;
    }
    let Some(desc_bytes) = objects::read_descriptor(state, host, task_id, &entry) else {
        crate::observe::fail(format!(
            "load_buffer miss desc task={task_id} ref={buffer_ref}"
        ));
        return None;
    };
    let Ok(desc) = decode_buffer_descriptor(&desc_bytes) else {
        crate::observe::fail(format!(
            "load_buffer decode fail task={task_id} ref={buffer_ref} desc_len={}",
            desc_bytes.len()
        ));
        return None;
    };
    let Some((gva, size)) = desc.backing_gva_size(state.page_shift) else {
        crate::observe::fail(format!(
            "load_buffer no backing task={task_id} ref={buffer_ref} shift={}",
            state.page_shift
        ));
        return None;
    };
    Some(BufferBacking { gva, size })
}

/// CPU staging read of a pre-resolved buffer backing at `offset`. Reads guest
/// RAM directly (reflects guest CPU writes), no host-store flush — the CPU path
/// has always read the pages as-is (the zero-copy rail owns the flush).
fn read_buffer_bytes_resolved<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    backing: &BufferBacking,
    offset: u64,
) -> Option<Vec<u8>> {
    let (gva, size) = (backing.gva, backing.size);
    if offset >= size {
        crate::observe::fail(format!(
            "load_buffer offset oob task={task_id} off={offset} size={size}"
        ));
        return None;
    }
    let avail = size - offset;
    let want = host_alloc_len(avail).filter(|&n| n > 0)?;
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
fn load_buffer_bytes<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    buffer_ref: u32,
    offset: u64,
) -> Option<Vec<u8>> {
    let backing = resolve_buffer_backing(state, host, task_id, buffer_ref)?;
    read_buffer_bytes_resolved(state, host, task_id, &backing, offset)
}

/// If `texture_ref` is a type-8 object whose descriptor is a buffer-backed
/// texture (view_opcode 9, `newTextureWithDescriptor:offset:bytesPerRow:`, or
/// its `TextureDescriptor2` form), return its decoded descriptor. `None` for a
/// non-type-8 object or a real texture VIEW (opcode 7/8/0x1b) — those stay on
/// the view path silently.
fn buffer_texture_descriptor<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    texture_ref: u32,
    entry: Option<ListObjectEntry>,
) -> Option<BufferTextureDescriptor> {
    // Reuse a caller-resolved object-list entry when supplied: the guest object
    // list is immutable for the life of a draw (the device never writes those
    // pages), so a precomputed entry is byte-identical to a fresh lookup and
    // saves a redundant guest-DMA read+decode per sampled bind.
    let entry = entry.or_else(|| objects::lookup_list_entry(state, host, task_id, texture_ref))?;
    if entry.object_type != OBJECT_TYPE_TEXTURE_VIEW {
        return None;
    }
    let desc_bytes = objects::read_descriptor(state, host, task_id, &entry)?;
    if !matches!(
        texture_type8_opcode(&desc_bytes),
        Some(TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE) | Some(TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE_WIDE)
    ) {
        return None;
    }
    decode_buffer_texture_descriptor(&desc_bytes).ok()
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
    state: &DeviceState,
    host: &M,
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

fn index_elem_size(index_type: u32) -> Option<usize> {
    match index_type {
        0 => Some(2), // MTLIndexTypeUInt16
        1 => Some(4), // MTLIndexTypeUInt32
        _ => None,
    }
}

/// Load the index bytes a bound indexed draw references, returning the **specific**
/// reason on failure. Metal emits it directly; Vulkan delegates it through
/// `DrawPreparationDecline::IndexLoad`, so both rails keep one reason vocabulary.
/// Runs on the drain worker (off main core); only reached when `req.indexed` is
/// set, so it cannot flood a 2D-UI boot.
fn load_index_bytes_reason<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    info: &IndexedDrawInfo,
) -> Result<Vec<u8>, IndexLoadReason> {
    use IndexLoadReason as R;
    let elem = index_elem_size(info.index_type).ok_or(R::TypeUnsupported)?;
    let need = (info.index_count as usize)
        .checked_mul(elem)
        .ok_or(R::CountOverflow)?;
    if need == 0 {
        return Err(R::CountZero);
    }
    let entry = objects::lookup_list_entry(state, host, task_id, info.index_buffer_ref)
        .ok_or(R::EntryMissing)?;
    if entry.object_type != 1 {
        return Err(R::ObjectType);
    }
    let desc_bytes = objects::read_descriptor(state, host, task_id, &entry).ok_or(R::DescRead)?;
    let desc = decode_buffer_descriptor(&desc_bytes).map_err(|_| R::DescDecode)?;
    let (gva, size) = desc
        .backing_gva_size(state.page_shift)
        .ok_or(R::BackingMissing)?;
    let end = info
        .index_buffer_offset
        .checked_add(need as u64)
        .ok_or(R::OffsetOverflow)?;
    if end > size {
        return Err(R::OutOfBounds);
    }
    let mut buf = vec![0u8; need];
    gva_mem::read_task_gva_by_id(
        host,
        &state.tasks,
        task_id,
        gva + info.index_buffer_offset,
        &mut buf,
        state.page_shift,
    )
    .map_err(|_| R::ReadFail)?;
    Ok(buf)
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn null_apv_buffer() -> crate::backend::metal::abi::ReimsVgpuBuffer {
    use crate::backend::metal::abi::ReimsVgpuBuffer;
    ReimsVgpuBuffer {
        binding: 0,
        data: std::ptr::null_mut(),
        len: 0,
        attribute_stride: 0,
        has_attribute_stride: 0,
        reserved0: 0,
        backing_data: std::ptr::null_mut(),
        backing_len: 0,
        backing_offset: 0,
    }
}

/// Encode one draw with Metal when vert/frag MTLBs resolve; writeback BGRA to mapping.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
pub fn encode_draw_and_writeback<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    req: &mut DrawEncodeRequest,
) -> EncodeStatus {
    encode_draw_chain(state, host, req, true, false).0
}

/// Encode one draw; optionally store to guest. Returns color0 tight RGBA8 for
/// multi-draw chaining (archive DrawJob threads output → next initial content).
///
/// `force_full_store`: when true, ignore scissor-local store even if Load+partial
/// scissor (required for multi-draw final writeback after in-process chaining).
///
/// Takes `&mut req` so multi-MiB Load seeds can be **moved** into the encoder
/// (no extra full-frame clone on the multi-draw chain).
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
pub fn encode_draw_chain<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    req: &mut DrawEncodeRequest,
    writeback_guest: bool,
    force_full_store: bool,
) -> (EncodeStatus, Option<Vec<u8>>) {
    encode_draw_chain_inner(state, host, req, writeback_guest, force_full_store)
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn encode_draw_chain_inner<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    req: &mut DrawEncodeRequest,
    writeback_guest: bool,
    force_full_store: bool,
) -> (EncodeStatus, Option<Vec<u8>>) {
    use crate::backend::metal::abi::{
        ReimsVgpuBlendState, ReimsVgpuBuffer, ReimsVgpuDepthAttachment, ReimsVgpuDepthBiasState,
        ReimsVgpuIndexedDraw, ReimsVgpuRasterState, ReimsVgpuSampledImage, ReimsVgpuSampler,
        ReimsVgpuScissor, ReimsVgpuStencilAttachment, ReimsVgpuStencilReferenceState,
        ReimsVgpuViewport, REIMS_VGPU_BINDING_SAMPLER_BASE, REIMS_VGPU_BINDING_TEXTURE_BASE,
        REIMS_VGPU_MTL_PIXEL_FORMAT_DEPTH32_FLOAT, REIMS_VGPU_MTL_PIXEL_FORMAT_STENCIL8,
    };
    use crate::backend::metal::render::{render_core_mrt, ColorRt};
    use crate::backend::metal::util::ErrOut;

    if req.colors.is_empty() {
        return (EncodeStatus::BadArgs("draw_mtl_no_color_target"), None);
    }
    // Move multi-MiB Load seeds out **before** cloning color metadata so multi-draw
    // chain frames are not duplicated (clone of empty Option is cheap).
    let mut color_seeds: Vec<Option<Vec<u8>>> = req
        .colors
        .iter_mut()
        .map(|c| c.target_seed_rgba.take())
        .collect();
    let color_list: Vec<ColorRtRequest> = req.colors.clone();
    let width = color_list[0].width;
    let height = color_list[0].height;
    if width == 0 || height == 0 {
        return (EncodeStatus::BadArgs("draw_mtl_zero_geom"), None);
    }
    // Metal pass requires matching RT dimensions.
    if color_list
        .iter()
        .any(|c| c.width != width || c.height != height || (c.mapping_id == 0 && c.target_gva == 0))
    {
        return (EncodeStatus::BadArgs("draw_mtl_mrt_geom_mismatch"), None);
    }
    // Pages each attachment's GVA Store may reach, resolved here rather than at
    // writeback: `render_core_mrt` below submits and waits, and the guest keeps
    // running on its own vCPUs across that. Indexed by attachment because MRT
    // stores every color target, not just slot 0.
    let sync_store_pages: Vec<Option<std::collections::HashSet<u64>>> = if writeback_guest {
        color_list
            .iter()
            .map(|c| sync_store_target_pages(state, host, req.task_id, c))
            .collect()
    } else {
        Vec::new()
    };
    let is_indexed = req
        .indexed
        .as_ref()
        .map(|i| i.index_count > 0)
        .unwrap_or(false);
    if !is_indexed && req.vertex_count == 0 {
        return (EncodeStatus::BadArgs("draw_mtl_no_vertices"), None);
    }

    let Some(pipeline) = load_render_pipeline(state, host, req.task_id, req.pipeline_ref) else {
        crate::observe::fail(format!(
            "metal_draw MissingPipeline pipe={}",
            req.pipeline_ref
        ));
        return (
            EncodeStatus::MissingPipeline("draw_mtl_pipeline_load"),
            None,
        );
    };
    let Some(vert) = load_mtlb(state, host, req.task_id, pipeline.vertex_func_ref) else {
        crate::observe::fail(format!(
            "metal_draw MissingMtlb vert_func={} pipe={}",
            pipeline.vertex_func_ref, req.pipeline_ref
        ));
        return (EncodeStatus::MissingMtlb("draw_mtl_vertex_mtlb_load"), None);
    };
    let Some(frag) = load_mtlb(state, host, req.task_id, pipeline.fragment_func_ref) else {
        crate::observe::fail(format!(
            "metal_draw MissingMtlb frag_func={} pipe={}",
            pipeline.fragment_func_ref, req.pipeline_ref
        ));
        return (
            EncodeStatus::MissingMtlb("draw_mtl_fragment_mtlb_load"),
            None,
        );
    };

    // Materialize buffer backs (storage first, then ReimsVgpuBuffer views).
    // Archive apple-pv-gpu-exec: a non-zero bound buffer that does not resolve
    // sets all_binds_ok=false and gates the draw (never feeds garbage geometry).
    let mut vtx_storage: Vec<Vec<u8>> = Vec::new();
    let mut frag_storage: Vec<Vec<u8>> = Vec::new();
    let mut vtx_bind_idx: Vec<u32> = Vec::new();
    let mut frag_bind_idx: Vec<u32> = Vec::new();
    for b in &req.vertex_buffers {
        if b.index >= MAX_BIND_SLOTS || b.buffer_ref == 0 {
            continue;
        }
        let Some(bytes) = load_buffer_bytes(state, host, req.task_id, b.buffer_ref, b.offset)
        else {
            crate::observe::fail(format!(
                "metal_draw gate: vertex buffer miss ref={} idx={} off={}",
                b.buffer_ref, b.index, b.offset
            ));
            return (
                EncodeStatus::MetalFailed("draw_mtl_vertex_buffer_miss"),
                None,
            );
        };
        vtx_bind_idx.push(b.index);
        vtx_storage.push(bytes);
    }
    for b in &req.fragment_buffers {
        if b.index >= MAX_BIND_SLOTS || b.buffer_ref == 0 {
            continue;
        }
        let Some(bytes) = load_buffer_bytes(state, host, req.task_id, b.buffer_ref, b.offset)
        else {
            crate::observe::fail(format!(
                "metal_draw gate: fragment buffer miss ref={} idx={} off={}",
                b.buffer_ref, b.index, b.offset
            ));
            return (
                EncodeStatus::MetalFailed("draw_mtl_fragment_buffer_miss"),
                None,
            );
        };
        frag_bind_idx.push(b.index);
        frag_storage.push(bytes);
    }

    // Stage-in attrs: layout always comes from the type-7 pipeline vertex
    // block (ICB path already does this). Host bytes attach when the stream
    // bound that buffer index; otherwise Metal still needs the descriptor or
    // PSO create fails with "Vertex function has input attributes but no
    // vertex descriptor was set".
    let stage_in_indices: std::collections::BTreeSet<u32> = pipeline
        .vertex_attributes
        .iter()
        .filter(|a| a.format != 0 && a.stride != 0)
        .map(|a| a.buffer_index)
        .collect();

    // Build ReimsVgpuVertexAttr list from pipeline vertex block + optional buffer storage.
    use crate::backend::metal::abi::ReimsVgpuVertexAttr;
    let mut attrs: Vec<ReimsVgpuVertexAttr> = Vec::new();
    let mut stage_in_with_data: std::collections::BTreeSet<u32> = Default::default();
    for a in &pipeline.vertex_attributes {
        if a.format == 0 || a.stride == 0 {
            continue;
        }
        let (data_ptr, len) =
            if let Some(pos) = vtx_bind_idx.iter().position(|&bi| bi == a.buffer_index) {
                let data = &vtx_storage[pos];
                if !data.is_empty() {
                    stage_in_with_data.insert(a.buffer_index);
                    (data.as_ptr(), data.len())
                } else {
                    (std::ptr::null(), 0)
                }
            } else {
                (std::ptr::null(), 0)
            };
        attrs.push(ReimsVgpuVertexAttr {
            location: a.location,
            format: a.format,
            offset: a.offset,
            buffer_index: a.buffer_index,
            stride: a.stride,
            data: data_ptr,
            len,
            has_step_function: if a.has_step_function { 1 } else { 0 },
            step_function: a.step_function,
            has_step_rate: if a.has_step_rate { 1 } else { 0 },
            step_rate: a.step_rate,
        });
    }

    // Bind non-stage-in buffers always; stage-in buffers only when not already
    // carried as ReimsVgpuVertexAttr host bytes (avoid double-bind).
    let mut vtx_bufs: Vec<ReimsVgpuBuffer> = Vec::new();
    for (i, data) in vtx_storage.iter().enumerate() {
        let binding = vtx_bind_idx[i];
        if stage_in_with_data.contains(&binding) {
            continue;
        }
        // Stage-in layout without bytes: still setVertexBuffer so the PSO
        // descriptor's buffer index has a bound buffer at draw time.
        let _ = stage_in_indices.contains(&binding);
        let mut ab = null_apv_buffer();
        ab.binding = binding;
        ab.data = data.as_ptr() as *mut u8;
        ab.len = data.len();
        vtx_bufs.push(ab);
    }
    let mut frag_bufs: Vec<ReimsVgpuBuffer> = Vec::with_capacity(frag_storage.len());
    for (i, data) in frag_storage.iter().enumerate() {
        let mut ab = null_apv_buffer();
        ab.binding = frag_bind_idx[i];
        ab.data = data.as_ptr() as *mut u8;
        ab.len = data.len();
        frag_bufs.push(ab);
    }

    // Sampled textures: type-11 mapping pages, then type-2/3 linear GVA.
    struct TexItem {
        index: u32,
        w: u32,
        h: u32,
        rgba: Vec<u8>,
    }
    // Archive apple-pv-gpu-exec: a bound texture that does not resolve gates the
    // draw (never samples black/garbage). Same for vertex-stage textures.
    let mut vtx_tex_items: Vec<TexItem> = Vec::new();
    let mut frag_tex_items: Vec<TexItem> = Vec::new();
    for t in &req.vertex_textures {
        if t.index >= MAX_BIND_SLOTS {
            continue;
        }
        let Some((w, h, rgba)) = load_sampled_rgba(state, host, req.task_id, t.texture_ref) else {
            crate::observe::fail(format!(
                "metal_draw gate: vertex texture miss ref={} {}",
                t.texture_ref,
                sample_miss_detail(state, host, req.task_id, t.texture_ref)
            ));
            return (
                EncodeStatus::MetalFailed("draw_mtl_vertex_texture_miss"),
                None,
            );
        };
        vtx_tex_items.push(TexItem {
            index: t.index,
            w,
            h,
            rgba,
        });
    }
    for t in &req.fragment_textures {
        if t.index >= MAX_BIND_SLOTS {
            continue;
        }
        let Some((w, h, rgba)) = load_sampled_rgba(state, host, req.task_id, t.texture_ref) else {
            crate::observe::fail(format!(
                "metal_draw gate: fragment texture miss ref={} {}",
                t.texture_ref,
                sample_miss_detail(state, host, req.task_id, t.texture_ref)
            ));
            return (
                EncodeStatus::MetalFailed("draw_mtl_fragment_texture_miss"),
                None,
            );
        };
        frag_tex_items.push(TexItem {
            index: t.index,
            w,
            h,
            rgba,
        });
    }
    let vtx_imgs: Vec<ReimsVgpuSampledImage> = vtx_tex_items
        .iter()
        .map(|it| {
            let data = it.rgba.as_ptr();
            let len = it.rgba.len();
            ReimsVgpuSampledImage {
                binding: REIMS_VGPU_BINDING_TEXTURE_BASE + it.index,
                width: it.w,
                height: it.h,
                rgba8: data,
                len,
                pixel_format: 0,
                bytes_per_row: it.w.saturating_mul(RGBA8_BPP),
                data,
                data_len: len,
            }
        })
        .collect();
    let frag_imgs: Vec<ReimsVgpuSampledImage> = frag_tex_items
        .iter()
        .map(|it| {
            let data = it.rgba.as_ptr();
            let len = it.rgba.len();
            ReimsVgpuSampledImage {
                binding: REIMS_VGPU_BINDING_TEXTURE_BASE + it.index,
                width: it.w,
                height: it.h,
                rgba8: data,
                len,
                pixel_format: 0,
                bytes_per_row: it.w.saturating_mul(RGBA8_BPP),
                data,
                data_len: len,
            }
        })
        .collect();

    // Samplers: type-7 subtype 0x03 when present. A nonzero ref is an explicit
    // guest bind; if it cannot be resolved, keep the correct fallback but make
    // the degradation visible with the exact resolver reason.
    let mut vtx_samps: Vec<ReimsVgpuSampler> = Vec::new();
    let mut frag_samps: Vec<ReimsVgpuSampler> = Vec::new();
    for s in &req.vertex_samplers {
        if s.index < MAX_BIND_SLOTS && s.sampler_ref != 0 {
            let sampler = load_sampler(state, host, req.task_id, s.sampler_ref, s.index)
                .unwrap_or_else(|error| {
                    crate::observe::Emit::decline("metal_draw_sampler_fallback", &error)
                        .field("task", req.task_id)
                        .field("pipe", req.pipeline_ref)
                        .field("stage", "vertex")
                        .fail_once(
                            (u64::from(s.sampler_ref) << 32) | (1_u64 << 30) | u64::from(s.index),
                        );
                    default_sampler(REIMS_VGPU_BINDING_SAMPLER_BASE + s.index)
                });
            vtx_samps.push(sampler);
        }
    }
    for s in &req.fragment_samplers {
        if s.index < MAX_BIND_SLOTS && s.sampler_ref != 0 {
            let sampler = load_sampler(state, host, req.task_id, s.sampler_ref, s.index)
                .unwrap_or_else(|error| {
                    crate::observe::Emit::decline("metal_draw_sampler_fallback", &error)
                        .field("task", req.task_id)
                        .field("pipe", req.pipeline_ref)
                        .field("stage", "fragment")
                        .fail_once(
                            (u64::from(s.sampler_ref) << 32) | (1_u64 << 29) | u64::from(s.index),
                        );
                    default_sampler(REIMS_VGPU_BINDING_SAMPLER_BASE + s.index)
                });
            frag_samps.push(sampler);
        }
    }

    let viewports: Vec<ReimsVgpuViewport> = req
        .viewport
        .map(|v| {
            vec![ReimsVgpuViewport {
                x: v[0] as f32,
                y: v[1] as f32,
                width: v[2] as f32,
                height: v[3] as f32,
                znear: v[4] as f32,
                zfar: v[5] as f32,
            }]
        })
        .unwrap_or_default();
    let scissors: Vec<ReimsVgpuScissor> = req
        .scissor
        .map(|(x, y, w, h)| {
            vec![ReimsVgpuScissor {
                x,
                y,
                width: w,
                height: h,
            }]
        })
        .unwrap_or_default();

    // Pipeline color0 blend + optional stream blend color.
    let mut blend = ReimsVgpuBlendState {
        enable: if pipeline.color0.blending_enabled {
            1
        } else {
            0
        },
        src_rgb: pipeline.color0.src_rgb,
        dst_rgb: pipeline.color0.dst_rgb,
        op_rgb: pipeline.color0.op_rgb,
        src_alpha: pipeline.color0.src_alpha,
        dst_alpha: pipeline.color0.dst_alpha,
        op_alpha: pipeline.color0.op_alpha,
        has_blend_color: 0,
        blend_color: [0.0; 4],
    };
    if let Some(c) = req.blend_color {
        blend.has_blend_color = 1;
        blend.blend_color = c;
    }
    // Pass blend when pipeline enables it or the stream set a constant blend color
    // (constant factors only take effect when enable is also set by the pipeline).
    let blend_opt = if blend.enable != 0 || blend.has_blend_color != 0 {
        Some(&blend)
    } else {
        None
    };

    let mut raster = ReimsVgpuRasterState {
        has_cull_mode: 0,
        cull_mode: 0,
        has_depth_clip_mode: 0,
        depth_clip_mode: 0,
        has_front_facing_winding: 0,
        front_facing_winding: 0,
        has_triangle_fill_mode: 0,
        triangle_fill_mode: 0,
        has_line_width: 0,
        line_width: 1.0,
    };
    if let Some(c) = req.cull_mode {
        raster.has_cull_mode = 1;
        raster.cull_mode = c;
    }
    if let Some(f) = req.front_facing {
        raster.has_front_facing_winding = 1;
        raster.front_facing_winding = f;
    }
    let raster_opt = if raster.has_cull_mode != 0 || raster.has_front_facing_winding != 0 {
        Some(&raster)
    } else {
        None
    };

    let depth_bias_state = req.depth_bias.map(|d| ReimsVgpuDepthBiasState {
        depth_bias: d[0],
        slope_scale: d[1],
        clamp: d[2],
    });
    let depth_bias_opt = depth_bias_state.as_ref();

    // Type-7 depth-stencil object + optional stencil reference.
    let depth_stencil_state = if req.depth_stencil_ref != 0 {
        match load_depth_stencil_state(state, host, req.task_id, req.depth_stencil_ref) {
            Ok(depth_stencil) => Some(depth_stencil),
            Err(error) => {
                crate::observe::Emit::decline("metal_draw_depth_stencil_fallback", &error)
                    .field("task", req.task_id)
                    .field("pipe", req.pipeline_ref)
                    .fail_once(u64::from(req.depth_stencil_ref));
                None
            }
        }
    } else {
        None
    };
    let depth_stencil_opt = depth_stencil_state.as_ref();
    let stencil_ref_state = req
        .stencil_ref
        .map(|(f, b)| ReimsVgpuStencilReferenceState { front: f, back: b });
    let stencil_ref_opt = stencil_ref_state.as_ref();

    // Host-side depth/stencil attachment buffers (guest LOAD / clear seed, STORE writeback).
    let mut depth_storage: Option<Vec<u8>> = None;
    let mut depth_attach_api: Option<ReimsVgpuDepthAttachment> = None;
    let mut depth_mapping_id = 0u32;
    if let Some(da) = &req.depth_attach {
        if da.present
            && da.level == 0
            && da.resolve_texture_ref == 0
            && da.load_action <= PASS_LOAD_ACTION_CLEAR
            && (da.store_action == PASS_STORE_ACTION_DONT_CARE
                || da.store_action == PASS_STORE_ACTION_STORE)
        {
            // The pass extent, same as every other attachment in it — the depth
            // buffer's rows and its row count have to come from one geometry.
            let row = width.saturating_mul(4);
            let depth_len = (row as usize).saturating_mul(height as usize);
            let mut buf = vec![0u8; depth_len];
            let mid =
                objects::resolve_type11_ref(state, host, req.task_id, da.texture_ref).unwrap_or(0);
            if mid != 0 {
                let _ = mapper::ensure_resolved_for_scanout(state, host, mid);
            }
            match da.load_action {
                x if x == PASS_LOAD_ACTION_CLEAR => {
                    fill_depth32(&mut buf, da.clear_depth as f32);
                }
                x if x == PASS_LOAD_ACTION_LOAD => {
                    let ok = if mid != 0 {
                        mapping_write::read_raw_rows(
                            state, host, mid, &mut buf, row, row, width, height,
                        )
                    } else {
                        load_linear_raw(
                            state,
                            host,
                            req.task_id,
                            da.texture_ref,
                            &mut buf,
                            row,
                            row,
                            width,
                            height,
                        )
                    };
                    if !ok {
                        // The guest asked to load prior depth and this device
                        // could not read it, so the pass runs against clear
                        // values instead: depth tests decide against content the
                        // guest never wrote. The load action deliberately stays
                        // LOAD — the seeded buffer *is* what Metal loads, so
                        // switching it to CLEAR would describe the same bytes
                        // twice — but the substitution is a loss of guest state
                        // and says so. The Vulkan arm reports the same class
                        // through `shader_state_degraded`; this one did not.
                        fill_depth32(&mut buf, da.clear_depth as f32);
                        if degrade_log_first(req.pipeline_ref, "depth_load_readback_failed") {
                            crate::observe::fail(format!(
                                "shader_state_degraded reason=depth_load_readback_failed \
                                 pipe={} task={} ds_ref={} mid={mid} {width}x{height} \
                                 (guest depth unreadable; pass seeded with clear_depth)",
                                req.pipeline_ref, req.task_id, da.texture_ref
                            ));
                        }
                    }
                }
                _ => {}
            }
            let data_ptr = buf.as_mut_ptr();
            depth_storage = Some(buf);
            depth_mapping_id = mid;
            depth_attach_api = Some(ReimsVgpuDepthAttachment {
                pixel_format: REIMS_VGPU_MTL_PIXEL_FORMAT_DEPTH32_FLOAT,
                load_action: map_load_action(req.pipeline_ref, da.load_action),
                store_action: map_store_action(da.store_action),
                clear_depth: da.clear_depth,
                data: data_ptr,
                len: depth_len,
            });
        }
    }

    let mut stencil_storage: Option<Vec<u8>> = None;
    let mut stencil_attach_api: Option<ReimsVgpuStencilAttachment> = None;
    let mut stencil_mapping_id = 0u32;
    if let Some(sa) = &req.stencil_attach {
        if sa.present
            && sa.level == 0
            && sa.resolve_texture_ref == 0
            && sa.load_action <= PASS_LOAD_ACTION_CLEAR
            && (sa.store_action == PASS_STORE_ACTION_DONT_CARE
                || sa.store_action == PASS_STORE_ACTION_STORE)
        {
            let stencil_len = (width as usize).saturating_mul(height as usize);
            let mut buf = vec![0u8; stencil_len];
            let mid =
                objects::resolve_type11_ref(state, host, req.task_id, sa.texture_ref).unwrap_or(0);
            if mid != 0 {
                let _ = mapper::ensure_resolved_for_scanout(state, host, mid);
            }
            match sa.load_action {
                x if x == PASS_LOAD_ACTION_CLEAR => {
                    buf.fill(sa.clear_stencil as u8);
                }
                x if x == PASS_LOAD_ACTION_LOAD => {
                    let ok = if mid != 0 {
                        mapping_write::read_raw_rows(
                            state, host, mid, &mut buf, width, width, width, height,
                        )
                    } else {
                        load_linear_raw(
                            state,
                            host,
                            req.task_id,
                            sa.texture_ref,
                            &mut buf,
                            width,
                            width,
                            width,
                            height,
                        )
                    };
                    if !ok {
                        // Same substitution and the same loss as the depth arm
                        // above: the guest's stencil contents are replaced by
                        // clear_stencil, so every stencil test in the pass reads
                        // state the guest never wrote.
                        buf.fill(sa.clear_stencil as u8);
                        if degrade_log_first(req.pipeline_ref, "stencil_load_readback_failed") {
                            crate::observe::fail(format!(
                                "shader_state_degraded reason=stencil_load_readback_failed \
                                 pipe={} task={} ds_ref={} mid={mid} {width}x{height} \
                                 (guest stencil unreadable; pass seeded with clear_stencil)",
                                req.pipeline_ref, req.task_id, sa.texture_ref
                            ));
                        }
                    }
                }
                _ => {}
            }
            let data_ptr = buf.as_mut_ptr();
            stencil_storage = Some(buf);
            stencil_mapping_id = mid;
            stencil_attach_api = Some(ReimsVgpuStencilAttachment {
                pixel_format: REIMS_VGPU_MTL_PIXEL_FORMAT_STENCIL8,
                load_action: map_load_action(req.pipeline_ref, sa.load_action),
                store_action: map_store_action(sa.store_action),
                clear_stencil: sa.clear_stencil,
                data: data_ptr,
                len: stencil_len,
            });
        }
    }

    let mut index_storage: Option<Vec<u8>> = None;
    let indexed_draw: Option<ReimsVgpuIndexedDraw> = if let Some(info) = &req.indexed {
        if info.index_count == 0 || info.index_buffer_ref == 0 {
            None
        } else {
            match load_index_bytes_reason(state, host, req.task_id, info) {
                Ok(bytes) => {
                    index_storage = Some(bytes);
                    let b = index_storage.as_ref().unwrap();
                    Some(ReimsVgpuIndexedDraw {
                        index_type: info.index_type,
                        index_count: info.index_count as usize,
                        base_vertex: info.base_vertex,
                        indices: b.as_ptr(),
                        indices_len: b.len(),
                        indirect: std::ptr::null(),
                    })
                }
                Err(reason) => {
                    // The reason itself is the line; `EncodeStatus` carries it
                    // onward so the boundary counter names it too. Latched per
                    // index buffer: an app whose index buffer never resolves
                    // re-submits the same draw every frame.
                    use crate::observe::Decline;
                    crate::observe::Emit::decline("metal_draw_index", &reason)
                        .field("task", req.task_id)
                        .field("pipe", req.pipeline_ref)
                        .field("buf", info.index_buffer_ref)
                        .field("off", info.index_buffer_offset)
                        .field("count", info.index_count)
                        .fail_once(info.index_buffer_ref as u64);
                    return (EncodeStatus::MetalFailed(reason.slug()), None);
                }
            }
        }
    } else {
        None
    };

    // Owned RGBA out buffers per color RT (host encode always RGBA8).
    // Seeds were moved into `color_seeds` above.
    let need = (width as usize)
        .saturating_mul(height as usize)
        .saturating_mul(RGBA8_BPP as usize);
    let mut color_outs: Vec<Vec<u8>> = (0..color_list.len()).map(|_| vec![0u8; need]).collect();

    // For indexed draws, pass index_count as vertex_count for the early gate.
    let vertex_count = if is_indexed {
        req.indexed.as_ref().map(|i| i.index_count).unwrap_or(0) as usize
    } else {
        req.vertex_count as usize
    };

    // Type-11 color targets render into a host RT and are written back by the
    // CPU. The guest-backed attachment that used to sit here aliased the
    // mapping's `mach_vm_remap` view with `newBufferWithBytesNoCopy`, so Load
    // read and Store wrote guest pages in place; that is exactly the access the
    // host GPU must not have, and the alias is gone. What runs now is the same
    // seed-and-write-back path the alias already fell through to on every
    // contract refusal (unaligned offset or row stride, span out of range, no
    // device), so this is a rung the rail has always had.
    for (i, c) in color_list.iter().enumerate() {
        if c.mapping_id == 0 {
            continue;
        }
        if c.load_action == PASS_LOAD_ACTION_LOAD && color_seeds[i].is_none() {
            color_seeds[i] =
                seed_color_load(state, host, req.task_id, c.texture_ref, 0, width, height);
            if color_seeds[i].is_none() {
                crate::observe::fail(format!(
                    "metal_draw guest_attachment_fallback_seed fail \
                     reason=load_seed_unresolved task={} pipe={} mid={} ref={} fmt={:#x} {}x{}",
                    req.task_id,
                    req.pipeline_ref,
                    c.mapping_id,
                    c.texture_ref,
                    c.format,
                    width,
                    height
                ));
            }
        }
    }

    // Build ColorRt views with raw pointers into seeds/outs (disjoint mut slices).
    let mut color_rts: Vec<ColorRt<'_>> = Vec::with_capacity(color_list.len());
    for (i, c) in color_list.iter().enumerate() {
        // Every target encodes host RGBA8 for writeback conversion.
        let out_ptr = color_outs[i].as_mut_ptr();
        let out_len = color_outs[i].len();
        let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, out_len) };
        // This slot's own entry and no other. `slot` is the index the guest
        // declared on the entry, so the vector is keyed by the same numbering
        // `c.slot` uses and `find` is an exact lookup rather than a search over
        // positions. An `or_else(first())` here could only ever fire for a slot
        // with no entry of its own, which is exactly the case where borrowing
        // another slot's blend state invents one. The compat `color0` alias it
        // looked like it served is served by the `or` below, which tests
        // `c.slot == 0`.
        let slot_blend = pipeline
            .color_attachments
            .iter()
            .find(|a| a.slot == c.slot)
            .filter(|a| a.blending_enabled)
            .map(|a| ReimsVgpuBlendState {
                enable: 1,
                src_rgb: a.src_rgb,
                dst_rgb: a.dst_rgb,
                op_rgb: a.op_rgb,
                src_alpha: a.src_alpha,
                dst_alpha: a.dst_alpha,
                op_alpha: a.op_alpha,
                has_blend_color: 0,
                blend_color: [0.0; 4],
            })
            .or({
                if pipeline.color0.blending_enabled && c.slot == 0 {
                    Some(ReimsVgpuBlendState {
                        enable: 1,
                        src_rgb: pipeline.color0.src_rgb,
                        dst_rgb: pipeline.color0.dst_rgb,
                        op_rgb: pipeline.color0.op_rgb,
                        src_alpha: pipeline.color0.src_alpha,
                        dst_alpha: pipeline.color0.dst_alpha,
                        op_alpha: pipeline.color0.op_alpha,
                        has_blend_color: 0,
                        blend_color: [0.0; 4],
                    })
                } else {
                    None
                }
            });
        color_rts.push(ColorRt {
            slot: c.slot,
            // Host RT: 0 = RGBA8Unorm (writeback conversion path).
            pixel_format: 0,
            seed_rgba8: color_seeds[i].as_deref(),
            out_rgba8: Some(out),
            clear_r: c.clear_color[0],
            clear_g: c.clear_color[1],
            clear_b: c.clear_color[2],
            clear_a: c.clear_color[3],
            load_action: map_load_action(req.pipeline_ref, c.load_action),
            blend: slot_blend,
            // Read without the `blending_enabled` filter the blend resolve
            // above applies: an unblended masked attachment still leaves its
            // unwritten channels alone. No `first()` fallback either — a
            // secondary slot with no entry of its own writes every channel,
            // which is what the absent tag means.
            write_mask: pipeline
                .color_attachments
                .iter()
                .find(|a| a.slot == c.slot)
                .map(|a| a.write_mask)
                .unwrap_or_default()
                .bits(),
        });
    }

    let mut err_buf = [0i8; 256];
    let err: ErrOut<'_> = (err_buf.as_mut_ptr(), err_buf.len());
    let st = render_core_mrt(
        &vert,
        &frag,
        width,
        height,
        vertex_count,
        req.first_vertex as usize,
        req.instance_count.max(1) as usize,
        req.base_instance as usize,
        req.primitive_type,
        None,
        indexed_draw.as_ref(),
        &attrs,
        &vtx_bufs,
        &frag_bufs,
        &vtx_imgs,
        &vtx_samps,
        &frag_imgs,
        &frag_samps,
        &viewports,
        &scissors,
        raster_opt,
        depth_bias_opt,
        depth_stencil_opt,
        stencil_ref_opt,
        depth_attach_api.as_mut(),
        stencil_attach_api.as_mut(),
        blend_opt,
        &mut color_rts,
        err,
    );
    // Keep owned storage live through render_core_mrt (ReimsVgpu* hold raw pointers).
    let _ = (
        &vtx_storage,
        &frag_storage,
        &vtx_tex_items,
        &frag_tex_items,
        &index_storage,
        &attrs,
        &pipeline,
        &depth_storage,
        &stencil_storage,
        &depth_stencil_state,
    );
    if !st.is_ok() {
        return (EncodeStatus::MetalBackend(st), None);
    }

    // Convert each color RT RGBA8 → guest format and writeback (type-11 mapping
    // or type-2/3 GVA — archive write_type11_rgba / write_gva_rgba).
    // Multi-draw intermediate records skip guest store (archive one writeback).
    let mut any_write = false;
    if !writeback_guest {
        // Still log + early paint latch only when storing; chain returns RGBA.
        return (EncodeStatus::Ok, color_outs.first().cloned());
    }
    for (i, c) in color_list.iter().enumerate() {
        if c.store_action == PASS_STORE_ACTION_DONT_CARE {
            continue;
        }
        let out_rgba = &color_outs[i];
        // Type-2/3 GVA keeps archive image_changed via store_seed_policy.
        let load_seed = color_seeds.get(i).and_then(|s| s.as_deref());
        let seed_for_store = store_seed_policy(force_full_store, c.load_action, load_seed);
        let gva_partial = seed_for_store.is_some()
            && req
                .scissor
                .map(|(x, y, w, h)| x > 0 || y > 0 || w < width || h < height)
                .unwrap_or(false);
        let wrote = if c.mapping_id != 0 {
            if gva_partial {
                let (sx, sy, sw, sh) = req.scissor.unwrap();
                write_mapping_rgba8_rect(
                    state,
                    host,
                    c.mapping_id,
                    width,
                    height,
                    c.format,
                    out_rgba,
                    sx,
                    sy,
                    sw,
                    sh,
                )
            } else {
                mapping_write::write_rgba8_image_changed(
                    state,
                    host,
                    c.mapping_id,
                    out_rgba,
                    seed_for_store,
                    width,
                    height,
                )
            }
        } else if c.target_gva != 0 {
            let allowed = sync_store_pages.get(i).and_then(|p| p.as_ref());
            if gva_partial {
                let (sx, sy, sw, sh) = req.scissor.unwrap();
                write_gva_rgba8_rect(
                    state,
                    host,
                    req.task_id,
                    c.target_gva,
                    width,
                    height,
                    c.row_stride,
                    c.format,
                    out_rgba,
                    sx,
                    sy,
                    sw,
                    sh,
                    allowed,
                )
            } else {
                write_gva_rgba8_within(
                    state,
                    host,
                    req.task_id,
                    c.target_gva,
                    width,
                    height,
                    c.row_stride,
                    c.format,
                    out_rgba,
                    allowed,
                )
                .is_ok()
            }
        } else {
            false
        };
        if wrote {
            any_write = true;
            // Early-boot logo+pill: paint type-11 front before first DisplaySwap.
            if c.mapping_id != 0 {
                crate::runtime::scanout::note_front_buffer_writeback(
                    state,
                    host,
                    c.mapping_id,
                    width,
                    height,
                    c.format,
                );
            }
        } else {
            let (nz, maxb) = crate::observe::nonzero_stats(out_rgba);
            crate::observe::fail(format!(
                "metal_draw writeback fail mid={} gva={:#x} fmt={:#x} {}x{} rgba_nz={} max={}",
                c.mapping_id, c.target_gva, c.format, width, height, nz, maxb
            ));
        }
    }
    // Only a total writeback failure is an error: a partial MRT writeback is Ok
    // if at least one RT landed, and each RT that did not has already emitted its
    // own `metal_draw writeback fail` line above.
    if !any_write {
        return (
            EncodeStatus::WritebackFailed("draw_mtl_writeback_none"),
            None,
        );
    }

    // Optional depth/stencil store writeback into type-11 mappings.
    if let (Some(da), Some(buf)) = (&req.depth_attach, &depth_storage) {
        if da.store_action == PASS_STORE_ACTION_STORE && depth_mapping_id != 0 {
            let row = width.saturating_mul(4);
            let _ = mapper::ensure_resolved_for_scanout(state, host, depth_mapping_id);
            let _ = mapping_write::write_raw_rows(
                state,
                host,
                depth_mapping_id,
                buf,
                row,
                row,
                width,
                height,
            );
        }
    }
    if let (Some(sa), Some(buf)) = (&req.stencil_attach, &stencil_storage) {
        if sa.store_action == PASS_STORE_ACTION_STORE && stencil_mapping_id != 0 {
            let _ = mapper::ensure_resolved_for_scanout(state, host, stencil_mapping_id);
            let _ = mapping_write::write_raw_rows(
                state,
                host,
                stencil_mapping_id,
                buf,
                width,
                width,
                width,
                height,
            );
        }
    }
    let color0_rgba = color_outs.first().cloned();
    (EncodeStatus::Ok, color0_rgba)
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn fill_depth32(buf: &mut [u8], depth: f32) {
    let bits = depth.to_bits().to_le_bytes();
    for i in 0..(buf.len() / 4) {
        buf[i * 4..i * 4 + 4].copy_from_slice(&bits);
    }
}

/// Type-2/3 linear GVA raw image read (tight dst rows of `row_bytes`).
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
// A raw image read is addressed by texture, level, geometry and destination
// stride; every one of those is a separate wire-decoded value.
#[allow(clippy::too_many_arguments)]
fn load_linear_raw<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    texture_ref: u32,
    dst: &mut [u8],
    dst_stride: u32,
    row_bytes: u32,
    width: u32,
    height: u32,
) -> bool {
    if texture_ref == 0 || width == 0 || height == 0 || row_bytes == 0 || dst_stride < row_bytes {
        return false;
    }
    let entry = match objects::lookup_list_entry(state, host, task_id, texture_ref) {
        Some(e) => e,
        None => return false,
    };
    if entry.object_type != OBJECT_TYPE_TEXTURE && entry.object_type != OBJECT_TYPE_TEXTURE_VARIANT
    {
        return false;
    }
    let desc_bytes = match objects::read_descriptor(state, host, task_id, &entry) {
        Some(d) => d,
        None => return false,
    };
    let tex = match decode_texture_descriptor(&desc_bytes) {
        Ok(t) => t,
        Err(_) => return false,
    };
    if !tex.has_width
        || !tex.has_height
        || !tex.has_row_stride
        || tex.width != width
        || tex.height != height
        || tex.row_stride < row_bytes
    {
        return false;
    }
    let (gva, alloc) = match tex.backing_gva_size(state.page_shift) {
        Some(v) => v,
        None => return false,
    };
    let need = (tex.row_stride as u64).saturating_mul(height as u64);
    if need > alloc.saturating_sub(tex.data_offset as u64) {
        return false;
    }
    let need_dst = (height as u64).saturating_mul(dst_stride as u64) as usize;
    if dst.len() < need_dst {
        return false;
    }
    let mut row = vec![0u8; row_bytes as usize];
    for y in 0..height {
        let row_gva = match gva.checked_add((y as u64).saturating_mul(tex.row_stride as u64)) {
            Some(a) => a,
            None => return false,
        };
        if gva_mem::read_task_gva_by_id(
            host,
            &state.tasks,
            task_id,
            row_gva,
            &mut row,
            state.page_shift,
        )
        .is_err()
        {
            return false;
        }
        let off = (y as usize) * (dst_stride as usize);
        dst[off..off + row_bytes as usize].copy_from_slice(&row);
    }
    true
}

/// Guest Store seed for type-11 `image_changed` / GVA partial writeback.
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
    if force_full_store || load_action != PASS_LOAD_ACTION_LOAD {
        None
    } else {
        load_seed
    }
}

// A six-way type-11 Load resolver stood here — `Type11LoadChoice`,
// `Type11LoadDecision` and `resolve_type11_load_decision` — deciding whether a
// LOAD should read the resident target, upload a CPU seed, or clear black, and
// naming *which* check decided so the always-on census could separate the three
// routes to a seed.
//
// It existed because a type-11 Store that landed by host-pointer import left the
// attachment resident-only with no CPU bytes, so a later LOAD had to work out
// which resident held the frame the guest's compositor computes its damage
// against. Stores read back and seed from guest pages now, so every LOAD has its
// bytes and there is nothing to resolve. Its one caller was inside the
// `try_import` branch and went out with it.
//
// Worth noting what this resolver was for, because it was the honest half of a
// problem that is now moot: its `PresentBoundary` arm was derived from live
// forensics rather than any decoded field — the guest sends no "re-seed from the
// front buffer" instruction — and the typed decisions existed so that arm could
// be told apart from the two legitimate seed paths and weighed for deletion.

/// Premultiplied `src` over `dst` with Metal factors **One / OneMinusSrcAlpha**.
///
/// Live x86 class (2026-07-13 serial-210321): pipe-17 Load fills mid=1 solid gray
/// (`rgb_nz=2073600`), then pipe-26 Load+blend seeds correctly but stores chrome-
/// only (`rgb_nz=6018`) — engine fragment coverage is opaque black (A=255)
/// outside chrome so alpha0-hole composite fills 0 and wipes the desktop base.
///
/// Contract: when color0 blend is One/OneMinusSrcAlpha, the **attachment Load
/// composite** is `src + dst*(1-src.a)` (premult). Applying that in software to
/// the pure fragment color (draw over black clear) recovers seed under true
/// transparent fragments. Opaque black still wins (intentional Clear-like black).
///
/// Returns `(pixels, blended_texels)` where blended counts texels with `src.a < 255`.
/// Software premult One/OMSA composite. **The product path does not call this**
/// — the hardware does Load+blend — and its two unit tests only check it against
/// hand-written constants, so it reads as dead on both of the obvious checks.
/// It is not. `premult_one_omsa_gpu_matches_software_composite` in
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

/// Decoded `MTLLoadAction` → the Metal C ABI value.
///
/// `MTLLoadAction` has exactly three values, so all three are spelled out and
/// there is no catch-all. There used to be one, `_ => DONT_CARE`, and it was
/// the most destructive default available: DONT_CARE tells Metal the previous
/// attachment contents may be discarded, so a decode that read the wrong offset
/// produced a *discarded framebuffer* and no log line at all. An unrecognised
/// value now says so once per `(pipeline, slug)`.
///
/// The fallback stays DONT_CARE rather than becoming LOAD or CLEAR. Out of
/// contract means this crate misread the field, not that the guest asked for
/// something exotic — every alternative is equally a guess, and inventing
/// semantics for an unknown wire value is what the ground rules forbid. What
/// changes is that the guess is now visible.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn map_load_action(pipeline_ref: u32, a: u16) -> u32 {
    use crate::backend::metal::abi::{
        REIMS_VGPU_MTL_LOAD_ACTION_CLEAR, REIMS_VGPU_MTL_LOAD_ACTION_DONT_CARE,
        REIMS_VGPU_MTL_LOAD_ACTION_LOAD,
    };
    match a {
        PASS_LOAD_ACTION_DONT_CARE => REIMS_VGPU_MTL_LOAD_ACTION_DONT_CARE,
        PASS_LOAD_ACTION_LOAD => REIMS_VGPU_MTL_LOAD_ACTION_LOAD,
        PASS_LOAD_ACTION_CLEAR => REIMS_VGPU_MTL_LOAD_ACTION_CLEAR,
        other => {
            if metal_degrade_log_first(pipeline_ref, "load_action_unmapped") {
                crate::observe::fail(format!(
                    "pass_state_degraded reason=load_action_unmapped \
                     pipe={pipeline_ref} load_action={other} \
                     (not one of MTLLoadAction 0/1/2; attachment treated as DontCare)"
                ));
            }
            REIMS_VGPU_MTL_LOAD_ACTION_DONT_CARE
        }
    }
}

/// `degrade_log_first`'s Metal-arm twin — the Vulkan one is cfg'd to the engine
/// path, and a per-`(pipeline, slug)` latch is what keeps a recurring pass-state
/// degradation from flooding `/tmp/reims-vgpu-fail.log` on every draw.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn metal_degrade_log_first(pipeline_ref: u32, slug: &'static str) -> bool {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(u32, &'static str)>>> = Mutex::new(None);
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    seen.get_or_insert_with(HashSet::new)
        .insert((pipeline_ref, slug))
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn map_store_action(a: u16) -> u32 {
    use crate::backend::metal::abi::{
        REIMS_VGPU_MTL_STORE_ACTION_DONT_CARE, REIMS_VGPU_MTL_STORE_ACTION_STORE,
    };
    if a == PASS_STORE_ACTION_STORE {
        REIMS_VGPU_MTL_STORE_ACTION_STORE
    } else {
        REIMS_VGPU_MTL_STORE_ACTION_DONT_CARE
    }
}

/// Exact failures while resolving stream state for a direct-Metal encoder.
///
/// A nonzero sampler/depth-stencil ref is an explicit guest bind. Falling back
/// to a default sampler or disabling depth after one of these checks fails is a
/// real degradation, not the speculative `ref == 0` path.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MetalStateDecline {
    SamplerEntryMissing {
        sampler_ref: u32,
        index: u32,
    },
    SamplerObjectType {
        sampler_ref: u32,
        index: u32,
        object_type: u8,
    },
    SamplerDescriptorMissing {
        sampler_ref: u32,
        index: u32,
    },
    SamplerDecode {
        sampler_ref: u32,
        index: u32,
        reason: DecodeStatus,
    },
    DepthStencilEntryMissing {
        depth_stencil_ref: u32,
    },
    DepthStencilObjectType {
        depth_stencil_ref: u32,
        object_type: u8,
    },
    DepthStencilDescriptorMissing {
        depth_stencil_ref: u32,
    },
    DepthStencilDecode {
        depth_stencil_ref: u32,
        reason: DecodeStatus,
    },
    IcbDepthStencilUnsupported {
        depth_stencil_ref: u32,
        depth_attachment: bool,
        stencil_attachment: bool,
    },
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
impl crate::observe::Decline for MetalStateDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::SamplerEntryMissing { .. } => "metal_sampler_entry_missing",
            Self::SamplerObjectType { .. } => "metal_sampler_object_type",
            Self::SamplerDescriptorMissing { .. } => "metal_sampler_descriptor_missing",
            Self::SamplerDecode { reason, .. } => reason.slug(),
            Self::DepthStencilEntryMissing { .. } => "metal_depth_stencil_entry_missing",
            Self::DepthStencilObjectType { .. } => "metal_depth_stencil_object_type",
            Self::DepthStencilDescriptorMissing { .. } => "metal_depth_stencil_descriptor_missing",
            Self::DepthStencilDecode { reason, .. } => reason.slug(),
            Self::IcbDepthStencilUnsupported { .. } => "metal_icb_depth_stencil_unsupported",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::SamplerEntryMissing { sampler_ref, index }
            | Self::SamplerDescriptorMissing { sampler_ref, index } => vec![
                ("sampler_ref", sampler_ref.to_string()),
                ("index", index.to_string()),
            ],
            Self::SamplerObjectType {
                sampler_ref,
                index,
                object_type,
            } => vec![
                ("sampler_ref", sampler_ref.to_string()),
                ("index", index.to_string()),
                ("object_type", object_type.to_string()),
            ],
            Self::SamplerDecode {
                sampler_ref,
                index,
                reason,
            } => {
                let mut fields = reason.fields();
                fields.push(("sampler_ref", sampler_ref.to_string()));
                fields.push(("index", index.to_string()));
                fields
            }
            Self::DepthStencilEntryMissing { depth_stencil_ref }
            | Self::DepthStencilDescriptorMissing { depth_stencil_ref } => {
                vec![("depth_stencil_ref", depth_stencil_ref.to_string())]
            }
            Self::DepthStencilObjectType {
                depth_stencil_ref,
                object_type,
            } => vec![
                ("depth_stencil_ref", depth_stencil_ref.to_string()),
                ("object_type", object_type.to_string()),
            ],
            Self::DepthStencilDecode {
                depth_stencil_ref,
                reason,
            } => {
                let mut fields = reason.fields();
                fields.push(("depth_stencil_ref", depth_stencil_ref.to_string()));
                fields
            }
            Self::IcbDepthStencilUnsupported {
                depth_stencil_ref,
                depth_attachment,
                stencil_attachment,
            } => vec![
                ("depth_stencil_ref", depth_stencil_ref.to_string()),
                ("depth_attachment", u8::from(*depth_attachment).to_string()),
                (
                    "stencil_attachment",
                    u8::from(*stencil_attachment).to_string(),
                ),
            ],
        }
    }
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn icb_depth_stencil_decline(req: &DrawEncodeRequest) -> Option<MetalStateDecline> {
    let depth_attachment = req
        .depth_attach
        .as_ref()
        .is_some_and(|attachment| attachment.present);
    let stencil_attachment = req
        .stencil_attach
        .as_ref()
        .is_some_and(|attachment| attachment.present);
    (req.depth_stencil_ref != 0 || depth_attachment || stencil_attachment).then_some(
        MetalStateDecline::IcbDepthStencilUnsupported {
            depth_stencil_ref: req.depth_stencil_ref,
            depth_attachment,
            stencil_attachment,
        },
    )
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn load_depth_stencil_state<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    ds_ref: u32,
) -> Result<crate::backend::metal::abi::ReimsVgpuDepthStencilState, MetalStateDecline> {
    use crate::backend::metal::abi::{ReimsVgpuDepthStencilFaceState, ReimsVgpuDepthStencilState};
    let entry = objects::lookup_list_entry(state, host, task_id, ds_ref).ok_or(
        MetalStateDecline::DepthStencilEntryMissing {
            depth_stencil_ref: ds_ref,
        },
    )?;
    if entry.object_type != OBJECT_TYPE_TYPE7 {
        return Err(MetalStateDecline::DepthStencilObjectType {
            depth_stencil_ref: ds_ref,
            object_type: entry.object_type,
        });
    }
    let desc = objects::read_descriptor(state, host, task_id, &entry).ok_or(
        MetalStateDecline::DepthStencilDescriptorMissing {
            depth_stencil_ref: ds_ref,
        },
    )?;
    let d = decode_depth_stencil_descriptor(&desc).map_err(|reason| {
        MetalStateDecline::DepthStencilDecode {
            depth_stencil_ref: ds_ref,
            reason,
        }
    })?;
    Ok(ReimsVgpuDepthStencilState {
        depth_compare_function: d.depth_compare_function,
        depth_write_enabled: if d.depth_write_enabled { 1 } else { 0 },
        front_stencil_enabled: if d.front_stencil_enabled { 1 } else { 0 },
        back_stencil_enabled: if d.back_stencil_enabled { 1 } else { 0 },
        front_face: ReimsVgpuDepthStencilFaceState {
            compare_function: d.front_face.compare_function,
            stencil_failure_operation: d.front_face.stencil_failure_operation,
            depth_failure_operation: d.front_face.depth_failure_operation,
            depth_stencil_pass_operation: d.front_face.depth_stencil_pass_operation,
            read_mask: d.front_face.read_mask,
            write_mask: d.front_face.write_mask,
        },
        back_face: ReimsVgpuDepthStencilFaceState {
            compare_function: d.back_face.compare_function,
            stencil_failure_operation: d.back_face.stencil_failure_operation,
            depth_failure_operation: d.back_face.depth_failure_operation,
            depth_stencil_pass_operation: d.back_face.depth_stencil_pass_operation,
            read_mask: d.back_face.read_mask,
            write_mask: d.back_face.write_mask,
        },
    })
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn default_sampler(binding: u32) -> crate::backend::metal::abi::ReimsVgpuSampler {
    use crate::backend::metal::abi::ReimsVgpuSampler;
    ReimsVgpuSampler {
        binding,
        unnormalized: 0,
        min_filter: 1, // linear
        mag_filter: 1,
        mip_filter: 0,     // not mipmapped
        s_address_mode: 0, // clamp to edge
        t_address_mode: 0,
        r_address_mode: 0,
        border_color: 0,
        compare_function: 0,
        lod_min_bits: 0f32.to_bits(),
        lod_max_bits: f32::MAX.to_bits(),
        max_anisotropy: 1,
        lod_average: 0,
        support_argument_buffers: 0,
        has_lod_clamp: 0,
        clamp_lod_min_bits: 0,
        clamp_lod_max_bits: 0,
    }
}

/// The Metal sampler ABI record for a decoded type-7 sampler descriptor.
///
/// One constructor for every encoder that builds this record — the render path,
/// the direct compute path, and both ICB-inherit paths. It is an eighteen-field
/// `repr(C)` mirror of a C struct, so a field added or reinterpreted in one
/// copy and not the others is a silent ABI disagreement rather than a build
/// error.
///
/// Two things the descriptor does not settle, and the caller does:
///
/// - `lod_clamp` is the clamp carried by the guest's *sampler binding* rather
///   than by the sampler object. When present it replaces the descriptor's own
///   clamp; the binding is the later statement.
/// - `argument_buffers` forces `support_argument_buffers` on for a sampler that
///   is resident in an argument buffer. That residency is a property of how the
///   pipeline binds it, which the type-7 descriptor cannot state.
///
/// `has_lod_clamp` is always 1: both clamp fields are filled on every path
/// here, from the binding when it carried one and from the descriptor
/// otherwise. [`default_sampler`] is the one record with no clamp to describe.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
pub(crate) fn sampler_record(
    binding: u32,
    sd: &crate::runtime::decode::resource::SamplerDescriptor,
    lod_clamp: Option<(u32, u32)>,
    argument_buffers: bool,
) -> crate::backend::metal::abi::ReimsVgpuSampler {
    use crate::backend::metal::abi::ReimsVgpuSampler;
    let (lod_min, lod_max) =
        lod_clamp.unwrap_or((sd.lod_min_clamp.to_bits(), sd.lod_max_clamp.to_bits()));
    ReimsVgpuSampler {
        binding,
        unnormalized: if sd.normalized_coordinates { 0 } else { 1 },
        min_filter: sd.min_filter,
        mag_filter: sd.mag_filter,
        mip_filter: sd.mip_filter,
        s_address_mode: sd.s_address,
        t_address_mode: sd.t_address,
        r_address_mode: sd.r_address,
        border_color: sd.border_color,
        compare_function: sd.compare_function,
        lod_min_bits: lod_min,
        lod_max_bits: lod_max,
        max_anisotropy: sd.max_anisotropy.max(1),
        lod_average: if sd.lod_average { 1 } else { 0 },
        support_argument_buffers: if argument_buffers || sd.support_argument_buffers {
            1
        } else {
            0
        },
        has_lod_clamp: 1,
        clamp_lod_min_bits: lod_min,
        clamp_lod_max_bits: lod_max,
    }
}

#[cfg(all(test, feature = "backend-metal", target_os = "macos"))]
mod sampler_record_tests {
    use crate::runtime::decode::resource::SamplerDescriptor;

    fn descriptor() -> SamplerDescriptor {
        SamplerDescriptor {
            min_filter: 1,
            mag_filter: 1,
            mip_filter: 2,
            s_address: 3,
            t_address: 4,
            r_address: 5,
            max_anisotropy: 0,
            lod_min_clamp: 0.25,
            lod_max_clamp: 8.0,
            compare_function: 6,
            border_color: 1,
            normalized_coordinates: true,
            support_argument_buffers: false,
            lod_average: true,
        }
    }

    /// The sampler *binding*'s clamp is the later statement and replaces the
    /// sampler object's own, in both the reported and the clamp field pair.
    #[test]
    fn the_binding_clamp_replaces_the_descriptor_clamp() {
        let sd = descriptor();
        let from_object = super::sampler_record(64, &sd, None, false);
        assert_eq!(from_object.lod_min_bits, 0.25f32.to_bits());
        assert_eq!(from_object.lod_max_bits, 8.0f32.to_bits());
        assert_eq!(from_object.clamp_lod_min_bits, from_object.lod_min_bits);
        assert_eq!(from_object.clamp_lod_max_bits, from_object.lod_max_bits);

        let from_binding = super::sampler_record(64, &sd, Some((7, 9)), false);
        assert_eq!(from_binding.lod_min_bits, 7);
        assert_eq!(from_binding.lod_max_bits, 9);
        assert_eq!(from_binding.clamp_lod_min_bits, 7);
        assert_eq!(from_binding.clamp_lod_max_bits, 9);
    }

    /// Argument-buffer residency is the caller's to state and can only add
    /// support, never withdraw what the descriptor already granted.
    #[test]
    fn argument_buffer_residency_only_adds_support() {
        let mut sd = descriptor();
        assert_eq!(
            super::sampler_record(64, &sd, None, false).support_argument_buffers,
            0
        );
        assert_eq!(
            super::sampler_record(64, &sd, None, true).support_argument_buffers,
            1
        );
        sd.support_argument_buffers = true;
        assert_eq!(
            super::sampler_record(64, &sd, None, false).support_argument_buffers,
            1
        );
    }

    /// Anisotropy of zero is not a Metal value; every encoder floored it at one
    /// and the shared record keeps doing so.
    #[test]
    fn anisotropy_is_floored_at_one() {
        let mut sd = descriptor();
        assert_eq!(
            super::sampler_record(64, &sd, None, false).max_anisotropy,
            1
        );
        sd.max_anisotropy = 4;
        assert_eq!(
            super::sampler_record(64, &sd, None, false).max_anisotropy,
            4
        );
    }
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
                Some(d) if d.len() < objects::TYPE5_MIN_LEN => {
                    format!("type=5 desc_len={desc_len} reason=short_desc")
                }
                Some(d) => {
                    let sid = ld32(&d[objects::TYPE5_SURFACE_ID..]);
                    match objects::decode_type5_texture_view(&d) {
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
        OBJECT_TYPE_IOSURFACE => {
            let Some(mid) = objects::resolve_type11_ref(state, host, task_id, texture_ref) else {
                return format!("type=11 desc_len={desc_len} reason=type11_resolve");
            };
            match state.mappings.get(&mid) {
                None => format!("type=11 mid={mid} desc_len={desc_len} reason=no_mapping"),
                Some(m) => format!(
                    "type=11 mid={mid} desc_len={desc_len} geom={} {}x{} fmt={:#x} mapped={} pages={} reason=type11_sample",
                    m.has_geom as u8,
                    m.width,
                    m.height,
                    m.format,
                    m.mapped as u8,
                    m.page_entries.len()
                )}
        }
        OBJECT_TYPE_TEXTURE_VIEW => {
            // Opcode-9 buffer-backed textures share the type-8 tag but are not views.
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
        OBJECT_TYPE_TEXTURE | OBJECT_TYPE_TEXTURE_VARIANT => {
            let Some(desc_bytes) = objects::read_descriptor(state, host, task_id, &entry) else {
                return format!("type={ot} desc_len={desc_len} reason=desc_read");
            };
            match decode_texture_descriptor(&desc_bytes) {
                Err(_) => format!("type={ot} desc_len={desc_len} reason=desc_decode"),
                Ok(tex) => {
                    let l0 = tex.level(0);
                    format!(
                        "type={ot} desc_len={desc_len} has_fmt={} fmt={:#x} mips={} handle={:#x} alloc={} L0={}x{} bpr={} reason=linear_sample",
                        tex.has_pixel_format as u8,
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

/// Load a sampled texture as tight RGBA8: type-11, type-8→base+mip+format+swizzle, or type-2/3.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn load_sampled_rgba<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
) -> Option<(u32, u32, Vec<u8>)> {
    if texture_ref == 0 {
        return None;
    }
    // Opcode-9 buffer-backed texture (type-8): sample the source buffer directly.
    if let Some(bt) = buffer_texture_descriptor(state, host, task_id, texture_ref, None) {
        return load_buffer_texture_rgba(state, host, task_id, texture_ref, &bt);
    }
    if let Some(v) = load_type11_rgba(state, host, task_id, texture_ref, None) {
        return Some(v);
    }
    // Type-8 view → base texture + selected mip + format override + optional swizzle.
    if let Some(view) = resolve_texture_view(state, host, task_id, texture_ref) {
        let mut loaded = if let Some(v) = load_type11_rgba(
            state,
            host,
            task_id,
            view.base_texture_ref,
            view.pixel_format,
        ) {
            // Type-11 IOSurface textures are single-level only: Metal rejects
            // mipmapped IOSurface descriptors. Non-zero view level_base fails.
            if view.level != 0 {
                return None;
            }
            v
        } else {
            load_linear_texture_rgba_at_level(
                state,
                host,
                task_id,
                view.base_texture_ref,
                view.level,
                view.pixel_format,
            )?
        };
        apply_view_swizzle_rgba8(&mut loaded.2, view.swizzle.as_ref(), texture_ref)?;
        return Some(loaded);
    }
    load_linear_texture_rgba_at_level(state, host, task_id, texture_ref, 0, None)
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn load_type11_rgba<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    format_override: Option<u16>,
) -> Option<(u32, u32, Vec<u8>)> {
    let mapping_id = objects::resolve_type11_ref(state, host, task_id, texture_ref)?;
    load_type11_mapping_rgba(state, host, mapping_id, format_override)
}

/// Sample a type-11 mapping as tight RGBA8 from guest pages.
///
/// Guest pages ARE the surface content: the CPU writeback lands Stores in them
/// and guest CPU writes are immediately visible. There is exactly one source;
/// no recovery ranking exists.
///
/// The resolve runs *before* the geometry read, not after. A mapping can be
/// mapped with a live `MappingInternal` and no latched W×H yet; resolving first
/// decodes the guest device-surface descriptor and latches the geometry, so the
/// sample succeeds instead of bailing out on `!has_geom` and dropping the bind.
fn load_type11_mapping_rgba<M: HostMemory + HostOps>(
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

/// Type-2/3 linear texture at mip `level`: strided guest rows → tight RGBA8.
///
/// `format_override` is the type-8 view pixel format when present. Base storage
/// geometry (row_stride / level layout) stays on the base texture; the sample
/// format must be bpp-compatible with the base (Metal texture-view contract).
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn load_linear_texture_rgba_at_level<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    level: u32,
    format_override: Option<u16>,
) -> Option<(u32, u32, Vec<u8>)> {
    let entry = objects::lookup_list_entry(state, host, task_id, texture_ref)?;
    if entry.object_type != OBJECT_TYPE_TEXTURE && entry.object_type != OBJECT_TYPE_TEXTURE_VARIANT
    {
        return None;
    }
    let desc_bytes = objects::read_descriptor(state, host, task_id, &entry)?;
    let tex = decode_texture_descriptor(&desc_bytes).ok()?;
    if !tex.has_pixel_format {
        return None;
    }
    let base_fmt = tex.pixel_format;
    let sample_fmt = effective_view_sample_format(base_fmt, format_override)?;
    let (gva, layout) = tex.level_gva(level, state.page_shift)?;
    let w = layout.width;
    let h = layout.height;
    let bpr = layout.row_stride;
    if bpr > u32::MAX as u64 {
        return None;
    }
    let bpr_u32 = bpr as u32;
    // Row geometry follows the base texture's bpp (allocation layout).
    let tight = pixel_format::tight_row_bytes(w, base_fmt)?;
    if bpr_u32 < tight || w == 0 || h == 0 {
        return None;
    }
    let need_rgba = (w as u64)
        .checked_mul(h as u64)?
        .checked_mul(RGBA8_BPP as u64)?;
    let need_rgba = host_alloc_len(need_rgba)?;
    let span = bpr.checked_mul(h as u64)?;
    if tex.allocation_size != 0 && layout.offset.saturating_add(span) > tex.allocation_size {
        return None;
    }
    let mut rgba = vec![0u8; need_rgba];
    let mut row = vec![0u8; tight as usize];
    for y in 0..h {
        let row_gva = gva.checked_add((y as u64).checked_mul(bpr)?)?;
        gva_mem::read_task_gva_by_id(
            host,
            &state.tasks,
            task_id,
            row_gva,
            &mut row,
            state.page_shift,
        )
        .ok()?;
        let dst_off = (y as usize) * (w as usize) * 4;
        if !pixel_format::convert_row_to_rgba8(sample_fmt, &row, w, &mut rgba[dst_off..]) {
            return None;
        }
    }
    Some((w, h, rgba))
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn load_sampler<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    sampler_ref: u32,
    slot: u32,
) -> Result<crate::backend::metal::abi::ReimsVgpuSampler, MetalStateDecline> {
    use crate::backend::metal::abi::REIMS_VGPU_BINDING_SAMPLER_BASE;
    let entry = objects::lookup_list_entry(state, host, task_id, sampler_ref).ok_or(
        MetalStateDecline::SamplerEntryMissing {
            sampler_ref,
            index: slot,
        },
    )?;
    if entry.object_type != OBJECT_TYPE_TYPE7 {
        return Err(MetalStateDecline::SamplerObjectType {
            sampler_ref,
            index: slot,
            object_type: entry.object_type,
        });
    }
    let desc = objects::read_descriptor(state, host, task_id, &entry).ok_or(
        MetalStateDecline::SamplerDescriptorMissing {
            sampler_ref,
            index: slot,
        },
    )?;
    let s =
        decode_sampler_descriptor(&desc).map_err(|reason| MetalStateDecline::SamplerDecode {
            sampler_ref,
            index: slot,
            reason,
        })?;
    Ok(sampler_record(
        REIMS_VGPU_BINDING_SAMPLER_BASE + slot,
        &s,
        None,
        false,
    ))
}

#[cfg(feature = "backend-vulkan")]
fn vulkan_sampler_resource(
    sampler_ref: u32,
    binding: u32,
    sampler: &crate::runtime::decode::resource::SamplerDescriptor,
) -> Result<crate::backend::vulkan::engine::SamplerResource, DrawPreparationDecline> {
    use crate::backend::vulkan::engine::SamplerResource;

    Ok(SamplerResource {
        binding,
        min_filter: translate::sampler::filter(sampler.min_filter).map_err(|reason| {
            DrawPreparationDecline::SamplerMinFilterTranslation {
                sampler_ref,
                binding,
                reason,
            }
        })?,
        mag_filter: translate::sampler::filter(sampler.mag_filter).map_err(|reason| {
            DrawPreparationDecline::SamplerMagFilterTranslation {
                sampler_ref,
                binding,
                reason,
            }
        })?,
        mip_filter: translate::sampler::mip_filter(sampler.mip_filter).map_err(|reason| {
            DrawPreparationDecline::SamplerMipFilterTranslation {
                sampler_ref,
                binding,
                reason,
            }
        })?,
        address_mode_u: translate::sampler::address_mode(sampler.s_address).map_err(|reason| {
            DrawPreparationDecline::SamplerAddressSTranslation {
                sampler_ref,
                binding,
                reason,
            }
        })?,
        address_mode_v: translate::sampler::address_mode(sampler.t_address).map_err(|reason| {
            DrawPreparationDecline::SamplerAddressTTranslation {
                sampler_ref,
                binding,
                reason,
            }
        })?,
        address_mode_w: translate::sampler::address_mode(sampler.r_address).map_err(|reason| {
            DrawPreparationDecline::SamplerAddressRTranslation {
                sampler_ref,
                binding,
                reason,
            }
        })?,
        border_color: translate::sampler::border_color(sampler.border_color).map_err(|reason| {
            DrawPreparationDecline::SamplerBorderColorTranslation {
                sampler_ref,
                binding,
                reason,
            }
        })?,
        // Metal reuses `MTLCompareFunction` for depth, stencil and sampler
        // compare, so this is `raster`'s table rather than `sampler`'s — one
        // Metal enum, one home.
        compare_function: translate::raster::compare_function(sampler.compare_function).map_err(
            |reason| DrawPreparationDecline::SamplerCompareFunctionTranslation {
                sampler_ref,
                binding,
                reason,
            },
        )?,
        lod_min: sampler.lod_min_clamp.to_bits(),
        lod_max: sampler.lod_max_clamp.to_bits(),
        max_anisotropy: sampler.max_anisotropy.max(1),
        unnormalized_coordinates: !sampler.normalized_coordinates,
    })
}

#[cfg(feature = "backend-vulkan")]
pub fn reflected_static_sampler_resource(
    stage: &'static str,
    binding: u32,
    sampler: metal2vulkan::reflect::StaticSamplerState,
) -> Result<crate::backend::vulkan::engine::SamplerResource, DrawPreparationDecline> {
    use crate::backend::vulkan::engine::{
        SamplerAddressMode, SamplerBorderColor, SamplerCompareFunction, SamplerFilter,
        SamplerMipFilter, SamplerResource,
    };
    use metal2vulkan::reflect::{
        SamplerAddressMode as ReflectedAddress, SamplerBorderColor as ReflectedBorder,
        SamplerCompareFunction as ReflectedCompare, SamplerCoordinates,
        SamplerFilter as ReflectedFilter, SamplerMipFilter as ReflectedMip, SamplerReduction,
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

#[cfg(feature = "backend-vulkan")]
pub(crate) fn load_vulkan_sampler<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    sampler_ref: u32,
    binding: u32,
) -> Result<crate::backend::vulkan::engine::SamplerResource, DrawPreparationDecline> {
    let entry = objects::lookup_list_entry(state, host, task_id, sampler_ref).ok_or(
        DrawPreparationDecline::SamplerEntryMissing {
            sampler_ref,
            binding,
        },
    )?;
    if entry.object_type != OBJECT_TYPE_TYPE7 {
        return Err(DrawPreparationDecline::SamplerObjectType {
            sampler_ref,
            binding,
            object_type: entry.object_type,
        });
    }
    let desc = objects::read_descriptor(state, host, task_id, &entry).ok_or(
        DrawPreparationDecline::SamplerDescriptorMissing {
            sampler_ref,
            binding,
        },
    )?;
    let descriptor_len = desc.len();
    let tag = desc.get(..4).map(ld32);
    let declared_len = desc.get(4..8).map(ld32);
    let sampler = decode_sampler_descriptor(&desc).map_err(|status| match status {
        DecodeStatus::ErrShort(_) => DrawPreparationDecline::SamplerDescriptorShort {
            sampler_ref,
            binding,
            descriptor_len,
        },
        DecodeStatus::ErrUnknownType(_) => DrawPreparationDecline::SamplerDescriptorUnknownType {
            sampler_ref,
            binding,
            descriptor_len,
            tag,
        },
        DecodeStatus::ErrUnsupported(_) => DrawPreparationDecline::SamplerDescriptorUnsupported {
            sampler_ref,
            binding,
            descriptor_len,
            tag,
            declared_len,
        },
    })?;
    vulkan_sampler_resource(sampler_ref, binding, &sampler)
}

/// Metal-direct builds never arm GVA windows — nothing to supersede.
#[cfg(not(feature = "backend-vulkan"))]
pub(crate) fn supersede_gva_window<M: HostMemory + HostOps>(
    _state: &mut DeviceState,
    _host: &mut M,
    _gva: u64,
    _width: u32,
    _height: u32,
    _by: &str,
) {
}

/// Store encode RGBA8 into **texture_ref** host cache as BGRA (not surface_id).
#[cfg(test)]
fn host_cache_store_rgba8(
    state: &mut DeviceState,
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
    crate::runtime::surface_cache::store_texture(state, texture_ref, width, height, bgra);
}

/// Advance the guest-visible publish milestones for a type-11 Store whose
/// pixels have landed in the mapping's guest pages.
///
/// Route-independent: the synchronous `cpu_portability` Store calls it inline,
/// and the deferred render rail calls it from the flush that finally performs
/// the same write (`storage_flush::flush_render_one`). Both have just proved
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
    state.note_surface_composite(mapping_id);
    state.note_dense_frame_published(mapping_id, width, height);
    crate::runtime::scanout::note_front_buffer_writeback(
        state, host, mapping_id, width, height, format,
    );
}

pub fn writeback_chain_rgba<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    color_slots: &[(u32, crate::runtime::decode::render::ColorAttachment)],
    rgba: &[u8],
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
            "writeback_chain_rgba fail reason={why} task={task_id} slots={} bytes={} \
             (the abandoned chain's last frame is not landing; guest pages keep stale bytes)",
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
    let Some((mapping_id, gva, w, h, bpr, fmt)) =
        lookup_render_target(state, host, task_id, att.texture_ref)
    else {
        return lost("render_target_unresolved");
    };
    let need = (w as usize).saturating_mul(h as usize).saturating_mul(4);
    if rgba.len() < need {
        return lost("readback_short");
    }
    if gva != 0 {
        supersede_gva_window(state, host, gva, w, h, "chain_land");
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
    // the resident readback into the type-11 mapping, publish the Composite
    // Store, and keep the degradation fail-visible.
    crate::observe::fail(format!(
        "writeback_chain_rgba reason=resident_chain_abandoned_cpu_recovery \
         mid={mapping_id} {w}x{h} fmt={fmt:#x}"
    ));
    let wrote = mapping_write::write_rgba8_image_changed(state, host, mapping_id, rgba, None, w, h);
    if wrote {
        publish_surface_store(state, host, mapping_id, w, h, fmt);
    }
    wrote
}

/// Builds without a Metal encode path have no host ICB to execute.
#[cfg(feature = "backend-vulkan")]
pub fn encode_icb_execute_and_writeback<M: HostMemory + HostOps>(
    _state: &mut DeviceState,
    _host: &mut M,
    _req: &DrawEncodeRequest,
    _icb_ref: u32,
    _range_location: u64,
    _range_length: u64,
) -> EncodeStatus {
    EncodeStatus::NoMetal("icb_exec_no_metal_build")
}

/// The colour render target's base format for a **type-4** surface, or nothing.
///
/// On this arm `m.format == 0` is not "unset", it is a decoded refusal:
/// [`objects::apply_type4_backing`] is the only writer of it, and it stores 0 for
/// a multi-plane surface and for a single-plane one whose FourCC it does not
/// know, saying why — "stage/paint must not invent BGRA".
/// [`objects::iosurface_pixel_format_to_mtl`] repeats it twice more, and the
/// compute staging path honours it, declining with typed `multiplane` /
/// `fmt_unknown` reasons.
///
/// This resolve used to invent BGRA8 from it, so one surface was refused by the
/// compute path and rendered into as BGRA8 by this one. That is not a survivable
/// disagreement for a multi-plane surface: BGRA8 over a `'420f'` allocation
/// describes the wrong stride and the wrong bytes, and every downstream window is
/// built from what this returns.
///
/// It now declines. Refusing was held back because it drops a colour attachment
/// — a compositing layer going black — on a class nothing had counted, so the
/// class was counted first: `rt_base_fmt_invent` read **0 on two driven
/// x86/Vulkan boots** (Safari window drag plus the web-content probe). The arm is
/// unreached on this workload, so declining costs nothing measurable and stops
/// the device from silently rendering a format the guest did not declare. The
/// counter stays and the fail line stays, because "unreached on this workload" is
/// not "unreachable" — the first surface to take it will now be named and
/// refused rather than named and rendered wrong.
///
/// The **type-11** arm deliberately does not come through here. A type-11
/// mapping's format has other writers, so its 0 can mean "not latched yet" rather
/// than "refused", and BGRA8 is the display contract's stated default for that
/// case ([`crate::runtime::compute_exec`]'s `or_bgra8` writes the same rule down).
/// Those are different zeros and only this one is provably a refusal.
fn rt_type4_base_format(format: u16, mapping_id: u32) -> Option<u16> {
    if format != 0 {
        return Some(format);
    }
    crate::runtime::drain::note_store_route("rt_base_fmt_declined");
    if crate::observe::first_sight("rt_base_fmt_declined", mapping_id as u64) {
        crate::observe::fail(format!(
            "rt_base_fmt_declined mapping={mapping_id} \
             (the mapping's format is the type-4 decoder's multi-plane / \
             unknown-FourCC refusal, so this surface is not a single-format \
             colour attachment and no format is invented for it)"
        ));
    }
    None
}

/// Report a type-5 colour attachment whose view record disagrees with the base
/// mapping it is resolved through.
///
/// This resolve reads only `surfaceID@0` out of a type-5 descriptor and takes
/// geometry and format from the mapping. [`objects::decode_type5_texture_view`]'s
/// own contract forbids that — "callers must not replace it with base mapping
/// geometry merely because the surface itself is otherwise stageable" — and the
/// live case it names is real: the BGRA8 desktop target is also exposed as a
/// row-byte-equivalent quarter-width RGBA32Uint view. Every other type-5
/// consumer binds the view's own geometry.
///
/// It is harmless exactly while view == base, so the question is how often that
/// holds for a *render target* specifically, which nothing has measured.
/// `rt_type5_view_differs` against `rt_type5_view_same` answers it. Reported
/// rather than repaired: taking the view's geometry here changes what every
/// type-5 colour attachment renders into, and that is not a change to make on an
/// unmeasured population.
///
/// **Read on two driven x86/Vulkan boots: `same` 20 273 and 24 360, `differs`
/// 0, `undecoded` 0.** So on this workload every type-5 colour attachment's view
/// agrees with the base mapping in width, height and format, and resolving
/// through the base loses nothing. The reinterpretation view the contract names
/// is real traffic elsewhere — the compute staging path sees it — but it is not
/// bound as a render target here.
///
/// That is a reason not to change the resolve, and not a reason to stop asking.
/// `differs` is a healthy zero: the first non-zero line names a surface being
/// rendered at the wrong geometry, which no other counter in this path could
/// report.
fn note_rt_type5_view(
    view: Option<objects::Type5TextureView>,
    surface_id: u32,
    base: (u32, u32, u16),
) {
    let Some(view) = view else {
        crate::runtime::drain::note_store_route("rt_type5_view_undecoded");
        return;
    };
    let (base_w, base_h, base_fmt) = base;
    if view.width == base_w && view.height == base_h && view.pixel_format == base_fmt {
        crate::runtime::drain::note_store_route("rt_type5_view_same");
        return;
    }
    crate::runtime::drain::note_store_route("rt_type5_view_differs");
    if crate::observe::first_sight("rt_type5_view_differs", surface_id as u64) {
        crate::observe::fail(format!(
            "rt_type5_view_differs sid={surface_id} view={}x{} fmt={:#x} plane={} \
             base={base_w}x{base_h} fmt={base_fmt:#x} (the colour attachment is \
             resolved with the base mapping's geometry, not the view's)",
            view.width, view.height, view.pixel_format, view.plane_index
        ));
    }
}

/// Archive `apple_pv_gpu_lookup_render_target`: type-11 first, else type-2/3 GVA.
///
/// Wallpaper/background intermediates are type-2/3 guest-VA; type-11-only resolve
/// drops those passes (black wallpaper). Color RT formats are the Metal color-
/// renderable set admitted by [`pixel_format::render_target_bpp`] (RGBA8 family,
/// BGRA8 family, RGBA16Float) — bring-up only listed compositor BGRA8/0x73.
///
/// Type-8 texture views (archive `resource_resolve_texture` view chain): resolve
/// to the base texture. Swizzled views are rejected as RTs (archive
/// `resolve_texture` requires `!has_swizzle`). Level 0 only for color RT
/// materialization (mip RT not supported). Without this, UI passes that bind a
/// type-8 view as color attachment fail MRT (`mrt_request fail slots=[211]`) and
/// drop entire draws (blank App Store sidebar / missing chrome labels).
fn lookup_render_target<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &M,
    task_id: u32,
    texture_ref: u32,
) -> Option<(u32, u64, u32, u32, u32, u16)> {
    if texture_ref == 0 {
        return None;
    }
    // Type-8 view → base (archive resource_resolve_texture view chain).
    let (resolved_ref, view_fmt_override, view_level) =
        if let Some(view) = resolve_texture_view(state, host, task_id, texture_ref) {
            // Archive resolve_texture rejects swizzled views for linear resolve.
            if let Some(plan) = view.swizzle.as_ref() {
                if !pixel_format::swizzle_is_identity(plan) {
                    return None;
                }
            }
            (view.base_texture_ref, view.pixel_format, view.level)
        } else {
            (texture_ref, None, 0)
        };
    if resolved_ref == 0 {
        return None;
    }
    // Archive lookup order is by **live** object-list type + descriptor, not a
    // sticky cache: type-11 first, else type-2/3. Guest reuses object refs;
    // two failure modes for a stale `texture_to_mapping` latch:
    // 1) live type is now type-2/3 → must not force type-11 (live residual
    //    mrt color RT resolve fail ref=199 type=2 fmt=0x73 480x64).
    // 2) live type is still type-11 but descriptor mapping_id changed (or a
    //    recycled ref now names a different mid) → must re-read the live
    //    descriptor, not prefer the latch. Preferring latch routed dual-mid
    //    full-screen desktop composites onto only one mid (mid=3 nz=6M vs
    //    mid=4 stuck logo nz=1.97M; damage rects then preserved logo via Load).
    let live = objects::lookup_list_entry(state, host, task_id, resolved_ref);
    let live_type = live.as_ref().map(|e| e.object_type);
    if let Some(ot) = live_type {
        if ot != OBJECT_TYPE_IOSURFACE {
            // Live list says not type-11 — drop any recycled-ref latch.
            state.texture_to_mapping.remove(&(task_id, resolved_ref));
        }
    }
    let try_type11 = live_type == Some(OBJECT_TYPE_IOSURFACE)
        || (live_type.is_none()
            && state
                .texture_to_mapping
                .contains_key(&(task_id, resolved_ref)));
    if try_type11 {
        // Type-11 sample windows carry planes, not mip levels — a mip>0 view
        // of an IOSurface has no contract-backed layout; fail visibly.
        if view_level != 0 {
            return None;
        }
        // Live list is source of truth for mapping_id when the entry is type-11.
        // Latch is only a fallback when the list entry is transiently missing
        // (resolve_type11_ref refreshes the latch from the live descriptor).
        let mapping_id = if live_type == Some(OBJECT_TYPE_IOSURFACE) {
            objects::resolve_type11_ref(state, host, task_id, resolved_ref).or_else(|| {
                state
                    .texture_to_mapping
                    .get(&(task_id, resolved_ref))
                    .copied()
            })?
        } else {
            state
                .texture_to_mapping
                .get(&(task_id, resolved_ref))
                .copied()
                .or_else(|| objects::resolve_type11_ref(state, host, task_id, resolved_ref))?
        };
        let _ = mapper::ensure_resolved_for_scanout(state, host, mapping_id);
        if let Some(m) = state.mappings.get(&mapping_id) {
            if m.has_geom && m.width > 0 && m.height > 0 {
                // Not `rt_type4_base_format`: a type-11 mapping's format has
                // writers other than the type-4 decoder, so 0 here can mean "not
                // latched yet" rather than "refused", and BGRA8 is the display
                // contract's default for that case. See that function.
                let base_fmt = if m.format != 0 {
                    m.format
                } else {
                    MTL_FORMAT_BGRA8_UNORM
                };
                let fmt =
                    effective_view_sample_format(base_fmt, view_fmt_override).unwrap_or(base_fmt);
                if pixel_format::render_target_bpp(fmt).is_some() {
                    return Some((mapping_id, 0, m.width, m.height, 0, fmt));
                }
            }
        }
        // Live type-11 that failed geom: do not decode as type-2.
        if live_type == Some(OBJECT_TYPE_IOSURFACE) {
            return None;
        }
    }
    // x86 Ventura/Tahoe type-4 surface/backing (present IOSurface). Object-list
    // index == surface_id (ResourceHeap addObject type=4 objectId=getSurfaceID).
    // Without this, clear-only streams and Store writebacks never touch display
    // mids — guest pages stay empty and dual-mid thrash paints black.
    // Type-4: object-list index is surface_id. Type-5 RefTextureHandle: surfaceID@0
    // (allocateRefTextureHandle) — product color RTs are type-5 wrapping type-4.
    let mut type5_view: Option<objects::Type5TextureView> = None;
    let type4_sid = if live_type == Some(objects::OBJECT_TYPE_SURFACE) {
        Some(resolved_ref)
    } else if live_type == Some(objects::OBJECT_TYPE_REF_TEXTURE) {
        let entry = live.as_ref()?;
        let desc = objects::read_descriptor(state, host, task_id, entry)?;
        if desc.len() < objects::TYPE5_MIN_LEN {
            return None;
        }
        let sid = ld32(&desc[objects::TYPE5_SURFACE_ID..]);
        if sid == 0 {
            return None;
        }
        type5_view = objects::decode_type5_texture_view(&desc);
        Some(sid)
    } else {
        None
    };
    if let Some(surface_id) = type4_sid {
        if view_level != 0 {
            return None;
        }
        if !objects::resolve_type4_surface(state, host, surface_id) {
            crate::observe::fail(format!(
                "rt_resolve FAIL type4 tex_ref={resolved_ref} sid={surface_id} live_type={live_type:?}"
            ));
            return None;
        }
        let m = state.mappings.get(&surface_id)?;
        if !m.has_geom || m.width == 0 || m.height == 0 || m.page_entries.is_empty() {
            crate::observe::fail(format!(
                "rt_resolve FAIL type4_geom tex_ref={resolved_ref} sid={surface_id} has_geom={} pages={}",
                m.has_geom,
                m.page_entries.len()
            ));
            return None;
        }
        let (base_w, base_h, base_raw_fmt) = (m.width, m.height, m.format);
        if live_type == Some(objects::OBJECT_TYPE_REF_TEXTURE) {
            note_rt_type5_view(type5_view, surface_id, (base_w, base_h, base_raw_fmt));
        }
        let base_fmt = rt_type4_base_format(base_raw_fmt, surface_id)?;
        let fmt = effective_view_sample_format(base_fmt, view_fmt_override).unwrap_or(base_fmt);
        pixel_format::render_target_bpp(fmt)?;
        // mapping_id = surface_id; no linear GVA.
        return Some((surface_id, 0, m.width, m.height, 0, fmt));
    }
    // type-2/3 linear GVA (wallpaper/background layers, UI intermediate RTs).
    let entry = live?;
    if entry.object_type != OBJECT_TYPE_TEXTURE && entry.object_type != OBJECT_TYPE_TEXTURE_VARIANT
    {
        return None;
    }
    let desc_bytes = objects::read_descriptor(state, host, task_id, &entry)?;
    let tex = decode_texture_descriptor(&desc_bytes).ok()?;
    if !tex.has_pixel_format || !tex.has_width || !tex.has_height || !tex.has_row_stride {
        return None;
    }
    let base_fmt = tex.pixel_format;
    let fmt = effective_view_sample_format(base_fmt, view_fmt_override).unwrap_or(base_fmt);
    // Refuses a format with no known bytes-per-texel; the value is not needed.
    pixel_format::render_target_bpp(fmt)?;
    // Mip>0 view of a linear texture: the RT is that level's plane inside the
    // base allocation (archive collapses view mip into linear geometry —
    // compositor blur/backdrop pyramids render into successive levels).
    let (gva, w, h, bpr) = if view_level != 0 {
        let (level_gva, layout) = tex.level_gva(view_level, state.page_shift)?;
        if layout.row_stride > u32::MAX as u64 {
            return None;
        }
        // Full level span must fit the allocation (same check as the sample
        // path) — writing rows past it would corrupt adjacent guest memory.
        let span = layout.row_stride.checked_mul(layout.height as u64)?;
        if tex.allocation_size != 0 && layout.offset.saturating_add(span) > tex.allocation_size {
            return None;
        }
        (
            level_gva,
            layout.width,
            layout.height,
            layout.row_stride as u32,
        )
    } else {
        let (gva, alloc) = tex.backing_gva_size(state.page_shift)?;
        let span = (tex.row_stride as u64).checked_mul(tex.height as u64)?;
        let tight0 = pixel_format::tight_row_bytes(tex.width, fmt)?;
        // Exclusive last-row end (archive): height-1 * bpr + tight may fit
        // tighter allocs; accept if bpr*height fits allocation.
        if alloc > 0 && span > alloc {
            let alt = if tex.height > 0 {
                (tex.row_stride as u64)
                    .saturating_mul((tex.height - 1) as u64)
                    .saturating_add(tight0 as u64)
            } else {
                0
            };
            if alt > alloc {
                return None;
            }
        }
        (gva, tex.width, tex.height, tex.row_stride)
    };
    let tight = pixel_format::tight_row_bytes(w, fmt)?;
    if bpr < tight || w == 0 || h == 0 {
        return None;
    }
    Some((0, gva, w, h, bpr, fmt))
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
    color_texture_ref: u32,
    pipeline_ref: u32,
    vertex_count: u32,
    instance_count: u32,
    primitive_type: u32,
    first_vertex: u32,
    base_instance: u32,
) -> Option<DrawEncodeRequest> {
    let (mapping_id, gva, w, h, bpr, fmt) =
        lookup_render_target(state, host, task_id, color_texture_ref)?;
    let c0 = ColorRtRequest {
        slot: 0,
        texture_ref: color_texture_ref,
        mapping_id,
        target_gva: gva,
        row_stride: bpr,
        width: w,
        height: h,
        format: fmt,
        load_action: 0,
        store_action: PASS_STORE_ACTION_STORE,
        clear_color: [0.0; 4],
        target_seed_rgba: None,
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
    color_slots: &[(u32, crate::runtime::decode::render::ColorAttachment)],
    clears: &[crate::runtime::decode::render::ColorAttachment],
    vertex_count: u32,
    instance_count: u32,
    primitive_type: u32,
    first_vertex: u32,
    base_instance: u32,
) -> Option<DrawEncodeRequest> {
    if color_slots.is_empty() {
        return None;
    }
    let mut colors = Vec::new();
    let mut base_w = 0u32;
    let mut base_h = 0u32;
    for &(slot, att) in color_slots {
        if att.texture_ref == 0 {
            continue;
        }
        let Some((mapping_id, gva, mw, mh, bpr, mfmt)) =
            lookup_render_target(state, host, task_id, att.texture_ref)
        else {
            // One unresolvable color attachment drops the whole pass (Metal
            // would not form the encoder with a null RT). Fail-visible detail
            // (type-8 view/base, type-11 geom, linear GVA) for residual slots.
            crate::observe::fail(format!(
                "mrt color RT resolve fail task={task_id} pipe={pipeline_ref} slot={slot} ref={} {}",
                att.texture_ref,
                sample_miss_detail(state, host, task_id, att.texture_ref)
            ));
            return None;
        };
        if base_w == 0 {
            base_w = mw;
            base_h = mh;
        } else if mw != base_w || mh != base_h {
            // Metal MRT requires matching dimensions; skip mismatched extras.
            continue;
        }
        let mut load_action = att.load_action;
        let mut clear_color = att.clear_color;
        let mut seed = None;
        if let Some(cl) = clears.iter().find(|a| a.texture_ref == att.texture_ref) {
            // Clear-only stream record for this attachment: real Metal Clear.
            load_action = PASS_LOAD_ACTION_CLEAR;
            clear_color = cl.clear_color;
            if mapping_id == 0 {
                seed = Some(solid_rgba_local(mw, mh, &cl.clear_color));
            }
        } else if att.load_action == PASS_LOAD_ACTION_CLEAR {
            if mapping_id == 0 {
                seed = Some(solid_rgba_local(mw, mh, &att.clear_color));
            }
        } else if att.load_action == PASS_LOAD_ACTION_LOAD && mapping_id == 0 {
            // GVA linear target: ephemeral host RT needs a CPU seed (archive
            // reims_vgpu_backend_metal; NULL seed → Metal Clear invent, still encode).
            // Type-11 is seeded later instead, at the attachment site in
            // `encode_draw` — the same place the guest-backed alias used to be
            // built, and the same seed it already took whenever the alias was
            // refused. Seeding here would need the mapping read twice.
            //
            // A deferred GVA Store window at this exact target geometry means
            // the engine resident is the authoritative prior content — skip
            // the CPU seed; the encode-side Load resolves it (LoadFromTarget
            // or flush-then-cache). A geometry mismatch flushes inside
            // seed_color_load before its cache/guest reads.
            let deferred_resident = gva != 0
                && state
                    .gva_deferred_flush
                    .get(&gva)
                    .is_some_and(|e| e.width == mw && e.height == mh);
            if !deferred_resident {
                seed = seed_color_load(state, host, task_id, att.texture_ref, gva, mw, mh);
                if seed.is_none() {
                    crate::observe::fail(format!(
                        "color LOAD seed miss ref={} {}x{} fmt={:#x} gva={:#x} (archive: still encode)",
                        att.texture_ref, mw, mh, mfmt, gva
                    ));
                }
            }
        }
        colors.push(ColorRtRequest {
            slot,
            texture_ref: att.texture_ref,
            mapping_id,
            target_gva: gva,
            row_stride: bpr,
            width: mw,
            height: mh,
            format: mfmt,
            load_action,
            store_action: att.store_action,
            clear_color,
            target_seed_rgba: seed,
        });
    }
    if colors.is_empty() {
        return None;
    }
    Some(DrawEncodeRequest {
        task_id,
        pipeline_ref,
        vertex_count,
        instance_count,
        primitive_type,
        first_vertex,
        base_instance,
        colors,
        ..Default::default()
    })
}

/// Archive `apple_pv_gpu_write_gva_rgba`: tight RGBA8 → native rows at GVA.
/// Packed contig HostOps view when possible; else multi-import per row
/// ([`gva_view::write_span`]) — no `write_gpa` walk.
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
) -> Option<std::collections::HashSet<u64>> {
    if c.target_gva == 0
        || c.store_action != PASS_STORE_ACTION_STORE
        || c.width == 0
        || c.height == 0
    {
        return None;
    }
    let span = (c.row_stride as u64).checked_mul(c.height as u64)?;
    let pages = crate::runtime::gva_mem::task_gva_page_gpa_set(
        host,
        &state.tasks,
        task_id,
        c.target_gva,
        span,
        state.page_shift,
    );
    if pages.is_empty() {
        crate::runtime::drain::note_store_route("sync_store_unbounded");
        return None;
    }
    crate::runtime::drain::note_store_route("sync_store_bound");
    Some(pages)
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
/// This is what closes the gap `storage_flush::deferred_pages_still_ours`
/// leaves open. That guard walks, decides, and returns; the writer then walks
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
    let need = rgba_row.saturating_mul(height as usize);
    if rgba.len() < need {
        return Err(MemError::BadArgs);
    }
    let span = (height as u64).saturating_mul(bpr as u64);
    let mut row = vec![0u8; tight as usize];
    // Guest writes resolve through a fresh PT walk at write time — never a
    // cached view (stale-view heap-corruption class; see gva_view::write_span) —
    // and that walk carries `allowed`, so a deferred window cannot alias a page
    // outside itself even if the guest re-points the range mid-flush.
    if let Some(span_map) =
        crate::runtime::gva_view::map_fresh_span_within(state, host, task_id, gva, span, allowed)
    {
        // The payload census, sampled: the run of `0xff` bytes this device puts
        // into guest RAM is the predicate the panic census implies, and a rail
        // that skips this is missing from a count whose whole use is saying
        // whether white is ours. Read off the RGBA8 source rather than the
        // converted rows because that is where the whole image is in one piece;
        // the conversion is per-channel and full-scale maps to full-scale, so an
        // all-`0xff` source region is all-`0xff` in guest RAM for every format
        // here, and a source that is not white cannot become white.
        crate::observe::footprint::note_written_payload(rgba);
        let (base, avail) = (span_map.ptr, span_map.avail);
        let mut res = Ok(());
        for y in 0..height as usize {
            let src = &rgba[y * rgba_row..y * rgba_row + rgba_row];
            if !pixel_format::convert_rgba8_to_row(format, src, width, &mut row) {
                res = Err(MemError::BadArgs);
                break;
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
    // Fragmented GVA: multi-import each converted row via write_span.
    for y in 0..height as usize {
        let src = &rgba[y * rgba_row..y * rgba_row + rgba_row];
        if !pixel_format::convert_rgba8_to_row(format, src, width, &mut row) {
            return Err(MemError::BadArgs);
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

/// Store only the Metal scissor rect of a full-size tight RGBA8 buffer to GVA,
/// bounded to the pages the Store's target resolved to before the GPU ran.
/// Packed contig view when possible; else multi-import each rect row.
///
/// Only the Metal encode path issues a scissored guest store today, but nothing
/// here is Metal-specific: it is plain page-table walking over [`HostMemory`],
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
    reason = "the archive writer mirrors the target GVA, native row geometry and the scissor rect"
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
    origin_x: u32,
    origin_y: u32,
    rect_w: u32,
    rect_h: u32,
    allowed: crate::runtime::gva_view::WindowPages<'_>,
) -> bool {
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
        // The payload census, sampled: the run of `0xff` bytes this device puts
        // into guest RAM is the predicate the panic census implies, and a rail
        // that skips this is missing from a count whose whole use is saying
        // whether white is ours. Read off the RGBA8 source rather than the
        // converted rows because that is where the whole image is in one piece;
        // the conversion is per-channel and full-scale maps to full-scale, so an
        // all-`0xff` source region is all-`0xff` in guest RAM for every format
        // here, and a source that is not white cannot become white.
        crate::observe::footprint::note_written_payload(rgba);
        let (base, avail) = (span_map.ptr, span_map.avail);
        let mut ok = true;
        for dy in 0..rect_h as usize {
            let y = origin_y as usize + dy;
            let src_full = &rgba[y * rgba_row + (origin_x as usize) * 4
                ..y * rgba_row + (origin_x as usize) * 4 + (rect_w as usize) * 4];
            src_rgba.copy_from_slice(src_full);
            if !pixel_format::convert_rgba8_to_row(format, &src_rgba, rect_w, &mut row) {
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
        if !pixel_format::convert_rgba8_to_row(format, &src_rgba, rect_w, &mut row) {
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

/// Store scissor rect of tight RGBA8 into a type-11 mapping (BGRA host → guest fmt).
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
// Source geometry, destination geometry and the scissor rect, which are three
// independent rectangles; collapsing them into one struct would invite exactly
// the mix-up the separate names prevent.
#[allow(clippy::too_many_arguments)]
fn write_mapping_rgba8_rect<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    full_w: u32,
    full_h: u32,
    format: u16,
    rgba: &[u8],
    origin_x: u32,
    origin_y: u32,
    rect_w: u32,
    rect_h: u32,
) -> bool {
    if origin_x.saturating_add(rect_w) > full_w || origin_y.saturating_add(rect_h) > full_h {
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
    let tight = (rect_w as usize).saturating_mul(bpp as usize);
    let mut raw = vec![0u8; tight.saturating_mul(rect_h as usize)];
    let mut guest_row = vec![0u8; tight];
    for dy in 0..rect_h as usize {
        let y = origin_y as usize + dy;
        let src = &rgba[y * rgba_row + (origin_x as usize) * 4
            ..y * rgba_row + (origin_x as usize) * 4 + (rect_w as usize) * 4];
        // Guest store is native format; convert from tight RGBA8 (same as full write_gva path).
        if !pixel_format::convert_rgba8_to_row(format, src, rect_w, &mut guest_row) {
            return false;
        }
        raw[dy * tight..dy * tight + tight].copy_from_slice(&guest_row);
    }
    mapping_write::write_rect_raw(
        state,
        host,
        mapping_id,
        origin_x,
        origin_y,
        rect_w,
        rect_h,
        &raw,
        tight as u32,
    )
}

fn solid_rgba_local(w: u32, h: u32, clear: &[f64; 4]) -> Vec<u8> {
    use crate::contract::pixel_format::f64_to_unorm8;
    let r = f64_to_unorm8(clear[0]);
    let g = f64_to_unorm8(clear[1]);
    let b = f64_to_unorm8(clear[2]);
    let a = f64_to_unorm8(clear[3]);
    let px = [r, g, b, a];
    let n = (w as usize).saturating_mul(h as usize).saturating_mul(4);
    let mut img = vec![0u8; n];
    for i in 0..(w * h) as usize {
        img[i * 4..i * 4 + 4].copy_from_slice(&px);
    }
    img
}

/// Seed color RT LOAD from guest type-11 (BGRA→RGBA) or type-2/3/view linear RGBA.
///
/// Every color RT is an ephemeral host RT now, so every `Load` needs this: the
/// type-11 guest-memory alias that let Metal Load read the surface bytes in
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
    // A deferred GVA Store window here can only be a geometry mismatch (the
    // request-build path skips the seed for a matching window): land it so
    // the cache/guest reads below serve the Store's bytes, not stale ones.
    if target_gva != 0 && state.gva_deferred_flush.contains_key(&target_gva) {
        crate::runtime::storage_flush::flush_gva_exact(state, host, target_gva, true, "load_seed");
    }
    // Discrete GPU: exact target GVA is the strongest identity across object-ref
    // recycling. Fall back to the type-2/3 texture namespace, never the
    // unrelated type-4 surface_id namespace. Guest memory is last.
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
        // from here; the other 1 462 of `load_seed_ok` are type-11 and take
        // `resolve_type11_load_seed`.
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
        // Keep this asking. It is the denominator that makes the level census
        // interpretable, and its other three legs are pinned by
        // `surface_cache`'s unit tests, so a zero here is a measured zero and
        // not a probe that cannot tell the cases apart.
        if target_gva != 0
            && crate::runtime::surface_cache::has_gva(state, target_gva, width, height)
        {
            use crate::runtime::surface_cache::GvaBackingState as B;
            crate::runtime::drain::note_store_route(
                match crate::runtime::surface_cache::gva_backing_state(state, host, target_gva) {
                    B::Unrecorded => "gva_seed_backing_unrecorded",
                    B::Same => "gva_seed_backing_same",
                    B::Unmapped => "gva_seed_backing_unmapped",
                    B::Moved => "gva_seed_backing_moved",
                },
            );
        }
        let cached = if target_gva != 0 {
            crate::runtime::surface_cache::get_gva(state, target_gva, width, height)
        } else {
            None
        }
        .or_else(|| {
            (texture_ref != 0)
                .then(|| {
                    crate::runtime::surface_cache::get_texture(state, texture_ref, width, height)
                })
                .flatten()
        });
        if let Some(bgra) = cached {
            return Some(swap_rb_channels(bgra));
        }
    }
    // Type-2/3 (or type-8 base) linear GVA → convert to RGBA8.
    let rgba = load_sampled_rgba_static(state, host, task_id, texture_ref)?;
    Some(rgba)
}

/// Resolve sampled texture RGBA without requiring Metal feature (color LOAD seed path).
///
/// Type-8 views with a non-identity swizzle are rejected here: RT materialization does not
/// rematerialize through a remapped view (contract: swizzled views fail for RT/blit).
/// View `pixel_format` still overrides the base format when bpp-compatible.
fn load_sampled_rgba_static<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
) -> Option<Vec<u8>> {
    // Opcode-9 buffer-backed texture (type-8): sample the source buffer directly.
    if let Some(bt) = buffer_texture_descriptor(state, host, task_id, texture_ref, None) {
        return load_buffer_texture_rgba(state, host, task_id, texture_ref, &bt).map(|(_, _, r)| r);
    }
    // Type-11 path via resolve.
    if let Some(mid) = objects::resolve_type11_ref(state, host, task_id, texture_ref) {
        return load_type11_mapping_rgba(state, host, mid, None).map(|(_, _, r)| r);
    }
    // Type-8 view → base texture + mip + format. The view's SWIZZLE is
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
    // Type-11 base through a view (format override may reinterpret BGRA storage).
    if let Some(mid) = objects::resolve_type11_ref(state, host, task_id, tex_ref) {
        if level != 0 {
            return None;
        }
        return load_type11_mapping_rgba(state, host, mid, fmt_override).map(|(_, _, r)| r);
    }
    load_linear_texture_rgba_host(state, host, task_id, tex_ref, level, fmt_override)
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
#[cfg(feature = "backend-vulkan")]
fn prepare_memo_scratch(scratch: &mut Vec<u8>, span: usize, filled: usize) {
    let filled = filled.min(span);
    scratch.resize(span, 0);
    scratch[filled..].fill(0);
}

/// Byte-exact revalidated memo for the type-11 mapping-backed guest-page sampled
/// path. Same contract as [`load_linear_guest_memoized`] / the type-5 view memo:
/// re-read the native BGRA rect every bind (a guest CPU write is always
/// observed — neither `map_generation` nor `content_generation` tracks in-place
/// guest writes), memcmp against the memo, and on an unchanged hit return the
/// cached RGBA `Arc` + a namespaced content identity so BOTH the CPU convert/
/// alloc AND the engine's content hash + GPU upload are skipped. A dock-
/// magnification burst re-binds the same static icons ~1000×, so this collapses
/// the `t11_guest` CPU copies that saturate the serial drain worker (the
/// dock-hover whole-VM freeze). Returns `(rgba, identity)`.
#[cfg(feature = "backend-vulkan")]
fn load_type11_rgba_memoized<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mid: u32,
) -> Option<(std::sync::Arc<Vec<u8>>, LinearSampleIdentity)> {
    let (w, h) = {
        let m = state.mappings.get(&mid)?;
        if !m.has_geom || m.width == 0 || m.height == 0 {
            return None;
        }
        (m.width, m.height)
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
    let mut scratch = std::mem::take(&mut state.type11_memo_scratch);
    prepare_memo_scratch(
        &mut scratch,
        span,
        (stride as usize).saturating_mul(h as usize),
    );
    if !{
        crate::runtime::scanout::read_mapping_bgra8(state, host, mid, &mut scratch, stride, w, h)
    } {
        state.type11_memo_scratch = scratch;
        return None;
    }
    // Identity key namespace: bits 63+62 mark type-11 memo content, distinct from
    // raw-GVA keys (bit 63 clear) and type-5 view keys (bit 63 set, bit 62 clear).
    // Every producer draws its generation from
    // `DeviceState::next_sampled_content_generation`, so a (key, generation)
    // pair is unique device-wide and content can never alias on a collision.
    let identity_key = (1u64 << 63) | (1u64 << 62) | mid as u64;
    let key = (mid, w, h);
    if let Some(m) = state.type11_memo.get_touch(&key) {
        if m.native == scratch {
            let rgba = m.rgba.clone();
            let generation = m.generation;
            state.type11_memo_scratch = scratch;
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
        state.type11_memo_scratch = scratch;
        return None;
    }
    let rgba = std::sync::Arc::new(rgba);
    let generation = state.next_sampled_content_generation();
    let entry_bytes = scratch.len() + rgba.len();
    state.type11_memo.insert(
        key,
        crate::model::GuestLinearMemo {
            native: scratch,
            rgba: rgba.clone(),
            bgra8: false,
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

#[cfg(all(test, feature = "backend-vulkan"))]
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
            crate::model::DeviceState::new(crate::model::DeviceId(1), crate::model::PAGE_SHIFT_X86);
        let mut host = crate::runtime::host::FakeHost::new();

        // No source at all: the commonest way this rail is reached with nothing
        // to land, and previously the quietest.
        let n = store_route_count("chain_land_refused");
        assert!(!super::writeback_chain_rgba(
            &mut state,
            &mut host,
            1,
            &[],
            &[1u8; 4]
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
            &[1u8; 4]
        ));
        assert_eq!(store_route_count("chain_land_refused"), n + 1);
    }
}
