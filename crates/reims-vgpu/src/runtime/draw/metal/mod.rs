//! The Metal rail's draw encode: a [`DrawEncodeRequest`] into `MTLRenderCommandEncoder`.
//!
//! The sibling of [`super::vulkan`], and the two are deliberately shaped alike:
//! each owns one `encode_draw_chain` with the same signature, each translates
//! the backend-neutral request its parent built, and neither is reachable except
//! through [`crate::backend::Backend`]. Everything here names a Metal ABI type,
//! an `MTL*` ordinal or a host-side attachment buffer that only this rail keeps.
//!
//! This lived in [`super`] behind a `cfg` on every item. What that cost was not
//! readability: the Vulkan rail's items were re-exported *flat* into
//! `runtime::draw`, so both rails spelled their entry point
//! `runtime::draw::encode_draw_chain` and a build could only ever contain one of
//! them. Naming this rail is what lets a single binary carry both and run one
//! guest stream through each.

use super::*;

use crate::backend::metal::render::{RetainedColorTarget, RetainedColorTexture};
use crate::runtime::chain_phase;
use reims_vgpu_protocol::pass_action::MTL_STORE_ACTION_DONT_CARE;

// The Metal ICB execute half of this rail.
pub mod icb;
pub use icb::*;

// The host-side depth/stencil attachment buffers — this rail's own working
// parts, so they are not re-exported.
mod depth_stencil;
use depth_stencil::{seed_host_depth_stencil, DepthStencilAspect, HostAttachment};

/// This rail's retention decision for one colour attachment, made before the
/// seed is built and spent by the encode and then by the Store.
///
/// One value carries all three because the three questions are one: whether a
/// texture already exists, whether its pixels are this pass's prior content, and
/// which cache frame the Store must publish it as. Asked separately they drift —
/// the hazard is a pass that skips the seed on one answer and publishes against
/// another.
struct ResidentPlan {
    key: crate::backend::metal::resident::ResidentColorKey,
    /// The retained texture, when this rail still held one. `None` is the first
    /// draw into a surface, or one whose target the byte budget evicted.
    texture: Option<::metal::Texture>,
    /// Whether `texture`'s pixels already are this pass's prior content, so the
    /// LOAD seed is a copy of bytes the texture holds.
    ///
    /// Only ever true when [`crate::runtime::draw::published_surface_frame`]
    /// answered and the registry's generation matched it. Taking the texture
    /// retired that claim, so this is the one and only reading of it.
    holds_prior: bool,
    /// The cache generation this plan was made against, and the comparison the
    /// Store publishes on: a writeback that leaves this generation in place did
    /// not refresh the cache, so the target's new pixels correspond to no frame
    /// the cache holds and the claim must stay retired.
    ///
    /// `0` when there was no entry to compare against, which a refreshing
    /// writeback then differs from as well.
    asked_generation: u64,
}

/// Decide how this attachment's render target is obtained and retained.
///
/// Asked for every attachment with a mapping, whatever its load action: a Clear
/// pass renders into a target too, and retaining that target is what makes the
/// *next* pass's Load free.
///
/// An attachment [`crate::runtime::draw::published_surface_frame`] declines is
/// still retained — the texture is reused as an allocation and the seed is
/// uploaded into it as before. Only the content claim is refused, and the two
/// are separate answers for exactly that reason.
fn plan_resident_target<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    c: &ColorRtRequest,
    width: u32,
    height: u32,
) -> ResidentPlan {
    use crate::backend::metal::resident::{self, ResidentColorKey};
    use crate::runtime::draw::NoPublishedFrame;

    let key = ResidentColorKey::for_surface(c.mapping_id, width, height);
    let generation = match crate::runtime::draw::published_surface_frame(
        state,
        host,
        task_id,
        c.texture_ref,
        width,
        height,
    ) {
        Ok(frame) => {
            crate::runtime::drain::note_store_route("metal_resident_frame_current");
            frame.generation
        }
        Err(decline) => {
            crate::runtime::drain::note_store_route(match decline {
                NoPublishedFrame::NotMapped => "metal_resident_frame_unmapped",
                NoPublishedFrame::Uncurrent(..) => "metal_resident_frame_uncurrent",
                NoPublishedFrame::Unpublished(_) => "metal_resident_frame_unpublished",
            });
            0
        }
    };
    let taken = resident::take(&key, generation);
    let holds_prior = taken.as_ref().is_some_and(|(_, current)| *current);
    crate::runtime::drain::note_store_route(match (&taken, holds_prior) {
        (None, _) => "metal_resident_absent",
        (Some(_), true) => "metal_resident_holds_prior",
        (Some(_), false) => "metal_resident_allocation_only",
    });
    ResidentPlan {
        key,
        texture: taken.map(|(texture, _)| texture),
        holds_prior,
        asked_generation: generation,
    }
}

/// Give the retained target back its claim to be the surface, if — and only if
/// — this Store actually refreshed the cache with the pixels it now holds.
///
/// The test is the generation moving. Every writer of `host_surfaces` takes a
/// fresh one in the same breath as it changes the bytes, so a generation that
/// did not move is a writeback that published nothing this target could be said
/// to hold: the partial-store rail retires the entry outright rather than
/// rebuilding it, and a `DontCare` slot never reaches here at all. Leaving the
/// claim retired in those cases costs the next draw the upload it pays today.
fn publish_resident_target(state: &DeviceState, plan: &ResidentPlan) {
    let Some(generation) = crate::runtime::surface_cache::frame_generation(
        state,
        plan.key.mapping_id,
        plan.key.width,
        plan.key.height,
    ) else {
        crate::runtime::drain::note_store_route("metal_resident_store_unpublished");
        return;
    };
    if generation == plan.asked_generation {
        crate::runtime::drain::note_store_route("metal_resident_store_stale");
        return;
    }
    crate::runtime::drain::note_store_route("metal_resident_store_published");
    crate::backend::metal::resident::published(&plan.key, generation);
}

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

/// Encode one draw; optionally store to guest. Returns color0 tight RGBA8 for
/// multi-draw chaining (archive DrawJob threads output → next initial content).
///
/// `force_full_store`: when true, ignore scissor-local store even if Load+partial
/// scissor (required for multi-draw final writeback after in-process chaining).
///
/// Takes `&mut req` so multi-MiB Load seeds can be **moved** into the encoder
/// (no extra full-frame clone on the multi-draw chain).
pub fn encode_draw_chain<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    req: &mut DrawEncodeRequest,
    writeback_guest: bool,
    force_full_store: bool,
) -> (EncodeStatus, Option<Vec<u8>>) {
    encode_draw_chain_inner(state, host, req, writeback_guest, force_full_store)
}

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
    use crate::backend::metal::render::{render_core_mrt, ColorRt, VisibilityQuery};
    use crate::backend::metal::util::ErrOut;

    // Opened before the first refusal check, so a chain that declines is charged
    // to whichever phase was open rather than vanishing from the division. See
    // `chain_phase`'s doc on early returns.
    let _phase = chain_phase::ChainTimer::start();
    chain_phase::enter(chain_phase::Phase::Prep);
    if req.colors.is_empty() {
        return (EncodeStatus::BadArgs("draw_mtl_no_color_target"), None);
    }
    if let Some(color) = req
        .colors
        .iter()
        .find(|color| color.multisample_source_ref != 0)
    {
        crate::observe::fail(format!(
            "metal_draw reason=draw_mtl_multisample_resolve_unsupported pipe={} \
             source={} resolve={} store_action={}",
            req.pipeline_ref, color.multisample_source_ref, color.texture_ref, color.store_action
        ));
        return (
            EncodeStatus::BadArgs("draw_mtl_multisample_resolve_unsupported"),
            None,
        );
    }
    // Before anything is resolved or staged: a bind naming a slot past its
    // argument table refuses the draw, once, for all three classes and both
    // stages. Metal answers an out-of-range argument-table index with a
    // process-aborting exception, so this is the one place the encode may not
    // continue past. Every consumer below therefore takes the slot as in-range.
    if let Some(bind) = first_bind_past_table(req) {
        crate::observe::fail(format!(
            "metal_draw reason=draw_mtl_bind_slot_past_table pipe={} class={} stage={} \
             index={} table={} ref={}",
            req.pipeline_ref,
            bind.class.name(),
            bind.stage_name(),
            bind.index,
            bind.class.table(),
            bind.resource_ref
        ));
        return (EncodeStatus::BadArgs("draw_mtl_bind_slot_past_table"), None);
    }
    // Move multi-MiB Load seeds out **before** cloning color metadata so multi-draw
    // chain frames are not duplicated (clone of empty Option is cheap).
    let mut color_seeds: Vec<Option<Vec<u8>>> = req
        .colors
        .iter_mut()
        .map(|c| {
            c.target_seed
                .take()
                .filter(|seed| seed.layout == crate::protocol::pixel_format::TexelLayout::Rgba8)
                .map(|seed| seed.bytes)
        })
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
    chain_phase::enter(chain_phase::Phase::PrepPages);
    let sync_store_pages: Vec<Option<StoreTargetPages>> = if writeback_guest {
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

    chain_phase::enter(chain_phase::Phase::PipelineDesc);
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
    // `load_render_pipeline` declared it; this rail is about to turn the
    // guest's shader form into the host's. Unlike the Vulkan rail this one
    // retains no pipeline state, so it walks the same three steps on every
    // draw and the second draw's `advance` declines — counted, not ignored.
    crate::runtime::draw::advance_pipeline(
        state,
        host,
        req.task_id,
        req.pipeline_ref,
        reims_vgpu_core::pipeline::PipelineState::Translating,
    );
    chain_phase::enter(chain_phase::Phase::PipelineMtlb);
    let Some(vert) = load_mtlb(
        state,
        host,
        req.task_id,
        pipeline.vertex_func_ref,
        AirLoadRail::Draw,
    ) else {
        crate::observe::fail(format!(
            "metal_draw MissingMtlb vert_func={} pipe={}",
            pipeline.vertex_func_ref, req.pipeline_ref
        ));
        crate::runtime::draw::refuse_pipeline(
            state,
            host,
            req.task_id,
            req.pipeline_ref,
            reims_vgpu_core::pipeline::RefusalReason::CompilationFailed("vertex_mtlb_missing"),
        );
        return (EncodeStatus::MissingMtlb("draw_mtl_vertex_mtlb_load"), None);
    };
    let Some(frag) = load_mtlb(
        state,
        host,
        req.task_id,
        pipeline.fragment_func_ref,
        AirLoadRail::Draw,
    ) else {
        crate::observe::fail(format!(
            "metal_draw MissingMtlb frag_func={} pipe={}",
            pipeline.fragment_func_ref, req.pipeline_ref
        ));
        crate::runtime::draw::refuse_pipeline(
            state,
            host,
            req.task_id,
            req.pipeline_ref,
            reims_vgpu_core::pipeline::RefusalReason::CompilationFailed("fragment_mtlb_missing"),
        );
        return (
            EncodeStatus::MissingMtlb("draw_mtl_fragment_mtlb_load"),
            None,
        );
    };
    // Both stages loaded. On this rail there is no further step that is about
    // the *pipeline* rather than about one draw's resources — the MTLBs go to
    // the shim, which builds the pipeline state as part of encoding — so the
    // pipeline becomes usable here. Deferring `Ready` to the encode's own
    // result would let one unrelated texture miss park every exec that leased
    // this pipeline forever.
    for step in [
        reims_vgpu_core::pipeline::PipelineState::Compiling,
        reims_vgpu_core::pipeline::PipelineState::Ready,
    ] {
        crate::runtime::draw::advance_pipeline(state, host, req.task_id, req.pipeline_ref, step);
    }

    // Materialize buffer backs (storage first, then ReimsVgpuBuffer views).
    // Archive apple-pv-gpu-exec: a non-zero bound buffer that does not resolve
    // sets all_binds_ok=false and gates the draw (never feeds garbage geometry).
    chain_phase::enter(chain_phase::Phase::Binds);
    let mut vtx_storage: Vec<Vec<u8>> = Vec::new();
    let mut frag_storage: Vec<Vec<u8>> = Vec::new();
    let mut vtx_bind_idx: Vec<u32> = Vec::new();
    let mut frag_bind_idx: Vec<u32> = Vec::new();
    for b in req.vertex_buffers.iter() {
        if b.buffer_ref == 0 {
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
    for b in req.fragment_buffers.iter() {
        if b.buffer_ref == 0 {
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

    // Stage-in attrs: layout always comes from the serializer-object pipeline vertex
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
            // A plain vertex descriptor's layouts default to `PerVertex`; the
            // post-tessellation default belongs to the ICB path, which names it
            // itself.
            step_function: a
                .step_function_ordinal(::metal::MTLVertexStepFunction::PerVertex as u32),
            step_rate: a.step_rate(),
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
        // The bind's own stride, where the record carried one. The ABI has
        // always had these two fields — the compute rail fills them and
        // `raw_metal::set_buffer_with_attribute_stride` reads them — and the
        // render path wrote zeros into them because nothing above it carried a
        // stride to write.
        // The `Option` itself, not `bind_attribute_stride`. That function
        // answers "which stride is in force", which needs a pipeline stride to
        // fall back to and this rail has none to hand — and collapsing the
        // absent case onto a zero would lose `Some(0)`, a legal Metal request
        // that fetches every vertex from one address. `has_attribute_stride`
        // is precisely the `is_some`, which is why the ABI carries both fields.
        if let Some(stride) = req
            .vertex_buffers
            .iter()
            .find(|b| b.index == binding)
            .and_then(|b| b.attribute_stride)
        {
            ab.attribute_stride = stride;
            ab.has_attribute_stride = 1;
        }
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

    // Sampled textures: mapper-ref-texture mapping pages, then normal-texture linear GVA.
    struct TexItem {
        index: u32,
        w: u32,
        h: u32,
        rgba: Vec<u8>,
    }
    // Archive apple-pv-gpu-exec: a bound texture that does not resolve gates the
    // draw (never samples black/garbage). Same for vertex-stage textures.
    chain_phase::enter(chain_phase::Phase::Sampled);
    // `sampled_us` is this rail's largest bar and, unlike the Vulkan rail's, has
    // never been divided — `runtime::sampled_phase` splits the other rail's.
    // Two spans and two magnitudes, because a bar this size has two candidate
    // shapes and they have opposite fixes: many small binds (per-bind overhead)
    // and few large ones (byte movement).
    let mut vtx_tex_items: Vec<TexItem> = Vec::new();
    let mut frag_tex_items: Vec<TexItem> = Vec::new();
    let span_sampled = chain_phase::CostSpan::new("metal_sampled_load_us");
    for t in req.vertex_textures.iter() {
        if t.texture_ref == 0 {
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
        crate::runtime::drain::note_store_route("metal_sampled_binds");
        crate::runtime::drain::note_store_route_n("metal_sampled_bytes", rgba.len() as u64);
        vtx_tex_items.push(TexItem {
            index: t.index,
            w,
            h,
            rgba,
        });
    }
    for t in req.fragment_textures.iter() {
        if t.texture_ref == 0 {
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
        crate::runtime::drain::note_store_route("metal_sampled_binds");
        crate::runtime::drain::note_store_route_n("metal_sampled_bytes", rgba.len() as u64);
        frag_tex_items.push(TexItem {
            index: t.index,
            w,
            h,
            rgba,
        });
    }
    drop(span_sampled);
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

    // Samplers: serializer-object subtype 0x03 when present. A nonzero ref is an explicit
    // guest bind; if it cannot be resolved, keep the correct fallback but make
    // the degradation visible with the exact resolver reason.
    let span_samplers = chain_phase::CostSpan::new("metal_sampled_smp_us");
    let mut vtx_samps: Vec<ReimsVgpuSampler> = Vec::new();
    let mut frag_samps: Vec<ReimsVgpuSampler> = Vec::new();
    for s in req.vertex_samplers.iter() {
        if s.sampler_ref != 0 {
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
            vtx_samps.push(with_bind_lod_clamp(sampler, s.lod_clamp));
        }
    }
    for s in req.fragment_samplers.iter() {
        if s.sampler_ref != 0 {
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
            frag_samps.push(with_bind_lod_clamp(sampler, s.lod_clamp));
        }
    }
    drop(span_samplers);

    // Both lists were built exactly one entry long from an `Option`, while the
    // backend ABI beneath them has always taken a slice and `apply_viewports`
    // has always called `setViewports:count:`. The only thing bounded to one
    // was the field above; the count these carry is now the guest's own, and
    // the backend refuses a count past `REIMS_VGPU_BACKEND_MAX_VIEWPORTS`
    // rather than truncating it.
    chain_phase::enter(chain_phase::Phase::Assemble);
    let viewports: Vec<ReimsVgpuViewport> = req
        .viewports
        .iter()
        .map(|v| ReimsVgpuViewport {
            x: v[0] as f32,
            y: v[1] as f32,
            width: v[2] as f32,
            height: v[3] as f32,
            znear: v[4] as f32,
            zfar: v[5] as f32,
        })
        .collect();
    let scissors: Vec<ReimsVgpuScissor> = req
        .scissors
        .iter()
        .map(|r| ReimsVgpuScissor {
            x: r.x,
            y: r.y,
            width: r.width,
            height: r.height,
        })
        .collect();

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
        has_front_facing_winding: 0,
        front_facing_winding: 0,
        has_fill_mode: 0,
        fill_mode: 0,
        has_depth_clip_mode: 0,
        depth_clip_mode: 0,
    };
    if let Some(c) = req.cull_mode {
        raster.has_cull_mode = 1;
        raster.cull_mode = c;
    }
    if let Some(f) = req.front_facing {
        raster.has_front_facing_winding = 1;
        raster.front_facing_winding = f;
    }
    if let Some(f) = req.fill_mode {
        raster.has_fill_mode = 1;
        raster.fill_mode = f;
    }
    if let Some(d) = req.depth_clip_mode {
        raster.has_depth_clip_mode = 1;
        raster.depth_clip_mode = d;
    }
    // A record is worth encoding when the stream bound any one of the four.
    // Spelled as a method on the struct rather than as an `||` chain here,
    // because a field added to the struct and not to the chain is a state the
    // guest set and this arm silently declines to send.
    let raster_opt = if raster.any_bound() {
        Some(&raster)
    } else {
        None
    };

    // The fifth encoder raster state, and the one this rail cannot send.
    // `setLineWidth:` is on the wire and `MTLRenderCommandEncoder` has no
    // public setter for it, so a width other than the default is guest work
    // this arm drops — and it says so by name rather than by a counter,
    // because a number cannot say which pipeline drew a hairline where the
    // guest asked for a thick line. Once per `(pipeline, slug)`: the guest
    // sets a width per encoder and draws many times under it.
    //
    // Only for a width that differs from Metal's own default. `None` is a
    // stream that set none and 1.0 is a stream that set the default, and
    // neither loses anything — reporting them would flood a log with draws
    // that rasterize exactly as asked.
    if let Some(width) = req.line_width {
        if width != 1.0 && degrade_log_first(req.pipeline_ref, "metal_line_width_unsupported") {
            crate::observe::fail(format!(
                "metal_line_width_unsupported pipe={} width={width} \
                 (MTLRenderCommandEncoder has no public line-width setter; \
                 lines rasterize at 1.0)",
                req.pipeline_ref
            ));
        }
    }

    let depth_bias_state = req.depth_bias.map(|d| ReimsVgpuDepthBiasState {
        depth_bias: d[0],
        slope_scale: d[1],
        clamp: d[2],
    });
    let depth_bias_opt = depth_bias_state.as_ref();

    // Serializer-object depth-stencil object + optional stencil reference.
    chain_phase::enter(chain_phase::Phase::AssembleDepth);
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
    chain_phase::enter(chain_phase::Phase::Assemble);
    let mut depth_attach_api: Option<ReimsVgpuDepthAttachment> = None;
    let depth_storage = req.depth_attach.as_ref().and_then(|da| {
        let mut seeded = seed_host_depth_stencil(
            state,
            host,
            req,
            DepthStencilAspect::Depth {
                clear: da.clear_depth,
            },
            HostAttachment::from(*da),
            (width, height),
        )?;
        depth_attach_api = Some(ReimsVgpuDepthAttachment {
            pixel_format: REIMS_VGPU_MTL_PIXEL_FORMAT_DEPTH32_FLOAT,
            load_action: map_load_action(req.pipeline_ref, da.load_action),
            store_action: map_store_action(req.pipeline_ref, da.store_action),
            clear_depth: da.clear_depth,
            data: seeded.data.as_mut_ptr(),
            len: seeded.data.len(),
        });
        Some(seeded)
    });

    let mut stencil_attach_api: Option<ReimsVgpuStencilAttachment> = None;
    let stencil_storage = req.stencil_attach.as_ref().and_then(|sa| {
        let mut seeded = seed_host_depth_stencil(
            state,
            host,
            req,
            DepthStencilAspect::Stencil {
                clear: sa.clear_stencil,
            },
            HostAttachment::from(*sa),
            (width, height),
        )?;
        stencil_attach_api = Some(ReimsVgpuStencilAttachment {
            pixel_format: REIMS_VGPU_MTL_PIXEL_FORMAT_STENCIL8,
            load_action: map_load_action(req.pipeline_ref, sa.load_action),
            store_action: map_store_action(req.pipeline_ref, sa.store_action),
            clear_stencil: sa.clear_stencil,
            data: seeded.data.as_mut_ptr(),
            len: seeded.data.len(),
        });
        Some(seeded)
    });

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
    chain_phase::enter(chain_phase::Phase::Seed);
    // The two halves of this rail's `seed_us`, which a driven macos-13 boot put
    // at 4.79 ms a draw — the largest bar on the rail. They are counters beside
    // the bar rather than a finer cut of it, so the bar keeps comparing across
    // boots; see [`chain_phase::CostSpan`].
    let mut color_outs: Vec<Vec<u8>> = {
        let _outs = chain_phase::CostSpan::new("metal_seed_outs_us");
        (0..color_list.len()).map(|_| vec![0u8; need]).collect()
    };

    // For indexed draws, pass index_count as vertex_count for the early gate.
    let vertex_count = if is_indexed {
        req.indexed.as_ref().map(|i| i.index_count).unwrap_or(0)
    } else {
        req.vertex_count
    };

    // Mapper-ref-texture color targets render into a host RT and are written back by the
    // CPU. The guest-backed attachment that used to sit here aliased the
    // mapping's `mach_vm_remap` view with `newBufferWithBytesNoCopy`, so Load
    // read and Store wrote guest pages in place; that is exactly the access the
    // host GPU must not have, and the alias is gone. What runs now is the same
    // seed-and-write-back path the alias already fell through to on every
    // contract refusal (unaligned offset or row stride, span out of range, no
    // device), so this is a rung the rail has always had.
    //
    // The seed is skipped entirely for an attachment this rail already holds a
    // render target for whose pixels are the frame the cache would have handed
    // over. See [`crate::backend::metal::resident`]: that is not a new claim
    // about content, it is "do not copy bytes into a texture that holds them".
    let mut resident_plan: Vec<Option<ResidentPlan>> =
        (0..color_list.len()).map(|_| None).collect();
    {
        let _seed_span = chain_phase::CostSpan::new("metal_seed_load_us");
        for (i, c) in color_list.iter().enumerate() {
            if c.mapping_id == 0 {
                continue;
            }
            // Asked for every attachment with a mapping, not only the ones that
            // Load: a Clear pass renders into a target too, and retaining that
            // target is what makes the *next* pass's Load free. The generation
            // is read whatever the load action, because it is also what the
            // Store publishes against.
            resident_plan[i] = Some(plan_resident_target(
                state,
                host,
                req.task_id,
                c,
                width,
                height,
            ));
            if c.load_action != MTL_LOAD_ACTION_LOAD || color_seeds[i].is_some() {
                continue;
            }
            if resident_plan[i]
                .as_ref()
                .is_some_and(|plan| plan.holds_prior)
            {
                crate::runtime::drain::note_store_route("metal_seed_from_resident");
                continue;
            }
            crate::runtime::drain::note_store_route("metal_seed_load_asked");
            color_seeds[i] = seed_color_load(
                state,
                host,
                req.task_id,
                c.texture_ref,
                0,
                width,
                height,
                // The Metal arm's seed consumer reads the bytes as RGBA8.
                crate::runtime::draw::texture_view::NativeUploads::NONE,
            );
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
    chain_phase::enter(chain_phase::Phase::Assemble);
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
        // What this rail is about to do to the guest's declaration, counted
        // before it does it. `pixel_format: 0` below is `RGBA8Unorm` for every
        // target whatever the destination declared, and until this call there
        // was nothing on the always-on channel to say when that quantised a
        // wider one. See `runtime::draw::ColorTargetNarrowing`.
        crate::runtime::draw::note_store_narrowing(c.format, width, height);
        color_rts.push(ColorRt {
            slot: c.slot,
            // Host RT: 0 = RGBA8Unorm (writeback conversion path). The one
            // place this rail decides a colour target's format; the narrowing
            // it can cause is named and counted directly above.
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
            // Moved out of the plan rather than borrowed: the backend takes
            // ownership of the handle for the pass, and a plan left behind
            // holding a second one would keep an evicted texture alive past the
            // registry that stopped counting its bytes.
            // The handle moves and the plan stays: the Store below needs the
            // key and the generation this plan was made against, and a second
            // live handle here would keep an evicted texture alive past the
            // registry that stopped counting its bytes.
            retained: resident_plan[i].as_mut().map(|plan| RetainedColorTarget {
                key: plan.key,
                texture: match (plan.texture.take(), plan.holds_prior) {
                    (None, _) => RetainedColorTexture::Absent,
                    (Some(texture), true) => RetainedColorTexture::Prior(texture),
                    (Some(texture), false) => RetainedColorTexture::Allocation(texture),
                },
            }),
        });
    }

    let mut err_buf = [0i8; 256];
    let err: ErrOut<'_> = (err_buf.as_mut_ptr(), err_buf.len());
    // The guest's offset stays here: the backend answers one draw at a time and
    // `runtime::exec` sums the answers per offset, which is the same split the
    // Vulkan rail takes for the same reason.
    let mut visibility = req.visibility.map(|arming| VisibilityQuery {
        mode: arming.mode,
        samples: None,
    });
    chain_phase::enter(chain_phase::Phase::Engine);
    let st = render_core_mrt(
        &vert,
        &frag,
        width,
        height,
        crate::protocol::draw::DrawArgs {
            vertex_count,
            instance_count: req.instance_count,
            primitive_type: req.primitive_type,
            first_vertex: req.first_vertex,
            base_instance: req.base_instance,
        },
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
        visibility.as_mut(),
        err,
    );
    chain_phase::enter(chain_phase::Phase::Store);
    // Read before the status is matched, the way `runtime::exec` reads the
    // field it lands in: the backend only fills `samples` on a pass that ran to
    // completion, so a refusal leaves the query unanswered and says so.
    if let Some(query) = visibility.as_ref() {
        req.visibility_samples = query.samples;
    }
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
        return (EncodeStatus::RailRefused(st), None);
    }

    // Convert each color RT RGBA8 → guest format and writeback (mapper-ref-texture mapping
    // or normal-texture GVA — archive write_mapper_ref_texture_rgba / write_gva_rgba).
    // Multi-draw intermediate records skip guest store (archive one writeback).
    let mut any_write = false;
    if !writeback_guest {
        // Still log + early paint latch only when storing; chain returns RGBA.
        // Moved out rather than cloned: `color_outs` is this call's own storage
        // and dies with the frame, so a clone was a second full-frame
        // allocation and copy for a buffer nothing else would read.
        return (
            EncodeStatus::Ok,
            std::mem::take(&mut color_outs).into_iter().next(),
        );
    }
    for (i, c) in color_list.iter().enumerate() {
        if c.store_action == MTL_STORE_ACTION_DONT_CARE {
            continue;
        }
        let out_rgba = &color_outs[i];
        // normal-texture GVA keeps archive image_changed via store_seed_policy.
        let load_seed = color_seeds.get(i).and_then(|s| s.as_deref());
        let seed_for_store = store_seed_policy(force_full_store, c.load_action, load_seed);
        // The same coverage question the draw census asks, from the same
        // helper: a scissor that reaches every texel is not a partial store.
        //
        // Exactly one rect, because this writes back *only* that rect and the
        // rest of the attachment keeps its seed. With several rects the union
        // is what the draw could have written, and storing the first alone
        // would drop every texel the others covered — pixels the guest drew and
        // this device then failed to publish, which is worse than storing more
        // than was needed. So a multi-rect draw takes the full store, and the
        // narrowing stays available for the single-rect case that has always
        // used it.
        let store_rect = match req.scissors.as_slice() {
            [r] if !r.covers(width, height) => Some(*r),
            _ => None,
        };
        let gva_partial = seed_for_store.is_some() && store_rect.is_some();
        let wrote = if c.mapping_id != 0 {
            if gva_partial {
                let r = store_rect.expect("gva_partial implies exactly one narrowing rect");
                write_mapping_rgba8_rect(
                    state,
                    host,
                    c.mapping_id,
                    width,
                    height,
                    c.format,
                    out_rgba,
                    mapping_write::Rect {
                        origin_x: r.x,
                        origin_y: r.y,
                        width: r.width,
                        height: r.height,
                    },
                )
            } else {
                // The host copy of this frame is worth its 12.6 ms a flush
                // only when nothing else holds the frame. This rail's own
                // resident render target holds exactly `out_rgba` — it is the
                // texture the pass rendered into and the readback came out of —
                // and `publish_resident_target` below stamps it as this
                // mapping's published frame the moment the write lands. So when
                // there is a plan, the copy is a second one and the two readers
                // that want it take
                // `backend::metal::resident::read_published_bgra8` instead.
                //
                // Keyed on the plan and not on `holds_prior`: `holds_prior` is
                // about the *prior* frame and is false on the very draw that
                // creates the target, which is precisely a draw whose Store does
                // publish a resident.
                mapping_write::write_rgba8_image_changed(
                    state,
                    host,
                    c.mapping_id,
                    out_rgba,
                    seed_for_store,
                    width,
                    height,
                    if resident_plan.get(i).is_some_and(Option::is_some) {
                        mapping_write::FramePublication::RailResident
                    } else {
                        mapping_write::FramePublication::HostCache
                    },
                )
            }
        } else if c.target_gva != 0 {
            let allowed = sync_store_pages
                .get(i)
                .and_then(|p| p.as_ref())
                .map(StoreTargetPages::membership);
            if gva_partial {
                let r = store_rect.expect("gva_partial implies exactly one narrowing rect");
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
                    mapping_write::Rect {
                        origin_x: r.x,
                        origin_y: r.y,
                        width: r.width,
                        height: r.height,
                    },
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
            if let Some(plan) = resident_plan.get(i).and_then(|p| p.as_ref()) {
                publish_resident_target(state, plan);
            }
            // Early-boot logo+pill: paint mapper-ref-texture front before first DisplaySwap.
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

    // Optional depth/stencil store writeback into mapper-ref-texture mappings.
    for seeded in [depth_storage.as_ref(), stencil_storage.as_ref()]
        .into_iter()
        .flatten()
    {
        seeded.store_back(state, host, (width, height));
    }
    // Moved, not cloned; see the early return above.
    let color0_rgba = std::mem::take(&mut color_outs).into_iter().next();
    (EncodeStatus::Ok, color0_rgba)
}

/// normal-texture linear GVA raw image read (tight dst rows of `row_bytes`).
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
    let Ok((_entry, desc_bytes)) = objects::resolve_descriptor(
        state,
        host,
        task_id,
        texture_ref,
        &[OBJECT_TYPE_TEXTURE, OBJECT_TYPE_TEXTURE_GENERATE_MIPMAPS],
    ) else {
        return false;
    };
    let Ok(tex) = decode_texture_descriptor(&desc_bytes) else {
        return false;
    };
    let stride_covers_row = tex
        .declared_row_stride()
        .is_some_and(|stride| stride >= row_bytes);
    if tex.extent() != Some((width, height)) || !stride_covers_row {
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

/// Decoded `MTLLoadAction` → the Metal C ABI value.
///
/// This maps nothing. `protocol::pass_action` and `backend::metal::abi` declare
/// the same three ordinals, in `u16` and `u32`, and `const` assertions in the
/// mirror pin them equal — so every arm below is a widening. It reads as a
/// translation table because it had to be one: until those two declarations
/// were related, this `match` was the only thing in the tree claiming they
/// agreed, and it claimed it on this arm alone.
///
/// What the function is really for is the guard above it. An out-of-contract
/// value used to fall out of a `_ => DONT_CARE` catch-all, the most destructive
/// default available: DONT_CARE tells Metal the previous attachment contents may
/// be discarded, so a decode that read the wrong offset produced a *discarded
/// framebuffer* and no log line at all. An unrecognised value now says so once
/// per `(pipeline, slug)`.
///
/// The answer stays DONT_CARE rather than becoming LOAD or CLEAR. Out of
/// contract means this crate misread the field, not that the guest asked for
/// something exotic — every alternative is equally a guess, and inventing
/// semantics for an unknown wire value is what the ground rules forbid. What
/// changes is that the guess is now visible.
fn map_load_action(pipeline_ref: u32, a: u16) -> u32 {
    use crate::backend::metal::abi::{
        REIMS_VGPU_MTL_LOAD_ACTION_CLEAR, REIMS_VGPU_MTL_LOAD_ACTION_DONT_CARE,
        REIMS_VGPU_MTL_LOAD_ACTION_LOAD,
    };
    if !load_action_in_contract(pipeline_ref, a) {
        return REIMS_VGPU_MTL_LOAD_ACTION_DONT_CARE;
    }
    match a {
        MTL_LOAD_ACTION_LOAD => REIMS_VGPU_MTL_LOAD_ACTION_LOAD,
        MTL_LOAD_ACTION_CLEAR => REIMS_VGPU_MTL_LOAD_ACTION_CLEAR,
        // DontCare, and — because `a` is a `u16` and the contract is three
        // ordinals of it — the values the guard above has already reported.
        _ => REIMS_VGPU_MTL_LOAD_ACTION_DONT_CARE,
    }
}

fn map_store_action(pipeline_ref: u32, a: u16) -> u32 {
    use crate::backend::metal::abi::{
        REIMS_VGPU_MTL_STORE_ACTION_DONT_CARE, REIMS_VGPU_MTL_STORE_ACTION_STORE,
    };
    // Reports and returns; the answer for an out-of-contract value is the same
    // DontCare it always was.
    let _ = store_action_in_contract(pipeline_ref, a);
    if a == MTL_STORE_ACTION_STORE {
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

impl crate::observe::Decline for MetalStateDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::SamplerEntryMissing { .. } => {
                crate::observe::ladder_slug!("metal_sampler", no_list_entry)
            }
            Self::SamplerObjectType { .. } => {
                crate::observe::ladder_slug!("metal_sampler", wrong_type)
            }
            Self::SamplerDescriptorMissing { .. } => {
                crate::observe::ladder_slug!("metal_sampler", desc_read)
            }
            Self::SamplerDecode { reason, .. } => reason.slug(),
            Self::DepthStencilEntryMissing { .. } => {
                crate::observe::ladder_slug!("metal_depth_stencil", no_list_entry)
            }
            Self::DepthStencilObjectType { .. } => {
                crate::observe::ladder_slug!("metal_depth_stencil", wrong_type)
            }
            Self::DepthStencilDescriptorMissing { .. } => {
                crate::observe::ladder_slug!("metal_depth_stencil", desc_read)
            }
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

fn icb_depth_stencil_decline(req: &DrawEncodeRequest) -> Option<MetalStateDecline> {
    let depth_attachment = req
        .depth_attach
        .as_ref()
        .is_some_and(|attachment| attachment.texture_ref != 0);
    let stencil_attachment = req
        .stencil_attach
        .as_ref()
        .is_some_and(|attachment| attachment.texture_ref != 0);
    (req.depth_stencil_ref != 0 || depth_attachment || stencil_attachment).then_some(
        MetalStateDecline::IcbDepthStencilUnsupported {
            depth_stencil_ref: req.depth_stencil_ref,
            depth_attachment,
            stencil_attachment,
        },
    )
}

fn load_depth_stencil_state<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    ds_ref: u32,
) -> Result<crate::backend::metal::abi::ReimsVgpuDepthStencilState, MetalStateDecline> {
    use crate::backend::metal::abi::{ReimsVgpuDepthStencilFaceState, ReimsVgpuDepthStencilState};
    let (_entry, desc) = objects::resolve_descriptor(
        state,
        host,
        task_id,
        ds_ref,
        &[OBJECT_TYPE_SERIALIZER_OBJECT],
    )
    .map_err(|rung| match rung {
        objects::LadderRung::NoListEntry | objects::LadderRung::NoTaskSpace => {
            MetalStateDecline::DepthStencilEntryMissing {
                depth_stencil_ref: ds_ref,
            }
        }
        objects::LadderRung::WrongType { got } => MetalStateDecline::DepthStencilObjectType {
            depth_stencil_ref: ds_ref,
            object_type: got,
        },
        objects::LadderRung::DescRead { .. } => MetalStateDecline::DepthStencilDescriptorMissing {
            depth_stencil_ref: ds_ref,
        },
    })?;
    let d = decode_depth_stencil_descriptor(&desc).map_err(|reason| {
        MetalStateDecline::DepthStencilDecode {
            depth_stencil_ref: ds_ref,
            reason,
        }
    })?;
    crate::runtime::draw::name_depth_stencil(state, host, task_id, ds_ref);
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

/// Apply a bind record's own LOD clamps over whatever the sampler object
/// declared.
///
/// `setVertexSamplerStates:lodMinClamps:lodMaxClamps:withRange:` and its
/// fragment sibling let one sampler state be bound at several slots with a
/// different clamp at each, which is the whole reason the pair rides on the
/// bind rather than on the object. `None` leaves the object's own clamps in
/// force, which is what the plain `setVertexSamplerStates:` means.
///
/// Both spellings are written, because [`ReimsVgpuSampler`] carries the clamp
/// twice — `lod_min_bits` beside the rest of the descriptor, and
/// `clamp_lod_min_bits` under `has_lod_clamp` — and [`sampler_record`] fills
/// both from one value for exactly that reason. Writing one of the two would
/// hand the shim a descriptor that disagrees with itself.
///
/// [`ReimsVgpuSampler`]: crate::backend::metal::abi::ReimsVgpuSampler
fn with_bind_lod_clamp(
    mut sampler: crate::backend::metal::abi::ReimsVgpuSampler,
    lod_clamp: Option<(u32, u32)>,
) -> crate::backend::metal::abi::ReimsVgpuSampler {
    if let Some((min_bits, max_bits)) = lod_clamp {
        sampler.lod_min_bits = min_bits;
        sampler.lod_max_bits = max_bits;
        sampler.has_lod_clamp = 1;
        sampler.clamp_lod_min_bits = min_bits;
        sampler.clamp_lod_max_bits = max_bits;
    }
    sampler
}

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

/// The Metal sampler ABI record for a decoded serializer-object sampler descriptor.
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
///   pipeline binds it, which the serializer-object descriptor cannot state.
///
/// `has_lod_clamp` is always 1: both clamp fields are filled on every path
/// here, from the binding when it carried one and from the descriptor
/// otherwise. [`default_sampler`] is the one record with no clamp to describe.
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
        max_anisotropy: sd.max_anisotropy,
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

#[cfg(test)]
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
            max_anisotropy: 1,
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

    /// The record binds the descriptor's anisotropy through, because
    /// [`crate::runtime::decode::resource::decode_sampler_descriptor`] is where
    /// the floor lives and this type has
    /// no other producer.
    #[test]
    fn anisotropy_is_carried_from_the_descriptor() {
        let mut sd = descriptor();
        sd.max_anisotropy = 4;
        assert_eq!(
            super::sampler_record(64, &sd, None, false).max_anisotropy,
            4
        );
    }
}

/// Load a sampled texture as tight RGBA8: mapper-ref-texture, texture-view→base+mip+format+swizzle,
/// or normal-texture.
fn load_sampled_rgba<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
) -> Option<(u32, u32, Vec<u8>)> {
    if texture_ref == 0 {
        return None;
    }
    // Opcode-9 buffer-backed texture (texture-view): sample the source buffer directly.
    if let Some(bt) = buffer_texture_descriptor(state, host, task_id, texture_ref, None) {
        return load_buffer_texture_rgba(state, host, task_id, texture_ref, &bt);
    }
    if let Some(v) = load_mapper_ref_texture_rgba(state, host, task_id, texture_ref, None) {
        return Some(v);
    }
    // Texture-view view → base texture + selected mip + format override + optional swizzle.
    if let Some(view) = resolve_texture_view(state, host, task_id, texture_ref) {
        let mut loaded = if let Some(v) = load_mapper_ref_texture_rgba(
            state,
            host,
            task_id,
            view.base_texture_ref,
            view.pixel_format,
        ) {
            // Mapper-ref-texture IOSurface textures are single-level only: Metal rejects
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

fn load_mapper_ref_texture_rgba<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    format_override: Option<u16>,
) -> Option<(u32, u32, Vec<u8>)> {
    let mapping_id = objects::resolve_mapper_ref_texture(state, host, task_id, texture_ref)?;
    if let Some(served) = sampled_from_published_surface(state, host, mapping_id, format_override) {
        return Some(served);
    }
    load_mapper_ref_texture_mapping_rgba(state, host, mapping_id, format_override)
}

/// A sampled read served from this device's own last publication of the surface,
/// instead of from the surface's guest pages.
///
/// # Why this is the whole of the bar
///
/// This rail's sampled path had no cache rung at all: every bind walked the
/// surface's pages and decoded every texel, from scratch, every draw. Measured
/// on a driven macos-13 boot, `metal_sampled_load_us` was 100.0 % of
/// `sampled_us` and `sampled_us` was 48 % of the whole chain — 2.15 MB a draw at
/// **150 MB/s**, against 2.4 GB/s for `replace_region` pushing the identical
/// bytes back out. The gap is the per-texel decode, not the copy.
///
/// The cache holds those same texels already, in BGRA, published by the Store
/// that rendered them, and reaching them costs a refcount and a four-byte
/// shuffle.
///
/// # Why the bytes are the same bytes
///
/// [`crate::runtime::draw::load_mapper_ref_texture_mapping_rgba`] keys on the
/// mapping's *latched* geometry — which is what
/// [`crate::runtime::surface_cache`] is keyed by — reads BGRA8 out of the
/// mapping, and converts it to RGBA8. With no view format override that
/// conversion is exactly a red/blue swap, so this arm and that one produce the
/// same image or this one does not fire.
///
/// A format override is refused rather than reinterpreted: the override
/// reinterprets the storage, and a cache entry records no such
/// reinterpretation. Counted, so the size of what is being left on the table is
/// visible rather than assumed.
///
/// # Which evidence standard, and why the strict one again
///
/// A stale sampled texture is less bad than a stale LOAD seed — nothing writes
/// a sampled read back, so the next frame re-samples — but the frame it
/// corrupts *is* written back by its pass's Store, so the error still reaches
/// the guest's pages. And the rung below this one reads the guest's pages, so a
/// refusal costs exactly what this path costs today.
///
/// Same door, same standard, one gate: widening is a decision to take once, on
/// the counters, for both readers — not twice, differently, here.
fn sampled_from_published_surface<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &M,
    mapping_id: u32,
    format_override: Option<u16>,
) -> Option<(u32, u32, Vec<u8>)> {
    use crate::runtime::draw::NoPublishedFrame;
    use crate::runtime::surface_currency::SurfaceCurrency;

    if format_override.is_some() {
        crate::runtime::drain::note_store_route("metal_sampled_surface_override");
        return None;
    }
    let (w, h) = {
        let m = state.mappings.get(&mapping_id)?;
        if !m.has_geom || m.width == 0 || m.height == 0 {
            // The read below latches the geometry as it goes, so the next bind
            // of this surface reaches the door. Not a refusal of the frame.
            crate::runtime::drain::note_store_route("metal_sampled_surface_ungeometried");
            return None;
        }
        (m.width, m.height)
    };
    crate::runtime::drain::note_store_route("metal_sampled_surface_asked");
    let published =
        match crate::runtime::draw::published_mapping_frame(state, host, mapping_id, w, h) {
            Ok(frame) => frame,
            Err(decline) => {
                crate::runtime::drain::note_store_route(match decline {
                    NoPublishedFrame::Uncurrent(_, SurfaceCurrency::Unwritten(_)) => {
                        "metal_sampled_surface_unwatched"
                    }
                    NoPublishedFrame::Uncurrent(_, SurfaceCurrency::WrotePixels(_)) => {
                        "metal_sampled_surface_repainted"
                    }
                    NoPublishedFrame::Uncurrent(_, SurfaceCurrency::WroteUnknown) => {
                        "metal_sampled_surface_unknown"
                    }
                    NoPublishedFrame::Unpublished(_) => "metal_sampled_surface_empty",
                    // `serves` admits `WroteElsewhere` and the mapping and geometry were
                    // both checked above, so either arm here is a contradiction between
                    // this match and the gate rather than a guest behaviour.
                    NoPublishedFrame::Uncurrent(_, SurfaceCurrency::WroteElsewhere)
                    | NoPublishedFrame::NotMapped => "metal_sampled_surface_impossible",
                });
                return None;
            }
        };
    // Two sources for one frame, in the order of what they cost. The cache hands
    // over an `Arc` and a whole-frame channel exchange; the resident hands over a
    // GPU readback and no exchange. Which one holds the frame is decided by the
    // Store that published it — see `mapping_write::FramePublication` — and both
    // are gated on the *same* generation the door above just read, so neither can
    // serve a frame the other has superseded.
    if let Some(bgra) = crate::runtime::surface_cache::get_shared(state, mapping_id, w, h) {
        crate::runtime::drain::note_store_route("metal_sampled_from_surface");
        crate::runtime::drain::note_store_route_n("metal_sampled_surface_bytes", bgra.len() as u64);
        return Some((w, h, swap_rb_channels(&bgra)));
    }
    let rgba = crate::backend::metal::resident::read_published_rgba8(
        &crate::backend::metal::resident::ResidentColorKey::for_surface(mapping_id, w, h),
        published.generation,
    )?;
    crate::runtime::drain::note_store_route("metal_sampled_from_resident");
    crate::runtime::drain::note_store_route_n("metal_sampled_surface_bytes", rgba.len() as u64);
    Some((w, h, rgba))
}

/// normal-texture linear texture at mip `level`: strided guest rows → tight RGBA8.
///
/// `format_override` is the texture-view pixel format when present. Base storage
/// geometry (row_stride / level layout) stays on the base texture; the sample
/// format must be bpp-compatible with the base (Metal texture-view contract).
fn load_linear_texture_rgba_at_level<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    level: u32,
    format_override: Option<u16>,
) -> Option<(u32, u32, Vec<u8>)> {
    let (_entry, desc_bytes) = objects::resolve_descriptor(
        state,
        host,
        task_id,
        texture_ref,
        &[OBJECT_TYPE_TEXTURE, OBJECT_TYPE_TEXTURE_GENERATE_MIPMAPS],
    )
    .ok()?;
    let tex = decode_texture_descriptor(&desc_bytes).ok()?;
    // A descriptor that declares no pixel format is not a texture this can
    // sample; the field's own value is read below only once that is settled.
    tex.declared_pixel_format()?;
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
    // The extent this actually reads: the loop below walks `gva + y * bpr` for
    // `tight` bytes, so the last row's trailing padding is never touched.
    // `row_stride * height` charges for it, and as a bound against the guest's
    // own `allocation_size` that refuses images the guest sized correctly —
    // `TextureLevelLayout::read_span` carries the measured case, a 27x27
    // RG8Unorm window mask that this arm was rejecting whole.
    let span = layout.read_span(tight)?;
    if tex.allocation_size != 0 && layout.offset.saturating_add(span) > tex.allocation_size {
        return None;
    }
    // Parsed once, above the row loop. This loop is this rail's sampled bar:
    // `metal_sampled_load_us` was 100 % of `sampled_us` on a driven macos-13
    // boot, and re-deciding the format ordinal per texel was most of it. See
    // `pixel_format::RowToRgba8`.
    let row_rail = pixel_format::RowToRgba8::for_format(sample_fmt)?;
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
        if !row_rail.convert(&row, w, &mut rgba[dst_off..]) {
            return None;
        }
    }
    Some((w, h, rgba))
}

fn load_sampler<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    sampler_ref: u32,
    slot: u32,
) -> Result<crate::backend::metal::abi::ReimsVgpuSampler, MetalStateDecline> {
    use crate::backend::metal::abi::REIMS_VGPU_BINDING_SAMPLER_BASE;
    let sampler =
        objects::resolve_sampler_state(state, host, task_id, sampler_ref).map_err(|failure| {
            match failure {
                objects::SamplerResolveError::Rung(rung) => match rung {
                    objects::LadderRung::NoListEntry | objects::LadderRung::NoTaskSpace => {
                        MetalStateDecline::SamplerEntryMissing {
                            sampler_ref,
                            index: slot,
                        }
                    }
                    objects::LadderRung::WrongType { got } => {
                        MetalStateDecline::SamplerObjectType {
                            sampler_ref,
                            index: slot,
                            object_type: got,
                        }
                    }
                    objects::LadderRung::DescRead { .. } => {
                        MetalStateDecline::SamplerDescriptorMissing {
                            sampler_ref,
                            index: slot,
                        }
                    }
                },
                objects::SamplerResolveError::Decode { status, .. } => {
                    MetalStateDecline::SamplerDecode {
                        sampler_ref,
                        index: slot,
                        reason: status,
                    }
                }
            }
        })?;
    Ok(sampler_record(
        REIMS_VGPU_BINDING_SAMPLER_BASE + slot,
        &sampler.descriptor,
        None,
        false,
    ))
}

/// Store scissor rect of tight RGBA8 into a mapper-ref-texture mapping (BGRA host → guest fmt).
// Source geometry, destination geometry and the scissor rect are three
// independent rectangles and stay three: collapsing them into one struct would
// invite exactly the mix-up the separate names prevent.
//
// Giving the scissor one a `Rect` serves that same argument rather than
// undoing it. Its four fields used to sit adjacent to `full_w`/`full_h` as six
// interchangeable `u32`s, so the mix-up the comment warns about was writable at
// every call; now the scissor rectangle is the one thing here with a type, and
// the destination extent is what remains loose beside it.
#[allow(clippy::too_many_arguments)]
fn write_mapping_rgba8_rect<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    full_w: u32,
    full_h: u32,
    format: u16,
    rgba: &[u8],
    rect: mapping_write::Rect,
) -> bool {
    let mapping_write::Rect {
        origin_x,
        origin_y,
        width: rect_w,
        height: rect_h,
    } = rect;
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
    let Some(store_rail) = pixel_format::Rgba8ToRow::for_format(format) else {
        return false;
    };
    let mut raw = vec![0u8; tight.saturating_mul(rect_h as usize)];
    let mut guest_row = vec![0u8; tight];
    for dy in 0..rect_h as usize {
        let y = origin_y as usize + dy;
        let src = &rgba[y * rgba_row + (origin_x as usize) * 4
            ..y * rgba_row + (origin_x as usize) * 4 + (rect_w as usize) * 4];
        // Guest store is native format; convert from tight RGBA8 (same as full write_gva path).
        if !store_rail.convert(src, rect_w, &mut guest_row) {
            return false;
        }
        raw[dy * tight..dy * tight + tight].copy_from_slice(&guest_row);
    }
    mapping_write::write_rect_raw(
        state,
        host,
        mapping_id,
        mapping_write::Rect {
            origin_x,
            origin_y,
            width: rect_w,
            height: rect_h,
        },
        &raw,
        tight as u32,
    )
}

/// The Metal rail's own checks.
///
/// These lived in `super::tests` under a per-test `cfg`. They are here for the
/// same reason the rail is: they reach this module's private state vocabulary
/// (`MetalStateDecline`, `MetalIcbInheritanceDecline`), and reaching it from a
/// sibling would have meant widening those types' visibility so that a test
/// could see them.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
    use crate::runtime::host::FakeHost;

    /// A sampled read of a mapper-ref-texture surface is served from this
    /// device's own published frame while the hypervisor's witness says the
    /// guest has not repainted it, and from the surface's own pages otherwise.
    ///
    /// The rung this adds is the whole of the rail's largest bar: measured on a
    /// driven macos-13 boot, `metal_sampled_load_us` was 100.0 % of
    /// `sampled_us`, moving 2.15 MB a draw at 150 MB/s through a per-texel
    /// decode of pages this device had just written itself.
    ///
    /// Three legs, and the middle one is the reason the strict evidence
    /// standard is used here as well as on the LOAD seed: a rail whose
    /// dirty-tracking witness never arms answers `NoStamp` to every ask, and
    /// under the permissive standard this door would then hand every shader
    /// whatever the cache held, forever.
    #[test]
    fn a_sampled_mapper_ref_texture_serves_a_published_frame_only_on_a_watched_clean_witness() {
        use crate::model::PAGE_SHIFT_X86;
        use crate::protocol::endian::{st16, st32};
        use crate::protocol::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
        use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        use crate::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM;
        use crate::runtime::decode::resource::{
            list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_MAPPER_REF_TEXTURE,
        };
        use crate::runtime::gva_mem;
        use crate::runtime::host::{HostMemory, HostOps};

        let (w, h) = (2u32, 2u32);
        let (mid, texture_ref, task_id) = (71u32, 2u32, 1u32);

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();

        let (dir_pfn, root_pfn) = (2u32, 3u32);
        let (dir_gpa, root_gpa) = (
            (dir_pfn as u64) << PAGE_SHIFT_X86,
            (root_pfn as u64) << PAGE_SHIFT_X86,
        );
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x1000, 0);
        let mut dir = [0u8; 8];
        st32(&mut dir[DIRECTORY_ROOT_PFN as usize..], root_pfn);
        st32(&mut dir[DIRECTORY_DEPTH as usize..], 1);
        assert!(host.write_gpa(dir_gpa, &dir).is_ok());
        for i in 0..3u32 {
            let pfn = 4 + i;
            host.map_range((pfn as u64) << PAGE_SHIFT_X86, 0x1000, 0);
            let mut pte = [0u8; 4];
            st32(&mut pte, pfn);
            assert!(host.write_gpa(root_gpa + (i as u64) * 4, &pte).is_ok());
        }
        state.define_task(task_id, 0x1000, dir_pfn);
        assert!(state.set_object_list(task_id, 0, 32));

        const DESC_LEN: usize = 0x20;
        let desc_gva = 0x1000u64;
        let mut desc = vec![0u8; DESC_LEN];
        st32(&mut desc[0..], mid);
        st16(&mut desc[0x16..], MTL_FORMAT_BGRA8_UNORM);
        st32(&mut desc[0x18..], w);
        st32(&mut desc[0x1c..], h);
        assert!(gva_mem::write_task_gva(
            &mut host,
            &state.tasks[task_id],
            desc_gva,
            &desc,
            PAGE_SHIFT_X86
        )
        .is_ok());
        let mut list_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
        st32(
            &mut list_entry,
            u32::from(OBJECT_TYPE_MAPPER_REF_TEXTURE) | ((DESC_LEN as u32) << 8),
        );
        list_entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
        assert!(gva_mem::write_task_gva(
            &mut host,
            &state.tasks[task_id],
            list_object_entry_offset(texture_ref, 32).unwrap(),
            &list_entry,
            PAGE_SHIFT_X86
        )
        .is_ok());

        let page_gpa = 6u64 << PAGE_SHIFT_X86;
        assert!(state.map_surface(mid));
        {
            let m = state.mappings.get_mut(&mid).expect("mapped above");
            m.mapped = true;
            m.mapping_internal = 1;
            m.page_entries = vec![(6u32 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        }
        assert!(state.set_mapping_geom(mid, w, h, MTL_FORMAT_BGRA8_UNORM));
        // What the surface's own pages hold, and what this device last
        // published for it — different colours, so "served from the cache" and
        // "read from the guest" are told apart by value and not by a counter.
        assert!(host
            .write_gpa(
                page_gpa,
                &[0x11u8, 0x22, 0x33, 0xff].repeat((w * h) as usize)
            )
            .is_ok());
        const GUEST_RGBA: [u8; 4] = [0x33, 0x22, 0x11, 0xff];
        crate::runtime::surface_cache::store(
            &mut state,
            mid,
            w,
            h,
            [0xAAu8, 0xBB, 0xCC, 0xff].repeat((w * h) as usize),
        );
        const PUBLISHED_RGBA: [u8; 4] = [0xCC, 0xBB, 0xAA, 0xff];

        // Unwatched: the witness has not been armed, so the strict standard
        // refuses and the guest's pages answer.
        let (gw, gh, rgba) = load_sampled_rgba(&mut state, &mut host, task_id, texture_ref)
            .expect("the surface's own pages are always a sampled source");
        assert_eq!((gw, gh), (w, h));
        assert_eq!(
            &rgba[..4],
            &GUEST_RGBA,
            "an unstamped surface has no witness to spend, and the cache must not be served"
        );

        // Watched and clean, which is what `mapping_write` leaves behind when it
        // publishes: the device's own frame is the surface.
        let token = crate::runtime::mapper::ensure_guest_write_token(&mut state, &mut host, mid)
            .expect("FakeHost observes guest writes");
        state
            .mappings
            .get_mut(&mid)
            .expect("mapped above")
            .guest_write_gen_at_store = host.guest_write_gen(token).expect("a live token has one");
        let (_, _, served) = load_sampled_rgba(&mut state, &mut host, task_id, texture_ref)
            .expect("a published frame under a clean witness is the surface");
        assert_eq!(
            &served[..4],
            &PUBLISHED_RGBA,
            "with the witness clean the device's own publication is the surface, and the \
             per-texel decode of the guest's pages is the cost this door exists to remove"
        );

        // Repainted by the guest CPU, with no device operation at all — this
        // witness is the only thing that sees it.
        host.guest_wrote_page(page_gpa);
        let (_, _, repainted) = load_sampled_rgba(&mut state, &mut host, task_id, texture_ref)
            .expect("a repainted surface still samples, from its own pages");
        assert_eq!(
            &repainted[..4],
            &GUEST_RGBA,
            "the cache is a frame the guest has since painted over"
        );
    }

    #[test]
    fn explicit_metal_sampler_and_depth_binds_return_typed_missing_entry_declines() {
        use crate::observe::Emit;

        let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let host = FakeHost::new();

        let sampler = load_sampler(&state, &host, 4, 77, 3)
            .expect_err("a nonzero sampler ref in an empty object list must decline");
        assert_eq!(
            sampler,
            MetalStateDecline::SamplerEntryMissing {
                sampler_ref: 77,
                index: 3,
            }
        );
        assert_eq!(
            Emit::decline("metal_draw_sampler_fallback", &sampler)
                .field("task", 4)
                .field("pipe", 19)
                .field("stage", "fragment")
                .render(),
            format!(
                "metal_draw_sampler_fallback reason={} \
                 sampler_ref=77 index=3 task=4 pipe=19 stage=fragment",
                crate::observe::ladder_slug!("metal_sampler", no_list_entry)
            )
        );

        let depth = load_depth_stencil_state(&state, &host, 4, 88)
            .expect_err("a nonzero depth-stencil ref in an empty object list must decline");
        assert_eq!(
            depth,
            MetalStateDecline::DepthStencilEntryMissing {
                depth_stencil_ref: 88,
            }
        );
        assert_eq!(
            Emit::decline("metal_draw_depth_stencil_fallback", &depth)
                .field("task", 4)
                .field("pipe", 19)
                .render(),
            format!(
                "metal_draw_depth_stencil_fallback reason={} \
                 depth_stencil_ref=88 task=4 pipe=19",
                crate::observe::ladder_slug!("metal_depth_stencil", no_list_entry)
            )
        );
    }

    #[test]
    fn metal_state_decode_declines_delegate_the_exact_resource_decoder_reason() {
        use crate::observe::{Decline as _, Emit};

        let sampler = MetalStateDecline::SamplerDecode {
            sampler_ref: 41,
            index: 6,
            reason: DecodeStatus::ErrShort("res_sampler_short"),
        };
        assert_eq!(sampler.slug(), "res_sampler_short");
        assert_eq!(
            Emit::decline("metal_draw_sampler_fallback", &sampler).render(),
            "metal_draw_sampler_fallback reason=res_sampler_short \
                 class=short sampler_ref=41 index=6"
        );

        let depth = MetalStateDecline::DepthStencilDecode {
            depth_stencil_ref: 52,
            reason: DecodeStatus::ErrShort("res_depth_stencil_short"),
        };
        assert_eq!(depth.slug(), "res_depth_stencil_short");
        assert_eq!(
            Emit::decline("metal_draw_depth_stencil_fallback", &depth).render(),
            "metal_draw_depth_stencil_fallback reason=res_depth_stencil_short \
                 class=short depth_stencil_ref=52"
        );
    }

    #[test]
    fn every_metal_icb_inheritance_check_is_unique_namespaced_and_log_safe() {
        use crate::observe::Decline as _;

        let all = vec![
            MetalIcbInheritanceDecline::CullModeUnsupported { value: 3 },
            MetalIcbInheritanceDecline::FrontFacingUnsupported { value: 2 },
            MetalIcbInheritanceDecline::BindSlotPastTable {
                bind: PastTableBind {
                    class: BindTableClass::Buffer,
                    stage: reims_vgpu_protocol::render::ShaderStage::Vertex,
                    index: MAX_BUFFER_BIND_SLOTS,
                    resource_ref: 1,
                },
            },
            MetalIcbInheritanceDecline::VertexBufferMissing {
                buffer_ref: 3,
                index: 4,
                offset: 5,
            },
            MetalIcbInheritanceDecline::FragmentBufferMissing {
                buffer_ref: 6,
                index: 7,
                offset: 8,
            },
            MetalIcbInheritanceDecline::VertexTextureMissing {
                texture_ref: 11,
                index: 12,
                detail: "no list entry".into(),
            },
            MetalIcbInheritanceDecline::FragmentTextureMissing {
                texture_ref: 13,
                index: 14,
                detail: "guest\nread failed".into(),
            },
            MetalIcbInheritanceDecline::PipelineRefZero,
            MetalIcbInheritanceDecline::PipelineMissing { pipeline_ref: 17 },
            MetalIcbInheritanceDecline::VertexMtlbMissing { function_ref: 18 },
            MetalIcbInheritanceDecline::FragmentMtlbMissing { function_ref: 19 },
            MetalIcbInheritanceDecline::VertexLibraryLoad {
                function_ref: 20,
                detail: "Metal error".into(),
            },
            MetalIcbInheritanceDecline::FragmentLibraryLoad {
                function_ref: 21,
                detail: "Metal error".into(),
            },
            MetalIcbInheritanceDecline::VertexFunctionCount {
                function_ref: 22,
                count: 2,
            },
            MetalIcbInheritanceDecline::FragmentFunctionCount {
                function_ref: 23,
                count: 0,
            },
            MetalIcbInheritanceDecline::VertexFunctionGet {
                function_ref: 24,
                detail: "function missing".into(),
            },
            MetalIcbInheritanceDecline::FragmentFunctionGet {
                function_ref: 25,
                detail: "function missing".into(),
            },
            MetalIcbInheritanceDecline::VertexDescriptorMissing {
                pipeline_ref: 26,
                attribute_count: 3,
            },
            // The sibling of the entry above, and the pair is the point: an empty
            // vertex block and a partial one are different refusals, and the
            // builder used to report the second as a success.
            MetalIcbInheritanceDecline::VertexAttributeUnencodable {
                pipeline_ref: 26,
                location: 3,
                value: 99,
            },
            MetalIcbInheritanceDecline::RenderPipelineCreate {
                pipeline_ref: 27,
                detail: "pipeline failed".into(),
            },
            MetalIcbInheritanceDecline::AllocationFailed {
                what: "inherited_texture",
            },
        ];

        // Coverage is checked by an exhaustive `match`, not by a count. A literal
        // here is a second spelling of the variant list: when the enum lost five
        // checks this assertion still said 26 and failed for that reason alone,
        // reporting a shrunk vocabulary as a fixture gap. `variant_name` has no `_`
        // arm, so a new check stops this test compiling until the fixture carries
        // one — which is what "the fixture must cover every check" was asking for.
        fn variant_name(decline: &MetalIcbInheritanceDecline) -> &'static str {
            use MetalIcbInheritanceDecline as D;
            match decline {
                D::CullModeUnsupported { .. } => "CullModeUnsupported",
                D::FrontFacingUnsupported { .. } => "FrontFacingUnsupported",
                D::BindSlotPastTable { .. } => "BindSlotPastTable",
                D::VertexBufferMissing { .. } => "VertexBufferMissing",
                D::FragmentBufferMissing { .. } => "FragmentBufferMissing",
                D::VertexTextureMissing { .. } => "VertexTextureMissing",
                D::FragmentTextureMissing { .. } => "FragmentTextureMissing",
                D::PipelineRefZero => "PipelineRefZero",
                D::PipelineMissing { .. } => "PipelineMissing",
                D::VertexMtlbMissing { .. } => "VertexMtlbMissing",
                D::FragmentMtlbMissing { .. } => "FragmentMtlbMissing",
                D::VertexLibraryLoad { .. } => "VertexLibraryLoad",
                D::FragmentLibraryLoad { .. } => "FragmentLibraryLoad",
                D::VertexFunctionCount { .. } => "VertexFunctionCount",
                D::FragmentFunctionCount { .. } => "FragmentFunctionCount",
                D::VertexFunctionGet { .. } => "VertexFunctionGet",
                D::FragmentFunctionGet { .. } => "FragmentFunctionGet",
                D::VertexDescriptorMissing { .. } => "VertexDescriptorMissing",
                D::VertexAttributeUnencodable { .. } => "VertexAttributeUnencodable",
                D::RenderPipelineCreate { .. } => "RenderPipelineCreate",
                D::AllocationFailed { .. } => "AllocationFailed",
            }
        }
        let mut covered = all.iter().map(variant_name).collect::<Vec<_>>();
        covered.sort_unstable();
        covered.dedup();
        assert_eq!(
            covered.len(),
            all.len(),
            "the fixture names one check twice: {covered:?}"
        );

        let mut slugs = all.iter().map(Decline::slug).collect::<Vec<_>>();
        for decline in &all {
            assert!(decline.slug().starts_with("metal_icb_inherit_"));
            for (key, value) in decline.fields() {
                assert!(!key.is_empty());
                assert!(
                    !value.chars().any(char::is_whitespace),
                    "{} rendered a non-token field {key}={value:?}",
                    decline.slug()
                );
            }
        }
        slugs.sort_unstable();
        slugs.dedup();
        assert_eq!(slugs.len(), all.len(), "two ICB checks share one reason");
    }

    #[test]
    fn metal_icb_inheritance_line_keeps_pipeline_and_sanitized_driver_detail() {
        use crate::observe::Emit;

        let decline = MetalIcbInheritanceDecline::RenderPipelineCreate {
            pipeline_ref: 71,
            detail: "Error Domain=MTLLibrary Code=3".into(),
        };
        assert_eq!(
            Emit::decline("metal_icb_inheritance", &decline)
                .field("task", 2)
                .field("pipe", 71)
                .field("icb", 99)
                .render(),
            "metal_icb_inheritance reason=metal_icb_inherit_render_pipeline_create \
                 pipeline_ref=71 detail=Error_Domain=MTLLibrary_Code=3 \
                 task=2 pipe=71 icb=99"
        );
    }
}
