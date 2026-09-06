//! The Vulkan rail's compute dispatch: staged binds into an `engine::ComputeRequest`.
//!
//! The sibling of [`super::metal`]. [`super`] resolves the guest's dispatch and
//! hands the result here; this module is the half that turns staged bytes into
//! engine resources, decides where a storage image's output lands, and asks the
//! engine to execute.
//!
//! It also owns this rail's *residency* policy — the mirror of guest windows the
//! engine already holds, which lets a staging loop skip a guest read
//! ([`resident_serve`]). The Metal rail keeps no such mirror, which is why every
//! staging site there resolves to "no resident to serve from".

use super::*;
use crate::runtime::draw::vulkan::gva_span_identity;

/// The sampled-image bindings that need a neutral texture: those the module
/// statically uses and `bound` does not cover.
///
/// Vulkan requires the pipeline layout to contain a descriptor for every
/// resource the module statically uses, and the layout this device builds is
/// assembled from what the guest bound — so a texture the kernel samples and the
/// guest left empty is absent from the layout entirely, not an unwritten slot in
/// it. Besides being undefined by the specification, that hole is fatal on one
/// of the two iGPU vendors this device supports: Mesa's Intel driver scores each
/// used binding as `(use_count << 7) / array_size` over an array it sized to
/// `max_binding + 1` and zero-filled, so it divides by zero and the host process
/// dies of `SIGFPE` inside `vkCreateComputePipelines`.
///
/// Only [`DescriptorUse::Used`] is returned, which is the bar the specification
/// actually sets. A declared-and-never-referenced variable is legal to omit and
/// must stay omitted, or the census that separated those two populations cannot
/// tell them apart any more. `Ambiguous` — two variables on one binding — is its
/// own defect and is not repaired by picking one of them.
pub(super) fn neutral_sampled_image_bindings(spirv: &[u32], bound: &[u32]) -> Vec<u32> {
    crate::runtime::spirv_bind::sampled_image_bindings(spirv)
        .into_iter()
        .filter(|binding| {
            !bound.contains(binding)
                && crate::runtime::spirv_bind::descriptor_static_use(spirv, *binding).is_violation()
        })
        .collect()
}

/// Side length of the texture substituted for a sampled image the kernel
/// samples and the guest never bound.
///
/// One texel, because there is nothing to derive a size from: the guest supplied
/// no texture, and any larger extent would be a number chosen to look plausible.
/// A kernel that asks this image its size gets 1×1 and that is reported, rather
/// than a guess that reads as data.
pub(super) const NEUTRAL_SAMPLED_IMAGE_EXTENT: u32 = 1;

/// A sampled image the kernel statically uses and the guest never bound, given a
/// neutral transparent texture so the pipeline layout can describe it.
///
/// **A repair that succeeded, not a success**, which is why it goes to the fail
/// channel: the kernel samples a texture whose contents this device invented, and
/// the reliance has to stay measurable so a later session can find out whether
/// the guest ever depended on what was in it. Nothing here claims the read did
/// not matter.
///
/// Omitting the binding instead is not the cheaper option. It is a specification
/// violation, and on Mesa's Intel driver it is a `SIGFPE` that kills the host
/// process — see the walk in [`crate::runtime::spirv_bind::sampled_image_bindings`].
pub(super) struct NeutralSampledImage {
    binding: u32,
    width: u32,
    height: u32,
}

impl crate::observe::Decline for NeutralSampledImage {
    fn slug(&self) -> &'static str {
        "compute_neutral_sampled_image_unbound"
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("binding", self.binding.to_string()),
            ("width", self.width.to_string()),
            ("height", self.height.to_string()),
        ]
    }
}

/// Where one compute storage image's output should land.
///
/// `GuestPages` when this device can put the dispatch's own copy straight into
/// the guest's RAM, `Host` when it has to read the pixels back and write them
/// itself. **Every decline here costs a device→host crossing and no guest
/// work**: `Host` is the general path, is what a host without the guest-RAM
/// import runs for everything, and lands identical bytes. So these are routed
/// on the `OFF` channel as a census rather than refused on the fail channel —
/// nothing is lost, and the counters are what say how much of a boot's compute
/// readback this arm can actually reach.
///
/// Two conditions, and each names a contract term rather than an observation:
///
/// - the writeback must be a **guest-linear plane**. A mapper-ref-texture destination is a
///   tiled surface mapping, which [`crate::runtime::render_writeback::vulkan::GvaPlaneDestination`]
///   cannot describe and the licence therefore cannot walk. It is the largest
///   class this arm does not reach, so [`note_mapper_ref_texture_shape`] bands how much of
///   it a raw copy could ever serve — see that function for why the route
///   counter alone does not say.
/// - the licence must be granted. That is where the format, the complete page
///   walk, the texel alignment and the guest-RAM references are all checked, in
///   the one place both GPU-side writers of a guest plane meet them.
///
/// # Residency is not a third condition
///
/// It was, for one boot, and that restriction reached 81 of the 89 linear
/// windows a driven macos-13 boot produces — so a rule written to be safe was
/// most of the traffic this arm exists to remove. What it was protecting against
/// is real but is not the reclaim: both reclaim paths already skip a resident
/// whose `gpu_only_content` holds, and every executed dispatch sets that flag.
/// The actual window is a **re-key**, which destroys the held image when the same
/// identity arrives at a new shape, and the pin is what refuses it. The engine
/// now takes that pin itself when it arms the write debt — see
/// `GuestWriteSource::ResidentStorage` — and releases it from the ring slot's
/// cleanup, after the fence. So a resident is held for exactly the window a
/// submitted-not-waited copy needs, and the destination no longer has to care.
pub(super) fn direct_destination<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    tex: &StagedTexture<VulkanStage>,
    held: ash::vk::Format,
) -> crate::backend::vulkan::engine::ComputeImageDestination {
    use crate::backend::vulkan::engine::ComputeImageDestination;
    let TextureWriteback::Linear {
        texture_ref,
        gva,
        pixel_format,
        row_stride,
        width,
        height,
        pages,
        ..
    } = &tex.writeback
    else {
        return mapper_ref_texture_destination(state, host, tex, held);
    };
    let Ok(row_stride) = u32::try_from(*row_stride) else {
        crate::runtime::drain::note_store_route("compute_dst_host_stride_width");
        return ComputeImageDestination::Host;
    };
    let plane = crate::runtime::render_writeback::vulkan::GvaPlaneDestination {
        target_gva: *gva,
        width: *width,
        height: *height,
        row_stride,
        format: *pixel_format,
        texture_ref: *texture_ref,
    };
    match crate::runtime::render_writeback::vulkan::licence_gva_plane(
        state,
        host,
        held,
        &plane,
        Some(pages),
    ) {
        Ok(licence) => {
            if let Some((need, have)) = licence.target.window_shortfall() {
                if crate::observe::first_sight(
                    "compute_dst_window_too_small",
                    u64::from(tex.binding),
                ) {
                    crate::observe::fail(format!(
                        "compute_dst_window_too_small bind={} gva={gva:#x} dims={width}x{height} \
                         fmt={pixel_format:#x} need={need} have={have} (the storage image names \
                         more texels than its guest pages hold; a direct landing would write \
                         past them, so it takes the pooled readback)",
                        tex.binding
                    ));
                }
                crate::runtime::drain::note_store_route("compute_dst_host_window_too_small");
                return ComputeImageDestination::Host;
            }
            crate::runtime::drain::note_store_route("compute_dst_guest_pages");
            // A split of the line above, so the two add up to it. Worth counting
            // separately because the resident half is the half that needs the
            // engine's pin, and it is the half that used to read back: a boot
            // where it stays at zero is a boot where the pin never had to work.
            crate::runtime::drain::note_store_route(if tex.rail.residency.is_some() {
                "compute_dst_guest_pages_resident"
            } else {
                "compute_dst_guest_pages_transient"
            });
            ComputeImageDestination::GuestPages {
                target: Box::new(licence.target),
                pages: licence.gpas,
            }
        }
        Err(decline) => {
            // Named, because the reasons are not interchangeable and the census
            // above cannot tell them apart: a format the copy cannot land raw is
            // a different thing to learn about this rail than a page walk that
            // came up short.
            crate::observe::off(format!(
                "compute_dst host bind={} gva={gva:#x} dims={width}x{height} fmt={pixel_format:#x} reason={decline:?}",
                tex.binding
            ));
            crate::runtime::drain::note_store_route("compute_dst_host_unlicensed");
            ComputeImageDestination::Host
        }
    }
}

/// [`direct_destination`] for a mapper-ref-texture surface mapping.
///
/// A tiled surface mapping is not a guest-linear plane and the GVA licence
/// cannot describe one — but it is not therefore unreachable, and treating it as
/// such is what this arm used to do. It answered `Host` before looking at
/// anything, and on a driven macos-13 boot that was 35 of the 51 storage
/// destinations, every one of them a device→host crossing.
///
/// The destination that *can* describe it already existed on the render rail,
/// resolving the sample window, walking the mapping's page entries and building
/// the same [`crate::backend::vulkan::engine::GuestPageTarget`] this rail wants.
/// It is now [`crate::runtime::mapping_write::vulkan::licence_mapper_ref_texture_surface`] and both
/// rails ask it, so the surface geometry, the format rule, the page walk and the
/// guest-RAM references have one spelling rather than two.
///
/// Every decline is a routing answer on the `OFF` channel, not a loss: readback
/// lands identical bytes, and on a host without the guest-RAM import it is the
/// only rail there is.
pub(super) fn mapper_ref_texture_destination<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    tex: &StagedTexture<VulkanStage>,
    held: ash::vk::Format,
) -> crate::backend::vulkan::engine::ComputeImageDestination {
    use crate::backend::vulkan::engine::ComputeImageDestination;
    let TextureWriteback::MapperRefTexture {
        mapping_id,
        surface_offset,
        surface_bpr,
        span_end,
        width,
        height,
        format,
        ..
    } = &tex.writeback
    else {
        // A storage image the guest gave nowhere to land. Not a destination this
        // arm declined — there is no destination.
        crate::runtime::drain::note_store_route("compute_dst_no_writeback");
        return ComputeImageDestination::Host;
    };
    // The window this bind staged against, not one resolved here. It is already
    // plane-correct for a ref-texture view and already a sub-rectangle where the
    // dispatch writes one, and it is the same window the readback rail lands
    // through — so the two rails cannot name different bytes of one surface.
    match crate::runtime::mapping_write::vulkan::licence_mapper_ref_texture_surface(
        state,
        host,
        held,
        &crate::runtime::mapping_write::vulkan::MapperRefTextureSurfaceDestination {
            mapping_id: *mapping_id,
            base_off: *surface_offset,
            bpr: *surface_bpr,
            span_end: *span_end,
            width: *width,
            height: *height,
            format: *format,
        },
    ) {
        Ok(licence) => {
            if let Some((need, have)) = licence.target.window_shortfall() {
                if crate::observe::first_sight(
                    "compute_dst_window_too_small",
                    u64::from(tex.binding),
                ) {
                    crate::observe::fail(format!(
                        "compute_dst_window_too_small bind={} mid={mapping_id} dims={width}x{height} \
                         need={need} have={have} (the storage image names more texels than its \
                         guest pages hold; a direct landing would write past them, so it takes \
                         the pooled readback)",
                        tex.binding
                    ));
                }
                crate::runtime::drain::note_store_route("compute_dst_host_window_too_small");
                return ComputeImageDestination::Host;
            }
            crate::runtime::drain::note_store_route("compute_dst_guest_pages");
            crate::runtime::drain::note_store_route(if tex.rail.residency.is_some() {
                "compute_dst_guest_pages_mapper_ref_texture_resident"
            } else {
                "compute_dst_guest_pages_mapper_ref_texture_transient"
            });
            ComputeImageDestination::GuestPages {
                target: Box::new(licence.target),
                pages: licence.gpas,
            }
        }
        Err(decline) => {
            // Named, because the reasons are not interchangeable and the route
            // counter cannot tell them apart. The one that dominates is
            // `ResidentFormatMismatch` — a storage image's format comes from the
            // specialized SPIR-V texel format and owes the mapping's declaration
            // nothing — and no copy can serve those, so a boot where it is most
            // of this counter is this arm working rather than failing.
            crate::observe::off(format!(
                "compute_dst_mapper_ref_texture bind={} mid={mapping_id} dims={width}x{height} held={held:?} reason={decline:?}",
                tex.binding
            ));
            crate::runtime::drain::note_store_route(
                "compute_dst_host_mapper_ref_texture_unlicensed",
            );
            ComputeImageDestination::Host
        }
    }
}

/// This rail's half of a staged compute texture. See [`RailStage`].
///
/// [`RailStage`]: crate::runtime::compute_exec::RailStage
#[derive(Debug, Default)]
pub(crate) struct VulkanStage {
    /// Which element of the descriptor binding's array this fills.
    pub(crate) array_element: u32,
    /// How many descriptors the binding declares.
    pub(crate) descriptor_count: u32,
    /// The storage-mirror window this staging corresponds to, so the writeback
    /// can register the resident it produced under the same key.
    pub(crate) residency: Option<ComputeStorageResidencyCandidate>,
    /// What the engine could already serve for this binding, so the stage-time
    /// guest read was skipped and `bytes` is a zero placeholder.
    ///
    /// [`ResidentServe::Seed`] — a storage binding whose resident the engine
    /// holds at a verified generation; it must never be seeded from the
    /// placeholder. [`ResidentServe::Sample`] — a sampled input whose window is
    /// a prior dispatch's storage output; the engine seeds the sampled image by
    /// copy-on-sample from that resident, again never from the bytes.
    ///
    /// One field rather than the `bool` and `Option` pair it replaces: those
    /// were the variant tag and the payload of this enum stored apart, so every
    /// producer had to rebuild both halves and nothing made a producer that set
    /// one without the other fail to compile.
    pub(crate) serve: Option<ResidentServe>,
    /// The retained multisample render target this binding is served from.
    ///
    /// Set only for a kernel-declared `texture2d_ms<T, access::read>`, and it is
    /// exclusive with everything above it by construction: `bytes` is empty,
    /// `serve` is `None`, and nothing is staged, because
    /// `engine::types::SampledResource::multisampled` says linear bytes cannot
    /// be uploaded into a multisample image at all. The engine binds this
    /// target's own view.
    pub(crate) multisample_target: Option<crate::backend::vulkan::engine::TargetIdentity>,
}

impl RailStage for VulkanStage {
    /// The guest ref is not kept: this rail reaches its images through the
    /// engine's own registry and never names the object the guest bound.
    fn stage(
        _texture_ref: u32,
        residency: Option<ComputeStorageResidencyCandidate>,
        serve: Option<ResidentServe>,
    ) -> Self {
        Self {
            array_element: 0,
            descriptor_count: 1,
            residency,
            serve,
            multisample_target: None,
        }
    }
}

/// Bound on mirror entries per mapping: a ping-pong canvas needs 2, planar
/// layouts a few more; anything beyond is assumed to be stale-key debris.
///
/// **The 8 is not derived, and the eviction below is the only thing standing
/// between this map and unbounded growth.** If every stale key were already
/// invalidated — `invalidate_storage_residency_window` runs on every overlap,
/// and every guest-page writer calls it — no cap would be needed at all, and
/// this would be a mechanism covering for an incomplete invalidation rather
/// than a bound. Which of those it is has never been measured, because the
/// eviction was silent.
///
/// `compute_mirror_evicted` is that measurement, and its **first reading is
/// zero**: one driven x86/Vulkan boot — Chess, Maps, the WebGL aquarium,
/// Wikipedia and apple.com, with page-downs and title-bar drags — evicted
/// nothing. So the cap does not bind on this workload and is a runaway guard,
/// not a working policy.
///
/// That is one boot and one workload, which is not enough to delete a guard
/// that is the only bound on this map. What would be: the same zero across a
/// boot that drives multiplanar video and several ping-pong canvases at once,
/// which is the case the "planar layouts a few more" guess was aimed at.
///
/// How close the population came was measured separately, and reads **2**
/// across every boot in a 72 MB accumulated log. That is exactly the ping-pong
/// canvas this doc predicted needs 2, so the shape of the guess is confirmed
/// while the number is not: 8 is 4x the observed high-water mark, and nothing
/// has yet produced the "planar layouts a few more" case that chose it.
pub(super) const STORAGE_RESIDENCY_WINDOWS_PER_MAPPING: usize = 8;

pub(super) fn note_storage_residency_writeback(
    state: &mut DeviceState,
    texture: &StagedTexture<VulkanStage>,
) {
    let Some(candidate) = texture.rail.residency else {
        return;
    };
    // Linear windows keep their authority in the host_linear_textures entry
    // (resident_gen), never in the mapping-keyed mirror.
    if candidate.key.is_linear() {
        return;
    }
    if candidate.key.is_heap() {
        state.compute_storage_residency.insert(
            candidate.key,
            next_mapping_content_generation(candidate.seed_generation),
        );
        return;
    }
    // The engine registered the resident at exactly next(seed_generation)
    // (ComputeStorageResidency::output_generation). The mirror must store the
    // same currency — not the mapping-level content generation — so disjoint
    // sibling-window writebacks (ping-pong canvases) cannot desync the pair.
    let generation = next_mapping_content_generation(candidate.seed_generation);
    // Drop intersecting windows (normally already gone, because the writeback
    // wrote guest pages and every guest-page writer calls the same overlap
    // invalidation — kept here as defense in depth); keep disjoint siblings
    // (ping-pong canvases) but bound the count.
    let mapping_id = candidate.key.mapping_id;
    state.invalidate_storage_residency_window(
        mapping_id,
        candidate.key.surface_offset,
        candidate.key.span_end,
    );
    let siblings: Vec<crate::model::ComputeStorageResidencyKey> = state
        .compute_storage_residency
        .keys()
        .filter(|key| key.mapping_id == mapping_id && **key != candidate.key)
        .cloned()
        .collect();
    // Counting the window inserted below, so the cap bounds the population this
    // mapping actually holds.
    let over_cap = (siblings.len() + 1).saturating_sub(STORAGE_RESIDENCY_WINDOWS_PER_MAPPING);
    for victim in siblings.iter().take(over_cap) {
        state.compute_storage_residency.remove(victim);
        // Dropping a mirror entry costs the next read of that window its
        // resident and sends it back to guest pages. That is safe, but it is
        // not free and it must not be invisible.
        crate::observe::off(format!(
            "compute_mirror_evicted mid={mapping_id} off={} end={} siblings={} cap={}",
            victim.surface_offset,
            victim.span_end,
            siblings.len(),
            STORAGE_RESIDENCY_WINDOWS_PER_MAPPING
        ));
    }
    state
        .compute_storage_residency
        .insert(candidate.key, generation);
}

pub(super) fn next_mapping_content_generation(current: u32) -> u32 {
    let next = current.wrapping_add(1);
    if next == 0 {
        1
    } else {
        next
    }
}

/// Measure storage-image seed traffic by structurally reflected content access.
///
/// `write_only` is intentionally still seeded: access alone does not prove a
/// dispatch overwrites every texel. The proxy makes that retained transfer
/// cost visible while preserving partial-write semantics.
pub(super) fn log_storage_image_access(pipe: u32, binding: u32, access: &str, bytes: u64) {
    crate::observe::off(format!(
        "compute_linux storage_access pipe={pipe} bind={binding} access={access} seed=1 bytes={bytes}"
    ));
}

/// The gate every staging rail applies before falling back to a guest read.
///
/// `mirror_generation` is the runtime's residency mirror for `key`; the engine
/// must agree with it, because the mirror can outlive an evicted resident. A
/// sampled binding additionally needs the resident's vk format to equal the one
/// the view will bind — the engine's resident-bind path guards that equality
/// and would fail the whole request on mismatch.
///
/// `None` means the resident cannot serve this binding: the caller either reads
/// the guest window or, where the resident is the only copy, names the loss.
pub(crate) fn resident_serve(
    key: crate::model::ComputeStorageResidencyKey,
    mirror_generation: u32,
    is_storage: bool,
    pixel_format: u16,
) -> Option<ResidentServe> {
    if is_storage {
        return (crate::backend::vulkan::engine::compute_resident_storage_generation(&key)
            == Some(mirror_generation))
        .then_some(ResidentServe::Seed(mirror_generation));
    }
    let (engine_generation, engine_format) =
        crate::backend::vulkan::engine::compute_resident_sample_source(&key)?;
    (engine_generation == mirror_generation
        && mtl_to_engine_sampled(pixel_format)
            .is_some_and(|f| f.vk_format() == engine_format.vk_format()))
    .then_some(ResidentServe::Sample(key, mirror_generation))
}

/// Linux product compute path (doorbell / BQL).
///
/// Stages buffers/textures with device `page_shift`, translates the kernel AIR
/// via [`crate::runtime::m2v_cache::translate_cached_kernel_reflected`], dispatches on the
/// process-global [`crate::backend::vulkan::engine`] (shared GRAPHICS|COMPUTE
/// device), then writebacks GVA / mapper-ref-texture.
///
/// Nested/ICB/stage-in stay Unsupported (engine surface is storage buffers +
/// storage images only).
pub(crate) fn execute_dispatch_linux<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    acc: &ComputeAccum,
    dispatch: &DispatchRecord,
) -> ComputeStatus {
    use crate::backend::vulkan::engine::{
        self as vk_engine, ComputeBufferResource, ComputeImageResult, ComputeRequest,
        ComputeSampledImageResource, ComputeSampledSource, ComputeStorageImageResource, DrawError,
    };

    if acc.pipeline_ref == 0 {
        return ComputeStatus::MissingPipeline("compute_vk_pipeline_ref_zero");
    }
    let Some(pipeline) = load_compute_pipeline(state, host, task_id, acc.pipeline_ref) else {
        return ComputeStatus::MissingPipeline("compute_vk_pipeline_load");
    };
    if let Some(stage_input) = pipeline.stage_input.as_ref() {
        if crate::observe::first_sight("compute_stage_input_contract", u64::from(acc.pipeline_ref))
        {
            crate::observe::off(format!(
                "compute_stage_input_contract pipe={} attrs={:?} layouts={:?} index_type={} \
                 index_buffer={}",
                acc.pipeline_ref,
                stage_input.attributes,
                stage_input.layouts,
                stage_input.index_type,
                stage_input.index_buffer_index,
            ));
        }
    }
    // A stage-in region this rail proceeds past — see
    // `linux_stage_input_or_imageblock_unsupported`, which explains why that is
    // lossless on a pipeline with no stage input. Counted rather than assumed.
    if acc.stage_in_region.is_some() || acc.stage_in_region_indirect.is_some() {
        crate::runtime::drain::note_store_route("compute_stage_in_region_unused");
    }
    if linux_stage_input_or_imageblock_unsupported(pipeline.stage_input.is_some(), acc) {
        crate::observe::fail(format!(
            "compute_linux unsupported pipe={} stage_in_desc={} stage_in_direct={} \
             stage_in_indirect={} imageblock={} (need SPI parity)",
            acc.pipeline_ref,
            pipeline.stage_input.is_some() as u8,
            acc.stage_in_region.is_some() as u8,
            acc.stage_in_region_indirect.is_some() as u8,
            acc.imageblock.is_some() as u8
        ));
        return ComputeStatus::Unsupported("linux_stage_in_imageblock");
    }
    // Dims first (cheap; proves sentinel recovery without m2v/vk).
    let DispatchDims {
        grid,
        threadgroup: tg,
        dispatch_threads,
    } = match resolve_dispatch_dims_reported(state, host, task_id, dispatch, acc) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let (grid_x, grid_y, grid_z) = (grid.x, grid.y, grid.z);
    let (tg_x, tg_y, tg_z) = (tg.x, tg.y, tg.z);
    // Resolved here, before any staging, so a record with no work costs nothing
    // — but computed by the same function that refuses the zero, because the two
    // are one rule. See [`crate::protocol::dispatch::workgroup_counts`] for why
    // splitting them put an unreachable `.max(1)` on the quotients.
    let Some([wg_x, wg_y, wg_z]) = crate::protocol::dispatch::workgroup_counts(
        [grid_x, grid_y, grid_z],
        [tg_x, tg_y, tg_z],
        dispatch_threads,
    ) else {
        return ComputeStatus::BadGrid("compute_vk_zero_dims");
    };

    // Translate before staging buffers. The final adopted SPIR-V carries the
    // conservative byte footprint that decides how much of each allocation the
    // dispatch can touch; staging first discarded that answer and copied every
    // bind through the end of its allocation.
    //
    // MTLB → AIR → SPIR-V (LocalSize = threadgroup dims).
    let Some(mtlb) = load_mtlb(
        state,
        host,
        task_id,
        pipeline.kernel_func_ref,
        AirLoadRail::Compute,
    ) else {
        return ComputeStatus::MissingMtlb("compute_vk_mtlb_load");
    };
    // The function blob is an MTLB container; llvm-dis needs the wrapped AIR
    // bitcode member (same extract the render path does — passing the raw
    // container was the live `llvm-dis: file doesn't start with bitcode
    // header` MetalFailed class).
    let air = match crate::runtime::mtlb::extract_air(&mtlb) {
        Ok(a) => a,
        Err(e) => {
            crate::observe::Emit::decline("compute_linux_air_extract", &e)
                .field("pipe", acc.pipeline_ref)
                .fail_once(acc.pipeline_ref as u64);
            return ComputeStatus::MetalFailed("compute_vk_air_extract");
        }
    };
    let kernel_shader = match crate::runtime::m2v_cache::translate_cached_kernel_reflected(
        air,
        [tg_x, tg_y, tg_z],
        acc.pipeline_ref,
    ) {
        Ok(b) => b,
        Err(e) => {
            // What the ordering plane thought of this pipeline when the
            // dispatch was refused --- the compute twin of the render arm's
            // field in `draw::vulkan`, and it is here because that field is
            // what made the render losses legible and this one had no
            // equivalent. A driven macos-26 desktop leaves three compute
            // refusals that cannot be classified without it: `ready` is the
            // ordering plane releasing work against a shader still being
            // translated, `retired` is a packet recorded before the guest's
            // delete and a loss this device chose to take, and they are
            // opposite conclusions from the same line.
            let model_pipeline = state
                .object_name(task_id, acc.pipeline_ref)
                .map_or("unnamed", |id| {
                    state.pipeline_state(id).unwrap_or("unleased")
                });
            crate::observe::Emit::decline("compute_linux_m2v", &e)
                .field("pipe", acc.pipeline_ref)
                .field("model_pipeline", model_pipeline)
                .fail_once(acc.pipeline_ref as u64);
            return ComputeStatus::MetalFailed("compute_vk_translate");
        }
    };
    if let Some(unsupported) = crate::runtime::spirv_bind::first_unsupported_vulkan_interface(
        &kernel_shader.reflection,
        metal2vulkan::reflect::ShaderStage::Kernel,
    ) {
        let reason = ComputeReflectionDecline::ReflectedInterfaceUnsupported {
            pipeline_ref: acc.pipeline_ref,
            feature: unsupported.feature,
            count: unsupported.count,
        };
        crate::observe::Emit::decline("compute_linux_reflection", &reason)
            .fail_once(u64::from(acc.pipeline_ref));
        return ComputeStatus::Unsupported(crate::observe::Decline::slug(&reason));
    }
    if let Some(resource) =
        crate::runtime::spirv_bind::first_unsupported_vulkan_resource(&kernel_shader.reflection)
    {
        let kind = crate::runtime::spirv_bind::unsupported_vulkan_resource_kind_name(resource.kind)
            .expect("helper returned an unsupported Vulkan resource");
        let reason = ComputeReflectionDecline::ReflectedResourceUnsupported {
            pipeline_ref: acc.pipeline_ref,
            index: resource.metal_index,
            binding: resource.descriptor.map(|descriptor| descriptor.binding),
            kind,
        };
        crate::observe::Emit::decline("compute_linux_reflection", &reason)
            .fail_once((u64::from(acc.pipeline_ref) << 32) | u64::from(resource.metal_index));
        return ComputeStatus::Unsupported(crate::observe::Decline::slug(&reason));
    }
    let reflected_local_size = kernel_shader
        .reflection
        .local_size
        .expect("kernel cache admits only the requested reflected local size");
    let Some(kernel_dispatch) = kernel_shader.reflection.kernel_dispatch else {
        return ComputeStatus::Unsupported("compute_kernel_dispatch_missing");
    };
    let dispatch = match kernel_dispatch_launch(
        kernel_dispatch,
        reflected_local_size,
        [wg_x, wg_y, wg_z],
        [tg_x, tg_y, tg_z],
        dispatch_threads.then_some([grid_x, grid_y, grid_z]),
    ) {
        Ok(dispatch) => dispatch,
        Err(decline) => {
            let (status, detail) = match &decline {
                KernelDispatchDecline::GridOverflow => {
                    (ComputeStatus::BadGrid("compute_vk_grid_overflow"), None)
                }
                KernelDispatchDecline::PushRangeUnavailable => (
                    ComputeStatus::Unsupported("compute_kernel_dispatch_push_range"),
                    None,
                ),
                KernelDispatchDecline::PlanRefused(detail) => (
                    ComputeStatus::BadGrid("compute_vk_dispatch_plan"),
                    Some(detail.replace(char::is_whitespace, "_")),
                ),
            };
            let reason = decline.reason(acc.pipeline_ref);
            let mut emit = crate::observe::Emit::decline("compute_linux_kernel_dispatch", &reason);
            if let Some(detail) = detail {
                emit = emit.field("detail", detail);
            }
            emit.fail_once(u64::from(acc.pipeline_ref));
            return status;
        }
    };
    let mut spirv = match spirv_words_le(&kernel_shader.spirv) {
        Ok(w) => w,
        Err(e) => {
            crate::observe::Emit::decline("compute_linux_spirv_parse", &e)
                .field("pipe", acc.pipeline_ref)
                .fail_once(acc.pipeline_ref as u64);
            return ComputeStatus::MetalFailed("compute_vk_spirv_parse");
        }
    };

    // Stage buffers only after translation has published the final-module
    // footprint and access. No Vulkan work occurs until every declared resource
    // has staged successfully. A bind reflection calls `Unused` or does not
    // declare is skipped before resolving its descriptor, walking its pages, or
    // allocating its staging Vec.
    let mut staged_bufs: Vec<StagedBuffer> = Vec::new();
    let mut buffer_accesses = Vec::with_capacity(acc.buffers.len());
    let mut buffer_readonly_count = 0usize;
    let mut buffer_writable_count = 0usize;
    let mut buffer_unused_count = 0usize;
    let mut buffer_absent_count = 0usize;
    let mut buffer_unknown_count = 0usize;
    for b in &acc.buffers {
        use crate::runtime::spirv_bind::ReflectedBufferAccess;
        let access =
            crate::runtime::spirv_bind::reflected_buffer_access(&kernel_shader.reflection, b.index);
        let writable = match access {
            ReflectedBufferAccess::Unused => {
                buffer_unused_count += 1;
                crate::runtime::drain::note_store_route("compute_buffer_unused_reflected");
                continue;
            }
            ReflectedBufferAccess::Absent => {
                buffer_unused_count += 1;
                buffer_absent_count += 1;
                crate::runtime::drain::note_store_route("compute_buffer_absent_reflected");
                continue;
            }
            ReflectedBufferAccess::ReadOnly => {
                buffer_readonly_count += 1;
                false
            }
            ReflectedBufferAccess::Writable => {
                buffer_writable_count += 1;
                true
            }
            ReflectedBufferAccess::Unknown => {
                // A declared descriptor with no access answer stays on the
                // conservative read/write arm. The per-translate reflection
                // guard names the malformed fact; this count shows how often a
                // dispatch had to pay for it.
                buffer_writable_count += 1;
                buffer_unknown_count += 1;
                true
            }
        };
        let extent = crate::runtime::spirv_bind::reflected_compute_buffer_extent(
            &kernel_shader.reflection,
            b.index,
            [wg_x, wg_y, wg_z],
            reflected_local_size,
        );
        match stage_buffer_with_extent(state, host, task_id, b, extent) {
            Ok(s) => {
                buffer_accesses.push((b.index, writable));
                staged_bufs.push(s);
            }
            Err(e) => {
                // `st={e:?}` alone was not greppable: the Debug spelling was
                // the only handle on which of stage_buffer's checks refused.
                // `reason=` names it.
                crate::observe::fail(format!(
                    "compute_linux stage_buf fail reason={} pipe={} idx={} ref={} off={:#x} class={}",
                    e.reason(),
                    acc.pipeline_ref,
                    b.index,
                    b.buffer_ref,
                    b.offset,
                    e.class()
                ));
                return e;
            }
        }
    }

    let mut staged_tex: Vec<StagedTexture<VulkanStage>> = Vec::new();
    let mut storage_writeonly_count = 0usize;
    for t in &acc.textures {
        use crate::runtime::spirv_bind::{
            ImageAccess, ReflectedComputeTexture, StorageImageAccess,
        };
        let Some(descriptor) = crate::runtime::spirv_bind::reflected_texture_descriptor(
            &kernel_shader.reflection,
            t.index,
        ) else {
            crate::observe::line(format!(
                "compute_linux texture_unused pipe={} i={} ref={}",
                acc.pipeline_ref, t.index, t.texture_ref
            ));
            continue;
        };
        let binding = descriptor.binding;
        // Both the sampled-vs-storage class and the shape come solely from the
        // translator's reflection — the declared Metal texture type, exact at
        // translate time. The always-on `census_reflection_wellformed` guard
        // proves the reflection is internally consistent per translate.
        let is_storage = match crate::runtime::spirv_bind::reflected_compute_texture(
            &kernel_shader.reflection,
            binding,
        ) {
            ReflectedComputeTexture::Plain2d(ImageAccess::Sampled) => false,
            ReflectedComputeTexture::Plain2d(ImageAccess::Storage) => true,
            ReflectedComputeTexture::Multisample2d => {
                // Not staged, because there is nothing to stage from and
                // nothing to stage into: a multisample image is filled by
                // rendering and by nothing else. The retained target that
                // rendered those samples is the whole source, so this arm
                // resolves it and skips `stage_texture_raw` entirely.
                match multisample_sampled_texture(
                    state,
                    host,
                    task_id,
                    acc.pipeline_ref,
                    t.texture_ref,
                    binding,
                    descriptor,
                ) {
                    Ok(staged) => {
                        staged_tex.push(staged);
                        continue;
                    }
                    // Already fail-visible, by the name of the rung that
                    // refused; the caller must not print a second line for one
                    // refusal.
                    Err(status) => return status,
                }
            }
            ReflectedComputeTexture::Absent => {
                // Metal permits unused bound resources. If reflection lists no
                // texture shape at this binding, the shader does not sample/write
                // it — do not stage or invent access/writeback semantics for it.
                crate::observe::line(format!(
                    "compute_linux texture_unused pipe={} i={} ref={} bind={}",
                    acc.pipeline_ref, t.index, t.texture_ref, binding
                ));
                continue;
            }
            ReflectedComputeTexture::UnstageableShape { axis } => {
                // The rail stages one flat plane window or one linear GVA level
                // per binding, so it can only ever produce a single-layer 2D
                // image. Binding that to a shader image declared with a slice,
                // depth, or sample axis is a descriptor-type mismatch — refuse
                // by name instead of dispatching against the wrong view.
                crate::observe::fail(format!(
                    "compute_linux texture_shape fail reason=unstageable_{axis} pipe={} i={} ref={} bind={binding}",
                    acc.pipeline_ref, t.index, t.texture_ref
                ));
                return ComputeStatus::Unsupported("texture_shape_unstageable");
            }
        };
        let storage_access = if is_storage {
            match crate::runtime::spirv_bind::storage_image_access(&spirv, binding) {
                Some(StorageImageAccess::WriteOnly) => Some("write_only"),
                Some(StorageImageAccess::ReadOnly) => Some("read_only"),
                Some(StorageImageAccess::ReadWrite) => Some("read_write"),
                Some(StorageImageAccess::Unknown) => Some("unknown"),
                Some(StorageImageAccess::AmbiguousBinding) => {
                    crate::observe::fail(format!(
                        "compute_linux texture_access fail reason=spirv_storage_ambiguous_binding pipe={} i={} ref={} bind={binding}",
                        acc.pipeline_ref, t.index, t.texture_ref
                    ));
                    return ComputeStatus::Unsupported("texture_spirv_storage_ambiguous_binding");
                }
                None => {
                    crate::observe::fail(format!(
                        "compute_linux texture_access fail reason=spirv_storage_access_missing pipe={} i={} ref={} bind={binding}",
                        acc.pipeline_ref, t.index, t.texture_ref
                    ));
                    return ComputeStatus::Unsupported("texture_spirv_storage_access_missing");
                }
            }
        } else {
            None
        };
        match stage_texture_raw::<VulkanStage, _>(
            state,
            host,
            task_id,
            t.texture_ref,
            binding,
            is_storage,
        ) {
            Ok(mut s) => {
                s.rail.array_element = descriptor.array_element;
                s.rail.descriptor_count = descriptor.descriptor_count;
                if let Some(storage_access) = storage_access {
                    if storage_access == "write_only" {
                        storage_writeonly_count += 1;
                    }
                    let bytes = (s.width as u64)
                        .saturating_mul(s.height as u64)
                        .saturating_mul(
                            pixel_format::bytes_per_pixel(s.pixel_format).unwrap_or(0) as u64
                        );
                    log_storage_image_access(acc.pipeline_ref, binding, storage_access, bytes);
                }
                staged_tex.push(s);
            }
            Err(e) => {
                let ot = objects::lookup_list_entry(state, host, task_id, t.texture_ref)
                    .map(|en| en.object_type)
                    .unwrap_or(0);
                crate::observe::fail(format!(
                    "compute_linux stage_tex fail reason={} pipe={} i={} ref={} ot={} bind={} access={} class={}",
                    e.reason(),
                    acc.pipeline_ref,
                    t.index,
                    t.texture_ref,
                    ot,
                    binding,
                    if is_storage { "storage" } else { "sampled" },
                    e.class()
                ));
                return e;
            }
        }
    }

    let mut sampled_count = 0usize;
    let mut storage_count = 0usize;
    for t in &staged_tex {
        if t.is_storage {
            storage_count += 1;
        } else {
            sampled_count += 1;
        }
    }
    // A dispatch that staged its resources is expected control flow; the
    // refusals on this path each emit their own typed decline.
    crate::observe::line(format!(
        "compute_linux stage_ok pipe={} nbuf={} bro={} brw={} bunused={} babsent={} bunknown={} ntex={} sampled={} storage={} swo={} grid=[{grid_x},{grid_y},{grid_z}] tg=[{tg_x},{tg_y},{tg_z}] encode=engine",
        acc.pipeline_ref,
        staged_bufs.len(),
        buffer_readonly_count,
        buffer_writable_count,
        buffer_unused_count,
        buffer_absent_count,
        buffer_unknown_count,
        staged_tex.len(),
        sampled_count,
        storage_count,
        storage_writeonly_count,
    ));

    let mut storage_buffers = Vec::with_capacity(buffer_accesses.len());
    for s in &mut staged_bufs {
        let Some((_, writable)) = buffer_accesses
            .iter()
            .find(|(binding, _)| *binding == s.bind.index)
        else {
            continue;
        };
        storage_buffers.push(ComputeBufferResource {
            binding: s.bind.index,
            bytes: std::mem::take(&mut s.bytes),
            writable: *writable,
        });
    }
    let mut sampled_images = Vec::with_capacity(sampled_count);
    let mut storage_images = Vec::with_capacity(storage_count);
    let mut storage_formats = Vec::with_capacity(storage_count);
    // Device support for format-less storage writes decides whether a guest
    // BGRA8Unorm storage surface can composite into a B8G8R8A8_UNORM view (no
    // R/B swap) or must degrade to the swapped Rgba8Unorm view.
    let write_without_format = vk_engine::supports_storage_image_write_without_format();
    for t in staged_tex.iter().filter(|texture| texture.is_storage) {
        let Some(selector) = t.storage_selector else {
            crate::observe::fail(format!(
                "compute_linux unsupported storage_format reason=no_storage_selector pipe={} bind={} fmt={:#x}",
                acc.pipeline_ref, t.binding, t.pixel_format
            ));
            return ComputeStatus::Unsupported("storage_no_selector_specialize");
        };
        let guest_fmt = selector_to_engine_storage(selector);
        let Some(shader_decl) = crate::runtime::spirv_bind::reflected_storage_image_format(
            &kernel_shader.reflection,
            t.binding,
        ) else {
            crate::observe::fail(format!(
                "compute_linux storage_format fail reason=reflection_format_missing pipe={} bind={} guest={guest_fmt:?} simg={}",
                acc.pipeline_ref, t.binding, selector as u32
            ));
            return ComputeStatus::Unsupported("storage_reflection_format_missing");
        };
        let specialized = match specialized_storage_image_format(
            guest_fmt,
            shader_decl,
            write_without_format,
        ) {
            Ok(format) => format,
            Err(reason) => {
                crate::observe::fail(format!(
                        "compute_linux storage_format fail reason={reason} pipe={} bind={} spirv={shader_decl:?} guest={guest_fmt:?} simg={} guest_bpp={} shader_bpp={}",
                        acc.pipeline_ref,
                        t.binding,
                        selector as u32,
                        guest_fmt.bytes_per_texel(),
                        spirv_image_format_to_engine_storage(shader_decl)
                            .map(|format| format.bytes_per_texel())
                            .unwrap_or(0)
                    ));
                return ComputeStatus::Unsupported("storage_format_specialize_mismatch");
            }
        };
        storage_formats.push((t.binding, guest_fmt, shader_decl, specialized));
    }
    let specialization_requests: Vec<_> = storage_formats
        .iter()
        .map(|(binding, _, _, specialized)| (*binding, *specialized))
        .collect();
    if let Err(error) =
        crate::runtime::spirv_bind::specialize_image_formats(&mut spirv, &specialization_requests)
    {
        let error: crate::runtime::spirv_bind::ImageFormatSpecializeError = error;
        crate::observe::Emit::decline("compute_linux_storage_format", &error)
            .field("pipe", acc.pipeline_ref)
            .fail();
        return ComputeStatus::Unsupported("storage_format_specialize_error");
    }
    // A guest BGRA8Unorm storage surface retargets to an `Unknown`-format
    // storage image (viewed B8G8R8A8_UNORM) so the composite writes land in the
    // guest's channel order — that write is only legal if the module declares
    // `StorageImageWriteWithoutFormat`. Inject it once when any binding took the
    // Unknown path (idempotent; the translator declares only Shader/Float16/…).
    if storage_formats.iter().any(|(_, _, _, specialized)| {
        matches!(
            specialized,
            crate::runtime::spirv_bind::ImageFormat::Unknown
        )
    }) {
        crate::runtime::spirv_bind::ensure_storage_write_without_format_capability(&mut spirv);
    }
    // Compute-side analog of the render resident gates: a deferred storage
    // writeback leaves guest-visible bytes GPU-resident-only until a flush
    // choke point lands them, so it requires the device's
    // `deferred_gpu_only_content` capability (off on portability-subset /
    // MoltenVK, where guest pages stay authoritative and the writeback runs
    // synchronously in this call).
    for t in &mut staged_tex {
        if t.is_storage {
            let Some(selector) = t.storage_selector else {
                crate::observe::fail(format!(
                    "compute_linux unsupported storage_format reason=no_storage_selector pipe={} bind={} fmt={:#x}",
                    acc.pipeline_ref, t.binding, t.pixel_format
                ));
                return ComputeStatus::Unsupported("storage_no_selector_writeback");
            };
            let guest_fmt = selector_to_engine_storage(selector);
            let Some((_, _, shader_decl, specialized)) = storage_formats
                .iter()
                .find(|(binding, _, _, _)| *binding == t.binding)
            else {
                crate::observe::fail(format!(
                    "compute_linux storage_format fail reason=spirv_format_specialize_internal pipe={} bind={} simg={}",
                    acc.pipeline_ref, t.binding, selector as u32
                ));
                return ComputeStatus::Unsupported("storage_format_specialize_internal");
            };
            // An `Unknown`-format storage image carries no SPIR-V texel format;
            // its engine format (and thus VkImageView) is the guest surface's
            // own format — here BGRA8Unorm → B8G8R8A8_UNORM — so the composite
            // write lands in guest channel order (the R/B-swap fix). Every other
            // format takes its engine format from the specialized SPIR-V format.
            let shader_fmt = if matches!(
                specialized,
                crate::runtime::spirv_bind::ImageFormat::Unknown
            ) {
                // Always-on proxy for the BGRA-storage-composite R/B class: this
                // line fires only on the corrected (without_format) path. Its
                // absence together with a `degraded_rb_swap` line below is the
                // regression signal that a swap is being emitted.
                crate::observe::off(format!(
                    "compute_linux bgra_storage_composite pipe={} bind={} mode=without_format guest={guest_fmt:?} view=B8G8R8A8_UNORM {}x{}",
                    acc.pipeline_ref, t.binding, t.width, t.height
                ));
                guest_fmt
            } else {
                let Some(fmt) = spirv_image_format_to_engine_storage(*specialized) else {
                    crate::observe::fail(format!(
                        "compute_linux storage_format fail reason=spirv_storage_format_unsupported pipe={} bind={} spirv={specialized:?} guest={guest_fmt:?} simg={}",
                        acc.pipeline_ref, t.binding, selector as u32
                    ));
                    return ComputeStatus::Unsupported("storage_spirv_format_unsupported");
                };
                // Degraded path: a BGRA8Unorm guest fell back to a Rgba8Unorm
                // view because `shaderStorageImageWriteWithoutFormat` is absent —
                // the composite output is R/B-swapped. Fail-visible so the class
                // is never silent on an unsupported device.
                if matches!(
                    guest_fmt,
                    crate::backend::vulkan::engine::StorageImageFormat::Bgra8Unorm
                ) && matches!(
                    fmt,
                    crate::backend::vulkan::engine::StorageImageFormat::Rgba8Unorm
                ) {
                    crate::observe::fail(format!(
                        "compute_linux bgra_storage_composite pipe={} bind={} mode=degraded_rb_swap reason=no_storage_image_write_without_format {}x{}",
                        acc.pipeline_ref, t.binding, t.width, t.height
                    ));
                }
                fmt
            };
            if specialized != shader_decl {
                crate::observe::off(format!(
                    "compute_linux storage_format_specialize pipe={} bind={} spirv={shader_decl:?} specialized={specialized:?} engine={shader_fmt:?} guest={guest_fmt:?} simg={} guest_bpp={} shader_bpp={}",
                    acc.pipeline_ref,
                    t.binding,
                    selector as u32,
                    guest_fmt.bytes_per_texel(),
                    spirv_image_format_to_engine_storage(*shader_decl)
                        .map(|format| format.bytes_per_texel())
                        .unwrap_or(0)
                ));
            }
            storage_images.push(ComputeStorageImageResource {
                binding: t.binding,
                array_element: t.rail.array_element,
                descriptor_count: t.rail.descriptor_count,
                format: shader_fmt,
                width: t.width,
                height: t.height,
                bytes: std::mem::take(&mut t.bytes),
                // The guest window this output belongs to is on `t.writeback`,
                // so the destination is decided from the window rather than
                // from anything about this dispatch. `Host` needs no host
                // capability; the direct arm needs the guest-RAM import, and
                // where that is absent the licence declines by name and this
                // reads back exactly as it always did.
                destination: direct_destination(state, host, t, shader_fmt.vk_format()),
                residency: t.rail.residency.map(|candidate| {
                    crate::backend::vulkan::engine::ComputeStorageResidency {
                        identity: candidate.key,
                        seed_generation: candidate.seed_generation,
                        output_generation: next_mapping_content_generation(
                            candidate.seed_generation,
                        ),
                    }
                }),
                seed_skipped: t
                    .rail
                    .serve
                    .and_then(ResidentServe::seed_generation)
                    .is_some(),
            });
        } else {
            let Some(sampled_fmt) = mtl_to_engine_sampled(t.pixel_format) else {
                crate::observe::fail(format!(
                    "compute_linux sampled_format fail reason=mtl_format_unsupported pipe={} bind={} fmt={:#x}",
                    acc.pipeline_ref, t.binding, t.pixel_format
                ));
                return ComputeStatus::Unsupported("sampled_format_unsupported");
            };
            sampled_images.push(ComputeSampledImageResource {
                binding: t.binding,
                array_element: t.rail.array_element,
                descriptor_count: t.rail.descriptor_count,
                format: sampled_fmt,
                width: t.width,
                height: t.height,
                mip_levels: t.mip_levels,
                // Asked in the order the sources exclude each other, not as a
                // pair: the producer that sets `multisample_target` is the one
                // rail that stages nothing, and it leaves `serve` and `bytes`
                // empty because there is nothing for either to hold.
                source: match t.rail.multisample_target.take() {
                    Some(identity) => ComputeSampledSource::MultisampleTarget(identity),
                    None => match t.rail.serve.and_then(ResidentServe::sample_source) {
                        Some((identity, generation)) => ComputeSampledSource::ResidentCopy(
                            crate::backend::vulkan::engine::ComputeResidentSampleBind {
                                identity,
                                generation,
                            },
                        ),
                        None => ComputeSampledSource::Bytes(std::mem::take(&mut t.bytes)),
                    },
                },
            });
        }
    }

    // Vulkan requires the pipeline layout to contain a descriptor for every
    // resource the module *statically uses*. The layout this device builds is
    // assembled from what the guest bound, so a texture the kernel samples and
    // the guest left empty is absent from the layout entirely — not an unwritten
    // slot in it. That is undefined behaviour by the specification and it is
    // worse than that in practice: Mesa's Intel driver sizes its binding array
    // to `max_binding + 1`, zero-fills every number nothing declared, and scores
    // each used binding as `(use_count << 7) / array_size` when it picks
    // binding-table slots. A hole under a used binding divides by zero, so the
    // whole process dies of `SIGFPE` inside `vkCreateComputePipelines` with no
    // error for this device to decline on. Fill it the way the sampler class
    // below already fills its own.
    //
    // Only `Used` is filled. A declared-and-unused variable is legal to omit and
    // must stay omitted, or the census that separated those two populations
    // cannot tell them apart any more; `Ambiguous` is two variables on one
    // binding, which is its own defect and is not repaired by picking one.
    let bound: Vec<u32> = sampled_images.iter().map(|img| img.binding).collect();
    for binding in neutral_sampled_image_bindings(&spirv, &bound) {
        crate::observe::Emit::decline(
            "compute_linux_sampled",
            &NeutralSampledImage {
                binding,
                width: NEUTRAL_SAMPLED_IMAGE_EXTENT,
                height: NEUTRAL_SAMPLED_IMAGE_EXTENT,
            },
        )
        .field("pipe", acc.pipeline_ref)
        .fail_once((u64::from(acc.pipeline_ref) << 32) | u64::from(binding));
        sampled_images.push(ComputeSampledImageResource {
            binding,
            array_element: 0,
            descriptor_count: 1,
            format: crate::backend::vulkan::engine::StorageImageFormat::Rgba8Unorm,
            width: NEUTRAL_SAMPLED_IMAGE_EXTENT,
            height: NEUTRAL_SAMPLED_IMAGE_EXTENT,
            // A stand-in for a binding the guest left empty is one level.
            mip_levels: 1,
            source: ComputeSampledSource::Bytes(pixel_format::solid_rgba8(
                NEUTRAL_SAMPLED_IMAGE_EXTENT,
                NEUTRAL_SAMPLED_IMAGE_EXTENT,
                &[0.0; 4],
            )),
        });
    }

    // Reflection is the sampler interface emitted alongside this exact module.
    // Derive it once per dispatch instead of walking every SPIR-V instruction
    // once to filter guest samplers and again to provision defaults.
    let reflected_samplers = kernel_shader.variant(false, false).samplers.clone();
    let mut samplers = Vec::new();
    for s in &acc.samplers {
        let binding = crate::runtime::spirv_bind::SAMPLER_BINDING_BASE + s.index;
        if reflected_samplers
            .binary_search_by_key(&binding, |sampler| sampler.binding)
            .is_err()
        {
            continue;
        }
        let mut sampler = match crate::runtime::draw::vulkan::load_vulkan_sampler(
            state,
            host,
            task_id,
            s.sampler_ref,
            binding,
        ) {
            Ok(v) => v,
            Err(reason) => {
                crate::observe::Emit::decline("compute_linux_sampler", &reason)
                    .field("pipe", acc.pipeline_ref)
                    .fail_once((u64::from(s.sampler_ref) << 32) | u64::from(binding));
                return ComputeStatus::MissingSampler("compute_vk_sampler_load");
            }
        };
        if s.has_lod_clamp {
            sampler.lod_min = s.lod_min_bits;
            sampler.lod_max = s.lod_max_bits;
        }
        samplers.push(sampler);
    }
    for reflected in reflected_samplers.iter() {
        if !samplers
            .iter()
            .any(|sampler| sampler.binding == reflected.binding)
        {
            if let Some(state) = reflected.static_state {
                let sampler = match crate::runtime::draw::vulkan::reflected_static_sampler_resource(
                    "kernel",
                    reflected.binding,
                    state,
                ) {
                    Ok(sampler) => sampler,
                    Err(reason) => {
                        crate::observe::Emit::decline("compute_linux_static_sampler", &reason)
                            .field("pipe", acc.pipeline_ref)
                            .fail_once(
                                (u64::from(acc.pipeline_ref) << 32) | u64::from(reflected.binding),
                            );
                        return ComputeStatus::Unsupported("compute_static_sampler_unsupported");
                    }
                };
                samplers.push(sampler);
            } else {
                samplers.push(
                    crate::backend::vulkan::engine::SamplerResource::normalized_default(
                        reflected.binding,
                    ),
                );
            }
        }
    }

    let req = ComputeRequest {
        spirv,
        entry: "main".into(),
        dispatch,
        storage_buffers,
        sampled_images,
        samplers,
        storage_images,
    };
    // The engine reaches this device's own object caches through the shared
    // borrow, so it is taken before the closure and released with it.
    let device: &DeviceState = state;
    let run_engine = |req: &ComputeRequest| {
        // The guard, not a boolean: the slot returns to service when the call
        // returns *or* unwinds, and there is no arming without one.
        let _watch = super::stall_watchdog::arm_compute_engine_stall_watchdog(
            acc.pipeline_ref,
            req,
            std::time::Duration::from_millis(super::stall_watchdog::COMPUTE_ENGINE_STALL_PROXY_MS),
        );
        vk_engine::execute_compute_request(device, req)
    };
    let out_result = run_engine(&req);
    let out = match out_result {
        Ok(o) => o,
        Err(e) => {
            let unsupported = matches!(&e, DrawError::Unsupported(_));
            crate::observe::Emit::decline("compute_linux_engine", &e)
                .field("pipe", acc.pipeline_ref)
                .fail_once(u64::from(acc.pipeline_ref));
            if unsupported {
                return ComputeStatus::Unsupported("engine_run_unsupported");
            }
            return ComputeStatus::MetalFailed("compute_vk_engine_run");
        }
    };
    if out.buffers.len() != buffer_writable_count || out.images.len() != storage_count {
        crate::observe::fail(format!(
            "compute_linux readback count mismatch pipe={} buf={}/{} img={}/{}",
            acc.pipeline_ref,
            out.buffers.len(),
            buffer_writable_count,
            out.images.len(),
            storage_count
        ));
        return ComputeStatus::MetalFailed("compute_vk_readback_count");
    }
    let vk_engine::ComputeOutput {
        buffers: output_buffers,
        images: output_images,
    } = out;
    for buffer in output_buffers {
        let Some(s) = staged_bufs
            .iter_mut()
            .find(|staged| staged.bind.index == buffer.binding)
        else {
            crate::observe::fail(format!(
                "compute_linux readback binding mismatch pipe={} bind={} bytes={}",
                acc.pipeline_ref,
                buffer.binding,
                buffer.bytes.len()
            ));
            return ComputeStatus::MetalFailed("compute_vk_readback_binding");
        };
        s.bytes = buffer.bytes;
        if let Err(e) = writeback_buffer(
            state,
            host,
            task_id,
            Some(acc.pipeline_ref),
            "vulkan_dispatch",
            s,
        ) {
            return e;
        }
    }
    for (t, result) in staged_tex
        .iter_mut()
        .filter(|texture| texture.is_storage)
        .zip(output_images)
    {
        match result {
            ComputeImageResult::Bytes(bytes) => {
                t.bytes = bytes;
                if let Err(e) = writeback_texture(state, host, task_id, t) {
                    return e;
                }
            }
            // The engine copied straight into the guest's pages, so there is no
            // writeback to do and no bytes to do it from.
            ComputeImageResult::Landed { bytes } => {
                crate::runtime::drain::note_store_route("compute_wb_landed");
                let _ = bytes;
                // The guest's pages are the only place this frame exists now,
                // so no host cache may go on naming one. This arm writes
                // neither cache — both are on the readback path — but a
                // previous dispatch's readback may have left an entry, and it
                // is stale by exactly one frame. Same call, same reason, as
                // both arms of the render rail's GVA Store.
                match &t.writeback {
                    TextureWriteback::Linear {
                        gva, texture_ref, ..
                    } => crate::runtime::surface_cache::forget_gva_copies(
                        state,
                        task_id,
                        *gva,
                        *texture_ref,
                    ),
                    // The surface-keyed rail owes more than a cache forget — a
                    // resident storage window over these bytes and the mapping's
                    // own written mark — and the render Store that shares this
                    // destination owes exactly the same set. Both call it.
                    //
                    // The offsets are the staged ones rather than the licence's,
                    // and they are the same offsets: `licence_mapper_ref_texture_surface`
                    // resolves the window through `mapper_ref_texture_sample_window`, which
                    // is where these came from when the texture was staged.
                    TextureWriteback::MapperRefTexture {
                        mapping_id,
                        surface_offset,
                        span_end,
                        ..
                    } => crate::runtime::mapping_write::vulkan::note_mapper_ref_texture_landed(
                        state,
                        *mapping_id,
                        *surface_offset,
                        *span_end,
                    ),
                    TextureWriteback::None => {}
                }
            }
        }
        // The output is in the guest's pages now, so the engine's image has
        // stopped being the only copy and the reclaim paths may take it. The
        // deferred branch above reaches the same edge through its own flush;
        // without this one a synchronously-written resident stayed flagged
        // unreproducible forever and no reclaim could ever touch it.
        if let Some(candidate) = t.rail.residency {
            crate::backend::vulkan::engine::note_resident_storage_copied_out(&candidate.key);
        }
        note_storage_residency_writeback(state, t);
    }

    // A dispatch that completed is expected control flow; every refusal on this
    // path emits its own typed decline. The fields are this dispatch's own
    // shape — process-cumulative engine totals belong to the parity tests that
    // take a snapshot around a known workload, not to a per-dispatch line that
    // would pay a global engine lock to print them.
    crate::observe::line(format!(
        "compute_linux ok pipe={} wg=[{wg_x},{wg_y},{wg_z}] nbuf={} bro={} brw={} bunused={} ntex={}",
        acc.pipeline_ref,
        staged_bufs.len(),
        buffer_readonly_count,
        buffer_writable_count,
        buffer_unused_count,
        staged_tex.len(),
    ));
    ComputeStatus::Ok
}

/// Whether this dispatch asks for something the Vulkan compute rail cannot do.
///
/// Only two things refuse: a pipeline carrying a stage-input descriptor, and an
/// imageblock. Both name storage this rail has no representation for, so the
/// dispatch would compute against memory that does not exist.
///
/// **`stage_in_region` and `stage_in_region_indirect` deliberately do not
/// refuse.** They bound the stage-in grid a stage-input pipeline walks, so on a
/// pipeline that declares no stage input there is nothing for them to bound and
/// executing the dispatch loses no guest work. That is a claim about the
/// contract rather than a measurement, which is why the caller counts the case
/// (`compute_stage_in_region_unused`) instead of staying silent about it: if a
/// guest ever pairs a region with a stage-input-free pipeline *and* depends on
/// it, the counter is what says so.
pub(crate) fn linux_stage_input_or_imageblock_unsupported(
    pipeline_stage_input: bool,
    acc: &ComputeAccum,
) -> bool {
    pipeline_stage_input || acc.imageblock.is_some()
}

/// Why a reflected kernel dispatch contract could not become device work.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum KernelDispatchDecline {
    /// A `dispatchThreadgroups` record whose workgroup count times its
    /// threadgroup size does not fit the logical thread grid's `u32`.
    GridOverflow,
    /// The reflected exact-thread payload has no representable byte range.
    PushRangeUnavailable,
    /// The translator refused to plan this launch's regions.
    PlanRefused(String),
}

impl KernelDispatchDecline {
    fn reason(&self, pipeline_ref: u32) -> ComputeReflectionDecline {
        match self {
            Self::GridOverflow | Self::PlanRefused(_) => {
                ComputeReflectionDecline::DispatchPlanRefused { pipeline_ref }
            }
            Self::PushRangeUnavailable => {
                ComputeReflectionDecline::DispatchPushRangeUnavailable { pipeline_ref }
            }
        }
    }
}

/// Turn one translated kernel's reflected dispatch contract into the device
/// work this launch performs.
///
/// The contract, not the record, decides the shape. A module translated for
/// whole workgroups baked its local size and can only be dispatched as one
/// rounded grid. A module translated for exact threads left its local size
/// specializable, and the translator decomposes the logical thread grid into
/// the interior plus each axis's boundary slab — at most eight regions, each
/// its own dispatch at its own workgroup size. Issuing such a module as a
/// single rounded dispatch would run invocations past the guest's grid; issuing
/// only some of its regions would drop guest threads. Both are why this returns
/// the whole plan rather than a grid and a correction.
///
/// `dispatch_threads_grid` is the exact thread count of a Metal
/// `dispatchThreads` record, and `None` is a `dispatchThreadgroups` record —
/// whose `workgroups * threadgroup` threads decompose to exactly one region at
/// the nominal local size, so one cached translation serves both Metal forms.
pub(crate) fn kernel_dispatch_launch(
    kernel_dispatch: metal2vulkan::reflect::KernelDispatch,
    nominal_local_size: [u32; 3],
    workgroups: [u32; 3],
    threadgroup: [u32; 3],
    dispatch_threads_grid: Option<[u32; 3]>,
) -> Result<crate::backend::vulkan::engine::ComputeDispatch, KernelDispatchDecline> {
    use crate::backend::vulkan::engine as vk_engine;
    use metal2vulkan::reflect::KernelDispatch;

    if matches!(kernel_dispatch, KernelDispatch::Workgroups) {
        return Ok(vk_engine::ComputeDispatch::Workgroups(workgroups));
    }
    let threads_per_grid = match dispatch_threads_grid {
        Some(grid) => grid,
        None => {
            let mut threads = [0u32; 3];
            for dimension in 0..3 {
                threads[dimension] = workgroups[dimension]
                    .checked_mul(threadgroup[dimension])
                    .ok_or(KernelDispatchDecline::GridOverflow)?;
            }
            threads
        }
    };
    // The reflected range, not a constructed one: `ThreadsFixed` puts its
    // payload at the translator's default offset while `ThreadsDynamic` names
    // its own, and an offset whose range would not fit is refused rather than
    // truncated — a short range is a shader reading bytes no one wrote.
    let range = kernel_dispatch
        .push_constant_range()
        .ok_or(KernelDispatchDecline::PushRangeUnavailable)?;
    let plan = kernel_dispatch
        .plan(nominal_local_size, Some(threads_per_grid))
        .map_err(KernelDispatchDecline::PlanRefused)?;
    Ok(vk_engine::ComputeDispatch::Regions {
        push_offset: range.offset,
        threadgroups_per_grid: plan.threadgroups_per_grid,
        regions: plan
            .regions
            .iter()
            .map(|region| vk_engine::ComputeDispatchRegion {
                local_size: region.local_size,
                group_count: region.group_count,
                push_constants: plan.push_constants(*region),
            })
            .collect(),
    })
}

pub(super) fn spirv_words_le(bytes: &[u8]) -> Result<Vec<u32>, ComputeSpirvDecline> {
    const HEADER_LEN: usize = 20;
    const WORD_ALIGNMENT: usize = 4;
    if bytes.len() < HEADER_LEN {
        return Err(ComputeSpirvDecline::HeaderTooShort {
            len: bytes.len(),
            minimum: HEADER_LEN,
        });
    }
    if !bytes.len().is_multiple_of(WORD_ALIGNMENT) {
        return Err(ComputeSpirvDecline::LengthMisaligned {
            len: bytes.len(),
            alignment: WORD_ALIGNMENT,
        });
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Thin `Option` adapters over the canonical tables in
/// [`crate::backend::vulkan::translate::pixel`].
///
/// These two used to *be* the tables — a second copy of the selector→engine and
/// Metal→engine mappings living in the compute path, where nothing checked them
/// against the pixel table they had to agree with. The call sites below are all
/// `if let Some(..)` / `let Some(..) else`, so the adapters keep that shape; the
/// decision itself now happens in exactly one place.
/// The engine's storage format for a contract selector.
///
/// Total, because the translate layer's map is. It used to take the selector's
/// `u32` ordinal and hand back an `Option`, and both of its call sites carried a
/// `reason=selector_unknown` refusal for the `None` — a decline that could only
/// have fired if two enums in this crate had drifted, which is not a thing the
/// guest can cause and not a thing a run-time check should be watching for.
/// Those two refusals are gone with the `Option`.
pub(super) fn selector_to_engine_storage(
    selector: pixel_format::StorageImageSelector,
) -> crate::backend::vulkan::engine::StorageImageFormat {
    crate::backend::vulkan::translate::pixel::storage_image_from_selector(selector)
}

pub(super) fn mtl_to_engine_sampled(
    format: u16,
) -> Option<crate::backend::vulkan::engine::StorageImageFormat> {
    // The *sampled* admission, not the storage one. Asking `storage_image` here
    // cost macOS 14 and macOS 15 a whole `DispatchThreadgroups` a boot on
    // `MTLPixelFormatR16Unorm`, which is sampleable everywhere and is not a
    // storage format — see `translate::pixel::sampled_image`.
    crate::backend::vulkan::translate::pixel::sampled_image(format).ok()
}

pub(super) fn spirv_image_format_to_engine_storage(
    format: crate::runtime::spirv_bind::ImageFormat,
) -> Option<crate::backend::vulkan::engine::StorageImageFormat> {
    use crate::backend::vulkan::engine::StorageImageFormat as V;
    use crate::runtime::spirv_bind::ImageFormat as S;
    Some(match format {
        S::Rgba32Float => V::Rgba32Float,
        S::Rgba16Float => V::Rgba16Float,
        S::R16Float => V::R16Float,
        S::Rgba16Uint => V::Rgba16Uint,
        S::Rgba8Uint => V::Rgba8Uint,
        S::Rgba8Sint => V::Rgba8Sint,
        S::Rgba8Unorm => V::Rgba8Unorm,
        S::Rg16Float => V::Rg16Float,
        S::R8Unorm => V::R8Unorm,
        S::Rg8Unorm => V::Rg8Unorm,
        S::Rgba32Uint => V::Rgba32Uint,
        S::R32Float => V::R32Float,
        S::R32ui => V::R32Uint,
        // Format-less (`Unknown`) storage images carry no engine texel format —
        // their view format comes from the guest surface, resolved by the caller.
        S::Unknown | S::Unsupported(_) => return None,
    })
}

/// Numeric class of a guest storage format: 0 normalized/float, 1 unsigned
/// integer, 2 signed integer.
///
/// Kept apart from the specialization table below because that table also
/// refuses formats whose storage path is unproven, and the class of a format is
/// a fact about it that holds whether or not we are willing to target it.
fn guest_numeric_class(guest: crate::backend::vulkan::engine::StorageImageFormat) -> u8 {
    use crate::backend::vulkan::engine::StorageImageFormat as V;
    match guest {
        V::Rgba32Float
        | V::Rgba16Float
        | V::R16Float
        | V::Rgba8Unorm
        | V::Bgra8Unorm
        | V::Rg16Float
        | V::R8Unorm
        | V::Rg8Unorm
        | V::R32Float
        | V::Rgb9e5Ufloat
        | V::R16Unorm
        | V::Rg16Unorm
        | V::Rgba16Unorm
        | V::Rgb10a2Unorm
        | V::Bgr10a2Unorm
        | V::A8Unorm
        | V::Rg11b10Float => 0,
        V::Rgba16Uint | V::Rgba8Uint | V::Rgba32Uint | V::R32Uint | V::Rg16Uint => 1,
        V::Rgba8Sint | V::R32Sint => 2,
    }
}

pub(super) fn specialized_storage_image_format(
    guest: crate::backend::vulkan::engine::StorageImageFormat,
    shader: crate::runtime::spirv_bind::ImageFormat,
    write_without_format: bool,
) -> Result<crate::runtime::spirv_bind::ImageFormat, &'static str> {
    use crate::backend::vulkan::engine::StorageImageFormat as V;
    use crate::runtime::spirv_bind::ImageFormat as S;

    let Some(shader_engine) = spirv_image_format_to_engine_storage(shader) else {
        return Err("spirv_storage_format_unsupported");
    };
    // A guest BGRA8Unorm surface written by a normalized (float/unorm-class)
    // shader is a color store. SPIR-V has no `Bgra8` storage format, so a
    // concrete `Rgba8Unorm` view would store the shader's red at the guest's
    // blue byte — the resolution-independent R/B swap. Retarget to a format-less
    // `Unknown` storage image; the engine views it `B8G8R8A8_UNORM` (guest
    // channel order) and the GPU converts the written vec4 to BGRA natively, so
    // every downstream consumer (writeback, resident export, sampling) sees the
    // correct bytes with no per-frame swizzle. Requires
    // `StorageImageWriteWithoutFormat`; when absent we degrade to the swapped
    // `Rgba8Unorm` view and the caller logs the degraded class.
    //
    // A uint/sint shader over BGRA is instead a deliberate raw byte view (byte
    // order preserved, no conversion) and must keep its raw format — it falls
    // through to the raw-view / class-matched logic below, unchanged.
    if matches!(guest, V::Bgra8Unorm) {
        let normalized_color_store = matches!(
            shader,
            S::Rgba8Unorm
                | S::Rgba32Float
                | S::Rgba16Float
                | S::R16Float
                | S::R32Float
                | S::Rg16Float
                | S::R8Unorm
                | S::Rg8Unorm
        );
        if normalized_color_store {
            return Ok(if write_without_format {
                S::Unknown
            } else {
                S::Rgba8Unorm
            });
        }
    }
    // Nothing to specialize when the translator already named the guest's own
    // format. Stated before the class rules below so a guest surface whose
    // storage path is otherwise unproven (`R32Float`, `R32Sint`) is not refused
    // for a shader that declares exactly it.
    if shader_engine == guest {
        return Ok(shader);
    }

    let shader_class = match shader {
        S::Rgba32Float
        | S::Rgba16Float
        | S::R16Float
        | S::R32Float
        | S::Rgba8Unorm
        | S::Rg16Float
        | S::R8Unorm
        | S::Rg8Unorm => 0,
        S::Rgba32Uint | S::Rgba16Uint | S::Rgba8Uint | S::R32ui => 1,
        S::Rgba8Sint => 2,
        // A shader that itself declared `Unknown` (format-less) storage is not a
        // class we specialize by numeric class; the caller only mints `Unknown`
        // deliberately for the BGRA path, which returns above.
        S::Unknown | S::Unsupported(_) => return Err("spirv_storage_format_unsupported"),
    };
    // An integer-class shader over a normalized/float-class guest surface of the
    // same texel width is a deliberate raw byte view — Metal `BGRA8Unorm` bound
    // to a `texture2d<uint, write>` and translated as `Rgba8Uint` writes bytes,
    // not colours, and re-targeting it would convert values that were never meant
    // to be converted. The reverse (a float shader over an integer surface) has
    // never been captured and is refused below rather than guessed at.
    //
    // Within one class equal width means nothing, and the store is a value store:
    // `R32Float` and `Rg16Float` are both four float bytes and mean different
    // things. A `float4` written through the former stores lane `.x` as one f32,
    // which the guest then reads as two halves — so a two-channel write loses its
    // second channel outright and corrupts its first. That is measured, not
    // hypothetical: the guest's decode-time HEIC downsample writes chroma with
    // `OpVectorShuffle … 1 2 1 2` into an `Rg16Float` surface the translator
    // declared `R32f`, and the picture speckles.
    //
    // The `R32Uint` guest case used to be carved out of a bare width test by name
    // for the same reason (`Rgba8Uint` declared over one 32-bit uint channel,
    // storing only the low byte of each lane). It needs no exception now: uint
    // over uint is one class, so it reaches the class-matched table below.
    if guest.bytes_per_texel() == shader_engine.bytes_per_texel()
        && shader_class != 0
        && guest_numeric_class(guest) == 0
    {
        return Ok(shader);
    }

    let (guest_class, specialized) = match guest {
        // R32-single-channel: R32Uint is supported as a storage image by
        // re-targeting the SPIR-V to `R32ui` (its class must still match the
        // shader's numeric class below — a uint-write shader). The remaining
        // R32 sint/float and the packed Rgb9e5 stay sampled-only until a live
        // capture justifies enabling their storage path.
        V::R32Uint => (1, S::R32ui),
        V::R32Sint
        | V::R32Float
        | V::Rgb9e5Ufloat
        | V::R16Unorm
        | V::Rg16Unorm
        // The integer member of that family, sampled-only for the same reason
        // and not for its class: `STORAGE_IMAGE` is no more mandatory for
        // `R16G16_UINT` than for `R16G16_UNORM`.
        | V::Rg16Uint
        | V::Rgba16Unorm
        // The packed 32-bit colour formats join them: Vulkan mandates no
        // `STORAGE_IMAGE` support for any of the three, and one of them is not
        // in the mandatory table at all.
        | V::Rgb10a2Unorm
        | V::Bgr10a2Unorm
        // `A8Unorm` joins them by contract rather than by capability:
        // `storage_selector` has no entry for it, so no guest storage binding
        // can name it and its view mapping would be illegal on one.
        | V::A8Unorm
        | V::Rg11b10Float => {
            return Err("spirv_sampled_only_format_as_storage");
        }
        V::Rgba32Float => (0, S::Rgba32Float),
        V::Rgba16Float => (0, S::Rgba16Float),
        V::R16Float => (0, S::R16Float),
        // Bgra8Unorm normally returns above (Unknown/B8G8R8A8 view, or the
        // degraded Rgba8Unorm) before reaching here; this arm is only the
        // class/bytes fallthrough for Rgba8Unorm and a defensive default.
        V::Rgba8Unorm | V::Bgra8Unorm => (0, S::Rgba8Unorm),
        V::Rg16Float => (0, S::Rg16Float),
        V::R8Unorm => (0, S::R8Unorm),
        V::Rg8Unorm => (0, S::Rg8Unorm),
        V::Rgba32Uint => (1, S::Rgba32Uint),
        V::Rgba16Uint => (1, S::Rgba16Uint),
        V::Rgba8Uint => (1, S::Rgba8Uint),
        V::Rgba8Sint => (2, S::Rgba8Sint),
    };
    if shader_class != guest_class {
        return Err("spirv_guest_numeric_class_mismatch");
    }
    Ok(specialized)
}

/// Resolve a kernel-declared `texture2d_ms<T, access::read>` binding to the
/// retained multisample target that holds its samples.
///
/// # Why this rail exists beside `stage_texture_raw` rather than inside it
///
/// Every other compute texture binding is *staged*: guest texels are read into
/// a host buffer and uploaded into a pooled transient. A multisample image
/// cannot be filled that way at all —
/// `engine::types::SampledResource::multisampled` states the rule, "such an
/// image can only come from a retained multisample target; linear bytes cannot
/// be uploaded into one with a buffer-to-image copy" — so a staging function
/// asked for one has nothing to do and no way to say so except by refusing.
///
/// That is exactly what the compute rail did: `reflected_compute_texture`
/// classified the shape as `UnstageableShape { axis: "multisampled" }`, beside
/// the 1D, 3D, cube, buffer and arrayed axes, and the dispatch was refused
/// whole. For those five the premise holds — the rail produces a single-layer
/// 2D rectangle and binding it to another declared shape is a descriptor-type
/// mismatch. For this one the premise is about bytes that were never wanted.
///
/// # What it resolves, and why through the same span the render rail uses
///
/// The samples live in the engine resident the render pass wrote, keyed by that
/// target's `TargetIdentity`. It is named through `draw::vulkan::gva_span_identity`,
/// which is the identity half of the currency test itself, and not rebuilt
/// here: a second derivation of the same registry key is how two rails come to
/// name different residents for one texture.
///
/// What this rail does *not* take is the other half —
/// `draw::gva_resident_if_current`, the currency test the single-sample
/// resident rails share. That test asks whether
/// anything has written the target's guest pages since the Store, making the
/// resident stale against them. A multisample target has no such second copy:
/// no rail of this device writes a multisample target's guest pages, and this
/// device is the only reader of them. With nothing to compare, the witness
/// cannot answer — it reports no observed write rather than a quiet span — and
/// a refusal for want of an answer would cost the guest its dispatch while
/// protecting nothing. The hazard it stands in for on other rails, an absent or
/// unready resident, is carried here by the engine's own `MultisampleSample*`
/// declines at bind time.
///
/// The target's own geometry comes from `draw::color_target_request`, which is
/// the same resolver the render pass resolved its attachment through, so the
/// bind and the render cannot disagree about extent, format, or sample count.
///
/// Every refusal is fail-visible and named for the rung that refused. A guest
/// kernel that reaches here and gets nothing has lost work.
#[allow(
    clippy::too_many_arguments,
    reason = "the binding's descriptor identity plus the guest object it names"
)]
fn multisample_sampled_texture<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    pipeline_ref: u32,
    texture_ref: u32,
    binding: u32,
    descriptor: crate::runtime::spirv_bind::ReflectedTextureDescriptor,
) -> Result<StagedTexture<VulkanStage>, ComputeStatus> {
    let refuse = |reason: &'static str, detail: String, status: ComputeStatus| {
        crate::observe::fail(format!(
            "compute_linux texture_multisample fail reason={reason} pipe={pipeline_ref} \
             ref={texture_ref} bind={binding} {detail}"
        ));
        status
    };
    // The attachment resolver, not a second reading of the descriptor: the
    // render pass that produced these samples resolved its target through this
    // same function, so its geometry, format and sample count are the ones the
    // resident was created from.
    let Some(req) = crate::runtime::draw::color_target_request(
        state,
        host,
        task_id,
        crate::runtime::render_pass::ColorAttachment {
            texture_ref,
            ..Default::default()
        },
        0,
        0,
        1,
        0,
        0,
        0,
    ) else {
        return Err(refuse(
            "target_unresolved",
            String::new(),
            ComputeStatus::MissingTexture("compute_multisample_target_unresolved"),
        ));
    };
    let c0 = req
        .colors
        .first()
        .expect("color_target_request builds exactly one colour");
    // The texture's own declaration, decoded from its descriptor's trailer. A
    // kernel declaring `texture2d_ms` against a texture that declares one
    // sample is a disagreement between two guest statements, and this device
    // must not pick a side by binding either shape.
    if c0.sample_count <= 1 {
        return Err(refuse(
            "texture_is_single_sample",
            format!(
                "samples={} {}x{} gva={:#x}",
                c0.sample_count, c0.width, c0.height, c0.target_gva
            ),
            ComputeStatus::Unsupported("compute_multisample_texture_is_single_sample"),
        ));
    }
    // Asked here rather than left to the request builder below, which would
    // refuse the whole dispatch under a name that says nothing about which
    // texture carried the format.
    if vulkan::mtl_to_engine_sampled(c0.format).is_none() {
        return Err(refuse(
            "mtl_format_unsupported",
            format!("fmt={:#x}", c0.format),
            ComputeStatus::Unsupported("compute_multisample_format_unsupported"),
        ));
    }
    // The geometry the resident was created from, read once off the resolved
    // attachment and used by the refusals, the identity and the bind alike.
    let (sample_count, width, height, format, target_gva, row_stride) = (
        c0.sample_count,
        c0.width,
        c0.height,
        c0.format,
        c0.target_gva,
        c0.row_stride,
    );
    let span = crate::runtime::draw::GvaSpan {
        texture_ref,
        gva: target_gva,
        row_stride,
        width,
        height,
        format,
    };
    let Some(identity) = gva_span_identity(state, host, task_id, span) else {
        return Err(refuse(
            "resident_unnamed",
            format!("samples={sample_count} {width}x{height} gva={target_gva:#x}"),
            ComputeStatus::MissingTexture("compute_multisample_resident_unnamed"),
        ));
    };
    crate::runtime::drain::note_store_route("compute_multisample_resident_bind");
    Ok(StagedTexture {
        binding,
        pixel_format: format,
        // A multisample image is never a storage image on this rail, so it has
        // no storage selector to carry.
        storage_selector: None,
        width,
        height,
        // A multisample texture has one level by construction.
        mip_levels: 1,
        bytes: Vec::new(),
        is_storage: false,
        // Read-only: the kernel declares `access::read` or this shape would
        // have been refused as `multisampled_storage` before reaching here.
        writeback: TextureWriteback::None,
        rail: VulkanStage {
            array_element: descriptor.array_element,
            descriptor_count: descriptor.descriptor_count,
            residency: None,
            serve: None,
            multisample_target: Some(identity),
        },
    })
}
