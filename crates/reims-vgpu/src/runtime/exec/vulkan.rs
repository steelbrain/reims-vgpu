//! Work this rail does before a submission may be consumed.
//!
//! Cold AIR translation is immutable CPU work with no protocol ownership, and
//! this rail wants it done before the packet is taken: a stream whose render or
//! compute pipelines are still translating is deferred rather than executed, so
//! a replay cannot duplicate clears, fences, dispatches or guest writeback.
//!
//! A rail that translates nothing has nothing to preflight, which is why the
//! whole of this is reached through
//! [`crate::backend::Backend::preflight_translations`] and `runtime::exec`
//! never names it.

use super::*;

/// `pipelines` is the **model's own lease list** for the packet, not a second
/// scan of its bytes.
///
/// The two used to be different answers to "which pipelines does this packet
/// bind": the walk collected `ResolvedOperation::pipeline_lease` and this
/// re-framed the stream looking for `SetPipeline` records. That difference is
/// load-bearing rather than cosmetic — admission readies *the walk's* leases on
/// *this* function's whole-submission verdict, so a pipeline the scan did not
/// reach was a lease declared ready on an answer that never examined it, which
/// is the shape of the `m2v_translation_pending_at_sync_boundary` loss measured
/// on a driven macos-15 desktop. Taking the list makes the two sets the same
/// set by construction, and there is nothing left to keep in step.
pub(crate) fn preflight_render_translations<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    pipelines: &[u32],
    pending: &mut Vec<u32>,
) {
    use crate::runtime::drain::{note_preflight_part, note_preflight_pipe, PreflightPart};
    for &pipeline_ref in pipelines {
        note_preflight_pipe();
        // The draw path's own memo already knows whether these two shaders are
        // translated, and answers for ~0.6 us against the 4.3 us of guest
        // resolves below. `translations_ready` states why that is not a weaker
        // answer — chiefly that the translate cache never evicts, so a shader
        // this memo saw translated is still translated.
        if crate::backend::vulkan::pipeline_resolve::translations_ready(
            state,
            host,
            task_id,
            pipeline_ref,
        ) {
            continue;
        }
        let air_started = std::time::Instant::now();
        // The MTLB containers, not owned copies of the AIR inside them: the two
        // `ensure_cached_async` calls below borrow, digest and drop, so copying
        // first would allocate twice per pipeline ref for bytes nothing keeps.
        let pair = draw::vulkan::load_render_mtlb_pair(state, host, task_id, pipeline_ref);
        note_preflight_part(PreflightPart::Air, air_started.elapsed().as_nanos() as u64);
        let Ok((v_mtlb, f_mtlb)) = pair else {
            // Normal execution emits the precise pipeline/MTLB failure. A
            // missing plan input is deterministic, not asynchronous work.
            //
            // Counted, because "not pending" is what this arm says and it is
            // the answer that readies the packet's pipeline leases at
            // admission. A draw has been measured reaching a shader that was
            // still translating while its lease read `ready`, and a pipeline
            // this pre-scan silently stopped examining is one of the two ways
            // that can happen — the other being a pipeline the scan never
            // reached at all. They were one silence.
            crate::runtime::drain::note_store_route("preflight_mtlb_unloadable");
            continue;
        };
        // A container whose AIR will not extract is the same "deterministic
        // missing plan input" as one that would not load: normal execution
        // reports it precisely, and there is no asynchronous work to await.
        let f_air = f_mtlb
            .as_ref()
            .map(|f_mtlb| crate::runtime::mtlb::extract_air(f_mtlb))
            .transpose();
        let (Ok(v_air), Ok(f_air)) = (crate::runtime::mtlb::extract_air(&v_mtlb), f_air) else {
            crate::runtime::drain::note_store_route("preflight_air_unextractable");
            continue;
        };
        let cache_started = std::time::Instant::now();
        // `|` and not `||`: both stages must be *started*, so they translate in
        // parallel and the packet is retried once rather than once per stage.
        let translated = crate::runtime::m2v_cache::ensure_cached_async(
            v_air,
            metal2vulkan::passes::Stage::Vertex,
            pipeline_ref,
        ) & f_air.map_or(true, |f_air| {
            // A depth/stencil-only pipeline has no fragment AIR; its stand-in
            // module is a constant and needs no translation.
            crate::runtime::m2v_cache::ensure_cached_async(
                f_air,
                metal2vulkan::passes::Stage::Fragment,
                pipeline_ref,
            )
        });
        note_preflight_part(
            PreflightPart::Cache,
            cache_started.elapsed().as_nanos() as u64,
        );
        if !translated {
            if !pending.contains(&pipeline_ref) {
                pending.push(pipeline_ref);
            }
            continue;
        }
        publish_render_usage(state, host, task_id, pipeline_ref, v_air, f_air);
    }
}

/// Tell the ordering plane what this pipeline's two stages do with each bound
/// slot, from the same pass that established they are translated.
///
/// # Why here, when `pipeline_resolve::ready` already publishes
///
/// `ready` runs at the pipeline's first *draw-time* resolve, which is after the
/// transaction that draws it was admitted — and the walk asks for a pipeline's
/// reflection once, at admission, and never again. So every draw of the packet
/// that first uses a pipeline gets a footprint that is `Unknown` for every slot
/// it binds. A driven macos-15 boot measured that at **1 244 draws** against
/// 69 448 answered by both stages: small, and entirely avoidable, because this
/// pass runs *before* admission readies the packet's leases.
///
/// The compute rail has published from its own pre-scan since it had one. This
/// is the render half, so both rails answer at the same point in a packet's
/// life and the two counters are read against the same denominator.
///
/// [`crate::model::DeviceState::pipeline_usage_published`] is what keeps the
/// two sites from being two answers: whichever reaches the pipeline first
/// publishes, the other finds the gate closed, and the gate is the model's own
/// record rather than a memo either site keeps.
fn publish_render_usage<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    pipeline_ref: u32,
    v_air: &[u8],
    f_air: Option<&[u8]>,
) {
    use crate::backend::vulkan::pipeline_resolve::VertexBindPlan;

    let Some(name) = crate::runtime::objects::name_resource(state, host, task_id, pipeline_ref)
    else {
        crate::runtime::drain::note_store_route("pipeline_usage_unnamed");
        return;
    };
    if state.pipeline_usage_published(name) {
        crate::runtime::drain::note_store_route("render_usage_already_published");
        return;
    }
    // The descriptor, for the vertex stage's attribute slots — the buffers the
    // fixed-function fetch reads, which the reflection does not name. Loaded
    // only on the pass that publishes, which is once per build.
    let Some(desc) = crate::runtime::draw::load_render_pipeline(state, host, task_id, pipeline_ref)
    else {
        crate::runtime::drain::note_store_route("render_usage_descriptor_lost");
        return;
    };
    // No fragment AIR is a depth/stencil-only pipeline; it publishes the same
    // stand-in stage the resolver builds, whose reflection names no resources.
    let fragment = match f_air {
        Some(f_air) => crate::runtime::m2v_cache::translate_cached_reflected(
            f_air,
            metal2vulkan::passes::Stage::Fragment,
            pipeline_ref,
        ),
        None => Ok(crate::runtime::m2v_cache::empty_fragment()),
    };
    let (Ok(vertex), Ok(fragment)) = (
        crate::runtime::m2v_cache::translate_cached_reflected(
            v_air,
            metal2vulkan::passes::Stage::Vertex,
            pipeline_ref,
        ),
        fragment,
    ) else {
        // Cached a moment ago and not now: the entry failed or was forgotten
        // between the two calls. Nothing is owed — the draw path reports the
        // translation failure precisely, and an unpublished pipeline is a
        // conservative footprint rather than a wrong one.
        crate::runtime::drain::note_store_route("render_usage_translation_lost");
        return;
    };
    let plan = VertexBindPlan::build(&desc);
    let (usage, refusals) = crate::backend::vulkan::binding_usage::render(
        &vertex.reflection,
        &plan,
        &fragment.reflection,
    );
    for refusal in refusals.into_iter().flatten() {
        crate::runtime::drain::note_store_route(refusal.slug());
    }
    crate::runtime::draw::publish_pipeline_usage(state, host, task_id, pipeline_ref, usage);
}

/// `dispatches` is the **model's own record of what this packet dispatches**,
/// not a second scan of its bytes.
///
/// The same correction the render arm took in the commit before this one, and
/// for the same reason: admission readies the walk's compute leases on this
/// function's whole-submission verdict, so a kernel the scan did not reach was
/// a lease declared ready on an answer that never examined it. The scan had
/// three ways not to reach one --- a selector carrying no threadgroup size, a
/// dispatch after an encoder continuation, and a framing refusal it swallowed
/// --- and [`reims_vgpu_core::exec::ExecWork::compute_dispatch_translations`]
/// has none of them, because it reads records the walk already resolved.
pub(crate) fn preflight_compute_translations<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    dispatches: &[(u32, [u32; 3])],
    pending: &mut Vec<u32>,
) {
    use crate::runtime::drain::{note_preflight_part, note_preflight_pipe, PreflightPart};
    for &(pipeline_ref, local_size) in dispatches {
        note_preflight_pipe();
        let air_started = std::time::Instant::now();
        let loaded = compute_exec::load_compute_pipeline(state, host, task_id, pipeline_ref)
            .and_then(|pipeline| {
                crate::runtime::mtlb::load_mtlb(
                    state,
                    host,
                    task_id,
                    pipeline.kernel_func_ref,
                    crate::runtime::mtlb::AirLoadRail::Compute,
                )
            });
        note_preflight_part(PreflightPart::Air, air_started.elapsed().as_nanos() as u64);
        let Some(mtlb) = loaded else {
            continue;
        };
        let Ok(air) = crate::runtime::mtlb::extract_air(&mtlb) else {
            continue;
        };
        let cache_started = std::time::Instant::now();
        let cached =
            crate::runtime::m2v_cache::ensure_cached_kernel_async(air, local_size, pipeline_ref);
        note_preflight_part(
            PreflightPart::Cache,
            cache_started.elapsed().as_nanos() as u64,
        );
        if !cached {
            if !pending.contains(&pipeline_ref) {
                pending.push(pipeline_ref);
            }
            continue;
        }
        publish_kernel_usage(state, host, task_id, pipeline_ref, air, local_size);
    }
}

/// Tell the ordering plane what this kernel does with each bound slot.
///
/// # Why here, and not where the dispatch runs
///
/// The walk asks for a pipeline's reflection **once, at admission**, so a
/// transaction that leases a kernel before its usage lands carries
/// [`reims_vgpu_core::access::AccessMode::Unknown`] for every slot it binds for
/// the life of that transaction. The compute rail's other point of contact with
/// a reflection is `compute_exec::vulkan`, which runs after the transaction has
/// been released — too late to be the answer that transaction was admitted on.
/// This pre-scan is the earliest site that holds the AIR, and it runs before
/// admission readies the packet's leases.
///
/// # Once per build, not once per packet
///
/// Enumerating a binding set allocates two vectors, and this loop reaches every
/// dispatch of every packet. [`DeviceState::pipeline_usage_published`] is the
/// gate, and it is the model's own record rather than a set kept here: a
/// withdrawal or a retirement drops the published usage, and a rail-side memo
/// would go on claiming the pipeline was covered after the answer had been
/// thrown away.
///
/// The translation is already cached — the caller has just established that —
/// so the call below is a lock and an `Arc` clone, taken only on the pass that
/// actually publishes.
fn publish_kernel_usage<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    pipeline_ref: u32,
    air: &[u8],
    local_size: [u32; 3],
) {
    let Some(name) = crate::runtime::objects::name_resource(state, host, task_id, pipeline_ref)
    else {
        crate::runtime::drain::note_store_route("pipeline_usage_unnamed");
        return;
    };
    if state.pipeline_usage_published(name) {
        crate::runtime::drain::note_store_route("compute_usage_already_published");
        return;
    }
    let Ok(shader) =
        crate::runtime::m2v_cache::translate_cached_kernel_reflected(air, local_size, pipeline_ref)
    else {
        // The entry the caller found cached is gone or failed between the two
        // calls. Nothing is owed: the dispatch's own path reports the
        // translation failure precisely, and an unpublished kernel is a
        // conservative footprint rather than a wrong one.
        crate::runtime::drain::note_store_route("compute_usage_translation_lost");
        return;
    };
    match crate::backend::vulkan::binding_usage::compute(&shader.reflection) {
        Ok(usage) => {
            crate::runtime::draw::publish_pipeline_usage(state, host, task_id, pipeline_ref, usage)
        }
        // A stage that cannot be published leaves the kernel at `Unknown`,
        // which costs ordering and never correctness. Counted rather than
        // failed, and the count is what says which reason is worth closing.
        Err(refusal) => crate::runtime::drain::note_store_route(refusal.slug()),
    }
}

/// [`crate::backend::Backend::preflight_translations`] for this rail: the
/// pipeline refs this packet binds that the rail does not hold a translation
/// for yet, deduplicated.
///
/// **The refs and not a bool**, because the caller has two different things to
/// do with the two halves of the answer. A lease this rail cannot serve must
/// stop being ready, and a lease it can serve must become ready; a single
/// "something is pending" cannot tell the caller which lease is which, so it
/// had to treat the whole packet as one — and the pipelines that *were* ready
/// stayed ready either way.
///
/// Every ref is examined, not just up to the first pending one: the point is
/// to *start* every cold translation this packet needs, so they proceed in
/// parallel and the packet is retried once rather than once per shader.
pub fn preflight_translations<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    render_pipelines: &[u32],
    compute_dispatches: &[(u32, [u32; 3])],
) -> Vec<u32> {
    // Both halves are asked once for the whole packet, and both are asked
    // about what the walk resolved. Neither reads a stream: the packet's bytes
    // were read once, and a second reading is a second answer to "what does
    // this packet run".
    let mut pending = Vec::new();
    preflight_render_translations(state, host, task_id, render_pipelines, &mut pending);
    preflight_compute_translations(state, host, task_id, compute_dispatches, &mut pending);
    pending
}
