//! Render pipeline states constructed once and retained in their task's
//! pipeline-reference namespace.
//!
//! # Contract
//!
//! Pipeline construction resolves the serialized descriptor and both function
//! objects, translates the functions, derives the bind plan, and registers that
//! immutable state under `(task, pipeline_ref)`. An encoder bind retrieves the
//! registered state by reference. The render-pipeline destroy record removes
//! one reference; task deletion or redefinition removes the task's entire
//! namespace. Resource-list deletion belongs to a separate reference space and
//! cannot retire a pipeline merely because its integer collides.
//!
//! The broad serializer-resource serializer object is not itself a retained
//! resource: its sibling descriptor classes are mutable serializer state. A
//! render pipeline *constructed
//! from* that serializer is a distinct, immutable object with the explicit
//! lifetime above. Keeping pipelines in their own typed registry preserves that
//! distinction instead of either retaining every serializer resource or re-reading
//! pipeline construction input on every draw.
//!
//! This used to be an insertion-bounded process-global memo. Every hit re-read
//! the pipeline and both function object-list entries to infer whether the
//! object was still alive. That saved the full eight-walk construction path but
//! still cost about 0.9 us per draw on the macOS 13 x86 Vulkan window-drag rail.
//! It also invented capacity eviction, and its key omitted the device.
//!
//! There is no content freshness check here because a pipeline state is
//! immutable after construction. Replacement is deletion followed by a new
//! construction, and deletion retires the old state before that reference can
//! resolve again. The held descriptor and shaders are host-owned copies, so no
//! guest page remains borrowed for the pipeline lifetime.
//!
//! `REIMS_VGPU_PIPELINE_MEMO=off` remains an ablation that narrows the device
//! back to reconstructing every draw. The `pipe_memo_*` and
//! `preflight_memo_*` route names remain for longitudinal log compatibility;
//! a hit now means a retained pipeline-state lookup, not a byte comparison.

use std::sync::{Arc, OnceLock};

use crate::runtime::drain::note_store_route;
use crate::runtime::draw::DrawPreparationDecline;
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::mtlb::{load_mtlb, AirLoadRail};
use crate::runtime::Device;
use reims_vgpu_core::{ResolvedRenderPipeline, VertexBindPlan};
#[cfg(test)]
use reims_vgpu_protocol::RenderPipelineDescriptor;

#[cfg(test)]
pub(crate) fn retained_pipeline_with_desc_for_test(
    desc: RenderPipelineDescriptor,
) -> Arc<ResolvedRenderPipeline> {
    let desc = Arc::new(desc);
    Arc::new(ResolvedRenderPipeline {
        pipeline_lifetime: Some(reims_vgpu_core::ResourceLifetime::new()),
        bind_plan: Arc::new(VertexBindPlan::build(&desc)),
        desc,
        vertex: reims_vgpu_vulkan::m2v_cache::prepare_render_shader(
            &reims_vgpu_vulkan::m2v_cache::empty_test_shader(
                reims_vgpu_vulkan::m2v_cache::RenderTranslationStage::Vertex,
            ),
            reims_vgpu_vulkan::m2v_cache::RenderTranslationStage::Vertex,
        ),
        fragment: reims_vgpu_vulkan::m2v_cache::prepare_render_shader(
            &reims_vgpu_vulkan::m2v_cache::empty_test_shader(
                reims_vgpu_vulkan::m2v_cache::RenderTranslationStage::Fragment,
            ),
            reims_vgpu_vulkan::m2v_cache::RenderTranslationStage::Fragment,
        ),
    })
}

#[cfg(test)]
pub(crate) fn retained_pipeline_for_test() -> Arc<ResolvedRenderPipeline> {
    retained_pipeline_with_desc_for_test(RenderPipelineDescriptor::default())
}

/// Whether retained pipeline states are on. See [`crate::env::PIPELINE_MEMO`].
fn memo_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            crate::env::read(crate::env::PIPELINE_MEMO).0,
            crate::env::Switch::Off
        )
    })
}

/// Whether `pipeline_ref`'s two shaders are **already translated**, answered
/// from the retained pipeline registry alone and without resolving, translating
/// or reading any AIR.
///
/// # Why the exec preflight can ask this instead of loading the AIR
///
/// `ExecPhase::Preflight` exists to answer one question before any record of a
/// packet runs: will executing this stream have to wait for a translation? It
/// answered it by resolving each pipeline's AIR out of guest memory and offering
/// it to `m2v_cache::ensure_cached_async` — three guest resolves at **4.3 us a
/// pipeline ref, 12 700 refs a second, ~54 ms of every second**.
///
/// A retained-state hit answers the same question without reading guest memory,
/// and it is not a weaker
/// answer:
///
/// - an entry is only ever filed after a successful [`resolve_uncached`], and it
///   holds the two semantic prepared-shader families that resolution produced —
///   so **an entry existing means those shaders were translated and published**;
/// - the m2v translate cache is **unbounded and nothing evicts it** (its only
///   removal is `forget_if_transient`, dropping a transient failure so it can be
///   retried), so a shader translated once is translated for the life of the
///   process.
///
/// Returns `false` whenever the memo is switched off, so
/// `REIMS_VGPU_PIPELINE_MEMO=off` takes the preflight back down its full path
/// along with everything else this module short-circuits.
///
pub fn translations_ready<M: HostMemory + HostOps>(
    state: &Device,
    _host: &M,
    task_id: u32,
    pipeline_ref: u32,
) -> bool {
    if !memo_enabled() {
        return false;
    }
    if !state.task_objects.render_pipelines.contains(
        task_id,
        reims_vgpu_protocol::SerializerRef::new(pipeline_ref),
    ) {
        note_store_route("preflight_memo_absent");
        return false;
    }
    note_store_route("preflight_memo_ready");
    true
}

/// Resolve `pipeline_ref` to its descriptor and both translated shaders.
///
/// Serves the task's retained pipeline state when it has already been
/// constructed. The returned `Arc` is the encoder's ownership of that immutable
/// state while it assembles and executes the draw.
pub fn resolve<M: HostMemory + HostOps>(
    state: &Device,
    host: &M,
    task_id: u32,
    pipeline_ref: u32,
) -> Result<Arc<ResolvedRenderPipeline>, DrawPreparationDecline> {
    if !memo_enabled() {
        note_store_route("pipe_memo_off");
        return resolve_uncached(state, host, task_id, pipeline_ref).map(Arc::new);
    }

    if let Some(resolved) = state.task_objects.render_pipelines.get(
        task_id,
        reims_vgpu_protocol::SerializerRef::new(pipeline_ref),
    ) {
        note_store_route("pipe_memo_hit");
        return Ok(resolved);
    }
    note_store_route("pipe_memo_miss");

    let mut resolved = resolve_uncached(state, host, task_id, pipeline_ref)?;
    resolved.pipeline_lifetime = Some(reims_vgpu_core::ResourceLifetime::new());
    let resolved = Arc::new(resolved);
    Ok(state.task_objects.render_pipelines.register(
        task_id,
        reims_vgpu_protocol::SerializerRef::new(pipeline_ref),
        resolved,
    ))
}

/// The sample count an attachment bound with this pipeline must carry.
///
/// The serialized allocation dimensions available while resolving a linear
/// texture do not repeat the immutable texture-creation sample count. The
/// render contract still supplies a total answer: every attached render target
/// must match the bound pipeline's `rasterSampleCount`, while a distinct resolve
/// destination is single-sampled. Ask retained pipeline state first; on its
/// first draw, decode only the pipeline descriptor and let [`resolve`] retain
/// the complete translated state when encoding begins.
pub fn attachment_sample_count<M: HostMemory + HostOps>(
    state: &Device,
    host: &M,
    task_id: u32,
    pipeline_ref: u32,
) -> Option<u32> {
    if memo_enabled() {
        if let Some(resolved) = state.task_objects.render_pipelines.get(
            task_id,
            reims_vgpu_protocol::SerializerRef::new(pipeline_ref),
        ) {
            return Some(resolved.desc.raster_sample_count.max(1));
        }
    }
    crate::runtime::draw::load_render_pipeline(state, host, task_id, pipeline_ref)
        .map(|desc| desc.raster_sample_count.max(1))
}

/// The full path: object list → descriptor → decode → MTLB → AIR → SPIR-V, for
/// the pipeline and both of its functions.
///
/// This is the only place a draw's pipeline resolution can fail, and each of its
/// seven refusals keeps the `DrawPreparationDecline` variant it always had — the
/// memo in front of it neither adds a failure nor renames one.
fn resolve_uncached<M: HostMemory + HostOps>(
    state: &Device,
    host: &M,
    task_id: u32,
    pipeline_ref: u32,
) -> Result<ResolvedRenderPipeline, DrawPreparationDecline> {
    let desc = crate::runtime::draw::load_render_pipeline(state, host, task_id, pipeline_ref)
        .ok_or(DrawPreparationDecline::PipelineMissing {
            task_id,
            pipeline_ref,
        })?;
    // The same three sub-phases the call site used to open around this work,
    // moved in with it. They are inert outside a live `ChainTimer`, so the two
    // non-draw callers of the loaders below are unaffected — and on the draw
    // rail `pl_desc_us` brackets the task registry lookup on a hit and this
    // construction on a miss, so the lifecycle correction remains measurable.
    use crate::runtime::chain_phase::{enter, Phase};
    enter(Phase::PipelineMtlb);
    let v_mtlb = load_mtlb(
        state,
        host,
        task_id,
        desc.vertex_func_ref,
        AirLoadRail::Draw,
    )
    .ok_or(DrawPreparationDecline::VertexMtlbMissing {
        task_id,
        function_ref: desc.vertex_func_ref,
    })?;
    let f_mtlb = load_mtlb(
        state,
        host,
        task_id,
        desc.fragment_func_ref,
        AirLoadRail::Draw,
    )
    .ok_or(DrawPreparationDecline::FragmentMtlbMissing {
        task_id,
        function_ref: desc.fragment_func_ref,
    })?;
    enter(Phase::PipelineAir);
    let v_air = crate::runtime::mtlb::extract_air(&v_mtlb).map_err(|reason| {
        DrawPreparationDecline::VertexAirExtract {
            function_ref: desc.vertex_func_ref,
            reason,
        }
    })?;
    let f_air = crate::runtime::mtlb::extract_air(&f_mtlb).map_err(|reason| {
        DrawPreparationDecline::FragmentAirExtract {
            function_ref: desc.fragment_func_ref,
            reason,
        }
    })?;
    enter(Phase::PipelineXlate);
    let vertex = state
        .executor
        .prepare_render_translation(
            v_air,
            reims_vgpu_core::ShaderStage::Vertex,
            desc.raster_sample_count.max(1),
            pipeline_ref,
        )
        .map_err(|reason| DrawPreparationDecline::VertexTranslate {
            pipeline_ref,
            reason,
        })?;
    let fragment = state
        .executor
        .prepare_render_translation(
            f_air,
            reims_vgpu_core::ShaderStage::Fragment,
            desc.raster_sample_count.max(1),
            pipeline_ref,
        )
        .map_err(|reason| DrawPreparationDecline::FragmentTranslate {
            pipeline_ref,
            reason,
        })?;
    let bind_plan = Arc::new(VertexBindPlan::build(&desc));
    Ok(ResolvedRenderPipeline {
        pipeline_lifetime: None,
        desc: Arc::new(desc),
        vertex,
        fragment,
        bind_plan,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::decode::resource::VertexAttribute;

    /// An empty retained registry must answer **not ready**, and that direction
    /// is the whole safety of asking it.
    ///
    /// `translations_ready` gates whether the exec preflight skips loading a
    /// pipeline's AIR. A wrong `false` costs the resolve it was trying to save;
    /// a wrong `true` tells the packet its shaders are translated when nothing
    /// has translated them, and the draw then meets an untranslated pipeline
    /// with the packet already committed. So the absent case is pinned
    /// explicitly rather than left to follow from the `Option` being `None`.
    #[test]
    fn an_absent_retained_pipeline_is_never_reported_ready() {
        use crate::model::DeviceId;
        use crate::runtime::host::FakeHost;

        let state = Device::new(DeviceId(1), 12);
        let host = FakeHost::new();
        assert!(
            !translations_ready(&state, &host, 7, 9),
            "a pipeline this registry has never resolved must send the preflight \
             down its own path, not be waved through as translated"
        );
    }

    /// A pipeline state has the pipeline API's lifetime, not the lifetime of the
    /// serializer bytes that constructed it. Re-pointing the object list changes
    /// future construction input; explicit resource deletion and task teardown
    /// end states that already exist.
    #[test]
    fn pipeline_states_survive_list_changes_and_retire_on_explicit_lifetime_events() {
        use crate::model::DeviceId;

        let mut state = Device::new(DeviceId(1), 12);
        state.define_task(3, 1 << 20, 7);
        let first = state.task_objects.render_pipelines.register(
            3,
            reims_vgpu_protocol::SerializerRef::new(9),
            retained_pipeline_for_test(),
        );
        let first_id = first
            .pipeline_lifetime
            .as_ref()
            .expect("a retained state has an object identity")
            .id();

        assert!(state.set_object_list(3, 11, 64));
        assert!(Arc::ptr_eq(
            &first,
            &state
                .task_objects
                .render_pipelines
                .get(3, reims_vgpu_protocol::SerializerRef::new(9))
                .unwrap()
        ));
        assert!(state
            .task_objects
            .render_pipelines
            .delete(3, reims_vgpu_protocol::SerializerRef::new(9)));
        assert!(!state
            .task_objects
            .render_pipelines
            .contains(3, reims_vgpu_protocol::SerializerRef::new(9)));
        assert_eq!(
            Arc::strong_count(&first),
            1,
            "the encoder owner remains valid"
        );

        let replacement = state.task_objects.render_pipelines.register(
            3,
            reims_vgpu_protocol::SerializerRef::new(9),
            retained_pipeline_for_test(),
        );
        assert_ne!(
            replacement
                .pipeline_lifetime
                .as_ref()
                .expect("the replacement is retained")
                .id(),
            first_id,
            "reusing a guest reference constructs a new pipeline object"
        );
        state.define_task(3, 1 << 20, 8);
        assert!(
            !state
                .task_objects
                .render_pipelines
                .contains(3, reims_vgpu_protocol::SerializerRef::new(9)),
            "task redefinition ends the old task namespace"
        );

        state.task_objects.render_pipelines.register(
            3,
            reims_vgpu_protocol::SerializerRef::new(9),
            retained_pipeline_for_test(),
        );
        assert!(state.delete_task(3).is_some());
        assert!(!state
            .task_objects
            .render_pipelines
            .contains(3, reims_vgpu_protocol::SerializerRef::new(9)));
    }

    /// The two sets [`VertexBindPlan`] carries used to be rebuilt inside the
    /// draw path from the same attribute list, and this pins the classification
    /// they replaced rather than the shape of the code that does it.
    ///
    /// The interesting rows are the ones a rewrite can get wrong. A Constant-step
    /// attribute whose `format` is zero is **not** constant step for this
    /// purpose: the draw's attribute walk skips it, so a bind of its buffer stays
    /// eligible for the zero-copy rail.
    ///
    /// A **zero `stride` is different, and it used to be filtered here too.** The
    /// walk does not read `a.stride`; it reads `draw::bind_attribute_stride`,
    /// which prefers the per-draw `attributeStride` the guest sent with the bind.
    /// So a pipeline declaring stride 0 for a dynamic layout is still walked,
    /// still emits a `Constant` step, and still needs the CPU base-instance
    /// prefix — and this set saying otherwise put the bind on the zero-copy rail
    /// and lost the whole draw to `ConstantVertexRequiresCpuBytes`. The rule the
    /// `attribute` field's doc states applies to both sets: a set derived from
    /// the pipeline alone may not depend on a field the draw re-derives.
    ///
    /// Both zero rows still count as *named* by the attribute list, because that
    /// set is deliberately unfiltered.
    #[test]
    fn the_bind_plan_separates_constant_step_from_merely_named() {
        const CONSTANT: Option<u32> = Some(0);
        const PER_INSTANCE: Option<u32> = Some(2);
        let attr = |buffer_index, format, stride, declared_step_function| VertexAttribute {
            location: 0,
            format,
            offset: 0,
            buffer_index,
            stride,
            declared_step_function,
            declared_step_rate: None,
        };
        let desc = RenderPipelineDescriptor {
            vertex_attributes: vec![
                attr(1, 0x21, 16, CONSTANT),     // constant, and it counts
                attr(2, 0x21, 16, PER_INSTANCE), // named, not constant
                attr(3, 0, 16, CONSTANT),        // format 0: the walk skips it
                attr(4, 0x21, 0, CONSTANT),      // stride 0: the draw supplies one
                attr(1, 0x21, 32, PER_INSTANCE), // a second attribute on buffer 1
                attr(5, 0x21, 16, None),         // undeclared step is per-vertex
            ],
            ..Default::default()
        };
        let plan = VertexBindPlan::build(&desc);

        assert!(
            plan.is_constant_step(1),
            "declared Constant with real bytes"
        );
        assert!(
            plan.is_constant_step(4),
            "a dynamic layout declares stride 0 and the draw supplies the real \
             one, so this attribute is walked and needs the base-instance \
             prefix; calling it zero-copy-eligible loses the draw"
        );
        for index in [2, 3, 5] {
            assert!(
                !plan.is_constant_step(index),
                "buffer {index} must keep the zero-copy rail"
            );
        }
        // Unfiltered: every index the list mentions, skipped by the walk or not.
        for index in 1..=5 {
            assert!(plan.feeds_stage_in(index), "buffer {index} is named");
        }
        assert!(!plan.feeds_stage_in(0), "an index the list never names");
        assert!(!plan.feeds_stage_in(6));
    }

    /// A pipeline with no vertex block answers "no" to both questions rather
    /// than panicking on an empty search, which is the shape every fullscreen
    /// pass that builds its vertices in the shader takes.
    #[test]
    fn an_empty_attribute_list_names_nothing() {
        let plan = VertexBindPlan::build(&RenderPipelineDescriptor::default());
        assert!(!plan.is_constant_step(0));
        assert!(!plan.feeds_stage_in(0));
    }
}
