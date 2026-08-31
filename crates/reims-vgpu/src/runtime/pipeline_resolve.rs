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
//! The broad type-7 serializer object is not itself a retained resource: other
//! type-7 subtypes are mutable serializer state. A render pipeline *constructed
//! from* that serializer is a distinct, immutable object with the explicit
//! lifetime above. Keeping pipelines in their own typed registry preserves that
//! distinction instead of either retaining every type-7 descriptor or re-reading
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

use crate::backend::vulkan::engine::DrawPreparationDecline;
use crate::model::DeviceState;
use crate::runtime::decode::resource::RenderPipelineDescriptor;
use crate::runtime::drain::note_store_route;
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::m2v_cache::CachedShader;
use crate::runtime::mtlb::{load_mtlb, AirLoadRail};

/// The two buffer-index sets every draw of one pipeline used to rebuild from
/// that pipeline's attribute list.
///
/// Both are functions of `RenderPipelineDescriptor::vertex_attributes` alone —
/// no field of the draw request reaches either — so they belong to the
/// resolution and not to the draw, and building them per draw was two heap
/// allocations and two tree builds on the path `chain_phase` reports as
/// `binds_us`, this draw path's largest column.
///
/// Sorted slices rather than `BTreeSet`s. The population is the attribute list's
/// distinct buffer indices, which a real pipeline runs in the single digits, and
/// at that size a sorted `binary_search` is a cache line and a compare where a
/// tree is a pointer chase per level. The sort also makes the two sets
/// canonical, so an equality between two resolutions means what it reads as.
/// # The measurement did not confirm it, and this is what it said
///
/// The twelve interleaved boots quoted on
/// [`crate::runtime::m2v_cache::ShaderVariant::samplers`] carried this
/// change too, and `binds_us` — the column this targets — moved the **wrong
/// way**: 2.477 [2.418..2.525] before against 2.636 [2.510..2.695] after, per
/// draw. The ranges touch rather than separate, and the sub-column where the two
/// set builds used to sit (`binds_us` less `bind_phase`'s three parts) rose 0.04
/// us, which removing work cannot cause. So the honest reading is that
/// `binds_us`'s boot-to-boot spread is wider than what this change is worth, not
/// that the change costs anything.
///
/// It stays because it is strictly less work — two heap allocations and two tree
/// builds a draw become two `binary_search`es over data the pipeline resolution
/// already holds — and because per-draw allocation churn is a jitter source as
/// well as a mean one. **No claim is made that it bought time.** If a future
/// session wants one, `bind_phase` would need a fourth `Part` bracketing the
/// lookups themselves; the three it has today do not reach them.
pub struct VertexBindPlan {
    /// Buffer indices feeding at least one Constant-step attribute. A bind of
    /// one of these may not take the zero-copy rail: the engine prepends a CPU
    /// base-instance prefix to those bytes at prepare time.
    constant_step: Box<[u32]>,
    /// Every buffer index the attribute list names, whatever the attribute's
    /// format or stride turns out to be.
    ///
    /// Unfiltered on purpose, and this is the one place that reasoning now
    /// lives. An attribute with `format == 0` or a zero stride is skipped by the
    /// draw's attribute walk and reads no bytes, but excluding those here would
    /// make this set depend on the same two fields the walk re-derives through
    /// `bind_attribute_stride`, and the two would drift apart the first time
    /// that derivation changed. Listing an index the walk turns out to skip
    /// costs one gather and never correctness, which is the direction this set
    /// is allowed to be wrong in.
    attribute: Box<[u32]>,
}

impl VertexBindPlan {
    fn build(desc: &RenderPipelineDescriptor) -> Self {
        let mut constant_step: Vec<u32> = desc
            .vertex_attributes
            .iter()
            // No stride term, for the reason `attribute` below states in full:
            // the draw walk does not use `a.stride`, it uses
            // `draw::bind_attribute_stride`, which prefers the per-draw
            // `attributeStride` the guest sent with the bind and falls back to
            // the pipeline's only when there is none. A pipeline declaring
            // stride 0 for a dynamic layout is therefore still walked, still
            // emits a `Constant` step, and still needs the CPU base-instance
            // prefix — while this set, filtered on the pipeline's stride, would
            // say the bind may take the zero-copy rail. `execute_draw_inner`
            // then refuses it with `ConstantVertexRequiresCpuBytes` and the draw
            // is lost.
            //
            // Listing an index the walk turns out to skip costs one CPU staging
            // read and never correctness, which is the direction a set derived
            // from the pipeline alone is allowed to be wrong in.
            .filter(|a| {
                a.format != 0
                    && crate::backend::vulkan::translate::vertex::step_function(
                        a.declared_step_function,
                    ) == Ok(crate::backend::vulkan::engine::VertexStepFunction::Constant)
            })
            .map(|a| a.buffer_index)
            .collect();
        constant_step.sort_unstable();
        constant_step.dedup();
        let mut attribute: Vec<u32> = desc
            .vertex_attributes
            .iter()
            .map(|a| a.buffer_index)
            .collect();
        attribute.sort_unstable();
        attribute.dedup();
        Self {
            constant_step: constant_step.into_boxed_slice(),
            attribute: attribute.into_boxed_slice(),
        }
    }

    /// Whether a bind of this buffer index feeds a Constant-step attribute, and
    /// so must stay on the CPU staging read.
    pub fn is_constant_step(&self, buffer_index: u32) -> bool {
        self.constant_step.binary_search(&buffer_index).is_ok()
    }

    /// Whether the pipeline's attribute list names this buffer index at all.
    pub fn feeds_stage_in(&self, buffer_index: u32) -> bool {
        self.attribute.binary_search(&buffer_index).is_ok()
    }
}

/// Everything a draw chain needs from its pipeline ref, resolved once.
///
/// The registry owns this structure behind one `Arc`, so a hit acquires only
/// that object. The fields are also shared with downstream engine/cache owners;
/// in particular, `RenderPipelineDescriptor` owns two `Vec`s that must never be
/// cloned per draw.
#[derive(Clone)]
pub struct ResolvedRenderPipeline {
    /// Present only for a state registered under the guest pipeline object's
    /// lifetime. The memo-off ablation reconstructs per draw and therefore has
    /// no retained identity to publish to a backend cache.
    pub pipeline_object: Option<crate::backend::vulkan::engine::PipelineObjectIdentity>,
    pub desc: Arc<RenderPipelineDescriptor>,
    pub vertex: Arc<CachedShader>,
    pub fragment: Arc<CachedShader>,
    /// Derived from `desc` and memoized with it — see [`VertexBindPlan`].
    pub bind_plan: Arc<VertexBindPlan>,
}

#[cfg(test)]
pub(crate) fn retained_pipeline_with_desc_for_test(
    desc: RenderPipelineDescriptor,
) -> Arc<ResolvedRenderPipeline> {
    use metal2vulkan::reflect::{ShaderReflection, ShaderStage, REFLECTION_VERSION};

    let reflection = |stage| {
        Arc::new(ShaderReflection {
            reflection_version: REFLECTION_VERSION,
            descriptor_layout: Default::default(),
            stage,
            entry_point: None,
            bindings: vec![],
            argument_buffer_fields: vec![],
            vertex_attributes: vec![],
            varyings: vec![],
            render_targets: vec![],
            depth_members: vec![],
            depth_qualifier: None,
            stencil_members: vec![],
            local_size: None,
            kernel_dispatch: None,
            vertex_builtins: None,
            tessellation: None,
            imageblock_layouts: vec![],
            implicit_imageblock_attachments: vec![],
            fragment_imageblock: None,
            datalayout: None,
            runtime_sampler_specializations: vec![],
            runtime_storage_image_specializations: vec![],
            function_constants: vec![],
        })
    };
    let desc = Arc::new(desc);
    Arc::new(ResolvedRenderPipeline {
        pipeline_object: Some(crate::backend::vulkan::engine::PipelineObjectIdentity::new()),
        bind_plan: Arc::new(VertexBindPlan::build(&desc)),
        desc,
        vertex: Arc::new(CachedShader::new(
            Vec::new(),
            reflection(ShaderStage::Vertex),
        )),
        fragment: Arc::new(CachedShader::new(
            Vec::new(),
            reflection(ShaderStage::Fragment),
        )),
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
///   holds the two `Arc<CachedShader>` that resolution produced — so **an entry
///   existing means those shaders were translated**;
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
    state: &DeviceState,
    _host: &M,
    task_id: u32,
    pipeline_ref: u32,
) -> bool {
    if !memo_enabled() {
        return false;
    }
    if !state
        .task_render_pipeline_states
        .contains(task_id, pipeline_ref)
    {
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
    state: &DeviceState,
    host: &M,
    task_id: u32,
    pipeline_ref: u32,
) -> Result<Arc<ResolvedRenderPipeline>, DrawPreparationDecline> {
    if !memo_enabled() {
        note_store_route("pipe_memo_off");
        return resolve_uncached(state, host, task_id, pipeline_ref).map(Arc::new);
    }

    if let Some(resolved) = state.task_render_pipeline_states.get(task_id, pipeline_ref) {
        note_store_route("pipe_memo_hit");
        return Ok(resolved);
    }
    note_store_route("pipe_memo_miss");

    let mut resolved = resolve_uncached(state, host, task_id, pipeline_ref)?;
    resolved.pipeline_object = Some(crate::backend::vulkan::engine::PipelineObjectIdentity::new());
    let resolved = Arc::new(resolved);
    Ok(state
        .task_render_pipeline_states
        .register(task_id, pipeline_ref, resolved))
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
    state: &DeviceState,
    host: &M,
    task_id: u32,
    pipeline_ref: u32,
) -> Option<u32> {
    if memo_enabled() {
        if let Some(resolved) = state.task_render_pipeline_states.get(task_id, pipeline_ref) {
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
    state: &DeviceState,
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
    let vertex = crate::runtime::m2v_cache::translate_cached_reflected(
        v_air,
        metal2vulkan::passes::Stage::Vertex,
        pipeline_ref,
    )
    .map_err(|reason| DrawPreparationDecline::VertexTranslate {
        pipeline_ref,
        reason,
    })?;
    let fragment = crate::runtime::m2v_cache::translate_cached_reflected(
        f_air,
        metal2vulkan::passes::Stage::Fragment,
        pipeline_ref,
    )
    .map_err(|reason| DrawPreparationDecline::FragmentTranslate {
        pipeline_ref,
        reason,
    })?;
    let bind_plan = Arc::new(VertexBindPlan::build(&desc));
    Ok(ResolvedRenderPipeline {
        pipeline_object: None,
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

        let state = DeviceState::new(DeviceId(1), 12);
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

        let mut state = DeviceState::new(DeviceId(1), 12);
        state.define_task(3, 1 << 20, 7);
        let first = state
            .task_render_pipeline_states
            .register(3, 9, retained_pipeline_for_test());
        let first_id = first
            .pipeline_object
            .as_ref()
            .expect("a retained state has an object identity")
            .id();

        assert!(state.set_object_list(3, 11, 64));
        assert!(Arc::ptr_eq(
            &first,
            &state.task_render_pipeline_states.get(3, 9).unwrap()
        ));
        assert!(state.task_render_pipeline_states.delete(3, 9));
        assert!(!state.task_render_pipeline_states.contains(3, 9));
        assert_eq!(
            Arc::strong_count(&first),
            1,
            "the encoder owner remains valid"
        );

        let replacement =
            state
                .task_render_pipeline_states
                .register(3, 9, retained_pipeline_for_test());
        assert_ne!(
            replacement
                .pipeline_object
                .as_ref()
                .expect("the replacement is retained")
                .id(),
            first_id,
            "reusing a guest reference constructs a new pipeline object"
        );
        state.define_task(3, 1 << 20, 8);
        assert!(
            !state.task_render_pipeline_states.contains(3, 9),
            "task redefinition ends the old task namespace"
        );

        state
            .task_render_pipeline_states
            .register(3, 9, retained_pipeline_for_test());
        assert!(state.delete_task(3));
        assert!(!state.task_render_pipeline_states.contains(3, 9));
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
