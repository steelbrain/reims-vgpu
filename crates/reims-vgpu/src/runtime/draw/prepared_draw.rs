//! Immutable handoff from semantic draw planning to native execution.
//!
//! Request construction decides both the executor input and the one completion
//! route that may consume its output. Store routing does not re-derive that
//! decision after submission from mutable device state.

use crate::model::TargetIdentity;
use crate::runtime::executor::{self, ExecutorDiagnostic};
use crate::runtime::Device;

#[derive(Clone, Debug)]
pub(super) enum DrawCompletionRoute {
    Pixels,
    EffectsOnly,
    ResidentChain(TargetIdentity),
    ResidentGvaReadback(TargetIdentity),
    ResidentGvaStore(TargetIdentity),
    ResidentSurfaceStore(TargetIdentity),
}

impl DrawCompletionRoute {
    pub(super) const fn kind(&self) -> &'static str {
        match self {
            Self::Pixels => "pixels",
            Self::EffectsOnly => "effects_only",
            Self::ResidentChain(_) => "resident_chain",
            Self::ResidentGvaReadback(_) => "resident_gva_readback",
            Self::ResidentGvaStore(_) => "resident_gva_store",
            Self::ResidentSurfaceStore(_) => "resident_surface_store",
        }
    }

    pub(super) fn claim(
        &mut self,
        next: Self,
    ) -> Result<(), reims_vgpu_core::draw_preparation::CompletionRouteConflict> {
        if !matches!(self, Self::Pixels) {
            return Err(reims_vgpu_core::draw_preparation::CompletionRouteConflict {
                current: self.kind(),
                requested: next.kind(),
            });
        }
        *self = next;
        Ok(())
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "each argument is an independently resolved semantic target fact"
)]
pub(super) fn plan_target_completion(
    state: &Device,
    request: &super::DrawEncodeRequest,
    executor_request: &mut reims_vgpu_core::DrawRequest,
    gva_allocation_generation: u64,
    store_publishes: bool,
    store_preserves: bool,
    writeback_guest: bool,
    surface_target: Option<TargetIdentity>,
    load_from_resident: bool,
    gva_load_target: Option<TargetIdentity>,
    width: u32,
    height: u32,
) -> Result<DrawCompletionRoute, super::DrawPreparationDecline> {
    use super::DrawPreparationDecline;

    let mut route = DrawCompletionRoute::Pixels;
    let mut claim = |next| {
        route
            .claim(next)
            .map_err(|conflict| DrawPreparationDecline::CompletionRouteConflict { conflict })
    };

    // Intermediate records retain their protocol-keyed target for the next
    // record. A final multisample Store does the same because its named image,
    // not single-sample guest bytes, is the pass result.
    if request.chain_from_resident || (store_preserves && (!writeback_guest || !store_publishes)) {
        if let Some(identity) =
            super::render_chain_identity(state, request, gva_allocation_generation)
        {
            executor_request.target_identity = Some(identity.clone());
            if store_preserves && (!writeback_guest || !store_publishes) {
                executor_request.skip_readback = true;
                claim(DrawCompletionRoute::ResidentChain(identity))?;
            }
        }
    }

    // A final GVA Store may leave its exact generational resident authoritative
    // until synchronization or a guest reader requires page bytes.
    let guest_backing_available = executor_request
        .target_guest
        .as_ref()
        .is_some_and(|target| target.memory().is_some());
    // Execution identity and completion authority are separate. A copied GVA
    // target still renders into its exact generational resident so concurrent
    // targets and a following LOAD cannot alias a pooled native image. That
    // resident may complete the Store only when it is the guest allocation.
    // Otherwise the guest may CPU-write a subregion as soon as completion is
    // observed, and a later validity notification cannot merge that region
    // with GPU-only pixels without overwriting one writer.
    if store_publishes
        && writeback_guest
        && super::gva_store_defer_eligible(request, gva_allocation_generation)
    {
        if let Some(identity) =
            super::gva_chain_identity(state.executor.as_ref(), request, gva_allocation_generation)
        {
            executor_request.target_identity = Some(identity.clone());
            executor_request.skip_readback = true;
            if guest_backing_available {
                claim(DrawCompletionRoute::ResidentGvaStore(identity))?;
            } else {
                claim(DrawCompletionRoute::ResidentGvaReadback(identity))?;
            }
        }
    }

    // A surface Store may skip readback only when the executor target is the
    // exact surface identity its completion will publish.
    if executor_request.target_identity.is_none() {
        executor_request.target_identity = surface_target.clone();
    }
    let renders_into_surface =
        surface_target.is_some() && executor_request.target_identity == surface_target;
    if store_publishes && renders_into_surface && !executor_request.skip_readback {
        executor_request.skip_readback = true;
        claim(DrawCompletionRoute::ResidentSurfaceStore(
            surface_target.expect("the equality above established a surface identity"),
        ))?;
    }

    // A resident LOAD must name the same target the executor will load. This is
    // independent of whether the pass publishes a Store.
    if load_from_resident {
        if executor_request.target_identity.is_none() {
            executor_request.target_identity = gva_load_target;
        }
        if executor_request.target_identity.is_none() {
            return Err(DrawPreparationDecline::ChainResidentIdentityMissing {
                target_gva: request
                    .colors
                    .first()
                    .map(|color| color.target_gva())
                    .unwrap_or(0),
                width,
                height,
            });
        }
        executor_request.load_from_target = true;
        executor_request.target_rgba8 = None;
    }

    let depth_stencil_store =
        executor_request
            .depth_attachment
            .as_ref()
            .is_some_and(|attachment| {
                attachment
                    .depth
                    .is_some_and(|depth| depth.store_action.publishes_single_sample())
                    || attachment
                        .stencil
                        .is_some_and(|stencil| stencil.store_action.publishes_single_sample())
            });
    if matches!(route, DrawCompletionRoute::Pixels) && !store_publishes && depth_stencil_store {
        executor_request.skip_readback = true;
        route = DrawCompletionRoute::EffectsOnly;
    }

    Ok(route)
}

pub(super) struct PreparedDraw {
    request: reims_vgpu_core::DrawRequest,
    route: DrawCompletionRoute,
    render_target_resource: Option<std::sync::Arc<crate::model::TaskResource>>,
}

/// Semantic work owed only after the executor accepts one prepared draw.
///
/// Kept apart from [`PreparedDraw::request`] so immutable executor input can
/// cross an ownership boundary without granting the executor access to
/// `Device`. Applying this value is the sole transition that records backend
/// materialization and render-target use.
pub(super) struct DrawCompletionPlan {
    route: DrawCompletionRoute,
    render_target_resource: Option<std::sync::Arc<crate::model::TaskResource>>,
}

/// One packet-owned render transaction: immutable executor requests paired
/// with the semantic completion plans that consume their ordered outputs.
pub(super) struct PreparedDrawSubmission {
    context: reims_vgpu_core::SubmissionContext,
    draws: PreparedDraws,
}

enum PreparedDraws {
    Empty,
    One(PreparedDraw),
    Many(Vec<PreparedDraw>),
}

pub(super) struct PreparedDrawProgress {
    pub completed: Vec<CompletedDraw>,
    pub failure: Option<ExecutorDiagnostic>,
}

impl PreparedDrawSubmission {
    pub fn new(context: reims_vgpu_core::SubmissionContext) -> Self {
        Self {
            context,
            draws: PreparedDraws::Empty,
        }
    }

    pub fn push(&mut self, draw: PreparedDraw) {
        self.draws = match std::mem::replace(&mut self.draws, PreparedDraws::Empty) {
            PreparedDraws::Empty => PreparedDraws::One(draw),
            PreparedDraws::One(first) => PreparedDraws::Many(vec![first, draw]),
            PreparedDraws::Many(mut draws) => {
                draws.push(draw);
                PreparedDraws::Many(draws)
            }
        };
    }

    pub fn execute(self, state: &mut Device) -> Result<PreparedDrawProgress, ExecutorDiagnostic> {
        let draws = match self.draws {
            PreparedDraws::Empty => {
                return Ok(PreparedDrawProgress {
                    completed: Vec::new(),
                    failure: None,
                });
            }
            PreparedDraws::One(draw) => {
                let completed = draw.execute_direct(state, self.context)?;
                return Ok(PreparedDrawProgress {
                    completed: vec![completed],
                    failure: None,
                });
            }
            PreparedDraws::Many(draws) => draws,
        };
        let executor = std::sync::Arc::clone(&state.executor);
        let mut requests = Vec::with_capacity(draws.len());
        let mut completion_plans = Vec::with_capacity(draws.len());
        for draw in draws {
            let (request, completion) = draw.into_executor_parts();
            requests.push(request);
            completion_plans.push(completion);
        }
        let engine_started = std::time::Instant::now();
        let progress = executor::execute_draws_progress(executor.as_ref(), self.context, requests)
            .map_err(|error| ExecutorDiagnostic::from_decline(&error));
        crate::runtime::chain_phase::note_detached(
            crate::runtime::chain_phase::Phase::Engine,
            engine_started.elapsed(),
        );
        let progress = progress?;
        state
            .task_objects
            .resources
            .record_gpu_materializations(progress.gpu_materialized.iter().copied());
        let completed = completion_plans
            .into_iter()
            .zip(progress.output)
            .map(|(plan, output)| plan.apply_output(progress.submission, output))
            .collect();
        Ok(PreparedDrawProgress {
            completed,
            failure: progress
                .failure
                .map(|error| ExecutorDiagnostic::from_decline(&error)),
        })
    }
}

impl PreparedDraw {
    pub(super) fn resident_chain_identity(&self) -> Option<&TargetIdentity> {
        match &self.route {
            DrawCompletionRoute::ResidentChain(identity) => Some(identity),
            _ => None,
        }
    }

    pub(super) fn is_surface_store(&self) -> bool {
        matches!(self.route, DrawCompletionRoute::ResidentSurfaceStore(_))
    }

    pub(super) fn new(
        request: reims_vgpu_core::DrawRequest,
        route: DrawCompletionRoute,
        render_target_resource: Option<std::sync::Arc<crate::model::TaskResource>>,
    ) -> Self {
        let render_target_resource = request
            .target_identity
            .is_some()
            .then_some(render_target_resource)
            .flatten();
        Self {
            request,
            route,
            render_target_resource,
        }
    }

    pub(super) fn execute(
        self,
        state: &mut Device,
        task_id: u32,
    ) -> Result<CompletedDraw, ExecutorDiagnostic> {
        let context = executor::context_for(state, task_id);
        let mut submission = PreparedDrawSubmission::new(context);
        submission.push(self);
        let mut progress = submission.execute(state)?;
        if let Some(failure) = progress.failure {
            return Err(failure);
        }
        Ok(progress
            .completed
            .pop()
            .expect("one prepared draw returns one completion"))
    }

    fn execute_direct(
        self,
        state: &mut Device,
        submission: reims_vgpu_core::SubmissionContext,
    ) -> Result<CompletedDraw, ExecutorDiagnostic> {
        let (request, completion) = self.into_executor_parts();
        let executor = std::sync::Arc::clone(&state.executor);
        let receipt = executor::execute_draw(executor.as_ref(), submission, request)
            .map_err(|error| ExecutorDiagnostic::from_decline(&error))?;
        Ok(completion.apply(state, receipt))
    }

    /// Split immutable executor input from the semantic transition its
    /// successful completion owes. Neither half mutates device state.
    pub(super) fn into_executor_parts(self) -> (reims_vgpu_core::DrawRequest, DrawCompletionPlan) {
        let Self {
            request,
            route,
            render_target_resource,
        } = self;
        (
            request,
            DrawCompletionPlan {
                route,
                render_target_resource,
            },
        )
    }
}

impl DrawCompletionPlan {
    /// Apply only an executor-validated completion. A refusal never produces a
    /// receipt and therefore cannot reach this transition.
    pub(super) fn apply(
        self,
        state: &mut Device,
        receipt: executor::ExecutionReceipt<reims_vgpu_core::DrawOutput>,
    ) -> CompletedDraw {
        state
            .task_objects
            .resources
            .record_gpu_materializations(receipt.gpu_materialized.iter().copied());
        self.apply_output(receipt.submission, receipt.output)
    }

    fn apply_output(
        self,
        submission: reims_vgpu_protocol::SubmissionIdentity,
        output: reims_vgpu_core::DrawOutput,
    ) -> CompletedDraw {
        if let Some(resource) = self.render_target_resource {
            resource.note_render_target_use();
        }
        CompletedDraw {
            submission: submission.id,
            output,
            route: self.route,
        }
    }
}

pub(super) struct CompletedDraw {
    pub(super) submission: reims_vgpu_protocol::SubmissionId,
    pub(super) output: reims_vgpu_core::DrawOutput,
    pub(super) route: DrawCompletionRoute,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gva(gva: u64) -> TargetIdentity {
        TargetIdentity::Gva {
            gva,
            width: 8,
            height: 8,
            generation: 1,
            format: reims_vgpu_core::pixel_format::TexelLayout::Rgba8,
        }
    }

    #[test]
    fn one_prepared_draw_carries_exactly_one_completion_route() {
        let identity = gva(0x1000);
        let mut route = DrawCompletionRoute::Pixels;
        route
            .claim(DrawCompletionRoute::ResidentGvaStore(identity.clone()))
            .expect("the first semantic route owns completion");
        let conflict = route
            .claim(DrawCompletionRoute::ResidentSurfaceStore(identity))
            .expect_err("a second route cannot silently win by branch order");
        assert_eq!(conflict.current, "resident_gva_store");
        assert_eq!(conflict.requested, "resident_surface_store");
    }

    #[test]
    fn a_depth_store_without_a_color_store_has_an_effects_only_completion() {
        use reims_vgpu_protocol::pass_action::{LoadAction, StoreAction};

        let state = Device::new(crate::model::DeviceId(1), crate::model::PAGE_SHIFT_X86);
        let owner = std::sync::Arc::new(crate::model::TaskResource::new(
            reims_vgpu_protocol::ObjectListEntry::new(
                crate::runtime::decode::resource::ObjectKind::Texture,
                0,
                0,
            ),
            std::sync::Arc::from([]),
        ));
        let mut executor_request = reims_vgpu_core::DrawRequest {
            depth_attachment: Some(reims_vgpu_core::DepthAttachment {
                identity: gva(0x4000),
                resource_lifetime: owner.lifetime_ref(),
                depth: Some(reims_vgpu_core::DepthAspectAttachment {
                    load_action: LoadAction::Clear,
                    store_action: StoreAction::Store,
                    clear_value: 0.0,
                }),
                stencil: None,
            }),
            ..reims_vgpu_core::DrawRequest::default()
        };
        let route = plan_target_completion(
            &state,
            &super::super::DrawEncodeRequest::default(),
            &mut executor_request,
            0,
            false,
            false,
            true,
            None,
            false,
            None,
            8,
            8,
        )
        .expect("a completed depth Store has one typed outcome");

        assert!(matches!(route, DrawCompletionRoute::EffectsOnly));
        assert!(
            executor_request.skip_readback,
            "no colour Store means there are no colour pixels to read back"
        );
    }

    #[test]
    fn prepared_draw_retains_a_target_resource_only_for_a_resident_request() {
        let resource = std::sync::Arc::new(crate::model::TaskResource::new(
            reims_vgpu_protocol::ObjectListEntry::new(
                crate::runtime::decode::resource::ObjectKind::Texture,
                0,
                0,
            ),
            std::sync::Arc::from([]),
        ));
        let without_target = PreparedDraw::new(
            reims_vgpu_core::DrawRequest::default(),
            DrawCompletionRoute::Pixels,
            Some(resource.clone()),
        );
        assert!(without_target.render_target_resource.is_none());

        let with_target = PreparedDraw::new(
            reims_vgpu_core::DrawRequest {
                target_identity: Some(gva(0x3000)),
                ..reims_vgpu_core::DrawRequest::default()
            },
            DrawCompletionRoute::Pixels,
            Some(resource.clone()),
        );
        assert!(std::sync::Arc::ptr_eq(
            with_target
                .render_target_resource
                .as_ref()
                .expect("the resident request retains its semantic target"),
            &resource,
        ));
        assert!(!resource.was_render_target());
    }

    #[test]
    fn target_planning_couples_executor_flags_to_the_completion_route() {
        let state = Device::new(crate::model::DeviceId(1), crate::model::PAGE_SHIFT_X86);
        let request = super::super::DrawEncodeRequest::default();
        let surface = TargetIdentity::Surface {
            id: 7,
            width: 8,
            height: 8,
            generation: 2,
            format: reims_vgpu_core::pixel_format::TexelLayout::Bgra8,
        };
        let mut executor_request = reims_vgpu_core::DrawRequest::default();
        let route = plan_target_completion(
            &state,
            &request,
            &mut executor_request,
            0,
            true,
            true,
            true,
            Some(surface.clone()),
            false,
            None,
            8,
            8,
        )
        .expect("an exact surface target has one completion route");
        assert!(matches!(
            route,
            DrawCompletionRoute::ResidentSurfaceStore(identity) if identity == surface
        ));
        assert_eq!(executor_request.target_identity, Some(surface));
        assert!(executor_request.skip_readback);

        let load_target = gva(0x2000);
        let mut executor_request = reims_vgpu_core::DrawRequest {
            target_rgba8: Some(std::sync::Arc::new(vec![1, 2, 3, 4])),
            ..reims_vgpu_core::DrawRequest::default()
        };
        let route = plan_target_completion(
            &state,
            &request,
            &mut executor_request,
            0,
            false,
            false,
            true,
            None,
            true,
            Some(load_target.clone()),
            8,
            8,
        )
        .expect("a resident LOAD names its executor target");
        assert!(matches!(route, DrawCompletionRoute::Pixels));
        assert_eq!(executor_request.target_identity, Some(load_target));
        assert!(executor_request.load_from_target);
        assert!(executor_request.target_rgba8.is_none());

        let mut executor_request = reims_vgpu_core::DrawRequest::default();
        assert!(matches!(
            plan_target_completion(
                &state,
                &request,
                &mut executor_request,
                0,
                false,
                false,
                true,
                None,
                true,
                None,
                8,
                8,
            ),
            Err(
                super::super::DrawPreparationDecline::ChainResidentIdentityMissing {
                    target_gva: 0,
                    width: 8,
                    height: 8,
                }
            )
        ));
        assert!(!executor_request.load_from_target);
    }

    /// A device-local image cannot be the completion of a Store into a
    /// guest-addressable texture.
    ///
    /// The guest may CPU-write only part of the allocation immediately after
    /// completion. Once that write has happened, invalidating the resident
    /// cannot preserve both its untouched GPU pixels and the guest's new
    /// pixels. With no guest-backed executor target, the Store therefore owns
    /// a synchronous pixel completion.
    #[test]
    fn a_gva_store_without_guest_backing_reads_its_exact_resident_at_completion() {
        use reims_vgpu_protocol::pass_action::StoreAction;

        let state = Device::new(crate::model::DeviceId(1), crate::model::PAGE_SHIFT_X86);
        let request = super::super::DrawEncodeRequest {
            colors: vec![super::super::ColorRtRequest {
                texture_ref: 9,
                storage: super::super::ColorTargetStorage::Linear(
                    reims_vgpu_core::LinearColorTarget {
                        allocation_gva: 0x1000,
                        allocation_size: 8 * 8 * 4,
                        plane_offset: 0,
                        row_stride: 8 * 4,
                    },
                ),
                width: 8,
                height: 8,
                format: reims_vgpu_core::pixel_format::MTL_FORMAT_RGBA8_UNORM,
                sample_count: 1,
                store_action: StoreAction::Store,
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut executor_request = reims_vgpu_core::DrawRequest::default();
        let route = plan_target_completion(
            &state,
            &request,
            &mut executor_request,
            1,
            true,
            true,
            true,
            None,
            false,
            None,
            8,
            8,
        )
        .expect("a GVA Store has one completion route");

        assert!(matches!(route, DrawCompletionRoute::ResidentGvaReadback(_)));
        assert!(executor_request.skip_readback);
        assert_eq!(executor_request.target_identity, Some(gva(0x1000)));
    }

    #[test]
    fn a_multisample_store_retains_samples_without_guest_readback() {
        use reims_vgpu_protocol::pass_action::StoreAction;

        let state = Device::new(crate::model::DeviceId(1), crate::model::PAGE_SHIFT_X86);
        let request = super::super::DrawEncodeRequest {
            colors: vec![super::super::ColorRtRequest {
                texture_ref: 9,
                storage: super::super::ColorTargetStorage::Linear(
                    reims_vgpu_core::LinearColorTarget {
                        allocation_gva: 0x1000,
                        allocation_size: 8 * 8 * 4,
                        plane_offset: 0,
                        row_stride: 8 * 4,
                    },
                ),
                width: 8,
                height: 8,
                format: reims_vgpu_core::pixel_format::MTL_FORMAT_RGBA8_UNORM,
                sample_count: 4,
                store_action: StoreAction::Store,
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(!request.colors[0].publishes_single_sample());
        assert!(request.colors[0].preserves_attachment_samples());

        let mut executor_request = reims_vgpu_core::DrawRequest::default();
        let route = plan_target_completion(
            &state,
            &request,
            &mut executor_request,
            1,
            false,
            true,
            true,
            None,
            false,
            None,
            8,
            8,
        )
        .expect("a multisample Store has one resident completion");

        assert!(matches!(route, DrawCompletionRoute::ResidentChain(_)));
        assert!(executor_request.skip_readback);
        assert_eq!(executor_request.target_identity, Some(gva(0x1000)));
    }

    #[test]
    fn executor_input_and_completion_plan_split_without_applying_state() {
        let identity = gva(0x2000);
        let request = reims_vgpu_core::DrawRequest {
            target_identity: Some(identity.clone()),
            ..Default::default()
        };
        let prepared = PreparedDraw::new(
            request,
            DrawCompletionRoute::ResidentGvaStore(identity.clone()),
            None,
        );

        let (request, completion) = prepared.into_executor_parts();
        assert_eq!(request.target_identity, Some(identity.clone()));
        assert!(matches!(
            completion.route,
            DrawCompletionRoute::ResidentGvaStore(found) if found == identity
        ));
        assert!(completion.render_target_resource.is_none());
    }
}
