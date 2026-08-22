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
    ResidentChain(TargetIdentity),
    ResidentGvaStore(TargetIdentity),
    ResidentSurfaceStore(TargetIdentity),
}

impl DrawCompletionRoute {
    pub(super) const fn kind(&self) -> &'static str {
        match self {
            Self::Pixels => "pixels",
            Self::ResidentChain(_) => "resident_chain",
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
    gpu_only_content_allowed: bool,
    store_publishes: bool,
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

    // Intermediate records use a protocol-keyed resident but leave no
    // guest-visible GPU-only content: the final record owns publication.
    if request.chain_from_resident || (store_publishes && !writeback_guest) {
        if let Some(identity) =
            super::render_chain_identity(state, request, gva_allocation_generation)
        {
            executor_request.target_identity = Some(identity.clone());
            if store_publishes && !writeback_guest {
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
    if (guest_backing_available || gpu_only_content_allowed) && store_publishes && writeback_guest {
        if let Some(identity) =
            super::gva_chain_identity(state.executor.as_ref(), request, gva_allocation_generation)
        {
            if super::gva_store_defer_eligible(request, gva_allocation_generation) {
                executor_request.target_identity = Some(identity.clone());
                executor_request.skip_readback = true;
                claim(DrawCompletionRoute::ResidentGvaStore(identity))?;
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
    if renders_into_surface && !executor_request.skip_readback {
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

    Ok(route)
}

pub(super) struct PreparedDraw {
    request: reims_vgpu_core::DrawRequest,
    route: DrawCompletionRoute,
    render_target_resource: Option<std::sync::Arc<crate::model::TaskResource>>,
}

impl PreparedDraw {
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
        let Self {
            request,
            route,
            render_target_resource,
        } = self;
        let executor = std::sync::Arc::clone(&state.executor);
        let submission = executor::context_for(state, task_id);
        let receipt = executor::execute_draw(executor.as_ref(), submission, request)
            .map_err(|error| ExecutorDiagnostic::from_decline(&error))?;
        state
            .task_objects
            .resources
            .record_gpu_materializations(receipt.gpu_materialized.iter().copied());
        if let Some(resource) = render_target_resource {
            resource.note_render_target_use();
        }
        Ok(CompletedDraw {
            submission: receipt.submission.id,
            output: receipt.output,
            route,
        })
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
            false,
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
}
