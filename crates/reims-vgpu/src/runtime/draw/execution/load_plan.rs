//! Color-attachment LOAD source and resident-target planning.
//!
//! Guest seeds, host seeds, resident currency, clear state, and the target
//! identity consumed by completion planning are resolved together. Request
//! assembly cannot combine a seed from one decision with a resident target
//! from another observation of mutable state.

use std::sync::Arc;

use super::*;

pub(super) struct LoadPlan {
    pub target_rgba8: Option<Arc<Vec<u8>>>,
    pub target_guest: Option<reims_vgpu_memory::GuestTargetPlan>,
    pub target_clear: [f32; 4],
    pub color_load_action: reims_vgpu_core::ColorLoadAction,
    pub target_seed_order: reims_vgpu_core::SeedOrder,
    pub gpu_only_content_allowed: bool,
    pub surface_target: Option<crate::model::TargetIdentity>,
    pub load_from_target: bool,
    pub gva_load_identity: Option<crate::model::TargetIdentity>,
}

pub(super) fn plan_load<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    request: &DrawEncodeRequest,
    gva_allocation_generation: u64,
    writeback_guest: bool,
    width: u32,
    height: u32,
) -> LoadPlan {
    crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::Seed);
    let mut target_rgba8 = None;
    let mut target_clear = [0.0; 4];
    let mut color_load_action = reims_vgpu_core::ColorLoadAction::Clear;
    let mut target_seed_order = reims_vgpu_core::SeedOrder::Rgba8;
    let gpu_only_content_allowed = state.executor.capabilities().deferred_gpu_only_content;

    let surface_render_target = iosurface_texture_render_identity(state, request);
    let surface_target = writeback_guest
        .then(|| surface_render_target.clone())
        .flatten();
    let gva_guest_backing = gva_guest_target_backing(state, host, request);
    let surface_guest_backing = surface_render_target.as_ref().and_then(|identity| {
        let color = request.colors.first()?;
        try_iosurface_texture_target_guest_memory(
            state,
            host,
            color.mapping_id(),
            width,
            height,
            identity.resident_layout(),
        )
    });
    let mut load_from_target = request.chain_from_resident
        && render_chain_identity(state, request, gva_allocation_generation).is_some();
    let gva_load = resolve_gva_load_source(
        state,
        host,
        request,
        gva_allocation_generation,
        gva_guest_backing.as_ref(),
        &mut load_from_target,
    );
    let gva_load_identity = gva_load.identity;
    let mut target_guest_seed = gva_load.guest_seed;
    let cpu_seed = gva_load.cpu_seed;

    if !load_from_target {
        if let Some((identity, mapping_epoch)) =
            iosurface_texture_load_currency_query(state, request)
        {
            let resident_current = iosurface_texture_load_resident_is_current(|| {
                let resident_epoch = state.executor.resident_read_plan(&identity).content_epoch;
                iosurface_texture_resident_is_current(mapping_epoch, resident_epoch)
            });
            if resident_current {
                load_from_target = true;
                crate::runtime::drain::note_store_route("iosurface_texture_seed_elided");
                note_iosurface_texture_elision_extent(width, height);
            } else {
                crate::runtime::drain::note_store_route("iosurface_texture_seed_provided");
            }
        }
    }

    if let Some(color) = request.colors.first() {
        let pixel_count = u64::from(width).saturating_mul(u64::from(height));
        let (count_route, area_route) = color.load_action.census_routes();
        crate::runtime::drain::note_store_route(count_route);
        crate::runtime::drain::note_store_route_n(area_route, pixel_count);
        match color.load_action {
            reims_vgpu_protocol::pass_action::LoadAction::Load if load_from_target => {}
            reims_vgpu_protocol::pass_action::LoadAction::Clear => {
                target_clear = color.clear_color.map(|component| component as f32);
            }
            reims_vgpu_protocol::pass_action::LoadAction::Load => {
                let mut seed_door = "none";
                if target_guest_seed.is_some() {
                    seed_door = "gva_guest";
                } else if let Some(seed) = cpu_seed.as_ref().or(color.target_seed_rgba.as_ref()) {
                    seed_door = "color_seed";
                    if seed.len() == (width as usize) * (height as usize) * 4 {
                        target_rgba8 = Some(Arc::new(seed.clone()));
                    }
                } else if color.mapping_id() != 0 {
                    seed_door = "mapping";
                    let target_format = iosurface_texture_render_identity(state, request)
                        .as_ref()
                        .map(crate::model::TargetIdentity::resident_layout)
                        .unwrap_or(reims_vgpu_core::pixel_format::TexelLayout::Rgba8);
                    match resolve_iosurface_texture_load_seed(
                        state,
                        host,
                        color.mapping_id(),
                        width,
                        height,
                        target_format,
                    ) {
                        Some(IOSurfaceLoadSeed::Host(bytes, order)) => {
                            target_rgba8 = Some(bytes);
                            target_seed_order = order;
                        }
                        Some(IOSurfaceLoadSeed::Guest(seed)) => target_guest_seed = Some(seed),
                        None => {}
                    }
                }
                note_load_seed_outcome(
                    seed_door,
                    target_rgba8.is_some() || target_guest_seed.is_some(),
                    color,
                    width,
                    height,
                );
            }
            reims_vgpu_protocol::pass_action::LoadAction::DontCare => {
                color_load_action = reims_vgpu_core::ColorLoadAction::DontCare;
            }
        }
        if color.load_action == reims_vgpu_protocol::pass_action::LoadAction::Load {
            color_load_action = reims_vgpu_core::ColorLoadAction::Load;
        }
    }

    if let Some(scissor) = request.scissors.first() {
        note_draw_coverage(
            *scissor,
            width,
            height,
            request.colors.first().map(|color| color.load_action),
            target_rgba8.is_some() || target_guest_seed.is_some(),
            load_from_target,
        );
    }

    let target_guest = if let Some(memory) = gva_guest_backing.or(surface_guest_backing) {
        Some(reims_vgpu_memory::GuestTargetPlan::Backing {
            memory,
            seed: target_guest_seed,
        })
    } else {
        target_guest_seed.map(reims_vgpu_memory::GuestTargetPlan::Seed)
    };

    LoadPlan {
        target_rgba8,
        target_guest,
        target_clear,
        color_load_action,
        target_seed_order,
        gpu_only_content_allowed,
        surface_target,
        load_from_target,
        gva_load_identity,
    }
}
