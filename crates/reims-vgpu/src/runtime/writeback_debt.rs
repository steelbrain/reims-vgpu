//! Resource-validity ownership for render targets.
//!
//! A render Store preserves pixels in the host attachment. It does not imply a
//! host-to-guest transfer. The guest makes that transfer observable by naming
//! the resource in `CmdSynchronizeResources`, or this device needs the guest
//! bytes itself for a fallback reader. Until then, [`PendingWritebacks`] records
//! that the engine image is authoritative and repeated Stores into the resource
//! replace one another without touching guest RAM.
//!
//! # A resource owns its transfer backing
//!
//! Debts carry the generational resource identity, GVA declaration, geometry,
//! format, and resource generation. The live GVA resource separately retains the
//! ordered physical pages of its transfer backing. Ordinary task unmap changes
//! virtual-address bookkeeping but does not retarget that resource. Explicit
//! discard drops the transfer backing, and the next prepare or synchronize
//! resolves it again without replacing the host texture.
//!
//! This is the safety property the former deferred-window design lacked: it
//! parked raw host pointers across guest execution. This model retains page
//! identities, not pointers; every transfer still constructs bounded
//! `GuestSlice`s from the owning RAMBlock import.
//!
//! # Validity transitions decide direction
//!
//! A GPU Store makes the host image authoritative. A later guest write makes
//! the transfer backing newer; payment then abandons the host image rather than
//! overwriting the guest's work. Task-GVA resources use canonical resource
//! identity and content generations.
//!
//! A named synchronize pays only its object list through
//! [`submit_for_resources`]. Readers that know a texture call
//! [`pay_for_texture`]. Only a genuinely unnameable reader uses [`pay_all`].
//! Completion stamps alone do not publish resources.
//!
//! The engine's `gpu_only_content` flag keeps an unpaid image alive. A
//! successful payment calls `note_resident_content_copied_out`; replacement,
//! invalidation, task retirement, and generation movement release the same
//! ownership without inventing a guest write.
//!
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::Device;

pub use reims_vgpu_core::{GvaPlaneKey, GvaResourceKey, GvaWritebackDebt, PendingWritebacks};

pub(crate) fn resource_key(
    state: &Device,
    task_id: u32,
    texture_ref: u32,
) -> Option<GvaResourceKey> {
    Some(GvaResourceKey {
        task_id,
        resource: state
            .task_objects
            .resources
            .identity(task_id, texture_ref)?,
    })
}

fn resource_owner(state: &Device, key: GvaResourceKey) -> Option<(u32, u32)> {
    let (task, object) = state.task_objects.resources.owner(key.resource)?;
    (task.get() == key.task_id).then_some((task.get(), object.get()))
}

/// Pay every owed GVA resource frame.
pub fn pay_all<M: HostMemory + HostOps>(state: &mut Device, host: &mut M) {
    if state.content.pending_writebacks.is_empty() {
        return;
    }
    for plane in state.content.pending_writebacks.gvas_by_age() {
        let Some(debt) = state.content.pending_writebacks.take_gva_plane(plane) else {
            continue;
        };
        let _ = pay_gva(state, host, plane, debt, GvaPaySite::All);
    }
}

/// Pay every plane owed by one task-local GVA resource.
pub fn pay_for_texture<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
) {
    if state.content.pending_writebacks.is_empty() {
        return;
    }
    let Some(gva_key) = resource_key(state, task_id, texture_ref) else {
        return;
    };
    // Every plane the reference owes, not the one that sorts first: a sampled
    // read names the resource, and a mip pyramid's levels are separate debts.
    for (plane, debt) in state.content.pending_writebacks.take_gva(gva_key) {
        let _ = pay_gva(state, host, plane, debt, GvaPaySite::Named);
    }
}

/// The stable host-texture identity for the GVA resource a draw is declaring.
///
/// The first successful resolution retains the ordered physical pages that the
/// resource's transfer buffer names. Later calls return the same generation and
/// backing even if the task removes its virtual mapping. After explicit
/// discard, the next call may establish a replacement transfer backing while
/// preserving the host texture's generation.
///
/// # A changed declaration ends one lifetime and begins the next
///
/// This used to answer `0` and emit `gva_resource_refused
/// reason=declaration_changed` when the draw's `(gva, span)` differed from the
/// one the entry was established with, on the reading that a live resource
/// cannot move. The reading is right and the response was not: the resource did
/// not move, the *reference* was reused, and the entry describing the retired
/// object is the thing that has to go.
///
/// Answering `0` never recovered. The entry stayed, so every later draw into
/// that reference compared against the same dead declaration and refused again —
/// one macos-26 report carried 5 197 of these lines over 280 references, one of
/// them refused 803 times in a single boot. What `0` costs depends on which
/// caller asked: `draw::execution`'s resident resolve turns it into
/// `GvaResidentRefusal::NoGeneration` and loses the frame, while the secondary
/// MRT builder puts it straight into [`TargetIdentity::Gva`], where generation
/// zero is the one value that cannot distinguish two allocations — the
/// wrong-content class that identity exists to close.
///
/// So a differing declaration is handled as what it is, a lifetime boundary,
/// through the same [`retire_gva_resource`] that `CmdDeleteResource` uses: the
/// old generation's unpaid frame is released rather than written into storage
/// the retired object no longer owns — the rule [`retire_gva_for_task`] already
/// states for task teardown — and [`PendingWritebacks::ensure_gva_resource`]
/// then establishes the new object's own generation.
///
/// It stays fail-visible, because a *frequent* redeclaration would say something
/// different: that some producer in this device describes one live resource two
/// ways, in which case each draw would mint a generation and no resident could
/// ever be reused. The line names both declarations so that reading can be made
/// from a log rather than from a rebuild.
///
/// [`TargetIdentity::Gva`]: crate::model::TargetIdentity::Gva
pub fn gva_resource_generation<M: HostMemory>(
    state: &mut Device,
    host: &M,
    key: GvaResourceKey,
    gva: u64,
    span: u64,
) -> u64 {
    if let Some((generation, declared_span, has_pages)) = state
        .content
        .pending_writebacks
        .gva_resource_status(key.plane(gva))
    {
        if declared_span == span {
            if has_pages {
                return generation;
            }
        } else {
            crate::observe::Emit::decline(
                "gva_resource_redeclared",
                &GvaResourceRedeclared {
                    gva,
                    was_span: declared_span,
                    now_span: span,
                },
            )
            .field("task", key.task_id)
            .field("resource", key.resource.index())
            .field("resource_generation", key.resource.generation())
            .fail();
            retire_gva_resource_by_key(state, key);
        }
    }
    let page_size = state.page_size();
    let ordered = crate::runtime::gva_mem::task_gva_page_gpas(
        host,
        &state.tasks,
        key.task_id,
        gva,
        span,
        state.page_shift,
    );
    let want = reims_vgpu_paging::span::pages_spanned(gva, span, page_size);
    let pages = (ordered.len() as u64 == want).then_some(ordered);
    state
        .content
        .pending_writebacks
        .ensure_gva_resource(key, gva, span, pages)
}

/// One plane of a reference observed at two different lengths.
///
/// A *different address* under one reference is not this: that is another plane
/// of the same resource — a mip level — and [`GvaPlaneKey`] gives it its own
/// entry. What remains here is one address whose length moved, which the
/// contract has no room for, so the reference has been reused for a second
/// object.
///
/// Carries both lengths because neither alone says anything: the question a
/// reader has is whether they are *stable* — a reference reused, ordinary guest
/// lifetime — or whether they alternate, which would be this device describing
/// one plane two ways.
struct GvaResourceRedeclared {
    gva: u64,
    was_span: u64,
    now_span: u64,
}

impl crate::observe::Decline for GvaResourceRedeclared {
    fn slug(&self) -> &'static str {
        "gva_resource_declaration_changed"
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("gva", format!("{:#x}", self.gva)),
            ("was_span", self.was_span.to_string()),
            ("now_span", self.now_span.to_string()),
        ]
    }
}

crate::observe::decline_display!(GvaResourceRedeclared);

/// Re-establish the transfer backing of the plane a debt names, without any
/// power to declare one.
///
/// The payment path's counterpart to [`gva_resource_generation`]. It asks only
/// the question payment has standing to ask — "does this plane still exist, and
/// does it still have its pages" — using the plane's *own* span, never the
/// debt's. A debt that outlived its plane therefore finds nothing here and is
/// released by the caller, where before it reached
/// [`PendingWritebacks::ensure_gva_resource`] and could re-create the retired
/// object at the dead declaration it was carrying.
fn reback_gva_resource<M: HostMemory>(state: &mut Device, host: &M, plane: GvaPlaneKey) -> bool {
    let Some((_, span, has_pages)) = state.content.pending_writebacks.gva_resource_status(plane)
    else {
        return false;
    };
    if has_pages {
        return true;
    }
    let page_size = state.page_size();
    let ordered = crate::runtime::gva_mem::task_gva_page_gpas(
        host,
        &state.tasks,
        plane.resource.task_id,
        plane.gva,
        span,
        state.page_shift,
    );
    let want = reims_vgpu_paging::span::pages_spanned(plane.gva, span, page_size);
    let pages = (ordered.len() as u64 == want).then_some(ordered);
    state
        .content
        .pending_writebacks
        .reback_gva_resource(plane, pages)
}

/// Record a GVA render result as host-authoritative without touching guest
/// pages. Returns `false` when the attachment has no resource identity and must
/// use the eager transfer path.
pub fn arm_gva<M: HostMemory + HostOps>(
    state: &mut Device,
    _host: &mut M,
    task_id: u32,
    c0: &crate::runtime::draw::ColorRtRequest,
    identity: &crate::model::TargetIdentity,
    submission: reims_vgpu_protocol::SubmissionId,
) -> bool {
    let Some((generation, resident_layout)) = (match *identity {
        crate::model::TargetIdentity::Gva {
            generation, format, ..
        } => Some((generation, format)),
        _ => None,
    }) else {
        return false;
    };
    if c0.texture_ref == 0 || generation == 0 {
        return false;
    }
    // Every older host-side spelling of this resource is stale as soon as the
    // render finishes. In particular, a compute storage resident and the
    // linear byte cache can otherwise sit above the guest-page reader and serve
    // the frame that preceded this Store indefinitely.
    state.invalidate_object_host_copies(task_id, c0.texture_ref);
    crate::runtime::surface_cache::evict_gva(state, c0.target_gva());
    let Some(linear) = c0.linear_target().copied() else {
        return false;
    };
    let Some(content) = state.task_objects.resources.record_completed_gpu_store(
        task_id,
        c0.texture_ref,
        submission,
    ) else {
        return false;
    };
    let key = GvaResourceKey {
        task_id,
        resource: content.0,
    };
    let debt = GvaWritebackDebt {
        linear,
        width: c0.width,
        height: c0.height,
        format: c0.format,
        resident_layout,
        generation,
        content: Some(content),
        guest_write: state.resource_write_stamp(task_id, c0.texture_ref),
        seq: 0,
    };
    let previous = state.content.pending_writebacks.arm_gva(key, debt);
    if previous.is_some() {
        crate::runtime::drain::note_store_route("gvadebt_superseded");
    }
    crate::runtime::drain::note_store_route("gvadebt_armed");
    if let Some(previous) = previous.filter(|previous| !same_gva_identity(*previous, debt)) {
        release_gva(state.executor.as_ref(), previous);
    }
    true
}

/// Whether this exact GVA resident is the host-authoritative copy named by an
/// unpaid resource debt.
pub fn gva_resident_authoritative(state: &Device, identity: &crate::model::TargetIdentity) -> bool {
    let Some((plane, debt)) = state.content.pending_writebacks.gva_for_identity(identity) else {
        return false;
    };
    state
        .resource_write_stamp_for(plane.resource.resource)
        .is_some_and(|stamp| stamp.quiet_since(debt.guest_write))
}

/// Retire host-authoritative resources whose task-local references are about to
/// be replaced. The pixels are deliberately not copied: after this lifecycle
/// transition the old object no longer names guest storage to synchronize.
pub fn retire_gva_for_task(state: &mut Device, task_id: u32) -> usize {
    let keys = state.content.pending_writebacks.gvas_for_task(task_id);
    let mut retired = 0;
    for key in keys {
        let (_, debts) = state.content.pending_writebacks.retire_gva_resource(key);
        retired += 1;
        for debt in debts {
            release_gva(state.executor.as_ref(), debt);
        }
    }
    if retired != 0 {
        crate::runtime::drain::note_store_route_n("gvadebt_retired_task", retired as u64);
    }
    retired
}

/// Retire one resource at its explicit lifetime boundary.
pub fn retire_gva_resource(state: &mut Device, task_id: u32, texture_ref: u32) -> bool {
    let Some(key) = resource_key(state, task_id, texture_ref) else {
        return false;
    };
    retire_gva_resource_by_key(state, key)
}

fn retire_gva_resource_by_key(state: &mut Device, key: GvaResourceKey) -> bool {
    let (existed, debts) = state.content.pending_writebacks.retire_gva_resource(key);
    let owed = !debts.is_empty();
    for debt in debts {
        release_gva(state.executor.as_ref(), debt);
    }
    existed || owed
}

/// Release named resources' retained transfer backings.
pub fn discard_gva_resources(state: &mut Device, task_id: u32, object_ids: &[u32]) -> usize {
    let resources: Vec<_> = object_ids
        .iter()
        .filter_map(|&object_id| resource_key(state, task_id, object_id))
        .collect();
    state
        .content
        .pending_writebacks
        .discard_gva_resources(resources)
}

fn same_gva_identity(a: GvaWritebackDebt, b: GvaWritebackDebt) -> bool {
    a.linear.target_gva() == b.linear.target_gva()
        && a.width == b.width
        && a.height == b.height
        && a.generation == b.generation
        && a.resident_layout == b.resident_layout
}

/// The engine resident one armed GVA debt names.
///
/// `pub(crate)` because a debt is not only something to pay: a reader that wants
/// the *content* rather than the guest's copy of it — the blit rail's whole-plane
/// GPU arm — needs exactly this identity, and deriving a second one from the same
/// debt fields is how two spellings of one resident start disagreeing. There is
/// one derivation and it is here.
pub(crate) fn gva_identity(debt: GvaWritebackDebt) -> crate::model::TargetIdentity {
    crate::model::TargetIdentity::Gva {
        gva: debt.linear.target_gva(),
        width: debt.width,
        height: debt.height,
        generation: debt.generation,
        format: debt.resident_layout,
    }
}

fn release_gva(executor: &dyn crate::runtime::executor::Executor, debt: GvaWritebackDebt) {
    executor.note_resident_content_copied_out(&gva_identity(debt));
}

/// Wait only for submitted writes that can reach one mapping's pages.
///
/// The page set comes from [`Device::mapping_reach_pages`], the same rule
/// the write path uses for its destination. A mapping that cannot name its pages
/// answers `None`, which conservatively waits.
pub fn settle_for_mapping(
    state: &mut Device,
    mapping_id: u32,
    site: crate::runtime::render_writeback::SettleSite,
) {
    let reach_started = std::time::Instant::now();
    let s = &*state;
    crate::runtime::render_writeback::settle_guest_writes_unless_disjoint(
        state.executor.as_ref(),
        site,
        || {
            crate::runtime::drain::note_store_route("wbdebt_reach_walk_n");
            s.mapping_reach_pages(mapping_id)
        },
    );
    crate::runtime::drain::note_store_route_us(
        "wbdebt_reach_us",
        reach_started.elapsed().as_micros() as u64,
    );
}

/// Materialize one named GVA resource, then wait for submitted writes that can
/// reach the task-GVA span a CPU reader is about to access.
pub fn settle_for_texture<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    gva: u64,
    span: u64,
    site: crate::runtime::render_writeback::SettleSite,
) {
    pay_for_texture(state, host, task_id, texture_ref);
    let (tasks, page_shift, page_size) = (&state.tasks, state.page_shift, state.page_size());
    crate::runtime::render_writeback::settle_guest_writes_unless_disjoint(
        state.executor.as_ref(),
        site,
        || {
            let want = reims_vgpu_paging::span::pages_spanned(gva, span, page_size);
            let gpas = crate::runtime::gva_mem::task_gva_page_gpas(
                host, tasks, task_id, gva, span, page_shift,
            );
            (gpas.len() as u64 == want).then_some(gpas)
        },
    );
}

/// [`settle_for_mapping`] for a caller that cannot name the mapping it is about
/// to touch, so it owes every debt.
pub fn settle_unnamed<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    site: crate::runtime::render_writeback::SettleSite,
) {
    pay_all(state, host);
    crate::runtime::render_writeback::settle_guest_writes(state.executor.as_ref(), site);
}

/// Submit exactly the resources named by an asynchronous synchronize command.
///
/// The object list is the scope of the API operation; an unrelated host-valid
/// texture remains resident-authoritative. Completion belongs to the FIFO: the
/// transfers recorded here precede that packet's queue point, and its pending
/// stamp publishes only after that point completes. Waiting here would turn the
/// asynchronous command into a device-wide drain and then make the stamp wait a
/// second time for work already known complete.
pub fn submit_for_resources<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    object_ids: &[u32],
) {
    for &object_id in object_ids {
        pay_for_texture(state, host, task_id, object_id);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GvaPaySite {
    Named,
    All,
}

impl GvaPaySite {
    fn route(self) -> &'static str {
        match self {
            Self::Named => "gvadebt_paid_named",
            Self::All => "gvadebt_paid_all",
        }
    }
}

/// Materialize one host-authoritative GVA resource into its retained transfer
/// backing. After explicit discard, synchronize lazily recreates that backing;
/// ordinary virtual-memory unmap does not participate in resource lifetime.
fn pay_gva<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    plane: GvaPlaneKey,
    debt: GvaWritebackDebt,
    site: GvaPaySite,
) -> bool {
    let key = plane.resource;
    let identity = gva_identity(debt);
    let Some((task_id, texture_ref)) = resource_owner(state, key) else {
        crate::runtime::drain::note_store_route("gvadebt_resource_retired");
        release_gva(state.executor.as_ref(), debt);
        return true;
    };
    let Some(now) = state.resource_write_stamp_for(key.resource) else {
        crate::runtime::drain::note_store_route("gvadebt_resource_retired");
        release_gva(state.executor.as_ref(), debt);
        return true;
    };
    if !now.quiet_since(debt.guest_write) {
        crate::runtime::drain::note_store_route("gvadebt_abandoned_guest_wrote");
        release_gva(state.executor.as_ref(), debt);
        return true;
    }
    let Some(span) = u64::from(debt.linear.row_stride).checked_mul(u64::from(debt.height)) else {
        crate::observe::fail(format!(
            "gvadebt_pay_lost task={} texture={} reason=span_overflow",
            task_id, texture_ref
        ));
        release_gva(state.executor.as_ref(), debt);
        return true;
    };
    // The resource's own declaration decides whether its pages come back, not
    // this debt's — see [`reback_gva_resource`]. A debt whose resource is gone
    // names storage that object no longer owns, so it is released here rather
    // than restored: restoring one would park it in the ledger forever, since
    // nothing retired can grow pages back.
    if !reback_gva_resource(state, host, plane) {
        crate::runtime::drain::note_store_route("gvadebt_resource_retired");
        release_gva(state.executor.as_ref(), debt);
        return true;
    }
    let Some((backing_generation, backing_span, ordered)) =
        state.content.pending_writebacks.gva_resource_backing(plane)
    else {
        state.content.pending_writebacks.restore_gva(plane, debt);
        crate::runtime::drain::note_store_route(match site {
            GvaPaySite::Named => "gvadebt_named_unmapped",
            GvaPaySite::All => "gvadebt_all_unmapped",
        });
        if site == GvaPaySite::Named {
            crate::observe::fail(format!(
                "gvadebt_pay_blocked task={} texture={} reason=span_unresolved",
                task_id, texture_ref
            ));
        }
        return false;
    };
    // The plane key already carries the address, so a mismatched one cannot
    // reach here: it would have found no plane at all above.
    if backing_generation != debt.generation || backing_span != span {
        crate::runtime::drain::note_store_route("gvadebt_generation_moved");
        release_gva(state.executor.as_ref(), debt);
        return true;
    }
    let pages = crate::runtime::draw::StoreTargetPages::from_ordered(&ordered, span);
    let request = crate::runtime::draw::ColorRtRequest {
        texture_ref,
        storage: crate::runtime::draw::ColorTargetStorage::Linear(debt.linear),
        width: debt.width,
        height: debt.height,
        format: debt.format,
        store_action: reims_vgpu_protocol::pass_action::StoreAction::Store,
        ..Default::default()
    };
    crate::runtime::drain::note_store_route(site.route());
    if let Err(reason) = crate::runtime::render_writeback::store_gva_frame(
        state,
        host,
        task_id,
        &identity,
        &request,
        texture_ref,
        Some(&pages),
    ) {
        // Through the builder rather than by interpolating the decline, which
        // renders its own `reason=` and produced `reason=reason=<slug>` — a line
        // the standard ranking grep drops. The builder also carries the
        // decline's own fields, so the `via=` that says which check inside the
        // store refused now reaches the log instead of being formatted away.
        crate::observe::Emit::decline("gvadebt_pay_lost", &reason)
            .field("task", task_id)
            .field("texture", texture_ref)
            .fail();
        release_gva(state.executor.as_ref(), debt);
    } else if let Some((resource, version)) = debt.content {
        if !state
            .task_objects
            .resources
            .record_gpu_to_guest_copy(resource, version)
        {
            crate::observe::fail(format!(
                "gvadebt_content_transition task={} texture={} reason=stale_content_version",
                task_id, texture_ref
            ));
        }
    }
    true
}

#[cfg(test)]
mod tests {

    use super::*;

    fn key(task_id: u32, reference: u32) -> GvaResourceKey {
        GvaResourceKey {
            task_id,
            resource: reims_vgpu_protocol::ResourceId::new(reference, 1),
        }
    }

    fn register_key(state: &mut Device, task_id: u32, reference: u32) -> GvaResourceKey {
        let resource = std::sync::Arc::new(crate::model::TaskResource::new(
            reims_vgpu_protocol::ObjectListEntry::new(
                reims_vgpu_protocol::ObjectKind::Buffer,
                0,
                0,
            ),
            std::sync::Arc::from([]),
        ));
        let resource = state
            .task_objects
            .resources
            .register(task_id, reference, resource);
        GvaResourceKey {
            task_id,
            resource: resource.semantic_id().unwrap(),
        }
    }

    fn gva_debt(generation: u64) -> GvaWritebackDebt {
        GvaWritebackDebt {
            linear: crate::runtime::draw::LinearColorTarget {
                allocation_gva: 0x4000,
                allocation_size: 64 * 256,
                plane_offset: 0,
                row_stride: 256,
            },
            width: 64,
            height: 64,
            format: reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM,
            resident_layout: reims_vgpu_core::pixel_format::TexelLayout::Bgra8,
            generation,
            content: None,
            guest_write: Default::default(),
            seq: 0,
        }
    }

    /// The resource reference, not the GVA, owns coherence. Reusing the same
    /// resource for another Store replaces its debt exactly as repeated Stores
    /// into one IOSurface do.
    #[test]
    fn a_second_gva_store_on_one_resource_replaces_the_first() {
        let mut pending = PendingWritebacks::default();
        let key = key(3, 19);
        assert_eq!(pending.arm_gva(key, gva_debt(7)), None);
        let previous = pending.arm_gva(key, gva_debt(8));
        assert_eq!(previous.map(|debt| debt.generation), Some(7));
        assert_eq!(pending.len(), 1);
        assert_eq!(pending.get_gva(key).map(|debt| debt.generation), Some(8));
    }

    /// GVA resources have protocol lifetime, not an arbitrary ledger capacity.
    #[test]
    fn gva_debts_are_not_evicted_by_capacity() {
        let mut pending = PendingWritebacks::default();
        const DISTINCT_RESOURCES: u32 = 64;
        for texture_ref in 1..=DISTINCT_RESOURCES {
            let key = key(2, texture_ref);
            pending.ensure_gva_resource(
                key,
                u64::from(texture_ref) << 16,
                4096,
                Some(vec![u64::from(texture_ref) << 12]),
            );
            assert_eq!(pending.arm_gva(key, gva_debt(texture_ref.into())), None);
        }
        assert_eq!(pending.len(), DISTINCT_RESOURCES as usize);
        assert_eq!(pending.gvas_by_age().len(), DISTINCT_RESOURCES as usize);
    }

    /// Ordinary virtual-memory bookkeeping does not retarget a live resource.
    /// A repeated prepare with a different walk keeps the original transfer
    /// backing until the protocol explicitly discards it.
    #[test]
    fn a_live_resource_retains_its_backing_until_discard() {
        let mut pending = PendingWritebacks::default();
        let key = key(3, 19);
        let generation = pending.ensure_gva_resource(key, 0x4000, 4096, Some(vec![0x9000]));
        assert_eq!(
            pending.ensure_gva_resource(key, 0x4000, 4096, Some(vec![0xa000])),
            generation
        );
        assert_eq!(
            &*pending.gva_resource_backing(key.plane(0x4000)).unwrap().2,
            &[0x9000]
        );

        assert_eq!(pending.discard_gva_resources([key]), 1);
        assert!(pending.gva_resource_backing(key.plane(0x4000)).is_none());
        assert_eq!(
            pending.ensure_gva_resource(key, 0x4000, 4096, Some(vec![0xa000])),
            generation,
            "discard replaces the transfer backing, not the host texture"
        );
        assert_eq!(
            &*pending.gva_resource_backing(key.plane(0x4000)).unwrap().2,
            &[0xa000]
        );
    }

    /// Delete is the resource lifetime boundary. Reusing the same task-local
    /// reference after delete receives a new host-texture identity.
    #[test]
    fn deleting_and_recreating_a_resource_changes_its_generation() {
        let mut pending = PendingWritebacks::default();
        let key = key(3, 19);
        let first = pending.ensure_gva_resource(key, 0x4000, 4096, Some(vec![0x9000]));
        assert!(pending.retire_gva_resource(key).0);
        let second = pending.ensure_gva_resource(key, 0x4000, 4096, Some(vec![0xa000]));
        assert_ne!(first, second);
    }

    /// Delete is the *announced* lifetime boundary; a plane's length moving at
    /// one address is the same boundary observed instead of announced. A
    /// plane's length is fixed for its life, so this is a different object in a
    /// reused slot and it must get a different host texture.
    ///
    /// Asserting the third call is what makes that visible: a fix that only
    /// stopped refusing, without replacing the entry, still fails here.
    #[test]
    fn one_plane_redeclared_at_a_new_length_is_a_new_resource() {
        let mut pending = PendingWritebacks::default();
        let key = key(3, 19);
        let first = pending.ensure_gva_resource(key, 0x4000, 4096, Some(vec![0x9000]));
        let second = pending.ensure_gva_resource(key, 0x4000, 8192, Some(vec![0xa000, 0xb000]));
        assert_ne!(first, second, "a new length is a new host texture");
        assert_eq!(
            &*pending.gva_resource_backing(key.plane(0x4000)).unwrap().2,
            &[0xa000, 0xb000],
            "the new object's pages replace the retired one's"
        );
        assert_eq!(
            pending.ensure_gva_resource(key, 0x4000, 8192, None),
            second,
            "the new declaration is the live one, so it is stable"
        );
    }

    /// A mip pyramid is one resource with several live planes, and the ledger
    /// has to hold all of them at once.
    ///
    /// Measured on a driven macos-26 boot: one reference cycling three
    /// contiguous declarations in exact 4:1 ratios — 256x192, 128x96, 64x48 of
    /// one RGBA8 allocation, the compositor's blur/backdrop pyramid. Keyed by
    /// the reference, each level change replaced the entry, so no level's
    /// resident could ever be reused and arming one level's Store dropped the
    /// previous level's unpaid frame. Both halves are asserted here: the
    /// generations are distinct **and** stable, and three debts coexist.
    #[test]
    fn the_levels_of_one_pyramid_are_separate_planes_of_one_resource() {
        let mut pending = PendingWritebacks::default();
        let key = key(1, 135);
        let levels = [
            (0x11af000_u64, 196_608_u64),
            (0x11df000, 49_152),
            (0x11eb000, 12_288),
        ];
        let generations: Vec<u64> = levels
            .iter()
            .map(|&(gva, span)| pending.ensure_gva_resource(key, gva, span, Some(vec![gva])))
            .collect();
        let mut distinct = generations.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), 3, "each level is its own host texture");

        // The cycle the boot showed: re-declaring level 0 after levels 1 and 2
        // must return level 0's own generation, not mint a fourth.
        for (i, &(gva, span)) in levels.iter().enumerate() {
            assert_eq!(
                pending.ensure_gva_resource(key, gva, span, None),
                generations[i],
                "a live plane is stable across its siblings"
            );
        }

        for (i, &(gva, _)) in levels.iter().enumerate() {
            let mut debt = gva_debt(generations[i]);
            debt.linear.allocation_gva = gva;
            assert_eq!(
                pending.arm_gva(key, debt),
                None,
                "arming one level must not supersede another"
            );
        }
        assert_eq!(
            pending.take_gva(key).len(),
            3,
            "the resource owes all three"
        );
    }

    /// A guest validity transition after the Store makes guest memory newer
    /// than the held resident. The debt remains available for an orderly
    /// abandon, but it must immediately stop licensing host-resident reads.
    #[test]
    fn a_guest_write_revokes_gva_resident_authority() {
        let mut state = Device::new(crate::model::DeviceId::default(), 12);
        let key = register_key(&mut state, 4, 12);
        let mut debt = gva_debt(99);
        debt.guest_write = state.resource_write_stamp_for(key.resource).unwrap();
        let _ = state.content.pending_writebacks.arm_gva(key, debt);
        let identity = gva_identity(debt);
        assert!(gva_resident_authoritative(&state, &identity));
        state
            .task_objects
            .resources
            .note_guest_write_by_id(key.resource);
        assert!(!gva_resident_authoritative(&state, &identity));
        assert!(state.content.pending_writebacks.get_gva(key).is_some());
    }
}
