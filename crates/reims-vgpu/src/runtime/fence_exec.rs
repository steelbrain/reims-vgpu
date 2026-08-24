//! Product-path event + encoder fence sync (event / blit / compute / render).
//!
//! Uses [`reims_vgpu_core::synchronization`] for planning and
//! device-owned typed event and fence namespaces for storage. Unsatisfied waits are
//! soft-pending and do not block the drain (unified-memory in-order path).
//! Event timeouts are fail-closed as unsupported (no deferred timer).

use crate::runtime::decode::event::{Command as EventCommand, Kind as EventCmdKind};
use crate::runtime::Device;
use reims_vgpu_core::{
    plan_event, plan_fence, EventKind, FenceAction, SynchronizationDecision as Decision,
    SynchronizationDomain as Domain,
};

/// Outcome of a product-path event or encoder fence operation.
///
/// `Unsupported` **carries the check that refused**. Seven distinct causes reach
/// it — a bad fence domain, an event on the fence path, either wait-with-timeout
/// form, either invalid plan, an unknown event kind, and a blit reason forwarded
/// by the encoder remap — and while the reason was named in a hand-rolled
/// `format!` at each site, the *value* lost it the moment it was returned. So
/// `blit_exec`'s remap back into `BlitStatus` flattened all seven into one
/// `fence_unsupported` slug, and no caller could tell them apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FenceStatus {
    Ok,
    /// Wait with no prior update/signal, or signal value not yet reached (soft).
    Pending,
    /// Zero ref.
    Missing,
    /// Refused; the payload is the registered slug naming which check refused.
    Unsupported(&'static str),
}

impl crate::observe::Refusal for FenceStatus {
    /// `Ok` and `Pending` are control flow. `Missing` is `ref == 0` — the
    /// genuinely-unbound case `AGENTS.md` carves out by name, and the guest polls
    /// it, so logging it would flood.
    ///
    /// One caveat a reader needs: `exec`'s blit-fence arm remaps
    /// `BlitStatus::MissingResource` into `Missing`, which is a real failure
    /// rather than an unbound ref. That site logs it through the blit rail's own
    /// reason channel, so it is not silent — but the two meanings do share this
    /// variant, and separating them belongs with `BlitStatus`'s own migration.
    fn refusal(&self) -> Option<&'static str> {
        match self {
            Self::Ok | Self::Pending | Self::Missing => None,
            Self::Unsupported(slug) => Some(slug),
        }
    }
}

/// Emit a refused [`FenceStatus`] once per `(reason, ref)` and return it
/// unchanged, so a refusing site is one `return refused(..)` rather than a
/// `format!` beside a bare variant.
///
/// This replaces the file's own `fail_once` + reason-slug latch, which duplicated
/// [`crate::observe::Emit::fail_once`] down to the `HashSet`. The latch key
/// changes with the move, deliberately: it was the reason alone, so the *second*
/// ref to hit a class was silent and a guest with two bad fences was
/// indistinguishable from a guest with one. Keying on the ref is what `AGENTS.md`
/// prescribes, and the per-op fail counters (`event_ops_fail`, `*_fences_fail`)
/// still carry ongoing magnitude.
///
/// Runs on the drain worker, off the QEMU main core.
fn refused(
    status: FenceStatus,
    reference: u32,
    fields: impl FnOnce(crate::observe::Emit) -> crate::observe::Emit,
) -> FenceStatus {
    if let Some(e) = crate::observe::Emit::refusal("fence_exec_fail", &status) {
        fields(e).fail_once(u64::from(reference));
    }
    status
}

/// Execute fence update or wait on the given encoder domain (blit/compute/render).
pub fn execute_fence(
    state: &mut Device,
    task_id: u32,
    domain: Domain,
    fence_ref: u32,
    action: FenceAction,
) -> FenceStatus {
    execute_fence_scoped(state, task_id, domain, fence_ref, action, None)
}

/// Execute a render fence while retaining the exact producer stages carried by
/// `updateFence:afterStages:`. Wait stages belong to the dependency emitted by
/// the render stream and are therefore not stored here.
pub fn execute_render_fence(
    state: &mut Device,
    task_id: u32,
    fence_ref: u32,
    action: FenceAction,
    stages: reims_vgpu_core::RenderBarrierStages,
) -> FenceStatus {
    execute_fence_scoped(
        state,
        task_id,
        Domain::RenderFence,
        fence_ref,
        action,
        Some(stages),
    )
}

fn execute_fence_scoped(
    state: &mut Device,
    task_id: u32,
    domain: Domain,
    fence_ref: u32,
    action: FenceAction,
    render_stages: Option<reims_vgpu_core::RenderBarrierStages>,
) -> FenceStatus {
    if fence_ref == 0 {
        return FenceStatus::Missing;
    }
    if domain == Domain::Unknown {
        return refused(
            FenceStatus::Unsupported("fence_domain_unknown"),
            fence_ref,
            |e| {
                e.field("task", task_id)
                    .field("domain", format!("{domain:?}"))
                    .field("action", format!("{action:?}"))
            },
        );
    }
    if domain == Domain::Event {
        return refused(
            FenceStatus::Unsupported("fence_event_in_fence_path"),
            fence_ref,
            |e| {
                e.field("task", task_id)
                    .field("action", format!("{action:?}"))
            },
        );
    }
    let current = state.fence_generation(task_id, fence_ref);
    let plan = plan_fence(action, domain, current);
    if plan.updates_state {
        let signal = match domain {
            Domain::BlitFence => reims_vgpu_core::FenceSignal::Blit,
            Domain::ComputeFence => reims_vgpu_core::FenceSignal::Compute,
            Domain::RenderFence => reims_vgpu_core::FenceSignal::Render(
                render_stages.unwrap_or(reims_vgpu_core::RenderBarrierStages::ALL),
            ),
            Domain::Unknown | Domain::Event => unreachable!("fence domain checked above"),
        };
        state.set_fence_update(task_id, fence_ref, plan.update_value, signal);
    }
    let status = match plan.decision {
        Decision::SignalUpdate | Decision::SignalNoop | Decision::WaitSatisfied => FenceStatus::Ok,
        Decision::WaitPending => FenceStatus::Pending,
        Decision::WaitTimeoutUnsupported => refused(
            FenceStatus::Unsupported("fence_wait_timeout_unsupported"),
            fence_ref,
            |e| {
                e.field("task", task_id)
                    .field("domain", format!("{domain:?}"))
            },
        ),
        Decision::Invalid => refused(
            FenceStatus::Unsupported("fence_plan_invalid"),
            fence_ref,
            |e| {
                e.field("task", task_id)
                    .field("domain", format!("{domain:?}"))
                    .field("action", format!("{action:?}"))
            },
        ),
    };
    note_fence_route(domain, action, status);
    status
}

/// Count the encoder, operation, and outcome at the one shared fence seam.
///
/// Successful render and compute fences otherwise leave no observable trace:
/// their stream handlers intercept the records before the encoder-specific
/// counters see them. Keeping this census beside the state transition makes a
/// boot able to say which domains actually rely on the fence contract, while
/// the always-on failure path remains reserved for work that was lost.
fn note_fence_route(domain: Domain, action: FenceAction, status: FenceStatus) {
    let route = match (domain, action, status) {
        (Domain::BlitFence, FenceAction::Update, FenceStatus::Ok) => "fence_blit_update_ok",
        (Domain::BlitFence, FenceAction::Wait, FenceStatus::Ok) => "fence_blit_wait_ok",
        (Domain::BlitFence, FenceAction::Wait, FenceStatus::Pending) => "fence_blit_wait_pending",
        (Domain::ComputeFence, FenceAction::Update, FenceStatus::Ok) => "fence_compute_update_ok",
        (Domain::ComputeFence, FenceAction::Wait, FenceStatus::Ok) => "fence_compute_wait_ok",
        (Domain::ComputeFence, FenceAction::Wait, FenceStatus::Pending) => {
            "fence_compute_wait_pending"
        }
        (Domain::RenderFence, FenceAction::Update, FenceStatus::Ok) => "fence_render_update_ok",
        (Domain::RenderFence, FenceAction::Wait, FenceStatus::Ok) => "fence_render_wait_ok",
        (Domain::RenderFence, FenceAction::Wait, FenceStatus::Pending) => {
            "fence_render_wait_pending"
        }
        _ => return,
    };
    crate::runtime::drain::note_store_route(route);
}

/// Execute a decoded ch-event segment command (signal / wait / wait-timeout).
///
/// Signal advances the Event-domain table with the explicit wire value (monotonic
/// advance only). Wait is satisfied when the stored value is present and
/// `>= target`. Wait-with-timeout is unsupported (no host timer). Soft-pending
/// waits do not block drain.
pub fn execute_event(state: &mut Device, task_id: u32, cmd: &EventCommand) -> FenceStatus {
    if cmd.event_ref == 0 {
        return FenceStatus::Missing;
    }
    let kind = match cmd.kind {
        EventCmdKind::SignalEvent => EventKind::Signal,
        EventCmdKind::WaitEvent => EventKind::Wait,
        EventCmdKind::Unknown => {
            return refused(
                FenceStatus::Unsupported("event_kind_unknown"),
                cmd.event_ref,
                |e| e.field("task", task_id).field("value", cmd.value),
            );
        }
    };
    let current = state.event_generation(task_id, cmd.event_ref);
    let plan = plan_event(kind, cmd.value, cmd.has_timeout, current);
    if plan.updates_state {
        state.set_event_generation(task_id, cmd.event_ref, plan.update_value);
    }
    match plan.decision {
        Decision::SignalUpdate | Decision::SignalNoop | Decision::WaitSatisfied => FenceStatus::Ok,
        Decision::WaitPending => FenceStatus::Pending,
        Decision::WaitTimeoutUnsupported => refused(
            FenceStatus::Unsupported("event_wait_timeout_unsupported"),
            cmd.event_ref,
            |e| e.field("task", task_id).field("timeout", cmd.timeout),
        ),
        Decision::Invalid => refused(
            FenceStatus::Unsupported("event_plan_invalid"),
            cmd.event_ref,
            |e| {
                e.field("task", task_id)
                    .field("value", cmd.value)
                    .field("has_timeout", cmd.has_timeout)
            },
        ),
    }
}

#[cfg(test)]
mod tests {
    use reims_vgpu_wire::OP_HEADER_LEN;

    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
    use crate::runtime::decode::event::{
        OP_SIGNAL_EVENT, OP_WAIT_EVENT, OP_WAIT_EVENT_TIMEOUT, SIGNAL_WAIT_PAYLOAD_LEN, TIMEOUT,
        WAIT_TIMEOUT_PAYLOAD_LEN,
    };
    use reims_vgpu_core::endian::{st32, st64};

    fn event_cmd(opcode: u32, event_ref: u32, value: u64, timeout: Option<u32>) -> EventCommand {
        let mut payload = if timeout.is_some() {
            vec![0u8; WAIT_TIMEOUT_PAYLOAD_LEN]
        } else {
            vec![0u8; SIGNAL_WAIT_PAYLOAD_LEN]
        };
        st32(&mut payload[0..4], event_ref);
        st64(&mut payload[4..12], value);
        if let Some(t) = timeout {
            st32(&mut payload[TIMEOUT..TIMEOUT + 4], t);
        }
        let mut bytes = vec![0u8; OP_HEADER_LEN + payload.len()];
        st32(&mut bytes[0..4], opcode);
        st32(&mut bytes[4..8], (OP_HEADER_LEN + payload.len()) as u32);
        bytes[OP_HEADER_LEN..].copy_from_slice(&payload);
        crate::runtime::decode::event::decode(&bytes).expect("build event cmd")
    }

    #[test]
    fn blit_compute_and_render_share_one_fence_object() {
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        // The encoder selects the operation syntax, not a separate allocator.
        assert_eq!(
            execute_fence(&mut state, 1, Domain::BlitFence, 5, FenceAction::Update),
            FenceStatus::Ok
        );
        assert_eq!(
            execute_fence(&mut state, 1, Domain::ComputeFence, 5, FenceAction::Wait,),
            FenceStatus::Ok
        );
        assert_eq!(
            execute_fence(&mut state, 1, Domain::RenderFence, 5, FenceAction::Update,),
            FenceStatus::Ok
        );
        assert_eq!(state.fence_generation(1, 5), Some(2));
        // Any encoder can wait on the shared fence object.
        for d in [Domain::BlitFence, Domain::ComputeFence, Domain::RenderFence] {
            assert_eq!(
                execute_fence(&mut state, 1, d, 5, FenceAction::Wait),
                FenceStatus::Ok
            );
        }
        // Wait on never-updated ref is pending.
        assert_eq!(
            execute_fence(&mut state, 1, Domain::ComputeFence, 9, FenceAction::Wait),
            FenceStatus::Pending
        );
    }

    #[test]
    fn encoder_fence_routes_name_domain_action_and_outcome() {
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let cases = [
            (
                Domain::BlitFence,
                101,
                "fence_blit_update_ok",
                "fence_blit_wait_ok",
                "fence_blit_wait_pending",
            ),
            (
                Domain::ComputeFence,
                102,
                "fence_compute_update_ok",
                "fence_compute_wait_ok",
                "fence_compute_wait_pending",
            ),
            (
                Domain::RenderFence,
                103,
                "fence_render_update_ok",
                "fence_render_wait_ok",
                "fence_render_wait_pending",
            ),
        ];

        for (domain, reference, update_route, wait_route, pending_route) in cases {
            let update_before = crate::runtime::drain::store_route_count(update_route);
            let wait_before = crate::runtime::drain::store_route_count(wait_route);
            let pending_before = crate::runtime::drain::store_route_count(pending_route);

            assert_eq!(
                execute_fence(&mut state, 1, domain, reference, FenceAction::Update),
                FenceStatus::Ok
            );
            assert_eq!(
                execute_fence(&mut state, 1, domain, reference, FenceAction::Wait),
                FenceStatus::Ok
            );
            assert_eq!(
                execute_fence(&mut state, 1, domain, reference + 1000, FenceAction::Wait,),
                FenceStatus::Pending
            );

            assert_eq!(
                crate::runtime::drain::store_route_count(update_route),
                update_before + 1
            );
            assert_eq!(
                crate::runtime::drain::store_route_count(wait_route),
                wait_before + 1
            );
            assert_eq!(
                crate::runtime::drain::store_route_count(pending_route),
                pending_before + 1
            );
        }
    }

    #[test]
    fn zero_ref_missing() {
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        assert_eq!(
            execute_fence(&mut state, 1, Domain::RenderFence, 0, FenceAction::Update,),
            FenceStatus::Missing
        );
        let cmd = event_cmd(OP_SIGNAL_EVENT, 0, 1, None);
        assert_eq!(execute_event(&mut state, 1, &cmd), FenceStatus::Missing);
    }

    #[test]
    fn event_signal_then_wait() {
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let sig = event_cmd(OP_SIGNAL_EVENT, 7, 100, None);
        assert_eq!(execute_event(&mut state, 1, &sig), FenceStatus::Ok);
        assert_eq!(state.event_generation(1, 7), Some(100));

        let wait_ok = event_cmd(OP_WAIT_EVENT, 7, 100, None);
        assert_eq!(execute_event(&mut state, 1, &wait_ok), FenceStatus::Ok);

        // Wait for higher value is soft-pending.
        let wait_hi = event_cmd(OP_WAIT_EVENT, 7, 101, None);
        assert_eq!(execute_event(&mut state, 1, &wait_hi), FenceStatus::Pending);

        // Advance signal, then wait satisfied.
        let sig2 = event_cmd(OP_SIGNAL_EVENT, 7, 101, None);
        assert_eq!(execute_event(&mut state, 1, &sig2), FenceStatus::Ok);
        assert_eq!(state.event_generation(1, 7), Some(101));
        assert_eq!(execute_event(&mut state, 1, &wait_hi), FenceStatus::Ok);
    }

    #[test]
    fn event_stale_signal_noop_and_independent_of_fence() {
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        assert_eq!(
            execute_event(&mut state, 1, &event_cmd(OP_SIGNAL_EVENT, 3, 50, None)),
            FenceStatus::Ok
        );
        // Stale / equal: no regression.
        assert_eq!(
            execute_event(&mut state, 1, &event_cmd(OP_SIGNAL_EVENT, 3, 40, None)),
            FenceStatus::Ok
        );
        assert_eq!(
            execute_event(&mut state, 1, &event_cmd(OP_SIGNAL_EVENT, 3, 50, None)),
            FenceStatus::Ok
        );
        assert_eq!(state.event_generation(1, 3), Some(50));

        // An equal ref in the fence namespace is independent of the event.
        assert_eq!(
            execute_fence(&mut state, 1, Domain::BlitFence, 3, FenceAction::Update),
            FenceStatus::Ok
        );
        assert_eq!(state.fence_generation(1, 3), Some(1));
        assert_eq!(state.event_generation(1, 3), Some(50));
    }

    #[test]
    fn task_teardown_retires_event_and_fence_namespaces() {
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        state.define_task(1, 0x1000, 2);
        assert_eq!(
            execute_fence(&mut state, 1, Domain::RenderFence, 3, FenceAction::Update),
            FenceStatus::Ok
        );
        assert_eq!(
            execute_event(&mut state, 1, &event_cmd(OP_SIGNAL_EVENT, 3, 50, None)),
            FenceStatus::Ok
        );
        assert!(state.delete_task(1).is_some());
        assert_eq!(state.fence_generation(1, 3), None);
        assert_eq!(state.event_generation(1, 3), None);
    }

    #[test]
    fn event_wait_timeout_unsupported() {
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let wait = event_cmd(OP_WAIT_EVENT_TIMEOUT, 1, 1, Some(42));
        // Asserting the reason, not just the coarse status: this path shares
        // `Unsupported` with six other checks.
        assert_eq!(
            execute_event(&mut state, 1, &wait),
            FenceStatus::Unsupported("event_wait_timeout_unsupported")
        );
        // Satisfied path even with timeout flag is still WaitSatisfied if value present.
        assert_eq!(
            execute_event(&mut state, 1, &event_cmd(OP_SIGNAL_EVENT, 1, 5, None)),
            FenceStatus::Ok
        );
        let wait_ok = event_cmd(OP_WAIT_EVENT_TIMEOUT, 1, 5, Some(42));
        assert_eq!(execute_event(&mut state, 1, &wait_ok), FenceStatus::Ok);
    }

    #[test]
    fn event_wait_missing_signal_pending() {
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let wait = event_cmd(OP_WAIT_EVENT, 99, 1, None);
        assert_eq!(execute_event(&mut state, 1, &wait), FenceStatus::Pending);
    }

    /// Every refusal on these two paths must name a *different* check, or the
    /// coarse status is back and the log cannot say which one fired. Seven
    /// distinct causes, seven distinct slugs.
    #[test]
    fn no_two_fence_checks_answer_with_the_same_reason() {
        use crate::observe::Refusal;
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);

        let mut seen: Vec<&'static str> = Vec::new();
        let mut record = |st: FenceStatus| {
            seen.push(st.refusal().unwrap_or_else(|| {
                panic!("expected a refusal, got {st:?}");
            }));
        };

        // Fence path: unknown domain, an event ref on the encoder-fence path,
        // wait-with-timeout, and an invalid plan.
        record(execute_fence(
            &mut state,
            1,
            Domain::Unknown,
            4,
            FenceAction::Update,
        ));
        record(execute_fence(
            &mut state,
            1,
            Domain::Event,
            4,
            FenceAction::Update,
        ));
        // Event path: unknown kind, wait-with-timeout.
        record(execute_event(
            &mut state,
            1,
            &event_cmd(OP_WAIT_EVENT_TIMEOUT, 4, 9, Some(42)),
        ));

        let mut unique = seen.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(
            unique.len(),
            seen.len(),
            "two fence checks share a reason: {seen:?}"
        );
        assert!(
            seen.contains(&"fence_domain_unknown")
                && seen.contains(&"fence_event_in_fence_path")
                && seen.contains(&"event_wait_timeout_unsupported"),
            "the reasons no longer name their checks: {seen:?}"
        );
    }

    /// `Ok`, `Pending` and a zero ref are control flow, not refusals. Logging
    /// them would flood the always-on sink on every poll — I2's carve-out, made
    /// a compile-time `match` rather than a comment.
    #[test]
    fn success_pending_and_unbound_refs_are_never_logged() {
        use crate::observe::Refusal;
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);

        assert_eq!(
            execute_fence(&mut state, 1, Domain::BlitFence, 5, FenceAction::Update).refusal(),
            None,
            "a successful update is not a refusal"
        );
        assert_eq!(
            execute_fence(&mut state, 1, Domain::ComputeFence, 9, FenceAction::Wait).refusal(),
            None,
            "a soft-pending wait is re-polled every drain; logging it floods"
        );
        assert_eq!(
            execute_fence(&mut state, 1, Domain::RenderFence, 0, FenceAction::Update,).refusal(),
            None,
            "ref==0 is the genuinely-unbound case AGENTS.md carves out"
        );
    }

    /// The encoder remap in `blit_exec` re-derives a blit reason from the fence
    /// status. It used to write a flat `fence_unsupported`, collapsing all seven
    /// causes into one slug; the reason now rides in the value, so the specific
    /// check survives the hop.
    #[test]
    fn a_refusal_carries_its_reason_across_the_blit_remap() {
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let st = execute_fence(&mut state, 1, Domain::Unknown, 7, FenceAction::Update);
        assert_eq!(st, FenceStatus::Unsupported("fence_domain_unknown"));

        let remapped = crate::runtime::blit_exec::blit_status_from_fence(st);
        assert_eq!(
            crate::runtime::blit_exec::blit_fail_reason(),
            "fence_domain_unknown",
            "the remap flattened the fence reason; got {remapped:?}"
        );
    }
}
