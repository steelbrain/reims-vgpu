//! Event/fence synchronization planner.
//!
//! A pure decision function and nothing else: given the generation currently
//! stored for a task-local event or fence reference and a requested signal/wait,
//! decide whether to advance the stored value, treat the operation as already
//! satisfied, or leave it pending. The planner owns no storage and performs no
//! I/O; composition applies its result to the device-owned task namespace.

use crate::{ReferenceNamespace, ResourceLifetime};
use reims_vgpu_protocol::{
    EventObject, FenceObject, ResourceId, ResourceObject, SerializerRef, TaskId,
};
use std::collections::BTreeMap;

pub const FENCE_INITIAL_GENERATION: u64 = 1;

/// Resource classes covered by a Metal memory barrier.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryBarrierScope(u16);

impl MemoryBarrierScope {
    pub const BUFFERS: Self = Self(1);
    pub const TEXTURES: Self = Self(2);
    pub const RENDER_TARGETS: Self = Self(4);
    pub const ALL: Self = Self(Self::BUFFERS.0 | Self::TEXTURES.0 | Self::RENDER_TARGETS.0);

    pub fn from_bits(bits: u16) -> Option<Self> {
        (bits & !Self::ALL.0 == 0).then_some(Self(bits))
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// Canonical identity and lifetime retained by a resource-list barrier.
#[derive(Clone, Debug)]
pub struct BarrierResource {
    pub id: ResourceId<ResourceObject>,
    pub lifetime: ResourceLifetime,
}

impl PartialEq for BarrierResource {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.lifetime.id() == other.lifetime.id()
    }
}

impl Eq for BarrierResource {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum Domain {
    #[default]
    Unknown = 0,
    Event = 1,
    BlitFence = 2,
    ComputeFence = 3,
    RenderFence = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum Decision {
    #[default]
    Invalid = 0,
    SignalUpdate = 1,
    SignalNoop = 2,
    WaitSatisfied = 3,
    WaitPending = 4,
    WaitTimeoutUnsupported = 5,
}

/// Why the [`Decision`] came out the way it did.
///
/// Finer-grained than `Decision` on purpose: "signal ignored because it repeated
/// the current value" and "signal ignored because it went backwards" are the
/// same decision and a different contract, and only this field separates them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum Reason {
    #[default]
    Invalid = 0,
    SignalFirst = 1,
    SignalAdvance = 2,
    SignalEqualIgnored = 3,
    SignalStaleIgnored = 4,
    WaitReached = 5,
    WaitMissingSignal = 6,
    WaitBelowTarget = 7,
    WaitTimeoutUnsupported = 8,
    FenceUpdateFirst = 9,
    FenceUpdateAdvance = 10,
    FenceUpdateAtMax = 11,
    FenceWaitReached = 12,
    FenceWaitMissingUpdate = 13,
    BadFenceDomain = 14,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventKind {
    Signal,
    Wait,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FenceAction {
    Update,
    Wait,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Plan {
    pub decision: Decision,
    pub reason: Reason,
    /// Caller must write `update_value` back to the generation store.
    pub updates_state: bool,
    pub update_value: u64,
}

impl Plan {
    fn signal(reason: Reason, update_value: u64) -> Self {
        Self {
            decision: Decision::SignalUpdate,
            reason,
            updates_state: true,
            update_value,
        }
    }

    fn noop(reason: Reason) -> Self {
        Self {
            decision: Decision::SignalNoop,
            reason,
            updates_state: false,
            update_value: 0,
        }
    }

    fn decided(decision: Decision, reason: Reason) -> Self {
        Self {
            decision,
            reason,
            updates_state: false,
            update_value: 0,
        }
    }
}

fn is_fence_domain(domain: Domain) -> bool {
    matches!(
        domain,
        Domain::BlitFence | Domain::ComputeFence | Domain::RenderFence
    )
}

/// Plan a guest event signal or wait.
///
/// Signals carry an explicit wire value and advance monotonically; a repeated or
/// backwards value is ignored rather than rejected. A wait is satisfied once the
/// stored value reaches the target. Wait-with-timeout is refused outright — there
/// is no deferred host timer, so honouring the timeout is not possible and
/// silently treating it as an untimed wait would change the guest's contract.
pub fn plan_event(kind: EventKind, value: u64, has_timeout: bool, current: Option<u64>) -> Plan {
    match kind {
        EventKind::Signal => match current {
            None => Plan::signal(Reason::SignalFirst, value),
            Some(cur) if value > cur => Plan::signal(Reason::SignalAdvance, value),
            Some(cur) if value == cur => Plan::noop(Reason::SignalEqualIgnored),
            Some(_) => Plan::noop(Reason::SignalStaleIgnored),
        },
        EventKind::Wait => match current {
            Some(cur) if cur >= value => {
                Plan::decided(Decision::WaitSatisfied, Reason::WaitReached)
            }
            _ if has_timeout => Plan::decided(
                Decision::WaitTimeoutUnsupported,
                Reason::WaitTimeoutUnsupported,
            ),
            Some(_) => Plan::decided(Decision::WaitPending, Reason::WaitBelowTarget),
            None => Plan::decided(Decision::WaitPending, Reason::WaitMissingSignal),
        },
    }
}

/// Plan an encoder fence update or wait.
///
/// Unlike events, fences carry no wire value: an update is an implicit
/// increment of a generation counter that starts at
/// [`FENCE_INITIAL_GENERATION`], and a wait is satisfied by the existence of any
/// prior update (the drain is in-order, so a generation that exists has already
/// been reached).
pub fn plan_fence(action: FenceAction, domain: Domain, current: Option<u64>) -> Plan {
    if !is_fence_domain(domain) {
        return Plan::decided(Decision::Invalid, Reason::BadFenceDomain);
    }
    match action {
        FenceAction::Update => match current {
            None => Plan::signal(Reason::FenceUpdateFirst, FENCE_INITIAL_GENERATION),
            Some(u64::MAX) => Plan::noop(Reason::FenceUpdateAtMax),
            Some(cur) => Plan::signal(Reason::FenceUpdateAdvance, cur + 1),
        },
        FenceAction::Wait => match current {
            Some(_) => Plan::decided(Decision::WaitSatisfied, Reason::FenceWaitReached),
            None => Plan::decided(Decision::WaitPending, Reason::FenceWaitMissingUpdate),
        },
    }
}

#[derive(Debug)]
struct TaskGenerationState<M> {
    id: ResourceId<M>,
    value: u64,
}

/// Mutable monotonic values in one API-specific task/reference namespace.
///
/// Equal fence and event integers remain different typed namespaces. Render,
/// compute, and blit fence operations share the same fence namespace rather
/// than creating operation-site-specific copies.
#[derive(Debug)]
pub struct TaskGenerationStates<M> {
    values: BTreeMap<(u32, u32), TaskGenerationState<M>>,
    namespace: ReferenceNamespace<M>,
}

impl<M> Default for TaskGenerationStates<M> {
    fn default() -> Self {
        Self {
            values: BTreeMap::new(),
            namespace: ReferenceNamespace::default(),
        }
    }
}

impl<M> TaskGenerationStates<M> {
    pub fn generation(&self, task_id: u32, reference: u32) -> Option<u64> {
        let state = self.values.get(&(task_id, reference))?;
        debug_assert_eq!(
            self.namespace
                .resolve(TaskId::new(task_id), SerializerRef::new(reference)),
            Some(state.id)
        );
        Some(state.value)
    }

    pub fn set_generation(&mut self, task_id: u32, reference: u32, value: u64) {
        if let Some(state) = self.values.get_mut(&(task_id, reference)) {
            state.value = value;
            return;
        }
        let id = self
            .namespace
            .publish(TaskId::new(task_id), SerializerRef::new(reference))
            .expect("synchronization identity space remains available");
        self.values
            .insert((task_id, reference), TaskGenerationState { id, value });
    }

    pub fn delete(&mut self, task_id: u32, reference: u32) -> bool {
        let removed = self.values.remove(&(task_id, reference)).is_some();
        if removed {
            assert!(self
                .namespace
                .release(TaskId::new(task_id), SerializerRef::new(reference)));
        }
        removed
    }

    pub fn delete_task(&mut self, task_id: u32) -> usize {
        let before = self.values.len();
        self.values.retain(|&(task, _), _| task != task_id);
        let removed = before - self.values.len();
        assert_eq!(removed, self.namespace.release_task(TaskId::new(task_id)));
        removed
    }

    pub fn identity(&self, task_id: u32, reference: u32) -> Option<ResourceId<M>> {
        self.values.get(&(task_id, reference)).map(|state| state.id)
    }
}

/// Encoder domain and stage scope whose completed work one fence update
/// publishes to a later wait.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FenceSignal {
    Blit,
    Compute,
    Render(crate::RenderBarrierStages),
}

/// Generational fence values plus the exact producer scope of the latest
/// update. Keeping both under one owner prevents a reused fence reference from
/// inheriting the previous lifetime's render stages.
#[derive(Debug, Default)]
pub struct TaskFenceStates {
    generations: TaskGenerationStates<FenceObject>,
    signals: BTreeMap<(u32, u32), FenceSignal>,
}

impl TaskFenceStates {
    pub fn generation(&self, task_id: u32, reference: u32) -> Option<u64> {
        self.generations.generation(task_id, reference)
    }

    pub fn set_update(&mut self, task_id: u32, reference: u32, value: u64, signal: FenceSignal) {
        self.generations.set_generation(task_id, reference, value);
        self.signals.insert((task_id, reference), signal);
    }

    pub fn signal(&self, task_id: u32, reference: u32) -> Option<FenceSignal> {
        self.signals.get(&(task_id, reference)).copied()
    }

    pub fn delete(&mut self, task_id: u32, reference: u32) -> bool {
        self.signals.remove(&(task_id, reference));
        self.generations.delete(task_id, reference)
    }

    pub fn delete_task(&mut self, task_id: u32) -> usize {
        self.signals.retain(|&(task, _), _| task != task_id);
        self.generations.delete_task(task_id)
    }

    pub fn identity(&self, task_id: u32, reference: u32) -> Option<ResourceId<FenceObject>> {
        self.generations.identity(task_id, reference)
    }
}

pub type TaskEventStates = TaskGenerationStates<EventObject>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_first_and_advance() {
        let plan = plan_event(EventKind::Signal, 5, false, None);
        assert!(plan.updates_state);
        assert_eq!(plan.update_value, 5);
        assert_eq!(plan.reason, Reason::SignalFirst);

        let plan2 = plan_event(EventKind::Signal, 7, false, Some(5));
        assert_eq!(plan2.reason, Reason::SignalAdvance);
        assert_eq!(plan2.update_value, 7);

        let plan3 = plan_event(EventKind::Signal, 5, false, Some(5));
        assert!(!plan3.updates_state);
        assert_eq!(plan3.reason, Reason::SignalEqualIgnored);

        let plan4 = plan_event(EventKind::Signal, 4, false, Some(5));
        assert!(!plan4.updates_state);
        assert_eq!(plan4.reason, Reason::SignalStaleIgnored);
    }

    #[test]
    fn wait_pending_and_satisfied() {
        let p = plan_event(EventKind::Wait, 3, false, None);
        assert_eq!(p.decision, Decision::WaitPending);
        assert_eq!(p.reason, Reason::WaitMissingSignal);

        let p2 = plan_event(EventKind::Wait, 3, false, Some(3));
        assert_eq!(p2.decision, Decision::WaitSatisfied);

        let p3 = plan_event(EventKind::Wait, 3, false, Some(2));
        assert_eq!(p3.reason, Reason::WaitBelowTarget);

        // A timeout is refused even when the wait would otherwise be pending,
        // but never when it is already satisfied.
        let p4 = plan_event(EventKind::Wait, 3, true, None);
        assert_eq!(p4.decision, Decision::WaitTimeoutUnsupported);
        let p5 = plan_event(EventKind::Wait, 3, true, Some(3));
        assert_eq!(p5.decision, Decision::WaitSatisfied);
    }

    #[test]
    fn fence_generation() {
        let p = plan_fence(FenceAction::Update, Domain::BlitFence, None);
        assert_eq!(p.update_value, FENCE_INITIAL_GENERATION);
        let p2 = plan_fence(FenceAction::Update, Domain::BlitFence, Some(1));
        assert_eq!(p2.update_value, 2);
        let p3 = plan_fence(FenceAction::Update, Domain::BlitFence, Some(u64::MAX));
        assert!(!p3.updates_state);
        assert_eq!(p3.reason, Reason::FenceUpdateAtMax);

        let w = plan_fence(FenceAction::Wait, Domain::ComputeFence, Some(4));
        assert_eq!(w.decision, Decision::WaitSatisfied);
        let w2 = plan_fence(FenceAction::Wait, Domain::RenderFence, None);
        assert_eq!(w2.decision, Decision::WaitPending);
    }

    #[test]
    fn fence_rejects_non_fence_domains() {
        for d in [Domain::Event, Domain::Unknown] {
            let bad = plan_fence(FenceAction::Update, d, None);
            assert_eq!(bad.decision, Decision::Invalid);
            assert_eq!(bad.reason, Reason::BadFenceDomain);
            assert!(!bad.updates_state);
        }
    }

    #[test]
    fn fence_and_event_namespaces_do_not_alias_and_reuse_advances_identity() {
        let mut fences = TaskFenceStates::default();
        let mut events = TaskEventStates::default();
        fences.set_update(
            3,
            9,
            1,
            FenceSignal::Render(crate::RenderBarrierStages::FRAGMENT),
        );
        events.set_generation(3, 9, 40);

        let first_fence = fences.identity(3, 9).expect("fence identity");
        assert_eq!(fences.generation(3, 9), Some(1));
        assert_eq!(
            fences.signal(3, 9),
            Some(FenceSignal::Render(crate::RenderBarrierStages::FRAGMENT))
        );
        assert_eq!(events.generation(3, 9), Some(40));
        assert!(fences.delete(3, 9));
        fences.set_update(3, 9, 2, FenceSignal::Compute);
        assert_ne!(fences.identity(3, 9), Some(first_fence));
        assert_eq!(fences.signal(3, 9), Some(FenceSignal::Compute));
        assert_eq!(events.generation(3, 9), Some(40));
    }
}
