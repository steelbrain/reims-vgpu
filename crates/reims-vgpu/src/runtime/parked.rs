//! The bytes an admitted packet is executed from, held while the model decides
//! when it runs.
//!
//! # Why there is a store at all
//!
//! Today a drain loop that cannot run a packet leaves the ring head where it
//! is, so the bytes stay in the guest's ring and the whole timeline stops
//! behind them. The semantic model replaces that: `SessionModel::admit` gives
//! an unready packet an ordering position and lets later packets proceed, and
//! `take_ready` hands the position back when it may run. Between those two
//! moments the ring has been consumed and the guest may have overwritten it, so
//! whatever the execution reads has to have been taken out of the ring first
//! and kept.
//!
//! # Why it belongs to the device and not to the model
//!
//! `reims_vgpu_core` holds ordering, admission, hazards and publication; it can
//! name no guest-RAM pointer and no decoded device payload, and its dependency
//! list is that claim. An `IngressOrdinal` is the model's, the bytes are this
//! device's, and this map is the join — which is exactly the shape that keeps
//! the model free of them.
//!
//! # What the store owns, and what it deliberately does not
//!
//! It owns the *identity* of a parked position: at most one entry per ordinal,
//! taken out exactly once. It does **not** decide whether an ordinal may run —
//! that is `take_ready`'s — and it does not know whether an ordinal was
//! admitted, because only the model can answer that.
//!
//! It also owns the accounting the switch introduces. Leaving the head unmoved
//! used to be this device's backpressure; parking removes it, so a position the
//! model never releases becomes retained host memory instead of a stalled ring.
//! That is not a reason to refuse work — the contract provides no lawful loss
//! here — but it is a reason to make the retention *countable*, which
//! [`ParkedWork::retained_bytes`] and [`ParkedStore::retained_bytes`] are.

use std::collections::{BTreeMap, BTreeSet};

use reims_vgpu_core::identity::{DeviceEpoch, IngressOrdinal, ResourceId};

use crate::runtime::drain::Packet;
use crate::runtime::exec::ExecSubmission;
use reims_vgpu_core::exec::ExecWork;

/// One admitted packet's retained inputs.
///
/// The decoded packet is always here; it owns its payload `Vec<u8>`, so
/// retaining it is retaining the bytes. The submission is here only for the
/// exec class, because it is the only class whose inputs are *not* all in the
/// packet: `read_submission` loads the command buffers out of task GVA, and a
/// device that re-read them at execution would walk whatever the guest put
/// there after the packet was admitted.
#[derive(Debug)]
pub struct ParkedWork {
    /// The ordering domain the packet arrived in, which is the channel its
    /// execution and its completion word belong to. Carried because the ordinal
    /// alone does not spell it and the model hands back ordinals.
    domain: u32,
    /// The host device incarnation this packet was admitted into.
    ///
    /// Retained rather than looked up at completion, because submission is not
    /// completion: a device loss ends every position admitted into the lost
    /// epoch, and a completion that read the *current* epoch would report work
    /// from a dead incarnation as work from the live one. See
    /// `crate::model::DeviceState::complete_transaction`.
    epoch: DeviceEpoch,
    packet: Packet,
    submission: Option<ExecSubmission>,
    /// What the model's ingress walk resolved this packet's records into.
    ///
    /// **Moved out of the transaction `SessionModel::admit` handed back, not
    /// copied beside it.** The model keeps a transaction's accesses in its
    /// dependency graph and returns everything else to this device, so before
    /// this field the resolved records were built, read for their pipeline
    /// leases, and dropped at the admission site — the walk ran in production
    /// and its answer was thrown away, leaving the executor to decode the same
    /// bytes a second time.
    ///
    /// Holding it here is what makes the pipeline leases a *derivation* rather
    /// than a second list: [`Self::leases`] reads `pipeline_leases` out of this
    /// work, so what the pump advances cannot disagree with what the
    /// transaction was admitted waiting on. It is `None` for every class that
    /// resolves to something other than exec work, which is every class that
    /// leases no pipeline.
    resolved: Option<ExecWork>,
}

impl ParkedWork {
    /// Retain a packet that names no host-side inputs beyond its own payload.
    #[must_use]
    pub const fn new(domain: u32, epoch: DeviceEpoch, packet: Packet) -> Self {
        Self {
            domain,
            epoch,
            packet,
            submission: None,
            resolved: None,
        }
    }

    /// Retain an exec packet together with the command buffers already read out
    /// of guest memory for it.
    #[must_use]
    pub const fn with_submission(
        domain: u32,
        epoch: DeviceEpoch,
        packet: Packet,
        submission: ExecSubmission,
        resolved: ExecWork,
    ) -> Self {
        Self {
            domain,
            epoch,
            packet,
            submission: Some(submission),
            resolved: Some(resolved),
        }
    }

    #[must_use]
    pub const fn domain(&self) -> u32 {
        self.domain
    }

    /// The incarnation this packet was admitted into, which is the one its
    /// completion belongs to.
    #[must_use]
    pub const fn epoch(&self) -> DeviceEpoch {
        self.epoch
    }

    #[must_use]
    pub const fn packet(&self) -> &Packet {
        &self.packet
    }

    /// The command buffers this packet was admitted against, for the exec class.
    #[must_use]
    pub const fn submission(&self) -> Option<&ExecSubmission> {
        self.submission.as_ref()
    }

    /// The pipelines this packet's records bind.
    ///
    /// Read off the resolved work rather than stored beside it; see
    /// [`Self::resolved`].
    #[must_use]
    pub fn leases(&self) -> &[ResourceId] {
        self.resolved
            .as_ref()
            .map_or(&[][..], |work| &work.pipeline_leases)
    }

    /// The records this packet resolved to, for an executor that would
    /// otherwise decode its bytes again.
    #[must_use]
    pub const fn resolved(&self) -> Option<&ExecWork> {
        self.resolved.as_ref()
    }

    /// Everything this position retained beyond its own packet, for the
    /// executor.
    ///
    /// One answer rather than two accessors a caller could take separately:
    /// the two halves were parked together or not at all, and a caller that
    /// could ask for one alone could run bytes against records that are not
    /// theirs.
    #[must_use]
    pub fn retained(&self) -> Option<crate::runtime::exec::RetainedInputs<'_>> {
        Some(crate::runtime::exec::RetainedInputs {
            submission: self.submission.as_ref()?,
            resolved: self.resolved.as_ref()?,
        })
    }

    /// Host bytes this position is holding.
    ///
    /// The payload, the command buffers and the resolved records, which are the
    /// three allocations parking keeps alive; the stamp-wait records are
    /// bounded by the header's `u16` count and are counted with the payload
    /// rather than being a fourth term nobody could act on separately.
    #[must_use]
    pub fn retained_bytes(&self) -> usize {
        self.packet.payload.len()
            + self.packet.stamp_waits.len()
                * std::mem::size_of::<crate::runtime::drain::StampWait>()
            + self
                .submission
                .as_ref()
                .map_or(0, ExecSubmission::retained_bytes)
            + self.resolved.as_ref().map_or(0, resolved_bytes)
    }
}

/// The heap a resolved packet's records occupy.
///
/// Counted term by term rather than as one `size_of` of the struct, because
/// every one of them is a separate `Vec` whose length is the guest's: a packet
/// with ten thousand records retains ten thousand records' worth of host
/// memory, and a measure that reported the struct's inline size would report
/// the same number for a packet with one.
fn resolved_bytes(work: &ExecWork) -> usize {
    use reims_vgpu_core::exec::StreamRecord;
    use std::mem::size_of;

    let arenas = &work.arenas;
    work.streams
        .iter()
        .map(|stream| stream.records.len() * size_of::<StreamRecord>())
        .sum::<usize>()
        + work.accesses.len() * size_of::<reims_vgpu_core::access::AccessIntent>()
        + work.pipeline_leases.len() * size_of::<ResourceId>()
        + work.prerequisites.len() * size_of::<reims_vgpu_core::exec::Prerequisite>()
        + arenas.buffer_bindings.len() * size_of::<reims_vgpu_core::bind::BufferBinding>()
        + arenas.object_bindings.len() * size_of::<reims_vgpu_core::bind::ObjectBinding>()
        + arenas.resources.len() * size_of::<ResourceId>()
        + arenas.pass_descriptors.len() * size_of::<reims_vgpu_core::pass::PassDescriptor>()
        + arenas.viewports.len() * size_of::<reims_vgpu_core::render::Viewport>()
        + arenas.scissors.len() * size_of::<reims_vgpu_core::render::ScissorRect>()
}

/// Why a parked position was let go without running.
///
/// Named rather than collapsed into one `remove`, because the two reach this
/// store from different model calls and a store that could not tell them apart
/// would report a device loss as ordinary progress.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Release {
    /// `take_ready` said it may run; the bytes go to the executor.
    Ready,
    /// The model withdrew it — a device loss, a closed generation's pipeline
    /// lease, or a refusal the caller is naming on its failure channel. The
    /// bytes are dropped and nothing runs.
    Withdrawn,
    /// A pipeline it waits on will never build. The bytes are dropped and
    /// nothing runs, but unlike [`Self::Withdrawn`] the position is completed,
    /// so its word publishes in its channel's order.
    Stranded,
}

impl Release {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Ready => "parked_taken_ready",
            Self::Withdrawn => "parked_withdrawn",
            Self::Stranded => "parked_stranded",
        }
    }
}

/// Every admitted position whose work has not run yet.
///
/// Keyed by `IngressOrdinal` and not by channel head: the ordinal is what the
/// model hands back, and the head has moved on by the time it does. That is
/// also why a held-packet cache keyed on a channel's consumer pointer is not
/// this and could not become it.
#[derive(Debug, Default)]
pub struct ParkedStore {
    work: BTreeMap<IngressOrdinal, ParkedWork>,
    /// Positions the model has released to run and this device has not run
    /// yet.
    ///
    /// Separate from `work` because "may run" and "has bytes" are two facts
    /// with two owners: the model decides the first through `take_ready`, and
    /// this store holds the second. A position can be parked and not runnable
    /// (waiting on a hazard, a stamp or a pipeline) and it can be runnable and
    /// not yet run, which is the state this set exists to make expressible —
    /// see [`Self::ready_in_order`].
    ready: BTreeSet<IngressOrdinal>,
    retained_bytes: usize,
    /// The most bytes ever parked at once, so a boot can report the retention
    /// the switch introduced instead of only its instantaneous value.
    peak_bytes: usize,
    peak_len: usize,
}

impl ParkedStore {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Hold a position's inputs until the model releases it.
    ///
    /// # Panics
    ///
    /// If the ordinal is already parked. An `IngressOrdinal` is issued once per
    /// admission and never reused, so a collision is this device parking one
    /// arrival twice — which would drop the first packet's bytes and run the
    /// second's under the first's ordering position. There is no answer that
    /// keeps both, so it is not an error a caller could handle.
    pub fn park(&mut self, ingress: IngressOrdinal, work: ParkedWork) {
        let bytes = work.retained_bytes();
        assert!(
            self.work.insert(ingress, work).is_none(),
            "ingress ordinal parked twice"
        );
        self.retained_bytes += bytes;
        self.peak_bytes = self.peak_bytes.max(self.retained_bytes);
        self.peak_len = self.peak_len.max(self.work.len());
    }

    /// Take a position's inputs out, once.
    ///
    /// Consuming, and that is the whole guard against a transaction running
    /// twice on this side: the model's `take_ready` is the one door work leaves
    /// by, and this is the one door its bytes leave by. A borrowing lookup
    /// beside it would be a second door.
    ///
    /// `None` for an ordinal this store never held, which is not an error here:
    /// a class that is admitted and executed in the same breath never parks,
    /// and it still completes through the same call.
    #[must_use = "a parked position taken out and not run is a packet that never runs"]
    pub fn release(&mut self, ingress: IngressOrdinal, why: Release) -> Option<ParkedWork> {
        let work = self.work.remove(&ingress)?;
        self.ready.remove(&ingress);
        self.retained_bytes = self.retained_bytes.saturating_sub(work.retained_bytes());
        crate::runtime::drain::note_store_route(why.slug());
        match why {
            Release::Ready => Some(work),
            // Dropped here rather than handed back, so a withdrawal cannot be
            // spelled as a run by a caller that ignores the reason it passed.
            Release::Withdrawn | Release::Stranded => {
                drop(work);
                None
            }
        }
    }

    /// Record that the model has released this position to run.
    ///
    /// `false` when this store is not holding the ordinal, which the caller
    /// counts rather than trusting: the two are kept in step by the drain
    /// parking every packet it admits, and a `take_ready` naming something
    /// unparked means that has stopped being true. It is not an assertion,
    /// because this is reached from the QEMU shim and no panic may cross that
    /// boundary.
    #[must_use = "a ready position this store does not hold is a packet nothing will run"]
    pub fn mark_ready(&mut self, ingress: IngressOrdinal) -> bool {
        if !self.work.contains_key(&ingress) {
            crate::runtime::drain::note_store_route("parked_ready_unheld");
            return false;
        }
        self.ready.insert(ingress);
        true
    }

    /// The positions that may run, in ingress order.
    ///
    /// A list rather than a `next()`, because **the caller may decline any one
    /// of them and must still reach the rest.** This device has two execution
    /// gates the model does not model — host paint two presents behind, and a
    /// translation whose AIR is not built — and both are properties of the
    /// device's own readiness rather than of the packet's ordering. A door that
    /// handed out one position at a time would make declining it block every
    /// later one, which is the head-of-line stall the switch exists to remove.
    ///
    /// A declined position stays here and is offered again on the next drain;
    /// nothing has to put it back, because only running it takes it out.
    #[must_use]
    pub fn ready_in_order(&self) -> Vec<IngressOrdinal> {
        self.ready.iter().copied().collect()
    }

    /// How many positions may run but have not.
    #[must_use]
    pub fn ready_len(&self) -> usize {
        self.ready.len()
    }

    /// One parked position's planning inputs, without taking them out.
    ///
    /// **Not a second way out.** The bytes leave by `release` and only by
    /// `release`; this is a shared borrow, so nothing reached through it can
    /// run the packet. It exists because the plan step — `preflight` — takes a
    /// shared `&DeviceState` and answers whether a position's translations are
    /// done, which is a question that has to be asked *before* deciding to take
    /// the bytes out.
    #[must_use]
    pub fn planning(&self, ingress: IngressOrdinal) -> Option<(&ExecSubmission, &ExecWork)> {
        let work = self.work.get(&ingress)?;
        Some((work.submission.as_ref()?, work.resolved.as_ref()?))
    }

    /// The completion slots a parked position is waiting on.
    ///
    /// The model holds the waits and does not hold the guest's page; this
    /// device holds the page and does not decide the waits. So the slots come
    /// from the packet the position retained, which is where the guest wrote
    /// them, and what is done with them — reading the page and telling the
    /// plane what it holds — stays the caller's.
    #[must_use]
    pub fn awaited_slots(&self, ingress: IngressOrdinal) -> Vec<u32> {
        self.work
            .get(&ingress)
            .map(|w| w.packet.stamp_waits.iter().map(|wait| wait.index).collect())
            .unwrap_or_default()
    }

    /// The ordering domain of a parked position, for a caller that has the
    /// ordinal and needs the channel.
    #[must_use]
    pub fn domain_of(&self, ingress: IngressOrdinal) -> Option<u32> {
        self.work.get(&ingress).map(ParkedWork::domain)
    }

    /// The incarnation a parked position was admitted into, for a caller that
    /// completes it without running it.
    #[must_use]
    pub fn epoch_of(&self, ingress: IngressOrdinal) -> Option<DeviceEpoch> {
        self.work.get(&ingress).map(ParkedWork::epoch)
    }

    /// The opcode of a parked position, for the gates that route on it.
    #[must_use]
    pub fn opcode(&self, ingress: IngressOrdinal) -> Option<u16> {
        self.work.get(&ingress).map(|w| w.packet.opcode)
    }

    /// Every parked position the model has not released, in ingress order.
    ///
    /// What the pump walks. A position is here because something it waits on
    /// has not happened; for an exec packet that something can be a translation
    /// this device itself has to finish, and nothing but a later pass will.
    #[must_use]
    pub fn waiting_in_order(&self) -> Vec<IngressOrdinal> {
        self.work
            .keys()
            .filter(|o| !self.ready.contains(o))
            .copied()
            .collect()
    }

    /// Whether this ordinal is holding bytes. For a caller deciding whether a
    /// completion names parked work, never for one about to run it.
    #[must_use]
    pub fn is_parked(&self, ingress: IngressOrdinal) -> bool {
        self.work.contains_key(&ingress)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.work.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.work.is_empty()
    }

    /// Host bytes every parked position is holding together.
    #[must_use]
    pub const fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    /// The high-water marks, as `(positions, bytes)`.
    #[must_use]
    pub const fn peak(&self) -> (usize, usize) {
        (self.peak_len, self.peak_bytes)
    }

    /// Drop everything, for the events that end every position at once — a
    /// device loss, a session reset that strands work, a channel teardown that
    /// takes its domain with it. Returns what was dropped, so the caller can
    /// say so on its failure channel; the ordinals come back because the guest
    /// is owed a typed reason per position and this is the only place they are
    /// still enumerable.
    #[must_use = "the guest is owed a typed reason for every position this dropped"]
    pub fn drain_all(&mut self) -> Vec<IngressOrdinal> {
        self.retained_bytes = 0;
        self.ready.clear();
        std::mem::take(&mut self.work).into_keys().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packet(payload: usize) -> Packet {
        Packet {
            opcode: 0x37,
            stamp_waits: Vec::new(),
            total_size: 0,
            completion_stamp: 0,
            payload: vec![0u8; payload],
            next_head: 0,
        }
    }

    const EPOCH: DeviceEpoch = DeviceEpoch::FIRST;

    const fn ordinal(n: u64) -> IngressOrdinal {
        IngressOrdinal(n)
    }

    /// The bytes handed back are the bytes parked, not a re-read of anything.
    #[test]
    fn a_parked_packet_comes_back_whole() {
        let mut store = ParkedStore::new();
        let mut p = packet(8);
        p.payload[3] = 0xab;
        store.park(ordinal(1), ParkedWork::new(5, EPOCH, p));

        let taken = store
            .release(ordinal(1), Release::Ready)
            .expect("parked a moment ago");
        assert_eq!(taken.domain(), 5);
        assert_eq!(taken.packet().payload[3], 0xab);
    }

    /// One position's bytes leave by one door, once.
    #[test]
    fn a_position_is_taken_out_once() {
        let mut store = ParkedStore::new();
        store.park(ordinal(2), ParkedWork::new(1, EPOCH, packet(4)));

        assert!(store.release(ordinal(2), Release::Ready).is_some());
        assert!(store.release(ordinal(2), Release::Ready).is_none());
        assert!(store.is_empty());
    }

    /// A withdrawal is not a run: the store empties and hands nothing back, so
    /// a caller that ignored its own reason cannot execute the packet anyway.
    #[test]
    fn a_withdrawal_hands_nothing_back() {
        let mut store = ParkedStore::new();
        store.park(ordinal(3), ParkedWork::new(1, EPOCH, packet(4096)));

        assert!(store.release(ordinal(3), Release::Withdrawn).is_none());
        assert!(!store.is_parked(ordinal(3)));
        assert_eq!(store.retained_bytes(), 0);
    }

    /// The retention the switch introduces is countable while it is held and
    /// after it is let go.
    #[test]
    fn retention_is_counted_up_and_down() {
        let mut store = ParkedStore::new();
        store.park(ordinal(1), ParkedWork::new(1, EPOCH, packet(1000)));
        store.park(ordinal(2), ParkedWork::new(1, EPOCH, packet(24)));
        assert_eq!(store.retained_bytes(), 1024);
        assert_eq!(store.peak(), (2, 1024));

        let _ = store.release(ordinal(1), Release::Ready);
        assert_eq!(store.retained_bytes(), 24);
        // The peak is a high-water mark and does not follow the release down.
        assert_eq!(store.peak(), (2, 1024));
    }

    /// Positions come out in ingress order, because that is the order the guest
    /// is owed its typed reasons in.
    #[test]
    fn everything_dropped_at_once_comes_back_in_order() {
        let mut store = ParkedStore::new();
        for n in [7, 2, 5] {
            store.park(ordinal(n), ParkedWork::new(1, EPOCH, packet(16)));
        }

        assert_eq!(store.drain_all(), vec![ordinal(2), ordinal(5), ordinal(7)],);
        assert!(store.is_empty());
        assert_eq!(store.retained_bytes(), 0);
    }

    /// An ordinal is issued once per admission; parking one twice would run the
    /// second packet under the first's ordering position.
    #[test]
    #[should_panic(expected = "ingress ordinal parked twice")]
    fn one_ordinal_cannot_hold_two_packets() {
        let mut store = ParkedStore::new();
        store.park(ordinal(4), ParkedWork::new(1, EPOCH, packet(4)));
        store.park(ordinal(4), ParkedWork::new(1, EPOCH, packet(4)));
    }
    /// A position the device declines to run stays runnable, and the ones
    /// behind it still run.
    ///
    /// **The head-of-line stall the switch exists to remove.** This device has
    /// two execution gates the model does not model — host paint two presents
    /// behind, and an exec packet whose AIR translation is not built — and
    /// today each of them stops the whole channel by leaving the ring head
    /// where it is. Declining one ready position must not do that.
    #[test]
    fn a_declined_position_does_not_stop_the_ones_behind_it() {
        let mut store = ParkedStore::new();
        for n in [1, 2, 3] {
            store.park(ordinal(n), ParkedWork::new(1, EPOCH, packet(8)));
            assert!(store.mark_ready(ordinal(n)));
        }

        // The device runs 2 and 3 and declines 1 — a present held behind paint.
        assert_eq!(
            store.ready_in_order(),
            vec![ordinal(1), ordinal(2), ordinal(3)]
        );
        for n in [2, 3] {
            assert!(store.release(ordinal(n), Release::Ready).is_some());
        }

        // Nothing put it back, because nothing took it out.
        assert_eq!(store.ready_in_order(), vec![ordinal(1)]);
        assert!(store.release(ordinal(1), Release::Ready).is_some());
        assert!(store.is_empty());
        assert_eq!(store.ready_len(), 0);
    }

    /// Running a position takes it out of both facts at once, so a second
    /// drain cannot offer bytes that have already executed.
    #[test]
    fn running_a_position_takes_it_off_the_ready_list_too() {
        let mut store = ParkedStore::new();
        store.park(ordinal(5), ParkedWork::new(1, EPOCH, packet(8)));
        assert!(store.mark_ready(ordinal(5)));

        assert!(store.release(ordinal(5), Release::Ready).is_some());
        assert!(store.ready_in_order().is_empty());
    }

    /// A withdrawal takes it off too: work the model took away must not be
    /// offered as runnable.
    #[test]
    fn a_withdrawal_takes_a_ready_position_off_the_list() {
        let mut store = ParkedStore::new();
        store.park(ordinal(6), ParkedWork::new(1, EPOCH, packet(8)));
        assert!(store.mark_ready(ordinal(6)));

        assert!(store.release(ordinal(6), Release::Withdrawn).is_none());
        assert!(store.ready_in_order().is_empty());
    }

    /// A ready position this store never held is answered and not asserted,
    /// because this is reached from the QEMU shim.
    #[test]
    fn a_ready_position_with_no_bytes_is_answered_not_asserted() {
        let mut store = ParkedStore::new();
        assert!(!store.mark_ready(ordinal(8)));
        assert!(store.ready_in_order().is_empty());
    }

    fn leased(refs: &[u32]) -> ExecWork {
        ExecWork {
            pipeline_leases: refs
                .iter()
                .map(|&r| ResourceId {
                    slot: reims_vgpu_core::identity::ObjectListRef(r),
                    generation: reims_vgpu_core::identity::SlotGeneration(1),
                })
                .collect(),
            ..ExecWork::default()
        }
    }

    /// The leases the pump advances are the resolved work's own field, so
    /// there is no second list to keep in step with it.
    #[test]
    fn the_leases_a_position_advances_are_the_resolved_works() {
        let work = ParkedWork::with_submission(
            2,
            EPOCH,
            packet(4),
            ExecSubmission::stated(7, Vec::new()),
            leased(&[11, 12]),
        );

        let resolved = work.resolved().expect("an exec position resolved");
        assert_eq!(work.leases(), resolved.pipeline_leases.as_slice());
        assert_eq!(work.leases().len(), 2);
    }

    /// A class that resolves to no exec work leases nothing, rather than
    /// leaving a caller to distinguish "no leases" from "not exec".
    #[test]
    fn a_position_with_no_resolved_work_leases_nothing() {
        let work = ParkedWork::new(1, EPOCH, packet(4));
        assert!(work.resolved().is_none());
        assert!(work.leases().is_empty());
    }

    /// Retention counts the records a position holds, so a packet that
    /// resolved to more of them is measured as holding more.
    #[test]
    fn retention_counts_the_resolved_records() {
        let bare = ParkedWork::with_submission(
            2,
            EPOCH,
            packet(4),
            ExecSubmission::stated(7, Vec::new()),
            ExecWork::default(),
        );
        let leases = ParkedWork::with_submission(
            2,
            EPOCH,
            packet(4),
            ExecSubmission::stated(7, Vec::new()),
            leased(&[11, 12, 13]),
        );

        assert!(
            leases.retained_bytes() > bare.retained_bytes(),
            "three leases are three leases' worth of host memory: {} vs {}",
            leases.retained_bytes(),
            bare.retained_bytes()
        );
    }
}
