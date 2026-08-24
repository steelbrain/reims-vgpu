//! Read an empty object-list slot again later, and say whether it fills.
//!
//! # The question this answers, and the one it refuses to ask
//!
//! `ListMiss::SlotEmpty` is the whole of macos-26's lost draws — the task
//! exists and is active, its object list is registered where the guest said,
//! the ref is inside the declared count, the guest read *succeeds*, and the
//! sixteen bytes are zero. Two readings fit that, with opposite fixes:
//!
//! 1. **A race this device should tolerate.** The guest referenced the object
//!    in a packet it submitted before publishing the slot. Then the answer is to
//!    defer the packet the way an unfinished AIR translation is deferred, and
//!    dropping the draw is the defect.
//! 2. **This device looked in the wrong list.** The object lives under another
//!    task and the ref was meant to resolve against it.
//!
//! The obvious discriminator — "does another live task hold a real object at
//! this slot?" — was built, and it answered *yes* to every miss of a boot. That
//! is not a verdict for reading 2. **Every task registers its object list at the
//! same `pfn = 1`** and the refs in play are small and dense, so "somebody else
//! has something at slot 3" is close to a tautology on a busy guest. Banding the
//! claimant count against the live task count showed 82 % of the answer sitting
//! in the uninformative band.
//!
//! This module asks the question that no other task's address can confound:
//! **re-read the same slot, in the same list, later.** If it becomes non-zero
//! the guest published late and reading 1 is right. If it is still zero when the
//! task dies, reading 1 is dead and reading 2 (or a fourth thing) is live.
//!
//! # Why there is no timeout
//!
//! The terminal verdict is the guest's own task teardown, not a wall clock. A
//! horizon would have to come from somewhere, and the deferral machinery this
//! feeds has none to borrow: `ChildPacketDisposition::Deferred` leaves the
//! packet at the FIFO head and it is retried every drain until the translation
//! lands. So a watch here ends when the guest ends the task, and
//! [`Verdict::fill_us`] reports the age at which a fill was actually seen —
//! which is the number that says whether deferring is affordable, rather than a
//! number chosen in advance that decides it.
//!
//! # What it costs
//!
//! One quiet probe read per *distinct* watched `(task, ref)` per drain tranche,
//! and nothing at all when nothing is watched — which is every rail except
//! macos-26, where rails 11 through 15 record zero `list_miss_slot_empty` in a
//! driven boot.

use crate::runtime::decode::resource::{decode_list_object_entry, OBJECT_LIST_ENTRY_LEN};
use crate::runtime::gva_mem;
use crate::runtime::host::HostMemory;
use crate::runtime::Device;

use super::{list_entry_or_miss, ListLookup, ListMiss};

/// One slot being watched, keyed by the `(task, ref)` the guest named.
type WatchKey = (u32, u32);

/// When the watch started, in both clocks it is read against.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Watch {
    /// For [`Verdict::fill_us`]. `crate::observe::elapsed_us`.
    recorded_us: u64,
    /// The index of the sweep that closes the tranche the miss happened in.
    ///
    /// A watch becomes due **after** that sweep, not at it, so the first re-read
    /// is genuinely a later tranche rather than the tail of the one that
    /// produced the miss. Without the distinction a slot published microseconds
    /// after the packet would read as filled at an age of zero, and a slot the
    /// guest never publishes would be indistinguishable from it on the first
    /// sample.
    closing_sweep: u64,
}

/// The watch set, with its capacity in the type that carries it.
///
/// The bound is real and so is the overflow: a ledger that silently stopped
/// admitting would report "every miss was watched" while watching a prefix, and
/// the resulting `filled`/`still_empty` ratio would be a statement about the
/// first N refs of a boot wearing the name of a statement about all of them.
/// [`Ledger::admit`] therefore returns whether the watch was taken and the
/// caller counts the refusals.
///
/// The capacity is not derived from the contract, because the contract does not
/// bound it — the guest declares an object list of 2^20 slots and may name any
/// of them. It is sized against the measured population instead: a driven
/// macos-26 boot produces ~170 misses spread over **eight** distinct refs across
/// four tasks, so a thousand distinct live watches is three orders of magnitude
/// of headroom, and `slot_recheck_dropped` is what says if that ever stops being
/// true.
struct Ledger {
    watches: std::collections::HashMap<WatchKey, Watch>,
    /// How many sweeps have run. A watch admitted now belongs to the tranche the
    /// next one closes, which is what [`Watch::closing_sweep`] records.
    sweep: u64,
}

impl Ledger {
    const CAPACITY: usize = 1024;

    fn new() -> Self {
        Self {
            watches: std::collections::HashMap::new(),
            sweep: 0,
        }
    }

    /// Start watching `key`, or report that the ledger is full.
    ///
    /// A repeat miss on a slot already watched is the *same* watch — the guest
    /// re-issuing a packet against a ref it still has not published — so it
    /// keeps the original `recorded_us` and does not consume a second entry.
    /// Overwriting it would reset the age and make every fill look instant.
    fn admit(&mut self, key: WatchKey, now_us: u64) -> bool {
        if self.watches.contains_key(&key) {
            return true;
        }
        if self.watches.len() >= Self::CAPACITY {
            return false;
        }
        self.watches.insert(
            key,
            Watch {
                recorded_us: now_us,
                closing_sweep: self.sweep.saturating_add(1),
            },
        );
        true
    }

    /// Open a sweep and hand back the watches whose own tranche has already
    /// closed.
    ///
    /// The strict `<` is the whole of "read it again **later**": a watch
    /// admitted during tranche *N* is skipped by the sweep that closes *N* and
    /// read for the first time by the one that closes *N+1*. Without it the
    /// first re-read is the same instant as the miss, so a slot the guest
    /// publishes a microsecond afterwards is indistinguishable from one it never
    /// publishes.
    fn begin_sweep(&mut self) -> Vec<(WatchKey, Watch)> {
        self.sweep = self.sweep.saturating_add(1);
        let sweep_now = self.sweep;
        self.watches
            .iter()
            .filter(|(_, w)| w.closing_sweep < sweep_now)
            .map(|(k, w)| (*k, *w))
            .collect()
    }

    /// The level line, or `None` when nothing is watched. Takes `now` rather
    /// than reading the clock so the age it reports is testable.
    fn outstanding_line(&self, now_us: u64) -> Option<String> {
        let oldest = self
            .watches
            .values()
            .map(|w| now_us.saturating_sub(w.recorded_us))
            .max()?;
        Some(format!(
            "slot_recheck_watching n={} oldest_us={oldest} capacity={}",
            self.watches.len(),
            Self::CAPACITY,
        ))
    }
}

/// Which `(task, ref)` pairs have ever resolved to a real object in this boot.
///
/// # Why a bitmap and not a set
///
/// This is written on the **success** path of every named object-list lookup,
/// which runs per ref per draw — thousands of times a second under a window
/// drag. A `Mutex<HashSet>` there would be measuring the instrument. A fixed
/// bitmap of atomics is one `fetch_or` with no lock and no allocation, and its
/// bound is the array rather than a number written down somewhere.
///
/// # What it is for
///
/// Two readings are left for macos-26's lost draws, and this separates them:
/// the guest names a pipeline ref that **never** existed in this list, or one
/// that existed and stopped. The second is a device defect with an obvious
/// mechanism — macOS 26 re-issues `define_task` for a live tid with a new
/// page-table root, so GVA page 1 afterwards is a different physical page and
/// everything published into the old one reads as zero.
struct ResolvedBits {
    /// `[task][word]`, one bit per ref.
    words: [[std::sync::atomic::AtomicU64; Self::WORDS]; Self::TASKS],
    /// Refs or tasks past the array. Counted rather than dropped, because a
    /// silent miss here would make "never resolved" the answer for everything
    /// out of range and that is the reading the whole instrument turns on.
    out_of_range: std::sync::atomic::AtomicU64,
}

impl ResolvedBits {
    /// Live task ids observed on this rail run to 21 within a boot and the
    /// table is indexed directly, so this is headroom rather than a fit.
    const TASKS: usize = 64;
    /// One 4 KiB page of a 12-byte-per-entry object list is 341 refs, which is
    /// the window every other instrument here reads; six words covers it with
    /// the same rounding.
    const WORDS: usize = 6;
    const REFS: u32 = (Self::WORDS * 64) as u32;

    fn new() -> Self {
        Self {
            words: std::array::from_fn(|_| std::array::from_fn(|_| Default::default())),
            out_of_range: Default::default(),
        }
    }

    fn locate(task_id: u32, ref_: u32) -> Option<(usize, usize, u64)> {
        if task_id as usize >= Self::TASKS || ref_ >= Self::REFS {
            return None;
        }
        Some((task_id as usize, (ref_ / 64) as usize, 1u64 << (ref_ % 64)))
    }

    fn set(&self, task_id: u32, ref_: u32) {
        use std::sync::atomic::Ordering;
        let Some((t, w, bit)) = Self::locate(task_id, ref_) else {
            self.out_of_range.fetch_add(1, Ordering::Relaxed);
            return;
        };
        self.words[t][w].fetch_or(bit, Ordering::Relaxed);
    }

    fn get(&self, task_id: u32, ref_: u32) -> Option<bool> {
        use std::sync::atomic::Ordering;
        let (t, w, bit) = Self::locate(task_id, ref_)?;
        Some(self.words[t][w].load(Ordering::Relaxed) & bit != 0)
    }
}

fn resolved_bits() -> &'static ResolvedBits {
    static BITS: std::sync::OnceLock<ResolvedBits> = std::sync::OnceLock::new();
    BITS.get_or_init(ResolvedBits::new)
}

/// Remember that `ref_` resolved to a real object under `task_id`.
///
/// On the success path of a named lookup. See [`ResolvedBits`] for why this is
/// an atomic bit and not a set.
pub(super) fn note_ref_resolved(task_id: u32, ref_: u32) {
    resolved_bits().set(task_id, ref_);
}

fn ledger() -> &'static std::sync::Mutex<Ledger> {
    use std::sync::{Mutex, OnceLock};
    static LEDGER: OnceLock<Mutex<Ledger>> = OnceLock::new();
    LEDGER.get_or_init(|| Mutex::new(Ledger::new()))
}

/// How one watch ended.
///
/// [`Self::StillWaiting`] is the only arm that is not terminal, and the only one
/// with no route of its own: a slot that is still zero is the ledger's residue,
/// reported as a level by [`outstanding_census`] rather than counted once per
/// tranche it survives.
///
/// The other seven are [`ListMiss`]'s own variants, carried rather than banded.
/// The first version of this module folded four of them into one
/// `slot_recheck_unreadable` and the first driven macos-26 boot put every one of
/// its 20 terminal verdicts there — a route that says a watch ended and cannot
/// say why, which is the shape [`ListMiss`] was introduced to retire.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Verdict {
    /// The slot became a real entry. **Reading 1**: the guest published after
    /// naming the ref, and the draw was dropped for a race rather than for a
    /// missing object.
    Filled,
    /// Still zero, and still worth asking again.
    StillWaiting,
    /// The watch ended without the slot ever being published, and this is the
    /// check that ended it. `NoTask` / `TaskInactive` / `NoObjectList` are the
    /// guest tearing the task down; the other four say the list moved under the
    /// watch, and for a slot that read cleanly once they are a finding in their
    /// own right rather than a shrug.
    Ended(ListMiss),
}

/// Classify one re-read.
///
/// Split from the sweep so the one judgement this module makes is testable
/// without a guest: which single [`ListMiss`] means "keep asking".
fn verdict_of(read: Result<(), ListMiss>) -> Verdict {
    match read {
        Ok(()) => Verdict::Filled,
        Err(ListMiss::SlotEmpty) => Verdict::StillWaiting,
        Err(ended) => Verdict::Ended(ended),
    }
}

/// Record that the guest named `ref_` against `task_id` and the slot was zero.
///
/// Called only from the `Named` arm of the lookup — a probe misses on every task
/// that does not own the ref, which is how it finds the one that does, and
/// watching those would fill the ledger with the search.
pub(super) fn note_slot_empty<M: HostMemory>(state: &Device, host: &M, task_id: u32, ref_: u32) {
    let now = crate::observe::elapsed_us();
    let admitted = ledger()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .admit((task_id, ref_), now);
    if !admitted {
        crate::runtime::drain::note_store_route("slot_recheck_dropped");
        return;
    }
    note_list_population(state, host, task_id, ref_);
}

/// How far the guest has actually populated the list this ref missed in.
///
/// The watch says the slot never fills; this says whether the guest's writes
/// ever reach that far. Two shapes, and they point at different defects:
///
/// - **The ref sits above everything populated.** The list is a prefix and this
///   ref is past its end, so the number the device is treating as an index into
///   this list is not one — a decode question, not a lifetime one.
/// - **The ref sits inside a populated region, as a hole.** The list really does
///   have a gap where the guest named an object, and the object is somewhere
///   else.
///
/// Reads the list's **first page** in one go rather than probing per ref: 341
/// entries at a 4 KiB page and 12 bytes an entry, against 341 page-table walks
/// for the same answer. Emitted once per newly-watched `(task, ref)`, which a
/// driven macos-26 boot reaches a few dozen times — the misses themselves are
/// three times that, and repeats of a slot already watched cost nothing.
fn note_list_population<M: HostMemory>(state: &Device, host: &M, task_id: u32, ref_: u32) {
    let Some(pop) = first_page_population(state, host, task_id) else {
        // The list's own first page did not read, which the per-slot walk would
        // have reported as `Unreadable` rather than `SlotEmpty` — so reaching
        // here means the two disagree and neither reading is usable.
        crate::runtime::drain::note_store_route("slot_recheck_population_unreadable");
        return;
    };
    // The route is what ranks the two shapes against each other across a boot;
    // the line is what names the task and the reach behind one reading.
    crate::runtime::drain::note_store_route(if i64::from(ref_) > pop.highest {
        "slot_empty_ref_above_reach"
    } else {
        "slot_empty_ref_within_reach"
    });
    // The remaining fork, and the only one both other searches leave open: did
    // this exact slot ever hold a real object earlier in the boot?
    let before = match resolved_bits().get(task_id, ref_) {
        Some(true) => {
            crate::runtime::drain::note_store_route("slot_empty_ref_was_resolved");
            "yes"
        }
        Some(false) => {
            crate::runtime::drain::note_store_route("slot_empty_ref_never_resolved");
            "no"
        }
        None => {
            crate::runtime::drain::note_store_route("slot_empty_ref_untracked");
            "untracked"
        }
    };
    crate::observe::off(format!(
        "slot_empty_population task={task_id} ref={ref_} {pop} resolved_before={before} \
         (the first page of this task's own object list, at the moment the ref missed)"
    ));
}

/// What one task's object list holds in the first page of it.
pub(super) struct Population {
    pub populated: usize,
    /// The last occupied index, or `-1` for an empty page — so `ref > highest`
    /// is a total test with no special case for "nothing here at all".
    pub highest: i64,
    slots: usize,
    occupied: Vec<usize>,
}

impl std::fmt::Display for Population {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "populated={} highest_ref={} slots_read={} occupied={:?}{}",
            self.populated,
            self.highest,
            self.slots,
            self.occupied,
            if self.populated > self.occupied.len() {
                "+"
            } else {
                ""
            }
        )
    }
}

/// Count and locate the objects in the first page of `task_id`'s object list.
///
/// One guest read of a page, decoded locally: 341 entries at a 4 KiB page and 12
/// bytes an entry, against 341 page-table walks for the same answer. `None` when
/// the page does not read.
///
/// Shared with the claimant search, which is the point. "Task 1 holds the ref
/// task 19 missed" means one thing if task 1 holds six objects and the opposite
/// if it holds three hundred, and a claim measured against an occupancy counted
/// some other way would not settle it. The two sites count the same way because
/// there is one count.
pub(super) fn first_page_population<M: HostMemory>(
    state: &Device,
    host: &M,
    task_id: u32,
) -> Option<Population> {
    let task = state.tasks.get(task_id)?;
    let page_size = 1usize << state.page_shift;
    let base = (task.object_list_pfn as u64) << state.page_shift;
    let mut page = vec![0u8; page_size];
    gva_mem::try_read_task_gva_by_id(
        host,
        &state.tasks,
        task_id,
        base,
        &mut page,
        state.page_shift,
    )
    .ok()?;
    // Never past what the guest declared, however many entries the page holds.
    let slots = (page_size / OBJECT_LIST_ENTRY_LEN).min(task.object_list_count as usize);
    let mut pop = Population {
        populated: 0,
        highest: -1,
        slots,
        occupied: Vec::new(),
    };
    for i in 0..slots {
        let raw = &page[i * OBJECT_LIST_ENTRY_LEN..(i + 1) * OBJECT_LIST_ENTRY_LEN];
        let Ok(entry) = decode_list_object_entry(raw) else {
            continue;
        };
        if entry.descriptor_length != 0 && entry.descriptor_gva != 0 {
            pop.populated += 1;
            pop.highest = i as i64;
            if pop.occupied.len() < OCCUPIED_SHOWN {
                pop.occupied.push(i);
            }
        }
    }
    Some(pop)
}

/// How many occupied indices the population line prints.
///
/// The indices are the reading, not the count: a list occupied at
/// `{0, 4, 8, 12}` while the guest names 3 and 9 is a stride or base this device
/// has wrong, and one occupied at `{2, 5, 11, 26}` while the guest names 3 and 9
/// is not. No count can tell those apart. Truncation is marked with a trailing
/// `+` on the list rather than left to be inferred from `populated`, because a
/// silently clipped set reads as a complete one and would answer the stride
/// question wrongly with no sign that it had.
///
/// Measured lists are 4 to 18 entries in a first page; this shows every one of
/// those whole and clips only something an order of magnitude denser, where the
/// prefix is still enough to see a stride.
const OCCUPIED_SHOWN: usize = 24;

/// One line per watch that ended, naming the check and the age.
///
/// The counted route ranks the ends against each other; this says which
/// `(task, ref)` and how long it waited, which is what a reader needs to line an
/// end up against the guest's own `set_object_list` / `define_task` traffic.
/// [`ListMiss::Unreadable`]'s payload is spelled out because that arm is
/// fifteen page-table checks wearing one name, and on a re-read the check that
/// refused is the finding rather than a footnote.
///
/// Latched per `(task, ref)` — the sweep can only end a watch once, so this is
/// belt and braces against a slot re-entering the ledger and ending the same way
/// every frame the guest re-issues the packet.
fn note_ended_detail(task_id: u32, ref_: u32, miss: ListMiss, age_us: u64) {
    if !crate::observe::first_sight(
        miss.recheck_route(),
        (u64::from(task_id) << 32) | u64::from(ref_),
    ) {
        return;
    }
    // The walk's refusal names itself through the same `Decline` vocabulary
    // every other guest read reports through, so a reader can match this
    // `check=` against a `gva_read_refused reason=` without a second table.
    use crate::observe::Decline;
    let check = match miss {
        ListMiss::Unreadable(why) => why.slug(),
        _ => miss.recheck_route(),
    };
    crate::observe::off(format!(
        "slot_recheck_ended task={task_id} ref={ref_} verdict={} check={check} age_us={age_us} \
         (the slot read and decoded when the guest named it; this is what the same read \
          found one or more tranches later)",
        miss.recheck_route(),
    ));
}

/// Re-read every watched slot that was recorded before this sweep.
///
/// Runs at the tail of a drain tranche, with the same `state` and host the
/// lookup used. Returns early with the lock untouched when nothing is watched,
/// which is every tranche on every rail that does not produce the miss.
pub fn sweep<M: HostMemory>(state: &Device, host: &M) {
    let due = ledger()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .begin_sweep();
    if due.is_empty() {
        return;
    }

    let now = crate::observe::elapsed_us();
    let mut retired: Vec<WatchKey> = Vec::new();
    for (key, watch) in due {
        let (task_id, ref_) = key;
        let read = list_entry_or_miss(state, host, task_id, ref_, ListLookup::Probe).map(|_| ());
        let age_us = now.saturating_sub(watch.recorded_us);
        match verdict_of(read) {
            Verdict::StillWaiting => continue,
            Verdict::Filled => {
                crate::runtime::drain::note_store_route("slot_recheck_filled");
                // The age at which the fill was *seen*, which is an upper bound
                // on the age at which it happened — the slot is only sampled
                // once per tranche. Quoted against `slot_recheck_filled` from
                // the same census window, so the mean is `fill_us / filled`.
                crate::runtime::drain::note_store_route_us("slot_recheck_fill_us", age_us);
            }
            Verdict::Ended(miss) => {
                crate::runtime::drain::note_store_route(miss.recheck_route());
                note_ended_detail(task_id, ref_, miss, age_us);
                // Beside the verdict and on the same cadence: how long the slot
                // stayed unpublished before the watch ended is what says whether
                // the guest had time to publish and chose not to, or whether the
                // task was gone before a deferral could have helped.
                crate::runtime::drain::note_store_route_us("slot_recheck_ended_us", age_us);
            }
        }
        retired.push(key);
    }

    if retired.is_empty() {
        return;
    }
    let mut guard = ledger().lock().unwrap_or_else(|e| e.into_inner());
    for key in retired {
        guard.watches.remove(&key);
    }
}

/// The watches still open, as a **level**, for the census to emit beside the
/// terminal routes.
///
/// The terminal routes are per-window sums and this is not: it is what the
/// ledger holds at the moment it is read, and the last sample is the answer
/// rather than the total. It exists because the residue has no other spelling —
/// a slot that is still zero is skipped by every sweep it survives, so counting
/// it per tranche would report one long wait as thousands of them, and not
/// counting it at all leaves "45 misses, 20 verdicts" with no account of the
/// other 25.
///
/// `oldest_us` is the age of the longest-running watch, which is the reading
/// that says whether the residue is a queue draining or a set of slots the
/// guest is never going to publish.
pub fn outstanding_census() -> Option<String> {
    ledger()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .outstanding_line(crate::observe::elapsed_us())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exactly one miss means "ask again"; every other one ends the watch and
    /// must arrive at the census carrying which check it was. A second variant
    /// landing on `StillWaiting` would keep a dead task in the ledger forever,
    /// and any variant reaching the census banded would rebuild the collapse
    /// that made this module's first driven boot unreadable.
    #[test]
    fn only_an_empty_slot_keeps_a_watch_alive_and_every_other_end_names_its_check() {
        assert_eq!(verdict_of(Ok(())), Verdict::Filled);
        let mut ended = std::collections::HashSet::new();
        for miss in ListMiss::ALL {
            match verdict_of(Err(miss)) {
                Verdict::StillWaiting => assert_eq!(miss, ListMiss::SlotEmpty, "{miss:?}"),
                Verdict::Ended(seen) => {
                    assert_eq!(seen, miss);
                    assert!(
                        ended.insert(seen.recheck_route()),
                        "{miss:?} shares a route with another ended verdict"
                    );
                }
                Verdict::Filled => panic!("{miss:?} cannot be a fill"),
            }
        }
        assert_eq!(ended.len(), ListMiss::ALL.len() - 1);
    }

    /// The residue has no per-tranche route, so the level is the only account of
    /// it — and an empty ledger must say nothing rather than say zero, or a rail
    /// that never misses would carry a line claiming it was watching.
    #[test]
    fn the_outstanding_level_is_silent_until_something_is_watched() {
        let mut ledger = Ledger::new();
        assert_eq!(ledger.outstanding_line(1_000), None);
        ledger.admit((9, 9), 400);
        ledger.admit((9, 10), 900);
        // Two watches, and the age reported is the *oldest* — the youngest would
        // read as a healthy queue however long the first slot had been stuck.
        let line = ledger.outstanding_line(1_000).expect("watching");
        assert!(line.contains(" n=2 "), "{line}");
        assert!(line.contains("oldest_us=600"), "{line}");
    }

    /// A repeat miss on a watched slot must not restart its clock. The guest
    /// re-issues the packet every frame it still wants the object, so an
    /// overwrite would report every fill at roughly one tranche of age however
    /// long the guest actually took.
    #[test]
    fn a_repeat_miss_keeps_the_first_sighting_and_costs_no_capacity() {
        let mut ledger = Ledger::new();
        assert!(ledger.admit((7, 3), 1_000));
        assert!(ledger.admit((7, 3), 9_000));
        assert_eq!(ledger.watches.len(), 1);
        assert_eq!(ledger.watches[&(7, 3)].recorded_us, 1_000);
    }

    /// The bound refuses rather than truncating, and says so to its caller.
    #[test]
    fn a_full_ledger_refuses_a_new_watch_instead_of_evicting_one() {
        let mut ledger = Ledger::new();
        for i in 0..Ledger::CAPACITY as u32 {
            assert!(ledger.admit((1, i), 0), "admit {i}");
        }
        assert!(!ledger.admit((1, Ledger::CAPACITY as u32), 0));
        assert_eq!(ledger.watches.len(), Ledger::CAPACITY);
        // The already-watched slot still resolves: a full ledger stops taking
        // new work, it does not stop tracking what it has.
        assert!(ledger.admit((1, 0), 0));
    }

    /// A watch may not be re-read by the sweep that is running when it is
    /// recorded, or "still empty" would be asserted of an instant.
    #[test]
    fn a_watch_is_not_due_in_the_sweep_that_closes_the_tranche_it_was_recorded_in() {
        let mut ledger = Ledger::new();
        // Tranche 1: the miss is recorded, then the tranche's own sweep runs.
        ledger.admit((2, 5), 0);
        assert!(ledger.begin_sweep().is_empty());
        // Tranche 2's sweep is the first that may read it, and it hands back the
        // watch as recorded — the age it reports is measured from the miss.
        let recorded = ledger.watches[&(2, 5)];
        assert_eq!(ledger.begin_sweep(), vec![((2, 5), recorded)]);
        // And it stays due until something retires it — a slot the guest never
        // publishes must keep being asked until the task dies.
        assert_eq!(ledger.begin_sweep().len(), 1);
    }

    /// A miss recorded by a *later* tranche must not be answered by the sweep
    /// that is already running for the earlier ones.
    #[test]
    fn a_sweep_does_not_pick_up_a_watch_admitted_after_it_began() {
        let mut ledger = Ledger::new();
        ledger.admit((1, 1), 0);
        assert!(ledger.begin_sweep().is_empty());
        ledger.admit((1, 2), 0);
        // Only the older watch: `(1, 2)` was admitted during this sweep's own
        // tranche and has not yet had a later one.
        assert_eq!(
            ledger
                .begin_sweep()
                .into_iter()
                .map(|(k, _)| k)
                .collect::<Vec<_>>(),
            vec![(1, 1)]
        );
    }
}
