//! The platform's stamps of a command buffer's trip (reims-vgpu-graphics' `trip`): when rt's
//! completed handler entered, and the span of rt's commit call, each on the clock the host
//! installs ([`set_clock`], the device's), handed to the decoder on the thread that runs them;
//! and, from rt's own times of the command buffer (on rt's host clock), how long the GPU sat idle
//! before it and its pickup ([`gpu`]). A diagnostic: before a clock is installed nothing is
//! stamped, and each site costs one load.
//!
//! **The card's record** (`Card`, [`take_cards`]): each command buffer committed to rt, every
//! queue's, its depth as committed (the device's buffers committed and not completed, every
//! queue's), when rt's commit call returned, the decoder's stamps of its way up ([`Chain`]), and
//! its GPU start and end, put on the installed clock as rt's handler enters.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use crate::core::sync::Mutex;

/// The clock the stamps read, in nanoseconds.
static CLOCK: OnceLock<fn() -> u64> = OnceLock::new();

/// The latest GPU end of the command buffers whose handlers ran, on rt's host clock.
static LAST_GPU_END: AtomicU64 = AtomicU64::new(0);

/// The device's command buffers committed to rt whose handlers have not entered, every queue's
/// ([`Card`]).
static IN_FLIGHT: AtomicU64 = AtomicU64::new(0);

/// The card's records since the take before ([`take_cards`]).
static CARDS: Mutex<Vec<CardRecord>> = Mutex::new(Vec::new());

thread_local! {
    /// When rt's completed handler this thread runs entered, and whether it was handed on from
    /// rt's thread.
    static HANDLER: Cell<(u64, bool)> = const { Cell::new((0, false)) };
    /// Its command buffer's idle before it and its pickup ([`gpu`]).
    static GPU: Cell<Option<(u64, u64)>> = const { Cell::new(None) };
    /// Its command buffer's GPU end on the installed clock ([`gpu_end`]).
    static END: Cell<u64> = const { Cell::new(0) };
    /// The decoder's stamps of the buffer this thread commits next ([`set_chain`]).
    static CHAIN: Cell<Option<Chain>> = const { Cell::new(None) };
    /// When this thread's last commit called rt's, and how long that call took.
    static COMMIT: Cell<(u64, u64)> = const { Cell::new((0, 0)) };
}

/// Stamps from now on, on `clock`; the first clock installed stays.
pub fn set_clock(clock: fn() -> u64) {
    let _ = CLOCK.set(clock);
}

/// The installed clock's time, 0 before one is installed.
pub(crate) fn stamp() -> u64 {
    CLOCK.get().map_or(0, |clock| clock())
}

/// rt's completed handler this thread runs entered at `at` (0: none), handed on where `handed`,
/// its command buffer's [`gpu`] `gpu` and its [`gpu_end`] `end`.
pub(crate) fn set_handler(at: u64, handed: bool, gpu: Option<(u64, u64)>, end: u64) {
    HANDLER.set((at, handed));
    GPU.set(gpu);
    END.set(end);
}

/// The GPU end of the command buffer whose completed handler this thread runs, on the installed
/// clock; 0 where rt holds no GPU times of it, or before a clock is installed.
pub fn gpu_end() -> u64 {
    END.get()
}

/// The decoder's stamps of a command buffer's way up, on the installed clock, each 0 where
/// unknown ([`set_chain`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Chain {
    /// The GPU end of the completion whose interrupt its queue's thread took it after, where none
    /// of its queue's buffers was in flight.
    pub gpu_end: u64,
    /// That interrupt's raise.
    pub raised: u64,
    /// The guest's commit call of it.
    pub committed: u64,
    /// Its take off its queue's ring.
    pub taken: u64,
}

/// The command buffer this thread commits next carries `chain` into its card's record (`None`:
/// nothing).
pub fn set_chain(chain: Option<Chain>) {
    CHAIN.set(chain);
}

/// A command buffer's record on the card ([`take_cards`]), on the installed clock, each time 0
/// where unknown.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CardRecord {
    /// Its GPU start.
    pub start: u64,
    /// Its GPU end.
    pub end: u64,
    /// The device's command buffers committed and not completed as it was committed, every
    /// queue's.
    pub depth: u64,
    /// When rt's commit call of it returned.
    pub returned: u64,
    /// The decoder's stamps of its way up.
    pub chain: Chain,
}

/// A command buffer's place on the card from its commit to its handler's entry ([`card`]).
pub(crate) struct Card {
    chain: Chain,
    depth: AtomicU64,
    returned: AtomicU64,
}

/// The card's place of a command buffer about to be committed on this thread, with this thread's
/// [`Chain`], taken; `None` before a clock is installed.
pub(crate) fn card() -> Option<Arc<Card>> {
    CLOCK.get()?;
    Some(Arc::new(Card {
        chain: CHAIN.take().unwrap_or_default(),
        depth: AtomicU64::new(0),
        returned: AtomicU64::new(0),
    }))
}

impl Card {
    /// Committed now: in flight, its depth those in flight before it.
    pub(crate) fn commits(&self) {
        let before = IN_FLIGHT.fetch_add(1, Ordering::Relaxed);
        self.depth.store(before, Ordering::Relaxed);
    }

    /// rt's commit call returned at `at`.
    pub(crate) fn returned(&self, at: u64) {
        self.returned.store(at, Ordering::Relaxed);
    }

    /// rt refused the commit: never in flight.
    pub(crate) fn refused(&self) {
        IN_FLIGHT.fetch_sub(1, Ordering::Relaxed);
    }

    /// rt's handler of it entered at `entered` on the installed clock and at `rt_now` on rt's host
    /// clock, the GPU having run it from `start` to `end` on rt's (0: no GPU times): out of
    /// flight and recorded. Its GPU end on the installed clock, 0 where unknown.
    pub(crate) fn completed(&self, entered: u64, rt_now: u64, start: u64, end: u64) -> u64 {
        IN_FLIGHT.fetch_sub(1, Ordering::Relaxed);
        let on = |t: u64| {
            if t == 0 || entered == 0 {
                0
            } else {
                entered.saturating_sub(rt_now.saturating_sub(t))
            }
        };
        let record = CardRecord {
            start: on(start),
            end: on(end),
            depth: self.depth.load(Ordering::Relaxed),
            returned: self.returned.load(Ordering::Relaxed),
            chain: self.chain,
        };
        CARDS.lock().push(record);
        record.end
    }
}

/// The card's records since the take before, in the order their handlers entered.
pub fn take_cards() -> Vec<CardRecord> {
    std::mem::take(&mut *CARDS.lock())
}

/// A command buffer's idle and pickup from rt's times of it, in nanoseconds of rt's host clock:
/// its scheduling (`kernel_end`, stamped as its last segment joins a submission, before the
/// `vkQueueSubmit2` call that carries it) and its GPU's start and end. The idle is its GPU start
/// less the latest GPU end of those before it whose handlers ran, 0 where it starts before that;
/// the pickup its GPU start less its scheduling, so the submit call is in it. `None` where rt
/// holds no GPU times of it, or no command buffer's end came before.
pub(crate) fn gpu_times(kernel_end: u64, start: u64, end: u64) -> Option<(u64, u64)> {
    if start == 0 || end == 0 || kernel_end == 0 {
        return None;
    }
    let last = LAST_GPU_END.fetch_max(end, Ordering::Relaxed);
    (last != 0).then(|| (start.saturating_sub(last), start.saturating_sub(kernel_end)))
}

/// The idle and pickup ([`gpu_times`]) of the command buffer whose completed handler this thread
/// runs.
pub fn gpu() -> Option<(u64, u64)> {
    GPU.get()
}

/// When rt's completed handler this thread runs entered (0 before a clock is installed), and
/// whether it was handed on from rt's thread.
pub fn handler() -> (u64, bool) {
    HANDLER.get()
}

/// This thread's commit called rt's at `at`, which took `ns`.
pub(crate) fn set_commit(at: u64, ns: u64) {
    COMMIT.set((at, ns));
}

/// When this thread's last commit called rt's and how long that call took, taken: `None` where it
/// made none since the last take, or before a clock is installed.
pub fn take_commit() -> Option<(u64, u64)> {
    let (at, ns) = COMMIT.replace((0, 0));
    (at != 0).then_some((at, ns))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A commit's stamps are taken once, on the thread that made them
    #[test]
    fn a_commits_stamps_are_taken_once() {
        assert_eq!(take_commit(), None);
        set_commit(5, 7);
        std::thread::scope(|scope| {
            scope.spawn(|| assert_eq!(take_commit(), None));
        });
        assert_eq!(take_commit(), Some((5, 7)));
        assert_eq!(take_commit(), None);
    }

    /// A card's record puts rt's GPU times on the installed clock, with its depth among those
    /// committed and not completed, its commit's return and its chain, once
    #[test]
    fn a_card_record_puts_rt_times_on_the_installed_clock() {
        assert!(card().is_none(), "no clock");
        set_clock(|| 7);
        let chain = Chain {
            gpu_end: 10,
            raised: 20,
            committed: 30,
            taken: 40,
        };
        set_chain(Some(chain));
        let first = card().unwrap();
        let second = card().unwrap();
        first.commits();
        second.commits();
        first.returned(900);
        // Entered at 1 ms on the installed clock, at 5 us on rt's: its GPU ran 1 us to 0.5 us
        // before.
        assert_eq!(first.completed(1_000_000, 5_000, 4_000, 4_500), 999_500);
        let third = card().unwrap();
        third.commits();
        third.refused();
        assert_eq!(second.completed(2_000_000, 6_000, 0, 0), 0);
        let records = take_cards();
        assert_eq!(
            records,
            [
                CardRecord {
                    start: 999_000,
                    end: 999_500,
                    depth: 0,
                    returned: 900,
                    chain,
                },
                CardRecord {
                    depth: 1,
                    ..CardRecord::default()
                },
            ]
        );
        assert_eq!(IN_FLIGHT.load(Ordering::Relaxed), 0);
        assert!(take_cards().is_empty(), "taken once");
    }

    /// A command buffer's idle runs from the latest end before it, and its pickup from its
    /// scheduling
    #[test]
    fn a_command_buffers_idle_runs_from_the_latest_end_before_it() {
        // The first: no end before it.
        assert_eq!(gpu_times(10, 20, 30), None);
        assert_eq!(gpu_times(40, 50, 60), Some((20, 10)));
        // One that starts before the latest end: no idle.
        assert_eq!(gpu_times(45, 55, 70), Some((0, 10)));
        // No GPU times: nothing.
        assert_eq!(gpu_times(80, 0, 0), None);
    }
}
