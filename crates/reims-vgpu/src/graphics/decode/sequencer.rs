//! The order of commits that `CB_SUBMITTED` promises (docs/graphics-kext/design.md, "Submitted"):
//! a command buffer answered as it is taken, before its walk, is committed before anything its
//! process sends after the answer is carried out where that could see the commit's place.
//!
//! - **Command buffers.** A buffer answered at its take holds a ticket ([`Ticket`]), issued before
//!   its answer is posted and given back once it is committed (or its stand-in is, or neither can
//!   be), with its context. Every command buffer marks the tickets issued as it is begun
//!   ([`Sequencer::mark`]), and is committed only once each of them of its own context is given
//!   back ([`Sequencer::wait_for`]). A buffer the guest sends after an answer is begun after it,
//!   so it is committed after that buffer, on any queue of the answered buffer's context, as the
//!   platform orders a tracked resource by commit; a buffer begun before the answer is not held
//!   up, as under one lock it could have been carried out first.
//! - **Other contexts.** A buffer of another context waits for no ticket of this one: it cannot
//!   name the context's objects, which are its process's alone, and Metal orders a tracked
//!   resource by commit only within a process (docs/graphics-kext/dgpu-oracle.md, Q6). What two
//!   processes share is an IOSurface, whose accesses the order of IOSurface accesses
//!   (`SURFACE_ORDER`, [`super::surface_order`]) orders on the GPU, and a surface's device cache,
//!   whose fills and copies wait for its users ([`super::surface_cache`]); neither reads the
//!   commit order, as a Mac's order of IOSurface accesses does not (Q16). So one process's
//!   answered buffer whose walk is long (a cold make in a draw) holds up only its own process's
//!   later commits, as on a Mac, where each process's queues run apart. A ticket or a buffer of
//!   no context is waited for by, and waits for, every one.
//! - **Range messages.** `BUFFER_DID_MODIFY_RANGE` and `TEXTURE_REPLACE_REGION` of a `Managed`
//!   object reach the platform's next submission, so one is carried out only once every ticket of
//!   its context issued before it is given back ([`Sequencer::after_context`]): its object is its
//!   process's alone, so no other context's buffer can see it. Ring 0 does not wait for it: it is
//!   carried out where the last of those tickets is given back, after its buffer's commit, the
//!   context's range messages one at a time in the order they came, and every command buffer of
//!   the context begun after it is committed after it ([`Sequencer::ranges_asked`],
//!   [`Sequencer::wait_for_ranges`]), as one begun after ring 0 carried it out was.
//! - **Cost.** A mark is one atomic load, and so is a commit's check where no ticket is out, and
//!   a begin's check of range messages where none waits: a device where nothing asks for
//!   `CB_SUBMITTED` pays nothing more.
//!
//! A ticket is issued after its buffer's mark, so a buffer never waits for its own or a later
//! one, and a buffer holding one waits only for older ones; a range message waits only for the
//! tickets issued before it, and a buffer only for the range messages that came before its begin:
//! the waits end.
//!
//! A doorbell's thread issues a ticket allocating nothing ([`Sequencer::issue`]): the order keeps
//! room for [`TICKETS_ROOM`] tickets out from its make, and refuses one there past that.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use reims_vgpu_vulkan::core::sync::{Condvar, Mutex, MutexGuard};

/// A range message's work, carried out once its context's tickets before it are given back.
pub(crate) type RangeOp = Box<dyn FnOnce() + Send>;

/// How many tickets out the order has room for from its make, so a doorbell's thread issues one
/// allocating nothing.
const TICKETS_ROOM: usize = 256;

/// The tickets out, and each context's range messages.
struct Open {
    /// From the oldest ticket out to the last issued, in the order issued (so by number, one
    /// apart): each one's number, its context, and whether it is given back, as one given back
    /// before an older one stays until that one is.
    tickets: VecDeque<(u64, Option<u32>, bool)>,
    ranges: HashMap<u32, Ranges>,
}

impl Default for Open {
    fn default() -> Self {
        Open {
            tickets: VecDeque::with_capacity(TICKETS_ROOM),
            ranges: HashMap::new(),
        }
    }
}

impl Open {
    /// The tickets out up to `mark`, each with its context.
    fn out_through(&self, mark: u64) -> impl Iterator<Item = Option<u32>> + '_ {
        (self.tickets.iter())
            .take_while(move |&&(number, _, _)| number <= mark)
            .filter(|&&(_, _, given_back)| !given_back)
            .map(|&(_, context, _)| context)
    }
}

/// Whether a command buffer of `context` is committed after a ticket of `holder`'s (the module
/// doc): one of its own context, or either of no context.
fn orders(context: Option<u32>, holder: Option<u32>) -> bool {
    context.is_none() || holder.is_none() || holder == context
}

/// One context's range messages ([`Sequencer::after_context`]).
#[derive(Default)]
struct Ranges {
    /// Each not yet carried out, with the last ticket issued when it came, in the order they came.
    waiting: VecDeque<(u64, RangeOp)>,
    /// A thread is carrying them out.
    running: bool,
    /// How many came, and how many were carried out.
    asked: u64,
    done: u64,
}

/// The device's commit order for `CB_SUBMITTED` (the module doc).
#[derive(Default)]
pub(crate) struct Sequencer {
    /// The last ticket issued; changed under `open`'s lock.
    issued: AtomicU64,
    /// Every ticket up to this one is given back.
    done_through: AtomicU64,
    /// Range messages that came and are not yet carried out, on every context.
    ranges_waiting: AtomicUsize,
    open: Mutex<Open>,
    given_back: Condvar,
}

/// A command buffer's place in the commit order, given back as it drops.
pub(crate) struct Ticket {
    sequencer: Arc<Sequencer>,
    number: u64,
}

impl std::fmt::Debug for Ticket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Ticket").field(&self.number).finish()
    }
}

impl Sequencer {
    /// The tickets issued so far: a command buffer begun now is committed after each.
    pub(crate) fn mark(&self) -> u64 {
        self.issued.load(Ordering::Acquire)
    }

    /// A ticket for a command buffer of `context` answered at its take; `None` where `try_only`
    /// and the order is busy, or has no room left for a ticket allocating nothing (the caller
    /// answers it from a thread that may wait).
    pub(crate) fn issue(self: &Arc<Self>, context: Option<u32>, try_only: bool) -> Option<Ticket> {
        let mut open = if try_only {
            let open = self.open.try_lock()?;
            if open.tickets.len() == open.tickets.capacity() {
                return None;
            }
            open
        } else {
            self.open.lock()
        };
        let number = self.issued.load(Ordering::Acquire) + 1;
        self.issued.store(number, Ordering::Release);
        open.tickets.push_back((number, context, false));
        Some(Ticket {
            sequencer: Arc::clone(self),
            number,
        })
    }

    /// [`Self::issue`] for a command buffer answered on a doorbell's thread, waiting for nothing
    /// and allocating nothing, with the buffer's mark: every ticket issued before its own
    /// ([`Self::mark`]). `None` where the order is busy or has no room left.
    pub(crate) fn issue_marked(self: &Arc<Self>, context: u32) -> Option<(u64, Ticket)> {
        let ticket = self.issue(Some(context), true)?;
        Some((ticket.number - 1, ticket))
    }

    /// How many tickets are out, not yet given back (for the tests).
    #[cfg(test)]
    pub(crate) fn out(&self) -> usize {
        let open = self.open.lock();
        open.tickets.iter().filter(|t| !t.2).count()
    }

    /// Waits, with no lock of the caller's held, until every ticket up to `mark` that a command
    /// buffer of `context` is committed after is given back: its own context's, and those of no
    /// context (the module doc). A buffer of no context waits for every one.
    pub(crate) fn wait_for(&self, context: Option<u32>, mark: u64) {
        if self.done_through.load(Ordering::Acquire) >= mark {
            return;
        }
        let mut open = self.open.lock();
        while self.done_through.load(Ordering::Acquire) < mark
            && open.out_through(mark).any(|holder| orders(context, holder))
        {
            self.given_back.wait(&mut open);
        }
    }

    /// Whether a command buffer of `context` marked at `mark` would be committed now, waiting for
    /// no ticket ([`Self::wait_for`]); `None` where the order is busy. Waits for nothing.
    pub(crate) fn reached(&self, context: Option<u32>, mark: u64) -> Option<bool> {
        if self.done_through.load(Ordering::Acquire) >= mark {
            return Some(true);
        }
        let open = self.open.try_lock()?;
        Some(!open.out_through(mark).any(|holder| orders(context, holder)))
    }

    /// Whether `asked` range messages of `context` are carried out, so a command buffer begun
    /// after them waits for none ([`Self::wait_for_ranges`]); `None` where the order is busy.
    /// Waits for nothing.
    pub(crate) fn ranges_done(&self, context: u32, asked: u64) -> Option<bool> {
        let open = self.open.try_lock()?;
        Some(open.ranges.get(&context).is_none_or(|r| r.done >= asked))
    }

    /// Carries out `op`, a range message of `context`'s, once every ticket of `context` issued
    /// before now is given back, after the context's range messages before it: here where none
    /// is out, else where the last of them is given back, on that thread, waiting for nothing
    /// here.
    pub(crate) fn after_context(&self, context: u32, op: RangeOp) {
        let mut open = self.open.lock();
        let mark = self.issued.load(Ordering::Acquire);
        let ranges = open.ranges.entry(context).or_default();
        ranges.asked += 1;
        ranges.waiting.push_back((mark, op));
        self.ranges_waiting.fetch_add(1, Ordering::AcqRel);
        self.carry_out(open, context);
    }

    /// How many range messages of `context` came before now, where any is still to be carried
    /// out on the device: a command buffer begun now is committed after each
    /// ([`Self::wait_for_ranges`]); `Some(None)` where none waits. `None` where `try_only` and
    /// the order is busy.
    pub(crate) fn ranges_asked(&self, context: u32, try_only: bool) -> Option<Option<u64>> {
        if self.ranges_waiting.load(Ordering::Acquire) == 0 {
            return Some(None);
        }
        let open = if try_only {
            self.open.try_lock()?
        } else {
            self.open.lock()
        };
        Some(open.ranges.get(&context).map(|r| r.asked))
    }

    /// Waits, with no lock of the caller's held, until `asked` range messages of `context` are
    /// carried out.
    pub(crate) fn wait_for_ranges(&self, context: u32, asked: u64) {
        let mut open = self.open.lock();
        while open.ranges.get(&context).is_some_and(|r| r.done < asked) {
            self.given_back.wait(&mut open);
        }
    }

    /// Carries out, on this thread and one at a time, each range message of `context` whose
    /// tickets are all given back, oldest first, unless another thread is carrying them out.
    fn carry_out<'s>(&'s self, mut open: MutexGuard<'s, Open>, context: u32) {
        loop {
            let Some(r) = open.ranges.get(&context) else {
                return;
            };
            if r.running {
                return;
            }
            let ready = (r.waiting.front()).is_some_and(|&(mark, _)| {
                !open.out_through(mark).any(|holder| holder == Some(context))
            });
            if !ready {
                return;
            }
            let Some(r) = open.ranges.get_mut(&context) else {
                return;
            };
            let Some((_, op)) = r.waiting.pop_front() else {
                return;
            };
            r.running = true;
            drop(open);
            op();
            open = self.open.lock();
            if let Some(r) = open.ranges.get_mut(&context) {
                r.running = false;
                r.done += 1;
            }
            self.ranges_waiting.fetch_sub(1, Ordering::AcqRel);
            self.given_back.notify_all();
        }
    }

    /// Ticket `number` is given back.
    fn give_back(&self, number: u64) {
        let mut open = self.open.lock();
        let at = (open.tickets.front())
            .and_then(|&(first, _, _)| usize::try_from(number.checked_sub(first)?).ok());
        let holder = match at.and_then(|at| open.tickets.get_mut(at)) {
            Some(ticket) => {
                ticket.2 = true;
                ticket.1
            }
            None => None,
        };
        while open
            .tickets
            .front()
            .is_some_and(|&(_, _, given_back)| given_back)
        {
            open.tickets.pop_front();
        }
        let through = open.tickets.front().map_or_else(
            || self.issued.load(Ordering::Acquire),
            |&(first, _, _)| first - 1,
        );
        self.done_through.store(through, Ordering::Release);
        self.given_back.notify_all();
        match holder {
            Some(context)
                if open
                    .ranges
                    .get(&context)
                    .is_some_and(|r| !r.waiting.is_empty()) =>
            {
                self.carry_out(open, context);
            }
            _ => drop(open),
        }
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        self.sequencer.give_back(self.number);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;
    use std::time::Duration;

    use super::*;

    /// A commit marked after a ticket waits for it, and one marked before does not.
    #[test]
    fn a_commit_waits_only_for_the_tickets_issued_before_its_begin() {
        let s = Arc::new(Sequencer::default());
        let before = s.mark();
        let ticket = s.issue(Some(1), false).unwrap();
        let after = s.mark();
        // Marked before the ticket: not held up.
        s.wait_for(Some(1), before);
        let went = AtomicBool::new(false);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                s.wait_for(Some(1), after);
                went.store(true, Ordering::Release);
            });
            std::thread::sleep(Duration::from_millis(50));
            assert!(!went.load(Ordering::Acquire), "held up by the open ticket");
            drop(ticket);
        });
        assert!(went.load(Ordering::Acquire));
    }

    /// Tickets given back out of order: the order holds through the oldest still out.
    #[test]
    fn tickets_given_back_out_of_order_hold_through_the_oldest() {
        let s = Arc::new(Sequencer::default());
        let first = s.issue(None, false).unwrap();
        let second = s.issue(None, false).unwrap();
        let mark = s.mark();
        drop(second);
        assert!(s.done_through.load(Ordering::Acquire) < mark);
        drop(first);
        assert_eq!(s.done_through.load(Ordering::Acquire), mark);
        s.wait_for(None, mark);
    }

    /// A commit waits for its own context's tickets alone: one of context 1 marked after context
    /// 2's ticket goes at once, and one of context 2 waits for it; a ticket of no context holds
    /// up every context's, and a buffer of no context waits for every context's ticket.
    #[test]
    fn a_commit_waits_for_its_own_context_s_tickets_alone() {
        let s = Arc::new(Sequencer::default());
        let other = s.issue(Some(2), false).unwrap();
        let mark = s.mark();
        // Another context's ticket: not held up.
        s.wait_for(Some(1), mark);
        let held_up = |context: Option<u32>, mark: u64, ticket: Ticket| {
            let went = AtomicBool::new(false);
            std::thread::scope(|scope| {
                scope.spawn(|| {
                    s.wait_for(context, mark);
                    went.store(true, Ordering::Release);
                });
                std::thread::sleep(Duration::from_millis(50));
                let early = went.load(Ordering::Acquire);
                drop(ticket);
                assert!(
                    !early,
                    "{context:?} went past an open ticket it is ordered after"
                );
            });
            assert!(went.load(Ordering::Acquire));
        };
        // Its own context's ticket holds it up.
        held_up(Some(2), mark, other);
        // A ticket of no context holds up every context's commit.
        let none = s.issue(None, false).unwrap();
        held_up(Some(1), s.mark(), none);
        // A buffer of no context waits for every context's ticket.
        let own = s.issue(Some(3), false).unwrap();
        held_up(None, s.mark(), own);
    }

    /// A range message waits for its own context's tickets alone, holding up no thread: it is
    /// carried out where the last of them is given back, after the range messages before it, and
    /// a buffer begun after it is committed after it.
    #[test]
    fn a_range_message_is_carried_out_once_its_context_s_tickets_are_given_back() {
        let s = Arc::new(Sequencer::default());
        let done = Arc::new(Mutex::new(Vec::new()));
        let range = |n: u32| {
            let done = Arc::clone(&done);
            Box::new(move || done.lock().push(n)) as RangeOp
        };
        let other = s.issue(Some(2), false).unwrap();
        assert_eq!(s.ranges_asked(1, false), Some(None), "none waits");
        s.after_context(1, range(1));
        assert_eq!(
            *done.lock(),
            [1],
            "another context's ticket holds nothing up"
        );
        let own = s.issue(Some(1), false).unwrap();
        s.after_context(1, range(2));
        s.after_context(1, range(3));
        assert_eq!(
            *done.lock(),
            [1],
            "carried out with its context's ticket out"
        );
        let asked = s.ranges_asked(1, false).flatten().unwrap();
        let went = AtomicBool::new(false);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                s.wait_for_ranges(1, asked);
                went.store(true, Ordering::Release);
            });
            std::thread::sleep(Duration::from_millis(50));
            assert!(
                !went.load(Ordering::Acquire),
                "committed before the range messages"
            );
            drop(own);
        });
        assert!(went.load(Ordering::Acquire));
        assert_eq!(*done.lock(), [1, 2, 3], "in the order they came");
        assert_eq!(s.ranges_asked(1, false), Some(None));
        drop(other);
    }
}
