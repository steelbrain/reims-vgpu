//! The order of commits that `CB_SUBMITTED` promises (docs/graphics-kext/design.md, "Submitted"):
//! a command buffer answered as it is taken, before its walk, is committed before anything the
//! guest sends after the answer is carried out where that could see the commit's place.
//!
//! - **Command buffers.** A buffer answered at its take holds a ticket ([`Ticket`]), issued before
//!   its answer is posted and given back once it is committed (or its stand-in is, or neither can
//!   be). Every command buffer marks the tickets issued as it is begun ([`Sequencer::mark`]), and
//!   is committed only once each of them is given back ([`Sequencer::wait_for`]). A buffer the
//!   guest sends after an answer is begun after it, so it is committed after that buffer, on any
//!   queue of any context, as the platform orders a tracked resource by commit; a buffer begun
//!   before the answer is not held up, as under one lock it could have been carried out first.
//! - **Range messages.** `BUFFER_DID_MODIFY_RANGE` and `TEXTURE_REPLACE_REGION` of a `Managed`
//!   object reach the platform's next submission, so one is carried out only once every ticket of
//!   its context issued before it is given back ([`Sequencer::wait_for_context`]): its object is
//!   its process's alone, so no other context's buffer can see it.
//! - **Cost.** A mark is one atomic load, and so is a commit's check where no ticket is out: a
//!   device where nothing asks for `CB_SUBMITTED` pays nothing more.
//!
//! A ticket is issued after its buffer's begin, so a buffer never waits for its own or a later
//! one, and a buffer holding one waits only for older ones: the waits end.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use reims_vgpu_core::sync::{Condvar, Mutex};

/// The tickets out, by number, with the context of each.
#[derive(Default)]
struct Open(BTreeMap<u64, Option<u32>>);

/// The device's commit order for `CB_SUBMITTED` (the module doc).
#[derive(Default)]
pub(crate) struct Sequencer {
    /// The last ticket issued; changed under `open`'s lock.
    issued: AtomicU64,
    /// Every ticket up to this one is given back.
    done_through: AtomicU64,
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
    /// and the order is busy (the caller answers it from a thread that may wait).
    pub(crate) fn issue(self: &Arc<Self>, context: Option<u32>, try_only: bool) -> Option<Ticket> {
        let mut open = if try_only {
            self.open.try_lock()?
        } else {
            self.open.lock()
        };
        let number = self.issued.load(Ordering::Acquire) + 1;
        self.issued.store(number, Ordering::Release);
        open.0.insert(number, context);
        Some(Ticket {
            sequencer: Arc::clone(self),
            number,
        })
    }

    /// Waits, with no lock of the caller's held, until every ticket up to `mark` is given back.
    pub(crate) fn wait_for(&self, mark: u64) {
        if self.done_through.load(Ordering::Acquire) >= mark {
            return;
        }
        let mut open = self.open.lock();
        while self.done_through.load(Ordering::Acquire) < mark {
            self.given_back.wait(&mut open);
        }
    }

    /// Waits until every ticket of `context` up to `mark` is given back; whether it waited.
    pub(crate) fn wait_for_context(&self, context: u32, mark: u64) -> bool {
        if self.done_through.load(Ordering::Acquire) >= mark {
            return false;
        }
        let mut open = self.open.lock();
        let mut waited = false;
        while open
            .0
            .range(..=mark)
            .any(|(_, holder)| *holder == Some(context))
        {
            waited = true;
            self.given_back.wait(&mut open);
        }
        waited
    }

    /// Ticket `number` is given back.
    fn give_back(&self, number: u64) {
        let mut open = self.open.lock();
        open.0.remove(&number);
        let through = open.0.first_key_value().map_or_else(
            || self.issued.load(Ordering::Acquire),
            |(&first, _)| first - 1,
        );
        self.done_through.store(through, Ordering::Release);
        drop(open);
        self.given_back.notify_all();
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
        s.wait_for(before);
        let went = AtomicBool::new(false);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                s.wait_for(after);
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
        s.wait_for(mark);
    }

    /// A range message waits for its own context's tickets alone.
    #[test]
    fn a_range_message_waits_for_its_context_alone() {
        let s = Arc::new(Sequencer::default());
        let other = s.issue(Some(2), false).unwrap();
        let mark = s.mark();
        assert!(
            !s.wait_for_context(1, mark),
            "another context's ticket holds nothing up"
        );
        let own = s.issue(Some(1), false).unwrap();
        let mark = s.mark();
        std::thread::scope(|scope| {
            let waiter = scope.spawn(|| s.wait_for_context(1, mark));
            std::thread::sleep(Duration::from_millis(50));
            assert!(!waiter.is_finished());
            drop(own);
            assert!(
                waiter.join().unwrap(),
                "it waited for its own context's ticket"
            );
        });
        drop(other);
    }
}
