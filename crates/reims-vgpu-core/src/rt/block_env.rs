//! `BlockEnv`: what every callout of core's sees, whichever thread runs it.
//!
//! A queue's block, a source's handler, a once's block and a Metal completion handler are callouts
//! an exception never unwinds out of: a raise in one with no handler of its own dies at the raise,
//! whatever handlers the thread had (measured for `dispatch_sync` on macOS 26.7). Its code holds
//! under [`Hold::NONE`](crate::rt::Hold::NONE), with no holder: a callout is `Send` and its
//! caller's hold is not, so none reaches it. While a queue's block runs, the thread owns the queue,
//! and a `sync` onto a queue it owns runs its block at once, as a nested call (`rt::queue`'s
//! module doc). So a queue's block or a completion handler enters:
//! - the thread's ownership chain: the queue a worker's block is of, a sync caller's chain plus
//!   the queue, none on a foreign thread;
//! - the "ours" mark, so a panic there aborts before unwinding (6.5): on a platform worker, and
//!   on Metal's thread for a completion handler, which would otherwise unwind into Metal.
//!
//! Dropping the environment restores the caller's chain and mark.

use std::cell::Cell;

use crate::rt::thread::{OursGuard, enter_ours};

/// One queue of a thread's ownership chain: its id and label.
pub(crate) type Owned = (u64, &'static str);

thread_local! {
    /// The queues this thread is running a block of (its ownership chain), outermost first.
    static OWNED: Cell<Vec<Owned>> = const { Cell::new(Vec::new()) };
}

/// This thread's ownership chain.
pub(crate) fn owned_chain() -> Vec<Owned> {
    OWNED.with(|o| {
        let v = o.take();
        o.set(v.clone());
        v
    })
}

fn swap_owned(v: Vec<Owned>) -> Vec<Owned> {
    OWNED.with(|o| o.replace(v))
}

/// The innermost queue this thread runs a block of (tests): what
/// `dispatch_queue_get_label(DISPATCH_CURRENT_QUEUE_LABEL)` reads there.
#[cfg(test)]
pub(crate) fn current_label() -> Option<&'static str> {
    OWNED.with(|o| {
        let v = o.take();
        let label = v.last().map(|&(_, l)| l);
        o.set(v);
        label
    })
}

/// The environment of one callout, entered when it starts and restored when it ends, on a
/// return and on an unwind alike. Thread-bound (`!Send`: it restores the thread's own state).
#[derive(Debug)]
pub struct BlockEnv {
    /// The caller's chain, put back at the drop.
    prev_owned: Vec<Owned>,
    _ours: Option<OursGuard>,
}

impl BlockEnv {
    /// A block of a queue: the thread owns `owned` while it runs. `ours` marks a platform
    /// worker as ours for the block.
    pub(crate) fn block(owned: Vec<Owned>, ours: bool) -> BlockEnv {
        let ours = ours.then(enter_ours);
        BlockEnv {
            prev_owned: swap_owned(owned),
            _ours: ours,
        }
    }

    /// A callout on a foreign thread (Metal's completion thread): it owns no queue, and runs
    /// marked as ours, so a panic in it aborts before it unwinds into the foreign thread's frames.
    pub fn foreign() -> BlockEnv {
        BlockEnv::block(Vec::new(), true)
    }
}

impl Drop for BlockEnv {
    fn drop(&mut self) {
        swap_owned(std::mem::take(&mut self.prev_owned));
    }
}

/// A block as it is submitted: the chain it will own, whether its thread is marked ours, and, in
/// tests, the submitter's death slot, which follows the work (a death under test capture unwinds on
/// and ends the worker, holding the queue, where the process would end).
#[derive(Clone)]
pub(crate) struct Submitted {
    owned: Vec<Owned>,
    ours: bool,
    #[cfg(any(test, feature = "testkit"))]
    slot: Option<std::sync::Arc<crate::testkit::capture::DeathSlot>>,
}

impl Submitted {
    /// A block that will own `owned`, submitted from this thread.
    pub(crate) fn new(owned: Vec<Owned>, ours: bool) -> Submitted {
        Submitted {
            owned,
            ours,
            #[cfg(any(test, feature = "testkit"))]
            slot: crate::testkit::capture::current_slot(),
        }
    }

    /// Runs `f` in the block's environment.
    pub(crate) fn run<R>(self, f: impl FnOnce() -> R) -> R {
        let Submitted {
            owned,
            ours,
            #[cfg(any(test, feature = "testkit"))]
            slot,
        } = self;
        let run = move || {
            let _env = BlockEnv::block(owned, ours);
            f()
        };
        #[cfg(any(test, feature = "testkit"))]
        return crate::testkit::capture::with_slot(slot, run);
        #[cfg(not(any(test, feature = "testkit")))]
        run()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rt::thread::is_our_thread;

    /// A block owns its chain, marked ours; the caller's come back after
    #[test]
    fn a_block_enters_and_restores() {
        {
            let _env = BlockEnv::block(vec![(7, "reims-vgpu.test-queue")], true);
            assert_eq!(current_label(), Some("reims-vgpu.test-queue"));
            assert!(is_our_thread());
        }
        assert_eq!(current_label(), None);
        assert!(!is_our_thread());
    }

    /// A foreign callout owns no queue and is marked ours; the caller's chain comes back after
    #[test]
    fn foreign_owns_nothing() {
        let _outer = BlockEnv::block(vec![(3, "reims-vgpu.reporting")], false);
        {
            let _env = BlockEnv::foreign();
            assert_eq!(current_label(), None);
            assert!(is_our_thread());
        }
        assert!(!is_our_thread());
        assert_eq!(current_label(), Some("reims-vgpu.reporting"));
    }
}
