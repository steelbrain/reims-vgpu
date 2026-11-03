//! Where a queue's thread waits for ring 0 before it hands a command buffer to the host: until
//! the control thread has taken every message the buffer waits for (`control_done` reaches the
//! buffer's count), then until its context's lane has carried out the context's messages among
//! them ([`Lanes`](super::lanes::Lanes)).
//!
//! Each waiting thread sleeps on a wake of its own, filed by what it waits for, and is woken only
//! once that holds: the control thread's move to count `done` wakes the threads waiting for a
//! count at or below `done` ([`Gate::counted`]), and a context's lane carrying its messages out
//! through count `done` wakes that context's threads waiting for its lane through a count at or
//! below `done` ([`Gate::carried_out`]). So a thread is woken at most twice for a buffer, whatever
//! the other processes' messages, where one wake for every waiting thread at each message any
//! thread took or carried out woke each of a device's waiting queues at every message.
//!
//! A thread checks what it waits for with the gate's lock held and files itself under it, and
//! whoever moves the count or the lane does so before it takes the lock to wake, so no wake is
//! lost: either the thread sees the move, or it was filed when the waker looked. A reset, the
//! device's end and anything else that ends a wait other than the move it waits for wake every
//! waiting thread ([`Gate::every`]), which looks again.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use parking_lot::{Condvar, Mutex};

/// The queues' threads waiting for ring 0 (the module doc).
#[derive(Debug, Default)]
pub(super) struct Gate {
    state: Mutex<State>,
}

/// What the gate's lock guards.
#[derive(Debug, Default)]
struct State {
    /// The threads waiting for the control thread's count, by the count each waits for, then by
    /// its number.
    counts: BTreeMap<(u64, u64), Arc<Condvar>>,
    /// The threads waiting for their context's lane, by context, then by the count each waits
    /// for the lane through, then by its number.
    lanes: HashMap<u32, BTreeMap<(u64, u64), Arc<Condvar>>>,
    /// The next waiting thread's number.
    next: u64,
    /// How many times a waiting thread was woken.
    woken: u64,
}

/// What a waiting thread finds when it looks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Look {
    /// The control thread has not taken every message it waits for: it waits for count `0`.
    Count(u64),
    /// The control thread has, and the lane of its context `0` has not carried out the
    /// context's messages through count `1`.
    Lane(u32, u64),
    /// It waits no more.
    Done,
}

impl Gate {
    /// Waits until `look` finds nothing more to wait for, looking again at each wake, each time
    /// with the gate's lock held.
    pub(super) fn wait(&self, mut look: impl FnMut() -> Look) {
        let wake = Arc::new(Condvar::new());
        let mut state = self.state.lock();
        loop {
            let filed = match look() {
                Look::Done => return,
                Look::Count(count) => {
                    let key = (count, state.next);
                    state.counts.insert(key, Arc::clone(&wake));
                    Filed::Count(key)
                }
                Look::Lane(context, through) => {
                    let key = (through, state.next);
                    let lane = state.lanes.entry(context).or_default();
                    lane.insert(key, Arc::clone(&wake));
                    Filed::Lane(context, key)
                }
            };
            state.next += 1;
            wake.wait(&mut state);
            // Woken by what it waited for, its file is gone; by a wake of every thread or a
            // spurious one, it goes now.
            match filed {
                Filed::Count(key) => {
                    state.counts.remove(&key);
                }
                Filed::Lane(context, key) => {
                    if let Some(lane) = state.lanes.get_mut(&context) {
                        lane.remove(&key);
                        if lane.is_empty() {
                            state.lanes.remove(&context);
                        }
                    }
                }
            }
        }
    }

    /// The control thread's count moved to `done`: the threads waiting for a count at or below it
    /// look again.
    pub(super) fn counted(&self, done: u64) {
        let mut state = self.state.lock();
        let Some(&(first, _)) = state.counts.keys().next() else {
            return;
        };
        if first > done {
            return;
        }
        let later = state.counts.split_off(&(done.saturating_add(1), 0));
        let due = std::mem::replace(&mut state.counts, later);
        Self::wake(&mut state, due.into_values());
    }

    /// Context `context`'s lane has carried out the context's messages through count `done`
    /// ([`Lanes::finished`](super::lanes::Lanes::finished)): its threads waiting for the lane
    /// through a count at or below it look again.
    pub(super) fn carried_out(&self, context: u32, done: u64) {
        let mut state = self.state.lock();
        let Some(lane) = state.lanes.get_mut(&context) else {
            return;
        };
        let later = lane.split_off(&(done.saturating_add(1), 0));
        let due = std::mem::replace(lane, later);
        if lane.is_empty() {
            state.lanes.remove(&context);
        }
        Self::wake(&mut state, due.into_values());
    }

    /// A reset, the device's end, or a queue's end: every waiting thread looks again.
    pub(super) fn every(&self) {
        let mut state = self.state.lock();
        let counts = std::mem::take(&mut state.counts);
        let lanes = std::mem::take(&mut state.lanes);
        let all: Vec<Arc<Condvar>> = (counts.into_values())
            .chain(lanes.into_values().flat_map(BTreeMap::into_values))
            .collect();
        Self::wake(&mut state, all.into_iter());
    }

    /// Wakes each of `wakes`, their files already taken out.
    fn wake(state: &mut State, wakes: impl Iterator<Item = Arc<Condvar>>) {
        for wake in wakes {
            state.woken += 1;
            wake.notify_one();
        }
    }

    /// How many times a waiting thread was woken, and how many wait now (the tests).
    #[cfg(test)]
    pub(super) fn counts(&self) -> (u64, usize) {
        let state = self.state.lock();
        let waiting = state.counts.len() + state.lanes.values().map(BTreeMap::len).sum::<usize>();
        (state.woken, waiting)
    }
}

/// Where a waiting thread filed itself.
#[derive(Clone, Copy)]
enum Filed {
    Count((u64, u64)),
    Lane(u32, (u64, u64)),
}

#[cfg(test)]
mod tests;
