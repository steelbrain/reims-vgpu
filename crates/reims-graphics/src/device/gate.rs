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
//!
//! **An offer.** A queue's thread that waits with a `COMMAND_BUFFER` taken files it with itself
//! ([`Offer`], [`Gate::wait_offering`]). A lane whose move ends the wait offers the buffer to the
//! host on its own thread before it wakes the waiting one ([`Gate::carried_out_offering`]), so a
//! buffer the host answers at its take is answered there, a thread's wake sooner, as a doorbell's
//! thread answers one that need wait for nothing. Each offer is made once at most, and never once
//! its thread has gone on ([`Offer::withdraw`]): its thread hands the buffer over only once no
//! offer of it is under way, and the host, which keeps what an answer took for the buffer, takes
//! it there in place of its own answer.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use parking_lot::{Condvar, Mutex};

use super::QueueShared;

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
    counts: BTreeMap<(u64, u64), File>,
    /// The threads waiting for their context's lane, by context, then by the count each waits
    /// for the lane through, then by its number.
    lanes: HashMap<u32, BTreeMap<(u64, u64), File>>,
    /// The next waiting thread's number.
    next: u64,
    /// How many times a waiting thread was woken.
    woken: u64,
}

/// A waiting thread's file: its wake, and the command buffer it offers.
#[derive(Debug)]
struct File {
    wake: Arc<Condvar>,
    offer: Option<Arc<Offer>>,
}

/// A `COMMAND_BUFFER` a queue's thread took off its ring and waits at the gate to hand to the
/// host, which the lane's thread that ends the wait offers to the host first (the module doc).
pub(super) struct Offer {
    /// The buffer's queue.
    pub(super) queue: Arc<QueueShared>,
    /// Its message, which its thread takes back as it goes on ([`Self::withdraw`]), and an
    /// offer reads in between.
    bytes: Mutex<Vec<u8>>,
    /// What its thread waits for: the control thread's count, then its context's lane through
    /// `own`.
    pub(super) waits_for: u64,
    pub(super) own: u64,
    claim: Mutex<Claim>,
    settled: Condvar,
}

impl std::fmt::Debug for Offer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Offer")
            .field("queue", &self.queue.obj_id)
            .field("bytes", &self.bytes.lock().len())
            .field("waits_for", &self.waits_for)
            .field("own", &self.own)
            .field("claim", &*self.claim.lock())
            .finish()
    }
}

/// Where an [`Offer`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Claim {
    /// Not offered yet.
    Open,
    /// Being offered.
    Claimed,
    /// Offered, or gone back to its thread unoffered: it is offered no more.
    Done,
}

impl Offer {
    /// `bytes`, a `COMMAND_BUFFER` of `queue`'s whose thread waits for the control thread's count
    /// `waits_for`, then for its context's lane through `own`.
    pub(super) fn new(queue: Arc<QueueShared>, bytes: Vec<u8>, waits_for: u64, own: u64) -> Offer {
        Offer {
            queue,
            bytes: Mutex::new(bytes),
            waits_for,
            own,
            claim: Mutex::new(Claim::Open),
            settled: Condvar::new(),
        }
    }

    /// The waker's take of the offer: whether no one offered it yet and its thread has not gone
    /// on, in which case it is the waker's to offer, once, then [`Self::settle`].
    fn claim(&self) -> bool {
        let mut claim = self.claim.lock();
        let open = *claim == Claim::Open;
        if open {
            *claim = Claim::Claimed;
        }
        open
    }

    /// The waker's offer has ended: made (`offered`), so it is offered no more, or not, as its
    /// thread waits for more than the waker's move, so a later waker may make it.
    fn settle(&self, offered: bool) {
        *self.claim.lock() = if offered { Claim::Done } else { Claim::Open };
        self.settled.notify_all();
    }

    /// The message, for the waker to offer while it holds the claim ([`Self::claim`]).
    pub(super) fn message(&self) -> parking_lot::MutexGuard<'_, Vec<u8>> {
        self.bytes.lock()
    }

    /// Its thread goes on: waits for an offer under way to end, lets no other begin, and takes
    /// its message back, so it hands the host the copy it took off the ring, copying it no more.
    pub(super) fn withdraw(&self) -> Vec<u8> {
        let mut claim = self.claim.lock();
        while *claim == Claim::Claimed {
            self.settled.wait(&mut claim);
        }
        *claim = Claim::Done;
        drop(claim);
        std::mem::take(&mut *self.bytes.lock())
    }
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
    #[cfg(test)]
    pub(super) fn wait(&self, look: impl FnMut() -> Look) {
        self.wait_offering(look, None);
    }

    /// [`Self::wait`], filed with `offer`, the buffer the waiting thread holds, which the lane
    /// whose move ends the wait may offer to the host before it wakes the thread (the module
    /// doc). The thread withdraws it once this returns ([`Offer::withdraw`]).
    pub(super) fn wait_offering(&self, mut look: impl FnMut() -> Look, offer: Option<&Arc<Offer>>) {
        let wake = Arc::new(Condvar::new());
        let file = || File {
            wake: Arc::clone(&wake),
            offer: offer.cloned(),
        };
        let mut state = self.state.lock();
        loop {
            let filed = match look() {
                Look::Done => return,
                Look::Count(count) => {
                    let key = (count, state.next);
                    state.counts.insert(key, file());
                    Filed::Count(key)
                }
                Look::Lane(context, through) => {
                    let key = (through, state.next);
                    let lane = state.lanes.entry(context).or_default();
                    lane.insert(key, file());
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
        Self::wake(&mut state, due.into_values().map(|file| file.wake));
    }

    /// Context `context`'s lane has carried out the context's messages through count `done`
    /// ([`Lanes::finished`](super::lanes::Lanes::finished)): its threads waiting for the lane
    /// through a count at or below it look again.
    #[cfg(test)]
    pub(super) fn carried_out(&self, context: u32, done: u64) {
        self.carried_out_offering(context, done, |_| false);
    }

    /// [`Self::carried_out`], on the lane's thread: the buffer each thread it wakes files is
    /// handed to `offer` first, with the gate's lock let go, as the host's answer may raise the
    /// interrupt, and the raise call the device; then each thread is woken. `offer` answers
    /// whether it offered the buffer to the host, which it does once at most, and only where
    /// nothing more is waited for; one it did not offer stays open for a later waker.
    pub(super) fn carried_out_offering(
        &self,
        context: u32,
        done: u64,
        offer: impl Fn(&Offer) -> bool,
    ) {
        let due = {
            let mut state = self.state.lock();
            let Some(lane) = state.lanes.get_mut(&context) else {
                return;
            };
            let later = lane.split_off(&(done.saturating_add(1), 0));
            let due = std::mem::replace(lane, later);
            if lane.is_empty() {
                state.lanes.remove(&context);
            }
            state.woken += due.len() as u64;
            due
        };
        for file in due.values() {
            if let Some(offered) = &file.offer
                && offered.claim()
            {
                offered.settle(offer(offered));
            }
        }
        for file in due.into_values() {
            file.wake.notify_one();
        }
    }

    /// A reset, the device's end, or a queue's end: every waiting thread looks again.
    pub(super) fn every(&self) {
        let mut state = self.state.lock();
        let counts = std::mem::take(&mut state.counts);
        let lanes = std::mem::take(&mut state.lanes);
        let all: Vec<Arc<Condvar>> = (counts.into_values())
            .chain(lanes.into_values().flat_map(BTreeMap::into_values))
            .map(|file| file.wake)
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
