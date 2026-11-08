//! Shared events across processes (docs/graphics-kext/design.md, "Shared events across
//! processes"): a process's shared event that another process makes an event of its own from,
//! by the port the kext gave out for it (`SHARED_EVENT_NAME`, then `SHARED_EVENT_IMPORT`).
//!
//! **One host event.** Every holder of a name (the process that made the event, and each that
//! made one from the name) has its own id in its own table, and each id is the one host event:
//! every holder's command buffers signal and wait on it, so a wait in one process is released by
//! a signal or a CPU set in another on the GPU's own timeline, as Metal's are on a Mac.
//!
//! **Values to the other holders.** Each process keeps the value as it knows it (the bundle's
//! `RgfxSharedEvent`), which the host's `EVENT_SIGNALLED` raise. The value one holder gives the
//! event, by a command buffer's signal (where the host posts it to that holder,
//! [`cmdbuf`](super::cmdbuf)'s `Signals`) or by its CPU (`SHARED_EVENT_SET_VALUE`, as the host
//! carries it out), is posted to every other holder, for its own id, on the completion ring of
//! the queue it named as it took the name ([`Holder::listen`]). Each post is tried at once,
//! waiting for nothing: where the ring has no room, or another thread posts there, it is kept,
//! its value only rising, and posted again by a thread of its own ([`Posts`]), so a process that
//! stops taking its rings holds up no other process's posts, nor the thread that signalled. A
//! value only rises, as Metal's does, so a post that comes late, or after a later one, moves
//! nothing.
//!
//! **What holds the event.** Each holder holds it until a destroy of its id or its context's
//! close, and each name until the kext lets go of it (`SHARED_EVENT_UNNAME`), once no process
//! holds the port it gave out for the name: so a process that holds the port makes an event from
//! it after every holder has gone, as a Mac's port holds its kernel event. The host's event goes
//! once its last holder and its last name have; a later import of the name is answered
//! `NOT_FOUND`.
//!
//! **Closes.** A context's close sets each of its shared events past every value
//! (`GraphicsDecoder::release_events`), so its own command buffers that wait on one finish; an
//! event another live context or a name still holds is left as it is, its value another
//! process's to move, as a Mac's event goes on in the other process. The last hold's going, by a
//! destroy, a close or an unname, then sets it past every value, so a closed context's buffer
//! that still waits on it finishes.

use std::collections::{HashMap, HashSet};
use std::mem::size_of;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use reims_vgpu_core::platform::event::SharedEvent as _;
use reims_vgpu_core::platform::{Handle, Platform};
use reims_vgpu_core::sync::Mutex;
use reims_vgpu_graphics::wire::{
    RGFX_CONTEXT_OBJ_ID_SHIFT, RGFX_EVT_EVENT_SIGNALLED, RGFX_EVT_SHARED_EVENT_VALUE,
    RGFX_STATUS_INVALID_ARGUMENT, RGFX_STATUS_NOT_FOUND, RGFX_STATUS_OK, RgfxEventSignalled,
    RgfxMsgHeader, RgfxSharedEventImport, RgfxSharedEventName, RgfxSharedEventUnname,
    RgfxSharedEventValue,
};

use super::{Entry, GraphicsDecoder, ObjHandle, read};

/// Posts one `EVENT_SIGNALLED` onto a holder's listen queue's completion ring, waiting for
/// nothing: false, with nothing posted, where the ring has no room or another thread is posting
/// onto it; true where it is posted, or dropped because the ring is gone.
pub(crate) type ListenSink = Arc<dyn Fn(&RgfxEventSignalled) -> bool + Send + Sync>;

/// Where a command buffer's signal of a shared event goes besides its own process
/// ([`cmdbuf`](super::cmdbuf)'s `Signals`): every other holder of the event's name, where it has
/// one.
pub(crate) trait FanOut: Send + Sync {
    /// The host posted shared event `obj_id`'s value `value` to its own holder.
    fn reached(&self, obj_id: u64, value: u64);
}

/// The `EVENT_SIGNALLED` of `value` for holder `obj_id`.
fn signalled(obj_id: u64, value: u64) -> RgfxEventSignalled {
    RgfxEventSignalled {
        header: RgfxMsgHeader {
            r#type: RGFX_EVT_EVENT_SIGNALLED,
            size: size_of::<RgfxEventSignalled>() as u32,
            seqno: 0,
        },
        obj_id,
        value,
    }
}

/// The context an object id is of.
fn context_of(obj_id: u64) -> u32 {
    (obj_id >> RGFX_CONTEXT_OBJ_ID_SHIFT) as u32
}

/// One holder of a shared event: its id, and where the other holders' values go.
struct Holder {
    obj_id: u64,
    /// Its listen queue's completion ring.
    listen: ListenSink,
}

/// One host event with a name.
struct Share<P: Platform> {
    event: P::SharedEvent,
    holders: Vec<Holder>,
    /// Its names, each held until its unname.
    names: Vec<u64>,
    /// A context closed while it held the event and another hold was left: its value was not set
    /// past every value then, so the last hold's going sets it.
    released_late: bool,
}

impl<P: Platform> Share<P> {
    /// Whether nothing holds it.
    fn unheld(&self) -> bool {
        self.holders.is_empty() && self.names.is_empty()
    }
}

/// [`Shares`]' state.
struct State<P: Platform> {
    /// By share number.
    shares: HashMap<u64, Share<P>>,
    /// The share each name names.
    by_name: HashMap<u64, u64>,
    /// The share each holder's id holds.
    by_obj: HashMap<u64, u64>,
    /// The next share's number.
    next: u64,
}

impl<P: Platform> Default for State<P> {
    fn default() -> Self {
        State {
            shares: HashMap::new(),
            by_name: HashMap::new(),
            by_obj: HashMap::new(),
            next: 1,
        }
    }
}

impl<P: Platform> State<P> {
    /// Takes `obj_id`'s hold of its share: the share's event where it was the last hold and a
    /// closed context's buffers may still wait on it (`released_late`), to set past every value.
    fn let_go(&mut self, obj_id: u64) -> Option<P::SharedEvent> {
        let number = self.by_obj.remove(&obj_id)?;
        let share = self.shares.get_mut(&number)?;
        share.holders.retain(|h| h.obj_id != obj_id);
        self.forget_if_unheld(number)
    }

    /// Forgets share `number` where nothing holds it: its event where a closed context's buffers
    /// may still wait on it, to set past every value.
    fn forget_if_unheld(&mut self, number: u64) -> Option<P::SharedEvent> {
        if !self.shares.get(&number)?.unheld() {
            return None;
        }
        let share = self.shares.remove(&number)?;
        share.released_late.then_some(share.event)
    }
}

/// The names of shared events and their holders (the module doc).
pub(crate) struct Shares<P: Platform> {
    state: Mutex<State<P>>,
    /// The holders, read with no lock: none is the case of every event but a shared one.
    held: AtomicUsize,
    posts: Arc<Posts>,
}

impl<P: Platform> Default for Shares<P> {
    fn default() -> Self {
        Shares {
            state: Mutex::new(State::default()),
            held: AtomicUsize::new(0),
            posts: Arc::default(),
        }
    }
}

impl<P: Platform> Shares<P> {
    /// Names `obj_id`'s `event` `name` (`SHARED_EVENT_NAME`), its other holders' values going to
    /// `listen`: a holder from now on. A second name of an event it already holds is one more of
    /// its names, and its listen queue is the latest named.
    pub(crate) fn name(&self, obj_id: u64, event: &P::SharedEvent, name: u64, listen: ListenSink) {
        let mut state = self.state.lock();
        let state = &mut *state;
        let number = match state.by_obj.get(&obj_id) {
            Some(&number) => number,
            None => {
                let number = state.next;
                state.next += 1;
                state.shares.insert(
                    number,
                    Share {
                        event: event.clone_ref(),
                        holders: Vec::new(),
                        names: Vec::new(),
                        released_late: false,
                    },
                );
                state.by_obj.insert(obj_id, number);
                number
            }
        };
        // A name given again (the kext numbers its names afresh after a device reset) names the
        // event given now.
        if let Some(old) = state.by_name.insert(name, number)
            && old != number
            && let Some(share) = state.shares.get_mut(&old)
        {
            share.names.retain(|&n| n != name);
            drop(state.forget_if_unheld(old));
        }
        let Some(share) = state.shares.get_mut(&number) else {
            return;
        };
        if !share.names.contains(&name) {
            share.names.push(name);
        }
        share.holders.retain(|h| h.obj_id != obj_id);
        share.holders.push(Holder { obj_id, listen });
        self.held.store(state.by_obj.len(), Ordering::Release);
    }

    /// Makes `obj_id` a holder of the event `name` names (`SHARED_EVENT_IMPORT`), its other
    /// holders' values going to `listen`: the event, for its table, and its value as the host's
    /// event has it; `None` where the name was let go of, or never given.
    pub(crate) fn import(
        &self,
        obj_id: u64,
        name: u64,
        listen: ListenSink,
        value: impl FnOnce(&P::SharedEvent) -> u64,
    ) -> Option<(P::SharedEvent, u64)> {
        let mut state = self.state.lock();
        let state = &mut *state;
        let number = *state.by_name.get(&name)?;
        // An id that held another share holds this one now.
        if state.by_obj.get(&obj_id).is_some_and(|&n| n != number) {
            drop(state.let_go(obj_id));
        }
        let share = state.shares.get_mut(&number)?;
        share.holders.retain(|h| h.obj_id != obj_id);
        share.holders.push(Holder { obj_id, listen });
        let event = share.event.clone_ref();
        let at = value(&event);
        state.by_obj.insert(obj_id, number);
        self.held.store(state.by_obj.len(), Ordering::Release);
        Some((event, at))
    }

    /// `obj_id` is destroyed: its hold goes. The event, where it was the last hold and a closed
    /// context's buffers may wait on it, which the caller sets past every value.
    pub(crate) fn forget(&self, obj_id: u64) -> Option<P::SharedEvent> {
        if self.held.load(Ordering::Acquire) == 0 {
            return None;
        }
        let mut state = self.state.lock();
        let event = state.let_go(obj_id);
        self.held.store(state.by_obj.len(), Ordering::Release);
        event
    }

    /// The kext let go of `name` (`SHARED_EVENT_UNNAME`): no import finds it from now on. The
    /// event, where it was the last hold and a closed context's buffers may wait on it, which the
    /// caller sets past every value.
    pub(crate) fn unname(&self, name: u64) -> Option<P::SharedEvent> {
        let mut state = self.state.lock();
        let number = state.by_name.remove(&name)?;
        state.shares.get_mut(&number)?.names.retain(|&n| n != name);
        state.forget_if_unheld(number)
    }

    /// `context` closes: each of its holds goes. The ids of its whose events another context or a
    /// name still holds, whose values the close leaves as they are.
    pub(crate) fn close(&self, context: u32) -> Vec<u64> {
        if self.held.load(Ordering::Acquire) == 0 {
            return Vec::new();
        }
        let mut state = self.state.lock();
        let state = &mut *state;
        let mine: Vec<u64> = (state.by_obj.keys())
            .copied()
            .filter(|&id| context_of(id) == context)
            .collect();
        let mut kept = Vec::new();
        for id in mine {
            let others = (state.by_obj.get(&id))
                .and_then(|n| state.shares.get_mut(n))
                .map(|share| {
                    let others = !share.names.is_empty()
                        || share
                            .holders
                            .iter()
                            .any(|h| context_of(h.obj_id) != context);
                    share.released_late |= others;
                    others
                })
                .unwrap_or(false);
            if others {
                kept.push(id);
            }
            // Its own event is set past every value by the close where nothing else holds it, so
            // nothing is handed back here.
            drop(state.let_go(id));
        }
        self.held.store(state.by_obj.len(), Ordering::Release);
        kept
    }

    /// The device's reset: every name and hold goes, and the posts not yet made.
    pub(crate) fn reset(&self) {
        let state = std::mem::take(&mut *self.state.lock());
        self.held.store(0, Ordering::Release);
        self.posts.clear();
        drop(state);
    }

    /// The CPU of `obj_id`'s holder set the event to `value` (`SHARED_EVENT_SET_VALUE`, carried
    /// out): posted to every other holder.
    pub(crate) fn set(&self, obj_id: u64, value: u64) {
        self.reached(obj_id, value);
    }

    /// Whether `obj_id` holds a named event (the tests' look).
    #[cfg(test)]
    pub(crate) fn holds(&self, obj_id: u64) -> bool {
        self.state.lock().by_obj.contains_key(&obj_id)
    }

    /// Waits until every post kept for later is made (the tests' look).
    #[cfg(test)]
    pub(crate) fn flush(&self) {
        self.posts.flush();
    }
}

impl<P: Platform> FanOut for Shares<P> {
    fn reached(&self, obj_id: u64, value: u64) {
        if self.held.load(Ordering::Acquire) == 0 {
            return;
        }
        let others: Vec<(u64, ListenSink)> = {
            let state = self.state.lock();
            let Some(share) = (state.by_obj.get(&obj_id)).and_then(|n| state.shares.get(n)) else {
                return;
            };
            (share.holders.iter())
                .filter(|h| h.obj_id != obj_id)
                .map(|h| (h.obj_id, Arc::clone(&h.listen)))
                .collect()
        };
        for (holder, listen) in others {
            self.posts.post(holder, value, listen);
        }
    }
}

impl<P: Platform> GraphicsDecoder<P> {
    /// A `SHARED_EVENT_NAME` from ring 0, on its context's lane: the sender's shared event named
    /// device-wide, the values the other holders give it going to `listen`, its listen queue's
    /// completion ring, which the caller found by the message's `listen_queue_obj_id` (`None`:
    /// no queue the host holds). Its seqno and status, which the caller answers in a `COMPLETE`:
    /// `NOT_FOUND` where the sender holds no shared event of the id (one whose create the host
    /// refused among them) or no such queue, and nothing is named.
    pub(crate) fn name_shared_event(
        &self,
        message: &[u8],
        listen: Option<ListenSink>,
    ) -> (Option<u64>, u32) {
        let seqno = read::<RgfxMsgHeader>(message).map(|h| h.seqno);
        let Some(m) = read::<RgfxSharedEventName>(message) else {
            return (seqno, RGFX_STATUS_INVALID_ARGUMENT);
        };
        if m.name == 0 || context_of(m.listen_queue_obj_id) != context_of(m.obj_id) {
            return (seqno, RGFX_STATUS_INVALID_ARGUMENT);
        }
        let (Some(listen), Some(ObjHandle::SharedEvent(event))) = (listen, self.named(m.obj_id))
        else {
            return (seqno, RGFX_STATUS_NOT_FOUND);
        };
        self.shared.shares.name(m.obj_id, event, m.name, listen);
        (seqno, RGFX_STATUS_OK)
    }

    /// A `SHARED_EVENT_IMPORT` from ring 0, on its context's lane: the sender's new shared event
    /// is the host event the name names, stored under its id, the values the other holders give
    /// it going to `listen` (as [`Self::name_shared_event`]'s). Its answer
    /// (`SHARED_EVENT_VALUE`), with the event's value as the host's event has it, or
    /// `NOT_FOUND` where no holder of the name is left or no such queue, and nothing is made;
    /// `None` for a message too short to name its seqno.
    pub(crate) fn import_shared_event(
        &self,
        message: &[u8],
        listen: Option<ListenSink>,
    ) -> Option<RgfxSharedEventValue> {
        let seqno = read::<RgfxMsgHeader>(message)?.seqno;
        let answer = |status, value| RgfxSharedEventValue {
            header: RgfxMsgHeader {
                r#type: RGFX_EVT_SHARED_EVENT_VALUE,
                size: size_of::<RgfxSharedEventValue>() as u32,
                seqno,
            },
            status,
            reserved_14: 0,
            value,
        };
        let Some(m) = read::<RgfxSharedEventImport>(message) else {
            return Some(answer(RGFX_STATUS_INVALID_ARGUMENT, 0));
        };
        if m.obj_id == 0
            || !self.owned(m.obj_id)
            || context_of(m.listen_queue_obj_id) != context_of(m.obj_id)
        {
            return Some(answer(RGFX_STATUS_INVALID_ARGUMENT, 0));
        }
        let Some(listen) = listen else {
            return Some(answer(RGFX_STATUS_NOT_FOUND, 0));
        };
        let imported =
            (self.shared.shares).import(m.obj_id, m.name, listen, |e| e.signaled_value());
        let Some((event, value)) = imported else {
            return Some(answer(RGFX_STATUS_NOT_FOUND, 0));
        };
        self.store(m.obj_id, Entry::of(ObjHandle::SharedEvent(event)));
        Some(answer(RGFX_STATUS_OK, value))
    }

    /// A `SHARED_EVENT_UNNAME` from ring 0: the name's hold goes, and the event, where it was
    /// the last hold and a closed context's command buffers may wait on it, is set past every
    /// value. Not answered; a message too short to name a name names none.
    pub(crate) fn unname_shared_event(&self, message: &[u8]) {
        let Some(m) = read::<RgfxSharedEventUnname>(message) else {
            return;
        };
        if let Some(event) = self.shared.shares.unname(m.name)
            && let Err(raise) = super::refusing(|| event.set_signaled_value(u64::MAX))
        {
            log::warn!(
                target: "reims_vgpu::graphics",
                "shared event named {:#x} was not set past every value at its last hold's going: \
                 {raise}",
                m.name
            );
        }
    }
}

/// How long the thread that makes the kept posts sleeps between its tries.
const RETRY: Duration = Duration::from_millis(1);

/// How many rounds of tries with nothing kept the thread waits before it ends.
const IDLE_ROUNDS: u32 = 100;

/// The posts to other holders a ring had no room for, by holder: the highest value kept for each,
/// and its sink. A thread of its own, made as the first is kept and gone once none has been kept
/// for [`IDLE_ROUNDS`] rounds, tries them again every [`RETRY`].
#[derive(Default)]
struct Posts {
    state: Mutex<PostsState>,
}

#[derive(Default)]
struct PostsState {
    kept: HashMap<u64, (u64, ListenSink)>,
    /// The holders whose posts the thread took out of `kept` and is trying: a post for one waits
    /// behind its try, so a holder's posts are made in the order given.
    trying: HashSet<u64>,
    /// Whether the thread runs.
    running: bool,
    /// Counts the rounds the thread has run since the last reset, so a thread of before a reset
    /// ends there.
    epoch: u64,
}

impl Posts {
    /// Posts `value` to holder `obj_id` through `listen`, at once where nothing is kept or tried
    /// for it and its ring takes it, else kept for the thread.
    fn post(self: &Arc<Self>, obj_id: u64, value: u64, listen: ListenSink) {
        let mut state = self.state.lock();
        if !state.kept.contains_key(&obj_id) && !state.trying.contains(&obj_id) {
            drop(state);
            if listen(&signalled(obj_id, value)) {
                return;
            }
            state = self.state.lock();
        }
        let entry = state
            .kept
            .entry(obj_id)
            .or_insert_with(|| (value, Arc::clone(&listen)));
        entry.0 = entry.0.max(value);
        if state.running {
            return;
        }
        state.running = true;
        let epoch = state.epoch;
        drop(state);
        let posts = Arc::clone(self);
        let spawned = std::thread::Builder::new()
            .name("reims-vgpu.graphics-shared-events".into())
            .spawn(move || posts.run(epoch));
        if spawned.is_err() {
            // Tried again at the next post that finds no room.
            self.state.lock().running = false;
        }
    }

    /// The thread: tries each kept post until none is left for [`IDLE_ROUNDS`] rounds.
    fn run(&self, epoch: u64) {
        let mut idle = 0;
        loop {
            let kept: Vec<(u64, u64, ListenSink)> = {
                let mut state = self.state.lock();
                if state.epoch != epoch {
                    return;
                }
                if state.kept.is_empty() {
                    idle += 1;
                    if idle > IDLE_ROUNDS {
                        state.running = false;
                        return;
                    }
                }
                let kept: Vec<_> = (state.kept.drain())
                    .map(|(id, (value, sink))| (id, value, sink))
                    .collect();
                state.trying = kept.iter().map(|&(id, _, _)| id).collect();
                kept
            };
            if !kept.is_empty() {
                idle = 0;
            }
            for (obj_id, value, listen) in kept {
                let posted = listen(&signalled(obj_id, value));
                let mut state = self.state.lock();
                if state.epoch != epoch {
                    return;
                }
                state.trying.remove(&obj_id);
                if !posted {
                    let entry = state.kept.entry(obj_id).or_insert((value, listen));
                    entry.0 = entry.0.max(value);
                }
            }
            std::thread::sleep(RETRY);
        }
    }

    /// Drops every kept post, and ends the thread.
    fn clear(&self) {
        let mut state = self.state.lock();
        state.kept.clear();
        state.trying.clear();
        state.running = false;
        state.epoch += 1;
    }

    /// Waits until nothing is kept (the tests' look).
    #[cfg(test)]
    fn flush(&self) {
        while {
            let state = self.state.lock();
            !state.kept.is_empty() || !state.trying.is_empty()
        } {
            std::thread::sleep(RETRY);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use reims_vgpu_core::platform::device::GpuDevice;
    use reims_vgpu_core::testkit::FakePlatform;
    use zerocopy::IntoBytes;

    use super::*;

    type Posted = Arc<Mutex<Vec<(u64, u64)>>>;

    /// A sink that takes every post, and what it took.
    fn sink() -> (ListenSink, Posted) {
        let posted: Posted = Arc::default();
        let into = Arc::clone(&posted);
        (
            Arc::new(move |m: &RgfxEventSignalled| {
                into.lock().push((m.obj_id, m.value));
                true
            }),
            posted,
        )
    }

    fn event() -> <FakePlatform as Platform>::SharedEvent {
        FakePlatform::new().device().new_shared_event().unwrap()
    }

    const A: u64 = 1 << RGFX_CONTEXT_OBJ_ID_SHIFT | 7;
    const B: u64 = 2 << RGFX_CONTEXT_OBJ_ID_SHIFT | 9;
    const C: u64 = 3 << RGFX_CONTEXT_OBJ_ID_SHIFT | 4;

    /// A value one holder gives the event goes to each other holder, for its own id, and not back
    /// to the one that gave it.
    #[test]
    fn a_value_goes_to_every_other_holder() {
        let shares = Shares::<FakePlatform>::default();
        let (a, to_a) = sink();
        let (b, to_b) = sink();
        let (c, to_c) = sink();
        shares.name(A, &event(), 5, a);
        assert!(shares.import(B, 5, b, |_| 0).is_some());
        assert!(shares.import(C, 5, c, |_| 0).is_some());
        shares.reached(B, 3);
        shares.set(A, 4);
        assert_eq!(*to_a.lock(), [(A, 3)]);
        assert_eq!(*to_b.lock(), [(B, 4)]);
        assert_eq!(*to_c.lock(), [(C, 3), (C, 4)]);
    }

    /// An import answers the host event's value as it is, through the event it names.
    #[test]
    fn an_import_answers_the_event_and_its_value() {
        let shares = Shares::<FakePlatform>::default();
        let made = event();
        shares.name(A, &made, 5, sink().0);
        let (event, value) = shares.import(B, 5, sink().0, |_| 11).unwrap();
        assert_eq!(value, 11);
        assert_eq!(event.id(), made.id());
        assert!(
            shares.import(B, 6, sink().0, |_| 0).is_none(),
            "a name never given"
        );
    }

    /// A name holds its event as a holder does: with every holder gone the name still makes an
    /// event of it, as a Mac's port does, and its unname, the last hold's going, forgets it, so a
    /// later import finds none. An event of one holder posts nothing.
    #[test]
    fn a_name_holds_its_event_until_its_unname() {
        let shares = Shares::<FakePlatform>::default();
        let (a, to_a) = sink();
        let made = event();
        shares.name(A, &made, 5, a);
        shares.reached(A, 2);
        assert!(to_a.lock().is_empty());
        assert!(shares.import(B, 5, sink().0, |_| 0).is_some());
        assert!(shares.forget(A).is_none());
        assert!(shares.forget(B).is_none());
        assert!(!shares.holds(A) && !shares.holds(B));
        let (event, _) = shares
            .import(C, 5, sink().0, |_| 0)
            .expect("the name holds it");
        assert_eq!(event.id(), made.id());
        assert!(shares.unname(5).is_none(), "C still holds it");
        assert!(
            shares.import(A, 5, sink().0, |_| 0).is_none(),
            "the name is gone"
        );
        assert!(shares.forget(C).is_none(), "no closed context waited on it");
        assert!(shares.unname(5).is_none(), "an unname of a name gone");
    }

    /// A close leaves an event another context or a name holds as it is, and the last hold's
    /// going then hands it back to be set past every value: a holder's destroy, or an unname.
    #[test]
    fn a_close_keeps_an_event_another_context_or_a_name_holds() {
        let shares = Shares::<FakePlatform>::default();
        shares.name(A, &event(), 5, sink().0);
        assert!(shares.import(B, 5, sink().0, |_| 0).is_some());
        assert_eq!(shares.close(1), [A]);
        assert!(!shares.holds(A));
        assert!(shares.unname(5).is_none(), "B holds it");
        assert!(shares.forget(B).is_some(), "the last hold after a close");
        // The name alone holds it past its namer's close.
        shares.name(A, &event(), 6, sink().0);
        assert_eq!(shares.close(1), [A]);
        assert!(shares.unname(6).is_some(), "the last hold after a close");
        // A close that leaves no other hold keeps nothing: its own release sets the event.
        shares.name(A, &event(), 7, sink().0);
        assert!(shares.unname(7).is_none());
        assert!(shares.close(1).is_empty());
        assert!(shares.import(B, 7, sink().0, |_| 0).is_none());
    }

    /// A `SHARED_EVENT_NAME` or `SHARED_EVENT_IMPORT` as the kext builds it.
    fn share(r#type: u32, obj_id: u64, name: u64, listen_queue_obj_id: u64) -> Vec<u8> {
        RgfxSharedEventName {
            header: RgfxMsgHeader {
                r#type,
                size: size_of::<RgfxSharedEventName>() as u32,
                seqno: 40,
            },
            obj_id,
            name,
            listen_queue_obj_id,
        }
        .as_bytes()
        .to_vec()
    }

    /// Through the decoder: a name and an import make one host event of two processes' ids, the
    /// import answered the event's value, and a CPU set by one is posted to the other. The
    /// namer's close leaves the event as it is, and so does the importer's destroy, the name
    /// holding it still; its unname, the last hold's going, sets it past every value, after which
    /// the name finds nothing. A name of no shared event,
    /// and an id or a listen queue of another context's, are refused, naming and making nothing.
    #[test]
    fn a_named_event_is_one_host_event_in_two_processes() {
        use reims_vgpu_graphics::wire::{
            RGFX_CMD_OBJECT_DESTROY, RGFX_CMD_SHARED_EVENT_CREATE, RGFX_CMD_SHARED_EVENT_IMPORT,
            RGFX_CMD_SHARED_EVENT_NAME, RGFX_CMD_SHARED_EVENT_SET_VALUE,
            RGFX_CMD_SHARED_EVENT_UNNAME, RgfxObjectDestroy, RgfxSharedEventCreate,
            RgfxSharedEventSetValue,
        };

        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        let mut dec = GraphicsDecoder::<FakePlatform>::new(
            Arc::clone(&p),
            Arc::new(reims_vgpu_graphics::region::Region::new(0x10000).unwrap()),
        );
        let obj = |context: u64, n: u64| context << RGFX_CONTEXT_OBJ_ID_SHIFT | n;
        let header = |r#type: u32, size: usize| RgfxMsgHeader {
            r#type,
            size: size as u32,
            seqno: 0,
        };
        let create = RgfxSharedEventCreate {
            header: header(
                RGFX_CMD_SHARED_EVENT_CREATE,
                size_of::<RgfxSharedEventCreate>(),
            ),
            obj_id: obj(1, 3),
            initial_value: 3,
        };
        dec.set_context(Some(1));
        assert_eq!(dec.decode(&device, create.as_bytes()).unwrap(), Ok(()));
        let (a, to_a) = sink();
        let named = share(RGFX_CMD_SHARED_EVENT_NAME, obj(1, 3), 7, obj(1, 1));
        assert_eq!(
            dec.name_shared_event(&named, Some(a)),
            (Some(40), RGFX_STATUS_OK)
        );
        // No such shared event; another context's listen queue.
        let none = share(RGFX_CMD_SHARED_EVENT_NAME, obj(1, 4), 8, obj(1, 1));
        assert_eq!(
            dec.name_shared_event(&none, Some(sink().0)),
            (Some(40), RGFX_STATUS_NOT_FOUND)
        );
        let foreign = share(RGFX_CMD_SHARED_EVENT_NAME, obj(1, 3), 8, obj(2, 1));
        assert_eq!(
            dec.name_shared_event(&foreign, Some(sink().0)),
            (Some(40), RGFX_STATUS_INVALID_ARGUMENT)
        );
        let made = match dec.named(obj(1, 3)) {
            Some(ObjHandle::SharedEvent(e)) => e.id(),
            _ => panic!("no shared event"),
        };

        dec.set_context(Some(2));
        let (b, to_b) = sink();
        let import = share(RGFX_CMD_SHARED_EVENT_IMPORT, obj(2, 5), 7, obj(2, 1));
        let answer = dec.import_shared_event(&import, Some(b)).unwrap();
        assert_eq!(
            (answer.header.r#type, answer.header.seqno),
            (RGFX_EVT_SHARED_EVENT_VALUE, 40)
        );
        assert_eq!((answer.status, answer.value), (RGFX_STATUS_OK, 3));
        match dec.named(obj(2, 5)) {
            Some(ObjHandle::SharedEvent(e)) => assert_eq!(e.id(), made),
            _ => panic!("the import made no shared event"),
        }
        // Another context's id, and a name never given.
        let theirs = share(RGFX_CMD_SHARED_EVENT_IMPORT, obj(1, 6), 7, obj(1, 1));
        let refused = dec.import_shared_event(&theirs, Some(sink().0)).unwrap();
        assert_eq!(refused.status, RGFX_STATUS_INVALID_ARGUMENT);
        let unknown = share(RGFX_CMD_SHARED_EVENT_IMPORT, obj(2, 6), 9, obj(2, 1));
        let refused = dec.import_shared_event(&unknown, Some(sink().0)).unwrap();
        assert_eq!((refused.status, refused.value), (RGFX_STATUS_NOT_FOUND, 0));
        assert!(dec.named(obj(2, 6)).is_none());

        // Process 2's CPU sets it: process 1 is posted the value, for its own id.
        let set = RgfxSharedEventSetValue {
            header: header(
                RGFX_CMD_SHARED_EVENT_SET_VALUE,
                size_of::<RgfxSharedEventSetValue>(),
            ),
            obj_id: obj(2, 5),
            value: 9,
        };
        assert_eq!(dec.decode(&device, set.as_bytes()).unwrap(), Ok(()));
        assert_eq!(*to_a.lock(), [(obj(1, 3), 9)]);
        assert!(to_b.lock().is_empty());
        dec.set_context(None);

        let sets = |mark| -> Vec<(u64, u64)> {
            (p.call_log().since(mark).iter())
                .filter(|c| c.sel == "setSignaledValue:" && c.obj.0 == made.0)
                .map(|c| (c.obj.0, c.args[0].as_u64().unwrap()))
                .collect()
        };
        // The namer's close leaves the event to process 2.
        let mark = p.call_log().mark();
        dec.drop_context(&device, 1);
        assert_eq!(sets(mark), []);
        // Process 2's destroy leaves the name's hold; the unname, the last hold's going, sets it
        // past every value.
        let mark = p.call_log().mark();
        let destroy = RgfxObjectDestroy {
            header: header(RGFX_CMD_OBJECT_DESTROY, size_of::<RgfxObjectDestroy>()),
            obj_id: obj(2, 5),
        };
        dec.set_context(Some(2));
        assert_eq!(dec.decode(&device, destroy.as_bytes()).unwrap(), Ok(()));
        assert_eq!(sets(mark), []);
        dec.set_context(Some(0));
        let unname = RgfxSharedEventUnname {
            header: header(
                RGFX_CMD_SHARED_EVENT_UNNAME,
                size_of::<RgfxSharedEventUnname>(),
            ),
            reserved_10: 0,
            name: 7,
        };
        dec.unname_shared_event(unname.as_bytes());
        assert_eq!(sets(mark), [(made.0, u64::MAX)]);
        dec.set_context(Some(2));
        let again = share(RGFX_CMD_SHARED_EVENT_IMPORT, obj(2, 7), 7, obj(2, 1));
        let refused = dec.import_shared_event(&again, Some(sink().0)).unwrap();
        assert_eq!(refused.status, RGFX_STATUS_NOT_FOUND);
        dec.set_context(None);
    }

    /// A post a ring has no room for is made again by the posts' thread, its value the highest
    /// kept, and a later post waits behind it.
    #[test]
    fn a_post_with_no_room_is_made_later() {
        let shares = Shares::<FakePlatform>::default();
        let room = Arc::new(AtomicBool::new(false));
        let posted: Posted = Arc::default();
        let (into, open) = (Arc::clone(&posted), Arc::clone(&room));
        let b: ListenSink = Arc::new(move |m: &RgfxEventSignalled| {
            if !open.load(Ordering::Acquire) {
                return false;
            }
            into.lock().push((m.obj_id, m.value));
            true
        });
        shares.name(A, &event(), 5, sink().0);
        assert!(shares.import(B, 5, b, |_| 0).is_some());
        shares.reached(A, 3);
        shares.reached(A, 7);
        assert!(posted.lock().is_empty());
        room.store(true, Ordering::Release);
        shares.flush();
        // The thread may have been trying 3 as 7 came, and then made both, in that order.
        let posted = posted.lock().clone();
        assert!(
            posted == [(B, 7)] || posted == [(B, 3), (B, 7)],
            "posted {posted:?}"
        );
    }
}
