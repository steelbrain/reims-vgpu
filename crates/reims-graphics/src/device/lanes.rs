//! Ring 0's lanes: what of ring 0 is carried out off the control thread, each context's messages
//! in their ring order on a thread of the context's own, so one context's slow message holds up
//! only that context's later messages (docs/graphics-kext/design.md, "Ring 0's lanes").
//!
//! **Lanes.** The control thread takes ring 0 in ring order, as before, and hands each
//! object-lifecycle message of a context's ([`backlog`](super::backlog)'s `Whose::Context`) to
//! the context's lane, and each `PRESENT_SURFACE` to the presents' lane. A lane carries its jobs
//! out one at a time, in the order they were handed, on a thread made as a job comes to a lane
//! with none, which goes once the lane has had no job for [`IDLE`]. The control thread carries
//! the rest out itself: the display commands, a `SURFACE_DOWNLOAD`, and a message no lane is
//! for, which the host refuses.
//!
//! **What waits for a lane.** A command buffer of context `c` waits for every message of `c`'s
//! published before its queue took it ([`Lanes::through`]), as it waited for the control thread
//! before. A job waits only for jobs handed before it: its own lane's, by the lane's order, and
//! another lane's where its message reads what that lane's messages change, by the edges it is
//! handed with ([`Lanes::hand`]):
//! - a `PRESENT_SURFACE` of surface `id`, for each lane's `TEXTURE_FROM_IOSURFACE` of `id`'s
//!   plane 0, its destroys of a texture the host names `id` by, or one such made there, and its
//!   `CONTEXT_CLOSE` where its context holds one: the host shows the newest live texture over
//!   the surface's plane 0, or refuses the present where it holds none;
//! - a `BLOCK_RELEASE` of a block shared with other contexts, for the lanes of each context that
//!   held it: the host refuses the release while an object of any context lies in the block,
//!   and an earlier holder's destroys and `BLOCK_UNSHARE` are answered by no one;
//! - a `BLOCK_GRANT` in the scanout's span, for every lane: a grant takes the span from whoever
//!   held it, and WindowServer's contexts are granted it in turn;
//! - a `RESOURCE_ID_CHUNK` of a chunk another context held, for that context's `CONTEXT_CLOSE`,
//!   and a context's first message, for every `CONTEXT_CLOSE` not yet carried out (whatever its
//!   number: the kext numbers contexts at their opens, which come in another order than their
//!   first messages): the kext frees a closed context's chunks and its slot of GPU addresses once
//!   the close is answered, or once it gives up waiting for the answer, so a close not yet carried
//!   out may still hold textures at the chunk's ids and buffers in the slot.
//!
//! Each wait is for a job handed before the waiter, and each lane with a job has a thread, so the
//! oldest job not yet carried out is never waiting: the waits end, by induction on the ring
//! order.
//!
//! **A reset** drops every job not yet begun ([`Lanes::reset`]); one begun carries on, and the
//! control thread tells the host of the reset only once it has ended ([`Lanes::wait_begun`]).

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::{Condvar, Mutex};
use zerocopy::FromBytes;

use super::backlog::{Whose, whose};
use crate::ring::Message;
use crate::wire::*;

/// How long a lane's thread waits for a job before it goes.
pub(super) const IDLE: Duration = Duration::from_secs(1);

/// Which lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Key {
    /// A context's: its object-lifecycle messages.
    Context(u32),
    /// The presents': every `PRESENT_SURFACE`, in ring order.
    Presents,
}

impl Key {
    /// The lane `message` is handed to, where one is: a context's for its object-lifecycle
    /// messages, the presents' for a `PRESENT_SURFACE`; `None` for what the control thread
    /// carries out itself.
    pub(super) fn of(message: &Message) -> Option<Key> {
        if message.header.r#type == RGFX_CMD_PRESENT_SURFACE {
            return Some(Key::Presents);
        }
        match whose(message.header.r#type, field(message)) {
            Whose::Context(context) => Some(Key::Context(context)),
            Whose::Every | Whose::NoOne => None,
        }
    }

    /// The name of the lane's thread.
    pub(super) fn thread_name(self) -> String {
        match self {
            Key::Context(context) => format!("reims-vgpu.graphics-context-{context}"),
            Key::Presents => "reims-vgpu.graphics-presents".to_owned(),
        }
    }
}

/// A message handed to a lane.
#[derive(Debug)]
pub(super) struct Job {
    /// The message.
    pub(super) message: Message,
    /// Ring 0's count through it.
    pub(super) through: u64,
    /// The reset count it was taken at.
    pub(super) epoch: u64,
    /// The lanes it waits for first, each through a count.
    after: Vec<(Key, u64)>,
}

/// What a lane's edges read of a job.
#[derive(Clone, Copy, Debug)]
struct Note {
    through: u64,
    epoch: u64,
    r#type: u32,
    /// The 8 bytes at offset 16: an object's id, or a context.
    field: u64,
    /// A `TEXTURE_FROM_IOSURFACE`'s surface and plane.
    surface: Option<(u32, u32)>,
}

impl Note {
    fn of(job: &Job) -> Note {
        let message = &job.message;
        let surface = (message.header.r#type == RGFX_CMD_TEXTURE_FROM_IOSURFACE)
            .then(|| RgfxTextureFromIOSurface::read_from_prefix(&message.bytes).ok())
            .flatten()
            .map(|(m, _)| (m.iosurface_id, m.plane));
        Note {
            through: job.through,
            epoch: job.epoch,
            r#type: message.header.r#type,
            field: field(message).unwrap_or(0),
            surface,
        }
    }
}

/// A lane's jobs.
#[derive(Debug, Default)]
struct Lane {
    /// Handed and not yet begun, in the order they came.
    jobs: VecDeque<(Job, Note)>,
    /// The job its thread carries out now.
    running: Option<Note>,
    /// Ring 0's count through the last job handed to it.
    handed: u64,
    /// A thread serves it.
    served: bool,
    /// Its thread waits here for a job, or for the lanes its first job waits for: a lane is woken
    /// only for what it waits for, as each context's lane stays made for [`IDLE`] after its last
    /// job, and a device of many processes holds many such lanes.
    wake: Arc<Condvar>,
}

impl Lane {
    /// Its jobs not yet carried out, of reset count `epoch`, oldest first.
    fn unfinished(&self, epoch: u64) -> impl Iterator<Item = &Note> {
        (self.running.iter())
            .chain(self.jobs.iter().map(|(_, note)| note))
            .filter(move |note| note.epoch == epoch)
    }

    /// Whether every job of reset count `epoch` it was handed through count `through` is carried
    /// out.
    fn through(&self, epoch: u64, through: u64) -> bool {
        self.unfinished(epoch)
            .next()
            .is_none_or(|n| n.through > through)
    }
}

/// What the lanes' lock guards.
#[derive(Debug, Default)]
struct State {
    lanes: HashMap<Key, Lane>,
    /// The reset count the lanes' jobs are of.
    epoch: u64,
    /// Each shared block's holders since it was granted, made or released, by its offset.
    holders: HashMap<u64, HashSet<u32>>,
    /// The context each texture resource id chunk was last granted to.
    chunks: HashMap<u32, u32>,
    /// The contexts a message was handed for since the device was made or reset. The kext numbers
    /// contexts once each, at their opens, but a context may post its first message after one
    /// opened later posts its own.
    seen: HashSet<u32>,
    /// The lanes whose thread waits for other lanes to carry out the jobs its first job waits
    /// for.
    blocked: HashSet<Key>,
    /// The device ended.
    ended: bool,
}

impl State {
    /// Every lane's wake.
    fn wakes(&self) -> Vec<Arc<Condvar>> {
        (self.lanes.values())
            .map(|lane| Arc::clone(&lane.wake))
            .collect()
    }

    /// Whether every job of the lane `key` handed through `through` is carried out.
    fn through(&self, key: Key, through: u64) -> bool {
        self.lanes
            .get(&key)
            .is_none_or(|lane| lane.through(self.epoch, through))
    }

    /// Each context lane's count through its last job not yet carried out that `matches`.
    fn last_matching(&self, matches: impl Fn(&Note) -> bool) -> Vec<(Key, u64)> {
        let epoch = self.epoch;
        self.lanes
            .iter()
            .filter(|(key, _)| matches!(key, Key::Context(_)))
            .filter_map(|(key, lane)| {
                (lane.unfinished(epoch).filter(|note| matches(note)))
                    .map(|note| note.through)
                    .max()
                    .map(|through| (*key, through))
            })
            .collect()
    }

    /// Every context's lane but `own` with a job not yet carried out, through its last job
    /// handed.
    fn every_lane_but(&self, own: Key) -> Vec<(Key, u64)> {
        let epoch = self.epoch;
        self.lanes
            .iter()
            .filter(|(key, lane)| {
                matches!(key, Key::Context(_))
                    && **key != own
                    && lane.unfinished(epoch).next().is_some()
            })
            .map(|(key, lane)| (*key, lane.handed))
            .collect()
    }

    /// The edges a job `note` of the lane `key` is handed with (the module doc), and what the
    /// lanes keep of it. `surface_textures` answers the objects the host names a surface by.
    fn edges(
        &mut self,
        key: Key,
        message: &Message,
        note: &Note,
        surface_textures: impl FnOnce(u32) -> Vec<u64>,
    ) -> Vec<(Key, u64)> {
        let Key::Context(context) = key else {
            return self.present_edges(message, surface_textures);
        };
        let mut edges = Vec::new();
        if self.seen.insert(context) {
            edges.extend(self.last_matching(|n| n.r#type == RGFX_CMD_CONTEXT_CLOSE));
        }
        match note.r#type {
            RGFX_CMD_BLOCK_SHARE => {
                if let Some(share) = read::<RgfxBlockGrant>(message) {
                    self.holders
                        .entry(share.offset)
                        .or_default()
                        .insert(context);
                }
            }
            RGFX_CMD_BLOCK_GRANT => {
                if let Some(grant) = read::<RgfxBlockGrant>(message) {
                    self.holders.remove(&grant.offset);
                    if grant.offset < RGFX_CONTEXT_ALLOCATIONS_OFFSET {
                        edges.extend(self.every_lane_but(key));
                    }
                }
            }
            RGFX_CMD_SURFACE_PAGES => {
                if let Some(pages) = read::<RgfxSurfacePages>(message) {
                    self.holders.remove(&pages.block_offset);
                }
            }
            RGFX_CMD_BLOCK_RELEASE => {
                if let Some(release) = read::<RgfxBlockRelease>(message) {
                    let held = self.holders.remove(&release.offset).unwrap_or_default();
                    for holder in held.into_iter().filter(|&h| h != context) {
                        let lane = Key::Context(holder);
                        if let Some(handed) = (self.lanes.get(&lane))
                            .filter(|l| l.unfinished(self.epoch).next().is_some())
                            .map(|l| l.handed)
                        {
                            edges.push((lane, handed));
                        }
                    }
                }
            }
            RGFX_CMD_RESOURCE_ID_CHUNK => {
                if let Some(chunk) = read::<RgfxResourceIdChunk>(message) {
                    let last = self.chunks.insert(chunk.chunk, context);
                    if let Some(holder) = last.filter(|&h| h != context) {
                        edges.extend(self.last_matching(|n| {
                            n.r#type == RGFX_CMD_CONTEXT_CLOSE && n.field as u32 == holder
                        }));
                    }
                }
            }
            _ => {}
        }
        edges.retain(|(lane, _)| *lane != key);
        edges
    }

    /// A `PRESENT_SURFACE`'s edges (the module doc).
    fn present_edges(
        &self,
        message: &Message,
        surface_textures: impl FnOnce(u32) -> Vec<u64>,
    ) -> Vec<(Key, u64)> {
        let Some(present) = read::<RgfxPresentSurface>(message) else {
            return Vec::new();
        };
        let id = present.iosurface_id;
        if id == 0 {
            return Vec::new();
        }
        let epoch = self.epoch;
        // What names the surface: the host's textures over it, and those still to be made over it,
        // or made again under one of those ids.
        let mut named: HashSet<u64> = surface_textures(id).into_iter().collect();
        let makes = |note: &Note, named: &HashSet<u64>| {
            note.r#type == RGFX_CMD_TEXTURE_FROM_IOSURFACE
                && (note.surface == Some((id, 0)) || named.contains(&note.field))
        };
        for lane in self.lanes.values() {
            for note in lane.unfinished(epoch) {
                if makes(note, &named) {
                    named.insert(note.field);
                }
            }
        }
        let contexts: HashSet<u32> = named
            .iter()
            .map(|obj| (obj >> RGFX_CONTEXT_OBJ_ID_SHIFT) as u32)
            .collect();
        self.last_matching(|note| match note.r#type {
            RGFX_CMD_TEXTURE_FROM_IOSURFACE => makes(note, &named),
            RGFX_CMD_OBJECT_DESTROY => named.contains(&note.field),
            RGFX_CMD_CONTEXT_CLOSE => contexts.contains(&(note.field as u32)),
            _ => false,
        })
    }
}

/// Ring 0's lanes (the module doc).
#[derive(Debug, Default)]
pub(super) struct Lanes {
    state: Mutex<State>,
    /// A job was carried out, a reset came, or the device ended ([`Self::wait_begun`]).
    moved: Condvar,
}

impl Lanes {
    /// Hands `message`, ring 0's count through it `through`, taken at reset count `epoch`, to the
    /// lane `key`, with the edges it waits for; `surface_textures` answers the objects the host
    /// names an IOSurface by, for a present's. Whether the lane has no thread, so the caller
    /// starts one ([`Self::unserved`] where it cannot).
    pub(super) fn hand(
        &self,
        key: Key,
        message: Message,
        through: u64,
        epoch: u64,
        surface_textures: impl FnOnce(u32) -> Vec<u64>,
    ) -> bool {
        let mut state = self.state.lock();
        if state.epoch != epoch {
            // Taken before a reset the lanes were told of: moot.
            return false;
        }
        let mut job = Job {
            message,
            through,
            epoch,
            after: Vec::new(),
        };
        let note = Note::of(&job);
        job.after = state.edges(key, &job.message, &note, surface_textures);
        let lane = state.lanes.entry(key).or_default();
        lane.handed = through;
        lane.jobs.push_back((job, note));
        let start = !lane.served;
        lane.served = true;
        let wake = Arc::clone(&lane.wake);
        drop(state);
        // A job handed makes no other lane's job ready: each waits only for jobs handed before.
        wake.notify_one();
        start
    }

    /// The lane `key` has no thread after all: its jobs are carried out by whoever calls
    /// [`Self::next`] with `wait` false.
    pub(super) fn unserved(&self, key: Key) {
        if let Some(lane) = self.state.lock().lanes.get_mut(&key) {
            lane.served = false;
        }
    }

    /// The next job of the lane `key`, once each lane it waits for is through its edge, for the
    /// lane's thread: carried out, it is [`Self::finished`]. `None` once the device ends, or where
    /// the lane has had no job for [`IDLE`] (at once, where not `wait`): the lane then has no
    /// thread. A job taken before a reset is dropped.
    pub(super) fn next(&self, key: Key, wait: bool) -> Option<Job> {
        let mut state = self.state.lock();
        loop {
            if state.ended {
                return None;
            }
            let epoch = state.epoch;
            let lane = state.lanes.get(&key)?;
            let wake = Arc::clone(&lane.wake);
            let ready = match lane.jobs.front() {
                Some((job, _)) => {
                    job.epoch != epoch || job.after.iter().all(|&(k, t)| state.through(k, t))
                }
                None => {
                    let gone = !wait || wake.wait_for(&mut state, IDLE).timed_out();
                    let empty = state
                        .lanes
                        .get(&key)
                        .is_none_or(|lane| lane.jobs.is_empty());
                    if gone && empty {
                        if let Some(lane) = state.lanes.get_mut(&key) {
                            lane.served = false;
                            if lane.running.is_none() {
                                state.lanes.remove(&key);
                            }
                        }
                        return None;
                    }
                    continue;
                }
            };
            if !ready {
                state.blocked.insert(key);
                wake.wait(&mut state);
                state.blocked.remove(&key);
                continue;
            }
            let lane = state.lanes.get_mut(&key)?;
            let (job, note) = lane.jobs.pop_front()?;
            if job.epoch != epoch {
                continue;
            }
            lane.running = Some(note);
            return Some(job);
        }
    }

    /// The lane `key`'s thread carried out the job [`Self::next`] gave it.
    pub(super) fn finished(&self, key: Key) {
        let blocked: Vec<Arc<Condvar>> = {
            let mut state = self.state.lock();
            if let Some(lane) = state.lanes.get_mut(&key) {
                lane.running = None;
            }
            (state.blocked.iter())
                .filter_map(|k| state.lanes.get(k).map(|lane| Arc::clone(&lane.wake)))
                .collect()
        };
        for wake in blocked {
            wake.notify_one();
        }
        self.moved.notify_all();
    }

    /// Whether every message of context `context`'s handed through ring 0's count `through` is
    /// carried out: a command buffer of the context waits for it, once the control thread has
    /// handed out every message it waits for.
    pub(super) fn through(&self, context: u32, through: u64) -> bool {
        self.state.lock().through(Key::Context(context), through)
    }

    /// [`Self::through`], waiting for nothing: `None` where the lanes' lock is held elsewhere.
    pub(super) fn try_through(&self, context: u32, through: u64) -> Option<bool> {
        let state = self.state.try_lock()?;
        Some(state.through(Key::Context(context), through))
    }

    /// The device's reset, to reset count `epoch`: every job not yet begun is dropped, and what
    /// the lanes keep of the guest's contexts, blocks and chunks is forgotten. A job begun goes
    /// on ([`Self::wait_begun`]).
    pub(super) fn reset(&self, epoch: u64) {
        let mut state = self.state.lock();
        state.epoch = epoch;
        for lane in state.lanes.values_mut() {
            lane.jobs.clear();
            lane.handed = 0;
        }
        state.holders.clear();
        state.chunks.clear();
        state.seen.clear();
        let wakes = state.wakes();
        drop(state);
        for wake in wakes {
            wake.notify_one();
        }
        self.moved.notify_all();
    }

    /// Waits until no lane carries out a job taken before reset count `epoch`, or the device ends.
    pub(super) fn wait_begun(&self, epoch: u64) {
        let mut state = self.state.lock();
        while !state.ended
            && (state.lanes.values())
                .any(|lane| lane.running.is_some_and(|note| note.epoch != epoch))
        {
            self.moved.wait(&mut state);
        }
    }

    /// The device ended: the lanes' threads go at their next look.
    pub(super) fn end(&self) {
        let wakes = {
            let mut state = self.state.lock();
            state.ended = true;
            state.wakes()
        };
        for wake in wakes {
            wake.notify_one();
        }
        self.moved.notify_all();
    }

    /// Whether no lane has a job not yet carried out (the tests).
    /// Whether the lane `key` has a thread (the tests).
    #[cfg(test)]
    pub(super) fn served(&self, key: Key) -> bool {
        self.state
            .lock()
            .lanes
            .get(&key)
            .is_some_and(|lane| lane.served)
    }

    #[cfg(test)]
    pub(super) fn idle(&self) -> bool {
        let state = self.state.lock();
        let epoch = state.epoch;
        state
            .lanes
            .values()
            .all(|lane| lane.unfinished(epoch).next().is_none())
    }
}

/// The 8 bytes at offset 16 of `message`, where it holds them.
fn field(message: &Message) -> Option<u64> {
    (message.bytes.get(16..24))
        .and_then(|bytes| bytes.try_into().ok())
        .map(u64::from_le_bytes)
}

/// `message`'s bytes as `T`, where they are as many.
fn read<T: FromBytes>(message: &Message) -> Option<T> {
    T::read_from_prefix(&message.bytes).ok().map(|(t, _)| t)
}
