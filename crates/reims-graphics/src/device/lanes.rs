//! Ring 0's lanes: what of ring 0 is carried out off the control thread, each context's messages
//! in their ring order on a thread of the context's own, so one context's slow message holds up
//! only that context's later messages (docs/graphics-kext/design.md, "Ring 0's lanes").
//!
//! **Lanes.** The control thread takes ring 0 in ring order, as before, and hands each
//! `SURFACE_PAGES` and `SURFACE_DOWNLOAD` to the lane of the block it names, each other
//! object-lifecycle message of a context's ([`backlog`](super::backlog)'s `Whose::Context`) to
//! the context's lane, and each `PRESENT_SURFACE`, with the `DISPLAY_COLOR`s that stage its
//! colour, to the presents' lane. A lane carries its jobs out one at a time, in the order they
//! were handed, on one of ring 0's threads (below). The control thread carries the rest out
//! itself: the display commands, and a message no lane is for, which the host refuses. It first
//! offers the host a `SURFACE_DOWNLOAD` of a block whose lane has nothing left to carry out, which
//! the host carries out there where it need wait for nothing (`Host::try_surface_download`): the
//! download keeps its place behind the block's earlier messages, all carried out, and no thread is
//! woken for it. One the host leaves goes to the block's lane.
//!
//! A block's lane keeps the messages that make and copy what the host holds of the block's pages
//! (its view and device cache) in ring order, whichever process posted them and for whichever
//! context, and holds them up behind no context's lane: a `SURFACE_PAGES` the kext posts for a
//! surface's maker from another process's CPU lock, or once the maker has closed, and the
//! `SURFACE_DOWNLOAD` or `BLOCK_RELEASE` it posts next without waiting for the answer, are
//! carried out in that order while the maker's lane is behind a slow message.
//!
//! **What waits for a lane.** A command buffer of context `c` waits for every message of `c`'s
//! published before its queue took it ([`Lanes::through`]), on `c`'s lane and on the blocks'
//! lanes, as it waited for the control thread before. A job waits only for jobs handed before it:
//! its own lane's, by the lane's order, and another lane's where its message reads what that
//! lane's messages change, by the edges it is handed with ([`Lanes::hand`]):
//! - a `PRESENT_SURFACE` of surface `id`, for each lane's `TEXTURE_FROM_IOSURFACE` of `id`'s
//!   plane 0, its destroys of a texture the host names `id` by, or one such made there, and its
//!   `CONTEXT_CLOSE` where its context holds one: the host shows the newest live texture over
//!   the surface's plane 0, or refuses the present where it holds none;
//! - a `BLOCK_RELEASE` of a block shared with other contexts, for the lanes of each context that
//!   held it: the host refuses the release while an object of any context lies in the block,
//!   and an earlier holder's destroys and `BLOCK_UNSHARE` are answered by no one;
//! - a `BLOCK_GRANT` in the scanout's span, for every lane: a grant takes the span from whoever
//!   held it, and WindowServer's contexts are granted it in turn;
//! - a `RESOURCE_ID_CHUNK` or `RESOURCE_ID_SMALL_CHUNK` of ids another context held, for that
//!   context's `CONTEXT_CLOSE`, and a context's first message, for every `CONTEXT_CLOSE` not yet
//!   carried out (whatever its number: the kext numbers contexts at their opens, which come in
//!   another order than their first messages): the kext frees a closed context's chunks and its
//!   slot of GPU addresses once the close is answered, so a close not yet carried out may still
//!   hold textures at the chunk's ids and buffers in the slot;
//! - a message of context `c`'s on `c`'s lane, for each block's lane through its last
//!   `SURFACE_PAGES` for `c` handed before it: `c`'s later messages see the block as its pages
//!   left it, as when they shared one lane;
//! - a `BUFFER_OVER_PAGES`, `HEAP_OVER_PAGES`, `BUFFER_OVER_SURFACE_PAGES` or `BLOCK_RELEASE`
//!   of a block, for the block's lane: each reads the block's view or device cache, which the
//!   block's lane makes and copies (below);
//! - a `SURFACE_PAGES` of a block for context `c`, for `c`'s lane through its last message of
//!   those four kinds that names the block: a block described again is described after what was
//!   made over it before.
//!
//! **What names a block.** Every message that names a region block, or the pages it describes, by
//! its offset, and what orders it against the block's lane:
//! - `SURFACE_PAGES` (every flavour: the lanes read no flag) and `SURFACE_DOWNLOAD`: the block's
//!   lane, in ring order. A download waits for no other lane, as on the control thread before: it
//!   reads the device cache and writes the pages, which no lane's message writes (command buffers
//!   write the cache, and the host orders the copy after those that used it).
//! - `BUFFER_OVER_PAGES`, `HEAP_OVER_PAGES`, `BUFFER_OVER_SURFACE_PAGES` and `BLOCK_RELEASE`: the
//!   block's lane through their position (above).
//! - `BLOCK_GRANT`, `BLOCK_SHARE` and `BLOCK_UNSHARE`: no edge. They change only the region's
//!   grants and holds, the holders the lanes keep at the hand-over, and, at an unshare, the host's
//!   count of the holder's command buffers the block's release waits for, which the release reads
//!   after the holder's lane; no block lane's job reads any of them. The kext posts a block's
//!   grant or share only once the block's first `SURFACE_PAGES` is answered, and gives a block
//!   out again only once its `BLOCK_RELEASE`, which waited for the block's lane, is answered OK.
//! - `TEXTURE_FROM_IOSURFACE`: names the block only through its buffer, which a
//!   `BUFFER_OVER_SURFACE_PAGES` of the context's made before it on the same lane, after the
//!   block's lane; a `SURFACE_PAGES` its own context posted before it comes first by the edge
//!   above. What it reads of the block (whether it has a device cache) the block's first
//!   `SURFACE_PAGES` decides, which is answered before any process learns of the block.
//! - `QUEUE_CREATE` (its rings), `COMMAND_BUFFER_IN_REGION` and `COMMAND_BUFFER_PART` (their
//!   bytes), `BUFFER_AT_OFFSET`, `BUFFER_OVER_HOST_MEMORY`, `HEAP_OVER_HOST_MEMORY`,
//!   `LIBRARY_FROM_REGION`, `ARGUMENT_LAYOUT` and `ICB_SLOTS` (their spans): spans of the
//!   process's own blocks, checked against the region's grants and holds, which no block lane's
//!   job changes; the host waits for no block's pages for them.
//! - A command buffer's `RGFX_OP_SURFACE_USE`s: the buffer waits for its own context's messages on
//!   every lane (above), and the host waits for a block's `SURFACE_PAGES` being carried out
//!   before it uses the block. `RGFX_OP_SURFACE_ORDER` names queues and seqnos, no block.
//! - A CPU lock posts a `SURFACE_DOWNLOAD` after a GPU write, and, where the block's pages were
//!   put off, the `SURFACE_PAGES` that hands them over before it, on the block's lane; an unlock
//!   posts nothing (its CPU generation reaches the host in the next command buffer's
//!   `RGFX_OP_SURFACE_USE`). A synchronize (`BUFFER_DID_MODIFY_RANGE`, a blit's) names objects.
//! - `PRESENT_SURFACE` names a surface, not a block: it waits for the textures over the surface
//!   (above) and, in the host, for the block's `SURFACE_PAGES` being carried out; a present that
//!   races a CPU lock's download in the guest races it here too, as before.
//! - `SET_SCANOUT` and `SET_CURSOR` name the scanout's and the cursor's spans, which hold no
//!   process's block.
//!
//! Each wait is for a job handed before the waiter, and each lane with a job has a thread, so the
//! oldest job not yet carried out is never waiting: the waits end, by induction on the ring
//! order.
//!
//! **Threads.** Ring 0's threads (`reims-vgpu.graphics-ring0-<i>`) serve the lanes: a lane that is
//! handed a job and has no thread is given the thread that went idle last ([`Lanes::hand`]), or
//! one made for it where none waits, and keeps it until it has no job left; the thread then waits
//! for the next lane, and goes once it has waited [`KEEP`] (the [`keep`](crate::keep) module). A
//! lane with nothing to do holds no thread, and a thread is woken only for a lane it is given. So
//! the device holds as many threads as it had lanes busy at once in the last [`KEEP`], whichever
//! contexts and blocks they were: with a thread of each lane's own, made as a job came to a lane
//! with none and gone once the lane had had none for 1 s, every block's first `SURFACE_PAGES` made
//! a thread, and so did every lane's job after a pause of a second, 345 threads in 30 s of a
//! WindowServer probe (car 12's B1, msi RTX 5080 host; measured). A thread is never refused to a
//! lane with a job, so the waits still end.
//!
//! **A reset** drops every job not yet begun ([`Lanes::reset`]); one begun carries on, and the
//! control thread tells the host of the reset only once it has ended ([`Lanes::wait_begun`]).

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;

use parking_lot::{Condvar, Mutex};
use zerocopy::FromBytes;

use super::backlog::{Whose, whose};
#[cfg(doc)]
use crate::keep::KEEP;
use crate::keep::Keep;
use crate::ring::Message;
use crate::wire::*;

/// The name of ring 0's thread `index` (the module doc, "Threads").
pub(super) fn thread_name(index: usize) -> String {
    format!("reims-vgpu.graphics-ring0-{index}")
}

/// Which lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Key {
    /// A context's: its object-lifecycle messages but those of a block's lane.
    Context(u32),
    /// The presents': every `PRESENT_SURFACE`, and every `DISPLAY_COLOR`, which stages the
    /// colour of the present after it, in ring order.
    Presents,
    /// The block's at this region offset: every `SURFACE_PAGES` and `SURFACE_DOWNLOAD` of it, in
    /// ring order.
    Block(u64),
}

impl Key {
    /// The lane `message` is handed to, where one is: a block's for a `SURFACE_PAGES` or a
    /// `SURFACE_DOWNLOAD` long enough to name it, a context's for its other object-lifecycle
    /// messages, the presents' for a `PRESENT_SURFACE` or a `DISPLAY_COLOR`; `None` for what the
    /// control thread carries out itself.
    pub(super) fn of(message: &Message) -> Option<Key> {
        match message.header.r#type {
            RGFX_CMD_PRESENT_SURFACE | RGFX_CMD_DISPLAY_COLOR => return Some(Key::Presents),
            RGFX_CMD_SURFACE_PAGES | RGFX_CMD_SURFACE_DOWNLOAD => {
                if let Some(block) = block_of(message) {
                    return Some(Key::Block(block));
                }
            }
            _ => {}
        }
        match whose(message.header.r#type, field(message)) {
            Whose::Context(context) => Some(Key::Context(context)),
            Whose::Every | Whose::NoOne => None,
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
    /// The block it names by offset, for the kinds a block's lane orders ([`block_of`]).
    block: Option<u64>,
    /// The context whose command buffers wait for it (`Whose::Context`).
    context: Option<u32>,
}

impl Note {
    fn of(job: &Job) -> Note {
        let message = &job.message;
        let surface = (message.header.r#type == RGFX_CMD_TEXTURE_FROM_IOSURFACE)
            .then(|| RgfxTextureFromIOSurface::read_from_prefix(&message.bytes).ok())
            .flatten()
            .map(|(m, _)| (m.iosurface_id, m.plane));
        let field = field(message);
        let context = match whose(message.header.r#type, field) {
            Whose::Context(context) => Some(context),
            Whose::Every | Whose::NoOne => None,
        };
        Note {
            through: job.through,
            epoch: job.epoch,
            r#type: message.header.r#type,
            field: field.unwrap_or(0),
            surface,
            block: block_of(message),
            context,
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
    /// A thread serves it, or was given it.
    served: bool,
    /// Its thread waits here for the lanes its first job waits for: a lane is woken only for what
    /// it waits for, as a device of many processes holds many lanes.
    wake: Arc<Condvar>,
}

/// One of ring 0's threads (the module doc, "Threads").
#[derive(Debug, Default)]
struct Ring0Thread {
    /// The lane it was given to serve next, while it waited for one.
    given: Option<Key>,
    /// It waits here for a lane.
    wake: Arc<Condvar>,
    /// When it last began to wait for a lane, on [`Lanes::keep`]'s clock: read as it joins the
    /// spares, under their lock, so a test that sees it there and moves the clock moves it past
    /// its wait's start.
    since: u64,
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
    /// Each shared block's holders since it was granted or released, by its offset. A
    /// `SURFACE_PAGES` leaves them: one the kext posts for a block it has handed out, from a CPU
    /// lock or before the block's release, finds the block held.
    holders: HashMap<u64, HashSet<u32>>,
    /// The context each small chunk of the texture resource ids ([`RGFX_RESOURCE_ID_SMALL_CHUNK`]
    /// ids) was last granted to, by a `RESOURCE_ID_SMALL_CHUNK` or its chunk's
    /// `RESOURCE_ID_CHUNK`.
    chunks: HashMap<u32, u32>,
    /// The contexts a message was handed for since the device was made or reset. The kext numbers
    /// contexts once each, at their opens, but a context may post its first message after one
    /// opened later posts its own.
    seen: HashSet<u32>,
    /// The jobs on the blocks' lanes not yet carried out that a context's command buffers wait
    /// for (its `SURFACE_PAGES`), by context, then ring 0's count through each, then block.
    pending: BTreeSet<(u32, u64, u64)>,
    /// The lanes whose thread waits for other lanes to carry out the jobs its first job waits
    /// for: each is woken once they have ([`Lanes::finished`]).
    blocked: HashSet<Key>,
    /// Ring 0's threads, by the index in their names; `None` where none has it.
    threads: Vec<Option<Ring0Thread>>,
    /// The threads that wait for a lane, by index, the one that went idle last at the end.
    spares: Vec<usize>,
    /// The device ended.
    ended: bool,
    /// How many of ring 0's threads were made, and how many went (the tests).
    #[cfg(test)]
    made: u64,
    #[cfg(test)]
    gone: u64,
    /// How many times a blocked lane was woken (the tests).
    #[cfg(test)]
    woken: u64,
}

impl State {
    /// Whether `job` can begin: every lane it waits for is through its edge, or it was taken
    /// before the last reset, which drops it.
    fn ready(&self, job: &Job) -> bool {
        job.epoch != self.epoch || job.after.iter().all(|&(k, t)| self.through(k, t))
    }

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

    /// Ring 0's count through which every message of context `context`'s handed is carried out,
    /// on its lane and on the blocks' (`u64::MAX` where none is left): one below the first not
    /// yet carried out.
    fn context_done(&self, context: u32) -> u64 {
        let epoch = self.epoch;
        let lane = (self.lanes.get(&Key::Context(context)))
            .and_then(|lane| lane.unfinished(epoch).next())
            .map(|note| note.through);
        let block = (self.pending)
            .range((context, 0, 0)..=(context, u64::MAX, u64::MAX))
            .next()
            .map(|&(_, through, _)| through);
        lane.into_iter()
            .chain(block)
            .min()
            .map_or(u64::MAX, |through| through.saturating_sub(1))
    }

    /// The lane `key`'s count through its last job handed, where it has a job not yet carried
    /// out.
    fn busy_through(&self, key: Key) -> Option<u64> {
        (self.lanes.get(&key))
            .filter(|lane| lane.unfinished(self.epoch).next().is_some())
            .map(|lane| lane.handed)
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
        let context = match key {
            Key::Context(context) => context,
            Key::Presents => return self.present_edges(message, surface_textures),
            Key::Block(block) => return self.block_edges(block, note),
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
            RGFX_CMD_RESOURCE_ID_CHUNK | RGFX_CMD_RESOURCE_ID_SMALL_CHUNK => {
                if let Some(chunk) = read::<RgfxResourceIdChunk>(message) {
                    // By small chunk: a chunk of 64 ids is four.
                    let per = if note.r#type == RGFX_CMD_RESOURCE_ID_SMALL_CHUNK {
                        1
                    } else {
                        (RGFX_RESOURCE_ID_CHUNK / RGFX_RESOURCE_ID_SMALL_CHUNK) as u32
                    };
                    let first = chunk.chunk.saturating_mul(per);
                    let mut holders = Vec::new();
                    for small in first..first.saturating_add(per) {
                        let last = self.chunks.insert(small, context);
                        if let Some(holder) = last.filter(|&h| h != context)
                            && !holders.contains(&holder)
                        {
                            holders.push(holder);
                        }
                    }
                    for holder in holders {
                        edges.extend(self.last_matching(|n| {
                            n.r#type == RGFX_CMD_CONTEXT_CLOSE && n.field as u32 == holder
                        }));
                    }
                }
            }
            _ => {}
        }
        // The block's lane, for a message that reads what it makes of the block.
        if let Some(block) = note.block
            && let Some(handed) = self.busy_through(Key::Block(block))
        {
            edges.push((Key::Block(block), handed));
        }
        // The blocks' lanes, through the context's last `SURFACE_PAGES` on each.
        let mut last: HashMap<u64, u64> = HashMap::new();
        for &(_, through, block) in
            (self.pending).range((context, 0, 0)..=(context, u64::MAX, u64::MAX))
        {
            let at = last.entry(block).or_default();
            *at = (*at).max(through);
        }
        edges.extend(last.into_iter().map(|(block, t)| (Key::Block(block), t)));
        edges.retain(|(lane, _)| *lane != key);
        edges
    }

    /// The edges of a job `note` of block `block`'s lane (the module doc): a `SURFACE_PAGES` for
    /// context `c` waits for `c`'s lane through its last message not yet carried out that names
    /// the block; a `SURFACE_DOWNLOAD` waits for no other lane.
    fn block_edges(&self, block: u64, note: &Note) -> Vec<(Key, u64)> {
        let (RGFX_CMD_SURFACE_PAGES, Some(context)) = (note.r#type, note.context) else {
            return Vec::new();
        };
        let lane = Key::Context(context);
        let epoch = self.epoch;
        (self.lanes.get(&lane))
            .and_then(|l| {
                (l.unfinished(epoch))
                    .filter(|n| n.block == Some(block))
                    .map(|n| n.through)
                    .max()
            })
            .map(|through| (lane, through))
            .into_iter()
            .collect()
    }

    /// A `PRESENT_SURFACE`'s edges (the module doc).
    fn present_edges(
        &self,
        message: &Message,
        surface_textures: impl FnOnce(u32) -> Vec<u64>,
    ) -> Vec<(Key, u64)> {
        // A `DISPLAY_COLOR` waits for nothing but the presents' lane's order.
        if message.header.r#type != RGFX_CMD_PRESENT_SURFACE {
            return Vec::new();
        }
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
    /// How long a thread with no lane waits for one before it goes: [`KEEP`], but in the tests.
    keep: Keep,
}

/// What [`Lanes::hand`] asks of its caller.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Hand {
    /// Nothing: the lane has a thread, or was given one that waited for a lane.
    Served,
    /// To start ring 0's thread `index` ([`thread_name`]), to serve the lane first
    /// ([`Lanes::unstarted`] where it cannot).
    Start(usize),
}

impl Lanes {
    /// Lanes whose threads wait `keep` for a lane before they go (the tests; else [`KEEP`]).
    #[cfg(test)]
    pub(super) fn with_keep(keep: Keep) -> Lanes {
        Lanes {
            keep,
            ..Lanes::default()
        }
    }

    /// Hands `message`, ring 0's count through it `through`, taken at reset count `epoch`, to the
    /// lane `key`, with the edges it waits for; `surface_textures` answers the objects the host
    /// names an IOSurface by, for a present's. Where the lane has no thread, it is given the
    /// thread that went idle last, or the caller starts one for it.
    pub(super) fn hand(
        &self,
        key: Key,
        message: Message,
        through: u64,
        epoch: u64,
        surface_textures: impl FnOnce(u32) -> Vec<u64>,
    ) -> Hand {
        let mut state = self.state.lock();
        if state.epoch != epoch {
            // Taken before a reset the lanes were told of: moot.
            return Hand::Served;
        }
        let mut job = Job {
            message,
            through,
            epoch,
            after: Vec::new(),
        };
        let note = Note::of(&job);
        job.after = state.edges(key, &job.message, &note, surface_textures);
        if let (Key::Block(block), Some(context)) = (key, note.context) {
            state.pending.insert((context, through, block));
        }
        let lane = state.lanes.entry(key).or_default();
        lane.handed = through;
        lane.jobs.push_back((job, note));
        // A job handed makes no lane's job ready, its own lane's first among them: each waits only
        // for jobs handed before. So a lane with a thread has nothing to be woken for.
        if lane.served {
            return Hand::Served;
        }
        lane.served = true;
        if let Some(index) = state.spares.pop()
            && let Some(thread) = state.threads.get_mut(index).and_then(Option::as_mut)
        {
            thread.given = Some(key);
            let wake = Arc::clone(&thread.wake);
            drop(state);
            wake.notify_one();
            return Hand::Served;
        }
        let threads = &mut state.threads;
        let index = threads
            .iter()
            .position(Option::is_none)
            .unwrap_or(threads.len());
        if index == threads.len() {
            threads.push(None);
        }
        if let Some(slot) = threads.get_mut(index) {
            *slot = Some(Ring0Thread::default());
        }
        #[cfg(test)]
        {
            state.made += 1;
        }
        Hand::Start(index)
    }

    /// Whether the lane `key` has no job of the current reset count not yet carried out: none
    /// handed and not begun, and none running.
    pub(super) fn lane_idle(&self, key: Key) -> bool {
        self.state.lock().busy_through(key).is_none()
    }

    /// Ring 0's thread `index`, which [`Self::hand`] asked for to serve the lane `key`, could not
    /// be started: the lane has no thread after all, and its jobs are carried out by whoever calls
    /// [`Self::next`] with no thread's index.
    pub(super) fn unstarted(&self, index: usize, key: Key) {
        let mut state = self.state.lock();
        if let Some(slot) = state.threads.get_mut(index) {
            *slot = None;
        }
        #[cfg(test)]
        {
            state.made -= 1;
        }
        if let Some(lane) = state.lanes.get_mut(&key) {
            lane.served = false;
        }
    }

    /// The next job of the lane `key`, once each lane it waits for is through its edge, for the
    /// thread that serves it, ring 0's thread `spare` where it is one: carried out, it is
    /// [`Self::finished`]. `None` once the device ends, or where the lane has no job left: the lane
    /// then has no thread, and ring 0's thread `spare` waits for another lane
    /// ([`Self::park`]), from here, so a job handed to the lane meanwhile is given to it. A job
    /// taken before a reset is dropped.
    pub(super) fn next(&self, key: Key, spare: Option<usize>) -> Option<Job> {
        let mut state = self.state.lock();
        loop {
            if state.ended {
                return None;
            }
            let epoch = state.epoch;
            let lane = state.lanes.get(&key)?;
            let wake = Arc::clone(&lane.wake);
            let ready = match lane.jobs.front() {
                Some((job, _)) => state.ready(job),
                None => {
                    if let Some(lane) = state.lanes.get_mut(&key) {
                        lane.served = false;
                        if lane.running.is_none() {
                            state.lanes.remove(&key);
                        }
                    }
                    if let Some(index) = spare {
                        self.spare(&mut state, index);
                    }
                    return None;
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

    /// The lane `key`'s thread carried out the job [`Self::next`] gave it. Where the job was a
    /// context's, one whose command buffers wait for it, that context, and ring 0's count through
    /// which every message of the context's handed is carried out ([`Self::through`] holds for
    /// each count up to it; `u64::MAX` where none is left).
    pub(super) fn finished(&self, key: Key) -> Option<(u32, u64)> {
        let (blocked, done): (Vec<Arc<Condvar>>, Option<(u32, u64)>) = {
            let mut state = self.state.lock();
            let epoch = state.epoch;
            let ran = (state.lanes.get_mut(&key)).and_then(|lane| lane.running.take());
            let context = match key {
                Key::Context(context) => Some(context),
                Key::Presents => None,
                Key::Block(block) => ran
                    .filter(|note| note.epoch == epoch)
                    .and_then(|note| note.context.map(|c| (c, note.through)))
                    .map(|(context, through)| {
                        state.pending.remove(&(context, through, block));
                        context
                    }),
            };
            let done = context.map(|context| (context, state.context_done(context)));
            // Only the blocked lanes this job let go: a lane waits for other lanes' jobs handed
            // before its own, and only their carrying out moves what it waits for. A device of
            // many processes may hold many blocked lanes while other lanes finish jobs.
            let blocked: Vec<Arc<Condvar>> = (state.blocked.iter())
                .filter_map(|k| state.lanes.get(k))
                .filter(|lane| lane.jobs.front().is_none_or(|(job, _)| state.ready(job)))
                .map(|lane| Arc::clone(&lane.wake))
                .collect();
            #[cfg(test)]
            {
                state.woken += blocked.len() as u64;
            }
            (blocked, done)
        };
        for wake in blocked {
            wake.notify_one();
        }
        self.moved.notify_all();
        done
    }

    /// Ring 0's thread `index`, with no lane to serve, waits for one, the threads that went idle
    /// after it given one first ([`Self::hand`]): the lane it is given, or `None` once it has
    /// waited [`KEEP`] for one, or the device ended; it then goes.
    pub(super) fn park(&self, index: usize) -> Option<Key> {
        let mut state = self.state.lock();
        let wake = (state.threads.get(index))
            .and_then(Option::as_ref)
            .map(|thread| Arc::clone(&thread.wake))?;
        let given = |state: &mut State| {
            (state.threads.get_mut(index))
                .and_then(Option::as_mut)
                .and_then(|thread| thread.given.take())
        };
        // A thread that left its lane without a look at it ([`Self::next`]) waits from here.
        let has_given = (state.threads.get(index))
            .and_then(Option::as_ref)
            .is_some_and(|thread| thread.given.is_some());
        if !has_given && !state.spares.contains(&index) {
            self.spare(&mut state, index);
        }
        let since = (state.threads.get(index))
            .and_then(Option::as_ref)
            .map_or(0, |thread| thread.since);
        let mut kept = false;
        loop {
            if let Some(key) = given(&mut state) {
                return Some(key);
            }
            if state.ended || kept {
                if let Some(at) = state.spares.iter().rposition(|&i| i == index) {
                    state.spares.remove(at);
                }
                if let Some(slot) = state.threads.get_mut(index) {
                    *slot = None;
                }
                #[cfg(test)]
                {
                    state.gone += 1;
                }
                return None;
            }
            kept = self.keep.wait(&wake, &mut state, since);
        }
    }

    /// Ring 0's thread `index` joins the spares, the last to go idle, from now.
    fn spare(&self, state: &mut State, index: usize) {
        if let Some(thread) = state.threads.get_mut(index).and_then(Option::as_mut) {
            thread.since = self.keep.now();
            state.spares.push(index);
        }
    }

    /// Whether every message of context `context`'s handed through ring 0's count `through` is
    /// carried out: a command buffer of the context waits for it, once the control thread has
    /// handed out every message it waits for.
    pub(super) fn through(&self, context: u32, through: u64) -> bool {
        through <= self.state.lock().context_done(context)
    }

    /// [`Self::through`], waiting for nothing: `None` where the lanes' lock is held elsewhere.
    pub(super) fn try_through(&self, context: u32, through: u64) -> Option<bool> {
        let state = self.state.try_lock()?;
        Some(through <= state.context_done(context))
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
        state.pending.clear();
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

    /// The device ended: the lanes' threads go at their next look, and so do those that wait for
    /// a lane.
    pub(super) fn end(&self) {
        let wakes = {
            let mut state = self.state.lock();
            state.ended = true;
            let mut wakes = state.wakes();
            wakes.extend((state.threads.iter().flatten()).map(|thread| Arc::clone(&thread.wake)));
            wakes
        };
        for wake in wakes {
            wake.notify_one();
        }
        self.moved.notify_all();
    }

    /// Whether the lane `key` has a thread (the tests).
    #[cfg(test)]
    pub(super) fn served(&self, key: Key) -> bool {
        self.state
            .lock()
            .lanes
            .get(&key)
            .is_some_and(|lane| lane.served)
    }

    /// Whether the lane `key`'s thread waits for other lanes' jobs, and how many times a blocked
    /// lane has been woken (the tests).
    #[cfg(test)]
    pub(super) fn blocked(&self, key: Key) -> (bool, u64) {
        let state = self.state.lock();
        (state.blocked.contains(&key), state.woken)
    }

    /// How many of ring 0's threads there are, how many of them wait for a lane, and how many
    /// were made and went (the tests).
    #[cfg(test)]
    pub(super) fn threads(&self) -> Ring0Threads {
        let state = self.state.lock();
        Ring0Threads {
            live: state.threads.iter().flatten().count(),
            spares: state.spares.len(),
            made: state.made,
            gone: state.gone,
        }
    }

    /// Whether no lane has a job not yet carried out (the tests).
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

/// Ring 0's threads, as [`Lanes::threads`] counts them (the tests).
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Ring0Threads {
    /// Those running.
    pub(super) live: usize,
    /// Those that wait for a lane.
    pub(super) spares: usize,
    /// Those made and gone, since the device was made.
    pub(super) made: u64,
    pub(super) gone: u64,
}

/// The 8 bytes at offset 16 of `message`, where it holds them.
fn field(message: &Message) -> Option<u64> {
    (message.bytes.get(16..24))
        .and_then(|bytes| bytes.try_into().ok())
        .map(u64::from_le_bytes)
}

/// The block `message` names by offset, for the kinds a block's lane orders (the module doc):
/// `SURFACE_PAGES`, `SURFACE_DOWNLOAD`, `BUFFER_OVER_PAGES`, `HEAP_OVER_PAGES`,
/// `BUFFER_OVER_SURFACE_PAGES` and `BLOCK_RELEASE`, where it is long enough to.
fn block_of(message: &Message) -> Option<u64> {
    match message.header.r#type {
        RGFX_CMD_SURFACE_PAGES => read::<RgfxSurfacePages>(message).map(|m| m.block_offset),
        RGFX_CMD_SURFACE_DOWNLOAD => read::<RgfxSurfaceDownload>(message).map(|m| m.block_offset),
        RGFX_CMD_BUFFER_OVER_PAGES => read::<RgfxBufferOverPages>(message).map(|m| m.block_offset),
        RGFX_CMD_HEAP_OVER_PAGES => read::<RgfxHeapOverPages>(message).map(|m| m.block_offset),
        RGFX_CMD_BUFFER_OVER_SURFACE_PAGES => {
            read::<RgfxBufferOverSurfacePages>(message).map(|m| m.block_offset)
        }
        RGFX_CMD_BLOCK_RELEASE => read::<RgfxBlockRelease>(message).map(|m| m.offset),
        _ => None,
    }
}

/// `message`'s bytes as `T`, where they are as many.
fn read<T: FromBytes>(message: &Message) -> Option<T> {
    T::read_from_prefix(&message.bytes).ok().map(|(t, _)| t)
}
