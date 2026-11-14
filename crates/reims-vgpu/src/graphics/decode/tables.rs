//! Each context's objects (docs/graphics-kext/design.md, "The host's decoder"): the table of a
//! process's objects, by id, with what each lies over, and the views through which a message
//! names them. No lock of the device's is held while a message of one context is carried out, so
//! a command buffer's walk and commit never hold up another context's, nor ring 0.
//!
//! **What a message sees.** A message takes a view of its context's table ([`View`]) when it first
//! names an object, and a command buffer takes one as it is begun: it then sees the table as it
//! was when the view was taken, and the changes the message itself made. A create, a destroy or a
//! close carried out meanwhile on another thread (ring 0's, or a compile's) is not seen, so every
//! outcome is the one the message would have had had it been carried out whole at the view's
//! taking, with nothing between, as under one lock: an object destroyed before the view is not
//! named (a command buffer that names it fails at the op that names it first), one destroyed after
//! it is named whole, and one made after it is not named. A command buffer's view is taken after
//! ring 0 has carried out every message published before the buffer was taken off its ring
//! (`reims_vgpu_graphics`'s order), so it sees every create its process posted before it.
//!
//! **What an entry is.** An object's handle, the region span it lies over, the CPU copy of a
//! `Managed` one, where a texture over an IOSurface plane lies in its buffer, and a queue's counts:
//! all set when it is made, and never changed while it is in the table. Each entry is boxed, so
//! it stays where it is while the table changes, and a view lends it out without a count taken or
//! a lock held.
//!
//! **When an entry goes.** An entry removed (a destroy, an id made again, the context's close or
//! the device's reset) while a view that names it is open is kept on the table's gone list, where
//! only such views find it, and let go of when the last of them closes, on that view's thread,
//! with the others it frees in the order they were removed: as an object destroyed after a whole
//! message under one lock is let go of after it. A view names an entry where it was taken at or
//! after the entry's make, or by the message that made it ([`View::made`]), and before its removal.
//! Else it is let go of at once, by whoever removed it, as one is that only a view taken before its
//! make saw nothing of: an older view open meanwhile (a message that waits) keeps only what it
//! names, so the list holds no more than the open views name. Each release is made in a refusal
//! scope ([`super::drop_refused`]).

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use parking_lot::RwLock;
use reims_vgpu_vulkan::core::collections::hasher::KeyBuild;
use reims_vgpu_vulkan::core::host::HostPtr;
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::sync::Mutex;

use super::lane::Lane;
use super::managed::ManagedCopy;
use super::surface_cache::Repitched;
use super::video::PlanePlace;
use super::{Drain, ObjHandle, drop_refused};

/// A command queue's counts of the host's command buffers made on it ([`super::cmdbuf`]), and its
/// lane ([`super::lane`]), which closes as the queue's entry goes.
#[derive(Default)]
pub(crate) struct QueueCounts {
    /// How many of its command buffers that wait for an event are begun and not yet scheduled. A
    /// buffer committed behind one is not submitted until it is, as Metal's is not.
    pub(crate) held: Arc<AtomicUsize>,
    /// How many command buffers the host made on it have not completed (a take on a doorbell's
    /// thread).
    pub(crate) in_flight: Arc<AtomicUsize>,
    /// Its marks in the surface order, which a command buffer that holds no op reads and writes
    /// with no lock ([`super::surface_order::Marks`]).
    pub(crate) order: Arc<super::surface_order::Marks>,
    /// Its committed command buffers' records the host may post ahead of the platform's
    /// submission, in commit order ([`super::cmdbuf::Line`]).
    pub(crate) line: Arc<super::cmdbuf::Line>,
    /// Where its command buffers' walks and commits run, each in its turn; `None` for a queue
    /// whose buffers are carried out whole where they are taken (the decoder's own tests).
    pub(crate) lane: Option<Arc<Lane>>,
    /// The command buffer at the head of its ring answered on a doorbell's thread, until its
    /// begin takes what the answer took for it ([`super::cmdbuf::Answered`]); where it is never
    /// begun, let go of as its context's close takes the entry out
    /// ([`super::GraphicsDecoder::close_tables`]), else with the queue's entry.
    pub(crate) answered: Mutex<Option<super::cmdbuf::Answered>>,
}

impl QueueCounts {
    /// A queue's counts, its command buffers carried out on `lane`, its marks in the surface order
    /// `order`.
    pub(crate) fn on(lane: Arc<Lane>, order: Arc<super::surface_order::Marks>) -> QueueCounts {
        QueueCounts {
            held: Arc::default(),
            in_flight: Arc::default(),
            order,
            line: Arc::default(),
            lane: Some(lane),
            answered: Mutex::default(),
        }
    }
}

impl Drop for QueueCounts {
    fn drop(&mut self) {
        if let Some(lane) = &self.lane {
            lane.close();
        }
    }
}

/// Where a buffer over the guest's memory has its contents, as the host reaches them: `length`
/// bytes from `at`, past `ram` where it has one (a byte of the guest's RAM mapping, which lives as
/// long as the device, or of a pages block's view, which the buffer keeps mapped), else at region
/// offset `at`. What a `Managed` texture's CPU copy is read through ([`super::managed`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GuestBytes {
    /// The first byte `at` counts from; `None` for the region's.
    pub(crate) ram: Option<HostPtr>,
    /// Where the contents start, from `ram`'s byte or in the region.
    pub(crate) at: u64,
    /// The contents' length.
    pub(crate) length: u64,
}

/// One object of a context's, as made. Its fields drop in their order: a `Managed` object's CPU
/// copy before the object.
pub(crate) struct Entry<P: Platform> {
    /// The CPU copy of a `Managed` buffer or texture ([`super::managed`]).
    pub(crate) managed: Option<ManagedCopy<P>>,
    /// The object.
    pub(crate) handle: ObjHandle<P>,
    /// The region span (offset, length) it lies over, for an object over the region's memory: a
    /// buffer or heap over host memory, and what is placed in one (a buffer at an offset, a texture
    /// in the heap or over the buffer), with its parent's span. A block is not given out again
    /// while an object lies in it.
    pub(crate) span: Option<(u64, u64)>,
    /// Where a texture over an IOSurface plane lies in its buffer: what a video encoder's picture
    /// is written through.
    pub(crate) place: Option<PlanePlace>,
    /// A command queue's counts.
    pub(crate) queue: Option<QueueCounts>,
    /// A heap over a pages block's pages (`HEAP_OVER_PAGES`): the host bytes it lies in, in the
    /// guest's RAM or the block's view, its first byte and length, which `span`, the block's, does
    /// not name.
    pub(crate) ram: Option<(HostPtr, u64)>,
    /// A buffer whose contents are the guest's bytes: where they lie, which `span` names only for
    /// a buffer over its own block (`BUFFER_OVER_HOST_MEMORY`); `None` for any other object.
    pub(crate) bytes: Option<GuestBytes>,
    /// A buffer over a block's surface view (`BUFFER_OVER_SURFACE_PAGES`), whose drop may be the
    /// last hold of the view's mapping and lets go of an import, and gives back nothing the guest
    /// picked (the host places it): it is let go of on the surface worker
    /// ([`super::surface_pages`]).
    pub(crate) over_view: bool,
    /// An indirect command buffer's commands whose draws and dispatches are dropped
    /// ([`super::icb::Dropped`]); `None` for any other object.
    pub(crate) icb_dropped: Option<Box<super::icb::Dropped>>,
    /// A buffer over a block's device cache that lays its surface's planes out at the GPU's
    /// pitch: that layout, by which each texture over the buffer is placed in it
    /// ([`Repitched`]); `None` for any other object.
    pub(crate) laid_out: Option<Arc<Repitched>>,
    /// A texture over a plane of a device cache laid out at the GPU's pitch: its block, and
    /// whether it lies in the cache or in the block's wide home ([`LaidUse`]); `None` for any
    /// other object.
    pub(crate) laid_use: Option<LaidUse>,
    /// The table's clock at its make.
    made: u64,
    /// The first clock of a view that names it: its make's, or the making message's view's where
    /// that was taken before ([`Context::insert`]'s `seen_from`).
    seen: u64,
}

impl<P: Platform> Entry<P> {
    /// An entry of `handle` alone.
    pub(crate) fn of(handle: ObjHandle<P>) -> Self {
        Entry {
            managed: None,
            handle,
            span: None,
            place: None,
            queue: None,
            ram: None,
            bytes: None,
            over_view: false,
            icb_dropped: None,
            laid_out: None,
            laid_use: None,
            made: 0,
            seen: 0,
        }
    }

    /// The entry, lying over `span`.
    pub(crate) fn over(mut self, span: Option<(u64, u64)>) -> Self {
        self.span = span;
        self
    }

    /// The entry, a buffer over a block's surface view (the field `over_view`).
    pub(crate) fn over_view(mut self) -> Self {
        self.over_view = true;
        self
    }

    /// The entry, a buffer over a device cache laid out as `laid_out` says (the field
    /// `laid_out`).
    pub(crate) fn laid_out(mut self, laid_out: Option<Arc<Repitched>>) -> Self {
        self.laid_out = laid_out;
        self
    }

    /// The entry, a buffer whose contents are the guest's `bytes` (the field `bytes`).
    pub(crate) fn with_bytes(mut self, bytes: Option<GuestBytes>) -> Self {
        self.bytes = bytes;
        self
    }
}

/// A texture over a plane of a surface whose device cache lays its planes out at the GPU's pitch
/// ([`Repitched`]), as a command buffer's walk notes each use of it
/// ([`super::surface_cache::Caches::reconcile`]): the block, whether the texture lies in the
/// block's wide home rather than in the cache (`super::surface_cache`, "Wider than the plane"), and
/// whether the use may write it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LaidUse {
    pub(crate) block: u64,
    pub(crate) wide: bool,
    pub(crate) writes: bool,
}

/// An entry the table owns: boxed, and freed only by its drop, so a view's pointer to it stays
/// good while the view may find it (the module doc).
struct Owned<P: Platform>(NonNull<Entry<P>>);

// SAFETY: an `Owned` is a `Box<Entry>`, which is `Send` and `Sync` where the entry is; the entry's
// handles are the platform's (`Handle: Send + Sync`), its counts atomics.
unsafe impl<P: Platform> Send for Owned<P> where Entry<P>: Send + Sync {}
// SAFETY: as for `Send`; a shared `Owned` lends only `&Entry`.
unsafe impl<P: Platform> Sync for Owned<P> where Entry<P>: Send + Sync {}

impl<P: Platform> Owned<P> {
    fn new(entry: Entry<P>) -> Self {
        Owned(NonNull::from(Box::leak(Box::new(entry))))
    }

    fn entry(&self) -> &Entry<P> {
        // SAFETY: the box lives until this `Owned` drops.
        unsafe { self.0.as_ref() }
    }

    /// Whether the entry is an object the host made, not an id that stands for a refused create
    /// (for the tests).
    #[cfg(any(test, feature = "testkit"))]
    fn made(&self) -> bool {
        !matches!(self.entry().handle, ObjHandle::Refused(_))
    }

    /// The entry, out of its box.
    fn into_entry(self) -> Entry<P> {
        let this = std::mem::ManuallyDrop::new(self);
        // SAFETY: the pointer came from `Box::leak`, and `this` is never dropped, so the box is
        // taken back once.
        *unsafe { Box::from_raw(this.0.as_ptr()) }
    }
}

impl<P: Platform> Drop for Owned<P> {
    fn drop(&mut self) {
        // SAFETY: the pointer came from `Box::leak`, and is freed here once.
        drop(unsafe { Box::from_raw(self.0.as_ptr()) });
    }
}

/// An entry removed from the table, owned by whoever lets it go.
pub(crate) struct Removed<P: Platform> {
    id: u64,
    owned: Owned<P>,
}

impl<P: Platform> Removed<P> {
    /// Its object's id.
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// The entry.
    pub(crate) fn entry(&self) -> &Entry<P> {
        self.owned.entry()
    }

    /// The entry, to let go of in parts.
    pub(crate) fn into_entry(self) -> Entry<P> {
        self.owned.into_entry()
    }
}

/// Lets go of `removed`, in order: each one's CPU copy and then its object, each in a refusal scope
/// of its own.
pub(crate) fn let_go<P: Platform>(removed: Vec<Removed<P>>) {
    for r in removed {
        let id = r.id;
        let Entry {
            managed, handle, ..
        } = r.into_entry();
        if let Some(copy) = managed {
            drop_refused(id, copy);
        }
        drop_refused(id, handle);
    }
}

/// An entry stored: where it lies, and what it replaced, to let go of where no view needs it.
pub(crate) type Made<P> = (NonNull<Entry<P>>, Option<Removed<P>>);

/// What a view found, and the message's own changes (`None`: removed by it), by id.
type Seen<P> = HashMap<u64, Option<NonNull<Entry<P>>>, KeyBuild>;

/// A context's table, under its lock.
struct Tables<P: Platform> {
    /// The objects, by id: hashed by core's fixed integer hasher, so the order a context's objects
    /// drop in at its close is the same in every run.
    live: HashMap<u64, Owned<P>, KeyBuild>,
    /// The entries removed while a view taken before the removal was open, in the order removed,
    /// each with the clock at its removal.
    gone: Vec<(Removed<P>, u64)>,
    /// Moves at each make and removal.
    clock: u64,
    /// The open views, by the clock at which each was taken: how many at each.
    views: BTreeMap<u64, usize>,
    /// The region spans the live entries lie over ([`Entry::span`]), each with how many lie over
    /// it: what [`Context::span_in`] looks through, a few keys about the range it asks of, where
    /// the entries themselves are thousands. It cannot go stale: an entry's span is counted where
    /// the entry goes into `live` ([`Context::insert`], [`Context::insert_if`]) and taken out where
    /// it leaves ([`Self::retire`], which every removal from `live` goes through: a remove, a make
    /// over its id again, a close's removal of every entry), each under the table's write lock with
    /// the change itself, and a span is set at the entry's make and never changed while it is in
    /// the table (the module doc). A context dropped takes its spans with it.
    spans: BTreeMap<(u64, u64), usize>,
    /// The longest span `spans` ever counted: how far before a range a span that reaches into it
    /// may start.
    longest: u64,
}

impl<P: Platform> Tables<P> {
    /// Counts `span`, a live entry's, in [`Self::spans`].
    fn count_span(&mut self, span: Option<(u64, u64)>) {
        if let Some(span) = span {
            *self.spans.entry(span).or_default() += 1;
            self.longest = self.longest.max(span.1);
        }
    }

    /// Takes `span`, an entry's that leaves the live ones, out of [`Self::spans`].
    fn uncount_span(&mut self, span: Option<(u64, u64)>) {
        if let Some(span) = span
            && let Some(n) = self.spans.get_mut(&span)
        {
            *n -= 1;
            if *n == 0 {
                self.spans.remove(&span);
            }
        }
    }

    /// The entry of `id` a view taken at `at` sees: in the table and made by then, else removed
    /// since and made by then.
    fn find(&self, id: u64, at: u64) -> Option<NonNull<Entry<P>>> {
        if let Some(owned) = self.live.get(&id).filter(|o| o.entry().made <= at) {
            return Some(owned.0);
        }
        self.gone.iter().rev().find_map(|(r, removed_at)| {
            (r.id == id && r.owned.entry().made <= at && at < *removed_at).then_some(r.owned.0)
        })
    }

    /// Removes `owned`, `id`'s entry: onto the gone list where an open view names it (each open
    /// view was taken before this), else handed back to let go of.
    fn retire(&mut self, id: u64, owned: Owned<P>) -> Option<Removed<P>> {
        self.clock += 1;
        self.uncount_span(owned.entry().span);
        let removed = Removed { id, owned };
        if self
            .views
            .range(removed.owned.entry().seen..)
            .next()
            .is_none()
        {
            return Some(removed);
        }
        self.gone.push((removed, self.clock));
        None
    }

    /// The gone entries no open view names any more, in the order removed: a view names one where
    /// it was taken from the entry's `seen` up to its removal.
    fn sweep(&mut self) -> Vec<Removed<P>> {
        if self.gone.is_empty() {
            return Vec::new();
        }
        if self.views.is_empty() {
            return std::mem::take(&mut self.gone)
                .into_iter()
                .map(|(r, _)| r)
                .collect();
        }
        let views = &self.views;
        let (keep, out): (Vec<_>, Vec<_>) =
            std::mem::take(&mut self.gone)
                .into_iter()
                .partition(|(r, removed_at)| {
                    views
                        .range(r.owned.entry().seen..*removed_at)
                        .next()
                        .is_some()
                });
        self.gone = keep;
        out.into_iter().map(|(r, _)| r).collect()
    }
}

/// One context's state: its objects, and the drain its command buffers join.
pub(crate) struct Context<P: Platform> {
    /// The context's id.
    pub(crate) id: u32,
    tables: RwLock<Tables<P>>,
    /// The command buffers committed since the context's last drain began: each holds a clone
    /// until its completion handler has run ([`super::GraphicsDecoder::drain_context`]).
    pub(crate) drain: Mutex<Arc<Drain>>,
    /// The device's recoveries its two-copy `Managed` textures were last looked at for
    /// (`GraphicsDecoder::refill_managed`): a command buffer's begin past one refills those made
    /// before it, under this lock, so no command buffer of the context is begun between.
    pub(crate) managed_recoveries: Mutex<u64>,
}

impl<P: Platform> Context<P> {
    /// A context with no objects.
    pub(crate) fn new(id: u32) -> Self {
        Context {
            id,
            tables: RwLock::new(Tables {
                live: HashMap::default(),
                gone: Vec::new(),
                clock: 0,
                views: BTreeMap::new(),
                spans: BTreeMap::new(),
                longest: 0,
            }),
            drain: Mutex::new(Arc::default()),
            managed_recoveries: Mutex::new(0),
        }
    }

    /// Stores `entry` under `id`, in place of what the id named, and the pointer to it; what it
    /// replaced, to let go of where no view needs it. `seen_from` is the clock of the making
    /// message's view of this table, where it has one ([`View::at`]), which names the entry
    /// though taken before its make ([`View::made`]).
    pub(crate) fn insert(&self, id: u64, mut entry: Entry<P>, seen_from: Option<u64>) -> Made<P> {
        let mut t = self.tables.write();
        t.clock += 1;
        entry.made = t.clock;
        entry.seen = seen_from.map_or(t.clock, |at| at.min(t.clock));
        t.count_span(entry.span);
        let owned = Owned::new(entry);
        let at = owned.0;
        let replaced = t.live.insert(id, owned).and_then(|old| t.retire(id, old));
        (at, replaced)
    }

    /// [`Self::insert`], where `live` answers, under the table's lock, that the entry is still to
    /// be stored; `None` where it is not.
    pub(crate) fn insert_if(
        &self,
        id: u64,
        mut entry: Entry<P>,
        seen_from: Option<u64>,
        live: impl FnOnce() -> bool,
    ) -> Option<Made<P>> {
        let mut t = self.tables.write();
        if !live() {
            return None;
        }
        t.clock += 1;
        entry.made = t.clock;
        entry.seen = seen_from.map_or(t.clock, |at| at.min(t.clock));
        t.count_span(entry.span);
        let owned = Owned::new(entry);
        let at = owned.0;
        let replaced = t.live.insert(id, owned).and_then(|old| t.retire(id, old));
        Some((at, replaced))
    }

    /// Removes `id`'s entry, where it has one: whether it was a texture, and it, to let go of,
    /// where no view needs it.
    pub(crate) fn remove(&self, id: u64) -> Option<(bool, Option<Removed<P>>)> {
        let mut t = self.tables.write();
        let owned = t.live.remove(&id)?;
        let texture = matches!(owned.entry().handle, ObjHandle::Texture(_));
        Some((texture, t.retire(id, owned)))
    }

    /// Removes every entry, in the table's order: those to let go of now, where no view needs
    /// them, and how many there were.
    pub(crate) fn remove_all(&self) -> (Vec<Removed<P>>, usize) {
        let mut t = self.tables.write();
        let ids: Vec<u64> = t.live.keys().copied().collect();
        let count = ids.len();
        let mut out = Vec::new();
        for id in ids {
            if let Some(owned) = t.live.remove(&id)
                && let Some(removed) = t.retire(id, owned)
            {
                out.push(removed);
            }
        }
        (out, count)
    }

    /// Whether any entry in the table lies in the `size` bytes at region offset `offset`: a span
    /// the live entries lie over that starts before the range ends and ends past its start, looked
    /// for among those that start no further before it than the longest span counted.
    pub(crate) fn span_in(&self, offset: u64, size: u64) -> bool {
        let end = offset.saturating_add(size);
        let t = self.tables.read();
        let from = offset.saturating_sub(t.longest);
        (t.spans.range((from, 0)..(end, 0)))
            .any(|(&(at, length), _)| offset < at.saturating_add(length))
    }

    /// Calls `f` with each entry the table holds now, with its id, under its read lock: what a
    /// close does with every object while a view open on the table still keeps what
    /// [`Self::remove_all`] retires out of its answer.
    pub(crate) fn for_each_live(&self, mut f: impl FnMut(u64, &Entry<P>)) {
        let t = self.tables.read();
        for (&id, owned) in &t.live {
            f(id, owned.entry());
        }
    }

    /// What `f` makes of `id`'s entry as the table holds it now, under its read lock.
    pub(crate) fn with_live<R>(&self, id: u64, f: impl FnOnce(&Entry<P>) -> R) -> Option<R> {
        self.tables.read().live.get(&id).map(|o| f(o.entry()))
    }

    /// [`Self::with_live`], waiting for nothing and allocating nothing (a doorbell's thread): no
    /// view is taken. `None` where the table is busy or holds no entry of `id`.
    pub(crate) fn try_with_live<R>(&self, id: u64, f: impl FnOnce(&Entry<P>) -> R) -> Option<R> {
        self.tables.try_read()?.live.get(&id).map(|o| f(o.entry()))
    }

    /// The ids of the objects in the table, not those that stand for a refused create (for the
    /// tests).
    #[cfg(test)]
    pub(crate) fn ids(&self) -> Vec<u64> {
        let t = self.tables.read();
        let made = t.live.iter().filter(|(_, o)| o.made());
        made.map(|(&id, _)| id).collect()
    }

    /// The ids that stand for a refused create ([`super::refused`]; for the tests).
    #[cfg(test)]
    pub(crate) fn refused_ids(&self) -> Vec<u64> {
        let t = self.tables.read();
        let refused = t.live.iter().filter(|(_, o)| !o.made());
        refused.map(|(&id, _)| id).collect()
    }

    /// How many objects the table holds, not counting the ids that stand for a refused create
    /// (for the tests).
    #[cfg(any(test, feature = "testkit"))]
    pub(crate) fn len(&self) -> usize {
        self.tables
            .read()
            .live
            .values()
            .filter(|o| o.made())
            .count()
    }

    /// How many removed entries open views keep (for the tests).
    #[cfg(test)]
    pub(crate) fn gone_len(&self) -> usize {
        self.tables.read().gone.len()
    }
}

impl<P: Platform> Drop for Context<P> {
    /// Every entry left is let go of in a refusal scope.
    fn drop(&mut self) {
        let t = self.tables.get_mut();
        let mut out: Vec<Removed<P>> = t
            .live
            .drain()
            .map(|(id, owned)| Removed { id, owned })
            .collect();
        out.extend(std::mem::take(&mut t.gone).into_iter().map(|(r, _)| r));
        let_go(out);
    }
}

/// The names a view's map of what it found holds before it grows: a command buffer of a few
/// dispatches names tens of objects, and each growth moves every name found so far.
const SEEN_ROOM: usize = 32;

/// A message's view of one context's table (the module doc): open from its taking until it drops,
/// which lets go of the removed entries only it still kept. It lends the entries it finds for its
/// own life, and keeps each found, and each change of the message's own, so a name it looks up
/// again is found without the table's lock.
pub(crate) struct View<P: Platform> {
    context: Arc<Context<P>>,
    /// The table's clock at the view's taking.
    at: u64,
    /// The entries found, and the message's own changes (`None`: removed by it), by id.
    seen: RefCell<Seen<P>>,
}

// SAFETY: a view is used on one thread at a time (it is not `Sync`); moved to another, it moves its
// pointers, to entries that are `Send + Sync`, and its context, which is.
unsafe impl<P: Platform> Send for View<P> where Entry<P>: Send + Sync {}

impl<P: Platform> View<P> {
    /// A view of `context`'s table, taken now.
    pub(crate) fn take(context: Arc<Context<P>>) -> Self {
        let at = {
            let mut t = context.tables.write();
            let at = t.clock;
            *t.views.entry(at).or_default() += 1;
            at
        };
        View {
            context,
            at,
            seen: RefCell::new(HashMap::with_capacity_and_hasher(
                SEEN_ROOM,
                KeyBuild::default(),
            )),
        }
    }

    /// The table's clock at the view's taking.
    pub(crate) fn at(&self) -> u64 {
        self.at
    }

    /// The context viewed.
    pub(crate) fn context(&self) -> &Arc<Context<P>> {
        &self.context
    }

    /// The entry of `id` the view sees: looked up in what the view found and changed with one
    /// hash of `id`, which the table's answer then goes in at.
    pub(crate) fn get(&self, id: u64) -> Option<&Entry<P>> {
        let found = match self.seen.borrow_mut().entry(id) {
            std::collections::hash_map::Entry::Occupied(seen) => *seen.get(),
            std::collections::hash_map::Entry::Vacant(slot) => {
                let found = self.context.tables.read().find(id, self.at)?;
                *slot.insert(Some(found))
            }
        };
        // SAFETY: the entry was in the table, made by the view's taking, or removed after it; or
        // it is the message's own make, whose `seen` is this view's clock or earlier (the
        // decoder's `view_at`), removed (if at all) after this view's taking. Either way this view
        // names it, so it is freed only once no such view is open, and this one is until it
        // drops: it outlives `&self`.
        found.map(|p| unsafe { p.as_ref() })
    }

    /// The message made `id`'s entry at `at`: the view sees it from now on.
    pub(crate) fn made(&self, id: u64, at: NonNull<Entry<P>>) {
        self.seen.borrow_mut().insert(id, Some(at));
    }

    /// The message removed `id`'s entry: the view no longer sees it.
    pub(crate) fn removed(&self, id: u64) {
        self.seen.borrow_mut().insert(id, None);
    }
}

impl<P: Platform> Drop for View<P> {
    fn drop(&mut self) {
        let out = {
            let mut t = self.context.tables.write();
            if let Some(n) = t.views.get_mut(&self.at) {
                *n -= 1;
                if *n == 0 {
                    t.views.remove(&self.at);
                }
            }
            t.sweep()
        };
        let_go(out);
    }
}

#[cfg(test)]
mod tests {
    use reims_vgpu_vulkan::core::testkit::FakePlatform;

    use super::*;

    fn fence(p: &FakePlatform) -> ObjHandle<FakePlatform> {
        use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
        let device = p.device();
        ObjHandle::Fence(device.new_fence().unwrap().unwrap())
    }

    /// A view sees the table as of its taking: an entry removed after it is still found, one made
    /// after it is not, and the removed one goes once the view closes.
    #[test]
    fn a_view_sees_the_table_as_of_its_taking() {
        let p = FakePlatform::new();
        let ctx = Arc::new(Context::<FakePlatform>::new(1));
        let (_, replaced) = ctx.insert(1, Entry::of(fence(&p)), None);
        assert!(replaced.is_none());
        let view = View::take(Arc::clone(&ctx));
        assert!(
            ctx.remove(1).is_some_and(|(_, now)| now.is_none()),
            "kept for the open view"
        );
        let (_, replaced) = ctx.insert(2, Entry::of(fence(&p)), None);
        assert!(replaced.is_none());
        assert!(view.get(1).is_some(), "removed after the view: found");
        assert!(view.get(2).is_none(), "made after the view: not found");
        assert_eq!(ctx.gone_len(), 1);
        drop(view);
        assert_eq!(
            ctx.gone_len(),
            0,
            "let go of with the last view that kept it"
        );
        let late = View::take(Arc::clone(&ctx));
        assert!(late.get(1).is_none());
        assert!(late.get(2).is_some());
    }

    /// One view taken long ago and still open (a message that waits) while frames of messages
    /// each take a view, remove the objects the frame before made and make as many: the table
    /// keeps only what an open view can name, the removed objects the old view saw, and every
    /// frame's removals go with the frame's view.
    #[test]
    fn an_old_view_keeps_only_what_it_can_name() {
        let p = FakePlatform::new();
        let ctx = Arc::new(Context::<FakePlatform>::new(1));
        for id in 0..4 {
            ctx.insert(id, Entry::of(fence(&p)), None);
        }
        let old = View::take(Arc::clone(&ctx));
        // One object the old view saw, removed while it is open: kept for it.
        assert!(ctx.remove(0).is_some_and(|(_, now)| now.is_none()));
        let mut next = 100;
        for _ in 0..1000 {
            let frame = View::take(Arc::clone(&ctx));
            for id in next - 4..next {
                ctx.remove(id);
            }
            for id in next..next + 4 {
                ctx.insert(id, Entry::of(fence(&p)), None);
            }
            next += 4;
            drop(frame);
        }
        assert!(
            old.get(0).is_some(),
            "removed after the old view: still found"
        );
        assert!(old.get(1).is_some(), "live, made before the old view");
        assert!(
            old.get(next - 1).is_none(),
            "made after the old view: not found"
        );
        assert_eq!(
            ctx.gone_len(),
            1,
            "only the removed object the old view can name is kept"
        );
        drop(old);
        assert_eq!(ctx.gone_len(), 0);
    }

    /// An object a message made, while its view of the table is open, is named by that view
    /// though it was made after the view's taking: removed by another message meanwhile, it is
    /// kept until the maker's view closes.
    #[test]
    fn a_message_s_own_make_stays_for_its_view() {
        let p = FakePlatform::new();
        let ctx = Arc::new(Context::<FakePlatform>::new(1));
        let maker = View::take(Arc::clone(&ctx));
        let (at, _) = ctx.insert(7, Entry::of(fence(&p)), Some(maker.at()));
        maker.made(7, at);
        let other = View::take(Arc::clone(&ctx));
        assert!(ctx.remove(7).is_some_and(|(_, now)| now.is_none()));
        drop(other);
        assert_eq!(ctx.gone_len(), 1, "kept: the maker's view names it");
        assert!(maker.get(7).is_some());
        drop(maker);
        assert_eq!(ctx.gone_len(), 0);
    }

    /// What frames of messages cost while one old view stays open, over time (a CPU bench, run by
    /// name with `--ignored`): each frame takes a view, removes the 8 objects the frame before
    /// made and makes 8, and drops its view; each 2,000 frames, how many removed entries the
    /// table keeps, a frame's cost, and a lookup's of an id the table never held (a miss looks
    /// through every kept entry).
    #[test]
    #[ignore = "a bench: run it by name"]
    #[allow(clippy::disallowed_methods)] // a measurement: the wall clock's time
    fn gone_cost_with_an_old_view_open() {
        let p = FakePlatform::new();
        let ctx = Arc::new(Context::<FakePlatform>::new(1));
        let old = View::take(Arc::clone(&ctx));
        let mut next = 1_000_000;
        for round in 1..=10 {
            let t = std::time::Instant::now();
            for _ in 0..2000 {
                let frame = View::take(Arc::clone(&ctx));
                for id in next - 8..next {
                    ctx.remove(id);
                }
                for id in next..next + 8 {
                    ctx.insert(id, Entry::of(fence(&p)), None);
                }
                next += 8;
                drop(frame);
            }
            let per_frame = t.elapsed() / 2000;
            let t = std::time::Instant::now();
            for i in 0..1000u64 {
                let miss = View::take(Arc::clone(&ctx));
                assert!(miss.get(5_000_000_000 + i).is_none());
            }
            let per_miss = t.elapsed() / 1000;
            eprintln!(
                "frames {:6}: gone {:6}, a frame {:9.1?}, a missed lookup in a new view {:9.1?}",
                round * 2000,
                ctx.gone_len(),
                per_frame,
                per_miss
            );
        }
        drop(old);
    }

    /// What a command buffer's view of its table costs, a command buffer (a measurement, not a
    /// check; run it by name with `--run-ignored only --no-capture` in a release build): Super
    /// Resolution's 6,847 command buffers of about 6 dispatches each, each binding buffers that
    /// lie in heaps. Over a table of `TABLES_COST_LIVE` objects (default 4,096), each of
    /// `TABLES_COST_CBS` command buffers (default 2,000) takes a view, names `TABLES_COST_NAMED`
    /// objects of the table (default 64, picked at random) for the first time, then
    /// `TABLES_COST_AGAIN` more times each (default 3: the names a dispatch's binds take again
    /// that the decoder's cache of recent entries misses), and drops the view. Prints the
    /// thread's user-space instructions and cycles a command buffer and a name, the median of 9
    /// runs.
    #[test]
    #[ignore = "a measurement: run it by name"]
    fn a_command_buffer_s_view_costs() {
        use crate::graphics::decode::draw_cost::{counts, spread};
        let env = |name: &str, default: u64| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        };
        let live = env("TABLES_COST_LIVE", 4096);
        let cbs = env("TABLES_COST_CBS", 2000);
        let named = env("TABLES_COST_NAMED", 64);
        let again = env("TABLES_COST_AGAIN", 3);
        let p = FakePlatform::new();
        let ctx = Arc::new(Context::<FakePlatform>::new(1));
        for id in 0..live {
            ctx.insert(0x1000 + id, Entry::of(fence(&p)), None);
        }
        let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = move |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        let ids: Vec<Vec<u64>> = (0..cbs)
            .map(|_| (0..named).map(|_| 0x1000 + next(live)).collect())
            .collect();
        let mut figures: [Vec<f64>; 2] = Default::default();
        for _ in 0..9 {
            let before = counts();
            for cb in &ids {
                let view = View::take(Arc::clone(&ctx));
                for _ in 0..=again {
                    for &id in cb {
                        assert!(view.get(std::hint::black_box(id)).is_some());
                    }
                }
                drop(view);
            }
            let spent = counts() - before;
            for (f, figure) in figures.iter_mut().enumerate() {
                figure.push(spent.0[f + 1] as f64 / cbs as f64);
            }
        }
        eprintln!(
            "{cbs} views over {live} objects, each naming {named} objects {} times: user-space \
             instructions and cycles a command buffer, then a name (median, lowest, highest of 9)",
            again + 1
        );
        for (figure, name) in figures.iter_mut().zip(["instructions", "cycles"]) {
            let [median, low, high] = spread(figure);
            let names = (named * (again + 1)) as f64;
            eprintln!(
                "  {name:<12} {median:9.0} ({low:.0} to {high:.0}) a command buffer, {:6.1} a \
                 name",
                median / names
            );
        }
    }

    /// Through random makes, makes over an id again, removes and a close's removal of every
    /// entry, with views open over some of them, whether an entry lies in a range answers what a
    /// look at every live entry's span answers: spans of blocks apart, of one block shared by the
    /// objects placed in it, of no bytes, and ranges of no bytes, over, across and beside them; and the index of spans holds the live entries' spans, recounted from them at each step
    #[test]
    fn span_in_answers_as_a_look_at_every_live_entry() {
        let p = FakePlatform::new();
        let ctx = Arc::new(Context::<FakePlatform>::new(1));
        let mut seed: u64 = 0x2545_f491_4f6c_dd1d;
        let mut next = move |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        let every = |ctx: &Context<FakePlatform>, offset: u64, size: u64| {
            let end = offset.saturating_add(size);
            let mut any = false;
            ctx.for_each_live(|_, e| {
                any |= e
                    .span
                    .is_some_and(|(at, length)| at < end && offset < at.saturating_add(length));
            });
            any
        };
        let mut views = Vec::new();
        for round in 0..3000 {
            let id = next(64);
            match next(8) {
                0..=3 => {
                    let span = match next(4) {
                        0 => None,
                        1 => Some((0x1_0000 + next(32) * 0x1000, 0)),
                        _ => Some((0x1_0000 + next(32) * 0x1000, (1 + next(4)) * 0x1000)),
                    };
                    let (_, replaced) = ctx.insert(id, Entry::of(fence(&p)).over(span), None);
                    let_go(replaced.into_iter().collect());
                }
                4 | 5 => {
                    if let Some((_, Some(removed))) = ctx.remove(id) {
                        let_go(vec![removed]);
                    }
                }
                6 => match views.len() {
                    0..4 => views.push(View::take(Arc::clone(&ctx))),
                    _ => views.clear(),
                },
                _ if round % 500 == 0 => let_go(ctx.remove_all().0),
                _ => {}
            }
            // The index holds the live entries' spans, each counted as many times as entries lie
            // over it.
            let mut counted: BTreeMap<(u64, u64), usize> = BTreeMap::new();
            ctx.for_each_live(|_, e| {
                if let Some(span) = e.span {
                    *counted.entry(span).or_default() += 1;
                }
            });
            assert_eq!(ctx.tables.read().spans, counted, "the index of spans");
            for _ in 0..4 {
                let (offset, size) = (0xf000 + next(40) * 0x800, next(4) * 0x800);
                assert_eq!(
                    ctx.span_in(offset, size),
                    every(&ctx, offset, size),
                    "{offset:#x}+{size:#x}"
                );
            }
        }
    }

    /// What a block's release's check that no object lies in it costs ([`Context::span_in`];
    /// a measurement, not a check; run it by name with `--run-ignored only --no-capture` in a
    /// release build): Particle Physics' 106 releases spent 2.31 us a dispatch there on gbp-iris,
    /// about 0.5 ms each. A table of `TABLES_COST_LIVE` objects (default 8,192), each over a
    /// block of its own of 64 KiB, one in four over the block of the object before (a buffer
    /// placed in a heap), and 2,000 checks of blocks among them and past them. Prints the
    /// thread's user-space instructions and cycles a check, the median of 9 runs.
    #[test]
    #[ignore = "a measurement: run it by name"]
    fn a_block_release_s_check_costs() {
        use crate::graphics::decode::draw_cost::{counts, spread};
        let live: u64 = std::env::var("TABLES_COST_LIVE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8192);
        let p = FakePlatform::new();
        let ctx = Arc::new(Context::<FakePlatform>::new(1));
        let block = |i: u64| (0x10_0000 + i * 0x1_0000, 0x1_0000);
        let mut at = 0;
        for id in 0..live {
            if id % 4 != 3 {
                at += 1;
            }
            ctx.insert(id, Entry::of(fence(&p)).over(Some(block(at))), None);
        }
        let checks: Vec<(u64, u64)> = (0..2000u64)
            .map(|i| block((i * 7919) % (at + at / 8)))
            .collect();
        let mut figures: [Vec<f64>; 2] = Default::default();
        let mut held = 0;
        for _ in 0..9 {
            let before = counts();
            for &(offset, size) in &checks {
                held += usize::from(ctx.span_in(std::hint::black_box(offset), size));
            }
            let spent = counts() - before;
            for (f, figure) in figures.iter_mut().enumerate() {
                figure.push(spent.0[f + 1] as f64 / checks.len() as f64);
            }
        }
        assert!(held > 0);
        eprintln!(
            "span_in over {live} objects: user-space instructions and cycles a check (median, \
             lowest, highest of 9)"
        );
        for (figure, name) in figures.iter_mut().zip(["instructions", "cycles"]) {
            let [median, low, high] = spread(figure);
            eprintln!("  {name:<12} {median:9.0} ({low:.0} to {high:.0})");
        }
    }

    /// An entry removed with no view open is handed back at once, and a view taken after a
    /// removal never keeps it.
    #[test]
    fn an_entry_no_view_needs_is_handed_back() {
        let p = FakePlatform::new();
        let ctx = Arc::new(Context::<FakePlatform>::new(1));
        ctx.insert(1, Entry::of(fence(&p)), None);
        assert!(ctx.remove(1).is_some_and(|(_, now)| now.is_some()));
        ctx.insert(1, Entry::of(fence(&p)), None);
        let older = View::take(Arc::clone(&ctx));
        assert!(ctx.remove(1).is_some_and(|(_, now)| now.is_none()));
        let newer = View::take(Arc::clone(&ctx));
        assert!(newer.get(1).is_none());
        drop(older);
        assert_eq!(
            ctx.gone_len(),
            0,
            "the newer view does not keep what went before it"
        );
        drop(newer);
    }
}
