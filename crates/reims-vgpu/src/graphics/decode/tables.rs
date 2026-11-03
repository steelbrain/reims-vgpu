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
//! (`reims_graphics`'s order), so it sees every create its process posted before it.
//!
//! **What an entry is.** An object's handle, the region span it lies over, the CPU copy of a
//! `Managed` one, where a texture over an IOSurface plane lies in its buffer, and a queue's counts:
//! all set when it is made, and never changed while it is in the table. Each entry is boxed, so
//! it stays where it is while the table changes, and a view lends it out without a count taken or
//! a lock held.
//!
//! **When an entry goes.** An entry removed (a destroy, an id made again, the context's close or
//! the device's reset) while a view taken before the removal is open is kept on the table's gone
//! list, where only such views find it, and let go of when the last of them closes, on that view's
//! thread, in the order the entries were removed: as an object destroyed after a whole message
//! under one lock is let go of after it. Else it is let go of at once, by whoever removed it. Each
//! release is made in a refusal scope ([`super::drop_refused`]).

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use parking_lot::RwLock;
use reims_vgpu_core::collections::hasher::KeyBuild;
use reims_vgpu_core::host::HostPtr;
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::sync::Mutex;

use super::lane::Lane;
use super::managed::ManagedCopy;
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
    /// begin takes what the answer took for it ([`super::cmdbuf::Answered`]); let go of with the
    /// queue's entry where it is never begun.
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
    /// A heap over a pages block's guest RAM (`HEAP_OVER_PAGES`): the host bytes it lies in, its
    /// first byte and length, which `span`, the block's, does not name.
    pub(crate) ram: Option<(HostPtr, u64)>,
    /// A buffer over a block's surface view (`BUFFER_OVER_SURFACE_PAGES`), whose drop may be the
    /// last hold of the view's mapping and lets go of an import, and gives back nothing the guest
    /// picked (the host places it): it is let go of on the surface worker
    /// ([`super::surface_pages`]).
    pub(crate) over_view: bool,
    /// The table's clock at its make.
    made: u64,
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
            over_view: false,
            made: 0,
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
}

impl<P: Platform> Tables<P> {
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

    /// Removes `owned`, `id`'s entry: onto the gone list where a view is open (each was taken
    /// before this), else handed back to let go of.
    fn retire(&mut self, id: u64, owned: Owned<P>) -> Option<Removed<P>> {
        self.clock += 1;
        let removed = Removed { id, owned };
        if self.views.is_empty() {
            return Some(removed);
        }
        self.gone.push((removed, self.clock));
        None
    }

    /// The gone entries no open view can find any more, in the order removed.
    fn sweep(&mut self) -> Vec<Removed<P>> {
        let Some((&oldest, _)) = self.views.first_key_value() else {
            return std::mem::take(&mut self.gone)
                .into_iter()
                .map(|(r, _)| r)
                .collect();
        };
        let (keep, out): (Vec<_>, Vec<_>) = std::mem::take(&mut self.gone)
            .into_iter()
            .partition(|(_, removed_at)| oldest < *removed_at);
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
            }),
            drain: Mutex::new(Arc::default()),
        }
    }

    /// Stores `entry` under `id`, in place of what the id named, and the pointer to it; what it
    /// replaced, to let go of where no view needs it.
    pub(crate) fn insert(&self, id: u64, mut entry: Entry<P>) -> Made<P> {
        let mut t = self.tables.write();
        t.clock += 1;
        entry.made = t.clock;
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
        live: impl FnOnce() -> bool,
    ) -> Option<Made<P>> {
        let mut t = self.tables.write();
        if !live() {
            return None;
        }
        t.clock += 1;
        entry.made = t.clock;
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

    /// Whether any entry in the table lies in the `size` bytes at region offset `offset`.
    pub(crate) fn span_in(&self, offset: u64, size: u64) -> bool {
        let end = offset.saturating_add(size);
        let t = self.tables.read();
        t.live.values().any(|o| {
            o.entry()
                .span
                .is_some_and(|(at, length)| at < end && offset < at.saturating_add(length))
        })
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

    /// The ids in the table (for the tests).
    #[cfg(test)]
    pub(crate) fn ids(&self) -> Vec<u64> {
        self.tables.read().live.keys().copied().collect()
    }

    /// How many entries the table holds (for the tests).
    #[cfg(any(test, feature = "testkit"))]
    pub(crate) fn len(&self) -> usize {
        self.tables.read().live.len()
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
            seen: RefCell::new(HashMap::default()),
        }
    }

    /// The context viewed.
    pub(crate) fn context(&self) -> &Arc<Context<P>> {
        &self.context
    }

    /// The entry of `id` the view sees.
    pub(crate) fn get(&self, id: u64) -> Option<&Entry<P>> {
        let seen = self.seen.borrow().get(&id).copied();
        let found = match seen {
            Some(found) => found,
            None => {
                let found = self.context.tables.read().find(id, self.at)?;
                self.seen.borrow_mut().insert(id, Some(found));
                Some(found)
            }
        };
        // SAFETY: the entry was in the table, made by the view's taking, or removed after it, so
        // it is freed only once no view taken before its removal is open, and this one is until
        // it drops; or it is the message's own make, removed (if at all) after this view's taking.
        // Either way it outlives `&self`.
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
    use reims_vgpu_core::testkit::FakePlatform;

    use super::*;

    fn fence(p: &FakePlatform) -> ObjHandle<FakePlatform> {
        use reims_vgpu_core::platform::device::GpuDevice as _;
        let device = p.device();
        ObjHandle::Fence(device.new_fence().unwrap().unwrap())
    }

    /// A view sees the table as of its taking: an entry removed after it is still found, one made
    /// after it is not, and the removed one goes once the view closes.
    #[test]
    fn a_view_sees_the_table_as_of_its_taking() {
        let p = FakePlatform::new();
        let ctx = Arc::new(Context::<FakePlatform>::new(1));
        let (_, replaced) = ctx.insert(1, Entry::of(fence(&p)));
        assert!(replaced.is_none());
        let view = View::take(Arc::clone(&ctx));
        assert!(
            ctx.remove(1).is_some_and(|(_, now)| now.is_none()),
            "kept for the open view"
        );
        let (_, replaced) = ctx.insert(2, Entry::of(fence(&p)));
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

    /// An entry removed with no view open is handed back at once, and a view taken after a
    /// removal never keeps it.
    #[test]
    fn an_entry_no_view_needs_is_handed_back() {
        let p = FakePlatform::new();
        let ctx = Arc::new(Context::<FakePlatform>::new(1));
        ctx.insert(1, Entry::of(fence(&p)));
        assert!(ctx.remove(1).is_some_and(|(_, now)| now.is_some()));
        ctx.insert(1, Entry::of(fence(&p)));
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
