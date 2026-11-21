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
//! only such views find it, and let go of once the last of them closes, on the thread of a view
//! closing then, with the others it frees in the order they were removed: as an object destroyed after a whole
//! message under one lock is let go of after it. A view names an entry where it was taken at or
//! after the entry's make, or by the message that made it ([`View::made`]), and before its removal.
//! Else it is let go of at once, by whoever removed it, as one is that only a view taken before its
//! make saw nothing of: an older view open meanwhile (a message that waits) keeps only what it
//! names, so the list holds no more than the open views name. Each release is made in a refusal
//! scope ([`super::drop_refused`]).
//!
//! **After the buffers posted before it.** A destroy or a make over a live id is carried out on
//! ring 0's lane as ring 0's order reaches it, while a command buffer its process posted before it,
//! on any of its queues' rings, may not have taken its view yet (the queue's thread takes it
//! later, or its begin has not come to its view): that buffer would see the entry gone, and its
//! work, which reaches objects through argument buffers and heaps it names no entry of, would run
//! after the platform let go of the object. So the removal reads, for each of the context's queues,
//! the highest seqno of a command buffer its ring has published, taken or not, and where the
//! queue's commit mark has not reached it, the entry waits on the gone list for that buffer's
//! commit too ([`Pending`]): it is let go of once every such buffer is committed, stood in for or
//! failed (its queue's commit mark, [`super::commit_order`]), and no open view names it, never at
//! the buffers' completion, which the platform's release waits for itself. Ring 0 waits for
//! nothing meanwhile: the entry is only moved, and a make of any id goes into the table at once.
//! Whoever lets go of a waiting entry is the thread that sweeps then: a buffer's end past its
//! commit ([`Context::settle`]), a view's close, or a queue's end. A queue that ends (its destroy,
//! or its context's close) commits nothing more, so a wait for its buffers turns into a wait for
//! the views open then, a buffer begun after the removal among them, each closed only after its
//! commit. A close itself waits for no buffer: one its process posted before the close and not yet
//! begun finds its context closed or its queue gone, and runs nothing.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use parking_lot::RwLock;
use reims_vgpu_vulkan::core::collections::hasher::KeyBuild;
use reims_vgpu_vulkan::core::host::HostPtr;
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::command::EscapedCopies;
use reims_vgpu_vulkan::core::sync::Mutex;

use super::lane::Lane;
use super::managed::ManagedCopy;
use super::surface_cache::Repitched;
use super::surface_cache::shapes::ShapeUse;
use super::surface_order::Marks;
use super::video::PlanePlace;
use super::{Drain, ObjHandle, TableObject, drop_refused};

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
    /// How many of those are committed: work of the queue's on the GPU, or on its way there, whose
    /// completion is not yet seen. A walk behind one cuts its buffer's parts by their estimated
    /// GPU time ([`super::split`], "How many parts").
    pub(crate) out: Arc<AtomicUsize>,
    /// Its last rate of GPU time to walk time, as a walk in parts of one of its buffers measured
    /// it ([`super::split`], "How far ahead of the GPU").
    pub(crate) rate: Arc<super::split::QueueRate>,
    /// Its marks in the surface order, which a command buffer that holds no op reads and writes
    /// with no lock ([`super::surface_order::Marks`]).
    pub(crate) order: Arc<super::surface_order::Marks>,
    /// Its committed command buffers' records the host may post ahead of the platform's
    /// submission, in commit order ([`super::cmdbuf::Line`]).
    pub(crate) line: Arc<super::cmdbuf::Line>,
    /// Its command buffers' turns of their `CB_COMPLETE` posts, in commit order
    /// ([`super::completions`]).
    pub(crate) completions: Arc<super::completions::Completions>,
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
            out: Arc::default(),
            rate: Arc::default(),
            order,
            line: Arc::default(),
            completions: Arc::default(),
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

/// What keeps bytes the platform imports mapped (a lease on the guest's RAM, a pages block's view,
/// a keep of a pool piece's block): a deallocator over them holds it, so they are unmapped only
/// once the platform let go of its import of them, wherever the last of its holds goes.
pub(crate) type BytesKeep = Arc<dyn std::any::Any + Send + Sync>;

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
    /// What keeps `ram`'s bytes mapped, which the heap's deallocator holds too: the lease on the
    /// guest's RAM for a pages block of one run, the block's view for one of more, or a keep of
    /// the pool piece's block for a heap over the pool. What a no-copy buffer over them holds
    /// (`BUFFER_MANAGED_AT_OFFSET`).
    pub(crate) ram_kept: Option<BytesKeep>,
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
    /// A texture, or a view of one, that is a view of a shape's image over a device cache in the
    /// per-shape form, and the reference it holds to the shape ([`ShapeUse`]); in the default
    /// form, one over a cache's buffer, and its reference to the shape the per-shape form would
    /// give it, in the cache's shadow, for the counts alone. `None` for any other object.
    pub(crate) shape: Option<ShapeUse>,
    /// An object over a device cache's buffer itself, in the per-shape form not through a
    /// shape's image: the buffer over a block's surface pages that is its cache, and each
    /// texture or view over it that is no view of a shape's image. The GPU's work through it
    /// writes the cache's buffer, which the cache's record of its written bytes holds none of: a
    /// command buffer whose walk names it and that may write its block makes the record whole
    /// ([`GraphicsDecoder::note_over_cache`]).
    ///
    /// [`GraphicsDecoder::note_over_cache`]: super::GraphicsDecoder::note_over_cache
    pub(crate) over_cache: bool,
    /// A no-copy buffer whose identity the guest may hand on with no message naming it, and
    /// whether it did ([`super::escape`]); `None` for any other object.
    pub(crate) escape: Option<super::escape::Escape>,
    /// A heap over a copy of its pages on its process's escaped copies, which it leaves at its drop
    /// ([`super::escape::Listed`]); `None` for any other object.
    pub(crate) listed: Option<super::escape::Listed>,
    /// A texture over an IOSurface's plane 0, which a present shows: the display's view of it
    /// ([`crate::graphics::ShownView`]), kept while the entry lives and while the display shows
    /// the texture; `None` for any other object.
    pub(crate) shown_view: Option<Arc<crate::graphics::ShownView<P>>>,
    /// The table's clock at its make.
    made: u64,
    /// The first clock of a view that names it: its make's, or the making message's view's where
    /// that was taken before ([`Context::insert_after`]'s `seen_from`).
    seen: u64,
    /// An object counted as made by its kind, live while its entry is ([`super::unkept`]); `None`
    /// where none is counted. It drops after the object.
    live: Option<super::unkept::Live>,
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
            ram_kept: None,
            bytes: None,
            over_view: false,
            icb_dropped: None,
            laid_out: None,
            laid_use: None,
            shape: None,
            over_cache: false,
            escape: None,
            listed: None,
            shown_view: None,
            made: 0,
            seen: 0,
            live: None,
        }
    }

    /// The entry, lying over `span`.
    pub(crate) fn over(mut self, span: Option<(u64, u64)>) -> Self {
        self.span = span;
        self
    }

    /// The entry, an object over a device cache's buffer itself where `over` (the field
    /// `over_cache`).
    pub(crate) fn over_cache(mut self, over: bool) -> Self {
        self.over_cache = over;
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

    /// The entry, of an object counted as made while it lives, where `live` is given (the field
    /// `live`).
    pub(crate) fn counted(mut self, live: Option<super::unkept::Live>) -> Self {
        self.live = live;
        self
    }

    /// The entry, a no-copy buffer whose identity the guest notes as it escapes (the field
    /// `escape`).
    pub(crate) fn escapes(mut self, escape: super::escape::Escape) -> Self {
        self.escape = Some(escape);
        self
    }
}

/// A table's objects by kind as they are now, for the host's summary of each ten seconds
/// (`crate::graphics::summary::kept`; [`Context::counts`]), so a boot's reader can tell the
/// platform's buffers and textures a guest id names from those it holds for no id. Each count is
/// of the live entries; those removed and kept for an open view or a command buffer posted before
/// the removal are counted apart (`*_gone`). A diagnostic.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct TableCounts {
    /// Textures and views of textures; of the live ones, the views of a device cache's shape's
    /// image ([`Entry::shape`]), and those the display made its own view of
    /// ([`Entry::shown_view`]).
    pub(crate) textures: u64,
    pub(crate) textures_gone: u64,
    pub(crate) shape_views: u64,
    pub(crate) shown_views: u64,
    /// Buffers; of the live ones, those over a device cache's buffer itself
    /// ([`Entry::over_cache`]), which are the cache's buffer, and those over a block's surface view
    /// ([`Entry::over_view`]).
    pub(crate) buffers: u64,
    pub(crate) buffers_gone: u64,
    pub(crate) over_cache: u64,
    pub(crate) over_view: u64,
    /// Heaps, live and gone.
    pub(crate) heaps: u64,
    /// The buffers over two-copy `Managed` objects' CPU copies.
    pub(crate) managed_copies: u64,
}

impl TableCounts {
    /// Counts `entry`, removed and kept where `gone`.
    fn add<P: Platform>(&mut self, entry: &Entry<P>, gone: bool) {
        let cpu = (entry.managed.as_ref()).is_some_and(|copy| copy.cpu.is_some());
        self.managed_copies += u64::from(cpu);
        match (&entry.handle, gone) {
            (ObjHandle::Texture(_), true) => self.textures_gone += 1,
            (ObjHandle::Texture(_), false) => {
                self.textures += 1;
                self.shape_views += u64::from(entry.shape.is_some());
                let shown = (entry.shown_view.as_ref()).is_some_and(|view| view.is_made());
                self.shown_views += u64::from(shown);
            }
            (ObjHandle::Buffer(_), true) => self.buffers_gone += 1,
            (ObjHandle::Buffer(_), false) => {
                self.buffers += 1;
                self.over_cache += u64::from(entry.over_cache);
                self.over_view += u64::from(entry.over_view);
            }
            (ObjHandle::Heap(_), _) => self.heaps += 1,
            _ => {}
        }
    }

    /// Adds `other`'s counts to these.
    pub(crate) fn sum(&mut self, other: &TableCounts) {
        self.textures += other.textures;
        self.textures_gone += other.textures_gone;
        self.shape_views += other.shape_views;
        self.shown_views += other.shown_views;
        self.buffers += other.buffers;
        self.buffers_gone += other.buffers_gone;
        self.over_cache += other.over_cache;
        self.over_view += other.over_view;
        self.heaps += other.heaps;
        self.managed_copies += other.managed_copies;
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

/// A command buffer an entry removed from its table waits for before it is let go of (the module
/// doc, "After the buffers posted before it"): its queue's marks, and its seqno. Counted among the
/// marks' waiting entries while it waits ([`Marks::park`]), so a buffer's end looks at its table
/// only where an entry waits for its queue.
pub(crate) struct Pending {
    marks: Arc<Marks>,
    seqno: u64,
}

impl Pending {
    /// A wait for the buffer at `seqno` of the queue whose marks are `marks`: counted among them
    /// before the removal looks at the queue's commit mark under the table's lock, so a buffer's
    /// end that raises the mark and then finds no entry waiting is seen by that look.
    pub(crate) fn new(marks: Arc<Marks>, seqno: u64) -> Self {
        marks.park();
        Pending { marks, seqno }
    }

    /// The buffer was committed, stood in for or failed.
    fn committed(&self) -> bool {
        self.marks.commit.committed_through(self.seqno)
    }

    /// Its queue ended: nothing of it is committed any more.
    fn ended(&self) -> bool {
        self.marks.commit.has_ended()
    }
}

impl Drop for Pending {
    fn drop(&mut self) {
        self.marks.unpark();
    }
}

/// An entry removed from the table while a view or a command buffer may still reach it (the module
/// doc).
struct Gone<P: Platform> {
    removed: Removed<P>,
    /// The table's clock at its removal.
    at: u64,
    /// The command buffers posted before its removal whose commits it waits for.
    pending: Vec<Pending>,
    /// Kept while a view taken before this clock is open: a queue whose buffers it waited for
    /// ended; 0 for none.
    open_before: u64,
}

impl<P: Platform> Gone<P> {
    /// Whether the entry is still to be kept, with the views `open` open and the table's clock at
    /// `clock`: a buffer it waits for is not yet committed, or an open view names it. A wait for a
    /// buffer of a queue that ended turns into a wait for every view open now.
    fn kept(&mut self, open: &BTreeMap<u64, usize>, clock: u64) -> bool {
        self.pending.retain(|p| !p.committed());
        if self.pending.iter().any(Pending::ended) {
            self.pending.retain(|p| !p.ended());
            self.open_before = self.open_before.max(clock + 1);
        }
        !self.pending.is_empty()
            || open
                .range(self.removed.owned.entry().seen..self.at)
                .next()
                .is_some()
            || open.range(..self.open_before).next().is_some()
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
    /// The entries removed while a view taken before the removal was open, or while a command
    /// buffer posted before it was not yet committed, in the order removed ([`Gone`]).
    gone: Vec<Gone<P>>,
    /// Moves at each make and removal.
    clock: u64,
    /// The region spans the live entries lie over ([`Entry::span`]), each with how many lie over
    /// it: what [`Context::span_in`] looks through, a few keys about the range it asks of, where
    /// the entries themselves are thousands. It cannot go stale: an entry's span is counted where
    /// the entry goes into `live` ([`Context::insert_after`], [`Context::insert_if`]) and taken
    /// out where it leaves ([`Self::retire`], which every removal from `live` goes through: a
    /// remove, a make over its id again, a close's removal of every entry), each under the table's
    /// write lock with the change itself, and a span is set at the entry's make and never changed
    /// while it is in the table (the module doc). A context dropped takes its spans with it.
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
        self.gone.iter().rev().find_map(|g| {
            let r = &g.removed;
            (r.id == id && r.owned.entry().made <= at && at < g.at).then_some(r.owned.0)
        })
    }

    /// Removes `owned`, `id`'s entry: onto the gone list where an open view of `views` names it
    /// (each open view was taken before this) or a command buffer of `pending` is not yet
    /// committed, else handed back to let go of.
    fn retire(
        &mut self,
        id: u64,
        owned: Owned<P>,
        views: &mut Views,
        pending: Vec<Pending>,
    ) -> Option<Removed<P>> {
        self.clock += 1;
        self.uncount_span(owned.entry().span);
        let mut gone = Gone {
            removed: Removed { id, owned },
            at: self.clock,
            pending,
            open_before: 0,
        };
        if !gone.kept(&views.open, self.clock) {
            return Some(gone.removed);
        }
        self.gone.push(gone);
        views.gone = true;
        None
    }

    /// The gone entries no open view of `views` names any more and that wait for no command
    /// buffer's commit, in the order removed: a view names one where it was taken from the entry's
    /// `seen` up to its removal ([`Gone::kept`]).
    fn sweep(&mut self, views: &mut Views) -> Vec<Removed<P>> {
        if self.gone.is_empty() {
            return Vec::new();
        }
        let clock = self.clock;
        let mut out = Vec::new();
        let mut keep = Vec::new();
        for mut g in std::mem::take(&mut self.gone) {
            if g.kept(&views.open, clock) {
                keep.push(g);
            } else {
                out.push(g.removed);
            }
        }
        self.gone = keep;
        views.gone = !self.gone.is_empty();
        out
    }
}

/// The open views of a context's table, by the clock at which each was taken: how many at each,
/// and whether the table's gone list holds an entry. A view is counted with the table's read lock
/// held, so no make or removal (each under its write lock) comes between its reading of the clock
/// and its count, and a removal, which reads them with the write lock held, finds every view open
/// then; a view's close uncounts it under this lock alone, and takes the write lock to sweep the
/// gone list only where the list holds an entry. So a command buffer's view, taken at its begin
/// on its queue's thread and closed after its walk on its lane's, holds up neither the lanes'
/// lookups of the table nor ring 0's makes: as the write lock, the view's taking was 1.2 µs of a
/// Super Resolution buffer's begin on the Iris Xe host, up to 9.8 µs in one of a hundred
/// (**measured** 2026-11-14, laps of the begin, gpre m3-max-31335).
#[derive(Default)]
struct Views {
    /// How many views are open at each clock.
    open: BTreeMap<u64, usize>,
    /// The table's gone list holds an entry: set as one goes on it, cleared as a sweep empties
    /// it, both with the write lock held too.
    gone: bool,
}

impl<P: Platform> Tables<P> {
    /// Removes `id`'s live entry, where it has one, as [`Self::retire`] does.
    fn retire_live(
        &mut self,
        id: u64,
        views: &mut Views,
        pending: Vec<Pending>,
    ) -> Option<Option<Removed<P>>> {
        let owned = self.live.remove(&id)?;
        Some(self.retire(id, owned, views, pending))
    }
}

/// What [`Context::each_object`] calls with each entry of the table: its id, whether it was
/// removed and is kept for an open view, its handle and its `Managed` CPU copy's buffer.
pub(super) type EachEntry<'a, P> =
    dyn FnMut(u64, bool, &ObjHandle<P>, Option<&<P as Platform>::Buffer>) + 'a;

/// One context's state: its objects, and the drain its command buffers join.
pub(crate) struct Context<P: Platform> {
    /// The context's id.
    pub(crate) id: u32,
    tables: RwLock<Tables<P>>,
    /// The table's open views ([`Views`]), taken with `tables`' lock held where both are.
    views: Mutex<Views>,
    /// What a test runs at a step of a view's taking or close ([`ViewStep`]).
    #[cfg(test)]
    view_step: Mutex<Option<ViewStepHook>>,
    /// The command buffers committed since the context's last drain began: each holds a clone
    /// until its completion handler has run ([`super::GraphicsDecoder::drain_context`]).
    pub(crate) drain: Mutex<Arc<Drain>>,
    /// The device's recoveries its two-copy `Managed` textures were last looked at for
    /// (`GraphicsDecoder::refill_managed`): a command buffer's begin past one refills those made
    /// before it, under this lock, so no command buffer of the context is begun between.
    pub(crate) managed_recoveries: Mutex<u64>,
    /// The context's live buffers over a copy of their read-only pages whose identity escaped
    /// ([`super::escape`]): each adds itself at its first note and leaves at its drop.
    pub(crate) escaped_copies: Arc<AtomicUsize>,
    /// The platform's copies under the context's objects whose identity escaped, which each of
    /// its command buffers syncs at its commit ([`super::escape`]): each object adds its copy at
    /// its first note, a heap at its make, and takes it off at its drop.
    pub(crate) escapes: EscapedCopies,
}

/// Tells the object `entry` holds, which a table stores under `id`, and its `Managed` CPU copy,
/// where it has one, that id, which the rt platform's device names them by in its history of
/// released ranges and its lines of the command buffers in flight at its loss
/// (`super::super::fault_walk::name`): for a buffer, texture or heap, a downcast and an atomic
/// store on the rt platform, two for a CPU copy; nothing on another platform or for another
/// object.
fn name_for_loss<P: Platform>(id: u64, entry: &Entry<P>) {
    let object = match &entry.handle {
        ObjHandle::Buffer(buffer) => TableObject::<P>::Buffer(buffer),
        ObjHandle::Texture(texture) => TableObject::Texture(texture),
        ObjHandle::Heap(heap) => TableObject::Heap(heap),
        _ => return,
    };
    super::super::fault_walk::name(id, object);
    if let Some(cpu) = (entry.managed.as_ref()).and_then(|copy| copy.cpu.as_ref()) {
        super::super::fault_walk::name(id, TableObject::<P>::ManagedCopy(cpu));
    }
}

impl<P: Platform> Context<P> {
    /// Calls `f` with each entry of the table, its id, whether it was removed and is kept for an
    /// open view, its handle and its `Managed` CPU copy's buffer, under a read of the table's lock
    /// had within `bound`; false where the lock was not had in it (a device's loss,
    /// `super::TableWalk`).
    pub(super) fn each_object(&self, bound: std::time::Duration, f: &mut EachEntry<'_, P>) -> bool {
        let Some(tables) = self.tables.try_read_for(bound) else {
            return false;
        };
        fn cpu<P: Platform>(entry: &Entry<P>) -> Option<&P::Buffer> {
            (entry.managed.as_ref()).and_then(|copy| copy.cpu.as_ref())
        }
        for (&id, owned) in &tables.live {
            let entry = owned.entry();
            f(id, false, &entry.handle, cpu(entry));
        }
        for g in &tables.gone {
            let entry = g.removed.entry();
            f(g.removed.id(), true, &entry.handle, cpu(entry));
        }
        true
    }

    /// The table's objects by kind as they are now ([`TableCounts`]), under a read of the table's
    /// lock had within `bound`; `None` where the lock was not had in it.
    pub(crate) fn counts(&self, bound: std::time::Duration) -> Option<TableCounts> {
        let tables = self.tables.try_read_for(bound)?;
        let mut counts = TableCounts::default();
        let live = tables.live.values().map(|owned| (owned.entry(), false));
        let gone = tables.gone.iter().map(|g| (g.removed.entry(), true));
        for (entry, gone) in live.chain(gone) {
            counts.add(entry, gone);
        }
        Some(counts)
    }

    /// A context with no objects.
    pub(crate) fn new(id: u32) -> Self {
        Context {
            id,
            tables: RwLock::new(Tables {
                live: HashMap::default(),
                gone: Vec::new(),
                clock: 0,
                spans: BTreeMap::new(),
                longest: 0,
            }),
            views: Mutex::default(),
            #[cfg(test)]
            view_step: Mutex::new(None),
            drain: Mutex::new(Arc::default()),
            managed_recoveries: Mutex::new(0),
            escaped_copies: Arc::default(),
            escapes: EscapedCopies::default(),
        }
    }

    /// [`Self::insert_after`] with no command buffer to wait for.
    #[cfg(test)]
    pub(crate) fn insert(&self, id: u64, entry: Entry<P>, seen_from: Option<u64>) -> Made<P> {
        self.insert_after(id, entry, seen_from, Vec::new())
    }

    /// Stores `entry` under `id`, in place of what the id named, and the pointer to it; what it
    /// replaced, to let go of where no view needs it and every command buffer of `pending` is
    /// committed (the module doc, "After the buffers posted before it"). `seen_from` is the clock
    /// of the making message's view of this table, where it has one ([`View::at`]), which names
    /// the entry though taken before its make ([`View::made`]). The entry's object is told `id`
    /// for its device's loss ([`name_for_loss`]).
    pub(crate) fn insert_after(
        &self,
        id: u64,
        mut entry: Entry<P>,
        seen_from: Option<u64>,
        pending: Vec<Pending>,
    ) -> Made<P> {
        let mut t = self.tables.write();
        name_for_loss(id, &entry);
        t.clock += 1;
        entry.made = t.clock;
        entry.seen = seen_from.map_or(t.clock, |at| at.min(t.clock));
        t.count_span(entry.span);
        let owned = Owned::new(entry);
        let at = owned.0;
        let replaced = (t.live.insert(id, owned))
            .and_then(|old| t.retire(id, old, &mut self.views.lock(), pending));
        (at, replaced)
    }

    /// [`Self::insert_after`], where `live` answers, under the table's lock, that the entry is
    /// still to be stored; `None` where it is not.
    pub(crate) fn insert_if(
        &self,
        id: u64,
        mut entry: Entry<P>,
        seen_from: Option<u64>,
        pending: Vec<Pending>,
        live: impl FnOnce() -> bool,
    ) -> Option<Made<P>> {
        let mut t = self.tables.write();
        if !live() {
            return None;
        }
        name_for_loss(id, &entry);
        t.clock += 1;
        entry.made = t.clock;
        entry.seen = seen_from.map_or(t.clock, |at| at.min(t.clock));
        t.count_span(entry.span);
        let owned = Owned::new(entry);
        let at = owned.0;
        let replaced = (t.live.insert(id, owned))
            .and_then(|old| t.retire(id, old, &mut self.views.lock(), pending));
        Some((at, replaced))
    }

    /// [`Self::remove_after`] with no command buffer to wait for.
    #[cfg(test)]
    pub(crate) fn remove(&self, id: u64) -> Option<(bool, Option<Removed<P>>)> {
        self.remove_after(id, Vec::new())
    }

    /// Removes `id`'s entry, where it has one: whether it was a texture, and it, to let go of
    /// where no view needs it and every command buffer of `pending` is committed (the module doc,
    /// "After the buffers posted before it").
    pub(crate) fn remove_after(
        &self,
        id: u64,
        pending: Vec<Pending>,
    ) -> Option<(bool, Option<Removed<P>>)> {
        let mut t = self.tables.write();
        let texture = matches!(t.live.get(&id)?.entry().handle, ObjHandle::Texture(_));
        let removed = t.retire_live(id, &mut self.views.lock(), pending)?;
        Some((texture, removed))
    }

    /// Removes every entry, in the table's order: those to let go of now, where no view needs
    /// them, and how many there were. They wait for no command buffer's commit: a buffer posted
    /// before the close that has not taken its view finds its context closed or its queue gone,
    /// and runs nothing (the module doc).
    pub(crate) fn remove_all(&self) -> (Vec<Removed<P>>, usize) {
        let mut t = self.tables.write();
        let mut views = self.views.lock();
        let ids: Vec<u64> = t.live.keys().copied().collect();
        let count = ids.len();
        let mut out = Vec::new();
        for id in ids {
            if let Some(Some(removed)) = t.retire_live(id, &mut views, Vec::new()) {
                out.push(removed);
            }
        }
        (out, count)
    }

    /// Lets go of the removed entries no open view names and no command buffer's commit holds any
    /// more: at a buffer's end past its commit, where an entry waits for its queue
    /// ([`Marks::has_parked`]), and at a queue's end (the module doc, "After the buffers posted
    /// before it").
    pub(crate) fn settle(&self) {
        if !self.views.lock().gone {
            return;
        }
        let out = {
            let mut t = self.tables.write();
            t.sweep(&mut self.views.lock())
        };
        let_go(out);
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
        out.extend(std::mem::take(&mut t.gone).into_iter().map(|g| g.removed));
        let_go(out);
    }
}

/// The names a view's map of what it found holds before it grows: a command buffer of a few
/// dispatches names tens of objects, and each growth moves every name found so far.
const SEEN_ROOM: usize = 32;

/// How many entries [`View::find_all`] looks for under one hold of the table's lock: a command
/// buffer of a few dispatches finds every name it has in one, and the table's makes and removals
/// wait no longer than as many lookups.
pub(super) const FOUND_A_HOLD: usize = 32;

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
            let t = context.tables.read();
            let at = t.clock;
            #[cfg(test)]
            context.step(ViewStep::Taking);
            *context.views.lock().open.entry(at).or_default() += 1;
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

    /// A view of the same table at this one's clock, with what this one found and the message's
    /// own changes it holds: it names what this one names, for chunks of the message's command
    /// buffer walked on another thread beside the rest ([`super::split`], "Two threads at once").
    /// The entries it keeps are let go of once both have closed, as any open view's are.
    pub(crate) fn twin(&self) -> Self {
        // This view, open at the same clock, holds its count there: no removal's or sweep's answer
        // changes with the twin's, so no lock of the table's is needed for it.
        *self.context.views.lock().open.entry(self.at).or_default() += 1;
        View {
            context: Arc::clone(&self.context),
            at: self.at,
            seen: RefCell::new(self.seen.borrow().clone()),
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

    /// Whether the view has its answer for `id` already, found, made or removed: a
    /// [`Self::get`] of it takes no lock of the table's.
    pub(crate) fn found(&self, id: u64) -> bool {
        self.seen.borrow().contains_key(&id)
    }

    /// Finds each of `ids` the view has not found yet, [`FOUND_A_HOLD`] of them under each hold
    /// of the table's lock, so [`Self::get`] answers them from what the view found. A view's
    /// answer for an id is set at its taking (the module doc): finding it now or at its first
    /// [`Self::get`] finds the same entry, or none. An id the table holds no entry of for the
    /// view is not kept, and [`Self::get`] looks it up again; one found that no op of the message
    /// reaches changes nothing the table keeps.
    pub(crate) fn find_all(&self, ids: impl IntoIterator<Item = u64>) {
        let mut ids = ids.into_iter();
        while self.find_some(&mut ids) {}
    }

    /// [`Self::find_all`]'s next [`FOUND_A_HOLD`] lookups of `ids` in the table, under one hold
    /// of its lock, let go of before it answers: whether `ids` may hold more.
    fn find_some(&self, ids: &mut impl Iterator<Item = u64>) -> bool {
        let mut seen = self.seen.borrow_mut();
        let mut tables = None;
        let mut held = 0;
        for id in ids {
            if let std::collections::hash_map::Entry::Vacant(slot) = seen.entry(id) {
                let t = tables.get_or_insert_with(|| self.context.tables.read());
                if let Some(found) = t.find(id, self.at) {
                    slot.insert(Some(found));
                }
                held += 1;
                if held == FOUND_A_HOLD {
                    return true;
                }
            }
        }
        false
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
    /// Uncounts the view; then, where the table's gone list holds an entry, lets go of those no
    /// open view names any more ([`Views`]).
    fn drop(&mut self) {
        let gone = {
            let mut views = self.context.views.lock();
            if let Some(n) = views.open.get_mut(&self.at) {
                *n -= 1;
                if *n == 0 {
                    views.open.remove(&self.at);
                }
            }
            views.gone
        };
        if !gone {
            return;
        }
        #[cfg(test)]
        self.context.step(ViewStep::Uncounted);
        let out = {
            let mut t = self.context.tables.write();
            t.sweep(&mut self.context.views.lock())
        };
        let_go(out);
    }
}

/// A step of a view's taking or close, at which a test runs what it set
/// ([`Context::on_view_step`]), to put another thread's work there.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ViewStep {
    /// The table's clock read for the view, its read lock held, the view not yet counted.
    Taking,
    /// The view uncounted, the gone list holding an entry, before the sweep.
    Uncounted,
}

/// What a test runs at a [`ViewStep`].
#[cfg(test)]
type ViewStepHook = Arc<dyn Fn(ViewStep) + Send + Sync>;

#[cfg(test)]
impl<P: Platform> Context<P> {
    /// Runs `hook` at each step of a view's taking or close from now on.
    pub(crate) fn on_view_step(&self, hook: impl Fn(ViewStep) + Send + Sync + 'static) {
        *self.view_step.lock() = Some(Arc::new(hook));
    }

    fn step(&self, step: ViewStep) {
        let hook = self.view_step.lock().clone();
        if let Some(hook) = hook {
            hook(step);
        }
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

    /// The table's counts tell its textures and buffers by kind, live apart from those removed and
    /// kept for an open view, and leave out every other kind of object.
    #[test]
    fn the_counts_tell_live_textures_and_buffers_apart_from_the_kept() {
        use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
        use reims_vgpu_vulkan::core::platform::values::TextureDesc;
        use reims_vgpu_vulkan::core::raise::OnFault;
        let p = FakePlatform::new();
        let device = p.device();
        let buffer = || {
            let made = device.new_buffer_of_length(OnFault::Die, 64, 0);
            ObjHandle::Buffer(made.unwrap().unwrap())
        };
        let desc = p.texture_descriptor(&TextureDesc {
            width: 4,
            height: 4,
            ..TextureDesc::default()
        });
        let texture = || {
            let made = device.new_texture(OnFault::Die, &desc);
            ObjHandle::Texture(made.unwrap().unwrap())
        };
        let ctx = Arc::new(Context::<FakePlatform>::new(1));
        drop(ctx.insert(1, Entry::of(texture()), None));
        drop(ctx.insert(2, Entry::of(texture()), None));
        drop(ctx.insert(3, Entry::of(buffer()).over_cache(true), None));
        drop(ctx.insert(4, Entry::of(buffer()).over_view(), None));
        drop(ctx.insert(5, Entry::of(fence(&p)), None));
        let view = View::take(Arc::clone(&ctx));
        assert!(view.get(2).is_some());
        assert!(ctx.remove(2).is_some_and(|(_, now)| now.is_none()), "kept for the view");
        let counts = ctx.counts(std::time::Duration::from_secs(1)).unwrap();
        assert_eq!((counts.textures, counts.textures_gone), (1, 1));
        assert_eq!((counts.buffers, counts.buffers_gone), (2, 0));
        assert_eq!((counts.over_cache, counts.over_view), (1, 1));
        assert_eq!((counts.shape_views, counts.shown_views, counts.heaps), (0, 0, 0));
        drop(view);
        let mut sum = ctx.counts(std::time::Duration::from_secs(1)).unwrap();
        assert_eq!(sum.textures_gone, 0, "let go of with the view");
        sum.sum(&counts);
        assert_eq!((sum.textures, sum.buffers), (2, 4));
    }

    /// Holds the first view step of `want` a thread comes to, there, until the test lets it go:
    /// the step's answer on `at` once there, and the thread goes on once `go` is sent to. Every
    /// later step runs on.
    fn hold_first(
        ctx: &Context<FakePlatform>,
        want: ViewStep,
    ) -> (std::sync::mpsc::Receiver<()>, std::sync::mpsc::Sender<()>) {
        let (at_tx, at) = std::sync::mpsc::channel();
        let (go, go_rx) = std::sync::mpsc::channel::<()>();
        let held = std::sync::atomic::AtomicBool::new(false);
        let (at_tx, go_rx) = (Mutex::new(at_tx), Mutex::new(go_rx));
        ctx.on_view_step(move |step| {
            if step == want && !held.swap(true, std::sync::atomic::Ordering::AcqRel) {
                at_tx.lock().send(()).unwrap();
                go_rx.lock().recv().unwrap();
            }
        });
        (at, go)
    }

    /// A view's taking reads the table's clock and counts the view with the table's read lock
    /// held: a removal, which takes its write lock, comes wholly before it or after it. Held
    /// between the two, the taking keeps the write lock from every removal; the removal after it
    /// keeps the entry for the view, which finds it, and lets go of it once the view closes. A
    /// removal before a taking is not seen by its view.
    #[test]
    fn a_views_taking_and_a_removal_come_one_after_the_other() {
        let p = FakePlatform::new();
        let ctx = Arc::new(Context::<FakePlatform>::new(1));
        for id in [1, 2] {
            let (_, replaced) = ctx.insert(id, Entry::of(fence(&p)), None);
            assert!(replaced.is_none());
        }
        let (at, go) = hold_first(&ctx, ViewStep::Taking);
        std::thread::scope(|scope| {
            let taking = scope.spawn(|| View::take(Arc::clone(&ctx)));
            at.recv().unwrap();
            assert!(
                ctx.tables.try_write().is_none(),
                "no removal while the taking holds the read lock"
            );
            assert!(
                ctx.views.lock().open.is_empty(),
                "not counted yet, the clock read"
            );
            go.send(()).unwrap();
            let view = taking.join().unwrap();
            assert!(
                ctx.remove(1).is_some_and(|(_, now)| now.is_none()),
                "kept for the view taken before the removal"
            );
            assert!(view.get(1).is_some(), "the view finds it");
            assert_eq!(ctx.gone_len(), 1);
            drop(view);
            assert_eq!(ctx.gone_len(), 0, "let go of as the view closes");
        });
        assert!(
            ctx.remove(2).is_some_and(|(_, now)| now.is_some()),
            "no view open: let go of at once"
        );
        let late = View::take(Arc::clone(&ctx));
        assert!(late.get(2).is_none(), "removed before the view: not seen");
    }

    /// A view's close uncounts it before it sweeps the gone list, with no lock of the table's
    /// held between: a removal there finds the view closed and lets go of its entry at once, a
    /// view closed meanwhile sweeps what neither names, and the sweep after lets go of what is
    /// left, each entry once.
    #[test]
    fn a_removal_between_a_views_uncounting_and_its_sweep_lets_go_at_once() {
        let p = FakePlatform::new();
        let ctx = Arc::new(Context::<FakePlatform>::new(1));
        for id in [1, 2, 3] {
            let (_, replaced) = ctx.insert(id, Entry::of(fence(&p)), None);
            assert!(replaced.is_none());
        }
        let first = View::take(Arc::clone(&ctx));
        let second = View::take(Arc::clone(&ctx));
        assert!(first.get(1).is_some() && second.get(1).is_some());
        assert!(
            ctx.remove(3).is_some_and(|(_, now)| now.is_none()),
            "kept for the open views"
        );
        let (at, go) = hold_first(&ctx, ViewStep::Uncounted);
        std::thread::scope(|scope| {
            let closing = scope.spawn(move || drop(first));
            at.recv().unwrap();
            assert!(
                ctx.tables.try_write().is_some(),
                "no lock of the table's held between the two"
            );
            assert_eq!(
                ctx.views.lock().open.values().sum::<usize>(),
                1,
                "the first uncounted"
            );
            drop(second);
            assert_eq!(
                ctx.gone_len(),
                0,
                "the second's sweep let go of what neither names"
            );
            assert!(
                ctx.remove(1).is_some_and(|(_, now)| now.is_some()),
                "both closed: let go of at once"
            );
            go.send(()).unwrap();
            closing.join().unwrap();
        });
        assert_eq!(ctx.gone_len(), 0, "the first's sweep found nothing more");
        let views = ctx.views.lock();
        assert!(
            views.open.is_empty() && !views.gone,
            "nothing open, nothing gone"
        );
        drop(views);
        let late = View::take(Arc::clone(&ctx));
        assert!(late.get(1).is_none() && late.get(3).is_none());
        assert!(late.get(2).is_some());
    }

    /// A view that finds its names at once ([`View::find_all`]) answers each as one taken at the
    /// same clock that looks it up only after the table changed: through random makes, makes over
    /// an id again and removes, each view of a pair taken together sees the same entry, or none,
    /// for every id, and what the table keeps for them goes once both close.
    #[test]
    fn a_view_finds_at_once_what_it_would_find_later() {
        let p = FakePlatform::new();
        let ctx = Arc::new(Context::<FakePlatform>::new(1));
        let mut seed: u64 = 0x1234_5678_9abc_def1;
        let mut next = move |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        let change = |ctx: &Context<FakePlatform>, next: &mut dyn FnMut(u64) -> u64| {
            let id = next(48);
            if next(3) == 0 {
                if let Some((_, Some(removed))) = ctx.remove(id) {
                    let_go(vec![removed]);
                }
            } else {
                let (_, replaced) = ctx.insert(id, Entry::of(fence(&p)), None);
                let_go(replaced.into_iter().collect());
            }
        };
        for _ in 0..200 {
            for _ in 0..next(12) {
                change(&ctx, &mut next);
            }
            let early = View::take(Arc::clone(&ctx));
            let late = View::take(Arc::clone(&ctx));
            assert_eq!(early.at(), late.at());
            early.find_all(0..64);
            for _ in 0..next(24) {
                change(&ctx, &mut next);
            }
            for id in 0..64 {
                assert_eq!(
                    early.get(id).map(std::ptr::from_ref),
                    late.get(id).map(std::ptr::from_ref),
                    "id {id}"
                );
            }
            drop((early, late));
            assert_eq!(ctx.gone_len(), 0, "nothing kept once both close");
        }
    }

    /// A view finds its names [`FOUND_A_HOLD`] at a time, the table's lock let go of between: a
    /// make goes through after each hold, before the next, and is not seen by the view; names it
    /// found already, or found before, take no lookup; and a name the table holds nothing of is
    /// not kept, so a later [`View::get`] looks it up again.
    #[test]
    fn a_view_finds_its_names_a_hold_at_a_time() {
        let p = FakePlatform::new();
        let ctx = Arc::new(Context::<FakePlatform>::new(1));
        for id in 0..100 {
            ctx.insert(id, Entry::of(fence(&p)), None);
        }
        let view = View::take(Arc::clone(&ctx));
        assert!(view.get(5).is_some());
        // 0 to 99 twice, and 1,000, which the table does not hold: 100 lookups, as 5 was found.
        let mut ids = (0..100).chain(0..100).chain([1_000]);
        let mut holds = 0;
        loop {
            let more = view.find_some(&mut ids);
            holds += 1;
            assert!(ctx.tables.try_write().is_some(), "let go of after a hold");
            ctx.insert(10_000 + holds, Entry::of(fence(&p)), None);
            if !more {
                break;
            }
        }
        assert_eq!(holds, 4, "100 lookups, 32 a hold");
        assert!((0..100).all(|id| view.get(id).is_some()));
        assert!(
            (1..=4).all(|n| view.get(10_000 + n).is_none()),
            "made after it"
        );
        assert!(!view.seen.borrow().contains_key(&1_000), "not kept");
        ctx.insert(1_000, Entry::of(fence(&p)), None);
        assert!(view.get(1_000).is_none(), "made after it, looked up again");
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

    /// What a command buffer's view of its table costs beside the threads that share the table
    /// (a measurement, not a check; run it by name with `--run-ignored only --no-capture` in a
    /// release build): [`a_command_buffer_s_view_costs`]'s command buffers, while ring 0's thread
    /// makes an object and removes one made `TABLES_COST_KEEP` makes before (default 64) every
    /// `TABLES_COST_SPIN` spins (default 2,000; 0 makes none), and `TABLES_COST_LANES` more lanes
    /// (default 1) walk command buffers of their own. Each command buffer names
    /// `TABLES_COST_FRESH` of ring 0's latest makes (default 4) among its `TABLES_COST_NAMED`
    /// (default 26), as Super Resolution's name the buffers made in its heaps for them; with
    /// `TABLES_COST_AT_ONCE=1`, each view finds its names at once ([`View::find_all`]). Prints
    /// the measured lane's user-space instructions and cycles a command buffer and a first
    /// lookup, the median of 9 runs.
    #[test]
    #[ignore = "a measurement: run it by name"]
    #[allow(clippy::disallowed_methods)] // a measurement: real threads contend for the view's lock
    fn a_command_buffer_s_view_costs_beside_ring_0() {
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

        use reims_vgpu_vulkan::core::platform::Handle as _;
        use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;

        use crate::graphics::decode::draw_cost::{counts, spread};
        // Ring 0's latest makes, as ids from `FIRST` on.
        const FIRST: u64 = 1 << 32;
        let env = |name: &str, default: u64| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        };
        let live = env("TABLES_COST_LIVE", 4096);
        let cbs = env("TABLES_COST_CBS", 2000);
        let named = env("TABLES_COST_NAMED", 26);
        let fresh = env("TABLES_COST_FRESH", 4).min(named);
        let again = env("TABLES_COST_AGAIN", 3);
        let keep = env("TABLES_COST_KEEP", 64).max(fresh + 1);
        let spin = env("TABLES_COST_SPIN", 2000);
        let lanes = env("TABLES_COST_LANES", 1);
        let at_once = env("TABLES_COST_AT_ONCE", 0) == 1;
        let p = FakePlatform::new();
        let fence = p.device().new_fence().unwrap().unwrap();
        let of = |f: &<FakePlatform as Platform>::Fence| Entry::of(ObjHandle::Fence(f.clone_ref()));
        let ctx = Arc::new(Context::<FakePlatform>::new(1));
        for id in 0..live {
            ctx.insert(0x1000 + id, of(&fence), None);
        }
        for id in FIRST..FIRST + keep {
            ctx.insert(id, of(&fence), None);
        }
        // The newest of ring 0's makes.
        let latest = Arc::new(AtomicU64::new(FIRST + keep - 1));
        let stop = Arc::new(AtomicBool::new(false));
        let mut threads = Vec::new();
        if spin > 0 {
            let (ctx, latest, stop) = (Arc::clone(&ctx), Arc::clone(&latest), Arc::clone(&stop));
            let fence = fence.clone_ref();
            threads.push(std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let next = latest.load(Ordering::Relaxed) + 1;
                    let (_, replaced) = ctx.insert(next, of(&fence), None);
                    let_go(replaced.into_iter().collect());
                    latest.store(next, Ordering::Release);
                    if let Some((_, Some(removed))) = ctx.remove(next - keep) {
                        let_go(vec![removed]);
                    }
                    for _ in 0..spin {
                        std::hint::spin_loop();
                    }
                }
            }));
        }
        let rng = |mut seed: u64| {
            move |n: u64| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed % n
            }
        };
        // A command buffer: a view, its names looked up `again` + 1 times, the view dropped.
        let walk = move |ctx: &Arc<Context<FakePlatform>>,
                         latest: u64,
                         next: &mut dyn FnMut(u64) -> u64| {
            let ids: Vec<u64> = (0..named)
                .map(|i| {
                    if i < fresh {
                        latest - i
                    } else {
                        0x1000 + next(live)
                    }
                })
                .collect();
            let view = View::take(Arc::clone(ctx));
            if at_once {
                view.find_all(ids.iter().copied());
            }
            for _ in 0..=again {
                for &id in &ids {
                    std::hint::black_box(view.get(std::hint::black_box(id)));
                }
            }
            drop(view);
        };
        for lane in 0..lanes {
            let (ctx, latest, stop) = (Arc::clone(&ctx), Arc::clone(&latest), Arc::clone(&stop));
            threads.push(std::thread::spawn(move || {
                let mut next = rng(0x2545_f491_4f6c_dd1d ^ (lane + 1));
                while !stop.load(Ordering::Relaxed) {
                    walk(&ctx, latest.load(Ordering::Acquire), &mut next);
                }
            }));
        }
        let mut next = rng(0x9e37_79b9_7f4a_7c15);
        let mut figures: [Vec<f64>; 2] = Default::default();
        for _ in 0..9 {
            let before = counts();
            for _ in 0..cbs {
                walk(&ctx, latest.load(Ordering::Acquire), &mut next);
            }
            let spent = counts() - before;
            for (f, figure) in figures.iter_mut().enumerate() {
                figure.push(spent.0[f + 1] as f64 / cbs as f64);
            }
        }
        stop.store(true, Ordering::Relaxed);
        for t in threads {
            t.join().unwrap();
        }
        eprintln!(
            "{cbs} views over {live} objects, each naming {named} objects ({fresh} of ring 0's \
             latest) {} times{}, ring 0 making and removing every {spin} spins, {lanes} more \
             lanes: user-space instructions and cycles a command buffer, then a first lookup \
             (median, lowest, highest of 9)",
            again + 1,
            if at_once { ", found at once" } else { "" }
        );
        for (figure, name) in figures.iter_mut().zip(["instructions", "cycles"]) {
            let [median, low, high] = spread(figure);
            eprintln!(
                "  {name:<12} {median:9.0} ({low:.0} to {high:.0}) a command buffer, {:7.1} a \
                 first lookup",
                median / named as f64
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
