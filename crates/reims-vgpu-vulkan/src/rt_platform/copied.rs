//! The rt platform's copies of host pages, the floor of an import of them (`super::buffer`'s
//! docs, "A copy at the refusal point"): pages that rt imports neither by their pointer nor as a
//! dma-buf of their files, and pages whose GPU writes are dropped, are imported as a copy
//! ([`import`]), a Shared block of rt's own that holds their bytes.
//!
//! - **The copy.** A Shared block rt takes for it (rt's `take_block`), the pages' bytes at its
//!   start and zeros past them to the block's end. A texture over a buffer in it lies in place
//!   where the device's linear image takes the texture, as over any Shared buffer of rt's, not in
//!   an image of its own that rt copies the buffer's bytes into at each submission that uses it,
//!   as over an import (rt's `Buffer::new_texture`). The device answers the pages, not the copy,
//!   as a buffer's `contents` and as the block's mapping (`super::buffer::KnownBlocks`).
//! - **The base.** For each copy, the bytes the pages and the copy last agreed on, in the
//!   process's memory: at first the bytes copied.
//! - **Synced for the command buffers that use it** ([`Used`]), as Metal makes resident for a
//!   command buffer the resources its encoders set and those it declares, and no others: a
//!   buffer, a texture or a heap over the copy that an encoder sets (a buffer or texture bound, a
//!   blit's source or destination, an attachment of its pass, an index or tessellation factor
//!   buffer), or that it declares with `useResource:` or `useHeap:`, which Metal asks for a
//!   resource a shader reaches through an address (an argument buffer's). Each object over a copy
//!   holds it from its make, so a use costs one check; a command buffer that uses no copy syncs
//!   none. Each use notes whether it can write the copy: a buffer bound to a shader's slot, a
//!   blit's destination or fill, an attachment, a visibility result buffer, a heap, a texture of
//!   shader-write or atomic usage bound to a shader's slot and a resource declared for writing
//!   can; an index, indirect or tessellation factor buffer, a blit's source, another texture
//!   bound to a shader's slot and a resource declared for reading alone cannot.
//! - **And for every command buffer of its process once it escaped** ([`Used::escaped`]): a copy
//!   under an object whose identity left the guest's Metal with no message naming it (core's
//!   `EscapedCopies`: a buffer's `gpuAddress` read, written by an argument encoder, under a
//!   texture whose `gpuResourceID` was read, or added to a residency set; a heap from its make)
//!   may be reached through memory with no use naming it, so each commit of its process's command
//!   buffers, and of their parts, syncs it as a use that can write it, from the escape until the
//!   last such object goes. A copy no object of which escaped is reached through no address the
//!   guest holds.
//! - **Copied in at the commits that use it** ([`copy_in`], from `super::command`'s commit,
//!   before rt is handed the command buffer, so before the work that reads it): every byte of the
//!   pages that differs from the base is written into the copy and the base. A byte the CPU did
//!   not change is never written, so a GPU write to the copy by work still in flight stays. rt's
//!   memory of a Shared block is coherent, so the GPU reads the bytes written with no flush.
//! - **Merged back at the completions of the command buffers that can write it** ([`merging`],
//!   the first of the command buffer's completed handlers, so before core's and before a waiter
//!   returns): every byte of the copy that differs from the base is written into the pages where
//!   the pages' byte still is the base's, and the base takes the copy's byte. A byte both sides
//!   changed keeps the CPU's, which the next commit that uses it copies in, as Metal keeps the
//!   later of two writes. A byte the guest writes between the merge's read of it and its store is
//!   lost only where the GPU changed it too. A command buffer whose uses of a copy can only read
//!   it neither merges it back nor puts back its bytes: its work wrote none of them.
//! - **Compared a chunk at a time** (`CHUNK`, 4 KiB): a chunk of one side is copied out with a raw
//!   copy, as memory the guest writes at any time is read, and compared with the base's, and only
//!   a chunk that differs is compared a word and then a byte at a time, and stored byte by byte
//!   where not all of a word's bytes change.
//! - **Writes dropped** where the copy is of pages whose GPU writes Metal drops ([`import`]'s
//!   `drops`: pages their process maps read-only, `super::buffer`'s docs, "Read-only memory"): at
//!   each completion of a command buffer that can write it every byte the GPU changed in the copy
//!   is put back from the base instead, and nothing is ever written into the pages, so the CPU and
//!   the command buffers committed after the completion read the pages' bytes, as the M3 Max, the
//!   5300M and the UHD 630 answer (`crate::gpu::command`'s docs). Within a command buffer, and
//!   between command buffers in flight at once, a read after the GPU's write sees the write until
//!   the completion that drops it.
//! - **Unchanging pages** ([`Kind::Unchanging`]): a copy of pages whose GPU writes are dropped
//!   that are a file's own on a volume mounted read-only, which no one writes while the block
//!   lives (`RGFX_SURFACE_PAGES_UNCHANGING`, the guest's word, read once as the copy is made).
//!   Such a copy has no base and is never copied in: its bytes are the pages' as it was made. At
//!   each completion of a command buffer that can write it, the whole of the copy is written
//!   again from the pages, which drops the GPU's writes as above. Where the guest's word was
//!   false (a process that wrote the pages after all), the GPU reads the bytes the pages held as
//!   the copy was made, or at the last such completion, until the next one: documented Metal says
//!   nothing of a GPU's reads of memory a process writes after it said it would not.
//! - **Released** as the last of its holders lets go of it: the objects over it, the command
//!   buffers that used it, until their completed handlers ran, the device's known blocks while
//!   its block is known, and its process's escaped copies. Then its import's release runs (the
//!   buffer's deallocator among it), with no sync in progress, and its block goes back to rt,
//!   which frees it once no buffer of rt's over it and no command buffer that used it remains. A
//!   copy just made is held by [`import`] too, until the make that asked for it holds it
//!   ([`Fresh`]).
//!
//! Counted: the bytes written into copies ([`IN_BYTES_CODE`]) and into pages ([`BACK_BYTES_CODE`]),
//! the bytes of the GPU's writes dropped ([`DROPPED_BYTES_CODE`]), the bytes of unchanging pages
//! written into their copies again ([`RESTORED_BYTES_CODE`]), the render passes that store into a
//! copy whose writes are dropped ([`STORE_OUTS_CODE`]), and the nanoseconds the syncs took
//! ([`SYNC_NS_CODE`]); and as each copy of pages whose writes are dropped is made, its KiB, by
//! whether the pages are unchanging and how their process maps them
//! ([`UNCHANGING_KIB_CODE`], [`UNCHANGING_COW_KIB_CODE`], [`READ_ONLY_KIB_CODE`]).

use std::collections::BTreeMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

use metal2vulkan_runtime as rt;

use crate::core::num::{ToU64 as _, ToUsize as _};
use crate::core::platform::blit::ResourceRef;
use crate::core::platform::device::NoCopyDeallocator;
use crate::core::platform::os::Clock as _;
use crate::core::platform::os::clock::MonoTime;
use crate::core::platform::values::PagesAccess;
use crate::core::sync::Mutex;

use super::RtPlatform;
use super::buffer::RtBuffer;
use super::heap::RtHeap;

/// The code whose count is the bytes the CPU changed in copied pages that commits wrote into their
/// copies ([`copy_in`]).
pub const IN_BYTES_CODE: &str = "rt-copied-in-bytes";

/// The code whose count is the bytes the GPU changed in copies that completions wrote into their
/// pages ([`merging`]).
pub const BACK_BYTES_CODE: &str = "rt-copied-back-bytes";

/// The code whose count is the bytes the GPU changed in copies whose writes are dropped that
/// completions put back from their bases ([`merging`], the module's docs, "Writes dropped").
pub const DROPPED_BYTES_CODE: &str = "rt-copied-dropped-bytes";

/// The code whose count is the bytes of unchanging pages that completions wrote into their copies
/// again, putting back the GPU's writes ([`merging`], the module's docs, "Unchanging pages").
pub const RESTORED_BYTES_CODE: &str = "rt-copied-restored-bytes";

/// The code whose count is the KiB of the copies made of unchanging pages whose process maps them
/// with no write in their maximum protection ([`import`], [`Kind::Unchanging`]).
pub const UNCHANGING_KIB_CODE: &str = "rt-copied-unchanging-kib";

/// The code whose count is the KiB of the copies made of unchanging pages whose process maps them
/// private and copy-on-write ([`import`], [`Kind::Unchanging`]'s `copy_on_write`).
pub const UNCHANGING_COW_KIB_CODE: &str = "rt-copied-unchanging-cow-kib";

/// The code whose count is the KiB of the copies made of pages whose GPU writes are dropped and
/// which may change ([`import`], [`Kind::Drops`]).
pub const READ_ONLY_KIB_CODE: &str = "rt-copied-read-only-kib";

/// The code whose count is the render passes with an attachment over a copy whose writes are
/// dropped ([`store_outs`]): the 5300M and the UHD 630 complete such a pass, its store dropped
/// with the copy's other writes, as rt's does; the M3 Max faults it (`crate::gpu::command`'s docs).
pub const STORE_OUTS_CODE: &str = "rt-copied-store-outs";

/// The code whose count is the nanoseconds the copies' syncs took, on the committing thread and
/// on the thread that runs the command buffers' completed handlers.
pub const SYNC_NS_CODE: &str = "rt-copied-sync-cpu-ns";

/// The bytes a sync compares at a time.
const CHUNK: usize = 4096;

/// A chunk's 8-byte words.
const WORDS: usize = CHUNK / 8;

/// `MTLResourceUsageRead`.
const USAGE_READ: u64 = 1;

/// `MTLResourceUsageSample`.
const USAGE_SAMPLE: u64 = 4;

/// The live copies, by their blocks (the module's docs, "Released").
static COPIES: Mutex<BTreeMap<rt::BlockId, Entry>> = Mutex::new(BTreeMap::new());

/// How many copies are live: none, and a commit or a buffer's `contents` asks nothing more.
static LIVE: AtomicUsize = AtomicUsize::new(0);

/// A live copy as [`COPIES`] keeps it: as its holders hold it, and a copy just made held here too
/// until the make that asked for it holds it ([`Fresh`]). No copy is dropped while [`COPIES`] is
/// locked, since its drop locks it.
struct Entry {
    /// The copy, while a holder holds it.
    copy: Weak<Copied>,
    /// The copy, until the make that asked for it holds it.
    fresh: Option<Arc<Copied>>,
}

/// What a copy does with the GPU's writes to it and with the CPU's to its pages (the module's
/// docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    /// The CPU's writes are copied in at the commits, and the GPU's merged back at the
    /// completions ("Copied in", "Merged back").
    Merges,
    /// The CPU's writes are copied in at the commits, and the GPU's dropped at the completions
    /// ("Writes dropped").
    Drops,
    /// The pages never change, and the GPU's writes are dropped at the completions ("Unchanging
    /// pages"); `copy_on_write` where their process maps them private and copy-on-write.
    Unchanging {
        /// The process maps the pages private and copy-on-write.
        copy_on_write: bool,
    },
}

impl Kind {
    /// The kind of a copy of pages their process maps as `access`.
    pub(super) fn of(access: PagesAccess) -> Kind {
        match access {
            PagesAccess::Writable => Kind::Merges,
            PagesAccess::ReadOnly => Kind::Drops,
            PagesAccess::Unchanging { copy_on_write } => Kind::Unchanging { copy_on_write },
        }
    }

    /// Whether the GPU's writes to the copy are dropped.
    pub(super) fn drops(self) -> bool {
        self != Kind::Merges
    }
}

/// Why no copy was made ([`import`]).
#[derive(Debug)]
pub(super) enum Uncopied {
    /// The process has no memory for the copy's base.
    NoMemory,
    /// rt refused the copy's block.
    Rt(rt::Error),
    /// rt answered no mapping of the copy's block.
    Unmapped,
}

impl std::fmt::Display for Uncopied {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Uncopied::NoMemory => f.write_str("no host memory for the copy's base"),
            Uncopied::Rt(error) => {
                write!(f, "metal2vulkan-runtime refused the copy's block: {error}")
            }
            Uncopied::Unmapped => {
                f.write_str("metal2vulkan-runtime mapped none of the copy's block")
            }
        }
    }
}

/// A copy of the `len` bytes of whole pages at `pages`, made on `device` (the module's docs), of
/// `kind`: its block, the pages' first byte at its start, the GPU's writes to it dropped at each
/// completion where the kind drops them ("Writes dropped", "Unchanging pages"), counted with its
/// KiB where it does. `release` runs as the copy goes ("Released"), after it stopped syncing. The
/// copy is held here until the caller's [`Fresh`] of the block goes, by which time an object the
/// caller made over it holds it ([`of`]). Else why none was made, `release` dropped unrun.
///
/// # Safety
///
/// The `len` bytes at `pages` are mapped readable until `release` runs, and writable too unless
/// the kind drops the GPU's writes.
pub(super) unsafe fn import(
    device: &rt::Device,
    (pages, len): (NonNull<u8>, u64),
    kind: Kind,
    release: NoCopyDeallocator,
) -> Result<rt::Block, Uncopied> {
    let block = (device.take_block(rt::MemoryClass::Shared, len)).map_err(Uncopied::Rt)?;
    let copy = block.contents.ok_or(Uncopied::Unmapped)?;
    let (id, taken, cap) = (block.id, block.clone(), block.len.to_usize());
    // SAFETY: the caller's; the block's `cap` bytes from `copy` are rt's mapping of it, 64 KiB
    // aligned and taken for the copy alone, mapped while a handle of it lives, which the copy
    // holds.
    let copied = unsafe { Copied::new(pages, len.to_usize(), kind, (copy, cap, Some(block))) };
    let copied = copied.ok_or(Uncopied::NoMemory)?;
    let code = match kind {
        Kind::Merges => None,
        Kind::Drops => Some(READ_ONLY_KIB_CODE),
        Kind::Unchanging {
            copy_on_write: false,
        } => Some(UNCHANGING_KIB_CODE),
        Kind::Unchanging {
            copy_on_write: true,
        } => Some(UNCHANGING_COW_KIB_CODE),
    };
    if let Some(code) = code {
        crate::answer::count_kind_by(code, len.div_ceil(1024));
    }
    *copied.release.lock() = Some(release);
    let copied = Arc::new(copied);
    let entry = Entry {
        copy: Arc::downgrade(&copied),
        fresh: Some(copied),
    };
    let replaced = COPIES.lock().insert(id, entry);
    drop(replaced);
    LIVE.fetch_add(1, Ordering::Release);
    Ok(taken)
}

/// Whether a copy is live.
pub(super) fn any() -> bool {
    LIVE.load(Ordering::Acquire) > 0
}

/// Whether the block `id` is a live copy whose GPU writes are dropped ("Writes dropped").
pub(super) fn drops(id: rt::BlockId) -> bool {
    of(id).is_some_and(|copied| copied.kind.drops())
}

/// The live copy whose block is `id`, for an object made over it to hold ("Synced for the command
/// buffers that use it"); `None` where the block is no live copy.
pub(super) fn of(id: rt::BlockId) -> Option<Arc<Copied>> {
    if !any() {
        return None;
    }
    let copies = COPIES.lock();
    copies.get(&id).and_then(|entry| entry.copy.upgrade())
}

/// [`import`]'s hold of the copy just made in the block it names, which it lets go of as it
/// drops: the make that asked for the copy holds it by then where it made an object over it, and
/// else the copy goes with it ("Released").
pub(super) struct Fresh(pub(super) rt::BlockId);

impl Drop for Fresh {
    fn drop(&mut self) {
        if !any() {
            return;
        }
        let fresh = (COPIES.lock().get_mut(&self.0)).and_then(|entry| entry.fresh.take());
        drop(fresh);
    }
}

/// A copy a command buffer uses, and whether a use of it can write it ([`Used`]).
#[derive(Clone)]
pub(super) struct CopyUse {
    /// The copy.
    copied: Arc<Copied>,
    /// Whether a use of it can write it.
    writes: bool,
}

/// The copies a command buffer uses, each once, with whether a use of it can write it ("Synced
/// for the command buffers that use it"), noted by its encoders as they set or declare an object
/// over one, and taken at its commit.
#[derive(Clone, Default)]
pub struct Used(Arc<Mutex<Vec<CopyUse>>>);

impl Used {
    /// The command buffer uses `copied`, where it is a copy (an object over none holds `None`), in
    /// a use that can write it where `writes`.
    pub(super) fn add(&self, copied: Option<&Arc<Copied>>, writes: bool) {
        let Some(copied) = copied else {
            return;
        };
        let mut used = self.0.lock();
        match used.iter_mut().find(|u| Arc::ptr_eq(&u.copied, copied)) {
            Some(u) => u.writes |= writes,
            None => used.push(CopyUse {
                copied: Arc::clone(copied),
                writes,
            }),
        }
    }

    /// The command buffer uses `buffer` where its work can write it (bound to a shader's slot, a
    /// blit's destination or fill): its copy, where it lies over one.
    pub(super) fn buffer(&self, buffer: &RtBuffer) {
        self.add(buffer.copied(), true);
    }

    /// The command buffer's work reads `buffer` alone (as indices, as an indirect or tessellation
    /// factor buffer, as a blit's source): its copy, where it lies over one.
    pub(super) fn buffer_read(&self, buffer: &RtBuffer) {
        self.add(buffer.copied(), false);
    }

    /// The command buffer uses `heap`, whose resources its work can write: its copy, where it lies
    /// over one.
    pub(super) fn heap(&self, heap: &RtHeap) {
        self.add(heap.copied(), true);
    }

    /// The command buffer declares `resource` for `usage` (`useResource:usage:`): its copy, where
    /// it lies over one, in a use that can write it unless `usage` is reads alone
    /// (`MTLResourceUsageRead`, `Sample`).
    pub(super) fn resource(&self, resource: &ResourceRef<'_, RtPlatform>, usage: u64) {
        let reads_alone = usage != 0 && usage & !(USAGE_READ | USAGE_SAMPLE) == 0;
        match resource {
            ResourceRef::Buffer(buffer) => self.add(buffer.copied(), !reads_alone),
            ResourceRef::Texture(texture) => self.add(texture.copied(), !reads_alone),
            ResourceRef::Icb(_) => {}
        }
    }

    /// The command buffer is of the process whose escaped copies are `escapes`: it uses each
    /// copy `escapes` holds now, in a use that can write it ("And for every command buffer of its
    /// process once it escaped").
    pub(super) fn escaped(&self, escapes: &crate::core::platform::command::EscapedCopies) {
        for copy in escapes.copies() {
            if let Ok(copied) = copy.downcast::<Copied>() {
                self.add(Some(&copied), true);
            }
        }
    }

    /// The copies used, taken: none is used since.
    pub(super) fn take(&self) -> Vec<CopyUse> {
        std::mem::take(&mut *self.0.lock())
    }
}

/// Whether a use of any of `copies`, those a command buffer uses, can write it: the command
/// buffer's completion then merges them back ([`merging`]).
pub(super) fn writes(copies: &[CopyUse]) -> bool {
    copies.iter().any(|u| u.writes)
}

/// Counts a render pass `sel` made whose attachments lie in the blocks `blocks` where one is a
/// live copy whose GPU writes are dropped ([`STORE_OUTS_CODE`]); the process's first with a line.
pub(super) fn store_outs(sel: &'static str, mut blocks: impl Iterator<Item = rt::BlockId>) {
    if !any() || !blocks.any(drops) {
        return;
    }
    crate::answer::count_kind(STORE_OUTS_CODE);
    if !STORE_OUT_SAID.swap(true, Ordering::Relaxed) {
        log::warn!(
            target: super::TARGET,
            "{sel}: a render pass stores into a texture over pages whose GPU writes are dropped; \
             the store is dropped with them, as the 5300M and the UHD 630 drop it, where the M3 \
             Max's GPU faults it ({STORE_OUTS_CODE}), counted from here"
        );
    }
}

/// Whether the process logged its first render pass over a copy whose writes are dropped.
static STORE_OUT_SAID: AtomicBool = AtomicBool::new(false);

/// Writes into each of `copies`, those a command buffer uses, the bytes the CPU changed in its
/// pages since they last agreed (the module's docs): at its commit, before rt is handed it.
/// Nothing where it uses none.
pub(super) fn copy_in(copies: &[CopyUse]) {
    if copies.is_empty() {
        return;
    }
    let clock = crate::os::OsClock::new();
    let began = clock.now_mono();
    let written: u64 = copies.iter().map(|u| u.copied.copy_in()).sum();
    synced(IN_BYTES_CODE, written, &clock, began);
}

/// The completed handler, to run first of a command buffer's, that writes back into the pages
/// of each of `copies`, those it uses, that a use of it can write the bytes the GPU changed in
/// the copy, or puts back from the base those of one whose writes are dropped, or writes the
/// whole of one of unchanging pages again from them (the module's docs), and that holds every
/// one of `copies` until it ran ("Released"); `None` where it uses none.
pub(super) fn merging(
    copies: &[CopyUse],
) -> Option<impl FnOnce(&rt::CommandBuffer) + Send + 'static> {
    let copies = copies.to_vec();
    (!copies.is_empty()).then_some(move |_: &rt::CommandBuffer| merge_back(&copies))
}

/// Writes back into the pages of each of `copies` that a use of it can write the bytes the GPU
/// changed in the copy, puts back from its base those of a copy whose writes are dropped, and
/// writes the whole of a copy of unchanging pages again from them; of the others, nothing.
fn merge_back(copies: &[CopyUse]) {
    if !writes(copies) {
        return;
    }
    let clock = crate::os::OsClock::new();
    let began = clock.now_mono();
    let (mut written, mut dropped, mut restored) = (0, 0, 0);
    for u in copies.iter().filter(|u| u.writes) {
        match u.copied.kind {
            Kind::Merges => written += u.copied.merge_back(),
            Kind::Drops => dropped += u.copied.drop_back(),
            Kind::Unchanging { .. } => restored += u.copied.restore(),
        }
    }
    if dropped > 0 {
        crate::answer::count_kind_by(DROPPED_BYTES_CODE, dropped);
    }
    if restored > 0 {
        crate::answer::count_kind_by(RESTORED_BYTES_CODE, restored);
    }
    synced(BACK_BYTES_CODE, written, &clock, began);
}

/// Counts a sync's `written` bytes as `code`, and its time since `began` on `clock`.
fn synced(code: &'static str, written: u64, clock: &crate::os::OsClock, began: MonoTime) {
    crate::answer::count_kind_by(code, written);
    let took = clock.now_mono().saturating_duration_since(began).as_nanos();
    crate::answer::count_kind_by(SYNC_NS_CODE, u64::try_from(took).unwrap_or(u64::MAX));
}

/// A copy of pages and its base (the module's docs).
pub struct Copied {
    /// The pages' first byte.
    home: NonNull<u8>,
    /// The copy's first byte.
    copy: NonNull<u8>,
    /// The base, a word per 8 bytes of the pages; none for a copy of unchanging pages, whose lock
    /// its writes of the pages' bytes take all the same.
    base: Mutex<Box<[u64]>>,
    /// What it does with the GPU's writes and the CPU's ([`Kind`]).
    kind: Kind,
    /// The bytes of the pages it copies.
    len: usize,
    /// The copy's memory, rt's Shared block taken for it alone ([`import`]), whose handle keeps
    /// it until the copy goes; `None` in the tests, whose memory is their own.
    block: Option<rt::Block>,
    /// What runs as the copy goes, after it stopped syncing ([`import`]'s `release`).
    release: Mutex<Option<NoCopyDeallocator>>,
}

// SAFETY: the pointers are to the pages, which stay mapped until the copy's release runs, and to
// the copy's memory, the copy's own, which its `block` keeps; each is read and written only under
// the base's lock, or by rt through its block.
unsafe impl Send for Copied {}
// SAFETY: as for `Send`.
unsafe impl Sync for Copied {}

impl Drop for Copied {
    /// No longer live, then its release (the module's docs, "Released"); its block goes back to
    /// rt after, with its handle. No sync runs: each holds the copy.
    fn drop(&mut self) {
        if let Some(block) = &self.block {
            let mut copies = COPIES.lock();
            let ours = (copies.get(&block.id)).is_some_and(|e| std::ptr::eq(e.copy.as_ptr(), self));
            let gone = ours.then(|| copies.remove(&block.id)).flatten();
            drop(copies);
            if gone.is_some() {
                LIVE.fetch_sub(1, Ordering::Release);
            }
        }
        let release = self.release.lock().take();
        if let Some(release) = release {
            release();
        }
    }
}

impl Copied {
    /// A copy of the `len` bytes at `home` in the memory `block` holds where given, whose first
    /// byte is `copy` and which holds `cap` bytes, the copy's bytes past `len` zeros, and its base,
    /// the bytes copied, of `kind` (one of unchanging pages with no base); `None` where `len` is
    /// 0, no whole number of chunks or past `cap`, or where the process has no memory for the
    /// base.
    ///
    /// # Safety
    ///
    /// The `len` bytes at `home` are mapped readable until the copy goes, and writable too unless
    /// `kind` drops the GPU's writes; the `cap` bytes at `copy`, 8-aligned, are memory of the
    /// copy's alone, mapped readable and writable until the copy goes.
    unsafe fn new(
        home: NonNull<u8>,
        len: usize,
        kind: Kind,
        (copy, cap, block): (NonNull<u8>, usize, Option<rt::Block>),
    ) -> Option<Copied> {
        if len == 0 || !len.is_multiple_of(CHUNK) || len > cap {
            return None;
        }
        let mut base = Vec::new();
        let words = match kind {
            Kind::Unchanging { .. } => 0,
            Kind::Merges | Kind::Drops => len / 8,
        };
        base.try_reserve_exact(words).ok()?;
        base.resize(words, 0_u64);
        // SAFETY: the pages' `len` bytes are readable, and the copy's `cap` bytes, then the
        // base's `words` words, are memory of the copy's own (the caller's).
        unsafe {
            std::ptr::copy_nonoverlapping(home.as_ptr(), copy.as_ptr(), len);
            copy.as_ptr().add(len).write_bytes(0, cap - len);
            let to = base.as_mut_ptr().cast::<u8>();
            std::ptr::copy_nonoverlapping(copy.as_ptr(), to, 8 * words);
        }
        Some(Copied {
            home,
            copy,
            base: Mutex::new(base.into_boxed_slice()),
            kind,
            len,
            block,
            release: Mutex::new(None),
        })
    }

    /// Writes into the copy the bytes of the pages that differ from the base, and the base takes
    /// them: the bytes written. Nothing for a copy of unchanging pages ("Unchanging pages").
    fn copy_in(&self) -> u64 {
        if matches!(self.kind, Kind::Unchanging { .. }) {
            return 0;
        }
        let mut base = self.base.lock();
        let mut now = [0_u64; WORDS];
        let mut written = 0;
        for (chunk, was) in base.as_chunks_mut::<WORDS>().0.iter_mut().enumerate() {
            let at = chunk * CHUNK;
            // SAFETY: the chunk lies in the pages, mapped until the copy goes (`new`'s contract).
            unsafe { read_chunk(self.home.as_ptr().add(at), &mut now) };
            if now == *was {
                continue;
            }
            for (i, (now, was)) in now.iter().zip(was.iter_mut()).enumerate() {
                if now != was {
                    // SAFETY: the word lies in the copy's first bytes, the length of the pages,
                    // 8-aligned as the copy is.
                    let to = unsafe { self.copy.as_ptr().add(at + 8 * i) };
                    // SAFETY: as above.
                    written += unsafe { store(to, *now, *was, None) };
                    *was = *now;
                }
            }
        }
        written
    }

    /// Writes into the pages the bytes of the copy that differ from the base where the pages'
    /// still are the base's, and the base takes the copy's: the bytes written.
    fn merge_back(&self) -> u64 {
        let mut base = self.base.lock();
        let (mut gpu, mut cpu) = ([0_u64; WORDS], [0_u64; WORDS]);
        let mut written = 0;
        for (chunk, was) in base.as_chunks_mut::<WORDS>().0.iter_mut().enumerate() {
            let at = chunk * CHUNK;
            // SAFETY: the chunk lies in the copy's first bytes, the length of the pages.
            unsafe { read_chunk(self.copy.as_ptr().add(at), &mut gpu) };
            if gpu == *was {
                continue;
            }
            // SAFETY: the chunk lies in the pages, mapped until the copy goes.
            unsafe { read_chunk(self.home.as_ptr().add(at), &mut cpu) };
            let words = gpu.iter().zip(cpu.iter()).zip(was.iter_mut());
            for (i, ((gpu, cpu), was)) in words.enumerate() {
                if gpu != was {
                    // SAFETY: the word lies in the pages, 8-aligned as they are.
                    let to = unsafe { self.home.as_ptr().add(at + 8 * i) };
                    // SAFETY: as above.
                    written += unsafe { store(to, *gpu, *was, Some(*cpu)) };
                    *was = *gpu;
                }
            }
        }
        written
    }

    /// Puts back into the copy from the base the bytes of the copy that differ from it, the
    /// GPU's writes, which are dropped ("Writes dropped"); the pages are never written: the
    /// bytes put back.
    fn drop_back(&self) -> u64 {
        let base = self.base.lock();
        let mut gpu = [0_u64; WORDS];
        let mut dropped = 0;
        for (chunk, was) in base.as_chunks::<WORDS>().0.iter().enumerate() {
            let at = chunk * CHUNK;
            // SAFETY: the chunk lies in the copy's first bytes, the length of the pages.
            unsafe { read_chunk(self.copy.as_ptr().add(at), &mut gpu) };
            if gpu == *was {
                continue;
            }
            for (i, (gpu, was)) in gpu.iter().zip(was.iter()).enumerate() {
                if gpu != was {
                    // SAFETY: the word lies in the copy's first bytes, 8-aligned as the copy is.
                    let to = unsafe { self.copy.as_ptr().add(at + 8 * i) };
                    // SAFETY: as above.
                    dropped += unsafe { store(to, *was, *gpu, None) };
                }
            }
        }
        dropped
    }

    /// Writes the whole of the pages into the copy again, the GPU's writes to a copy of unchanging
    /// pages put back ("Unchanging pages"); the pages are never written: the bytes written.
    fn restore(&self) -> u64 {
        let _held = self.base.lock();
        // SAFETY: the pages' `len` bytes are readable until the copy goes (`new`'s contract), and
        // the copy's first `len` bytes its own; a raw copy, as memory others write at any time is
        // read.
        unsafe { std::ptr::copy_nonoverlapping(self.home.as_ptr(), self.copy.as_ptr(), self.len) };
        self.len.to_u64()
    }
}

/// The chunk at `from` read into `into` with a raw copy, as memory others write at any time is
/// read.
///
/// # Safety
///
/// `from` is `CHUNK` readable bytes.
unsafe fn read_chunk(from: *const u8, into: &mut [u64; WORDS]) {
    // SAFETY: the caller's, and `into` is `CHUNK` bytes of the caller's own.
    unsafe { std::ptr::copy_nonoverlapping(from, into.as_mut_ptr().cast::<u8>(), CHUNK) };
}

/// Writes into the 8 bytes at `to` each byte of `new` that differs from `was`'s, and where `other`
/// is given only those whose byte of `other` still is `was`'s: in one store where all 8 are, else
/// byte by byte. The bytes written.
///
/// # Safety
///
/// `to` is 8 writable bytes, 8-aligned.
unsafe fn store(to: *mut u8, new: u64, was: u64, other: Option<u64>) -> u64 {
    let (n, w) = (new.to_ne_bytes(), was.to_ne_bytes());
    let o = other.map(u64::to_ne_bytes);
    let writes = |j: usize| n[j] != w[j] && o.is_none_or(|o| o[j] == w[j]);
    if (0..8).all(&writes) {
        // SAFETY: the caller's.
        unsafe { to.cast::<u64>().write(new) };
        return 8;
    }
    let mut written = 0;
    for j in (0..8).filter(|&j| writes(j)) {
        // SAFETY: the caller's; `j` is below 8.
        unsafe { to.add(j).write(n[j]) };
        written += 1;
    }
    written
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `n` chunks of the process's own memory, 8-aligned.
    fn chunks(n: usize) -> Box<[u64]> {
        vec![0_u64; n * WORDS].into_boxed_slice()
    }

    /// The bytes at `at`, `len` of them.
    fn bytes(at: NonNull<u8>, len: usize) -> Vec<u8> {
        // SAFETY: the callers' pointers are a copy's or the test's pages, at least `len` long.
        unsafe { std::slice::from_raw_parts(at.as_ptr(), len) }.to_vec()
    }

    /// The bytes past the copy's first `len` in its memory, as [`copy_of`] makes it.
    const TAIL: usize = 64;

    /// A copy of the `len` bytes at `home` in memory of the test's own, never freed, [`TAIL`]
    /// bytes of 0xee past them, of `kind` (`Copied::new`).
    ///
    /// # Safety
    ///
    /// As `Copied::new`'s for `home`.
    unsafe fn copy_of(home: NonNull<u8>, len: usize, kind: Kind) -> Option<Copied> {
        let memory = vec![0xeeee_eeee_eeee_eeee_u64; (len + TAIL) / 8].leak();
        let cap = memory.len() * 8;
        let copy = NonNull::new(memory.as_mut_ptr().cast::<u8>()).expect("the memory");
        // SAFETY: the caller's for `home`; the memory is the copy's alone, never freed, 8-aligned.
        unsafe { Copied::new(home, len, kind, (copy, cap, None)) }
    }

    /// A copy holds its pages' bytes and zeros past them; a commit's copy in writes the bytes the
    /// CPU changed and leaves the copy's others, which the GPU may have written meanwhile; a
    /// completion's merge back writes the bytes the GPU changed where the CPU did not, a byte both
    /// changed keeping the CPU's, which the next copy in takes; each counts the bytes it wrote.
    #[test]
    fn a_copy_syncs_the_bytes_each_side_changed() {
        let mut pages = chunks(2);
        let home = NonNull::new(pages.as_mut_ptr().cast::<u8>()).expect("the pages");
        let len = 2 * CHUNK;
        // SAFETY: the pages are the test's own, written below only through `home`.
        unsafe { home.as_ptr().write_bytes(0x11, len) };
        // SAFETY: the pages live until the test's end, past the copy.
        let copied = unsafe { copy_of(home, len, Kind::Merges) }.expect("a copy");
        assert_eq!(bytes(copied.copy, len), vec![0x11; len]);
        // SAFETY: the copy's memory holds `TAIL` bytes past the pages'.
        let tail = bytes(unsafe { copied.copy.add(len) }, TAIL);
        assert_eq!(tail, vec![0; TAIL], "zeros past the pages' bytes");
        assert_eq!((copied.copy_in(), copied.merge_back()), (0, 0));

        // The CPU writes 3 bytes of the pages; the GPU, meanwhile, a byte of the copy beside one.
        // SAFETY: inside the pages and the copy.
        unsafe {
            home.as_ptr().add(5).write_bytes(0x22, 3);
            copied.copy.as_ptr().add(8).write(0x33);
        }
        assert_eq!(copied.copy_in(), 3);
        let copy = bytes(copied.copy, 16);
        assert_eq!(copy[..5], [0x11; 5]);
        assert_eq!(copy[5..8], [0x22; 3]);
        assert_eq!(copy[8..], [0x33, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11]);

        // The GPU writes a whole word in the second chunk and a byte the CPU also writes; the CPU
        // keeps its byte, the pages take the GPU's others, and the next copy in takes the CPU's.
        let far = CHUNK + 64;
        // SAFETY: inside the pages and the copy.
        unsafe {
            copied.copy.as_ptr().add(far).write_bytes(0x44, 8);
            copied.copy.as_ptr().add(far + 8).write(0x55);
            home.as_ptr().add(far + 8).write(0x66);
        }
        assert_eq!(copied.merge_back(), 1 + 8);
        let back = bytes(home, len);
        assert_eq!(back[8], 0x33);
        assert_eq!(back[far..far + 8], [0x44; 8]);
        assert_eq!(back[far + 8], 0x66);
        assert_eq!(copied.copy_in(), 1);
        assert_eq!(bytes(copied.copy, len), back);
        assert_eq!((copied.copy_in(), copied.merge_back()), (0, 0));
    }

    /// A copy whose writes are dropped never writes its pages: a completion puts back from the
    /// base every byte the GPU changed in the copy, a byte both sides changed among them, counting
    /// them, and the next commit's copy in takes the CPU's bytes.
    #[test]
    fn a_dropping_copy_puts_the_gpus_bytes_back_and_never_writes_its_pages() {
        let mut pages = chunks(2);
        let home = NonNull::new(pages.as_mut_ptr().cast::<u8>()).expect("the pages");
        let len = 2 * CHUNK;
        // SAFETY: the pages are the test's own, written below only through `home`.
        unsafe { home.as_ptr().write_bytes(0x11, len) };
        // SAFETY: the pages live until the test's end, past the copy.
        let copied = unsafe { copy_of(home, len, Kind::Drops) }.expect("a copy");
        assert!(copied.kind.drops());

        // The GPU writes a word in the first chunk and a byte in the second, which the CPU also
        // writes, and the CPU 2 bytes elsewhere.
        let far = CHUNK + 64;
        // SAFETY: inside the pages and the copy.
        unsafe {
            copied.copy.as_ptr().add(16).write_bytes(0x44, 8);
            copied.copy.as_ptr().add(far).write(0x55);
            home.as_ptr().add(far).write(0x66);
            home.as_ptr().add(32).write_bytes(0x22, 2);
        }
        assert_eq!((copied.drop_back(), copied.drop_back()), (8 + 1, 0));
        let mut expect = vec![0x11; len];
        expect[far] = 0x66;
        expect[32..34].fill(0x22);
        assert_eq!(
            bytes(home, len),
            expect,
            "the pages hold only the CPU's bytes"
        );
        assert_eq!(bytes(copied.copy, len), vec![0x11; len], "the base's bytes");
        assert_eq!(copied.copy_in(), 1 + 2);
        assert_eq!(bytes(copied.copy, len), expect);
        assert_eq!((copied.copy_in(), copied.drop_back()), (0, 0));
    }

    /// A command buffer's completion merges back, or puts back, only the copies a use of which
    /// can write them: one whose uses all read it keeps the bytes the GPU changed in it (none, in
    /// Metal's run of such work) and its pages theirs, whether or not its writes are dropped; a
    /// use that can write a copy noted after reads of it makes its uses writing.
    #[test]
    fn a_completion_merges_only_the_copies_its_uses_can_write() {
        let len = CHUNK;
        let mut made = Vec::new();
        for kind in [Kind::Merges, Kind::Drops, Kind::Merges] {
            let mut pages = chunks(1);
            let home = NonNull::new(pages.as_mut_ptr().cast::<u8>()).expect("the pages");
            // SAFETY: the pages are the test's own, written below only through `home`.
            unsafe { home.as_ptr().write_bytes(0x11, len) };
            // SAFETY: the pages live until the test's end, past the copy.
            let copied = Arc::new(unsafe { copy_of(home, len, kind) }.expect("a copy"));
            // SAFETY: inside the copy, as the GPU writes it.
            unsafe { copied.copy.as_ptr().write_bytes(0x44, 8) };
            made.push((pages, home, copied));
        }
        let used = Used::default();
        let [
            (_, read_home, read),
            (_, dropped_home, dropped),
            (_, written_home, written),
        ] = &made[..]
        else {
            unreachable!("three copies");
        };
        used.add(Some(read), false);
        used.add(Some(dropped), false);
        used.add(Some(written), false);
        used.add(Some(written), true);
        used.add(None, true);
        let copies = used.take();
        let uses: Vec<bool> = copies.iter().map(|u| u.writes).collect();
        assert_eq!(uses, [false, false, true], "each copy once");
        assert!(writes(&copies));
        assert!(used.take().is_empty(), "taken");
        merge_back(&copies);
        assert_eq!(bytes(*read_home, 8), [0x11; 8], "a read copy's pages");
        assert_eq!(
            bytes(read.copy, 8),
            [0x44; 8],
            "a read copy keeps its bytes"
        );
        assert_eq!(bytes(*dropped_home, 8), [0x11; 8]);
        assert_eq!(bytes(dropped.copy, 8), [0x44; 8], "nothing put back");
        assert_eq!(bytes(*written_home, 8), [0x44; 8], "merged back");
        assert!(!writes(&copies[..2]));
    }

    /// A copy's release runs as the copy goes, once.
    #[test]
    fn a_copy_s_release_runs_as_it_goes() {
        let mut pages = chunks(1);
        let home = NonNull::new(pages.as_mut_ptr().cast::<u8>()).expect("the pages");
        // SAFETY: the pages live until the test's end, past the copy.
        let copied = unsafe { copy_of(home, CHUNK, Kind::Merges) }.expect("a copy");
        let ran = Arc::new(AtomicUsize::new(0));
        let runs = Arc::clone(&ran);
        *copied.release.lock() = Some(Box::new(move || {
            runs.fetch_add(1, Ordering::SeqCst);
        }));
        let held = Arc::new(copied);
        let other = Arc::clone(&held);
        drop(held);
        assert_eq!(ran.load(Ordering::SeqCst), 0, "held still");
        drop(other);
        assert_eq!(ran.load(Ordering::SeqCst), 1);
    }

    /// A store writes the bytes that changed, and of those only where the other side's byte is
    /// still the old one: all 8 in one store, else byte by byte.
    #[test]
    fn a_store_writes_only_the_bytes_to_be_written() {
        let mut word = [0_u64; 1];
        let to = word.as_mut_ptr().cast::<u8>();
        let (was, new) = (0x0101_0101_0101_0101_u64, 0x0202_0202_0202_0202_u64);
        word[0] = was;
        // SAFETY: `to` is the word's 8 bytes, 8-aligned.
        assert_eq!(unsafe { store(to, new, was, None) }, 8);
        assert_eq!(word[0], new);
        let mut changed = was.to_ne_bytes();
        changed[3] = 0x09;
        let other = u64::from_ne_bytes(changed);
        word[0] = other;
        // SAFETY: as above.
        assert_eq!(unsafe { store(to, new, was, Some(other)) }, 7);
        let mut kept = new.to_ne_bytes();
        kept[3] = 0x09;
        assert_eq!(word[0], u64::from_ne_bytes(kept));
    }

    /// A length of no whole number of chunks takes no copy, and nor does one past the memory.
    #[test]
    fn a_copy_is_of_whole_chunks() {
        let mut pages = chunks(1);
        let home = NonNull::new(pages.as_mut_ptr().cast::<u8>()).expect("the pages");
        // SAFETY: the pages are the test's own, past any copy.
        assert!(unsafe { copy_of(home, CHUNK / 2, Kind::Merges) }.is_none());
        // SAFETY: as above.
        assert!(unsafe { copy_of(home, 0, Kind::Merges) }.is_none());
        let mut memory = chunks(1);
        let copy = NonNull::new(memory.as_mut_ptr().cast::<u8>()).expect("the memory");
        let small = (copy, CHUNK / 2, None);
        // SAFETY: the pages are the test's own; the memory's first half is the copy's.
        assert!(unsafe { Copied::new(home, CHUNK, Kind::Merges, small) }.is_none());
    }

    /// A copy of unchanging pages keeps no base and is never copied in: its bytes are the pages'
    /// as it was made, whatever a process that said they would not change writes there since,
    /// until a completion of a command buffer that can write it, which writes the whole of the
    /// pages into it again, dropping the GPU's writes; the pages are never written. A completion
    /// whose uses only read it leaves it as it is.
    #[test]
    fn an_unchanging_copy_reads_its_pages_once_until_a_write_is_dropped() {
        let mut pages = chunks(2);
        let home = NonNull::new(pages.as_mut_ptr().cast::<u8>()).expect("the pages");
        let len = 2 * CHUNK;
        // SAFETY: the pages are the test's own, written below only through `home`.
        unsafe { home.as_ptr().write_bytes(0x11, len) };
        let kind = Kind::Unchanging {
            copy_on_write: true,
        };
        // SAFETY: the pages live until the test's end, past the copy.
        let copied = Arc::new(unsafe { copy_of(home, len, kind) }.expect("a copy"));
        assert!(copied.kind.drops());
        assert!(copied.base.lock().is_empty(), "no base");
        assert_eq!(bytes(copied.copy, len), vec![0x11; len]);

        // A process that said the pages would not change writes 3 bytes of them; the GPU, a word
        // of the copy.
        // SAFETY: inside the pages and the copy.
        unsafe {
            home.as_ptr().add(5).write_bytes(0x22, 3);
            copied.copy.as_ptr().add(CHUNK + 16).write_bytes(0x44, 8);
        }
        let read = Used::default();
        read.add(Some(&copied), false);
        let reads = read.take();
        copy_in(&reads);
        assert_eq!(copied.copy_in(), 0, "never copied in");
        merge_back(&reads);
        let mut gpu = vec![0x11; len];
        gpu[CHUNK + 16..CHUNK + 24].fill(0x44);
        assert_eq!(
            bytes(copied.copy, len),
            gpu,
            "the make's bytes and the GPU's"
        );

        let written = Used::default();
        written.add(Some(&copied), true);
        merge_back(&written.take());
        let mut cpu = vec![0x11; len];
        cpu[5..8].fill(0x22);
        assert_eq!(
            bytes(copied.copy, len),
            cpu,
            "the pages' bytes, the GPU's dropped"
        );
        assert_eq!(bytes(home, len), cpu, "the pages are never written");
        assert_eq!(copied.restore(), len.to_u64());
    }
}
