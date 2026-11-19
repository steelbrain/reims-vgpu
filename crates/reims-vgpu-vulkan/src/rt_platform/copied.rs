//! The rt platform's copies of host pages, the floor of an import of them (`super::buffer`'s
//! docs, "A copy at the refusal point"): pages that rt imports neither by their pointer nor as a
//! dma-buf of their files are imported as a copy ([`import`]), memory of the platform's own that
//! holds their bytes, which rt imports by its pointer.
//!
//! - **The copy.** Anonymous memory of the process's, `ALIGN`-aligned and a whole number of
//!   `ALIGN`s long (64 KiB, a multiple of every driver's `minImportedHostPointerAlignment` we
//!   met), the pages' bytes at its start; rt's block over it is Shared, and the device answers
//!   the pages, not the copy, as a buffer's `contents` and as the block's mapping
//!   (`super::buffer::KnownBlocks`).
//! - **The base.** For each copy, the bytes the pages and the copy last agreed on, in the
//!   process's memory: at first the bytes copied.
//! - **Copied in at each commit** ([`copy_in`], from `super::command`'s commit, before rt is
//!   handed the command buffer): every byte of the pages that differs from the base is written
//!   into the copy and the base, of every live copy, as a command buffer may reach any of them
//!   through an address. A byte the CPU did not change is never written, so a GPU write to the
//!   copy by work still in flight stays.
//! - **Merged back at each completion** ([`merging`], the first of the command buffer's completed
//!   handlers, so before core's and before a waiter returns): every byte of the copy that differs
//!   from the base is written into the pages where the pages' byte still is the base's, and the
//!   base takes the copy's byte. A byte both sides changed keeps the CPU's, which the next commit
//!   copies in, as Metal keeps the later of two writes. A byte the guest writes between the
//!   merge's read of it and its store is lost only where the GPU changed it too.
//! - **Compared a chunk at a time** (`CHUNK`, 4 KiB): a chunk of one side is copied out with a raw
//!   copy, as memory the guest writes at any time is read, and compared with the base's, and only
//!   a chunk that differs is compared a word and then a byte at a time, and stored byte by byte
//!   where not all of a word's bytes change.
//! - **Released** as rt frees the import: the copy stops syncing, waiting for a sync in progress,
//!   before the import's own release runs (the buffer's deallocator among it), and its memory goes
//!   with the last handle of it.
//!
//! Counted: the bytes written into copies ([`IN_BYTES_CODE`]) and into pages ([`BACK_BYTES_CODE`]),
//! and the nanoseconds the syncs took ([`SYNC_NS_CODE`]).

use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use metal2vulkan_rt as rt;

use crate::core::num::{ToU64 as _, ToUsize as _};
use crate::core::platform::device::NoCopyDeallocator;
use crate::core::sync::Mutex;

/// The code whose count is the bytes the CPU changed in copied pages that commits wrote into their
/// copies ([`copy_in`]).
pub const IN_BYTES_CODE: &str = "rt-copied-in-bytes";

/// The code whose count is the bytes the GPU changed in copies that completions wrote into their
/// pages ([`merging`]).
pub const BACK_BYTES_CODE: &str = "rt-copied-back-bytes";

/// The code whose count is the nanoseconds the copies' syncs took, on the committing thread and
/// on rt's completion thread.
pub const SYNC_NS_CODE: &str = "rt-copied-sync-cpu-ns";

/// The bytes a sync compares at a time.
const CHUNK: usize = 4096;

/// A chunk's 8-byte words.
const WORDS: usize = CHUNK / 8;

/// A copy's alignment, and what its length is a multiple of.
const ALIGN: usize = 64 << 10;

/// The live copies, in the order made.
static COPIES: Mutex<Vec<Arc<Copied>>> = Mutex::new(Vec::new());

/// How many copies are live: none, and a commit or a buffer's `contents` asks nothing more.
static LIVE: AtomicUsize = AtomicUsize::new(0);

/// Why no copy was imported ([`import`]).
#[derive(Debug)]
pub(super) enum Uncopied {
    /// The process has no memory for the copy or its base.
    NoMemory,
    /// rt refused the copy's import.
    Rt(rt::Error),
}

impl std::fmt::Display for Uncopied {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Uncopied::NoMemory => f.write_str("no host memory for the copy"),
            Uncopied::Rt(error) => write!(f, "metal2vulkan-rt refused the copy's import: {error}"),
        }
    }
}

/// rt's import of a copy of the `len` bytes of whole pages at `pages`, made on `device` (the
/// module's docs): the copy's block, the pages' first byte at its start. rt runs `release` once it
/// freed the import, after the copy stopped syncing. Else why none was made, `release` dropped
/// unrun.
///
/// # Safety
///
/// The `len` bytes at `pages` are mapped readable and writable until `release` runs.
pub(super) unsafe fn import(
    device: &rt::Device,
    (pages, len): (NonNull<u8>, u64),
    release: NoCopyDeallocator,
) -> Result<rt::Block, Uncopied> {
    // SAFETY: the caller's.
    let copied = unsafe { Copied::new(pages, len.to_usize()) }.ok_or(Uncopied::NoMemory)?;
    let copied = Arc::new(copied);
    let held = Arc::clone(&copied);
    let release: NoCopyDeallocator = Box::new(move || {
        unregister(&held);
        release();
    });
    let size = copied.layout.size().to_u64();
    // SAFETY: the copy's `size` bytes are the process's own memory, mapped readable and writable,
    // and freed only as the last handle of the copy goes, which the release holds until it runs.
    match unsafe { device.import_host_memory(copied.copy, size, Some(release)) } {
        Ok((block, _)) => {
            COPIES.lock().push(copied);
            LIVE.fetch_add(1, Ordering::Release);
            Ok(block)
        }
        Err(error) => Err(Uncopied::Rt(error)),
    }
}

/// Whether a copy is live.
pub(super) fn any() -> bool {
    LIVE.load(Ordering::Acquire) > 0
}

/// Writes into every live copy the bytes the CPU changed in its pages since they last agreed
/// (the module's docs): at a commit, before rt is handed the command buffer.
pub(super) fn copy_in() {
    if !any() {
        return;
    }
    let started = Instant::now();
    let copies = COPIES.lock().clone();
    let written: u64 = copies.iter().map(|copied| copied.copy_in()).sum();
    synced(IN_BYTES_CODE, written, started);
}

/// The completed handler that writes back into every live copy's pages the bytes the GPU changed
/// in the copy (the module's docs), to run first of a command buffer's; `None` while no copy is
/// live, as at the command buffer's commit.
pub(super) fn merging() -> Option<impl FnOnce(&rt::CommandBuffer) + Send + 'static> {
    any().then_some(|_: &rt::CommandBuffer| merge_back())
}

/// Writes back into every live copy's pages the bytes the GPU changed in the copy.
fn merge_back() {
    if !any() {
        return;
    }
    let started = Instant::now();
    let copies = COPIES.lock().clone();
    let written: u64 = copies.iter().map(|copied| copied.merge_back()).sum();
    synced(BACK_BYTES_CODE, written, started);
}

/// Counts a sync's `written` bytes as `code`, and its time since `started`.
fn synced(code: &'static str, written: u64, started: Instant) {
    crate::answer::count_kind_by(code, written);
    let took = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    crate::answer::count_kind_by(SYNC_NS_CODE, took);
}

/// `copied` released: no longer live, and once a sync of it in progress ends, never synced again.
fn unregister(copied: &Arc<Copied>) {
    let mut copies = COPIES.lock();
    if let Some(i) = copies.iter().position(|c| Arc::ptr_eq(c, copied)) {
        copies.swap_remove(i);
        LIVE.fetch_sub(1, Ordering::Release);
    }
    drop(copies);
    *copied.base.lock() = None;
}

/// A copy of pages and its base (the module's docs).
struct Copied {
    /// The pages' first byte.
    home: NonNull<u8>,
    /// The copy's first byte.
    copy: NonNull<u8>,
    /// The copy's memory.
    layout: Layout,
    /// The base, a word per 8 bytes of the pages; `None` once released, when nothing syncs it.
    base: Mutex<Option<Box<[u64]>>>,
}

// SAFETY: the pointers are to the pages, which stay mapped until the copy is released, and to the
// copy's memory, the copy's own; each is read and written only under the base's lock, or by rt
// through its import.
unsafe impl Send for Copied {}
// SAFETY: as for `Send`.
unsafe impl Sync for Copied {}

impl Drop for Copied {
    fn drop(&mut self) {
        // SAFETY: the copy's memory was allocated with this layout, and nothing reaches it now:
        // its import's release, or the refusal of its import, dropped the last handle of it.
        unsafe { std::alloc::dealloc(self.copy.as_ptr(), self.layout) };
    }
}

impl Copied {
    /// A copy of the `len` bytes at `home`, and its base, each those bytes; `None` where `len` is
    /// 0 or no whole number of chunks, or where the process has no memory for them.
    ///
    /// # Safety
    ///
    /// The `len` bytes at `home` are mapped readable and writable until the copy is released.
    unsafe fn new(home: NonNull<u8>, len: usize) -> Option<Copied> {
        if len == 0 || !len.is_multiple_of(CHUNK) {
            return None;
        }
        let layout = Layout::from_size_align(len.checked_next_multiple_of(ALIGN)?, ALIGN).ok()?;
        // SAFETY: the layout's size is not 0.
        let copy = NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) })?;
        let copied = Copied {
            home,
            copy,
            layout,
            base: Mutex::new(None),
        };
        let mut base = Vec::new();
        base.try_reserve_exact(len / 8).ok()?;
        base.resize(len / 8, 0_u64);
        // SAFETY: the pages' `len` bytes are readable (the caller's), and the copy's first `len`
        // bytes, then the base's, are memory of the copy's own.
        unsafe {
            std::ptr::copy_nonoverlapping(home.as_ptr(), copy.as_ptr(), len);
            let to = base.as_mut_ptr().cast::<u8>();
            std::ptr::copy_nonoverlapping(copy.as_ptr(), to, len);
        }
        *copied.base.lock() = Some(base.into_boxed_slice());
        Some(copied)
    }

    /// Writes into the copy the bytes of the pages that differ from the base, and the base takes
    /// them: the bytes written.
    fn copy_in(&self) -> u64 {
        let mut base = self.base.lock();
        let Some(base) = base.as_deref_mut() else {
            return 0;
        };
        let mut now = [0_u64; WORDS];
        let mut written = 0;
        for (chunk, was) in base.chunks_exact_mut(WORDS).enumerate() {
            let at = chunk * CHUNK;
            // SAFETY: the chunk lies in the pages, mapped until the copy is released (`new`'s
            // contract).
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
        let Some(base) = base.as_deref_mut() else {
            return 0;
        };
        let (mut gpu, mut cpu) = ([0_u64; WORDS], [0_u64; WORDS]);
        let mut written = 0;
        for (chunk, was) in base.chunks_exact_mut(WORDS).enumerate() {
            let at = chunk * CHUNK;
            // SAFETY: the chunk lies in the copy's first bytes, the length of the pages.
            unsafe { read_chunk(self.copy.as_ptr().add(at), &mut gpu) };
            if gpu == *was {
                continue;
            }
            // SAFETY: the chunk lies in the pages, mapped until the copy is released.
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

    /// A copy holds its pages' bytes; a commit's copy in writes the bytes the CPU changed and
    /// leaves the copy's others, which the GPU may have written meanwhile; a completion's merge
    /// back writes the bytes the GPU changed where the CPU did not, a byte both changed keeping
    /// the CPU's, which the next copy in takes; each counts the bytes it wrote; a released copy
    /// syncs nothing.
    #[test]
    fn a_copy_syncs_the_bytes_each_side_changed() {
        let mut pages = chunks(2);
        let home = NonNull::new(pages.as_mut_ptr().cast::<u8>()).expect("the pages");
        let len = 2 * CHUNK;
        // SAFETY: the pages are the test's own, written below only through `home`.
        unsafe { home.as_ptr().write_bytes(0x11, len) };
        // SAFETY: the pages live until the test's end, past the copy.
        let copied = unsafe { Copied::new(home, len) }.expect("a copy");
        assert!(copied.layout.size() >= len && copied.layout.align() == ALIGN);
        assert_eq!(bytes(copied.copy, len), vec![0x11; len]);
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

        // Released, it syncs nothing.
        *copied.base.lock() = None;
        // SAFETY: inside the pages.
        unsafe { home.as_ptr().write(0x77) };
        assert_eq!((copied.copy_in(), copied.merge_back()), (0, 0));
        assert_eq!(bytes(copied.copy, 1), [0x11]);
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

    /// A length of no whole number of chunks takes no copy.
    #[test]
    fn a_copy_is_of_whole_chunks() {
        let mut pages = chunks(1);
        let home = NonNull::new(pages.as_mut_ptr().cast::<u8>()).expect("the pages");
        // SAFETY: the pages are the test's own, past any copy.
        assert!(unsafe { Copied::new(home, CHUNK / 2) }.is_none());
        // SAFETY: as above.
        assert!(unsafe { Copied::new(home, 0) }.is_none());
    }
}
