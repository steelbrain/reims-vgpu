//! Buffers and heaps over the guest's pages: the alias a heap over address ranges is made over
//! (`setAddressRanges:`), and the one each replace of a detached backing gives a buffer or a heap
//! (`replaceBackingWithRanges:readOnly:`); and what `detachBacking` keeps.
//!
//! - **An alias per attach.** The ranges' pages are remapped, shared, into a fresh range of this
//!   platform's own (`mach_vm_remap`, one call per run of host-contiguous ranges), and the
//!   runtime is given that range with a keep-alive that frees it (`new_heap_with_bytes_no_copy`,
//!   `rebind_bytes_no_copy`, [`keep_alive`]), and no deallocator: no one but this platform sees
//!   the alias go. A write through the alias shows in the guest's pages and one through the
//!   guest's pages in the alias, so the GPU's writes land in the guest's memory at once and it
//!   reads the guest's latest bytes, as Metal's do. The runtime keeps an alias, and so the pages
//!   under it, until every command buffer committed with it completed and it freed its import of
//!   them, which it does before it drops the keep-alive: a later replace, a detach or
//!   the object's release never takes pages a committed command buffer still uses or the device
//!   still imports, and the guest's memory stays mapped under it whatever the guest's own mapping
//!   does meanwhile. A replace never remaps under a range the runtime holds: the runtime keeps
//!   the pages a range had when it imported them (`tests/no_copy_remap.rs`).
//! - **Ranges off whole pages.** Core's ranges are whole pages, each starting on a page; Metal
//!   takes others, a heap's and a replace's alike, and this platform takes them as Metal does
//!   (measured on the M3 Max, macOS 26.7, `tests/detach_backing.rs`'s `unaligned`): ranges whose
//!   bytes follow on from the previous range's are one run of bytes, and each run of bytes takes
//!   the pages it lies in, whole, from the page of its first byte through the page of its last
//!   ([`spans`]). A range of part of a page takes all of it, the page's bytes past the range
//!   included (the CPU and the GPU read them, and the GPU's fill writes them), and a later range
//!   starting off its page takes it from the page's start. A first range starting off its page dies
//!   where it is given ([`off_the_page`]): Metal's `contents` answers its bytes from the range's
//!   own address, and its GPU reads and writes them from its page's start, which no alias gives at
//!   once.
//! - **Read-only.** A read-only alias (the replace's `readOnly:`, or an object made with
//!   `MTLResourceOptions` bit 24) has its maximum protection lowered to read (`mach_vm_protect`
//!   with `set_maximum`), as Metal's alias of a read-only replace of several ranges is
//!   (measured: current and maximum protection read). The GPU's writes never reach such pages,
//!   and the command buffer completes: later commands of the same command buffer read what it
//!   wrote, and a later command buffer the guest's bytes, as under a read-only replace on Metal
//!   (blit fills, measured on both platforms by `tests/detach_backing.rs`; kernel stores are
//!   dropped alike on Metal). On Linux the runtime makes the object over a copy of such an
//!   alias's pages, or of a no-copy object's pages made with bit 24 or, but for a buffer core
//!   makes over bytes, mapped read-only, whose GPU writes it drops, as the views' gate refuses
//!   read-only pages (`crate::os::import_view`): a shader's, a blit's into a buffer and a
//!   visibility result's are sent, and dropped as Metal's are, while a pass's store into the
//!   object's memory and a write of a texture over it are answered where they are encoded
//!   ([`Backing::read_only_now`], `crate::gpu::writes`). A buffer core makes over bytes it maps
//!   read-only is imported through a writable twin of them instead, as a surface's read-only
//!   alias is, whose writes die where they are encoded ([`Backing::twinned_now`],
//!   [`ImportSite::twins`]).
//! - **Detach** marks the object detached; its `contents` and its GPU address stay, as Metal's do
//!   (measured: `contents` stays the alias of the last replace, still over its pages). A
//!   submission of a command buffer that uses it fails while it is detached, as do those of
//!   command buffers made before it was released detached ([`submission_check`],
//!   `crate::gpu::command`'s docs). A command that names the object, encoded before a replace,
//!   and whose command buffer is first submitted after it (committed after it, or committed
//!   before it and held behind an event wait), reaches the new pages as Metal's does where the
//!   runtime encodes its compute, blit or render encoder again from its calls; in an encoder it
//!   cannot (a compute encoder of the runtime's command predication that shares its unit, holds
//!   a loop or leaves work out, which no record of ours reaches, a render pass that executes an
//!   indirect command buffer or whose end captured its draws' tile reach, one with a call that
//!   answers otherwise than at its encoding) it fails, as the runtime resolved the use as the
//!   command was encoded
//!   (`CommandBufferError::EncodedUseRebound`): that dies where its failure is first read
//!   (`crate::gpu::command`), where Metal runs the command over the pages the object lies in as
//!   it submits. The runtime decides each of the command buffer's
//!   submissions as it submits it, after the submission check, so one the check fails keeps its
//!   code: a use detached behind a wait, in the first submission or a later one (an event's
//!   split), fails with code 8, as Metal's (`tests/detach_backing.rs`). A rebound use in a later
//!   submission dies all the same, the command buffer keeping that code, where Metal runs it over
//!   the new pages. One submitted before keeps the pages it had, as Metal's does. Detaching a
//!   buffer placed in a heap over address ranges detaches the heap, as Metal's does (measured):
//!   every resource placed in it fails from then on, and a replace of the placed buffer answers
//!   NO, leaving the heap detached. A heap made over no address ranges, and a buffer placed in
//!   one, never detach: Metal answers NO to the detach and to a replace, and their command
//!   buffers run (measured, `tests/detach_backing.rs`). Core detaches neither: its detached
//!   buffers are no-copy buffers of their own, and its detached heaps placement heaps over
//!   address ranges. Any other detach succeeds but one in flight (below), and detaching a
//!   detached object is no change.
//! - **No memory while detached.** A detach is the runtime's (`detach_backing`): the
//!   object lies in no memory until a replace gives it some, so the runtime holds the memory it
//!   was over no more, the guest's pages a replace gave it or what it was made over (core's
//!   temporary allocation, which core makes its detached buffers and heaps over and detaches at
//!   once). The detach ends the object's use: a command that reaches it without naming it (an
//!   address a kernel loads), which no submission check sees, reads 0 and its writes are dropped,
//!   the command buffer completing with no error, until a command buffer uses the object after a
//!   replace, where it would reach pages the guest has given up, or core's memory. Metal answers
//!   so too: a detached buffer, used or not, replaced or not, and a buffer placed in a detached
//!   heap, reached through an indirect command's buffer address or a loaded pointer, read 0 and
//!   drop the write, the command buffer completing with no error (measured on the M3 Max, macOS
//!   26.7: through a loaded pointer on both platforms in `tests/detach_backing.rs`'s
//!   `detached_reach` and `never_replaced`, and through an indirect command's buffer address by a
//!   probe outside the tree). No memory is imported for it, so nothing the GPU writes there is
//!   kept on any host. A texture over the object's bytes stays, and presents a stand-in of its
//!   image's plan, zero bytes, until they lie in pages again (after Metal's
//!   answer measured on an M4: such a texture reads zero bytes and drops its writes). Where a
//!   texture over the bytes has no exact stand-in (a multisample or multi-planar one, or one with
//!   a depth or decoded twin) or is being made, the runtime refuses the detach, and the process
//!   ends there ([`refused_detach`]). `contents` stays what it was: the alias of the last
//!   replace, or of the ranges a heap was made over, or the memory a no-copy buffer was made over
//!   ([`Own`]), which the detach does not free: Metal runs a no-copy buffer's deallocator at its
//!   first replace, inside the call, or as it is destroyed, never at a detach, so the run the
//!   runtime hands over as it lets go of the memory at the detach waits until then ([`Held`]). A
//!   buffer neither replaced nor made over memory of the caller's by `newBufferWithDescriptor:`
//!   (one of the device's own memory, one made on a surface, one `newBufferWithBytesNoCopy:`
//!   made) dies at the detach, where what Metal's `contents` answers and when it runs the
//!   deallocator are not measured; core detaches none. A replace's rebind ends the use too, as
//!   Metal's replace does, so after it such a command reads 0 and drops the write until a command
//!   buffer uses the object again, and then reaches its new pages (measured on Metal so: a buffer
//!   or a heap used, detached and replaced; both platforms in `tests/detach_backing.rs`). A
//!   command buffer not yet committed that used the object, its encoder ended or still open,
//!   leaves the detach as it is: it fails with code 8 as it is submitted, and a command that
//!   reaches the object unnamed meanwhile reads 0, as on Metal (measured,
//!   `tests/detach_backing.rs`'s `held`). The runtime resolves what a command buffer reaches
//!   unnamed as it is first submitted, so one committed before the detach and
//!   submitted after it reads 0 there too, and one submitted before it keeps the pages it reached
//!   until it completed, as Metal's does.
//! - **In flight.** Metal answers NO to a detach, and the object stays attached, while a command
//!   buffer submitted and not yet completed took it (a command that named it, a buffer placed in
//!   the heap included, or made it resident): a heap's at once, a buffer's once each such command
//!   buffer completed, the call blocking until then. A buffer's answer after that wait is a race
//!   in the kernel's one call, NO in most runs and YES in some (13 of 150 on 2026-10-11); a heap's
//!   is always NO (measured on the M3 Max, macOS 26.7, 25G229, `tests/detach_backing.rs`'s
//!   `in_flight` and `expected_in_flight`). This platform answers NO, one of Metal's two.
//!
//!   Once the command buffer completed, Metal lets go of what it took as it processes the
//!   completion, which races the start of its completed handlers, unseen by any caller: a detach
//!   made before then answers NO, one made after YES (measured on the M3 Max, macOS 27.0.1,
//!   26A434, 2026-10-23, a heap's and a buffer's, used by a blit fill, a kernel's write, a
//!   `useResource:` or a draw, by a probe outside the tree; `tests/detach_backing.rs`'s
//!   `in_a_completed_handler`). Made at once in the first or a second completed handler, a detach
//!   answered NO in 257 of 1132 runs, from 1 in 50 to 40 in 40 per batch, the odds moving with
//!   the host's load from one minute to the next, in the same layouts and with the same work in
//!   the process (197 of 200 on macOS 26.7); and YES in all 134 where a draw used the object.
//!   Made 20 µs or more into the handler, it answered YES in all 684 runs; with 32 command
//!   buffers queued before and after this one, from 100 µs in (all 152; NO in 17 of 40 at 60 µs).
//!   Every NO detached when made again, 10 to 101 µs later, within 117 µs of the handler's start
//!   (all 467). Made on another thread as `waitUntilCompleted` returned, a detach answered NO in
//!   21 of 184 runs (9 of 200 on macOS 26.7, and none of 200 made 2 ms later). A command buffer
//!   that completes with an error, its status `Error`, is let go of before its completed handlers
//!   run: Metal answers YES to a detach made in its completed handler, whether the encoders that
//!   took the object ran or ran nothing (measured on the M3 Max, macOS 27.0.1, 26A434: 2796 of
//!   2800 detaches of a heap's and a buffer's, in command buffers failed with code 1 or 16 by
//!   protection options and with code 8 by a detached use; `tests/detach_backing.rs`'s
//!   `expected_in_a_failed_completed_handler`). So once a command buffer completed, YES is one
//!   of Metal's answers at every moment, and NO only at its handlers' start, by a race.
//!
//!   The runtime's own detach answers NO while the work of a submission that took the object has
//!   not completed, and lets go of what a command buffer's submissions took as it processes its
//!   completion, before its completed handlers run, failed or not (the runtime's `detach.rs` and
//!   `CommandBufferShared::complete`). This platform answers first: each submission the runtime submits is noted with
//!   its encoders as its check decides it ([`Opened::submit`]), until the command buffer closes,
//!   after its completed handlers ran; a detach asks the runtime what those encoders use
//!   ([`in_flight`]), of the command buffers with a submission noted since the object was made
//!   alone that have not completed, their status neither `Completed` nor `Error`, which the runtime
//!   sets before it runs the completed handlers; a buffer's waits by reading it until each such
//!   command buffer completed. One Metal's GPU fails at its commit for what its encoders did is
//!   noted nowhere, as it is committed with no check (`crate::gpu::command`). Where none took it,
//!   the runtime's detach answers, by its rule over what it holds, a NO of its own answered as
//!   Metal's NO, the object attached: a detach made once a command buffer completed, in its
//!   completed handlers or after, answers YES, whichever of its submissions took the object, and
//!   whether it completed or failed, those of encoders its commit failed included: Metal's answer
//!   later in the handlers and in a failed one's, and one of its two at their start
//!   (`tests/detach_backing.rs`'s `in_a_completed_handler`). Core never reads the answer: after a NO, its next
//!   prepare has the replace answer NO, which raises (`BufferBackingRefused`,
//!   `HeapBackingRefused`), and a command buffer that uses the object before it runs. The GPU's own
//!   failures (a hang, a fault, a victim of its recovery), which no test here may run, follow the
//!   failed command buffer's rule unmeasured. A detach made while a command buffer that fails is in
//!   flight is unmeasured too, and answers as for any other.
//!
//! What Metal answers, measured on the M3 Max (macOS 26.7, 25G229), and this platform with it:
//! - **A replace** answers NO, and leaves the object detached, for an attached object, an empty
//!   array, a NULL or unmapped range, a range with a page mapped without access, and ranges
//!   whose lengths total other than the buffer's length or the heap descriptor's size (a heap of
//!   descriptor size 0x4010, size 0x8000, takes ranges of 0x4010 bytes, and neither 0x4000 nor
//!   0x8000; a buffer of two pages refuses a range of 0x1000 bytes, or of a page and 0x40 bytes,
//!   before one of a page or two, whose pages would fill it). One that answers YES keeps the
//!   object's GPU address, and runs the deallocator of a buffer's memory before it inside the
//!   call (this platform's runtime once no committed command buffer still uses it).
//! - **A heap over address ranges** is made over ranges of any total, shorter or longer than its
//!   size: this platform's heap takes the ranges' pages up to its size, and zero-filled pages of
//!   its own past them. None is made over an empty array or a NULL or unmapped range, or one
//!   with a page mapped without access, past its size too, nor of size 0. Read-only pages make one,
//!   and replace, read-only or not (`tests/detach_backing.rs`, `pages_without_access`).
//! - **`contents`** answers the alias: the bytes are the guest's either way, and the address is
//!   the host's alone. Metal's differs in where it points: a buffer replaced by one range answers
//!   that range's own address (writable whatever `readOnly:` says), and the buffers of a heap
//!   made over one range, as core makes its heaps, answer their offsets in that range after any
//!   replace, over its pages and not the replacing ones; core frees that range after the heap's
//!   first replace. A CPU read through such a heap buffer's `contents` after a replace (a texture
//!   filled from its bytes) reads the guest's bytes here, where Metal reads memory core has
//!   freed.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::num::NonZeroU64;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

use metal2vulkan_runtime as rt;
use reims_vgpu_core::host::HostPtr;
use reims_vgpu_core::num::ToU64;
use reims_vgpu_core::platform::os::vm::{HostVm, KernReturn, VmProt};
use reims_vgpu_core::platform::types::ResourceAddressRange;
use reims_vgpu_core::raise::RaiseResult;
use reims_vgpu_core::raise::death::{DeathId, DeathKind, fatal_with_detail};
use reims_vgpu_core::rt::BlockEnv;
use reims_vgpu_core::sync::Mutex;
use reims_vgpu_core::tasks::isolation::TaskLife;

use crate::answer::unrepresentable;
use crate::os::OsVm;

/// `replaceBackingWithRanges:readOnly:`.
pub(crate) const REPLACE: &str = "replaceBackingWithRanges:readOnly:";

/// `detachBacking`.
const DETACH: &str = "detachBacking";

/// The runtime rounds a heap's size up to a multiple of this (16 KiB), and the heap's memory
/// covers all of it.
pub(crate) const HEAP_GRANULE: u64 = 0x4000;

/// Whether a buffer or heap is detached (`detachBacking`), and the read-only bit it was made
/// with, which every alias it takes keeps. Core sends one object's detaches and replaces one at a
/// time. While it is detached, and once it is released detached, the submissions that use it
/// fail ([`submission_check`]), by its runtime identity.
#[derive(Debug)]
pub struct Backing {
    detached: AtomicBool,
    read_only: bool,
    /// Whether the object is a heap, whose detach answers NO at once where a command buffer in
    /// flight took it; a buffer's waits for each such command buffer first ([`Backing::detach`]).
    heap: bool,
    /// Its number among the buffers and heaps made ([`MADE`]): a submission noted before it was
    /// made never took it ([`in_flight`]).
    made: u64,
    /// The runtime object's identity, which names it in [`DETACHED`]; `None` for one a detach
    /// never detaches: a heap made over no address ranges, and a buffer placed in a heap, whose
    /// detach is its heap's (module docs).
    id: Option<rt::ObjectId>,
    /// The memory `contents` stays over while the object is detached.
    alias: Mutex<Aliased>,
    /// Whether the memory the object lies in now is read-only where the runtime does not import
    /// read-only memory as it is mapped ([`crate::os::COPIES_READ_ONLY`]): the alias of a
    /// read-only replace, or of the ranges a heap was made over with bit 24, or the memory a
    /// no-copy buffer or heap was made over, mapped read-only or made with bit 24; counted while it
    /// holds ([`crate::gpu::writes::set_kept`]).
    kept: AtomicBool,
    /// Whether, of that, the runtime imports the memory through a writable twin of its pages,
    /// whose GPU writes reach them ([`Backing::twinned_now`]): a buffer core made over bytes it
    /// maps read-only, without bit 24 ([`ImportSite::twins`]); else it copies them, dropping the
    /// GPU's writes.
    twinned: AtomicBool,
    /// The guest's pages the object's memory is over, and whether a detach left it in no memory
    /// since its last replace, which work encoded reads without the object
    /// ([`crate::gpu::writes::Kept`]).
    pages: Arc<Pages>,
}

/// The memory a buffer's or heap's `contents` stays over while a detach left it in no memory
/// ([`Pages::zeroed`]): the alias of its last replace, or of the address ranges a heap was made
/// over ([`Alias`]), which the runtime holds until the object is detached or rebound and no
/// command buffer submitted uses it, and [`Backing`] until another takes its place; or the memory
/// a no-copy buffer was made over ([`Own`]), until its first replace.
#[derive(Debug, Default)]
struct Aliased {
    /// The alias of the last replace, or the one a heap over address ranges was made over;
    /// `None` for a buffer never replaced.
    last: Option<Arc<Alias>>,
    /// The memory of the caller's a no-copy buffer was made over, until its first replace;
    /// `None` for any other object.
    own: Option<Own>,
}

/// The memory of the caller's a no-copy buffer was made over (`newBufferWithDescriptor:`, as core
/// makes its detached buffers over a temporary allocation): its first byte, which `contents`
/// answers while the buffer is detached, as Metal's stays, and the hold on the run of its
/// deallocator ([`Held`]).
#[derive(Debug)]
pub(crate) struct Own {
    /// The first byte.
    at: NonNull<u8>,
    held: Arc<Held>,
}

// SAFETY: `at` is an address only, which nothing here reads through: it is answered as
// `contents`.
unsafe impl Send for Own {}

// SAFETY: as for `Send`.
unsafe impl Sync for Own {}

impl Own {
    /// The memory of `at`, whose deallocator's run, which the runtime makes once it lets go of
    /// the memory, goes through `held`.
    pub(crate) fn new(at: NonNull<u8>, held: Arc<Held>) -> Own {
        Own { at, held }
    }
}

/// The run of a no-copy buffer's deallocator. Metal makes it at the buffer's first replace,
/// inside the call, or as the buffer is destroyed, and never at a detach (measured on the M3 Max,
/// macOS 26.7: `tests/detach_backing.rs`'s `buffer_life`). The runtime makes it once it lets go of
/// the memory, which a rebind does, and a detach: so a run the runtime hands over after such a
/// detach is held ([`Held::hold`]) until the replace that follows, or until the buffer is
/// destroyed ([`Held::release`], [`Backing`]'s drop).
#[derive(Default)]
pub(crate) struct Held(Mutex<Hold>);

/// Whether [`Held`] holds a run.
#[derive(Default)]
enum Hold {
    /// A run handed over is made at once.
    #[default]
    Runs,
    /// A run handed over waits for [`Held::release`]: the run, once handed over.
    Holding(Option<Box<dyn FnOnce() + Send>>),
}

impl std::fmt::Debug for Held {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let holding = matches!(*self.0.lock(), Hold::Holding(_));
        f.debug_struct("Held").field("holding", &holding).finish()
    }
}

impl Held {
    /// The run of the deallocator, which the runtime hands over: made now, or held.
    pub(crate) fn run(&self, run: Box<dyn FnOnce() + Send>) {
        let mut hold = self.0.lock();
        if let Hold::Holding(held) = &mut *hold {
            *held = Some(run);
            return;
        }
        drop(hold);
        run();
    }

    /// A run handed over from now on is held.
    fn hold(&self) {
        let mut hold = self.0.lock();
        if matches!(*hold, Hold::Runs) {
            *hold = Hold::Holding(None);
        }
    }

    /// A run held is made now, and one handed over later at once.
    fn release(&self) {
        let held = match std::mem::take(&mut *self.0.lock()) {
            Hold::Holding(held) => held,
            Hold::Runs => None,
        };
        if let Some(run) = held {
            run();
        }
    }
}

/// The guest's pages a buffer's or heap's memory is over now, as the host maps them, and whether a
/// detach left it in no memory since its last replace, which its [`Backing`] keeps and work
/// encoded reads without the object ([`crate::gpu::writes::Kept`]): a reference that revived a
/// heap or buffer could run its release on the thread that reads.
#[derive(Debug, Default)]
pub struct Pages {
    /// The pages.
    paged: Mutex<Paged>,
    /// Whether a detach left the object in no memory since its last replace, where the GPU reads
    /// zeros through it: written with `paged` locked, and read without it, as each bind asks.
    zeroed: AtomicBool,
    /// Whether it is over the guest's pages now: `paged` has pages and it is not zeroed.
    guests: AtomicBool,
}

/// The runs of [`Pages`].
#[derive(Debug, Default)]
struct Paged {
    /// The guest's pages, as the host maps them (each a first byte and an end): those of the
    /// object's last replace, or of the address ranges a heap was made over.
    pages: Vec<std::ops::Range<u64>>,
    /// Where each of `pages` ends in the object's bytes, laid out over them in their order: the
    /// lengths up to it added up, so the run holding an object's byte is found by a search, not
    /// a walk from the first (a heap over scattered pages holds a run a page).
    ends: Vec<u64>,
}

impl Paged {
    /// `pages`, laid out in their order.
    fn of(pages: Vec<std::ops::Range<u64>>) -> Paged {
        let ends = (pages.iter())
            .scan(0u64, |at, run| {
                *at = at.saturating_add(run.end - run.start);
                Some(*at)
            })
            .collect();
        Paged { pages, ends }
    }

    /// The host ranges that hold the object's bytes `bytes`, as a walk of `pages` from the first
    /// finds them (`os::surface::in_runs`), from the first run whose end lies past `bytes`' start.
    fn in_runs(&self, bytes: &std::ops::Range<u64>) -> Vec<std::ops::Range<u64>> {
        let first = self.ends.partition_point(|&end| end <= bytes.start);
        let mut held = Vec::new();
        let mut at = first
            .checked_sub(1)
            .and_then(|before| self.ends.get(before).copied())
            .unwrap_or(0);
        for (run, &end) in self.pages.iter().zip(&self.ends).skip(first) {
            if at >= bytes.end {
                break;
            }
            let (from, to) = (bytes.start.max(at), bytes.end.min(end));
            if from < to {
                held.push(run.start + (from - at)..run.start + (to - at));
            }
            at = end;
        }
        held
    }

    /// Whether [`Paged::in_runs`] finds any of `bytes`: the runs, laid out one after another,
    /// hold the object's bytes from 0 to the last run's end, so whether `bytes` meet those.
    fn meets(&self, bytes: &std::ops::Range<u64>) -> bool {
        let total = self.ends.last().copied().unwrap_or(0);
        bytes.start < bytes.end && bytes.start < total
    }
}

impl Pages {
    /// The object is over `pages` from now on.
    fn set(&self, pages: Vec<std::ops::Range<u64>>) {
        let guests = !pages.is_empty();
        let mut paged = self.paged.lock();
        *paged = Paged::of(pages);
        self.zeroed.store(false, Ordering::Release);
        self.guests.store(guests, Ordering::Release);
    }

    /// A detach left the object in no memory: the GPU reads zeros through it, and its writes
    /// reach no guest page.
    fn zero(&self) {
        let _paged = self.paged.lock();
        self.zeroed.store(true, Ordering::Release);
        self.guests.store(false, Ordering::Release);
    }

    /// The host memory of the object's bytes `bytes` in the guest's pages it is over now, read
    /// under its lock with no copy of the pages; `None` for one over memory of its own, or
    /// detached, answered without the lock.
    fn in_pages(&self, bytes: &std::ops::Range<u64>) -> Option<Vec<std::ops::Range<u64>>> {
        if !self.guests.load(Ordering::Acquire) {
            return None;
        }
        let paged = self.paged.lock();
        (!self.zeroed() && !paged.pages.is_empty()).then(|| paged.in_runs(bytes))
    }

    /// Whether [`Pages::in_pages`] finds any memory of `bytes`, read as it reads it, with nothing
    /// collected; `None` where it answers `None`.
    fn meets_pages(&self, bytes: &std::ops::Range<u64>) -> Option<bool> {
        if !self.guests.load(Ordering::Acquire) {
            return None;
        }
        let paged = self.paged.lock();
        (!self.zeroed() && !paged.pages.is_empty()).then(|| paged.meets(bytes))
    }

    /// Whether a detach left the object in no memory since its last replace.
    pub(crate) fn zeroed(&self) -> bool {
        self.zeroed.load(Ordering::Acquire)
    }
}

impl Backing {
    /// The attached backing of the runtime buffer `id`, over the memory it was made with;
    /// `read_only` for one made with `MTLResourceOptions` bit 24.
    pub(crate) fn new(read_only: bool, id: rt::ObjectId) -> Backing {
        Backing::of(read_only, false, id)
    }

    /// The attached backing of the runtime heap `id`, made over address ranges, as
    /// [`Backing::new`]'s.
    pub(crate) fn heap(read_only: bool, id: rt::ObjectId) -> Backing {
        Backing::of(read_only, true, id)
    }

    /// The attached backing of the runtime buffer or `heap` `id`.
    fn of(read_only: bool, heap: bool, id: rt::ObjectId) -> Backing {
        Backing {
            detached: AtomicBool::new(false),
            read_only,
            heap,
            made: MADE.fetch_add(1, Ordering::SeqCst),
            id: Some(id),
            alias: Mutex::new(Aliased::default()),
            kept: AtomicBool::new(false),
            twinned: AtomicBool::new(false),
            pages: Arc::default(),
        }
    }

    /// The backing of an object a detach never detaches: it stays attached, and a replace
    /// answers NO, as Metal's does (module docs).
    pub(crate) fn fixed() -> Backing {
        Backing {
            detached: AtomicBool::new(false),
            read_only: false,
            heap: false,
            made: 0,
            id: None,
            alias: Mutex::new(Aliased::default()),
            kept: AtomicBool::new(false),
            twinned: AtomicBool::new(false),
            pages: Arc::default(),
        }
    }

    /// `detachBacking` of the object, which `detach`, the runtime's detach of its runtime object
    /// (`detach_backing`), detaches: the object is detached from now on, but for one that never
    /// detaches ([`Backing::fixed`]), and one a command buffer submitted and not yet completed
    /// took, which stays attached, Metal's NO: a heap's at once, a buffer's once each such command
    /// buffer completed ([`in_flight`], module docs). The runtime's object then lies in no memory,
    /// so the GPU no longer reaches the memory it was over through it, the guest's pages a replace
    /// gave it or what it was made over, and a command reaches it unnamed nowhere until a replace;
    /// where the runtime refuses that, the process ends here ([`refused_detach`]), and where it
    /// answers NO, holding what a submission whose work has not completed took (one submitted as
    /// the detach is made), it stays attached. A buffer neither replaced nor made over memory of the caller's by
    /// `newBufferWithDescriptor:` dies here ([`unmeasured_detach`]).
    pub(crate) fn detach(&self, detach: impl FnOnce() -> Result<bool, rt::Error>) {
        let Some(id) = self.id else {
            return;
        };
        if self.is_detached() {
            return;
        }
        let held = {
            let aliased = self.alias.lock();
            match (&aliased.last, &aliased.own) {
                (None, None) => unmeasured_detach(),
                // A buffer never replaced: the run of its deallocator, which the runtime hands
                // over as it lets go of the memory, waits for the first replace or the release.
                (None, Some(own)) => Some(Arc::clone(&own.held)),
                (Some(_), _) => None,
            }
        };
        let takers = in_flight(id, self.made);
        if !takers.is_empty() {
            if !self.heap {
                takers.iter().for_each(completed);
            }
            return;
        }
        if let Some(held) = &held {
            held.hold();
        }
        match detach() {
            Ok(true) => {}
            Ok(false) => {
                if let Some(held) = held {
                    held.release();
                }
                return;
            }
            Err(refused) => fatal_with_detail(
                DeathId::Platform {
                    kind: DeathKind::Abort,
                    call: DETACH,
                },
                &refused_detach(&refused),
            ),
        }
        self.detached.store(true, Ordering::Release);
        DETACHED.detached(id);
        self.pages.zero();
    }

    /// Where the object's `contents` stays while it is detached: the first byte of the alias of
    /// its last replace or of the ranges a heap was made over, or of the memory a no-copy buffer
    /// was made over, which it had when it was detached ([`Backing::detach`]); `None` for one
    /// attached, whose runtime object answers.
    pub(crate) fn kept_contents(&self) -> Option<NonNull<u8>> {
        if !self.pages.zeroed() {
            return None;
        }
        let aliased = self.alias.lock();
        match (&aliased.last, &aliased.own) {
            (Some(alias), _) => Some(alias.at),
            (None, Some(own)) => Some(own.at),
            (None, None) => None,
        }
    }

    /// The object, a heap, was made over `alias`, the alias of `ranges` of the guest's pages
    /// ([`Backing::in_pages`]), which its `contents` stays over while it is detached: read-only
    /// where it was made with bit 24 ([`gather`]).
    pub(crate) fn made_over(&self, ranges: &[ResourceAddressRange], alias: Option<Arc<Alias>>) {
        let kept = crate::os::COPIES_READ_ONLY && alias.is_some() && self.read_only;
        self.alias.lock().last = alias;
        self.twinned.store(false, Ordering::Release);
        crate::gpu::writes::set_kept(&self.kept, kept);
        self.pages.set(pages(ranges));
    }

    /// The object, a no-copy buffer, was made over `own`, memory of the caller's.
    pub(crate) fn made_over_own(&self, own: Own) {
        self.alias.lock().own = Some(own);
    }

    /// The object, a no-copy buffer or heap, was made over memory of the caller's that is
    /// read-only where `read_only` (mapped so, [`crate::os::ImportView::read_only`], or made with
    /// bit 24), and that the runtime imports through a writable twin of its pages where `twinned`
    /// ([`crate::os::ImportView::twinned`], a buffer core makes over bytes without bit 24): its
    /// memory is read-only then, and where it was made with bit 24
    /// ([`Backing::read_only_now`]), twinned where both say so ([`Backing::twinned_now`]).
    pub(crate) fn imported(&self, read_only: bool, twinned: bool) {
        let kept = crate::os::COPIES_READ_ONLY && (read_only || self.read_only);
        let twinned = kept && twinned && !self.read_only;
        self.twinned.store(twinned, Ordering::Release);
        crate::gpu::writes::set_kept(&self.kept, kept);
    }

    /// Whether the memory the object lies in now is read-only where the runtime does not import
    /// read-only memory as it is mapped (on Linux, [`crate::os::COPIES_READ_ONLY`]): it imports
    /// it through a writable twin ([`Backing::twinned_now`]), or makes the object over a copy of
    /// it, dropping the GPU's writes, where Metal's GPU keeps them from the memory itself and
    /// faults a pass that stores into it; not while it is detached, in no memory.
    pub(crate) fn read_only_now(&self) -> bool {
        self.kept.load(Ordering::Acquire) && !self.pages.zeroed()
    }

    /// Whether, of [`Backing::read_only_now`], the runtime imports the memory through a writable
    /// twin of its pages, whose GPU writes reach them, where Metal drops them: every write
    /// through the object dies where it is encoded (`crate::gpu::writes::kept_write`).
    pub(crate) fn twinned_now(&self) -> bool {
        self.twinned.load(Ordering::Acquire) && self.read_only_now()
    }

    /// [`Backing::read_only_now`], asked again as work that may write through a texture over the
    /// object is encoded ([`crate::gpu::writes::Kept`]).
    pub(crate) fn kept(&self) -> crate::gpu::writes::Kept {
        match self.read_only_now() {
            true => crate::gpu::writes::Kept::Until(Arc::clone(&self.pages)),
            false => crate::gpu::writes::Kept::No,
        }
    }

    /// [`Backing::twinned_now`], asked again as work that may write through the object, a
    /// buffer, is encoded ([`crate::gpu::writes::Kept`]): a buffer over read-only memory the
    /// runtime copies keeps nothing, as its GPU writes are dropped, as Metal's are.
    pub(crate) fn twin_kept(&self) -> crate::gpu::writes::Kept {
        match self.twinned_now() {
            true => crate::gpu::writes::Kept::Until(Arc::clone(&self.pages)),
            false => crate::gpu::writes::Kept::No,
        }
    }

    /// The host memory of the object's bytes `bytes` in the guest's pages its memory is over now,
    /// as the host maps them (each a first byte and an end): those of its last replace, or of the
    /// address ranges a heap was made over; `None` for one over memory of its own, or detached,
    /// whose writes reach no guest page ([`Pages::in_pages`]).
    pub(crate) fn in_pages(
        &self,
        bytes: &std::ops::Range<u64>,
    ) -> Option<Vec<std::ops::Range<u64>>> {
        self.pages.in_pages(bytes)
    }

    /// Whether [`Backing::in_pages`] finds any memory of `bytes`, with nothing collected
    /// ([`Pages::meets_pages`]).
    pub(crate) fn meets_pages(&self, bytes: &std::ops::Range<u64>) -> Option<bool> {
        self.pages.meets_pages(bytes)
    }

    /// Whether a detach detaches the object: a buffer not placed in a heap, or a heap made over
    /// address ranges; not one [`Backing::fixed`] made.
    pub(crate) fn detaches(&self) -> bool {
        self.id.is_some()
    }

    /// Whether the object is detached.
    pub(crate) fn is_detached(&self) -> bool {
        self.detached.load(Ordering::Acquire)
    }

    /// `replaceBackingWithRanges:readOnly:` of an object whose ranges must total `length` bytes
    /// (a buffer's length, a heap descriptor's size) and whose runtime object covers `size`
    /// (its length, the heap's size): NO for an attached object, ranges of another total, and
    /// ranges [`gather`] cannot alias, as Metal answers; else `rebind` gives the runtime's object
    /// the alias, and YES. The alias is freed by its keep-alive, which a refused `rebind` drops
    /// at once. A no-copy buffer's first YES runs its deallocator inside the call, as Metal's
    /// does, where the runtime let go of the memory it was made over ([`Held`]).
    pub(crate) fn replace(
        &self,
        ranges: &[ResourceAddressRange],
        read_only: bool,
        length: u64,
        size: u64,
        rebind: impl FnOnce(NonNull<u8>, rt::KeepAlive) -> RaiseResult<()>,
    ) -> RaiseResult<bool> {
        if !self.is_detached() || total(ranges) != Some(length) {
            return Ok(false);
        }
        let Some(alias) = gather(REPLACE, ranges, size, read_only || self.read_only) else {
            return Ok(false);
        };
        let alias = Arc::new(alias);
        let (first, keep) = Arc::clone(&alias).into_runtime(ImportSite::Replace);
        rebind(first, keep)?;
        let own = {
            let mut aliased = self.alias.lock();
            let kept = crate::os::COPIES_READ_ONLY && (read_only || self.read_only);
            self.twinned.store(false, Ordering::Release);
            crate::gpu::writes::set_kept(&self.kept, kept);
            aliased.last = Some(alias);
            self.pages.set(pages(ranges));
            aliased.own.take()
        };
        // The memory the buffer was made over goes now, inside the call, as Metal's does.
        if let Some(own) = own {
            own.held.release();
        }
        self.detached.store(false, Ordering::Release);
        if let Some(id) = self.id {
            DETACHED.attached(id);
        }
        Ok(true)
    }
}

impl Drop for Backing {
    /// The object's release: the command buffers made before it that use it, detached, still
    /// fail. The runtime object may go before or after: no other object takes its identity. A
    /// no-copy buffer never replaced runs its deallocator now where a detach held the run, as
    /// Metal's runs it as the buffer is destroyed ([`Held`]).
    fn drop(&mut self) {
        if self.is_detached()
            && let Some(id) = self.id
        {
            DETACHED.released(id);
        }
        let own = self.alias.lock().own.take();
        if let Some(own) = own {
            own.held.release();
        }
        crate::gpu::writes::set_kept(&self.kept, false);
    }
}

/// The `MTLCommandBufferError` code of a submission that uses a detached buffer or heap:
/// `MTLCommandBufferErrorOutOfMemory`, as Metal fails it (module docs of
/// [`crate::gpu::command`]).
pub(crate) const DETACHED_USE: u64 = 8;

/// The detached objects, the command buffers they may fail, and those open.
static DETACHED: Registry = Registry::new();

/// How many buffers and heaps a detach may detach were made ([`Backing::new`],
/// [`Backing::heap`]): each takes the count before it as its number, and a submission notes the
/// count as it is checked ([`Submitted::made`]). A command takes an object only once the object
/// was made, so a submission noted with a count of at most an object's number never took it.
static MADE: AtomicU64 = AtomicU64::new(0);

/// [`Detached`], under its lock, and what a submission reads of it without the lock, each
/// written under the lock as it changes: whether an object is detached, and the latest release
/// kept, so a command buffer no object can fail takes no lock of the process's; and the most of
/// [`MADE`] a submission noted, so a detach of an object made since takes none either.
#[derive(Debug)]
struct Registry {
    /// How many objects are detached now: the length of [`Detached::detached`].
    detached: AtomicUsize,
    /// The number of the latest release [`Detached::releases`] keeps, plus one; 0 for none.
    latest: AtomicU64,
    /// The most of [`MADE`] a submission noted ([`Submitted::made`]), taken before the note is
    /// written, under its command buffer's lock: a detach of an object made as the count reached
    /// it, or after, finds no command buffer that took it ([`in_flight`]).
    noted: AtomicU64,
    state: Mutex<Detached>,
    /// For tests: run in [`Registry::released`] between its stores of `latest` and `detached`,
    /// where a reader without the lock may run.
    #[cfg(test)]
    between: Option<fn(&Registry)>,
}

/// The objects detached now and those released detached since the oldest open command buffer
/// was made, and the open command buffers, numbered in one sequence of makes and releases. An
/// object is named by its runtime identity ([`rt::Buffer::object_id`], [`rt::Heap::object_id`]),
/// which no other object ever takes, as the runtime lists a command's uses
/// ([`rt::UsedResources`]): a buffer not placed in a heap, or a heap. `released` holds an object
/// at most once, so it grows at most to the objects released while the oldest open command buffer
/// stays open; whether a command buffer can fail by one is read from the latest alone, in
/// constant time, as each submission asks it ([`Registry::any`]).
#[derive(Debug)]
struct Detached {
    /// The next number of the sequence.
    next: u64,
    /// The objects detached now.
    detached: BTreeSet<rt::ObjectId>,
    /// The objects released detached: the number of the release. A command buffer made before
    /// it may hold a use of the object.
    released: BTreeMap<rt::ObjectId, u64>,
    /// The releases of `released`, in the order of their numbers, the oldest first.
    releases: VecDeque<(u64, rt::ObjectId)>,
    /// The open command buffers, by number, until they close ([`Opened`]). One made by an isolated
    /// task that ended and never committed holds nothing ([`Open::let_go`]): an ended task's
    /// commits no more, but for a commit already past its check of the task's end, which puts it
    /// back ([`Registry::commit`]), and one of a task that died is never released ([`TaskLife`]).
    /// Such ones go as they reach the front, the oldest open, at a close, and all of them once the
    /// open ones grew to twice `fewest`, at a close: in O(1) a close, over the closes.
    open: BTreeMap<u64, Arc<Open>>,
    /// The fewest open command buffers since those that hold nothing were last let go of.
    fewest: usize,
}

/// An open command buffer ([`Opened`]): its number, the isolated task that made it, if one did,
/// until it commits, and what the runtime submitted of it.
#[derive(Debug)]
struct Open {
    /// Its number in the sequence of [`Detached`].
    number: u64,
    /// The isolated task that made it, if one did.
    task: Option<Weak<TaskLife>>,
    /// Whether it is open, not committed ([`OPEN`]), committed, open until it completed whatever
    /// its task does ([`COMMITTED`]), or let go of as holding nothing ([`DROPPED`]): each change
    /// is one exchange, so a commit and a close's letting go never both pass unseen.
    state: AtomicU8,
    /// Its submissions the runtime submitted, until it closes: each decided and noted under this
    /// lock ([`Opened::submit`]), which a detach reads them under ([`in_flight`]), a note for
    /// each part of it the platform committed apart ([`Opened::next_part`]). A note holds its
    /// part's command buffer, which holds its check, which holds this: the part's close takes it.
    submitted: Mutex<Vec<Submitted>>,
    /// The places of its parts not yet closed ([`Opened`]): it closes with the last.
    parts: AtomicUsize,
}

/// [`Open::state`]: open, not committed.
const OPEN: u8 = 0;
/// [`Open::state`]: committed.
const COMMITTED: u8 = 1;
/// [`Open::state`]: let go of, not committed, as holding nothing ([`Open::let_go`]).
const DROPPED: u8 = 2;

impl Open {
    /// Whether it holds nothing, and is let go of now: made by an isolated task that ended, and
    /// not committed. A commit of its command buffer after this, which a task's end after the
    /// commit's check of it lets through, finds it let go of and puts it back
    /// ([`Registry::commit`]).
    fn let_go(&self) -> bool {
        (self.task.as_ref()).is_some_and(|task| task.upgrade().is_none_or(|task| task.has_ended()))
            && (self.state)
                .compare_exchange(OPEN, DROPPED, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
    }
}

/// A command buffer whose submissions the runtime submitted, until it closes ([`Opened`]): it,
/// and the encoders of each submission, by their places among its ended encoders
/// ([`rt::Submission::encoders`]).
#[derive(Debug)]
struct Submitted {
    /// Which part of the command buffer it is, the first 0 ([`Opened::next_part`]).
    part: usize,
    cb: rt::CommandBuffer,
    encoders: Vec<std::ops::Range<usize>>,
    /// The count of [`MADE`] as its latest submission was checked: it took no object made since.
    made: u64,
}

impl Detached {
    /// The next number of the sequence.
    fn number(&mut self) -> u64 {
        let number = self.next;
        self.next += 1;
        number
    }

    /// The objects a command buffer made at `made` fails by where it uses one: those detached
    /// now, and those released detached after it was made.
    fn failing(&self, made: u64) -> Vec<rt::ObjectId> {
        let released = (self.released.iter())
            .filter(|&(_, &at)| at > made)
            .map(|(&id, _)| id);
        self.detached.iter().copied().chain(released).collect()
    }

    /// The open command buffers that hold nothing let go of ([`Detached::open`]).
    fn prune(&mut self) {
        while let Some(oldest) = self.open.first_entry()
            && oldest.get().let_go()
        {
            oldest.remove();
        }
        self.fewest = self.fewest.min(self.open.len());
        if self.open.len() >= 2 * self.fewest.max(16) {
            self.open.retain(|_, open| !open.let_go());
            self.fewest = self.open.len();
        }
    }
}

impl Registry {
    /// None detached, released or open.
    const fn new() -> Registry {
        Registry {
            detached: AtomicUsize::new(0),
            latest: AtomicU64::new(0),
            noted: AtomicU64::new(0),
            state: Mutex::new(Detached {
                next: 0,
                detached: BTreeSet::new(),
                released: BTreeMap::new(),
                releases: VecDeque::new(),
                open: BTreeMap::new(),
                fewest: 0,
            }),
            #[cfg(test)]
            between: None,
        }
    }

    /// A command buffer made now by `task`, or by none: its place, open until it closes.
    fn open(&self, task: Option<Weak<TaskLife>>) -> Arc<Open> {
        let mut state = self.state.lock();
        let open = Arc::new(Open {
            number: state.number(),
            task,
            state: AtomicU8::new(OPEN),
            submitted: Mutex::new(Vec::new()),
            parts: AtomicUsize::new(1),
        });
        state.open.insert(open.number, Arc::clone(&open));
        open
    }

    /// The command buffer of `open` commits now: open until it completed, whatever its task does.
    /// One a close let go of as holding nothing, its task having ended after the commit's check
    /// of the task's end, is open again, so a detach finds what it submits
    /// ([`Registry::noted_since`]); a release made while it was let go of may have been forgotten
    /// already.
    fn commit(&self, open: &Arc<Open>) {
        let committed =
            (open.state).compare_exchange(OPEN, COMMITTED, Ordering::AcqRel, Ordering::Acquire);
        if committed == Err(DROPPED) {
            let mut state = self.state.lock();
            open.state.store(COMMITTED, Ordering::Release);
            state.open.insert(open.number, Arc::clone(open));
        }
    }

    /// Object `id` detached.
    fn detached(&self, id: rt::ObjectId) {
        let mut state = self.state.lock();
        state.detached.insert(id);
        self.detached.store(state.detached.len(), Ordering::SeqCst);
    }

    /// Object `id`, detached, replaced: attached again. A command buffer committed before the
    /// replace that names it fails in the runtime, which resolved the use as it was encoded
    /// (module docs).
    fn attached(&self, id: rt::ObjectId) {
        let mut state = self.state.lock();
        state.detached.remove(&id);
        self.detached.store(state.detached.len(), Ordering::SeqCst);
    }

    /// Object `id` released detached: kept while a command buffer is open, which may have been
    /// made before it. The release is stored before the count of the objects detached: a reader
    /// without the lock that sees the object no longer detached sees it released, so a command
    /// buffer open before the release can fail by the object at every moment ([`Registry::any`]),
    /// and no close forgets the release while that command buffer is open.
    fn released(&self, id: rt::ObjectId) {
        let mut state = self.state.lock();
        state.detached.remove(&id);
        let number = state.number();
        if !state.open.is_empty() {
            state.released.insert(id, number);
            state.releases.push_back((number, id));
            self.latest.store(number + 1, Ordering::SeqCst);
        }
        #[cfg(test)]
        if let Some(between) = self.between {
            between(self);
        }
        self.detached.store(state.detached.len(), Ordering::SeqCst);
    }

    /// Whether a command buffer made at `made` can fail by any object: one is detached, or the
    /// latest release kept came after it. Read without the lock, the count of the objects
    /// detached first, in the order opposite to [`Registry::released`]'s stores.
    fn any(&self, made: u64) -> bool {
        self.detached.load(Ordering::SeqCst) > 0
            || self.latest.load(Ordering::SeqCst) > made.saturating_add(1)
    }

    /// The objects a command buffer made at `made` fails by where it uses one
    /// ([`Detached::failing`]).
    fn failing(&self, made: u64) -> Vec<rt::ObjectId> {
        self.state.lock().failing(made)
    }

    /// Whether a command buffer made at `made` fails where it uses any of `ids`, as the runtime
    /// answers of what it uses and [`Registry::failing`] ([`Opened::submit`]).
    #[cfg(test)]
    fn fails(&self, ids: &[rt::ObjectId], made: u64) -> bool {
        let failing = self.failing(made);
        ids.iter().any(|id| failing.contains(id))
    }

    /// Command buffer `number` closed, and those that hold nothing let go of: the releases no
    /// open command buffer was made before are forgotten. Its place is answered, to be let go of
    /// without the lock.
    fn closed(&self, number: u64) -> Option<Arc<Open>> {
        let mut state = self.state.lock();
        let open = state.open.remove(&number);
        state.prune();
        let oldest = state.open.keys().next().copied().unwrap_or(u64::MAX);
        while let Some(&(at, id)) = state.releases.front()
            && at <= oldest
        {
            state.releases.pop_front();
            state.released.remove(&id);
        }
        if state.releases.is_empty() {
            self.latest.store(0, Ordering::SeqCst);
        }
        open
    }

    /// The open command buffers with a submission noted when `made` buffers and heaps had been
    /// made, or more ([`Submitted::made`]): each, and the encoders of its submissions.
    fn noted_since(&self, made: u64) -> Vec<(rt::CommandBuffer, Vec<std::ops::Range<usize>>)> {
        if self.noted.load(Ordering::SeqCst) <= made {
            return Vec::new();
        }
        let open: Vec<Arc<Open>> = self.state.lock().open.values().cloned().collect();
        // Read under each command buffer's lock, without the registry's.
        open.iter()
            .flat_map(|open| {
                let submitted = open.submitted.lock();
                (submitted.iter())
                    .filter(|s| s.made > made)
                    .map(|s| (s.cb.clone(), s.encoders.clone()))
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

/// A command buffer's place in the sequence of [`Detached`], where it was made, open until it
/// drops: when the command buffer is released uncommitted, when it commits with no check, or
/// when it completes, as the runtime then drops its check ([`submission_check`]). The parts of
/// a command buffer the platform commits apart share its place, each open until it closes so,
/// and the place until the last of them closes ([`Opened::next_part`]).
#[derive(Debug)]
pub(crate) struct Opened {
    open: Arc<Open>,
    /// Which part of its command buffer it is the place of, the first 0.
    part: usize,
}

impl Opened {
    /// The place of a command buffer made now, by the isolated task whose code runs, if one is.
    pub(crate) fn new() -> Opened {
        let task = TaskLife::current().map(|task| Arc::downgrade(&task));
        Opened {
            open: DETACHED.open(task),
            part: 0,
        }
    }

    /// The place of the command buffer's next part, which goes on from this one's
    /// (`CommandBuffer::split`): the same place, as the parts are one command buffer of Metal's,
    /// whose submissions fail by the objects released detached since the first part was made.
    pub(crate) fn next_part(&self) -> Opened {
        self.open.parts.fetch_add(1, Ordering::SeqCst);
        Opened {
            open: Arc::clone(&self.open),
            part: self.part + 1,
        }
    }

    /// The command buffer commits now: open until it completed, whatever its task does
    /// ([`Registry::commit`]).
    pub(crate) fn commit(self) -> Opened {
        DETACHED.commit(&self.open);
        self
    }

    /// The submission check of `submission` of the command buffer ([`submission_check`]): it
    /// fails with [`DETACHED_USE`] where it uses a buffer or heap detached now, or one released
    /// detached since the command buffer was made ([`Detached::failing`]); else it is noted
    /// submitted, with its encoders, until the command buffer closes. It is decided and noted at
    /// once, under the command buffer's lock, which a detach reads the notes under
    /// ([`in_flight`]), so a detach either fails it or sees it. Whether it uses one is asked only
    /// where an object could fail it, of the runtime (`rt::Submission::uses`), which lists
    /// nothing and holds no handle.
    fn submit(&self, submission: &rt::Submission<'_>) -> Option<NonZeroU64> {
        let objects = MADE.load(Ordering::SeqCst);
        let number = self.open.number;
        let mut submitted = self.open.submitted.lock();
        if DETACHED.any(number) && submission.uses(&DETACHED.failing(number)) {
            return NonZeroU64::new(DETACHED_USE);
        }
        DETACHED.noted.fetch_max(objects, Ordering::SeqCst);
        let at = match submitted.iter().position(|s| s.part == self.part) {
            Some(at) => at,
            None => {
                submitted.push(Submitted {
                    part: self.part,
                    cb: submission.command_buffer().clone(),
                    encoders: Vec::new(),
                    made: objects,
                });
                submitted.len() - 1
            }
        };
        let noted = &mut submitted[at];
        noted.encoders.push(submission.encoders());
        noted.made = noted.made.max(objects);
        None
    }
}

impl Drop for Opened {
    /// Its part closes, its note let go of; the last part's close closes the place.
    fn drop(&mut self) {
        let last = self.open.parts.fetch_sub(1, Ordering::SeqCst) == 1;
        let open = last.then(|| DETACHED.closed(self.open.number));
        // Let go of without the registry's lock: it holds the command buffer, whose last handle
        // it may be.
        let submitted = {
            let mut submitted = self.open.submitted.lock();
            let at = submitted.iter().position(|s| s.part == self.part);
            at.map(|at| submitted.swap_remove(at))
        };
        drop((submitted, open));
    }
}

/// Whether command buffer `cb`, made at `opened`, fails: it uses a buffer or heap detached now,
/// or one released detached since it was made ([`Detached::failing`]). Asked of the runtime
/// (`rt::CommandBuffer::uses`) only where an object could fail it.
pub(crate) fn uses_detached(cb: &rt::CommandBuffer, opened: &Opened) -> bool {
    let number = opened.open.number;
    DETACHED.any(number) && cb.uses(&DETACHED.failing(number))
}

/// The check of each submission of the command buffer committed as `cb`
/// ([`rt::CommandBuffer::set_submission_check`]), which the runtime runs once the submission's
/// event waits are met, just before it submits it, on the thread that submits it (the
/// committing thread inside its commit, or the runtime's scheduler thread): one that uses a
/// detached buffer or heap, or one released detached since `cb` was made, fails with
/// [`DETACHED_USE`]; any other is noted submitted ([`Opened::submit`]). The runtime drops it,
/// and so closes `cb`, once the command buffer completed, after its completed handlers ran.
pub(crate) fn submission_check(cb: Opened) -> rt::SubmissionCheck {
    Arc::new(move |submission: &rt::Submission<'_>| {
        // A callout inside the runtime's submission: marked ours, so a panic here aborts.
        let _env = BlockEnv::foreign();
        cb.submit(submission)
    })
}

/// The command buffers submitted and not yet completed that took object `id`, a buffer not
/// placed in a heap or a heap, made as the `made`th of them ([`MADE`]): through a command of a
/// submission the runtime submitted that named it (a buffer placed in the heap included) or made
/// it resident, as the runtime lists what each of its ended encoders uses until it lets go of
/// those uses, after the completed handlers ran ([`rt::CommandBuffer::ended_encoders`]). Only the
/// command buffers with a submission noted since the object was made are asked: none noted
/// before it took it, and none is asked where no submission was noted since
/// ([`Registry::noted_since`]). A submission noted after they were read is decided by the
/// runtime's own detach, which makes the object's uses in submissions after it fail as Metal's
/// do, and answers NO where one submitted before it took the object ([`Backing::detach`]). One
/// that completed, its status `Completed` or `Error` ([`has_completed`]), took nothing, though
/// it stays noted until it closes: Metal lets go of what a command buffer used once it completed,
/// as it starts its completed handlers, and of a failed one's before them (module docs, "In
/// flight").
fn in_flight(id: rt::ObjectId, made: u64) -> Vec<rt::CommandBuffer> {
    // The command buffers not taken are let go of here, under no lock.
    (DETACHED.noted_since(made).into_iter())
        .filter(|(cb, _)| !has_completed(cb))
        .filter(|(cb, encoders)| took(cb, encoders, id))
        .map(|(cb, _)| cb)
        .collect()
}

/// Whether `cb`'s ended encoders at `encoders`, those of its submissions the runtime submitted,
/// use object `id` ([`in_flight`]), as the runtime answers it (`rt::CommandBuffer::encoders_use`,
/// which lists nothing).
fn took(cb: &rt::CommandBuffer, encoders: &[std::ops::Range<usize>], id: rt::ObjectId) -> bool {
    (encoders.iter()).any(|at| cb.encoders_use(at.clone(), &[id]))
}

/// Whether `cb` completed, its status `Completed` or `Error`, which the runtime sets before it
/// runs its completed handlers: a handler of its own sees it so.
fn has_completed(cb: &rt::CommandBuffer) -> bool {
    matches!(
        cb.status(),
        rt::CommandBufferStatus::Completed | rt::CommandBufferStatus::Error
    )
}

/// Returns once `cb` completed ([`has_completed`]). The status is read every
/// [`COMPLETION_POLL`].
fn completed(cb: &rt::CommandBuffer) {
    while !has_completed(cb) {
        std::thread::sleep(COMPLETION_POLL);
    }
}

/// How often [`completed`] reads a command buffer's status.
const COMPLETION_POLL: std::time::Duration = std::time::Duration::from_micros(100);

/// The host's pages `ranges` lie in, in their order, as an alias of them maps them ([`spans`]),
/// each a first byte and an end, those that meet merged.
fn pages(ranges: &[ResourceAddressRange]) -> Vec<std::ops::Range<u64>> {
    let mut runs: Vec<std::ops::Range<u64>> = Vec::new();
    for span in spans(ranges, OsVm::new().page_size()).unwrap_or_default() {
        match runs.last_mut() {
            Some(run) if run.end == span.start => run.end = span.end,
            _ => runs.push(span),
        }
    }
    runs
}

/// The ranges' lengths added up; `None` past `u64::MAX`.
fn total(ranges: &[ResourceAddressRange]) -> Option<u64> {
    ranges
        .iter()
        .try_fold(0u64, |sum, range| sum.checked_add(range.length))
}

/// A range of this platform's address space whose pages are the guest's, shared, and zeros past
/// them ([`gather`]), freed when it drops; or a range another holder maps, which the alias
/// borrows ([`Alias::borrowing`]): a surface's own view of the guest's pages on Linux, which
/// holds the same pages, zeros past them included, and lets the surface's buffers lie over it
/// with no second view of its pages (`crate::os::surface_alias`).
#[derive(Debug)]
pub(crate) struct Alias {
    at: NonNull<u8>,
    length: u64,
    /// What the runtime imports for it while an import of it lives: its pages, or, a surface's,
    /// their writable twin ([`crate::os::import_view`]), one for every import made meanwhile
    /// ([`Alias::into_runtime`]).
    import: Mutex<Weak<crate::os::ImportView>>,
    /// The hold on the memory of a borrowed alias, which its holder maps and frees, dropped with
    /// the alias; `None` for one of this platform's own, which the alias frees.
    borrowed: Option<Box<dyn std::fmt::Debug + Send + Sync>>,
}

// SAFETY: an `Alias` is an address range this platform owns or borrows; nothing reads through it
// here, and it is freed or let go of once, by whichever thread drops it.
unsafe impl Send for Alias {}

// SAFETY: as for `Send`: a shared `Alias` only answers where it is, and imports it.
unsafe impl Sync for Alias {}

impl Alias {
    /// An alias of this platform's own, `length` bytes at `at`, which it frees.
    fn own(at: NonNull<u8>, length: u64) -> Alias {
        Alias::of(at, length, None)
    }

    /// The `length` bytes at `at`, mapped by another holder while `hold` lives, as an alias that
    /// holds `hold` and frees nothing: what it shows is the holder's (`crate::os::surface_alias`).
    #[cfg_attr(
        target_os = "macos",
        expect(dead_code, reason = "only Linux's surfaces lend their own view")
    )]
    pub(crate) fn borrowing(
        at: NonNull<u8>,
        length: u64,
        hold: impl std::fmt::Debug + Send + Sync + 'static,
    ) -> Alias {
        Alias::of(at, length, Some(Box::new(hold)))
    }

    /// The alias of `length` bytes at `at`, borrowed where `borrowed` holds them.
    fn of(
        at: NonNull<u8>,
        length: u64,
        borrowed: Option<Box<dyn std::fmt::Debug + Send + Sync>>,
    ) -> Alias {
        Alias {
            at,
            length,
            import: Mutex::new(Weak::new()),
            borrowed,
        }
    }

    /// The first byte the runtime imports, and the runtime's keep-alive of the alias, which lets
    /// go of it as it drops: once the runtime freed its import, at the buffer's release where
    /// it lies in a live buffer's import ([`keep_alive`]), or at once where the call was
    /// refused. The alias is freed once nothing else holds it.
    ///
    /// The runtime imports the alias's pages, or, a surface's ([`ImportSite::twins`]), where the
    /// host's runtime cannot import them as they are mapped (read-only, on Linux), a writable
    /// twin of them ([`crate::os::import_view`]), whose first byte this answers, and which the
    /// keep-alive holds too: the GPU's writes then reach the pages, where Metal drops them
    /// (`crate::gpu::writes::kept_write`). A read-only alias of any other site is given as it
    /// is, which the runtime copies, dropping the GPU's writes. Every import of the alias made
    /// while another lives is given the same memory, its twin included, so the buffers over one
    /// surface's alias lie at one first byte, in one import of the runtime's (its
    /// `host_aliases.rs`), with one twin's mappings, where each would take a twin of its own.
    pub(crate) fn into_runtime(self: Arc<Self>, site: ImportSite) -> (NonNull<u8>, rt::KeepAlive) {
        let first = self.at;
        let view = {
            let mut import = self.import.lock();
            match import.upgrade() {
                Some(view) => view,
                None => {
                    let view = crate::os::import_view(first, self.length, site.twins());
                    let view = Arc::new(view);
                    *import = Arc::downgrade(&view);
                    view
                }
            }
        };
        let imported = view.at();
        let traced = ImportTrace::new(site, first, self.length, imported);
        (imported, keep_alive((view, self, traced)))
    }

    /// The first byte.
    #[cfg(test)]
    pub(crate) fn first(&self) -> NonNull<u8> {
        self.at
    }
}

impl Drop for Alias {
    /// Frees the alias of this platform's own; lets go of a borrowed one's hold.
    fn drop(&mut self) {
        if self.borrowed.take().is_none() {
            let _ = OsVm::new().deallocate(HostPtr::from_ptr(self.at.as_ptr()), self.length);
        }
    }
}

/// Where host memory was handed to the runtime, as the trace names it ([`ImportTrace`]).
#[derive(Debug, Clone, Copy)]
pub(crate) enum ImportSite {
    /// A no-copy buffer over host memory core gives (`newBufferWithBytesNoCopy`: task pages, the
    /// EFI framebuffer).
    NoCopyBytes,
    /// A no-copy buffer a descriptor names (a detached buffer).
    NoCopyDescriptor,
    /// A buffer over a surface's alias.
    Surface,
    /// A heap over guest ranges.
    Heap,
    /// A buffer's replaced backing.
    Replace,
    /// A buffer over a surface, at its base address.
    SurfaceBase,
}

impl ImportSite {
    /// Whether read-only memory handed to the runtime here is imported through a writable twin
    /// of its pages where the runtime cannot import them as mapped ([`crate::os::import_view`]),
    /// in place, its writes dying where they are encoded (`crate::gpu::writes::kept_write`):
    /// - a buffer over a surface's alias, whose buffers then lie in one import of its pages, so a
    ///   writer's GPU writes through another import reach its readers' reads, which a copy made
    ///   at each commit would miss while that writer runs;
    /// - a buffer core makes over bytes it maps read-only (`newBufferWithBytesNoCopy`), which
    ///   only core's own work reaches and none of it writes through the GPU, so no death comes,
    ///   where a copy costs every commit and completion a compare of all its bytes. Measured on
    ///   the Radeon 780M (RADV), macOS 26.7.2 on apple-gfx at 1920 × 1080, 2026-11-07: 37 to 44
    ///   such buffers of 29.5 MiB a boot, every read-only import of the six boots that named the
    ///   site, three of them of 0x7e9000 bytes; copied rather than twinned, a Finder window's
    ///   two drags drew 1.1 and 1.7 per cent fewer frames (27.35 and 35.7 frames a second over
    ///   six boots, against 27.65 and 36.35 over four), QEMU took about 25 points of a core
    ///   more, and Safari's scroll up a long page drew about 8 per cent fewer (four boots
    ///   against two).
    ///
    /// Elsewhere the runtime copies such memory, dropping the GPU's writes, as Metal drops them:
    /// the guest's read-only objects (a replace `readOnly:YES`, a heap over address ranges or a
    /// no-copy buffer or heap made with bit 24).
    pub(crate) fn twins(self) -> bool {
        matches!(self, Self::Surface | Self::NoCopyBytes)
    }

    /// The site's number in the trace.
    fn code(self) -> u64 {
        match self {
            Self::NoCopyBytes => 1,
            Self::NoCopyDescriptor => 2,
            Self::Surface => 3,
            Self::Heap => 4,
            Self::Replace => 5,
            Self::SurfaceBase => 6,
        }
    }
}

/// The trace of the host memory this platform hands the runtime for one no-copy buffer, heap or
/// rebind, with `REIMS_VGPU_TRACE` set ([`reims_vgpu_core::trace`]): `imp` as its keep-alive is
/// made (the site, the first byte given, its length, the first byte handed over), and `impfree`
/// as the keep-alive goes ([`keep_alive`]). It marks each buffer, not each import: a buffer
/// made over exactly the bytes a live no-copy buffer was made over lies in that one's import
/// (the runtime's `host_aliases.rs`), and its keep-alive goes at its release; the runtime's
/// `WorkCounts` count those (`host_keep_alives_covered`). A diagnostic: it holds nothing and
/// reads nothing with an effect, and the keep-alive drops it last.
pub(crate) struct ImportTrace(u64);

impl ImportTrace {
    pub(crate) fn new(
        site: ImportSite,
        given: NonNull<u8>,
        length: u64,
        imported: NonNull<u8>,
    ) -> Self {
        let at = imported.as_ptr().addr().to_u64();
        let given = given.as_ptr().addr().to_u64();
        reims_vgpu_core::trace::event("imp", &[site.code(), given, length, at]);
        Self(at)
    }
}

impl Drop for ImportTrace {
    fn drop(&mut self) {
        reims_vgpu_core::trace::event("impfree", &[self.0]);
    }
}

/// The runtime's keep-alive of `kept`, what keeps the memory of a no-copy buffer's or heap's
/// import mapped (an [`Alias`], a twin, an [`UnmapHold`](crate::remap::UnmapHold)): the runtime
/// drops it once it freed the device's import of the memory, after the deallocator ran, on
/// the thread that frees the import, which may be one of the runtime's; or, for a buffer whose
/// import keeps another buffer's keep-alive already (one that lies in a live buffer's import),
/// at the buffer's release, after its deallocator (the runtime's `host_memory.rs`, "A
/// keep-alive"). Its drop runs marked ours, so a panic there aborts.
pub(crate) fn keep_alive(kept: impl Send + 'static) -> rt::KeepAlive {
    /// `kept`, dropped marked ours.
    struct Kept<T>(Option<T>);
    impl<T> Drop for Kept<T> {
        fn drop(&mut self) {
            // A callout on the runtime's thread: marked ours, so a panic here aborts.
            let _env = BlockEnv::foreign();
            drop(self.0.take());
        }
    }
    Box::new(Kept(Some(kept)))
}

/// One remap of a gather: `length` bytes from `source`, at `offset` in the alias.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Run {
    /// Where in the alias.
    pub(crate) offset: u64,
    /// The first byte of the guest's pages.
    pub(crate) source: u64,
    /// The bytes, whole pages.
    pub(crate) length: u64,
}

/// The pages of `page` bytes `ranges` lie in, as Metal maps them (the module docs), each a first
/// byte and an end: ranges whose bytes follow on from the previous range's are one run of bytes,
/// and each run of bytes takes the pages it lies in, from the page of its first byte through the
/// page of its last; one of no bytes takes none. `None` past `u64::MAX`.
fn spans(ranges: &[ResourceAddressRange], page: u64) -> Option<Vec<std::ops::Range<u64>>> {
    let mut bytes: Vec<std::ops::Range<u64>> = Vec::new();
    for r in ranges {
        let end = r.address.checked_add(r.length)?;
        match bytes.last_mut() {
            Some(run) if run.end == r.address => run.end = end,
            _ => bytes.push(r.address..end),
        }
    }
    bytes
        .into_iter()
        .filter(|run| !run.is_empty())
        .map(|run| Some(run.start - run.start % page..run.end.checked_next_multiple_of(page)?))
        .collect()
}

/// The remaps of `ranges` into an alias of `size` bytes, whole pages of `page` bytes: the pages
/// of [`spans`] in order, up to `size`, one remap per run of them that follow on in the host's
/// memory. `None` for an empty list or a NULL range, which Metal refuses, and past `u64::MAX`. A
/// first range that starts off its page dies at `sel` ([`off_the_page`]).
pub(crate) fn runs(
    sel: &'static str,
    ranges: &[ResourceAddressRange],
    size: u64,
    page: u64,
) -> Option<Vec<Run>> {
    let first = ranges.first()?;
    if ranges.iter().any(|range| range.address == 0) {
        return None;
    }
    if first.address % page != 0 {
        off_the_page(sel, first.address, first.address % page);
    }
    let mut runs: Vec<Run> = Vec::new();
    let mut offset = 0u64;
    for span in spans(ranges, page)? {
        let length = (span.end - span.start).min(size.saturating_sub(offset));
        if length == 0 {
            break;
        }
        match runs.last_mut() {
            Some(run) if run.source.checked_add(run.length) == Some(span.start) => {
                run.length += length;
            }
            _ => runs.push(Run {
                offset,
                source: span.start,
                length,
            }),
        }
        offset += length;
    }
    Some(runs)
}

/// The death at `sel` of address ranges whose first starts at `address`, `into` bytes into its
/// page ([`runs`]). Metal's `contents` answers the range's bytes from its own address, and its
/// GPU reads and writes them from the start of its page, `into` bytes before it (measured on the
/// M3 Max, macOS 26.7: one range from 0x40 and from 0x1000 into its page, a heap's and a
/// replace's alike, `tests/detach_backing.rs`); a buffer of the runtime's has one address for
/// the CPU and the GPU, the first byte of the memory it imports, so no alias of this platform's
/// gives both.
fn off_the_page(sel: &'static str, address: u64, into: u64) -> ! {
    unrepresentable(
        sel,
        &format!(
            "address ranges whose first starts at {address:#x}, {into:#x} bytes into its page: \
             Metal's contents answers its bytes from that address, and its GPU reads and writes \
             them from the page's start, {into:#x} bytes before it; the runtime's buffer has one \
             address for the CPU and the GPU"
        ),
    )
}

/// The guest's pages of `ranges`, gathered into a fresh alias of `size` bytes rounded up to the
/// host's page ([`runs`]), zero-filled past the ranges' pages, read-only when `read_only` (module
/// docs). `None`, with nothing left mapped, for ranges Metal refuses (an empty list, a NULL or an
/// unmapped range, one with a page the host maps without read access, which the runtime refuses
/// to import, past `size` too). Where the host refuses the alias what it needs past those checks
/// (address space, a remap of the guest's pages, which on Linux takes pages with a file behind
/// them, the block of zeros, a protection), the process ends at `sel` ([`refused_by_host`]):
/// Metal takes the ranges. [`gather_or_nil`] answers nil there instead.
///
/// Where the runtime imports a surface's read-only alias through a writable twin of its files
/// ([`crate::os::TWINS_READ_ONLY`], Linux), a read-only alias's zeros past the ranges' pages map
/// the block of zeros ([`map_zeros`]), as a twin can be made of a file alone: the twin's are
/// zeros of its own then ([`crate::os::import_view`]); an alias the runtime copies reads the
/// block's zeros. Elsewhere they are the fresh range's own,
/// which a writable alias with zeros past its ranges' pages takes from
/// [`crate::os::allocate_aliasable`]: on Linux a file of its own, as the runtime imports a range
/// either as host memory, which RADV refuses for any page with a file behind it, or as dma-bufs,
/// which udmabuf makes of files alone (`crate::os::host_memory`), so an alias of the guest's file
/// pages and anonymous zeros would import neither way; on Mach any allocation.
pub(crate) fn gather(
    sel: &'static str,
    ranges: &[ResourceAddressRange],
    size: u64,
    read_only: bool,
) -> Option<Alias> {
    gather_short(sel, ranges, size, read_only).unwrap_or_else(|short| refused_by_host(sel, &short))
}

/// [`gather`] for a call whose refusal is a nil object (a buffer over a surface, a heap over
/// address ranges): where the host refuses the alias what it needs, the call answers nil, as
/// Metal's does when memory runs out, nothing left mapped
/// (`crate::answer::out_of_host_memory`, counted; in strict mode the process ends at `sel`).
pub(crate) fn gather_or_nil(
    sel: &'static str,
    ranges: &[ResourceAddressRange],
    size: u64,
    read_only: bool,
) -> Option<Alias> {
    gather_short(sel, ranges, size, read_only).unwrap_or_else(|short| {
        crate::answer::out_of_host_memory(sel, || short.0);
        None
    })
}

/// What the host refused [`gather`]'s alias past the checks of its ranges, where Metal's call
/// goes on: the line that names it.
#[derive(Debug)]
struct HostShort(String);

impl HostShort {
    /// The host's refusal of `what`, of `length` bytes, with `kr`.
    fn of(what: &str, length: u64, kr: KernReturn) -> HostShort {
        HostShort(format!(
            "the host refused {what} of {length:#x} bytes (kern_return {}), where Metal's call \
             goes on",
            kr.0
        ))
    }
}

/// [`gather`], `Err` where the host refuses the alias what it needs, nothing left mapped.
fn gather_short(
    sel: &'static str,
    ranges: &[ResourceAddressRange],
    size: u64,
    read_only: bool,
) -> Result<Option<Alias>, HostShort> {
    let vm = OsVm::new();
    let page = vm.page_size();
    let Some(length) = size.checked_next_multiple_of(page) else {
        return Ok(None);
    };
    // Every page of the ranges, those past `size` too, as Metal refuses one there (measured);
    // checked before the alias takes address space, which a hole among the ranges could hold,
    // and before a refusal of the host's.
    let Some(spans) = spans(ranges, page) else {
        return Ok(None);
    };
    for span in spans {
        let Ok(start) = usize::try_from(span.start) else {
            return Ok(None);
        };
        if !vm.readable(HostPtr::from_addr(start), span.end - span.start) {
            return Ok(None);
        }
    }
    let Some(runs) = runs(sel, ranges, length, page) else {
        return Ok(None);
    };
    // The ranges' pages fill the alias from its start; zeros follow them.
    let pages = runs.last().map_or(0, |run| run.offset + run.length);
    let zero_block = read_only && crate::os::TWINS_READ_ONLY;
    let at = if pages < length && !zero_block {
        crate::os::allocate_aliasable(length)
    } else {
        vm.allocate_anywhere(length)
    }
    .map_err(|kr| HostShort::of("address space for an alias", length, kr))?;
    let Some(first) = NonNull::new(at.as_ptr::<u8>()) else {
        return Ok(None);
    };
    // Unmapped as it drops, so nothing is left mapped where the host refuses the rest.
    let alias = Alias::own(first, length);
    // The runs follow on from the alias's start ([`runs`]), so they are one view, made at once:
    // on Linux through dma-bufs of the pages where the host makes them (`crate::os`'s VM).
    let mut sources = Vec::with_capacity(runs.len());
    for run in &runs {
        let Ok(source) = usize::try_from(run.source) else {
            return Ok(None);
        };
        sources.push((HostPtr::from_addr(source), run.length));
    }
    if let Err((i, kr)) = vm.remap_runs_shared_overwrite(at, &sources) {
        let what = "a remap of the guest's pages into an alias";
        return Err(HostShort::of(what, runs[i].length, kr));
    }
    if read_only {
        if zero_block {
            map_zeros(&vm, at, pages, length)
                .map_err(|ZerosRefused { what, length, kr }| HostShort::of(what, length, kr))?;
        }
        if let Err(kr) = vm.protect_max(at, length, VmProt::READ) {
            return Err(HostShort::of("an alias's read-only protection", length, kr));
        }
    }
    Ok(Some(alias))
}

/// The death at `sel` of what the host refused ([`HostShort`]), where Metal's call goes on: this
/// platform's alias of the guest's pages, or its zeros, need what Metal's own call does not, and
/// no answer of Metal's fits without them (a NO where Metal answers YES, or a detach whose object
/// stays over the guest's pages).
fn refused_by_host(sel: &'static str, short: &HostShort) -> ! {
    fatal_with_detail(
        DeathId::Platform {
            kind: DeathKind::Abort,
            call: sel,
        },
        &short.0,
    )
}

/// The line of the death at `detachBacking` of an object the runtime refused to detach
/// ([`Backing::detach`]), `refused` its answer: a texture over its bytes whose image no stand-in
/// of zero bytes presents exactly (a multisample or multi-planar texture, or one with a depth or
/// decoded twin), or one being made over them (`RuntimeGap::DetachTextures`); memory for a
/// stand-in; a lost device. Metal detaches it, and such a texture reads zero bytes and drops its
/// writes (measured on an M4), which the runtime has no form for there.
fn refused_detach(refused: &rt::Error) -> String {
    format!(
        "{}: the detach of a buffer or heap, which the runtime refuses where a texture over its \
         bytes has no stand-in of zero bytes (a multisample or multi-planar texture, or one with \
         a depth or decoded twin) or is being made, or for memory or a lost device; Metal \
         detaches it, and such a texture reads zeros and drops its writes",
        crate::answer::refusal(refused)
    )
}

/// The death at `detachBacking` of a buffer neither replaced nor made over memory of the
/// caller's by `newBufferWithDescriptor:` ([`Backing::detach`]): one of the device's own memory,
/// one made on a surface, or one `newBufferWithBytesNoCopy:length:options:deallocator:` made.
/// What Metal's `contents` answers after such a detach, and when it runs such a buffer's
/// deallocator, is not measured, and core detaches none of them.
fn unmeasured_detach() -> ! {
    fatal_with_detail(
        DeathId::Platform {
            kind: DeathKind::Abort,
            call: DETACH,
        },
        "a detach of a buffer neither replaced nor made by newBufferWithDescriptor: over memory \
         of the caller's (one of the device's own memory, one made on a surface, or one \
         newBufferWithBytesNoCopy: made): what Metal's contents answers after it, and when \
         Metal runs the buffer's deallocator, is not measured",
    )
}

/// The bytes of [`zero_block`]'s block: 1 MiB, whole pages on every host.
const ZERO_BLOCK: u64 = 0x10_0000;

/// A block of [`ZERO_BLOCK`] bytes of zeros, read-only, made once and never freed, that a
/// read-only alias's zeros past its ranges' pages map where the runtime twins read-only memory
/// ([`map_zeros`]); `Err` with the host's answer where it gave none, which a later call asks
/// again, as the host may have room by then. A view can be made of it
/// ([`crate::os::allocate_aliasable`]).
fn zero_block() -> Result<HostPtr, KernReturn> {
    static BLOCK: Mutex<Option<usize>> = Mutex::new(None);
    let mut block = BLOCK.lock();
    if let Some(at) = *block {
        return Ok(HostPtr::from_addr(at));
    }
    let vm = OsVm::new();
    let at = crate::os::allocate_zeros(ZERO_BLOCK)?;
    if let Err(kr) = vm.protect_max(at, ZERO_BLOCK, VmProt::READ) {
        let _ = vm.deallocate(at, ZERO_BLOCK);
        return Err(kr);
    }
    *block = Some(at.addr());
    Ok(at)
}

/// What the host refused [`map_zeros`]: what, of how many bytes, and its answer.
#[derive(Debug)]
pub(crate) struct ZerosRefused {
    pub(crate) what: &'static str,
    pub(crate) length: u64,
    pub(crate) kr: KernReturn,
}

/// Maps the block of [`zero_block`] over the bytes from `from` to `length` of the range at `at`,
/// whole pages, read-only: one remap maps the block, and each further one doubles what is mapped.
/// Where the runtime imports a surface's read-only alias through a writable twin
/// ([`crate::os::TWINS_READ_ONLY`]), the twin of these is zeros of its own
/// ([`crate::os::import_view`]). `Err` where the host gave no block or refused a remap, what it
/// mapped left as it is.
pub(crate) fn map_zeros(
    vm: &OsVm,
    at: HostPtr,
    from: u64,
    length: u64,
) -> Result<(), ZerosRefused> {
    if from >= length {
        return Ok(());
    }
    let block = zero_block().map_err(|kr| ZerosRefused {
        what: "the block of zeros",
        length: ZERO_BLOCK,
        kr,
    })?;
    let (first, length) = (at.wrapping_add(from), length - from);
    let mut mapped = length.min(ZERO_BLOCK);
    if let Err(kr) = vm.remap_shared_overwrite(first, block, mapped) {
        return Err(ZerosRefused {
            what: "a remap of the block of zeros",
            length: mapped,
            kr,
        });
    }
    while mapped < length {
        let more = mapped.min(length - mapped);
        if let Err(kr) = vm.remap_shared_overwrite(first.wrapping_add(mapped), first, more) {
            return Err(ZerosRefused {
                what: "a remap of zeros",
                length: more,
                kr,
            });
        }
        mapped += more;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::os::PAGE;

    fn range(address: u64, length: u64) -> ResourceAddressRange {
        ResourceAddressRange { address, length }
    }

    fn run(offset: u64, source: u64, length: u64) -> Run {
        Run {
            offset,
            source,
            length,
        }
    }

    /// Fresh pages of the host a view can be made of, each byte `byte`.
    fn pages(count: u64, byte: u8) -> HostPtr {
        let at = crate::os::allocate_aliasable(count * PAGE).unwrap();
        // SAFETY: the pages were just allocated, readable and writable.
        unsafe {
            std::ptr::write_bytes(
                at.as_ptr::<u8>(),
                byte,
                usize::try_from(count * PAGE).unwrap(),
            );
        };
        at
    }

    fn read(at: NonNull<u8>, offset: u64) -> u8 {
        // SAFETY: test memory this test mapped readable.
        unsafe {
            at.as_ptr()
                .add(usize::try_from(offset).unwrap())
                .read_volatile()
        }
    }

    fn write(at: HostPtr, offset: u64, byte: u8) {
        // SAFETY: test memory this test mapped writable.
        unsafe { at.wrapping_add(offset).as_ptr::<u8>().write_volatile(byte) }
    }

    fn address(at: HostPtr) -> u64 {
        u64::try_from(at.addr()).unwrap()
    }

    /// A command buffer's parts share its place among the releases of detached objects, which
    /// stays open until the last of them closes, whichever closes first
    #[test]
    fn a_command_buffer_s_parts_share_its_place_until_the_last_closes() {
        let open = |number| DETACHED.state.lock().open.contains_key(&number);
        for first_closes_first in [true, false] {
            let first = Opened::new();
            let second = first.next_part();
            let third = second.next_part();
            let number = first.open.number;
            assert_eq!((second.open.number, third.open.number), (number, number));
            assert_eq!((second.part, third.part), (1, 2));
            let (early, late) = if first_closes_first {
                (first, third)
            } else {
                (third, first)
            };
            drop(early);
            drop(second);
            assert!(open(number), "closed with a part still open");
            drop(late);
            assert!(!open(number), "open once its parts closed");
        }
    }

    /// Host-contiguous ranges make one run, as core cuts its runs at two pages; the ranges'
    /// bytes are cut at the size, and a short last range is remapped to its page's end
    #[test]
    fn runs_join_contiguous_ranges_up_to_the_size() {
        let at = 0x10_0000;
        let ranges = [
            range(at, PAGE),
            range(at + PAGE, PAGE),
            range(at + 0x10_0000, PAGE),
        ];
        assert_eq!(
            runs(REPLACE, &ranges, 3 * PAGE, PAGE),
            Some(vec![
                run(0, at, 2 * PAGE),
                run(2 * PAGE, at + 0x10_0000, PAGE)
            ])
        );
        assert_eq!(
            runs(REPLACE, &ranges, PAGE, PAGE),
            Some(vec![run(0, at, PAGE)])
        );
        assert_eq!(
            runs(REPLACE, &[range(at, 0x10)], PAGE, PAGE),
            Some(vec![run(0, at, PAGE)])
        );
        assert_eq!(runs(REPLACE, &[], PAGE, PAGE), None);
        assert_eq!(
            runs(REPLACE, &[range(at, PAGE), range(0, PAGE)], 2 * PAGE, PAGE),
            None
        );
    }

    /// Each run of ranges whose bytes follow on takes the pages it lies in, whole, as Metal maps
    /// them: a range of part of a page all of it, before another too, and a later range starting
    /// off its page from the page's start, through the page of its last byte; up to the size,
    /// those that follow on in the host's memory one remap: the shapes `tests/detach_backing.rs`
    /// compares with Metal's (its `unaligned_shape`'s 2 to 7), and more
    #[test]
    fn runs_take_the_pages_each_run_of_bytes_lies_in() {
        let g = 0x10_0000;
        let cases = [
            // A range of a quarter page before one of two pages: its page, then the next one's.
            (
                vec![range(g, PAGE / 4), range(g + 2 * PAGE, 2 * PAGE)],
                2 * PAGE,
                vec![run(0, g, PAGE), run(PAGE, g + 2 * PAGE, PAGE)],
            ),
            // A range of a page and 0x40 bytes: two pages.
            (
                vec![range(g, PAGE + 0x40), range(g + 3 * PAGE, PAGE)],
                2 * PAGE,
                vec![run(0, g, 2 * PAGE)],
            ),
            // A later range from 0x40 into its page: from the page's start.
            (
                vec![range(g, PAGE), range(g + 2 * PAGE + 0x40, PAGE)],
                2 * PAGE,
                vec![run(0, g, PAGE), run(PAGE, g + 2 * PAGE, PAGE)],
            ),
            // The same before a third range: through the page of its last byte.
            (
                vec![
                    range(g, PAGE),
                    range(g + 2 * PAGE + 0x40, PAGE),
                    range(g + 5 * PAGE, PAGE),
                ],
                3 * PAGE,
                vec![run(0, g, PAGE), run(PAGE, g + 2 * PAGE, 2 * PAGE)],
            ),
            // Ranges whose bytes follow on across a page: one run of bytes.
            (
                vec![
                    range(g, PAGE + 0x40),
                    range(g + PAGE + 0x40, 2 * PAGE - 0x40),
                ],
                3 * PAGE,
                vec![run(0, g, 3 * PAGE)],
            ),
            // A middle range of a quarter page from its page's start.
            (
                vec![
                    range(g, PAGE),
                    range(g + 2 * PAGE, PAGE / 4),
                    range(g + 4 * PAGE, PAGE),
                ],
                3 * PAGE,
                vec![
                    run(0, g, PAGE),
                    run(PAGE, g + 2 * PAGE, PAGE),
                    run(2 * PAGE, g + 4 * PAGE, PAGE),
                ],
            ),
            // Ranges that do not follow on but share a page: the page twice.
            (
                vec![range(g, PAGE / 4), range(g + PAGE / 2, PAGE)],
                3 * PAGE,
                vec![run(0, g, PAGE), run(PAGE, g, 2 * PAGE)],
            ),
            // A range of part of a page before the next page's range: one remap.
            (
                vec![range(g, PAGE / 4), range(g + PAGE, PAGE)],
                2 * PAGE,
                vec![run(0, g, 2 * PAGE)],
            ),
            // A range of no bytes takes nothing.
            (
                vec![
                    range(g, PAGE),
                    range(g + 0x40 + 4 * PAGE, 0),
                    range(g + 2 * PAGE, PAGE),
                ],
                2 * PAGE,
                vec![run(0, g, PAGE), run(PAGE, g + 2 * PAGE, PAGE)],
            ),
        ];
        for (ranges, size, expected) in cases {
            assert_eq!(
                runs(REPLACE, &ranges, size, PAGE),
                Some(expected),
                "{ranges:x?}"
            );
        }
    }

    /// A first range starting off its page dies at the creator, naming where it starts and what
    /// Metal's CPU and GPU see; ranges Metal refuses (a NULL one, an unmapped page) are refused
    /// first, as Metal makes nothing of them
    #[test]
    fn a_first_range_off_its_page_dies() {
        use reims_vgpu_core::raise::death::DeathKind;
        use reims_vgpu_core::testkit::capture::catch_death;
        let guest = pages(3, 0);
        for (into, sel) in [(0x40, REPLACE), (PAGE / 4, "newHeapWithDescriptor:")] {
            let ranges = [range(address(guest) + into, 2 * PAGE)];
            *crate::answer::LAST_LINE.lock() = None;
            let site = catch_death(|| gather(sel, &ranges, 2 * PAGE, false).is_some()).unwrap_err();
            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{line}");
            assert!(
                line.contains(&format!("{into:#x} bytes into its page"))
                    && line.contains("Metal's contents answers its bytes from that address"),
                "{line}"
            );
        }
        let null = range(0, PAGE);
        let off = range(address(guest) + 0x40, PAGE);
        assert_eq!(runs(REPLACE, &[off, null], 2 * PAGE, PAGE), None);
        let vm = OsVm::new();
        assert!(
            vm.deallocate(guest.wrapping_add(2 * PAGE), PAGE)
                .is_success()
        );
        let unmapped = [range(address(guest) + 0x40, 2 * PAGE)];
        assert!(gather(REPLACE, &unmapped, 2 * PAGE, false).is_none());
        assert!(vm.deallocate(guest, 2 * PAGE).is_success());
    }

    /// The alias shows the guest's pages, in order, and the guest's pages its writes; past the
    /// ranges it is zero-filled pages of its own
    #[test]
    fn an_alias_is_the_guests_pages_both_ways() {
        let guest = pages(4, 0xa1);
        write(guest, 3 * PAGE, 0xb2);
        let ranges = [
            range(address(guest), PAGE),
            range(address(guest) + PAGE, PAGE),
            range(address(guest) + 3 * PAGE, PAGE),
        ];
        let alias = gather(REPLACE, &ranges, 4 * PAGE, false).unwrap();
        let first = alias.first();
        assert_eq!(
            [0, PAGE, 2 * PAGE, 3 * PAGE].map(|at| read(first, at)),
            [0xa1, 0xa1, 0xb2, 0]
        );
        write(guest, 5, 0x5c);
        assert_eq!(read(first, 5), 0x5c, "the guest's write shows in the alias");
        write(HostPtr::from_ptr(first.as_ptr()), 2 * PAGE + 7, 0x6d);
        // SAFETY: the guest's pages are mapped.
        let seen = unsafe {
            guest
                .wrapping_add(3 * PAGE + 7)
                .as_ptr::<u8>()
                .read_volatile()
        };
        assert_eq!(seen, 0x6d, "the alias's write shows in the guest's page");
        drop(alias);
        assert!(OsVm::new().deallocate(guest, 4 * PAGE).is_success());
    }

    /// A NULL or unmapped range, one with a page mapped without access, within the size or past
    /// it, or none, gathers nothing; read-only pages gather
    #[test]
    fn refused_ranges_gather_nothing() {
        use reims_vgpu_core::platform::os::vm::VmProt;
        let vm = OsVm::new();
        // A hole of one page between two mapped ones, where no alias of two pages fits.
        let around = pages(3, 0);
        let hole = around.wrapping_add(PAGE);
        assert!(vm.deallocate(hole, PAGE).is_success());
        let shut = pages(2, 0);
        vm.protect_max(shut.wrapping_add(PAGE), PAGE, VmProt(0))
            .unwrap();
        // A page unmapped before free address space, where the alias would fit.
        let last = pages(2, 0);
        assert!(vm.deallocate(last.wrapping_add(PAGE), PAGE).is_success());
        for ranges in [
            vec![],
            vec![range(0, PAGE)],
            vec![range(address(around), PAGE), range(address(hole), PAGE)],
            vec![range(address(shut), 2 * PAGE)],
            vec![range(address(last), 2 * PAGE)],
        ] {
            assert!(
                gather(REPLACE, &ranges, 2 * PAGE, false).is_none(),
                "{ranges:?}"
            );
        }
        // Past the size, as within it.
        for ranges in [
            vec![range(address(shut), 2 * PAGE)],
            vec![
                range(address(shut), PAGE),
                range(address(shut) + PAGE, PAGE),
            ],
            vec![range(address(last), 2 * PAGE)],
        ] {
            assert!(
                gather(REPLACE, &ranges, PAGE, false).is_none(),
                "{ranges:?}"
            );
        }
        // A later range that ends in its page before an unmapped one: only its page is read.
        let ends = [
            range(address(around), PAGE),
            range(address(last) + 0x40, PAGE - 0x40),
        ];
        assert!(gather(REPLACE, &ends, 2 * PAGE, false).is_some());
        let read_only = pages(1, 0);
        vm.protect_max(read_only, PAGE, VmProt::READ).unwrap();
        assert!(gather(REPLACE, &[range(address(read_only), PAGE)], PAGE, false).is_some());
        for page in [around, around.wrapping_add(2 * PAGE), last, read_only] {
            assert!(vm.deallocate(page, PAGE).is_success());
        }
        assert!(vm.deallocate(shut, 2 * PAGE).is_success());
    }

    /// Runtime identities of `N` objects, made and released here, which no other object takes.
    fn ids<const N: usize>() -> [rt::ObjectId; N] {
        let dev = crate::gpu::device::tests::dev();
        std::array::from_fn(|_| {
            dev.raw
                .new_buffer(16, rt::ResourceOptions::default())
                .unwrap()
                .object_id()
        })
    }

    /// A registry of its own, empty.
    fn registry() -> Registry {
        Registry::new()
    }

    /// A command buffer made now in `d` by `task`, or by none: its number. Its place stays in
    /// `d` until it closes there.
    fn made(d: &Registry, task: Option<&Arc<TaskLife>>) -> u64 {
        d.open(task.map(Arc::downgrade)).number
    }

    /// Command buffer `made` committed now in `d`.
    fn committed(d: &Registry, made: u64) -> u64 {
        let open = Arc::clone(&d.state.lock().open[&made]);
        d.commit(&open);
        made
    }

    /// The numbers of the command buffers open in `d`.
    fn open(d: &Registry) -> Vec<u64> {
        d.state.lock().open.keys().copied().collect()
    }

    /// A replace answers NO for an attached object and for ranges of another total, and YES once
    /// the runtime took the alias, which leaves the object attached. A detach is the runtime's,
    /// once; `contents` then stays the memory the object was over: what a no-copy buffer was made
    /// over, whose deallocator's run, which the runtime hands over as it lets go of it, is held
    /// until the first replace, or a replace's alias. The runtime's NO leaves the object
    /// attached, and its refusal ends the process at `detachBacking`, naming it and what Metal's
    /// textures read; so does a detach of a buffer neither replaced nor made over memory of the
    /// caller's
    #[test]
    fn replace_answers_as_metal() {
        use reims_vgpu_core::testkit::capture::catch_death;
        let guest = pages(2, 0xa1);
        let ranges = [range(address(guest), 2 * PAGE)];
        let [id, other] = ids();
        let site = catch_death(|| Backing::new(false, other).detach(|| Ok(true)));
        assert_eq!(
            site.map_err(|site| (site.kind, site.file)),
            Err((DeathKind::Abort, DETACH)),
            "no memory of the caller's"
        );
        let backing = Backing::new(false, id);
        let made_over = pages(2, 0x11);
        let made_at = NonNull::new(made_over.as_ptr::<u8>()).unwrap();
        let held = Arc::new(Held::default());
        backing.made_over_own(Own::new(made_at, Arc::clone(&held)));
        let ran = Arc::new(AtomicBool::new(false));
        let deallocate = || {
            let ran = Arc::clone(&ran);
            Box::new(move || ran.store(true, Ordering::Release)) as Box<dyn FnOnce() + Send>
        };
        let rebinds = std::cell::Cell::new(0);
        let rebind = |_: NonNull<u8>, _: rt::KeepAlive| {
            rebinds.set(rebinds.get() + 1);
            Ok(())
        };
        // The runtime's detaches, and its answer to each.
        let detaches = std::cell::Cell::new(0);
        let detach = |answer: Result<bool, rt::Error>| {
            let detaches = &detaches;
            move || {
                detaches.set(detaches.get() + 1);
                answer
            }
        };
        let refusal = || rt::Error::NotImplemented(rt::RuntimeGap::DetachTextures);
        assert_eq!(
            backing.replace(&ranges, false, 2 * PAGE, 2 * PAGE, rebind),
            Ok(false),
            "attached"
        );
        backing.detach(detach(Ok(false)));
        assert!(!backing.is_detached(), "the runtime's NO");
        assert_eq!(backing.kept_contents(), None);
        held.run(deallocate());
        assert!(ran.load(Ordering::Acquire), "not held past a NO");
        ran.store(false, Ordering::Release);
        backing.detach(detach(Ok(true)));
        backing.detach(detach(Ok(true)));
        assert!(backing.is_detached());
        assert_eq!(detaches.take(), 2, "never replaced: detached once");
        assert_eq!(backing.kept_contents(), Some(made_at), "contents stays");
        held.run(deallocate());
        assert!(!ran.load(Ordering::Acquire), "held at the detach");
        for length in [PAGE, 3 * PAGE] {
            assert_eq!(
                backing.replace(&ranges, false, length, length, rebind),
                Ok(false),
                "{length:#x}"
            );
        }
        assert_eq!(rebinds.get(), 0);
        assert!(!ran.load(Ordering::Acquire), "held past refused replaces");
        assert_eq!(
            backing.replace(&ranges, false, 2 * PAGE, 2 * PAGE, rebind),
            Ok(true)
        );
        assert_eq!(rebinds.get(), 1);
        assert!(ran.load(Ordering::Acquire), "run inside the first replace");
        assert!(!backing.is_detached());
        let alias = backing.alias.lock().last.as_ref().map(|alias| alias.at);
        backing.detach(detach(Ok(true)));
        assert_eq!(detaches.take(), 1);
        assert_eq!(backing.kept_contents(), alias, "contents stays the alias");
        assert_eq!(
            backing.replace(&ranges, false, 2 * PAGE, 2 * PAGE, rebind),
            Ok(true)
        );
        assert_eq!(backing.kept_contents(), None);
        let site = catch_death(|| backing.detach(detach(Err(refusal())))).unwrap_err();
        assert_eq!((site.kind, site.file), (DeathKind::Abort, DETACH));
        assert_eq!(detaches.take(), 1);
        let line = refused_detach(&refusal());
        assert!(
            line.starts_with("the Vulkan runtime refused it: ")
                && line.contains("Metal detaches it, and such a texture reads zeros"),
            "{line}"
        );
        let fixed = Backing::fixed();
        fixed.detach(detach(Ok(true)));
        assert!(!fixed.is_detached(), "a fixed backing never detaches");
        assert_eq!(detaches.take(), 0);
        assert_eq!(
            fixed.replace(&ranges, false, 2 * PAGE, 2 * PAGE, rebind),
            Ok(false)
        );
        let vm = OsVm::new();
        assert!(vm.deallocate(guest, 2 * PAGE).is_success());
        assert!(vm.deallocate(made_over, 2 * PAGE).is_success());
    }

    /// A no-copy buffer never replaced runs its deallocator as it goes where a detach held the
    /// run, as Metal's runs it as the buffer is destroyed
    #[test]
    fn a_held_run_is_made_as_the_buffer_goes() {
        let [id] = ids();
        let backing = Backing::new(false, id);
        let made_over = pages(1, 0x11);
        let held = Arc::new(Held::default());
        let made_at = NonNull::new(made_over.as_ptr::<u8>()).unwrap();
        backing.made_over_own(Own::new(made_at, Arc::clone(&held)));
        backing.detach(|| Ok(true));
        let ran = Arc::new(AtomicBool::new(false));
        let run = Arc::clone(&ran);
        held.run(Box::new(move || run.store(true, Ordering::Release)));
        assert!(!ran.load(Ordering::Acquire), "held at the detach");
        drop(backing);
        assert!(ran.load(Ordering::Acquire), "run as it went");
        assert!(OsVm::new().deallocate(made_over, PAGE).is_success());
    }

    /// A deallocator's run is made at once but while a detach holds it; then at the release,
    /// once, and at once again after it
    #[test]
    fn a_held_run_is_made_at_its_release() {
        let held = Held::default();
        let runs = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let run = || {
            let runs = Arc::clone(&runs);
            Box::new(move || {
                runs.fetch_add(1, Ordering::AcqRel);
            }) as Box<dyn FnOnce() + Send>
        };
        let count = || runs.load(Ordering::Acquire);
        held.run(run());
        assert_eq!(count(), 1);
        held.hold();
        held.run(run());
        assert_eq!(count(), 1, "held");
        held.release();
        assert_eq!(count(), 2, "released");
        held.release();
        assert_eq!(count(), 2, "once");
        held.run(run());
        assert_eq!(count(), 3, "at once after it");
    }

    /// A read-only alias reads the guest's pages, and zeros past them, and its pages cannot be
    /// made writable
    #[test]
    fn a_read_only_alias_reads_zeros_past_its_pages() {
        let guest = pages(1, 0xa5);
        let alias = gather(REPLACE, &[range(address(guest), PAGE)], 3 * PAGE, true).unwrap();
        assert_eq!(alias.length, 3 * PAGE);
        assert_eq!(read(alias.at, 0), 0xa5);
        for offset in [PAGE, 2 * PAGE + 5, 3 * PAGE - 1] {
            assert_eq!(read(alias.at, offset), 0, "{offset:#x}");
        }
        let at = HostPtr::from_ptr(alias.at.as_ptr());
        assert!(!crate::os::writable(at, alias.length), "writable");
        drop(alias);
        assert!(OsVm::new().deallocate(guest, PAGE).is_success());
    }

    /// An object fails the command buffers that use it while it is detached; released detached,
    /// those made before the release until they close; replaced, none here, committed before the
    /// replace or not (the runtime fails one that names it, module docs). No release is kept
    /// while no command buffer is open
    #[test]
    fn detached_objects_fail_their_command_buffers() {
        let d = registry();
        let [buffer, heap] = ids();
        let first = made(&d, None);
        assert!(!d.any(first));
        d.detached(buffer);
        assert!(d.any(first) && d.fails(&[buffer], first) && !d.fails(&[heap], first));
        d.attached(buffer);
        assert!(!d.any(first) && !d.fails(&[buffer], first), "replaced");
        let first = committed(&d, first);
        d.detached(buffer);
        d.released(buffer);
        let second = made(&d, None);
        assert!(d.state.lock().detached.is_empty());
        assert!(d.fails(&[buffer], first), "made before the release");
        assert!(
            !d.fails(&[buffer], second) && !d.any(second),
            "made after it"
        );
        d.detached(heap);
        let second = committed(&d, second);
        assert!(
            d.fails(&[heap], second) && d.fails(&[heap], first),
            "detached"
        );
        assert!(d.fails(&[buffer, heap], second) && !d.fails(&[], second));
        d.attached(heap);
        assert!(!d.fails(&[heap], second), "committed before the replace");
        d.closed(first);
        assert!(
            d.state.lock().released.is_empty(),
            "no open command buffer made before it"
        );
        d.closed(second);
        assert!(open(&d).is_empty());
        d.detached(buffer);
        d.released(buffer);
        let state = d.state.lock();
        assert!(state.detached.is_empty() && state.released.is_empty());
    }

    /// A command buffer open before an object's release, the only object detached, can fail by
    /// it at every moment of the release, as a reader without the lock sees it ([`Registry::any`]):
    /// between the release's stores as before and after them, the object is detached or released
    #[test]
    fn a_release_is_seen_by_a_reader_at_every_moment() {
        static SEEN: Mutex<Vec<bool>> = Mutex::new(Vec::new());
        let d = Registry {
            between: Some(|d| SEEN.lock().push(d.any(0))),
            ..Registry::new()
        };
        let [buffer] = ids();
        let open = made(&d, None);
        assert_eq!(open, 0);
        let open = committed(&d, open);
        d.detached(buffer);
        assert!(d.any(open));
        d.released(buffer);
        assert_eq!(*SEEN.lock(), [true], "between the stores");
        assert!(d.any(open) && d.fails(&[buffer], open), "after them");
        d.closed(open);
    }

    /// Releases made while an old command buffer stays open fail the command buffers made before
    /// each, and none made after the last; as the oldest open command buffers close, the releases
    /// none of those still open was made before are forgotten, in order, the later ones kept
    #[test]
    fn releases_are_forgotten_in_order_as_the_oldest_command_buffers_close() {
        let d = registry();
        let objects: [rt::ObjectId; 6] = ids();
        let old = made(&d, None);
        let [first, second, third, rest @ ..] = objects;
        for id in [first, second] {
            d.detached(id);
            d.released(id);
        }
        let middle = made(&d, None);
        for id in std::iter::once(third).chain(rest) {
            d.detached(id);
            d.released(id);
        }
        let new = made(&d, None);
        assert!(d.any(old) && d.any(middle) && !d.any(new));
        assert!(d.fails(&[first], old) && !d.fails(&[first], middle) && d.fails(&[third], middle));
        d.closed(old);
        assert!(
            !d.fails(&[first], old) && !d.fails(&[second], old),
            "made before the oldest open"
        );
        assert_eq!(d.state.lock().released.len(), 4);
        assert!(d.fails(&[third], middle) && d.any(middle));
        d.closed(middle);
        let state = d.state.lock();
        assert!(state.released.is_empty() && state.releases.is_empty());
        drop(state);
        assert!(!d.any(new) && !d.any(0), "no release kept");
        d.closed(new);
    }

    /// A command buffer committed while a buffer it fills is detached fails with code 8 as it is
    /// submitted, as Metal's (`tests/detach_backing.rs`'s `never_replaced`), on every host: the
    /// detached buffer lies in no memory, so no host keeps what a command that reaches it unnamed
    /// writes
    #[test]
    fn a_commit_while_detached_fails_on_every_host() {
        use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder};
        use reims_vgpu_core::platform::command::{CommandBuffer, CommandQueue};
        use reims_vgpu_core::platform::device::GpuDevice;
        use reims_vgpu_core::platform::resource::Buffer;
        use reims_vgpu_core::platform::types::NsRange;
        use reims_vgpu_core::platform::values::{BufferDesc, OptionSet};
        use reims_vgpu_core::raise::OnFault;
        const TEST: &str = "gpu::backing::tests::a_commit_while_detached_fails_on_every_host";
        if std::env::var("RVG_CHILD").as_deref() == Ok(TEST) {
            reims_vgpu_core::testkit::capture::set_capture_enabled(false);
            let d = crate::gpu::device::tests::dev();
            let memory = pages(1, 0xd0);
            // SAFETY: the pages stay mapped for the process, which exits below.
            let contents = unsafe { reims_vgpu_core::platform::HostRange::new(memory, PAGE) };
            let buffer = d
                .new_buffer(BufferDesc {
                    length: PAGE,
                    options: vec![OptionSet::StorageMode(0)],
                    pinned_gpu_address: None,
                    contents: Some(contents),
                    no_copy: true,
                    deallocator: None,
                })
                .unwrap()
                .unwrap();
            buffer.detach_backing().unwrap();
            let queue = d.new_command_queue().unwrap();
            let cb = queue.command_buffer(OnFault::Return).unwrap().unwrap();
            let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
            for command in [
                BlitCmd::FillBuffer {
                    buffer: &buffer,
                    range: NsRange::new(0, 16),
                    value: 0x5c,
                },
                BlitCmd::EndEncoding,
            ] {
                e.encode(OnFault::Return, command).unwrap();
            }
            cb.commit(OnFault::Return).unwrap();
            cb.wait_until_completed(OnFault::Return).unwrap();
            let code = cb.error().map_or(0, |e| e.code);
            println!("status {} code {code}", cb.status().0);
            std::process::exit(0);
        }
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", TEST, "--nocapture", "--test-threads=1"])
            .env("RVG_CHILD", TEST)
            .output()
            .unwrap();
        let (stdout, stderr) = (
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
        let death = stderr
            .lines()
            .find(|l| l.starts_with("reims-vgpu: fatal: "));
        assert!(out.status.success() && death.is_none(), "{out:?}");
        assert!(stdout.contains("status 5 code 8"), "{stdout}");
    }

    /// A command buffer took the object from its submission that used it until it completed,
    /// whether it completed with an error or not: a detach finds it in flight while it waits on
    /// an event after its fill of the object, and none of its noted submissions in flight in its
    /// completed handler, as Metal lets go of what a command buffer used as its completed
    /// handlers start, and of a failed one's before them ([`in_flight`], module docs, "In
    /// flight"); the failed one's fill of the object ran, and a later submission failed with
    /// code 8 by a detached use
    #[test]
    fn a_command_buffer_took_the_object_until_it_completed() {
        use crate::gpu::command::VulkanCommandBuffer;
        use crate::gpu::resource::VulkanBuffer;
        use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder};
        use reims_vgpu_core::platform::command::{CommandBuffer, CommandQueue};
        use reims_vgpu_core::platform::device::GpuDevice;
        use reims_vgpu_core::platform::event::{EventRef, SharedEvent};
        use reims_vgpu_core::platform::resource::Buffer;
        use reims_vgpu_core::platform::types::NsRange;
        use reims_vgpu_core::platform::values::{BufferDesc, OptionSet};
        use reims_vgpu_core::raise::OnFault;
        fn fill(cb: &VulkanCommandBuffer, buffer: &VulkanBuffer) {
            let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
            for command in [
                BlitCmd::FillBuffer {
                    buffer,
                    range: NsRange::new(0, 16),
                    value: 0x5c,
                },
                BlitCmd::EndEncoding,
            ] {
                e.encode(OnFault::Return, command).unwrap();
            }
        }
        let d = crate::gpu::device::tests::dev();
        // A no-copy buffer over a page of the test's, which stays mapped for the process.
        let no_copy = || {
            // SAFETY: the page stays mapped for the process.
            let contents = unsafe { reims_vgpu_core::platform::HostRange::new(pages(1, 0), PAGE) };
            d.new_buffer(BufferDesc {
                length: PAGE,
                options: vec![OptionSet::StorageMode(0)],
                pinned_gpu_address: None,
                contents: Some(contents),
                no_copy: true,
                deallocator: None,
            })
            .unwrap()
            .unwrap()
        };
        let used = no_copy();
        let other = no_copy();
        let detached = no_copy();
        detached.detach_backing().unwrap();
        let (id, made) = (used.backing.id.unwrap(), used.backing.made);
        let queue = d.new_command_queue().unwrap();
        let (event, gate) = (d.new_shared_event().unwrap(), d.new_shared_event().unwrap());
        let mut before = Vec::new();
        let mut seen = Vec::new();
        for fails in [false, true] {
            let cb = queue.command_buffer(OnFault::Return).unwrap().unwrap();
            let (taken, handler) = std::sync::mpsc::channel();
            cb.add_completed_handler(
                OnFault::Return,
                Box::new(move |info| {
                    let _ = taken.send((info.status.0, in_flight(id, made).len()));
                }),
            )
            .unwrap();
            fill(&cb, &used);
            if fails {
                cb.encode_signal(Some(EventRef::Shared(&event)), 1).unwrap();
                fill(&cb, &detached);
            } else {
                cb.encode_wait(Some(EventRef::Shared(&gate)), 1).unwrap();
                fill(&cb, &other);
            }
            cb.commit(OnFault::Return).unwrap();
            if !fails {
                // The first submission is noted as the runtime submits it, read for up to 5 s;
                // the second waits for the gate, so the command buffer has not completed.
                let mut taking = in_flight(id, made).len();
                for _ in 0..5000 {
                    if taking != 0 {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(1));
                    taking = in_flight(id, made).len();
                }
                before.push((cb.status().0 < 4, taking));
                gate.set_signaled_value(1).unwrap();
            }
            cb.wait_until_completed(OnFault::Return).unwrap();
            seen.push(handler.recv().unwrap());
        }
        assert_eq!(
            (before, seen),
            (vec![(true, 1)], vec![(4, 0), (5, 0)]),
            "(not completed, command buffers taking it) waiting; (status, them) in the handlers"
        );
    }

    /// A command buffer of an isolated task that ended holds no release: one that died mid-encode
    /// is never released, and never commits. One that committed holds it until it completed
    #[test]
    fn an_ended_tasks_command_buffer_holds_no_release() {
        let d = registry();
        let task = TaskLife::new();
        let leaked = made(&d, Some(&task));
        let sent = made(&d, Some(&task));
        let sent = committed(&d, sent);
        let host = made(&d, None);
        let [buffer] = ids();
        d.detached(buffer);
        d.released(buffer);
        assert!(d.fails(&[buffer], leaked) && d.fails(&[buffer], sent));
        d.closed(host);
        assert!(d.fails(&[buffer], leaked), "the task is running");
        let other = made(&d, None);
        assert!(task.exit());
        d.closed(other);
        assert_eq!(open(&d), [sent]);
        assert!(d.fails(&[buffer], sent), "committed");
        d.closed(sent);
        assert!(open(&d).is_empty() && d.state.lock().released.is_empty());
    }

    /// A command buffer whose task ended after its commit's check of the task's end, so that a
    /// close let go of its place as holding nothing before the commit, is open again from its
    /// commit until it closes, so a detach finds what it submits ([`Registry::noted_since`]); a
    /// later close keeps it
    #[test]
    fn a_commit_after_its_tasks_end_keeps_its_place() {
        let d = registry();
        let task = TaskLife::new();
        let late = d.open(Some(Arc::downgrade(&task)));
        let other = made(&d, None);
        assert!(task.exit());
        d.closed(other);
        assert!(
            open(&d).is_empty(),
            "let go of: its task ended, uncommitted"
        );
        d.commit(&late);
        assert_eq!(open(&d), [late.number], "open again");
        let after = made(&d, None);
        d.closed(after);
        assert_eq!(open(&d), [late.number], "committed: kept");
        d.closed(late.number);
        assert!(open(&d).is_empty());
    }

    /// The command buffers of an ended task that lie behind an older open one go as the open ones
    /// double since the last such sweep, at a close, and not one a close before: each close costs
    /// O(1) over the closes, where a walk of every open one at each close cost O(open)
    #[test]
    fn an_ended_tasks_command_buffers_behind_an_open_one_go_as_the_open_ones_double() {
        let d = registry();
        let task = TaskLife::new();
        let oldest = made(&d, None);
        let leaked: Vec<u64> = (0..20).map(|_| made(&d, Some(&task))).collect();
        assert!(task.exit());
        let mut host: Vec<u64> = (0..11).map(|_| made(&d, None)).collect();
        d.closed(host.remove(0));
        assert_eq!(open(&d).len(), 31, "under twice 16 open after the close");
        assert!(open(&d).contains(&leaked[0]));
        host.extend([made(&d, None), made(&d, None)]);
        d.closed(host.pop().unwrap());
        assert_eq!(
            open(&d).len(),
            12,
            "swept at twice 16: the oldest and eleven of the host's"
        );
        assert!(leaked.iter().all(|n| !open(&d).contains(n)));
        assert_eq!(d.state.lock().fewest, 12);
        d.closed(oldest);
        for n in host {
            d.closed(n);
        }
        assert!(open(&d).is_empty());
    }

    /// The length of a run of pages in [`Pages`]' tests.
    const RUN: u64 = 0x4000;

    /// Whether an object's bytes meet its pages is what its memory there answers: none over memory
    /// of its own, whether any of them lie in its pages, and none once a detach zeroes it, until
    /// a replace gives it pages again
    #[test]
    #[expect(clippy::single_range_in_vec_init, reason = "lists of host ranges")]
    fn meeting_pages_is_whether_the_bytes_lie_in_them() {
        let pages = Arc::new(Pages::default());
        let all = [
            0..0,
            0..1,
            0..RUN,
            RUN - 1..RUN + 1,
            2 * RUN..3 * RUN,
            3 * RUN..4 * RUN,
        ];
        let agree = |pages: &Pages| {
            for bytes in &all {
                let memory = pages.in_pages(bytes).map(|ranges| !ranges.is_empty());
                assert_eq!(pages.meets_pages(bytes), memory, "{bytes:x?}");
            }
        };
        agree(&pages);
        assert_eq!(pages.meets_pages(&(0..RUN)), None);
        pages.set(vec![
            0x10_0000..0x10_0000 + RUN,
            0x30_0000..0x30_0000 + 2 * RUN,
        ]);
        agree(&pages);
        assert_eq!(pages.meets_pages(&(RUN - 1..RUN + 1)), Some(true));
        assert_eq!(pages.meets_pages(&(3 * RUN..4 * RUN)), Some(false));
        pages.zero();
        agree(&pages);
        assert_eq!(pages.meets_pages(&(0..RUN)), None);
        pages.set(vec![0x50_0000..0x50_0000 + RUN]);
        agree(&pages);
        assert_eq!(pages.meets_pages(&(0..1)), Some(true));
    }

    /// An object's bytes are found in its pages by a search from the run that holds their first
    /// byte, as a walk of the runs from the first finds them: in each run they reach, in order, at
    /// each run's edges, past their end, over runs of no length, and for no bytes; and they meet
    /// the pages where that finds any
    #[test]
    #[expect(clippy::single_range_in_vec_init, reason = "lists of host ranges")]
    fn bytes_in_pages_are_found_by_a_search_as_by_a_walk() {
        let scattered: Vec<std::ops::Range<u64>> = (0..64u64)
            .map(|i| {
                let at = 0x100_0000 + ((i * 37) % 64) * 2 * RUN;
                at..at + RUN * (1 + i % 3)
            })
            .collect();
        let lists = [
            vec![
                0x10_000..0x14_000,
                0x50_000..0x50_000,
                0x40_000..0x48_000,
                0x20_000..0x24_000,
            ],
            vec![0x10_000..0x14_000],
            vec![0x50_000..0x50_000],
            vec![],
            scattered,
        ];
        for runs in lists {
            let paged = Paged::of(runs.clone());
            let total: u64 = runs.iter().map(|r| r.end - r.start).sum();
            let mut edges = vec![0, 1, 0x3fff, 0x4000, 0x4001, 0xbfff, 0xc000, 0xffff];
            edges.extend([total.saturating_sub(1), total, total + 1]);
            edges.extend((1..12).map(|i| i * total / 11 + i % 3));
            for &start in &edges {
                for &end in &edges {
                    let bytes = start..end;
                    let walked = crate::os::in_runs(&runs, &bytes);
                    assert_eq!(
                        paged.meets(&bytes),
                        !walked.is_empty(),
                        "{runs:x?} {bytes:x?}"
                    );
                    assert_eq!(paged.in_runs(&bytes), walked, "{runs:x?} {bytes:x?}");
                }
            }
        }
    }
}
