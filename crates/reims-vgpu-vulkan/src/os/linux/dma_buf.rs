//! Dma-bufs of the pages behind a range the backing table resolves ([`super::files`]), for a
//! device that imports no shared memory as host memory.
//!
//! RADV imports host memory as an amdgpu userptr, which Mesa asks for with
//! `AMDGPU_GEM_USERPTR_ANONONLY`, so the kernel refuses every page with a file behind it: the
//! guest's RAM (QEMU's `memory-backend-memfd`) and every range the library makes a view of are
//! shared mappings of memfds, and none of them imports. `/dev/udmabuf` makes a dma-buf of those
//! same memfd pages (`UDMABUF_CREATE_LIST`), and RADV imports that
//! (`VK_EXT_external_memory_dma_buf`). Measured on a Radeon 780M (RADV of Mesa 26.0.8, Linux 7.0,
//! 2026-10-11), with no runtime between:
//! - a sealed memfd mapped shared: `vkGetMemoryHostPointerPropertiesEXT` answers
//!   `VK_ERROR_UNKNOWN`; private anonymous memory imports;
//! - a udmabuf of two pieces of that memfd 6 MiB apart imports as types 5 and 10 (host-visible,
//!   coherent and cached); the GPU read the bytes the CPU wrote through the shared mapping, and a
//!   fill across both pieces reached the file's pages, seen through that mapping after the fence,
//!   with no map, flush or invalidate of the import;
//! - a udmabuf of one piece takes about 9 ms per 64 MiB, the kernel pinning its pages; one of
//!   16,201 scattered 4 KiB pieces (with the limits below raised) 19 ms the first time and 3 ms
//!   after, and RADV imports it in about 50 µs; two over the same pages both import.
//!
//! **Chunks.** udmabuf takes at most `list_limit` pieces (1024 by default) and `size_limit_mb` MiB
//! (64) in one dma-buf, parameters of the `udmabuf` module only root can raise, and the library
//! asks no one to. An import of a macOS 26 guest's boot took up to 16,201 runs of file pages
//! (`docs/qemu.md`, "Linux hosts: AMD GPUs and udmabuf"), so a window is cut into chunks, each a
//! dma-buf within the limits the host has now ([`limits`]), cut only where the runtime can bind
//! the next ([`chunks`]); the runtime binds them in order behind one buffer.
//!
//! **Costs** (measured on the same host, 2026-10-12): a udmabuf pins every page it takes,
//! allocating those never touched (64 MiB of untouched memory in 18 ms), and keeps showing a page
//! the file later drops. Each import the device holds adds about 0.55 µs to every submission
//! (RADV passes each in a list on every submit): 1,024 live imports cost 642 µs a submit, 4,096
//! about 2.5 ms. So the fewest chunks are made, and an import goes with its buffer.
//!
//! **Groups.** Views of guest pages that no-copy buffers lie over are imported one each, so a
//! process with many of them pays for each at every submission. The graphics device's surface
//! worker gathers the views a process makes at once and hands them to [`group_views`]
//! (`crate::core::platform::os::vm::HostVm::group_views`): their pages go into group dma-bufs
//! ([`of_group`]), each as many views back to back as fit udmabuf's limits, and each view's place
//! is registered ([`GROUPED`]) while its membership lives. The import source then answers a
//! buffer's or a heap's request over a registered view's pages with that piece of the group's
//! file ([`piece_of`], `super::import`'s `host_memory`), the file shared by its `Arc`, and the
//! runtime imports the file once for all of them. A member's place goes as its view does, before
//! the view is unmapped, so a view mapped later at its address is never answered with its pages;
//! a request after it, or once the group's file is gone (a device's recovery asking again), is
//! answered as one over no group is (below). While any member's import lives, every member's
//! pages stay pinned in the file and mapped in the device's import: a group holds at most
//! [`GROUP_BYTES`], 64 MiB, whatever udmabuf takes.
//!
//! **Regions.** A buffer's or a heap's request over no grouped view, whose pages are the guest's
//! RAM mapped writable (the leases' registrations, `crate::os::files`), is answered with pieces of
//! the RAM's region dma-bufs where they fit the request ([`ram_pieces`]): each file of the RAM is
//! cut into regions of [`REGION_BYTES`], 64 MiB at multiples of it in the file (udmabuf's
//! `size_limit_mb`, down to a power of two, where a host has it lower), each one dma-buf of its
//! one run of pages, made at its first piece and shared by its `Arc`, so a device that lists
//! every import in every submission (RADV) lists one a region whatever objects lie in it. A
//! request takes one piece where its pages are one run of the file inside one region, at an
//! offset its alignment divides: a heap over a pages block of one run, which lies in the RAM
//! itself, a buffer over the RAM, a view of one run. Pages across two regions, or several runs,
//! take a piece of each where the runtime binds several chunks (each boundary a multiple of its
//! granule from the start, and the bytes whole granules, as a pad would be a region's other
//! pages). Everything else, a view of scattered pages among it (4 KiB runs against RADV's 64 KiB
//! granule), takes a dma-buf of its own, as memory of the library's own and read-only pages do.
//! The table of regions ([`REGIONS`]) holds each file weakly: the runtime's imports of it hold
//! it, and so does each import source that answered with a piece of it, which the runtime keeps
//! with the imports it made (`super::import`'s `host_memory`), and so do the imports of it the
//! devices keep (below, "Kept imports"), so it lives while any buffer or heap bound in it does and
//! for their linger after the last, and a later request in the region makes it again (the trace's
//! `ramregion` writes each made, and `rampiece` each request's answer). A region's first
//! piece from its first page is so a piece of a shared file, never a dma-buf of its own (the
//! runtime takes a chunk at offset 0 whose `Arc` it alone holds as the buffer's own). While it
//! lives it pins every page of its region and maps
//! them in the device's GPU address space, the pages no object covers too: at most 64 MiB a
//! region any object lies in, of the guest's RAM, which `reims-ctl` makes resident from the start
//! (QEMU's `prealloc=on`), so the pins hold no page the RAM did not. Making one costs about 9 ms
//! (below, "Costs"), once while objects lie in its region.
//!
//! What regions save (**measured** 2026-11-15, Radeon 780M, RADV of Mesa 26.2.6, a 26.7.2 guest's
//! desktop walk, Finder then Launchpad three times, two walks with regions and two with a dma-buf
//! an object): at the walk's end the device held 259 to 260 imports, 19 to 21 of them region
//! dma-bufs, against 360 to 361, and Launchpad's submissions took 7.2 to 7.4 ms a present against
//! 8.6 to 9.0 (263 to 271 us a submission against 319 to 333), its presents a second (53.5 to
//! 53.6) and their take to show (a median of 12.4 to 12.7 ms) as before. A walk made 61 to 67
//! region dma-bufs, at most 31 alive at once: 1.9 GiB of the guest's 8 GiB pinned, none of it
//! memory the RAM did not already hold.
//!
//! **Kept imports.** Each region a no-copy buffer's or heap's source answered with is kept
//! imported on the device that made the object ([`keep_regions`], on the thread that made it,
//! once the runtime's call returned): the device's one import of the region's dma-buf, which the
//! object's piece made or joined, is held by the device's keep (`rt::Device::import_shared_dma_buf`)
//! besides the objects bound in it, so the region's next object joins it rather than importing the
//! dma-buf again. Without the keep, a region whose objects come and go one at a time was imported
//! again for each, and its dma-buf made again: on a Radeon 780M the import call took 820 to 907 us
//! and the next submission started on the GPU about 3 ms after it, RADV mapping the import into
//! the GPU's page tables first, against 25 to 69 us and 104 to 145 us joining a kept import
//! (**measured** by metal2vulkan's lane, its `kept_file_bench`, RUNTIME 2389). A region is in use
//! while a source that answered with it lives ([`RegionUse`]), which the runtime keeps with the
//! imports it made through it, so while an object over a piece of it lives; once its last use
//! goes, its kept imports linger [`LINGER_NS`], two seconds, then go, swept by a thread of their
//! own ([`SWEEP`]), and with them the region's dma-buf, unless an object comes first. So a device
//! holds an import of each region an object lay in within the last two seconds, at most the
//! guest's RAM over 64 MiB of them (128 for 8 GiB), each in every submission's list on RADV, and
//! the regions' pins hold no page the guest's RAM did not (`prealloc=on`, above). While one is
//! kept, the runtime's residency thread releases no import of the region's objects: the keep is a
//! hold of the file that is none of theirs, and the release would take nothing off the list. Kept
//! only on a device the process adopted, which the platform's let-go lets go of with its keeps
//! ([`take_region_keeps`]); the device's recovery carries its keeps to its new `VkDevice`, which
//! imports the dma-buf again at the first object that needs it (the runtime's `kept.rs`). Nothing
//! of it reaches the guest: an object lies at the same address over the same pages whether its
//! piece imported the dma-buf or joined a kept import.
//!
//! **The pages' size** (**measured** 2026-10-26, RADV of Mesa 26.2.4, raw Vulkan with no runtime
//! between, kernels of our own reading 64 MiB and writing 64 MiB, the median GPU time of five in
//! ms; what else the pages cost is in `super::import`):
//! - on the RX 7900 XTX, 16 scattered loads a thread within a 4 MiB window took 1.2 ms over 2 MiB
//!   pages, a memfd's huge pages through a dma-buf as anonymous ones by pointer, 3.6 ms through a
//!   dma-buf of those pages in 64 KiB pieces out of order, and 7.3 (by pointer) to 11.6 (dma-buf)
//!   over 4 KiB pages; each workgroup's 64 rows from a 4 MiB window 1.0 ms over 2 MiB pages and
//!   1.5 to 1.7 over 4 KiB ones. Over all 64 MiB the pages' size changed nothing there;
//! - on the Radeon 780M scattered loads over all 64 MiB took 8.9 ms over anonymous 2 MiB pages by
//!   pointer (9.2 in device-local memory), and 16.8 to 16.9 over 4 KiB pages and through a dma-buf
//!   of a memfd's 2 MiB pages alike; within a 4 MiB window that dma-buf took 3.3 ms, as 2 MiB
//!   pages by pointer did (3.1), against 5.3 over 4 KiB pages.
//!
//! So a dma-buf of huge pages helps the RX 7900 XTX as the pages themselves would, and the Radeon
//! 780M only within a window of a few MiB. Whether guest RAM's memfd pages are huge is the
//! host's: QEMU advises it on every RAM block (`MADV_HUGEPAGE`, `vendor/qemu/system/physmem.c`),
//! which the kernel takes for shared memory only where
//! `/sys/kernel/mm/transparent_hugepage/shmem_enabled` allows it (`advise` where these were
//! measured), and the library sets nothing of the host's.
//!
//! What udmabuf takes besides ([`Refusal`]): memfds sealed against shrinking (`F_SEAL_SHRINK`,
//! which QEMU's `seal=on`, its default, sets, and [`super::vm::memfd`] sets on the library's own),
//! at page offsets. The device node is the `kvm` group's on Ubuntu.

use std::collections::BTreeMap;
use std::fmt;
use std::io;
use std::os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};

use crate::core::num::ToU64 as _;
use crate::core::platform::os::vm::VmProt;
use crate::core::sync::Mutex;
use metal2vulkan_runtime::{self as rt, DmaBufChunk};

use super::vm;
use crate::os::files;

/// `UDMABUF_FLAGS_CLOEXEC`.
const FLAGS_CLOEXEC: u32 = 0x01;

/// `UDMABUF_CREATE_LIST`, `_IOW('u', 0x43, struct udmabuf_create_list)`: the header's 8 bytes.
const CREATE_LIST: u64 = (1 << 30) | (8 << 16) | ((b'u' as u64) << 8) | 0x43;

/// udmabuf's defaults, taken where the system shows no parameters.
const DEFAULT_LIST_LIMIT: u64 = 1024;
const DEFAULT_SIZE_LIMIT_MB: u64 = 64;

/// Why no dma-bufs were made of a range.
#[derive(Debug)]
pub(crate) enum Refusal {
    /// A byte of the range has no file behind it in the backing table.
    Unresolved,
    /// `/dev/udmabuf` would not open: absent (the module not loaded), or not the process's to
    /// open (Ubuntu gives it to the `kvm` group).
    Device(Arc<io::Error>),
    /// The range cannot be cut into chunks within udmabuf's limits at the runtime's granule: a
    /// granule holds more pieces or bytes than one dma-buf may, or the runtime binds no chunks
    /// (its granule is the whole window) and the window is past the limits.
    Limit {
        pieces: u64,
        bytes: u64,
        granule: u64,
        limits: Limits,
    },
    /// The kernel refused the call otherwise (a file it takes no pages of, unsealed or not a
    /// memfd, or no memory to pin the pages).
    Kernel(io::Error),
    /// The kernel gave no file for zeros the range takes, a pad's ([`vm::memfd`], which counts it
    /// as the host's memory running out).
    NoFile(io::Error),
    /// A view would write through a dma-buf of a file the process holds open read-only, whose
    /// pages the kernel maps writable from the file for no one ([`of_view`]).
    ReadOnlyFile,
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refusal::Unresolved => write!(f, "a byte of the range has no file behind it"),
            Refusal::Device(e) => write!(
                f,
                "/dev/udmabuf would not open ({e}): load the udmabuf module, and put the user in \
                 the group that owns the node (kvm on Ubuntu)"
            ),
            Refusal::Limit {
                pieces,
                bytes,
                granule,
                limits,
            } => {
                write!(f, "{pieces} pieces of {bytes:#x} bytes")?;
                if granule < bytes {
                    write!(f, " cut at {granule:#x}")?;
                }
                write!(
                    f,
                    " do not fit udmabuf's list_limit {} and size_limit_mb {}",
                    limits.pieces,
                    limits.bytes >> 20,
                )
            }
            Refusal::Kernel(e) => write!(f, "UDMABUF_CREATE_LIST failed: {e}"),
            Refusal::NoFile(e) => write!(f, "the kernel gave no file for zeros of the range: {e}"),
            Refusal::ReadOnlyFile => write!(f, "a writable view of a file open read-only"),
        }
    }
}

impl Refusal {
    /// Whether every range of pages that holds the one refused, asked for as one dma-buf, is
    /// refused too while the pages stay mapped as they are, as the runtime's standing refusal
    /// asks (`rt::DmaBufStandingRefusal`, which the import source answers for a texture's pages,
    /// `super::import`'s `host_memory`):
    /// - a byte with no file behind it ([`Refusal::Unresolved`]) lies in every such range;
    /// - `/dev/udmabuf` not opening ([`Refusal::Device`]) is the process's, opened once;
    /// - a range one dma-buf cannot hold ([`Refusal::Limit`] at a granule of its length, a
    ///   texture's) is as many runs of file pages and bytes in a wider range, or more: a range
    ///   only gains runs and bytes as it widens. Cut at a granule shorter than the range, a wider
    ///   range is cut elsewhere, so it may fit;
    /// - a writable piece of a file open read-only ([`Refusal::ReadOnlyFile`]) is in every such
    ///   range.
    ///
    /// The kernel's refusal ([`Refusal::Kernel`]) does not stand: it answers `EINVAL` for a list
    /// past limits lowered since they were read as for a file it takes no pages of, and `ENOMEM`
    /// for pages it could not pin now. Nor does the kernel giving no file for a pad's zeros
    /// ([`Refusal::NoFile`]), the host's memory running out now.
    pub(crate) fn holds_wider(&self) -> bool {
        match self {
            Refusal::Unresolved | Refusal::Device(_) | Refusal::ReadOnlyFile => true,
            Refusal::Limit { bytes, granule, .. } => granule >= bytes,
            Refusal::Kernel(_) | Refusal::NoFile(_) => false,
        }
    }
}

/// What one dma-buf may hold: pieces, and bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Limits {
    pub(crate) pieces: u64,
    pub(crate) bytes: u64,
}

/// udmabuf's limits as the host has them, its defaults where it shows none: read from the module's
/// parameters at most once a second ([`LIMITS`]), so they are the host's within a second of now,
/// and read again at a refusal they may have caused ([`within_limits`]). Each read of the two
/// parameters cost a texture over a no-copy buffer's pages about 9% of its making's CPU on a
/// Radeon 780M (**measured** 2026-11-15, Geekbench 7.1.0's compute pass, 606 such textures).
pub(super) fn limits() -> Limits {
    LIMITS.at(mono_now())
}

/// How long a read of udmabuf's limits stands, in nanoseconds ([`limits`]).
const LIMITS_STAND_NS: u64 = 1_000_000_000;

/// udmabuf's limits as last read from the host ([`limits`]).
static LIMITS: KeptLimits = KeptLimits::new(host_limits);

/// Limits read by `read` and kept for [`LIMITS_STAND_NS`] ([`limits`]).
struct KeptLimits {
    read: fn() -> Limits,
    /// The limits last read, and when, on the monotonic clock ([`mono_now`]).
    kept: Mutex<Option<(u64, Limits)>>,
}

impl KeptLimits {
    const fn new(read: fn() -> Limits) -> KeptLimits {
        KeptLimits {
            read,
            kept: Mutex::new(None),
        }
    }

    /// The limits read less than [`LIMITS_STAND_NS`] before `now`, else read now.
    fn at(&self, now: u64) -> Limits {
        if let Some((read, limits)) = *self.kept.lock()
            && now.saturating_sub(read) < LIMITS_STAND_NS
        {
            return limits;
        }
        self.read_now(now)
    }

    /// The limits read now, kept from `now`.
    fn read_now(&self, now: u64) -> Limits {
        let limits = (self.read)();
        *self.kept.lock() = Some((now, limits));
        limits
    }
}

/// The monotonic clock now, in nanoseconds.
fn mono_now() -> u64 {
    use crate::core::platform::os::Clock as _;
    super::clock::LinuxClock::new().now_mono().as_nanos()
}

/// `attempt` within udmabuf's limits ([`limits`]); where they refuse it ([`Refusal::Limit`]), or
/// the kernel does (it refuses a list past limits lowered since they were read), once more within
/// the limits read from the host now, where those differ from the ones it was refused at: a limit
/// raised or lowered while the process runs is taken at the first refusal it would change, as
/// when every call read the host's limits.
fn within_limits<T>(attempt: impl FnMut(Limits) -> Result<T, Refusal>) -> Result<T, Refusal> {
    within(&LIMITS, mono_now(), attempt)
}

/// [`within_limits`] at `now`, of the limits `kept`.
fn within<T>(
    kept: &KeptLimits,
    now: u64,
    mut attempt: impl FnMut(Limits) -> Result<T, Refusal>,
) -> Result<T, Refusal> {
    let limits = kept.at(now);
    let refused = match attempt(limits) {
        Err(refused @ (Refusal::Limit { .. } | Refusal::Kernel(_))) => refused,
        answered => return answered,
    };
    let host = kept.read_now(now);
    if host == limits {
        return Err(refused);
    }
    attempt(host)
}

/// udmabuf's limits as the host has them now, its defaults where it shows none.
fn host_limits() -> Limits {
    let parameter = |name: &str, default: u64| {
        let path = format!("/sys/module/udmabuf/parameters/{name}");
        (std::fs::read_to_string(path).ok())
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(default)
    };
    Limits {
        pieces: parameter("list_limit", DEFAULT_LIST_LIMIT),
        bytes: parameter("size_limit_mb", DEFAULT_SIZE_LIMIT_MB) << 20,
    }
}

/// `/dev/udmabuf`, opened once for the process.
fn device() -> Result<&'static OwnedFd, Refusal> {
    static DEVICE: OnceLock<Result<OwnedFd, Arc<io::Error>>> = OnceLock::new();
    let opened = DEVICE.get_or_init(|| {
        // SAFETY: a NUL-terminated path.
        let raw = unsafe { libc::open(c"/dev/udmabuf".as_ptr(), libc::O_RDWR | libc::O_CLOEXEC) };
        if raw < 0 {
            return Err(Arc::new(io::Error::last_os_error()));
        }
        // SAFETY: a descriptor just opened, owned from here on.
        Ok(unsafe { OwnedFd::from_raw_fd(raw) })
    });
    opened.as_ref().map_err(|e| Refusal::Device(Arc::clone(e)))
}

/// Whether `/dev/udmabuf` is the process's to open ([`device`]).
pub(super) fn opens() -> bool {
    device().is_ok()
}

/// A run of file pages: `length` bytes of `file` from `offset`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Run<F> {
    file: F,
    offset: u64,
    length: u64,
}

/// `runs`, the window's pages in order, cut into chunks: each at most `limits.pieces` runs and
/// `limits.bytes` bytes, and each but the last a multiple of `granule` long, so every cut lies a
/// multiple of `granule` from the window's start. A chunk takes as much as fits, so the fewest
/// chunks are made; a run is cut where a chunk ends inside it. `None` where a granule does not fit
/// one chunk.
fn chunks<F: Clone>(runs: &[Run<F>], granule: u64, limits: Limits) -> Option<Vec<Vec<Run<F>>>> {
    let length: u64 = runs.iter().map(|r| r.length).sum();
    let mut made = Vec::new();
    // The next byte to place, as a run and an offset into it, and its offset in the window.
    let (mut run, mut into, mut at) = (0, 0, 0);
    while at < length {
        // The farthest the limits reach from `at`.
        let room = limits.bytes.min(length - at);
        let (mut reach, mut pieces, mut r, mut i) = (0, 0, run, into);
        while reach < room && pieces < limits.pieces {
            let take = (runs[r].length - i).min(room - reach);
            reach += take;
            pieces += 1;
            (r, i) = if i + take == runs[r].length {
                (r + 1, 0)
            } else {
                (r, i + take)
            };
        }
        // Cut there, or at the granule before it unless the window ends there.
        let size = if at + reach == length {
            reach
        } else {
            reach - reach % granule
        };
        if size == 0 {
            return None;
        }
        let mut chunk = Vec::new();
        let mut left = size;
        while left > 0 {
            let take = (runs[run].length - into).min(left);
            chunk.push(Run {
                file: runs[run].file.clone(),
                offset: runs[run].offset + into,
                length: take,
            });
            left -= take;
            (run, into) = if into + take == runs[run].length {
                (run + 1, 0)
            } else {
                (run, into + take)
            };
        }
        made.push(chunk);
        at += size;
    }
    Some(made)
}

/// [`chunks`] of `runs`, padded where they are more than one: then they come to the window's
/// length rounded up to `granule`, as the runtime binds each chunk at a multiple of it, and the
/// last ends with a run `pad` makes of the bytes past the window, a file of the import's own, so a
/// GPU write past the window's end reaches nothing else. `None` where they fit no chunks, or
/// `pad` makes none.
fn padded<F: Clone>(
    runs: &mut Vec<Run<F>>,
    granule: u64,
    limits: Limits,
    pad: impl FnOnce(u64) -> Option<Run<F>>,
) -> Option<Vec<Vec<Run<F>>>> {
    let cut = chunks(runs, granule, limits)?;
    let length: u64 = runs.iter().map(|r| r.length).sum();
    let past = length.next_multiple_of(granule) - length;
    if cut.len() == 1 || past == 0 {
        return Some(cut);
    }
    runs.push(pad(past)?);
    chunks(runs, granule, limits)
}

/// The runs of file pages behind the `length` bytes from `first`, whole pages, each resolved
/// through the backing table, added to `runs` in order: one for each run of a file's pages that
/// follow on, the one before included. Refused with `runs` as they were where a byte has no file
/// behind it.
///
/// The table's stretches are joined here as they are walked ([`files::each_stretch`]), a file's
/// `Arc` cloned only for a run that starts: resolved as pieces first, each holding a clone, and
/// joined after, the runs took 42% of the CPU of a texture's making over a no-copy buffer's pages
/// on a Radeon 780M (**measured** 2026-11-15, Geekbench 7.1.0's compute pass).
fn add_runs(runs: &mut Vec<Run<Arc<OwnedFd>>>, first: u64, length: u64) -> Result<(), Refusal> {
    let page = vm::page_size();
    debug_assert!(first.is_multiple_of(page) && length.is_multiple_of(page));
    let (count, last_length) = (runs.len(), runs.last().map(|r| r.length));
    let resolved = files::each_stretch(first, length, |file, offset, bytes| {
        if let Some(last) = runs.last_mut()
            && Arc::ptr_eq(&last.file, file)
            && last.offset + last.length == offset
        {
            last.length += bytes;
            return;
        }
        runs.push(Run {
            file: Arc::clone(file),
            offset,
            length: bytes,
        });
    });
    if !resolved {
        runs.truncate(count);
        if let (Some(run), Some(length)) = (runs.last_mut(), last_length) {
            run.length = length;
        }
        return Err(Refusal::Unresolved);
    }
    Ok(())
}

/// One dma-buf of the pages of `ranges`, each a first byte and a length, whole pages in
/// ascending order and apart, which in order are those pages (as [`add_runs`] resolves them);
/// refused with udmabuf's limits where they are more than one dma-buf takes.
pub(crate) fn of_ranges(ranges: &[(u64, u64)]) -> Result<OwnedFd, Refusal> {
    let mut runs = Vec::new();
    for &(first, length) in ranges {
        add_runs(&mut runs, first, length)?;
    }
    let device = device()?;
    let bytes: u64 = ranges.iter().map(|&(_, length)| length).sum();
    within_limits(|limits| {
        if runs.len() as u64 > limits.pieces || bytes > limits.bytes {
            return Err(Refusal::Limit {
                pieces: runs.len() as u64,
                bytes,
                granule: bytes,
                limits,
            });
        }
        create(device, &runs)
    })
}

/// Whether one dma-buf takes the whole pages of the `length` bytes from `first`, as the runtime
/// asks an import's source for a texture's own pages, at a granule of their length
/// (`super::import`'s `host_memory`): refused, with udmabuf's limits named, where they are more
/// runs of file pages, or more bytes, than one takes, and where the backing table resolves a byte
/// to no file. It counts the runs as [`add_runs`] makes them, and makes nothing.
pub(crate) fn fits_one(first: u64, length: u64) -> Result<(), Refusal> {
    let page = vm::page_size();
    let start = first & !(page - 1);
    let end = first.saturating_add(length).next_multiple_of(page);
    fits_one_of(std::slice::from_ref(&(start..end)))
}

/// Whether one dma-buf takes the pages of `ranges`, host addresses of whole pages, in order: the
/// pages a texture over a view of them binds, as [`fits_one`] counts a view's, asked before the
/// view is made. Refused as [`fits_one`] is; a run of file pages that goes on from one range into
/// the next counts once, as [`add_runs`] makes it one.
pub(crate) fn fits_one_of(ranges: &[std::ops::Range<u64>]) -> Result<(), Refusal> {
    let mut pieces: u64 = 0;
    let mut bytes: u64 = 0;
    // The file and the offset past the last run's end.
    let mut last: Option<(Arc<OwnedFd>, u64)> = None;
    for range in ranges {
        let length = range.end.saturating_sub(range.start);
        let resolved = files::pieces(range.start, length).ok_or(Refusal::Unresolved)?;
        for piece in resolved {
            let follows = matches!(
                &last,
                Some((fd, end)) if Arc::ptr_eq(fd, &piece.backing.file)
                    && *end == piece.backing.offset
            );
            if !follows {
                pieces += 1;
            }
            let end = piece.backing.offset + piece.length;
            last = Some((piece.backing.file, end));
        }
        bytes = bytes.saturating_add(length);
    }
    fits_pieces(pieces, bytes)
}

/// Whether one dma-buf takes `pieces` pieces of `bytes` bytes in all, within udmabuf's limits as
/// the host has them now, asked of pages that do not exist yet: refused as [`fits_one_of`] refuses
/// pages past the limits.
pub(crate) fn fits_pieces(pieces: u64, bytes: u64) -> Result<(), Refusal> {
    within_limits(|limits| {
        if pieces > limits.pieces || bytes > limits.bytes {
            return Err(Refusal::Limit {
                pieces,
                bytes,
                granule: bytes,
                limits,
            });
        }
        Ok(())
    })
}

/// Dma-bufs of the `length` bytes from `first`, whole pages, each resolved through the backing
/// table, which in order are those bytes ([`add_runs`]), cut into chunks within udmabuf's limits
/// at multiples of `granule` from `first` (module docs).
pub(crate) fn of_pages(first: u64, length: u64, granule: u64) -> Result<Vec<DmaBufChunk>, Refusal> {
    of_pages_counted(first, length, granule).map(|(chunks, _)| chunks)
}

/// [`of_pages`], and the runs of file pages the chunks hold, the pad's among them: what the
/// import source's trace event writes (`super::import`'s `host_memory`).
pub(crate) fn of_pages_counted(
    first: u64,
    length: u64,
    granule: u64,
) -> Result<(Vec<DmaBufChunk>, u64), Refusal> {
    let page = vm::page_size();
    let mut runs: Vec<Run<Arc<OwnedFd>>> = Vec::new();
    add_runs(&mut runs, first, length)?;
    let device = device()?;
    let granule = granule.max(page);
    let resolved = runs.len();
    within_limits(|limits| {
        // A pad an attempt before this one added goes.
        runs.truncate(resolved);
        // Runs one dma-buf takes are one chunk of them all, as `chunks` cuts them, made of the
        // runs themselves: a texture's pages, at a granule of their length, always are.
        if !runs.is_empty() && resolved.to_u64() <= limits.pieces && length <= limits.bytes {
            let chunk = DmaBufChunk {
                file: Arc::new(create(device, &runs)?),
                offset: 0,
                length,
            };
            return Ok((vec![chunk], resolved.to_u64()));
        }
        let refused = |runs: &[Run<Arc<OwnedFd>>]| Refusal::Limit {
            pieces: runs.len() as u64,
            bytes: length,
            granule,
            limits,
        };
        // The kernel's refusal of the pad's file, where it gave none ([`vm::memfd`]).
        let no_file = std::cell::Cell::new(None);
        let pad = |pad| match vm::memfd(pad) {
            Ok(file) => Some(Run {
                file,
                offset: 0,
                length: pad,
            }),
            Err(e) => {
                no_file.set(Some(e));
                None
            }
        };
        let cut = padded(&mut runs, granule, limits, pad).ok_or_else(|| {
            no_file
                .take()
                .map_or_else(|| refused(&runs), Refusal::NoFile)
        })?;
        let made = cut.iter().map(|chunk| {
            let fd = create(device, chunk)?;
            Ok(DmaBufChunk {
                file: Arc::new(fd),
                offset: 0,
                length: chunk.iter().map(|r| r.length).sum(),
            })
        });
        Ok((made.collect::<Result<_, _>>()?, runs.len().to_u64()))
    })
}

/// A group dma-buf ([`of_group`]): several members' pages back to back in one file, and where
/// each member starts in it.
#[derive(Debug)]
pub(crate) struct Group {
    /// The file every member's piece is of.
    pub(crate) file: Arc<OwnedFd>,
    /// The first byte of each member taken, in the order given, each a multiple of the group's
    /// alignment: `offsets.len()` members were taken, the first that fit.
    pub(crate) offsets: Vec<u64>,
    /// The file's bytes.
    pub(crate) length: u64,
}

/// One dma-buf of the pages of the first of `members` that fit in it, each member a first byte
/// and a length (whole pages, resolved through the backing table as [`add_runs`] resolves them),
/// in order: each member starts at a multiple of `align` (a multiple of the host page; the host
/// page where less), and the bytes between two members are a file of zeros of the group's own,
/// so the file holds no page of other memory. A group takes members while they fit `limits`
/// (within udmabuf's as the host has them, [`limits`]), counting each member's runs apart (two
/// members whose pages follow on in a file are two runs), and stops at the first member that
/// does not fit or whose
/// pages resolve to no file, leaving it and those after it to another group. Refused where the
/// first member alone does not fit, where its pages resolve to no file, where `/dev/udmabuf` does
/// not open, or where the kernel refuses the list. A gap whose file of zeros the kernel does not
/// give ends the group before the member after it.
pub(crate) fn of_group(
    members: &[(u64, u64)],
    align: u64,
    limits: Limits,
) -> Result<Group, Refusal> {
    let page = vm::page_size();
    let align = align.max(page);
    debug_assert!(align.is_multiple_of(page));
    let device = device()?;
    let mut runs: Vec<Run<Arc<OwnedFd>>> = Vec::new();
    let mut offsets = Vec::new();
    let mut length: u64 = 0;
    // The file of zeros the gaps between members take, made at the first gap: a gap is less than
    // `align`.
    let mut gaps: Option<Arc<OwnedFd>> = None;
    for &(first, bytes) in members {
        let start = length.next_multiple_of(align);
        let mut member = Vec::new();
        if start > length {
            let file = match &gaps {
                Some(file) => Arc::clone(file),
                // The first member starts the file, at no gap: where the kernel gives no file for
                // a later one's, the group ends before that member.
                None => match vm::memfd(align - page) {
                    Ok(file) => Arc::clone(gaps.insert(file)),
                    Err(_) => break,
                },
            };
            member.push(Run {
                file,
                offset: 0,
                length: start - length,
            });
        }
        if let Err(refusal) = add_runs(&mut member, first, bytes) {
            if offsets.is_empty() {
                return Err(refusal);
            }
            break;
        }
        let pieces = (runs.len() + member.len()) as u64;
        let end = start.saturating_add(bytes);
        if pieces > limits.pieces || end > limits.bytes {
            if offsets.is_empty() {
                return Err(Refusal::Limit {
                    pieces,
                    bytes: end,
                    granule: end,
                    limits,
                });
            }
            break;
        }
        runs.append(&mut member);
        offsets.push(start);
        length = end;
    }
    let file = Arc::new(create(device, &runs)?);
    Ok(Group {
        file,
        offsets,
        length,
    })
}

/// The views [`group_views`] put in group dma-bufs whose membership lives ([`Membership`]), by
/// first byte: what the import source answers a request over their pages with
/// ([`piece_of`]).
static GROUPED: Mutex<BTreeMap<u64, Grouped>> = Mutex::new(BTreeMap::new());

/// A view in [`GROUPED`].
#[derive(Debug)]
struct Grouped {
    /// The view's bytes.
    length: u64,
    /// Its group's file: the runtime's imports of it, and the caller of [`group_views`] until it
    /// has made each member's import, hold it.
    file: Weak<OwnedFd>,
    /// Where the view's pages start in the file.
    offset: u64,
    /// Its membership's number, so a membership drops its own entry alone.
    number: u64,
}

/// The largest alignment a request of the runtime's has asked a shared piece's offset to be a
/// multiple of ([`asked_alignment`]), which each group's members start at, at least the host page.
static ALIGNMENT: AtomicU64 = AtomicU64::new(0);

/// A request of the runtime's asked for pieces at offsets that are multiples of `alignment`: the
/// groups made from now on start their members at a multiple of it, at least.
pub(crate) fn asked_alignment(alignment: u64) {
    ALIGNMENT.fetch_max(alignment, Ordering::Relaxed);
}

/// A view's place in its group dma-buf ([`GROUPED`]), until it drops: the view's owner drops it
/// before the view is unmapped, so a view mapped later at the same address is never answered with
/// the old view's pages.
#[derive(Debug)]
pub(crate) struct Membership {
    first: u64,
    number: u64,
}

impl Drop for Membership {
    fn drop(&mut self) {
        let mut grouped = GROUPED.lock();
        if grouped
            .get(&self.first)
            .is_some_and(|g| g.number == self.number)
        {
            grouped.remove(&self.first);
        }
    }
}

/// The most bytes one group dma-buf holds, whatever udmabuf takes: a member whose buffers went
/// keeps its pages pinned and mapped in the device's import while another member's import lives,
/// so a group's bytes bound what such members hold (`super::dma_buf`'s module docs, "Groups").
/// udmabuf's default `size_limit_mb`, which CachyOS's 7.3 kernels raise to 2147483647 as they boot
/// (`super::host_ram`'s module docs), where `list_limit` alone would let a group of large views
/// grow to gigabytes.
pub(crate) const GROUP_BYTES: u64 = 64 << 20;

/// What one group dma-buf holds at most: udmabuf's limits as the host has them ([`limits`]), its
/// bytes at most [`GROUP_BYTES`].
pub(crate) fn group_limits() -> Limits {
    group_limits_of(limits())
}

/// What one group dma-buf holds at most within udmabuf's `limits`: its bytes at most
/// [`GROUP_BYTES`].
fn group_limits_of(limits: Limits) -> Limits {
    Limits {
        pieces: limits.pieces,
        bytes: limits.bytes.min(GROUP_BYTES),
    }
}

/// What [`group_views`] made: the group dma-bufs, which the caller holds until it has made each
/// member's import, and each view's membership, `None` for a view left alone.
pub(crate) type Groups = (Vec<Arc<OwnedFd>>, Vec<Option<Membership>>);

/// Puts `views`, each a first byte and a length (whole pages of the library's views, mapped),
/// into group dma-bufs ([`of_group`]), in order, each as many as fit, the members starting at the
/// alignment the runtime's requests have asked for ([`asked_alignment`]), and registers each
/// member's place ([`GROUPED`]). A view a group refuses as its first member (its pages past the
/// limits alone, or resolving to no file) is left alone and the next starts a group, and a group
/// that would take only its first member is not made: its import is made as before, a dma-buf
/// of its own.
pub(crate) fn group_views(views: &[(u64, u64)]) -> Groups {
    group_views_at(views, ALIGNMENT.load(Ordering::Relaxed))
}

/// [`group_views`], the members starting at multiples of `align`, at least the host page.
fn group_views_at(views: &[(u64, u64)], align: u64) -> Groups {
    static NUMBER: AtomicU64 = AtomicU64::new(0);
    let mut files = Vec::new();
    let mut members: Vec<Option<Membership>> = views.iter().map(|_| None).collect();
    let align = align.max(vm::page_size());
    let mut at = 0;
    while views.len() - at > 1 {
        let made = within_limits(|limits| of_group(&views[at..], align, group_limits_of(limits)));
        let group = match made {
            Ok(group) => group,
            Err(refusal) => {
                log::debug!(
                    "a view of {:#x} bytes at {:#x} starts no group dma-buf: {refusal}",
                    views[at].1,
                    views[at].0
                );
                at += 1;
                continue;
            }
        };
        let taken = group.offsets.len();
        if taken > 1 {
            let mut grouped = GROUPED.lock();
            for (i, &offset) in group.offsets.iter().enumerate() {
                let (first, length) = views[at + i];
                let number = NUMBER.fetch_add(1, Ordering::Relaxed);
                let file = Arc::downgrade(&group.file);
                grouped.insert(
                    first,
                    Grouped {
                        length,
                        file,
                        offset,
                        number,
                    },
                );
                members[at + i] = Some(Membership { first, number });
            }
            drop(grouped);
            log::debug!(
                "{taken} views in a group dma-buf of {:#x} bytes, from the view at {:#x}",
                group.length,
                views[at].0
            );
            files.push(group.file);
        }
        at += taken;
    }
    (files, members)
}

/// The piece of a group dma-buf that holds the `length` bytes from `first` ([`GROUPED`]): its
/// file and the bytes' offset in it, where they lie in one view whose membership lives, its file
/// held still, and that offset is a multiple of `alignment`.
pub(crate) fn piece_of(first: u64, length: u64, alignment: u64) -> Option<(Arc<OwnedFd>, u64)> {
    let grouped = GROUPED.lock();
    let (&base, view) = grouped.range(..=first).next_back()?;
    let end = first.checked_add(length)?;
    if end > base.saturating_add(view.length) {
        return None;
    }
    let offset = view.offset + (first - base);
    if !offset.is_multiple_of(alignment.max(1)) {
        return None;
    }
    Some((view.file.upgrade()?, offset))
}

/// The bytes of a region of a guest RAM file (module docs, "Regions"), where udmabuf takes them in
/// one dma-buf: udmabuf's default `size_limit_mb`, and [`GROUP_BYTES`], which bounds what a
/// shared file pins that no object covers, whatever a host's udmabuf takes.
pub(crate) const REGION_BYTES: u64 = 64 << 20;

/// The bytes of a region on this host: [`REGION_BYTES`], or udmabuf's `size_limit_mb` where the
/// host has it lower, down to a power of two, so a region of a file lies at a multiple of every
/// alignment up to its own; `None` where that is less than a page.
fn region_bytes() -> Option<u64> {
    let bytes = 1u64 << REGION_BYTES.min(limits().bytes).checked_ilog2()?;
    (bytes >= vm::page_size()).then_some(bytes)
}

/// A region of a guest RAM file (module docs, "Regions"): the file's identity (its `Arc`'s
/// address), the regions' bytes and the region's index in the file.
type RegionKey = (usize, u64, u64);

/// The region dma-bufs made and alive now, by [`RegionKey`] (module docs, "Regions").
static REGIONS: Mutex<BTreeMap<RegionKey, Region>> = Mutex::new(BTreeMap::new());

/// A region's dma-buf in [`REGIONS`].
#[derive(Debug)]
struct Region {
    /// The RAM's file, held weakly: its `Arc`'s allocation, whose address keys the entry, is not
    /// let go of while the entry lives, so no other file is ever found under the key.
    _ram: Weak<OwnedFd>,
    /// The region's dma-buf: the runtime's imports of it, the sources that answered with it
    /// ([`RegionUse`]) and its kept imports ([`Region::kept`]) hold it.
    file: Weak<OwnedFd>,
    /// How many [`RegionUse`]s of it live: the sources that answered with a piece of it, which the
    /// runtime keeps while an object it made through one lives.
    users: u64,
    /// When the last of them went, on the monotonic clock, where an import of it is kept: its
    /// kept imports go [`LINGER_NS`] after (module docs, "Kept imports").
    idle_since: Option<u64>,
    /// Its import kept on each device an object over a piece of it was made on, by the device's
    /// registry ID (`rt::Device::import_shared_dma_buf`, module docs, "Kept imports").
    kept: Vec<(u64, rt::SharedDmaBufImport)>,
    /// The devices, by registry ID, that refused to keep its import: none is asked again.
    refused: Vec<u64>,
}

/// How many region dma-bufs were made (module docs, "Regions"): a diagnostic, which the trace
/// writes as each is made.
static REGIONS_MADE: AtomicU64 = AtomicU64::new(0);

/// A source's use of a region's dma-buf ([`ram_pieces`]): the dma-buf held, and the region counted
/// in use ([`Region::users`]) while it lives. The import source keeps one for each region it
/// answered with (`super::import`'s `host_memory`), and the runtime keeps the source while an
/// import made through it lives, so a region is in use while an object over a piece of it lives.
#[derive(Debug)]
pub(crate) struct RegionUse {
    key: RegionKey,
    file: Arc<OwnedFd>,
}

impl RegionUse {
    /// Whether `self` and `other` are uses of one region's dma-buf.
    pub(crate) fn same(&self, other: &RegionUse) -> bool {
        Arc::ptr_eq(&self.file, &other.file)
    }

    /// The region, weakly: what [`keep_regions`] keeps on the device an object over a piece of
    /// it was made on.
    pub(crate) fn region(&self) -> RegionRef {
        RegionRef {
            key: self.key,
            file: Arc::downgrade(&self.file),
        }
    }
}

impl Drop for RegionUse {
    /// The region's last use going starts its kept imports' linger (module docs, "Kept imports").
    fn drop(&mut self) {
        let mut regions = REGIONS.lock();
        let Some(region) = (regions.get_mut(&self.key))
            .filter(|r| std::ptr::eq(r.file.as_ptr(), Arc::as_ptr(&self.file)))
        else {
            return;
        };
        region.users = region.users.saturating_sub(1);
        let lingers = region.users == 0 && !region.kept.is_empty();
        if lingers {
            region.idle_since = Some(mono_now());
        }
        drop(regions);
        if lingers {
            SWEEP.wake();
        }
    }
}

/// A region a source answered with, weakly ([`RegionUse::region`]).
#[derive(Clone, Debug)]
pub(crate) struct RegionRef {
    key: RegionKey,
    file: Weak<OwnedFd>,
}

/// What [`ram_pieces`] answers: the pieces, in order, and a use of each region they lie in.
#[derive(Debug)]
pub(crate) struct RegionPieces {
    pub(crate) chunks: Vec<DmaBufChunk>,
    pub(crate) uses: Vec<RegionUse>,
}

/// Pieces of the guest RAM's region dma-bufs that hold the `length` bytes from `first` (whole
/// pages) in order, as a no-copy buffer's or heap's request at `granule` and `alignment` takes
/// them (module docs, "Regions"): each piece one run of a file's pages inside one region, at its
/// offset in the region's dma-buf, made at the region's first piece, with a use of each region
/// ([`RegionUse`]). `None` where a byte shows no page of the guest's RAM mapped writable, where the
/// pieces break the request's rules (a piece at an offset its alignment does not divide; several,
/// where a boundary between two lies off the granule, or the length is not whole granules, which
/// would take a pad of other pages), or where udmabuf makes no region's dma-buf: the request is
/// answered as before then.
pub(crate) fn ram_pieces(
    first: u64,
    length: u64,
    granule: u64,
    alignment: u64,
) -> Option<RegionPieces> {
    ram_pieces_in(first, length, granule, alignment, region_bytes()?)
}

/// Why [`ram_pieces`] answered none.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NoPieces {
    /// A byte shows no page of the guest's RAM mapped writable.
    NotRam,
    /// The pieces break the request's rules.
    Rules,
    /// udmabuf made no region's dma-buf.
    Made,
}

impl NoPieces {
    /// Its number in the trace's `rampiece` event.
    fn code(self) -> u64 {
        match self {
            NoPieces::NotRam => 1,
            NoPieces::Rules => 2,
            NoPieces::Made => 3,
        }
    }
}

/// [`ram_pieces`] in regions of `region` bytes, a power of two of whole pages.
fn ram_pieces_in(
    first: u64,
    length: u64,
    granule: u64,
    alignment: u64,
    region: u64,
) -> Option<RegionPieces> {
    let answer = cut_in_regions(first, length, granule, alignment, region);
    // The trace's `rampiece`: the request's first byte, its length, the pieces answered, and
    // why none were where none were.
    let (count, why) = match &answer {
        Ok(pieces) => (pieces.chunks.len().to_u64(), 0),
        Err(no) => (0, no.code()),
    };
    crate::core::trace::event("rampiece", &[first, length, count, why]);
    answer.ok()
}

/// The pieces [`ram_pieces_in`] answers, or why there are none.
fn cut_in_regions(
    first: u64,
    length: u64,
    granule: u64,
    alignment: u64,
    region: u64,
) -> Result<RegionPieces, NoPieces> {
    debug_assert!(region.is_power_of_two() && region.is_multiple_of(vm::page_size()));
    let pieces = files::guest_ram_pieces(first, length).ok_or(NoPieces::NotRam)?;
    // Each piece cut where its pages cross into another region: (the RAM's file, the region's
    // index, the offset in the region, the bytes).
    let mut cut: Vec<(Arc<OwnedFd>, u64, u64, u64)> = Vec::new();
    for piece in &pieces {
        let (mut offset, mut left) = (piece.backing.offset, piece.length);
        while left > 0 {
            let into = offset % region;
            let take = left.min(region - into);
            cut.push((Arc::clone(&piece.backing.file), offset / region, into, take));
            offset += take;
            left -= take;
        }
    }
    let shape: Vec<(u64, u64)> = cut.iter().map(|&(_, _, into, n)| (into, n)).collect();
    if !takes(&shape, length, granule, alignment) {
        return Err(NoPieces::Rules);
    }
    let device = device().map_err(|_| NoPieces::Made)?;
    let mut uses: Vec<RegionUse> = Vec::new();
    let mut chunks = Vec::with_capacity(cut.len());
    for (ram, index, into, bytes) in cut {
        let key = (Arc::as_ptr(&ram).addr(), region, index);
        let file = match uses.iter().find(|u| u.key == key) {
            Some(used) => Arc::clone(&used.file),
            None => {
                let used = region_use(device, &ram, index, region).ok_or(NoPieces::Made)?;
                let file = Arc::clone(&used.file);
                uses.push(used);
                file
            }
        };
        chunks.push(DmaBufChunk {
            file,
            offset: into,
            length: bytes,
        });
    }
    Ok(RegionPieces { chunks, uses })
}

/// Whether pieces of `shape`, each an offset in its file and a length, in order, answer a request
/// of `length` bytes at `granule` and `alignment` as the runtime's contract allows
/// (`metal2vulkan_runtime::DmaBufSource`): each piece at a multiple of `alignment`; one piece the
/// request's bytes; several, every boundary between two a multiple of `granule` from the start,
/// and the bytes whole granules, as the runtime binds several in whole granules and a pad would
/// be a region's other pages.
fn takes(shape: &[(u64, u64)], length: u64, granule: u64, alignment: u64) -> bool {
    let (granule, alignment) = (granule.max(1), alignment.max(1));
    if shape.is_empty() || !shape.iter().all(|&(at, _)| at.is_multiple_of(alignment)) {
        return false;
    }
    let total: u64 = shape.iter().map(|&(_, n)| n).sum();
    if total != length {
        return false;
    }
    if shape.len() == 1 {
        return true;
    }
    let mut end = 0u64;
    for &(_, n) in &shape[..shape.len() - 1] {
        end += n;
        if !end.is_multiple_of(granule) {
            return false;
        }
    }
    length.is_multiple_of(granule)
}

/// A use of region `index` of the RAM's file `ram`, `region` bytes from `index * region` (fewer at
/// the file's end): of its dma-buf alive in [`REGIONS`], or of one made now and entered there. A
/// region made meanwhile by another thread is the one used, and this one goes. `None` where the
/// file's size is not known or the kernel refuses the dma-buf.
fn region_use(device: &OwnedFd, ram: &Arc<OwnedFd>, index: u64, region: u64) -> Option<RegionUse> {
    let key = (Arc::as_ptr(ram).addr(), region, index);
    // A use of the region's live dma-buf, counted, its linger ended.
    let used = |regions: &mut BTreeMap<RegionKey, Region>| {
        let entry = regions.get_mut(&key)?;
        let file = entry.file.upgrade()?;
        entry.users += 1;
        entry.idle_since = None;
        Some(RegionUse { key, file })
    };
    if let Some(used) = used(&mut REGIONS.lock()) {
        return Some(used);
    }
    let start = index.checked_mul(region)?;
    let stat = stat_of(ram)?;
    let bytes = region.min(u64::try_from(stat.st_size).ok()?.checked_sub(start)?);
    let run = Run {
        file: Arc::clone(ram),
        offset: start,
        length: bytes,
    };
    let made = match create(device, std::slice::from_ref(&run)) {
        Ok(fd) => Arc::new(fd),
        Err(refusal) => {
            log::debug!(
                "no region dma-buf of the guest's RAM, {bytes:#x} bytes at {start:#x} of its \
                 file: {refusal}"
            );
            return None;
        }
    };
    let mut regions = REGIONS.lock();
    if let Some(used) = used(&mut regions) {
        return Some(used);
    }
    // An entry whose dma-buf went holds no kept import: each holds the dma-buf.
    regions.retain(|_, r| r.file.strong_count() > 0);
    regions.insert(
        key,
        Region {
            _ram: Arc::downgrade(ram),
            file: Arc::downgrade(&made),
            users: 1,
            idle_since: None,
            kept: Vec::new(),
            refused: Vec::new(),
        },
    );
    let live = regions.len().to_u64();
    drop(regions);
    // The trace's `ramregion`: the region's first byte in the file, its bytes, the regions made
    // so far, those alive now, the bytes of pages the file held before and after the dma-buf was
    // made (`st_blocks`): what it gave pages to that nothing had touched, the guest's writes
    // meanwhile among them.
    let made_so_far = REGIONS_MADE.fetch_add(1, Ordering::Relaxed) + 1;
    if crate::core::trace::on() {
        let held = |stat: &libc::stat| u64::try_from(stat.st_blocks).unwrap_or(0) * 512;
        let after = stat_of(ram).map_or(0, |s| held(&s));
        let event = [start, bytes, made_so_far, live, held(&stat), after];
        crate::core::trace::event("ramregion", &event);
    }
    Some(RegionUse { key, file: made })
}

/// How long a region's kept imports stay once no object lies in it, in nanoseconds (module docs,
/// "Kept imports"): two seconds, the first window the runtime's residency thread keeps an idle
/// import of a no-copy buffer for before it releases it (`host_residency.rs`'s `IDLE`).
pub(crate) const LINGER_NS: u64 = 2_000_000_000;

/// Keeps the import of each region of `answered`, those a source answered with as an object was
/// made on `device`, on that device (module docs, "Kept imports"): its one import of the region's
/// dma-buf, which the object's piece joined, held from now on by the device's keep
/// (`rt::Device::import_shared_dma_buf`), so the region's next object joins it rather than
/// importing the dma-buf again. Only on a device the process adopted, which the platform's let-go
/// lets go of ([`take_region_keeps`]); only for a region in use, as one whose source the runtime
/// dropped had no import made through it; once per device; and never on a device that refused
/// it once. Called on the thread that made the object, once the runtime's call returned.
pub(crate) fn keep_regions(device: &rt::Device, answered: Vec<RegionRef>) {
    if answered.is_empty() || !crate::gpu::device::adopted(device) {
        return;
    }
    let id = device.registry_id();
    for region in answered {
        // The region's dma-buf, where it is in use and this device keeps none of it yet.
        let file = {
            let regions = REGIONS.lock();
            let Some(entry) =
                (regions.get(&region.key)).filter(|r| Weak::ptr_eq(&r.file, &region.file))
            else {
                continue;
            };
            let asked = entry.kept.iter().any(|&(at, _)| at == id) || entry.refused.contains(&id);
            if asked || entry.users == 0 {
                continue;
            }
            let Some(file) = region.file.upgrade() else {
                continue;
            };
            file
        };
        let made = device.import_shared_dma_buf(&file, rt::ResourceOptions::default());
        let mut regions = REGIONS.lock();
        let entry = (regions.get_mut(&region.key)).filter(|r| Weak::ptr_eq(&r.file, &region.file));
        // Whether it lingers already: its last use went while the keep was made.
        let mut lingers = false;
        let left = match (made, entry) {
            (Ok(kept), Some(entry)) if !entry.kept.iter().any(|&(at, _)| at == id) => {
                entry.kept.push((id, kept));
                if entry.users == 0 {
                    entry.idle_since = Some(mono_now());
                    lingers = true;
                }
                None
            }
            (Ok(kept), _) => Some(kept),
            (Err(error), entry) => {
                if let Some(entry) = entry {
                    entry.refused.push(id);
                }
                log::debug!(
                    target: "reims_vgpu_vulkan::os",
                    "the device keeps no import of a region dma-buf of the guest's RAM: {error}"
                );
                None
            }
        };
        drop(regions);
        // A keep made meanwhile by another thread is the one kept; this one goes, with no lock.
        drop(left);
        drop(file);
        if lingers {
            SWEEP.wake();
        }
    }
}

/// Every region import the devices keep, taken out of [`REGIONS`], for the platform's let-go of
/// its devices (`crate::let_go`): each holds a handle of its device, which goes with the keep, and
/// the imports go with their last buffer, or at once where none is bound in them.
pub(crate) fn take_region_keeps() -> Vec<rt::SharedDmaBufImport> {
    let mut regions = REGIONS.lock();
    let mut taken = Vec::new();
    for region in regions.values_mut() {
        taken.extend(region.kept.drain(..).map(|(_, kept)| kept));
        region.idle_since = None;
    }
    taken
}

/// The kept imports of the regions whose linger ended by `now` ([`LINGER_NS`]), taken out of
/// [`REGIONS`], and when the next linger ends, where one goes on.
fn lingered(now: u64) -> (Vec<rt::SharedDmaBufImport>, Option<u64>) {
    let mut regions = REGIONS.lock();
    let mut taken = Vec::new();
    let mut next: Option<u64> = None;
    for region in regions.values_mut() {
        let Some(since) = region.idle_since.filter(|_| region.users == 0) else {
            continue;
        };
        let ends = since.saturating_add(LINGER_NS);
        if ends <= now {
            taken.extend(region.kept.drain(..).map(|(_, kept)| kept));
            region.idle_since = None;
        } else {
            next = Some(next.map_or(ends, |n| n.min(ends)));
        }
    }
    (taken, next)
}

/// Ends every region's linger now, as [`LINGER_NS`] after its last use would, its kept imports let
/// go of: for tests, which wait no two seconds.
#[cfg(test)]
pub(crate) fn end_lingers() {
    drop(lingered(u64::MAX).0);
}

/// The sweep of the regions' kept imports whose linger ended ([`lingered`]), on a thread of its
/// own, `reims-vgpu.region-keeps`, started with the first keep: it waits until the next linger
/// ends, or with none for a keep or a region's last use, takes what lingered out and lets go of it
/// with no lock held. It holds nothing of a device but what [`REGIONS`] holds.
static SWEEP: Sweep = Sweep::new();

/// [`SWEEP`]'s state.
struct Sweep {
    /// Set as a keep or a last use wakes the thread, cleared as it sweeps.
    woken: Mutex<bool>,
    /// Raised as it is woken.
    wake: crate::core::sync::Condvar,
    /// Whether the thread was started.
    started: OnceLock<()>,
}

impl Sweep {
    const fn new() -> Sweep {
        Sweep {
            woken: Mutex::new(false),
            wake: crate::core::sync::Condvar::new(),
            started: OnceLock::new(),
        }
    }

    /// Wakes the thread, starting it first where it was not.
    fn wake(&'static self) {
        self.started.get_or_init(|| {
            let spawned = std::thread::Builder::new()
                .name("reims-vgpu.region-keeps".into())
                .spawn(move || self.run());
            if let Err(error) = spawned {
                log::warn!(
                    target: "reims_vgpu_vulkan::os",
                    "no thread lets go of the guest RAM's region imports kept idle: {error}"
                );
            }
        });
        *self.woken.lock() = true;
        self.wake.notify_one();
    }

    /// The thread's loop ([`SWEEP`]).
    fn run(&self) {
        use crate::core::platform::os::Clock as _;
        use crate::core::platform::os::clock::{Deadline, MonoTime};
        let clock = super::clock::LinuxClock::new();
        loop {
            let (gone, next) = lingered(clock.now_mono().as_nanos());
            drop(gone);
            let mut woken = self.woken.lock();
            while !*woken {
                match next {
                    Some(ends) => {
                        let deadline = Deadline::Mono(MonoTime::from_nanos(ends));
                        if clock
                            .wait_until(&self.wake, &mut woken, deadline)
                            .timed_out()
                        {
                            break;
                        }
                    }
                    None => clock.wait(&self.wake, &mut woken),
                }
            }
            *woken = false;
        }
    }
}

/// The status of the file `fd`.
fn stat_of(fd: &OwnedFd) -> Option<libc::stat> {
    let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: the descriptor is open for the call, and `fstat` writes the structure.
    if unsafe { libc::fstat(fd.as_raw_fd(), stat.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: the call succeeded, so it wrote the structure.
    Some(unsafe { stat.assume_init() })
}

/// One dma-buf of a view's pages ([`of_view`]): `length` bytes of the view from `at`, to map with
/// protection `prot`.
#[derive(Debug)]
pub(crate) struct ViewBuf {
    pub(crate) fd: OwnedFd,
    pub(crate) at: u64,
    pub(crate) length: u64,
    pub(crate) prot: VmProt,
}

/// Dma-bufs of the pages `pieces` show, in order, which mapped one after another are those pages
/// (`super::vm`'s module docs, "Views through dma-bufs"): each piece's file pages, cut into
/// chunks within udmabuf's limits and where the protection changes, the fewest that fit. Refused
/// where `/dev/udmabuf` would not open or the kernel refuses a piece's file (one not a memfd, or
/// not sealed against shrinking), with nothing made; and where a writable piece's file is open
/// read-only, which udmabuf does not ask, as a map of the file shared and writable is refused
/// (`EACCES`), so the view is no way past that.
pub(crate) fn of_view(pieces: &[files::Piece]) -> Result<Vec<ViewBuf>, Refusal> {
    let device = device()?;
    let mut writable: Vec<&Arc<OwnedFd>> = Vec::new();
    for piece in pieces
        .iter()
        .filter(|p| p.prot.raw() & libc::PROT_WRITE != 0)
    {
        let fd = &piece.backing.file;
        if writable.iter().any(|w| Arc::ptr_eq(w, fd)) {
            continue;
        }
        // SAFETY: a query of an open descriptor's flags.
        let flags = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GETFL) };
        if flags < 0 || flags & libc::O_ACCMODE != libc::O_RDWR {
            return Err(Refusal::ReadOnlyFile);
        }
        writable.push(fd);
    }
    let page = vm::page_size();
    within_limits(|limits| {
        let mut made = Vec::new();
        let mut from = 0;
        while from < pieces.len() {
            // The pieces from `from` of one protection.
            let prot = pieces[from].prot;
            let to = (from..pieces.len())
                .find(|&i| pieces[i].prot != prot)
                .unwrap_or(pieces.len());
            let runs: Vec<Run<Arc<OwnedFd>>> = pieces[from..to]
                .iter()
                .map(|p| Run {
                    file: Arc::clone(&p.backing.file),
                    offset: p.backing.offset,
                    length: p.length,
                })
                .collect();
            let mut at = pieces[from].at;
            let cut = chunks(&runs, page, limits).ok_or(Refusal::Limit {
                pieces: runs.len() as u64,
                bytes: runs.iter().map(|r| r.length).sum(),
                granule: page,
                limits,
            })?;
            for chunk in cut {
                let length = chunk.iter().map(|r| r.length).sum();
                made.push(ViewBuf {
                    fd: create(device, &chunk)?,
                    at,
                    length,
                    prot,
                });
                at += length;
            }
            from = to;
        }
        Ok(made)
    })
}

/// One dma-buf of `runs`, in order (`UDMABUF_CREATE_LIST`).
fn create(device: &OwnedFd, runs: &[Run<Arc<OwnedFd>>]) -> Result<OwnedFd, Refusal> {
    // `struct udmabuf_create_list` of Linux's `include/uapi/linux/udmabuf.h` as little-endian
    // words: `flags` and `count`, then each `struct udmabuf_create_item`'s `memfd` and its
    // padding, `offset` and `size`.
    let count = u32::try_from(runs.len()).unwrap_or(u32::MAX);
    let mut call = Vec::with_capacity(1 + 3 * runs.len());
    call.push(u64::from(FLAGS_CLOEXEC) | (u64::from(count) << 32));
    for run in runs {
        let memfd = u32::try_from(run.file.as_raw_fd()).unwrap_or(u32::MAX);
        call.extend([u64::from(memfd), run.offset, run.length]);
    }
    // SAFETY: the list describes open memfds the backing table holds through the call; the kernel reads the header and its items.
    let made = unsafe { libc::ioctl(device.as_raw_fd(), CREATE_LIST as _, call.as_ptr()) };
    if made < 0 {
        return Err(Refusal::Kernel(io::Error::last_os_error()));
    }
    #[cfg(test)]
    MADE.fetch_add(1, Ordering::Relaxed);
    // SAFETY: the dma-buf the call just made, owned from here on.
    Ok(unsafe { OwnedFd::from_raw_fd(made) })
}

/// How many dma-bufs the process made ([`create`]), for tests that count them.
#[cfg(test)]
pub(super) static MADE: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
mod tests {
    use crate::core::num::ToUsize as _;
    use crate::core::platform::os::vm::{HostAddr, HostVm as _};

    use super::*;

    /// Whether this host has a udmabuf device this process opens: where it has none, the tests
    /// say so and check nothing.
    fn udmabuf() -> bool {
        match device() {
            Ok(_) => true,
            Err(refusal) => {
                eprintln!("skipped: {refusal}");
                false
            }
        }
    }

    /// The dma-buf's byte at `offset`, read through a mapping of it.
    fn read_through(chunk: &DmaBufChunk, offset: u64) -> u8 {
        let length = usize::try_from(chunk.length).unwrap();
        // SAFETY: a fresh read-only mapping of the dma-buf, read once and unmapped.
        unsafe {
            let at = libc::mmap(
                std::ptr::null_mut(),
                length,
                libc::PROT_READ,
                libc::MAP_SHARED,
                chunk.file.as_raw_fd(),
                0,
            );
            assert_ne!(at, libc::MAP_FAILED);
            let byte = (at.cast::<u8>())
                .add(usize::try_from(offset).unwrap())
                .read_volatile();
            libc::munmap(at, length);
            byte
        }
    }

    /// The byte at `offset` of the dma-buf `file`, read through a mapping of it from its start.
    fn read_file(file: &OwnedFd, offset: u64) -> u8 {
        let length = (offset + 1).next_multiple_of(vm::page_size()).to_usize();
        // SAFETY: a fresh read-only mapping of the dma-buf, read once and unmapped.
        unsafe {
            let at = libc::mmap(
                std::ptr::null_mut(),
                length,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            );
            assert_ne!(at, libc::MAP_FAILED);
            let byte = (at.cast::<u8>()).add(offset.to_usize()).read_volatile();
            libc::munmap(at, length);
            byte
        }
    }

    /// Writes `byte` at `offset` of the dma-buf, through a mapping of it, as the GPU writes
    /// through its import.
    fn write_through(chunk: &DmaBufChunk, offset: u64, byte: u8) {
        let length = usize::try_from(chunk.length).unwrap();
        // SAFETY: a fresh shared mapping of the dma-buf, written once and unmapped.
        unsafe {
            let at = libc::mmap(
                std::ptr::null_mut(),
                length,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                chunk.file.as_raw_fd(),
                0,
            );
            assert_ne!(at, libc::MAP_FAILED);
            (at.cast::<u8>())
                .add(usize::try_from(offset).unwrap())
                .write_volatile(byte);
            libc::munmap(at, length);
        }
    }

    fn run(file: u32, offset: u64, length: u64) -> Run<u32> {
        Run {
            file,
            offset,
            length,
        }
    }

    /// Runs within the limits make one chunk; past them, chunks as full as the limits let, cut
    /// at granules from the window's start, a run cut where a chunk ends in it; a granule past
    /// the limits makes none
    #[test]
    fn a_window_is_cut_into_chunks_within_the_limits_at_granules() {
        let limits = Limits {
            pieces: 3,
            bytes: 0x10000,
        };
        let one = [run(1, 0, 0x1000), run(2, 0x5000, 0x2000)];
        assert_eq!(
            chunks(&one, 0x1000, limits),
            Some(vec![vec![run(1, 0, 0x1000), run(2, 0x5000, 0x2000)]])
        );
        // Five one-page runs, three a chunk.
        let scattered: Vec<_> = (0..5u32)
            .map(|i| run(i, u64::from(i) * 0x3000, 0x1000))
            .collect();
        assert_eq!(
            chunks(&scattered, 0x1000, limits),
            Some(vec![
                vec![
                    run(0, 0, 0x1000),
                    run(1, 0x3000, 0x1000),
                    run(2, 0x6000, 0x1000)
                ],
                vec![run(3, 0x9000, 0x1000), run(4, 0xc000, 0x1000)],
            ])
        );
        // At a granule of two pages the first chunk ends at two pages; the last ends with the
        // window, at no granule.
        assert_eq!(
            chunks(&scattered, 0x2000, limits),
            Some(vec![
                vec![run(0, 0, 0x1000), run(1, 0x3000, 0x1000)],
                vec![
                    run(2, 0x6000, 0x1000),
                    run(3, 0x9000, 0x1000),
                    run(4, 0xc000, 0x1000)
                ],
            ])
        );
        // A run longer than a chunk's bytes is cut where each chunk ends.
        let long = [run(7, 0x1000, 0x25000)];
        assert_eq!(
            chunks(&long, 0x1000, limits),
            Some(vec![
                vec![run(7, 0x1000, 0x10000)],
                vec![run(7, 0x11000, 0x10000)],
                vec![run(7, 0x21000, 0x5000)],
            ])
        );
        // A run cut at a granule inside it: three runs reach 0x5000, the granule 0x2000 cuts
        // the third.
        let mixed = [
            run(1, 0, 0x1000),
            run(2, 0, 0x2000),
            run(3, 0, 0x2000),
            run(4, 0, 0x1000),
        ];
        assert_eq!(
            chunks(&mixed, 0x2000, limits),
            Some(vec![
                vec![run(1, 0, 0x1000), run(2, 0, 0x2000), run(3, 0, 0x1000)],
                vec![run(3, 0x1000, 0x1000), run(4, 0, 0x1000)],
            ])
        );
        // A granule of four single pages past three pieces fits no chunk; the whole window as
        // its granule fits only within the limits.
        assert_eq!(chunks(&scattered, 0x4000, limits), None);
        assert_eq!(chunks(&scattered, 0x5000, limits), None);
        assert!(chunks(&one, 0x3000, limits).is_some());
        // The most runs one import of a macOS 26 guest's boot took, at udmabuf's defaults.
        let boot: Vec<_> = (0..16_201u32)
            .map(|i| run(i, u64::from(i) * 0x2000, 0x1000))
            .collect();
        let defaults = Limits {
            pieces: DEFAULT_LIST_LIMIT,
            bytes: DEFAULT_SIZE_LIMIT_MB << 20,
        };
        let cut = chunks(&boot, 0x1000, defaults).unwrap();
        assert_eq!(cut.len(), 16);
        assert!(cut.iter().all(|c| c.len() as u64 <= DEFAULT_LIST_LIMIT));
        assert_eq!(cut.iter().map(Vec::len).sum::<usize>(), 16_201);
    }

    /// Chunks past one come to the window's length rounded up to the granule, the last ending
    /// with the pad's run; one chunk, or a window of whole granules, takes no pad
    #[test]
    fn several_chunks_are_padded_to_the_granule() {
        let limits = Limits {
            pieces: 2,
            bytes: 0x10000,
        };
        let pad = |length| Some(run(99, 0, length));
        let mut one = vec![run(1, 0, 0x1000)];
        assert_eq!(
            padded(&mut one, 0x4000, limits, pad),
            Some(vec![vec![run(1, 0, 0x1000)]])
        );
        let mut whole: Vec<_> = (0..4u32).map(|i| run(i, 0, 0x2000)).collect();
        assert_eq!(padded(&mut whole, 0x4000, limits, pad).unwrap().len(), 2);
        assert_eq!(whole.len(), 4, "no pad");
        // Three single pages at a granule of two: two chunks, the last padded by a page.
        let mut three: Vec<_> = (0..3u32).map(|i| run(i, 0, 0x1000)).collect();
        assert_eq!(
            padded(&mut three, 0x2000, limits, pad),
            Some(vec![
                vec![run(0, 0, 0x1000), run(1, 0, 0x1000)],
                vec![run(2, 0, 0x1000), run(99, 0, 0x1000)],
            ])
        );
        // Five pages at two a chunk: the pad joins the last page's chunk.
        let mut five: Vec<_> = (0..5u32).map(|i| run(i, 0, 0x1000)).collect();
        let cut = padded(&mut five, 0x2000, limits, pad).unwrap();
        assert_eq!(cut.iter().flatten().map(|r| r.length).sum::<u64>(), 0x6000);
        assert_eq!(cut.last().unwrap().last().unwrap().file, 99);
    }

    /// The dma-bufs of a view show the pages the view shows, in the view's order, and a range
    /// with no file behind it is refused
    #[test]
    fn a_dma_buf_shows_the_pages_of_its_range() {
        if !udmabuf() {
            return;
        }
        let page = vm::page_size();
        let at = vm::allocate_aliasable(4 * page).unwrap().addr().to_u64();
        for i in 0..4 {
            // SAFETY: the range this test just allocated, read and write.
            unsafe {
                vm::ptr(at + i * page)
                    .cast::<u8>()
                    .write_volatile(0x10 + i as u8)
            };
        }
        let chunks = of_pages(at + page, 2 * page, page).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].length, 2 * page);
        assert_eq!(read_through(&chunks[0], 0), 0x11);
        assert_eq!(read_through(&chunks[0], page), 0x12);
        // SAFETY: as above.
        unsafe { vm::ptr(at + page).cast::<u8>().write_volatile(0x5a) };
        assert_eq!(
            read_through(&chunks[0], 0),
            0x5a,
            "the same pages, not a copy"
        );
        let private = vm::anonymous(None, page, "a test's private page").unwrap();
        assert!(matches!(
            of_pages(private, page, page),
            Err(Refusal::Unresolved)
        ));
    }

    /// The runs of a range's file pages join as the table is walked, a range's first to the run
    /// before it where its pages follow on in the file; a range a byte of which has no file behind
    /// it adds none, and leaves the run it would have joined as it was. It needs no udmabuf
    #[test]
    fn runs_join_as_the_table_is_walked_and_a_refusal_adds_none() {
        let page = vm::page_size();
        let at = vm::allocate_aliasable(3 * page).unwrap().addr().to_u64();
        let shape = |runs: &[Run<Arc<OwnedFd>>]| -> Vec<(u64, u64)> {
            runs.iter().map(|r| (r.offset, r.length)).collect()
        };
        let mut runs = Vec::new();
        add_runs(&mut runs, at, page).unwrap();
        assert_eq!(runs.len(), 1);
        let offset = runs[0].offset;
        // The table forgets the third page: no file is behind it from now on.
        files::forget(at + 2 * page, page);
        assert!(matches!(
            add_runs(&mut runs, at + page, 2 * page),
            Err(Refusal::Unresolved)
        ));
        assert_eq!(shape(&runs), [(offset, page)]);
        add_runs(&mut runs, at + page, page).unwrap();
        assert_eq!(shape(&runs), [(offset, 2 * page)]);
    }

    /// One dma-buf takes the pages of a list of ranges where they are no more runs of file pages
    /// than its limit, counted as a dma-buf of them makes them: ranges whose pages follow on in
    /// the file are one run, however they are cut, and every other page of a file one run each;
    /// one more than the limit is refused with the count, and pages no file is behind are refused
    /// as unresolved. It needs no udmabuf: it makes nothing
    #[test]
    fn a_list_of_ranges_fits_one_dma_buf_by_its_runs_of_file_pages() {
        let page = vm::page_size();
        let limits = limits();
        let base = vm::allocate_aliasable(8 * page).unwrap().addr().to_u64();
        let following = [base..base + page, base + page..base + 3 * page];
        assert!(fits_one_of(&following).is_ok());
        assert!(fits_one_of(std::slice::from_ref(&(base..base + 3 * page))).is_ok());
        let private = vm::anonymous(None, page, "a test's private page").unwrap();
        assert!(matches!(
            fits_one_of(&[base..base + page, private..private + page]),
            Err(Refusal::Unresolved)
        ));
        if limits.pieces > 1 << 16 {
            eprintln!(
                "skipped the limit: udmabuf takes {} pieces here",
                limits.pieces
            );
            return;
        }
        // Every other page of a range, so no two pieces follow on in the file.
        let count = limits.pieces + 1;
        let at = vm::allocate_aliasable(2 * count * page)
            .unwrap()
            .addr()
            .to_u64();
        let apart: Vec<std::ops::Range<u64>> = (0..count)
            .map(|i| at + 2 * i * page..at + (2 * i + 1) * page)
            .collect();
        assert!(fits_one_of(&apart[1..]).is_ok(), "as many as the limit");
        match fits_one_of(&apart) {
            Err(Refusal::Limit { pieces, .. }) => assert_eq!(pieces, count),
            other => panic!("one past the limit: {other:?}"),
        }
    }

    /// A refusal holds for every range that holds the one refused ([`Refusal::holds_wider`]) by
    /// its class: pages with no file behind them, as a request of them is refused, a range past
    /// one dma-buf's runs at a granule of its length, udmabuf not opening, and a writable piece of
    /// a file open read-only do; the same range cut at a shorter granule, the kernel's refusals and
    /// a pad's missing file do not. It needs no udmabuf
    #[test]
    fn refusals_hold_for_wider_ranges_by_their_class() {
        let page = vm::page_size();
        let private = vm::anonymous(None, 2 * page, "a test's private pages").unwrap();
        let unresolved = of_pages_counted(private, 2 * page, 2 * page).unwrap_err();
        assert!(
            matches!(unresolved, Refusal::Unresolved) && unresolved.holds_wider(),
            "{unresolved}"
        );
        let limits = Limits {
            pieces: 1024,
            bytes: 64 << 20,
        };
        let past = |granule| Refusal::Limit {
            pieces: 1025,
            bytes: 1025 * page,
            granule,
            limits,
        };
        assert!(past(1025 * page).holds_wider(), "one dma-buf of the range");
        assert!(!past(16 * page).holds_wider(), "cut at a shorter granule");
        let error = |code| io::Error::from_raw_os_error(code);
        assert!(Refusal::Device(Arc::new(error(libc::ENOENT))).holds_wider());
        assert!(Refusal::ReadOnlyFile.holds_wider());
        assert!(!Refusal::Kernel(error(libc::EINVAL)).holds_wider());
        assert!(!Refusal::Kernel(error(libc::ENOMEM)).holds_wider());
        assert!(!Refusal::NoFile(error(libc::ENOMEM)).holds_wider());
    }

    /// A view of more scattered pages than one dma-buf may hold is made of several, in order,
    /// each within the host's limits
    #[test]
    fn a_view_past_the_limits_is_made_of_chunks() {
        if !udmabuf() {
            return;
        }
        let page = vm::page_size();
        let limits = limits();
        // Every other page of a range, so no two pieces follow on in the file.
        let count = limits.pieces + 2;
        let source = vm::allocate_aliasable(2 * count * page).unwrap();
        let view = vm::allocate_aliasable(count * page).unwrap();
        let host = super::super::LinuxVm::new();
        for i in 0..count {
            let from = source.addr().to_u64() + 2 * i * page;
            // SAFETY: the source range this test allocated, read and write.
            unsafe { vm::ptr(from).cast::<u8>().write_volatile((i % 251) as u8) };
            let to = view.addr().to_u64() + i * page;
            let (to, from) = (
                HostAddr::from_addr(to.to_usize()),
                HostAddr::from_addr(from.to_usize()),
            );
            assert!(host.remap_shared_overwrite(to, from, page).is_ok());
        }
        let chunks = of_pages(view.addr().to_u64(), count * page, page).unwrap();
        assert!(chunks.len() >= 2);
        assert_eq!(chunks.iter().map(|c| c.length).sum::<u64>(), count * page);
        let mut at = 0;
        for chunk in &chunks {
            assert!(chunk.length <= limits.bytes);
            assert!(chunk.length / page <= limits.pieces);
            for offset in (0..chunk.length).step_by(usize::try_from(page).unwrap()) {
                let i = at + offset / page;
                assert_eq!(read_through(chunk, offset), (i % 251) as u8);
            }
            at += chunk.length / page;
        }
    }

    /// The limits the group tests make their dma-bufs within: udmabuf's defaults, whatever the
    /// host's.
    const LIMITS: Limits = Limits {
        pieces: DEFAULT_LIST_LIMIT,
        bytes: DEFAULT_SIZE_LIMIT_MB << 20,
    };

    /// The byte at `offset` of the group's file, read through a mapping of it.
    fn read_group(group: &Group, offset: u64) -> u8 {
        let chunk = DmaBufChunk {
            file: Arc::clone(&group.file),
            offset: 0,
            length: group.length,
        };
        read_through(&chunk, offset)
    }

    /// A range of `pages` pages of memory the backing table resolves, each page's first byte
    /// `0x40 + its index`.
    fn marked(pages: u64) -> u64 {
        let page = vm::page_size();
        let at = vm::allocate_aliasable(pages * page)
            .unwrap()
            .addr()
            .to_u64();
        for i in 0..pages {
            // SAFETY: the range just allocated, read and write.
            unsafe {
                vm::ptr(at + i * page)
                    .cast::<u8>()
                    .write_volatile(0x40 + i as u8)
            };
        }
        at
    }

    /// A group dma-buf holds its members' pages back to back, in the order given, each where its
    /// offset says, and they are the members' pages, not a copy: a write through the file shows
    /// in the member's memory
    #[test]
    fn a_group_dma_buf_holds_its_members_pages_back_to_back() {
        if !udmabuf() {
            return;
        }
        let page = vm::page_size();
        let at = marked(6);
        let members = [(at + page, 2 * page), (at + 4 * page, page), (at, page)];
        let group = of_group(&members, page, LIMITS).unwrap();
        assert_eq!(group.offsets, [0, 2 * page, 3 * page]);
        assert_eq!(group.length, 4 * page);
        let read: Vec<u8> = (0..4).map(|i| read_group(&group, i * page)).collect();
        assert_eq!(read, [0x41, 0x42, 0x44, 0x40]);
        let chunk = DmaBufChunk {
            file: Arc::clone(&group.file),
            offset: 0,
            length: group.length,
        };
        write_through(&chunk, 2 * page + 7, 0x5a);
        // SAFETY: the range `marked` allocated, readable.
        let shown = unsafe { vm::ptr(at + 4 * page + 7).cast::<u8>().read_volatile() };
        assert_eq!(shown, 0x5a, "the member's own page");
    }

    /// Each member of a group starts at a multiple of the group's alignment, and the bytes
    /// between two members read zeros, of a file of the group's own
    #[test]
    fn a_groups_members_start_at_its_alignment() {
        if !udmabuf() {
            return;
        }
        let page = vm::page_size();
        let at = marked(3);
        let members = [(at, page), (at + page, 2 * page)];
        let group = of_group(&members, 4 * page, LIMITS).unwrap();
        assert_eq!(group.offsets, [0, 4 * page]);
        assert_eq!(group.length, 6 * page);
        assert_eq!(read_group(&group, 0), 0x40);
        for gap in 1..4 {
            assert_eq!(read_group(&group, gap * page), 0, "the gap's page {gap}");
        }
        assert_eq!(read_group(&group, 4 * page), 0x41);
        assert_eq!(read_group(&group, 5 * page), 0x42);
    }

    /// A group takes members while they fit one dma-buf: the first member past the limits, its
    /// bytes or its runs, or whose pages no file is behind, ends it, and is left with those after
    /// it; a first member that does not fit alone, or resolves to no file, is refused. With limits
    /// of the test's own, udmabuf's defaults, so it holds whatever the host's are
    #[test]
    fn a_group_ends_at_the_first_member_that_does_not_fit() {
        if !udmabuf() {
            return;
        }
        let page = vm::page_size();
        let at = marked(4);
        // The whole of the limit and a page, untouched: the backing table resolves it, and only a
        // group made of it would pin its pages.
        let large = vm::allocate_aliasable(LIMITS.bytes + page)
            .unwrap()
            .addr()
            .to_u64();
        let group = of_group(
            &[(at, page), (large, LIMITS.bytes), (at + page, page)],
            page,
            LIMITS,
        )
        .unwrap();
        assert_eq!(group.offsets, [0], "the second is past the bytes");
        assert_eq!(group.length, page);
        assert!(matches!(
            of_group(&[(large, LIMITS.bytes + page)], page, LIMITS),
            Err(Refusal::Limit { .. })
        ));
        // Three runs apart, at two runs a dma-buf.
        let two = Limits {
            pieces: 2,
            ..LIMITS
        };
        let apart = [(at, page), (at + 2 * page, page), (at + 3 * page, page)];
        let group = of_group(&apart, page, two).unwrap();
        assert_eq!(group.offsets, [0, page], "the third is past the runs");
        let private = vm::anonymous(None, page, "a test's private page").unwrap();
        let group = of_group(
            &[(at, page), (private, page), (at + page, page)],
            page,
            LIMITS,
        )
        .unwrap();
        assert_eq!(
            group.offsets,
            [0],
            "a member no file is behind ends the group"
        );
        assert!(matches!(
            of_group(&[(private, page), (at, page)], page, LIMITS),
            Err(Refusal::Unresolved)
        ));
    }

    /// udmabuf's limits are read from the module's parameters where the host shows them, and
    /// are its defaults where it does not; a group dma-buf takes them, its bytes at most
    /// [`GROUP_BYTES`]
    #[test]
    fn the_limits_are_the_hosts_and_a_groups_bytes_at_most_64_mib() {
        let read = |name: &str| {
            std::fs::read_to_string(format!("/sys/module/udmabuf/parameters/{name}"))
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
        };
        let expected = Limits {
            pieces: read("list_limit").unwrap_or(DEFAULT_LIST_LIMIT),
            bytes: read("size_limit_mb").unwrap_or(DEFAULT_SIZE_LIMIT_MB) << 20,
        };
        assert_eq!(limits(), expected);
        let group = group_limits();
        assert_eq!(group.pieces, expected.pieces);
        assert_eq!(group.bytes, expected.bytes.min(64 << 20));
    }

    /// The limits are read from the host at most once a second: a read less than a second after
    /// the last answers the limits kept, one a second or more after it reads the host again
    #[test]
    fn the_limits_are_read_at_most_once_a_second() {
        static READS: AtomicU64 = AtomicU64::new(0);
        fn read() -> Limits {
            let reads = READS.fetch_add(1, Ordering::Relaxed) + 1;
            Limits {
                pieces: reads,
                bytes: 0x10000,
            }
        }
        let kept = KeptLimits::new(read);
        let t0 = 5_000_000_000;
        assert_eq!(kept.at(t0).pieces, 1);
        assert_eq!(kept.at(t0 + LIMITS_STAND_NS - 1).pieces, 1);
        assert_eq!(READS.load(Ordering::Relaxed), 1);
        assert_eq!(kept.at(t0 + LIMITS_STAND_NS).pieces, 2);
        assert_eq!(kept.at(t0 + LIMITS_STAND_NS + 1).pieces, 2);
        assert_eq!(READS.load(Ordering::Relaxed), 2);
    }

    /// A refusal at the limits kept tries once more at the host's limits read now, where they
    /// changed since: a list limit raised within the second is taken at the first refusal it
    /// changes, and so is one lowered, which the kernel refuses a list past. Where the host's
    /// limits are those it was refused at, and for a refusal not of the limits, the refusal
    /// stands after one attempt
    #[test]
    fn a_refusal_at_the_limits_kept_tries_again_at_the_hosts() {
        static HOST: AtomicU64 = AtomicU64::new(2);
        static READS: AtomicU64 = AtomicU64::new(0);
        fn read() -> Limits {
            READS.fetch_add(1, Ordering::Relaxed);
            Limits {
                pieces: HOST.load(Ordering::Relaxed),
                bytes: 0x10_0000,
            }
        }
        let kept = KeptLimits::new(read);
        let now = 1;
        let tries = std::cell::Cell::new(0);
        // Three pieces, past a list limit of 2; a list the kernel refuses past the host's limit.
        let three = |limits: Limits| {
            tries.set(tries.get() + 1);
            if limits.pieces > HOST.load(Ordering::Relaxed) {
                return Err(Refusal::Kernel(io::Error::from_raw_os_error(libc::EINVAL)));
            }
            if limits.pieces < 3 {
                return Err(Refusal::Limit {
                    pieces: 3,
                    bytes: 0x3000,
                    granule: 0x3000,
                    limits,
                });
            }
            Ok(limits.pieces)
        };
        // The host's limits, read now, refuse it: one attempt.
        assert!(matches!(
            within(&kept, now, three),
            Err(Refusal::Limit { .. })
        ));
        assert_eq!((tries.replace(0), READS.load(Ordering::Relaxed)), (1, 2));
        // Raised within the second: refused at the kept 2, taken at the host's 4.
        HOST.store(4, Ordering::Relaxed);
        assert_eq!(within(&kept, now, three).unwrap(), 4);
        assert_eq!(tries.replace(0), 2);
        assert_eq!(within(&kept, now, three).unwrap(), 4);
        assert_eq!(tries.replace(0), 1);
        // Lowered within the second, below what it needs: the kernel refuses the list made at the
        // kept 4, and the host's 1 refuses it.
        HOST.store(1, Ordering::Relaxed);
        assert!(matches!(
            within(&kept, now, three),
            Err(Refusal::Limit { .. })
        ));
        assert_eq!(tries.replace(0), 2);
        // A refusal not of the limits reads nothing.
        let reads = READS.load(Ordering::Relaxed);
        let unresolved = |_: Limits| -> Result<(), Refusal> { Err(Refusal::Unresolved) };
        assert!(matches!(
            within(&kept, now, unresolved),
            Err(Refusal::Unresolved)
        ));
        assert_eq!(READS.load(Ordering::Relaxed), reads);
    }

    /// Grouped views are answered with their pieces of one group dma-buf while their memberships
    /// live and the file is held: the same file for each, each view at its offset, a part of a
    /// view at its offset there, which holds the view's own pages, and none at an offset that is
    /// not a multiple of the alignment asked, nor past a view's end. A view whose membership
    /// went, and every view once the group's file is let go of (a device's recovery asking
    /// again after the imports went), is answered by no piece, so its pages take a dma-buf of
    /// their own, as before
    #[test]
    fn grouped_views_are_answered_with_their_pieces_while_their_group_lives() {
        if !udmabuf() {
            return;
        }
        let page = vm::page_size();
        let at = marked(6);
        let views = [(at, 2 * page), (at + 3 * page, page), (at + 5 * page, page)];
        let (files, members) = group_views_at(&views, page);
        assert_eq!(files.len(), 1, "one group");
        assert!(members.iter().all(Option::is_some), "each view a member");
        let pieces: Vec<(Arc<OwnedFd>, u64)> = (views.iter())
            .map(|&(first, length)| piece_of(first, length, page).expect("a piece"))
            .collect();
        assert!(pieces.iter().all(|(f, _)| Arc::ptr_eq(f, &files[0])));
        let offsets: Vec<u64> = pieces.iter().map(|p| p.1).collect();
        assert_eq!(offsets, [0, 2 * page, 3 * page]);
        let group = Group {
            file: Arc::clone(&files[0]),
            offsets: offsets.clone(),
            length: 4 * page,
        };
        assert_eq!(read_group(&group, 2 * page), 0x43, "the second view's page");
        // A part of the first view, and its alignment.
        let (file, offset) = piece_of(at + page, page, page).expect("a part's piece");
        assert!(Arc::ptr_eq(&file, &files[0]));
        assert_eq!(offset, page);
        assert!(
            piece_of(at + page, page, 2 * page).is_none(),
            "not a multiple of 2 pages"
        );
        assert!(
            piece_of(at + page, 2 * page, page).is_none(),
            "past the view's end"
        );
        // Asked again, as at a recovery, while the group lives: the same piece.
        let again = piece_of(views[1].0, views[1].1, page).expect("a piece again");
        assert!(Arc::ptr_eq(&again.0, &files[0]));
        assert_eq!(again.1, 2 * page);
        drop(again);
        // A membership gone: none for that view.
        let mut members = members;
        members[0] = None;
        assert!(
            piece_of(views[0].0, views[0].1, page).is_none(),
            "its membership went"
        );
        assert!(piece_of(views[2].0, views[2].1, page).is_some());
        // The group's file let go of: none for any, though their memberships live.
        drop((pieces, file, group, files));
        assert!(
            piece_of(views[1].0, views[1].1, page).is_none(),
            "the file went"
        );
        assert!(
            piece_of(views[2].0, views[2].1, page).is_none(),
            "the file went"
        );
        drop(members);
    }

    /// A group's members start at the alignment asked; a view a group cannot start with (its
    /// pages resolving to no file) is left alone and the next starts the group; and a group that
    /// would take one view alone is not made
    #[test]
    fn views_no_group_takes_are_left_alone() {
        if !udmabuf() {
            return;
        }
        let page = vm::page_size();
        let at = marked(4);
        let (files, members) = group_views_at(&[(at, page), (at + 2 * page, page)], 4 * page);
        assert_eq!(files.len(), 1);
        assert!(members.iter().all(Option::is_some));
        assert_eq!(piece_of(at + 2 * page, page, 4 * page).unwrap().1, 4 * page);
        drop((files, members));
        let private = vm::anonymous(None, page, "a test's private page").unwrap();
        let views = [(private, page), (at, page), (at + page, page)];
        let (files, members) = group_views_at(&views, page);
        assert_eq!(files.len(), 1);
        assert!(members[0].is_none(), "no file behind it: alone");
        assert!(members[1].is_some() && members[2].is_some());
        assert!(piece_of(private, page, page).is_none());
        drop((files, members));
        let (files, members) = group_views_at(&[(at + 3 * page, page)], page);
        assert!(files.is_empty() && members[0].is_none(), "one view alone");
    }

    /// Pieces of every region's file the way the runtime takes them: one piece at an offset the
    /// alignment divides; several only with each boundary on the granule and whole granules, as a
    /// pad would be a region's other pages
    #[test]
    fn region_pieces_keep_the_runtimes_rules() {
        let page = 0x1000;
        // One piece: any page, where its alignment divides its offset.
        assert!(takes(&[(3 * page, 2 * page)], 2 * page, 2 * page, page));
        assert!(!takes(
            &[(3 * page, 2 * page)],
            2 * page,
            2 * page,
            2 * page
        ));
        // Two pieces across a region's end: the boundary on the granule, whole granules.
        let across = [(12 * page, 4 * page), (0, 4 * page)];
        assert!(takes(&across, 8 * page, 4 * page, 4 * page));
        assert!(
            !takes(&across, 8 * page, 8 * page, 8 * page),
            "a granule of its length"
        );
        assert!(
            !takes(
                &[(12 * page, 4 * page), (0, 3 * page)],
                7 * page,
                4 * page,
                4 * page
            ),
            "a pad past the bytes"
        );
        assert!(
            !takes(
                &[(13 * page, 3 * page), (0, 5 * page)],
                8 * page,
                4 * page,
                page
            ),
            "a boundary off the granule"
        );
        assert!(!takes(&[], 0, page, page));
        assert!(
            !takes(&[(0, page)], 2 * page, page, page),
            "short of the bytes"
        );
    }

    /// A range of the guest's RAM for a test: a memfd of `pages` pages mapped shared, each page's
    /// first byte its number, and registered as the guest's RAM while it lives.
    struct TestRam {
        at: u64,
        length: u64,
        _registration: files::Registration,
    }

    impl TestRam {
        fn new(pages: u64) -> TestRam {
            let page = vm::page_size();
            let length = pages * page;
            let file = vm::memfd(length).unwrap();
            // SAFETY: a fresh shared mapping of the file, unmapped as the range drops.
            let start = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    length.to_usize(),
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    file.as_raw_fd(),
                    0,
                )
            };
            assert_ne!(start, libc::MAP_FAILED);
            let at = start.addr().to_u64();
            for i in 0..pages {
                // SAFETY: a page of the mapping above.
                unsafe {
                    vm::ptr(at + i * page)
                        .cast::<u8>()
                        .write_volatile(u8::try_from(i).unwrap())
                };
            }
            let backing = files::Backing {
                file: Arc::clone(&file),
                offset: 0,
            };
            let registration = files::Registration::guest_ram(at, length, backing);
            TestRam {
                at,
                length,
                _registration: registration,
            }
        }
    }

    impl Drop for TestRam {
        fn drop(&mut self) {
            // SAFETY: the mapping `new` made.
            unsafe { libc::munmap(vm::ptr(self.at), self.length.to_usize()) };
        }
    }

    /// The uses ([`Region::users`]) and the kept imports of the region whose dma-buf is `file`.
    fn region_of(file: &Arc<OwnedFd>) -> Option<(u64, usize)> {
        let regions = REGIONS.lock();
        (regions.values())
            .find(|r| std::ptr::eq(r.file.as_ptr(), Arc::as_ptr(file)))
            .map(|r| (r.users, r.kept.len()))
    }

    /// When the region whose dma-buf is `file` began to linger ([`Region::idle_since`]).
    fn idle_since(file: &Arc<OwnedFd>) -> Option<u64> {
        let regions = REGIONS.lock();
        (regions.values())
            .find(|r| std::ptr::eq(r.file.as_ptr(), Arc::as_ptr(file)))
            .and_then(|r| r.idle_since)
    }

    /// A region an object was made over is kept imported on the device that made it, once
    /// ([`keep_regions`]): its kept import lingers past its last use, which a use ends, the
    /// region's own dma-buf answered again; the linger ends [`LINGER_NS`] after the last use, the
    /// keep and the dma-buf with it. A region no source uses any more is not kept. Where udmabuf
    /// does not open, or the device keeps no import of the dma-buf (NVIDIA's driver refuses one
    /// dedicated to nothing this small), it says so and checks no more
    #[test]
    fn a_regions_kept_import_lingers_past_its_last_use() {
        if !udmabuf() {
            return;
        }
        let d = crate::gpu::device::tests::dev();
        let page = vm::page_size();
        let region = 16 * page;
        let ram = TestRam::new(32);
        let made = || REGIONS_MADE.load(Ordering::Relaxed);
        let one = ram_pieces_in(ram.at, page, page, page, region).unwrap();
        let file = Arc::clone(&one.chunks[0].file);
        keep_regions(&d.raw, refs_of(&one));
        if region_of(&file) != Some((1, 1)) {
            eprintln!(
                "skipped: {} keeps no import of a region's dma-buf ({:?})",
                d.raw.properties().name,
                region_of(&file)
            );
            return;
        }
        keep_regions(&d.raw, refs_of(&one));
        assert_eq!(region_of(&file), Some((1, 1)), "kept once a device");
        let two = ram_pieces_in(ram.at + page, page, page, page, region).unwrap();
        assert_eq!(region_of(&file), Some((2, 1)));
        drop((one, two));
        assert_eq!(region_of(&file), Some((0, 1)), "kept past its last use");
        let since = idle_since(&file).expect("lingering");
        let (gone, next) = lingered(since);
        assert!(gone.is_empty(), "lingering still");
        assert_eq!(next, Some(since + LINGER_NS));
        let before = made();
        let three = ram_pieces_in(ram.at + 2 * page, page, page, page, region).unwrap();
        assert!(
            Arc::ptr_eq(&three.chunks[0].file, &file),
            "its own dma-buf again"
        );
        assert_eq!(made(), before, "none made");
        assert_eq!((region_of(&file), idle_since(&file)), (Some((1, 1)), None));
        drop(three);
        let since = idle_since(&file).expect("lingering again");
        let (gone, next) = lingered(since + LINGER_NS);
        assert_eq!((gone.len(), next), (1, None), "its linger ended");
        drop(gone);
        let held = Arc::downgrade(&file);
        drop(file);
        assert!(held.upgrade().is_none(), "the dma-buf went with its keep");
        // A region whose last use went before the object's making returned is not kept.
        let four = ram_pieces_in(ram.at + 17 * page, page, page, page, region).unwrap();
        let (file, refs) = (Arc::clone(&four.chunks[0].file), refs_of(&four));
        drop(four);
        keep_regions(&d.raw, refs);
        assert_eq!(region_of(&file), Some((0, 0)), "not kept");
    }

    /// The regions `pieces` lie in, weakly ([`RegionUse::region`]).
    fn refs_of(pieces: &RegionPieces) -> Vec<RegionRef> {
        pieces.uses.iter().map(RegionUse::region).collect()
    }

    /// Two requests in one region of the guest's RAM are pieces of one file, at their pages'
    /// offsets in the region, which show those pages; one in another region is a piece of another
    /// file; the region's file is made once while a piece of it is held, and made again once the
    /// last went
    #[test]
    fn requests_in_one_region_share_its_file() {
        if !udmabuf() {
            return;
        }
        let page = vm::page_size();
        // Regions of 16 pages, four of them.
        let region = 16 * page;
        let ram = TestRam::new(64);
        let made = || REGIONS_MADE.load(Ordering::Relaxed);
        let before = made();
        let one = ram_pieces_in(ram.at + 2 * page, 3 * page, 3 * page, page, region).unwrap();
        let two = ram_pieces_in(ram.at + 9 * page, page, page, page, region).unwrap();
        assert_eq!((one.chunks.len(), two.chunks.len()), (1, 1));
        assert!(
            Arc::ptr_eq(&one.chunks[0].file, &two.chunks[0].file),
            "one file"
        );
        assert_eq!(
            (one.chunks[0].offset, one.chunks[0].length),
            (2 * page, 3 * page)
        );
        assert_eq!(
            (two.chunks[0].offset, two.chunks[0].length),
            (9 * page, page)
        );
        assert_eq!(made() - before, 1, "made once");
        assert_eq!(
            region_of(&one.chunks[0].file),
            Some((2, 0)),
            "a use of it each answer, kept nowhere"
        );
        assert_eq!(
            read_file(&one.chunks[0].file, 2 * page),
            2,
            "the region's own pages"
        );
        assert_eq!(read_file(&one.chunks[0].file, 9 * page), 9);
        let other = ram_pieces_in(ram.at + 17 * page, page, page, page, region).unwrap();
        assert!(
            !Arc::ptr_eq(&one.chunks[0].file, &other.chunks[0].file),
            "another region"
        );
        assert_eq!(other.chunks[0].offset, page);
        assert_eq!(read_file(&other.chunks[0].file, page), 17);
        assert_eq!(made() - before, 2);
        // The first region's pieces let go of: its file goes, and the next request makes it again.
        let held = Arc::downgrade(&one.chunks[0].file);
        drop((one, two));
        assert!(
            held.upgrade().is_none(),
            "the file went with its last piece"
        );
        let again = ram_pieces_in(ram.at + 4 * page, page, page, page, region).unwrap();
        assert_eq!(made() - before, 3, "made again");
        assert_eq!(read_file(&again.chunks[0].file, 4 * page), 4);
        drop((again, other));
    }

    /// A request whose pages cross into the next region takes a piece of each region's file
    /// where the runtime binds several chunks at a granule the crossing lies on, and none where
    /// it takes one chunk alone (its granule its length), or the bytes are not whole granules;
    /// memory of the library's own and an offset the alignment does not divide take none
    #[test]
    fn requests_across_regions_and_off_the_ram() {
        if !udmabuf() {
            return;
        }
        let page = vm::page_size();
        let region = 16 * page;
        let ram = TestRam::new(64);
        let across = ram_pieces_in(ram.at + 12 * page, 8 * page, 4 * page, 4 * page, region);
        let across = across.expect("a piece of each region");
        assert_eq!(
            (across.chunks.len(), across.uses.len()),
            (2, 2),
            "a use of each"
        );
        assert_eq!(
            (across.chunks[0].offset, across.chunks[0].length),
            (12 * page, 4 * page)
        );
        assert_eq!(
            (across.chunks[1].offset, across.chunks[1].length),
            (0, 4 * page)
        );
        assert!(!Arc::ptr_eq(&across.chunks[0].file, &across.chunks[1].file));
        assert_eq!(read_file(&across.chunks[1].file, 3 * page), 19);
        assert!(
            ram_pieces_in(ram.at + 12 * page, 8 * page, 8 * page, page, region).is_none(),
            "one chunk alone"
        );
        assert!(
            ram_pieces_in(ram.at + 12 * page, 7 * page, 4 * page, 4 * page, region).is_none(),
            "a pad past the bytes"
        );
        assert!(
            ram_pieces_in(ram.at + 2 * page, page, page, 4 * page, region).is_none(),
            "off the alignment"
        );
        let own = vm::allocate_aliasable(4 * page).unwrap().addr().to_u64();
        assert!(
            ram_pieces_in(own, page, page, page, region).is_none(),
            "memory of the library's own"
        );
        // The file's last region is cut at its end: 24 pages in regions of 16.
        let short = TestRam::new(24);
        let tail = ram_pieces_in(short.at + 18 * page, 2 * page, 2 * page, page, region).unwrap();
        assert_eq!(tail.chunks[0].offset, 2 * page);
        // SAFETY: a dma-buf answers `SEEK_END` with its size.
        let bytes = unsafe { libc::lseek(tail.chunks[0].file.as_raw_fd(), 0, libc::SEEK_END) };
        assert_eq!(u64::try_from(bytes).unwrap(), 8 * page);
        assert_eq!(read_file(&tail.chunks[0].file, 3 * page), 19);
    }
}
