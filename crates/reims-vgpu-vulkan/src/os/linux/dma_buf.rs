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
//! with the imports it made (`super::import`'s `host_memory`), so it lives while any buffer or heap
//! bound in it does and goes with the last, and a later request in the region makes it again (the
//! trace's `ramregion` writes each made, and `rampiece` each request's answer). A region's first
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
use metal2vulkan_runtime::DmaBufChunk;

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

/// What one dma-buf may hold: pieces, and bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Limits {
    pub(crate) pieces: u64,
    pub(crate) bytes: u64,
}

/// udmabuf's limits as the host has them now, its defaults where it shows none.
pub(super) fn limits() -> Limits {
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
/// follow on, the one before included.
fn add_runs(runs: &mut Vec<Run<Arc<OwnedFd>>>, first: u64, length: u64) -> Result<(), Refusal> {
    let page = vm::page_size();
    debug_assert!(first.is_multiple_of(page) && length.is_multiple_of(page));
    let resolved = files::pieces(first, length).ok_or(Refusal::Unresolved)?;
    for piece in resolved {
        let (backing, bytes) = (piece.backing, piece.length);
        match runs.last_mut() {
            Some(last)
                if Arc::ptr_eq(&last.file, &backing.file)
                    && last.offset + last.length == backing.offset =>
            {
                last.length += bytes;
            }
            _ => runs.push(Run {
                file: backing.file,
                offset: backing.offset,
                length: bytes,
            }),
        }
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
    let limits = limits();
    let bytes: u64 = ranges.iter().map(|&(_, length)| length).sum();
    if runs.len() as u64 > limits.pieces || bytes > limits.bytes {
        return Err(Refusal::Limit {
            pieces: runs.len() as u64,
            bytes,
            granule: bytes,
            limits,
        });
    }
    create(device, &runs)
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
    let limits = limits();
    if pieces > limits.pieces || bytes > limits.bytes {
        return Err(Refusal::Limit {
            pieces,
            bytes,
            granule: bytes,
            limits,
        });
    }
    Ok(())
}

/// Dma-bufs of the `length` bytes from `first`, whole pages, each resolved through the backing
/// table, which in order are those bytes ([`add_runs`]), cut into chunks within udmabuf's limits
/// at multiples of `granule` from `first` (module docs).
pub(crate) fn of_pages(first: u64, length: u64, granule: u64) -> Result<Vec<DmaBufChunk>, Refusal> {
    let page = vm::page_size();
    let mut runs: Vec<Run<Arc<OwnedFd>>> = Vec::new();
    add_runs(&mut runs, first, length)?;
    let device = device()?;
    let limits = limits();
    let granule = granule.max(page);
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
    made.collect()
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
    let limits = limits();
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
    let limits = group_limits();
    let mut files = Vec::new();
    let mut members: Vec<Option<Membership>> = views.iter().map(|_| None).collect();
    let align = align.max(vm::page_size());
    let mut at = 0;
    while views.len() - at > 1 {
        let group = match of_group(&views[at..], align, limits) {
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

/// The region dma-bufs made and alive now, by the guest RAM file's identity (its `Arc`'s
/// address), the regions' bytes and the region's index in the file (module docs, "Regions").
static REGIONS: Mutex<BTreeMap<(usize, u64, u64), Region>> = Mutex::new(BTreeMap::new());

/// A region's dma-buf in [`REGIONS`].
#[derive(Debug)]
struct Region {
    /// The RAM's file, held weakly: its `Arc`'s allocation, whose address keys the entry, is not
    /// let go of while the entry lives, so no other file is ever found under the key.
    _ram: Weak<OwnedFd>,
    /// The region's dma-buf: the runtime's imports of it, and an answer not yet imported, hold it.
    file: Weak<OwnedFd>,
}

/// How many region dma-bufs were made (module docs, "Regions"): a diagnostic, which the trace
/// writes as each is made.
static REGIONS_MADE: AtomicU64 = AtomicU64::new(0);

/// Pieces of the guest RAM's region dma-bufs that hold the `length` bytes from `first` (whole
/// pages) in order, as a no-copy buffer's or heap's request at `granule` and `alignment` takes
/// them (module docs, "Regions"): each piece one run of a file's pages inside one region, at its
/// offset in the region's dma-buf, made at the region's first piece. `None` where a byte shows no
/// page of the guest's RAM mapped writable, where the pieces break the request's rules (a piece at
/// an offset its alignment does not divide; several, where a boundary between two lies off the
/// granule, or the length is not whole granules, which would take a pad of other pages), or where
/// udmabuf makes no region's dma-buf: the request is answered as before then.
pub(crate) fn ram_pieces(
    first: u64,
    length: u64,
    granule: u64,
    alignment: u64,
) -> Option<Vec<DmaBufChunk>> {
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
) -> Option<Vec<DmaBufChunk>> {
    let answer = cut_in_regions(first, length, granule, alignment, region);
    // The trace's `rampiece`: the request's first byte, its length, the pieces answered, and
    // why none were where none were.
    let (count, why) = match &answer {
        Ok(chunks) => (chunks.len().to_u64(), 0),
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
) -> Result<Vec<DmaBufChunk>, NoPieces> {
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
    cut.into_iter()
        .map(|(ram, index, into, bytes)| {
            let file = region_file(device, &ram, index, region).ok_or(NoPieces::Made)?;
            Ok(DmaBufChunk {
                file,
                offset: into,
                length: bytes,
            })
        })
        .collect()
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

/// The dma-buf of region `index` of the RAM's file `ram`, `region` bytes from `index * region`
/// (fewer at the file's end): the one alive in [`REGIONS`], or one made now and entered there. A
/// region made meanwhile by another thread is the one answered, and this one goes. `None` where
/// the file's size is not known or the kernel refuses the dma-buf.
fn region_file(
    device: &OwnedFd,
    ram: &Arc<OwnedFd>,
    index: u64,
    region: u64,
) -> Option<Arc<OwnedFd>> {
    let key = (Arc::as_ptr(ram).addr(), region, index);
    let alive = |regions: &BTreeMap<(usize, u64, u64), Region>| {
        regions.get(&key).and_then(|r| r.file.upgrade())
    };
    if let Some(file) = alive(&REGIONS.lock()) {
        return Some(file);
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
    if let Some(file) = alive(&regions) {
        return Some(file);
    }
    regions.retain(|_, r| r.file.strong_count() > 0);
    regions.insert(
        key,
        Region {
            _ram: Arc::downgrade(ram),
            file: Arc::downgrade(&made),
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
    Some(made)
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
    let limits = limits();
    let page = vm::page_size();
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
}

/// One dma-buf of `runs`, in order (`UDMABUF_CREATE_LIST`).
fn create(device: &OwnedFd, runs: &[Run<Arc<OwnedFd>>]) -> Result<OwnedFd, Refusal> {
    // `struct udmabuf_create_list` of Linux's `include/uapi/linux/udmabuf.h` as little-endian
    // words: `flags` and `count`, then each `struct udmabuf_create_item`'s `memfd` and its
    // padding, `offset` and `size`.
    let count = u32::try_from(runs.len()).unwrap_or(u32::MAX);
    let mut call = vec![u64::from(FLAGS_CLOEXEC) | (u64::from(count) << 32)];
    for run in runs {
        let memfd = u32::try_from(run.file.as_raw_fd()).unwrap_or(u32::MAX);
        call.extend([u64::from(memfd), run.offset, run.length]);
    }
    // SAFETY: the list describes open memfds the backing table holds through the call; the kernel reads the header and its items.
    let made = unsafe { libc::ioctl(device.as_raw_fd(), CREATE_LIST as _, call.as_ptr()) };
    if made < 0 {
        return Err(Refusal::Kernel(io::Error::last_os_error()));
    }
    // SAFETY: the dma-buf the call just made, owned from here on.
    Ok(unsafe { OwnedFd::from_raw_fd(made) })
}

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
        assert_eq!((one.len(), two.len()), (1, 1));
        assert!(Arc::ptr_eq(&one[0].file, &two[0].file), "one file");
        assert_eq!((one[0].offset, one[0].length), (2 * page, 3 * page));
        assert_eq!((two[0].offset, two[0].length), (9 * page, page));
        assert_eq!(made() - before, 1, "made once");
        assert_eq!(
            read_file(&one[0].file, 2 * page),
            2,
            "the region's own pages"
        );
        assert_eq!(read_file(&one[0].file, 9 * page), 9);
        let other = ram_pieces_in(ram.at + 17 * page, page, page, page, region).unwrap();
        assert!(!Arc::ptr_eq(&one[0].file, &other[0].file), "another region");
        assert_eq!(other[0].offset, page);
        assert_eq!(read_file(&other[0].file, page), 17);
        assert_eq!(made() - before, 2);
        // The first region's pieces let go of: its file goes, and the next request makes it again.
        let held = Arc::downgrade(&one[0].file);
        drop((one, two));
        assert!(
            held.upgrade().is_none(),
            "the file went with its last piece"
        );
        let again = ram_pieces_in(ram.at + 4 * page, page, page, page, region).unwrap();
        assert_eq!(made() - before, 3, "made again");
        assert_eq!(read_file(&again[0].file, 4 * page), 4);
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
        assert_eq!(across.len(), 2);
        assert_eq!((across[0].offset, across[0].length), (12 * page, 4 * page));
        assert_eq!((across[1].offset, across[1].length), (0, 4 * page));
        assert!(!Arc::ptr_eq(&across[0].file, &across[1].file));
        assert_eq!(read_file(&across[1].file, 3 * page), 19);
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
        assert_eq!(tail[0].offset, 2 * page);
        // SAFETY: a dma-buf answers `SEEK_END` with its size.
        let bytes = unsafe { libc::lseek(tail[0].file.as_raw_fd(), 0, libc::SEEK_END) };
        assert_eq!(u64::try_from(bytes).unwrap(), 8 * page);
        assert_eq!(read_file(&tail[0].file, 3 * page), 19);
    }
}
