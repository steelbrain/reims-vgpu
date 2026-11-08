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
//!
//! Pieces of the block of zeros read-only aliases share ([`super::import::allocate_zeros`]) are
//! made of files of zeros of the import's own instead ([`Zeros`]), as an import's twin maps private
//! zeros there: the GPU's writes stay in them and never reach the block. Its source answers the
//! same pieces of the same files for every request over the same pages, as the runtime's contract
//! asks (`rt::HostMemory::with_dma_buf`), so a texture's dma-buf over the import's pages shows
//! what the GPU wrote there through the import's.

use std::collections::BTreeMap;
use std::fmt;
use std::io;
use std::os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd};
use std::sync::{Arc, OnceLock};

use crate::core::sync::Mutex;
use metal2vulkan_runtime::DmaBufChunk;

use super::files::{self, Backing};
use super::vm;

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
    /// The kernel gave no file for zeros the range takes, a pad's or the block of zeros' pieces
    /// ([`vm::memfd`], which counts it as the host's memory running out).
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

/// The files of zeros one import's dma-buf source answers the block of zeros' pages with
/// ([`of_pages`]), each by the first address of the pages it holds: a page asked for again is
/// answered by the same piece of the same file, and pages asked for the first time by a file made
/// for them, so every request over the same pages answers the same files (module docs).
#[derive(Debug, Default)]
pub(crate) struct Zeros(Mutex<BTreeMap<u64, (u64, Arc<OwnedFd>)>>);

impl Zeros {
    /// The pieces of its files that answer the `length` bytes of the block of zeros at `at`, in
    /// order, each with its length: those its files hold already, and a new file for each run of
    /// the rest; the kernel's refusal where it gives no file for one ([`vm::memfd`]).
    fn pieces(&self, at: u64, length: u64) -> Result<Vec<(Backing, u64)>, Refusal> {
        let end = at.saturating_add(length);
        let mut files = self.0.lock();
        let held: Vec<(u64, u64, Arc<OwnedFd>)> = (files.range(..end))
            .filter(|&(&first, &(bytes, _))| first.saturating_add(bytes) > at)
            .map(|(&first, (bytes, fd))| (first, *bytes, Arc::clone(fd)))
            .collect();
        let made = |files: &mut BTreeMap<u64, (u64, Arc<OwnedFd>)>, first: u64, bytes: u64| {
            let fd = vm::memfd(bytes).map_err(Refusal::NoFile)?;
            files.insert(first, (bytes, Arc::clone(&fd)));
            Ok::<_, Refusal>((Backing { fd, offset: 0 }, bytes))
        };
        let mut pieces = Vec::new();
        let mut next = at;
        for (first, bytes, fd) in held {
            if first > next {
                pieces.push(made(&mut files, next, first - next)?);
            }
            let from = first.max(next);
            let to = first.saturating_add(bytes).min(end);
            let offset = from - first;
            pieces.push((Backing { fd, offset }, to - from));
            next = to;
        }
        if next < end {
            pieces.push(made(&mut files, next, end - next)?);
        }
        Ok(pieces)
    }
}

/// The runs of file pages behind the `length` bytes from `first`, whole pages, each resolved
/// through the backing table, added to `runs` in order: one for each run of a file's pages that
/// follow on, the one before included, the block of zeros' pieces from `zeros`, the files of zeros
/// of the import's own.
fn add_runs(
    runs: &mut Vec<Run<Arc<OwnedFd>>>,
    first: u64,
    length: u64,
    zeros: &Zeros,
) -> Result<(), Refusal> {
    let page = vm::page_size();
    debug_assert!(first.is_multiple_of(page) && length.is_multiple_of(page));
    let resolved = files::pieces(first, length).ok_or(Refusal::Unresolved)?;
    let of_zeros = super::import::of_zeros(&resolved);
    for (piece, of_zeros) in resolved.into_iter().zip(of_zeros) {
        let backings = if of_zeros {
            zeros.pieces(first + piece.at, piece.length)?
        } else {
            vec![(piece.backing, piece.length)]
        };
        for (backing, bytes) in backings {
            match runs.last_mut() {
                Some(last)
                    if Arc::ptr_eq(&last.file, &backing.fd)
                        && last.offset + last.length == backing.offset =>
                {
                    last.length += bytes;
                }
                _ => runs.push(Run {
                    file: backing.fd,
                    offset: backing.offset,
                    length: bytes,
                }),
            }
        }
    }
    Ok(())
}

/// One dma-buf of the pages of `ranges`, each a first byte and a length, whole pages in
/// ascending order and apart, which in order are those pages (as [`add_runs`] resolves them);
/// refused with udmabuf's limits where they are more than one dma-buf takes.
pub(crate) fn of_ranges(ranges: &[(u64, u64)], zeros: &Zeros) -> Result<OwnedFd, Refusal> {
    let mut runs = Vec::new();
    for &(first, length) in ranges {
        add_runs(&mut runs, first, length, zeros)?;
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
/// to no file. It counts the runs as [`add_runs`] makes them, and makes nothing: each piece of the
/// block of zeros counts as a run of its own, as a file of zeros of the import's own holds it.
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
        let of_zeros = super::import::of_zeros(&resolved);
        for (piece, of_zeros) in resolved.into_iter().zip(of_zeros) {
            let follows = matches!(
                &last,
                Some((fd, end)) if !of_zeros && Arc::ptr_eq(fd, &piece.backing.fd)
                    && *end == piece.backing.offset
            );
            if !follows {
                pieces += 1;
            }
            let end = piece.backing.offset + piece.length;
            last = (!of_zeros).then_some((piece.backing.fd, end));
        }
        bytes = bytes.saturating_add(length);
    }
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
pub(crate) fn of_pages(
    first: u64,
    length: u64,
    granule: u64,
    zeros: &Zeros,
) -> Result<Vec<DmaBufChunk>, Refusal> {
    let page = vm::page_size();
    let mut runs: Vec<Run<Arc<OwnedFd>>> = Vec::new();
    add_runs(&mut runs, first, length, zeros)?;
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
            fd,
            length: chunk.iter().map(|r| r.length).sum(),
        })
    });
    made.collect()
}

/// One dma-buf of a view's pages ([`of_view`]): `length` bytes of the view from `at`, to map with
/// protection `prot`.
#[derive(Debug)]
pub(crate) struct ViewBuf {
    pub(crate) fd: OwnedFd,
    pub(crate) at: u64,
    pub(crate) length: u64,
    pub(crate) prot: i32,
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
    for piece in pieces.iter().filter(|p| p.prot & libc::PROT_WRITE != 0) {
        let fd = &piece.backing.fd;
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
                file: Arc::clone(&p.backing.fd),
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
    // SAFETY: the list describes open memfds the backing table and the import's zeros hold
    // through the call; the kernel reads the header and its items.
    let made = unsafe { libc::ioctl(device.as_raw_fd(), CREATE_LIST as _, call.as_ptr()) };
    if made < 0 {
        return Err(Refusal::Kernel(io::Error::last_os_error()));
    }
    // SAFETY: the dma-buf the call just made, owned from here on.
    Ok(unsafe { OwnedFd::from_raw_fd(made) })
}

#[cfg(test)]
mod tests {
    use crate::core::num::{ToU64 as _, ToUsize as _};
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
                chunk.fd.as_raw_fd(),
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
                chunk.fd.as_raw_fd(),
                0,
            );
            assert_ne!(at, libc::MAP_FAILED);
            (at.cast::<u8>())
                .add(usize::try_from(offset).unwrap())
                .write_volatile(byte);
            libc::munmap(at, length);
        }
    }

    /// A source's files of zeros answer pages asked for again with the same pieces of the same
    /// files, whatever the requests' extents, and pages asked for the first time with files of
    /// their own
    #[test]
    fn zero_pages_asked_again_take_the_same_pieces() {
        let zeros = Zeros::default();
        let shape = |pieces: &[(Backing, u64)]| -> Vec<(u64, u64)> {
            pieces.iter().map(|(b, n)| (b.offset, *n)).collect()
        };
        let middle = zeros.pieces(0x3000, 0x2000).unwrap();
        assert_eq!(shape(&middle), [(0, 0x2000)]);
        let inside = zeros.pieces(0x4000, 0x1000).unwrap();
        assert_eq!(shape(&inside), [(0x1000, 0x1000)]);
        assert!(Arc::ptr_eq(&inside[0].0.fd, &middle[0].0.fd));
        // Pages on both sides, asked for the first time, take files of their own around it.
        let around = zeros.pieces(0x1000, 0x6000).unwrap();
        assert_eq!(shape(&around), [(0, 0x2000), (0, 0x2000), (0, 0x2000)]);
        assert!(Arc::ptr_eq(&around[1].0.fd, &middle[0].0.fd));
        assert!(!Arc::ptr_eq(&around[0].0.fd, &around[2].0.fd));
        let again = zeros.pieces(0x2000, 0x4000).unwrap();
        assert_eq!(shape(&again), [(0x1000, 0x1000), (0, 0x2000), (0, 0x1000)]);
        for (a, b) in again.iter().zip(&around) {
            assert!(Arc::ptr_eq(&a.0.fd, &b.0.fd));
        }
        // Another import's source answers files of its own.
        let other = Zeros::default().pieces(0x3000, 0x2000).unwrap();
        assert!(!Arc::ptr_eq(&other[0].0.fd, &middle[0].0.fd));
    }

    /// Where the kernel gives no file for zeros, the request is refused (`Refusal::NoFile`),
    /// counted by default and keeping nothing; the pages asked for again take a file then
    #[test]
    fn zero_pages_the_kernel_gives_no_file_for_are_refused() {
        let zeros = Zeros::default();
        vm::TEST_NO_FILE.with(|no| no.set(true));
        let refused = crate::answer::relaxed(|| zeros.pieces(0x3000, 0x2000));
        vm::TEST_NO_FILE.with(|no| no.set(false));
        assert!(matches!(refused, Err(Refusal::NoFile(_))), "{refused:?}");
        assert!(zeros.0.lock().is_empty(), "nothing kept");
        assert_eq!(zeros.pieces(0x3000, 0x2000).unwrap().len(), 1);
    }

    /// Every request of an import's source over the same pages of the block of zeros answers the
    /// same pieces of the same files, as the runtime's contract asks
    /// (`rt::HostMemory::with_dma_buf`): a write through the import's dma-buf there is what the
    /// dma-buf of a texture's pages inside it reads. The write never reaches the block, nor
    /// another import's dma-buf of the same pages
    #[test]
    fn a_sources_requests_over_zero_pages_answer_the_same_files() {
        if !udmabuf() {
            return;
        }
        let page = vm::page_size();
        let at = super::super::import::allocate_zeros(4 * page).unwrap();
        let first = at.addr().to_u64();
        let zeros = Zeros::default();
        let whole = of_pages(first, 4 * page, page, &zeros).unwrap();
        let texture = of_pages(first + page, 2 * page, 2 * page, &zeros).unwrap();
        let other = of_pages(first, 4 * page, page, &Zeros::default()).unwrap();
        assert_eq!((whole.len(), texture.len(), other.len()), (1, 1, 1));
        write_through(&whole[0], 2 * page, 0x5a);
        assert_eq!(read_through(&texture[0], page), 0x5a, "the same pages");
        assert_eq!(read_through(&other[0], 2 * page), 0, "another import's");
        // SAFETY: the block's page, mapped readable, which nothing writes now.
        let block = unsafe { vm::ptr(first + 2 * page).cast::<u8>().read_volatile() };
        assert_eq!(block, 0);
        let vm = super::super::LinuxVm::new();
        assert!(vm.deallocate(at, 4 * page).is_success());
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
        let chunks = of_pages(at + page, 2 * page, page, &Zeros::default()).unwrap();
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
            of_pages(private, page, page, &Zeros::default()),
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
        let chunks = of_pages(view.addr().to_u64(), count * page, page, &Zeros::default()).unwrap();
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
}
