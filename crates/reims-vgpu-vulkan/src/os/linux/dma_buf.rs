//! Dma-bufs of the pages behind a range the backing table resolves ([`crate::os::files`]), for a
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
//! dma-buf within the limits the host has now ([`within_limits`]), cut only where the runtime can bind
//! the next ([`chunks`]); the runtime binds them in order behind one buffer.
//!
//! **Costs** (measured on the same host, 2026-10-12): a udmabuf pins every page it takes,
//! allocating those never touched (64 MiB of untouched memory in 18 ms), and keeps showing a page
//! the file later drops. Each import the device holds adds about 0.55 µs to every submission
//! (RADV passes each in a list on every submit): 1,024 live imports cost 642 µs a submit, 4,096
//! about 2.5 ms. So the fewest chunks are made, and an import goes with its buffer.
//!
//! **Several pieces.** RADV binds each chunk of a sparse buffer as a GPU VA operation of its own,
//! and amdgpu updates its page tables for each, at the bind and again at the release, besides
//! the import's place in every submission's list ("Costs", above): another reason the fewest
//! chunks are made.
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

use std::fmt;
use std::io;
use std::os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd};
#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use crate::core::num::ToU64 as _;
use crate::core::platform::os::vm::VmProt;
use crate::core::sync::Mutex;

use super::vm;
use crate::os::files;

/// `UDMABUF_FLAGS_CLOEXEC`.
const FLAGS_CLOEXEC: u32 = 0x01;

/// `UDMABUF_CREATE_LIST`, `_IOW('u', 0x43, struct udmabuf_create_list)`: the header's 8 bytes.
const CREATE_LIST: u64 = (1 << 30) | (8 << 16) | ((b'u' as u64) << 8) | 0x43;

/// udmabuf's defaults, taken where the system shows no parameters.
const DEFAULT_LIST_LIMIT: u64 = 1024;
const DEFAULT_SIZE_LIMIT_MB: u64 = 64;

/// A dma-buf of pages ([`of_pages`]): `length` bytes of `file`.
#[derive(Debug)]
pub(crate) struct DmaBufChunk {
    pub(crate) file: Arc<OwnedFd>,
    pub(crate) length: u64,
}

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

/// udmabuf's limits as the host has them, its defaults where it shows none: read from the module's
/// parameters at most once a second ([`LIMITS`]), so they are the host's within a second of now,
/// and read again at a refusal they may have caused ([`within_limits`]). Each read of the two
/// parameters cost a texture over a no-copy buffer's pages about 9% of its making's CPU on a
/// Radeon 780M (**measured** 2026-11-15, Geekbench 7.1.0's compute pass, 606 such textures). The
/// tests read them so; the library reads them through [`within_limits`].
#[cfg(test)]
fn limits() -> Limits {
    LIMITS.at(mono_now())
}

/// How long a read of udmabuf's limits stands, in nanoseconds (`limits`).
const LIMITS_STAND_NS: u64 = 1_000_000_000;

/// udmabuf's limits as last read from the host (`limits`).
static LIMITS: KeptLimits = KeptLimits::new(host_limits);

/// Limits read by `read` and kept for [`LIMITS_STAND_NS`] ([`within_limits`]).
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

/// `attempt` within udmabuf's limits, read through [`LIMITS`]; where they refuse it ([`Refusal::Limit`]), or
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

/// Whether one dma-buf takes the pages `pieces` show, in order, a view's: refused, with udmabuf's
/// limits named, where they are more runs of file pages, or more bytes, than one takes. A run of
/// file pages that goes on from one piece into the next counts once, as [`add_runs`] makes it
/// one. Asked as the view is made (`super::vm`'s `map_view`).
pub(crate) fn fits_view(pieces: &[files::Piece]) -> Result<(), Refusal> {
    let mut runs: u64 = 0;
    let mut bytes: u64 = 0;
    // The last piece's file and the offset past its end.
    let mut last: Option<(&Arc<OwnedFd>, u64)> = None;
    for piece in pieces {
        let follows = matches!(
            last,
            Some((fd, end)) if Arc::ptr_eq(fd, &piece.backing.file) && end == piece.backing.offset
        );
        if !follows {
            runs += 1;
        }
        last = Some((&piece.backing.file, piece.backing.offset + piece.length));
        bytes = bytes.saturating_add(piece.length);
    }
    fits_pieces(runs, bytes)
}

/// Whether one dma-buf takes `pieces` pieces of `bytes` bytes in all, within udmabuf's limits as
/// the host has them now: refused, with the limits named, past them.
fn fits_pieces(pieces: u64, bytes: u64) -> Result<(), Refusal> {
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

/// [`of_pages`], and the runs of file pages the chunks hold, the pad's among them.
fn of_pages_counted(
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
                length: chunk.iter().map(|r| r.length).sum(),
            })
        });
        Ok((made.collect::<Result<_, _>>()?, runs.len().to_u64()))
    })
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

    /// udmabuf's limits are read from the module's parameters where the host shows them, and
    /// are its defaults where it does not
    #[test]
    fn the_limits_are_the_hosts() {
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

    /// A range of the guest's RAM for a test: a memfd of `pages` pages mapped shared, each page's
    /// first byte its number (modulo 256), and registered as the guest's RAM while it lives.
    struct TestRam {
        at: u64,
        length: u64,
        file: Arc<OwnedFd>,
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
                unsafe { vm::ptr(at + i * page).cast::<u8>().write_volatile(i as u8) };
            }
            let backing = files::Backing {
                file: Arc::clone(&file),
                offset: 0,
            };
            let registration = files::Registration::guest_ram(at, length, backing);
            TestRam {
                at,
                length,
                file,
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

    /// A view of [`TestRam`] pages for a test: its runs, each a RAM, its first page and its
    /// pages, mapped shared one after another from the RAM's files and registered as showing them
    /// while it lives, or recorded as one view's runs, as the library's views of the guest's pages
    /// are.
    struct TestView {
        at: u64,
        length: u64,
        registrations: Vec<files::Registration>,
        recorded: bool,
    }

    impl TestView {
        fn new(runs: &[(&TestRam, u64, u64)]) -> TestView {
            TestView::made(runs, false)
        }

        fn made(runs: &[(&TestRam, u64, u64)], recorded: bool) -> TestView {
            let page = vm::page_size();
            let length: u64 = runs.iter().map(|&(_, _, n)| n * page).sum();
            // SAFETY: a fresh reservation, which the runs are mapped over and the view unmaps.
            let start = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    length.to_usize(),
                    libc::PROT_NONE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            assert_ne!(start, libc::MAP_FAILED);
            let at = start.addr().to_u64();
            let mut registrations = Vec::new();
            let mut pieces = Vec::new();
            let mut to = at;
            for &(ram, first, n) in runs {
                // SAFETY: a shared mapping of the RAM's file over the view's own reservation.
                let mapped = unsafe {
                    libc::mmap(
                        vm::ptr(to),
                        (n * page).to_usize(),
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_SHARED | libc::MAP_FIXED,
                        ram.file.as_raw_fd(),
                        libc::off_t::try_from(first * page).unwrap(),
                    )
                };
                assert_eq!(mapped.addr().to_u64(), to);
                let backing = files::Backing {
                    file: Arc::clone(&ram.file),
                    offset: first * page,
                };
                match recorded {
                    true => pieces.push(files::Piece {
                        at: to - at,
                        length: n * page,
                        backing,
                        prot: VmProt::READ | VmProt::WRITE,
                    }),
                    false => registrations.push(files::Registration::new(to, n * page, backing)),
                }
                to += n * page;
            }
            if recorded {
                files::record_pieces(at, pieces);
            }
            TestView {
                at,
                length,
                registrations,
                recorded,
            }
        }
    }

    impl Drop for TestView {
        fn drop(&mut self) {
            // Its registrations, or its runs, go before its pages are unmapped.
            self.registrations.clear();
            if self.recorded {
                files::forget(self.at, self.length);
            }
            // SAFETY: the mapping `new` made.
            unsafe { libc::munmap(vm::ptr(self.at), self.length.to_usize()) };
        }
    }

    /// A buffer's or heap's pages in several runs, of one file or of several, take one dma-buf of
    /// their own, one list of their runs (`UDMABUF_CREATE_LIST` takes runs of several memfds) at
    /// offset 0, whatever the granule a split would take, which shows the pages in their order:
    /// the runtime binds a plain buffer to it, with no sparse binding. So does one run across a
    /// region's end (module docs, "Regions")
    #[test]
    fn pages_in_several_runs_take_one_dma_buf_of_their_own() {
        if !udmabuf() {
            return;
        }
        let page = vm::page_size();
        let ram = TestRam::new(32);
        let other = TestRam::new(8);
        let view = TestView::new(&[(&ram, 16, 16), (&ram, 0, 16), (&other, 3, 2)]);
        let length = 34 * page;
        let (chunks, runs) = of_pages_counted(view.at, length, 16 * page).unwrap();
        assert_eq!(
            (chunks.len(), runs),
            (1, 3),
            "one dma-buf of the three runs"
        );
        assert_eq!(chunks[0].length, length);
        assert_eq!(
            Arc::strong_count(&chunks[0].file),
            1,
            "a dma-buf of its own"
        );
        let shown: Vec<u8> = [0, 15, 16, 31, 32, 33]
            .iter()
            .map(|&i| read_through(&chunks[0], i * page))
            .collect();
        assert_eq!(
            shown,
            [16, 31, 0, 15, 3, 4],
            "the pages in the view's order"
        );
        drop(view);
        let (across, runs) = of_pages_counted(ram.at + 12 * page, 8 * page, 4 * page).unwrap();
        assert_eq!(
            (across.len(), runs),
            (1, 1),
            "one run across a region's end"
        );
        assert_eq!(read_through(&across[0], 7 * page), 19);
    }

    /// A buffer's or heap's pages in more runs than one dma-buf takes (udmabuf's `list_limit` as
    /// the host has it) take chunks, the fewest within the limits, each a dma-buf of its own: the
    /// only requests the runtime binds in several chunks. Where the limits take more runs than the
    /// test makes, it says so and checks nothing
    #[test]
    fn pages_past_the_list_limit_take_chunks() {
        if !udmabuf() {
            return;
        }
        let page = vm::page_size();
        let limits = limits();
        let n = limits.pieces + 1;
        if n > 4096 || n * page > limits.bytes {
            eprintln!(
                "skipped: udmabuf takes {} runs of {:#x} bytes",
                limits.pieces, limits.bytes
            );
            return;
        }
        let ram = TestRam::new(2 * n);
        let runs: Vec<(&TestRam, u64, u64)> = (0..n).map(|i| (&ram, 2 * i, 1)).collect();
        let view = TestView::new(&runs);
        let (chunks, made) = of_pages_counted(view.at, n * page, page).unwrap();
        assert_eq!((chunks.len(), made), (2, n), "two chunks");
        assert_eq!(
            (chunks[0].length, chunks[1].length),
            (limits.pieces * page, page)
        );
        assert_eq!(read_through(&chunks[1], 0), (2 * (n - 1)) as u8);
    }
}
