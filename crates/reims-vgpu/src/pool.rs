//! The guest's Shared pool: a range of guest physical addresses the host reserved with no RAM
//! under it ([`PoolConfig`]), which the device fills with the block source's host-visible blocks
//! as it grows, one piece a block, and gives back as the guest empties them.
//!
//! # Growth
//!
//! An ask for more pool ([`Pool::grow`]: the host's `rvg_graphics_device_pool_grow`, and the
//! headroom rule below) wakes the pool's taker, `reims-vgpu.graphics-pool-ahead`, which takes a
//! Shared block from the source ([`BlockSource::take_block`], which waits only when no block is
//! ready) for each such ask in turn, oldest first. A take that fits no piece takes a block of its
//! own length on its context's worker (`takes`), so only that take waits for the source's
//! allocation. The pool's thread, `reims-vgpu.graphics-pool`, places each block taken, the takes'
//! before the taker's, at the lowest free address of the range that holds it, 2 MiB aligned, so
//! pieces go out in ascending addresses, each its block's length, the source's rounding of the
//! length asked, which need not be whole 2 MiB ([`PIECE_ALIGN`]), carves a take's block for it at
//! its piece's first byte, and asks the host to map it there ([`PoolHost::add`]) with no lock of
//! the pool's held: it waits for no source, so neither another context's take nor the headroom
//! waits behind a take's allocation, nor a take behind the headroom's. The host maps it (QEMU: a
//! RAM region of its own over the block, added on its main loop) and says so ([`Pool::added`], the
//! C API's "range added"); only then does [`Pool::find`] find it. A block the source hands out
//! again after the pool gave it back is placed and mapped anew, as any block is: its piece went
//! before it went back.
//!
//! # Headroom and giving blocks back
//!
//! A new piece is empty: nothing of the guest's lies in it. The guest's processes take blocks of
//! the pool and give them back (`takes`): a piece is used from its first block's take and empty
//! again once its last block went back. The pool keeps [`HEADROOM`] bytes of empty pieces as
//! headroom, each [`BLOCK_LEN`] long, metal2vulkan-runtime's unit: where the guest's use of a piece
//! leaves the empty pieces, with the asks for more pool no take waits on, fewer bytes than that,
//! the pool asks for a block more for each piece short ([`State::keep_headroom`]). It keeps them
//! from the moment the guest's driver takes the pool (`PoolLedger::keep_headroom`), before any
//! take, so the guest's first takes find pieces made and mapped. The headroom is the lowest empty
//! pieces the guest may use, each one whose empty pieces below it hold fewer than [`HEADROOM`]
//! bytes ([`State::empties`]). An empty piece above them that stays empty for [`RELEASE_DELAY`]
//! goes back to the source ([`BlockSource::release_block`]), as the release below says, and the
//! source keeps the block ready or frees it. The delay rides out churn, where a guest empties and
//! fills a piece again within seconds. Where the trace is on, each piece that goes back is an
//! event of it, `poolback`, with its length and how long it stood empty.
//!
//! # Release
//!
//! The pool gives a block back to the source only once the guest's mapping of its piece is gone.
//! It takes the piece out of the lookups, and once every hold on it ([`PieceHold`]) has gone, asks
//! the host to unmap it ([`PoolHost::remove`]), and gives the block back once the host says it is
//! out of the guest's address space and no host thread of its own touches it ([`Pool::removed`]).
//! The pool's thread takes each step as it comes due, waiting for none of them, so a piece on its
//! way back holds up no other piece's place or add. So the source may free a block the moment it
//! gets it back: no guest access, no thread of the host's and no thread of the library's touches
//! it.
//!
//! A piece is held, too, while an import of its bytes lives ([`PieceKeep`]): a heap over a
//! context's block that the platform made over rt's import of the block's mapping, rather than in
//! the block itself, whose release, which may run well after the heap went, holds the keep. So the
//! block goes back, and the source may free it, only once every import of its bytes is freed, as
//! rt's `import_host_memory` asks of the memory it imports. The last keep's drop rings the pool's
//! bell ([`Bell`]), which wakes the pool's thread, so the block goes back as that keep goes:
//! nothing looks for it on a timer, and nothing the guest waits for waits for it.
//!
//! At the device's release the host has removed every piece it added (QEMU does so as it tears
//! the device down, before the release): from then on ([`Pool::close`]) the pool waits only for
//! the holds and the keeps and asks the host nothing, and it gives every block back before it lets
//! go of the source. It waits for them at most the release's 5 s ([`crate::LET_GO_BOUND`]): past
//! them, where a hold or a keep is left (an import whose GPU work never completes, on a GPU that
//! hung), it gives back only the blocks no hold or keep is on and forgets the source, so a block
//! still imported never goes back, by the pool or by the source's drop, and the source, with the
//! device it holds, stays to the process's exit.
//!
//! # The block source
//!
//! [`BlockSource`] is metal2vulkan-runtime's block calls as its API names them
//! (`Device::take_block`, `release_block`, [`BlockInfo`]). On the rt platform the source is the
//! runtime's blocks ([`rt_blocks::RtBlocks`], the platform's `ApiPlatform::block_source`), each
//! known to its device while handed out. A platform that gives none (the testkit's) takes
//! [`HostBlocks`], blocks of the process's own memory, which hands blocks out, keeps them ready and
//! frees them as the runtime does.
//!
//! The host maps a block into the guest whole host pages at a time: a block whose mapping does
//! not start at a host page, or whose length is no whole number of them, goes back to the source
//! unmapped, counted ([`UNALIGNED_CODE`]), and no piece comes of the ask.

// Guest addresses flow through here: arithmetic is spelt out.
#![cfg_attr(not(test), deny(clippy::arithmetic_side_effects))]

mod addresses;
mod host_blocks;
mod private_blocks;
mod rt_blocks;
mod takes;
mod write_combined;

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use parking_lot::{Condvar, Mutex, MutexGuard};
use reims_vgpu_vulkan::answer::count_kind;
use reims_vgpu_vulkan::core::num::ToU64 as _;
use reims_vgpu_vulkan::core::platform::os::clock::MonoTime;
use reims_vgpu_vulkan::core::platform::os::vm::HostVm as _;
use reims_vgpu_vulkan::core::trace;
use reims_vgpu_vulkan::os::OsVm;

pub(crate) use addresses::HostAddresses;
#[cfg(test)]
pub(crate) use addresses::{
    HOST_BASE, HOST_END, IMPORT_BASE, IMPORT_END, PRIVATE_BASE, PRIVATE_END,
};
pub(crate) use host_blocks::HostBlocks;
pub(crate) use private_blocks::{
    PrivateBlocks, PrivateGive, TakeAnswer as PrivateTakeAnswer, TakeError as PrivateTakeError,
};
pub(crate) use rt_blocks::RtBlocks;
pub(crate) use takes::{Give, PoolLedger, TAKE_MAX};
#[cfg(test)]
use takes::{TAKE_ALIGN, TakeAnswer, Taken};
pub(crate) use write_combined::WriteCombined;

use crate::Error;
use crate::trail;

/// The code of a block the source handed out whose mapping is not whole host pages from a host
/// page, which goes back unmapped (the module doc's "The block source").
pub const UNALIGNED_CODE: &str = "rt-pool-block-unaligned";

/// The code of the bytes the take workers zeroed through their pieces' mappings (`takes`),
/// counted by the byte.
pub const TAKE_ZEROED_BYTES_CODE: &str = "pool-take-zeroed-bytes";

/// The code of a carve's bytes the take workers did not zero, as the pool's zeroer zeroed them
/// ahead of the take and no carve held them since (`takes`, "Zeros ahead"). Counted by the byte.
pub const TAKE_ZERO_PREZEROED_BYTES_CODE: &str = "pool-take-zero-skipped-prezeroed-bytes";

/// The code of the bytes the pool's zeroer zeroed ahead of their takes (`takes`, "Zeros ahead"),
/// counted by the byte.
pub const ZEROED_AHEAD_BYTES_CODE: &str = "pool-zeroed-ahead-bytes";

/// The code of the nanoseconds the pool's zeroer spent writing its chunks' zeros, on its own
/// thread with no lock held (`takes`, "Zeros ahead"), counted in all.
pub const ZERO_WRITE_NS_CODE: &str = "pool-zero-write-ns";

/// The code of the longest write of one chunk's zeros by the pool's zeroer (`takes`, "Zeros
/// ahead"), in nanoseconds: its count rises to it.
pub const ZERO_WRITE_MAX_NS_CODE: &str = "pool-zero-write-max-ns";

/// The code of the longest the pool's zeroer held the pool's lock for one chunk (`takes`, "Zeros
/// ahead"), in nanoseconds: its count rises to it.
pub const ZERO_HOLD_MAX_NS_CODE: &str = "pool-zero-hold-max-ns";

/// The codes of the longest each part of the zeroer's hold of the pool's lock for one chunk took
/// so far (`takes`, "Zeros ahead"), in nanoseconds, their counts rising to it: the chunk's pick
/// and claim (`State::next_zero`), and its mark as reading zero (`State::zeroed`) with the choice
/// of whether to wake the pool's waiters. Its write lies between them, and the wake after the
/// mark, each with the lock let go of.
pub const ZERO_HOLD_PART_MAX_NS_CODES: [&str; 2] =
    ["pool-zero-hold-pick-max-ns", "pool-zero-hold-mark-max-ns"];

/// The code of the zeroer's holds of the pool's lock for one chunk longer than 50 µs (`takes`,
/// [`ZERO_HOLD_LONG`]), counted by the hold.
pub const ZERO_HOLDS_LONG_CODE: &str = "pool-zero-holds-over-50us";

/// The codes of the nanoseconds the holds [`ZERO_HOLDS_LONG_CODE`] counts spent in each part, in
/// the order of [`ZERO_HOLD_PART_MAX_NS_CODES`].
pub const ZERO_LONG_HOLD_PART_NS_CODES: [&str; 2] =
    ["pool-zero-long-hold-pick-ns", "pool-zero-long-hold-mark-ns"];

/// A zeroer's hold longer than this is counted with its parts ([`ZERO_HOLDS_LONG_CODE`]): the
/// longest the walks of 11-19 saw, with 16 KiB written inside the hold, was 64 µs.
pub const ZERO_HOLD_LONG: std::time::Duration = std::time::Duration::from_micros(50);

/// The code of the takes that waited for the zeroer's claimed chunk, which their first fit held
/// bytes of, to be written (`takes`, "Zeros ahead"), counted by the take.
pub const TAKE_CLAIM_WAITS_CODE: &str = "pool-take-claim-waits";

/// The code of the nanoseconds the takes [`TAKE_CLAIM_WAITS_CODE`] counts waited, in all.
pub const TAKE_CLAIM_WAIT_NS_CODE: &str = "pool-take-claim-wait-ns";

/// The code of the longest wait [`TAKE_CLAIM_WAITS_CODE`] counts, in nanoseconds.
pub const TAKE_CLAIM_WAIT_MAX_NS_CODE: &str = "pool-take-claim-wait-max-ns";

/// The code of the takes answered, whose waits [`TAKE_WAIT_CODES`] count (`takes`).
pub const TAKE_WAITS_CODE: &str = "pool-take-waits";

/// The codes of the answered takes' waits, from each take's carry-out to its answer (`takes`):
/// the whole wait, then its parts behind the context's earlier takes, for a piece of the take's
/// own, and the take worker's zeroing, which for a piece of its own runs while the host maps the
/// piece and is left out of the wait for it; then, of the wait for a piece, the source's takes of
/// its blocks ([`BlockSource::take_block`]), the rest being their place and the host's map; each
/// the nanoseconds in all, then the longest.
pub const TAKE_WAIT_CODES: [(&str, &str); 5] = [
    ("pool-take-wait-ns", "pool-take-wait-max-ns"),
    ("pool-take-wait-queue-ns", "pool-take-wait-queue-max-ns"),
    ("pool-take-wait-growth-ns", "pool-take-wait-growth-max-ns"),
    ("pool-take-wait-zero-ns", "pool-take-wait-zero-max-ns"),
    (
        "pool-take-wait-growth-source-ns",
        "pool-take-wait-growth-source-max-ns",
    ),
];

/// The length of each piece the pool asks for as headroom (the module doc's "Headroom and giving
/// blocks back"), and of every block [`HostBlocks`] hands out for an ask no longer:
/// metal2vulkan-runtime's unit, the length of the Shared allocations its block maker makes ahead of
/// need ([`MemoryClass::block_len`], 16 MiB), read from rt.
///
/// rt hands out a block of the length asked, rounded up to 64 KiB. A take of up to a unit is the
/// first free range of rt's allocations that fits it: a piece, a whole unit, lands in a unit the
/// maker made ahead where one is wholly free, with no Vulkan call on the pool's thread, else in a
/// unit's allocation the take makes; given back, it is a unit's range again, which rt keeps as its
/// reserve or frees as one unit's call. A longer take makes an allocation of its own length on
/// the pool's thread, which serves later takes of that length alone and which rt frees as one
/// call of that length once it has been idle 250 ms. A driver may hold the device's submissions
/// for the whole of an allocation's or a free's call, longer the longer it is (rt's
/// `ALLOCATION_LEN`: on an RTX 3090 a 16 MiB Shared allocation held one for 3.3 ms at the median,
/// one of 256 MiB for 51 ms, **measured**). So a piece is one unit, and a take longer than a piece
/// has a piece of its own length, rounded up to 64 KiB, its allocation made in its growth.
///
/// On a Radeon 780M a growth of 256 MiB (rt's take, the host's add and its populate) took at
/// most 10.5, 2.3 and 76.0 ms, and a boot with a desktop, a browser, video and two 3D benchmarks
/// took nothing longer than 16 MiB and carved at most 73 MiB at once outside the benchmarks
/// (**measured**, 11-19).
///
/// [`MemoryClass::block_len`]: metal2vulkan_runtime::MemoryClass::block_len
pub const BLOCK_LEN: u64 = metal2vulkan_runtime::MemoryClass::Shared.block_len();

/// The bytes of empty pieces the pool keeps ahead of the guest's takes, its headroom (the module
/// doc's "Headroom and giving blocks back"): 64 MiB, in pieces of [`BLOCK_LEN`]. On an RTX 3090
/// and a Radeon 780M, walks of the desktop, browser scroll, WebGL and 4K video, a GL scene and
/// two 3D benchmarks raised the guest's carved memory by at most 28 MiB net within a window, and
/// no take waited on a growth, with one 64 MiB piece kept (**measured**, 11-19).
pub const HEADROOM: u64 = 64 << 20;

/// The alignment of a piece's guest physical address and of a pool's range: 2 MiB, a large page of
/// the guest's and of the hypervisors' second-level tables. A piece's length need not be a multiple
/// of it: it is its block's, the length asked rounded up to the source's granule (rt's 64 KiB, or
/// the GPU's `bufferImageGranularity` where that is larger; [`HostBlocks`]' unit, or 2 MiB past
/// one), and the next piece starts at the first multiple of it past the piece ([`first_fit`]).
pub const PIECE_ALIGN: u64 = 2 << 20;

/// How long a piece stays empty, with a lower piece empty too, before its block goes back to the
/// source (the module doc's "Headroom and giving blocks back").
pub const RELEASE_DELAY: Duration = Duration::from_secs(5);

/// A block's storage mode as the guest's buffers in it see it (metal2vulkan-runtime's
/// `MemoryClass`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(
    dead_code,
    reason = "metal2vulkan-runtime's shape, kept whole so only the source changes when it lands"
)]
pub enum MemoryClass {
    /// Device-local, never mapped, no host pointer.
    Private,
    /// Host-visible, host-coherent and host-cached: mapped into the guest.
    Shared,
    /// Host-visible and host-coherent, and not host-cached where the GPU has such memory
    /// (metal2vulkan-runtime's `SharedWriteCombined`): mapped into the guest where the host asks
    /// for write-combined blocks ([`WriteCombined`]).
    SharedWriteCombined,
}

/// A block's number, never reused while its source lives (metal2vulkan-runtime's `BlockId`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u32);

impl BlockId {
    /// The number.
    #[allow(
        dead_code,
        reason = "metal2vulkan-runtime's shape, kept whole so only the source changes when it lands"
    )]
    pub fn get(self) -> u32 {
        self.0
    }
}

/// A block a source handed out (metal2vulkan-runtime's `BlockInfo`): plain data.
#[derive(Clone, Copy, Debug)]
#[allow(
    dead_code,
    reason = "metal2vulkan-runtime's shape, kept whole so only the source changes when it lands"
)]
pub struct BlockInfo {
    /// Its number.
    pub id: BlockId,
    /// Its class.
    pub class: MemoryClass,
    /// Its length: at least the length asked ([`HostBlocks`]' [`BLOCK_LEN`] for a shorter ask).
    pub len: u64,
    /// The GPU address of its first byte; 0 from a source with no GPU ([`HostBlocks`]).
    pub gpu_address: u64,
    /// Its host mapping, for a Shared block; `None` for a Private one.
    pub contents: Option<NonNull<u8>>,
}

// SAFETY: plain data; `contents` is an address, and dereferencing it is the holder's unsafe.
unsafe impl Send for BlockInfo {}
// SAFETY: as Send.
unsafe impl Sync for BlockInfo {}

/// A source refused a block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockError {
    /// A length past what the source allocates in one block.
    Length,
    /// The host or the device has no memory for it.
    OutOfMemory,
    /// A block the caller does not hold.
    NotHeld,
}

/// Where the pool's blocks come from (metal2vulkan-runtime's `Device::take_block` and
/// `release_block`).
pub trait BlockSource: Send + Sync {
    /// A block of `class` at least `len` long ([`BlockInfo::len`]). Waits only when no block is
    /// ready.
    fn take_block(&self, class: MemoryClass, len: u64) -> Result<BlockInfo, BlockError>;
    /// Gives a block back, which no mapping of the guest's and no thread of the caller's touches
    /// any more: it goes back to the ready set, with its id, address and host mapping, or is
    /// freed.
    fn release_block(&self, block: BlockId) -> Result<(), BlockError>;
}

/// The guest physical range the host reserved for the pool, with no RAM under it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolConfig {
    /// Its first guest physical address, a multiple of [`PIECE_ALIGN`].
    pub physical_address: u64,
    /// Its length, a multiple of [`PIECE_ALIGN`].
    pub physical_length: u64,
}

impl PoolConfig {
    /// Whether a pool takes the range: not empty, both ends multiples of [`PIECE_ALIGN`], and not
    /// past the address space.
    pub fn is_valid(self) -> bool {
        self.physical_length != 0
            && self.physical_address.is_multiple_of(PIECE_ALIGN)
            && self.physical_length.is_multiple_of(PIECE_ALIGN)
            && self
                .physical_address
                .checked_add(self.physical_length)
                .is_some()
    }

    /// Whether the range overlaps `length` bytes from `address`.
    pub fn overlaps(self, address: u64, length: u64) -> bool {
        let end = self.physical_address.saturating_add(self.physical_length);
        address < end && self.physical_address < address.saturating_add(length)
    }

    /// The range a pool takes of `self`, the guest physical addresses the host can give it, on a
    /// GPU whose memory heaps hold `heaps` bytes, every heap summed
    /// (`GpuDevice::memory_heaps_total`): from `self`'s start, `heaps` rounded up to
    /// [`PIECE_ALIGN`], at most `self`'s length; `None` for heaps of 0, where no block of the
    /// pool's could come from the GPU. The range is guest physical addresses, which RAM fills only
    /// as the pool takes blocks: it bounds the pool by what the GPU's memory can ever hold, and
    /// the driver, which refuses a block it cannot give, bounds the RAM.
    pub fn sized(self, heaps: u64) -> Option<PoolConfig> {
        if heaps == 0 {
            return None;
        }
        let want = heaps
            .checked_next_multiple_of(PIECE_ALIGN)
            .unwrap_or(u64::MAX);
        Some(PoolConfig {
            physical_address: self.physical_address,
            physical_length: want.min(self.physical_length),
        })
    }
}

/// `(physical_address, host_address, length)`: map `length` bytes of host memory at
/// `host_address` into the guest at `physical_address`, then say so ([`Pool::added`]). Called on
/// the pool's thread with no lock of the device's held; it must not wait for a lock the host holds
/// while it calls the device.
pub type PoolAddFn = Arc<dyn Fn(u64, *mut u8, u64) + Send + Sync>;

/// `(physical_address, length)`: unmap the piece at `physical_address`, then say so once no
/// thread of the host's touches it ([`Pool::removed`]). Called on the pool's thread before it
/// gives the piece's block back, with no lock of the device's held; it must not wait for a lock
/// the host holds while it calls the device.
pub type PoolRemoveFn = Arc<dyn Fn(u64, u64) + Send + Sync>;

/// The platform's monotonic clock, which times an empty piece's delay.
pub(crate) type PoolNow = Box<dyn Fn() -> MonoTime + Send + Sync>;

/// The host's side of the pool.
#[derive(Clone)]
pub struct PoolHost {
    /// Maps a piece.
    pub add: PoolAddFn,
    /// Unmaps a piece.
    pub remove: PoolRemoveFn,
}

/// Where a piece's mapping is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mapped {
    /// Asked of the host, not yet added.
    Adding,
    /// In the guest's address space.
    Live,
    /// Being unmapped before its block goes back, out of the lookups; `asked` once the host was
    /// asked to remove it.
    Removing { asked: bool },
    /// Out of the guest's address space, and no thread of the host's touches it.
    Removed,
}

/// A piece on its way back to the source (the module doc's "Release").
#[derive(Clone, Copy, Debug)]
struct Going {
    /// Since when.
    since: MonoTime,
    /// Whether the pool's thread warned that it still waits.
    warned: bool,
}

/// One block in the range, which the pool holds until its piece is out of the guest.
#[derive(Debug)]
struct Piece {
    block: BlockInfo,
    mapped: Mapped,
    /// Since when nothing of the guest's lies in it; `None` while something does.
    empty_since: Option<MonoTime>,
    /// Holds on it ([`PieceHold`]).
    holds: u64,
    /// The keeps of it, each a clone ([`PieceKeep`]): it is kept while this is not the only one.
    kept: Arc<()>,
    /// The contexts' blocks in it (`takes`).
    carves: u64,
    /// The bytes of it that read zero, as the zeroer zeroed them ahead of their takes, by offset
    /// in the block; none of them is carved (`takes`, "Zeros ahead").
    zeros: takes::Ranges,
    /// Whether some of its bytes may be neither carved nor in `zeros`: set at its place and at
    /// each carve's free, and cleared once the zeroer finds none (`takes`, "Zeros ahead").
    zero_pending: bool,
    /// Since when it is on its way back to the source, where it is: only a piece the pool gives
    /// back, never one the host removed of its own.
    going: Option<Going>,
}

impl Piece {
    /// Whether the guest may put something in it: not being unmapped.
    fn usable(&self) -> bool {
        matches!(self.mapped, Mapped::Adding | Mapped::Live)
    }

    /// Whether an import of its bytes keeps it ([`PieceKeep`]).
    fn kept(&self) -> bool {
        Arc::strong_count(&self.kept) > 1
    }

    /// Whether a hold or a keep is on it, so its block does not go back.
    fn held(&self) -> bool {
        self.holds > 0 || self.kept()
    }
}

/// What wakes the pool's thread and the release's end ([`Shared::wake`]): a count of the rings,
/// under a lock of its own that is held for nothing else, so that a keep's drop takes no lock of
/// the pool's ([`PieceKeep`]). A waiter reads the count before it looks at what it waits for, and
/// sleeps only while the count is still the one it read, so a ring between its look and its
/// sleep is not lost.
#[derive(Default)]
struct Bell {
    rung: Mutex<u64>,
    cv: Condvar,
    /// How many times a waiter went to sleep: for the tests, which wait for the release's end to
    /// sleep on a keep.
    #[cfg(test)]
    slept: std::sync::atomic::AtomicU64,
}

impl Bell {
    /// The rings so far.
    fn rung(&self) -> u64 {
        *self.rung.lock()
    }

    /// One ring more, waking every waiter.
    fn ring(&self) {
        let mut rung = self.rung.lock();
        *rung = rung.wrapping_add(1);
        drop(rung);
        self.cv.notify_all();
    }

    /// Sleeps while no ring came past `seen`, at most `most` where given (a waiter looks again
    /// after any wake).
    fn wait(&self, seen: u64, most: Option<Duration>) {
        let mut rung = self.rung.lock();
        if *rung != seen {
            return;
        }
        #[cfg(test)]
        self.slept.fetch_add(1, Ordering::SeqCst);
        match most {
            Some(most) => {
                let _ = self.cv.wait_for(&mut rung, most);
            }
            None => self.cv.wait(&mut rung),
        }
    }

    /// How many times a waiter went to sleep.
    #[cfg(test)]
    fn slept(&self) -> u64 {
        self.slept.load(Ordering::SeqCst)
    }
}

/// A keep of a piece's block for an import of its bytes, which the import's release holds
/// ([`PoolLedger::over_held`]): while it lives, the block does not go back to the source, which
/// may free it. The release may run on any thread, rt's block-maker thread among them, where it
/// makes no call into the runtime and drops none of its handles; and a thread may hold the
/// pool's lock across a call into rt (`over_held`'s), which can wait for that thread. So the
/// keep's drop takes no lock of the pool's and holds nothing whose drop could be the pool's last,
/// whose clock holds the platform: it lets go of its count of the piece and rings the pool's bell
/// ([`Bell`]), which wakes the pool's thread to give the block back.
pub(crate) struct PieceKeep {
    kept: Option<Arc<()>>,
    bell: Arc<Bell>,
}

impl Drop for PieceKeep {
    fn drop(&mut self) {
        drop(self.kept.take());
        self.bell.ring();
    }
}

impl std::fmt::Debug for PieceKeep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PieceKeep").finish_non_exhaustive()
    }
}

/// What the pool's lock holds.
#[derive(Default)]
struct State {
    /// By guest physical address.
    pieces: BTreeMap<u64, Piece>,
    /// Each piece's address, by its block.
    by_block: HashMap<BlockId, u64>,
    /// Asks for more pool no take waits on, the headroom's and the host's, oldest first: the
    /// pool's taker takes a block for each ([`Shared::take_ahead`]).
    asks: VecDeque<Ask>,
    /// The length of the ask the pool's taker serves, from its take off [`Self::asks`] until its
    /// block is in [`Self::ahead`] or no block came of it; 0 while it serves none.
    serving: u64,
    /// The blocks the pool's taker took, oldest first, for the pool's thread to place after
    /// [`Self::own`]'s.
    ahead: VecDeque<BlockInfo>,
    /// The blocks the take workers took for takes that fit no piece, oldest first, for the pool's
    /// thread to place before any of [`Self::ahead`]'s (`takes`).
    own: VecDeque<Own>,
    /// The takes waiting for their own blocks' places, by number, with where each one's piece
    /// went once it was placed (`takes`).
    waits: HashMap<u64, Option<Placed>>,
    /// The source, for the take workers' own blocks: let go of at the device's release.
    source: Option<Arc<dyn BlockSource>>,
    /// The contexts' blocks of the pool, by guest physical address (`takes`).
    carves: BTreeMap<u64, takes::Carve>,
    /// The contexts with a take in flight or a worker, by context (`takes`).
    takers: HashMap<u32, takes::Taker>,
    /// The GPU addresses given to the carves of pieces whose blocks have none (`addresses`).
    addresses: addresses::HostAddresses,
    /// The last number a carve, a context's takes or an ask took.
    serial: u64,
    /// The take workers running, and the zeroer (`takes`).
    workers: usize,
    /// The chunk the zeroer is writing, which no carve takes: its piece's address and its
    /// offsets in the piece's block (`takes`, "Zeros ahead").
    claim: Option<(u64, std::ops::Range<u64>)>,
    /// The takes whose first fit holds bytes of the claimed chunk, waiting for its write: the
    /// zeroer claims no other meanwhile (`takes`, "Zeros ahead").
    claim_waits: usize,
    /// The zeroer runs.
    zeroer: bool,
    /// For the tests: the zeroer claims no chunk.
    #[cfg(test)]
    zero_shut: bool,
    /// For the tests: the zeroer waits, its chunk claimed, before its write.
    #[cfg(test)]
    zero_paused: bool,
    /// For the tests: the bytes the take workers zeroed, then the zeroer.
    #[cfg(test)]
    zeroed_by: [u64; 2],
    /// The ledgers given out and not dropped ([`PoolLedger`]).
    keepers: usize,
    /// The source, handed over by the device's release while a ledger lived: the last ledger's
    /// drop gives every block back to it ([`Pool::close`]).
    left: Option<Arc<dyn BlockSource>>,
    /// The device was released: the host removed every piece it added.
    closed: bool,
    /// The pool's thread logged the range's being full once.
    full_logged: bool,
}

/// An ask for at least `length` more bytes of pool that no take waits on.
#[derive(Clone, Copy, Debug)]
struct Ask {
    length: u64,
}

/// A block a take worker took for its take (`takes`), and the take.
#[derive(Clone, Copy, Debug)]
struct Own {
    block: BlockInfo,
    take: OwnTake,
}

/// The take a take worker took a block for: the number of its wait on the block's place, and the
/// take, `len` bytes of `context`'s takes numbered `life`, which the pool's thread carves at the
/// block's piece's first byte as it places it.
#[derive(Clone, Copy, Debug)]
struct OwnTake {
    wait: u64,
    context: u32,
    life: u64,
    len: u64,
}

/// Where a take's own block went (`takes`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Placed {
    /// No piece came of it.
    Nowhere,
    /// Its piece's address, and the take's carve at the piece's first byte, by number, where the
    /// pool's thread carved it there.
    At(u64, Option<u64>),
}

impl State {
    /// A number no carve, context's takes or ask took before.
    fn next_serial(&mut self) -> u64 {
        self.serial = self.serial.saturating_add(1);
        self.serial
    }

    /// The guest put something in the piece at `at`: the pool keeps its headroom
    /// ([`Self::keep_headroom`]).
    fn use_piece(&mut self, at: u64) {
        let Some(p) = self.pieces.get_mut(&at) else {
            return;
        };
        p.empty_since = None;
        self.keep_headroom();
    }

    /// Where the empty pieces the guest may use, the asks for more pool no take waits on, the one
    /// the pool's taker serves among them, and the blocks it took that wait for their places,
    /// hold fewer than [`HEADROOM`] bytes, asks for a block of [`BLOCK_LEN`] for each piece short,
    /// so they hold that many: whether it asked.
    fn keep_headroom(&mut self) -> bool {
        let empty = (self.empties()).fold(0, |all: u64, (_, p, _)| all.saturating_add(p.block.len));
        let asked =
            (self.asks.iter()).fold(self.serving, |all: u64, a| all.saturating_add(a.length));
        let taken = (self.ahead.iter()).fold(asked, |all: u64, b| all.saturating_add(b.len));
        let mut held = empty.saturating_add(taken);
        let ask = held < HEADROOM;
        while held < HEADROOM {
            self.asks.push_back(Ask { length: BLOCK_LEN });
            held = held.saturating_add(BLOCK_LEN);
        }
        ask
    }

    /// The empty pieces the guest may use, lowest address first, each with whether it is of the
    /// headroom: each one whose empty pieces below it hold fewer than [`HEADROOM`] bytes.
    fn empties(&self) -> impl Iterator<Item = (u64, &Piece, bool)> {
        let mut below: u64 = 0;
        (self.pieces.iter())
            .filter(|(_, p)| p.usable() && p.empty_since.is_some())
            .map(move |(&at, p)| {
                let headroom = below < HEADROOM;
                below = below.saturating_add(p.block.len);
                (at, p, headroom)
            })
    }

    /// The guest emptied the piece at `at`, at `now`: nothing of the guest's lies in it, and no
    /// guest mapping of it is left.
    fn empty_piece(&mut self, at: u64, now: MonoTime) {
        if let Some(p) = self.pieces.get_mut(&at)
            && p.empty_since.is_none()
        {
            p.empty_since = Some(now);
        }
    }

    /// Sets the piece at `at` on its way back to the source, at `now`: out of the lookups, its
    /// remove asked once no hold is left on it ([`Self::next_remove`]), and its block given back
    /// once the host removed it ([`Self::next_back`]).
    fn send_back(&mut self, at: u64, now: MonoTime) {
        if let Some(p) = self.pieces.get_mut(&at) {
            p.mapped = Mapped::Removing { asked: false };
            p.going = Some(Going {
                since: now,
                warned: false,
            });
        }
    }

    /// A piece on its way back that no hold is on and whose remove the host was not asked yet,
    /// marked asked: its address and length.
    fn next_remove(&mut self) -> Option<(u64, u64)> {
        let (&at, p) = (self.pieces.iter_mut()).find(|(_, p)| {
            p.going.is_some() && p.holds == 0 && p.mapped == Mapped::Removing { asked: false }
        })?;
        p.mapped = Mapped::Removing { asked: true };
        Some((at, p.block.len))
    }

    /// The block of a piece on its way back that the host removed and no hold or keep is on, the
    /// piece taken out of the pool.
    fn next_back(&mut self) -> Option<BlockId> {
        let at = (self.pieces.iter())
            .find(|(_, p)| p.going.is_some() && !p.held() && p.mapped == Mapped::Removed)
            .map(|(&at, _)| at)?;
        self.take_piece(at)
    }

    /// At the device's release, once the host removed every piece: the blocks of the pieces on
    /// their way back that no hold or keep is on, the pieces taken out of the pool. The rest wait
    /// for their holds and keeps at the release's end ([`Shared::take_all_out`]).
    fn take_out_going(&mut self) -> Vec<BlockId> {
        let going: Vec<u64> = (self.pieces.iter())
            .filter(|(_, p)| p.going.is_some() && !p.held())
            .map(|(&at, _)| at)
            .collect();
        (going.into_iter())
            .filter_map(|at| self.take_piece(at))
            .collect()
    }

    /// Takes the piece at `at` out of the pool: its block.
    fn take_piece(&mut self, at: u64) -> Option<BlockId> {
        let p = self.pieces.remove(&at)?;
        self.by_block.remove(&p.block.id);
        Some(p.block.id)
    }

    /// Warns once of each piece on its way back that still waits 1 s after it set out, saying
    /// what for, at `now`: how long until the next of them is due its warning.
    fn warn_going(&mut self, now: MonoTime) -> Option<Duration> {
        let mut next: Option<Duration> = None;
        for (&at, p) in &mut self.pieces {
            let Some(going) = p.going.as_mut().filter(|g| !g.warned) else {
                continue;
            };
            let when = going.since.saturating_add(Duration::from_secs(1));
            if when > now {
                let after = when.saturating_duration_since(now);
                next = Some(next.map_or(after, |n| n.min(after)));
                continue;
            }
            going.warned = true;
            log::warn!(
                "pool: the piece at {at:#x} still waits to go back after 1 s: {}",
                if p.holds > 0 {
                    "a thread of the device's holds it"
                } else if p.kept() {
                    "an import of its bytes, not yet freed, keeps it"
                } else {
                    "the host has not removed it"
                }
            );
        }
        next
    }
}

/// The pool's shared part: what the thread, the device and the holds hold.
struct Shared {
    range: PoolConfig,
    host: PoolHost,
    /// [`RELEASE_DELAY`], shorter in the tests.
    release_delay: Duration,
    /// The platform's monotonic clock.
    now: PoolNow,
    state: Mutex<State>,
    changed: Condvar,
    /// What the pool's thread and the release's end sleep on ([`Bell`]), rung at every change
    /// ([`Shared::wake`]) and by a keep's drop ([`PieceKeep`]).
    bell: Arc<Bell>,
    /// The take workers' and the zeroer's threads, which the device's release waits for.
    threads: Arc<Threads>,
}

/// Worker threads, the pool's ([`takes`]: its take workers and its zeroer) or the Private blocks'
/// ([`private_blocks`]), each counted from its spawn until it holds what it works on no more. A
/// worker's handle may be the last of it, whose drop drops what that holds: the block source of
/// the Private blocks, or the pool's clock, which holds the platform; rt's device's drop calls
/// the driver. The device's release waits for every one ([`Pool::close`],
/// [`private_blocks::PrivateBlocks::close`]), so that neither is dropped on a worker's thread past
/// the release, where the process may be exiting.
#[derive(Default)]
struct Threads {
    /// The threads counted.
    live: Mutex<usize>,
    /// Told when one goes.
    gone: Condvar,
}

impl Threads {
    /// One thread goes.
    fn go(&self) {
        let mut live = self.live.lock();
        *live = live.saturating_sub(1);
        drop(live);
        self.gone.notify_all();
    }

    /// Waits for every thread counted to go.
    fn wait(&self) {
        let mut live = self.live.lock();
        while *live > 0 {
            self.gone.wait(&mut live);
        }
    }
}

/// A worker's count among the [`Threads`]: it goes at this one's drop.
struct Live(Arc<Threads>);

impl Live {
    /// One thread more among `threads`.
    fn new(threads: &Arc<Threads>) -> Live {
        let mut live = threads.live.lock();
        *live = live.saturating_add(1);
        Live(Arc::clone(threads))
    }
}

impl Drop for Live {
    fn drop(&mut self) {
        self.0.go();
    }
}

/// A hold on a live piece: while it lives, its block does not go back.
#[allow(
    dead_code,
    reason = "the guest's side of the pool (its ring messages) calls it once it lands"
)]
pub struct PieceHold {
    shared: Arc<Shared>,
    physical_address: u64,
    /// The block.
    pub block: BlockId,
    /// The byte the lookup named, in the block.
    pub offset: u64,
    /// That byte's host address.
    pub host: NonNull<u8>,
}

impl std::fmt::Debug for PieceHold {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PieceHold")
            .field("block", &self.block)
            .field("offset", &self.offset)
            .finish_non_exhaustive()
    }
}

impl Drop for PieceHold {
    fn drop(&mut self) {
        let mut st = self.shared.state.lock();
        if let Some(p) = st.pieces.get_mut(&self.physical_address) {
            p.holds = p.holds.saturating_sub(1);
        }
        drop(st);
        self.shared.wake();
    }
}

/// The guest's Shared pool of one device.
pub struct Pool {
    shared: Arc<Shared>,
    /// The source. Let go of once the threads that take from it have ended and every block went
    /// back to it ([`Pool::close`]); it frees its blocks once nothing else holds it.
    source: Option<Arc<dyn BlockSource>>,
    /// The pool's thread, which places pieces and gives them back ([`Shared::serve`]).
    thread: Option<JoinHandle<()>>,
    /// The pool's taker, which takes the blocks of the asks no take waits on
    /// ([`Shared::take_ahead`]).
    ahead: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for Pool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pool")
            .field("range", &self.shared.range)
            .finish_non_exhaustive()
    }
}

impl Pool {
    /// A pool over `range`, timed by `now`, its blocks from `source`, and its thread and taker
    /// started. [`Error::InvalidArgument`] for a range [`PoolConfig::is_valid`] refuses,
    /// [`Error::HostResources`] where a thread is refused.
    pub(crate) fn new(
        range: PoolConfig,
        host: PoolHost,
        now: PoolNow,
        source: Arc<dyn BlockSource>,
    ) -> Result<Pool, Error> {
        Pool::with_delay(range, host, RELEASE_DELAY, now, source)
    }

    /// [`Pool::new`], its pieces given back after `release_delay`.
    fn with_delay(
        range: PoolConfig,
        host: PoolHost,
        release_delay: Duration,
        now: PoolNow,
        source: Arc<dyn BlockSource>,
    ) -> Result<Pool, Error> {
        if !range.is_valid() {
            return Err(Error::InvalidArgument);
        }
        let shared = Arc::new(Shared {
            range,
            host,
            release_delay,
            now,
            state: Mutex::new(State {
                source: Some(Arc::clone(&source)),
                ..State::default()
            }),
            changed: Condvar::new(),
            bell: Arc::default(),
            threads: Arc::default(),
        });
        let spawn = |name: &str, f: fn(&Arc<Shared>, &dyn BlockSource)| {
            let shared = Arc::clone(&shared);
            let source = Arc::clone(&source);
            (std::thread::Builder::new().name(name.into()))
                .spawn(move || f(&shared, source.as_ref()))
                .ok()
        };
        let thread = spawn("reims-vgpu.graphics-pool", Shared::serve);
        let ahead = spawn("reims-vgpu.graphics-pool-ahead", Shared::take_ahead);
        let mut pool = Pool {
            shared,
            source: Some(source),
            thread,
            ahead,
        };
        if pool.thread.is_none() || pool.ahead.is_none() {
            // The release ends the one made.
            pool.close();
            return Err(Error::HostResources);
        }
        Ok(pool)
    }

    /// The range of guest physical addresses the pool takes.
    pub fn range(&self) -> PoolConfig {
        self.shared.range
    }

    /// Asks for at least `length` more bytes of pool: the pool's taker takes a block, and the
    /// pool's thread has the host map it. Waits for nothing.
    pub fn grow(&self, length: u64) {
        let ask = Ask { length };
        self.shared.state.lock().asks.push_back(ask);
        self.shared.wake();
    }

    /// A ledger of the pool's blocks the guest's contexts hold ([`PoolLedger`]): takes, gives and
    /// lookups. While one lives, the device's release ([`Pool::close`]) leaves every block with
    /// the pool, and the last one's drop gives them back.
    pub(crate) fn ledger(&self) -> PoolLedger {
        PoolLedger::of(&self.shared)
    }

    /// The host mapped the piece at `physical_address` it was asked to add: lookups find it from
    /// now on. [`Error::InvalidArgument`] for an address the pool asked no add at, or one the host
    /// said it added already.
    pub fn added(&self, physical_address: u64) -> Result<(), Error> {
        let mut st = self.shared.state.lock();
        let p = (st.pieces.get_mut(&physical_address)).ok_or(Error::InvalidArgument)?;
        match p.mapped {
            Mapped::Adding => p.mapped = Mapped::Live,
            // Its give-back came first: the host's remove follows its add.
            Mapped::Removing { .. } => {}
            Mapped::Live | Mapped::Removed => return Err(Error::InvalidArgument),
        }
        log::info!(
            "pool: {} MiB at {physical_address:#x} mapped into the guest",
            p.block.len >> 20
        );
        let len = p.block.len;
        drop(st);
        let piece = trail::Held::over(None, None, [(physical_address, len)]);
        trail::record(trail::Kind::PoolMapped, piece);
        // A take may wait for it.
        self.shared.wake();
        Ok(())
    }

    /// The host took the piece at `physical_address` out of the guest's address space, and no
    /// thread of its own touches it: asked to ([`PoolHost::remove`]), or of its own, as it does
    /// before it releases the device. [`Error::InvalidArgument`] for an address the pool asked
    /// no add at.
    pub fn removed(&self, physical_address: u64) -> Result<(), Error> {
        let mut st = self.shared.state.lock();
        let p = (st.pieces.get_mut(&physical_address)).ok_or(Error::InvalidArgument)?;
        p.mapped = Mapped::Removed;
        drop(st);
        self.shared.wake();
        Ok(())
    }

    /// The guest put something in the piece at `physical_address`, as a context's first block
    /// in it does (`takes`): the pool keeps its headroom ([`State::keep_headroom`]).
    /// [`Error::InvalidArgument`] for an address where no piece the guest may use starts. For the
    /// tests, which drive a piece's use without the takes.
    #[cfg(test)]
    pub fn piece_used(&self, physical_address: u64) -> Result<(), Error> {
        let mut st = self.shared.state.lock();
        let p = (st.pieces.get(&physical_address)).ok_or(Error::InvalidArgument)?;
        if !p.usable() {
            return Err(Error::InvalidArgument);
        }
        st.use_piece(physical_address);
        drop(st);
        self.shared.wake();
        Ok(())
    }

    /// The guest emptied the piece at `physical_address`, as a context's last block in it going
    /// back does (`takes`): no guest mapping of it is left, and nothing of the guest's lies in it.
    /// [`RELEASE_DELAY`] later, if it is still empty and above the headroom, its block goes back to
    /// the source, and the guest may not use it again until the device hands it out anew.
    /// [`Error::InvalidArgument`] for an address where no piece the guest may use starts. For the
    /// tests, as [`Pool::piece_used`].
    #[cfg(test)]
    pub fn piece_empty(&self, physical_address: u64) -> Result<(), Error> {
        let mut st = self.shared.state.lock();
        let p = (st.pieces.get(&physical_address)).ok_or(Error::InvalidArgument)?;
        if !p.usable() {
            return Err(Error::InvalidArgument);
        }
        let now = (self.shared.now)();
        st.empty_piece(physical_address, now);
        drop(st);
        self.shared.wake();
        Ok(())
    }

    /// The live piece the guest may use that holds `length` bytes from `physical_address`, held
    /// until the answer drops; `None` where none holds all of them.
    #[allow(
        dead_code,
        reason = "the guest's side of the pool (its ring messages) calls it once it lands"
    )]
    pub fn find(&self, physical_address: u64, length: u64) -> Option<PieceHold> {
        let mut st = self.shared.state.lock();
        let (&start, p) = st.pieces.range_mut(..=physical_address).next_back()?;
        let offset = physical_address.checked_sub(start)?;
        let end = offset.checked_add(length)?;
        if p.mapped != Mapped::Live || end > p.block.len {
            return None;
        }
        let contents = p.block.contents?;
        p.holds = p.holds.checked_add(1)?;
        let block = p.block.id;
        drop(st);
        let offset_bytes = usize::try_from(offset).ok()?;
        // SAFETY: `offset` is inside the block's mapping, checked above.
        let host = unsafe { NonNull::new_unchecked(contents.as_ptr().add(offset_bytes)) };
        Some(PieceHold {
            shared: Arc::clone(&self.shared),
            physical_address: start,
            block,
            offset,
            host,
        })
    }

    /// The pieces the guest may use and the host mapped, by address: `(physical_address,
    /// length)`.
    #[allow(
        dead_code,
        reason = "the guest's side of the pool (its ring messages) calls it once it lands"
    )]
    pub fn live(&self) -> Vec<(u64, u64)> {
        let st = self.shared.state.lock();
        (st.pieces.iter())
            .filter(|(_, p)| p.mapped == Mapped::Live)
            .map(|(&a, p)| (a, p.block.len))
            .collect()
    }

    /// The device's release: the host removed every piece it added. Ends the pool's thread (which
    /// gives back the pieces on their way back that no hold is on), its taker, its take workers
    /// (a take in flight is answered never) and its zeroer, and waits until their threads let go
    /// of the pool ([`Threads`]); gives back the blocks taken and not placed, with no word to the
    /// host; then, where no ledger lives, waits for every hold on the pieces left, gives each
    /// one's block back and lets go of the source, or past the release's bound gives back the
    /// free ones' alone and forgets the source ([`Shared::give_all_back`]). Where a ledger lives
    /// (the decoder's, whose heaps over the contexts' blocks may outlive the device), the last
    /// one's drop does that.
    pub(crate) fn close(&mut self) {
        let mut st = self.shared.state.lock();
        st.closed = true;
        st.source = None;
        drop(st);
        self.shared.wake();
        for t in [self.thread.take(), self.ahead.take()]
            .into_iter()
            .flatten()
        {
            let _ = t.join();
        }
        self.shared.wait_for_workers();
        self.shared.threads.wait();
        let Some(source) = self.source.take() else {
            return;
        };
        let mut st = self.shared.state.lock();
        let own = std::mem::take(&mut st.own);
        let ahead = std::mem::take(&mut st.ahead);
        drop(st);
        let unplaced = (own.into_iter().map(|o| o.block)).chain(ahead);
        for block in unplaced {
            if let Err(e) = source.release_block(block.id) {
                log::warn!(kind:? = e; "pool: the source refused a block back: {e:?}");
            }
        }
        let mut st = self.shared.state.lock();
        if st.keepers > 0 {
            st.left = Some(source);
            return;
        }
        drop(st);
        self.shared.give_all_back(source);
    }

    /// The block source, for the tests.
    #[cfg(test)]
    fn source(&self) -> Option<&Arc<dyn BlockSource>> {
        self.source.as_ref()
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.close();
    }
}

impl Shared {
    /// Tells the pool's waiters its state changed: those on its lock's condvar, and the pool's
    /// thread and the release's end, which sleep on its bell ([`Bell`]).
    fn wake(&self) {
        self.changed.notify_all();
        self.bell.ring();
    }

    /// The pool's thread: places each block taken, the take workers' first, and has the host map
    /// it, making the zeroer where none runs (`takes`, "Zeros ahead"); and gives back the pieces
    /// that stayed empty past their delay, a step at a time as each comes due, waiting for no
    /// hold, no remove and no keep; until the pool closes, where it gives back those on their way
    /// back that no hold or keep is on. It sleeps on the pool's bell, which every change and a
    /// keep's drop ring ([`Bell`]).
    fn serve(self: &Arc<Self>, source: &dyn BlockSource) {
        let mut st = self.state.lock();
        loop {
            let seen = self.bell.rung();
            if st.closed {
                let back = st.take_out_going();
                drop(st);
                for block in back {
                    if let Err(e) = source.release_block(block) {
                        log::warn!(kind:? = e; "pool: the source refused a block back: {e:?}");
                    }
                }
                return;
            }
            let next = match st.own.pop_front() {
                Some(own) => Some((own.block, Some(own.take))),
                None => st.ahead.pop_front().map(|block| (block, None)),
            };
            if let Some((block, take)) = next {
                self.place(&mut st, source, block, take);
                continue;
            }
            if let Some((at, len)) = st.next_remove() {
                MutexGuard::unlocked(&mut st, || (self.host.remove)(at, len));
                continue;
            }
            if let Some(block) = st.next_back() {
                MutexGuard::unlocked(&mut st, || {
                    if let Err(e) = source.release_block(block) {
                        log::warn!(kind:? = e; "pool: the source refused a block back: {e:?}");
                    }
                });
                continue;
            }
            let now = (self.now)();
            let (due, next) = self.due(&st, now);
            if let Some(at) = due {
                let piece = (st.pieces.get(&at)).map(|p| (p.block.len, p.empty_since));
                st.send_back(at, now);
                MutexGuard::unlocked(&mut st, || {
                    log::info!("pool: the empty piece at {at:#x} goes back to the source");
                    if let Some((len, since)) = piece {
                        let piece = trail::Held::over(None, None, [(at, len)]);
                        trail::record(trail::Kind::PoolBack, piece);
                        let empty = since
                            .map_or(Duration::ZERO, |since| now.saturating_duration_since(since));
                        let empty = u64::try_from(empty.as_nanos()).unwrap_or(u64::MAX);
                        // poolback: the piece's address and length, and the nanoseconds it
                        // stood empty
                        trace::event("poolback", &[at, len, empty]);
                    }
                });
                continue;
            }
            let next = match (next, st.warn_going(now)) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
            MutexGuard::unlocked(&mut st, || self.bell.wait(seen, next));
        }
    }

    /// The pool's taker: takes a Shared block for each ask no take waits on ([`State::asks`]),
    /// oldest first, with the pool's lock let go of, and hands it to the pool's thread to place
    /// ([`State::ahead`]), until the pool closes. A block it takes once the device's release began
    /// goes back with no word to the host, which removed its pieces before the release.
    fn take_ahead(self: &Arc<Self>, source: &dyn BlockSource) {
        let mut st = self.state.lock();
        loop {
            if st.closed {
                return;
            }
            let Some(Ask { length }) = st.asks.pop_front() else {
                self.changed.wait(&mut st);
                continue;
            };
            st.serving = length;
            let block =
                MutexGuard::unlocked(&mut st, || source.take_block(MemoryClass::Shared, length));
            st.serving = 0;
            match block {
                Ok(block) if !st.closed => {
                    st.ahead.push_back(block);
                    self.wake();
                }
                Ok(block) => MutexGuard::unlocked(&mut st, || {
                    let _ = source.release_block(block.id);
                }),
                Err(e) => MutexGuard::unlocked(&mut st, || {
                    log::warn!(kind:? = e; "pool: no block of {length} bytes: {e:?}");
                }),
            }
        }
    }

    /// The piece whose block goes back now, if any, and how long until the next may: of the
    /// empty pieces the guest may use, every one above the headroom ([`State::empties`]), once
    /// empty for the delay.
    fn due(&self, st: &State, now: MonoTime) -> (Option<u64>, Option<Duration>) {
        let empty = (st.empties()).filter_map(|(at, p, headroom)| {
            let since = p.empty_since.filter(|_| !headroom)?;
            Some((at, since.saturating_add(self.release_delay)))
        });
        let mut next: Option<Duration> = None;
        for (at, when) in empty {
            if when <= now {
                return (Some(at), None);
            }
            let after = when.saturating_duration_since(now);
            next = Some(next.map_or(after, |n| n.min(after)));
        }
        (None, next)
    }

    /// Places `block` at the lowest free address that holds it, a multiple of [`PIECE_ALIGN`], as a
    /// piece of `block`'s length, whatever multiple of its source's granule that is, and asks the
    /// host to map it, with the lock `st` holds let go of for the add. Where `take` still waits on
    /// it, the take is carved at the piece's first byte, and learns where it went before the add,
    /// so its worker zeroes the carve while the host maps the piece (`takes`). No piece comes of
    /// it, and it goes back, where the range is full or the block is not whole host pages
    /// ([`UNALIGNED_CODE`]).
    fn place(
        self: &Arc<Self>,
        st: &mut MutexGuard<'_, State>,
        source: &dyn BlockSource,
        block: BlockInfo,
        take: Option<OwnTake>,
    ) {
        let placed = self.fit_piece(st, source, block);
        if let Some(take) = take
            && st.waits.contains_key(&take.wait)
        {
            let outcome = match placed {
                Some((at, _)) => {
                    let carved = (st.stands(take.context, take.life))
                        .then(|| st.carve_first(take.context, take.len, at))
                        .flatten();
                    Placed::At(at, carved)
                }
                None => Placed::Nowhere,
            };
            st.waits.insert(take.wait, Some(outcome));
        }
        let start = placed.is_some() && st.start_zeroer();
        MutexGuard::unlocked(st, || {
            if start {
                takes::spawn_zeroer(self);
            }
            // The take waiting on it, where it still does, looks where the piece went.
            self.wake();
            if let Some((at, contents)) = placed {
                (self.host.add)(at, contents.as_ptr(), block.len);
            }
        });
    }

    /// [`Self::place`]'s piece: `block`'s, at the lowest free address that holds it, its address
    /// and host mapping, or `None` where the block went back, the range full or the block not
    /// whole host pages. The pool's lock `st` holds is let go of where the block goes back.
    fn fit_piece(
        &self,
        st: &mut MutexGuard<'_, State>,
        source: &dyn BlockSource,
        block: BlockInfo,
    ) -> Option<(u64, NonNull<u8>)> {
        let back = |st: &mut MutexGuard<'_, State>| {
            MutexGuard::unlocked(st, || {
                let _ = source.release_block(block.id);
            });
        };
        let Some(contents) = block.contents else {
            back(st);
            return None;
        };
        let page = OsVm::new().page_size().max(1);
        let at = contents.as_ptr().addr();
        if !(at.to_u64().is_multiple_of(page) && block.len.is_multiple_of(page)) {
            // The process's first with a line, each counted.
            static LOGGED: AtomicBool = AtomicBool::new(false);
            if !LOGGED.swap(true, Ordering::Relaxed) {
                log::warn!(
                    "pool: block {} of {:#x} bytes at host address {at:#x} is not whole host \
                     pages of {page:#x} bytes: it goes back unmapped ({UNALIGNED_CODE})",
                    block.id.0,
                    block.len,
                );
            }
            count_kind(UNALIGNED_CODE);
            back(st);
            return None;
        }
        let Some(at) = first_fit(self.range, &st.pieces, block.len) else {
            if !std::mem::replace(&mut st.full_logged, true) {
                log::warn!(
                    "pool: the range at {:#x} of {} MiB holds no {} MiB more: the block goes back",
                    self.range.physical_address,
                    self.range.physical_length >> 20,
                    block.len >> 20
                );
            }
            back(st);
            return None;
        };
        st.pieces.insert(
            at,
            Piece {
                block,
                mapped: Mapped::Adding,
                empty_since: Some((self.now)()),
                holds: 0,
                kept: Arc::new(()),
                carves: 0,
                zeros: takes::Ranges::new(),
                zero_pending: true,
                going: None,
            },
        );
        st.by_block.insert(block.id, at);
        Some((at, contents))
    }

    /// [`Pool::close`]'s part, once the pool's thread has ended: waits for every hold and keep on
    /// the pieces left to go, asleep on the pool's bell, which their drops ring ([`Bell`]), at most
    /// the release's bound ([`crate::LET_GO_BOUND`]), then takes them all out of the pool and
    /// answers the blocks no hold or keep is on, and whether that is every block. A keep goes once
    /// the platform freed its import, which it does once no GPU work uses the bytes. The host
    /// removed every piece before the release, so it is asked nothing.
    fn take_all_out(&self) -> (Vec<BlockId>, bool) {
        const WARN_AFTER: Duration = Duration::from_secs(1);
        let bound = crate::LET_GO_BOUND;
        let mut st = self.state.lock();
        let mut warned = false;
        let started = (self.now)();
        loop {
            let seen = self.bell.rung();
            let Some((&at, p)) = (st.pieces.iter()).find(|(_, p)| p.held()) else {
                break;
            };
            let waited = (self.now)().saturating_duration_since(started);
            if waited >= bound {
                break;
            }
            if !warned && waited > WARN_AFTER {
                warned = true;
                let what = match p.holds > 0 {
                    true => "a thread of the device's that holds",
                    false => "the free of an import of the bytes of",
                };
                log::warn!(
                    "pool: the release still waits after 1 s for {what} the piece at {at:#x}"
                );
            }
            // Until the hold's or the keep's drop rings the bell, or the warning or the bound is
            // due.
            let due = if warned { bound } else { WARN_AFTER };
            let most = due.saturating_sub(waited).max(Duration::from_millis(1));
            MutexGuard::unlocked(&mut st, || self.bell.wait(seen, Some(most)));
        }
        st.by_block.clear();
        st.carves.clear();
        st.takers.clear();
        let pieces = std::mem::take(&mut st.pieces);
        drop(st);
        // Its holds are no longer counted down once it is out of the pool: a piece held here, past
        // the bound, stays held.
        let (held, free): (Vec<_>, Vec<_>) = pieces.into_iter().partition(|(_, p)| p.held());
        if !held.is_empty() {
            let named: Vec<String> = (held.iter())
                .map(|(at, p)| match p.holds > 0 {
                    true => format!("{at:#x} (held by a thread of the device's)"),
                    false => format!("{at:#x} (an import of its bytes not freed)"),
                })
                .collect();
            log::warn!(
                "pool: the release waited {bound:?}, its bound, for the pieces at {}: their blocks \
                 stay out, and the block source and its device are kept to the exit",
                named.join(", ")
            );
        }
        let blocks = (free.into_iter()).map(|(_, p)| p.block.id).collect();
        (blocks, held.is_empty())
    }

    /// The device's release's end, once no ledger lives: the blocks no hold or keep is on go back
    /// to `source` ([`Self::take_all_out`]), which is then let go of where they were all of them.
    /// Where a hold or a keep outlasted the release's bound, the source is forgotten instead, so
    /// neither the pool nor the source's drop gives back a block still held or imported (the
    /// runtime's source gives back every block it holds as it drops, `rt_blocks`), and the source
    /// stays to the process's exit. Nothing reaches it after: the pool keeps it nowhere else
    /// ([`Pool::close`] took it out of the state before), and a hold's or a keep's drop finds its
    /// piece gone.
    fn give_all_back(&self, source: Arc<dyn BlockSource>) {
        let (blocks, all) = self.take_all_out();
        for block in blocks {
            if let Err(e) = source.release_block(block) {
                log::warn!(kind:? = e; "pool: the source refused a block back: {e:?}");
            }
        }
        if all {
            drop(source);
        } else {
            std::mem::forget(source);
        }
    }
}

/// The lowest address in `range`, a multiple of [`PIECE_ALIGN`], where `length` bytes overlap no
/// piece of `pieces`; `None` where none is left.
fn first_fit(range: PoolConfig, pieces: &BTreeMap<u64, Piece>, length: u64) -> Option<u64> {
    let end = range.physical_address.checked_add(range.physical_length)?;
    let mut at = range.physical_address;
    for (&start, p) in pieces {
        if at.checked_add(length)? <= start {
            break;
        }
        let past = start.checked_add(p.block.len)?;
        at = at.max(past.checked_next_multiple_of(PIECE_ALIGN)?);
    }
    (at.checked_add(length)? <= end).then_some(at)
}

#[cfg(test)]
mod tests;
