//! The guest's Shared pool: a range of guest physical addresses the host reserved with no RAM
//! under it ([`PoolConfig`]), which the device fills with the block source's host-visible blocks
//! as it grows, one piece a block, and gives back as the guest empties them.
//!
//! # Growth
//!
//! An ask for more pool ([`Pool::grow`]: the host's `rvg_graphics_device_pool_grow`, and the
//! headroom rule below) wakes the pool's thread, `reims-vgpu.graphics-pool`. It takes a Shared
//! block from the source ([`BlockSource::take_block`], which waits only when no block is ready),
//! places it at the lowest free address of the range that holds it, 2 MiB aligned, so pieces go
//! out in ascending addresses, and asks the host to map it there ([`PoolHost::add`]) with no lock
//! of the pool's held. The host maps it (QEMU: a RAM region of its own over the block, added on
//! its main loop) and says so ([`Pool::added`], the C API's "range added"); only then does
//! [`Pool::find`] find it. A block the source hands out again after the pool gave it back is
//! placed and mapped anew, as any block is: its piece went before it went back.
//!
//! # Headroom and giving blocks back
//!
//! A new piece is empty: nothing of the guest's lies in it. The guest's processes take blocks of
//! the pool and give them back (`takes`): a piece is used from its first block's take and empty
//! again once its last block went back. The pool keeps one empty piece as headroom: when the guest
//! uses the last empty one, the pool asks for a block more. It keeps one from the moment the
//! guest's driver takes the pool (`PoolLedger::keep_headroom`), before any take, so the guest's
//! first take finds a piece made and mapped. A piece that stays empty for [`RELEASE_DELAY`] while a lower one is empty too goes
//! back to the source ([`BlockSource::release_block`]), as the release below says, and the source
//! keeps the block ready or frees it; the lowest empty piece stays as the headroom. The delay
//! rides out churn, where a guest empties and fills a piece again within seconds.
//!
//! # Release
//!
//! The pool gives a block back to the source only once the guest's mapping of its piece is gone.
//! It takes the piece out of the lookups, waits for every hold on it ([`PieceHold`]) to go, asks
//! the host to unmap it ([`PoolHost::remove`]), and gives the block back once the host says it is
//! out of the guest's address space and no host thread of its own touches it ([`Pool::removed`]).
//! So the source may free a block the moment it gets it back: no guest access, no thread of the
//! host's and no thread of the library's touches it.
//!
//! At the device's release the host has removed every piece it added (QEMU does so as it tears
//! the device down, before the release): from then on ([`Pool::close`]) the pool waits only for
//! the holds and asks the host nothing, and it gives every block back before it lets go of the
//! source.
//!
//! # The block source
//!
//! [`BlockSource`] is metal2vulkan-rt's block calls as its API names them (`Device::take_block`,
//! `release_block`, [`BlockInfo`]). On the Vulkan platform the source is [`HostBlocks`], blocks of
//! the process's own memory, which hands blocks out, keeps them ready and frees them as the
//! runtime does. On the rt platform it is the runtime's blocks
//! ([`rt_blocks::RtBlocks`], the platform's `ApiPlatform::block_source`), each known to
//! its device while handed out.
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

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use parking_lot::{Condvar, Mutex};
use reims_vgpu_vulkan::answer::count_kind;
use reims_vgpu_vulkan::core::memory::placed::Placed;
use reims_vgpu_vulkan::core::num::ToU64 as _;
use reims_vgpu_vulkan::core::platform::os::clock::MonoTime;
use reims_vgpu_vulkan::core::platform::os::vm::HostVm as _;
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
#[cfg(feature = "rt")]
pub(crate) use rt_blocks::RtBlocks;
pub(crate) use takes::{Give, PoolLedger, TAKE_MAX};
#[cfg(test)]
use takes::{TAKE_ALIGN, TakeAnswer, Taken};

use crate::Error;

/// The code of a block the source handed out whose mapping is not whole host pages from a host
/// page, which goes back unmapped (the module doc's "The block source").
pub const UNALIGNED_CODE: &str = "rt-pool-block-unaligned";

/// The code of the bytes the take workers zeroed through their pieces' mappings (`takes`),
/// counted by the byte.
pub const TAKE_ZEROED_BYTES_CODE: &str = "pool-take-zeroed-bytes";

/// The code of a carve's bytes the take workers did not zero, as the piece's source said its
/// block read zero as it handed it out and no carve of the piece held them since
/// ([`BlockInfo::zeroed`], `takes`). Counted by the byte.
pub const TAKE_ZERO_SKIPPED_BYTES_CODE: &str = "pool-take-zero-skipped-bytes";

/// The code of a carve's bytes the take workers did not zero, as the pool's zeroer zeroed them
/// after an earlier carve of them was freed and no carve held them since (`takes`, "Zeros after
/// a free"). Counted by the byte.
pub const TAKE_ZERO_PREZEROED_BYTES_CODE: &str = "pool-take-zero-skipped-prezeroed-bytes";

/// The code of the bytes the pool's zeroer zeroed after their carves' frees (`takes`, "Zeros
/// after a free"), counted by the byte.
pub const ZEROED_AFTER_FREE_BYTES_CODE: &str = "pool-zeroed-after-free-bytes";

/// The code of the longest the pool's zeroer held the pool's lock for one chunk (`takes`, "Zeros
/// after a free"), in nanoseconds: its count rises to it.
pub const ZERO_HOLD_MAX_NS_CODE: &str = "pool-zero-hold-max-ns";

/// The codes of the longest each part of the zeroer's hold of the pool's lock for one chunk took
/// so far (`takes`, "Zeros after a free"), in nanoseconds, their counts rising to it: the chunk's
/// pick (`State::next_zero`), its write, and its mark as reading zero (`State::zeroed`).
pub const ZERO_HOLD_PART_MAX_NS_CODES: [&str; 3] = [
    "pool-zero-hold-pick-max-ns",
    "pool-zero-hold-write-max-ns",
    "pool-zero-hold-mark-max-ns",
];

/// The code of the zeroer's holds of the pool's lock for one chunk longer than 50 µs (`takes`,
/// [`ZERO_HOLD_LONG`]), counted by the hold.
pub const ZERO_HOLDS_LONG_CODE: &str = "pool-zero-holds-over-50us";

/// The codes of the nanoseconds the holds [`ZERO_HOLDS_LONG_CODE`] counts spent in each part, in
/// the order of [`ZERO_HOLD_PART_MAX_NS_CODES`].
pub const ZERO_LONG_HOLD_PART_NS_CODES: [&str; 3] = [
    "pool-zero-long-hold-pick-ns",
    "pool-zero-long-hold-write-ns",
    "pool-zero-long-hold-mark-ns",
];

/// A zeroer's hold longer than this is counted with its parts ([`ZERO_HOLDS_LONG_CODE`]): the
/// longest the walks of 11-19 saw at 16 KiB a chunk was 64 µs.
pub const ZERO_HOLD_LONG: std::time::Duration = std::time::Duration::from_micros(50);

/// The code of the takes answered, whose waits [`TAKE_WAIT_CODES`] count (`takes`).
pub const TAKE_WAITS_CODE: &str = "pool-take-waits";

/// The codes of the answered takes' waits, from each take's carry-out to its answer (`takes`):
/// the whole wait, then its parts behind the context's earlier takes, for a piece of the take's
/// own, and the take worker's zeroing; each the nanoseconds in all, then the longest.
pub const TAKE_WAIT_CODES: [(&str, &str); 4] = [
    ("pool-take-wait-ns", "pool-take-wait-max-ns"),
    ("pool-take-wait-queue-ns", "pool-take-wait-queue-max-ns"),
    ("pool-take-wait-growth-ns", "pool-take-wait-growth-max-ns"),
    ("pool-take-wait-zero-ns", "pool-take-wait-zero-max-ns"),
];

/// The length of the piece the pool keeps ahead of the guest's takes, its headroom (the module
/// doc's "Headroom and giving blocks back"), and of every block [`HostBlocks`] hands out for an
/// ask no longer: 256 MiB. It is the pool's own: metal2vulkan-rt hands out a block of the length
/// asked, rounded up to 64 KiB, carved from its 16 MiB allocations (its
/// `MemoryClass::block_len`) where it fits one, else at the start of an allocation of its own.
pub const BLOCK_LEN: u64 = 256 << 20;

/// The alignment of a piece's guest physical address and of a pool's range: 2 MiB, a large page
/// of the guest's and of the hypervisors' second-level tables.
pub const PIECE_ALIGN: u64 = 2 << 20;

/// How long a piece stays empty, with a lower piece empty too, before its block goes back to the
/// source (the module doc's "Headroom and giving blocks back").
pub const RELEASE_DELAY: Duration = Duration::from_secs(5);

/// A block's storage mode as the guest's buffers in it see it (metal2vulkan-rt's `MemoryClass`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(
    dead_code,
    reason = "metal2vulkan-rt's shape, kept whole so only the source changes when it lands"
)]
pub enum MemoryClass {
    /// Device-local, never mapped, no host pointer.
    Private,
    /// Host-visible, host-coherent and host-cached: the only class mapped into the guest.
    Shared,
}

/// A block's number, never reused while its source lives (metal2vulkan-rt's `BlockId`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u32);

impl BlockId {
    /// The number.
    #[allow(
        dead_code,
        reason = "metal2vulkan-rt's shape, kept whole so only the source changes when it lands"
    )]
    pub fn get(self) -> u32 {
        self.0
    }
}

/// A block a source handed out (metal2vulkan-rt's `BlockInfo`): plain data.
#[derive(Clone, Copy, Debug)]
#[allow(
    dead_code,
    reason = "metal2vulkan-rt's shape, kept whole so only the source changes when it lands"
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
    /// Whether every byte of it reads zero as the source hands it out (metal2vulkan-rt's
    /// `BlockInfo::zeroed`: the driver made its allocation zeroed, and no block held a byte at or
    /// past its first before): the take workers then zero only the bytes of a carve some carve of
    /// the piece held before (`takes`, [`Placed`]).
    pub zeroed: bool,
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

/// Where the pool's blocks come from (metal2vulkan-rt's `Device::take_block` and
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

/// One block in the range, which the pool holds until its piece is out of the guest.
#[derive(Debug)]
struct Piece {
    block: BlockInfo,
    mapped: Mapped,
    /// Since when nothing of the guest's lies in it; `None` while something does.
    empty_since: Option<MonoTime>,
    /// Holds on it ([`PieceHold`]).
    holds: u64,
    /// The contexts' blocks in it (`takes`).
    carves: u64,
    /// The mark of the bytes its carves ever held, from its block's take ([`Placed`]): those at
    /// or past it are as the source handed them out.
    placed: Placed,
    /// The bytes below the mark that read zero, as the zeroer zeroed them after their carve's
    /// free, by offset in the block; none of them is carved (`takes`, "Zeros after a free").
    zeros: takes::Ranges,
}

impl Piece {
    /// Whether the guest may put something in it: not being unmapped.
    fn usable(&self) -> bool {
        matches!(self.mapped, Mapped::Adding | Mapped::Live)
    }
}

/// What the pool's lock holds.
#[derive(Default)]
struct State {
    /// By guest physical address.
    pieces: BTreeMap<u64, Piece>,
    /// Each piece's address, by its block.
    by_block: HashMap<BlockId, u64>,
    /// Asks for more pool, oldest first.
    asks: VecDeque<Ask>,
    /// The asks a take waits on, by number, with where each one's piece went once it was served:
    /// `Some(None)` where no piece came of it (`takes`).
    waits: HashMap<u64, Option<Option<u64>>>,
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
    /// Freed carves' bytes the zeroer zeroes, oldest first: each its piece's address and the
    /// bytes' offsets in the piece's block (`takes`, "Zeros after a free").
    to_zero: VecDeque<(u64, std::ops::Range<u64>)>,
    /// The zeroer runs.
    zeroer: bool,
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

/// An ask for at least `length` more bytes of pool, and the number of the take's wait on it.
#[derive(Clone, Copy, Debug)]
struct Ask {
    length: u64,
    wait: Option<u64>,
}

impl State {
    /// A number no carve, context's takes or ask took before.
    fn next_serial(&mut self) -> u64 {
        self.serial = self.serial.saturating_add(1);
        self.serial
    }

    /// The guest put something in the piece at `at`: where no other piece is empty, the pool
    /// asks for a block more, so one stays as headroom.
    fn use_piece(&mut self, at: u64) {
        let Some(p) = self.pieces.get_mut(&at) else {
            return;
        };
        p.empty_since = None;
        self.keep_headroom();
    }

    /// Where no piece the guest may use is empty and no ask waits, asks for a block more, so one
    /// stays as headroom: whether it asked.
    fn keep_headroom(&mut self) -> bool {
        let headroom = (self.pieces.values()).any(|p| p.usable() && p.empty_since.is_some());
        let ask = !headroom && self.asks.is_empty();
        if ask {
            self.asks.push_back(Ask {
                length: BLOCK_LEN,
                wait: None,
            });
        }
        ask
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
        self.shared.changed.notify_all();
    }
}

/// The guest's Shared pool of one device.
pub struct Pool {
    shared: Arc<Shared>,
    /// The source. Let go of once the thread that takes from it has ended and every block went
    /// back to it ([`Pool::close`]); it frees its blocks once nothing else holds it.
    source: Option<Arc<dyn BlockSource>>,
    thread: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for Pool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pool")
            .field("range", &self.shared.range)
            .finish_non_exhaustive()
    }
}

impl Pool {
    /// A pool over `range`, timed by `now`, its blocks from `source`, and its thread started.
    /// [`Error::InvalidArgument`] for a range [`PoolConfig::is_valid`] refuses,
    /// [`Error::HostResources`] where the thread is refused.
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
            state: Mutex::new(State::default()),
            changed: Condvar::new(),
            threads: Arc::default(),
        });
        let thread = {
            let shared = Arc::clone(&shared);
            let source = Arc::clone(&source);
            std::thread::Builder::new()
                .name("reims-vgpu.graphics-pool".into())
                .spawn(move || shared.serve(source.as_ref()))
                .map_err(|_| Error::HostResources)?
        };
        Ok(Pool {
            shared,
            source: Some(source),
            thread: Some(thread),
        })
    }

    /// The range of guest physical addresses the pool takes.
    pub fn range(&self) -> PoolConfig {
        self.shared.range
    }

    /// Asks for at least `length` more bytes of pool: the pool's thread takes a block and has the
    /// host map it. Waits for nothing.
    pub fn grow(&self, length: u64) {
        let ask = Ask { length, wait: None };
        self.shared.state.lock().asks.push_back(ask);
        self.shared.changed.notify_all();
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
        drop(st);
        // A take may wait for it.
        self.shared.changed.notify_all();
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
        self.shared.changed.notify_all();
        Ok(())
    }

    /// The guest put something in the piece at `physical_address`, as a context's first block
    /// in it does (`takes`). Where no other piece is empty, the pool asks for a block more, so one
    /// stays as headroom. [`Error::InvalidArgument`] for an address where no piece the guest may
    /// use starts. For the tests, which drive a piece's use without the takes.
    #[cfg(test)]
    pub fn piece_used(&self, physical_address: u64) -> Result<(), Error> {
        let mut st = self.shared.state.lock();
        let p = (st.pieces.get(&physical_address)).ok_or(Error::InvalidArgument)?;
        if !p.usable() {
            return Err(Error::InvalidArgument);
        }
        st.use_piece(physical_address);
        drop(st);
        self.shared.changed.notify_all();
        Ok(())
    }

    /// The guest emptied the piece at `physical_address`, as a context's last block in it going
    /// back does (`takes`): no guest mapping of it is left, and nothing of the guest's lies in it.
    /// [`RELEASE_DELAY`] later, if it is still empty and a lower piece is empty too, its block
    /// goes back to the source, and the guest may not use it again until the device hands it out
    /// anew. [`Error::InvalidArgument`] for an address where no piece the guest may use starts.
    /// For the tests, as [`Pool::piece_used`].
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
        self.shared.changed.notify_all();
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

    /// The device's release: the host removed every piece it added. Ends the pool's thread (a
    /// piece it is giving back waits only for its holds from now on), its take workers (a take
    /// in flight is answered never) and its zeroer, and waits until their threads let go of the
    /// pool ([`Threads`]), then, where no ledger lives, waits for every hold on the
    /// pieces left, gives each one's block back and lets go of the source. Where a ledger lives
    /// (the decoder's, whose heaps over the contexts' blocks may outlive the device), the last
    /// one's drop does that.
    pub(crate) fn close(&mut self) {
        self.shared.state.lock().closed = true;
        self.shared.changed.notify_all();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        self.shared.wait_for_workers();
        self.shared.threads.wait();
        let Some(source) = self.source.take() else {
            return;
        };
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
    /// The pool's thread: takes a block for each ask and has the host map it, and gives back the
    /// pieces that stayed empty past their delay, until the pool closes.
    fn serve(&self, source: &dyn BlockSource) {
        loop {
            let mut st = self.state.lock();
            if st.closed {
                return;
            }
            if let Some(Ask { length, wait }) = st.asks.pop_front() {
                drop(st);
                let placed = match source.take_block(MemoryClass::Shared, length) {
                    Ok(block) => self.place(source, block),
                    Err(e) => {
                        log::warn!(kind:? = e; "pool: no block of {length} bytes: {e:?}");
                        None
                    }
                };
                // The take waiting on it, where it still does, looks where the piece went.
                if let Some(wait) = wait
                    && let Some(outcome) = self.state.lock().waits.get_mut(&wait)
                {
                    *outcome = Some(placed);
                }
                self.changed.notify_all();
                continue;
            }
            let now = (self.now)();
            let (due, next) = self.due(&st, now);
            if let Some((at, block)) = due {
                drop(st);
                log::info!("pool: the empty piece at {at:#x} goes back to the source");
                self.take_out(block);
                if let Err(e) = source.release_block(block) {
                    log::warn!(kind:? = e; "pool: the source refused a block back: {e:?}");
                }
                continue;
            }
            match next {
                Some(after) => {
                    let _ = self.changed.wait_for(&mut st, after);
                }
                None => self.changed.wait(&mut st),
            }
        }
    }

    /// The piece whose block goes back now, if any, and how long until the next may: of the
    /// empty pieces the guest may use, every one but the lowest, once empty for the delay.
    fn due(&self, st: &State, now: MonoTime) -> (Option<(u64, BlockId)>, Option<Duration>) {
        let mut empty = (st.pieces.iter()).filter_map(|(&at, p)| {
            let since = p.empty_since.filter(|_| p.usable())?;
            Some((at, p.block.id, since.saturating_add(self.release_delay)))
        });
        // The lowest empty piece is the headroom.
        let _ = empty.next();
        let mut next: Option<Duration> = None;
        for (at, block, when) in empty {
            if when <= now {
                return (Some((at, block)), None);
            }
            let after = when.saturating_duration_since(now);
            next = Some(next.map_or(after, |n| n.min(after)));
        }
        (None, next)
    }

    /// Places `block` at the lowest free address that holds it and asks the host to map it: the
    /// address, or `None` where the block went back, the range full, the pool closed or the
    /// block not whole host pages ([`UNALIGNED_CODE`]).
    fn place(&self, source: &dyn BlockSource, block: BlockInfo) -> Option<u64> {
        let Some(contents) = block.contents else {
            let _ = source.release_block(block.id);
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
            let _ = source.release_block(block.id);
            return None;
        }
        let mut st = self.state.lock();
        // A block made while the device was released goes back with no word to the host, which
        // removed its pieces before the release.
        if st.closed {
            drop(st);
            let _ = source.release_block(block.id);
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
            drop(st);
            let _ = source.release_block(block.id);
            return None;
        };
        st.pieces.insert(
            at,
            Piece {
                block,
                mapped: Mapped::Adding,
                empty_since: Some((self.now)()),
                holds: 0,
                carves: 0,
                placed: Placed::default(),
                zeros: takes::Ranges::new(),
            },
        );
        st.by_block.insert(block.id, at);
        drop(st);
        (self.host.add)(at, contents.as_ptr(), block.len);
        Some(at)
    }

    /// Takes `block`'s piece out before the block goes back: out of the lookups, then, once no
    /// hold is left, out of the guest (the module doc's "Release"). Returns once the host said it
    /// is out, or, after the device's release, once no hold is left.
    fn take_out(&self, block: BlockId) {
        let mut st = self.state.lock();
        if !st.by_block.contains_key(&block) {
            return;
        }
        let mut warned = false;
        let started = (self.now)();
        loop {
            let closed = st.closed;
            let Some(&at) = st.by_block.get(&block) else {
                return;
            };
            let Some(p) = st.pieces.get_mut(&at) else {
                return;
            };
            match p.mapped {
                Mapped::Adding | Mapped::Live => p.mapped = Mapped::Removing { asked: false },
                Mapped::Removing { asked: false } if p.holds == 0 && !closed => {
                    p.mapped = Mapped::Removing { asked: true };
                    let len = p.block.len;
                    drop(st);
                    (self.host.remove)(at, len);
                    st = self.state.lock();
                }
                Mapped::Removing { .. } if p.holds == 0 && closed => p.mapped = Mapped::Removed,
                Mapped::Removed if p.holds == 0 => {
                    st.pieces.remove(&at);
                    st.by_block.remove(&block);
                    return;
                }
                _ => {
                    let waited = (self.now)().saturating_duration_since(started);
                    if !warned && waited > Duration::from_secs(1) {
                        warned = true;
                        log::warn!(
                            "pool: the piece at {at:#x} still waits to go back after 1 s: {}",
                            if p.holds > 0 {
                                "a thread of the device's holds it"
                            } else {
                                "the host has not removed it"
                            }
                        );
                    }
                    let _ = self.changed.wait_for(&mut st, Duration::from_millis(250));
                }
            }
        }
    }

    /// [`Pool::close`]'s part, once the pool's thread has ended: waits for every hold on the
    /// pieces left to go, then takes them all out of the pool and answers their blocks. The host
    /// removed every piece before the release, so it is asked nothing.
    fn take_all_out(&self) -> Vec<BlockId> {
        let mut st = self.state.lock();
        let mut warned = false;
        let started = (self.now)();
        while let Some(&at) = (st.pieces.iter()).find_map(|(at, p)| (p.holds > 0).then_some(at)) {
            let waited = (self.now)().saturating_duration_since(started);
            if !warned && waited > Duration::from_secs(1) {
                warned = true;
                log::warn!(
                    "pool: the release still waits after 1 s for a thread of the device's that \
                     holds the piece at {at:#x}"
                );
            }
            let _ = self.changed.wait_for(&mut st, Duration::from_millis(250));
        }
        st.by_block.clear();
        st.carves.clear();
        st.takers.clear();
        let pieces = std::mem::take(&mut st.pieces);
        drop(st);
        pieces.into_values().map(|p| p.block.id).collect()
    }

    /// The device's release's end, once no ledger lives: every block goes back to `source`
    /// ([`Self::take_all_out`]), which is then let go of.
    fn give_all_back(&self, source: Arc<dyn BlockSource>) {
        for block in self.take_all_out() {
            if let Err(e) = source.release_block(block) {
                log::warn!(kind:? = e; "pool: the source refused a block back: {e:?}");
            }
        }
        drop(source);
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
