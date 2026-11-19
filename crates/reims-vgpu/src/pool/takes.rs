//! The guest's contexts' blocks of the pool: each context (a process of the guest's) takes blocks
//! of the pool for its buffers' slabs, and gives each back once it is done with it
//! (`POOL_TAKE`, `POOL_GIVE`, docs/graphics-kext/design.md, "The Shared pool"). This is their
//! ledger: who holds which block, each a *carve* of a live piece, at a multiple of
//! [`TAKE_ALIGN`].
//!
//! # A take
//!
//! [`PoolLedger::take`] carves the block at once where a live piece has room (the lowest address
//! that fits, first fit), on the thread that carries the take out, with the pool's lock held and
//! nothing else done: it never allocates and never zeroes. The take then goes to its context's
//! worker (`reims-vgpu.graphics-pool-take`), which zeroes the block through the piece's host
//! mapping and answers. Where nothing fits, the worker asks for a piece of the take's length, waits
//! for the pool's thread to place it and the host to map it, and carves again. A context's takes
//! are answered in its own order, one worker a context with a take in flight, so a take waits only
//! behind its own context's earlier ones, never behind another's growth or zeroing. A worker goes
//! after [`WORKER_IDLE`] with no take. Every block reads zero at its answer: its worker zeroes
//! it at its take, off the thread that carried the take out, its bytes counted
//! ([`super::TAKE_ZEROED_BYTES_CODE`]), but for those the pool's zeroer zeroed after an earlier
//! carve's free (below, "Zeros after a free"). A piece's source says nothing of its block's
//! bytes as it hands it out, so every other byte of a carve is zeroed at its take.
//!
//! Each answered take's wait, from its carry-out to its answer, is counted by its cause
//! ([`super::TAKE_WAIT_CODES`]): behind its context's earlier takes (its worker's start among
//! them), for a piece of its own, and its worker's zeroing; the whole wait apart, as the three
//! leave out the waits for the pool's lock. Each is counted in nanoseconds in all and at its
//! longest, and a take whose whole wait is the longest yet is logged with its three.
//!
//! # Zeros after a free
//!
//! A freed carve's bytes go to the pool's zeroer (`reims-vgpu.graphics-pool-zero`, one a pool,
//! made at a free that finds none and gone after [`WORKER_IDLE`] with nothing to zero), which
//! zeroes them through the piece's mapping off every take's path, [`ZERO_CHUNK`] at a time, each
//! chunk with the pool's lock held, and records them in the piece as reading zero ([`Ranges`]),
//! counted ([`super::ZEROED_AFTER_FREE_BYTES_CODE`]), the longest it held the lock for a chunk
//! counted too ([`super::ZERO_HOLD_MAX_NS_CODE`]) with the longest of each of its parts, the
//! chunk's pick, its write and its mark, and each hold longer than [`super::ZERO_HOLD_LONG`]
//! counted with its parts' time ([`super::ZERO_HOLDS_LONG_CODE`]); it lets the lock go between
//! chunks. A carve
//! is made with the same lock held, so it never meets a chunk half zeroed: it takes its bytes out
//! of the record, and its worker zeroes only those the record lacks; the rest it leaves, counted ([`super::TAKE_ZERO_PREZEROED_BYTES_CODE`]). A carve's bytes still waiting for
//! the zeroer are no longer free, so the zeroer passes them over, and its worker zeroes them as
//! before. So the zeroer waits for no take worker and none waits for it, the pool's nor the
//! Private blocks' ([`super::private_blocks`], whose ledger, lock and workers it shares nothing
//! with): a holder of the pool's lock waits at most one chunk's zeroing for it, as for any other
//! holder. The zeroer zeroes nothing in a piece the guest emptied, which may go back whole, nor
//! in one going out: an empty piece's bytes are zeroed at its next take, as before. The device's
//! release ends the zeroer at its next chunk, and waits for it and for the take workers to let go
//! of the pool ([`super::Threads`]). The guest writes no freed carve's bytes: the kext takes a
//! block's mapping out of the process before its give, and the device frees it once the
//! context's command buffers have finished.
//!
//! A take ends only at its answer, its context's close or the device's reset: there is no
//! timeout. A take whose context closed, or that the reset ended, before its answer is given back
//! here and never answered.
//!
//! # A give, a close and a reset
//!
//! A give ([`PoolLedger::give`]) marks the block going: the device answers it once the context's
//! committed command buffers have finished, and frees it then ([`PoolLedger::free`]). A context's
//! close hands back every block it holds the same way ([`PoolLedger::context_closed`]), freed
//! once its objects are let go of and its command buffers have finished. The device's reset frees
//! every block at once ([`PoolLedger::reset`]), its objects already dropped. A block is freed only
//! by the number it was carved under, so a free that comes after a reset freed it frees nothing
//! carved there since.
//!
//! A piece holds its first carve from its use ([`State::use_piece`]) and is empty again once its
//! last carve is freed ([`State::empty_piece`]): the pool's headroom and release rules (the
//! module doc) then run on the takes alone.
//!
//! # Its GPU address
//!
//! Each carve has a GPU address from its carving, which its take's answer gives the guest
//! (`POOL_ADDRESSES`): where its piece's block has one (metal2vulkan-rt's), the block's address
//! at the carve's offset in it, so a heap the guest places there is a range of the block; where it
//! has none (`HostBlocks`), the host's pick ([`super::addresses`]), freed with the carve. A carve
//! with no room left for its pick is not carved, as one with no room in a piece.

use std::collections::{BTreeMap, VecDeque};
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use parking_lot::MutexGuard;
use reims_vgpu_vulkan::answer::count_kind_by;

use super::{
    Ask, Error, Live, Mapped, MonoTime, Shared, State, TAKE_WAIT_CODES, TAKE_WAITS_CODE,
    TAKE_ZERO_PREZEROED_BYTES_CODE, TAKE_ZEROED_BYTES_CODE, ZERO_HOLD_LONG, ZERO_HOLD_MAX_NS_CODE,
    ZERO_HOLD_PART_MAX_NS_CODES, ZERO_HOLDS_LONG_CODE, ZERO_LONG_HOLD_PART_NS_CODES,
    ZEROED_AFTER_FREE_BYTES_CODE,
};

/// A block's alignment and its length's granule: 64 KiB, the wire's `RGFX_POOL_BLOCK_ALIGN`.
pub const TAKE_ALIGN: u64 = 64 << 10;

/// The longest take: 4 GiB, the wire's `RGFX_POOL_TAKE_MAX`.
pub const TAKE_MAX: u64 = 1 << 32;

/// How long a context's take worker waits for its next take before it goes, and the zeroer for
/// its next bytes.
const WORKER_IDLE: Duration = Duration::from_secs(2);

/// The most bytes the zeroer zeroes at once, with the pool's lock held (the module doc's "Zeros
/// after a free"): 16 KiB, so a holder of the lock waits for little.
pub(super) const ZERO_CHUNK: u64 = 16 << 10;

/// Byte ranges of a piece's block, by offset: each range's first byte to its end, none touching
/// another.
pub(super) type Ranges = BTreeMap<u64, u64>;

/// Where a carve is in its life.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Carving {
    /// Carved for a take not yet answered: its worker zeroes it, and frees it where the take
    /// ended unanswered.
    Taking,
    /// The context's: answered.
    Held,
    /// Given back or let go of at its context's close, freed once the device's answer is due.
    Giving,
}

/// A context's block of a live piece.
#[derive(Debug)]
pub(super) struct Carve {
    context: u32,
    len: u64,
    serial: u64,
    state: Carving,
    /// The GPU address of its first byte (the module doc's "Its GPU address").
    gpu_address: GpuAddress,
    /// The bytes its worker zeroes, by offset from its first byte: those the zeroer did not zero
    /// after an earlier carve's free (the module doc's "Zeros after a free").
    zero: Vec<Range<u64>>,
}

/// Where a carve's GPU address comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GpuAddress {
    /// Its piece's block's address, at the carve's offset in it.
    OfBlock(u64),
    /// The host's pick, freed with the carve ([`super::addresses`]).
    Picked(u64),
}

impl GpuAddress {
    fn get(self) -> u64 {
        match self {
            GpuAddress::OfBlock(a) | GpuAddress::Picked(a) => a,
        }
    }
}

/// What came of a give ([`PoolLedger::give`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Give {
    /// The block is going, to be freed under this name.
    Given(CarveRef),
    /// The context holds no such block.
    NotFound,
    /// An object of the host's lies in it: it stays the context's.
    InUse,
}

/// A carve as a give or a close names it until it is freed ([`PoolLedger::free`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CarveRef {
    at: u64,
    serial: u64,
}

/// A take's answer: the block ([`Taken`]), or `None` where the pool has no room for it. Called
/// once, from the take's worker, with no lock of the pool's held; never for a take that ended
/// unanswered.
pub(crate) type TakeAnswer = Box<dyn FnOnce(Option<Taken>) + Send>;

/// A block a take gave its context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Taken {
    /// Its offset in the pool.
    pub(crate) offset: u64,
    /// Its length.
    pub(crate) length: u64,
    /// The GPU address of its first byte, never 0 (the module doc's "Its GPU address").
    pub(crate) gpu_address: u64,
}

/// A take waiting for its context's worker.
struct Job {
    /// Its context's takes' number when it was taken: the take is stale once they took another.
    life: u64,
    len: u64,
    /// Its carve, where one fit at the take: its address and number.
    carved: Option<(u64, u64)>,
    answer: TakeAnswer,
    /// When it was carried out, before the pool's lock: its wait's start.
    asked: MonoTime,
}

/// The longest the zeroer held the pool's lock for a chunk so far, in nanoseconds, which
/// [`ZERO_HOLD_MAX_NS_CODE`]'s count rises to.
static ZERO_HOLD_MAX_NS: AtomicU64 = AtomicU64::new(0);

/// The longest of each part of the zeroer's hold so far, in nanoseconds, which
/// [`ZERO_HOLD_PART_MAX_NS_CODES`]' counts rise to.
static ZERO_HOLD_PART_MAX_NS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// The longest of each of [`TAKE_WAIT_CODES`]' waits so far, in nanoseconds, which their
/// `-max-ns` codes' counts rise to.
static TAKE_WAIT_MAX_NS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Counts an answered take's wait ([`TAKE_WAIT_CODES`]): its whole, then behind its context's
/// earlier takes, for a piece of its own and zeroing; the take itself last
/// ([`TAKE_WAITS_CODE`]), once its waits are counted. Logs the take, `len` bytes of context
/// `context`'s, where its whole wait is the longest yet.
fn count_wait(context: u32, len: u64, waits: [Duration; 4]) {
    let mut longest = false;
    for (i, wait) in waits.iter().enumerate() {
        let ns = u64::try_from(wait.as_nanos()).unwrap_or(u64::MAX);
        let (all, max) = TAKE_WAIT_CODES[i];
        count_kind_by(all, ns);
        let was = TAKE_WAIT_MAX_NS[i].fetch_max(ns, Ordering::Relaxed);
        if ns > was {
            count_kind_by(max, ns.saturating_sub(was));
            longest |= i == 0;
        }
    }
    if longest {
        let ms = |i: usize| waits[i].as_secs_f64() * 1000.0;
        log::info!(
            "pool: context {context}'s take of {} KiB waited {:.2} ms, the longest yet: {:.2} ms \
             behind its earlier takes, {:.2} ms for a piece, {:.2} ms zeroing",
            len >> 10,
            ms(0),
            ms(1),
            ms(2),
            ms(3)
        );
    }
    count_kind_by(TAKE_WAITS_CODE, 1);
}

/// One hold of the pool's lock by the zeroer for a chunk, in its `parts`: the chunk's pick
/// (`State::next_zero`), its write and its mark (`State::zeroed`). Counted: the longest hold and
/// the longest of each part, and, where the hold is longer than [`ZERO_HOLD_LONG`], the hold
/// with its parts' nanoseconds ([`ZERO_HOLDS_LONG_CODE`]).
pub(super) fn zero_held(parts: [Duration; 3]) {
    let ns = parts.map(|part| u64::try_from(part.as_nanos()).unwrap_or(u64::MAX));
    let held = ns
        .iter()
        .fold(0, |all: u64, &part| all.saturating_add(part))
        .max(1);
    let was = ZERO_HOLD_MAX_NS.fetch_max(held, Ordering::Relaxed);
    if held > was {
        count_kind_by(ZERO_HOLD_MAX_NS_CODE, held.saturating_sub(was));
    }
    let mosts = ns.iter().zip(&ZERO_HOLD_PART_MAX_NS);
    for ((&part, most), code) in mosts.zip(ZERO_HOLD_PART_MAX_NS_CODES) {
        let was = most.fetch_max(part, Ordering::Relaxed);
        if part > was {
            count_kind_by(code, part.saturating_sub(was));
        }
    }
    if u128::from(held) > ZERO_HOLD_LONG.as_nanos() {
        count_kind_by(ZERO_HOLDS_LONG_CODE, 1);
        for (&part, code) in ns.iter().zip(ZERO_LONG_HOLD_PART_NS_CODES) {
            count_kind_by(code, part);
        }
    }
}

/// A context's takes.
pub(super) struct Taker {
    /// Their number, which the context's close and the device's reset change.
    life: u64,
    /// Its takes not yet taken up by its worker, in order.
    jobs: VecDeque<Job>,
    /// Whether its worker runs.
    working: bool,
}

/// What came of a carve that waited for the pool to grow.
enum Grown {
    /// Carved at this address, under this number, and how long the take waited for a piece of
    /// its own where it asked for one.
    At(u64, u64, Option<Duration>),
    /// The pool has no room for it.
    NoRoom,
    /// The take ended unanswered: its context closed, the device was reset or released.
    Stale,
}

/// A handle on the pool's ledger of the contexts' blocks. The device's release leaves every block
/// with the pool while one lives, and the last one's drop gives them back to the source
/// ([`super::Pool::close`]): the decoder holds one, and its heaps over the contexts' blocks may
/// outlive the device.
pub(crate) struct PoolLedger {
    shared: Arc<Shared>,
}

impl std::fmt::Debug for PoolLedger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PoolLedger")
            .field("range", &self.shared.range)
            .finish_non_exhaustive()
    }
}

impl Clone for PoolLedger {
    fn clone(&self) -> Self {
        PoolLedger::of(&self.shared)
    }
}

impl Drop for PoolLedger {
    fn drop(&mut self) {
        let mut st = self.shared.state.lock();
        st.keepers = st.keepers.saturating_sub(1);
        let left = if st.keepers == 0 {
            st.left.take()
        } else {
            None
        };
        drop(st);
        if let Some(source) = left {
            self.shared.give_all_back(source);
        }
    }
}

impl PoolLedger {
    /// A ledger over `shared`, counted among those that keep the blocks at the release.
    pub(super) fn of(shared: &Arc<Shared>) -> PoolLedger {
        let mut st = shared.state.lock();
        st.keepers = st.keepers.saturating_add(1);
        drop(st);
        PoolLedger {
            shared: Arc::clone(shared),
        }
    }

    /// The guest's driver took the pool: its processes' takes may come from now on, so the pool
    /// keeps one empty piece ahead of them, as it does once a piece is used (the module doc's
    /// "Headroom and giving blocks back"). It asks for a block where no piece the guest may use
    /// is empty and no ask waits: at a boot's take, not after a reset that left an empty piece.
    /// Whether it asked; waits for nothing.
    pub(crate) fn keep_headroom(&self) -> bool {
        let asked = self.shared.state.lock().keep_headroom();
        self.shared.changed.notify_all();
        log::info!(
            "pool: the guest's driver took the pool: {}",
            if asked {
                "a piece is asked ahead of its first take"
            } else {
                "a piece is empty or asked ahead of its first take"
            }
        );
        asked
    }

    /// The pool's first guest physical address, which a block's offset counts from.
    fn base(&self) -> u64 {
        self.shared.range.physical_address
    }

    /// `POOL_TAKE`: context `context` asks for a block of at least `length` bytes. Its block is
    /// carved now where a live piece has room, and `answer` is given once it is zeroed, after
    /// the context's earlier takes are answered (the module doc's "A take"); never where the
    /// device was released. [`Error::InvalidArgument`] for a `length` of 0 or past
    /// [`TAKE_MAX`], with `answer` dropped.
    pub(crate) fn take(&self, context: u32, length: u64, answer: TakeAnswer) -> Result<(), Error> {
        if length == 0 || length > TAKE_MAX {
            return Err(Error::InvalidArgument);
        }
        let len = length
            .checked_next_multiple_of(TAKE_ALIGN)
            .ok_or(Error::InvalidArgument)?;
        let asked = (self.shared.now)();
        let mut st = self.shared.state.lock();
        if st.closed {
            return Ok(());
        }
        if !st.takers.contains_key(&context) {
            let life = st.next_serial();
            st.takers.insert(
                context,
                Taker {
                    life,
                    jobs: VecDeque::new(),
                    working: false,
                },
            );
        }
        let carved = st.carve(context, len);
        let Some(taker) = st.takers.get_mut(&context) else {
            return Ok(());
        };
        taker.jobs.push_back(Job {
            life: taker.life,
            len,
            carved,
            answer,
            asked,
        });
        if taker.working {
            drop(st);
            self.shared.changed.notify_all();
            return Ok(());
        }
        taker.working = true;
        st.workers = st.workers.saturating_add(1);
        drop(st);
        // The carve may have asked the pool's thread for a block, as headroom.
        self.shared.changed.notify_all();
        let hold = ThreadHold::of(&self.shared);
        let spawned = std::thread::Builder::new()
            .name("reims-vgpu.graphics-pool-take".into())
            .spawn(move || hold.work(context));
        if spawned.is_err() {
            log::warn!("pool: no thread for context {context}'s takes: they are refused");
            let mut st = self.shared.state.lock();
            st.workers = st.workers.saturating_sub(1);
            let jobs = match st.takers.remove(&context) {
                Some(taker) => taker.jobs,
                None => VecDeque::new(),
            };
            let now = (self.shared.now)();
            for job in &jobs {
                if let Some((at, serial)) = job.carved {
                    st.uncarve(at, serial, now);
                }
            }
            drop(st);
            self.shared.changed.notify_all();
            for job in jobs {
                (job.answer)(None);
            }
        }
        Ok(())
    }

    /// `POOL_GIVE`: context `context` gives back its block of `length` bytes at pool offset
    /// `offset`, as its take answered it, where `in_use` says no object of the host's lies in its
    /// bytes (their pool offset and length). It is the context's no more, and is freed at
    /// [`Self::free`], once the context's committed command buffers have finished. `in_use` is
    /// asked with the pool's lock held, so no heap is made over the block
    /// ([`Self::over_held`]) between the answer and the give.
    pub(crate) fn give(
        &self,
        context: u32,
        offset: u64,
        length: u64,
        in_use: impl FnOnce(u64, u64) -> bool,
    ) -> Give {
        let Some(at) = self.base().checked_add(offset) else {
            return Give::NotFound;
        };
        let mut st = self.shared.state.lock();
        let held = st
            .carves
            .get(&at)
            .is_some_and(|c| c.context == context && c.len == length && c.state == Carving::Held);
        if !held {
            return Give::NotFound;
        }
        if in_use(offset, length) {
            return Give::InUse;
        }
        let Some(c) = st.carves.get_mut(&at) else {
            return Give::NotFound;
        };
        c.state = Carving::Giving;
        Give::Given(CarveRef {
            at,
            serial: c.serial,
        })
    }

    /// Frees the blocks `carves` names, which a give or a close let go of, where the device's
    /// reset has not freed them since: a piece whose last block this was is empty again, and the
    /// bytes of one in a piece still holding a carve go to the zeroer (the module doc's "Zeros
    /// after a free"), made here where none runs.
    pub(crate) fn free(&self, carves: &[CarveRef]) {
        if carves.is_empty() {
            return;
        }
        let now = (self.shared.now)();
        let mut st = self.shared.state.lock();
        for c in carves {
            st.free_carve(c.at, c.serial, now);
        }
        let start = st.start_zeroer();
        drop(st);
        self.shared.changed.notify_all();
        if start {
            spawn_zeroer(&self.shared);
        }
    }

    /// Context `context` closed: its takes not yet answered end unanswered, and its blocks are
    /// let go of, returned to be freed ([`Self::free`]) once its objects are let go of and its
    /// command buffers have finished.
    pub(crate) fn context_closed(&self, context: u32) -> Vec<CarveRef> {
        let now = (self.shared.now)();
        let mut st = self.shared.state.lock();
        let ended = st.end_takes(Some(context), now);
        let mut gone = Vec::new();
        for (&at, c) in &mut st.carves {
            if c.context == context && c.state == Carving::Held {
                c.state = Carving::Giving;
            }
            if c.context == context && c.state == Carving::Giving {
                gone.push(CarveRef {
                    at,
                    serial: c.serial,
                });
            }
        }
        drop(st);
        self.shared.changed.notify_all();
        drop(ended);
        gone
    }

    /// The device's reset: every take not yet answered ends unanswered, and every block is freed
    /// now, every object of the guest's already dropped.
    pub(crate) fn reset(&self) {
        let now = (self.shared.now)();
        let mut st = self.shared.state.lock();
        let ended = st.end_takes(None, now);
        let freed: Vec<(u64, u64)> = (st.carves.iter())
            .filter(|(_, c)| c.state != Carving::Taking)
            .map(|(&at, c)| (at, c.serial))
            .collect();
        for (at, serial) in freed {
            st.uncarve(at, serial, now);
        }
        drop(st);
        self.shared.changed.notify_all();
        drop(ended);
    }

    /// What `f` makes of the host address of the `length` bytes from pool offset `offset`, where
    /// they lie in one block context `context` holds, called with the pool's lock held, so the
    /// block is not given back ([`Self::give`]) before what `f` makes over it is in the decoder's
    /// tables. `None`, with `f` not called, where they do not.
    pub(crate) fn over_held<R>(
        &self,
        context: u32,
        offset: u64,
        length: u64,
        f: impl FnOnce(NonNull<u8>) -> R,
    ) -> Option<R> {
        let at = self.base().checked_add(offset)?;
        let end = at.checked_add(length)?;
        let st = self.shared.state.lock();
        let (&carve_at, c) = st.carves.range(..=at).next_back()?;
        if c.context != context || c.state != Carving::Held || end > carve_at.checked_add(c.len)? {
            return None;
        }
        let host = st.host_of(at)?;
        Some(f(host))
    }

    /// The pool's guest physical range: its first address and its length.
    pub(crate) fn range(&self) -> (u64, u64) {
        (
            self.shared.range.physical_address,
            self.shared.range.physical_length,
        )
    }

    /// The blocks the contexts hold, taken or being taken, and their bytes (for the tests).
    #[cfg(test)]
    pub(crate) fn counts(&self) -> (usize, u64) {
        let st = self.shared.state.lock();
        let bytes = st.carves.values().map(|c| c.len).sum();
        (st.carves.len(), bytes)
    }

    /// The GPU addresses the host picked and has not freed (for the tests).
    #[cfg(test)]
    pub(crate) fn picked(&self) -> usize {
        self.shared.state.lock().addresses.held()
    }
}

impl State {
    /// The host address of the byte at guest physical address `at`, in the live piece that holds
    /// it.
    fn host_of(&self, at: u64) -> Option<NonNull<u8>> {
        let (&start, p) = self.pieces.range(..=at).next_back()?;
        let offset = at.checked_sub(start)?;
        if offset >= p.block.len {
            return None;
        }
        let contents = p.block.contents?;
        let offset = usize::try_from(offset).ok()?;
        // SAFETY: `offset` is inside the block's mapping, checked above.
        Some(unsafe { NonNull::new_unchecked(contents.as_ptr().add(offset)) })
    }

    /// Frees the carve at `at` where it is still the one numbered `serial`, at `now`: its piece
    /// is empty again where it was the last.
    fn uncarve(&mut self, at: u64, serial: u64, now: MonoTime) {
        if self.carves.get(&at).is_none_or(|c| c.serial != serial) {
            return;
        }
        if let Some(Carve {
            gpu_address: GpuAddress::Picked(address),
            ..
        }) = self.carves.remove(&at)
        {
            self.addresses.free(address);
        }
        let Some((&start, p)) = self.pieces.range_mut(..=at).next_back() else {
            return;
        };
        p.carves = p.carves.saturating_sub(1);
        if p.carves == 0 {
            self.empty_piece(start, now);
        }
    }

    /// Carves `len` bytes for `context` at the lowest address of a live piece that has room for
    /// them, first fit, and marks the piece used where it was empty: the address and the carve's
    /// number, or `None` where no live piece has room.
    fn carve(&mut self, context: u32, len: u64) -> Option<(u64, u64)> {
        let at = (self.pieces.iter())
            .filter(|(_, p)| p.mapped == Mapped::Live && p.block.len >= len)
            .find_map(|(&start, p)| {
                let end = start.checked_add(p.block.len)?;
                let mut at = start;
                for (&carve_at, c) in self.carves.range(start..end) {
                    if at.checked_add(len)? <= carve_at {
                        break;
                    }
                    at = at.max(carve_at.checked_add(c.len)?);
                }
                (at.checked_add(len)? <= end).then_some((start, at))
            });
        let (start, at) = at?;
        let piece = self.pieces.get_mut(&start)?;
        let block = piece.block.gpu_address;
        let offset = at.checked_sub(start)?;
        let end = offset.checked_add(len)?;
        // Of the carve's bytes, those the zeroer zeroed since an earlier carve's free are zero;
        // the rest its worker zeroes. They are the carve's now, out of the record.
        let zero = missing(&piece.zeros, offset..end)
            .into_iter()
            .map(|r| r.start.saturating_sub(offset)..r.end.saturating_sub(offset))
            .collect();
        remove(&mut piece.zeros, offset..end);
        let gpu_address = if block != 0 {
            GpuAddress::OfBlock(block.checked_add(offset)?)
        } else {
            GpuAddress::Picked(self.addresses.pick(len)?)
        };
        let serial = self.next_serial();
        self.carves.insert(
            at,
            Carve {
                context,
                len,
                serial,
                state: Carving::Taking,
                gpu_address,
                zero,
            },
        );
        let p = self.pieces.get_mut(&start)?;
        p.carves = p.carves.saturating_add(1);
        if p.carves == 1 {
            self.use_piece(start);
        }
        Some((at, serial))
    }

    /// Frees the carve at `at` numbered `serial` ([`Self::uncarve`]): its bytes go to the
    /// zeroer where its piece is live and still holds a carve (the module doc's "Zeros after a
    /// free").
    fn free_carve(&mut self, at: u64, serial: u64, now: MonoTime) {
        let freed = (self.carves.get(&at))
            .filter(|c| c.serial == serial)
            .map(|c| c.len);
        self.uncarve(at, serial, now);
        let Some(len) = freed else {
            return;
        };
        let Some((&start, p)) = self.pieces.range(..=at).next_back() else {
            return;
        };
        if p.carves == 0 || p.mapped != Mapped::Live {
            return;
        }
        if let Some(offset) = at.checked_sub(start)
            && let Some(end) = offset.checked_add(len)
        {
            self.to_zero.push_back((start, offset..end));
        }
    }

    /// Whether a zeroer is to be made: bytes wait for one, none runs and the device is not
    /// released. Counted among the workers where it is.
    fn start_zeroer(&mut self) -> bool {
        let start = !self.zeroer && !self.closed && !self.to_zero.is_empty();
        if start {
            self.zeroer = true;
            self.workers = self.workers.saturating_add(1);
        }
        start
    }

    /// The zeroer's next bytes, its piece's address, their offsets in its block and their host
    /// address: at most [`ZERO_CHUNK`] of the first bytes waiting, in a live piece that holds a
    /// carve, carved by none and not zero already. `None` where
    /// none wait. They stay so while the pool's lock is held.
    pub(super) fn next_zero(&mut self) -> Option<(u64, Range<u64>, NonNull<u8>)> {
        while let Some((piece, range)) = self.to_zero.pop_front() {
            let Some(p) = self.pieces.get(&piece) else {
                continue;
            };
            let Some(contents) = p.block.contents.filter(|_| p.mapped == Mapped::Live) else {
                continue;
            };
            if p.carves == 0 {
                continue;
            }
            let range = range.start..range.end.min(p.block.len);
            let free = self.uncarved(piece, missing(&p.zeros, range.clone()));
            let Some(first) = free.first() else {
                continue;
            };
            let chunk = first.start..first.end.min(first.start.saturating_add(ZERO_CHUNK));
            let Ok(offset) = usize::try_from(chunk.start) else {
                continue;
            };
            if chunk.end < range.end {
                self.to_zero.push_front((piece, chunk.end..range.end));
            }
            // SAFETY: the chunk lies below the block's length, inside its mapping.
            let host = unsafe { NonNull::new_unchecked(contents.as_ptr().add(offset)) };
            return Some((piece, chunk, host));
        }
        None
    }

    /// The zeroer zeroed `range` of the block of the piece at `piece`, with the pool's lock held
    /// since [`Self::next_zero`]: its bytes read zero.
    pub(super) fn zeroed(&mut self, piece: u64, range: Range<u64>) {
        if let Some(p) = self.pieces.get_mut(&piece) {
            insert(&mut p.zeros, range);
        }
    }

    /// `parts`, offsets in the block of the piece at `piece`, less the bytes its carves hold.
    fn uncarved(&self, piece: u64, mut parts: Vec<Range<u64>>) -> Vec<Range<u64>> {
        let Some(p) = self.pieces.get(&piece) else {
            return Vec::new();
        };
        let end = piece.saturating_add(p.block.len);
        for (&at, c) in self.carves.range(piece..end) {
            let first = at.saturating_sub(piece);
            parts = cut(parts, first..first.saturating_add(c.len));
        }
        parts
    }

    /// Whether the takes of `context` numbered `life` still stand: neither its close nor the
    /// device's reset or release came since.
    fn stands(&self, context: u32, life: u64) -> bool {
        !self.closed && self.takers.get(&context).is_some_and(|t| t.life == life)
    }

    /// Ends the takes of `context`, or of every context, not yet answered: their contexts' takes
    /// take a new number, so the take a worker carries out is stale, and those waiting for the
    /// worker are dropped here, their carves freed. Their answers are returned, to drop with no
    /// lock held.
    fn end_takes(&mut self, context: Option<u32>, now: MonoTime) -> Vec<TakeAnswer> {
        let contexts: Vec<u32> = match context {
            Some(context) => vec![context],
            None => self.takers.keys().copied().collect(),
        };
        let mut ended = Vec::new();
        for context in contexts {
            let life = self.next_serial();
            let Some(taker) = self.takers.get_mut(&context) else {
                continue;
            };
            taker.life = life;
            let jobs = std::mem::take(&mut taker.jobs);
            if !taker.working {
                self.takers.remove(&context);
            }
            for job in jobs {
                if let Some((at, serial)) = job.carved {
                    self.uncarve(at, serial, now);
                }
                ended.push(job.answer);
            }
        }
        ended
    }
}

impl Shared {
    /// Context `context`'s take worker: takes up its takes in order, carves each that did not fit
    /// at its take (growing the pool), zeroes it and answers, until its takes stay none for
    /// [`WORKER_IDLE`] or the device is released.
    fn work(&self, context: u32) {
        let mut st = self.state.lock();
        let mut idle_since = (self.now)();
        loop {
            if st.closed {
                break;
            }
            let job = st.takers.get_mut(&context).and_then(|t| t.jobs.pop_front());
            let Some(job) = job else {
                let idle = (self.now)().saturating_duration_since(idle_since);
                match WORKER_IDLE.checked_sub(idle).filter(|left| !left.is_zero()) {
                    Some(left) => {
                        let _ = self.changed.wait_for(&mut st, left);
                        continue;
                    }
                    None => break,
                }
            };
            let started = (self.now)();
            let carved = match job.carved {
                Some(carved) => Grown::At(carved.0, carved.1, None),
                None => self.carve_grown(&mut st, context, job.life, job.len),
            };
            let grown = (self.now)().saturating_duration_since(started);
            let (at, serial, waited) = match carved {
                Grown::At(at, serial, waited) => (at, serial, waited),
                Grown::NoRoom => {
                    MutexGuard::unlocked(&mut st, || {
                        log::warn!(
                            "pool: no room for context {context}'s block of {} KiB: the take is \
                             refused",
                            job.len >> 10
                        );
                        (job.answer)(None);
                    });
                    idle_since = (self.now)();
                    continue;
                }
                Grown::Stale => {
                    idle_since = (self.now)();
                    continue;
                }
            };
            let host = st.host_of(at);
            let carve = st.carves.get(&at).filter(|c| c.serial == serial);
            let zero =
                carve.map_or_else(|| std::iter::once(0..job.len).collect(), |c| c.zero.clone());
            let zeroing = (self.now)();
            MutexGuard::unlocked(&mut st, || {
                let Some(host) = host else {
                    return;
                };
                for r in &zero {
                    let first = usize::try_from(r.start);
                    let bytes = usize::try_from(r.end.saturating_sub(r.start));
                    if let (Ok(first), Ok(bytes)) = (first, bytes) {
                        // SAFETY: the carve lies in its live piece's mapping, and the piece's
                        // block stays the pool's while the carve does, which only this worker
                        // frees until it is answered; `r` lies in its `job.len` bytes.
                        unsafe { std::ptr::write_bytes(host.as_ptr().add(first), 0, bytes) };
                    }
                }
                let zeroes = total(&zero);
                if zeroes > 0 {
                    count_kind_by(TAKE_ZEROED_BYTES_CODE, zeroes);
                }
                if job.len > zeroes {
                    count_kind_by(
                        TAKE_ZERO_PREZEROED_BYTES_CODE,
                        job.len.saturating_sub(zeroes),
                    );
                }
            });
            let zeroed = (self.now)().saturating_duration_since(zeroing);
            let stands = st.stands(context, job.life);
            if stands
                && let Some(c) = st.carves.get_mut(&at)
                && c.serial == serial
            {
                c.state = Carving::Held;
                let taken = Taken {
                    offset: at.saturating_sub(self.range.physical_address),
                    length: job.len,
                    gpu_address: c.gpu_address.get(),
                };
                MutexGuard::unlocked(&mut st, || {
                    let whole = (self.now)().saturating_duration_since(job.asked);
                    let queued = started.saturating_duration_since(job.asked);
                    (job.answer)(Some(taken));
                    count_wait(context, job.len, [whole, queued, grown, zeroed]);
                    // Each wait for a piece, in every boot's log: the first line, then a count.
                    if let Some(waited) = waited {
                        log::info!(
                            kind:% = "pool-take-waited";
                            "pool: context {context}'s take of {} KiB waited {:.1} ms for a piece",
                            job.len >> 10,
                            waited.as_secs_f64() * 1000.0
                        );
                    }
                });
            } else {
                let now = (self.now)();
                st.uncarve(at, serial, now);
                self.changed.notify_all();
                MutexGuard::unlocked(&mut st, || drop(job.answer));
            }
            idle_since = (self.now)();
        }
        // Gone: a take handed to it after this starts another.
        let closed = st.closed;
        if let Some(taker) = st.takers.get_mut(&context) {
            taker.working = false;
            if closed {
                taker.jobs.clear();
            }
            if taker.jobs.is_empty() {
                st.takers.remove(&context);
            }
        }
        st.workers = st.workers.saturating_sub(1);
        drop(st);
        self.changed.notify_all();
    }

    /// A carve of `len` bytes for the take of `context` numbered `life`, which none fitted at its
    /// take: asks the pool for a piece of `len`, waits for its thread to place it and the host to
    /// map it, and carves again, until a carve fits, no piece comes, or the take is stale.
    fn carve_grown(
        &self,
        st: &mut MutexGuard<'_, State>,
        context: u32,
        life: u64,
        len: u64,
    ) -> Grown {
        let mut asked: Option<MonoTime> = None;
        loop {
            if !st.stands(context, life) {
                return Grown::Stale;
            }
            if let Some((at, serial)) = st.carve(context, len) {
                // The carve may have asked the pool's thread for a block, as headroom.
                self.changed.notify_all();
                let waited = asked.map(|since| (self.now)().saturating_duration_since(since));
                return Grown::At(at, serial, waited);
            }
            asked = asked.or_else(|| Some((self.now)()));
            let wait = st.next_serial();
            st.waits.insert(wait, None);
            st.asks.push_back(Ask {
                length: len,
                wait: Some(wait),
            });
            self.changed.notify_all();
            let placed = loop {
                if !st.stands(context, life) {
                    st.waits.remove(&wait);
                    return Grown::Stale;
                }
                if let Some(&Some(placed)) = st.waits.get(&wait) {
                    st.waits.remove(&wait);
                    break placed;
                }
                self.changed.wait(st);
            };
            let Some(piece) = placed else {
                return Grown::NoRoom;
            };
            // Once the host mapped it, or it went: a carve looks at live pieces alone.
            while st
                .pieces
                .get(&piece)
                .is_some_and(|p| p.mapped == Mapped::Adding)
            {
                if !st.stands(context, life) {
                    return Grown::Stale;
                }
                self.changed.wait(st);
            }
        }
    }

    /// The pool's zeroer (the module doc's "Zeros after a free"): zeroes the freed carves' bytes
    /// waiting, a chunk at a time with the pool's lock held, letting it go between chunks, until
    /// none wait for [`WORKER_IDLE`] or the device is released.
    fn zero_work(&self) {
        let mut st = self.state.lock();
        let mut idle_since = (self.now)();
        loop {
            if st.closed {
                break;
            }
            let picking = (self.now)();
            let Some((piece, range, host)) = st.next_zero() else {
                let idle = (self.now)().saturating_duration_since(idle_since);
                match WORKER_IDLE.checked_sub(idle).filter(|left| !left.is_zero()) {
                    Some(left) => {
                        let _ = self.changed.wait_for(&mut st, left);
                        continue;
                    }
                    None => break,
                }
            };
            let picked = (self.now)();
            let len = range.end.saturating_sub(range.start);
            let Ok(bytes) = usize::try_from(len) else {
                continue;
            };
            // SAFETY: the bytes lie in their live piece's mapping, and no carve holds them; the
            // pool's lock, held since `next_zero`, keeps the piece live and any carve off them, so
            // neither the guest nor the device reaches them meanwhile.
            unsafe { std::ptr::write_bytes(host.as_ptr(), 0, bytes) };
            let written = (self.now)();
            st.zeroed(piece, range);
            let marked = (self.now)();
            let parts = [
                picked.saturating_duration_since(picking),
                written.saturating_duration_since(picked),
                marked.saturating_duration_since(written),
            ];
            MutexGuard::unlocked(&mut st, || {
                count_kind_by(ZEROED_AFTER_FREE_BYTES_CODE, len);
                zero_held(parts);
            });
            idle_since = (self.now)();
        }
        st.zeroer = false;
        if st.closed {
            st.to_zero.clear();
        }
        st.workers = st.workers.saturating_sub(1);
        drop(st);
        self.changed.notify_all();
    }

    /// Waits for every take worker to go: they go once the device's release closed the pool.
    pub(super) fn wait_for_workers(&self) {
        let mut st = self.state.lock();
        while st.workers > 0 {
            let _ = self.changed.wait_for(&mut st, Duration::from_millis(250));
        }
    }
}

/// What a pool thread holds, a take worker's or the zeroer's: the pool, then its count among the
/// [`super::Threads`], which a struct's fields' order drops in that order, at the thread's return or
/// unwind, so the device's release returns only once the thread let go of the pool.
struct ThreadHold {
    shared: Arc<Shared>,
    _live: Live,
}

impl ThreadHold {
    /// A hold on `shared`, counted among its threads.
    fn of(shared: &Arc<Shared>) -> ThreadHold {
        ThreadHold {
            shared: Arc::clone(shared),
            _live: Live::new(&shared.threads),
        }
    }

    /// Runs context `context`'s take worker ([`Shared::work`]), then lets go of the pool, then of
    /// the thread's count.
    fn work(self, context: u32) {
        self.shared.work(context);
    }

    /// Runs the zeroer ([`Shared::zero_work`]), then lets go of the pool, then of the thread's
    /// count.
    fn zero(self) {
        self.shared.zero_work();
    }
}

/// Makes the pool's zeroer ([`Shared::zero_work`]), which [`State::start_zeroer`] counted; where
/// no thread is made, the bytes waiting are let go of, and their takes zero them as before.
fn spawn_zeroer(shared: &Arc<Shared>) {
    let hold = ThreadHold::of(shared);
    let spawned = std::thread::Builder::new()
        .name("reims-vgpu.graphics-pool-zero".into())
        .spawn(move || hold.zero());
    if spawned.is_err() {
        log::warn!("pool: no thread to zero freed blocks: their next takes zero them");
        let mut st = shared.state.lock();
        st.zeroer = false;
        st.to_zero.clear();
        st.workers = st.workers.saturating_sub(1);
        drop(st);
        shared.changed.notify_all();
    }
}

/// The bytes of `parts`.
fn total(parts: &[Range<u64>]) -> u64 {
    (parts.iter())
        .map(|r| r.end.saturating_sub(r.start))
        .fold(0, u64::saturating_add)
}

/// The parts of `range` that `set` lacks, in order.
fn missing(set: &Ranges, range: Range<u64>) -> Vec<Range<u64>> {
    let mut parts = Vec::new();
    if range.start >= range.end {
        return parts;
    }
    let mut at = range.start;
    let before = set.range(..range.start).next_back();
    for (&first, &end) in before.into_iter().chain(set.range(range.start..range.end)) {
        if end <= at {
            continue;
        }
        if first > at {
            parts.push(at..first);
        }
        at = at.max(end);
        if at >= range.end {
            return parts;
        }
    }
    parts.push(at..range.end);
    parts
}

/// `parts` less the bytes of `cut`.
fn cut(parts: Vec<Range<u64>>, cut: Range<u64>) -> Vec<Range<u64>> {
    let mut left = Vec::with_capacity(parts.len().saturating_add(1));
    for part in parts {
        if part.end <= cut.start || cut.end <= part.start {
            left.push(part);
            continue;
        }
        if part.start < cut.start {
            left.push(part.start..cut.start);
        }
        if cut.end < part.end {
            left.push(cut.end..part.end);
        }
    }
    left
}

/// Takes the bytes of `range` out of `set`.
fn remove(set: &mut Ranges, range: Range<u64>) {
    if range.start >= range.end {
        return;
    }
    let before = (set.range(..range.start).next_back()).map(|(&first, &end)| (first, end));
    let within: Vec<(u64, u64)> = (set.range(range.start..range.end))
        .map(|(&first, &end)| (first, end))
        .collect();
    for (first, end) in before.into_iter().chain(within) {
        if end <= range.start {
            continue;
        }
        set.remove(&first);
        if first < range.start {
            set.insert(first, range.start);
        }
        if range.end < end {
            set.insert(range.end, end);
        }
    }
}

/// Puts the bytes of `range` in `set`, joined with the ranges they overlap or touch.
fn insert(set: &mut Ranges, range: Range<u64>) {
    if range.start >= range.end {
        return;
    }
    let (mut first, mut end) = (range.start, range.end);
    let joined: Vec<(u64, u64)> = (set.range(..=range.end).rev())
        .take_while(|&(_, &e)| e >= range.start)
        .map(|(&f, &e)| (f, e))
        .collect();
    for (f, e) in joined {
        set.remove(&f);
        first = first.min(f);
        end = end.max(e);
    }
    set.insert(first, end);
}
