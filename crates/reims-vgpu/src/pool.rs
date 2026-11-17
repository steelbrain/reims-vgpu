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
//! [`Pool::find`] find it. A block the source hands out again after the pool gave it back keeps
//! its id, so its piece keeps its address and mapping, and the host is asked nothing.
//!
//! # Headroom and giving blocks back
//!
//! A new piece is empty: nothing of the guest's lies in it. The guest says when it puts something
//! in a piece ([`Pool::piece_used`]) and when it has emptied one ([`Pool::piece_empty`]). The pool
//! keeps one empty piece as headroom: when the guest uses the last empty one, the pool asks for a
//! block more. A piece that stays empty for [`RELEASE_DELAY`] while a lower one is empty too goes
//! back to the source ([`BlockSource::release_block`]), which keeps it ready, mapped, or frees it
//! through the unmap below; the lowest empty piece stays as the headroom. The delay rides out
//! churn, where a guest empties and fills a piece again within seconds.
//!
//! # Release
//!
//! The source calls [`BlockMappings::unmap`] before it frees any block it handed out, teardown
//! included. For a block of the pool's, unmap takes the piece out of the lookups, waits for every
//! hold on it ([`PieceHold`]) to go, asks the host to unmap it ([`PoolHost::remove`]), and returns
//! once the host says it is out of the guest's address space and no host thread of its own touches
//! it ([`Pool::removed`]). So no guest access, no thread of the host's and no thread of the
//! library's touches a block once the source frees it.
//!
//! At the device's release the host has removed every piece it added (QEMU does so as it tears
//! the device down, before the release): from then on ([`Pool::close`]) an unmap waits only for
//! the holds, and asks the host nothing.
//!
//! # The block source
//!
//! [`BlockSource`] and [`BlockMappings`] are metal2vulkan-rt's block calls and hook as its API
//! names them (`Device::take_block`, `release_block`, `BlockMappings::unmap`, [`BlockInfo`]).
//! Until the device runs on that runtime, the source is [`HostBlocks`], blocks of the process's
//! own memory, which hands blocks out, keeps them ready and frees them as the runtime does.

// Guest addresses flow through here: arithmetic is spelt out.
#![cfg_attr(not(test), deny(clippy::arithmetic_side_effects))]

mod host_blocks;

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ptr::NonNull;
use std::sync::{Arc, Weak};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

pub(crate) use host_blocks::HostBlocks;

use crate::Error;

/// A pooled block's length, metal2vulkan-rt's `BLOCK_LEN`: 256 MiB.
pub const BLOCK_LEN: u64 = 256 << 20;

/// The alignment of a piece's guest physical address and of a pool's range: 2 MiB, a large page
/// of the guest's and of the hypervisors' second-level tables.
pub const PIECE_ALIGN: u64 = 2 << 20;

/// How long a piece stays empty, with a lower piece empty too, before its block goes back to the
/// source (the module doc's "Headroom and giving blocks back").
pub const RELEASE_DELAY: Duration = Duration::from_secs(5);

/// A block's storage mode as the guest's buffers in it see it (metal2vulkan-rt's `MemoryClass`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
    pub fn get(self) -> u32 {
        self.0
    }
}

/// A block a source handed out (metal2vulkan-rt's `BlockInfo`): plain data.
#[derive(Clone, Copy, Debug)]
pub struct BlockInfo {
    /// Its number.
    pub id: BlockId,
    /// Its class.
    pub class: MemoryClass,
    /// Its length: [`BLOCK_LEN`] for a pooled block, more for a dedicated one.
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

/// Where the pool's blocks come from (metal2vulkan-rt's `Device::take_block` and
/// `release_block`).
pub trait BlockSource: Send + Sync {
    /// A block of `class` at least `len` long: a pooled one up to [`BLOCK_LEN`], a dedicated one
    /// past it. Waits only when no block is ready.
    fn take_block(&self, class: MemoryClass, len: u64) -> Result<BlockInfo, BlockError>;
    /// Gives a block back: it goes back to the ready set, with its id, address and mapping, or is
    /// freed after [`BlockMappings::unmap`].
    fn release_block(&self, block: BlockId) -> Result<(), BlockError>;
}

/// The hook a source calls before it frees any block it handed out, teardown included
/// (metal2vulkan-rt's `BlockMappings`).
pub trait BlockMappings: Send + Sync {
    /// Returns once every mapping of `block` made of it is gone and no thread touches it. Calls
    /// nothing of the source's.
    fn unmap(&self, block: BlockId);
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
            && self.physical_address % PIECE_ALIGN == 0
            && self.physical_length % PIECE_ALIGN == 0
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
}

/// `(physical_address, host_address, length)`: map `length` bytes of host memory at
/// `host_address` into the guest at `physical_address`, then say so ([`Pool::added`]). Called on
/// the pool's thread with no lock of the device's held; it must not wait for a lock the host holds
/// while it calls the device.
pub type PoolAddFn = Arc<dyn Fn(u64, *mut u8, u64) + Send + Sync>;

/// `(physical_address, length)`: unmap the piece at `physical_address`, then say so once no
/// thread of the host's touches it ([`Pool::removed`]). Called inside [`BlockMappings::unmap`],
/// on the source's thread, with no lock of the device's held; it must not wait for a lock the host
/// holds while it calls the device.
pub type PoolRemoveFn = Arc<dyn Fn(u64, u64) + Send + Sync>;

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
    /// Being unmapped, out of the lookups; `asked` once the host was asked to remove it.
    Removing { asked: bool },
    /// Out of the guest's address space, and no thread of the host's touches it.
    Removed,
}

/// One block in the range.
#[derive(Debug)]
struct Piece {
    block: BlockInfo,
    mapped: Mapped,
    /// The pool holds the block: taken, not given back. A piece given back stays mapped until the
    /// source unmaps it, and nothing finds it.
    held: bool,
    /// Since when nothing of the guest's lies in it; `None` while something does.
    empty_since: Option<Instant>,
    /// Holds on it ([`PieceHold`]).
    holds: u64,
}

impl Piece {
    /// Whether the guest may put something in it: held and not being unmapped.
    fn usable(&self) -> bool {
        self.held && matches!(self.mapped, Mapped::Adding | Mapped::Live)
    }
}

/// What the pool's lock holds.
#[derive(Default)]
struct State {
    /// By guest physical address.
    pieces: BTreeMap<u64, Piece>,
    /// Each piece's address, by its block.
    by_block: HashMap<BlockId, u64>,
    /// Asks for more pool, in bytes, oldest first.
    asks: VecDeque<u64>,
    /// The device was released: the host removed every piece it added.
    closed: bool,
    /// The pool's thread logged the range's being full once.
    full_logged: bool,
}

/// The pool's shared part: what the thread, the device and the source's hook hold.
struct Shared {
    range: PoolConfig,
    host: PoolHost,
    /// [`RELEASE_DELAY`], shorter in the tests.
    release_delay: Duration,
    state: Mutex<State>,
    changed: Condvar,
}

/// A hold on a live piece: while it lives, an unmap of its block waits.
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

/// The pool's [`BlockMappings`], which the source holds: weak, so the source, which the pool
/// holds, makes no cycle.
struct Mappings(Weak<Shared>);

impl BlockMappings for Mappings {
    fn unmap(&self, block: BlockId) {
        if let Some(shared) = self.0.upgrade() {
            shared.unmap(block);
        }
    }
}

/// The guest's Shared pool of one device.
pub struct Pool {
    shared: Arc<Shared>,
    /// The source. Dropped after the thread that takes from it ends ([`Pool::close`]); it frees
    /// its blocks once nothing else holds it, after unmapping each.
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
    /// A pool over `range`, its blocks from what `source` makes with the pool's hook, and its
    /// thread started. [`Error::InvalidArgument`] for a range [`PoolConfig::is_valid`] refuses,
    /// [`Error::HostResources`] where the thread is refused.
    pub(crate) fn new(
        range: PoolConfig,
        host: PoolHost,
        source: impl FnOnce(Arc<dyn BlockMappings>) -> Arc<dyn BlockSource>,
    ) -> Result<Pool, Error> {
        Pool::with_delay(range, host, RELEASE_DELAY, source)
    }

    /// [`Pool::new`], its pieces given back after `release_delay`.
    fn with_delay(
        range: PoolConfig,
        host: PoolHost,
        release_delay: Duration,
        source: impl FnOnce(Arc<dyn BlockMappings>) -> Arc<dyn BlockSource>,
    ) -> Result<Pool, Error> {
        if !range.is_valid() {
            return Err(Error::InvalidArgument);
        }
        let shared = Arc::new(Shared {
            range,
            host,
            release_delay,
            state: Mutex::new(State::default()),
            changed: Condvar::new(),
        });
        let source = source(Arc::new(Mappings(Arc::downgrade(&shared))));
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

    /// Asks for at least `length` more bytes of pool: the pool's thread takes a block and has the
    /// host map it. Waits for nothing.
    pub fn grow(&self, length: u64) {
        self.shared.state.lock().asks.push_back(length);
        self.shared.changed.notify_all();
    }

    /// The host mapped the piece at `physical_address` it was asked to add: lookups find it from
    /// now on. [`Error::InvalidArgument`] for an address the pool asked no add at, or one the host
    /// said it added already.
    pub fn added(&self, physical_address: u64) -> Result<(), Error> {
        let mut st = self.shared.state.lock();
        let p = (st.pieces.get_mut(&physical_address)).ok_or(Error::InvalidArgument)?;
        match p.mapped {
            Mapped::Adding => p.mapped = Mapped::Live,
            // An unmap came first: the host's remove follows its add.
            Mapped::Removing { .. } => {}
            Mapped::Live | Mapped::Removed => return Err(Error::InvalidArgument),
        }
        log::info!(
            "pool: {} MiB at {physical_address:#x} mapped into the guest",
            p.block.len >> 20
        );
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

    /// The guest put something in the piece at `physical_address`. Where no other piece is
    /// empty, the pool asks for a block more, so one stays as headroom.
    /// [`Error::InvalidArgument`] for an address where no piece the guest may use starts.
    pub fn piece_used(&self, physical_address: u64) -> Result<(), Error> {
        let mut st = self.shared.state.lock();
        let p = (st.pieces.get_mut(&physical_address)).ok_or(Error::InvalidArgument)?;
        if !p.usable() {
            return Err(Error::InvalidArgument);
        }
        p.empty_since = None;
        let headroom = (st.pieces.values()).any(|p| p.usable() && p.empty_since.is_some());
        if !headroom && st.asks.is_empty() {
            st.asks.push_back(BLOCK_LEN);
        }
        drop(st);
        self.shared.changed.notify_all();
        Ok(())
    }

    /// The guest emptied the piece at `physical_address`: no guest mapping of it is left (the
    /// kext unmapped every process's mapping into it and holds none of its own), and nothing of
    /// the guest's lies in it. From now on the guest puts nothing there until it says
    /// [`Pool::piece_used`]: [`RELEASE_DELAY`] later, if it is still empty and a lower piece is
    /// empty too, its block goes back to the source, and the guest may not use it again until the
    /// device hands it out anew. [`Error::InvalidArgument`] for an address where no piece the
    /// guest may use starts.
    pub fn piece_empty(&self, physical_address: u64) -> Result<(), Error> {
        let mut st = self.shared.state.lock();
        let p = (st.pieces.get_mut(&physical_address)).ok_or(Error::InvalidArgument)?;
        if !p.usable() {
            return Err(Error::InvalidArgument);
        }
        if p.empty_since.is_none() {
            p.empty_since = Some(Instant::now());
        }
        drop(st);
        self.shared.changed.notify_all();
        Ok(())
    }

    /// The live piece the guest may use that holds `length` bytes from `physical_address`, held
    /// until the answer drops; `None` where none holds all of them.
    pub fn find(&self, physical_address: u64, length: u64) -> Option<PieceHold> {
        let mut st = self.shared.state.lock();
        let (&start, p) = st.pieces.range_mut(..=physical_address).next_back()?;
        let offset = physical_address.checked_sub(start)?;
        let end = offset.checked_add(length)?;
        if p.mapped != Mapped::Live || !p.held || end > p.block.len {
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
    pub fn live(&self) -> Vec<(u64, u64)> {
        let st = self.shared.state.lock();
        (st.pieces.iter())
            .filter(|(_, p)| p.mapped == Mapped::Live && p.held)
            .map(|(&a, p)| (a, p.block.len))
            .collect()
    }

    /// The device's release: the host removed every piece it added. Ends the pool's thread, then
    /// lets go of the source; from now on an unmap waits only for the holds on its piece.
    pub(crate) fn close(&mut self) {
        self.shared.state.lock().closed = true;
        self.shared.changed.notify_all();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        self.source = None;
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
            if let Some(length) = st.asks.pop_front() {
                drop(st);
                match source.take_block(MemoryClass::Shared, length) {
                    Ok(block) => self.place(source, block),
                    Err(e) => log::warn!(kind:? = e; "pool: no block of {length} bytes: {e:?}"),
                }
                continue;
            }
            let now = Instant::now();
            let (due, next) = self.due(&st, now);
            if let Some((at, block)) = due {
                if let Some(p) = st.pieces.get_mut(&at) {
                    p.held = false;
                    p.empty_since = None;
                }
                drop(st);
                log::info!("pool: the empty piece at {at:#x} goes back to the source");
                if let Err(e) = source.release_block(block) {
                    log::warn!(kind:? = e; "pool: the source refused a block back: {e:?}");
                }
                continue;
            }
            match next {
                Some(when) => {
                    let _ = self.changed.wait_until(&mut st, when);
                }
                None => self.changed.wait(&mut st),
            }
        }
    }

    /// The piece whose block goes back now, if any, and when the next may: of the empty pieces
    /// the guest may use, every one but the lowest, once empty for the delay.
    fn due(&self, st: &State, now: Instant) -> (Option<(u64, BlockId)>, Option<Instant>) {
        let mut empty = (st.pieces.iter()).filter_map(|(&at, p)| {
            let since = p.empty_since.filter(|_| p.usable())?;
            Some((at, p.block.id, since.checked_add(self.release_delay)?))
        });
        // The lowest empty piece is the headroom.
        let _ = empty.next();
        let mut next: Option<Instant> = None;
        for (at, block, when) in empty {
            if when <= now {
                return (Some((at, block)), None);
            }
            next = Some(next.map_or(when, |n| n.min(when)));
        }
        (None, next)
    }

    /// Places `block` at the lowest free address that holds it and asks the host to map it; a
    /// block the pool placed before (given back, then taken again) keeps its piece, mapped.
    fn place(&self, source: &dyn BlockSource, block: BlockInfo) {
        let mut st = self.state.lock();
        if let Some(&at) = st.by_block.get(&block.id) {
            if let Some(p) = st.pieces.get_mut(&at) {
                p.held = true;
                p.empty_since = Some(Instant::now());
            }
            return;
        }
        let Some(contents) = block.contents else {
            drop(st);
            let _ = source.release_block(block.id);
            return;
        };
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
            return;
        };
        st.pieces.insert(
            at,
            Piece {
                block,
                mapped: Mapped::Adding,
                held: true,
                empty_since: Some(Instant::now()),
                holds: 0,
            },
        );
        st.by_block.insert(block.id, at);
        drop(st);
        (self.host.add)(at, contents.as_ptr(), block.len);
    }

    /// [`BlockMappings::unmap`] of the pool's: the module doc's "Release".
    fn unmap(&self, block: BlockId) {
        let mut st = self.state.lock();
        if !st.by_block.contains_key(&block) {
            return;
        }
        let mut warned = false;
        let started = Instant::now();
        loop {
            let closed = st.closed;
            let Some(&at) = st.by_block.get(&block) else {
                return;
            };
            let Some(p) = st.pieces.get_mut(&at) else {
                return;
            };
            match p.mapped {
                Mapped::Adding | Mapped::Live => {
                    p.mapped = Mapped::Removing { asked: false };
                    p.held = false;
                }
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
                    if !warned && started.elapsed() > Duration::from_secs(1) {
                        warned = true;
                        log::warn!(
                            "pool: the unmap of the piece at {at:#x} still waits after 1 s: {}",
                            if p.holds > 0 {
                                "a thread of the device's holds it"
                            } else {
                                "the host has not removed it"
                            }
                        );
                    }
                    let _ = self
                        .changed
                        .wait_for(&mut st, Duration::from_millis(250));
                }
            }
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
