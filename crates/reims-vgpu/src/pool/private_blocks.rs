//! The guest's contexts' Private blocks: each context (a process of the guest's) takes blocks of
//! GPU addresses for its Private objects and gives each back once it is done with it
//! (`PRIVATE_TAKE`, `PRIVATE_GIVE`, docs/graphics-kext/design.md, "Private blocks"). This is
//! their ledger: who holds which block, by its address.
//!
//! # A block
//!
//! On a source that gives Private blocks (metal2vulkan-rt's device), a take is one block of the
//! source's ([`BlockSource::take_block`] of [`MemoryClass::Private`]): device memory no one maps,
//! at the block's own GPU address, the whole of it the context's (a pooled block is the source's
//! `BLOCK_LEN` long whatever the take's length). The source may wait for the memory, on the
//! context's lane, where the context waits for the answer anyway. On a source with none
//! (`HostBlocks`, the Vulkan platform's), a block is a range of addresses the host picks
//! ([`HostAddresses::private`], `[2^44, 2^45)`) with no memory of its own, the take's length
//! rounded up to [`TAKE_ALIGN`]: the platform makes each object the guest places there as it makes
//! one at the guest's own picks, and pins it at its address. Nothing here zeroes anything: the
//! decoder makes each Private buffer zeroed at its create.
//!
//! # A give, a close and a reset
//!
//! A give ([`PrivateBlocks::give`]) marks the block going, where no object of the host's lies in
//! it: the device answers it once the context's committed command buffers have finished, and
//! frees it then ([`PrivateBlocks::free`]). A context's close hands back every block it holds the
//! same way ([`PrivateBlocks::context_closed`]), freed once its objects are let go of and its
//! command buffers have finished. The device's reset frees every block at once
//! ([`PrivateBlocks::reset`]), its objects already dropped. A block is freed only by the number it
//! was taken under, so a free that comes after a reset freed it frees nothing taken since.

use std::collections::BTreeMap;
use std::sync::Arc;

use parking_lot::Mutex;

use super::addresses::HostAddresses;
use super::{BlockId, BlockSource, MemoryClass};

/// A block's length is a multiple of this: 64 KiB, the wire's `RGFX_PRIVATE_BLOCK_ALIGN`.
pub const TAKE_ALIGN: u64 = 64 << 10;

/// The longest take: 1 TiB, the wire's `RGFX_PRIVATE_TAKE_MAX`.
pub const TAKE_MAX: u64 = 1 << 40;

/// Why a take got no block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TakeError {
    /// A length of 0 or past [`TAKE_MAX`].
    Invalid,
    /// No block or address left for it.
    Refused,
}

/// What came of a give ([`PrivateBlocks::give`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PrivateGive {
    /// The block is going, to be freed under this name.
    Given(HeldRef),
    /// The context holds no such block.
    NotFound,
    /// An object of the host's lies in it: it stays the context's.
    InUse,
}

/// A block as a give or a close names it until it is freed ([`PrivateBlocks::free`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HeldRef {
    at: u64,
    serial: u64,
}

/// A context's block.
#[derive(Debug)]
struct Held {
    context: u32,
    len: u64,
    serial: u64,
    /// Given back or let go of at its context's close, freed once the device's answer is due.
    going: bool,
    /// The source's block it is, where the source gave one; `None` for the host's pick.
    block: Option<BlockId>,
}

/// What the ledger's lock holds.
#[derive(Debug)]
struct State {
    /// The addresses picked for blocks where the source gives none.
    addresses: HostAddresses,
    /// The blocks held, by address.
    held: BTreeMap<u64, Held>,
    /// The number the next block is taken under.
    serial: u64,
}

/// The contexts' Private blocks ([module doc](self)): a handle, which clones share.
#[derive(Clone)]
pub(crate) struct PrivateBlocks {
    /// The source of Private blocks, where it gives them.
    source: Option<Arc<dyn BlockSource>>,
    state: Arc<Mutex<State>>,
}

impl PrivateBlocks {
    /// A ledger with no block held, whose blocks are `source`'s, or the host's picks where it
    /// has none.
    pub(crate) fn new(source: Option<Arc<dyn BlockSource>>) -> PrivateBlocks {
        PrivateBlocks {
            source,
            state: Arc::new(Mutex::new(State {
                addresses: HostAddresses::private(),
                held: BTreeMap::new(),
                serial: 0,
            })),
        }
    }

    /// Context `context`'s new block of at least `length` bytes: its GPU address and its length, a
    /// multiple of [`TAKE_ALIGN`]: `length` rounded up to it, or the source's block's whole length
    /// rounded down to it.
    pub(crate) fn take(&self, context: u32, length: u64) -> Result<(u64, u64), TakeError> {
        if length == 0 || length > TAKE_MAX {
            return Err(TakeError::Invalid);
        }
        let len = length
            .checked_next_multiple_of(TAKE_ALIGN)
            .ok_or(TakeError::Invalid)?;
        // The source's block, taken with no lock of the ledger's held.
        let block = match &self.source {
            Some(source) => {
                let info = (source.take_block(MemoryClass::Private, len))
                    .map_err(|_| TakeError::Refused)?;
                if info.gpu_address == 0 || info.len < len {
                    let _ = source.release_block(info.id);
                    return Err(TakeError::Refused);
                }
                // `len` is a multiple of TAKE_ALIGN no longer than the block's length.
                let whole = (info.len.checked_rem(TAKE_ALIGN))
                    .and_then(|rest| info.len.checked_sub(rest))
                    .unwrap_or(len);
                Some((info.id, info.gpu_address, whole))
            }
            None => None,
        };
        let mut state = self.state.lock();
        let (at, len) = match block {
            Some((_, at, whole)) => (at, whole),
            None => (state.addresses.pick(len).ok_or(TakeError::Refused)?, len),
        };
        state.serial = state.serial.wrapping_add(1);
        let serial = state.serial;
        state.held.insert(
            at,
            Held {
                context,
                len,
                serial,
                going: false,
                block: block.map(|(id, ..)| id),
            },
        );
        Ok((at, len))
    }

    /// Context `context` gives back its block of `length` bytes at `gpu_address`: not found
    /// where it holds none there of that length, or holds it going already; in use where
    /// `in_use(gpu_address, length)` finds an object of the host's in it, and the block stays the
    /// context's; else going, to be freed under the name returned.
    pub(crate) fn give(
        &self,
        context: u32,
        gpu_address: u64,
        length: u64,
        in_use: impl FnOnce(u64, u64) -> bool,
    ) -> PrivateGive {
        let mut state = self.state.lock();
        let Some(held) = state.held.get_mut(&gpu_address) else {
            return PrivateGive::NotFound;
        };
        if held.context != context || held.len != length || held.going {
            return PrivateGive::NotFound;
        }
        if in_use(gpu_address, length) {
            return PrivateGive::InUse;
        }
        held.going = true;
        PrivateGive::Given(HeldRef {
            at: gpu_address,
            serial: held.serial,
        })
    }

    /// Every block context `context` holds, going from now on, to be freed once its objects are
    /// let go of and its command buffers have finished ([`Self::free`]).
    pub(crate) fn context_closed(&self, context: u32) -> Vec<HeldRef> {
        let mut state = self.state.lock();
        (state.held.iter_mut())
            .filter(|(_, held)| held.context == context && !held.going)
            .map(|(&at, held)| {
                held.going = true;
                HeldRef {
                    at,
                    serial: held.serial,
                }
            })
            .collect()
    }

    /// Frees `blocks`: each that is still held under its name goes back to the source, or its
    /// address to the picks.
    pub(crate) fn free(&self, blocks: &[HeldRef]) {
        let mut released = Vec::new();
        {
            let mut state = self.state.lock();
            for name in blocks {
                if state
                    .held
                    .get(&name.at)
                    .is_none_or(|held| held.serial != name.serial)
                {
                    continue;
                }
                let held = state.held.remove(&name.at);
                match held.and_then(|held| held.block) {
                    Some(block) => released.push(block),
                    None => state.addresses.free(name.at),
                }
            }
        }
        self.release(released);
    }

    /// The device's reset: every block freed at once, its objects already dropped.
    pub(crate) fn reset(&self) {
        let released: Vec<BlockId> = {
            let mut state = self.state.lock();
            let held = std::mem::take(&mut state.held);
            let mut released = Vec::new();
            for (at, held) in held {
                match held.block {
                    Some(block) => released.push(block),
                    None => state.addresses.free(at),
                }
            }
            released
        };
        self.release(released);
    }

    /// Gives `blocks` back to the source, with no lock of the ledger's held.
    fn release(&self, blocks: Vec<BlockId>) {
        if let Some(source) = &self.source {
            for block in blocks {
                // The ledger gives each block back once, as it took it.
                let _ = source.release_block(block);
            }
        }
    }

    /// The blocks held, by address: their contexts, lengths and whether each is going (for the
    /// tests).
    #[cfg(test)]
    pub(crate) fn held(&self) -> Vec<(u64, u32, u64, bool)> {
        let state = self.state.lock();
        (state.held.iter())
            .map(|(&at, held)| (at, held.context, held.len, held.going))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::super::addresses::{PRIVATE_BASE, PRIVATE_END};
    use super::super::{BlockError, BlockInfo};
    use super::*;

    /// A source of Private blocks at addresses from 2^40, a block of `len` bytes rounded up to
    /// 1 MiB, plus 1 KiB, at the next multiple of 1 GiB, which records what it hands out and is
    /// given back.
    #[derive(Default)]
    struct Source {
        next: AtomicU32,
        out: Mutex<Vec<BlockId>>,
        back: Mutex<Vec<BlockId>>,
        refuse: std::sync::atomic::AtomicBool,
    }

    impl BlockSource for Source {
        fn take_block(&self, class: MemoryClass, len: u64) -> Result<BlockInfo, BlockError> {
            assert_eq!(class, MemoryClass::Private);
            if self.refuse.load(Ordering::Relaxed) {
                return Err(BlockError::OutOfMemory);
            }
            let n = self.next.fetch_add(1, Ordering::Relaxed) + 1;
            let id = BlockId(n);
            self.out.lock().push(id);
            Ok(BlockInfo {
                id,
                class,
                len: len.next_multiple_of(1 << 20) + 0x400,
                gpu_address: (1 << 40) + u64::from(n) * (1 << 30),
                contents: None,
            })
        }
        fn release_block(&self, block: BlockId) -> Result<(), BlockError> {
            self.back.lock().push(block);
            Ok(())
        }
    }

    /// With no source, a take is the host's pick in the Private range, its length rounded up to
    /// 64 KiB; a length of 0 or past 1 TiB is invalid; and each block is the taker's alone.
    #[test]
    fn a_take_with_no_source_is_a_pick_in_the_private_range() {
        let blocks = PrivateBlocks::new(None);
        assert_eq!(blocks.take(3, 0x1000), Ok((PRIVATE_BASE, 0x1_0000)));
        assert_eq!(
            blocks.take(4, 0x10_0000),
            Ok((PRIVATE_BASE + 0x1_8000, 0x10_0000))
        );
        assert_eq!(blocks.take(3, 0), Err(TakeError::Invalid));
        assert_eq!(blocks.take(3, TAKE_MAX + 1), Err(TakeError::Invalid));
        assert_eq!(
            blocks.held(),
            [
                (PRIVATE_BASE, 3, 0x1_0000, false),
                (PRIVATE_BASE + 0x1_8000, 4, 0x10_0000, false)
            ]
        );
        assert!(blocks.held().iter().all(|&(at, ..)| at < PRIVATE_END));
    }

    /// A give names the block by its address and length, and only its context's: another's, an
    /// unknown address or a wrong length is not found, and so is one going already. A block an
    /// object lies in stays held. A given block is freed only at its free, and its address is
    /// picked again only once the picks wrap.
    #[test]
    fn a_give_frees_the_block_only_at_its_free_and_only_where_no_object_lies_in_it() {
        let blocks = PrivateBlocks::new(None);
        let (at, len) = blocks.take(3, 0x1_0000).unwrap();
        assert_eq!(blocks.give(4, at, len, |_, _| false), PrivateGive::NotFound);
        assert_eq!(
            blocks.give(3, at + 0x8000, len, |_, _| false),
            PrivateGive::NotFound
        );
        assert_eq!(
            blocks.give(3, at, 0x2_0000, |_, _| false),
            PrivateGive::NotFound
        );
        let mut asked = None;
        assert_eq!(
            blocks.give(3, at, len, |a, l| {
                asked = Some((a, l));
                true
            }),
            PrivateGive::InUse
        );
        assert_eq!(asked, Some((at, len)));
        assert_eq!(blocks.held(), [(at, 3, len, false)]);
        let PrivateGive::Given(name) = blocks.give(3, at, len, |_, _| false) else {
            panic!("given");
        };
        assert_eq!(blocks.held(), [(at, 3, len, true)]);
        assert_eq!(blocks.give(3, at, len, |_, _| false), PrivateGive::NotFound);
        blocks.free(&[name]);
        assert!(blocks.held().is_empty());
        // Freed twice frees nothing.
        blocks.free(&[name]);
        assert_ne!(blocks.take(3, 0x1_0000).unwrap().0, at);
    }

    /// A close hands back every block of the context not going already, and frees them at its
    /// free; the device's reset frees every block at once, and a free of a block the reset freed
    /// frees nothing taken since.
    #[test]
    fn a_close_hands_back_the_context_s_blocks_and_a_reset_frees_every_block() {
        let blocks = PrivateBlocks::new(None);
        let (a, _) = blocks.take(3, 0x1_0000).unwrap();
        let (b, _) = blocks.take(3, 0x2_0000).unwrap();
        let (c, _) = blocks.take(4, 0x1_0000).unwrap();
        let PrivateGive::Given(given) = blocks.give(3, a, 0x1_0000, |_, _| false) else {
            panic!("given");
        };
        let closed = blocks.context_closed(3);
        assert_eq!(closed.len(), 1, "only the block not going already");
        assert_eq!(
            blocks.held(),
            [
                (a, 3, 0x1_0000, true),
                (b, 3, 0x2_0000, true),
                (c, 4, 0x1_0000, false)
            ]
        );
        blocks.free(&closed);
        blocks.free(&[given]);
        assert_eq!(blocks.held(), [(c, 4, 0x1_0000, false)]);
        let gone = blocks.context_closed(4);
        blocks.reset();
        assert!(blocks.held().is_empty());
        let (d, _) = blocks.take(5, 0x1_0000).unwrap();
        blocks.free(&gone);
        assert_eq!(blocks.held(), [(d, 5, 0x1_0000, false)]);
    }

    /// With a source, a take is one of the source's Private blocks, at its address, as long as
    /// the block rounded down to 64 KiB; a refusal of the source's is the take's; a block is
    /// given back to the source at its free and at a reset, and never twice.
    #[test]
    fn a_take_with_a_source_is_its_whole_block_at_its_address() {
        let source = Arc::new(Source::default());
        let blocks = PrivateBlocks::new(Some(source.clone()));
        let (a, len) = blocks.take(3, 0x2_1000).unwrap();
        assert_eq!((a, len), ((1 << 40) + (1 << 30), 0x10_0000));
        let (b, _) = blocks.take(3, 0x1_0000).unwrap();
        assert_eq!(b, (1 << 40) + (2 << 30));
        source.refuse.store(true, Ordering::Relaxed);
        assert_eq!(blocks.take(3, 0x1_0000), Err(TakeError::Refused));
        let PrivateGive::Given(name) = blocks.give(3, a, len, |_, _| false) else {
            panic!("given");
        };
        assert!(source.back.lock().is_empty(), "not before its free");
        blocks.free(&[name]);
        assert_eq!(*source.back.lock(), [BlockId(1)]);
        blocks.free(&[name]);
        blocks.reset();
        assert_eq!(*source.back.lock(), [BlockId(1), BlockId(2)]);
        assert!(blocks.held().is_empty());
    }
}
