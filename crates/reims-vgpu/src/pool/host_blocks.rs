//! [`HostBlocks`]: a [`BlockSource`] of the process's own memory, which hands blocks out, keeps
//! them ready and frees them as metal2vulkan-runtime's device does, until the pool takes its blocks
//! from that runtime: then only the source changes.
//!
//! Each block is zeroed memory of the process's, aligned to 2 MiB and touched a byte a page before
//! it goes out, as the runtime touches a Shared block. A block given back goes to the ready set
//! with its id and mapping, and reads zero again when it goes out anew, as a new Metal buffer
//! does; past two ready blocks, the oldest is freed, on the thread that gave the block back. The
//! pool gives a block back only once the guest's mapping of it is gone, so a block is freed with
//! no word to anyone. At the source's end, once nothing holds it, every block it handed out or
//! keeps ready is freed. Its blocks have no GPU: `gpu_address` is 0.

use std::alloc::Layout;
use std::collections::{HashMap, VecDeque};
use std::ptr::NonNull;

use parking_lot::Mutex;

use super::{BLOCK_LEN, BlockError, BlockId, BlockInfo, BlockSource, MemoryClass};

/// The alignment of a block's memory: 2 MiB, a large page.
const ALIGN: usize = 2 << 20;

/// How many blocks given back stay ready: past them the oldest is freed.
const READY_MAX: usize = 2;

/// One block of the process's memory.
struct Block {
    info: BlockInfo,
    layout: Layout,
}

impl Block {
    /// A zeroed block of `len` bytes, every page touched; `None` where the host refuses it.
    fn new(id: BlockId, len: u64) -> Option<Block> {
        let layout = Layout::from_size_align(usize::try_from(len).ok()?, ALIGN).ok()?;
        // SAFETY: `layout` has a non-zero size: `len` is at least a page.
        let ptr = NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) })?;
        // The first CPU write to a fresh mapping is the slow one: done here, off the guest's path.
        for page in (0..layout.size()).step_by(4096) {
            // SAFETY: inside the allocation.
            unsafe { ptr.as_ptr().add(page).write_volatile(0) };
        }
        Some(Block {
            info: BlockInfo {
                id,
                class: MemoryClass::Shared,
                len,
                gpu_address: 0,
                contents: Some(ptr),
            },
            layout,
        })
    }
}

impl Block {
    /// Zeroes the whole block: a block taken again reads as a new one.
    fn zero(&self) {
        if let Some(ptr) = self.info.contents {
            // SAFETY: the block's own allocation, which nothing else uses while it is ready.
            unsafe { std::ptr::write_bytes(ptr.as_ptr(), 0, self.layout.size()) };
        }
    }
}

impl Drop for Block {
    fn drop(&mut self) {
        if let Some(ptr) = self.info.contents {
            // SAFETY: allocated with this layout in `Block::new`.
            unsafe { std::alloc::dealloc(ptr.as_ptr(), self.layout) };
        }
    }
}

/// What the source's lock holds.
#[derive(Default)]
struct Blocks {
    /// The blocks handed out, by id.
    out: HashMap<BlockId, Block>,
    /// The pooled blocks given back, oldest first.
    ready: VecDeque<Block>,
    /// The id the next new block takes.
    next: u32,
}

/// Blocks of the process's own memory ([module doc](self)).
pub(crate) struct HostBlocks {
    blocks: Mutex<Blocks>,
}

impl HostBlocks {
    /// A source with no block yet.
    pub(crate) fn new() -> HostBlocks {
        HostBlocks {
            blocks: Mutex::new(Blocks {
                next: 1,
                ..Blocks::default()
            }),
        }
    }

    /// How many blocks it has, handed out and ready: for the tests.
    #[cfg(test)]
    pub(crate) fn counts(&self) -> (usize, usize) {
        let b = self.blocks.lock();
        (b.out.len(), b.ready.len())
    }
}

impl BlockSource for HostBlocks {
    fn take_block(&self, class: MemoryClass, len: u64) -> Result<BlockInfo, BlockError> {
        if class != MemoryClass::Shared {
            return Err(BlockError::Length);
        }
        let mut b = self.blocks.lock();
        if len <= BLOCK_LEN
            && let Some(block) = b.ready.pop_front()
        {
            drop(b);
            block.zero();
            let info = block.info;
            self.blocks.lock().out.insert(info.id, block);
            return Ok(info);
        }
        let id = BlockId(b.next);
        b.next = b.next.checked_add(1).ok_or(BlockError::OutOfMemory)?;
        drop(b);
        let len = if len <= BLOCK_LEN {
            BLOCK_LEN
        } else {
            len.checked_next_multiple_of(ALIGN as u64)
                .ok_or(BlockError::Length)?
        };
        let block = Block::new(id, len).ok_or(BlockError::OutOfMemory)?;
        let info = block.info;
        self.blocks.lock().out.insert(id, block);
        Ok(info)
    }

    fn release_block(&self, block: BlockId) -> Result<(), BlockError> {
        let mut b = self.blocks.lock();
        let given = b.out.remove(&block).ok_or(BlockError::NotHeld)?;
        if given.info.len > BLOCK_LEN {
            drop(b);
            drop(given);
            return Ok(());
        }
        b.ready.push_back(given);
        let surplus = (b.ready.len() > READY_MAX)
            .then(|| b.ready.pop_front())
            .flatten();
        drop(b);
        drop(surplus);
        Ok(())
    }
}
