//! [`RtBlocks`], the pool's [`BlockSource`] over metal2vulkan-rt's blocks (`Device::take_block`
//! and `release_block`), the runtime the device runs on from the cutover. No path of the
//! library's makes one before then; its tests run it on the GPU each test host holds, but on the
//! Mac, whose Vulkan, MoltenVK, rt's floor refuses.
//!
//! rt hands out a `Block` handle and takes a block back by its handle, and the pool names a block
//! by its number ([`BlockId`]). The source keeps each handle it hands out by its number, the
//! block's only handle (the pool holds plain data, [`BlockInfo`]), and a release takes the handle
//! out of the map and gives it to rt, which returns the block's range to the free ranges of the
//! allocation it was taken from. A later take may land there, under a new number: rt numbers
//! each take anew.
//!
//! A block's bytes are undefined, new or taken again: rt zeroes nothing ahead of need, and
//! nothing here zeroes them. rt maps and unmaps whole allocations through the `BlockMappings`
//! hook its device was made with (`DeviceOptions::block_mappings`), not through the pool: `map`
//! before any block of an allocation goes out, `unmap` before the allocation is freed.

use std::collections::HashMap;

use metal2vulkan_rt as rt;
use parking_lot::Mutex;

use super::{BlockError, BlockId, BlockInfo, BlockSource, MemoryClass};

/// The pool's blocks from an rt device, with the handle of each one handed out.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the pool's source over rt, which no device runs on yet"
    )
)]
pub(crate) struct RtBlocks {
    device: rt::Device,
    /// The handle of each block handed out, by its number.
    held: Mutex<HashMap<BlockId, rt::Block>>,
}

impl RtBlocks {
    /// A source of `device`'s blocks, with none handed out.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the pool's source over rt, which no device runs on yet"
        )
    )]
    pub(crate) fn new(device: rt::Device) -> RtBlocks {
        RtBlocks {
            device,
            held: Mutex::new(HashMap::new()),
        }
    }
}

impl BlockSource for RtBlocks {
    /// rt's block of `class` at least `len` long. Err: [`BlockError::Length`] past the device's
    /// largest block (rt's `Limit::BlockLen`); [`BlockError::OutOfMemory`] for rt's every other
    /// answer: its other limits, a device out of memory or lost, or one quiesced.
    fn take_block(&self, class: MemoryClass, len: u64) -> Result<BlockInfo, BlockError> {
        let rt_class = match class {
            MemoryClass::Private => rt::MemoryClass::Private,
            MemoryClass::Shared => rt::MemoryClass::Shared,
        };
        let block = (self.device.take_block(rt_class, len)).map_err(|error| match error {
            rt::Error::Limit(rt::Limit::BlockLen { .. }) => BlockError::Length,
            _ => BlockError::OutOfMemory,
        })?;
        let info = BlockInfo {
            id: BlockId(block.id.get()),
            class,
            len: block.len,
            gpu_address: block.gpu_address,
            contents: block.contents,
        };
        self.held.lock().insert(info.id, block);
        Ok(info)
    }

    /// Gives `block`'s handle back to rt, out of the map. Err: [`BlockError::NotHeld`] for a
    /// number the source did not hand out, or gave back already.
    fn release_block(&self, block: BlockId) -> Result<(), BlockError> {
        let handle = self.held.lock().remove(&block);
        let handle = handle.ok_or(BlockError::NotHeld)?;
        // The source's own device made it, so rt's one refusal, another device's block, cannot
        // be; rt drops the handle all the same.
        (self.device.release_block(handle)).map_err(|_| BlockError::NotHeld)
    }
}

impl Drop for RtBlocks {
    /// Gives every block still handed out back to rt, then the device.
    fn drop(&mut self) {
        let held = std::mem::take(self.held.get_mut());
        for block in held.into_values() {
            let _ = self.device.release_block(block);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    /// A block is taken from rt and given back by its number alone: the map hands rt its handle,
    /// and a number not handed out, or given back already, is not held. A long block given back
    /// is taken again by the next take of its length, under a new number.
    #[test]
    #[cfg_attr(
        target_os = "macos",
        ignore = "MoltenVK: rt's floor refuses it (no nullDescriptor), so no rt device on the Mac before KosmicKrisp at the cutover; ledger: rt on the Mac"
    )]
    fn on_vulkan_rt_blocks_give_a_block_back_by_its_number() {
        let (device, name) = reims_vgpu_vulkan::seam::test_device().unwrap();
        println!("rt's device: {name}");
        let blocks = RtBlocks::new(device.clone());
        let shared = blocks.take_block(MemoryClass::Shared, MIB).unwrap();
        assert_eq!(shared.class, MemoryClass::Shared);
        assert!(shared.len >= MIB, "{shared:?}");
        assert!(shared.contents.is_some(), "{shared:?}");
        assert_ne!(shared.gpu_address, 0, "{shared:?}");
        let private = blocks.take_block(MemoryClass::Private, MIB).unwrap();
        assert_eq!(private.class, MemoryClass::Private);
        assert!(private.contents.is_none(), "{private:?}");
        assert_ne!(private.id, shared.id);
        assert_eq!(
            blocks.take_block(MemoryClass::Shared, u64::MAX).err(),
            Some(BlockError::Length)
        );

        assert_eq!(blocks.release_block(shared.id), Ok(()));
        assert_eq!(blocks.release_block(shared.id), Err(BlockError::NotHeld));
        assert_eq!(
            blocks.release_block(BlockId(u32::MAX)),
            Err(BlockError::NotHeld)
        );
        assert_eq!(blocks.release_block(private.id), Ok(()));
        // A long block takes an allocation of its own. Given back, it is a free range of that
        // allocation at once, which the next take of its length lands in, under a new number:
        // rt counts one long reuse, and the take makes no allocation.
        let long = blocks.take_block(MemoryClass::Shared, 64 * MIB).unwrap();
        assert_eq!(blocks.release_block(long.id), Ok(()));
        let before = device.counters().metrics();
        let again = blocks.take_block(MemoryClass::Shared, 64 * MIB).unwrap();
        let after = device.counters().metrics();
        assert!(again.id > long.id, "{again:?} after {long:?}");
        assert_eq!(
            again.gpu_address, long.gpu_address,
            "{again:?} after {long:?}"
        );
        assert_eq!(
            (
                after.shared_long_reuses - before.shared_long_reuses,
                after.consumer_allocations - before.consumer_allocations
            ),
            (1, 0),
            "long reuses, and allocations the take made"
        );
        assert_eq!(blocks.release_block(again.id), Ok(()));
    }
}
