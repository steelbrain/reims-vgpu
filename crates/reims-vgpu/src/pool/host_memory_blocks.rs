//! [`HostMemoryBlocks`], the guest's Shared pool's [`BlockSource`] where the host asks for blocks
//! of host memory (`GraphicsConfig::pool_host_memory`, the C API's `pool_host_memory`), on Linux:
//! each block is memory of the library's own, a memfd it maps shared
//! (`reims_vgpu_vulkan::os::RegionMemory`), imported into the rt device as a dma-buf of its pages
//! (`rt_platform::import_host_block`, rt's `Device::import_dma_buf`). The pool hands the host the
//! memfd's mapping to map into the guest (`PoolHost::add`; QEMU: a RAM region over it), as it does
//! the guest's RAM.
//!
//! The pool maps each of its blocks into the guest. Linux's KVM on arm64 takes a cacheable
//! `VM_PFNMAP` mapping, as a GPU driver's mapping of its own memory is, as a memory slot only
//! where the CPU has stage-2 forced write-back and `CACHE_DIC` (`kvm_arch_prepare_memory_region`,
//! which userspace reads as `KVM_CAP_ARM_CACHEABLE_PFNMAP_SUPPORTED`), and refuses the slot
//! elsewhere, where QEMU then exits at the pool's first piece: on an M1 Pro under Fedora Asahi
//! Remix 45's kernel `7.3.4-401.asahi.fc45`, whose Honeykrisp (Mesa 26.2.7) maps all of its
//! memory so (**measured** 2026-11-24, issue #174). A memfd's mapping is ordinary shared memory,
//! which KVM maps as guest memory as it maps the guest's RAM, cached; the host's mapping is
//! cached too, and the GPU reaches the pages through the dma-buf's import, as it reaches the
//! guest's no-copy buffers. So every view of a block is the same cached memory.
//!
//! A block is known to the rt platform's device from its take until just before it goes back
//! (`rt_platform::KnownBlocks`, as [`super::RtBlocks`]' are), which also keeps its host address,
//! the memfd's mapping, for the device's reads of the guest's bytes in it. A block given back is
//! dropped: rt frees the import once nothing of the GPU's holds it, and the import's release then
//! unmaps the memory. Each block is one import, one of the device's `maxMemoryAllocationCount`
//! allocations; the pool's blocks are 16 MiB ([`super::BLOCK_LEN`]) but for longer takes.
//!
//! The source makes no Private blocks: the guest's Private blocks (`PrivateBlocks`) never reach
//! the guest's address space and take the platform's own source.

use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};

use metal2vulkan_runtime as rt;
use parking_lot::Mutex;
use reims_vgpu_vulkan::rt_platform::{RtDevice, RtObj};

use super::rt_blocks::Knowing;
use super::{BlockError, BlockId, BlockInfo, BlockSource, MemoryClass};

/// A block's length is a multiple of this: 64 KiB, the C API's `pool_add` unit and rt's
/// granule.
const GRANULE: u64 = 64 << 10;

/// Whether the process logged an import the device refused.
static REFUSAL_SAID: AtomicBool = AtomicBool::new(false);

/// The pool's blocks of host memory, imported into an rt device, with the handle of each one
/// handed out (the module's docs).
pub(crate) struct HostMemoryBlocks {
    /// The platform's handle of the device, which its let-go counts (`RtBlocks`' docs).
    device: RtObj<RtDevice>,
    /// The blocks the platform's device knows, where each block handed out is known.
    known: Box<dyn Knowing>,
    /// The handle of each block handed out, by its number.
    held: Mutex<HashMap<BlockId, rt::Block>>,
}

impl HostMemoryBlocks {
    /// A source of blocks of host memory imported into `device`, with none handed out, each known
    /// in `known` while it is; holding `device` until it goes.
    pub(crate) fn new(device: RtObj<RtDevice>, known: Box<dyn Knowing>) -> HostMemoryBlocks {
        HostMemoryBlocks {
            device,
            known,
            held: Mutex::new(HashMap::new()),
        }
    }
}

impl BlockSource for HostMemoryBlocks {
    /// A block of `len` bytes rounded up to 64 KiB of new host memory, zeros, imported into the
    /// device, known from now. Err: [`BlockError::Length`] for a Private block, which the source
    /// makes none of, and for a length of 0 or past the address space;
    /// [`BlockError::OutOfMemory`] where the host gives no memory, or the device refuses the
    /// import or states no GPU address for it, the process's first refusal logged.
    fn take_block(&self, class: MemoryClass, len: u64) -> Result<BlockInfo, BlockError> {
        if class != MemoryClass::Shared || len == 0 {
            return Err(BlockError::Length);
        }
        let len = len
            .checked_next_multiple_of(GRANULE)
            .ok_or(BlockError::Length)?;
        let memory =
            reims_vgpu_vulkan::os::RegionMemory::new(len).ok_or(BlockError::OutOfMemory)?;
        let contents = NonNull::new(memory.base());
        let refused = |why: &dyn std::fmt::Display| {
            if !REFUSAL_SAID.swap(true, Ordering::Relaxed) {
                log::warn!(
                    "pool: a block of {len} bytes of host memory was not imported into the GPU: \
                     {why}; the take is refused (logged once)"
                );
            }
            BlockError::OutOfMemory
        };
        let block = reims_vgpu_vulkan::rt_platform::import_host_block(&self.device, memory, len)
            .map_err(|why| refused(&why))?;
        let Some(gpu_address) = block.gpu_address else {
            let _ = self.device.rt().release_block(block);
            return Err(refused(&"rt states no GPU address for it"));
        };
        let info = BlockInfo {
            id: BlockId(block.id.get()),
            class,
            len,
            gpu_address,
            contents,
        };
        self.known.know(block.clone());
        self.held.lock().insert(info.id, block);
        Ok(info)
    }

    /// Gives `block`'s handle back to rt, out of the map, forgotten first: rt frees the import
    /// once nothing of the GPU's holds it, and its memory goes then. Err:
    /// [`BlockError::NotHeld`] for a number the source did not hand out, or gave back already.
    fn release_block(&self, block: BlockId) -> Result<(), BlockError> {
        let handle = self.held.lock().remove(&block);
        let handle = handle.ok_or(BlockError::NotHeld)?;
        self.known.forget(handle.id);
        (self.device.rt().release_block(handle)).map_err(|_| BlockError::NotHeld)
    }
}

impl Drop for HostMemoryBlocks {
    /// Gives every block still handed out back to rt, each forgotten first, then the device.
    fn drop(&mut self) {
        let held: Vec<rt::Block> = self.held.get_mut().drain().map(|(_, b)| b).collect();
        for block in held {
            self.known.forget(block.id);
            let _ = self.device.rt().release_block(block);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use reims_vgpu_vulkan::core::platform::Handle as _;

    use super::*;
    use crate::platform::ApiPlatform as _;

    const MIB: u64 = 1 << 20;

    /// A set of known blocks the test reads: each block's address and length, by its number.
    #[derive(Default, Clone)]
    struct Known(Arc<Mutex<HashMap<u32, (u64, u64)>>>);

    impl Knowing for Known {
        fn know(&self, block: rt::Block) {
            let at = block.gpu_address.expect("a taken block's address");
            self.0.lock().insert(block.id.get(), (at, block.len));
        }

        fn forget(&self, id: rt::BlockId) {
            self.0.lock().remove(&id.get());
        }
    }

    /// A Shared block is host memory imported into the device: its length the ask rounded up to
    /// 64 KiB, at a GPU address, its host mapping zeros the test writes and reads back, known from
    /// its take until its release, and given back by its number once. A Private block, and one
    /// of no bytes, are refused.
    #[test]
    fn on_vulkan_host_memory_blocks_are_imported_host_memory() {
        let platform = crate::rt_test_end::platform();
        let device = platform.metal_device(None).expect("rt's device on the GPU");
        println!("rt's device: {}", device.facts().name);
        let known = Known::default();
        let blocks = HostMemoryBlocks::new(device.clone_ref(), Box::new(known.clone()));
        let block = blocks.take_block(MemoryClass::Shared, MIB + 1).unwrap();
        assert_eq!(block.class, MemoryClass::Shared);
        assert_eq!(block.len, MIB + GRANULE);
        assert_ne!(block.gpu_address, 0, "{block:?}");
        let contents = block.contents.expect("a host block's mapping");
        // SAFETY: the mapping spans the block, which the test holds while it reads and writes.
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(contents.as_ptr(), block.len as usize) };
        assert!(bytes.iter().all(|&b| b == 0), "new memory is zeros");
        bytes.fill(0xA5);
        assert!(bytes.iter().all(|&b| b == 0xA5));
        assert_eq!(
            known.0.lock().get(&block.id.0),
            Some(&(block.gpu_address, block.len))
        );
        assert_eq!(
            blocks.take_block(MemoryClass::Private, MIB).err(),
            Some(BlockError::Length)
        );
        assert_eq!(
            blocks.take_block(MemoryClass::Shared, 0).err(),
            Some(BlockError::Length)
        );
        assert_eq!(blocks.release_block(block.id), Ok(()));
        assert!(known.0.lock().is_empty());
        assert_eq!(blocks.release_block(block.id), Err(BlockError::NotHeld));
    }
}
