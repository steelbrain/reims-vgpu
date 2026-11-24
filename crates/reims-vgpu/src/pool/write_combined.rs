//! [`WriteCombined`], the guest's Shared pool's [`BlockSource`] where the host asks for
//! write-combined blocks (`GraphicsConfig::pool_write_combined`, the C API's
//! `pool_write_combined`): a source whose every Shared take is a take of
//! [`MemoryClass::SharedWriteCombined`] from the source under it, and which hands every other
//! take and every release through.
//!
//! The pool maps each of its blocks into the guest through the host (`PoolHost::add`; QEMU: a RAM
//! region over the block's host mapping). Linux's KVM on arm64 takes a cacheable `VM_PFNMAP`
//! mapping, as a GPU driver's mapping of its memory is, as guest memory only where the CPU has
//! stage-2 forced write-back and `CACHE_DIC` (`kvm_arch_prepare_memory_region`, which userspace
//! reads as `KVM_CAP_ARM_CACHEABLE_PFNMAP_SUPPORTED`), and refuses the slot elsewhere, where QEMU
//! then exits at the pool's first piece: on an M1 Pro under Fedora Asahi Remix 45's kernel
//! `7.3.4-401.asahi.fc45` with Mesa 26.2.7's Honeykrisp, whose host-cached memory type is such a
//! mapping and whose uncached type's mapping KVM takes (**measured** 2026-11-24, issue #174).
//! There the pool's blocks are metal2vulkan-runtime's write-combined Shared memory: the host's
//! mapping is uncached, and the guest reaches the same pages through its own mapping, which KVM
//! maps cacheable at stage 2 where the pages are RAM in the kernel's linear map (read from code,
//! `arch/arm64/kvm/mmu.c`).
//!
//! The guest's Private blocks (`PrivateBlocks`) never reach the guest's address space and take
//! the source under this one, unwrapped.

use std::sync::Arc;

use super::{BlockError, BlockId, BlockInfo, BlockSource, MemoryClass};

/// A source whose Shared takes are write-combined takes of the source under it (the module's
/// docs).
pub(crate) struct WriteCombined {
    source: Arc<dyn BlockSource>,
}

impl WriteCombined {
    /// `source`'s blocks, its Shared ones write-combined.
    pub(crate) fn new(source: Arc<dyn BlockSource>) -> WriteCombined {
        WriteCombined { source }
    }
}

impl BlockSource for WriteCombined {
    /// The source's block of [`MemoryClass::SharedWriteCombined`] for a Shared take, and of
    /// `class` for any other, with the source's answer.
    fn take_block(&self, class: MemoryClass, len: u64) -> Result<BlockInfo, BlockError> {
        let class = match class {
            MemoryClass::Shared => MemoryClass::SharedWriteCombined,
            other => other,
        };
        self.source.take_block(class, len)
    }

    /// The source's release of `block`.
    fn release_block(&self, block: BlockId) -> Result<(), BlockError> {
        self.source.release_block(block)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;

    /// A source that answers each take with a block of the class asked, numbered in turn, and
    /// records the takes and releases it saw.
    #[derive(Default)]
    struct Seen {
        takes: Mutex<Vec<(MemoryClass, u64)>>,
        releases: Mutex<Vec<BlockId>>,
    }

    impl BlockSource for Seen {
        fn take_block(&self, class: MemoryClass, len: u64) -> Result<BlockInfo, BlockError> {
            let mut takes = self.takes.lock();
            takes.push((class, len));
            Ok(BlockInfo {
                id: BlockId(u32::try_from(takes.len()).unwrap()),
                class,
                len,
                gpu_address: 0,
                contents: None,
            })
        }

        fn release_block(&self, block: BlockId) -> Result<(), BlockError> {
            self.releases.lock().push(block);
            Ok(())
        }
    }

    /// A Shared take is the source's write-combined take of the same length; a Private take, and
    /// a release, go to the source as they came.
    #[test]
    fn a_shared_take_is_the_source_s_write_combined_take() {
        let seen = Arc::new(Seen::default());
        let source = WriteCombined::new(Arc::clone(&seen) as Arc<dyn BlockSource>);
        let shared = source.take_block(MemoryClass::Shared, 16 << 20).unwrap();
        let private = source.take_block(MemoryClass::Private, 1 << 20).unwrap();
        assert_eq!(shared.class, MemoryClass::SharedWriteCombined);
        assert_eq!(private.class, MemoryClass::Private);
        source.release_block(shared.id).unwrap();
        assert_eq!(
            *seen.takes.lock(),
            [
                (MemoryClass::SharedWriteCombined, 16 << 20),
                (MemoryClass::Private, 1 << 20)
            ]
        );
        assert_eq!(*seen.releases.lock(), [shared.id]);
    }
}
