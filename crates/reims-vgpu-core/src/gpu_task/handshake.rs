//! The handshake: the task's setup, synchronously on the connection's thread, under the task
//! lock throughout.
//!
//! The task opens the Metal device whose registry ID the host's configuration carries
//! (`Platform::open_device`, which in one process answers the host's own device). With none it
//! logs, replies 2 and goes on without a device: the parts are built over nil and every
//! capability answers no.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Weak};

use super::DeathKind;
use super::service::GpuTask;
use crate::deserializer::Deserializer;
use crate::fatal_unless;
use crate::guest::GuestPtr;
use crate::guest::records::{APV_PAGE_SHIFT, APV_PAGE_SIZE, GuestFeatures, TaskRoot2};
use crate::host::PhysRange;
use crate::memory::map::MemoryMap;
use crate::memory::mapping::Mapping;
use crate::memory::page_table::{INDEX_BITS, PageTableRead, PageTableWalker, host_error_log};
use crate::num;
use crate::platform::device::GpuDevice;
use crate::platform::{Handle, Platform};
use crate::raise::{OnFault, RaiseResult};
use crate::resources::host::ResourceHost;
use crate::resources::manager::ResourceManager;
use crate::rt::Hold;
use crate::tasks::object_lists::{ObjectListKind, ObjectLists};
use crate::tasks::shared::{TaskCore, TaskParts};
use crate::tasks::wire::{Reply, TaskConfiguration};

/// The task root: one guest page.
const TASK_ROOT_LENGTH: u64 = APV_PAGE_SIZE;

/// The task root's fields, as byte offsets into its page (packed).
const ROOT_PAGE_TABLE_ROOT: u64 = num::const_u64(std::mem::offset_of!(TaskRoot2, page_table_root));
const ROOT_PAGE_TABLE_DEPTH: u64 =
    num::const_u64(std::mem::offset_of!(TaskRoot2, page_table_depth));
pub(super) const ROOT_OBJECT_LIST_OFFSET: u64 =
    num::const_u64(std::mem::offset_of!(TaskRoot2, object_list_offset));
pub(super) const ROOT_OBJECT_LIST_LENGTH: u64 =
    num::const_u64(std::mem::offset_of!(TaskRoot2, object_list_length));
pub(super) const ROOT_TASK_LENGTH: u64 =
    num::const_u64(std::mem::offset_of!(TaskRoot2, task_length));
const ROOT_PLACEMENT_LIST_OFFSET: u64 =
    num::const_u64(std::mem::offset_of!(TaskRoot2, placement_list_offset));
const ROOT_PLACEMENT_LIST_LENGTH: u64 =
    num::const_u64(std::mem::offset_of!(TaskRoot2, placement_list_length));

impl<P: Platform> GpuTask<P> {
    /// Sets the task up: the device, the features, the memory map (a copy of the device's;
    /// `None` is nil), the length, the task root and
    /// its page table, the object lists (a queued job sets them from the task root), then the
    /// queues, the paging event and the parts. Replies `(status, pid)`: 2 for a missing map, a
    /// zero length or a task root that fails, else 1.
    ///
    /// With no device it replies 2 and carries on, building the parts over nil (no queues, no
    /// event, every capability no), then replies 1 (the host takes the first reply).
    #[expect(clippy::too_many_arguments, reason = "the handshake message's fields")]
    pub(super) fn handshake(
        &self,
        task_root: u32,
        task_size: u64,
        task_id: u32,
        configuration: TaskConfiguration,
        features: GuestFeatures,
        memory_map: Option<Arc<MemoryMap<P>>>,
        reply: &Reply<(u32, i32)>,
    ) -> RaiseResult<()> {
        let p = &*self.platform;
        let _lock = self.handshake_lock.lock();
        let pid = std::process::id().cast_signed();
        let device = p.open_device(configuration.registry_id).map(Arc::new);
        *self.device.lock() = device.as_deref().map(Handle::clone_ref);
        if device.is_none() {
            log::error!(
                "task {task_id}: no Metal device of registry id {:#x}; going on without one",
                configuration.registry_id
            );
            reply.call((2, pid));
        }
        if let Some(d) = &device {
            d.allow_foreign_libraries();
        }
        self.features.store(&features);
        self.task_id.store(task_id, Ordering::SeqCst);
        // The task keeps its own copy of the device's map
        let map = memory_map.map(|m| m.copy());
        *self.memory_map.lock() = map.clone();
        if map.is_none() {
            log::error!("task {task_id} refused: the device has no memory map");
            reply.call((2, pid));
            return Ok(());
        }
        // No length limit but the page table's coverage.
        self.length.store(task_size, Ordering::SeqCst);
        if task_size == 0 {
            log::warn!("task {task_id} refused: its length is 0");
            reply.call((2, pid));
            return Ok(());
        }
        // Task root 0 fails the define here instead of faulting.
        if task_root == 0 || !self.setup_task_root(task_root) {
            log::warn!("task {task_id} refused: its root at page {task_root:#x} was not set up");
            reply.call((2, pid));
            return Ok(());
        }
        // Queued: the job takes the task lock, which this handshake holds, so it runs after
        let root = self.task_root_ptr().unwrap_or_default();
        let object_list_offset = root.read_u64(ROOT_OBJECT_LIST_OFFSET, DeathKind::NullTaskRoot);
        let object_list_length = root.read_u32(ROOT_OBJECT_LIST_LENGTH, DeathKind::NullTaskRoot);
        if self.features.load().supports_argument_buffers {
            let placement_list_offset =
                root.read_u64(ROOT_PLACEMENT_LIST_OFFSET, DeathKind::NullTaskRoot);
            let placement_list_length =
                root.read_u32(ROOT_PLACEMENT_LIST_LENGTH, DeathKind::NullTaskRoot);
            self.set_object_list(
                object_list_offset,
                object_list_length,
                placement_list_offset,
                placement_list_length,
                Reply::new(|_status| {}),
            );
        } else {
            self.set_object_list(
                object_list_offset,
                object_list_length,
                0,
                0,
                Reply::new(|_status| {}),
            );
        }
        let features = self.features.load();
        let host: Weak<dyn ResourceHost<P>> = self.me.clone();
        let list_argument_buffers =
            features.supports_argument_buffers & configuration.supports_argument_buffers;
        // The lists need no device: built either way. A second handshake keeps the first.
        let object_lists = {
            let mut slot = self.object_lists.lock();
            match slot.as_ref() {
                Some(lists) => Arc::clone(lists),
                None => {
                    let lists = ObjectLists::new(
                        ObjectListKind::MemoryMap,
                        task_size,
                        list_argument_buffers,
                        host.clone(),
                    );
                    *slot = Some(Arc::clone(&lists));
                    lists
                }
            }
        };
        // Both shared: the task's command buffers are one guest process's own work, as an app's
        // queues are in the graphics lane, and nothing here tells WindowServer's task from an
        // app's; its paging is that work's bulk copies.
        let command_queue = match &device {
            Some(d) => d.new_command_queue(),
            None => None,
        };
        *self.command_queue.lock() = command_queue;
        let paging_queue = match &device {
            Some(d) => d.new_command_queue(),
            None => None,
        };
        let paging_event = match &device {
            Some(d) => d.new_event(OnFault::Die)?,
            None => None,
        };
        // The manager shares the task's reference to the device
        let resource_manager = ResourceManager::new(
            self.platform.clone(),
            device.clone(),
            host,
            object_lists.clone(),
            features.supports_indirect_command_buffers,
        );
        let deserializer = Deserializer::new(
            self.platform.clone(),
            device.as_deref().map(Handle::clone_ref),
            resource_manager.object_delegate(),
        );
        // Every capability no without a device.
        let flags = device.as_ref().map(|d| d.caps().flags).unwrap_or_default();
        let supports_buffer_with_address_ranges = flags.supports_buffer_with_address_ranges;
        let supports_heap_with_address_ranges = flags.supports_heap_with_address_ranges;
        let supports_resource_detach_backing = flags.supports_resource_detach_backing;
        let core = TaskCore::new(
            device,
            task_size,
            paging_queue,
            paging_event,
            // The service's paging counter starts at 0: its first signal is 1
            0,
            supports_buffer_with_address_ranges,
            supports_heap_with_address_ranges,
            supports_resource_detach_backing,
        );
        core.set_parts(TaskParts {
            deserializer,
            resource_manager,
            object_list_manager: object_lists,
        });
        // A second handshake keeps the first core, and releases the new one's parts here
        let _ = self.core.set(core);
        reply.call((1, pid));
        Ok(())
    }

    /// A mapping of `length` bytes of guest RAM from `page`, contiguous; `None` outside RAM
    /// or with no memory map.
    fn new_contiguous_mapping(&self, page: u32, length: u64) -> Option<Arc<Mapping<P>>> {
        let range = PhysRange {
            physical_address: u64::from(page).wrapping_shl(APV_PAGE_SHIFT),
            physical_length: length,
        };
        self.map()?.new_range_mapping(&[range], length, 0)
    }

    /// Maps the task root at `task_root` and makes the page table it describes; writes the
    /// task's length into it. False for a root outside RAM (no line), or a depth outside 1 to 3
    /// or a length past the page table's coverage (logged).
    pub(super) fn setup_task_root(&self, task_root: u32) -> bool {
        // The handshake refuses 0 before
        fatal_unless!(task_root != 0, DeathKind::ZeroTaskRoot);
        let mapping = self.new_contiguous_mapping(task_root, TASK_ROOT_LENGTH);
        *self.root_mapping.lock() = mapping.clone();
        let Some(mapping) = mapping else {
            return false;
        };
        let root = GuestPtr::new(mapping.virtual_address());
        *self.task_root.lock() = Some(root);
        let depth = root.read_u32(ROOT_PAGE_TABLE_DEPTH, DeathKind::NullTaskRoot);
        if !(1..=3).contains(&depth) {
            log::warn!("task root refused: page table depth {depth} is not 1 to 3");
            return false;
        }
        let page_table_root = root.read_u32(ROOT_PAGE_TABLE_ROOT, DeathKind::NullTaskRoot);
        let task_length = self.task_length();
        // A table of depth d covers ENTRIES^d pages
        let mut entries: u64 = 1;
        for _ in 0..depth {
            entries = entries.wrapping_shl(INDEX_BITS);
        }
        if task_length > entries.wrapping_shl(APV_PAGE_SHIFT) {
            log::warn!(
                "task root refused: {task_length:#x} bytes are more than a page table of depth \
                 {depth} covers"
            );
            return false;
        }
        // The walk reads guest RAM through the task's map as it is at each read
        let this = self.me.clone();
        let read: PageTableRead = Arc::new(
            move |_: Hold<'_>, _: OnFault, address: u64, dst: &mut [u8]| {
                let map = this.upgrade().and_then(|t| t.map());
                Ok(map.is_some_and(|m| m.read(address, dst)))
            },
        );
        // Holes refused; never nil
        let walker = PageTableWalker::new(
            page_table_root,
            depth,
            false,
            read,
            host_error_log(self.platform.clone()),
        );
        *self.page_table.lock() = Some(Arc::new(walker));
        root.write_u64(ROOT_TASK_LENGTH, task_length, DeathKind::NullTaskRoot);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::*;
    use super::*;
    use crate::testkit::capture::catch_death;
    use crate::testkit::platform::FAKE_REGISTRY_ID;

    /// The handshake builds the task: the task length goes into the task root, the sends come
    /// in order, and the paging counter starts at 0
    #[test]
    fn builds_the_task() {
        let tx = fixture(1, &rw_pages(4));
        let r = handshake_with(&tx, ROOT_PAGE, TASK_LEN, FAKE_REGISTRY_ID, true);
        assert_eq!(r, vec![(1, pid())]);
        let root = tx.task.task_root_ptr().unwrap();
        assert_eq!(
            root.read_u64(ROOT_TASK_LENGTH, DeathKind::NullTaskRoot),
            TASK_LEN
        );
        let sels = tx.sels();
        let pos = |s: &str| sels.iter().position(|x| *x == s).unwrap();
        assert!(pos("allowLibrariesFromOtherPlatforms") < pos("newCommandQueue"));
        assert_eq!(sels.iter().filter(|s| **s == "newCommandQueue").count(), 2);
        assert!(pos("newCommandQueue") < pos("newEvent"));
        assert!(tx.task.core.get().is_some());
        assert_eq!(tx.task.task_id.load(std::sync::atomic::Ordering::SeqCst), 7);
        assert_eq!(
            tx.task
                .core()
                .paging_counter
                .load(std::sync::atomic::Ordering::SeqCst),
            0
        );
    }

    /// A nil memory map replies 2
    #[test]
    fn without_a_memory_map_replies_2() {
        let tx = fixture(1, &rw_pages(4));
        let r = handshake_with(&tx, ROOT_PAGE, TASK_LEN, FAKE_REGISTRY_ID, false);
        assert_eq!(r, vec![(2, pid())]);
        assert!(tx.task.memory_map.lock().is_none());
        assert!(tx.task.core.get().is_none());
    }

    /// A zero task length replies 2
    #[test]
    fn zero_length_replies_2() {
        let tx = fixture(1, &rw_pages(4));
        let r = handshake_with(&tx, ROOT_PAGE, 0, FAKE_REGISTRY_ID, true);
        assert_eq!(r, vec![(2, pid())]);
        assert!(tx.task.task_root_ptr().is_none());
        assert!(tx.task.core.get().is_none());
    }

    /// Task root 0 and a root outside RAM reply 2, with one line
    #[test]
    fn task_root_failures_reply_2() {
        let lines = Lines::capture();
        let tx = fixture(1, &rw_pages(4));
        assert_eq!(
            handshake_with(&tx, 0, TASK_LEN, FAKE_REGISTRY_ID, true),
            vec![(2, pid())]
        );
        assert_eq!(
            lines.take(),
            ["task 7 refused: its root at page 0x0 was not set up"]
        );
        let tx = fixture(1, &rw_pages(4));
        assert_eq!(
            handshake_with(&tx, 0x10, TASK_LEN, FAKE_REGISTRY_ID, true),
            vec![(2, pid())]
        );
        // the mapping failed: the root setup fails without its own line
        assert_eq!(
            lines.take(),
            ["task 7 refused: its root at page 0x10 was not set up"]
        );
    }

    /// A depth outside 1 to 3, or a length past the coverage, fails
    #[test]
    fn invalid_depth_replies_2() {
        let lines = Lines::capture();
        let tx = fixture(4, &rw_pages(4));
        assert_eq!(
            handshake_with(&tx, ROOT_PAGE, TASK_LEN, FAKE_REGISTRY_ID, true),
            vec![(2, pid())]
        );
        // one line of the root setup's, then the handshake's
        let not_set_up = format!("task 7 refused: its root at page {ROOT_PAGE:#x} was not set up");
        assert_eq!(
            lines.take(),
            [
                "task root refused: page table depth 4 is not 1 to 3".to_owned(),
                not_set_up.clone()
            ]
        );
        // depth 1 covers ENTRIES pages: one byte more fails
        let covered = crate::memory::page_table::ENTRIES << APV_PAGE_SHIFT;
        let tx = fixture(1, &rw_pages(4));
        assert_eq!(
            handshake_with(&tx, ROOT_PAGE, covered + 1, FAKE_REGISTRY_ID, true),
            vec![(2, pid())]
        );
        assert_eq!(
            lines.take(),
            [
                format!(
                    "task root refused: {:#x} bytes are more than a page table of depth 1 covers",
                    covered + 1
                ),
                not_set_up
            ]
        );
        let tx = fixture(1, &rw_pages(4));
        assert_eq!(
            handshake_with(&tx, ROOT_PAGE, covered, FAKE_REGISTRY_ID, true),
            vec![(1, pid())]
        );
        assert!(lines.take().is_empty());
    }

    /// No device: reply 2, carry on, reply 1; the parts are built over nil, and nothing reaches
    /// Metal
    #[test]
    fn without_a_device_carries_on() {
        let tx = fixture(1, &rw_pages(4));
        let r = handshake_with(&tx, ROOT_PAGE, TASK_LEN, 0x42, true);
        assert_eq!(r, vec![(2, pid()), (1, pid())]);
        let core = tx.task.core.get().expect("parts over nil");
        assert!(core.mtl_device().is_none());
        assert!(core.paging_queue.is_none());
        assert!(core.paging_event.is_none());
        assert!(core.parts().is_some());
        assert!(tx.task.object_lists.lock().is_some());
        assert!(!tx.sels().contains(&"newCommandQueue"));
    }

    /// With no device the lists are still built, so the handshake's queued set-object-list runs
    /// and, refusing a list that ends at the task's end, zeroes the root's list fields
    #[test]
    fn without_a_device_still_sets_the_object_list() {
        let tx = fixture(1, &rw_pages(4));
        tx.ram
            .write_u64(page_pa(ROOT_PAGE) + ROOT_OBJECT_LIST_OFFSET, TASK_LEN);
        tx.ram
            .write_u32(page_pa(ROOT_PAGE) + ROOT_OBJECT_LIST_LENGTH, 12);
        let r = handshake_with(&tx, ROOT_PAGE, TASK_LEN, 0x42, true);
        assert_eq!(r, vec![(2, pid()), (1, pid())]);
        let root = tx.task.task_root_ptr().unwrap();
        assert_eq!(
            root.read_u64(ROOT_OBJECT_LIST_OFFSET, DeathKind::NullTaskRoot),
            0
        );
        assert_eq!(
            root.read_u32(ROOT_OBJECT_LIST_LENGTH, DeathKind::NullTaskRoot),
            0
        );
    }

    /// The root setup's assert on a zero root
    #[test]
    fn setup_task_root_zero_dies() {
        let tx = fixture(1, &rw_pages(4));
        assert_eq!(
            catch_death(|| tx.task.setup_task_root(0)).unwrap_err(),
            DeathKind::ZeroTaskRoot.site()
        );
    }
}
