//! Task memory through the guest's page table: mappings of task ranges, copies out of the
//! task, cursors over it, address ranges and buffers over task pages, and the object lists'
//! mapping.
//!
//! Range rules (each with its test):
//! - a copy refuses a range ending exactly at the task's end (`>=`), while mappings, cursors
//!   and address ranges accept it (`>`);
//! - a failed physical read in a copy is logged and skipped, and the copy's position does not
//!   advance;
//! - a run of pages is cut when it reaches two pages;
//! - one read-only leaf makes a whole mapping read-only.
//!
//! A task page whose guest page lies outside every range of the memory map has a NULL host
//! address, which a cursor or an address range takes as it is, with no line and no raise. A read
//! or a reply through it is a NULL guest access, a death in the task's own code, which ends the
//! task (the suite's `service_gap_cmdbuf_outside_ram` and `covgap_mem_reply_in_hole`).

use std::sync::Arc;

use super::service::GpuTask;
use super::{DeathKind, RaiseKind, exception, raise_carrying, raise_caught, raise_on};
use crate::device::tasks::MAX_LOCAL_TASK_LENGTH;
use crate::guest::records::{APV_PAGE_MASK, APV_PAGE_SHIFT, APV_PAGE_SIZE};
use crate::host::{HostPtr, PhysRange};
use crate::memory::cursor::{Cursor, CursorRange, WritableCursor};
use crate::memory::mapping::Mapping;
use crate::num::{ToU64, ToUsize};
use crate::platform::Platform;
use crate::platform::device::GpuDevice;
use crate::platform::types::ResourceAddressRange;
use crate::raise::{Kept, keep_forever};
use crate::raise::{OnFault, RaiseResult};
use crate::resources::host::MappedAddress;
use crate::rt::Hold;
use crate::{fatal, fatal_unless};

/// The guest page: 16 KiB (4 KiB on x86-64).
pub(super) const PAGE: u64 = APV_PAGE_SIZE;
/// The in-page mask.
pub(super) const PAGE_MASK: u64 = APV_PAGE_MASK;

impl<P: Platform> GpuTask<P> {
    /// Raises unless `offset..offset + length` lies in the task (a range ending at its end
    /// passes).
    fn check_task_range(offset: u64, length: u64, task_length: u64) -> RaiseResult<()> {
        let (end, overflow) = offset.overflowing_add(length);
        if overflow || end > task_length {
            return raise_caught(
                RaiseKind::OutOfRange,
                format_args!(
                    "{length:#x} bytes at task offset {offset:#x} run past the task's \
                     {task_length:#x}"
                ),
            );
        }
        Ok(())
    }

    /// A mapping of the task's pages `offset..offset + length` (page-aligned, not empty), one
    /// run of guest pages per contiguous stretch; `None` for a writable request over a
    /// read-only leaf, or with no memory map.
    pub(super) fn new_virtual_mapping(
        &self,
        hold: Hold<'_>,
        offset: u64,
        length: u64,
        need_writable: bool,
    ) -> RaiseResult<Option<Arc<Mapping<P>>>> {
        Self::check_task_range(offset, length, self.task_length())?;
        if self.task_root_ptr().is_none() {
            return raise_caught(
                RaiseKind::NoPageTable,
                "no page table to map task memory through",
            );
        }
        if offset & PAGE_MASK != 0 {
            return raise_caught(
                RaiseKind::UnalignedOffset,
                format_args!("task offset {offset:#x} to map is not page-aligned"),
            );
        }
        if length & PAGE_MASK != 0 {
            return raise_caught(
                RaiseKind::UnalignedLength,
                format_args!("length {length:#x} to map is not page-aligned"),
            );
        }
        if length == 0 {
            return raise_caught(
                RaiseKind::ZeroLength,
                format_args!("map of 0 bytes at task offset {offset:#x}"),
            );
        }
        let mut ranges: Vec<PhysRange> = Vec::new();
        let mut writable = true;
        let mut range = PhysRange::default();
        // With no page table there is nothing to visit
        if let Some(walker) = self.walker() {
            walker.visit_pages_in_range(
                hold,
                OnFault::Return,
                offset,
                length,
                &mut |_address, page, page_writable| {
                    let physical = u64::from(page).wrapping_shl(APV_PAGE_SHIFT);
                    // One read-only leaf makes the whole mapping read-only.
                    if writable && page_writable == 0 {
                        writable = false;
                    }
                    // The run's start plus one page, from an initial {0, 0}.
                    if physical == range.physical_address.wrapping_add(PAGE) {
                        range.physical_length = range.physical_length.wrapping_add(PAGE);
                    } else {
                        if range.physical_length != 0 {
                            ranges.push(range);
                        }
                        range.physical_address = physical;
                        range.physical_length = PAGE;
                    }
                    Ok(true)
                },
            )?;
        }
        if range.physical_length != 0 {
            ranges.push(range);
        }
        if need_writable && !writable {
            return Ok(None);
        }
        Ok(self
            .map()
            .and_then(|m| m.new_range_mapping(&ranges, length, u32::from(!writable))))
    }

    /// The host address of task offset `offset`, through a mapping of the pages
    /// `offset..offset + length` covers, which comes back with it: the caller holds it while it
    /// uses the address. No range check of its own.
    pub(super) fn mapped_address_for_offset(
        &self,
        hold: Hold<'_>,
        offset: u64,
        length: u64,
        need_writable: bool,
    ) -> RaiseResult<MappedAddress<P>> {
        let start = offset & !PAGE_MASK;
        // Wrapping, unchecked
        let end = offset.wrapping_add(length).wrapping_add(PAGE_MASK) & !PAGE_MASK;
        let mapping =
            self.new_virtual_mapping(hold, start, end.wrapping_sub(start), need_writable)?;
        let Some(mapping) = mapping else {
            return raise_caught(
                RaiseKind::NoMapping,
                format_args!("no mapping of {length:#x} bytes at task offset {offset:#x}"),
            );
        };
        if need_writable && !mapping.writable() {
            // The raise carries the mapping to where it ends (the job's holder), so it goes
            // after the handler's reply.
            return raise_carrying(
                RaiseKind::ReadOnlyMapping,
                format_args!("the mapping at task offset {offset:#x} is read-only"),
                mapping,
            );
        }
        Ok(MappedAddress {
            address: mapping.virtual_address().wrapping_add(offset & PAGE_MASK),
            mapping: Some(mapping),
        })
    }

    /// Copies `dst.len()` bytes of the task from `offset` into `dst`, page by page through the
    /// page table.
    ///
    /// A copy ending exactly at the task's end is refused (`>=`). A page whose read fails is
    /// logged and skipped, and the next page's bytes land where that page's would have. Its
    /// raises are as `on` says: a copy is reached both under a handler and with none (the
    /// rate-map reset job).
    pub(super) fn copy_from_virtual_offset(
        &self,
        hold: Hold<'_>,
        on: OnFault,
        offset: u64,
        dst: &mut [u8],
    ) -> RaiseResult<()> {
        let length = dst.len().to_u64();
        let task_length = self.task_length();
        let (end, overflow) = offset.overflowing_add(length);
        if overflow || end >= task_length {
            return raise_on(
                on,
                RaiseKind::OutOfRange,
                format_args!(
                    "copy of {length:#x} bytes at task offset {offset:#x} reaches the task's \
                     end, {task_length:#x}"
                ),
            );
        }
        let page_offset = offset & PAGE_MASK;
        let start = offset & !PAGE_MASK;
        let size = (end.wrapping_add(PAGE_MASK) & !PAGE_MASK).wrapping_sub(start);
        let mut remaining = length;
        // Where the next page's bytes go in `dst`
        let mut current: u64 = 0;
        let map = self.map();
        if let Some(walker) = self.walker() {
            walker.visit_pages_in_range(hold, on, start, size, &mut |address, page, _flags| {
                let mut physical = u64::from(page).wrapping_shl(APV_PAGE_SHIFT);
                let page_end = physical.wrapping_add(PAGE);
                if address == start {
                    physical = physical.wrapping_add(page_offset);
                }
                let chunk_length = page_end.wrapping_sub(physical).min(remaining);
                fatal_unless!(chunk_length <= length, DeathKind::CopyChunk);
                let from = current.to_usize();
                let to = current.wrapping_add(chunk_length).to_usize();
                let Some(chunk) = dst.get_mut(from..to) else {
                    fatal!(DeathKind::CopyChunk)
                };
                let ok = map.as_ref().is_some_and(|m| m.read(physical, chunk));
                if ok {
                    current = current.wrapping_add(chunk_length);
                    remaining = remaining.wrapping_sub(chunk_length);
                } else {
                    log::warn!(
                        "task {}: {chunk_length:#x} bytes of guest memory at {physical:#x} not \
                         read, for byte {current:#x} of a copy",
                        self.id()
                    );
                }
                // The walk ignores the answer
                Ok(ok)
            })?;
        }
        Ok(())
    }

    /// A fresh buffer with `length` bytes of the task from `offset`.
    ///
    /// A length past the largest task (32 GiB) raises before anything is allocated, the fault
    /// the guest saw when the host could not allocate the buffer. The buffer is made zeroed
    /// (lazily mapped for a large one), so a copy that fails its range check raises before any
    /// page is touched, and a raise in the copy frees it. Its raises are as `on` says.
    pub(super) fn data_from_virtual_offset(
        &self,
        hold: Hold<'_>,
        on: OnFault,
        offset: u64,
        length: u64,
    ) -> RaiseResult<Option<Vec<u8>>> {
        if length > MAX_LOCAL_TASK_LENGTH {
            return raise_on(
                on,
                RaiseKind::BufferLength,
                format_args!("buffer of {length:#x} bytes is longer than any task"),
            );
        }
        let mut buffer = vec![0u8; length.to_usize()];
        self.copy_from_virtual_offset(hold, on, offset, &mut buffer)?;
        Ok(Some(buffer))
    }

    /// The object lists mapped. The test that they are comes before the lock and again under
    /// it; a failure to map raises. The first success of the task rebuilds its resource
    /// indices (once per task).
    pub(super) fn ensure_object_list_mapped(&self, hold: Hold<'_>) -> RaiseResult<()> {
        let mapped = |lists: &crate::tasks::object_lists::ObjectLists<P>| {
            lists.is_object_list_mapped()
                && (lists.is_placement_list_mapped() || lists.placement_list_length() == 0)
        };
        if mapped(&self.lists()) {
            return Ok(());
        }
        let _lock = self.object_lists_lock.lock();
        let lists = self.lists();
        if mapped(&lists) {
            return Ok(());
        }
        if !lists.ensure_mapped(hold) {
            return raise_caught(
                RaiseKind::ObjectListsUnmapped,
                "the task's object lists could not be mapped",
            );
        }
        let restored = &self.restored;
        if !restored.load(std::sync::atomic::Ordering::SeqCst) {
            restored.store(true, std::sync::atomic::Ordering::SeqCst);
            self.core().rebuild_task(hold)?;
        }
        Ok(())
    }

    /// A cursor over the task's `offset..offset + length`, reading and writing guest RAM in
    /// place, at use; with `writable`, one that can write, or `None` over a read-only leaf.
    /// `None` (logged) before the task root is set up, for a range past the task's end or an
    /// empty one, and when the walk raises.
    ///
    /// A range ending exactly at the task's end is accepted (`>`), where the in-process copy
    /// refuses it (testing.md 7.1).
    pub(super) fn cursor(
        &self,
        hold: Hold<'_>,
        offset: u64,
        length: u64,
        writable: bool,
    ) -> RaiseResult<Option<Cursor>> {
        if self.task_root_ptr().is_none() {
            log::warn!(
                "task {}: no cursor before the task root is set up",
                self.id()
            );
            return Ok(None);
        }
        let (end, overflow) = offset.overflowing_add(length);
        if overflow || end > self.task_length() {
            log::warn!(
                "task {}: no cursor of {length:#x} bytes at {offset:#x}: past the task's end",
                self.id()
            );
            return Ok(None);
        }
        if length == 0 {
            log::warn!("task {}: no cursor of 0 bytes at {offset:#x}", self.id());
            return Ok(None);
        }
        let map = self.map();
        let walker = self.walker();
        let walked = (|| -> RaiseResult<_> {
            let start = offset & !PAGE_MASK;
            let size = (end.wrapping_add(PAGE_MASK) & !PAGE_MASK).wrapping_sub(start);
            let mut ranges: Vec<CursorRange> = Vec::new();
            let mut range = CursorRange::default();
            let mut page_offset = offset & PAGE_MASK;
            let mut length_remaining = length;
            let mut last_address = u64::MAX;
            let mut all_writable = true;
            if let Some(walker) = &walker {
                walker.visit_pages_in_range(
                    hold,
                    OnFault::Return,
                    start,
                    size,
                    &mut |_address, page, page_writable| {
                        fatal_unless!(length_remaining != 0, DeathKind::CursorLength);
                        let physical = u64::from(page).wrapping_shl(APV_PAGE_SHIFT);
                        // NULL outside RAM, not an error: the range is used as it is
                        let host = map.as_ref().map_or(HostPtr::NULL, |m| {
                            m.virtual_address_for_physical(physical, PAGE)
                        });
                        let length_in_page = PAGE.wrapping_sub(page_offset).min(length_remaining);
                        let start_address = host.wrapping_add(page_offset);
                        let host_value = host.addr().to_u64();
                        if last_address.wrapping_add(PAGE) == host_value {
                            range.length = range.length.wrapping_add(length_in_page);
                        } else {
                            if range.length != 0 {
                                ranges.push(range);
                            }
                            range = CursorRange {
                                address: start_address,
                                length: length_in_page,
                            };
                        }
                        last_address = host_value;
                        page_offset = 0;
                        fatal_unless!(length_remaining >= length_in_page, DeathKind::CursorLength);
                        length_remaining = length_remaining.wrapping_sub(length_in_page);
                        if page_writable == 0 {
                            all_writable = false;
                        }
                        Ok(true)
                    },
                )?;
            }
            if range.length != 0 {
                ranges.push(range);
            }
            if writable {
                if all_writable {
                    return Ok(Some(WritableCursor::with_ranges(ranges).0));
                }
                return Ok(None);
            }
            Ok(Some(Cursor::with_ranges(ranges)))
        })();
        match walked {
            Ok(cursor) => Ok(cursor),
            Err(t) => {
                log::warn!(
                    "task {}: no cursor of {length:#x} bytes at {offset:#x}: {}",
                    self.id(),
                    exception(&t)
                );
                hold.land(t);
                Ok(None)
            }
        }
    }

    /// The host ranges behind the task's pages `page..` for `length` bytes (page-aligned, not
    /// empty), one per run of host-contiguous pages; `read_only` tells whether any leaf is.
    pub(super) fn address_ranges_for_virtual_page(
        &self,
        hold: Hold<'_>,
        page: u32,
        length: u64,
        read_only: &mut bool,
    ) -> RaiseResult<Option<P::AddressRangeArray>> {
        let offset = u64::from(page).wrapping_mul(PAGE);
        let task_length = self.task_length();
        let (end, overflow) = offset.overflowing_add(length);
        if overflow || end > task_length {
            return raise_caught(
                RaiseKind::OutOfRange,
                format_args!(
                    "{length:#x} bytes at task offset {offset:#x} run past the task's \
                     {task_length:#x}"
                ),
            );
        }
        if self.task_root_ptr().is_none() {
            return raise_caught(
                RaiseKind::NoPageTable,
                "no page table to map task memory through",
            );
        }
        if length & PAGE_MASK != 0 {
            return raise_caught(
                RaiseKind::UnalignedLength,
                format_args!("length {length:#x} to map is not page-aligned"),
            );
        }
        if length == 0 {
            return raise_caught(
                RaiseKind::ZeroLength,
                format_args!("map of 0 bytes at task offset {offset:#x}"),
            );
        }
        let mut ranges: Vec<ResourceAddressRange> = Vec::new();
        let mut writable = true;
        let mut range = ResourceAddressRange::default();
        let map = self.map();
        if let Some(walker) = self.walker() {
            walker.visit_pages_in_range(
                hold,
                OnFault::Return,
                offset,
                length,
                &mut |_address, guest_page, page_writable| {
                    let physical = u64::from(guest_page).wrapping_shl(APV_PAGE_SHIFT);
                    if writable && page_writable == 0 {
                        writable = false;
                    }
                    let host = map
                        .as_ref()
                        .map_or(HostPtr::NULL, |m| {
                            m.virtual_address_for_physical(physical, PAGE)
                        })
                        .addr()
                        .to_u64();
                    // The run's start plus one page: runs are cut at two pages.
                    if range.address.wrapping_add(PAGE) == host {
                        range.length = range.length.wrapping_add(PAGE);
                    } else {
                        if range.length != 0 {
                            ranges.push(range);
                        }
                        range = ResourceAddressRange {
                            address: host,
                            length: PAGE,
                        };
                    }
                    Ok(true)
                },
            )?;
        }
        if range.length != 0 {
            ranges.push(range);
        }
        *read_only = !writable;
        Ok(Some(self.platform.new_address_range_array(&ranges)?))
    }

    /// A Metal buffer over the task's pages from `page`, for `length` bytes, without a copy:
    /// the buffer's deallocator releases the mapping. The mapping is kept for good when there
    /// is no device or Metal makes no buffer: its address space stays taken.
    pub(super) fn new_buffer_for_virtual_page(
        &self,
        hold: Hold<'_>,
        page: u32,
        length: u64,
    ) -> RaiseResult<P::Buffer> {
        let mapping = self.new_virtual_mapping(
            hold,
            u64::from(page).wrapping_shl(APV_PAGE_SHIFT),
            length,
            false,
        )?;
        let Some(mapping) = mapping else {
            return raise_caught(
                RaiseKind::NoBufferMapping,
                format_args!("no mapping of {length:#x} bytes at task page {page:#x}"),
            );
        };
        let address = mapping.virtual_address();
        // The reference the deallocator's release ends
        let kept = Arc::clone(&mapping);
        let Some(device) = self.device_handle() else {
            keep_forever(kept, Kept::AddressSpace);
            drop(mapping);
            return raise_caught(
                RaiseKind::NoBuffer,
                format_args!("Metal made no buffer over task page {page:#x}"),
            );
        };
        let buffer = Self::no_copy_buffer(&device, address, length, mapping)?;
        let Some(buffer) = buffer else {
            // Metal released the deallocator's reference; the kept one stays
            keep_forever(kept, Kept::AddressSpace);
            return raise_caught(
                RaiseKind::NoBuffer,
                format_args!("Metal made no buffer over task page {page:#x}"),
            );
        };
        // The buffer's deallocator owns the mapping from here
        drop(kept);
        Ok(buffer)
    }

    /// A buffer over `address..address + length`, whose deallocator releases `mapping`.
    #[allow(unsafe_code)] // architecture 17.3: the call of an `unsafe fn` over host memory
    fn no_copy_buffer(
        device: &P::Device,
        address: HostPtr,
        length: u64,
        mapping: Arc<Mapping<P>>,
    ) -> RaiseResult<Option<P::Buffer>> {
        // SAFETY: `address..address+length` is `mapping`'s VM range, and the deallocator owns
        // `mapping`, so the range stays mapped until Metal frees the buffer and runs it.
        let bytes = unsafe { crate::platform::HostRange::new(address, length) };
        device.new_buffer_over_host_memory(
            OnFault::Return,
            bytes,
            0,
            Some(Box::new(move || drop(mapping))),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::*;
    use super::*;
    use crate::guest::GuestPtr;
    use crate::raise::death::DeathKind as Fault;
    use crate::resources::host::ResourceHost;
    use crate::testkit::capture::catch_death;
    use crate::testkit::objects::Model;
    use crate::testkit::script::Step;

    /// The raise a call dies with, with no handler live.
    fn raised<T>(f: impl FnOnce() -> RaiseResult<T>) -> Option<&'static str> {
        catch_death(f).err().map(|d| {
            assert_eq!(d.kind, Fault::UnhandledRaise);
            d.file
        })
    }

    /// The raise a call answers under a handler, by the site it dies at with none live.
    fn caught<T>(f: impl for<'i> FnOnce(Hold<'i>) -> RaiseResult<T>) -> Option<&'static str> {
        let hold = Hold::NONE;
        f(hold).err().map(|r| {
            let d = r.site();
            hold.land(r);
            assert_eq!(d.kind, Fault::UnhandledRaise);
            d.file
        })
    }

    /// A copy reads across pages, and `>=` refuses one ending at the task's end
    #[test]
    fn copy_reads_across_pages_and_refuses_the_end() {
        let tx = ready(&rw_pages(4));
        let src: Vec<u8> = (0..0x100u32).map(|i| i as u8).collect();
        tx.ram.write(page_pa(DATA_PAGE) + PAGE - 0x80, &src[..0x80]);
        tx.ram.write(page_pa(DATA_PAGE + 1), &src[0x80..]);
        let mut dst = vec![0u8; 0x100];
        tx.task
            .copy_from_virtual_offset(Hold::NONE, OnFault::Die, PAGE - 0x80, &mut dst)
            .unwrap();
        assert_eq!(dst, src);
        let mut last = vec![0u8; 0x10];
        assert_eq!(
            raised(|| tx.task.copy_from_virtual_offset(
                Hold::NONE,
                OnFault::Die,
                TASK_LEN - 0x10,
                &mut last
            )),
            Some(RaiseKind::OutOfRange.id())
        );
    }

    /// Mappings check their range, alignment and length; one read-only leaf is nil for a
    /// writable request, read-only otherwise
    #[test]
    fn new_virtual_mapping_checks_and_writability() {
        let mut entries = rw_pages(4);
        entries[2] &= !W;
        let tx = ready(&entries);
        let throws =
            |off: u64, len: u64| caught(|hold| tx.task.new_virtual_mapping(hold, off, len, false));
        assert_eq!(throws(0, TASK_LEN + PAGE), Some(RaiseKind::OutOfRange.id()));
        assert_eq!(throws(0x10, PAGE), Some(RaiseKind::UnalignedOffset.id()));
        assert_eq!(throws(0, 0x10), Some(RaiseKind::UnalignedLength.id()));
        assert_eq!(throws(0, 0), Some(RaiseKind::ZeroLength.id()));
        let m = tx
            .task
            .new_virtual_mapping(Hold::NONE, 0, 2 * PAGE, true)
            .unwrap()
            .unwrap();
        assert!(m.writable());
        assert!(
            tx.task
                .new_virtual_mapping(Hold::NONE, 0, 4 * PAGE, true)
                .unwrap()
                .is_none()
        );
        let m = tx
            .task
            .new_virtual_mapping(Hold::NONE, 0, 4 * PAGE, false)
            .unwrap()
            .unwrap();
        assert!(!m.writable());
    }

    /// Before the handshake there is no page table: a mapping raises
    #[test]
    fn mapping_before_the_handshake_raises() {
        let tx = fixture(1, &rw_pages(4));
        tx.task
            .length
            .store(TASK_LEN, std::sync::atomic::Ordering::SeqCst);
        assert_eq!(
            caught(|hold| tx.task.new_virtual_mapping(hold, 0, PAGE, false)),
            Some(RaiseKind::NoPageTable.id())
        );
    }

    /// The mapped address points into the mapping, which comes back with it; a writable request
    /// over a read-only page raises
    #[test]
    fn mapped_address_for_offset_points_into_the_mapping() {
        let mut entries = rw_pages(4);
        entries[3] &= !W;
        let tx = ready(&entries);
        tx.ram.write(page_pa(DATA_PAGE) + 0x123, &[9, 8, 7]);
        let a = tx
            .task
            .mapped_address_for_offset(Hold::NONE, 0x123, 3, false)
            .unwrap();
        assert!(a.mapping.is_some());
        assert_eq!(
            GuestPtr::new(a.address).read_u32(0, DeathKind::NullReply) & 0xff_ffff,
            0x07_0809
        );
        drop(a);
        assert_eq!(
            caught(|hold| tx
                .task
                .mapped_address_for_offset(hold, 3 * PAGE, 4, true)
                .map(|_| ())),
            Some(RaiseKind::NoMapping.id())
        );
    }

    /// `>`: a cursor ending at the task's end is accepted; past it, or empty, is logged and nil;
    /// a read-only page refuses a writable cursor without a line
    #[test]
    fn cursors_follow_the_page_table() {
        let mut entries = rw_pages(4);
        entries[3] &= !W;
        let tx = ready(&entries);
        let lines = Lines::capture();
        assert!(
            tx.task
                .cursor(Hold::NONE, 0, TASK_LEN, false)
                .unwrap()
                .is_some()
        );
        assert!(
            tx.task
                .cursor(Hold::NONE, 0, TASK_LEN + 1, false)
                .unwrap()
                .is_none()
        );
        assert!(tx.task.cursor(Hold::NONE, 0, 0, false).unwrap().is_none());
        assert!(
            tx.task
                .writable_cursor_from_virtual_offset(Hold::NONE, 0, 3 * PAGE)
                .unwrap()
                .is_some()
        );
        assert!(
            tx.task
                .writable_cursor_from_virtual_offset(Hold::NONE, 0, TASK_LEN)
                .unwrap()
                .is_none()
        );
        assert_eq!(
            lines.take(),
            [
                format!(
                    "task 7: no cursor of {:#x} bytes at 0x0: past the task's end",
                    TASK_LEN + 1
                ),
                "task 7: no cursor of 0 bytes at 0x0".to_owned()
            ]
        );
    }

    /// A cursor before the handshake set the task root is nil
    #[test]
    fn cursor_before_the_handshake_is_nil() {
        let tx = fixture(1, &rw_pages(4));
        assert!(
            tx.task
                .cursor(Hold::NONE, 0, PAGE, false)
                .unwrap()
                .is_none()
        );
    }

    /// A hole in the page table is caught: the cursor is nil
    #[test]
    fn cursor_over_a_hole_is_nil() {
        let mut entries = rw_pages(4);
        entries[1] = 0;
        let tx = ready(&entries);
        assert!(
            tx.task
                .cursor(Hold::NONE, 0, 2 * PAGE, false)
                .unwrap()
                .is_none()
        );
    }

    /// A run is cut when it reaches two pages; any read-only leaf is reported; the checks raise
    #[test]
    fn address_ranges_cut_runs_at_two_pages() {
        let mut entries = rw_pages(4);
        entries[3] &= !W;
        let tx = ready(&entries);
        let mut ro = false;
        let arr = tx
            .task
            .address_ranges_for_virtual_page(Hold::NONE, 0, 3 * PAGE, &mut ro)
            .unwrap()
            .unwrap();
        assert!(!ro);
        let host = |i: u32| tx.ram.host_addr(page_pa(DATA_PAGE + i)).unwrap().addr() as u64;
        let Model::Ranges(r) = &arr.state().model else {
            panic!("not ranges")
        };
        let got: Vec<(u64, u64)> = r.iter().map(|x| (x.address, x.length)).collect();
        assert_eq!(got, vec![(host(0), 2 * PAGE), (host(2), PAGE)]);
        tx.task
            .address_ranges_for_virtual_page(Hold::NONE, 0, 4 * PAGE, &mut ro)
            .unwrap();
        assert!(ro);
        let raises = |page: u32, len: u64| {
            let mut ro = false;
            caught(|hold| {
                tx.task
                    .address_ranges_for_virtual_page(hold, page, len, &mut ro)
            })
        };
        assert_eq!(raises(4, PAGE), Some(RaiseKind::OutOfRange.id()));
        assert_eq!(raises(0, 0x10), Some(RaiseKind::UnalignedLength.id()));
        assert_eq!(raises(0, 0), Some(RaiseKind::ZeroLength.id()));
    }

    /// A buffer wraps the mapping without a copy; a refused buffer raises
    #[test]
    fn new_buffer_for_virtual_page_wraps_the_mapping() {
        let tx = ready(&rw_pages(4));
        let b = tx
            .task
            .new_buffer_for_virtual_page(Hold::NONE, 1, PAGE)
            .unwrap();
        drop(b);
        assert!(
            tx.sels()
                .contains(&"newBufferWithBytesNoCopy:length:options:deallocator:")
        );
        tx.p.script().push(
            "newBufferWithBytesNoCopy:length:options:deallocator:",
            Step::Nil,
        );
        assert_eq!(
            caught(|hold| tx.task.new_buffer_for_virtual_page(hold, 1, PAGE)),
            Some(RaiseKind::NoBuffer.id())
        );
    }

    #[test]
    fn data_from_virtual_offset_copies() {
        let tx = ready(&rw_pages(4));
        tx.ram.write(page_pa(DATA_PAGE) + 4, &[1, 2, 3]);
        let d = ResourceHost::data_from_virtual_offset(&*tx.task, Hold::NONE, 4, 3).unwrap();
        assert_eq!(d, Some(vec![1, 2, 3]));
    }

    /// A 4 GiB read out of range raises without writing the buffer (the rate-map reset job's
    /// read, with no handler)
    #[test]
    #[allow(clippy::disallowed_methods)] // wall time is the point: a 4 GiB fill takes seconds
    fn data_from_virtual_offset_large_out_of_range_raises() {
        let tx = ready(&rw_pages(4));
        let started = std::time::Instant::now();
        assert_eq!(
            raised(|| tx
                .task
                .data_from_virtual_offset(Hold::NONE, OnFault::Die, 4, 0xffff_ffff)),
            Some(RaiseKind::OutOfRange.id())
        );
        // writing 4 GiB of zeros takes seconds; a lazy zeroed allocation does not
        assert!(started.elapsed() < std::time::Duration::from_secs(1));
    }

    /// A length past the largest task raises before anything is allocated (the rate-map reset
    /// job's read, with no handler)
    #[test]
    fn a_length_past_any_task_raises() {
        let tx = ready(&rw_pages(4));
        for length in [MAX_LOCAL_TASK_LENGTH + 1, u64::MAX] {
            assert_eq!(
                raised(|| tx
                    .task
                    .data_from_virtual_offset(Hold::NONE, OnFault::Die, 0, length)),
                Some(RaiseKind::BufferLength.id())
            );
        }
    }

    /// A function whose IR sits over a hole in the task's page table, IR the guest has not
    /// mapped: the resolver's lookup raises the function lookup's failure, where a nil function
    /// would reach Metal and end the process, and makes the function once the guest maps the page
    #[test]
    fn a_function_over_a_hole_raises_its_lookup_until_the_page_is_mapped() {
        use crate::raise::RaiseId;
        use crate::resources::manager::create::function::ObjectFunction;
        use crate::resources::manager::records::OperationHeader;
        use crate::resources::manager::resolver::function_with_function_ref;
        use crate::resources::manager::{ObjectKind, RaiseKind as ManagerRaise};
        use crate::testkit::platform::FAKE_REGISTRY_ID;
        use zerocopy::IntoBytes;

        let mut entries = rw_pages(4);
        entries[2] = 0;
        let tx = fixture(1, &entries);
        // entry 1 at task offset 12: a function whose 20-byte record is at task offset 0x100
        tx.ram
            .write_u32(page_pa(ROOT_PAGE) + ROOT_OBJECT_LIST_LENGTH, 24);
        tx.ram.write_u32(page_pa(DATA_PAGE) + 12, 6 | (20 << 8));
        tx.ram.write_u64(page_pa(DATA_PAGE) + 16, 0x100);
        let ir_offset = 2 * PAGE + 0x10;
        let function = ObjectFunction {
            offset: ir_offset,
            length: 4,
        };
        let header = OperationHeader {
            command_type: 15,
            field4: 0,
        };
        tx.ram
            .write(page_pa(DATA_PAGE) + 0x100, function.as_bytes());
        tx.ram.write(page_pa(DATA_PAGE) + 0x10c, header.as_bytes());
        tx.ram.write(page_pa(DATA_PAGE + 2) + 0x10, &[9, 8, 7, 6]);
        assert_eq!(
            handshake_with(&tx, ROOT_PAGE, TASK_LEN, FAKE_REGISTRY_ID, true),
            vec![(1, pid())]
        );
        tx.task.ensure_object_list_mapped(Hold::NONE).unwrap();
        let (_, manager, _) = tx.task.parts();
        // The resource manager's lines, which `Lines` leaves out
        let capture = crate::testkit::logs::capture();
        let raised = function_with_function_ref(Hold::NONE, &manager, 1)
            .err()
            .unwrap();
        assert_eq!(
            raised.id(),
            RaiseId::Manager(ManagerRaise::Lookup(ObjectKind::Function))
        );
        let want = format!(
            "function 1 not made: its IR, 0x4 bytes at task offset {ir_offset:#x}, is not \
             mapped: a hole in the page table"
        );
        let lines: Vec<String> = capture.take().into_iter().map(|(_, _, m)| m).collect();
        assert!(lines.contains(&want), "{want}: {lines:?}");
        assert!(!tx.sels().contains(&"newLibraryWithData:error:"));
        // the guest maps the page: the next lookup makes the function from its IR
        tx.ram.write_u32(page_pa(PT_PAGE) + 8, (DATA_PAGE + 2) | W);
        tx.p.world().set_library_functions(vec!["main0".to_owned()]);
        assert!(
            function_with_function_ref(Hold::NONE, &manager, 1)
                .unwrap()
                .is_some()
        );
        let made = tx.p.call_log().calls_to("newLibraryWithData:error:");
        assert_eq!(
            made.last().unwrap().args,
            vec![crate::testkit::calllog::Arg::Bytes(vec![9, 8, 7, 6])]
        );
    }

    /// A failed read is logged with the byte of the copy where the page's bytes would have
    /// gone, and the copy goes on
    #[test]
    fn failed_read_logs_where_it_would_have_gone() {
        // page 1 of the task names a page outside guest RAM
        let mut entries = rw_pages(4);
        entries[1] = ((RAM_PA + RAM_LEN) >> APV_PAGE_SHIFT) as u32 | W;
        let tx = ready(&entries);
        let lines = Lines::capture();
        let mut dst = vec![0u8; (PAGE + 0x10).to_usize()];
        tx.task
            .copy_from_virtual_offset(Hold::NONE, OnFault::Die, PAGE - 0x10, &mut dst)
            .unwrap();
        // page 0 gave 0x10 bytes, so the position is byte 0x10 when page 1 fails
        let want = format!(
            "task 7: {PAGE:#x} bytes of guest memory at {:#x} not read, for byte 0x10 of a copy",
            RAM_PA + RAM_LEN
        );
        assert!(lines.take().contains(&want), "{want}");
    }
}
