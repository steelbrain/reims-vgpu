//! The MMIO register file: six 64-bit registers and the read-only features offered.
//!
//! Reads are unsynchronised seq_cst loads. The ring base, ring size, descriptor base and enable
//! writes run on the device's queue through a `sync` from the writing thread, so each waits for a
//! running ring batch; the write counter (the doorbell) and the read counter are stored on the
//! caller's thread.

// Guest data flows through here: arithmetic is spelt out.
#![cfg_attr(not(test), deny(clippy::arithmetic_side_effects))]

use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::guest::GuestPtr;
use crate::host::PhysRange;
use crate::iosurface::DeathKind;
use crate::iosurface::descriptor::Form;
use crate::iosurface::device::{HostDeviceCore, QueueState};
use crate::iosurface::ring::RING_COMMAND_SIZE;
use crate::iosurface::table::{MAPPING_ID_COUNT, TABLE_ENTRY_SIZE};
use crate::num::ToUsize;
use crate::platform::Platform;
use crate::raise::{Kept, keep_forever};

/// The ring's guest physical base.
pub(super) const REG_RING_BASE: u64 = 0x1000;
/// The ring's length in commands.
pub(super) const REG_RING_SIZE: u64 = 0x1008;
/// The descriptor table's guest physical base.
pub(super) const REG_DESCRIPTOR_BASE: u64 = 0x1010;
/// The ring's write counter: a write is the doorbell.
pub(super) const REG_RING_WR_COUNTER: u64 = 0x1018;
/// The ring's read counter.
pub(super) const REG_RING_RD_COUNTER: u64 = 0x1020;
/// The features the device offers, read-only: bit 0, the second descriptor form
/// ([`HostDeviceCore::features_offered`]). Name is ours: the guest's mapper reads it before it
/// writes the features it enables to [`REG_DESC_ENABLE`], and takes the second form only when
/// bit 0 is set.
pub(super) const REG_FEATURES: u64 = 0x1028;
/// The descriptor form (bit 0: the second form).
pub(super) const REG_DESC_ENABLE: u64 = 0x1030;

/// The six registers.
pub(super) struct Registers {
    /// [`REG_RING_BASE`].
    pub(super) ring_base: AtomicU64,
    /// [`REG_RING_SIZE`].
    pub(super) ring_size: AtomicU64,
    /// [`REG_DESCRIPTOR_BASE`].
    pub(super) descriptor_base: AtomicU64,
    /// [`REG_RING_WR_COUNTER`]: free-running.
    pub(super) ring_wr_counter: AtomicU64,
    /// [`REG_RING_RD_COUNTER`]: free-running.
    pub(super) ring_rd_counter: AtomicU64,
    /// [`REG_DESC_ENABLE`].
    pub(super) desc_enable: AtomicU64,
}

impl Registers {
    /// Every register 0.
    pub(super) fn new() -> Registers {
        Registers {
            ring_base: AtomicU64::new(0),
            ring_size: AtomicU64::new(0),
            descriptor_base: AtomicU64::new(0),
            ring_wr_counter: AtomicU64::new(0),
            ring_rd_counter: AtomicU64::new(0),
            desc_enable: AtomicU64::new(0),
        }
    }

    /// The registers as `Debug` fields.
    pub(super) fn debug_fields(&self, d: &mut std::fmt::DebugStruct<'_, '_>) {
        d.field("reg_ring_base", &self.ring_base)
            .field("reg_ring_size", &self.ring_size)
            .field("reg_descriptor_base", &self.descriptor_base)
            .field("reg_ring_wr_counter", &self.ring_wr_counter)
            .field("reg_ring_rd_counter", &self.ring_rd_counter)
            .field("reg_desc_enable", &self.desc_enable);
    }
}

impl<P: Platform> HostDeviceCore<P> {
    /// The register at `offset`; all ones for an offset that names none.
    pub fn mmio_read(&self, offset: u64) -> u64 {
        // The registers are 64 bits wide; QEMU's 32-bit declaration truncates (testing.md 7.3).
        let r = &self.regs;
        match offset {
            REG_RING_BASE => r.ring_base.load(Ordering::SeqCst),
            REG_RING_SIZE => r.ring_size.load(Ordering::SeqCst),
            REG_DESCRIPTOR_BASE => r.descriptor_base.load(Ordering::SeqCst),
            REG_RING_WR_COUNTER => r.ring_wr_counter.load(Ordering::SeqCst),
            REG_RING_RD_COUNTER => r.ring_rd_counter.load(Ordering::SeqCst),
            REG_FEATURES => self.features_offered(),
            REG_DESC_ENABLE => r.desc_enable.load(Ordering::SeqCst),
            _ => u64::MAX,
        }
    }

    /// The descriptor table's guest physical base, [`REG_DESCRIPTOR_BASE`]: 0 until the guest's
    /// mapper writes one, and again after a write of 0 or of the descriptor enable. Read-only, an
    /// unsynchronised load, as the guest's read is.
    pub fn descriptor_base(&self) -> u64 {
        self.regs.descriptor_base.load(Ordering::SeqCst)
    }

    /// Writes the register at `offset`; the features offered and every other offset ignore it.
    pub fn mmio_write(&self, offset: u64, value: u64) {
        // The registers are 64 bits wide; QEMU's 32-bit declaration zero-extends (testing.md
        // 7.3).
        match offset {
            REG_RING_BASE => self.on_queue(move |this, st| this.set_ring_base(st, value)),
            REG_RING_SIZE => self.on_queue(move |this, st| this.set_ring_size(st, value)),
            REG_DESCRIPTOR_BASE => {
                self.on_queue(move |this, st| this.set_descriptor_base(st, value));
            }
            REG_RING_WR_COUNTER => {
                self.regs.ring_wr_counter.store(value, Ordering::SeqCst);
                if let Some(doorbell) = &*self.doorbell.lock() {
                    doorbell.merge(1);
                }
            }
            REG_RING_RD_COUNTER => {
                // Stored only while no ring is mapped, read off the queue.
                let no_ring = self.state.lock().ring.is_none();
                if no_ring {
                    self.regs.ring_rd_counter.store(value, Ordering::SeqCst);
                }
            }
            REG_DESC_ENABLE => {
                self.on_queue(move |this, st| this.set_descriptor_enable(st, value));
            }
            _ => {}
        }
    }

    /// The features [`REG_FEATURES`] reads: bit 0, the second descriptor form, only where it
    /// leaves no mapping id without a descriptor ([`second_form_holds_every_id`]).
    pub(super) fn features_offered(&self) -> u64 {
        u64::from(second_form_holds_every_id(self.page_size()))
    }

    /// Runs `f` on the device's queue over its state, waiting for it.
    fn on_queue(&self, f: impl FnOnce(&Self, &mut QueueState<P>) + Send + 'static) {
        let Some(this) = self.block_self() else {
            return;
        };
        self.hw_queue.sync(move || {
            let mut st = this.state.lock();
            f(&this, &mut st);
        });
    }

    /// The descriptors' form, as the enable register has it.
    pub(super) fn form(&self) -> Form {
        Form::of(self.regs.desc_enable.load(Ordering::SeqCst))
    }

    /// A new ring base unmaps the ring and keeps the size: the ring stays unmapped until the
    /// size is written again.
    pub(super) fn set_ring_base(&self, st: &mut QueueState<P>, base: u64) {
        if st.ring.is_some() {
            self.maps.lock().ring = None;
            st.ring = None;
        }
        self.regs.ring_base.store(base, Ordering::SeqCst);
    }

    /// A size needs a base, and is stored before the ring is mapped: an overflowing length or a
    /// refused mapping leaves the register at the new size and the ring at the old one. The
    /// mapping asserts a whole number of guest pages: a size that is not a multiple of 1024
    /// commands (256 on x86-64) is host death.
    pub(super) fn set_ring_size(&self, st: &mut QueueState<P>, size: u64) {
        let ring_base = self.regs.ring_base.load(Ordering::SeqCst);
        if size == 0 || ring_base == 0 {
            return;
        }
        self.regs.ring_size.store(size, Ordering::SeqCst);
        let ring_size = self.regs.ring_size.load(Ordering::SeqCst);
        let (length, overflow) = ring_size.overflowing_mul(RING_COMMAND_SIZE);
        if overflow {
            log::warn!(
                "ring size {ring_size} is too large: its length in bytes overflows; the ring stays \
                 as it was"
            );
            return;
        }
        let range = PhysRange {
            physical_address: self.regs.ring_base.load(Ordering::SeqCst),
            physical_length: length,
        };
        let mapped = {
            let mut maps = self.maps.lock();
            // A length that is not whole guest pages asserts in the map.
            let mapping = maps
                .memory_map
                .as_ref()
                .and_then(|m| m.new_range_mapping(&[range], length, 0));
            // The mapping replaced is kept for good, and the ring may still point into it when
            // the new one is refused, so it stays mapped.
            if let Some(old) = maps.ring.take() {
                keep_forever(old, Kept::StaleReader);
            }
            maps.ring = mapping;
            maps.ring.as_ref().map(|m| m.virtual_address())
        };
        match mapped {
            Some(address) => {
                let ring = GuestPtr::new(address);
                st.ring = Some(ring);
                // The ring's first word is read only when the debug line is enabled; no `maps`
                // guard is held while it logs.
                log::debug!(
                    "ring mapped at {:#x}, first word {:08x}",
                    address.addr(),
                    self.mapped_access(ring, 4, DeathKind::RingPeek, || ring
                        .read_u32(0, DeathKind::RingPeek))
                );
            }
            None => {
                // The ring is unchanged: still the old one.
                log::warn!(
                    "ring of {length:#x} bytes at {:#x} could not be mapped; the ring stays as it \
                     was",
                    range.physical_address
                );
            }
        }
    }

    /// Keeps bit 0, drops the descriptor table (tearing every surface down) and sets the
    /// descriptors per page for the form.
    pub(super) fn set_descriptor_enable(&self, st: &mut QueueState<P>, enable: u64) {
        log::debug!("descriptor enable {enable:#x}");
        self.regs.desc_enable.store(enable & 1, Ordering::SeqCst);
        self.set_descriptor_base(st, 0);
        st.descriptors_per_page = self.page_size() / self.form().size();
    }

    /// Tears every surface down, drops the table and every cached descriptor page, stores the
    /// base and maps one page of table there. A table the map refuses is left unmapped.
    pub(super) fn set_descriptor_base(&self, st: &mut QueueState<P>, base: u64) {
        self.teardown_surfaces(st);
        let mut maps = self.maps.lock();
        if st.table.is_some() {
            maps.table = None;
            st.table = None;
        }
        let count = st.table_len.to_usize();
        for (page, mapping) in st.pages.iter_mut().zip(maps.pages.iter_mut()).take(count) {
            if page.is_some() {
                *mapping = None;
                *page = None;
            }
        }
        self.regs.descriptor_base.store(base, Ordering::SeqCst);
        if base != 0 {
            maps.table = maps.new_contiguous_mapping(base, self.page_size());
            st.table = maps
                .table
                .as_ref()
                .map(|m| GuestPtr::new(m.virtual_address()));
            drop(maps);
            match st.table {
                Some(table) => log::debug!(
                    "descriptor table at {base:#x} mapped at {:#x}",
                    table.addr().addr()
                ),
                None => log::debug!("descriptor table at {base:#x} could not be mapped"),
            }
        }
    }
}

/// Whether one page of descriptor table, `page / 4` entries, each naming a page of second-form
/// descriptors (`page / 0x400`), reaches every mapping id. On 16 KiB pages it holds 65,536; on
/// x86-64's 4 KiB ones 4,096, short of the 8192 ids, where the first form's 0x200 bytes give
/// exactly 8192. The guest's mapper takes the second form whenever it is offered, and then
/// refuses to start when its own table, sized the same way, holds fewer than 8192 mappings: a
/// 4 KiB-page guest starts only on the first form.
fn second_form_holds_every_id(page: u64) -> bool {
    let entries = page / const { NonZeroU64::new(TABLE_ENTRY_SIZE).unwrap() };
    let per_page = page / Form::Second.size();
    entries.saturating_mul(per_page) >= MAPPING_ID_COUNT
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::iosurface::ring::IoReturn;
    use crate::iosurface::testing::*;
    use crate::testkit::calllog::Arg;
    use crate::testkit::capture::expect_death;
    use crate::testkit::logs;
    use crate::testkit::script::Step;
    use crate::testkit::vm::KERN_NO_SPACE;

    mod mmio {
        use super::*;

        /// Every offset; 0x1028 reads the features offered (the second form on 16 KiB pages,
        /// none on 4 KiB ones) and ignores writes; other offsets read all ones. The registers
        /// are 64 bits wide (testing.md 7.3).
        #[test]
        fn register_file() {
            let r = rig();
            r.dev.mmio_write(REG_RING_BASE, 0x1_0000_4000);
            assert_eq!(r.dev.mmio_read(REG_RING_BASE), 0x1_0000_4000);
            let offered = u64::from(PAGE == 0x4000);
            assert_eq!(r.dev.mmio_read(REG_FEATURES), offered);
            r.dev.mmio_write(REG_FEATURES, 7);
            assert_eq!(r.dev.mmio_read(REG_FEATURES), offered);
            for off in [0, 0x1004, 0x1038, 0xfff, u64::MAX] {
                r.dev.mmio_write(off, 9);
                assert_eq!(r.dev.mmio_read(off), u64::MAX);
            }
            r.dev.mmio_write(REG_DESC_ENABLE, 0xff);
            assert_eq!(r.dev.mmio_read(REG_DESC_ENABLE), 1);
            r.dev.mmio_write(REG_DESCRIPTOR_BASE, TABLE);
            assert_eq!(r.dev.mmio_read(REG_DESCRIPTOR_BASE), TABLE);
        }

        /// 0x1020 is stored only while no ring is mapped.
        #[test]
        fn read_counter_only_without_ring() {
            let r = rig();
            r.dev.mmio_write(REG_RING_RD_COUNTER, 5);
            assert_eq!(r.dev.mmio_read(REG_RING_RD_COUNTER), 5);
            r.dev.mmio_write(REG_RING_BASE, RING);
            r.dev.mmio_write(REG_RING_SIZE, 1024);
            r.dev.mmio_write(REG_RING_RD_COUNTER, 7);
            assert_eq!(r.dev.mmio_read(REG_RING_RD_COUNTER), 5);
        }

        /// 0x1018 stores the counter and rings the doorbell: a mapped ring runs.
        #[test]
        fn write_counter_wakes() {
            let r = rig();
            r.setup(true);
            r.command(0, 7, 1);
            r.doorbell(1);
            assert_eq!(r.dev.mmio_read(REG_RING_WR_COUNTER), 1);
            assert_eq!(r.dev.mmio_read(REG_RING_RD_COUNTER), 1);
            assert_eq!(r.status(0), IoReturn::Unsupported.code());
            assert_eq!(*r.irqs.lock(), vec![0]);
        }
    }

    mod features {
        use super::*;

        /// The second form is offered on 16 KiB pages, whose table holds 65,536 of its
        /// descriptors, and not on 4 KiB ones, whose table holds 4,096 of the 8192 ids; the
        /// first form would hold exactly 8192 there.
        #[test]
        fn second_form_by_page_size() {
            assert!(second_form_holds_every_id(0x4000));
            assert!(!second_form_holds_every_id(0x1000));
            assert_eq!(
                (0x1000 / TABLE_ENTRY_SIZE) * (0x1000 / Form::First.size()),
                MAPPING_ID_COUNT
            );
        }
    }

    mod ring {
        use super::*;

        /// A size needs a base (else nothing, the register kept); base and size map the ring
        /// (16 bytes a command); a new base unmaps it and keeps the size.
        #[test]
        fn size_base_and_unmap() {
            let r = rig();
            r.dev.mmio_write(REG_RING_SIZE, 1024);
            assert_eq!(r.dev.mmio_read(REG_RING_SIZE), 0);
            r.dev.mmio_write(REG_RING_BASE, RING);
            r.dev.mmio_write(REG_RING_SIZE, 0);
            assert_eq!(r.dev.mmio_read(REG_RING_SIZE), 0);
            let mark = r.p.call_log().mark();
            r.dev.mmio_write(REG_RING_SIZE, 1024);
            assert_eq!(r.dev.mmio_read(REG_RING_SIZE), 1024);
            let alloc = r.p.call_log().since(mark);
            assert_eq!(alloc[0].sel, "mach_vm_allocate");
            assert_eq!(alloc[0].args[0], Arg::U64(0x4000));
            assert!(r.dev.state.lock().ring.is_some());
            let mark = r.p.call_log().mark();
            r.dev.mmio_write(REG_RING_BASE, 0);
            assert!(r.dev.state.lock().ring.is_none());
            assert!(r.dev.maps.lock().ring.is_none());
            assert_eq!(r.p.call_log().sels_since(mark), vec!["mach_vm_deallocate"]);
            assert_eq!(r.dev.mmio_read(REG_RING_SIZE), 1024);
        }

        /// A second size leaks the first mapping (no deallocate); a refused mapping keeps the
        /// old ring pointer; an overflowing size stores the register and maps nothing.
        #[test]
        fn replaced_refused_and_overflowing_mappings() {
            let r = rig();
            r.dev.mmio_write(REG_RING_BASE, RING);
            r.dev.mmio_write(REG_RING_SIZE, 1024);
            let first = r.dev.state.lock().ring;
            let mark = r.p.call_log().mark();
            r.dev.mmio_write(REG_RING_SIZE, 2048);
            assert_eq!(r.p.call_log().count("mach_vm_deallocate"), 0);
            let second = r.dev.state.lock().ring;
            assert_ne!(first, second);
            assert!(!r.p.call_log().sels_since(mark).is_empty());
            r.p.script()
                .push("mach_vm_allocate", Step::value(KERN_NO_SPACE));
            r.dev.mmio_write(REG_RING_SIZE, 1024);
            {
                let st = r.dev.state.lock();
                assert_eq!(st.ring, second);
                assert!(r.dev.maps.lock().ring.is_none());
            }
            assert_eq!(r.dev.mmio_read(REG_RING_SIZE), 1024);
            let mark = r.p.call_log().mark();
            r.dev.mmio_write(REG_RING_SIZE, 1 << 60);
            assert_eq!(r.dev.mmio_read(REG_RING_SIZE), 1 << 60);
            assert!(r.p.call_log().sels_since(mark).is_empty());
            assert_eq!(r.dev.state.lock().ring, second);
        }

        /// The ring-mapped debug line prints the ring's host address and its first command word
        /// in eight hex digits, read from the guest ring.
        #[test]
        fn mapped_ring_logs_its_first_command() {
            let r = rig();
            r.ram.write_u32(RING, 0xab);
            r.sync();
            let logs = logs::capture();
            r.dev.mmio_write(REG_RING_BASE, RING);
            r.dev.mmio_write(REG_RING_SIZE, 1024);
            let ring = r.dev.state.lock().ring.unwrap().addr().addr();
            assert_eq!(
                lines_from(&logs, module_path!()),
                [(
                    log::Level::Debug,
                    format!("ring mapped at {ring:#x}, first word 000000ab")
                )]
            );
        }

        /// A ring that is not a whole number of 16 KiB pages asserts in the memory map.
        #[test]
        fn unaligned_size_asserts() {
            expect_death(
                crate::memory::DeathKind::MappingLengthUnaligned.site(),
                || {
                    let r = rig();
                    r.dev.mmio_write(REG_RING_BASE, RING);
                    r.dev.mmio_write(REG_RING_SIZE, 1);
                },
            );
        }
    }

    mod descriptors {
        use super::*;

        /// The enable keeps bit 0, sets the form's descriptors per page and drops the table.
        #[test]
        fn enable_sets_the_form() {
            let r = rig();
            r.dev.mmio_write(REG_DESCRIPTOR_BASE, TABLE);
            assert!(r.dev.state.lock().table.is_some());
            r.dev.mmio_write(REG_DESC_ENABLE, 3);
            assert_eq!(r.dev.mmio_read(REG_DESC_ENABLE), 1);
            assert_eq!(r.dev.state.lock().descriptors_per_page, PAGE / 0x400);
            assert!(r.dev.state.lock().table.is_none());
            assert_eq!(r.dev.mmio_read(REG_DESCRIPTOR_BASE), 0);
            r.dev.mmio_write(REG_DESC_ENABLE, 2);
            assert_eq!(r.dev.mmio_read(REG_DESC_ENABLE), 0);
            assert_eq!(r.dev.state.lock().descriptors_per_page, PAGE / 0x200);
        }

        /// A base maps one page of table; a new base tears the surfaces down and drops the
        /// cached descriptor pages; a base the map refuses leaves the table unmapped.
        #[test]
        fn base_maps_and_drops() {
            let r = rig();
            r.mapped_v2(1, 1);
            assert!(r.dev.state.lock().pages[0].is_some());
            let mark = r.p.call_log().mark();
            r.dev.mmio_write(REG_DESCRIPTOR_BASE, HOLE);
            let sels = r.p.call_log().sels_since(mark);
            assert!(sels.contains(&"CFRelease"));
            // The table and the descriptor page, then the refused new table's allocation.
            assert_eq!(
                sels.iter().filter(|s| **s == "mach_vm_deallocate").count(),
                3
            );
            let st = r.dev.state.lock();
            assert!(st.pages.iter().all(Option::is_none));
            assert!(st.table.is_none());
            let maps = r.dev.maps.lock();
            assert!(maps.table.is_none());
            assert!(maps.pages.iter().all(Option::is_none));
            drop(maps);
            assert!(st.slots[1].surface.is_none());
            drop(st);
            assert_eq!(r.dev.mmio_read(REG_DESCRIPTOR_BASE), HOLE);
        }
    }
}
