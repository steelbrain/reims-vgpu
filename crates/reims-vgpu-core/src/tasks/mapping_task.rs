//! The mapping task: a VM reservation standing in for the host's task when the device has a
//! memory map.

use crate::num::ToUsize;
use std::sync::Arc;

use crate::guest::records::APV_PAGE_SIZE;
use crate::host::{HostPtr, PhysRange};
use crate::memory::map::MemoryMap;
use crate::platform::Platform;
use crate::platform::os::vm::VmProt;
use crate::tasks::DeathKind;

/// Whether `value` is a multiple of `alignment` (a power of two).
fn aligned_to(value: u64, alignment: u64) -> bool {
    value & alignment.wrapping_sub(1) == 0
}

/// A VM reservation of the task's size, with no access, that the memory map maps guest memory
/// into: the task the device's and the EFI display's `createTask` make over a memory map.
/// Dropping it frees the reservation, then releases the map.
pub struct MappingTask<P: Platform> {
    size: u64,
    base_address: HostPtr,
    memory_map: Arc<MemoryMap<P>>,
    /// The platform whose VM makes the reservation's calls on this process's task.
    platform: Arc<P>,
}

impl<P: Platform> MappingTask<P> {
    /// Reserves `size` bytes anywhere and sets their maximum protection to none; `None` if
    /// either call fails (a reservation made is freed by the drop). A size that is not a whole
    /// number of pages ends the process first.
    pub fn new(platform: Arc<P>, size: u64, memory_map: Arc<MemoryMap<P>>) -> Option<Arc<Self>> {
        crate::fatal_unless!(aligned_to(size, APV_PAGE_SIZE), DeathKind::MappingSize);
        let mut task = MappingTask {
            size,
            base_address: HostPtr::NULL,
            memory_map,
            platform,
        };
        let vm = task.platform.vm();
        match vm.reserve_noaccess(size) {
            Ok(a) => task.base_address = a,
            // The base is still NULL, so the drop frees nothing
            Err(_) => return None,
        }
        Some(Arc::new(task))
    }

    /// The reservation's base.
    pub fn base_address(&self) -> HostPtr {
        self.base_address
    }

    /// The reservation's size.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Maps `ranges` at `base + offset` through the memory map (no range check of its own; the
    /// offset adds with wrapping).
    pub fn map_memory_at_virtual_offset(
        &self,
        offset: u64,
        ranges: &[PhysRange],
        read_only: bool,
    ) -> bool {
        let address = self.base_address.wrapping_add(offset);
        self.memory_map
            .map_memory_at_virtual_address(address, ranges, read_only)
    }

    /// Replaces the range with fresh zero-fill (fixed, overwrite), then sets its maximum
    /// protection to none.
    ///
    /// The allocation writes back the address it used, `base + offset` truncated to the host's
    /// page (measured: `base + 0x2000` comes back `base`, `base + 0x5000` `base + 0x4000`), and
    /// the protection starts there, with the length it was given. The VM's
    /// `allocate_fixed_overwrite` does not answer that address, so the truncation is made here
    /// with the host's page size.
    pub fn unmap_memory_at_virtual_offset(&self, offset: u64, length: u64) -> bool {
        let vm = self.platform.vm();
        let address = self.base_address.wrapping_add(offset);
        if vm.allocate_fixed_overwrite(address, length).is_err() {
            return false;
        }
        let page_mask = vm.page_size().wrapping_sub(1);
        let address = HostPtr::from_addr(address.addr() & !page_mask.to_usize());
        vm.protect_max(address, length, VmProt::NONE).is_ok()
    }
}

impl<P: Platform> Drop for MappingTask<P> {
    /// Frees the reservation (its result ignored); the map is released after, with the fields.
    fn drop(&mut self) {
        if !self.base_address.is_null() {
            let _ = self.platform.vm().deallocate(self.base_address, self.size);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::descriptor::{GuestPhysicalRange, MemoryMapDescriptor};
    use crate::platform::os::vm::KernReturn;
    use crate::raise::death::DeathSite;
    use crate::testkit::calllog::{Arg, Call};
    use crate::testkit::capture::catch_death;
    use crate::testkit::guest_ram::GuestRam;
    use crate::testkit::platform::FakePlatform;
    use crate::testkit::script::Step;

    type Task = MappingTask<FakePlatform>;
    /// The host's page.
    const PAGE: u64 = crate::testkit::guest_ram::PAGE_SIZE;

    fn plat() -> Arc<FakePlatform> {
        Arc::new(FakePlatform::new())
    }

    /// A memory map over one fake RAM range at guest physical `0x10_0000`.
    fn map(p: &Arc<FakePlatform>, len: u64) -> (GuestRam, Arc<MemoryMap<FakePlatform>>) {
        let ram = GuestRam::with_range(0x10_0000, len).unwrap();
        p.fake_vm().adopt(&ram);
        let mut d = MemoryMapDescriptor::new();
        for r in ram.ranges() {
            d.add_range(GuestPhysicalRange {
                physical_address: r.physical_address,
                physical_length: r.region.len(),
                host_address: r.host(),
            });
        }
        let m = MemoryMap::new(p.clone(), &d).unwrap();
        (ram, m)
    }

    fn vm_calls(p: &FakePlatform) -> Vec<Call> {
        p.call_log()
            .calls()
            .into_iter()
            .filter(|c| c.class == "mach_vm")
            .collect()
    }

    fn sels(calls: &[Call]) -> Vec<&'static str> {
        calls.iter().map(|c| c.sel).collect()
    }

    mod aligned_to {
        use super::*;

        #[test]
        fn page_multiples_only() {
            assert!(aligned_to(0, PAGE));
            assert!(aligned_to(PAGE, PAGE));
            assert!(aligned_to(7 * PAGE, PAGE));
            assert!(!aligned_to(1, PAGE));
            assert!(!aligned_to(PAGE + PAGE / 2, PAGE));
            assert!(aligned_to(u64::MAX - (PAGE - 1), PAGE));
        }
    }

    mod new {
        use super::*;

        const ASSERT: DeathSite = DeathKind::MappingSize.site();

        /// Allocate anywhere, then protect none
        #[test]
        fn reserves_with_no_access() {
            let p = plat();
            let (_ram, m) = map(&p, PAGE);
            let before = vm_calls(&p).len();
            let t = Task::new(p.clone(), 4 * PAGE, m).unwrap();
            let calls = vm_calls(&p).split_off(before);
            assert_eq!(sels(&calls), vec!["mach_vm_allocate", "mach_vm_protect"]);
            assert_eq!(
                calls[0].args,
                vec![Arg::U64(4 * PAGE), Arg::Str("ANYWHERE".into())]
            );
            assert_eq!(
                calls[1].args,
                vec![
                    Arg::Ptr(t.base_address().addr()),
                    Arg::U64(4 * PAGE),
                    Arg::Bool(true),
                    Arg::I32(0)
                ]
            );
            assert!(!t.base_address().is_null());
            assert_eq!(t.size(), 4 * PAGE);
            assert_eq!(p.fake_vm().protection(t.base_address()), Some(VmProt::NONE));
        }

        /// A size that is not a page multiple dies
        #[test]
        fn unaligned_size_dies_before_any_call() {
            let p = plat();
            let (_ram, m) = map(&p, PAGE);
            let before = vm_calls(&p).len();
            let r = catch_death(|| Task::new(p.clone(), PAGE + 1, m).map(|_| ()));
            assert_eq!(r.err(), Some(ASSERT));
            assert_eq!(vm_calls(&p).len(), before);
        }

        /// A failed allocate is nil; nothing to deallocate
        #[test]
        fn allocate_failure_is_nil() {
            let p = plat();
            let (_ram, m) = map(&p, PAGE);
            p.script()
                .push("mach_vm_allocate", Step::value(KernReturn(3)));
            let before = vm_calls(&p).len();
            assert!(Task::new(p.clone(), PAGE, m.clone()).is_none());
            assert_eq!(
                sels(&vm_calls(&p).split_off(before)),
                vec!["mach_vm_allocate"]
            );
            // The map was released with the task: only the test holds it now.
            assert_eq!(Arc::strong_count(&m), 1);
        }

        /// A failed protect is nil; the release deallocates the reservation
        #[test]
        fn protect_failure_is_nil_and_frees() {
            let p = plat();
            let (_ram, m) = map(&p, PAGE);
            p.script()
                .push("mach_vm_protect", Step::value(KernReturn(2)));
            let before = vm_calls(&p).len();
            assert!(Task::new(p.clone(), 2 * PAGE, m).is_none());
            let calls = vm_calls(&p).split_off(before);
            assert_eq!(
                sels(&calls),
                vec!["mach_vm_allocate", "mach_vm_protect", "mach_vm_deallocate"]
            );
            assert_eq!(calls[2].args[1], Arg::U64(2 * PAGE));
            assert_eq!(calls[1].args[0], calls[2].args[0]);
        }
    }

    mod drop {
        use super::*;

        /// Deallocates the reservation, then releases the map
        #[test]
        fn deallocates_then_releases_map() {
            let p = plat();
            let (_ram, m) = map(&p, PAGE);
            let t = Task::new(p.clone(), 2 * PAGE, m.clone()).unwrap();
            let base = t.base_address();
            assert_eq!(Arc::strong_count(&m), 2);
            let before = vm_calls(&p).len();
            drop(t);
            let calls = vm_calls(&p).split_off(before);
            assert_eq!(sels(&calls), vec!["mach_vm_deallocate"]);
            assert_eq!(
                calls[0].args,
                vec![Arg::Ptr(base.addr()), Arg::U64(2 * PAGE)]
            );
            assert_eq!(Arc::strong_count(&m), 1);
            assert!(!p.fake_vm().is_mapped(base, PAGE));
        }
    }

    mod map_memory_at_virtual_offset {
        use super::*;

        /// Maps at base + offset through the memory map, with its protection
        #[test]
        fn maps_through_the_memory_map() {
            let p = plat();
            let (ram, m) = map(&p, 4 * PAGE);
            let t = Task::new(p.clone(), 8 * PAGE, m).unwrap();
            assert!(ram.write_u32(0x10_0000 + PAGE, 0xfeed_f00d));
            let ranges = [PhysRange {
                physical_address: 0x10_0000 + PAGE,
                physical_length: 2 * PAGE,
            }];
            assert!(t.map_memory_at_virtual_offset(3 * PAGE, &ranges, true));
            let at = t.base_address().wrapping_add(3 * PAGE);
            assert_eq!(
                p.fake_vm().read(at, 4),
                Some(0xfeed_f00du32.to_le_bytes().to_vec())
            );
            assert!(!p.fake_vm().is_writable(at, 4));
            assert!(t.map_memory_at_virtual_offset(0, &ranges, false));
            assert!(p.fake_vm().is_writable(t.base_address(), 4));
        }

        /// The map's refusal is returned
        #[test]
        fn refusal_passes_through() {
            let p = plat();
            let (_ram, m) = map(&p, PAGE);
            let t = Task::new(p.clone(), 2 * PAGE, m).unwrap();
            let ranges = [PhysRange {
                physical_address: 0,
                physical_length: PAGE,
            }];
            assert!(!t.map_memory_at_virtual_offset(0, &ranges, false));
            assert!(t.map_memory_at_virtual_offset(0, &[], false));
        }
    }

    mod unmap_memory_at_virtual_offset {
        use super::*;

        /// Fresh zero fill, then no access
        #[test]
        fn overwrites_then_protects_none() {
            let p = plat();
            let (_ram, m) = map(&p, PAGE);
            let t = Task::new(p.clone(), 4 * PAGE, m).unwrap();
            let at = t.base_address().wrapping_add(PAGE);
            let before = vm_calls(&p).len();
            assert!(t.unmap_memory_at_virtual_offset(PAGE, 2 * PAGE));
            let calls = vm_calls(&p).split_off(before);
            assert_eq!(sels(&calls), vec!["mach_vm_allocate", "mach_vm_protect"]);
            assert_eq!(
                calls[0].args,
                vec![
                    Arg::Ptr(at.addr()),
                    Arg::U64(2 * PAGE),
                    Arg::Str("FIXED|OVERWRITE".into())
                ]
            );
            assert_eq!(
                calls[1].args,
                vec![
                    Arg::Ptr(at.addr()),
                    Arg::U64(2 * PAGE),
                    Arg::Bool(true),
                    Arg::I32(0)
                ]
            );
            assert_eq!(p.fake_vm().protection(at), Some(VmProt::NONE));
        }

        /// An unaligned offset: the protect starts at the page-truncated address the kernel writes
        /// back
        #[test]
        fn unaligned_offset_protects_truncated_address() {
            let p = plat();
            let (_ram, m) = map(&p, PAGE);
            let t = Task::new(p.clone(), 4 * PAGE, m).unwrap();
            let before = vm_calls(&p).len();
            assert!(t.unmap_memory_at_virtual_offset(PAGE + PAGE / 2, PAGE));
            let calls = vm_calls(&p).split_off(before);
            assert_eq!(sels(&calls), vec!["mach_vm_allocate", "mach_vm_protect"]);
            assert_eq!(
                calls[0].args[0],
                Arg::Ptr(t.base_address().wrapping_add(PAGE + PAGE / 2).addr())
            );
            assert_eq!(
                calls[1].args,
                vec![
                    Arg::Ptr(t.base_address().wrapping_add(PAGE).addr()),
                    Arg::U64(PAGE),
                    Arg::Bool(true),
                    Arg::I32(0)
                ]
            );
        }

        /// A failed allocate is NO with no protect
        #[test]
        fn allocate_failure_is_no() {
            let p = plat();
            let (_ram, m) = map(&p, PAGE);
            let t = Task::new(p.clone(), 2 * PAGE, m).unwrap();
            p.script()
                .push("mach_vm_allocate", Step::value(KernReturn(1)));
            let before = vm_calls(&p).len();
            assert!(!t.unmap_memory_at_virtual_offset(0, PAGE));
            assert_eq!(
                sels(&vm_calls(&p).split_off(before)),
                vec!["mach_vm_allocate"]
            );
        }

        /// A failed protect is NO
        #[test]
        fn protect_failure_is_no() {
            let p = plat();
            let (_ram, m) = map(&p, PAGE);
            let t = Task::new(p.clone(), 2 * PAGE, m).unwrap();
            p.script()
                .push("mach_vm_protect", Step::value(KernReturn(2)));
            assert!(!t.unmap_memory_at_virtual_offset(0, PAGE));
        }
    }

    mod accessors {
        use super::*;

        #[test]
        fn synthesized_getters() {
            let p = plat();
            let (_ram, m) = map(&p, PAGE);
            let t = Task::new(p.clone(), 3 * PAGE, m).unwrap();
            assert_eq!(t.size(), 3 * PAGE);
            assert!(p.fake_vm().protection(t.base_address()).is_some());
        }
    }
}
