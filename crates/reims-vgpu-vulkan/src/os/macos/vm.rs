//! `HostVm` over `mach_vm_*`.
//!
//! Every call is on `mach_task_self()`; the `kern_return_t` goes back raw. Only `== KERN_SUCCESS`
//! is ever tested.

use mach2::kern_return::{KERN_SUCCESS, kern_return_t};
use mach2::traps::mach_task_self;
use mach2::vm::{mach_vm_allocate, mach_vm_deallocate, mach_vm_protect, mach_vm_remap};
use mach2::vm_inherit::VM_INHERIT_COPY;
use mach2::vm_prot::vm_prot_t;
use mach2::vm_statistics::{VM_FLAGS_ANYWHERE, VM_FLAGS_FIXED, VM_FLAGS_OVERWRITE};
use mach2::vm_types::mach_vm_address_t;
use reims_vgpu_core::num::{ToU64, ToUsize};
use reims_vgpu_core::platform::os::vm::{HostAddr, KernReturn, VmProt};

/// Mach VM (`mach_vm_*`, `vm_page_size`).
#[derive(Debug, Default)]
pub struct MacVm;

impl MacVm {
    /// The process's HostVm.
    pub fn new() -> MacVm {
        MacVm
    }

    /// Whether every page of the `len` bytes at `addr` is mapped with read access, by
    /// `mach_vm_region`'s current protection of each region over them: `false` where a page is
    /// unmapped or mapped without read access, and for a region the kernel answers none of. A
    /// `len` of 0 is `true`.
    pub fn readable(&self, addr: HostAddr, len: u64) -> bool {
        use mach2::message::mach_msg_type_number_t;
        use mach2::vm_prot::VM_PROT_READ;
        use mach2::vm_region::{VM_REGION_BASIC_INFO_64, vm_region_basic_info_64};
        let end = vm_addr(addr).saturating_add(len);
        let mut at = vm_addr(addr);
        while at < end {
            let mut region = at;
            let mut size = 0;
            let mut info = vm_region_basic_info_64::default();
            let mut count: mach_msg_type_number_t = vm_region_basic_info_64::count();
            let mut object = 0;
            // SAFETY: every out-parameter is live and sized for the flavour.
            let kr = unsafe {
                mach2::vm::mach_vm_region(
                    mach_task_self(),
                    &mut region,
                    &mut size,
                    VM_REGION_BASIC_INFO_64,
                    std::ptr::from_mut(&mut info).cast::<std::ffi::c_int>(),
                    &mut count,
                    &mut object,
                )
            };
            // The kernel answers the region at `at`, or the first one past it: past it is a
            // hole.
            if kr != KERN_SUCCESS || region > at || info.protection & VM_PROT_READ == 0 {
                return false;
            }
            at = region.saturating_add(size);
        }
        true
    }
}

/// A host address as `mach_vm_address_t`.
fn vm_addr(a: HostAddr) -> mach_vm_address_t {
    a.addr().to_u64()
}

/// A `mach_vm_address_t` the kernel wrote back, as a host address.
fn host_addr(a: mach_vm_address_t) -> HostAddr {
    HostAddr::from_addr(a.to_usize())
}

/// `kr == KERN_SUCCESS` as a `Result` over the raw value.
fn check(kr: kern_return_t) -> Result<(), KernReturn> {
    if kr == KERN_SUCCESS {
        Ok(())
    } else {
        Err(KernReturn(kr))
    }
}

impl reims_vgpu_core::platform::os::vm::HostVm for MacVm {
    /// `vm_page_size`: 19 uses.
    fn page_size(&self) -> u64 {
        // SAFETY: `vm_page_size` is a libsystem_kernel global, set before `main` and never
        // written after.
        (unsafe { mach2::vm_page_size::vm_page_size }).to_u64()
    }

    /// `vm_page_shift`: and the other page-number shifts
    fn page_shift(&self) -> u32 {
        // SAFETY: as for `vm_page_size`.
        (unsafe { mach2::vm_page_size::vm_page_shift }).cast_unsigned()
    }

    /// `mach_vm_allocate(mach_task_self(), &address, len, VM_FLAGS_ANYWHERE)` with `address = 0`.
    fn allocate_anywhere(&self, len: u64) -> Result<HostAddr, KernReturn> {
        let mut address: mach_vm_address_t = 0;
        // SAFETY: `address` is a live out-parameter; an anywhere allocation touches no existing
        // mapping.
        let kr =
            unsafe { mach_vm_allocate(mach_task_self(), &mut address, len, VM_FLAGS_ANYWHERE) };
        check(kr).map(|()| host_addr(address))
    }

    /// `mach_vm_allocate(mach_task_self(), &address, length, VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE)`:
    /// the address the kernel writes back is the caller's own, which it then protects; a fixed
    /// allocation leaves it unchanged.
    fn allocate_fixed_overwrite(&self, addr: HostAddr, len: u64) -> Result<(), KernReturn> {
        let mut address = vm_addr(addr);
        // SAFETY: replacing `addr..addr+len` is the caller's decision: the range is the mapping
        // task's own reservation.
        let kr = unsafe {
            mach_vm_allocate(
                mach_task_self(),
                &mut address,
                len,
                VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE,
            )
        };
        check(kr)
    }

    /// `mach_vm_remap(task, &target, length, 0, VM_FLAGS_OVERWRITE, task, source, false, &cur,
    /// &max, VM_INHERIT_COPY)`, a shared alias (`copy = false`). Core ignores the written-back
    /// address and both protections; the address is returned for the trait's callers.
    fn remap_shared_overwrite(
        &self,
        dst: HostAddr,
        src: HostAddr,
        len: u64,
    ) -> Result<HostAddr, KernReturn> {
        let mut target = vm_addr(dst);
        let mut cur: vm_prot_t = 0;
        let mut max: vm_prot_t = 0;
        // SAFETY: overwriting `dst..dst+len` is the caller's decision (the range lies inside the
        // mapping it allocated for it); every out-parameter is live.
        let kr = unsafe {
            mach_vm_remap(
                mach_task_self(),
                &mut target,
                len,
                0,
                VM_FLAGS_OVERWRITE,
                mach_task_self(),
                vm_addr(src),
                0,
                &mut cur,
                &mut max,
                VM_INHERIT_COPY,
            )
        };
        check(kr).map(|()| host_addr(target))
    }

    /// `set_maximum` is TRUE at every site.
    fn protect_max(&self, addr: HostAddr, len: u64, prot: VmProt) -> Result<(), KernReturn> {
        // SAFETY: lowering a range's maximum protection is the caller's decision; a later access
        // the protection forbids faults (a process death).
        let kr = unsafe { mach_vm_protect(mach_task_self(), vm_addr(addr), len, 1, prot.raw()) };
        check(kr)
    }

    /// `mach_vm_deallocate(mach_task_self(), address, length)`: every caller ignores the result.
    fn deallocate(&self, addr: HostAddr, len: u64) -> KernReturn {
        // SAFETY: the range is one core allocated and no longer uses (its owner's drop).
        KernReturn(unsafe { mach_vm_deallocate(mach_task_self(), vm_addr(addr), len) })
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use reims_vgpu_core::platform::os::vm::HostVm;

    /// `KERN_INVALID_ADDRESS`.
    const KERN_INVALID_ADDRESS: i32 = 1;
    /// `KERN_INVALID_ARGUMENT`.
    const KERN_INVALID_ARGUMENT: i32 = 4;
    /// `KERN_PROTECTION_FAILURE`.
    const KERN_PROTECTION_FAILURE: i32 = 2;

    fn vm() -> MacVm {
        MacVm::new()
    }

    fn page() -> u64 {
        vm().page_size()
    }

    fn read(a: HostAddr, off: u64) -> u8 {
        // SAFETY: test memory this test mapped readable.
        unsafe { a.wrapping_add(off).as_ptr::<u8>().read_volatile() }
    }

    fn write(a: HostAddr, off: u64, v: u8) {
        // SAFETY: test memory this test mapped writable.
        unsafe { a.wrapping_add(off).as_ptr::<u8>().write_volatile(v) }
    }

    /// Region info: (current, maximum) protection of the region containing `a`.
    fn region_prot(a: HostAddr) -> (i32, i32) {
        use mach2::message::mach_msg_type_number_t;
        use mach2::vm_region::{VM_REGION_BASIC_INFO_64, vm_region_basic_info_64};
        let mut addr = vm_addr(a);
        let mut size = 0u64;
        let mut info = vm_region_basic_info_64::default();
        let mut count: mach_msg_type_number_t = vm_region_basic_info_64::count();
        let mut object = 0;
        // SAFETY: every out-parameter is live and sized for the flavour.
        let kr = unsafe {
            mach2::vm::mach_vm_region(
                mach_task_self(),
                &mut addr,
                &mut size,
                VM_REGION_BASIC_INFO_64,
                std::ptr::from_mut(&mut info).cast::<std::ffi::c_int>(),
                &mut count,
                &mut object,
            )
        };
        assert_eq!(kr, KERN_SUCCESS);
        (info.protection, info.max_protection)
    }

    /// `readable`: a fresh mapping is, over all of it and over part of it; a page of no access,
    /// a range over it, a deallocated page and a range into one are not
    #[test]
    fn readable_answers_by_each_regions_protection() {
        let v = vm();
        let p = page();
        let a = v.allocate_anywhere(4 * p).unwrap();
        assert!(v.readable(a, 4 * p));
        assert!(v.readable(a.wrapping_add(p + 1), p));
        assert!(v.readable(a, 0));
        // SAFETY: lowering the current protection of this test's own mapping.
        let kr = unsafe {
            mach2::vm::mach_vm_protect(mach_task_self(), vm_addr(a.wrapping_add(p)), p, 0, 0)
        };
        assert_eq!(kr, KERN_SUCCESS);
        assert!(v.readable(a, p));
        assert!(!v.readable(a, 2 * p));
        assert!(!v.readable(a.wrapping_add(p), 1));
        assert!(v.deallocate(a.wrapping_add(3 * p), p).is_success());
        assert!(v.readable(a.wrapping_add(2 * p), p));
        assert!(!v.readable(a.wrapping_add(2 * p), 2 * p));
        assert!(!v.readable(a.wrapping_add(3 * p), 1));
        assert!(v.deallocate(a, 3 * p).is_success());
        assert!(!v.readable(a, 1));
    }

    mod page_size {
        use super::*;

        /// vm_page_size / vm_page_shift (16 KiB / 14 on the suite's host)
        #[test]
        fn page_size_and_shift_agree() {
            let v = vm();
            assert_eq!(v.page_size(), 1u64 << v.page_shift());
            assert_eq!(
                v.page_size(),
                u64::try_from(unsafe { libc::vm_page_size }).unwrap()
            );
            #[cfg(target_arch = "aarch64")]
            {
                assert_eq!(v.page_size(), 16384);
                assert_eq!(v.page_shift(), 14);
            }
        }
    }

    mod allocate {
        use super::*;

        /// mach_vm_allocate(VM_FLAGS_ANYWHERE) gives page-aligned zero-filled memory
        #[test]
        fn anywhere_is_zero_filled_and_aligned() {
            let v = vm();
            let len = 2 * page();
            let a = v.allocate_anywhere(len).unwrap();
            assert!(!a.is_null());
            assert_eq!(vm_addr(a) % page(), 0);
            assert_eq!(read(a, 0), 0);
            assert_eq!(read(a, len - 1), 0);
            write(a, 7, 0xa5);
            assert_eq!(read(a, 7), 0xa5);
            assert_eq!(v.deallocate(a, len), KernReturn::SUCCESS);
        }

        /// A failed mach_vm_allocate hands the raw kern_return_t back
        #[test]
        fn anywhere_failure_is_the_raw_kr() {
            let e = vm().allocate_anywhere(u64::MAX - page()).unwrap_err();
            assert!(!e.is_success());
            assert_ne!(e.0, 0);
        }

        /// mach_vm_allocate(0): measured, KERN_SUCCESS with address 0
        #[test]
        fn anywhere_zero_length_measured() {
            // What a range mapping gets for totalLength 0.
            assert_eq!(vm().allocate_anywhere(0), Ok(HostAddr::NULL));
        }

        /// VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE replaces the range with zero fill
        #[test]
        fn fixed_overwrite_zero_fills_in_place() {
            let v = vm();
            let len = 3 * page();
            let a = v.allocate_anywhere(len).unwrap();
            write(a, 0, 1);
            write(a, page(), 2);
            write(a, 2 * page(), 3);
            let mid = a.wrapping_add(page());
            v.allocate_fixed_overwrite(mid, page()).unwrap();
            assert_eq!(read(a, 0), 1);
            assert_eq!(read(a, page()), 0);
            assert_eq!(read(a, 2 * page()), 3);
            v.deallocate(a, len);
        }

        /// A fixed overwrite the address space cannot hold is refused with the raw kr
        #[test]
        fn fixed_overwrite_failure_is_the_raw_kr() {
            let v = vm();
            let a = v.allocate_anywhere(page()).unwrap();
            let e = v
                .allocate_fixed_overwrite(a, u64::MAX - page())
                .unwrap_err();
            assert!(!e.is_success());
            // The page is untouched by the refused call.
            write(a, 0, 9);
            assert_eq!(read(a, 0), 9);
            v.deallocate(a, page());
        }
    }

    mod mapping_pieces {
        use super::*;

        /// The remap that maps a guest mapping's pieces (core's `memory::map`) is a shared alias
        /// both ways
        #[test]
        fn remap_is_shared_both_ways() {
            let v = vm();
            let len = 2 * page();
            let src = v.allocate_anywhere(len).unwrap();
            let dst = v.allocate_anywhere(len).unwrap();
            write(src, 5, 0x11);
            let got = v.remap_shared_overwrite(dst, src, len).unwrap();
            assert_eq!(got, dst);
            assert_eq!(read(dst, 5), 0x11);
            write(src, len - 1, 0x22);
            assert_eq!(read(dst, len - 1), 0x22);
            write(dst, 9, 0x33);
            assert_eq!(read(src, 9), 0x33);
            v.deallocate(dst, len);
            v.deallocate(src, len);
        }

        /// Remapping an unmapped source (a NULL host range) fails
        #[test]
        fn remap_of_unmapped_source_fails() {
            let v = vm();
            let dst = v.allocate_anywhere(page()).unwrap();
            let e = v
                .remap_shared_overwrite(dst, HostAddr::NULL, page())
                .unwrap_err();
            assert_eq!(e, KernReturn(KERN_INVALID_ADDRESS));
            v.deallocate(dst, page());
        }
    }

    mod protect {
        use super::*;

        /// set_maximum = TRUE lowers the maximum protection (and the current with it)
        #[test]
        fn protect_sets_the_maximum() {
            let v = vm();
            let a = v.allocate_anywhere(page()).unwrap();
            assert_eq!(region_prot(a), (3, 7));
            v.protect_max(a, page(), VmProt::READ).unwrap();
            assert_eq!(region_prot(a), (1, 1));
            // The maximum is now read: read-write can no longer be granted.
            assert_eq!(
                v.protect_max(a, page(), VmProt::READ | VmProt::WRITE),
                Err(KernReturn(KERN_PROTECTION_FAILURE))
            );
            v.deallocate(a, page());
        }

        /// A `MappingTask`'s reservation is VM_PROT_NONE
        #[test]
        fn protect_none() {
            let v = vm();
            let a = v.allocate_anywhere(page()).unwrap();
            v.protect_max(a, page(), VmProt::NONE).unwrap();
            assert_eq!(region_prot(a), (0, 0));
            // `MappingTask`'s unmap: fixed overwrite then NONE again.
            v.allocate_fixed_overwrite(a, page()).unwrap();
            v.protect_max(a, page(), VmProt::NONE).unwrap();
            assert_eq!(region_prot(a), (0, 0));
            v.deallocate(a, page());
        }

        /// Protecting an unmapped range fails with the raw kr (the log prints it)
        #[test]
        fn protect_unmapped_fails() {
            let e = vm()
                .protect_max(HostAddr::NULL, page(), VmProt::READ)
                .unwrap_err();
            assert_eq!(e, KernReturn(KERN_INVALID_ADDRESS));
        }
    }

    mod deallocate {
        use super::*;

        /// mach_vm_deallocate's result is handed back (callers ignore it)
        #[test]
        fn deallocate_returns_the_kr() {
            let v = vm();
            let a = v.allocate_anywhere(page()).unwrap();
            assert_eq!(v.deallocate(a, page()), KernReturn::SUCCESS);
            // Deallocating nothing succeeds; a wrapping range is an invalid argument.
            assert_eq!(v.deallocate(a, 0), KernReturn::SUCCESS);
            assert_eq!(
                v.deallocate(HostAddr::from_addr(usize::MAX & !0xfff), 2 * page()),
                KernReturn(KERN_INVALID_ARGUMENT)
            );
        }
    }
}
