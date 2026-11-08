//! Platform trait: `HostVm`: mach_vm allocate/remap/protect/deallocate, `vm_page_size`.
//!
//! Every call is on `mach_task_self()`. Only `KERN_SUCCESS` is ever tested, so the error side
//! carries the raw `kern_return_t` for the one site that logs it.

use std::ops::BitOr;

use crate::core::host::HostPtr;

/// A host virtual address (`mach_vm_address_t`). Host addresses are [`HostPtr`]s throughout core,
/// so that `guest` is the only place that turns one into a pointer.
pub type HostAddr = HostPtr;

/// `kern_return_t`, raw.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct KernReturn(pub i32);

impl KernReturn {
    /// `KERN_SUCCESS`.
    pub const SUCCESS: KernReturn = KernReturn(0);

    /// `kr == KERN_SUCCESS`, the only test core makes.
    pub const fn is_success(self) -> bool {
        self.0 == 0
    }
}

/// `vm_prot_t`, raw.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct VmProt(pub i32);

impl VmProt {
    /// `VM_PROT_NONE`.
    pub const NONE: VmProt = VmProt(0);
    /// `VM_PROT_READ`.
    pub const READ: VmProt = VmProt(1);
    /// `VM_PROT_WRITE`.
    pub const WRITE: VmProt = VmProt(2);

    /// The raw value.
    pub const fn raw(self) -> i32 {
        self.0
    }
}

impl BitOr for VmProt {
    type Output = VmProt;
    /// `VM_PROT_READ | VM_PROT_WRITE`.
    fn bitor(self, rhs: VmProt) -> VmProt {
        VmProt(self.0 | rhs.0)
    }
}

/// Mach VM on the current task (`mach_task_self()`). The fake is a model over real memory
/// (`testkit::vm`).
pub trait HostVm: Send + Sync {
    /// `vm_page_size` (16 KiB on the suite's host).
    fn page_size(&self) -> u64;
    /// `vm_page_shift` (14 on the suite's host): core reads it as its own global (and the other
    /// page-number shifts).
    fn page_shift(&self) -> u32;
    /// `mach_vm_allocate(task, &a, len, VM_FLAGS_ANYWHERE)`: a fresh zero-filled range.
    fn allocate_anywhere(&self, len: u64) -> Result<HostAddr, KernReturn>;
    /// The same call, [`HostVm::allocate_anywhere`], for a range core then names in address
    /// ranges, which the platform maps again as an alias of its pages (the detach heap's
    /// temporary). On Mach any range can be aliased; a platform whose fresh ranges cannot be
    /// (Linux's anonymous memory) gives one that can.
    fn allocate_aliasable(&self, len: u64) -> Result<HostAddr, KernReturn> {
        self.allocate_anywhere(len)
    }
    /// `mach_vm_allocate(task, &addr, len, VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE)`: replaces
    /// `addr..addr+len` with zero fill.
    fn allocate_fixed_overwrite(&self, addr: HostAddr, len: u64) -> Result<(), KernReturn>;
    /// `mach_vm_remap(task, &dst, len, 0, VM_FLAGS_OVERWRITE, task, src, FALSE, &cur, &max, VM_INHERIT_COPY)`:
    /// a **shared** alias of `src` at `dst` (a write through one shows through the other); fails
    /// for an unmapped source. Returns the address the kernel wrote back (core ignores it and the
    /// two protections).
    fn remap_shared_overwrite(
        &self,
        dst: HostAddr,
        src: HostAddr,
        len: u64,
    ) -> Result<HostAddr, KernReturn>;
    /// A view of `runs` laid one after another from `dst`: for each `(src, len)` in order,
    /// [`HostVm::remap_shared_overwrite`] of `len` bytes from `src` at `dst` plus the lengths of
    /// the runs before it, each a whole number of pages. It stops at the first refusal, which it
    /// answers with the run's index and code, the runs before it left mapped, as those remaps one
    /// at a time leave them. Mach's calls are those remaps; a platform whose views take a kernel
    /// mapping for each run of pages (Linux) maps the whole view at once in fewer, where a
    /// refusal leaves the runs before it mapped or none.
    fn remap_runs_shared_overwrite(
        &self,
        dst: HostAddr,
        runs: &[(HostAddr, u64)],
    ) -> Result<(), (usize, KernReturn)> {
        let mut at = dst;
        for (i, &(src, len)) in runs.iter().enumerate() {
            self.remap_shared_overwrite(at, src, len)
                .map_err(|kr| (i, kr))?;
            at = at.wrapping_add(len);
        }
        Ok(())
    }
    /// `mach_vm_protect(task, addr, len, TRUE, prot)`: sets the **maximum** protection
    /// (`set_maximum = TRUE`), not mprotect semantics. A later write to a read-only range is a
    /// SIGBUS, a process death.
    fn protect_max(&self, addr: HostAddr, len: u64, prot: VmProt) -> Result<(), KernReturn>;
    /// `mach_vm_deallocate(task, addr, len)`. Every caller ignores the result.
    fn deallocate(&self, addr: HostAddr, len: u64) -> KernReturn;
    /// No Mach counterpart: the most kernel mappings that views of guest pages no object holds
    /// may keep, kept only to be taken again (the graphics device's cache of IOSurface views),
    /// where each run of a view ([`HostVm::remap_runs_shared_overwrite`]) takes a mapping of its
    /// own and the process has a budget of them. `None`, the default, where views take none that
    /// is counted (Mach's).
    fn idle_view_mappings(&self) -> Option<u64> {
        None
    }
    /// No Mach counterpart: how many more kernel mappings views of guest pages may take, now,
    /// before the platform maps a view another way than a mapping a run (through dma-bufs, which
    /// a device that pins a view's pages cannot import) or refuses one, where each run of a view
    /// ([`HostVm::remap_runs_shared_overwrite`]) takes a mapping of its own and the process has a
    /// budget of them. `None`, the default, where views take none that is counted (Mach's).
    fn view_room(&self) -> Option<u64> {
        None
    }
    /// No Mach counterpart: how many of the kernel mappings [`HostVm::view_room`] counts the view
    /// of guest pages at `at`, `len` bytes, holds now, which its teardown gives back: a mapping a
    /// run where its runs are mapped one by one, fewer where the platform mapped them another way
    /// (Linux's dma-bufs, a mapping per 4 MiB). `None`, the default, where views take none that is
    /// counted (Mach's).
    fn view_mappings(&self, at: HostAddr, len: u64) -> Option<u64> {
        let _ = (at, len);
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only KERN_SUCCESS is success
    #[test]
    fn kern_return_is_success() {
        assert!(KernReturn::SUCCESS.is_success());
        assert!(KernReturn(0).is_success());
        assert!(!KernReturn(1).is_success()); // KERN_INVALID_ADDRESS
        assert!(!KernReturn(-1).is_success());
        assert_eq!(KernReturn::default(), KernReturn::SUCCESS);
    }

    /// The protections have mach's values and combine by or
    #[test]
    fn vm_prot_values_and_or() {
        assert_eq!(VmProt::NONE.raw(), 0);
        assert_eq!(VmProt::READ.raw(), 1);
        assert_eq!(VmProt::WRITE.raw(), 2);
        assert_eq!((VmProt::READ | VmProt::WRITE).raw(), 3);
        assert_eq!(VmProt::NONE | VmProt::READ, VmProt::READ);
    }
}
