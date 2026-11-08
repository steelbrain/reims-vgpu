//! Fake: `FakeVm`: Mach VM over real memory, with remap aliasing and maximum protections
//! (architecture 13.2), for `platform::os::vm::HostVm`.
//!
//! Every address `FakeVm` hands out is real memory (a [`RawRegion`]), because core dereferences
//! host addresses (D18). It knows its own allocations and the [`GuestRam`] a test adopts; any other
//! address is unmapped (`KERN_INVALID_ADDRESS`), as a NULL host source is for a remap.
//!
//! **Aliasing.** `mach_vm_remap` makes a shared alias: a write through one side shows through
//! the other (measured on the suite's host). Core has no way to share pages without the kernel,
//! so the fake copies at the remap and keeps the two sides coherent at sync points: every
//! `FakeVm` call, and [`FakeVm::sync`], copy each byte that changed on one side since the last
//! sync to the other. A test that writes guest RAM after a mapping exists, or reads guest RAM
//! after core wrote through a mapping, calls `sync()` first. Writes through [`FakeVm::write`]
//! reach every alias at once.
//!
//! **Protection.** `mach_vm_protect` sets the maximum protection per page. A write to a page
//! without `VM_PROT_WRITE` through [`FakeVm::write`] is a SIGBUS: a death marker at
//! [`VM_WRITE_FAULT`]. Deallocated pages become unmapped; their memory is kept until the `FakeVm`
//! is dropped, so a stale raw pointer in core reads zeros or old bytes instead of faulting the test
//! binary.

use std::sync::Arc;

use crate::core::sync::Mutex;

use super::calllog::{Arg, CallLog};
use super::guest_ram::{GuestRam, PAGE_SIZE, RawRegion, copy_between};
use super::script::Script;
use crate::core::host::HostPtr;
use crate::core::num::{self, ToU64, ToUsize};
use crate::core::platform::ObjId;
use crate::core::platform::os::vm::{HostAddr, HostVm, KernReturn, VmProt};
use crate::core::raise::death::DeathKind;
use crate::core::raise::death::DeathSite;
use crate::core::raise::death::fatal;

/// `KERN_INVALID_ADDRESS`.
pub const KERN_INVALID_ADDRESS: KernReturn = KernReturn(1);
/// `KERN_NO_SPACE`.
pub const KERN_NO_SPACE: KernReturn = KernReturn(3);
/// `KERN_INVALID_ARGUMENT`.
pub const KERN_INVALID_ARGUMENT: KernReturn = KernReturn(4);

/// The death marker of a write to memory without `VM_PROT_WRITE`: a SIGBUS, which ends the
/// process on the real platform.
pub const VM_WRITE_FAULT: DeathSite = DeathSite {
    kind: DeathKind::Abort,
    file: "SIGBUS",
    line: 0,
};

/// The page shift of [`PAGE_SIZE`].
pub const PAGE_SHIFT: u32 = PAGE_SIZE.trailing_zeros();

const CLASS: &str = "mach_vm";

#[derive(Debug)]
struct Area {
    region: Arc<RawRegion>,
    prot: Vec<VmProt>,
    mapped: Vec<bool>,
}

impl Area {
    fn new(region: Arc<RawRegion>) -> Area {
        let pages = pages_of(region.len()).to_usize();
        Area {
            region,
            prot: vec![VmProt::READ | VmProt::WRITE; pages],
            mapped: vec![true; pages],
        }
    }

    /// The page indices covering `off..off+len` (whole pages).
    fn pages(&self, off: u64, len: u64) -> std::ops::Range<usize> {
        let first = num::div0(off, PAGE_SIZE);
        let end = num::div0(
            num::align_up_wrapping(off.wrapping_add(len), PAGE_SIZE),
            PAGE_SIZE,
        );
        first.to_usize()..end.to_usize().min(self.prot.len())
    }

    fn allows(&self, off: u64, len: u64, bit: VmProt) -> bool {
        self.pages(off, len).all(|p| {
            self.mapped.get(p).copied().unwrap_or(false)
                && self.prot.get(p).is_some_and(|v| v.raw() & bit.raw() != 0)
        })
    }
}

#[derive(Debug)]
struct Alias {
    dst: Arc<RawRegion>,
    dst_off: u64,
    src: Arc<RawRegion>,
    src_off: u64,
    shadow: Vec<u8>,
}

impl Alias {
    fn len(&self) -> u64 {
        self.shadow.len().to_u64()
    }

    /// Copies each byte changed on one side since the last sync to the other (the destination
    /// wins when both changed).
    fn sync(&mut self) {
        let n = self.len();
        let (Some(d), Some(s)) = (
            self.dst.read_vec(self.dst_off, n),
            self.src.read_vec(self.src_off, n),
        ) else {
            return;
        };
        for (i, ((dv, sv), sh)) in d
            .iter()
            .zip(s.iter())
            .zip(self.shadow.iter_mut())
            .enumerate()
        {
            let at = i.to_u64();
            if dv != sh {
                let _ = self.src.write(self.src_off.wrapping_add(at), &[*dv]);
                *sh = *dv;
            } else if sv != sh {
                let _ = self.dst.write(self.dst_off.wrapping_add(at), &[*sv]);
                *sh = *sv;
            }
        }
    }

    fn overlaps_dst(&self, region: &Arc<RawRegion>, off: u64, len: u64) -> bool {
        Arc::ptr_eq(&self.dst, region)
            && off < self.dst_off.wrapping_add(self.len())
            && self.dst_off < off.wrapping_add(len)
    }
}

#[derive(Debug, Default)]
struct VmState {
    areas: Vec<Area>,
    aliases: Vec<Alias>,
    /// What [`HostVm::idle_view_mappings`] answers ([`FakeVm::set_idle_view_mappings`]).
    idle_view_mappings: Option<u64>,
    /// What [`HostVm::view_room`] answers ([`FakeVm::set_view_room`]).
    view_room: Option<u64>,
}

impl VmState {
    fn find(&self, addr: HostPtr, len: u64) -> Option<(usize, u64)> {
        self.areas
            .iter()
            .enumerate()
            .find_map(|(i, a)| a.region.offset_of(addr, len).map(|off| (i, off)))
    }

    fn sync(&mut self) {
        for a in &mut self.aliases {
            a.sync();
        }
    }
}

fn pages_of(len: u64) -> u64 {
    num::div0(num::align_up_wrapping(len, PAGE_SIZE), PAGE_SIZE)
}

/// Mach VM on fake memory. `Clone` shares the address space.
#[derive(Debug, Clone)]
pub struct FakeVm {
    state: Arc<Mutex<VmState>>,
    log: CallLog,
    script: Script,
}

impl FakeVm {
    /// An empty address space recording into `log`, scripted by `script`.
    pub fn new(log: CallLog, script: Script) -> FakeVm {
        FakeVm {
            state: Arc::new(Mutex::new(VmState::default())),
            log,
            script,
        }
    }

    /// Has [`HostVm::idle_view_mappings`] answer `mappings` from now on (`None` at first, as
    /// Mach's).
    pub fn set_idle_view_mappings(&self, mappings: Option<u64>) {
        self.state.lock().idle_view_mappings = mappings;
    }

    /// Has [`HostVm::view_room`] answer `mappings` from now on (`None` at first, as Mach's).
    pub fn set_view_room(&self, mappings: Option<u64>) {
        self.state.lock().view_room = mappings;
    }

    /// The same address space with its own, discarded log and an empty script: the host's (QEMU's)
    /// own mach calls, which are not core's platform calls.
    pub fn unlogged(&self) -> FakeVm {
        FakeVm {
            state: self.state.clone(),
            log: CallLog::new(),
            script: Script::new(),
        }
    }

    fn call(&self, sel: &'static str, args: Vec<Arg>) -> super::script::Step {
        self.log.push(ObjId(0), CLASS, sel, args);
        self.script.next(ObjId(0), sel)
    }

    /// Makes `ram`'s ranges known memory (readable and writable), as QEMU's RAM is mapped in
    /// the host.
    pub fn adopt(&self, ram: &GuestRam) {
        let mut g = self.state.lock();
        for r in ram.ranges() {
            if !g.areas.iter().any(|a| Arc::ptr_eq(&a.region, &r.region)) {
                g.areas.push(Area::new(r.region));
            }
        }
    }

    /// Brings every alias pair up to date (see the module comment).
    pub fn sync(&self) {
        self.state.lock().sync();
    }

    /// Whether every page of `addr..addr+len` is mapped.
    pub fn is_mapped(&self, addr: HostPtr, len: u64) -> bool {
        let g = self.state.lock();
        g.find(addr, len).is_some_and(|(i, off)| {
            g.areas.get(i).is_some_and(|a| {
                a.pages(off, len)
                    .all(|p| a.mapped.get(p).copied().unwrap_or(false))
            })
        })
    }

    /// Whether every page of `addr..addr+len` is mapped and writable.
    pub fn is_writable(&self, addr: HostPtr, len: u64) -> bool {
        let g = self.state.lock();
        g.find(addr, len).is_some_and(|(i, off)| {
            g.areas
                .get(i)
                .is_some_and(|a| a.allows(off, len, VmProt::WRITE))
        })
    }

    /// The maximum protection of the page holding `addr`.
    pub fn protection(&self, addr: HostPtr) -> Option<VmProt> {
        let g = self.state.lock();
        let (i, off) = g.find(addr, 0)?;
        let a = g.areas.get(i)?;
        a.prot.get(num::div0(off, PAGE_SIZE).to_usize()).copied()
    }

    /// Reads `len` bytes at `addr` after a sync; `None` if unmapped or not readable.
    pub fn read(&self, addr: HostPtr, len: u64) -> Option<Vec<u8>> {
        let mut g = self.state.lock();
        g.sync();
        let (i, off) = g.find(addr, len)?;
        let a = g.areas.get(i)?;
        if !a.allows(off, len, VmProt::READ) {
            return None;
        }
        a.region.read_vec(off, len)
    }

    /// Writes `src` at `addr` as a raw store would, reaching every alias. A write to a page without
    /// `VM_PROT_WRITE` is a SIGBUS: a death at [`VM_WRITE_FAULT`]. `false` if the address is not
    /// known memory.
    pub fn write(&self, addr: HostPtr, src: &[u8]) -> bool {
        let mut g = self.state.lock();
        g.sync();
        let len = src.len().to_u64();
        let Some((i, off)) = g.find(addr, len) else {
            return false;
        };
        let Some(a) = g.areas.get(i) else {
            return false;
        };
        if !a.allows(off, len, VmProt::WRITE) {
            drop(g);
            fatal(VM_WRITE_FAULT)
        }
        let ok = a.region.write(off, src);
        g.sync();
        ok
    }
}

impl HostVm for FakeVm {
    fn page_size(&self) -> u64 {
        PAGE_SIZE
    }

    fn page_shift(&self) -> u32 {
        PAGE_SHIFT
    }

    fn allocate_anywhere(&self, len: u64) -> Result<HostAddr, KernReturn> {
        let step = self.call(
            "mach_vm_allocate",
            vec![Arg::U64(len), Arg::Str("ANYWHERE".into())],
        );
        step.getter("mach_vm_allocate");
        if let Some(kr) = step.get::<KernReturn>() {
            return Err(kr);
        }
        let mut g = self.state.lock();
        g.sync();
        let size = num::align_up_wrapping(len, PAGE_SIZE);
        let region = RawRegion::new(size).ok_or(KERN_NO_SPACE)?;
        let addr = region.addr();
        g.areas.push(Area::new(region));
        Ok(addr)
    }

    fn allocate_fixed_overwrite(&self, addr: HostAddr, len: u64) -> Result<(), KernReturn> {
        let step = self.call(
            "mach_vm_allocate",
            vec![
                Arg::Ptr(addr.addr()),
                Arg::U64(len),
                Arg::Str("FIXED|OVERWRITE".into()),
            ],
        );
        step.getter("mach_vm_allocate");
        if let Some(kr) = step.get::<KernReturn>() {
            return Err(kr);
        }
        let mut g = self.state.lock();
        g.sync();
        let (i, off) = g.find(addr, len).ok_or(KERN_INVALID_ADDRESS)?;
        let region = match g.areas.get(i) {
            Some(a) => a.region.clone(),
            None => return Err(KERN_INVALID_ADDRESS),
        };
        g.aliases.retain(|al| !al.overlaps_dst(&region, off, len));
        if let Some(a) = g.areas.get_mut(i) {
            let _ = a.region.zero(off, len);
            for p in a.pages(off, len) {
                if let (Some(m), Some(pr)) = (a.mapped.get_mut(p), a.prot.get_mut(p)) {
                    *m = true;
                    *pr = VmProt::READ | VmProt::WRITE;
                }
            }
        }
        Ok(())
    }

    fn remap_shared_overwrite(
        &self,
        dst: HostAddr,
        src: HostAddr,
        len: u64,
    ) -> Result<HostAddr, KernReturn> {
        let step = self.call(
            "mach_vm_remap",
            vec![Arg::Ptr(dst.addr()), Arg::Ptr(src.addr()), Arg::U64(len)],
        );
        step.getter("mach_vm_remap");
        if let Some(kr) = step.get::<KernReturn>() {
            return Err(kr);
        }
        let mut g = self.state.lock();
        g.sync();
        let (si, soff) = g.find(src, len).ok_or(KERN_INVALID_ADDRESS)?;
        let (di, doff) = g.find(dst, len).ok_or(KERN_INVALID_ADDRESS)?;
        let (src_region, src_ok) = match g.areas.get(si) {
            Some(a) => (
                a.region.clone(),
                a.pages(soff, len)
                    .all(|p| a.mapped.get(p).copied().unwrap_or(false)),
            ),
            None => return Err(KERN_INVALID_ADDRESS),
        };
        if !src_ok {
            return Err(KERN_INVALID_ADDRESS);
        }
        let dst_region = match g.areas.get(di) {
            Some(a) => a.region.clone(),
            None => return Err(KERN_INVALID_ADDRESS),
        };
        g.aliases
            .retain(|al| !al.overlaps_dst(&dst_region, doff, len));
        if !copy_between(&src_region, soff, &dst_region, doff, len) {
            return Err(KERN_INVALID_ADDRESS);
        }
        let shadow = src_region.read_vec(soff, len).ok_or(KERN_INVALID_ADDRESS)?;
        if let Some(a) = g.areas.get_mut(di) {
            for p in a.pages(doff, len) {
                if let (Some(m), Some(pr)) = (a.mapped.get_mut(p), a.prot.get_mut(p)) {
                    *m = true;
                    *pr = VmProt::READ | VmProt::WRITE;
                }
            }
        }
        g.aliases.push(Alias {
            dst: dst_region,
            dst_off: doff,
            src: src_region,
            src_off: soff,
            shadow,
        });
        Ok(dst)
    }

    fn protect_max(&self, addr: HostAddr, len: u64, prot: VmProt) -> Result<(), KernReturn> {
        let step = self.call(
            "mach_vm_protect",
            vec![
                Arg::Ptr(addr.addr()),
                Arg::U64(len),
                Arg::Bool(true),
                Arg::I32(prot.raw()),
            ],
        );
        step.getter("mach_vm_protect");
        if let Some(kr) = step.get::<KernReturn>() {
            return Err(kr);
        }
        let mut g = self.state.lock();
        g.sync();
        let (i, off) = g.find(addr, len).ok_or(KERN_INVALID_ADDRESS)?;
        if let Some(a) = g.areas.get_mut(i) {
            for p in a.pages(off, len) {
                if let Some(pr) = a.prot.get_mut(p) {
                    *pr = prot;
                }
            }
        }
        Ok(())
    }

    fn deallocate(&self, addr: HostAddr, len: u64) -> KernReturn {
        let step = self.call(
            "mach_vm_deallocate",
            vec![Arg::Ptr(addr.addr()), Arg::U64(len)],
        );
        step.getter("mach_vm_deallocate");
        if let Some(kr) = step.get::<KernReturn>() {
            return kr;
        }
        let mut g = self.state.lock();
        g.sync();
        let Some((i, off)) = g.find(addr, len) else {
            return KernReturn::SUCCESS;
        };
        let region = match g.areas.get(i) {
            Some(a) => a.region.clone(),
            None => return KernReturn::SUCCESS,
        };
        g.aliases.retain(|al| !al.overlaps_dst(&region, off, len));
        if let Some(a) = g.areas.get_mut(i) {
            for p in a.pages(off, len) {
                if let Some(m) = a.mapped.get_mut(p) {
                    *m = false;
                }
            }
        }
        KernReturn::SUCCESS
    }

    /// As set ([`FakeVm::set_idle_view_mappings`]); unlogged, as no Mach call answers it.
    fn idle_view_mappings(&self) -> Option<u64> {
        self.state.lock().idle_view_mappings
    }

    /// As set ([`FakeVm::set_view_room`]); unlogged, as no Mach call answers it.
    fn view_room(&self) -> Option<u64> {
        self.state.lock().view_room
    }
}

#[cfg(test)]
mod tests {
    use super::super::script::Step;
    use super::*;
    use crate::core::testkit::capture::catch_death;

    fn vm() -> (FakeVm, CallLog, Script) {
        let (log, script) = (CallLog::new(), Script::new());
        (FakeVm::new(log.clone(), script.clone()), log, script)
    }

    /// Allocate gives zeroed, page-rounded, writable memory and logs the call
    #[test]
    fn allocate_anywhere() {
        let (vm, log, _) = vm();
        let a = vm.allocate_anywhere(100).unwrap();
        assert_eq!(a.addr() as u64 % PAGE_SIZE, 0);
        assert!(vm.is_mapped(a, PAGE_SIZE));
        assert!(vm.is_writable(a, PAGE_SIZE));
        assert_eq!(vm.read(a, 4).unwrap(), vec![0; 4]);
        assert!(vm.write(a, &[1, 2]));
        assert_eq!(vm.read(a, 2).unwrap(), vec![1, 2]);
        assert_eq!(log.calls_to("mach_vm_allocate")[0].args[0], Arg::U64(100));
        // The host's page: 16 KiB on arm64, 4 KiB on x86-64.
        let page = if cfg!(target_arch = "x86_64") {
            (0x1000, 12)
        } else {
            (0x4000, 14)
        };
        assert_eq!((vm.page_size(), vm.page_shift()), page);
    }

    /// A scripted kern_return fails the call
    #[test]
    fn scripted_failure() {
        let (vm, _, script) = vm();
        script.push("mach_vm_allocate", Step::value(KERN_NO_SPACE));
        assert_eq!(vm.allocate_anywhere(0x4000), Err(KERN_NO_SPACE));
        assert!(vm.allocate_anywhere(0x4000).is_ok());
        script.push("mach_vm_deallocate", Step::value(KernReturn(9)));
        assert_eq!(vm.deallocate(HostPtr::NULL, 1), KernReturn(9));
    }

    /// Remap is a shared alias at sync points, both ways; an unknown source fails
    #[test]
    fn remap_aliases() {
        let (vm, _, _) = vm();
        let ram = GuestRam::with_range(0x1000_0000, 0x8000).unwrap();
        vm.adopt(&ram);
        ram.write_u32(0x1000_4000, 0xfeed_face);
        let src = ram.host_addr(0x1000_4000).unwrap();
        let dst = vm.allocate_anywhere(0x8000).unwrap();
        let dst2 = dst.wrapping_add(0x4000);
        assert_eq!(vm.remap_shared_overwrite(dst2, src, 0x4000), Ok(dst2));
        assert_eq!(vm.read(dst2, 4).unwrap(), 0xfeed_faceu32.to_le_bytes());
        // guest writes after the mapping show through at a sync
        ram.write_u32(0x1000_4004, 7);
        vm.sync();
        assert_eq!(
            vm.read(dst2.wrapping_add(4), 4).unwrap(),
            7u32.to_le_bytes()
        );
        // writes through the mapping reach guest RAM
        assert!(vm.write(dst2.wrapping_add(8), &[9]));
        assert_eq!(ram.read_vec(0x1000_4008, 1).unwrap(), vec![9]);
        // an unmapped (NULL) source fails
        assert_eq!(
            vm.remap_shared_overwrite(dst, HostPtr::NULL, 0x4000),
            Err(KERN_INVALID_ADDRESS)
        );
        // overwrite breaks the alias
        vm.allocate_fixed_overwrite(dst2, 0x4000).unwrap();
        ram.write_u32(0x1000_4000, 1);
        vm.sync();
        assert_eq!(vm.read(dst2, 4).unwrap(), vec![0; 4]);
    }

    /// A view of runs is their remaps one after another, in order, each from the end of the
    /// one before; a refused run answers its index, the runs before it mapped
    #[test]
    fn a_view_of_runs_is_their_remaps_in_order() {
        let (vm, log, _) = vm();
        let ram = GuestRam::with_range(0x1000_0000, 0x10000).unwrap();
        vm.adopt(&ram);
        ram.write_u32(0x1000_8000, 0xa);
        ram.write_u32(0x1000_0000, 0xb);
        let host = |pa| ram.host_addr(pa).unwrap();
        let dst = vm.allocate_anywhere(0xc000).unwrap();
        let mark = log.mark();
        let runs = [(host(0x1000_8000), 0x4000), (host(0x1000_0000), 0x8000)];
        assert_eq!(vm.remap_runs_shared_overwrite(dst, &runs), Ok(()));
        assert_eq!(log.sels_since(mark), ["mach_vm_remap", "mach_vm_remap"]);
        assert_eq!(vm.read(dst, 4).unwrap(), 0xau32.to_le_bytes());
        assert_eq!(
            vm.read(dst.wrapping_add(0x4000), 4).unwrap(),
            0xbu32.to_le_bytes()
        );
        let refused = [(host(0x1000_8000), 0x4000), (HostPtr::NULL, 0x4000)];
        assert_eq!(
            vm.remap_runs_shared_overwrite(dst, &refused),
            Err((1, KERN_INVALID_ADDRESS))
        );
        assert_eq!(vm.read(dst, 4).unwrap(), 0xau32.to_le_bytes());
    }

    /// Maximum protection; a write to a read-only page is the SIGBUS marker
    #[test]
    fn protect_and_fault() {
        let (vm, log, _) = vm();
        let a = vm.allocate_anywhere(0x8000).unwrap();
        vm.protect_max(a, 0x4000, VmProt::READ).unwrap();
        assert_eq!(vm.protection(a), Some(VmProt::READ));
        assert_eq!(
            vm.protection(a.wrapping_add(0x4000)),
            Some(VmProt::READ | VmProt::WRITE)
        );
        assert!(!vm.is_writable(a, 1));
        assert!(vm.is_writable(a.wrapping_add(0x4000), 1));
        let v2 = vm.clone();
        assert_eq!(catch_death(move || v2.write(a, &[1])), Err(VM_WRITE_FAULT));
        vm.protect_max(a.wrapping_add(0x4000), 0x4000, VmProt::NONE)
            .unwrap();
        assert!(vm.read(a.wrapping_add(0x4000), 1).is_none());
        assert_eq!(
            vm.protect_max(HostPtr::from_addr(0x10), 1, VmProt::READ),
            Err(KERN_INVALID_ADDRESS)
        );
        let c = &log.calls_to("mach_vm_protect")[0];
        assert_eq!(c.args[2], Arg::Bool(true));
        assert_eq!(c.args[3], Arg::I32(1));
    }

    /// Deallocate unmaps but keeps the memory; unknown ranges succeed
    #[test]
    fn deallocate_unmaps() {
        let (vm, _, _) = vm();
        let a = vm.allocate_anywhere(0x4000).unwrap();
        assert_eq!(vm.deallocate(a, 0x4000), KernReturn::SUCCESS);
        assert!(!vm.is_mapped(a, 1));
        assert!(vm.read(a, 1).is_none());
        assert_eq!(vm.deallocate(HostPtr::from_addr(8), 8), KernReturn::SUCCESS);
        assert!(!vm.write(HostPtr::from_addr(8), &[0]));
        // fixed overwrite maps it again, zeroed
        vm.allocate_fixed_overwrite(a, 0x4000).unwrap();
        assert!(vm.is_mapped(a, 0x4000));
        assert_eq!(
            vm.allocate_fixed_overwrite(HostPtr::from_addr(8), 8),
            Err(KERN_INVALID_ADDRESS)
        );
    }
}
