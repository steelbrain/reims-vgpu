//! Windows page aliases for the ported Apple paravirtual GPU.
//!
//! A guest IOSurface can list 4 KiB pages in any order. Ordinary MapViewOfFile requires
//! 64 KiB offsets and cannot represent that list. MapViewOfFile3 replacing a placeholder
//! accepts page-aligned offsets. Each independently replaceable page keeps its own view,
//! so changing one page never temporarily unmaps neighbouring pages used by a GPU import.
//!
//! This module uses only the Windows kernel and std; its tests can run before the Vulkan
//! dependency build completes. It does not copy guest bytes or poll dirty-page bitmaps.

use std::ffi::c_void;
use std::io;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::ptr;
use std::sync::Arc;

/// Page size of the supported Windows x86-64 host, checked by the kernel mapping tests.
pub const PAGE: usize = 4096;
const MEM_RESERVE: u32 = 0x2000;
const MEM_RELEASE: u32 = 0x8000;
const MEM_RESERVE_PLACEHOLDER: u32 = 0x40000;
const MEM_REPLACE_PLACEHOLDER: u32 = 0x4000;
const MEM_PRESERVE_PLACEHOLDER: u32 = 2;
const PAGE_NOACCESS: u32 = 1;
const PAGE_READONLY: u32 = 2;
const PAGE_READWRITE: u32 = 4;

/// Win64 MEMORY_BASIC_INFORMATION; queried without dereferencing potentially inaccessible RAM.
#[repr(C)]
struct MemoryInfo {
    base: usize,
    allocation_base: usize,
    allocation_protection: u32,
    partition: u16,
    length: usize,
    state: u32,
    protection: u32,
    kind: u32,
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn VirtualQuery(address: *const c_void, info: *mut MemoryInfo, size: usize) -> usize;
    fn CreateFileMappingW(
        file: *mut c_void,
        attributes: *const c_void,
        protect: u32,
        high: u32,
        low: u32,
        name: *const u16,
    ) -> *mut c_void;
    fn VirtualFree(address: *mut c_void, size: usize, kind: u32) -> i32;
    fn VirtualProtect(address: *mut c_void, size: usize, protect: u32, old: *mut u32) -> i32;
}

/// Whether all bytes in this range currently permit a CPU read; guard pages are not read.
pub fn readable(start: u64, end: u64) -> bool {
    permitted(start, end, false)
}

/// Whether the current mappings can be imported by a GPU requiring writable CPU pages.
pub fn writable(start: u64, end: u64) -> bool {
    permitted(start, end, true)
}

fn permitted(start: u64, end: u64, write: bool) -> bool {
    let mut at = start;
    while at < end {
        let mut info = std::mem::MaybeUninit::<MemoryInfo>::uninit();
        // Count actual kernel queries in native regression tests. The counter is
        // thread-local so unrelated mappings tested concurrently cannot affect it.
        #[cfg(test)]
        QUERY_COUNT.with(|count| count.set(count.get() + 1));
        // SAFETY: the kernel inspects the address and writes only the local SDK-layout output.
        let size =
            unsafe { VirtualQuery(at as *const _, info.as_mut_ptr(), size_of::<MemoryInfo>()) };
        if size != size_of::<MemoryInfo>() {
            return false;
        }
        // SAFETY: VirtualQuery returned a complete initialized structure.
        let info = unsafe { info.assume_init() };
        let access = info.protection & 0xff;
        let allowed = if write {
            matches!(access, 0x04 | 0x08 | 0x40 | 0x80)
        } else {
            matches!(access, 0x02 | 0x04 | 0x08 | 0x20 | 0x40 | 0x80)
        };
        if info.state != 0x1000 || info.protection & 0x100 != 0 || !allowed {
            return false;
        }
        at = (info.base + info.length) as u64;
    }
    true
}

#[cfg(test)]
thread_local! { pub(super) static QUERY_COUNT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) }; }

#[link(name = "onecore")]
unsafe extern "system" {
    fn VirtualAlloc2(
        process: *mut c_void,
        base: *mut c_void,
        size: usize,
        kind: u32,
        protect: u32,
        parameters: *const c_void,
        count: u32,
    ) -> *mut c_void;
    fn MapViewOfFile3(
        section: *mut c_void,
        process: *mut c_void,
        base: *mut c_void,
        offset: u64,
        size: usize,
        kind: u32,
        protect: u32,
        parameters: *const c_void,
        count: u32,
    ) -> *mut c_void;
    fn UnmapViewOfFile2(process: *mut c_void, base: *mut c_void, flags: u32) -> i32;
}

/// An owned pagefile-backed section. Aliases hold it until their last view disappears.
#[derive(Debug)]
pub struct Section {
    handle: OwnedHandle,
    length: usize,
}

impl Section {
    /// Create a section over the same file that QEMU maps as guest RAM. Its existing bytes
    /// are neither resized nor copied; the section keeps the backing alive independently.
    pub fn of_file(file: &std::fs::File) -> io::Result<Arc<Self>> {
        let length = usize::try_from(file.metadata()?.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "guest RAM file too large"))?;
        // SAFETY: file stays open for the call; zero sizes preserve its current extent.
        let raw = unsafe {
            CreateFileMappingW(
                file.as_raw_handle(),
                ptr::null(),
                PAGE_READWRITE,
                0,
                0,
                ptr::null(),
            )
        };
        if raw.is_null() {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: the returned section handle is newly owned here.
        Ok(Arc::new(Self {
            handle: unsafe { OwnedHandle::from_raw_handle(raw) },
            length,
        }))
    }

    /// Allocates zero-filled backing shared by every view of this section.
    pub fn new(length: usize) -> io::Result<Arc<Self>> {
        if length == 0 || !length.is_multiple_of(PAGE) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "section length must be whole pages",
            ));
        }
        let size = length as u64;
        // SAFETY: INVALID_HANDLE_VALUE requests pagefile backing; no name or attributes.
        let raw = unsafe {
            CreateFileMappingW(
                -1isize as *mut c_void,
                ptr::null(),
                PAGE_READWRITE,
                (size >> 32) as u32,
                size as u32,
                ptr::null(),
            )
        };
        if raw.is_null() {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: the successful call returned a newly owned handle, closed once by OwnedHandle.
        Ok(Arc::new(Self {
            handle: unsafe { OwnedHandle::from_raw_handle(raw) },
            length,
        }))
    }
}

/// One reserved placeholder, or a view replacing it. The address remains reserved until Drop.
pub(super) struct Page {
    pub(super) address: usize,
    pub(super) length: usize,
    pub(super) backing: Option<(Arc<Section>, usize)>,
    pub(super) protection: u32,
}

impl Page {
    /// Releases this page to a placeholder without releasing its address reservation.
    pub(super) fn unmap(&mut self) -> io::Result<()> {
        if self.backing.is_some() {
            // SAFETY: this object owns the view. Unlike MapViewOfFile3, UnmapViewOfFile2
            // requires an actual process handle; -1 is the current-process pseudo handle.
            if unsafe {
                UnmapViewOfFile2(
                    -1isize as *mut _,
                    self.address as *mut _,
                    MEM_PRESERVE_PLACEHOLDER,
                )
            } == 0
            {
                return Err(io::Error::last_os_error());
            }
            self.backing = None;
        }
        Ok(())
    }

    /// Replaces this exact placeholder. Failure leaves it reserved and unmapped.
    pub(super) fn map(
        &mut self,
        section: Arc<Section>,
        offset: usize,
        protection: u32,
    ) -> io::Result<()> {
        // SAFETY: the owned placeholder matches the view length and base; Section owns its handle.
        let mapped = unsafe {
            MapViewOfFile3(
                section.handle.as_raw_handle(),
                ptr::null_mut(),
                self.address as *mut _,
                offset as u64,
                self.length,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                ptr::null(),
                0,
            )
        };
        if mapped.is_null() {
            return Err(io::Error::last_os_error());
        }
        assert_eq!(
            mapped as usize, self.address,
            "Windows replaced a different placeholder"
        );
        self.backing = Some((section, offset));
        self.protection = PAGE_READWRITE;
        // A view's initial access is its ceiling for subsequent VirtualProtect calls. Keep
        // that ceiling writable and apply the current access separately, so remapping a
        // read-only or inaccessible page does not permanently prevent restoring its access.
        if let Err(error) = self.protect(protection) {
            self.unmap().unwrap_or_else(|error| {
                super::failure::failed(error, "undo newly mapped GPU page")
            });
            return Err(error);
        }
        Ok(())
    }

    /// Change only current access; the section and the view's initial ceiling stay writable.
    pub(super) fn protect(&mut self, protection: u32) -> io::Result<()> {
        if self.protection == protection {
            return Ok(());
        }
        let mut old = 0;
        // SAFETY: this object owns the mapped page and the writable out parameter.
        if unsafe { VirtualProtect(self.address as *mut _, self.length, protection, &mut old) } == 0
        {
            return Err(io::Error::last_os_error());
        }
        self.protection = protection;
        Ok(())
    }
}

impl Drop for Page {
    /// Every page releases its own view and placeholder, including a partly built arena.
    fn drop(&mut self) {
        self.unmap()
            .unwrap_or_else(|error| super::failure::failed(error, "unmap owned GPU page"));
        // SAFETY: this object owns the whole placeholder; size zero releases it.
        assert_ne!(
            unsafe { VirtualFree(self.address as *mut _, 0, MEM_RELEASE) },
            0,
            "releasing an owned GPU placeholder failed: {}",
            io::Error::last_os_error()
        );
    }
}

/// A contiguous virtual address range made of independently replaceable shared page views.
pub struct PageArena {
    base: usize,
    pages: Vec<Option<Page>>,
    live: usize,
}

impl PageArena {
    /// Reserves the range and maps zero-filled, aliasable pages over it.
    pub fn new(length: usize) -> io::Result<Self> {
        let section = Section::new(length)?;
        // SAFETY: a fresh reservation in this process; no fixed address or extended parameters.
        let base = unsafe {
            VirtualAlloc2(
                ptr::null_mut(),
                ptr::null_mut(),
                length,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                ptr::null(),
                0,
            )
        };
        if base.is_null() {
            return Err(io::Error::last_os_error());
        }
        let mut tail = Page {
            address: base as usize,
            length,
            backing: None,
            protection: PAGE_NOACCESS,
        };
        let mut pages = Vec::with_capacity(length / PAGE);
        while tail.length > PAGE {
            // SAFETY: split only this owned placeholder; preserving both resulting reservations.
            if unsafe {
                VirtualFree(
                    tail.address as *mut _,
                    PAGE,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                )
            } == 0
            {
                return Err(io::Error::last_os_error());
            }
            pages.push(Page {
                address: tail.address,
                length: PAGE,
                backing: None,
                protection: PAGE_NOACCESS,
            });
            tail.address += PAGE;
            tail.length -= PAGE;
        }
        pages.push(tail);
        for (i, page) in pages.iter_mut().enumerate() {
            page.map(Arc::clone(&section), i * PAGE, PAGE_READWRITE)?;
        }
        Ok(Self {
            base: base as usize,
            live: pages.len(),
            pages: pages.into_iter().map(Some).collect(),
        })
    }

    /// The range's stable address. Dereferencing it requires the caller's memory synchronization.
    pub fn address(&self) -> usize {
        self.base
    }

    /// The original reservation's extent, including any pages explicitly released later.
    pub fn length(&self) -> usize {
        self.pages.len() * PAGE
    }

    /// Whether a page remains mapped; partial deallocation creates holes in an arena.
    pub fn contains(&self, page: usize) -> bool {
        self.pages.get(page).is_some_and(Option::is_some)
    }

    /// Releases one complete page without disturbing any neighbour's reservation or mapping.
    pub fn release(&mut self, page: usize) {
        if let Some(view) = self.pages[page].take() {
            drop(view);
            self.live -= 1;
        }
    }

    /// Constant-time empty check: teardown must not rescan a large arena after every page.
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// The backing of a page; keeping it alive lets another arena outlive this one.
    pub fn backing(&self, page: usize) -> (Arc<Section>, usize) {
        self.pages[page]
            .as_ref()
            .unwrap_or_else(|| super::failure::abort("arena page is mapped"))
            .backing
            .as_ref()
            .unwrap_or_else(|| super::failure::abort("live arena page has backing"))
            .clone()
    }

    /// Rebinds one page to the same physical bytes as a section offset, with no byte copy.
    /// Other pages stay mapped throughout. The caller excludes access to the replaced page.
    pub fn replace(&mut self, page: usize, section: Arc<Section>, offset: usize) -> io::Result<()> {
        if !offset.is_multiple_of(PAGE)
            || offset
                .checked_add(PAGE)
                .is_none_or(|end| end > section.length)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "section offset is not a valid page",
            ));
        }
        let target = self.pages[page]
            .as_mut()
            .unwrap_or_else(|| super::failure::abort("remap target is mapped"));
        let previous = target
            .backing
            .clone()
            .unwrap_or_else(|| super::failure::abort("live arena page is mapped"));
        let protection = target.protection;
        target.unmap()?;
        if let Err(error) = target.map(section, offset, protection) {
            // Restore the old mapping before reporting the kernel refusal. Returning an unmapped
            // GPU address would hide the cause and turn a recoverable allocation error into a fault.
            target
                .map(previous.0, previous.1, protection)
                .unwrap_or_else(|error| {
                    super::failure::failed(error, "restore GPU page after refused remap")
                });
            return Err(error);
        }
        Ok(())
    }

    /// Changes CPU page access. Imports must observe the same read-only contract.
    #[cfg(test)]
    pub fn protect(&mut self, page: usize, writable: bool) -> io::Result<()> {
        self.protect_access(page, if writable { 3 } else { 1 })
    }

    /// Mach VM read/write bits mapped to Windows access, including inaccessible guard pages.
    pub fn protect_access(&mut self, page: usize, access: i32) -> io::Result<()> {
        let target = self.pages[page]
            .as_mut()
            .unwrap_or_else(|| super::failure::abort("protection target is mapped"));
        let protection = match access {
            0 => PAGE_NOACCESS,
            1 => PAGE_READONLY,
            2 | 3 => PAGE_READWRITE,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "unsupported page access",
                ));
            }
        };
        target.protect(protection)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A guard page remains a valid remap destination without becoming CPU-readable.
    #[test]
    fn inaccessible_page_can_be_remapped_then_restored() {
        let source = PageArena::new(PAGE).unwrap();
        let mut target = PageArena::new(PAGE).unwrap();
        // SAFETY: the test exclusively owns this live writable mapping.
        unsafe { (source.address() as *mut u8).write_volatile(0x91) };
        target.protect_access(0, 0).unwrap();
        let (section, offset) = source.backing(0);
        target.replace(0, section, offset).unwrap();
        target.protect_access(0, 1).unwrap();
        // SAFETY: only after explicitly restoring read access may the test inspect the alias.
        assert_eq!(
            unsafe { (target.address() as *const u8).read_volatile() },
            0x91
        );
    }

    /// The guest's second 4 KiB page must be a true shared alias despite not being 64 KiB aligned.
    #[test]
    fn four_kib_offset_is_shared_in_both_directions() {
        let source = PageArena::new(4 * PAGE).unwrap();
        let mut view = PageArena::new(PAGE).unwrap();
        let (section, offset) = source.backing(1);
        assert_eq!(offset, PAGE);
        view.replace(0, section, offset).unwrap();
        // SAFETY: both arenas own mapped writable pages; this test is the sole accessor.
        unsafe {
            let source_byte = (source.address() + PAGE + 37) as *mut u8;
            let alias_byte = (view.address() + 37) as *mut u8;
            source_byte.write_volatile(0x61);
            assert_eq!(alias_byte.read_volatile(), 0x61);
            alias_byte.write_volatile(0xb7);
            assert_eq!(source_byte.read_volatile(), 0xb7);
        }
    }

    /// IOSurface page order and repeated pages survive source unmapping, without stale copies.
    #[test]
    fn reordered_pages_and_alias_of_alias_keep_the_same_memory() {
        let source = PageArena::new(4 * PAGE).unwrap();
        let mut view = PageArena::new(4 * PAGE).unwrap();
        for (i, from) in [3, 1, 2, 1].into_iter().enumerate() {
            let (section, offset) = source.backing(from);
            view.replace(i, section, offset).unwrap();
        }
        let mut second = PageArena::new(PAGE).unwrap();
        let (section, offset) = view.backing(1);
        second.replace(0, section, offset).unwrap();
        drop(source);
        // SAFETY: the aliases retain the section and own these mapped writable pages.
        unsafe {
            (second.address() as *mut u8).write_volatile(0xe3);
            assert_eq!(((view.address() + PAGE) as *const u8).read_volatile(), 0xe3);
            assert_eq!(
                ((view.address() + 3 * PAGE) as *const u8).read_volatile(),
                0xe3
            );
            assert_eq!((view.address() as *const u8).read_volatile(), 0);
        }
        drop(view);
        // SAFETY: the last alias still owns its mapped page and the section.
        assert_eq!(
            unsafe { (second.address() as *const u8).read_volatile() },
            0xe3
        );
    }

    /// Invalid guest offsets fail before a live mapping is touched; protection can be restored.
    #[test]
    fn refused_remap_keeps_previous_bytes() {
        let mut view = PageArena::new(2 * PAGE).unwrap();
        // SAFETY: uniquely accessed writable arena.
        unsafe { (view.address() as *mut u8).write_volatile(0x42) };
        let (section, _) = view.backing(1);
        assert!(view.replace(0, section, 17).is_err());
        view.protect(0, false).unwrap();
        // SAFETY: the page remains mapped readable.
        assert_eq!(
            unsafe { (view.address() as *const u8).read_volatile() },
            0x42
        );
        view.protect(0, true).unwrap();
    }
}
