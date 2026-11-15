//! What the runtime imports for host memory on Windows: the memory itself, by its pointer
//! (`VK_EXT_external_memory_host`), every range of the VM's a view of a section ([`super::vm`]).
//!
//! A Windows driver refused a view mapped read-only at its import and imported a read-write view
//! of the same pages (measured on the UHD Graphics 630 host, Intel's 101.2145 driver, Windows 11
//! 26H2, 2026-11-15; NVIDIA's and AMD's drivers on Windows unmeasured), as Linux drivers refuse
//! read-only pages. So, as on Linux, the runtime imports no read-only memory: the views' gate
//! refuses pages the backing table shows read-only ([`import_gate`]), and the runtime makes the
//! buffer or heap over a copy of them whose GPU writes it drops, as Metal's GPU drops them
//! ([`ImportView::read_only`]; `COPIES_READ_ONLY`). Nothing else is gated: Windows counts no
//! mappings against a limit, and its views are never mapped through dma-bufs.

use std::ptr::NonNull;
use std::sync::Arc;

use crate::core::num::ToU64 as _;
use crate::os::files;
use metal2vulkan_runtime as rt;

use super::vm;

/// Whether the runtime makes read-only memory, and memory made with `MTLResourceOptions` bit 24,
/// over a copy of it whose GPU writes it drops, where Metal's GPU keeps them from the memory
/// itself: on Windows, as on Linux (module docs).
pub(crate) const COPIES_READ_ONLY: bool = true;

/// The memory the runtime imports for `length` bytes at `at` (module docs).
#[derive(Debug)]
pub(crate) struct ImportView {
    at: NonNull<u8>,
    /// The table shows every page of the import mapped writable ([`ImportView::vouched`]).
    vouched: bool,
    /// The table shows a page of the memory mapped read-only ([`ImportView::read_only`]).
    read_only: bool,
}

// SAFETY: `at` is an address only, which nothing here reads through: it is handed to the runtime.
unsafe impl Send for ImportView {}

// SAFETY: as for `Send`: a shared view only answers where it is.
unsafe impl Sync for ImportView {}

impl ImportView {
    /// The first byte the runtime imports.
    pub(crate) fn at(&self) -> NonNull<u8> {
        self.at
    }

    /// Whether the table shows a page of the memory mapped read-only: the gate refuses the pages
    /// ([`import_gate`]), so the runtime makes the object over a copy of them whose GPU writes it
    /// drops (module docs).
    pub(crate) fn read_only(&self) -> bool {
        self.read_only
    }

    /// Whether the runtime may be told the import's pages are writable ([`host_memory_vouched`]):
    /// every page of the memory the table shows mapped with write permission now. Where it does
    /// not know a page, or a page is read-only, the runtime asks Windows as before.
    pub(crate) fn vouched(&self) -> bool {
        self.vouched
    }

    /// The word that the runtime's import of the memory was made ([`ImportMade`]): nothing is
    /// waiting for it on Windows.
    pub(crate) fn made(&self) -> ImportMade {
        ImportMade
    }

    /// The runtime's deallocator of the import, and what keeps the import's memory mapped beside
    /// the bytes themselves: `inner` as it is, and nothing more on Windows.
    pub(crate) fn release(
        self,
        inner: Option<rt::Deallocator>,
    ) -> (Option<rt::Deallocator>, Option<rt::KeepAlive>) {
        (inner, None)
    }
}

/// The word that the runtime's import of a view was made, which the caller gives once the call
/// that imports it returned: nothing waits for it on Windows.
#[derive(Debug)]
pub(crate) struct ImportMade;

impl ImportMade {
    /// Gives the word: nothing on Windows.
    pub(crate) fn made(self, _device: &rt::Device) {}
}

/// The views' gate over the runtime's import of a no-copy buffer's or heap's pages, by their
/// pointer (`rt::HostMemory::with_import_gate`), which every [`host_memory`] carries: it refuses
/// pages the table shows read-only, so the runtime copies them, dropping the GPU's writes (module
/// docs).
fn import_gate(request: rt::ImportRequest) -> bool {
    gate(request.start.as_ptr().addr().to_u64(), request.length)
}

/// [`import_gate`]'s answer for the `length` bytes of pages from `first`.
fn gate(first: u64, length: u64) -> bool {
    files::writable(first, length) != Some(false)
}

/// The runtime's host memory at `at`, the first byte of an import ([`ImportView::at`]), with the
/// views' gate ([`import_gate`]).
///
/// # Safety
///
/// `rt::HostMemory::new`'s contract, for the memory at `at`.
pub(crate) unsafe fn host_memory(at: NonNull<u8>) -> rt::HostMemory {
    static GATE: std::sync::LazyLock<rt::ImportGate> =
        std::sync::LazyLock::new(|| Arc::new(import_gate));
    // SAFETY: the caller's.
    unsafe { rt::HostMemory::new(at) }.with_import_gate(Arc::clone(&GATE))
}

/// [`host_memory`], its pages' write permission vouched for where `vouched`
/// ([`ImportView::vouched`]): the runtime then asks Windows nothing of the pages before it imports
/// them.
///
/// # Safety
///
/// [`host_memory`]'s; and where `vouched`, every page of the import is mapped writable by this
/// process now (the table's record, [`ImportView::vouched`]), and stays mapped so while it is
/// imported, as `rt::HostMemory::assume_writable` asks: the library lowers no protection of a view
/// or of the guest's RAM once made, and an unmap of its own waits for the import's free
/// (`crate::remap`'s `UnmapHold`).
pub(crate) unsafe fn host_memory_vouched(at: NonNull<u8>, vouched: bool) -> rt::HostMemory {
    // SAFETY: the caller's.
    let memory = unsafe { host_memory(at) };
    match vouched {
        // SAFETY: the caller's.
        true => unsafe { memory.assume_writable() },
        false => memory,
    }
}

/// Why this thread's last dma-buf source made none: never, as Windows imports no dma-buf.
pub(crate) fn dma_buf_refusal() -> Option<String> {
    None
}

/// The memory the runtime imports for `length` bytes at `at`: those bytes, where every page of
/// them is mapped writable or one has no section behind it (which the runtime then imports as it
/// would anywhere), and where one is mapped read-only, which the gate refuses, so the runtime
/// copies them, dropping the GPU's writes (module docs).
pub(crate) fn import_view(at: NonNull<u8>, length: u64) -> ImportView {
    let plain = ImportView {
        at,
        vouched: false,
        read_only: false,
    };
    let start = at.as_ptr().addr().to_u64();
    let page = vm::page_size();
    let first = start & !(page - 1);
    let Some(end) = start
        .checked_add(length.max(1))
        .and_then(|e| e.checked_next_multiple_of(page))
    else {
        return plain;
    };
    match files::writable(first, end - first) {
        Some(false) => ImportView {
            read_only: true,
            ..plain
        },
        writable => ImportView {
            vouched: writable == Some(true),
            ..plain
        },
    }
}

#[cfg(test)]
mod tests {
    use crate::core::platform::os::vm::{HostVm as _, KernReturn, VmProt};

    use super::super::WindowsVm;
    use super::*;

    /// A range of the VM's is vouched for and imported in place; one read-only is not, and the
    /// gate refuses its pages; memory the table does not know is neither
    #[test]
    fn imports_follow_the_tables_protections() {
        let v = WindowsVm::new();
        let p = vm::page_size();
        let a = v.allocate_anywhere(2 * p).unwrap();
        let at = NonNull::new(a.as_ptr::<u8>()).unwrap();
        let view = import_view(at, 2 * p);
        assert!(view.vouched() && !view.read_only());
        let first = a.addr().to_u64();
        assert!(gate(first, 2 * p));
        assert_eq!(v.protect_max(a, p, VmProt::READ), Ok(()));
        let view = import_view(at, 2 * p);
        assert!(!view.vouched() && view.read_only());
        assert!(!gate(first, 2 * p));
        let stack = [0u8; 16];
        let foreign = NonNull::from(&stack).cast::<u8>();
        let view = import_view(foreign, 16);
        assert!(!view.vouched() && !view.read_only());
        assert_eq!(v.deallocate(a, 2 * p), KernReturn::SUCCESS);
    }
}
