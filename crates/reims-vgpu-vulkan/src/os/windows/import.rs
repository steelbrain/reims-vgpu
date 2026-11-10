//! Windows GPU-import views of read-only guest pages.
//!
//! Vulkan's host-pointer import requires writable mappings. Keep the guest/task
//! mapping read-only and map its section pages a second time for the import, as
//! the Linux backend does. Lifetime and pointer translation are explicit: the
//! runtime's deallocator sees the original address, after its import has ended.

use super::{mapping::PAGE, vm};
use metal2vulkan_runtime as rt;
use reims_vgpu_core::platform::os::vm::HostAddr;
use reims_vgpu_core::sync::Mutex;
use std::collections::BTreeMap;
use std::ptr::NonNull;

/// GPU writes to a writable twin reach its backing; advertise the same deviation as Linux.
pub(crate) const TWINS_READ_ONLY: bool = true;
static TWINS: Mutex<BTreeMap<usize, (usize, usize)>> = Mutex::new(BTreeMap::new());

struct Twin {
    memory: vm::OwnMemory,
}
impl Drop for Twin {
    fn drop(&mut self) {
        TWINS.lock().remove(&(self.memory.first() as usize));
        // OwnMemory drops after this method, after Vulkan released its imported allocation.
    }
}

pub(crate) struct ImportView {
    at: NonNull<u8>,
    twin: Option<Twin>,
}
// SAFETY: at is only an address; its owner and the runtime synchronize byte access.
unsafe impl Send for ImportView {}
// SAFETY: shared access returns an address and never dereferences it.
unsafe impl Sync for ImportView {}

impl ImportView {
    pub(crate) fn at(&self) -> NonNull<u8> {
        self.at
    }
    pub(crate) fn writes_kept(&self) -> bool {
        self.twin.is_some()
    }

    /// Keep the alias until GPU use ends, but invoke the caller with its original pointer.
    pub(crate) fn release(
        self,
        original: NonNull<u8>,
        inner: Option<rt::Deallocator>,
    ) -> (Option<rt::Deallocator>, Option<rt::KeepAlive>) {
        let Some(twin) = self.twin else {
            return (inner, None);
        };
        let original = original.as_ptr() as usize;
        let inner = inner.map(|inner| -> rt::Deallocator {
            Box::new(move |_, length| {
                inner(
                    NonNull::new(original as *mut u8).unwrap_or_else(|| {
                        super::failure::abort(concat!(module_path!(), ":", line!()))
                    }),
                    length,
                )
            })
        });
        (inner, Some(Box::new(twin)))
    }
}

/// Map only readable, non-writable ranges; writable pages are imported in place.
pub(crate) fn import_view(at: NonNull<u8>, length: u64) -> ImportView {
    let address = at.as_ptr() as usize;
    let alias =
        vm::writable_import_alias(HostAddr::from_addr(address), length).unwrap_or_else(|error| {
            super::failure::failed(error, "Windows could not map a writable GPU import alias")
        });
    let Some(memory) = alias else {
        return ImportView { at, twin: None };
    };
    let first = address / PAGE * PAGE;
    let extent = ((address - first) as u64 + length).next_multiple_of(PAGE as u64) as usize;
    let mapped = memory.first() as usize;
    TWINS.lock().insert(mapped, (extent, first));
    ImportView {
        at: NonNull::new((mapped + address - first) as *mut u8)
            .unwrap_or_else(|| super::failure::abort(concat!(module_path!(), ":", line!()))),
        twin: Some(Twin { memory }),
    }
}

/// Metal's contents pointer is the original range, even when Vulkan imports its twin.
pub(crate) fn shown_at(at: NonNull<u8>) -> NonNull<u8> {
    let address = at.as_ptr() as usize;
    let twins = TWINS.lock();
    let Some((&base, &(length, original))) = twins.range(..=address).next_back() else {
        return at;
    };
    if address >= base + length {
        return at;
    }
    NonNull::new((original + address - base) as *mut u8)
        .unwrap_or_else(|| super::failure::abort(concat!(module_path!(), ":", line!())))
}
