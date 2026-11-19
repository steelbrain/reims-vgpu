//! Another API's memory imported into CUDA ([`ImportedBuffer`]), for a picture to land in a
//! device's own memory by one copy within the GPU ([`crate::nvidia::AnnexBPicture::write_into_device`]),
//! and for NVENC to read a frame where the device's GPU wrote it
//! ([`crate::nvidia::NvencEncoder::encode_device`]):
//! a Vulkan device's buffer whose memory it exported as an opaque file descriptor
//! (`VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT`), imported with `cuImportExternalMemory` and
//! mapped with `cuExternalMemoryGetMappedBuffer`. The memory is the exporter's GPU's, so
//! the import is made on the CUDA device of that GPU; CUDA refuses another's.
//!
//! **Mapped from the allocation's first byte.** The mapping spans the allocation from its first
//! byte to the buffer's last, and the buffer's address is the mapping's plus its offset, so a
//! buffer at any offset maps however its allocation was made. CUDA maps a buffer of an import
//! made dedicated (`CUDA_EXTERNAL_MEMORY_DEDICATED`, which an exporter's dedicated allocation
//! must be imported as) only from offset 0: a mapping from 224, 256, 4096 or 65536 bytes in is
//! refused `CUDA_ERROR_INVALID_VALUE`, while one of any length from offset 0 maps, the
//! allocation's whole length among them; a shared allocation's buffers map at any offset
//! (**measured**, an RTX 2080, and at 224 an RTX 3090, 2026-11-19).
//!
//! The import takes the descriptor where it succeeds, as CUDA's reference says of an opaque
//! file descriptor, and closes it where it fails. The mapping is freed (`cuMemFree`) and the
//! import destroyed as the [`ImportedBuffer`] drops, with its GPU's context pushed. The exporter's
//! buffer must outlive it: what the GPU writes through an import of freed memory lands nowhere
//! anyone owns. An import may hold the exporter's buffer itself ([`ImportedBuffer::holding`]),
//! which it lets go of only once the import is destroyed.

use std::any::Any;
use std::os::fd::{IntoRawFd as _, OwnedFd};
use std::sync::Arc;

use crate::nvidia::cuda::Gpu;
use crate::nvidia::cuda::ffi::{
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD, CUDA_EXTERNAL_MEMORY_BUFFER_DESC,
    CUDA_EXTERNAL_MEMORY_DEDICATED, CUDA_EXTERNAL_MEMORY_HANDLE_DESC, CUdeviceptr,
    CUexternalMemory, CudaExternalMemoryHandle,
};
use crate::nvidia::error::Error;

/// A buffer of another API's memory, imported on a GPU and mapped there (module docs).
pub struct ImportedBuffer {
    gpu: Arc<Gpu>,
    memory: CUexternalMemory,
    /// The mapping, from the allocation's first byte (module docs).
    mapped: CUdeviceptr,
    /// The buffer's first byte, in the GPU's address space: `offset` bytes into the mapping.
    address: CUdeviceptr,
    /// Its bytes.
    length: usize,
    /// What it holds until it is destroyed ([`ImportedBuffer::holding`]).
    owner: Option<Box<dyn Any + Send + Sync>>,
}

// SAFETY: the import and the mapping are the driver's handles, which any thread may use with
// the GPU's context pushed; nothing here is written after the import.
unsafe impl Send for ImportedBuffer {}
// SAFETY: as above; `&self` reads the handles alone.
unsafe impl Sync for ImportedBuffer {}

impl std::fmt::Debug for ImportedBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImportedBuffer")
            .field("gpu", &self.gpu.info().pci_bus_id)
            .field("address", &format_args!("{:#x}", self.address))
            .field("mapped", &format_args!("{:#x}", self.mapped))
            .field("length", &self.length)
            .finish()
    }
}

impl ImportedBuffer {
    /// The memory `fd` describes, imported on `gpu`: an allocation of `allocation_size` bytes,
    /// the exporter's dedicated allocation where `dedicated`, of which the buffer is `length`
    /// bytes from `offset`, mapped from the allocation's first byte (module docs). The import
    /// takes `fd` where it succeeds, and closes it where it fails.
    pub fn import(
        gpu: &Arc<Gpu>,
        fd: OwnedFd,
        allocation_size: u64,
        offset: u64,
        length: u64,
        dedicated: bool,
    ) -> Result<Self, Error> {
        let cuda = gpu.cuda();
        let external = (cuda.external.as_ref())
            .ok_or_else(|| Error::Unsupported("a driver with no cuImportExternalMemory".into()))?;
        let fits = offset
            .checked_add(length)
            .is_some_and(|end| end <= allocation_size);
        let Some((start, length)) = fits
            .then(|| Some((usize::try_from(offset).ok()?, usize::try_from(length).ok()?)))
            .flatten()
        else {
            return Err(Error::Unsupported(format!(
                "a buffer of {length} bytes at {offset} in an allocation of {allocation_size}"
            )));
        };
        let _current = gpu.push()?;
        let mut description = CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
            r#type: CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD,
            handle: CudaExternalMemoryHandle { fd: 0 },
            size: allocation_size,
            flags: if dedicated {
                CUDA_EXTERNAL_MEMORY_DEDICATED
            } else {
                0
            },
            reserved: [0; 16],
        };
        let raw = fd.into_raw_fd();
        description.handle = CudaExternalMemoryHandle { fd: raw };
        let mut memory: CUexternalMemory = std::ptr::null_mut();
        // SAFETY: the description is live for the call, and names a descriptor this function
        // owns, whose memory is `allocation_size` bytes (the exporter's word); the GPU's context
        // is current.
        let result =
            unsafe { (external.cuImportExternalMemory)(&raw mut memory, &raw const description) };
        if let Err(error) = cuda.check("cuImportExternalMemory", result) {
            // SAFETY: the import failed, so the descriptor is still this function's own.
            drop(unsafe { <OwnedFd as std::os::fd::FromRawFd>::from_raw_fd(raw) });
            return Err(error);
        }
        // From the allocation's first byte, which a dedicated import alone maps from (module
        // docs), to the buffer's last.
        let buffer = CUDA_EXTERNAL_MEMORY_BUFFER_DESC {
            offset: 0,
            size: offset + length as u64,
            flags: 0,
            reserved: [0; 16],
        };
        let mut mapped: CUdeviceptr = 0;
        // SAFETY: the import is live; the range lies inside its allocation (checked).
        let result = unsafe {
            (external.cuExternalMemoryGetMappedBuffer)(&raw mut mapped, memory, &raw const buffer)
        };
        if let Err(error) = cuda.check("cuExternalMemoryGetMappedBuffer", result) {
            // SAFETY: imported above, mapped nowhere, destroyed once.
            let _ = unsafe { (external.cuDestroyExternalMemory)(memory) };
            return Err(error);
        }
        Ok(Self {
            gpu: Arc::clone(gpu),
            memory,
            mapped,
            address: mapped + start as CUdeviceptr,
            length,
            owner: None,
        })
    }

    /// The import, holding `owner`, the exporter's buffer, until the import is destroyed, so
    /// that the memory outlives every use of the import (module docs). An import that cannot be
    /// destroyed as it drops (its GPU's context not to be had) never lets go of it.
    #[must_use]
    pub fn holding(mut self, owner: Box<dyn Any + Send + Sync>) -> Self {
        self.owner = Some(owner);
        self
    }

    /// Its bytes.
    pub fn length(&self) -> usize {
        self.length
    }

    /// The GPU it was imported on.
    pub(crate) fn gpu(&self) -> &Arc<Gpu> {
        &self.gpu
    }

    /// The address of its byte at `offset`, in its GPU's address space.
    pub(crate) fn address_at(&self, offset: usize) -> CUdeviceptr {
        self.address + offset as CUdeviceptr
    }
}

impl Drop for ImportedBuffer {
    fn drop(&mut self) {
        let cuda = self.gpu.cuda();
        let Some(external) = cuda.external.as_ref() else {
            return;
        };
        let Ok(_current) = self.gpu.push() else {
            log::warn!("an imported buffer stays imported: its GPU's context is not to be had");
            std::mem::forget(self.owner.take());
            return;
        };
        // SAFETY: the mapping `cuExternalMemoryGetMappedBuffer` gave, freed once, as CUDA's
        // reference says a mapped buffer is.
        let freed = unsafe { (cuda.f.cuMemFree_v2)(self.mapped) };
        if let Err(error) = cuda.check("cuMemFree", freed) {
            log::warn!(kind:% = error; "{error}");
        }
        // SAFETY: imported by `import`, destroyed once, its mapping freed above.
        let destroyed = unsafe { (external.cuDestroyExternalMemory)(self.memory) };
        if let Err(error) = cuda.check("cuDestroyExternalMemory", destroyed) {
            log::warn!(kind:% = error; "{error}");
        }
    }
}

#[cfg(test)]
mod tests;
