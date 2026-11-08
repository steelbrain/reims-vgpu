//! Another API's memory imported into CUDA ([`ImportedBuffer`]), for a picture to land in a
//! device's own memory by one copy within the GPU ([`crate::nvidia::AnnexBPicture::write_into_device`]):
//! a Vulkan device's buffer whose memory it exported as an opaque file descriptor
//! (`VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT`), imported with `cuImportExternalMemory` and
//! mapped whole with `cuExternalMemoryGetMappedBuffer`. The memory is the exporter's GPU's, so
//! the import is made on the CUDA device of that GPU; CUDA refuses another's.
//!
//! The import takes the descriptor where it succeeds, as CUDA's reference says of an opaque
//! file descriptor, and closes it where it fails. The mapping is freed (`cuMemFree`) and the
//! import destroyed as the [`ImportedBuffer`] drops, with its GPU's context pushed. The exporter's
//! buffer must outlive it: what the GPU writes through an import of freed memory lands nowhere
//! anyone owns.

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
    /// The buffer's first byte, in the GPU's address space.
    address: CUdeviceptr,
    /// Its bytes.
    length: usize,
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
            .field("length", &self.length)
            .finish()
    }
}

impl ImportedBuffer {
    /// The memory `fd` describes, imported on `gpu`: an allocation of `allocation_size` bytes,
    /// the exporter's dedicated allocation where `dedicated`, of which the buffer is `length`
    /// bytes from `offset`. The import takes `fd` where it succeeds, and closes it where it fails.
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
        let Some(length) = fits.then(|| usize::try_from(length).ok()).flatten() else {
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
        let buffer = CUDA_EXTERNAL_MEMORY_BUFFER_DESC {
            offset,
            size: length as u64,
            flags: 0,
            reserved: [0; 16],
        };
        let mut address: CUdeviceptr = 0;
        // SAFETY: the import is live; the range lies inside its allocation (checked).
        let result = unsafe {
            (external.cuExternalMemoryGetMappedBuffer)(&raw mut address, memory, &raw const buffer)
        };
        if let Err(error) = cuda.check("cuExternalMemoryGetMappedBuffer", result) {
            // SAFETY: imported above, mapped nowhere, destroyed once.
            let _ = unsafe { (external.cuDestroyExternalMemory)(memory) };
            return Err(error);
        }
        Ok(Self {
            gpu: Arc::clone(gpu),
            memory,
            address,
            length,
        })
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
            return;
        };
        // SAFETY: the mapping `cuExternalMemoryGetMappedBuffer` gave, freed once, as CUDA's
        // reference says a mapped buffer is.
        let freed = unsafe { (cuda.f.cuMemFree_v2)(self.address) };
        if let Err(error) = cuda.check("cuMemFree", freed) {
            log::warn!("{error}");
        }
        // SAFETY: imported by `import`, destroyed once, its mapping freed above.
        let destroyed = unsafe { (external.cuDestroyExternalMemory)(self.memory) };
        if let Err(error) = cuda.check("cuDestroyExternalMemory", destroyed) {
            log::warn!("{error}");
        }
    }
}
