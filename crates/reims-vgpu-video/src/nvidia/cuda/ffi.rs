//! The CUDA driver API's declarations this crate uses, written from the
//! interface and checked against ffnvcodec's `dynlink_cuda.h` (13.1.15.0)
//! by `crate::nvidia::layout`'s test: each type's size and offsets, and each
//! function's signature.

#![allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    clippy::upper_case_acronyms,
    reason = "NVIDIA's names, kept for grepping the headers and for the layout check"
)]

use std::ffi::{c_char, c_int, c_uint, c_ulonglong, c_void};

use crate::nvidia::dl::functions;

/// `CUresult`: 0 is success.
pub(crate) type CUresult = c_int;
/// `CUdevice`: an ordinal's handle.
pub(crate) type CUdevice = c_int;
/// `CUdeviceptr`: a device address, 64 bits on a 64-bit host.
pub(crate) type CUdeviceptr = c_ulonglong;
/// `CUcontext`.
pub(crate) type CUcontext = *mut c_void;
/// `CUstream`.
pub(crate) type CUstream = *mut c_void;
/// `CUarray`.
pub(crate) type CUarray = *mut c_void;
/// `CUdevice_attribute`.
pub(crate) type CUdevice_attribute = c_int;
/// `CUmemorytype`.
pub(crate) type CUmemorytype = c_int;

pub(crate) const CUDA_SUCCESS: CUresult = 0;
/// `CUDA_ERROR_NOT_READY`: work queued on a stream is not done yet
/// (`cuStreamQuery`).
pub(crate) const CUDA_ERROR_NOT_READY: CUresult = 600;

/// `CU_STREAM_NON_BLOCKING`: the stream does not wait for the null stream,
/// nor it for this one.
pub(crate) const CU_STREAM_NON_BLOCKING: c_uint = 0x1;

pub(crate) const CU_MEMORYTYPE_HOST: CUmemorytype = 1;
pub(crate) const CU_MEMORYTYPE_DEVICE: CUmemorytype = 2;

pub(crate) const CU_DEVICE_ATTRIBUTE_PCI_BUS_ID: CUdevice_attribute = 33;
pub(crate) const CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID: CUdevice_attribute = 34;
pub(crate) const CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID: CUdevice_attribute = 50;
pub(crate) const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: CUdevice_attribute = 75;
pub(crate) const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: CUdevice_attribute = 76;

/// `CUDA_MEMCPY2D`: a 2D copy between host and device memory.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CUDA_MEMCPY2D {
    pub(crate) srcXInBytes: usize,
    pub(crate) srcY: usize,
    pub(crate) srcMemoryType: CUmemorytype,
    pub(crate) srcHost: *const c_void,
    pub(crate) srcDevice: CUdeviceptr,
    pub(crate) srcArray: CUarray,
    pub(crate) srcPitch: usize,
    pub(crate) dstXInBytes: usize,
    pub(crate) dstY: usize,
    pub(crate) dstMemoryType: CUmemorytype,
    pub(crate) dstHost: *mut c_void,
    pub(crate) dstDevice: CUdeviceptr,
    pub(crate) dstArray: CUarray,
    pub(crate) dstPitch: usize,
    pub(crate) WidthInBytes: usize,
    pub(crate) Height: usize,
}

functions! {
    /// libcuda's functions, by the names its header binds them to (the
    /// `_v2` entry points where it names those).
    pub(crate) struct CudaFunctions {
        cuInit: fn(c_uint) -> CUresult = c"cuInit";
        cuDriverGetVersion: fn(*mut c_int) -> CUresult = c"cuDriverGetVersion";
        cuDeviceGetCount: fn(*mut c_int) -> CUresult = c"cuDeviceGetCount";
        cuDeviceGet: fn(*mut CUdevice, c_int) -> CUresult = c"cuDeviceGet";
        cuDeviceGetName: fn(*mut c_char, c_int, CUdevice) -> CUresult = c"cuDeviceGetName";
        cuDeviceGetAttribute: fn(*mut c_int, CUdevice_attribute, CUdevice) -> CUresult
            = c"cuDeviceGetAttribute";
        cuDeviceGetPCIBusId: fn(*mut c_char, c_int, CUdevice) -> CUresult
            = c"cuDeviceGetPCIBusId";
        cuDeviceGetByPCIBusId: fn(*mut CUdevice, *const c_char) -> CUresult
            = c"cuDeviceGetByPCIBusId";
        cuDevicePrimaryCtxRetain: fn(*mut CUcontext, CUdevice) -> CUresult
            = c"cuDevicePrimaryCtxRetain";
        cuDevicePrimaryCtxRelease: fn(CUdevice) -> CUresult = c"cuDevicePrimaryCtxRelease";
        cuCtxPushCurrent_v2: fn(CUcontext) -> CUresult = c"cuCtxPushCurrent_v2";
        cuCtxPopCurrent_v2: fn(*mut CUcontext) -> CUresult = c"cuCtxPopCurrent_v2";
        cuMemFree_v2: fn(CUdeviceptr) -> CUresult = c"cuMemFree_v2";
        cuMemHostAlloc: fn(*mut *mut c_void, usize, c_uint) -> CUresult = c"cuMemHostAlloc";
        cuMemFreeHost: fn(*mut c_void) -> CUresult = c"cuMemFreeHost";
        cuMemcpy2DAsync_v2: fn(*const CUDA_MEMCPY2D, CUstream) -> CUresult
            = c"cuMemcpy2DAsync_v2";
        cuStreamCreate: fn(*mut CUstream, c_uint) -> CUresult = c"cuStreamCreate";
        cuStreamQuery: fn(CUstream) -> CUresult = c"cuStreamQuery";
        cuStreamDestroy_v2: fn(CUstream) -> CUresult = c"cuStreamDestroy_v2";
        cuGetErrorName: fn(CUresult, *mut *const c_char) -> CUresult = c"cuGetErrorName";
        cuGetErrorString: fn(CUresult, *mut *const c_char) -> CUresult = c"cuGetErrorString";
    }
}

functions! {
    /// libcuda's registration of host memory, which ffnvcodec's header does
    /// not declare, so `crate::nvidia::layout` cannot check it: written from the
    /// CUDA driver API's reference (`cuMemHostRegister`, which `cuda.h`
    /// binds to `cuMemHostRegister_v2`, and `cuMemHostUnregister`), whose
    /// symbols libcuda 615.71.09 exports (`nm -D` on x86-64 Linux, 2026-10-25).
    /// Loaded apart from [`CudaFunctions`], so that a driver without them
    /// loses zero copy alone (`crate::nvidia::zero_copy`).
    pub(crate) struct CudaRegistration {
        cuMemHostRegister_v2: fn(*mut c_void, usize, c_uint) -> CUresult
            = c"cuMemHostRegister_v2";
        cuMemHostUnregister: fn(*mut c_void) -> CUresult = c"cuMemHostUnregister";
    }
}

functions! {
    /// libcuda's import of another API's memory (a Vulkan device's, as an
    /// opaque file descriptor), as ffnvcodec's header declares it. Loaded
    /// apart from [`CudaFunctions`], so that a driver without it loses only
    /// the landing in a device's memory (`crate::nvidia::external`).
    pub(crate) struct CudaExternalMemory {
        cuImportExternalMemory:
            fn(*mut CUexternalMemory, *const CUDA_EXTERNAL_MEMORY_HANDLE_DESC) -> CUresult
            = c"cuImportExternalMemory";
        cuExternalMemoryGetMappedBuffer: fn(
            *mut CUdeviceptr,
            CUexternalMemory,
            *const CUDA_EXTERNAL_MEMORY_BUFFER_DESC,
        ) -> CUresult = c"cuExternalMemoryGetMappedBuffer";
        cuDestroyExternalMemory: fn(CUexternalMemory) -> CUresult = c"cuDestroyExternalMemory";
    }
}

/// `CUexternalMemory`.
pub(crate) type CUexternalMemory = *mut c_void;
/// `CUexternalMemoryHandleType`.
pub(crate) type CUexternalMemoryHandleType = c_int;

/// `CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD`: a POSIX file descriptor of
/// memory a Vulkan device exported (`VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT`).
pub(crate) const CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD: CUexternalMemoryHandleType = 1;
/// `CUDA_EXTERNAL_MEMORY_DEDICATED`: the memory is a dedicated allocation. CUDA's `cuda.h`
/// defines it; ffnvcodec's headers, which the layout test reads, do not.
pub(crate) const CUDA_EXTERNAL_MEMORY_DEDICATED: c_uint = 0x1;

/// `CUDA_EXTERNAL_MEMORY_HANDLE_DESC`'s `win32` member of its `handle`
/// union, which only sizes the union here.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CudaExternalMemoryWin32 {
    pub(crate) handle: *mut c_void,
    pub(crate) name: *const c_void,
}

/// `CUDA_EXTERNAL_MEMORY_HANDLE_DESC`'s `handle` union.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) union CudaExternalMemoryHandle {
    pub(crate) fd: c_int,
    pub(crate) win32: CudaExternalMemoryWin32,
}

/// `CUDA_EXTERNAL_MEMORY_HANDLE_DESC`: the memory `cuImportExternalMemory`
/// imports.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
    pub(crate) r#type: CUexternalMemoryHandleType,
    pub(crate) handle: CudaExternalMemoryHandle,
    pub(crate) size: c_ulonglong,
    pub(crate) flags: c_uint,
    pub(crate) reserved: [c_uint; 16],
}

/// `CUDA_EXTERNAL_MEMORY_BUFFER_DESC`: the range of imported memory
/// `cuExternalMemoryGetMappedBuffer` maps.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC {
    pub(crate) offset: c_ulonglong,
    pub(crate) size: c_ulonglong,
    pub(crate) flags: c_uint,
    pub(crate) reserved: [c_uint; 16],
}

/// `CU_MEMHOSTREGISTER_PORTABLE`: the registered memory is pinned memory to
/// every context, not just the one current at the call (the reference).
pub(crate) const CU_MEMHOSTREGISTER_PORTABLE: c_uint = 0x01;

// The size ffnvcodec 13.1.15.0's header gives on x86-64 Linux
// (2026-10-24); `crate::nvidia::layout`'s test checks every offset too.
#[cfg(target_pointer_width = "64")]
const _: () = assert!(size_of::<CUDA_MEMCPY2D>() == 128);
// The C layout on x86-64 (the handle's 16-byte union at 8, the size at 24, the flags at 32, then
// 16 reserved words), which `crate::nvidia::layout`'s test checks against ffnvcodec's header.
#[cfg(target_pointer_width = "64")]
const _: () = assert!(size_of::<CUDA_EXTERNAL_MEMORY_HANDLE_DESC>() == 104);
#[cfg(target_pointer_width = "64")]
const _: () = assert!(size_of::<CUDA_EXTERNAL_MEMORY_BUFFER_DESC>() == 88);
