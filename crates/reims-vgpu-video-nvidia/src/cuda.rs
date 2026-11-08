//! The CUDA driver API (libcuda), loaded at run time: the library, the
//! host's GPUs, a GPU's primary context made current around each call, and
//! the 2D copies that move frames between the GPU and host memory.
//!
//! NVDEC and NVENC both run on a CUDA context. Each [`Gpu`] retains its
//! device's primary context, the one the CUDA runtime would share, and
//! pushes it onto the calling thread for each operation, so that any
//! thread may drive it: a context may be current on several threads at
//! once, and CUDA's driver API may be called from any thread. The push
//! lasts as long as a [`Current`], which pops it on the thread that pushed
//! it, on every path, a panic's unwinding included: it cannot be sent to
//! another thread, whose stack of contexts is its own. A [`Stream`] is
//! work queued in order on a context, which a decoder's maps and copies run
//! on, so that their end is asked after rather than waited for.

pub(crate) mod ffi;

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::marker::PhantomData;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use crate::dl::Library;
use crate::error::Error;
use ffi::{CUcontext, CUdevice, CUresult, CudaFunctions, CudaRegistration};

/// libcuda, loaded and initialised once per process.
pub(crate) struct Cuda {
    pub(crate) f: CudaFunctions,
    /// Its registration of host memory, where it has it.
    pub(crate) registration: Option<CudaRegistration>,
    driver_version: i32,
    _library: Library,
}

static CUDA: OnceLock<Result<Cuda, Error>> = OnceLock::new();

impl Cuda {
    /// libcuda, loading and initialising it on the first call. On a host
    /// without NVIDIA's driver every call returns the same load error.
    pub(crate) fn get() -> Result<&'static Self, Error> {
        CUDA.get_or_init(Self::load).as_ref().map_err(Clone::clone)
    }

    fn load() -> Result<Self, Error> {
        let library = Library::open(c"libcuda.so.1")?;
        let f = CudaFunctions::load(&library)?;
        let registration = CudaRegistration::load(&library)
            .inspect_err(|e| log::info!("{e}: zero copy into guest memory is off"))
            .ok();
        let cuda = Self {
            f,
            registration,
            driver_version: 0,
            _library: library,
        };
        // SAFETY: cuInit takes flags, which must be 0.
        cuda.check("cuInit", unsafe { (cuda.f.cuInit)(0) })?;
        let mut version: c_int = 0;
        // SAFETY: the pointer is to a live c_int.
        cuda.check("cuDriverGetVersion", unsafe {
            (cuda.f.cuDriverGetVersion)(&raw mut version)
        })?;
        Ok(Self {
            driver_version: version,
            ..cuda
        })
    }

    /// `Ok` for `CUDA_SUCCESS`, else the error with the driver's name and
    /// text for the code.
    pub(crate) fn check(&self, call: &'static str, result: CUresult) -> Result<(), Error> {
        if result == ffi::CUDA_SUCCESS {
            return Ok(());
        }
        Err(Error::Cuda {
            call,
            code: result,
            message: self.describe(result),
        })
    }

    /// `cuGetErrorName: cuGetErrorString` of a code.
    fn describe(&self, result: CUresult) -> String {
        let text = |describe: unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult| {
            let mut text: *const c_char = std::ptr::null();
            // SAFETY: both functions take any code and write a pointer to
            // a static NUL-terminated string, or fail and write nothing.
            let ok = unsafe { describe(result, &raw mut text) } == ffi::CUDA_SUCCESS;
            if ok && !text.is_null() {
                // SAFETY: non-null, static and NUL-terminated, as above.
                unsafe { CStr::from_ptr(text) }
                    .to_string_lossy()
                    .into_owned()
            } else {
                "unknown".to_owned()
            }
        };
        format!(
            "{}: {}",
            text(self.f.cuGetErrorName),
            text(self.f.cuGetErrorString)
        )
    }

    /// The driver's CUDA version, as `cuDriverGetVersion` numbers it
    /// (13040 for 13.4).
    pub(crate) fn driver_version(&self) -> i32 {
        self.driver_version
    }

    /// The host's GPUs, in CUDA's order.
    pub(crate) fn devices(&self) -> Result<Vec<DeviceInfo>, Error> {
        let mut count: c_int = 0;
        // SAFETY: the pointer is to a live c_int.
        self.check("cuDeviceGetCount", unsafe {
            (self.f.cuDeviceGetCount)(&raw mut count)
        })?;
        (0..count)
            .map(|ordinal| {
                let mut device: CUdevice = 0;
                // SAFETY: the pointer is to a live CUdevice; the ordinal
                // is below the count.
                self.check("cuDeviceGet", unsafe {
                    (self.f.cuDeviceGet)(&raw mut device, ordinal)
                })?;
                self.device_info(ordinal, device)
            })
            .collect()
    }

    fn device_info(&self, ordinal: i32, device: CUdevice) -> Result<DeviceInfo, Error> {
        let string = |call: &'static str,
                      get: unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult|
         -> Result<String, Error> {
            let mut buffer: [c_char; 256] = [0; 256];
            // The driver writes at most this many bytes, its NUL included.
            let len = c_int::try_from(buffer.len()).unwrap_or(0);
            // SAFETY: the buffer holds `len` bytes, which bounds the
            // driver's write, and the device is a handle cuDeviceGet
            // returned.
            self.check(call, unsafe { get(buffer.as_mut_ptr(), len, device) })?;
            // The last byte is NUL whatever the driver wrote, so the string
            // ends inside the buffer.
            if let Some(last) = buffer.last_mut() {
                *last = 0;
            }
            // SAFETY: NUL-terminated within the buffer, as above.
            Ok(unsafe { CStr::from_ptr(buffer.as_ptr()) }
                .to_string_lossy()
                .into_owned())
        };
        let attribute = |attribute| -> Result<i32, Error> {
            let mut value: c_int = 0;
            // SAFETY: the pointer is to a live c_int and the device is a
            // handle cuDeviceGet returned.
            self.check("cuDeviceGetAttribute", unsafe {
                (self.f.cuDeviceGetAttribute)(&raw mut value, attribute, device)
            })?;
            Ok(value)
        };
        Ok(DeviceInfo {
            ordinal,
            name: string("cuDeviceGetName", self.f.cuDeviceGetName)?,
            pci_bus_id: string("cuDeviceGetPCIBusId", self.f.cuDeviceGetPCIBusId)?,
            compute_capability: (
                attribute(ffi::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
                attribute(ffi::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
            ),
            pci: (
                attribute(ffi::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID)?,
                attribute(ffi::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)?,
                attribute(ffi::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)?,
            ),
        })
    }
}

/// One of the host's GPUs, as CUDA reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    /// Its CUDA ordinal.
    pub ordinal: i32,
    /// `cuDeviceGetName`.
    pub name: String,
    /// `cuDeviceGetPCIBusId`: `domain:bus:device.function`, such as
    /// `0000:01:00.0`.
    pub pci_bus_id: String,
    /// The compute capability, major and minor (12.0 for Blackwell).
    pub compute_capability: (i32, i32),
    /// The PCI domain, bus and device numbers.
    pub pci: (i32, i32, i32),
}

/// Which GPU to open.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuSelector {
    /// The first GPU CUDA lists.
    First,
    /// A CUDA ordinal.
    Ordinal(i32),
    /// A PCI address, `0000:01:00.0` or `01:00.0`.
    PciBusId(String),
    /// A DRM node of the GPU, such as `/dev/dri/renderD128`, found through
    /// its PCI address in sysfs.
    DrmNode(String),
}

impl GpuSelector {
    /// Reads a selector as a host's configuration names a GPU: a DRM node
    /// path, a PCI address, a CUDA ordinal, or nothing for the first GPU.
    pub fn parse(name: &str) -> Self {
        if name.is_empty() {
            Self::First
        } else if name.starts_with("/dev/") {
            Self::DrmNode(name.to_owned())
        } else if let Ok(ordinal) = name.parse() {
            Self::Ordinal(ordinal)
        } else {
            Self::PciBusId(name.to_owned())
        }
    }
}

/// The PCI address of a DRM node, from sysfs: the name of the PCI device
/// its `device` link points at.
fn drm_node_pci_address(node: &str) -> Result<String, Error> {
    let name = Path::new(node)
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| Error::NoDevice(format!("at {node}")))?;
    let link = std::fs::read_link(format!("/sys/class/drm/{name}/device"))
        .map_err(|e| Error::NoDevice(format!("at {node}: {e}")))?;
    link.file_name()
        .and_then(|address| address.to_str())
        .map(str::to_owned)
        .ok_or_else(|| Error::NoDevice(format!("at {node}: no PCI address")))
}

/// A GPU, with its primary context retained until it is dropped.
pub struct Gpu {
    cuda: &'static Cuda,
    device: CUdevice,
    context: CUcontext,
    info: DeviceInfo,
}

// SAFETY: a CUDA context is not bound to a thread: each operation pushes
// it onto the calling thread and pops it after ("A context may be current
// to one or more host threads"), and the handles are never mutated.
unsafe impl Send for Gpu {}
// SAFETY: as above; the driver serialises what it must on a context.
unsafe impl Sync for Gpu {}

impl Gpu {
    /// Opens the GPU `selector` names, retaining its primary context.
    pub fn open(selector: &GpuSelector) -> Result<Self, Error> {
        let cuda = Cuda::get()?;
        let devices = cuda.devices()?;
        let normalise = |address: &str| {
            let lower = address.to_ascii_lowercase();
            if lower.matches(':').count() == 1 {
                format!("0000:{lower}")
            } else {
                lower
            }
        };
        let info = match selector {
            GpuSelector::First => devices.into_iter().next(),
            GpuSelector::Ordinal(ordinal) => devices.into_iter().find(|d| d.ordinal == *ordinal),
            GpuSelector::PciBusId(address) => {
                let address = normalise(address);
                devices
                    .into_iter()
                    .find(|d| d.pci_bus_id.to_ascii_lowercase() == address)
            }
            GpuSelector::DrmNode(node) => {
                let address = normalise(&drm_node_pci_address(node)?);
                devices
                    .into_iter()
                    .find(|d| d.pci_bus_id.to_ascii_lowercase() == address)
            }
        }
        .ok_or_else(|| Error::NoDevice(format!("{selector:?}")))?;
        let bus_id = CString::new(info.pci_bus_id.clone())
            .map_err(|_| Error::NoDevice(format!("{selector:?}")))?;
        let mut device: CUdevice = 0;
        // SAFETY: the pointer is to a live CUdevice; the bus ID is
        // NUL-terminated.
        cuda.check("cuDeviceGetByPCIBusId", unsafe {
            (cuda.f.cuDeviceGetByPCIBusId)(&raw mut device, bus_id.as_ptr())
        })?;
        let mut context: CUcontext = std::ptr::null_mut();
        // SAFETY: the pointer is to a live CUcontext, and the device is a
        // handle the driver returned.
        cuda.check("cuDevicePrimaryCtxRetain", unsafe {
            (cuda.f.cuDevicePrimaryCtxRetain)(&raw mut context, device)
        })?;
        Ok(Self {
            cuda,
            device,
            context,
            info,
        })
    }

    /// What CUDA reports of the GPU.
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// The driver's CUDA version (13040 for 13.4).
    pub fn driver_version(&self) -> i32 {
        self.cuda.driver_version()
    }

    pub(crate) fn cuda(&self) -> &'static Cuda {
        self.cuda
    }

    /// The context, for the APIs that take it (NVENC's session).
    pub(crate) fn context(&self) -> CUcontext {
        self.context
    }

    /// Makes the GPU's context current on this thread until the guard is
    /// dropped.
    pub(crate) fn push(&self) -> Result<Current<'_>, Error> {
        // SAFETY: the context is retained for as long as `self` lives,
        // which the guard borrows.
        self.cuda.check("cuCtxPushCurrent", unsafe {
            (self.cuda.f.cuCtxPushCurrent_v2)(self.context)
        })?;
        Ok(Current {
            gpu: self,
            _thread: PhantomData,
        })
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // SAFETY: the device's primary context was retained once by
        // `open`, and is released once here; nothing made on it outlives
        // the Gpu, since everything that uses it holds it.
        let result = unsafe { (self.cuda.f.cuDevicePrimaryCtxRelease)(self.device) };
        if let Err(e) = self.cuda.check("cuDevicePrimaryCtxRelease", result) {
            log::warn!("{e}");
        }
    }
}

/// A CUDA stream on a GPU's context, destroyed when dropped: the work
/// queued on it runs in order, apart from other streams' and the null
/// stream's (`CU_STREAM_NON_BLOCKING`), and [`Stream::is_done`] asks
/// whether all of it has run, without waiting (`crate::wait`).
pub(crate) struct Stream {
    gpu: Arc<Gpu>,
    handle: ffi::CUstream,
}

// SAFETY: a stream is not bound to a thread: each call that takes it pushes
// its GPU's context on the calling thread, and the handle is never mutated.
unsafe impl Send for Stream {}
// SAFETY: as above; the driver serialises what it must on a stream.
unsafe impl Sync for Stream {}

impl Stream {
    /// A new stream on `gpu`'s context.
    pub(crate) fn new(gpu: &Arc<Gpu>) -> Result<Self, Error> {
        let cuda = gpu.cuda();
        let _current = gpu.push()?;
        let mut handle: ffi::CUstream = std::ptr::null_mut();
        // SAFETY: the pointer is to a live CUstream; the GPU's context is
        // current, which the stream is made on.
        cuda.check("cuStreamCreate", unsafe {
            (cuda.f.cuStreamCreate)(&raw mut handle, ffi::CU_STREAM_NON_BLOCKING)
        })?;
        Ok(Self {
            gpu: Arc::clone(gpu),
            handle,
        })
    }

    /// The handle, to queue work on, with its GPU's context current.
    pub(crate) fn raw(&self) -> ffi::CUstream {
        self.handle
    }

    /// Whether everything queued on it has run (`cuStreamQuery`): `Ok(false)`
    /// while some has not, and the error of work that failed.
    pub(crate) fn is_done(&self) -> Result<bool, Error> {
        let cuda = self.gpu.cuda();
        let _current = self.gpu.push()?;
        // SAFETY: a live stream of the GPU's context, current on this
        // thread.
        match unsafe { (cuda.f.cuStreamQuery)(self.handle) } {
            ffi::CUDA_ERROR_NOT_READY => Ok(false),
            result => cuda.check("cuStreamQuery", result).map(|()| true),
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        let cuda = self.gpu.cuda();
        let Ok(_current) = self.gpu.push() else {
            log::warn!("a CUDA stream is left undestroyed: its GPU's context is not to be had");
            return;
        };
        // SAFETY: made by cuStreamCreate, destroyed once. Work still queued
        // on it runs, and the driver frees the stream after it (the
        // driver API's reference, `cuStreamDestroy`): the call does not
        // wait for it.
        let result = unsafe { (cuda.f.cuStreamDestroy_v2)(self.handle) };
        if let Err(e) = cuda.check("cuStreamDestroy", result) {
            log::warn!("{e}");
        }
    }
}

/// Page-locked host memory of CUDA's (`cuMemHostAlloc`), freed as it
/// drops.
pub(crate) struct Pinned {
    gpu: Arc<Gpu>,
    at: *mut u8,
    len: usize,
}

// SAFETY: host memory, reached through `&self` and `&mut self` alone, and
// freed with the GPU's context pushed on whichever thread drops it.
unsafe impl Send for Pinned {}
// SAFETY: as above.
unsafe impl Sync for Pinned {}

impl Pinned {
    pub(crate) fn new(gpu: &Arc<Gpu>, len: usize) -> Result<Self, Error> {
        let cuda = gpu.cuda();
        let _current = gpu.push()?;
        let mut at: *mut c_void = std::ptr::null_mut();
        // SAFETY: the out pointer is live; no flags.
        cuda.check("cuMemHostAlloc", unsafe {
            (cuda.f.cuMemHostAlloc)(&raw mut at, len, 0)
        })?;
        if at.is_null() {
            return Err(Error::Unsupported(
                "cuMemHostAlloc gave no memory".to_owned(),
            ));
        }
        // CUDA's memory comes uninitialised, and `bytes` lends it as bytes:
        // zeroed once, so that every byte is one before any is read.
        // SAFETY: `len` bytes CUDA allocated, writable and not yet lent.
        unsafe { std::ptr::write_bytes(at.cast::<u8>(), 0, len) };
        Ok(Self {
            gpu: Arc::clone(gpu),
            at: at.cast::<u8>(),
            len,
        })
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        // SAFETY: `len` bytes CUDA allocated, initialised by `new`, alive
        // until the drop, and written only through `bytes_mut`'s borrow.
        unsafe { std::slice::from_raw_parts(self.at, self.len) }
    }

    pub(crate) fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as above; `&mut self` makes this the only borrow. The
        // GPU's copies into it run inside that borrow (as
        // `nvdec::decoder`'s and `jpeg`'s read backs make them).
        unsafe { std::slice::from_raw_parts_mut(self.at, self.len) }
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        let cuda = self.gpu.cuda();
        let Ok(_current) = self.gpu.push() else {
            return;
        };
        // SAFETY: allocated by cuMemHostAlloc, freed once.
        let result = unsafe { (cuda.f.cuMemFreeHost)(self.at.cast::<c_void>()) };
        if let Err(e) = cuda.check("cuMemFreeHost", result) {
            log::warn!("{e}");
        }
    }
}

/// The GPU's context, current on this thread while this lives. It is
/// neither `Send` nor `Sync`: it pops the context it pushed off the stack
/// of the thread that pushed it.
pub(crate) struct Current<'a> {
    gpu: &'a Gpu,
    _thread: PhantomData<*const ()>,
}

impl Drop for Current<'_> {
    fn drop(&mut self) {
        let mut popped: CUcontext = std::ptr::null_mut();
        // SAFETY: `push` made the context current on this thread, which
        // the guard never leaves, and every push in the crate is popped in
        // the reverse order by its own guard, so the top of the stack is
        // ours; the pointer is to a live CUcontext.
        let result = unsafe { (self.gpu.cuda.f.cuCtxPopCurrent_v2)(&raw mut popped) };
        match self.gpu.cuda.check("cuCtxPopCurrent", result) {
            Err(e) => log::warn!("{e}"),
            Ok(()) if popped != self.gpu.context => {
                log::warn!("cuCtxPopCurrent popped another context than the GPU's");
            }
            Ok(()) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selectors_read_as_a_host_names_its_gpu() {
        assert_eq!(GpuSelector::parse(""), GpuSelector::First);
        assert_eq!(GpuSelector::parse("1"), GpuSelector::Ordinal(1));
        assert_eq!(
            GpuSelector::parse("/dev/dri/renderD128"),
            GpuSelector::DrmNode("/dev/dri/renderD128".to_owned())
        );
        assert_eq!(
            GpuSelector::parse("0000:01:00.0"),
            GpuSelector::PciBusId("0000:01:00.0".to_owned())
        );
    }
}
