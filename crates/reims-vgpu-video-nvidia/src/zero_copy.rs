//! Zero copy on NVIDIA: guest buffers registered with CUDA, so that the
//! GPU copies a decoded frame straight into a guest's buffer, with no
//! staging and no copy by the CPU.
//!
//! The host lends a view of the guest buffer's pages, contiguous in its
//! address space (reims-vgpu's view of a surface over the guest's RAM).
//! The view is registered with CUDA once per mapping
//! (`cuMemHostRegister`, portable to every context), which pins its pages
//! and makes them memory the GPU copies into by DMA, and kept by the
//! mapping's identity in core's [`DeviceImports`], process-wide, since a
//! range of host memory is registered once in a process whichever GPU
//! registered it. The first decoders made make the process's, on their
//! runtime, whose thread drops a registration once its mapping goes. It is
//! unregistered when the mapping goes, before the hold that keeps the view
//! mapped is dropped.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Arc, OnceLock};

use reims_vgpu_video_core::device_memory::{DeviceImports, DeviceMemory, HostView};
use reims_vgpu_video_core::runtime::Runtime;

use crate::cuda::Gpu;
use crate::cuda::ffi::CU_MEMHOSTREGISTER_PORTABLE;

/// The most registrations kept at once: a guest's decoders use a few pools
/// of 10 to 30 buffers each, and each registration pins its buffer's pages
/// (core's [`DeviceImports`] drops the one used longest ago past it).
const MOST_REGISTRATIONS: usize = 96;

/// The process's registrations of guest buffers, from the first decoders'
/// [`make_registrations`].
static REGISTRATIONS: OnceLock<DeviceImports<Registration>> = OnceLock::new();

/// Makes the process's registrations, dropped on a thread of `runtime`'s,
/// where no decoders have made them yet.
pub(crate) fn make_registrations(runtime: &Arc<dyn Runtime>) {
    REGISTRATIONS.get_or_init(|| DeviceImports::new(runtime, "nvidia", MOST_REGISTRATIONS));
}

/// A guest buffer's view registered with CUDA: unregistered as it drops,
/// with the GPU that registered it current, and the view's hold dropped
/// after.
pub(crate) struct Registration {
    gpu: Arc<Gpu>,
    /// The view, held mapped while this lives; dropped after the
    /// unregistration.
    view: HostView,
}

impl Registration {
    /// `memory`'s view registered on `gpu`; `Err` says why not, for the
    /// log.
    fn new(gpu: &Arc<Gpu>, memory: &dyn DeviceMemory) -> Result<Self, String> {
        let view = memory
            .host_view()
            .map_err(|error| format!("no view of the buffer: {error}"))?;
        let fits = (view.offset.checked_add(memory.length())).is_some_and(|end| end <= view.size);
        if !fits {
            return Err("the host's view is shorter than the buffer".to_owned());
        }
        let cuda = gpu.cuda();
        let registration = (cuda.registration.as_ref())
            .ok_or("this driver registers no host memory (no cuMemHostRegister_v2)")?;
        let _current = gpu.push().map_err(|error| error.to_string())?;
        // SAFETY: the view's whole pages, mapped readable and writable
        // while its hold lives, which this registration keeps until it has
        // unregistered them.
        let result = unsafe {
            (registration.cuMemHostRegister_v2)(
                view.first.as_ptr().cast::<c_void>(),
                view.size,
                CU_MEMHOSTREGISTER_PORTABLE,
            )
        };
        cuda.check("cuMemHostRegister", result)
            .map_err(|error| format!("CUDA refused to register the buffer: {error}"))?;
        log::debug!(
            "a guest buffer registered: {} bytes, its first {} into its first page",
            view.size,
            view.offset
        );
        Ok(Self {
            gpu: Arc::clone(gpu),
            view,
        })
    }

    /// The host address of the buffer's first byte, which the GPU writes
    /// by DMA.
    pub(crate) fn first(&self) -> NonNull<u8> {
        // SAFETY: the offset is inside the view (checked at registration).
        unsafe { self.view.first.add(self.view.offset) }
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        let cuda = self.gpu.cuda();
        let Some(registration) = cuda.registration.as_ref() else {
            return;
        };
        let Ok(_current) = self.gpu.push() else {
            log::warn!("a guest buffer stays registered: its GPU's context is not to be had");
            return;
        };
        // SAFETY: registered by `new`, unregistered once; the view is still
        // mapped, its hold dropped after this.
        let result = unsafe {
            (registration.cuMemHostUnregister)(self.view.first.as_ptr().cast::<c_void>())
        };
        if let Err(error) = cuda.check("cuMemHostUnregister", result) {
            log::warn!("{error}");
        }
    }
}

/// `memory`'s view registered on `gpu`: the process's registration of its
/// mapping, made now where there is none. `Err` says why there is none: no
/// decoders made the process's registrations, the mapping went, or CUDA
/// refused it.
pub(crate) fn registered(
    gpu: &Arc<Gpu>,
    memory: &dyn DeviceMemory,
) -> Result<Arc<Registration>, String> {
    // Made with the decoders whose picture is written.
    let registrations =
        (REGISTRATIONS.get()).ok_or("no decoders made the process's registrations")?;
    registrations.get_or_make(
        &memory.identity(),
        0,
        || Registration::new(gpu, memory),
        || "the buffer's mapping went while it was held".to_owned(),
    )
}
