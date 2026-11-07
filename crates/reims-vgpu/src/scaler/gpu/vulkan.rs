//! The scaler's kernel on the Vulkan platform (`reims_vgpu_vulkan::gpu::scaler`), on the runtime's
//! system default device, the one the library's GPU device draws on.
//!
//! A surface is reached in place through the host's view of its pages ([`DeviceMemory::host_view`],
//! on Linux): imported once a mapping, with the view's hold as the import's keep-alive, and kept
//! by the mapping's identity until the mapping goes ([`DeviceImports`]), so a transform after the
//! first on the same surfaces imports nothing. Where the host lends no view, as on macOS, the
//! surface is staged.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use reims_vgpu_video_core::device_memory::{DeviceImports, DeviceMemory};
use reims_vgpu_video_core::runtime::Runtime;
use reims_vgpu_vulkan::gpu::scaler::{GpuBuffer, Kernel as GpuKernel, Pass, ScalerGpu, Submitted};

use super::{Dispatch, Gpu, Kernel};

/// The most surfaces' imports kept at once.
const IMPORTS_KEPT: usize = 32;

/// The kernel on the Vulkan platform, and its imports of surfaces' pages.
pub(crate) struct VulkanGpu {
    gpu: ScalerGpu,
    imports: DeviceImports<Imported>,
}

/// A surface's pages imported, and where its first byte lies in them.
pub(crate) struct Imported {
    buffer: GpuBuffer,
    base: usize,
}

/// A buffer the kernel binds: one of the GPU's own, or a surface's import, shared with the
/// imports kept.
pub(crate) enum VulkanBuffer {
    Own(GpuBuffer),
    Import(Arc<Imported>),
}

impl VulkanBuffer {
    fn get(&self) -> &GpuBuffer {
        match self {
            VulkanBuffer::Own(buffer) => buffer,
            VulkanBuffer::Import(imported) => &imported.buffer,
        }
    }
}

impl VulkanGpu {
    /// The process's, made at its first use, its imports dropped on threads `runtime` starts:
    /// `Err` with why where the runtime finds no device or refuses the kernel, which every later
    /// ask answers the same.
    pub(crate) fn process(runtime: &Arc<dyn Runtime>) -> Result<Arc<VulkanGpu>, String> {
        static GPU: OnceLock<Result<Arc<VulkanGpu>, String>> = OnceLock::new();
        GPU.get_or_init(|| {
            ScalerGpu::new().map(|gpu| {
                Arc::new(VulkanGpu {
                    gpu,
                    imports: DeviceImports::new(runtime, "scaler", IMPORTS_KEPT),
                })
            })
        })
        .clone()
    }
}

impl Gpu for VulkanGpu {
    type Buffer = VulkanBuffer;
    type Run = Submitted;

    fn name(&self) -> &str {
        self.gpu.name()
    }

    fn shared(&self, length: usize) -> Result<VulkanBuffer, String> {
        self.gpu.shared_buffer(length).map(VulkanBuffer::Own)
    }

    fn table(&self, bytes: &[u8]) -> Result<Self::Buffer, String> {
        self.gpu.table_buffer(bytes).map(VulkanBuffer::Own)
    }

    fn scratch(&self, length: usize) -> Result<VulkanBuffer, String> {
        self.gpu.private_buffer(length).map(VulkanBuffer::Own)
    }

    fn contents(buffer: &mut VulkanBuffer) -> Option<&mut [u8]> {
        match buffer {
            VulkanBuffer::Own(buffer) => buffer.contents(),
            VulkanBuffer::Import(_) => None,
        }
    }

    fn memory(
        &self,
        memory: &dyn DeviceMemory,
        _writes: bool,
    ) -> Result<Option<(VulkanBuffer, usize)>, String> {
        let identity = memory.identity();
        let imported = self.imports.get_or_make(
            &identity,
            0,
            || {
                let view = memory.host_view().map_err(|e| e.to_string())?;
                // SAFETY: the view is whole pages of the host's mapping of the surface, which
                // the host lends only where it may be written, kept mapped by its hold, which the
                // import keeps until the runtime lets go of it.
                let buffer = unsafe { self.gpu.memory_buffer(view.first, view.size, view.hold) }?;
                Ok(Imported {
                    buffer,
                    base: view.offset,
                })
            },
            || "the surface's mapping is gone".to_owned(),
        );
        match imported {
            Ok(imported) => {
                let base = imported.base;
                Ok(Some((VulkanBuffer::Import(imported), base)))
            }
            Err(why) => Err(why),
        }
    }

    fn submit(
        &self,
        dispatches: &[Dispatch<'_, VulkanBuffer>],
        done: Box<dyn FnOnce() + Send>,
    ) -> Result<Submitted, String> {
        self.gpu.submit(&passes(dispatches), done)
    }

    fn submit_apart(
        &self,
        dispatches: &[Dispatch<'_, VulkanBuffer>],
        done: Box<dyn FnOnce() + Send>,
    ) -> Result<Submitted, String> {
        self.gpu.submit_apart(&passes(dispatches), done)
    }

    fn finished(run: &Submitted) -> Option<Result<(), String>> {
        run.finished()
    }

    fn gpu_time(run: &Submitted) -> Option<Duration> {
        run.gpu_time()
    }
}

impl VulkanGpu {
    /// The runtime's counts of translations, makings and the driver cache's hits and misses, and
    /// its shader cache's directory, for the probes (`ScalerGpu::cache_counts`).
    #[cfg(test)]
    pub(crate) fn cache_counts(&self) -> ([u64; 5], std::path::PathBuf) {
        self.gpu.cache_counts()
    }
}

/// The platform's passes of `dispatches`.
fn passes<'a>(dispatches: &[Dispatch<'a, VulkanBuffer>]) -> Vec<Pass<'a>> {
    dispatches
        .iter()
        .map(|d| Pass {
            kernel: kernel(d.kernel),
            source: d.source.get(),
            destination: d.destination.get(),
            taps: d.taps.get(),
            params: d.params,
            lines: d.lines.get(),
            threads: d.threads,
            group: d.group,
        })
        .collect()
}

/// The platform's kernel of `kernel`.
fn kernel(kernel: Kernel) -> GpuKernel {
    match kernel {
        Kernel::Lines => GpuKernel::Lines,
        Kernel::Transform => GpuKernel::Transform,
        Kernel::Across => GpuKernel::Across,
        Kernel::Shared => GpuKernel::Shared,
        Kernel::Quads => GpuKernel::Quads,
        Kernel::TransformAdaptive => GpuKernel::TransformAdaptive,
        Kernel::TransformBothPlanes => GpuKernel::TransformBothPlanes,
    }
}
