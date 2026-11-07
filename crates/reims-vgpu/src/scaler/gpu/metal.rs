//! The scaler's kernel on Metal (`reims_vgpu_metal::gpu::scaler`), on the system's default
//! device, the one the library's GPU device draws on.
//!
//! A surface is reached in place through the IOSurface the host lends
//! ([`DeviceMemory::io_surface`]): a buffer over it starts at the surface's first byte, as the
//! lend's bytes do, and runs past them to the allocation's size (`reims_vgpu_metal::gpu::scaler`),
//! so the planes are bound at their own offsets and only the lend's bytes are reached. The buffer
//! retains the IOSurface, a hold the guest sees, so it is made for one transform and dropped
//! before the transform returns, or kept with the device's hold on the surfaces where the GPU is
//! not done by the deadline. A surface the host says it may not write is not bound as a
//! destination: it is staged, and its lend refuses the write. Where Metal makes no buffer over an
//! IOSurface, the surface is staged too; over a guest's pages, which the host's IOSurfaces of a
//! VM lie in, no such buffer has been made yet, as no VM with the device has run on a Mac.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use reims_vgpu_metal::gpu::scaler::{GpuBuffer, Kernel as GpuKernel, Pass, ScalerGpu, Submitted};
use reims_vgpu_video_core::device_memory::DeviceMemory;

use super::{Dispatch, Gpu, Kernel};

/// The kernel on Metal.
pub(crate) struct MetalGpu(ScalerGpu);

impl MetalGpu {
    /// The process's, made at its first use: `Err` with why where Metal has no device or refuses
    /// the kernel, which every later ask answers the same.
    pub(crate) fn process() -> Result<Arc<MetalGpu>, String> {
        static GPU: OnceLock<Result<Arc<MetalGpu>, String>> = OnceLock::new();
        GPU.get_or_init(|| ScalerGpu::new().map(|gpu| Arc::new(MetalGpu(gpu))))
            .clone()
    }
}

impl Gpu for MetalGpu {
    type Buffer = GpuBuffer;
    type Run = Submitted;

    fn name(&self) -> &str {
        self.0.name()
    }

    fn shared(&self, length: usize) -> Result<GpuBuffer, String> {
        self.0.shared_buffer(length)
    }

    fn table(&self, bytes: &[u8]) -> Result<Self::Buffer, String> {
        self.0.table_buffer(bytes)
    }

    fn scratch(&self, length: usize) -> Result<GpuBuffer, String> {
        self.0.private_buffer(length)
    }

    fn contents(buffer: &mut GpuBuffer) -> Option<&mut [u8]> {
        buffer.contents()
    }

    fn memory(
        &self,
        memory: &dyn DeviceMemory,
        writes: bool,
    ) -> Result<Option<(GpuBuffer, usize)>, String> {
        let Ok(lent) = memory.io_surface() else {
            return Ok(None);
        };
        if writes && !lent.writable() {
            return Ok(None);
        }
        // SAFETY: the lend keeps the IOSurface live while `memory` lives, which the caller's hold
        // on the surface outlives, and the buffer goes before that hold (module docs); the host
        // lends it for writing where it is bound as a destination.
        let buffer = unsafe { self.0.surface_buffer(lent.as_ptr()) }?;
        if buffer.length() < memory.length() {
            return Err(format!(
                "a buffer of {} bytes over a surface of {}",
                buffer.length(),
                memory.length()
            ));
        }
        Ok(Some((buffer, 0)))
    }

    fn submit(
        &self,
        dispatches: &[Dispatch<'_, GpuBuffer>],
        done: Box<dyn FnOnce() + Send>,
    ) -> Result<Submitted, String> {
        self.0.submit(&passes(dispatches), done)
    }

    fn submit_apart(
        &self,
        dispatches: &[Dispatch<'_, GpuBuffer>],
        done: Box<dyn FnOnce() + Send>,
    ) -> Result<Submitted, String> {
        self.0.submit_apart(&passes(dispatches), done)
    }

    fn finished(run: &Submitted) -> Option<Result<(), String>> {
        run.finished()
    }

    fn gpu_time(run: &Submitted) -> Option<Duration> {
        run.gpu_time()
    }
}

/// The platform's passes of `dispatches`.
fn passes<'a>(dispatches: &[Dispatch<'a, GpuBuffer>]) -> Vec<Pass<'a>> {
    dispatches
        .iter()
        .map(|d| Pass {
            kernel: kernel(d.kernel),
            source: d.source,
            destination: d.destination,
            taps: d.taps,
            params: d.params,
            lines: d.lines,
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
