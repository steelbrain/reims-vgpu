//! The scaler device's kernels on the Vulkan platform: the metal crate's `scaler.metal`, which
//! `build.rs` compiled with fast math off into `$OUT_DIR/scaler.metallib`, made on the runtime's
//! system default device, the one the library's GPU device draws on (the loader's pin included,
//! `crate::host_gpus` on Linux). reims-vgpu's scaler backend works out each dispatch's kernel,
//! parameters and taps from a transform's plan; this makes the buffers it binds, encodes its
//! dispatches in one command buffer of a queue of its own, in order, each after the one before (a
//! serial encoder, which the runtime orders with the barriers their buffers need), and says when
//! the GPU is done. A second queue takes the dispatches that make each kernel's pipeline ahead of
//! a guest's first transform ([`ScalerGpu::submit_apart`]): the runtime holds a command buffer's
//! batch in its queue's order until the pipelines its dispatches name are made, and orders a batch
//! of another queue after it only where the two share a buffer one of them writes, so no
//! transform's work waits behind theirs.
//!
//! - **Its own buffers** lie outside the Metal address space ([`crate::gpu::own_buffer`]), so no
//!   guest buffer's `gpuAddress` depends on a transform.
//! - **A guest surface is reached where it is** ([`ScalerGpu::memory_buffer`]): a buffer over
//!   the host's view of its pages, imported as the GPU device imports guest memory
//!   ([`crate::os::host_memory`]): the pointer, or where the device refuses it (RADV refuses a
//!   page with a file behind it) dma-bufs of the same pages at stock `udmabuf` limits, several
//!   bound into one sparse buffer where the pages need more than one.

use std::ptr::NonNull;
use std::time::Duration;

use metal2vulkan_runtime as rt;
use reims_vgpu_core::num::ToU64;

use crate::platform::device_options;

/// The kernel's metallib, which `build.rs` compiled from the metal crate's `scaler.metal`.
pub(crate) const METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/scaler.metallib"));

/// The kernels' names in it, as [`Kernel`] orders them.
pub(crate) const KERNELS: [&str; 6] = [
    "scaler_lines",
    "scaler_transform",
    "scaler_across",
    "scaler_across_shared",
    "scaler_quads",
    "scaler_transform_adaptive",
];

/// The kernel a dispatch runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// `scaler_lines`: columns of the turned source down a band of the window, into the lines
    /// buffer.
    Lines,
    /// `scaler_transform`: a plane of the destination, of a transform that is not separable.
    Transform,
    /// `scaler_across`: a plane of the destination, from the lines buffer.
    Across,
    /// `scaler_across_shared`: the same, its pixels worked out into threadgroup memory first.
    Shared,
    /// `scaler_quads`: the RGB plane of a copy of a Y'CbCr source, a thread two by two pixels, in
    /// whole threadgroups.
    Quads,
    /// `scaler_transform_adaptive`: `scaler_transform` for a transform whose passes the adaptive
    /// filter moves.
    TransformAdaptive,
}

/// The kernels on the GPU the library draws on, a queue of their own, and one apart.
pub struct ScalerGpu {
    device: rt::Device,
    queue: rt::CommandQueue,
    /// The queue of [`ScalerGpu::submit_apart`].
    apart: rt::CommandQueue,
    pipelines: [rt::ComputePipelineState; 6],
    name: String,
}

/// A buffer the kernel binds: one of the GPU's own memory, which the CPU reaches at
/// [`GpuBuffer::contents`], or one over a surface's pages.
pub struct GpuBuffer {
    buffer: rt::Buffer,
    contents: Option<NonNull<u8>>,
    length: usize,
}

// SAFETY: the contents pointer is the buffer's, which lives as long as this; the CPU writes it
// only through `&mut self`, and reads it only while no GPU work that writes it runs.
unsafe impl Send for GpuBuffer {}
// SAFETY: as above.
unsafe impl Sync for GpuBuffer {}

impl GpuBuffer {
    /// Its bytes.
    pub fn length(&self) -> usize {
        self.length
    }

    /// Its bytes as the CPU reaches them: a buffer of the GPU's own memory's; `None` for one over
    /// a surface.
    pub fn contents(&mut self) -> Option<&mut [u8]> {
        // SAFETY: the buffer's own memory, `length` bytes, which nothing else writes while the
        // caller holds `&mut self` and no work of the kernel's on it runs.
        self.contents
            .map(|at| unsafe { std::slice::from_raw_parts_mut(at.as_ptr(), self.length) })
    }
}

/// One dispatch: its kernel, its buffers, its parameters' bytes (`ScalerParams`) and its
/// threads.
pub struct Pass<'a> {
    /// Its kernel.
    pub kernel: Kernel,
    /// Bound at 0: the source.
    pub source: &'a GpuBuffer,
    /// Bound at 1: the destination.
    pub destination: &'a GpuBuffer,
    /// Bound at 2: the taps.
    pub taps: &'a GpuBuffer,
    /// Bound at 3, as bytes.
    pub params: &'a [u8],
    /// Bound at 4: the lines.
    pub lines: &'a GpuBuffer,
    /// Its threads across and down, rounded up to whole threadgroups.
    pub threads: (u32, u32),
    /// The threads of a threadgroup across and down.
    pub group: (u32, u32),
}

/// A command buffer committed: its dispatches on the GPU.
pub struct Submitted {
    command_buffer: rt::CommandBuffer,
}

impl Submitted {
    /// `None` while the GPU runs it; then `Ok`, or why it failed.
    pub fn finished(&self) -> Option<Result<(), String>> {
        match self.command_buffer.status() {
            rt::CommandBufferStatus::Completed => Some(Ok(())),
            rt::CommandBufferStatus::Error => Some(Err(self.command_buffer.error().map_or_else(
                || "the command buffer failed".to_owned(),
                |e| format!("{e:?}"),
            ))),
            _ => None,
        }
    }

    /// How long the GPU took, from its start to its end, once it is done: as the runtime times
    /// a command buffer on a device made with `gpu_times_from_host`, from when it submitted the
    /// work to when it saw it complete.
    pub fn gpu_time(&self) -> Option<Duration> {
        let start = self.command_buffer.gpu_start_time().ok()?;
        let end = self.command_buffer.gpu_end_time().ok()?;
        (end >= start && start > 0.0).then(|| Duration::from_secs_f64(end - start))
    }
}

fn refused(what: &str) -> impl FnOnce(rt::Error) -> String + '_ {
    move |e| format!("the runtime refused {what}: {e}")
}

impl ScalerGpu {
    /// The kernels on the runtime's system default device; `Err` with why where there is none,
    /// or the runtime refuses a kernel.
    pub fn new() -> Result<ScalerGpu, String> {
        let device = rt::Device::system_default_with(device_options())
            .map_err(refused("the system default device"))?;
        let library = device
            .new_library_with_data(METALLIB)
            .map_err(refused("the scaler's library"))?;
        let pipeline = |kernel: &str| {
            let function = library
                .new_function_with_name(kernel)
                .map_err(refused("a scaler's kernel"))?;
            device
                .new_compute_pipeline_state_with_function(
                    &function,
                    &rt::PipelineOptions::default(),
                )
                .map_err(refused("a scaler's pipeline"))
        };
        let pipelines = [
            pipeline(KERNELS[0])?,
            pipeline(KERNELS[1])?,
            pipeline(KERNELS[2])?,
            pipeline(KERNELS[3])?,
            pipeline(KERNELS[4])?,
            pipeline(KERNELS[5])?,
        ];
        // Shared: the scaler's work is a guest app's video work, which shares a GPU queue as an
        // app's own queues do.
        let new_queue = || {
            device
                .new_command_queue()
                .map_err(refused("a command queue"))
        };
        let (queue, apart) = (new_queue()?, new_queue()?);
        let name = device.name().to_owned();
        Ok(ScalerGpu {
            device,
            queue,
            apart,
            pipelines,
            name,
        })
    }

    /// The GPU's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// A buffer of `length` bytes of the GPU's own memory, shared with the CPU, zeroed.
    pub fn shared_buffer(&self, length: usize) -> Result<GpuBuffer, String> {
        let buffer = crate::gpu::own_buffer(
            &self.device,
            length.max(4).to_u64(),
            rt::ResourceOptions::shared(),
            rt::BufferContents::Zero,
        )
        .map_err(refused("a buffer"))?;
        let contents = buffer.contents();
        Ok(GpuBuffer {
            buffer,
            contents,
            length,
        })
    }

    /// A buffer of the GPU's own memory holding `bytes`, which the CPU writes once and never
    /// reads: write-combined, as the taps' buffer is, so that a GPU without unified memory reads
    /// it from its own memory where the device has host-visible memory there, not across the bus.
    pub fn table_buffer(&self, bytes: &[u8]) -> Result<GpuBuffer, String> {
        let options = rt::ResourceOptions::new(
            rt::StorageMode::Shared,
            rt::CpuCacheMode::WriteCombined,
            rt::HazardTrackingMode::Default,
        );
        let buffer = crate::gpu::own_buffer(
            &self.device,
            bytes.len().max(4).to_u64(),
            options,
            rt::BufferContents::Copy(bytes),
        )
        .map_err(refused("a buffer"))?;
        Ok(GpuBuffer {
            buffer,
            contents: None,
            length: bytes.len(),
        })
    }

    /// A buffer of `length` bytes of the GPU's own memory, which only the GPU reaches.
    pub fn private_buffer(&self, length: usize) -> Result<GpuBuffer, String> {
        let buffer = crate::gpu::own_buffer(
            &self.device,
            length.max(4).to_u64(),
            rt::ResourceOptions::private(),
            rt::BufferContents::Zero,
        )
        .map_err(refused("a buffer"))?;
        Ok(GpuBuffer {
            buffer,
            contents: None,
            length,
        })
    }

    /// A buffer over the `length` bytes at `first`, imported as they are, kept mapped by `keep`
    /// until the runtime lets go of its import.
    ///
    /// # Safety
    ///
    /// The bytes are a whole number of pages the process maps for reading and writing, which
    /// `keep` keeps mapped, with the same backing, until it is dropped; the GPU may read them, and
    /// write them where the caller binds the buffer as a destination.
    pub unsafe fn memory_buffer(
        &self,
        first: NonNull<u8>,
        length: usize,
        keep: Box<dyn Send + 'static>,
    ) -> Result<GpuBuffer, String> {
        // SAFETY: the caller's: mapped, writable pages, kept mapped by `keep`.
        let memory = unsafe { crate::os::host_memory(first, true) };
        let contents = rt::BufferContents::NoCopy {
            memory,
            deallocator: None,
            keep_alive: Some(keep),
        };
        let descriptor =
            rt::BufferDescriptor::new(length.to_u64(), rt::ResourceOptions::shared(), contents)
                .with_placement(rt::AddressPlacement::Outside);
        let buffer = self
            .device
            .new_buffer_with_descriptor(descriptor)
            .map_err(|e| {
                let why = crate::os::dma_buf_refusal().unwrap_or_default();
                format!("the runtime refused a buffer over the surface's pages: {e} {why}")
            })?;
        Ok(GpuBuffer {
            buffer,
            contents: None,
            length,
        })
    }

    /// Encodes `passes` in one command buffer, in order, and commits it; `done` runs once the GPU
    /// is done with it, on one of the runtime's handler threads.
    pub fn submit(
        &self,
        passes: &[Pass<'_>],
        done: Box<dyn FnOnce() + Send>,
    ) -> Result<Submitted, String> {
        self.submit_on(&self.queue, passes, done)
    }

    /// [`Self::submit`], on a queue apart from the transforms' (module docs).
    pub fn submit_apart(
        &self,
        passes: &[Pass<'_>],
        done: Box<dyn FnOnce() + Send>,
    ) -> Result<Submitted, String> {
        self.submit_on(&self.apart, passes, done)
    }

    /// What the runtime counted of its device so far, of every state on it, the scaler's and the
    /// GPU device's alike, for the probes of a first transform: the functions it translated, and
    /// those it found translated (in its memo, an archive or the shader cache on disk); the
    /// `VkPipeline`s it made; and of those, the ones the driver says it found in the cache the
    /// making was handed, and the ones it says it did not. With the shader cache's directory.
    pub fn cache_counts(&self) -> ([u64; 5], std::path::PathBuf) {
        let work = self.device.work_counts();
        let cache = self.device.shader_cache_statistics();
        (
            [
                work.functions_translated,
                work.functions_shared,
                work.variant_pipelines_made,
                cache.driver_pipeline_hits,
                cache.driver_pipeline_misses,
            ],
            self.device.shader_cache_directory().to_owned(),
        )
    }

    fn submit_on(
        &self,
        queue: &rt::CommandQueue,
        passes: &[Pass<'_>],
        done: Box<dyn FnOnce() + Send>,
    ) -> Result<Submitted, String> {
        let command_buffer = queue
            .command_buffer()
            .map_err(refused("a command buffer"))?;
        let mut encoder = command_buffer
            .compute_command_encoder()
            .map_err(refused("a compute encoder"))?;
        let mut current = None;
        for pass in passes {
            if current != Some(pass.kernel) {
                let pipeline = match pass.kernel {
                    Kernel::Lines => &self.pipelines[0],
                    Kernel::Transform => &self.pipelines[1],
                    Kernel::Across => &self.pipelines[2],
                    Kernel::Shared => &self.pipelines[3],
                    Kernel::Quads => &self.pipelines[4],
                    Kernel::TransformAdaptive => &self.pipelines[5],
                };
                encoder
                    .set_compute_pipeline_state(pipeline)
                    .map_err(refused("a scaler's pipeline"))?;
                current = Some(pass.kernel);
            }
            let buffers = [pass.source, pass.destination, pass.taps];
            let buffers = (0u32..)
                .zip(buffers)
                .chain([(4, pass.lines), (5, pass.lines)]);
            for (index, buffer) in buffers {
                encoder
                    .set_buffer(Some(&buffer.buffer), 0, index)
                    .map_err(refused("a buffer's binding"))?;
            }
            encoder
                .set_bytes(pass.params, 3)
                .map_err(refused("the parameters"))?;
            let size = pass.group;
            let groups = rt::Size {
                width: u64::from(pass.threads.0.div_ceil(size.0)),
                height: u64::from(pass.threads.1.div_ceil(size.1)),
                depth: 1,
            };
            let group = rt::Size {
                width: u64::from(size.0),
                height: u64::from(size.1),
                depth: 1,
            };
            if groups.width > 0 && groups.height > 0 {
                encoder
                    .dispatch_threadgroups(groups, group)
                    .map_err(refused("a dispatch"))?;
            }
        }
        encoder
            .end_encoding()
            .map_err(refused("the encoder's end"))?;
        command_buffer
            .add_completed_handler(Box::new(move |_| done()))
            .map_err(refused("a completed handler"))?;
        command_buffer.commit().map_err(refused("the commit"))?;
        Ok(Submitted { command_buffer })
    }
}

impl std::fmt::Debug for ScalerGpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScalerGpu")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

/// The page size of a host whose pages the tests import.
#[cfg(test)]
fn page() -> usize {
    use reims_vgpu_core::num::ToUsize as _;
    crate::os::PAGE.to_usize()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The kernel loads on the runtime's default device, and writes a word through a buffer over
    /// host pages: a 4x1 BGRA copy, whose one dispatch writes its row of four pixels
    #[test]
    fn the_kernel_copies_a_row_into_imported_pages() {
        let Ok(gpu) = ScalerGpu::new() else {
            eprintln!("no Vulkan device: skipped");
            return;
        };
        assert!(!gpu.name().is_empty());
        let page = page();
        let layout = std::alloc::Layout::from_size_align(page, page).unwrap();
        // SAFETY: a non-zero size.
        let at = NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) }).unwrap();
        struct Freed(NonNull<u8>, std::alloc::Layout);
        // SAFETY: the pointer is the allocation's, freed once.
        unsafe impl Send for Freed {}
        impl Drop for Freed {
            fn drop(&mut self) {
                // SAFETY: allocated above with this layout.
                unsafe { std::alloc::dealloc(self.0.as_ptr(), self.1) };
            }
        }
        let mut source = gpu.shared_buffer(16).unwrap();
        source
            .contents()
            .unwrap()
            .copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        // SAFETY: a page of the process's own, writable, freed only when the import lets go.
        let destination =
            unsafe { gpu.memory_buffer(at, page, Box::new(Freed(at, layout))) }.unwrap();
        let mut taps = gpu.shared_buffer(64).unwrap();
        let table: Vec<u8> = (0u32..4)
            .flat_map(|i| i.to_le_bytes().into_iter().chain(1.0f32.to_le_bytes()))
            .chain(0u32.to_le_bytes().into_iter().chain(1.0f32.to_le_bytes()))
            .collect();
        taps.contents().unwrap()[..table.len()].copy_from_slice(&table);
        // ScalerParams for a 4x1 BGRA copy (gpu/scaler.metal), word by word.
        let mut words = [0u32; 92];
        let one = 1.0f32.to_bits();
        words[0] = 2; // source_packing: RGB
        words[1] = 1; // source_sample_bytes
        words[2..6].copy_from_slice(&[2, 1, 0, 3]); // source_order
        words[7] = 16; // source_luma_pitch
        words[9] = 16; // source_chroma_pitch
        words[11] = one; // source_luma_scale
        words[13] = one; // source_chroma_scale
        words[14..20].copy_from_slice(&[1, 0, 0, 0, 1, 0]); // luma_map
        words[20..24].copy_from_slice(&[3, 0, 1, 1]); // last pixel, subsampling
        words[24..30].copy_from_slice(&[1, 1, 0, 0, 3, 0]); // chroma steps, first and last
        words[30] = 1; // rgb
        words[40..44].copy_from_slice(&[0, 1, 4, 1]); // across first, count, down first, count
        words[44..48].copy_from_slice(&[0, 0, 4, 1]); // window
        words[50..52].copy_from_slice(&[1, 1]); // subsampling
        words[52] = 5; // plane_kind: RGB32
        words[53..57].copy_from_slice(&[2, 1, 0, 3]); // plane_order
        words[58] = 16; // plane_pitch
        words[60] = 1; // rows
        words[62] = 16; // end_byte
        words[63] = 4; // words
        words[68] = 255.0f32.to_bits(); // code_max
        words[75..79].copy_from_slice(&[0, 1, 4, 1]); // the chroma's taps: the luma's
        let params: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let submitted = gpu
            .submit(
                &[Pass {
                    kernel: Kernel::Transform,
                    source: &source,
                    destination: &destination,
                    taps: &taps,
                    params: &params,
                    lines: &taps,
                    threads: (4, 1),
                    group: (16, 4),
                }],
                Box::new(|| {}),
            )
            .unwrap();
        // 10 s of 1 ms pauses at most.
        let finished = (0..10_000)
            .find_map(|_| {
                let done = submitted.finished();
                if done.is_none() {
                    std::thread::sleep(Duration::from_millis(1));
                }
                done
            })
            .expect("the GPU never finished");
        finished.unwrap();
        drop(submitted);
        // SAFETY: the page is the import's, which the GPU wrote and is done with.
        let written = unsafe { std::slice::from_raw_parts(at.as_ptr(), 16) };
        assert_eq!(
            written,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
        drop((destination, source, taps));
    }
}
