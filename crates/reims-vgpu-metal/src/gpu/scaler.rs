//! The scaler device's kernels on Metal: [`SOURCE`] (`scaler.metal`), compiled at run time with
//! fast math off, on the system's default device, the one the library's GPU device draws on.
//! reims-vgpu's scaler backend works out each dispatch's kernel, parameters and taps from a
//! transform's plan; this makes the buffers it binds, encodes its dispatches in one command
//! buffer of a queue of its own, in order, each after the one before (a serial encoder), and says
//! when the GPU is done. A second queue takes the dispatches that make each kernel's pipeline
//! ahead of a guest's first transform ([`ScalerGpu::submit_apart`]), so that no transform's work
//! waits behind theirs.
//!
//! A guest surface is reached where it is: [`ScalerGpu::surface_buffer`] makes a buffer over the
//! surface's own IOSurface (`newBufferWithIOSurface:`). The buffer's first byte is the surface's
//! first byte, its base address, `kIOSurfaceOffset` into its allocation, and its length is the
//! allocation's size, that offset more than the surface's own bytes, so only the surface's bytes
//! from its first are bound: measured on the M3 Max, macOS 27.0.1, 2026-10-25, with surfaces at
//! offsets 0, 0x40 and 0x4000 into 64 KiB, where a row the kernel writes at a plane offset is the
//! row a lend of the surface reads there. The buffer retains the IOSurface, a hold the guest
//! sees, so it is dropped before the hold its maker has on the surface goes.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::time::Duration;

use block2::RcBlock;
use objc2::rc::autoreleasepool;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_io_surface::IOSurfaceRef;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder as _, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder as _, MTLComputePipelineState, MTLDevice,
    MTLLibrary as _, MTLMathMode, MTLResourceOptions, MTLSize,
};
use reims_vgpu_core::num::ToUsize as _;
use reims_vgpu_core::sync::Mutex;

use super::device::create_or_die;
use crate::msg;
use crate::platform::{MetalPlatform, Obj};
use crate::spi::methods::DeviceSpi;

/// The kernel's Metal Shading Language.
pub const SOURCE: &str = include_str!("scaler.metal");

/// The kernels' names in [`SOURCE`], as [`Kernel`] orders them.
const KERNELS: [&str; 7] = [
    "scaler_lines",
    "scaler_transform",
    "scaler_across",
    "scaler_across_shared",
    "scaler_quads",
    "scaler_transform_adaptive",
    "scaler_transform_both_planes",
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
    /// `scaler_transform_both_planes`: `scaler_transform` for both planes of a 4:2:0 window whose
    /// every edge is even, which the adaptive filter does not move.
    TransformBothPlanes,
}

/// The kernels on the GPU the library draws on, a queue of their own, and one apart.
pub struct ScalerGpu {
    device: Obj<ProtocolObject<dyn MTLDevice>>,
    queue: Obj<ProtocolObject<dyn MTLCommandQueue>>,
    /// The queue of [`ScalerGpu::submit_apart`].
    apart: Obj<ProtocolObject<dyn MTLCommandQueue>>,
    pipelines: [Obj<ProtocolObject<dyn MTLComputePipelineState>>; 7],
    name: String,
}

/// A buffer the kernel binds: one of the GPU's own memory, which the CPU reaches at
/// [`GpuBuffer::contents`], or one over a surface.
pub struct GpuBuffer {
    buffer: Obj<ProtocolObject<dyn MTLBuffer>>,
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

    /// Its bytes as the CPU reaches them: a buffer of the GPU's own memory's; `None` for one over a
    /// surface.
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
    command_buffer: Obj<ProtocolObject<dyn MTLCommandBuffer>>,
}

impl Submitted {
    /// `None` while the GPU runs it; then `Ok`, or why it failed.
    pub fn finished(&self) -> Option<Result<(), String>> {
        let cb = self.command_buffer.get();
        match msg::get("status", || cb.status()) {
            MTLCommandBufferStatus::Completed => Some(Ok(())),
            MTLCommandBufferStatus::Error => Some(Err(msg::get("error", || cb.error())
                .map_or_else(
                    || "the command buffer failed".to_owned(),
                    |e| {
                        msg::string_text(&msg::get("localizedDescription", || {
                            e.localizedDescription()
                        }))
                    },
                ))),
            _ => None,
        }
    }

    /// How long the GPU took, from its start to its end, once it is done.
    pub fn gpu_time(&self) -> Option<Duration> {
        let cb = self.command_buffer.get();
        let (start, end) = (
            msg::get("GPUStartTime", || cb.GPUStartTime()),
            msg::get("GPUEndTime", || cb.GPUEndTime()),
        );
        (end >= start && start > 0.0).then(|| Duration::from_secs_f64(end - start))
    }
}

impl ScalerGpu {
    /// The kernels on the system's default device, compiled now; `Err` with why where there is
    /// no device, or Metal refuses a kernel.
    pub fn new() -> Result<ScalerGpu, String> {
        autoreleasepool(|_| {
            let device = MetalPlatform::system_default_device()
                .ok_or_else(|| "Metal has no default device".to_owned())?;
            let d = device.get();
            let name = msg::get("name", || d.name()).to_string();
            let source = msg::new_string(SOURCE);
            let options = msg::get("new", MTLCompileOptions::new);
            // The reference's arithmetic as it is written: no reassociation, no fused or
            // approximate operations.
            msg::get("setMathMode:", || options.setMathMode(MTLMathMode::Safe));
            let library = create_or_die("newLibraryWithSource:options:error:", |e| {
                // SAFETY: `create_or_die`'s out-parameter.
                unsafe {
                    DeviceSpi::newLibraryWithSource_options_error(d, &source, Some(&options), e)
                }
            })
            .map_err(|e| format!("Metal refused the scaler's kernel: {}", e.description))?
            .ok_or_else(|| "Metal made no library of the scaler's kernel".to_owned())?;
            let pipeline = |kernel: &str| {
                let function = msg::get("newFunctionWithName:", || {
                    library
                        .get()
                        .newFunctionWithName(&NSString::from_str(kernel))
                })
                .ok_or_else(|| format!("the scaler's library has no {kernel}"))?;
                create_or_die("newComputePipelineStateWithFunction:error:", |e| {
                    // SAFETY: `create_or_die`'s out-parameter.
                    unsafe {
                        DeviceSpi::newComputePipelineStateWithFunction_error(d, Some(&function), e)
                    }
                })
                .map_err(|e| format!("Metal refused the scaler's {kernel}: {}", e.description))?
                .ok_or_else(|| format!("Metal made no pipeline of the scaler's {kernel}"))
            };
            let pipelines = [
                pipeline(KERNELS[0])?,
                pipeline(KERNELS[1])?,
                pipeline(KERNELS[2])?,
                pipeline(KERNELS[3])?,
                pipeline(KERNELS[4])?,
                pipeline(KERNELS[5])?,
                pipeline(KERNELS[6])?,
            ];
            let new_queue = || {
                msg::get("newCommandQueue", || d.newCommandQueue())
                    .map(Obj::from_retained)
                    .ok_or_else(|| "Metal made no command queue".to_owned())
            };
            let (queue, apart) = (new_queue()?, new_queue()?);
            Ok(ScalerGpu {
                device,
                queue,
                apart,
                pipelines,
                name,
            })
        })
    }

    /// The GPU's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// A buffer of `length` bytes of the GPU's own memory, shared with the CPU, zeroed.
    pub fn shared_buffer(&self, length: usize) -> Result<GpuBuffer, String> {
        let d = self.device.get();
        let buffer = msg::get("newBufferWithLength:options:", || {
            d.newBufferWithLength_options(length.max(4), MTLResourceOptions::StorageModeShared)
        })
        .map(Obj::from_retained)
        .ok_or_else(|| format!("Metal made no buffer of {length} bytes"))?;
        let contents = NonNull::new(msg::get("contents", || buffer.get().contents()).as_ptr())
            .map(NonNull::cast::<u8>);
        Ok(GpuBuffer {
            buffer,
            contents,
            length,
        })
    }

    /// A buffer of the GPU's own memory holding `bytes`, which the CPU writes once and never
    /// reads: write-combined, as the taps' buffer is.
    pub fn table_buffer(&self, bytes: &[u8]) -> Result<GpuBuffer, String> {
        let d = self.device.get();
        let length = bytes.len();
        let options =
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeWriteCombined;
        let buffer = msg::get("newBufferWithLength:options:", || {
            d.newBufferWithLength_options(length.max(4), options)
        })
        .map(Obj::from_retained)
        .ok_or_else(|| format!("Metal made no buffer of {length} bytes"))?;
        let contents = NonNull::new(msg::get("contents", || buffer.get().contents()).as_ptr())
            .map(NonNull::cast::<u8>)
            .ok_or_else(|| format!("Metal's buffer of {length} bytes has no contents"))?;
        // SAFETY: a shared buffer's contents are its `length.max(4)` bytes, which nothing else
        // reaches yet.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), length) };
        Ok(GpuBuffer {
            buffer,
            contents: None,
            length,
        })
    }

    /// A buffer of `length` bytes of the GPU's own memory, which only the GPU reaches.
    pub fn private_buffer(&self, length: usize) -> Result<GpuBuffer, String> {
        let d = self.device.get();
        let buffer = msg::get("newBufferWithLength:options:", || {
            d.newBufferWithLength_options(length.max(4), MTLResourceOptions::StorageModePrivate)
        })
        .map(Obj::from_retained)
        .ok_or_else(|| format!("Metal made no buffer of {length} bytes"))?;
        Ok(GpuBuffer {
            buffer,
            contents: None,
            length,
        })
    }

    /// A buffer over all of `surface`'s allocation, from its first byte: its planes lie at their
    /// offsets from its base address, plus its `kIOSurfaceOffset`.
    ///
    /// # Safety
    ///
    /// `surface` is a live `IOSurfaceRef`, which its caller may let the GPU read, and write where
    /// it binds the buffer as a destination; the buffer retains it, and is dropped before the
    /// caller's hold on the surface goes.
    pub unsafe fn surface_buffer(&self, surface: NonNull<c_void>) -> Result<GpuBuffer, String> {
        let d = self.device.get();
        // SAFETY: the caller's: a live IOSurfaceRef.
        let surface: &IOSurfaceRef = unsafe { surface.cast().as_ref() };
        let buffer = msg::get("newBufferWithIOSurface:", || {
            d.newBufferWithIOSurface(Some(surface))
        })
        .map(Obj::from_retained)
        .ok_or_else(|| "Metal made no buffer over the surface".to_owned())?;
        let length = msg::get("length", || buffer.get().length());
        Ok(GpuBuffer {
            buffer,
            contents: None,
            length,
        })
    }

    /// Encodes `passes` in one command buffer, in order, and commits it; `done` runs once the GPU
    /// is done with it, on Metal's completion thread.
    pub fn submit(
        &self,
        passes: &[Pass<'_>],
        done: Box<dyn FnOnce() + Send>,
    ) -> Result<Submitted, String> {
        self.submit_on(&self.queue, passes, done)
    }

    /// [`Self::submit`], on a queue apart from the transforms': Metal orders nothing of the
    /// transforms' after its work.
    pub fn submit_apart(
        &self,
        passes: &[Pass<'_>],
        done: Box<dyn FnOnce() + Send>,
    ) -> Result<Submitted, String> {
        self.submit_on(&self.apart, passes, done)
    }

    fn submit_on(
        &self,
        queue: &Obj<ProtocolObject<dyn MTLCommandQueue>>,
        passes: &[Pass<'_>],
        done: Box<dyn FnOnce() + Send>,
    ) -> Result<Submitted, String> {
        autoreleasepool(|_| {
            let q = queue.get();
            let command_buffer = msg::get("commandBuffer", || q.commandBuffer())
                .map(Obj::from_retained)
                .ok_or_else(|| "Metal made no command buffer".to_owned())?;
            let cb = command_buffer.get();
            let encoder = msg::get("computeCommandEncoder", || cb.computeCommandEncoder())
                .ok_or_else(|| "Metal made no compute encoder".to_owned())?;
            let mut current = None;
            for pass in passes {
                if current != Some(pass.kernel) {
                    let pipeline = match pass.kernel {
                        Kernel::Lines => self.pipelines[0].get(),
                        Kernel::Transform => self.pipelines[1].get(),
                        Kernel::Across => self.pipelines[2].get(),
                        Kernel::Shared => self.pipelines[3].get(),
                        Kernel::Quads => self.pipelines[4].get(),
                        Kernel::TransformAdaptive => self.pipelines[5].get(),
                        Kernel::TransformBothPlanes => self.pipelines[6].get(),
                    };
                    msg::get("setComputePipelineState:", || {
                        encoder.setComputePipelineState(pipeline);
                    });
                    current = Some(pass.kernel);
                }
                let buffers = [pass.source, pass.destination, pass.taps];
                let buffers = buffers
                    .into_iter()
                    .enumerate()
                    .chain([(4, pass.lines), (5, pass.lines)]);
                for (index, buffer) in buffers {
                    msg::get("setBuffer:offset:atIndex:", || {
                        // SAFETY: a live buffer at an index the kernel declares.
                        unsafe {
                            encoder.setBuffer_offset_atIndex(Some(buffer.buffer.get()), 0, index);
                        }
                    });
                }
                let Some(bytes) = NonNull::new(pass.params.as_ptr().cast_mut().cast::<c_void>())
                else {
                    continue;
                };
                msg::get("setBytes:length:atIndex:", || {
                    // SAFETY: `params` holds its bytes for the call, which copies them.
                    unsafe { encoder.setBytes_length_atIndex(bytes, pass.params.len(), 3) };
                });
                let size = pass.group;
                let groups = MTLSize {
                    width: pass.threads.0.div_ceil(size.0).to_usize(),
                    height: pass.threads.1.div_ceil(size.1).to_usize(),
                    depth: 1,
                };
                let group = MTLSize {
                    width: size.0.to_usize(),
                    height: size.1.to_usize(),
                    depth: 1,
                };
                if groups.width > 0 && groups.height > 0 {
                    msg::get("dispatchThreadgroups:threadsPerThreadgroup:", || {
                        encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, group);
                    });
                }
            }
            msg::get("endEncoding", || encoder.endEncoding());
            let once = Mutex::new(Some(done));
            let handler = RcBlock::new(move |_: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                // A foreign callout: marked ours, so a panic here aborts before it unwinds into
                // Metal.
                let _env = reims_vgpu_core::rt::BlockEnv::foreign();
                if let Some(done) = once.lock().take() {
                    done();
                }
            });
            let handler = RcBlock::as_ptr(&handler);
            // SAFETY: a live block, which Metal copies.
            msg::get("addCompletedHandler:", || unsafe {
                cb.addCompletedHandler(handler)
            });
            msg::get("commit", || cb.commit());
            Ok(Submitted { command_buffer })
        })
    }
}

#[cfg(test)]
mod tests {

    use reims_vgpu_core::platform::os::surface::{IoSurfaces as _, SurfaceSpec};

    use super::*;
    use crate::os::iosurface::MetalSurfaces;

    /// `ScalerParams` (`scaler.metal`) for a copy of one row of `pixels` BGRA pixels into a
    /// plane at `offset`, word by word.
    fn copy_row(pixels: u32, offset: u32) -> Vec<u8> {
        let mut words = [0u32; 92];
        let one = 1.0f32.to_bits();
        words[0] = 2; // source_packing: RGB
        words[1] = 1; // source_sample_bytes
        words[2..6].copy_from_slice(&[2, 1, 0, 3]); // source_order
        words[7] = pixels * 4; // source_luma_pitch
        words[9] = pixels * 4; // source_chroma_pitch
        words[11] = one; // source_luma_scale
        words[13] = one; // source_chroma_scale
        words[14..20].copy_from_slice(&[1, 0, 0, 0, 1, 0]); // luma_map
        words[20..24].copy_from_slice(&[pixels - 1, 0, 1, 1]); // last pixel, subsampling
        words[24..30].copy_from_slice(&[1, 1, 0, 0, pixels - 1, 0]); // chroma steps, first and last
        words[30] = 1; // rgb
        words[40..44].copy_from_slice(&[0, 1, pixels, 1]); // the taps: one a pixel
        words[44..48].copy_from_slice(&[0, 0, pixels, 1]); // window
        words[50..52].copy_from_slice(&[1, 1]); // subsampling
        words[52] = 5; // plane_kind: RGB32
        words[53..57].copy_from_slice(&[2, 1, 0, 3]); // plane_order
        words[57] = offset; // plane_offset
        words[58] = pixels * 4; // plane_pitch
        words[60] = 1; // rows
        words[62] = pixels * 4; // end_byte
        words[63] = pixels; // words
        words[68] = 255.0f32.to_bits(); // code_max
        words[75..79].copy_from_slice(&[0, 1, pixels, 1]); // the chroma's taps: the luma's
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// The kernel compiles on this Mac's GPU, and a buffer over a surface made at an offset into
    /// its allocation starts at the surface's first byte and is the allocation's size: a row the
    /// kernel writes at the buffer's first byte is the first a lend of the surface reads. A
    /// buffer over a surface has no CPU address (`contents` answers NULL)
    #[test]
    fn a_buffer_over_a_surface_starts_at_the_surfaces_first_byte() {
        let gpu = ScalerGpu::new().unwrap();
        assert!(!gpu.name().is_empty());
        let row: Vec<u8> = (1..=16).collect();
        let mut source = gpu.shared_buffer(16).unwrap();
        source.contents().unwrap().copy_from_slice(&row);
        let mut taps = gpu.shared_buffer(40).unwrap();
        let table: Vec<u8> = (0u32..4)
            .chain([0])
            .flat_map(|i| i.to_le_bytes().into_iter().chain(1.0f32.to_le_bytes()))
            .collect();
        taps.contents().unwrap().copy_from_slice(&table);
        for offset in [0u32, 0x40, 0x4000] {
            let spec = SurfaceSpec {
                pixel_format: u32::from_be_bytes(*b"BGRA"),
                alloc_size: 0x10000,
                offset: u64::from(offset),
                width: 64,
                height: (0x10000 - u64::from(offset)) / 256,
                bytes_per_element: 4,
                bytes_per_row: 256,
                element_width: 1,
                element_height: 1,
                ..SurfaceSpec::default()
            };
            let surface = MetalSurfaces::new().create(&spec).unwrap();
            let at = NonNull::from(surface.as_iosurface_ref()).cast();
            // SAFETY: a live surface, which outlives the buffer.
            let buffer = unsafe { gpu.surface_buffer(at) }.unwrap();
            assert_eq!(buffer.length(), 0x10000, "offset {offset:#x}");
            assert!(buffer.contents.is_none());
            let params = copy_row(4, 0);
            let submitted = gpu
                .submit(
                    &[Pass {
                        kernel: Kernel::Transform,
                        source: &source,
                        destination: &buffer,
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
            let done = (0..10_000)
                .find_map(|_| {
                    let done = submitted.finished();
                    if done.is_none() {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    done
                })
                .expect("the GPU never finished");
            done.unwrap();
            assert!(submitted.gpu_time().is_some());
            drop((submitted, buffer));
            let mut first = Vec::new();
            crate::os::surface_bytes::lend_surface_bytes_for_reading(&surface, &mut |bytes| {
                assert_eq!(
                    bytes.len(),
                    0x10000 - offset.to_usize(),
                    "offset {offset:#x}"
                );
                first.extend_from_slice(&bytes[..16]);
            })
            .unwrap();
            assert_eq!(first, row, "offset {offset:#x}");
        }
    }
}
