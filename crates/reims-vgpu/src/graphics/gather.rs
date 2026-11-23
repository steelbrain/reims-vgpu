//! A 4:4:4 picture written by the runtime device's GPU into the two planes of a surface the guest
//! lent, on Linux ([`Gatherer`]): VideoToolbox's paravirtual decoders' pictures of HEVC's 4:4:4
//! at 8 bits, which NVDEC decodes into three planes, luma, Cb and Cr (`super::nvidia`), and
//! VA-API's drivers into XYUV surfaces of packed pixels (`super::vaapi::xyuv`), land in a surface
//! of CoreVideo's 4:4:4 formats ('444v', '444f'), whose second plane holds each pixel's Cb then
//! Cr.
//!
//! - **The source** is a buffer of the device's that holds the picture's samples, each component
//!   in rows at one pitch, a step of bytes from one sample to the next ([`Planar`]): NVDEC's
//!   decoder's scratch, which CUDA wrote, three planes of a byte a sample; or a VA-API decoder's,
//!   which the driver's video processing wrote, one plane of XYUV's four bytes a pixel.
//! - **The destination** is a Shared buffer of the device's over the surface's pages, the
//!   surface's first byte its first, which VideoToolbox's decoder makes for a 4:4:4 surface and
//!   keeps with its view of the pages (`super::vt`).
//! - **The pass** is the platform library's `videoGather444` kernel (`shaders.metal`) on a queue
//!   of the gatherer's own: one dispatch a plane, a thread a 32-bit word of the plane's rows, each
//!   of its bytes the sample of its pixel and component, read by 32-bit loads. A row's last word
//!   that holds 1 to 3 of its bytes keeps the surface's other bytes. The pass writes what the
//!   CPU's interleave of the picture read back writes (`nvidia`'s `interleave_planes`): each
//!   plane's rows as far as both the picture's crop and the plane hold them, the chroma's in
//!   whole Cb/Cr pairs, and nothing else of the surface. Then it is waited for.
//! - **Refusals** ([`gathers`]) come before anything is submitted, and the picture is then copied
//!   by the CPU: a picture past 8 bits, a plane off a 32-bit word, a plane whose rows lie past the
//!   surface or the source, a value past 32 bits; and the device's refusal of the kernel's
//!   pipeline, a queue, a command buffer or a buffer of its arguments.
//!
//! Each picture written so is counted ([`GATHERED_CODE`]): a GPU pass a picture that a Mac,
//! whose decoder writes the guest's surface, does not make.

use std::sync::{Arc, OnceLock};

use reims_vgpu_video::core::device_memory::TargetPlane;
use reims_vgpu_vulkan::core::platform::Handle as _;
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::blit::{BlitCmd, BlitEncoder as _};
use reims_vgpu_vulkan::core::platform::command::{CommandBuffer as _, CommandQueue as _};
use reims_vgpu_vulkan::core::platform::compute::{ComputeCmd, ComputeEncoder as _};
use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
use reims_vgpu_vulkan::core::platform::resource::{Buffer as _, Library as _};
use reims_vgpu_vulkan::core::platform::types::{CommandBufferStatus, MtlSize, NsRange};
use reims_vgpu_vulkan::core::raise::{OnFault, Raise};

/// The platform library's kernel that writes a plane of the surface (`shaders.metal`).
const GATHER_KERNEL: &str = "videoGather444";

/// The code of a 4:4:4 picture the GPU wrote into a surface's two planes (the module doc),
/// counted per picture by [`Gatherer::gather`].
pub(crate) const GATHERED_CODE: &str = "video-pictures-gpu-gathered";

/// The threads of a dispatch's threadgroup, along a row.
const GROUP_WIDTH: u64 = 64;

/// The bytes between the chroma plane's arguments and the luma's in the arguments' buffer: a
/// constant buffer's binding offset is a multiple of 256 on any device.
const ARGUMENTS_APART: u64 = 256;

/// What `videoGather444` takes of a plane (`shaders.metal`'s `VideoGatherPlane`): in the source,
/// the first byte of each component the plane takes in the picture's first row (luma's twice, or
/// Cb's and Cr's), the bytes from one of the picture's rows to the next and from one sample of a
/// component to the next, and the components a pixel of the plane holds; then the plane's bytes
/// written a row, its rows written, its words a row written, its first word in the destination,
/// and its words from one row to the next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GatherPlane {
    first: u32,
    second: u32,
    source_pitch: u32,
    step: u32,
    components: u32,
    row_bytes: u32,
    rows: u32,
    row_words: u32,
    out_first: u32,
    pitch_words: u32,
}

impl GatherPlane {
    /// Its bytes, as the kernel reads them.
    fn bytes(&self) -> [u8; 40] {
        let words = [
            self.first,
            self.second,
            self.source_pitch,
            self.step,
            self.components,
            self.row_bytes,
            self.rows,
            self.row_words,
            self.out_first,
            self.pitch_words,
        ];
        let mut bytes = [0; 40];
        for (to, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            *to = word.to_le_bytes();
        }
        bytes
    }

    /// The threadgroups of its dispatch: a thread a word of the plane's rows written.
    fn threadgroups(&self) -> MtlSize {
        let across = u64::from(self.row_words).div_ceil(GROUP_WIDTH);
        MtlSize::new(across, u64::from(self.rows), 1)
    }
}

/// Where a 4:4:4 picture's three components lie in its source buffer: the first byte of each in
/// the first row, luma, Cb and Cr, the bytes from one row to the next, and from one sample of a
/// component to the next, the same for the three, and the buffer's bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Planar {
    pub(crate) planes: [u64; 3],
    pub(crate) pitch: u64,
    pub(crate) step: u64,
    pub(crate) length: u64,
}

impl Planar {
    /// An XYUV picture's components in a buffer of `length` bytes, its one plane of rows `pitch`
    /// apart from the buffer's first byte: each pixel four bytes, Cr, Cb, Y and one that holds
    /// nothing, in that order (the VA crate's `PackedTarget`).
    pub(crate) fn xyuv(pitch: u64, length: u64) -> Self {
        Self {
            planes: [2, 1, 0],
            pitch,
            step: 4,
            length,
        }
    }
}

/// The dispatches that write a 4:4:4 picture of 8-bit samples, `size` its crop's samples as the
/// source holds them, from `source` into `planes`, a surface's luma and chroma planes, the
/// destination's bytes from the surface's first, of which it has `surface` (module doc): the
/// rows and bytes the CPU's interleave writes. `Err` says why none is made.
pub(crate) fn gathers(
    source: Planar,
    (width, height): (u32, u32),
    planes: &[TargetPlane; 2],
    surface: u64,
) -> Result<[GatherPlane; 2], String> {
    let narrow = |v: u64, what: &str| u32::try_from(v).map_err(|_| format!("{what} of {v}"));
    let wide = |v: usize| u64::try_from(v).unwrap_or(u64::MAX);
    let (width, height) = (u64::from(width), u64::from(height));
    // A plane's dispatch: `components` samples a pixel from the source planes `from`, into
    // `plane`.
    let plane = |plane: &TargetPlane, components: u64, from: [usize; 2], which: &str| {
        let rows = height.min(u64::from(plane.rows));
        let row_bytes = (width * components).min(wide(plane.row_bytes));
        let row_bytes = row_bytes - row_bytes % components;
        let (offset, bytes_per_row) = (wide(plane.offset), wide(plane.bytes_per_row));
        if rows == 0 || row_bytes == 0 {
            return Err(format!("an empty {which} plane"));
        }
        if !offset.is_multiple_of(4) || !bytes_per_row.is_multiple_of(4) {
            return Err(format!(
                "a {which} plane at byte {offset} of rows {bytes_per_row} apart: not on words"
            ));
        }
        // The kernel writes whole words: a row's last, which may hold bytes past the row, lies
        // in the destination.
        let end = (rows - 1).checked_mul(bytes_per_row).and_then(|last| {
            (offset.checked_add(last)?).checked_add(row_bytes.next_multiple_of(4))
        });
        if !end.is_some_and(|end| end <= surface) {
            return Err(format!(
                "a {which} plane of {rows} rows of {row_bytes} bytes, {bytes_per_row} apart, from \
                 byte {offset}, past the surface's {surface}"
            ));
        }
        // And reads whole words: the word that holds the last row's last sample of each source
        // component lies in the source.
        let samples = row_bytes / components;
        let span = (samples - 1)
            .checked_mul(source.step)
            .and_then(|s| s.checked_add(1));
        let reach = (from.into_iter().map(|at| source.planes[at]).max())
            .and_then(|first| first.checked_add((rows - 1).checked_mul(source.pitch)?))
            .and_then(|last| last.checked_add(span?))
            .map(|end| end.next_multiple_of(4));
        let row_fits = samples
            .checked_mul(source.step)
            .is_some_and(|row| row <= source.pitch);
        if !reach.is_some_and(|reach| reach <= source.length) || !row_fits || source.step == 0 {
            return Err(format!(
                "a {which} plane whose rows of {samples} samples, {} apart, lie past its source's \
                 {} bytes",
                source.pitch, source.length
            ));
        }
        narrow(reach.unwrap_or(u64::MAX), "a source's reach")?;
        Ok(GatherPlane {
            first: narrow(source.planes[from[0]], "a first byte")?,
            second: narrow(source.planes[from[1]], "a first byte")?,
            source_pitch: narrow(source.pitch, "a pitch")?,
            step: narrow(source.step, "a step")?,
            components: narrow(components, "components")?,
            row_bytes: narrow(row_bytes, "a row's bytes")?,
            rows: narrow(rows, "rows")?,
            row_words: narrow(row_bytes.div_ceil(4), "a row's words")?,
            out_first: narrow(offset / 4, "a first word")?,
            pitch_words: narrow(bytes_per_row / 4, "a pitch's words")?,
        })
    };
    Ok([
        plane(&planes[0], 1, [0, 0], "luma")?,
        plane(&planes[1], 2, [1, 2], "chroma")?,
    ])
}

/// The kernel's pipeline on a device, made at its first gather and kept for the device's
/// decoders.
pub(crate) struct GatherPipeline<P: Platform> {
    device: P::Device,
    made: OnceLock<Result<P::ComputePipeline, String>>,
}

impl<P: Platform> GatherPipeline<P> {
    /// The kernel's pipeline on `device`, made at its first use.
    pub(crate) fn new(device: &P::Device) -> Self {
        Self {
            device: device.clone_ref(),
            made: OnceLock::new(),
        }
    }

    /// The pipeline; `Err` says why the device makes none.
    fn get(&self) -> Result<P::ComputePipeline, String> {
        let made = self.made.get_or_init(|| {
            let library = match self.device.new_shader_library() {
                Ok(Some(library)) => library,
                Ok(None) => return Err("the platform's library is not made".to_owned()),
                Err(error) => return Err(format!("the platform's library: {error}")),
            };
            let function = match library.new_function(OnFault::Return, GATHER_KERNEL) {
                Ok(Some(function)) => function,
                Ok(None) => {
                    return Err(format!("the platform's library holds no {GATHER_KERNEL}"));
                }
                Err(raise) => return Err(format!("{GATHER_KERNEL}: {raise}")),
            };
            match self
                .device
                .new_compute_pipeline_state_for_function(Some(&function))
            {
                Ok(Some(pipeline)) => Ok(pipeline),
                Ok(None) => Err(format!("{GATHER_KERNEL}'s pipeline is refused")),
                Err(error) => Err(format!("{GATHER_KERNEL}'s pipeline: {error}")),
            }
        });
        made.as_ref().map(|p| p.clone_ref()).map_err(Clone::clone)
    }
}

/// One decoder's gathers of its 4:4:4 pictures into the surfaces it lands them in (module doc),
/// one at a time, each waited for.
pub(crate) struct Gatherer<P: Platform> {
    pipeline: Arc<GatherPipeline<P>>,
    queue: Option<P::CommandQueue>,
    /// The kernel's arguments, the luma plane's and the chroma's, which the CPU writes before each
    /// gather: the last one was waited for.
    arguments: Option<P::Buffer>,
}

impl<P: Platform> Gatherer<P> {
    /// A decoder's gatherer on `pipeline`'s device.
    pub(crate) fn new(pipeline: Arc<GatherPipeline<P>>) -> Self {
        Self {
            pipeline,
            queue: None,
            arguments: None,
        }
    }

    /// The device its passes run on.
    pub(crate) fn device(&self) -> &P::Device {
        &self.pipeline.device
    }

    /// The gatherer's queue, made at its first use; `Err` where the device makes none.
    fn queue(&mut self) -> Result<&P::CommandQueue, String> {
        if self.queue.is_none() {
            let queue = (self.pipeline.device.new_command_queue()).ok_or("no command queue")?;
            queue.set_label("reims-vgpu.graphics-video-gather");
            self.queue = Some(queue);
        }
        self.queue
            .as_ref()
            .ok_or_else(|| "no command queue".to_owned())
    }

    /// Fills `length` bytes of `buffer` with zeros and waits for it: a new buffer that another
    /// API writes before the device's next command buffer, which the platform's fill of a new
    /// buffer's zeros, ahead of that command buffer, would then come after. `Err` says why not.
    pub(crate) fn zero(&mut self, buffer: &P::Buffer, length: u64) -> Result<(), String> {
        let queue = self.queue()?;
        let cb = (queue.command_buffer(OnFault::Return))
            .map_err(|raise| format!("the zeros' command buffer: {raise}"))?
            .ok_or("no command buffer for the zeros")?;
        let blit = (cb.blit_encoder(OnFault::Return))
            .map_err(|raise| format!("the zeros' blit encoder: {raise}"))?
            .ok_or("no blit encoder for the zeros")?;
        let fill = BlitCmd::FillBuffer {
            buffer,
            range: NsRange {
                location: 0,
                length,
            },
            value: 0,
        };
        let fault = |what: &str, raise: Raise| format!("the zeros: {what}: {raise}");
        blit.encode(OnFault::Return, fill)
            .map_err(|r| fault("the fill", r))?;
        (blit.encode(OnFault::Return, BlitCmd::EndEncoding)).map_err(|r| fault("its end", r))?;
        cb.commit(OnFault::Return)
            .map_err(|r| fault("the commit", r))?;
        (cb.wait_until_completed(OnFault::Return)).map_err(|r| fault("the wait", r))?;
        if cb.status() != CommandBufferStatus::COMPLETED {
            let error = cb.error().map(|e| format!("{e:?}")).unwrap_or_default();
            return Err(format!("the zeros failed {error}"));
        }
        Ok(())
    }

    /// Writes `planes` from `source` into `destination`, both bound from their first byte, by the
    /// kernel, and waits for it, on the gatherer's queue (module doc), counting the picture
    /// ([`GATHERED_CODE`]). `Err` says why it was not written: nothing submitted, or a pass that
    /// failed, after which the destination may hold part of the picture.
    pub(crate) fn gather(
        &mut self,
        source: &P::Buffer,
        destination: &P::Buffer,
        planes: &[GatherPlane; 2],
    ) -> Result<(), String> {
        let pipeline = self.pipeline.get()?;
        if self.arguments.is_none() {
            let device = &self.pipeline.device;
            let made = device.new_buffer_of_length(OnFault::Return, 2 * ARGUMENTS_APART, 0);
            match made {
                Ok(Some(buffer)) => self.arguments = Some(buffer),
                Ok(None) => return Err("no buffer for the kernel's arguments".to_owned()),
                Err(raise) => return Err(format!("the arguments' buffer: {raise}")),
            }
        }
        self.queue()?;
        let (Some(queue), Some(arguments)) = (self.queue.as_ref(), self.arguments.as_ref()) else {
            return Err("no command queue".to_owned());
        };
        let contents = arguments.contents();
        if contents.is_null() || arguments.length() < 2 * ARGUMENTS_APART {
            return Err("the arguments' buffer has no host bytes for both planes".to_owned());
        }
        let contents = contents.as_ptr::<u8>();
        for (n, plane) in planes.iter().enumerate() {
            let bytes = plane.bytes();
            // SAFETY: the buffer's host bytes, two planes' arguments apart at least, hold each
            // plane's 40 at its multiple of 256; and no GPU work reads them: the gather before was
            // waited for.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    contents.add(n * ARGUMENTS_APART as usize),
                    bytes.len(),
                );
            }
        }
        let cb = match queue.command_buffer(OnFault::Return) {
            Ok(Some(cb)) => cb,
            Ok(None) => return Err("no command buffer".to_owned()),
            Err(raise) => return Err(format!("its command buffer: {raise}")),
        };
        let compute = match cb.compute_encoder(OnFault::Return) {
            Ok(Some(compute)) => compute,
            Ok(None) => return Err("no compute encoder".to_owned()),
            Err(raise) => return Err(format!("its compute encoder: {raise}")),
        };
        let state = ComputeCmd::SetComputePipelineState {
            state: Some(&pipeline),
        };
        compute
            .encode(OnFault::Return, state)
            .map_err(|raise| format!("its pipeline: {raise}"))?;
        for (n, plane) in planes.iter().enumerate() {
            let commands = [
                ComputeCmd::SetBuffers {
                    buffers: &[Some(source), Some(destination), Some(arguments)],
                    offsets: &[0, 0, n as u64 * ARGUMENTS_APART],
                    range: NsRange {
                        location: 0,
                        length: 3,
                    },
                },
                ComputeCmd::DispatchThreadgroups {
                    threadgroups_per_grid: plane.threadgroups(),
                    threads_per_threadgroup: MtlSize::new(GROUP_WIDTH, 1, 1),
                },
            ];
            for command in commands {
                compute
                    .encode(OnFault::Return, command)
                    .map_err(|raise| format!("a plane's dispatch: {raise}"))?;
            }
        }
        (compute.encode(OnFault::Return, ComputeCmd::EndEncoding))
            .map_err(|raise| format!("its compute end: {raise}"))?;
        cb.commit(OnFault::Return)
            .map_err(|raise| format!("its commit: {raise}"))?;
        (cb.wait_until_completed(OnFault::Return)).map_err(|raise| format!("its wait: {raise}"))?;
        if cb.status() != CommandBufferStatus::COMPLETED {
            let error = cb.error().map(|e| format!("{e:?}")).unwrap_or_default();
            return Err(format!("the GPU's pass failed {error}"));
        }
        reims_vgpu_vulkan::answer::count_kind(GATHERED_CODE);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
