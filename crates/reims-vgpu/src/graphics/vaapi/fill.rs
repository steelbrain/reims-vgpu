//! An encode session's frames written by the runtime device's GPU into inputs of the VA crate's
//! encoder, on Linux ([`DeviceFills`], the arm's [`GpuCopies`]): each input is a buffer of the
//! device's whose memory the encoder's display imports, and the encoder codes the frame from it
//! (`VideoEncoder::encode_target`), as NVENC's filler writes NVENC's input
//! (`nvidia::encode::fill`).
//!
//! - **The input** is a buffer in a dma-buf block of its own
//!   (`new_dma_buf_buffer_of_length_padding_unobserved`), made on a session's first frame, its
//!   memory exported once and imported on the encoder's display as a surface of the encoder's
//!   input shape, its format and coded size (`VideoEncoder::import_input`), laid out as the
//!   encoder says (`VideoEncoder::input_layout`, asked once a shape): as the driver lays out its
//!   own inputs where it lays them out in Intel's Y tiles, as iHD does, since iHD codes a linear
//!   input's frames otherwise than it codes their upload (the VA crate's encoder docs, "Imported
//!   inputs"); linear otherwise, the luma plane's coded rows from its first byte, then the chroma
//!   plane's, half as many, at one pitch, the coded row's bytes rounded up to 256, as the arm's
//!   imports of device caches lie (the VA crate's `target`). The import holds the buffer. The
//!   session keeps its
//!   input while the encoder's shape stays and the encoder no longer holds it, and makes another
//!   otherwise: an encoder made for a new sequence of another shape, or a stalled picture's,
//!   which the encoder keeps while the GPU may read it.
//! - **A frame** is written by the platform library's `videoEncodeFill` kernel (`shaders.metal`),
//!   on a queue of the filler's own: one dispatch a plane, a thread a 32-bit word of the input,
//!   stored where the input's layout puts it.
//!   Each word holds the source's bytes, read where they lie, in the surface's pages or its device
//!   caches, by 32-bit loads, the source bound from its plane's first byte rounded down to 256. The
//!   kernel reads the rows the VA crate's upload reads: `height` luma rows and `height / 2`
//!   chroma rows, `width` samples' bytes of each. It fills the margin to the coded size as the
//!   upload fills it: each row past the frame's repeats its last, and each byte past a row's bytes
//!   repeats the row's last sample, or Cb/Cr pair. So the encoder codes the same bytes either way.
//!   A blit then hands the input to the driver (`BlitEncoder::release_to_external`, rt's release
//!   of its block), and the fill is waited for before the encoder reads the input. The driver never
//!   writes the input, and nothing reads it through the device, so the next frame's fill writes it
//!   anew with no acquire, as rt's release lets it (`rt_platform::export`, "The release").
//! - **Refusals** ([`Filled::Declined`]) come before anything is submitted, and the frame is
//!   uploaded instead. The device or the driver may refuse to make an input, or the device the
//!   kernel's pipeline; neither is tried again in the session. A plane in its cache's surface,
//!   which no encode's source plane is, is declined too. A fill that failed once
//!   submitted is an error: its input is kept out of use while the session lives, as the GPU may
//!   write it still.
//!
//! Each session logs at its end how many frames went this way, and the fills' times.

use std::os::fd::AsFd as _;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use reims_vgpu_video::core::runtime::Runtime;
use reims_vgpu_video::vaapi::{
    InputLayout, TARGET_ALIGNMENT, Target, TargetPlane, TargetShape, TargetTiling, VideoEncoder,
};
use reims_vgpu_vulkan::core::platform::Handle as _;
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::blit::{BlitCmd, BlitEncoder as _};
use reims_vgpu_vulkan::core::platform::command::{CommandBuffer as _, CommandQueue as _};
use reims_vgpu_vulkan::core::platform::compute::{ComputeCmd, ComputeEncoder as _};
use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
use reims_vgpu_vulkan::core::platform::resource::{Buffer as _, Library as _};
use reims_vgpu_vulkan::core::platform::types::{CommandBufferStatus, MtlSize, NsRange};
use reims_vgpu_vulkan::core::raise::OnFault;

use super::super::video_host::{BackendError, OutputPlane};
use super::{Filled, GpuCopier, GpuCopies, GpuFiller, NAME, STORAGE_PRIVATE};

/// The platform library's kernel that writes a plane of the encoder's input (`shaders.metal`).
const FILL_KERNEL: &str = "videoEncodeFill";

/// The threads of a fill's threadgroup, along a row.
const GROUP_WIDTH: u64 = 64;

/// Where a plane's source is bound: its first byte rounded down to a multiple of this, which a
/// buffer binding's offset takes on any device.
const BINDING_ALIGNMENT: u64 = 256;

/// The bytes between the chroma plane's arguments and the luma's in the arguments' buffer: a
/// constant buffer's binding offset is a multiple of 256 on any device.
const ARGUMENTS_APART: u64 = 256;

/// What `videoEncodeFill` takes of a plane (`shaders.metal`'s `VideoFillPlane`): where the plane
/// lies in its source buffer as bound, its rows and the bytes its margin repeats, the input's
/// words a row, its rows and its words from one row to the next, and how the input lays its words
/// out ([`TILING_LINEAR`] or [`TILING_INTEL_Y`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FillPlane {
    first: u32,
    source_pitch: u32,
    row_bytes: u32,
    rows: u32,
    unit: u32,
    coded_words: u32,
    coded_rows: u32,
    pitch_words: u32,
    tiling: u32,
}

/// `VideoFillPlane`'s `tiling` of a linear input.
const TILING_LINEAR: u32 = 0;
/// `VideoFillPlane`'s `tiling` of an input in Intel's Y tiles.
const TILING_INTEL_Y: u32 = 1;

impl FillPlane {
    /// Its bytes, as the kernel reads them.
    fn bytes(&self) -> [u8; 36] {
        let words = [
            self.first,
            self.source_pitch,
            self.row_bytes,
            self.rows,
            self.unit,
            self.coded_words,
            self.coded_rows,
            self.pitch_words,
            self.tiling,
        ];
        let mut bytes = [0; 36];
        for (to, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            *to = word.to_le_bytes();
        }
        bytes
    }

    /// The threadgroups of its dispatch: a thread a word of the input.
    fn threadgroups(&self) -> MtlSize {
        let across = u64::from(self.coded_words).div_ceil(GROUP_WIDTH);
        MtlSize::new(across, u64::from(self.coded_rows), 1)
    }
}

/// Where a source plane lies: its buffer's bytes, its first byte, the bytes from one of its rows
/// to the next, the bytes of a row, and the rows it holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Source {
    length: u64,
    offset: u64,
    bytes_per_row: u64,
    row_bytes: u64,
    height: u64,
}

impl Source {
    fn of<P: Platform>(plane: &OutputPlane<P>) -> Self {
        Self {
            length: plane.buffer.length(),
            offset: plane.offset,
            bytes_per_row: plane.bytes_per_row,
            row_bytes: plane.row_bytes,
            height: plane.height,
        }
    }
}

/// One plane's dispatch: its source bound from byte `bound`, the input's bytes from `at`, and
/// what the kernel takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PlaneFill {
    bound: u64,
    at: u64,
    plane: FillPlane,
}

/// The rows of a Y tile: an input plane's rows in Intel's Y tiles take whole rows of tiles.
const Y_TILE_ROWS: u64 = 32;
/// The bytes of a Y tile's row: an input's pitch in Intel's Y tiles is a whole number of tiles.
const Y_TILE_WIDTH: u64 = 128;

/// The dispatches that write a frame of `size` (luma samples) from `sources`, its luma and chroma
/// planes, into an input of `shape` laid out as `layout` (module doc): `Err` says why none can.
/// The rows read are the upload's: `height` of luma and `height / 2` of chroma, each `width`
/// samples' bytes, its margin repeating a sample, or a Cb/Cr pair.
fn plane_fills(
    sources: [Source; 2],
    (width, height): (u32, u32),
    shape: TargetShape,
    layout: InputLayout,
) -> Result<[PlaneFill; 2], String> {
    let (coded_width, coded_height) = shape.coded;
    if width < 2 || height < 2 || coded_width < width || coded_height < height {
        return Err(format!(
            "a frame of {width}x{height} for an input of {coded_width}x{coded_height}"
        ));
    }
    let sample = shape.format.bytes_per_sample();
    let row_bytes = u64::from(width) * sample;
    let coded_bytes = u64::from(coded_width) * sample;
    let InputLayout {
        tiling,
        pitch,
        chroma_at,
        length,
    } = layout;
    if coded_bytes % 4 != 0 {
        return Err(format!(
            "coded rows of {coded_bytes} bytes, not whole words"
        ));
    }
    // The kernel's word for the tiling, the rows a plane's rows are rounded up to and what its
    // pitch is a multiple of.
    let (tiling, rows_of, pitch_of) = match tiling {
        TargetTiling::Linear => (TILING_LINEAR, 1, 4),
        TargetTiling::IntelY => (TILING_INTEL_Y, Y_TILE_ROWS, Y_TILE_WIDTH),
    };
    if pitch < coded_bytes || pitch % pitch_of != 0 || chroma_at % BINDING_ALIGNMENT != 0 {
        return Err(format!(
            "an input of {coded_bytes} bytes a row at a pitch of {pitch}, its chroma at byte \
             {chroma_at}, as {:?}",
            layout.tiling
        ));
    }
    let narrow = |v: u64, what: &str| u32::try_from(v).map_err(|_| format!("{what} of {v}"));
    // A plane's dispatch: `rows` of the source's rows into the input's `coded_rows` from its byte
    // `at`, in `room` bytes, its margin repeating `unit` bytes.
    let plane = |source: Source,
                 (rows, coded_rows): (u64, u64),
                 unit: u64,
                 (at, room): (u64, u64),
                 which: &str| {
        if source.row_bytes < row_bytes || source.bytes_per_row < row_bytes || source.height < rows
        {
            return Err(format!(
                "a {which} plane of {} rows of {} bytes, {} apart, for a frame of {rows} of \
                 {row_bytes}",
                source.height, source.row_bytes, source.bytes_per_row
            ));
        }
        let bound = source.offset - source.offset % BINDING_ALIGNMENT;
        let first = source.offset - bound;
        // The kernel reads whole words: the word that holds the last byte lies in the buffer.
        let reach = (first + (rows - 1) * source.bytes_per_row + row_bytes).next_multiple_of(4);
        if bound + reach > source.length {
            return Err(format!(
                "a {which} plane whose last row's word ends at byte {} of a buffer of {}",
                bound + reach,
                source.length
            ));
        }
        let pitch_words = pitch / 4;
        narrow(reach, "a source plane's reach")?;
        // The input's bytes the plane's rows take, whole rows of tiles where it is in tiles.
        let span = pitch * coded_rows.next_multiple_of(rows_of);
        if span > room {
            return Err(format!(
                "a {which} plane of {span} bytes in {room} of the input's"
            ));
        }
        narrow(span / 4, "an input plane's words")?;
        Ok(PlaneFill {
            bound,
            at,
            plane: FillPlane {
                first: narrow(first, "a first byte")?,
                source_pitch: narrow(source.bytes_per_row, "a pitch")?,
                row_bytes: narrow(row_bytes, "a row's bytes")?,
                rows: narrow(rows, "rows")?,
                unit: narrow(unit, "a unit")?,
                coded_words: narrow(coded_bytes / 4, "a coded row's words")?,
                coded_rows: narrow(coded_rows, "coded rows")?,
                pitch_words: narrow(pitch_words, "a pitch's words")?,
                tiling,
            },
        })
    };
    let (height, coded_height) = (u64::from(height), u64::from(coded_height));
    Ok([
        plane(
            sources[0],
            (height, coded_height),
            sample,
            (0, chroma_at),
            "luma",
        )?,
        plane(
            sources[1],
            (height / 2, coded_height / 2),
            2 * sample,
            (chroma_at, length.saturating_sub(chroma_at)),
            "chroma",
        )?,
    ])
}

/// The kernel's pipeline on a device, made at a session's first fill and kept for the device's
/// sessions.
struct Pipeline<P: Platform> {
    device: P::Device,
    made: OnceLock<Result<P::ComputePipeline, String>>,
}

impl<P: Platform> Pipeline<P> {
    /// The pipeline; `Err` says why the device makes none.
    fn get(&self) -> Result<P::ComputePipeline, String> {
        let made = self.made.get_or_init(|| {
            let library = match self.device.new_shader_library() {
                Ok(Some(library)) => library,
                Ok(None) => return Err("the platform's library is not made".to_owned()),
                Err(error) => return Err(format!("the platform's library: {error}")),
            };
            let function = match library.new_function(OnFault::Return, FILL_KERNEL) {
                Ok(Some(function)) => function,
                Ok(None) => return Err(format!("the platform's library holds no {FILL_KERNEL}")),
                Err(raise) => return Err(format!("{FILL_KERNEL}: {raise}")),
            };
            match self
                .device
                .new_compute_pipeline_state_for_function(Some(&function))
            {
                Ok(Some(pipeline)) => Ok(pipeline),
                Ok(None) => Err(format!("{FILL_KERNEL}'s pipeline is refused")),
                Err(error) => Err(format!("{FILL_KERNEL}'s pipeline: {error}")),
            }
        });
        made.as_ref().map(|p| p.clone_ref()).map_err(Clone::clone)
    }
}

/// The GPU fills of a device's encode sessions (module doc), and no GPU copies of decoded
/// pictures: a decode's picture lands by the driver's video processing or the CPU's copy, as it
/// does with no `GpuCopies` at all.
pub(super) struct DeviceFills<P: Platform> {
    pipeline: Arc<Pipeline<P>>,
    runtime: Arc<dyn Runtime>,
}

impl<P: Platform> DeviceFills<P> {
    /// The fills of `device`'s sessions, timed on `runtime`'s clock.
    pub(super) fn new(device: &P::Device, runtime: &Arc<dyn Runtime>) -> Self {
        Self {
            pipeline: Arc::new(Pipeline {
                device: device.clone_ref(),
                made: OnceLock::new(),
            }),
            runtime: Arc::clone(runtime),
        }
    }
}

impl<P: Platform> GpuCopies<P> for DeviceFills<P> {
    fn copier(&self) -> Result<Box<dyn GpuCopier<P>>, String> {
        Err("the platform makes no GPU copies".to_owned())
    }

    fn filler(&self) -> Result<Box<dyn GpuFiller<P>>, String> {
        Ok(Box::new(DeviceFiller {
            pipeline: Arc::clone(&self.pipeline),
            queue: None,
            arguments: None,
            layout: None,
            inputs: Vec::new(),
            refused: None,
            failed: Vec::new(),
            filled: 0,
            runtime: Arc::clone(&self.runtime),
            total: Duration::ZERO,
            longest: Duration::ZERO,
        }))
    }
}

/// An input of the session's (module doc).
struct Input<P: Platform> {
    buffer: P::Buffer,
    /// Its import on the encoder's display, which holds the buffer.
    target: Target,
    shape: TargetShape,
}

/// One encode session's fills of its frames into its encoder's inputs (module doc).
struct DeviceFiller<P: Platform> {
    pipeline: Arc<Pipeline<P>>,
    queue: Option<P::CommandQueue>,
    /// The kernel's arguments, the luma plane's and the chroma's, which the CPU writes before each
    /// fill: the last fill was waited for.
    arguments: Option<P::Buffer>,
    /// The layout the encoder gives its inputs of a shape, asked once a shape: it makes a surface
    /// of the driver's to read it.
    layout: Option<(TargetShape, InputLayout)>,
    inputs: Vec<Input<P>>,
    /// Why no input or pipeline is made, once one was refused: the session's later frames are
    /// declined with it, and no make is tried again.
    refused: Option<String>,
    /// Inputs whose fill failed on the GPU, which may write them still.
    failed: Vec<Input<P>>,
    /// Frames filled, and the time of their fills from submission to their wait's end.
    filled: u64,
    /// The clock the fills are timed on: the video's runtime's.
    runtime: Arc<dyn Runtime>,
    total: Duration,
    longest: Duration,
}

impl<P: Platform> GpuFiller<P> for DeviceFiller<P> {
    fn fill(
        &mut self,
        encoder: &VideoEncoder,
        luma: &OutputPlane<P>,
        chroma: &OutputPlane<P>,
    ) -> Result<Filled, BackendError> {
        if luma.in_surface() || chroma.in_surface() {
            return Ok(Filled::Declined(
                "a plane in its cache's surface".to_owned(),
            ));
        }
        let shape = encoder.input_shape();
        let settings = encoder.settings();
        let size = (settings.width, settings.height);
        let layout = self.layout_for(encoder, shape);
        let sources = [Source::of(luma), Source::of(chroma)];
        let fills = match plane_fills(sources, size, shape, layout) {
            Ok(fills) => fills,
            Err(why) => return Ok(Filled::Declined(why)),
        };
        if let Some(why) = &self.refused {
            return Ok(Filled::Declined(why.clone()));
        }
        let pipeline = match self.pipeline.get() {
            Ok(pipeline) => pipeline,
            Err(why) => return Ok(Filled::Declined(self.refused.insert(why).clone())),
        };
        let index = match self.input_for(encoder, shape, layout) {
            Ok(index) => index,
            Err(why) => return Ok(Filled::Declined(self.refused.insert(why).clone())),
        };
        let (destination, target) = {
            let input = &self.inputs[index];
            (input.buffer.clone_ref(), input.target.clone())
        };
        let start = self.runtime.now();
        let sources = [&luma.buffer, &chroma.buffer];
        match self.write(&pipeline, &destination, sources, &fills) {
            Ok(None) => {
                let took = self.runtime.elapsed_since(start);
                self.filled += 1;
                self.total += took;
                self.longest = self.longest.max(took);
                Ok(Filled::Written(target))
            }
            Ok(Some(why)) => Ok(Filled::Declined(why)),
            Err(error) => {
                self.failed.push(self.inputs.swap_remove(index));
                Err(error)
            }
        }
    }
}

impl<P: Platform> DeviceFiller<P> {
    /// The layout `encoder` gives its inputs of `shape` (module doc), asked where the session has
    /// not asked it of this shape.
    fn layout_for(&mut self, encoder: &VideoEncoder, shape: TargetShape) -> InputLayout {
        match self.layout {
            Some((of, layout)) if of == shape => layout,
            _ => self.layout.insert((shape, encoder.input_layout())).1,
        }
    }

    /// The index of the session's input of `shape` that `encoder` no longer holds, made where
    /// there is none (module doc), laid out as `layout` says; the others are let go of. `Err`
    /// says why none is made.
    fn input_for(
        &mut self,
        encoder: &VideoEncoder,
        shape: TargetShape,
        layout: InputLayout,
    ) -> Result<usize, String> {
        self.inputs
            .retain(|input| input.shape == shape && !input.target.is_held());
        if self.inputs.is_empty() {
            let input = self.make_input(encoder, shape, layout)?;
            self.inputs.push(input);
        }
        Ok(0)
    }

    /// A new input of `shape`, laid out as `layout` says, imported on `encoder`'s display.
    fn make_input(
        &self,
        encoder: &VideoEncoder,
        shape: TargetShape,
        layout: InputLayout,
    ) -> Result<Input<P>, String> {
        let InputLayout {
            tiling,
            pitch,
            chroma_at,
            length,
        } = layout;
        let made = self
            .pipeline
            .device
            .new_dma_buf_buffer_of_length_padding_unobserved(
                OnFault::Return,
                length,
                STORAGE_PRIVATE,
                Some(0),
            );
        let buffer = match made {
            Ok(Some(buffer)) => buffer,
            Ok(None) => return Err("the device makes no buffer of a dma-buf".to_owned()),
            Err(raise) => return Err(format!("a buffer of a dma-buf's make: {raise}")),
        };
        let exported = buffer.export_dma_buf()?;
        if exported.offset % TARGET_ALIGNMENT != 0 {
            return Err(format!(
                "an input at byte {} of its dma-buf, not on {TARGET_ALIGNMENT}",
                exported.offset
            ));
        }
        let planes = [
            TargetPlane {
                offset: exported.offset,
                bytes_per_row: pitch,
            },
            TargetPlane {
                offset: exported.offset + chroma_at,
                bytes_per_row: pitch,
            },
        ];
        let target = encoder
            .import_input(
                exported.fd.as_fd(),
                exported.allocation_size,
                planes,
                tiling,
                Box::new(buffer.clone_ref()),
            )
            .map_err(|error| format!("the import of an input: {error}"))?;
        Ok(Input {
            buffer,
            target,
            shape,
        })
    }

    /// Writes `fills` from `sources`, the luma plane's buffer and the chroma's, into
    /// `destination` by the kernel's `pipeline`, releases it to the driver and waits for it, on
    /// the filler's queue. `Some` says why nothing was submitted; `Err` is a fill that failed once
    /// submitted.
    fn write(
        &mut self,
        pipeline: &P::ComputePipeline,
        destination: &P::Buffer,
        sources: [&P::Buffer; 2],
        fills: &[PlaneFill; 2],
    ) -> Result<Option<String>, BackendError> {
        let device = &self.pipeline.device;
        if self.queue.is_none() {
            let Some(queue) = device.new_command_queue() else {
                return Ok(Some("no command queue".to_owned()));
            };
            queue.set_label("reims-vgpu.graphics-video-fill");
            self.queue = Some(queue);
        }
        if self.arguments.is_none() {
            let made = device.new_buffer_of_length(OnFault::Return, 2 * ARGUMENTS_APART, 0);
            match made {
                Ok(Some(buffer)) => self.arguments = Some(buffer),
                Ok(None) => return Ok(Some("no buffer for the kernel's arguments".to_owned())),
                Err(raise) => return Ok(Some(format!("the arguments' buffer: {raise}"))),
            }
        }
        let (Some(queue), Some(arguments)) = (self.queue.as_ref(), self.arguments.as_ref()) else {
            return Ok(Some("no command queue".to_owned()));
        };
        let cb = match queue.command_buffer(OnFault::Return) {
            Ok(Some(cb)) => cb,
            Ok(None) => return Ok(Some("no command buffer".to_owned())),
            Err(raise) => return Ok(Some(format!("its command buffer: {raise}"))),
        };
        let planes = Planes {
            pipeline,
            arguments,
            destination,
            sources,
        };
        if let Some(why) = encode_planes::<P>(&cb, &planes, fills) {
            return Ok(Some(why));
        }
        let blit = match cb.blit_encoder(OnFault::Return) {
            Ok(Some(blit)) => blit,
            Ok(None) => return Ok(Some("no blit encoder".to_owned())),
            Err(raise) => return Ok(Some(format!("its blit encoder: {raise}"))),
        };
        if let Err(why) = blit.release_to_external(destination) {
            return Ok(Some(format!("its release to the driver: {why}")));
        }
        if let Err(raise) = blit.encode(OnFault::Return, BlitCmd::EndEncoding) {
            return Ok(Some(format!("its end: {raise}")));
        }
        if let Err(raise) = cb.commit(OnFault::Return) {
            return Ok(Some(format!("its commit: {raise}")));
        }
        let failed = match cb.wait_until_completed(OnFault::Return) {
            Err(raise) => Some(format!("its wait: {raise}")),
            Ok(()) if cb.status() != CommandBufferStatus::COMPLETED => {
                let error = cb.error().map(|e| format!("{e:?}")).unwrap_or_default();
                Some(format!("the fill failed {error}"))
            }
            Ok(()) => None,
        };
        match failed {
            Some(failed) => {
                Err(format!("the GPU's fill of a frame into the encoder's input: {failed}").into())
            }
            None => Ok(None),
        }
    }
}

/// What a fill's dispatches bind: the kernel's pipeline, the buffer of its arguments, the input
/// they write, and the luma plane's buffer and the chroma's they read.
struct Planes<'a, P: Platform> {
    pipeline: &'a P::ComputePipeline,
    arguments: &'a P::Buffer,
    destination: &'a P::Buffer,
    sources: [&'a P::Buffer; 2],
}

/// Writes each of `fills`' arguments into `planes.arguments`, 256 bytes apart, and encodes into
/// `cb` the pass that writes them (module doc): one dispatch a plane. `Some` says why it could
/// not, with nothing committed.
fn encode_planes<P: Platform>(
    cb: &P::CommandBuffer,
    planes: &Planes<'_, P>,
    fills: &[PlaneFill; 2],
) -> Option<String> {
    let contents = planes.arguments.contents();
    if contents.is_null() || planes.arguments.length() < 2 * ARGUMENTS_APART {
        return Some("the arguments' buffer has no host bytes for both planes".to_owned());
    }
    let contents = contents.as_ptr::<u8>();
    for (n, fill) in fills.iter().enumerate() {
        let bytes = fill.plane.bytes();
        // SAFETY: the buffer's host bytes, two planes' arguments apart at least, hold each plane's
        // 36 at its multiple of 256; and no GPU work reads them: the fill before was waited for.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                contents.add(n * ARGUMENTS_APART as usize),
                bytes.len(),
            );
        }
    }
    let compute = match cb.compute_encoder(OnFault::Return) {
        Ok(Some(compute)) => compute,
        Ok(None) => return Some("no compute encoder".to_owned()),
        Err(raise) => return Some(format!("its compute encoder: {raise}")),
    };
    let state = ComputeCmd::SetComputePipelineState {
        state: Some(planes.pipeline),
    };
    if let Err(raise) = compute.encode(OnFault::Return, state) {
        return Some(format!("its pipeline: {raise}"));
    }
    for ((n, fill), source) in fills.iter().enumerate().zip(planes.sources) {
        let commands = [
            ComputeCmd::SetBuffers {
                buffers: &[
                    Some(source),
                    Some(planes.destination),
                    Some(planes.arguments),
                ],
                offsets: &[fill.bound, fill.at, n as u64 * ARGUMENTS_APART],
                range: NsRange {
                    location: 0,
                    length: 3,
                },
            },
            ComputeCmd::DispatchThreadgroups {
                threadgroups_per_grid: fill.plane.threadgroups(),
                threads_per_threadgroup: MtlSize::new(GROUP_WIDTH, 1, 1),
            },
        ];
        for command in commands {
            if let Err(raise) = compute.encode(OnFault::Return, command) {
                return Some(format!("a plane's dispatch: {raise}"));
            }
        }
    }
    if let Err(raise) = compute.encode(OnFault::Return, ComputeCmd::EndEncoding) {
        return Some(format!("its compute end: {raise}"));
    }
    None
}

impl<P: Platform> Drop for DeviceFiller<P> {
    fn drop(&mut self) {
        if self.filled == 0 {
            return;
        }
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        let mean = self.total / u32::try_from(self.filled).unwrap_or(u32::MAX);
        log::info!(
            "hardware video: {NAME}: an encode session's frames written by the GPU into the \
             encoder's inputs: {}, {:.3} ms each from submission to the wait's end, {:.3} ms at \
             most",
            self.filled,
            ms(mean),
            ms(self.longest)
        );
    }
}

#[cfg(test)]
mod tests;
