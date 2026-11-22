//! A decode session's pictures copied by the runtime device's GPU into their outputs' device
//! caches, on Linux ([`ScratchCopier`], the copier of `fill::DeviceFills`: the second way a
//! picture lands, in `super`'s module doc), where the driver's video processing declined to
//! write the picture into its cache in place: radeonsi imports no plane off a multiple of 256
//! bytes of its memory, as CoreVideo lays its planes 32 bytes into their surface, and the RX 550's
//! imports no rows narrower than its own pitch (`docs/video/devices.md`, "Output layouts").
//!
//! - **The scratch** is a buffer of the session's in a dma-buf block of its own
//!   (`new_dma_buf_buffer_of_length_padding_unobserved`), the platform's, made at the session's
//!   first picture and made again for a picture of another size: the picture's luma plane from
//!   its first byte and its chroma plane right after the luma's rows, at one pitch, the even
//!   width's bytes rounded up to 256 (`TARGET_ALIGNMENT`), as the arm's dma-buf caches lie where
//!   the driver writes them in place. It is made as zeros, waited for, before the driver first
//!   writes it, so that the platform's fill of a new buffer's zeros, ahead of its next command
//!   buffer, comes before that write. No guest message names it and no guest maps it: it holds
//!   what the driver wrote of the session's last picture, and the host reads nothing of it but
//!   the rows the kernel copies.
//! - **A picture** is written into the scratch by the driver's video processing (the VA crate's
//!   `Frame::write_to_guest`, as the first way writes a cache, through the scratch's memory as
//!   [`CacheBytes`]), acquired (`Buffer::external_write_done`), then copied by the platform
//!   library's `videoEncodeFill` kernel (`shaders.metal`, `fill`'s) on a queue of the copier's
//!   own: one dispatch a plane, a thread a 32-bit word, each word a plain copy, as the plane's
//!   coded row is its rows' bytes. Each plane's rows of the crop the CPU's copy writes
//!   (`ReadBack::copy_into`) are copied into the plane's cache at its own first byte and pitch,
//!   and nothing else of the cache; then the copy is waited for. A picture is copied so only
//!   where that crop fills the planes' rows, every byte of them that the CPU's copy and its
//!   staging's copy into the cache write: so the cache holds the same bytes either way.
//! - **Refusals** ([`DeviceWrite::Declined`]) come before anything is written into the cache, and
//!   the picture is copied by the CPU: a plane in the guest's pages or in its cache's surface, a
//!   crop that does not fill the planes' rows, rows or a first byte or pitch that are not whole
//!   32-bit words, the driver's video processing's decline of the scratch, a submission refused.
//!   The device's or the driver's refusal to make the scratch, and the device's of the kernel's
//!   pipeline, are not tried again in the session. A copy that failed once submitted is an error,
//!   and its scratch is kept out of use while the session lives, as the GPU may read it still.
//!
//! Each picture copied is counted ([`COPIED_CODE`]), with the KiB of its planes' crop
//! ([`COPIED_KIB_CODE`]): a GPU pass a picture that a Mac, whose decoder writes the guest's
//! surface, does not make. Each session logs at its end how many pictures went this way, and the
//! time from the driver's write to the copy's end.

use std::sync::Arc;
use std::time::Duration;

use reims_vgpu_video::core::backend::DeviceWrite;
use reims_vgpu_video::core::device_memory::{GuestTarget, MemoryLife};
use reims_vgpu_video::core::runtime::Runtime;
use reims_vgpu_video::core::surfaces::SurfaceLayout;
use reims_vgpu_video::vaapi::{Frame, GuestWrites, TARGET_ALIGNMENT};
use reims_vgpu_vulkan::core::platform::Handle as _;
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::blit::{BlitCmd, BlitEncoder as _};
use reims_vgpu_vulkan::core::platform::command::{CommandBuffer as _, CommandQueue as _};
use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
use reims_vgpu_vulkan::core::platform::types::{CommandBufferStatus, NsRange};
use reims_vgpu_vulkan::core::raise::{OnFault, Raise};
use reims_vgpu_vulkan::core::sync::Mutex;

use super::super::video_host::{BackendError, OutputPlane};
use super::fill::{
    ARGUMENTS_APART, FillPlane, Pipeline, PlaneFill, Planes, Source, TILING_LINEAR, encode_planes,
};
use super::{CacheBytes, CacheMemory, CacheOf, GpuCopier, NAME, Place, STORAGE_PRIVATE};

/// The code of a picture the GPU copied from the scratch into its output's caches (the module
/// doc), counted per picture.
pub(crate) const COPIED_CODE: &str = "video-pictures-gpu-copied";

/// The code of the KiB of the planes' crops the GPU copied (the module doc), counted per picture.
pub(crate) const COPIED_KIB_CODE: &str = "video-gpu-copied-kib";

/// How the scratch of a picture lies (module doc): its even width and height, the bytes of a
/// sample, the pitch of both planes, the chroma plane's first byte, and the bytes of both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ScratchLayout {
    width: u64,
    height: u64,
    sample: u64,
    pitch: u64,
    chroma_at: u64,
    length: u64,
}

impl ScratchLayout {
    /// The scratch of a picture of `width` by `height` at `bit_depth`; `Err` says why there is
    /// none.
    fn of(width: u32, height: u32, bit_depth: u32) -> Result<Self, String> {
        let sample = match bit_depth {
            8 => 1,
            10 => 2,
            depth => return Err(format!("a {depth}-bit picture")),
        };
        if width == 0 || height == 0 {
            return Err("an empty picture".to_owned());
        }
        let (width, height) = (u64::from(width), u64::from(height));
        let (width, height) = (width.next_multiple_of(2), height.next_multiple_of(2));
        let pitch = (width * sample).next_multiple_of(TARGET_ALIGNMENT);
        let chroma_at = pitch * height;
        Ok(Self {
            width,
            height,
            sample,
            pitch,
            chroma_at,
            length: chroma_at + pitch * (height / 2),
        })
    }

    /// The layout the driver's video processing writes the picture in, of `bit_depth`
    /// (`in_place_layout`): its even width and height, each plane at its first byte and pitch.
    fn surface(&self, bit_depth: u32) -> Result<SurfaceLayout, String> {
        let row_bytes = self.width * self.sample;
        let place = |offset: u64, width: u64, height: u64| Place {
            offset,
            bytes_per_row: self.pitch,
            row_bytes,
            width,
            height,
            rows_held: height,
        };
        super::in_place_layout(
            place(0, self.width, self.height),
            place(self.chroma_at, self.width / 2, self.height / 2),
            bit_depth,
        )
    }
}

/// The dispatches that copy a picture of `size` (its crop's luma samples), lying in a scratch
/// laid out as `scratch`, into `outputs`, its luma and chroma planes (module doc): each plane's
/// rows of the crop the CPU's copy writes, bound from the scratch's plane and the output plane's
/// first bytes. `Err` says why none is made: a crop that does not fill the planes' rows, or rows
/// that are not whole words.
fn plane_copies(
    scratch: ScratchLayout,
    (width, height): (u64, u64),
    outputs: [Source; 2],
) -> Result<[PlaneFill; 2], String> {
    let [luma, chroma] = outputs;
    let sample = scratch.sample;
    // The CPU's copy's crop into these planes (the VA crate's `ReadBack::copy_into`): the
    // columns each plane's rows hold, a chroma row a pair for every two, and the rows each
    // holds, a chroma row for every two.
    let fit = |plane: Source| plane.row_bytes.min(plane.bytes_per_row) / sample;
    let width = width.min(fit(luma)).min((fit(chroma) & !1).max(1));
    let height = height.min(luma.height).min(chroma.height.saturating_mul(2));
    let crop = [
        (width * sample, height),
        (width.div_ceil(2) * 2 * sample, height.div_ceil(2)),
    ];
    if width > scratch.width || height > scratch.height {
        return Err(format!(
            "a crop of {width}x{height} past its scratch's {}x{}",
            scratch.width, scratch.height
        ));
    }
    let narrow = |v: u64, what: &str| u32::try_from(v).map_err(|_| format!("{what} of {v}"));
    let plane = |output: Source,
                 (row_bytes, rows): (u64, u64),
                 unit: u64,
                 from: u64,
                 which: &str|
     -> Result<PlaneFill, String> {
        if row_bytes != output.row_bytes || rows != output.height || rows == 0 {
            return Err(format!(
                "a crop of {width}x{height} whose {which} rows, {rows} of {row_bytes} bytes, do \
                 not fill its plane's {} of {}",
                output.height, output.row_bytes
            ));
        }
        if !row_bytes.is_multiple_of(4)
            || !output.offset.is_multiple_of(4)
            || !output.bytes_per_row.is_multiple_of(4)
        {
            return Err(format!(
                "a {which} plane at byte {} of rows of {row_bytes} bytes, {} apart: not whole \
                 words",
                output.offset, output.bytes_per_row
            ));
        }
        let end = (rows - 1)
            .checked_mul(output.bytes_per_row)
            .and_then(|last| output.offset.checked_add(last)?.checked_add(row_bytes));
        if !end.is_some_and(|end| end <= output.length) {
            return Err(format!(
                "a {which} plane whose rows end past its buffer of {}",
                output.length
            ));
        }
        Ok(PlaneFill {
            bound: from,
            at: output.offset,
            plane: FillPlane {
                first: 0,
                source_pitch: narrow(scratch.pitch, "a pitch")?,
                row_bytes: narrow(row_bytes, "a row's bytes")?,
                rows: narrow(rows, "rows")?,
                unit: narrow(unit, "a unit")?,
                coded_words: narrow(row_bytes / 4, "a row's words")?,
                coded_rows: narrow(rows, "rows")?,
                pitch_words: narrow(output.bytes_per_row / 4, "a pitch's words")?,
                tiling: TILING_LINEAR,
            },
        })
    };
    Ok([
        plane(luma, crop[0], sample, 0, "luma")?,
        plane(chroma, crop[1], 2 * sample, scratch.chroma_at, "chroma")?,
    ])
}

/// The KiB of a copy's planes' crop.
fn copied_kib(fills: &[PlaneFill; 2]) -> u64 {
    let bytes: u64 = (fills.iter())
        .map(|fill| u64::from(fill.plane.row_bytes) * u64::from(fill.plane.rows))
        .sum();
    bytes >> 10
}

/// A session's scratch (module doc).
struct Scratch<P: Platform> {
    layout: ScratchLayout,
    buffer: P::Buffer,
    /// Its memory as the driver's video processing writes it, by `life`'s identity, so that the
    /// display's import of it goes as the scratch does.
    bytes: CacheBytes<P>,
    /// The life of `bytes`' identity, held while the scratch is.
    _life: MemoryLife,
}

/// One decode session's GPU copies of its pictures into their outputs' device caches (module
/// doc).
pub(super) struct ScratchCopier<P: Platform> {
    pipeline: Arc<Pipeline<P>>,
    queue: Option<P::CommandQueue>,
    /// The kernel's arguments, the luma plane's and the chroma's, which the CPU writes before each
    /// copy: the last copy was waited for.
    arguments: Option<P::Buffer>,
    scratch: Option<Scratch<P>>,
    /// Why no scratch or pipeline is made, once one was refused: the session's later pictures are
    /// declined with it, and no make is tried again.
    refused: Option<String>,
    /// Scratches whose copy failed on the GPU, which may read them still.
    failed: Vec<Scratch<P>>,
    /// Pictures copied, and the time from the driver's write to the copy's wait's end.
    copied: u64,
    /// The clock the copies are timed on: the video's runtime's.
    runtime: Arc<dyn Runtime>,
    total: Duration,
    longest: Duration,
}

impl<P: Platform> ScratchCopier<P> {
    /// A session's copier on `pipeline`'s device, timed on `runtime`'s clock.
    pub(super) fn new(pipeline: Arc<Pipeline<P>>, runtime: Arc<dyn Runtime>) -> Self {
        Self {
            pipeline,
            queue: None,
            arguments: None,
            scratch: None,
            refused: None,
            failed: Vec::new(),
            copied: 0,
            runtime,
            total: Duration::ZERO,
            longest: Duration::ZERO,
        }
    }
}

impl<P: Platform> GpuCopier<P> for ScratchCopier<P> {
    fn copy(
        &mut self,
        frame: &Frame,
        imports: &GuestWrites,
        luma: &OutputPlane<P>,
        chroma: &OutputPlane<P>,
    ) -> Result<DeviceWrite, BackendError> {
        let declined = |why: &str| Ok(DeviceWrite::Declined(why.to_owned()));
        if luma.in_surface() || chroma.in_surface() {
            return declined("a plane in its cache's surface");
        }
        if !luma.cached || !chroma.cached {
            return declined("a plane in the guest's pages");
        }
        let bit_depth = frame.bit_depth();
        let layout = match ScratchLayout::of(frame.width(), frame.height(), bit_depth) {
            Ok(layout) => layout,
            Err(why) => return declined(&why),
        };
        let size = (u64::from(frame.width()), u64::from(frame.height()));
        let fills = match plane_copies(layout, size, [Source::of(luma), Source::of(chroma)]) {
            Ok(fills) => fills,
            Err(why) => return declined(&why),
        };
        let surface = match layout.surface(bit_depth) {
            Ok(surface) => surface,
            Err(why) => return declined(&why),
        };
        if let Some(why) = &self.refused {
            return declined(why);
        }
        let pipeline = match self.pipeline.get() {
            Ok(pipeline) => pipeline,
            Err(why) => return declined(self.refused.insert(why)),
        };
        if let Err(why) = self.scratch_for(layout) {
            return declined(self.refused.insert(why));
        }
        let Some(scratch) = self.scratch.as_ref() else {
            return declined("no scratch");
        };
        let start = self.runtime.now();
        let written = match GuestTarget::new(&surface, &scratch.bytes) {
            Ok(target) => frame.write_to_guest(&target, imports)?,
            Err(why) => return declined(&format!("the scratch's target: {why}")),
        };
        if let DeviceWrite::Declined(why) = written {
            let why = format!("the driver's video processing into its scratch: {why}");
            return declined(&why);
        }
        if let Err(why) = scratch.buffer.external_write_done() {
            let why = format!("the scratch's acquire: {why}");
            return declined(self.refused.insert(why));
        }
        let source = scratch.buffer.clone_ref();
        let destinations = [&luma.buffer, &chroma.buffer];
        match self.write(&pipeline, &source, destinations, &fills) {
            Ok(None) => {
                let took = self.runtime.elapsed_since(start);
                self.copied += 1;
                self.total += took;
                self.longest = self.longest.max(took);
                reims_vgpu_vulkan::answer::count_kind(COPIED_CODE);
                reims_vgpu_vulkan::answer::count_kind_by(COPIED_KIB_CODE, copied_kib(&fills));
                Ok(DeviceWrite::Written)
            }
            Ok(Some(why)) => declined(&why),
            Err(error) => {
                self.failed.extend(self.scratch.take());
                Err(error)
            }
        }
    }
}

impl<P: Platform> ScratchCopier<P> {
    /// Makes the session's scratch for a picture laid out as `layout`, where it has none of that
    /// layout (module doc): `Err` says why none is made.
    fn scratch_for(&mut self, layout: ScratchLayout) -> Result<(), String> {
        if self.scratch.as_ref().is_some_and(|s| s.layout == layout) {
            return Ok(());
        }
        self.scratch = None;
        let device = &self.pipeline.device;
        let made = device.new_dma_buf_buffer_of_length_padding_unobserved(
            OnFault::Return,
            layout.length,
            STORAGE_PRIVATE,
            Some(0),
        );
        let buffer = match made {
            Ok(Some(buffer)) => buffer,
            Ok(None) => return Err("the device makes no scratch of a dma-buf".to_owned()),
            Err(raise) => return Err(format!("a scratch of a dma-buf's make: {raise}")),
        };
        let exported = buffer.export_dma_buf()?;
        if exported.offset % TARGET_ALIGNMENT != 0 {
            return Err(format!(
                "a scratch at byte {} of its dma-buf, not on {TARGET_ALIGNMENT}",
                exported.offset
            ));
        }
        self.zero(&buffer, layout.length)?;
        let life = MemoryLife::new();
        let bytes = CacheBytes {
            memory: Arc::new(CacheMemory {
                dma_buf: exported.fd,
                size: exported.allocation_size,
                offset: exported.offset,
                cache: CacheOf::Buffer(buffer.clone_ref()),
                last: Mutex::new(None),
            }),
            identity: life.identity(),
        };
        self.scratch = Some(Scratch {
            layout,
            buffer,
            bytes,
            _life: life,
        });
        Ok(())
    }

    /// The copier's queue, made at its first use; `None` where the device makes none.
    fn queue(&mut self) -> Option<&P::CommandQueue> {
        if self.queue.is_none() {
            let queue = self.pipeline.device.new_command_queue()?;
            queue.set_label("reims-vgpu.graphics-video-gpu-copy");
            self.queue = Some(queue);
        }
        self.queue.as_ref()
    }

    /// Fills `length` bytes of a new scratch, `buffer`, with zeros and waits for it (module doc):
    /// `Err` says why not.
    fn zero(&mut self, buffer: &P::Buffer, length: u64) -> Result<(), String> {
        let queue = self.queue().ok_or("no command queue")?;
        let cb = (queue.command_buffer(OnFault::Return))
            .map_err(|raise| format!("the scratch's zeros' command buffer: {raise}"))?
            .ok_or("no command buffer for the scratch's zeros")?;
        let blit = (cb.blit_encoder(OnFault::Return))
            .map_err(|raise| format!("the scratch's zeros' blit encoder: {raise}"))?
            .ok_or("no blit encoder for the scratch's zeros")?;
        let fill = BlitCmd::FillBuffer {
            buffer,
            range: NsRange {
                location: 0,
                length,
            },
            value: 0,
        };
        let fault = |what: &str, raise: Raise| format!("the scratch's zeros: {what}: {raise}");
        blit.encode(OnFault::Return, fill)
            .map_err(|r| fault("the fill", r))?;
        (blit.encode(OnFault::Return, BlitCmd::EndEncoding)).map_err(|r| fault("its end", r))?;
        cb.commit(OnFault::Return)
            .map_err(|r| fault("the commit", r))?;
        (cb.wait_until_completed(OnFault::Return)).map_err(|r| fault("the wait", r))?;
        if cb.status() != CommandBufferStatus::COMPLETED {
            let error = cb.error().map(|e| format!("{e:?}")).unwrap_or_default();
            return Err(format!("the scratch's zeros failed {error}"));
        }
        Ok(())
    }

    /// Copies `fills` from `source`, the scratch, into `destinations`, the luma plane's buffer and
    /// the chroma's, by the kernel's `pipeline`, and waits for it, on the copier's queue. `Some`
    /// says why nothing was submitted; `Err` is a copy that failed once submitted.
    fn write(
        &mut self,
        pipeline: &P::ComputePipeline,
        source: &P::Buffer,
        destinations: [&P::Buffer; 2],
        fills: &[PlaneFill; 2],
    ) -> Result<Option<String>, BackendError> {
        if self.arguments.is_none() {
            let device = &self.pipeline.device;
            let made = device.new_buffer_of_length(OnFault::Return, 2 * ARGUMENTS_APART, 0);
            match made {
                Ok(Some(buffer)) => self.arguments = Some(buffer),
                Ok(None) => return Ok(Some("no buffer for the kernel's arguments".to_owned())),
                Err(raise) => return Ok(Some(format!("the arguments' buffer: {raise}"))),
            }
        }
        if self.queue().is_none() {
            return Ok(Some("no command queue".to_owned()));
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
            destinations,
            sources: [source, source],
        };
        if let Some(why) = encode_planes::<P>(&cb, &planes, fills) {
            return Ok(Some(why));
        }
        if let Err(raise) = cb.commit(OnFault::Return) {
            return Ok(Some(format!("its commit: {raise}")));
        }
        let failed = match cb.wait_until_completed(OnFault::Return) {
            Err(raise) => Some(format!("its wait: {raise}")),
            Ok(()) if cb.status() != CommandBufferStatus::COMPLETED => {
                let error = cb.error().map(|e| format!("{e:?}")).unwrap_or_default();
                Some(format!("the copy failed {error}"))
            }
            Ok(()) => None,
        };
        match failed {
            Some(failed) => Err(format!(
                "the GPU's copy of a picture from its scratch into its caches: {failed}"
            )
            .into()),
            None => Ok(None),
        }
    }
}

impl<P: Platform> Drop for ScratchCopier<P> {
    fn drop(&mut self) {
        if self.copied == 0 {
            return;
        }
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        let mean = self.total / u32::try_from(self.copied).unwrap_or(u32::MAX);
        log::info!(
            "hardware video: {NAME}: a decode session's pictures copied by the GPU into their \
             device caches from a scratch the driver's video processing wrote: {}, {:.3} ms each \
             from the driver's write to the copy's end, {:.3} ms at most",
            self.copied,
            ms(mean),
            ms(self.longest)
        );
    }
}

#[cfg(test)]
mod tests;
