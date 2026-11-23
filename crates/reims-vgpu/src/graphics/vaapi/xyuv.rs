//! A VideoToolbox decoder's 4:4:4 pictures, which the driver decodes into XYUV surfaces of packed
//! pixels, written by the runtime device's GPU into the two planes of the surface the guest lent,
//! on Linux ([`XyuvGathers`]), as NVDEC's are (`super::super::gather`): where the display's import
//! check passed XYUV (the VA crate's `ImportWrites::xyuv`), and else read back and split by the
//! CPU, as before (`super::VaGuestDecoder`).
//!
//! - **The scratch** is a buffer of the decoder's in a dma-buf block of its own
//!   (`new_dma_buf_buffer_of_length_padding_unobserved`), made at its first 4:4:4 picture and made
//!   again, as large as both, for a larger one: a picture's pixels from its first byte, four bytes
//!   each, in rows of the even width's bytes rounded up to 256 (`TARGET_ALIGNMENT`), so that a
//!   driver that writes an import's rows of whole alignments alone, from the dma-buf's first byte
//!   (iHD's, docs/video/bugs.md U26), takes it. It is made as zeros, waited for, before the driver
//!   first writes it, as `gpu_copy`'s scratch is. No guest message names it and no guest maps it.
//! - **A picture** is written into the scratch by the driver's video processing
//!   (`Frame::write_4_4_4_to`), its crop's even width and height from the crop's first pixel,
//!   through the scratch's memory as [`CacheBytes`]; acquired (`Buffer::external_write_done`);
//!   then `videoGather444` writes the crop's samples into the surface's two planes through the
//!   Shared buffer over its pages that VideoToolbox's decoder hands for a 4:4:4 surface
//!   (`super::super::vt`), and is waited for: the rows and bytes that the CPU's split of the read
//!   back writes (`ReadBack::copy_4_4_4_into`), and nothing else of the surface.
//! - **Refusals** come before anything is written into the surface, and the picture is then read
//!   back and split by the CPU, the reason logged once a decoder: no buffer over the surface, a
//!   crop the kernel does not write (`gather::gathers`), the driver's decline of the scratch
//!   (rows of a width whose bytes are not whole alignments, where it writes those alone; an odd
//!   size past the decoded picture), and a pass that failed. The device's refusal to make the
//!   scratch, or to acquire it, ends the route for the decoder ([`Gathered::Ended`],
//!   [`XyuvGathers::takes_pictures`]), whose later 4:4:4 surfaces are then handed with no buffer
//!   over them.
//!
//! Each picture written so is counted (`video-pictures-gpu-gathered`, the gatherer's), and each
//! decoder logs at its end how many went so, and the time from the driver's write to the pass's
//! end.

use std::sync::Arc;
use std::time::Duration;

use reims_vgpu_video::core::backend::DeviceWrite;
use reims_vgpu_video::core::device_memory::{MemoryLife, TargetPlane};
use reims_vgpu_video::core::runtime::Runtime;
use reims_vgpu_video::vaapi::{
    Error as VaError, Frame, GuestWrites, ImportWrites, PackedTarget, TARGET_ALIGNMENT,
};
use reims_vgpu_vulkan::core::platform::Handle as _;
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
use reims_vgpu_vulkan::core::raise::OnFault;
use reims_vgpu_vulkan::core::sync::Mutex;

use super::super::gather::{GatherPipeline, Gatherer, Planar, gathers};
use super::{CacheBytes, CacheMemory, CacheOf, NAME, STORAGE_PRIVATE};

/// A decoder's route for its 4:4:4 pictures by the GPU, where the display's import check passed
/// XYUV (module doc): the trial's own answer decides. `Err` says why there is none, for the log of
/// the CPU's splits.
pub(super) fn route<P: Platform>(
    writes: &ImportWrites,
    pipeline: &Arc<GatherPipeline<P>>,
    runtime: &Arc<dyn Runtime>,
) -> Result<XyuvGathers<P>, String> {
    match writes.xyuv() {
        Ok(_) => Ok(XyuvGathers::new(Arc::clone(pipeline), Arc::clone(runtime))),
        Err(why) => Err(format!(
            "the driver's video processing writes no XYUV import right: {why}"
        )),
    }
}

/// What became of a picture offered to [`XyuvGathers::land`].
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Gathered {
    /// Written into the surface.
    Written,
    /// Not written, with why: the CPU splits it, and the route takes the next.
    Declined(String),
    /// Not written, with why, nor any later picture of the decoder: the device refused the
    /// scratch.
    Ended(String),
}

impl Gathered {
    /// `Ok` where the picture was written, else why not, for the log of the CPU's split.
    pub(super) fn written(self) -> Result<(), String> {
        match self {
            Self::Written => Ok(()),
            Self::Declined(why) => Err(why),
            Self::Ended(why) => Err(format!("the GPU's writes ended: {why}")),
        }
    }
}

/// How a scratch of a picture of an even `width` by `height` lies (module doc): its bytes from one
/// row to the next, and its rows.
fn scratch_shape((width, height): (u32, u32)) -> (u64, u64) {
    let row = u64::from(width) * PackedTarget::PIXEL_BYTES as u64;
    (row.next_multiple_of(TARGET_ALIGNMENT), u64::from(height))
}

/// A decoder's scratch (module doc).
struct Scratch<P: Platform> {
    pitch: u64,
    rows: u64,
    buffer: P::Buffer,
    /// Its memory as the driver's video processing writes it, by `life`'s identity, so that the
    /// display's import of it goes as the scratch does.
    bytes: CacheBytes<P>,
    _life: MemoryLife,
}

/// One decoder's GPU writes of its 4:4:4 pictures (module doc).
pub(super) struct XyuvGathers<P: Platform> {
    gatherer: Gatherer<P>,
    scratch: Option<Scratch<P>>,
    /// Scratches whose picture's write failed, which the driver may write still: kept out of use
    /// while the decoder lives.
    failed: Vec<Scratch<P>>,
    /// Why the route ended ([`Gathered::Ended`]), after which it takes no picture.
    ended: Option<String>,
    /// Pictures written, and the time from the driver's write to the pass's end, on `runtime`'s
    /// clock (the video's runtime's).
    gathered: u64,
    runtime: Arc<dyn Runtime>,
    total: Duration,
    longest: Duration,
}

impl<P: Platform> XyuvGathers<P> {
    /// A decoder's route on `pipeline`'s device, timed on `runtime`'s clock: nothing is made
    /// before its first picture.
    fn new(pipeline: Arc<GatherPipeline<P>>, runtime: Arc<dyn Runtime>) -> Self {
        Self {
            gatherer: Gatherer::new(pipeline),
            scratch: None,
            failed: Vec::new(),
            ended: None,
            gathered: 0,
            runtime,
            total: Duration::ZERO,
            longest: Duration::ZERO,
        }
    }

    /// Whether it takes the decoder's 4:4:4 pictures: until the device refuses its scratch
    /// ([`Gathered::Ended`]).
    pub(super) fn takes_pictures(&self) -> bool {
        self.ended.is_none()
    }

    /// Writes `frame`, a 4:4:4 one, into `planes`, the luma and chroma planes of the surface
    /// `surface` lies over from its first byte, by the GPU (module doc). The `Err` is the frame's,
    /// as `Frame::write_4_4_4_to`'s: a decode that failed, or a write past its deadline.
    pub(super) fn land(
        &mut self,
        frame: &Frame,
        imports: &GuestWrites,
        planes: &[TargetPlane; 2],
        surface: Option<&P::Buffer>,
    ) -> Result<Gathered, VaError> {
        let declined = |why: &str| Ok(Gathered::Declined(why.to_owned()));
        if let Some(why) = &self.ended {
            return Ok(Gathered::Declined(format!("the GPU's writes ended: {why}")));
        }
        let Some(surface) = surface else {
            return declined("no buffer of the device's over the surface");
        };
        if !frame.is_4_4_4() {
            return declined("a 4:2:0 picture");
        }
        let crop = (frame.width(), frame.height());
        let even = (crop.0.next_multiple_of(2), crop.1.next_multiple_of(2));
        if let Err(why) = self.scratch_for(even) {
            return Ok(self.end(why));
        }
        let Some(scratch) = self.scratch.as_ref() else {
            return declined("no scratch");
        };
        let length = scratch.pitch * scratch.rows;
        let passes = match gathers(Planar::xyuv(scratch.pitch, length), crop, planes, surface.length())
        {
            Ok(passes) => passes,
            Err(why) => return Ok(Gathered::Declined(why)),
        };
        let start = self.runtime.now();
        let Ok(pitch) = usize::try_from(scratch.pitch) else {
            return declined("a scratch past the address space");
        };
        let target = match PackedTarget::new(&scratch.bytes, 0, pitch, even) {
            Ok(target) => target,
            Err(why) => return Ok(Gathered::Declined(format!("the scratch's target: {why}"))),
        };
        match frame.write_4_4_4_to(&target, imports) {
            Ok(DeviceWrite::Written) => {}
            Ok(DeviceWrite::Declined(why)) => {
                return Ok(Gathered::Declined(format!(
                    "the driver's video processing into its scratch: {why}"
                )));
            }
            Err(error) => {
                drop(target);
                self.failed.extend(self.scratch.take());
                return Err(error);
            }
        }
        if let Err(why) = scratch.buffer.external_write_done() {
            return Ok(self.end(format!("the scratch's acquire: {why}")));
        }
        let source = scratch.buffer.clone_ref();
        if let Err(why) = self.gatherer.gather(&source, surface, &passes) {
            return Ok(Gathered::Declined(why));
        }
        let took = self.runtime.elapsed_since(start);
        self.gathered += 1;
        self.total += took;
        self.longest = self.longest.max(took);
        Ok(Gathered::Written)
    }

    /// Ends the route, for `why` ([`Gathered::Ended`]): its scratches are kept while it lives.
    fn end(&mut self, why: String) -> Gathered {
        self.ended = Some(why.clone());
        Gathered::Ended(why)
    }

    /// Makes the decoder's scratch for a picture of an even `size`, where it has none as large
    /// (module doc): `Err` says why none is made.
    fn scratch_for(&mut self, size: (u32, u32)) -> Result<(), String> {
        let (pitch, rows) = scratch_shape(size);
        if (self.scratch.as_ref()).is_some_and(|s| s.pitch >= pitch && s.rows >= rows) {
            return Ok(());
        }
        let (pitch, rows) = match self.scratch.take() {
            Some(kept) => (kept.pitch.max(pitch), kept.rows.max(rows)),
            None => (pitch, rows),
        };
        let length = pitch.checked_mul(rows).ok_or("a scratch past 2^64 bytes")?;
        let made = self.gatherer.device().new_dma_buf_buffer_of_length_padding_unobserved(
            OnFault::Return,
            length,
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
        self.gatherer.zero(&buffer, length)?;
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
            pitch,
            rows,
            buffer,
            bytes,
            _life: life,
        });
        Ok(())
    }
}

impl<P: Platform> Drop for XyuvGathers<P> {
    fn drop(&mut self) {
        if self.gathered == 0 {
            return;
        }
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        log::info!(
            "hardware video: {NAME}: a VideoToolbox session's 4:4:4 pictures written by the GPU \
             into their surfaces' two planes: {}, {:.3} ms each on average from the driver's \
             write to the pass's end, {:.3} ms the longest",
            self.gathered,
            ms(self.total) / self.gathered as f64,
            ms(self.longest)
        );
    }
}

#[cfg(test)]
mod tests;
