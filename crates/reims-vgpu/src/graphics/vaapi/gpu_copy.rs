//! The VA-API arm's second way to land a picture (`super`'s module doc): a copy by the runtime
//! device's GPU from the driver's decode surface into the output's planes, where they lie, at
//! their own rows and offsets: the guest's pages, imported in place, or the surface's device
//! cache. The decode surface's planes are its export as dma-bufs (the VA crate's
//! `Frame::export`), imported as images by the Vulkan platform's copier
//! (`reims_vgpu_vulkan::os::PlaneCopier`), one copier a session, on a runtime queue of its own.
//!
//! The copy writes what the CPU's copy would: the crop of the decoded picture, at each plane's
//! first byte, as many columns and rows as both the picture and the planes hold
//! ([`super::crop_extent`]), and nothing past them, so a guest's planes hold the same bytes either
//! way. It waits for the decode first, as the CPU's copy does, and the session signals the decode
//! only once the copy is done on the GPU.
//!
//! Each session logs at its end how many pictures went this way, and their GPU time.

use std::sync::Arc;
use std::time::Duration;

use reims_vgpu_video::core::backend::DeviceWrite;
use reims_vgpu_video::vaapi::Frame;
use reims_vgpu_vulkan::VulkanPlatform;
use reims_vgpu_vulkan::gpu::device::VulkanDevice;
use reims_vgpu_vulkan::os::{CopyError, DmaBufPlane, PlaneCopier, PlaneCopy, PlaneFormat};

use super::super::video_host::{BackendError, OutputPlane};
use super::{GpuCopier, GpuCopies, NAME, Room, crop_extent};

/// How long a copy waits for the GPU: a picture's copy takes under a millisecond of it.
const COPY_WAIT: Duration = Duration::from_secs(2);

/// The copies of a runtime device's sessions.
pub(super) struct RuntimeCopies {
    device: VulkanDevice,
}

impl RuntimeCopies {
    pub(super) fn new(device: &VulkanDevice) -> Self {
        use reims_vgpu_vulkan::core::platform::Handle as _;
        Self {
            device: device.clone_ref(),
        }
    }
}

impl GpuCopies<VulkanPlatform> for RuntimeCopies {
    fn copier(&self) -> Result<Box<dyn GpuCopier<VulkanPlatform>>, String> {
        let copier = PlaneCopier::new(&self.device)?;
        Ok(Box::new(RuntimeCopier {
            copier,
            copied: 0,
            gpu: Duration::ZERO,
            timed: 0,
            longest: Duration::ZERO,
        }))
    }
}

/// One session's copies, and their counts.
struct RuntimeCopier {
    copier: PlaneCopier,
    /// Pictures copied, and the GPU time of those the queue timed.
    copied: u64,
    gpu: Duration,
    timed: u64,
    longest: Duration,
}

impl GpuCopier<VulkanPlatform> for RuntimeCopier {
    fn copy(
        &mut self,
        frame: &Frame,
        luma: &OutputPlane<VulkanPlatform>,
        chroma: &OutputPlane<VulkanPlatform>,
    ) -> Result<DeviceWrite, BackendError> {
        let (formats, sample) = match frame.bit_depth() {
            8 => ([PlaneFormat::R8, PlaneFormat::Rg8], 1),
            10 => ([PlaneFormat::R16, PlaneFormat::Rg16], 2),
            depth => return Ok(DeviceWrite::Declined(format!("a {depth}-bit frame"))),
        };
        let size = (frame.width(), frame.height());
        let Some((width, height)) = crop_extent(size, sample, Room::of(luma), Room::of(chroma))
        else {
            return Ok(DeviceWrite::Declined("an empty crop".to_owned()));
        };
        // The decode first, as the CPU's copy waits for it: a decode that failed or ran past its
        // deadline fails the picture.
        frame.sync()?;
        let export = match frame.export() {
            Ok(export) => export,
            Err(why) => {
                return Ok(DeviceWrite::Declined(format!(
                    "the decode surface's export: {why}"
                )));
            }
        };
        let (surface_width, surface_height) = export.size();
        let (x, y) = frame.crop_origin();
        // A key a plane of this export: its life's address, unique while it lives.
        let key = u64::try_from(Arc::as_ptr(&export).addr()).unwrap_or(0);
        let planes = [
            (
                luma,
                [x, y],
                [width, height],
                [surface_width, surface_height],
            ),
            (
                chroma,
                [x / 2, y / 2],
                [width.div_ceil(2), height.div_ceil(2)],
                [surface_width.div_ceil(2), surface_height.div_ceil(2)],
            ),
        ];
        let copies: Vec<PlaneCopy<'_>> = (planes.iter().zip(formats).enumerate())
            .map(|(i, (&(plane, origin, extent, size), format))| {
                let exported = export.plane(i);
                PlaneCopy {
                    source: DmaBufPlane {
                        key: key.wrapping_mul(2).wrapping_add(u64::from(i == 1)),
                        alive: export.alive(),
                        fd: export.fd(i),
                        modifier: exported.modifier,
                        offset: exported.offset,
                        pitch: exported.pitch,
                        format,
                        width: size[0],
                        height: size[1],
                    },
                    origin,
                    extent,
                    to: &plane.buffer,
                    offset: plane.offset,
                    bytes_per_row: plane.bytes_per_row,
                }
            })
            .collect();
        match self.copier.copy(&copies, COPY_WAIT, Box::new(frame.hold())) {
            Ok(gpu) => {
                self.copied += 1;
                if let Some(gpu) = gpu {
                    self.gpu += gpu;
                    self.timed += 1;
                    self.longest = self.longest.max(gpu);
                }
                Ok(DeviceWrite::Written)
            }
            Err(CopyError::Refused(why)) => Ok(DeviceWrite::Declined(why)),
            Err(CopyError::Failed(why)) => {
                Err(format!("the GPU's copy of a picture: {why}").into())
            }
        }
    }
}

impl Drop for RuntimeCopier {
    fn drop(&mut self) {
        if self.copied == 0 {
            return;
        }
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        let mean = self.gpu / u32::try_from(self.timed.max(1)).unwrap_or(u32::MAX);
        log::info!(
            "hardware video: {NAME}: a session's pictures copied by the GPU into their outputs: \
             {}, {:.3} ms of GPU time each, {:.3} ms at most ({} timed)",
            self.copied,
            ms(mean),
            ms(self.longest),
            self.timed
        );
    }
}
