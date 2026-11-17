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
//! only once the copy is done on the GPU, however long the GPU takes (the copier's wait has no
//! bound of its own): a GPU slowed by other work slows the picture, as a Mac's does, and fails it
//! only where the device is lost.
//!
//! On Windows the decode surface is a shared Direct3D 12 texture of the VA crate's pool
//! (`Frame::texture`), imported by its NT handle as one two-plane image by the Vulkan platform's
//! copier there, which copies each plane's aspect the same way; the encode sessions' fills are not
//! made there, so an encode's frames go by the VA crate's upload.
//!
//! Each session logs at its end how many pictures went this way, and their GPU time.
//!
//! An encode session's frames go the other way the same way (`super::encode`'s module doc): the
//! runtime device's GPU copies the source's planes, where they lie, into an input surface of the
//! VA crate's encoder, exported as dma-bufs for writing, with the copier's fill
//! (`PlaneCopier::fill`), which fills the margin to the coded size as the VA crate's upload
//! fills it. The fill waits for the GPU as the copy does; the encode follows once it is done.
//! Each encode session logs at its end how many frames went this way, and their GPU time.

use std::sync::Arc;
use std::time::Duration;

use reims_vgpu_video::core::backend::DeviceWrite;
use reims_vgpu_video::vaapi::Frame;
#[cfg(unix)]
use reims_vgpu_video::vaapi::InputSurface;
use reims_vgpu_vulkan::VulkanPlatform;
use reims_vgpu_vulkan::gpu::device::VulkanDevice;
use reims_vgpu_vulkan::os::{CopyError, PlaneCopier, PlaneCopy};
#[cfg(unix)]
use reims_vgpu_vulkan::os::{DmaBufPlane, PlaneFill, PlaneFormat};
#[cfg(windows)]
use reims_vgpu_vulkan::os::{SharedPlanes, TextureFormat};

use super::super::video_host::{BackendError, OutputPlane};
use super::{GpuCopier, GpuCopies, GpuFiller, NAME, Room, crop_extent};

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

    #[cfg(unix)]
    fn filler(&self) -> Result<Box<dyn GpuFiller<VulkanPlatform>>, String> {
        let copier = PlaneCopier::new(&self.device)?;
        Ok(Box::new(RuntimeFiller {
            copier,
            times: Times::default(),
        }))
    }

    #[cfg(windows)]
    fn filler(&self) -> Result<Box<dyn GpuFiller<VulkanPlatform>>, String> {
        Err("the GPU fills no encoder's input surface on Windows".to_owned())
    }
}

/// The GPU times of a session's copies: how many, their sum over those the queue timed, and the
/// longest.
#[cfg(unix)]
#[derive(Default)]
struct Times {
    count: u64,
    gpu: Duration,
    timed: u64,
    longest: Duration,
}

#[cfg(unix)]
impl Times {
    fn add(&mut self, gpu: Option<Duration>) {
        self.count += 1;
        if let Some(gpu) = gpu {
            self.gpu += gpu;
            self.timed += 1;
            self.longest = self.longest.max(gpu);
        }
    }

    /// Their count, mean GPU time and longest, as the log lines say them.
    fn said(&self) -> String {
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        let mean = self.gpu / u32::try_from(self.timed.max(1)).unwrap_or(u32::MAX);
        format!(
            "{}, {:.3} ms of GPU time each, {:.3} ms at most ({} timed)",
            self.count,
            ms(mean),
            ms(self.longest),
            self.timed
        )
    }
}

/// One encode session's fills of its encoder's input surfaces, and their times.
#[cfg(unix)]
struct RuntimeFiller {
    copier: PlaneCopier,
    times: Times,
}

#[cfg(unix)]
impl GpuFiller<VulkanPlatform> for RuntimeFiller {
    fn fill(
        &mut self,
        surface: &InputSurface,
        luma: &OutputPlane<VulkanPlatform>,
        chroma: &OutputPlane<VulkanPlatform>,
        size: (u32, u32),
        sample: u32,
    ) -> Result<DeviceWrite, BackendError> {
        let formats = match sample {
            1 => [PlaneFormat::R8, PlaneFormat::Rg8],
            2 => [PlaneFormat::R16, PlaneFormat::Rg16],
            bytes => return Ok(DeviceWrite::Declined(format!("samples of {bytes} bytes"))),
        };
        let export = surface.export();
        let (coded_width, coded_height) = export.size();
        let (width, height) = size;
        // A key a plane of this export: its life's address, unique while it lives.
        let key = u64::try_from(Arc::as_ptr(export).addr()).unwrap_or(0);
        let planes = [
            (luma, [width, height], [coded_width, coded_height]),
            (
                chroma,
                [width / 2, height / 2],
                [coded_width / 2, coded_height / 2],
            ),
        ];
        let fills: Vec<PlaneFill<'_>> = (planes.iter().zip(formats).enumerate())
            .map(|(i, (&(plane, extent, size), format))| {
                let exported = export.plane(i);
                PlaneFill {
                    into: DmaBufPlane {
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
                    from: &plane.buffer,
                    offset: plane.offset,
                    bytes_per_row: plane.bytes_per_row,
                    extent,
                }
            })
            .collect();
        match self.copier.fill(&fills, Box::new(Arc::clone(export))) {
            Ok(gpu) => {
                self.times.add(gpu);
                Ok(DeviceWrite::Written)
            }
            Err(CopyError::Refused(why)) => Ok(DeviceWrite::Declined(why)),
            Err(CopyError::Failed(why)) => Err(format!("the GPU's copy of a frame: {why}").into()),
        }
    }
}

#[cfg(unix)]
impl Drop for RuntimeFiller {
    fn drop(&mut self) {
        if self.times.count > 0 {
            log::info!(
                "hardware video: {NAME}: an encode session's frames copied by the GPU into the \
                 encoder's input surfaces: {}",
                self.times.said()
            );
        }
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

impl RuntimeCopier {
    /// What a copy's answer makes of the picture: written, its GPU time counted; declined where
    /// the copier refused it, nothing submitted; or failed.
    fn landed(
        &mut self,
        copied: Result<Option<Duration>, CopyError>,
    ) -> Result<DeviceWrite, BackendError> {
        match copied {
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

#[cfg(windows)]
impl GpuCopier<VulkanPlatform> for RuntimeCopier {
    fn copy(
        &mut self,
        frame: &Frame,
        luma: &OutputPlane<VulkanPlatform>,
        chroma: &OutputPlane<VulkanPlatform>,
    ) -> Result<DeviceWrite, BackendError> {
        let (format, sample) = match frame.bit_depth() {
            8 => (TextureFormat::Nv12, 1),
            10 => (TextureFormat::P010, 2),
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
        let Some(texture) = frame.texture() else {
            return Ok(DeviceWrite::Declined(
                "the decode surface is no shared texture".to_owned(),
            ));
        };
        let (texture_width, texture_height) = texture.size();
        let (x, y) = frame.crop_origin();
        let source = SharedPlanes {
            // A key a texture: its life's address, unique while it lives.
            key: u64::try_from(Arc::as_ptr(&texture).addr()).unwrap_or(0),
            alive: texture.alive(),
            handle: texture.handle(),
            format,
            width: texture_width,
            height: texture_height,
        };
        let copies = [
            PlaneCopy {
                plane: 0,
                origin: [x, y],
                extent: [width, height],
                to: &luma.buffer,
                offset: luma.offset,
                bytes_per_row: luma.bytes_per_row,
            },
            PlaneCopy {
                plane: 1,
                origin: [x / 2, y / 2],
                extent: [width.div_ceil(2), height.div_ceil(2)],
                to: &chroma.buffer,
                offset: chroma.offset,
                bytes_per_row: chroma.bytes_per_row,
            },
        ];
        let copied = self.copier.copy(&source, &copies, Box::new(frame.hold()));
        self.landed(copied)
    }
}

#[cfg(unix)]
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
        let copied = self.copier.copy(&copies, Box::new(frame.hold()));
        self.landed(copied)
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
