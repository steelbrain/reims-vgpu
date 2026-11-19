//! The rt platform's GPU copies of the VA-API arm's pictures on Windows (`super::open_on_rt`):
//! none yet. A picture there is decoded into a shared Direct3D 12 texture of the VA crate's pool
//! (`Frame::texture`), which rt imports no image of, so [`RtCopies`]' copier declines each
//! picture at the GPU's copy, the module doc's second way, and counts it ([`DECLINED_CODE`]); the
//! picture then lands by the CPU's copy, the third way: read back by `vaGetImage` and copied into
//! the session's staging, which is copied into the output's device cache.
//!
//! **Measured** (2026-11-19, Intel UHD 630, Vulkan driver 101.2145, Mesa 26.2.7's `vaon12`): the
//! Vulkan platform's GPU copy of a 1080p NV12 picture took 1.47 ms at the mean, a 4K NV12
//! picture's 4.18 ms and a 4K P010 picture's 10.36 ms; the CPU's copy took 18.77 ms, 68.35 ms
//! and 79.28 ms, of which the read back took 15.09 ms, 57.00 ms and 60.49 ms.

use reims_vgpu_video::core::backend::DeviceWrite;
use reims_vgpu_video::vaapi::Frame;
use reims_vgpu_vulkan::RtPlatform;

use super::super::video_host::{BackendError, OutputPlane};
use super::{GpuCopier, GpuCopies, GpuFiller};

/// The code of a picture the rt platform's GPU copy declined on Windows, which the CPU copied
/// (docs/conventions.md, "Each degradation is logged once per boot, then counted").
pub(crate) const DECLINED_CODE: &str = "rt-video-gpu-copy-declined";

/// Why the rt platform's GPU copy declines a picture on Windows.
const WHY: &str = "the rt platform imports no shared Direct3D 12 texture to copy from";

/// The rt platform's GPU copies on Windows: a copier that declines each picture (module doc), and
/// no encoder's fills, as the arm answers no encoder there.
pub(super) struct RtCopies;

impl GpuCopies<RtPlatform> for RtCopies {
    fn copier(&self) -> Result<Box<dyn GpuCopier<RtPlatform>>, String> {
        Ok(Box::new(RtCopier))
    }

    fn filler(&self) -> Result<Box<dyn GpuFiller<RtPlatform>>, String> {
        Err(WHY.to_owned())
    }
}

/// One session's copier: each picture declined and counted (module doc).
struct RtCopier;

impl GpuCopier<RtPlatform> for RtCopier {
    fn copy(
        &mut self,
        _frame: &Frame,
        _luma: &OutputPlane<RtPlatform>,
        _chroma: &OutputPlane<RtPlatform>,
    ) -> Result<DeviceWrite, BackendError> {
        Ok(declined())
    }
}

/// A picture's GPU copy declined, and counted (module doc).
fn declined() -> DeviceWrite {
    reims_vgpu_vulkan::answer::count_kind(DECLINED_CODE);
    DeviceWrite::Declined(WHY.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A session has a copier, whose every picture is declined with why and counted by its code,
    /// so that the CPU's copy lands each; and its fills are refused with why (module doc).
    #[test]
    fn each_picture_is_declined_with_why_and_counted_and_the_fills_are_refused() {
        let counted = || {
            (reims_vgpu_vulkan::answer::counted_kinds().into_iter())
                .find(|&(code, _)| code == DECLINED_CODE)
                .map_or(0, |(_, n)| n)
        };
        assert!(RtCopies.copier().is_ok());
        assert_eq!(RtCopies.filler().err().as_deref(), Some(WHY));
        let before = counted();
        for n in 1..=2 {
            assert_eq!(declined(), DeviceWrite::Declined(WHY.to_owned()));
            assert_eq!(counted(), before + n);
        }
    }
}
