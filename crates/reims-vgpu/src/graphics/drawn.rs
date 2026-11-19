//! A video cache's planes over Direct3D 11 textures that the host's video draws its pictures
//! into, on Windows (`decode::surface_cache`'s `exported`, `vaapi`).
//!
//! A cache made for the host's video takes a surface at the first texture the guest makes over
//! one of its planes whole, as on Unix hosts. On Windows, the surface's planes are the platform's
//! textures over two shared Direct3D 11 textures: a luma texture of `R8_UNORM` (8-bit samples)
//! or `R16_UNORM` (16-bit samples), and a chroma texture of `R16_UINT` or `R32_UINT` whose
//! element is one packed pair. reims-vgpu-video's `PlaneDraw` makes them ([`DrawnPlanes::new`])
//! and the platform imports them (`GpuDevice::import_d3d11_surface`). A decode whose planes are
//! the surface's then lands its picture by one draw from the shared Direct3D 12 texture that
//! VA-API decoded it into ([`DrawnPlanes::draw`]). The draw holds the targets' keyed mutexes
//! across its write, waits for it on the landing's thread and releases them, and the landing then
//! says the writes are done (`ImportedSurfaceMemory::external_writes_done`).
//!
//! Each draw is counted by picture ([`DRAWN_CODE`]) and by the KiB of its planes' crop
//! ([`DRAWN_KIB_CODE`]). A real Mac decodes straight into the surface the guest reads, so the
//! draw is one GPU pass per picture that a Mac does not make.

use std::sync::Arc;

use reims_vgpu_video::core::device_memory::MemoryIdentity;
use reims_vgpu_video::vaapi::{Frame, PlaneDraw, PlaneTargets};
use reims_vgpu_vulkan::core::platform::Handle as _;
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::resource::ImportedSurfaceMemory;

/// The code of a picture drawn into a cache's planes over Direct3D 11 textures (the module doc),
/// counted per picture.
pub(crate) const DRAWN_CODE: &str = "video-pictures-drawn";

/// The code of the KiB of the planes' crops drawn (the module doc), counted per picture.
pub(crate) const DRAWN_KIB_CODE: &str = "video-drawn-kib";

/// The Direct3D 11 textures under a cache's planes, and the draws that write a picture into them.
pub(crate) struct DrawnPlanes {
    /// The draws, one set per GPU's video.
    draw: Arc<PlaneDraw>,
    /// The textures.
    targets: PlaneTargets,
}

impl DrawnPlanes {
    /// The textures of a picture of `size`, of 16-bit samples where `ten_bit`, made by `draw`;
    /// `Err` says why none were made.
    pub(crate) fn new(
        draw: &Arc<PlaneDraw>,
        size: (u32, u32),
        ten_bit: bool,
    ) -> Result<DrawnPlanes, String> {
        Ok(DrawnPlanes {
            draw: Arc::clone(draw),
            targets: draw.targets(size, ten_bit)?,
        })
    }

    /// The textures.
    pub(crate) fn targets(&self) -> &PlaneTargets {
        &self.targets
    }

    /// Draws `frame`'s crop into the textures once its decode is done (the module doc), and
    /// counts it. `Err` says why it was not drawn.
    pub(crate) fn draw(&self, frame: &Frame) -> Result<(), String> {
        frame.sync().map_err(|error| error.to_string())?;
        let texture = (frame.texture()).ok_or("the picture lies in no shared texture")?;
        let extent = (frame.width(), frame.height());
        self.draw
            .draw(&texture, frame.crop_origin(), &self.targets, extent)?;
        reims_vgpu_vulkan::answer::count_kind(DRAWN_CODE);
        reims_vgpu_vulkan::answer::count_kind_by(DRAWN_KIB_CODE, crop_bytes(frame) >> 10);
        Ok(())
    }
}

/// The bytes of `frame`'s crop in its two planes: the luma's samples and the chroma's pairs.
fn crop_bytes(frame: &Frame) -> u64 {
    let sample: u64 = if frame.bit_depth() > 8 { 2 } else { 1 };
    let (width, height) = (u64::from(frame.width()), u64::from(frame.height()));
    let luma = width * height * sample;
    let chroma = width.div_ceil(2) * height.div_ceil(2) * 2 * sample;
    luma + chroma
}

/// One plane of a video cache's surface over Direct3D 11 textures, as a decode's picture lands
/// in it (the module doc).
pub(crate) struct DrawnTarget<P: Platform> {
    /// The surface's memory, which says the draws' writes are done.
    pub(crate) memory: Arc<dyn ImportedSurfaceMemory>,
    /// The textures under both planes, and their draws.
    pub(crate) planes: Arc<DrawnPlanes>,
    /// The surface's identity, which lives as long as its cache.
    pub(crate) identity: MemoryIdentity,
    /// The plane's image, the cache's plane shape's, which the CPU's copy writes where the draw
    /// is not made.
    pub(crate) texture: P::Texture,
}

impl<P: Platform> DrawnTarget<P> {
    /// Another target of the same plane.
    pub(crate) fn clone_ref(&self) -> Self {
        DrawnTarget {
            memory: Arc::clone(&self.memory),
            planes: Arc::clone(&self.planes),
            identity: self.identity.clone(),
            texture: self.texture.clone_ref(),
        }
    }
}
