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
//! VA-API decoded it into ([`DrawnPlanes::draw`]). The landing waits for the decode, tells of it
//! done (`Landing::decode_done`), and then draws: a command buffer of the decode alone completes
//! as the decode is done, with the draw running behind it, and every reader of the output waits
//! for the draw (`decode`'s video module docs, "Completed at the decode"). The draw submits its
//! write and waits for it on the landing's thread, and the landing then says the writes are done
//! (`ImportedSurfaceMemory::external_writes_done`), before the session signals the decode. The
//! targets are shared with no keyed mutex: the wait and that acquire order the write before the
//! platform's reads.
//!
//! A new cache's plane shapes owe a fill from its buffer, its make's zeros. A decode that takes
//! the cache's first fill as met takes theirs as met too (`decode::surface_cache`'s
//! `exported`), so the textures are to read zeros wherever its picture's draw does not write
//! them, as that fill would have left them ([`DrawnPlanes::fill_taken`]): the draw clears them
//! to zeros first where its picture is smaller than the planes (reims-vgpu-video's
//! `PlaneTargets::owe_zeros`). Where no picture is drawn, they are cleared alone
//! ([`DrawnPlanes::unwritten`]): a draw that fails, before the CPU's copy, and a command buffer
//! whose decode answered nothing, at its finish. A fill of the shapes from the buffer since
//! leaves them owing nothing ([`DrawnPlanes::filled`]). A picture smaller than the planes owes
//! the zeros whatever the textures held, another picture's: its draw clears them first, so that
//! no texel past it keeps what it held (`video_host`'s module docs, "A short picture").
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

    /// Draws `frame`'s crop into the textures once its decode is done, telling `decoded` between
    /// the two (the module doc), over zeros where the crop is smaller than them, and counts it.
    /// `Err` says why it was not drawn, the textures
    /// cleared to the zeros they owe first ([`Self::unwritten`]); `decoded` is told only of a
    /// decode that is done.
    pub(crate) fn draw(&self, frame: &Frame, decoded: &mut dyn FnMut()) -> Result<(), String> {
        let drawn = frame
            .sync()
            .map_err(|error| error.to_string())
            .and_then(|()| {
                decoded();
                let texture = (frame.texture()).ok_or("the picture lies in no shared texture")?;
                let extent = (frame.width(), frame.height());
                let size = self.targets.size();
                if extent.0 < size.0 || extent.1 < size.1 {
                    self.targets.owe_zeros();
                }
                self.draw
                    .draw(&texture, frame.crop_origin(), &self.targets, extent)
            });
        if let Err(why) = drawn {
            self.unwritten();
            return Err(why);
        }
        reims_vgpu_vulkan::answer::count_kind(DRAWN_CODE);
        reims_vgpu_vulkan::answer::count_kind_by(DRAWN_KIB_CODE, crop_bytes(frame) >> 10);
        Ok(())
    }

    /// A decode took the fill the cache's plane shapes owe as met: the textures are to read
    /// zeros wherever no picture's draw writes them (the module doc).
    pub(crate) fn fill_taken(&self) {
        self.targets.owe_zeros();
    }

    /// A fill of the plane shapes' images from the cache's buffer was made: the textures owe no
    /// zeros (the module doc).
    pub(crate) fn filled(&self) {
        self.targets.written();
    }

    /// Whether the textures owe zeros ([`Self::fill_taken`]).
    pub(crate) fn owes_zeros(&self) -> bool {
        self.targets.owes_zeros()
    }

    /// Clears the textures to the zeros they owe, where no picture is drawn into them (the module
    /// doc), and returns once the GPU did; nothing where they owe none. A failure is logged.
    pub(crate) fn unwritten(&self) {
        if let Err(why) = self.draw.zero_owed(&self.targets) {
            log::warn!(
                target: "reims_vgpu::graphics", kind:% = why;
                "a video cache's Direct3D 11 textures were not cleared to the zeros they owe: \
                 {why}"
            );
        }
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
