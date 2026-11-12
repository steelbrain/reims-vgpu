//! [`DisplayRead`]: what a graphics device's display shows, read back for the host
//! ([`super::GraphicsDevice::read_display`]; QEMU's console, and its screendump, read it).
//!
//! **What a read draws.** The window's own draw ([`ScanoutDraw::encode_shown`]): the guest's
//! scanout, or the IOSurface its display pipe presents, through the display's colour, into a
//! texture of the display's mode, the size the guest is told (the window's size in pixels, or
//! 1920 × 1080 without a window), placed and clipped as in a drawable of that size. Where the
//! guest has no scanout and presents no surface the display shows black, and so does the read.
//! While the display's panel is off (the guest's display asleep) the draw clears the texture to
//! black, and the read is black whatever the guest shows.
//! The texture is BGRA8, shared, a render target, and is read
//! once the draw has finished on the GPU: rows of `width * 4` bytes, B, G, R and A in memory.
//!
//! **What it touches.** Nothing the guest or the window sees. Its draws go on a queue of their
//! own, which the platform may share with other command queues ([`ScanoutDraw::for_reads`]),
//! made at the first read over the scanout and the presents the window draws, and answer no
//! present; the mode the guest is told is the screen's, which a read only
//! asks. A read holds the presented surface's block while its draw reads it, as a window's draw
//! does, so a release of the block is answered once that GPU copy has finished. Reads run one
//! at a time, and keep their texture for the next read of the same size.

use std::sync::{Arc, OnceLock};

use reims_vgpu_graphics::Screen;
use reims_vgpu_graphics::region::Region;
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::command::CommandBuffer as _;
use reims_vgpu_vulkan::core::platform::descriptor::TextureDescriptor as _;
use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
use reims_vgpu_vulkan::core::platform::types::CommandBufferStatus;
use reims_vgpu_vulkan::core::raise::OnFault;
use reims_vgpu_vulkan::core::sync::Mutex;
use reims_vgpu_window::PixelRect;

use super::sink::ScanoutDraw;
use super::{Presented, Shown};
use crate::entry::no_catch;
use crate::error::Error;
use crate::platform::ApiPlatform;

/// The largest width or height of a read's texture.
const MAX_READ_SIZE: u32 = 16384;

/// `MTLPixelFormatBGRA8Unorm`.
const BGRA8_UNORM: u64 = 80;
/// `MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget`, a frame target's.
const READ_USAGE: u64 = 0x1 | 0x4;
/// `MTLResourceStorageModeShared`, a frame target's.
const READ_RESOURCE_OPTIONS: u64 = 0;
/// A pixel the display shows where it shows nothing: opaque black, B, G, R, A.
const BLACK: [u8; 4] = [0, 0, 0, 0xff];

/// A read's texture, with its size.
type ReadTexture<P> = ((u32, u32), <P as Platform>::Texture);

/// Reads of what a graphics device's display shows (the module doc).
pub(crate) struct DisplayRead<P: ApiPlatform> {
    /// The platform: its default device, which the decoder's textures are on, makes the reads'.
    platform: Arc<P>,
    /// The device's region, whose scanout the draw imports.
    region: Arc<Region>,
    /// The scanout the guest set, as the screen keeps it.
    shown: Arc<Shown>,
    /// The surface the display pipe presents, and the display's colour.
    presented: Arc<Presented<P>>,
    /// The screen, whose mode a read has.
    screen: Arc<dyn Screen>,
    /// The reads' draw, made at the first; `None` inside where the platform refused it.
    draw: OnceLock<Option<ScanoutDraw<P>>>,
    /// The last read's texture and its size: held through a read, so reads run one at a time.
    texture: Mutex<Option<ReadTexture<P>>>,
}

impl<P: ApiPlatform> DisplayRead<P> {
    /// Reads of the display `screen` shows, over `region`, the scanout `shown` keeps and the
    /// surfaces `presented` shows; nothing is made until the first read.
    pub(crate) fn new(
        platform: &Arc<P>,
        region: &Arc<Region>,
        shown: &Arc<Shown>,
        presented: &Arc<Presented<P>>,
        screen: Arc<dyn Screen>,
    ) -> DisplayRead<P> {
        DisplayRead {
            platform: Arc::clone(platform),
            region: Arc::clone(region),
            shown: Arc::clone(shown),
            presented: Arc::clone(presented),
            screen,
            draw: OnceLock::new(),
            texture: Mutex::new(None),
        }
    }

    /// Reads of the display `self` reads, at `screen`'s mode: the Vulkan platform's tests' read at
    /// the size of a window, which a device without one does not have.
    #[cfg(test)]
    pub(crate) fn at_screen(&self, screen: Arc<dyn Screen>) -> DisplayRead<P> {
        DisplayRead::new(
            &self.platform,
            &self.region,
            &self.shown,
            &self.presented,
            screen,
        )
    }

    /// What the display shows now, into the first `width * height * 4` bytes of `dst`, rows of
    /// `width * 4` bytes, BGRA8 (the module doc), with the display's mode, `(width, height)`,
    /// which is returned whatever the result. [`Error::InvalidArgument`], with nothing written,
    /// where `dst` holds fewer bytes; [`Error::Metal`] where the platform refuses the texture or
    /// the draw; [`Error::Gpu`] where the draw ended with an error. Waits for the draw on the GPU.
    pub(crate) fn read(&self, dst: &mut [u8]) -> ((u32, u32), Result<(), Error>) {
        let mode = self.screen.mode();
        let size = (mode.width, mode.height);
        (size, self.read_at(size, dst))
    }

    /// [`Self::read`] at the mode's size `(width, height)`.
    fn read_at(&self, (width, height): (u32, u32), dst: &mut [u8]) -> Result<(), Error> {
        let row = usize::try_from(width)
            .ok()
            .and_then(|w| w.checked_mul(4))
            .ok_or(Error::InvalidArgument)?;
        let length = usize::try_from(height)
            .ok()
            .and_then(|h| h.checked_mul(row))
            .ok_or(Error::InvalidArgument)?;
        let dst = dst.get_mut(..length).ok_or(Error::InvalidArgument)?;
        if !(1..=MAX_READ_SIZE).contains(&width) || !(1..=MAX_READ_SIZE).contains(&height) {
            return Err(Error::Metal);
        }
        let draw = self
            .draw
            .get_or_init(|| {
                ScanoutDraw::for_reads(
                    Arc::clone(&self.platform),
                    &self.region,
                    &self.shown,
                    &self.presented,
                )
            })
            .as_ref()
            .ok_or(Error::Metal)?;
        let mut kept = self.texture.lock();
        if !kept.as_ref().is_some_and(|(at, _)| *at == (width, height)) {
            *kept = None;
            *kept = Some(((width, height), self.texture_of(draw, width, height)?));
        }
        let Some((_, texture)) = kept.as_ref() else {
            return Err(Error::Metal);
        };
        let whole = PixelRect {
            x: 0,
            y: 0,
            width: u64::from(width),
            height: u64::from(height),
        };
        // The draw, in its refusal scope, as the window's is (`sink`'s module doc, "A refused
        // draw"): a refused one answers that the platform refused the draw.
        let drawn = draw
            .scoped(|| {
                let Some((command_buffer, hold)) = draw.encode_shown(texture, whole) else {
                    return Some(None);
                };
                no_catch(command_buffer.commit(OnFault::Die));
                no_catch(command_buffer.wait_until_completed(OnFault::Die));
                // The draw has read the surface's block.
                drop(hold);
                Some(Some(command_buffer.status()))
            })
            .ok_or(Error::Metal)?;
        let Some(status) = drawn else {
            // No scanout, no surface: the display shows black.
            for pixel in dst.as_chunks_mut::<4>().0 {
                pixel.copy_from_slice(&BLACK);
            }
            return Ok(());
        };
        if status == CommandBufferStatus::ERROR {
            return Err(Error::Gpu);
        }
        self.platform.read_pixels(texture, dst, row);
        Ok(())
    }

    /// A texture of `width` × `height` for the reads, on the draw's device; [`Error::Metal`]
    /// where the platform refuses it.
    fn texture_of(
        &self,
        draw: &ScanoutDraw<P>,
        width: u32,
        height: u32,
    ) -> Result<P::Texture, Error> {
        // Metal's `+0` descriptor goes at the entry's pool.
        let descriptor = no_catch(self.platform.texture_2d_descriptor(
            OnFault::Die,
            BGRA8_UNORM,
            u64::from(width),
            u64::from(height),
            false,
        ));
        descriptor.set_usage(READ_USAGE);
        descriptor.set_resource_options(READ_RESOURCE_OPTIONS);
        no_catch(draw.device.new_texture(OnFault::Die, &descriptor)).ok_or(Error::Metal)
    }
}
