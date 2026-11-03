//! [`ScanoutDraw`]: what a graphics display's window sink draws with, on any platform: the
//! region's pages under the guest's scanout, imported as a no-copy buffer, and a command queue
//! of its own on the platform's default device. Each draw blits the guest's scanout out of the
//! buffer into the drawable's texture, at the image's place, in one command buffer: the GPU reads
//! the guest's pixels in place, and the CPU copies nothing.
//!
//! **Sizes.** The guest sets a scanout of the size the display offers, the window's size in
//! pixels, so the image is the whole drawable and the blit covers it. While the two differ (the
//! window resized, the guest not yet following), the blit copies what fits of the scanout at the
//! image's top left, clipped to the drawable, after a clear of the drawable to black: a blit
//! cannot scale, and the guest's next mode makes the sizes equal again.
//!
//! **A presented surface.** While the display pipe shows an IOSurface (`PRESENT_SURFACE`), the
//! draw puts the texture the decoder made over it into the drawable instead, on the GPU: a copy,
//! texture to texture, where the two have one pixel format (BGRA8, with or without sRGB), and
//! otherwise a draw of the display's present shaders (core's [`DisplayPipelineCache`]), which
//! sample the surface in its own format (WindowServer's display surfaces are 10-bit,
//! `BGR10A2Unorm`) and write the drawable's. The first surface of each format is logged.
//!
//! **The import.** The buffer covers the scanout's pages only: the host pages from the one its
//! first row starts in to the one `stride × height` bytes on ends in, kept to the region. It is
//! made at the first draw of a scanout and again only when the guest moves the scanout or changes
//! its size, so the region's other pages stay the guest's alone: an import pins what it covers
//! (the Vulkan platform's does, on every driver), and the region's untouched pages then cost the
//! host nothing (`Region`). A refused import is logged once for its pages, and the window shows
//! nothing of that scanout. The buffer's deallocator holds the region, which keeps the region
//! mapped while a command buffer still reads it: Metal (or the runtime) runs it once the buffer
//! has gone, after the sink, a later import and every command buffer that used it.

use std::sync::Arc;

use reims_graphics::Scanout;
use reims_graphics::region::Region;
use reims_vgpu_core::host::HostPtr;
use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
use reims_vgpu_core::platform::descriptor::RenderPassDescriptor as _;
use reims_vgpu_core::platform::device::GpuDevice as _;
use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};
use reims_vgpu_core::platform::resource::Texture as _;
use reims_vgpu_core::platform::types::{ClearColor, MtlOrigin, MtlSize};
use reims_vgpu_core::platform::values::ColorAttachmentSet;
use reims_vgpu_core::platform::{Handle, HostRange, Platform};
use reims_vgpu_core::raise::OnFault;
use reims_vgpu_core::sync::Mutex;
use reims_vgpu_window::PixelRect;

use reims_vgpu_core::display::pipeline::{DisplayPipelineCache, PipelineKey};
use reims_vgpu_core::platform::resource::Buffer as _;
use reims_vgpu_core::platform::types::MtlViewport;

use super::{DrawnPresents, Presented, Shown};
use crate::display::no_catch;
use crate::platform::ApiPlatform;

/// `MTLLoadActionClear`.
const LOAD_ACTION_CLEAR: u64 = 2;
/// `MTLStoreActionStore`.
const STORE_ACTION_STORE: u64 = 1;

/// What a graphics display's window sink draws with (the module doc).
pub(crate) struct ScanoutDraw<P: Platform> {
    /// The platform, whose render pass descriptors clear the drawable.
    platform: Arc<P>,
    /// The device the queue and the buffer are on: the platform's default. Only the Metal sink,
    /// on macOS, reads it.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub(crate) device: P::Device,
    /// The sink's command queue, unshared ([`new_command_queue_unshared`]): it puts the guest's
    /// scanout and presented surfaces into the window, each draw on a present's path.
    ///
    /// [`new_command_queue_unshared`]: reims_vgpu_core::platform::device::GpuDevice::new_command_queue_unshared
    pub(crate) queue: P::CommandQueue,
    /// The device's region.
    region: Arc<Region>,
    /// The host's page size, which an import's start and length are multiples of.
    page: u64,
    /// The pages under the scanout last drawn, as a no-copy buffer (the module doc).
    import: Mutex<Option<ScanoutImport<P>>>,
    /// What the window shows.
    shown: Arc<Shown>,
    /// The surface the display pipe shows, if any, in place of the scanout.
    presented: Arc<Presented<P>>,
    /// The present shaders' pipelines and the strip's two buffers, made at the first surface
    /// whose format a copy cannot carry; `None` inside where the platform refused them.
    converter: std::sync::OnceLock<Option<Converter<P>>>,
}

/// The region's pages under a scanout, imported (the module doc).
pub(crate) struct ScanoutImport<P: Platform> {
    /// The first page's offset in the region.
    pub(crate) start: u64,
    /// The pages' length in bytes.
    pub(crate) length: u64,
    /// The pages as a no-copy buffer; `None` where the platform refused them.
    pub(crate) buffer: Option<P::Buffer>,
}

/// What draws a presented surface of another format into the drawable: core's display pipelines
/// and the full-viewport strip they draw (positions at vertex buffer 0, texture coordinates at 1).
struct Converter<P: Platform> {
    /// The display's present pipelines, by the drawable's format.
    pipelines: DisplayPipelineCache<P>,
    /// The strip's positions, float4 × 4.
    positions: P::Buffer,
    /// Its texture coordinates, float2 × 4, the surface's top row at the top.
    coordinates: P::Buffer,
}

/// The strip's positions and texture coordinates, as core's display draws them.
const STRIP_POSITIONS: [f32; 16] = [
    -1.0, -1.0, 0.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
];
const STRIP_COORDINATES: [f32; 8] = [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
/// `MTLPrimitiveTypeTriangleStrip`.
const TRIANGLE_STRIP: u64 = 4;

/// `MTLPixelFormatBGRA8Unorm` and its sRGB form, which copy into each other.
const BGRA8_FORMATS: [u64; 2] = [80, 81];

impl<P: ApiPlatform> ScanoutDraw<P> {
    /// The draw over `region` on `platform`'s default device, sharing `shown` with the window's
    /// screen; `None` where the platform has no device, or refuses a queue. It imports nothing
    /// until it draws a scanout.
    pub(crate) fn new(
        platform: Arc<P>,
        region: &Arc<Region>,
        shown: &Arc<Shown>,
        presented: &Arc<Presented<P>>,
    ) -> Option<ScanoutDraw<P>> {
        let device = platform.metal_device(None)?;
        let queue = device.new_command_queue_unshared()?;
        // SAFETY: sysconf reads a constant of the host's.
        let page = u64::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
            .ok()
            .filter(|p| p.is_power_of_two())
            .unwrap_or(0x4000);
        Some(ScanoutDraw {
            platform,
            device,
            queue,
            region: Arc::clone(region),
            page,
            import: Mutex::new(None),
            shown: Arc::clone(shown),
            presented: Arc::clone(presented),
            converter: std::sync::OnceLock::new(),
        })
    }

    /// The buffer over `scanout`'s pages and where its first row starts in it, imported where
    /// the last import covers other pages (the module doc); `None` where the platform refuses
    /// them.
    pub(crate) fn scanout_buffer(&self, scanout: &Scanout) -> Option<(P::Buffer, u64)> {
        let size = self.region.size();
        let first = scanout.offset.min(size);
        let end = scanout
            .offset
            .saturating_add(u64::from(scanout.stride) * u64::from(scanout.height));
        let start = first & !(self.page - 1);
        let length = end
            .checked_next_multiple_of(self.page)
            .unwrap_or(u64::MAX)
            .min(size)
            .saturating_sub(start);
        let mut import = self.import.lock();
        if !import
            .as_ref()
            .is_some_and(|i| (i.start, i.length) == (start, length))
        {
            *import = Some(ScanoutImport {
                start,
                length,
                buffer: self.import_pages(start, length),
            });
        }
        let buffer = import.as_ref()?.buffer.as_ref()?.clone_ref();
        Some((buffer, first - start))
    }

    /// The `length` bytes of the region at `start` as a no-copy buffer; `None`, logged, where
    /// the platform refuses them.
    fn import_pages(&self, start: u64, length: u64) -> Option<P::Buffer> {
        if length == 0 {
            return None;
        }
        let keep = Arc::clone(&self.region);
        // SAFETY: the pages lie inside the region (`scanout_buffer` keeps them to it), which is
        // mapped from its making until its last reference goes, and the deallocator holds one
        // until the platform runs it, once the buffer has gone.
        let bytes = unsafe {
            HostRange::new(
                HostPtr::from_ptr(self.region.base().add(start as usize)),
                length,
            )
        };
        let buffer = no_catch(self.device.new_buffer_over_host_memory(
            OnFault::Die,
            bytes,
            0,
            Some(Box::new(move || drop(keep))),
        ));
        if buffer.is_none() {
            log::warn!(
                target: "reims_vgpu::graphics",
                "the platform refused the scanout's pages (0x{length:x} bytes at 0x{start:x} in \
                 the region): the window shows nothing of this scanout"
            );
        }
        buffer
    }

    /// The window's render thread draws into a drawable of `size` pixels: the window's size,
    /// which the display offers the guest.
    pub(crate) fn drawn(&self, size: (u64, u64)) {
        *self.shown.drawable.lock() = Some(size);
    }

    /// A command buffer of the sink's queue with the scanout blitted into `texture` at `region`
    /// (the module doc), not committed, and the presents it shows, which the caller hands it
    /// before its commit ([`DrawnPresents::after`]); `None`, with nothing encoded, where the
    /// guest has no scanout or the queue gives no command buffer. The presents the draw shows are
    /// those made so far, whatever it draws: where it shows the scanout in place of a surface it
    /// cannot draw, or nothing, they are answered at the next refresh, so the guest's compositor
    /// does not wait on a window that cannot show them.
    pub(crate) fn encode(
        &self,
        texture: &P::Texture,
        region: PixelRect,
    ) -> Option<(P::CommandBuffer, DrawnPresents<P>)> {
        let mut presents = self.presented.drawn();
        if let Some((surface, hold)) = self.presented.shown() {
            let (from, to) = (surface.pixel_format(), texture.pixel_format());
            self.presented.seen_format(from, to);
            let drawn = if BGRA8_FORMATS.contains(&from) && BGRA8_FORMATS.contains(&to) {
                self.encode_surface(&surface, texture, region)
            } else {
                self.encode_converted(&surface, texture, region)
            };
            if let Some(drawn) = drawn {
                // The surface's block stays until the draw has read it.
                presents.reading(hold);
                return Some((drawn, presents));
            }
            self.presented.refuse_format(from, to);
        }
        let command_buffer = self.encode_scanout(texture, region)?;
        Some((command_buffer, presents))
    }

    /// A command buffer of the sink's queue with the scanout blitted into `texture` at `region`,
    /// not committed; `None` where the guest has no scanout or the queue gives no command buffer.
    fn encode_scanout(&self, texture: &P::Texture, region: PixelRect) -> Option<P::CommandBuffer> {
        let scanout = (*self.shown.scanout.lock())?;
        let (buffer, source_offset) = self.scanout_buffer(&scanout)?;
        let command_buffer = no_catch(self.queue.command_buffer(OnFault::Die))?;
        let (texture_width, texture_height) = (texture.width(), texture.height());
        let x = region.x.min(texture_width);
        let y = region.y.min(texture_height);
        let width = u64::from(scanout.width)
            .min(region.width)
            .min(texture_width - x);
        let height = u64::from(scanout.height)
            .min(region.height)
            .min(texture_height - y);
        if (x, y, width, height) != (0, 0, texture_width, texture_height) {
            self.clear(&command_buffer, texture);
        }
        if width > 0 && height > 0 {
            let blit = no_catch(command_buffer.blit_encoder(OnFault::Die))?;
            let stride = u64::from(scanout.stride);
            no_catch(blit.encode(
                OnFault::Die,
                BlitCmd::CopyBufferToTexture {
                    source_buffer: &buffer,
                    source_offset,
                    source_bytes_per_row: stride,
                    source_bytes_per_image: stride * height,
                    source_size: MtlSize::new(width, height, 1),
                    destination_texture: texture,
                    destination_slice: 0,
                    destination_level: 0,
                    destination_origin: MtlOrigin::new(x, y, 0),
                },
            ));
            no_catch(blit.encode(OnFault::Die, BlitCmd::EndEncoding));
        }
        Some(command_buffer)
    }

    /// A command buffer of the sink's queue with the presented `surface` copied into `texture` at
    /// `region`, clipped to both, after a clear where it does not cover the texture; not
    /// committed.
    fn encode_surface(
        &self,
        surface: &P::Texture,
        texture: &P::Texture,
        region: PixelRect,
    ) -> Option<P::CommandBuffer> {
        let command_buffer = no_catch(self.queue.command_buffer(OnFault::Die))?;
        let (texture_width, texture_height) = (texture.width(), texture.height());
        let x = region.x.min(texture_width);
        let y = region.y.min(texture_height);
        let width = surface.width().min(region.width).min(texture_width - x);
        let height = surface.height().min(region.height).min(texture_height - y);
        if (x, y, width, height) != (0, 0, texture_width, texture_height) {
            self.clear(&command_buffer, texture);
        }
        if width > 0 && height > 0 {
            let blit = no_catch(command_buffer.blit_encoder(OnFault::Die))?;
            no_catch(blit.encode(
                OnFault::Die,
                BlitCmd::CopyTextureRegion {
                    source_texture: surface,
                    source_slice: 0,
                    source_level: 0,
                    source_origin: MtlOrigin::new(0, 0, 0),
                    source_size: MtlSize::new(width, height, 1),
                    destination_texture: texture,
                    destination_slice: 0,
                    destination_level: 0,
                    destination_origin: MtlOrigin::new(x, y, 0),
                },
            ));
            no_catch(blit.encode(OnFault::Die, BlitCmd::EndEncoding));
        }
        Some(command_buffer)
    }

    /// The converter, made at its first use; `None` where the platform refuses its library, a
    /// pipeline or a buffer.
    fn converter(&self) -> Option<&Converter<P>> {
        self.converter
            .get_or_init(|| {
                let pipelines = DisplayPipelineCache::new(self.device.clone_ref())?;
                let buffer = |floats: &[f32]| -> Option<P::Buffer> {
                    let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
                    let length = u64::try_from(bytes.len()).ok()?;
                    let buffer =
                        no_catch(self.device.new_buffer_of_length(OnFault::Die, length, 0))?;
                    let contents: *mut u8 = buffer.contents().as_ptr();
                    if contents.is_null() {
                        return None;
                    }
                    // SAFETY: the buffer is `length` bytes of shared memory, its contents mapped
                    // while it lives, and nothing else writes it yet.
                    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents, bytes.len()) };
                    Some(buffer)
                };
                Some(Converter {
                    positions: buffer(&STRIP_POSITIONS)?,
                    coordinates: buffer(&STRIP_COORDINATES)?,
                    pipelines,
                })
            })
            .as_ref()
    }

    /// A command buffer of the sink's queue with the presented `surface` drawn into `texture` at
    /// `region` by the display's present shaders, which convert its format to the drawable's,
    /// after a clear of the drawable to black; not committed. `None`, with nothing committed,
    /// where the converter or a pipeline for the drawable's format is refused.
    fn encode_converted(
        &self,
        surface: &P::Texture,
        texture: &P::Texture,
        region: PixelRect,
    ) -> Option<P::CommandBuffer> {
        let converter = self.converter()?;
        let pipeline = converter.pipelines.pipeline_for(PipelineKey {
            format: texture.pixel_format(),
            gamma: false,
        })?;
        let command_buffer = no_catch(self.queue.command_buffer(OnFault::Die))?;
        let pass = self.platform.render_pass_descriptor();
        no_catch(pass.set_color_attachment(
            OnFault::Die,
            0,
            &ColorAttachmentSet {
                texture: Some(Some(texture)),
                load_action: Some(LOAD_ACTION_CLEAR),
                store_action: Some(STORE_ACTION_STORE),
                clear_color: Some(ClearColor {
                    red: 0.0,
                    green: 0.0,
                    blue: 0.0,
                    alpha: 1.0,
                }),
                ..ColorAttachmentSet::default()
            },
        ));
        let encoder = no_catch(command_buffer.render_encoder(OnFault::Die, &pass))?;
        let (texture_width, texture_height) = (texture.width(), texture.height());
        let x = region.x.min(texture_width);
        let y = region.y.min(texture_height);
        let width = region.width.min(texture_width - x);
        let height = region.height.min(texture_height - y);
        #[expect(clippy::as_conversions, reason = "pixel counts far below 2^52")]
        let viewport = MtlViewport {
            origin_x: x as f64,
            origin_y: y as f64,
            width: width as f64,
            height: height as f64,
            znear: 0.0,
            zfar: 1.0,
        };
        if width > 0 && height > 0 {
            for cmd in [
                RenderCmd::SetRenderPipelineState {
                    pipeline_state: Some(&pipeline),
                },
                RenderCmd::SetViewport { viewport },
                RenderCmd::SetVertexBuffer {
                    buffer: Some(&converter.positions),
                    offset: 0,
                    index: 0,
                },
                RenderCmd::SetVertexBuffer {
                    buffer: Some(&converter.coordinates),
                    offset: 0,
                    index: 1,
                },
                RenderCmd::SetFragmentTexture {
                    texture: Some(surface),
                    index: 0,
                },
                RenderCmd::DrawPrimitives {
                    primitive_type: TRIANGLE_STRIP,
                    vertex_start: 0,
                    vertex_count: 4,
                },
            ] {
                no_catch(encoder.encode(OnFault::Die, cmd));
            }
        }
        no_catch(encoder.encode(OnFault::Die, RenderCmd::EndEncoding));
        Some(command_buffer)
    }

    /// Clears all of `texture` to opaque black, in a render pass of `command_buffer` with no
    /// draw.
    fn clear(&self, command_buffer: &P::CommandBuffer, texture: &P::Texture) {
        let pass = self.platform.render_pass_descriptor();
        no_catch(pass.set_color_attachment(
            OnFault::Die,
            0,
            &ColorAttachmentSet {
                texture: Some(Some(texture)),
                load_action: Some(LOAD_ACTION_CLEAR),
                store_action: Some(STORE_ACTION_STORE),
                clear_color: Some(ClearColor {
                    red: 0.0,
                    green: 0.0,
                    blue: 0.0,
                    alpha: 1.0,
                }),
                ..ColorAttachmentSet::default()
            },
        ));
        if let Some(encoder) = no_catch(command_buffer.render_encoder(OnFault::Die, &pass)) {
            no_catch(encoder.encode(OnFault::Die, RenderCmd::EndEncoding));
        }
    }
}
