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
//! **The display's colour.** While any stage of the display pipe's colour is on (`color`: gamma
//! tables and a colour matrix a transaction set), the draw puts what the display shows through it:
//! a presented surface is drawn into the drawable by the colour's fragment function
//! (`displayPipeColorFragment`), in place of the copy or the plain present shaders, and the
//! scanout by its twin (`displayPipeColorScanoutFragment`), which reads the guest's pixels in
//! place from the imported pages, as the blit does, one drawable pixel for each, clipped as the
//! blit clips. The colour's buffers are made again only when a present changes it. With every
//! stage off, the draw is as before.
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
//!
//! **A refused draw.** What a draw reads is the guest's (its scanout's pages, the surface it
//! presented and its format, the display's colour), so a draw runs in a refusal scope of its
//! own ([`ScanoutDraw::scoped`]): a death in it, the platform refusing an encode or a create
//! among them, refuses that draw alone, as the lane refuses a guest op
//! (docs/graphics-kext/design.md, "Guest input the host refuses"). Nothing of it is committed, an
//! encoder it left open is ended on the way ([`Open`]), as Metal asserts at an encoder released
//! without its end, and the window keeps the frame it shows. The presents the draw took are
//! answered at the next refresh, as for a draw that shows nothing, and the next draw is tried as
//! any is. The first refusal is logged at WARN, later ones at debug.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

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
use reims_vgpu_core::raise::{OnFault, RaiseResult, refusals};
use reims_vgpu_core::sync::Mutex;
use reims_vgpu_window::PixelRect;

use reims_vgpu_core::display::pipeline::{DisplayPipelineCache, PipelineKey};
use reims_vgpu_core::platform::resource::Buffer as _;
use reims_vgpu_core::platform::types::MtlViewport;

use super::color::DisplayColor;
use super::{DisplayHold, DrawnPresents, Presented, Shown};
use crate::display::no_catch;
use crate::platform::ApiPlatform;
use reims_graphics::wire::{
    RGFX_COLOR_MATRIX, RGFX_COLOR_PLANE_GAMMA, RGFX_COLOR_POST_GAMMA, RGFX_COLOR_PRE_GAMMA,
};
use reims_vgpu_core::platform::resource::Library as _;
use reims_vgpu_core::platform::values::RenderPipelineDesc;

/// `MTLLoadActionClear`.
const LOAD_ACTION_CLEAR: u64 = 2;
/// `MTLStoreActionStore`.
const STORE_ACTION_STORE: u64 = 1;

/// What a graphics display's window sink draws with (the module doc).
pub(crate) struct ScanoutDraw<P: Platform> {
    /// The platform, whose render pass descriptors clear the drawable.
    platform: Arc<P>,
    /// The device the queue and the buffer are on: the platform's default. The Metal sink, on
    /// macOS, and the host's reads of the display (`read`), which make their texture on it, read
    /// it.
    pub(crate) device: P::Device,
    /// The sink's command queue, unshared ([`new_command_queue_unshared`]): it puts the guest's
    /// scanout and presented surfaces into the window, each draw on a present's path. The host's
    /// reads of the display, which no present waits on, draw on a queue the platform may share
    /// ([`ScanoutDraw::for_reads`]).
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
    /// The colour's fragment functions, a surface's and the scanout's, made at the first draw
    /// through a colour; `None` inside where the platform's library holds them not.
    color_functions: std::sync::OnceLock<Option<[P::Function; 2]>>,
    /// The colour's pipelines, by what they draw and the drawable's pixel format.
    color_pipelines: Mutex<ColorPipelines<P>>,
    /// The colour's buffers, made for the colour of that count of changes.
    color_buffers: Mutex<Option<(u64, ColorBuffers<P>)>>,
    /// Whether a draw was refused yet: the first refusal is logged at WARN ([`Self::scoped`]).
    refused: AtomicBool,
}

/// An encoder a draw holds open, ended by [`Open::end`]. One a death in the draw unwinds past
/// (the draw's scope takes the death, [`ScanoutDraw::scoped`]) is ended as the unwind drops it, in
/// a refusal scope of its own: Metal asserts at an encoder released without its end, and the
/// command buffer it encodes into is never committed.
struct Open<E> {
    /// The encoder, until it is ended.
    encoder: Option<E>,
    /// Its end: `endEncoding`.
    end: fn(&E) -> RaiseResult<()>,
}

impl<E> Open<E> {
    /// `encoder`, open, which `end` ends.
    fn new(encoder: E, end: fn(&E) -> RaiseResult<()>) -> Open<E> {
        Open {
            encoder: Some(encoder),
            end,
        }
    }

    /// The encoder, set from [`Open::new`] until [`Open::end`] or the drop takes it.
    fn get(&self) -> &E {
        match &self.encoder {
            Some(encoder) => encoder,
            None => unreachable!("an encoder is open until it is taken"),
        }
    }

    /// Ends the encoder.
    fn end(mut self) {
        if let Some(encoder) = self.encoder.take() {
            no_catch((self.end)(&encoder));
        }
    }
}

impl<E> Drop for Open<E> {
    fn drop(&mut self) {
        if let Some(encoder) = self.encoder.take() {
            let end = self.end;
            if let Err(refusal) = refusals::refusing(|| no_catch(end(&encoder))) {
                log::debug!(
                    target: "reims_vgpu::graphics",
                    "the end of a refused draw's encoder was refused too: {refusal}"
                );
            }
        }
    }
}

/// A render encoder's end.
fn end_render<P: Platform>(encoder: &P::RenderEncoder) -> RaiseResult<()> {
    encoder.encode(OnFault::Die, RenderCmd::EndEncoding)
}

/// A blit encoder's end.
fn end_blit<P: Platform>(encoder: &P::BlitEncoder) -> RaiseResult<()> {
    encoder.encode(OnFault::Die, BlitCmd::EndEncoding)
}

/// The colour's pipelines made so far, each with what it draws and the drawable's pixel format.
type ColorPipelines<P> = Vec<((ColorSource, u64), <P as Platform>::RenderPipeline)>;

/// What a draw through the colour reads: a presented surface's texture, or the scanout's pages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ColorSource {
    /// `displayPipeColorFragment`.
    Surface,
    /// `displayPipeColorScanoutFragment`.
    Scanout,
}

/// Where a draw through the colour reads the scanout (`displayPipeColorScanoutFragment`'s
/// buffers 4 and 5): the imported pages, and the drawable pixel its top left goes to, its row's
/// 4-byte words and its first pixel's word in the buffer.
struct ScanoutRead<P: Platform> {
    buffer: P::Buffer,
    origin: (u32, u32),
    stride_words: u32,
    first_word: u32,
}

/// The colour's buffers (the module doc): the properties, `displayPipeColorFragment`'s buffer 0
/// (the matrix's rows, then each stage's entry count), and the three gamma stages' tables, its
/// buffers 1 to 3 (an empty stage's a buffer of one zero entry).
struct ColorBuffers<P: Platform> {
    properties: P::Buffer,
    tables: [P::Buffer; 3],
}

impl<P: Platform> ColorBuffers<P> {
    /// The same buffers, retained again.
    fn clone_refs(&self) -> ColorBuffers<P> {
        ColorBuffers {
            properties: self.properties.clone_ref(),
            tables: self.tables.each_ref().map(|b| b.clone_ref()),
        }
    }
}

/// What a draw through the colour reads (`ScanoutDraw::encode_color_pass`).
enum Read<'a, P: Platform> {
    /// A presented surface's texture.
    Surface(&'a P::Texture),
    /// The scanout's pages.
    Scanout(&'a ScanoutRead<P>),
}

impl<P: Platform> Clone for Read<'_, P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<P: Platform> Copy for Read<'_, P> {}

/// The colour's fragment functions: a surface's, and the scanout's.
const COLOR_FRAGMENTS: [&str; 2] = [
    "displayPipeColorFragment",
    "displayPipeColorScanoutFragment",
];
/// The colour pipelines' label.
const COLOR_LABEL: &str = "reims-vgpu.display-pipe-color";

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
        Self::on_queue(platform, region, shown, presented, |device| {
            device.new_command_queue_unshared()
        })
    }

    /// A draw as [`Self::new`]'s, for the host's reads of the display (`read`), on a queue the
    /// platform may share with other command queues: no present waits on a read, so it holds no
    /// GPU queue of its own for the device's life.
    pub(crate) fn for_reads(
        platform: Arc<P>,
        region: &Arc<Region>,
        shown: &Arc<Shown>,
        presented: &Arc<Presented<P>>,
    ) -> Option<ScanoutDraw<P>> {
        Self::on_queue(platform, region, shown, presented, |device| {
            device.new_command_queue()
        })
    }

    /// [`Self::new`], on the queue `queue` makes on the device.
    fn on_queue(
        platform: Arc<P>,
        region: &Arc<Region>,
        shown: &Arc<Shown>,
        presented: &Arc<Presented<P>>,
        queue: impl FnOnce(&P::Device) -> Option<P::CommandQueue>,
    ) -> Option<ScanoutDraw<P>> {
        let device = platform.metal_device(None)?;
        let queue = queue(&device)?;
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
            color_functions: std::sync::OnceLock::new(),
            color_pipelines: Mutex::default(),
            color_buffers: Mutex::default(),
            refused: AtomicBool::new(false),
        })
    }

    /// Runs `draw`, a draw of the display, in a refusal scope (the module doc, "A refused draw"):
    /// what it answers, or `None` where a death in it refused it there, logged.
    pub(crate) fn scoped<T>(&self, draw: impl FnOnce() -> Option<T>) -> Option<T> {
        match refusals::refusing(draw) {
            Ok(drawn) => drawn,
            Err(refusal) => {
                if self.refused.swap(true, Ordering::AcqRel) {
                    log::debug!(
                        target: "reims_vgpu::graphics",
                        "a draw of the display was refused: {refusal}"
                    );
                } else {
                    log::warn!(
                        target: "reims_vgpu::graphics",
                        "a draw of the display was refused, and the window keeps the frame it \
                         shows: {refusal} (later refusals are logged at debug)"
                    );
                }
                None
            }
        }
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
    ///
    /// It runs in the draw's refusal scope ([`Self::scoped`]): `None` too where a death in it
    /// refused it, nothing committed, and its presents are answered at the next refresh.
    pub(crate) fn encode(
        &self,
        texture: &P::Texture,
        region: PixelRect,
    ) -> Option<(P::CommandBuffer, DrawnPresents<P>)> {
        self.scoped(|| {
            let mut presents = self.presented.drawn();
            let (command_buffer, hold) = self.encode_shown(texture, region)?;
            if let Some(hold) = hold {
                // The surface's block stays until the draw has read it.
                presents.reading(hold);
            }
            Some((command_buffer, presents))
        })
    }

    /// A command buffer of the sink's queue with what the display shows drawn into `texture` at
    /// `region`, as [`Self::encode`] draws it, not committed, and the hold of the block of the
    /// presented surface it reads, where it draws one: the caller keeps it until the buffer has
    /// finished. `None`, with nothing encoded, where the guest has no scanout or the queue gives
    /// no command buffer. It answers no present, so the host's reads of the display
    /// (`graphics::read`) draw with it.
    pub(crate) fn encode_shown(
        &self,
        texture: &P::Texture,
        region: PixelRect,
    ) -> Option<(P::CommandBuffer, Option<Arc<DisplayHold>>)> {
        let (changes, color) = self.presented.color();
        let color = (!color.is_identity()).then_some((changes, color));
        if let Some((surface, hold)) = self.presented.shown() {
            let (from, to) = (surface.pixel_format(), texture.pixel_format());
            self.presented.seen_format(from, to);
            let drawn = if let Some((changes, color)) = &color {
                self.encode_colored(*changes, color, &surface, texture, region)
            } else if BGRA8_FORMATS.contains(&from) && BGRA8_FORMATS.contains(&to) {
                self.encode_surface(&surface, texture, region)
            } else {
                self.encode_converted(&surface, texture, region)
            };
            if let Some(drawn) = drawn {
                return Some((drawn, Some(hold)));
            }
            self.presented.refuse_format(from, to);
        }
        let command_buffer = match &color {
            Some((changes, color)) => self.encode_scanout_colored(*changes, color, texture, region),
            None => self.encode_scanout(texture, region),
        }?;
        Some((command_buffer, None))
    }

    /// A command buffer of the sink's queue with the scanout drawn into `texture` at `region`
    /// through `color` (the module doc), clipped as [`Self::encode_scanout`] clips its blit, after
    /// a clear of the drawable to black; not committed. `None` where the guest has no scanout, or
    /// the platform refuses its pages, the colour's pipeline or buffers.
    fn encode_scanout_colored(
        &self,
        changes: u64,
        color: &DisplayColor,
        texture: &P::Texture,
        region: PixelRect,
    ) -> Option<P::CommandBuffer> {
        let scanout = (*self.shown.scanout.lock())?;
        let (buffer, source_offset) = self.scanout_buffer(&scanout)?;
        let (texture_width, texture_height) = (texture.width(), texture.height());
        let x = region.x.min(texture_width);
        let y = region.y.min(texture_height);
        let width = u64::from(scanout.width)
            .min(region.width)
            .min(texture_width - x);
        let height = u64::from(scanout.height)
            .min(region.height)
            .min(texture_height - y);
        // The scanout's offset and stride are multiples of 4 (`reims_graphics`'s `SET_SCANOUT`),
        // and its pages' import starts at a page.
        let read = ScanoutRead {
            buffer,
            origin: (u32::try_from(x).ok()?, u32::try_from(y).ok()?),
            stride_words: scanout.stride / 4,
            first_word: u32::try_from(source_offset / 4).ok()?,
        };
        let at = PixelRect {
            x,
            y,
            width,
            height,
        };
        let command_buffer = no_catch(self.queue.command_buffer(OnFault::Die))?;
        self.encode_color_pass(
            &command_buffer,
            changes,
            color,
            Read::Scanout(&read),
            texture,
            at,
        )?;
        Some(command_buffer)
    }

    /// A command buffer of the sink's queue with the presented `surface` drawn into `texture` at
    /// `region` through `color` (the module doc), after a clear of the drawable to black; not
    /// committed. `None`, with nothing committed, where the platform refuses the colour's
    /// pipeline or buffers.
    fn encode_colored(
        &self,
        changes: u64,
        color: &DisplayColor,
        surface: &P::Texture,
        texture: &P::Texture,
        region: PixelRect,
    ) -> Option<P::CommandBuffer> {
        let command_buffer = no_catch(self.queue.command_buffer(OnFault::Die))?;
        let (texture_width, texture_height) = (texture.width(), texture.height());
        let x = region.x.min(texture_width);
        let y = region.y.min(texture_height);
        let at = PixelRect {
            x,
            y,
            width: region.width.min(texture_width - x),
            height: region.height.min(texture_height - y),
        };
        self.encode_color_pass(
            &command_buffer,
            changes,
            color,
            Read::Surface(surface),
            texture,
            at,
        )?;
        Some(command_buffer)
    }

    /// The colour's pipeline for `source` into a drawable of `format`, made and kept at its first
    /// use; `None` where the platform's library holds not the colour's functions, or refuses the
    /// pipeline.
    fn color_pipeline(
        &self,
        converter: &Converter<P>,
        source: ColorSource,
        format: u64,
    ) -> Option<P::RenderPipeline> {
        let functions = self
            .color_functions
            .get_or_init(|| {
                let library = self.device.new_shader_library().ok().flatten()?;
                let function = |name| no_catch(library.new_function(OnFault::Die, name));
                Some([function(COLOR_FRAGMENTS[0])?, function(COLOR_FRAGMENTS[1])?])
            })
            .as_ref()?;
        let mut pipelines = self.color_pipelines.lock();
        if let Some((_, pipeline)) = pipelines.iter().find(|(key, _)| *key == (source, format)) {
            return Some(pipeline.clone_ref());
        }
        let fragment = match source {
            ColorSource::Surface => &functions[0],
            ColorSource::Scanout => &functions[1],
        };
        let made = self
            .device
            .new_render_pipeline_state(&RenderPipelineDesc {
                label: Some(COLOR_LABEL),
                vertex: &converter.pipelines.vertex_function,
                fragment,
                color0_pixel_format: format,
                blend_over: false,
                resource_index: None,
            })
            .ok()
            .flatten()?;
        pipelines.push(((source, format), made.clone_ref()));
        Some(made)
    }

    /// The colour's buffers for `color`, the colour of `changes` changes, made again where the
    /// colour changed since the last made; `None` where the platform refuses one.
    fn color_buffers(&self, changes: u64, color: &DisplayColor) -> Option<ColorBuffers<P>> {
        let mut kept = self.color_buffers.lock();
        if let Some((made_for, buffers)) = kept.as_ref()
            && *made_for == changes
        {
            return Some(buffers.clone_refs());
        }
        let bytes = |entries: &[[f32; 4]]| -> Vec<u8> {
            entries
                .iter()
                .flatten()
                .flat_map(|f| f.to_le_bytes())
                .collect()
        };
        let stage = |s: u32| {
            let s = usize::try_from(s).ok()?;
            color.stages.get(s).and_then(Option::as_ref)
        };
        let count = |s: u32| stage(s).map_or(0, |t| u32::try_from(t.len()).unwrap_or(0));
        let mut properties = match stage(RGFX_COLOR_MATRIX) {
            Some(rows) => bytes(rows),
            None => bytes(&[[0.0; 4]; 3]),
        };
        for s in [
            RGFX_COLOR_PLANE_GAMMA,
            RGFX_COLOR_PRE_GAMMA,
            RGFX_COLOR_MATRIX,
            RGFX_COLOR_POST_GAMMA,
        ] {
            properties.extend_from_slice(&count(s).to_le_bytes());
        }
        let table = |s: u32| match stage(s) {
            Some(entries) => self.buffer_of(&bytes(entries)),
            None => self.buffer_of(&[0; 16]),
        };
        let made = ColorBuffers {
            properties: self.buffer_of(&properties)?,
            tables: [
                table(RGFX_COLOR_PLANE_GAMMA)?,
                table(RGFX_COLOR_PRE_GAMMA)?,
                table(RGFX_COLOR_POST_GAMMA)?,
            ],
        };
        *kept = Some((changes, made.clone_refs()));
        Some(made)
    }

    /// A shared buffer holding `bytes`; `None` where the platform refuses it.
    fn buffer_of(&self, bytes: &[u8]) -> Option<P::Buffer> {
        let length = u64::try_from(bytes.len()).ok()?;
        let buffer = no_catch(self.device.new_buffer_of_length(OnFault::Die, length, 0))?;
        let contents: *mut u8 = buffer.contents().as_ptr();
        if contents.is_null() {
            return None;
        }
        // SAFETY: the buffer is `length` bytes of shared memory, its contents mapped while it
        // lives, and nothing else writes it yet.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents, bytes.len()) };
        Some(buffer)
    }

    /// Encodes into `command_buffer` a render pass that clears `texture` to black and draws
    /// `read` over `at` through `color`, the colour of `changes` changes; `None`, with no pass
    /// encoded, where the platform refuses the colour's pipeline or buffers.
    fn encode_color_pass(
        &self,
        command_buffer: &P::CommandBuffer,
        changes: u64,
        color: &DisplayColor,
        read: Read<'_, P>,
        texture: &P::Texture,
        at: PixelRect,
    ) -> Option<()> {
        let converter = self.converter()?;
        let source = match read {
            Read::Surface(_) => ColorSource::Surface,
            Read::Scanout(_) => ColorSource::Scanout,
        };
        let pipeline = self.color_pipeline(converter, source, texture.pixel_format())?;
        let buffers = self.color_buffers(changes, color)?;
        let scanout_at = match read {
            Read::Scanout(r) => {
                let words = [r.origin.0, r.origin.1, r.stride_words, r.first_word];
                let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
                Some(self.buffer_of(&bytes)?)
            }
            Read::Surface(_) => None,
        };
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
        let encoder = Open::new(
            no_catch(command_buffer.render_encoder(OnFault::Die, &pass))?,
            end_render::<P>,
        );
        #[expect(clippy::as_conversions, reason = "pixel counts far below 2^52")]
        let viewport = MtlViewport {
            origin_x: at.x as f64,
            origin_y: at.y as f64,
            width: at.width as f64,
            height: at.height as f64,
            znear: 0.0,
            zfar: 1.0,
        };
        if at.width > 0 && at.height > 0 {
            let mut cmds = vec![
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
                RenderCmd::SetFragmentBuffer {
                    buffer: Some(&buffers.properties),
                    offset: 0,
                    index: 0,
                },
            ];
            for (index, table) in (1..).zip(&buffers.tables) {
                cmds.push(RenderCmd::SetFragmentBuffer {
                    buffer: Some(table),
                    offset: 0,
                    index,
                });
            }
            match (read, &scanout_at) {
                (Read::Surface(surface), _) => cmds.push(RenderCmd::SetFragmentTexture {
                    texture: Some(surface),
                    index: 0,
                }),
                (Read::Scanout(r), Some(scanout_at)) => {
                    cmds.push(RenderCmd::SetFragmentBuffer {
                        buffer: Some(&r.buffer),
                        offset: 0,
                        index: 4,
                    });
                    cmds.push(RenderCmd::SetFragmentBuffer {
                        buffer: Some(scanout_at),
                        offset: 0,
                        index: 5,
                    });
                }
                (Read::Scanout(_), None) => {}
            }
            cmds.push(RenderCmd::DrawPrimitives {
                primitive_type: TRIANGLE_STRIP,
                vertex_start: 0,
                vertex_count: 4,
            });
            for cmd in cmds {
                no_catch(encoder.get().encode(OnFault::Die, cmd));
            }
        }
        encoder.end();
        Some(())
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
            let blit = Open::new(
                no_catch(command_buffer.blit_encoder(OnFault::Die))?,
                end_blit::<P>,
            );
            let stride = u64::from(scanout.stride);
            no_catch(blit.get().encode(
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
            blit.end();
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
            let blit = Open::new(
                no_catch(command_buffer.blit_encoder(OnFault::Die))?,
                end_blit::<P>,
            );
            no_catch(blit.get().encode(
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
            blit.end();
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
        let encoder = Open::new(
            no_catch(command_buffer.render_encoder(OnFault::Die, &pass))?,
            end_render::<P>,
        );
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
                no_catch(encoder.get().encode(OnFault::Die, cmd));
            }
        }
        encoder.end();
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

#[cfg(test)]
mod tests {
    use reims_vgpu_core::platform::ObjId;
    use reims_vgpu_core::platform::values::TextureDesc;
    use reims_vgpu_core::testkit::objects::kind;
    use reims_vgpu_core::testkit::{Arg, FakePlatform};

    use super::super::color::Table;
    use super::super::{DisplayHold, ShownTexture};
    use super::*;

    const FRAGMENT_BUFFER: &str = "setFragmentBuffer:offset:atIndex:";
    const DRAW: &str = "drawPrimitives:vertexStart:vertexCount:";
    const COPY_IN: &str = "copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:\
                           sourceSize:toTexture:destinationSlice:destinationLevel:\
                           destinationOrigin:";
    const NEW_BUFFER: &str = "newBufferWithLength:options:";
    const NEW_TEXTURE: &str = "newTextureWithDescriptor:";
    const NEW_PIPELINE: &str = "newRenderPipelineStateWithDescriptor:error:";
    const NAMED: &str = "newFunctionWithName:";

    /// The scanout `rig` shows.
    const SCANOUT: Scanout = Scanout {
        width: 64,
        height: 32,
        stride: 256,
        offset: reims_graphics::wire::RGFX_ALLOCATIONS_OFFSET,
    };

    /// A sink over a fresh region, showing a 64 × 32 scanout at the region's allocations, and a
    /// BGRA8 drawable of that size.
    fn rig() -> (
        Arc<FakePlatform>,
        ScanoutDraw<FakePlatform>,
        Arc<Presented<FakePlatform>>,
        <FakePlatform as Platform>::Texture,
    ) {
        let platform = Arc::new(FakePlatform::new());
        let region = Arc::new(Region::new(reims_graphics::REGION_SIZE as usize).unwrap());
        let shown = Arc::new(Shown::default());
        *shown.scanout.lock() = Some(SCANOUT);
        let presented = Arc::new(Presented::<FakePlatform>::default());
        let draw = ScanoutDraw::new(Arc::clone(&platform), &region, &shown, &presented).unwrap();
        let drawable = texture(&draw, 80);
        (platform, draw, presented, drawable)
    }

    /// A 64 × 32 texture of `format` on the sink's device.
    fn texture(
        draw: &ScanoutDraw<FakePlatform>,
        format: u64,
    ) -> <FakePlatform as Platform>::Texture {
        let descriptor = draw.platform.texture_descriptor(&TextureDesc {
            pixel_format: format,
            width: 64,
            height: 32,
            ..TextureDesc::default()
        });
        draw.device
            .new_texture(OnFault::Die, &descriptor)
            .unwrap()
            .unwrap()
    }

    /// The bytes of a buffer the fake platform made.
    fn bytes_of(buffer: &<FakePlatform as Platform>::Buffer) -> Vec<u8> {
        let length = usize::try_from(buffer.length()).unwrap();
        let contents: *const u8 = buffer.contents().as_ptr();
        // SAFETY: the fake buffer's contents are `length` bytes it holds while it lives.
        unsafe { std::slice::from_raw_parts(contents, length) }.to_vec()
    }

    fn floats(entries: &[[f32; 4]]) -> Vec<u8> {
        entries
            .iter()
            .flatten()
            .flat_map(|f| f.to_le_bytes())
            .collect()
    }

    /// The display's colour becomes `stages`, its `changes`-th change.
    fn set_color(presented: &Presented<FakePlatform>, changes: u64, stages: [Option<Table>; 4]) {
        *presented.color.lock() = (changes, Arc::new(DisplayColor { stages }));
    }

    /// The fragment buffers a draw since `mark` bound, by index.
    fn fragment_buffers(platform: &FakePlatform, mark: u64) -> Vec<(u64, ObjId)> {
        platform
            .call_log()
            .since(mark)
            .into_iter()
            .filter(|c| c.sel == FRAGMENT_BUFFER)
            .map(|c| (c.args[2].as_u64().unwrap(), c.args[0].as_obj().unwrap()))
            .collect()
    }

    /// The buffers a test keeps alive.
    type Made = Mutex<Vec<<FakePlatform as Platform>::Buffer>>;

    /// The buffers the fake platform makes from here on, kept alive (up to `n`).
    fn keep_new_buffers(platform: &FakePlatform, n: usize) -> Arc<Made> {
        let made: Arc<Made> = Arc::default();
        for _ in 0..n {
            let made = Arc::clone(&made);
            platform
                .world()
                .on_new::<kind::Buffer>(move |b| made.lock().push(b.clone_ref()));
        }
        made
    }

    /// The four words of the scanout's place a draw bound at fragment buffer 5.
    fn scanout_words(made: &Made, id: ObjId) -> Vec<u32> {
        let made = made.lock();
        let buffer = made
            .iter()
            .find(|b| b.id() == id)
            .expect("the scanout's place");
        let bytes = bytes_of(buffer);
        let (words, _) = bytes.as_chunks::<4>();
        words.iter().map(|w| u32::from_le_bytes(*w)).collect()
    }

    /// A death in a draw (the platform's, at the scanout's copy, or at a draw through the
    /// display's colour) refuses that draw alone: it answers nothing, the encoder it held open is
    /// ended after the death, nothing is committed, and the next draw draws
    #[test]
    fn a_death_in_a_draw_refuses_that_draw_alone() {
        use reims_vgpu_core::testkit::Step;
        let (platform, draw, presented, drawable) = rig();
        let log = platform.call_log();
        let region = PixelRect {
            x: 0,
            y: 0,
            width: 64,
            height: 32,
        };
        let post: Table = Arc::from(vec![[0.0, 0.0, 0.0, 0.0], [0.5, 0.25, 1.0, 0.0]]);
        for (at, color) in [(COPY_IN, None), (DRAW, Some(post))] {
            if let Some(post) = color {
                set_color(&presented, 1, [None, None, None, Some(post)]);
            }
            platform.script().push(at, Step::crash());
            let mark = log.mark();
            assert!(draw.encode(&drawable, region).is_none());
            let sels = log.sels_since(mark);
            let died = sels
                .iter()
                .position(|s| *s == at)
                .expect("the call that died");
            assert_eq!(&sels[died + 1..], ["endEncoding"], "{sels:?}");
            assert!(!sels.contains(&"commit"));

            let mark = log.mark();
            let (command_buffer, presents) = draw.encode(&drawable, region).expect("a draw");
            presents.after(&command_buffer);
            command_buffer.commit(OnFault::Die).unwrap();
            let sels = log.sels_since(mark);
            assert_eq!(sels.iter().filter(|s| **s == at).count(), 1, "{sels:?}");
            assert!(sels.contains(&"commit"));
        }
    }

    /// With every stage off, the scanout is blitted into the drawable and nothing is drawn; with
    /// a stage on, it is drawn into the drawable by the scanout's colour function, reading the
    /// imported pages in place (no copy), with the properties at fragment buffer 0 (the matrix's
    /// rows, each stage's entry count), the gamma tables at 1 to 3, the pages at 4 and the
    /// scanout's place at 5 (the drawable pixel of its top left, its row and first pixel in
    /// words); the next draw of the same colour makes no table, texture or pipeline, and a change
    /// of the colour makes its buffers again
    #[test]
    fn a_color_draws_the_scanout_through_its_tables() {
        let (platform, draw, presented, drawable) = rig();
        let log = platform.call_log();
        let region = PixelRect {
            x: 0,
            y: 0,
            width: 64,
            height: 32,
        };

        let mark = log.mark();
        let _ = draw.encode(&drawable, region).unwrap();
        let blits: Vec<_> = log
            .since(mark)
            .into_iter()
            .filter(|c| c.sel == COPY_IN)
            .collect();
        assert_eq!(blits.len(), 1);
        assert!(blits[0].args.contains(&Arg::obj(&drawable)));
        assert_eq!(
            log.sels_since(mark).iter().filter(|s| **s == DRAW).count(),
            0
        );
        let (pages, _) = draw.scanout_buffer(&SCANOUT).unwrap();

        let post: Table = Arc::from(vec![[0.0, 0.0, 0.0, 0.0], [0.5, 0.25, 1.0, 0.0]]);
        set_color(&presented, 1, [None, None, None, Some(Arc::clone(&post))]);
        let made = keep_new_buffers(&platform, 16);
        let mark = log.mark();
        let _ = draw.encode(&drawable, region).unwrap();
        let calls = log.since(mark);
        let named: Vec<_> = calls
            .iter()
            .filter(|c| c.sel == NAMED)
            .map(|c| c.args[0].clone())
            .collect();
        // After the present shaders' (the converter's, made at its first use), the colour's two.
        assert!(
            named.ends_with(&COLOR_FRAGMENTS.map(|f| Arg::Str(f.to_owned()))),
            "{named:?}"
        );
        assert!(
            !calls.iter().any(|c| c.sel == COPY_IN),
            "the pages are read in place"
        );
        assert_eq!(calls.iter().filter(|c| c.sel == NEW_TEXTURE).count(), 0);
        assert_eq!(calls.iter().filter(|c| c.sel == DRAW).count(), 1);
        let bound = fragment_buffers(&platform, mark);
        assert_eq!(
            bound.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            [0, 1, 2, 3, 4, 5]
        );
        assert_eq!(bound[4].1, pages.id());
        assert_eq!(scanout_words(&made, bound[5].1), [0, 0, 64, 0]);

        let buffers = draw.color_buffers(1, &presented.color().1).unwrap();
        assert_eq!(buffers.properties.id(), bound[0].1);
        let mut properties = floats(&[[0.0; 4]; 3]);
        for count in [0u32, 0, 0, 2] {
            properties.extend_from_slice(&count.to_le_bytes());
        }
        assert_eq!(bytes_of(&buffers.properties), properties);
        assert_eq!(bytes_of(&buffers.tables[0]), [0; 16]);
        assert_eq!(bytes_of(&buffers.tables[1]), [0; 16]);
        assert_eq!(bytes_of(&buffers.tables[2]), floats(&post));
        assert_eq!(buffers.tables[2].id(), bound[3].1);

        // The same colour again: only the scanout's place is made.
        let mark = log.mark();
        let _ = draw.encode(&drawable, region).unwrap();
        assert_eq!(
            log.sels_since(mark)
                .iter()
                .filter(|s| **s == NEW_BUFFER)
                .count(),
            1
        );
        for made in [NEW_TEXTURE, NEW_PIPELINE, NAMED] {
            assert_eq!(
                log.sels_since(mark).iter().filter(|s| **s == made).count(),
                0,
                "{made}"
            );
        }
        assert_eq!(fragment_buffers(&platform, mark)[..5], bound[..5]);

        // A matrix and a plane table, into a region off the drawable's corner: the buffers are
        // made again, and the scanout's place names the region's corner.
        let rows: Table = Arc::from(vec![
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.25],
        ]);
        let plane: Table = Arc::from(vec![[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]);
        set_color(
            &presented,
            2,
            [
                Some(Arc::clone(&plane)),
                None,
                Some(Arc::clone(&rows)),
                Some(post),
            ],
        );
        let corner = PixelRect {
            x: 8,
            y: 4,
            width: 64,
            height: 32,
        };
        let mark = log.mark();
        let _ = draw.encode(&drawable, corner).unwrap();
        assert_eq!(
            log.sels_since(mark)
                .iter()
                .filter(|s| **s == NEW_BUFFER)
                .count(),
            5
        );
        let buffers = draw.color_buffers(2, &presented.color().1).unwrap();
        let mut properties = floats(&rows);
        for count in [2u32, 0, 3, 2] {
            properties.extend_from_slice(&count.to_le_bytes());
        }
        assert_eq!(bytes_of(&buffers.properties), properties);
        assert_eq!(bytes_of(&buffers.tables[0]), floats(&plane));
        let bound = fragment_buffers(&platform, mark);
        assert_eq!(bound[0].1, buffers.properties.id());
        assert_eq!(scanout_words(&made, bound[5].1), [8, 4, 64, 0]);

        // Every stage off again: the blit, as before.
        set_color(&presented, 3, Default::default());
        let mark = log.mark();
        let _ = draw.encode(&drawable, region).unwrap();
        assert_eq!(
            log.sels_since(mark).iter().filter(|s| **s == DRAW).count(),
            0
        );
        assert!(
            log.since(mark)
                .iter()
                .any(|c| c.sel == COPY_IN && c.args.contains(&Arg::obj(&drawable)))
        );
    }

    /// A presented surface is drawn through the colour (a stage on) by the surface's colour
    /// function, in place of the copy a BGRA8 surface takes, sampled straight from the surface
    #[test]
    fn a_color_draws_a_presented_surface_through_its_tables() {
        let (platform, draw, presented, drawable) = rig();
        let log = platform.call_log();
        let surface = texture(&draw, 80);
        *presented.texture.lock() = Some(ShownTexture {
            texture: surface.clone_ref(),
            hold: Arc::new(DisplayHold {
                span: None,
                answers: Mutex::default(),
            }),
        });
        let region = PixelRect {
            x: 0,
            y: 0,
            width: 64,
            height: 32,
        };
        let pre: Table = Arc::from(vec![[0.25, 0.25, 0.25, 0.0]]);
        set_color(&presented, 1, [None, Some(pre), None, None]);
        let mark = log.mark();
        let _ = draw.encode(&drawable, region).unwrap();
        let calls = log.since(mark);
        assert_eq!(calls.iter().filter(|c| c.sel == DRAW).count(), 1);
        assert!(
            !calls
                .iter()
                .any(|c| c.sel == COPY_IN || c.sel.starts_with("copyFromTexture"))
        );
        let sampled: Vec<_> = calls
            .iter()
            .filter(|c| c.sel == "setFragmentTexture:atIndex:")
            .map(|c| c.args[0].as_obj())
            .collect();
        assert_eq!(sampled, [Some(surface.id())]);
        let bound = fragment_buffers(&platform, mark);
        assert_eq!(
            bound.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );
    }
}
