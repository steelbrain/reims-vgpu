//! [`ScanoutDraw`]: what a graphics display's window sink draws with, on any platform: the
//! region's pages under the guest's scanout, imported as a no-copy buffer, and a command queue
//! of its own on the platform's default device. Each draw blits the guest's scanout out of the
//! buffer into the drawable's texture, at the image's place, in one command buffer: the GPU reads
//! the guest's pixels in place, and the CPU copies nothing.
//!
//! **Sizes.** The guest sets a scanout of the size the display offers, the window's size in
//! pixels, its width rounded down to a multiple of 16 and its height to even, so the image is the
//! whole drawable, or, for a drawable of other pixels, all of it but up to 15 columns and a row,
//! which stay black: the layout draws that mode a guest pixel to a pixel, at the drawable's top
//! left (`geometry::offered_mode`). While the two differ
//! otherwise (the window resized, the guest not yet following), the blit copies what fits of the
//! scanout at the image's top left, clipped to the drawable, after a clear of the drawable to
//! black: a blit cannot scale, and the guest's next mode makes the sizes match again.
//!
//! **A presented surface.** While the display pipe shows an IOSurface (`PRESENT_SURFACE`), the
//! draw puts the texture the decoder made over it into the drawable instead, on the GPU: a copy,
//! texture to texture, where the two have one pixel format (BGRA8, with or without sRGB), and
//! otherwise a draw of the display's present shaders (core's [`DisplayPipelineCache`]), which
//! sample the surface in its own format (WindowServer's display surfaces are 10-bit,
//! `BGR10A2Unorm`) and write the drawable's. The first surface of each format is logged.
//!
//! **The display's view.** The passes that sample a presented surface (the present shaders'
//! and the colour's) read its texture through the display's own view of it
//! ([`super::ShownView`]): the whole texture in its own pixel format, at one of the platform's
//! own indices, not the texture's own index, which is the guest's. The guest's indices are its
//! processes': as one ends (a logout, a window server's restart) its indices go back to the
//! guest's kext, and the next process's texture made at one takes its element at once, while the
//! display may still show, and draw again at each refresh, the texture that held it. The view is
//! made at the texture's first such draw, and kept while the decoder holds the texture and while
//! the display shows it, so a present of a texture shown before costs nothing more; the copy
//! reads the texture's image, through no index, and makes none. Where the platform refuses the
//! view, the draw is `None` ([`ScanoutDraw::encode_shown`]), counted (`display-view-refused`):
//! the window keeps the frame it shows, and the host's read of the display answers black, as
//! where the display shows nothing; the next draw asks for the view again.
//!
//! **Clears.** A draw clears the drawable to black first only where what it draws does not cover
//! all of it: a pass whose draw covers the drawable writes each of its pixels, so it loads none
//! of the drawable's old ones ([`pass_load`]), and its pixels are those it would draw after a
//! clear. On a Radeon 780M (RADV), a 3840 x 2160 surface of WindowServer's at 30 frames a second,
//! the pass took 0.78 ms of the GPU's time with its clear and 0.80 ms without (measured
//! 2026-11-08, the median of about 330 draws each): the clear there costs nothing measurable.
//!
//! **The panel.** While the display's panel is off (`SET_DISPLAY_POWER`, the guest's display
//! asleep), a draw clears the drawable to black and draws nothing of the guest's: the presents it
//! takes are answered as any draw's, so the guest's compositor runs on as with a lit panel, and
//! its first draw once the panel is on shows what the guest shows then.
//!
//! **The guest's cursor.** Where the window draws the guest's cursor itself, the guest moving it
//! on its own (`reims_vgpu_window::handle`'s module doc), the window's draw puts the guest's glyph
//! over what the display shows, by its alpha, where the guest put its hot spot, in a render pass
//! of its own after the frame's, in the frame's command buffer (`cursor`). The host's reads of
//! the display draw no cursor.
//!
//! **A display the guest does not drive.** The window of a display after display 0
//! ([`Shown`]'s `black_undriven`) draws black while the guest shows nothing on it, no scanout and
//! no presented surface: a guest whose driver drives display 0 alone leaves the others so, and a
//! compositor shows no window that was never drawn into (Wayland maps a surface at its first
//! buffer), so the window would not show at all. Display 0's draw is `None` then, as before.
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
//! **The colour as a table.** A colour with no matrix maps each channel on its own: its output
//! in red depends only on the input's red, and so on. Where the drawable is 8-bit unorm and what
//! the draw reads takes only `n` values per channel (the scanout's bytes, 256; an 8-bit unorm
//! surface, 256; a 10-bit unorm one, 1024, drawn at its own size, so each drawable pixel samples
//! one of its texels), the draw then looks each channel up in a table of every input value
//! (`displayPipeLutFragment`, `displayPipeLutScanoutFragment`) in place of running the stages at
//! each pixel. The table is `n` × 1 pixels of the drawable's format, which the colour's function
//! itself draws, through the very pipeline the pass would use, from a ramp of every value in the
//! read's format (a texture of the surface's format, or 256 scanout pixels of one byte each):
//! entry k is what the pass draws for a pixel of value k, so the drawable's bytes are the
//! pass's. The draw that first needs a table draws it into a new one, ahead of its own pass, in
//! its own command buffer, and so does each draw after it until a command buffer that drew it
//! has completed without error: a draw dropped before its commit, or failed on the GPU, never
//! leaves a table undrawn. A new colour,
//! another read or another drawable format makes the table again. A colour with a matrix, a
//! surface of another format or size, or a drawable of another format, take the pass.
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
//! **A recovered device.** Where the platform recovers the device's GPU (on the Vulkan platform,
//! its lost `VkDevice` made again, `device-recovered`), the textures, pipelines and pooled
//! buffers made before it do not carry the draws' work on: the runtime refuses them on the new
//! device, or reads a `Private` one's contents as zero. So a draw that finds the device's count
//! of recoveries changed since the draws made what they keep lets go of it all (the converter, the
//! colour's functions, pipelines, buffers and table, and the ramps), and makes each again as it
//! needs it ([`ScanoutDraw::forget_if_recovered`]): the window shows the guest's frames through
//! the display's colour as before the recovery, where the kept objects would otherwise refuse
//! every draw, or draw the colour's table from a zero ramp.
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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use reims_vgpu_graphics::Scanout;
use reims_vgpu_graphics::region::Region;
use reims_vgpu_vulkan::core::host::HostPtr;
use reims_vgpu_vulkan::core::platform::blit::{BlitCmd, BlitEncoder as _};
use reims_vgpu_vulkan::core::platform::command::{CommandBuffer as _, CommandQueue as _};
use reims_vgpu_vulkan::core::platform::descriptor::RenderPassDescriptor as _;
use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
use reims_vgpu_vulkan::core::platform::os::vm::HostVm as _;
use reims_vgpu_vulkan::core::platform::render::{RenderCmd, RenderEncoder as _};
use reims_vgpu_vulkan::core::platform::resource::Texture as _;
use reims_vgpu_vulkan::core::platform::types::{
    ClearColor, CommandBufferStatus, MtlOrigin, MtlRegion, MtlSize,
};
use reims_vgpu_vulkan::core::platform::values::{ColorAttachmentSet, TextureDesc};
use reims_vgpu_vulkan::core::platform::{Handle, HostRange, Platform};
use reims_vgpu_vulkan::core::raise::{OnFault, RaiseResult, refusals};
use reims_vgpu_vulkan::core::sync::Mutex;
use reims_vgpu_vulkan::os::OsVm;
use reims_vgpu_window::{DrawnCursor, PixelRect};

use reims_vgpu_vulkan::core::display::pipeline::{DisplayPipelineCache, PipelineKey};
use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
use reims_vgpu_vulkan::core::platform::types::MtlViewport;

use self::cursor::CursorPass;
use super::color::DisplayColor;
use super::decode::unkept::Kind;
use super::{DisplayHold, Drawable, DrawnPresents, Presented, Shown};
use crate::entry::no_catch;
use crate::platform::ApiPlatform;
use reims_vgpu_graphics::wire::{
    RGFX_COLOR_MATRIX, RGFX_COLOR_PLANE_GAMMA, RGFX_COLOR_POST_GAMMA, RGFX_COLOR_PRE_GAMMA,
};
use reims_vgpu_vulkan::core::platform::resource::Library as _;
use reims_vgpu_vulkan::core::platform::values::RenderPipelineDesc;

mod cursor;

/// The code of a draw of a presented surface that sampled nothing, the platform having refused
/// the display's view of the surface's texture (the module doc, "The display's view"), counted
/// per draw.
pub(crate) const VIEW_REFUSED_CODE: &str = "display-view-refused";

/// `MTLLoadActionDontCare`.
const LOAD_ACTION_DONT_CARE: u64 = 0;
/// `MTLLoadActionLoad`.
const LOAD_ACTION_LOAD: u64 = 1;
/// `MTLLoadActionClear`.
const LOAD_ACTION_CLEAR: u64 = 2;
/// `MTLStoreActionStore`.
const STORE_ACTION_STORE: u64 = 1;

/// The load action of a render pass that draws over `at` of `texture`, clipped to it: none where
/// `at` covers every pixel of it, which the draw then writes (no fragment function of the
/// display's discards, and none blends), else a clear to black first (the module doc, "Clears").
fn pass_load<P: Platform>(texture: &P::Texture, at: PixelRect) -> u64 {
    let whole = PixelRect {
        x: 0,
        y: 0,
        width: texture.width(),
        height: texture.height(),
    };
    if at == whole && at.width > 0 && at.height > 0 {
        LOAD_ACTION_DONT_CARE
    } else {
        LOAD_ACTION_CLEAR
    }
}

/// What a graphics display's window sink draws with (the module doc).
pub(crate) struct ScanoutDraw<P: Platform> {
    /// The platform, whose render pass descriptors clear the drawable.
    platform: Arc<P>,
    /// The device the queue and the buffer are on: the platform's default. The host's reads of
    /// the display (`read`), which make their texture on it, read it.
    pub(crate) device: P::Device,
    /// The sink's command queue, unshared ([`new_command_queue_unshared`]): it puts the guest's
    /// scanout and presented surfaces into the window, each draw on a present's path. The host's
    /// reads of the display, which no present waits on, draw on a queue the platform may share
    /// ([`ScanoutDraw::for_reads`]).
    ///
    /// [`new_command_queue_unshared`]: reims_vgpu_vulkan::core::platform::device::GpuDevice::new_command_queue_unshared
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
    /// The device's recoveries ([`GpuDevice::recoveries`]) when the objects below were made: a
    /// draw that finds the count changed makes them again ([`Self::forget_if_recovered`]).
    ///
    /// [`GpuDevice::recoveries`]: reims_vgpu_vulkan::core::platform::device::GpuDevice::recoveries
    made_at: AtomicU64,
    /// The present shaders' pipelines and the strip's two buffers, made at the first surface
    /// whose format a copy cannot carry; `None` inside where the platform refused them.
    converter: Kept<Converter<P>>,
    /// The colour's fragment functions ([`COLOR_FRAGMENTS`]), made at the first draw through a
    /// colour; `None` inside where the platform's library holds them not.
    color_functions: Kept<[P::Function; 4]>,
    /// The colour's pipelines, by what they draw and the drawable's pixel format.
    color_pipelines: Mutex<ColorPipelines<P>>,
    /// The colour's buffers, made for the colour of that count of changes.
    color_buffers: Mutex<Option<(u64, ColorBuffers<P>)>>,
    /// The colour's table last drawn (the module doc).
    color_table: Mutex<Option<ColorTable<P>>>,
    /// The ramps the tables of surfaces are drawn from, by pixel format, made at their first use.
    ramps: Mutex<Vec<(u64, P::Texture)>>,
    /// The scanout's ramp, made at its first use; `None` inside where the platform refused it.
    scanout_ramp: Kept<ScanoutRead<P>>,
    /// What draws the guest's cursor where the window draws it (`cursor`): its pipelines and its
    /// glyph's texture, made at their first use.
    cursor: Mutex<CursorPass<P>>,
    /// For the tests: every colour takes the pass, with no table.
    #[cfg(test)]
    tables_off: AtomicBool,
}

/// An encoder a draw holds open, ended by [`Open::end`]. One a death in the draw unwinds past
/// (the draw's scope takes the death, [`ScanoutDraw::scoped`]) is ended as the unwind drops it, in
/// a refusal scope of its own: Metal asserts at an encoder released without its end, and the
/// command buffer it encodes into is never committed.
pub(super) struct Open<E> {
    /// The encoder, until it is ended.
    encoder: Option<E>,
    /// Its end: `endEncoding`.
    end: fn(&E) -> RaiseResult<()>,
}

impl<E> Open<E> {
    /// `encoder`, open, which `end` ends.
    pub(super) fn new(encoder: E, end: fn(&E) -> RaiseResult<()>) -> Open<E> {
        Open {
            encoder: Some(encoder),
            end,
        }
    }

    /// The encoder, set from [`Open::new`] until [`Open::end`] or the drop takes it.
    pub(super) fn get(&self) -> &E {
        match &self.encoder {
            Some(encoder) => encoder,
            None => unreachable!("an encoder is open until it is taken"),
        }
    }

    /// Ends the encoder.
    pub(super) fn end(mut self) {
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
pub(super) fn end_blit<P: Platform>(encoder: &P::BlitEncoder) -> RaiseResult<()> {
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
    /// `displayPipeLutFragment`: a surface through the colour's table.
    SurfaceTable,
    /// `displayPipeLutScanoutFragment`: the scanout through the colour's table.
    ScanoutTable,
}

impl ColorSource {
    /// Its fragment function's place in [`COLOR_FRAGMENTS`].
    fn function(self) -> usize {
        match self {
            ColorSource::Surface => 0,
            ColorSource::Scanout => 1,
            ColorSource::SurfaceTable => 2,
            ColorSource::ScanoutTable => 3,
        }
    }
}

/// An object a draw makes at its first use and keeps for the next draws, made again after the
/// device's recovery ([`ScanoutDraw::forget_if_recovered`]): `None` inside where the platform
/// refused it, not asked again until then.
struct Kept<T>(Mutex<Option<Option<Arc<T>>>>);

impl<T> Kept<T> {
    /// Nothing made yet.
    fn new() -> Kept<T> {
        Kept(Mutex::new(None))
    }

    /// The object, made by `make` at the first call since the last [`Kept::forget`]; `None`
    /// where `make` answered none.
    fn get_or_make(&self, make: impl FnOnce() -> Option<T>) -> Option<Arc<T>> {
        let mut kept = self.0.lock();
        kept.get_or_insert_with(|| make().map(Arc::new)).clone()
    }

    /// Lets go of the object, or of the refusal: the next [`Kept::get_or_make`] makes it again.
    fn forget(&self) {
        *self.0.lock() = None;
    }
}

/// What a colour's table is drawn from: the scanout's bytes, or a surface of a pixel format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TableSource {
    Scanout,
    Surface(u64),
}

/// The colour's table last drawn (the module doc): what it is for, the colour's count of
/// changes, what it reads and the drawable's pixel format; the table; and whether a command
/// buffer that drew it has completed.
struct ColorTable<P: Platform> {
    made_for: (u64, TableSource, u64),
    texture: P::Texture,
    drawn: Arc<AtomicBool>,
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

/// The colour's fragment functions: a surface's and the scanout's, then their twins that look the
/// colour up in its table.
const COLOR_FRAGMENTS: [&str; 4] = [
    "displayPipeColorFragment",
    "displayPipeColorScanoutFragment",
    "displayPipeLutFragment",
    "displayPipeLutScanoutFragment",
];
/// The drawables' pixel formats a colour's table is drawn in: `MTLPixelFormatRGBA8Unorm` and
/// `MTLPixelFormatBGRA8Unorm`, whose bytes a draw reads back as it wrote them.
const TABLE_DRAWABLES: [u64; 2] = [70, 80];
/// The values each channel of the scanout takes, and so its table's entries.
const SCANOUT_VALUES: u64 = 256;
/// `MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget`, and storage mode private: a table.
const TABLE_USAGE: u64 = 0x1 | 0x4;
const PRIVATE: u64 = 0x20;
/// `MTLTextureUsageShaderRead`: a ramp.
const RAMP_USAGE: u64 = 0x1;

/// The bytes of a ramp's pixel of value `k` in every channel.
type RampPixel = fn(u32) -> [u8; 4];

/// The values each channel of a surface of pixel format `format` takes, where it is one a table
/// can carry: `MTLPixelFormatRGBA8Unorm` (70) and `BGRA8Unorm` (80), 256; `RGB10A2Unorm` (90) and
/// `BGR10A2Unorm` (94), 1024; with the bytes of a ramp's pixel of value `k` in every channel. An
/// sRGB or float format samples to values off the n-step grid, and takes the pass.
fn table_values(format: u64) -> Option<(u64, RampPixel)> {
    match format {
        70 | 80 => Some((256, |k| {
            let k = u8::try_from(k).unwrap_or(u8::MAX);
            [k, k, k, u8::MAX]
        })),
        90 | 94 => Some((1024, |k| {
            let k = k.min(1023);
            (k | (k << 10) | (k << 20) | (3 << 30)).to_le_bytes()
        })),
        _ => None,
    }
}
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
        Some(Self::made(
            platform, device, queue, region, shown, presented,
        ))
    }

    /// The draw on `device` and its `queue`.
    fn made(
        platform: Arc<P>,
        device: P::Device,
        queue: P::CommandQueue,
        region: &Arc<Region>,
        shown: &Arc<Shown>,
        presented: &Arc<Presented<P>>,
    ) -> ScanoutDraw<P> {
        let page = OsVm::new().page_size();
        let made_at = AtomicU64::new(device.recoveries());
        ScanoutDraw {
            platform,
            device,
            queue,
            region: Arc::clone(region),
            page,
            import: Mutex::new(None),
            shown: Arc::clone(shown),
            presented: Arc::clone(presented),
            made_at,
            converter: Kept::new(),
            color_functions: Kept::new(),
            color_pipelines: Mutex::default(),
            color_buffers: Mutex::default(),
            color_table: Mutex::new(None),
            ramps: Mutex::default(),
            scanout_ramp: Kept::new(),
            cursor: Mutex::default(),
            #[cfg(test)]
            tables_off: AtomicBool::new(false),
        }
    }

    /// For the tests: every colour takes the pass from now on, with no table.
    #[cfg(test)]
    pub(crate) fn without_tables(&self) {
        self.tables_off.store(true, Ordering::Relaxed);
    }

    /// Lets go of the objects the draws made and keep (the converter, the colour's functions,
    /// pipelines, buffers and table, the ramps, and the cursor's pipelines and glyph texture)
    /// where the device recovered since they were made ([`GpuDevice::recoveries`], the module
    /// doc, "A recovered device"): the next draw that needs one makes it again on the device's
    /// new GPU device. The scanout's import is a no-copy buffer over the region's pages, which a
    /// recovery carries.
    ///
    /// [`GpuDevice::recoveries`]: reims_vgpu_vulkan::core::platform::device::GpuDevice::recoveries
    fn forget_if_recovered(&self) {
        let recoveries = self.device.recoveries();
        if self.made_at.swap(recoveries, Ordering::AcqRel) == recoveries {
            return;
        }
        log::info!(
            target: "reims_vgpu::graphics",
            "the device recovered: the display's draws make their pipelines, buffers, tables, \
             ramps and cursor again"
        );
        self.converter.forget();
        self.color_functions.forget();
        self.color_pipelines.lock().clear();
        *self.color_buffers.lock() = None;
        *self.color_table.lock() = None;
        self.ramps.lock().clear();
        self.scanout_ramp.forget();
        *self.cursor.lock() = CursorPass::default();
    }

    /// Runs `draw`, a draw of the display, in a refusal scope (the module doc, "A refused draw"):
    /// what it answers, or `None` where a death in it refused it there, logged.
    pub(crate) fn scoped<T>(&self, draw: impl FnOnce() -> Option<T>) -> Option<T> {
        match refusals::refusing(draw) {
            Ok(drawn) => drawn,
            Err(refusal) => {
                // Once per kind of refusal; later ones at debug, counted (`core::logging`).
                log::warn!(
                    target: "reims_vgpu::graphics", kind:% = refusal;
                    "a draw of the display was refused, and the window keeps the frame it shows: \
                     {refusal}"
                );
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
        let buffer = Kind::Scanout.making(|| {
            no_catch(self.device.new_buffer_over_host_memory(
                OnFault::Die,
                bytes,
                0,
                Some(Box::new(move || drop(keep))),
            ))
        });
        if buffer.is_none() {
            log::warn!(
                target: "reims_vgpu::graphics",
                "the platform refused the scanout's pages (0x{length:x} bytes at 0x{start:x} in \
                 the region): the window shows nothing of this scanout"
            );
        }
        buffer
    }

    /// The window's render thread draws into a drawable of `size` pixels, `scale` of them a
    /// point: the window's size, which the display offers the guest, and its screen's scale,
    /// which it tells the guest.
    pub(crate) fn drawn(&self, size: (u64, u64), scale: f64) {
        *self.shown.drawable.lock() = Some(Drawable {
            pixels: size,
            scale,
        });
    }

    /// A command buffer of the sink's queue with the scanout blitted into `texture` at `region`
    /// (the module doc), not committed, and the presents it shows, which the caller hands it
    /// before its commit ([`DrawnPresents::after`]); `None`, with nothing encoded, where the
    /// guest has no scanout or the queue gives no command buffer. The presents the draw shows are
    /// those made so far, whatever it draws: where it shows the scanout in place of a surface it
    /// cannot draw, or nothing, they are answered at the next refresh, so the guest's compositor
    /// does not wait on a window that cannot show them.
    ///
    /// Where the window draws the guest's cursor itself (`cursor`, the guest moving it on its
    /// own), `cursor` is drawn over what the display shows, in a pass of its own after it
    /// ([`Self::encode_cursor`]); never while the display's panel is off.
    ///
    /// It runs in the draw's refusal scope ([`Self::scoped`]): `None` too where a death in it
    /// refused it, nothing committed, and its presents are answered at the next refresh.
    pub(crate) fn encode(
        &self,
        texture: &P::Texture,
        region: PixelRect,
        cursor: Option<&DrawnCursor>,
    ) -> Option<(P::CommandBuffer, DrawnPresents<P>)> {
        self.scoped(|| {
            let mut presents = self.presented.drawn();
            let (command_buffer, hold) = self.encode_shown(texture, region)?;
            if let Some(hold) = hold {
                // The surface's block stays until the draw has read it.
                presents.reading(hold);
            }
            if let Some(cursor) = cursor
                && !self.shown.asleep()
            {
                self.encode_cursor(&command_buffer, texture, region, cursor);
            }
            Some((command_buffer, presents))
        })
    }

    /// The guest's `cursor` drawn into `texture` over the frame `command_buffer` drew there at
    /// `region`, in a render pass of its own that loads it, with the converter's strip and the
    /// cursor's pipeline and glyph texture (`cursor`'s module doc); nothing where the cursor would
    /// not show, or where the platform refuses the converter, the pipeline or the texture.
    fn encode_cursor(
        &self,
        command_buffer: &P::CommandBuffer,
        texture: &P::Texture,
        region: PixelRect,
        cursor: &DrawnCursor,
    ) {
        let Some(converter) = self.converter() else {
            return;
        };
        let prepared = self.cursor.lock().prepare(
            &self.platform,
            &self.device,
            &converter,
            texture,
            region,
            cursor,
        );
        let Some(prepared) = prepared else {
            return;
        };
        let pass = self.platform.render_pass_descriptor();
        no_catch(pass.set_color_attachment(
            OnFault::Die,
            0,
            &ColorAttachmentSet {
                texture: Some(Some(texture)),
                load_action: Some(LOAD_ACTION_LOAD),
                store_action: Some(STORE_ACTION_STORE),
                ..ColorAttachmentSet::default()
            },
        ));
        let Some(encoder) = no_catch(command_buffer.render_encoder(OnFault::Die, &pass)) else {
            return;
        };
        let encoder = Open::new(encoder, end_render::<P>);
        for cmd in [
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&prepared.pipeline),
            },
            RenderCmd::SetViewport {
                viewport: prepared.viewport,
            },
            RenderCmd::SetScissorRect {
                rect: prepared.scissor,
            },
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
                texture: Some(&prepared.texture),
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
        encoder.end();
    }

    /// A command buffer of the sink's queue with what the display shows drawn into `texture` at
    /// `region`, as [`Self::encode`] draws it, not committed, and the hold of the block of the
    /// presented surface it reads, where it draws one: the caller keeps it until the buffer has
    /// finished; all of `texture` cleared to black while the display's panel is off (the module
    /// doc, "The panel"), and on a display after display 0 the guest shows nothing on ("A display
    /// the guest does not drive"). `None`, with nothing encoded, where the guest has no scanout on
    /// display 0 or the queue gives no command buffer, and where the platform refuses the
    /// display's view of the presented surface a pass would sample (the module doc, "The
    /// display's view"). It answers no present, so the host's reads of the display
    /// (`graphics::read`) draw with it.
    pub(crate) fn encode_shown(
        &self,
        texture: &P::Texture,
        region: PixelRect,
    ) -> Option<(P::CommandBuffer, Option<Arc<DisplayHold>>)> {
        self.forget_if_recovered();
        if self.shown.asleep() {
            // The display's panel is off: black, whatever the guest shows.
            let command_buffer = no_catch(self.queue.command_buffer(OnFault::Die))?;
            self.clear(&command_buffer, texture);
            return Some((command_buffer, None));
        }
        let (changes, color) = self.presented.color();
        let color = (!color.is_identity()).then_some((changes, color));
        if let Some((surface, view, hold)) = self.presented.shown() {
            let (from, to) = (surface.pixel_format(), texture.pixel_format());
            self.presented.seen_format(from, to);
            let copied = BGRA8_FORMATS.contains(&from) && BGRA8_FORMATS.contains(&to);
            let drawn = if color.is_none() && copied {
                self.encode_surface(&surface, texture, region)
            } else {
                // A pass samples the surface through the display's view of it (the module doc,
                // "The display's view").
                let Some(read) = view.of(&surface) else {
                    reims_vgpu_vulkan::answer::count_kind(VIEW_REFUSED_CODE);
                    log::warn!(
                        target: "reims_vgpu::graphics", kind:% = VIEW_REFUSED_CODE;
                        "the platform refused the display's view of the presented IOSurface's \
                         texture (pixel format {from}): the window keeps the frame it shows \
                         ({VIEW_REFUSED_CODE})"
                    );
                    return None;
                };
                match &color {
                    Some((changes, color)) => {
                        self.encode_colored(*changes, color, read, texture, region)
                    }
                    None => self.encode_converted(read, texture, region),
                }
            };
            if let Some(drawn) = drawn {
                return Some((drawn, Some(hold)));
            }
            self.presented.refuse_format(from, to);
        }
        if self.shown.black_undriven && self.shown.scanout.lock().is_none() {
            // A display the guest does not drive: black (the module doc).
            let command_buffer = no_catch(self.queue.command_buffer(OnFault::Die))?;
            self.clear(&command_buffer, texture);
            return Some((command_buffer, None));
        }
        let command_buffer = match &color {
            Some((changes, color)) => self.encode_scanout_colored(*changes, color, texture, region),
            None => self.encode_scanout(texture, region),
        }?;
        Some((command_buffer, None))
    }

    /// A command buffer of the sink's queue for a draw of a presented surface.
    fn surface_command_buffer(&self) -> Option<P::CommandBuffer> {
        no_catch(self.queue.command_buffer(OnFault::Die))
    }

    /// A command buffer of the sink's queue with the scanout drawn into `texture` at `region`
    /// through `color` (the module doc), clipped as [`Self::encode_scanout`] clips its blit, after
    /// a clear of the drawable to black where that does not cover it; not committed. `None` where the guest has no scanout, or
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
        // The scanout's offset and stride are multiples of 4 (`reims_vgpu_graphics`'s `SET_SCANOUT`),
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
    /// `region` through `color` (the module doc), after a clear of the drawable to black where
    /// `region` does not cover it; not committed. `None`, with nothing committed, where the platform refuses the colour's
    /// pipeline or buffers.
    fn encode_colored(
        &self,
        changes: u64,
        color: &DisplayColor,
        surface: &P::Texture,
        texture: &P::Texture,
        region: PixelRect,
    ) -> Option<P::CommandBuffer> {
        let command_buffer = self.surface_command_buffer()?;
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
        let functions = self.color_functions.get_or_make(|| {
            let library = self.device.new_shader_library().ok().flatten()?;
            let function = |name| no_catch(library.new_function(OnFault::Die, name));
            Some([
                function(COLOR_FRAGMENTS[0])?,
                function(COLOR_FRAGMENTS[1])?,
                function(COLOR_FRAGMENTS[2])?,
                function(COLOR_FRAGMENTS[3])?,
            ])
        })?;
        let mut pipelines = self.color_pipelines.lock();
        if let Some((_, pipeline)) = pipelines.iter().find(|(key, _)| *key == (source, format)) {
            return Some(pipeline.clone_ref());
        }
        let fragment = &functions[source.function()];
        let made = self
            .device
            .new_render_pipeline_state(&RenderPipelineDesc {
                label: Some(COLOR_LABEL),
                vertex: &converter.pipelines.vertex_function,
                fragment,
                color0_pixel_format: format,
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

    /// Encodes into `command_buffer` a render pass that clears `texture` to black where `at` does
    /// not cover it ([`pass_load`]) and draws `read` over `at` through `color`, the colour of
    /// `changes` changes: through the colour's
    /// table where it has one for them (the module doc), drawn first where it is not yet, else by
    /// the colour's pass; `None`, with no pass encoded, where the platform refuses the colour's
    /// pipeline or buffers.
    fn encode_color_pass(
        &self,
        command_buffer: &P::CommandBuffer,
        changes: u64,
        color: &DisplayColor,
        read: Read<'_, P>,
        texture: &P::Texture,
        at: PixelRect,
    ) -> Option<()> {
        let table = self.color_table(command_buffer, changes, color, read, texture, at);
        self.encode_color_draw(
            command_buffer,
            changes,
            color,
            read,
            table.as_ref(),
            texture,
            at,
        )
    }

    /// The colour's table for drawing `read` into `drawable` over `at` through `color`, the
    /// colour of `changes` changes, drawn into `command_buffer` first where no command buffer
    /// that drew it has completed; `None` where the draw takes the pass (the module doc), or the
    /// platform refuses the table, its ramp or its draw.
    fn color_table(
        &self,
        command_buffer: &P::CommandBuffer,
        changes: u64,
        color: &DisplayColor,
        read: Read<'_, P>,
        drawable: &P::Texture,
        at: PixelRect,
    ) -> Option<P::Texture> {
        #[cfg(test)]
        if self.tables_off.load(Ordering::Relaxed) {
            return None;
        }
        let matrix = usize::try_from(RGFX_COLOR_MATRIX).ok()?;
        let format = drawable.pixel_format();
        if color.stages.get(matrix)?.is_some() || !TABLE_DRAWABLES.contains(&format) {
            return None;
        }
        let (source, values) = match read {
            Read::Scanout(_) => (TableSource::Scanout, SCANOUT_VALUES),
            Read::Surface(surface) => {
                let (values, _) = table_values(surface.pixel_format())?;
                if (surface.width(), surface.height()) != (at.width, at.height) {
                    return None;
                }
                (TableSource::Surface(surface.pixel_format()), values)
            }
        };
        let made_for = (changes, source, format);
        let mut kept = self.color_table.lock();
        let (texture, drawn) = match kept.as_ref() {
            Some(table) if table.made_for == made_for => {
                if table.drawn.load(Ordering::Acquire) {
                    return Some(table.texture.clone_ref());
                }
                (table.texture.clone_ref(), Arc::clone(&table.drawn))
            }
            _ => {
                let texture = self.new_texture(format, values, TABLE_USAGE, Some(PRIVATE))?;
                let drawn = Arc::new(AtomicBool::new(false));
                *kept = Some(ColorTable {
                    made_for,
                    texture: texture.clone_ref(),
                    drawn: Arc::clone(&drawn),
                });
                (texture, drawn)
            }
        };
        drop(kept);
        let whole = PixelRect {
            x: 0,
            y: 0,
            width: values,
            height: 1,
        };
        match source {
            TableSource::Scanout => {
                let ramp = self.scanout_ramp()?;
                self.encode_color_draw(
                    command_buffer,
                    changes,
                    color,
                    Read::Scanout(&*ramp),
                    None,
                    &texture,
                    whole,
                )?;
            }
            TableSource::Surface(surface) => {
                let ramp = self.ramp(surface)?;
                self.encode_color_draw(
                    command_buffer,
                    changes,
                    color,
                    Read::Surface(&ramp),
                    None,
                    &texture,
                    whole,
                )?;
            }
        }
        no_catch(command_buffer.add_completed_handler(
            OnFault::Die,
            Box::new(move |info| {
                if info.status == CommandBufferStatus::COMPLETED {
                    drawn.store(true, Ordering::Release);
                }
            }),
        ));
        Some(texture)
    }

    /// A texture of `format`, `width` × 1 pixels, of `usage` and `options`, on the sink's
    /// device; `None` where the platform refuses it.
    fn new_texture(
        &self,
        format: u64,
        width: u64,
        usage: u64,
        options: Option<u64>,
    ) -> Option<P::Texture> {
        let descriptor = self.platform.texture_descriptor(&TextureDesc {
            pixel_format: format,
            width,
            height: 1,
            usage,
            resource_options: options,
            ..TextureDesc::default()
        });
        no_catch(self.device.new_texture(OnFault::Die, &descriptor))
    }

    /// The ramp a table of a surface of pixel format `format` is drawn from: a texture of that
    /// format whose pixel k holds value k in every channel, made at its first use; `None` where
    /// the format takes no table or the platform refuses the texture.
    fn ramp(&self, format: u64) -> Option<P::Texture> {
        let mut ramps = self.ramps.lock();
        if let Some((_, ramp)) = ramps.iter().find(|(f, _)| *f == format) {
            return Some(ramp.clone_ref());
        }
        let (values, pixel) = table_values(format)?;
        let ramp = self.new_texture(format, values, RAMP_USAGE, None)?;
        let bytes: Vec<u8> = (0..u32::try_from(values).ok()?).flat_map(pixel).collect();
        let length = u64::try_from(bytes.len()).ok()?;
        // SAFETY: the bytes live across the call, which copies them out and keeps no pointer.
        let pixels =
            unsafe { HostRange::new(HostPtr::from_ptr(bytes.as_ptr().cast_mut()), length) };
        no_catch(ramp.replace_region(
            MtlRegion {
                origin: MtlOrigin::new(0, 0, 0),
                size: MtlSize::new(values, 1, 1),
            },
            0,
            0,
            pixels,
            length,
            length,
        ));
        ramps.push((format, ramp.clone_ref()));
        Some(ramp)
    }

    /// The scanout's ramp: 256 pixels whose pixel k holds byte k in every channel, read from the
    /// top left of a table as the scanout's colour function reads the scanout; made at its first
    /// use, `None` where the platform refuses its buffer.
    fn scanout_ramp(&self) -> Option<Arc<ScanoutRead<P>>> {
        self.scanout_ramp.get_or_make(|| {
            let bytes: Vec<u8> = (0..=u8::MAX).flat_map(|k| [k; 4]).collect();
            Some(ScanoutRead {
                buffer: self.buffer_of(&bytes)?,
                origin: (0, 0),
                stride_words: u32::try_from(SCANOUT_VALUES).ok()?,
                first_word: 0,
            })
        })
    }

    /// Encodes into `command_buffer` a render pass that clears `texture` to black where `at` does
    /// not cover it ([`pass_load`]) and draws `read` over `at`, through `table`, the colour's table, where it is given, else through
    /// `color`, the colour of `changes` changes, by the colour's pass; `None`, with no pass
    /// encoded, where the platform refuses the colour's pipeline or buffers.
    #[expect(
        clippy::too_many_arguments,
        reason = "one pass, of the colour's or its table's"
    )]
    fn encode_color_draw(
        &self,
        command_buffer: &P::CommandBuffer,
        changes: u64,
        color: &DisplayColor,
        read: Read<'_, P>,
        table: Option<&P::Texture>,
        texture: &P::Texture,
        at: PixelRect,
    ) -> Option<()> {
        let converter = self.converter()?;
        let source = match (read, table) {
            (Read::Surface(_), None) => ColorSource::Surface,
            (Read::Scanout(_), None) => ColorSource::Scanout,
            (Read::Surface(_), Some(_)) => ColorSource::SurfaceTable,
            (Read::Scanout(_), Some(_)) => ColorSource::ScanoutTable,
        };
        let pipeline = self.color_pipeline(&converter, source, texture.pixel_format())?;
        let buffers = match table {
            None => Some(self.color_buffers(changes, color)?),
            Some(_) => None,
        };
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
                load_action: Some(pass_load::<P>(texture, at)),
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
            ];
            if let Some(buffers) = &buffers {
                cmds.push(RenderCmd::SetFragmentBuffer {
                    buffer: Some(&buffers.properties),
                    offset: 0,
                    index: 0,
                });
                for (index, table) in (1..).zip(&buffers.tables) {
                    cmds.push(RenderCmd::SetFragmentBuffer {
                        buffer: Some(table),
                        offset: 0,
                        index,
                    });
                }
            }
            if let Some(table) = table {
                cmds.push(RenderCmd::SetFragmentTexture {
                    texture: Some(table),
                    index: 1,
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
        let command_buffer = self.surface_command_buffer()?;
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
    fn converter(&self) -> Option<Arc<Converter<P>>> {
        self.converter.get_or_make(|| {
            let pipelines = DisplayPipelineCache::new(self.device.clone_ref())?;
            let buffer = |floats: &[f32]| -> Option<P::Buffer> {
                let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
                let length = u64::try_from(bytes.len()).ok()?;
                let buffer = no_catch(self.device.new_buffer_of_length(OnFault::Die, length, 0))?;
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
    }

    /// A command buffer of the sink's queue with the presented `surface` drawn into `texture` at
    /// `region` by the display's present shaders, which convert its format to the drawable's,
    /// after a clear of the drawable to black where `region` does not cover it ([`pass_load`]);
    /// not committed. `None`, with nothing committed,
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
        let command_buffer = self.surface_command_buffer()?;
        let (texture_width, texture_height) = (texture.width(), texture.height());
        let x = region.x.min(texture_width);
        let y = region.y.min(texture_height);
        let width = region.width.min(texture_width - x);
        let height = region.height.min(texture_height - y);
        let at = PixelRect {
            x,
            y,
            width,
            height,
        };
        let pass = self.platform.render_pass_descriptor();
        no_catch(pass.set_color_attachment(
            OnFault::Die,
            0,
            &ColorAttachmentSet {
                texture: Some(Some(texture)),
                load_action: Some(pass_load::<P>(texture, at)),
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
    use reims_vgpu_vulkan::core::platform::ObjId;
    use reims_vgpu_vulkan::core::platform::types::MtlScissorRect;
    use reims_vgpu_vulkan::core::testkit::calllog::Call;
    use reims_vgpu_vulkan::core::testkit::objects::{CompleteOn, kind};
    use reims_vgpu_vulkan::core::testkit::{Arg, FakePlatform};

    use super::super::color::Table;
    use super::super::{DisplayHold, ShownTexture, ShownView};
    use super::*;

    const FRAGMENT_BUFFER: &str = "setFragmentBuffer:offset:atIndex:";
    const DRAW: &str = "drawPrimitives:vertexStart:vertexCount:";
    const COPY_IN: &str = "copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:\
                           sourceSize:toTexture:destinationSlice:destinationLevel:\
                           destinationOrigin:";
    const NEW_BUFFER: &str = "newBufferWithLength:options:";
    const NEW_TEXTURE: &str = "newTextureWithDescriptor:";
    const NEW_VIEW: &str = "newTextureViewWithPixelFormat:";
    const NEW_PIPELINE: &str = "newRenderPipelineStateWithDescriptor:error:";
    const NAMED: &str = "newFunctionWithName:";
    const FRAGMENT_TEXTURE: &str = "setFragmentTexture:atIndex:";
    const REPLACE: &str = "replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:";

    /// The scanout `rig` shows.
    const SCANOUT: Scanout = Scanout {
        width: 64,
        height: 32,
        stride: 256,
        offset: reims_vgpu_graphics::wire::RGFX_ALLOCATIONS_OFFSET,
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
        let region = Arc::new(Region::new(reims_vgpu_graphics::REGION_SIZE as usize).unwrap());
        let shown = Arc::new(Shown::default());
        *shown.scanout.lock() = Some(SCANOUT);
        let presented = Arc::new(Presented::<FakePlatform>::default());
        let draw = ScanoutDraw::new(Arc::clone(&platform), &region, &shown, &presented).unwrap();
        let drawable = texture(&draw, 80);
        (platform, draw, presented, drawable)
    }

    /// The whole 64 × 32 drawable.
    const WHOLE: PixelRect = PixelRect {
        x: 0,
        y: 0,
        width: 64,
        height: 32,
    };

    /// `presented` shows `surface`, under a hold of its own, whose writes are all the platform's
    /// work where `queued` ([`DisplayHold`]): the hold, weakly, whose strong count is 1 while the
    /// display alone holds it.
    fn show_held(
        presented: &Presented<FakePlatform>,
        surface: &<FakePlatform as Platform>::Texture,
        queued: bool,
    ) -> std::sync::Weak<DisplayHold> {
        let hold = Arc::new(DisplayHold {
            span: None,
            answers: Mutex::default(),
            queued,
        });
        let weak = Arc::downgrade(&hold);
        *presented.texture.lock() = Some(ShownTexture {
            texture: surface.clone_ref(),
            view: Arc::default(),
            hold,
        });
        weak
    }

    /// The committed command buffers not yet completed, once there are `count` of them.
    #[allow(clippy::disallowed_methods)] // test scaffolding: real time bounds the wait for the thread
    fn committed(platform: &FakePlatform, count: usize) -> Vec<ObjId> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            let committed = platform.world().committed();
            if committed.len() >= count {
                assert_eq!(committed.len(), count, "more command buffers than expected");
                return committed;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "{committed:?} of {count}"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    /// On the runtime's own device the draw reads the presented surface itself: the surface's
    /// block goes once the draw has finished, not before, past its scheduling (the testkit's
    /// commit), whether every write of the surface is the platform's work, where the answers the
    /// draw holds go at its scheduling, or not (`DrawnPresents::after`)
    #[test]
    fn on_the_runtimes_device_the_surface_s_block_goes_with_the_draw() {
        for queued in [true, false] {
            let (platform, draw, presented, drawable) = rig();
            let surface = texture(&draw, 80);
            let hold = show_held(&presented, &surface, queued);
            let (command_buffer, presents) = draw.encode(&drawable, WHOLE, None).unwrap();
            assert_eq!(hold.strong_count(), 2, "the draw holds the surface");
            presents.after(&command_buffer);
            no_catch(command_buffer.commit(OnFault::Die));
            assert_eq!(committed(&platform, 1), [command_buffer.id()]);
            assert_eq!(
                hold.strong_count(),
                2,
                "queued {queued}: while the draw runs"
            );
            assert!(platform.world().complete(
                command_buffer.id(),
                CommandBufferStatus::COMPLETED,
                None,
                CompleteOn::Thread
            ));
            assert_eq!(
                hold.strong_count(),
                1,
                "queued {queued}: until it has finished"
            );
        }
    }

    /// With no scanout and no presented surface, display 0's window draws nothing, and the window
    /// of a display after it clears its drawable to black: one render pass, which copies and
    /// draws nothing of the guest's; once the guest sets a scanout there, it blits the scanout
    #[test]
    fn a_display_after_display_0_draws_black_while_the_guest_shows_nothing_there() {
        let platform = Arc::new(FakePlatform::new());
        let region = Arc::new(Region::new(reims_vgpu_graphics::REGION_SIZE as usize).unwrap());
        let presented = Arc::new(Presented::<FakePlatform>::default());
        let count =
            |calls: &[Call], sel: &str| calls.iter().filter(|c| c.sel.starts_with(sel)).count();
        let display_0 = Arc::new(Shown::default());
        let draw =
            ScanoutDraw::new(Arc::clone(&platform), &region, &display_0, &presented).unwrap();
        let drawable = texture(&draw, 80);
        assert!(draw.encode_shown(&drawable, WHOLE).is_none());

        let display_1 = Arc::new(Shown {
            black_undriven: true,
            ..Shown::default()
        });
        let draw =
            ScanoutDraw::new(Arc::clone(&platform), &region, &display_1, &presented).unwrap();
        let log = platform.call_log();
        let mark = log.mark();
        let (_black, hold) = draw.encode_shown(&drawable, WHOLE).expect("a black draw");
        assert!(hold.is_none());
        let calls = log.since(mark);
        assert_eq!(count(&calls, "renderCommandEncoderWithDescriptor:"), 1);
        assert_eq!(count(&calls, "copyFrom"), 0);
        assert_eq!(count(&calls, "drawPrimitives"), 0);

        *display_1.scanout.lock() = Some(SCANOUT);
        let mark = log.mark();
        let _scanout = draw
            .encode_shown(&drawable, WHOLE)
            .expect("the scanout's draw");
        let calls = log.since(mark);
        assert_eq!(count(&calls, "copyFrom"), 1, "{calls:?}");
    }

    /// On the runtime's own device a presented surface is drawn from the runtime's texture, with
    /// no import of its pages and no compare kernel
    #[test]
    fn on_the_runtimes_device_a_surface_is_drawn_from_its_texture() {
        let (platform, draw, presented, drawable) = rig();
        let surface = texture(&draw, 80);
        *presented.texture.lock() = Some(ShownTexture {
            texture: surface.clone_ref(),
            view: Arc::default(),
            hold: Arc::new(DisplayHold {
                span: None,
                answers: Mutex::default(),
                queued: false,
            }),
        });
        let log = platform.call_log();
        let mark = log.mark();
        let _ = draw.encode(&drawable, WHOLE, None).unwrap();
        let calls = log.since(mark);
        assert!(
            !calls
                .iter()
                .any(|c| c.sel.starts_with("newBufferWithBytesNoCopy")
                    || c.sel.starts_with("dispatchThreadgroups"))
        );
        let copies: Vec<_> = (calls.iter())
            .filter(|c| c.sel.starts_with("copyFromTexture"))
            .map(|c| c.args[0].as_obj())
            .collect();
        assert_eq!(copies, [Some(surface.id())]);
    }

    /// A cursor of a 4 × 4 glyph, its hot spot at (1, 1), at (10, 20) in a 64 × 32 mode.
    fn drawn_cursor() -> DrawnCursor {
        DrawnCursor {
            glyph: Arc::new(reims_vgpu_window::Glyph {
                width: 4,
                height: 4,
                hot_spot: (1, 1),
                rgba: vec![0x80; 64],
            }),
            mode: (64, 32),
            at: (10, 20),
        }
    }

    /// Where the window draws the guest's cursor, the draw puts its glyph over the frame, after
    /// the frame's blit, in a render pass of its own: a pipeline of the present shaders that
    /// blends, made once for the drawable's format; the glyph's texture, made and filled once
    /// for each glyph; the glyph's rectangle as the viewport, its hot spot at the cursor, and the
    /// scissor; and one draw of the strip. The next frame with the same glyph makes nothing; a
    /// new glyph makes its texture alone. With no cursor, and while the display's panel is off,
    /// the draw has no pass of the cursor's
    #[test]
    fn the_guest_s_cursor_is_drawn_over_the_frame_in_a_pass_of_its_own() {
        const BEGIN: &str = "renderCommandEncoderWithDescriptor:";
        let (platform, draw, _, drawable) = rig();
        let log = platform.call_log();
        let cursor = drawn_cursor();

        let mark = log.mark();
        let _ = draw.encode(&drawable, WHOLE, Some(&cursor)).unwrap();
        let calls = log.since(mark);
        let sels: Vec<_> = calls.iter().map(|c| c.sel).collect();
        let blit = sels.iter().position(|s| *s == COPY_IN).expect("the blit");
        let begin = sels.iter().position(|s| *s == BEGIN).expect("the pass");
        assert!(blit < begin, "{sels:?}");
        assert_eq!(sels.iter().filter(|s| **s == BEGIN).count(), 1);
        let pipelines: Vec<_> = (calls.iter())
            .filter(|c| c.sel == NEW_PIPELINE)
            .map(|c| format!("{:?}", c.args[0]))
            .collect();
        assert_eq!(pipelines.len(), 1, "{pipelines:?}");
        assert!(pipelines[0].contains("reims-vgpu.display-cursor"));
        assert!(pipelines[0].contains("blending_enabled: true"));
        let filled: Vec<_> = (calls.iter())
            .filter(|c| c.sel == REPLACE)
            .map(|c| c.obj)
            .collect();
        assert_eq!(filled.len(), 1);
        assert_eq!(count_since(&platform, mark, NEW_TEXTURE), 1);
        assert_eq!(fragment_textures(&platform, mark), [(filled[0], 0)]);
        let arg = |sel: &str| {
            (calls.iter())
                .filter(|c| c.sel == sel)
                .map(|c| c.args[0].clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            arg("setViewport:"),
            [Arg::val(&MtlViewport {
                origin_x: 9.0,
                origin_y: 19.0,
                width: 4.0,
                height: 4.0,
                znear: 0.0,
                zfar: 1.0,
            })]
        );
        assert_eq!(
            arg("setScissorRect:"),
            [Arg::val(&MtlScissorRect {
                x: 9,
                y: 19,
                width: 4,
                height: 4,
            })]
        );
        let draws: Vec<_> = (calls.iter()).filter(|c| c.sel == DRAW).collect();
        assert_eq!(draws.len(), 1);
        assert_eq!(
            draws[0].args,
            [Arg::U64(TRIANGLE_STRIP), Arg::U64(0), Arg::U64(4)]
        );

        // The same glyph again: nothing made.
        let mark = log.mark();
        let _ = draw.encode(&drawable, WHOLE, Some(&cursor)).unwrap();
        for made in [NEW_PIPELINE, NEW_TEXTURE, REPLACE] {
            assert_eq!(count_since(&platform, mark, made), 0, "{made}");
        }
        assert_eq!(fragment_textures(&platform, mark), [(filled[0], 0)]);
        assert_eq!(count_since(&platform, mark, DRAW), 1);

        // A new glyph: its texture alone.
        let moved = DrawnCursor {
            at: (11, 20),
            ..drawn_cursor()
        };
        let mark = log.mark();
        let _ = draw.encode(&drawable, WHOLE, Some(&moved)).unwrap();
        assert_eq!(count_since(&platform, mark, NEW_PIPELINE), 0);
        assert_eq!(count_since(&platform, mark, NEW_TEXTURE), 1);
        assert_eq!(count_since(&platform, mark, REPLACE), 1);
        assert_ne!(fragment_textures(&platform, mark), [(filled[0], 0)]);

        // No cursor, and the panel off.
        let mark = log.mark();
        let _ = draw.encode(&drawable, WHOLE, None).unwrap();
        assert_eq!(count_since(&platform, mark, BEGIN), 0);
        draw.shown
            .asleep
            .store(true, std::sync::atomic::Ordering::Release);
        let mark = log.mark();
        let _ = draw.encode(&drawable, WHOLE, Some(&moved)).unwrap();
        assert_eq!(count_since(&platform, mark, DRAW), 0);
    }

    /// A death in the cursor's pass refuses that draw alone: the pass's encoder is ended after
    /// it, nothing is committed, and the next draw draws the frame and the cursor and commits
    #[test]
    fn a_death_in_the_cursor_s_pass_refuses_that_draw_alone() {
        use reims_vgpu_vulkan::core::testkit::Step;
        let (platform, draw, _, drawable) = rig();
        let log = platform.call_log();
        let cursor = drawn_cursor();
        platform.script().push(DRAW, Step::crash());
        let mark = log.mark();
        assert!(draw.encode(&drawable, WHOLE, Some(&cursor)).is_none());
        let sels = log.sels_since(mark);
        let died = sels.iter().position(|s| *s == DRAW).expect("the draw");
        assert_eq!(&sels[died + 1..], ["endEncoding"], "{sels:?}");
        assert!(!sels.contains(&"commit"));

        let mark = log.mark();
        let (command_buffer, presents) =
            (draw.encode(&drawable, WHOLE, Some(&cursor))).expect("a draw");
        presents.after(&command_buffer);
        command_buffer.commit(OnFault::Die).unwrap();
        let sels = log.sels_since(mark);
        assert_eq!(
            sels.iter().filter(|s| **s == COPY_IN).count(),
            1,
            "{sels:?}"
        );
        assert_eq!(sels.iter().filter(|s| **s == DRAW).count(), 1, "{sels:?}");
        assert!(sels.contains(&"commit"));
    }

    /// On the host's GPU, the cursor's pass puts the guest's glyph over the frame by its alpha where the
    /// guest put its hot spot, at a pixel of the drawable to a pixel of the glyph, and keeps
    /// every other pixel of the frame: an opaque pixel of the glyph replaces the frame's, a half
    /// transparent one is blended over it, a transparent one leaves it
    #[test]
    fn on_vulkan_the_cursor_s_pass_blends_the_glyph_over_the_frame() {
        let platform = crate::rt_test_end::platform();
        crate::entry::run(|_| cursor_blends_over_the_frame(&platform));
    }

    /// [`on_vulkan_the_cursor_s_pass_blends_the_glyph_over_the_frame`]'s body, inside the
    /// library's entry.
    fn cursor_blends_over_the_frame<P: ApiPlatform>(platform: &Arc<P>) {
        use reims_vgpu_graphics::wire::RGFX_ALLOCATIONS_OFFSET;

        let (width, height) = (64_u32, 32_u32);
        let region = Arc::new(crate::platform::test_region(
            &**platform,
            reims_vgpu_graphics::REGION_SIZE as usize,
        ));
        // The scanout, BGRA: each pixel its own blue and green, red 0x40.
        let frame = |x: u32, y: u32| [(x * 4) as u8, (y * 8) as u8, 0x40, 0xff];
        let scanout: Vec<u8> = (0..height)
            .flat_map(|y| (0..width).flat_map(move |x| frame(x, y)))
            .collect();
        // SAFETY: the scanout lies inside the region, mapped while it lives.
        unsafe {
            let base = region.base().add(RGFX_ALLOCATIONS_OFFSET as usize);
            std::ptr::copy_nonoverlapping(scanout.as_ptr(), base, scanout.len());
        }
        let shown = Arc::new(Shown::default());
        *shown.scanout.lock() = Some(Scanout {
            width,
            height,
            stride: width * 4,
            offset: RGFX_ALLOCATIONS_OFFSET,
        });
        let presented = Arc::new(Presented::<P>::default());
        let draw = ScanoutDraw::new(Arc::clone(platform), &region, &shown, &presented).unwrap();
        // A 4 × 2 glyph, RGBA: opaque white, opaque red, half transparent white, transparent;
        // then opaque blue, half transparent black, transparent, opaque green.
        let glyph: Vec<u8> = [
            [0xff, 0xff, 0xff, 0xff],
            [0xff, 0x00, 0x00, 0xff],
            [0xff, 0xff, 0xff, 0x80],
            [0x12, 0x34, 0x56, 0x00],
            [0x00, 0x00, 0xff, 0xff],
            [0x00, 0x00, 0x00, 0x80],
            [0xff, 0xff, 0xff, 0x00],
            [0x00, 0xff, 0x00, 0xff],
        ]
        .concat();
        let cursor = DrawnCursor {
            glyph: Arc::new(reims_vgpu_window::Glyph {
                width: 4,
                height: 2,
                hot_spot: (1, 1),
                rgba: glyph.clone(),
            }),
            mode: (64, 32),
            at: (21, 11),
        };
        let descriptor = platform.texture_descriptor(&TextureDesc {
            pixel_format: 80,
            width: u64::from(width),
            height: u64::from(height),
            usage: 0x1 | 0x4,
            resource_options: Some(0),
            ..TextureDesc::default()
        });
        let target = (platform.metal_device(None).unwrap())
            .new_texture(OnFault::Die, &descriptor)
            .unwrap()
            .unwrap();
        let image = PixelRect {
            x: 0,
            y: 0,
            width: u64::from(width),
            height: u64::from(height),
        };
        let (command_buffer, presents) = draw.encode(&target, image, Some(&cursor)).unwrap();
        command_buffer.commit(OnFault::Die).unwrap();
        command_buffer.wait_until_completed(OnFault::Die).unwrap();
        drop(presents);
        let mut bytes = vec![0; (width * height * 4) as usize];
        platform.read_pixels(&target, &mut bytes, (width * 4) as usize);
        // The glyph's top left is at (20, 10).
        for y in 0..height {
            for x in 0..width {
                let at = ((y * width + x) * 4) as usize;
                let got = &bytes[at..at + 4];
                let under = frame(x, y);
                let expected = if (20..24).contains(&x) && (10..12).contains(&y) {
                    let g = (((y - 10) * 4 + (x - 20)) * 4) as usize;
                    let [r, gr, b, a] = [glyph[g], glyph[g + 1], glyph[g + 2], glyph[g + 3]];
                    let a = f32::from(a) / 255.0;
                    let over = |c: u8, d: u8| f32::from(c) * a + f32::from(d) * (1.0 - a);
                    // The drawable is BGRA.
                    [
                        over(b, under[0]),
                        over(gr, under[1]),
                        over(r, under[2]),
                        255.0,
                    ]
                } else {
                    under.map(f32::from)
                };
                let off = (got.iter().zip(expected))
                    .map(|(g, e)| (f32::from(*g) - e).abs())
                    .fold(0.0, f32::max);
                assert!(
                    off <= 2.0,
                    "pixel ({x}, {y}): {got:?}, expected {expected:?}"
                );
            }
        }
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
    /// ended after the death, nothing is committed, and the next draw draws. The colour's pass,
    /// with no table (a death in a draw through the colour's table:
    /// `a_death_in_a_table_s_draw_refuses_that_draw_alone`)
    #[test]
    fn a_death_in_a_draw_refuses_that_draw_alone() {
        use reims_vgpu_vulkan::core::testkit::Step;
        let (platform, draw, presented, drawable) = rig();
        draw.without_tables();
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
            assert!(draw.encode(&drawable, region, None).is_none());
            let sels = log.sels_since(mark);
            let died = sels
                .iter()
                .position(|s| *s == at)
                .expect("the call that died");
            assert_eq!(&sels[died + 1..], ["endEncoding"], "{sels:?}");
            assert!(!sels.contains(&"commit"));

            let mark = log.mark();
            let (command_buffer, presents) = draw.encode(&drawable, region, None).expect("a draw");
            presents.after(&command_buffer);
            command_buffer.commit(OnFault::Die).unwrap();
            let sels = log.sels_since(mark);
            assert_eq!(sels.iter().filter(|s| **s == at).count(), 1, "{sels:?}");
            assert!(sels.contains(&"commit"));
        }
    }

    /// A death in a draw through the colour's table refuses that draw alone, wherever it comes:
    /// at the upload of the ramp the table is drawn from, at the draw of the table, or at the draw
    /// of the frame through it. Every encoder the draw began is ended, the one open at the death
    /// after it, nothing is committed, and the next draw draws the table and the frame through it
    /// and commits
    #[test]
    fn a_death_in_a_table_s_draw_refuses_that_draw_alone() {
        use reims_vgpu_vulkan::core::testkit::Step;
        const BEGIN: &str = "renderCommandEncoderWithDescriptor:";
        let (platform, draw, presented, drawable) = rig();
        let log = platform.call_log();
        let region = PixelRect {
            x: 0,
            y: 0,
            width: 64,
            height: 32,
        };
        let surface = texture(&draw, 94);
        *presented.texture.lock() = Some(ShownTexture {
            texture: surface.clone_ref(),
            view: Arc::default(),
            hold: Arc::new(DisplayHold {
                span: None,
                answers: Mutex::default(),
                queued: false,
            }),
        });
        let post: Table = Arc::from(vec![[0.0, 0.0, 0.0, 0.0], [0.5, 0.25, 1.0, 0.0]]);
        // Where it dies, and the calls of that selector before it that go through.
        for (changes, (at, before)) in (1..).zip([(REPLACE, 0), (DRAW, 0), (DRAW, 1)]) {
            set_color(
                &presented,
                changes,
                [None, None, None, Some(Arc::clone(&post))],
            );
            for _ in 0..before {
                platform.script().push(at, Step::Default);
            }
            platform.script().push(at, Step::crash());
            let mark = log.mark();
            assert!(
                draw.encode(&drawable, region, None).is_none(),
                "{at} {before}"
            );
            let sels = log.sels_since(mark);
            let died = sels
                .iter()
                .enumerate()
                .filter(|(_, s)| **s == at)
                .nth(before)
                .map(|(i, _)| i)
                .expect("the call that died");
            let ended = sels.iter().filter(|s| **s == "endEncoding").count();
            assert_eq!(
                sels.iter().filter(|s| **s == BEGIN).count(),
                ended,
                "every encoder ended: {sels:?}"
            );
            if at == DRAW {
                assert_eq!(&sels[died + 1..], ["endEncoding"], "{sels:?}");
            }
            assert!(!sels.contains(&"commit"), "{sels:?}");

            let mark = log.mark();
            let (command_buffer, presents) = draw.encode(&drawable, region, None).expect("a draw");
            presents.after(&command_buffer);
            command_buffer.commit(OnFault::Die).unwrap();
            let sels = log.sels_since(mark);
            assert_eq!(count_since(&platform, mark, DRAW), 2, "{sels:?}");
            assert!(sels.contains(&"commit"), "{sels:?}");
        }
    }

    /// With every stage off, the scanout is blitted into the drawable and nothing is drawn; with
    /// a stage on, it is drawn into the drawable by the scanout's colour function, reading the
    /// imported pages in place (no copy), with the properties at fragment buffer 0 (the matrix's
    /// rows, each stage's entry count), the gamma tables at 1 to 3, the pages at 4 and the
    /// scanout's place at 5 (the drawable pixel of its top left, its row and first pixel in
    /// words); the next draw of the same colour makes no table, texture or pipeline, and a change
    /// of the colour makes its buffers again. The colour's pass, with no table (the module doc;
    /// `a_color_with_no_matrix_draws_the_scanout_through_its_table`)
    #[test]
    fn a_color_draws_the_scanout_through_its_tables() {
        let (platform, draw, presented, drawable) = rig();
        draw.without_tables();
        let log = platform.call_log();
        let region = PixelRect {
            x: 0,
            y: 0,
            width: 64,
            height: 32,
        };

        let mark = log.mark();
        let _ = draw.encode(&drawable, region, None).unwrap();
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
        let _ = draw.encode(&drawable, region, None).unwrap();
        let calls = log.since(mark);
        let named: Vec<_> = calls
            .iter()
            .filter(|c| c.sel == NAMED)
            .map(|c| c.args[0].clone())
            .collect();
        // After the present shaders' (the converter's, made at its first use), the colour's four.
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
        let _ = draw.encode(&drawable, region, None).unwrap();
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
        let _ = draw.encode(&drawable, corner, None).unwrap();
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
        let _ = draw.encode(&drawable, region, None).unwrap();
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
    /// function, in place of the copy a BGRA8 surface takes, sampled through the display's view
    /// of the surface (the module doc, "The display's view"). The colour's pass, with no table
    /// (`a_surface_takes_the_table_at_its_own_size_in_unorm`)
    #[test]
    fn a_color_draws_a_presented_surface_through_its_tables() {
        let (platform, draw, presented, drawable) = rig();
        draw.without_tables();
        let log = platform.call_log();
        let surface = texture(&draw, 80);
        let view = Arc::<ShownView<FakePlatform>>::default();
        *presented.texture.lock() = Some(ShownTexture {
            texture: surface.clone_ref(),
            view: Arc::clone(&view),
            hold: Arc::new(DisplayHold {
                span: None,
                answers: Mutex::default(),
                queued: false,
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
        let _ = draw.encode(&drawable, region, None).unwrap();
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
        let made = view.made().expect("the display's view");
        assert_eq!(sampled, [Some(made.id())]);
        assert_ne!(made.id(), surface.id());
        let bound = fragment_buffers(&platform, mark);
        assert_eq!(
            bound.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );
    }

    /// The textures a draw since `mark` bound to its fragment stage, with their indices.
    fn fragment_textures(platform: &FakePlatform, mark: u64) -> Vec<(ObjId, u64)> {
        platform
            .call_log()
            .since(mark)
            .into_iter()
            .filter(|c| c.sel == FRAGMENT_TEXTURE)
            .map(|c| (c.args[0].as_obj().unwrap(), c.args[1].as_u64().unwrap()))
            .collect()
    }

    /// How many calls of `sel` since `mark`.
    fn count_since(platform: &FakePlatform, mark: u64, sel: &str) -> usize {
        (platform.call_log().sels_since(mark).iter())
            .filter(|s| **s == sel)
            .count()
    }

    /// A colour with no matrix draws the scanout through its table (the module doc). The first
    /// draw makes the table, 256 × 1 of the drawable's format, and draws it by the scanout's
    /// colour function with the colour's buffers (0 to 3) from the ramp of every byte (its pixels
    /// at 4, its place at 5: the table's top left, a row of 256 words); then it draws the scanout
    /// through the table by the table's function: the pages at 4, their place at 5, the table at
    /// texture 1, and no colour buffer. Each draw draws the table again until a command buffer
    /// that drew it has completed, then the pass alone runs; a new colour makes a new table, and
    /// a colour with a matrix takes the colour's pass
    #[test]
    fn a_color_with_no_matrix_draws_the_scanout_through_its_table() {
        let (platform, draw, presented, drawable) = rig();
        let log = platform.call_log();
        let region = PixelRect {
            x: 0,
            y: 0,
            width: 64,
            height: 32,
        };
        let (pages, _) = draw.scanout_buffer(&SCANOUT).unwrap();
        let post: Table = Arc::from(vec![[0.0, 0.0, 0.0, 0.0], [0.5, 0.25, 1.0, 0.0]]);
        set_color(&presented, 1, [None, None, None, Some(Arc::clone(&post))]);
        let made = keep_new_buffers(&platform, 32);

        let mark = log.mark();
        let (first, _) = draw.encode(&drawable, region, None).unwrap();
        assert_eq!(
            count_since(&platform, mark, DRAW),
            2,
            "the table, then the pass"
        );
        assert_eq!(count_since(&platform, mark, NEW_TEXTURE), 1);
        let table = (draw.color_table.lock().as_ref())
            .map(|t| t.texture.clone_ref())
            .unwrap();
        assert_eq!(
            (table.width(), table.height(), table.pixel_format()),
            (256, 1, 80)
        );
        let bound = fragment_buffers(&platform, mark);
        assert_eq!(
            bound.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            [0, 1, 2, 3, 4, 5, 4, 5]
        );
        let ramp = draw.scanout_ramp().unwrap();
        assert_eq!(bound[4].1, ramp.buffer.id());
        assert_eq!(
            bytes_of(&ramp.buffer),
            (0..=u8::MAX).flat_map(|k| [k; 4]).collect::<Vec<_>>()
        );
        assert_eq!(scanout_words(&made, bound[5].1), [0, 0, 256, 0]);
        assert_eq!(bound[6].1, pages.id());
        assert_eq!(scanout_words(&made, bound[7].1), [0, 0, 64, 0]);
        assert_eq!(fragment_textures(&platform, mark), [(table.id(), 1)]);

        // Not completed yet: the next draw draws the table again, into the same texture.
        let mark = log.mark();
        let _ = draw.encode(&drawable, region, None).unwrap();
        assert_eq!(count_since(&platform, mark, DRAW), 2);
        assert_eq!(count_since(&platform, mark, NEW_TEXTURE), 0);

        // Once a buffer that drew it has completed, the pass alone.
        platform.world().set_auto_complete(Some(CompleteOn::Caller));
        first.commit(OnFault::Die).unwrap();
        let mark = log.mark();
        let _ = draw.encode(&drawable, region, None).unwrap();
        assert_eq!(count_since(&platform, mark, DRAW), 1);
        assert_eq!(
            count_since(&platform, mark, NEW_BUFFER),
            1,
            "the scanout's place"
        );
        assert_eq!(count_since(&platform, mark, NEW_TEXTURE), 0);
        assert_eq!(count_since(&platform, mark, NEW_PIPELINE), 0);
        assert_eq!(
            fragment_buffers(&platform, mark)
                .iter()
                .map(|(i, _)| *i)
                .collect::<Vec<_>>(),
            [4, 5]
        );
        assert_eq!(fragment_textures(&platform, mark), [(table.id(), 1)]);

        // A new colour: a new table.
        let plane: Table = Arc::from(vec![[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]);
        set_color(&presented, 2, [Some(plane), None, None, None]);
        let mark = log.mark();
        let _ = draw.encode(&drawable, region, None).unwrap();
        assert_eq!(count_since(&platform, mark, DRAW), 2);
        assert_eq!(count_since(&platform, mark, NEW_TEXTURE), 1);

        // A matrix: the colour's pass, its buffers bound, no table.
        let rows: Table = Arc::from(vec![
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]);
        set_color(&presented, 3, [None, None, Some(rows), Some(post)]);
        let mark = log.mark();
        let _ = draw.encode(&drawable, region, None).unwrap();
        assert_eq!(count_since(&platform, mark, DRAW), 1);
        assert_eq!(count_since(&platform, mark, NEW_TEXTURE), 0);
        assert_eq!(
            fragment_buffers(&platform, mark)
                .iter()
                .map(|(i, _)| *i)
                .collect::<Vec<_>>(),
            [0, 1, 2, 3, 4, 5]
        );
        assert!(fragment_textures(&platform, mark).is_empty());
    }

    /// A presented surface takes the colour's table where its format is 8-bit or 10-bit unorm
    /// and it is drawn at its own size: the table, of 256 or 1024 entries, is drawn from a ramp
    /// of the surface's format (made once for each format, its width the table's, pixel k value
    /// k in every channel) by the surface's colour function, then the surface through the table
    /// at texture 1, sampled through the display's view of it, made at the first draw alone. A
    /// surface drawn at another size, or of an sRGB format, takes the pass
    #[test]
    fn a_surface_takes_the_table_at_its_own_size_in_unorm() {
        let (platform, draw, presented, drawable) = rig();
        let log = platform.call_log();
        let region = PixelRect {
            x: 0,
            y: 0,
            width: 64,
            height: 32,
        };
        let post: Table = Arc::from(vec![[0.0, 0.0, 0.0, 0.0], [0.5, 0.25, 1.0, 0.0]]);
        set_color(&presented, 1, [None, None, None, Some(post)]);
        let show = |surface: &<FakePlatform as Platform>::Texture| {
            let view = Arc::<ShownView<FakePlatform>>::default();
            *presented.texture.lock() = Some(ShownTexture {
                texture: surface.clone_ref(),
                view: Arc::clone(&view),
                hold: Arc::new(DisplayHold {
                    span: None,
                    answers: Mutex::default(),
                    queued: false,
                }),
            });
            view
        };
        for (format, values) in [(80, 256), (70, 256), (94, 1024), (90, 1024)] {
            let surface = texture(&draw, format);
            let view = show(&surface);
            let mark = log.mark();
            let _ = draw.encode(&drawable, region, None).unwrap();
            assert_eq!(count_since(&platform, mark, DRAW), 2, "{format}");
            assert_eq!(count_since(&platform, mark, NEW_VIEW), 1, "{format}");
            let viewed = view.made().expect("the display's view").clone_ref();
            assert_eq!(viewed.pixel_format(), format);
            assert_eq!(
                count_since(&platform, mark, NEW_TEXTURE),
                2,
                "{format}: ramp, table"
            );
            assert_eq!(count_since(&platform, mark, REPLACE), 1, "{format}");
            let ramp = (draw.ramps.lock().iter())
                .find(|(f, _)| *f == format)
                .map(|(_, r)| r.clone_ref())
                .unwrap();
            let table = (draw.color_table.lock().as_ref())
                .map(|t| t.texture.clone_ref())
                .unwrap();
            assert_eq!(
                (ramp.width(), ramp.height(), ramp.pixel_format()),
                (values, 1, format)
            );
            assert_eq!(
                (table.width(), table.height(), table.pixel_format()),
                (values, 1, 80)
            );
            assert_eq!(
                fragment_textures(&platform, mark),
                [(ramp.id(), 0), (table.id(), 1), (viewed.id(), 0)],
                "{format}"
            );
            // Again: the ramp and the view are kept.
            let mark = log.mark();
            let _ = draw.encode(&drawable, region, None).unwrap();
            assert_eq!(count_since(&platform, mark, NEW_TEXTURE), 0, "{format}");
            assert_eq!(count_since(&platform, mark, NEW_VIEW), 0, "{format}");
            assert_eq!(
                fragment_textures(&platform, mark),
                [(ramp.id(), 0), (table.id(), 1), (viewed.id(), 0)],
                "{format}"
            );
        }
        let (_, pixel) = table_values(94).unwrap();
        assert_eq!(
            pixel(5),
            (5u32 | (5 << 10) | (5 << 20) | (3 << 30)).to_le_bytes()
        );
        let (_, pixel) = table_values(80).unwrap();
        assert_eq!(pixel(7), [7, 7, 7, 0xff]);

        // At another size, and in sRGB: the pass.
        let surface = texture(&draw, 80);
        show(&surface);
        let quarter = PixelRect {
            x: 0,
            y: 0,
            width: 32,
            height: 16,
        };
        let srgb = texture(&draw, 81);
        for (shown, at) in [(&surface, quarter), (&srgb, region)] {
            let view = show(shown);
            let mark = log.mark();
            let _ = draw.encode(&drawable, at, None).unwrap();
            assert_eq!(count_since(&platform, mark, DRAW), 1);
            let viewed = view.made().expect("the display's view");
            assert_eq!(fragment_textures(&platform, mark), [(viewed.id(), 0)]);
        }
    }

    /// A presented surface a pass draws, one the guest placed at its index 7, is sampled through
    /// the display's view of it, a texture at an index of the platform's own, made at the pass's
    /// first draw and kept: once the guest has let go of the surface and placed a new texture at
    /// 7, the next draw makes no view and binds the same one, neither texture at 7
    #[test]
    fn a_pass_samples_a_presented_surface_through_a_view_at_an_index_of_its_own() {
        let (platform, draw, presented, drawable) = rig();
        let log = platform.call_log();
        let placed = || {
            let descriptor = draw.platform.texture_descriptor(&TextureDesc {
                pixel_format: 70,
                width: 64,
                height: 32,
                resource_index: Some((7, true)),
                ..TextureDesc::default()
            });
            (draw.device.new_texture(OnFault::Die, &descriptor))
                .unwrap()
                .unwrap()
        };
        let surface = placed();
        assert_eq!(surface.resource_index(), 7);
        let view = Arc::<ShownView<FakePlatform>>::default();
        *presented.texture.lock() = Some(ShownTexture {
            texture: surface.clone_ref(),
            view: Arc::clone(&view),
            hold: Arc::new(DisplayHold {
                span: None,
                answers: Mutex::default(),
                queued: false,
            }),
        });
        let mark = log.mark();
        let _ = draw.encode(&drawable, WHOLE, None).expect("a draw");
        assert_eq!(count_since(&platform, mark, NEW_VIEW), 1);
        let viewed = view.made().expect("the display's view").clone_ref();
        assert_ne!(viewed.resource_index(), 7);
        assert_eq!(viewed.pixel_format(), 70);
        assert_eq!(fragment_textures(&platform, mark), [(viewed.id(), 0)]);

        // The guest lets go of the surface, which the display still shows, and places a new
        // texture at its index.
        drop(surface);
        let next = placed();
        assert_eq!(next.resource_index(), 7);
        let mark = log.mark();
        let _ = draw.encode(&drawable, WHOLE, None).expect("a draw");
        assert_eq!(count_since(&platform, mark, NEW_VIEW), 0);
        assert_eq!(fragment_textures(&platform, mark), [(viewed.id(), 0)]);
    }

    /// Where the platform refuses the display's view of a presented surface a pass would sample,
    /// the draw answers nothing, the refusal counted: nothing is encoded or committed, and the
    /// scanout is not drawn in its place. The next draw makes the view and draws through it
    #[test]
    fn a_refused_view_of_a_presented_surface_draws_nothing_and_is_counted() {
        use reims_vgpu_vulkan::core::testkit::Step;
        let (platform, draw, presented, drawable) = rig();
        let log = platform.call_log();
        let surface = texture(&draw, 70);
        let view = Arc::<ShownView<FakePlatform>>::default();
        *presented.texture.lock() = Some(ShownTexture {
            texture: surface.clone_ref(),
            view: Arc::clone(&view),
            hold: Arc::new(DisplayHold {
                span: None,
                answers: Mutex::default(),
                queued: false,
            }),
        });
        let refused = || {
            (reims_vgpu_vulkan::answer::degradation_counts().into_iter())
                .find(|(code, _)| code == VIEW_REFUSED_CODE)
                .map_or(0, |(_, n)| n)
        };
        let before = refused();
        platform.script().push(NEW_VIEW, Step::Nil);
        let mark = log.mark();
        assert!(draw.encode(&drawable, WHOLE, None).is_none());
        assert_eq!(refused(), before + 1);
        assert!(view.made().is_none());
        let sels = log.sels_since(mark);
        assert_eq!(sels, [NEW_VIEW], "{sels:?}");

        let mark = log.mark();
        let (command_buffer, presents) = draw.encode(&drawable, WHOLE, None).expect("a draw");
        presents.after(&command_buffer);
        command_buffer.commit(OnFault::Die).unwrap();
        assert_eq!(refused(), before + 1);
        let viewed = view.made().expect("the display's view");
        assert_eq!(fragment_textures(&platform, mark), [(viewed.id(), 0)]);
        assert!(log.sels_since(mark).contains(&"commit"));
    }

    /// While the display shows a presented surface the guest placed at its index, the guest lets
    /// go of it and places a new texture of other pixels at that index: the display's next read
    /// still reads the shown surface's pixels, through the display's view of it, and the new
    /// texture's once the display shows it
    #[test]
    fn on_vulkan_a_shown_surface_reads_its_own_pixels_after_its_index_is_placed_again() {
        let platform = crate::rt_test_end::platform();
        crate::entry::run(|_| shown_surface_reads_its_own_pixels(&platform));
    }

    /// [`on_vulkan_a_shown_surface_reads_its_own_pixels_after_its_index_is_placed_again`]'s body,
    /// inside the library's entry.
    fn shown_surface_reads_its_own_pixels<P: ApiPlatform>(platform: &Arc<P>) {
        const W: u32 = 64;
        const H: u32 = 32;
        /// The guest's index the surfaces are placed at.
        const PLACED: u64 = 4321;
        let device = platform.metal_device(None).unwrap();
        device.reserve_resource_indices(0, &[PLACED]).unwrap();
        let region = Arc::new(crate::platform::test_region(
            &**platform,
            reims_vgpu_graphics::REGION_SIZE as usize,
        ));
        let shown = Arc::new(Shown::default());
        let presented = Arc::new(Presented::<P>::default());
        let draw = ScanoutDraw::new(Arc::clone(platform), &region, &shown, &presented).unwrap();
        let whole = PixelRect {
            x: 0,
            y: 0,
            width: u64::from(W),
            height: u64::from(H),
        };
        // An RGBA8 texture of one colour at the placed index: the display converts it.
        let placed = |rgba: [u8; 4]| {
            let descriptor = platform.texture_descriptor(&TextureDesc {
                pixel_format: 70,
                width: u64::from(W),
                height: u64::from(H),
                usage: 0x1,
                resource_options: Some(0),
                resource_index: Some((PLACED, true)),
                ..TextureDesc::default()
            });
            let texture = (device.new_texture(OnFault::Die, &descriptor))
                .unwrap()
                .unwrap();
            assert_eq!(texture.resource_index(), PLACED);
            let bytes: Vec<u8> = rgba.repeat((W * H) as usize);
            let length = bytes.len() as u64;
            // SAFETY: the bytes live across the call, which copies them out.
            let range =
                unsafe { HostRange::new(HostPtr::from_ptr(bytes.as_ptr().cast_mut()), length) };
            texture
                .replace_region(
                    MtlRegion {
                        origin: MtlOrigin::new(0, 0, 0),
                        size: MtlSize::new(u64::from(W), u64::from(H), 1),
                    },
                    0,
                    0,
                    range,
                    u64::from(W) * 4,
                    length,
                )
                .unwrap();
            texture
        };
        let show = |texture: &P::Texture| {
            *presented.texture.lock() = Some(ShownTexture {
                texture: texture.clone_ref(),
                view: Arc::default(),
                hold: Arc::new(DisplayHold {
                    span: None,
                    answers: Mutex::default(),
                    queued: false,
                }),
            });
        };
        // The display's read, into a BGRA8 texture.
        let read = || -> Vec<u8> {
            let descriptor = platform.texture_descriptor(&TextureDesc {
                pixel_format: 80,
                width: u64::from(W),
                height: u64::from(H),
                usage: 0x1 | 0x4,
                resource_options: Some(0),
                ..TextureDesc::default()
            });
            let target = (device.new_texture(OnFault::Die, &descriptor))
                .unwrap()
                .unwrap();
            let (command_buffer, hold) = draw.encode_shown(&target, whole).expect("a draw");
            command_buffer.commit(OnFault::Die).unwrap();
            command_buffer.wait_until_completed(OnFault::Die).unwrap();
            drop(hold);
            let mut bytes = vec![0; (W * H * 4) as usize];
            platform.read_pixels(&target, &mut bytes, (W * 4) as usize);
            bytes
        };
        let first = [0x10, 0x80, 0xf0, 0xff];
        let second = [0xe0, 0x30, 0x50, 0xff];
        let bgra = |[r, g, b, a]: [u8; 4]| [b, g, r, a].repeat((W * H) as usize);

        let surface = placed(first);
        show(&surface);
        assert_eq!(read(), bgra(first), "the shown surface");
        // The guest lets go of the surface, which the display still shows, and places a new
        // texture at its index.
        drop(surface);
        let next = placed(second);
        assert_eq!(
            read(),
            bgra(first),
            "the shown surface, its index placed again"
        );
        show(&next);
        assert_eq!(read(), bgra(second), "the new texture, shown");
    }

    /// The colour's table draws the drawable's bytes the colour's pass draws: the scanout, and a
    /// surface of each format a table carries, holding every value in each channel across a width
    /// off a power of two (3840), through colours of one stage and of three, of 1 to 1024 entries,
    /// none the identity
    #[test]
    fn on_vulkan_the_table_draws_what_the_pass_draws() {
        the_table_draws_what_the_pass_draws(crate::rt_test_end::platform());
    }

    /// `on_vulkan_the_table_draws_what_the_pass_draws`, on `platform`.
    fn the_table_draws_what_the_pass_draws<P: ApiPlatform>(platform: Arc<P>) {
        crate::entry::run(|_| table_draws_what_the_pass_draws(&platform));
    }

    /// [`the_table_draws_what_the_pass_draws`]'s body, inside the library's entry.
    fn table_draws_what_the_pass_draws<P: ApiPlatform>(platform: &Arc<P>) {
        use reims_vgpu_graphics::wire::RGFX_ALLOCATIONS_OFFSET;

        const W: u32 = 3840;
        const H: u32 = 4;
        let region = Arc::new(crate::platform::test_region(
            &**platform,
            reims_vgpu_graphics::REGION_SIZE as usize,
        ));
        // Channel c of pixel (x, y) at `n` values: each takes every value along a row.
        let value = |c: u32, x: u32, y: u32, n: u32| match c {
            0 => x % n,
            1 => (x * 3 + y * 7) % n,
            _ => n - 1 - (x + y) % n,
        };
        let pixels = |n: u32, pixel: &dyn Fn([u32; 3], u32) -> [u8; 4]| -> Vec<u8> {
            (0..H)
                .flat_map(|y| (0..W).map(move |x| (x, y)))
                .flat_map(|(x, y)| pixel([0, 1, 2].map(|c| value(c, x, y, n)), x % 251))
                .collect()
        };
        let eight = |order: [usize; 3]| {
            move |v: [u32; 3], a: u32| -> [u8; 4] {
                let b = v.map(|c| u8::try_from(c).unwrap());
                [
                    b[order[0]],
                    b[order[1]],
                    b[order[2]],
                    u8::try_from(a).unwrap(),
                ]
            }
        };
        let ten = |v: [u32; 3], a: u32| {
            (v[0] | (v[1] << 10) | (v[2] << 20) | ((a % 4) << 30)).to_le_bytes()
        };
        // The scanout: BGRA bytes.
        let scanout = pixels(256, &eight([2, 1, 0]));
        // SAFETY: the scanout lies inside the region, mapped while it lives.
        unsafe {
            let at = region.base().add(RGFX_ALLOCATIONS_OFFSET as usize);
            std::ptr::copy_nonoverlapping(scanout.as_ptr(), at, scanout.len());
        }
        let shown = Arc::new(Shown::default());
        *shown.scanout.lock() = Some(Scanout {
            width: W,
            height: H,
            stride: W * 4,
            offset: RGFX_ALLOCATIONS_OFFSET,
        });
        let presented = Arc::new(Presented::<P>::default());
        let tabled = ScanoutDraw::new(Arc::clone(platform), &region, &shown, &presented).unwrap();
        let passed = ScanoutDraw::new(Arc::clone(platform), &region, &shown, &presented).unwrap();
        passed.without_tables();
        let new_texture = |format: u64, usage: u64| {
            let descriptor = platform.texture_descriptor(&TextureDesc {
                pixel_format: format,
                width: u64::from(W),
                height: u64::from(H),
                usage,
                resource_options: Some(0),
                ..TextureDesc::default()
            });
            (platform.metal_device(None).unwrap())
                .new_texture(OnFault::Die, &descriptor)
                .unwrap()
                .unwrap()
        };
        let whole = PixelRect {
            x: 0,
            y: 0,
            width: u64::from(W),
            height: u64::from(H),
        };
        let read = |draw: &ScanoutDraw<P>| -> Vec<u8> {
            let target = new_texture(80, 0x1 | 0x4);
            let (command_buffer, hold) = draw.encode_shown(&target, whole).unwrap();
            command_buffer.commit(OnFault::Die).unwrap();
            command_buffer.wait_until_completed(OnFault::Die).unwrap();
            drop(hold);
            let mut bytes = vec![0; (W * H * 4) as usize];
            platform.read_pixels(&target, &mut bytes, (W * 4) as usize);
            bytes
        };
        let curve = |n: usize, power: [f32; 3]| -> Table {
            #[expect(clippy::as_conversions, reason = "small counts")]
            let top = n.saturating_sub(1).max(1) as f32;
            #[expect(clippy::as_conversions, reason = "small counts")]
            let entries = (0..n)
                .map(|k| {
                    let x = k as f32 / top;
                    [x.powf(power[0]), x.powf(power[1]), x.powf(power[2]), 0.0]
                })
                .collect::<Vec<_>>();
            Arc::from(entries)
        };
        let colours: [[Option<Table>; 4]; 4] = [
            // A display profile's plane gamma.
            [
                Some(curve(256, [1.0 / 2.2, 1.0 / 1.8, 1.2])),
                None,
                None,
                None,
            ],
            // Three stages of odd and even counts.
            [
                Some(curve(17, [0.5, 1.5, 1.0])),
                Some(curve(1024, [2.0, 0.7, 1.1])),
                None,
                Some(Arc::from(vec![
                    [1.0, 0.0, 0.25, 0.0],
                    [0.0, 1.0, 0.75, 0.0],
                ])),
            ],
            // One entry: a constant.
            [
                None,
                Some(Arc::from(vec![[0.25, 0.5, 0.75, 0.0]])),
                None,
                None,
            ],
            // A post-gamma past the ends.
            [None, None, None, Some(curve(257, [3.0, 0.3, 1.0]))],
        ];
        let surfaces: [(u64, Vec<u8>); 4] = [
            (80, pixels(256, &eight([2, 1, 0]))),
            (70, pixels(256, &eight([0, 1, 2]))),
            (94, pixels(1024, &ten)),
            (90, pixels(1024, &ten)),
        ];
        let mut changes = 0;
        for source in [None].into_iter().chain(surfaces.iter().map(Some)) {
            *presented.texture.lock() = source.map(|(format, bytes)| {
                let texture = new_texture(*format, 0x1);
                let length = bytes.len() as u64;
                // SAFETY: the bytes live across the call, which copies them out.
                let range =
                    unsafe { HostRange::new(HostPtr::from_ptr(bytes.as_ptr().cast_mut()), length) };
                texture
                    .replace_region(
                        MtlRegion {
                            origin: MtlOrigin::new(0, 0, 0),
                            size: MtlSize::new(u64::from(W), u64::from(H), 1),
                        },
                        0,
                        0,
                        range,
                        u64::from(W) * 4,
                        length,
                    )
                    .unwrap();
                ShownTexture {
                    texture,
                    view: Arc::default(),
                    hold: Arc::new(DisplayHold {
                        span: None,
                        answers: Mutex::default(),
                        queued: false,
                    }),
                }
            });
            let name = source.map_or("the scanout".to_owned(), |(f, _)| format!("format {f}"));
            for (i, stages) in colours.iter().enumerate() {
                changes += 1;
                *presented.color.lock() = (
                    changes,
                    Arc::new(DisplayColor {
                        stages: stages.clone(),
                    }),
                );
                let through = read(&passed);
                let looked_up = read(&tabled);
                assert!(
                    (tabled.color_table.lock().as_ref()).is_some_and(|t| t.made_for.0 == changes),
                    "{name}, colour {i}: no table drawn"
                );
                let differ = (through.iter().zip(&looked_up)).position(|(a, b)| a != b);
                assert_eq!(
                    differ,
                    None,
                    "{name}, colour {i}: byte {differ:?}: pass {:?}, table {:?}",
                    differ.map(|d| &through[d & !3..(d & !3) + 4]),
                    differ.map(|d| &looked_up[d & !3..(d & !3) + 4]),
                );
                // The colour changed the bytes: the comparison is not of two copies.
                let original = source.map_or(&scanout, |(_, b)| b);
                assert!(
                    source.is_some_and(|(f, _)| *f >= 90)
                        || through
                            .chunks(4)
                            .zip(original.chunks(4))
                            .any(|(a, b)| a[..3] != b[..3]),
                    "{name}, colour {i}: the pass changed nothing"
                );
            }
        }
    }

    /// A pass whose draw covers the drawable loads none of its old pixels, and one whose draw
    /// covers part of it, a guest mode smaller than the drawable, clears the rest to black first
    /// ([`pass_load`]): drawn over a drawable that held one value in every pixel and over one that
    /// held another, the drawable reads the same, with black outside the guest's image. A
    /// presented 10-bit surface through the present shaders, through a colour's table and through
    /// a colour's pass, and the scanout through a colour, at odd sizes and at a mode the display
    /// offers in each (its width down to a multiple of 16, its height to even)
    #[test]
    fn on_vulkan_a_pass_that_skips_its_clear_writes_every_pixel_of_the_drawable() {
        let platform = crate::rt_test_end::platform();
        crate::entry::run(|_| a_pass_writes_every_pixel(&platform));
    }

    /// [`on_vulkan_a_pass_that_skips_its_clear_writes_every_pixel_of_the_drawable`]'s body, inside
    /// the library's entry.
    fn a_pass_writes_every_pixel<P: ApiPlatform>(platform: &Arc<P>) {
        use reims_vgpu_graphics::wire::RGFX_ALLOCATIONS_OFFSET;

        let region = Arc::new(crate::platform::test_region(
            &**platform,
            reims_vgpu_graphics::REGION_SIZE as usize,
        ));
        let new_texture = |format: u64, usage: u64, (width, height): (u32, u32)| {
            let descriptor = platform.texture_descriptor(&TextureDesc {
                pixel_format: format,
                width: u64::from(width),
                height: u64::from(height),
                usage,
                resource_options: Some(0),
                ..TextureDesc::default()
            });
            (platform.metal_device(None).unwrap())
                .new_texture(OnFault::Die, &descriptor)
                .unwrap()
                .unwrap()
        };
        let fill = |texture: &P::Texture, bytes: &[u8], (width, height): (u32, u32)| {
            let length = bytes.len() as u64;
            // SAFETY: the bytes live across the call, which copies them out.
            let range =
                unsafe { HostRange::new(HostPtr::from_ptr(bytes.as_ptr().cast_mut()), length) };
            texture
                .replace_region(
                    MtlRegion {
                        origin: MtlOrigin::new(0, 0, 0),
                        size: MtlSize::new(u64::from(width), u64::from(height), 1),
                    },
                    0,
                    0,
                    range,
                    u64::from(width) * 4,
                    length,
                )
                .unwrap();
        };
        let swap: Table = Arc::from(vec![
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]);
        let gamma: Table = Arc::from(vec![[0.0, 0.0, 0.0, 0.0], [1.0, 0.5, 0.25, 0.0]]);
        let colours: [(&str, [Option<Table>; 4]); 3] = [
            ("no colour", [None, None, None, None]),
            ("a table", [Some(gamma), None, None, None]),
            ("a matrix", [None, None, Some(swap), None]),
        ];
        let mut changes = 0;
        for drawable in [(1001_u32, 7_u32), (67, 33), (17, 5), (64, 32)] {
            let modes = [drawable, (drawable.0 / 16 * 16, drawable.1 / 2 * 2)];
            for (i, mode) in modes.into_iter().enumerate() {
                if i == 1 && mode == drawable {
                    continue;
                }
                let (width, height) = mode;
                let pixels = |pixel: &dyn Fn((u32, u32)) -> [u8; 4]| -> Vec<u8> {
                    (0..height)
                        .flat_map(|y| (0..width).map(move |x| (x, y)))
                        .flat_map(pixel)
                        .collect()
                };
                let scanout = pixels(&|(x, y)| {
                    [
                        (x % 256) as u8,
                        (y * 7 % 256) as u8,
                        ((x + y) % 256) as u8,
                        255,
                    ]
                });
                // SAFETY: the scanout lies inside the region, mapped while it lives.
                unsafe {
                    let base = region.base().add(RGFX_ALLOCATIONS_OFFSET as usize);
                    std::ptr::copy_nonoverlapping(scanout.as_ptr(), base, scanout.len());
                }
                let shown = Arc::new(Shown::default());
                *shown.scanout.lock() = Some(Scanout {
                    width,
                    height,
                    stride: width * 4,
                    offset: RGFX_ALLOCATIONS_OFFSET,
                });
                let ten = pixels(&|(x, y)| {
                    ((x % 1024)
                        | ((y * 13 % 1024) << 10)
                        | (((x * 3 + y) % 1024) << 20)
                        | (3 << 30))
                        .to_le_bytes()
                });
                let surface = new_texture(94, 0x1, mode);
                fill(&surface, &ten, mode);
                let presented = Arc::new(Presented::<P>::default());
                let draw =
                    ScanoutDraw::new(Arc::clone(platform), &region, &shown, &presented).unwrap();
                let image = PixelRect {
                    x: 0,
                    y: 0,
                    width: u64::from(width),
                    height: u64::from(height),
                };
                let read = |held: [u8; 4]| -> Vec<u8> {
                    let target = new_texture(80, 0x1 | 0x4, drawable);
                    let old: Vec<u8> = (0..drawable.0 * drawable.1).flat_map(|_| held).collect();
                    fill(&target, &old, drawable);
                    let expected = if mode == drawable {
                        LOAD_ACTION_DONT_CARE
                    } else {
                        LOAD_ACTION_CLEAR
                    };
                    assert_eq!(
                        pass_load::<P>(&target, image),
                        expected,
                        "{drawable:?} {mode:?}"
                    );
                    let (command_buffer, hold) = draw.encode_shown(&target, image).unwrap();
                    command_buffer.commit(OnFault::Die).unwrap();
                    command_buffer.wait_until_completed(OnFault::Die).unwrap();
                    drop(hold);
                    let mut bytes = vec![0; (drawable.0 * drawable.1 * 4) as usize];
                    platform.read_pixels(&target, &mut bytes, (drawable.0 * 4) as usize);
                    bytes
                };
                for (source, colour) in [
                    ("the surface", 0),
                    ("the surface", 1),
                    ("the surface", 2),
                    ("the scanout", 2),
                ] {
                    *presented.texture.lock() = (source == "the surface").then(|| ShownTexture {
                        texture: Handle::clone_ref(&surface),
                        view: Arc::default(),
                        hold: Arc::new(DisplayHold {
                            span: None,
                            answers: Mutex::default(),
                            queued: false,
                        }),
                    });
                    changes += 1;
                    let (name, stages) = &colours[colour];
                    *presented.color.lock() = (
                        changes,
                        Arc::new(DisplayColor {
                            stages: stages.clone(),
                        }),
                    );
                    let case = format!("{source}, {name}, mode {mode:?} in {drawable:?}");
                    let first = read([0x11, 0x22, 0x33, 0x44]);
                    let second = read([0xee, 0xdd, 0xcc, 0xbb]);
                    let differ = (first.chunks(4).zip(second.chunks(4))).position(|(a, b)| a != b);
                    assert_eq!(
                        differ,
                        None,
                        "{case}: pixel {:?} kept what the drawable held: {:?} and {:?}",
                        differ.map(|d| (d % drawable.0 as usize, d / drawable.0 as usize)),
                        differ.map(|d| &first[d * 4..d * 4 + 4]),
                        differ.map(|d| &second[d * 4..d * 4 + 4]),
                    );
                    let outside = (first.chunks(4).enumerate()).find(|(p, px)| {
                        let (x, y) = (*p as u32 % drawable.0, *p as u32 / drawable.0);
                        (x >= width || y >= height) && *px != [0, 0, 0, 255]
                    });
                    assert_eq!(outside, None, "{case}: not black outside the image");
                }
            }
        }
    }
}
