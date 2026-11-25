//! The guest's cursor drawn into the window's frame where the window draws it itself, the guest
//! moving it on its own ([`DrawnCursor`], `reims_vgpu_window::handle`'s module doc): the glyph
//! over the frame the sink drew, by its alpha, its hot spot where the guest's `MOVE_CURSOR` put it.
//!
//! **The pass.** After the frame's blit or pass, in the frame's command buffer, a render pass that
//! loads the drawable draws the converter's strip (positions at vertex buffer 0, texture
//! coordinates at 1) over a viewport that is the glyph's rectangle in the drawable, its hot spot
//! at the cursor's position, a glyph pixel to a guest pixel at the image's scale, with a scissor
//! that clips it to the image and the drawable ([`placement`]). Its pipeline is the present
//! shaders' (`displayPresentVertex` and `displayPresentFragment`, which samples the glyph filtered
//! where the image scales it), blending over by alpha, premultiplied: both source factors `one`,
//! both destination factors `oneMinusSourceAlpha`. The guest's alpha is straight, and the texture
//! holds the glyph premultiplied, so that the filter, where the image scales the glyph, does not
//! darken its edges with the colour of its transparent pixels: at one texel a pixel, each pixel is
//! the glyph's colour over the frame's by its alpha, as with straight alpha. The glyph's colours
//! are drawn as the guest gives them, through no colour of the display pipe's, as the host's
//! cursor shows them over the image. The host's reads of the display (`read`) draw no cursor:
//! QEMU's console takes the cursor apart from the frame.
//!
//! **What the guest reads.** The pass makes no buffer: it binds the converter's. The glyph's
//! texture is the platform's own ([`ApiPlatform::own_texture`]): on the Vulkan platform one that
//! takes no resource ID and no range of Metal's space, so that no address or ID the guest reads
//! moves for it. It is made again only for a new glyph, and the pipeline once per drawable format;
//! a recovered device makes both again (`sink`'s module doc, "A recovered device").
//!
//! **The window's overlay** ([`DrawnOverlay`]: on Linux the question a close asks) is drawn by a
//! pass of the same kind, after the cursor's, a pixel of its image to a pixel of the drawable at
//! the overlay's place, clipped to the drawable alone ([`overlay_placement`]), with a pipeline and
//! a texture of its own ([`CursorPass::overlay`]), made again only for a new image.
//!
//! **Refusals.** What the platform refuses of them leaves the frame without the guest's cursor,
//! logged (once a kind, `core::logging`); the host's cursor stays hidden over the image while the
//! guest leads, so neither shows. The draw's values are ones the platforms take: a scissor inside
//! the drawable, and a viewport inside the least bounds Vulkan sets every device (the runtime hands
//! a viewport to Vulkan as it is), [`VIEWPORT_BOUNDS`] and [`VIEWPORT_DIMENSION`]; a glyph that
//! would reach past them is not drawn (only near the far edge of a window over 8000 pixels
//! across). Nothing here reaches the guest or the C API: the cursor is in the window's pixels
//! alone.

use std::sync::Arc;

use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
use reims_vgpu_vulkan::core::platform::resource::Texture as _;
use reims_vgpu_vulkan::core::platform::types::{
    MtlOrigin, MtlRegion, MtlScissorRect, MtlSize, MtlViewport,
};
use reims_vgpu_vulkan::core::platform::values::{
    ColorAttachmentDesc, FullRenderPipelineDesc, TextureDesc,
};
use reims_vgpu_vulkan::core::platform::{Handle as _, HostRange, Platform};
use reims_vgpu_window::{DrawnCursor, DrawnOverlay, Glyph, PixelRect};

use super::Converter;
use crate::entry::no_catch;
use crate::platform::ApiPlatform;

/// The least `viewportBoundsRange` Vulkan sets every device: a viewport's edges lie inside it.
const VIEWPORT_BOUNDS: (f64, f64) = (-8192.0, 8191.0);
/// The least `maxViewportDimensions` Vulkan sets every device: a viewport's width and height.
const VIEWPORT_DIMENSION: f64 = 4096.0;

/// `MTLPixelFormatRGBA8Unorm`, the glyph's texture's format: the guest's bytes as they come.
const GLYPH_FORMAT: u64 = 70;
/// `MTLTextureUsageShaderRead`.
const GLYPH_USAGE: u64 = 1;
/// The cursor's pipelines' label.
pub(super) const CURSOR_PIPELINE_LABEL: &str = "reims-vgpu.display-cursor";
/// The overlay's pipelines' label.
pub(super) const OVERLAY_PIPELINE_LABEL: &str = "reims-vgpu.display-overlay";

/// What a sink keeps to draw the guest's cursor, or the window's overlay, from one frame to the
/// next.
pub(super) struct CursorPass<P: Platform> {
    /// Its pipelines' label.
    label: &'static str,
    /// What it draws, for its refusals' lines.
    draws: &'static str,
    /// The blending pipelines made so far, by the drawable's pixel format.
    pipelines: Vec<(u64, P::RenderPipeline)>,
    /// The glyph last drawn, and its texture.
    glyph: Option<(Arc<Glyph>, P::Texture)>,
}

impl<P: Platform> CursorPass<P> {
    /// The guest's cursor's pass.
    pub(super) fn cursor() -> Self {
        CursorPass::new(CURSOR_PIPELINE_LABEL, "guest cursor")
    }

    /// The window's overlay's pass.
    pub(super) fn overlay() -> Self {
        CursorPass::new(OVERLAY_PIPELINE_LABEL, "overlay")
    }

    /// A pass whose pipelines are labelled `label`, drawing what `draws` names.
    fn new(label: &'static str, draws: &'static str) -> Self {
        CursorPass {
            label,
            draws,
            pipelines: Vec::new(),
            glyph: None,
        }
    }
}

/// What a frame's cursor draw binds: the pipeline for the drawable's format, the glyph's texture,
/// and where it draws ([`placement`]).
pub(super) struct Prepared<P: Platform> {
    /// The blending pipeline.
    pub(super) pipeline: P::RenderPipeline,
    /// The glyph's texture.
    pub(super) texture: P::Texture,
    /// The glyph's rectangle in the drawable.
    pub(super) viewport: MtlViewport,
    /// Its clip to the image and the drawable.
    pub(super) scissor: MtlScissorRect,
}

impl<P: ApiPlatform> CursorPass<P> {
    /// What drawing `glyph` over the frame drawn into `target` binds, at `placed` (its viewport
    /// and scissor: [`placement`], [`overlay_placement`]), with `converter`'s functions on
    /// `device` of `platform`: the pipeline for the target's format, made the first time, and the
    /// glyph's texture, made again only for a new glyph. `None` where the platform refuses the
    /// pipeline or the texture (logged).
    pub(super) fn prepare(
        &mut self,
        platform: &Arc<P>,
        device: &P::Device,
        converter: &Converter<P>,
        target: &P::Texture,
        placed: (MtlViewport, MtlScissorRect),
        glyph: &Arc<Glyph>,
    ) -> Option<Prepared<P>> {
        let (viewport, scissor) = placed;
        let format = target.pixel_format();
        let pipeline = match self.pipelines.iter().find(|(f, _)| *f == format) {
            Some((_, pipeline)) => pipeline.clone_ref(),
            None => {
                let made = device
                    .new_render_pipeline_state_full(&FullRenderPipelineDesc {
                        label: Some(self.label),
                        colors: vec![(0, over(format))],
                        ..FullRenderPipelineDesc::new(
                            &converter.pipelines.vertex_function,
                            Some(&converter.pipelines.fragment_function),
                        )
                    })
                    .ok()
                    .flatten();
                let Some(pipeline) = made else {
                    return refused(self.draws, "a pipeline for the drawable's format");
                };
                self.pipelines.push((format, pipeline.clone_ref()));
                pipeline
            }
        };
        let texture = match &self.glyph {
            Some((drawn, texture)) if Arc::ptr_eq(drawn, glyph) => texture.clone_ref(),
            _ => {
                let Some(texture) = glyph_texture(platform, device, glyph) else {
                    return refused(self.draws, "a texture for its image");
                };
                self.glyph = Some((Arc::clone(glyph), texture.clone_ref()));
                texture
            }
        };
        Some(Prepared {
            pipeline,
            texture,
            viewport,
            scissor,
        })
    }
}

/// Logs that the platform refused `what` for the pass that draws `draws`: the frame shows none.
fn refused<T>(draws: &str, what: &str) -> Option<T> {
    let kind = format!("{draws}: {what}");
    // Once per kind; later ones at debug, counted (`core::logging`).
    log::warn!(
        target: "reims_vgpu::graphics", kind:% = kind;
        "the window draws no {draws}: the platform refused {what}"
    );
    None
}

/// A colour attachment of `pixel_format` that blends each fragment, its colour premultiplied,
/// over the target by its alpha: both source factors `one` (1), both destination factors
/// `oneMinusSourceAlpha` (5).
fn over(pixel_format: u64) -> ColorAttachmentDesc {
    ColorAttachmentDesc {
        pixel_format,
        blending_enabled: true,
        source_rgb_blend_factor: 1,
        destination_rgb_blend_factor: 5,
        source_alpha_blend_factor: 1,
        destination_alpha_blend_factor: 5,
        ..ColorAttachmentDesc::default()
    }
}

/// Where `cursor` draws over `region` of a target of `size` pixels: the viewport, the glyph's
/// rectangle in the target, its hot spot at the cursor's position in the mode's pixels, a glyph
/// pixel to a guest pixel at the image's scale; and the scissor, the target's pixels that
/// rectangle touches inside the image and the target. The viewport may reach past the target,
/// which only the scissor clips. `None` where nothing would show: an empty mode, glyph or
/// region, or a glyph wholly off the image; and where the viewport would reach past
/// [`VIEWPORT_BOUNDS`] or be larger than [`VIEWPORT_DIMENSION`] (the module doc).
pub(super) fn placement(
    region: PixelRect,
    size: (u64, u64),
    cursor: &DrawnCursor,
) -> Option<(MtlViewport, MtlScissorRect)> {
    let (mode, glyph) = (cursor.mode, &cursor.glyph);
    if mode.0 == 0 || mode.1 == 0 || glyph.width == 0 || glyph.height == 0 {
        return None;
    }
    #[expect(clippy::as_conversions, reason = "u64 to f64 rounds to nearest")]
    let [x, y, width, height, target_width, target_height] = [
        region.x,
        region.y,
        region.width,
        region.height,
        size.0,
        size.1,
    ]
    .map(|v| v as f64);
    let (per_x, per_y) = (width / f64::from(mode.0), height / f64::from(mode.1));
    let left = x + (f64::from(cursor.at.0) - f64::from(glyph.hot_spot.0)) * per_x;
    let top = y + (f64::from(cursor.at.1) - f64::from(glyph.hot_spot.1)) * per_y;
    let (glyph_width, glyph_height) = (
        f64::from(glyph.width) * per_x,
        f64::from(glyph.height) * per_y,
    );
    let (x0, y0) = (left.floor().max(x), top.floor().max(y));
    let x1 = (left + glyph_width).ceil().min(x + width).min(target_width);
    let y1 = (top + glyph_height)
        .ceil()
        .min(y + height)
        .min(target_height);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    let (low, high) = VIEWPORT_BOUNDS;
    let within = |start: f64, length: f64| {
        start >= low && start + length <= high && length <= VIEWPORT_DIMENSION
    };
    if !(within(left, glyph_width) && within(top, glyph_height)) {
        return None;
    }
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "whole pixels inside the target, which u64 holds"
    )]
    let [sx, sy, sw, sh] = [x0, y0, x1 - x0, y1 - y0].map(|v| v as u64);
    let viewport = MtlViewport {
        origin_x: left,
        origin_y: top,
        width: glyph_width,
        height: glyph_height,
        znear: 0.0,
        zfar: 1.0,
    };
    let scissor = MtlScissorRect {
        x: sx,
        y: sy,
        width: sw,
        height: sh,
    };
    Some((viewport, scissor))
}

/// Where `overlay` draws over a target of `size` pixels: the viewport, its image's rectangle at
/// its place, a pixel of it to a pixel of the target; and the scissor, the target's pixels that
/// rectangle covers. `None` where nothing would show: an empty image, or one wholly off the
/// target; and where the viewport would reach past [`VIEWPORT_BOUNDS`] or be larger than
/// [`VIEWPORT_DIMENSION`] (the module doc).
pub(super) fn overlay_placement(
    size: (u64, u64),
    overlay: &DrawnOverlay,
) -> Option<(MtlViewport, MtlScissorRect)> {
    let (width, height) = (
        u64::from(overlay.image.width),
        u64::from(overlay.image.height),
    );
    let (x, y) = overlay.at;
    let (x1, y1) = (
        x.saturating_add(width).min(size.0),
        y.saturating_add(height).min(size.1),
    );
    if x1 <= x || y1 <= y {
        return None;
    }
    #[expect(clippy::as_conversions, reason = "u64 to f64 rounds to nearest")]
    let [left, top, width, height] = [x, y, width, height].map(|v| v as f64);
    let (_, high) = VIEWPORT_BOUNDS;
    let within = |start: f64, length: f64| start + length <= high && length <= VIEWPORT_DIMENSION;
    if !(within(left, width) && within(top, height)) {
        return None;
    }
    let viewport = MtlViewport {
        origin_x: left,
        origin_y: top,
        width,
        height,
        znear: 0.0,
        zfar: 1.0,
    };
    let scissor = MtlScissorRect {
        x,
        y,
        width: x1 - x,
        height: y1 - y,
    };
    Some((viewport, scissor))
}

/// `glyph`'s pixels in a texture of the platform's own (the module doc), RGBA8, premultiplied,
/// read by shaders, a row its bytes do not reach transparent; `None` for a glyph with no pixels,
/// or where the platform makes no texture.
fn glyph_texture<P: ApiPlatform>(
    platform: &Arc<P>,
    device: &P::Device,
    glyph: &Glyph,
) -> Option<P::Texture> {
    let (width, height) = (u64::from(glyph.width), u64::from(glyph.height));
    let row = width.checked_mul(4)?;
    let length = usize::try_from(row.checked_mul(height)?).ok()?;
    if length == 0 {
        return None;
    }
    let mut bytes = glyph.rgba.get(..length).unwrap_or(&glyph.rgba).to_vec();
    bytes.resize(length, 0);
    premultiply(&mut bytes);
    let desc = TextureDesc {
        pixel_format: GLYPH_FORMAT,
        width,
        height,
        usage: GLYPH_USAGE,
        ..TextureDesc::default()
    };
    let texture = platform.own_texture(device, &desc)?;
    no_catch(texture.replace_region(
        MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(width, height, 1)),
        0,
        0,
        HostRange::of_slice(&bytes),
        row,
        row * height,
    ));
    Some(texture)
}

/// Premultiplies RGBA `pixels` by their alpha, each channel rounded to nearest.
fn premultiply(pixels: &mut [u8]) {
    for [red, green, blue, alpha] in pixels.as_chunks_mut::<4>().0 {
        let alpha = u16::from(*alpha);
        for channel in [red, green, blue] {
            let scaled = (u16::from(*channel) * alpha + 127) / 255;
            *channel = u8::try_from(scaled).unwrap_or(u8::MAX);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cursor of a `width` by `height` glyph with its hot spot at `hot_spot`, at `at` in `mode`.
    fn cursor(
        (width, height): (u32, u32),
        hot_spot: (u16, u16),
        mode: (u16, u16),
        at: (u16, u16),
    ) -> DrawnCursor {
        DrawnCursor {
            glyph: Arc::new(Glyph {
                width,
                height,
                hot_spot,
                rgba: vec![0xff; usize::try_from(width * height * 4).unwrap()],
            }),
            mode,
            at,
        }
    }

    /// A region of the target.
    fn rect(x: u64, y: u64, width: u64, height: u64) -> PixelRect {
        PixelRect {
            x,
            y,
            width,
            height,
        }
    }

    /// The viewport puts the glyph's hot spot at the guest's cursor position, a glyph pixel to a
    /// guest pixel at the image's scale, in a letterboxed image too, and the scissor is the
    /// glyph's rectangle where it lies inside the image
    #[test]
    fn the_viewport_puts_the_hot_spot_at_the_cursor() {
        let c = cursor((10, 5), (10, 0), (100, 50), (60, 20));
        let (v, s) = placement(rect(0, 0, 100, 50), (100, 50), &c).unwrap();
        assert_eq!(
            (v.origin_x, v.origin_y, v.width, v.height),
            (50.0, 20.0, 10.0, 5.0)
        );
        assert_eq!((s.x, s.y, s.width, s.height), (50, 20, 10, 5));
        // Twice the size, 30 pixels from the left and 10 from the top.
        let (v, s) = placement(rect(30, 10, 200, 100), (260, 120), &c).unwrap();
        assert_eq!(
            (v.origin_x, v.origin_y, v.width, v.height),
            (130.0, 50.0, 20.0, 10.0)
        );
        assert_eq!((s.x, s.y, s.width, s.height), (130, 50, 20, 10));
        // At a scale of 1.5 the rectangle's edges fall inside pixels, which the scissor takes.
        let (v, s) = placement(rect(0, 0, 150, 75), (150, 75), &c).unwrap();
        assert_eq!((v.origin_x, v.width), (75.0, 15.0));
        assert_eq!((v.origin_y, v.height), (30.0, 7.5));
        assert_eq!((s.x, s.y, s.width, s.height), (75, 30, 15, 8));
    }

    /// A glyph partly off the image keeps its whole viewport, past the target where it reaches
    /// there, and the scissor clips it to the image; one wholly off it, an empty glyph and an
    /// empty mode draw nothing
    #[test]
    fn the_scissor_clips_the_glyph_to_the_image() {
        let region = rect(20, 10, 100, 100);
        let size = (140, 120);
        let g = (16, 16);
        let (v, s) = placement(region, size, &cursor(g, (4, 4), (100, 100), (0, 0))).unwrap();
        assert_eq!((v.origin_x, v.origin_y, v.width), (16.0, 6.0, 16.0));
        assert_eq!((s.x, s.y, s.width, s.height), (20, 10, 12, 12));
        let (v, s) = placement(region, size, &cursor(g, (0, 0), (100, 100), (95, 95))).unwrap();
        assert_eq!((v.origin_x, v.origin_y), (115.0, 105.0));
        assert_eq!((s.x, s.y, s.width, s.height), (115, 105, 5, 5));
        let at_edge = cursor(g, (8, 8), (100, 100), (0, 0));
        let (v, s) = placement(rect(0, 0, 100, 100), (100, 100), &at_edge).unwrap();
        assert_eq!((v.origin_x, v.origin_y), (-8.0, -8.0));
        assert_eq!((s.x, s.y, s.width, s.height), (0, 0, 8, 8));
        let none = |c: &DrawnCursor| placement(region, size, c).is_none();
        assert!(none(&cursor(g, (0, 0), (100, 100), (100, 50))));
        assert!(none(&cursor(g, (0, 0), (100, 100), (50, 100))));
        assert!(none(&cursor(g, (0, 0), (100, 100), (u16::MAX, u16::MAX))));
        assert!(none(&cursor(g, (20, 20), (100, 100), (0, 0))));
        assert!(none(&cursor((0, 16), (0, 0), (100, 100), (50, 50))));
        assert!(none(&cursor(g, (0, 0), (0, 100), (50, 50))));
        assert!(
            placement(
                rect(20, 10, 0, 100),
                size,
                &cursor(g, (0, 0), (100, 100), (5, 5))
            )
            .is_none()
        );
    }

    /// A glyph whose viewport would reach past Vulkan's least viewport bounds, or be larger than
    /// its least viewport size, is not drawn, though it shows in the image
    #[test]
    fn a_viewport_past_vulkan_s_least_bounds_draws_nothing() {
        let wide = rect(0, 0, 8192, 100);
        let size = (8192, 100);
        let g = (16, 16);
        assert!(placement(wide, size, &cursor(g, (0, 0), (8192, 100), (8170, 10))).is_some());
        assert!(placement(wide, size, &cursor(g, (0, 0), (8192, 100), (8180, 10))).is_none());
        // A glyph of 256 pixels at 17 pixels to a guest pixel is 4352 wide.
        let scaled = rect(0, 0, 1700, 1700);
        let big = cursor((256, 1), (0, 0), (100, 100), (0, 0));
        assert!(placement(scaled, (1700, 1700), &big).is_none());
        let fits = cursor((240, 1), (0, 0), (100, 100), (0, 0));
        assert!(placement(scaled, (1700, 1700), &fits).is_some());
    }

    /// An overlay is drawn a pixel of its image to a pixel of the target at its place, clipped to
    /// the target alone; one wholly off the target, an empty one, and one past Vulkan's least
    /// viewport bounds draw nothing
    #[test]
    fn an_overlay_is_drawn_pixel_for_pixel_at_its_place() {
        let overlay = |(width, height): (u32, u32), at: (u64, u64)| DrawnOverlay {
            image: Arc::new(Glyph {
                width,
                height,
                hot_spot: (0, 0),
                rgba: Vec::new(),
            }),
            at,
        };
        let (v, s) = overlay_placement((1920, 1080), &overlay((624, 248), (648, 416))).unwrap();
        assert_eq!(
            (v.origin_x, v.origin_y, v.width, v.height),
            (648.0, 416.0, 624.0, 248.0)
        );
        assert_eq!((s.x, s.y, s.width, s.height), (648, 416, 624, 248));
        let (v, s) = overlay_placement((320, 200), &overlay((312, 248), (4, 0))).unwrap();
        assert_eq!((v.width, v.height), (312.0, 248.0));
        assert_eq!((s.x, s.y, s.width, s.height), (4, 0, 312, 200));
        let none = |o: &DrawnOverlay| overlay_placement((320, 200), o).is_none();
        assert!(none(&overlay((10, 10), (320, 0))));
        assert!(none(&overlay((10, 10), (0, 200))));
        assert!(none(&overlay((0, 10), (0, 0))));
        assert!(none(&overlay((4097, 10), (0, 0))));
        assert!(overlay_placement((9000, 100), &overlay((10, 10), (8190, 0))).is_none());
    }

    /// Premultiplying scales each colour channel by the alpha, to nearest, and keeps the alpha
    #[test]
    fn premultiplying_scales_the_colour_by_the_alpha() {
        let mut pixels = [
            0xff, 0x80, 0x00, 0xff, 0xff, 0x80, 0x10, 0x80, 0xff, 0xff, 0xff, 0x00,
        ];
        premultiply(&mut pixels);
        assert_eq!(
            pixels,
            [
                0xff, 0x80, 0x00, 0xff, 0x80, 0x40, 0x08, 0x80, 0x00, 0x00, 0x00, 0x00
            ]
        );
    }
}
