//! Where the guest's image sits in a window, and where the pointer is on it.
//!
//! Every value here is in the view's own space, y down: the origin is the view's top left, as
//! in a flipped `NSView`. Points are `f64`; the drawable's pixels are whole numbers
//! ([`PixelRect`]). The one exception is [`resized_frame`], which places a window on its screen,
//! in the screen's points, y down too.
//!
//! The image is the guest mode aspect-fit into the view, less its insets (a screen's camera
//! housing), centred between black bars. It is fitted in the drawable's pixels, so its edges
//! fall on whole pixels, and [`Layout::image`] is that pixel rectangle again in points. The
//! pointer is mapped through the same rectangle, so the guest pixel under the host's pointer is
//! the one drawn there.
//!
//! **The guest's modes are a multiple of 16 pixels wide** ([`MODE_WIDTH_STEP`], [`offered_mode`]).
//! The guest's desktop surfaces are its mode's width in BGRA8 pixels, 4 bytes each, and a
//! surface over the guest's RAM is drawn in place only where each of its rows starts and ends on
//! a line of the GPU's memory: on Intel's Vulkan driver (ANV) a render pass writes back whole
//! 64-byte lines, so the bytes past a row's last pixel on its last line would be written too.
//! 16 pixels are 64 bytes. Measured by metal2vulkan's lane on homelab-ms01's Iris Xe
//! (2026-11-07): the guest's 3662-pixel desktop surface, 14648 bytes a row (ending 56 bytes
//! into a line), stayed a copy of the guest's pages, 2.54 ms of the GPU's time a plain frame and
//! 3.94 ms a frame that reads its own pixels. Padding a row's bytes alone does not help: the
//! row's last line still holds bytes past its last pixel.

use crate::config::{BUTTON_EXTRA, BUTTON_LEFT, BUTTON_MIDDLE, BUTTON_RIGHT, BUTTON_SIDE};
use crate::handle::PixelRect;

/// A rectangle in points, y down.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rect {
    /// Points from the left.
    pub x: f64,
    /// Points from the top.
    pub y: f64,
    /// Points wide.
    pub width: f64,
    /// Points high.
    pub height: f64,
}

/// Room kept clear at each edge of a view, in points: the image is fitted inside it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Insets {
    /// From the top (a camera housing's height).
    pub top: f64,
    /// From the left.
    pub left: f64,
    /// From the bottom.
    pub bottom: f64,
    /// From the right.
    pub right: f64,
}

/// Where a view's drawable and image are, for one size, scale and guest mode.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Layout {
    /// The drawable's size in pixels, width then height: the view's bounds times its backing
    /// scale, rounded, at least 1×1.
    pub drawable: (u64, u64),
    /// The image, in the view's points.
    pub image: Rect,
    /// The image, in the drawable's pixels: inside the drawable, and inside the insets unless
    /// they leave no room.
    pub pixels: PixelRect,
}

/// `value` where it is finite and not below 0, else 0.
fn non_negative(value: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        0.0
    }
}

/// `scale` where it is finite and above 0, else 1.
fn valid_scale(scale: f64) -> f64 {
    if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    }
}

/// Points to whole pixels: `points × scale`, rounded, never below 0.
fn to_pixels(points: f64, scale: f64) -> u64 {
    (non_negative(points) * scale).round() as u64
}

/// Whole pixels as a float, for the fit's arithmetic.
fn as_f64(pixels: u64) -> f64 {
    pixels as f64
}

/// The view's layout: its bounds in points `(width, height)`, its backing scale (pixels per
/// point), the room kept clear at its edges, and the guest's mode `(width, height)` once the
/// guest has set one.
///
/// The drawable is the bounds in pixels. The image is the guest mode aspect-fit into the
/// drawable less the insets (in points, so scaled with the bounds), on whole pixels, centred;
/// where the insets leave no room it is fitted into the whole drawable. With no guest mode yet
/// (or an empty one) the image is that whole area.
pub fn layout(bounds: (f64, f64), scale: f64, insets: Insets, guest: Option<(u16, u16)>) -> Layout {
    let scale = valid_scale(scale);
    let drawable = (
        to_pixels(bounds.0, scale).max(1),
        to_pixels(bounds.1, scale).max(1),
    );
    let (left, top) = (to_pixels(insets.left, scale), to_pixels(insets.top, scale));
    let (right, bottom) = (
        to_pixels(insets.right, scale),
        to_pixels(insets.bottom, scale),
    );
    let inset_width = drawable.0.saturating_sub(left).saturating_sub(right);
    let inset_height = drawable.1.saturating_sub(top).saturating_sub(bottom);
    let area = if inset_width == 0 || inset_height == 0 {
        PixelRect {
            x: 0,
            y: 0,
            width: drawable.0,
            height: drawable.1,
        }
    } else {
        PixelRect {
            x: left,
            y: top,
            width: inset_width,
            height: inset_height,
        }
    };
    let pixels = match guest {
        Some((w, h)) if w > 0 && h > 0 => fit(area, (w, h)),
        _ => area,
    };
    Layout {
        drawable,
        image: Rect {
            x: as_f64(pixels.x) / scale,
            y: as_f64(pixels.y) / scale,
            width: as_f64(pixels.width) / scale,
            height: as_f64(pixels.height) / scale,
        },
        pixels,
    }
}

/// The pixels the width of every mode offered the guest is a multiple of: 16 BGRA8 pixels are
/// a 64-byte line of the GPU's memory, so each row of a guest surface of the mode's width starts
/// and ends on a line (the module doc).
pub const MODE_WIDTH_STEP: u64 = 16;

/// The pixels the height of every mode offered the guest is a multiple of: the guest kext
/// offers a mode HiDPI only where its pixels halve exactly (guest/kext's `RgfxMode::hiDpi`).
pub const MODE_HEIGHT_STEP: u64 = 2;

/// The guest mode a window that keeps its size offers the guest, for a drawable of `drawable`
/// pixels: its width rounded down to a multiple of [`MODE_WIDTH_STEP`] and its height to a
/// multiple of [`MODE_HEIGHT_STEP`] (a side below its step is kept). Both halve exactly, as a
/// HiDPI mode's pixels must. A window's pixels are often neither: a resize of the person's, or a
/// borderless window at a fractional scale, as on a 3840 × 2160 screen at 2.1, the screen's
/// 1828.57 × 1028.57 points rounded to 1829 × 1029, which SDL's Wayland backend makes 3841 × 2161
/// pixels (points times the scale, rounded), offered 3840 × 2160, the screen's own pixels. The
/// layout draws that mode a guest pixel to a pixel at the drawable's top left ([`fit`]).
pub fn offered_mode(drawable: (u64, u64)) -> (u64, u64) {
    let down = |pixels: u64, step: u64| {
        if pixels >= step {
            pixels - pixels % step
        } else {
            pixels
        }
    };
    (
        down(drawable.0, MODE_WIDTH_STEP),
        down(drawable.1, MODE_HEIGHT_STEP),
    )
}

/// The largest rectangle of `guest`'s ratio in `area`, on whole pixels, centred (to within half
/// a pixel). One side fills the area; the other is the ratio's, rounded, at least 1. A guest
/// mode of the area's pixels, or fewer by less than a step each way ([`MODE_WIDTH_STEP`],
/// [`MODE_HEIGHT_STEP`]), is drawn a guest pixel to a pixel, at the area's top left: that is the
/// mode a window offers for its pixels ([`offered_mode`]), which shows whole and unscaled,
/// beside up to 15 columns and a row of black.
fn fit(area: PixelRect, guest: (u16, u16)) -> PixelRect {
    let (gw, gh) = (u64::from(guest.0), u64::from(guest.1));
    if gw <= area.width
        && gh <= area.height
        && area.width - gw < MODE_WIDTH_STEP
        && area.height - gh < MODE_HEIGHT_STEP
    {
        return PixelRect {
            x: area.x,
            y: area.y,
            width: gw,
            height: gh,
        };
    }
    let (aw, ah) = (as_f64(area.width), as_f64(area.height));
    let (gw, gh) = (f64::from(guest.0), f64::from(guest.1));
    let (width, height) = if aw * gh <= ah * gw {
        (
            area.width,
            ((aw * gh / gw).round() as u64).clamp(1, area.height),
        )
    } else {
        (
            ((ah * gw / gh).round() as u64).clamp(1, area.width),
            area.height,
        )
    };
    PixelRect {
        x: area.x + (area.width - width) / 2,
        y: area.y + (area.height - height) / 2,
        width,
        height,
    }
}

/// The guest pixel under `point`, for an image at `image` showing a guest mode `guest` (width,
/// height): `x` in `0..width`, `y` in `0..height`. A point outside the image holds at its
/// nearest edge, so a drag that leaves the image keeps the guest's pointer at that edge.
/// `(0, 0)` for an empty image or mode.
pub fn pointer_to_guest(point: (f64, f64), image: Rect, guest: (u16, u16)) -> (u16, u16) {
    let axis = |p: f64, origin: f64, extent: f64, size: u16| -> u16 {
        if size == 0 || extent.is_nan() || extent <= 0.0 {
            return 0;
        }
        let at = ((p - origin) / extent * f64::from(size)).floor();
        at.clamp(0.0, f64::from(size - 1)) as u16
    };
    (
        axis(point.0, image.x, image.width, guest.0),
        axis(point.1, image.y, image.height, guest.1),
    )
}

/// Whether `point` is on the image: its left and top edges are in, its right and bottom out.
pub fn contains(image: Rect, point: (f64, f64)) -> bool {
    point.0 >= image.x
        && point.0 < image.x + image.width
        && point.1 >= image.y
        && point.1 < image.y + image.height
}

/// The `BUTTON_*` bit of the host's mouse button `ns_button_number` (`NSEvent.buttonNumber`):
/// 0 left, 1 right, 2 middle, 3 and 4 the side buttons. 0 for a button past those: the guest's
/// tablet has five.
pub fn button_bit(ns_button_number: i64) -> u32 {
    match ns_button_number {
        0 => BUTTON_LEFT,
        1 => BUTTON_RIGHT,
        2 => BUTTON_MIDDLE,
        3 => BUTTON_SIDE,
        4 => BUTTON_EXTRA,
        _ => 0,
    }
}

/// A new window's content size in points, for a guest mode `guest` (width, height), with
/// `visible` the room there is for the content (the screen's visible area less the window's
/// title bar): one guest pixel per host point, shrunk with its ratio kept where it does not fit.
/// So on a Retina screen the window's drawable is twice the mode each way. At least one point
/// each way; `visible` with no room is ignored.
pub fn initial_content_size(guest: (u16, u16), visible: (f64, f64)) -> (f64, f64) {
    let (width, height) = (f64::from(guest.0), f64::from(guest.1));
    let (room_w, room_h) = (non_negative(visible.0), non_negative(visible.1));
    let shrink = if room_w > 0.0 && room_h > 0.0 && width > 0.0 && height > 0.0 {
        (room_w / width).min(room_h / height).min(1.0)
    } else {
        1.0
    };
    ((width * shrink).max(1.0), (height * shrink).max(1.0))
}

/// A window's frame `frame` resized to `size` (width, height), as a window following the guest's
/// mode is: its top left corner kept, then moved as little as it takes to be inside `visible`,
/// its screen's visible area. Where it is wider or taller than `visible`, its left or top edge is
/// `visible`'s, so its title bar stays on the screen. Both rectangles are in the screen's points,
/// y down.
pub fn resized_frame(frame: Rect, size: (f64, f64), visible: Rect) -> Rect {
    let (width, height) = size;
    // The far edge pulled back inside first, then the near edge, which wins.
    let inside =
        |at: f64, extent: f64, start: f64, room: f64| at.min(start + room - extent).max(start);
    Rect {
        x: inside(frame.x, width, visible.x, visible.width),
        y: inside(frame.y, height, visible.y, visible.height),
        width,
        height,
    }
}

/// Points of the view per guest pixel, for an image at `image` showing a guest mode `guest`:
/// the size the guest's cursor is drawn at. 1 for an empty image or mode.
pub fn cursor_points_per_pixel(image: Rect, guest: (u16, u16)) -> f64 {
    let per_pixel = image.width / f64::from(guest.0);
    if per_pixel.is_finite() && per_pixel > 0.0 {
        per_pixel
    } else {
        1.0
    }
}

/// The host cursor's image size and hot spot, in points.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CursorSize {
    /// Points wide, then high.
    pub size: (f64, f64),
    /// The hot spot, in points from the image's top left.
    pub hot_spot: (f64, f64),
}

/// The host cursor for a guest glyph `glyph` pixels (width, height) with its hot spot at
/// `hot_spot` pixels from its top left, drawn at `points_per_pixel`
/// ([`cursor_points_per_pixel`]): the guest's cursor at the size the image shows it. The hot
/// spot is kept on the glyph (a hot spot past its edge holds at the last pixel).
pub fn cursor_size(glyph: (u32, u32), hot_spot: (u16, u16), points_per_pixel: f64) -> CursorSize {
    let scale = valid_scale(points_per_pixel);
    let on_glyph = |spot: u16, extent: u32| -> f64 {
        f64::from(u32::from(spot).min(extent.saturating_sub(1)))
    };
    CursorSize {
        size: (f64::from(glyph.0) * scale, f64::from(glyph.1) * scale),
        hot_spot: (
            on_glyph(hot_spot.0, glyph.0) * scale,
            on_glyph(hot_spot.1, glyph.1) * scale,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NO_INSETS: Insets = Insets {
        top: 0.0,
        left: 0.0,
        bottom: 0.0,
        right: 0.0,
    };

    /// The ratios, and windows wide, tall and of odd sizes, at 1× and 2×.
    const GUESTS: [(u16, u16); 5] = [
        (1920, 1200),
        (2560, 1600),
        (1920, 1080),
        (1024, 768),
        (1366, 768),
    ];
    const WINDOWS: [(f64, f64); 6] = [
        (1728.0, 1080.0),
        (2000.0, 600.0),
        (500.0, 1400.0),
        (1001.0, 777.0),
        (333.5, 999.25),
        (1.0, 1.0),
    ];

    /// The image's side that does not fill the area is the guest ratio's, to within half a
    /// pixel
    fn assert_ratio(pixels: PixelRect, area: (u64, u64), guest: (u16, u16)) {
        let (gw, gh) = (f64::from(guest.0), f64::from(guest.1));
        let (w, h) = (pixels.width as f64, pixels.height as f64);
        if pixels.width == area.0 {
            assert!(
                (h - (w * gh / gw).max(1.0)).abs() <= 0.5 || pixels.height == area.1,
                "{pixels:?} in {area:?} for {guest:?}"
            );
        } else {
            assert_eq!(
                pixels.height, area.1,
                "{pixels:?} in {area:?} for {guest:?}"
            );
            assert!(
                (w - (h * gw / gh).max(1.0)).abs() <= 0.5,
                "{pixels:?} in {area:?} for {guest:?}"
            );
        }
    }

    /// 16:10, 16:9 and 4:3 guests keep their ratio in wide, tall and odd windows
    #[test]
    fn keeps_the_guest_ratio() {
        for scale in [1.0, 2.0, 3.0] {
            for window in WINDOWS {
                for guest in GUESTS {
                    let l = layout(window, scale, NO_INSETS, Some(guest));
                    assert_ratio(l.pixels, l.drawable, guest);
                }
            }
        }
        // A 16:10 guest in a wide window: full height, bars left and right.
        let l = layout((2000.0, 600.0), 1.0, NO_INSETS, Some((1920, 1200)));
        assert_eq!(
            l.pixels,
            PixelRect {
                x: 520,
                y: 0,
                width: 960,
                height: 600
            }
        );
        // A 4:3 guest in a tall window: full width, bars above and below.
        let l = layout((500.0, 1400.0), 2.0, NO_INSETS, Some((1024, 768)));
        assert_eq!(
            l.pixels,
            PixelRect {
                x: 0,
                y: 1025,
                width: 1000,
                height: 750
            }
        );
    }

    /// The image is centred between the bars, to within half a pixel
    #[test]
    fn centres_the_image() {
        for scale in [1.0, 2.0] {
            for window in WINDOWS {
                for guest in GUESTS {
                    let l = layout(window, scale, NO_INSETS, Some(guest));
                    let p = l.pixels;
                    let (left, right) = (p.x, l.drawable.0 - p.x - p.width);
                    let (top, bottom) = (p.y, l.drawable.1 - p.y - p.height);
                    assert!(right - left <= 1, "{l:?}");
                    assert!(bottom - top <= 1, "{l:?}");
                }
            }
        }
    }

    /// The image's edges fall on whole pixels, inside the drawable, and `image` is those pixels
    /// in points
    #[test]
    fn the_image_is_on_whole_pixels() {
        for scale in [1.0, 1.5, 2.0] {
            for window in WINDOWS {
                for guest in GUESTS {
                    let l = layout(window, scale, NO_INSETS, Some(guest));
                    let p = l.pixels;
                    assert!(p.width >= 1 && p.height >= 1);
                    assert!(p.x + p.width <= l.drawable.0 && p.y + p.height <= l.drawable.1);
                    let near = |a: f64, b: u64| (a * scale - b as f64).abs() < 1e-9;
                    assert!(near(l.image.x, p.x) && near(l.image.y, p.y), "{l:?}");
                    assert!(near(l.image.width, p.width), "{l:?}");
                    assert!(near(l.image.height, p.height), "{l:?}");
                }
            }
        }
    }

    /// The drawable is the bounds in pixels, rounded, and never empty
    #[test]
    fn the_drawable_is_the_bounds_in_pixels() {
        let l = layout((1728.0, 1117.0), 2.0, NO_INSETS, None);
        assert_eq!(l.drawable, (3456, 2234));
        let l = layout((100.3, 50.7), 1.0, NO_INSETS, None);
        assert_eq!(l.drawable, (100, 51));
        let l = layout((0.0, -5.0), 2.0, NO_INSETS, None);
        assert_eq!(l.drawable, (1, 1));
        let l = layout((f64::NAN, f64::INFINITY), f64::NAN, NO_INSETS, None);
        assert_eq!(l.drawable, (1, 1));
    }

    /// The mode offered for a drawable is its width rounded down to a multiple of 16 pixels and
    /// its height to even, and a guest at that mode is drawn a guest pixel to a pixel, at the top
    /// left, beside up to 15 columns and a row of black
    #[test]
    fn a_drawable_offers_a_mode_a_multiple_of_16_wide_drawn_unscaled() {
        assert_eq!(offered_mode((3841, 2161)), (3840, 2160));
        assert_eq!(offered_mode((3662, 2039)), (3648, 2038));
        assert_eq!(offered_mode((3663, 2040)), (3648, 2040));
        assert_eq!(offered_mode((3840, 2160)), (3840, 2160));
        assert_eq!(offered_mode((1366, 768)), (1360, 768));
        assert_eq!(offered_mode((15, 9)), (15, 8));
        assert_eq!(offered_mode((1, 1)), (1, 1));
        // Every mode a wide drawable offers starts each BGRA8 row on a 64-byte line, and halves
        // exactly.
        for width in 16..4000 {
            let (w, h) = offered_mode((width, width / 2 + 1));
            assert_eq!((w * 4) % 64, 0, "{width}");
            assert!(width - w < MODE_WIDTH_STEP && h % 2 == 0 && (w / 2) % 8 == 0);
        }
        // Borderless on a 3840 × 2160 screen at 2.1: 1829 × 1029 points, 3841 × 2161 pixels,
        // which show the 3840 × 2160 offered whole.
        let l = layout(
            (1829.0, 1029.0),
            3841.0 / 1829.0,
            NO_INSETS,
            Some((3840, 2160)),
        );
        assert_eq!(l.drawable, (3841, 2161));
        let whole = |width, height| PixelRect {
            x: 0,
            y: 0,
            width,
            height,
        };
        assert_eq!(l.pixels, whole(3840, 2160));
        // A window of 3662 × 2039 pixels shows 3648 × 2038 whole, beside 14 columns and a row.
        let l = layout((3662.0, 2039.0), 1.0, NO_INSETS, Some((3648, 2038)));
        assert_eq!(l.pixels, whole(3648, 2038));
        // A guest 16 pixels narrower, or 2 lower, is fitted as any other: centred, filling a side.
        let l = layout((3856.0, 2160.0), 1.0, NO_INSETS, Some((3840, 2160)));
        assert_eq!(l.pixels.height, 2160);
        assert_ne!(l.pixels, whole(3840, 2160));
        let l = layout((3850.0, 2162.0), 1.0, NO_INSETS, Some((3840, 2160)));
        assert_eq!(l.pixels.height, 2162);
    }

    /// With no guest mode yet, the image is the whole area inside the insets
    #[test]
    fn with_no_mode_the_image_is_the_inset_area() {
        let insets = Insets {
            top: 32.0,
            ..NO_INSETS
        };
        let l = layout((1728.0, 1117.0), 2.0, insets, None);
        assert_eq!(
            l.pixels,
            PixelRect {
                x: 0,
                y: 64,
                width: 3456,
                height: 2170
            }
        );
        assert_eq!(
            l.image,
            Rect {
                x: 0.0,
                y: 32.0,
                width: 1728.0,
                height: 1085.0
            }
        );
        assert_eq!(
            layout((10.0, 10.0), 1.0, NO_INSETS, Some((0, 768)))
                .pixels
                .width,
            10
        );
    }

    /// On a screen with a camera housing (a notch-like top inset), the image fits below it
    #[test]
    fn the_image_fits_below_a_camera_housing() {
        // A borderless window on a 3456×2234 built-in screen, 32 points of housing.
        let insets = Insets {
            top: 32.0,
            ..NO_INSETS
        };
        let l = layout((1728.0, 1117.0), 2.0, insets, Some((3456, 2160)));
        assert_eq!(l.drawable, (3456, 2234));
        assert!(l.pixels.y >= 64, "{l:?}");
        assert!(l.image.y >= 32.0, "{l:?}");
        assert_ratio(l.pixels, (3456, 2170), (3456, 2160));
        // It fills the width and is centred in the 2170 rows below the housing.
        assert_eq!(
            l.pixels,
            PixelRect {
                x: 0,
                y: 64 + 5,
                width: 3456,
                height: 2160
            }
        );
        // Every edge's inset is kept.
        let insets = Insets {
            top: 10.0,
            left: 20.0,
            bottom: 30.0,
            right: 40.0,
        };
        let l = layout((1000.0, 1000.0), 1.0, insets, Some((1000, 1000)));
        assert_eq!(
            l.pixels,
            PixelRect {
                x: 20,
                y: 10 + 10,
                width: 940,
                height: 940
            }
        );
        // Insets that leave no room are ignored.
        let insets = Insets {
            top: 600.0,
            bottom: 600.0,
            ..NO_INSETS
        };
        let l = layout((1000.0, 1000.0), 1.0, insets, Some((1000, 1000)));
        assert_eq!(l.pixels.height, 1000);
    }

    /// A guest mode change re-fits the image; the drawable (the window's size) is unchanged
    #[test]
    fn a_mode_change_refits_the_image() {
        let before = layout((1200.0, 800.0), 2.0, NO_INSETS, Some((1920, 1200)));
        let after = layout((1200.0, 800.0), 2.0, NO_INSETS, Some((1024, 768)));
        assert_eq!(before.drawable, after.drawable);
        assert_eq!(
            before.pixels,
            PixelRect {
                x: 0,
                y: 50,
                width: 2400,
                height: 1500
            }
        );
        assert_eq!(
            after.pixels,
            PixelRect {
                x: 133,
                y: 0,
                width: 2133,
                height: 1600
            }
        );
        assert_ratio(after.pixels, after.drawable, (1024, 768));
    }

    /// The pointer maps to the guest pixel under it
    #[test]
    fn the_pointer_maps_to_the_guest_pixel_under_it() {
        let image = Rect {
            x: 100.0,
            y: 50.0,
            width: 960.0,
            height: 600.0,
        };
        let guest = (1920, 1200);
        assert_eq!(pointer_to_guest((100.0, 50.0), image, guest), (0, 0));
        assert_eq!(pointer_to_guest((580.0, 350.0), image, guest), (960, 600));
        assert_eq!(pointer_to_guest((100.4, 50.6), image, guest), (0, 1));
        assert_eq!(
            pointer_to_guest((1059.9, 649.9), image, guest),
            (1919, 1199)
        );
        assert!(contains(image, (100.0, 50.0)));
        assert!(contains(image, (1059.9, 649.9)));
        assert!(!contains(image, (1060.0, 300.0)));
        assert!(!contains(image, (99.9, 300.0)));
        assert!(!contains(image, (500.0, 650.0)));
        assert!(!contains(image, (500.0, 49.0)));
    }

    /// A point outside the image holds at its nearest edge, on all four sides
    #[test]
    fn the_pointer_holds_at_the_nearest_edge() {
        let image = Rect {
            x: 100.0,
            y: 50.0,
            width: 960.0,
            height: 600.0,
        };
        let guest = (1920, 1200);
        // Left, right, top, bottom.
        assert_eq!(pointer_to_guest((-500.0, 350.0), image, guest), (0, 600));
        assert_eq!(pointer_to_guest((5000.0, 350.0), image, guest), (1919, 600));
        assert_eq!(pointer_to_guest((580.0, -20.0), image, guest), (960, 0));
        assert_eq!(pointer_to_guest((580.0, 9000.0), image, guest), (960, 1199));
        // The corners, and what is not a number.
        assert_eq!(pointer_to_guest((-1.0, -1.0), image, guest), (0, 0));
        assert_eq!(pointer_to_guest((1e9, 1e9), image, guest), (1919, 1199));
        assert_eq!(pointer_to_guest((f64::NAN, f64::NAN), image, guest), (0, 0));
        // An empty image or mode.
        assert_eq!(pointer_to_guest((5.0, 5.0), Rect::default(), guest), (0, 0));
        assert_eq!(pointer_to_guest((500.0, 300.0), image, (0, 0)), (0, 0));
    }

    /// The host's buttons 0 to 4 are left, right, middle and the two side buttons
    #[test]
    fn buttons_map_to_the_five_bits() {
        assert_eq!(button_bit(0), BUTTON_LEFT);
        assert_eq!(button_bit(1), BUTTON_RIGHT);
        assert_eq!(button_bit(2), BUTTON_MIDDLE);
        assert_eq!(button_bit(3), BUTTON_SIDE);
        assert_eq!(button_bit(4), BUTTON_EXTRA);
        assert_eq!(button_bit(5), 0);
        assert_eq!(button_bit(31), 0);
        assert_eq!(button_bit(-1), 0);
        let all = [0, 1, 2, 3, 4].map(button_bit);
        assert_eq!(all.iter().fold(0, |a, b| a | b).count_ones(), 5);
    }

    /// A new window shows a guest pixel per host point
    #[test]
    fn a_new_window_shows_a_guest_point_per_host_point() {
        assert_eq!(
            initial_content_size((960, 600), (1728.0, 1080.0)),
            (960.0, 600.0)
        );
        assert_eq!(
            initial_content_size((1280, 800), (1920.0, 1050.0)),
            (1280.0, 800.0)
        );
        // On a 1× screen its drawable is the guest mode, and the image fills it.
        let size = initial_content_size((1920, 1200), (1920.0, 1200.0));
        let l = layout(size, 1.0, NO_INSETS, Some((1920, 1200)));
        assert_eq!(l.drawable, (1920, 1200));
        assert_eq!(
            l.pixels,
            PixelRect {
                x: 0,
                y: 0,
                width: 1920,
                height: 1200
            }
        );
        // 1920×1080 on a Retina screen opens at 1920×1080 points where they fit: its drawable is
        // twice the mode each way, and the image, scaled on the GPU, fills it.
        let size = initial_content_size((1920, 1080), (1920.0, 1200.0));
        assert_eq!(size, (1920.0, 1080.0));
        let l = layout(size, 2.0, NO_INSETS, Some((1920, 1080)));
        assert_eq!(l.drawable, (3840, 2160));
        assert_eq!(
            l.pixels,
            PixelRect {
                x: 0,
                y: 0,
                width: 3840,
                height: 2160
            }
        );
    }

    /// A guest larger than the screen's visible area is shrunk to fit it, ratio kept
    #[test]
    fn a_new_window_shrinks_to_the_visible_area() {
        let (w, h) = initial_content_size((1728, 1080), (1728.0, 1050.0));
        assert!(
            (w - 1680.0).abs() < 1e-9 && (h - 1050.0).abs() < 1e-9,
            "{w}x{h}"
        );
        let (w, h) = initial_content_size((3840, 2160), (1920.0, 1200.0));
        assert_eq!((w, h), (1920.0, 1080.0));
        // No room given: left at 1:1.
        assert_eq!(
            initial_content_size((1024, 768), (0.0, 0.0)),
            (1024.0, 768.0)
        );
        // Never below a point.
        assert_eq!(initial_content_size((0, 0), (100.0, 100.0)), (1.0, 1.0));
    }

    /// A window following the guest's mode keeps its top left corner where the new size fits
    #[test]
    fn a_resize_keeps_the_top_left_corner() {
        // The built-in screen's visible area below the menu bar, y down.
        let visible = Rect {
            x: 0.0,
            y: 38.0,
            width: 1728.0,
            height: 1079.0,
        };
        let frame = Rect {
            x: 100.0,
            y: 200.0,
            width: 960.0,
            height: 568.0,
        };
        let smaller = resized_frame(frame, (640.0, 428.0), visible);
        assert_eq!((smaller.x, smaller.y), (100.0, 200.0));
        assert_eq!((smaller.width, smaller.height), (640.0, 428.0));
        let larger = resized_frame(frame, (1600.0, 900.0), visible);
        assert_eq!(
            larger,
            Rect {
                x: 100.0,
                y: 200.0,
                width: 1600.0,
                height: 900.0
            }
        );
    }

    /// A window following the guest's mode is moved inside the screen's visible area, and where
    /// it does not fit, its left and top edges are the area's: its title bar stays on the screen
    #[test]
    fn a_resize_stays_in_the_visible_area() {
        // A second screen, right of the first, y down.
        let visible = Rect {
            x: 1728.0,
            y: 0.0,
            width: 1920.0,
            height: 1050.0,
        };
        let frame = Rect {
            x: 2800.0,
            y: 500.0,
            width: 700.0,
            height: 400.0,
        };
        // Past the right and bottom edges: moved left and up, just inside.
        let r = resized_frame(frame, (1000.0, 800.0), visible);
        assert_eq!((r.x, r.y), (2648.0, 250.0));
        // Wider and taller than the area: at its left and top.
        let r = resized_frame(frame, (2000.0, 1200.0), visible);
        assert_eq!((r.x, r.y), (1728.0, 0.0));
        assert_eq!((r.width, r.height), (2000.0, 1200.0));
        // Above or left of the area: moved down or right into it.
        let above = Rect {
            x: 1700.0,
            y: -30.0,
            ..frame
        };
        let r = resized_frame(above, (700.0, 400.0), visible);
        assert_eq!((r.x, r.y), (1728.0, 0.0));
    }

    /// The guest's cursor is drawn at the image's points per guest pixel, its hot spot scaled
    /// with it
    #[test]
    fn the_cursor_scales_with_the_image() {
        // A 1920-wide guest shown 960 points wide: half a point per guest pixel.
        let image = Rect {
            x: 0.0,
            y: 0.0,
            width: 960.0,
            height: 600.0,
        };
        let ppp = cursor_points_per_pixel(image, (1920, 1200));
        assert_eq!(ppp, 0.5);
        assert_eq!(
            cursor_size((64, 64), (10, 20), ppp),
            CursorSize {
                size: (32.0, 32.0),
                hot_spot: (5.0, 10.0)
            }
        );
        // The window doubled: the cursor doubles.
        let image = Rect {
            width: 1920.0,
            height: 1200.0,
            ..image
        };
        let ppp = cursor_points_per_pixel(image, (1920, 1200));
        assert_eq!(
            cursor_size((64, 64), (10, 20), ppp),
            CursorSize {
                size: (64.0, 64.0),
                hot_spot: (10.0, 20.0)
            }
        );
        // A hot spot past the glyph holds at its last pixel.
        assert_eq!(cursor_size((16, 8), (40, 40), 2.0).hot_spot, (30.0, 14.0));
        // An empty mode draws at a point per pixel.
        assert_eq!(cursor_points_per_pixel(image, (0, 0)), 1.0);
    }
}
