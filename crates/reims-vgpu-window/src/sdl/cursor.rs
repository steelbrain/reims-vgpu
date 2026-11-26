//! The host's cursor over a window's image: the guest's glyph as SDL's colour cursor, sized as the
//! image shows it, or hidden while the guest hides its own. UI thread only.
//!
//! The glyph's RGBA bytes are straight alpha, as the guest's glyph comes, so a guest's cursor
//! looks as it does on macOS and in QEMU's own displays. SDL takes a cursor's image in its sizes
//! ([`images`]). Where they are points (Wayland, macOS), that is the image at the screen's 100%
//! scale, with an alternate image for the screen's own scale, in pixels
//! (`SDL_AddSurfaceAlternateImage`), which SDL draws on a screen of that scale: at 190% a guest one
//! to one with the screen's pixels gets one glyph pixel per screen pixel. Where they are pixels
//! (X11, Windows), it is the image at the window's pixels alone, which SDL draws as it is: read
//! from SDL 3.4.18's code, its X11 driver never scales a cursor's image, and its Windows driver
//! scales it by the monitor's content scale only under `SDL_HINT_MOUSE_DPI_SCALE_CURSORS`, which
//! the window leaves off there (`native/sdl_thread/ui.rs`): on a Windows screen at 150% that would
//! draw the guest's cursor half as large again, its hot spot not scaled with it. Each image is the
//! glyph resampled to its size by nearest pixel ([`resample`]), so a 1× guest's glyph keeps its
//! hard edges as the image scales it.

use sdl3_sys::everything::{
    SDL_AddSurfaceAlternateImage, SDL_CreateColorCursor, SDL_CreateSurface, SDL_Cursor,
    SDL_DestroyCursor, SDL_DestroySurface, SDL_PIXELFORMAT_RGBA32, SDL_Surface,
};

use crate::geometry;
use crate::handle::Glyph;

/// A cursor SDL made, destroyed with this.
#[derive(Debug)]
pub(crate) struct Cursor(*mut SDL_Cursor);

impl Cursor {
    /// SDL's cursor.
    pub(crate) fn raw(&self) -> *mut SDL_Cursor {
        self.0
    }
}

impl Drop for Cursor {
    fn drop(&mut self) {
        // SAFETY: the cursor is SDL's, made on the UI thread, where this runs; SDL falls back to
        // its default cursor where this one is set.
        unsafe { SDL_DestroyCursor(self.0) };
    }
}

/// The cursor built from a glyph, and what it was built from.
#[derive(Debug)]
struct Built {
    /// The glyph.
    glyph: Glyph,
    /// Points per guest pixel.
    points_per_pixel: f64,
    /// The screen's pixels per point.
    density: f64,
    /// The cursor.
    cursor: Cursor,
}

/// The cursor built for one window, built again when the glyph or a scale changes.
#[derive(Debug, Default)]
pub(crate) struct CursorCache {
    /// The last cursor built from the guest's glyph.
    built: Option<Built>,
}

impl CursorCache {
    /// The guest's `glyph` as a cursor drawn at `points_per_pixel` points per guest pixel, on a
    /// screen of `density` pixels per point; `None` (the host's arrow) before the guest has set
    /// a glyph, for one with no pixels, or where SDL makes none. An unchanged request answers the
    /// cursor it answered before.
    pub(crate) fn cursor(
        &mut self,
        glyph: Option<&Glyph>,
        points_per_pixel: f64,
        density: f64,
    ) -> Option<&Cursor> {
        let glyph = glyph?;
        let same = self.built.as_ref().is_some_and(|b| {
            b.points_per_pixel.to_bits() == points_per_pixel.to_bits()
                && b.density.to_bits() == density.to_bits()
                && b.glyph == *glyph
        });
        if !same {
            self.built = build(glyph, points_per_pixel, density).map(|cursor| Built {
                glyph: glyph.clone(),
                points_per_pixel,
                density,
                cursor,
            });
        }
        self.built.as_ref().map(|b| &b.cursor)
    }
}

/// `glyph` as a cursor drawn at `points_per_pixel` points per glyph pixel on a screen of
/// `density` (the module doc), or `None`.
fn build(glyph: &Glyph, points_per_pixel: f64, density: f64) -> Option<Cursor> {
    let size = geometry::cursor_size(
        (glyph.width, glyph.height),
        glyph.hot_spot,
        points_per_pixel,
    );
    let (points, pixels) = images(size.size, density);
    let base = surface(glyph, points)?;
    if let Some(pixels) = pixels
        && let Some(alternate) = surface(glyph, pixels)
    {
        // SAFETY: both surfaces are live; SDL takes a reference to the alternate.
        unsafe {
            SDL_AddSurfaceAlternateImage(base, alternate);
            SDL_DestroySurface(alternate);
        }
    }
    let hot = |v: f64, extent: u32| (v.round() as i32).clamp(0, extent.saturating_sub(1) as i32);
    // SAFETY: `base` is a live RGBA surface; SDL copies what it needs into the cursor.
    let cursor = unsafe {
        let cursor = SDL_CreateColorCursor(
            base,
            hot(size.hot_spot.0, points.0),
            hot(size.hot_spot.1, points.1),
        );
        SDL_DestroySurface(base);
        cursor
    };
    (!cursor.is_null()).then_some(Cursor(cursor))
}

/// The pixels of the images SDL takes for a cursor of `size` in SDL's sizes, on a window of
/// `density` pixels a unit of them (the module doc): the image at SDL's sizes, and where the
/// density is above 1 (points, on a screen above 100%) an alternate at the window's pixels; each
/// on whole pixels, at least 1 a side.
fn images(size: (f64, f64), density: f64) -> ((u32, u32), Option<(u32, u32)>) {
    let whole = |v: f64| v.round().max(1.0) as u32;
    let base = (whole(size.0), whole(size.1));
    let alternate = (density.is_finite() && density > 1.0)
        .then(|| (whole(size.0 * density), whole(size.1 * density)));
    (base, alternate)
}

/// An RGBA surface of `size` pixels holding `glyph` resampled to it, or `None` for a glyph with
/// no pixels, or where SDL makes no surface.
fn surface(glyph: &Glyph, size: (u32, u32)) -> Option<*mut SDL_Surface> {
    let pixels = resample(glyph, size)?;
    let (width, height) = (i32::try_from(size.0).ok()?, i32::try_from(size.1).ok()?);
    // SAFETY: a valid size and format.
    let surface = unsafe { SDL_CreateSurface(width, height, SDL_PIXELFORMAT_RGBA32) };
    if surface.is_null() {
        return None;
    }
    // SAFETY: the surface is SDL's new one, `height` rows of at least `width * 4` bytes, `pitch`
    // apart, which nothing else touches.
    unsafe {
        let s = &*surface;
        let pitch = usize::try_from(s.pitch).ok()?;
        let row = usize::try_from(size.0).ok()?.checked_mul(4)?;
        for (y, line) in pixels.chunks_exact(row).enumerate() {
            let at = s.pixels.cast::<u8>().add(y * pitch);
            std::ptr::copy_nonoverlapping(line.as_ptr(), at, row);
        }
    }
    Some(surface)
}

/// `glyph`'s pixels resampled to `size` by nearest pixel: `size.1` rows of `size.0 * 4` RGBA
/// bytes. A pixel the glyph's bytes do not cover is transparent. `None` for a glyph or a size
/// with no pixels.
pub(crate) fn resample(glyph: &Glyph, size: (u32, u32)) -> Option<Vec<u8>> {
    let (gw, gh) = (
        usize::try_from(glyph.width).ok()?,
        usize::try_from(glyph.height).ok()?,
    );
    let (w, h) = (usize::try_from(size.0).ok()?, usize::try_from(size.1).ok()?);
    if gw == 0 || gh == 0 || w == 0 || h == 0 {
        return None;
    }
    let mut out = vec![0u8; w.checked_mul(h)?.checked_mul(4)?];
    for (y, row) in out.chunks_exact_mut(w * 4).enumerate() {
        let sy = y * gh / h;
        for (x, px) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let sx = x * gw / w;
            let at = (sy * gw + sx) * 4;
            if let Some(src) = glyph.rgba.get(at..at + 4) {
                px.copy_from_slice(src);
            }
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2×2 glyph: red, green / blue, clear.
    fn glyph() -> Glyph {
        Glyph {
            width: 2,
            height: 2,
            hot_spot: (0, 0),
            rgba: vec![
                255, 0, 0, 255, 0, 255, 0, 255, //
                0, 0, 255, 255, 0, 0, 0, 0,
            ],
        }
    }

    /// At its own size the glyph is its own pixels; doubled, each pixel is a 2×2 block
    #[test]
    fn a_glyph_is_resampled_by_nearest_pixel() {
        assert_eq!(resample(&glyph(), (2, 2)), Some(glyph().rgba));
        let doubled = resample(&glyph(), (4, 4)).unwrap();
        let px = |x: usize, y: usize| &doubled[(y * 4 + x) * 4..(y * 4 + x) * 4 + 4];
        assert_eq!(px(0, 0), &[255, 0, 0, 255]);
        assert_eq!(px(1, 1), &[255, 0, 0, 255]);
        assert_eq!(px(2, 0), &[0, 255, 0, 255]);
        assert_eq!(px(0, 3), &[0, 0, 255, 255]);
        assert_eq!(px(3, 3), &[0, 0, 0, 0]);
    }

    /// SDL takes the guest's cursor at the size the image shows it, on every host: where SDL's
    /// sizes are pixels (Windows and X11, whose windows have a density of 1 at any scale), the
    /// image at the window's pixels alone, which SDL draws as it is (SDL is not told to scale
    /// cursors: `native::sdl_thread::ui`); where they are points (Wayland, macOS), the image in
    /// points and one at the screen's pixels, which SDL draws on a screen of that scale
    #[test]
    fn a_cursor_is_handed_sdl_at_the_guest_s_size_in_sdl_s_units() {
        // A 32 × 32 glyph of a guest of `guest` pixels shown on an image `width` wide in SDL's
        // sizes, on a window of `density` pixels a unit of them: the image SDL takes, and the
        // alternate.
        let handed = |width: f64, guest: (u16, u16), density: f64| {
            let image = geometry::Rect {
                width,
                ..geometry::Rect::default()
            };
            let per_pixel = geometry::cursor_points_per_pixel(image, guest);
            let size = geometry::cursor_size((32, 32), (0, 0), per_pixel).size;
            images(size, density)
        };
        // Windows at 150%: a guest of 2880 × 1620 on an image of 2880 pixels, a guest pixel a
        // screen pixel, hands SDL its glyph's 32 pixels, which SDL draws as they are, not at 48.
        assert_eq!(handed(2880.0, (2880, 1620), 1.0), ((32, 32), None));
        // Windows at 125%: a guest 1920 pixels wide on an image of 2400 pixels, 40.
        assert_eq!(handed(2400.0, (1920, 1080), 1.0), ((40, 40), None));
        // X11 the same way: a guest at a screen pixel a pixel, and one at two.
        assert_eq!(handed(2560.0, (2560, 1440), 1.0), ((32, 32), None));
        assert_eq!(handed(2560.0, (1280, 720), 1.0), ((64, 64), None));
        // Wayland and macOS at 2: a guest of 2880 × 1800 on an image of 1440 points, 16 points
        // and 32 pixels.
        assert_eq!(
            handed(1440.0, (2880, 1800), 2.0),
            ((16, 16), Some((32, 32)))
        );
        // Wayland at 2.1: 1829 points, 3841 pixels, over a guest 3840 pixels wide.
        assert_eq!(
            handed(1829.0, (3840, 2160), 3841.0 / 1829.0),
            ((15, 15), Some((32, 32)))
        );
        // macOS at 1: a point a guest pixel.
        assert_eq!(handed(1440.0, (1440, 900), 1.0), ((32, 32), None));
        // No pixels to speak of, or no density known: one pixel a side, and no alternate.
        assert_eq!(images((0.2, f64::NAN), f64::NAN), ((1, 1), None));
    }

    /// A glyph short of its bytes is transparent where they run out; one with no pixels is none
    #[test]
    fn a_short_or_empty_glyph() {
        let mut short = glyph();
        short.rgba.truncate(8);
        let out = resample(&short, (2, 2)).unwrap();
        assert_eq!(&out[8..], &[0; 8]);
        let empty = Glyph::default();
        assert_eq!(resample(&empty, (2, 2)), None);
        assert_eq!(resample(&glyph(), (0, 2)), None);
    }
}
