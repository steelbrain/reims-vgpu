//! The host's cursor over a window's image: the guest's glyph as an `NSCursor`, sized as the
//! guest draws it on the image, or an invisible cursor while the guest hides its own.
//!
//! The glyph's pixels are an `NSBitmapImageRep`'s own copy of its RGBA bytes: 8 bits a sample,
//! four samples, alpha last, device RGB, as the guest's glyph comes. The alpha is straight, not
//! premultiplied: a guest's cursor then looks as it does under Virtualization.framework and in
//! QEMU's own displays, which both draw the glyph's colour and alpha as they are. The
//! representation's image is sized in points at the image's scale ([`geometry::cursor_size`]),
//! and keeps its pixels: AppKit draws the cursor from them at the screen's scale, so on a Retina
//! screen that shows the guest one to one, each glyph pixel is one screen pixel.

use objc2::rc::Retained;
use objc2::{AllocAnyThread, MainThreadMarker};
use objc2_app_kit::{NSBitmapFormat, NSBitmapImageRep, NSCursor, NSDeviceRGBColorSpace, NSImage};
use objc2_foundation::{NSPoint, NSSize};
use reims_vgpu_vulkan::os::msg;

use crate::geometry;
use crate::handle::Glyph;

/// The invisible cursor's glyph: one pixel, fully transparent.
fn invisible_glyph() -> Glyph {
    Glyph {
        width: 1,
        height: 1,
        hot_spot: (0, 0),
        rgba: vec![0; 4],
    }
}

/// A cursor built from a glyph, and what it was built from.
#[derive(Debug)]
struct Built {
    /// The glyph.
    glyph: Glyph,
    /// Points per guest pixel.
    points_per_pixel: f64,
    /// The cursor.
    cursor: Retained<NSCursor>,
}

/// The cursors built for one window, rebuilt when the glyph or the image's scale changes.
#[derive(Debug, Default)]
pub(crate) struct CursorCache {
    /// The invisible cursor, once built.
    invisible: Option<Retained<NSCursor>>,
    /// The last cursor built from the guest's glyph.
    built: Option<Built>,
}

impl CursorCache {
    /// An empty cache.
    pub(crate) fn new() -> CursorCache {
        CursorCache::default()
    }

    /// The cursor over the image: the guest's `glyph` drawn at `points_per_pixel` points per
    /// guest pixel, hot spot included, while `visible`; an invisible cursor while the guest hides
    /// its own; `None` (the host's arrow) before the guest has set a glyph, or for a glyph with
    /// no pixels. An unchanged request answers the cursor it answered before. Made on the main
    /// thread (`mtm`), where AppKit sets cursors.
    pub(crate) fn cursor(
        &mut self,
        glyph: Option<&Glyph>,
        visible: bool,
        points_per_pixel: f64,
        mtm: MainThreadMarker,
    ) -> Option<Retained<NSCursor>> {
        let _ = mtm;
        self.lookup(glyph, visible, points_per_pixel)
    }

    /// [`CursorCache::cursor`], on any thread.
    fn lookup(
        &mut self,
        glyph: Option<&Glyph>,
        visible: bool,
        points_per_pixel: f64,
    ) -> Option<Retained<NSCursor>> {
        let glyph = glyph?;
        if !visible {
            if self.invisible.is_none() {
                self.invisible = build(&invisible_glyph(), 1.0);
            }
            return self.invisible.clone();
        }
        if let Some(built) = &self.built
            && built.points_per_pixel.to_bits() == points_per_pixel.to_bits()
            && built.glyph == *glyph
        {
            return Some(built.cursor.clone());
        }
        let cursor = build(glyph, points_per_pixel)?;
        self.built = Some(Built {
            glyph: glyph.clone(),
            points_per_pixel,
            cursor: cursor.clone(),
        });
        Some(cursor)
    }
}

/// `glyph` as a cursor drawn at `points_per_pixel` points per glyph pixel, or `None` for a glyph
/// with no pixels.
fn build(glyph: &Glyph, points_per_pixel: f64) -> Option<Retained<NSCursor>> {
    let rep = bitmap(glyph)?;
    let size = geometry::cursor_size(
        (glyph.width, glyph.height),
        glyph.hot_spot,
        points_per_pixel,
    );
    let image = msg::get("initWithSize:", || {
        NSImage::initWithSize(NSImage::alloc(), NSSize::new(size.size.0, size.size.1))
    });
    msg::get("addRepresentation:", || image.addRepresentation(&rep));
    let hot_spot = NSPoint::new(size.hot_spot.0, size.hot_spot.1);
    Some(msg::get("initWithImage:hotSpot:", || {
        NSCursor::initWithImage_hotSpot(NSCursor::alloc(), &image, hot_spot)
    }))
}

/// A bitmap of `glyph`'s pixels (module doc), holding its own copy of them: `height` rows of
/// `width * 4` bytes, a row the glyph's bytes do not cover being transparent. `None` for a
/// glyph with no pixels, or where AppKit makes no bitmap.
fn bitmap(glyph: &Glyph) -> Option<Retained<NSBitmapImageRep>> {
    let (width, height) = (
        usize::try_from(glyph.width).ok()?,
        usize::try_from(glyph.height).ok()?,
    );
    if width == 0 || height == 0 {
        return None;
    }
    let row = width.checked_mul(4)?;
    let (pixels_wide, pixels_high, bytes_per_row) = (
        isize::try_from(width).ok()?,
        isize::try_from(height).ok()?,
        isize::try_from(row).ok()?,
    );
    // SAFETY: the colour space name is AppKit's constant.
    let device_rgb = unsafe { NSDeviceRGBColorSpace };
    let rep = msg::get(
        "initWithBitmapDataPlanes:pixelsWide:pixelsHigh:bitsPerSample:samplesPerPixel:hasAlpha:\
         isPlanar:colorSpaceName:bitmapFormat:bytesPerRow:bitsPerPixel:",
        || {
            // SAFETY: no planes: the representation allocates its own pixels, `bytes_per_row`
            // bytes a row, `pixels_high` rows.
            unsafe {
                NSBitmapImageRep::initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bitmapFormat_bytesPerRow_bitsPerPixel(
                    NSBitmapImageRep::alloc(),
                    std::ptr::null_mut(),
                    pixels_wide,
                    pixels_high,
                    8,
                    4,
                    true,
                    false,
                    device_rgb,
                    NSBitmapFormat::AlphaNonpremultiplied,
                    bytes_per_row,
                    32,
                )
            }
        },
    )?;
    let data = msg::get("bitmapData", || rep.bitmapData());
    let stride = usize::try_from(msg::get("bytesPerRow", || rep.bytesPerRow())).ok()?;
    if data.is_null() || stride < row {
        return None;
    }
    let length = stride.checked_mul(height)?;
    // SAFETY: a meshed bitmap's data is `bytesPerRow` bytes a row, `pixelsHigh` rows, owned by
    // `rep`, which outlives the slice; nothing else reads or writes it meanwhile.
    let pixels = unsafe { std::slice::from_raw_parts_mut(data, length) };
    for (y, dst) in pixels.chunks_exact_mut(stride).enumerate() {
        let src = glyph.rgba.get(y * row..).unwrap_or_default();
        let src = src.get(..row).unwrap_or(src);
        let (copied, rest) = dst.split_at_mut(src.len());
        copied.copy_from_slice(src);
        rest.fill(0);
    }
    Some(rep)
}

#[cfg(test)]
mod tests {
    use objc2_app_kit::NSImageRep;

    use super::*;

    /// A `width` by `height` glyph whose pixel `i` is `[i, 2i, 3i, alpha]`.
    fn glyph(width: u32, height: u32, hot_spot: (u16, u16), alpha: u8) -> Glyph {
        let rgba = (0..width * height)
            .flat_map(|i| {
                let i = i as u8;
                [i, i.wrapping_mul(2), i.wrapping_mul(3), alpha]
            })
            .collect();
        Glyph {
            width,
            height,
            hot_spot,
            rgba,
        }
    }

    /// The cursor's one bitmap: its pixel size, its format and its bytes.
    fn bitmap_of(cursor: &NSCursor) -> (isize, isize, NSBitmapFormat, Vec<u8>) {
        let reps = cursor.image().representations();
        assert_eq!(reps.len(), 1);
        let rep: Retained<NSImageRep> = reps.objectAtIndex(0);
        let rep = rep.downcast::<NSBitmapImageRep>().unwrap();
        let (w, h) = (rep.pixelsWide(), rep.pixelsHigh());
        let stride = rep.bytesPerRow() as usize;
        // SAFETY: the bitmap's data is `bytesPerRow` bytes a row, `pixelsHigh` rows.
        let data = unsafe { std::slice::from_raw_parts(rep.bitmapData(), stride * h as usize) };
        let bytes = data
            .chunks_exact(stride)
            .flat_map(|row| &row[..w as usize * 4])
            .copied()
            .collect();
        (w, h, rep.bitmapFormat(), bytes)
    }

    /// The cursor is the guest's glyph: its pixels, straight alpha, drawn at the image's points
    /// per guest pixel with its hot spot scaled
    #[test]
    fn the_cursor_is_the_glyph_at_the_images_scale() {
        let mut cache = CursorCache::new();
        let g = glyph(4, 2, (3, 1), 0x80);
        let cursor = cache.lookup(Some(&g), true, 2.5).unwrap();
        let size = cursor.image().size();
        assert_eq!((size.width, size.height), (10.0, 5.0));
        let hot = cursor.hotSpot();
        assert_eq!((hot.x, hot.y), (7.5, 2.5));
        let (w, h, format, bytes) = bitmap_of(&cursor);
        assert_eq!((w, h), (4, 2));
        assert!(format.contains(NSBitmapFormat::AlphaNonpremultiplied));
        assert!(!format.contains(NSBitmapFormat::AlphaFirst));
        assert_eq!(bytes, g.rgba);
    }

    /// Rows the glyph's bytes do not reach are transparent
    #[test]
    fn a_short_glyph_is_transparent_past_its_bytes() {
        let mut cache = CursorCache::new();
        let mut g = glyph(2, 2, (0, 0), 0xff);
        g.rgba.truncate(12);
        let cursor = cache.lookup(Some(&g), true, 1.0).unwrap();
        let (_, _, _, bytes) = bitmap_of(&cursor);
        assert_eq!(&bytes[..12], &g.rgba[..]);
        assert_eq!(&bytes[12..], &[0; 4]);
    }

    /// An unchanged request answers the same cursor; a new glyph or scale builds another
    #[test]
    fn an_unchanged_request_answers_the_same_cursor() {
        let mut cache = CursorCache::new();
        let g = glyph(8, 8, (1, 1), 0xff);
        let first = cache.lookup(Some(&g), true, 1.0).unwrap();
        let again = cache.lookup(Some(&g.clone()), true, 1.0).unwrap();
        assert!(std::ptr::eq(&*first, &*again));
        let scaled = cache.lookup(Some(&g), true, 0.5).unwrap();
        assert!(!std::ptr::eq(&*first, &*scaled));
        let other = glyph(8, 8, (2, 2), 0xff);
        let changed = cache.lookup(Some(&other), true, 0.5).unwrap();
        assert!(!std::ptr::eq(&*scaled, &*changed));
    }

    /// While the guest hides its cursor the host's is invisible: one transparent pixel, the
    /// same cursor each time
    #[test]
    fn a_hidden_guest_cursor_is_invisible() {
        let mut cache = CursorCache::new();
        let g = glyph(8, 8, (1, 1), 0xff);
        let hidden = cache.lookup(Some(&g), false, 2.0).unwrap();
        let (w, h, _, bytes) = bitmap_of(&hidden);
        assert_eq!((w, h), (1, 1));
        assert_eq!(bytes, vec![0; 4]);
        let again = cache.lookup(Some(&g), false, 1.0).unwrap();
        assert!(std::ptr::eq(&*hidden, &*again));
    }

    /// Before the guest sets a glyph, and for a glyph with no pixels, the host's arrow stays
    #[test]
    fn with_no_glyph_the_host_keeps_its_arrow() {
        let mut cache = CursorCache::new();
        assert!(cache.lookup(None, true, 1.0).is_none());
        assert!(cache.lookup(None, false, 1.0).is_none());
        assert!(
            cache
                .lookup(Some(&glyph(0, 4, (0, 0), 0xff)), true, 1.0)
                .is_none()
        );
    }
}
