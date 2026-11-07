//! SDL's sizes, and a windowed window's first size in them. SDL's sizes are points on Wayland and
//! on macOS, and pixels on X11, where a desktop at a content scale above 1 has more than one of
//! them to the host's point: read from SDL 3.4.18's code, its X11 driver makes a window of the
//! size asked for in pixels, its display bounds and usable bounds are pixels, its modes have a
//! pixel density of 1, and the content scale is the desktop's (`Xft.dpi` over 96, else GDK's
//! scale). Wayland's modes have the output's scale as their density, with SDL's sizes in points.
//!
//! Linux sizes its windows here (`native/linux/sdl.rs`); macOS's SDL windows open in the frames
//! AppKit's window takes, in points.

#![cfg_attr(
    all(not(target_os = "linux"), not(test)),
    expect(dead_code, reason = "macOS's SDL windows open in AppKit's frames")
)]

use super::shell::MIN_CONTENT;
use crate::geometry;

/// A screen's pixels per host point, from its current mode's `pixel_density` and SDL's
/// `content_scale` of it: the density where it is above 1 (Wayland at a scale above 1), else the
/// content scale (X11, where the desktop's scale is the content scale alone), else 1.
pub(crate) fn pixels_per_point(pixel_density: f32, content_scale: f32) -> f64 {
    let density = f64::from(pixel_density);
    if density > 1.0 {
        density
    } else if content_scale.is_finite() && content_scale > 0.0 {
        f64::from(content_scale)
    } else {
        1.0
    }
}

/// What SDL's sizes on a screen are (the module doc).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Units {
    /// Pixels per unit of SDL's sizes: the mode's pixel density, at least 1.
    pixels: f64,
    /// Pixels per host point ([`pixels_per_point`]).
    scale: f64,
}

impl Units {
    /// The units of a screen whose current mode has `pixel_density` and whose content scale is
    /// `content_scale`. A density above 1 (Wayland's) is taken to the nearest 120th, the unit of
    /// Wayland's fractional scales, by which SDL's Wayland backend sizes a window's buffer: the
    /// mode's density is a float (2.1 reads as 2.0999999), and at a size whose pixels end in a
    /// half (965 points at 2.1 are 2026.5) the float rounds the other way.
    pub(crate) fn new(pixel_density: f32, content_scale: f32) -> Units {
        let in_120ths = |scale: f64| (scale * 120.0).round() / 120.0;
        let pixels = f64::from(pixel_density).max(1.0);
        let scale = pixels_per_point(pixel_density, content_scale);
        if pixels > 1.0 {
            Units {
                pixels: in_120ths(pixels),
                scale: in_120ths(scale),
            }
        } else {
            Units { pixels, scale }
        }
    }

    /// Units per host point.
    fn per_point(self) -> f64 {
        self.scale / self.pixels
    }

    /// `points` host points, on whole pixels, as a size SDL takes.
    fn whole(self, points: f64) -> i32 {
        ((points * self.scale).floor() / self.pixels).round() as i32
    }

    /// The pixels SDL makes of `size`, one of its sizes: the size times the pixels per unit,
    /// rounded halfway away from zero, as SDL's Wayland backend rounds a window's buffer from its
    /// size in points (its `PointToPixel`, read from SDL 3.4.18's code); on X11 the size itself.
    fn pixels_of(self, size: i32) -> i64 {
        (f64::from(size) * self.pixels + 1e-6).round() as i64
    }

    /// The largest of SDL's sizes up to `size`, and not below `min`, whose pixels are even: at a
    /// fractional scale some sizes in points come to an odd number of pixels (at 2.1, 971 points
    /// are 2039 pixels, and the ten sizes from 965 to 974 points all come out odd).
    fn even(self, size: i32, min: i32) -> i32 {
        let mut even = size;
        while even > min && self.pixels_of(even) % 2 != 0 {
            even -= 1;
        }
        even
    }

    /// [`MIN_CONTENT`] as a size SDL takes.
    pub(crate) fn min_content(self) -> (i32, i32) {
        let side = |points: i32| (f64::from(points) * self.per_point()).round() as i32;
        (side(MIN_CONTENT.0), side(MIN_CONTENT.1))
    }
}

/// The content size, in `units`, of a windowed window for a guest mode `mode` at `guest_scale`,
/// on a screen whose usable bounds are `usable` in those units, where known: a guest point per
/// host point ([`geometry::initial_content_size`]), shrunk to fit the usable bounds less
/// `title_room` points, on whole pixels, made smaller where it takes that to come to an even
/// number of pixels each way, and at least [`MIN_CONTENT`] points. The guest is offered the
/// window's pixels rounded down to even (the window crate's `geometry`), and a HiDPI mode is
/// one whose pixels halve exactly, so a window of even pixels shows the guest's whole mode.
pub(crate) fn windowed_size(
    mode: (u16, u16),
    guest_scale: u8,
    usable: Option<(i32, i32)>,
    title_room: f64,
    units: Units,
) -> (i32, i32) {
    let per_point = units.per_point();
    let room = usable.map_or((0.0, 0.0), |(w, h)| {
        (
            f64::from(w) / per_point,
            f64::from(h) / per_point - title_room,
        )
    });
    let (width, height) = geometry::initial_content_size(mode, f64::from(guest_scale.max(1)), room);
    let min = units.min_content();
    (
        units.even(units.whole(width).max(min.0), min.0),
        units.even(units.whole(height).max(min.1), min.1),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A screen's scale is its mode's density above 1, else its content scale
    #[test]
    fn a_screen_s_scale_is_its_density_else_its_content_scale() {
        assert_eq!(pixels_per_point(2.0, 1.0), 2.0);
        assert_eq!(pixels_per_point(1.5, 1.0), 1.5);
        assert_eq!(pixels_per_point(1.0, 2.0), 2.0);
        assert_eq!(pixels_per_point(1.0, 1.25), 1.25);
        assert_eq!(pixels_per_point(1.0, f32::NAN), 1.0);
        assert_eq!(pixels_per_point(0.0, 0.0), 1.0);
    }

    /// A window opens at a guest point per host point in SDL's sizes: points on Wayland, and on
    /// X11 pixels, the content scale of them to the point, so a window at a content scale of 2
    /// is not opened at half its size; on whole pixels, shrunk to fit, and never below the
    /// smallest content
    #[test]
    fn a_window_opens_a_guest_point_per_host_point_in_sdl_s_sizes() {
        const ROOM: f64 = 48.0;
        let fhd = (1920, 1080);
        // Wayland at 1×, and at 2×: points, the usable bounds in points.
        let wayland = Units::new(1.0, 1.0);
        assert_eq!(
            windowed_size(fhd, 1, Some((2560, 1400)), ROOM, wayland),
            (1920, 1080)
        );
        let wayland_2x = Units::new(2.0, 1.0);
        assert_eq!(
            windowed_size((3840, 2160), 2, Some((1920, 1080)), ROOM, wayland_2x),
            (1835, 1032)
        );
        // Wayland at 150%: whole pixels, 1280 points of 1.5 pixels.
        let wayland_150 = Units::new(1.5, 1.0);
        assert_eq!(
            windowed_size((1280, 800), 1, Some((1707, 960)), ROOM, wayland_150),
            (1280, 800)
        );
        // X11 at a content scale of 2: pixels, two to the point; a 1× guest of 1920 × 1080
        // shrinks to fit 1920 × 1008 points, 3584 × 2016 pixels.
        let x11_2x = Units::new(1.0, 2.0);
        assert_eq!(
            windowed_size(fhd, 1, Some((3840, 2112)), ROOM, x11_2x),
            (3584, 2016)
        );
        // A HiDPI guest there is a guest pixel per screen pixel.
        assert_eq!(
            windowed_size((2560, 1600), 2, Some((3840, 2112)), ROOM, x11_2x),
            (2560, 1600)
        );
        // X11 at 150%: 1280 × 800 points are 1920 × 1200 pixels.
        let x11_150 = Units::new(1.0, 1.5);
        assert_eq!(
            windowed_size((1280, 800), 1, Some((2560, 1392)), ROOM, x11_150),
            (1920, 1200)
        );
        // The smallest content, 320 × 200 points, in pixels at 2.
        assert_eq!(x11_2x.min_content(), (640, 400));
        assert_eq!(wayland_2x.min_content(), (320, 200));
        assert_eq!(windowed_size((400, 300), 2, None, ROOM, x11_2x), (640, 400));
    }

    /// At the fractional scales desktops offer, a window opens on an even number of pixels each
    /// way, as SDL rounds its sizes to pixels: on Wayland, points made smaller until they come to
    /// an even number of pixels; on X11, the pixels themselves
    #[test]
    fn a_window_opens_on_an_even_number_of_pixels_at_fractional_scales() {
        const ROOM: f64 = 48.0;
        let modes = [
            ((1920, 1080), 1),
            ((2560, 1600), 1),
            ((3840, 2160), 2),
            ((1366, 768), 1),
            ((1280, 800), 1),
        ];
        for scale in [1.25_f32, 1.5, 1.75, 2.1] {
            // A 3840 × 2160 screen: its points on Wayland, its pixels on X11.
            let points = (
                (3840.0 / f64::from(scale)).round() as i32,
                (2160.0 / f64::from(scale)).round() as i32,
            );
            let wayland = Units::new(scale, 1.0);
            let x11 = Units::new(1.0, scale);
            for (mode, guest_scale) in modes {
                for (units, usable) in [(wayland, points), (x11, (3840, 2160))] {
                    let size = windowed_size(mode, guest_scale, Some(usable), ROOM, units);
                    let pixels = (units.pixels_of(size.0), units.pixels_of(size.1));
                    assert!(
                        pixels.0 % 2 == 0 && pixels.1 % 2 == 0,
                        "{mode:?} at {scale}: {size:?}, {pixels:?} pixels"
                    );
                }
            }
        }
        // Wayland at 2.1: 1570 points are 3297 pixels, and 1564 the next size down that is even.
        let wayland_21 = Units::new(2.1, 1.0);
        let size = windowed_size((2560, 1600), 1, Some((1829, 1029)), ROOM, wayland_21);
        assert_eq!(size, (1564, 981));
        assert_eq!(
            (wayland_21.pixels_of(size.0), wayland_21.pixels_of(size.1)),
            (3284, 2060)
        );
        // At 1.5: 2227 points are 3341 pixels (3340.5 rounded away from zero), 2225 are 3338.
        let wayland_15 = Units::new(1.5, 1.0);
        assert_eq!(
            windowed_size((2560, 1600), 1, Some((2560, 1440)), ROOM, wayland_15),
            (2225, 1392)
        );
        // X11 at 2.1: 3660 × 2059 pixels, one fewer high.
        let x11_21 = Units::new(1.0, 2.1);
        assert_eq!(
            windowed_size((1920, 1080), 1, Some((3840, 2160)), ROOM, x11_21),
            (3660, 2058)
        );
        // The ten sizes from 965 to 974 points are odd at 2.1 (2027 to 2045 pixels).
        assert_eq!(wayland_21.even(974, 200), 964);
        assert_eq!(wayland_21.pixels_of(964), 2024);
    }
}
