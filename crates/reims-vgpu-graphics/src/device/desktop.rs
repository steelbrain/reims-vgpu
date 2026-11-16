//! The guest's desktop as the host's captured pointer crosses it (docs/graphics-kext/design.md,
//! "Multiple displays", "A drag across displays"): each display's rectangle in WindowServer's
//! global space, in points, as the guest's `DISPLAY_BOUNDS` gave it, beside the display's
//! scanout, whose pixels lie over that rectangle as the kext places `POINTER_OVER`'s point
//! (`RgfxPointerOver.hpp`). A point of the desktop goes onto the display whose rectangle holds
//! it; one no rectangle holds is held on the display it came from, as a Mac holds its cursor
//! where no display meets the edge.

use crate::wire::RgfxDisplayBounds;

/// A display's rectangle in the guest's desktop, in points: from `min` up to `max`, the corner
/// past it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Bounds {
    /// The left and top edges.
    pub(crate) min: (f64, f64),
    /// The right and bottom edges, past the last column and row.
    pub(crate) max: (f64, f64),
}

impl Bounds {
    /// The rectangle a `DISPLAY_BOUNDS` gives, where it has an area.
    pub(crate) fn of(command: &RgfxDisplayBounds) -> Option<Bounds> {
        (command.max_x > command.min_x && command.max_y > command.min_y).then(|| Bounds {
            min: (f64::from(command.min_x), f64::from(command.min_y)),
            max: (f64::from(command.max_x), f64::from(command.max_y)),
        })
    }

    /// Whether `point` lies in it: on its left and top edges, short of its right and bottom.
    fn holds(&self, point: (f64, f64)) -> bool {
        (self.min.0..self.max.0).contains(&point.0) && (self.min.1..self.max.1).contains(&point.1)
    }

    /// Its size in points.
    fn size(&self) -> (f64, f64) {
        (self.max.0 - self.min.0, self.max.1 - self.min.1)
    }
}

/// Where a display lies in the desktop and what it shows: its rectangle, and its scanout's
/// pixels, which lie over it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Place {
    /// Its rectangle.
    pub(crate) bounds: Bounds,
    /// Its scanout's width and height, neither 0.
    pub(crate) pixels: (u32, u32),
}

impl Place {
    /// The desktop's points a pixel of the scanout spans, across and down.
    pub(crate) fn points_per_pixel(&self) -> (f64, f64) {
        let size = self.bounds.size();
        (
            size.0 / f64::from(self.pixels.0),
            size.1 / f64::from(self.pixels.1),
        )
    }

    /// The desktop's point at `pixel`, a place over the scanout's pixels (a pixel's centre is its
    /// column and row plus a half).
    pub(crate) fn point_of(&self, pixel: (f64, f64)) -> (f64, f64) {
        let per = self.points_per_pixel();
        (
            self.bounds.min.0 + pixel.0 * per.0,
            self.bounds.min.1 + pixel.1 * per.1,
        )
    }

    /// The scanout's pixel under the desktop's `point`, held inside the scanout: the pixel the
    /// guest's cursor is at there.
    pub(crate) fn pixel_of(&self, point: (f64, f64)) -> (u32, u32) {
        let per = self.points_per_pixel();
        let along = |at: f64, min: f64, per: f64, pixels: u32| {
            let pixel = ((at - min) / per).floor();
            if pixel.is_nan() || pixel < 0.0 {
                0
            } else {
                (pixel as u64).min(u64::from(pixels) - 1) as u32
            }
        };
        (
            along(point.0, self.bounds.min.0, per.0, self.pixels.0),
            along(point.1, self.bounds.min.1, per.1, self.pixels.1),
        )
    }

    /// `point` held inside the rectangle, a hair short of its right and bottom edges.
    fn hold(&self, point: (f64, f64)) -> (f64, f64) {
        let inside = |at: f64, min: f64, max: f64| at.clamp(min, (max - HAIR).max(min));
        (
            inside(point.0, self.bounds.min.0, self.bounds.max.0),
            inside(point.1, self.bounds.min.1, self.bounds.max.1),
        )
    }
}

/// How far short of a rectangle's right and bottom edges a held point stays, in points: inside,
/// on the last pixel's column or row at any density a mode has.
const HAIR: f64 = 1.0 / 1024.0;

/// The display whose rectangle holds `point`, among `places` (display `i`'s at `i`, `None` where
/// the guest gave none or it has no scanout), `first` looked at before the others.
pub(crate) fn display_at(
    places: &[Option<Place>],
    first: usize,
    point: (f64, f64),
) -> Option<usize> {
    let holds = |i: usize| {
        places
            .get(i)
            .copied()
            .flatten()
            .is_some_and(|p| p.bounds.holds(point))
    };
    if holds(first) {
        return Some(first);
    }
    (0..places.len()).find(|&i| holds(i))
}

/// Where the pointer, at `from` on display `on`, goes when it moves to `to`: onto the display
/// that holds `to`; where none does, onto another display that holds the move across alone
/// (`to`'s column, `from`'s row) or down alone (`from`'s column, `to`'s row), as the pointer
/// slides along the edge it met; else held inside display `on`, at its edge. The display and the
/// point; `None` where display `on` has no place.
pub(crate) fn step(
    places: &[Option<Place>],
    on: usize,
    from: (f64, f64),
    to: (f64, f64),
) -> Option<(usize, (f64, f64))> {
    let current = places.get(on).copied().flatten()?;
    if let Some(display) = display_at(places, on, to) {
        return Some((display, to));
    }
    for point in [(to.0, from.1), (from.0, to.1)] {
        if let Some(display) = display_at(places, on, point).filter(|&d| d != on) {
            return Some((display, point));
        }
    }
    Some((on, current.hold(to)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A display at (`x`, `y`) of `w` by `h` points, `scale` pixels a point.
    fn place(x: f64, y: f64, w: f64, h: f64, scale: u32) -> Option<Place> {
        Some(Place {
            bounds: Bounds {
                min: (x, y),
                max: (x + w, y + h),
            },
            pixels: (w as u32 * scale, h as u32 * scale),
        })
    }

    /// A pixel's centre maps to the point it covers, and back, at 1x and at 2x
    #[test]
    fn a_pixel_and_its_point_map_to_each_other() {
        let one = place(1920.0, 0.0, 1920.0, 1080.0, 1).unwrap();
        assert_eq!(one.point_of((0.5, 10.5)), (1920.5, 10.5));
        assert_eq!(one.pixel_of((1920.5, 10.5)), (0, 10));
        let two = place(-1280.0, 0.0, 1280.0, 800.0, 2).unwrap();
        assert_eq!(two.point_of((2559.5, 0.5)), (-0.25, 0.25));
        assert_eq!(two.pixel_of((-0.25, 0.25)), (2559, 0));
        // Held inside the scanout.
        assert_eq!(two.pixel_of((5.0, -3.0)), (2559, 0));
    }

    /// A point goes onto the display that holds it; past an edge no display meets, it slides
    /// along the edge or is held on the display it was on
    #[test]
    fn a_point_goes_onto_the_display_that_holds_it() {
        // Display 1 right of display 0, its top 200 points lower.
        let places = [
            place(0.0, 0.0, 1920.0, 1080.0, 1),
            place(1920.0, 200.0, 1920.0, 1080.0, 1),
        ];
        assert_eq!(
            step(&places, 0, (1919.5, 500.0), (1921.0, 500.0)),
            Some((1, (1921.0, 500.0)))
        );
        // Above display 1's top: held at display 0's right edge.
        let (on, at) = step(&places, 0, (1919.5, 100.0), (1925.0, 100.0)).unwrap();
        assert_eq!(on, 0);
        assert!(at.0 < 1920.0 && at.0 > 1919.99 && at.1 == 100.0, "{at:?}");
        // Up past display 1's top edge, where nothing is: slides along it.
        assert_eq!(
            step(&places, 1, (1925.0, 210.0), (1930.0, 190.0)),
            Some((1, (1930.0, 200.0)))
        );
        // Right and down from display 0 past the bottom of a shorter display 1, into nothing:
        // the move across alone lands on display 1.
        let short = [places[0], place(1920.0, 0.0, 1920.0, 600.0, 1)];
        assert_eq!(
            step(&short, 0, (1919.0, 599.0), (1921.0, 601.0)),
            Some((1, (1921.0, 599.0)))
        );
        // Back left across the edge.
        assert_eq!(
            step(&places, 1, (1920.5, 500.0), (1919.0, 500.0)),
            Some((0, (1919.0, 500.0)))
        );
        // A display the guest gave no rectangle for holds nothing.
        assert_eq!(step(&[None, places[1]], 0, (0.0, 0.0), (1.0, 1.0)), None);
    }
}
