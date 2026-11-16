//! A captured pointer past the image's edge, onto the guest's other displays: what a window
//! decides of it, with no OS in it (the SDL window's input, `native/sdl_thread/input.rs`, drives
//! it on Linux and Windows).
//!
//! Captured, the host's pointer is held over the image (`capture`). Where the guest has several
//! displays and its desktop puts another display past an edge the pointer moves toward, within
//! [`REACH`] of it ([`PointerBeyond::adjoins`]), the pointer leaves: the window holds the host's
//! pointer still where it is ([`Hold::Lock`]: the OS's pointer lock, which hides it and reports
//! its moves as motion alone) and hands each move, in its display's pixels, to the guest's
//! desktop ([`PointerBeyond::moved`]), which sends the guest the pointer wherever it lands, on
//! whichever display, through that display's own window: the guest's cursor crosses onto the
//! other display as a Mac's does, and a drag goes on there. Where it lands back on the window's
//! own display, out of reach of an edge another display meets, the pointer is the window's again
//! ([`Hold::Unlock`]: the host's pointer let go there, over the image where the guest's cursor
//! is). Letting go of capture (the release chord, the keyboard lost, the window's end) lets go of
//! it too, the host's pointer where it was held; the guest's cursor stays where it is.
//!
//! The pointer leaves within [`REACH`] of the edge, not only at it, as a compositor may hold a
//! confined pointer short of the edge (KWin, [`REACH`]); the guest's cursor goes on from where it
//! is, so it crosses at the edge all the same.

use crate::config::{Landed, PointerBeyond};
use crate::geometry::{self, Rect};

/// What the window does with the host's pointer as it leaves and comes back.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Hold {
    /// Holds it still where it is, hidden, its moves reported as motion alone, as the desktop
    /// accelerates them.
    Lock,
    /// Holds it still where it is, hidden, its moves reported as the device's own counts, as no
    /// acceleration has changed them: the pointer for games ([`crate::game`]).
    Raw,
    /// Lets it go at the window's point (in points), over the image.
    Unlock((f64, f64)),
}

/// A pointer come back to the window ([`Beyond::moved`], [`Beyond::let_go`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Back {
    /// Where the host's pointer goes: [`Hold::Unlock`]'s point.
    pub point: (f64, f64),
    /// The guest pixel there, sent already where the pointer came back over the window's display.
    pub at: (u16, u16),
}

/// The pointer gone past the edge.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Away {
    /// Where the host's pointer was held as it left, in the window's points.
    held: (f64, f64),
    /// The guest pixel there.
    at: (u16, u16),
    /// The guest's mode it left in.
    mode: (u16, u16),
}

/// Whether the window's captured pointer has gone past the image's edge (the module doc).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Beyond {
    /// Where it left, while it is away.
    away: Option<Away>,
}

/// How near an edge another display meets, in the window's points, the captured pointer moving
/// toward it leaves for that display. Not a point: a compositor may hold a confined pointer short
/// of the edge. KWin keeps it where it was when a move would take it out of the region, rather
/// than sliding it to the region's edge, so a push stops short by up to one move's length:
/// measured on KWin on Wayland at a scale of 1.9 (lane dragcross, potluck job m3-max-35056), a
/// mouse pushed left at about 5 points a move stopped 2.6 points short of the image's left edge
/// and stayed there, and a pointer that had to come within a point of the edge never left. A fast
/// hand's moves are tens of points.
pub const REACH: f64 = 32.0;

/// The guest pixels [`REACH`] points of `image` span of `mode`, across and down, rounded up, at
/// least 1 and at most the mode's: the pointer within that many of an edge's last column or row
/// is within reach of the edge.
pub fn reach(image: Rect, mode: (u16, u16)) -> (u16, u16) {
    let along = |pixels: u16, points: f64| {
        let per = f64::from(pixels) / points;
        if per.is_finite() && per > 0.0 {
            (REACH * per).ceil().clamp(1.0, f64::from(pixels.max(1))) as u16
        } else {
            1
        }
    };
    (along(mode.0, image.width), along(mode.1, image.height))
}

/// Whether guest pixel `at` of `mode` lies within `reach` of an edge another display meets, where
/// `moving` (the pixels the pointer moved, across and down) is toward it, or with `moving` `None`
/// in any case: within `reach` of an edge's last column or row, and `beyond` puts a display past
/// it (across, down, or past the corner where `at` is within reach of two edges).
pub fn at_shared_edge(
    beyond: &dyn PointerBeyond,
    at: (u16, u16),
    mode: (u16, u16),
    reach: (u16, u16),
    moving: Option<(i32, i32)>,
) -> bool {
    let toward = |at: u16, pixels: u16, reach: u16, moving: Option<i32>| -> i8 {
        if at < reach && moving.is_none_or(|by| by < 0) {
            -1
        } else if u32::from(at) + u32::from(reach) >= u32::from(pixels)
            && moving.is_none_or(|by| by > 0)
        {
            1
        } else {
            0
        }
    };
    let (x, y) = (
        toward(at.0, mode.0, reach.0, moving.map(|by| by.0)),
        toward(at.1, mode.1, reach.1, moving.map(|by| by.1)),
    );
    let corner = if x != 0 && y != 0 { (x, y) } else { (0, 0) };
    [(x, 0), (0, y), corner]
        .into_iter()
        .filter(|&outward| outward != (0, 0))
        .any(|outward| beyond.adjoins(at, mode, outward))
}

/// The window's point at the centre of guest pixel `at` of `mode`, over `image`.
pub(crate) fn point_of(at: (u16, u16), image: Rect, mode: (u16, u16)) -> (f64, f64) {
    (
        image.x + (f64::from(at.0) + 0.5) * image.width / f64::from(mode.0.max(1)),
        image.y + (f64::from(at.1) + 0.5) * image.height / f64::from(mode.1.max(1)),
    )
}

impl Beyond {
    /// Not away.
    pub fn new() -> Beyond {
        Beyond::default()
    }

    /// Whether the pointer has gone past the edge: its moves go to [`Self::moved`].
    pub fn is_away(&self) -> bool {
        self.away.is_some()
    }

    /// The captured pointer moved to `point`, in the window's points, over `image` showing
    /// `mode`, with `buttons` held, from guest pixel `from`, the last the window sent: where it
    /// moved toward an edge another display meets and is within reach of it, it leaves for the
    /// guest's desktop, which is sent its place there, and the window holds the host's pointer
    /// ([`Hold::Lock`]). `None` where it stays the window's, for the window to send, and where the
    /// window sent no pixel yet.
    pub fn reached(
        &mut self,
        beyond: &dyn PointerBeyond,
        point: (f64, f64),
        image: Rect,
        mode: (u16, u16),
        buttons: u32,
        from: Option<(u16, u16)>,
    ) -> Option<Hold> {
        if self.away.is_some() || mode.0 == 0 || mode.1 == 0 {
            return None;
        }
        let from = from?;
        let at = geometry::pointer_to_guest(point, image, mode);
        let moving = (
            i32::from(at.0) - i32::from(from.0),
            i32::from(at.1) - i32::from(from.1),
        );
        if !at_shared_edge(beyond, at, mode, reach(image, mode), Some(moving))
            || !beyond.leave(at, mode)
        {
            return None;
        }
        self.away = Some(Away {
            held: point,
            at,
            mode,
        });
        beyond.moved((0.0, 0.0), buttons);
        Some(Hold::Lock)
    }

    /// The pointer away moved by `by`, in the window's points, over `image`, with `buttons`
    /// held: the guest's desktop takes it on. Where it comes back to the window: over the
    /// window's display, out of reach of any edge another display meets (the host's pointer let
    /// go there), or where the desktop let it go (the host's pointer where it was held). `None`
    /// while it stays away, or was not.
    pub fn moved(
        &mut self,
        beyond: &dyn PointerBeyond,
        by: (f64, f64),
        image: Rect,
        buttons: u32,
    ) -> Option<Back> {
        let away = self.away?;
        let mode = away.mode;
        let per = |pixels: u16, points: f64| {
            let per = f64::from(pixels) / points;
            if per.is_finite() { per } else { 0.0 }
        };
        let by = (
            by.0 * per(mode.0, image.width),
            by.1 * per(mode.1, image.height),
        );
        match beyond.moved(by, buttons) {
            Landed::Here { at, mode: of } => {
                let at = rescale(at, of, mode);
                if at_shared_edge(beyond, at, mode, reach(image, mode), None) {
                    return None;
                }
                self.away = None;
                beyond.come_back();
                Some(Back {
                    point: point_of(at, image, mode),
                    at,
                })
            }
            Landed::Elsewhere => None,
            Landed::Lost => {
                self.away = None;
                Some(Back {
                    point: away.held,
                    at: away.at,
                })
            }
        }
    }

    /// Where the host's pointer was held when it went past the edge, while it is away.
    pub fn held(&self) -> Option<(f64, f64)> {
        self.away.map(|away| away.held)
    }

    /// A button went down or up while away: the pointer's new `buttons`, where it is.
    pub fn buttons(&self, beyond: &dyn PointerBeyond, buttons: u32) {
        if self.away.is_some() {
            beyond.moved((0.0, 0.0), buttons);
        }
    }

    /// Capture let go while away (the release chord, the keyboard lost, the window's end): the
    /// pointer is the window's again, the host's pointer where it was held, and the guest's
    /// cursor stays where it is. `None` where it was not away.
    pub fn let_go(&mut self, beyond: Option<&dyn PointerBeyond>) -> Option<Back> {
        let away = self.away.take()?;
        if let Some(beyond) = beyond {
            beyond.come_back();
        }
        Some(Back {
            point: away.held,
            at: away.at,
        })
    }
}

/// Pixel `at` of mode `from` as a pixel of mode `to`: the same pixel where the two are one.
fn rescale(at: (u16, u16), from: (u16, u16), to: (u16, u16)) -> (u16, u16) {
    let along = |at: u16, from: u16, to: u16| {
        if from == to || from == 0 || to == 0 {
            at.min(to.saturating_sub(1))
        } else {
            ((u32::from(at) * u32::from(to)) / u32::from(from)).min(u32::from(to) - 1) as u16
        }
    };
    (along(at.0, from.0, to.0), along(at.1, from.1, to.1))
}

#[cfg(test)]
mod tests {
    use reims_vgpu_vulkan::core::sync::Mutex;

    use super::*;
    use crate::config::BUTTON_LEFT;

    /// A guest desktop of two displays side by side in one mode, display 1 right of display 0,
    /// the window display 0's, recording what it is told and moving a point across them in that
    /// mode's pixels.
    struct TwoSideBySide {
        /// Its mode.
        mode: (u16, u16),
        /// The pointer's place across both displays, while away.
        at: Mutex<Option<(f64, f64)>>,
        /// What it was told, in order.
        told: Mutex<Vec<String>>,
    }

    impl TwoSideBySide {
        fn new(mode: (u16, u16)) -> TwoSideBySide {
            TwoSideBySide {
                mode,
                at: Mutex::new(None),
                told: Mutex::new(Vec::new()),
            }
        }

        fn told(&self) -> Vec<String> {
            std::mem::take(&mut *self.told.lock())
        }
    }

    impl PointerBeyond for TwoSideBySide {
        fn adjoins(&self, _at: (u16, u16), mode: (u16, u16), outward: (i8, i8)) -> bool {
            assert_eq!(mode, self.mode);
            outward == (1, 0)
        }

        fn leave(&self, at: (u16, u16), _mode: (u16, u16)) -> bool {
            self.told.lock().push(format!("leave {at:?}"));
            *self.at.lock() = Some((f64::from(at.0) + 0.5, f64::from(at.1) + 0.5));
            true
        }

        fn moved(&self, by: (f64, f64), buttons: u32) -> Landed {
            let mut at = self.at.lock();
            let Some(point) = at.as_mut() else {
                return Landed::Lost;
            };
            let width = f64::from(self.mode.0);
            point.0 = (point.0 + by.0).clamp(0.0, 2.0 * width - 0.001);
            point.1 = (point.1 + by.1).clamp(0.0, f64::from(self.mode.1) - 0.001);
            let (x, y) = (point.0.floor() as u16, point.1.floor() as u16);
            if point.0 < width {
                self.told
                    .lock()
                    .push(format!("0 at {x},{y} buttons {buttons}"));
                Landed::Here {
                    at: (x, y),
                    mode: self.mode,
                }
            } else {
                let x = x - self.mode.0;
                self.told
                    .lock()
                    .push(format!("1 at {x},{y} buttons {buttons}"));
                Landed::Elsewhere
            }
        }

        fn come_back(&self) {
            self.told.lock().push("back".into());
            *self.at.lock() = None;
        }
    }

    /// A 960 x 540 image at (10, 20) showing a 1920 x 1080 mode: two pixels a point.
    const IMAGE: Rect = Rect {
        x: 10.0,
        y: 20.0,
        width: 960.0,
        height: 540.0,
    };
    const MODE: (u16, u16) = (1920, 1080);

    /// The captured pointer moving toward the image's edge where the guest's desktop puts another
    /// display, within reach of it, leaves for it, the host's pointer locked; its moves go on over
    /// the other display, a button's change with them, and back across the edge, out of reach of
    /// it, it is the window's again, the host's pointer let go where the guest's cursor is
    #[test]
    fn the_pointer_near_a_shared_edge_goes_on_onto_the_other_display() {
        let desktop = TwoSideBySide::new(MODE);
        let mut beyond = Beyond::new();
        // Inside the image, at its bottom edge, which no display meets, and within reach of the
        // right edge but moving away from it: the window's.
        let inside = Some((900, 560));
        assert_eq!(
            beyond.reached(&desktop, (500.0, 300.0), IMAGE, MODE, 0, inside),
            None
        );
        assert_eq!(
            beyond.reached(&desktop, (500.0, 559.9), IMAGE, MODE, 0, Some((980, 560))),
            None
        );
        assert_eq!(
            beyond.reached(&desktop, (950.0, 300.0), IMAGE, MODE, 0, Some((1900, 560))),
            None
        );
        assert_eq!(
            beyond.reached(&desktop, (960.0, 300.0), IMAGE, MODE, 0, None),
            None,
            "no pixel sent yet"
        );
        assert!(desktop.told().is_empty());
        // Within reach of the right edge, moving toward it: the pointer leaves, at its pixel.
        assert_eq!(
            beyond.reached(
                &desktop,
                (960.0, 300.0),
                IMAGE,
                MODE,
                BUTTON_LEFT,
                Some((1880, 560))
            ),
            Some(Hold::Lock)
        );
        assert!(beyond.is_away());
        assert_eq!(
            desktop.told(),
            ["leave (1900, 560)", "0 at 1900,560 buttons 1"]
        );
        // Moves in points, two pixels each: across the edge and on.
        assert_eq!(
            beyond.moved(&desktop, (10.0, 0.0), IMAGE, BUTTON_LEFT),
            None
        );
        assert_eq!(
            beyond.moved(&desktop, (50.0, 5.0), IMAGE, BUTTON_LEFT),
            None
        );
        assert_eq!(
            desktop.told(),
            ["1 at 0,560 buttons 1", "1 at 100,570 buttons 1"]
        );
        beyond.buttons(&desktop, 0);
        assert_eq!(desktop.told(), ["1 at 100,570 buttons 0"]);
        // Back over display 0 but within reach of the shared edge: still away.
        assert_eq!(beyond.moved(&desktop, (-60.0, 0.0), IMAGE, 0), None);
        assert_eq!(desktop.told(), ["0 at 1900,570 buttons 0"]);
        // Further in: the window's again, the host's pointer at that pixel's centre.
        let back = beyond.moved(&desktop, (-25.0, 0.0), IMAGE, 0).unwrap();
        assert_eq!(back.at, (1850, 570));
        assert_eq!(back.point, (935.25, 305.25));
        assert!(!beyond.is_away());
        assert_eq!(desktop.told(), ["0 at 1850,570 buttons 0", "back"]);
        assert_eq!(
            beyond.moved(&desktop, (1.0, 0.0), IMAGE, 0),
            None,
            "not away"
        );
    }

    /// Letting go of capture while away gives the window the pointer where it was held, the
    /// desktop told; a desktop that let go itself does the same; one that will not take the
    /// pointer leaves it the window's
    #[test]
    fn letting_go_while_away_gives_the_pointer_back_where_it_was_held() {
        let desktop = TwoSideBySide::new(MODE);
        let mut beyond = Beyond::new();
        assert_eq!(beyond.let_go(Some(&desktop)), None, "not away");
        let from = Some((1900, 160));
        beyond.reached(&desktop, (969.9, 100.0), IMAGE, MODE, 0, from);
        beyond.moved(&desktop, (40.0, 0.0), IMAGE, 0);
        desktop.told();
        assert_eq!(
            beyond.let_go(Some(&desktop)),
            Some(Back {
                point: (969.9, 100.0),
                at: (1919, 160)
            })
        );
        assert_eq!(desktop.told(), ["back"]);

        beyond.reached(&desktop, (969.9, 100.0), IMAGE, MODE, 0, from);
        // The desktop let go (a reset).
        *desktop.at.lock() = None;
        assert_eq!(
            beyond.moved(&desktop, (3.0, 0.0), IMAGE, 0),
            Some(Back {
                point: (969.9, 100.0),
                at: (1919, 160)
            })
        );
        assert!(!beyond.is_away());

        struct Refuses;
        impl PointerBeyond for Refuses {
            fn adjoins(&self, _: (u16, u16), _: (u16, u16), _: (i8, i8)) -> bool {
                true
            }
            fn leave(&self, _: (u16, u16), _: (u16, u16)) -> bool {
                false
            }
            fn moved(&self, _: (f64, f64), _: u32) -> Landed {
                panic!("never left")
            }
            fn come_back(&self) {}
        }
        assert_eq!(
            beyond.reached(&Refuses, (969.9, 100.0), IMAGE, MODE, 0, from),
            None
        );
        assert!(!beyond.is_away());
    }

    /// The reach is 32 points of the image's pixels at an edge: 64 at two pixels a point, 32 at
    /// one, and at least a pixel; the pointer within it moving toward an edge asks about that
    /// edge alone, and with no move given, about each edge it is within reach of and past the
    /// corner
    #[test]
    fn the_edge_is_within_reach_of_its_last_pixel() {
        assert_eq!(reach(IMAGE, MODE), (64, 64));
        let one = Rect {
            width: 1920.0,
            height: 1080.0,
            ..IMAGE
        };
        assert_eq!(reach(one, MODE), (32, 32));
        assert_eq!(reach(IMAGE, (40, 20)), (2, 2));
        let asked = Mutex::new(Vec::new());
        struct Asks<'a>(&'a Mutex<Vec<(i8, i8)>>);
        impl PointerBeyond for Asks<'_> {
            fn adjoins(&self, _: (u16, u16), _: (u16, u16), outward: (i8, i8)) -> bool {
                self.0.lock().push(outward);
                false
            }
            fn leave(&self, _: (u16, u16), _: (u16, u16)) -> bool {
                false
            }
            fn moved(&self, _: (f64, f64), _: u32) -> Landed {
                Landed::Lost
            }
            fn come_back(&self) {}
        }
        let asks = Asks(&asked);
        assert!(!at_shared_edge(&asks, (1, 1078), MODE, (64, 64), None));
        assert_eq!(*asked.lock(), [(-1, 0), (0, 1), (-1, 1)]);
        asked.lock().clear();
        assert!(!at_shared_edge(
            &asks,
            (1, 1078),
            MODE,
            (64, 64),
            Some((-3, 0))
        ));
        assert_eq!(*asked.lock(), [(-1, 0)], "moving left alone");
        asked.lock().clear();
        assert!(!at_shared_edge(
            &asks,
            (1, 1078),
            MODE,
            (64, 64),
            Some((3, -1))
        ));
        assert!(asked.lock().is_empty(), "moving away from both edges");
        assert!(!at_shared_edge(&asks, (64, 500), MODE, (64, 64), None));
        assert!(asked.lock().is_empty(), "64 pixels in is out of reach");
    }
}
