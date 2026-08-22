//! Presentation geometry shared by native output and host input adapters.

/// Where a source frame lands inside a destination drawable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PresentationViewport {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl PresentationViewport {
    pub fn covers(self, dst: (u32, u32)) -> bool {
        self.x == 0 && self.y == 0 && self.width == dst.0 && self.height == dst.1
    }
}

/// Largest centered source-aspect rectangle inside `dst`.
///
/// Zero dimensions are clamped to one so a transition frame cannot divide by
/// zero or produce an empty native blit.
pub fn aspect_fit_viewport(src: (u32, u32), dst: (u32, u32)) -> PresentationViewport {
    let (sw, sh) = (u64::from(src.0.max(1)), u64::from(src.1.max(1)));
    let (dw, dh) = (u64::from(dst.0.max(1)), u64::from(dst.1.max(1)));
    let (width, height) = if sw * dh >= sh * dw {
        (dw, (sh * dw / sw).max(1))
    } else {
        ((sw * dh / sh).max(1), dh)
    };
    PresentationViewport {
        x: ((dw - width) / 2) as u32,
        y: ((dh - height) / 2) as u32,
        width: width as u32,
        height: height as u32,
    }
}

/// Map a window-space pointer through the presentation viewport into guest
/// pixels. Letterbox positions clamp to the nearest visible guest edge.
pub fn pointer_to_guest(pos: (f64, f64), window: (u32, u32), guest: (u32, u32)) -> (u32, u32) {
    let viewport = aspect_fit_viewport(guest, window);
    let x = (pos.0 - f64::from(viewport.x)) * f64::from(guest.0.max(1))
        / f64::from(viewport.width.max(1));
    let y = (pos.1 - f64::from(viewport.y)) * f64::from(guest.1.max(1))
        / f64::from(viewport.height.max(1));
    (
        (x.max(0.0) as u32).min(guest.0.saturating_sub(1)),
        (y.max(0.0) as u32).min(guest.1.saturating_sub(1)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aspect_fit_and_pointer_inverse_share_one_transform() {
        let viewport = aspect_fit_viewport((1440, 1080), (1920, 1080));
        assert_eq!(
            viewport,
            PresentationViewport {
                x: 240,
                y: 0,
                width: 1440,
                height: 1080,
            }
        );
        assert!(!viewport.covers((1920, 1080)));
        assert_eq!(
            pointer_to_guest((960.0, 540.0), (1920, 1080), (1440, 1080)),
            (720, 540)
        );
        assert_eq!(
            pointer_to_guest((10.0, 540.0), (1920, 1080), (1440, 1080)),
            (0, 540)
        );
    }

    #[test]
    fn matching_and_degenerate_extents_stay_total() {
        assert!(aspect_fit_viewport((1920, 1080), (960, 540)).covers((960, 540)));
        let viewport = aspect_fit_viewport((0, 0), (1920, 1080));
        assert!(viewport.width >= 1 && viewport.height >= 1);
        assert_eq!(pointer_to_guest((5.0, 5.0), (10, 10), (0, 0)), (0, 0));
    }
}
