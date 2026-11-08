//! The device's display as the guest drives it, and [`Screen`], what shows it on the host: the
//! window of `reims-vgpu`'s glue, or a fixed mode with nothing shown ([`FixedScreen`]).
//!
//! **Modes.** The display offers one mode: the screen's size and refresh rate ([`Screen::mode`]),
//! which for a window is its size in pixels at its host screen's rate. The guest runs at the host
//! screen's rate. When the screen's mode changes (the window resized), the device tells the guest
//! with `DISPLAY_CHANGED`, and the guest asks `GET_DISPLAYS` again (`refresh`).
//!
//! **Scanout.** `SET_SCANOUT` names the pixels the display shows, in the scanout's room of the
//! region (`RGFX_SCANOUT_SPAN` bytes at `RGFX_ALLOCATIONS_OFFSET`, before the processes' blocks,
//! so the display never shows a process's memory); the screen reads them in place. It shows them
//! again at each refresh while the guest has a scanout ([`Screen::present`], which `refresh`
//! calls), as the guest may have written them since. A reset of the device keeps the scanout
//! (`device`'s module doc); only `SET_SCANOUT` changes it.
//!
//! **Cursor.** `SET_CURSOR` copies the guest's image, premultiplied BGRA8, out of the region,
//! and hands the screen straight-alpha RGBA ([`straight_rgba`]); `MOVE_CURSOR` tells it where the
//! cursor is and whether it shows.

/// A display mode: pixels each way and a refresh rate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mode {
    /// Pixels a row.
    pub width: u32,
    /// Rows.
    pub height: u32,
    /// The refresh rate, in thousandths of a hertz.
    pub refresh_mhz: u32,
}

impl Mode {
    /// The time between two refreshes, in nanoseconds: 60 Hz's where the rate is 0.
    pub fn period_ns(&self) -> u64 {
        let mhz = if self.refresh_mhz == 0 {
            60_000
        } else {
            self.refresh_mhz
        };
        1_000_000_000_000 / u64::from(mhz)
    }
}

/// The pixels a display shows: `height` rows of `width` BGRA8 pixels, `stride` bytes apart,
/// from `offset` in the region, every byte of them inside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scanout {
    /// Pixels a row, above 0.
    pub width: u32,
    /// Rows, above 0.
    pub height: u32,
    /// Bytes from one row's start to the next's.
    pub stride: u32,
    /// Where the first row starts in the region.
    pub offset: u64,
}

impl Scanout {
    /// The bytes from the first row's start to the last row's end: the range the guest's writes
    /// to which are a new frame.
    pub fn span(&self) -> u64 {
        u64::from(self.stride) * u64::from(self.height - 1)
            + u64::from(self.width) * u64::from(crate::wire::RGFX_SCANOUT_BYTES_PER_PIXEL)
    }
}

/// A cursor image, copied from the guest's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CursorImage {
    /// Pixels a row.
    pub width: u32,
    /// Rows.
    pub height: u32,
    /// The hot spot, in pixels from the top left.
    pub hot_spot: (u32, u32),
    /// `height` rows of `width * 4` bytes, RGBA, the alpha straight.
    pub rgba: Vec<u8>,
}

/// What shows the device's display on the host. Its methods are called on the device's threads
/// (the control thread and the refresh thread, never inside a call of the host's) and must not
/// wait for the device.
pub trait Screen: Send + Sync + 'static {
    /// The one mode the display offers now: for a window, its size in pixels at its host
    /// screen's refresh rate. Asked at `GET_DISPLAYS`, and at each refresh, where a change is
    /// told to the guest.
    fn mode(&self) -> Mode;

    /// The guest set the pixels the display shows, or blanked it (`None`). The screen shows
    /// them at once, and then at each refresh ([`Screen::present`]), across the device's resets,
    /// until the guest sets others.
    fn scanout(&self, scanout: Option<Scanout>);

    /// A refresh while the guest has a scanout: the screen shows its pixels again, as the guest
    /// may have written them since, where it shows the scanout and not something of the host's
    /// in its place.
    fn present(&self);

    /// The guest set its cursor image, or removed it (`None`).
    fn cursor_image(&self, image: Option<CursorImage>);

    /// The guest moved its cursor's hot spot to (`x`, `y`) in the display's pixels, and shows it
    /// where `visible`.
    fn cursor_moved(&self, x: i32, y: i32, visible: bool);
}

/// A screen that shows nothing, with a fixed mode: a display with no window.
#[derive(Clone, Copy, Debug)]
pub struct FixedScreen(pub Mode);

impl Screen for FixedScreen {
    fn mode(&self) -> Mode {
        self.0
    }

    fn scanout(&self, _scanout: Option<Scanout>) {}

    fn present(&self) {}

    fn cursor_image(&self, _image: Option<CursorImage>) {}

    fn cursor_moved(&self, _x: i32, _y: i32, _visible: bool) {}
}

/// The guest's premultiplied BGRA8 pixels as straight-alpha RGBA: each colour divided by the
/// alpha, rounded to nearest and kept to 255, and a pixel of alpha 0 all zeros.
pub fn straight_rgba(bgra_premultiplied: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bgra_premultiplied.len());
    for &[b, g, r, a] in bgra_premultiplied.as_chunks::<4>().0 {
        if a == 0 {
            out.extend_from_slice(&[0, 0, 0, 0]);
            continue;
        }
        let a16 = u16::from(a);
        let straight = |c: u8| {
            let v = (u16::from(c) * 255 + a16 / 2) / a16;
            u8::try_from(v.min(255)).unwrap_or(u8::MAX)
        };
        out.extend_from_slice(&[straight(r), straight(g), straight(b), a]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Premultiplied BGRA becomes straight RGBA; transparent pixels are zeros, and colours past
    /// their alpha are kept to 255
    #[test]
    fn a_premultiplied_cursor_becomes_straight_rgba() {
        let bgra = [
            10, 20, 30, 255, // opaque
            64, 32, 0, 128, // half
            9, 9, 9, 0, // transparent
            200, 0, 0, 100, // past its alpha
        ];
        assert_eq!(
            straight_rgba(&bgra),
            [30, 20, 10, 255, 0, 64, 128, 128, 0, 0, 0, 0, 0, 0, 255, 100]
        );
    }

    /// A mode's period is its rate's; a rate of 0 is 60 Hz
    #[test]
    fn a_mode_s_period_follows_its_rate() {
        let mode = |refresh_mhz| Mode {
            width: 1,
            height: 1,
            refresh_mhz,
        };
        assert_eq!(mode(120_000).period_ns(), 8_333_333);
        assert_eq!(mode(0).period_ns(), 16_666_666);
        assert_eq!(mode(59_940).period_ns(), 16_683_350);
    }

    /// A scanout's range runs from its first row's start to its last row's end
    #[test]
    fn a_scanout_s_range_ends_at_its_last_pixel() {
        let s = Scanout {
            width: 10,
            height: 3,
            stride: 64,
            offset: 0,
        };
        assert_eq!(s.span(), 64 * 2 + 40);
    }
}
