//! The device's display as the guest drives it, and [`Screen`], what shows it on the host: the
//! window of `reims-vgpu`'s glue, or a fixed mode with nothing shown ([`FixedScreen`]).
//!
//! **Modes.** The display offers one mode: the screen's size and refresh rate ([`Screen::mode`]),
//! which for a window is its size in pixels at its host screen's rate, with the host screen's
//! scale, which a guest that took `DISPLAY_SCALE` is told after the display's record
//! (`RgfxDisplayScale`). The guest runs at the host screen's rate. When the screen's mode changes
//! (the window resized, or moved to a screen of another scale), the device tells the guest with
//! `DISPLAY_CHANGED`, and the guest asks `GET_DISPLAYS` again (`refresh`).
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
//!
//! **The panel.** `SET_DISPLAY_POWER` turns the display's panel off while the guest's display
//! sleeps and on at its wake ([`Screen::power`]): while it is off, the screen shows black and no
//! cursor, and everything else goes on (the scanout, presents and their answers, the cursor's
//! commands, `VBLANK`), so the screen shows what the guest set meanwhile once it is on. A reset
//! turns it on.
//!
//! **Several displays.** A device has 1 to `RGFX_DISPLAYS_MAX` displays, fixed when it is made,
//! each with a [`Screen`] of its own (docs/graphics-kext/design.md, "Multiple displays"). Each
//! display's scanout lies in a room of its own: the scanout's room split evenly among them
//! ([`scanout_room`], [`scanout_room_at`]), so one display keeps the whole room, as before any
//! display had a room of its own. On a device of more than one display each display offers only
//! a mode whose scanout fits its room ([`offered_in_room`]); with one, the screen's mode as it is.

use crate::wire::{
    RGFX_ALLOCATIONS_OFFSET, RGFX_SCANOUT_BYTES_PER_PIXEL, RGFX_SCANOUT_ROOM_ALIGN,
    RGFX_SCANOUT_SPAN, RGFX_SCANOUT_STRIDE_ALIGN,
};

/// The bytes of each display's scanout room on a device of `count` displays:
/// `RGFX_SCANOUT_SPAN / count`, rounded down to a multiple of `RGFX_SCANOUT_ROOM_ALIGN`. The whole
/// span for one display (and for a `count` of 0, which no device has).
pub fn scanout_room(count: u32) -> u64 {
    let room = RGFX_SCANOUT_SPAN / u64::from(count.max(1));
    room - room % RGFX_SCANOUT_ROOM_ALIGN
}

/// Where display `display`'s scanout room starts in the region, on a device of `count`
/// displays: `RGFX_ALLOCATIONS_OFFSET` for display 0, and each next display's a room on.
pub fn scanout_room_at(display: u32, count: u32) -> u64 {
    RGFX_ALLOCATIONS_OFFSET + u64::from(display) * scanout_room(count)
}

/// The bytes a scanout of `width` x `height` takes in a room, laid out as the guest's kexts and
/// the option ROM lay theirs: `height` rows of `width` pixels, each row rounded up to a multiple
/// of `RGFX_SCANOUT_STRIDE_ALIGN` bytes.
fn laid_out(width: u32, height: u32) -> u64 {
    let row = u64::from(width) * u64::from(RGFX_SCANOUT_BYTES_PER_PIXEL);
    row.next_multiple_of(u64::from(RGFX_SCANOUT_STRIDE_ALIGN)) * u64::from(height)
}

/// The mode a display whose screen offers `mode` offers its guest on a device of `count`
/// displays: `mode` itself on a device of one, or where its scanout fits a display's room
/// ([`scanout_room`], rows laid out as [`laid_out`] says); otherwise the largest mode of
/// `mode`'s ratio whose scanout fits, a multiple of 16 pixels wide and an even number high (the
/// steps of a window's offered modes, `reims-vgpu-window`'s `geometry::offered_mode`), at
/// `mode`'s rate and scale. Four displays take up to 3840 x 2160 each, two up to 5120 x 2880.
pub fn offered_in_room(mode: Mode, count: u32) -> Mode {
    let room = scanout_room(count);
    if count <= 1 || laid_out(mode.width, mode.height) <= room {
        return mode;
    }
    let (w, h) = (u64::from(mode.width.max(1)), u64::from(mode.height.max(1)));
    // The height of a mode `width` wide at `mode`'s ratio, rounded down to even.
    let height_of = |width: u64| (width * h / w) & !1;
    let fits = |width: u64| {
        let height = height_of(width);
        let (Ok(width), Ok(height)) = (u32::try_from(width), u32::try_from(height)) else {
            return false;
        };
        laid_out(width, height) <= room
    };
    // Every width from `mode`'s down, in steps of 16: the first that fits is the largest.
    let mut width = w & !15;
    while width > 16 && !fits(width) {
        width -= 16;
    }
    let width = width.max(16);
    let clamp = |v: u64| u32::try_from(v).unwrap_or(u32::MAX);
    Mode {
        width: clamp(width),
        height: clamp(height_of(width).max(2)),
        ..mode
    }
}

/// A display mode: pixels each way and a refresh rate, and the scale of the host screen that
/// shows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mode {
    /// Pixels a row.
    pub width: u32,
    /// Rows.
    pub height: u32,
    /// The refresh rate, in thousandths of a hertz.
    pub refresh_mhz: u32,
    /// The host screen's pixels a point, in 120ths (`RgfxDisplayScale`'s `scale_120ths`): 120 at
    /// 1x, 240 at 2x; 0 where no screen shows the display.
    pub scale_120ths: u32,
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
    /// screen's refresh rate, with that screen's scale. Asked at `GET_DISPLAYS`, and at each
    /// refresh, where a change is told to the guest.
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

    /// The display's panel went off (`on` false: the guest's display sleeps) or on again
    /// (`SET_DISPLAY_POWER`, or a reset of a device whose panel was off). While it is off the
    /// screen shows black, with no cursor, whatever the guest sets; on, it shows what the guest
    /// set, and its cursor, again.
    fn power(&self, on: bool);
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

    fn power(&self, _on: bool) {}
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
            scale_120ths: 120,
        };
        assert_eq!(mode(120_000).period_ns(), 8_333_333);
        assert_eq!(mode(0).period_ns(), 16_666_666);
        assert_eq!(mode(59_940).period_ns(), 16_683_350);
    }

    /// The scanout's room is split evenly, in 64 KiB steps: 128 MiB for one display, 64 MiB each
    /// for two, 0x2aa_0000 bytes each for three and 32 MiB each for four, display 0's where the
    /// one display's always was
    #[test]
    fn the_scanout_s_room_is_split_evenly_among_the_displays() {
        assert_eq!(scanout_room(1), RGFX_SCANOUT_SPAN);
        assert_eq!(scanout_room(2), 64 << 20);
        assert_eq!(scanout_room(3), 0x2aa_0000);
        assert_eq!(scanout_room(4), 32 << 20);
        for count in 1..=crate::wire::RGFX_DISPLAYS_MAX {
            assert_eq!(scanout_room_at(0, count), RGFX_ALLOCATIONS_OFFSET);
            let last = scanout_room_at(count - 1, count) + scanout_room(count);
            assert!(last <= crate::wire::RGFX_CONTEXT_ALLOCATIONS_OFFSET);
        }
        assert_eq!(scanout_room_at(3, 4), RGFX_ALLOCATIONS_OFFSET + (96 << 20));
    }

    /// One display offers its screen's mode, however large; several offer each the largest of
    /// its ratio that fits a room, 16 pixels wide and 2 high a step, at its rate and scale
    #[test]
    fn a_display_offers_the_largest_mode_of_its_ratio_its_room_holds() {
        let mode = |width, height| Mode {
            width,
            height,
            refresh_mhz: 120_000,
            scale_120ths: 240,
        };
        // One display: as it is, 8K included.
        assert_eq!(offered_in_room(mode(7680, 4320), 1), mode(7680, 4320));
        // What fits stays: four take up to 3840 x 2160 (31.6 MiB of 32), two 5120 x 2880.
        assert_eq!(offered_in_room(mode(3840, 2160), 4), mode(3840, 2160));
        assert_eq!(offered_in_room(mode(5120, 2880), 2), mode(5120, 2880));
        assert_eq!(offered_in_room(mode(1920, 1080), 4), mode(1920, 1080));
        // Too large: the largest of its ratio that fits.
        assert_eq!(offered_in_room(mode(7680, 4320), 4), mode(3840, 2160));
        assert_eq!(offered_in_room(mode(7680, 4320), 2), mode(5440, 3060));
        assert_eq!(offered_in_room(mode(5120, 2880), 4), mode(3840, 2160));
        // Each a step of the window's modes, and inside its room.
        for (w, h) in [(6016, 3384), (5120, 1440), (4096, 4096), (7680, 2160)] {
            for count in 2..=4 {
                let offered = offered_in_room(mode(w, h), count);
                assert_eq!(offered.width % 16, 0, "{w}x{h} on {count}");
                assert_eq!(offered.height % 2, 0, "{w}x{h} on {count}");
                assert!(laid_out(offered.width, offered.height) <= scanout_room(count));
                assert!(offered.width <= w && offered.height <= h);
                if offered == mode(w, h) {
                    continue;
                }
                // The next step up does not fit.
                let up = offered.width + 16;
                let up_height = (u64::from(up) * u64::from(h) / u64::from(w)) as u32 & !1;
                assert!(
                    laid_out(up, up_height) > scanout_room(count),
                    "{w}x{h} on {count}"
                );
                assert_eq!((offered.refresh_mhz, offered.scale_120ths), (120_000, 240));
            }
        }
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
