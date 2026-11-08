//! The host's screens, which the OS's side lists (`native`): "screen i" is the i-th of them, the
//! main one first, and the window of the display on port i opens on screen i
//! ([`screen_for_port`]). On macOS they come from CoreGraphics, which answers on any thread; on
//! Linux from SDL, on the library's UI thread, which takes a snapshot of them
//! (`native/linux/ui.rs`). Where there are none (no window system to reach), there is no screen
//! to open on ([`screen_for_port`]).
//!
//! A screen's refresh rate is the one it runs at ([`screen_refresh`]): its current mode's, so a
//! screen the person set below its highest rate gives the guest the rate it was set to, as the
//! guest's modes run at the host screen's rate, never raised or lowered for frames.

use crate::env;
use crate::native;

/// The rate a screen runs at where the OS gives none, in Hz ([`screen_refresh`]).
const DEFAULT_REFRESH_HZ: f64 = 60.0;

/// A host screen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HostScreen {
    /// Its display ID: CoreGraphics' on macOS, SDL's on Linux.
    pub id: u32,
    /// Its current mode's size in pixels, width then height.
    pub pixels: (u32, u32),
    /// Pixels per point.
    pub scale: f64,
    /// The refresh rate it runs at, in Hz ([`screen_refresh`]).
    pub refresh_hz: f64,
}

/// The host's screens, the main one first. Empty where the OS lists none.
pub fn host_screens() -> Vec<HostScreen> {
    native::host_screens()
}

/// The index of the screen the display on `port` opens on, among `count` screens:
/// `list[port]` where the list names one that exists, else `port` where that exists, else 0,
/// the main screen.
pub fn pick_screen(port: u64, list: &[usize], count: usize) -> usize {
    let port = usize::try_from(port).ok();
    let listed = port.and_then(|port| list.get(port)).copied();
    listed
        .filter(|&index| index < count)
        .or(port.filter(|&port| port < count))
        .unwrap_or(0)
}

/// The refresh rate, in Hz, of a screen whose current mode is `pixels` at `current_hz`, among
/// `modes`, each mode's pixels and rate as the OS lists them: the current mode's rate, the rate
/// the screen runs at, whatever higher ones the screen offers at its size. A rate of 0, or one
/// that is not a number, is unknown; where the current mode's is, the highest of the modes of
/// the same pixels stands in, as the most a variable-rate screen runs at, the way ProMotion's
/// current mode reports the top of its range. A mode of other pixels does not count, as the
/// screen does not run its rate at its current size (a 4K screen listing 1920×1080 at 120 Hz runs
/// 4K at 60). [`DEFAULT_REFRESH_HZ`] where every rate is unknown.
pub(crate) fn screen_refresh(
    pixels: (u32, u32),
    current_hz: f64,
    modes: impl IntoIterator<Item = ((u32, u32), f64)>,
) -> f64 {
    let known = |hz: &f64| hz.is_finite() && *hz > 0.0;
    if known(&current_hz) {
        return current_hz;
    }
    let highest = modes
        .into_iter()
        .filter_map(|(size, hz)| (size == pixels).then_some(hz))
        .filter(known)
        .fold(0.0, f64::max);
    if highest > 0.0 {
        highest
    } else {
        DEFAULT_REFRESH_HZ
    }
}

/// The screen the window of the display on `port` opens on: the one `REIMS_VGPU_WINDOW_SCREENS`
/// names for it, else screen `port`, else the main screen. `None` where there is no screen.
pub fn screen_for_port(port: u64) -> Option<HostScreen> {
    let screens = host_screens();
    let index = pick_screen(port, &env::settings().screens, screens.len());
    screens.get(index).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Display i opens on screen i when there is one
    #[test]
    fn display_i_opens_on_screen_i() {
        assert_eq!(pick_screen(0, &[], 3), 0);
        assert_eq!(pick_screen(1, &[], 3), 1);
        assert_eq!(pick_screen(2, &[], 3), 2);
    }

    /// Display i opens on the main screen when there is no screen i
    #[test]
    fn a_missing_screen_is_the_main_one() {
        assert_eq!(pick_screen(1, &[], 1), 0);
        assert_eq!(pick_screen(7, &[], 2), 0);
        assert_eq!(pick_screen(u64::MAX, &[], 2), 0);
        assert_eq!(pick_screen(0, &[], 0), 0);
    }

    /// The list names each display's screen; an entry naming no screen falls back to screen i,
    /// then the main one
    #[test]
    fn the_list_names_each_display_its_screen() {
        assert_eq!(pick_screen(0, &[1, 0], 2), 1);
        assert_eq!(pick_screen(1, &[1, 0], 2), 0);
        // The list is shorter than the ports: screen i.
        assert_eq!(pick_screen(1, &[1], 2), 1);
        // The entry names a screen that is not there: screen i, else the main one.
        assert_eq!(pick_screen(1, &[0, 5], 2), 1);
        assert_eq!(pick_screen(2, &[0, 0, 5], 2), 0);
    }

    /// A screen runs at its current mode's rate, a screen set below its highest rate included;
    /// where that is unknown, at the highest of the modes of its own pixels, never that of a
    /// mode of other pixels; with none known, 60 Hz
    #[test]
    fn a_screen_runs_at_its_current_mode_s_rate() {
        const UHD: (u32, u32) = (3840, 2160);
        const FHD: (u32, u32) = (1920, 1080);
        // ProMotion, as the M3 Max's screen lists its modes (macOS 27.0.1): set to ProMotion its
        // current mode is the 120 Hz one; set to 60 Hz, the guest gets 60, not 120.
        let promotion = (3456, 2234);
        let rates = [120.0, 60.0, 59.94, 50.0, 48.0, 47.95];
        let modes = rates.map(|hz| (promotion, hz));
        assert_eq!(screen_refresh(promotion, 120.0, modes), 120.0);
        assert_eq!(screen_refresh(promotion, 60.0, modes), 60.0);
        assert_eq!(screen_refresh(promotion, 59.94, modes), 59.94);
        // A current mode of no rate: the highest of its size.
        assert_eq!(screen_refresh(promotion, 0.0, modes), 120.0);
        // A 4K screen at 60 Hz that lists 1920×1080 at 120 and 144 Hz runs 4K at 60, and with
        // its rate unknown still not at the rates of other sizes.
        let modes = [(UHD, 60.0), (UHD, 30.0), (FHD, 120.0), (FHD, 144.0)];
        assert_eq!(screen_refresh(UHD, 60.0, modes), 60.0);
        assert_eq!(screen_refresh(UHD, 0.0, modes), 60.0);
        // Unknown rates: 0, and what is not a number, count for nothing.
        let unknown = [(UHD, 0.0), (UHD, f64::NAN), (FHD, 75.0)];
        assert_eq!(screen_refresh(UHD, f64::INFINITY, unknown), 60.0);
        assert_eq!(screen_refresh(UHD, 0.0, []), 60.0);
    }
}
