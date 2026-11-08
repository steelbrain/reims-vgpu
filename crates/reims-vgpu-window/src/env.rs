//! The `REIMS_VGPU_WINDOW_*` variables, read once per process. An unset or unparsable value is
//! the default.
//!
//! - `REIMS_VGPU_WINDOW_BORDERLESS`: `1`, `true`, `yes` or `on`, in any case, makes each
//!   window a borderless one that fills its screen. Default: titled windows.
//! - `REIMS_VGPU_WINDOW_SCREENS`: a comma-separated list of host screen indices (spaces
//!   allowed): display i opens on the screen the list's i-th entry names. One bad entry unsets
//!   the whole list. Default: display i on screen i (`screens::pick_screen`).
//! - `REIMS_VGPU_WINDOW_IMPL`: on macOS, `sdl`, in any case, opens SDL3's window (`sdl`) in
//!   place of AppKit's, in a build with the `macos-sdl` feature, on by default (one without it
//!   logs that, and opens AppKit's). Default: AppKit's. Linux has SDL3's alone, and reads no
//!   value.

use std::sync::OnceLock;

/// The borderless switch's variable.
pub const BORDERLESS_VAR: &str = "REIMS_VGPU_WINDOW_BORDERLESS";
/// The screen list's variable.
pub const SCREENS_VAR: &str = "REIMS_VGPU_WINDOW_SCREENS";
/// The window implementation's variable (macOS).
pub const IMPL_VAR: &str = "REIMS_VGPU_WINDOW_IMPL";

/// What the variables say.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Settings {
    /// Borderless windows that fill their screens.
    pub borderless: bool,
    /// The host screen of each display, by port; empty when unset.
    pub screens: Vec<usize>,
    /// Which window a display opens on macOS.
    pub window_impl: WindowImpl,
}

/// Which window a display opens on macOS (`REIMS_VGPU_WINDOW_IMPL`). Linux has SDL3's alone.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WindowImpl {
    /// AppKit's window, view and events (`shell`, `view`).
    #[default]
    AppKit,
    /// SDL3's window, over Cocoa, with the macOS extensions (`sdl`).
    Sdl,
}

/// The window implementation a value names: SDL3's for `sdl`, in any case and spaces aside, else
/// AppKit's.
pub fn parse_impl(value: Option<&str>) -> WindowImpl {
    match value {
        Some(value) if value.trim().eq_ignore_ascii_case("sdl") => WindowImpl::Sdl,
        _ => WindowImpl::AppKit,
    }
}

/// The settings the variables' values give, each `None` where unset.
pub fn parse(borderless: Option<&str>, screens: Option<&str>) -> Settings {
    Settings {
        borderless: borderless.is_some_and(parse_switch),
        screens: screens.and_then(parse_screens).unwrap_or_default(),
        window_impl: WindowImpl::AppKit,
    }
}

/// Whether a switch's value turns it on.
fn parse_switch(value: &str) -> bool {
    let value = value.trim();
    ["1", "true", "yes", "on"]
        .iter()
        .any(|on| value.eq_ignore_ascii_case(on))
}

/// A list of screen indices, or `None` if any entry is not one.
fn parse_screens(value: &str) -> Option<Vec<usize>> {
    value
        .split(',')
        .map(|entry| entry.trim().parse::<usize>().ok())
        .collect()
}

/// The process's settings, read from its environment the first time they are asked for. A
/// value that is not UTF-8 counts as unset.
pub fn settings() -> &'static Settings {
    static SETTINGS: OnceLock<Settings> = OnceLock::new();
    SETTINGS.get_or_init(|| {
        let var = |name: &str| std::env::var(name).ok();
        let (borderless, screens) = (var(BORDERLESS_VAR), var(SCREENS_VAR));
        Settings {
            window_impl: parse_impl(var(IMPL_VAR).as_deref()),
            ..parse(borderless.as_deref(), screens.as_deref())
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unset or unparsable values are the defaults
    #[test]
    fn unset_or_unparsable_values_are_the_defaults() {
        assert_eq!(parse(None, None), Settings::default());
        assert_eq!(parse(Some("maybe"), Some("one,two")), Settings::default());
        assert_eq!(parse(Some(""), Some("")), Settings::default());
    }

    /// Borderless is on for 1, true, yes and on, in any case
    #[test]
    fn borderless_takes_the_usual_words() {
        for on in ["1", "true", "TRUE", "Yes", "on", "ON", " on "] {
            assert!(parse(Some(on), None).borderless, "{on:?}");
        }
        for off in ["0", "false", "no", "off", "2", "enabled", "y"] {
            assert!(!parse(Some(off), None).borderless, "{off:?}");
        }
    }

    /// The screen list is comma-separated indices, spaces allowed; one bad entry unsets it
    #[test]
    fn the_screen_list_is_all_or_nothing() {
        assert_eq!(parse(None, Some("1,0")).screens, vec![1, 0]);
        assert_eq!(parse(None, Some(" 2 , 0,1 ")).screens, vec![2, 0, 1]);
        assert_eq!(parse(None, Some("3")).screens, vec![3]);
        for bad in ["1,,0", "1,0,", "1;0", "-1,0", "1,x", "1.5", ","] {
            assert_eq!(
                parse(None, Some(bad)).screens,
                Vec::<usize>::new(),
                "{bad:?}"
            );
        }
    }

    /// The window is AppKit's unless the variable names SDL3's
    #[test]
    fn the_window_is_appkits_unless_sdl_is_named() {
        for sdl in ["sdl", "SDL", " Sdl "] {
            assert_eq!(parse_impl(Some(sdl)), WindowImpl::Sdl, "{sdl:?}");
        }
        for other in ["", "appkit", "sdl3", "1", "cocoa"] {
            assert_eq!(parse_impl(Some(other)), WindowImpl::AppKit, "{other:?}");
        }
        assert_eq!(parse_impl(None), WindowImpl::AppKit);
    }

    /// The settings are read once and kept
    #[test]
    fn the_settings_are_read_once() {
        assert!(std::ptr::eq(settings(), settings()));
    }
}
