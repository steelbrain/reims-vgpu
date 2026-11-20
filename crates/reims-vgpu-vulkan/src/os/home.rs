//! The user's cache directory on Linux and macOS: `.cache` under `HOME` (`crate::os`'s
//! `user_cache_directory`).

use std::ffi::OsString;
use std::path::PathBuf;

/// `.cache` under `HOME`; `None` where `HOME` is not set.
pub(super) fn cache_directory() -> Option<PathBuf> {
    in_home(std::env::var_os("HOME"))
}

/// `.cache` under `home`, where there is one.
fn in_home(home: Option<OsString>) -> Option<PathBuf> {
    Some(PathBuf::from(home?).join(".cache"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The user's cache directory is `.cache` under `HOME`, and none without it, so the library's
    /// cache directory is `~/.cache/reims-vgpu-vulkan` (a test build's
    /// `~/.cache/reims-vgpu-vulkan-tests`)
    #[test]
    fn the_cache_directory_is_dot_cache_under_home() {
        assert_eq!(
            in_home(Some("/home/u".into())),
            Some(PathBuf::from("/home/u/.cache"))
        );
        assert_eq!(in_home(None), None);
        let home = PathBuf::from(std::env::var_os("HOME").expect("HOME is set"));
        assert_eq!(crate::os::user_cache_directory(), Some(home.join(".cache")));
        assert_eq!(
            crate::os::cache_directory(),
            Some(home.join(".cache/reims-vgpu-vulkan-tests"))
        );
    }
}
