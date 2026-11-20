//! The user's cache directory on Windows: `reims-vgpu` in the user's local application data folder
//! (`crate::os`'s `user_cache_directory`), as the shell names that folder for the user
//! (`SHGetKnownFolderPath` of `FOLDERID_LocalAppData`), whatever the process's environment holds:
//! a process started outside MSYS2's shell, as a user's QEMU is, has no `HOME`, and a service may
//! lack `LOCALAPPDATA`.

use std::ffi::OsString;
use std::os::windows::ffi::OsStringExt as _;
use std::path::PathBuf;

use windows_sys::Win32::Foundation::S_OK;
use windows_sys::Win32::System::Com::CoTaskMemFree;
use windows_sys::Win32::UI::Shell::{FOLDERID_LocalAppData, SHGetKnownFolderPath};

/// The library's directory in the folder.
const DIRECTORY: &str = "reims-vgpu";

/// `reims-vgpu` in the user's local application data folder; `None` where the shell names no
/// such folder.
pub(crate) fn cache_directory() -> Option<PathBuf> {
    let mut path: windows_sys::core::PWSTR = std::ptr::null_mut();
    // SAFETY: the folder of the process's own user (no token, no flags), into a live
    // out-parameter the call sets to a string it allocates, freed below whatever the call returned.
    let result =
        unsafe { SHGetKnownFolderPath(&FOLDERID_LocalAppData, 0, std::ptr::null_mut(), &mut path) };
    let folder = (result == S_OK && !path.is_null()).then(|| {
        // SAFETY: a NUL-terminated wide string the call allocated.
        let len = (0..).take_while(|&i| unsafe { *path.add(i) } != 0).count();
        // SAFETY: `len` wide characters before the NUL.
        OsString::from_wide(unsafe { std::slice::from_raw_parts(path, len) })
    });
    // SAFETY: the string the call allocated with the COM allocator, or null, which the free
    // ignores; freed once.
    unsafe { CoTaskMemFree(path.cast()) };
    Some(PathBuf::from(folder?).join(DIRECTORY))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The user's cache directory is `reims-vgpu` in the user's local application data folder,
    /// the folder Windows names in the process's `LOCALAPPDATA` (a path is the same in any case),
    /// and the library's stores lie in it under the names they have under `~/.cache` on Linux and
    /// macOS.
    #[test]
    fn the_cache_directory_is_reims_vgpu_in_the_local_application_data_folder() {
        let dir = cache_directory().expect("a local application data folder");
        let folder = PathBuf::from(std::env::var_os("LOCALAPPDATA").expect("LOCALAPPDATA is set"));
        let lower = |path: &std::path::Path| path.to_string_lossy().to_lowercase();
        assert_eq!(lower(&dir), lower(&folder.join("reims-vgpu")));
        assert_eq!(crate::os::user_cache_directory(), Some(dir.clone()));
        assert_eq!(
            crate::os::cache_directory(),
            Some(dir.join("reims-vgpu-vulkan-tests"))
        );
    }
}
