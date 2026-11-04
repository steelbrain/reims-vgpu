//! Where this crate's dynamic library is, for the tests that link C to it or read its symbols.

use std::env::consts::{DLL_PREFIX, DLL_SUFFIX};
use std::path::{Path, PathBuf};

/// The library's file name: `libreims_vgpu_c.dylib` on macOS, `libreims_vgpu_c.so` on Linux.
pub fn file_name() -> String {
    format!("{DLL_PREFIX}reims_vgpu_c{DLL_SUFFIX}")
}

/// The directory holding the library a build of this test made. Cargo's build directory has two
/// layouts. In the first, every unit's output is in the profile's `deps/`, this test binary's own
/// directory, where the library's name carries no hash. In the second (`-Zbuild-dir-new-layout`,
/// which nightly-2026-11-03's Cargo uses by default, measured), each unit writes to a directory of
/// its own, `<profile>/build/<package>/<hash>/out/`, and each build of the library with other
/// features or dependencies keeps its own: the one written last is taken, as `deps/` held the
/// library written last.
pub fn dir() -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    let own = exe.parent().unwrap();
    let name = file_name();
    if own.join(&name).exists() {
        return own.to_path_buf();
    }
    // `<profile>/build/reims-vgpu-c/<hash>/out/<this test>`
    let package = own
        .parent()
        .and_then(Path::parent)
        .unwrap_or_else(|| panic!("{}: not in a unit's directory", exe.display()));
    std::fs::read_dir(package)
        .unwrap_or_else(|e| panic!("{}: {e}", package.display()))
        .filter_map(|unit| {
            let out = unit.ok()?.path().join("out");
            let written = out.join(&name).metadata().ok()?.modified().ok()?;
            Some((written, out))
        })
        .max()
        .map(|(_, out)| out)
        .unwrap_or_else(|| panic!("{name} is missing beside {}", exe.display()))
}
