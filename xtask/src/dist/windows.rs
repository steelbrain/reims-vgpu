//! `cargo xtask dist` on Windows: refused, as no dist tree is laid out for Windows yet. The
//! shared library's extension is the one Windows gives a DLL, which the tree's layout names
//! (`super::layout`).

use std::path::{Path, PathBuf};

use super::Options;
use crate::tools::Tools;
use crate::util::Res;

/// The shared library's extension, `dll`.
pub const SHARED_LIBRARY_EXTENSION: &str = "dll";

/// Refuses, building and writing nothing: no dist tree is made on Windows.
pub fn dist(
    _tools: &mut dyn Tools,
    _o: &Options,
    _target_dir: &Path,
    _header: &Path,
) -> Res<PathBuf> {
    Err("cargo xtask dist: no dist tree is made on Windows yet".to_owned())
}

#[cfg(test)]
mod tests {
    use super::super::testing::Log;
    use super::*;
    use crate::tools::Profile;

    /// `dist` on Windows refuses, and runs no tool
    #[test]
    fn dist_is_refused() {
        let mut log = Log::default();
        let o = Options {
            profile: Profile("release".into()),
            out: PathBuf::from("out"),
        };
        let e = dist(&mut log, &o, Path::new("target"), Path::new("reims_vgpu.h")).unwrap_err();
        assert!(e.contains("Windows"), "{e}");
        assert!(log.lines().is_empty());
    }
}
