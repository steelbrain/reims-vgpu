//! The library's own shaders: `shaders.metal` (the display's present functions), compiled by
//! Apple's tools when this crate is built (`build.rs`) into [`METALLIB`], which the rt platform
//! loads as the library core asks for with `newLibraryWithSource:options:error:`
//! (`crate::rt_platform::library`).

/// The metallib `build.rs` compiled `shaders.metal` into.
pub const METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders.metallib"));

#[cfg(test)]
mod tests {
    /// The metallibs the repository holds for a build without Apple's tools (`metallibs/`,
    /// `build.rs`) are this tree's: their record is the one `build.rs` writes of this tree's
    /// commands and sources, and the directory holds every metallib it names and no other. So a
    /// source changed on a Mac without `cargo xtask metallibs` fails here, not on Linux. The
    /// directory is the crate's, which cargo and nextest give the test process: read at run time,
    /// so the test is the same binary in every worktree (docs/development.md, "Building through
    /// kache").
    #[test]
    fn the_committed_metallibs_are_this_trees() {
        let crate_dir = std::env::var_os("CARGO_MANIFEST_DIR")
            .expect("CARGO_MANIFEST_DIR is unset: run the tests with cargo or nextest");
        let dir = std::path::Path::new(&crate_dir).join("metallibs");
        let mend = "run `cargo xtask metallibs` on a Mac and commit what it writes";
        let tree = include_str!(concat!(env!("OUT_DIR"), "/metallibs.toml"));
        let committed = std::fs::read_to_string(dir.join("metallibs.toml"))
            .unwrap_or_else(|e| panic!("{}: {e}: {mend}", dir.display()));
        assert_eq!(
            committed,
            tree,
            "{}'s record is not this tree's: {mend}",
            dir.display()
        );
        let mut named: Vec<&str> = tree
            .lines()
            .filter_map(|line| line.split_once(" = "))
            .map(|(file, _)| file.trim_matches('"'))
            .collect();
        named.sort_unstable();
        let mut held: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name != "metallibs.toml")
            .collect();
        held.sort_unstable();
        assert_eq!(held, named, "{}: {mend}", dir.display());
    }
}
