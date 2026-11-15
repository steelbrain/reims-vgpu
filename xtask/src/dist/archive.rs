//! What the Linux and Windows dists share about the archive: rustc's note of the native libraries
//! a static link of it needs, `nm`'s listing of the global symbols a file defines, and the check
//! that the archive installed keeps the C API's functions alone global.

use std::path::Path;

use super::strs_with;
use crate::util::Res;

/// The directory under the target dir the archive is localized in.
pub const LOCALIZED: &str = "dist-localized";

/// The prefix of every symbol the archive keeps global: the C API's functions.
pub const API_PREFIX: &str = "rvg_";

/// `Libs.private` from rustc's `native-static-libs: ...` note in `cargo`'s output, in rustc's
/// order. A note that names none is refused: the archive always needs the C library.
pub fn native_static_libs(cargo: &str) -> Res<Vec<String>> {
    const NOTE: &str = "native-static-libs: ";
    let Some(at) = cargo.find(NOTE) else {
        return Err(format!(
            "rustc printed no native-static-libs note:\n{}",
            cargo.lines().rev().take(20).collect::<Vec<_>>().join("\n")
        ));
    };
    let rest = &cargo[at + NOTE.len()..];
    // The note ends its JSON string (`"`), or its line, before an escape (`\n`).
    let end = rest.find(['"', '\\', '\n']).unwrap_or(rest.len());
    let libs: Vec<String> = rest[..end].split_whitespace().map(str::to_owned).collect();
    if libs.is_empty() {
        return Err("rustc's native-static-libs note names no library".to_owned());
    }
    Ok(libs)
}

/// `nm`'s arguments that list the global symbols `file` defines, one a line, in POSIX's format:
/// the name, then its type.
pub fn nm_args(file: &Path) -> Vec<String> {
    strs_with(&["-g", "--defined-only", "-P"], file)
}

/// The names in `nm -g --defined-only -P`'s output, in its order: each line of a name and a
/// one-letter type, past the lines that name an archive's member.
pub fn defined_globals(nm: &str) -> Vec<String> {
    nm.lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let name = fields.next()?;
            let kind = fields.next()?;
            (kind.len() == 1).then(|| name.to_owned())
        })
        .collect()
}

/// The C API's functions among the globals `nm` lists ([`defined_globals`]) for `object`, in its
/// order. Fails where it lists none: `object` holds no C API.
pub fn api_functions(nm: &str, object: &Path) -> Res<Vec<String>> {
    let api: Vec<String> = defined_globals(nm)
        .into_iter()
        .filter(|name| name.starts_with(API_PREFIX))
        .collect();
    if api.is_empty() {
        return Err(format!(
            "{} defines no {API_PREFIX} function: it holds no C API",
            object.display()
        ));
    }
    Ok(api)
}

/// Fails where `globals`, global symbols `archive` defines, holds any not of the C API, naming
/// the first few.
pub fn check_globals(archive: &Path, globals: impl IntoIterator<Item = String>) -> Res<()> {
    let others: Vec<String> = globals
        .into_iter()
        .filter(|name| !name.starts_with(API_PREFIX))
        .collect();
    if others.is_empty() {
        return Ok(());
    }
    Err(format!(
        "{} defines {} global symbols besides the C API's, such as {}: a host's own copies would \
         clash with them",
        archive.display(),
        others.len(),
        others
            .iter()
            .take(8)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

#[cfg(test)]
pub(super) mod tests {
    use super::super::strs;
    use super::*;

    /// What `cargo rustc --message-format json -- --print native-static-libs` prints on Linux:
    /// the note, once as its message and once rendered, among the other messages.
    pub const CARGO: &str = r#"{"reason":"compiler-artifact","package_id":"path+file:///r/crates/reims-vgpu-vulkan#0.1.0","filenames":["/t/release/deps/libreims_vgpu_vulkan.rlib"],"fresh":true}
{"reason":"compiler-message","package_id":"path+file:///r/crates/reims-vgpu-c#0.1.0","message":{"rendered":"note: Link against the following native artifacts when linking against this static library. The order and any duplication can be significant on some platforms.\n\n","children":[],"level":"note","message":"Link against the following native artifacts when linking against this static library. The order and any duplication can be significant on some platforms.","spans":[]}}
{"reason":"compiler-message","package_id":"path+file:///r/crates/reims-vgpu-c#0.1.0","message":{"rendered":"note: native-static-libs: -lgcc_s -lutil -lrt -lpthread -lm -ldl -lc\n\n","children":[],"level":"note","message":"native-static-libs: -lgcc_s -lutil -lrt -lpthread -lm -ldl -lc","spans":[]}}
{"reason":"build-finished","success":true}"#;

    #[test]
    fn native_static_libs_reads_the_note_in_order() {
        assert_eq!(
            native_static_libs(CARGO).unwrap(),
            strs(&[
                "-lgcc_s",
                "-lutil",
                "-lrt",
                "-lpthread",
                "-lm",
                "-ldl",
                "-lc"
            ])
        );
        // rustc's own text, as a terminal shows it.
        assert_eq!(
            native_static_libs("note: native-static-libs: -lgcc_s -lc\n\nwarning: x").unwrap(),
            strs(&["-lgcc_s", "-lc"])
        );
    }

    #[test]
    fn native_static_libs_refuses_no_note_and_an_empty_one() {
        let e = native_static_libs(r#"{"reason":"build-finished","success":true}"#).unwrap_err();
        assert!(e.contains("no native-static-libs note"), "{e}");
        let e = native_static_libs(r#"{"message":"native-static-libs: "}"#).unwrap_err();
        assert!(e.contains("names no library"), "{e}");
    }

    /// nm's POSIX lines give their names, past an archive's member lines
    #[test]
    fn defined_globals_reads_nms_names() {
        let listing = "libreims_vgpu.a[reims_vgpu.o]:\n\
                       rvg_graphics_device_create T 1000 20\nrvg_status_message T 1020 8\n";
        assert_eq!(
            defined_globals(listing),
            strs(&["rvg_graphics_device_create", "rvg_status_message"])
        );
        assert_eq!(defined_globals(""), Vec::<String>::new());
    }

    /// The C API's functions alone pass; any other global fails, counted and named
    #[test]
    fn check_globals_refuses_any_global_but_the_c_apis() {
        let archive = Path::new("lib/libreims_vgpu.a");
        check_globals(archive, strs(&["rvg_a", "rvg_b"])).unwrap();
        check_globals(archive, Vec::new()).unwrap();
        let e = check_globals(archive, strs(&["rvg_a", "SDL_Init", "SDL_Quit"])).unwrap_err();
        assert!(e.contains("2 global symbols besides the C API's"), "{e}");
        assert!(e.contains("SDL_Init, SDL_Quit"), "{e}");
    }
}
