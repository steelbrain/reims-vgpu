//! `cargo xtask header`: generate the C headers with cbindgen, each from its crate's
//! `cbindgen.toml`, or, with `--check`, check that each committed header equals cbindgen's output:
//! the C API's, `crates/reims-vgpu-c/include/reims_vgpu.h`, and the reims-graphics device's wire
//! format, `crates/reims-vgpu-graphics/include/reims_graphics_wire.h`, which the guest's kext and option
//! ROM include.
//!
//! cbindgen is an external binary (`cargo install cbindgen --version 0.29.4`, or Homebrew's).
//! Its output depends on its version, so a different version is reported rather than compared,
//! and fails `--check`: the committed header is the one [`CBINDGEN_VERSION`] makes.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::util::{Res, read, repo_root, run_capture, write_if_changed};

/// The cbindgen that made the committed header.
pub const CBINDGEN_VERSION: &str = "0.29.4";

/// Each header's crate and file name.
pub const HEADERS: [(&str, &str); 2] = [
    ("reims-vgpu-c", "reims_vgpu.h"),
    ("reims-vgpu-graphics", "reims_graphics_wire.h"),
];

/// A crate cbindgen reads.
fn crate_dir(krate: &str) -> PathBuf {
    repo_root().join("crates").join(krate)
}

/// A committed header: `crates/<krate>/include/<file>`.
pub fn header_path(krate: &str, file: &str) -> PathBuf {
    crate_dir(krate).join("include").join(file)
}

/// The version `cbindgen --version` reports (`cbindgen 0.29.4` -> `0.29.4`).
fn parse_version(out: &str) -> Option<&str> {
    out.split_whitespace().nth(1)
}

/// A scratch file path unique to this process and call.
fn scratch(name: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!("xtask-{}-{n}-{name}", std::process::id()))
}

/// cbindgen's header for a crate, as a string.
fn generate(cbindgen: &Path, krate: &str, file: &str) -> Res<String> {
    let dir = crate_dir(krate);
    let tmp = scratch(file);
    run_capture(
        Command::new(cbindgen)
            .current_dir(&dir)
            .arg("--quiet")
            .arg("--config")
            .arg(dir.join("cbindgen.toml"))
            .arg("--output")
            .arg(&tmp)
            .arg(&dir),
    )?;
    let text = read(&tmp);
    let _ = std::fs::remove_file(&tmp);
    text
}

/// The cbindgen drift check (`None`: cbindgen is not the committed header's version).
fn drift(cbindgen: &Path, check: bool) -> Res<Option<bool>> {
    let version = run_capture(Command::new(cbindgen).arg("--version")).map_err(|e| {
        format!("header: cbindgen not found ({e}); install cbindgen {CBINDGEN_VERSION}")
    })?;
    let version = parse_version(&version).unwrap_or("?").to_owned();
    if version != CBINDGEN_VERSION {
        eprintln!(
            "header: cbindgen {version}, but the header is made by {CBINDGEN_VERSION}; not compared"
        );
        return Ok(None);
    }
    let mut ok = true;
    for (krate, file) in HEADERS {
        let generated = generate(cbindgen, krate, file)?;
        let path = header_path(krate, file);
        if check {
            let committed = read(&path)?;
            if committed != generated {
                eprintln!(
                    "header: {} differs from cbindgen's output; run `cargo xtask header`",
                    path.display()
                );
                ok = false;
            }
        } else if write_if_changed(&path, &generated)? {
            println!("wrote {}", path.display());
        }
    }
    if check && ok {
        println!("header: ok");
    }
    Ok(Some(ok))
}

/// `header [--check] [--cbindgen <path>]`.
pub fn cmd_header(args: &[String]) -> Res<bool> {
    let check = args.iter().any(|a| a == "--check");
    let cbindgen = args
        .iter()
        .position(|a| a == "--cbindgen")
        .and_then(|i| args.get(i + 1))
        .map_or_else(|| PathBuf::from("cbindgen"), PathBuf::from);
    // A cbindgen of another version writes nothing, and fails the check.
    Ok(drift(&cbindgen, check)?.unwrap_or(!check))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_parse() {
        assert_eq!(parse_version("cbindgen 0.29.4\n"), Some("0.29.4"));
        assert_eq!(parse_version(""), None);
    }

    #[test]
    fn headers_live_in_their_crates() {
        let [(api, api_h), (wire, wire_h)] = HEADERS;
        assert!(header_path(api, api_h).ends_with("crates/reims-vgpu-c/include/reims_vgpu.h"));
        assert!(
            header_path(wire, wire_h)
                .ends_with("crates/reims-vgpu-graphics/include/reims_graphics_wire.h")
        );
    }
}
