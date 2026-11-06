//! Embeds the guest's kexts for reims-vgpu's own device (src/prebuilt.rs): the set
//! guest/prebuilt/set.toml lists, which `make -C guest prebuilt` writes. Each of its `[[kext]]`
//! entries names a bundle, `guest/prebuilt/<BundlePath>`, whose every file is included, in the
//! set's order, with the entry's `ExecutablePath` and the Darwin versions it loads on
//! (`MinKernel`, `MaxKernel`); its other keys are OpenCore's, which reims-util writes itself. A
//! checkout without the set builds a reims-ctl that carries none, which refuses to make an image
//! for reims-vgpu rather than make one without the guest's drivers.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

fn main() {
    let crate_dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets it"));
    let prebuilt = crate_dir.join("../../guest/prebuilt");
    let manifest = prebuilt.join("set.toml");
    println!("cargo::rerun-if-changed={}", manifest.display());
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets it")).join("set.rs");
    let code = match std::fs::read_to_string(&manifest) {
        Ok(text) => {
            let prebuilt = prebuilt
                .canonicalize()
                .expect("guest/prebuilt holds set.toml");
            format!("Some(&[\n{}])", kexts(&prebuilt, &text))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => "None".to_string(),
        Err(e) => panic!("read {}: {e}", manifest.display()),
    };
    let set = format!(
        "/// guest/prebuilt/set.toml's set, in its order.\n\
         pub static SET: Option<&[Kext]> = {code};\n"
    );
    std::fs::write(&out, set).unwrap_or_else(|e| panic!("write {}: {e}", out.display()));
}

/// The set's entries, as `Kext` expressions, a line each.
fn kexts(prebuilt: &Path, text: &str) -> String {
    let set: toml::Table = text
        .parse()
        .unwrap_or_else(|e| panic!("guest/prebuilt/set.toml: {e}"));
    let entries = set
        .get("kext")
        .and_then(toml::Value::as_array)
        .expect("guest/prebuilt/set.toml has no [[kext]] entries");
    let mut code = String::new();
    for entry in entries {
        let key = |k: &str| entry.get(k).and_then(toml::Value::as_str);
        let bundle = key("BundlePath").expect("a [[kext]] entry has no BundlePath");
        assert!(
            bundle.ends_with(".kext") && !bundle.contains('/') && !bundle.starts_with('.'),
            "guest/prebuilt/set.toml: BundlePath {bundle:?} is not a bundle's name"
        );
        let some = |k: &str| match key(k).filter(|v| !v.is_empty()) {
            Some(v) => format!("Some({v:?})"),
            None => "None".to_string(),
        };
        let executable = some("ExecutablePath");
        let bounds = [some("MinKernel"), some("MaxKernel")];
        for b in ["MinKernel", "MaxKernel"].into_iter().filter_map(key) {
            assert!(
                b.is_empty()
                    || (b.split('.').count() == 3
                        && b.split('.').all(|n| n.parse::<u32>().is_ok())),
                "guest/prebuilt/set.toml: {bundle}'s kernel bound {b:?} is not X.Y.Z"
            );
        }
        let dir = prebuilt.join(bundle);
        let mut files = Vec::new();
        walk(&dir, &dir, &mut files);
        files.sort();
        assert!(!files.is_empty(), "{} holds no files", dir.display());
        let mut list = String::new();
        for (rel, path) in &files {
            println!("cargo::rerun-if-changed={}", path.display());
            let path = path.to_str().expect("a UTF-8 path");
            write!(list, "({rel:?}, include_bytes!({path:?})), ").unwrap();
        }
        writeln!(
            code,
            "    Kext {{ bundle: {bundle:?}, executable: {executable}, min_kernel: {}, \
             max_kernel: {}, files: &[{list}] }},",
            bounds[0], bounds[1]
        )
        .unwrap();
    }
    code
}

/// Every file under `dir`, by its path relative to `root` with '/' between its names: a bundle's
/// directories hold files and directories alone.
fn walk(root: &Path, dir: &Path, files: &mut Vec<(String, PathBuf)>) {
    println!("cargo::rerun-if-changed={}", dir.display());
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display()));
    for e in entries {
        let e = e.unwrap_or_else(|e| panic!("read {}: {e}", dir.display()));
        let path = e.path();
        let kind = e
            .file_type()
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        if kind.is_dir() {
            walk(root, &path, files);
        } else if kind.is_file() {
            let rel = path.strip_prefix(root).expect("under the bundle");
            let rel: Vec<_> = rel
                .iter()
                .map(|n| n.to_str().expect("a UTF-8 name"))
                .collect();
            files.push((rel.join("/"), path));
        } else {
            panic!("{} is neither a file nor a directory", path.display());
        }
    }
}
