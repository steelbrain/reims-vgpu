//! Embeds what reims-ctl carries of the guest's side for reims-vgpu's own device (src/prebuilt.rs):
//! - the x86-64 guest's kexts, the set guest/prebuilt/set.toml lists, which `make -C guest
//!   prebuilt` writes, and the arm64 guest's, the set guest/prebuilt-arm64e/set.toml lists, which
//!   `make -C guest prebuilt-arm64e` writes. Each of a set's `[[kext]]` entries names a bundle
//!   beside its set.toml, `<BundlePath>`, whose every file is included, in the set's order, with
//!   the entry's `ExecutablePath` and the Darwin versions it loads on (`MinKernel`,
//!   `MaxKernel`); its other keys are OpenCore's, which reims-util writes itself. A checkout
//!   without a set builds a reims-ctl that carries none of it, which refuses to make an image or
//!   a bundle for reims-vgpu rather than make one without the guest's drivers;
//! - the device's driver module for Ambrosia, firmware/ambrosia's ReimsGraphicsModule.c, built
//!   here as its Makefile builds `build/ReimsGraphics.elf`: with clang and the Makefile's
//!   `CFLAGS`, and linked with the `LDFLAGS` by the toolchain's own lld (rust-lld, as GNU's
//!   `ld.lld`), so the build needs clang and no lld of the host's. Where clang can't build it,
//!   reims-ctl carries no module, says so as it builds, and refuses to start an arm64 VM.
//!
//! A reims-ctl for Windows, which runs no arm64 VM (src/host.rs), carries neither the arm64 set
//! nor the module.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let crate_dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets it"));
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets it"));
    let out = out_dir.join("set.rs");
    // A reims-ctl for Windows runs no arm64 VM (src/host.rs), and carries no arm64 guest.
    let windows = std::env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "windows");
    let (set_arm64e, module) = if windows {
        ("None".to_string(), "None".to_string())
    } else {
        (
            set_code(
                &crate_dir.join("../../guest/prebuilt-arm64e"),
                "guest/prebuilt-arm64e",
            ),
            module_code(&crate_dir.join("../../firmware/ambrosia"), &out_dir),
        )
    };
    let set = format!(
        "/// guest/prebuilt/set.toml's set, in its order.\n\
         pub static SET: Option<&[Kext]> = {};\n\
         /// guest/prebuilt-arm64e/set.toml's set, in its order.\n\
         pub static SET_ARM64E: Option<&[Kext]> = {set_arm64e};\n\
         /// The device's driver module for Ambrosia, firmware/ambrosia's ReimsGraphics.elf.\n\
         pub static AMBROSIA_MODULE: Option<&[u8]> = {module};\n",
        set_code(&crate_dir.join("../../guest/prebuilt"), "guest/prebuilt"),
    );
    std::fs::write(&out, set).unwrap_or_else(|e| panic!("write {}: {e}", out.display()));
}

/// The set in `prebuilt` (`name` in messages), as an `Option<&[Kext]>` expression.
fn set_code(prebuilt: &Path, name: &str) -> String {
    let manifest = prebuilt.join("set.toml");
    println!("cargo::rerun-if-changed={}", manifest.display());
    match std::fs::read_to_string(&manifest) {
        Ok(text) => {
            let prebuilt = prebuilt
                .canonicalize()
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            format!("Some(&[\n{}])", kexts(&prebuilt, &text, name))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => "None".to_string(),
        Err(e) => panic!("read {}: {e}", manifest.display()),
    }
}

/// The module built from `fw` into `out`, as an `Option<&[u8]>` expression: None, with a warning,
/// where it can't be built.
fn module_code(fw: &Path, out: &Path) -> String {
    let wire = fw.join("../../crates/reims-vgpu-graphics/include");
    for f in [
        fw.join("Makefile"),
        fw.join("ReimsGraphicsModule.c"),
        fw.join("AmbrosiaGraphics.h"),
        wire.join("reims_graphics_wire.h"),
    ] {
        println!("cargo::rerun-if-changed={}", f.display());
    }
    match build_module(fw, &wire, out) {
        Ok(elf) => format!(
            "Some(include_bytes!({:?}))",
            elf.to_str().expect("a UTF-8 path")
        ),
        Err(why) => {
            println!(
                "cargo::warning=this reims-ctl carries no driver module for Ambrosia, and starts \
                 no arm64 VM: {why}"
            );
            "None".to_string()
        }
    }
}

fn build_module(fw: &Path, wire: &Path, out: &Path) -> Result<PathBuf, String> {
    let makefile = std::fs::read_to_string(fw.join("Makefile"))
        .map_err(|e| format!("read {}: {e}", fw.join("Makefile").display()))?;
    let wire = wire.to_str().ok_or("a path that is not UTF-8")?;
    let cflags: Vec<String> = make_var(&makefile, "CFLAGS")?
        .into_iter()
        .map(|f| f.replace("$(WIRE)", wire))
        .collect();
    let ldflags = make_var(&makefile, "LDFLAGS")?;
    let obj = out.join("ReimsGraphicsModule.o");
    let elf = out.join("ReimsGraphics.elf");
    run(Command::new("clang")
        .args(&cflags)
        .arg("-c")
        .arg(fw.join("ReimsGraphicsModule.c"))
        .arg("-o")
        .arg(&obj))?;
    run(Command::new(rust_lld()?)
        .args(["-flavor", "gnu"])
        .args(&ldflags)
        .arg("-o")
        .arg(&elf)
        .arg(&obj))?;
    Ok(elf)
}

/// The words of the Makefile's `name = ...` assignment, its continued lines joined.
fn make_var(makefile: &str, name: &str) -> Result<Vec<String>, String> {
    let mut lines = makefile.lines();
    let first = lines
        .find_map(|l| {
            l.strip_prefix(name)
                .and_then(|r| r.trim_start().strip_prefix('='))
        })
        .ok_or_else(|| format!("firmware/ambrosia/Makefile has no {name}"))?;
    let mut text = String::new();
    let mut line = first;
    loop {
        match line.strip_suffix('\\') {
            Some(l) => {
                text.push_str(l);
                text.push(' ');
                line = lines.next().unwrap_or_default();
            }
            None => {
                text.push_str(line);
                break;
            }
        }
    }
    Ok(text.split_whitespace().map(str::to_string).collect())
}

/// The toolchain's lld, `rust-lld` in its sysroot, for this host.
fn rust_lld() -> Result<PathBuf, String> {
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let host = std::env::var("HOST").map_err(|e| format!("HOST: {e}"))?;
    let out = Command::new(&rustc)
        .args(["--print", "sysroot"])
        .output()
        .map_err(|e| format!("run rustc --print sysroot: {e}"))?;
    let sysroot = String::from_utf8(out.stdout).map_err(|e| format!("rustc's sysroot: {e}"))?;
    let exe = if host.contains("windows") { ".exe" } else { "" };
    let lld = Path::new(sysroot.trim())
        .join("lib/rustlib")
        .join(&host)
        .join("bin")
        .join(format!("rust-lld{exe}"));
    if lld.is_file() {
        Ok(lld)
    } else {
        Err(format!("the toolchain has no {}", lld.display()))
    }
}

fn run(cmd: &mut Command) -> Result<(), String> {
    let what = format!("{cmd:?}");
    let out = cmd.output().map_err(|e| format!("{what}: {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(format!(
            "{what} failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ))
    }
}

/// The set's entries, as `Kext` expressions, a line each.
fn kexts(prebuilt: &Path, text: &str, name: &str) -> String {
    let set: toml::Table = text
        .parse()
        .unwrap_or_else(|e| panic!("{name}/set.toml: {e}"));
    let entries = set
        .get("kext")
        .and_then(toml::Value::as_array)
        .unwrap_or_else(|| panic!("{name}/set.toml has no [[kext]] entries"));
    let mut code = String::new();
    for entry in entries {
        let key = |k: &str| entry.get(k).and_then(toml::Value::as_str);
        let bundle = key("BundlePath").expect("a [[kext]] entry has no BundlePath");
        assert!(
            bundle.ends_with(".kext") && !bundle.contains('/') && !bundle.starts_with('.'),
            "{name}/set.toml: BundlePath {bundle:?} is not a bundle's name"
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
                "{name}/set.toml: {bundle}'s kernel bound {b:?} is not X.Y.Z"
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
