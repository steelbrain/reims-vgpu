//! Builds `tests/c/harness.c` against the committed `include/reims_vgpu.h`, links it to this
//! crate's dynamic library, and runs it. The harness uses the platform the library is built on, as
//! a C host does: `MetalPlatform`, or `VulkanPlatform` with the `vulkan` feature.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::env::consts::{DLL_PREFIX, DLL_SUFFIX};
use std::os::unix::process::ExitStatusExt as _;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The directory holding this crate's dynamic library (this test binary's own).
fn deps_dir() -> PathBuf {
    std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// The crate's directory, which cargo and nextest give a test process: read at run time, so the
/// test is the same binary in every worktree (docs/development.md, "Building through kache").
fn crate_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .expect("CARGO_MANIFEST_DIR: run the tests with cargo or nextest")
}

/// Compiles `source` into `out`; panics with the compiler's output.
fn compile(source: &Path, out: &Path) {
    let deps = deps_dir();
    let status = Command::new("cc")
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic"])
        .arg("-I")
        .arg(crate_dir().join("include"))
        .arg(source)
        .arg("-L")
        .arg(&deps)
        .arg("-lreims_vgpu_c")
        .arg(format!("-Wl,-rpath,{}", deps.display()))
        .arg("-o")
        .arg(out)
        .output()
        .unwrap();
    assert!(
        status.status.success(),
        "cc failed:\n{}",
        String::from_utf8_lossy(&status.stderr)
    );
}

/// The harness binary, built into a directory of its own; the directory is removed with the
/// returned guard.
struct Harness {
    dir: PathBuf,
    bin: PathBuf,
}

impl Drop for Harness {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn harness(name: &str) -> Harness {
    let dylib = deps_dir().join(format!("{DLL_PREFIX}reims_vgpu_c{DLL_SUFFIX}"));
    assert!(dylib.exists(), "{} is missing", dylib.display());
    let dir = std::env::temp_dir().join(format!("rvg-api-harness-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let bin = dir.join("harness");
    compile(&crate_dir().join("tests/c/harness.c"), &bin);
    Harness { dir, bin }
}

/// The C API from C, end to end on the library's platform — every check of tests/c/harness.c passes
#[test]
fn c_harness_passes() {
    let h = harness("run");
    let out = Command::new(&h.bin).output().unwrap();
    assert!(
        out.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("harness: ok"));
}

/// A NULL handle given to a function without a status ends the process (SIGABRT)
#[test]
fn c_harness_null_handle_aborts() {
    let h = harness("die");
    let out = Command::new(&h.bin).arg("die-null").output().unwrap();
    assert_eq!(out.status.signal(), Some(6), "{out:?}");
    assert!(!String::from_utf8_lossy(&out.stdout).contains("survived"));
}

/// Tests/c/harness.c calls every function include/reims_vgpu.h declares
#[test]
fn c_harness_calls_every_function() {
    let header = std::fs::read_to_string(crate_dir().join("include/reims_vgpu.h")).unwrap();
    let harness = std::fs::read_to_string(crate_dir().join("tests/c/harness.c")).unwrap();
    let declared: Vec<&str> = header
        .lines()
        .filter(|l| !l.starts_with("//") && !l.starts_with(' '))
        .filter_map(|l| {
            let i = l.find("rvg_")?;
            let rest = &l[i..];
            let end = rest.find('(')?;
            Some(&rest[..end])
        })
        .collect();
    assert_eq!(declared.len(), 48, "{declared:?}");
    let missing: Vec<&str> = declared
        .iter()
        .copied()
        .filter(|f| !harness.contains(&format!("{f}(")))
        .collect();
    assert!(missing.is_empty(), "harness.c never calls {missing:?}");
}

/// The header also compiles as C++ and, where the compiler has it (macOS), as Objective-C, alone
#[test]
fn header_compiles_as_cxx_and_objc() {
    let include = crate_dir().join("include");
    let objc = cfg!(target_os = "macos").then_some(("objective-c", "-std=c11"));
    for (lang, std) in std::iter::once(("c++", "-std=c++17")).chain(objc) {
        let out = Command::new("cc")
            .args(["-x", lang, std, "-fsyntax-only", "-Wall", "-Werror", "-I"])
            .arg(&include)
            .arg("-")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .and_then(|mut c| {
                use std::io::Write as _;
                c.stdin.take().unwrap().write_all(
                    b"#include \"reims_vgpu.h\"\nint main(void) { return RVG_MAX_DISPLAY_PORTS == 8 ? 0 : 1; }\n",
                )?;
                c.wait_with_output()
            })
            .unwrap();
        assert!(
            out.status.success(),
            "{lang}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

/// The committed header is cbindgen's output for this crate (the drift check of
/// `cargo xtask header --check`), when the pinned cbindgen is installed
#[test]
fn header_has_no_drift() {
    let Ok(v) = Command::new("cbindgen").arg("--version").output() else {
        eprintln!("cbindgen not installed: skipped");
        return;
    };
    if !String::from_utf8_lossy(&v.stdout).contains("0.29.4") {
        eprintln!("cbindgen is not 0.29.4: skipped");
        return;
    }
    let dir = crate_dir();
    let tmp = std::env::temp_dir().join(format!("rvg-api-drift-{}.h", std::process::id()));
    let out = Command::new("cbindgen")
        .current_dir(&dir)
        .arg("--quiet")
        .arg("--config")
        .arg(dir.join("cbindgen.toml"))
        .arg("--output")
        .arg(&tmp)
        .arg(&dir)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let generated = std::fs::read_to_string(&tmp).unwrap();
    let _ = std::fs::remove_file(&tmp);
    let committed = std::fs::read_to_string(dir.join("include/reims_vgpu.h")).unwrap();
    assert!(
        generated == committed,
        "include/reims_vgpu.h is stale: run `cargo xtask header`"
    );
}
