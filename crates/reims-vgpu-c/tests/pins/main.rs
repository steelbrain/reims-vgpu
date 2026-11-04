//! The C API's pins: builds `tests/pins/api_pins.c` against the committed `include/reims_vgpu.h`,
//! links it to this crate's dylib, runs it, and compares its whole output with
//! `tests/pins/expected.tsv`, recorded from this build. `api_pins.c`'s comments say what each pin
//! observes.
//!
//! A change meant to alter what a pin observes re-records the file and commits its diff:
//! `RVG_PINS_RECORD=1 cargo test -p reims-vgpu-c --test pins` writes this build's output to
//! `expected.tsv` instead of comparing. A pin whose output varies between runs is a broken pin:
//! make it print something deterministic, or remove it.
//!
//! The pins run on macOS only: `api_pins.c` throws NSExceptions through the Objective-C runtime,
//! and the recording is the Metal platform's.

#![cfg(target_os = "macos")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::PathBuf;
use std::process::Command;

#[path = "../support/cdylib.rs"]
mod cdylib;

/// The crate's directory, which cargo and nextest give a test process: read at run time, so the
/// test is the same binary in every worktree (docs/development.md, "Building through kache").
fn crate_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .expect("CARGO_MANIFEST_DIR: run the tests with cargo or nextest")
}

/// Set to record `expected.tsv` from this build instead of comparing with it.
const RECORD: &str = "RVG_PINS_RECORD";

/// Builds `api_pins.c` against this crate's dylib and runs it: its standard output.
fn run_pins() -> String {
    let dylib = cdylib::dir().join(cdylib::file_name());
    assert!(dylib.exists(), "{} is missing", dylib.display());
    let dir = std::env::temp_dir().join(format!("rvg-api-pins-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let bin = dir.join("api_pins");
    let deps = cdylib::dir();
    let cc = Command::new("cc")
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic", "-I"])
        .arg(crate_dir().join("include"))
        .arg(crate_dir().join("tests/pins/api_pins.c"))
        .arg("-L")
        .arg(&deps)
        .arg("-lreims_vgpu_c")
        .arg(format!("-Wl,-rpath,{}", deps.display()))
        .args(["-framework", "Foundation", "-lobjc", "-o"])
        .arg(&bin)
        .output()
        .unwrap();
    assert!(
        cc.status.success(),
        "cc failed:\n{}",
        String::from_utf8_lossy(&cc.stderr)
    );
    let out = Command::new(&bin).output().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success(),
        "api_pins did not run to its end:\n{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    stdout
}

/// The C API as a host sees it: every pin of tests/pins/api_pins.c observes what `expected.tsv`
/// records (the order and thread of the device's callbacks, each table's one destroy, the
/// display thread, a display's life through a frame, a window display with no run loop, the
/// refusal of add_trace_range alone, the registers, a resume's refusals, the EFI display's
/// traced range, resume-child writes past the child FIFOs, the feature level and the
/// connection types as the guest reads them, the log hook, and the deaths that remain)
#[test]
fn c_api_pins_match_the_recording() {
    let got = run_pins();
    let path = crate_dir().join("tests/pins/expected.tsv");
    if std::env::var_os(RECORD).is_some() {
        std::fs::write(&path, &got).unwrap();
        eprintln!(
            "recorded {} lines to {}",
            got.lines().count(),
            path.display()
        );
        return;
    }
    let want = std::fs::read_to_string(&path).unwrap();
    let (got, want): (Vec<&str>, Vec<&str>) = (got.lines().collect(), want.lines().collect());
    assert!(want.len() >= 20, "{}", want.len());
    let diff: Vec<String> = (0..got.len().max(want.len()))
        .filter(|&i| got.get(i) != want.get(i))
        .map(|i| {
            format!(
                "line {}:\n  recorded: {}\n  now:      {}",
                i + 1,
                want.get(i).unwrap_or(&"(none)"),
                got.get(i).unwrap_or(&"(none)")
            )
        })
        .collect();
    assert!(
        diff.is_empty(),
        "the pins differ from tests/pins/expected.tsv ({RECORD}=1 re-records it):\n{}",
        diff.join("\n")
    );
}
