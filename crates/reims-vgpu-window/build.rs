//! On macOS with SDL's window (`macos-sdl`): links compiler-rt's builtins (`libclang_rt.osx.a`,
//! from `xcrun clang --print-runtime-dir`), unbundled, so they reach each final link of a crate
//! that links SDL and nothing else.
//!
//! SDL built for a macOS older than a check it makes (`@available`) calls
//! `__isPlatformVersionAtLeast`, as `cargo xtask dist`'s SDL does, which is built for macOS 11.0.
//! rustc links with no default libraries, so no compiler-rt, and the definition a link finds is
//! the standard library's: a link with LTO drops it from an executable, whose exports are its
//! own, and SDL's calls are left undefined (**measured** with rustc 1.98.1: `cargo nextest run
//! --release -p reims-vgpu --no-run` after `cargo xtask dist --release` in the same target
//! directory). A link that keeps the standard library's definition, a dev build's or the C API's
//! libraries', takes it as before; compiler-rt's member is loaded only where the symbol is still
//! undefined. The archive `dist` installs bundles none of it: a host's own link adds compiler-rt
//! (clang does by default).

use std::process::Command;

fn main() -> Result<(), String> {
    println!("cargo:rerun-if-changed=build.rs");
    let macos = std::env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "macos");
    if !macos || std::env::var_os("CARGO_FEATURE_MACOS_SDL").is_none() {
        return Ok(());
    }
    let out = Command::new("xcrun")
        .args(["clang", "--print-runtime-dir"])
        .output()
        .map_err(|e| {
            format!("xcrun clang --print-runtime-dir: {e}: the build needs Xcode's clang")
        })?;
    if !out.status.success() {
        return Err(format!(
            "xcrun clang --print-runtime-dir failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let stdout = String::from_utf8(out.stdout)
        .map_err(|e| format!("xcrun clang --print-runtime-dir: not UTF-8: {e}"))?;
    let dir = stdout.trim();
    // Run again where Xcode's clang moves (a new version's directory), as the old one goes.
    println!("cargo:rerun-if-changed={dir}/libclang_rt.osx.a");
    println!("cargo:rustc-link-search=native={dir}");
    println!("cargo:rustc-link-lib=static:-bundle=clang_rt.osx");
    Ok(())
}
