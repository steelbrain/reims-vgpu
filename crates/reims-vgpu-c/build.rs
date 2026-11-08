//! On macOS: links the cdylib with the install name `@rpath/libreims_vgpu_c.dylib`.
//!
//! Without it the linker gives the dylib its output path as its install name (`LC_ID_DYLIB`), an
//! absolute path in the build directory, which a host linked to it records and dyld then loads,
//! whatever the host's rpaths. kache restores a cached link's output in another target directory
//! with the bytes it was first linked with, so the C harness of a new worktree linked to a dylib
//! that named the worktree it came from, and failed to load once that worktree was removed. With
//! `@rpath/` a host finds the dylib through its own rpaths, wherever the build ran. `cargo xtask
//! dist` names the dylib it installs `@rpath/libreims_vgpu.dylib` (`xtask/src/dist/macos.rs`).
//! Linux needs nothing: a cdylib built here has no soname, and a host linked to it records its
//! file name.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "macos") {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-install_name,@rpath/libreims_vgpu_c.dylib");
    }
}
