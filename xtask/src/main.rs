//! `cargo xtask <command>`: the workspace's lints and packaging (architecture 2.1, 13.1, 11.3).

// xtask is tooling, not product behaviour: it keeps its own tables in std's maps and sorts freely.
#![allow(clippy::disallowed_types, clippy::disallowed_methods)]

mod dist;
mod header;
mod lint;
mod metallibs;
mod smoke;
mod tools;
mod util;

use std::process::ExitCode;
use util::{Res, read, repo_root};

/// One subcommand: its name and what it does.
struct Command {
    name: &'static str,
    about: &'static str,
    run: Runner,
}

/// A command's body: its arguments in, whether it passed out.
type Runner = fn(&[String]) -> Res<bool>;

const COMMANDS: &[Command] = &[
    Command {
        name: "testkit",
        about: "core's testkit module is gated, and only dev-dependencies turn its feature on (D50)",
        run: cmd_testkit,
    },
    Command {
        name: "header",
        about: "regenerate include/reims_vgpu.h with cbindgen; --check fails when the committed header differs from cbindgen's output",
        run: header::cmd_header,
    },
    Command {
        name: "dist",
        about: "the library as a plain library in target/dist: libreims_vgpu.a and .dylib, reims_vgpu.h, reims-vgpu.pc and the option ROM built from firmware/; --profile <p>, --out DIR, --platform vulkan (the proof of concept, in target/dist-vulkan)",
        run: dist::cmd_dist,
    },
    Command {
        name: "metallibs",
        about: "the Vulkan platform's metallibs and their record, metallibs.toml, compiled on a Mac into crates/reims-vgpu-vulkan/metallibs, which a build elsewhere takes them from, or into <dir> (RVG_METALLIB_DIR=<dir>)",
        run: metallibs::cmd_metallibs,
    },
    Command {
        name: "smoke",
        about: "a guest workload linked statically with the dist tree's library, as QEMU links it, against its expected values; --dist <dir>",
        run: smoke::cmd_smoke,
    },
    Command {
        name: "check",
        about: "testkit, header --check, rustfmt, clippy, cargo nextest and the doc-tests; on a Mac, also clippy for x86-64 Linux and, where Rosetta runs x86-64 code, core's tests as x86-64",
        run: cmd_check,
    },
    Command {
        name: "check-ctl",
        about: "rustfmt, clippy and tests for the separate reims-ctl workspace",
        run: cmd_check_ctl,
    },
];

fn usage() {
    eprintln!("usage: cargo xtask <command> [args...]\n\ncommands:");
    for c in COMMANDS {
        eprintln!("  {:<13} {}", c.name, c.about);
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(name) = args.first() else {
        usage();
        return ExitCode::from(2);
    };
    match COMMANDS.iter().find(|c| c.name == name) {
        Some(c) => match (c.run)(args.get(1..).unwrap_or(&[])) {
            Ok(true) => ExitCode::SUCCESS,
            Ok(false) => ExitCode::FAILURE,
            Err(e) => {
                eprintln!("xtask {name}: {e}");
                ExitCode::FAILURE
            }
        },
        None => {
            eprintln!("xtask: unknown command `{name}`");
            usage();
            ExitCode::from(2)
        }
    }
}

fn flag(args: &[String], f: &str) -> bool {
    args.iter().any(|a| a == f)
}

/// Prints a report; true when it has no errors.
fn show(name: &str, rep: &lint::Report) -> bool {
    for e in &rep.errors {
        eprintln!("error: {e}");
    }
    if rep.errors.is_empty() {
        println!("{name}: ok");
    } else {
        eprintln!("{name}: {} errors", rep.errors.len());
    }
    rep.errors.is_empty()
}

// ---------------------------------------------------------------------------------------------

fn cmd_testkit(_args: &[String]) -> Res<bool> {
    let root = repo_root();
    let lib = read(&root.join("crates/reims-vgpu-core/src/lib.rs"))?;
    let mut manifests = Vec::new();
    for p in util::walk(&root.join("crates"), &["Cargo.toml"]) {
        manifests.push((util::rel(&p, &root), read(&p)?));
    }
    Ok(show("testkit", &lint::testkit_manifests(&lib, &manifests)))
}

fn cargo(args: &[&str]) -> Res<bool> {
    cargo_with(args, &[])
}

/// [`cargo`], with `env` added to Cargo's environment.
fn cargo_with(args: &[&str], env: &[(&str, &std::path::Path)]) -> Res<bool> {
    let shown: String = env
        .iter()
        .map(|(k, v)| format!("{k}={} ", v.display()))
        .collect();
    println!("$ {shown}cargo {}", args.join(" "));
    let status =
        std::process::Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
            .args(args)
            .envs(env.iter().copied())
            .current_dir(repo_root())
            .status()
            .map_err(|e| format!("cargo: {e}"))?;
    Ok(status.success())
}

/// The features `check` tests with: every feature of the workspace's crates but the C API's
/// `vulkan`, which makes every handle of the C API the Vulkan platform's, so that the pins and
/// the C harness, which drive the product's C API, would run on that platform instead
/// (architecture 13.1). Clippy lints with every feature as well, the C API's on Vulkan among them.
const TESTED_FEATURES: &str = "reims-vgpu-core/testkit,reims-vgpu/testkit,reims-vgpu/vulkan";

/// The features on Linux, where the Vulkan platform is the only one, the C API's included.
const LINUX_FEATURES: &str =
    "reims-vgpu-core/testkit,reims-vgpu/testkit,reims-vgpu/vulkan,reims-vgpu-c/vulkan";

/// The Linux the library ships for, which a Mac type-checks.
const LINUX_TARGET: &str = "x86_64-unknown-linux-gnu";

/// The features a Mac type-checks Linux with: [`LINUX_FEATURES`], and SDL's sources left
/// unbuilt, as a Mac has no Linux C toolchain for them and nothing links.
const LINUX_CHECK_FEATURES: &str = "reims-vgpu-core/testkit,reims-vgpu/testkit,reims-vgpu/vulkan,\
    reims-vgpu-c/vulkan,reims-vgpu-window/sdl-no-link";

/// The variable jemalloc's build script (`tikv-jemalloc-sys`, the C API's allocator on
/// [`LINUX_TARGET`]) reads for that target as a library to link in place of the one it builds. A
/// Mac's type-check of Linux sets it to an empty archive ([`empty_archive`]), as the Mac has no
/// Linux C toolchain to build jemalloc with and nothing links.
const LINUX_JEMALLOC_OVERRIDE: &str = "X86_64_UNKNOWN_LINUX_GNU_JEMALLOC_OVERRIDE";

/// An archive with no members, `libjemalloc.a` under the target directory, written for
/// [`LINUX_JEMALLOC_OVERRIDE`].
fn empty_archive() -> Res<std::path::PathBuf> {
    let dir = tools::target_dir(&mut tools::System)?.join("linux-check");
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let path = dir.join("libjemalloc.a");
    std::fs::write(&path, b"!<arch>\n").map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(path)
}

/// The macOS core's tests run as under Rosetta, with the x86-64 guest's pages.
const X86_64_MACOS_TARGET: &str = "x86_64-apple-darwin";

/// Whether this Mac runs x86-64 code: an x86-64 `true` run under `arch -x86_64` exits 0, which
/// it does where Rosetta is installed (and on an Intel Mac); without Rosetta the launch fails.
fn runs_x86_64() -> bool {
    std::process::Command::new("/usr/bin/arch")
        .args(["-x86_64", "/usr/bin/true"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

fn cmd_check(args: &[String]) -> Res<bool> {
    let mut ok = true;
    ok &= cmd_testkit(&[])?;
    ok &= header::cmd_header(&["--check".to_owned()])?;
    if !flag(args, "--no-cargo") {
        let features = if cfg!(target_os = "linux") {
            LINUX_FEATURES
        } else {
            TESTED_FEATURES
        };
        ok &= cargo(&["fmt", "--all", "--check"])?;
        let clippy = |more: &[&str], env: &[(&str, &std::path::Path)]| {
            let mut args = vec!["clippy", "--workspace", "--all-targets"];
            args.extend_from_slice(more);
            args.extend(["--", "-D", "warnings"]);
            cargo_with(&args, env)
        };
        // Every feature, SDL3's macOS window among them; then the tested features without the
        // default ones, so the macOS library without SDL (`macos-sdl` off,
        // `native/macos/sdl_off.rs`) is checked too.
        ok &= clippy(&["--all-features"], &[])?;
        ok &= clippy(&["--no-default-features", "--features", features], &[])?;
        if cfg!(target_os = "macos") {
            // Every Linux file, type-checked with no Linux host: build scripts run on the Mac,
            // and nothing links.
            let jemalloc = empty_archive()?;
            ok &= clippy(
                &["--target", LINUX_TARGET, "--features", LINUX_CHECK_FEATURES],
                &[(LINUX_JEMALLOC_OVERRIDE, &jemalloc)],
            )?;
        }
        ok &= cargo(&[
            "nextest",
            "run",
            "--workspace",
            "--features",
            features,
            "--no-fail-fast",
        ])?;
        if cfg!(target_os = "macos") {
            if runs_x86_64() {
                ok &= cargo(&[
                    "nextest",
                    "run",
                    "-p",
                    "reims-vgpu-core",
                    "--target",
                    X86_64_MACOS_TARGET,
                    "--no-fail-fast",
                ])?;
            } else {
                println!(
                    "check: core's tests as {X86_64_MACOS_TARGET} skipped: this Mac runs no \
                     x86-64 code (`softwareupdate --install-rosetta` installs Rosetta)"
                );
            }
        }
        // nextest runs no doc-tests.
        ok &= cargo(&["test", "--workspace", "--features", features, "--doc"])?;
    }
    println!("check: {}", if ok { "ok" } else { "FAILED" });
    Ok(ok)
}

fn cmd_check_ctl(_args: &[String]) -> Res<bool> {
    const MANIFEST: &str = "crates/reims-ctl/Cargo.toml";
    let mut ok = true;
    ok &= cargo(&["fmt", "--manifest-path", MANIFEST, "--all", "--", "--check"])?;
    ok &= cargo(&[
        "clippy",
        "--manifest-path",
        MANIFEST,
        "--all-targets",
        "--",
        "-D",
        "warnings",
    ])?;
    ok &= cargo(&["test", "--manifest-path", MANIFEST])?;
    println!("check-ctl: {}", if ok { "ok" } else { "FAILED" });
    Ok(ok)
}
