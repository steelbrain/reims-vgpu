//! `cargo xtask dist` on macOS: the library for arm64 macOS [`DEPLOYMENT_TARGET`], on the Vulkan
//! platform over MoltenVK, a build for development and tests. The dylib's install name is
//! `@rpath/libreims_vgpu.dylib`, and it is signed ad hoc again after the name changes.
//! `Libs.private` names every library the dylib links (`otool -L`).
//!
//! **The archive's symbols and classes.** The archive bundles what the library links statically:
//! Rust's standard library and SDL3, which draws a display's window. A host that links the
//! archive next to its own SDL would have its `SDL_*` calls bind to the archive's SDL3: QEMU
//! links Homebrew's SDL2 for `-display sdl`, and lists the archive first. So the archive
//! installed is one object, the built archive's members linked into it (`ld -r`), whose global
//! symbols are the C API's `rvg_*` functions alone: every other one is made a static symbol
//! ([`localize`]). SDL's Objective-C classes are registered by name, whatever their symbols, and
//! a host's SDL2 that is sdl2-compat loads Homebrew's SDL3, whose classes have the same names: so
//! the build gives SDL's classes, and the class method its `NSCursor` category adds, names of
//! the library's own ([`SDL_OBJC_NAMES`], through the C flags of SDL's build). `dist` then checks
//! the archive installed: any other global it defines, or a class that keeps an SDL name, fails
//! the build. The dylib needs no localizing: rustc exports the C API's functions alone.
//!
//! Three things the build must make sure of for that:
//! - SDL's build script reruns when SDL's sources change, never when its C flags do. So `dist`
//!   records the flags SDL was last built with for the profile (`sdl-cflags-<profile>` beside the
//!   object), and drops SDL's build first (`cargo clean -p sdl3-sys`) where they differ.
//! - kache, where a developer's builds run through it (`scripts/internal/dev-env.sh`), caches a
//!   build script's run by the inputs Cargo reruns it for (kache 2.0.0, `src/build_script.rs`),
//!   and SDL's script declares none of the flags. So the run SDL built again would be served from
//!   one with no flags, of any target directory on the host: a dev dist then left SDL's common
//!   symbols global, its objects built for the SDK's macOS (**measured** on the M3 Max). The
//!   build turns kache's build-script cache off (`KACHE_BUILD_SCRIPT_CACHE=0`, an `[env]` entry
//!   that the rustc wrapper and the scripts see), so its scripts run as without kache, and a run
//!   with the library's flags is not served to another build either.
//! - Clang makes a global that C declares with no value a common symbol, which `ld -r` leaves
//!   global: SDL is built with `-fno-common`.
//!
//! The object `ld -r` writes holds no DWARF of its own, only a debug map that names the built
//! archive's members (`<target>/<profile>/libreims_vgpu_c.a`): a debugger finds the library's
//! lines in a host linked with it while that build is the target dir's latest.

use std::path::{Path, PathBuf};

use super::{LIB, Options, c_api_args, install, layout, pc_file, strs, strs_with, write};
use crate::tools::Tools;
use crate::util::Res;

/// The shared library's extension: `lib/libreims_vgpu.dylib`.
pub const SHARED_LIBRARY_EXTENSION: &str = "dylib";
/// The macOS the library is built for (`MACOSX_DEPLOYMENT_TARGET`): rustc's own default for
/// `aarch64-apple-darwin`. The C a dependency compiles (objc2's exception helper, `try_catch.o`)
/// would otherwise be built for the SDK's version (`cc` asks `xcrun --show-sdk-version`), and a
/// host that targets an older macOS, as QEMU does, would get a linker warning for it. A
/// `MACOSX_DEPLOYMENT_TARGET` set in the environment wins.
pub const DEPLOYMENT_TARGET: &str = "11.0";
/// The dylib's install name.
pub const INSTALL_NAME: &str = "@rpath/libreims_vgpu.dylib";

/// The Objective-C names SDL 3.4.18's Cocoa video defines: its classes, then the class method its
/// category adds to `NSCursor`. Each is given [`SDL_OBJC_PREFIX`] in the library's build (module
/// doc).
pub const SDL_OBJC_NAMES: [&str; 13] = [
    "Cocoa_PasteboardDataProvider",
    "SDL3AppDelegate",
    "SDL3Application",
    "SDL3Cocoa_WindowListener",
    "SDL3MessageBoxPresenter",
    "SDL3OpenGLContext",
    "SDL3TranslatorResponder",
    "SDL3View",
    "SDL3Window",
    "SDL3_cocoametalview",
    "SDL_CocoaVideoData",
    "SDL_CocoaWindowData",
    "invisibleCursor",
];
/// The prefix SDL's Objective-C names take in the library's build.
pub const SDL_OBJC_PREFIX: &str = "RVG";
/// The variable whose C flags the C an arm64 macOS build compiles takes (the `cc` crate's), which
/// SDL's CMake build passes on to its Objective-C.
const TARGET_CFLAGS: &str = "CFLAGS_aarch64_apple_darwin";

/// The variables that, set in the environment, would give SDL's build other C flags than
/// [`sdl_cflags`]. The `cc` crate (1.5.1, as locked) reads all four of its target's names and
/// joins their flags: `CFLAGS` first, then `HOST_CFLAGS` (`TARGET_CFLAGS` when cross-compiling),
/// then the underscored [`TARGET_CFLAGS`], ours, and the dashed `CFLAGS_aarch64-apple-darwin`
/// last. A dashed one's flags come after ours, and can undo or redefine them (our `-D` renames
/// among them); an underscored one in the environment replaces ours, as `--config env` sets no
/// variable the environment already holds. The first two come before ours, which then win.
const SHADOWING_CFLAGS: [&str; 2] = ["CFLAGS_aarch64-apple-darwin", TARGET_CFLAGS];

/// The first of [`SHADOWING_CFLAGS`] that `is_set` says the environment holds.
fn shadowing_cflags(is_set: impl Fn(&str) -> bool) -> Option<&'static str> {
    SHADOWING_CFLAGS.into_iter().find(|name| is_set(name))
}

/// The `cargo build` arguments: `reims-vgpu-c`'s libraries (cdylib and staticlib) in the profile,
/// on the platform, for [`DEPLOYMENT_TARGET`], with SDL's C flags ([`sdl_cflags`]) and kache's
/// build-script cache off ([`KACHE_BUILD_SCRIPT_CACHE`]): each an `[env]` entry, which rustc, its
/// wrapper and every build script see, and which a value set in the environment overrides.
pub fn cargo_args(o: &Options) -> Vec<String> {
    let mut args = vec!["build".to_owned()];
    args.extend(c_api_args(o));
    args.push("--config".to_owned());
    args.push(format!(
        "env.MACOSX_DEPLOYMENT_TARGET=\"{DEPLOYMENT_TARGET}\""
    ));
    args.push("--config".to_owned());
    args.push(format!("env.{TARGET_CFLAGS}=\"{}\"", sdl_cflags()));
    args.push("--config".to_owned());
    args.push(format!("env.{KACHE_BUILD_SCRIPT_CACHE}=\"0\""));
    args
}

/// The variable that, `0`, turns kache's build-script cache off (module doc): kache then installs
/// no launcher in front of a script it compiles, and runs a script it installed one for as Cargo
/// would. Unread where the build does not run through kache.
const KACHE_BUILD_SCRIPT_CACHE: &str = "KACHE_BUILD_SCRIPT_CACHE";

/// The C flags SDL's build takes: a `-D` for each of its Objective-C names, giving it
/// [`SDL_OBJC_PREFIX`], and `-fno-common`, as `ld -r` leaves a common symbol (a global SDL
/// declares with no value) global ([`localize`]).
pub fn sdl_cflags() -> String {
    let mut flags: Vec<String> = SDL_OBJC_NAMES
        .iter()
        .map(|name| format!("-D{name}={SDL_OBJC_PREFIX}{name}"))
        .collect();
    flags.push("-fno-common".to_owned());
    flags.join(" ")
}

/// The `cargo clean` arguments that drop SDL's build in the profile, which then builds again with
/// [`sdl_cflags`]: SDL's build script reruns when its sources change, never when its C flags do.
pub fn sdl_clean_args(o: &Options) -> Vec<String> {
    strs(&["clean", "--profile", o.profile.0.as_str(), "-p", "sdl3-sys"])
}

/// `Libs.private` from `otool -L` of the dylib, in otool's order: `-framework X` for each
/// framework under `/System/Library/Frameworks` (`-weak_framework X` for a weak one), and `-lX`
/// for each `/usr/lib/libX[.version].dylib`. The dylib's own install name, `own`, is skipped;
/// anything else is refused, since the library links only the system.
pub fn libs_private(otool: &str, own: &str) -> Res<Vec<String>> {
    let mut libs = Vec::new();
    // The first line names the file otool read.
    for line in otool.lines().skip(1) {
        let Some((path, versions)) = line.trim().split_once(" (") else {
            continue;
        };
        if path == own {
            continue;
        }
        let framework = path
            .strip_prefix("/System/Library/Frameworks/")
            .and_then(|p| p.split_once(".framework/"))
            .map(|(name, _)| name);
        let usr_lib = path
            .strip_prefix("/usr/lib/lib")
            .filter(|f| !f.contains('/') && f.ends_with(".dylib"))
            .and_then(|f| f.split('.').next());
        if let Some(name) = framework {
            let weak = versions.trim_end_matches(')').ends_with(", weak");
            libs.push(
                if weak {
                    "-weak_framework"
                } else {
                    "-framework"
                }
                .to_owned(),
            );
            libs.push(name.to_owned());
        } else if let Some(name) = usr_lib {
            libs.push(format!("-l{name}"));
        } else {
            return Err(format!(
                "the dylib links {path}, which is neither a system framework nor in /usr/lib"
            ));
        }
    }
    if libs.is_empty() {
        return Err(format!("otool -L listed nothing the dylib links:\n{otool}"));
    }
    Ok(libs)
}

/// Builds and writes the tree with `tools`; `header` is the committed header. Returns the output
/// directory.
pub fn dist(tools: &mut dyn Tools, o: &Options, target_dir: &Path, header: &Path) -> Res<PathBuf> {
    if !cfg!(target_arch = "aarch64") {
        return Err("the library is arm64 macOS only".to_owned());
    }
    if let Some(name) = shadowing_cflags(|name| std::env::var_os(name).is_some()) {
        return Err(format!(
            "{name} is set in the environment, and SDL's build would take it in place of the \
             flags that give SDL's Objective-C classes the library's names: unset it"
        ));
    }
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    // SDL builds again where the flags it was last built with here are not today's, or unknown.
    // The stamp is written once the archive is checked, so an SDL the flags did not reach is
    // built again by the next dist.
    let stamp = target_dir
        .join(LOCALIZED)
        .join(format!("sdl-cflags-{}", o.profile.dir()));
    let cflags = sdl_cflags();
    if std::fs::read_to_string(&stamp).ok().as_deref() != Some(cflags.as_str()) {
        tools.run(&cargo, &sdl_clean_args(o))?;
    }
    tools.run(&cargo, &cargo_args(o))?;
    let built = target_dir.join(o.profile.dir());
    let [archive, dylib, pc, include] = layout().map(|p| o.out.join(p));

    localize(
        tools,
        &built.join("libreims_vgpu_c.a"),
        &target_dir.join(LOCALIZED),
        &archive,
    )?;
    write(&stamp, &cflags)?;
    install(&built.join("libreims_vgpu_c.dylib"), &dylib, 0o755)?;
    // Cargo names the cdylib after the crate; the library's install name is its own.
    tools.run(
        "xcrun",
        &strs_with(&["install_name_tool", "-id", INSTALL_NAME], &dylib),
    )?;
    tools.run("codesign", &strs_with(&["-s", "-", "-f"], &dylib))?;
    install(header, &include, 0o644)?;
    let otool = tools.output("xcrun", &strs_with(&["otool", "-L"], &dylib))?;
    let libs = libs_private(&otool, INSTALL_NAME)?;
    write(&pc, &pc_file(env!("CARGO_PKG_VERSION"), &libs))?;
    Ok(o.out.clone())
}

/// The directory under the target dir the archive is localized in ([`localize`]).
const LOCALIZED: &str = "dist-localized";

/// The prefix of every symbol the archive keeps global: the C API's functions, as Mach-O names
/// them.
const API_PREFIX: &str = "_rvg_";

/// Writes `archive`, the archive installed, from `built`, Cargo's, in `work` (module doc): its
/// members linked into one object whose globals are the C API's alone, archived alone, with mode
/// 0644. Fails where the archive written defines no `rvg_` function, another global, or a class
/// that keeps an SDL name ([`check_archive`]).
fn localize(tools: &mut dyn Tools, built: &Path, work: &Path, archive: &Path) -> Res<()> {
    std::fs::create_dir_all(work).map_err(|e| format!("{}: {e}", work.display()))?;
    let exported = work.join("exported-symbols");
    write(&exported, &format!("{API_PREFIX}*\n"))?;
    let object = work.join(format!("{LIB}.o"));
    let mut link = strs(&[
        "ld",
        "-r",
        "-arch",
        "arm64",
        "-platform_version",
        "macos",
        DEPLOYMENT_TARGET,
        DEPLOYMENT_TARGET,
    ]);
    link.extend(strs_with(&["-exported_symbols_list"], &exported));
    link.extend(strs_with(&["-force_load"], built));
    link.extend(strs_with(&["-o"], &object));
    tools.run("xcrun", &link)?;
    let packed = work.join(format!("lib{LIB}.a"));
    if std::fs::symlink_metadata(&packed).is_ok() {
        std::fs::remove_file(&packed).map_err(|e| format!("{}: {e}", packed.display()))?;
    }
    let mut pack = strs_with(&["libtool", "-static", "-o"], &packed);
    pack.push(object.to_string_lossy().into_owned());
    tools.run("xcrun", &pack)?;
    install(&packed, archive, 0o644)?;
    let nm = tools.output(
        "xcrun",
        &strs_with(&["nm", "--defined-only", "-P"], archive),
    )?;
    check_archive(&nm, archive)
}

/// The symbols in `nm --defined-only -P`'s output, in its order: each line of a name and a
/// one-letter type, past the lines that name an archive's member. The type is upper case for a
/// global, lower case for a static symbol.
fn defined_symbols(nm: &str) -> Vec<(&str, char)> {
    nm.lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let name = fields.next()?;
            let mut kind = fields.next()?.chars();
            let letter = kind.next()?;
            kind.next().is_none().then_some((name, letter))
        })
        .collect()
}

/// Fails where `nm`, the archive's symbols, has no C API function, a global not of the C API, or
/// an Objective-C class named as SDL names its own, naming the first few.
fn check_archive(nm: &str, archive: &Path) -> Res<()> {
    let symbols = defined_symbols(nm);
    let globals: Vec<&str> = symbols
        .iter()
        .filter(|(_, kind)| kind.is_ascii_uppercase())
        .map(|&(name, _)| name)
        .collect();
    if !globals.iter().any(|name| name.starts_with(API_PREFIX)) {
        return Err(format!(
            "{} defines no {API_PREFIX} function: the C API is missing",
            archive.display()
        ));
    }
    let first = |names: &[&str]| names.iter().take(8).copied().collect::<Vec<_>>().join(", ");
    let others: Vec<&str> = globals
        .iter()
        .copied()
        .filter(|name| !name.starts_with(API_PREFIX))
        .collect();
    if !others.is_empty() {
        return Err(format!(
            "{} defines {} global symbols besides the C API's, such as {}: a host's own copies \
             would clash with them",
            archive.display(),
            others.len(),
            first(&others)
        ));
    }
    let sdl_named: Vec<&str> = symbols
        .iter()
        .filter_map(|&(name, _)| name.strip_prefix("_OBJC_CLASS_$_"))
        .filter(|class| class.starts_with("SDL") || SDL_OBJC_NAMES.contains(class))
        .collect();
    if !sdl_named.is_empty() {
        return Err(format!(
            "{} defines {} Objective-C classes under SDL's names, such as {}: a host that loads \
             SDL3 would have two of each",
            archive.display(),
            sdl_named.len(),
            first(&sdl_named)
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::testing::{Log, mode, put, tmp};
    use super::super::{DEFAULT_OUT, parse_args, strs};
    use super::*;
    use crate::tools::Profile;

    const OTOOL: &str = "\
/t/dist/lib/libreims_vgpu.dylib:
\t@rpath/libreims_vgpu.dylib (compatibility version 0.0.0, current version 0.0.0)
\t/System/Library/Frameworks/AppKit.framework/Versions/C/AppKit (compatibility version 45.0.0, current version 2685.60.104, weak)
\t/System/Library/Frameworks/Metal.framework/Versions/A/Metal (compatibility version 1.0.0, current version 373.2.0)
\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1356.0.0)
\t/usr/lib/libiconv.2.dylib (compatibility version 7.0.0, current version 7.0.0)
\t/usr/lib/libobjc.A.dylib (compatibility version 1.0.0, current version 228.0.0)
";

    #[test]
    fn parse_args_defaults() {
        let o = parse_args(&[], Path::new("/t")).unwrap();
        assert_eq!(o.profile, Profile("release".into()));
        assert_eq!(o.out, PathBuf::from("/t/dist-vulkan"));
    }

    /// The `--config` entry that renames SDL's Objective-C names.
    /// SDL's C flags, then kache's build-script cache off, as `cargo_args` ends.
    fn renames() -> String {
        format!(
            "--config env.CFLAGS_aarch64_apple_darwin=\"{}\" \
             --config env.KACHE_BUILD_SCRIPT_CACHE=\"0\"",
            sdl_cflags()
        )
    }

    #[test]
    fn cargo_args_build_the_c_api_crate_only() {
        let o = parse_args(&[], Path::new("/t")).unwrap();
        assert_eq!(
            cargo_args(&o).join(" "),
            format!(
                "build --profile release -p reims-vgpu-c --lib \
                 --config env.MACOSX_DEPLOYMENT_TARGET=\"11.0\" {}",
                renames()
            )
        );
    }

    /// Each of SDL's Objective-C names becomes the library's own, the first and last as written,
    /// and SDL has no common symbols
    #[test]
    fn sdl_cflags_prefix_each_name() {
        let flags = sdl_cflags();
        assert_eq!(flags.split(' ').count(), SDL_OBJC_NAMES.len() + 1);
        assert!(flags.starts_with(
            "-DCocoa_PasteboardDataProvider=RVGCocoa_PasteboardDataProvider -DSDL3AppDelegate="
        ));
        assert!(flags.ends_with(" -DinvisibleCursor=RVGinvisibleCursor -fno-common"));
    }

    /// Either name whose flags the `cc` crate joins after the flags dist gives SDL, or reads in
    /// their place, is refused where set; the ones whose flags come before them are not
    #[test]
    fn shadowing_cflags_are_the_targets_own_names() {
        assert_eq!(shadowing_cflags(|_| false), None);
        assert_eq!(
            shadowing_cflags(|n| n == "CFLAGS_aarch64-apple-darwin"),
            Some("CFLAGS_aarch64-apple-darwin")
        );
        assert_eq!(
            shadowing_cflags(|n| n == TARGET_CFLAGS),
            Some(TARGET_CFLAGS)
        );
        assert_eq!(
            shadowing_cflags(|n| n == "CFLAGS" || n == "TARGET_CFLAGS"),
            None
        );
    }

    #[test]
    fn sdl_clean_args_drop_sdls_build_in_the_profile() {
        let o = parse_args(&[], Path::new("/t")).unwrap();
        assert_eq!(
            sdl_clean_args(&o).join(" "),
            "clean --profile release -p sdl3-sys"
        );
    }

    #[test]
    fn libs_private_names_frameworks_and_usr_lib_in_order() {
        assert_eq!(
            libs_private(OTOOL, INSTALL_NAME).unwrap(),
            strs(&[
                "-weak_framework",
                "AppKit",
                "-framework",
                "Metal",
                "-lSystem",
                "-liconv",
                "-lobjc"
            ])
        );
    }

    #[test]
    fn libs_private_refuses_what_is_not_the_system() {
        let otool = format!(
            "{OTOOL}\t/opt/homebrew/lib/libz.1.dylib (compatibility version 1.0.0, current version 1.3.1)\n"
        );
        let e = libs_private(&otool, INSTALL_NAME).unwrap_err();
        assert!(e.contains("/opt/homebrew/lib/libz.1.dylib"), "{e}");
        // The dylib's own install name, under another name, is not its own.
        let e = libs_private(OTOOL, "@rpath/other.dylib").unwrap_err();
        assert!(e.contains("@rpath/libreims_vgpu.dylib"), "{e}");
        assert!(libs_private("/x.dylib:\n", INSTALL_NAME).is_err());
    }

    #[test]
    fn layout_names_the_dylib() {
        assert_eq!(layout()[1], "lib/libreims_vgpu.dylib");
    }

    /// `nm --defined-only -P`'s listing of the localized archive: its member, the C API's
    /// functions, then static symbols, SDL's and a renamed class among them.
    const LOCALIZED_ARCHIVE: &str = "libreims_vgpu.a[reims_vgpu.o]:\n\
        _rvg_graphics_device_create T 1000 20\n_rvg_status_message T 1020 8\n\
        _SDL_Init t 2000 40\n_OBJC_CLASS_$_RVGSDL3Window s 3000 28\n";

    /// The programs as `dist` runs them: each call recorded; `libtool` writes the archive it
    /// names, `nm` lists `nm`, and anything else answers otool's listing.
    struct Programs {
        log: Log,
        nm: String,
    }

    impl Programs {
        fn new(nm: &str) -> Programs {
            Programs {
                log: Log {
                    answer: OTOOL.to_owned(),
                    ..Log::default()
                },
                nm: nm.to_owned(),
            }
        }
    }

    impl Tools for Programs {
        fn run(&mut self, program: &str, args: &[String]) -> Res<()> {
            if args.first().map(String::as_str) == Some("libtool") {
                put(Path::new(&args[3]), "localized");
            }
            self.log.run(program, args)
        }
        fn output(&mut self, program: &str, args: &[String]) -> Res<String> {
            let answer = self.log.output(program, args)?;
            Ok(if args.first().map(String::as_str) == Some("nm") {
                self.nm.clone()
            } else {
                answer
            })
        }
    }

    /// A tree the build would write: the libraries Cargo built and the header.
    fn built(name: &str) -> (PathBuf, PathBuf, PathBuf) {
        let root = tmp(name);
        let target = root.join("target");
        put(&target.join("release/libreims_vgpu_c.a"), "archive");
        put(&target.join("release/libreims_vgpu_c.dylib"), "dylib");
        let header = root.join("reims_vgpu.h");
        put(&header, "header");
        (root.clone(), target, header)
    }

    #[test]
    fn dist_writes_the_tree_and_renames_the_dylib() {
        let (root, target, header) = built("tree");
        let o = parse_args(&[], &target).unwrap();
        // A previous tree's files are replaced.
        put(&o.out.join("lib/libreims_vgpu.a"), "old");
        let mut tools = Programs::new(LOCALIZED_ARCHIVE);
        let out = dist(&mut tools, &o, &target, &header).unwrap();
        assert_eq!(out, target.join(DEFAULT_OUT));
        let read = |p: &str| std::fs::read_to_string(out.join(p)).unwrap();
        assert_eq!(read("lib/libreims_vgpu.a"), "localized");
        assert_eq!(read("lib/libreims_vgpu.dylib"), "dylib");
        assert_eq!(read("include/reims_vgpu.h"), "header");
        assert!(read("lib/pkgconfig/reims-vgpu.pc").contains(
            "Libs.private: -weak_framework AppKit -framework Metal -lSystem -liconv -lobjc\n"
        ));
        assert_eq!(mode(&out.join("lib/libreims_vgpu.dylib")), 0o755);
        assert_eq!(mode(&out.join("lib/libreims_vgpu.a")), 0o644);
        let dylib = out.join("lib/libreims_vgpu.dylib");
        let d = dylib.to_str().unwrap();
        let work = target.join("dist-localized");
        let exported = work.join("exported-symbols");
        assert_eq!(std::fs::read_to_string(&exported).unwrap(), "_rvg_*\n");
        let object = work.join("reims_vgpu.o");
        let packed = work.join("libreims_vgpu.a");
        let archive = out.join("lib/libreims_vgpu.a");
        let calls = tools.log.lines();
        // SDL builds again, as no flags are known for it here.
        assert!(
            calls[0].ends_with(" clean --profile release -p sdl3-sys"),
            "{}",
            calls[0]
        );
        assert!(calls[1].ends_with(&format!(
            " build --profile release -p reims-vgpu-c --lib \
             --config env.MACOSX_DEPLOYMENT_TARGET=\"11.0\" {}",
            renames()
        )));
        assert_eq!(
            std::fs::read_to_string(work.join("sdl-cflags-release")).unwrap(),
            sdl_cflags()
        );
        assert_eq!(
            calls[2..],
            [
                format!(
                    "xcrun ld -r -arch arm64 -platform_version macos 11.0 11.0 \
                     -exported_symbols_list {} -force_load {} -o {}",
                    exported.display(),
                    target.join("release/libreims_vgpu_c.a").display(),
                    object.display()
                ),
                format!(
                    "xcrun libtool -static -o {} {}",
                    packed.display(),
                    object.display()
                ),
                format!("xcrun nm --defined-only -P {}", archive.display()),
                format!("xcrun install_name_tool -id @rpath/libreims_vgpu.dylib {d}"),
                format!("codesign -s - -f {d}"),
                format!("xcrun otool -L {d}"),
            ]
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// SDL is not built again where it was built with today's flags, and is where with others
    #[test]
    fn dist_rebuilds_sdl_only_for_other_flags() {
        let (root, target, header) = built("stamp");
        let o = parse_args(&[], &target).unwrap();
        let stamp = target.join("dist-localized/sdl-cflags-release");
        let cleans = |tools: &Programs| {
            tools
                .log
                .lines()
                .iter()
                .filter(|call| call.ends_with(" clean --profile release -p sdl3-sys"))
                .count()
        };
        put(&stamp, &sdl_cflags());
        let mut tools = Programs::new(LOCALIZED_ARCHIVE);
        dist(&mut tools, &o, &target, &header).unwrap();
        assert_eq!(cleans(&tools), 0);
        put(&stamp, "-DSDL3Window=OtherSDL3Window");
        let mut tools = Programs::new(LOCALIZED_ARCHIVE);
        dist(&mut tools, &o, &target, &header).unwrap();
        assert_eq!(cleans(&tools), 1);
        assert_eq!(std::fs::read_to_string(&stamp).unwrap(), sdl_cflags());
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A global the archive still defines besides the C API's fails the dist, naming it
    #[test]
    fn dist_fails_where_the_archive_keeps_another_global() {
        let (root, target, header) = built("leak");
        let o = parse_args(&[], &target).unwrap();
        let leaky = "libreims_vgpu.a[reims_vgpu.o]:\n_rvg_graphics_device_create T 1000 20\n\
                     _SDL_Init T 2000 40\n";
        let mut tools = Programs::new(leaky);
        let e = dist(&mut tools, &o, &target, &header).unwrap_err();
        assert!(e.contains("1 global symbols besides the C API's"), "{e}");
        assert!(e.contains("_SDL_Init"), "{e}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A class that keeps an SDL name, static or not, fails the dist, naming it
    #[test]
    fn dist_fails_where_a_class_keeps_sdls_name() {
        let (root, target, header) = built("class");
        let o = parse_args(&[], &target).unwrap();
        for class in ["SDL3Window", "Cocoa_PasteboardDataProvider", "SDL4Window"] {
            let named = format!(
                "libreims_vgpu.a[reims_vgpu.o]:\n_rvg_graphics_device_create T 1000 20\n\
                 _OBJC_CLASS_$_{class} s 3000 28\n"
            );
            let mut tools = Programs::new(&named);
            let e = dist(&mut tools, &o, &target, &header).unwrap_err();
            assert!(e.contains("1 Objective-C classes under SDL's names"), "{e}");
            assert!(e.contains(class), "{e}");
        }
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// An archive with no `rvg_` function is refused
    #[test]
    fn dist_refuses_an_archive_without_the_c_api() {
        let (root, target, header) = built("noapi");
        let o = parse_args(&[], &target).unwrap();
        let mut tools = Programs::new("libreims_vgpu.a[reims_vgpu.o]:\n_SDL_Init t 2000 40\n");
        let e = dist(&mut tools, &o, &target, &header).unwrap_err();
        assert!(e.contains("defines no _rvg_ function"), "{e}");
        std::fs::remove_dir_all(&root).unwrap();
    }
}
