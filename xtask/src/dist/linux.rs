//! `cargo xtask dist` on Linux: the library on the Vulkan platform, the only one Linux builds.
//! The shared library is `lib/libreims_vgpu.so`, whose soname is its own name; nothing is signed
//! and nothing has an install name. `Libs.private` is what rustc prints for linking the archive
//! (`--print native-static-libs`). The ROM is built here, by the host's clang, so its bytes are
//! that compiler's, not the Mac's: `dist` prints its sha256 and the clang that built it.
//!
//! The library's shaders are metallibs that only a Mac compiles: the build copies those the
//! repository holds (`crates/reims-vgpu-vulkan/metallibs`), or those in the directory
//! `RVG_METALLIB_DIR` names, which `cargo xtask metallibs <dir>` wrote on a Mac.
//!
//! **The archive's symbols.** The archive bundles what the library links statically: Rust's
//! standard library and SDL3, which draws a display's window. A host that links the archive
//! next to its own SDL (QEMU's `-display sdl`, or an SDL 2 that is sdl2-compat over a shared
//! SDL3) would have its `SDL_*` calls bind to the archive's SDL3, or SDL's own calls bind across
//! the two. So the archive installed is one object, the built archive's members linked into it
//! (`ld -r`), whose global symbols are the C API's `rvg_*` functions alone: `objcopy` makes every
//! other one local ([`localize`]). `dist` then checks the archive installed: any other global it
//! defines fails the build. The shared library needs none of this: rustc's version script
//! exports the C API's functions alone.

use std::path::{Path, PathBuf};

use super::{
    FIRMWARE, LIB, Options, Platform, ROM, c_api_args, firmware_args, install, layout, pc_file,
    strs, strs_with, write,
};
use crate::tools::Tools;
use crate::util::Res;

/// The platform `dist` builds when `--platform` names none: Vulkan, the only one on Linux.
pub const DEFAULT_PLATFORM: Platform = Platform::Vulkan;
/// The shared library's extension: `lib/libreims_vgpu.so`.
pub const SHARED_LIBRARY_EXTENSION: &str = "so";

/// The shared library's soname: its file name in the tree, which a host that links it records.
fn soname() -> String {
    format!("lib{LIB}.so")
}

/// The `cargo rustc` arguments that build `reims-vgpu-c`'s libraries (cdylib and staticlib) in
/// the profile, on the platform. Cargo names the cdylib after the crate, so the link gives it the
/// library's own soname.
pub fn cargo_args(o: &Options) -> Vec<String> {
    let mut args = vec!["rustc".to_owned()];
    args.extend(c_api_args(o));
    args.extend(strs(&["--", "-C"]));
    args.push(format!("link-arg=-Wl,-soname,{}", soname()));
    args
}

/// The `cargo rustc` arguments that make rustc print the native libraries a static link of the
/// archive needs, in a note among Cargo's JSON messages on stdout.
pub fn static_libs_args(o: &Options) -> Vec<String> {
    let mut args = vec!["rustc".to_owned()];
    args.extend(c_api_args(o));
    args.extend(strs(&[
        "--crate-type",
        "staticlib",
        "--message-format",
        "json",
        "--",
        "--print",
        "native-static-libs",
    ]));
    args
}

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

/// Builds and writes the tree with `tools`; `header` is the committed header and `firmware` the
/// option ROM's driver. Returns the output directory.
pub fn dist(
    tools: &mut dyn Tools,
    o: &Options,
    target_dir: &Path,
    header: &Path,
    firmware: &Path,
) -> Res<PathBuf> {
    if o.platform == Platform::Metal {
        return Err(
            "--platform metal: the Metal platform is macOS only; on Linux the library is built \
             on Vulkan, the default"
                .to_owned(),
        );
    }
    // The ROM first: it takes a second, and a host without lld fails before the library builds.
    let firmware_build = target_dir.join(FIRMWARE);
    tools
        .run("make", &firmware_args(firmware, &firmware_build))
        .map_err(|e| format!("{e}: the option ROM did not build (it needs clang and lld)"))?;
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    tools.run(&cargo, &cargo_args(o))?;
    let built = target_dir.join(o.profile.dir());
    let [archive, shared, pc, include, rom] = layout().map(|p| o.out.join(p));

    localize(
        tools,
        &built.join("libreims_vgpu_c.a"),
        &target_dir.join(LOCALIZED),
        &archive,
    )?;
    install(&built.join("libreims_vgpu_c.so"), &shared, 0o755)?;
    install(header, &include, 0o644)?;
    // Asked after the tree has its libraries: this build of the archive alone replaces the one
    // in the target dir, not the one installed.
    let libs = native_static_libs(&tools.output(&cargo, &static_libs_args(o))?)?;
    write(&pc, &pc_file(env!("CARGO_PKG_VERSION"), &libs))?;
    install(&firmware_build.join(ROM), &rom, 0o644)?;
    println!("{}", rom_line(tools, &rom)?);
    Ok(o.out.clone())
}

/// The directory under the target dir the archive is localized in ([`localize`]).
const LOCALIZED: &str = "dist-localized";

/// The prefix of every symbol the archive keeps global: the C API's functions.
const API_PREFIX: &str = "rvg_";

/// Writes `archive`, the archive installed, from `built`, Cargo's, in `work` (module doc): its
/// members linked into one object, every global symbol but the C API's made local, archived
/// alone, with mode 0644. Fails where the object defines no `rvg_` function, or where the archive
/// written still defines another global ([`check_globals`]).
fn localize(tools: &mut dyn Tools, built: &Path, work: &Path, archive: &Path) -> Res<()> {
    std::fs::create_dir_all(work).map_err(|e| format!("{}: {e}", work.display()))?;
    let object = work.join(format!("{LIB}.o"));
    let mut link = strs(&["-r", "--whole-archive"]);
    link.push(built.to_string_lossy().into_owned());
    link.extend(strs_with(&["-o"], &object));
    tools.run("ld", &link)?;
    let api: Vec<String> = defined_globals(&tools.output("nm", &nm_args(&object))?)
        .into_iter()
        .filter(|name| name.starts_with(API_PREFIX))
        .collect();
    if api.is_empty() {
        return Err(format!(
            "{} defines no {API_PREFIX} function: nothing would stay global",
            object.display()
        ));
    }
    let keep = work.join("keep-global-symbols");
    write(&keep, &(api.join("\n") + "\n"))?;
    let mut strip = vec![format!("--keep-global-symbols={}", keep.display())];
    strip.push(object.to_string_lossy().into_owned());
    tools.run("objcopy", &strip)?;
    if std::fs::symlink_metadata(archive).is_ok() {
        std::fs::remove_file(archive).map_err(|e| format!("{}: {e}", archive.display()))?;
    }
    super::mkdir_parent(archive)?;
    let mut pack = strs_with(&["rcs"], archive);
    pack.push(object.to_string_lossy().into_owned());
    tools.run("ar", &pack)?;
    set_mode(archive, 0o644)?;
    check_globals(&tools.output("nm", &nm_args(archive))?, archive)
}

/// `nm`'s arguments that list the global symbols `file` defines, one a line, in POSIX's format:
/// the name, then its type.
fn nm_args(file: &Path) -> Vec<String> {
    strs_with(&["-g", "--defined-only", "-P"], file)
}

/// The names in `nm -g --defined-only -P`'s output, in its order: each line of a name and a
/// one-letter type, past the lines that name an archive's member.
fn defined_globals(nm: &str) -> Vec<String> {
    nm.lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let name = fields.next()?;
            let kind = fields.next()?;
            (kind.len() == 1).then(|| name.to_owned())
        })
        .collect()
}

/// Fails where `nm`, the archive's globals, names any not of the C API, naming the first few.
fn check_globals(nm: &str, archive: &Path) -> Res<()> {
    let others: Vec<String> = defined_globals(nm)
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

/// Sets `file`'s mode.
fn set_mode(file: &Path, mode: u32) -> Res<()> {
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::set_permissions(file, std::fs::Permissions::from_mode(mode))
        .map_err(|e| format!("{}: {e}", file.display()))
}

/// The line that names the ROM a boot used: its sha256 and the clang that built it. The ROM's
/// bytes depend on the compiler, so a ROM built here differs from the Mac's, and from one an
/// earlier clang built.
fn rom_line(tools: &mut dyn Tools, rom: &Path) -> Res<String> {
    let sha = tools.output("sha256sum", &strs_with(&[], rom))?;
    let clang = tools.output("clang", &strs(&["--version"]))?;
    Ok(format!(
        "dist: {ROM} sha256 {}, built by {}",
        sha.split_whitespace().next().unwrap_or_default(),
        clang.lines().next().unwrap_or_default()
    ))
}

#[cfg(test)]
mod tests {
    use super::super::parse_args;
    use super::super::testing::{Log, mode, put, tmp};
    use super::*;
    use crate::tools::Profile;

    /// What `cargo rustc --message-format json -- --print native-static-libs` prints: the note,
    /// once as its message and once rendered, among the other messages.
    const CARGO: &str = r#"{"reason":"compiler-artifact","package_id":"path+file:///r/crates/reims-vgpu-core#0.1.0","filenames":["/t/release/deps/libreims_vgpu_core.rlib"],"fresh":true}
{"reason":"compiler-message","package_id":"path+file:///r/crates/reims-vgpu-c#0.1.0","message":{"rendered":"note: Link against the following native artifacts when linking against this static library. The order and any duplication can be significant on some platforms.\n\n","children":[],"level":"note","message":"Link against the following native artifacts when linking against this static library. The order and any duplication can be significant on some platforms.","spans":[]}}
{"reason":"compiler-message","package_id":"path+file:///r/crates/reims-vgpu-c#0.1.0","message":{"rendered":"note: native-static-libs: -lgcc_s -lutil -lrt -lpthread -lm -ldl -lc\n\n","children":[],"level":"note","message":"native-static-libs: -lgcc_s -lutil -lrt -lpthread -lm -ldl -lc","spans":[]}}
{"reason":"build-finished","success":true}"#;

    fn options(args: &[&str], target: &Path) -> Options {
        parse_args(&strs(args), target).unwrap()
    }

    #[test]
    fn parse_args_defaults_to_vulkan() {
        let o = parse_args(&[], Path::new("/t")).unwrap();
        assert_eq!(o.profile, Profile("release".into()));
        assert_eq!(o.platform, Platform::Vulkan);
        assert_eq!(o.out, PathBuf::from("/t/dist-vulkan"));
    }

    #[test]
    fn cargo_args_build_the_c_api_crate_with_its_soname() {
        assert_eq!(
            cargo_args(&options(&[], Path::new("/t"))).join(" "),
            "rustc --profile release -p reims-vgpu-c --lib --features reims-vgpu-c/vulkan \
             -- -C link-arg=-Wl,-soname,libreims_vgpu.so"
        );
    }

    #[test]
    fn static_libs_args_build_the_archive_alone_and_print_its_libraries() {
        assert_eq!(
            static_libs_args(&options(&["--profile", "dev"], Path::new("/t"))).join(" "),
            "rustc --profile dev -p reims-vgpu-c --lib --features reims-vgpu-c/vulkan \
             --crate-type staticlib --message-format json -- --print native-static-libs"
        );
    }

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

    #[test]
    fn rom_line_names_the_roms_hash_and_compiler() {
        struct Answers;
        impl Tools for Answers {
            fn run(&mut self, _: &str, _: &[String]) -> Res<()> {
                Ok(())
            }
            fn output(&mut self, program: &str, _: &[String]) -> Res<String> {
                Ok(match program {
                    "sha256sum" => "9b4d4a9d  /t/ReimsVgpu.rom".to_owned(),
                    _ => "clang version 23.1.1\nTarget: x86_64-pc-linux-gnu".to_owned(),
                })
            }
        }
        assert_eq!(
            rom_line(&mut Answers, Path::new("/t/ReimsVgpu.rom")).unwrap(),
            "dist: ReimsVgpu.rom sha256 9b4d4a9d, built by clang version 23.1.1"
        );
    }

    #[test]
    fn layout_names_the_shared_object() {
        assert_eq!(layout()[1], "lib/libreims_vgpu.so");
    }

    /// A tree the build would write: the libraries Cargo built, the header and the ROM.
    fn built(name: &str) -> (PathBuf, PathBuf, PathBuf, PathBuf) {
        let root = tmp(name);
        let target = root.join("target");
        put(&target.join("release/libreims_vgpu_c.a"), "archive");
        put(&target.join("release/libreims_vgpu_c.so"), "shared");
        let header = root.join("reims_vgpu.h");
        put(&header, "header");
        // What the firmware's make writes.
        put(&target.join("firmware/ReimsVgpu.rom"), "rom");
        (root.clone(), target, header, root.join("firmware"))
    }

    /// The programs as `dist` runs them: each call recorded; `ar` writes the archive it names,
    /// and `nm` lists `object` for the linked object and `archive` for the archive.
    struct Programs {
        log: Log,
        object: String,
        archive: String,
    }

    impl Programs {
        fn new(object: &str, archive: &str) -> Programs {
            Programs {
                log: Log {
                    answer: CARGO.to_owned(),
                    ..Log::default()
                },
                object: object.to_owned(),
                archive: archive.to_owned(),
            }
        }
    }

    impl Tools for Programs {
        fn run(&mut self, program: &str, args: &[String]) -> Res<()> {
            if program == "ar" {
                put(Path::new(&args[1]), "localized");
            }
            self.log.run(program, args)
        }
        fn output(&mut self, program: &str, args: &[String]) -> Res<String> {
            let answer = self.log.output(program, args)?;
            let last = args.last().map_or("", String::as_str);
            Ok(match program {
                "nm" if last.ends_with(".o") => self.object.clone(),
                "nm" => self.archive.clone(),
                "sha256sum" => "9b4d4a9d  ReimsVgpu.rom".to_owned(),
                "clang" => "clang version 23.1.1".to_owned(),
                _ => answer,
            })
        }
    }

    /// `nm -P`'s listing of the linked object: the C API's functions, Rust's and SDL's.
    const OBJECT: &str = "rvg_graphics_device_create T 1000 20\nrvg_status_message T 1020 8\n\
                          SDL_Init T 2000 40\n_ZN4core3fmt5write17h0E W 3000 10\n";
    /// `nm -P`'s listing of the localized archive: its member, then the C API's functions.
    const LOCALIZED_ARCHIVE: &str = "libreims_vgpu.a[reims_vgpu.o]:\n\
                                     rvg_graphics_device_create T 1000 20\nrvg_status_message T 1020 8\n";

    #[test]
    fn dist_writes_the_tree_with_rustcs_libraries() {
        let (root, target, header, firmware) = built("tree");
        let o = options(&[], &target);
        // A previous tree's files are replaced.
        put(&o.out.join("lib/libreims_vgpu.so"), "old");
        put(&o.out.join("lib/libreims_vgpu.a"), "old");
        let mut tools = Programs::new(OBJECT, LOCALIZED_ARCHIVE);
        let out = dist(&mut tools, &o, &target, &header, &firmware).unwrap();
        assert_eq!(out, target.join("dist-vulkan"));
        let read = |p: &str| std::fs::read_to_string(out.join(p)).unwrap();
        assert_eq!(read("lib/libreims_vgpu.a"), "localized");
        assert_eq!(read("lib/libreims_vgpu.so"), "shared");
        assert_eq!(read("include/reims_vgpu.h"), "header");
        assert_eq!(read("share/reims-vgpu/ReimsVgpu.rom"), "rom");
        assert!(
            read("lib/pkgconfig/reims-vgpu.pc")
                .contains("Libs.private: -lgcc_s -lutil -lrt -lpthread -lm -ldl -lc\n")
        );
        assert_eq!(mode(&out.join("lib/libreims_vgpu.so")), 0o755);
        assert_eq!(mode(&out.join("lib/libreims_vgpu.a")), 0o644);
        assert_eq!(mode(&out.join("share/reims-vgpu/ReimsVgpu.rom")), 0o644);
        let work = target.join("dist-localized");
        let keep = work.join("keep-global-symbols");
        assert_eq!(
            std::fs::read_to_string(&keep).unwrap(),
            "rvg_graphics_device_create\nrvg_status_message\n"
        );
        let object = work.join("reims_vgpu.o");
        let archive = out.join("lib/libreims_vgpu.a");
        let calls = tools.log.lines();
        assert_eq!(
            calls[0],
            format!(
                "make -C {} BUILD={} rom",
                firmware.display(),
                target.join("firmware").display()
            )
        );
        assert!(
            calls[1].ends_with(" -- -C link-arg=-Wl,-soname,libreims_vgpu.so"),
            "{}",
            calls[1]
        );
        assert_eq!(
            calls[2..7],
            [
                format!(
                    "ld -r --whole-archive {} -o {}",
                    target.join("release/libreims_vgpu_c.a").display(),
                    object.display()
                ),
                format!("nm -g --defined-only -P {}", object.display()),
                format!(
                    "objcopy --keep-global-symbols={} {}",
                    keep.display(),
                    object.display()
                ),
                format!("ar rcs {} {}", archive.display(), object.display()),
                format!("nm -g --defined-only -P {}", archive.display()),
            ]
        );
        assert!(
            calls[7].ends_with(" -- --print native-static-libs"),
            "{}",
            calls[7]
        );
        assert_eq!(
            calls[8..],
            [
                format!(
                    "sha256sum {}",
                    out.join("share/reims-vgpu/ReimsVgpu.rom").display()
                ),
                "clang --version".to_owned(),
            ]
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A global the archive still defines besides the C API's fails the dist, naming it
    #[test]
    fn dist_fails_where_the_archive_keeps_another_global() {
        let (root, target, header, firmware) = built("leak");
        let o = options(&[], &target);
        let leaky = "libreims_vgpu.a[reims_vgpu.o]:\nrvg_graphics_device_create T 1000 20\n\
                     SDL_Init T 2000 40\n";
        let mut tools = Programs::new(OBJECT, leaky);
        let e = dist(&mut tools, &o, &target, &header, &firmware).unwrap_err();
        assert!(e.contains("1 global symbols besides the C API's"), "{e}");
        assert!(e.contains("SDL_Init"), "{e}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// An object with no `rvg_` function is refused before anything is made local
    #[test]
    fn dist_refuses_an_archive_without_the_c_api() {
        let (root, target, header, firmware) = built("noapi");
        let o = options(&[], &target);
        let mut tools = Programs::new("SDL_Init T 2000 40\n", LOCALIZED_ARCHIVE);
        let e = dist(&mut tools, &o, &target, &header, &firmware).unwrap_err();
        assert!(e.contains("defines no rvg_ function"), "{e}");
        assert!(
            !tools.log.lines().iter().any(|l| l.starts_with("objcopy")),
            "{:?}",
            tools.log.lines()
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// nm's POSIX lines give their names, past an archive's member lines
    #[test]
    fn defined_globals_reads_nms_names() {
        assert_eq!(
            defined_globals(LOCALIZED_ARCHIVE),
            strs(&["rvg_graphics_device_create", "rvg_status_message"])
        );
        assert_eq!(defined_globals(""), Vec::<String>::new());
    }

    #[test]
    fn dist_refuses_metal_before_building() {
        let (root, target, header, firmware) = built("refusals");
        let mut log = Log::default();
        let metal = options(&["--platform", "metal"], &target);
        let e = dist(&mut log, &metal, &target, &header, &firmware).unwrap_err();
        assert!(e.contains("--platform metal"), "{e}");
        assert!(log.calls.is_empty(), "{:?}", log.calls);
        assert!(!metal.out.exists());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn dist_stops_before_the_library_when_the_rom_does_not_build() {
        let (root, target, header, firmware) = built("norom");
        let o = options(&[], &target);
        let mut log = Log {
            fail: Some("make"),
            ..Log::default()
        };
        let e = dist(&mut log, &o, &target, &header, &firmware).unwrap_err();
        assert!(e.contains("make failed"), "{e}");
        assert!(e.contains("clang and lld"), "{e}");
        assert_eq!(log.calls.len(), 1, "{:?}", log.calls);
        assert!(!o.out.exists());
        std::fs::remove_dir_all(&root).unwrap();
    }
}
