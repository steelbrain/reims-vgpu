//! `cargo xtask dist` on Windows, run in MSYS2's CLANG64 environment: the library for
//! `x86_64-pc-windows-gnullvm` ([`TARGET`]), the target whose archive QEMU's MinGW build links
//! (docs/qemu.md, "Windows hosts"). The archive is rustc's staticlib for that target with its
//! objects' symbols renamed ([`localize`]); the shared library, `lib/libreims_vgpu.dll`, is linked
//! from that archive and exports the C API's functions alone (a module-definition file names
//! them). `Libs.private` is what rustc prints for linking the archive
//! (`--print native-static-libs`).
//!
//! No import library is installed, so a host links the archive: `-lreims_vgpu` takes
//! `libreims_vgpu.a` in lld's and GNU ld's MinGW search, which take `libreims_vgpu.dll.a` before
//! it where there is one, and so does meson's static lookup (QEMU's `static: true` dependency),
//! as `libreims_vgpu.a` comes first among its static names.
//!
//! The library's shaders are metallibs that only a Mac compiles: the build copies those the
//! repository holds (`crates/reims-vgpu-vulkan/metallibs`), or those in the directory
//! `RVG_METALLIB_DIR` names.
//!
//! **The archive's symbols.** As on Linux (`linux.rs`), the archive bundles Rust's standard
//! library and SDL3, and a host that links it beside its own SDL (QEMU's `-display sdl`, SDL 2)
//! would have its `SDL_*` calls bind to the archive's SDL3. Linux links the archive's objects into
//! one and makes every global but the C API's local. That does not work for COFF, measured on a
//! toolchain of 2026-11 (LLVM's tools of MSYS2's CLANG64, and its GNU binutils 2.47): lld makes
//! no relocatable object, and GNU `ld -r` resolves the objects' COMDAT sections, leaving static
//! symbols in section 0 and relocations to symbols it dropped, which lld refuses
//! (`GCC_except_table36 should not refer to special section 0`). So the objects stay as built,
//! and `llvm-objcopy --redefine-syms` gives every global they define but the C API's functions
//! the suffix [`RENAMED`] (`SDL_Init.rvg`), in each object that defines or uses it, and its
//! `__imp_` name with it: no name of the host's own (a C name, or another Rust library's) is one
//! of them, while each of the archive's references binds where it bound before. Of the archive's
//! 2561 members, 729 are objects (67176 globals, 3203 of them SDL's); the other 1832 are
//! rustc's raw-dylib imports, short import members named after their DLL, which stay as they
//! were. `dist` checks that the objects define no other global but the C API's functions.

use std::path::{Path, PathBuf};

use std::collections::BTreeSet;

use super::archive::{
    API_PREFIX, LOCALIZED, api_functions, check_globals, defined_globals, native_static_libs,
    nm_args,
};
use super::{LIB, Options, c_api_args, install, layout, pc_file, strs, strs_with, write};
use crate::tools::Tools;
use crate::util::Res;

/// The shared library's extension, `dll`: `lib/libreims_vgpu.dll`.
pub const SHARED_LIBRARY_EXTENSION: &str = "dll";

/// The target the library is built for: MinGW's, with LLVM's runtime, which QEMU's Windows build
/// links.
pub const TARGET: &str = "x86_64-pc-windows-gnullvm";

/// LLVM's archiver, symbol lister and object copier, as MSYS2's CLANG64 environment names them.
const AR: &str = "llvm-ar";
const NM: &str = "llvm-nm";
const OBJCOPY: &str = "llvm-objcopy";
/// The C compiler driver that links the shared library: the one rustc links [`TARGET`] with.
const CC: &str = "x86_64-w64-mingw32-clang";

/// The suffix of every global the archive's objects define but the C API's functions (module
/// doc). A `.` is in no C name.
const RENAMED: &str = ".rvg";

/// The `cargo rustc` arguments that build `reims-vgpu-c`'s archive for [`TARGET`] in the
/// profile, with rustc's note of the native libraries a static link of it needs.
pub fn cargo_args(o: &Options) -> Vec<String> {
    let mut args = vec!["rustc".to_owned()];
    args.extend(c_api_args(o));
    args.extend(strs(&[
        "--target",
        TARGET,
        "--crate-type",
        "staticlib",
        "--",
        "--print",
        "native-static-libs",
    ]));
    args
}

/// [`cargo_args`] with Cargo's JSON messages on stdout: the same build, which Cargo finds fresh
/// and whose messages, rustc's note among them, it prints again.
pub fn static_libs_args(o: &Options) -> Vec<String> {
    let mut args = cargo_args(o);
    let rustc = args.iter().position(|a| a == "--").unwrap_or(args.len());
    args.splice(rustc..rustc, strs(&["--message-format", "json"]));
    args
}

/// Builds and writes the tree with `tools`; `header` is the committed header. Returns the output
/// directory.
pub fn dist(tools: &mut dyn Tools, o: &Options, target_dir: &Path, header: &Path) -> Res<PathBuf> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    tools.run(&cargo, &cargo_args(o))?;
    let built = target_dir.join(TARGET).join(o.profile.dir());
    let [archive, shared, pc, include] = layout().map(|p| o.out.join(p));
    let work = target_dir.join(LOCALIZED);

    let api = localize(tools, &built.join("libreims_vgpu_c.a"), &work, &archive)?;
    let libs = native_static_libs(&tools.output(&cargo, &static_libs_args(o))?)?;
    link_shared(tools, &archive, &api, &libs, &work, &shared)?;
    install(header, &include, 0o644)?;
    write(&pc, &pc_file(env!("CARGO_PKG_VERSION"), &libs))?;
    Ok(o.out.clone())
}

/// Whether an archive's member is one of rustc's raw-dylib import members, which it names after
/// their DLL.
fn is_import(member: &str) -> bool {
    member.to_ascii_lowercase().ends_with(".dll")
}

/// Writes `archive`, the archive installed, from `built`, Cargo's, in `work` (module doc): its
/// objects with every global but the C API's renamed, after its import members as built, and
/// installed with mode 0644. `llvm-ar` and `llvm-objcopy` keep the members' zero dates, so the
/// same build gives the same bytes and leaves the installed archive as it was. Returns the C
/// API's functions. Fails where the objects define no `rvg_` function, or where the archive
/// written still defines another global outside its import members.
fn localize(tools: &mut dyn Tools, built: &Path, work: &Path, archive: &Path) -> Res<Vec<String>> {
    std::fs::create_dir_all(work).map_err(|e| format!("{}: {e}", work.display()))?;
    let members = tools.output(AR, &strs_with(&["t"], built))?;
    let (imports, objects): (Vec<&str>, Vec<&str>) = members.lines().partition(|m| is_import(m));

    let built_objects = work.join("objects.a");
    copy(built, &built_objects)?;
    remove_members(tools, &built_objects, &imports, &work.join("imports.rsp"))?;
    let listing = tools.output(NM, &nm_args(&built_objects))?;
    let api = api_functions(&listing, &built_objects)?;
    let renames = work.join("rename-symbols");
    write(&renames, &rename_list(&defined_globals(&listing)))?;
    let renamed = work.join("renamed.a");
    let mut rename = vec![format!("--redefine-syms={}", renames.display())];
    rename.push(built_objects.to_string_lossy().into_owned());
    rename.push(renamed.to_string_lossy().into_owned());
    tools.run(OBJCOPY, &rename)?;

    let packed = work.join(format!("lib{LIB}.a"));
    copy(built, &packed)?;
    remove_members(tools, &packed, &objects, &work.join("objects.rsp"))?;
    let mut pack = strs_with(&["qLsD"], &packed);
    pack.push(renamed.to_string_lossy().into_owned());
    tools.run(AR, &pack)?;
    install(&packed, archive, 0o644)?;
    let listing = tools.output(NM, &nm_args(archive))?;
    let globals = member_globals(&listing)
        .into_iter()
        .filter(|(member, name)| !is_import(member) && !name.ends_with(RENAMED))
        .map(|(_, name)| name);
    check_globals(archive, globals)?;
    Ok(api)
}

/// `llvm-objcopy --redefine-syms`' list for `globals`, the globals the archive's objects define:
/// each but the C API's functions, once, with its [`RENAMED`] name, then its `__imp_` name with
/// that name's, a line each, in name order.
fn rename_list(globals: &[String]) -> String {
    let names: BTreeSet<&str> = globals
        .iter()
        .map(String::as_str)
        .filter(|name| !name.starts_with(API_PREFIX))
        .collect();
    let mut list = String::new();
    for name in names {
        list.push_str(&format!(
            "{name} {name}{RENAMED}\n__imp_{name} __imp_{name}{RENAMED}\n"
        ));
    }
    list
}

/// Copies Cargo's archive to `to`, a file of `work` that `llvm-ar` then changes.
fn copy(built: &Path, to: &Path) -> Res<()> {
    std::fs::copy(built, to)
        .map(drop)
        .map_err(|e| format!("{} to {}: {e}", built.display(), to.display()))
}

/// Removes `names` from `archive` with `llvm-ar d`, one member for each time a name is listed, so
/// an archive's members of the same name all go. The names go in a response file, `rsp`: an
/// archive's objects are more than a command line holds.
fn remove_members(tools: &mut dyn Tools, archive: &Path, names: &[&str], rsp: &Path) -> Res<()> {
    let mut list = names.join("\n");
    list.push('\n');
    write(rsp, &list)?;
    let mut args = strs_with(&["d"], archive);
    args.push(format!("@{}", rsp.display()));
    tools.run(AR, &args)
}

/// The global symbols an archive's members define, from `llvm-nm -g --defined-only -P`'s listing
/// of it, as (member, name) in its order. Each member's lines follow a line of its name and a
/// colon, `member:` or `archive[member]:`.
fn member_globals(nm: &str) -> Vec<(String, String)> {
    let mut member = String::new();
    let mut globals = Vec::new();
    for line in nm.lines() {
        if let [name] = defined_globals(line).as_slice() {
            globals.push((member.clone(), name.clone()));
        } else if let Some(head) = line.trim().strip_suffix(':') {
            member = match head.strip_suffix(']').and_then(|h| h.split_once('[')) {
                Some((_, inner)) => inner.to_owned(),
                None => head.to_owned(),
            };
        }
    }
    globals
}

/// Links `shared`, the shared library installed, from `archive`, the archive installed, with the
/// native libraries `libs`, in `work`: a module-definition file there exports `api`, the C API's
/// functions, alone, and the link writes no timestamp, so the same build gives the same bytes.
/// The exports pull the object in; the archive is not linked whole, as its import members hold
/// one import descriptor of a DLL for each crate that imports from it, which a whole link would
/// define more than once (measured: `duplicate symbol: __IMPORT_DESCRIPTOR_kernel32`).
fn link_shared(
    tools: &mut dyn Tools,
    archive: &Path,
    api: &[String],
    libs: &[String],
    work: &Path,
    shared: &Path,
) -> Res<()> {
    let def = work.join(format!("{LIB}.def"));
    write(&def, &format!("EXPORTS\n{}\n", api.join("\n")))?;
    let linked = work.join(format!("lib{LIB}.dll"));
    let mut args = strs_with(&["-shared", "-o"], &linked);
    args.push(def.to_string_lossy().into_owned());
    args.push("-Wl,--no-insert-timestamp".to_owned());
    args.push(archive.to_string_lossy().into_owned());
    args.extend(libs.iter().cloned());
    tools.run(CC, &args)?;
    install(&linked, shared, 0o755)
}

#[cfg(test)]
mod tests {
    use super::super::parse_args;
    use super::super::testing::{LONG_AGO, Log, age, mtime, put, tmp};
    use super::*;

    /// What `cargo rustc --message-format json ... -- --print native-static-libs` prints for
    /// [`TARGET`]: the note, as rustc printed it for the library on 2026-11-15, among the other
    /// messages.
    const CARGO: &str = r#"{"reason":"compiler-artifact","package_id":"path+file:///C:/r/crates/reims-vgpu-vulkan#0.1.0","fresh":true}
{"reason":"compiler-message","package_id":"path+file:///C:/r/crates/reims-vgpu-c#0.1.0","message":{"rendered":"note: native-static-libs: -lm -lkernel32 -luser32 -lgdi32 -lwinmm -limm32 -lole32 -loleaut32 -lversion -luuid -ladvapi32 -lsetupapi -lshell32 -lkernel32 -lntdll -luserenv -lws2_32 -ldbghelp -lsynchronization\n\n","children":[],"level":"note","message":"native-static-libs: -lm -lkernel32 -luser32 -lgdi32 -lwinmm -limm32 -lole32 -loleaut32 -lversion -luuid -ladvapi32 -lsetupapi -lshell32 -lkernel32 -lntdll -luserenv -lws2_32 -ldbghelp -lsynchronization","spans":[]}}
{"reason":"build-finished","success":true}"#;

    /// The members `llvm-ar t` lists for Cargo's archive: objects, and import members that share
    /// their DLL's name.
    const MEMBERS: &str = "std-0a.std.0b-cgu.0.rcgu.o\nbcryptprimitives.dll\n\
                           reims_vgpu_c-1c.0d-cgu.0.rcgu.o\nbcryptprimitives.dll\nSDL.c.obj\n";

    fn options(args: &[&str], target: &Path) -> Options {
        parse_args(&strs(args), target).unwrap()
    }

    #[test]
    fn cargo_args_build_the_archive_for_gnullvm_and_print_its_libraries() {
        assert_eq!(
            cargo_args(&options(&[], Path::new("t"))).join(" "),
            "rustc --profile release -p reims-vgpu-c --lib \
             --target x86_64-pc-windows-gnullvm --crate-type staticlib \
             -- --print native-static-libs"
        );
    }

    /// The JSON build is the same build, so Cargo finds it fresh
    #[test]
    fn static_libs_args_are_the_build_with_json_messages() {
        let o = options(&["--profile", "dev"], Path::new("t"));
        let mut json = static_libs_args(&o);
        assert_eq!(
            json.join(" "),
            "rustc --profile dev -p reims-vgpu-c --lib \
             --target x86_64-pc-windows-gnullvm --crate-type staticlib \
             --message-format json -- --print native-static-libs"
        );
        json.retain(|a| a != "--message-format" && a != "json");
        assert_eq!(json, cargo_args(&o));
    }

    #[test]
    fn layout_names_the_dll() {
        assert_eq!(layout()[1], "lib/libreims_vgpu.dll");
    }

    /// A tree the build would write: the archive Cargo built for the target, and the header.
    fn built(name: &str) -> (PathBuf, PathBuf, PathBuf) {
        let root = tmp(name);
        let target = root.join("target");
        put(
            &target.join("x86_64-pc-windows-gnullvm/release/libreims_vgpu_c.a"),
            "archive",
        );
        let header = root.join("reims_vgpu.h");
        put(&header, "header");
        (root.clone(), target, header)
    }

    /// The programs as `dist` runs them: each call recorded; `llvm-ar qLsD` writes the archive it
    /// names and the compiler driver the library it links; `llvm-nm` lists `objects` for the
    /// built archive's objects and `archive` for the archive installed.
    struct Programs {
        log: Log,
        objects: String,
        archive: String,
    }

    impl Programs {
        fn new(objects: &str, archive: &str) -> Programs {
            Programs {
                log: Log {
                    answer: CARGO.to_owned(),
                    ..Log::default()
                },
                objects: objects.to_owned(),
                archive: archive.to_owned(),
            }
        }
    }

    impl Tools for Programs {
        fn run(&mut self, program: &str, args: &[String]) -> Res<()> {
            match (program, args[0].as_str()) {
                (AR, "qLsD") => put(Path::new(&args[1]), "localized"),
                (CC, _) => put(Path::new(&args[2]), "dll"),
                _ => {}
            }
            self.log.run(program, args)
        }
        fn output(&mut self, program: &str, args: &[String]) -> Res<String> {
            let answer = self.log.output(program, args)?;
            let last = args.last().map_or("", String::as_str);
            Ok(match program {
                AR => MEMBERS.to_owned(),
                NM if last.ends_with("objects.a") => self.objects.clone(),
                NM => self.archive.clone(),
                _ => answer,
            })
        }
    }

    /// `llvm-nm -P`'s listing of the built archive's objects: the C API's functions, Rust's and
    /// SDL's, a COMDAT function that two objects define.
    const OBJECTS: &str = "reims_vgpu_c-1c.0d-cgu.0.rcgu.o:\n\
                           rvg_graphics_device_create T 1000 20\nrvg_status_message T 1020 8\n\
                           _ZN4core3fmt5write17h0E T 3000 10\n\
                           std-0a.std.0b-cgu.0.rcgu.o:\n_ZN4core3fmt5write17h0E T 0 10\n\
                           SDL.c.obj:\nSDL_Init T 2000 40\n";
    /// `llvm-nm -P`'s listing of the archive installed: the import members' symbols, then the
    /// objects': the C API's functions and the renamed globals.
    const LOCALIZED_ARCHIVE: &str = "bcryptprimitives.dll:\n__imp_ProcessPrng T 0 0\n\
                                     ProcessPrng T 0 0\n\
                                     bcryptprimitives.dll:\n\
                                     __IMPORT_DESCRIPTOR_bcryptprimitives D 0 0\n\
                                     reims_vgpu_c-1c.0d-cgu.0.rcgu.o:\n\
                                     rvg_graphics_device_create T 1000 20\n\
                                     rvg_status_message T 1020 8\n\
                                     _ZN4core3fmt5write17h0E.rvg T 3000 10\n\
                                     SDL.c.obj:\nSDL_Init.rvg T 2000 40\n";

    #[test]
    fn dist_writes_the_tree_with_rustcs_libraries() {
        let (root, target, header) = built("tree");
        let o = options(&[], &target);
        // A previous tree's files are replaced.
        put(&o.out.join("lib/libreims_vgpu.dll"), "old");
        put(&o.out.join("lib/libreims_vgpu.a"), "old");
        let mut tools = Programs::new(OBJECTS, LOCALIZED_ARCHIVE);
        let out = dist(&mut tools, &o, &target, &header).unwrap();
        assert_eq!(out, target.join("dist-vulkan"));
        let read = |p: &str| std::fs::read_to_string(out.join(p)).unwrap();
        assert_eq!(read("lib/libreims_vgpu.a"), "localized");
        assert_eq!(read("lib/libreims_vgpu.dll"), "dll");
        assert_eq!(read("include/reims_vgpu.h"), "header");
        assert!(read("lib/pkgconfig/reims-vgpu.pc").contains(
            "Libs.private: -lm -lkernel32 -luser32 -lgdi32 -lwinmm -limm32 -lole32 -loleaut32 \
             -lversion -luuid -ladvapi32 -lsetupapi -lshell32 -lkernel32 -lntdll -luserenv \
             -lws2_32 -ldbghelp -lsynchronization\n"
        ));
        assert!(!out.join("lib/libreims_vgpu.dll.a").exists());

        let work = target.join("dist-localized");
        let work_file = |name: &str| std::fs::read_to_string(work.join(name)).unwrap();
        assert_eq!(
            work_file("imports.rsp"),
            "bcryptprimitives.dll\nbcryptprimitives.dll\n"
        );
        assert_eq!(
            work_file("objects.rsp"),
            "std-0a.std.0b-cgu.0.rcgu.o\nreims_vgpu_c-1c.0d-cgu.0.rcgu.o\nSDL.c.obj\n"
        );
        assert_eq!(
            work_file("rename-symbols"),
            "SDL_Init SDL_Init.rvg\n__imp_SDL_Init __imp_SDL_Init.rvg\n\
             _ZN4core3fmt5write17h0E _ZN4core3fmt5write17h0E.rvg\n\
             __imp__ZN4core3fmt5write17h0E __imp__ZN4core3fmt5write17h0E.rvg\n"
        );
        assert_eq!(
            work_file("reims_vgpu.def"),
            "EXPORTS\nrvg_graphics_device_create\nrvg_status_message\n"
        );

        let built = target
            .join(TARGET)
            .join("release")
            .join("libreims_vgpu_c.a");
        let (objects, renamed) = (work.join("objects.a"), work.join("renamed.a"));
        let packed = work.join("libreims_vgpu.a");
        let archive = out.join("lib/libreims_vgpu.a");
        let calls = tools.log.lines();
        assert!(
            calls[0].ends_with(" --crate-type staticlib -- --print native-static-libs"),
            "{}",
            calls[0]
        );
        assert_eq!(
            calls[1..8],
            [
                format!("llvm-ar t {}", built.display()),
                format!(
                    "llvm-ar d {} @{}",
                    objects.display(),
                    work.join("imports.rsp").display()
                ),
                format!("llvm-nm -g --defined-only -P {}", objects.display()),
                format!(
                    "llvm-objcopy --redefine-syms={} {} {}",
                    work.join("rename-symbols").display(),
                    objects.display(),
                    renamed.display()
                ),
                format!(
                    "llvm-ar d {} @{}",
                    packed.display(),
                    work.join("objects.rsp").display()
                ),
                format!("llvm-ar qLsD {} {}", packed.display(), renamed.display()),
                format!("llvm-nm -g --defined-only -P {}", archive.display()),
            ]
        );
        assert!(
            calls[8].ends_with(" --message-format json -- --print native-static-libs"),
            "{}",
            calls[8]
        );
        assert_eq!(
            calls[9],
            format!(
                "x86_64-w64-mingw32-clang -shared -o {} {} -Wl,--no-insert-timestamp {} \
                 -lm -lkernel32 -luser32 -lgdi32 -lwinmm -limm32 -lole32 -loleaut32 -lversion \
                 -luuid -ladvapi32 -lsetupapi -lshell32 -lkernel32 -lntdll -luserenv -lws2_32 \
                 -ldbghelp -lsynchronization",
                work.join("libreims_vgpu.dll").display(),
                work.join("reims_vgpu.def").display(),
                archive.display()
            )
        );
        assert_eq!(calls.len(), 10, "{calls:?}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A dist of the same build again leaves every file of the tree as it was, mtime included,
    /// so what is built from the tree is not built again
    #[test]
    fn dist_again_keeps_the_tree() {
        let (root, target, header) = built("again");
        let o = options(&[], &target);
        let run = || {
            let mut tools = Programs::new(OBJECTS, LOCALIZED_ARCHIVE);
            dist(&mut tools, &o, &target, &header).unwrap();
        };
        run();
        for p in layout() {
            age(&o.out.join(p));
        }
        run();
        for p in layout() {
            assert_eq!(mtime(&o.out.join(&p)), LONG_AGO, "{p}");
        }
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A global an object still defines under its own name besides the C API's fails the dist,
    /// naming it, before the shared library is linked; the import members' globals are theirs
    #[test]
    fn dist_fails_where_the_archive_keeps_another_global() {
        let (root, target, header) = built("leak");
        let o = options(&[], &target);
        let leaky = "bcryptprimitives.dll:\n__imp_ProcessPrng T 0 0\n\
                     libreims_vgpu.a[SDL.c.obj]:\nrvg_graphics_device_create T 1000 20\n\
                     _ZN4core3fmt5write17h0E.rvg T 3000 10\nSDL_Init T 2000 40\n";
        let mut tools = Programs::new(OBJECTS, leaky);
        let e = dist(&mut tools, &o, &target, &header).unwrap_err();
        assert!(e.contains("1 global symbols besides the C API's"), "{e}");
        assert!(e.contains("SDL_Init"), "{e}");
        assert!(
            !tools.log.lines().iter().any(|l| l.starts_with(CC)),
            "{:?}",
            tools.log.lines()
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Objects with no `rvg_` function are refused before anything is renamed
    #[test]
    fn dist_refuses_an_archive_without_the_c_api() {
        let (root, target, header) = built("noapi");
        let o = options(&[], &target);
        let mut tools = Programs::new("SDL.c.obj:\nSDL_Init T 2000 40\n", LOCALIZED_ARCHIVE);
        let e = dist(&mut tools, &o, &target, &header).unwrap_err();
        assert!(e.contains("defines no rvg_ function"), "{e}");
        assert!(
            !tools.log.lines().iter().any(|l| l.contains("objcopy")),
            "{:?}",
            tools.log.lines()
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Import members are the DLL-named ones, in any case
    #[test]
    fn is_import_takes_dll_names() {
        assert!(is_import("bcryptprimitives.dll"));
        assert!(is_import("KERNEL32.DLL"));
        assert!(!is_import("std-0a.std.0b-cgu.0.rcgu.o"));
        assert!(!is_import("dll.o"));
    }

    /// Each global but the C API's is renamed once, with its `__imp_` name, in name order
    #[test]
    fn rename_list_renames_every_global_but_the_c_apis_once() {
        let globals = strs(&["SDL_Quit", "rvg_status_message", "SDL_Init", "SDL_Quit"]);
        assert_eq!(
            rename_list(&globals),
            "SDL_Init SDL_Init.rvg\n__imp_SDL_Init __imp_SDL_Init.rvg\n\
             SDL_Quit SDL_Quit.rvg\n__imp_SDL_Quit __imp_SDL_Quit.rvg\n"
        );
        assert_eq!(rename_list(&strs(&["rvg_status_message"])), "");
    }

    /// Each global goes with the member whose line heads it, in either form llvm-nm writes
    #[test]
    fn member_globals_reads_each_members_names() {
        let pairs = |nm: &str| -> Vec<String> {
            member_globals(nm)
                .into_iter()
                .map(|(m, n)| format!("{m} {n}"))
                .collect()
        };
        assert_eq!(
            pairs(LOCALIZED_ARCHIVE),
            strs(&[
                "bcryptprimitives.dll __imp_ProcessPrng",
                "bcryptprimitives.dll ProcessPrng",
                "bcryptprimitives.dll __IMPORT_DESCRIPTOR_bcryptprimitives",
                "reims_vgpu_c-1c.0d-cgu.0.rcgu.o rvg_graphics_device_create",
                "reims_vgpu_c-1c.0d-cgu.0.rcgu.o rvg_status_message",
                "reims_vgpu_c-1c.0d-cgu.0.rcgu.o _ZN4core3fmt5write17h0E.rvg",
                "SDL.c.obj SDL_Init.rvg",
            ])
        );
        assert_eq!(
            pairs("libreims_vgpu.a[reims_vgpu.o]:\nrvg_a T 0 0\n"),
            strs(&["reims_vgpu.o rvg_a"])
        );
        assert!(member_globals("").is_empty());
    }
}
