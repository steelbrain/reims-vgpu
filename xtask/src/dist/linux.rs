//! `cargo xtask dist` on Linux: the library on the Vulkan platform, its only one.
//! The shared library is `lib/libreims_vgpu.so`, whose soname is its own name; nothing is signed
//! and nothing has an install name. `Libs.private` is what rustc prints for linking the archive
//! (`--print native-static-libs`).
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
//!
//! **The archive's order.** The same `objcopy` renames the section of each function the
//! link order lists (`crates/reims-vgpu-c/link-order.txt`, [`order`]) to `.text.sorted.NNNNNN`
//! in the list's order, which GNU ld lays out ahead of the rest of `.text` when QEMU links the
//! archive. `dist` prints how many of the list's functions it placed; one the archive no longer
//! has is skipped. The shared library keeps rustc's order.

use std::path::{Path, PathBuf};

use super::archive::{
    LOCALIZED, api_functions, check_globals, defined_globals, native_static_libs, nm_args,
};
use super::order;
use super::{
    LIB, Options, c_api_args, helper_args, install, install_helper, layout, pc_file, strs,
    strs_with, write,
};
use crate::tools::Tools;
use crate::util::Res;

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

/// Builds and writes the tree with `tools`; `header` is the committed header. Returns the output
/// directory.
pub fn dist(tools: &mut dyn Tools, o: &Options, target_dir: &Path, header: &Path) -> Res<PathBuf> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    tools.run(&cargo, &cargo_args(o))?;
    let built = target_dir.join(o.profile.dir());
    let [archive, shared, pc, include, helper] = layout().map(|p| o.out.join(p));

    localize(
        tools,
        &built.join("libreims_vgpu_c.a"),
        &target_dir.join(LOCALIZED),
        &archive,
        &o.link_order,
    )?;
    install(&built.join("libreims_vgpu_c.so"), &shared, 0o755)?;
    install(header, &include, 0o644)?;
    // Asked after the tree has its libraries: this build of the archive alone replaces the one
    // in the target dir, not the one installed.
    let libs = native_static_libs(&tools.output(&cargo, &static_libs_args(o))?)?;
    write(&pc, &pc_file(env!("CARGO_PKG_VERSION"), &libs))?;
    install_helper(tools, &helper_build_args(o), &built, &helper)?;
    Ok(o.out.clone())
}

/// The `cargo build` arguments that build the helper ([`super::HELPER`]) in the profile.
pub fn helper_build_args(o: &Options) -> Vec<String> {
    let mut args = vec!["build".to_owned()];
    args.extend(helper_args(o));
    args
}

/// Writes `archive`, the archive installed, from `built`, Cargo's, in `work` (module doc): its
/// members linked into one object, every global symbol but the C API's made local and the
/// functions `link_order` lists laid out first ([`order_renames`]), archived alone in `work`
/// with zero dates (`ar`'s `D`), and installed with mode 0644, so the same build gives the
/// same bytes and leaves the installed archive as it was. Fails where the object defines no
/// `rvg_` function, or where the archive written still defines another global.
fn localize(
    tools: &mut dyn Tools,
    built: &Path,
    work: &Path,
    archive: &Path,
    link_order: &Path,
) -> Res<()> {
    std::fs::create_dir_all(work).map_err(|e| format!("{}: {e}", work.display()))?;
    let object = work.join(format!("{LIB}.o"));
    let mut link = strs(&["-r", "--whole-archive"]);
    link.push(built.to_string_lossy().into_owned());
    link.extend(strs_with(&["-o"], &object));
    tools.run("ld", &link)?;
    let api = api_functions(&tools.output("nm", &nm_args(&object))?, &object)?;
    let keep = work.join("keep-global-symbols");
    write(&keep, &(api.join("\n") + "\n"))?;
    let mut strip = vec![format!("--keep-global-symbols={}", keep.display())];
    if let Some(renames) = order_renames(tools, &object, link_order, work)? {
        strip.push(format!("@{}", renames.display()));
    }
    strip.push(object.to_string_lossy().into_owned());
    tools.run("objcopy", &strip)?;
    // `ar r` adds to an archive that is there: the one packed starts empty.
    let packed = work.join(format!("lib{LIB}.a"));
    if std::fs::symlink_metadata(&packed).is_ok() {
        std::fs::remove_file(&packed).map_err(|e| format!("{}: {e}", packed.display()))?;
    }
    let mut pack = strs_with(&["rcsD"], &packed);
    pack.push(object.to_string_lossy().into_owned());
    tools.run("ar", &pack)?;
    install(&packed, archive, 0o644)?;
    check_globals(
        archive,
        defined_globals(&tools.output("nm", &nm_args(archive))?),
    )
}

/// The `objcopy` response file, written in `work`, that renames the sections of `object`'s
/// functions `link_order` lists to [`order::SORTED`] sections in its order ([`order::renames`]), and
/// prints how many of them it placed. None where the list is missing or empty, or names none of
/// `object`'s functions: the archive then keeps rustc's order.
fn order_renames(
    tools: &mut dyn Tools,
    object: &Path,
    link_order: &Path,
    work: &Path,
) -> Res<Option<PathBuf>> {
    let Ok(text) = std::fs::read_to_string(link_order) else {
        println!("dist: no link order at {}", link_order.display());
        return Ok(None);
    };
    let wanted = order::entries(&text);
    if wanted.is_empty() {
        return Ok(None);
    }
    let raw = tools.output("nm", &order::nm_args(object, false))?;
    let demangled = tools.output("nm", &order::nm_args(object, true))?;
    let (args, placed) = order::renames(&wanted, &order::functions(&raw, &demangled)?);
    println!(
        "dist: link order: {placed} of the {} functions {} lists placed",
        wanted.len(),
        link_order.display()
    );
    if args.is_empty() {
        return Ok(None);
    }
    let file = work.join("link-order-renames");
    write(&file, &(args.join("\n") + "\n"))?;
    Ok(Some(file))
}

#[cfg(test)]
mod tests {
    use super::super::archive::tests::CARGO;
    use super::super::parse_args;
    use super::super::testing::{LONG_AGO, Log, age, mode, mtime, put, tmp};
    use super::*;
    use crate::tools::Profile;

    /// The options of `args`, with no link order: `target`'s `link-order.txt`, which a test
    /// writes where it wants one.
    fn options(args: &[&str], target: &Path) -> Options {
        Options {
            link_order: target.join("link-order.txt"),
            ..parse_args(&strs(args), target).unwrap()
        }
    }

    #[test]
    fn parse_args_defaults() {
        let o = parse_args(&[], Path::new("/t")).unwrap();
        assert_eq!(o.profile, Profile("release".into()));
        assert_eq!(o.out, PathBuf::from("/t/dist-vulkan"));
    }

    #[test]
    fn cargo_args_build_the_c_api_crate_with_its_soname() {
        assert_eq!(
            cargo_args(&options(&[], Path::new("/t"))).join(" "),
            "rustc --profile release -p reims-vgpu-c --lib \
             -- -C link-arg=-Wl,-soname,libreims_vgpu.so"
        );
    }

    #[test]
    fn static_libs_args_build_the_archive_alone_and_print_its_libraries() {
        assert_eq!(
            static_libs_args(&options(&["--profile", "dev"], Path::new("/t"))).join(" "),
            "rustc --profile dev -p reims-vgpu-c --lib \
             --crate-type staticlib --message-format json -- --print native-static-libs"
        );
    }

    #[test]
    fn layout_names_the_shared_object() {
        assert_eq!(layout()[1], "lib/libreims_vgpu.so");
    }

    /// A tree the build would write: the libraries Cargo built and the header.
    fn built(name: &str) -> (PathBuf, PathBuf, PathBuf) {
        let root = tmp(name);
        let target = root.join("target");
        put(&target.join("release/libreims_vgpu_c.a"), "archive");
        put(&target.join("release/libreims_vgpu_c.so"), "shared");
        put(&target.join("release/reims-vgpu-compile"), "helper");
        let header = root.join("reims_vgpu.h");
        put(&header, "header");
        (root.clone(), target, header)
    }

    /// The programs as `dist` runs them: each call recorded; `ar` writes the archive it names,
    /// and `nm` lists `object` for the linked object's globals, `listing` for its every symbol
    /// (`-p`: as named, then demangled with `-C`), and `archive` for the archive.
    struct Programs {
        log: Log,
        object: String,
        listing: (String, String),
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
                listing: Default::default(),
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
            let has = |flag: &str| args.iter().any(|a| a == flag);
            Ok(match program {
                "nm" if has("-p") && has("-C") => self.listing.1.clone(),
                "nm" if has("-p") => self.listing.0.clone(),
                "nm" if last.ends_with(".o") => self.object.clone(),
                "nm" => self.archive.clone(),
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
        let (root, target, header) = built("tree");
        let o = options(&[], &target);
        // A previous tree's files are replaced.
        put(&o.out.join("lib/libreims_vgpu.so"), "old");
        put(&o.out.join("lib/libreims_vgpu.a"), "old");
        let mut tools = Programs::new(OBJECT, LOCALIZED_ARCHIVE);
        let out = dist(&mut tools, &o, &target, &header).unwrap();
        assert_eq!(out, target.join("dist-vulkan"));
        let read = |p: &str| std::fs::read_to_string(out.join(p)).unwrap();
        assert_eq!(read("lib/libreims_vgpu.a"), "localized");
        assert_eq!(read("lib/libreims_vgpu.so"), "shared");
        assert_eq!(read("include/reims_vgpu.h"), "header");
        assert!(
            read("lib/pkgconfig/reims-vgpu.pc")
                .contains("Libs.private: -lgcc_s -lutil -lrt -lpthread -lm -ldl -lc\n")
        );
        assert_eq!(read("bin/reims-vgpu-compile"), "helper");
        assert_eq!(mode(&out.join("lib/libreims_vgpu.so")), 0o755);
        assert_eq!(mode(&out.join("lib/libreims_vgpu.a")), 0o644);
        assert_eq!(mode(&out.join("bin/reims-vgpu-compile")), 0o755);
        let work = target.join("dist-localized");
        let keep = work.join("keep-global-symbols");
        assert_eq!(
            std::fs::read_to_string(&keep).unwrap(),
            "rvg_graphics_device_create\nrvg_status_message\n"
        );
        let object = work.join("reims_vgpu.o");
        let packed = work.join("libreims_vgpu.a");
        let archive = out.join("lib/libreims_vgpu.a");
        let calls = tools.log.lines();
        assert!(
            calls[0].ends_with(" -- -C link-arg=-Wl,-soname,libreims_vgpu.so"),
            "{}",
            calls[0]
        );
        assert_eq!(
            calls[1..6],
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
                format!("ar rcsD {} {}", packed.display(), object.display()),
                format!("nm -g --defined-only -P {}", archive.display()),
            ]
        );
        assert!(
            calls[6].ends_with(" -- --print native-static-libs"),
            "{}",
            calls[6]
        );
        assert!(
            calls[7].ends_with(
                " build --profile release -p reims-vgpu-c -p reims-vgpu-vulkan \
                 --bin reims-vgpu-compile"
            ),
            "{}",
            calls[7]
        );
        assert_eq!(calls.len(), 8, "{calls:?}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A dist of the same build again leaves every file of the tree as it was, mtime included,
    /// so what is built from the tree is not built again
    #[test]
    fn dist_again_keeps_the_tree() {
        let (root, target, header) = built("again");
        let o = options(&[], &target);
        let run = || {
            let mut tools = Programs::new(OBJECT, LOCALIZED_ARCHIVE);
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

    /// The object's listing of every symbol: two functions and a datum, as named and as `nm -C`
    /// prints them.
    fn listing() -> (String, String) {
        (
            "0000000000001000 t _RNvCs1_1a1f\n0000000000002000 t _RNvCs1_1b1g.llvm.7\n\
             0000000000003000 d _RNvCs1_1a4DATA\n"
                .to_owned(),
            "0000000000001000 t a[0123456789abcdef]::f\n0000000000002000 t b[0123456789abcdef]::g\n\
             0000000000003000 d a[0123456789abcdef]::DATA\n"
                .to_owned(),
        )
    }

    /// The functions the link order lists get sections that sort in its order, renamed in the
    /// `objcopy` call that localizes the object; one the object lacks is skipped
    #[test]
    fn dist_lays_out_the_link_orders_functions_in_its_order() {
        let (root, target, header) = built("order");
        let o = options(&[], &target);
        put(&o.link_order, "# a comment\nb::g\ngone::x\na::f\n");
        let mut tools = Programs::new(OBJECT, LOCALIZED_ARCHIVE);
        tools.listing = listing();
        dist(&mut tools, &o, &target, &header).unwrap();
        let work = target.join("dist-localized");
        let renames = work.join("link-order-renames");
        assert_eq!(
            std::fs::read_to_string(&renames).unwrap(),
            "--rename-section\n.text._RNvCs1_1b1g.llvm.7=.text.sorted.000000\n\
             --rename-section\n.text._RNvCs1_1a1f=.text.sorted.000002\n"
        );
        let object = work.join("reims_vgpu.o");
        let calls = tools.log.lines();
        assert_eq!(
            calls[2..6],
            [
                format!("nm -g --defined-only -P {}", object.display()),
                format!("nm -p --defined-only {}", object.display()),
                format!("nm -p --defined-only -C {}", object.display()),
                format!(
                    "objcopy --keep-global-symbols={} @{} {}",
                    work.join("keep-global-symbols").display(),
                    renames.display(),
                    object.display()
                ),
            ]
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A link order that names none of the object's functions localizes it as without one: a
    /// stale list costs the order, never the build
    #[test]
    fn dist_with_a_stale_link_order_keeps_rustcs_order() {
        let (root, target, header) = built("stale");
        let o = options(&[], &target);
        put(&o.link_order, "gone::x\ngone::y\n");
        let mut tools = Programs::new(OBJECT, LOCALIZED_ARCHIVE);
        tools.listing = listing();
        dist(&mut tools, &o, &target, &header).unwrap();
        let work = target.join("dist-localized");
        assert!(!work.join("link-order-renames").exists());
        let object = work.join("reims_vgpu.o");
        let calls = tools.log.lines();
        assert_eq!(
            calls[5],
            format!(
                "objcopy --keep-global-symbols={} {}",
                work.join("keep-global-symbols").display(),
                object.display()
            )
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A global the archive still defines besides the C API's fails the dist, naming it
    #[test]
    fn dist_fails_where_the_archive_keeps_another_global() {
        let (root, target, header) = built("leak");
        let o = options(&[], &target);
        let leaky = "libreims_vgpu.a[reims_vgpu.o]:\nrvg_graphics_device_create T 1000 20\n\
                     SDL_Init T 2000 40\n";
        let mut tools = Programs::new(OBJECT, leaky);
        let e = dist(&mut tools, &o, &target, &header).unwrap_err();
        assert!(e.contains("1 global symbols besides the C API's"), "{e}");
        assert!(e.contains("SDL_Init"), "{e}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// An object with no `rvg_` function is refused before anything is made local
    #[test]
    fn dist_refuses_an_archive_without_the_c_api() {
        let (root, target, header) = built("noapi");
        let o = options(&[], &target);
        let mut tools = Programs::new("SDL_Init T 2000 40\n", LOCALIZED_ARCHIVE);
        let e = dist(&mut tools, &o, &target, &header).unwrap_err();
        assert!(e.contains("defines no rvg_ function"), "{e}");
        assert!(
            !tools.log.lines().iter().any(|l| l.starts_with("objcopy")),
            "{:?}",
            tools.log.lines()
        );
        std::fs::remove_dir_all(&root).unwrap();
    }
}
