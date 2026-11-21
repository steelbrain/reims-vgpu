//! `cargo xtask dist`: the library as a plain library, which a host links the way it links its
//! other dependencies, through pkg-config (docs/architecture.md 11.3). It writes
//! `<target>/dist-vulkan/` ([`DEFAULT_OUT`]) unless `--out` names another directory:
//!
//! - `lib/libreims_vgpu.a`, and the shared library beside it, built from `reims-vgpu-c`, the
//!   crate of the C API: `lib/libreims_vgpu.dylib` on macOS, `lib/libreims_vgpu.so` on Linux,
//!   `lib/libreims_vgpu.dll` on Windows. How each OS builds and names them is its own module's:
//!   `macos`, `linux`, `windows`.
//! - `include/reims_vgpu.h`, the committed header.
//! - `bin/reims-vgpu-compile` (`.exe` on Windows), the program the library starts beside its
//!   host's executable to compile a kernel past rt's stage bound ([`HELPER`]). A host's install
//!   puts it beside the host's executable.
//! - `lib/pkgconfig/reims-vgpu.pc`. Its `Libs.private` names every library a static link of the
//!   archive needs.
//! - `share/doc/reims-vgpu/`: the license, `LICENSE` and the texts in `LICENSES/` it points at,
//!   copied from the repository under the same names ([`install_license`]).
//!
//! On Linux the archive lays out the functions [`order::ORDER_FILE`] lists first, in its order
//! ([`order`]).
//!
//! QEMU (`make qemu`) links the archive; the C API's tests link the shared library. A file whose
//! bytes are those the tree already holds is left as it was, mtime included
//! ([`replace_if_changed`]), so a dist of an unchanged build has QEMU's build do nothing.
//!
//! Its Cargo steps run with [`BUILD_COMMIT`] naming the commit the tree is at ([`build_commit`]),
//! which the recorder writes into each recording's stats file, so a dist at another commit
//! compiles the Vulkan crate again.
//!
//! The library is built on the Vulkan platform (architecture 2.2), its only one: natively on
//! Linux and Windows, and on macOS over Mesa's KosmicKrisp where the distribution ships it.
//! `--platform vulkan` names it, for the builds that pass it; any other platform is refused.

use std::path::{Path, PathBuf};

use crate::tools::{Profile, System, Tools, target_dir};
use crate::util::{Res, repo_root};

#[cfg(any(target_os = "linux", windows))]
mod archive;
#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
pub mod order;
#[cfg(windows)]
mod windows;
#[cfg(target_os = "linux")]
use linux as native;
#[cfg(target_os = "macos")]
use macos as native;
#[cfg(windows)]
use windows as native;

/// The library's name: `lib/libreims_vgpu.{a,dylib,so,dll}`, linked with `-lreims_vgpu`.
pub const LIB: &str = "reims_vgpu";
/// The pkg-config package: `lib/pkgconfig/reims-vgpu.pc`, and `share/reims-vgpu`.
pub const PACKAGE: &str = "reims-vgpu";
/// The header, under the name it is committed and installed with.
pub const HEADER: &str = "reims_vgpu.h";
/// The stage helper's program (`reims_vgpu_vulkan::rt_platform::stage`), `reims-vgpu-vulkan`'s,
/// in the tree's `bin` with the OS's executable suffix ([`helper_file`]).
pub const HELPER: &str = "reims-vgpu-compile";
/// The file that states the license, in the repository and in the tree's [`doc_dir`].
pub const LICENSE: &str = "LICENSE";
/// The directory of the license's texts, `<SPDX identifier>.txt` each, in the repository and in
/// the tree's [`doc_dir`].
pub const LICENSES: &str = "LICENSES";

/// The tree's directory under the target dir when `--out` names none.
pub const DEFAULT_OUT: &str = "dist-vulkan";

/// The variable `dist` sets for its Cargo steps: the commit the library is built from
/// ([`build_commit`]), which `crates/reims-vgpu-vulkan/src/recorder.rs` compiles in as
/// `recorder::COMMIT`. Cargo and kache key the crate on it.
pub const BUILD_COMMIT: &str = "RVG_BUILD_COMMIT";

/// The commit the tree is at, `git rev-parse HEAD`, with `-dirty` where a tracked file differs
/// from it or where git cannot say. Left aside: submodules, `crates/reims-ctl` (a workspace of
/// its own, linking nothing of the library), and `Cargo.lock`, which is committed without
/// metal2vulkan's git `source` lines and which every build writes them back into. `None` where
/// there is no commit to name (no git, or no checkout).
pub fn build_commit(tools: &mut dyn Tools) -> Option<String> {
    let head = tools.output("git", &strs(&["rev-parse", "HEAD"])).ok()?;
    if head.is_empty() {
        return None;
    }
    let changed = tools.output(
        "git",
        &strs(&[
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--ignore-submodules",
            "--",
            ".",
            ":(exclude)crates/reims-ctl",
            ":(exclude)Cargo.lock",
        ]),
    );
    Some(match changed {
        Ok(changed) if changed.is_empty() => head,
        _ => format!("{head}-dirty"),
    })
}

/// `cargo xtask dist`'s arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Options {
    pub profile: Profile,
    /// The directory the tree is written into.
    pub out: PathBuf,
    /// The list of the library's functions Linux's archive lays out first, in its order
    /// ([`order`]): the repository's [`order::ORDER_FILE`].
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub link_order: PathBuf,
}

/// The usage line.
pub const USAGE: &str = "usage: cargo xtask dist [--platform vulkan] \
[--profile dev|release|checked] [--release] [--out DIR]";

/// Parses the arguments. The default is a release build into `<target>/dist-vulkan`
/// ([`DEFAULT_OUT`]); `--out` wins over it.
pub fn parse_args(args: &[String], target_dir: &Path) -> Res<Options> {
    let mut profile = Profile("release".to_owned());
    let mut out = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut value = |name: &str| {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{name} needs a value\n{USAGE}"))
        };
        match a.as_str() {
            "--platform" => match value("--platform")?.as_str() {
                "vulkan" => {}
                other => {
                    return Err(format!(
                        "--platform {other}: the library builds on vulkan alone\n{USAGE}"
                    ));
                }
            },
            "--profile" => profile = Profile(value("--profile")?),
            "--release" => profile = Profile("release".to_owned()),
            "--out" => out = Some(PathBuf::from(value("--out")?)),
            other => return Err(format!("unknown argument `{other}`\n{USAGE}")),
        }
    }
    Ok(Options {
        profile,
        out: out.unwrap_or_else(|| target_dir.join(DEFAULT_OUT)),
        link_order: repo_root().join(order::ORDER_FILE),
    })
}

/// The Cargo arguments every OS's build of the library starts its own with, after the
/// subcommand: `reims-vgpu-c`'s libraries in the profile.
pub fn c_api_args(o: &Options) -> Vec<String> {
    strs(&[
        "--profile",
        o.profile.0.as_str(),
        "-p",
        "reims-vgpu-c",
        "--lib",
    ])
}

/// The helper's file name, in the build and in the tree: [`HELPER`] with the OS's executable
/// suffix.
pub fn helper_file() -> String {
    format!("{HELPER}{}", std::env::consts::EXE_SUFFIX)
}

/// The Cargo arguments every OS's build of the helper starts its own with, after the subcommand:
/// [`HELPER`] in the profile, with `reims-vgpu-c` selected beside its package, so the crates the
/// two share are resolved with the library's build's features, and built once.
pub fn helper_args(o: &Options) -> Vec<String> {
    strs(&[
        "--profile",
        o.profile.0.as_str(),
        "-p",
        "reims-vgpu-c",
        "-p",
        "reims-vgpu-vulkan",
        "--bin",
        HELPER,
    ])
}

/// Builds the helper with Cargo's `args` and installs it from `built`, the profile's directory of
/// the build, to `to`, with mode 0755.
fn install_helper(tools: &mut dyn Tools, args: &[String], built: &Path, to: &Path) -> Res<()> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    tools.run(&cargo, args)?;
    install(&built.join(helper_file()), to, 0o755)
}

/// `reims-vgpu.pc`. Its prefix is the file's grandparent directory (`${pcfiledir}`), so the tree
/// can be moved.
pub fn pc_file(version: &str, libs_private: &[String]) -> String {
    format!(
        "# Written by `cargo xtask dist`.\n\
         prefix=${{pcfiledir}}/../..\n\
         libdir=${{prefix}}/lib\n\
         includedir=${{prefix}}/include\n\
         \n\
         Name: {PACKAGE}\n\
         Description: The host side of the reims-graphics device, the GPU macOS guests drive\n\
         Version: {version}\n\
         Libs: -L${{libdir}} -l{LIB}\n\
         Libs.private: {}\n\
         Cflags: -I${{includedir}}\n",
        libs_private.join(" ")
    )
}

/// The files `dist` writes under its output directory, relative to it: the archive, the shared
/// library (`native::SHARED_LIBRARY_EXTENSION`), the package, the header and the helper.
pub fn layout() -> [String; 5] {
    [
        format!("lib/lib{LIB}.a"),
        format!("lib/lib{LIB}.{}", native::SHARED_LIBRARY_EXTENSION),
        format!("lib/pkgconfig/{PACKAGE}.pc"),
        format!("include/{HEADER}"),
        format!("bin/{}", helper_file()),
    ]
}

/// The package's documentation directory in the tree, relative to it: the license.
pub fn doc_dir() -> String {
    format!("share/doc/{PACKAGE}")
}

/// Copies the license from the repository at `root` into `out`'s [`doc_dir`], under the same
/// names, so the paths [`LICENSE`] gives the texts hold there too: [`LICENSE`], then each
/// `.txt` file of [`LICENSES`] in name order, with mode 0644, replacing what is there. Returns
/// the files written, relative to `out`, in that order. Fails where [`LICENSES`] holds no text.
pub fn install_license(root: &Path, out: &Path) -> Res<Vec<String>> {
    let texts = root.join(LICENSES);
    let mut names = Vec::new();
    for entry in std::fs::read_dir(&texts).map_err(|e| format!("{}: {e}", texts.display()))? {
        let name = entry
            .map_err(|e| format!("{}: {e}", texts.display()))?
            .file_name()
            .to_string_lossy()
            .into_owned();
        if name.ends_with(".txt") {
            names.push(name);
        }
    }
    if names.is_empty() {
        return Err(format!("{}: no license text in it", texts.display()));
    }
    names.sort();
    let doc = doc_dir();
    let files = std::iter::once(LICENSE.to_owned())
        .chain(names.iter().map(|name| format!("{LICENSES}/{name}")));
    let mut written = Vec::new();
    for file in files {
        let to = format!("{doc}/{file}");
        install(&root.join(&file), &out.join(&to), 0o644)?;
        written.push(to);
    }
    Ok(written)
}

/// `cargo xtask dist`.
pub fn cmd_dist(args: &[String]) -> Res<bool> {
    let target = target_dir(&mut System)?;
    let o = parse_args(args, &target)?;
    if let Some(commit) = build_commit(&mut System) {
        // SAFETY: xtask runs on one thread, and every program it has run so far has exited:
        // nothing reads the environment while it changes. The Cargo steps below inherit it.
        unsafe { std::env::set_var(BUILD_COMMIT, commit) };
    }
    let header = repo_root().join("crates/reims-vgpu-c/include").join(HEADER);
    let out = native::dist(&mut System, &o, &target, &header)?;
    let license = install_license(&repo_root(), &out)?;
    for p in layout().into_iter().chain(license) {
        println!("dist: {}", out.join(p).display());
    }
    Ok(true)
}

fn strs(args: &[&str]) -> Vec<String> {
    args.iter().map(|s| (*s).to_owned()).collect()
}

fn strs_with(args: &[&str], p: &Path) -> Vec<String> {
    let mut v = strs(args);
    v.push(p.to_string_lossy().into_owned());
    v
}

fn mkdir_parent(p: &Path) -> Res<()> {
    match p.parent() {
        Some(dir) => std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display())),
        None => Ok(()),
    }
}

/// Writes `content` to `p` with mode 0644 ([`replace_if_changed`]).
fn write(p: &Path, content: &str) -> Res<()> {
    replace_if_changed(p, content.as_bytes(), 0o644).map(drop)
}

/// Copies a built file to `to` with `mode` ([`replace_if_changed`]).
fn install(from: &Path, to: &Path, mode: u32) -> Res<()> {
    let bytes = std::fs::read(from).map_err(|e| format!("{}: {e}", from.display()))?;
    replace_if_changed(to, &bytes, mode).map(drop)
}

/// Every file `dist` writes is written by this: `to` gets `bytes` and `mode` (on Unix; Windows
/// keeps no mode bits), and is written only where it is not already a file holding `bytes`. An
/// unchanged file keeps its mtime, so what is built from the tree is not built again: `make qemu`
/// recompiles `reims-graphics-pci.c` where the header is newer, and relinks QEMU where the
/// archive is. A file that changed is removed
/// first and written anew, so a process that maps the old one (a host running the shared
/// library) keeps its pages. Returns whether it wrote.
fn replace_if_changed(to: &Path, bytes: &[u8], mode: u32) -> Res<bool> {
    let err = |e: std::io::Error| format!("{}: {e}", to.display());
    let old = std::fs::symlink_metadata(to).ok();
    let same = old
        .as_ref()
        .is_some_and(|m| m.is_file() && m.len() == bytes.len() as u64)
        && std::fs::read(to).map_err(err)? == bytes;
    if !same {
        mkdir_parent(to)?;
        if old.is_some() {
            std::fs::remove_file(to).map_err(err)?;
        }
        std::fs::write(to, bytes).map_err(err)?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(to, std::fs::Permissions::from_mode(mode)).map_err(err)?;
    }
    #[cfg(not(unix))]
    let _ = mode;
    Ok(!same)
}

/// What the OSes' tests share.
#[cfg(test)]
mod testing {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::SystemTime;

    use crate::tools::Tools;
    use crate::util::Res;

    /// Records every call; `output` answers with `answer`, and `run` of `fail` fails.
    #[derive(Default)]
    pub struct Log {
        pub calls: Vec<(String, Vec<String>)>,
        pub answer: String,
        pub fail: Option<&'static str>,
    }

    impl Tools for Log {
        fn run(&mut self, program: &str, args: &[String]) -> Res<()> {
            self.calls.push((program.to_owned(), args.to_vec()));
            if self.fail == Some(program) {
                return Err(format!("{program} failed"));
            }
            Ok(())
        }
        fn output(&mut self, program: &str, args: &[String]) -> Res<String> {
            self.calls.push((program.to_owned(), args.to_vec()));
            Ok(self.answer.clone())
        }
    }

    impl Log {
        /// Each call as one line: the program, then its arguments.
        pub fn lines(&self) -> Vec<String> {
            self.calls
                .iter()
                .map(|(p, a)| format!("{p} {}", a.join(" ")))
                .collect()
        }
    }

    pub fn tmp(name: &str) -> PathBuf {
        static N: AtomicUsize = AtomicUsize::new(0);
        let p = std::env::temp_dir().join(format!(
            "xtask-dist-{}-{}-{name}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    pub fn put(p: &Path, content: &str) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, content).unwrap();
    }

    #[cfg(unix)]
    pub fn mode(p: &Path) -> u32 {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::metadata(p).unwrap().permissions().mode() & 0o777
    }

    /// A time long past, which no file a test writes has as its mtime unless set to it.
    pub const LONG_AGO: SystemTime = SystemTime::UNIX_EPOCH;

    /// Sets `p`'s mtime to [`LONG_AGO`].
    pub fn age(p: &Path) {
        let file = std::fs::File::options().write(true).open(p).unwrap();
        file.set_modified(LONG_AGO).unwrap();
    }

    pub fn mtime(p: &Path) -> SystemTime {
        std::fs::metadata(p).unwrap().modified().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Answers `git rev-parse HEAD` with `head` and `git status` with `status`.
    struct Git {
        head: Res<String>,
        status: Res<String>,
    }

    impl Tools for Git {
        fn run(&mut self, program: &str, _: &[String]) -> Res<()> {
            Err(format!("{program}: not run here"))
        }
        fn output(&mut self, program: &str, args: &[String]) -> Res<String> {
            assert_eq!(program, "git");
            match args.first().map(String::as_str) {
                Some("rev-parse") => self.head.clone(),
                Some("status") => self.status.clone(),
                _ => Err(format!("git {args:?}")),
            }
        }
    }

    /// The build commit is HEAD's, marked dirty where a tracked file differs or git cannot say,
    /// and absent with no commit to name.
    #[test]
    fn build_commit_names_head_and_marks_a_changed_tree() {
        let head = "0123456789abcdef0123456789abcdef01234567".to_owned();
        let at = |head: Res<String>, status: Res<String>| build_commit(&mut Git { head, status });
        assert_eq!(at(Ok(head.clone()), Ok(String::new())), Some(head.clone()));
        let dirty = Some(format!("{head}-dirty"));
        assert_eq!(at(Ok(head.clone()), Ok(" M Cargo.toml".into())), dirty);
        assert_eq!(at(Ok(head.clone()), Err("git status failed".into())), dirty);
        assert_eq!(
            at(Err("not a git repository".into()), Ok(String::new())),
            None
        );
        assert_eq!(at(Ok(String::new()), Ok(String::new())), None);
    }

    #[test]
    fn parse_args_every_option() {
        let o = parse_args(&strs(&["--profile", "dev", "--out", "/o"]), Path::new("/t")).unwrap();
        assert_eq!(o.profile, Profile("dev".into()));
        assert_eq!(o.out, PathBuf::from("/o"));
        let o = parse_args(&strs(&["--profile", "dev", "--release"]), Path::new("/t")).unwrap();
        assert_eq!(o.profile, Profile("release".into()));
    }

    /// `--platform vulkan` changes nothing: the tree goes into dist-vulkan unless --out names
    /// one, whichever comes first
    #[test]
    fn parse_args_takes_vulkan_alone() {
        let o = parse_args(&strs(&["--platform", "vulkan"]), Path::new("/t")).unwrap();
        assert_eq!(o, parse_args(&[], Path::new("/t")).unwrap());
        assert_eq!(o.out, PathBuf::from("/t/dist-vulkan"));
        for args in [
            ["--platform", "vulkan", "--out", "/o"],
            ["--out", "/o", "--platform", "vulkan"],
        ] {
            let o = parse_args(&strs(&args), Path::new("/t")).unwrap();
            assert_eq!(o.out, PathBuf::from("/o"));
        }
        let metal = parse_args(&strs(&["--platform", "metal"]), Path::new("/t")).unwrap_err();
        assert!(metal.contains("--platform metal"), "{metal}");
    }

    #[test]
    fn parse_args_rejects_unknown_and_missing_values() {
        let unknown = parse_args(&strs(&["--skin"]), Path::new("/t")).unwrap_err();
        assert!(unknown.contains("--skin"), "{unknown}");
        let missing = parse_args(&strs(&["--out"]), Path::new("/t")).unwrap_err();
        assert!(missing.contains("--out needs"), "{missing}");
        let platform = parse_args(&strs(&["--platform", "opengl"]), Path::new("/t")).unwrap_err();
        assert!(platform.contains("--platform opengl"), "{platform}");
        let missing = parse_args(&strs(&["--platform"]), Path::new("/t")).unwrap_err();
        assert!(missing.contains("--platform needs"), "{missing}");
    }

    #[test]
    fn pc_file_links_the_library() {
        let pc = pc_file("1.2.3", &strs(&["-framework", "Metal", "-lobjc"]));
        for line in [
            "prefix=${pcfiledir}/../..",
            "libdir=${prefix}/lib",
            "includedir=${prefix}/include",
            "Name: reims-vgpu",
            "Version: 1.2.3",
            "Libs: -L${libdir} -lreims_vgpu",
            "Libs.private: -framework Metal -lobjc",
            "Cflags: -I${includedir}",
        ] {
            assert!(pc.lines().any(|l| l == line), "{line} missing from\n{pc}");
        }
    }

    /// The license and each text land in the doc dir under their own names, replacing an older
    /// copy, and nothing in `LICENSES` but its texts is copied
    #[test]
    fn install_license_copies_the_license_and_its_texts() {
        let root = testing::tmp("license");
        let repo = root.join("repo");
        testing::put(&repo.join("LICENSE"), "either, at your option");
        testing::put(&repo.join("LICENSES/B-2.0.txt"), "b");
        testing::put(&repo.join("LICENSES/A-1.0.txt"), "a");
        testing::put(&repo.join("LICENSES/.DS_Store"), "finder");
        let out = root.join("dist");
        testing::put(&out.join("share/doc/reims-vgpu/LICENSE"), "old");
        let written = install_license(&repo, &out).unwrap();
        assert_eq!(
            written,
            strs(&[
                "share/doc/reims-vgpu/LICENSE",
                "share/doc/reims-vgpu/LICENSES/A-1.0.txt",
                "share/doc/reims-vgpu/LICENSES/B-2.0.txt",
            ])
        );
        let read = |p: &str| std::fs::read_to_string(out.join(p)).unwrap();
        assert_eq!(read(&written[0]), "either, at your option");
        assert_eq!(read(&written[1]), "a");
        assert_eq!(read(&written[2]), "b");
        #[cfg(unix)]
        for p in &written {
            assert_eq!(testing::mode(&out.join(p)), 0o644, "{p}");
        }
        assert!(!out.join("share/doc/reims-vgpu/LICENSES/.DS_Store").exists());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn install_license_refuses_a_repository_without_texts() {
        let root = testing::tmp("nolicense");
        testing::put(&root.join("LICENSE"), "either");
        let e = install_license(&root, &root.join("dist")).unwrap_err();
        assert!(e.contains("LICENSES"), "{e}");
        testing::put(&root.join("LICENSES/README"), "not a text");
        let e = install_license(&root, &root.join("dist")).unwrap_err();
        assert!(e.contains("no license text"), "{e}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// The repository's own license, copied, has every text its `LICENSE` names and every one
    /// the workspace's license expression names
    #[test]
    fn install_license_of_the_repository_has_every_text_it_names() {
        let root = testing::tmp("repo-license");
        let written = install_license(&repo_root(), &root).unwrap();
        let doc = root.join(doc_dir());
        let license = std::fs::read_to_string(doc.join(LICENSE)).unwrap();
        let named: Vec<&str> = license
            .split_whitespace()
            .filter(|word| word.starts_with("LICENSES/"))
            .map(|word| word.trim_end_matches([';', ',', '.', ')']))
            .collect();
        assert!(!named.is_empty(), "{license}");
        let manifest = std::fs::read_to_string(repo_root().join("Cargo.toml")).unwrap();
        let expression = manifest
            .lines()
            .find_map(|l| l.strip_prefix("license = \""))
            .and_then(|l| l.strip_suffix('"'))
            .unwrap();
        let spdx = expression
            .split(" OR ")
            .map(|id| format!("{LICENSES}/{id}.txt"));
        for text in named.iter().map(|t| (*t).to_owned()).chain(spdx) {
            assert!(doc.join(&text).is_file(), "{text} missing from the tree");
            assert!(
                written.contains(&format!("{}/{text}", doc_dir())),
                "{text} not reported"
            );
        }
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Bytes a file already holds leave it as it was, mtime included, and only its mode is set;
    /// other bytes, of another length or the same, are written, and so is a missing file
    #[test]
    fn replace_if_changed_writes_only_other_bytes() {
        let root = testing::tmp("replace");
        let p = root.join("lib/libx.a");
        let read = || std::fs::read_to_string(&p).unwrap();
        assert!(replace_if_changed(&p, b"one", 0o644).unwrap());
        assert_eq!(read(), "one");
        testing::age(&p);
        assert!(!replace_if_changed(&p, b"one", 0o755).unwrap());
        assert_eq!(testing::mtime(&p), testing::LONG_AGO);
        #[cfg(unix)]
        assert_eq!(testing::mode(&p), 0o755);
        for other in ["two", "a longer one"] {
            testing::age(&p);
            assert!(replace_if_changed(&p, other.as_bytes(), 0o644).unwrap());
            assert_eq!(read(), other);
            assert_ne!(testing::mtime(&p), testing::LONG_AGO);
            #[cfg(unix)]
            assert_eq!(testing::mode(&p), 0o644);
        }
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A file that changed is a new file: another name for the old one, as a process that maps
    /// it holds, keeps the old bytes. A symlink (on Unix, where the test makes one) is replaced by
    /// a file, even one to the same bytes
    #[test]
    fn replace_if_changed_replaces_rather_than_writes_through() {
        let root = testing::tmp("replace-new");
        let p = root.join("libx.so");
        testing::put(&p, "old");
        let link = root.join("held");
        std::fs::hard_link(&p, &link).unwrap();
        assert!(replace_if_changed(&p, b"new", 0o755).unwrap());
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "new");
        assert_eq!(std::fs::read_to_string(&link).unwrap(), "old");
        #[cfg(unix)]
        {
            let target = root.join("target");
            testing::put(&target, "same");
            let sym = root.join("sym");
            std::os::unix::fs::symlink(&target, &sym).unwrap();
            assert!(replace_if_changed(&sym, b"same", 0o644).unwrap());
            assert!(std::fs::symlink_metadata(&sym).unwrap().is_file());
        }
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A built file installed again unchanged keeps its mtime; one that changed is copied
    #[test]
    fn install_copies_only_a_changed_file() {
        let root = testing::tmp("install");
        let (from, to) = (root.join("built.a"), root.join("dist/lib/libx.a"));
        testing::put(&from, "built");
        install(&from, &to, 0o644).unwrap();
        testing::age(&to);
        install(&from, &to, 0o644).unwrap();
        assert_eq!(testing::mtime(&to), testing::LONG_AGO);
        testing::put(&from, "built again");
        install(&from, &to, 0o644).unwrap();
        assert_eq!(std::fs::read_to_string(&to).unwrap(), "built again");
        assert_ne!(testing::mtime(&to), testing::LONG_AGO);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn layout_matches_what_the_pc_file_names() {
        let pc = pc_file("0", &[]);
        let [archive, _shared, pcpath, header, helper] = layout();
        assert_eq!(archive, "lib/libreims_vgpu.a");
        assert_eq!(pcpath, "lib/pkgconfig/reims-vgpu.pc");
        assert_eq!(header, "include/reims_vgpu.h");
        let suffix = if cfg!(windows) { ".exe" } else { "" };
        assert_eq!(helper, format!("bin/reims-vgpu-compile{suffix}"));
        assert!(
            pc.contains("libdir=${prefix}/lib\n") && pc.contains("includedir=${prefix}/include\n")
        );
    }
}
