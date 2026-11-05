//! `cargo xtask dist`: the library as a plain library, which a host links the way it links its
//! other dependencies, through pkg-config (docs/architecture.md 11.3). It writes
//! `<target>/dist/`:
//!
//! - `lib/libreims_vgpu.a`, and the shared library beside it, built from `reims-vgpu-c`, the
//!   crate of the C API: `lib/libreims_vgpu.dylib` on macOS, `lib/libreims_vgpu.so` on Linux.
//!   How each OS builds and names them is its own module's: `macos`, `linux`.
//! - `include/reims_vgpu.h`, the committed header.
//! - `lib/pkgconfig/reims-vgpu.pc`. Its `Libs.private` names every library a static link of the
//!   archive needs; its `optionrom` variable names the ROM below.
//! - `share/reims-vgpu/ReimsVgpu.rom`: the EFI option ROM, the driver in [`FIRMWARE`] built by
//!   its own Makefile (clang and lld) into `<target>/firmware`. QEMU's `apple-gfx-pci` loads it
//!   as its default `romfile`.
//! - `share/doc/reims-vgpu/`: the license, `LICENSE` and the texts in `LICENSES/` it points at,
//!   copied from the repository under the same names ([`install_license`]).
//!
//! QEMU (`make qemu`) and the smoke link the archive; the suite's runner links the dylib, and
//! loads it from the tree when it runs.
//!
//! `--platform vulkan` builds the library on the Vulkan platform instead (`reims-vgpu-c`'s
//! `vulkan` feature, architecture 2.2), into `<target>/dist-vulkan` unless `--out` says
//! otherwise, so the two trees sit side by side. The tree is laid out the same. On macOS the
//! default is Metal; on Linux, where only the Vulkan platform builds, it is Vulkan.

use std::path::{Path, PathBuf};

use crate::tools::{Profile, System, target_dir};
use crate::util::{Res, repo_root};

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "linux")]
use linux as native;
#[cfg(target_os = "macos")]
use macos as native;

/// The library's name: `lib/libreims_vgpu.{a,dylib,so}`, linked with `-lreims_vgpu`.
pub const LIB: &str = "reims_vgpu";
/// The pkg-config package: `lib/pkgconfig/reims-vgpu.pc`, and `share/reims-vgpu`.
pub const PACKAGE: &str = "reims-vgpu";
/// The header, under the name it is committed and installed with.
pub const HEADER: &str = "reims_vgpu.h";
/// The option ROM's file name: the firmware's build output, and its name in `share/reims-vgpu`
/// and in QEMU's data directory.
pub const ROM: &str = "ReimsVgpu.rom";
/// The EFI driver the option ROM packages, in the repository, and the directory under the target
/// dir it is built into.
pub const FIRMWARE: &str = "firmware";
/// The file that states the license, in the repository and in the tree's [`doc_dir`].
pub const LICENSE: &str = "LICENSE";
/// The directory of the license's texts, `<SPDX identifier>.txt` each, in the repository and in
/// the tree's [`doc_dir`].
pub const LICENSES: &str = "LICENSES";

/// The platform the tree's library is built on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    /// Metal: the product.
    Metal,
    /// The Vulkan platform, over metal2vulkan's runtime: `reims-vgpu-c`'s `vulkan` feature.
    Vulkan,
}

impl Platform {
    /// The platform `--platform` names.
    pub fn parse(name: &str) -> Res<Platform> {
        match name {
            "metal" => Ok(Platform::Metal),
            "vulkan" => Ok(Platform::Vulkan),
            other => Err(format!("--platform {other}: not metal or vulkan\n{USAGE}")),
        }
    }

    /// The tree's directory under the target dir when `--out` names none.
    pub fn default_out(self) -> &'static str {
        match self {
            Platform::Metal => "dist",
            Platform::Vulkan => "dist-vulkan",
        }
    }

    /// The `cargo build` features that select the platform: none for Metal, the default.
    pub fn features(self) -> &'static [&'static str] {
        match self {
            Platform::Metal => &[],
            Platform::Vulkan => &["--features", "reims-vgpu-c/vulkan"],
        }
    }
}

/// `cargo xtask dist`'s arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Options {
    pub profile: Profile,
    pub platform: Platform,
    /// The directory the tree is written into.
    pub out: PathBuf,
}

/// The usage line.
pub const USAGE: &str = "usage: cargo xtask dist [--platform metal|vulkan] \
[--profile dev|release|checked] [--release] [--out DIR]";

/// Parses the arguments. The default is a release build on the OS's default platform
/// (`native::DEFAULT_PLATFORM`), into `<target>/dist` on Metal and `<target>/dist-vulkan` on
/// Vulkan; `--out` wins over either.
pub fn parse_args(args: &[String], target_dir: &Path) -> Res<Options> {
    let mut profile = Profile("release".to_owned());
    let mut platform = native::DEFAULT_PLATFORM;
    let mut out = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut value = |name: &str| {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{name} needs a value\n{USAGE}"))
        };
        match a.as_str() {
            "--platform" => platform = Platform::parse(&value("--platform")?)?,
            "--profile" => profile = Profile(value("--profile")?),
            "--release" => profile = Profile("release".to_owned()),
            "--out" => out = Some(PathBuf::from(value("--out")?)),
            other => return Err(format!("unknown argument `{other}`\n{USAGE}")),
        }
    }
    Ok(Options {
        profile,
        platform,
        out: out.unwrap_or_else(|| target_dir.join(platform.default_out())),
    })
}

/// The Cargo arguments every OS's build of the library starts its own with, after the
/// subcommand: `reims-vgpu-c`'s libraries in the profile, on the platform.
pub fn c_api_args(o: &Options) -> Vec<String> {
    let mut args = strs(&[
        "--profile",
        o.profile.0.as_str(),
        "-p",
        "reims-vgpu-c",
        "--lib",
    ]);
    args.extend(strs(o.platform.features()));
    args
}

/// The `make` arguments that build the option ROM from `firmware`, the driver's directory, into
/// `build`.
pub fn firmware_args(firmware: &Path, build: &Path) -> Vec<String> {
    let mut args = strs_with(&["-C"], firmware);
    args.push(format!("BUILD={}", build.display()));
    args.push("rom".to_owned());
    args
}

/// `reims-vgpu.pc`. Its prefix is the file's grandparent directory (`${pcfiledir}`), so the tree
/// can be moved.
pub fn pc_file(version: &str, libs_private: &[String]) -> String {
    format!(
        "# Written by `cargo xtask dist`.\n\
         prefix=${{pcfiledir}}/../..\n\
         libdir=${{prefix}}/lib\n\
         includedir=${{prefix}}/include\n\
         # The EFI option ROM for the PCI device, 106b:eeee.\n\
         optionrom=${{prefix}}/share/{PACKAGE}/{ROM}\n\
         \n\
         Name: {PACKAGE}\n\
         Description: The host side of the paravirtualized GPU that macOS guests drive\n\
         Version: {version}\n\
         Libs: -L${{libdir}} -l{LIB}\n\
         Libs.private: {}\n\
         Cflags: -I${{includedir}}\n",
        libs_private.join(" ")
    )
}

/// The files `dist` writes under its output directory, relative to it: the archive, the shared
/// library (`native::SHARED_LIBRARY_EXTENSION`), the package, the header and the ROM.
pub fn layout() -> [String; 5] {
    [
        format!("lib/lib{LIB}.a"),
        format!("lib/lib{LIB}.{}", native::SHARED_LIBRARY_EXTENSION),
        format!("lib/pkgconfig/{PACKAGE}.pc"),
        format!("include/{HEADER}"),
        format!("share/{PACKAGE}/{ROM}"),
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
    let header = repo_root().join("crates/reims-vgpu-c/include").join(HEADER);
    let firmware = repo_root().join(FIRMWARE);
    let out = native::dist(&mut System, &o, &target, &header, &firmware)?;
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

fn write(p: &Path, content: &str) -> Res<()> {
    mkdir_parent(p)?;
    std::fs::write(p, content).map_err(|e| format!("{}: {e}", p.display()))
}

/// Copies a built file to `to` with `mode`, replacing what is there.
fn install(from: &Path, to: &Path, mode: u32) -> Res<()> {
    use std::os::unix::fs::PermissionsExt as _;
    mkdir_parent(to)?;
    if std::fs::symlink_metadata(to).is_ok() {
        std::fs::remove_file(to).map_err(|e| format!("{}: {e}", to.display()))?;
    }
    std::fs::copy(from, to).map_err(|e| format!("{} -> {}: {e}", from.display(), to.display()))?;
    std::fs::set_permissions(to, std::fs::Permissions::from_mode(mode))
        .map_err(|e| format!("{}: {e}", to.display()))
}

/// What the OSes' tests share.
#[cfg(test)]
mod testing {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

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

    pub fn mode(p: &Path) -> u32 {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::metadata(p).unwrap().permissions().mode() & 0o777
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_args_every_option() {
        let o = parse_args(&strs(&["--profile", "dev", "--out", "/o"]), Path::new("/t")).unwrap();
        assert_eq!(o.profile, Profile("dev".into()));
        assert_eq!(o.out, PathBuf::from("/o"));
        let o = parse_args(&strs(&["--profile", "dev", "--release"]), Path::new("/t")).unwrap();
        assert_eq!(o.profile, Profile("release".into()));
        let o = parse_args(&strs(&["--platform", "metal"]), Path::new("/t")).unwrap();
        assert_eq!(
            (o.platform, o.out),
            (Platform::Metal, PathBuf::from("/t/dist"))
        );
    }

    /// Vulkan's tree goes beside Metal's unless --out names one, whichever comes first
    #[test]
    fn parse_args_vulkan_goes_beside_metal() {
        let o = parse_args(&strs(&["--platform", "vulkan"]), Path::new("/t")).unwrap();
        assert_eq!(o.platform, Platform::Vulkan);
        assert_eq!(o.out, PathBuf::from("/t/dist-vulkan"));
        for args in [
            ["--platform", "vulkan", "--out", "/o"],
            ["--out", "/o", "--platform", "vulkan"],
        ] {
            let o = parse_args(&strs(&args), Path::new("/t")).unwrap();
            assert_eq!((o.platform, o.out), (Platform::Vulkan, PathBuf::from("/o")));
        }
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
    fn firmware_args_build_the_rom_into_the_build_dir() {
        assert_eq!(
            firmware_args(Path::new("/r/firmware"), Path::new("/t/firmware")).join(" "),
            "-C /r/firmware BUILD=/t/firmware rom"
        );
    }

    #[test]
    fn pc_file_links_the_library_and_names_the_rom() {
        let pc = pc_file("1.2.3", &strs(&["-framework", "Metal", "-lobjc"]));
        for line in [
            "prefix=${pcfiledir}/../..",
            "libdir=${prefix}/lib",
            "includedir=${prefix}/include",
            "optionrom=${prefix}/share/reims-vgpu/ReimsVgpu.rom",
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

    #[test]
    fn layout_matches_what_the_pc_file_names() {
        let pc = pc_file("0", &[]);
        let [archive, _shared, pcpath, header, rom] = layout();
        assert_eq!(archive, "lib/libreims_vgpu.a");
        assert_eq!(pcpath, "lib/pkgconfig/reims-vgpu.pc");
        assert_eq!(header, "include/reims_vgpu.h");
        assert!(pc.contains(&format!("${{prefix}}/{rom}")));
    }
}
