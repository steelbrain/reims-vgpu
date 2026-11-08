//! reims-util (<https://github.com/steelbrain/reims-util>) makes what a new x86-64 VM boots from:
//! an OpenCore image, with OVMF's code and variables beside it, and a macOS installer, downloaded
//! from Apple and checked against Apple's chunklists. It also makes an existing VM a new OpenCore
//! image presenting the same Mac (opencore.rs). An image holds Lilu, which reims-util adds
//! itself, and, for a VM on reims-vgpu's own device, the guest's kexts reims-ctl carries, each
//! given by `--add-kext` in the set's order ([`kext_args`]). reims-ctl runs it
//! with `--json`: each line on its stdout is an event, `step`, `progress` or `note`, and the last
//! is `done`, with what it made, or `error`.
//!
//! It is installed beside reims-ctl, at [`REV`], by `cargo install` from its repository
//! ([`install`]). `reims-ctl __record-install`, which scripts/install-ctl.sh runs last, installs
//! it on Linux, where x86-64 VMs need it, and on any host where one is beside reims-ctl already
//! ([`beside`]), which moves to [`REV`]. So the update that brings a reims-ctl that needs it
//! installs it too, though that update runs the script it started with, which predates
//! reims-util. `REIMS_CTL_UTIL` names a binary to use instead.
//!
//! An install builds it in an empty target directory: cargo 1.98.1 judges a git package's binary
//! up to date across commits (measured 2026-10-13): from a target directory an earlier commit was
//! built in, `cargo install --rev` of a later one installs the earlier commit's binary and records
//! it as the later one's. cargo replaces a reims-util installed from another commit or from a
//! path, and leaves one already at [`REV`] as it is, building nothing (measured 2026-10-26, cargo
//! 1.98.1), so an install moves it to the pin and no further. The build takes 12 s on an M3 Max.
//!
//! Installers go in the data directory's `installers`, shared by every VM: a VM uses an offline
//! installer in place, and a raw copy of an online one, a DMG (create.rs). A download stopped
//! part way is continued by the next, and an installer already there is checked against Apple's
//! chunklists and kept. One reims-util at a time writes there ([`lock`]).

use crate::opencore::Identity;
use crate::paths;
use crate::prebuilt::{self, Kext};
use crate::vm::Graphics;
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::ffi::OsStr;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub const REPO: &str = "https://github.com/steelbrain/reims-util";
/// The commit [`install`] installs.
pub const REV: &str = "5ca3b10683be28938c01f977b89af5ffa8685754";

/// The reims-util binary: `REIMS_CTL_UTIL`, else the one beside reims-ctl, where both are
/// installed, else the first on PATH.
pub fn binary() -> Result<PathBuf> {
    if let Some(p) = std::env::var_os("REIMS_CTL_UTIL") {
        return Ok(PathBuf::from(p));
    }
    let beside = beside();
    let on_path: Vec<PathBuf> = std::env::var_os("PATH")
        .map(|p| {
            std::env::split_paths(&p)
                .map(|d| d.join("reims-util"))
                .collect()
        })
        .unwrap_or_default();
    beside
        .into_iter()
        .chain(on_path)
        .find(|p| p.is_file())
        .context("reims-util is not installed: run reims-ctl update, which installs it")
}

/// The reims-util beside the running reims-ctl, in the bin directory cargo installed both into.
pub fn beside() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(|d| d.join("reims-util")))
        .filter(|p| p.is_file())
}

/// `reims-util --version`: "reims-util 0.1.0".
pub fn version(bin: &Path) -> Option<String> {
    let o = Command::new(bin).arg("--version").output().ok()?;
    let v = String::from_utf8_lossy(&o.stdout).trim().to_string();
    (o.status.success() && !v.is_empty()).then_some(v)
}

/// Installs reims-util at [`REV`] with `cargo install`, into the bin directory it installed
/// reims-ctl into, building afresh in the cache directory from `repo`, the reims-vgpu checkout,
/// whose rust-toolchain.toml picks the toolchain. One from another commit or source is replaced,
/// and one at [`REV`] is left as it is (the module doc).
pub fn install(repo: &Path) -> Result<()> {
    let cargo = which("cargo")
        .or_else(|| {
            let home = std::env::var_os("CARGO_HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| paths::home().join(".cargo"));
            Some(home.join("bin/cargo")).filter(|p| p.is_file())
        })
        .context("cargo is not installed")?;
    // An earlier commit's build there would be installed as this one's (the module's doc).
    let target = paths::cache_dir().join("util-target");
    let _ = std::fs::remove_dir_all(&target);
    let status = Command::new(&cargo)
        .args([
            "install", "--locked", "--quiet", "--git", REPO, "--rev", REV,
        ])
        .arg("reims-util")
        .env("CARGO_TARGET_DIR", &target)
        .current_dir(repo)
        .stdin(Stdio::null())
        .status()
        .with_context(|| format!("run {}", cargo.display()))?;
    let _ = std::fs::remove_dir_all(&target);
    if !status.success() {
        bail!("cargo install of reims-util failed ({status})");
    }
    Ok(())
}

fn which(name: &str) -> Option<PathBuf> {
    std::env::split_paths(&std::env::var_os("PATH")?)
        .map(|d| d.join(name))
        .find(|p| p.is_file())
}

/// A macOS reims-util has installers for, as `online-installer list` gives it.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Version {
    /// "macOS Tahoe 26".
    pub name: String,
    /// What names it on reims-util's command line: "tahoe".
    pub slug: String,
}

impl Version {
    /// Its number: "26", "10.15".
    pub fn number(&self) -> &str {
        self.name.rsplit(' ').next().unwrap_or_default()
    }

    /// Whether Apple has full installers for it, which an offline installer holds: macOS 11 and
    /// later.
    pub fn has_offline(&self) -> bool {
        major(self.number()) >= 11
    }
}

/// The versions there are installers for, newest first. reims-util knows them without the
/// network.
pub fn versions() -> Result<Vec<Version>> {
    let r = run(&["online-installer".as_ref(), "list".as_ref()], &mut |_| {})?;
    serde_json::from_value(r).context("read reims-util's list of versions")
}

/// macOS to download for a new VM.
#[derive(Clone, Debug, PartialEq)]
pub struct Download {
    /// What reims-util takes: a name ("tahoe"), a number ("26") or `latest`, and for an offline
    /// installer also a release ("26.6.2") or a build ("25G83").
    pub version: String,
    /// An offline installer, of one release, which installs with no network, rather than an
    /// online one, Apple's recovery image, whose Reinstall macOS downloads the newest release.
    pub offline: bool,
}

/// An installer reims-util made.
#[derive(Clone, Debug)]
pub struct Installer {
    pub image: PathBuf,
    /// Its macOS's version: the macOS's number ("26") for an online installer, the release
    /// ("26.6.2") for an offline one.
    pub version: String,
}

#[derive(Deserialize)]
struct OnlineResult {
    title: String,
    image: PathBuf,
}

#[derive(Deserialize)]
struct OfflineResult {
    installer: OfflineRelease,
    image: PathBuf,
}

#[derive(Deserialize)]
struct OfflineRelease {
    version: String,
}

/// Downloads `d`'s installer into the installers directory, or continues its download, or
/// checks the one there.
pub fn installer(d: &Download, progress: &mut dyn FnMut(&str)) -> Result<Installer> {
    let v = d.version.trim();
    if v.is_empty() || v.eq_ignore_ascii_case("list") {
        bail!(
            "give the macOS to download: a name (tahoe), a number (26) or latest; \
             reims-util online-installer list lists them"
        );
    }
    let dir = paths::installers_dir();
    std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    let _lock = lock(&dir, progress)?;
    let kind = if d.offline {
        "offline-installer"
    } else {
        "online-installer"
    };
    let r = run(
        &[kind.as_ref(), v.as_ref(), "-o".as_ref(), dir.as_os_str()],
        progress,
    )?;
    Ok(if d.offline {
        let o: OfflineResult = serde_json::from_value(r).context("read reims-util's result")?;
        Installer {
            image: o.image,
            version: o.installer.version,
        }
    } else {
        let o: OnlineResult = serde_json::from_value(r).context("read reims-util's result")?;
        Installer {
            image: o.image,
            version: o.title.rsplit(' ').next().unwrap_or_default().to_string(),
        }
    })
}

#[derive(Deserialize)]
struct OpencoreResult {
    smbios: Smbios,
}

#[derive(Deserialize)]
struct Smbios {
    model: String,
    serial: String,
}

/// The arguments that add kexts to the OpenCore image of a VM on `graphics`, beyond Lilu, which
/// reims-util adds itself: on reims-vgpu's own device, each of the guest's
/// kexts ([`prebuilt::guest_kexts`], written to the data directory first), in the set's order,
/// which is the order OpenCore loads them in; none on another.
pub fn kext_args(graphics: Graphics) -> Result<Vec<String>> {
    if graphics != Graphics::ReimsVgpu {
        return Ok(Vec::new());
    }
    add_kexts(prebuilt::set()?, &prebuilt::guest_kexts()?)
}

/// `--add-kext PATH[,MIN[,MAX]]` for each of `set`, at `bundles`: its bundle's path, then the
/// Darwin versions it loads on, as reims-util takes them. A bound only of the newest is given
/// with the oldest, 0.0.0, before it.
fn add_kexts(set: &[Kext], bundles: &[PathBuf]) -> Result<Vec<String>> {
    let mut args = Vec::new();
    for (k, b) in set.iter().zip(bundles) {
        let mut v = b
            .to_str()
            .filter(|p| !p.contains(','))
            .with_context(|| {
                format!(
                    "{} is not a path reims-util's --add-kext takes (UTF-8, with no ','): set \
                     XDG_DATA_HOME to a directory whose path is",
                    b.display()
                )
            })?
            .to_string();
        match (k.min_kernel, k.max_kernel) {
            (None, None) => {}
            (Some(min), None) => v.push_str(&format!(",{min}")),
            (min, Some(max)) => v.push_str(&format!(",{},{max}", min.unwrap_or("0.0.0"))),
        }
        args.extend(["--add-kext".to_string(), v]);
    }
    Ok(args)
}

/// Makes an OpenCore image at `image`, with `OVMF_CODE.fd` and `OVMF_VARS.fd` beside it, for a VM
/// whose network card is `mac`, which becomes its ROM, as a Mac's ROM is its first network
/// card's address, and which installs macOS `version` where it is known; `kexts` adds kexts
/// ([`kext_args`]).
pub fn opencore(
    image: &Path,
    mac: &str,
    version: Option<&str>,
    kexts: &[String],
    progress: &mut dyn FnMut(&str),
) -> Result<()> {
    let mut args = vec!["--rom", mac];
    if let Some(m) = version.and_then(model_for) {
        args.extend(["--model", m]);
    }
    args.extend(kexts.iter().map(String::as_str));
    make_opencore(image, &args, progress)
}

/// Makes an OpenCore image at `image` presenting the Mac `id`, for a VM that has its OVMF: no
/// OVMF is written. `show_picker` shows OpenCore's boot picker at each start, which reims-util
/// hides by default; `kexts` adds kexts ([`kext_args`]).
pub fn opencore_as(
    image: &Path,
    id: &Identity,
    show_picker: bool,
    kexts: &[String],
    progress: &mut dyn FnMut(&str),
) -> Result<()> {
    let mut args = vec![
        "--model",
        &id.model,
        "--serial",
        &id.serial,
        "--mlb",
        &id.mlb,
        "--uuid",
        &id.uuid,
        "--rom",
        &id.rom,
        "--no-ovmf",
    ];
    if show_picker {
        args.push("--show-picker");
    }
    args.extend(kexts.iter().map(String::as_str));
    make_opencore(image, &args, progress)
}

fn make_opencore(image: &Path, args: &[&str], progress: &mut dyn FnMut(&str)) -> Result<()> {
    let mut all: Vec<&OsStr> = vec!["opencore".as_ref(), "-o".as_ref(), image.as_os_str()];
    all.extend(args.iter().map(OsStr::new));
    let r = run(&all, progress)?;
    if let Ok(o) = serde_json::from_value::<OpencoreResult>(r) {
        progress(&format!(
            "OpenCore presents the VM as an {}, serial {}",
            o.smbios.model, o.smbios.serial
        ));
    }
    Ok(())
}

/// The Mac model an OpenCore image presents for macOS `version` ("10.14", "26.6.2"), where it is
/// not reims-util's default: the iMac20,1 runs macOS 10.15.6 to 26, and the iMacPro1,1 10.13 to
/// 15.
pub fn model_for(version: &str) -> Option<&'static str> {
    let mut parts = version.trim().split('.');
    let minor = parts.nth(1).and_then(|m| m.parse::<u32>().ok());
    match (major(version), minor) {
        (10, Some(m)) if m < 15 => Some("iMacPro1,1"),
        _ => None,
    }
}

fn major(version: &str) -> u32 {
    version
        .trim()
        .split('.')
        .next()
        .and_then(|m| m.parse().ok())
        .unwrap_or(0)
}

#[derive(Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum Event {
    Step { message: String },
    Progress { done: u64, total: Option<u64> },
    Note { message: String },
    Done { result: serde_json::Value },
    Error { message: String },
}

/// Runs reims-util with `--json` and `args`, passing its steps and notes to `progress`, and its
/// progress as a whole percent of its step, at most once a second but for the first and the
/// last; returns its result.
fn run(args: &[&OsStr], progress: &mut dyn FnMut(&str)) -> Result<serde_json::Value> {
    let bin = binary()?;
    let mut cmd = Command::new(&bin);
    cmd.arg("--json")
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    end_with_parent(&mut cmd);
    let mut child = cmd
        .spawn()
        .with_context(|| format!("run {}", bin.display()))?;
    let stderr = child.stderr.take();
    let err_thread = std::thread::spawn(move || {
        let mut s = String::new();
        if let Some(mut e) = stderr {
            let _ = e.read_to_string(&mut s);
        }
        s
    });
    let mut step = String::new();
    let mut shown: Option<(u64, Instant)> = None;
    let mut result = None;
    let stdout = child.stdout.take().context("reims-util's output")?;
    for line in BufReader::new(stdout).lines() {
        let Ok(event) = serde_json::from_str::<Event>(&line?) else {
            continue;
        };
        match event {
            Event::Step { message } => {
                step = lower_first(&message);
                shown = None;
                progress(&step);
            }
            Event::Note { message } => progress(&message),
            Event::Progress { done, total } => {
                let (at, last, text) = match total {
                    Some(t) if t > 0 => {
                        let pct = done.min(t) * 100 / t;
                        (pct, pct == 100, format!("{step}: {pct}% of {}", size(t)))
                    }
                    // Nothing yet of a size not known yet says nothing.
                    _ if done == 0 => continue,
                    _ => (done / 100_000_000, false, format!("{step}: {}", size(done))),
                };
                let due = match shown {
                    None => true,
                    Some((a, t)) => a != at && (last || t.elapsed() >= Duration::from_secs(1)),
                };
                if due {
                    shown = Some((at, Instant::now()));
                    progress(&text);
                }
            }
            Event::Done { result: r } => result = Some(Ok(r)),
            Event::Error { message } => result = Some(Err(message)),
        }
    }
    let status = child.wait()?;
    let err = err_thread.join().unwrap_or_default();
    match result {
        Some(Ok(r)) if status.success() => Ok(r),
        Some(Err(m)) => bail!("{m}"),
        _ => bail!("reims-util failed ({status}): {}", err.trim()),
    }
}

/// "Downloading OpenCore 1.0.8" as reims-ctl's steps read: "downloading OpenCore 1.0.8". A step
/// that starts with a name, "OVMF …", stays as it is.
fn lower_first(s: &str) -> String {
    let mut c = s.chars();
    match (c.next(), c.next()) {
        (Some(a), Some(b)) if a.is_uppercase() && b.is_lowercase() => {
            a.to_lowercase().chain(s[a.len_utf8()..].chars()).collect()
        }
        _ => s.to_string(),
    }
}

/// "6 MB", "1.1 GB".
fn size(bytes: u64) -> String {
    if bytes < 1_000_000_000 {
        format!("{} MB", bytes.div_ceil(1_000_000))
    } else {
        format!("{:.1} GB", bytes as f64 / 1e9)
    }
}

/// On Linux, has reims-util end when the reims-ctl thread that started it does, which waits for
/// it, so with reims-ctl: a download stops where it is when the interface quits, and the next
/// one continues it.
fn end_with_parent(cmd: &mut Command) {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::process::CommandExt;
        let parent = std::process::id() as libc::pid_t;
        // SAFETY: prctl, getppid and _exit are async-signal-safe, as pre_exec requires.
        unsafe {
            cmd.pre_exec(move || {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                // The parent ended before the signal was asked for.
                if libc::getppid() != parent {
                    libc::_exit(1);
                }
                Ok(())
            });
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = cmd;
}

/// The installers directory's lock, held while it lives: two VMs made from the same macOS at
/// once share one download, rather than both writing its file. The lock goes with reims-ctl,
/// and reims-util, which does not inherit it, ends with reims-ctl ([`end_with_parent`]).
struct Lock {
    _file: std::fs::File,
}

fn lock(dir: &Path, progress: &mut dyn FnMut(&str)) -> Result<Lock> {
    use std::os::fd::AsRawFd;
    let path = dir.join(".lock");
    let f = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&path)
        .with_context(|| format!("open {}", path.display()))?;
    // SAFETY: flock on a descriptor this function holds open.
    if unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        progress("waiting for another download of a macOS installer to end");
        // SAFETY: as above.
        if unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX) } != 0 {
            bail!(
                "lock {}: {}",
                path.display(),
                std::io::Error::last_os_error()
            );
        }
    }
    Ok(Lock { _file: f })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_macos_gets_a_model_that_runs_it() {
        assert_eq!(model_for("10.13"), Some("iMacPro1,1"));
        assert_eq!(model_for("10.14"), Some("iMacPro1,1"));
        assert_eq!(model_for("10.15"), None);
        assert_eq!(model_for("26"), None);
        assert_eq!(model_for("26.6.2"), None);
        assert_eq!(model_for(""), None);
    }

    #[test]
    fn versions_read_reims_utils_list() {
        let list: Vec<Version> = serde_json::from_str(
            r#"[{"name":"macOS Tahoe 26","slug":"tahoe","board_id":"Mac-CFF7D910A743CAAF"},
                {"name":"macOS Catalina 10.15","slug":"catalina","board_id":"Mac-00BE6ED71E35EB86"}]"#,
        )
        .unwrap();
        assert_eq!(list[0].slug, "tahoe");
        assert_eq!(list[0].number(), "26");
        assert!(list[0].has_offline());
        assert_eq!(list[1].number(), "10.15");
        assert!(!list[1].has_offline());
    }

    /// Each kext of the set is added in its order, by its bundle's path and its bounds; a path
    /// reims-util would read past is refused.
    #[test]
    fn kexts_are_added_in_the_sets_order_with_their_bounds() {
        static SET: &[Kext] = &[
            Kext {
                bundle: "A.kext",
                executable: None,
                min_kernel: Some("25.4.0"),
                max_kernel: Some("25.99.99"),
                files: &[],
            },
            Kext {
                bundle: "B.kext",
                executable: None,
                min_kernel: None,
                max_kernel: Some("25.99.99"),
                files: &[],
            },
            Kext {
                bundle: "C.kext",
                executable: None,
                min_kernel: Some("25.4.0"),
                max_kernel: None,
                files: &[],
            },
            Kext {
                bundle: "D.kext",
                executable: None,
                min_kernel: None,
                max_kernel: None,
                files: &[],
            },
        ];
        let at: Vec<PathBuf> = ["A", "B", "C", "D"]
            .iter()
            .map(|n| PathBuf::from(format!("/d/kexts-1/{n}.kext")))
            .collect();
        assert_eq!(
            add_kexts(SET, &at).unwrap(),
            [
                "--add-kext",
                "/d/kexts-1/A.kext,25.4.0,25.99.99",
                "--add-kext",
                "/d/kexts-1/B.kext,0.0.0,25.99.99",
                "--add-kext",
                "/d/kexts-1/C.kext,25.4.0",
                "--add-kext",
                "/d/kexts-1/D.kext",
            ]
        );
        let comma = vec![PathBuf::from("/d,e/A.kext")];
        assert!(add_kexts(&SET[..1], &comma).is_err());
        assert!(kext_args(Graphics::VmwareSvga).unwrap().is_empty());
    }

    #[test]
    fn steps_read_as_reims_ctls() {
        assert_eq!(
            lower_first("Downloading OpenCore 1.0.8"),
            "downloading OpenCore 1.0.8"
        );
        assert_eq!(lower_first("OVMF's code"), "OVMF's code");
        assert_eq!(size(6_200_000), "7 MB");
        assert_eq!(size(1_100_000_000), "1.1 GB");
    }

    /// A stand-in for reims-util that prints `lines` and exits with `code`.
    fn fake(dir: &Path, lines: &str, code: i32) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let p = dir.join("reims-util");
        std::fs::write(
            &p,
            format!("#!/bin/sh\ncat <<'EOF'\n{lines}\nEOF\necho oops >&2\nexit {code}\n"),
        )
        .unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        p
    }

    #[test]
    fn run_passes_steps_and_percents_a_second_apart_and_returns_the_result() {
        let d = std::env::temp_dir().join(format!("reims-ctl-util-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let lines = r#"{"event":"step","message":"Downloading the macOS Tahoe 26 recovery image"}
{"event":"progress","done":0,"total":2000000000}
{"event":"progress","done":10000000,"total":2000000000}
{"event":"progress","done":20000000,"total":2000000000}
{"event":"progress","done":40000000,"total":2000000000}
{"event":"progress","done":2000000000,"total":2000000000}
{"event":"step","message":"Downloading OpenCore 1.0.8"}
{"event":"progress","done":0,"total":null}
{"event":"progress","done":11000000,"total":11000000}
{"event":"note","message":"Wrote it"}
not json
{"event":"done","result":{"title":"macOS Tahoe 26","image":"/x.dmg"}}"#;
        let bin = fake(&d, lines, 0);
        let env = crate::testenv::Env::new("util-run");
        env.set("REIMS_CTL_UTIL", &bin);
        let mut seen = Vec::new();
        let r = run(&["online-installer".as_ref()], &mut |s| {
            seen.push(s.to_string())
        })
        .unwrap();
        assert_eq!(r["image"], "/x.dmg");
        assert_eq!(
            seen,
            [
                "downloading the macOS Tahoe 26 recovery image",
                "downloading the macOS Tahoe 26 recovery image: 0% of 2.0 GB",
                "downloading the macOS Tahoe 26 recovery image: 100% of 2.0 GB",
                "downloading OpenCore 1.0.8",
                "downloading OpenCore 1.0.8: 100% of 11 MB",
                "Wrote it",
            ]
        );
        // Its error event is the error; without one, its stderr.
        fake(
            &d,
            r#"{"event":"error","message":"no online installer for \"x\""}"#,
            1,
        );
        let e = run(&[], &mut |_| {}).unwrap_err();
        assert_eq!(format!("{e:#}"), "no online installer for \"x\"");
        fake(&d, "", 1);
        let e = format!("{:#}", run(&[], &mut |_| {}).unwrap_err());
        assert!(e.ends_with(": oops"), "{e}");
        drop(env);
        std::fs::remove_dir_all(&d).unwrap();
    }
}
