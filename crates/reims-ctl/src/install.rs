//! What scripts/install-ctl.sh installed: `install.toml` in the config directory names the
//! repository it built from, and the QEMU builds live in the data directory's `qemu/<build>`,
//! each a whole install prefix (QEMU finds its data files relative to its binary), with
//! `qemu/current` linking the newest. An update installs a new build beside the others and moves
//! the link, so a running VM's binary is never written.
//!
//! `REIMS_CTL_QEMU` names a QEMU binary to use instead (a build tree's, such as `make qemu`'s
//! `build/qemu/qemu-system-x86_64`); `qemu-img` is taken from beside it.

use crate::host::Profile;
use crate::paths;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// The script that installs reims-ctl and its QEMU, by its path in the checkout, for the messages
/// that send the user there: on Windows scripts\install.ps1, which runs scripts/install-ctl.sh in
/// MSYS2's CLANG64 shell.
pub const INSTALL_SCRIPT: &str = if cfg!(windows) {
    r"scripts\install.ps1"
} else {
    "scripts/install-ctl.sh"
};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Install {
    /// The reims-vgpu checkout the install built from, which `update` pulls.
    pub repo: PathBuf,
    /// Its commit at the install.
    #[serde(default)]
    pub commit: String,
    /// The QEMU build installed (a directory under the data directory's `qemu`).
    #[serde(default)]
    pub qemu: PathBuf,
    #[serde(default)]
    pub installed_at: String,
    /// The bash that runs the install script on Windows, MSYS2's (`os::install_script`); empty
    /// elsewhere, where `bash` is on PATH.
    #[serde(default, skip_serializing_if = "is_empty")]
    pub shell: PathBuf,
}

fn is_empty(p: &Path) -> bool {
    p.as_os_str().is_empty()
}

impl Install {
    pub fn load() -> Option<Install> {
        let text = std::fs::read_to_string(paths::install_file()).ok()?;
        toml::from_str(&text).ok()
    }

    pub fn save(&self) -> Result<()> {
        let file = paths::install_file();
        std::fs::create_dir_all(file.parent().unwrap())?;
        std::fs::write(&file, toml::to_string_pretty(self)?)
            .with_context(|| format!("write {}", file.display()))
    }
}

/// The QEMU binary for `profile`.
pub fn qemu_binary(profile: Profile) -> Result<PathBuf> {
    if let Some(p) = std::env::var_os("REIMS_CTL_QEMU") {
        return Ok(paths::native(PathBuf::from(p)));
    }
    let bin = paths::qemu_root()
        .join("current")
        .join("bin")
        .join(profile.qemu_binary());
    if !bin.exists() {
        bail!(
            "QEMU is not installed ({} is missing): run {INSTALL_SCRIPT} in your reims-vgpu \
             checkout, or set REIMS_CTL_QEMU to a qemu-system binary",
            bin.display()
        );
    }
    Ok(bin)
}

/// The QEMU binary for `profile`, or where it would be, for printing a command.
pub fn qemu_binary_or_default(profile: Profile) -> PathBuf {
    qemu_binary(profile).unwrap_or_else(|_| {
        paths::qemu_root()
            .join("current")
            .join("bin")
            .join(profile.qemu_binary())
    })
}

/// `qemu-img`, beside the QEMU binary.
pub fn qemu_img(profile: Profile) -> Result<PathBuf> {
    let bin = qemu_binary(profile)?;
    let img = bin.with_file_name(if cfg!(windows) {
        "qemu-img.exe"
    } else {
        "qemu-img"
    });
    if !img.exists() {
        bail!(
            "{} is missing: reinstall with {INSTALL_SCRIPT}",
            img.display()
        );
    }
    Ok(img)
}

/// The audio backends `qemu` was built with (`-audiodev help`), cached per binary.
pub fn audio_backends(qemu: &Path) -> Vec<String> {
    backends(qemu, "-audiodev")
}

/// The display backends `qemu` was built with (`-display help`), cached per binary.
pub fn display_backends(qemu: &Path) -> Vec<String> {
    backends(qemu, "-display")
}

/// What `qemu <option> help` listed, by binary and option.
type Lists = HashMap<(PathBuf, &'static str), Vec<String>>;

/// What `qemu <option> help` lists, a name to a line.
fn backends(qemu: &Path, option: &'static str) -> Vec<String> {
    static CACHE: OnceLock<Mutex<Lists>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    let key = (qemu.to_path_buf(), option);
    if let Some(v) = cache.lock().unwrap().get(&key) {
        return v.clone();
    }
    let out = std::process::Command::new(qemu)
        .args([option, "help"])
        .output();
    let list: Vec<String> = match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout)
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.contains(' ') && !l.ends_with(':'))
            .map(String::from)
            .collect(),
        Err(_) => Vec::new(),
    };
    cache.lock().unwrap().insert(key, list.clone());
    list
}

/// What `qemu -device NAME,help` answered, by binary and device.
type Probes = HashMap<(PathBuf, &'static str), Option<bool>>;

/// Whether `qemu` was built with the device `name` (`-device NAME,help` lists its properties,
/// and fails for a device it does not have), cached per binary; None where it does not run.
pub fn has_device(qemu: &Path, name: &'static str) -> Option<bool> {
    static CACHE: OnceLock<Mutex<Probes>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    let key = (qemu.to_path_buf(), name);
    if let Some(v) = cache.lock().unwrap().get(&key) {
        return *v;
    }
    let has = std::process::Command::new(qemu)
        .args(["-device", &format!("{name},help")])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .ok()
        .map(|s| s.success());
    cache.lock().unwrap().insert(key, has);
    has
}

/// The audio backend to use where the VM names none: the host's native one, among those built
/// (DirectSound's on Windows).
pub fn default_audio_backend(available: &[String]) -> String {
    for w in ["pipewire", "pa", "alsa", "dsound", "sdl"] {
        if available.iter().any(|a| a == w) {
            return w.into();
        }
    }
    if available.is_empty() {
        // QEMU could not be asked (not installed yet): its usual choice.
        return if cfg!(windows) { "dsound" } else { "pipewire" }.into();
    }
    "none".into()
}

/// `qemu --version`'s first line.
pub fn qemu_version(qemu: &Path) -> Option<String> {
    let o = std::process::Command::new(qemu)
        .arg("--version")
        .output()
        .ok()?;
    String::from_utf8_lossy(&o.stdout)
        .lines()
        .next()
        .map(String::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shell function `name` of scripts/install-ctl.sh, as the script defines it.
    fn script_function(script: &str, name: &str) -> String {
        let start = script.find(&format!("\n{name}() {{")).expect(name) + 1;
        let len = script[start..].find("\n}\n").expect(name) + 3;
        script[start..start + len].to_string()
    }

    /// What the prune test runs as a QEMU: this test binary, copied into a build, idling here.
    #[test]
    fn idle_as_a_qemu() {
        if std::env::var_os("REIMS_CTL_TEST_IDLE").is_some() {
            std::thread::sleep(std::time::Duration::from_secs(60));
        }
    }

    /// The installer's prune keeps a build older than the newest three while a QEMU runs from
    /// it, started as reims-ctl starts it, through `qemu/current`, which a newer build then
    /// took; and removes the old builds nothing runs. The installer runs on Linux alone, whose
    /// /proc names each process's executable.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_installers_prune_keeps_a_build_a_running_qemu_executes() {
        // The crate's directory, which cargo and nextest give a test process: read at run time, so
        // the test is the same binary in every worktree (docs/development.md, "Building through
        // kache").
        let dir = std::env::var_os("CARGO_MANIFEST_DIR")
            .expect("CARGO_MANIFEST_DIR: run the tests with cargo");
        let script = Path::new(&dir).join("../../scripts/install-ctl.sh");
        let text = std::fs::read_to_string(&script).unwrap();
        let functions: String = ["qemu_executables", "prune_qemu"]
            .iter()
            .map(|f| script_function(&text, f))
            .collect();
        let root = std::env::temp_dir().join(format!("reims-ctl-prune-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let qemu = root.join("qemu");
        let builds = [
            "20260101-000000-a",
            "20260102-000000-b",
            "20260103-000000-c",
        ]
        .into_iter()
        .chain(["20260104-000000-d", "20260105-000000-e"])
        .collect::<Vec<_>>();
        for b in &builds {
            std::fs::create_dir_all(qemu.join(b).join("bin")).unwrap();
        }
        let bin = "bin/qemu-system-test";
        std::fs::copy(
            std::env::current_exe().unwrap(),
            qemu.join(builds[0]).join(bin),
        )
        .unwrap();
        let link = |b: &str| {
            let _ = std::fs::remove_file(qemu.join("current"));
            std::os::unix::fs::symlink(b, qemu.join("current")).unwrap();
        };
        link(builds[0]);
        let spawn = || {
            std::process::Command::new(qemu.join("current").join(bin))
                .args(["--exact", "install::tests::idle_as_a_qemu"])
                .env("REIMS_CTL_TEST_IDLE", "1")
                .stdout(std::process::Stdio::null())
                .spawn()
        };
        // The copy is busy for as long as a process another test forked before its copy closed
        // holds it open, until that process execs: Linux refuses to exec it then (ETXTBSY).
        let mut tries = 0;
        let mut running = loop {
            match spawn() {
                Err(e) if e.kind() == std::io::ErrorKind::ExecutableFileBusy && tries < 200 => {
                    tries += 1;
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                r => break r.unwrap(),
            }
        };
        link(builds[4]);
        let pruned = std::process::Command::new("bash")
            .arg("-c")
            .arg(format!("set -euo pipefail\n{functions}prune_qemu \"$1\""))
            .arg("prune")
            .arg(&qemu)
            .status();
        let _ = running.kill();
        let _ = running.wait();
        assert!(pruned.unwrap().success());
        let left: Vec<_> = builds.iter().filter(|b| qemu.join(b).exists()).collect();
        let _ = std::fs::remove_dir_all(&root);
        assert_eq!(left, [&builds[0], &builds[2], &builds[3], &builds[4]]);
    }

    /// On Windows the installer's prune removes no build while a QEMU runs from the QEMU
    /// directory, as reims-ctl starts it, through `qemu/current`, a junction there: Windows names
    /// a running program by the path it started from, which a newer build then took. With none
    /// running it keeps the newest three.
    #[cfg(windows)]
    #[test]
    fn the_installers_prune_keeps_every_build_while_a_qemu_runs_on_windows() {
        let dir = std::env::var_os("CARGO_MANIFEST_DIR")
            .expect("CARGO_MANIFEST_DIR: run the tests with cargo");
        let script = Path::new(&dir).join("../../scripts/install-ctl.sh");
        let text = std::fs::read_to_string(&script).unwrap();
        let functions: String = ["qemu_executables", "prune_qemu"]
            .iter()
            .map(|f| script_function(&text, f))
            .collect();
        let root = std::env::temp_dir().join(format!("reims-ctl-prune-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let qemu = root.join("qemu");
        let builds = [
            "20260101-000000-a",
            "20260102-000000-b",
            "20260103-000000-c",
            "20260104-000000-d",
            "20260105-000000-e",
        ];
        for b in &builds {
            std::fs::create_dir_all(qemu.join(b).join("bin")).unwrap();
        }
        let bin = "bin/qemu-system-test.exe";
        std::fs::copy(
            std::env::current_exe().unwrap(),
            qemu.join(builds[2]).join(bin),
        )
        .unwrap();
        let current = qemu.join("current");
        let link = |b: &str| {
            let _ = std::process::Command::new("cmd")
                .args(["/c", "rmdir"])
                .arg(&current)
                .output();
            let made = std::process::Command::new("cmd")
                .args(["/c", "mklink", "/J"])
                .arg(&current)
                .arg(qemu.join(b))
                .output()
                .unwrap();
            assert!(made.status.success(), "{made:?}");
        };
        link(builds[2]);
        let mut running = std::process::Command::new(current.join(bin))
            .args(["--exact", "install::tests::idle_as_a_qemu"])
            .env("REIMS_CTL_TEST_IDLE", "1")
            .stdout(std::process::Stdio::null())
            .spawn()
            .unwrap();
        link(builds[4]);
        let prune = || {
            std::process::Command::new("bash")
                .arg("-c")
                .arg(format!(
                    "set -euo pipefail\nWINDOWS=1\n{functions}prune_qemu \"$(cygpath -u \"$1\")\""
                ))
                .arg("prune")
                .arg(&qemu)
                .env("MSYSTEM", "CLANG64")
                .status()
        };
        let pruned_running = prune();
        let left_running: Vec<_> = builds.iter().filter(|b| qemu.join(b).exists()).collect();
        let _ = running.kill();
        let _ = running.wait();
        let pruned = prune();
        let left: Vec<_> = builds.iter().filter(|b| qemu.join(b).exists()).collect();
        let _ = std::process::Command::new("cmd")
            .args(["/c", "rmdir"])
            .arg(&current)
            .output();
        let _ = std::fs::remove_dir_all(&root);
        assert!(pruned_running.unwrap().success());
        assert_eq!(left_running, builds.iter().collect::<Vec<_>>());
        assert!(pruned.unwrap().success());
        assert_eq!(left, [&builds[2], &builds[3], &builds[4]]);
    }

    /// The installer's --skip-qemu keeps the installed QEMU: with none installed it ends at
    /// once, before it builds or installs anything, saying what to do, as it does on an arm64
    /// host, which has no QEMU. On a host reims-ctl has no VMs for (an Intel Mac) it ends before
    /// that, saying where it runs and makes them.
    #[test]
    fn the_installer_skips_qemu_only_where_one_is_installed() {
        let dir = std::env::var_os("CARGO_MANIFEST_DIR")
            .expect("CARGO_MANIFEST_DIR: run the tests with cargo");
        let script = Path::new(&dir).join("../../scripts/install-ctl.sh");
        let root = std::env::temp_dir().join(format!("reims-ctl-skip-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let out = std::process::Command::new("bash")
            .arg(&script)
            .arg("--skip-qemu")
            .env("XDG_DATA_HOME", root.join("data"))
            .env("XDG_CACHE_HOME", root.join("cache"))
            // On Windows the script runs in MSYS2's CLANG64 shell, which builds QEMU there.
            .env("MSYSTEM", "CLANG64")
            .stdin(std::process::Stdio::null())
            .output()
            .unwrap();
        let _ = std::fs::remove_dir_all(&root);
        let err = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(1), "{err}");
        if cfg!(all(
            any(target_os = "linux", windows),
            target_arch = "x86_64"
        )) {
            assert!(
                err.contains("--skip-qemu keeps the installed QEMU, and none is installed")
                    && err.contains("run this without --skip-qemu, which builds it"),
                "{err}"
            );
        } else if cfg!(target_arch = "aarch64") {
            assert!(
                err.contains("--skip-qemu keeps the installed QEMU, and this host has none")
                    && err.contains("run this without --skip-qemu"),
                "{err}"
            );
        } else {
            assert!(err.contains(crate::host::UNSUPPORTED), "{err}");
        }
    }

    /// What the installer says of the QEMU build --skip-qemu keeps: nothing where it was built
    /// from what this checkout builds it from, which a change to reims-ctl alone leaves as it
    /// was; else that QEMU or the library changed since its commit, or that the checkout lacks
    /// that commit.
    #[test]
    fn the_installer_says_when_the_qemu_it_keeps_is_not_this_checkouts() {
        let dir = std::env::var_os("CARGO_MANIFEST_DIR")
            .expect("CARGO_MANIFEST_DIR: run the tests with cargo");
        let text =
            std::fs::read_to_string(Path::new(&dir).join("../../scripts/install-ctl.sh")).unwrap();
        let function = script_function(&text, "kept_qemu");
        let repo = std::env::temp_dir().join(format!("reims-ctl-kept-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&repo);
        let git = |args: &[&str]| {
            let out = crate::update::git_command(&repo)
                .args([
                    "-c",
                    "user.name=t",
                    "-c",
                    "user.email=t@t",
                    "-c",
                    "commit.gpgsign=false",
                ])
                .args(args)
                .output()
                .unwrap();
            assert!(out.status.success(), "{args:?}: {out:?}");
            String::from_utf8_lossy(&out.stdout).trim().to_string()
        };
        let commit = |file: &str| {
            let path = repo.join(file);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            let text = std::fs::read_to_string(&path).unwrap_or_default();
            std::fs::write(&path, text + "x\n").unwrap();
            git(&["add", file]);
            git(&["commit", "-q", "-m", file]);
            git(&["rev-parse", "--short", "HEAD"])
        };
        std::fs::create_dir_all(&repo).unwrap();
        git(&["init", "-q"]);
        commit("vendor/qemu/configure");
        let built = commit("crates/reims-vgpu/src/lib.rs");
        commit("crates/reims-ctl/src/main.rs");
        commit("docs/qemu.md");
        let kept = |build: &str| {
            let mut bash = std::process::Command::new("bash");
            for v in crate::update::GIT_REPOSITORY_ENV {
                bash.env_remove(v);
            }
            let out = bash
                .arg("-c")
                .arg(format!("set -euo pipefail\n{function}kept_qemu \"$1\""))
                .arg("kept")
                .arg(format!("/data/reims-ctl/qemu/{build}"))
                .env("ROOT", &repo)
                .output()
                .unwrap();
            assert!(out.status.success(), "{out:?}");
            String::from_utf8(out.stdout).unwrap()
        };
        let this = kept(&format!("20261107-120000-{built}"));
        let unknown = kept("20261107-120000-0000000");
        commit("crates/reims-vgpu/src/lib.rs");
        let changed = kept(&format!("20261107-120000-{built}"));
        let _ = std::fs::remove_dir_all(&repo);
        assert_eq!(this, "");
        assert_eq!(
            unknown,
            "the installed QEMU was built at 0000000, which this checkout does not have: run this \
             without --skip-qemu to build this checkout's\n"
        );
        assert_eq!(
            changed,
            format!(
                "the installed QEMU was built at {built}, and QEMU or the library it links changed \
                 since: run this without --skip-qemu to build this checkout's\n"
            )
        );
    }

    #[test]
    fn audio_backend_prefers_the_native_one() {
        let v = |s: &[&str]| s.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        assert_eq!(
            default_audio_backend(&v(&["none", "alsa", "pa", "pipewire"])),
            "pipewire"
        );
        assert_eq!(default_audio_backend(&v(&["none", "alsa"])), "alsa");
        assert_eq!(default_audio_backend(&v(&["none", "dbus"])), "none");
    }
}
