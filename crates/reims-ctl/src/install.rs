//! What scripts/install-ctl.sh installed: `install.toml` in the config directory names the
//! repository it built from, and the QEMU builds live in the data directory's `qemu/<build>`,
//! each a whole install prefix (QEMU finds its data files relative to its binary), with
//! `qemu/current` linking the newest. An update installs a new build beside the others and moves
//! the link, so a running VM's binary is never written.
//!
//! `REIMS_CTL_QEMU` names a QEMU binary to use instead (a build tree's, such as `make qemu`'s
//! `build/qemu/qemu-system-aarch64`); `qemu-img` is taken from beside it.

use crate::host::Profile;
use crate::paths;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

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
        return Ok(PathBuf::from(p));
    }
    let bin = paths::qemu_root()
        .join("current/bin")
        .join(profile.qemu_binary());
    if !bin.exists() {
        bail!(
            "QEMU is not installed ({} is missing): run scripts/install-ctl.sh in your reims-vgpu \
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
            .join("current/bin")
            .join(profile.qemu_binary())
    })
}

/// `qemu-img`, beside the QEMU binary.
pub fn qemu_img(profile: Profile) -> Result<PathBuf> {
    let bin = qemu_binary(profile)?;
    let img = bin.with_file_name("qemu-img");
    if !img.exists() {
        bail!(
            "{} is missing: reinstall with scripts/install-ctl.sh",
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

/// The audio backend to use where the VM names none: the host's native one, among those built.
pub fn default_audio_backend(available: &[String]) -> String {
    let wanted: &[&str] = if cfg!(target_os = "macos") {
        &["coreaudio"]
    } else {
        &["pipewire", "pa", "alsa", "sdl"]
    };
    for w in wanted {
        if available.iter().any(|a| a == w) {
            return (*w).into();
        }
    }
    if available.is_empty() {
        // QEMU could not be asked (not installed yet): its usual choice.
        return if cfg!(target_os = "macos") {
            "coreaudio"
        } else {
            "pipewire"
        }
        .into();
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
    /// took; and removes the old builds nothing runs.
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

    #[test]
    fn audio_backend_prefers_the_native_one() {
        let v = |s: &[&str]| s.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        if cfg!(target_os = "macos") {
            assert_eq!(
                default_audio_backend(&v(&["none", "coreaudio"])),
                "coreaudio"
            );
        } else {
            assert_eq!(
                default_audio_backend(&v(&["none", "alsa", "pa", "pipewire"])),
                "pipewire"
            );
            assert_eq!(default_audio_backend(&v(&["none", "alsa"])), "alsa");
        }
        assert_eq!(default_audio_backend(&v(&["none", "dbus"])), "none");
    }
}
