//! What scripts/install-ctl.sh installed: the QEMU builds live in the data directory's
//! `qemu/<build>`, each a whole install prefix (QEMU finds its data files relative to its
//! binary), with `qemu/current` linking the newest, and `install.toml` beside them names the
//! repository the install built from, its commit and the QEMU build. An update installs a new
//! build beside the others and moves the link, so a running VM's binary is never written.
//!
//! The record is where the install's files are: an install into another data directory (an
//! `XDG_DATA_HOME` of its own, as a scratch install has) records itself there, and leaves the
//! user's record as it was. A reims-ctl before 2026-11-18 kept the record in the config
//! directory, which such an install overwrote while its QEMU went elsewhere. A record there is
//! still read where the data directory has none, and an install's record replaces it where it
//! named a QEMU build of the same data directory ([`Install::save`]).
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
    /// The data directory's record, else an earlier reims-ctl's in the config directory.
    pub fn load() -> Option<Install> {
        let text = std::fs::read_to_string(paths::install_file())
            .or_else(|_| std::fs::read_to_string(paths::config_install_file()))
            .ok()?;
        toml::from_str(&text).ok()
    }

    /// Writes the record into the data directory, and removes an earlier reims-ctl's record in
    /// the config directory where that one named a QEMU build of this data directory: the same
    /// install's, which this replaces. One of another data directory's install stays.
    pub fn save(&self) -> Result<()> {
        let file = paths::install_file();
        std::fs::create_dir_all(file.parent().unwrap())?;
        std::fs::write(&file, toml::to_string_pretty(self)?)
            .with_context(|| format!("write {}", file.display()))?;
        let old = paths::config_install_file();
        if let Ok(text) = std::fs::read_to_string(&old)
            && let Ok(i) = toml::from_str::<Install>(&text)
            && i.qemu.starts_with(paths::qemu_root())
        {
            let _ = std::fs::remove_file(&old);
        }
        Ok(())
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

/// What `qemu -device qemu64-x86_64-cpu,help` listed, by binary and property.
type CpuProbes = HashMap<(PathBuf, &'static str), Option<bool>>;

/// Whether `qemu`'s x86-64 CPUs have the property `name`, which `-device qemu64-x86_64-cpu,help`
/// lists (every x86-64 CPU model has the same properties), cached per binary; None where QEMU does
/// not run. A QEMU ends at its start on a `-cpu` property it does not know.
pub fn cpu_has_property(qemu: &Path, name: &'static str) -> Option<bool> {
    static CACHE: OnceLock<Mutex<CpuProbes>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    let key = (qemu.to_path_buf(), name);
    if let Some(v) = cache.lock().unwrap().get(&key) {
        return *v;
    }
    let has = std::process::Command::new(qemu)
        .args(["-device", "qemu64-x86_64-cpu,help"])
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| lists_property(&String::from_utf8_lossy(&o.stdout), name));
    cache.lock().unwrap().insert(key, has);
    has
}

/// Whether `-device TYPE,help`'s output lists the property `name`, on a line `  name=<type>`.
fn lists_property(out: &str, name: &str) -> bool {
    out.lines().any(|l| {
        l.trim_start()
            .strip_prefix(name)
            .is_some_and(|r| r.starts_with("=<"))
    })
}

/// What `qemu -smbios type=17,memory_type=26 -version` answered, by binary and its modification
/// time.
type SmbiosProbes = HashMap<(PathBuf, Option<std::time::SystemTime>), bool>;

/// Whether `qemu` takes `-smbios type=17`'s `memory_type`, the memory devices' SMBIOS Memory
/// Type, which reims-vgpu's QEMU has since vendor/qemu's `0c91b471`. A QEMU without it, upstream's
/// or one built from an older tree (`REIMS_CTL_QEMU` names any), ends at its start on the option.
/// QEMU takes its options in order, so it checks `-smbios` and exits with an error before
/// `-version` prints and exits with 0. Cached per binary, by its path and modification time; false
/// where QEMU does not run.
pub fn smbios_takes_memory_type(qemu: &Path) -> bool {
    static CACHE: OnceLock<Mutex<SmbiosProbes>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    let mtime = std::fs::metadata(qemu).and_then(|m| m.modified()).ok();
    let key = (qemu.to_path_buf(), mtime);
    if let Some(v) = cache.lock().unwrap().get(&key) {
        return *v;
    }
    let takes = std::process::Command::new(qemu)
        .args(["-smbios", "type=17,memory_type=26", "-version"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    cache.lock().unwrap().insert(key, takes);
    takes
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

    /// A CPU property is found by its whole name in `-device qemu64-x86_64-cpu,help`'s lines, as
    /// QEMU's object_property_help writes them, and not by a name it starts or ends.
    #[test]
    fn a_cpu_property_is_listed_by_its_whole_name() {
        let out = "qemu64-x86_64-cpu options:\n  \
                   avx512f=<bool>\n  \
                   x-avx512-xstate=<bool>  - on/off (default: false)\n  \
                   xsave=<bool>\n";
        assert!(lists_property(out, "x-avx512-xstate"));
        assert!(lists_property(out, "avx512f"));
        assert!(!lists_property(out, "avx512"));
        assert!(!lists_property(out, "avx512-xstate"));
        assert!(!lists_property(out, "x-avx512"));
        let none = "There are no options for qemu64-x86_64-cpu.\n";
        assert!(!lists_property(none, "x-avx512-xstate"));
    }

    /// A QEMU that does not run, or that refuses the probe's options, takes no `memory_type`:
    /// this test binary stands for one that refuses them, as libtest ends on an option it does
    /// not know.
    #[test]
    fn a_qemu_that_refuses_the_probe_takes_no_memory_type() {
        let missing = std::env::temp_dir().join("reims-ctl-no-such-qemu");
        assert!(!smbios_takes_memory_type(&missing));
        let refuses = std::env::current_exe().unwrap();
        assert!(!smbios_takes_memory_type(&refuses));
    }

    /// The installer's prune keeps a build older than the newest three while a QEMU runs from
    /// it, started as reims-ctl starts it, through `qemu/current`, which a newer build then
    /// took; and removes the old builds nothing runs. The test runs on Linux alone, whose /proc
    /// names each process's executable.
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

    /// Runs `body` in bash with the installer's `set_aside` and `put_back` defined, in `dir`;
    /// returns its standard output.
    fn with_set_aside(dir: &Path, body: &str) -> String {
        let manifest = std::env::var_os("CARGO_MANIFEST_DIR")
            .expect("CARGO_MANIFEST_DIR: run the tests with cargo");
        let text =
            std::fs::read_to_string(Path::new(&manifest).join("../../scripts/install-ctl.sh"))
                .unwrap();
        let functions: String = ["set_aside", "put_back"]
            .iter()
            .map(|f| script_function(&text, f))
            .collect();
        let out = std::process::Command::new("bash")
            .arg("-c")
            .arg(format!("set -euo pipefail\n{functions}{body}"))
            .current_dir(dir)
            // On Windows, MSYS2's bash, as the installer runs in.
            .env("MSYSTEM", "CLANG64")
            .env("CHERE_INVOKING", "1")
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap()
    }

    /// The installer's set-aside of a program it is about to install over: the program renamed
    /// to the first free `NAME.old-N` and a copy of it left at its path; after the install, the
    /// original put back where the install failed and left the copy or nothing, and kept aside
    /// where it wrote a new program; nothing for a program not installed yet.
    #[test]
    fn the_installer_sets_a_program_aside_and_puts_it_back_where_the_install_wrote_none() {
        let d = std::env::temp_dir().join(format!("reims-ctl-aside-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let read = |n: &str| std::fs::read_to_string(d.join(n)).ok();
        std::fs::write(d.join("prog.exe"), "one").unwrap();

        // An install that failed and left the copy: the original goes back.
        assert_eq!(with_set_aside(&d, "set_aside prog.exe"), "prog.exe.old-0");
        assert_eq!(read("prog.exe").as_deref(), Some("one"));
        assert_eq!(read("prog.exe.old-0").as_deref(), Some("one"));
        with_set_aside(&d, "put_back prog.exe prog.exe.old-0");
        assert_eq!(read("prog.exe").as_deref(), Some("one"));
        assert_eq!(read("prog.exe.old-0"), None);

        // An install that wrote a new program: the original stays aside, for a start to remove.
        let aside = with_set_aside(
            &d,
            "a=$(set_aside prog.exe); echo two >prog.exe; \
             put_back prog.exe \"$a\"; printf %s \"$a\"",
        );
        assert_eq!(aside, "prog.exe.old-0");
        assert_eq!(read("prog.exe").as_deref(), Some("two\n"));
        assert_eq!(read("prog.exe.old-0").as_deref(), Some("one"));

        // An install that removed the program and failed: the original goes back, under the next
        // free name while the last is still aside.
        let aside = with_set_aside(
            &d,
            "a=$(set_aside prog.exe); rm prog.exe; \
             put_back prog.exe \"$a\"; printf %s \"$a\"",
        );
        assert_eq!(aside, "prog.exe.old-1");
        assert_eq!(read("prog.exe").as_deref(), Some("two\n"));
        assert_eq!(read("prog.exe.old-1"), None);
        assert_eq!(read("prog.exe.old-0").as_deref(), Some("one"));

        // A program not installed yet: nothing set aside.
        assert_eq!(with_set_aside(&d, "set_aside none.exe"), "");
        assert!(!d.join("none.exe").exists());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Windows refuses to replace a running program's file on NTFS, as cargo install renames its
    /// new build over it ("Access is denied"; a ReFS Dev Drive allows it), and lets the
    /// installer's set-aside rename it: the install then replaces the copy, the running program
    /// keeps running the file set aside, and a start removes that file once the program has
    /// ended.
    #[cfg(windows)]
    #[test]
    fn a_running_program_set_aside_on_windows_is_replaced_and_removed_once_it_ends() {
        let d = std::env::temp_dir().join(format!("reims-ctl-aside-win-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let prog = d.join("reims-ctl.exe");
        std::fs::copy(std::env::current_exe().unwrap(), &prog).unwrap();
        let mut running = std::process::Command::new(&prog)
            .args(["--exact", "install::tests::idle_as_a_qemu"])
            .env("REIMS_CTL_TEST_IDLE", "1")
            .stdout(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let aside = with_set_aside(&d, "set_aside reims-ctl.exe");
        // cargo install's rename of its new build over the program.
        let new = d.join("reims-ctl.exe.new");
        std::fs::write(&new, "new").unwrap();
        let replaced = std::fs::rename(&new, &prog);
        with_set_aside(&d, &format!("put_back reims-ctl.exe {aside}"));
        let kept = std::fs::read(&prog).ok();
        // The file set aside is the one the program runs: on NTFS, the same rename over it is
        // refused.
        let control = d.join("control.exe");
        std::fs::write(&control, "control").unwrap();
        let refused = std::fs::rename(&control, d.join(&aside));
        crate::update::remove_set_aside(&d);
        let still_running = running.try_wait().unwrap().is_none();
        let _ = running.kill();
        let _ = running.wait();
        // Windows lets go of an ended program's file a moment after the process ends.
        let start = std::time::Instant::now();
        while d.join(&aside).exists() && start.elapsed() < std::time::Duration::from_secs(10) {
            crate::update::remove_set_aside(&d);
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        let aside_after = d.join(&aside).exists();
        let _ = std::fs::remove_dir_all(&d);
        if let Err(e) = &refused {
            assert_eq!(e.kind(), std::io::ErrorKind::PermissionDenied, "{e}");
        }
        assert_eq!(aside, "reims-ctl.exe.old-0");
        replaced.unwrap();
        assert_eq!(kept.as_deref(), Some(&b"new"[..]));
        assert!(still_running, "the set-aside ended the running program");
        assert!(!aside_after);
    }

    /// The installer's --skip-qemu keeps the installed QEMU: with none installed it ends at
    /// once, before it builds or installs anything, saying what to do. On a host reims-ctl has
    /// no VMs for (an Intel Mac) it ends before that, saying where it runs them.
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
        if cfg!(any(
            all(any(target_os = "linux", windows), target_arch = "x86_64"),
            all(
                any(target_os = "linux", target_os = "macos"),
                target_arch = "aarch64"
            )
        )) {
            assert!(
                err.contains("--skip-qemu keeps the installed QEMU, and none is installed")
                    && err.contains("run this without --skip-qemu, which builds it"),
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

    /// The install's record is in the data directory, beside the QEMU builds it names: an
    /// install into a scratch data directory records itself there, and the user's record stays
    /// as it was, an earlier reims-ctl's in the config directory too, which the user's own
    /// reims-ctl still reads; the user's next install moves that one to the data directory.
    #[test]
    fn an_install_is_recorded_beside_its_qemu_builds() {
        let env = crate::testenv::Env::new("install-record");
        let record = |commit: &str| Install {
            repo: env.root.join("repo"),
            commit: commit.into(),
            qemu: paths::qemu_root().join(format!("20261118-000000-{commit}")),
            installed_at: String::new(),
            shell: PathBuf::new(),
        };
        let own = env.root.join("data");
        let old = paths::config_install_file();
        std::fs::create_dir_all(old.parent().unwrap()).unwrap();
        std::fs::write(&old, toml::to_string_pretty(&record("aaaaaaa")).unwrap()).unwrap();
        let old_text = std::fs::read(&old).unwrap();
        assert_eq!(Install::load().unwrap().commit, "aaaaaaa");

        let scratch = env.root.join("scratch");
        env.set("REIMS_CTL_DATA_DIR", &scratch);
        record("bbbbbbb").save().unwrap();
        assert!(scratch.join("install.toml").is_file());
        assert_eq!(Install::load().unwrap().commit, "bbbbbbb");
        assert_eq!(std::fs::read(&old).unwrap(), old_text);

        env.set("REIMS_CTL_DATA_DIR", &own);
        assert_eq!(Install::load().unwrap().commit, "aaaaaaa");
        record("ccccccc").save().unwrap();
        assert!(!old.exists());
        assert!(own.join("install.toml").is_file());
        assert_eq!(Install::load().unwrap().commit, "ccccccc");
        env.set("REIMS_CTL_DATA_DIR", &scratch);
        assert_eq!(Install::load().unwrap().commit, "bbbbbbb");
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
