//! Tests that set reims-ctl's environment (its directories, the QEMU and reims-util it runs)
//! hold an [`Env`] while they run: the environment is the process's, shared by the tests that
//! run at once.

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

static LOCK: Mutex<()> = Mutex::new(());

const VARS: [&str; 4] = [
    "REIMS_CTL_CONFIG_DIR",
    "REIMS_CTL_DATA_DIR",
    "REIMS_CTL_QEMU",
    "REIMS_CTL_UTIL",
];

/// A temporary directory holding reims-ctl's config and data directories, `config` and
/// `data`, named in the environment until it drops, with no QEMU or reims-util named.
pub struct Env {
    pub root: PathBuf,
    saved: Vec<(&'static str, Option<OsString>)>,
    _lock: MutexGuard<'static, ()>,
}

impl Env {
    pub fn new(tag: &str) -> Env {
        let lock = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let root = std::env::temp_dir().join(format!("reims-ctl-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let saved = VARS.iter().map(|v| (*v, std::env::var_os(v))).collect();
        let env = Env {
            root,
            saved,
            _lock: lock,
        };
        env.set("REIMS_CTL_CONFIG_DIR", env.root.join("config"));
        env.set("REIMS_CTL_DATA_DIR", env.root.join("data"));
        // SAFETY: the tests that set the environment hold LOCK.
        unsafe {
            std::env::remove_var("REIMS_CTL_QEMU");
            std::env::remove_var("REIMS_CTL_UTIL");
        }
        env
    }

    /// Sets `var`, one of those the Env puts back, until it drops.
    pub fn set(&self, var: &str, value: impl AsRef<OsStr>) {
        assert!(VARS.contains(&var), "{var} is not put back");
        // SAFETY: the tests that set the environment hold LOCK.
        unsafe { std::env::set_var(var, value) };
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        for (var, old) in &self.saved {
            // SAFETY: as in set.
            unsafe {
                match old {
                    Some(v) => std::env::set_var(var, v),
                    None => std::env::remove_var(var),
                }
            }
        }
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// `p` as the inside of a TOML string: Windows' `\` escaped.
pub fn toml_path(p: &Path) -> String {
    p.display().to_string().replace('\\', "\\\\")
}

/// A stand-in program at `path` that runs `script`, a POSIX shell script, and the path to run it
/// by: `path` itself, made executable. On Windows, which runs no script by its `#!` line, `path`
/// holds the script and `path.cmd` runs it in MSYS2's or Git's `sh`, the path returned.
#[cfg(unix)]
pub fn program(path: &Path, script: &str) -> PathBuf {
    std::fs::write(path, script).unwrap();
    crate::os::set_mode(path, 0o755).unwrap();
    path.to_path_buf()
}

/// A stand-in program at `path` that runs `script`, a POSIX shell script, and the path to run it
/// by: `path` itself, made executable. On Windows, which runs no script by its `#!` line, `path`
/// holds the script and `path.cmd` runs it in MSYS2's or Git's `sh`, the path returned.
#[cfg(windows)]
pub fn program(path: &Path, script: &str) -> PathBuf {
    let on_path: Vec<PathBuf> = std::env::var_os("PATH")
        .map(|p| {
            std::env::split_paths(&p)
                .map(|d| d.join("sh.exe"))
                .collect()
        })
        .unwrap_or_default();
    let sh = on_path
        .into_iter()
        .chain([
            PathBuf::from(r"C:\msys64\usr\bin\sh.exe"),
            PathBuf::from(r"C:\Program Files\Git\usr\bin\sh.exe"),
        ])
        .find(|p| p.is_file())
        .expect("the tests' stand-in programs are sh scripts: install MSYS2 or Git for Windows");
    std::fs::write(path, script).unwrap();
    let name = path.file_name().unwrap().to_string_lossy();
    let cmd = path.with_file_name(format!("{name}.cmd"));
    std::fs::write(
        &cmd,
        format!("@\"{}\" \"%~dp0{name}\" %*\r\n", sh.display()),
    )
    .unwrap();
    cmd
}
