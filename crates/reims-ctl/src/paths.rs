//! Where reims-ctl keeps things, the same on macOS and Linux (the XDG layout), so the paths
//! carry no spaces and a printed QEMU command reads the same on both:
//! - `$XDG_CONFIG_HOME/reims-ctl` (`~/.config/reims-ctl`): `vms.toml`, the VMs this user has,
//!   each by its name and directory, and `install.toml`, what scripts/install-ctl.sh installed;
//! - `$XDG_DATA_HOME/reims-ctl` (`~/.local/share/reims-ctl`): `vms/<name>`, the default home of
//!   a new VM, `qemu/<build>`, the QEMU builds, `qemu/current` naming the newest, and
//!   `installers`, the macOS installers reims-util downloaded, which VMs use in place;
//! - `$XDG_CACHE_HOME/reims-ctl` (`~/.cache/reims-ctl`): builds, scripts/install-ctl.sh's and
//!   reims-util's (reims_util.rs);
//! - the runtime directory, for a QMP socket whose path in the VM's directory would be too long
//!   for a Unix socket: `$XDG_RUNTIME_DIR/reims-ctl`, else `$TMPDIR/reims-ctl-<uid>`; where even
//!   a short socket name does not fit there, `/tmp/reims-ctl-<uid>` (`short_runtime_dir`).
//!
//! `REIMS_CTL_CONFIG_DIR` and `REIMS_CTL_DATA_DIR` replace the first two (tests use them).
//!
//! On Windows the layout is the same, in the user's profile (`%USERPROFILE%`), and the runtime
//! directory is `reims-ctl` in the user's own temporary directory (`%TEMP%`), short enough for
//! any socket's name.

use std::path::{Path, PathBuf};

#[cfg(unix)]
pub fn home() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/"))
}

/// The user's profile: `%USERPROFILE%`, which a shell's `HOME` (MSYS2's, Git's) does not replace,
/// so reims-ctl finds the same files from every shell.
#[cfg(windows)]
pub fn home() -> PathBuf {
    std::env::var_os("USERPROFILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("\\"))
}

fn xdg(var: &str, fallback: &str) -> PathBuf {
    match std::env::var_os(var) {
        Some(v) if Path::new(&v).is_absolute() => PathBuf::from(v),
        _ => home().join(fallback),
    }
}

pub fn config_dir() -> PathBuf {
    match std::env::var_os("REIMS_CTL_CONFIG_DIR") {
        Some(v) => PathBuf::from(v),
        None => xdg("XDG_CONFIG_HOME", ".config").join("reims-ctl"),
    }
}

pub fn data_dir() -> PathBuf {
    match std::env::var_os("REIMS_CTL_DATA_DIR") {
        Some(v) => PathBuf::from(v),
        None => xdg("XDG_DATA_HOME", ".local/share").join("reims-ctl"),
    }
}

pub fn cache_dir() -> PathBuf {
    xdg("XDG_CACHE_HOME", ".cache").join("reims-ctl")
}

pub fn installers_dir() -> PathBuf {
    data_dir().join("installers")
}

pub fn default_vms_dir() -> PathBuf {
    data_dir().join("vms")
}

pub fn qemu_root() -> PathBuf {
    data_dir().join("qemu")
}

pub fn registry_file() -> PathBuf {
    config_dir().join("vms.toml")
}

pub fn install_file() -> PathBuf {
    config_dir().join("install.toml")
}

#[cfg(unix)]
pub fn runtime_dir() -> PathBuf {
    if let Some(v) = std::env::var_os("XDG_RUNTIME_DIR").filter(|v| Path::new(v).is_absolute()) {
        return PathBuf::from(v).join("reims-ctl");
    }
    let tmp = std::env::var_os("TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    tmp.join(format!("reims-ctl-{}", crate::os::uid()))
}

/// The runtime directory for a socket that a long `$XDG_RUNTIME_DIR` or `$TMPDIR` leaves no
/// room for: `/tmp/reims-ctl-<uid>`, at most 46 bytes with its socket's name.
#[cfg(unix)]
pub fn short_runtime_dir() -> PathBuf {
    PathBuf::from(format!("/tmp/reims-ctl-{}", crate::os::uid()))
}

/// `reims-ctl` in the user's own temporary directory, `%TEMP%`.
#[cfg(windows)]
pub fn runtime_dir() -> PathBuf {
    std::env::temp_dir().join("reims-ctl")
}

/// The runtime directory, whose sockets' paths fit (`vm::qmp_socket_in`).
#[cfg(windows)]
pub fn short_runtime_dir() -> PathBuf {
    runtime_dir()
}

/// Where a bug report's zip goes: `$XDG_DOWNLOAD_DIR`, else `~/Downloads`, else the home.
pub fn downloads_dir() -> PathBuf {
    if let Some(v) = std::env::var_os("XDG_DOWNLOAD_DIR").filter(|v| Path::new(v).is_absolute()) {
        return PathBuf::from(v);
    }
    let d = home().join("Downloads");
    if d.is_dir() { d } else { home() }
}

/// `~` and `~/...` expanded, the rest as given.
pub fn expand(p: &str) -> PathBuf {
    let p = p.trim();
    if p == "~" {
        home()
    } else if let Some(rest) = p.strip_prefix("~/") {
        home().join(rest)
    } else {
        PathBuf::from(p)
    }
}

/// `p` made absolute against the current directory, `~` expanded, without resolving links.
pub fn absolute(p: &str) -> PathBuf {
    let e = expand(p);
    if e.is_absolute() {
        e
    } else {
        std::env::current_dir().map(|d| d.join(&e)).unwrap_or(e)
    }
}

/// `p` with the home shown as `~`, for display.
pub fn tilde(p: &Path) -> String {
    let h = home();
    match p.strip_prefix(&h) {
        Ok(rest) if !h.as_os_str().is_empty() && h != Path::new("/") => {
            if rest.as_os_str().is_empty() {
                "~".into()
            } else {
                format!("~/{}", rest.display())
            }
        }
        _ => p.display().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_takes_the_tilde_only_at_the_start() {
        let h = home();
        assert_eq!(expand("~"), h);
        assert_eq!(expand("~/vms/a"), h.join("vms/a"));
        assert_eq!(expand("/a/~/b"), PathBuf::from("/a/~/b"));
        assert_eq!(expand("~user/x"), PathBuf::from("~user/x"));
    }

    #[test]
    fn tilde_shows_paths_under_home() {
        let h = home();
        assert_eq!(tilde(&h.join("a/b")), "~/a/b");
        assert_eq!(tilde(Path::new("/elsewhere/x")), "/elsewhere/x");
    }
}
