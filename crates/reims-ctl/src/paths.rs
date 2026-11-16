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
//! any socket's name. Paths there are in Windows' own form, `\` between their parts: those
//! reims-ctl makes, those a variable above or the user gives it ([`native`]), and those it
//! prints with the home as `~` ([`tilde`]).

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

/// `p` in the host's own form: on Windows with `\` for each `/` between its parts, as a path a
/// user or a Unix-like shell gives (`C:/vms`) has them, which a part joined to it would print
/// after a `\`; elsewhere as it is.
pub fn native(p: PathBuf) -> PathBuf {
    if cfg!(windows) {
        p.components().collect()
    } else {
        p
    }
}

/// The directory the XDG variable `var` names where it is absolute, else `fallback`'s parts in
/// the home.
fn xdg(var: &str, fallback: &[&str]) -> PathBuf {
    match std::env::var_os(var) {
        Some(v) if Path::new(&v).is_absolute() => native(PathBuf::from(v)),
        _ => fallback.iter().fold(home(), |p, part| p.join(part)),
    }
}

pub fn config_dir() -> PathBuf {
    match std::env::var_os("REIMS_CTL_CONFIG_DIR") {
        Some(v) => native(PathBuf::from(v)),
        None => xdg("XDG_CONFIG_HOME", &[".config"]).join("reims-ctl"),
    }
}

pub fn data_dir() -> PathBuf {
    match std::env::var_os("REIMS_CTL_DATA_DIR") {
        Some(v) => native(PathBuf::from(v)),
        None => xdg("XDG_DATA_HOME", &[".local", "share"]).join("reims-ctl"),
    }
}

pub fn cache_dir() -> PathBuf {
    xdg("XDG_CACHE_HOME", &[".cache"]).join("reims-ctl")
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
        return native(PathBuf::from(v));
    }
    let d = home().join("Downloads");
    if d.is_dir() { d } else { home() }
}

/// `~` and `~/...` expanded (on Windows `~\...` too), the rest as given, in the host's own form
/// ([`native`]).
pub fn expand(p: &str) -> PathBuf {
    let p = p.trim();
    let rest = p
        .strip_prefix("~/")
        .or_else(|| p.strip_prefix("~\\").filter(|_| cfg!(windows)));
    native(if p == "~" {
        home()
    } else if let Some(rest) = rest {
        home().join(rest)
    } else {
        PathBuf::from(p)
    })
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

/// `p` with the home shown as `~`, for display, in the host's own form ([`native`]): `~/a/b`, and
/// on Windows `~\a\b`.
pub fn tilde(p: &Path) -> String {
    let h = home();
    match p.strip_prefix(&h) {
        Ok(rest) if !h.as_os_str().is_empty() && h != Path::new("/") => {
            if rest.as_os_str().is_empty() {
                "~".into()
            } else {
                format!(
                    "~{}{}",
                    std::path::MAIN_SEPARATOR,
                    native(rest.to_path_buf()).display()
                )
            }
        }
        _ => native(p.to_path_buf()).display().to_string(),
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
        assert_eq!(tilde(&h.join("a").join("b")), format!("~{SEP}a{SEP}b"));
        let elsewhere = Path::new(SEP).join("elsewhere").join("x");
        assert_eq!(tilde(&elsewhere), format!("{SEP}elsewhere{SEP}x"));
    }

    use std::path::MAIN_SEPARATOR_STR as SEP;

    /// On Windows, the paths reims-ctl makes and prints have `\` alone between their parts:
    /// those of its own directories, a path with `/` a user or a variable gives, and one the home
    /// as `~` starts; and `~\` expands as `~/` does.
    #[cfg(windows)]
    #[test]
    fn windows_paths_have_backslashes_alone() {
        let h = home();
        let shown = |p: &Path| p.display().to_string();
        for p in [
            config_dir(),
            data_dir(),
            cache_dir(),
            qemu_root(),
            default_vms_dir(),
        ] {
            assert!(!shown(&p).contains('/'), "{}", shown(&p));
        }
        assert_eq!(shown(&expand("C:/vms/a")), r"C:\vms\a");
        assert_eq!(shown(&expand("~/vms/a")), format!(r"{}\vms\a", shown(&h)));
        assert_eq!(shown(&expand(r"~\vms\a")), format!(r"{}\vms\a", shown(&h)));
        assert_eq!(
            tilde(&h.join(".local/share").join("reims-ctl")),
            r"~\.local\share\reims-ctl"
        );
        assert_eq!(tilde(Path::new("C:/vms/a")), r"C:\vms\a");
    }

    /// Elsewhere a path is printed as given, and `~\` is a name like any other.
    #[cfg(unix)]
    #[test]
    fn unix_paths_are_kept_as_given() {
        assert_eq!(native(PathBuf::from("a//b/./c")), PathBuf::from("a//b/./c"));
        assert_eq!(expand(r"~\vms"), PathBuf::from(r"~\vms"));
        assert_eq!(tilde(&home().join(".local/share")), "~/.local/share");
    }
}
