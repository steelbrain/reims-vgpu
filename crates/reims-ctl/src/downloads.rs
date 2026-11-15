//! What reims-ctl keeps downloaded for new VMs: macOS installers, which reims-util downloads
//! (reims_util.rs), and IPSWs (ipsw.rs), in the data directory's `installers`, and the IPSWs the
//! previous reims-ctl kept in its `ipsw`. Each is kept for the next VM of its macOS, and a
//! download that stopped leaves its part for the next to continue. `reims-ctl downloads` and the
//! interface's Downloads view list them with the space each takes, and delete one: a later VM
//! of its macOS downloads it again.
//!
//! A file a VM boots from (an offline installer, which a VM uses where it is) isn't deleted, nor
//! one being downloaded.

use crate::ipsw;
use crate::paths;
use crate::vm::{Registry, Vm};
use anyhow::{Context, Result, bail};
use reims_restore::catalog;
use std::path::{Path, PathBuf};

/// A file kept for new VMs.
#[derive(Clone, Debug, PartialEq)]
pub struct Item {
    pub path: PathBuf,
    /// What it is, as the list says it: "macOS 27.2.1 (26B131) IPSW".
    pub what: String,
    /// The space it takes on disk.
    pub bytes: u64,
}

impl Item {
    pub fn name(&self) -> String {
        self.path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned()
    }
}

/// The directories downloads are kept in.
pub fn dirs() -> [PathBuf; 2] {
    [paths::installers_dir(), ipsw::legacy_dir()]
}

/// The files kept, by name.
pub fn list() -> Vec<Item> {
    let mut out = Vec::new();
    for d in dirs() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            // reims-util's lock, and anything else hidden
            if name.starts_with('.') {
                continue;
            }
            let Ok(m) = e.metadata() else { continue };
            if !m.is_file() {
                continue;
            }
            use std::os::unix::fs::MetadataExt as _;
            out.push(Item {
                path: e.path(),
                what: what(&name),
                // what it takes on disk: a sparse image takes less than its length
                bytes: (m.blocks() * 512).min(m.len()),
            });
        }
    }
    out.sort_by_key(Item::name);
    out
}

/// What the file `name` is, from its name.
pub fn what(name: &str) -> String {
    if let Some(base) = name.strip_suffix(".part") {
        return format!("{}, partly downloaded", what(base));
    }
    if let Some((v, b)) = catalog::name_parts(name) {
        return format!("macOS {v} ({b}) IPSW");
    }
    if let Some(n) = name
        .strip_prefix("macOS-")
        .and_then(|n| n.strip_suffix("-online-installer.dmg"))
    {
        return format!("macOS {} online installer", n.replace('-', " "));
    }
    if let Some(n) = name
        .strip_prefix("macOS-")
        .and_then(|n| n.strip_suffix("-offline-installer.img"))
        && let [build, version, rest] = n.rsplitn(3, '-').collect::<Vec<_>>()[..]
    {
        return format!(
            "macOS {} {version} ({build}) offline installer",
            rest.replace('-', " ")
        );
    }
    "a file reims-ctl didn't download".into()
}

/// Space, as the list shows it: "25.6 GB".
pub fn size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1e9)
    } else {
        format!("{} MB", bytes.div_ceil(1_000_000))
    }
}

/// Deletes the kept file named `name`; what it freed.
pub fn delete(name: &str) -> Result<String> {
    let item = list()
        .into_iter()
        .find(|i| i.name() == name)
        .with_context(|| format!("no download is named {name}: reims-ctl downloads lists them"))?;
    if let Some(vm) = used_by(&item.path) {
        bail!("{vm} boots from {name}: take it out first (reims-ctl eject {vm}), or delete {vm}");
    }
    if being_downloaded(&item.path) {
        bail!("{name} is being downloaded now: delete it once its download ends or is stopped");
    }
    std::fs::remove_file(&item.path)
        .with_context(|| format!("delete {}", paths::tilde(&item.path)))?;
    Ok(format!("deleted {name}, freeing {}", size(item.bytes)))
}

/// The VM that boots from `path`, its installer, if any.
fn used_by(path: &Path) -> Option<String> {
    let same = |p: &Path| match (p.canonicalize(), path.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => p == path,
    };
    let reg = Registry::load().ok()?;
    reg.vms.iter().find_map(|e| {
        let vm = Vm::load(&e.dir).ok()?;
        let installer = vm.config.q35.as_ref()?.installer.as_deref()?;
        same(&vm.path(installer)).then(|| e.name.clone())
    })
}

/// Whether a download is writing `path`: an IPSW's part, whose download holds its lock file
/// (reims-restore's download.rs), or an installer while reims-util downloads into the
/// installers directory, which it locks.
fn being_downloaded(path: &Path) -> bool {
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    let lock = if let Some(base) = name.strip_suffix(".part") {
        reims_restore::download::lock_path(path.parent().unwrap_or(Path::new(".")), base)
    } else if catalog::name_parts(&path.file_name().unwrap_or_default().to_string_lossy()).is_some()
    {
        // a whole IPSW is written no more
        return false;
    } else {
        path.with_file_name(".lock")
    };
    let Ok(f) = std::fs::OpenOptions::new().read(true).open(&lock) else {
        return false;
    };
    use std::os::fd::AsRawFd as _;
    // SAFETY: flock on a descriptor this function holds; closing it lets go.
    unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) != 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_download_says_what_it_is() {
        assert_eq!(
            what("UniversalMac_27.2.1_26B131_Restore.ipsw"),
            "macOS 27.2.1 (26B131) IPSW"
        );
        assert_eq!(
            what("UniversalMac_27.2.1_26B131_Restore.ipsw.part"),
            "macOS 27.2.1 (26B131) IPSW, partly downloaded"
        );
        assert_eq!(
            what("macOS-Tahoe-online-installer.dmg"),
            "macOS Tahoe online installer"
        );
        assert_eq!(
            what("macOS-Tahoe-26.6.2-25G83-offline-installer.img"),
            "macOS Tahoe 26.6.2 (25G83) offline installer"
        );
        assert_eq!(what("notes.txt"), "a file reims-ctl didn't download");
        assert_eq!(size(25_595_954_706), "25.6 GB");
        assert_eq!(size(950_000_000), "950 MB");
    }

    /// The list holds both directories' files, but no hidden one; delete removes one, but not
    /// one a VM boots from or one being downloaded.
    #[test]
    fn downloads_are_listed_and_deleted() {
        let _env = crate::testenv::Env::new("downloads");
        let [inst, old] = dirs();
        std::fs::create_dir_all(&inst).unwrap();
        std::fs::create_dir_all(&old).unwrap();
        let ipsw = "UniversalMac_27.2.1_26B131_Restore.ipsw";
        let part = "UniversalMac_28.0_27A100_Restore.ipsw.part";
        let offline = "macOS-Tahoe-26.6.2-25G83-offline-installer.img";
        std::fs::write(inst.join(ipsw), vec![1u8; 10_000]).unwrap();
        std::fs::write(inst.join(part), b"p").unwrap();
        std::fs::write(inst.join(offline), b"o").unwrap();
        std::fs::write(inst.join(".lock"), b"").unwrap();
        std::fs::write(old.join("UniversalMac_14.6.1_23G93_Restore.ipsw"), b"x").unwrap();
        let names: Vec<_> = list().iter().map(Item::name).collect();
        assert_eq!(
            names,
            [
                "UniversalMac_14.6.1_23G93_Restore.ipsw",
                ipsw,
                part,
                offline
            ]
        );

        // a VM boots from the offline installer
        let mut c = crate::vm::tests::x86_64_config();
        c.q35.as_mut().unwrap().installer = Some(inst.join(offline));
        let dir = _env.root.join("vms/t");
        std::fs::create_dir_all(&dir).unwrap();
        Vm {
            dir: dir.clone(),
            config: c,
        }
        .save()
        .unwrap();
        let mut reg = Registry::load().unwrap();
        reg.add("t", &dir).unwrap();
        reg.save().unwrap();
        let e = delete(offline).unwrap_err().to_string();
        assert!(e.starts_with("t boots from"), "{e}");

        // a part being downloaded
        let base = part.strip_suffix(".part").unwrap();
        let held = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(reims_restore::download::lock_path(&inst, base))
            .unwrap();
        use std::os::fd::AsRawFd as _;
        // SAFETY: flock on an open descriptor.
        assert_eq!(
            unsafe { libc::flock(held.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );
        let e = delete(part).unwrap_err().to_string();
        assert!(e.contains("is being downloaded now"), "{e}");
        drop(held);
        assert!(
            delete(part)
                .unwrap()
                .starts_with(&format!("deleted {part}, freeing"))
        );

        assert!(
            delete(ipsw)
                .unwrap()
                .starts_with(&format!("deleted {ipsw}"))
        );
        assert!(!inst.join(ipsw).exists());
        let e = delete("nothing.ipsw").unwrap_err().to_string();
        assert!(e.starts_with("no download is named nothing.ipsw"), "{e}");
    }
}
