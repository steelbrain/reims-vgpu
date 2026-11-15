//! The IPSWs arm64 VMs are restored from (restore.rs). A create names its IPSW by its path, or
//! by `latest`, the current macOS Apple's catalog offers (reims-restore's catalog.rs), which is
//! downloaded once into the installers directory (reims-restore's download.rs), where reims-util
//! keeps its installers too (reims_util.rs), and every later VM of that release finds it. A
//! download that stops is continued by the next. Apple offers only its current release: another
//! is given by its IPSW's path. One downloaded before is also found by its version or build,
//! there or in the `ipsw` directory the previous reims-ctl downloaded IPSWs into.

use crate::paths;
use anyhow::{Context, Result};
use reims_restore::catalog::{self, Entry, OnDisk};
use reims_restore::http::Client;
use reims_restore::pipeline::Event;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

/// Where IPSWs are downloaded: the installers directory.
pub fn dir() -> PathBuf {
    paths::installers_dir()
}

/// Where the previous reims-ctl downloaded IPSWs, read where they are.
pub fn legacy_dir() -> PathBuf {
    paths::data_dir().join("ipsw")
}

/// The IPSWs downloaded, newest first.
pub fn on_disk() -> Vec<OnDisk> {
    catalog::on_disk(&[dir(), legacy_dir()])
}

/// Whether a create's `--ipsw` names a file rather than a release: a path, or a name that is
/// there.
pub fn is_path(s: &str) -> bool {
    s.contains('/') || s.ends_with(".ipsw") || Path::new(s).exists()
}

/// The IPSW a create named.
#[derive(Clone, Debug, PartialEq)]
pub enum Found {
    /// Here: given by its path, or downloaded before.
    Here(PathBuf),
    /// To download: its file's length, and how much of it a stopped download left.
    Download { entry: Entry, total: u64, have: u64 },
}

/// Where a create's IPSW comes from: Apple's catalog and CDN ([`Apple`]), or a test's.
pub trait Source {
    /// Finds the IPSW `want` names: a path, `latest`, or a release downloaded before.
    fn find(&self, want: &str) -> Result<Found>;
    /// Downloads `e`'s IPSW into [`dir`], or continues its download; the file.
    fn download(
        &self,
        e: &Entry,
        progress: &mut dyn FnMut(&str),
        cancel: &AtomicBool,
    ) -> Result<PathBuf>;
}

/// Apple's catalog, and its CDN.
pub struct Apple;

impl Source for Apple {
    fn find(&self, want: &str) -> Result<Found> {
        if is_path(want) {
            return here(want);
        }
        let client = Client::apple();
        let (listed, unread) = match catalog::fetch(&client, catalog::URL) {
            Ok(l) => (l, None),
            Err(e) => (Vec::new(), Some(e)),
        };
        let found = catalog::pick(want, &on_disk(), &listed).map_err(|e| match unread {
            Some(why) => e.context(format!("{why:#}")),
            None => e,
        })?;
        Ok(match found {
            catalog::Found::OnDisk(o) => Found::Here(o.path),
            catalog::Found::Download(entry) => {
                let (total, have) = reims_restore::download::remaining(&client, &entry, &dir())?;
                Found::Download { entry, total, have }
            }
        })
    }

    fn download(
        &self,
        e: &Entry,
        progress: &mut dyn FnMut(&str),
        cancel: &AtomicBool,
    ) -> Result<PathBuf> {
        let mut shown = crate::restore::Shown::default();
        reims_restore::download::download(
            &Client::apple(),
            e,
            &dir(),
            &mut |ev| match ev {
                Event::Step(s) | Event::Note(s) => progress(&s),
                Event::Progress { what, done, total } => {
                    if let Some(line) = shown.line(&what, done, total) {
                        progress(&line);
                    }
                }
            },
            cancel,
        )
    }
}

/// The IPSW at `path`, which must be a file.
pub fn here(path: &str) -> Result<Found> {
    let p = PathBuf::from(path);
    let m = std::fs::metadata(&p).with_context(|| format!("the IPSW {path} does not exist"))?;
    anyhow::ensure!(m.is_file(), "the IPSW {path} is not a file");
    Ok(Found::Here(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_ipsw_is_named_by_its_path_or_a_release() {
        assert!(is_path("~/Downloads/x.ipsw"));
        assert!(is_path("./a"));
        assert!(is_path("UniversalMac_27.2.1_26B131_Restore.ipsw"));
        assert!(!is_path("latest"));
        assert!(!is_path("27.2.1"));
        assert!(!is_path("26B131"));
    }

    /// IPSWs downloaded go in the installers directory, and the previous reims-ctl's are found
    /// where it kept them.
    #[test]
    fn ipsws_are_found_in_both_directories() {
        let env = crate::testenv::Env::new("ipsw-dirs");
        assert_eq!(dir(), paths::installers_dir());
        std::fs::create_dir_all(dir()).unwrap();
        std::fs::create_dir_all(legacy_dir()).unwrap();
        std::fs::write(dir().join("UniversalMac_27.2.1_26B131_Restore.ipsw"), b"a").unwrap();
        std::fs::write(
            legacy_dir().join("UniversalMac_14.6.1_23G93_Restore.ipsw"),
            b"b",
        )
        .unwrap();
        let found: Vec<_> = on_disk().into_iter().map(|o| o.build).collect();
        assert_eq!(found, ["26B131", "23G93"]);
        assert!(env.root.exists());
        // a release downloaded before is found with no network
        assert_eq!(
            catalog::pick("14.6.1", &on_disk(), &[]).unwrap(),
            catalog::Found::OnDisk(on_disk()[1].clone())
        );
    }
}
