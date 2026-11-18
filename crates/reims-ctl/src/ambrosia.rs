//! Ambrosia (<https://github.com/steelbrain/ambrosia>), the boot firmware of an arm64 VM on QEMU's
//! vmapple machine: the VM's QEMU runs its image as `-bios`, and Ambrosia boots the guest's macOS
//! with the configuration bundle reims-ctl gives it (prebuilt.rs).
//!
//! reims-ctl runs the image of Ambrosia's release [`VERSION`]: the zip its release page attaches,
//! `ambrosia-<version>.zip`, is downloaded once and checked against [`ZIP_SHA256`], and its
//! `ambrosia-<version>/ambrosia.bin`, checked against [`IMAGE_SHA256`], is kept in the data
//! directory's `ambrosia` ([`image`]). An image there whose bytes are not the pinned ones is
//! downloaded again. `reims-ctl __record-install` downloads it on a host that runs arm64 VMs, so a
//! VM's start needs no network, and a start downloads it where it is missing.
//!
//! A release's zip is the same bytes on any host that builds it (Ambrosia's docs/building.md), so
//! its SHA-256 pins the release: a new Ambrosia is a new [`VERSION`] and both hashes, together.

use crate::paths;
use anyhow::{Context, Result, bail, ensure};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// The release reims-ctl runs.
pub const VERSION: &str = "0.1.3";
/// The SHA-256 of the release's zip, `ambrosia-<VERSION>.zip`.
pub const ZIP_SHA256: &str = "72e5d354e295e86652b7bdd799b389625aa6e3fdbf704fd43cf4d2dd4865ef55";
/// The SHA-256 of its image, `ambrosia.bin`, as the zip's `SHA256SUMS` gives it.
pub const IMAGE_SHA256: &str = "617ac54c841c8f2a7342ae064d58c9b9f306fba89760dabb39cd1b8ce775985b";

/// The most a download of the zip reads: the 0.1.3 zip is 304,700 bytes.
const ZIP_LIMIT: u64 = 16 << 20;

/// Where the release's zip is: its GitHub release page's asset.
pub fn url() -> String {
    format!(
        "https://github.com/steelbrain/ambrosia/releases/download/v{VERSION}/ambrosia-{VERSION}.zip"
    )
}

/// Where the image is kept: `ambrosia-<VERSION>.bin` in the data directory's `ambrosia`.
pub fn path() -> PathBuf {
    paths::data_dir()
        .join("ambrosia")
        .join(format!("ambrosia-{VERSION}.bin"))
}

/// The image, downloaded first where it is missing or not the pinned bytes.
pub fn image() -> Result<PathBuf> {
    let file = path();
    if std::fs::read(&file).is_ok_and(|b| sha256(&b) == IMAGE_SHA256) {
        return Ok(file);
    }
    let zip = download().with_context(|| format!("download Ambrosia {VERSION} from {}", url()))?;
    let image = image_from_zip(&zip)?;
    write(&file, &image)?;
    Ok(file)
}

/// Whether the image is kept already, with the pinned bytes.
pub fn is_kept() -> bool {
    std::fs::read(path()).is_ok_and(|b| sha256(&b) == IMAGE_SHA256)
}

/// The release's zip, checked against [`ZIP_SHA256`]. A request ends where its connection, its
/// answer's head or any read of its body stalls for a minute.
fn download() -> Result<Vec<u8>> {
    let minute = Some(Duration::from_secs(60));
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .https_only(true)
        .timeout_connect(minute)
        .timeout_recv_response(minute)
        .timeout_per_read(minute)
        .build()
        .into();
    let mut resp = agent.get(&url()).call()?;
    let zip = resp
        .body_mut()
        .with_config()
        .limit(ZIP_LIMIT)
        .read_to_vec()?;
    let got = sha256(&zip);
    ensure!(
        got == ZIP_SHA256,
        "the zip's SHA-256 is {got}, not {ZIP_SHA256}, the release's"
    );
    Ok(zip)
}

/// The image in `zip`, `ambrosia-<VERSION>/ambrosia.bin`, checked against [`IMAGE_SHA256`].
fn image_from_zip(zip: &[u8]) -> Result<Vec<u8>> {
    let name = format!("ambrosia-{VERSION}/ambrosia.bin");
    let mut archive =
        zip::ZipArchive::new(std::io::Cursor::new(zip)).context("read Ambrosia's zip")?;
    let mut member = archive
        .by_name(&name)
        .with_context(|| format!("Ambrosia's zip has no {name}"))?;
    let mut image = Vec::new();
    member
        .read_to_end(&mut image)
        .with_context(|| format!("read {name}"))?;
    let got = sha256(&image);
    if got != IMAGE_SHA256 {
        bail!("{name}'s SHA-256 is {got}, not {IMAGE_SHA256}, the release's");
    }
    Ok(image)
}

/// Writes `bytes` to `file` through a file beside it renamed into place, so a reader never sees
/// it half-written.
fn write(file: &Path, bytes: &[u8]) -> Result<()> {
    let dir = file.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let mut tmp = file.as_os_str().to_os_string();
    tmp.push(format!(".{}.tmp", std::process::id()));
    let tmp = PathBuf::from(tmp);
    let written = std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, file));
    if let Err(e) = written {
        let _ = std::fs::remove_file(&tmp);
        return Err(e).with_context(|| format!("write {}", file.display()));
    }
    Ok(())
}

/// `bytes`' SHA-256, in lowercase hex.
fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    /// A zip of `members`, stored, as Ambrosia's release zip stores them.
    fn zip_of(members: &[(&str, &[u8])]) -> Vec<u8> {
        let mut w = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        for (name, bytes) in members {
            w.start_file(*name, opts).unwrap();
            w.write_all(bytes).unwrap();
        }
        w.finish().unwrap().into_inner()
    }

    /// The image is the zip's `ambrosia-<VERSION>/ambrosia.bin`, and one whose bytes are not the
    /// release's is refused, as is a zip without it.
    #[test]
    fn the_image_is_the_releases() {
        let name = format!("ambrosia-{VERSION}/ambrosia.bin");
        let e = image_from_zip(&zip_of(&[(&name, b"not the image")])).unwrap_err();
        assert!(e.to_string().contains(IMAGE_SHA256), "{e:#}");
        let e = image_from_zip(&zip_of(&[("ambrosia.bin", b"x")])).unwrap_err();
        assert!(e.to_string().contains(&name), "{e:#}");
        assert!(image_from_zip(b"not a zip").is_err());
    }

    /// The URL is the release's asset, by its tag and its zip's name.
    #[test]
    fn the_url_names_the_release() {
        assert_eq!(
            url(),
            "https://github.com/steelbrain/ambrosia/releases/download/v0.1.3/ambrosia-0.1.3.zip"
        );
        assert_eq!(
            sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        for h in [ZIP_SHA256, IMAGE_SHA256] {
            assert!(h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit()));
        }
    }

    /// A write leaves the file whole, and nothing beside it.
    #[test]
    fn the_image_is_written_whole() {
        let d = std::env::temp_dir().join(format!("reims-ctl-ambrosia-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let f = d.join("ambrosia").join("a.bin");
        write(&f, b"one").unwrap();
        write(&f, b"two").unwrap();
        assert_eq!(std::fs::read(&f).unwrap(), b"two");
        assert_eq!(std::fs::read_dir(f.parent().unwrap()).unwrap().count(), 1);
        std::fs::remove_dir_all(&d).unwrap();
    }
}
