//! An IPSW's disk images, opened where they lie: a stored member read in place, decrypted if it
//! is an Apple Encrypted Archive, then unpacked if it is a UDIF image.

use std::io::{self, Read};
use std::sync::Arc;

use anyhow::{Context, Result, bail};

use crate::aea::{self, Aea, KeyService};
use crate::ipsw::Ipsw;
use crate::source::{Cursor, ReadAt, SubRange};
use crate::udif::Udif;

/// What an image member is, layer by layer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Layers {
    /// An Apple Encrypted Archive around the rest.
    pub encrypted: bool,
    /// A UDIF image (inside the archive, if any).
    pub udif: bool,
}

/// A source of an image's bytes, any of the layers.
pub type Source = Arc<dyn ReadAt>;

/// Opens member `name` of `ipsw`, a stored image: returns its plaintext (the archive decrypted,
/// with its key from `keys`) and what it was.
pub fn open_plain(ipsw: &Ipsw, name: &str, keys: &dyn KeyService) -> Result<(Source, Layers)> {
    let range = ipsw
        .range(name)?
        .with_context(|| format!("{name} is deflated in the IPSW, so it can't be read in place"))?;
    let mut head = [0u8; 4];
    range
        .read_exact_at(0, &mut head)
        .with_context(|| format!("read {name}"))?;
    if aea::is_aea(&head) {
        let key = aea::archive_key(&range, keys).with_context(|| format!("get {name}'s key"))?;
        let plain = Aea::open(range, &key).with_context(|| format!("decrypt {name}"))?;
        Ok((
            Arc::new(plain),
            Layers {
                encrypted: true,
                udif: false,
            },
        ))
    } else {
        Ok((Arc::new(range), Layers::default()))
    }
}

/// Member `name`'s plaintext as a stream, front to back, and its length: an Apple Encrypted
/// Archive decrypted (stored, as every IPSW keeps its archives), a deflated member inflated (as
/// 14's IPSWs keep their BaseSystem and Cryptex1 images, and every IPSW its Cryptex1 AppOS and
/// 27's its RosettaOS), any other member as stored. A member read from the zip has its CRC-32
/// checked at its end; an archive's segments are checked as they are decrypted.
pub fn plain_stream(
    ipsw: &Ipsw,
    name: &str,
    keys: &dyn KeyService,
) -> Result<(u64, Box<dyn Read + Send>)> {
    let size = ipsw.member(name)?.size;
    let mut r = ipsw.reader(name)?;
    let mut head = Vec::new();
    (&mut r)
        .take(4)
        .read_to_end(&mut head)
        .with_context(|| format!("read {name}"))?;
    if !aea::is_aea(&head) {
        return Ok((size, Box::new(io::Cursor::new(head).chain(r))));
    }
    drop(r);
    if ipsw.range(name)?.is_none() {
        bail!("{name} is an encrypted archive deflated in the IPSW, which no IPSW has been");
    }
    let (plain, _) = open_plain(ipsw, name, keys)?;
    Ok((plain.len(), Box::new(Cursor::new(plain))))
}

/// Opens member `name` as a disk: its plaintext, unpacked if it is a UDIF image.
pub fn open_disk(ipsw: &Ipsw, name: &str, keys: &dyn KeyService) -> Result<(Source, Layers)> {
    let (plain, mut layers) = open_plain(ipsw, name, keys)?;
    if is_udif(&*plain)? {
        let img = Udif::open(plain).with_context(|| format!("open {name} as a UDIF image"))?;
        layers.udif = true;
        Ok((Arc::new(img), layers))
    } else {
        Ok((plain, layers))
    }
}

/// Opens member `name`'s APFS container: the `Apple_APFS` partition of its disk, or the whole
/// disk where it is the container itself.
pub fn open_apfs(ipsw: &Ipsw, name: &str, keys: &dyn KeyService) -> Result<(Source, Layers)> {
    let (plain, mut layers) = open_plain(ipsw, name, keys)?;
    if is_udif(&*plain)? {
        let img = Udif::open(plain).with_context(|| format!("open {name} as a UDIF image"))?;
        layers.udif = true;
        let p = img
            .partition("Apple_APFS")
            .with_context(|| format!("find {name}'s APFS container"))?
            .clone();
        let img: Source = Arc::new(img);
        let part = SubRange::new(img, p.offset, p.len)?;
        check_nxsb(&part, name)?;
        Ok((Arc::new(part), layers))
    } else {
        check_nxsb(&*plain, name)?;
        Ok((plain, layers))
    }
}

/// An APFS reader's view of one of our sources.
struct Adapt(Source);

impl reims_apfs::dev::Source for Adapt {
    fn read_at(&self, off: u64, buf: &mut [u8]) -> std::io::Result<()> {
        self.0.read_exact_at(off, buf)
    }

    fn len(&self) -> u64 {
        self.0.len()
    }
}

/// Opens member `name`'s APFS container for reading files out of it.
pub fn open_container(
    ipsw: &Ipsw,
    name: &str,
    keys: &dyn KeyService,
) -> Result<reims_apfs::Container> {
    let (src, _) = open_apfs(ipsw, name, keys)?;
    reims_apfs::open(Arc::new(Adapt(src))).with_context(|| format!("open {name}'s APFS container"))
}

/// Opens member `name`'s APFS container for reading files and records out of it, with the
/// source it reads: a block `b` of the container is at `b * block_size` of the source, so a
/// copy can read a stream's extents straight from it.
pub fn open_container_source(
    ipsw: &Ipsw,
    name: &str,
    keys: &dyn KeyService,
) -> Result<(Source, reims_apfs::Container)> {
    let (src, _) = open_apfs(ipsw, name, keys)?;
    let c = reims_apfs::open(Arc::new(Adapt(src.clone())))
        .with_context(|| format!("open {name}'s APFS container"))?;
    Ok((src, c))
}

/// Whether `src` ends in a UDIF trailer.
pub fn is_udif(src: &dyn ReadAt) -> Result<bool> {
    let len = src.len();
    if len < 512 {
        return Ok(false);
    }
    let mut koly = [0u8; 4];
    src.read_exact_at(len - 512, &mut koly)
        .context("read an image's last sector")?;
    Ok(&koly == b"koly")
}

/// Fails unless `src` starts with an APFS container superblock.
fn check_nxsb(src: &dyn ReadAt, name: &str) -> Result<()> {
    let mut b = [0u8; 36];
    src.read_exact_at(0, &mut b)
        .with_context(|| format!("read {name}'s container"))?;
    if &b[32..36] != b"NXSB" {
        bail!("{name} holds no APFS container");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aea::AppleKeyService;
    use crate::ipsw::tests::zip_with;

    fn read_all(ipsw: &Ipsw, name: &str) -> Result<(u64, Vec<u8>)> {
        let (len, mut r) = plain_stream(ipsw, name, &AppleKeyService::default())?;
        let mut out = Vec::new();
        r.read_to_end(&mut out)?;
        Ok((len, out))
    }

    #[test]
    fn a_stored_or_deflated_image_streams_as_it_is() {
        let big: Vec<u8> = (0..300_000u32).flat_map(|i| i.to_le_bytes()).collect();
        let p = zip_with(&[
            ("stored.dmg", &big, false),
            ("deflated.dmg", &big, true),
            ("tiny", b"ab", true),
        ]);
        let ipsw = Ipsw::open(&p.0).unwrap();
        for name in ["stored.dmg", "deflated.dmg"] {
            let (len, got) = read_all(&ipsw, name).unwrap();
            assert_eq!(len, big.len() as u64);
            assert!(got == big, "{name}");
        }
        assert_eq!(read_all(&ipsw, "tiny").unwrap(), (2, b"ab".to_vec()));
    }

    #[test]
    fn a_deflated_encrypted_archive_is_refused() {
        let p = zip_with(&[("x.dmg.aea", b"AEA1 and the rest", true)]);
        let ipsw = Ipsw::open(&p.0).unwrap();
        let e = read_all(&ipsw, "x.dmg.aea").unwrap_err();
        assert!(format!("{e:#}").contains("deflated"), "{e:#}");
    }
}
