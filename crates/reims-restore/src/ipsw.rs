//! An IPSW, read where it lies: its zip directory, its small members whole, its large ones as
//! streams, and stored ones at random offsets, with no extraction to disk.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};

use crate::source::{Cursor, ReadAt, SubRange};

/// How a member is stored in the zip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// Stored as it is: its bytes can be read at any offset.
    Stored,
    /// Deflated: its bytes can only be streamed.
    Deflated,
}

/// One member of the IPSW's zip.
#[derive(Clone, Debug)]
pub struct Member {
    /// Its name in the zip.
    pub name: String,
    /// Where its data starts in the file.
    pub data_start: u64,
    /// Its size in the zip.
    pub compressed_size: u64,
    /// Its size once read.
    pub size: u64,
    /// How it is stored.
    pub method: Method,
    /// The CRC-32 of its contents.
    pub crc32: u32,
    /// Its Unix mode (type and permission bits), where the zip records one.
    pub mode: Option<u32>,
    /// Its modification time, Unix seconds: the zip's extended timestamp where it has one, else
    /// its DOS time read as UTC.
    pub mtime: u32,
    /// Whether it is a directory.
    pub dir: bool,
}

/// A stored member's bytes, read at offsets.
pub type Range = SubRange<Arc<dyn ReadAt>>;

/// An open IPSW.
pub struct Ipsw {
    path: PathBuf,
    src: Arc<dyn ReadAt>,
    members: HashMap<String, Member>,
    /// Its directories' entries, by name (ending with `/`).
    dirs: HashMap<String, Member>,
}

impl Ipsw {
    /// Opens `path` and reads its zip directory.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        Self::from_source(Arc::new(crate::source::FileRange::whole(file)?), path)
    }

    /// Reads the zip directory of an IPSW whose bytes `src` gives, such as a ranged download.
    /// `path` names it in messages.
    pub fn from_source(src: Arc<dyn ReadAt>, path: &Path) -> Result<Self> {
        let mut zip =
            zip::ZipArchive::new(BufReader::new(Cursor::new(src.clone()))).with_context(|| {
                format!(
                    "{} is not a zip archive: is it a whole IPSW?",
                    path.display()
                )
            })?;
        let mut members = HashMap::new();
        let mut dirs = HashMap::new();
        for i in 0..zip.len() {
            let f = zip
                .by_index_raw(i)
                .with_context(|| format!("read member {i} of {}", path.display()))?;
            let name = f.name().context("a member's name")?.into_owned();
            let mtime = mtime(&f);
            if f.is_dir() {
                dirs.insert(
                    name.clone(),
                    Member {
                        name,
                        data_start: 0,
                        compressed_size: 0,
                        size: 0,
                        method: Method::Stored,
                        crc32: 0,
                        mode: f.unix_mode(),
                        mtime,
                        dir: true,
                    },
                );
                continue;
            }
            let method = match f.compression() {
                zip::CompressionMethod::Stored => Method::Stored,
                zip::CompressionMethod::Deflated => Method::Deflated,
                other => bail!("{name} is compressed with {other:?}, which an IPSW doesn't use"),
            };
            if f.encrypted() {
                bail!("{name} is encrypted in the zip, which an IPSW never is");
            }
            let data_start = f
                .data_start()
                .with_context(|| format!("{name}: no data offset"))?;
            members.insert(
                name.clone(),
                Member {
                    name,
                    data_start,
                    compressed_size: f.compressed_size(),
                    size: f.size(),
                    method,
                    crc32: f.crc32(),
                    mode: f.unix_mode(),
                    mtime,
                    dir: false,
                },
            );
        }
        let len = src.len();
        for m in members.values() {
            if m.data_start
                .checked_add(m.compressed_size)
                .is_none_or(|end| end > len)
            {
                bail!(
                    "{} ends past the end of {} ({len} bytes): the IPSW is truncated",
                    m.name,
                    path.display()
                );
            }
        }
        Ok(Self {
            path: path.to_owned(),
            src,
            members,
            dirs,
        })
    }

    /// The IPSW's path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The member called `name`.
    pub fn member(&self, name: &str) -> Result<&Member> {
        self.members
            .get(name)
            .with_context(|| format!("the IPSW has no {name}"))
    }

    /// Whether the IPSW has a member called `name`.
    pub fn has(&self, name: &str) -> bool {
        self.members.contains_key(name)
    }

    /// The members whose names start with `prefix`, directories too, by name.
    pub fn entries(&self, prefix: &str) -> Vec<&Member> {
        let mut v: Vec<&Member> = self
            .members
            .values()
            .chain(self.dirs.values())
            .filter(|m| m.name.starts_with(prefix))
            .collect();
        v.sort_by(|a, b| a.name.cmp(&b.name));
        v
    }

    /// Every member's name.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.members.keys().map(String::as_str)
    }

    /// Reads a member whole into memory, checking its CRC-32: for manifests, property lists and
    /// the components iBoot takes whole (the restore ramdisk is 373 MB on 27.2.1). Images are
    /// read with [`Ipsw::reader`] or [`Ipsw::range`] instead.
    pub fn read(&self, name: &str) -> Result<Vec<u8>> {
        let m = self.member(name)?;
        let size = usize::try_from(m.size)
            .with_context(|| format!("{name} is {} bytes, too large for memory", m.size))?;
        let mut out = Vec::new();
        out.try_reserve_exact(size)
            .with_context(|| format!("no memory for {name}'s {size} bytes"))?;
        self.reader(name)?
            .read_to_end(&mut out)
            .with_context(|| format!("read {name} from the IPSW"))?;
        Ok(out)
    }

    /// A stream of a member's contents, which fails at its end, and on every read after, if its
    /// CRC-32 or size is wrong.
    pub fn reader(&self, name: &str) -> Result<Box<dyn Read + Send>> {
        let m = self.member(name)?.clone();
        let raw = Cursor::new(SubRange::new(
            self.src.clone(),
            m.data_start,
            m.compressed_size,
        )?);
        let inner: Box<dyn Read + Send> = match m.method {
            Method::Stored => Box::new(raw),
            Method::Deflated => Box::new(flate2::read::DeflateDecoder::new(
                BufReader::with_capacity(1 << 20, raw),
            )),
        };
        Ok(Box::new(Checked {
            inner,
            name: m.name,
            want_size: m.size,
            want_crc: m.crc32,
            size: 0,
            crc: crc32fast::Hasher::new(),
            end: None,
        }))
    }

    /// A stored member's bytes at random offsets, or `None` if it is deflated.
    pub fn range(&self, name: &str) -> Result<Option<Range>> {
        let m = self.member(name)?;
        if m.method != Method::Stored {
            return Ok(None);
        }
        Ok(Some(SubRange::new(self.src.clone(), m.data_start, m.size)?))
    }

    /// A stored member read in place, as [`Ipsw::range`], whose CRC-32 is checked as it is read
    /// front to back: see [`CheckedRange`]. `None` if the member is compressed.
    pub fn checked_range(&self, name: &str) -> Result<Option<CheckedRange>> {
        let m = self.member(name)?;
        Ok(self.range(name)?.map(|range| CheckedRange {
            range,
            name: m.name.clone(),
            want_crc: m.crc32,
            front: Mutex::new(Front {
                next: 0,
                crc: crc32fast::Hasher::new(),
                damaged: None,
            }),
        }))
    }
}

/// A stored member read in place whose CRC-32 is checked as it is read front to back, as ASR
/// streams the system image: the read that carries the front to the member's end fails if the
/// member's CRC-32 is wrong, and so does every read after it. A read behind the front, or past a
/// gap ahead of it, is served unchecked and leaves the front where it is.
pub struct CheckedRange {
    range: Range,
    name: String,
    want_crc: u32,
    front: Mutex<Front>,
}

/// How far a [`CheckedRange`] has been read front to back, and the CRC-32 of that much.
struct Front {
    next: u64,
    crc: crc32fast::Hasher,
    damaged: Option<String>,
}

impl ReadAt for CheckedRange {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.range.read_at(offset, buf)?;
        let mut f = self.front.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(msg) = &f.damaged {
            return Err(io::Error::new(io::ErrorKind::InvalidData, msg.clone()));
        }
        let end = offset + n as u64;
        if offset <= f.next && end > f.next {
            let from = (f.next - offset) as usize;
            f.crc.update(&buf[from..n]);
            f.next = end;
            if f.next == self.range.len() {
                let crc = std::mem::take(&mut f.crc).finalize();
                if crc != self.want_crc {
                    let msg = format!(
                        "{}'s CRC-32 is {crc:08x}, not {:08x}: the IPSW is damaged",
                        self.name, self.want_crc
                    );
                    f.damaged = Some(msg.clone());
                    return Err(io::Error::new(io::ErrorKind::InvalidData, msg));
                }
            }
        }
        Ok(n)
    }

    fn len(&self) -> u64 {
        self.range.len()
    }
}

/// A zip entry's modification time, Unix seconds.
fn mtime<R: std::io::Read>(f: &zip::read::ZipFile<'_, R>) -> u32 {
    for e in f.extra_data_fields() {
        if let zip::ExtraField::ExtendedTimestamp(t) = e
            && let Some(t) = t.mod_time()
        {
            return t;
        }
    }
    f.last_modified().map_or(0, |d| {
        unix_seconds(
            d.year().into(),
            d.month().into(),
            d.day().into(),
            d.hour().into(),
            d.minute().into(),
            d.second().into(),
        )
    })
}

/// A UTC civil time as Unix seconds (0 before 1970, clamped to u32).
fn unix_seconds(y: i64, mo: i64, d: i64, h: i64, mi: i64, s: i64) -> u32 {
    // days from 1970-01-01, by the proleptic Gregorian calendar
    let y = if mo <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (mo + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    (days * 86400 + h * 3600 + mi * 60 + s).clamp(0, u32::MAX.into()) as u32
}

/// A member's stream that checks its size and CRC-32 at its end.
struct Checked {
    inner: Box<dyn Read + Send>,
    name: String,
    want_size: u64,
    want_crc: u32,
    size: u64,
    crc: crc32fast::Hasher,
    /// The end's verdict once a read has reached it, which every later read returns:
    /// `Read` lets a caller read again past the end.
    end: Option<Result<(), (io::ErrorKind, String)>>,
}

impl Checked {
    /// The verdict on a stream that has ended at `self.size` bytes.
    fn verdict(&mut self) -> Result<(), (io::ErrorKind, String)> {
        if self.size != self.want_size {
            return Err((
                io::ErrorKind::UnexpectedEof,
                format!(
                    "{} ends after {} of its {} bytes: the IPSW is damaged",
                    self.name, self.size, self.want_size
                ),
            ));
        }
        let crc = std::mem::take(&mut self.crc).finalize();
        if crc != self.want_crc {
            return Err((
                io::ErrorKind::InvalidData,
                format!(
                    "{}'s CRC-32 is {crc:08x}, not {:08x}: the IPSW is damaged",
                    self.name, self.want_crc
                ),
            ));
        }
        Ok(())
    }
}

impl Read for Checked {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if let Some(end) = &self.end {
            return match end {
                Ok(()) => Ok(0),
                Err((kind, msg)) => Err(io::Error::new(*kind, msg.clone())),
            };
        }
        let n = self.inner.read(buf)?;
        self.size += n as u64;
        if self.size > self.want_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{} holds more than its {} bytes: the IPSW is damaged",
                    self.name, self.want_size
                ),
            ));
        }
        self.crc.update(&buf[..n]);
        if n == 0 && !buf.is_empty() {
            let end = self.verdict();
            self.end = Some(end);
            return self.read(buf);
        }
        Ok(n)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Write;

    /// A zip of `members`: name, contents, and whether deflated.
    pub(crate) fn zip_with(members: &[(&str, &[u8], bool)]) -> tempfile_path::TempPath {
        let p = tempfile_path::TempPath::new("zip");
        let f = File::create(&p.0).unwrap();
        let mut w = zip::ZipWriter::new(f);
        for (name, data, deflate) in members {
            let opts = zip::write::SimpleFileOptions::default().compression_method(if *deflate {
                zip::CompressionMethod::Deflated
            } else {
                zip::CompressionMethod::Stored
            });
            w.start_file(*name, opts).unwrap();
            w.write_all(data).unwrap();
        }
        w.finish().unwrap();
        p
    }

    pub(crate) mod tempfile_path {
        use std::path::PathBuf;
        /// A path in the system's temporary directory, removed when dropped.
        pub struct TempPath(pub PathBuf);
        impl TempPath {
            pub fn new(ext: &str) -> Self {
                let mut b = [0u8; 8];
                getrandom::fill(&mut b).unwrap();
                Self(std::env::temp_dir().join(format!(
                    "reims-restore-test-{}.{ext}",
                    u64::from_le_bytes(b)
                )))
            }
        }
        impl Drop for TempPath {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
    }

    #[test]
    fn stored_and_deflated_members_read_back() {
        let big: Vec<u8> = (0..100_000u32).flat_map(|i| i.to_le_bytes()).collect();
        let p = zip_with(&[("a.txt", b"hello", false), ("dir/b.bin", &big, true)]);
        let ipsw = Ipsw::open(&p.0).unwrap();
        assert_eq!(ipsw.read("a.txt").unwrap(), b"hello");
        assert_eq!(ipsw.read("dir/b.bin").unwrap(), big);
        assert!(ipsw.range("dir/b.bin").unwrap().is_none());
        let r = ipsw.range("a.txt").unwrap().unwrap();
        let mut buf = [0u8; 3];
        crate::source::ReadAt::read_exact_at(&r, 2, &mut buf).unwrap();
        assert_eq!(&buf, b"llo");
        assert!(ipsw.read("missing").is_err());
    }

    #[test]
    fn a_members_unix_mode_is_kept() {
        let p = tempfile_path::TempPath::new("zip");
        let mut w = zip::ZipWriter::new(File::create(&p.0).unwrap());
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored)
            .unix_permissions(0o755);
        w.start_file("tool", opts).unwrap();
        w.write_all(b"#!").unwrap();
        w.finish().unwrap();
        let ipsw = Ipsw::open(&p.0).unwrap();
        assert_eq!(
            ipsw.member("tool").unwrap().mode.map(|m| m & 0o777),
            Some(0o755)
        );
    }

    #[test]
    fn entries_list_directories_and_times() {
        let p = tempfile_path::TempPath::new("zip");
        let mut w = zip::ZipWriter::new(File::create(&p.0).unwrap());
        let t = zip::DateTime::from_date_and_time(2024, 7, 29, 12, 30, 10).unwrap();
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored)
            .last_modified_time(t);
        w.add_directory("Firmware/", opts.unix_permissions(0o755))
            .unwrap();
        w.start_file("Firmware/a.im4p", opts).unwrap();
        w.write_all(b"abc").unwrap();
        w.start_file("other", opts).unwrap();
        w.finish().unwrap();
        let ipsw = Ipsw::open(&p.0).unwrap();
        let e = ipsw.entries("Firmware/");
        let names: Vec<_> = e.iter().map(|m| (m.name.as_str(), m.dir, m.size)).collect();
        assert_eq!(
            names,
            [("Firmware/", true, 0), ("Firmware/a.im4p", false, 3)]
        );
        // 2024-07-29 12:30:10 UTC
        assert_eq!(e[1].mtime, 1_722_256_210);
        assert!(!ipsw.has("Firmware/"));
    }

    #[test]
    fn civil_times_become_unix_seconds() {
        assert_eq!(unix_seconds(1970, 1, 1, 0, 0, 0), 0);
        assert_eq!(unix_seconds(2000, 3, 1, 0, 0, 0), 951_868_800);
        assert_eq!(unix_seconds(1960, 1, 1, 0, 0, 0), 0);
    }

    #[test]
    fn a_damaged_member_fails_its_crc() {
        let p = zip_with(&[("a.txt", b"hello world", false)]);
        let ipsw = Ipsw::open(&p.0).unwrap();
        let start = ipsw.member("a.txt").unwrap().data_start;
        drop(ipsw);
        let mut bytes = std::fs::read(&p.0).unwrap();
        bytes[start as usize] ^= 1;
        std::fs::write(&p.0, &bytes).unwrap();
        let ipsw = Ipsw::open(&p.0).unwrap();
        let err = ipsw.read("a.txt").unwrap_err();
        assert!(format!("{err:#}").contains("CRC-32"), "{err:#}");
    }

    #[test]
    fn reads_past_the_end_repeat_its_verdict() {
        let p = zip_with(&[("a.txt", b"hello", false), ("b.txt", b"world", true)]);
        let ipsw = Ipsw::open(&p.0).unwrap();
        for name in ["a.txt", "b.txt"] {
            let mut s = ipsw.reader(name).unwrap();
            let mut all = Vec::new();
            s.read_to_end(&mut all).unwrap();
            assert_eq!(all.len(), 5);
            let mut buf = [0u8; 16];
            assert_eq!(s.read(&mut buf).unwrap(), 0);
            assert_eq!(s.read(&mut buf).unwrap(), 0);
        }
        let start = ipsw.member("a.txt").unwrap().data_start;
        drop(ipsw);
        let mut bytes = std::fs::read(&p.0).unwrap();
        bytes[start as usize] ^= 1;
        std::fs::write(&p.0, &bytes).unwrap();
        let ipsw = Ipsw::open(&p.0).unwrap();
        let mut s = ipsw.reader("a.txt").unwrap();
        let mut all = Vec::new();
        let first = s.read_to_end(&mut all).unwrap_err().to_string();
        let mut buf = [0u8; 16];
        let again = s.read(&mut buf).unwrap_err().to_string();
        assert_eq!(first, again);
        assert!(!first.contains("00000000"), "{first}");
    }

    #[test]
    fn a_checked_range_fails_where_its_front_reaches_a_damaged_end() {
        let data: Vec<u8> = (0..70_000u32).map(|i| (i * 13 % 251) as u8).collect();
        let p = zip_with(&[("img", &data, false)]);
        let ipsw = Ipsw::open(&p.0).unwrap();
        // Sound: read front to back in uneven pieces, with a look behind and one far ahead.
        let r = ipsw.checked_range("img").unwrap().unwrap();
        let mut buf = vec![0u8; 9_000];
        r.read_exact_at(60_000, &mut buf[..100]).unwrap();
        let mut off = 0u64;
        while off < r.len() {
            let n = (r.len() - off).min(9_000) as usize;
            r.read_exact_at(off, &mut buf[..n]).unwrap();
            r.read_exact_at(off / 2, &mut buf[..10]).unwrap();
            off += n as u64;
        }
        r.read_exact_at(0, &mut buf[..10]).unwrap();
        // Damaged: one bit flipped in the middle.
        let start = ipsw.member("img").unwrap().data_start;
        drop((r, ipsw));
        let mut bytes = std::fs::read(&p.0).unwrap();
        bytes[start as usize + 35_000] ^= 2;
        std::fs::write(&p.0, &bytes).unwrap();
        let ipsw = Ipsw::open(&p.0).unwrap();
        let r = ipsw.checked_range("img").unwrap().unwrap();
        let mut off = 0u64;
        let err = loop {
            let n = (r.len() - off).min(9_000) as usize;
            if let Err(e) = r.read_exact_at(off, &mut buf[..n]) {
                break e;
            }
            off += n as u64;
        };
        assert!(off + 9_000 >= r.len(), "failed before the end, at {off}");
        assert!(err.to_string().contains("CRC-32"), "{err}");
        assert!(r.read_exact_at(0, &mut buf[..10]).is_err());
    }

    #[test]
    fn a_truncated_ipsw_is_refused() {
        let p = zip_with(&[("a.txt", &[7u8; 5000], false)]);
        let bytes = std::fs::read(&p.0).unwrap();
        // Keep the directory but drop the member's data: the directory then points past the
        // end once the tail is moved up.
        let q = tempfile_path::TempPath::new("zip");
        std::fs::write(&q.0, &bytes[..bytes.len() / 2]).unwrap();
        assert!(Ipsw::open(&q.0).is_err());
    }
}
