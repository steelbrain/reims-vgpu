//! What the integration tests share: containers ferrosys formats, with a System volume of
//! directories, links, a large file, every compressed layout and many small files, and a sparse
//! disk in memory to write them to.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Arc;

use ferrosys::apfs::{FormatOptions, VolumeGroup, VolumeName, VolumeRequest, format_to};
use ferrosys::{CompressedFile, Darwin, EntryKind, Metadata, SourceEntry, Timestamp};
use reims_apfs::dev::Source;

pub const FRAMEWORK: &str = "/System/Library/Frameworks/Virtualization.framework";
pub const BOOTER: &str = "/System/Library/Frameworks/Virtualization.framework/Versions/A/Resources/AVPBooter.vmapple2.bin";
pub const MANY: usize = 5000;

pub fn time() -> Timestamp {
    Timestamp::from_secs(1_700_000_000)
}

/// `n` bytes no compressor shrinks.
pub fn noise(n: usize, seed: u32) -> Vec<u8> {
    let mut x = seed.wrapping_mul(2_654_435_761) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x as u8
        })
        .collect()
}

pub fn header(kind: u32, size: usize, tail: &[u8]) -> Vec<u8> {
    let mut h = b"fpmc".to_vec();
    h.extend_from_slice(&kind.to_le_bytes());
    h.extend_from_slice(&(size as u64).to_le_bytes());
    h.extend_from_slice(tail);
    h
}

pub fn zlib(b: &[u8]) -> Vec<u8> {
    let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    z.write_all(b).unwrap();
    z.finish().unwrap()
}

/// An LZFSE stream of one uncompressed block.
pub fn lzfse_raw(b: &[u8]) -> Vec<u8> {
    let mut s = b"bvx-".to_vec();
    s.extend_from_slice(&(b.len() as u32).to_le_bytes());
    s.extend_from_slice(b);
    s.extend_from_slice(b"bvx$");
    s
}

/// A resource fork of chunks behind a table of 32-bit offsets (types 8, 10 and 12).
pub fn offset_fork(chunks: &[Vec<u8>]) -> Vec<u8> {
    let mut at = 4 * (chunks.len() + 1);
    let mut fork = Vec::new();
    for c in chunks.iter().map(Vec::len).chain([0]) {
        fork.extend_from_slice(&(at as u32).to_le_bytes());
        at += c;
    }
    for c in chunks {
        fork.extend_from_slice(c);
    }
    fork
}

/// A classic resource fork holding one resource of chunks (type 4): the data's offset
/// (big-endian) at 0, the resource's length there, then its chunk count and each chunk's
/// offset and length from the resource's start (little-endian).
pub fn classic_fork(chunks: &[Vec<u8>]) -> Vec<u8> {
    let mut res = (chunks.len() as u32).to_le_bytes().to_vec();
    let mut at = 4 + 8 * chunks.len();
    for c in chunks {
        res.extend_from_slice(&(at as u32).to_le_bytes());
        res.extend_from_slice(&(c.len() as u32).to_le_bytes());
        at += c.len();
    }
    for c in chunks {
        res.extend_from_slice(c);
    }
    let mut fork = 0x100u32.to_be_bytes().to_vec();
    fork.resize(0x100, 0);
    fork.extend_from_slice(&(res.len() as u32).to_be_bytes());
    fork.extend_from_slice(&res);
    fork
}

pub fn chunks_of(content: &[u8], f: impl Fn(&[u8]) -> Vec<u8>) -> Vec<Vec<u8>> {
    content.chunks(64 * 1024).map(f).collect()
}

/// The compressed files: their paths, content and stored forms.
pub fn compressed() -> Vec<(&'static str, Vec<u8>, CompressedFile)> {
    let text = b"the restore reads this file through its compression header ".repeat(4000);
    let short = b"hello zlib, a header held in a stream ".repeat(12);
    let lzvn = b"hello".to_vec();
    let mut lzvn_stream = vec![0xe0 | 5];
    lzvn_stream.extend_from_slice(&lzvn);
    lzvn_stream.extend_from_slice(&[0x06, 0, 0, 0, 0, 0, 0, 0]);
    let fork_noise = noise(150_000, 7);
    vec![
        (
            "/c/zlib-inline",
            short.clone(),
            CompressedFile::new(header(3, short.len(), &zlib(&short)), None),
        ),
        (
            "/c/lzvn-inline",
            lzvn.clone(),
            CompressedFile::new(header(7, lzvn.len(), &lzvn_stream), None),
        ),
        (
            "/c/stored-inline",
            b"abc".to_vec(),
            CompressedFile::new(header(9, 3, &[0xcc, b'a', b'b', b'c']), None),
        ),
        (
            "/c/lzfse-fork",
            fork_noise.clone(),
            CompressedFile::new(
                header(12, fork_noise.len(), &[]),
                Some(offset_fork(&chunks_of(&fork_noise, lzfse_raw))),
            ),
        ),
        (
            "/c/zlib-fork",
            text.clone(),
            CompressedFile::new(
                header(4, text.len(), &[]),
                Some(classic_fork(&chunks_of(&text, zlib))),
            ),
        ),
        (
            "/c/stored-fork",
            fork_noise.clone(),
            CompressedFile::new(
                header(10, fork_noise.len(), &[]),
                Some(offset_fork(&chunks_of(&fork_noise, |c| {
                    [&[0xcc][..], c].concat()
                }))),
            ),
        ),
        (
            "/c/lzvn-raw-fork",
            fork_noise.clone(),
            CompressedFile::new(
                header(8, fork_noise.len(), &[]),
                Some(offset_fork(&chunks_of(&fork_noise, |c| {
                    [&[0x06][..], c].concat()
                }))),
            ),
        ),
    ]
}

pub fn dir(path: &str) -> SourceEntry {
    SourceEntry::new(path, EntryKind::Directory, Metadata::new(0o755, time()))
}

pub fn file(path: &str, content: Vec<u8>) -> SourceEntry {
    SourceEntry::new(
        path,
        EntryKind::File(content.into()),
        Metadata::new(0o644, time()),
    )
}

pub fn symlink(path: &str, target: &str) -> SourceEntry {
    SourceEntry::new(
        path,
        EntryKind::Symlink(target.as_bytes().to_vec()),
        Metadata::new(0o755, time()),
    )
}

pub fn system_entries() -> Vec<SourceEntry> {
    let mut e = vec![];
    let mut p = String::new();
    for c in [
        "System",
        "Library",
        "Frameworks",
        "Virtualization.framework",
        "Versions",
        "A",
        "Resources",
    ] {
        p = format!("{p}/{c}");
        e.push(dir(&p));
    }
    e.push(symlink(&format!("{FRAMEWORK}/Versions/Current"), "A"));
    e.push(symlink(
        &format!("{FRAMEWORK}/Resources"),
        "Versions/Current/Resources",
    ));
    e.push(symlink("/abs", "/System/Library"));
    e.push(symlink("/loop", "loop"));
    e.push(file(BOOTER, noise(300_000, 1)));
    e.push(file("/empty", Vec::new()));
    e.push(dir("/c"));
    for (path, content, form) in compressed() {
        e.push(file(path, content).with_darwin(Darwin::new(0).compressed(form)));
    }
    e.push(dir("/many"));
    for i in 0..MANY {
        e.push(file(&format!("/many/f{i:05}"), format!("{i}").into_bytes()));
    }
    e
}

/// A disk image in memory that keeps only the pages written, the rest reading as zeros; writes
/// and seeks are from `base`, so a container can be written into a partition.
#[derive(Clone, Default)]
pub struct Sparse {
    pub pages: BTreeMap<u64, Vec<u8>>,
    pub len: u64,
    pub base: u64,
    pub pos: u64,
}

pub const PAGE: u64 = 4096;

impl Sparse {
    pub fn new(len: u64) -> Self {
        Self {
            len,
            ..Self::default()
        }
    }

    pub fn page(&mut self, at: u64) -> &mut Vec<u8> {
        self.pages
            .entry(at / PAGE)
            .or_insert_with(|| vec![0; PAGE as usize])
    }

    pub fn get(&self, off: u64, len: usize) -> Vec<u8> {
        let mut b = vec![0; len];
        self.read_at(off, &mut b).unwrap();
        b
    }
}

impl Write for Sparse {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let at = self.base + self.pos;
        let n = buf.len().min((PAGE - at % PAGE) as usize);
        self.page(at)[(at % PAGE) as usize..][..n].copy_from_slice(&buf[..n]);
        self.pos += n as u64;
        self.len = self.len.max(self.base + self.pos);
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Read for Sparse {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let at = self.base + self.pos;
        let n = buf.len().min(self.len.saturating_sub(at) as usize);
        self.read_at(at, &mut buf[..n])?;
        self.pos += n as u64;
        Ok(n)
    }
}

impl Seek for Sparse {
    fn seek(&mut self, to: SeekFrom) -> std::io::Result<u64> {
        self.pos = match to {
            SeekFrom::Start(x) => x,
            SeekFrom::End(x) => (self.len - self.base).saturating_add_signed(x),
            SeekFrom::Current(x) => self.pos.saturating_add_signed(x),
        };
        Ok(self.pos)
    }
}

impl Source for Sparse {
    fn read_at(&self, off: u64, buf: &mut [u8]) -> std::io::Result<()> {
        if off + buf.len() as u64 > self.len {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        for (i, b) in buf.iter_mut().enumerate() {
            let at = off + i as u64;
            *b = self
                .pages
                .get(&(at / PAGE))
                .map_or(0, |p| p[(at % PAGE) as usize]);
        }
        Ok(())
    }

    fn len(&self) -> u64 {
        self.len
    }
}

/// The containers' size: two volumes take a container of at least 1 GiB.
pub const CONTAINER: u64 = 1 << 30;

/// Format a container of a Data volume in slot 0 and a System volume in slot 1, joined as a
/// group, into `disk` from its `base`.
pub fn format_into(disk: &mut Sparse) {
    let data = VolumeRequest::new(VolumeName::new("Data").unwrap(), [0x33; 16]);
    let system = VolumeRequest::new(VolumeName::new("Macintosh HD").unwrap(), [0x22; 16])
        .contents(system_entries());
    let options = FormatOptions::new([0x11; 16], time(), data)
        .volume(system)
        .volume_group(VolumeGroup::new(1, 0));
    disk.pos = 0;
    format_to(
        &mut *disk,
        vec![file("/marker", b"data".to_vec())],
        CONTAINER,
        options,
    )
    .expect("formats");
}

pub fn container() -> Sparse {
    let mut disk = Sparse::new(CONTAINER);
    format_into(&mut disk);
    disk
}

pub fn source(b: Sparse) -> Arc<dyn Source> {
    Arc::new(b)
}
