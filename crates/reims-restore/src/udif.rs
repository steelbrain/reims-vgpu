//! UDIF disk images (`.dmg`), read at random offsets without unpacking them: the koly trailer,
//! the `blkx` tables of its resource fork, and each chunk decompressed when first read, with the
//! last few kept.

use std::collections::VecDeque;
use std::io::{self, Read};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail, ensure};
use plist::Value;

use crate::source::ReadAt;

const SECTOR: u64 = 512;
/// The most bytes one chunk may expand to: a UDIF chunk is at most a few MiB.
const MAX_CHUNK: u64 = 64 << 20;
/// How many decompressed chunks are kept.
const CACHE_CHUNKS: usize = 48;

/// What a chunk holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Zero,
    Raw,
    Adc,
    Zlib,
    Bzip2,
    Lzfse,
    Lzma,
}

/// One chunk of the image: sectors `start..start + count`, from `len` bytes at `offset`.
#[derive(Clone, Copy, Debug)]
struct Chunk {
    kind: Kind,
    start: u64,
    count: u64,
    offset: u64,
    len: u64,
}

/// One partition the image's `blkx` tables describe.
#[derive(Clone, Debug)]
pub struct Partition {
    /// Its name, such as `Apple_APFS : 4` or `Protective Master Boot Record (MBR : 0)`.
    pub name: String,
    /// Its first byte in the image.
    pub offset: u64,
    /// Its length in bytes.
    pub len: u64,
}

/// A UDIF image over a source of its bytes.
pub struct Udif<S> {
    src: S,
    chunks: Vec<Chunk>,
    partitions: Vec<Partition>,
    len: u64,
    cache: Mutex<VecDeque<(usize, Arc<Vec<u8>>)>>,
}

fn be32(b: &[u8], o: usize) -> u32 {
    u32::from_be_bytes(b[o..o + 4].try_into().unwrap())
}
fn be64(b: &[u8], o: usize) -> u64 {
    u64::from_be_bytes(b[o..o + 8].try_into().unwrap())
}

impl<S: ReadAt> Udif<S> {
    /// Opens the image `src` holds: reads its trailer and chunk tables.
    pub fn open(src: S) -> Result<Self> {
        let total = src.len();
        ensure!(
            total >= 512,
            "the image is {total} bytes, too short for a UDIF trailer"
        );
        let mut koly = [0u8; 512];
        src.read_exact_at(total - 512, &mut koly)
            .context("read the image's trailer")?;
        ensure!(&koly[..4] == b"koly", "not a UDIF image: no koly trailer");
        let data_fork = be64(&koly, 24);
        let xml_offset = be64(&koly, 216);
        let xml_len = be64(&koly, 224);
        ensure!(
            xml_len > 0 && xml_len <= 64 << 20,
            "the image's plist is {xml_len} bytes"
        );
        ensure!(
            xml_offset.checked_add(xml_len).is_some_and(|e| e <= total),
            "the image's plist lies past its end"
        );
        let mut xml = vec![0u8; xml_len as usize];
        src.read_exact_at(xml_offset, &mut xml)
            .context("read the image's plist")?;
        let plist = Value::from_reader(io::Cursor::new(&xml)).context("parse the image's plist")?;
        let blkx = plist
            .as_dictionary()
            .and_then(|d| d.get("resource-fork"))
            .and_then(Value::as_dictionary)
            .and_then(|d| d.get("blkx"))
            .and_then(Value::as_array)
            .context("the image's plist has no resource-fork blkx")?;

        let mut chunks = Vec::new();
        let mut partitions = Vec::new();
        for entry in blkx {
            let d = entry
                .as_dictionary()
                .context("a blkx entry is not a dictionary")?;
            let name = d
                .get("Name")
                .or_else(|| d.get("CFName"))
                .and_then(Value::as_string)
                .unwrap_or("")
                .to_owned();
            let data = d
                .get("Data")
                .and_then(Value::as_data)
                .context("a blkx entry has no Data")?;
            ensure!(
                data.len() >= 204 && &data[..4] == b"mish",
                "blkx {name:?} is not a mish table"
            );
            let first = be64(data, 8);
            let count = be64(data, 16);
            let base = data_fork + be64(data, 24);
            let n = be32(data, 200) as usize;
            ensure!(
                data.len() >= 204 + n * 40,
                "blkx {name:?} is shorter than its {n} chunks"
            );
            for i in 0..n {
                let c = &data[204 + i * 40..204 + (i + 1) * 40];
                let kind = match be32(c, 0) {
                    0x0000_0000 | 0x0000_0002 => Kind::Zero,
                    0x0000_0001 => Kind::Raw,
                    0x8000_0004 => Kind::Adc,
                    0x8000_0005 => Kind::Zlib,
                    0x8000_0006 => Kind::Bzip2,
                    0x8000_0007 => Kind::Lzfse,
                    0x8000_0008 => Kind::Lzma,
                    0x7fff_fffe | 0xffff_ffff => continue,
                    t => bail!(
                        "blkx {name:?} has a chunk of type {t:#x}, which this reader doesn't know"
                    ),
                };
                let ch = Chunk {
                    kind,
                    start: first + be64(c, 8),
                    count: be64(c, 16),
                    offset: base + be64(c, 24),
                    len: be64(c, 32),
                };
                // Zero and raw chunks are read in place, at any size; a compressed one is
                // expanded whole.
                ensure!(
                    matches!(kind, Kind::Zero | Kind::Raw) || ch.count * SECTOR <= MAX_CHUNK,
                    "blkx {name:?} has a compressed chunk of {} sectors",
                    ch.count
                );
                ensure!(
                    ch.offset.checked_add(ch.len).is_some_and(|e| e <= total),
                    "blkx {name:?} has a chunk past the image's end: the image is truncated"
                );
                if ch.count > 0 {
                    chunks.push(ch);
                }
            }
            partitions.push(Partition {
                name,
                offset: first * SECTOR,
                len: count * SECTOR,
            });
        }
        chunks.sort_by_key(|c| c.start);
        for w in chunks.windows(2) {
            ensure!(
                w[0].start + w[0].count <= w[1].start,
                "the image's chunks overlap"
            );
        }
        let len = chunks.last().map_or(0, |c| (c.start + c.count) * SECTOR);
        Ok(Self {
            src,
            chunks,
            partitions,
            len,
            cache: Mutex::new(VecDeque::new()),
        })
    }

    /// The partitions its `blkx` tables name.
    pub fn partitions(&self) -> &[Partition] {
        &self.partitions
    }

    /// The one partition whose name contains `what`, such as `Apple_APFS`.
    pub fn partition(&self, what: &str) -> Result<&Partition> {
        let mut found = self.partitions.iter().filter(|p| p.name.contains(what));
        let p = found
            .next()
            .with_context(|| format!("the image has no {what} partition"))?;
        ensure!(
            found.next().is_none(),
            "the image has more than one {what} partition"
        );
        Ok(p)
    }

    /// The decompressed bytes of chunk `i`.
    fn chunk(&self, i: usize) -> io::Result<Arc<Vec<u8>>> {
        if let Some((_, d)) = self.cache.lock().unwrap().iter().find(|(j, _)| *j == i) {
            return Ok(d.clone());
        }
        let c = self.chunks[i];
        let want = (c.count * SECTOR) as usize;
        let mut packed = vec![0u8; c.len as usize];
        self.src.read_exact_at(c.offset, &mut packed)?;
        let bad = |what: &str| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("a {what} chunk of the image is damaged"),
            )
        };
        let mut out = Vec::with_capacity(want);
        match c.kind {
            Kind::Zlib => {
                flate2::read::ZlibDecoder::new(&packed[..])
                    .take(want as u64)
                    .read_to_end(&mut out)
                    .map_err(|_| bad("zlib"))?;
            }
            Kind::Bzip2 => {
                bzip2::read::BzDecoder::new(&packed[..])
                    .take(want as u64)
                    .read_to_end(&mut out)
                    .map_err(|_| bad("bzip2"))?;
            }
            Kind::Lzfse => {
                lzfse_rust::decode_bytes(&packed, &mut out).map_err(|_| bad("LZFSE"))?;
            }
            Kind::Lzma => {
                lzma_rs::xz_decompress(&mut &packed[..], &mut out).map_err(|_| bad("LZMA"))?;
            }
            Kind::Adc => adc_decode(&packed, &mut out, want).map_err(|_| bad("ADC"))?,
            Kind::Zero | Kind::Raw => unreachable!("read in place"),
        }
        if out.len() != want {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "a chunk of the image expands to {} bytes, not {want}",
                    out.len()
                ),
            ));
        }
        let out = Arc::new(out);
        let mut cache = self.cache.lock().unwrap();
        if cache.len() >= CACHE_CHUNKS {
            cache.pop_front();
        }
        cache.push_back((i, out.clone()));
        Ok(out)
    }
}

impl<S: ReadAt> ReadAt for Udif<S> {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        if offset >= self.len || buf.is_empty() {
            return Ok(0);
        }
        let sector = offset / SECTOR;
        // The chunk holding `sector`, or the gap before the next one, which reads as zeroes.
        let i = self.chunks.partition_point(|c| c.start + c.count <= sector);
        let Some(c) = self.chunks.get(i).copied() else {
            return Ok(0);
        };
        if sector < c.start {
            let n = buf.len().min((c.start * SECTOR - offset) as usize);
            buf[..n].fill(0);
            return Ok(n);
        }
        let within = offset - c.start * SECTOR;
        let n = buf.len().min((c.count * SECTOR - within) as usize);
        match c.kind {
            Kind::Zero => buf[..n].fill(0),
            Kind::Raw => {
                // A raw chunk may hold fewer bytes than its sectors; the rest reads as zeroes.
                let have = c.len.saturating_sub(within).min(n as u64) as usize;
                self.src
                    .read_exact_at(c.offset + within, &mut buf[..have])?;
                buf[have..n].fill(0);
            }
            _ => {
                let d = self.chunk(i)?;
                buf[..n].copy_from_slice(&d[within as usize..within as usize + n]);
            }
        }
        Ok(n)
    }

    fn len(&self) -> u64 {
        self.len
    }
}

/// Apple Data Compression, the oldest UDIF codec.
fn adc_decode(input: &[u8], out: &mut Vec<u8>, max: usize) -> Result<(), ()> {
    let mut i = 0;
    while i < input.len() && out.len() < max {
        let b = input[i];
        if b & 0x80 != 0 {
            let n = (b & 0x7f) as usize + 1;
            out.extend_from_slice(input.get(i + 1..i + 1 + n).ok_or(())?);
            i += 1 + n;
        } else if b & 0x40 != 0 {
            let n = (b & 0x3f) as usize + 4;
            let d = u16::from_be_bytes([*input.get(i + 1).ok_or(())?, *input.get(i + 2).ok_or(())?])
                as usize
                + 1;
            copy_back(out, d, n)?;
            i += 3;
        } else {
            let n = ((b & 0x3c) >> 2) as usize + 3;
            let d = (((b & 0x03) as usize) << 8 | *input.get(i + 1).ok_or(())? as usize) + 1;
            copy_back(out, d, n)?;
            i += 2;
        }
    }
    Ok(())
}

fn copy_back(out: &mut Vec<u8>, distance: usize, n: usize) -> Result<(), ()> {
    if distance > out.len() {
        return Err(());
    }
    for _ in 0..n {
        out.push(out[out.len() - distance]);
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Write;

    /// A UDIF image written from scratch: one partition holding `data`, in 4-sector chunks
    /// of each kind in turn, with a run of zeroes left as a zero-fill chunk.
    pub fn synthetic(data: &[u8]) -> Vec<u8> {
        assert_eq!(data.len() % 2048, 0);
        let mut fork = Vec::new();
        let mut table = Vec::new();
        let sectors = data.len() as u64 / SECTOR;
        let mut chunks: Vec<[u8; 40]> = Vec::new();
        for (n, piece) in data.chunks(2048).enumerate() {
            let start = n as u64 * 4;
            let (kind, packed): (u32, Vec<u8>) = if piece.iter().all(|&b| b == 0) {
                (0x2, Vec::new())
            } else {
                match n % 4 {
                    0 => (0x1, piece.to_vec()),
                    1 => {
                        let mut e = flate2::write::ZlibEncoder::new(
                            Vec::new(),
                            flate2::Compression::default(),
                        );
                        e.write_all(piece).unwrap();
                        (0x8000_0005, e.finish().unwrap())
                    }
                    2 => {
                        let mut e =
                            bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::default());
                        e.write_all(piece).unwrap();
                        (0x8000_0006, e.finish().unwrap())
                    }
                    _ => {
                        let mut out = Vec::new();
                        lzfse_rust::encode_bytes(piece, &mut out).unwrap();
                        (0x8000_0007, out)
                    }
                }
            };
            let mut c = [0u8; 40];
            c[0..4].copy_from_slice(&kind.to_be_bytes());
            c[8..16].copy_from_slice(&start.to_be_bytes());
            c[16..24].copy_from_slice(&4u64.to_be_bytes());
            c[24..32].copy_from_slice(&(fork.len() as u64).to_be_bytes());
            c[32..40].copy_from_slice(&(packed.len() as u64).to_be_bytes());
            fork.extend_from_slice(&packed);
            chunks.push(c);
        }
        let mut end = [0u8; 40];
        end[0..4].copy_from_slice(&0xffff_ffffu32.to_be_bytes());
        chunks.push(end);
        table.extend_from_slice(b"mish");
        table.extend_from_slice(&1u32.to_be_bytes());
        table.extend_from_slice(&0u64.to_be_bytes());
        table.extend_from_slice(&sectors.to_be_bytes());
        table.extend_from_slice(&0u64.to_be_bytes());
        table.resize(200, 0);
        table.extend_from_slice(&(chunks.len() as u32).to_be_bytes());
        for c in &chunks {
            table.extend_from_slice(c);
        }
        let mut part = plist::Dictionary::new();
        part.insert("Name".into(), Value::String("Apple_APFS : 1".into()));
        part.insert("Data".into(), Value::Data(table));
        let mut rf = plist::Dictionary::new();
        rf.insert("blkx".into(), Value::Array(vec![Value::Dictionary(part)]));
        let mut root = plist::Dictionary::new();
        root.insert("resource-fork".into(), Value::Dictionary(rf));
        let mut xml = Vec::new();
        Value::Dictionary(root).to_writer_xml(&mut xml).unwrap();
        let mut img = fork;
        let xml_off = img.len() as u64;
        img.extend_from_slice(&xml);
        let mut koly = [0u8; 512];
        koly[0..4].copy_from_slice(b"koly");
        koly[4..8].copy_from_slice(&4u32.to_be_bytes());
        koly[8..12].copy_from_slice(&512u32.to_be_bytes());
        koly[32..40].copy_from_slice(&xml_off.to_be_bytes());
        koly[216..224].copy_from_slice(&xml_off.to_be_bytes());
        koly[224..232].copy_from_slice(&(xml.len() as u64).to_be_bytes());
        koly[492..500].copy_from_slice(&sectors.to_be_bytes());
        img.extend_from_slice(&koly);
        img
    }

    fn pattern(len: usize) -> Vec<u8> {
        let mut v: Vec<u8> = (0..len as u32)
            .map(|i| (i.wrapping_mul(2654435761) >> 13) as u8)
            .collect();
        // A zero run that becomes a zero-fill chunk.
        if len >= 2048 * 6 {
            v[2048 * 5..2048 * 6].fill(0);
        }
        v
    }

    #[test]
    fn reads_every_chunk_kind_at_any_offset() {
        let data = pattern(2048 * 12);
        let img = Udif::open(synthetic(&data)).unwrap();
        assert_eq!(img.len(), data.len() as u64);
        assert_eq!(img.partition("Apple_APFS").unwrap().len, data.len() as u64);
        for (off, n) in [
            (0usize, 100usize),
            (2000, 5000),
            (2048 * 5 - 3, 2060),
            (data.len() - 7, 7),
            (1, data.len() - 1),
        ] {
            let mut buf = vec![0u8; n];
            img.read_exact_at(off as u64, &mut buf).unwrap();
            assert_eq!(buf, data[off..off + n], "at {off}+{n}");
        }
        let mut buf = [0u8; 4];
        assert_eq!(img.read_at(data.len() as u64, &mut buf).unwrap(), 0);
    }

    #[test]
    fn a_damaged_chunk_is_an_error_not_wrong_bytes() {
        let data = pattern(2048 * 4);
        let mut raw = synthetic(&data);
        // Chunk 1 is zlib; it starts right after chunk 0's 2048 raw bytes.
        raw[2048 + 10] ^= 0xff;
        let img = Udif::open(raw).unwrap();
        let mut buf = vec![0u8; 2048];
        assert!(img.read_exact_at(2048, &mut buf).is_err());
    }

    #[test]
    fn a_truncated_image_is_refused() {
        let data = pattern(2048 * 4);
        let raw = synthetic(&data);
        assert!(Udif::open(raw[100..].to_vec()).is_err());
        assert!(Udif::open(raw[..raw.len() - 600].to_vec()).is_err());
    }

    #[test]
    fn adc_decodes_literals_and_copies() {
        // "abc" literal, then a 4-byte copy from 3 back: "abcabca".
        let input = [0x82, b'a', b'b', b'c', 0x40, 0x00, 0x02];
        let mut out = Vec::new();
        adc_decode(&input, &mut out, 100).unwrap();
        assert_eq!(out, b"abcabca");
    }
}
