//! The system volume's canonical metadata (`Ap,SystemVolumeCanonicalMetadata`, IM4P `msys`): an
//! mtree of the sealed volume. A seal sets every inode's four times and every directory entry's
//! date added to one instant, its root directory's `time=` (docs/design.md, "The system volume's
//! seal"); this reads that instant.
//!
//! The payload is `pbze`: a big-endian block size, then blocks, each its decoded and stored sizes
//! (big-endian) and its bytes, an LZFSE stream where it is stored smaller than it decodes and the
//! bytes themselves where not **[measured]**. Decoded, it is an Apple Archive (`AA01`) of three
//! entries on 27.2.1, a directory, `digest.db` (135 MB) and `mtree.txt` (90 MB) **[ipsw]**; the
//! mtree's text has its root directory's entry first: `. type=dir time=<seconds>.<nanoseconds>`.
//! An archive entry is `AA01`, its header's size (16-bit little-endian, the magic included), and
//! fields each a three-letter key, a type letter and a value, its blobs following the header in
//! the order their fields name them.

use anyhow::{Context, Result, bail, ensure};

use crate::img4;

/// The archive entry holding the mtree's text.
const MTREE: &str = "mtree.txt";
/// The most of the mtree's text read looking for the root's entry: it comes first.
const SEARCHED: usize = 1 << 20;

/// The root directory's `time=` of the mtree IM4P `im4p`, in nanoseconds since 1970.
///
/// # Errors
///
/// The IM4P is not `msys`, its payload is not `pbze` or fails to decode, its archive holds no
/// `mtree.txt`, or no root entry with a time is found in that text's first MiB.
pub fn root_time(im4p: &[u8]) -> Result<u64> {
    let p = img4::parse_im4p(im4p).context("read the canonical metadata's IM4P")?;
    ensure!(
        &p.tag == b"msys",
        "the canonical metadata is IM4P {}, not msys",
        String::from_utf8_lossy(&p.tag)
    );
    let mut z = Pbze::new(&p.payload)?;
    let text = archive_member(&mut z, MTREE, SEARCHED)?;
    find_root_time(&text).context("find the mtree root's time")
}

/// A `pbze` payload read front to back, skipping whole blocks without decoding them.
struct Pbze<'a> {
    b: &'a [u8],
    /// The next block's header.
    at: usize,
    block: Vec<u8>,
    pos: usize,
}

impl<'a> Pbze<'a> {
    fn new(b: &'a [u8]) -> Result<Self> {
        ensure!(b.len() >= 12 && &b[..4] == b"pbze", "not a pbze payload");
        Ok(Self {
            b,
            at: 12,
            block: Vec::new(),
            pos: 0,
        })
    }

    /// The next block's decoded and stored sizes, or `None` at the end.
    fn next_sizes(&self) -> Result<Option<(usize, usize)>> {
        if self.at >= self.b.len() {
            return Ok(None);
        }
        let h = self
            .b
            .get(self.at..self.at + 16)
            .context("a pbze block header runs past the payload")?;
        let size = u64::from_be_bytes(h[..8].try_into().unwrap()) as usize;
        let stored = u64::from_be_bytes(h[8..].try_into().unwrap()) as usize;
        ensure!(
            self.at + 16 + stored <= self.b.len(),
            "a pbze block runs past the payload"
        );
        Ok(Some((size, stored)))
    }

    /// Decode the next block into `block`.
    fn decode_next(&mut self) -> Result<()> {
        let (size, stored) = self.next_sizes()?.context("the pbze payload ends early")?;
        let data = &self.b[self.at + 16..self.at + 16 + stored];
        self.at += 16 + stored;
        self.block.clear();
        self.pos = 0;
        if stored < size {
            lzfse_rust::decode_bytes(data, &mut self.block)
                .map_err(|e| anyhow::anyhow!("a pbze block's LZFSE: {e}"))?;
            ensure!(
                self.block.len() == size,
                "a pbze block decodes to {} of {size} bytes",
                self.block.len()
            );
        } else {
            self.block.extend_from_slice(data);
        }
        Ok(())
    }

    /// Pass over `n` bytes, decoding only the block they end in.
    fn skip(&mut self, mut n: u64) -> Result<()> {
        let here = (self.block.len() - self.pos) as u64;
        if n <= here {
            self.pos += n as usize;
            return Ok(());
        }
        n -= here;
        self.pos = self.block.len();
        while let Some((size, stored)) = self.next_sizes()? {
            if n < size as u64 {
                self.decode_next()?;
                self.pos = n as usize;
                return Ok(());
            }
            n -= size as u64;
            self.at += 16 + stored;
        }
        ensure!(n == 0, "the pbze payload ends early");
        Ok(())
    }

    fn read_exact(&mut self, out: &mut [u8]) -> Result<()> {
        let mut done = 0;
        while done < out.len() {
            if self.pos == self.block.len() {
                self.decode_next()?;
            }
            let n = (out.len() - done).min(self.block.len() - self.pos);
            out[done..done + n].copy_from_slice(&self.block[self.pos..self.pos + n]);
            self.pos += n;
            done += n;
        }
        Ok(())
    }
}

/// The first `limit` bytes (or all) of the data of archive member `name`.
fn archive_member(z: &mut Pbze, name: &str, limit: usize) -> Result<Vec<u8>> {
    loop {
        let mut head = [0u8; 6];
        z.read_exact(&mut head)
            .with_context(|| format!("the archive holds no {name}"))?;
        ensure!(
            &head[..4] == b"AA01",
            "not an Apple Archive entry: {:02x?}",
            &head[..4]
        );
        let size = usize::from(u16::from_le_bytes([head[4], head[5]]));
        ensure!(size >= 6, "an archive header of {size} bytes");
        let mut h = vec![0u8; size - 6];
        z.read_exact(&mut h)?;
        let (path, blobs) = parse_header(&h)?;
        for (key, len) in blobs {
            if path == name && key == *b"DAT" {
                let n = usize::try_from(len).unwrap_or(usize::MAX).min(limit);
                let mut out = vec![0u8; n];
                z.read_exact(&mut out)?;
                return Ok(out);
            }
            z.skip(len)?;
        }
    }
}

/// An archive entry's blobs: each one's key and size, in order.
type Blobs = Vec<([u8; 3], u64)>;

/// An archive header's path and its blobs' keys and sizes, in order.
fn parse_header(h: &[u8]) -> Result<(String, Blobs)> {
    let mut path = String::new();
    let mut blobs = Vec::new();
    let mut i = 0;
    while i < h.len() {
        let f = h
            .get(i..i + 4)
            .context("an archive field runs past its header")?;
        let key = [f[0], f[1], f[2]];
        let ty = f[3];
        i += 4;
        let width = match ty {
            b'*' => 0,
            b'1' | b'2' | b'4' | b'8' => usize::from(ty - b'0'),
            b'A' => 2,
            b'B' => 4,
            b'C' => 8,
            b'P' => {
                let l = h
                    .get(i..i + 2)
                    .context("an archive path runs past its header")?;
                2 + usize::from(u16::from_le_bytes([l[0], l[1]]))
            }
            b'S' => 8,
            b'T' => 12,
            b'F' => 4,
            b'G' => 20,
            b'H' => 32,
            b'I' => 48,
            b'J' => 64,
            _ => bail!(
                "an archive field {}{} of a type this reader doesn't know",
                String::from_utf8_lossy(&key),
                ty as char
            ),
        };
        let v = h
            .get(i..i + width)
            .context("an archive field runs past its header")?;
        match ty {
            b'P' if key == *b"PAT" => path = String::from_utf8_lossy(&v[2..]).into_owned(),
            b'A'..=b'C' => {
                let mut n = [0u8; 8];
                n[..width].copy_from_slice(v);
                blobs.push((key, u64::from_le_bytes(n)));
            }
            _ => {}
        }
        i += width;
    }
    Ok((path, blobs))
}

/// The `time=` of the first entry naming `.` as a directory, the mtree's root.
fn find_root_time(text: &[u8]) -> Result<u64> {
    // An entry runs over lines joined by a trailing backslash; the root's starts with `.` and
    // white space at the start of a line.
    let mut entry = String::new();
    for line in text.split(|&c| c == b'\n') {
        let line = String::from_utf8_lossy(line);
        let starts = entry.is_empty();
        if starts && !(line.starts_with(". ") || line.starts_with(".\t")) {
            continue;
        }
        entry.push_str(line.trim_end_matches('\\'));
        entry.push(' ');
        if line.ends_with('\\') {
            continue;
        }
        let words: Vec<&str> = entry.split_whitespace().collect();
        if words.contains(&"type=dir")
            && let Some(t) = words.iter().find_map(|w| w.strip_prefix("time="))
        {
            return parse_time(t);
        }
        entry.clear();
    }
    bail!("no root directory entry with a time")
}

/// `<seconds>.<nanoseconds>` in nanoseconds.
fn parse_time(t: &str) -> Result<u64> {
    let (s, ns) = t.split_once('.').unwrap_or((t, "0"));
    ensure!(ns.len() <= 9, "time {t} has more than nanoseconds");
    let s: u64 = s.parse().with_context(|| format!("time {t}"))?;
    let ns: u64 = format!("{ns:0<9}")
        .parse()
        .with_context(|| format!("time {t}"))?;
    s.checked_mul(1_000_000_000)
        .and_then(|x| x.checked_add(ns))
        .with_context(|| format!("time {t} out of range"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pbze payload of `text`, cut in `block`-byte blocks, each LZFSE where that is smaller.
    fn pbze(text: &[u8], block: usize) -> Vec<u8> {
        let mut out = b"pbze".to_vec();
        out.extend_from_slice(&(block as u64).to_be_bytes());
        for chunk in text.chunks(block) {
            let mut z = Vec::new();
            lzfse_rust::encode_bytes(chunk, &mut z).unwrap();
            let stored: &[u8] = if z.len() < chunk.len() { &z } else { chunk };
            out.extend_from_slice(&(chunk.len() as u64).to_be_bytes());
            out.extend_from_slice(&(stored.len() as u64).to_be_bytes());
            out.extend_from_slice(stored);
        }
        out
    }

    /// An archive entry: a header of `fields` (key and type letter, value), then `blobs`.
    fn entry(fields: &[(&str, Vec<u8>)], blobs: &[&[u8]]) -> Vec<u8> {
        let mut h = Vec::new();
        for (k, v) in fields {
            h.extend_from_slice(k.as_bytes());
            h.extend_from_slice(v);
        }
        let mut out = b"AA01".to_vec();
        out.extend_from_slice(&((6 + h.len()) as u16).to_le_bytes());
        out.extend_from_slice(&h);
        for b in blobs {
            out.extend_from_slice(b);
        }
        out
    }

    fn path(p: &str) -> Vec<u8> {
        let mut v = (p.len() as u16).to_le_bytes().to_vec();
        v.extend_from_slice(p.as_bytes());
        v
    }

    // An mtree of the form the canonical metadata holds, written here: a header, a `/set`, then
    // the root's entry over continued lines, then others with times of their own.
    fn mtree() -> Vec<u8> {
        let mut t = b"#\t   user: root\n\n# .\n/set type=file\n".to_vec();
        t.extend_from_slice(
            b".               type=dir time=1700000123.450000000 \\\n                btime=1 ctime=1 inode=2 nxattr=0\n",
        );
        t.extend_from_slice(b"    a     type=dir time=99.000000000 inode=16\n");
        for i in 0..5000 {
            t.extend_from_slice(format!("    f{i}  time=5.0 inode={}\n", 17 + i).as_bytes());
        }
        t
    }

    // An archive as the canonical metadata's: a directory, a large database the reader passes
    // over, then the mtree, with fields of several types around the path and the blob.
    fn archive() -> Vec<u8> {
        let ts = vec![0u8; 12];
        let db: Vec<u8> = (0..300_000u32).map(|i| (i * 7 + i / 251) as u8).collect();
        let text = mtree();
        let mut a = entry(
            &[
                ("TYP1", b"D".to_vec()),
                ("PATP", path("")),
                ("MOD2", vec![0xed, 1]),
                ("CTMT", ts.clone()),
            ],
            &[],
        );
        a.extend(entry(
            &[
                ("TYP1", b"F".to_vec()),
                ("PATP", path("digest.db")),
                ("UID1", vec![0]),
                ("MTMT", ts.clone()),
                ("XATA", (5u16).to_le_bytes().to_vec()),
                ("DATB", (db.len() as u32).to_le_bytes().to_vec()),
            ],
            &[b"xattr", &db],
        ));
        a.extend(entry(
            &[
                ("TYP1", b"F".to_vec()),
                ("PATP", path("mtree.txt")),
                ("SH2H", vec![0xab; 32]),
                ("DATB", (text.len() as u32).to_le_bytes().to_vec()),
            ],
            &[&text],
        ));
        a
    }

    #[test]
    fn the_root_time_is_read_through_every_block_form() {
        for block in [64usize, 4096, 1 << 20] {
            let b = pbze(&archive(), block);
            let mut z = Pbze::new(&b).unwrap();
            let text = archive_member(&mut z, MTREE, SEARCHED).unwrap();
            assert_eq!(
                find_root_time(&text).unwrap(),
                1_700_000_123_450_000_000,
                "{block}"
            );
        }
    }

    #[test]
    fn times_parse_to_nanoseconds() {
        assert_eq!(
            parse_time("1793942531.000000000").unwrap(),
            1_793_942_531_000_000_000
        );
        assert_eq!(parse_time("1.5").unwrap(), 1_500_000_000);
        assert_eq!(parse_time("7").unwrap(), 7_000_000_000);
        assert!(parse_time("1.0000000001").is_err());
    }

    #[test]
    fn an_mtree_without_a_root_time_is_refused() {
        assert!(find_root_time(b". type=dir inode=2\n    a time=3.0\n").is_err());
        assert!(Pbze::new(b"pbzx\0\0\0\0\0\0\0\0").is_err());
        // An archive without the mtree.
        let b = pbze(&archive()[..300_100], 4096);
        assert!(archive_member(&mut Pbze::new(&b).unwrap(), MTREE, SEARCHED).is_err());
    }
}
