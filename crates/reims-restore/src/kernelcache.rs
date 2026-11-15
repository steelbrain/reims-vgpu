//! The boot kernel collection an IPSW carries (its install identity's `KernelCache`, IM4P
//! `krnl`): the payload decompressed — LZFSE (`bvx2`) or LZSS (`complzss`) — and read as a
//! Mach-O file set (`MH_FILESET`), each kext found by its `LC_FILESET_ENTRY` and its segments by
//! its own load commands. Ambrosia loads this same kernel collection from the guest's Preboot and
//! patches kexts' segments in place; the LocalPolicy's trust anchors are read from it
//! ([`crate::anchors`]).
//!
//! A file set's offsets are from the start of the whole collection: an entry's Mach-O header
//! is at its `fileoff`, and its segments' and sections' file offsets are the collection's.

use anyhow::{Context, Result, bail, ensure};

use crate::img4;

const MH_MAGIC_64: u32 = 0xfeed_facf;
const MH_FILESET: u32 = 0xc;
const LC_SEGMENT_64: u32 = 0x19;
const LC_FILESET_ENTRY: u32 = 0x8000_0035;
const MACH_HEADER_64: usize = 32;
const SEGMENT_COMMAND_64: usize = 72;
const SECTION_64: usize = 80;

/// The LZSS payload's header: `complzss`, an Adler-32 of the decompressed bytes, their size
/// and the compressed size (each 32-bit big-endian), then padding; the stream starts at 0x180.
const LZSS_MAGIC: &[u8; 8] = b"complzss";
const LZSS_DATA: usize = 0x180;

/// The kernel collection IM4P `im4p` holds, decompressed.
///
/// # Errors
///
/// The IM4P is not `krnl`, its payload is neither compressed form nor a Mach-O, or it does not
/// decompress to the size it records.
pub fn decompress(im4p: &[u8]) -> Result<Vec<u8>> {
    let p = img4::parse_im4p(im4p).context("read the kernel cache's IM4P")?;
    ensure!(
        &p.tag == b"krnl",
        "the kernel cache is IM4P {}, not krnl",
        String::from_utf8_lossy(&p.tag)
    );
    let b = &p.payload;
    let out = if b.starts_with(b"bvx2") || b.starts_with(b"bvx1") || b.starts_with(b"bvxn") {
        let mut out = Vec::new();
        lzfse_rust::decode_bytes(b, &mut out)
            .map_err(|e| anyhow::anyhow!("the kernel cache's LZFSE: {e}"))?;
        if let Some(c) = p.compression {
            ensure!(
                out.len() as u64 == c.size,
                "the kernel cache decompresses to {} bytes, not the {} its IM4P records",
                out.len(),
                c.size
            );
        }
        out
    } else if b.starts_with(LZSS_MAGIC) {
        lzss_payload(b)?
    } else if b.get(..4) == Some(&MH_MAGIC_64.to_le_bytes()) {
        b.clone()
    } else {
        bail!("the kernel cache's payload is neither LZFSE, LZSS nor a Mach-O")
    };
    Ok(out)
}

/// A `complzss` payload, decompressed and checked against its header.
fn lzss_payload(b: &[u8]) -> Result<Vec<u8>> {
    ensure!(
        b.len() >= LZSS_DATA,
        "an LZSS payload shorter than its header"
    );
    let be = |at: usize| u32::from_be_bytes(b[at..at + 4].try_into().unwrap());
    let (sum, size, stored) = (be(8), be(12) as usize, be(16) as usize);
    let data = b
        .get(LZSS_DATA..LZSS_DATA + stored)
        .context("an LZSS payload shorter than its compressed size")?;
    let out = lzss(data, size);
    ensure!(
        out.len() == size,
        "an LZSS payload decompresses to {} of {size} bytes",
        out.len()
    );
    ensure!(adler32(&out) == sum, "an LZSS payload fails its checksum");
    Ok(out)
}

/// LZSS as Apple's kernel caches use it: a 4 KiB ring first filled with spaces, a flag byte
/// before each eight items, a set bit a literal byte and a clear one a 12-bit ring position and
/// a 4-bit length of three to eighteen bytes. Stops at `size` bytes or the input's end.
fn lzss(src: &[u8], size: usize) -> Vec<u8> {
    const N: usize = 4096;
    const F: usize = 18;
    const THRESHOLD: usize = 2;
    let mut ring = [b' '; N];
    let mut r = N - F;
    let mut out = Vec::with_capacity(size);
    let mut flags = 0u32;
    let mut i = 0;
    while out.len() < size {
        flags >>= 1;
        if flags & 0x100 == 0 {
            let Some(&f) = src.get(i) else { break };
            i += 1;
            flags = u32::from(f) | 0xff00;
        }
        if flags & 1 != 0 {
            let Some(&c) = src.get(i) else { break };
            i += 1;
            out.push(c);
            ring[r] = c;
            r = (r + 1) % N;
        } else {
            let (Some(&a), Some(&b)) = (src.get(i), src.get(i + 1)) else {
                break;
            };
            i += 2;
            let pos = usize::from(a) | (usize::from(b & 0xf0) << 4);
            let len = usize::from(b & 0x0f) + THRESHOLD + 1;
            for k in 0..len {
                let c = ring[(pos + k) % N];
                out.push(c);
                ring[r] = c;
                r = (r + 1) % N;
            }
        }
    }
    out.truncate(size);
    out
}

fn adler32(d: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for chunk in d.chunks(5552) {
        for &x in chunk {
            a += u32::from(x);
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    b << 16 | a
}

/// A kernel collection read as a Mach-O file set: each entry's identifier and where its Mach-O
/// header is.
pub struct FileSet<'a> {
    b: &'a [u8],
    entries: Vec<(String, usize)>,
}

fn u32_at(b: &[u8], at: usize) -> Result<u32> {
    let x = b
        .get(at..at + 4)
        .context("a Mach-O field past the file's end")?;
    Ok(u32::from_le_bytes(x.try_into().unwrap()))
}

fn u64_at(b: &[u8], at: usize) -> Result<u64> {
    let x = b
        .get(at..at + 8)
        .context("a Mach-O field past the file's end")?;
    Ok(u64::from_le_bytes(x.try_into().unwrap()))
}

/// A fixed-width, NUL-padded name.
fn name16(b: &[u8], at: usize) -> Result<&[u8]> {
    let n = b
        .get(at..at + 16)
        .context("a Mach-O name past the file's end")?;
    Ok(&n[..n.iter().position(|&c| c == 0).unwrap_or(16)])
}

/// The load commands of the Mach-O header at `at`: each command's type and where it starts.
fn load_commands(b: &[u8], at: usize) -> Result<Vec<(u32, usize)>> {
    ensure!(
        u32_at(b, at)? == MH_MAGIC_64,
        "no 64-bit Mach-O header at {at:#x}"
    );
    let n = u32_at(b, at + 16)? as usize;
    let size = u32_at(b, at + 20)? as usize;
    let end = at + MACH_HEADER_64 + size;
    ensure!(end <= b.len(), "Mach-O load commands past the file's end");
    let mut out = Vec::with_capacity(n);
    let mut c = at + MACH_HEADER_64;
    for _ in 0..n {
        let cmd = u32_at(b, c)?;
        let len = u32_at(b, c + 4)? as usize;
        ensure!(
            len >= 8 && c + len <= end,
            "a Mach-O load command runs past the commands"
        );
        out.push((cmd, c));
        c += len;
    }
    Ok(out)
}

impl<'a> FileSet<'a> {
    /// Read `b` as a file set.
    ///
    /// # Errors
    ///
    /// `b` is not a 64-bit Mach-O file set, or its commands run past it.
    pub fn parse(b: &'a [u8]) -> Result<Self> {
        ensure!(
            u32_at(b, 0)? == MH_MAGIC_64 && u32_at(b, 12)? == MH_FILESET,
            "the kernel cache is not a Mach-O file set"
        );
        let mut entries = Vec::new();
        for (cmd, c) in load_commands(b, 0)? {
            if cmd != LC_FILESET_ENTRY {
                continue;
            }
            let fileoff = usize::try_from(u64_at(b, c + 16)?)?;
            let id_at = c + u32_at(b, c + 24)? as usize;
            let len = u32_at(b, c + 4)? as usize;
            let id = b
                .get(id_at..c + len)
                .context("a file set entry's name past its command")?;
            let id = &id[..id.iter().position(|&x| x == 0).unwrap_or(id.len())];
            entries.push((String::from_utf8_lossy(id).into_owned(), fileoff));
        }
        Ok(Self { b, entries })
    }

    /// The entries' identifiers, in the order the file set lists them.
    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(id, _)| id.as_str())
    }

    fn header(&self, id: &str) -> Result<usize> {
        self.entries
            .iter()
            .find(|(e, _)| e == id)
            .map(|&(_, at)| at)
            .with_context(|| format!("the kernel cache holds no {id}"))
    }

    /// The bytes of each of entry `id`'s segments that the file holds, in its commands' order.
    ///
    /// # Errors
    ///
    /// No entry is `id`, or its header or segments run past the file.
    pub fn segments(&self, id: &str) -> Result<Vec<(String, &'a [u8])>> {
        let mut out = Vec::new();
        for (cmd, c) in load_commands(self.b, self.header(id)?)? {
            if cmd != LC_SEGMENT_64 {
                continue;
            }
            let name = String::from_utf8_lossy(name16(self.b, c + 8)?).into_owned();
            let off = usize::try_from(u64_at(self.b, c + 40)?)?;
            let size = usize::try_from(u64_at(self.b, c + 48)?)?;
            let bytes = self
                .b
                .get(off..off + size)
                .with_context(|| format!("{id}'s segment {name} past the file's end"))?;
            out.push((name, bytes));
        }
        Ok(out)
    }

    /// Section `sect` of segment `seg` of entry `id`.
    ///
    /// # Errors
    ///
    /// No entry is `id`, it has no such section, or the section runs past the file.
    pub fn section(&self, id: &str, seg: &str, sect: &str) -> Result<&'a [u8]> {
        for (cmd, c) in load_commands(self.b, self.header(id)?)? {
            if cmd != LC_SEGMENT_64 || name16(self.b, c + 8)? != seg.as_bytes() {
                continue;
            }
            let n = u32_at(self.b, c + 64)? as usize;
            for i in 0..n {
                let s = c + SEGMENT_COMMAND_64 + i * SECTION_64;
                if name16(self.b, s)? != sect.as_bytes() {
                    continue;
                }
                let size = usize::try_from(u64_at(self.b, s + 40)?)?;
                let off = u32_at(self.b, s + 48)? as usize;
                return self
                    .b
                    .get(off..off + size)
                    .with_context(|| format!("{id}'s {seg},{sect} past the file's end"));
            }
        }
        bail!("{id} has no {seg},{sect}")
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    // An LZSS stream written by hand: three literals, then six bytes from the ring where the
    // first literal went (N - F = 4078 = 0xfee).
    #[test]
    fn lzss_reads_literals_and_ring_references() {
        let src = [0x07, b'a', b'b', b'c', 0xee, 0xf3];
        assert_eq!(lzss(&src, 9), b"abcabcabc");
        // Before anything is written the ring reads as spaces.
        assert_eq!(lzss(&[0x00, 0x00, 0x00], 3), b"   ");
    }

    #[test]
    fn an_lzss_payload_is_checked_by_its_header() {
        let mut p = LZSS_MAGIC.to_vec();
        p.extend_from_slice(&adler32(b"abcabcabc").to_be_bytes());
        p.extend_from_slice(&9u32.to_be_bytes());
        p.extend_from_slice(&6u32.to_be_bytes());
        p.resize(LZSS_DATA, 0);
        p.extend_from_slice(&[0x07, b'a', b'b', b'c', 0xee, 0xf3]);
        assert_eq!(lzss_payload(&p).unwrap(), b"abcabcabc");
        p[8] ^= 1;
        assert!(lzss_payload(&p).is_err());
    }

    /// A Mach-O header of `filetype` with `cmds`.
    fn macho(filetype: u32, cmds: &[Vec<u8>]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&MH_MAGIC_64.to_le_bytes());
        b.extend_from_slice(&0x0100_000cu32.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&filetype.to_le_bytes());
        b.extend_from_slice(&(cmds.len() as u32).to_le_bytes());
        b.extend_from_slice(&(cmds.iter().map(Vec::len).sum::<usize>() as u32).to_le_bytes());
        b.extend_from_slice(&[0; 8]);
        for c in cmds {
            b.extend_from_slice(c);
        }
        b
    }

    fn name(s: &str) -> [u8; 16] {
        let mut n = [0u8; 16];
        n[..s.len()].copy_from_slice(s.as_bytes());
        n
    }

    /// A segment command of one section, both at file offset `off` for `len` bytes.
    fn segment(seg: &str, sect: &str, off: u64, len: u64) -> Vec<u8> {
        let mut c = Vec::new();
        c.extend_from_slice(&LC_SEGMENT_64.to_le_bytes());
        c.extend_from_slice(&((SEGMENT_COMMAND_64 + SECTION_64) as u32).to_le_bytes());
        c.extend_from_slice(&name(seg));
        c.extend_from_slice(&[0; 16]); // vmaddr, vmsize
        c.extend_from_slice(&off.to_le_bytes());
        c.extend_from_slice(&len.to_le_bytes());
        c.extend_from_slice(&[0; 8]); // maxprot, initprot
        c.extend_from_slice(&1u32.to_le_bytes());
        c.extend_from_slice(&0u32.to_le_bytes());
        c.extend_from_slice(&name(sect));
        c.extend_from_slice(&name(seg));
        c.extend_from_slice(&[0; 8]); // addr
        c.extend_from_slice(&len.to_le_bytes());
        c.extend_from_slice(&(off as u32).to_le_bytes());
        c.extend_from_slice(&[0; 28]);
        c
    }

    fn entry(id: &str, fileoff: u64) -> Vec<u8> {
        let mut c = Vec::new();
        let len = (32 + id.len() + 1).div_ceil(8) * 8;
        c.extend_from_slice(&LC_FILESET_ENTRY.to_le_bytes());
        c.extend_from_slice(&(len as u32).to_le_bytes());
        c.extend_from_slice(&0u64.to_le_bytes());
        c.extend_from_slice(&fileoff.to_le_bytes());
        c.extend_from_slice(&32u32.to_le_bytes());
        c.extend_from_slice(&0u32.to_le_bytes());
        c.extend_from_slice(id.as_bytes());
        c.resize(len, 0);
        c
    }

    /// A file set of `kexts` (identifier, its `__TEXT,__const` bytes), written here: the set's
    /// header and entries, then each kext's header and its constants.
    pub(crate) fn fileset(kexts: &[(&str, &[u8])]) -> Vec<u8> {
        let head_len = macho(
            MH_FILESET,
            &kexts.iter().map(|(id, _)| entry(id, 0)).collect::<Vec<_>>(),
        )
        .len();
        let kext_head = macho(0xb, &[segment("__TEXT", "__const", 0, 0)]).len();
        let mut at = head_len as u64;
        let mut entries = Vec::new();
        let mut bodies = Vec::new();
        for (id, consts) in kexts {
            entries.push(entry(id, at));
            let data_at = at + kext_head as u64;
            let mut k = macho(
                0xb,
                &[segment("__TEXT", "__const", data_at, consts.len() as u64)],
            );
            k.extend_from_slice(consts);
            at += k.len() as u64;
            bodies.push(k);
        }
        let mut b = macho(MH_FILESET, &entries);
        for k in bodies {
            b.extend(k);
        }
        b
    }

    #[test]
    fn a_file_sets_kexts_and_sections_are_found() {
        let b = fileset(&[
            ("com.example.a", b"first kext's constants"),
            ("com.example.b", b"second"),
        ]);
        let fs = FileSet::parse(&b).unwrap();
        assert_eq!(
            fs.ids().collect::<Vec<_>>(),
            ["com.example.a", "com.example.b"]
        );
        assert_eq!(
            fs.section("com.example.b", "__TEXT", "__const").unwrap(),
            b"second"
        );
        assert_eq!(
            fs.segments("com.example.a").unwrap(),
            [("__TEXT".to_owned(), &b"first kext's constants"[..])]
        );
        assert!(fs.section("com.example.c", "__TEXT", "__const").is_err());
        assert!(fs.section("com.example.a", "__TEXT", "__cstring").is_err());
        assert!(FileSet::parse(&b[..40]).is_err());
    }
}
