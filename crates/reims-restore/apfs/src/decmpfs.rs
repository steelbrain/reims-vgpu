//! Transparent file compression (decmpfs): a compressed file's content, decoded from its
//! `com.apple.decmpfs` attribute and, where the type keeps it there, its resource fork.
//!
//! The attribute starts with a 16-byte header: `fpmc`, the compression type (little-endian
//! 32-bit) and the content's size (little-endian 64-bit). The content is cut into 64 KiB chunks,
//! each one stream of the type's encoding. Types 3 (zlib), 7 (LZVN), 9 (stored) and 11 (LZFSE)
//! keep their one chunk after the header; types 4 (zlib), 8 (LZVN), 10 (stored) and 12 (LZFSE)
//! keep their chunks in the resource fork. Types 8, 10 and 12 open the fork with a little-endian
//! 32-bit offset per chunk and one for the end; type 4's fork is a classic resource fork whose
//! one resource is a chunk count and each chunk's offset and length. A zlib chunk opening with
//! 0xFF, an LZVN chunk opening with 0x06 and a stored chunk opening with 0xCC hold their content
//! raw after that byte. (The layouts as ferrosys, MIT OR Apache-2.0, records them.)

use crate::{Error, Result};

pub const MAGIC: &[u8; 4] = b"fpmc";
pub const HEADER_LEN: usize = 16;
pub const CHUNK: usize = 64 * 1024;

/// The attribute that holds a compressed file's header.
pub const XATTR: &[u8] = b"com.apple.decmpfs";
/// The attribute that holds a compressed file's resource fork.
pub const RESOURCE_FORK: &[u8] = b"com.apple.ResourceFork";
/// The BSD flag a compressed file carries.
pub const UF_COMPRESSED: u32 = 0x20;

/// A compression header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub kind: u32,
    pub size: u64,
}

impl Header {
    /// Parse the header at the start of a `com.apple.decmpfs` attribute.
    #[must_use]
    pub fn parse(attr: &[u8]) -> Option<Self> {
        (attr.len() >= HEADER_LEN && &attr[..4] == MAGIC).then(|| Self {
            kind: u32::from_le_bytes(attr[4..8].try_into().unwrap()),
            size: u64::from_le_bytes(attr[8..16].try_into().unwrap()),
        })
    }

    /// Whether the type keeps its content in the resource fork.
    #[must_use]
    pub const fn in_resource_fork(&self) -> bool {
        matches!(self.kind, 4 | 8 | 10 | 12 | 14)
    }
}

#[derive(Clone, Copy)]
enum Enc {
    Zlib,
    Lzvn,
    Stored,
    Lzfse,
}

fn encoding(kind: u32) -> Result<Enc> {
    Ok(match kind {
        3 | 4 => Enc::Zlib,
        7 | 8 => Enc::Lzvn,
        9 | 10 => Enc::Stored,
        11 | 12 => Enc::Lzfse,
        k => return Err(Error::Unsupported(format!("decmpfs compression type {k}"))),
    })
}

fn bad(what: &str) -> Error {
    Error::Invalid(format!("compressed file: {what}"))
}

/// Decode one chunk of `enc` into `out`, which is exactly the chunk's share of the content.
fn chunk(enc: Enc, input: &[u8], out: &mut [u8]) -> Result<()> {
    let raw =
        |marker: u8| -> Option<&[u8]> { (input.first() == Some(&marker)).then(|| &input[1..]) };
    let n = match enc {
        Enc::Stored => {
            let r = raw(0xcc).ok_or_else(|| bad("stored chunk without its marker"))?;
            let n = r.len().min(out.len());
            out[..n].copy_from_slice(&r[..n]);
            n
        }
        Enc::Zlib => {
            if let Some(r) = raw(0xff) {
                let n = r.len().min(out.len());
                out[..n].copy_from_slice(&r[..n]);
                n
            } else {
                use std::io::Read;
                let mut d = flate2::read::ZlibDecoder::new(input);
                let mut n = 0;
                while n < out.len() {
                    let k = d
                        .read(&mut out[n..])
                        .map_err(|e| bad(&format!("zlib: {e}")))?;
                    if k == 0 {
                        break;
                    }
                    n += k;
                }
                n
            }
        }
        Enc::Lzvn => {
            if let Some(r) = raw(0x06) {
                let n = r.len().min(out.len());
                out[..n].copy_from_slice(&r[..n]);
                n
            } else {
                ferrosys_lzfse::decompress_lzvn(input, out)
                    .map_err(|e| bad(&format!("LZVN: {e}")))?
            }
        }
        Enc::Lzfse => {
            ferrosys_lzfse::decompress_lzfse(input, out).map_err(|e| bad(&format!("LZFSE: {e}")))?
        }
    };
    if n != out.len() {
        return Err(bad(&format!(
            "a chunk decoded to {n} bytes, {} expected",
            out.len()
        )));
    }
    Ok(())
}

/// The content of a compressed file, from its `com.apple.decmpfs` attribute and, for the types
/// that keep it there, its resource fork.
///
/// # Errors
///
/// The header is malformed, the type is not one this crate decodes, or a chunk does not decode
/// to its share of the content.
pub fn decode(attr: &[u8], fork: Option<&[u8]>) -> Result<Vec<u8>> {
    let h = Header::parse(attr).ok_or_else(|| bad("no fpmc header"))?;
    let enc = encoding(h.kind)?;
    let size = usize::try_from(h.size).map_err(|_| bad("size"))?;
    if !h.in_resource_fork() {
        if size > CHUNK {
            return Err(bad("an inline chunk of more than 64 KiB"));
        }
        let mut out = vec![0u8; size];
        if size > 0 {
            chunk(enc, &attr[HEADER_LEN..], &mut out)?;
        }
        return Ok(out);
    }
    let fork = fork.ok_or_else(|| bad("no resource fork"))?;
    let chunks = size.div_ceil(CHUNK);
    let ranges: Vec<(usize, usize)> = if h.kind == 4 {
        // A classic resource fork: the data offset (big-endian) at 0; there a length, then the
        // resource: a chunk count and (offset, length) pairs, little-endian, from its start.
        let be = |o: usize| -> Result<usize> {
            fork.get(o..o + 4)
                .map(|b| u32::from_be_bytes(b.try_into().unwrap()) as usize)
                .ok_or_else(|| bad("short resource fork"))
        };
        let le = |o: usize| -> Result<usize> {
            fork.get(o..o + 4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
                .ok_or_else(|| bad("short resource fork"))
        };
        let res = be(0)? + 4;
        let n = le(res)?;
        if n != chunks {
            return Err(bad(&format!("{n} chunks for {size} bytes")));
        }
        (0..n)
            .map(|i| Ok((res + le(res + 4 + i * 8)?, le(res + 8 + i * 8)?)))
            .map(|r: Result<(usize, usize)>| r.map(|(o, l)| (o, o + l)))
            .collect::<Result<_>>()?
    } else {
        let word = |i: usize| -> Result<usize> {
            fork.get(i * 4..i * 4 + 4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
                .ok_or_else(|| bad("short chunk table"))
        };
        (0..chunks)
            .map(|i| Ok((word(i)?, word(i + 1)?)))
            .collect::<Result<_>>()?
    };
    // Allocated once the fork is known to hold a range for every chunk, so a size no fork
    // backs is refused before it is asked for.
    let mut out = vec![0u8; size];
    for (i, (s, e)) in ranges.into_iter().enumerate() {
        let input = fork
            .get(s..e)
            .filter(|_| s <= e)
            .ok_or_else(|| bad("a chunk outside the resource fork"))?;
        let at = i * CHUNK;
        chunk(enc, input, &mut out[at..(at + CHUNK).min(size)])?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(kind: u32, size: u64) -> Vec<u8> {
        let mut h = MAGIC.to_vec();
        h.extend_from_slice(&kind.to_le_bytes());
        h.extend_from_slice(&size.to_le_bytes());
        h
    }

    #[test]
    fn stored_inline_and_in_the_fork_decode() {
        let mut a = header(9, 5);
        a.push(0xcc);
        a.extend_from_slice(b"hello");
        assert_eq!(decode(&a, None).unwrap(), b"hello");
        // Two chunks in the fork, each stored.
        let size = CHUNK + 3;
        let c0: Vec<u8> = std::iter::once(0xcc)
            .chain((0..CHUNK).map(|i| i as u8))
            .collect();
        let c1 = vec![0xcc, 7, 8, 9];
        let table = 12u32;
        let mut fork = Vec::new();
        for o in [
            table,
            table + c0.len() as u32,
            table + (c0.len() + c1.len()) as u32,
        ] {
            fork.extend_from_slice(&o.to_le_bytes());
        }
        fork.extend_from_slice(&c0);
        fork.extend_from_slice(&c1);
        let out = decode(&header(10, size as u64), Some(&fork)).unwrap();
        assert_eq!(out.len(), size);
        assert_eq!(&out[CHUNK..], &[7, 8, 9]);
        assert_eq!(out[1], 1);
    }

    #[test]
    fn zlib_and_raw_lzvn_chunks_decode() {
        use std::io::Write;
        let content = b"the quick brown fox jumps over the lazy dog".repeat(20);
        let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        z.write_all(&content).unwrap();
        let mut a = header(3, content.len() as u64);
        a.extend_from_slice(&z.finish().unwrap());
        assert_eq!(decode(&a, None).unwrap(), content);
        let mut b = header(7, 3);
        b.extend_from_slice(&[0x06, 1, 2, 3]);
        assert_eq!(decode(&b, None).unwrap(), [1, 2, 3]);
        assert!(decode(&header(99, 1), None).is_err());
    }
}
