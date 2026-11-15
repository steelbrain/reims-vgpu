//! The hash a directory entry is keyed by: 22 bits of a CRC-32C over the name's normalized code
//! points.
//!
//! A name is normalized to its canonical decomposition, canonically ordered; on a volume that
//! folds case, it is then case-folded (full folding, statuses C and F of Unicode's
//! `CaseFolding.txt`) and decomposed again. The hash is the CRC-32C of those code points as
//! little-endian 32-bit words, with no terminator, complemented, in its low 22 bits. These are
//! the rules ferrosys (MIT OR Apache-2.0) records as observed on volumes macOS 15, 26 and 27
//! wrote; the Apple File System Reference's description hashes the terminator, which macOS does
//! not. The tables are Unicode's, from ferrosys-ucd.

use ferrosys_ucd::{CASE_FOLDING, COMBINING_CLASS, DECOMPOSITION};

const CRC32C: crc::Crc<u32> = crc::Crc::<u32>::new(&crc::CRC_32_ISCSI);

/// The bits a directory entry's hash takes.
pub const HASH_MASK: u32 = 0x3f_ffff;

const S_BASE: u32 = 0xAC00;
const L_BASE: u32 = 0x1100;
const V_BASE: u32 = 0x1161;
const T_BASE: u32 = 0x11A7;
const T_COUNT: u32 = 28;
const N_COUNT: u32 = 21 * T_COUNT;
const S_COUNT: u32 = 19 * N_COUNT;

/// The normalized form of `name`, or `None` where it is not UTF-8.
#[must_use]
pub fn normalize(name: &[u8], fold: bool) -> Option<Vec<u32>> {
    let text = std::str::from_utf8(name).ok()?;
    let d = decompose(text.chars().map(u32::from));
    if !fold {
        return Some(d);
    }
    let folded = d.into_iter().flat_map(|cp| {
        CASE_FOLDING
            .binary_search_by_key(&cp, |(c, _)| *c)
            .map_or_else(|_| vec![cp], |i| CASE_FOLDING[i].1.to_vec())
    });
    Some(decompose(folded))
}

/// The 22-bit hash a directory entry for `name` is keyed by, on a volume that folds case
/// (`fold`) or keeps it. A name that is not UTF-8 hashes its bytes as they are.
#[must_use]
pub fn hash(name: &[u8], fold: bool) -> u32 {
    let bytes: Vec<u8> = match normalize(name, fold) {
        Some(form) => form.iter().flat_map(|cp| cp.to_le_bytes()).collect(),
        None => name.to_vec(),
    };
    !CRC32C.checksum(&bytes) & HASH_MASK
}

/// Whether two names are one name on a volume that folds case (`fold`) or keeps it.
#[must_use]
pub fn same(a: &[u8], b: &[u8], fold: bool) -> bool {
    match (normalize(a, fold), normalize(b, fold)) {
        (Some(x), Some(y)) => x == y,
        _ => a == b,
    }
}

fn decompose(code_points: impl Iterator<Item = u32>) -> Vec<u32> {
    let mut out = Vec::new();
    for cp in code_points {
        if (S_BASE..S_BASE + S_COUNT).contains(&cp) {
            let s = cp - S_BASE;
            out.push(L_BASE + s / N_COUNT);
            out.push(V_BASE + (s % N_COUNT) / T_COUNT);
            if !s.is_multiple_of(T_COUNT) {
                out.push(T_BASE + s % T_COUNT);
            }
        } else if let Ok(i) = DECOMPOSITION.binary_search_by_key(&cp, |(c, _)| *c) {
            out.extend_from_slice(DECOMPOSITION[i].1);
        } else {
            out.push(cp);
        }
    }
    // Canonical ordering: each run of nonzero combining classes, stably sorted by class.
    let mut i = 0;
    while i < out.len() {
        if class(out[i]) == 0 {
            i += 1;
            continue;
        }
        let start = i;
        while i < out.len() && class(out[i]) != 0 {
            i += 1;
        }
        out[start..i].sort_by_key(|cp| class(*cp));
    }
    out
}

fn class(cp: u32) -> u8 {
    COMBINING_CLASS
        .binary_search_by_key(&cp, |(c, _)| *c)
        .map_or(0, |i| COMBINING_CLASS[i].1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composed_and_decomposed_names_are_one_and_hash_alike() {
        for fold in [false, true] {
            assert!(same("caf\u{e9}".as_bytes(), "cafe\u{301}".as_bytes(), fold));
            assert_eq!(
                hash("caf\u{e9}".as_bytes(), fold),
                hash("cafe\u{301}".as_bytes(), fold)
            );
            assert!(same(
                "\u{d55c}".as_bytes(),
                "\u{1112}\u{1161}\u{11ab}".as_bytes(),
                fold
            ));
        }
        assert!(same(b"System", b"SYSTEM", true));
        assert!(!same(b"System", b"SYSTEM", false));
        assert_eq!(hash(b"Library", true), hash(b"LIBRARY", true));
        assert_ne!(hash(b"Library", false), hash(b"LIBRARY", false));
        assert!(hash(b"x", true) <= HASH_MASK);
    }
}
