//! QuickTime-style atoms: a big-endian u32 size that counts the 8-byte
//! header, a big-endian type, then the payload. Messages, parameters,
//! sample buffer serializations and image description extensions are all
//! built from them.

use crate::core::fourcc::FourCc;
use crate::paravirt::WireError;

/// The size of an atom's header: its size and its type.
pub const ATOM_HEADER_SIZE: usize = 8;

/// One atom found in a byte range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Atom<'a> {
    /// The atom's type.
    pub kind: FourCc,
    /// Where the atom starts, relative to the range it was found in.
    pub offset: usize,
    /// The whole atom, header included.
    pub bytes: &'a [u8],
}

impl<'a> Atom<'a> {
    /// What follows the header.
    pub fn payload(&self) -> &'a [u8] {
        self.bytes.get(ATOM_HEADER_SIZE..).unwrap_or_default()
    }
}

/// The atoms that follow each other in `bytes`, each read as a sibling.
///
/// A size under 8 or past the end ends the walk with an error, as
/// FigAtomStream's walk does; a tail shorter than a header ends it quietly.
pub fn siblings(bytes: &[u8]) -> Siblings<'_> {
    Siblings {
        bytes,
        offset: 0,
        failed: false,
    }
}

/// The iterator [`siblings`] returns.
#[derive(Clone, Debug)]
pub struct Siblings<'a> {
    bytes: &'a [u8],
    offset: usize,
    failed: bool,
}

impl<'a> Iterator for Siblings<'a> {
    type Item = Result<Atom<'a>, WireError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }
        let rest = self.bytes.get(self.offset..)?;
        if rest.len() < ATOM_HEADER_SIZE {
            return None;
        }
        let size = usize::try_from(read_u32(rest, 0)?).ok()?;
        if size < ATOM_HEADER_SIZE || size > rest.len() {
            self.failed = true;
            return Some(Err(WireError::Malformed("atom size")));
        }
        let atom = Atom {
            kind: FourCc(read_u32(rest, 4)?),
            offset: self.offset,
            bytes: rest.get(..size)?,
        };
        self.offset += size;
        Some(Ok(atom))
    }
}

/// The big-endian u32 at `offset`, if the bytes reach that far.
pub fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    let field = bytes.get(offset..offset.checked_add(4)?)?;
    Some(u32::from_be_bytes(field.try_into().ok()?))
}

/// The big-endian u64 at `offset`, if the bytes reach that far.
pub fn read_u64(bytes: &[u8], offset: usize) -> Option<u64> {
    let field = bytes.get(offset..offset.checked_add(8)?)?;
    Some(u64::from_be_bytes(field.try_into().ok()?))
}

/// The big-endian u16 at `offset`, if the bytes reach that far.
pub fn read_u16(bytes: &[u8], offset: usize) -> Option<u16> {
    let field = bytes.get(offset..offset.checked_add(2)?)?;
    Some(u16::from_be_bytes(field.try_into().ok()?))
}

/// Writes atoms into a growing buffer, nesting as FigAtomWriter does: each
/// atom's size is filled in when it ends.
#[derive(Debug, Default)]
pub struct AtomWriter {
    bytes: Vec<u8>,
    open: Vec<usize>,
}

impl AtomWriter {
    /// An empty writer.
    pub fn new() -> Self {
        Self::default()
    }

    /// A writer that appends to `bytes`.
    pub fn with_bytes(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            open: Vec::new(),
        }
    }

    /// Starts an atom of type `kind`; its payload is what is written until
    /// the matching [`end`](Self::end).
    pub fn begin(&mut self, kind: FourCc) -> &mut Self {
        self.open.push(self.bytes.len());
        self.bytes.extend_from_slice(&[0; 4]);
        self.bytes.extend_from_slice(&kind.to_be_bytes());
        self
    }

    /// Appends payload bytes to the open atom.
    pub fn append(&mut self, payload: &[u8]) -> &mut Self {
        self.bytes.extend_from_slice(payload);
        self
    }

    /// Ends the innermost open atom, writing its size.
    ///
    /// An atom larger than 4 GiB cannot be written; it is truncated to the
    /// u32 the header has room for, which no message here comes near (the
    /// guest's buffers are 0xa00000 bytes at most).
    pub fn end(&mut self) -> &mut Self {
        if let Some(start) = self.open.pop() {
            let size = u32::try_from(self.bytes.len() - start).unwrap_or(u32::MAX);
            if let Some(field) = self.bytes.get_mut(start..start + 4) {
                field.copy_from_slice(&size.to_be_bytes());
            }
        }
        self
    }

    /// A whole atom: `kind`, then `payload`.
    pub fn atom(&mut self, kind: FourCc, payload: &[u8]) -> &mut Self {
        self.begin(kind).append(payload).end()
    }

    /// The bytes so far, to patch a field already written.
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
    }

    /// The bytes so far.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether nothing is written yet.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// The bytes, with every atom still open ended.
    pub fn finish(mut self) -> Vec<u8> {
        while !self.open.is_empty() {
            self.end();
        }
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_atoms_get_their_sizes() {
        let mut writer = AtomWriter::new();
        writer.begin(FourCc::new(*b"outr"));
        writer.atom(FourCc::new(*b"innr"), &[1, 2, 3]);
        writer.end();
        let bytes = writer.finish();
        assert_eq!(
            bytes,
            [
                0, 0, 0, 19, b'o', b'u', b't', b'r', 0, 0, 0, 11, b'i', b'n', b'n', b'r', 1, 2, 3
            ]
        );
        let outer: Vec<_> = siblings(&bytes).collect();
        assert_eq!(outer.len(), 1);
        let Some(Ok(outer)) = outer.first().cloned() else {
            panic!("one atom");
        };
        let inner: Vec<_> = siblings(outer.payload()).collect();
        assert_eq!(
            inner,
            [Ok(Atom {
                kind: FourCc::new(*b"innr"),
                offset: 0,
                bytes: &bytes[8..19],
            })]
        );
    }

    #[test]
    fn an_atom_past_the_end_is_malformed() {
        let bytes = [0, 0, 0, 12, b'a', b'b', b'c', b'd', 0, 0];
        let found: Vec<_> = siblings(&bytes).collect();
        assert_eq!(found, [Err(WireError::Malformed("atom size"))]);
    }

    #[test]
    fn an_atom_smaller_than_its_header_is_malformed() {
        let bytes = [0, 0, 0, 4, b'a', b'b', b'c', b'd'];
        let found: Vec<_> = siblings(&bytes).collect();
        assert_eq!(found, [Err(WireError::Malformed("atom size"))]);
    }

    #[test]
    fn a_short_tail_ends_the_walk() {
        let bytes = [0, 0, 0, 8, b'a', b'b', b'c', b'd', 0, 0, 0];
        assert_eq!(siblings(&bytes).count(), 1);
    }
}
