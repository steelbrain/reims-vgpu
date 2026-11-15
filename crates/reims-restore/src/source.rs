//! Random access to bytes: a file, a range of one, or an image inside another, read at offsets
//! from several places at once.

use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::sync::Arc;

/// Bytes read at offsets, from any thread.
pub trait ReadAt: Send + Sync {
    /// Reads up to `buf.len()` bytes at `offset`; returns how many, 0 only at or past the end.
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize>;

    /// The number of bytes.
    fn len(&self) -> u64;

    /// Whether there are no bytes.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Fills `buf` from `offset`, or fails with `UnexpectedEof`.
    fn read_exact_at(&self, mut offset: u64, mut buf: &mut [u8]) -> io::Result<()> {
        while !buf.is_empty() {
            let n = self.read_at(offset, buf)?;
            if n == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("short read at {offset} of {} bytes", self.len()),
                ));
            }
            offset += n as u64;
            buf = &mut buf[n..];
        }
        Ok(())
    }
}

impl<T: ReadAt + ?Sized> ReadAt for Arc<T> {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        (**self).read_at(offset, buf)
    }
    fn len(&self) -> u64 {
        (**self).len()
    }
}

impl<T: ReadAt + ?Sized> ReadAt for Box<T> {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        (**self).read_at(offset, buf)
    }
    fn len(&self) -> u64 {
        (**self).len()
    }
}

impl ReadAt for [u8] {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        let Ok(start) = usize::try_from(offset) else {
            return Ok(0);
        };
        if start >= <[u8]>::len(self) {
            return Ok(0);
        }
        let n = buf.len().min(<[u8]>::len(self) - start);
        buf[..n].copy_from_slice(&self[start..start + n]);
        Ok(n)
    }
    fn len(&self) -> u64 {
        <[u8]>::len(self) as u64
    }
}

impl ReadAt for Vec<u8> {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        self.as_slice().read_at(offset, buf)
    }
    fn len(&self) -> u64 {
        <[u8]>::len(self) as u64
    }
}

/// A range of a file: `len` bytes from `start`.
pub struct FileRange {
    file: Arc<File>,
    start: u64,
    len: u64,
}

impl FileRange {
    /// The range `start..start + len` of `file`.
    pub fn new(file: Arc<File>, start: u64, len: u64) -> Self {
        Self { file, start, len }
    }

    /// The whole of `file`.
    pub fn whole(file: File) -> io::Result<Self> {
        let len = file.metadata()?.len();
        Ok(Self::new(Arc::new(file), 0, len))
    }
}

impl ReadAt for FileRange {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        if offset >= self.len {
            return Ok(0);
        }
        let n = (buf.len() as u64).min(self.len - offset) as usize;
        self.file.read_at(&mut buf[..n], self.start + offset)
    }
    fn len(&self) -> u64 {
        self.len
    }
}

/// A range of another source.
pub struct SubRange<S> {
    inner: S,
    start: u64,
    len: u64,
}

impl<S: ReadAt> SubRange<S> {
    /// `len` bytes of `inner` from `start`; fails if they don't lie within it.
    pub fn new(inner: S, start: u64, len: u64) -> io::Result<Self> {
        if start.checked_add(len).is_none_or(|end| end > inner.len()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("range {start}+{len} lies outside {} bytes", inner.len()),
            ));
        }
        Ok(Self { inner, start, len })
    }
}

impl<S: ReadAt> ReadAt for SubRange<S> {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        if offset >= self.len {
            return Ok(0);
        }
        let n = (buf.len() as u64).min(self.len - offset) as usize;
        self.inner.read_at(self.start + offset, &mut buf[..n])
    }
    fn len(&self) -> u64 {
        self.len
    }
}

/// A `Read` + `Seek` cursor over a source.
pub struct Cursor<S> {
    inner: S,
    pos: u64,
}

impl<S: ReadAt> Cursor<S> {
    /// A cursor at the start of `inner`.
    pub fn new(inner: S) -> Self {
        Self { inner, pos: 0 }
    }
}

impl<S: ReadAt> io::Read for Cursor<S> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read_at(self.pos, buf)?;
        self.pos += n as u64;
        Ok(n)
    }
}

impl<S: ReadAt> io::Seek for Cursor<S> {
    fn seek(&mut self, pos: io::SeekFrom) -> io::Result<u64> {
        let new = match pos {
            io::SeekFrom::Start(p) => Some(p),
            io::SeekFrom::End(d) => self.inner.len().checked_add_signed(d),
            io::SeekFrom::Current(d) => self.pos.checked_add_signed(d),
        };
        self.pos = new
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "seek before the start"))?;
        Ok(self.pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Seek, SeekFrom};

    #[test]
    fn sub_range_reads_only_its_bytes() {
        let data: Vec<u8> = (0..100).collect();
        let r = SubRange::new(data, 10, 20).unwrap();
        let mut buf = [0u8; 30];
        assert_eq!(r.read_at(15, &mut buf).unwrap(), 5);
        assert_eq!(&buf[..5], &[25, 26, 27, 28, 29]);
        assert_eq!(r.read_at(20, &mut buf).unwrap(), 0);
    }

    #[test]
    fn sub_range_outside_its_source_is_refused() {
        let data = vec![0u8; 10];
        assert!(SubRange::new(data, 5, 6).is_err());
    }

    #[test]
    fn cursor_reads_and_seeks() {
        let data: Vec<u8> = (0..10).collect();
        let mut c = Cursor::new(data);
        c.seek(SeekFrom::End(-3)).unwrap();
        let mut v = Vec::new();
        c.read_to_end(&mut v).unwrap();
        assert_eq!(v, [7, 8, 9]);
        assert!(c.seek(SeekFrom::Current(-11)).is_err());
    }
}
