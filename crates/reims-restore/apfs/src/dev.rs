//! Where a container's bytes come from: a [`Source`] read at offsets (a file, or a disk image
//! decoded on the fly), and the [`Partition`] of it a container occupies.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;

use crate::Result;
use crate::file_at::FileAt as _;

/// Bytes read at offsets, from any thread.
pub trait Source: Send + Sync {
    /// Fill `buf` from `off`.
    ///
    /// # Errors
    ///
    /// The range is past the end, or the read fails.
    fn read_at(&self, off: u64, buf: &mut [u8]) -> std::io::Result<()>;

    /// The source's length in bytes.
    fn len(&self) -> u64;

    /// Whether the source is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Source for File {
    fn read_at(&self, off: u64, buf: &mut [u8]) -> std::io::Result<()> {
        self.read_exact_at(buf, off)
    }

    fn len(&self) -> u64 {
        self.metadata().map_or(0, |m| m.len())
    }
}

impl Source for Vec<u8> {
    fn read_at(&self, off: u64, buf: &mut [u8]) -> std::io::Result<()> {
        let s = usize::try_from(off)
            .ok()
            .filter(|s| s.checked_add(buf.len()).is_some_and(|e| e <= self.len()));
        let s = s.ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "read past the end")
        })?;
        buf.copy_from_slice(&self[s..s + buf.len()]);
        Ok(())
    }

    fn len(&self) -> u64 {
        self.len() as u64
    }
}

/// `len` bytes of a source from `base`: a container's partition.
#[derive(Clone)]
pub struct Partition {
    src: Arc<dyn Source>,
    base: u64,
    len: u64,
}

impl std::fmt::Debug for Partition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Partition")
            .field("base", &self.base)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl Partition {
    #[must_use]
    pub fn new(src: Arc<dyn Source>, base: u64, len: u64) -> Self {
        Self { src, base, len }
    }

    /// The whole of `src`.
    #[must_use]
    pub fn whole(src: Arc<dyn Source>) -> Self {
        let len = src.len();
        Self::new(src, 0, len)
    }

    #[must_use]
    pub const fn len(&self) -> u64 {
        self.len
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[must_use]
    pub const fn base(&self) -> u64 {
        self.base
    }

    /// Read `buf.len()` bytes at `off`, relative to the partition.
    ///
    /// # Errors
    ///
    /// The range is past the partition, or the read fails.
    pub fn read_at(&self, off: u64, buf: &mut [u8]) -> Result<()> {
        if off
            .checked_add(buf.len() as u64)
            .is_none_or(|end| end > self.len)
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "bytes {off}..+{} are past the partition's {} bytes",
                    buf.len(),
                    self.len
                ),
            )
            .into());
        }
        self.src.read_at(self.base + off, buf)?;
        Ok(())
    }
}

/// A [`Source`] as `Read + Seek`, for readers that want a stream (the GPT's).
pub struct Cursor<'a> {
    src: &'a dyn Source,
    pos: u64,
}

impl<'a> Cursor<'a> {
    #[must_use]
    pub fn new(src: &'a dyn Source) -> Self {
        Self { src, pos: 0 }
    }
}

impl Read for Cursor<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = (self.src.len().saturating_sub(self.pos)).min(buf.len() as u64) as usize;
        self.src.read_at(self.pos, &mut buf[..n])?;
        self.pos += n as u64;
        Ok(n)
    }
}

impl Seek for Cursor<'_> {
    fn seek(&mut self, to: SeekFrom) -> std::io::Result<u64> {
        let end = i128::from(self.src.len());
        let at = match to {
            SeekFrom::Start(x) => i128::from(x),
            SeekFrom::End(x) => end + i128::from(x),
            SeekFrom::Current(x) => i128::from(self.pos) + i128::from(x),
        };
        self.pos = u64::try_from(at).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "seek before the start")
        })?;
        Ok(self.pos)
    }
}
