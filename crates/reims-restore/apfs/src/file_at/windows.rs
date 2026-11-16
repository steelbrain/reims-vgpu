//! [`FileAt`] on Windows: the methods of Unix's `FileExt` the crates call, over std's
//! `seek_read` and `seek_write`, each of which reads or writes once from an offset and leaves
//! the file's cursor after what it moved.

use std::fs::File;
use std::io;
use std::os::windows::fs::FileExt as _;

/// Unix's `FileExt`, on Windows.
pub trait FileAt {
    /// Reads into `buf` from `offset`, once: the bytes read, 0 at the end of the file.
    ///
    /// # Errors
    ///
    /// The read fails.
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize>;

    /// Writes `buf` at `offset`, once: the bytes written.
    ///
    /// # Errors
    ///
    /// The write fails.
    fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<usize>;

    /// Fills `buf` from `offset`.
    ///
    /// # Errors
    ///
    /// The file ends first (`UnexpectedEof`), or a read fails.
    fn read_exact_at(&self, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
        while !buf.is_empty() {
            match self.read_at(buf, offset) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "failed to fill whole buffer",
                    ));
                }
                Ok(n) => {
                    buf = &mut buf[n..];
                    offset += n as u64;
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// Writes all of `buf` at `offset`.
    ///
    /// # Errors
    ///
    /// A write fails, or writes nothing (`WriteZero`).
    fn write_all_at(&self, mut buf: &[u8], mut offset: u64) -> io::Result<()> {
        while !buf.is_empty() {
            match self.write_at(buf, offset) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "failed to write whole buffer",
                    ));
                }
                Ok(n) => {
                    buf = &buf[n..];
                    offset += n as u64;
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}

impl FileAt for File {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        self.seek_read(buf, offset)
    }

    fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<usize> {
        self.seek_write(buf, offset)
    }
}
