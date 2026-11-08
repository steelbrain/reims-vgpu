//! Memory of the process's own that the GPU can write and the CPU reads
//! cached: a memfd, made a dma-buf by udmabuf, and mapped. The staged copy
//! back imports it as a VA surface and has the GPU copy each decoded frame
//! into it (`super::readback::HostSurface`).
//!
//! A decode surface lives in video memory, which the CPU reads uncached
//! over PCIe: `vaGetImage` of a decoded 4K frame on radeonsi ran at about
//! 600 MB/s. Pages of a memfd are ordinary cached system memory, which the
//! GPU writes over PCIe as fast as it reads its own.

use std::ffi::{c_char, c_int, c_uint, c_ulong, c_void};
use std::fs::{File, OpenOptions};
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd};

use super::display::Error;

unsafe extern "C" {
    fn memfd_create(name: *const c_char, flags: c_uint) -> c_int;
    fn fcntl(fd: c_int, command: c_int, ...) -> c_int;
    fn ioctl(fd: c_int, request: c_ulong, ...) -> c_int;
    fn mmap(
        address: *mut c_void,
        length: usize,
        protection: c_int,
        flags: c_int,
        fd: c_int,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(address: *mut c_void, length: usize) -> c_int;
}

/// `MFD_CLOEXEC` and `MFD_ALLOW_SEALING` (linux/memfd.h).
const MFD_CLOEXEC: c_uint = 0x1;
const MFD_ALLOW_SEALING: c_uint = 0x2;
/// `F_ADD_SEALS` and `F_SEAL_SHRINK` (linux/fcntl.h): udmabuf takes only a
/// memfd that cannot shrink under it.
const F_ADD_SEALS: c_int = 1024 + 9;
const F_SEAL_SHRINK: c_int = 0x0002;
/// `PROT_READ`, `PROT_WRITE`, `MAP_SHARED` and `MAP_FAILED`.
const PROT_READ: c_int = 0x1;
const PROT_WRITE: c_int = 0x2;
const MAP_SHARED: c_int = 0x01;
const MAP_FAILED: *mut c_void = usize::MAX as *mut c_void;
/// `UDMABUF_CREATE` (linux/udmabuf.h): `_IOW('u', 0x42, struct
/// udmabuf_create)`, the struct being 24 bytes.
const UDMABUF_CREATE: c_ulong = (1 << 30) | (24 << 16) | ((b'u' as c_ulong) << 8) | 0x42;
/// `UDMABUF_FLAGS_CLOEXEC`.
const UDMABUF_FLAGS_CLOEXEC: u32 = 0x01;
const PAGE: usize = 4096;

/// `struct udmabuf_create`.
#[repr(C)]
struct UdmabufCreate {
    memfd: u32,
    flags: u32,
    offset: u64,
    size: u64,
}

/// A memfd of the process's own, a dma-buf over it, and its mapping.
pub(crate) struct HostMemory {
    /// The dma-buf, which a VA import takes.
    dma_buf: OwnedFd,
    /// The memfd behind it, kept open as long as the mapping.
    memfd: File,
    map: *mut u8,
    len: usize,
}

// SAFETY: the mapping is the memfd's pages, owned by this value; it is only
// read through `bytes`, under the borrow of the value.
unsafe impl Send for HostMemory {}

fn io_error(what: &str) -> Error {
    Error::Unsupported(format!("{what}: {}", std::io::Error::last_os_error()))
}

impl HostMemory {
    /// `len` bytes, rounded up to whole pages.
    pub(crate) fn new(len: usize) -> Result<Self, Error> {
        let len = len
            .checked_next_multiple_of(PAGE)
            .ok_or_else(|| Error::Unsupported(format!("host memory of {len} bytes")))?;
        // SAFETY: a NUL-terminated name and valid flags.
        let fd = unsafe {
            memfd_create(
                c"reims-vgpu-video-readback".as_ptr(),
                MFD_CLOEXEC | MFD_ALLOW_SEALING,
            )
        };
        if fd < 0 {
            return Err(io_error("memfd_create"));
        }
        // SAFETY: a descriptor memfd_create just returned, owned from here.
        let memfd = File::from(unsafe { OwnedFd::from_raw_fd(fd) });
        memfd
            .set_len(u64::try_from(len).unwrap_or(u64::MAX))
            .map_err(|e| Error::Unsupported(format!("sizing the memfd: {e}")))?;
        // SAFETY: an open memfd that allows sealing.
        if unsafe { fcntl(memfd.as_raw_fd(), F_ADD_SEALS, F_SEAL_SHRINK) } < 0 {
            return Err(io_error("sealing the memfd"));
        }
        let dma_buf = udmabuf(&memfd, 0, len)?;
        // SAFETY: a shared mapping of the whole memfd, which is `len` bytes.
        let map = unsafe {
            mmap(
                core::ptr::null_mut(),
                len,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                memfd.as_raw_fd(),
                0,
            )
        };
        if map == MAP_FAILED {
            return Err(io_error("mapping the memfd"));
        }
        Ok(Self {
            dma_buf,
            memfd,
            map: map.cast(),
            len,
        })
    }

    /// The dma-buf's descriptor, for an import.
    pub(crate) fn dma_buf(&self) -> c_int {
        self.dma_buf.as_raw_fd()
    }

    /// The dma-buf, borrowed, for a wait on its fences.
    pub(crate) fn dma_buf_fd(&self) -> BorrowedFd<'_> {
        self.dma_buf.as_fd()
    }

    /// Its size in bytes.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Its bytes, as the GPU last wrote them. The caller has waited for the
    /// GPU's write (`vaSyncSurface` of the surface over it).
    pub(crate) fn bytes(&self) -> &[u8] {
        // SAFETY: `len` mapped bytes, alive until drop; the GPU's write is
        // over, so nothing writes them while they are borrowed.
        unsafe { core::slice::from_raw_parts(self.map, self.len) }
    }

    /// Its bytes, to write before the GPU does. The caller has no GPU work
    /// on them in flight.
    pub(crate) fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: `len` mapped bytes, alive until drop, borrowed mutably
        // with the value, and no GPU work writes them meanwhile.
        unsafe { core::slice::from_raw_parts_mut(self.map, self.len) }
    }

    /// Another dma-buf, of the `len` bytes from `offset`, whole pages of the
    /// memfd: as the host's dma-buf of some of a guest buffer's pages is.
    pub(crate) fn dma_buf_of_pages(&self, offset: usize, len: usize) -> Result<OwnedFd, Error> {
        if !offset.is_multiple_of(PAGE)
            || !len.is_multiple_of(PAGE)
            || offset.saturating_add(len) > self.len
        {
            return Err(Error::Unsupported(format!(
                "a dma-buf of {len} bytes at {offset} of {} bytes of host memory",
                self.len
            )));
        }
        udmabuf(&self.memfd, offset, len)
    }
}

/// A dma-buf of the `len` bytes of `memfd` from `offset`, whole pages, made
/// by udmabuf.
fn udmabuf(memfd: &File, offset: usize, len: usize) -> Result<OwnedFd, Error> {
    let device = OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/udmabuf")
        .map_err(|e| Error::Unsupported(format!("/dev/udmabuf: {e}")))?;
    let create = UdmabufCreate {
        memfd: u32::try_from(memfd.as_raw_fd()).unwrap_or(u32::MAX),
        flags: UDMABUF_FLAGS_CLOEXEC,
        offset: u64::try_from(offset).unwrap_or(u64::MAX),
        size: u64::try_from(len).unwrap_or(u64::MAX),
    };
    // SAFETY: the request's argument is a `struct udmabuf_create`, which
    // lives for the call.
    let dma_buf = unsafe { ioctl(device.as_raw_fd(), UDMABUF_CREATE, &raw const create) };
    if dma_buf < 0 {
        return Err(io_error("UDMABUF_CREATE"));
    }
    // SAFETY: a descriptor the ioctl just returned, owned from here.
    Ok(unsafe { OwnedFd::from_raw_fd(dma_buf) })
}

impl Drop for HostMemory {
    fn drop(&mut self) {
        // SAFETY: the mapping made in `new`, unmapped once.
        unsafe { munmap(self.map.cast(), self.len) };
    }
}
