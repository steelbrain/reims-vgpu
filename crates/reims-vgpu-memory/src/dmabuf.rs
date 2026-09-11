//! Linux shared-page export. Backing identity comes from HostOps, never /proc maps.
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

/// Borrowed backing of one page resolved by the host. The host keeps this fd
/// and the named guest mapping stable until the synchronous export completes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SharedPage {
    pub fd: i32,
    pub reserved: u32,
    pub offset: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExportError {
    NoBacking,
    InvalidRange,
    Open(i32),
    Export(i32),
    SizeMismatch,
}
impl reims_vgpu_observe::Decline for ExportError {
    fn slug(&self) -> &'static str {
        match self {
            Self::NoBacking => "dmabuf_no_shared_backing",
            Self::InvalidRange => "dmabuf_invalid_range",
            Self::Open(_) => "dmabuf_exporter_unavailable",
            Self::Export(_) => "dmabuf_export_failed",
            Self::SizeMismatch => "dmabuf_size_mismatch",
        }
    }
    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Open(e) | Self::Export(e) => vec![("errno", e.to_string())],
            _ => Vec::new(),
        }
    }
}

#[derive(Debug)]
pub struct SharedDmaBuf {
    fd: OwnedFd,
    len: u64,
}
impl SharedDmaBuf {
    pub fn fd(&self) -> &OwnedFd {
        &self.fd
    }
    pub fn len(&self) -> u64 {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// # Safety
    /// Every page must describe the corresponding guest page, not merely an
    /// arbitrary valid file range. Descriptors and backing must remain stable
    /// until this call returns. The kernel validates seals and pins the payload.
    pub unsafe fn export(pages: &[SharedPage], page: u64) -> Result<Self, ExportError> {
        let ranges = coalesce(pages, page)?;
        let len = (pages.len() as u64)
            .checked_mul(page)
            .ok_or(ExportError::InvalidRange)?;
        let path = c"/dev/udmabuf";
        let raw = unsafe { libc::open(path.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC) };
        if raw < 0 {
            return Err(ExportError::Open(errno()));
        }
        let exporter = unsafe { OwnedFd::from_raw_fd(raw) };
        // Linux x86_64 _IOW('u', 0x43, struct udmabuf_create_list).
        let mut list = vec![0u64; 1 + ranges.len() * 3];
        list[0] = 1 | ((ranges.len() as u64) << 32);
        for (i, (fd, offset, len)) in ranges.into_iter().enumerate() {
            list[1 + i * 3] = fd as u64;
            list[2 + i * 3] = offset;
            list[3 + i * 3] = len;
        }
        let raw = unsafe {
            libc::ioctl(
                exporter.as_raw_fd(),
                0x40087543u64 as libc::c_ulong,
                list.as_ptr(),
            )
        };
        if raw < 0 {
            return Err(ExportError::Export(errno()));
        }
        Ok(Self {
            fd: unsafe { OwnedFd::from_raw_fd(raw) },
            len,
        })
    }
    pub fn sync_cpu(&self, end: bool) -> Result<(), i32> {
        let flags = 3u64 | if end { 4 } else { 0 };
        loop {
            let r =
                unsafe { libc::ioctl(self.fd.as_raw_fd(), 0x40086200u64 as libc::c_ulong, &flags) };
            if r == 0 {
                return Ok(());
            }
            let e = errno();
            if e != libc::EINTR && e != libc::EAGAIN {
                return Err(e);
            }
        }
    }
}
fn errno() -> i32 {
    std::io::Error::last_os_error()
        .raw_os_error()
        .unwrap_or(libc::EIO)
}
fn coalesce(pages: &[SharedPage], page: u64) -> Result<Vec<(i32, u64, u64)>, ExportError> {
    if page == 0 || !page.is_power_of_two() || pages.is_empty() || pages.len() > u32::MAX as usize {
        return Err(ExportError::InvalidRange);
    }
    let host_page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if host_page <= 0 || !page.is_multiple_of(host_page as u64) {
        return Err(ExportError::InvalidRange);
    }
    let mut runs: Vec<(i32, u64, u64)> = Vec::new();
    for p in pages {
        if p.fd < 0
            || p.reserved != 0
            || !p.offset.is_multiple_of(page)
            || p.offset.checked_add(page).is_none()
        {
            return Err(ExportError::InvalidRange);
        }
        if let Some(last) = runs.last_mut() {
            if last.0 == p.fd && last.1.checked_add(last.2) == Some(p.offset) {
                last.2 = last.2.checked_add(page).ok_or(ExportError::InvalidRange)?;
                continue;
            }
        }
        runs.push((p.fd, p.offset, page));
    }
    Ok(runs)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn p(fd: i32, offset: u64) -> SharedPage {
        SharedPage {
            fd,
            offset,
            reserved: 0,
        }
    }
    #[test]
    fn preserves_order_gaps_and_file_identity() {
        assert_eq!(
            coalesce(&[p(4, 8192), p(4, 12288), p(4, 0), p(5, 4096)], 4096).unwrap(),
            vec![(4, 8192, 8192), (4, 0, 4096), (5, 4096, 4096)]
        );
    }
    #[test]
    fn rejects_bad_geometry_and_overflow() {
        for pages in [
            vec![],
            vec![p(-1, 0)],
            vec![p(1, 1)],
            vec![p(1, u64::MAX - 4095)],
        ] {
            assert_eq!(coalesce(&pages, 4096), Err(ExportError::InvalidRange));
        }
        assert_eq!(coalesce(&[p(1, 0)], 0), Err(ExportError::InvalidRange));
    }
}
