//! Windows ownership for QEMU's CRT RAM file descriptors. Handles are never narrowed to i32.
use std::ffi::c_void;
use std::fs::File;
use std::io;
use std::num::NonZeroU64;
use std::os::windows::io::FromRawHandle;

type InvalidParameter =
    Option<unsafe extern "C" fn(*const u16, *const u16, *const u16, u32, usize)>;
unsafe extern "C" {
    fn _get_osfhandle(fd: i32) -> isize;
    fn _set_thread_local_invalid_parameter_handler(handler: InvalidParameter) -> InvalidParameter;
}
#[repr(C)]
struct SystemInfo {
    architecture: u16,
    reserved: u16,
    page_size: u32,
    minimum_address: usize,
    maximum_address: usize,
    processor_mask: usize,
    processor_count: u32,
    processor_type: u32,
    allocation_granularity: u32,
    processor_level: u16,
    processor_revision: u16,
}
#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetSystemInfo(info: *mut SystemInfo);
    fn GetFileSizeEx(file: *mut c_void, size: *mut i64) -> i32;
    fn DuplicateHandle(
        source_process: *mut c_void,
        source: *mut c_void,
        target_process: *mut c_void,
        target: *mut *mut c_void,
        access: u32,
        inherit: i32,
        options: u32,
    ) -> i32;
}

/// Let the one descriptor lookup return EBADF, as the C API promises for an invalid fd.
/// This handler is thread-local, installed only for that call, and never masks other errors.
unsafe extern "C" fn invalid_fd(_: *const u16, _: *const u16, _: *const u16, _: u32, _: usize) {}

fn borrowed_handle(fd: i32) -> io::Result<*mut c_void> {
    // SAFETY: the thread's handler is restored before returning; lookup borrows, never closes.
    let raw = unsafe {
        let old = _set_thread_local_invalid_parameter_handler(Some(invalid_fd));
        let raw = _get_osfhandle(fd);
        _set_thread_local_invalid_parameter_handler(old);
        raw
    };
    if raw == -1 || raw == -2 {
        return Err(io::Error::from_raw_os_error(6));
    }
    Ok(raw as *mut c_void)
}

/// Windows x86-64 pages are distinct from its larger allocation granularity.
pub(in crate::device) fn host_page_size() -> NonZeroU64 {
    let mut info = std::mem::MaybeUninit::<SystemInfo>::uninit();
    // SAFETY: GetSystemInfo always initializes this SDK-layout output structure.
    unsafe {
        GetSystemInfo(info.as_mut_ptr());
    }
    NonZeroU64::new(unsafe { info.assume_init() }.page_size as u64).unwrap_or_else(|| {
        // A zero OS page size cannot describe any valid RAM import. Use
        // core's fatal boundary rather than unwinding through the C API.
        use reims_vgpu_core::raise::death::{DeathId, DeathKind, fatal};
        fatal(DeathId::Platform {
            kind: DeathKind::Abort,
            call: "GetSystemInfo",
        })
    })
}

/// Query the complete 64-bit extent, without giving Rust ownership of QEMU's handle.
pub(in crate::device) fn file_size(fd: i32) -> io::Result<u64> {
    let handle = borrowed_handle(fd)?;
    let mut size = 0;
    // SAFETY: the caller keeps its descriptor open; size is writable for the call.
    if unsafe { GetFileSizeEx(handle, &mut size) } == 0 {
        return Err(io::Error::last_os_error());
    }
    u64::try_from(size)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "negative RAM file extent"))
}

/// Retain a non-inheritable handle independently of the caller's CRT descriptor.
pub(in crate::device) fn duplicate(fd: i32) -> io::Result<File> {
    let handle = borrowed_handle(fd)?;
    let mut copy = std::ptr::null_mut();
    let process = -1isize as *mut c_void;
    // SAFETY: both process handles denote this process; DUPLICATE_SAME_ACCESS preserves rights.
    if unsafe { DuplicateHandle(process, handle, process, &mut copy, 0, 0, 2) } == 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: the new handle has one owner and is a duplicate of the caller's file handle.
    Ok(unsafe { File::from_raw_handle(copy) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::windows::io::{AsRawHandle, IntoRawHandle};
    unsafe extern "C" {
        fn _open_osfhandle(handle: isize, flags: i32) -> i32;
        fn _close(fd: i32) -> i32;
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn DeviceIoControl(
            file: *mut c_void,
            code: u32,
            input: *const c_void,
            input_size: u32,
            output: *mut c_void,
            output_size: u32,
            returned: *mut u32,
            overlapped: *mut c_void,
        ) -> i32;
    }

    #[test]
    fn duplicate_retains_a_sparse_file_larger_than_four_gibibytes() {
        let path =
            std::env::temp_dir().join(format!("reims-file-contract-{}.tmp", std::process::id()));
        let file = File::options()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        // A sparse extent tests the 64-bit ABI without writing several GiB of disk data.
        let mut returned = 0;
        // SAFETY: FSCTL_SET_SPARSE needs no input/output buffer and changes this test file only.
        assert_ne!(
            unsafe {
                DeviceIoControl(
                    file.as_raw_handle(),
                    0x900c4,
                    std::ptr::null(),
                    0,
                    std::ptr::null_mut(),
                    0,
                    &mut returned,
                    std::ptr::null_mut(),
                )
            },
            0
        );
        file.set_len(5 * 1024 * 1024 * 1024).unwrap();
        let fd = unsafe { _open_osfhandle(file.into_raw_handle() as isize, 0x8000) };
        assert!(fd >= 0);
        assert_eq!(file_size(fd).unwrap(), 5 * 1024 * 1024 * 1024);
        let retained = duplicate(fd).unwrap();
        assert_eq!(unsafe { _close(fd) }, 0);
        assert_eq!(retained.metadata().unwrap().len(), 5 * 1024 * 1024 * 1024);
        assert!(file_size(fd).is_err());
        assert!(duplicate(i32::MAX).is_err());
        drop(retained);
        std::fs::remove_file(path).unwrap();
        assert_eq!(host_page_size().get(), 4096);
    }
}
