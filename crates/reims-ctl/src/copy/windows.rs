//! [`super`]'s file system calls on Windows: a block clone (FSCTL_DUPLICATE_EXTENTS_TO_FILE)
//! where the file system makes them, as ReFS does (a Dev Drive among it), and sparse files for
//! the holes of a full copy, which NTFS keeps.

use std::fs::{File, OpenOptions};
use std::io;
use std::os::windows::io::AsRawHandle;
use std::path::Path;
use windows_sys::Win32::System::IO::DeviceIoControl;

/// The errors of a refused clone between two volumes: ERROR_NOT_SAME_DEVICE.
pub(super) const APART: &[i32] = &[17];

/// The errors of a refused clone on a file system that makes none: ERROR_INVALID_FUNCTION and
/// ERROR_NOT_SUPPORTED.
pub(super) const UNSUPPORTED: &[i32] = &[1, 50];

/// A volume's flag for block clones (winnt.h).
const FILE_SUPPORTS_BLOCK_REFCOUNTING: u32 = 0x0800_0000;

/// The most one clone call takes: a gigabyte, a whole number of any volume's clusters.
const CLONE_CHUNK: u64 = 1 << 30;

/// Runs the file system control `code` on `f`, with `input` and room for `output`; the bytes
/// it answered.
fn control<I, O>(f: &File, code: u32, input: Option<&I>, output: &mut [O]) -> io::Result<usize> {
    let mut got = 0u32;
    // SAFETY: input and output are alive for the call, of the sizes given; the call is
    // synchronous on a handle opened without overlapped I/O.
    let ok = unsafe {
        DeviceIoControl(
            f.as_raw_handle(),
            code,
            input.map_or(std::ptr::null(), |i| (i as *const I).cast()),
            input.map_or(0, |_| std::mem::size_of::<I>() as u32),
            output.as_mut_ptr().cast(),
            std::mem::size_of_val(output) as u32,
            &mut got,
            std::ptr::null_mut(),
        )
    } != 0;
    if ok {
        Ok(got as usize)
    } else {
        Err(io::Error::last_os_error())
    }
}

/// `dst`, made empty and sparse, so the blocks a full copy leaves unwritten are holes.
pub(super) fn create(_from: &File, dst: &Path) -> io::Result<File> {
    let f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(dst)?;
    set_sparse(&f)?;
    Ok(f)
}

/// Makes `f` sparse: NTFS leaves the parts of a sparse file never written unallocated, where it
/// fills a file that is not with zeros, and tells only a sparse file's holes
/// ([`data_ranges`]).
pub(super) fn set_sparse(f: &File) -> io::Result<()> {
    use windows_sys::Win32::System::Ioctl::FSCTL_SET_SPARSE;
    control::<(), u8>(f, FSCTL_SET_SPARSE, None, &mut []).map(drop)
}

/// The volume `f` is on: its serial number and its flags.
fn volume(f: &File) -> io::Result<(u32, u32)> {
    use windows_sys::Win32::Storage::FileSystem::GetVolumeInformationByHandleW;
    let (mut serial, mut flags) = (0u32, 0u32);
    // SAFETY: f is open for the call; the buffers left out are null with a size of 0.
    let ok = unsafe {
        GetVolumeInformationByHandleW(
            f.as_raw_handle(),
            std::ptr::null_mut(),
            0,
            &mut serial,
            std::ptr::null_mut(),
            &mut flags,
            std::ptr::null_mut(),
            0,
        )
    } != 0;
    if ok {
        Ok((serial, flags))
    } else {
        Err(io::Error::last_os_error())
    }
}

/// The size of a cluster of the volume `p` is on.
fn cluster_size(p: &Path) -> io::Result<u64> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{GetDiskFreeSpaceW, GetVolumePathNameW};
    let wide: Vec<u16> = p.as_os_str().encode_wide().chain([0]).collect();
    let mut root = vec![0u16; 1024];
    // SAFETY: wide is NUL-terminated, and root holds its length in UTF-16 units.
    if unsafe { GetVolumePathNameW(wide.as_ptr(), root.as_mut_ptr(), root.len() as u32) } == 0 {
        return Err(io::Error::last_os_error());
    }
    let (mut sectors, mut bytes, mut free, mut total) = (0u32, 0u32, 0u32, 0u32);
    // SAFETY: root is a NUL-terminated volume path, and each count is a u32 the call fills.
    if unsafe {
        GetDiskFreeSpaceW(
            root.as_ptr(),
            &mut sectors,
            &mut bytes,
            &mut free,
            &mut total,
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    Ok(u64::from(sectors) * u64::from(bytes))
}

/// Makes `dst` a block clone of `src`, or returns why not: the two are on different volumes
/// (ERROR_NOT_SAME_DEVICE), the volume makes no clones (ERROR_NOT_SUPPORTED), or the clone's
/// own error. A clone is of whole clusters, the last one past the file's end, which the file's
/// length leaves out.
pub(super) fn clone(src: &Path, dst: &Path) -> io::Result<()> {
    use windows_sys::Win32::System::Ioctl::{
        DUPLICATE_EXTENTS_DATA, FSCTL_DUPLICATE_EXTENTS_TO_FILE,
    };
    let from = File::open(src)?;
    let to = create(&from, dst)?;
    let ((from_volume, flags), (to_volume, _)) = (volume(&from)?, volume(&to)?);
    if from_volume != to_volume {
        return Err(io::Error::from_raw_os_error(APART[0]));
    }
    if flags & FILE_SUPPORTS_BLOCK_REFCOUNTING == 0 {
        return Err(io::Error::from_raw_os_error(UNSUPPORTED[1]));
    }
    let cluster = cluster_size(src)?;
    let len = from.metadata()?.len();
    to.set_len(len)?;
    let mut at = 0;
    while at < len {
        let count = (len - at).min(CLONE_CHUNK).next_multiple_of(cluster);
        let data = DUPLICATE_EXTENTS_DATA {
            FileHandle: from.as_raw_handle(),
            SourceFileOffset: at as i64,
            TargetFileOffset: at as i64,
            ByteCount: count as i64,
        };
        control::<_, u8>(&to, FSCTL_DUPLICATE_EXTENTS_TO_FILE, Some(&data), &mut [])?;
        at += count;
    }
    Ok(())
}

/// The parts of `f`, `len` bytes long, that a full copy reads: all of it. NTFS's allocated ranges
/// (FSCTL_QUERY_ALLOCATED_RANGES) left out data just written: a test's sparse source, asked
/// just after it was written, came back without the range written first, so its copy was zeros
/// there (job m3-max-32291 on the mini, 2026-11-15; job 32290 of the same code got the range).
/// [`super::full`] leaves the blocks of zeros it reads holes, so a copy is as sparse either way.
pub(super) fn data_ranges(_f: &File, len: u64) -> Vec<(u64, u64)> {
    if len == 0 { Vec::new() } else { vec![(0, len)] }
}

/// The type of the file system `dir` is on: NTFS, ReFS, exFAT.
pub(super) fn file_system(dir: &Path) -> Option<String> {
    use std::os::windows::fs::OpenOptionsExt;
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_FLAG_BACKUP_SEMANTICS, GetVolumeInformationByHandleW,
    };
    // A directory opens with backup semantics, for no access but its attributes.
    let d = OpenOptions::new()
        .access_mode(0)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
        .open(dir)
        .ok()?;
    let mut name = [0u16; 64];
    // SAFETY: d is open for the call, name holds its length in UTF-16 units, and the other
    // buffers are left out.
    let ok = unsafe {
        GetVolumeInformationByHandleW(
            d.as_raw_handle(),
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            name.as_mut_ptr(),
            name.len() as u32,
        )
    } != 0;
    let len = name.iter().position(|&c| c == 0)?;
    ok.then(|| String::from_utf16_lossy(&name[..len]))
}
