//! Copying an image into a VM's directory: a reflink (Linux's btrfs and XFS) or a clone (APFS),
//! which shares the source's blocks and takes no time whatever its size, where the source and the
//! copy are on one file system that makes them; else a full copy, which reads and writes every
//! byte. A long full copy says so before it starts, with how much it copies and why it is no
//! reflink, then how far it is, as reims-util's downloads do (reims_util.rs). Either way the copy
//! is the source's bytes, its holes kept, and a file of its own: a write to one leaves the other
//! as it was.
//!
//! Measured with copies of a 57.9 GB qcow2 between two directories of one btrfs subvolume (Linux
//! 7.3.1, 2026-11-08): a reflink took 119 ms where it had 11,477 extents, and 785 ms where it had
//! 110,874.

use crate::paths;
use crate::reims_util::size;
use anyhow::{Context, Result};
use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt};
use std::path::Path;
use std::time::{Duration, Instant};

/// A full copy of more data than this says so first, and then how far it is: at an NVMe drive's
/// 2 GB/s, a gigabyte takes half a second. An OpenCore image or OVMF, a few megabytes, copies
/// without a word.
const LONG: u64 = 1_000_000_000;

/// Copies `src` to `dst`, a file that does not exist yet, with `src`'s permissions: a reflink or
/// clone where the file system makes one, else a full copy, which `progress` hears of first where
/// it is long, with its size and why it is no reflink, and then how far it is.
pub fn file(src: &Path, dst: &Path, progress: &mut dyn FnMut(&str)) -> Result<()> {
    copy(src, dst, LONG, progress)
}

/// [`file`], where a full copy of more than `long` bytes of data is told.
fn copy(src: &Path, dst: &Path, long: u64, progress: &mut dyn FnMut(&str)) -> Result<()> {
    let fail = || format!("could not copy {} to {}", src.display(), dst.display());
    let refused = match clone(src, dst) {
        Ok(()) => return Ok(()),
        Err(e) => e,
    };
    let from = File::open(src).with_context(fail)?;
    let len = from.metadata().with_context(fail)?.len();
    let to = create(&from, dst).with_context(fail)?;
    let ranges = data_ranges(&from, len);
    let total: u64 = ranges.iter().map(|(a, b)| b - a).sum();
    if total <= long {
        return full(&from, &to, &ranges, len, &mut |_| {}).with_context(fail);
    }
    let why = Why::of(&refused, src, dst);
    let name = src
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    progress(&format!(
        "copying {name} in full, {}, not as a reflink: {}",
        size(total),
        why.note(src, dst)
    ));
    // Each whole percent, at most once a second but for the first and the last.
    let short = why.short();
    let mut shown: Option<(u64, Instant)> = None;
    full(&from, &to, &ranges, len, &mut |done| {
        let pct = done.min(total) * 100 / total;
        let due = match shown {
            None => true,
            Some((p, t)) => p != pct && (pct == 100 || t.elapsed() >= Duration::from_secs(1)),
        };
        if due {
            shown = Some((pct, Instant::now()));
            progress(&format!(
                "copying {name} in full ({short}): {pct}% of {}",
                size(total)
            ));
        }
    })
    .with_context(fail)
}

/// `dst`, made empty, with the permissions of `from`, as `cp` makes its copy.
fn create(from: &File, dst: &Path) -> io::Result<File> {
    OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(from.metadata()?.mode() & 0o7777)
        .open(dst)
}

/// Makes `dst` a reflink of `src`, or returns the file system's refusal.
#[cfg(target_os = "linux")]
fn clone(src: &Path, dst: &Path) -> io::Result<()> {
    let from = File::open(src)?;
    let to = create(&from, dst)?;
    // SAFETY: FICLONE takes the source's descriptor; both are open for the call.
    if unsafe { libc::ioctl(to.as_raw_fd(), libc::FICLONE, from.as_raw_fd()) } == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// Makes `dst` a clone of `src`, or returns the file system's refusal.
#[cfg(target_os = "macos")]
fn clone(src: &Path, dst: &Path) -> io::Result<()> {
    let (s, d) = (c_path(src)?, c_path(dst)?);
    // SAFETY: both are NUL-terminated paths, alive for the call.
    if unsafe { libc::clonefile(s.as_ptr(), d.as_ptr(), 0) } == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(target_os = "macos")]
fn c_path(p: &Path) -> io::Result<std::ffi::CString> {
    use std::os::unix::ffi::OsStrExt;
    std::ffi::CString::new(p.as_os_str().as_bytes()).map_err(io::Error::other)
}

/// The parts of `f`, `len` bytes long, that hold data, as `[start, end)`: all of it where the file
/// system tells no holes.
fn data_ranges(f: &File, len: u64) -> Vec<(u64, u64)> {
    let seek = |at: u64, whence| {
        // SAFETY: lseek on an open descriptor.
        let r = unsafe { libc::lseek(f.as_raw_fd(), at as libc::off_t, whence) };
        if r < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(r as u64)
        }
    };
    let mut ranges = Vec::new();
    let mut at = 0;
    while at < len {
        let start = match seek(at, libc::SEEK_DATA) {
            Ok(s) => s,
            // No data after `at`.
            Err(e) if e.raw_os_error() == Some(libc::ENXIO) => break,
            Err(_) => at,
        };
        let end = seek(start, libc::SEEK_HOLE).unwrap_or(len).min(len);
        if end <= start {
            break;
        }
        ranges.push((start, end));
        at = end;
    }
    ranges
}

/// Copies `ranges` of `from` to `to`, then makes `to` `len` bytes long; `copied` hears how many
/// bytes of the ranges are copied, from 0. Blocks of zeros are left holes, as `cp --sparse=always`
/// leaves them.
fn full(
    from: &File,
    to: &File,
    ranges: &[(u64, u64)],
    len: u64,
    copied: &mut dyn FnMut(u64),
) -> io::Result<()> {
    const BLOCK: usize = 4096;
    let mut buf = vec![0u8; 1 << 20];
    let mut done = 0;
    copied(0);
    for &(start, end) in ranges {
        let mut at = start;
        while at < end {
            let n = ((end - at) as usize).min(buf.len());
            from.read_exact_at(&mut buf[..n], at)?;
            // Each run of blocks not all zeros, written at once.
            let mut run: Option<usize> = None;
            for (i, block) in buf[..n].chunks(BLOCK).enumerate() {
                match (block.iter().all(|&b| b == 0), run) {
                    (false, None) => run = Some(i * BLOCK),
                    (true, Some(r)) => {
                        to.write_all_at(&buf[r..i * BLOCK], at + r as u64)?;
                        run = None;
                    }
                    _ => {}
                }
            }
            if let Some(r) = run {
                to.write_all_at(&buf[r..n], at + r as u64)?;
            }
            at += n as u64;
            done += n as u64;
            copied(done);
        }
    }
    to.set_len(len)
}

/// Why a copy is no reflink, from the file system's refusal of one.
#[derive(Debug, PartialEq, Eq)]
enum Why {
    /// The source is on the first file system, and the copy on the second.
    Apart(String, String),
    /// Their file system makes none.
    Unsupported(String),
    /// Their file system refused one, with this error.
    Refused(String, String),
}

impl Why {
    fn of(e: &io::Error, src: &Path, dst: &Path) -> Why {
        let fs = |p: &Path| {
            p.parent()
                .and_then(file_system)
                .unwrap_or_else(|| "its file system".into())
        };
        Why::from_error(e, fs(src), fs(dst))
    }

    /// From `e`, how a reflink of a file on `from` to one on `to` ended.
    fn from_error(e: &io::Error, from: String, to: String) -> Why {
        match e.raw_os_error() {
            Some(libc::EXDEV) => Why::Apart(from, to),
            Some(c)
                if [libc::EOPNOTSUPP, libc::ENOTSUP, libc::ENOTTY, libc::ENOSYS].contains(&c) =>
            {
                Why::Unsupported(from)
            }
            _ => Why::Refused(from, e.to_string()),
        }
    }

    /// Why a full copy of `src` to `dst` is no reflink, as it says before it starts.
    fn note(&self, src: &Path, dst: &Path) -> String {
        let dir = |p: &Path| paths::tilde(p.parent().unwrap_or(p));
        match self {
            Why::Apart(a, b) => format!(
                "{} and {} are different file systems ({a} and {b}), and a reflink is made \
                 within one",
                dir(src),
                dir(dst)
            ),
            Why::Unsupported(fs) => format!("{fs}, which {} is on, makes none", dir(src)),
            Why::Refused(fs, e) => format!("{fs} refused one: {e}"),
        }
    }

    /// Why, in the few words each step of the copy says.
    fn short(&self) -> String {
        match self {
            Why::Apart(..) => "different file systems".into(),
            Why::Unsupported(fs) => format!("{fs} makes no reflinks"),
            Why::Refused(fs, _) => format!("{fs} refused a reflink"),
        }
    }
}

/// The type of the file system `dir` is on: btrfs, ext4, tmpfs.
#[cfg(target_os = "linux")]
fn file_system(dir: &Path) -> Option<String> {
    let mounts = std::fs::read_to_string("/proc/self/mountinfo").ok()?;
    mount_type(&mounts, &std::fs::canonicalize(dir).ok()?)
}

/// The type of the file system `dir` is on: apfs, hfs, exfat.
#[cfg(target_os = "macos")]
fn file_system(dir: &Path) -> Option<String> {
    let p = c_path(dir).ok()?;
    // SAFETY: statfs fills the struct it is given; all zeros is a valid one.
    let mut s: libc::statfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statfs(p.as_ptr(), &mut s) } != 0 {
        return None;
    }
    // SAFETY: f_fstypename is NUL-terminated.
    let name = unsafe { std::ffi::CStr::from_ptr(s.f_fstypename.as_ptr()) };
    Some(name.to_string_lossy().into_owned())
}

/// The type of the mount `path`, absolute and with no symbolic links, is on, by `mountinfo`, as
/// /proc/self/mountinfo lists the mounts: the last on the longest mount point above it, which is
/// the one over the others there.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn mount_type(mountinfo: &str, path: &Path) -> Option<String> {
    let mut best: Option<(usize, &str)> = None;
    for line in mountinfo.lines() {
        let Some((mount, rest)) = line.split_once(" - ") else {
            continue;
        };
        let (Some(point), Some(fs)) = (mount.split(' ').nth(4), rest.split(' ').next()) else {
            continue;
        };
        let point = unescape(point);
        if path.starts_with(&point) && best.is_none_or(|(n, _)| point.len() >= n) {
            best = Some((point.len(), fs));
        }
    }
    best.map(|(_, fs)| fs.to_string())
}

/// A mount point as mountinfo writes it, where a space, tab, newline or backslash is `\` and its
/// three octal digits.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn unescape(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b.get(i + 1..i + 4) {
            Some(d) if b[i] == b'\\' && d.iter().all(|c| (b'0'..=b'7').contains(c)) => {
                out.push((d[0] - b'0') * 64 + (d[1] - b'0') * 8 + (d[2] - b'0'));
                i += 4;
            }
            _ => {
                out.push(b[i]);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn dir(base: &Path, tag: &str) -> PathBuf {
        let d = base.join(format!("reims-ctl-copy-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A file of `len` bytes: data at 0, a run of zeros, then a hole to the end.
    fn sparse_file(p: &Path, len: u64) -> Vec<u8> {
        use std::os::unix::fs::PermissionsExt;
        let f = File::create(p).unwrap();
        let data: Vec<u8> = (0..300_000u32).map(|i| (i * 7 + 1) as u8).collect();
        f.write_all_at(&data, 0).unwrap();
        f.write_all_at(&[0u8; 65536], data.len() as u64).unwrap();
        f.write_all_at(b"tail", len / 2).unwrap();
        f.set_len(len).unwrap();
        std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o640)).unwrap();
        std::fs::read(p).unwrap()
    }

    /// A refusal names why: the file systems differ, theirs makes no reflinks, or theirs refused
    /// one for a reason of its own, which it gives.
    #[test]
    fn a_refused_reflink_says_why() {
        let why = |code, a: &str, b: &str| {
            Why::from_error(&io::Error::from_raw_os_error(code), a.into(), b.into())
        };
        let (src, dst) = (
            paths::home().join("vms/old/disk.qcow2"),
            paths::home().join(".local/share/reims-ctl/vms/mac/disk.qcow2"),
        );
        let apart = why(libc::EXDEV, "ext4", "btrfs");
        assert_eq!(apart, Why::Apart("ext4".into(), "btrfs".into()));
        assert_eq!(
            apart.note(&src, &dst),
            "~/vms/old and ~/.local/share/reims-ctl/vms/mac are different file systems (ext4 \
             and btrfs), and a reflink is made within one"
        );
        assert_eq!(apart.short(), "different file systems");
        for code in [libc::EOPNOTSUPP, libc::ENOTTY] {
            let none = why(code, "ext4", "ext4");
            assert_eq!(none, Why::Unsupported("ext4".into()));
            assert_eq!(
                none.note(&src, &dst),
                "ext4, which ~/vms/old is on, makes none"
            );
            assert_eq!(none.short(), "ext4 makes no reflinks");
        }
        let refused = why(libc::EINVAL, "btrfs", "btrfs");
        let e = io::Error::from_raw_os_error(libc::EINVAL).to_string();
        assert_eq!(refused, Why::Refused("btrfs".into(), e.clone()));
        assert_eq!(refused.note(&src, &dst), format!("btrfs refused one: {e}"));
        assert_eq!(refused.short(), "btrfs refused a reflink");
    }

    /// A path's file system is the last mount on the longest mount point above it, whose escapes
    /// are undone.
    #[test]
    fn the_mount_table_names_the_file_system_a_path_is_on() {
        let mounts = "\
22 1 0:21 /@ / rw,noatime shared:1 - btrfs /dev/nvme0n1p2 rw,compress=zstd:1
23 22 0:22 / /tmp rw,noatime shared:5 - tmpfs tmpfs rw,inode64
24 22 0:21 /@home /home rw,noatime shared:2 - btrfs /dev/nvme0n1p2 rw
25 24 8:1 / /home/u/my\\040vms rw,relatime shared:9 - ext4 /dev/sda1 rw
26 23 0:30 / /tmp rw,relatime shared:6 - ramfs ramfs rw
";
        let fs = |p: &str| mount_type(mounts, Path::new(p));
        assert_eq!(fs("/home/u/vms/disk.qcow2").as_deref(), Some("btrfs"));
        assert_eq!(fs("/home/u/my vms/disk.qcow2").as_deref(), Some("ext4"));
        assert_eq!(fs("/home/u/my vmsx").as_deref(), Some("btrfs"));
        assert_eq!(fs("/tmp/x").as_deref(), Some("ramfs"));
        assert_eq!(fs("/usr").as_deref(), Some("btrfs"));
        assert_eq!(unescape("a\\134b\\011c\\0"), "a\\b\tc\\0");
    }

    /// A full copy is the source's bytes and length, its holes and zeros left holes, with its
    /// permissions, and says how far it is from 0 to all of the data.
    #[test]
    fn a_full_copy_keeps_the_bytes_and_the_holes() {
        let d = dir(&std::env::temp_dir(), "full");
        let len = 64 << 20;
        let bytes = sparse_file(&d.join("src"), len);
        let from = File::open(d.join("src")).unwrap();
        let to = create(&from, &d.join("dst")).unwrap();
        let ranges = data_ranges(&from, len);
        let mut seen = Vec::new();
        full(&from, &to, &ranges, len, &mut |n| seen.push(n)).unwrap();
        assert_eq!(std::fs::read(d.join("dst")).unwrap(), bytes);
        let meta = std::fs::metadata(d.join("dst")).unwrap();
        assert_eq!(meta.mode() & 0o777, 0o640);
        assert!(
            meta.blocks() * 512 < 8 << 20,
            "{} bytes",
            meta.blocks() * 512
        );
        let total: u64 = ranges.iter().map(|(a, b)| b - a).sum();
        assert_eq!(seen.first(), Some(&0));
        assert_eq!(seen.last(), Some(&total));
        assert!(seen.windows(2).all(|w| w[0] < w[1]), "{seen:?}");
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// A copy within a directory of btrfs or APFS is a reflink or clone, and says nothing; one
    /// within tmpfs, which makes none, is a full copy that says so; one between file systems of
    /// two types is a full copy that says they differ. Each is the source's bytes. The test
    /// copies within and between the temporary directory, the test's own and, on Linux,
    /// /dev/shm: one APFS volume on a Mac, and btrfs and tmpfs on the Linux hosts that check
    /// it.
    #[test]
    fn a_copy_is_a_reflink_where_it_can_be_and_says_why_where_not() {
        let exe = std::env::current_exe().unwrap();
        let mut dirs = vec![
            dir(&std::env::temp_dir(), "tmp"),
            dir(exe.parent().unwrap(), "target"),
        ];
        // tmpfs on Linux, wherever the temporary directory is.
        if Path::new("/dev/shm").is_dir() {
            dirs.push(dir(Path::new("/dev/shm"), "shm"));
        }
        for (i, from) in dirs.iter().enumerate() {
            let bytes = sparse_file(&from.join("src"), 4 << 20);
            for (j, to) in dirs.iter().enumerate() {
                let dst = to.join(format!("copy-{i}-{j}"));
                let mut seen = Vec::new();
                copy(&from.join("src"), &dst, 0, &mut |s| {
                    seen.push(s.to_string())
                })
                .unwrap();
                assert_eq!(std::fs::read(&dst).unwrap(), bytes);
                assert_eq!(std::fs::metadata(&dst).unwrap().mode() & 0o777, 0o640);
                let (a, b) = (file_system(from).unwrap(), file_system(to).unwrap());
                let said = seen.first().map(String::as_str).unwrap_or("");
                if a != b {
                    assert!(
                        said.contains("are different file systems"),
                        "{a} {b}: {seen:?}"
                    );
                } else if i == j && ["btrfs", "apfs"].contains(&a.as_str()) {
                    assert!(seen.is_empty(), "{a}: {seen:?}");
                } else if i == j && a == "tmpfs" {
                    assert!(said.ends_with("makes none"), "{seen:?}");
                    assert!(said.starts_with("copying src in full, 1 MB, not as a reflink"));
                    assert_eq!(
                        seen.last().unwrap(),
                        "copying src in full (tmpfs makes no reflinks): 100% of 1 MB"
                    );
                }
                std::fs::remove_file(&dst).unwrap();
            }
        }
        for d in &dirs {
            std::fs::remove_dir_all(d).unwrap();
        }
    }
}
