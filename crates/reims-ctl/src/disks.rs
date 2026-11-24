//! Disk passthrough: the host's whole disks a VM's guest is given while the VM runs, on x86-64
//! VMs on Linux hosts.
//!
//! - A chosen disk is kept in the VM's settings ([`HostDiskChoice`], a `[[host_disk]]` table) by
//!   its link in `/dev/disk/by-id`, which udev names from the disk's model and serial number, so
//!   the choice follows the disk across the host's reboots and from port to port. A disk with no
//!   link there (some USB enclosures give their disks none) is refused: its name in `/dev` can
//!   lead to another disk after a reboot. A loop device, which has no serial number, is kept by
//!   its file in `/dev`, and the list says that name may change.
//! - The guest sees it as a SATA disk on the VM's AHCI controller, where OpenCore's image is, at
//!   a port the VM's own disks leave free ([`crate::vm::Q35::host_disk_ports`]), whatever
//!   controller the VM's system disk is on, with no boot index, so it never boots ahead of
//!   OpenCore's image. QEMU opens it as a raw disk, named so (`driver=raw` over
//!   `driver=host_device`), never probed for an image format, so an image header the guest
//!   writes on it is never read as one; with `cache.direct=on`, so the guest's writes go to the
//!   disk with no copy in the host's page cache; and with `locking=on`, so a second QEMU can't
//!   open it for writing. QEMU attaches no SATA disk to a running machine, so a disk given or
//!   taken back while the VM runs is the guest's, or the host's again, from the VM's next start.
//! - Two systems writing one file system destroy it, so the host must not use a disk the guest
//!   has. A disk is refused ([`refusal`]) while it or one of its partitions is mounted (the
//!   host's root and boot file systems among them), is swap, or is part of a device-mapper or
//!   md device (LVM, LUKS, RAID), and where it holds the VM's own disk image. A start also asks
//!   the kernel ([`claimed`]): it opens the disk exclusively, which fails while anything in the
//!   kernel claims the disk or a partition of it, as a file system on several disks (btrfs's)
//!   claims each.
//! - QEMU opens the disk's file, which is root's and the disk group's (`brw-rw----`). The disk
//!   group may write every disk of the host, its system disk too, so reims-ctl never names it:
//!   where this user can't open a disk, it gives the udev rule that makes that one disk this
//!   user's, by its link in `/dev/disk/by-id`, and the commands that add and apply it
//!   ([`fix`]).

use crate::host::{Host, Os};
use crate::vm::{HostDiskChoice, Vm};
use anyhow::{Result, bail};
use std::path::{Path, PathBuf};

/// What reims-ctl says where it passes no disk through.
pub const ONLY: &str = "passing a disk through is for x86-64 VMs on Linux hosts, for now";

/// A block device's number, major and minor.
pub type Dev = (u32, u32);

/// Where Linux lists the host's block devices, and udev's stable names for them.
const SYS: &str = "/sys";
const BY_ID: &str = "/dev/disk/by-id";

/// What the host uses its block devices for, read once.
#[derive(Clone, Debug, Default)]
pub struct HostUse {
    /// Each mount's point, with its block device where it has one (tmpfs and the like have
    /// none).
    pub mounts: Vec<(Option<Dev>, PathBuf)>,
    /// The block devices that are swap.
    pub swaps: Vec<Dev>,
}

impl HostUse {
    /// This host's, from the kernel's tables of its mounts and swap.
    pub fn current() -> HostUse {
        let read = |p: &str| std::fs::read_to_string(p).unwrap_or_default();
        HostUse {
            mounts: parse_mountinfo(&read("/proc/self/mountinfo"), block_dev),
            swaps: parse_swaps(&read("/proc/swaps"), block_dev),
        }
    }

    /// The block device the file `f` is on: its own, where it is one, else that of the mount
    /// that holds it. `stat` answers the first.
    fn device_of(&self, f: &Path, stat: impl Fn(&Path) -> Option<Dev>) -> Option<Dev> {
        if let Some(d) = stat(f) {
            return Some(d);
        }
        let f = std::fs::canonicalize(f).unwrap_or_else(|_| f.to_path_buf());
        self.mounts
            .iter()
            .filter(|(_, point)| f.starts_with(point))
            .max_by_key(|(_, point)| point.components().count())
            .and_then(|(d, _)| *d)
    }
}

/// A whole disk of the host's: a disk on a controller, or a loop device over a file.
#[derive(Clone, Debug, PartialEq)]
pub struct HostDisk {
    /// The kernel's name for it: `sda`, `nvme0n1`, `loop7`.
    pub kernel: String,
    /// The file a VM keeps it by: its link in `/dev/disk/by-id`, else its file in `/dev`.
    pub path: PathBuf,
    pub bytes: u64,
    /// Its model, or the file a loop device reads.
    pub name: String,
    /// The file a loop device reads.
    pub loop_file: Option<String>,
    /// Its device, its partitions', and those of the devices made of them (device-mapper's,
    /// md's): a file on any of them is on this disk.
    pub devs: Vec<Dev>,
    /// What the host uses it for, a phrase each: what keeps it from a guest.
    pub uses: Vec<String>,
}

impl HostDisk {
    /// Whether `c` names it: by its file, or by a file that leads to it, as `/dev/sdb` or
    /// another of its links does.
    pub fn is(&self, c: &HostDiskChoice) -> bool {
        c.path == self.path
            || std::fs::canonicalize(&c.path)
                .ok()
                .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
                .is_some_and(|n| n == self.kernel)
    }

    /// Whether the name it is kept by is its file in `/dev`, which a reboot may give another
    /// disk.
    pub fn named_by_kernel(&self) -> bool {
        self.path.parent() == Some(Path::new("/dev"))
    }

    /// Its name and size, as the list shows it.
    pub fn label(&self) -> String {
        format!("{}, {}", self.name, size(self.bytes))
    }

    /// Whether this process may open its file to drive it, as QEMU, which runs as this user,
    /// opens it.
    pub fn usable(&self) -> bool {
        crate::os::may_read_write(&self.path)
    }
}

/// A size as disks are sold by: GB, or TB from a thousand GB.
pub fn size(bytes: u64) -> String {
    let (n, unit) = if bytes >= 1_000_000_000_000 {
        (bytes as f64 / 1e12, "TB")
    } else {
        (bytes as f64 / 1e9, "GB")
    };
    let s = format!("{n:.1}");
    format!("{} {unit}", s.strip_suffix(".0").unwrap_or(&s))
}

/// The host's whole disks, with what the host uses each for; none on a host that has no Linux
/// sysfs.
pub fn host_disks() -> Vec<HostDisk> {
    read_disks(Path::new(SYS), Path::new(BY_ID), &HostUse::current())
}

/// The disks under `sys` (Linux's `/sys`: `block/` lists every block device, a disk's
/// partitions in its directory, and `class/block/` every one by its name), each kept by its
/// link in `by_id`, with what `used` says the host uses it for. Left out: optical drives
/// (`sr`), floppies, read-only and empty devices, the kernel's own (zram, device-mapper, md,
/// RAM disks, which have no `device`), and loop devices without a file.
fn read_disks(sys: &Path, by_id: &Path, used: &HostUse) -> Vec<HostDisk> {
    let Ok(dir) = std::fs::read_dir(sys.join("block")) else {
        return Vec::new();
    };
    let links = by_id_links(by_id);
    let mut names: Vec<String> = dir
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .collect();
    names.sort();
    let mut out = Vec::new();
    for kernel in names {
        let d = sys.join("block").join(&kernel);
        let read = |f: &Path| {
            std::fs::read_to_string(f)
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        };
        let at = |f: &str| read(&d.join(f));
        let loop_file = at("loop/backing_file");
        let physical = d.join("device").exists();
        if !(physical || loop_file.is_some())
            || kernel.starts_with("sr")
            || kernel.starts_with("fd")
            || at("ro").as_deref() == Some("1")
        {
            continue;
        }
        let bytes = at("size").and_then(|s| s.parse::<u64>().ok()).unwrap_or(0) * 512;
        let Some(dev) = at("dev").as_deref().and_then(parse_dev) else {
            continue;
        };
        if bytes == 0 {
            continue;
        }
        // The disk, then its partitions: each with its device and its directory.
        let mut parts = vec![(None, dev, d.clone())];
        if let Ok(entries) = std::fs::read_dir(&d) {
            let mut found: Vec<(String, Dev, PathBuf)> = entries
                .filter_map(|e| {
                    let e = e.ok()?;
                    let p = e.path();
                    let name = e.file_name().into_string().ok()?;
                    if !p.join("partition").exists() {
                        return None;
                    }
                    Some((name, parse_dev(&read(&p.join("dev"))?)?, p))
                })
                .collect();
            found.sort();
            parts.extend(found.into_iter().map(|(n, dev, p)| (Some(n), dev, p)));
        }
        let mut devs = Vec::new();
        let mut uses = Vec::new();
        for (part, pdev, pdir) in parts {
            devs.push(pdev);
            let who = part.map(|p| format!("{p} ")).unwrap_or_default();
            let mounted = |dev: Dev| -> Vec<String> {
                used.mounts
                    .iter()
                    .filter(|(d, _)| *d == Some(dev))
                    .map(|(_, p)| p.display().to_string())
                    .collect()
            };
            for point in mounted(pdev) {
                uses.push(format!("{who}mounted at {point}"));
            }
            if used.swaps.contains(&pdev) {
                uses.push(format!("{who}swap"));
            }
            let mut holders: Vec<String> = std::fs::read_dir(pdir.join("holders"))
                .map(|es| {
                    es.filter_map(|e| e.ok()?.file_name().into_string().ok())
                        .collect()
                })
                .unwrap_or_default();
            holders.sort();
            for h in holders {
                let hdev = read(&sys.join("class/block").join(&h).join("dev"))
                    .as_deref()
                    .and_then(parse_dev);
                let mut what = Vec::new();
                if let Some(hdev) = hdev {
                    devs.push(hdev);
                    what.extend(mounted(hdev).into_iter().map(|p| format!("mounted at {p}")));
                    if used.swaps.contains(&hdev) {
                        what.push("swap".into());
                    }
                }
                uses.push(if what.is_empty() {
                    format!("{who}in {h}")
                } else {
                    format!("{who}in {h} ({})", what.join(", "))
                });
            }
        }
        let name = match &loop_file {
            Some(f) => format!("loop device over {f}"),
            None => model(
                &at("device/vendor"),
                &at("device/model"),
                &at("device/name"),
            ),
        };
        let path = match links.iter().find(|(k, _)| *k == kernel) {
            Some((_, link)) => by_id.join(link),
            None => Path::new("/dev").join(&kernel),
        };
        out.push(HostDisk {
            kernel,
            path,
            bytes,
            name,
            loop_file,
            devs,
            uses,
        });
    }
    out
}

/// A disk's name from what its controller says: its vendor where its model does not name it
/// (SATA disks give `ATA` as theirs), or an SD card's name.
fn model(vendor: &Option<String>, model: &Option<String>, card: &Option<String>) -> String {
    match (vendor.as_deref(), model.as_deref().or(card.as_deref())) {
        (Some(v), Some(m))
            if v != "ATA" && !m.to_ascii_lowercase().contains(&v.to_ascii_lowercase()) =>
        {
            format!("{v} {m}")
        }
        (_, Some(m)) => m.to_string(),
        (Some(v), None) => format!("{v} disk"),
        (None, None) => "disk".into(),
    }
}

/// `by_id`'s links to whole disks, the one to keep each by: a link named from the disk's model
/// and serial number (`ata-`, `nvme-`, `usb-`) before one named from a number (`wwn-`,
/// `nvme-eui.`, `scsi-`), then the shortest, as NVMe's names without the namespace's `_1`.
/// Links to partitions end in `-partN` and lead to the partition, so they name no disk here.
fn by_id_links(by_id: &Path) -> Vec<(String, String)> {
    let Ok(dir) = std::fs::read_dir(by_id) else {
        return Vec::new();
    };
    let mut links: Vec<(String, String)> = dir
        .filter_map(|e| {
            let e = e.ok()?;
            let link = e.file_name().into_string().ok()?;
            let target = std::fs::read_link(e.path()).ok()?;
            Some((target.file_name()?.to_str()?.to_string(), link))
        })
        .collect();
    let rank = |l: &str| {
        let numbered = [
            "wwn-",
            "nvme-eui.",
            "nvme-nvme.",
            "scsi-",
            "dm-",
            "md-",
            "lvm-",
        ];
        (
            numbered.iter().any(|p| l.starts_with(p)),
            l.len(),
            l.to_string(),
        )
    };
    links.sort_by_key(|(k, l)| (k.clone(), rank(l)));
    links.dedup_by(|a, b| a.0 == b.0);
    links
}

/// A device number as sysfs gives it, `8:16`.
fn parse_dev(s: &str) -> Option<Dev> {
    let (a, b) = s.trim().split_once(':')?;
    Some((a.parse().ok()?, b.parse().ok()?))
}

/// A path as the kernel's tables write it: a space, tab, newline or backslash as `\` and three
/// octal digits.
fn unescape(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let octal = b
            .get(i + 1..i + 4)
            .filter(|d| d.iter().all(|c| (b'0'..=b'7').contains(c)));
        match octal {
            Some(d) if b[i] == b'\\' => {
                out.push(d.iter().fold(0u8, |n, c| n.wrapping_mul(8) + (c - b'0')));
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

/// The mounts of `/proc/self/mountinfo`'s text: each one's point, and its source's block
/// device, which `stat` gives where the source is a file (a btrfs mount's own device number is
/// one the kernel makes up), else its own where it is a real one (major above 0).
fn parse_mountinfo(text: &str, stat: impl Fn(&Path) -> Option<Dev>) -> Vec<(Option<Dev>, PathBuf)> {
    text.lines()
        .filter_map(|l| {
            let f: Vec<&str> = l.split(' ').collect();
            let sep = f.iter().position(|x| *x == "-")?;
            let point = PathBuf::from(unescape(f.get(4)?));
            let source = unescape(f.get(sep + 2)?);
            let own = parse_dev(f.get(2)?).filter(|d| d.0 != 0);
            let dev = if source.starts_with('/') {
                stat(Path::new(&source)).or(own)
            } else {
                own
            };
            Some((dev, point))
        })
        .collect()
}

/// The block devices `/proc/swaps`' text lists; a swap file is no device's.
fn parse_swaps(text: &str, stat: impl Fn(&Path) -> Option<Dev>) -> Vec<Dev> {
    text.lines()
        .skip(1)
        .filter_map(|l| stat(Path::new(&unescape(l.split_whitespace().next()?))))
        .collect()
}

/// The block device the file `p` is, if it is one.
#[cfg(target_os = "linux")]
pub fn block_dev(p: &Path) -> Option<Dev> {
    use std::os::unix::fs::{FileTypeExt, MetadataExt};
    let m = std::fs::metadata(p).ok()?;
    m.file_type()
        .is_block_device()
        .then(|| (libc::major(m.rdev()), libc::minor(m.rdev())))
}

/// The block device the file `p` is: none off Linux, which passes no disk through.
#[cfg(not(target_os = "linux"))]
pub fn block_dev(_p: &Path) -> Option<Dev> {
    None
}

/// Whether the kernel holds the disk at `p` for something of its own: opened exclusively, it
/// answers EBUSY while the disk or a partition of it is mounted, swap, or in another device.
#[cfg(target_os = "linux")]
pub fn claimed(p: &Path) -> bool {
    use std::os::unix::fs::OpenOptionsExt;
    let open = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_EXCL)
        .open(p);
    matches!(open, Err(e) if e.raw_os_error() == Some(libc::EBUSY))
}

/// Whether the kernel holds the disk at `p`: never asked off Linux.
#[cfg(not(target_os = "linux"))]
pub fn claimed(_p: &Path) -> bool {
    false
}

/// The block devices that hold the VM's own images: its disk's, and OpenCore's and the
/// installer's on an x86-64 VM.
pub fn own_devs(vm: &Vm, used: &HostUse) -> Vec<Dev> {
    let mut files = vec![vm.disk()];
    if let Some(q) = &vm.config.q35 {
        files.push(vm.path(&q.opencore));
        files.extend(q.installer.as_deref().map(|i| vm.path(i)));
    }
    files
        .iter()
        .filter_map(|f| used.device_of(f, block_dev))
        .collect()
}

/// Why `d` can't be a guest's, a sentence; None where it can. `own` are the devices that hold
/// the VM's images ([`own_devs`]).
pub fn refusal(d: &HostDisk, own: &[Dev]) -> Option<String> {
    let mut why = Vec::new();
    if d.devs.iter().any(|x| own.contains(x)) {
        why.push("it holds this VM's own disk".to_string());
    }
    why.extend(d.uses.iter().cloned());
    if !why.is_empty() {
        return Some(format!(
            "{} ({}) is this host's: {}. A guest writing a disk the host uses would destroy it",
            d.label(),
            d.path.display(),
            why.join(", ")
        ));
    }
    unnamed(d).then(|| {
        format!(
            "{} ({}) has no link in /dev/disk/by-id, from its model and serial number, which \
             some USB enclosures give their disks none of: its name in /dev can lead to another \
             disk after the host's next boot, so a VM can't keep it",
            d.label(),
            d.path.display()
        )
    })
}

/// Whether `d` is a disk with no stable name: no link in `/dev/disk/by-id`, and no loop device,
/// which is named by its file in `/dev` alone.
fn unnamed(d: &HostDisk) -> bool {
    d.loop_file.is_none() && d.named_by_kernel()
}

/// One line of a VM's choice: a disk of the host's, with the setting that gives it to the VM
/// if one does, or a disk the VM is given that is not attached.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub disk: Option<HostDisk>,
    pub chosen: Option<HostDiskChoice>,
}

impl Row {
    pub fn name(&self) -> String {
        match (&self.disk, &self.chosen) {
            (Some(d), _) => d.label(),
            (None, Some(c)) if !c.name.is_empty() => c.name.clone(),
            _ => "disk".into(),
        }
    }

    /// The file it is kept by.
    pub fn path(&self) -> PathBuf {
        match (&self.disk, &self.chosen) {
            (Some(d), _) => d.path.clone(),
            (None, Some(c)) => c.path.clone(),
            _ => PathBuf::new(),
        }
    }

    /// What to say of it beside its name and file: what the host uses it for, which keeps it
    /// from a guest, a name a reboot may change, and whether this user can open it. `own` are
    /// the devices that hold the VM's images.
    pub fn notes(&self, own: &[Dev]) -> Vec<String> {
        let Some(d) = &self.disk else {
            return vec!["not attached".into()];
        };
        let mut n = Vec::new();
        if d.devs.iter().any(|x| own.contains(x)) {
            n.push("holds this VM's own disk".to_string());
        }
        n.extend(d.uses.iter().cloned());
        if unnamed(d) {
            n.push("no link in /dev/disk/by-id".into());
        } else if d.named_by_kernel() {
            n.push("its name may change at the host's next boot".into());
        }
        if !d.usable() {
            n.push("you can't open it".into());
        }
        n
    }
}

/// The host's disks, each with the choice in `chosen` that gives it, then the chosen disks that
/// are not attached. Loop devices the host uses (a snap's) are left out unless chosen.
pub fn rows(chosen: &[HostDiskChoice], disks: &[HostDisk]) -> Vec<Row> {
    let mut rows: Vec<Row> = disks
        .iter()
        .map(|d| Row {
            disk: Some(d.clone()),
            chosen: chosen.iter().find(|c| d.is(c)).cloned(),
        })
        .filter(|r| {
            r.chosen.is_some()
                || r.disk
                    .as_ref()
                    .is_some_and(|d| d.loop_file.is_none() || d.uses.is_empty())
        })
        .collect();
    rows.extend(
        chosen
            .iter()
            .filter(|c| !disks.iter().any(|d| d.is(c)))
            .map(|c| Row {
                disk: None,
                chosen: Some(c.clone()),
            }),
    );
    rows
}

/// The choice that gives the VM `d`.
pub fn choice_of(d: &HostDisk) -> HostDiskChoice {
    HostDiskChoice {
        path: d.path.clone(),
        name: d.label(),
    }
}

/// `chosen` with `row` turned over: a disk the VM has is taken back, and one it hasn't is
/// given, unless the host uses it ([`refusal`]). `own` are the devices that hold the VM's
/// images.
pub fn toggle(chosen: &[HostDiskChoice], row: &Row, own: &[Dev]) -> Result<Vec<HostDiskChoice>> {
    if let Some(c) = &row.chosen {
        return Ok(chosen.iter().filter(|o| *o != c).cloned().collect());
    }
    let Some(d) = &row.disk else {
        return Ok(chosen.to_vec());
    };
    if let Some(why) = refusal(d, own) {
        bail!("{why}");
    }
    let mut c = chosen.to_vec();
    c.push(choice_of(d));
    Ok(c)
}

/// Gives the VM the disks `change` makes of those it has, in its settings as they are now, and
/// saves them. They are the guest's from the VM's next start. Returns the VM as saved, and what
/// happened, a sentence a disk.
pub fn apply(
    vm: &Vm,
    change: impl FnOnce(&[HostDiskChoice]) -> Result<Vec<HostDiskChoice>>,
) -> Result<(Vm, Vec<String>)> {
    let mut now = Vm::load(&vm.dir)?;
    let old = now.config.host_disks.clone();
    now.config.host_disks = change(&old)?;
    let name = now.name().to_string();
    if now.config.host_disks == old {
        return Ok((now, vec![format!("{name}: nothing changed")]));
    }
    now.save()?;
    let running = crate::runtime::running(&now);
    let label = |c: &HostDiskChoice| match c.name.as_str() {
        "" => c.path.display().to_string(),
        n => n.to_string(),
    };
    let mut said: Vec<String> = old
        .iter()
        .filter(|c| !now.config.host_disks.contains(c))
        .map(|c| {
            if running {
                format!("{name}: {} stays the guest's until the VM stops", label(c))
            } else {
                format!("{name}: {} is no longer the VM's", label(c))
            }
        })
        .collect();
    said.extend(
        now.config
            .host_disks
            .iter()
            .filter(|c| !old.contains(c))
            .map(|c| format!("{name}: {} is the guest's from its next start", label(c))),
    );
    Ok((now, said))
}

/// What lets QEMU, as this user, open the disk `d`, and it alone: a udev rule that makes it
/// this user's by its link in `/dev/disk/by-id`, in a file of its own, with the commands that
/// add and apply it; for a loop device, which has no such link, a change of its file's owner
/// that lasts while it is attached.
pub fn fix(d: &HostDisk) -> String {
    let user = user();
    if d.loop_file.is_some() || d.named_by_kernel() {
        return format!(
            "give it to yourself while it is attached: sudo chown {user} {}",
            d.path.display()
        );
    }
    let (rule, file) = udev_rule(&d.path, &user);
    format!(
        "give this disk alone to yourself with a udev rule of its own (the disk group, which could \
         open it too, may write every disk of this host):\n  echo '{rule}' | sudo tee {file}\n  \
         sudo udevadm control --reload && sudo udevadm trigger --action=change {} && sudo \
         udevadm settle",
        d.path.display()
    )
}

/// The udev rule that makes the disk whose link in `/dev/disk/by-id` is `path` the user
/// `user`'s, and the file it goes in: after systemd's 60-persistent-storage.rules, which makes
/// the link.
fn udev_rule(path: &Path, user: &str) -> (String, String) {
    let link = path.file_name().unwrap_or_default().to_string_lossy();
    let safe: String = link
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect();
    (
        format!(
            "SUBSYSTEM==\"block\", ENV{{DEVTYPE}}==\"disk\", SYMLINK==\"disk/by-id/{link}\", \
             OWNER=\"{user}\""
        ),
        format!("/etc/udev/rules.d/71-reims-ctl-disk-{safe}.rules"),
    )
}

/// This user's name, for a udev rule's owner: the login's, else the user's number.
fn user() -> String {
    std::env::var("USER")
        .or_else(|_| std::env::var("LOGNAME"))
        .ok()
        .filter(|u| !u.is_empty())
        .or_else(|| {
            let status = std::fs::read_to_string("/proc/self/status").ok()?;
            let line = status.lines().find(|l| l.starts_with("Uid:"))?;
            Some(line.split_whitespace().nth(1)?.to_string())
        })
        .unwrap_or_else(|| "$USER".into())
}

/// What keeps the VM's chosen disks from its guest now, a sentence each: one not attached, one
/// the host uses, one this user can't open.
pub fn problems(vm: &Vm) -> Vec<String> {
    if vm.config.host_disks.is_empty() {
        return Vec::new();
    }
    let used = HostUse::current();
    let disks = read_disks(Path::new(SYS), Path::new(BY_ID), &used);
    let own = own_devs(vm, &used);
    let mut p = Vec::new();
    for c in &vm.config.host_disks {
        let name = match c.name.as_str() {
            "" => c.path.display().to_string(),
            n => format!("{n} ({})", c.path.display()),
        };
        match disks.iter().find(|d| d.is(c)) {
            None => p.push(format!(
                "The disk {name} isn't attached: attach it, or take it back with reims-ctl \
                 disks {} --remove {}.",
                vm.name(),
                c.path.display()
            )),
            Some(d) => {
                if let Some(why) = refusal(d, &own) {
                    p.push(format!("{why}."));
                } else if !d.usable() {
                    p.push(format!(
                        "You can't open the disk {name}, so QEMU can't: {}",
                        fix(d)
                    ));
                }
            }
        }
    }
    p
}

/// Refuses a start whose disks the guest can't have: on a host or VM that passes none through,
/// the [`problems`], and a disk the kernel holds for the host ([`claimed`]).
pub fn check_start(vm: &Vm, host: Host) -> Result<()> {
    if vm.config.host_disks.is_empty() {
        return Ok(());
    }
    if host.os != Os::Linux || vm.config.q35.is_none() {
        bail!("{ONLY}");
    }
    if let Some(p) = problems(vm).into_iter().next() {
        bail!("{p}");
    }
    for c in &vm.config.host_disks {
        if claimed(&c.path) {
            bail!(
                "the host is using the disk {} ({}): the kernel holds it or a partition of it \
                 (mounted, swap, or part of another device), and a guest writing it would \
                 destroy it",
                c.name,
                c.path.display()
            );
        }
    }
    Ok(())
}

/// The row a command line's `DISK` names: its number in the list, its file or a file that
/// leads to it, its link's name in `/dev/disk/by-id`, its kernel name, or words of its name. A
/// number past the list names no disk, rather than one whose name holds its digits.
pub fn find(rows: &[Row], arg: &str) -> Result<usize> {
    let a = arg.trim();
    if let Ok(n) = a.parse::<usize>() {
        if (1..=rows.len()).contains(&n) {
            return Ok(n - 1);
        }
        bail!("there is no disk {n} in the list: reims-ctl disks lists them");
    }
    let low = a.to_ascii_lowercase();
    let given = HostDiskChoice {
        path: PathBuf::from(a),
        name: String::new(),
    };
    let by = |f: &dyn Fn(&Row) -> bool| -> Vec<usize> {
        rows.iter()
            .enumerate()
            .filter(|(_, r)| f(r))
            .map(|(i, _)| i)
            .collect()
    };
    for hits in [
        by(&|r| {
            r.path() == Path::new(a)
                || r.path().file_name() == Some(std::ffi::OsStr::new(a))
                || r.disk
                    .as_ref()
                    .is_some_and(|d| d.kernel == a || d.is(&given))
        }),
        by(&|r| r.name().to_ascii_lowercase().contains(&low)),
    ] {
        match hits.len() {
            0 => continue,
            1 => return Ok(hits[0]),
            _ => bail!("{a:?} names several disks: give its number or its file"),
        }
    }
    bail!("no disk of this host's matches {a:?}: reims-ctl disks lists them")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sysfs of block devices and a `/dev/disk/by-id` in a temporary directory, as Linux
    /// lists them.
    #[cfg(unix)]
    struct Fake(PathBuf);

    #[cfg(unix)]
    impl Fake {
        fn new(tag: &str) -> Fake {
            let d =
                std::env::temp_dir().join(format!("reims-ctl-disks-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&d);
            std::fs::create_dir_all(d.join("sys/block")).unwrap();
            std::fs::create_dir_all(d.join("sys/class/block")).unwrap();
            std::fs::create_dir_all(d.join("by-id")).unwrap();
            Fake(d)
        }

        fn sys(&self) -> PathBuf {
            self.0.join("sys")
        }

        fn by_id(&self) -> PathBuf {
            self.0.join("by-id")
        }

        fn write(&self, rel: &str, v: &str) -> &Fake {
            let f = self.0.join("sys").join(rel);
            std::fs::create_dir_all(f.parent().unwrap()).unwrap();
            std::fs::write(f, format!("{v}\n")).unwrap();
            self
        }

        /// A disk on a controller: its `device` directory holds its vendor and model.
        fn disk(&self, name: &str, dev: &str, sectors: u64, attrs: &[(&str, &str)]) -> &Fake {
            std::fs::create_dir_all(self.0.join(format!("sys/block/{name}/device"))).unwrap();
            self.write(&format!("block/{name}/dev"), dev)
                .write(&format!("block/{name}/size"), &sectors.to_string())
                .write(&format!("block/{name}/ro"), "0");
            for (k, v) in attrs {
                self.write(&format!("block/{name}/{k}"), v);
            }
            self
        }

        fn part(&self, disk: &str, name: &str, dev: &str) -> &Fake {
            self.write(&format!("block/{disk}/{name}/partition"), "1")
                .write(&format!("block/{disk}/{name}/dev"), dev)
        }

        /// `holder` (dm-0, md0) made of `part`'s device, at `dev`.
        fn holder(&self, part_dir: &str, holder: &str, dev: &str) -> &Fake {
            std::fs::create_dir_all(
                self.0
                    .join(format!("sys/block/{part_dir}/holders/{holder}")),
            )
            .unwrap();
            self.write(&format!("class/block/{holder}/dev"), dev)
        }

        fn link(&self, link: &str, kernel: &str) -> &Fake {
            std::os::unix::fs::symlink(format!("../../{kernel}"), self.by_id().join(link)).unwrap();
            self
        }
    }

    #[cfg(unix)]
    impl Drop for Fake {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A host: an NVMe system disk whose root is in LUKS (dm-0) and whose EFI partition is
    /// mounted, a SATA disk with a swap partition, a USB disk nothing uses, a loop device over
    /// a file nothing uses and a snap's mounted one, beside what is never a disk to give.
    #[cfg(unix)]
    fn host() -> (Fake, HostUse) {
        let f = Fake::new(&format!("host-{:?}", std::thread::current().id()));
        f.disk(
            "nvme0n1",
            "259:0",
            1_953_525_168,
            &[("device/model", "Samsung SSD 980 PRO 1TB")],
        )
        .part("nvme0n1", "nvme0n1p1", "259:1")
        .part("nvme0n1", "nvme0n1p2", "259:2")
        .holder("nvme0n1/nvme0n1p2", "dm-0", "253:0")
        .disk(
            "sda",
            "8:0",
            7_814_037_168,
            &[
                ("device/vendor", "ATA"),
                ("device/model", "WDC WD40EFRX-68N"),
            ],
        )
        .part("sda", "sda1", "8:1")
        .part("sda", "sda2", "8:2")
        .disk(
            "sdb",
            "8:16",
            62_530_624,
            &[
                ("device/vendor", "SanDisk"),
                ("device/model", "Cruzer Blade"),
            ],
        )
        .part("sdb", "sdb1", "8:17")
        .write("block/loop0/dev", "7:0")
        .write("block/loop0/size", "113712")
        .write("block/loop0/ro", "1")
        .write(
            "block/loop0/loop/backing_file",
            "/var/lib/snapd/snaps/core_1.snap",
        )
        .write("block/loop1/dev", "7:1")
        .write("block/loop1/size", "209715")
        .write("block/loop1/ro", "0")
        .write("block/loop1/loop/backing_file", "/home/u/snap.img")
        .write("block/loop7/dev", "7:7")
        .write("block/loop7/size", "2097152")
        .write("block/loop7/ro", "0")
        .write("block/loop7/loop/backing_file", "/srv/scratch/disk.img")
        // A loop device with no file, zram and dm-0 have no `device`.
        .write("block/loop2/dev", "7:2")
        .write("block/loop2/size", "0")
        .write("block/zram0/dev", "252:0")
        .write("block/zram0/size", "16777216")
        .write("block/dm-0/dev", "253:0")
        .write("block/dm-0/size", "1900000000")
        .disk("sr0", "11:0", 2_097_151, &[("device/model", "DVD-RW")])
        .disk("mmcblk0boot0", "179:8", 8192, &[("ro", "1")])
        // A card reader with no card.
        .disk("sdc", "8:32", 0, &[("device/model", "SD Reader")])
        // A virtio disk with no serial number, which udev links nowhere in by-id.
        .disk("vda", "254:0", 41_943_040, &[])
        .link("nvme-Samsung_SSD_980_PRO_1TB_S5GXNX0R123456A", "nvme0n1")
        .link("nvme-Samsung_SSD_980_PRO_1TB_S5GXNX0R123456A_1", "nvme0n1")
        .link("nvme-eui.002538b111b2c3d4", "nvme0n1")
        .link(
            "nvme-Samsung_SSD_980_PRO_1TB_S5GXNX0R123456A-part1",
            "nvme0n1p1",
        )
        .link("wwn-0x50014ee2b5c3d4e5", "sda")
        .link("ata-WDC_WD40EFRX-68N32N0_WD-WCC7K1234567", "sda")
        .link("ata-WDC_WD40EFRX-68N32N0_WD-WCC7K1234567-part2", "sda2")
        .link("usb-SanDisk_Cruzer_Blade_4C530001230512116021-0:0", "sdb");
        let used = HostUse {
            mounts: vec![
                (Some((253, 0)), "/".into()),
                (Some((259, 1)), "/boot/efi".into()),
                (None, "/tmp".into()),
                (Some((7, 1)), "/snap/core/1".into()),
            ],
            swaps: vec![(8, 2)],
        };
        (f, used)
    }

    /// The host's disks: each kept by its by-id link from its model and serial number, else
    /// its file in /dev; with what the host uses it or its partitions for; never an optical
    /// drive, a read-only, empty or kernel-made device, or a loop device with no file.
    #[cfg(unix)]
    #[test]
    fn host_disks_are_read_from_sysfs_with_what_the_host_uses_them_for() {
        let (f, used) = host();
        let d = read_disks(&f.sys(), &f.by_id(), &used);
        let got: Vec<(&str, String, &str, String, &[String])> = d
            .iter()
            .map(|d| {
                (
                    d.kernel.as_str(),
                    d.path
                        .strip_prefix(f.by_id())
                        .unwrap_or(&d.path)
                        .display()
                        .to_string(),
                    d.name.as_str(),
                    size(d.bytes),
                    d.uses.as_slice(),
                )
            })
            .collect();
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        assert_eq!(
            got,
            [
                (
                    "loop1",
                    "/dev/loop1".into(),
                    "loop device over /home/u/snap.img",
                    "0.1 GB".into(),
                    &s(&["mounted at /snap/core/1"])[..],
                ),
                (
                    "loop7",
                    "/dev/loop7".into(),
                    "loop device over /srv/scratch/disk.img",
                    "1.1 GB".into(),
                    &[][..],
                ),
                (
                    "nvme0n1",
                    "nvme-Samsung_SSD_980_PRO_1TB_S5GXNX0R123456A".into(),
                    "Samsung SSD 980 PRO 1TB",
                    "1 TB".into(),
                    &s(&[
                        "nvme0n1p1 mounted at /boot/efi",
                        "nvme0n1p2 in dm-0 (mounted at /)"
                    ])[..],
                ),
                (
                    "sda",
                    "ata-WDC_WD40EFRX-68N32N0_WD-WCC7K1234567".into(),
                    "WDC WD40EFRX-68N",
                    "4 TB".into(),
                    &s(&["sda2 swap"])[..],
                ),
                (
                    "sdb",
                    "usb-SanDisk_Cruzer_Blade_4C530001230512116021-0:0".into(),
                    "SanDisk Cruzer Blade",
                    "32 GB".into(),
                    &[][..],
                ),
                ("vda", "/dev/vda".into(), "disk", "21.5 GB".into(), &[][..]),
            ]
        );
        // A file on the LUKS root is on the NVMe disk.
        assert!(d[2].devs.contains(&(253, 0)));
        assert!(d[0].named_by_kernel() && !d[2].named_by_kernel());
        // No sysfs, as on macOS: no disks.
        assert!(read_disks(&f.0.join("none"), &f.by_id(), &used).is_empty());
    }

    /// The mount table's devices: a source file's own device (btrfs's mount number is made
    /// up), else the mount's where it is real; mount points with spaces unescaped.
    #[test]
    fn mounts_and_swap_are_read_from_the_kernels_tables() {
        let stat = |p: &Path| match p.to_str()? {
            "/dev/nvme0n1p2" => Some((259, 2)),
            "/dev/sda2" => Some((8, 2)),
            "/dev/sdb1" => Some((8, 17)),
            _ => None,
        };
        let mountinfo = "\
22 1 0:31 / / rw,relatime shared:1 - btrfs /dev/nvme0n1p2 rw,compress=zstd:1,subvol=/@
23 22 0:21 / /proc rw,nosuid shared:5 - proc proc rw
24 22 259:1 / /boot/efi rw,relatime shared:2 - vfat /dev/nvme0n1p1 rw
25 22 0:40 / /tmp rw shared:3 - tmpfs tmpfs rw
26 22 8:17 / /media/u/My\\040Disk rw,nosuid shared:4 - exfat /dev/sdb1 rw
27 22 253:3 / /data rw shared:6 opt:x - xfs /dev/mapper/data rw
";
        assert_eq!(
            parse_mountinfo(mountinfo, stat),
            [
                (Some((259, 2)), PathBuf::from("/")),
                (None, "/proc".into()),
                (Some((259, 1)), "/boot/efi".into()),
                (None, "/tmp".into()),
                (Some((8, 17)), "/media/u/My Disk".into()),
                (Some((253, 3)), "/data".into()),
            ]
        );
        let swaps = "Filename\t\t\t\tType\t\tSize\t\tUsed\t\tPriority\n\
                     /dev/sda2                               partition\t8388604\t\t0\t\t-2\n\
                     /swap/swapfile                          file\t\t4194300\t\t0\t\t-3\n";
        assert_eq!(parse_swaps(swaps, stat), [(8, 2)]);
        // A file is on its mount's device: the longest mount point that holds it.
        let used = HostUse {
            mounts: parse_mountinfo(mountinfo, stat),
            swaps: vec![],
        };
        let none = |_: &Path| None;
        assert_eq!(
            used.device_of(Path::new("/media/u/My Disk/vm/disk.qcow2"), none),
            Some((8, 17))
        );
        assert_eq!(
            used.device_of(Path::new("/home/u/vm/disk.qcow2"), none),
            Some((259, 2))
        );
        assert_eq!(used.device_of(Path::new("/tmp/vm/disk.qcow2"), none), None);
        // A disk used in place is its own device.
        assert_eq!(used.device_of(Path::new("/dev/sdb1"), stat), Some((8, 17)));
    }

    /// A disk the host uses is refused, with every use, and so is the disk that holds the VM's
    /// own image; a disk nothing uses is given and taken back.
    #[cfg(unix)]
    #[test]
    fn a_disk_the_host_uses_is_refused() {
        let (f, used) = host();
        let disks = read_disks(&f.sys(), &f.by_id(), &used);
        let r = rows(&[], &disks);
        // The snap's loop device is left out.
        let names: Vec<&str> = r
            .iter()
            .map(|r| r.disk.as_ref().unwrap().kernel.as_str())
            .collect();
        assert_eq!(names, ["loop7", "nvme0n1", "sda", "sdb", "vda"]);
        let root = refusal(&disks[2], &[]).unwrap();
        assert!(
            root.contains("nvme0n1p1 mounted at /boot/efi, nvme0n1p2 in dm-0 (mounted at /)"),
            "{root}"
        );
        assert!(refusal(&disks[3], &[]).unwrap().contains("sda2 swap"));
        let e = toggle(&[], &r[2], &[]).unwrap_err().to_string();
        assert!(e.contains("sda2 swap"), "{e}");
        // The USB disk is free, unless the VM's own image is on it.
        assert_eq!(refusal(&disks[4], &[]), None);
        let own = refusal(&disks[4], &[(8, 17)]).unwrap();
        assert!(own.contains("it holds this VM's own disk"), "{own}");
        assert!(
            r[3].notes(&[(8, 17)])
                .contains(&"holds this VM's own disk".to_string())
        );
        let given = toggle(&[], &r[3], &[]).unwrap();
        assert_eq!(given, [choice_of(&disks[4])]);
        assert_eq!(given[0].name, "SanDisk Cruzer Blade, 32 GB");
        let r = rows(&given, &disks);
        assert!(r[3].chosen.is_some());
        assert!(toggle(&given, &r[3], &[]).unwrap().is_empty());
        // A loop device is named by its file in /dev, which a reboot may change; any other disk
        // with no link in by-id is refused, since its name could lead to another disk.
        assert_eq!(refusal(&disks[1], &[]), None);
        assert!(
            r[0].notes(&[])
                .contains(&"its name may change at the host's next boot".to_string())
        );
        let vda = refusal(&disks[5], &[]).unwrap();
        assert!(vda.contains("has no link in /dev/disk/by-id"), "{vda}");
        assert!(
            r[4].notes(&[])
                .contains(&"no link in /dev/disk/by-id".to_string())
        );
        assert!(toggle(&[], &r[4], &[]).is_err());
        // A disk taken back from the host since it was given, as a snap's loop device mounted
        // since, still shows, so it can be taken back.
        let snap = vec![choice_of(&disks[0])];
        let r = rows(&snap, &disks);
        assert_eq!(r[0].disk.as_ref().unwrap().kernel, "loop1");
        assert!(toggle(&snap, &r[0], &[]).unwrap().is_empty());
    }

    /// A disk the VM has that is not attached has a row of its own, after the host's, which
    /// takes it back.
    #[cfg(unix)]
    #[test]
    fn a_disk_not_attached_shows_and_is_taken_back() {
        let (f, used) = host();
        let disks = read_disks(&f.sys(), &f.by_id(), &used);
        let away = HostDiskChoice {
            path: "/dev/disk/by-id/ata-Gone_Disk_123".into(),
            name: "Gone Disk, 500 GB".into(),
        };
        let r = rows(std::slice::from_ref(&away), &disks);
        let last = r.last().unwrap();
        assert_eq!(
            (last.disk.as_ref(), last.chosen.as_ref()),
            (None, Some(&away))
        );
        assert_eq!(last.name(), "Gone Disk, 500 GB");
        assert_eq!(last.notes(&[]), ["not attached"]);
        assert!(
            toggle(std::slice::from_ref(&away), last, &[])
                .unwrap()
                .is_empty()
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_command_line_names_a_disk_by_number_file_link_or_name() {
        let (f, used) = host();
        let disks = read_disks(&f.sys(), &f.by_id(), &used);
        let r = rows(&[], &disks);
        assert_eq!(find(&r, "1").unwrap(), 0);
        assert_eq!(find(&r, "/dev/loop7").unwrap(), 0);
        assert_eq!(find(&r, "loop7").unwrap(), 0);
        assert_eq!(find(&r, "sdb").unwrap(), 3);
        assert_eq!(
            find(&r, "usb-SanDisk_Cruzer_Blade_4C530001230512116021-0:0").unwrap(),
            3
        );
        assert_eq!(find(&r, "cruzer").unwrap(), 3);
        assert_eq!(find(&r, "WD40EFRX").unwrap(), 2);
        let e = format!("{:#}", find(&r, "s").unwrap_err());
        assert!(e.contains("several"), "{e}");
        // Not the disk whose model is WD40EFRX-68N.
        let e = format!("{:#}", find(&r, "6").unwrap_err());
        assert!(e.contains("no disk 6"), "{e}");
        assert!(find(&r, "/dev/sdz").is_err());
    }

    /// The rule that gives one disk to its user matches it by its link in by-id, after
    /// systemd's rules make the link, in a file of its own; a loop device, which has no link,
    /// is given by its file's owner.
    #[test]
    fn a_disk_is_given_to_its_user_alone_by_a_udev_rule() {
        let (rule, file) = udev_rule(
            Path::new("/dev/disk/by-id/usb-SanDisk_Cruzer_Blade_4C530001230512116021-0:0"),
            "u",
        );
        assert_eq!(
            rule,
            "SUBSYSTEM==\"block\", ENV{DEVTYPE}==\"disk\", \
             SYMLINK==\"disk/by-id/usb-SanDisk_Cruzer_Blade_4C530001230512116021-0:0\", OWNER=\"u\""
        );
        assert_eq!(
            file,
            "/etc/udev/rules.d/71-reims-ctl-disk-usb-SanDisk_Cruzer_Blade_4C530001230512116021-0_0.rules"
        );
        let d = HostDisk {
            kernel: "sdb".into(),
            path: "/dev/disk/by-id/ata-X_1".into(),
            bytes: 1,
            name: "X".into(),
            loop_file: None,
            devs: vec![(8, 16)],
            uses: vec![],
        };
        let f = fix(&d);
        assert!(f.contains("SYMLINK==\"disk/by-id/ata-X_1\""), "{f}");
        assert!(
            f.contains("sudo tee /etc/udev/rules.d/71-reims-ctl-disk-ata-X_1.rules"),
            "{f}"
        );
        assert!(
            f.contains("udevadm trigger --action=change /dev/disk/by-id/ata-X_1"),
            "{f}"
        );
        assert!(!f.contains("usermod"), "{f}");
        let l = HostDisk {
            kernel: "loop7".into(),
            path: "/dev/loop7".into(),
            loop_file: Some("/srv/d.img".into()),
            ..d
        };
        assert!(fix(&l).contains(" /dev/loop7"), "{}", fix(&l));
    }

    #[test]
    fn sizes_are_as_disks_are_sold() {
        assert_eq!(size(500_107_862_016), "500.1 GB");
        assert_eq!(size(4_000_787_030_016), "4 TB");
        assert_eq!(size(1_073_741_824), "1.1 GB");
    }

    #[test]
    fn names_from_the_controllers_strings() {
        let s = |x: &str| Some(x.to_string());
        assert_eq!(
            model(&s("ATA"), &s("WDC WD40EFRX-68N"), &None),
            "WDC WD40EFRX-68N"
        );
        assert_eq!(
            model(&s("SanDisk"), &s("Cruzer Blade"), &None),
            "SanDisk Cruzer Blade"
        );
        assert_eq!(
            model(&s("Samsung"), &s("Samsung SSD 870"), &None),
            "Samsung SSD 870"
        );
        assert_eq!(model(&None, &None, &s("SD64G")), "SD64G");
        assert_eq!(model(&None, &None, &None), "disk");
    }
}
