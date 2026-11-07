//! Which macOS an x86-64 VM runs, and the graphics that follows from it.
//!
//! reims-vgpu's own GPU device has a driver only in the guest's kexts reims-ctl carries
//! ([`crate::prebuilt`]), which OpenCore loads on the Darwin versions their MinKernel and MaxKernel
//! bound: macOS 26.4 and later releases of macOS 26. On an older macOS the device has no driver,
//! and the guest no GPU, where Apple's own driver in the guest drives apple-gfx. So a VM whose
//! settings name no graphics runs the device for the macOS it holds ([`graphics_for`]):
//! - at create, the macOS of the installer reims-util downloaded, or of the disk given;
//! - else at the first start that finds macOS on its disk ([`settle`]), which saves it by name, so
//!   that it no longer changes with the guest: an update of the guest to macOS 26.4 leaves it on
//!   apple-gfx, which still runs it, until its settings choose reims-vgpu.
//!
//! The disk tells its macOS through its APFS container's Preboot volume, which holds for each
//! install, in a directory named by its volume group's UUID (before volume groups, by its system
//! volume's), `System/Library/CoreServices/SystemVersion.plist`, a copy of the system's. Read from
//! an x86-64 macOS 26.7.2 (25G332) guest's disk: the file is 604 bytes, uncompressed, and its
//! directory is named by the volume group's UUID; the system volume itself is sealed, its tree
//! hashed. The reader follows Apple's File System Reference: the latest checkpoint's container
//! superblock, the container's object map, each volume's superblock, and the Preboot volume's
//! object map and file-system tree. It reads what macOS writes there and nothing else: no
//! encrypted volume, no compressed file, and no checkpoint area kept as a tree.

use crate::host::Profile;
use crate::opencore::{Disk, le16, le32, le64};
use crate::prebuilt;
use crate::vm::{Graphics, Vm};
use anyhow::{Context, Result, bail, ensure};
use std::collections::BTreeSet;
use std::path::Path;

/// A macOS release.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Macos {
    /// Its version, "26.7.2"; for an online installer only its macOS's number, "26", since its
    /// Reinstall macOS installs the newest release of it.
    pub version: String,
    /// Its build, "25G332", where known.
    pub build: Option<String>,
}

impl Macos {
    /// The Darwin version it runs, major and minor. A build names it: its number is Darwin's
    /// major version, and its letter the minor, A for 0 (25G332 runs Darwin 25.6). Without one,
    /// as an online installer's, only the major follows, from the macOS's number (macOS 26 and
    /// later run Darwin one below it, 11 to 15 nine above, and 10.Y Y + 4), and the minor is
    /// taken as 99: the newest release, as OpenCore's bound 25.99.99 stands for every release of
    /// Darwin 25.
    pub fn darwin(&self) -> Option<(u32, u32)> {
        if let Some(d) = self.build.as_deref().and_then(build_darwin) {
            return Some(d);
        }
        let mut parts = self.version.trim().split('.');
        let major: u32 = parts.next()?.parse().ok()?;
        let darwin = match major {
            10 => parts.next()?.parse::<u32>().ok()? + 4,
            11..=15 => major + 9,
            26.. => major - 1,
            _ => return None,
        };
        Some((darwin, 99))
    }

    /// What to call it: "macOS 26.7.2 (25G332)".
    pub fn describe(&self) -> String {
        match &self.build {
            Some(b) => format!("macOS {} ({b})", self.version),
            None => format!("macOS {}", self.version),
        }
    }
}

/// The Darwin version a build names: 25G332 is 25.6.
fn build_darwin(build: &str) -> Option<(u32, u32)> {
    let digits = build.bytes().take_while(u8::is_ascii_digit).count();
    let major = build[..digits].parse().ok()?;
    let letter = *build.as_bytes().get(digits)?;
    letter
        .is_ascii_uppercase()
        .then(|| (major, u32::from(letter - b'A')))
}

/// The macOS release that runs Darwin `major`.`minor`, as [`Macos::darwin`] maps them: "26.4"
/// for 25.4.
fn release_of(major: u32, minor: u32) -> String {
    match major {
        25.. => format!("{}.{minor}", major + 1),
        20..=24 => format!("{}.{minor}", major - 9),
        _ => format!("10.{}", major.saturating_sub(4)),
    }
}

/// The graphics a VM whose settings name none runs, for `found`, each macOS its disk holds or the
/// one its installer installs: reims-vgpu's own device where the guest's kexts load on each,
/// apple-gfx where they don't load on one, and None where nothing is found, or reims-ctl carries
/// no kexts.
pub fn graphics_for(found: &[Macos]) -> Option<Graphics> {
    if found.is_empty() {
        return None;
    }
    let mut runs = true;
    for m in found {
        runs &= prebuilt::loads_on(m.darwin()?)?;
    }
    Some(if runs {
        Graphics::ReimsVgpu
    } else {
        Graphics::AppleGfx
    })
}

/// Why reims-vgpu's device has no driver on `m`, a macOS its kexts don't load on: "older than
/// macOS 26.4, the first release reims-vgpu's macOS drivers run on".
fn too_old_or_new(m: &Macos) -> String {
    let (min, max) = prebuilt::bounds().unwrap_or(((0, 0), (u32::MAX, u32::MAX)));
    match m.darwin() {
        Some(d) if d > max => format!(
            "newer than macOS {}, the last release reims-vgpu's macOS drivers run on",
            release_of(max.0, max.1).trim_end_matches(".99")
        ),
        _ => format!(
            "older than macOS {}, the first release reims-vgpu's macOS drivers run on",
            release_of(min.0, min.1)
        ),
    }
}

/// What a create that chose `graphics` for the VM `name` tells of it, after its name, where its
/// macOS, `found`, decided it, or where the reims-vgpu given has no driver on it.
pub fn create_note(name: &str, found: &[Macos], graphics: Graphics, given: bool) -> Option<String> {
    let old = found
        .iter()
        .find(|m| m.darwin().and_then(prebuilt::loads_on) == Some(false))?;
    match (graphics, given) {
        (Graphics::AppleGfx, false) => Some(format!(
            "{} is {}, so it runs on apple-gfx",
            old.describe(),
            too_old_or_new(old)
        )),
        (Graphics::ReimsVgpu, true) => Some(format!(
            "its {} is {}: on reims-vgpu macOS has no driver for its GPU, and draws without one. \
             To run it on apple-gfx, run reims-ctl set {name} --graphics apple-gfx.",
            old.describe(),
            too_old_or_new(old)
        )),
        _ => None,
    }
}

/// What to tell of a VM on reims-vgpu whose guest, `m`, has no driver for its device.
fn no_driver(name: &str, m: &Macos) -> String {
    format!(
        "{name} runs on reims-vgpu, but its {} is {}: macOS has no driver for its GPU, and draws \
         without one. To run it on apple-gfx, run reims-ctl set {name} --graphics apple-gfx.",
        m.describe(),
        too_old_or_new(m)
    )
}

/// What a start does first for a stopped x86-64 VM on reims-vgpu: the macOS its disk holds.
#[derive(Debug, PartialEq)]
pub enum Settled {
    /// Nothing to tell: another device, no macOS on the disk yet, or a macOS that runs the
    /// device.
    Kept,
    /// Its settings named no graphics, and now name the one for its macOS, apple-gfx: what to
    /// tell of it, after its name.
    Saved(String),
    /// A warning: the device has no driver on its macOS, which its settings chose; or reims-ctl
    /// can't read its disk, and it runs the default.
    Warn(String),
}

/// For a stopped x86-64 VM on reims-vgpu, the macOS its disk holds: where its settings name no
/// graphics, the device for it is saved in them by name ([`graphics_for`]), and where they name
/// reims-vgpu on a macOS it has no driver on, that is a warning. A disk reims-ctl can't read is a
/// warning where the VM runs the default.
pub fn settle(vm: &mut Vm) -> Result<Settled> {
    if vm.config.profile != Profile::Q35X86_64 || vm.config.graphics() != Graphics::ReimsVgpu {
        return Ok(Settled::Kept);
    }
    let name = vm.name().to_string();
    let found = match on_disk(&vm.disk(), vm.config.disk_format) {
        Ok(f) => f,
        Err(e) if vm.config.graphics.is_none() => {
            return Ok(Settled::Warn(format!(
                "reims-ctl can't tell which macOS {name}'s disk holds ({e:#}), so it runs on \
                 reims-vgpu, whose macOS drivers need macOS {} or later. If it has an older \
                 macOS, run reims-ctl set {name} --graphics apple-gfx.",
                prebuilt::bounds()
                    .map(|(min, _)| release_of(min.0, min.1))
                    .unwrap_or_default()
            )));
        }
        Err(_) => return Ok(Settled::Kept),
    };
    let Some(graphics) = graphics_for(&found) else {
        return Ok(Settled::Kept);
    };
    if vm.config.graphics.is_some() {
        return Ok(match graphics {
            Graphics::ReimsVgpu => Settled::Kept,
            _ => match found
                .iter()
                .find(|m| m.darwin().and_then(prebuilt::loads_on) == Some(false))
            {
                Some(m) => Settled::Warn(no_driver(&name, m)),
                None => Settled::Kept,
            },
        });
    }
    // Saved in the settings as they are now, which another reims-ctl may have changed.
    let mut now = Vm::load(&vm.dir)?;
    if now.config.graphics.is_some() {
        *vm = now;
        return Ok(Settled::Kept);
    }
    now.config.graphics = Some(graphics);
    now.save()?;
    *vm = now;
    Ok(match graphics {
        Graphics::ReimsVgpu => Settled::Kept,
        _ => {
            let m = found
                .iter()
                .find(|m| m.darwin().and_then(prebuilt::loads_on) == Some(false))
                .unwrap_or(&found[0]);
            Settled::Saved(format!(
                "its disk holds {}, which is {}, so it runs on apple-gfx, now saved in its \
                 settings. Once its macOS is updated to {} or later, reims-ctl set {name} \
                 --graphics reims-vgpu moves it to reims-vgpu's own GPU device.",
                m.describe(),
                too_old_or_new(m),
                prebuilt::bounds()
                    .map(|(min, _)| format!("macOS {}", release_of(min.0, min.1)))
                    .unwrap_or_default()
            ))
        }
    })
}

/// The VM as a start would run it: where its settings name no graphics, on the device
/// [`settle`] would save for the macOS its disk holds. For what `reims-ctl command`, `show` and
/// `list` print.
pub fn as_started(vm: &Vm) -> Vm {
    let mut v = vm.clone();
    if v.config.profile == Profile::Q35X86_64
        && v.config.graphics.is_none()
        && !crate::runtime::running(vm)
        && let Ok(found) = on_disk(&vm.disk(), vm.config.disk_format)
    {
        v.config.graphics = graphics_for(&found);
    }
    v
}

/// Each macOS the disk image at `path` holds: none where it has no GPT, no APFS container, or no
/// macOS installed on one yet.
pub fn on_disk(path: &Path, format: crate::vm::ImageFormat) -> Result<Vec<Macos>> {
    let disk = Disk::open(path, format)?;
    let mut found = Vec::new();
    for start in crate::opencore::partitions(&disk, &APFS_TYPE)? {
        found.extend(Container::open(&disk, start)?.macos()?);
    }
    Ok(found)
}

/// An APFS partition's type GUID, 7C3457EF-0000-11AA-AA11-00306543ECAC, as GPT stores it.
const APFS_TYPE: [u8; 16] = [
    0xef, 0x57, 0x34, 0x7c, 0x00, 0x00, 0xaa, 0x11, 0xaa, 0x11, 0x00, 0x30, 0x65, 0x43, 0xec, 0xac,
];

const NX_MAGIC: &[u8; 4] = b"NXSB";
const APFS_MAGIC: &[u8; 4] = b"APSB";
/// `o_type`'s low 16 bits.
const OBJECT_TYPE_NX_SUPERBLOCK: u32 = 0x01;
const OBJECT_TYPE_BTREE: u32 = 0x02;
const OBJECT_TYPE_BTREE_NODE: u32 = 0x03;
const OBJECT_TYPE_OMAP: u32 = 0x0b;
const OBJECT_TYPE_FS: u32 = 0x0d;
/// `o_type`'s flag of an object found by its address, not through an object map.
const OBJ_PHYSICAL: u32 = 0x4000_0000;
const BTNODE_ROOT: u16 = 1;
const BTNODE_FIXED_KV_SIZE: u16 = 4;
/// `bt_flags`: child nodes found by their addresses.
const BTREE_PHYSICAL: u32 = 0x10;
/// The size of `btree_info_t`, at the end of a root node.
const BTREE_INFO: usize = 40;
/// `ov_flags`.
const OMAP_VAL_DELETED: u32 = 1;
const OMAP_VAL_ENCRYPTED: u32 = 4;
/// `apfs_incompatible_features`: file names compared without regard to case, or to Unicode
/// normalization, whose directory entries' keys hold a hash of the name.
const APFS_INCOMPAT_CASE_INSENSITIVE: u64 = 1;
const APFS_INCOMPAT_NORMALIZATION_INSENSITIVE: u64 = 8;
/// `apfs_fs_flags`.
const APFS_FS_UNENCRYPTED: u64 = 1;
const APFS_VOL_ROLE_PREBOOT: u16 = 0x10;
/// A file-system record's type, the top four bits of its key's first word.
const APFS_TYPE_INODE: u8 = 3;
const APFS_TYPE_FILE_EXTENT: u8 = 8;
const APFS_TYPE_DIR_REC: u8 = 9;
const OBJ_ID_MASK: u64 = 0x0fff_ffff_ffff_ffff;
/// The root directory's inode.
const ROOT_DIR_INO_NUM: u64 = 2;
/// An inode's extended field holding its data stream (`j_dstream_t`).
const INO_EXT_TYPE_DSTREAM: u8 = 8;
/// `bsd_flags`: the file's data is compressed (in its extended attributes).
const UF_COMPRESSED: u32 = 0x20;

/// The most nodes a lookup reads, past which a tree is taken as damaged.
const MOST_NODES: usize = 4096;
/// The most a file read is: SystemVersion.plist is under 1 KiB.
const MOST_FILE: u64 = 1 << 20;

/// An APFS container.
struct Container<'a> {
    disk: &'a Disk,
    start: u64,
    block: u64,
    /// The latest checkpoint's transaction.
    xid: u64,
    /// Its superblock.
    nx: Vec<u8>,
}

impl<'a> Container<'a> {
    fn open(disk: &'a Disk, start: u64) -> Result<Container<'a>> {
        let mut first = vec![0u8; 4096];
        disk.read(start, &mut first)?;
        ensure!(
            &first[32..36] == NX_MAGIC,
            "an APFS partition with no container"
        );
        let block = u64::from(le32(&first, 36));
        ensure!(
            (4096..=65536).contains(&block) && block.is_power_of_two(),
            "APFS blocks of {block} bytes"
        );
        let mut c = Container {
            disk,
            start,
            block,
            xid: 0,
            nx: Vec::new(),
        };
        // The latest valid superblock of the checkpoint descriptor area, else block 0's.
        let mut nx = c.block_at(0)?;
        let desc_blocks = le32(&nx, 104);
        if desc_blocks & 0x8000_0000 == 0 {
            let base = le64(&nx, 112);
            for i in 0..u64::from(desc_blocks.min(1 << 16)) {
                let Ok(b) = c.block_at(base + i) else {
                    continue;
                };
                if &b[32..36] == NX_MAGIC
                    && le32(&b, 24) & 0xffff == OBJECT_TYPE_NX_SUPERBLOCK
                    && checksum_ok(&b)
                    && le64(&b, 16) > le64(&nx, 16)
                {
                    nx = b;
                }
            }
        }
        ensure!(
            checksum_ok(&nx),
            "the APFS container's superblock is damaged"
        );
        c.xid = le64(&nx, 16);
        c.nx = nx;
        Ok(c)
    }

    /// Block `n` of the container.
    fn block_at(&self, n: u64) -> Result<Vec<u8>> {
        let mut b = vec![0u8; self.block as usize];
        let at = n
            .checked_mul(self.block)
            .and_then(|o| o.checked_add(self.start))
            .context("an APFS block past the disk's end")?;
        self.disk.read(at, &mut b)?;
        Ok(b)
    }

    /// The object at block `n`, checked by its checksum and its type's low bits.
    fn object(&self, n: u64, types: &[u32]) -> Result<Vec<u8>> {
        let b = self.block_at(n)?;
        ensure!(checksum_ok(&b), "a damaged APFS object at block {n}");
        ensure!(
            types.contains(&(le32(&b, 24) & 0xffff)),
            "an APFS object of type {:#x} at block {n}",
            le32(&b, 24)
        );
        Ok(b)
    }

    /// Where the object map at block `omap` puts virtual object `oid` at the latest checkpoint.
    fn resolve(&self, omap: u64, oid: u64) -> Result<u64> {
        let om = self.object(omap, &[OBJECT_TYPE_OMAP])?;
        let mut node = self.object(le64(&om, 48), &[OBJECT_TYPE_BTREE, OBJECT_TYPE_BTREE_NODE])?;
        for _ in 0..16 {
            let n = Node::parse(&node, self.block as usize)?;
            let mut hit = None;
            for i in 0..n.count {
                let (k, v) = n.entry(i, (16, 16))?;
                ensure!(k.len() >= 16 && v.len() >= 8, "a short object map entry");
                if (le64(k, 0), le64(k, 8)) <= (oid, self.xid) {
                    hit = Some((le64(k, 0), v));
                }
            }
            let Some((key, v)) = hit else { break };
            if n.level == 0 {
                ensure!(key == oid, "the object map has no object {oid}");
                ensure!(v.len() >= 16, "a short object map value");
                let flags = le32(v, 0);
                ensure!(flags & OMAP_VAL_DELETED == 0, "object {oid} is deleted");
                ensure!(flags & OMAP_VAL_ENCRYPTED == 0, "object {oid} is encrypted");
                return Ok(le64(v, 8));
            }
            node = self.object(le64(v, 0), &[OBJECT_TYPE_BTREE, OBJECT_TYPE_BTREE_NODE])?;
        }
        bail!("the object map has no object {oid}")
    }

    /// Each macOS its Preboot volumes hold, for an install whose volumes are in the container.
    fn macos(&self) -> Result<Vec<Macos>> {
        let omap = le64(&self.nx, 160);
        let count = (le32(&self.nx, 180) as usize).min(100);
        let mut volumes = Vec::new();
        for i in 0..count {
            let oid = le64(&self.nx, 184 + i * 8);
            if oid == 0 {
                continue;
            }
            let at = self.resolve(omap, oid)?;
            let sb = self.object(at, &[OBJECT_TYPE_FS])?;
            ensure!(
                &sb[32..36] == APFS_MAGIC,
                "an APFS volume with no superblock"
            );
            volumes.push(sb);
        }
        // The UUIDs an install's Preboot directory may be named by.
        let ids: BTreeSet<String> = volumes
            .iter()
            .flat_map(|sb| [uuid(&sb[240..256]), uuid(&sb[1008..1024])])
            .filter(|u| u != "00000000-0000-0000-0000-000000000000")
            .collect();
        let mut found = Vec::new();
        for sb in volumes.iter().filter(|sb| {
            le16(sb, 964) == APFS_VOL_ROLE_PREBOOT && le64(sb, 264) & APFS_FS_UNENCRYPTED != 0
        }) {
            let fs = Volume::open(self, sb)?;
            for (name, dir) in fs.children(ROOT_DIR_INO_NUM)? {
                if !ids.contains(&name.to_ascii_uppercase()) {
                    continue;
                }
                let mut at = Some(dir);
                for part in ["System", "Library", "CoreServices", "SystemVersion.plist"] {
                    at = match at {
                        Some(d) => fs.child(d, part)?,
                        None => break,
                    };
                }
                if let Some(file) = at {
                    found.push(parse_version(&fs.read(file)?)?);
                }
            }
        }
        Ok(found)
    }
}

/// A UUID's 16 bytes as text, as a Preboot directory names it: upper case, in groups.
fn uuid(b: &[u8]) -> String {
    let hex: String = b.iter().map(|x| format!("{x:02X}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..]
    )
}

/// SystemVersion.plist's ProductVersion and ProductBuildVersion.
fn parse_version(bytes: &[u8]) -> Result<Macos> {
    let v = plist::Value::from_reader(std::io::Cursor::new(bytes))
        .context("read SystemVersion.plist")?;
    let d = v
        .as_dictionary()
        .context("SystemVersion.plist holds no dictionary")?;
    let text = |k: &str| d.get(k).and_then(plist::Value::as_string).map(String::from);
    Ok(Macos {
        version: text("ProductVersion").context("SystemVersion.plist has no ProductVersion")?,
        build: text("ProductBuildVersion"),
    })
}

/// A B-tree node: its entries' table of contents, keys and values.
struct Node<'a> {
    b: &'a [u8],
    level: u16,
    count: usize,
    fixed: bool,
    toc: usize,
    keys: usize,
    values_end: usize,
}

impl<'a> Node<'a> {
    fn parse(b: &'a [u8], size: usize) -> Result<Node<'a>> {
        let flags = le16(b, 32);
        let toc = 56 + usize::from(le16(b, 40));
        let keys = toc + usize::from(le16(b, 42));
        let values_end = if flags & BTNODE_ROOT != 0 {
            size - BTREE_INFO
        } else {
            size
        };
        ensure!(keys <= values_end, "a damaged APFS B-tree node");
        Ok(Node {
            b,
            level: le16(b, 34),
            count: le32(b, 36) as usize,
            fixed: flags & BTNODE_FIXED_KV_SIZE != 0,
            toc,
            keys,
            values_end,
        })
    }

    /// Entry `i`'s key and value; `fixed` gives their sizes in a node of fixed-size entries. A
    /// value whose offset is 0xffff, none, is empty.
    fn entry(&self, i: usize, fixed: (usize, usize)) -> Result<(&'a [u8], &'a [u8])> {
        let (ko, kl, vo, vl) = if self.fixed {
            let e = self.toc + i * 4;
            (le16(self.b, e), fixed.0, le16(self.b, e + 2), fixed.1)
        } else {
            let e = self.toc + i * 8;
            (
                le16(self.b, e),
                usize::from(le16(self.b, e + 2)),
                le16(self.b, e + 4),
                usize::from(le16(self.b, e + 6)),
            )
        };
        let k = self.keys + usize::from(ko);
        let key = self
            .b
            .get(k..k + kl)
            .filter(|_| k + kl <= self.values_end)
            .context("a damaged APFS B-tree entry")?;
        if vo == 0xffff {
            return Ok((key, &[]));
        }
        // A non-leaf node's value is its child's object identifier, whatever size the tree's
        // values are.
        let vl = if self.level > 0 { 8 } else { vl };
        let v = self
            .values_end
            .checked_sub(usize::from(vo))
            .context("a damaged APFS B-tree entry")?;
        let value = self
            .b
            .get(v..v + vl)
            .context("a damaged APFS B-tree entry")?;
        Ok((key, value))
    }
}

/// A file-system record: its type, key and value.
type Record = (u8, Vec<u8>, Vec<u8>);

/// A volume's file-system tree.
struct Volume<'a, 'c> {
    c: &'c Container<'a>,
    omap: u64,
    root: u64,
    /// Whether the root, and the child nodes, are found by their addresses.
    root_physical: bool,
    children_physical: bool,
    hashed: bool,
    case_insensitive: bool,
}

impl<'a, 'c> Volume<'a, 'c> {
    fn open(c: &'c Container<'a>, sb: &[u8]) -> Result<Volume<'a, 'c>> {
        let incompat = le64(sb, 56);
        let mut v = Volume {
            c,
            omap: le64(sb, 128),
            root: le64(sb, 136),
            root_physical: le32(sb, 116) & OBJ_PHYSICAL != 0,
            children_physical: false,
            hashed: incompat
                & (APFS_INCOMPAT_CASE_INSENSITIVE | APFS_INCOMPAT_NORMALIZATION_INSENSITIVE)
                != 0,
            case_insensitive: incompat & APFS_INCOMPAT_CASE_INSENSITIVE != 0,
        };
        let root = v.node(v.root, v.root_physical)?;
        let size = c.block as usize;
        v.children_physical = le32(&root, size - BTREE_INFO) & BTREE_PHYSICAL != 0;
        Ok(v)
    }

    fn node(&self, oid: u64, physical: bool) -> Result<Vec<u8>> {
        let at = if physical {
            oid
        } else {
            self.c.resolve(self.omap, oid)?
        };
        self.c
            .object(at, &[OBJECT_TYPE_BTREE, OBJECT_TYPE_BTREE_NODE])
    }

    /// Every record of object `id`, in the tree's order: its type, key and value.
    fn records(&self, id: u64) -> Result<Vec<Record>> {
        let mut out = Vec::new();
        let mut read = 0;
        self.walk(
            self.node(self.root, self.root_physical)?,
            id,
            &mut out,
            &mut read,
            0,
        )?;
        Ok(out)
    }

    fn walk(
        &self,
        node: Vec<u8>,
        id: u64,
        out: &mut Vec<Record>,
        read: &mut usize,
        depth: u32,
    ) -> Result<()> {
        *read += 1;
        ensure!(
            *read <= MOST_NODES && depth < 16,
            "a damaged APFS file-system tree"
        );
        let n = Node::parse(&node, self.c.block as usize)?;
        let mut entries = Vec::with_capacity(n.count);
        for i in 0..n.count {
            let (k, v) = n.entry(i, (0, 0))?;
            ensure!(k.len() >= 8, "a short APFS file-system key");
            entries.push((le64(k, 0), k, v));
        }
        if n.level == 0 {
            for (word, k, v) in entries {
                if word & OBJ_ID_MASK == id {
                    out.push(((word >> 60) as u8, k.to_vec(), v.to_vec()));
                }
            }
            return Ok(());
        }
        // Child i holds the keys from its own to the next child's: id's records may span
        // several.
        for i in 0..entries.len() {
            let first = entries[i].0 & OBJ_ID_MASK;
            let next = entries.get(i + 1).map(|e| e.0 & OBJ_ID_MASK);
            if first <= id && next.is_none_or(|n| n >= id) {
                ensure!(entries[i].2.len() >= 8, "a short APFS B-tree pointer");
                let child = self.node(le64(entries[i].2, 0), self.children_physical)?;
                self.walk(child, id, out, read, depth + 1)?;
            }
        }
        Ok(())
    }

    /// The entries of directory `dir`: each name and the inode it names.
    fn children(&self, dir: u64) -> Result<Vec<(String, u64)>> {
        let mut out = Vec::new();
        for (t, k, v) in self.records(dir)? {
            if t != APFS_TYPE_DIR_REC || v.len() < 8 {
                continue;
            }
            let (start, len) = if self.hashed {
                ensure!(k.len() >= 12, "a short APFS directory entry");
                (12, (le32(&k, 8) & 0x3ff) as usize)
            } else {
                ensure!(k.len() >= 10, "a short APFS directory entry");
                (10, usize::from(le16(&k, 8)))
            };
            // The length counts the name's terminating NUL.
            let name = k
                .get(start..start + len.saturating_sub(1))
                .context("a short APFS directory entry")?;
            out.push((String::from_utf8_lossy(name).into_owned(), le64(&v, 0)));
        }
        Ok(out)
    }

    /// The inode `name` names in directory `dir`, matched as the volume compares names.
    fn child(&self, dir: u64, name: &str) -> Result<Option<u64>> {
        Ok(self
            .children(dir)?
            .into_iter()
            .find(|(n, _)| match self.case_insensitive {
                true => n.eq_ignore_ascii_case(name),
                false => n == name,
            })
            .map(|(_, id)| id))
    }

    /// The bytes of file `id`, uncompressed, at most [`MOST_FILE`].
    fn read(&self, id: u64) -> Result<Vec<u8>> {
        let records = self.records(id)?;
        let inode = records
            .iter()
            .find(|(t, ..)| *t == APFS_TYPE_INODE)
            .map(|(_, _, v)| v)
            .context("a file with no inode")?;
        ensure!(inode.len() >= 96, "a short APFS inode");
        ensure!(
            le32(inode, 68) & UF_COMPRESSED == 0,
            "SystemVersion.plist is compressed"
        );
        let size = dstream_size(inode).unwrap_or(0).min(MOST_FILE);
        let mut extents: Vec<(u64, u64, u64)> = Vec::new();
        for (t, k, v) in self.records(le64(inode, 8))? {
            if t == APFS_TYPE_FILE_EXTENT && k.len() >= 16 && v.len() >= 16 {
                extents.push((
                    le64(&k, 8),
                    le64(&v, 0) & 0x00ff_ffff_ffff_ffff,
                    le64(&v, 8),
                ));
            }
        }
        extents.sort();
        let mut data = vec![0u8; size as usize];
        for (logical, len, block) in extents {
            if logical >= size || block == 0 {
                continue;
            }
            let n = len.min(size - logical) as usize;
            let at = block
                .checked_mul(self.c.block)
                .and_then(|o| o.checked_add(self.c.start))
                .context("a file extent past the disk's end")?;
            self.c
                .disk
                .read(at, &mut data[logical as usize..logical as usize + n])?;
        }
        Ok(data)
    }
}

/// The size of the data stream an inode's extended fields hold.
fn dstream_size(inode: &[u8]) -> Option<u64> {
    let n = usize::from(le16(inode, 92));
    let mut data = 96 + n * 4;
    for i in 0..n {
        let e = 96 + i * 4;
        let (kind, size) = (
            *inode.get(e)?,
            usize::from(le16(inode.get(..e + 4)?, e + 2)),
        );
        if kind == INO_EXT_TYPE_DSTREAM {
            return inode.get(data..data + 8).map(|b| le64(b, 0));
        }
        data += size.next_multiple_of(8);
    }
    None
}

/// Whether an APFS object's Fletcher-64 checksum, its first 8 bytes, is that of the rest.
fn checksum_ok(b: &[u8]) -> bool {
    const M: u64 = 0xffff_ffff;
    let (mut s1, mut s2) = (0u64, 0u64);
    for w in b[8..].as_chunks::<4>().0 {
        s1 = (s1 + u64::from(u32::from_le_bytes(*w))) % M;
        s2 = (s2 + s1) % M;
    }
    let c1 = M - (s1 + s2) % M;
    let c2 = M - (s1 + c1) % M;
    u64::from(le32(b, 0)) == c1 && u64::from(le32(b, 4)) == c2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::ImageFormat;

    const BS: usize = 4096;

    /// Writes the checksum of `b`, an object, into its first 8 bytes.
    fn seal(b: &mut [u8]) {
        const M: u64 = 0xffff_ffff;
        let (mut s1, mut s2) = (0u64, 0u64);
        for w in b[8..].as_chunks::<4>().0 {
            s1 = (s1 + u64::from(u32::from_le_bytes(*w))) % M;
            s2 = (s2 + s1) % M;
        }
        let c1 = M - (s1 + s2) % M;
        let c2 = M - (s1 + c1) % M;
        b[..4].copy_from_slice(&(c1 as u32).to_le_bytes());
        b[4..8].copy_from_slice(&(c2 as u32).to_le_bytes());
    }

    fn put(b: &mut [u8], at: usize, bytes: &[u8]) {
        b[at..at + bytes.len()].copy_from_slice(bytes);
    }

    /// An object of type `kind` at `xid`.
    fn object(kind: u32, oid: u64, xid: u64) -> Vec<u8> {
        let mut b = vec![0u8; BS];
        put(&mut b, 8, &oid.to_le_bytes());
        put(&mut b, 16, &xid.to_le_bytes());
        put(&mut b, 24, &kind.to_le_bytes());
        b
    }

    /// A root leaf B-tree node of `entries`, its keys and values; of fixed 16-byte ones where
    /// `fixed`.
    fn leaf(entries: &[(Vec<u8>, Vec<u8>)], fixed: bool, oid: u64) -> Vec<u8> {
        let mut b = object(OBJECT_TYPE_BTREE | OBJ_PHYSICAL, oid, 1);
        let flags = BTNODE_ROOT | 2 | if fixed { BTNODE_FIXED_KV_SIZE } else { 0 };
        put(&mut b, 32, &flags.to_le_bytes());
        put(&mut b, 36, &(entries.len() as u32).to_le_bytes());
        let toc_len = entries.len() * if fixed { 4 } else { 8 };
        put(&mut b, 42, &(toc_len as u16).to_le_bytes());
        let keys = 56 + toc_len;
        let end = BS - BTREE_INFO;
        let (mut ko, mut vo) = (0usize, 0usize);
        for (i, (k, v)) in entries.iter().enumerate() {
            vo += v.len();
            put(&mut b, keys + ko, k);
            put(&mut b, end - vo, v);
            if fixed {
                put(&mut b, 56 + i * 4, &(ko as u16).to_le_bytes());
                put(&mut b, 56 + i * 4 + 2, &(vo as u16).to_le_bytes());
            } else {
                let toc = [ko as u16, k.len() as u16, vo as u16, v.len() as u16];
                for (j, x) in toc.iter().enumerate() {
                    put(&mut b, 56 + i * 8 + j * 2, &x.to_le_bytes());
                }
            }
            ko += k.len();
        }
        put(&mut b, BS - 36, &(BS as u32).to_le_bytes());
        b
    }

    /// An object map at block `at` whose tree, at `at + 1`, maps each (oid, block).
    fn omap(at: u64, map: &[(u64, u64)]) -> [Vec<u8>; 2] {
        let mut om = object(OBJECT_TYPE_OMAP | OBJ_PHYSICAL, at, 1);
        put(&mut om, 48, &(at + 1).to_le_bytes());
        let entries: Vec<_> = map
            .iter()
            .map(|&(oid, block)| {
                let k = [oid.to_le_bytes(), 1u64.to_le_bytes()].concat();
                let v = [0u32.to_le_bytes(), (BS as u32).to_le_bytes()].concat();
                (k, [v, block.to_le_bytes().to_vec()].concat())
            })
            .collect();
        [om, leaf(&entries, true, at + 1)]
    }

    fn key(id: u64, kind: u8) -> Vec<u8> {
        (id | u64::from(kind) << 60).to_le_bytes().to_vec()
    }

    /// A directory entry of `dir` naming `name`, inode `id`, with a hashed key.
    fn drec(dir: u64, name: &str, id: u64) -> (Vec<u8>, Vec<u8>) {
        let mut k = key(dir, APFS_TYPE_DIR_REC);
        k.extend(((name.len() as u32 + 1) | 0xabc << 10).to_le_bytes());
        k.extend(name.as_bytes());
        k.push(0);
        (
            k,
            [id.to_le_bytes(), [0; 8], [0; 8]].concat()[..18].to_vec(),
        )
    }

    /// A disk holding a GPT, one APFS partition from sector 40, and in it a Preboot volume
    /// holding each of `installs`, a directory name and SystemVersion.plist's bytes, beside a
    /// data volume whose group is `group`.
    fn disk(installs: &[(&str, &[u8])], group: [u8; 16]) -> Vec<u8> {
        const START: usize = 40 * 512;
        // Blocks: 0 the superblock, 1-2 the container's object map, 3 the Preboot volume, 4-5
        // its object map, 6 its tree, 7 the data volume, 8.. the files.
        let mut blocks: Vec<Vec<u8>> = vec![Vec::new(); 8];
        let mut nx = object(OBJECT_TYPE_NX_SUPERBLOCK | OBJ_PHYSICAL, 1, 1);
        put(&mut nx, 32, NX_MAGIC);
        put(&mut nx, 36, &(BS as u32).to_le_bytes());
        put(&mut nx, 160, &1u64.to_le_bytes());
        put(&mut nx, 180, &100u32.to_le_bytes());
        put(&mut nx, 184, &1026u64.to_le_bytes());
        put(&mut nx, 192, &1027u64.to_le_bytes());
        blocks[0] = nx;
        let [om, omt] = omap(1, &[(1026, 3), (1027, 7)]);
        (blocks[1], blocks[2]) = (om, omt);
        let mut sb = object(OBJECT_TYPE_FS, 1026, 1);
        put(&mut sb, 32, APFS_MAGIC);
        put(&mut sb, 56, &APFS_INCOMPAT_CASE_INSENSITIVE.to_le_bytes());
        put(&mut sb, 116, &OBJECT_TYPE_BTREE.to_le_bytes());
        put(&mut sb, 128, &4u64.to_le_bytes());
        put(&mut sb, 136, &1028u64.to_le_bytes());
        put(&mut sb, 264, &APFS_FS_UNENCRYPTED.to_le_bytes());
        put(&mut sb, 964, &APFS_VOL_ROLE_PREBOOT.to_le_bytes());
        blocks[3] = sb;
        let [om, omt] = omap(4, &[(1028, 6)]);
        (blocks[4], blocks[5]) = (om, omt);
        let mut data = object(OBJECT_TYPE_FS, 1027, 1);
        put(&mut data, 32, APFS_MAGIC);
        put(&mut data, 1008, &group);
        blocks[7] = data;
        // The tree: the root directory, then for each install its directories and file.
        let mut records = Vec::new();
        let mut next = 16;
        for (dir, plist) in installs {
            let ids: Vec<u64> = (next..next + 5).collect();
            next += 5;
            records.push(drec(ROOT_DIR_INO_NUM, dir, ids[0]));
            for (i, part) in ["System", "Library", "CoreServices", "SystemVersion.plist"]
                .iter()
                .enumerate()
            {
                records.push(drec(ids[i], part, ids[i + 1]));
            }
            let file = ids[4];
            let mut inode = vec![0u8; 96 + 4 + 40];
            put(&mut inode, 8, &(file + 100).to_le_bytes());
            put(&mut inode, 92, &1u16.to_le_bytes());
            put(&mut inode, 94, &40u16.to_le_bytes());
            inode[96] = INO_EXT_TYPE_DSTREAM;
            put(&mut inode, 98, &40u16.to_le_bytes());
            put(&mut inode, 100, &(plist.len() as u64).to_le_bytes());
            records.push((key(file, APFS_TYPE_INODE), inode));
            let block = blocks.len() as u64;
            let mut contents = plist.to_vec();
            contents.resize(BS, 0);
            blocks.push(contents);
            let ext = [(BS as u64).to_le_bytes(), block.to_le_bytes(), [0; 8]].concat();
            let ek = [
                key(file + 100, APFS_TYPE_FILE_EXTENT),
                0u64.to_le_bytes().to_vec(),
            ];
            records.push((ek.concat(), ext));
        }
        records.sort_by(|a, b| {
            let id = |k: &[u8]| (le64(k, 0) & OBJ_ID_MASK, le64(k, 0) >> 60);
            id(&a.0).cmp(&id(&b.0))
        });
        let mut tree = leaf(&records, false, 1028);
        put(&mut tree, 24, &OBJECT_TYPE_BTREE.to_le_bytes());
        blocks[6] = tree;
        for b in blocks.iter_mut().take(8) {
            seal(b);
        }
        let mut img = vec![0u8; START + blocks.len() * BS];
        put(&mut img, 512, b"EFI PART");
        put(&mut img, 512 + 72, &2u64.to_le_bytes());
        put(&mut img, 512 + 80, &1u32.to_le_bytes());
        put(&mut img, 512 + 84, &128u32.to_le_bytes());
        put(&mut img, 1024, &APFS_TYPE);
        put(&mut img, 1024 + 32, &40u64.to_le_bytes());
        for (i, b) in blocks.iter().enumerate() {
            put(&mut img, START + i * BS, b);
        }
        img
    }

    fn system_version(version: &str, build: &str) -> Vec<u8> {
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0"><dict>
  <key>ProductBuildVersion</key><string>{build}</string>
  <key>ProductName</key><string>macOS</string>
  <key>ProductVersion</key><string>{version}</string>
</dict></plist>"#
        )
        .into_bytes()
    }

    const GROUP: [u8; 16] = [
        0xd3, 0x6e, 0x3c, 0x34, 0xbd, 0xaf, 0x44, 0xaa, 0xbe, 0x65, 0x17, 0xc5, 0xdd, 0x68, 0x6b,
        0xc1,
    ];

    fn read(img: &[u8], name: &str) -> Result<Vec<Macos>> {
        let d = std::env::temp_dir().join(format!("reims-ctl-macos-{name}-{}", std::process::id()));
        std::fs::write(&d, img).unwrap();
        let r = on_disk(&d, ImageFormat::Raw);
        std::fs::remove_file(&d).unwrap();
        r
    }

    /// The macOS of an install whose Preboot directory names its volume group, read from its
    /// SystemVersion.plist; a directory naming no volume of the container, as an install
    /// removed leaves, is none.
    #[test]
    fn a_disk_tells_the_macos_its_preboot_volume_holds() {
        let ours = system_version("15.6.1", "24G90");
        let stale = system_version("14.0", "23A344");
        let img = disk(
            &[
                ("D36E3C34-BDAF-44AA-BE65-17C5DD686BC1", &ours),
                ("00112233-4455-6677-8899-AABBCCDDEEFF", &stale),
            ],
            GROUP,
        );
        let found = read(&img, "one").unwrap();
        assert_eq!(
            found,
            [Macos {
                version: "15.6.1".into(),
                build: Some("24G90".into())
            }]
        );
        // A damaged object is an error, not a disk without macOS.
        let mut bad = img.clone();
        bad[40 * 512 + 6 * BS + 100] ^= 1;
        assert!(read(&bad, "bad").is_err());
    }

    /// A disk with no GPT, as a new one is, or no APFS container, holds no macOS.
    #[test]
    fn a_new_disk_holds_no_macos() {
        assert_eq!(read(&vec![0u8; 1 << 16], "empty").unwrap(), []);
        let mut img = disk(&[], GROUP);
        assert_eq!(read(&img, "no-install").unwrap(), []);
        img[1024] ^= 0xff;
        assert_eq!(read(&img, "no-apfs").unwrap(), []);
    }

    /// The Darwin version of a release: its build's number and letter, else its macOS's number's
    /// major and the newest minor.
    #[test]
    fn a_release_runs_the_darwin_version_its_build_names() {
        let m = |v: &str, b: Option<&str>| Macos {
            version: v.into(),
            build: b.map(String::from),
        };
        assert_eq!(m("26.7.2", Some("25G332")).darwin(), Some((25, 6)));
        assert_eq!(m("26.4", Some("25E246")).darwin(), Some((25, 4)));
        assert_eq!(m("26.4", Some("25E5207k")).darwin(), Some((25, 4)));
        assert_eq!(m("26", None).darwin(), Some((25, 99)));
        assert_eq!(m("15", None).darwin(), Some((24, 99)));
        assert_eq!(m("15.6.1", None).darwin(), Some((24, 99)));
        assert_eq!(m("10.15", None).darwin(), Some((19, 99)));
        assert_eq!(m("tahoe", None).darwin(), None);
        assert_eq!(release_of(25, 4), "26.4");
        assert_eq!(release_of(24, 6), "15.6");
    }

    /// reims-vgpu where the guest's kexts load on every macOS found, which the tests' set bounds
    /// to Darwin 25.4.0 to 25.99.99; apple-gfx where they don't load on one; none for nothing
    /// found.
    #[test]
    fn the_graphics_follow_the_guests_macos() {
        let m = |v: &str, b: &str| Macos {
            version: v.into(),
            build: Some(b.into()),
        };
        let online = |v: &str| Macos {
            version: v.into(),
            build: None,
        };
        assert_eq!(graphics_for(&[]), None);
        assert_eq!(
            graphics_for(&[m("26.4", "25E246")]),
            Some(Graphics::ReimsVgpu)
        );
        assert_eq!(graphics_for(&[online("26")]), Some(Graphics::ReimsVgpu));
        assert_eq!(
            graphics_for(&[m("26.3.1", "25D2128")]),
            Some(Graphics::AppleGfx)
        );
        assert_eq!(graphics_for(&[online("15")]), Some(Graphics::AppleGfx));
        assert_eq!(
            graphics_for(&[m("26.7.2", "25G332"), m("15.6.1", "24G90")]),
            Some(Graphics::AppleGfx)
        );
        assert_eq!(graphics_for(&[online("27")]), Some(Graphics::AppleGfx));
        let note = create_note("t", &[m("15.6.1", "24G90")], Graphics::AppleGfx, false).unwrap();
        assert!(
            note.starts_with("macOS 15.6.1 (24G90) is older than macOS 26.4"),
            "{note}"
        );
        assert_eq!(
            create_note("t", &[m("26.4", "25E246")], Graphics::ReimsVgpu, false),
            None
        );
        let given = create_note("t", &[online("15")], Graphics::ReimsVgpu, true).unwrap();
        assert!(given.contains("--graphics apple-gfx"), "{given}");
        let newer = create_note("t", &[online("27")], Graphics::AppleGfx, false).unwrap();
        assert!(newer.contains("newer than macOS 26,"), "{newer}");
    }
}
