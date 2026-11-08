//! What reims-ctl reads of the macOS on an x86-64 VM's disk: AppleGVA's setting that
//! reims-vgpu's kexts leave in the guest ([`gva_note`]).
//!
//! The reader follows Apple's File System Reference: the latest checkpoint's container
//! superblock, the container's object map, each volume's superblock, and a volume's object map
//! and file-system tree. It reads what macOS writes there and nothing else: no encrypted volume,
//! no compressed file, and no checkpoint area kept as a tree. Read from an x86-64 macOS 26.7.2
//! (25G332) guest's disk: the system volume is sealed, its tree hashed, and the Data volume holds
//! the user's files.

use crate::opencore::{Disk, le16, le32, le64};
use crate::vm::{Graphics, Vm};
use anyhow::{Context, Result, bail, ensure};

/// AppleGVA's gvaForceAMDKE as a VM's macOS keeps it, in
/// /Library/Preferences/com.apple.AppleGVA.plist on its Data volume. RgfxGate, a kext of the
/// guest's set, sets it true at each boot on reims-vgpu's own device where that file is absent, an
/// XML plist or a binary one (guest/rgfxgate/src/RgfxGatePrefs.cpp), and nothing undoes it once
/// the VM moves to other graphics. Some, with how many other keys the file holds, where it is
/// true; None where the disk holds no such setting, or reims-ctl can't read it (an encrypted Data
/// volume).
fn gva_forced(vm: &Vm) -> Option<usize> {
    let read = || -> Result<Option<usize>> {
        let disk = Disk::open(&vm.disk(), vm.config.disk_format)?;
        for start in crate::opencore::partitions(&disk, &APFS_TYPE)? {
            let c = Container::open(&disk, start)?;
            for bytes in c.files(APFS_VOL_ROLE_DATA, GVA_PREFERENCES)? {
                let v = plist::Value::from_reader(std::io::Cursor::new(bytes))?;
                let Some(d) = v.as_dictionary() else {
                    continue;
                };
                let forced = d.get("gvaForceAMDKE").is_some_and(|k| {
                    k.as_boolean() == Some(true) || k.as_signed_integer().is_some_and(|n| n != 0)
                });
                if forced {
                    return Ok(Some(d.len() - 1));
                }
            }
        }
        Ok(None)
    };
    read().ok().flatten()
}

/// What a new OpenCore image without reims-vgpu's kexts, for an x86-64 VM on other graphics,
/// tells of what they left in its macOS, which reims-ctl can't change: the AppleGVA setting
/// ([`gva_forced`]), with the command that removes it, and RgfxGate's Metal bundle. Measured on
/// apple-gfx with a macOS 26.7.2 guest: VideoToolbox answers the same with the setting and
/// without it (no hardware decoding of H.264 or HEVC, no hardware encoder among its 15); with the
/// file removed, the next boot on reims-vgpu writes it again, as it sets it again in the empty
/// binary plist `defaults delete` leaves; root can't remove the bundle, which SIP protects. None
/// where the guest keeps no such setting.
pub fn gva_note(vm: &Vm) -> Option<String> {
    let graphics = vm.config.graphics();
    if graphics == Graphics::ReimsVgpu {
        return None;
    }
    let others = gva_forced(vm)?;
    let g = graphics.name();
    let file = format!("/{GVA_PREFERENCES}");
    let undo = if others == 0 {
        format!(
            "sudo rm {file} (the drivers write it again if the VM goes back to reims-vgpu's \
             device)"
        )
    } else {
        format!("sudo plutil -remove gvaForceAMDKE {file}")
    };
    Some(format!(
        "its macOS keeps a setting reims-vgpu's macOS drivers made, AppleGVA's gvaForceAMDKE, \
         which points VideoToolbox at an AMD video driver, as reims-vgpu's device has: {g} has \
         none, so it changes nothing there. To remove it anyway, run this in the guest's \
         Terminal: {undo}. The drivers' Metal bundle, /Library/GPUBundles/reims-graphics.bundle, \
         stays too: macOS protects it, and {g} doesn't use it"
    ))
}

/// AppleGVA's preferences, from the Data volume's root.
const GVA_PREFERENCES: &str = "Library/Preferences/com.apple.AppleGVA.plist";

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
const APFS_VOL_ROLE_DATA: u16 = 0x40;
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

    /// Its volumes' superblocks.
    fn volumes(&self) -> Result<Vec<Vec<u8>>> {
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
        Ok(volumes)
    }

    /// The file at `path`, from a volume's root, on each unencrypted volume of `role` that holds
    /// it.
    fn files(&self, role: u16, path: &str) -> Result<Vec<Vec<u8>>> {
        let mut out = Vec::new();
        for sb in self.volumes()? {
            if le16(&sb, 964) != role || le64(&sb, 264) & APFS_FS_UNENCRYPTED == 0 {
                continue;
            }
            let fs = Volume::open(self, &sb)?;
            if let Some(file) = fs.lookup(ROOT_DIR_INO_NUM, path)? {
                out.push(fs.read(file)?);
            }
        }
        Ok(out)
    }
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

    /// The inode `path`, names separated by '/', names from directory `dir`.
    fn lookup(&self, dir: u64, path: &str) -> Result<Option<u64>> {
        let mut at = dir;
        for part in path.split('/').filter(|p| !p.is_empty()) {
            match self.child(at, part)? {
                Some(id) => at = id,
                None => return Ok(None),
            }
        }
        Ok(Some(at))
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
            "a compressed file, which reims-ctl does not read"
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
pub(crate) mod tests {
    use super::*;
    use std::path::Path;

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

    /// A volume for [`disk`]: its role, UUID and volume group, and its files, each a path from
    /// its root and the file's bytes.
    struct Vol<'a> {
        role: u16,
        uuid: [u8; 16],
        group: [u8; 16],
        files: Vec<(&'a str, &'a [u8])>,
    }

    /// A disk holding a GPT, one APFS partition from sector 40, and in it `vols`, each with a
    /// file-system tree of one leaf.
    fn disk(vols: &[Vol]) -> Vec<u8> {
        const START: usize = 40 * 512;
        // Blocks: 0 the superblock, 1-2 the container's object map, then a volume's superblock,
        // object map (two) and tree at 3 + 4i, then the files' data.
        let mut blocks: Vec<Vec<u8>> = vec![Vec::new(); 3 + 4 * vols.len()];
        let mut nx = object(OBJECT_TYPE_NX_SUPERBLOCK | OBJ_PHYSICAL, 1, 1);
        put(&mut nx, 32, NX_MAGIC);
        put(&mut nx, 36, &(BS as u32).to_le_bytes());
        put(&mut nx, 160, &1u64.to_le_bytes());
        put(&mut nx, 180, &100u32.to_le_bytes());
        let mut map = Vec::new();
        for i in 0..vols.len() {
            put(&mut nx, 184 + i * 8, &(1026 + i as u64).to_le_bytes());
            map.push((1026 + i as u64, 3 + 4 * i as u64));
        }
        blocks[0] = nx;
        let [om, omt] = omap(1, &map);
        (blocks[1], blocks[2]) = (om, omt);
        for (i, v) in vols.iter().enumerate() {
            let at = 3 + 4 * i;
            let tree_oid = 2000 + i as u64;
            let mut sb = object(OBJECT_TYPE_FS, 1026 + i as u64, 1);
            put(&mut sb, 32, APFS_MAGIC);
            put(&mut sb, 56, &APFS_INCOMPAT_CASE_INSENSITIVE.to_le_bytes());
            put(&mut sb, 116, &OBJECT_TYPE_BTREE.to_le_bytes());
            put(&mut sb, 128, &(at as u64 + 1).to_le_bytes());
            put(&mut sb, 136, &tree_oid.to_le_bytes());
            put(&mut sb, 240, &v.uuid);
            put(&mut sb, 264, &APFS_FS_UNENCRYPTED.to_le_bytes());
            put(&mut sb, 964, &v.role.to_le_bytes());
            put(&mut sb, 1008, &v.group);
            blocks[at] = sb;
            let [om, omt] = omap(at as u64 + 1, &[(tree_oid, at as u64 + 3)]);
            (blocks[at + 1], blocks[at + 2]) = (om, omt);
            // The tree: each file's directories, once each, then its inode and extent.
            let mut records = Vec::new();
            let mut dirs: std::collections::BTreeMap<String, u64> = Default::default();
            let mut next = 16;
            for (path, bytes) in &v.files {
                let parts: Vec<&str> = path.split('/').collect();
                let mut parent = ROOT_DIR_INO_NUM;
                for (j, part) in parts.iter().enumerate() {
                    let prefix = parts[..=j].join("/");
                    if let Some(&id) = dirs.get(&prefix) {
                        parent = id;
                        continue;
                    }
                    let id = next;
                    next += 1;
                    records.push(drec(parent, part, id));
                    dirs.insert(prefix, id);
                    parent = id;
                }
                let file = parent;
                let mut inode = vec![0u8; 96 + 4 + 40];
                put(&mut inode, 8, &(file + 1000).to_le_bytes());
                put(&mut inode, 92, &1u16.to_le_bytes());
                put(&mut inode, 94, &40u16.to_le_bytes());
                inode[96] = INO_EXT_TYPE_DSTREAM;
                put(&mut inode, 98, &40u16.to_le_bytes());
                put(&mut inode, 100, &(bytes.len() as u64).to_le_bytes());
                records.push((key(file, APFS_TYPE_INODE), inode));
                let block = blocks.len() as u64;
                let mut contents = bytes.to_vec();
                contents.resize(BS, 0);
                blocks.push(contents);
                let ext = [(BS as u64).to_le_bytes(), block.to_le_bytes(), [0; 8]].concat();
                let ek = [
                    key(file + 1000, APFS_TYPE_FILE_EXTENT),
                    0u64.to_le_bytes().to_vec(),
                ];
                records.push((ek.concat(), ext));
            }
            records.sort_by(|a, b| {
                let id = |k: &[u8]| (le64(k, 0) & OBJ_ID_MASK, le64(k, 0) >> 60);
                id(&a.0).cmp(&id(&b.0))
            });
            let mut tree = leaf(&records, false, tree_oid);
            put(&mut tree, 24, &OBJECT_TYPE_BTREE.to_le_bytes());
            blocks[at + 3] = tree;
        }
        for b in blocks.iter_mut().take(3 + 4 * vols.len()) {
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

    /// An empty Preboot volume, as an install's container holds beside its Data volume.
    fn preboot<'a>() -> Vol<'a> {
        Vol {
            role: 0x10,
            uuid: [0; 16],
            group: [0; 16],
            files: Vec::new(),
        }
    }

    /// A data volume of the volume group [`GROUP`], holding `files`.
    /// Writes at `path` a raw disk whose Data volume keeps the AppleGVA setting RgfxGate writes,
    /// in a file of that key alone, as it makes one.
    pub(crate) fn gva_disk(path: &Path) {
        let pref = br#"<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0"><dict><key>gvaForceAMDKE</key><true/></dict></plist>"#;
        std::fs::write(
            path,
            disk(&[preboot(), data(vec![(GVA_PREFERENCES, pref)])]),
        )
        .unwrap();
    }

    fn data<'a>(files: Vec<(&'a str, &'a [u8])>) -> Vol<'a> {
        Vol {
            role: APFS_VOL_ROLE_DATA,
            uuid: GROUP,
            group: GROUP,
            files,
        }
    }

    const GROUP: [u8; 16] = [
        0xd3, 0x6e, 0x3c, 0x34, 0xbd, 0xaf, 0x44, 0xaa, 0xbe, 0x65, 0x17, 0xc5, 0xdd, 0x68, 0x6b,
        0xc1,
    ];

    /// A guest whose Data volume's AppleGVA preferences set gvaForceAMDKE, as RgfxGate leaves
    /// them, is told, with the file's other keys counted; one without the key, or without the
    /// file, is not.
    #[test]
    fn a_guest_tells_the_gva_setting_reims_vgpus_kexts_leave() {
        let pref = |body: &str| {
            format!(
                r#"<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0"><dict>{body}</dict></plist>"#
            )
            .into_bytes()
        };
        let forced = pref("<key>gvaForceAMDKE</key><true/>");
        let kept = pref("<key>gvaForceAMDKE</key><true/><key>other</key><integer>1</integer>");
        let off = pref("<key>other</key><integer>1</integer>");
        let env = crate::testenv::Env::new("gva");
        let dir = env.root.join("vm");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join(crate::vm::CONFIG_FILE),
            r#"
            name = "t"
            profile = "q35-x86_64"
            cpus = 2
            memory_mib = 4096
            disk = "disk.img"
            disk_format = "raw"
            graphics = "vmware-svga"
            [q35]
            opencore = "opencore.img"
            firmware_code = "OVMF_CODE.fd"
            firmware_vars = "OVMF_VARS.fd"
            "#,
        )
        .unwrap();
        let vm = Vm::load(&dir).unwrap();
        let note = || gva_note(&vm).unwrap_or_default();
        let with = |files: Vec<(&str, &[u8])>| {
            std::fs::write(dir.join("disk.img"), disk(&[preboot(), data(files)])).unwrap();
            gva_forced(&vm)
        };
        assert_eq!(with(vec![(GVA_PREFERENCES, &forced)]), Some(0));
        // A file of the key alone, as RgfxGate makes one, is removed; another loses the key.
        assert!(note().contains("sudo rm /Library/Preferences/com.apple.AppleGVA.plist"));
        assert_eq!(with(vec![(GVA_PREFERENCES, &kept)]), Some(1));
        assert!(note().contains(
            "sudo plutil -remove gvaForceAMDKE /Library/Preferences/com.apple.AppleGVA.plist"
        ));
        assert_eq!(with(vec![(GVA_PREFERENCES, &off)]), None);
        assert_eq!(gva_note(&vm), None);
        assert_eq!(
            with(vec![("Library/Preferences/other.plist", &forced)]),
            None
        );
        assert_eq!(with(Vec::new()), None);
    }
}
