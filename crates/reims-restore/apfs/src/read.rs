//! Reading a container: its newest checkpoint, its object maps, its volumes, and B-trees
//! searched from the root down.
//!
//! Every object read is checked: its Fletcher-64 checksum, and its identifier, type, subtype
//! and transaction against what led to it. Every node a search passes through also has its keys
//! checked to ascend strictly, its level to be one below its parent's, and its first key to be
//! no lower than its parent's key for it.

use std::cmp::Ordering;

use crate::btree::Node;
use crate::dev::Partition;
use crate::fs;
use crate::le::{u16_at, u32_at, u64_at};
use sha2::{Digest, Sha256};

use crate::ondisk::{
    ApfsSuperblock, BtreeInfo, NX_MAGIC, NxSuperblock, ObjHeader, OmapPhys, apfs, btree, cpm, im,
    obj, omap,
};
use crate::{Error, Result, corrupt, fletcher};

/// An object map entry's value: its flags, size and block address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OmapVal {
    pub flags: u32,
    pub size: u32,
    pub paddr: u64,
}

/// An object map: the root of its tree from (identifier, transaction) to [`OmapVal`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct Omap {
    tree: u64,
    /// The transaction the map was read at; no node of its tree is newer.
    xid: u64,
}

/// How a tree's child identifiers find their nodes.
#[derive(Clone, Copy)]
pub(crate) enum Addressing<'a> {
    /// Identifiers are block addresses.
    Physical,
    /// Identifiers map through an object map, as of a transaction.
    Virtual(&'a Omap, u64),
}

/// A key comparator.
pub(crate) type KeyOrder = fn(&[u8], &[u8]) -> Ordering;

/// A visitor of leaf records, by key and value, which returns whether to go on.
pub(crate) type Visit<'f> = dyn FnMut(&[u8], &[u8]) -> Result<bool> + 'f;

/// A leaf record's key and value.
pub(crate) type Record = (Vec<u8>, Vec<u8>);

/// An opened container, at its newest valid checkpoint.
pub struct Container {
    dev: Partition,
    block_size: u32,
    /// The newest valid checkpoint's superblock.
    nx: NxSuperblock,
    omap: Omap,
}

impl std::fmt::Debug for Container {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Container")
            .field("dev", &self.dev)
            .field("block_size", &self.block_size)
            .field("xid", &self.xid())
            .finish_non_exhaustive()
    }
}

impl Container {
    /// Open the container on `dev`, at the newest checkpoint whose superblock and maps check.
    ///
    /// # Errors
    ///
    /// Block 0 is not a container superblock, no checkpoint is valid, or the container is of a
    /// kind this crate does not read (Fusion, a checkpoint area in pieces).
    pub fn open(dev: Partition) -> Result<Self> {
        let mut head = vec![0u8; 4096];
        dev.read_at(0, &mut head)?;
        if u32_at(&head, 32) != NX_MAGIC {
            return Err(corrupt(0, "not an APFS container (no NXSB magic)"));
        }
        let block_size = u32_at(&head, 36);
        if !(4096..=65536).contains(&block_size) || !block_size.is_power_of_two() {
            return Err(corrupt(0, format!("block size {block_size}")));
        }
        let mut b0 = vec![0u8; block_size as usize];
        dev.read_at(0, &mut b0)?;
        check_object(&b0, 0, None, Some(obj::NX_SUPERBLOCK), u64::MAX)?;
        let nx0 = NxSuperblock::parse(&b0);
        if nx0.incompat & crate::ondisk::nx::INCOMPAT_FUSION != 0 {
            return Err(Error::Unsupported("a Fusion container".into()));
        }
        if nx0.xp_desc_blocks & 0x8000_0000 != 0 {
            return Err(Error::Unsupported(
                "a checkpoint descriptor area that is not contiguous".into(),
            ));
        }
        if nx0
            .block_count
            .checked_mul(u64::from(block_size))
            .is_none_or(|b| b > dev.len())
        {
            return Err(corrupt(
                0,
                format!(
                    "{} blocks do not fit the partition's {} bytes",
                    nx0.block_count,
                    dev.len()
                ),
            ));
        }
        let mut c = Self {
            dev,
            block_size,
            nx: nx0.clone(),
            omap: Omap { tree: 0, xid: 0 },
        };
        // The checkpoints in the descriptor area, newest first: the first whose superblock and
        // maps all check is the container's state.
        let (base, n) = (nx0.xp_desc_base, u64::from(nx0.xp_desc_blocks));
        let mut supers = Vec::new();
        for i in 0..n {
            let b = c.read_block(base + i)?;
            if ObjHeader::parse(&b).kind() == obj::NX_SUPERBLOCK
                && u32_at(&b, 32) == NX_MAGIC
                && fletcher::verify(&b)
            {
                supers.push((ObjHeader::parse(&b).xid, base + i, b));
            }
        }
        supers.sort_by_key(|s| std::cmp::Reverse(s.0));
        let mut last_err = corrupt(base, "no valid checkpoint superblock");
        for (xid, pa, b) in supers {
            let nx = NxSuperblock::parse(&b);
            match c.check_maps(&nx, pa, xid) {
                Ok(()) => {
                    c.nx = nx;
                    c.omap = c.load_omap(c.nx.omap_oid, xid)?;
                    return Ok(c);
                }
                Err(e) => last_err = e,
            }
        }
        Err(last_err)
    }

    /// Check the checkpoint maps of the checkpoint whose superblock `nx` is at `pa`.
    fn check_maps(&self, nx: &NxSuperblock, pa: u64, xid: u64) -> Result<()> {
        let (base, n) = (nx.xp_desc_base, u64::from(nx.xp_desc_blocks));
        if nx.xp_desc_len == 0 || u64::from(nx.xp_desc_len) > n {
            return Err(corrupt(
                pa,
                format!(
                    "checkpoint of {} blocks in a {n}-block area",
                    nx.xp_desc_len
                ),
            ));
        }
        for k in 0..u64::from(nx.xp_desc_len) {
            let at = base + (u64::from(nx.xp_desc_index) + k) % n;
            if at == pa {
                continue;
            }
            let m = self.read_object(at, None, Some(obj::CHECKPOINT_MAP), xid)?;
            if ObjHeader::parse(&m).xid != xid {
                return Err(corrupt(at, "checkpoint map from another transaction"));
            }
            if u32_at(&m, cpm::COUNT) as usize > (m.len() - cpm::MAP) / cpm::ENTRY_SIZE {
                return Err(corrupt(
                    at,
                    "checkpoint map with more entries than fit its block",
                ));
            }
        }
        Ok(())
    }

    /// The size of every block.
    #[must_use]
    pub const fn block_size(&self) -> u32 {
        self.block_size
    }

    /// The newest checkpoint's transaction.
    #[must_use]
    pub const fn xid(&self) -> u64 {
        self.nx.hdr.xid
    }

    /// Where the container starts in the source it was opened on, in bytes.
    #[must_use]
    pub const fn base(&self) -> u64 {
        self.dev.base()
    }

    /// The newest checkpoint's superblock.
    #[must_use]
    pub const fn superblock(&self) -> &NxSuperblock {
        &self.nx
    }

    /// Read block `paddr`.
    pub(crate) fn read_block(&self, paddr: u64) -> Result<Vec<u8>> {
        let mut b = vec![0u8; self.block_size as usize];
        self.read_bytes(paddr, 0, &mut b)?;
        Ok(b)
    }

    /// Read `buf.len()` bytes from `off` bytes into block `paddr`, all of them inside the
    /// container.
    pub(crate) fn read_bytes(&self, paddr: u64, off: u64, buf: &mut [u8]) -> Result<()> {
        let bs = u64::from(self.block_size);
        let start = paddr.checked_mul(bs).and_then(|s| s.checked_add(off));
        let end = start.and_then(|s| s.checked_add(buf.len() as u64));
        match (start, end) {
            (Some(s), Some(e)) if e <= self.nx.block_count * bs => self.dev.read_at(s, buf),
            _ => Err(corrupt(
                paddr,
                format!("a read past the container's {} blocks", self.nx.block_count),
            )),
        }
    }

    /// Read the object at `paddr`, checking its checksum, its identifier (`oid`, where known),
    /// its type (`kind`, where given) and that it is no newer than `max_xid`.
    pub(crate) fn read_object(
        &self,
        paddr: u64,
        oid: Option<u64>,
        kind: Option<u32>,
        max_xid: u64,
    ) -> Result<Vec<u8>> {
        let b = self.read_block(paddr)?;
        check_object(&b, paddr, oid, kind, max_xid)?;
        Ok(b)
    }

    /// The object map whose header is at `paddr`.
    fn load_omap(&self, paddr: u64, max_xid: u64) -> Result<Omap> {
        let b = self.read_object(paddr, Some(paddr), Some(obj::OMAP), max_xid)?;
        let phys = OmapPhys::parse(&b);
        if phys.tree_type & obj::STORAGE_MASK != obj::PHYSICAL {
            return Err(corrupt(
                paddr,
                format!("object map tree of type {:#x}", phys.tree_type),
            ));
        }
        Ok(Omap {
            tree: phys.tree_oid,
            xid: max_xid,
        })
    }

    /// The newest mapping of `oid` in `m` at or before `xid`.
    pub(crate) fn omap_lookup(
        &self,
        m: &Omap,
        oid: u64,
        xid: u64,
    ) -> Result<Option<(u64, OmapVal)>> {
        if m.tree == 0 {
            return Ok(None);
        }
        let mut key = [0u8; 16];
        key[..8].copy_from_slice(&oid.to_le_bytes());
        key[8..].copy_from_slice(&xid.to_le_bytes());
        let t = Tree {
            c: self,
            root: m.tree,
            addressing: Addressing::Physical,
            subtype: obj::OMAP,
            max_xid: m.xid,
            order: fs::compare_u64_pairs,
            root_hash: None,
        };
        let Some((k, v)) = t.last_le(&key)? else {
            return Ok(None);
        };
        if u64_at(&k, 0) != oid {
            return Ok(None);
        }
        if v.len() < 16 {
            return Err(corrupt(m.tree, "short object map value"));
        }
        Ok(Some((
            u64_at(&k, 8),
            OmapVal {
                flags: u32_at(&v, 0),
                size: u32_at(&v, 4),
                paddr: u64_at(&v, 8),
            },
        )))
    }

    /// The volume in slot `index`, as of the newest checkpoint.
    ///
    /// # Errors
    ///
    /// The slot is empty, or the volume's superblock or object map fail their checks.
    pub fn volume(&self, index: usize) -> Result<Volume<'_>> {
        let oid = *self
            .nx
            .fs_oid
            .get(index)
            .filter(|o| **o != 0)
            .ok_or_else(|| Error::Invalid(format!("volume slot {index} is empty")))?;
        let (vxid, v) = self
            .omap_lookup(&self.omap, oid, self.xid())?
            .ok_or_else(|| {
                corrupt(
                    0,
                    format!("volume {oid} is not in the container's object map"),
                )
            })?;
        if v.flags & omap::VAL_DELETED != 0 {
            return Err(corrupt(v.paddr, format!("volume {oid} is deleted")));
        }
        let b = self.read_object(v.paddr, Some(oid), Some(obj::FS), vxid)?;
        if u64_at(&b, 16) != vxid {
            return Err(corrupt(
                v.paddr,
                "volume superblock's xid differs from its object map entry",
            ));
        }
        if u32_at(&b, apfs::MAGIC) != crate::ondisk::APFS_MAGIC {
            return Err(corrupt(v.paddr, "volume superblock without APSB magic"));
        }
        let sb = ApfsSuperblock::parse(&b);
        let omap = self.load_omap(sb.omap_oid, self.xid())?;
        let root_hash = if sb.is_sealed() {
            self.seal(&sb, &omap)?
        } else {
            None
        };
        Ok(Volume {
            c: self,
            index,
            oid,
            sb,
            omap,
            root_hash,
        })
    }

    /// A sealed volume's root hash, the SHA-256 of its file-system tree's root node, from its
    /// integrity metadata; `None` where the seal is broken.
    fn seal(&self, sb: &ApfsSuperblock, omap: &Omap) -> Result<Option<Vec<u8>>> {
        let oid = sb.integrity_meta_oid;
        let (xid, v) = self.omap_lookup(omap, oid, self.xid())?.ok_or_else(|| {
            corrupt(
                0,
                format!("sealed volume {:?} without its integrity metadata", sb.name),
            )
        })?;
        let b = self.read_object(v.paddr, Some(oid), Some(obj::INTEGRITY_META), xid)?;
        if u32_at(&b, im::FLAGS) & im::SEAL_BROKEN != 0 {
            return Ok(None);
        }
        let kind = u32_at(&b, im::HASH_TYPE);
        if kind != im::HASH_SHA256 {
            return Err(Error::Unsupported(format!("a seal of hash type {kind}")));
        }
        let off = u32_at(&b, im::ROOT_HASH_OFFSET) as usize;
        let hash = b
            .get(off..off + HASH_LEN)
            .ok_or_else(|| corrupt(v.paddr, "root hash past the block"))?;
        Ok(Some(hash.to_vec()))
    }

    /// Every volume in the container, in slot order.
    ///
    /// # Errors
    ///
    /// A volume's superblock or object map fail their checks.
    pub fn volumes(&self) -> Result<Vec<Volume<'_>>> {
        let slots = (self.nx.max_file_systems as usize).min(self.nx.fs_oid.len());
        (0..slots)
            .filter(|i| self.nx.fs_oid[*i] != 0)
            .map(|i| self.volume(i))
            .collect()
    }

    /// The volume holding the operating system: the one whose role is System, or else the
    /// container's only volume.
    ///
    /// # Errors
    ///
    /// No volume has that role and the container holds more than one, or a volume fails its
    /// checks.
    pub fn system_volume(&self) -> Result<Volume<'_>> {
        let mut vols = self.volumes()?;
        if let Some(i) = vols.iter().position(|v| v.sb.role == apfs::ROLE_SYSTEM) {
            return Ok(vols.swap_remove(i));
        }
        match vols.len() {
            1 => Ok(vols.remove(0)),
            n => Err(Error::Invalid(format!(
                "no volume has the System role among the container's {n}"
            ))),
        }
    }
}

/// Check an object's checksum, identifier, type and transaction.
pub(crate) fn check_object(
    b: &[u8],
    paddr: u64,
    oid: Option<u64>,
    kind: Option<u32>,
    max_xid: u64,
) -> Result<()> {
    if !fletcher::verify(b) {
        return Err(Error::Checksum {
            block: paddr,
            stored: fletcher::stored(b),
            computed: fletcher::checksum(b),
        });
    }
    let h = ObjHeader::parse(b);
    if let Some(o) = oid
        && h.oid != o
    {
        return Err(corrupt(
            paddr,
            format!("object {} where {o} was expected", h.oid),
        ));
    }
    if let Some(k) = kind
        && h.kind() != k
    {
        return Err(corrupt(
            paddr,
            format!("object of type {:#x} where {k:#x} was expected", h.kind()),
        ));
    }
    if h.xid > max_xid {
        return Err(corrupt(
            paddr,
            format!("object from xid {} is newer than {max_xid}", h.xid),
        ));
    }
    Ok(())
}

/// A volume of a container.
pub struct Volume<'c> {
    pub(crate) c: &'c Container,
    /// The volume's slot in the container.
    pub index: usize,
    /// The volume's virtual identifier.
    pub oid: u64,
    /// The volume's superblock.
    pub sb: ApfsSuperblock,
    omap: Omap,
    /// A sealed volume's root hash.
    root_hash: Option<Vec<u8>>,
}

impl std::fmt::Debug for Volume<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Volume")
            .field("index", &self.index)
            .field("name", &self.sb.name)
            .field("role", &self.sb.role)
            .finish_non_exhaustive()
    }
}

impl<'c> Volume<'c> {
    /// The volume's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.sb.name
    }

    /// The volume's role (`APFS_VOL_ROLE_*`, [`apfs::ROLE_SYSTEM`] among them).
    #[must_use]
    pub const fn role(&self) -> u16 {
        self.sb.role
    }

    /// The container the volume is in.
    #[must_use]
    pub const fn container(&self) -> &'c Container {
        self.c
    }

    /// A sealed volume's root hash: the SHA-256 the integrity metadata records for its
    /// file-system tree's root node. `None` on an unsealed volume or one whose seal is broken.
    #[must_use]
    pub fn root_hash(&self) -> Option<&[u8]> {
        self.root_hash.as_deref()
    }

    /// The shape of every node of the file-system tree, parents before children, in key order:
    /// for studying how a sealed volume's hashed tree is packed, so a writer can reproduce it.
    ///
    /// # Errors
    ///
    /// The tree fails its checks.
    pub fn fs_tree_shape(&self) -> Result<Vec<NodeShape>> {
        let mut out = Vec::new();
        self.fs_tree()?.walk_nodes(&mut |s| {
            out.push(s.clone());
            Ok(())
        })?;
        Ok(out)
    }

    /// Every leaf record of the file-system tree, in key order, as (key, value): what a seal is
    /// built from (`canon`, `seal`). It holds them all in memory, about 300 MB for a 27.2.1
    /// system volume's 1.76 million records.
    pub fn fs_tree_records(&self) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let mut out = Vec::new();
        self.fs_tree()?.scan(&[0u8; 8], &mut |k, v| {
            out.push((k.to_vec(), v.to_vec()));
            Ok(true)
        })?;
        Ok(out)
    }

    /// Read-only facts about a sealed volume's seal, for showing a written volume reproduces it:
    /// the root hash, how many of each file-system record type the hashed tree holds (type `t`
    /// is [`crate::ondisk::rec::name`]`(t)`), and the physical file-extent tree that carries the
    /// block addresses the seal does not cover.
    ///
    /// # Errors
    ///
    /// A tree fails its checks.
    pub fn seal_facts(&self) -> Result<SealFacts> {
        let mut fs_record_types: std::collections::BTreeMap<u8, u64> =
            std::collections::BTreeMap::new();
        self.fs_tree()?.scan(&[0u8; 8], &mut |k, _| {
            *fs_record_types.entry(fs::key_header(k).1).or_default() += 1;
            Ok(true)
        })?;
        let mut facts = SealFacts {
            root_hash: self.root_hash.clone(),
            fs_record_types,
            fext_records: 0,
            fext_root_paddr: 0,
            fext_root_has_object_header: false,
        };
        if let Some(t) = self.fext_tree()? {
            t.scan(&[0u8; 16], &mut |_, _| {
                facts.fext_records += 1;
                Ok(true)
            })?;
            let paddr = t.root_block()?;
            let b = self.c.read_block(paddr)?;
            facts.fext_root_paddr = paddr;
            // A tree outside the seal carries an ordinary object header, Fletcher-checked; a
            // hashed (sealed) tree's nodes are headerless, their first 32 bytes zero.
            facts.fext_root_has_object_header =
                b[..32].iter().any(|x| *x != 0) && fletcher::verify(&b);
        }
        Ok(facts)
    }

    fn addressing(&self, tree_type: u32) -> Result<Addressing<'_>> {
        match tree_type & obj::STORAGE_MASK {
            obj::PHYSICAL => Ok(Addressing::Physical),
            obj::VIRTUAL => Ok(Addressing::Virtual(&self.omap, self.c.xid())),
            _ => Err(Error::Unsupported(format!(
                "a volume tree of type {tree_type:#x}"
            ))),
        }
    }

    /// The volume's file-system tree.
    pub(crate) fn fs_tree(&self) -> Result<Tree<'_>> {
        Ok(Tree {
            c: self.c,
            root: self.sb.root_tree_oid,
            addressing: self.addressing(self.sb.root_tree_type)?,
            subtype: obj::FSTREE,
            max_xid: self.c.xid(),
            order: fs::compare_keys,
            root_hash: self.root_hash.as_deref(),
        })
    }

    /// A sealed volume's file extent tree, which holds its files' extents in place of the
    /// file-system tree.
    pub(crate) fn fext_tree(&self) -> Result<Option<Tree<'_>>> {
        if self.sb.fext_tree_oid == 0 {
            return Ok(None);
        }
        Ok(Some(Tree {
            c: self.c,
            root: self.sb.fext_tree_oid,
            addressing: self.addressing(self.sb.fext_tree_type)?,
            subtype: obj::FEXT_TREE,
            max_xid: self.c.xid(),
            order: fs::compare_u64_pairs,
            root_hash: None,
        }))
    }
}

/// What a sealed volume's seal covers and what it does not ([`Volume::seal_facts`]).
#[derive(Clone, Debug, Default)]
pub struct SealFacts {
    /// The signed root hash: SHA-256 of the file-system tree's root node.
    pub root_hash: Option<Vec<u8>>,
    /// How many of each file-system record type the hashed tree holds. A sealed volume's holds
    /// no file-extent records (type [`crate::ondisk::rec::FILE_EXTENT`]); its files' extents are
    /// in the file-extent tree instead.
    pub fs_record_types: std::collections::BTreeMap<u8, u64>,
    /// How many physical extents the file-extent tree holds.
    pub fext_records: u64,
    /// The block address of the file-extent tree's root.
    pub fext_root_paddr: u64,
    /// Whether that root is an ordinary object (an object header, Fletcher-checked), which means
    /// the file-extent tree is outside the seal and its block addresses may be rewritten.
    pub fext_root_has_object_header: bool,
}

/// One node's shape, for studying how a tree is packed ([`Volume::fs_tree_shape`]).
#[derive(Clone, Debug)]
pub struct NodeShape {
    /// The node's level: 0 for a leaf, higher toward the root.
    pub level: u16,
    /// How many entries the node holds.
    pub nkeys: u32,
    /// Whether this is the root node (its value area ends 40 bytes early, holding the tree info).
    pub is_root: bool,
    /// The node's unused bytes (`btn_free_space.len`).
    pub free_space: u16,
    /// The node's first key.
    pub first_key: Vec<u8>,
    /// The node's second key, for telling whether an index node repeats its first child's key as
    /// a low fence.
    pub second_key: Vec<u8>,
    /// The node's last key.
    pub last_key: Vec<u8>,
}

/// A B-tree, searched from its root.
pub(crate) struct Tree<'a> {
    c: &'a Container,
    root: u64,
    addressing: Addressing<'a>,
    subtype: u32,
    max_xid: u64,
    order: KeyOrder,
    /// The SHA-256 the root node must have: a sealed volume's file-system tree's.
    root_hash: Option<&'a [u8]>,
}

/// The size of a seal's hashes: SHA-256's.
const HASH_LEN: usize = 32;

/// The deepest a B-tree may be. A node's level is a 16-bit field an attacker controls, and
/// [`Tree::child`] holds every child's level to be one below its parent's, so a walk recurses as
/// deep as the root's level claims. Real APFS trees are a handful of levels (a 4 KiB node holds
/// hundreds of keys); even at the smallest branching that keeps a tree, 64 levels span the whole
/// 2^64 key space, so a root deeper than this is corrupt and its walk is refused rather than run.
const MAX_TREE_DEPTH: u16 = 64;

/// A tree's root node, with what its walks need.
struct Root {
    node: Node,
    info: BtreeInfo,
    /// What a child identifier in an index entry is relative to.
    child_base: u64,
}

impl Tree<'_> {
    /// Call `f` on each leaf record from the first whose key is not below `start`, in key
    /// order, until it returns false.
    pub(crate) fn scan(&self, start: &[u8], f: &mut Visit<'_>) -> Result<()> {
        let r = self.root()?;
        self.scan_node(&r.node, &r, start, f).map(|_| ())
    }

    fn scan_node(&self, n: &Node, r: &Root, start: &[u8], f: &mut Visit<'_>) -> Result<bool> {
        let keys = self.keys(n)?;
        if n.is_leaf() {
            for (i, k) in keys.iter().enumerate() {
                if (self.order)(k, start) == Ordering::Less {
                    continue;
                }
                let (_, v) = n.entry(i as u32)?;
                let v =
                    v.ok_or_else(|| corrupt(n.block, format!("leaf entry {i} without a value")))?;
                if !f(k, v)? {
                    return Ok(false);
                }
            }
            return Ok(true);
        }
        let first = keys
            .iter()
            .rposition(|k| (self.order)(k, start) != Ordering::Greater)
            .unwrap_or(0);
        for i in first..keys.len() {
            let child = self.child(n, r, i)?;
            if !self.scan_node(&child, r, start, f)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// The leaf record with the greatest key not above `key`.
    pub(crate) fn last_le(&self, key: &[u8]) -> Result<Option<Record>> {
        let r = self.root()?;
        let mut n = r.node.clone();
        loop {
            let keys = self.keys(&n)?;
            let Some(i) = keys
                .iter()
                .rposition(|k| (self.order)(k, key) != Ordering::Greater)
            else {
                return Ok(None);
            };
            if n.is_leaf() {
                let (k, v) = n.entry(i as u32)?;
                let v =
                    v.ok_or_else(|| corrupt(n.block, format!("leaf entry {i} without a value")))?;
                return Ok(Some((k.to_vec(), v.to_vec())));
            }
            n = self.child(&n, &r, i)?;
        }
    }

    /// The block address of the tree's root node.
    pub(crate) fn root_block(&self) -> Result<u64> {
        Ok(self.root()?.node.block)
    }

    /// Visit every node of the tree, parents before children, in key order, reporting each
    /// node's shape for studying how it is packed.
    pub(crate) fn walk_nodes(&self, cb: &mut dyn FnMut(&NodeShape) -> Result<()>) -> Result<()> {
        let r = self.root()?;
        self.walk_nodes_at(&r.node, &r, cb)
    }

    fn walk_nodes_at(
        &self,
        n: &Node,
        r: &Root,
        cb: &mut dyn FnMut(&NodeShape) -> Result<()>,
    ) -> Result<()> {
        let keys = self.keys(n)?;
        let first = keys.first().map(|k| k.to_vec()).unwrap_or_default();
        let second = keys.get(1).map(|k| k.to_vec()).unwrap_or_default();
        let last = keys.last().map(|k| k.to_vec()).unwrap_or_default();
        cb(&NodeShape {
            level: n.level,
            nkeys: n.nkeys,
            is_root: n.is_root(),
            // btn_free_space is an nloc_t (off at FREE_SPACE, len two bytes after it); the
            // unused byte count is the len.
            free_space: u16_at(&n.bytes, btree::FREE_SPACE + 2),
            first_key: first,
            second_key: second,
            last_key: last,
        })?;
        if !n.is_leaf() {
            for i in 0..keys.len() {
                let child = self.child(n, r, i)?;
                self.walk_nodes_at(&child, r, cb)?;
            }
        }
        Ok(())
    }

    fn root(&self) -> Result<Root> {
        let (paddr, noheader) = self.resolve(self.root)?;
        let node = self.read_node(
            self.root,
            paddr,
            noheader,
            true,
            self.max_xid,
            None,
            self.root_hash,
        )?;
        if node.level >= MAX_TREE_DEPTH {
            return Err(corrupt(
                paddr,
                format!("root at level {}, deeper than any real B-tree", node.level),
            ));
        }
        let info = node
            .info()
            .ok_or_else(|| corrupt(paddr, "root node without tree info"))?;
        if info.node_size != self.c.block_size {
            return Err(Error::Unsupported(format!(
                "B-tree node size {} in a container of {}-byte blocks",
                info.node_size, self.c.block_size
            )));
        }
        // A hashed tree's index entries name their children relative to the root's identifier
        // (measured on the sealed system volumes of macOS 14 to 27).
        let child_base = if info.flags & btree::HASHED != 0 {
            self.root
        } else {
            0
        };
        Ok(Root {
            node,
            info,
            child_base,
        })
    }

    /// Node `n`'s keys, checked to ascend strictly.
    fn keys<'n>(&self, n: &'n Node) -> Result<Vec<&'n [u8]>> {
        let mut keys: Vec<&'n [u8]> = Vec::with_capacity(n.nkeys as usize);
        for i in 0..n.nkeys {
            let (k, _) = n.entry(i)?;
            if k.len() < 8 {
                return Err(corrupt(n.block, format!("key {i} of {} bytes", k.len())));
            }
            if let Some(prev) = keys.last()
                && (self.order)(prev, k) != Ordering::Less
            {
                return Err(corrupt(
                    n.block,
                    format!("key {i} is not above the one before it"),
                ));
            }
            keys.push(k);
        }
        Ok(keys)
    }

    /// The child entry `i` of index node `n` names, checked against `n`.
    fn child(&self, n: &Node, r: &Root, i: usize) -> Result<Node> {
        let (k, v) = n.entry(i as u32)?;
        let v = v
            .filter(|v| v.len() >= 8)
            .ok_or_else(|| corrupt(n.block, format!("index entry {i} without a child")))?;
        let oid = r
            .child_base
            .checked_add(u64_at(v, 0))
            .ok_or_else(|| corrupt(n.block, "child identifier overflows"))?;
        let (paddr, noheader) = self.resolve(oid)?;
        // A hashed tree's index entry holds its child's hash after the child's identifier.
        let hash = if r.info.flags & btree::HASHED != 0 {
            Some(v.get(8..8 + HASH_LEN).ok_or_else(|| {
                corrupt(n.block, format!("index entry {i} without its child's hash"))
            })?)
        } else {
            None
        };
        // A physical tree's child is never newer than its parent; a virtual tree's child is
        // rewritten through the object map without its parent.
        let node_xid = u64_at(&n.bytes, 16);
        let max_xid = match self.addressing {
            Addressing::Physical if node_xid != 0 => node_xid.min(self.max_xid),
            _ => self.max_xid,
        };
        let c = self.read_node(oid, paddr, noheader, false, max_xid, Some(&r.info), hash)?;
        if c.level + 1 != n.level {
            return Err(corrupt(
                paddr,
                format!(
                    "child at level {} under a node at level {}",
                    c.level, n.level
                ),
            ));
        }
        if c.nkeys == 0 {
            return Err(corrupt(paddr, "empty non-root node"));
        }
        let (ck, _) = c.entry(0)?;
        if ck.len() < 8 || (self.order)(k, ck) == Ordering::Greater {
            return Err(corrupt(
                paddr,
                "child's first key is below its parent's key for it",
            ));
        }
        Ok(c)
    }

    /// A node's block address, and whether it is stored without an object header.
    fn resolve(&self, oid: u64) -> Result<(u64, bool)> {
        match self.addressing {
            Addressing::Physical => Ok((oid, false)),
            Addressing::Virtual(m, xid) => {
                let (_, v) = self.c.omap_lookup(m, oid, xid)?.ok_or_else(|| {
                    corrupt(
                        0,
                        format!("virtual object {oid} is not in the object map at xid {xid}"),
                    )
                })?;
                if v.flags & omap::VAL_DELETED != 0 {
                    return Err(corrupt(v.paddr, format!("virtual object {oid} is deleted")));
                }
                if v.flags & omap::VAL_ENCRYPTED != 0 {
                    return Err(Error::Unsupported(format!("encrypted object {oid}")));
                }
                Ok((v.paddr, v.flags & omap::VAL_NOHEADER != 0))
            }
        }
    }

    /// Read node `oid` at `paddr`. A node with an object header has it checked; a node
    /// without one (a sealed volume's) is checked by `hash`, its SHA-256 as its parent or the
    /// seal records it, which a node with a header is held to as well where one is given.
    #[allow(clippy::too_many_arguments)]
    fn read_node(
        &self,
        oid: u64,
        paddr: u64,
        noheader: bool,
        root: bool,
        max_xid: u64,
        info: Option<&BtreeInfo>,
        hash: Option<&[u8]>,
    ) -> Result<Node> {
        let b = self.c.read_block(paddr)?;
        if let Some(h) = hash
            && Sha256::digest(&b)[..] != *h
        {
            return Err(corrupt(
                paddr,
                format!("node {oid}'s hash differs from the one recorded for it"),
            ));
        }
        if noheader || u16_at(&b, btree::FLAGS) & btree::NODE_NOHEADER != 0 {
            if hash.is_none() {
                return Err(Error::Unsupported(format!(
                    "headerless node {oid} without a hash to check it by"
                )));
            }
            if b[..32].iter().any(|x| *x != 0) {
                return Err(corrupt(paddr, "headerless node with a header"));
            }
            return self.parse_node(paddr, b, root, info);
        }
        let kind = if root { obj::BTREE } else { obj::BTREE_NODE };
        check_object(&b, paddr, Some(oid), Some(kind), max_xid)?;
        let st = u32_at(&b, 28);
        if st != self.subtype {
            return Err(corrupt(
                paddr,
                format!("tree node of subtype {st:#x}, expected {:#x}", self.subtype),
            ));
        }
        self.parse_node(paddr, b, root, info)
    }

    fn parse_node(
        &self,
        paddr: u64,
        b: Vec<u8>,
        root: bool,
        info: Option<&BtreeInfo>,
    ) -> Result<Node> {
        let flags = u16_at(&b, btree::FLAGS);
        if root != (flags & btree::NODE_ROOT != 0) {
            return Err(corrupt(
                paddr,
                if root {
                    "tree root without the root flag"
                } else {
                    "non-root node with the root flag"
                },
            ));
        }
        Node::parse(paddr, b, info)
    }
}
