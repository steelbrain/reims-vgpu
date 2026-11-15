//! Building one volume: its superblock, its object map, and its three trees — the file-system
//! tree (virtual, its nodes mapped through the volume's object map), and the extent-reference
//! and snapshot metadata trees (physical) — laid out as a freshly formatted volume's are.
//!
//! The caller gives the volume's file-system records (inodes, directory entries, attributes,
//! data-stream identifiers) and its data streams. The volume takes a run of blocks for each
//! stream, records where it is with a file extent in the file-system tree and a physical extent
//! in the extent-reference tree (one reference, owned by the stream's inode), and returns where
//! each stream goes, so the caller writes its bytes there.

use crate::fletcher;
use crate::fs::key_word;
use crate::ondisk::{APFS_MAGIC, apfs, obj, rec};

use super::geometry::BLOCK_SIZE;
use super::node::Layout;
use super::space::{XID, omap_record};
use super::tree::{self, Alloc, TreeError, TreeKind};

const BLOCK: usize = BLOCK_SIZE as usize;
/// The meta-crypto state a volume without encryption records: version 5, class 6, revision 1.
const META_CRYPTO: [u8; 20] = [
    0x05, 0, 0, 0, 0, 0, 0, 0, 0x06, 0, 0, 0, 0xb2, 0x01, 0x41, 0x1a, 0x01, 0, 0, 0,
];
/// A physical extent's kind: a new extent, as every extent a format writes is.
const PEXT_KIND_NEW: u64 = 1;

/// One data stream of the volume: its identifier (the inode's private identifier, or an
/// attribute stream's), the inode that owns it, and its length in bytes.
#[derive(Clone, Copy, Debug)]
pub struct Stream {
    pub id: u64,
    pub owner: u64,
    pub len: u64,
}

/// How many objects of each kind the volume holds, for its superblock.
#[derive(Clone, Copy, Debug, Default)]
pub struct Counts {
    pub files: u64,
    pub directories: u64,
    pub symlinks: u64,
    pub other: u64,
}

/// What a volume is built from.
pub struct VolumeSpec<'a> {
    /// The volume's slot in the container, from 0.
    pub index: u32,
    pub name: &'a str,
    pub uuid: [u8; 16],
    /// The volume's role (`apfs::ROLE_*`).
    pub role: u16,
    /// The volume's optional features (`0x2`, plus `0x10` for a volume-group System volume).
    pub features: u64,
    /// The volume's incompatible features (`0x1` case-insensitive).
    pub incompat: u64,
    /// The next object identifier the volume hands out.
    pub next_obj_id: u64,
    /// The volume group's identifier, or zero.
    pub group: [u8; 16],
    /// The instant the volume records as its own.
    pub t: u64,
    /// The file-system records but file extents, in any order.
    pub records: Vec<(Vec<u8>, Vec<u8>)>,
    /// The data streams, each given blocks of its own.
    pub streams: &'a [Stream],
    pub counts: Counts,
}

/// A built volume: its metadata blocks (block, bytes), the first block of each stream (in the
/// order given; 0 for an empty stream), and the container object-map record that maps the
/// superblock's virtual identifier to its block.
pub struct Volume {
    pub blocks: Vec<(u64, Vec<u8>)>,
    pub stream_blocks: Vec<u64>,
    pub sb_oid: u64,
    pub container_omap_record: (Vec<u8>, Vec<u8>),
}

/// A 4 KiB block being filled, little-endian.
pub(super) struct Block(pub(super) Vec<u8>);

impl Block {
    pub(super) fn new() -> Self {
        Self(vec![0u8; BLOCK])
    }
    pub(super) fn u16(&mut self, at: usize, v: u16) {
        self.0[at..at + 2].copy_from_slice(&v.to_le_bytes());
    }
    pub(super) fn u32(&mut self, at: usize, v: u32) {
        self.0[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }
    pub(super) fn u64(&mut self, at: usize, v: u64) {
        self.0[at..at + 8].copy_from_slice(&v.to_le_bytes());
    }
    pub(super) fn bytes(&mut self, at: usize, v: &[u8]) {
        self.0[at..at + v.len()].copy_from_slice(v);
    }
    pub(super) fn header(&mut self, oid: u64, xid: u64, type_and_flags: u32, subtype: u32) {
        self.u64(8, oid);
        self.u64(16, xid);
        self.u32(24, type_and_flags);
        self.u32(28, subtype);
    }
    pub(super) fn seal(mut self) -> Vec<u8> {
        fletcher::seal(&mut self.0);
        self.0
    }
}

/// The blocks a stream of `len` bytes takes.
#[must_use]
pub const fn blocks_of(len: u64) -> u64 {
    len.div_ceil(BLOCK_SIZE)
}

/// A file extent record: stream `id`'s bytes from `logical`, `len` bytes (whole blocks) at block
/// `phys`.
#[must_use]
pub fn file_extent(id: u64, logical: u64, len: u64, phys: u64) -> (Vec<u8>, Vec<u8>) {
    let mut k = key_word(id, rec::FILE_EXTENT).to_le_bytes().to_vec();
    k.extend_from_slice(&logical.to_le_bytes());
    let mut v = Vec::with_capacity(24);
    v.extend_from_slice(&len.to_le_bytes()); // len_and_flags, no flags
    v.extend_from_slice(&phys.to_le_bytes());
    v.extend_from_slice(&0u64.to_le_bytes()); // crypto_id
    (k, v)
}

/// A physical extent record: `blocks` blocks from `phys`, owned by `owner`, referenced once.
#[must_use]
pub fn phys_extent(phys: u64, blocks: u64, owner: u64) -> (Vec<u8>, Vec<u8>) {
    let k = key_word(phys, rec::EXTENT).to_le_bytes().to_vec();
    let mut v = Vec::with_capacity(20);
    v.extend_from_slice(&(blocks | (PEXT_KIND_NEW << 60)).to_le_bytes());
    v.extend_from_slice(&owner.to_le_bytes());
    v.extend_from_slice(&1i32.to_le_bytes());
    (k, v)
}

/// Build the volume `spec` describes, its blocks from `alloc` and its virtual identifiers from
/// `next_oid`.
///
/// # Errors
///
/// [`TreeError`] when the container has no room, or a record fits no node.
pub fn volume(
    spec: VolumeSpec,
    alloc: &mut Alloc,
    next_oid: &mut u64,
) -> Result<Volume, TreeError> {
    let first = alloc.next();
    let sb_oid = *next_oid;
    *next_oid += 1;

    // The streams' blocks, each one extent.
    let mut fs_records = spec.records;
    let mut extentref = Vec::new();
    let mut stream_blocks = Vec::with_capacity(spec.streams.len());
    let mut data_blocks = 0u64;
    for s in spec.streams {
        let n = blocks_of(s.len);
        if n == 0 {
            stream_blocks.push(0);
            continue;
        }
        let phys = alloc.take(n).ok_or(TreeError::NoSpace)?;
        stream_blocks.push(phys);
        data_blocks += n;
        fs_records.push(file_extent(s.id, 0, n * BLOCK_SIZE, phys));
        extentref.push(phys_extent(phys, n, s.owner));
    }
    fs_records.sort_by(|a, b| crate::fs::compare_keys(&a.0, &b.0));
    extentref.sort_by(|a, b| crate::fs::compare_keys(&a.0, &b.0));

    let mut blocks: Vec<(u64, Vec<u8>)> = Vec::new();
    let omap_obj = alloc.take(1).ok_or(TreeError::NoSpace)?;

    let physical = |subtype, tree_flags, layout| TreeKind {
        layout,
        subtype,
        tree_flags,
        virtual_nodes: false,
    };
    let er = tree::build(
        &extentref,
        physical(obj::BLOCKREFTREE, 0x52, Layout::Variable),
        alloc,
        next_oid,
    )?;
    let sm = tree::build(
        &[],
        physical(obj::SNAPMETATREE, 0x52, Layout::Variable),
        alloc,
        next_oid,
    )?;
    let fs = tree::build(
        &fs_records,
        TreeKind {
            layout: Layout::Variable,
            subtype: obj::FSTREE,
            tree_flags: 0x42,
            virtual_nodes: true,
        },
        alloc,
        next_oid,
    )?;
    let mut mapped: Vec<(Vec<u8>, Vec<u8>)> = fs
        .mapped
        .iter()
        .map(|&(oid, paddr)| omap_record(oid, XID, paddr))
        .collect();
    mapped.sort_by(|a, b| crate::fs::compare_u64_pairs(&a.0, &b.0));
    let om = tree::build(
        &mapped,
        physical(
            obj::OMAP,
            0x12,
            Layout::Fixed {
                key_size: 16,
                val_size: 16,
            },
        ),
        alloc,
        next_oid,
    )?;

    // The volume's object map (not manually managed), resolving the file-system tree's nodes.
    {
        let mut b = Block::new();
        b.header(omap_obj, XID, obj::PHYSICAL | obj::OMAP, 0);
        b.u32(32, 0); // om_flags
        b.u32(40, obj::PHYSICAL | obj::BTREE); // om_tree_type
        b.u32(44, obj::PHYSICAL | obj::BTREE); // om_snapshot_tree_type
        b.u64(48, om.root); // om_tree_oid
        blocks.push((omap_obj, b.seal()));
    }
    for t in [er.nodes, sm.nodes, fs.nodes, om.nodes] {
        blocks.extend(t);
    }

    // Every block the volume owns but its superblock.
    let alloc_count = alloc.next() - first;
    let sb = alloc.take(1).ok_or(TreeError::NoSpace)?;
    let mut b = superblock(&Super {
        index: spec.index,
        name: spec.name,
        uuid: spec.uuid,
        role: spec.role,
        features: spec.features,
        incompat: spec.incompat,
        fs_flags: apfs::FS_UNENCRYPTED,
        alloc_count,
        omap: omap_obj,
        root: fs.root,
        extentref: er.root,
        snapmeta: sm.root,
        next_obj_id: spec.next_obj_id,
        next_doc_id: 3,
        counts: spec.counts,
        data_blocks,
        group: spec.group,
        t: spec.t,
    });
    b.header(sb_oid, XID, obj::FS, 0);
    blocks.push((sb, b.seal()));

    Ok(Volume {
        blocks,
        stream_blocks,
        sb_oid,
        container_omap_record: omap_record(sb_oid, XID, sb),
    })
}

/// What a volume superblock records.
pub(super) struct Super<'a> {
    pub index: u32,
    pub name: &'a str,
    pub uuid: [u8; 16],
    pub role: u16,
    pub features: u64,
    pub incompat: u64,
    pub fs_flags: u64,
    /// Every block the volume owns but its superblock.
    pub alloc_count: u64,
    pub omap: u64,
    pub root: u64,
    pub extentref: u64,
    pub snapmeta: u64,
    pub next_obj_id: u64,
    pub next_doc_id: u32,
    pub counts: Counts,
    /// The blocks the volume's data streams take.
    pub data_blocks: u64,
    pub group: [u8; 16],
    pub t: u64,
}

/// A volume superblock of `s`, its object header left for the caller.
pub(super) fn superblock(s: &Super) -> Block {
    let mut b = Block::new();
    b.u32(apfs::MAGIC, APFS_MAGIC);
    b.u32(apfs::FS_INDEX, s.index);
    b.u64(apfs::FEATURES, s.features);
    b.u64(apfs::INCOMPAT, s.incompat);
    b.u64(apfs::ALLOC_COUNT, s.alloc_count);
    b.bytes(apfs::META_CRYPTO, &META_CRYPTO);
    b.u32(apfs::ROOT_TREE_TYPE, obj::BTREE); // virtual
    b.u32(apfs::EXTENTREF_TREE_TYPE, obj::PHYSICAL | obj::BTREE);
    b.u32(apfs::SNAP_META_TREE_TYPE, obj::PHYSICAL | obj::BTREE);
    b.u64(apfs::OMAP_OID, s.omap);
    b.u64(apfs::ROOT_TREE_OID, s.root);
    b.u64(apfs::EXTENTREF_TREE_OID, s.extentref);
    b.u64(apfs::SNAP_META_TREE_OID, s.snapmeta);
    b.u64(apfs::NEXT_OBJ_ID, s.next_obj_id);
    b.u64(apfs::NUM_FILES, s.counts.files);
    b.u64(apfs::NUM_DIRECTORIES, s.counts.directories);
    b.u64(apfs::NUM_SYMLINKS, s.counts.symlinks);
    b.u64(apfs::NUM_OTHER_FSOBJECTS, s.counts.other);
    b.u64(apfs::TOTAL_BLOCKS_ALLOCED, s.data_blocks);
    b.bytes(apfs::VOL_UUID, &s.uuid);
    b.u64(apfs::LAST_MOD_TIME, s.t);
    b.u64(apfs::FS_FLAGS, s.fs_flags);
    // The program that formatted it: a name, the instant, and this transaction.
    b.bytes(apfs::FORMATTED_BY, b"reims-restore");
    b.u64(apfs::FORMATTED_BY + 32, s.t);
    b.u64(apfs::FORMATTED_BY + 40, XID);
    b.bytes(apfs::VOLNAME, s.name.as_bytes());
    b.u32(apfs::NEXT_DOC_ID, s.next_doc_id);
    b.u16(apfs::ROLE, s.role);
    b.bytes(apfs::VOLUME_GROUP_ID, &s.group);
    b.u32(apfs::FEXT_TREE_TYPE, obj::PHYSICAL | obj::BTREE);
    b.u32(apfs::RESERVED_TYPE, obj::PHYSICAL | obj::BTREE);
    b
}
