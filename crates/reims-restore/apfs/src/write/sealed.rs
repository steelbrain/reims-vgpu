//! Building a sealed system volume: its hashed file-system tree ([`crate::seal`]), headerless
//! and mapped through the volume's object map, and around it what the seal leaves out — the
//! physical file-extent tree that says where each data stream is, the integrity metadata that
//! records the root hash, and the one snapshot the volume boots from, named for that hash.
//!
//! Laid out as a sealed system volume of a fresh VZ restore is **[vz]**:
//! - the live superblock: incompatible features case-insensitive and sealed (`0x21`), flags
//!   `0x201`, one snapshot, an empty extent-reference tree, the integrity metadata, the
//!   snapshot's extended metadata and a file-extent tree of its own (physical, fixed 16-byte
//!   keys and values), which it counts as allocated on top of what the snapshot holds;
//! - the snapshot: a superblock copy (physical, named by its block) like the live one but with
//!   no object map, extent-reference or snapshot trees, no snapshots, one object identifier
//!   fewer and its own file-extent tree, the same records as the live one's in other blocks
//!   (`fsck_apfs` finds one tree reached from both overlapping, and counts it twice); its own
//!   extent-reference tree, a physical extent per data stream owned by its inode; and two
//!   records in the snapshot metadata tree, its metadata and its name;
//! - the object map: the tree's nodes mapped without headers, the integrity metadata and the
//!   snapshot's extended metadata mapped as ordinary objects, and a snapshot tree of one entry,
//!   the snapshot's transaction.
//!
//! The snapshot is taken in the one transaction that writes the container, as a freshly sealed
//! image's is before anything mounts it.

use crate::fs::key_word;
use crate::ondisk::{obj, omap, rec};
use crate::seal::SealedTree;

use super::geometry::BLOCK_SIZE;
use super::node::Layout;
use super::space::XID;
use super::tree::{self, Alloc, TreeError, TreeKind};
use super::volume::{Block, Counts, Stream, Super, Volume, blocks_of, phys_extent, superblock};

/// A sealed volume's incompatible features: case-insensitive, sealed.
const INCOMPAT: u64 = 0x21;
/// A sealed volume's flags: unencrypted, and `0x200`, which every sealed system volume of a VZ
/// restore carries **[vz]** and the reference does not name.
const FS_FLAGS: u64 = 0x201;
/// The integrity metadata's version, and its hash type (SHA-256).
const INTEGRITY_VERSION: u32 = 2;
const HASH_SHA256: u32 = 1;
/// Where the integrity metadata keeps the root hash.
const ROOT_HASH_OFFSET: u32 = 128;
/// The snapshot extended metadata's version.
const SNAP_META_EXT_VERSION: u32 = 1;
/// The snapshot name record's identifier: every bit of an object identifier.
const SNAP_NAME_OID: u64 = rec::OBJ_ID_MASK;

/// What seals a volume: its hashed tree, its data streams (each given its blocks, one extent),
/// and its snapshot's name and identifier.
pub struct Seal {
    /// The hashed file-system tree, its identifiers from any base; placed, they are rebased.
    pub tree: SealedTree,
    /// The data streams, each `len` a whole number of blocks.
    pub streams: Vec<Stream>,
    /// The snapshot's name, `com.apple.os.update-<root hash>`.
    pub snapshot: String,
    pub snapshot_uuid: [u8; 16],
    /// The next object identifier when the snapshot was taken; the live volume's is one more.
    pub inum: u64,
    pub next_doc_id: u32,
}

/// What a sealed volume is built from.
pub struct SealedSpec<'a> {
    pub index: u32,
    pub name: &'a str,
    pub uuid: [u8; 16],
    pub role: u16,
    pub features: u64,
    pub group: [u8; 16],
    pub t: u64,
    pub counts: Counts,
    pub seal: Seal,
}

/// A file-extent tree record: stream `id`'s bytes from `logical`, `len` bytes at block `phys`.
fn fext(id: u64, logical: u64, len: u64, phys: u64) -> (Vec<u8>, Vec<u8>) {
    let mut k = Vec::with_capacity(16);
    k.extend_from_slice(&id.to_le_bytes());
    k.extend_from_slice(&logical.to_le_bytes());
    let mut v = Vec::with_capacity(16);
    v.extend_from_slice(&len.to_le_bytes()); // len_and_flags, no flags
    v.extend_from_slice(&phys.to_le_bytes());
    (k, v)
}

/// An object-map record with flags.
fn omap_entry(oid: u64, flags: u32, paddr: u64) -> (Vec<u8>, Vec<u8>) {
    let mut k = Vec::with_capacity(16);
    k.extend_from_slice(&oid.to_le_bytes());
    k.extend_from_slice(&XID.to_le_bytes());
    let mut v = Vec::with_capacity(16);
    v.extend_from_slice(&flags.to_le_bytes());
    v.extend_from_slice(&(BLOCK_SIZE as u32).to_le_bytes());
    v.extend_from_slice(&paddr.to_le_bytes());
    (k, v)
}

/// The snapshot metadata tree's two records: the snapshot's metadata and its name.
fn snapshot_records(
    name: &str,
    extentref: u64,
    sblock: u64,
    t: u64,
    inum: u64,
) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut cname = name.as_bytes().to_vec();
    cname.push(0);
    let name_len = cname.len() as u16;

    let meta_k = key_word(XID, rec::SNAP_METADATA).to_le_bytes().to_vec();
    let mut meta_v = Vec::with_capacity(50 + cname.len());
    meta_v.extend_from_slice(&extentref.to_le_bytes());
    meta_v.extend_from_slice(&sblock.to_le_bytes());
    meta_v.extend_from_slice(&t.to_le_bytes()); // create_time
    meta_v.extend_from_slice(&t.to_le_bytes()); // change_time
    meta_v.extend_from_slice(&inum.to_le_bytes());
    meta_v.extend_from_slice(&(obj::PHYSICAL | obj::BTREE).to_le_bytes());
    meta_v.extend_from_slice(&0u32.to_le_bytes()); // flags
    meta_v.extend_from_slice(&name_len.to_le_bytes());
    meta_v.extend_from_slice(&cname);

    let mut name_k = key_word(SNAP_NAME_OID, rec::SNAP_NAME)
        .to_le_bytes()
        .to_vec();
    name_k.extend_from_slice(&name_len.to_le_bytes());
    name_k.extend_from_slice(&cname);
    let name_v = XID.to_le_bytes().to_vec();

    vec![(meta_k, meta_v), (name_k, name_v)]
}

/// Build the sealed volume `spec` describes, its blocks from `alloc` and its virtual identifiers
/// from `next_oid`.
///
/// # Errors
///
/// [`TreeError`] when the container has no room, or a record fits no node.
pub fn sealed_volume(
    spec: SealedSpec,
    alloc: &mut Alloc,
    next_oid: &mut u64,
) -> Result<Volume, TreeError> {
    let SealedSpec { seal, .. } = spec;
    let Seal {
        mut tree,
        streams,
        snapshot,
        snapshot_uuid,
        inum,
        next_doc_id,
    } = seal;
    let first = alloc.next();
    let sb_oid = *next_oid;
    *next_oid += 1;
    let mut blocks: Vec<(u64, Vec<u8>)> = Vec::new();

    // The data streams, each one extent: the file-extent tree says where, the snapshot's
    // extent-reference tree that its inode owns it.
    let mut fexts = Vec::with_capacity(streams.len());
    let mut extentref = Vec::with_capacity(streams.len());
    let mut stream_blocks = Vec::with_capacity(streams.len());
    let mut data_blocks = 0u64;
    for s in &streams {
        let n = blocks_of(s.len);
        if n == 0 {
            stream_blocks.push(0);
            continue;
        }
        let phys = alloc.take(n).ok_or(TreeError::NoSpace)?;
        stream_blocks.push(phys);
        data_blocks += n;
        fexts.push(fext(s.id, 0, n * BLOCK_SIZE, phys));
        extentref.push(phys_extent(phys, n, s.owner));
    }
    fexts.sort_by(|a, b| crate::fs::compare_u64_pairs(&a.0, &b.0));
    extentref.sort_by(|a, b| crate::fs::compare_keys(&a.0, &b.0));

    let omap_obj = alloc.take(1).ok_or(TreeError::NoSpace)?;

    // The hashed tree's nodes, headerless, their identifiers from here.
    let root_oid = *next_oid;
    *next_oid += tree.nodes.len() as u64;
    tree.rebase(root_oid);
    let mut mapped = Vec::with_capacity(tree.nodes.len() + 2);
    for n in tree.nodes {
        let paddr = alloc.take(1).ok_or(TreeError::NoSpace)?;
        mapped.push(omap_entry(n.oid, omap::VAL_NOHEADER, paddr));
        blocks.push((paddr, n.block));
    }

    // The integrity metadata, recording the root hash.
    let im_oid = *next_oid;
    *next_oid += 1;
    let im = alloc.take(1).ok_or(TreeError::NoSpace)?;
    {
        let mut b = Block::new();
        b.header(im_oid, XID, obj::INTEGRITY_META, 0);
        b.u32(32, INTEGRITY_VERSION);
        b.u32(36, 0); // flags
        b.u32(40, HASH_SHA256);
        b.u32(44, ROOT_HASH_OFFSET);
        b.bytes(ROOT_HASH_OFFSET as usize, &tree.root_hash);
        blocks.push((im, b.seal()));
    }
    mapped.push(omap_entry(im_oid, 0, im));

    // The snapshot's extended metadata.
    let sme_oid = *next_oid;
    *next_oid += 1;
    let sme = alloc.take(1).ok_or(TreeError::NoSpace)?;
    {
        let mut b = Block::new();
        b.header(sme_oid, XID, obj::SNAP_META_EXT, 0);
        b.u32(32, SNAP_META_EXT_VERSION);
        b.u32(36, 0); // flags
        b.u64(40, XID); // the snapshot's transaction
        b.bytes(48, &snapshot_uuid);
        b.u64(64, 0); // token
        blocks.push((sme, b.seal()));
    }
    mapped.push(omap_entry(sme_oid, 0, sme));
    mapped.sort_by(|a, b| crate::fs::compare_u64_pairs(&a.0, &b.0));

    let physical = |layout, subtype, tree_flags| TreeKind {
        layout,
        subtype,
        tree_flags,
        virtual_nodes: false,
    };
    let fixed = |key_size, val_size| Layout::Fixed { key_size, val_size };
    // The snapshot's file-extent tree; the live volume gets a copy of its own, last.
    let ft_snap = tree::build(
        &fexts,
        physical(fixed(16, 16), obj::FEXT_TREE, 0x12),
        alloc,
        next_oid,
    )?;
    let er_live = tree::build(
        &[],
        physical(Layout::Variable, obj::BLOCKREFTREE, 0x52),
        alloc,
        next_oid,
    )?;
    let er_snap = tree::build(
        &extentref,
        physical(Layout::Variable, obj::BLOCKREFTREE, 0x52),
        alloc,
        next_oid,
    )?;
    let snap_sb = alloc.take(1).ok_or(TreeError::NoSpace)?;
    let sm = tree::build(
        &snapshot_records(&snapshot, er_snap.root, snap_sb, spec.t, inum),
        physical(Layout::Variable, obj::SNAPMETATREE, 0x52),
        alloc,
        next_oid,
    )?;
    let om = tree::build(
        &mapped,
        physical(fixed(16, 16), obj::OMAP, 0x12),
        alloc,
        next_oid,
    )?;
    let snaps = [(XID.to_le_bytes().to_vec(), vec![0u8; 16])];
    let ost = tree::build(
        &snaps,
        physical(fixed(8, 16), obj::OMAP_SNAPSHOT, 0x10),
        alloc,
        next_oid,
    )?;

    // The volume's object map, with its one snapshot.
    {
        let mut b = Block::new();
        b.header(omap_obj, XID, obj::PHYSICAL | obj::OMAP, 0);
        b.u32(omap::FLAGS, 0);
        b.u32(omap::SNAP_COUNT, 1);
        b.u32(omap::TREE_TYPE, obj::PHYSICAL | obj::BTREE);
        b.u32(omap::SNAPSHOT_TREE_TYPE, obj::PHYSICAL | obj::BTREE);
        b.u64(omap::TREE_OID, om.root);
        b.u64(omap::SNAPSHOT_TREE_OID, ost.root);
        b.u64(omap::MOST_RECENT_SNAP, XID);
        blocks.push((omap_obj, b.seal()));
    }
    // Every block the volume owned when the snapshot was taken, but the superblocks (as a
    // volume's count leaves out its own superblock, `fsck_apfs` leaves out the snapshot's copy
    // too); then the live volume's own file-extent tree, and with it.
    let snap_alloc_count = alloc.next() - first - 1;
    let ft_live = tree::build(
        &fexts,
        physical(fixed(16, 16), obj::FEXT_TREE, 0x12),
        alloc,
        next_oid,
    )?;
    let live_alloc_count = alloc.next() - first - 1;
    for t in [
        ft_snap.nodes,
        ft_live.nodes,
        er_live.nodes,
        er_snap.nodes,
        sm.nodes,
        om.nodes,
        ost.nodes,
    ] {
        blocks.extend(t);
    }

    // The superblocks: the snapshot's copy, then the live one.
    let sb = alloc.take(1).ok_or(TreeError::NoSpace)?;
    let fields = |snapshot: bool| Super {
        index: spec.index,
        name: spec.name,
        uuid: spec.uuid,
        role: spec.role,
        features: spec.features,
        incompat: INCOMPAT,
        fs_flags: FS_FLAGS,
        alloc_count: if snapshot {
            snap_alloc_count
        } else {
            live_alloc_count
        },
        omap: if snapshot { 0 } else { omap_obj },
        root: root_oid,
        extentref: if snapshot { 0 } else { er_live.root },
        snapmeta: if snapshot { 0 } else { sm.root },
        next_obj_id: if snapshot { inum } else { inum + 1 },
        next_doc_id,
        counts: spec.counts,
        data_blocks,
        group: spec.group,
        t: spec.t,
    };
    let sealed = |b: &mut Block, snapshot: bool| {
        b.u64(crate::ondisk::apfs::NUM_SNAPSHOTS, u64::from(!snapshot));
        b.u64(crate::ondisk::apfs::INTEGRITY_META_OID, im_oid);
        b.u64(
            crate::ondisk::apfs::FEXT_TREE_OID,
            if snapshot { ft_snap.root } else { ft_live.root },
        );
        if !snapshot {
            b.u64(crate::ondisk::apfs::SNAP_META_EXT_OID, sme_oid);
        }
    };
    {
        let mut b = superblock(&fields(true));
        sealed(&mut b, true);
        b.header(snap_sb, XID, obj::PHYSICAL | obj::FS, 0);
        blocks.push((snap_sb, b.seal()));
    }
    {
        let mut b = superblock(&fields(false));
        sealed(&mut b, false);
        b.header(sb_oid, XID, obj::FS, 0);
        blocks.push((sb, b.seal()));
    }

    Ok(Volume {
        blocks,
        stream_blocks,
        sb_oid,
        container_omap_record: super::space::omap_record(sb_oid, XID, sb),
    })
}
