//! Building a container's scaffold: the superblock, its one checkpoint, the space manager and its
//! internal pool, the reaper, and the container object map. These are the blocks every container
//! has before any volume, laid out as a freshly formatted container's are (the bare container a
//! `newfs_apfs` leaves, before a volume is added).
//!
//! The writer forms a single checkpoint, transaction 1, holding everything, with empty free
//! queues — the state a bare `newfs_apfs` container is in, which `fsck_apfs` accepts. The space
//! manager's main device marks every block through the internal pool, and then the blocks the
//! object map and the volumes take, as used in one chunk bitmap; the internal pool's own bitmap
//! marks the chunk-info and bitmap blocks it holds. Blocks the container does not touch stay zero,
//! so a file destination stays sparse.

use crate::fletcher;
use crate::ondisk::{NX_MAGIC, cpm, nx, obj};

use super::geometry::{BLOCK_SIZE, Geometry};
use super::node::{Layout, NodeSpec};

/// A block is 4 KiB.
const BLOCK: usize = BLOCK_SIZE as usize;
/// The space manager is ephemeral object 0x400, the reaper 0x401; volumes and their trees take
/// virtual identifiers from 0x402 up.
pub const SPACEMAN_OID: u64 = 0x400;
pub const REAPER_OID: u64 = 0x401;
pub const FIRST_VOLUME_OID: u64 = 0x402;
/// The one transaction a fresh container is written in.
pub const XID: u64 = 1;

/// A 4 KiB block being filled, little-endian.
struct Block(Vec<u8>);

impl Block {
    fn new() -> Self {
        Self(vec![0u8; BLOCK])
    }
    fn u16(&mut self, at: usize, v: u16) {
        self.0[at..at + 2].copy_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, at: usize, v: u32) {
        self.0[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, at: usize, v: u64) {
        self.0[at..at + 8].copy_from_slice(&v.to_le_bytes());
    }
    /// Write the object header (checksum last) for a non-headerless object.
    fn header(&mut self, oid: u64, xid: u64, type_and_flags: u32, subtype: u32) {
        self.u64(8, oid);
        self.u64(16, xid);
        self.u32(24, type_and_flags);
        self.u32(28, subtype);
    }
    fn seal(mut self) -> Vec<u8> {
        fletcher::seal(&mut self.0);
        self.0
    }
}

/// The container object map's record for a volume: its superblock's virtual identifier maps to
/// the block the superblock is at. Fixed 16-byte key `(oid, xid)` and 16-byte value `(flags,
/// size, paddr)`.
#[must_use]
pub fn omap_record(oid: u64, xid: u64, paddr: u64) -> (Vec<u8>, Vec<u8>) {
    let mut k = Vec::with_capacity(16);
    k.extend_from_slice(&oid.to_le_bytes());
    k.extend_from_slice(&xid.to_le_bytes());
    let mut v = Vec::with_capacity(16);
    v.extend_from_slice(&0u32.to_le_bytes()); // flags
    v.extend_from_slice(&(BLOCK as u32).to_le_bytes()); // size
    v.extend_from_slice(&paddr.to_le_bytes());
    (k, v)
}

/// Build a container's scaffold for `block_count` blocks, `uuid` its identifier, with the object
/// map holding `omap_entries` (already in key order) and `used_after_pool` blocks taken past the
/// internal pool by the object map and the volumes (so the main device's free count and chunk
/// bitmap are right). The object map's own two blocks are included in `used_after_pool` by the
/// caller.
#[must_use]
pub fn scaffold(
    g: &Geometry,
    uuid: [u8; 16],
    omap_entries: &[(Vec<u8>, Vec<u8>)],
    fs_oids: &[u64],
    next_oid: u64,
    used_after_pool: u64,
) -> Vec<(u64, Vec<u8>)> {
    let block_count = g.block_count;
    let mut out: Vec<(u64, Vec<u8>)> = Vec::new();

    // The object map sits first past the pool, its root next.
    let omap_oid = g.first_free;
    let omap_root = g.first_free + 1;
    // Blocks used on the main device: everything through the pool, then what the caller placed.
    let used = g.first_free + used_after_pool;

    // --- Container superblock (block 0 and the descriptor copy) ---
    let nxsb = |copy_oid: u64| -> Vec<u8> {
        let mut b = Block::new();
        b.header(copy_oid, XID, obj::EPHEMERAL | obj::NX_SUPERBLOCK, 0);
        b.u32(nx::MAGIC, NX_MAGIC);
        b.u32(nx::BLOCK_SIZE, BLOCK as u32);
        b.u64(nx::BLOCK_COUNT, block_count);
        b.u64(nx::INCOMPAT, nx::INCOMPAT_VERSION2);
        b.0[nx::UUID..nx::UUID + 16].copy_from_slice(&uuid);
        b.u64(nx::NEXT_OID, next_oid);
        b.u64(nx::NEXT_XID, XID + 1);
        b.u32(nx::XP_DESC_BLOCKS, g.desc_blocks as u32);
        b.u32(nx::XP_DATA_BLOCKS, g.data_blocks as u32);
        b.u64(nx::XP_DESC_BASE, 1);
        b.u64(nx::XP_DATA_BASE, g.data_base);
        b.u32(nx::XP_DESC_NEXT, 2);
        b.u32(nx::XP_DATA_NEXT, 2);
        b.u32(nx::XP_DESC_INDEX, 0);
        b.u32(nx::XP_DESC_LEN, 2);
        b.u32(nx::XP_DATA_INDEX, 0);
        b.u32(nx::XP_DATA_LEN, 2);
        b.u64(nx::SPACEMAN_OID, SPACEMAN_OID);
        b.u64(nx::OMAP_OID, omap_oid);
        b.u64(nx::REAPER_OID, REAPER_OID);
        b.u32(nx::MAX_FILE_SYSTEMS, g.max_volumes as u32);
        for (i, oid) in fs_oids.iter().enumerate() {
            b.u64(nx::FS_OID + 8 * i, *oid);
        }
        // The ephemeral object ring: minimum blocks and one ring, as a fresh container records.
        b.u64(nx::EPHEMERAL_INFO, 0x8_0004_0001);
        b.seal()
    };
    out.push((0, nxsb(1)));
    out.push((2, nxsb(1)));

    // --- Checkpoint map (block 1): maps the two ephemeral objects to their data blocks ---
    {
        let mut b = Block::new();
        b.header(1, XID, obj::PHYSICAL | obj::CHECKPOINT_MAP, 0);
        b.u32(cpm::FLAGS, cpm::LAST);
        b.u32(cpm::COUNT, 2);
        let mut entry = |i: usize, ty: u32, oid: u64, paddr: u64| {
            let o = cpm::MAP + i * cpm::ENTRY_SIZE;
            b.u32(o, ty); // cpm_type
            b.u32(o + 4, 0); // cpm_subtype
            b.u32(o + 8, BLOCK as u32); // cpm_size
            b.u32(o + 12, 0); // cpm_pad
            b.u64(o + 16, 0); // cpm_fs_oid
            b.u64(o + 24, oid); // cpm_oid
            b.u64(o + 32, paddr); // cpm_paddr
        };
        entry(0, obj::EPHEMERAL | obj::SPACEMAN, SPACEMAN_OID, g.data_base);
        entry(
            1,
            obj::EPHEMERAL | obj::NX_REAPER,
            REAPER_OID,
            g.data_base + 1,
        );
        out.push((1, b.seal()));
    }

    // --- Space manager (data block 0), its chunk-info block and bitmaps ---
    out.extend(space_manager(g, block_count, used));

    // --- Reaper (data block 1) ---
    {
        let mut b = Block::new();
        b.header(REAPER_OID, XID, obj::EPHEMERAL | obj::NX_REAPER, 0);
        b.u64(32, 1); // nr_next_reap_id
        b.u32(64, 1); // nr_flags: NR_BHM_FLAG, always set
        b.u32(108, (BLOCK - 112) as u32); // nr_state_buffer_size
        out.push((g.data_base + 1, b.seal()));
    }

    // --- Container object map and its (possibly empty) tree ---
    {
        let mut b = Block::new();
        b.header(omap_oid, XID, obj::PHYSICAL | obj::OMAP, 0);
        b.u32(32, 1); // om_flags: OMAP_MANUALLY_MANAGED
        b.u32(36, 0); // om_snap_count
        b.u32(40, obj::PHYSICAL | obj::BTREE); // om_tree_type
        b.u32(44, obj::PHYSICAL | obj::BTREE); // om_snapshot_tree_type
        b.u64(48, omap_root); // om_tree_oid
        out.push((omap_oid, b.seal()));
    }
    let longest = if omap_entries.is_empty() { 0 } else { 16 };
    out.push((
        omap_root,
        NodeSpec {
            obj_type: obj::PHYSICAL | obj::BTREE,
            subtype: obj::OMAP,
            oid: omap_root,
            xid: XID,
            level: 0,
            is_root: true,
            headerless: false,
            hashed: false,
            layout: Layout::Fixed {
                key_size: 16,
                val_size: 16,
            },
            tree_flags: 0x12,
            longest_key: longest,
            longest_val: longest,
            key_count: omap_entries.len() as u64,
            node_count: 1,
            entries: omap_entries,
        }
        .build(),
    ));

    out
}

/// The space manager object, its chunk-info block, the main device's one chunk bitmap and the
/// internal pool's bitmap — the blocks that record which of the container's blocks are used.
fn space_manager(g: &Geometry, block_count: u64, used: u64) -> Vec<(u64, Vec<u8>)> {
    let mut out = Vec::new();
    let bpc = super::geometry::BLOCKS_PER_CHUNK;
    let per_cib = super::geometry::CHUNKS_PER_CIB;
    assert!(
        g.cab_count == 0,
        "a container of more than {} chunk-info blocks",
        super::geometry::CIBS_PER_CAB
    );

    // The blocks of each chunk that are used: the used blocks are the prefix [0, used).
    let chunk_used = |c: u64| -> (u64, u64) {
        let start = c * bpc;
        let blocks = bpc.min(block_count - start);
        (blocks, used.saturating_sub(start).min(blocks))
    };
    // The internal pool holds the chunk-info blocks first, then a bitmap for each chunk with any
    // block used, in chunk order.
    let cib_addr = |i: u64| g.pool_base + i;
    let mut bitmap_addr = vec![0u64; g.chunk_count as usize];
    let mut next_pool = g.pool_base + g.cib_count;
    for c in 0..g.chunk_count {
        if chunk_used(c).1 > 0 {
            bitmap_addr[c as usize] = next_pool;
            next_pool += 1;
        }
    }
    let pool_used = next_pool - g.pool_base;
    assert!(
        pool_used <= g.pool_blocks,
        "the internal pool holds every chunk's bitmap"
    );

    // --- The space manager object itself (one block) ---
    {
        let mut b = Block::new();
        b.header(SPACEMAN_OID, XID, obj::EPHEMERAL | obj::SPACEMAN, 0);
        b.u32(32, BLOCK as u32); // sm_block_size
        b.u32(36, bpc as u32); // sm_blocks_per_chunk
        b.u32(40, per_cib as u32); // sm_chunks_per_cib
        b.u32(44, super::geometry::CIBS_PER_CAB as u32);
        // Main device (offset 48): block/chunk/cib/cab counts, free count, address offset.
        b.u64(48, block_count);
        b.u64(48 + 8, g.chunk_count);
        b.u32(48 + 16, g.cib_count as u32);
        b.u32(48 + 20, g.cab_count as u32);
        b.u64(48 + 24, block_count - used); // main device free count
        b.u32(48 + 32, g.address_offset as u32);
        // The second device, which a container on one device keeps empty, lists from where the
        // main device's list ends.
        b.u32(96 + 32, g.second_address_offset as u32);
        b.u32(144, 1); // sm_flags: SM_FLAG_VERSIONED
        b.u32(148, 16); // sm_ip_bm_tx_multiplier
        b.u64(152, g.pool_blocks); // sm_ip_block_count
        b.u32(160, g.pool_bitmaps as u32); // sm_ip_bm_size_in_blocks
        b.u32(164, g.pool_bitmap_slots() as u32); // sm_ip_bm_block_count
        b.u64(168, g.pool_bitmap_base); // sm_ip_bm_base
        b.u64(176, g.pool_base); // sm_ip_base
        // Three free queues (offsets 200/240/280), empty, with their node limits.
        b.u16(200 + 24, g.fq_pool_limit as u16);
        b.u16(240 + 24, g.fq_main_limit as u16);
        // The internal-pool bitmap ring: slot 0 is the live bitmap, slots 1..n a free list.
        let ring = g.pool_bitmap_slots();
        b.u16(320, 1); // sm_ip_bm_free_head
        b.u16(322, (ring - 1) as u16); // sm_ip_bm_free_tail
        b.u32(324, g.xid_array_offset as u32);
        b.u32(328, g.slot_array_offset as u32);
        b.u32(332, g.free_list_offset as u32);
        b.u32(336, 1); // sm_version
        b.u32(340, super::geometry::SPACEMAN_STRUCT_SIZE as u32);
        // Trailing arrays. The current internal-pool bitmap (one block) is in ring slot 0,
        // written in this transaction.
        b.u64(g.xid_array_offset as usize, XID);
        b.u16(g.slot_array_offset as usize, 0);
        // The free list threads slots 1..ring-1, each to the next, the last to none; slot 0,
        // in use, points nowhere.
        let fl = g.free_list_offset as usize;
        for slot in 0..ring as usize {
            let next = if slot == 0 || slot == ring as usize - 1 {
                0xffff
            } else {
                (slot + 1) as u16
            };
            b.u16(fl + slot * 2, next);
        }
        // The main device's address list: the chunk-info blocks.
        let ao = g.address_offset as usize;
        for i in 0..g.cib_count {
            b.u64(ao + i as usize * 8, cib_addr(i));
        }
        out.push((g.data_base, b.seal()));
    }

    // --- The chunk-info blocks: each describes up to 126 chunks ---
    for i in 0..g.cib_count {
        let first = i * per_cib;
        let count = per_cib.min(g.chunk_count - first);
        let mut b = Block::new();
        b.header(cib_addr(i), XID, obj::PHYSICAL | obj::SPACEMAN_CIB, 0);
        b.u32(32, i as u32); // cib_index
        b.u32(36, count as u32); // cib_chunk_info_count
        for j in 0..count {
            let c = first + j;
            let at = 40 + j as usize * 32;
            let (blocks, in_use) = chunk_used(c);
            b.u64(at, XID); // ci_xid
            b.u64(at + 8, c * bpc); // ci_addr
            b.u32(at + 16, blocks as u32); // ci_block_count
            b.u32(at + 20, (blocks - in_use) as u32); // ci_free_count
            b.u64(at + 24, bitmap_addr[c as usize]); // ci_bitmap_addr, 0 where all free
        }
        out.push((cib_addr(i), b.seal()));
    }

    // --- Each used chunk's bitmap: its used blocks' bits set ---
    for c in 0..g.chunk_count {
        let (_, in_use) = chunk_used(c);
        if in_use > 0 {
            let mut bm = vec![0u8; BLOCK];
            set_bits(&mut bm, in_use as usize);
            out.push((bitmap_addr[c as usize], bm));
        }
    }

    // --- The internal pool's bitmap (ring slot 0): its chunk-info and bitmap blocks used ---
    {
        let mut bm = vec![0u8; BLOCK];
        set_bits(&mut bm, pool_used as usize);
        out.push((g.pool_bitmap_base, bm));
    }

    out
}

/// Set the low `n` bits of a bitmap block (bit `i` is bit `i & 7` of byte `i >> 3`), the way a
/// chunk bitmap marks blocks `[0, n)` used.
fn set_bits(bm: &mut [u8], n: usize) {
    for byte in bm.iter_mut().take(n / 8) {
        *byte = 0xff;
    }
    let rem = n % 8;
    if rem != 0 {
        bm[n / 8] = ((1u16 << rem) - 1) as u8;
    }
}
