//! Where a freshly formatted APFS container's structures go, computed from its block count the
//! way `newfs_apfs` sizes them (and `ferrosys`, whose formulas these match, records as observed
//! on every container macOS formats).
//!
//! A container of 4 KiB blocks is laid out from block 0: the container superblock, then the
//! checkpoint descriptor area, the checkpoint data area, the space manager's internal-pool bitmap
//! ring and its internal pool, and then — past the pool — the object maps, volumes and trees. The
//! descriptor and data areas are sized to hold the checkpoints a container is formatted with and
//! the room its free queues may grow into; the internal pool holds every chunk-info, address and
//! bitmap block three times over. This module computes those sizes and the first block past the
//! pool, where the writer begins placing everything else.

/// Every block is 4 KiB; the writer forms no other size.
pub const BLOCK_SIZE: u64 = 4096;
/// A chunk is a block's worth of bits: 32768 blocks.
pub const BLOCKS_PER_CHUNK: u64 = BLOCK_SIZE * 8;
/// A chunk-info block describes this many chunks (past its 40-byte header, 32 bytes each).
pub const CHUNKS_PER_CIB: u64 = (BLOCK_SIZE - 40) / 32;
/// A chunk-info address block holds this many addresses (past its 40-byte header, 8 bytes each).
pub const CIBS_PER_CAB: u64 = (BLOCK_SIZE - 40) / 8;
/// The internal-pool bitmap ring keeps sixteen slots per pool bitmap block.
pub const POOL_BITMAP_SLOTS: u64 = 16;
/// The fixed part of a `spaceman_phys`, where its trailing arrays begin.
pub const SPACEMAN_STRUCT_SIZE: u64 = 2520;
/// The most volumes a container records room for.
pub const MAX_VOLUMES: u64 = 100;
/// A volume slot is kept for every 512 MiB of container.
const BYTES_PER_VOLUME_SLOT: u64 = 512 << 20;
/// A checkpoint-mapping block maps this many objects (past its 40-byte header, 40 bytes each).
const MAPPINGS_PER_BLOCK: u64 = (BLOCK_SIZE - 40) / 40;
/// A free-queue node holds this many entries (past the node header, tree info and an uncounted
/// head, 28 bytes each).
const QUEUE_ENTRIES_PER_NODE: u64 = (BLOCK_SIZE - 56 - 40 - 24) / 28;
/// A free queue is held to two mebibytes of nodes.
const QUEUE_NODE_CAP: u64 = (2 << 20) / BLOCK_SIZE;

/// A container's layout: the counts and block addresses a fresh format places.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub block_count: u64,
    pub chunk_count: u64,
    pub cib_count: u64,
    pub cab_count: u64,
    /// The internal pool's blocks (`sm_ip_block_count`).
    pub pool_blocks: u64,
    /// Pool bitmap blocks, each with sixteen ring slots (`sm_ip_bm_block_count` = 16× this).
    pub pool_bitmaps: u64,
    /// The checkpoint descriptor area, from block 1.
    pub desc_blocks: u64,
    /// The checkpoint data area, past the descriptor area.
    pub data_base: u64,
    pub data_blocks: u64,
    /// The internal-pool bitmap ring, past the data area (`sm_ip_bm_base`).
    pub pool_bitmap_base: u64,
    /// The internal pool, past the bitmap ring (`sm_ip_base`).
    pub pool_base: u64,
    /// The first block past the pool, where object maps, volumes and trees begin.
    pub first_free: u64,
    /// Volume slots the container records room for (`nx_max_file_systems`).
    pub max_volumes: u64,
    /// The smaller of the main free queue's node limit and eight.
    pub ephemeral_min_blocks: u64,
    /// The internal pool's and main device's free-queue node limits.
    pub fq_pool_limit: u64,
    pub fq_main_limit: u64,
    /// The space manager's fixed structure in blocks (one, for our sizes).
    pub manager_blocks: u64,
    /// The offsets of the space manager's trailing arrays.
    pub xid_array_offset: u64,
    pub slot_array_offset: u64,
    pub free_list_offset: u64,
    pub address_offset: u64,
    pub second_address_offset: u64,
}

/// The node limit a free queue of `entries` entries takes, held to the two-mebibyte cap: a queue
/// that needs two nodes is given three, as `newfs_apfs` does.
fn queue_nodes(entries: u64) -> u64 {
    let needed = entries.div_ceil(QUEUE_ENTRIES_PER_NODE).max(1);
    let needed = if needed == 2 { 3 } else { needed };
    needed.min(QUEUE_NODE_CAP)
}

/// The volume slots a container of `block_count` blocks is formatted with by default: one per
/// 512 MiB, at most 100. A container is given at least as many as the volumes it holds.
#[must_use]
pub fn volume_capacity(block_count: u64) -> u64 {
    (block_count * BLOCK_SIZE)
        .div_ceil(BYTES_PER_VOLUME_SLOT)
        .min(MAX_VOLUMES)
}

impl Geometry {
    /// The layout a container of `block_count` 4 KiB blocks is formatted with, with room for the
    /// default number of volumes.
    #[must_use]
    pub fn of(block_count: u64) -> Self {
        Self::with_max_volumes(block_count, volume_capacity(block_count))
    }

    /// The layout a container of `block_count` 4 KiB blocks is formatted with, with room for
    /// `max_volumes` volumes — which sizes the checkpoint areas, so the writer gives a container
    /// at least as many slots as it holds volumes.
    #[must_use]
    pub fn with_max_volumes(block_count: u64, max_volumes: u64) -> Self {
        let chunk_count = block_count.div_ceil(BLOCKS_PER_CHUNK);
        let cib_count = chunk_count.div_ceil(CHUNKS_PER_CIB);
        let cab_count = if cib_count > CIBS_PER_CAB {
            cib_count.div_ceil(CIBS_PER_CAB)
        } else {
            0
        };
        let pool_blocks = 3 * (chunk_count + cib_count + cab_count);
        let pool_bitmaps = pool_blocks.div_ceil(BLOCKS_PER_CHUNK);
        let slots = POOL_BITMAP_SLOTS * pool_bitmaps;
        let listed = if cab_count > 0 { cab_count } else { cib_count };

        // The space manager's trailing arrays, and the block it needs.
        let xid_array_offset = SPACEMAN_STRUCT_SIZE;
        let slot_array_offset = xid_array_offset + 8 * pool_bitmaps;
        let free_list_offset = slot_array_offset + (2 * pool_bitmaps).div_ceil(8) * 8;
        let address_offset = free_list_offset + (2 * slots).div_ceil(8) * 8;
        let second_address_offset = address_offset + 8 * listed;
        let manager_blocks = second_address_offset.div_ceil(BLOCK_SIZE);

        let fq_pool_limit = queue_nodes(pool_blocks.div_ceil(8));
        let share = match block_count {
            n if n < 1 << 18 => 32,
            n if n < 1 << 20 => 16,
            _ => 8,
        };
        let fq_main_limit = queue_nodes(block_count.div_ceil(share));
        let ephemeral_min_blocks = fq_main_limit.min(8);

        // The data area holds the checkpoints and the room the free queues may grow into; the
        // descriptor area holds enough mapping blocks to map all of it, rounded up to four, and
        // four more.
        let data_blocks = 4
            + 8 * manager_blocks
            + 32 * ephemeral_min_blocks * max_volumes
            + 4 * (fq_pool_limit + fq_main_limit);
        let map_blocks = data_blocks.div_ceil(MAPPINGS_PER_BLOCK);
        let desc_blocks = 4 * map_blocks.div_ceil(4) + 4;

        let data_base = 1 + desc_blocks;
        let pool_bitmap_base = data_base + data_blocks;
        let pool_base = pool_bitmap_base + slots;
        let first_free = pool_base + pool_blocks;

        Self {
            block_count,
            chunk_count,
            cib_count,
            cab_count,
            pool_blocks,
            pool_bitmaps,
            desc_blocks,
            data_base,
            data_blocks,
            pool_bitmap_base,
            pool_base,
            first_free,
            max_volumes,
            ephemeral_min_blocks,
            fq_pool_limit,
            fq_main_limit,
            manager_blocks,
            xid_array_offset,
            slot_array_offset,
            free_list_offset,
            address_offset,
            second_address_offset,
        }
    }

    /// The ring slots the pool bitmap area holds (`sm_ip_bm_block_count`).
    #[must_use]
    pub const fn pool_bitmap_slots(&self) -> u64 {
        POOL_BITMAP_SLOTS * self.pool_bitmaps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every value measured on a container diskutil/newfs formatted, or formatted by ferrosys
    // (which matches newfs): the iBoot system container (500 MiB), the recoveryOS container
    // (~5 GiB), a 60 GiB main container, and a 256 GiB one.
    #[test]
    fn geometry_matches_formatted_containers() {
        let isc = Geometry::of(128_000);
        assert_eq!(isc.chunk_count, 4);
        assert_eq!(isc.cib_count, 1);
        assert_eq!(isc.pool_blocks, 15);
        assert_eq!((isc.desc_blocks, isc.data_blocks), (8, 388));
        assert_eq!(
            (isc.data_base, isc.pool_bitmap_base, isc.pool_base),
            (9, 397, 413)
        );
        assert_eq!((isc.fq_pool_limit, isc.fq_main_limit), (1, 29));
        assert_eq!(isc.max_volumes, 1);
        assert_eq!(isc.address_offset, 2568);
        assert_eq!(isc.second_address_offset, 2576);

        let recos = Geometry::of(1_310_710);
        assert_eq!(
            (recos.chunk_count, recos.cib_count, recos.pool_blocks),
            (40, 1, 123)
        );
        assert_eq!((recos.desc_blocks, recos.data_blocks), (52, 4624));
        assert_eq!((recos.pool_bitmap_base, recos.pool_base), (4677, 4693));
        assert_eq!((recos.fq_pool_limit, recos.fq_main_limit), (1, 512));
        assert_eq!(recos.max_volumes, 10);

        let main = Geometry::of(15_333_160);
        assert_eq!(
            (main.chunk_count, main.cib_count, main.pool_blocks),
            (468, 4, 1416)
        );
        assert_eq!((main.desc_blocks, main.data_blocks), (280, 27672));
        assert_eq!((main.pool_bitmap_base, main.pool_base), (27953, 27969));
        assert_eq!((main.fq_pool_limit, main.fq_main_limit), (3, 512));
        assert_eq!(main.max_volumes, 100);
        assert_eq!(main.second_address_offset, 2600);

        let big = Geometry::of(67_108_864);
        assert_eq!(
            (big.chunk_count, big.cib_count, big.pool_blocks),
            (2048, 17, 6195)
        );
        assert_eq!((big.desc_blocks, big.data_blocks), (280, 27684));
        assert_eq!((big.fq_pool_limit, big.fq_main_limit), (6, 512));
    }

    #[test]
    fn the_pool_is_marked_used_and_everything_else_is_free() {
        // Chunk 0 holds every structure through the pool; the writer then places objects from
        // first_free. For the iBoot container the pool ends at 428, as the formatted one's does.
        let isc = Geometry::of(128_000);
        assert_eq!(isc.first_free, 428);
    }
}
