//! APFS's on-disk constants and the offsets of the fields this crate reads, as the
//! Apple File System Reference (2020-06-22) defines them.

use crate::le::{u16_at, u32_at, u64_at, uuid_at};

/// The object header every non-headerless object starts with.
pub const OBJ_HEADER_SIZE: usize = 32;

pub mod obj {
    pub const TYPE_MASK: u32 = 0x0000_ffff;
    pub const VIRTUAL: u32 = 0;
    pub const EPHEMERAL: u32 = 0x8000_0000;
    pub const PHYSICAL: u32 = 0x4000_0000;
    pub const NOHEADER: u32 = 0x2000_0000;
    pub const ENCRYPTED: u32 = 0x1000_0000;
    pub const NONPERSISTENT: u32 = 0x0800_0000;
    pub const STORAGE_MASK: u32 = 0xc000_0000;

    pub const NX_SUPERBLOCK: u32 = 0x01;
    pub const BTREE: u32 = 0x02;
    pub const BTREE_NODE: u32 = 0x03;
    pub const SPACEMAN: u32 = 0x05;
    pub const SPACEMAN_CAB: u32 = 0x06;
    pub const SPACEMAN_CIB: u32 = 0x07;
    pub const SPACEMAN_BITMAP: u32 = 0x08;
    pub const SPACEMAN_FREE_QUEUE: u32 = 0x09;
    pub const EXTENT_LIST_TREE: u32 = 0x0a;
    pub const OMAP: u32 = 0x0b;
    pub const CHECKPOINT_MAP: u32 = 0x0c;
    pub const FS: u32 = 0x0d;
    pub const FSTREE: u32 = 0x0e;
    pub const BLOCKREFTREE: u32 = 0x0f;
    pub const SNAPMETATREE: u32 = 0x10;
    pub const NX_REAPER: u32 = 0x11;
    pub const NX_REAP_LIST: u32 = 0x12;
    pub const OMAP_SNAPSHOT: u32 = 0x13;
    pub const EFI_JUMPSTART: u32 = 0x14;
    pub const ER_STATE: u32 = 0x18;
    pub const SNAP_META_EXT: u32 = 0x1d;
    pub const INTEGRITY_META: u32 = 0x1e;
    pub const FEXT_TREE: u32 = 0x1f;
    /// 'keys', the container keybag.
    pub const CONTAINER_KEYBAG: u32 = 0x7379_656b;
    /// 'recs', a volume keybag.
    pub const VOLUME_KEYBAG: u32 = 0x7363_6572;
    /// 'mkey', the media keybag.
    pub const MEDIA_KEYBAG: u32 = 0x7965_6b6d;
}

/// An object header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObjHeader {
    pub cksum: u64,
    pub oid: u64,
    pub xid: u64,
    pub type_and_flags: u32,
    pub subtype: u32,
}

impl ObjHeader {
    #[must_use]
    pub fn parse(b: &[u8]) -> Self {
        Self {
            cksum: u64_at(b, 0),
            oid: u64_at(b, 8),
            xid: u64_at(b, 16),
            type_and_flags: u32_at(b, 24),
            subtype: u32_at(b, 28),
        }
    }

    #[must_use]
    pub const fn kind(&self) -> u32 {
        self.type_and_flags & obj::TYPE_MASK
    }

    #[must_use]
    pub const fn storage(&self) -> u32 {
        self.type_and_flags & obj::STORAGE_MASK
    }
}

pub const NX_MAGIC: u32 = 0x4253_584e;
pub const APFS_MAGIC: u32 = 0x4253_5041;
pub const NX_MAX_FILE_SYSTEMS: usize = 100;
pub const NX_NUM_COUNTERS: usize = 32;

/// Container superblock field offsets.
pub mod nx {
    pub const MAGIC: usize = 32;
    pub const BLOCK_SIZE: usize = 36;
    pub const BLOCK_COUNT: usize = 40;
    pub const FEATURES: usize = 48;
    pub const RO_COMPAT: usize = 56;
    pub const INCOMPAT: usize = 64;
    pub const UUID: usize = 72;
    pub const NEXT_OID: usize = 88;
    pub const NEXT_XID: usize = 96;
    pub const XP_DESC_BLOCKS: usize = 104;
    pub const XP_DATA_BLOCKS: usize = 108;
    pub const XP_DESC_BASE: usize = 112;
    pub const XP_DATA_BASE: usize = 120;
    pub const XP_DESC_NEXT: usize = 128;
    pub const XP_DATA_NEXT: usize = 132;
    pub const XP_DESC_INDEX: usize = 136;
    pub const XP_DESC_LEN: usize = 140;
    pub const XP_DATA_INDEX: usize = 144;
    pub const XP_DATA_LEN: usize = 148;
    pub const SPACEMAN_OID: usize = 152;
    pub const OMAP_OID: usize = 160;
    pub const REAPER_OID: usize = 168;
    pub const TEST_TYPE: usize = 176;
    pub const MAX_FILE_SYSTEMS: usize = 180;
    pub const FS_OID: usize = 184;
    pub const COUNTERS: usize = 984;
    pub const BLOCKED_OUT: usize = 1240;
    pub const EVICT_MAPPING_TREE_OID: usize = 1256;
    pub const FLAGS: usize = 1264;
    pub const EFI_JUMPSTART: usize = 1272;
    pub const FUSION_UUID: usize = 1280;
    pub const KEYLOCKER: usize = 1296;
    pub const EPHEMERAL_INFO: usize = 1312;
    pub const TEST_OID: usize = 1344;
    pub const FUSION_MT_OID: usize = 1352;
    pub const FUSION_WBC_OID: usize = 1360;
    pub const FUSION_WBC: usize = 1368;
    pub const NEWEST_MOUNTED_VERSION: usize = 1384;
    pub const MKB_LOCKER: usize = 1392;
    pub const END: usize = 1408;

    pub const INCOMPAT_VERSION2: u64 = 0x2;
    pub const INCOMPAT_FUSION: u64 = 0x100;
    pub const FLAG_CRYPTO_SW: u64 = 0x4;
}

/// A container superblock, as the fields this crate uses.
#[derive(Clone, Debug)]
pub struct NxSuperblock {
    pub hdr: ObjHeader,
    pub block_size: u32,
    pub block_count: u64,
    pub features: u64,
    pub ro_compat: u64,
    pub incompat: u64,
    pub uuid: [u8; 16],
    pub next_oid: u64,
    pub next_xid: u64,
    pub xp_desc_blocks: u32,
    pub xp_data_blocks: u32,
    pub xp_desc_base: u64,
    pub xp_data_base: u64,
    pub xp_desc_next: u32,
    pub xp_data_next: u32,
    pub xp_desc_index: u32,
    pub xp_desc_len: u32,
    pub xp_data_index: u32,
    pub xp_data_len: u32,
    pub spaceman_oid: u64,
    pub omap_oid: u64,
    pub reaper_oid: u64,
    pub max_file_systems: u32,
    pub fs_oid: Vec<u64>,
    pub flags: u64,
    pub efi_jumpstart: u64,
    pub keylocker: (u64, u64),
    pub mkb_locker: (u64, u64),
    pub ephemeral_info: [u64; 4],
    pub newest_mounted_version: u64,
    pub blocked_out: (u64, u64),
    pub evict_mapping_tree_oid: u64,
}

impl NxSuperblock {
    #[must_use]
    pub fn parse(b: &[u8]) -> Self {
        Self {
            hdr: ObjHeader::parse(b),
            block_size: u32_at(b, nx::BLOCK_SIZE),
            block_count: u64_at(b, nx::BLOCK_COUNT),
            features: u64_at(b, nx::FEATURES),
            ro_compat: u64_at(b, nx::RO_COMPAT),
            incompat: u64_at(b, nx::INCOMPAT),
            uuid: uuid_at(b, nx::UUID),
            next_oid: u64_at(b, nx::NEXT_OID),
            next_xid: u64_at(b, nx::NEXT_XID),
            xp_desc_blocks: u32_at(b, nx::XP_DESC_BLOCKS),
            xp_data_blocks: u32_at(b, nx::XP_DATA_BLOCKS),
            xp_desc_base: u64_at(b, nx::XP_DESC_BASE),
            xp_data_base: u64_at(b, nx::XP_DATA_BASE),
            xp_desc_next: u32_at(b, nx::XP_DESC_NEXT),
            xp_data_next: u32_at(b, nx::XP_DATA_NEXT),
            xp_desc_index: u32_at(b, nx::XP_DESC_INDEX),
            xp_desc_len: u32_at(b, nx::XP_DESC_LEN),
            xp_data_index: u32_at(b, nx::XP_DATA_INDEX),
            xp_data_len: u32_at(b, nx::XP_DATA_LEN),
            spaceman_oid: u64_at(b, nx::SPACEMAN_OID),
            omap_oid: u64_at(b, nx::OMAP_OID),
            reaper_oid: u64_at(b, nx::REAPER_OID),
            max_file_systems: u32_at(b, nx::MAX_FILE_SYSTEMS),
            fs_oid: (0..NX_MAX_FILE_SYSTEMS)
                .map(|i| u64_at(b, nx::FS_OID + 8 * i))
                .collect(),
            flags: u64_at(b, nx::FLAGS),
            efi_jumpstart: u64_at(b, nx::EFI_JUMPSTART),
            keylocker: (u64_at(b, nx::KEYLOCKER), u64_at(b, nx::KEYLOCKER + 8)),
            mkb_locker: (u64_at(b, nx::MKB_LOCKER), u64_at(b, nx::MKB_LOCKER + 8)),
            ephemeral_info: std::array::from_fn(|i| u64_at(b, nx::EPHEMERAL_INFO + 8 * i)),
            newest_mounted_version: u64_at(b, nx::NEWEST_MOUNTED_VERSION),
            blocked_out: (u64_at(b, nx::BLOCKED_OUT), u64_at(b, nx::BLOCKED_OUT + 8)),
            evict_mapping_tree_oid: u64_at(b, nx::EVICT_MAPPING_TREE_OID),
        }
    }
}

/// Volume superblock field offsets.
pub mod apfs {
    pub const MAGIC: usize = 32;
    pub const FS_INDEX: usize = 36;
    pub const FEATURES: usize = 40;
    pub const RO_COMPAT: usize = 48;
    pub const INCOMPAT: usize = 56;
    pub const UNMOUNT_TIME: usize = 64;
    pub const RESERVE_BLOCK_COUNT: usize = 72;
    pub const QUOTA_BLOCK_COUNT: usize = 80;
    pub const ALLOC_COUNT: usize = 88;
    pub const META_CRYPTO: usize = 96;
    pub const ROOT_TREE_TYPE: usize = 116;
    pub const EXTENTREF_TREE_TYPE: usize = 120;
    pub const SNAP_META_TREE_TYPE: usize = 124;
    pub const OMAP_OID: usize = 128;
    pub const ROOT_TREE_OID: usize = 136;
    pub const EXTENTREF_TREE_OID: usize = 144;
    pub const SNAP_META_TREE_OID: usize = 152;
    pub const REVERT_TO_XID: usize = 160;
    pub const REVERT_TO_SBLOCK_OID: usize = 168;
    pub const NEXT_OBJ_ID: usize = 176;
    pub const NUM_FILES: usize = 184;
    pub const NUM_DIRECTORIES: usize = 192;
    pub const NUM_SYMLINKS: usize = 200;
    pub const NUM_OTHER_FSOBJECTS: usize = 208;
    pub const NUM_SNAPSHOTS: usize = 216;
    pub const TOTAL_BLOCKS_ALLOCED: usize = 224;
    pub const TOTAL_BLOCKS_FREED: usize = 232;
    pub const VOL_UUID: usize = 240;
    pub const LAST_MOD_TIME: usize = 256;
    pub const FS_FLAGS: usize = 264;
    pub const FORMATTED_BY: usize = 272;
    pub const MODIFIED_BY: usize = 320;
    /// The size of one `apfs_modified_by_t`.
    pub const MODIFIED_BY_SIZE: usize = 48;
    pub const VOLNAME: usize = 704;
    pub const NEXT_DOC_ID: usize = 960;
    pub const ROLE: usize = 964;
    pub const ROOT_TO_XID: usize = 968;
    pub const ER_STATE_OID: usize = 976;
    pub const CLONEINFO_ID_EPOCH: usize = 984;
    pub const CLONEINFO_XID: usize = 992;
    pub const SNAP_META_EXT_OID: usize = 1000;
    pub const VOLUME_GROUP_ID: usize = 1008;
    pub const INTEGRITY_META_OID: usize = 1024;
    pub const FEXT_TREE_OID: usize = 1032;
    pub const FEXT_TREE_TYPE: usize = 1040;
    pub const RESERVED_TYPE: usize = 1044;
    pub const RESERVED_OID: usize = 1048;
    /// The end of the fields the reference names.
    pub const END: usize = 1056;

    pub const FEATURE_DEFRAG: u64 = 0x4;
    pub const FEATURE_HARDLINK_MAP_RECORDS: u64 = 0x2;
    pub const FEATURE_STRICTATIME: u64 = 0x8;
    pub const FEATURE_VOLGRP_SYSTEM_INO_SPACE: u64 = 0x10;

    pub const INCOMPAT_CASE_INSENSITIVE: u64 = 0x1;
    pub const INCOMPAT_DATALESS_SNAPS: u64 = 0x2;
    pub const INCOMPAT_ENC_ROLLED: u64 = 0x4;
    pub const INCOMPAT_NORMALIZATION_INSENSITIVE: u64 = 0x8;
    pub const INCOMPAT_INCOMPLETE_RESTORE: u64 = 0x10;
    pub const INCOMPAT_SEALED_VOLUME: u64 = 0x20;

    pub const FS_UNENCRYPTED: u64 = 0x1;
    pub const FS_ONEKEY: u64 = 0x8;

    pub const ROLE_NONE: u16 = 0;
    pub const ROLE_SYSTEM: u16 = 0x1;
    pub const ROLE_USER: u16 = 0x2;
    pub const ROLE_RECOVERY: u16 = 0x4;
    pub const ROLE_VM: u16 = 0x8;
    pub const ROLE_PREBOOT: u16 = 0x10;
    pub const ROLE_INSTALLER: u16 = 0x20;
    pub const ROLE_DATA: u16 = 0x40;
    pub const ROLE_BASEBAND: u16 = 0x80;
    pub const ROLE_UPDATE: u16 = 0xc0;
    pub const ROLE_XART: u16 = 0x100;
    pub const ROLE_HARDWARE: u16 = 0x140;
    pub const ROLE_BACKUP: u16 = 0x180;
    pub const ROLE_ENTERPRISE: u16 = 0x240;
    pub const ROLE_PRELOGIN: u16 = 0x2c0;
}

/// A volume superblock, as the fields this crate uses. The raw block is kept so a rewrite keeps
/// every field this crate does not name.
#[derive(Clone, Debug)]
pub struct ApfsSuperblock {
    pub hdr: ObjHeader,
    pub raw: Vec<u8>,
    pub fs_index: u32,
    pub features: u64,
    pub ro_compat: u64,
    pub incompat: u64,
    pub reserve_block_count: u64,
    pub quota_block_count: u64,
    pub alloc_count: u64,
    pub root_tree_type: u32,
    pub extentref_tree_type: u32,
    pub snap_meta_tree_type: u32,
    pub omap_oid: u64,
    pub root_tree_oid: u64,
    pub extentref_tree_oid: u64,
    pub snap_meta_tree_oid: u64,
    pub revert_to_xid: u64,
    pub next_obj_id: u64,
    pub num_files: u64,
    pub num_directories: u64,
    pub num_symlinks: u64,
    pub num_other_fsobjects: u64,
    pub num_snapshots: u64,
    pub total_blocks_alloced: u64,
    pub total_blocks_freed: u64,
    pub uuid: [u8; 16],
    pub last_mod_time: u64,
    pub fs_flags: u64,
    pub name: String,
    pub next_doc_id: u32,
    pub role: u16,
    pub root_to_xid: u64,
    pub er_state_oid: u64,
    pub cloneinfo_id_epoch: u64,
    pub cloneinfo_xid: u64,
    pub snap_meta_ext_oid: u64,
    pub volume_group_id: [u8; 16],
    pub integrity_meta_oid: u64,
    pub fext_tree_oid: u64,
    pub fext_tree_type: u32,
}

impl ApfsSuperblock {
    #[must_use]
    pub fn parse(b: &[u8]) -> Self {
        let name_bytes = &b[apfs::VOLNAME..apfs::VOLNAME + 256];
        let end = name_bytes.iter().position(|c| *c == 0).unwrap_or(256);
        Self {
            hdr: ObjHeader::parse(b),
            raw: b.to_vec(),
            fs_index: u32_at(b, apfs::FS_INDEX),
            features: u64_at(b, apfs::FEATURES),
            ro_compat: u64_at(b, apfs::RO_COMPAT),
            incompat: u64_at(b, apfs::INCOMPAT),
            reserve_block_count: u64_at(b, apfs::RESERVE_BLOCK_COUNT),
            quota_block_count: u64_at(b, apfs::QUOTA_BLOCK_COUNT),
            alloc_count: u64_at(b, apfs::ALLOC_COUNT),
            root_tree_type: u32_at(b, apfs::ROOT_TREE_TYPE),
            extentref_tree_type: u32_at(b, apfs::EXTENTREF_TREE_TYPE),
            snap_meta_tree_type: u32_at(b, apfs::SNAP_META_TREE_TYPE),
            omap_oid: u64_at(b, apfs::OMAP_OID),
            root_tree_oid: u64_at(b, apfs::ROOT_TREE_OID),
            extentref_tree_oid: u64_at(b, apfs::EXTENTREF_TREE_OID),
            snap_meta_tree_oid: u64_at(b, apfs::SNAP_META_TREE_OID),
            revert_to_xid: u64_at(b, apfs::REVERT_TO_XID),
            next_obj_id: u64_at(b, apfs::NEXT_OBJ_ID),
            num_files: u64_at(b, apfs::NUM_FILES),
            num_directories: u64_at(b, apfs::NUM_DIRECTORIES),
            num_symlinks: u64_at(b, apfs::NUM_SYMLINKS),
            num_other_fsobjects: u64_at(b, apfs::NUM_OTHER_FSOBJECTS),
            num_snapshots: u64_at(b, apfs::NUM_SNAPSHOTS),
            total_blocks_alloced: u64_at(b, apfs::TOTAL_BLOCKS_ALLOCED),
            total_blocks_freed: u64_at(b, apfs::TOTAL_BLOCKS_FREED),
            uuid: uuid_at(b, apfs::VOL_UUID),
            last_mod_time: u64_at(b, apfs::LAST_MOD_TIME),
            fs_flags: u64_at(b, apfs::FS_FLAGS),
            name: String::from_utf8_lossy(&name_bytes[..end]).into_owned(),
            next_doc_id: u32_at(b, apfs::NEXT_DOC_ID),
            role: u16_at(b, apfs::ROLE),
            root_to_xid: u64_at(b, apfs::ROOT_TO_XID),
            er_state_oid: u64_at(b, apfs::ER_STATE_OID),
            cloneinfo_id_epoch: u64_at(b, apfs::CLONEINFO_ID_EPOCH),
            cloneinfo_xid: u64_at(b, apfs::CLONEINFO_XID),
            snap_meta_ext_oid: u64_at(b, apfs::SNAP_META_EXT_OID),
            volume_group_id: uuid_at(b, apfs::VOLUME_GROUP_ID),
            integrity_meta_oid: u64_at(b, apfs::INTEGRITY_META_OID),
            fext_tree_oid: u64_at(b, apfs::FEXT_TREE_OID),
            fext_tree_type: u32_at(b, apfs::FEXT_TREE_TYPE),
        }
    }

    #[must_use]
    pub fn is_sealed(&self) -> bool {
        self.incompat & apfs::INCOMPAT_SEALED_VOLUME != 0
    }

    #[must_use]
    pub fn case_insensitive(&self) -> bool {
        self.incompat & apfs::INCOMPAT_CASE_INSENSITIVE != 0
    }

    /// The `apfs_modified_by` entries that are set: who, when, and the last transaction.
    #[must_use]
    pub fn modified_by(&self) -> Vec<(String, u64, u64)> {
        let mut v = Vec::new();
        let one = |off: usize| {
            let id = &self.raw[off..off + 32];
            let end = id.iter().position(|c| *c == 0).unwrap_or(32);
            (
                String::from_utf8_lossy(&id[..end]).into_owned(),
                u64_at(&self.raw, off + 32),
                u64_at(&self.raw, off + 40),
            )
        };
        v.push(one(apfs::FORMATTED_BY));
        for i in 0..8 {
            let e = one(apfs::MODIFIED_BY + i * apfs::MODIFIED_BY_SIZE);
            if !e.0.is_empty() {
                v.push(e);
            }
        }
        v
    }
}

/// Object map field offsets and flags.
pub mod omap {
    pub const FLAGS: usize = 32;
    pub const SNAP_COUNT: usize = 36;
    pub const TREE_TYPE: usize = 40;
    pub const SNAPSHOT_TREE_TYPE: usize = 44;
    pub const TREE_OID: usize = 48;
    pub const SNAPSHOT_TREE_OID: usize = 56;
    pub const MOST_RECENT_SNAP: usize = 64;
    pub const PENDING_REVERT_MIN: usize = 72;
    pub const PENDING_REVERT_MAX: usize = 80;
    pub const END: usize = 88;

    pub const MANUALLY_MANAGED: u32 = 0x1;

    pub const VAL_DELETED: u32 = 0x1;
    pub const VAL_SAVED: u32 = 0x2;
    pub const VAL_ENCRYPTED: u32 = 0x4;
    pub const VAL_NOHEADER: u32 = 0x8;
}

/// An object map header.
#[derive(Clone, Copy, Debug)]
pub struct OmapPhys {
    pub hdr: ObjHeader,
    pub flags: u32,
    pub snap_count: u32,
    pub tree_type: u32,
    pub snapshot_tree_type: u32,
    pub tree_oid: u64,
    pub snapshot_tree_oid: u64,
    pub most_recent_snap: u64,
}

impl OmapPhys {
    #[must_use]
    pub fn parse(b: &[u8]) -> Self {
        Self {
            hdr: ObjHeader::parse(b),
            flags: u32_at(b, omap::FLAGS),
            snap_count: u32_at(b, omap::SNAP_COUNT),
            tree_type: u32_at(b, omap::TREE_TYPE),
            snapshot_tree_type: u32_at(b, omap::SNAPSHOT_TREE_TYPE),
            tree_oid: u64_at(b, omap::TREE_OID),
            snapshot_tree_oid: u64_at(b, omap::SNAPSHOT_TREE_OID),
            most_recent_snap: u64_at(b, omap::MOST_RECENT_SNAP),
        }
    }
}

/// B-tree node flags, tree flags and layout.
pub mod btree {
    pub const NODE_ROOT: u16 = 0x1;
    pub const NODE_LEAF: u16 = 0x2;
    pub const NODE_FIXED_KV_SIZE: u16 = 0x4;
    pub const NODE_HASHED: u16 = 0x8;
    pub const NODE_NOHEADER: u16 = 0x10;
    pub const NODE_CHECK_KOFF_INVAL: u16 = 0x8000;

    pub const UINT64_KEYS: u32 = 0x1;
    pub const SEQUENTIAL_INSERT: u32 = 0x2;
    pub const ALLOW_GHOSTS: u32 = 0x4;
    pub const EPHEMERAL: u32 = 0x8;
    pub const PHYSICAL: u32 = 0x10;
    pub const NONPERSISTENT: u32 = 0x20;
    pub const KV_NONALIGNED: u32 = 0x40;
    pub const HASHED: u32 = 0x80;
    pub const NOHEADER: u32 = 0x100;

    /// Where a node's flags, level, key count and area locations sit.
    pub const FLAGS: usize = 32;
    pub const LEVEL: usize = 34;
    pub const NKEYS: usize = 36;
    pub const TABLE_SPACE: usize = 40;
    pub const FREE_SPACE: usize = 44;
    pub const KEY_FREE_LIST: usize = 48;
    pub const VAL_FREE_LIST: usize = 52;
    /// The start of a node's data, past its header.
    pub const DATA: usize = 56;
    /// The size of `btree_info_t`, at the end of a root node.
    pub const INFO_SIZE: usize = 40;
    /// An offset marking a value that is not there.
    pub const OFF_INVALID: u16 = 0xffff;
}

/// The B-tree info at the end of a root node.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BtreeInfo {
    pub flags: u32,
    pub node_size: u32,
    pub key_size: u32,
    pub val_size: u32,
    pub longest_key: u32,
    pub longest_val: u32,
    pub key_count: u64,
    pub node_count: u64,
}

impl BtreeInfo {
    #[must_use]
    pub fn parse(b: &[u8]) -> Self {
        Self {
            flags: u32_at(b, 0),
            node_size: u32_at(b, 4),
            key_size: u32_at(b, 8),
            val_size: u32_at(b, 12),
            longest_key: u32_at(b, 16),
            longest_val: u32_at(b, 20),
            key_count: u64_at(b, 24),
            node_count: u64_at(b, 32),
        }
    }
}

/// Checkpoint map entries.
pub mod cpm {
    pub const FLAGS: usize = 32;
    pub const COUNT: usize = 36;
    pub const MAP: usize = 40;
    pub const ENTRY_SIZE: usize = 40;
    pub const LAST: u32 = 0x1;
}

/// Integrity metadata (a sealed volume's) field offsets and values.
pub mod im {
    pub const FLAGS: usize = 36;
    pub const HASH_TYPE: usize = 40;
    pub const ROOT_HASH_OFFSET: usize = 44;
    pub const SEAL_BROKEN: u32 = 0x1;
    pub const HASH_SHA256: u32 = 1;
}

/// File-system record types: the top four bits of a key's first word.
pub mod rec {
    pub const SNAP_METADATA: u8 = 1;
    pub const EXTENT: u8 = 2;
    pub const INODE: u8 = 3;
    pub const XATTR: u8 = 4;
    pub const SIBLING_LINK: u8 = 5;
    pub const DSTREAM_ID: u8 = 6;
    pub const CRYPTO_STATE: u8 = 7;
    pub const FILE_EXTENT: u8 = 8;
    pub const DIR_REC: u8 = 9;
    pub const DIR_STATS: u8 = 10;
    pub const SNAP_NAME: u8 = 11;
    pub const SIBLING_MAP: u8 = 12;
    pub const FILE_INFO: u8 = 13;

    pub const OBJ_ID_MASK: u64 = 0x0fff_ffff_ffff_ffff;
    pub const TYPE_SHIFT: u32 = 60;

    #[must_use]
    pub const fn name(t: u8) -> &'static str {
        match t {
            SNAP_METADATA => "snap_metadata",
            EXTENT => "phys_extent",
            INODE => "inode",
            XATTR => "xattr",
            SIBLING_LINK => "sibling_link",
            DSTREAM_ID => "dstream_id",
            CRYPTO_STATE => "crypto_state",
            FILE_EXTENT => "file_extent",
            DIR_REC => "dir_rec",
            DIR_STATS => "dir_stats",
            SNAP_NAME => "snap_name",
            SIBLING_MAP => "sibling_map",
            FILE_INFO => "file_info",
            _ => "unknown",
        }
    }
}

/// Well-known inode numbers.
pub mod ino {
    pub const ROOT_DIR_PARENT: u64 = 1;
    pub const ROOT_DIR: u64 = 2;
    pub const PRIV_DIR: u64 = 3;
    pub const SNAP_DIR: u64 = 6;
    pub const PURGEABLE_DIR: u64 = 7;
    pub const MIN_USER: u64 = 16;
    /// The first inode number of a volume group's System volume.
    pub const UNIFIED_ID_SPACE_MARK: u64 = 0x0800_0000_0000_0000;
    pub const SYSTEM_INO_SPACE_BASE: u64 = 0x0fff_ffff_0000_0000;
}

/// Partition type GUIDs, in their usual string form.
pub mod part {
    /// An APFS container (`Apple_APFS`).
    pub const APFS: &str = "7C3457EF-0000-11AA-AA11-00306543ECAC";
    /// The iBoot system container (`Apple_APFS_ISC`).
    pub const APFS_ISC: &str = "69646961-6700-11AA-AA11-00306543ECAC";
    /// The recoveryOS container (`Apple_APFS_Recovery`).
    pub const APFS_RECOVERY: &str = "52637672-7900-11AA-AA11-00306543ECAC";
}
