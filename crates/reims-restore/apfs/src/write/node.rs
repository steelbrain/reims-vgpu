//! Serialising one APFS B-tree node — its header, table of contents, keys and values, and, for a
//! root, the tree info at the end — laid out as `read.rs` reads one back and as Apple's own nodes
//! are (Apple File System Reference, "B-Trees").
//!
//! A node is one 4 KiB block. Its 32-byte object header carries a Fletcher-64 checksum, its
//! identifier, transaction, type and subtype, unless the tree is headerless (a sealed volume's
//! file-system tree), when the first 32 bytes stay zero and the node is hashed into its parent
//! instead. Past the header sit the flags, level and key count, the locations of the table of
//! contents, the free gap and the two free lists, then the data area: the table of contents, the
//! keys packed upward, the free gap, and the values packed downward from the end (before the tree
//! info of a root).

use crate::fletcher;
use crate::ondisk::btree;

/// A node is one block.
pub const BLOCK: usize = 4096;
/// The data area begins past the 56-byte node header.
const DATA: usize = btree::DATA;
/// The tree info at the end of a root node.
const INFO: usize = btree::INFO_SIZE;

/// How a node stores its keys and values.
#[derive(Clone, Copy, Debug)]
pub enum Layout {
    /// Keys and values of varying size; each table entry is four 16-bit offsets and lengths
    /// (key offset, key length, value offset, value length).
    Variable,
    /// Keys and values of one size each, kept in the tree info; each table entry is two 16-bit
    /// offsets (key offset, value offset).
    Fixed { key_size: u16, val_size: u16 },
}

impl Layout {
    /// The table of contents' size in bytes for a node of `nkeys` entries, for a root (or not).
    ///
    /// A variable tree grows its table in 64-byte chunks of eight 8-byte entries, with one chunk
    /// the minimum. A fixed tree sizes its table to the node's full capacity — as many fixed
    /// entries as the block holds past the header and the tree info — rounded up to 64 bytes, so
    /// a node can fill without growing it.
    fn toc_bytes(self, nkeys: usize, is_root: bool, level: u16) -> usize {
        match self {
            Self::Variable => nkeys.div_ceil(8).max(1) * 64,
            Self::Fixed { .. } => (self.capacity(is_root, level) * 4).div_ceil(64) * 64,
        }
    }

    /// The entries a node of a fixed tree holds: as many as fit past the header (and a root's
    /// tree info), each a 4-byte table entry, a key and a value — the value of an index node
    /// being a child's 8-byte identifier rather than a leaf's value. Unbounded for a variable
    /// tree, whose nodes fill by bytes.
    #[must_use]
    pub fn capacity(self, is_root: bool, level: u16) -> usize {
        match self {
            Self::Variable => usize::MAX,
            Self::Fixed { key_size, val_size } => {
                let tail = if is_root { INFO } else { 0 };
                let val = if level == 0 { usize::from(val_size) } else { 8 };
                (BLOCK - DATA - tail) / (4 + usize::from(key_size) + val)
            }
        }
    }

    /// Whether `nkeys` entries of `keybytes` and `valbytes` in all fit one node.
    #[must_use]
    pub fn fits(
        self,
        nkeys: usize,
        keybytes: usize,
        valbytes: usize,
        is_root: bool,
        level: u16,
    ) -> bool {
        match self {
            Self::Variable => {
                let tail = if is_root { INFO } else { 0 };
                DATA + self.toc_bytes(nkeys, is_root, level) + keybytes + valbytes + tail <= BLOCK
            }
            Self::Fixed { .. } => nkeys <= self.capacity(is_root, level),
        }
    }

    const fn is_fixed(self) -> bool {
        matches!(self, Self::Fixed { .. })
    }
}

/// What a node is built from.
pub struct NodeSpec<'a> {
    /// The object type with its storage flags (`0x4000_0002` physical, `0x0000_0002` virtual),
    /// or `0x03` for a non-root node. Ignored where the tree is headerless.
    pub obj_type: u32,
    /// The tree's subtype: omap `0x0b`, file-system tree `0x0e`, extent reference `0x0f`,
    /// snapshot metadata `0x10`, file-extent tree `0x1f`.
    pub subtype: u32,
    /// The node's identifier: a physical node's block address, a virtual node's object
    /// identifier. Ignored where the tree is headerless.
    pub oid: u64,
    /// The transaction that wrote it.
    pub xid: u64,
    /// 0 for a leaf, rising toward the root.
    pub level: u16,
    /// Whether this is the tree's root, which carries the tree info.
    pub is_root: bool,
    /// Whether the tree is headerless (a sealed file-system tree), its first 32 bytes left zero.
    pub headerless: bool,
    /// Whether each index entry carries the hash of its child (a sealed file-system tree).
    pub hashed: bool,
    /// How keys and values are stored.
    pub layout: Layout,
    /// The B-tree info flags, where this is a root (`0x12` omap, `0x42` file-system tree, `0x52`
    /// extent reference and snapshot metadata, `0x1c2` a sealed file-system tree).
    pub tree_flags: u32,
    /// The longest key and value in the whole tree, for the tree info.
    pub longest_key: u32,
    pub longest_val: u32,
    /// The number of leaf records and of nodes in the whole tree, for the tree info.
    pub key_count: u64,
    pub node_count: u64,
    /// The node's entries, already in key order.
    pub entries: &'a [(Vec<u8>, Vec<u8>)],
}

impl NodeSpec<'_> {
    /// Serialise the node into one block.
    #[must_use]
    pub fn build(&self) -> Vec<u8> {
        let mut b = vec![0u8; BLOCK];
        let nkeys = self.entries.len();
        let is_leaf = self.level == 0;

        let mut flags = 0u16;
        if self.is_root {
            flags |= btree::NODE_ROOT;
        }
        if is_leaf {
            flags |= btree::NODE_LEAF;
        }
        if self.layout.is_fixed() {
            flags |= btree::NODE_FIXED_KV_SIZE;
        }
        if self.hashed {
            flags |= btree::NODE_HASHED;
        }
        if self.headerless {
            flags |= btree::NODE_NOHEADER;
        }
        b[btree::FLAGS..btree::FLAGS + 2].copy_from_slice(&flags.to_le_bytes());
        b[btree::LEVEL..btree::LEVEL + 2].copy_from_slice(&self.level.to_le_bytes());
        b[btree::NKEYS..btree::NKEYS + 4].copy_from_slice(&(nkeys as u32).to_le_bytes());

        let toc = self.layout.toc_bytes(nkeys, self.is_root, self.level);
        // btn_table_space: offset 0 from the data area, length the table size.
        b[btree::TABLE_SPACE..btree::TABLE_SPACE + 2].copy_from_slice(&0u16.to_le_bytes());
        b[btree::TABLE_SPACE + 2..btree::TABLE_SPACE + 4]
            .copy_from_slice(&(toc as u16).to_le_bytes());

        let kbase = DATA + toc;
        let val_end = BLOCK - if self.is_root { INFO } else { 0 };
        let keybytes: usize = self.entries.iter().map(|(k, _)| k.len()).sum();
        let valbytes: usize = self.entries.iter().map(|(_, v)| v.len()).sum();
        // btn_free_space: offset from the key area where the gap starts (past the keys), length
        // the gap between the keys and the values.
        let free_off = keybytes;
        let free_len = (val_end - valbytes) - (kbase + keybytes);
        b[btree::FREE_SPACE..btree::FREE_SPACE + 2]
            .copy_from_slice(&(free_off as u16).to_le_bytes());
        b[btree::FREE_SPACE + 2..btree::FREE_SPACE + 4]
            .copy_from_slice(&(free_len as u16).to_le_bytes());
        // Both free lists empty: our nodes carry no freed fragments.
        b[btree::KEY_FREE_LIST..btree::KEY_FREE_LIST + 2]
            .copy_from_slice(&btree::OFF_INVALID.to_le_bytes());
        b[btree::KEY_FREE_LIST + 2..btree::KEY_FREE_LIST + 4].copy_from_slice(&0u16.to_le_bytes());
        b[btree::VAL_FREE_LIST..btree::VAL_FREE_LIST + 2]
            .copy_from_slice(&btree::OFF_INVALID.to_le_bytes());
        b[btree::VAL_FREE_LIST + 2..btree::VAL_FREE_LIST + 4].copy_from_slice(&0u16.to_le_bytes());

        // Keys packed upward from the key area; values packed downward from the value-area end.
        let fixed = self.layout.is_fixed();
        let mut koff = 0usize;
        let mut voff = 0usize;
        for (i, (k, v)) in self.entries.iter().enumerate() {
            let ko = koff;
            b[kbase + ko..kbase + ko + k.len()].copy_from_slice(k);
            voff += v.len();
            let vo = voff;
            b[val_end - vo..val_end - vo + v.len()].copy_from_slice(v);
            if fixed {
                let t = DATA + i * 4;
                b[t..t + 2].copy_from_slice(&(ko as u16).to_le_bytes());
                b[t + 2..t + 4].copy_from_slice(&(vo as u16).to_le_bytes());
            } else {
                let t = DATA + i * 8;
                b[t..t + 2].copy_from_slice(&(ko as u16).to_le_bytes());
                b[t + 2..t + 4].copy_from_slice(&(k.len() as u16).to_le_bytes());
                b[t + 4..t + 6].copy_from_slice(&(vo as u16).to_le_bytes());
                b[t + 6..t + 8].copy_from_slice(&(v.len() as u16).to_le_bytes());
            }
            koff += k.len();
        }

        if self.is_root {
            let (ks, vs) = match self.layout {
                Layout::Variable => (0u32, 0u32),
                Layout::Fixed { key_size, val_size } => (u32::from(key_size), u32::from(val_size)),
            };
            let info = BLOCK - INFO;
            b[info..info + 4].copy_from_slice(&self.tree_flags.to_le_bytes());
            b[info + 4..info + 8].copy_from_slice(&(BLOCK as u32).to_le_bytes());
            b[info + 8..info + 12].copy_from_slice(&ks.to_le_bytes());
            b[info + 12..info + 16].copy_from_slice(&vs.to_le_bytes());
            b[info + 16..info + 20].copy_from_slice(&self.longest_key.to_le_bytes());
            b[info + 20..info + 24].copy_from_slice(&self.longest_val.to_le_bytes());
            b[info + 24..info + 32].copy_from_slice(&self.key_count.to_le_bytes());
            b[info + 32..info + 40].copy_from_slice(&self.node_count.to_le_bytes());
        }

        if !self.headerless {
            b[8..16].copy_from_slice(&self.oid.to_le_bytes());
            b[16..24].copy_from_slice(&self.xid.to_le_bytes());
            b[24..28].copy_from_slice(&self.obj_type.to_le_bytes());
            b[28..32].copy_from_slice(&self.subtype.to_le_bytes());
            fletcher::seal(&mut b);
        }
        b
    }
}
