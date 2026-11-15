//! B-tree nodes: their table of contents, keys and values (Apple File System Reference,
//! "B-Trees").

use crate::le::{u16_at, u32_at};
use crate::ondisk::{BtreeInfo, btree};
use crate::{Result, corrupt};

/// The largest hash a hashed tree's index entry holds (`BTREE_NODE_HASH_SIZE_MAX`).
const MAX_HASH: usize = 64;

/// One node, parsed over its block.
#[derive(Clone, Debug)]
pub(crate) struct Node {
    pub block: u64,
    pub bytes: Vec<u8>,
    pub flags: u16,
    pub level: u16,
    pub nkeys: u32,
    key_start: usize,
    val_end: usize,
    toc_start: usize,
    /// Key and value sizes, for a node whose entries are fixed-size.
    fixed: Option<(usize, usize)>,
}

impl Node {
    /// Parse `bytes`, read from `block`. A fixed-size node's entry sizes come from its tree's
    /// info: `fixed` is the info's key and value sizes, where the tree has them (an index node
    /// of a fixed tree holds 8-byte child identifiers, plus a hash in a hashed tree).
    pub(crate) fn parse(block: u64, bytes: Vec<u8>, info: Option<&BtreeInfo>) -> Result<Self> {
        let len = bytes.len();
        let flags = u16_at(&bytes, btree::FLAGS);
        let level = u16_at(&bytes, btree::LEVEL);
        let nkeys = u32_at(&bytes, btree::NKEYS);
        let toc_off = usize::from(u16_at(&bytes, btree::TABLE_SPACE));
        let toc_len = usize::from(u16_at(&bytes, btree::TABLE_SPACE + 2));
        let toc_start = btree::DATA + toc_off;
        let key_start = toc_start + toc_len;
        let val_end = if flags & btree::NODE_ROOT != 0 {
            len - btree::INFO_SIZE
        } else {
            len
        };
        if key_start > val_end {
            return Err(corrupt(
                block,
                "B-tree node's table of contents runs past its value area",
            ));
        }
        let leaf = flags & btree::NODE_LEAF != 0;
        if leaf != (level == 0) {
            return Err(corrupt(
                block,
                format!("B-tree node flags {flags:#x} disagree with level {level}"),
            ));
        }
        let fixed = if flags & btree::NODE_FIXED_KV_SIZE != 0 {
            let root_info;
            let info = if flags & btree::NODE_ROOT != 0 {
                root_info = BtreeInfo::parse(&bytes[len - btree::INFO_SIZE..]);
                &root_info
            } else {
                info.ok_or_else(|| {
                    corrupt(block, "fixed-size B-tree node without its tree's info")
                })?
            };
            let vsize = if leaf {
                info.val_size as usize
            } else if info.flags & btree::HASHED != 0 {
                8 + MAX_HASH
            } else {
                8
            };
            Some((info.key_size as usize, vsize))
        } else {
            None
        };
        let entry = if fixed.is_some() { 4 } else { 8 };
        if (nkeys as usize) * entry > toc_len {
            return Err(corrupt(
                block,
                format!("{nkeys} keys do not fit a {toc_len}-byte table of contents"),
            ));
        }
        Ok(Self {
            block,
            bytes,
            flags,
            level,
            nkeys,
            key_start,
            val_end,
            toc_start,
            fixed,
        })
    }

    pub(crate) fn is_leaf(&self) -> bool {
        self.flags & btree::NODE_LEAF != 0
    }

    pub(crate) fn is_root(&self) -> bool {
        self.flags & btree::NODE_ROOT != 0
    }

    /// The tree info a root node ends with.
    pub(crate) fn info(&self) -> Option<BtreeInfo> {
        self.is_root()
            .then(|| BtreeInfo::parse(&self.bytes[self.bytes.len() - btree::INFO_SIZE..]))
    }

    /// Entry `i`'s key and value byte ranges; the value is `None` where its offset is invalid.
    pub(crate) fn entry(&self, i: u32) -> Result<(&[u8], Option<&[u8]>)> {
        let t = self.toc_start + i as usize * if self.fixed.is_some() { 4 } else { 8 };
        let (koff, klen, voff, vlen) = match self.fixed {
            Some((ks, vs)) => (
                usize::from(u16_at(&self.bytes, t)),
                ks,
                u16_at(&self.bytes, t + 2),
                vs,
            ),
            None => (
                usize::from(u16_at(&self.bytes, t)),
                usize::from(u16_at(&self.bytes, t + 2)),
                u16_at(&self.bytes, t + 4),
                usize::from(u16_at(&self.bytes, t + 6)),
            ),
        };
        let ks = self.key_start + koff;
        if ks + klen > self.val_end {
            return Err(corrupt(
                self.block,
                format!("entry {i}'s key runs past the key area"),
            ));
        }
        let key = &self.bytes[ks..ks + klen];
        if voff == btree::OFF_INVALID {
            return Ok((key, None));
        }
        let voff = usize::from(voff);
        if voff > self.val_end - self.key_start || vlen > voff {
            return Err(corrupt(
                self.block,
                format!("entry {i}'s value is outside the value area"),
            ));
        }
        let vs = self.val_end - voff;
        // A fixed tree's hashed index value is the child and a hash of the tree's hash size,
        // shorter than the maximum this crate assumes; clamp to the node.
        let ve = (vs + vlen).min(self.val_end);
        Ok((key, Some(&self.bytes[vs..ve])))
    }
}
