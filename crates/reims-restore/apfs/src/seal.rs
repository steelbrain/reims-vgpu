//! Building a sealed volume's hashed file-system tree: pack a stream of records, in key order,
//! into headerless nodes, and hash each node into its parent to a signed root
//! (Apple File System Reference, "Sealed Volumes").
//!
//! The packing here is our own — records filled into each node until the next does not fit — not
//! a copy of how Apple's kernel lays its nodes out. The result is a self-consistent hashed tree:
//! every index entry carries the SHA-256 of the block it points at, and the root hash is the
//! SHA-256 of the root block. A reader that walks it checks out.

use sha2::{Digest, Sha256};

use crate::fs;
use crate::ondisk::{btree, obj};
use crate::write::node::{Layout, NodeSpec};

/// A file-system block holds a node; the sealed system volumes we build are 4 KiB.
pub const BLOCK: usize = 4096;
/// A node's header, before its table of contents.
const HDR: usize = 56;
/// The tree info at the end of a root node.
const INFO: usize = 40;
/// The sealed file-system tree's flags: headerless, hashed, non-aligned, sequential-insert.
pub const TREE_FLAGS: u32 =
    btree::NOHEADER | btree::HASHED | btree::KV_NONALIGNED | btree::SEQUENTIAL_INSERT;

/// One leaf record, in key order.
#[derive(Clone)]
pub struct Record {
    pub key: Vec<u8>,
    pub val: Vec<u8>,
}

/// A built node: its identifier, level, first key (its parent's separator), the SHA-256 of its
/// block, and the block itself.
pub struct BuiltNode {
    pub oid: u64,
    pub level: u16,
    pub first_key: Vec<u8>,
    pub hash: [u8; 32],
    pub block: Vec<u8>,
}

/// A built hashed tree.
pub struct SealedTree {
    /// Every node, parents after children (leaves first, root last).
    pub nodes: Vec<BuiltNode>,
    /// The root node's identifier. Index values name children relative to it, so the tree's
    /// blocks and hashes are the same wherever its identifiers start ([`SealedTree::rebase`]).
    pub root_oid: u64,
    /// The signed root hash: SHA-256 of the root block.
    pub root_hash: [u8; 32],
    /// The tree info recorded in the root node.
    pub longest_key: u32,
    pub longest_val: u32,
    pub key_count: u64,
}

impl SealedTree {
    /// The same tree with its identifiers starting at `root_oid`: its blocks do not change.
    pub fn rebase(&mut self, root_oid: u64) {
        for n in &mut self.nodes {
            n.oid = n.oid - self.root_oid + root_oid;
        }
        self.root_oid = root_oid;
    }
}

/// Divide a level's entries, given by their key and value sizes, among nodes each holding as
/// many as fit; returns each node's end (exclusive).
fn pack(n: usize, size: impl Fn(usize) -> (usize, usize), level: u16) -> Vec<usize> {
    let mut bounds = Vec::new();
    let mut i = 0;
    while i < n {
        let (mut kb, mut vb, mut cnt) = (0, 0, 0);
        while i + cnt < n {
            let (k, v) = size(i + cnt);
            if !Layout::Variable.fits(cnt + 1, kb + k, vb + v, false, level) {
                break;
            }
            kb += k;
            vb += v;
            cnt += 1;
        }
        // A record larger than a node is refused before a tree is built ([`build`]).
        i += cnt.max(1);
        bounds.push(i);
    }
    bounds
}

/// An index entry's value: the child's identifier relative to the root's, then its hash.
fn index_val(rel: u64, hash: &[u8; 32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(8 + 32);
    v.extend_from_slice(&rel.to_le_bytes());
    v.extend_from_slice(hash);
    v
}

/// Build a sealed hashed tree from `records` in key order: leaves filled until the next record
/// does not fit, then index levels the same way, each index entry its child's first key, its
/// identifier relative to the root and its SHA-256, until a level fits one node, the root.
/// `root_oid` is the root's identifier; the others follow it, leaves first.
///
/// # Panics
///
/// If `records` is empty, or a record does not fit a node.
#[must_use]
pub fn build(records: &[Record], root_oid: u64) -> SealedTree {
    assert!(
        !records.is_empty(),
        "a sealed tree needs at least one record"
    );
    for r in records {
        assert!(
            Layout::Variable.fits(1, r.key.len(), r.val.len(), false, 0),
            "a record of {} key and {} value bytes fits no node",
            r.key.len(),
            r.val.len()
        );
    }
    let longest_key = records.iter().map(|r| r.key.len()).max().unwrap_or(0) as u32;
    let longest_val = records.iter().map(|r| r.val.len()).max().unwrap_or(0) as u32;
    let key_count = records.len() as u64;

    let mut nodes: Vec<BuiltNode> = Vec::new();
    let mut next_rel = 1u64;
    let mut node =
        |entries: &[(Vec<u8>, Vec<u8>)], level: u16, is_root: bool, nodes: &mut Vec<BuiltNode>| {
            let rel = if is_root {
                0
            } else {
                next_rel += 1;
                next_rel - 1
            };
            let block = NodeSpec {
                obj_type: 0,
                subtype: obj::FSTREE,
                oid: 0,
                xid: 0,
                level,
                is_root,
                headerless: true,
                hashed: true,
                layout: Layout::Variable,
                tree_flags: TREE_FLAGS,
                longest_key,
                longest_val,
                key_count,
                node_count: nodes.len() as u64 + 1,
                entries,
            }
            .build();
            let hash: [u8; 32] = Sha256::digest(&block).into();
            nodes.push(BuiltNode {
                oid: root_oid + rel,
                level,
                first_key: entries[0].0.clone(),
                hash,
                block,
            });
            (rel, hash)
        };

    // The leaves. A level that fits one node is the root.
    let all = |n: usize, size: &dyn Fn(usize) -> (usize, usize), level: u16| {
        let (kb, vb) = (0..n).fold((0, 0), |(k, v), i| {
            let (a, b) = size(i);
            (k + a, v + b)
        });
        Layout::Variable.fits(n, kb, vb, true, level)
    };
    let leaf_size = |i: usize| (records[i].key.len(), records[i].val.len());
    let pairs = |r: &[Record]| -> Vec<(Vec<u8>, Vec<u8>)> {
        r.iter().map(|r| (r.key.clone(), r.val.clone())).collect()
    };
    let mut above: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    if all(records.len(), &leaf_size, 0) {
        node(&pairs(records), 0, true, &mut nodes);
    } else {
        let mut start = 0;
        for end in pack(records.len(), leaf_size, 0) {
            let (rel, hash) = node(&pairs(&records[start..end]), 0, false, &mut nodes);
            above.push((records[start].key.clone(), index_val(rel, &hash)));
            start = end;
        }
        // Index levels until one holds a level whole.
        let mut level = 1u16;
        loop {
            let entries = std::mem::take(&mut above);
            let size = |i: usize| (entries[i].0.len(), entries[i].1.len());
            if all(entries.len(), &size, level) {
                node(&entries, level, true, &mut nodes);
                break;
            }
            let mut start = 0;
            for end in pack(entries.len(), size, level) {
                let (rel, hash) = node(&entries[start..end], level, false, &mut nodes);
                above.push((entries[start].0.clone(), index_val(rel, &hash)));
                start = end;
            }
            level += 1;
        }
    }
    let root_hash = nodes.last().expect("a root").hash;
    SealedTree {
        nodes,
        root_oid,
        root_hash,
        longest_key,
        longest_val,
        key_count,
    }
}

/// Check a built tree is a self-consistent Merkle tree: every index entry's stored hash is the
/// SHA-256 of the block it names, each block's stored hash matches its bytes, and the root hash
/// is the SHA-256 of the root block. Returns the number of index entries checked, or an error
/// string naming the first inconsistency.
pub fn verify(tree: &SealedTree) -> Result<u64, String> {
    use std::collections::HashMap;
    let by_oid: HashMap<u64, &BuiltNode> = tree.nodes.iter().map(|n| (n.oid, n)).collect();
    let mut checked = 0u64;
    let mut root_seen = false;
    for n in &tree.nodes {
        let h: [u8; 32] = Sha256::digest(&n.block).into();
        if h != n.hash {
            return Err(format!("node {}: stored hash != SHA-256(block)", n.oid));
        }
        let flags = u16::from_le_bytes([n.block[32], n.block[33]]);
        let is_root = flags & btree::NODE_ROOT != 0;
        if is_root {
            root_seen = true;
            if n.hash != tree.root_hash {
                return Err("root node hash != tree.root_hash".into());
            }
        }
        if flags & btree::NODE_LEAF != 0 {
            continue;
        }
        // Index node: each value is child-oid-relative + child hash.
        let nkeys = u32::from_le_bytes([n.block[36], n.block[37], n.block[38], n.block[39]]);
        let toc = HDR;
        let val_end = BLOCK - if is_root { INFO } else { 0 };
        for i in 0..nkeys as usize {
            let t = toc + i * 8;
            let vo = u16::from_le_bytes([n.block[t + 4], n.block[t + 5]]) as usize;
            let vs = val_end - vo;
            let rel = u64::from_le_bytes(n.block[vs..vs + 8].try_into().unwrap());
            let child_hash = &n.block[vs + 8..vs + 40];
            let child_oid = tree.root_oid + rel;
            let child = by_oid
                .get(&child_oid)
                .ok_or_else(|| format!("node {}: child oid {child_oid} missing", n.oid))?;
            if child.hash[..] != *child_hash {
                return Err(format!(
                    "node {}: entry {i} hash != child {child_oid}'s block hash",
                    n.oid
                ));
            }
            if child.level + 1 != n.level {
                return Err(format!("node {}: child {child_oid} at wrong level", n.oid));
            }
            checked += 1;
        }
    }
    if !root_seen {
        return Err("no root node".into());
    }
    let _ = fs::key_header;
    Ok(checked)
}
