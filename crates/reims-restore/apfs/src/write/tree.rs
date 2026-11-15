//! Building a whole B-tree from records in key order: leaves packed until the next record does
//! not fit, then each level of index nodes above them the same way, until one node — the root —
//! holds a level. Every node is a block of its own; the caller's allocator gives each its block.
//!
//! An index entry is its child's first key and the child's identifier, eight bytes: the child's
//! block address in a physical tree, its virtual identifier in a virtual tree (a volume's
//! file-system tree), which the volume's object map then resolves. A virtual tree's nodes are
//! returned with the object-map records that map them.

use crate::ondisk::obj;

use super::node::{Layout, NodeSpec};
use super::space::XID;

/// Hands out blocks one run after another, from a first block up to an end.
#[derive(Debug)]
pub struct Alloc {
    next: u64,
    end: u64,
}

impl Alloc {
    /// An allocator of the blocks `[start, end)`.
    #[must_use]
    pub const fn new(start: u64, end: u64) -> Self {
        Self { next: start, end }
    }

    /// The first of `n` contiguous blocks, or `None` when fewer than `n` are left.
    pub fn take(&mut self, n: u64) -> Option<u64> {
        let at = self.next;
        let next = at.checked_add(n)?;
        (next <= self.end).then(|| {
            self.next = next;
            at
        })
    }

    /// The first block not yet handed out.
    #[must_use]
    pub const fn next(&self) -> u64 {
        self.next
    }
}

/// What kind of tree to build.
#[derive(Clone, Copy, Debug)]
pub struct TreeKind {
    pub layout: Layout,
    /// The tree's subtype (`obj::OMAP`, `obj::FSTREE`, `obj::BLOCKREFTREE`, ...).
    pub subtype: u32,
    /// The B-tree info flags the root records.
    pub tree_flags: u32,
    /// Whether the tree's nodes are virtual objects (resolved by an object map) rather than
    /// physical ones (named by their block).
    pub virtual_nodes: bool,
}

/// A built tree: its root's identifier, every node as (block, bytes), and, for a virtual tree,
/// the object-map records (virtual identifier, block) of its nodes.
pub struct Built {
    pub root: u64,
    pub nodes: Vec<(u64, Vec<u8>)>,
    pub mapped: Vec<(u64, u64)>,
}

/// What ran out building a tree.
#[derive(Debug, thiserror::Error)]
pub enum TreeError {
    #[error("the container has no room for the tree's nodes")]
    NoSpace,
    #[error("a record of {key} key and {val} value bytes fits no node")]
    TooBig { key: usize, val: usize },
}

/// Divide `entries` among nodes of `level`, each holding as many as fit, the last the rest.
fn partition(
    entries: &[(Vec<u8>, Vec<u8>)],
    layout: Layout,
    level: u16,
) -> Result<Vec<std::ops::Range<usize>>, TreeError> {
    let mut out = Vec::new();
    let mut start = 0;
    while start < entries.len() {
        let (mut kb, mut vb) = (0usize, 0usize);
        let mut end = start;
        while end < entries.len() {
            let (k, v) = &entries[end];
            if !layout.fits(end - start + 1, kb + k.len(), vb + v.len(), false, level) {
                break;
            }
            kb += k.len();
            vb += v.len();
            end += 1;
        }
        if end == start {
            let (k, v) = &entries[start];
            return Err(TreeError::TooBig {
                key: k.len(),
                val: v.len(),
            });
        }
        out.push(start..end);
        start = end;
    }
    Ok(out)
}

/// Whether all of `entries` fit one root node at `level`.
fn fits_root(entries: &[(Vec<u8>, Vec<u8>)], layout: Layout, level: u16) -> bool {
    let kb: usize = entries.iter().map(|(k, _)| k.len()).sum();
    let vb: usize = entries.iter().map(|(_, v)| v.len()).sum();
    layout.fits(entries.len(), kb, vb, true, level)
}

/// Build a tree of `kind` from `records`, in key order, its nodes' blocks taken from `alloc` and,
/// for a virtual tree, their identifiers from `next_oid`.
///
/// # Errors
///
/// [`TreeError::NoSpace`] when `alloc` runs out, [`TreeError::TooBig`] for a record no node
/// holds.
pub fn build(
    records: &[(Vec<u8>, Vec<u8>)],
    kind: TreeKind,
    alloc: &mut Alloc,
    next_oid: &mut u64,
) -> Result<Built, TreeError> {
    let longest_key = records.iter().map(|(k, _)| k.len()).max().unwrap_or(0) as u32;
    let longest_val = records.iter().map(|(_, v)| v.len()).max().unwrap_or(0) as u32;

    // Lay out each level: the nodes' entry ranges, identifiers and blocks.
    struct Node {
        id: u64,
        paddr: u64,
        level: u16,
        entries: Vec<(Vec<u8>, Vec<u8>)>,
    }
    let mut nodes: Vec<Node> = Vec::new();
    let mut level_entries: Vec<(Vec<u8>, Vec<u8>)> = records.to_vec();
    let mut level = 0u16;
    let place = |alloc: &mut Alloc, next_oid: &mut u64| -> Result<(u64, u64), TreeError> {
        let paddr = alloc.take(1).ok_or(TreeError::NoSpace)?;
        let id = if kind.virtual_nodes {
            let oid = *next_oid;
            *next_oid += 1;
            oid
        } else {
            paddr
        };
        Ok((id, paddr))
    };
    loop {
        if fits_root(&level_entries, kind.layout, level) {
            let (id, paddr) = place(alloc, next_oid)?;
            nodes.push(Node {
                id,
                paddr,
                level,
                entries: level_entries,
            });
            break;
        }
        let mut above = Vec::new();
        for r in partition(&level_entries, kind.layout, level)? {
            let (id, paddr) = place(alloc, next_oid)?;
            above.push((level_entries[r.start].0.clone(), id.to_le_bytes().to_vec()));
            nodes.push(Node {
                id,
                paddr,
                level,
                entries: level_entries[r].to_vec(),
            });
        }
        level_entries = above;
        level += 1;
    }

    let node_count = nodes.len() as u64;
    let root_index = nodes.len() - 1;
    let storage = if kind.virtual_nodes { 0 } else { obj::PHYSICAL };
    let mut built = Built {
        root: nodes[root_index].id,
        nodes: Vec::with_capacity(nodes.len()),
        mapped: Vec::new(),
    };
    for (i, n) in nodes.iter().enumerate() {
        let is_root = i == root_index;
        let bytes = NodeSpec {
            obj_type: storage | if is_root { obj::BTREE } else { obj::BTREE_NODE },
            subtype: kind.subtype,
            oid: n.id,
            xid: XID,
            level: n.level,
            is_root,
            headerless: false,
            hashed: false,
            layout: kind.layout,
            tree_flags: kind.tree_flags,
            longest_key,
            longest_val,
            key_count: records.len() as u64,
            node_count,
            entries: &n.entries,
        }
        .build();
        built.nodes.push((n.paddr, bytes));
        if kind.virtual_nodes {
            built.mapped.push((n.id, n.paddr));
        }
    }
    Ok(built)
}
