//! The guest GPU page table, and the walk that resolves a GVA through it.
//!
//! # Provenance
//!
//! Unlike [`crate::ops`], this layout is not derived by perturbing a serializer,
//! and it cannot be: no serializer record carries a page table, so there is no
//! fixture that could pin one. It comes from the device contract, checked field
//! by field, and it is what the device's GVA resolution runs on — a wrong
//! constant anywhere in it would fail to resolve anything at all.
//!
//! The format:
//!
//! - A node holds an array of four-byte entries indexed directly by the index,
//!   so an entry is [`PTE_SIZE`] bytes.
//! - Bit 31 is a flag, [`PTE_FLAG_MASK`]. The frame number is the remaining
//!   [`PTE_PFN_MASK`], used raw with no further shift.
//! - **Zero is the sole not-present encoding**: an absent entry reads zero, and
//!   a removed one is cleared back to zero.
//! - A frame number never has bit 31 set, and the flag is only ever written
//!   together with a frame number — the guest builds an entry by OR-ing the two,
//!   never the flag alone.
//!
//! Those last two points are why [`WalkError::MalformedPte`] is a distinct error
//! rather than a pedantic split of [`WalkError::NotPresent`]. Together they say
//! the only entry with a zero frame-number field and a nonzero word is
//! [`PTE_FLAG_MASK`] exactly, which is a value the guest has no way to write.
//! Reading one means the page holding the table was corrupted, and collapsing
//! the two arms would discard that signal to save a branch.
//!
//! The argument deliberately does not rest on "physical page zero is never
//! mapped". That is true of the platform rather than of this format, nothing in
//! the guest's own construction enforces it, and it is not needed: the flag
//! never travelling alone is what closes the case.
//!
//! The fan-out is 1024 entries per node and byte lengths convert to pages with a
//! 12-bit shift. Both are the x86 pathway's values and both agree with
//! [`X86_64`] below. See [`Geometry::index_bits`] for why those two numbers are
//! not independent, and so why deriving one pathway's does not leave the other
//! one guessed.
//!
//! # Relationship to paging resolution and the device refusal adapter
//!
//! That module walks the same tree and reached the same constants
//! independently, which is why the agreement is worth something — including on
//! the subtle part, the two-arm split on a zero PFN.
//!
//! What stays outside this byte view is the part that is not byte interpretation:
//! task lookup, allocation-requiring span resolution, and the device's typed
//! refusal channel. This module owns the tree and nothing else.

use crate::mem::GuestMemory;

/// Byte width of one page-table entry.
///
/// The node's entry array is indexed with a four-byte scale.
pub const PTE_SIZE: u32 = 4;

/// Bit 31, a flag carried alongside the frame number.
///
/// This module never interprets it; it is preserved in [`Walk::raw_pte`] so a
/// caller that learns its meaning does not have to re-read the entry.
pub const PTE_FLAG_MASK: u32 = 0x8000_0000;

/// Bits `[30:0]`, the page frame number, used raw and unshifted.
pub const PTE_PFN_MASK: u32 = 0x7fff_ffff;

/// Offset of the root page number within a task's directory page.
pub const DIRECTORY_ROOT_PFN: u64 = 0x00;

/// Offset of the tree depth within a task's directory page.
///
/// Read per task rather than assumed. The x86 guest has only ever been observed
/// to say 3, but the field exists and a hardcoded depth would be a guess.
pub const DIRECTORY_DEPTH: u64 = 0x04;

/// Upper bound on the tree depth read from a task's directory page.
///
/// Not the depth itself — that is [`DIRECTORY_DEPTH`], read per task. This is
/// the bound a corrupt or hostile directory word is refused against, and it is
/// derived from the address space rather than picked: a walk of `d` levels
/// resolves `d * index_bits + page_shift` bits of guest virtual address, and
/// `index_bits` is `page_shift - 2` because a node is one page of four-byte
/// entries ([`Geometry::index_bits`]).
///
/// | geometry | index bits | depth 3 | depth 4 | depth 5 |
/// |---|---|---|---|---|
/// | x86_64, `page_shift` 12 | 10 | 42 bits | **52 bits** | 62 bits |
/// | arm64e, `page_shift` 14 | 12 | 50 bits | **62 bits** | 74 bits |
///
/// Four is the first depth that covers a 48-bit virtual address on **both**
/// geometries, which is the whole address space either guest can form: macOS
/// user VAs are 47-bit and the wire carries a `u64` that no guest fills. Depth 3
/// does not reach it on x86 (42 bits), so the bound cannot be lowered to the
/// only depth either guest has been observed to declare; depth 5 could not
/// describe an address the guest can construct, so nothing is lost by refusing
/// it.
///
/// Observed: a driven x86 boot reads `depth=3` on every task directory it walks
/// (32 of 32), and `WalkError::DepthTooDeep` has never been seen. The headroom
/// is one level, and it is the level that separates "enough for the address
/// space" from "enough for the guest we have measured".
pub const MAX_DEPTH: u32 = 4;

// The derivation above, as a build gate. Depth `MAX_DEPTH` must reach a 48-bit
// address on both pathways, and `MAX_DEPTH - 1` must not — the first half is the
// property the bound exists for, and the second is what keeps it the *smallest*
// such depth rather than a number that merely happens to be large enough.
const _: () = {
    const VA_BITS: u32 = 48;
    const fn reach(page_shift: u32, depth: u32) -> u32 {
        depth * (page_shift - 2) + page_shift
    }
    assert!(reach(12, MAX_DEPTH) >= VA_BITS);
    assert!(reach(14, MAX_DEPTH) >= VA_BITS);
    assert!(reach(12, MAX_DEPTH - 1) < VA_BITS);
};

/// Page-table shape for one guest pathway.
///
/// **One number is stored.** Everything else about the shape — entries per
/// table, index mask, page size — is derived, because they are not independent:
/// a node is exactly one page of four-byte entries, so the fan-out is fixed by
/// the page size. Storing any of them separately invites a struct whose fields
/// disagree, which is what [`Geometry::validate`] would then have to catch.
///
/// [`MAX_DEPTH`] used to be a field here, and it is the worked example of that
/// hazard: both pathway constants set it to `MAX_DEPTH`, nothing ever
/// constructed a `Geometry` with anything else, and half of `validate` existed
/// to catch a disagreement only a hand-built struct could create. The depth
/// bound is a property of the address space, not of a page size — its
/// derivation is the `const` assertion at its own declaration — so it is read
/// from there directly and there is no second copy to keep honest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    /// Guest page shift: 12 on x86_64, 14 on arm64e. Never defaulted.
    pub page_shift: u32,
}

/// x86_64 macOS guest: 4 KiB pages, 1024 entries per node, ten index bits.
pub const X86_64: Geometry = Geometry { page_shift: 12 };

/// arm64e macOS guest: 16 KiB pages, 4096 entries per node, twelve index bits.
pub const ARM64E: Geometry = Geometry { page_shift: 14 };

impl Geometry {
    /// Bytes per guest page.
    #[inline]
    pub const fn page_size(self) -> u64 {
        1u64 << self.page_shift
    }

    /// Mask selecting the byte offset within a page.
    #[inline]
    pub const fn page_offset_mask(self) -> u64 {
        self.page_size() - 1
    }

    /// Index bits per level.
    ///
    /// A node is one page of four-byte entries, so it holds
    /// `page_size / 4 == 2^(page_shift - 2)` of them. The `- 2` is
    /// `log2(PTE_SIZE)` and is the whole reason this is derived rather than
    /// stored: x86's ten bits and arm64e's twelve are both this expression, and
    /// the walk masks each index to this width.
    #[inline]
    pub const fn index_bits(self) -> u32 {
        self.page_shift - 2
    }

    /// Entries in one node.
    #[inline]
    pub const fn entries_per_table(self) -> u64 {
        1u64 << self.index_bits()
    }

    /// Mask selecting one level's index out of a page index.
    #[inline]
    pub const fn index_mask(self) -> u64 {
        self.entries_per_table() - 1
    }

    /// Guest page frame number to guest address.
    #[inline]
    pub const fn pfn_to_addr(self, pfn: u32) -> u64 {
        (pfn as u64) << self.page_shift
    }

    /// Reject a shape this walk cannot execute.
    ///
    /// The page shift is checked against the two pathways rather than against a
    /// range, because a third value would not be an untested configuration — it
    /// would mean the geometry was inferred from something other than the
    /// pathway, and every constant derived from it would be suspect.
    pub const fn validate(self) -> Result<(), WalkError> {
        if self.page_shift != 12 && self.page_shift != 14 {
            return Err(WalkError::UnsupportedGeometry);
        }
        Ok(())
    }
}

/// Why a walk stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WalkError {
    /// The [`Geometry`] is not one this walk can execute.
    UnsupportedGeometry,
    /// The task named no root page.
    ZeroRootPfn,
    /// The task's directory reported depth zero.
    ZeroDepth,
    /// The task's directory reported a depth past [`MAX_DEPTH`].
    DepthTooDeep,
    /// A page holding part of the tree could not be read.
    TableRead,
    /// The entry was zero: the guest has not mapped this address.
    ///
    /// Expected control flow, not a device defect — see the module docs.
    NotPresent,
    /// The entry was nonzero but named PFN zero.
    ///
    /// A working guest cannot produce this: a frame number never carries bit 31,
    /// and physical page zero is never mapped. Reading one means the table's own
    /// page is corrupt.
    MalformedPte,
}

/// A failed walk, with the position it failed at.
///
/// The position is carried because the device reports it: which level, which
/// entry, and the raw word read there is the difference between "this address
/// is not mapped" and a diagnosable corruption.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WalkFailure {
    pub error: WalkError,
    /// Level the walk stopped at, zero-based from the root.
    pub level: u32,
    /// Index within that level's node.
    pub entry_index: u32,
    /// The entry as read, before masking. Zero if the read itself failed.
    pub raw_pte: u32,
}

/// A resolved address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Walk {
    /// Page frame the walk arrived at.
    pub leaf_pfn: u32,
    /// Address of that page.
    pub addr_page: u64,
    /// Address of the requested byte within it.
    pub addr: u64,
    /// Page index of the input address.
    pub page_index: u64,
    /// The leaf entry as read, before masking, so a caller can read
    /// [`PTE_FLAG_MASK`] without walking again.
    pub raw_pte: u32,
}

/// Read a task's root page number and tree depth from its directory page.
///
/// Both come off the directory rather than from constants: the depth is a field
/// the guest writes, and treating it as known is how a device ends up walking
/// the wrong number of levels when a guest changes.
pub fn read_directory<M: GuestMemory>(
    mem: &M,
    geometry: Geometry,
    directory_pfn: u32,
) -> Result<(u32, u32), WalkError> {
    geometry.validate()?;
    if directory_pfn == 0 {
        return Err(WalkError::ZeroRootPfn);
    }
    let base = geometry.pfn_to_addr(directory_pfn);
    let root_pfn = mem
        .u32_at(base + DIRECTORY_ROOT_PFN)
        .ok_or(WalkError::TableRead)?;
    let depth = mem
        .u32_at(base + DIRECTORY_DEPTH)
        .ok_or(WalkError::TableRead)?;
    Ok((root_pfn, depth))
}

/// The node pages one descent reads, root first.
///
/// Every element is an **interior** page of the tree — a page whose contents are
/// page-table entries — and the leaf the walk resolves to is deliberately not
/// among them. That is the distinction the whole type exists to make: a caller
/// asking "is this guest page part of a page table?" must not be told yes about
/// the data page the table points at.
///
/// The capacity is [`MAX_DEPTH`] and the push is bounded by it, so a path can
/// never name more nodes than a walk can descend. `depth` is validated against
/// the same constant before the descent starts, which is what makes the bound
/// unreachable rather than merely checked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct NodePath {
    pfns: [u32; MAX_DEPTH as usize],
    len: u32,
}

impl NodePath {
    /// The nodes descended through, root first, in descent order.
    ///
    /// Short on a failed walk: a walk that refused at level 2 has read levels 0
    /// and 1, and those two are real nodes whether or not the address resolved.
    pub fn pfns(&self) -> &[u32] {
        &self.pfns[..self.len as usize]
    }

    /// Record one node. Silently keeps the first [`MAX_DEPTH`] and no more.
    fn push(&mut self, pfn: u32) {
        let at = self.len as usize;
        if at < self.pfns.len() {
            self.pfns[at] = pfn;
            self.len += 1;
        }
    }
}

/// Walk `gva` from `root_pfn` down `depth` levels.
///
/// Every level reads one entry and descends into the page frame it names; the
/// frame the last level names is the leaf. `depth` is validated to be at least
/// one, so the loop always runs and the returned `leaf_pfn` always came from an
/// entry rather than from `root_pfn`.
pub fn walk<M: GuestMemory>(
    mem: &M,
    geometry: Geometry,
    root_pfn: u32,
    depth: u32,
    gva: u64,
) -> Result<Walk, WalkFailure> {
    walk_recording_nodes(
        mem,
        geometry,
        root_pfn,
        depth,
        gva,
        &mut NodePath::default(),
    )
}

/// [`walk`], additionally reporting the interior nodes the descent read.
///
/// One descent serves both, rather than a second walker that would have to be
/// held to this one: the caller that wants the node set wants it *for the tree
/// this walk just read*, and a re-read could see a different tree.
///
/// `nodes` is filled as the walk descends, so a refusal still reports the levels
/// that were read before it. It is not cleared first — a caller reusing a path
/// across walks is accumulating deliberately.
pub fn walk_recording_nodes<M: GuestMemory>(
    mem: &M,
    geometry: Geometry,
    root_pfn: u32,
    depth: u32,
    gva: u64,
    nodes: &mut NodePath,
) -> Result<Walk, WalkFailure> {
    let fail = |error| WalkFailure {
        error,
        level: 0,
        entry_index: 0,
        raw_pte: 0,
    };
    geometry.validate().map_err(fail)?;
    if root_pfn == 0 {
        return Err(fail(WalkError::ZeroRootPfn));
    }
    if depth == 0 {
        return Err(fail(WalkError::ZeroDepth));
    }
    if depth > MAX_DEPTH {
        return Err(fail(WalkError::DepthTooDeep));
    }

    let page_index = gva >> geometry.page_shift;
    let page_off = gva & geometry.page_offset_mask();
    let mut current_pfn = root_pfn;
    let mut raw_pte = 0;

    for level in 0..depth {
        // `current_pfn` is the page this level reads its entry *out of*, so it
        // is an interior node by construction and the leaf never reaches here —
        // the loop ends before the frame the last entry names is recorded.
        nodes.push(current_pfn);
        // The root indexes by the most significant slice of the page index, so
        // the shift shrinks as the walk descends.
        let shift = (depth - 1 - level) * geometry.index_bits();
        let entry_index = ((page_index >> shift) & geometry.index_mask()) as u32;
        let entry_addr = geometry.pfn_to_addr(current_pfn) + (entry_index as u64) * PTE_SIZE as u64;

        let at = |error, raw_pte| WalkFailure {
            error,
            level,
            entry_index,
            raw_pte,
        };
        let pte = mem.u32_at(entry_addr).ok_or(at(WalkError::TableRead, 0))?;
        raw_pte = pte;

        let next_pfn = pte & PTE_PFN_MASK;
        if next_pfn == 0 {
            // An absent entry is written as zero, and a frame number never
            // carries bit 31, so these two cases have different causes.
            return Err(at(
                if pte == 0 {
                    WalkError::NotPresent
                } else {
                    WalkError::MalformedPte
                },
                pte,
            ));
        }
        current_pfn = next_pfn;
    }

    let addr_page = geometry.pfn_to_addr(current_pfn);
    Ok(Walk {
        leaf_pfn: current_pfn,
        addr_page,
        addr: addr_page + page_off,
        page_index,
        raw_pte,
    })
}

/// Deepest-level entries fetched per guest read by [`walk_run`].
///
/// The buffer is a stack array, so this is the only thing bounding it: 64
/// entries is 256 bytes, and it turns the one-read-per-page the level-reuse
/// leaves behind into one read per 64 pages. Raising it costs stack in every
/// frame that walks a run and buys a shrinking fraction of the reads that are
/// left; lowering it gives the per-page cost back.
///
/// It is not a limit on anything the guest can express — a run longer than a
/// batch simply takes several, and a run shorter than one reads only the words
/// its node has left. Nothing observable changes with this number, which is
/// why the equivalence test against [`walk`] is what holds it.
const LEAF_BATCH: usize = 64;

/// Walk a run of consecutive pages, re-reading only the levels whose entry
/// index changed.
///
/// A run of `pages` pages starting at `first_gva` shares every level of the tree
/// except the deepest for `1 << index_bits` pages at a time, so walking each one
/// with [`walk`] re-reads the same upper entries `depth - 1` times per page. On
/// a guest with a four-level tree that is four guest-memory reads per page where
/// one is needed, and the caller that motivates this — a licence check over a
/// 1080p surface's page list — pays it 2 025 times a flush.
///
/// The visitor is called once per page in ascending order with the page's index
/// within the run, and stops the walk by answering `false`. A failure is
/// reported for the page it happened on and does not stop the run: a caller
/// checking a cached list against the live table needs to know *which* pages
/// disagree, and a walk that stopped at the first would report a shorter
/// disagreement than there is.
///
/// # What the reuse assumes
///
/// That the tree does not change under the walk. It is the same assumption
/// [`walk`] makes within one descent, extended to the run — a guest that
/// rewrites an upper entry midway through is a guest editing a page table this
/// device is reading, and neither form of walk can be atomic against that.
/// A caller needing a coherent snapshot needs one from the hypervisor, not from
/// a re-read here.
///
/// The deepest level is read [`LEAF_BATCH`] entries at a time, which widens
/// that same assumption from the levels above a page to the `LEAF_BATCH` pages
/// either side of it. It does not introduce it.
pub fn walk_run<M: GuestMemory>(
    mem: &M,
    geometry: Geometry,
    root_pfn: u32,
    depth: u32,
    first_gva: u64,
    pages: u64,
    visit: &mut dyn FnMut(u64, Result<Walk, WalkFailure>) -> bool,
) {
    let fail = |error| WalkFailure {
        error,
        level: 0,
        entry_index: 0,
        raw_pte: 0,
    };
    if let Err(f) = geometry.validate().map_err(fail) {
        visit(0, Err(f));
        return;
    }
    if root_pfn == 0 {
        visit(0, Err(fail(WalkError::ZeroRootPfn)));
        return;
    }
    if depth == 0 {
        visit(0, Err(fail(WalkError::ZeroDepth)));
        return;
    }
    if depth > MAX_DEPTH {
        visit(0, Err(fail(WalkError::DepthTooDeep)));
        return;
    }

    // The entry index taken at each level of the previous page's descent, and
    // the frame that entry named. `held` is how many *leading* levels of that
    // record are still true: a level whose index differs invalidates itself and
    // everything under it, which is why one prefix length is enough and a
    // per-level valid bit is not.
    let mut seen_index = [0u32; MAX_DEPTH as usize];
    let mut seen_next = [0u32; MAX_DEPTH as usize];
    let mut held = 0usize;

    // The deepest level's entry index advances by one per page, so consecutive
    // pages read consecutive words of one node. Those are fetched a batch at a
    // time: the upper levels are already elided by `held`, which leaves one
    // guest read per page, and a batch turns 64 of them into one.
    //
    // `leaf_node` is the node the buffer holds words from — never zero for a
    // live batch, because PFN zero is not a page the format can name, so a zero
    // here means empty. A batch never crosses a node: it is clamped to the
    // words left in this one, which is also what keeps `leaf_first + leaf_len`
    // inside the node's own page.
    let mut leaf_buf = [0u8; LEAF_BATCH * PTE_SIZE as usize];
    let mut leaf_node = 0u32;
    let mut leaf_first = 0u32;
    let mut leaf_len = 0usize;
    // A node whose batch read failed. One unreadable byte fails a whole span,
    // so a batch cannot say *which* word was bad; without this the walk would
    // retry the failing batch for every page of the node before falling back.
    let mut leaf_unbatchable = 0u32;

    let first_page = first_gva >> geometry.page_shift;
    for i in 0..pages {
        let page_index = first_page + i;
        let gva = (page_index << geometry.page_shift) | (first_gva & geometry.page_offset_mask());
        let page_off = gva & geometry.page_offset_mask();
        let mut current_pfn = root_pfn;
        let mut raw_pte = 0u32;
        let mut failure = None;
        for level in 0..depth {
            let shift = (depth - 1 - level) * geometry.index_bits();
            let entry_index = ((page_index >> shift) & geometry.index_mask()) as u32;
            let lv = level as usize;
            if lv < held && seen_index[lv] == entry_index {
                current_pfn = seen_next[lv];
                continue;
            }
            let entry_addr =
                geometry.pfn_to_addr(current_pfn) + (entry_index as u64) * PTE_SIZE as u64;
            let at = |error, raw_pte| WalkFailure {
                error,
                level,
                entry_index,
                raw_pte,
            };
            let read = if level + 1 == depth {
                // Refill when this word is not one the buffer already holds.
                if leaf_node != current_pfn
                    || entry_index < leaf_first
                    || (entry_index - leaf_first) as usize >= leaf_len
                {
                    leaf_len = 0;
                    if leaf_unbatchable != current_pfn {
                        let left = geometry.entries_per_table() - entry_index as u64;
                        let want = (left as usize).min(LEAF_BATCH);
                        if mem.read_at(entry_addr, &mut leaf_buf[..want * PTE_SIZE as usize]) {
                            leaf_node = current_pfn;
                            leaf_first = entry_index;
                            leaf_len = want;
                        } else {
                            leaf_unbatchable = current_pfn;
                        }
                    }
                }
                if leaf_len == 0 {
                    // Batching is off for this node, so read the one word and
                    // let its own failure be attributed to its own page.
                    mem.u32_at(entry_addr)
                } else {
                    let off = (entry_index - leaf_first) as usize * PTE_SIZE as usize;
                    Some(u32::from_le_bytes([
                        leaf_buf[off],
                        leaf_buf[off + 1],
                        leaf_buf[off + 2],
                        leaf_buf[off + 3],
                    ]))
                }
            } else {
                mem.u32_at(entry_addr)
            };
            let Some(pte) = read else {
                failure = Some(at(WalkError::TableRead, 0));
                break;
            };
            raw_pte = pte;
            let next_pfn = pte & PTE_PFN_MASK;
            if next_pfn == 0 {
                failure = Some(at(
                    if pte == 0 {
                        WalkError::NotPresent
                    } else {
                        WalkError::MalformedPte
                    },
                    pte,
                ));
                break;
            }
            seen_index[lv] = entry_index;
            seen_next[lv] = next_pfn;
            held = lv + 1;
            current_pfn = next_pfn;
        }
        let result = match failure {
            // A failed descent leaves the record describing a tree the walk did
            // not finish reading, so the next page starts from the root.
            Some(f) => {
                held = 0;
                Err(f)
            }
            None => {
                let addr_page = geometry.pfn_to_addr(current_pfn);
                Ok(Walk {
                    leaf_pfn: current_pfn,
                    addr_page,
                    addr: addr_page + page_off,
                    page_index,
                    raw_pte,
                })
            }
        };
        if !visit(i, result) {
            return;
        }
    }
}

/// Builds page tables the way the guest does.
///
/// This exists so tests walk a tree assembled by the format's own rules rather
/// than one hand-written to satisfy the walker — the two agree only if the
/// walker is right. [`Builder::set_entry`] enforces both of the format's write
/// guards, so a test that tries to build a malformed entry fails at the build
/// rather than producing a tree the walker then quietly accepts.
///
/// It carves pages out of a caller-provided buffer rather than allocating, so
/// the crate's no-allocation invariant holds in tests as well as in the decode
/// path. `reims-vgpu`'s tests use it too, which is why it is not `#[cfg(test)]`.
pub struct Builder<'a> {
    geometry: Geometry,
    pages: &'a mut [u8],
    next_pfn: u32,
}

impl<'a> Builder<'a> {
    /// Take a buffer as the guest-physical image.
    ///
    /// Page zero is reserved immediately, so PFN 0 always means "no page" — the
    /// same reservation that makes zero a usable not-present encoding.
    ///
    /// Panics if the buffer is not a whole number of pages, which would let a
    /// frame at the tail be silently short.
    pub fn new(geometry: Geometry, pages: &'a mut [u8]) -> Self {
        let page = geometry.page_size() as usize;
        assert!(
            pages.len() >= page && pages.len().is_multiple_of(page),
            "the image must be a nonzero whole number of {page}-byte pages"
        );
        pages.fill(0);
        Self {
            geometry,
            pages,
            next_pfn: 1,
        }
    }

    /// Number of pages an image must hold for `frames` frames plus the reserved
    /// page zero. Use it to size the buffer passed to [`Builder::new`].
    pub const fn image_len(geometry: Geometry, frames: usize) -> usize {
        (frames + 1) * geometry.page_size() as usize
    }

    /// Claim the next zeroed page and return its frame number.
    pub fn alloc_page(&mut self) -> u32 {
        let pfn = self.next_pfn;
        let end = (pfn as usize + 1) * self.geometry.page_size() as usize;
        assert!(end <= self.pages.len(), "image too small for another frame");
        self.next_pfn += 1;
        pfn
    }

    fn slot(&mut self, node_pfn: u32, index: u32) -> &mut [u8] {
        let at =
            (self.geometry.pfn_to_addr(node_pfn) as usize) + (index as usize) * PTE_SIZE as usize;
        &mut self.pages[at..at + PTE_SIZE as usize]
    }

    /// Write one entry, enforcing the format's two write guards.
    ///
    /// Panics if the slot is already occupied or if `pfn` has bit 31 set —
    /// exactly the two entries a guest cannot produce.
    pub fn set_entry(&mut self, node_pfn: u32, index: u32, pfn: u32, flag: bool) {
        assert_eq!(pfn & PTE_FLAG_MASK, 0, "a PFN never has bit 31 already set");
        let entry = pfn | if flag { PTE_FLAG_MASK } else { 0 };
        let slot = self.slot(node_pfn, index);
        assert_eq!(
            u32::from_le_bytes(slot.try_into().unwrap()),
            0,
            "an entry is only written into an empty slot"
        );
        slot.copy_from_slice(&entry.to_le_bytes());
    }

    /// Write a raw word, bypassing the guards, to synthesize corruption.
    ///
    /// The guest cannot produce these; the point is to prove the walker reports
    /// them rather than descending into them.
    pub fn poke_entry(&mut self, node_pfn: u32, index: u32, raw: u32) {
        self.slot(node_pfn, index)
            .copy_from_slice(&raw.to_le_bytes());
    }

    /// Read an entry's frame number back, for node reuse.
    fn child_of(&mut self, node_pfn: u32, index: u32) -> u32 {
        u32::from_le_bytes(self.slot(node_pfn, index).try_into().unwrap()) & PTE_PFN_MASK
    }

    /// Build a `depth`-level tree mapping `page_index` to `leaf_pfn`.
    ///
    /// Returns the root frame number.
    pub fn map(&mut self, depth: u32, page_index: u64, leaf_pfn: u32) -> u32 {
        let root = self.alloc_page();
        self.map_into(root, depth, page_index, leaf_pfn);
        root
    }

    /// Add a mapping to an existing tree, reusing nodes already present.
    pub fn map_into(&mut self, root: u32, depth: u32, page_index: u64, leaf_pfn: u32) {
        let mut node = root;
        for level in 0..depth {
            let shift = (depth - 1 - level) * self.geometry.index_bits();
            let index = ((page_index >> shift) & self.geometry.index_mask()) as u32;
            if level == depth - 1 {
                self.set_entry(node, index, leaf_pfn, false);
                return;
            }
            node = match self.child_of(node, index) {
                0 => {
                    let child = self.alloc_page();
                    self.set_entry(node, index, child, false);
                    child
                }
                existing => existing,
            };
        }
    }

    /// The assembled guest-physical image.
    pub fn bytes(&self) -> &[u8] {
        self.pages
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mem::SliceMemory;

    #[test]
    fn the_derived_shape_matches_the_contracts_constants() {
        // The width each index is masked to, and four-byte entries filling
        // exactly one page.
        assert_eq!(X86_64.index_bits(), 10);
        assert_eq!(X86_64.index_mask(), 0x3ff);
        assert_eq!(X86_64.entries_per_table(), 1024);
        assert_eq!(
            X86_64.entries_per_table() * PTE_SIZE as u64,
            X86_64.page_size()
        );

        assert_eq!(ARM64E.index_bits(), 12);
        assert_eq!(ARM64E.index_mask(), 0xfff);
        assert_eq!(ARM64E.entries_per_table(), 4096);
        assert_eq!(
            ARM64E.entries_per_table() * PTE_SIZE as u64,
            ARM64E.page_size()
        );

        assert_eq!(PTE_PFN_MASK | PTE_FLAG_MASK, u32::MAX);
        assert_eq!(PTE_PFN_MASK & PTE_FLAG_MASK, 0);
    }

    #[test]
    fn a_geometry_off_either_pathway_is_refused_rather_than_walked() {
        for shift in [0, 11, 13, 15, 16] {
            let g = Geometry { page_shift: shift };
            assert_eq!(g.validate(), Err(WalkError::UnsupportedGeometry));
        }
        assert_eq!(X86_64.validate(), Ok(()));
        assert_eq!(ARM64E.validate(), Ok(()));
    }

    /// Enough frames for a full-depth tree at either page size. Sized for the
    /// larger page so one buffer type serves both geometries; it stays a whole
    /// number of pages at 4 KiB too, which [`Builder::new`] requires.
    const IMAGE: usize = Builder::image_len(ARM64E, 8);

    #[test]
    fn a_walk_over_a_validly_built_tree_finds_the_leaf() {
        for geometry in [X86_64, ARM64E] {
            for depth in 1..=MAX_DEPTH {
                let mut buf = [0u8; IMAGE];
                let mut b = Builder::new(geometry, &mut buf);
                let leaf = 0x2bc;
                // A page index with a distinct value in every level's slice, so
                // a walk that mixes two levels up cannot land by accident.
                let page_index = (0..depth)
                    .map(|l| ((l as u64) + 1) << (l * geometry.index_bits()))
                    .fold(0, |a, b| a | b);
                let root = b.map(depth, page_index, leaf);
                let mem = SliceMemory::new(b.bytes());

                let off = 0x21;
                let gva = (page_index << geometry.page_shift) | off;
                let w = walk(&mem, geometry, root, depth, gva).expect("mapped");
                assert_eq!(w.leaf_pfn, leaf);
                assert_eq!(w.page_index, page_index);
                assert_eq!(w.addr, geometry.pfn_to_addr(leaf) + off);
            }
        }
    }

    /// The reported node path is exactly the interior pages the descent read,
    /// root first, and the leaf is never one of them.
    ///
    /// The leaf half is the load-bearing one. A caller uses this set to decide
    /// that a guest page holds page-table entries and must not be written; if
    /// the leaf leaked into it, every data page the tree maps would be declared
    /// part of the table, which is the direction that reads as a thorough guard
    /// and refuses the device's ordinary work.
    #[test]
    fn the_nodes_a_walk_reports_are_its_interiors_and_never_its_leaf() {
        for geometry in [X86_64, ARM64E] {
            for depth in 1..=MAX_DEPTH {
                let mut buf = [0u8; IMAGE];
                let mut b = Builder::new(geometry, &mut buf);
                let leaf = 0x2bc;
                let page_index = (0..depth)
                    .map(|l| ((l as u64) + 1) << (l * geometry.index_bits()))
                    .fold(0, |a, b| a | b);
                let root = b.map(depth, page_index, leaf);

                // The same chain, walked with the builder's own reader, so the
                // expectation is derived from the tree rather than restated.
                let mut expected = [0u32; MAX_DEPTH as usize];
                let mut node = root;
                for level in 0..depth {
                    expected[level as usize] = node;
                    let shift = (depth - 1 - level) * geometry.index_bits();
                    let index = ((page_index >> shift) & geometry.index_mask()) as u32;
                    node = b.child_of(node, index);
                }
                assert_eq!(node, leaf, "the chain must end at the leaf");

                let mem = SliceMemory::new(b.bytes());
                let gva = page_index << geometry.page_shift;
                let mut nodes = NodePath::default();
                let w = walk_recording_nodes(&mem, geometry, root, depth, gva, &mut nodes)
                    .expect("mapped");

                assert_eq!(nodes.pfns(), &expected[..depth as usize]);
                assert_eq!(nodes.pfns()[0], root, "the root is always the first node");
                assert!(
                    !nodes.pfns().contains(&w.leaf_pfn),
                    "the leaf is not an interior node: {:?} contains {}",
                    nodes.pfns(),
                    w.leaf_pfn
                );
                // And the plain walk agrees about everything else.
                assert_eq!(walk(&mem, geometry, root, depth, gva), Ok(w));
            }
        }
    }

    /// A walk that refuses still reports the nodes it read before refusing.
    ///
    /// Those levels were really read, and a caller collecting the tree's pages
    /// wants them: an address that does not resolve is the *common* case while a
    /// guest is tearing a task down, which is exactly when the question is asked.
    #[test]
    fn a_refused_walk_still_reports_the_nodes_it_read() {
        let geometry = X86_64;
        let depth = 3;
        let bits = geometry.index_bits();
        let mut buf = [0u8; IMAGE];
        let mut b = Builder::new(geometry, &mut buf);
        let mapped = (1u64 << (2 * bits)) | (1u64 << bits) | 1;
        let root = b.map(depth, mapped, 0x2bc);
        let level1 = b.child_of(root, 1);
        let mem = SliceMemory::new(b.bytes());

        // Differs in the root's own slice: nothing below the root was read.
        let elsewhere = (2u64 << (2 * bits)) | (1u64 << bits) | 1;
        let mut nodes = NodePath::default();
        let r = walk_recording_nodes(
            &mem,
            geometry,
            root,
            depth,
            elsewhere << geometry.page_shift,
            &mut nodes,
        );
        assert_eq!(r.unwrap_err().level, 0);
        assert_eq!(nodes.pfns(), &[root]);

        // Shares the root's slice and differs below it: two nodes were read.
        let deeper = (1u64 << (2 * bits)) | (2u64 << bits) | 1;
        let mut nodes = NodePath::default();
        let r = walk_recording_nodes(
            &mem,
            geometry,
            root,
            depth,
            deeper << geometry.page_shift,
            &mut nodes,
        );
        assert_eq!(r.unwrap_err().level, 1);
        assert_eq!(nodes.pfns(), &[root, level1]);
    }

    /// A path cannot name more nodes than a walk can descend, however many times
    /// it is pushed — the bound is the array's own length and not a check a new
    /// caller could forget.
    #[test]
    fn a_node_path_never_grows_past_the_deepest_walk() {
        let mut path = NodePath::default();
        for pfn in 1..=(MAX_DEPTH * 3) {
            path.push(pfn);
        }
        assert_eq!(path.pfns().len(), MAX_DEPTH as usize);
        assert_eq!(path.pfns(), &[1, 2, 3, 4]);
    }

    /// `walk_run` and `walk` must answer identically for every page of a run,
    /// including the pages that do not resolve.
    ///
    /// This is the whole of `walk_run`'s contract. It exists only to avoid
    /// re-reading upper levels, so the moment it answers differently from the
    /// walk it optimises it is not an optimisation but a second, weaker walker —
    /// and the way it would fail is by carrying a stale upper level across an
    /// index boundary, which the run below crosses deliberately.
    #[test]
    fn a_run_walk_agrees_with_the_single_walk_on_every_page() {
        for geometry in [X86_64, ARM64E] {
            for depth in 1..=MAX_DEPTH {
                let mut buf = [0u8; IMAGE];
                let mut b = Builder::new(geometry, &mut buf);
                // Two pages whose indices differ above the deepest level, so the
                // run has to notice the upper entry changed, plus their
                // neighbours, which must reuse it.
                let stride = 1u64 << geometry.index_bits();
                let root = b.map(depth, 0, 0x11);
                b.map_into(root, depth, 1, 0x12);
                if depth > 1 {
                    b.map_into(root, depth, stride, 0x21);
                    b.map_into(root, depth, stride + 1, 0x22);
                }
                let mem = SliceMemory::new(b.bytes());

                // Covers both mapped clusters, the hole between them, and the
                // unmapped tail past the second.
                let pages = if depth > 1 { stride + 3 } else { 4 };
                let mut seen = 0u64;
                walk_run(&mem, geometry, root, depth, 0, pages, &mut |i, got| {
                    let gva = i << geometry.page_shift;
                    let want = walk(&mem, geometry, root, depth, gva);
                    match (&got, &want) {
                        (Ok(a), Ok(e)) => assert_eq!(a, e, "page {i} depth {depth}"),
                        (Err(a), Err(e)) => assert_eq!(a, e, "page {i} depth {depth}"),
                        _ => panic!("page {i} depth {depth}: {got:?} vs {want:?}"),
                    }
                    seen += 1;
                    true
                });
                assert_eq!(seen, pages, "every page of the run is visited, in order");
            }
        }
    }

    /// A run fetches its deepest level a batch at a time, so the guest-read
    /// count falls far below one per page.
    ///
    /// The proxy for the batching itself. `a_run_walk_agrees_with_the_single_walk_on_every_page`
    /// already pins the answers across batch and node boundaries — it walks
    /// 1027 pages on the x86 geometry — so what is unproven without counting is
    /// whether the batch is *taken*. A refill bug that re-read per page would
    /// answer identically and cost what the batch exists to save.
    #[test]
    fn a_long_run_reads_its_deepest_level_in_batches() {
        struct Counting<'a> {
            inner: SliceMemory<'a>,
            reads: core::cell::Cell<usize>,
        }
        impl GuestMemory for Counting<'_> {
            fn read_at(&self, addr: u64, out: &mut [u8]) -> bool {
                self.reads.set(self.reads.get() + 1);
                self.inner.read_at(addr, out)
            }
        }

        const PAGES: u64 = 4 * LEAF_BATCH as u64;
        let geometry = X86_64;
        let mut buf = [0u8; IMAGE];
        let mut b = Builder::new(geometry, &mut buf);
        // One node, every entry of the run mapped to a distinct frame, so a
        // batch that mixed neighbouring words up would be caught below.
        let root = b.map(1, 0, 0x100);
        for i in 1..PAGES {
            b.map_into(root, 1, i, 0x100 + i as u32);
        }
        let mem = Counting {
            inner: SliceMemory::new(b.bytes()),
            reads: core::cell::Cell::new(0),
        };

        let mut seen = 0u64;
        walk_run(&mem, geometry, root, 1, 0, PAGES, &mut |i, got| {
            assert_eq!(
                got.map(|w| w.leaf_pfn),
                Ok(0x100 + i as u32),
                "page {i} took the wrong word of its batch"
            );
            seen += 1;
            true
        });
        assert_eq!(seen, PAGES);
        assert_eq!(
            mem.reads.get() as u64,
            PAGES.div_ceil(LEAF_BATCH as u64),
            "one read per batch, not one per page"
        );
    }

    /// A node whose batch read is refused falls back to one read per word, and
    /// still answers what the single walk answers.
    ///
    /// One unreadable byte fails a whole span, so a batch cannot report *which*
    /// word was bad. Falling back is what keeps a failure attributed to the page
    /// that owns it — and on a host that refuses the wide read for its own
    /// reasons, what keeps the run answering at all.
    #[test]
    fn a_node_that_refuses_a_wide_read_falls_back_to_one_word_at_a_time() {
        struct NarrowOnly<'a> {
            inner: SliceMemory<'a>,
            reads: core::cell::Cell<usize>,
        }
        impl GuestMemory for NarrowOnly<'_> {
            fn read_at(&self, addr: u64, out: &mut [u8]) -> bool {
                self.reads.set(self.reads.get() + 1);
                out.len() <= PTE_SIZE as usize && self.inner.read_at(addr, out)
            }
        }

        const PAGES: u64 = 8;
        let geometry = X86_64;
        let mut buf = [0u8; IMAGE];
        let mut b = Builder::new(geometry, &mut buf);
        let root = b.map(2, 0, 0x11);
        for i in 1..PAGES {
            // Page 3 is left unmapped, so the fallback carries a refusal as
            // well as the frames either side of it.
            if i != 3 {
                b.map_into(root, 2, i, 0x11 + i as u32);
            }
        }
        let mem = NarrowOnly {
            inner: SliceMemory::new(b.bytes()),
            reads: core::cell::Cell::new(0),
        };

        let mut seen = 0u64;
        walk_run(&mem, geometry, root, 2, 0, PAGES, &mut |i, got| {
            let want = walk(&mem, geometry, root, 2, i << geometry.page_shift);
            assert_eq!(got, want, "page {i}");
            seen += 1;
            true
        });
        assert_eq!(seen, PAGES);
        assert!(
            mem.reads.get() >= PAGES as usize,
            "the fallback reads at least once per page"
        );
    }

    /// The visitor stops the run by answering `false`, and no page past it is
    /// walked. A caller checking a page list against the live table stops at the
    /// first disagreement it cares about, and a run that kept reading would cost
    /// the guest-memory reads the stop exists to avoid.
    #[test]
    fn a_run_walk_stops_when_the_visitor_says_so() {
        let geometry = X86_64;
        let mut buf = [0u8; IMAGE];
        let mut b = Builder::new(geometry, &mut buf);
        let root = b.map(2, 0, 0x11);
        b.map_into(root, 2, 1, 0x12);
        let mem = SliceMemory::new(b.bytes());

        let mut visited = 0;
        walk_run(&mem, geometry, root, 2, 0, 64, &mut |_, _| {
            visited += 1;
            visited < 2
        });
        assert_eq!(visited, 2);
    }

    #[test]
    fn an_unmapped_address_reports_not_present_rather_than_corruption() {
        let geometry = X86_64;
        let mut buf = [0u8; IMAGE];
        let mut b = Builder::new(geometry, &mut buf);
        let root = b.map(2, 0, 9);
        let mem = SliceMemory::new(b.bytes());

        // Sibling of the mapped entry at the deepest level.
        let gva = 1u64 << geometry.page_shift;
        let f = walk(&mem, geometry, root, 2, gva).unwrap_err();
        assert_eq!(f.error, WalkError::NotPresent);
        assert_eq!(f.level, 1);
        assert_eq!(f.entry_index, 1);
        assert_eq!(f.raw_pte, 0);
    }

    #[test]
    fn a_nonzero_entry_naming_no_page_is_corruption_not_absence() {
        // A frame number never carries bit 31, so the guest cannot write this
        // and the walker must not treat it as a hole.
        let geometry = X86_64;
        let mut buf = [0u8; IMAGE];
        let mut b = Builder::new(geometry, &mut buf);
        let root = b.map(1, 0, 9);
        b.poke_entry(root, 0, PTE_FLAG_MASK);
        let mem = SliceMemory::new(b.bytes());

        let f = walk(&mem, geometry, root, 1, 0).unwrap_err();
        assert_eq!(f.error, WalkError::MalformedPte);
        assert_eq!(f.raw_pte, PTE_FLAG_MASK);
    }

    #[test]
    #[should_panic(expected = "a PFN never has bit 31 already set")]
    fn the_builder_refuses_a_pfn_that_already_has_bit_31_set() {
        // Guards the guard: if `set_entry` stopped enforcing this, the test
        // above would be synthesizing corruption the guest could also write,
        // and `MalformedPte` would stop meaning corruption.
        let mut buf = [0u8; IMAGE];
        let mut b = Builder::new(X86_64, &mut buf);
        let root = b.alloc_page();
        b.set_entry(root, 0, PTE_FLAG_MASK | 1, false);
    }

    #[test]
    #[should_panic(expected = "an entry is only written into an empty slot")]
    fn the_builder_refuses_to_overwrite_a_live_entry() {
        let mut buf = [0u8; IMAGE];
        let mut b = Builder::new(X86_64, &mut buf);
        let root = b.alloc_page();
        b.set_entry(root, 0, 1, false);
        b.set_entry(root, 0, 2, false);
    }

    #[test]
    fn the_flag_bit_survives_to_the_caller_and_never_reaches_the_frame_number() {
        let geometry = X86_64;
        let mut buf = [0u8; IMAGE];
        let mut b = Builder::new(geometry, &mut buf);
        let root = b.alloc_page();
        b.set_entry(root, 0, 0x1234, true);
        let mem = SliceMemory::new(b.bytes());

        let w = walk(&mem, geometry, root, 1, 0).expect("mapped");
        assert_eq!(w.leaf_pfn, 0x1234);
        assert_eq!(w.raw_pte, PTE_FLAG_MASK | 0x1234);
        assert_eq!(w.addr_page, geometry.pfn_to_addr(0x1234));
    }

    #[test]
    fn a_depth_outside_the_bound_is_refused_before_any_page_is_read() {
        let geometry = X86_64;
        let mut buf = [0u8; IMAGE];
        let b = Builder::new(geometry, &mut buf);
        let mem = SliceMemory::new(b.bytes());
        assert_eq!(
            walk(&mem, geometry, 1, 0, 0).unwrap_err().error,
            WalkError::ZeroDepth
        );
        assert_eq!(
            walk(&mem, geometry, 1, MAX_DEPTH + 1, 0).unwrap_err().error,
            WalkError::DepthTooDeep
        );
        assert_eq!(
            walk(&mem, geometry, 0, 1, 0).unwrap_err().error,
            WalkError::ZeroRootPfn
        );
    }

    #[test]
    fn a_table_page_outside_the_image_reports_a_read_failure() {
        let geometry = X86_64;
        let mut buf = [0u8; IMAGE];
        let mut b = Builder::new(geometry, &mut buf);
        let root = b.alloc_page();
        // Point at a frame the image does not contain.
        b.set_entry(root, 0, 0x7fff, false);
        let mem = SliceMemory::new(b.bytes());

        let f = walk(&mem, geometry, root, 2, 0).unwrap_err();
        assert_eq!(f.error, WalkError::TableRead);
        assert_eq!(f.level, 1);
    }

    #[test]
    fn the_directory_supplies_root_and_depth_rather_than_a_constant() {
        let geometry = X86_64;
        let mut buf = [0u8; IMAGE];
        let mut b = Builder::new(geometry, &mut buf);
        let dir = b.alloc_page();
        b.poke_entry(dir, 0, 7); // root pfn
        b.poke_entry(dir, 1, 3); // depth
        let mem = SliceMemory::new(b.bytes());

        assert_eq!(read_directory(&mem, geometry, dir), Ok((7, 3)));
        assert_eq!(
            read_directory(&mem, geometry, 0),
            Err(WalkError::ZeroRootPfn)
        );
    }
}
