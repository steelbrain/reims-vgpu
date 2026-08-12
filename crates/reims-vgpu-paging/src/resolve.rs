//! Task GVA resolver: the device's walk above the wire page-table format.
//!
//! The format and the descent live in [`reims_vgpu_wire::page_table`], which
//! owns them — [`Geometry`] and the two pathway constants are re-exports, so
//! there is no second declaration to drift. What this module adds is the layer
//! the device consumes: task-root reads, the typed refusal statuses its
//! failure channel reports (the mapping onto that channel lives in the device,
//! next to the channel), and the run form the span readers use.

use reims_vgpu_wire::mem::GuestMemory;
use reims_vgpu_wire::page_table as wire_page_table;

pub use reims_vgpu_wire::page_table::{Geometry, ARM64E, X86_64};

/// The page-table geometry for a guest page shift, or `None`.
///
/// Only 12 (x86_64) and 14 (arm64e) are geometries this device walks, and the
/// answer is `None` for anything else — there is no arm default, because a walk
/// at the wrong stride reads a tree that is there and returns PFNs that are
/// wrong rather than failing.
///
/// This sits next to the two constants rather than in the device because the
/// device is not its only caller: [`crate::span`] takes a `Geometry` and every
/// device rail above it starts from a `page_shift`, so the selector would
/// otherwise be a device-side function that the crate holding both geometries
/// cannot use. It reads neither a fixture nor a host, which is what leaves it
/// here rather than a rung either side.
#[inline]
pub fn geometry_for_page_shift(page_shift: u32) -> Option<Geometry> {
    if page_shift == X86_64.page_shift {
        Some(X86_64)
    } else if page_shift == ARM64E.page_shift {
        Some(ARM64E)
    } else {
        None
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Task {
    pub active: bool,
    pub directory_pfn: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct TaskRoot {
    pub directory_pfn: u32,
    pub root_pfn: u32,
    pub depth: u32,
}

/// Which of the walk's checks refused, one variant per check.
///
/// The discriminants are stable on purpose — they have been reported since the
/// walk was written and readings of old logs are compared against new ones.
/// The device maps each variant onto its always-on failure channel; nothing
/// here emits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
#[derive(Default)]
pub enum ResolveStatus {
    #[default]
    Ok = 0,
    ErrArgs = 1,
    ErrInactiveTask = 2,
    ErrNoDirectory = 3,
    ErrDirectoryRead = 4,
    ErrZeroRootPfn = 5,
    ErrZeroDepth = 6,
    ErrDepthTooDeep = 7,
    // 8 was `ErrAddressOutOfRange`, removed rather than left as vocabulary no
    // boot could ever print. Nothing in this crate or the device constructed
    // it, and the format says nothing could: the walk masks each level's index
    // to `index_bits` and discards the bits of the GVA above the tree's reach —
    // which is what the guest's own descent does, so an address "past the end"
    // is not a condition either side recognises. The discriminant is left as a
    // hole rather than reused, because these numbers are compared across logs.
    ErrPageTableRead = 9,
    ErrZeroPfn = 10,
    ErrMalformedPte = 11,
    ErrUnsupportedGeometry = 14,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Translation {
    pub status: ResolveStatus,
    pub gva: u64,
    pub gpa: u64,
    pub gva_page_index: u64,
    pub gpa_page: u64,
    pub directory_pfn: u32,
    pub root_pfn: u32,
    pub depth: u32,
    pub leaf_pfn: u32,
    pub level: u32,
    pub entry_index: u32,
    pub raw_pte: u32,
}

/// Wire walk failures as the device's typed refusals.
///
/// The names differ in one place worth stating: the wire crate calls a zero
/// entry `NotPresent`, because the table's builder writes zero for absent.
/// This layer calls the same thing `ErrZeroPfn`, which is the guest saying
/// "not mapped here" rather than a device defect. The device's log slug is
/// unchanged so the fail log reads the same.
fn resolve_status_of(error: wire_page_table::WalkError) -> ResolveStatus {
    use wire_page_table::WalkError as W;
    match error {
        W::UnsupportedGeometry => ResolveStatus::ErrUnsupportedGeometry,
        W::ZeroRootPfn => ResolveStatus::ErrZeroRootPfn,
        W::ZeroDepth => ResolveStatus::ErrZeroDepth,
        W::DepthTooDeep => ResolveStatus::ErrDepthTooDeep,
        W::TableRead => ResolveStatus::ErrPageTableRead,
        W::NotPresent => ResolveStatus::ErrZeroPfn,
        W::MalformedPte => ResolveStatus::ErrMalformedPte,
    }
}

pub fn resolve_status_name(status: ResolveStatus) -> &'static str {
    match status {
        ResolveStatus::Ok => "ok",
        ResolveStatus::ErrArgs => "args",
        ResolveStatus::ErrInactiveTask => "inactive-task",
        ResolveStatus::ErrNoDirectory => "no-directory",
        ResolveStatus::ErrDirectoryRead => "directory-read",
        ResolveStatus::ErrZeroRootPfn => "zero-root-pfn",
        ResolveStatus::ErrZeroDepth => "zero-depth",
        ResolveStatus::ErrDepthTooDeep => "depth-too-deep",
        ResolveStatus::ErrPageTableRead => "page-table-read",
        ResolveStatus::ErrZeroPfn => "zero-pfn",
        ResolveStatus::ErrMalformedPte => "malformed-pte",
        ResolveStatus::ErrUnsupportedGeometry => "unsupported-geometry",
    }
}

pub fn read_task_root(
    mem: &dyn GuestMemory,
    task: &Task,
    geometry: Geometry,
) -> Result<TaskRoot, ResolveStatus> {
    if geometry.validate().is_err() {
        return Err(ResolveStatus::ErrUnsupportedGeometry);
    }
    if !task.active {
        return Err(ResolveStatus::ErrInactiveTask);
    }
    if task.directory_pfn == 0 {
        return Err(ResolveStatus::ErrNoDirectory);
    }
    // Geometry and the zero directory are checked above with this module's own
    // statuses, so the only failure left for the wire read is the read itself.
    let (root_pfn, depth) = wire_page_table::read_directory(&mem, geometry, task.directory_pfn)
        .map_err(|_| ResolveStatus::ErrDirectoryRead)?;
    Ok(TaskRoot {
        directory_pfn: task.directory_pfn,
        root_pfn,
        depth,
    })
}

/// The most interior nodes one descent can read, which is the tree's depth.
pub const MAX_TREE_NODES: usize = wire_page_table::MAX_DEPTH as usize;

/// The guest-physical pages holding page-table entries that `gva` descends
/// through under `task`, written into `out`, returning how many were read.
///
/// **Interior pages only.** The page the walk resolves *to* is data and is
/// deliberately absent, so a caller may read this as "these pages must not be
/// written" without that claim swallowing every page the tree maps.
///
/// Short, or empty, whenever the walk refused early — a task with no directory
/// reads zero, and an address that stops being mapped mid-teardown reads only
/// the levels above where it stopped. Those levels were really read and really
/// are nodes, so they are reported rather than discarded; a caller must not read
/// a short answer as "the tree has no more nodes than this".
pub fn task_node_gpas(
    mem: &dyn GuestMemory,
    geometry: Geometry,
    task: &Task,
    gva: u64,
    out: &mut [u64; MAX_TREE_NODES],
) -> usize {
    let Ok(root) = read_task_root(mem, task, geometry) else {
        return 0;
    };
    let mut nodes = wire_page_table::NodePath::default();
    // The walk's own verdict is discarded on purpose: an unresolvable address
    // still descended through real nodes, and those are the answer here.
    let _ = wire_page_table::walk_recording_nodes(
        &mem,
        geometry,
        root.root_pfn,
        root.depth,
        gva,
        &mut nodes,
    );
    let pfns = nodes.pfns();
    for (slot, &pfn) in out.iter_mut().zip(pfns) {
        *slot = geometry.pfn_to_addr(pfn);
    }
    pfns.len()
}

pub fn translate_root(
    mem: &dyn GuestMemory,
    geometry: Geometry,
    root_pfn: u32,
    depth: u32,
    gva: u64,
) -> Translation {
    let mut out = Translation {
        gva,
        root_pfn,
        depth,
        ..Default::default()
    };
    if geometry.validate().is_err() {
        out.status = ResolveStatus::ErrUnsupportedGeometry;
        return out;
    }
    out.gva_page_index = gva >> geometry.page_shift;

    // The descent itself lives in `reims_vgpu_wire::page_table`, which owns the
    // format. Keeping it there means there is one declaration rather than two
    // that could drift apart silently, and the tree walk gets exercised by that
    // crate's tests as well as by this module's.
    match wire_page_table::walk(&mem, geometry, root_pfn, depth, gva) {
        Ok(w) => {
            // On success the walker reports the deepest level it read, which is
            // where the loop this replaced left these fields.
            out.level = depth - 1;
            out.entry_index = (w.page_index & geometry.index_mask()) as u32;
            out.raw_pte = w.raw_pte;
            out.status = ResolveStatus::Ok;
            out.leaf_pfn = w.leaf_pfn;
            out.gpa_page = w.addr_page;
            out.gpa = w.addr;
        }
        Err(f) => {
            out.level = f.level;
            out.entry_index = f.entry_index;
            out.raw_pte = f.raw_pte;
            out.status = resolve_status_of(f.error);
        }
    }
    out
}

/// Translate a run of consecutive pages under one root, calling `visit` with
/// each page's GPA or the walk's refusal for a page the table cannot
/// translate.
///
/// The run form of [`translate_root`], and it exists for one reason: the
/// per-page form re-reads every upper level of the tree for every page, and a
/// run visits each page index exactly once. A 1080p surface covers around two
/// thousand consecutive pages whose upper levels are the same three or four
/// entries, and the descent — [`wire_page_table::walk_run`] — reads each
/// shared entry once.
///
/// The visitor stops the run by answering `false`. It is not called at all when
/// the root or geometry is unusable, so a caller must compare what it saw
/// against what it expected rather than reading a quiet return as agreement —
/// the same contract [`translate_root`] has by returning a status.
pub fn translate_root_run(
    mem: &dyn GuestMemory,
    geometry: Geometry,
    root_pfn: u32,
    depth: u32,
    first_gva: u64,
    pages: u64,
    visit: &mut dyn FnMut(u64, Result<u64, ResolveStatus>) -> bool,
) {
    if geometry.validate().is_err() || root_pfn == 0 || depth == 0 {
        return;
    }
    wire_page_table::walk_run(
        &mem,
        geometry,
        root_pfn,
        depth,
        first_gva,
        pages,
        &mut |index, walked| {
            visit(
                index,
                walked
                    .map(|w| w.addr)
                    .map_err(|f| resolve_status_of(f.error)),
            )
        },
    );
}

/// What a range's leaf entries looked like, page by page.
///
/// Counts rather than a verdict: this module answers what the tree says and the
/// caller decides what that means. See [`range_coverage`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct RangeCoverage {
    /// Pages the walk was asked for.
    pub pages: u64,
    /// Pages whose descent reached a leaf entry naming a frame.
    pub present: u64,
    /// Pages whose descent stopped on a **zero** entry, at any level.
    pub absent: u64,
    /// Pages whose descent could not be answered — a table page that would not
    /// read, or an entry the format cannot produce. Neither present nor absent.
    pub undecidable: u64,
    /// Index within the run of the first absent page, meaningful when
    /// [`Self::absent`] is non-zero.
    pub first_absent_index: u64,
    /// Level the first absent page's descent stopped at, zero-based from the
    /// root. `first_absent_level + 1 == depth` is a **leaf** entry that is zero;
    /// anything shallower is a whole absent subtree.
    pub first_absent_level: u32,
    /// The tree's depth, as the task's own directory declares it.
    pub depth: u32,
}

/// Walk `pages` consecutive pages from `first_gva` under `task` and count what
/// the tree holds for each.
///
/// # What this is for
///
/// One guest line's page-table teardown asserts, per page of the range it is
/// given, that the leaf entry is **not already zero** — and takes the whole
/// guest down when it is not. The range it tears down is the range it named on
/// the wire, and it tears it down after this device has replied. So walking the
/// range at that moment answers whether the guest is about to assert, before it
/// does, using a tree this device is already entitled to read.
///
/// The level is carried because it separates two different guest assertions: a
/// zero at the deepest level is a leaf entry the teardown refuses to clear, and
/// a zero above it is a missing interior node the same teardown refuses to
/// descend through. Both end the guest; they are not the same defect.
///
/// `None` when the task has no readable root — an inactive task, a task with no
/// directory, or a directory page that would not read. That is not a coverage
/// answer and must not be counted as one.
///
/// The cost is [`wire_page_table::walk_run`]'s: the upper levels are read once
/// per subtree rather than once per page, and the deepest level a batch at a
/// time. The guest's own teardown walks every one of these pages, so the reach
/// asked for here is never more than the reach the guest has already committed
/// to; bounding it is the caller's decision and not this function's.
pub fn range_coverage(
    mem: &dyn GuestMemory,
    geometry: Geometry,
    task: &Task,
    first_gva: u64,
    pages: u64,
) -> Option<RangeCoverage> {
    let root = read_task_root(mem, task, geometry).ok()?;
    if root.root_pfn == 0 || root.depth == 0 {
        return None;
    }
    let mut out = RangeCoverage {
        pages,
        depth: root.depth,
        ..RangeCoverage::default()
    };
    wire_page_table::walk_run(
        &mem,
        geometry,
        root.root_pfn,
        root.depth,
        first_gva,
        pages,
        &mut |index, walked| {
            match walked {
                Ok(_) => out.present += 1,
                // `NotPresent` is the entry reading zero, which is the whole
                // question. Every other refusal is a read this walk could not
                // complete and says nothing about what the guest will find.
                Err(f) if f.error == wire_page_table::WalkError::NotPresent => {
                    if out.absent == 0 {
                        out.first_absent_index = index;
                        out.first_absent_level = f.level;
                    }
                    out.absent += 1;
                }
                Err(_) => out.undecidable += 1,
            }
            true
        },
    );
    Some(out)
}

#[cfg(test)]
mod tests {

    use super::*;
    use alloc::collections::BTreeMap;
    use alloc::vec::Vec;
    use reims_vgpu_wire::page_table::{
        DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN, PTE_FLAG_MASK, PTE_SIZE,
    };

    const PAGE_SHIFT_ARM64E: u32 = ARM64E.page_shift;

    /// The two supported page shifts select their own geometry and every other
    /// shift selects none.
    ///
    /// The `None` half is the load-bearing one. A shift with no geometry that
    /// fell back to an arm default would walk a real tree at the wrong stride:
    /// the reads succeed, the indices come out of the wrong bits, and the PFNs
    /// returned are wrong rather than absent.
    #[test]
    fn a_page_shift_selects_its_own_geometry_and_nothing_else_selects_one() {
        assert_eq!(geometry_for_page_shift(X86_64.page_shift), Some(X86_64));
        assert_eq!(geometry_for_page_shift(ARM64E.page_shift), Some(ARM64E));
        // Between the two, below both, and past the width of an address.
        for shift in [0, 11, 13, 15, 16, 21, 64, u32::MAX] {
            assert_eq!(geometry_for_page_shift(shift), None, "shift {shift}");
        }
    }

    struct MapReader {
        map: BTreeMap<u64, u8>,
    }

    impl MapReader {
        fn new() -> Self {
            Self {
                map: BTreeMap::new(),
            }
        }
        fn put_u32(&mut self, gpa: u64, v: u32) {
            for (i, b) in v.to_le_bytes().iter().enumerate() {
                self.map.insert(gpa + i as u64, *b);
            }
        }
    }

    impl GuestMemory for MapReader {
        fn read_at(&self, gpa: u64, dst: &mut [u8]) -> bool {
            for (i, slot) in dst.iter_mut().enumerate() {
                match self.map.get(&(gpa + i as u64)) {
                    Some(b) => *slot = *b,
                    None => return false,
                }
            }
            true
        }
    }

    #[test]
    fn geometry_defaults() {
        assert!(ARM64E.validate().is_ok());
        assert!(X86_64.validate().is_ok());
        let mut bad = ARM64E;
        bad.page_shift = 13;
        assert!(bad.validate().is_err());
    }

    #[test]
    fn task_root_reads_directory_root_and_depth() {
        let mut r = MapReader::new();
        let dir_gpa = (2u64) << PAGE_SHIFT_ARM64E;
        r.put_u32(dir_gpa + DIRECTORY_ROOT_PFN, 1);
        r.put_u32(dir_gpa + DIRECTORY_DEPTH, 1);
        let task = Task {
            active: true,
            directory_pfn: 2,
        };
        let root = read_task_root(&r, &task, ARM64E).unwrap();
        assert_eq!(root.directory_pfn, 2);
        assert_eq!(root.root_pfn, 1);
        assert_eq!(root.depth, 1);
    }

    /// The node GPAs are the pages the descent read entries out of, and the
    /// page the address resolves to is not one of them.
    ///
    /// Written against a two-level tree so "interior" and "root" are not the
    /// same page: at depth 1 a walker that reported the leaf instead of the root
    /// would still return one address and could pass by accident.
    #[test]
    fn the_node_gpas_are_the_tables_and_not_the_page_they_map() {
        let mut r = MapReader::new();
        // depth 2, root pfn 1: root[0] -> pfn 3, and pfn 3's entry 0 -> pfn 9.
        let dir_gpa = 2u64 << PAGE_SHIFT_ARM64E;
        r.put_u32(dir_gpa + DIRECTORY_ROOT_PFN, 1);
        r.put_u32(dir_gpa + DIRECTORY_DEPTH, 2);
        r.put_u32(1u64 << PAGE_SHIFT_ARM64E, 3);
        r.put_u32(3u64 << PAGE_SHIFT_ARM64E, 9);
        let task = Task {
            active: true,
            directory_pfn: 2,
        };

        let mut out = [0u64; MAX_TREE_NODES];
        let n = task_node_gpas(&r, ARM64E, &task, 0x40, &mut out);
        assert_eq!(n, 2, "one node per level walked");
        assert_eq!(
            &out[..n],
            &[1u64 << PAGE_SHIFT_ARM64E, 3u64 << PAGE_SHIFT_ARM64E]
        );

        // The address really does resolve into pfn 9, and pfn 9 is not a node.
        let t = translate_root(&r, ARM64E, 1, 2, 0x40);
        assert_eq!(t.status, ResolveStatus::Ok);
        assert_eq!(t.leaf_pfn, 9);
        assert!(!out[..n].contains(&(9u64 << PAGE_SHIFT_ARM64E)));
    }

    /// A task the walk cannot even start on reports no nodes rather than a
    /// stale or invented one.
    #[test]
    fn a_task_with_no_directory_reports_no_nodes() {
        let r = MapReader::new();
        let mut out = [0u64; MAX_TREE_NODES];
        for task in [
            Task {
                active: false,
                directory_pfn: 2,
            },
            Task {
                active: true,
                directory_pfn: 0,
            },
        ] {
            assert_eq!(task_node_gpas(&r, ARM64E, &task, 0x40, &mut out), 0);
        }
    }

    /// An address that stops resolving still reports the levels above where it
    /// stopped — which is the case the guard exists for, because a guest tearing
    /// a task down unmaps from the bottom.
    #[test]
    fn an_unresolvable_address_still_reports_the_nodes_above_it() {
        let mut r = MapReader::new();
        let dir_gpa = 2u64 << PAGE_SHIFT_ARM64E;
        r.put_u32(dir_gpa + DIRECTORY_ROOT_PFN, 1);
        r.put_u32(dir_gpa + DIRECTORY_DEPTH, 2);
        // The root's entry names pfn 3, and pfn 3's own entry is written as the
        // format's not-present encoding — a real zero rather than a hole in the
        // reader, so the walk refuses on the guest's table and not on the test.
        r.put_u32(1u64 << PAGE_SHIFT_ARM64E, 3);
        r.put_u32(3u64 << PAGE_SHIFT_ARM64E, 0);
        let task = Task {
            active: true,
            directory_pfn: 2,
        };

        assert_eq!(
            translate_root(&r, ARM64E, 1, 2, 0x40).status,
            ResolveStatus::ErrZeroPfn
        );
        let mut out = [0u64; MAX_TREE_NODES];
        let n = task_node_gpas(&r, ARM64E, &task, 0x40, &mut out);
        assert_eq!(
            &out[..n],
            &[1u64 << PAGE_SHIFT_ARM64E, 3u64 << PAGE_SHIFT_ARM64E],
            "both levels were read before the walk refused"
        );
    }

    #[test]
    fn translate_one_level() {
        // depth=1, root_pfn=1, GVA page 0 -> leaf pfn 5
        let mut r = MapReader::new();
        // table at pfn 1: entry 0 = pfn 5
        let table_gpa = (1u64) << PAGE_SHIFT_ARM64E;
        r.put_u32(table_gpa, 5);
        let t = translate_root(&r, ARM64E, 1, 1, 0x100);
        assert_eq!(t.status, ResolveStatus::Ok);
        assert_eq!(t.leaf_pfn, 5);
        assert_eq!(t.gpa, ((5u64) << PAGE_SHIFT_ARM64E) + 0x100);
    }

    /// The run form answers page for page what the single form answers, and
    /// carries the same refusal for a page that does not translate.
    #[test]
    fn a_run_answers_as_the_single_form_does_and_keeps_the_refusal() {
        let mut r = MapReader::new();
        let table_gpa = (1u64) << PAGE_SHIFT_ARM64E;
        // Pages 0 and 2 map; page 1 is absent (entry reads zero, which the
        // reader models as an unreadable word — TableRead rather than
        // NotPresent, and either way a refusal the visitor must see).
        r.put_u32(table_gpa, 5);
        r.put_u32(table_gpa + 2 * PTE_SIZE as u64, 7);
        let mut seen = Vec::new();
        translate_root_run(&r, ARM64E, 1, 1, 0x100, 3, &mut |i, res| {
            seen.push((i, res));
            true
        });
        assert_eq!(seen.len(), 3);
        for (i, res) in &seen {
            let single = translate_root(&r, ARM64E, 1, 1, ((*i) << PAGE_SHIFT_ARM64E) + 0x100);
            match res {
                Ok(gpa) => {
                    assert_eq!(single.status, ResolveStatus::Ok);
                    assert_eq!(*gpa, single.gpa);
                }
                Err(status) => assert_eq!(*status, single.status),
            }
        }
        assert!(seen[1].1.is_err(), "the unmapped page carries its refusal");
    }

    /// A full-depth walk descends every level and takes the leaf from the last
    /// one.
    ///
    /// Every level of the walk reads its PTE identically, so the leaf is just
    /// the PFN the final level named. Nothing tested a walk deeper than one
    /// level before, which left that "just" unchecked: a walk that stopped one
    /// level early, or that took the leaf from the table PFN instead of the
    /// entry, resolves `depth == 1` correctly and every deeper `depth` wrong.
    /// The four indices here are distinct so an off-by-one level lands on the
    /// wrong table and reads nothing.
    #[test]
    fn a_four_level_walk_takes_its_leaf_from_the_deepest_entry() {
        const IDX: [u64; 4] = [1, 2, 3, 4];
        const TABLE_PFN: [u32; 4] = [10, 11, 12, 13];
        const LEAF_PFN: u32 = 0x555;
        const PAGE_OFF: u64 = 0x1234;

        let g = ARM64E;
        let page_index = (IDX[0] << 36) | (IDX[1] << 24) | (IDX[2] << 12) | IDX[3];

        let mut r = MapReader::new();
        for level in 0..4usize {
            let next = if level == 3 {
                // Flag bit set: the walk must mask it off, not fold it into the PFN.
                PTE_FLAG_MASK | LEAF_PFN
            } else {
                TABLE_PFN[level + 1]
            };
            let tbl = (TABLE_PFN[level] as u64) << g.page_shift;
            r.put_u32(tbl + IDX[level] * PTE_SIZE as u64, next);
        }

        let gva = (page_index << g.page_shift) | PAGE_OFF;
        let t = translate_root(&r, g, TABLE_PFN[0], 4, gva);

        assert_eq!(t.status, ResolveStatus::Ok);
        assert_eq!(t.leaf_pfn, LEAF_PFN);
        assert_eq!(t.gpa, ((LEAF_PFN as u64) << g.page_shift) + PAGE_OFF);
        assert_eq!(t.gpa_page, (LEAF_PFN as u64) << g.page_shift);
        assert_eq!(t.level, 3, "the last level walked");
        assert_eq!(t.entry_index as u64, IDX[3]);
        assert_eq!(t.raw_pte, PTE_FLAG_MASK | LEAF_PFN);
    }

    /// No address a walk can form overflows, at any geometry the wire crate's
    /// `validate` accepts.
    ///
    /// The walk used to carry five `u64::MAX - x < y` guards and a fallible
    /// PFN-to-GPA helper, all of which were dead: a PFN is a `u32` and the
    /// accepted page shifts are 12 and 14, so the widest address the walk can
    /// name is under 2^46 and every addend is under 2^17. Those guards are gone,
    /// which makes this the only thing holding the premise. It fails if a
    /// geometry with a wider shift is ever accepted, which is exactly when the
    /// guards would have needed to come back.
    #[test]
    fn accepted_geometries_cannot_form_an_address_that_overflows() {
        for g in [ARM64E, X86_64] {
            assert!(g.validate().is_ok());
            // The widest table or leaf base a `u32` PFN can name.
            let max_base = g.pfn_to_addr(u32::MAX);
            let max_entry_off = g.index_mask() * (PTE_SIZE as u64);
            let max_page_off = g.page_offset_mask();
            assert!(max_base.checked_add(max_entry_off).is_some());
            assert!(max_base.checked_add(max_page_off).is_some());
        }

        // And nothing wider is accepted. The shift is the only thing a
        // geometry carries, so it is the only thing that can decide this.
        for shift in 0..64u32 {
            let g = Geometry { page_shift: shift };
            if g.validate().is_ok() {
                assert!(
                    shift == ARM64E.page_shift || shift == X86_64.page_shift,
                    "unexpected page shift accepted"
                );
            }
        }
    }

    #[test]
    fn zero_pfn_and_inactive() {
        let r = MapReader::new();
        let task = Task {
            active: false,
            directory_pfn: 1,
        };
        assert_eq!(
            read_task_root(&r, &task, ARM64E).unwrap_err(),
            ResolveStatus::ErrInactiveTask
        );
        let t = translate_root(&r, ARM64E, 0, 1, 0);
        assert_eq!(t.status, ResolveStatus::ErrZeroRootPfn);
    }

    /// A one-level tree whose directory says so, with `entries` written into the
    /// root table starting at index 0. A zero entry is written as a real zero
    /// word rather than left out, because an absent byte reads as an unreadable
    /// table and that is a different answer from an entry that says nothing is
    /// mapped.
    fn one_level_task(entries: &[u32]) -> (MapReader, Task) {
        let mut r = MapReader::new();
        let dir_gpa = (2u64) << PAGE_SHIFT_ARM64E;
        r.put_u32(dir_gpa + DIRECTORY_ROOT_PFN, 1);
        r.put_u32(dir_gpa + DIRECTORY_DEPTH, 1);
        let table_gpa = (1u64) << PAGE_SHIFT_ARM64E;
        for (i, &e) in entries.iter().enumerate() {
            r.put_u32(table_gpa + i as u64 * PTE_SIZE as u64, e);
        }
        (
            r,
            Task {
                active: true,
                directory_pfn: 2,
            },
        )
    }

    /// A range every one of whose pages is mapped reports no absence, and the
    /// counts add up to the pages asked for.
    #[test]
    fn a_fully_mapped_range_is_covered() {
        let (r, task) = one_level_task(&[5, 6, 7, 8]);
        let c = range_coverage(&r, ARM64E, &task, 0, 4).unwrap();
        assert_eq!(c.pages, 4);
        assert_eq!(c.present, 4);
        assert_eq!(c.absent, 0);
        assert_eq!(c.undecidable, 0);
        assert_eq!(c.depth, 1);
    }

    /// A range whose leaf entries all read zero reports every page absent, and
    /// says the zero is at the deepest level.
    ///
    /// That level is the discriminator the caller needs: this is the shape whose
    /// guest teardown refuses to clear an entry that is already zero, and it is
    /// not the same defect as a subtree that is not there at all.
    #[test]
    fn a_range_of_zero_leaf_entries_is_absent_at_the_deepest_level() {
        let (r, task) = one_level_task(&[0, 0, 0]);
        let c = range_coverage(&r, ARM64E, &task, 0, 3).unwrap();
        assert_eq!(c.present, 0);
        assert_eq!(c.absent, 3);
        assert_eq!(c.first_absent_index, 0);
        assert_eq!(c.first_absent_level + 1, c.depth);
    }

    /// One zero among live entries is found, counted once, and located.
    ///
    /// The scattered shape and the wholly-absent shape are different guest
    /// stories — a mapping that was never fully wired against a range torn down
    /// twice — so a scan that only answered "some page is missing" would merge
    /// them.
    #[test]
    fn a_single_hole_is_located_within_a_live_range() {
        let (r, task) = one_level_task(&[5, 6, 0, 8, 9]);
        let c = range_coverage(&r, ARM64E, &task, 0, 5).unwrap();
        assert_eq!(c.present, 4);
        assert_eq!(c.absent, 1);
        assert_eq!(c.first_absent_index, 2);
        assert_eq!(c.first_absent_level + 1, c.depth);
    }

    /// A zero entry above the leaf reports the level it stopped at, which is not
    /// the leaf level.
    #[test]
    fn an_absent_subtree_reports_the_level_it_stopped_at() {
        let mut r = MapReader::new();
        let dir_gpa = (2u64) << PAGE_SHIFT_ARM64E;
        r.put_u32(dir_gpa + DIRECTORY_ROOT_PFN, 1);
        r.put_u32(dir_gpa + DIRECTORY_DEPTH, 2);
        // The root's entry 0 reads a real zero: the whole subtree under it is
        // gone, so no page of the range has a leaf entry to be zero.
        r.put_u32((1u64) << PAGE_SHIFT_ARM64E, 0);
        let task = Task {
            active: true,
            directory_pfn: 2,
        };
        let c = range_coverage(&r, ARM64E, &task, 0, 4).unwrap();
        assert_eq!(c.absent, 4);
        assert_eq!(c.depth, 2);
        assert_eq!(c.first_absent_level, 0);
        assert!(
            c.first_absent_level + 1 < c.depth,
            "an absent subtree is shallower than a zero leaf entry"
        );
    }

    /// A task with no readable root has no coverage answer at all, which is not
    /// the same as a range that is fully absent.
    #[test]
    fn a_task_with_no_root_has_no_coverage_answer() {
        let (r, _) = one_level_task(&[5]);
        for task in [
            Task {
                active: false,
                directory_pfn: 2,
            },
            Task {
                active: true,
                directory_pfn: 0,
            },
            Task {
                active: true,
                directory_pfn: 9999,
            },
        ] {
            assert_eq!(range_coverage(&r, ARM64E, &task, 0, 4), None);
        }
    }

    /// A table page that will not read is undecidable, and is counted apart from
    /// absence.
    ///
    /// Merging the two is the failure that matters: an unreadable table would
    /// otherwise be reported as a range the guest is about to assert on, and the
    /// alarm this feeds costs a session when it is wrong.
    #[test]
    fn an_unreadable_table_is_undecidable_and_not_absence() {
        let mut r = MapReader::new();
        let dir_gpa = (2u64) << PAGE_SHIFT_ARM64E;
        r.put_u32(dir_gpa + DIRECTORY_ROOT_PFN, 1);
        r.put_u32(dir_gpa + DIRECTORY_DEPTH, 1);
        // Nothing at all is written for the root table, so every entry read
        // fails rather than returning a word.
        let task = Task {
            active: true,
            directory_pfn: 2,
        };
        let c = range_coverage(&r, ARM64E, &task, 0, 4).unwrap();
        assert_eq!(c.absent, 0);
        assert_eq!(c.present, 0);
        assert_eq!(c.undecidable, 4);
    }
}
