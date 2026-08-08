//! What a decoded dispatch record means: its `MTLDispatchType`, the threadgroup
//! counts a compute grid resolves to, and the three extents a mesh draw encodes.
//!
//! All three are read off the wire and none is backend-specific, so each is
//! answered here once rather than at every backend's encode site. Each is a
//! *closed* rule with a substitution behind it, which is why they are functions
//! and not open-coded comparisons — see
//! [`crate::contract::dispatch::workgroup_counts`] for what the split version of
//! the second one cost.
//!
//! The two dimension rules are deliberately neighbours, because they disagree
//! and the disagreement is the contract. A compute dispatch may substitute
//! nothing: every extent is the guest's. A mesh draw may substitute exactly one
//! thing, the object threadgroup Metal demands even with no object stage, and it
//! reports having done so. Read either alone and the other looks like a bug.
//!
//! # The dispatch type
//!
//! Two ordinals, and the only interesting thing about them is that the accepted
//! set is *closed*: `MTLDispatchType` in the Metal SDK has exactly `Serial` and
//! `Concurrent`, so unlike a pixel format or a primitive type there is no
//! "value this device has no contract for yet" to leave room for. An ordinal
//! outside the pair is a malformed record or a wrong wire offset, not a guest
//! feature.
//!
//! # Why they live here rather than in the backend
//!
//! The value arrives on the wire, from the guest, and is decoded by
//! [`crate::runtime::decode::compute`] — none of which is backend-specific. It
//! was previously reachable only through `backend::metal::abi`, which is
//! `backend-metal`-gated, so the shared code that accepts the field could not
//! name the values it was accepting and the one place that narrowed it ran on a
//! single arm. `contract/` is where a number that comes from the wire and the
//! SDK belongs, per this module tree's own doc.
//!
//! `backend::metal::abi` keeps its own spelling of the pair, because that module
//! is a mirror of an archived C header and its provenance is the point. A `const`
//! assertion there pins the two spellings equal, so a divergence is a build
//! failure on every arm that compiles the mirror — including the cross-compiled
//! `--target aarch64-apple-darwin` clippy run `AGENTS.md` requires from Linux.

/// `MTLDispatchTypeSerial` — dispatches in a segment may not overlap.
pub const MTL_DISPATCH_TYPE_SERIAL: u32 = 0;
/// `MTLDispatchTypeConcurrent` — Metal may overlap dispatches in a segment.
pub const MTL_DISPATCH_TYPE_CONCURRENT: u32 = 1;

/// Whether `raw` is one of the two dispatch types the contract declares.
///
/// Beside the constants on purpose. The rule this answers used to be written
/// out at the site that consumed it, as
/// `if x == CONCURRENT { CONCURRENT } else { SERIAL }` — a comparison that reads
/// as a narrowing and is really an unreported substitution, and which nothing
/// could compare against the constants it was narrowing to.
#[must_use]
pub fn is_declared_dispatch_type(raw: u32) -> bool {
    raw == MTL_DISPATCH_TYPE_SERIAL || raw == MTL_DISPATCH_TYPE_CONCURRENT
}

/// The threadgroup counts a dispatch resolves to, or `None` if it has no work.
///
/// `grid` is the record's grid and `tg` its threads-per-threadgroup, both
/// straight off the wire. `grid_is_threads` distinguishes Metal's two spellings:
/// `DispatchThreadgroups` states the count directly, while `DispatchThreads`
/// states a total thread count that Metal divides by the threadgroup size,
/// rounding up — which is what `div_ceil` reproduces here.
///
/// # Why the zero test and the division are one function
///
/// They were two, about two hundred lines apart, and the distance is what made
/// the pair wrong: the division carried a `.max(1)` on each quotient, which the
/// zero test above it had already made unreachable. That clamp reads as
/// prudence and is the opposite. A `grid` component of zero is a guest asking
/// for no threads, and the only faithful answer is no dispatch — fabricating
/// one threadgroup runs the kernel, and a kernel that runs writes the storage
/// buffers and images bound to it. So the substitution this device must never
/// make is available only where the test that forbids it also lives.
///
/// A zero in `tg` is refused for a second reason: it is the divisor, so the
/// alternative to refusing it is a panic on a value that came off the wire.
#[must_use]
pub fn workgroup_counts(grid: [u32; 3], tg: [u32; 3], grid_is_threads: bool) -> Option<[u32; 3]> {
    if grid.iter().chain(&tg).any(|&d| d == 0) {
        return None;
    }
    if !grid_is_threads {
        return Some(grid);
    }
    Some([
        grid[0].div_ceil(tg[0]),
        grid[1].div_ceil(tg[1]),
        grid[2].div_ceil(tg[2]),
    ])
}

/// The three `MTLSize`s a mesh draw encodes, or `None` if it has no work.
///
/// A mesh draw carries a grid, an object threadgroup size and a mesh
/// threadgroup size, and Metal takes all three as `MTLSize` — a plain struct it
/// validates as strictly positive in every dimension. So a zero reaching
/// `drawMeshThreads:` or `drawMeshThreadgroups:` is not a small dispatch; it is
/// a validation assert on a debug layer and undefined driver behaviour without
/// one. There is no reading under which this device may pass one through, which
/// is why this returns `None` rather than clamping.
///
/// # Why `object_tg` is the one that may be zero
///
/// Metal requires `threadsPerObjectThreadgroup` to be supplied even when the
/// pipeline's `objectFunction` is nil, and there is no object stage to size in
/// that case. A guest that omits it leaves the field zero, and `MTLSize`'s own
/// convention for an unused dimension is 1 — so 1 is what the record means, not
/// a number invented to get past a check. Substituting it is reported by the
/// `object_tg_defaulted` flag rather than done silently, because the
/// substitution is only correct when there is no object stage: a pipeline that
/// *has* one and a record that sizes it at zero is a guest disagreeing with
/// itself, and this function cannot see the pipeline to tell the two apart.
///
/// # Why a triple's first component is not the triple
///
/// The check this replaces tested `grid[0]` and `mesh_tg[0]` and nothing else,
/// which reads as thorough and samples one third of each. It is the same trap
/// `AGENTS.md` records for a range guard that checks two endpoints and calls it
/// a span — the granularity the answer varies at is per-component, so the walk
/// has to be per-component.
#[must_use]
pub fn mesh_draw_dims(
    grid: [u32; 3],
    object_tg: [u32; 3],
    mesh_tg: [u32; 3],
) -> Option<MeshDrawDims> {
    if grid.iter().chain(&mesh_tg).any(|&d| d == 0) {
        return None;
    }
    let object_tg_defaulted = object_tg.contains(&0);
    Some(MeshDrawDims {
        grid,
        object_tg: object_tg.map(|d| if d == 0 { 1 } else { d }),
        mesh_tg,
        object_tg_defaulted,
    })
}

/// The dimensions a mesh draw encodes, after [`mesh_draw_dims`] has accepted it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshDrawDims {
    /// `threadsPerGrid` or `threadgroupsPerGrid`, per the record's opcode.
    pub grid: [u32; 3],
    /// `threadsPerObjectThreadgroup`, with any zero component read as 1.
    pub object_tg: [u32; 3],
    /// `threadsPerMeshThreadgroup`.
    pub mesh_tg: [u32; 3],
    /// Whether any `object_tg` component was zero and has been read as 1.
    ///
    /// Carried rather than discarded so the substitution stays countable. It is
    /// correct only for a pipeline with no object stage, and this type cannot
    /// see the pipeline — so a caller that can, can say so.
    pub object_tg_defaulted: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A zero anywhere in either triple means no dispatch, in both spellings.
    ///
    /// Exhaustive over which component is zero, because the refusal is what
    /// stands between a guest that asked for no threads and a kernel this
    /// device ran anyway. A predicate that checked only `grid[0]` — which is
    /// the shape the sibling mesh-draw path still has — passes a test that
    /// zeroes the first component and nothing else.
    #[test]
    fn a_zero_in_any_dimension_dispatches_nothing() {
        for grid_is_threads in [false, true] {
            for i in 0..6 {
                let mut grid = [4u32, 4, 4];
                let mut tg = [2u32, 2, 2];
                if i < 3 {
                    grid[i] = 0
                } else {
                    tg[i - 3] = 0
                }
                assert_eq!(
                    workgroup_counts(grid, tg, grid_is_threads),
                    None,
                    "grid={grid:?} tg={tg:?} threads={grid_is_threads}"
                );
            }
        }
    }

    /// `DispatchThreadgroups` passes the guest's count through untouched.
    #[test]
    fn a_threadgroup_count_is_not_divided() {
        assert_eq!(
            workgroup_counts([7, 3, 1], [8, 8, 1], false),
            Some([7, 3, 1])
        );
    }

    /// `DispatchThreads` rounds up, and never past what the guest asked for.
    ///
    /// The exact-multiple case is the one that would hide an off-by-one: 16
    /// threads in groups of 8 is two groups, not three. The partial case pins
    /// the rounding direction — Metal launches the group that covers the
    /// remainder, so a trailing thread is never dropped.
    #[test]
    fn a_thread_count_rounds_up_to_whole_threadgroups() {
        assert_eq!(
            workgroup_counts([16, 16, 1], [8, 8, 1], true),
            Some([2, 2, 1])
        );
        assert_eq!(
            workgroup_counts([17, 1, 1], [8, 1, 1], true),
            Some([3, 1, 1])
        );
        assert_eq!(
            workgroup_counts([1, 1, 1], [64, 64, 64], true),
            Some([1, 1, 1])
        );
        assert_eq!(
            workgroup_counts([u32::MAX, 1, 1], [1, 1, 1], true),
            Some([u32::MAX, 1, 1]),
            "the widest grid a u32 can carry divides without wrapping"
        );
    }

    /// A zero in any component of `grid` or `mesh_tg` refuses the draw.
    ///
    /// Exhaustive over all six, which is the whole point: the check this
    /// replaced tested component `[0]` of each and would pass a test that
    /// zeroed only that one. Every case here except two is a regression the old
    /// check let through to Metal as a zero `MTLSize`.
    #[test]
    fn a_mesh_draw_with_a_zero_extent_encodes_nothing() {
        for i in 0..6 {
            let mut grid = [4u32, 4, 4];
            let mut mesh_tg = [2u32, 2, 2];
            if i < 3 {
                grid[i] = 0
            } else {
                mesh_tg[i - 3] = 0
            }
            assert_eq!(
                mesh_draw_dims(grid, [1, 1, 1], mesh_tg),
                None,
                "grid={grid:?} mesh_tg={mesh_tg:?}"
            );
        }
    }

    /// A zero object threadgroup is read as 1, and the substitution is reported.
    ///
    /// The asymmetry with `grid` and `mesh_tg` is the contract: Metal wants the
    /// argument even with no object stage, so absent means 1 rather than
    /// invalid. `object_tg_defaulted` is what keeps that from being a silent
    /// fabrication — it is the only one of the three this device may supply.
    #[test]
    fn an_absent_object_threadgroup_is_read_as_one_and_says_so() {
        let all_zero = mesh_draw_dims([4, 4, 4], [0, 0, 0], [2, 2, 2]).expect("accepted");
        assert_eq!(all_zero.object_tg, [1, 1, 1]);
        assert!(all_zero.object_tg_defaulted);

        // A partially-sized object threadgroup: only the zero components move.
        let partial = mesh_draw_dims([4, 4, 4], [8, 0, 0], [2, 2, 2]).expect("accepted");
        assert_eq!(partial.object_tg, [8, 1, 1]);
        assert!(partial.object_tg_defaulted);

        let stated = mesh_draw_dims([4, 4, 4], [8, 2, 1], [2, 2, 2]).expect("accepted");
        assert_eq!(stated.object_tg, [8, 2, 1]);
        assert!(
            !stated.object_tg_defaulted,
            "nothing was substituted, so nothing may be reported as substituted"
        );
    }

    /// An accepted draw passes `grid` and `mesh_tg` through untouched.
    ///
    /// The counterpart to the clamp that `workgroup_counts` refuses to make:
    /// these two are the guest's own extents and this device may not round,
    /// clamp or default either of them.
    #[test]
    fn an_accepted_mesh_draw_alters_neither_extent() {
        let dims = mesh_draw_dims([7, 3, 1], [1, 1, 1], [32, 1, 1]).expect("accepted");
        assert_eq!(dims.grid, [7, 3, 1]);
        assert_eq!(dims.mesh_tg, [32, 1, 1]);
        assert!(!dims.object_tg_defaulted);
    }

    /// The accepted set is exactly the two declared ordinals.
    ///
    /// Worth pinning rather than assuming, because the predicate's whole job is
    /// to be *closed*: the substitution it guards is chosen for every value it
    /// rejects, so a predicate that accidentally accepted a third ordinal would
    /// pass that value through to a Metal enum conversion that has no arm for
    /// it. The sweep runs past both constants in both directions.
    #[test]
    fn only_the_two_declared_dispatch_types_are_accepted() {
        assert!(is_declared_dispatch_type(MTL_DISPATCH_TYPE_SERIAL));
        assert!(is_declared_dispatch_type(MTL_DISPATCH_TYPE_CONCURRENT));
        for raw in 2..=64u32 {
            assert!(
                !is_declared_dispatch_type(raw),
                "{raw} is not a declared MTLDispatchType"
            );
        }
        assert!(!is_declared_dispatch_type(u32::MAX));
    }
}
