//! Semantic compute and mesh dispatch geometry decoded from guest records.

pub const MTL_DISPATCH_TYPE_SERIAL: u32 = 0;
pub const MTL_DISPATCH_TYPE_CONCURRENT: u32 = 1;

#[must_use]
pub fn is_declared_dispatch_type(raw: u32) -> bool {
    matches!(raw, MTL_DISPATCH_TYPE_SERIAL | MTL_DISPATCH_TYPE_CONCURRENT)
}

/// What a Metal dispatch becomes once it is expressed in whole workgroups, and
/// the thread grid the translated entry point culls against.
///
/// `dispatchThreads:threadsPerThreadgroup:` names a **thread** count and Metal
/// launches exactly that many, however badly the count divides the threadgroup.
/// `vkCmdDispatch` names whole workgroups and has no partial one, so a grid
/// that does not divide its threadgroup can only be rounded up. The threads in
/// the excess do run, with a `thread_position_in_grid` outside the grid the
/// guest asked for, and every address such a thread computes is a store the
/// guest's kernel was written on the promise it would never make.
///
/// Nothing on the host side can suppress an invocation, so the cull is the
/// shader's: the translated entry point compares its invocation id against a
/// thread grid pushed by this device and returns early past it. That makes the
/// grid a value the dispatch *must* carry, not a diagnostic, which is why this
/// type refuses to hand out [`Self::counts`] without [`Self::threads_per_grid`]
/// beside it — a bare `[u32; 3]` let a caller dispatch the rounded-up counts
/// and never learn what the cull needed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkgroupPlan {
    /// Workgroup counts for `vkCmdDispatch`.
    pub counts: [u32; 3],
    /// The exact Metal thread grid, in threads, per axis.
    ///
    /// This is what a translated entry point compares its invocation id
    /// against to cull the surplus, so it is carried beside the counts rather
    /// than left for a consumer to reconstruct: a `dispatchThreads` grid is the
    /// guest's own thread count, while a `dispatchThreadgroups` grid is
    /// `counts[i] * group[i]` — every launched thread, nothing culled. Getting
    /// that second case wrong culls threads the guest asked for, which is the
    /// same class of loss as running ones it did not.
    pub threads_per_grid: [u32; 3],
}

impl Default for WorkgroupPlan {
    /// A plan that dispatches nothing.
    ///
    /// Deliberately not derived, and deliberately all zero: a zero in any axis
    /// is already the backend's named refusal, so a request that reaches
    /// execution still carrying this default is rejected rather than run at
    /// some invented size. There is no meaningful "default grid" — a dispatch
    /// either names one or is not a dispatch.
    fn default() -> Self {
        Self {
            counts: [0; 3],
            threads_per_grid: [0; 3],
        }
    }
}

#[must_use]
pub fn workgroup_counts(
    grid: [u32; 3],
    group: [u32; 3],
    grid_is_threads: bool,
) -> Option<WorkgroupPlan> {
    if grid.iter().chain(&group).any(|&dimension| dimension == 0) {
        return None;
    }
    if !grid_is_threads {
        return Some(WorkgroupPlan {
            counts: grid,
            // Whole workgroups: every thread of every group is inside the
            // grid the guest named, so the thread count is the product.
            threads_per_grid: [
                grid[0].saturating_mul(group[0]),
                grid[1].saturating_mul(group[1]),
                grid[2].saturating_mul(group[2]),
            ],
        });
    }
    let counts = [
        grid[0].div_ceil(group[0]),
        grid[1].div_ceil(group[1]),
        grid[2].div_ceil(group[2]),
    ];
    Some(WorkgroupPlan {
        counts,
        threads_per_grid: grid,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshDrawDims {
    pub grid: [u32; 3],
    pub object_tg: [u32; 3],
    pub mesh_tg: [u32; 3],
    pub object_tg_defaulted: bool,
}

#[must_use]
pub fn mesh_draw_dims(
    grid: [u32; 3],
    object_tg: [u32; 3],
    mesh_tg: [u32; 3],
) -> Option<MeshDrawDims> {
    if grid.iter().chain(&mesh_tg).any(|&dimension| dimension == 0) {
        return None;
    }
    let object_tg_defaulted = object_tg.contains(&0);
    Some(MeshDrawDims {
        grid,
        object_tg: object_tg.map(|dimension| if dimension == 0 { 1 } else { dimension }),
        mesh_tg,
        object_tg_defaulted,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_types_and_workgroups_are_total() {
        assert!(is_declared_dispatch_type(0));
        assert!(is_declared_dispatch_type(1));
        assert!(!(2..=64).any(is_declared_dispatch_type));
        assert_eq!(
            workgroup_counts([17, 1, 1], [8, 1, 1], true),
            Some(WorkgroupPlan {
                counts: [3, 1, 1],
                threads_per_grid: [17, 1, 1],
            })
        );
        // A threadgroup-count dispatch names workgroups already, so there is
        // nothing to round and nothing to overrun however the numbers sit.
        assert_eq!(
            workgroup_counts([7, 3, 1], [8, 8, 1], false),
            Some(WorkgroupPlan {
                counts: [7, 3, 1],
                threads_per_grid: [56, 24, 1],
            })
        );
    }

    /// The rounded-up counts cover the grid and overshoot it by less than one
    /// group per axis — which is the amount the translated entry point's guard
    /// culls. These are the shapes the conformance battery measures on
    /// hardware; a quotient one group short loses guest work outright and one
    /// group long doubles what the guard has to throw away.
    #[test]
    fn the_counts_cover_the_grid_and_overshoot_by_less_than_a_group() {
        for (grid, group, counts) in [
            ([218u32, 16, 1], [8u32, 8, 1], [28u32, 2, 1]),
            ([57, 9, 1], [8, 8, 1], [8, 2, 1]),
            // A grid that divides in every axis is covered exactly.
            ([64, 16, 1], [8, 8, 1], [8, 2, 1]),
        ] {
            let plan = workgroup_counts(grid, group, true).expect("valid grid");
            assert_eq!(plan.counts, counts);
            assert_eq!(plan.threads_per_grid, grid);
            for axis in 0..3 {
                let launched = plan.counts[axis] * group[axis];
                assert!(launched >= grid[axis], "axis {axis}: grid not covered");
                assert!(
                    launched - grid[axis] < group[axis],
                    "axis {axis}: a whole group past the grid is a group too many"
                );
            }
        }
    }

    #[test]
    fn zero_in_any_required_dimension_means_no_dispatch() {
        for index in 0..6 {
            let mut grid = [4, 4, 4];
            let mut group = [2, 2, 2];
            if index < 3 {
                grid[index] = 0;
            } else {
                group[index - 3] = 0;
            }
            assert_eq!(workgroup_counts(grid, group, true), None);
        }
    }

    /// A whole-workgroup dispatch still needs a thread grid, because the guard
    /// in the translated entry point is unconditional: what makes it cull
    /// nothing is the grid covering every launched thread. A plan that reported
    /// the guest's *group* counts here would cull all but the first thread of
    /// each group in every axis.
    #[test]
    fn a_workgroup_dispatch_reports_every_thread_it_launches() {
        let plan = workgroup_counts([7, 3, 2], [8, 8, 4], false).expect("valid grid");
        assert_eq!(plan.counts, [7, 3, 2]);
        assert_eq!(plan.threads_per_grid, [56, 24, 8]);
    }

    /// And a thread-count dispatch reports the guest's own count, which is the
    /// grid the guard compares against.
    #[test]
    fn a_thread_dispatch_reports_the_grid_the_guest_named() {
        for (grid, group) in [([218u32, 16, 1], [8u32, 8, 1]), ([1, 1, 1], [64, 1, 1])] {
            let plan = workgroup_counts(grid, group, true).expect("valid grid");
            assert_eq!(plan.threads_per_grid, grid);
        }
    }

    #[test]
    fn mesh_defaults_only_the_optional_object_group() {
        let dimensions =
            mesh_draw_dims([7, 3, 1], [8, 0, 0], [32, 1, 1]).expect("valid mesh dimensions");
        assert_eq!(dimensions.object_tg, [8, 1, 1]);
        assert!(dimensions.object_tg_defaulted);
        for index in 0..6 {
            let mut grid = [4, 4, 4];
            let mut mesh = [2, 2, 2];
            if index < 3 {
                grid[index] = 0;
            } else {
                mesh[index - 3] = 0;
            }
            assert_eq!(mesh_draw_dims(grid, [1, 1, 1], mesh), None);
        }
    }
}
