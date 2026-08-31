//! Linear resources resolved once per reference and held.
//!
//! # The shape this follows
//!
//! Apple's host resolves a guest object reference to a host buffer **once**,
//! when the object is created, and stores it on the task under that reference.
//! Its render decoder then reads a `{u32 reference, u64 offset}` record per
//! bound slot, asks the task for the buffer by reference, and hands Metal the
//! buffer and the offset. No address translation happens on the draw path at
//! all — the page-run computation on that side is reachable only from the
//! map/unmap handlers, never from a decoder.
//!
//! This device resolved the bind instead: every bound buffer of every draw
//! walked the task page table over the bound span, coalesced the GPA-contiguous
//! stretches, and asked the host to alias each one. That is the same answer
//! every time until the guest changes a mapping, and the guest changes mappings
//! about four orders of magnitude less often than it draws.
//!
//! So this holds the resolution and the draw path looks it up.
//!
//! Linear textures use the same packed allocation. Their level offset and row
//! pitch become image coordinates over it, while buffer records carry their
//! offset to the bind. Both are views of one resource-owned mapping; neither
//! needs a second task-page walk after that mapping is retained.
//!
//! # What a held resolution is, and is not
//!
//! It is an **address** resolution: which host spans back this reference's
//! bytes right now. It is not the bytes. The runs point into this process's
//! import of guest RAM and the GPU reads them when the command buffer executes,
//! so a guest CPU write to those pages is picked up with nothing invalidated —
//! the same property the walking rail had, and the reason `CmdInvalidateResources`
//! and the exec resource table's validity quad do not appear anywhere here.
//! Content invalidation is not this module's business.
//!
//! Only an **address** change matters, and the guest announces every one of
//! them:
//!
//! * `CmdMapMemory2` / `CmdUnmapMemory` — the guest mutates the task page table
//!   and then notifies, carrying the exact `(task, gva, length)` that moved.
//!   Retired by range.
//! * `CmdReplacePhysical` — a GPA behind a GVA changed.
//! * `CmdSetObjectList` / `CmdDeleteObject` — a reference now names something
//!   else, or nothing.
//! * `CmdDefineTask2` / `CmdDeleteTask` — the page table root changed or the
//!   task is gone.
//!
//! `CmdReplacePhysical` and `CmdDeleteObject` carry the task-local resource
//! reference and retire that reference. `CmdSetObjectList`, `CmdDefineTask2`
//! and `CmdDeleteTask` replace task-wide naming state and retire the whole task.
//!
//! # Why the fallback key carries the offset
//!
//! Apple keys purely by reference, because their buffer covers the whole
//! allocation and the offset rides to Metal beside it. An exact-window fallback
//! resolution here covers `[gva + offset, gva + size)` — the span the bind
//! actually asked for — so two binds of one reference at different offsets are
//! two resolutions.
//!
//! The packed-alias rail resolves the whole allocation once and supplies the
//! offset beside that retained source. It bypasses this map entirely, which is
//! what keeps resource-shaped state resource-shaped. It is an optional answer
//! beside the narrower fallback: if an unmapped tail prevents the whole
//! allocation from being reconstructed, the exact offset/cap window still
//! resolves here and gathers.
//!
//! The distinction is measured rather than aesthetic. Before packed resources
//! bypassed this map, one driven x86 window-drag run reached 33,828 fallback
//! entries over 48 `(task, reference)` pairs, with one reference accounting for
//! 3,080 offsets. The same workload with buffer-plus-offset binding held zero
//! fallback entries while the packed resources remained live. The offset is
//! therefore required for correctness only on the exact-window fallback; it is
//! not a sound identity for the normal resource registry.
//!
//! # Why the key also carries the shader's extent cap
//!
//! The three fields above all describe the *bind*. The fourth describes the
//! **shader**: how far reflection proved this draw's shader can read into the
//! buffer, which is what lets the resolution cover less than the rest of the
//! allocation. A resolution walked under a narrow cap covers fewer bytes than a
//! shader with a wider one needs, and serving it across would hand the GPU a
//! short buffer — wrong pixels, no error. So the cap is part of the identity of
//! the resolution rather than a property of it, and [`Key`] says the rest.
//!
//! # No capacity
//!
//! There is no cap and no eviction. The fallback population is one entry per
//! live `(task, reference, offset, extent cap)` whose whole resource could not
//! be reconstructed; the normal population is one packed entry per
//! `(task, reference)`. Every entry leaves through one of the retirement rules
//! above or through [`BoundBuffers::clear`] at device reset. A capacity here
//! would be a second, invisible reason for a resolution to disappear, and the
//! miss it caused would read as a mapping change that never happened.
//!
//! The extent cap widens that population only where two shaders declare
//! different extents over one bind; the retirement rules are all keyed on task
//! and reference and so are indifferent to it.

use std::collections::HashMap;
use std::sync::Arc;

use crate::backend::vulkan::engine::GuestRun;
use crate::runtime::guest_ram_map::GuestWindowRun;

/// One task buffer reconstructed as a stable, contiguous host allocation.
///
/// The allocation follows the buffer's task-virtual byte order even when its
/// guest-physical pages are scattered. Every offset bind can therefore slice
/// this one checked import instead of gathering the same pages into scratch.
#[derive(Clone, Debug)]
pub struct PackedBuffer {
    pub gva: u64,
    pub size: u64,
    /// Offset of `gva` inside the page-aligned host allocation.
    pub head: u64,
    pub import: Arc<crate::runtime::guest_ram::GuestRamImport>,
    /// Physical page list behind the packed alias, retained so resource views
    /// can build witnesses without walking the task page table again.
    pub gpas: Arc<Vec<u64>>,
    /// Persistent whole-buffer sources shared by every offset bind.
    pub runs: Arc<Vec<GuestRun>>,
    pub pages: Arc<Vec<GuestWindowRun>>,
}

#[derive(Clone, Debug)]
pub enum PackedBufferResolution {
    Available(PackedBuffer),
    /// The whole declared allocation could not be mapped. Narrow, individually
    /// walkable binds remain valid and use the existing gather rail.
    Unavailable {
        gva: u64,
        size: u64,
    },
}

impl PackedBufferResolution {
    fn overlaps(&self, gva: u64, len: u64) -> bool {
        let (base, span) = match self {
            Self::Available(buffer) => (buffer.gva, buffer.size),
            Self::Unavailable { gva, size } => (*gva, *size),
        };
        if len == 0 || span == 0 {
            return false;
        }
        base < gva.saturating_add(len) && gva < base.saturating_add(span)
    }
}

/// A resolved bind: where this reference's bytes live, as the engine binds them.
///
/// Both lists are `Arc`ed by the producer already, so a lookup hands the draw
/// path the same allocation the walk built rather than a copy of it.
#[derive(Clone, Debug)]
pub struct BoundBuffer {
    /// Guest VA the resolution starts at (the backing's `gva + offset`).
    pub gva: u64,
    /// Byte length the runs cover, and the bind's `total_len`.
    pub span: u64,
    /// First byte of this bind inside `runs` / `pages`.
    ///
    /// Exact-window resolutions use zero. A packed resource shares its one
    /// whole-buffer source and carries the guest's bind offset here.
    pub source_offset: u64,
    /// Host-pointer spans the CPU gather walks.
    pub runs: Arc<Vec<GuestRun>>,
    /// The same bytes as bounded references into this process's import, when
    /// the host can import guest RAM at all. `None` keeps the caller on the
    /// gathering arm exactly as a fresh resolution would.
    pub pages: Option<Arc<Vec<GuestWindowRun>>>,
}

impl BoundBuffer {
    /// Whether this resolution's bytes overlap `[gva, gva + len)`.
    ///
    /// Half-open on both sides. A zero-length range overlaps nothing, which is
    /// what a map notify carrying no length means.
    fn overlaps(&self, gva: u64, len: u64) -> bool {
        if len == 0 || self.span == 0 {
            return false;
        }
        let a_end = self.gva.saturating_add(self.span);
        let b_end = gva.saturating_add(len);
        self.gva < b_end && gva < a_end
    }
}

/// `(task, reference, offset, extent cap)` — see the module doc on why the
/// offset is here.
///
/// The cap is in the key because it is not a property of the bind: it is what
/// the *shader on this draw* proved about how far it can read
/// ([`crate::runtime::spirv_bind::reflected_buffer_extent`]). Two shaders may
/// bind one allocation at one offset and declare different extents, and a
/// resolution walked for the narrower one covers fewer bytes than the wider one
/// needs. Keyed without the cap, that resolution would be handed to the wider
/// shader as a hit and the GPU would read a short buffer — no error, wrong
/// pixels. Keyed with it, the two coexist and each pays its own walk.
///
/// `None` — the uncapped whole-allocation resolution — is a distinct key from
/// any capped one, which is what keeps the pre-existing behaviour reachable and
/// unchanged for every bind reflection does not bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Key {
    task: u32,
    buffer_ref: u32,
    offset: u64,
    cap: Option<u64>,
}

/// What [`BoundBuffers::shape`] measures. Levels, not per-interval counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RegistryShape {
    /// Held resolutions.
    pub entries: usize,
    /// Distinct `(task, reference)` pairs behind them — what the registry would
    /// hold if it were keyed the way Apple's is.
    pub pairs: usize,
    /// Pairs held at more than one offset. Zero means the offset in the key
    /// never separates two live entries.
    pub multi_offset_pairs: usize,
    /// The most offsets any one pair is held at.
    pub max_offsets: u32,
}

/// Report the registry's shape once per census interval, on the same one-second
/// cadence as `store_routes` so the two line up row for row.
///
/// Read against the `bb_retire_*` routes: those say how many resolutions a
/// retirement dropped, this says what the survivors look like. A miss is either
/// a retired key or a key never seen, and the two together are what decide
/// whether the 12.8x more fresh resolutions on an importing host are churn in
/// the retirement rules or churn in the keys.
pub fn note_registry_levels(state: &crate::model::DeviceState) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static LAST_MS: AtomicU64 = AtomicU64::new(0);
    static PEAK_ENTRIES: AtomicU64 = AtomicU64::new(0);

    let shape = state.bound_buffers.shape();
    let peak = PEAK_ENTRIES
        .fetch_max(shape.entries as u64, Ordering::Relaxed)
        .max(shape.entries as u64);

    let now = crate::observe::elapsed_ms() as u64;
    let last = LAST_MS.load(Ordering::Relaxed);
    if now.saturating_sub(last) < 1000 {
        return;
    }
    // Losing the race only costs a skipped interval, never a double line.
    if LAST_MS
        .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
        .is_err()
    {
        return;
    }
    crate::observe::off(format!(
        "bound_buffers (levels, not per-interval) entries={} peak={} pairs={} \
         multi_offset_pairs={} max_offsets={}",
        shape.entries, peak, shape.pairs, shape.multi_offset_pairs, shape.max_offsets
    ));
}

/// Every held bind resolution on this device.
#[derive(Default, Debug)]
pub struct BoundBuffers {
    held: HashMap<Key, BoundBuffer>,
    packed: HashMap<(u32, u32), PackedBufferResolution>,
}

impl BoundBuffers {
    /// The resolution for this bind, if one is held.
    pub fn get(
        &self,
        task_id: u32,
        buffer_ref: u32,
        offset: u64,
        cap: Option<u64>,
    ) -> Option<&BoundBuffer> {
        self.held.get(&Key {
            task: task_id,
            buffer_ref,
            offset,
            cap,
        })
    }

    /// Hold a freshly walked resolution.
    pub fn insert(
        &mut self,
        task_id: u32,
        buffer_ref: u32,
        offset: u64,
        cap: Option<u64>,
        bound: BoundBuffer,
    ) {
        self.held.insert(
            Key {
                task: task_id,
                buffer_ref,
                offset,
                cap,
            },
            bound,
        );
    }

    pub fn packed(&self, task_id: u32, buffer_ref: u32) -> Option<&PackedBufferResolution> {
        self.packed.get(&(task_id, buffer_ref))
    }

    /// Borrow the retained allocation when it still describes exactly this
    /// resource construction.
    ///
    /// The geometry check matters on the narrow window between a descriptor
    /// changing and its retirement packet being consumed: returning the old
    /// allocation there would make a warm lookup observably different from a
    /// fresh resolution. Returning a reference is equally deliberate. A warm
    /// encoder bind borrows its resource object and retains only the execution
    /// payload it hands onward; it does not acquire all of the construction
    /// state merely to inspect it.
    pub fn packed_available(
        &self,
        task_id: u32,
        resource_ref: u32,
        gva: u64,
        size: u64,
    ) -> Option<&PackedBuffer> {
        match self.packed(task_id, resource_ref)? {
            PackedBufferResolution::Available(packed)
                if packed.gva == gva && packed.size == size =>
            {
                Some(packed)
            }
            PackedBufferResolution::Available(_) | PackedBufferResolution::Unavailable { .. } => {
                None
            }
        }
    }

    pub fn insert_packed(&mut self, task_id: u32, buffer_ref: u32, packed: PackedBufferResolution) {
        self.packed.insert((task_id, buffer_ref), packed);
    }

    /// Drop everything held for one task.
    ///
    /// The answer for a page-table root change, a new object list, or a deleted
    /// task: in each case every reference may now name different bytes.
    pub fn retire_task(&mut self, task_id: u32) -> usize {
        let before = self.held.len();
        self.held.retain(|k, _| k.task != task_id);
        self.packed.retain(|(task, _), _| *task != task_id);
        before - self.held.len()
    }

    /// Drop everything held for one reference, at every offset.
    ///
    /// The `CmdDeleteObject` answer. That packet names the reference —
    /// `delete_object(task_id, ref_)` — and the rest of the device already
    /// scopes its response to it: `objects`, `invalidate_object_host_copies`
    /// and `texture_to_mapping` are all keyed `(task, ref)`. This registry
    /// retiring the whole task was the outlier, and a measured expensive one:
    /// one driven boot dropped 54 109 resolutions there, 95% of every bind miss
    /// on the device.
    ///
    /// # Why the narrower rule is sound
    ///
    /// A held resolution for reference `R` is built from three things: the
    /// object-list entry at index `R`, the descriptor that entry names, and the
    /// task page-table walk over the span the descriptor declares. Deleting
    /// object `X` where `X != R` touches none of them — the list is indexed by
    /// reference so entries do not shift, `X`'s descriptor is at its own
    /// address, and no page table changes. So no resolution but `R`'s own can
    /// be stale because of this packet.
    ///
    /// Two references aliasing one allocation are safe for the same reason:
    /// deleting one does not free the pages or move the other's descriptor. If
    /// the guest then reuses that address the announcement is a different
    /// packet — `CmdMapMemory2`, `CmdUnmapMemory` or `CmdReplacePhysical` — and
    /// those rules still retire by range or by task.
    pub fn retire_ref(&mut self, task_id: u32, buffer_ref: u32) -> usize {
        let before = self.held.len();
        self.held
            .retain(|k, _| k.task != task_id || k.buffer_ref != buffer_ref);
        self.packed.remove(&(task_id, buffer_ref));
        before - self.held.len()
    }

    /// Drop everything held for `task_id` whose bytes overlap `[gva, gva+len)`.
    ///
    /// The map/unmap answer, which carries the exact range that moved.
    pub fn retire_range(&mut self, task_id: u32, gva: u64, len: u64) -> usize {
        let before = self.held.len();
        self.held
            .retain(|k, b| k.task != task_id || !b.overlaps(gva, len));
        self.packed
            .retain(|(task, _), b| *task != task_id || !b.overlaps(gva, len));
        before - self.held.len()
    }

    /// Drop everything. Device reset, where no guest state survives.
    pub fn clear(&mut self) {
        self.held.clear();
        self.packed.clear();
    }

    /// How many resolutions are held, for the census.
    pub fn len(&self) -> usize {
        self.held.len()
    }

    /// The registry's shape: entries, the distinct `(task, reference)` pairs
    /// behind them, how many of those pairs are held at more than one offset,
    /// and the most offsets any single pair carries.
    ///
    /// This is the instrument for the question the module doc states and does
    /// not answer. Apple keys by reference alone; this keys by
    /// `(task, reference, offset)`, on the belief that a reference is bound at
    /// one offset and the two keys therefore describe the same registry. That
    /// belief has never been counted. `pairs == entries` says it holds and the
    /// extra field is inert; `pairs < entries` says references really are bound
    /// at several offsets, each paying its own walk, and the narrower key is
    /// costing exactly `entries - pairs` resolutions.
    ///
    /// Walked once per census interval rather than tracked incrementally: a
    /// second index would have to be maintained by every retirement rule, which
    /// is a correctness surface bought for a measurement, and the population is
    /// the guest's live working set rather than anything unbounded.
    pub fn shape(&self) -> RegistryShape {
        let mut per_pair: HashMap<(u32, u32), u32> = HashMap::new();
        for k in self.held.keys() {
            *per_pair.entry((k.task, k.buffer_ref)).or_default() += 1;
        }
        RegistryShape {
            entries: self.held.len(),
            pairs: per_pair.len(),
            multi_offset_pairs: per_pair.values().filter(|n| **n > 1).count(),
            max_offsets: per_pair.values().copied().max().unwrap_or(0),
        }
    }

    /// Whether nothing is held.
    pub fn is_empty(&self) -> bool {
        self.held.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bound(gva: u64, span: u64) -> BoundBuffer {
        BoundBuffer {
            gva,
            span,
            source_offset: 0,
            runs: Arc::new(Vec::new()),
            pages: None,
        }
    }

    /// The lookup is keyed by all three of task, reference and offset, so no
    /// two binds can collide onto one resolution.
    #[test]
    fn a_resolution_is_found_only_by_its_own_key() {
        let mut b = BoundBuffers::default();
        b.insert(7, 3, 0, None, bound(0x1000, 0x2000));
        assert!(b.get(7, 3, 0, None).is_some());
        assert!(b.get(7, 3, 0x100, None).is_none(), "a different offset");
        assert!(b.get(7, 4, 0, None).is_none(), "a different reference");
        assert!(b.get(8, 3, 0, None).is_none(), "a different task");
    }

    /// A resolution walked under one shader's extent cap is never served to a
    /// shader that proved a different one.
    ///
    /// This is the corruption guard for the narrowing rail, and it fails in the
    /// direction that has no other alarm: a 64-byte resolution handed to a
    /// shader entitled to 4096 does not error, it reads whatever the GPU finds
    /// past the end of a short buffer and draws it. The uncapped entry is a
    /// fourth distinct key rather than a wildcard, so a bind reflection could
    /// not bound never picks up a neighbour's narrowing either.
    #[test]
    fn a_resolution_is_never_served_across_a_different_extent_cap() {
        let mut b = BoundBuffers::default();
        b.insert(1, 1, 0, Some(64), bound(0x1000, 64));

        assert!(b.get(1, 1, 0, Some(64)).is_some(), "its own cap");
        assert!(
            b.get(1, 1, 0, Some(4096)).is_none(),
            "a shader entitled to more must not get the 64-byte walk"
        );
        assert!(
            b.get(1, 1, 0, None).is_none(),
            "an unbounded bind must not get a capped walk"
        );

        // The three coexist rather than evicting one another, so neither shader
        // re-walks on every draw because the other one ran in between.
        b.insert(1, 1, 0, Some(4096), bound(0x1000, 4096));
        b.insert(1, 1, 0, None, bound(0x1000, 0x10000));
        assert_eq!(b.len(), 3);
        assert_eq!(b.get(1, 1, 0, Some(64)).map(|r| r.span), Some(64));
        assert_eq!(b.get(1, 1, 0, Some(4096)).map(|r| r.span), Some(4096));
        assert_eq!(b.get(1, 1, 0, None).map(|r| r.span), Some(0x10000));

        // A retirement is keyed on task and reference, so it takes all three.
        assert_eq!(b.retire_ref(1, 1), 3);
        assert!(b.is_empty());
    }

    /// A map/unmap notify retires exactly the resolutions whose bytes moved,
    /// and leaves the neighbours that did not.
    #[test]
    fn a_range_retire_takes_the_overlapping_resolutions_only() {
        let mut b = BoundBuffers::default();
        b.insert(1, 1, 0, None, bound(0x1000, 0x1000)); // [0x1000,0x2000)
        b.insert(1, 2, 0, None, bound(0x2000, 0x1000)); // [0x2000,0x3000)
        b.insert(1, 3, 0, None, bound(0x9000, 0x1000)); // far away
        assert_eq!(b.retire_range(1, 0x1800, 0x1000), 2, "spans the first two");
        assert!(b.get(1, 3, 0, None).is_some(), "the far one survives");
        assert_eq!(b.len(), 1);
    }

    /// Whole-buffer alias answers share the reference lifecycle even when no
    /// offset resolution has been materialized yet.
    #[test]
    fn packed_alias_answers_retire_with_their_reference_and_mapping() {
        let mut b = BoundBuffers::default();
        b.insert_packed(
            1,
            7,
            PackedBufferResolution::Unavailable {
                gva: 0x4000,
                size: 0x3000,
            },
        );
        b.insert_packed(
            1,
            8,
            PackedBufferResolution::Unavailable {
                gva: 0x9000,
                size: 0x1000,
            },
        );
        assert!(b.packed(1, 7).is_some());
        assert_eq!(b.retire_range(1, 0x5000, 0x1000), 0);
        assert!(b.packed(1, 7).is_none(), "overlapping alias answer");
        assert!(b.packed(1, 8).is_some(), "unrelated alias answer");
        assert_eq!(b.retire_ref(1, 8), 0);
        assert!(b.packed(1, 8).is_none());
    }

    #[test]
    fn one_packed_import_serves_every_offset_of_a_reference() {
        let import = Arc::new(
            crate::runtime::guest_ram::GuestRamImport::new_host_allocation(
                0x7f00_0000_0000,
                0x8000,
                0x1000,
            )
            .expect("aligned allocation"),
        );
        let id = import.id();
        let mut b = BoundBuffers::default();
        b.insert_packed(
            3,
            9,
            PackedBufferResolution::Available(PackedBuffer {
                gva: 0x10800,
                size: 0x7000,
                head: 0x800,
                import: Arc::clone(&import),
                gpas: Arc::new(Vec::new()),
                runs: Arc::new(Vec::new()),
                pages: Arc::new(Vec::new()),
            }),
        );
        let PackedBufferResolution::Available(packed) = b.packed(3, 9).unwrap() else {
            panic!("available above")
        };
        for (offset, span) in [(0, 0x1000), (0x1800, 0x2000), (0x5000, 0x800)] {
            let slice = packed
                .import
                .slice(packed.head + offset, span)
                .expect("each bind lies in the one allocation");
            assert_eq!(slice.import(), id);
        }
        let owners = Arc::strong_count(&import);
        assert!(b.packed_available(3, 9, 0x10800, 0x7000).is_some());
        assert_eq!(
            Arc::strong_count(&import),
            owners,
            "a warm lookup borrows the resource rather than acquiring it"
        );
        assert!(
            b.packed_available(3, 9, 0x11800, 0x7000).is_none(),
            "a changed base cannot reuse the prior construction"
        );
        assert!(
            b.packed_available(3, 9, 0x10800, 0x6000).is_none(),
            "a changed allocation size cannot reuse the prior construction"
        );
        assert!(
            b.packed_available(4, 9, 0x10800, 0x7000).is_none(),
            "the same reference in another task is a different resource"
        );
    }

    /// A range retire is scoped to its task: the same GVA under another task is
    /// a different address space and must not be touched.
    #[test]
    fn a_range_retire_does_not_cross_tasks() {
        let mut b = BoundBuffers::default();
        b.insert(1, 1, 0, None, bound(0x1000, 0x1000));
        b.insert(2, 1, 0, None, bound(0x1000, 0x1000));
        assert_eq!(b.retire_range(1, 0x1000, 0x1000), 1);
        assert!(b.get(2, 1, 0, None).is_some());
    }

    /// A zero-length notify names no bytes and must retire nothing — otherwise
    /// a malformed packet would silently drop every resolution it touched.
    #[test]
    fn a_zero_length_range_retires_nothing() {
        let mut b = BoundBuffers::default();
        b.insert(1, 1, 0, None, bound(0x1000, 0x1000));
        assert_eq!(b.retire_range(1, 0x1000, 0), 0);
        assert_eq!(b.len(), 1);
    }

    /// Ranges that merely touch at an endpoint do not overlap, so an unmap of
    /// the page after a resolution does not retire it.
    #[test]
    fn abutting_ranges_do_not_overlap() {
        let mut b = BoundBuffers::default();
        b.insert(1, 1, 0, None, bound(0x1000, 0x1000)); // [0x1000,0x2000)
        assert_eq!(b.retire_range(1, 0x2000, 0x1000), 0, "starts where it ends");
        assert_eq!(b.retire_range(1, 0x0000, 0x1000), 0, "ends where it starts");
        assert_eq!(b.len(), 1);
    }

    /// A reference retire takes that reference at **every** offset, and nothing
    /// else.
    ///
    /// Both halves matter and they fail in opposite directions. Leaving one of
    /// the deleted reference's offsets behind serves bytes from an object the
    /// guest destroyed; taking a neighbour's is the whole-task rule this
    /// replaced, which is merely expensive. The offsets are the ones a driven
    /// boot actually produces — a single reference is held at 233 of them.
    #[test]
    fn a_reference_retire_takes_every_offset_of_that_reference_only() {
        let mut b = BoundBuffers::default();
        for off in [0u64, 0x400, 0x1000, 0x9000] {
            b.insert(1, 7, off, None, bound(0x1000 + off, 0x400));
        }
        // A neighbouring reference on the same task, and the same reference
        // under another task: neither is named by this packet.
        b.insert(1, 8, 0, None, bound(0x8000, 0x400));
        b.insert(2, 7, 0, None, bound(0x1000, 0x400));
        assert_eq!(b.len(), 6);

        assert_eq!(b.retire_ref(1, 7), 4, "every offset of reference 7");
        assert!(b.get(1, 7, 0, None).is_none());
        assert!(b.get(1, 7, 0x9000, None).is_none());
        assert!(
            b.get(1, 8, 0, None).is_some(),
            "a sibling reference survives"
        );
        assert!(b.get(2, 7, 0, None).is_some(), "another task's survives");
        assert_eq!(b.len(), 2);

        // A reference nothing is held for is not an error, and takes nothing.
        assert_eq!(b.retire_ref(1, 7), 0);
        assert_eq!(b.retire_ref(1, 99), 0);
        assert_eq!(b.len(), 2);
    }

    /// The shape separates entries from the `(task, reference)` pairs behind
    /// them, which is the whole reason it exists: `pairs == entries` says the
    /// offset in the key never distinguishes two live entries, and anything
    /// less counts the resolutions the narrower key is paying for.
    #[test]
    fn the_shape_counts_pairs_apart_from_entries() {
        let mut b = BoundBuffers::default();
        assert_eq!(b.shape(), RegistryShape::default(), "empty");

        // One reference at one offset each: the two keys would agree.
        b.insert(1, 1, 0, None, bound(0x1000, 0x1000));
        b.insert(1, 2, 0, None, bound(0x2000, 0x1000));
        let s = b.shape();
        assert_eq!((s.entries, s.pairs), (2, 2));
        assert_eq!((s.multi_offset_pairs, s.max_offsets), (0, 1));

        // The same reference at a second offset is a second entry and not a
        // second pair — exactly the divergence from Apple's key.
        b.insert(1, 1, 0x400, None, bound(0x1400, 0x400));
        let s = b.shape();
        assert_eq!((s.entries, s.pairs), (3, 2), "one pair now holds two");
        assert_eq!((s.multi_offset_pairs, s.max_offsets), (1, 2));

        // The same reference under another task is a different pair, because a
        // GVA has no meaning apart from the table it resolves against.
        b.insert(2, 1, 0, None, bound(0x1000, 0x1000));
        let s = b.shape();
        assert_eq!((s.entries, s.pairs), (4, 3));
        assert_eq!(s.multi_offset_pairs, 1);
    }

    /// A task retire takes that task's resolutions whatever their addresses,
    /// and leaves every other task alone.
    #[test]
    fn a_task_retire_takes_the_whole_task() {
        let mut b = BoundBuffers::default();
        b.insert(1, 1, 0, None, bound(0x1000, 0x1000));
        b.insert(1, 2, 0x40, None, bound(0x8000, 0x1000));
        b.insert(2, 1, 0, None, bound(0x1000, 0x1000));
        assert_eq!(b.retire_task(1), 2);
        assert_eq!(b.len(), 1);
        assert!(b.get(2, 1, 0, None).is_some());
    }
}
