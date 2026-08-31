//! Resource-validity ownership for render targets.
//!
//! A render Store preserves pixels in the host attachment. It does not imply a
//! host-to-guest transfer. The guest makes that transfer observable by naming
//! the resource in `CmdSynchronizeResources`, or this device needs the guest
//! bytes itself for a fallback reader. Until then, [`PendingWritebacks`] records
//! that the engine image is authoritative and repeated Stores into the resource
//! replace one another without touching guest RAM.
//!
//! # A resource owns its transfer backing
//!
//! Type-11 debts carry a mapping id, geometry, and map generation. GVA debts
//! carry the task-local texture reference, GVA declaration, geometry, format,
//! and resource generation. The live GVA resource separately retains the
//! ordered physical pages of its transfer backing. Ordinary task unmap changes
//! virtual-address bookkeeping but does not retarget that resource. Explicit
//! discard drops the transfer backing, and the next prepare or synchronize
//! resolves it again without replacing the host texture.
//!
//! This is the safety property the former deferred-window design lacked: it
//! parked raw host pointers across guest execution. This model retains page
//! identities, not pointers; every transfer still constructs bounded
//! `GuestSlice`s from the owning RAMBlock import.
//!
//! # Validity transitions decide direction
//!
//! A GPU Store makes the host image authoritative. A later guest
//! `clear_host_valid` makes the guest copy newer; payment then abandons the host
//! image rather than overwriting the guest's work. Surface resources use
//! `ResourceValidity`'s ordered sequence. Task-GVA resources use the validity
//! generation keyed by `(task, texture_ref)`, including the case where that
//! integer collides with an unrelated mapping id.
//!
//! A named synchronize pays only its object list through
//! [`submit_for_resources`]. Readers that know a mapping or texture call
//! [`pay_for_mapping`] or [`pay_for_texture`]. Only a genuinely unnameable
//! aliasing reader uses [`pay_all`]. Completion stamps alone do not publish
//! resources.
//!
//! The engine's `gpu_only_content` flag keeps an unpaid image alive. A
//! successful payment calls `note_resident_content_copied_out`; replacement,
//! invalidation, task retirement, and generation movement release the same
//! ownership without inventing a guest write.
//!
//! [`MAX_DEBTS`] bounds only anonymous type-11 surface debts. GVA resource
//! lifetime is explicit — resource discard/delete and task teardown — so an
//! unrelated capacity limit must not invent an early synchronization point.

use crate::model::DeviceState;
use crate::runtime::host::{HostMemory, HostOps};

/// Debts held at once, before an arm pays the oldest to make room.
///
/// This is the existing measured ceiling for the ledger, now shared by both
/// backing representations rather than duplicated per representation. An
/// insertion past it pays the oldest frame, so the bound can cost coalescing but
/// cannot lose pixels. `wbdebt_evicted` reports when a workload reaches it.
pub const MAX_DEBTS: usize = 32;

/// The engine identity of the resident a debt's frame lives in.
///
/// This module is compiled on every backend arm and the ledger is backend-
/// agnostic, but the identity of a resident is not: only the Vulkan engine has
/// one. An alias rather than a `cfg` on the field, so [`WritebackDebt`] and
/// [`PendingWritebacks::arm`] have **one** shape on every arm — two shapes is
/// how a struct starts disagreeing with itself across a feature boundary, and
/// nothing in the toolchain compares them.
///
/// Nothing arms a surface debt on a Metal build: the sole caller is
/// `runtime::draw::vulkan`. The placeholder is what lets the ledger's own tests
/// compile there anyway.
#[cfg(feature = "backend-vulkan")]
pub type ResidentIdentity = crate::backend::vulkan::engine::TargetIdentity;

/// See the Vulkan spelling above. A named zero-sized type rather than `()`,
/// because `()` as an argument is a clippy lint at every call site and a reader
/// cannot tell a deliberate placeholder from a function that forgot to return.
#[cfg(not(feature = "backend-vulkan"))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NoResidentIdentity;

/// See [`NoResidentIdentity`].
#[cfg(not(feature = "backend-vulkan"))]
pub type ResidentIdentity = NoResidentIdentity;

/// A synthetic resident identity, for tests that arm a debt without a device.
///
/// Here rather than in each test module because it is the one place that knows
/// which arm [`ResidentIdentity`] is on — two spellings of it would be a
/// divergence across a feature boundary, which is the thing the alias exists to
/// prevent.
#[cfg(test)]
#[cfg(feature = "backend-vulkan")]
pub(crate) fn test_resident_identity(
    id: u32,
    width: u32,
    height: u32,
    generation: u64,
) -> ResidentIdentity {
    crate::backend::vulkan::engine::TargetIdentity::Surface {
        id,
        width,
        height,
        generation,
        format: crate::backend::vulkan::translate::pixel::SCANOUT_FORMAT,
    }
}

/// The placeholder arm: every identity is the same value, which is exactly true
/// — a Metal build arms no surface debt at all.
#[cfg(test)]
#[cfg(not(feature = "backend-vulkan"))]
pub(crate) fn test_resident_identity(
    _id: u32,
    _width: u32,
    _height: u32,
    _generation: u64,
) -> ResidentIdentity {
    NoResidentIdentity
}

/// A frame owed to one type-11 mapping's guest pages.
///
/// Values only, and no memory. See the module doc: the rail this replaces held
/// resolved host pointers and corrupted the guest's page tables with them. A
/// [`crate::backend::vulkan::engine::TargetIdentity`] is `Copy` and every field
/// of it is a scalar the protocol handed over, so holding one keeps that rule.
///
/// `Clone` and not `Copy`, which the identity's own doc explains: it is a value
/// either way, and the debt is moved out of the ledger by `take` rather than
/// copied out of it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WritebackDebt {
    /// The resident the frame is *in*.
    ///
    /// Carried rather than re-derived, and that is the whole of a defect that
    /// lost every Apple Maps frame on the rail a discrete host has no
    /// alternative to. `pay` used to rebuild the identity from
    /// `present_identity::surface_identity`, which reads the mapping's
    /// generation *now*; the arm site already holds the identity the draw
    /// registered its resident under — it stamps that identity's content epoch
    /// one statement earlier — and the two are not the same key when the
    /// mapping's generation moved between the draw and the Store. A driven boot
    /// under `REIMS_VGPU_SHARED_TARGET=off` read
    /// `read_target_unknown_identity diverges=generation asked_gen=N held_gen=N-1`
    /// on three of four mappings, and the frame was refused with the resident
    /// holding it sitting in the registry.
    ///
    /// [`Self::map_generation`] beside it is a different question and both are
    /// needed: this says *which image the pixels are in*, that says *whether the
    /// pages are still the ones they were promised to*.
    pub identity: ResidentIdentity,
    /// Geometry the Store was taken at, and the geometry the payment writes.
    pub width: u32,
    pub height: u32,
    /// `MappingEntry::map_generation` at the arm.
    ///
    /// The payment refuses when the mapping's generation has moved since, so a
    /// surface the guest has remapped is void rather than paid into pages that
    /// now back something else. This is about the *destination*; see
    /// [`Self::identity`] for the source, which used to be inferred from this
    /// and cannot be.
    pub map_generation: u32,
    /// Arm order, for choosing which debt an over-full ledger pays first.
    pub seq: u64,
}

/// The guest resource that owns one GVA render attachment.
///
/// Unlike the address, this is also what `CmdSynchronizeResources` names. A
/// task is part of the key because object references are task-local.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct GvaResourceKey {
    pub task_id: u32,
    pub texture_ref: u32,
}

/// One render plane of a GVA resource.
///
/// # Why the ledger's unit is not the resource
///
/// A render pass targets exactly one mip level, and a level is a sub-range of
/// the resource's single allocation — `runtime::draw::render_target`'s
/// `level != 0` arm resolves it to that level's own `(gva, row_stride, height)`.
/// So one reference legitimately owns several live planes at once, and a ledger
/// keyed by the reference holds one entry where the guest is using three.
///
/// That was measured rather than reasoned. A driven macos-26 boot cycles one
/// reference through three declarations whose addresses are contiguous and
/// whose spans fall in exact 4:1 ratios — 256×192, 128×96, 64×48 of one RGBA8
/// allocation, the compositor's blur/backdrop pyramid. Keyed by the reference,
/// arming level 1's Store drops level 0's unpaid frame and every level change
/// mints a new generation, so no level's resident is ever reused.
///
/// [`GvaResourceKey`] stays the **resource**, because that is what
/// `CmdSynchronizeResources` and `CmdDeleteResource` name and what
/// `resource_validity` and `blit_exec` ask with — neither holds an address. The
/// derived `Ord` puts `resource` first, so `BTreeMap::range` over
/// [`GvaResourceKey::planes`] is every plane of one resource and the
/// resource-wide operations stay one lookup shape rather than a second map.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct GvaPlaneKey {
    pub resource: GvaResourceKey,
    pub gva: u64,
}

impl GvaResourceKey {
    /// This resource's plane at one guest address.
    pub fn plane(self, gva: u64) -> GvaPlaneKey {
        GvaPlaneKey {
            resource: self,
            gva,
        }
    }

    /// Every plane of this resource, as a `BTreeMap` range.
    ///
    /// Total by construction: the bounds are this resource's own lowest and
    /// highest representable plane, so no plane of it can sort outside them and
    /// no plane of another resource can sort inside.
    fn planes(self) -> std::ops::RangeInclusive<GvaPlaneKey> {
        self.plane(0)..=self.plane(u64::MAX)
    }
}

/// A frame held only by a GVA target's engine-resident image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GvaWritebackDebt {
    pub gva: u64,
    pub row_stride: u32,
    pub width: u32,
    pub height: u32,
    pub format: u16,
    pub generation: u64,
    pub guest_write: crate::runtime::buffer_write_gen::BufferWriteStamp,
    pub seq: u64,
}

/// The transfer backing retained by one live plane of a GVA texture resource.
///
/// The plane owns this physical-page identity after its virtual declaration has
/// been resolved. Task unmap changes the task's CPU mapping bookkeeping; it does
/// not retarget a live resource. An explicit resource discard drops only
/// `pages` — on every plane — allowing the next prepare/synchronize to establish
/// a new transfer backing without changing any host texture's identity.
///
/// The address is in the [`GvaPlaneKey`], so what remains here is what varies
/// per plane at one address: its length, its host-texture identity, and its
/// pages.
#[derive(Clone, Debug)]
struct GvaResourceState {
    generation: u64,
    span: u64,
    pages: Option<std::sync::Arc<[u64]>>,
}

/// One entry in the bounded ledger, irrespective of backing kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum WritebackKey {
    Mapping(u32),
}

/// Every render resource whose current frame exists only in a host resident.
///
/// Surface resources key by mapping id; GVA resources key by the plane of the
/// task-local reference a pass rendered into — see [`GvaPlaneKey`] for why the
/// plane and not the reference. In either representation, a second Store into
/// the same thing replaces the first rather than queueing another frame.
#[derive(Debug, Default)]
pub struct PendingWritebacks {
    debts: std::collections::BTreeMap<u32, WritebackDebt>,
    gva_debts: std::collections::BTreeMap<GvaPlaneKey, GvaWritebackDebt>,
    gva_resources: std::collections::BTreeMap<GvaPlaneKey, GvaResourceState>,
    next_seq: u64,
    next_gva_generation: u64,
}

impl PendingWritebacks {
    /// Mappings currently owed a frame.
    pub fn len(&self) -> usize {
        self.debts.len() + self.gva_debts.len()
    }

    /// Whether anything is owed at all — the check every reader makes, and the
    /// one that has to be free.
    pub fn is_empty(&self) -> bool {
        self.debts.is_empty() && self.gva_debts.is_empty()
    }

    /// What `mapping_id` is owed, if anything.
    pub fn get(&self, mapping_id: u32) -> Option<WritebackDebt> {
        self.debts.get(&mapping_id).cloned()
    }

    /// Take `mapping_id`'s debt, leaving it owed nothing.
    pub fn take(&mut self, mapping_id: u32) -> Option<WritebackDebt> {
        self.debts.remove(&mapping_id)
    }

    /// Every mapping owed a frame, oldest arm first.
    pub fn mappings_by_age(&self) -> Vec<u32> {
        let mut all: Vec<(u64, u32)> = self.debts.iter().map(|(id, d)| (d.seq, *id)).collect();
        all.sort_unstable();
        all.into_iter().map(|(_, id)| id).collect()
    }

    /// The surface mapping whose debt has been owed longest.
    fn oldest(&self) -> Option<WritebackKey> {
        self.debts
            .iter()
            .min_by_key(|(_, d)| d.seq)
            .map(|(id, _)| WritebackKey::Mapping(*id))
    }

    /// Record that `mapping_id` is owed a frame, returning the mapping whose
    /// debt the caller must pay to bring the ledger back under [`MAX_DEBTS`] —
    /// `None` in the ordinary case.
    ///
    /// A mapping already owed a frame is *replaced*: the later frame is the
    /// fresher answer and the earlier one has been superseded on the GPU
    /// already. That replacement is the whole saving, so it is counted.
    ///
    /// The over-full entry is left in the ledger and handed back by name rather
    /// than removed here, so [`PendingWritebacks::take`] stays the only way a
    /// debt leaves — a removal that is not a payment is a frame the guest asked
    /// for and never received.
    #[must_use = "an evicted mapping still owes a frame and the caller must pay it"]
    pub fn arm(
        &mut self,
        mapping_id: u32,
        identity: ResidentIdentity,
        width: u32,
        height: u32,
        map_generation: u32,
    ) -> Option<WritebackKey> {
        let evict = match self.debts.len() >= MAX_DEBTS && !self.debts.contains_key(&mapping_id) {
            true => self.oldest(),
            false => None,
        };
        let seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);
        let previous = self.debts.insert(
            mapping_id,
            WritebackDebt {
                identity,
                width,
                height,
                map_generation,
                seq,
            },
        );
        if previous.is_some() {
            crate::runtime::drain::note_store_route("wbdebt_superseded");
        }
        crate::runtime::drain::note_store_route("wbdebt_armed");
        evict
    }

    /// Record a host-authoritative frame for one plane of a GVA resource.
    ///
    /// A second Store into the same plane replaces the earlier debt. The
    /// returned previous debt names an older resident identity that the caller
    /// must release when the declaration changed.
    ///
    /// The debt's own `gva` picks the plane, so a pass into a *different* level
    /// of the same reference queues beside the first rather than dropping it.
    /// Keyed by the reference, arming a blur pyramid's level 1 discarded level
    /// 0's unpaid frame — see [`GvaPlaneKey`].
    #[must_use = "a replaced resource debt may own an older resident identity"]
    pub fn arm_gva(
        &mut self,
        key: GvaResourceKey,
        mut debt: GvaWritebackDebt,
    ) -> Option<GvaWritebackDebt> {
        debt.seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);
        let previous = self.gva_debts.insert(key.plane(debt.gva), debt);
        if previous.is_some() {
            crate::runtime::drain::note_store_route("gvadebt_superseded");
        }
        crate::runtime::drain::note_store_route("gvadebt_armed");
        previous
    }

    /// Establish or retrieve the lifetime identity of one live plane of a GVA
    /// resource.
    ///
    /// `pages` is accepted only on the first resolution after construction or
    /// explicit discard. Repeated draws and ordinary task unmaps keep the
    /// retained physical backing and the same host-texture generation.
    ///
    /// A plane of this reference at an address it has not used before is simply
    /// a new plane — the mip level case [`GvaPlaneKey`] describes — and gets its
    /// own generation without disturbing the others.
    ///
    /// # A second span at one address is a second resource
    ///
    /// A plane's length is fixed for its life: it comes from the creation
    /// descriptor and nothing in the protocol retargets it. So a draw naming
    /// this reference's plane at *this* address with a different span is not
    /// that plane at all — the guest retired the object and its object-list slot
    /// now holds another one, without the `CmdDeleteResource` that would have
    /// said so.
    ///
    /// That makes the new span a **lifetime boundary**, and the entry is
    /// replaced with a fresh generation rather than kept. Keeping it is what a
    /// caller cannot do anything correct with: the old generation names an image
    /// holding the old object's pixels, and the new declaration's image must not
    /// be that one.
    ///
    /// The surface rail has always worked this way — [`Self::arm`] replaces a
    /// debt and counts `wbdebt_superseded` — and this is the same rule spelled
    /// for the resource half of the ledger. The caller owns releasing whatever
    /// the old generation held; see [`gva_resource_generation`].
    pub fn ensure_gva_resource(
        &mut self,
        key: GvaResourceKey,
        gva: u64,
        span: u64,
        pages: Option<Vec<u64>>,
    ) -> u64 {
        let plane = key.plane(gva);
        if let Some(resource) = self.gva_resources.get_mut(&plane) {
            if resource.span == span {
                if resource.pages.is_none() {
                    resource.pages = pages.map(std::sync::Arc::from);
                }
                return resource.generation;
            }
        }
        self.next_gva_generation = self.next_gva_generation.wrapping_add(1);
        if self.next_gva_generation == 0 {
            self.next_gva_generation = 1;
        }
        let generation = self.next_gva_generation;
        self.gva_resources.insert(
            plane,
            GvaResourceState {
                generation,
                span,
                pages: pages.map(std::sync::Arc::from),
            },
        );
        generation
    }

    /// Give a live plane back the transfer backing an explicit discard took,
    /// without touching its declaration or its generation.
    ///
    /// This is what the payment path needs and all it may have. Payment names a
    /// plane it did not declare — the declaration it holds is the debt's,
    /// recorded when the frame was armed — so letting it reach
    /// [`Self::ensure_gva_resource`] gives a stale debt the power to resurrect a
    /// retired plane or to re-declare a live one out from under the draw that
    /// owns it. Asking here instead makes that unrepresentable: absent the
    /// plane, there is nothing to reback and nothing is created.
    ///
    /// `pages` is adopted only into a plane that has none, exactly as on the
    /// establishing path.
    #[cfg(feature = "backend-vulkan")]
    fn reback_gva_resource(&mut self, plane: GvaPlaneKey, pages: Option<Vec<u64>>) -> bool {
        let Some(resource) = self.gva_resources.get_mut(&plane) else {
            return false;
        };
        if resource.pages.is_none() {
            resource.pages = pages.map(std::sync::Arc::from);
        }
        true
    }

    #[cfg(any(feature = "backend-vulkan", test))]
    fn gva_resource_backing(
        &self,
        plane: GvaPlaneKey,
    ) -> Option<(u64, u64, std::sync::Arc<[u64]>)> {
        let resource = self.gva_resources.get(&plane)?;
        Some((
            resource.generation,
            resource.span,
            std::sync::Arc::clone(resource.pages.as_ref()?),
        ))
    }

    /// Gated on the arm that calls it. Unlike [`Self::gva_resource_backing`],
    /// which the tests in this module exercise directly, the only reader of
    /// this one is [`gva_resource_generation`], which is Vulkan-only — so
    /// admitting `test` here leaves it dead on the Metal arm's test build.
    #[cfg(feature = "backend-vulkan")]
    fn gva_resource_status(&self, plane: GvaPlaneKey) -> Option<(u64, u64, bool)> {
        self.gva_resources
            .get(&plane)
            .map(|resource| (resource.generation, resource.span, resource.pages.is_some()))
    }

    /// Release the transfer buffer of each named resource while preserving its
    /// host texture and lifetime identity.
    ///
    /// Every plane of a named resource, because the guest's discard names the
    /// resource and a resource holds all of its levels' backings.
    pub fn discard_gva_resources(&mut self, task_id: u32, object_ids: &[u32]) -> usize {
        let mut discarded = 0;
        for &texture_ref in object_ids {
            let key = GvaResourceKey {
                task_id,
                texture_ref,
            };
            for resource in self.gva_resources.range_mut(key.planes()) {
                discarded += usize::from(resource.1.pages.take().is_some());
            }
        }
        discarded
    }

    /// Every plane of one resource goes at once: `CmdDeleteResource` names the
    /// resource, and a level that outlived its allocation names nothing.
    fn retire_gva_resource(&mut self, key: GvaResourceKey) -> (bool, Vec<GvaWritebackDebt>) {
        let planes: Vec<GvaPlaneKey> = self
            .gva_resources
            .range(key.planes())
            .map(|(plane, _)| *plane)
            .chain(self.gva_debts.range(key.planes()).map(|(plane, _)| *plane))
            .collect();
        let mut existed = false;
        let mut debts = Vec::new();
        for plane in planes {
            existed |= self.gva_resources.remove(&plane).is_some();
            debts.extend(self.gva_debts.remove(&plane));
        }
        (existed, debts)
    }

    /// The one plane debt this resource owes, or `None` when it owes zero or
    /// several.
    ///
    /// The caller — `blit_exec`'s whole-plane GPU copy — names a resource and
    /// holds no address, so with several planes owed it cannot say which one its
    /// source level is. Declining costs it the GPU shortcut and nothing else:
    /// that path's own doc records that a fall-through spends a frame and cannot
    /// lose one.
    pub fn get_gva(&self, key: GvaResourceKey) -> Option<GvaWritebackDebt> {
        let mut owed = self.gva_debts.range(key.planes());
        let (_, only) = owed.next()?;
        match owed.next() {
            None => Some(*only),
            Some(_) => {
                crate::runtime::drain::note_store_route("gvadebt_resource_owes_many_planes");
                None
            }
        }
    }

    pub fn has_gva(&self, key: GvaResourceKey) -> bool {
        self.gva_debts.range(key.planes()).next().is_some()
    }

    /// Every plane debt this resource owes, taken out of the ledger.
    ///
    /// The resource is the unit the guest synchronizes and the unit a sampled
    /// read names, so a payment for it owes every level's frame and not the one
    /// that happened to sort first.
    pub fn take_gva(&mut self, key: GvaResourceKey) -> Vec<(GvaPlaneKey, GvaWritebackDebt)> {
        let planes: Vec<GvaPlaneKey> = self
            .gva_debts
            .range(key.planes())
            .map(|(plane, _)| *plane)
            .collect();
        planes
            .into_iter()
            .filter_map(|plane| self.gva_debts.remove(&plane).map(|debt| (plane, debt)))
            .collect()
    }

    fn take_gva_plane(&mut self, plane: GvaPlaneKey) -> Option<GvaWritebackDebt> {
        self.gva_debts.remove(&plane)
    }

    /// Put back a debt whose guest backing was temporarily unavailable.
    /// Preserves its original age: inability to pay does not make an old frame
    /// the newest member of the ledger.
    #[cfg(feature = "backend-vulkan")]
    fn restore_gva(&mut self, plane: GvaPlaneKey, debt: GvaWritebackDebt) {
        let previous = self.gva_debts.insert(plane, debt);
        debug_assert!(
            previous.is_none(),
            "a taken debt restores into its own hole"
        );
    }

    fn gvas_by_age(&self) -> Vec<GvaPlaneKey> {
        let mut all: Vec<(u64, GvaPlaneKey)> = self
            .gva_debts
            .iter()
            .map(|(key, debt)| (debt.seq, *key))
            .collect();
        all.sort_unstable();
        all.into_iter().map(|(_, key)| key).collect()
    }

    /// Distinct resources of one task, deduped across their planes: task
    /// teardown retires resources, and [`Self::retire_gva_resource`] already
    /// takes every plane of the one it is given.
    fn gvas_for_task(&self, task_id: u32) -> Vec<GvaResourceKey> {
        let mut all: Vec<GvaResourceKey> = self
            .gva_resources
            .keys()
            .map(|plane| plane.resource)
            .filter(|key| key.task_id == task_id)
            .collect();
        all.dedup();
        all
    }

    #[cfg(feature = "backend-vulkan")]
    fn gva_for_identity(
        &self,
        identity: &crate::backend::vulkan::engine::TargetIdentity,
    ) -> Option<(GvaPlaneKey, GvaWritebackDebt)> {
        let crate::backend::vulkan::engine::TargetIdentity::Gva {
            gva,
            width,
            height,
            generation,
            ..
        } = *identity
        else {
            return None;
        };
        self.gva_debts
            .iter()
            .find(|(_, debt)| {
                debt.gva == gva
                    && debt.width == width
                    && debt.height == height
                    && debt.generation == generation
            })
            .map(|(key, debt)| (*key, *debt))
    }
}

/// Whether the lazy rail is on for this process.
///
/// Read once. The rail changes *when* a frame reaches guest pages, not what the
/// bytes are, so a boot that flipped it midway would be two devices in one log.
pub fn lazy_writeback_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let (state, value) = crate::env::read(crate::env::LAZY_WRITEBACK);
        // Only an explicit `off` narrows to the eager Store. Unset, `on` and an
        // unrecognized value are all the shipping rail, which is what makes
        // `Switch::Unrecognized` — an operator's typo — fail toward the measured
        // default rather than silently selecting the arm it is 45 % slower on.
        let on = !matches!(state, crate::env::Switch::Off);
        crate::observe::off(format!(
            "lazy_writeback on={on} switch={state:?} value={}",
            value.unwrap_or_else(|| "<unset>".into())
        ));
        on
    })
}

/// Pay `mapping_id`'s owed frame, if it owes one.
///
/// The one call a reader of a named mapping's guest bytes makes before it reads
/// them. Free when nothing is owed — one `BTreeMap` emptiness check, which is
/// the answer on nearly every call.
pub fn pay_for_mapping<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
) {
    if state.pending_writebacks.is_empty() {
        return;
    }
    let Some(debt) = state.pending_writebacks.take(mapping_id) else {
        return;
    };
    pay(state, host, mapping_id, debt, "wbdebt_paid_named");
}

/// Pay every owed frame.
///
/// For a reader that cannot name the mapping it is about to read — a GVA span, a
/// buffer, a page walk that may alias a surface. Aliasing across the id
/// namespaces is real, so "cannot say" resolves to "owes all of them".
///
/// # Why the disjointness closures those readers already carry do not narrow it
///
/// The three GVA readers each build the exact page list they are about to touch,
/// and hand it to
/// [`crate::runtime::render_writeback::settle_guest_writes_unless_disjoint`] so a
/// reader somewhere else entirely does not wait for a surface's writeback. That
/// narrowing cannot be reused here, and the reason is the rail itself: the test
/// runs only when a copy is **outstanding**, and an owed frame has not been
/// submitted at all. With the lazy rail on, the common state is a clear debt
/// flag and a full ledger, where the closure never runs.
///
/// Narrowing this would need a page-extent hint held per debt, and a hint is the
/// beginning of holding resolved memory — which is what the module doc says this
/// rail must not do. [`note_unnamed_reach`] is the instrument that says whether
/// it would be worth it; read its doc before building one.
pub fn pay_all<M: HostMemory + HostOps>(state: &mut DeviceState, host: &mut M) {
    if state.pending_writebacks.is_empty() {
        return;
    }
    for mapping_id in state.pending_writebacks.mappings_by_age() {
        let Some(debt) = state.pending_writebacks.take(mapping_id) else {
            continue;
        };
        pay(state, host, mapping_id, debt, "wbdebt_paid_all");
    }
    for plane in state.pending_writebacks.gvas_by_age() {
        let Some(debt) = state.pending_writebacks.take_gva_plane(plane) else {
            continue;
        };
        let _ = pay_gva(state, host, plane, debt, GvaPaySite::All);
    }
}

/// One call in [`REACH_SAMPLE`] does the walk; the rest cost one modulo.
///
/// The walk is ~2 000 page-table descents for a 1080p span and the site that
/// dominates [`pay_all`] runs about 1 700 times a second, so measuring every call
/// would cost more than the rail saves and would be measuring the instrument.
/// A census wants a rate and not a total, and a rate converges on a 1-in-64
/// sample of a population this size: ~26 walks a second against ~1 700 calls.
const REACH_SAMPLE: u64 = 64;

/// Calls to [`note_unnamed_reach`], for the sample.
static REACH_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Does an unnameable reader that pays every debt actually read the pages it is
/// paying for?
///
/// # The question, and why it decides the rail's ceiling
///
/// The premise the lazy rail was built on read the settle census wrong. Those
/// counters count settles that **waited**, and on a driven macos-13
/// sustained-animation boot they total six a second — which is what "840 writes
/// consumed six times" was derived from. The *calls* are a different population:
/// `draw::vulkan::load_linear_guest_memoized` alone reaches its settle about
/// 1 700 times a second, reads the guest pages every one of them, and cannot name
/// a mapping, so it pays every owed frame. That is why the first driven on-arm
/// boot coalesced 130 Stores of 577 rather than the ~95 % the premise predicted.
///
/// But paying is only *owed* where the read and the surface share pages. A
/// compositor sampling a glyph atlas while three windows owe frames pays three
/// copies it will not look at. This counts which it is:
///
/// * `wbdebt_reach_overlap` — the sampled read touched a page some debt's
///   mapping holds. The payment was owed and no narrowing can remove it.
/// * `wbdebt_reach_disjoint` — it did not. The payment was pure waste, and the
///   ratio of these two is the prize a page-extent hint per debt would collect.
/// * `wbdebt_reach_unnamed` — the reader's own walk came back short, so nothing
///   could be ruled out. A narrowing must treat this as overlap.
///
/// `pages` is the reader's own closure, the same one it hands the disjointness
/// test, so both ends of the comparison stay one rule. It runs only on a sampled
/// call and only while something is owed.
/// Private, and that is the half of the repair a reader is most likely to undo.
/// The census is one of three terms a raw-GVA reader owes and it is worthless on
/// its own — it reports what the naming missed, so a site that censuses without
/// paying has measured its own omission and done nothing about it.
/// [`settle_for_texture`] is the only way to it, and that one spells all three.
fn note_unnamed_reach(state: &DeviceState, pages: impl FnOnce() -> Option<Vec<u64>>) {
    if state.pending_writebacks.is_empty() {
        return;
    }
    let n = REACH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if !n.is_multiple_of(REACH_SAMPLE) {
        return;
    }
    let Some(read) = pages() else {
        crate::runtime::drain::note_store_route("wbdebt_reach_unnamed");
        return;
    };
    let read: std::collections::BTreeSet<u64> = read.into_iter().collect();
    let overlap = state
        .pending_writebacks
        .mappings_by_age()
        .into_iter()
        .any(|mapping_id| {
            state
                .mapping_reach_pages(mapping_id)
                .is_some_and(|owed| owed.iter().any(|page| read.contains(page)))
        });
    match overlap {
        true => crate::runtime::drain::note_store_route("wbdebt_reach_overlap"),
        false => crate::runtime::drain::note_store_route("wbdebt_reach_disjoint"),
    }
}

/// Pay whatever a *texture* reference names, for a reader that reaches guest
/// bytes through a task GVA but knows which resource it is reading.
///
/// # Why a GVA reader is nameable after all, and what that measured
///
/// The three linear readers walk raw task GVAs, so the first cut of this rail
/// had them pay every owed frame. [`note_unnamed_reach`] priced that: **173
/// sampled payments over one driven macos-13 sustained-animation boot, 173
/// disjoint from every owed surface and not one overlap**, at a 1-in-64 sample
/// of ~11 000 payments. Meanwhile `wbdebt_paid_all` was 20 391 against
/// `wbdebt_paid_named` 755, so 96 % of all payments were the ones that read
/// nothing they paid for, and they cost `sampled_us` 1.64 → 8.49 us a chain.
///
/// They are nameable because the guest names them. A debt is keyed by mapping
/// id, and this device holds two ways from a texture reference to one:
/// `DeviceState::texture_to_mapping` for the per-task registration, and the id
/// itself where the guest uses one namespace for both.
/// [`crate::runtime::resource_validity::apply`] resolves a validity statement
/// through exactly this pair, so both now go through the one resolver that owns
/// it, [`crate::model::DeviceState::mappings_named_by`] — this used to be the
/// same question asked of the same two tables in two different spellings, and
/// the divergence is in that method's doc.
///
/// A reference that resolves to neither names no mapping this device holds, so
/// no debt can be about it. That is a statement about the registries and not
/// about a workload — but it is not a statement about raw *page* aliasing, where
/// a surface's pages are re-used as some other resource's backing with no
/// mapping entry. [`note_unnamed_reach`] stays wired at these sites as the
/// standing alarm for exactly that: it samples the read's own page walk against
/// every owed surface's pages, and `wbdebt_reach_overlap` above zero is a
/// payment this naming skipped and should not have.
pub fn pay_for_texture<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
) {
    if state.pending_writebacks.is_empty() {
        return;
    }
    let gva_key = GvaResourceKey {
        task_id,
        texture_ref,
    };
    let mut named = false;
    // Every plane the reference owes, not the one that sorts first: a sampled
    // read names the resource, and a mip pyramid's levels are separate debts.
    for (plane, debt) in state.pending_writebacks.take_gva(gva_key) {
        named = true;
        let _ = pay_gva(state, host, plane, debt, GvaPaySite::Named);
    }
    // Both surface spellings, from the one resolver `resource_validity::apply`
    // uses: a reference that is itself a mapping id, and the per-task
    // registration. Paying one leaves the ledger holding the other, so asking
    // for each costs a map lookup and cannot pay the wrong surface.
    let targets = state.mappings_named_by(task_id, texture_ref);
    for mapping_id in targets.iter() {
        if state.pending_writebacks.get(mapping_id).is_some() {
            named = true;
            pay_for_mapping(state, host, mapping_id);
        }
    }
    if !named {
        // "Nothing was owed" is two opposite findings, and this counter reported
        // them as one at 1.1 M a boot — the same volume as every sampled guest
        // import, which is what made it read as a healthy zero.
        //
        // `_resolved` says the ledger genuinely holds no debt for a surface this
        // reference does name. `_unresolved` says neither spelling named a live
        // surface at all: `texture_ref` is not a mapping this device holds and
        // `texture_to_mapping` names none either. A debt owed by the surface
        // behind such a reference cannot be found, so that arm is "we did not
        // look", not "there was nothing there" — and a sampled bind proceeding
        // past it reads the guest's pages while the newest frame is still in a
        // resident.
        //
        // The split asks [`DeviceState::names_live_mapping`], not whether the
        // per-task registration answered. It used to ask the latter, which the
        // reference-is-the-mapping-id spelling never populates, so the census
        // read 100 % `_unresolved` on both arms of a driven macos-13 boot —
        // 74 816/74 816 with the guest import on and 185 674/185 674 with it
        // off. A split whose two arms cannot both be reached measures nothing,
        // and this one was read as evidence that the naming never resolves.
        //
        // The split is emitted beside the total, so
        // `_resolved + _unresolved == wbdebt_texture_owes_nothing` is checkable
        // on the census itself.
        crate::runtime::drain::note_store_route(
            match state.names_live_mapping(task_id, texture_ref) {
                true => "wbdebt_texture_owes_nothing_resolved",
                false => "wbdebt_texture_owes_nothing_unresolved",
            },
        );
        crate::runtime::drain::note_store_route("wbdebt_texture_owes_nothing");
    }
}

/// The stable host-texture identity for the GVA resource a draw is declaring.
///
/// The first successful resolution retains the ordered physical pages that the
/// resource's transfer buffer names. Later calls return the same generation and
/// backing even if the task removes its virtual mapping. After explicit
/// discard, the next call may establish a replacement transfer backing while
/// preserving the host texture's generation.
///
/// # A changed declaration ends one lifetime and begins the next
///
/// This used to answer `0` and emit `gva_resource_refused
/// reason=declaration_changed` when the draw's `(gva, span)` differed from the
/// one the entry was established with, on the reading that a live resource
/// cannot move. The reading is right and the response was not: the resource did
/// not move, the *reference* was reused, and the entry describing the retired
/// object is the thing that has to go.
///
/// Answering `0` never recovered. The entry stayed, so every later draw into
/// that reference compared against the same dead declaration and refused again —
/// one macos-26 report carried 5 197 of these lines over 280 references, one of
/// them refused 803 times in a single boot. What `0` costs depends on which
/// caller asked: `draw::vulkan`'s resident resolve turns it into
/// `GvaResidentRefusal::NoGeneration` and loses the frame, while the secondary
/// MRT builder puts it straight into [`TargetIdentity::Gva`], where generation
/// zero is the one value that cannot distinguish two allocations — the
/// wrong-content class that identity exists to close.
///
/// So a differing declaration is handled as what it is, a lifetime boundary,
/// through the same [`retire_gva_resource`] that `CmdDeleteResource` uses: the
/// old generation's unpaid frame is released rather than written into storage
/// the retired object no longer owns — the rule [`retire_gva_for_task`] already
/// states for task teardown — and [`PendingWritebacks::ensure_gva_resource`]
/// then establishes the new object's own generation.
///
/// It stays fail-visible, because a *frequent* redeclaration would say something
/// different: that some producer in this device describes one live resource two
/// ways, in which case each draw would mint a generation and no resident could
/// ever be reused. The line names both declarations so that reading can be made
/// from a log rather than from a rebuild.
///
/// [`TargetIdentity::Gva`]: crate::backend::vulkan::engine::TargetIdentity::Gva
#[cfg(feature = "backend-vulkan")]
pub fn gva_resource_generation<M: HostMemory>(
    state: &mut DeviceState,
    host: &M,
    key: GvaResourceKey,
    gva: u64,
    span: u64,
) -> u64 {
    if let Some((generation, declared_span, has_pages)) =
        state.pending_writebacks.gva_resource_status(key.plane(gva))
    {
        if declared_span == span {
            if has_pages {
                return generation;
            }
        } else {
            crate::observe::Emit::decline(
                "gva_resource_redeclared",
                &GvaResourceRedeclared {
                    gva,
                    was_span: declared_span,
                    now_span: span,
                },
            )
            .field("task", key.task_id)
            .field("texture", key.texture_ref)
            .fail();
            retire_gva_resource(state, key.task_id, key.texture_ref);
        }
    }
    let page_size = state.page_size();
    let ordered = crate::runtime::gva_mem::task_gva_page_gpas(
        host,
        &state.tasks,
        key.task_id,
        gva,
        span,
        state.page_shift,
    );
    let want = reims_vgpu_paging::span::pages_spanned(gva, span, page_size);
    let pages = (ordered.len() as u64 == want).then_some(ordered);
    state
        .pending_writebacks
        .ensure_gva_resource(key, gva, span, pages)
}

/// One plane of a reference observed at two different lengths.
///
/// A *different address* under one reference is not this: that is another plane
/// of the same resource — a mip level — and [`GvaPlaneKey`] gives it its own
/// entry. What remains here is one address whose length moved, which the
/// contract has no room for, so the reference has been reused for a second
/// object.
///
/// Carries both lengths because neither alone says anything: the question a
/// reader has is whether they are *stable* — a reference reused, ordinary guest
/// lifetime — or whether they alternate, which would be this device describing
/// one plane two ways.
#[cfg(feature = "backend-vulkan")]
struct GvaResourceRedeclared {
    gva: u64,
    was_span: u64,
    now_span: u64,
}

#[cfg(feature = "backend-vulkan")]
impl crate::observe::Decline for GvaResourceRedeclared {
    fn slug(&self) -> &'static str {
        "gva_resource_declaration_changed"
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("gva", format!("{:#x}", self.gva)),
            ("was_span", self.was_span.to_string()),
            ("now_span", self.now_span.to_string()),
        ]
    }
}

#[cfg(feature = "backend-vulkan")]
crate::observe::decline_display!(GvaResourceRedeclared);

/// Re-establish the transfer backing of the plane a debt names, without any
/// power to declare one.
///
/// The payment path's counterpart to [`gva_resource_generation`]. It asks only
/// the question payment has standing to ask — "does this plane still exist, and
/// does it still have its pages" — using the plane's *own* span, never the
/// debt's. A debt that outlived its plane therefore finds nothing here and is
/// released by the caller, where before it reached
/// [`PendingWritebacks::ensure_gva_resource`] and could re-create the retired
/// object at the dead declaration it was carrying.
#[cfg(feature = "backend-vulkan")]
fn reback_gva_resource<M: HostMemory>(
    state: &mut DeviceState,
    host: &M,
    plane: GvaPlaneKey,
) -> bool {
    let Some((_, span, has_pages)) = state.pending_writebacks.gva_resource_status(plane) else {
        return false;
    };
    if has_pages {
        return true;
    }
    let page_size = state.page_size();
    let ordered = crate::runtime::gva_mem::task_gva_page_gpas(
        host,
        &state.tasks,
        plane.resource.task_id,
        plane.gva,
        span,
        state.page_shift,
    );
    let want = reims_vgpu_paging::span::pages_spanned(plane.gva, span, page_size);
    let pages = (ordered.len() as u64 == want).then_some(ordered);
    state.pending_writebacks.reback_gva_resource(plane, pages)
}

/// Record a GVA render result as host-authoritative without touching guest
/// pages. Returns `false` when the attachment has no resource identity and must
/// use the eager transfer path.
#[cfg(feature = "backend-vulkan")]
pub fn arm_gva<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    c0: &crate::runtime::draw::ColorRtRequest,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
) -> bool {
    let Some(generation) = (match *identity {
        crate::backend::vulkan::engine::TargetIdentity::Gva { generation, .. } => Some(generation),
        _ => None,
    }) else {
        return false;
    };
    if c0.texture_ref == 0 || generation == 0 {
        return false;
    }
    // Every older host-side spelling of this resource is stale as soon as the
    // render finishes. In particular, a compute storage resident and the
    // linear byte cache can otherwise sit above the guest-page reader and serve
    // the frame that preceded this Store indefinitely.
    state.invalidate_object_host_copies(task_id, c0.texture_ref);
    crate::runtime::surface_cache::evict_gva(state, c0.target_gva);
    let key = GvaResourceKey {
        task_id,
        texture_ref: c0.texture_ref,
    };
    let plane = key.plane(c0.target_gva);
    // Arm the guest-write witness *before* the ledger, because whether it arms
    // decides whether this frame may be deferred at all.
    //
    // A deferred frame is a host-authoritative copy of guest pages the guest CPU
    // may write at any moment with no device operation, and the only recovery
    // from such a write is to land the frame everywhere the guest did not touch
    // — which needs the hypervisor's per-page report. Deferring without it is
    // not a cheaper Store, it is a Store that a single guest write anywhere in
    // the plane deletes: the layer reverts to whatever its pages held and every
    // pixel the GPU rendered outside the guest's own rectangle is gone.
    //
    // So the rule is the writer's, and stricter than the reader's: a Store may
    // be deferred only while this device can still say what the guest wrote in
    // the meantime. Everything else takes the eager copying rail the caller
    // falls back to, which lands the frame in the guest's pages now and has
    // nothing left to lose.
    if !gva_writeback_is_recoverable(state, host, identity, plane) {
        return false;
    }
    let debt = GvaWritebackDebt {
        gva: c0.target_gva,
        row_stride: c0.row_stride,
        width: c0.width,
        height: c0.height,
        format: c0.format,
        generation,
        guest_write: state.buffer_write_gen.stamp(task_id, c0.texture_ref),
        seq: 0,
    };
    let previous = state.pending_writebacks.arm_gva(key, debt);
    if let Some(previous) = previous.filter(|previous| !same_gva_identity(*previous, debt)) {
        release_gva(previous);
    }
    true
}

/// Whether a frame deferred into `plane`'s resident could still be landed after
/// a guest CPU write into those pages.
///
/// The witness this arms used to be stamped only by the *eager* store, which is
/// the one rail with nothing outstanding: after an eager store the guest's pages
/// already hold the frame, so a guest write over them costs nothing. Here it
/// costs the frame, so this is where the question has to become answerable.
///
/// The hypervisor's set has an arming window — its generation reads back 0 until
/// a harvest has run over it — so the first Store into a fresh plane arms the
/// set and answers `false`, and the next Store into it can defer. Measured on
/// the macos-15 conformance battery: 229 of 1 349 GVA Stores were inside that
/// window, and every one of the 7 frames the ledger lost to a guest write was
/// one of them.
#[cfg(feature = "backend-vulkan")]
fn gva_writeback_is_recoverable<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    plane: GvaPlaneKey,
) -> bool {
    let Some((_, _, ordered)) = state.pending_writebacks.gva_resource_backing(plane) else {
        // No page list, so no set to watch and no destination to land into
        // later either. Both halves of a deferral are missing.
        crate::runtime::drain::note_store_route("gvadebt_arm_unbacked");
        return false;
    };
    let Some(key) = crate::runtime::gva_store_witness::GvaTargetKey::of(identity) else {
        crate::runtime::drain::note_store_route("gvadebt_arm_unnamed");
        return false;
    };
    crate::runtime::gva_store_witness::note_store(state, host, key, &ordered);
    if crate::runtime::gva_store_witness::armed(state, key) {
        true
    } else {
        crate::runtime::drain::note_store_route("gvadebt_arm_unwitnessed");
        false
    }
}

/// Whether this exact GVA resident is the host-authoritative copy named by an
/// unpaid resource debt.
#[cfg(feature = "backend-vulkan")]
pub fn gva_resident_authoritative(
    state: &DeviceState,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
) -> bool {
    let Some((plane, debt)) = state.pending_writebacks.gva_for_identity(identity) else {
        return false;
    };
    state
        .buffer_write_gen
        .stamp(plane.resource.task_id, plane.resource.texture_ref)
        .quiet_since(debt.guest_write)
}

/// Retire host-authoritative resources whose task-local references are about to
/// be replaced. The pixels are deliberately not copied: after this lifecycle
/// transition the old object no longer names guest storage to synchronize.
pub fn retire_gva_for_task(state: &mut DeviceState, task_id: u32) -> usize {
    let keys = state.pending_writebacks.gvas_for_task(task_id);
    let mut retired = 0;
    for key in keys {
        let (_, debts) = state.pending_writebacks.retire_gva_resource(key);
        retired += 1;
        #[cfg(feature = "backend-vulkan")]
        for debt in debts {
            release_gva(debt);
        }
        #[cfg(not(feature = "backend-vulkan"))]
        let _ = debts;
    }
    if retired != 0 {
        crate::runtime::drain::note_store_route_n("gvadebt_retired_task", retired as u64);
    }
    retired
}

/// Retire one resource at its explicit lifetime boundary.
pub fn retire_gva_resource(state: &mut DeviceState, task_id: u32, texture_ref: u32) -> bool {
    let key = GvaResourceKey {
        task_id,
        texture_ref,
    };
    let (existed, debts) = state.pending_writebacks.retire_gva_resource(key);
    let owed = !debts.is_empty();
    #[cfg(feature = "backend-vulkan")]
    for debt in debts {
        release_gva(debt);
    }
    #[cfg(not(feature = "backend-vulkan"))]
    let _ = debts;
    existed || owed
}

/// Release named resources' retained transfer backings.
pub fn discard_gva_resources(state: &mut DeviceState, task_id: u32, object_ids: &[u32]) -> usize {
    state
        .pending_writebacks
        .discard_gva_resources(task_id, object_ids)
}

#[cfg(feature = "backend-vulkan")]
fn same_gva_identity(a: GvaWritebackDebt, b: GvaWritebackDebt) -> bool {
    a.gva == b.gva
        && a.width == b.width
        && a.height == b.height
        && a.generation == b.generation
        && a.format == b.format
}

/// The engine resident one armed GVA debt names.
///
/// `pub(crate)` because a debt is not only something to pay: a reader that wants
/// the *content* rather than the guest's copy of it — the blit rail's whole-plane
/// GPU arm — needs exactly this identity, and deriving a second one from the same
/// debt fields is how two spellings of one resident start disagreeing. There is
/// one derivation and it is here.
#[cfg(feature = "backend-vulkan")]
pub(crate) fn gva_identity(
    debt: GvaWritebackDebt,
) -> crate::backend::vulkan::engine::TargetIdentity {
    crate::backend::vulkan::engine::TargetIdentity::Gva {
        gva: debt.gva,
        width: debt.width,
        height: debt.height,
        generation: debt.generation,
        format: crate::runtime::draw::gva_resident_format(debt.format),
    }
}

#[cfg(feature = "backend-vulkan")]
fn release_gva(debt: GvaWritebackDebt) {
    crate::backend::vulkan::engine::note_resident_content_copied_out(&gva_identity(debt));
}

#[cfg(feature = "backend-vulkan")]
pub(crate) fn pay_key<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    key: WritebackKey,
) -> bool {
    match key {
        WritebackKey::Mapping(mapping_id) => {
            if let Some(debt) = state.pending_writebacks.take(mapping_id) {
                pay(state, host, mapping_id, debt, "wbdebt_paid_evicted");
            }
            true
        }
    }
}

/// Pay `mapping_id`'s owed frame and then wait for every submitted guest-page
/// write **that can reach this mapping's pages** — the whole obligation of a
/// host-side reader or writer of one named mapping's bytes, in one call.
///
/// The two halves are one obligation and are spelled as one function so a new
/// site cannot discharge half of it. A site that settles without paying reads
/// the frame *before* the one the guest's own driver last asked for; a site that
/// pays without settling reads the frame it just submitted and has not waited
/// for. Both are stale pixels and neither shows up as a refusal.
///
/// # Naming a mapping is naming your reach
///
/// This used to wait on `settle_guest_writes`, which quiesces every outstanding
/// guest write wherever it lands, and a disjointness-aware second spelling sat
/// beside it for the two callers that had noticed. That split was the bug. A
/// caller that can name the mapping it is about to touch has, by naming it,
/// already said which pages it needs ordered; waiting for a writeback into some
/// other surface is an ordering requirement it does not have. So there is one
/// function again and the imprecise behaviour is unreachable rather than
/// available.
///
/// It cost the largest single number this device has measured. A driven macos-13
/// Maps leg took the quiesce on **2154 of 2154** `MappingRectWrite` calls for
/// **17.02 s** — 79 % of the blit rail and 350x the next-largest settle site —
/// while the payment beside it totalled 1.7 ms. The old doc here claimed "free
/// when nothing is owed and nothing is outstanding, which is the answer on
/// nearly every call"; on that site it was the answer on none of them.
///
/// The narrowing applies to the *wait* only, never the payment: an owed frame
/// has not been submitted, so there is nothing outstanding for the test to find
/// disjoint from, and a debt left unpaid here would be read straight past.
///
/// The page set is walked here rather than taken as a closure, and both of those
/// are deliberate. Walked *here* because the payment needs `state` mutably and
/// the disjointness test needs it shared, so a caller-supplied closure cannot
/// hold `state` across both. [`DeviceState::mapping_reach_pages`] because that is
/// the same function the writeback builds its own destination from, so the two
/// ends of the comparison are one rule rather than two spellings of it. It stays
/// lazy: `settle_guest_writes_unless_disjoint` runs the closure only when
/// something is outstanding, and a mapping that cannot name its pages answers
/// `None`, which waits.
pub fn settle_for_mapping<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    site: crate::runtime::render_writeback::SettleSite,
) {
    // Charged apart because subtracting the wait cannot tell them apart, and
    // after the wait went away the remainder was still 4.9 s on a driven leg:
    // that is either the payment doing work the quiesce used to have already
    // landed, or the reach walk itself, and those want opposite repairs.
    let pay_started = std::time::Instant::now();
    pay_for_mapping(state, host, mapping_id);
    crate::runtime::drain::note_store_route_us(
        "wbdebt_pay_us",
        pay_started.elapsed().as_micros() as u64,
    );
    let reach_started = std::time::Instant::now();
    let s = &*state;
    crate::runtime::render_writeback::settle_guest_writes_unless_disjoint(site, || {
        crate::runtime::drain::note_store_route("wbdebt_reach_walk_n");
        s.mapping_reach_pages(mapping_id)
    });
    crate::runtime::drain::note_store_route_us(
        "wbdebt_reach_us",
        reach_started.elapsed().as_micros() as u64,
    );
}

/// [`settle_for_mapping`] for a reader that names a **resource** and walks a raw
/// task GVA rather than a mapping — the whole obligation of a CPU read of one
/// named resource's guest bytes, in one call.
///
/// # Why this exists, which is the same reason [`settle_for_mapping`] does
///
/// That function's doc says the payment and the wait "are one obligation and are
/// spelled as one function so a new site cannot discharge half of it", and nine
/// mapping-named sites go through it. The resource-named readers never got the
/// equivalent, so each wrote the three terms out by hand: the
/// [`note_unnamed_reach`] census, the payment, and the disjointness-narrowed
/// settle, with the page walk spelled twice per site because the census and the
/// settle each take their own closure.
///
/// Three hand-written copies is the shape this repository keeps paying for, and
/// it failed here in the predicted direction. `draw::read_buffer_bytes_resolved`
/// carried the settle **alone** — no census and no payment — because it held
/// `DeviceState` shared and structurally could not pay, so a buffer-backed
/// sampled texture read the guest's pages with the rendered frame still sitting
/// in a host resident. A settle waits for writes already submitted; an owed frame
/// has not been submitted at all, so the wait returns at once and finds nothing.
///
/// The three terms are ordered and the order matters. The census runs **first**,
/// while the ledger still holds what the payment is about to clear — a census
/// after the payment reports an empty ledger and reads as a healthy zero. The
/// payment runs before the settle because it is what puts the owed frame on the
/// queue; settling first would wait for a copy nobody had issued yet.
///
/// The walk is done here rather than taken as a closure for the reason
/// [`settle_for_mapping`] gives: the payment needs `state` mutably and the
/// disjointness test needs it shared, so one caller-supplied closure cannot span
/// both. Both terms now get the same walk from the same expression, which is the
/// point — a site cannot census one span and settle a different one.
pub fn settle_for_texture<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    gva: u64,
    span: u64,
    site: crate::runtime::render_writeback::SettleSite,
) {
    // The reference names a resource and a surface debt is keyed by mapping id,
    // so the payment reaches only what this reference resolves to. The census is
    // the standing alarm for the one thing that naming cannot see — raw page
    // aliasing, where a surface's pages back some other resource with no mapping
    // entry — and `wbdebt_reach_overlap` must stay at zero.
    {
        let (tasks, page_shift) = (&state.tasks, state.page_shift);
        let page_size = state.page_size();
        note_unnamed_reach(state, || {
            let want = reims_vgpu_paging::span::pages_spanned(gva, span, page_size);
            let gpas = crate::runtime::gva_mem::task_gva_page_gpas(
                host, tasks, task_id, gva, span, page_shift,
            );
            (gpas.len() as u64 == want).then_some(gpas)
        });
    }
    pay_for_texture(state, host, task_id, texture_ref);
    let (tasks, page_shift, page_size) = (&state.tasks, state.page_shift, state.page_size());
    crate::runtime::render_writeback::settle_guest_writes_unless_disjoint(site, || {
        let want = reims_vgpu_paging::span::pages_spanned(gva, span, page_size);
        let gpas = crate::runtime::gva_mem::task_gva_page_gpas(
            host, tasks, task_id, gva, span, page_shift,
        );
        (gpas.len() as u64 == want).then_some(gpas)
    });
}

/// [`settle_for_mapping`] for a caller that is **about to land the owed frame
/// itself**, over the same window, while preserving ranges the payment would
/// overwrite.
///
/// There is exactly one such caller and the distinction is not a nicety. A debt
/// is armed at a Store and names the resident that Store produced;
/// [`pay_for_mapping`] discharges it by writing that resident over the **whole**
/// window with no exclusions. `merge_guest_writes_into_pages` exists to put the
/// same resident into every page the guest did *not* write and keep the pages it
/// did — so paying first destroys exactly the bytes the merge was called to
/// preserve, one statement before the merge writes everything else back around
/// them. The guest's repaint is then gone and `t11sample_resident_merged`
/// reports success.
///
/// So the debt is **dropped**, not paid: what the caller is about to write is the
/// same surface's newer content at the same geometry — `write_bgra8_inner`
/// refuses on `GeometryMoved` if it is not — so the owed frame is superseded
/// rather than lost. The settle half still runs, because writes this device has
/// already submitted into these pages must land before the caller reads or
/// writes them, and that is true whoever is writing.
///
/// Counted, because a zero here says the two never co-occur on a workload and a
/// non-zero says how much guest painting the old order was throwing away.
pub fn supersede_for_mapping(
    state: &mut DeviceState,
    mapping_id: u32,
    site: crate::runtime::render_writeback::SettleSite,
) {
    if state.pending_writebacks.take(mapping_id).is_some() {
        crate::runtime::drain::note_store_route("wbdebt_superseded_by_skipping_write");
    }
    crate::runtime::render_writeback::settle_guest_writes(site);
}

/// [`settle_for_mapping`] for a caller that cannot name the mapping it is about
/// to touch, so it owes every debt.
pub fn settle_unnamed<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    site: crate::runtime::render_writeback::SettleSite,
) {
    pay_all(state, host);
    crate::runtime::render_writeback::settle_guest_writes(site);
}

/// Submit exactly the resources named by an asynchronous synchronize command.
///
/// The object list is the scope of the API operation; an unrelated host-valid
/// texture remains resident-authoritative. Completion belongs to the FIFO: the
/// transfers recorded here precede that packet's queue point, and its pending
/// stamp publishes only after that point completes. Waiting here would turn the
/// asynchronous command into a device-wide drain and then make the stamp wait a
/// second time for work already known complete.
pub fn submit_for_resources<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    object_ids: &[u32],
) {
    for &object_id in object_ids {
        pay_for_texture(state, host, task_id, object_id);
    }
}

/// Run the Store the debt stands for, now.
///
/// Everything the copy needs is resolved here and not at the arm — the identity
/// from the mapping's *current* generation, the page walk inside
/// `store_render_frame`. Two answers other than writing, and both release the
/// resident's `gpu_only_content` where they can, because that flag is what keeps
/// the reclaim off an image holding pixels nothing else has:
///
/// * **The guest superseded the frame.** `clear_host_valid` after the arm means
///   the guest wrote these pages itself, and landing an older frame on top of
///   its work is the write-ordering hazard `render_writeback`'s doc names
///   fourth. [`crate::runtime::resource_validity::licence_of`] is the existing
///   happens-before and it is read rather than re-derived.
/// * **The mapping's generation moved.** The guest remapped the surface, so the
///   identity this debt was armed under names a resident that is now an orphan,
///   and the pages it would be written into belong to something else. There is
///   no way to name that orphan from here — the current generation resolves to a
///   different identity — so its `gpu_only_content` outlives it and one image
///   leaks per occurrence. `wbdebt_generation_moved` is how a boot says how many;
///   a reading above single digits is the signal to carry the arm's whole
///   identity rather than the four integers that re-derive it.
#[cfg(feature = "backend-vulkan")]
fn pay<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    debt: WritebackDebt,
    route: &'static str,
) {
    let Some(entry) = state.mappings.get(&mapping_id) else {
        crate::runtime::drain::note_store_route("wbdebt_generation_moved");
        return;
    };
    let (map_generation, validity) = (entry.map_generation, entry.validity);
    if map_generation != debt.map_generation {
        crate::runtime::drain::note_store_route("wbdebt_generation_moved");
        return;
    }
    // The resident the draw registered, not the one a fresh derivation would
    // name today. See `WritebackDebt::identity`.
    let identity = debt.identity;
    if crate::runtime::resource_validity::licence_of(validity)
        == crate::runtime::resource_validity::WritebackLicence::Superseded
    {
        crate::runtime::drain::note_store_route("wbdebt_abandoned_guest_wrote");
        crate::backend::vulkan::engine::note_resident_content_copied_out(&identity);
        return;
    }
    crate::runtime::drain::note_store_route(route);
    if !crate::runtime::render_writeback::store_render_frame(
        state,
        host,
        mapping_id,
        &identity,
        debt.width,
        debt.height,
    ) {
        // `store_render_frame` reports its own loss on the failure channel; this
        // names the rail that owed it, because a debt paid late and refused is a
        // different investigation from a Store refused where it was issued.
        crate::observe::fail(format!(
            "wbdebt_pay_lost mapping={mapping_id} {}x{} reason=store_refused",
            debt.width, debt.height
        ));
        crate::backend::vulkan::engine::note_resident_content_copied_out(&identity);
    }
    crate::runtime::mapper::stamp_guest_write_gen(state, host, mapping_id);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GvaPaySite {
    Named,
    All,
}

#[cfg(feature = "backend-vulkan")]
impl GvaPaySite {
    fn route(self) -> &'static str {
        match self {
            Self::Named => "gvadebt_paid_named",
            Self::All => "gvadebt_paid_all",
        }
    }
}

/// The plane-relative byte ranges the guest CPU owns, from the pages the
/// hypervisor reported it wrote.
///
/// `ordered` is the plane's own page list in plane order, so page `i` is the
/// bytes at `[i * page_size, (i + 1) * page_size)` — the same identity
/// `StoreTargetPages::from_ordered` builds the destination from, which is why
/// this cannot be derived from a GPA alone: the same physical page may appear at
/// more than one offset of a plane, and every appearance is the guest's.
///
/// The result is ascending, disjoint, and clamped to `span`, which is what
/// [`crate::runtime::mapping_write::SkipRanges`] promises its readers.
///
/// Gated the way [`PendingWritebacks::gva_resource_backing`] is: only the Vulkan
/// arm arms a GVA debt, so the Metal build has no caller — but the relation is
/// plain arithmetic over a page list and its tests are worth running on every
/// arm.
#[cfg(any(feature = "backend-vulkan", test))]
fn plane_offsets_of_pages(
    ordered: &[u64],
    page_size: u64,
    span: u64,
    written: &[u64],
) -> Vec<(u64, u64)> {
    let mut out: Vec<(u64, u64)> = Vec::new();
    for (i, gpa) in ordered.iter().enumerate() {
        if !written.contains(gpa) {
            continue;
        }
        let from = (i as u64).saturating_mul(page_size);
        if from >= span {
            continue;
        }
        let to = from.saturating_add(page_size).min(span);
        match out.last_mut() {
            // Adjacent pages coalesce, so a whole-plane write is one range
            // rather than one per page and the row walk below stays cheap.
            Some(last) if last.1 == from => last.1 = to,
            _ => out.push((from, to)),
        }
    }
    out
}

/// Where the guest CPU wrote inside the plane this debt owes a frame to, or
/// `None` when the host cannot say.
///
/// `None` covers every unknown, and each one means the same thing to the caller:
/// the guest declared a write, this device cannot name the region, so the
/// guest's pages keep what they hold. Serving the device's frame over an
/// unnamed guest write is the one answer that destroys work nothing can
/// recover.
///
/// An empty report is an unknown and not a finding. The hypervisor harvests at
/// its own points, so "no page of this target has moved" from a harvest that
/// has not yet run since the guest's store is indistinguishable from a guest
/// that wrote nothing — and the guest's own declaration already said it wrote.
/// Two witnesses disagreeing is not a licence to pick the convenient one.
#[cfg(feature = "backend-vulkan")]
fn guest_owned_plane_ranges<M: HostOps>(
    state: &DeviceState,
    host: &M,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    ordered: &[u64],
    span: u64,
) -> Option<Vec<(u64, u64)>> {
    let Some(key) = crate::runtime::gva_store_witness::GvaTargetKey::of(identity) else {
        crate::runtime::drain::note_store_route("gvadebt_merge_no_key");
        return None;
    };
    let written = crate::runtime::gva_store_witness::written_pages(state, host, key)?;
    if written.is_empty() {
        crate::runtime::drain::note_store_route("gvadebt_merge_no_pages");
        return None;
    }
    let ranges = plane_offsets_of_pages(ordered, state.page_size(), span, &written);
    if ranges.is_empty() {
        crate::runtime::drain::note_store_route("gvadebt_merge_no_offsets");
        return None;
    }
    Some(ranges)
}

/// Materialize one host-authoritative GVA resource into its retained transfer
/// backing. After explicit discard, synchronize lazily recreates that backing;
/// ordinary virtual-memory unmap does not participate in resource lifetime.
#[cfg(feature = "backend-vulkan")]
fn pay_gva<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    plane: GvaPlaneKey,
    debt: GvaWritebackDebt,
    site: GvaPaySite,
) -> bool {
    let key = plane.resource;
    let identity = gva_identity(debt);
    // Whether the guest has declared a CPU write to this resource since the
    // Store. It is not yet a verdict: the declaration is one resource-wide bit
    // (`shouldInvalidateHost`, a `lock btr` of the object's dirty flag) and the
    // API relation it stands for is per region. Where the guest wrote is
    // resolved below, once the plane's page list is back, because that is the
    // only coordinate system the hypervisor's answer arrives in.
    let guest_declared_write = !state
        .buffer_write_gen
        .stamp(key.task_id, key.texture_ref)
        .quiet_since(debt.guest_write);
    let Some(span) = u64::from(debt.row_stride).checked_mul(u64::from(debt.height)) else {
        crate::observe::fail(format!(
            "gvadebt_pay_lost task={} texture={} reason=span_overflow",
            key.task_id, key.texture_ref
        ));
        release_gva(debt);
        return true;
    };
    // The resource's own declaration decides whether its pages come back, not
    // this debt's — see [`reback_gva_resource`]. A debt whose resource is gone
    // names storage that object no longer owns, so it is released here rather
    // than restored: restoring one would park it in the ledger forever, since
    // nothing retired can grow pages back.
    if !reback_gva_resource(state, host, plane) {
        crate::runtime::drain::note_store_route("gvadebt_resource_retired");
        release_gva(debt);
        return true;
    }
    let Some((backing_generation, backing_span, ordered)) =
        state.pending_writebacks.gva_resource_backing(plane)
    else {
        state.pending_writebacks.restore_gva(plane, debt);
        crate::runtime::drain::note_store_route(match site {
            GvaPaySite::Named => "gvadebt_named_unmapped",
            GvaPaySite::All => "gvadebt_all_unmapped",
        });
        if site == GvaPaySite::Named {
            crate::observe::fail(format!(
                "gvadebt_pay_blocked task={} texture={} reason=span_unresolved",
                key.task_id, key.texture_ref
            ));
        }
        return false;
    };
    // The plane key already carries the address, so a mismatched one cannot
    // reach here: it would have found no plane at all above.
    if backing_generation != debt.generation || backing_span != span {
        crate::runtime::drain::note_store_route("gvadebt_generation_moved");
        release_gva(debt);
        return true;
    }
    // The third answer. Writing the whole frame over a plane the guest CPU wrote
    // part of loses the guest's stores; dropping the frame loses everything the
    // GPU rendered, which is the whole layer minus the guest's rectangle. Only
    // the hypervisor's per-page report separates them, and when it cannot the
    // guest's bytes win — the same direction every other consumer of that
    // witness fails in.
    let skip = if guest_declared_write {
        match guest_owned_plane_ranges(state, host, &identity, &ordered, span) {
            Some(ranges) => {
                crate::runtime::drain::note_store_route("gvadebt_merged_guest_wrote");
                ranges
            }
            None => {
                crate::runtime::drain::note_store_route("gvadebt_abandoned_guest_wrote");
                release_gva(debt);
                return true;
            }
        }
    } else {
        Vec::new()
    };
    let pages = crate::runtime::draw::StoreTargetPages::from_ordered(&ordered, span);
    let request = crate::runtime::draw::ColorRtRequest {
        texture_ref: key.texture_ref,
        target_gva: debt.gva,
        row_stride: debt.row_stride,
        width: debt.width,
        height: debt.height,
        format: debt.format,
        store_action: crate::contract::pass_action::MTL_STORE_ACTION_STORE,
        ..Default::default()
    };
    crate::runtime::drain::note_store_route(site.route());
    if let Err(reason) = crate::runtime::render_writeback::store_gva_frame(
        state,
        host,
        key.task_id,
        &identity,
        &request,
        key.texture_ref,
        Some(&pages),
        &skip,
    ) {
        // Through the builder rather than by interpolating the decline, which
        // renders its own `reason=` and produced `reason=reason=<slug>` — a line
        // the standard ranking grep drops. The builder also carries the
        // decline's own fields, so the `via=` that says which check inside the
        // store refused now reaches the log instead of being formatted away.
        crate::observe::Emit::decline("gvadebt_pay_lost", &reason)
            .field("task", key.task_id)
            .field("texture", key.texture_ref)
            .fail();
        release_gva(debt);
    }
    true
}

/// [`pay`] on an arm with no Vulkan engine to owe a frame to.
///
/// Unreachable rather than merely unused: the only arm site is the type-11
/// surface Store in `draw::vulkan`, so the ledger is empty on this arm and both
/// callers return at their emptiness check before reaching here. It exists so
/// the reader-side helpers can be one set of functions on both arms instead of
/// two spellings the settle sites would have to choose between.
#[cfg(not(feature = "backend-vulkan"))]
fn pay<M: HostMemory + HostOps>(
    _state: &mut DeviceState,
    _host: &mut M,
    _mapping_id: u32,
    _debt: WritebackDebt,
    _route: &'static str,
) {
}

#[cfg(not(feature = "backend-vulkan"))]
fn pay_gva<M: HostMemory + HostOps>(
    _state: &mut DeviceState,
    _host: &mut M,
    _plane: GvaPlaneKey,
    _debt: GvaWritebackDebt,
    _site: GvaPaySite,
) -> bool {
    true
}

#[cfg(test)]
mod tests {

    /// A synthetic resident identity for a mapping, so the ledger's tests can
    /// arm a debt without a device. The surface namespace and the mapping id
    /// are what the ledger keys on; the rest is only carried.
    use super::test_resident_identity as ident;
    use super::*;

    /// The coalescing the rail exists for, at the container: a second arm into
    /// one mapping replaces the first rather than queueing beside it, so N
    /// Stores between two reads cost one copy and not N.
    /// The ledger hands back the resident the frame is **in**, verbatim, even
    /// when the mapping's own generation has moved past it.
    ///
    /// This is the defect that lost every Apple Maps frame under
    /// `REIMS_VGPU_SHARED_TARGET=off`, and it is the whole of it: `pay` rebuilt
    /// the identity from `present_identity::surface_identity`, which reads the
    /// mapping's generation *now*, while the draw had registered its resident
    /// under the generation current when the stream started. A driven boot read
    /// `read_target_unknown_identity diverges=generation asked_gen=N held_gen=N-1`
    /// and refused the Store with the resident holding the pixels sitting in the
    /// registry.
    ///
    /// So the identity is armed and taken and never derived, and the divergence
    /// below is deliberate: `map_generation` is 9 and the resident is at 8,
    /// which is exactly the state the boot was in.
    #[test]
    #[cfg(feature = "backend-vulkan")]
    fn a_debt_remembers_the_resident_it_was_armed_with_and_not_the_mapping() {
        let mut pending = PendingWritebacks::default();
        let resident = ident(7, 1920, 1080, 8);
        assert_eq!(pending.arm(7, resident.clone(), 1920, 1080, 9), None);
        let debt = pending.take(7).expect("the debt was armed");
        assert_eq!(
            debt.identity, resident,
            "the payment would read a resident the draw never wrote"
        );
        assert_eq!(
            debt.identity.generation(),
            8,
            "the resident's generation, not the mapping's"
        );
        assert_eq!(
            debt.map_generation, 9,
            "the destination guard still watches the mapping"
        );
    }

    #[test]
    fn a_second_arm_into_one_mapping_replaces_the_first() {
        let mut pending = PendingWritebacks::default();
        assert_eq!(pending.arm(7, ident(7, 1920, 1080, 3), 1920, 1080, 3), None);
        assert_eq!(pending.arm(7, ident(7, 1920, 1080, 3), 1920, 1080, 3), None);
        assert_eq!(pending.len(), 1, "one mapping owes one frame");
        let debt = pending.take(7).expect("mapping 7 owes a frame");
        assert_eq!(debt.seq, 1, "the later Store is the one owed");
        assert!(pending.is_empty());
    }

    /// Geometry travels with the debt, because the payment writes at the
    /// geometry the Store was taken at and the mapping may have been re-declared
    /// since.
    #[test]
    fn a_debt_carries_the_geometry_its_store_was_taken_at() {
        let mut pending = PendingWritebacks::default();
        assert_eq!(pending.arm(4, ident(4, 800, 600, 11), 800, 600, 11), None);
        let debt = pending.get(4).expect("mapping 4 owes a frame");
        assert_eq!(
            (debt.width, debt.height, debt.map_generation),
            (800, 600, 11)
        );
    }

    /// The bound is the container's, and it hands the caller the mapping that
    /// has to be paid rather than dropping a frame to stay under it.
    #[test]
    fn arming_past_the_bound_evicts_the_oldest_and_says_so() {
        let mut pending = PendingWritebacks::default();
        for id in 0..MAX_DEBTS as u32 {
            assert_eq!(
                pending.arm(id, ident(id, 64, 64, 1), 64, 64, 1),
                None,
                "under the bound"
            );
        }
        assert_eq!(pending.len(), MAX_DEBTS);
        let evicted = pending.arm(
            MAX_DEBTS as u32,
            ident(MAX_DEBTS as u32, 64, 64, 1),
            64,
            64,
            1,
        );
        assert_eq!(
            evicted,
            Some(WritebackKey::Mapping(0)),
            "the oldest arm is the one handed back"
        );
        assert_eq!(
            pending.len(),
            MAX_DEBTS + 1,
            "the named debt is still owed until the caller pays it"
        );
        assert!(
            pending.take(0).is_some(),
            "and paying it is what brings the ledger back under the bound"
        );
        assert_eq!(pending.len(), MAX_DEBTS);
    }

    /// Re-arming a mapping already at the head of a full ledger must not evict:
    /// it is a replacement and the entry count does not grow.
    #[test]
    fn re_arming_a_held_mapping_never_evicts() {
        let mut pending = PendingWritebacks::default();
        for id in 0..MAX_DEBTS as u32 {
            assert_eq!(pending.arm(id, ident(id, 64, 64, 1), 64, 64, 1), None);
        }
        assert_eq!(
            pending.arm(0, ident(0, 64, 64, 1), 64, 64, 1),
            None,
            "a replacement makes no room"
        );
        assert_eq!(pending.len(), MAX_DEBTS);
    }

    /// Age order is arm order and not mapping id, because `pay_all` walks it and
    /// the oldest owed frame is the one most likely to be read.
    #[test]
    fn mappings_come_back_in_arm_order() {
        let mut pending = PendingWritebacks::default();
        assert_eq!(pending.arm(9, ident(9, 1, 1, 1), 1, 1, 1), None);
        assert_eq!(pending.arm(2, ident(2, 1, 1, 1), 1, 1, 1), None);
        assert_eq!(pending.arm(5, ident(5, 1, 1, 1), 1, 1, 1), None);
        assert_eq!(pending.mappings_by_age(), vec![9, 2, 5]);
    }

    fn gva_debt(generation: u64) -> GvaWritebackDebt {
        GvaWritebackDebt {
            gva: 0x4000,
            row_stride: 256,
            width: 64,
            height: 64,
            format: crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM,
            generation,
            guest_write: Default::default(),
            seq: 0,
        }
    }

    /// The resource reference, not the GVA, owns coherence. Reusing the same
    /// resource for another Store replaces its debt exactly as repeated Stores
    /// into one IOSurface do.
    #[test]
    fn a_second_gva_store_on_one_resource_replaces_the_first() {
        let mut pending = PendingWritebacks::default();
        let key = GvaResourceKey {
            task_id: 3,
            texture_ref: 19,
        };
        assert_eq!(pending.arm_gva(key, gva_debt(7)), None);
        let previous = pending.arm_gva(key, gva_debt(8));
        assert_eq!(previous.map(|debt| debt.generation), Some(7));
        assert_eq!(pending.len(), 1);
        assert_eq!(pending.get_gva(key).map(|debt| debt.generation), Some(8));
    }

    /// GVA resources have protocol lifetime, not an arbitrary ledger capacity.
    /// Holding more than the anonymous-surface coalescing bound must not invent
    /// a transfer or drop an older resource's host-authoritative frame.
    #[test]
    fn gva_resources_are_not_evicted_by_the_surface_debt_bound() {
        let mut pending = PendingWritebacks::default();
        for texture_ref in 1..=(MAX_DEBTS as u32 + 8) {
            let key = GvaResourceKey {
                task_id: 2,
                texture_ref,
            };
            pending.ensure_gva_resource(
                key,
                u64::from(texture_ref) << 16,
                4096,
                Some(vec![u64::from(texture_ref) << 12]),
            );
            assert_eq!(pending.arm_gva(key, gva_debt(texture_ref.into())), None);
        }
        assert_eq!(pending.len(), MAX_DEBTS + 8);
        assert_eq!(pending.gvas_by_age().len(), MAX_DEBTS + 8);
    }

    /// Ordinary virtual-memory bookkeeping does not retarget a live resource.
    /// A repeated prepare with a different walk keeps the original transfer
    /// backing until the protocol explicitly discards it.
    #[test]
    fn a_live_resource_retains_its_backing_until_discard() {
        let mut pending = PendingWritebacks::default();
        let key = GvaResourceKey {
            task_id: 3,
            texture_ref: 19,
        };
        let generation = pending.ensure_gva_resource(key, 0x4000, 4096, Some(vec![0x9000]));
        assert_eq!(
            pending.ensure_gva_resource(key, 0x4000, 4096, Some(vec![0xa000])),
            generation
        );
        assert_eq!(
            &*pending.gva_resource_backing(key.plane(0x4000)).unwrap().2,
            &[0x9000]
        );

        assert_eq!(pending.discard_gva_resources(3, &[19]), 1);
        assert!(pending.gva_resource_backing(key.plane(0x4000)).is_none());
        assert_eq!(
            pending.ensure_gva_resource(key, 0x4000, 4096, Some(vec![0xa000])),
            generation,
            "discard replaces the transfer backing, not the host texture"
        );
        assert_eq!(
            &*pending.gva_resource_backing(key.plane(0x4000)).unwrap().2,
            &[0xa000]
        );
    }

    /// Delete is the resource lifetime boundary. Reusing the same task-local
    /// reference after delete receives a new host-texture identity.
    #[test]
    fn deleting_and_recreating_a_resource_changes_its_generation() {
        let mut pending = PendingWritebacks::default();
        let key = GvaResourceKey {
            task_id: 3,
            texture_ref: 19,
        };
        let first = pending.ensure_gva_resource(key, 0x4000, 4096, Some(vec![0x9000]));
        assert!(pending.retire_gva_resource(key).0);
        let second = pending.ensure_gva_resource(key, 0x4000, 4096, Some(vec![0xa000]));
        assert_ne!(first, second);
    }

    /// Delete is the *announced* lifetime boundary; a plane's length moving at
    /// one address is the same boundary observed instead of announced. A
    /// plane's length is fixed for its life, so this is a different object in a
    /// reused slot and it must get a different host texture.
    ///
    /// Asserting the third call is what makes that visible: a fix that only
    /// stopped refusing, without replacing the entry, still fails here.
    #[test]
    fn one_plane_redeclared_at_a_new_length_is_a_new_resource() {
        let mut pending = PendingWritebacks::default();
        let key = GvaResourceKey {
            task_id: 3,
            texture_ref: 19,
        };
        let first = pending.ensure_gva_resource(key, 0x4000, 4096, Some(vec![0x9000]));
        let second = pending.ensure_gva_resource(key, 0x4000, 8192, Some(vec![0xa000, 0xb000]));
        assert_ne!(first, second, "a new length is a new host texture");
        assert_eq!(
            &*pending.gva_resource_backing(key.plane(0x4000)).unwrap().2,
            &[0xa000, 0xb000],
            "the new object's pages replace the retired one's"
        );
        assert_eq!(
            pending.ensure_gva_resource(key, 0x4000, 8192, None),
            second,
            "the new declaration is the live one, so it is stable"
        );
    }

    /// A mip pyramid is one resource with several live planes, and the ledger
    /// has to hold all of them at once.
    ///
    /// Measured on a driven macos-26 boot: one reference cycling three
    /// contiguous declarations in exact 4:1 ratios — 256x192, 128x96, 64x48 of
    /// one RGBA8 allocation, the compositor's blur/backdrop pyramid. Keyed by
    /// the reference, each level change replaced the entry, so no level's
    /// resident could ever be reused and arming one level's Store dropped the
    /// previous level's unpaid frame. Both halves are asserted here: the
    /// generations are distinct **and** stable, and three debts coexist.
    #[test]
    fn the_levels_of_one_pyramid_are_separate_planes_of_one_resource() {
        let mut pending = PendingWritebacks::default();
        let key = GvaResourceKey {
            task_id: 1,
            texture_ref: 135,
        };
        let levels = [
            (0x11af000_u64, 196_608_u64),
            (0x11df000, 49_152),
            (0x11eb000, 12_288),
        ];
        let generations: Vec<u64> = levels
            .iter()
            .map(|&(gva, span)| pending.ensure_gva_resource(key, gva, span, Some(vec![gva])))
            .collect();
        let mut distinct = generations.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), 3, "each level is its own host texture");

        // The cycle the boot showed: re-declaring level 0 after levels 1 and 2
        // must return level 0's own generation, not mint a fourth.
        for (i, &(gva, span)) in levels.iter().enumerate() {
            assert_eq!(
                pending.ensure_gva_resource(key, gva, span, None),
                generations[i],
                "a live plane is stable across its siblings"
            );
        }

        for (i, &(gva, _)) in levels.iter().enumerate() {
            let mut debt = gva_debt(generations[i]);
            debt.gva = gva;
            assert_eq!(
                pending.arm_gva(key, debt),
                None,
                "arming one level must not supersede another"
            );
        }
        assert_eq!(
            pending.take_gva(key).len(),
            3,
            "the resource owes all three"
        );
    }

    /// The plane's own page order decides the offsets, not the GPA order.
    ///
    /// A page list is what a plane *is*: the same physical page may appear at
    /// more than one offset, and a writeback that resolved offsets by sorting
    /// GPAs would skip the wrong bytes of a plane whose pages the guest
    /// allocated out of order — which is every plane, since guest RAM is not
    /// handed out contiguously.
    #[test]
    fn the_guests_pages_map_to_their_own_offsets_in_the_plane() {
        const P: u64 = 0x1000;
        // Descending GPAs, so an implementation that sorted them would produce
        // the reverse of this.
        let ordered = [9 * P, 4 * P, 7 * P, 2 * P];
        let span = 4 * P;
        assert_eq!(
            plane_offsets_of_pages(&ordered, P, span, &[7 * P]),
            vec![(2 * P, 3 * P)],
            "page 7 is the plane's third page"
        );
        // Plane-adjacent pages coalesce; plane-separated ones do not.
        assert_eq!(
            plane_offsets_of_pages(&ordered, P, span, &[4 * P, 7 * P]),
            vec![(P, 3 * P)]
        );
        assert_eq!(
            plane_offsets_of_pages(&ordered, P, span, &[9 * P, 7 * P]),
            vec![(0, P), (2 * P, 3 * P)]
        );
        // A page the plane does not own is not the plane's.
        assert_eq!(plane_offsets_of_pages(&ordered, P, span, &[5 * P]), vec![]);
    }

    /// The last page of a plane whose span ends inside it is clamped, because
    /// the skip list is read against the frame's own bytes and a range past the
    /// end would exclude bytes that do not exist.
    #[test]
    fn the_last_page_of_a_plane_is_clamped_to_its_span() {
        const P: u64 = 0x1000;
        let ordered = [3 * P, 8 * P];
        assert_eq!(
            plane_offsets_of_pages(&ordered, P, P + 64, &[8 * P]),
            vec![(P, P + 64)]
        );
        // A page wholly past the span belongs to no byte of the frame.
        assert_eq!(plane_offsets_of_pages(&ordered, P, P, &[8 * P]), vec![]);
    }

    /// A Store may not be deferred while this device cannot say what the guest
    /// writes to the plane in the meantime.
    ///
    /// The writer's rule, and stricter than the reader's. A deferred frame lives
    /// only in a resident, and the sole recovery from a guest CPU write into its
    /// pages is to land it everywhere the guest did not touch — which needs the
    /// hypervisor's per-page report. Without one, a single guest write anywhere
    /// in the plane deletes the whole frame, and on a compositing layer that is
    /// every pixel the GPU rendered outside the guest's own rectangle.
    ///
    /// Both unwitnessed shapes are asked, because they arrive from opposite
    /// directions: a host with no dirty bitmap at all, and the product host's
    /// arming window, in which a freshly tracked set reads its generation back
    /// as 0 until a harvest has run. The second is the one that bit — on the
    /// macos-15 battery every frame the ledger lost to a guest write was a
    /// first Store into a fresh plane.
    #[cfg(feature = "backend-vulkan")]
    #[test]
    fn an_unwitnessed_gva_store_is_not_deferred() {
        fn arm(host: &mut crate::runtime::FakeHost) -> (bool, DeviceState) {
            let mut state = DeviceState::new(crate::model::DeviceId::default(), 12);
            let debt = gva_debt(9);
            let identity = gva_identity(debt);
            let key = GvaResourceKey {
                task_id: 3,
                texture_ref: 12,
            };
            let span = u64::from(debt.row_stride) * u64::from(debt.height);
            let pages: Vec<u64> = (0..span.div_ceil(state.page_size()))
                .map(|i| 0x40_000 + i * state.page_size())
                .collect();
            let _ = state
                .pending_writebacks
                .ensure_gva_resource(key, debt.gva, span, Some(pages));
            let c0 = crate::runtime::draw::ColorRtRequest {
                texture_ref: key.texture_ref,
                target_gva: debt.gva,
                row_stride: debt.row_stride,
                width: debt.width,
                height: debt.height,
                format: debt.format,
                ..Default::default()
            };
            let armed = arm_gva(&mut state, host, key.task_id, &c0, &identity);
            (armed, state)
        }

        let mut blind = crate::runtime::FakeHost::new();
        blind.guest_writes_unobservable = true;
        let (armed, state) = arm(&mut blind);
        assert!(
            !armed,
            "a host with no dirty bitmap cannot recover a deferral"
        );
        assert_eq!(
            state.pending_writebacks.len(),
            0,
            "a refused deferral must leave no debt for the caller's eager Store to fight with"
        );

        let mut waking = crate::runtime::FakeHost::new();
        waking.guest_write_startup_window = true;
        let (armed, state) = arm(&mut waking);
        assert!(
            !armed,
            "a set still inside its arming window cannot date a frame"
        );
        assert_eq!(state.pending_writebacks.len(), 0);

        // The same plane once the host can answer: deferral is licensed again,
        // so this refusal is a gate and not a disabled rail.
        let mut ready = crate::runtime::FakeHost::new();
        let (armed, state) = arm(&mut ready);
        assert!(armed, "a witnessed plane still defers");
        assert_eq!(state.pending_writebacks.len(), 1);
    }

    /// A guest CPU write into part of a plane names that part, so the payment
    /// can keep both writers.
    ///
    /// The relation `cpu_write_after_render` asks for, at the seam that decides
    /// it. Before this, `pay_gva` had one answer to a guest write — release the
    /// debt — and a guest that wrote one page of a layer lost every pixel the
    /// GPU had rendered into the rest of it.
    #[cfg(feature = "backend-vulkan")]
    #[test]
    fn a_partial_guest_write_names_the_pages_it_owns() {
        let mut state = DeviceState::new(crate::model::DeviceId::default(), 12);
        let mut host = crate::runtime::FakeHost::new();
        let page = state.page_size();
        let ordered: Vec<u64> = (0..4).map(|i| (0x40 + i) * page).collect();
        let span = 4 * page;
        let debt = gva_debt(9);
        let identity = gva_identity(debt);
        let key = crate::runtime::gva_store_witness::GvaTargetKey::of(&identity)
            .expect("a GVA identity names a witness target");

        // Nothing armed: the host cannot name a page, so the guest keeps
        // everything — the direction every unknown answers in.
        assert_eq!(
            guest_owned_plane_ranges(&state, &host, &identity, &ordered, span),
            None,
            "an unstamped target has no extent to report"
        );

        crate::runtime::gva_store_witness::note_store(&mut state, &mut host, key, &ordered);
        // Still nothing written since the Store. That is not a licence either:
        // the guest's own declaration is what brought the caller here, and a
        // harvest that has not run yet reports the same empty list.
        assert_eq!(
            guest_owned_plane_ranges(&state, &host, &identity, &ordered, span),
            None,
            "an empty report is an unknown, not a finding"
        );

        host.guest_wrote_page(ordered[2] + 8);
        assert_eq!(
            guest_owned_plane_ranges(&state, &host, &identity, &ordered, span),
            Some(vec![(2 * page, 3 * page)]),
            "only the page the guest wrote is the guest's"
        );
    }

    /// A guest validity transition after the Store makes guest memory newer
    /// than the held resident. The debt remains available for an orderly
    /// abandon, but it must immediately stop licensing host-resident reads.
    #[cfg(feature = "backend-vulkan")]
    #[test]
    fn a_guest_write_revokes_gva_resident_authority() {
        let mut state = DeviceState::new(crate::model::DeviceId::default(), 12);
        let key = GvaResourceKey {
            task_id: 4,
            texture_ref: 12,
        };
        let debt = gva_debt(99);
        let _ = state.pending_writebacks.arm_gva(key, debt);
        let identity = gva_identity(debt);
        assert!(gva_resident_authoritative(&state, &identity));
        state
            .buffer_write_gen
            .note_write(key.task_id, key.texture_ref);
        assert!(!gva_resident_authoritative(&state, &identity));
        assert!(state.pending_writebacks.get_gva(key).is_some());
    }

    /// "This texture owes nothing" splits into a surface with no debt and a
    /// reference that named no surface at all.
    ///
    /// The second is not a reading about the ledger — it is a reading about the
    /// lookup. No spelling named a mapping this device holds, so a debt owed by
    /// the surface behind that reference could not have been found whether or
    /// not one existed, and a sampled bind proceeding past it reads the guest's
    /// pages while the newest frame is still in a resident. One counter reported
    /// both at 1.1 M a boot, which is every sampled guest import, and that
    /// volume is what made it read as a healthy zero.
    ///
    /// The middle case is the one this test exists for. A reference that **is**
    /// its own mapping id resolves perfectly and never populates
    /// `texture_to_mapping`, so a split that asked only that registration
    /// reported it as "we could not look" — and since that is the dominant
    /// spelling, the split read 100 % `_unresolved` on both arms of a driven
    /// boot and could not have read anything else.
    #[test]
    fn a_texture_that_owes_nothing_says_whether_it_named_a_surface_at_all() {
        use crate::runtime::drain::store_route_count;

        // Baselines rather than a clear: these counters are process-global and
        // another test in this binary may have moved them.
        let resolved0 = store_route_count("wbdebt_texture_owes_nothing_resolved");
        let unresolved0 = store_route_count("wbdebt_texture_owes_nothing_unresolved");
        let total0 = store_route_count("wbdebt_texture_owes_nothing");

        let mut state = DeviceState::new(crate::model::DeviceId::default(), 12);
        let mut host = crate::runtime::FakeHost::new();
        // The ledger has to be non-empty or `pay_for_texture` returns at its
        // emptiness check and neither counter is reached. Mapping 7 owes; the
        // three references below are about other surfaces.
        assert_eq!(
            state
                .pending_writebacks
                .arm(7, ident(7, 64, 64, 1), 64, 64, 1),
            None
        );
        // Reference 21 names mapping 9 through the per-task registration, and
        // this device holds mapping 9. It owes nothing.
        state.mappings.entry(9).or_default().mapped = true;
        state.texture_to_mapping.insert((1, 21), 9);
        pay_for_texture(&mut state, &mut host, 1, 21);
        assert_eq!(
            store_route_count("wbdebt_texture_owes_nothing_resolved") - resolved0,
            1
        );
        assert_eq!(
            store_route_count("wbdebt_texture_owes_nothing_unresolved") - unresolved0,
            0
        );

        // Reference 30 *is* a mapping this device holds. It owes nothing, and it
        // resolves — through the spelling that never touches the registration.
        state.mappings.entry(30).or_default().mapped = true;
        pay_for_texture(&mut state, &mut host, 1, 30);
        assert_eq!(
            store_route_count("wbdebt_texture_owes_nothing_resolved") - resolved0,
            2,
            "a reference that is its own mapping id has resolved"
        );
        assert_eq!(
            store_route_count("wbdebt_texture_owes_nothing_unresolved") - unresolved0,
            0
        );

        // Reference 22 names nothing: not a mapping this device holds, and no
        // `texture_to_mapping` entry.
        pay_for_texture(&mut state, &mut host, 1, 22);
        assert_eq!(
            store_route_count("wbdebt_texture_owes_nothing_resolved") - resolved0,
            2
        );
        assert_eq!(
            store_route_count("wbdebt_texture_owes_nothing_unresolved") - unresolved0,
            1
        );
        assert_eq!(
            store_route_count("wbdebt_texture_owes_nothing") - total0,
            3,
            "the split has to add up to the total it divides"
        );
    }

    /// A synchronize list is a scope, not merely a trigger. Publishing one
    /// object must leave an unrelated resource host-authoritative.
    #[test]
    fn asynchronous_resource_synchronization_submits_only_named_objects() {
        let mut state = DeviceState::new(crate::model::DeviceId::default(), 12);
        let mut host = crate::runtime::FakeHost::new();
        assert_eq!(
            state
                .pending_writebacks
                .arm(7, ident(7, 64, 64, 1), 64, 64, 1),
            None
        );
        assert_eq!(
            state
                .pending_writebacks
                .arm(8, ident(8, 64, 64, 1), 64, 64, 1),
            None
        );
        submit_for_resources(&mut state, &mut host, 1, &[7]);
        assert!(state.pending_writebacks.get(7).is_none());
        assert!(state.pending_writebacks.get(8).is_some());
    }
}
