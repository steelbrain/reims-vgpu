//! Backend-independent content observations which precede or outlive execution.
//!
//! These ledgers hold semantic ownership state. They deliberately contain no
//! host transfer code and emit no observations; orchestration records those at
//! the boundary where a transition is requested.

use std::collections::{BTreeMap, HashMap};
use std::ops::RangeInclusive;
use std::sync::Arc;

use reims_vgpu_protocol::{ContentVersion, ResourceId, ResourceObject};

/// A sampled guest-memory window in the namespace that owns its lifetime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GatherKey {
    TaskGva {
        task_id: u32,
        resource: ResourceId<ResourceObject>,
        gva: u64,
    },
    Mapping {
        mapping: reims_vgpu_protocol::MappingId,
        base_offset: reims_vgpu_protocol::ByteOffset,
    },
}

impl GatherKey {
    /// Stable diagnostic/cache name; uniqueness comes from the paired content generation.
    pub fn content_key(self) -> u64 {
        let mut hash = crate::fnv::FNV_OFFSET_BASIS;
        let mut fold = |value: u64| hash = crate::fnv::fold_u64(hash, value);
        match self {
            Self::TaskGva {
                task_id,
                resource,
                gva,
            } => {
                fold(1);
                fold(u64::from(task_id));
                fold(u64::from(resource.index()));
                fold(u64::from(resource.generation()));
                fold(gva);
            }
            Self::Mapping {
                mapping,
                base_offset,
            } => {
                fold(2);
                fold(u64::from(mapping.get()));
                fold(base_offset.get());
            }
        }
        hash
    }
}

/// Guest-declared generation in the identity space that owns a sampled window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatedGeneration {
    Mapping(u32),
    TaskResource(ResourceWriteStamp),
}

/// Why a page-exact write observation cannot call a window quiet.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostWriteVerdict {
    Quiet,
    Overlap,
    Unnamed,
}

impl HostWriteVerdict {
    pub fn wrote(self) -> bool {
        !matches!(self, Self::Quiet)
    }

    pub fn route(self) -> &'static str {
        match self {
            Self::Quiet => "gw_hw_quiet",
            Self::Overlap => "gw_hw_overlap",
            Self::Unnamed => "gw_hw_unnamed",
        }
    }
}

const EPOCH_CHUNK_BYTES: usize = 1usize << reims_vgpu_paging::resolve::X86_64.page_shift;
const EPOCHS_PER_CHUNK: usize = EPOCH_CHUNK_BYTES / std::mem::size_of::<u64>();
const _: () = assert!(EPOCHS_PER_CHUNK.is_power_of_two());

#[derive(Debug)]
struct EpochChunk {
    all_at: u64,
    cells: [u64; EPOCHS_PER_CHUNK],
}

impl Default for EpochChunk {
    fn default() -> Self {
        Self {
            all_at: 0,
            cells: [0; EPOCHS_PER_CHUNK],
        }
    }
}

#[derive(Debug, Default)]
struct PageEpochs {
    chunks: HashMap<u64, Box<EpochChunk>>,
    unnamed_at: u64,
}

impl PageEpochs {
    fn note_page_range(&mut self, mut page: u64, mut count: usize, epoch: u64) {
        while count != 0 {
            let chunk_key = page / EPOCHS_PER_CHUNK as u64;
            let slot = (page % EPOCHS_PER_CHUNK as u64) as usize;
            let take = count.min(EPOCHS_PER_CHUNK - slot);
            let chunk = self.chunks.entry(chunk_key).or_default();
            if slot == 0 && take == EPOCHS_PER_CHUNK {
                chunk.all_at = epoch;
            } else {
                chunk.cells[slot..slot + take].fill(epoch);
            }
            page += take as u64;
            count -= take;
        }
    }

    fn note_pages<I>(&mut self, pages: I, epoch: u64, page_shift: u32)
    where
        I: IntoIterator<Item = u64>,
    {
        for gpa in pages {
            let page = gpa >> page_shift;
            let chunk_key = page / EPOCHS_PER_CHUNK as u64;
            let slot = (page % EPOCHS_PER_CHUNK as u64) as usize;
            self.chunks.entry(chunk_key).or_default().cells[slot] = epoch;
        }
    }

    fn verdict(&self, since: u64, pages: &[u64], page_shift: u32) -> HostWriteVerdict {
        if since < self.unnamed_at {
            return HostWriteVerdict::Unnamed;
        }
        for &gpa in pages {
            let page = gpa >> page_shift;
            let chunk_key = page / EPOCHS_PER_CHUNK as u64;
            let slot = (page % EPOCHS_PER_CHUNK as u64) as usize;
            if self
                .chunks
                .get(&chunk_key)
                .is_some_and(|chunk| chunk.all_at > since || chunk.cells[slot] > since)
            {
                return HostWriteVerdict::Overlap;
            }
        }
        HostWriteVerdict::Quiet
    }
}

/// Exact page generations for writes the device makes into guest RAM.
#[derive(Debug)]
pub struct HostWrites {
    epoch: u64,
    page_shift: u32,
    pages: PageEpochs,
}

impl Default for HostWrites {
    fn default() -> Self {
        Self::new(reims_vgpu_paging::resolve::X86_64.page_shift)
    }
}

impl HostWrites {
    pub fn new(page_shift: u32) -> Self {
        Self {
            epoch: 0,
            page_shift,
            pages: PageEpochs::default(),
        }
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn note_mapping(&mut self, pages: Option<&[u64]>) {
        self.epoch = self.epoch.wrapping_add(1);
        match pages {
            Some(pages) => {
                self.pages
                    .note_pages(pages.iter().copied(), self.epoch, self.page_shift)
            }
            None => self.pages.unnamed_at = self.epoch,
        }
    }

    pub fn note_pages(&mut self, pages: Vec<u64>) {
        self.note_page_iter(pages);
    }

    pub fn note_page_iter<I>(&mut self, pages: I)
    where
        I: IntoIterator<Item = u64>,
    {
        self.epoch = self.epoch.wrapping_add(1);
        self.pages.note_pages(pages, self.epoch, self.page_shift);
    }

    /// Record page-number runs under this ledger's declared geometry.
    pub fn note_page_ranges<I>(&mut self, ranges: I)
    where
        I: IntoIterator<Item = (u64, usize)>,
    {
        self.epoch = self.epoch.wrapping_add(1);
        for (first_page, count) in ranges {
            self.pages.note_page_range(first_page, count, self.epoch);
        }
    }

    pub fn note_unknown(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        self.pages.unnamed_at = self.epoch;
    }

    pub fn wrote_any_since(&self, since: u64, pages: &[u64]) -> HostWriteVerdict {
        self.pages.verdict(since, pages, self.page_shift)
    }
}

/// One object's write generation before the canonical resource exists.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BufferWriteStamp {
    epoch: u64,
    generation: u64,
}

impl BufferWriteStamp {
    pub fn quiet_since(self, earlier: Self) -> bool {
        self.epoch == earlier.epoch && self.generation == earlier.generation
    }
}

/// Content currency recorded beside a derived copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceWriteStamp {
    Resolved {
        resource: ResourceId<ResourceObject>,
        version: ContentVersion,
    },
    Unresolved(BufferWriteStamp),
}

impl Default for ResourceWriteStamp {
    fn default() -> Self {
        Self::Unresolved(BufferWriteStamp::default())
    }
}

impl ResourceWriteStamp {
    pub fn quiet_since(self, earlier: Self) -> bool {
        match (self, earlier) {
            (
                Self::Resolved {
                    resource: now_resource,
                    version: now_version,
                },
                Self::Resolved {
                    resource: old_resource,
                    version: old_version,
                },
            ) => now_resource == old_resource && now_version == old_version,
            (Self::Unresolved(now), Self::Unresolved(old)) => now.quiet_since(old),
            _ => false,
        }
    }
}

/// Write generations in the guest's task-local pre-construction namespace.
#[derive(Default, Debug)]
pub struct BufferWriteGens {
    generations: HashMap<(u32, u32), u64>,
    epoch: u64,
}

impl BufferWriteGens {
    pub fn note_write(&mut self, task_id: u32, object_id: u32) {
        let generation = self.generations.entry((task_id, object_id)).or_insert(0);
        *generation = generation.wrapping_add(1);
    }

    pub fn stamp(&self, task_id: u32, object_id: u32) -> BufferWriteStamp {
        BufferWriteStamp {
            epoch: self.epoch,
            generation: self
                .generations
                .get(&(task_id, object_id))
                .copied()
                .unwrap_or(0),
        }
    }

    pub fn has_write(&self, task_id: u32, object_id: u32) -> bool {
        self.generations
            .get(&(task_id, object_id))
            .is_some_and(|generation| *generation != 0)
    }

    pub fn retire_object(&mut self, task_id: u32, object_id: u32) {
        if self.generations.remove(&(task_id, object_id)).is_some() {
            self.epoch = self.epoch.wrapping_add(1);
        }
    }

    pub fn retire_task(&mut self, task_id: u32) {
        let before = self.generations.len();
        self.generations.retain(|&(task, _), _| task != task_id);
        if self.generations.len() != before {
            self.epoch = self.epoch.wrapping_add(1);
        }
    }

    pub fn tracked(&self) -> usize {
        self.generations.len()
    }
}

/// One declared texture plane inside its parent allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearColorTarget {
    pub allocation_gva: u64,
    pub allocation_size: u64,
    pub plane_offset: u64,
    pub row_stride: u32,
}

impl LinearColorTarget {
    pub fn new(
        allocation_gva: u64,
        allocation_size: u64,
        plane_offset: u64,
        row_stride: u32,
    ) -> Option<Self> {
        if allocation_gva == 0
            || allocation_size == 0
            || row_stride == 0
            || plane_offset >= allocation_size
        {
            return None;
        }
        allocation_gva.checked_add(plane_offset)?;
        Some(Self {
            allocation_gva,
            allocation_size,
            plane_offset,
            row_stride,
        })
    }

    pub fn target_gva(&self) -> u64 {
        self.allocation_gva + self.plane_offset
    }

    pub fn whole(target_gva: u64, row_stride: u32, height: u32) -> Self {
        Self {
            allocation_gva: target_gva,
            allocation_size: u64::from(row_stride) * u64::from(height),
            plane_offset: 0,
            row_stride,
        }
    }
}

/// The generational resource lifetime that owns one GVA attachment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct GvaResourceKey {
    pub task_id: u32,
    pub resource: ResourceId<ResourceObject>,
}

/// One render plane of a GVA resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct GvaPlaneKey {
    pub resource: GvaResourceKey,
    pub gva: u64,
}

impl GvaResourceKey {
    pub fn plane(self, gva: u64) -> GvaPlaneKey {
        GvaPlaneKey {
            resource: self,
            gva,
        }
    }

    fn planes(self) -> RangeInclusive<GvaPlaneKey> {
        self.plane(0)..=self.plane(u64::MAX)
    }
}

/// A frame held only by a GVA target's executor-resident image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GvaWritebackDebt {
    pub linear: LinearColorTarget,
    pub width: u32,
    pub height: u32,
    pub format: u16,
    pub resident_layout: crate::pixel_format::TexelLayout,
    pub generation: u64,
    pub content: Option<(ResourceId<ResourceObject>, ContentVersion)>,
    pub guest_write: ResourceWriteStamp,
    pub seq: u64,
}

#[derive(Clone, Debug)]
struct GvaResourceState {
    generation: u64,
    span: u64,
    pages: Option<Arc<[u64]>>,
}

/// Semantic ownership of executor-authoritative GVA frames and their backing.
#[derive(Debug, Default)]
pub struct PendingWritebacks {
    debts: BTreeMap<GvaPlaneKey, GvaWritebackDebt>,
    resources: BTreeMap<GvaPlaneKey, GvaResourceState>,
    next_seq: u64,
    next_generation: u64,
}

/// The resource and executor-resident identity a GVA Store published.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct GvaTargetKey {
    pub task_id: u32,
    pub resource: ResourceId<ResourceObject>,
    pub gva: u64,
    pub generation: u64,
    pub width: u32,
    pub height: u32,
    pub bgra: bool,
}

impl GvaTargetKey {
    pub fn of(resource: GvaResourceKey, identity: &crate::TargetIdentity) -> Option<Self> {
        match *identity {
            crate::TargetIdentity::Gva {
                gva,
                width,
                height,
                generation,
                ..
            } if generation != 0 && gva != 0 => Some(Self {
                task_id: resource.task_id,
                resource: resource.resource,
                gva,
                generation,
                width,
                height,
                bgra: identity.is_bgra(),
            }),
            _ => None,
        }
    }
}

#[derive(Debug)]
struct GvaStoreEntry {
    pages: Vec<u64>,
    guest_write: ResourceWriteStamp,
    host_epoch: u64,
}

/// Why a copied GVA resident cannot stand in for its guest pages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GvaWriteReach {
    Quiet,
    NoEntry,
    GuestWrote,
    Host(HostWriteVerdict),
}

impl GvaWriteReach {
    pub fn route(self) -> &'static str {
        match self {
            Self::Quiet => "gvaw_quiet",
            Self::NoEntry => "gvaw_no_entry",
            Self::GuestWrote => "gvaw_guest_wrote",
            Self::Host(HostWriteVerdict::Quiet) => "gvaw_host_quiet",
            Self::Host(HostWriteVerdict::Overlap) => "gvaw_host_overlap",
            Self::Host(HostWriteVerdict::Unnamed) => "gvaw_host_unnamed",
        }
    }

    pub fn is_quiet(self) -> bool {
        matches!(self, Self::Quiet)
    }
}

/// Content witness for live named GVA targets.
#[derive(Default, Debug)]
pub struct GvaStoreWitness {
    entries: BTreeMap<GvaTargetKey, GvaStoreEntry>,
}

impl GvaStoreWitness {
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn retire_task(&mut self, task_id: u32) {
        self.entries.retain(|key, _| key.task_id != task_id);
    }

    pub fn retire_pages(&mut self, gone: &[u64]) {
        self.entries
            .retain(|_, entry| !entry.pages.iter().any(|page| gone.contains(page)));
    }

    pub fn note_store(
        &mut self,
        key: GvaTargetKey,
        pages: &[u64],
        guest_write: ResourceWriteStamp,
        host_epoch: u64,
    ) -> bool {
        if key.generation == 0 || pages.is_empty() {
            return false;
        }
        self.entries.insert(
            key,
            GvaStoreEntry {
                pages: pages.to_vec(),
                guest_write,
                host_epoch,
            },
        );
        true
    }

    pub fn reach(
        &self,
        key: GvaTargetKey,
        current_guest_write: ResourceWriteStamp,
        host_writes: &HostWrites,
    ) -> GvaWriteReach {
        let Some(entry) = self.entries.get(&key) else {
            return GvaWriteReach::NoEntry;
        };
        if !current_guest_write.quiet_since(entry.guest_write) {
            return GvaWriteReach::GuestWrote;
        }
        match host_writes.wrote_any_since(entry.host_epoch, &entry.pages) {
            HostWriteVerdict::Quiet => GvaWriteReach::Quiet,
            other => GvaWriteReach::Host(other),
        }
    }

    pub fn host_epoch_distance(&self, key: GvaTargetKey, current_epoch: u64) -> Option<u64> {
        Some(current_epoch.saturating_sub(self.entries.get(&key)?.host_epoch))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl PendingWritebacks {
    pub fn len(&self) -> usize {
        self.debts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.debts.is_empty()
    }

    #[must_use = "a replaced debt may own an older resident identity"]
    pub fn arm_gva(
        &mut self,
        key: GvaResourceKey,
        mut debt: GvaWritebackDebt,
    ) -> Option<GvaWritebackDebt> {
        debt.seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);
        self.debts.insert(key.plane(debt.linear.target_gva()), debt)
    }

    pub fn ensure_gva_resource(
        &mut self,
        key: GvaResourceKey,
        gva: u64,
        span: u64,
        pages: Option<Vec<u64>>,
    ) -> u64 {
        let plane = key.plane(gva);
        if let Some(resource) = self.resources.get_mut(&plane) {
            if resource.span == span {
                if resource.pages.is_none() {
                    resource.pages = pages.map(Arc::from);
                }
                return resource.generation;
            }
        }
        self.next_generation = self.next_generation.wrapping_add(1);
        if self.next_generation == 0 {
            self.next_generation = 1;
        }
        let generation = self.next_generation;
        self.resources.insert(
            plane,
            GvaResourceState {
                generation,
                span,
                pages: pages.map(Arc::from),
            },
        );
        generation
    }

    pub fn reback_gva_resource(&mut self, plane: GvaPlaneKey, pages: Option<Vec<u64>>) -> bool {
        let Some(resource) = self.resources.get_mut(&plane) else {
            return false;
        };
        if resource.pages.is_none() {
            resource.pages = pages.map(Arc::from);
        }
        true
    }

    pub fn gva_resource_backing(&self, plane: GvaPlaneKey) -> Option<(u64, u64, Arc<[u64]>)> {
        let resource = self.resources.get(&plane)?;
        Some((
            resource.generation,
            resource.span,
            Arc::clone(resource.pages.as_ref()?),
        ))
    }

    pub fn gva_resource_status(&self, plane: GvaPlaneKey) -> Option<(u64, u64, bool)> {
        self.resources
            .get(&plane)
            .map(|resource| (resource.generation, resource.span, resource.pages.is_some()))
    }

    pub fn discard_gva_resources(
        &mut self,
        resources: impl IntoIterator<Item = GvaResourceKey>,
    ) -> usize {
        let mut discarded = 0;
        for key in resources {
            for (_, resource) in self.resources.range_mut(key.planes()) {
                discarded += usize::from(resource.pages.take().is_some());
            }
        }
        discarded
    }

    pub fn retire_gva_resource(&mut self, key: GvaResourceKey) -> (bool, Vec<GvaWritebackDebt>) {
        let planes: Vec<_> = self
            .resources
            .range(key.planes())
            .map(|(plane, _)| *plane)
            .chain(self.debts.range(key.planes()).map(|(plane, _)| *plane))
            .collect();
        let mut existed = false;
        let mut debts = Vec::new();
        for plane in planes {
            existed |= self.resources.remove(&plane).is_some();
            debts.extend(self.debts.remove(&plane));
        }
        (existed, debts)
    }

    pub fn get_gva(&self, key: GvaResourceKey) -> Option<GvaWritebackDebt> {
        let mut owed = self.debts.range(key.planes());
        let (_, only) = owed.next()?;
        (owed.next().is_none()).then_some(*only)
    }

    pub fn has_gva(&self, key: GvaResourceKey) -> bool {
        self.debts.range(key.planes()).next().is_some()
    }

    pub fn take_gva(&mut self, key: GvaResourceKey) -> Vec<(GvaPlaneKey, GvaWritebackDebt)> {
        let planes: Vec<_> = self
            .debts
            .range(key.planes())
            .map(|(plane, _)| *plane)
            .collect();
        planes
            .into_iter()
            .filter_map(|plane| self.debts.remove(&plane).map(|debt| (plane, debt)))
            .collect()
    }

    pub fn take_gva_plane(&mut self, plane: GvaPlaneKey) -> Option<GvaWritebackDebt> {
        self.debts.remove(&plane)
    }

    pub fn restore_gva(&mut self, plane: GvaPlaneKey, debt: GvaWritebackDebt) {
        debug_assert!(self.debts.insert(plane, debt).is_none());
    }

    pub fn gvas_by_age(&self) -> Vec<GvaPlaneKey> {
        let mut all: Vec<_> = self
            .debts
            .iter()
            .map(|(key, debt)| (debt.seq, *key))
            .collect();
        all.sort_unstable();
        all.into_iter().map(|(_, key)| key).collect()
    }

    pub fn gvas_for_task(&self, task_id: u32) -> Vec<GvaResourceKey> {
        let mut all: Vec<_> = self
            .resources
            .keys()
            .map(|plane| plane.resource)
            .filter(|key| key.task_id == task_id)
            .collect();
        all.dedup();
        all
    }

    pub fn gva_for_identity(
        &self,
        identity: &crate::TargetIdentity,
    ) -> Option<(GvaPlaneKey, GvaWritebackDebt)> {
        let crate::TargetIdentity::Gva {
            gva,
            width,
            height,
            generation,
            ..
        } = *identity
        else {
            return None;
        };
        self.debts
            .iter()
            .find(|(_, debt)| {
                debt.linear.target_gva() == gva
                    && debt.width == width
                    && debt.height == height
                    && debt.generation == generation
            })
            .map(|(key, debt)| (*key, *debt))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_stamps_do_not_cross_task_retirement() {
        let mut writes = BufferWriteGens::default();
        writes.note_write(5, 7);
        let before = writes.stamp(5, 7);
        writes.retire_task(5);
        writes.note_write(5, 7);
        assert!(!writes.stamp(5, 7).quiet_since(before));
    }

    #[test]
    fn preconstruction_writes_retire_with_the_object_name() {
        let mut writes = BufferWriteGens::default();
        writes.note_write(5, 7);
        assert!(writes.has_write(5, 7));

        writes.retire_object(5, 7);

        assert!(!writes.has_write(5, 7));
        assert!(!writes.has_write(5, 8));
    }

    #[test]
    fn host_writes_are_page_exact_and_unbounded_by_age() {
        let mut writes = HostWrites::default();
        let before = writes.epoch();
        for page in 100..4200_u64 {
            writes.note_pages(vec![page << 12]);
        }
        assert_eq!(
            writes.wrote_any_since(before, &[7 << 12]),
            HostWriteVerdict::Quiet
        );
        assert_eq!(
            writes.wrote_any_since(before, &[101 << 12]),
            HostWriteVerdict::Overlap
        );
    }

    #[test]
    fn unnamed_host_write_fails_closed() {
        let mut writes = HostWrites::default();
        let before = writes.epoch();
        writes.note_unknown();
        assert_eq!(
            writes.wrote_any_since(before, &[7 << 12]),
            HostWriteVerdict::Unnamed
        );
    }

    #[test]
    fn gva_store_witness_checks_both_writer_namespaces() {
        let key = GvaTargetKey {
            task_id: 1,
            resource: ResourceId::new(7, 1),
            gva: 0x1000,
            generation: 2,
            width: 16,
            height: 16,
            bgra: true,
        };
        let mut guest = BufferWriteGens::default();
        let mut host = HostWrites::default();
        let mut witness = GvaStoreWitness::default();
        assert!(witness.note_store(
            key,
            &[0x1000],
            ResourceWriteStamp::Unresolved(guest.stamp(1, 7)),
            host.epoch()
        ));
        assert_eq!(
            witness.reach(
                key,
                ResourceWriteStamp::Unresolved(guest.stamp(1, 7)),
                &host
            ),
            GvaWriteReach::Quiet
        );
        guest.note_write(1, 7);
        assert_eq!(
            witness.reach(
                key,
                ResourceWriteStamp::Unresolved(guest.stamp(1, 7)),
                &host
            ),
            GvaWriteReach::GuestWrote
        );
        let current = ResourceWriteStamp::Unresolved(guest.stamp(1, 7));
        assert!(witness.note_store(key, &[0x1000], current, host.epoch()));
        host.note_pages(vec![0x1000]);
        assert_eq!(
            witness.reach(key, current, &host),
            GvaWriteReach::Host(HostWriteVerdict::Overlap)
        );
    }

    #[test]
    fn a_live_plane_retains_backing_until_discard() {
        let mut pending = PendingWritebacks::default();
        let key = GvaResourceKey {
            task_id: 3,
            resource: ResourceId::new(19, 1),
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
        assert_eq!(pending.discard_gva_resources([key]), 1);
        assert!(pending.gva_resource_backing(key.plane(0x4000)).is_none());
    }

    #[test]
    fn changed_span_mints_a_new_resource_generation() {
        let mut pending = PendingWritebacks::default();
        let key = GvaResourceKey {
            task_id: 3,
            resource: ResourceId::new(19, 1),
        };
        let first = pending.ensure_gva_resource(key, 0x4000, 4096, None);
        let second = pending.ensure_gva_resource(key, 0x4000, 8192, None);
        assert_ne!(first, second);
    }

    #[test]
    fn a_reused_object_slot_cannot_inherit_the_retired_resources_debt() {
        let mut pending = PendingWritebacks::default();
        let retired = GvaResourceKey {
            task_id: 3,
            resource: ResourceId::new(19, 1),
        };
        let replacement = GvaResourceKey {
            task_id: 3,
            resource: ResourceId::new(19, 2),
        };
        let mut debt = GvaWritebackDebt {
            linear: LinearColorTarget::whole(0x4000, 256, 64),
            width: 64,
            height: 64,
            format: 0,
            resident_layout: crate::pixel_format::TexelLayout::Rgba8,
            generation: 7,
            content: None,
            guest_write: ResourceWriteStamp::default(),
            seq: 0,
        };
        let _ = pending.arm_gva(retired, debt);
        debt.generation = 8;
        let _ = pending.arm_gva(replacement, debt);

        assert_eq!(pending.get_gva(retired).unwrap().generation, 7);
        assert_eq!(pending.get_gva(replacement).unwrap().generation, 8);
        pending.retire_gva_resource(retired);
        assert!(pending.get_gva(retired).is_none());
        assert_eq!(pending.get_gva(replacement).unwrap().generation, 8);
    }
}
