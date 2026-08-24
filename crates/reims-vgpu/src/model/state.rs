//! Device-owned state: registers, rings, tasks, mapper, present, fail log.

use crate::model::LruBytesMemo;
#[cfg(test)]
use reims_vgpu_core::ResourceNode;
use reims_vgpu_core::{
    CursorState, DeviceRegisters, DisplayHandshake, MapperCapture, MapperService, ResourceGraph,
    TaskEntry, TaskTable, WorkSchedulingState,
};
#[cfg(test)]
use reims_vgpu_protocol::FenceObject;
use reims_vgpu_protocol::SurfaceId;
use reims_vgpu_protocol::{
    ByteLength, ByteOffset, ComputePipelineObject, ComputeStageInputDescriptor, ContentVersion,
    FunctionObject, GuestVirtualAddress, HeapObject, MapperResolvedSurfaceId, MapperSurfaceRef,
    ObjectListEntry as ListObjectEntry, ObjectTableRef, PlaneIndex,
    ResourceDecodeError as ResourceDecodeStatus, ResourceDescriptor as Descriptor, ResourceId,
    ResourceObject, SamplerDescriptor, SamplerObject, SubmissionId, SurfaceBackingId, TaskId,
};
use reims_vgpu_protocol::{DepthStencilObject, RenderPipelineObject};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Opaque device instance id (QEMU handle).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DeviceId(pub u64);

/// Which check found a FIFO packet malformed.
///
/// One variant per distinct check, because the whole point of the vocabulary is
/// that `malformed packet` is not a diagnosis. These were thirteen hyphenated
/// `&'static str` literals passed by hand — informative to read, but not
/// greppable as slugs, not enumerable, and not countable, so nothing could tell
/// you whether the guest's ring had desynced or whether a header read had simply
/// failed.
///
/// Root-only and child-only checks are separate variants rather than one shared
/// slug plus a `channel=` field: they are genuinely different reads against
/// different registers, and collapsing them would put us back where we started.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PacketFault {
    /// Producer/consumer counters cannot describe a published byte range.
    DesyncedHeadTail,
    /// `total_size` outside `[header, ring]`, or short of its stamp list.
    BadSize,
    /// Guest read failed: root packet header.
    RootHeaderRead,
    /// Guest read failed: root packet snapshot.
    RootSnapRead,
    /// Guest write failed: root completion-stamp writeback.
    RootStampWriteback,
    /// Guest read failed: child packet header.
    ChildHeaderRead,
    /// Guest read failed: child ring register base.
    ChildRegsBaseRead,
    /// Guest read failed: child ring head register.
    ChildRegsHeadRead,
    /// Guest read failed: child ring stamp register.
    ChildRegsStampRead,
    /// Guest read failed: child packet snapshot.
    ChildSnapRead,
    /// Guest read failed: child ring tail.
    ChildTailRead,
    /// Guest write failed: child ring head writeback.
    ChildHeadWriteback,
    /// This device snapshotted less of the ring than the packet's own published
    /// `total_size`. Unlike every other variant here this accuses the host, not
    /// the guest, and it is a healthy zero — `packet_snapshot_len` cannot
    /// produce a snapshot that reaches it.
    ShortSnapshot,
}

impl PacketFault {
    pub fn slug(self) -> &'static str {
        match self {
            Self::DesyncedHeadTail => "packet_desynced_head_tail",
            Self::BadSize => "packet_bad_size",
            Self::RootHeaderRead => "packet_root_header_read",
            Self::RootSnapRead => "packet_root_snap_read",
            Self::RootStampWriteback => "packet_root_stamp_writeback",
            Self::ChildHeaderRead => "packet_child_header_read",
            Self::ChildRegsBaseRead => "packet_child_regs_base_read",
            Self::ChildRegsHeadRead => "packet_child_regs_head_read",
            Self::ChildRegsStampRead => "packet_child_regs_stamp_read",
            Self::ChildSnapRead => "packet_child_snap_read",
            Self::ChildTailRead => "packet_child_tail_read",
            Self::ChildHeadWriteback => "packet_child_head_writeback",
            Self::ShortSnapshot => "packet_short_snapshot",
        }
    }
}

/// Which check refused to execute a decoded child-channel command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecFault {
    /// A type-2 indirect exec packet shorter than its declared descriptor.
    Indirect2Short,
}

impl ExecFault {
    pub fn slug(self) -> &'static str {
        match self {
            Self::Indirect2Short => "exec_indirect2_short",
        }
    }
}

/// A command the reference host dispatches to a handler, which this device
/// decodes far enough to name but does not execute.
///
/// Kept apart from [`FailEvent::UnknownChildOpcode`] because the two say
/// different things to whoever reads the log. An unknown opcode is a hole in
/// this device's decode — nobody knows what the guest asked for. One of these is
/// a command whose contract is known and whose effect this device has chosen not
/// to implement, so the record names the command and the gap can be closed by
/// writing the handler rather than by more reverse engineering.
///
/// The variants that carry no risk of losing guest work say so in their own
/// docs. A reader ranking the fail log needs that distinction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnimplementedCommand {
    /// `CmdDebug` (`0x00`). A host-side trace marker; nothing is owed.
    Debug,
    /// `CmdDeleteObject` (`0x28`). The guest is retiring one object named by a
    /// serializer destroy record, and this device holds nothing that record can
    /// name.
    ///
    /// The record's ref lives in the **serializer's per-kind ref space**: the
    /// kind comes from the record's own opcode and each kind numbers its refs
    /// independently. This device tracks no object in that space. Its object
    /// table is keyed by the *kernel object-list* ref, a different namespace
    /// reached through a different command (`0x33 CmdSetObjectList`), and the
    /// caches that do hold the kinds this command names — samplers and pipeline
    /// states — are keyed by the object's own *state*, not by any ref, so they
    /// cannot be retired by one either.
    ///
    /// So nothing is owed and nothing leaks: acting on the ref would key the
    /// object-list namespace with a number from the serializer's, and the two
    /// overlap, so the only reachable effect is destroying an unrelated object
    /// that happens to share the integer. Declining is the correct behaviour
    /// until this device tracks serializer refs, not a gap to be closed by
    /// wiring the existing teardown call to it.
    DeleteObject,
    /// `CmdDisplaySleepState` (`0x09`). The guest's panel is entering or leaving
    /// sleep and this device's display model does not move with it.
    DisplaySleepState,
    /// `CmdDisplaySetProperties` (`0x0a`). A display property the guest set and
    /// this device does not apply.
    DisplaySetProperties,
    /// `CmdDelay` (`0x3d`). The guest asked the channel to be held; this device
    /// continues immediately, which reorders nothing but can race a guest that
    /// used the delay for settling.
    Delay,
    /// One of the reference host's retired opcodes. Its handler accepts the
    /// packet and does nothing with the payload, so matching it is fidelity
    /// rather than a gap — the record exists to say an old guest is still
    /// emitting one.
    Deprecated,
}

impl UnimplementedCommand {
    pub fn slug(self) -> &'static str {
        match self {
            Self::Debug => "cmd_debug_unimplemented",
            Self::DeleteObject => "cmd_delete_object_unimplemented",
            Self::DisplaySleepState => "cmd_display_sleep_state_unimplemented",
            Self::DisplaySetProperties => "cmd_display_set_properties_unimplemented",
            Self::Delay => "cmd_delay_unimplemented",
            Self::Deprecated => "cmd_deprecated",
        }
    }

    /// Apple's own name for the command, so a reader can find it in the
    /// dispatch table without going through this enum's spelling.
    pub fn command(self) -> &'static str {
        match self {
            Self::Debug => "CmdDebug",
            Self::DeleteObject => "CmdDeleteObject",
            Self::DisplaySleepState => "CmdDisplaySleepState",
            Self::DisplaySetProperties => "CmdDisplaySetProperties",
            Self::Delay => "CmdDelay",
            Self::Deprecated => "CmdDeprecated",
        }
    }
}

/// Fail-visible protocol event (unknown/malformed). Never invents semantics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FailEvent {
    UnknownRootOpcode {
        opcode: u16,
        total_size: u32,
    },
    /// A child opcode this device does not decode. The guest's work is dropped
    /// and its stamps are still retired, so the guest is told this succeeded —
    /// which makes the record the only trace the command ever existed.
    ///
    /// `total_size` alone cannot identify the command: it counts the header and
    /// the stamps as well as the payload, so a 24-byte packet is one stamp plus
    /// one payload word or no stamps and three, and those are different
    /// commands. `stamp_count` and `payload` separate them and carry the wire
    /// bytes needed to name the opcode, matching what the `map_family` echo
    /// beside this arm already reports for the opcodes it does decode.
    UnknownChildOpcode {
        channel: u32,
        opcode: u16,
        total_size: u32,
        stamp_count: u16,
        payload: Vec<u8>,
    },
    /// A command this device names but does not execute. See
    /// [`UnimplementedCommand`] for why this is not the unknown-opcode arm.
    ///
    /// Carries the same wire fields as its neighbour above, because the two get
    /// read side by side and a reader comparing them should not have to hold two
    /// field lists in their head.
    UnimplementedChildCommand {
        channel: u32,
        command: UnimplementedCommand,
        opcode: u16,
        total_size: u32,
        stamp_count: u16,
        payload: Vec<u8>,
    },
    MalformedRootPacket {
        fault: PacketFault,
        head: u32,
    },
    MalformedChildPacket {
        channel: u32,
        fault: PacketFault,
        head: u32,
    },
    UnsupportedExec {
        channel: u32,
        fault: ExecFault,
    },
    /// A gfx-window access whose width is neither 32 nor 64 bits.
    ///
    /// Only the gfx rail can raise this. The iosfc window's handlers mask the
    /// read to the requested width and ignore the width on write, so there is
    /// no size they refuse — which is why this carries no window discriminator:
    /// a field with one reachable value tells the log's reader nothing.
    BadMmioAccess {
        offset: u64,
        size: u32,
    },
}

/// A resource constructed from one task/object-list reference.
///
/// The object-list entry and its descriptor are construction input. Once the
/// resource exists, binds retrieve this retained object rather than consulting
/// guest memory again. The guest ends that lifetime explicitly by deleting the
/// resource or the task that owns it.
#[derive(Debug)]
struct TaskResourceConstruction {
    entry: ListObjectEntry,
    descriptor: Arc<[u8]>,
    /// Typed form of the construction descriptor, decoded exactly once for
    /// this resource lifetime.
    decoded: OnceLock<Result<Descriptor, ResourceDecodeStatus>>,
}

#[derive(Debug)]
struct TaskResourceRelations {
    /// Publication state for descriptor-declared graph relations.
    relation_publication: AtomicU8,
    /// Mapping association established by registered-surface construction.
    iosurface_mapping: OnceLock<SurfaceId>,
}

/// One retained task resource, split into immutable construction, canonical
/// identity, graph relations, and operational use state.
#[derive(Debug)]
pub struct TaskResource {
    construction: TaskResourceConstruction,
    /// Canonical generational identity assigned when the task namespace
    /// publishes this object.
    semantic_id: OnceLock<ResourceId<ResourceObject>>,
    relations: TaskResourceRelations,
    /// Identity whose strong lifetime is exactly this serialized resource.
    /// Direct backend objects keep only a weak reference, so deletion—not an
    /// arbitrary idle timeout—makes them reclaimable.
    lifetime: reims_vgpu_core::ResourceLifetime,
    /// Set after a successful draw used this texture as an attachment. A
    /// sampled-only texture cannot have an engine render-target resident, so
    /// its bind need not probe the mutable Store/witness registries. This is
    /// resource state carried by the decoded attachment use, not an inference
    /// from its address, shape, or contents.
    was_render_target: AtomicBool,
}

impl TaskResource {
    pub fn new(entry: ListObjectEntry, descriptor: Arc<[u8]>) -> Self {
        Self {
            construction: TaskResourceConstruction {
                entry,
                descriptor,
                decoded: OnceLock::new(),
            },
            semantic_id: OnceLock::new(),
            relations: TaskResourceRelations {
                relation_publication: AtomicU8::new(RELATIONS_UNPUBLISHED),
                iosurface_mapping: OnceLock::new(),
            },
            lifetime: reims_vgpu_core::ResourceLifetime::new(),
            was_render_target: AtomicBool::new(false),
        }
    }

    pub fn entry(&self) -> ListObjectEntry {
        self.construction.entry
    }

    pub fn descriptor(&self) -> &Arc<[u8]> {
        &self.construction.descriptor
    }

    pub fn semantic_id(&self) -> Option<ResourceId<ResourceObject>> {
        self.semantic_id.get().copied()
    }

    pub(crate) fn lifetime(&self) -> reims_vgpu_core::ResourceLifetime {
        self.lifetime.clone()
    }

    pub(crate) fn begin_relation_publication(&self) -> bool {
        self.relations
            .relation_publication
            .compare_exchange(
                RELATIONS_UNPUBLISHED,
                RELATIONS_PUBLISHING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    pub(crate) fn finish_relation_publication(&self, published: bool) {
        self.relations.relation_publication.store(
            if published {
                RELATIONS_PUBLISHED
            } else {
                RELATIONS_UNPUBLISHED
            },
            Ordering::Release,
        );
    }

    /// Cache one boundary-produced semantic construction descriptor.
    ///
    /// The model owns the immutable result and its lifetime; the protocol
    /// adapter supplied by the caller owns parsing the bytes.
    pub(crate) fn decoded_with(
        &self,
        decode: impl FnOnce() -> Result<Descriptor, ResourceDecodeStatus>,
    ) -> &Result<Descriptor, ResourceDecodeStatus> {
        self.construction.decoded.get_or_init(decode)
    }

    pub fn lifetime_ref(&self) -> TaskResourceLifetimeRef {
        self.lifetime.reference()
    }

    pub(crate) fn note_render_target_use(&self) {
        self.was_render_target.store(true, Ordering::Release);
    }

    pub(crate) fn was_render_target(&self) -> bool {
        self.was_render_target.load(Ordering::Acquire)
    }

    pub(crate) fn registered_iosurface_mapping(&self) -> Option<SurfaceId> {
        self.relations.iosurface_mapping.get().copied()
    }

    pub(crate) fn register_iosurface_mapping(&self, surface: SurfaceId) -> SurfaceId {
        *self.relations.iosurface_mapping.get_or_init(|| surface)
    }
}

const RELATIONS_UNPUBLISHED: u8 = 0;
const RELATIONS_PUBLISHING: u8 = 1;
const RELATIONS_PUBLISHED: u8 = 2;

/// Compatibility name while runtime requests migrate to core vocabulary.
pub type TaskResourceLifetimeRef = reims_vgpu_core::ResourceLifetimeRef;

/// Per-task resource objects, keyed by the guest's `(task, reference)` pair.
///
/// Interior synchronization keeps resource lookup available to encode helpers
/// that only borrow [`DeviceState`] immutably. Those helpers run while the
/// device already owns its state, but making the registry itself synchronized
/// also makes the lifetime rule explicit instead of relying on that outer
/// serialization.
#[derive(Debug, Default)]
struct TaskResourceRegistry {
    objects: BTreeMap<(u32, u32), Arc<TaskResource>>,
    graph: ResourceGraph,
}

pub type SubmissionResourceSnapshot = (
    ObjectTableRef<ResourceObject>,
    Option<ResourceId<ResourceObject>>,
    Option<ContentVersion>,
);

#[derive(Debug, Default)]
pub struct TaskResources(Mutex<TaskResourceRegistry>);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HeapResourceRetirement {
    pub resource: ResourceId<ResourceObject>,
    /// Present only when this resource is the final owner of the heap range.
    pub storage_origin: Option<ComputeStorageOrigin>,
}

impl TaskResources {
    pub fn get(&self, task_id: u32, ref_: u32) -> Option<Arc<TaskResource>> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .objects
            .get(&(task_id, ref_))
            .cloned()
    }

    pub fn identity(&self, task_id: u32, ref_: u32) -> Option<ResourceId<ResourceObject>> {
        self.get(task_id, ref_)?.semantic_id()
    }

    /// Recover composition routing metadata from a canonical resource lifetime.
    ///
    /// Resolved command envelopes carry only `ResourceId`; task-local object
    /// names remain inside the graph and are projected here only for legacy
    /// host materializations which have not yet adopted generational keys.
    pub(crate) fn owner(
        &self,
        id: ResourceId<ResourceObject>,
    ) -> Option<(TaskId, ObjectTableRef<ResourceObject>)> {
        let registry = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let node = registry.graph.resource(id)?;
        Some((node.task, node.object))
    }

    /// Publish a newly constructed object unless another lookup won the race.
    pub fn register(
        &self,
        task_id: u32,
        ref_: u32,
        resource: Arc<TaskResource>,
    ) -> Arc<TaskResource> {
        let mut registry = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(existing) = registry.objects.get(&(task_id, ref_)) {
            return Arc::clone(existing);
        }
        let id = registry
            .graph
            .create_resource(
                TaskId::new(task_id),
                ObjectTableRef::new(ref_),
                resource.entry().kind,
                None,
                [],
            )
            .expect("an unpublished task reference is free in the resource graph");
        resource
            .semantic_id
            .set(id)
            .expect("a resource receives one semantic identity");
        registry
            .objects
            .insert((task_id, ref_), Arc::clone(&resource));
        resource
    }

    pub fn delete(&self, task_id: u32, ref_: u32) -> bool {
        let mut registry = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let removed = registry.objects.remove(&(task_id, ref_));
        if removed.is_some() {
            registry
                .graph
                .release_reference(TaskId::new(task_id), ObjectTableRef::new(ref_))
                .expect("published resources have a graph reference");
        }
        removed.is_some()
    }

    pub fn delete_task(&self, task_id: u32) -> usize {
        let mut registry = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let before = registry.objects.len();
        registry.objects.retain(|&(task, _), _| task != task_id);
        let removed = before - registry.objects.len();
        let graph_removed = registry.graph.release_task(TaskId::new(task_id));
        debug_assert_eq!(removed, graph_removed);
        removed
    }

    #[cfg(test)]
    pub fn resource_node(&self, id: ResourceId<ResourceObject>) -> Option<ResourceNode> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .graph
            .resource(id)
            .cloned()
    }

    #[cfg(test)]
    pub fn storage_node(
        &self,
        id: reims_vgpu_protocol::StorageId,
    ) -> Option<reims_vgpu_core::StorageNode> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .graph
            .storage(id)
            .cloned()
    }

    /// Record a CPU write against the canonical resource identity, when that
    /// object has already been constructed.
    #[cfg(test)]
    pub fn note_guest_write(&self, task_id: u32, ref_: u32) -> Option<ContentVersion> {
        let registry = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let id = registry
            .graph
            .resolve(TaskId::new(task_id), ObjectTableRef::new(ref_))?;
        registry.graph.resource(id)?.content.guest_wrote().ok()
    }

    /// Record a CPU write after resolution has replaced the serializer ref.
    pub fn note_guest_write_by_id(&self, id: ResourceId<ResourceObject>) -> Option<ContentVersion> {
        let registry = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        registry.graph.guest_wrote_aliases(id)
    }

    /// Snapshot the canonical identity and content version of a constructed
    /// task resource.
    pub fn content_stamp(
        &self,
        task_id: u32,
        ref_: u32,
    ) -> Option<(ResourceId<ResourceObject>, ContentVersion)> {
        let registry = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let id = registry
            .graph
            .resolve(TaskId::new(task_id), ObjectTableRef::new(ref_))?;
        Some((id, registry.graph.resource(id)?.content.current()))
    }

    pub fn content_version_for(&self, id: ResourceId<ResourceObject>) -> Option<ContentVersion> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .graph
            .resource(id)
            .map(|node| node.content.current())
    }

    /// Snapshot content through an already resolved task resource.
    pub fn content_stamp_for(
        &self,
        resource: &TaskResource,
    ) -> Option<reims_vgpu_core::ContentStamp> {
        let id = resource.semantic_id()?;
        let registry = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Some(reims_vgpu_core::ContentStamp {
            resource: id,
            version: registry.graph.resource(id)?.content.current(),
        })
    }

    /// Apply persistent GPU materializations returned by a successful executor
    /// completion. Stale stamps are ignored: a newer guest write remains the
    /// sole current version and the old GPU copy cannot regain authority.
    pub fn record_gpu_materializations(
        &self,
        stamps: impl IntoIterator<Item = reims_vgpu_core::ContentStamp>,
    ) -> usize {
        let registry = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        stamps
            .into_iter()
            .filter(|stamp| {
                registry
                    .graph
                    .resource(stamp.resource)
                    .is_some_and(|node| node.content.gpu_materialized(stamp.version).is_ok())
            })
            .count()
    }

    /// Apply a completed GPU Store to the resource version state.
    ///
    /// The current executor reports render operations synchronously. Reserving
    /// and completing here therefore occurs only after its successful
    /// completion fact; no speculative version becomes authoritative.
    pub fn record_completed_gpu_store(
        &self,
        task_id: u32,
        ref_: u32,
        submission: SubmissionId,
    ) -> Option<(ResourceId<ResourceObject>, ContentVersion)> {
        let registry = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let id = registry
            .graph
            .resolve(TaskId::new(task_id), ObjectTableRef::new(ref_))?;
        let content = &registry.graph.resource(id)?.content;
        content.gpu_store_planned(submission).ok()?;
        let version = content.gpu_store_completed(submission).ok()?;
        Some((id, version))
    }

    /// Record successful materialization of one exact GPU content version.
    pub fn record_gpu_to_guest_copy(
        &self,
        id: ResourceId<ResourceObject>,
        version: ContentVersion,
    ) -> bool {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .graph
            .resource_mut(id)
            .is_some_and(|node| node.content.copy_gpu_to_guest_completed(version).is_ok())
    }

    /// Apply an ordered Store whose guest-memory destination is protected by
    /// the executor's write ledger until its submission fence retires.
    ///
    /// Resource currency advances when the command is accepted in submission
    /// order. Physical access to the guest replica is separately gated by that
    /// ledger, so no observer can consume the new version before it lands.
    pub fn record_ordered_materialized_store(
        &self,
        task_id: u32,
        ref_: u32,
        submission: SubmissionId,
    ) -> Option<(ResourceId<ResourceObject>, ContentVersion)> {
        let (id, version) = self.record_completed_gpu_store(task_id, ref_, submission)?;
        self.record_gpu_to_guest_copy(id, version)
            .then_some((id, version))
    }

    /// Resolve and enter every constructed resource declared by a submission.
    ///
    /// Residency tables legitimately contain objects which no command has
    /// constructed in this process yet. Those remain unresolved in the
    /// immutable envelope instead of being assigned a guessed identity.
    pub fn begin_submission(
        &self,
        task_id: u32,
        submission: SubmissionId,
        objects: impl IntoIterator<Item = ObjectTableRef<ResourceObject>>,
    ) -> Vec<SubmissionResourceSnapshot> {
        let mut registry = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let task = TaskId::new(task_id);
        objects
            .into_iter()
            .map(
                |object| match registry.graph.enter_submission(task, object, submission) {
                    Some((id, expected)) => (object, Some(id), Some(expected)),
                    None => (object, None, None),
                },
            )
            .collect()
    }

    /// Pair one submission's successful prepare/submit transitions.
    pub fn complete_submission(
        &self,
        submission: SubmissionId,
        resources: impl IntoIterator<Item = ResourceId<ResourceObject>>,
    ) {
        let mut registry = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let resources: BTreeSet<_> = resources.into_iter().collect();
        for id in resources {
            registry
                .graph
                .complete(id, submission)
                .expect("submission resources complete exactly once");
        }
    }

    pub fn attach_mapper_storage(
        &self,
        task_id: u32,
        ref_: u32,
        mapper_ref: MapperSurfaceRef,
        plane: PlaneIndex,
    ) -> bool {
        let mut registry = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(id) = registry
            .objects
            .get(&(task_id, ref_))
            .and_then(|resource| resource.semantic_id())
        else {
            return false;
        };
        let storage = registry
            .graph
            .mapper_storage(mapper_ref, plane)
            .expect("storage identity space remains available");
        registry.graph.attach_initial_storage(id, storage).is_ok()
    }

    pub fn attach_registered_surface(&self, task_id: u32, ref_: u32, surface_id: u32) -> bool {
        let mut registry = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(id) = registry
            .objects
            .get(&(task_id, ref_))
            .and_then(|resource| resource.semantic_id())
        else {
            return false;
        };
        let storage = registry
            .graph
            .registered_surface_storage(SurfaceBackingId::new(u64::from(surface_id)))
            .expect("storage identity space remains available");
        registry.graph.attach_initial_storage(id, storage).is_ok()
    }

    pub fn attach_task_address(&self, task_id: u32, ref_: u32, address: u64, length: u64) -> bool {
        let mut registry = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(id) = registry
            .objects
            .get(&(task_id, ref_))
            .and_then(|resource| resource.semantic_id())
        else {
            return false;
        };
        let storage = registry
            .graph
            .task_address_storage(
                TaskId::new(task_id),
                GuestVirtualAddress::new(address),
                ByteLength::new(length),
            )
            .expect("storage identity space remains available");
        registry.graph.attach_initial_storage(id, storage).is_ok()
    }

    pub fn link_view(
        &self,
        task_id: u32,
        view_ref: u32,
        parent_task: u32,
        parent_ref: u32,
    ) -> bool {
        let mut registry = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let child = registry
            .objects
            .get(&(task_id, view_ref))
            .and_then(|resource| resource.semantic_id());
        let parent = registry
            .objects
            .get(&(parent_task, parent_ref))
            .and_then(|resource| resource.semantic_id());
        match (child, parent) {
            (Some(child), Some(parent)) => registry.graph.link_parent(child, parent).is_ok(),
            _ => false,
        }
    }

    pub fn link_buffer_texture(
        &self,
        task_id: u32,
        texture_ref: u32,
        buffer_ref: u32,
        offset: u64,
        bytes_per_row: u64,
    ) -> bool {
        let mut registry = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let child = registry
            .objects
            .get(&(task_id, texture_ref))
            .and_then(|resource| resource.semantic_id());
        let parent = registry
            .objects
            .get(&(task_id, buffer_ref))
            .and_then(|resource| resource.semantic_id());
        match (child, parent) {
            (Some(child), Some(parent)) => registry
                .graph
                .link_buffer_range(
                    child,
                    parent,
                    ByteOffset::new(offset),
                    ByteLength::new(bytes_per_row),
                )
                .is_ok(),
            _ => false,
        }
    }

    pub fn link_heap_texture(
        &self,
        task_id: u32,
        texture_ref: u32,
        heap: ResourceId<HeapObject>,
        explicit: Option<(u64, u64)>,
    ) -> Result<(), reims_vgpu_core::GraphError> {
        let mut registry = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let texture = registry
            .objects
            .get(&(task_id, texture_ref))
            .and_then(|resource| resource.semantic_id())
            .ok_or(reims_vgpu_core::GraphError::ResourceAbsent)?;
        registry.graph.link_heap_texture(
            texture,
            heap,
            explicit.map(|(offset, length)| (ByteOffset::new(offset), ByteLength::new(length))),
        )
    }

    pub fn heap_storage_origin(
        &self,
        task_id: u32,
        texture_ref: u32,
    ) -> Option<ComputeStorageOrigin> {
        let registry = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let texture = registry
            .objects
            .get(&(task_id, texture_ref))?
            .semantic_id()?;
        let node = registry.graph.resource(texture)?;
        let storage = registry.graph.storage(node.storage?)?;
        match storage.backing {
            reims_vgpu_core::StorageBacking::HeapPlacement {
                heap,
                offset,
                length,
            } => Some(ComputeStorageOrigin::HeapPlacement {
                heap,
                offset: offset.get(),
                span_end: offset.get().checked_add(length.get())?,
            }),
            reims_vgpu_core::StorageBacking::HeapAllocation { heap, allocation } => {
                Some(ComputeStorageOrigin::HeapAllocation { heap, allocation })
            }
            _ => None,
        }
    }

    /// The residency identities a heap-texture deletion is entitled to end.
    ///
    /// Allocator-owned residents carry the resource generation directly. An
    /// explicit placement carries only its heap range, so it may be withdrawn
    /// only when the deleted resource is that storage node's final owner.
    pub(crate) fn heap_resource_retirement(
        &self,
        task_id: u32,
        texture_ref: u32,
    ) -> Option<HeapResourceRetirement> {
        let registry = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let resource = registry
            .objects
            .get(&(task_id, texture_ref))?
            .semantic_id()?;
        let node = registry.graph.resource(resource)?;
        let storage = registry.graph.storage(node.storage?)?;
        let origin = match storage.backing {
            reims_vgpu_core::StorageBacking::HeapPlacement {
                heap,
                offset,
                length,
            } => ComputeStorageOrigin::HeapPlacement {
                heap,
                offset: offset.get(),
                span_end: offset.get().checked_add(length.get())?,
            },
            reims_vgpu_core::StorageBacking::HeapAllocation { heap, allocation } => {
                ComputeStorageOrigin::HeapAllocation { heap, allocation }
            }
            _ => return None,
        };
        Some(HeapResourceRetirement {
            resource,
            storage_origin: (storage.owners.len() == 1 && storage.owners.contains(&resource))
                .then_some(origin),
        })
    }

    /// Resolve the retained buffer allocation behind one buffer-backed texture.
    ///
    /// The child-to-parent graph edge is generational. A deleted and reused raw
    /// buffer reference therefore cannot retarget a texture that still owns the
    /// original parent resource.
    pub(crate) fn buffer_texture_backing(
        &self,
        texture: &TaskResource,
    ) -> Option<(u64, u64, u64, u64)> {
        let texture_id = texture.semantic_id()?;
        let registry = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let texture_node = registry.graph.resource(texture_id)?;
        let texture_storage = registry.graph.storage(texture_node.storage?)?;
        let reims_vgpu_core::StorageBacking::BufferRange {
            buffer,
            offset,
            bytes_per_row,
        } = texture_storage.backing
        else {
            return None;
        };
        if !texture_node.parents.contains(&buffer) {
            return None;
        }
        let buffer_node = registry.graph.resource(buffer)?;
        let buffer_storage = registry.graph.storage(buffer_node.storage?)?;
        let reims_vgpu_core::StorageBacking::TaskAddress {
            address, length, ..
        } = buffer_storage.backing
        else {
            return None;
        };
        Some((
            address.get(),
            length.get(),
            offset.get(),
            bytes_per_row.get(),
        ))
    }
}

#[cfg(test)]
mod task_resource_graph_tests {
    use super::*;
    use reims_vgpu_protocol::ObjectKind;

    fn resource(kind: ObjectKind) -> Arc<TaskResource> {
        Arc::new(TaskResource::new(
            ListObjectEntry::new(kind, 0, 0),
            Arc::from([]),
        ))
    }

    #[test]
    fn task_resource_publication_assigns_one_canonical_identity() {
        let resources = TaskResources::default();
        let first = resources.register(4, 9, resource(ObjectKind::Buffer));
        let raced = resources.register(4, 9, resource(ObjectKind::Buffer));
        let id = first.semantic_id().expect("published identity");

        assert!(Arc::ptr_eq(&first, &raced));
        assert_eq!(raced.semantic_id(), Some(id));
        let node = resources.resource_node(id).expect("canonical node");
        assert_eq!(node.task, TaskId::new(4));
        assert_eq!(node.object, ObjectTableRef::new(9));
        assert_eq!(node.kind, ObjectKind::Buffer);
    }

    #[test]
    fn explicit_delete_and_reference_reuse_advance_the_canonical_generation() {
        let resources = TaskResources::default();
        let first = resources.register(4, 9, resource(ObjectKind::Texture));
        let first_id = first.semantic_id().unwrap();
        assert!(resources.delete(4, 9));
        let second = resources.register(4, 9, resource(ObjectKind::Texture));
        let second_id = second.semantic_id().unwrap();

        assert_eq!(first_id.index(), second_id.index());
        assert_ne!(first_id.generation(), second_id.generation());
        assert!(resources.resource_node(first_id).is_none());
        assert!(resources.resource_node(second_id).is_some());
    }

    #[test]
    fn only_the_last_heap_alias_may_retire_the_shared_storage_origin() {
        let resources = TaskResources::default();
        let first = resources.register(4, 9, resource(ObjectKind::TextureView));
        let alias = resources.register(4, 10, resource(ObjectKind::TextureView));
        let heap = ResourceId::<HeapObject>::new(7, 3);
        resources
            .link_heap_texture(4, 9, heap, Some((0x200, 0x800)))
            .unwrap();
        resources
            .link_heap_texture(4, 10, heap, Some((0x200, 0x800)))
            .unwrap();

        assert_eq!(
            resources.heap_resource_retirement(4, 9),
            Some(HeapResourceRetirement {
                resource: first.semantic_id().unwrap(),
                storage_origin: None,
            })
        );
        assert!(resources.delete(4, 9));
        assert_eq!(
            resources.heap_resource_retirement(4, 10),
            Some(HeapResourceRetirement {
                resource: alias.semantic_id().unwrap(),
                storage_origin: Some(ComputeStorageOrigin::HeapPlacement {
                    heap,
                    offset: 0x200,
                    span_end: 0xa00,
                }),
            })
        );
    }

    #[test]
    fn mapper_backed_texture_gets_storage_distinct_from_its_resource_identity() {
        let resources = TaskResources::default();
        let texture = resources.register(4, 9, resource(ObjectKind::IOSurfaceTexture));
        let id = texture.semantic_id().unwrap();

        assert!(resources.attach_mapper_storage(
            4,
            9,
            reims_vgpu_protocol::MapperSurfaceRef::new(12),
            PlaneIndex::new(2),
        ));
        let node = resources.resource_node(id).unwrap();
        assert!(node.storage.is_some());
        assert_eq!(node.backing_generation.get(), 1);
    }

    #[test]
    fn registered_surface_view_retains_and_shares_its_parents_storage() {
        let resources = TaskResources::default();
        let surface = resources.register(0, 12, resource(ObjectKind::SurfaceBacking));
        let view = resources.register(4, 9, resource(ObjectKind::IOSurfacePlaneView));
        let surface_id = surface.semantic_id().unwrap();
        let view_id = view.semantic_id().unwrap();

        assert!(resources.attach_registered_surface(0, 12, 12));
        assert!(resources.link_view(4, 9, 0, 12));

        let surface_node = resources.resource_node(surface_id).unwrap();
        let view_node = resources.resource_node(view_id).unwrap();
        assert_eq!(surface_node.storage, view_node.storage);
        assert!(view_node.parents.contains(&surface_id));
        assert!(resources.delete(0, 12));
        assert!(resources.resource_node(surface_id).is_some());
        assert!(resources.delete(4, 9));
        assert!(resources.resource_node(surface_id).is_none());
    }

    #[test]
    fn task_address_aliases_share_storage_but_not_resource_identity() {
        let resources = TaskResources::default();
        let buffer = resources.register(4, 9, resource(ObjectKind::Buffer));
        let texture = resources.register(4, 10, resource(ObjectKind::Texture));

        assert!(resources.attach_task_address(4, 9, 0x4000, 0x2000));
        assert!(resources.attach_task_address(4, 10, 0x4000, 0x2000));
        let buffer_node = resources
            .resource_node(buffer.semantic_id().unwrap())
            .unwrap();
        let texture_node = resources
            .resource_node(texture.semantic_id().unwrap())
            .unwrap();

        assert_ne!(buffer_node.id, texture_node.id);
        assert_eq!(buffer_node.storage, texture_node.storage);
    }

    #[test]
    fn buffer_texture_relation_retains_the_source_buffer() {
        let resources = TaskResources::default();
        let buffer = resources.register(4, 9, resource(ObjectKind::Buffer));
        let texture = resources.register(4, 10, resource(ObjectKind::TextureView));
        let buffer_id = buffer.semantic_id().unwrap();

        assert!(resources.attach_task_address(4, 9, 0x4000, 0x2000));
        assert!(resources.link_buffer_texture(4, 10, 9, 96, 512));
        assert!(resources.delete(4, 9));
        assert!(resources.resource_node(buffer_id).is_some());
        let replacement = resources.register(4, 9, resource(ObjectKind::Buffer));
        assert!(resources.attach_task_address(4, 9, 0x9000, 0x1000));
        assert_ne!(replacement.semantic_id(), Some(buffer_id));
        assert_eq!(
            resources.buffer_texture_backing(&texture),
            Some((0x4000, 0x2000, 96, 512)),
            "a live texture must keep naming its retained generational buffer"
        );
        assert!(resources.delete(4, 10));
        assert!(resources.resource_node(buffer_id).is_none());
    }

    #[test]
    fn submission_snapshot_retains_a_deleted_resource_until_completion() {
        let resources = TaskResources::default();
        let resource = resources.register(4, 9, resource(ObjectKind::Texture));
        let id = resource.semantic_id().unwrap();
        let submission = SubmissionId::new(12);

        let snapshot = resources.begin_submission(4, submission, [ObjectTableRef::new(9)]);
        assert_eq!(
            snapshot,
            vec![(
                ObjectTableRef::new(9),
                Some(id),
                Some(ContentVersion::new(1))
            )]
        );
        assert_eq!(
            resources.resource_node(id).unwrap().lifecycle,
            reims_vgpu_core::LifecycleState::InFlight
        );

        assert!(resources.delete(4, 9));
        assert!(resources.resource_node(id).is_some());
        resources.complete_submission(submission, [id]);
        assert!(resources.resource_node(id).is_none());
    }

    #[test]
    fn pre_submission_guest_write_is_the_expected_content_version() {
        let resources = TaskResources::default();
        let resource = resources.register(4, 9, resource(ObjectKind::Buffer));
        let id = resource.semantic_id().unwrap();

        let version = resources.note_guest_write(4, 9).unwrap();
        let snapshot =
            resources.begin_submission(4, SubmissionId::new(1), [ObjectTableRef::new(9)]);

        assert_eq!(
            snapshot[0],
            (ObjectTableRef::new(9), Some(id), Some(version))
        );
        resources.complete_submission(SubmissionId::new(1), [id]);
    }

    #[test]
    fn executor_materialization_applies_only_to_the_exact_stamped_version() {
        let resources = TaskResources::default();
        let resource = resources.register(4, 9, resource(ObjectKind::Texture));
        let stale = resources.content_stamp_for(resource.as_ref()).unwrap();
        let current = resources.note_guest_write(4, 9).unwrap();

        assert_eq!(resources.record_gpu_materializations([stale]), 0);
        assert!(!resources
            .resource_node(stale.resource)
            .unwrap()
            .content
            .snapshot()
            .current_in_gpu());

        let current = reims_vgpu_core::ContentStamp {
            resource: stale.resource,
            version: current,
        };
        assert_eq!(resources.record_gpu_materializations([current]), 1);
        assert!(resources
            .resource_node(current.resource)
            .unwrap()
            .content
            .snapshot()
            .current_in_gpu());
    }

    #[test]
    fn constructed_resource_content_supersedes_the_unresolved_write_fallback() {
        let mut state = DeviceState::new(DeviceId(1), crate::model::PAGE_SHIFT_X86);
        let unresolved = state.resource_write_stamp(4, 9);
        state.content.preconstruction_writes.note_write(4, 9);
        assert!(
            !state.resource_write_stamp(4, 9).quiet_since(unresolved),
            "the fallback must preserve writes which precede construction"
        );

        let resource = state
            .task_objects
            .resources
            .register(4, 9, resource(ObjectKind::Buffer));
        let initial = state.resource_write_stamp(4, 9);
        assert!(
            !initial.quiet_since(unresolved),
            "a generational resource identity cannot equal an unresolved slot"
        );

        state.content.preconstruction_writes.note_write(4, 9);
        assert!(
            state.resource_write_stamp(4, 9).quiet_since(initial),
            "the fallback counter cannot invalidate a constructed resource"
        );

        state.task_objects.resources.note_guest_write(4, 9).unwrap();
        assert!(
            !state.resource_write_stamp(4, 9).quiet_since(initial),
            "the canonical content version owns constructed-resource currency"
        );
        assert!(resource.semantic_id().is_some());
    }

    #[test]
    fn repeated_residency_records_complete_one_resource_participation() {
        let resources = TaskResources::default();
        let resource = resources.register(4, 9, resource(ObjectKind::Buffer));
        let id = resource.semantic_id().unwrap();
        let submission = SubmissionId::new(5);

        let snapshot = resources.begin_submission(
            4,
            submission,
            [ObjectTableRef::new(9), ObjectTableRef::new(9)],
        );
        assert_eq!(snapshot.len(), 2);
        resources.complete_submission(submission, snapshot.iter().filter_map(|item| item.1));
        assert_eq!(
            resources.resource_node(id).unwrap().lifecycle,
            reims_vgpu_core::LifecycleState::Created
        );
    }

    #[test]
    fn completed_gpu_store_and_copy_update_one_resource_version() {
        let resources = TaskResources::default();
        let resource = resources.register(4, 9, resource(ObjectKind::Texture));
        let id = resource.semantic_id().unwrap();

        let (stored_id, version) = resources
            .record_completed_gpu_store(4, 9, SubmissionId::new(3))
            .unwrap();
        assert_eq!(stored_id, id);
        let node = resources.resource_node(id).unwrap();
        let content = node.content.snapshot();
        assert_eq!(content.current, version);
        assert!(content.current_in_gpu());
        assert!(!content.current_in_guest());

        assert!(resources.record_gpu_to_guest_copy(id, version));
        assert!(resources
            .resource_node(id)
            .unwrap()
            .content
            .snapshot()
            .current_in_guest());
    }
}

pub type TaskReferenceStates<T, M> = reims_vgpu_core::TaskReferenceStates<T, M>;

/// Per-task sampler objects, keyed by the sampler API's reference space.
pub type TaskSamplerStates = TaskReferenceStates<SamplerDescriptor, SamplerObject>;

/// Immutable construction state retained by a compute-pipeline object.
#[derive(Clone, Debug)]
pub(crate) struct LoadedComputePipeline {
    pub kernel_func_ref: u32,
    /// Function payload retained by the pipeline lifetime. Releasing the
    /// function reference cannot invalidate a pipeline already constructed
    /// from it.
    pub kernel_mtlb: Arc<[u8]>,
    /// Product-ready stage-input. `None` means the descriptor declared none —
    /// and only that. A descriptor whose entries exceeded the decoder's caps
    /// refuses the pipeline rather than landing here as `None`, because the two
    /// are different guest programs.
    pub stage_input: Option<ComputeStageInputDescriptor>,
}

/// Per-task compute pipeline states, keyed by that API's reference space.
pub type TaskComputePipelineStates =
    TaskReferenceStates<LoadedComputePipeline, ComputePipelineObject>;

/// Immutable shader-function payload retained for the guest object lifetime.
#[derive(Debug)]
pub(crate) struct LoadedFunction {
    pub mtlb: Arc<[u8]>,
}

/// Per-task shader functions, keyed by the function API's reference space.
pub type TaskFunctionStates = TaskReferenceStates<LoadedFunction, FunctionObject>;

pub type TaskFenceStates = reims_vgpu_core::TaskFenceStates;
pub type TaskEventStates = reims_vgpu_core::TaskEventStates;
pub type TaskHeapStates = TaskReferenceStates<(), HeapObject>;

/// Per-task render pipeline states, keyed by the pipeline API's reference
/// space. A state owns its decoded descriptor, translated functions and derived
/// bind plan exactly as one native pipeline state owns its construction.
pub type TaskRenderPipelineStates =
    TaskReferenceStates<reims_vgpu_core::ResolvedRenderPipeline, RenderPipelineObject>;

/// Per-task depth-stencil states, keyed by that API's reference space.
///
/// A depth-stencil state is an immutable object with its own explicit delete
/// command (`OPCODE_DELETE_DEPTH_STENCIL_STATE`), exactly like a sampler state
/// and a render pipeline state, so it belongs in this namespace and not in
/// [`TaskResources`] — whose type mask deliberately excludes serializer resources,
/// because that tag is also worn by mutable serializer descriptors and two
/// reference spaces sharing one map would destroy each other's entries when
/// their integers collide.
///
/// It used to be resolved out of guest memory on **every draw that bound any
/// depth state**: an object-list lookup, a descriptor read, an `Arc<[u8]>`
/// allocation and a decode, measured at 0.43-0.47 µs of a 9.8 µs Maps chain.
/// The census that licensed retaining it counted the bytes rather than assuming
/// them — 1 878 843 reads of 32 distinct references over a driven boot, **every
/// one of them byte-identical to the previous read of the same reference and not
/// one changed**. The guest publishes the state once and binds it; the delete
/// command is the invalidation, which is why this needs no capacity. The
/// internal generation distinguishes a later reuse of the same guest ref; it
/// is not another guest-visible lifetime event.
pub type TaskDepthStencilStates =
    TaskReferenceStates<reims_vgpu_protocol::DepthStencilDescriptor, DepthStencilObject>;

#[cfg(test)]
mod task_reference_state_tests {
    use super::TaskReferenceStates;
    use reims_vgpu_protocol::{SamplerObject, SerializerRef};
    use std::sync::Arc;

    #[test]
    fn explicit_reference_and_task_deletion_are_the_only_retirement_events() {
        let states = TaskReferenceStates::<_, SamplerObject>::default();
        let seven = SerializerRef::new(7);
        let eight = SerializerRef::new(8);
        let first = states.register(1, seven, Arc::new(10u32));
        let raced = states.register(1, seven, Arc::new(11u32));
        states.register(1, eight, Arc::new(12u32));
        states.register(2, seven, Arc::new(13u32));

        assert!(Arc::ptr_eq(&first, &raced), "the first construction wins");
        let first_id = states.identity(1, seven).unwrap();
        assert_eq!(*states.get(1, seven).unwrap(), 10);
        assert!(states.delete(1, seven));
        assert!(!states.contains(1, seven));
        states.register(1, seven, Arc::new(14u32));
        let replacement_id = states.identity(1, seven).unwrap();
        assert_eq!(first_id.index(), replacement_id.index());
        assert_ne!(first_id.generation(), replacement_id.generation());
        assert!(states.contains(1, eight));
        assert!(states.contains(2, seven));

        assert_eq!(states.delete_task(1), 2);
        assert!(!states.contains(1, seven));
        assert!(!states.contains(1, eight));
        assert!(states.contains(2, seven));
        assert_eq!(
            *first, 10,
            "an encoder owner remains valid after registry deletion"
        );
    }

    #[test]
    fn a_live_reference_population_has_no_capacity_eviction() {
        let states = TaskReferenceStates::<_, SamplerObject>::default();
        for ref_ in 0..2048 {
            states.register(3, SerializerRef::new(ref_), Arc::new(ref_));
        }
        for ref_ in 0..2048 {
            assert_eq!(*states.get(3, SerializerRef::new(ref_)).unwrap(), ref_);
        }
    }
}

/// Why a `DeviceState` mutator refused a decoded guest record.
///
/// # The `*IdSentinel` five were `*IdRange`
///
/// Five of these named a *range* check, because `is_surface_mapping_id` used to be
/// `id >= 1 && id < MAX_MAPPINGS` and one variant covered both halves. The
/// ceiling is gone — `surface_mappings` is an unbounded registry over the full
/// wire `u32`, so it refused ids its own storage would have held — and the only value these can
/// now refuse is 0, the device-wide "no mapping" sentinel that `runtime::draw`
/// reads as "this attachment is addressed by GVA".
///
/// So the slugs say `_id_sentinel`. A name that still said `_id_range` would
/// tell a reader ranking the fail log that the guest overran a table, and send
/// them looking for a bound that does not exist. Four sibling `*TaskIdRange`
/// variants were deleted outright in the same move, for the same reason: the
/// task table is a map too, and there is no id it refuses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StateMutationDecline {
    SetObjectListTaskInactive {
        task_id: u32,
    },
    #[cfg(test)]
    InsertObjectTaskInactive {
        task_id: u32,
        object_ref: u32,
    },
    MapSurfaceIdSentinel {
        mapping_id: u32,
    },
    UnmapSurfaceIdSentinel {
        mapping_id: u32,
    },
    AttachMappingIdSentinel {
        mapping_id: u32,
    },
    AttachMappingInternalZero {
        mapping_id: u32,
    },
    MappingDeviceDescIdSentinel {
        mapping_id: u32,
    },
    MappingDeviceDescEmpty {
        mapping_id: u32,
    },
    MappingGeomIdSentinel {
        mapping_id: u32,
    },
    MappingGeomWidthZero {
        mapping_id: u32,
    },
    MappingGeomHeightZero {
        mapping_id: u32,
    },
    MappingGeomWidthRange {
        mapping_id: u32,
        width: u32,
    },
    MappingGeomHeightRange {
        mapping_id: u32,
        height: u32,
    },
}

impl StateMutationDecline {
    pub(crate) fn discriminant(self) -> u64 {
        match self {
            Self::SetObjectListTaskInactive { task_id }
            | Self::MapSurfaceIdSentinel {
                mapping_id: task_id,
            }
            | Self::UnmapSurfaceIdSentinel {
                mapping_id: task_id,
            }
            | Self::AttachMappingIdSentinel {
                mapping_id: task_id,
            }
            | Self::AttachMappingInternalZero {
                mapping_id: task_id,
            }
            | Self::MappingDeviceDescIdSentinel {
                mapping_id: task_id,
            }
            | Self::MappingDeviceDescEmpty {
                mapping_id: task_id,
            }
            | Self::MappingGeomIdSentinel {
                mapping_id: task_id,
            }
            | Self::MappingGeomWidthZero {
                mapping_id: task_id,
            }
            | Self::MappingGeomHeightZero {
                mapping_id: task_id,
            } => u64::from(task_id),
            #[cfg(test)]
            Self::InsertObjectTaskInactive {
                task_id,
                object_ref,
            } => (u64::from(task_id) << 32) | u64::from(object_ref),
            Self::MappingGeomWidthRange { mapping_id, width } => {
                (u64::from(mapping_id) << 32) | u64::from(width)
            }
            Self::MappingGeomHeightRange { mapping_id, height } => {
                (u64::from(mapping_id) << 32) | u64::from(height)
            }
        }
    }
}

/// The guest page table and GPU-VA base a mapping's [`SurfaceMappingEntry::
/// page_entries`] were walked from, when the list came from a surface backing surface
/// plan.
///
/// Latched at the one site that assigns those entries so the two cannot drift
/// apart. It exists so a later reader can *repeat* the walk without repeating
/// the search: `resolve_surface_backing_ex` finds the surface object by probing up
/// to 256 task object lists, and that cost is why the page list is cached rather
/// than re-derived. The walk itself is cheap — one page-table translation per
/// page — and it is the only thing that can say whether the cached list still
/// names the guest's memory.
/// It carries the mapping page generation it was latched at, and a
/// reader must check that before trusting it. Six sites clear or replace
/// `page_entries` and every one of them bumps the generation, so a carried-over
/// walk is unusable by construction rather than by every future writer
/// remembering to retire a second field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceBackingWalk {
    /// Task whose page table translated the backing pages.
    pub task_id: u32,
    /// `getGPUVirtualAddress() >> page_shift` of the surface backing — page `i`
    /// of the list is `(backing_pfn + i) << page_shift` in that task.
    pub backing_pfn: u32,
    /// `page_generation` of the list this walk produced.
    pub page_generation: u32,
}

use reims_vgpu_core::MappingContentState;
pub use reims_vgpu_core::ResourceValidity;

/// Ownership token for one host page view which must be returned through HostOps.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct HostPageView {
    ptr: usize,
    len: usize,
}

impl HostPageView {
    pub(crate) fn new(ptr: usize, len: usize) -> Option<Self> {
        (ptr != 0 && len != 0).then_some(Self { ptr, len })
    }

    pub(crate) fn ptr(&self) -> usize {
        self.ptr
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn release(self) -> (usize, usize) {
        (self.ptr, self.len)
    }
}

/// One packed host view over a mapping incarnation's guest pages.
///
/// Pointer, length, physical footprint, and optional GPU import are one state:
/// none may survive retirement without the others. The host/executor-specific
/// release operations are emitted when this value is removed from its mapping.
#[derive(Debug)]
pub(crate) struct SurfaceHostView {
    host: HostPageView,
    footprint: reims_vgpu_memory::GuestPageFootprint,
    import: Option<std::sync::Arc<reims_vgpu_memory::GuestRamImport>>,
}

impl SurfaceHostView {
    pub(crate) fn new(
        ptr: usize,
        len: usize,
        footprint: reims_vgpu_memory::GuestPageFootprint,
    ) -> Option<Self> {
        let footprint_len = footprint
            .pages()
            .len()
            .checked_mul(usize::try_from(footprint.page_size()).ok()?)?;
        let host = HostPageView::new(ptr, len)?;
        if footprint_len != len {
            return None;
        }
        Some(Self {
            host,
            footprint,
            import: None,
        })
    }

    pub(crate) fn ptr(&self) -> usize {
        self.host.ptr()
    }

    pub(crate) fn len(&self) -> usize {
        self.host.len()
    }

    pub(crate) fn footprint(&self) -> &reims_vgpu_memory::GuestPageFootprint {
        &self.footprint
    }

    pub(crate) fn import(&self) -> Option<&std::sync::Arc<reims_vgpu_memory::GuestRamImport>> {
        self.import.as_ref()
    }

    pub(crate) fn replace_import(
        &mut self,
        import: std::sync::Arc<reims_vgpu_memory::GuestRamImport>,
    ) -> Option<reims_vgpu_memory::ImportId> {
        self.import.replace(import).map(|old| {
            old.retire();
            old.id()
        })
    }

    pub(crate) fn into_release(mut self) -> ((usize, usize), Option<reims_vgpu_memory::ImportId>) {
        let import = self.import.take().map(|import| {
            import.retire();
            import.id()
        });
        (self.host.release(), import)
    }
}

/// IOSurface mapper registry entry keyed by mapping_id.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SurfaceMappingLifecycle {
    /// The guest currently exposes this mapping identity.
    pub(crate) active: bool,
    /// Logical incarnation; changes when the mapping identity is recycled.
    pub(crate) generation: u32,
    /// Guest address of the mapper object associated with this incarnation.
    pub(crate) internal_kva: u64,
}

/// IOSurface mapper registry entry keyed by mapping_id.
#[derive(Debug, Default)]
pub(crate) struct SurfacePageState {
    /// Version of this exact physical page plan.
    pub(crate) generation: u32,
    /// Guest page-table entries (valid bit + PFN); empty until resolved.
    pub(crate) entries: Vec<u32>,
    /// Guest address of the page-table source for the current plan.
    pub(crate) table_kva: u64,
    /// Contract derivation of [`Self::entries`] when it came from a surface
    /// backing. `None` for every other source.
    pub(crate) surface_walk: Option<SurfaceBackingWalk>,
}

/// Host representation of one exact [`SurfacePageState`] incarnation.
///
/// This state is topology policy output. Retiring it must not retire the guest
/// mapping, its semantic content, or its page plan.
#[derive(Debug, Default)]
pub(crate) struct SurfaceMaterialization {
    /// Contiguous ownership-bearing host view over the current page plan.
    contiguous: Option<SurfaceHostView>,
    /// Page generation whose plan the host refused to expose contiguously.
    refused_generation: Option<u32>,
}

impl SurfaceMaterialization {
    pub(crate) fn view(&self) -> Option<&SurfaceHostView> {
        self.contiguous.as_ref()
    }

    pub(crate) fn has_view(&self) -> bool {
        self.contiguous.is_some()
    }

    pub(crate) fn install(&mut self, view: SurfaceHostView) {
        self.contiguous = Some(view);
    }

    pub(crate) fn footprint(&self) -> Option<reims_vgpu_memory::GuestPageFootprint> {
        self.view().map(|view| view.footprint().clone())
    }

    pub(crate) fn replace_import(
        &mut self,
        import: std::sync::Arc<reims_vgpu_memory::GuestRamImport>,
    ) -> Option<reims_vgpu_memory::ImportId> {
        self.contiguous.as_mut()?.replace_import(import)
    }

    pub(crate) fn refused_for(&self, page_generation: u32) -> bool {
        self.refused_generation == Some(page_generation)
    }

    pub(crate) fn note_refused(&mut self, page_generation: u32) {
        self.refused_generation = Some(page_generation);
    }

    /// Detach backend import identity before returning the host view which may
    /// be unmapped. Neither operation changes the mapping or page plan.
    pub(crate) fn retire(
        &mut self,
    ) -> (Option<(usize, usize)>, Option<reims_vgpu_memory::ImportId>) {
        self.contiguous
            .take()
            .map(SurfaceHostView::into_release)
            .map_or((None, None), |(view, import)| (Some(view), import))
    }
}

/// One complete semantic declaration of a mapped surface.
///
/// Absence is represented by `None` in [`SurfaceDeclaration`]; there is no
/// separate validity bit which can disagree with these three fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceGeometry {
    pub width: u32,
    pub height: u32,
    pub format: u16,
}

/// Construction facts learned about one mapping independently of its logical,
/// page-table, content, and host-materialization lifetimes.
#[derive(Debug, Default)]
pub(crate) struct SurfaceDeclaration {
    geometry: Option<SurfaceGeometry>,
    /// Cached `sIOSurfaceDeviceDescriptor` from the mapping object. The exact
    /// declared record is exposed only through [`Self::device_desc_complete`].
    device_desc: Vec<u8>,
}

impl SurfaceDeclaration {
    pub(crate) fn geometry(&self) -> Option<SurfaceGeometry> {
        self.geometry
    }

    pub(crate) fn publish_geometry(&mut self, geometry: SurfaceGeometry) {
        self.geometry = Some(geometry);
    }

    pub(crate) fn clear(&mut self) {
        self.geometry = None;
        self.device_desc.clear();
    }

    pub(crate) fn publish_device_desc(&mut self, desc: &[u8]) {
        self.device_desc.clear();
        self.device_desc.extend_from_slice(desc);
    }

    pub(crate) fn device_desc_complete(&self) -> Option<&[u8]> {
        self.device_desc.get(..reims_vgpu_protocol::DEVICE_DESC_LEN)
    }
}

/// IOSurface mapper registry entry keyed by mapping_id.
#[derive(Debug, Default)]
pub struct SurfaceMappingEntry {
    /// Guest-visible mapping lifetime, separate from page and host lifetimes.
    pub(crate) lifecycle: SurfaceMappingLifecycle,
    /// Atomic declaration presence plus its optional device record.
    pub(crate) declaration: SurfaceDeclaration,
    /// Content currency is one transition object, separate from mapping,
    /// page-table, and host-materialization lifecycle.
    pub(crate) content: MappingContentState,
    /// Physical page-plan identity and its contract derivation.
    pub(crate) pages: SurfacePageState,
    /// Contiguous host-VA view over `page_entries` (`HostOps::map_pages`,
    /// mach_vm_remap of guest RAM). 0 = not built. This is the surface storage
    /// for the guest mapping. Guest CPU writes and host page reads see this
    /// allocation directly; on a capable unified-memory backend an imported
    /// render attachment retains the same view. Retired (never freed in place)
    /// whenever `page_entries` change; see [`PendingHostReleases`].
    /// Host/import materialization of the current page plan.
    pub(crate) materialization: SurfaceMaterialization,
    /// Task id that last owned this surface as a surface backing `OBJECT_TYPE_SURFACE`
    /// object (0 = no non-trivial hint; task 0 is always probed first anyway).
    /// `resolve_surface_backing_ex` probes this task right after task 0 so a
    /// per-bind present-path scan short-circuits instead of walking all 256
    /// task slots. Purely a search-order hint — a stale/wrong value only costs
    /// one extra probe before the full-table fallback re-finds the owner.
    pub owner_task_hint: u32,
}

/// Surface-mapping namespace owned by the device model.
///
/// These keys name IOSurface/registered-surface slots. They are deliberately
/// [`SurfaceId`]s rather than page-table [`reims_vgpu_protocol::MappingId`]s:
/// both arrive as `u32` values on existing runtime boundaries, but they name
/// independent lifetimes and must never share a registry merely because their
/// numeric values coincide.
#[derive(Debug, Default)]
pub(crate) struct SurfaceMappingRegistry {
    entries: BTreeMap<SurfaceId, SurfaceMappingEntry>,
    /// One ordering source for guest invalidation and device publication.
    /// It lives with every mapping content state whose stamps it issues.
    validity_sequence: u64,
}

impl SurfaceMappingRegistry {
    /// Issue a nonzero ordering stamp shared by both sides of content validity.
    fn next_validity_sequence(&mut self) -> u64 {
        self.validity_sequence = self.validity_sequence.saturating_add(1);
        self.validity_sequence
    }

    fn mark_written(&mut self, id: SurfaceId) -> u32 {
        let sequence = self.next_validity_sequence();
        let Some(mapping) = self.entries.get_mut(&id) else {
            return 0;
        };
        mapping.content.host_wrote_guest_pages(sequence)
    }

    fn apply_validity(
        &mut self,
        id: SurfaceId,
        ops: reims_vgpu_protocol::ResourceValidityOps,
    ) -> bool {
        if !self.entries.contains_key(&id) {
            return false;
        }
        let sequence = (ops.clear_host_valid != 0).then(|| self.next_validity_sequence());
        let mapping = self
            .entries
            .get_mut(&id)
            .expect("registered surface remains present for one state transition");
        if let Some(sequence) = sequence {
            mapping.content.guest_wrote(sequence);
        }
        mapping.content.apply_validity(ops);
        true
    }

    fn note_content_published(&mut self, id: SurfaceId) -> u32 {
        let sequence = self.next_validity_sequence();
        let Some(mapping) = self.entries.get_mut(&id) else {
            return 0;
        };
        mapping.content.host_published(sequence)
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub(crate) fn contains_key(&self, id: &u32) -> bool {
        self.entries.contains_key(&SurfaceId::new(*id))
    }

    pub(crate) fn get(&self, id: &u32) -> Option<&SurfaceMappingEntry> {
        self.entries.get(&SurfaceId::new(*id))
    }

    #[cfg(not(test))]
    fn get_mut(&mut self, id: &u32) -> Option<&mut SurfaceMappingEntry> {
        self.entries.get_mut(&SurfaceId::new(*id))
    }

    /// Mutable fixture access. Product code mutates surfaces only through
    /// [`DeviceState`] transitions.
    #[cfg(test)]
    pub(crate) fn get_mut(&mut self, id: &u32) -> Option<&mut SurfaceMappingEntry> {
        self.entries.get_mut(&SurfaceId::new(*id))
    }

    #[cfg(not(test))]
    fn entry(
        &mut self,
        id: u32,
    ) -> std::collections::btree_map::Entry<'_, SurfaceId, SurfaceMappingEntry> {
        self.entries.entry(SurfaceId::new(id))
    }

    #[cfg(test)]
    pub(crate) fn entry(
        &mut self,
        id: u32,
    ) -> std::collections::btree_map::Entry<'_, SurfaceId, SurfaceMappingEntry> {
        self.entries.entry(SurfaceId::new(id))
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (u32, &SurfaceMappingEntry)> {
        self.entries.iter().map(|(id, entry)| (id.get(), entry))
    }

    fn values_mut(&mut self) -> impl Iterator<Item = &mut SurfaceMappingEntry> {
        self.entries.values_mut()
    }

    #[cfg(test)]
    pub(crate) fn insert(
        &mut self,
        id: u32,
        entry: SurfaceMappingEntry,
    ) -> Option<SurfaceMappingEntry> {
        self.entries.insert(SurfaceId::new(id), entry)
    }
}

#[cfg(test)]
impl std::ops::Index<&u32> for SurfaceMappingRegistry {
    type Output = SurfaceMappingEntry;

    fn index(&self, id: &u32) -> &Self::Output {
        self.get(id)
            .unwrap_or_else(|| panic!("test indexed absent surface mapping {id}"))
    }
}

impl SurfaceMappingEntry {
    pub fn geometry(&self) -> Option<SurfaceGeometry> {
        self.declaration.geometry()
    }

    pub(crate) fn has_geometry(&self) -> bool {
        self.geometry().is_some()
    }

    pub(crate) fn width_or_zero(&self) -> u32 {
        self.geometry_or_zero().width
    }

    pub(crate) fn height_or_zero(&self) -> u32 {
        self.geometry_or_zero().height
    }

    pub(crate) fn format_or_zero(&self) -> u16 {
        self.geometry_or_zero().format
    }

    #[cfg(test)]
    pub(crate) fn publish_geometry_for_test(&mut self, width: u32, height: u32, format: u16) {
        self.declaration.publish_geometry(SurfaceGeometry {
            width,
            height,
            format,
        });
    }

    #[cfg(test)]
    pub(crate) fn with_geometry_for_test(mut self, width: u32, height: u32, format: u16) -> Self {
        self.publish_geometry_for_test(width, height, format);
        self
    }

    #[cfg(test)]
    pub(crate) fn clear_geometry_for_test(&mut self) {
        self.declaration.geometry = None;
    }

    #[cfg(test)]
    pub(crate) fn publish_device_desc_for_test(&mut self, desc: &[u8]) {
        self.declaration.publish_device_desc(desc);
    }

    /// Geometry projected for diagnostics which must print a value even before
    /// declaration. Behavioral decisions should use [`Self::geometry`] so
    /// absence cannot be mistaken for a zero-sized declaration.
    pub(crate) fn geometry_or_zero(&self) -> SurfaceGeometry {
        self.geometry().unwrap_or(SurfaceGeometry {
            width: 0,
            height: 0,
            format: 0,
        })
    }

    /// The cached `sIOSurfaceDeviceDescriptor`, but only when a whole one is
    /// there — `None` while nothing has published one, so a caller falls back
    /// on its own terms instead of reading a partial record.
    ///
    /// Three callers asked this in three spellings, two of which handed
    /// `device_desc.as_slice()` whole while the third handed
    /// `device_desc.get(..DEVICE_DESC_LEN)`. Those agree only because
    /// `mapper::resolve` reads into a `[0u8; DEVICE_DESC_LEN]` and so caches
    /// exactly that many bytes; `set_mapping_device_desc` enforces nothing but
    /// non-emptiness. `device_desc_plane` bounds every plane read against the
    /// slice it is handed and the plane table runs to `0x240`, past the record's
    /// own `0x200`, so a longer cached blob would make the whole-slice spelling
    /// decode an eighth plane the truncating one refuses. Truncation is the
    /// answer for all three: it is what the record declares.
    pub fn device_desc_complete(&self) -> Option<&[u8]> {
        self.declaration.device_desc_complete()
    }

    pub(crate) fn device_desc_bytes(&self) -> &[u8] {
        &self.declaration.device_desc
    }

    pub(crate) fn publish_device_desc(&mut self, desc: &[u8]) {
        self.declaration.publish_device_desc(desc);
    }
}

pub use reims_vgpu_core::{ComputeStorageOrigin, ComputeStorageResidencyKey};

/// Why a present is not backed by guest work, as reported by
/// [`DeviceState::note_present_backing`].
///
/// Two distinct findings, and the callee names which so the caller cannot supply
/// the word. Both are statements about **decoded Store bookkeeping only** —
/// the full-frame publication witness, advanced when a Store's pixels reached the mapping's guest
/// pages. Neither says what the viewer sees, and that limit is the point: on the
/// resident rail a Store renders into the registry without writing guest pages,
/// so a mapping can be "unbacked" here while a perfectly good resident carries
/// its present. What the viewer sees takes the carrier reading the emission site
/// pairs with this (`resident_presentable`), never this value alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentBacking {
    /// Presented again with no full-frame Store naming this mapping since its
    /// own previous present. Carries the unchanged publication sequence.
    Restaled { seq: u64 },
    /// First present since this mapping was created, and no full-frame Store has
    /// ever named it.
    NeverStored,
}

/// Decoded full-frame publication evidence for presented surface lifetimes.
#[derive(Clone, Debug, Default)]
struct PresentBackingEvidence {
    published: BTreeMap<u32, u64>,
    last_presented: BTreeMap<u32, u64>,
    sequence: u64,
}

/// Entry-side present backpressure and held-head episode ownership.
#[derive(Clone, Debug, Default)]
struct PresentBackpressureState {
    unpainted: u32,
    held_head: Option<(u32, u32)>,
    episodes: u64,
}

/// Whether the current present needs CPU pixels and how often each rail ran.
#[derive(Clone, Debug, Default)]
struct PresentCapturePolicy {
    current_present_resident_carried: bool,
    full_captures: u64,
    light_captures: u64,
}

/// Contract-owned mapping roles for presentation.
///
/// The mapping named by the current display transaction, the host action's
/// mapping, and the latest composited early front are different roles even when
/// their numeric values happen to agree. The content boundary is carried with
/// them because it changes which role may feed the console.
#[derive(Clone, Debug, Default)]
struct PresentRoutingState {
    presented: u32,
    host: u32,
    early_composite: u32,
    content_boundary: bool,
}

/// Host-console geometry, paint witness, and publication cadence.
#[derive(Clone, Debug, Default)]
struct PresentConsoleState {
    valid: bool,
    width: u32,
    height: u32,
    generation: u32,
    painted_mapping: u32,
    painted_generation: u32,
    window_active: bool,
    present_epoch: u64,
}

impl PresentConsoleState {
    fn note_present_started(&mut self) {
        self.valid = true;
    }

    fn establish(&mut self, width: u32, height: u32, generation: u32) {
        self.valid = true;
        self.width = width;
        self.height = height;
        self.generation = generation;
    }

    fn record_paint(&mut self, mapping: u32, width: u32, height: u32, generation: u32) {
        self.establish(width, height, generation);
        self.painted_mapping = mapping;
        self.painted_generation = generation;
    }

    fn record_painted_identity(&mut self, mapping: u32, generation: u32) {
        self.painted_mapping = mapping;
        self.painted_generation = generation;
    }

    fn already_painted(&self, mapping: u32, generation: u32) -> bool {
        self.painted_mapping == mapping && self.painted_generation == generation
    }

    #[cfg(test)]
    fn valid(&self) -> bool {
        self.valid
    }

    #[cfg(test)]
    fn width(&self) -> u32 {
        self.width
    }

    #[cfg(test)]
    fn height(&self) -> u32 {
        self.height
    }

    fn generation(&self) -> u32 {
        self.generation
    }

    fn geometry(&self) -> Option<(u32, u32)> {
        (self.valid && self.width > 0 && self.height > 0).then_some((self.width, self.height))
    }

    fn dimensions(&self) -> Option<(u32, u32)> {
        (self.width > 0 && self.height > 0).then_some((self.width, self.height))
    }

    fn matches_geometry(&self, width: u32, height: u32) -> bool {
        self.width == width && self.height == height
    }

    fn set_window_active(&mut self, active: bool) {
        self.window_active = active;
    }

    fn window_active(&self) -> bool {
        self.window_active
    }

    fn advance_epoch(&mut self) -> u64 {
        self.present_epoch = self.present_epoch.saturating_add(1);
        self.present_epoch
    }

    #[cfg(all(feature = "host-window", target_os = "macos"))]
    fn epoch(&self) -> u64 {
        self.present_epoch
    }

    #[cfg(test)]
    fn set_geometry_for_test(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    #[cfg(test)]
    fn set_generation_for_test(&mut self, generation: u32) {
        self.generation = generation;
    }

    #[cfg(test)]
    fn set_painted_generation_for_test(&mut self, generation: u32) {
        self.painted_generation = generation;
    }
}

impl PresentRoutingState {
    fn begin_present(&mut self, mapping: u32) {
        self.presented = mapping;
        self.host = mapping;
    }

    fn note_present_candidate(&mut self, mapping: u32) {
        self.presented = mapping;
    }

    fn note_early_composite(&mut self, mapping: u32) {
        self.early_composite = mapping;
    }

    fn cross_content_boundary(&mut self) {
        self.content_boundary = true;
    }

    fn presented(&self) -> u32 {
        self.presented
    }

    #[cfg(test)]
    fn host(&self) -> u32 {
        self.host
    }

    fn early_composite(&self) -> u32 {
        self.early_composite
    }

    fn content_boundary_crossed(&self) -> bool {
        self.content_boundary
    }

    fn is_current_present(&self, mapping: u32) -> bool {
        mapping == self.host || mapping == self.presented
    }
}

impl PresentCapturePolicy {
    fn set_current_present_resident_carried(&mut self, carried: bool) {
        self.current_present_resident_carried = carried;
    }

    fn current_present_resident_carried(&self) -> bool {
        self.current_present_resident_carried
    }

    fn note_full(&mut self) {
        self.full_captures = self.full_captures.wrapping_add(1);
    }

    fn note_light(&mut self) {
        self.light_captures = self.light_captures.wrapping_add(1);
    }

    fn counts(&self) -> (u64, u64) {
        (self.full_captures, self.light_captures)
    }
}

impl PresentBackpressureState {
    fn accepted(&mut self) {
        self.unpainted = self.unpainted.saturating_add(1);
    }

    fn consumed(&mut self) {
        self.unpainted = 0;
        self.held_head = None;
    }

    fn unpainted(&self) -> u32 {
        self.unpainted
    }

    fn at_cap(&self, cap: u32) -> bool {
        self.unpainted >= cap
    }

    fn hold(&mut self, channel: u32, head: u32) -> Option<(u32, u64)> {
        if self.held_head == Some((channel, head)) {
            return None;
        }
        self.held_head = Some((channel, head));
        self.episodes = self.episodes.saturating_add(1);
        Some((self.unpainted, self.episodes))
    }

    #[cfg(test)]
    fn set_unpainted(&mut self, count: u32) {
        self.unpainted = count;
    }

    #[cfg(test)]
    fn episodes(&self) -> u64 {
        self.episodes
    }
}

impl PresentBackingEvidence {
    fn publish(&mut self, mapping_id: u32) {
        self.sequence = self.sequence.saturating_add(1);
        self.published.insert(mapping_id, self.sequence);
    }

    fn present(&mut self, mapping_id: u32) -> Option<PresentBacking> {
        let sequence = self.published.get(&mapping_id).copied().unwrap_or(0);
        match self.last_presented.insert(mapping_id, sequence) {
            Some(previous) if previous == sequence => {
                Some(PresentBacking::Restaled { seq: sequence })
            }
            None if sequence == 0 => Some(PresentBacking::NeverStored),
            _ => None,
        }
    }

    fn retire(&mut self, mapping_id: u32) {
        self.published.remove(&mapping_id);
        self.last_presented.remove(&mapping_id);
    }

    #[cfg(test)]
    fn sequence_for(&self, mapping_id: u32) -> u64 {
        self.published.get(&mapping_id).copied().unwrap_or(0)
    }

    #[cfg(test)]
    fn copy_sequence(&mut self, source: u32, target: u32) {
        let sequence = self.sequence_for(source);
        self.published.insert(target, sequence);
    }
}

#[cfg(test)]
mod present_backing_evidence_tests {
    use super::{PresentBacking, PresentBackingEvidence};

    #[test]
    fn retirement_clears_publication_and_presented_witness_together() {
        let mut evidence = PresentBackingEvidence::default();
        evidence.publish(5);
        assert_eq!(evidence.present(5), None);
        assert!(matches!(
            evidence.present(5),
            Some(PresentBacking::Restaled { .. })
        ));

        evidence.retire(5);
        assert_eq!(evidence.present(5), Some(PresentBacking::NeverStored));
    }
}

#[cfg(test)]
mod present_backpressure_state_tests {
    use super::PresentBackpressureState;

    #[test]
    fn one_held_head_is_one_episode_until_paint_consumes_it() {
        let mut state = PresentBackpressureState::default();
        state.accepted();
        assert_eq!(state.hold(5, 464), Some((1, 1)));
        assert_eq!(state.hold(5, 464), None);

        state.consumed();
        assert_eq!(state.unpainted(), 0);
        assert_eq!(state.hold(5, 464), Some((0, 2)));
    }
}

/// HostOps view over a **task GVA range** (MapMemory2 / UnmapMemory lifecycle).
///
/// Distinct from a mapping's contiguous page-plan materialization (iosfc
/// `mapping_id` page list).
/// Published for MapMemory2, with on-demand construction retained for CPU-only
/// access; torn down on overlapping UnmapMemory / MapMemory2 / delete_task.
/// Does **not** own discrete encode content
/// (`host_gva_surfaces`) — that cache is retained across Unmap (wallpaper class).
#[derive(Debug, Default)]
pub struct GvaHostView {
    /// Task slot the walk used when the view was built (resolved active id).
    pub task_id: u32,
    /// Guest VA base of the registered span (not necessarily page-aligned).
    pub gva: u64,
    /// Byte length of the registered GVA span.
    pub length: u64,
    /// Ownership-bearing host page view. `None` exists only for an
    /// unverifiable synthetic fixture; product construction always supplies it.
    pub(crate) host_view: Option<HostPageView>,
    /// Exact page-table result the host alias was built from. An empty list is
    /// reserved for synthetic fixtures which cannot be revalidated.
    pub page_gpas: Arc<[u64]>,
    /// Backend-visible allocation over this view. A RAMBlock import is borrowed
    /// from the VM lifetime; a host-allocation import is owned by this mapping.
    pub(crate) import: Option<Arc<reims_vgpu_memory::GuestRamImport>>,
    pub(crate) import_head: u64,
}

impl GvaHostView {
    pub(crate) fn new(
        task_id: u32,
        gva: u64,
        length: u64,
        host_view: HostPageView,
        page_gpas: Arc<[u64]>,
    ) -> Self {
        Self {
            task_id,
            gva,
            length,
            host_view: Some(host_view),
            page_gpas,
            import: None,
            import_head: 0,
        }
    }

    #[cfg(test)]
    pub(crate) fn fixture(task_id: u32, gva: u64, length: u64, ptr: usize, ptr_len: usize) -> Self {
        Self::new(
            task_id,
            gva,
            length,
            HostPageView::new(ptr, ptr_len).expect("fixture host view"),
            Arc::from([]),
        )
    }

    pub(crate) fn ptr(&self) -> usize {
        self.host_view.as_ref().map_or(0, HostPageView::ptr)
    }

    pub(crate) fn ptr_len(&self) -> usize {
        self.host_view.as_ref().map_or(0, HostPageView::len)
    }

    pub(crate) fn take_host_view(&mut self) -> Option<(usize, usize)> {
        self.host_view.take().map(HostPageView::release)
    }

    pub(crate) fn install_import(
        &mut self,
        import: Arc<reims_vgpu_memory::GuestRamImport>,
        import_head: u64,
    ) {
        self.import = Some(import);
        self.import_head = import_head;
    }

    pub(crate) fn import(&self) -> Option<&Arc<reims_vgpu_memory::GuestRamImport>> {
        self.import.as_ref()
    }

    pub(crate) fn import_head(&self) -> u64 {
        self.import_head
    }

    fn take_owned_import(&mut self) -> Option<reims_vgpu_memory::ImportId> {
        self.import
            .take()
            .filter(|import| import.gpa_base().is_none())
            .map(|import| {
                import.retire();
                import.id()
            })
    }
}

/// Which guest pages a GVA-keyed encode was stored against.
///
/// [`HostReplicaState::gva_surfaces`] is keyed by guest **virtual** address, and
/// a GVA is only a name for whatever the guest's page table points it at right
/// now. The guest recycles those names hard — the deferred-window drift census
/// routinely reports every page of a GVA moving between arm and flush — so
/// "same gva, same geometry" does not mean "same allocation". This records the
/// physical backing the pixels were produced from, so a later lookup can tell a
/// mapping that churned and came back (the retained wallpaper class) from a name
/// the guest handed to a different resource.
///
/// The first page, not the whole list. This held a dense `Vec<u64>` — one slot
/// per guest page, holes included, so a permutation could not read as the same
/// mapping — and the store walked the entire span to fill it. Nothing ever read
/// past element 0. `surface_cache::gva_backing_state`, the one consumer that
/// decides anything, compares the first page and says so in its own doc; the
/// only reader of `len()` was the gauge reporting how many bytes the lists cost,
/// which is a measurement of its own overhead. `span` had no reader at all.
///
/// So the store now takes one `translate_task_gva`, exactly the call the check
/// makes, and a 4K entry costs one walk instead of ~2 025. Producer and consumer
/// ask the identical question, which is the property the dense list was reaching
/// for and did not have.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GvaBacking {
    /// Task whose page table the walk used.
    pub task_id: u32,
    /// Page-aligned leaf GPA of the span's first page when the pixels were
    /// stored.
    pub first_gpa: u64,
}

/// Host-owned BGRA8 frame for a surface_id (Linux/Vulkan render-cache, §8.5).
#[derive(Clone, Debug, Default)]
pub struct HostSurface {
    pub width: u32,
    pub height: u32,
    /// Tight BGRA8, stride = width * 4.
    ///
    /// Shared rather than owned so a holder that took the frame keeps it across
    /// a replacement of this entry: the two point at one allocation, and storing
    /// a new frame leaves the holder's pixels intact instead of orphaning them.
    pub bgra: std::sync::Arc<Vec<u8>>,
    /// Generation of the store that produced these bytes, issued by
    /// [`DeviceState::next_sampled_content_generation`] (independent of guest
    /// `content_generation`).
    ///
    /// Device-global rather than per-entry, because this value is half of the
    /// sampled-content identity the engine binds on. A per-entry counter is
    /// only unique while the entry lives, and this map's entries are removed
    /// and re-created on the routine deferred-Store arm path.
    pub host_gen: u64,
    /// Decoded object type that produced a GVA-keyed type-2/3 encode. Zero for
    /// surface/ref caches and for stores that did not record an owner.
    pub producer_object_type: u8,
    /// Recency stamp for the GVA cache's byte cap
    /// ([`GVA_ENCODE_CACHE_BYTE_CAP`]), from
    /// [`HostReplicaState`]'s recency transition. Bumped on store **and on every
    /// confirmed hit**, which is the half that matters: a wallpaper plane is
    /// stored once and sampled forever, so a stamp advanced only by stores
    /// would make the most-wanted entry in the map look like the coldest.
    /// Unused (and left at 0) by the surface_id and texture_ref caches, which
    /// have no cap.
    pub last_touch: u64,
    /// Guest pages these bytes were produced from, for GVA-keyed entries.
    /// `None` on the surface_id/texture_ref caches (their key is not a guest
    /// virtual address) and on any GVA store whose walk did not resolve.
    pub backing: Option<GvaBacking>,
    /// The target GVA the store that produced these bytes rendered into, for
    /// texture_ref-keyed entries. Zero when the producer had none, and unused by
    /// the GVA-keyed cache, whose key *is* that address.
    ///
    /// The ref cache is the fallback door of the colour LOAD seed, and a LOAD
    /// seed is the attachment's *prior content* — so serving one produced at a
    /// different address hands the pass another allocation's picture to
    /// composite onto, and the Store writes the result back. That is a fixpoint:
    /// the next frame loads what this one stored. This field is what lets the
    /// serve site say whether that happened, which is the reading the door has
    /// never had — `load_seed_ok_color` counts both doors as one.
    pub source_gva: u64,
    /// Whether the guest's own pages already hold these pixels.
    ///
    /// This is the field that decides whether the byte cap may evict the entry,
    /// and it exists because two rules in this device were relying on each other
    /// without either saying so.
    ///
    /// The render writeback stores into this cache on every outcome, because on
    /// the ones that did not reach guest RAM it is what holds the authoritative
    /// bytes. The page-ownership guard then argues that *refusing* a guest write is
    /// safe — permitting one would land pixels in whatever now owns those pages,
    /// which has been observed as guest heap corruption — and closes with "the
    /// caller keeps the content either way … so nothing renderable is lost by
    /// refusing".
    ///
    /// That closing clause is a claim about this map, and
    /// `surface_cache::enforce_gva_cache_cap` was free to falsify it: an entry
    /// that is the only copy of pixels the guest never received is an ordinary
    /// eviction candidate to a cap that only counts bytes.
    ///
    /// So `false` marks an entry the cap must not take: evicting it is the loss
    /// the refusal was allowed on the promise that it would not happen. `true`
    /// means the guest's pages have the same bytes, a later read can re-derive
    /// them from guest RAM, and eviction costs a re-read and nothing else.
    ///
    /// `true` for the surface_id and texture_ref caches, which have no cap.
    pub guest_holds_bytes: bool,
}

/// Raw type-2/3 texture content retained by the discrete backend.
///
/// Unlike [`HostSurface`], bytes stay in the guest Metal pixel format and are
/// tightly row-packed. The key is `(task_id, texture_ref)`; descriptor fields
/// below reject stale hits after a ref is rebound. UnmapMemory drops the guest
/// page-table alias, not this GPU-private texture body.
#[derive(Clone, Debug, Default)]
pub struct HostLinearTexture {
    pub gva: u64,
    pub pixel_format: u16,
    pub width: u32,
    pub height: u32,
    pub row_stride: u64,
    pub bytes: Vec<u8>,
    pub host_gen: u32,
    /// Nonzero ⇒ the engine's pinned resident storage image at this generation
    /// is the authoritative content and `bytes` is empty (deferred linear
    /// writeback). Cleared by any bytes store.
    pub resident_gen: u32,
}

/// Complete identity of one task-local native-format host replica.
///
/// The descriptor fields are meaningful only as a unit: a task/object key
/// reused with another address, format, extent, or stride names a different
/// replica window and must not inherit the prior bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearReplicaWindow {
    pub task_id: u32,
    pub texture_ref: u32,
    pub gva: u64,
    pub pixel_format: u16,
    pub width: u32,
    pub height: u32,
    pub row_stride: u64,
}

impl LinearReplicaWindow {
    fn key(self) -> (u32, u32) {
        (self.task_id, self.texture_ref)
    }

    fn storable_bpp(self) -> Option<u32> {
        let bpp = reims_vgpu_core::pixel_format::bytes_per_pixel(self.pixel_format)?;
        let valid = self.texture_ref != 0
            && self.gva != 0
            && self.width != 0
            && self.height != 0
            && self.row_stride >= (self.width as u64).saturating_mul(bpp as u64);
        valid.then_some(bpp)
    }

    fn tight_len(self, bpp: u32) -> Option<usize> {
        (self.width as usize)
            .checked_mul(self.height as usize)?
            .checked_mul(bpp as usize)
    }

    fn describes(self, entry: &HostLinearTexture) -> bool {
        entry.gva == self.gva
            && entry.pixel_format == self.pixel_format
            && entry.width == self.width
            && entry.height == self.height
            && entry.row_stride == self.row_stride
    }

    fn adopt(self, entry: &mut HostLinearTexture) {
        entry.gva = self.gva;
        entry.pixel_format = self.pixel_format;
        entry.width = self.width;
        entry.height = self.height;
        entry.row_stride = self.row_stride;
        entry.bytes.clear();
    }
}

/// Why resident native-format bytes could not be published into their replica.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearMaterializeDecline {
    /// The entry is gone or a newer resident generation replaced this one.
    Superseded { resident_gen: u32 },
    /// The retained format has no byte size.
    FormatUnsized { pixel_format: u16 },
    /// The retained tight extent cannot be represented by the host.
    TightSizeOverflow { width: u32, height: u32, bpp: u32 },
    /// The executor returned less than one complete tight image.
    ReadbackShort { got: usize, need: usize },
}

/// The latest `presentFrame` retain and the warm buffer used to replace it.
///
/// Pixels and their identity are one value: publishing a light resident-backed
/// frame deliberately clears the CPU bytes, while a failed full capture returns
/// its scratch without changing any part of the prior retain. Keeping those
/// transitions here prevents callers from publishing half of a new identity or
/// accidentally violating the keep-prior contract.
#[derive(Clone, Debug, Default)]
pub(crate) struct RetainedPresentFrame {
    bgra: Vec<u8>,
    mapping: u32,
    width: u32,
    height: u32,
    generation: u32,
    /// Semantic surface epoch: pixel identity beside the guest-page generation.
    content_epoch: u32,
    valid: bool,
    encode_pending: bool,
    /// Warm second buffer. It is storage only and is never read as frame content.
    scratch: Vec<u8>,
}

impl RetainedPresentFrame {
    pub(crate) fn pixels(&self) -> &[u8] {
        &self.bgra
    }

    pub(crate) fn mapping(&self) -> u32 {
        self.mapping
    }

    #[cfg(feature = "host-window")]
    pub(crate) fn width(&self) -> u32 {
        self.width
    }

    #[cfg(feature = "host-window")]
    pub(crate) fn height(&self) -> u32 {
        self.height
    }

    pub(crate) fn generation(&self) -> u32 {
        self.generation
    }

    #[cfg(feature = "host-window")]
    pub(crate) fn content_epoch(&self) -> u32 {
        self.content_epoch
    }

    pub(crate) fn is_valid(&self) -> bool {
        self.valid
    }

    pub(crate) fn encode_pending(&self) -> bool {
        self.encode_pending
    }

    pub(crate) fn matches_geometry(&self, width: u32, height: u32) -> bool {
        self.width == width && self.height == height
    }

    pub(crate) fn publish_light(
        &mut self,
        mapping: u32,
        width: u32,
        height: u32,
        generation: u32,
        content_epoch: u32,
    ) {
        self.bgra.clear();
        self.publish_identity(mapping, width, height, generation, content_epoch);
    }

    pub(crate) fn take_scratch(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.scratch)
    }

    pub(crate) fn return_scratch(&mut self, scratch: Vec<u8>) {
        self.scratch = scratch;
    }

    pub(crate) fn publish_full(
        &mut self,
        bgra: Vec<u8>,
        mapping: u32,
        width: u32,
        height: u32,
        generation: u32,
        content_epoch: u32,
    ) {
        self.scratch = std::mem::replace(&mut self.bgra, bgra);
        self.publish_identity(mapping, width, height, generation, content_epoch);
    }

    fn publish_identity(
        &mut self,
        mapping: u32,
        width: u32,
        height: u32,
        generation: u32,
        content_epoch: u32,
    ) {
        self.mapping = mapping;
        self.width = width;
        self.height = height;
        self.generation = generation;
        self.content_epoch = content_epoch;
        self.valid = true;
        self.encode_pending = true;
    }

    pub(crate) fn mark_encode_pending(&mut self) {
        self.encode_pending = true;
    }

    pub(crate) fn mark_encoded(&mut self) {
        self.encode_pending = false;
    }

    #[cfg(test)]
    pub(crate) fn invalidate(&mut self) {
        self.valid = false;
    }

    #[cfg(test)]
    pub(crate) fn validate(&mut self) {
        self.valid = true;
    }

    #[cfg(test)]
    pub(crate) fn scratch_len(&self) -> usize {
        self.scratch.len()
    }

    #[cfg(test)]
    pub(crate) fn clear_pixels(&mut self) {
        self.bgra.clear();
    }

    #[cfg(test)]
    fn set_identity_for_test(
        &mut self,
        mapping: u32,
        width: u32,
        height: u32,
        generation: u32,
        content_epoch: u32,
    ) {
        self.mapping = mapping;
        self.width = width;
        self.height = height;
        self.generation = generation;
        self.content_epoch = content_epoch;
    }

    #[cfg(test)]
    fn replace_pixels_for_test(&mut self, bgra: Vec<u8>) {
        self.bgra = bgra;
    }
}

/// Present / scanout model state.
#[derive(Clone, Debug, Default)]
pub struct PresentState {
    /// Last decoded write class for each live surface incarnation.
    ///
    /// This is presentation routing evidence, not mapping ownership. It is
    /// retired with the mapping incarnation so a reused numeric surface id
    /// cannot inherit a predecessor's Composite/ClearOnly classification.
    write_kind: BTreeMap<SurfaceId, SurfaceWriteKind>,
    /// Host-console geometry/current generation, successful-paint witness,
    /// window ownership, and publication epoch.
    console: PresentConsoleState,
    /// Presented, host-action, and early-composite mapping roles plus the
    /// content-boundary transition that changes which may feed the console.
    routing: PresentRoutingState,
    /// Full-frame publication, last-presented comparison, and mapping-lifetime
    /// retirement in one structural evidence ledger. This records decoded Store
    /// bookkeeping only; it does not infer resident content.
    backing_evidence: PresentBackingEvidence,
    /// Latest presentFrame retain (PGDisplay +0x188), including its semantic
    /// identity, CPU pixels when present, and capture scratch ownership.
    frame: RetainedPresentFrame,
    /// Accepted-but-unpainted count and held-head episode coalescing.
    backpressure: PresentBackpressureState,
    /// True when the current present's resident and attached engine presenter
    /// were prepared successfully before capture. When true,
    /// `capture_present_frame` skips the GPU→host readback because the window
    /// consumes that same resident directly.
    capture_policy: PresentCapturePolicy,
}

impl PresentState {
    pub(crate) fn establish_console(&mut self, width: u32, height: u32, generation: u32) {
        self.console.establish(width, height, generation);
    }

    pub(crate) fn record_console_paint(
        &mut self,
        mapping: u32,
        width: u32,
        height: u32,
        generation: u32,
    ) {
        self.console
            .record_paint(mapping, width, height, generation);
    }

    pub(crate) fn record_painted_identity(&mut self, mapping: u32, generation: u32) {
        self.console.record_painted_identity(mapping, generation);
    }

    pub(crate) fn console_already_painted(&self, mapping: u32, generation: u32) -> bool {
        self.console.already_painted(mapping, generation)
    }

    #[cfg(test)]
    pub(crate) fn console_valid(&self) -> bool {
        self.console.valid()
    }

    #[cfg(test)]
    pub(crate) fn console_width(&self) -> u32 {
        self.console.width()
    }

    #[cfg(test)]
    pub(crate) fn console_height(&self) -> u32 {
        self.console.height()
    }

    pub(crate) fn console_generation(&self) -> u32 {
        self.console.generation()
    }

    pub(crate) fn console_geometry(&self) -> Option<(u32, u32)> {
        self.console.geometry()
    }

    pub(crate) fn console_dimensions(&self) -> Option<(u32, u32)> {
        self.console.dimensions()
    }

    pub(crate) fn console_matches_geometry(&self, width: u32, height: u32) -> bool {
        self.console.matches_geometry(width, height)
    }

    pub(crate) fn set_window_active(&mut self, active: bool) {
        self.console.set_window_active(active);
    }

    pub(crate) fn window_active(&self) -> bool {
        self.console.window_active()
    }

    pub(crate) fn advance_present_epoch(&mut self) -> u64 {
        self.console.advance_epoch()
    }

    #[cfg(all(feature = "host-window", target_os = "macos"))]
    pub(crate) fn present_epoch(&self) -> u64 {
        self.console.epoch()
    }

    #[cfg(test)]
    pub(crate) fn set_console_geometry_for_test(&mut self, width: u32, height: u32) {
        self.console.set_geometry_for_test(width, height);
    }

    #[cfg(test)]
    pub(crate) fn set_console_generation_for_test(&mut self, generation: u32) {
        self.console.set_generation_for_test(generation);
    }

    #[cfg(test)]
    pub(crate) fn set_painted_generation_for_test(&mut self, generation: u32) {
        self.console.set_painted_generation_for_test(generation);
    }

    pub(crate) fn begin_present(&mut self, mapping: u32) {
        self.routing.begin_present(mapping);
        self.console.note_present_started();
    }

    pub(crate) fn note_present_candidate(&mut self, mapping: u32) {
        self.routing.note_present_candidate(mapping);
    }

    pub(crate) fn note_early_composite(&mut self, mapping: u32) {
        self.routing.note_early_composite(mapping);
    }

    pub(crate) fn cross_content_boundary(&mut self) {
        self.routing.cross_content_boundary();
    }

    pub(crate) fn presented_mapping(&self) -> u32 {
        self.routing.presented()
    }

    #[cfg(test)]
    pub(crate) fn host_mapping(&self) -> u32 {
        self.routing.host()
    }

    pub(crate) fn early_composite_mapping(&self) -> u32 {
        self.routing.early_composite()
    }

    pub(crate) fn content_boundary_crossed(&self) -> bool {
        self.routing.content_boundary_crossed()
    }

    pub(crate) fn is_current_present(&self, mapping: u32) -> bool {
        self.routing.is_current_present(mapping)
    }

    pub(crate) fn frame(&self) -> &RetainedPresentFrame {
        &self.frame
    }

    pub(crate) fn publish_light_frame(
        &mut self,
        mapping: u32,
        width: u32,
        height: u32,
        generation: u32,
        content_epoch: u32,
    ) {
        self.frame
            .publish_light(mapping, width, height, generation, content_epoch);
    }

    pub(crate) fn take_capture_scratch(&mut self) -> Vec<u8> {
        self.frame.take_scratch()
    }

    pub(crate) fn return_capture_scratch(&mut self, scratch: Vec<u8>) {
        self.frame.return_scratch(scratch);
    }

    pub(crate) fn publish_captured_frame(
        &mut self,
        bgra: Vec<u8>,
        mapping: u32,
        width: u32,
        height: u32,
        generation: u32,
        content_epoch: u32,
    ) {
        self.frame
            .publish_full(bgra, mapping, width, height, generation, content_epoch);
    }

    pub(crate) fn mark_frame_encode_pending(&mut self) {
        self.frame.mark_encode_pending();
    }

    pub(crate) fn mark_frame_encoded(&mut self) {
        self.frame.mark_encoded();
    }

    #[cfg(test)]
    pub(crate) fn invalidate_frame_for_test(&mut self) {
        self.frame.invalidate();
    }

    #[cfg(test)]
    pub(crate) fn validate_frame_for_test(&mut self) {
        self.frame.validate();
    }

    #[cfg(test)]
    pub(crate) fn clear_frame_pixels_for_test(&mut self) {
        self.frame.clear_pixels();
    }

    #[cfg(test)]
    pub(crate) fn set_frame_identity_for_test(
        &mut self,
        mapping: u32,
        width: u32,
        height: u32,
        generation: u32,
        content_epoch: u32,
    ) {
        self.frame
            .set_identity_for_test(mapping, width, height, generation, content_epoch);
    }

    #[cfg(test)]
    pub(crate) fn replace_frame_pixels_for_test(&mut self, bgra: Vec<u8>) {
        self.frame.replace_pixels_for_test(bgra);
    }

    pub(crate) fn note_present_accepted(&mut self) {
        self.backpressure.accepted();
    }

    pub(crate) fn note_paint_consumed(&mut self) {
        self.backpressure.consumed();
    }

    pub(crate) fn unpainted_presents(&self) -> u32 {
        self.backpressure.unpainted()
    }

    pub(crate) fn present_backpressure_at_cap(&self, cap: u32) -> bool {
        self.backpressure.at_cap(cap)
    }

    pub(crate) fn note_backpressure_hold(&mut self, channel: u32, head: u32) -> Option<(u32, u64)> {
        self.backpressure.hold(channel, head)
    }

    #[cfg(test)]
    pub(crate) fn set_unpainted_presents_for_test(&mut self, count: u32) {
        self.backpressure.set_unpainted(count);
    }

    #[cfg(test)]
    pub(crate) fn backpressure_hold_count_for_test(&self) -> u64 {
        self.backpressure.episodes()
    }

    pub(crate) fn set_current_present_resident_carried(&mut self, carried: bool) {
        self.capture_policy
            .set_current_present_resident_carried(carried);
    }

    pub(crate) fn current_present_resident_carried(&self) -> bool {
        self.capture_policy.current_present_resident_carried()
    }

    pub(crate) fn note_full_capture(&mut self) {
        self.capture_policy.note_full();
    }

    pub(crate) fn note_light_capture(&mut self) {
        self.capture_policy.note_light();
    }

    pub(crate) fn capture_counts(&self) -> (u64, u64) {
        self.capture_policy.counts()
    }
}

#[cfg(test)]
mod retained_present_frame_tests {
    use super::RetainedPresentFrame;

    #[test]
    fn capture_failure_keeps_the_whole_prior_retain_and_success_recycles_it() {
        let prior = vec![0x11; 16];
        let next = vec![0x22; 16];
        let mut frame = RetainedPresentFrame::default();
        frame.publish_full(prior.clone(), 3, 2, 2, 7, 9);

        let mut scratch = frame.take_scratch();
        scratch.extend_from_slice(&next);
        frame.return_scratch(scratch);
        assert_eq!(frame.pixels(), prior);
        assert_eq!((frame.mapping(), frame.generation()), (3, 7));

        frame.publish_full(next.clone(), 4, 2, 2, 8, 10);
        assert_eq!(frame.pixels(), next);
        assert_eq!((frame.mapping(), frame.generation()), (4, 8));
        assert_eq!(frame.scratch_len(), prior.len());

        frame.publish_light(5, 2, 2, 9, 11);
        assert!(frame.pixels().is_empty());
        assert_eq!((frame.mapping(), frame.generation()), (5, 9));
        assert!(frame.is_valid());
        assert!(frame.encode_pending());
    }
}

#[cfg(test)]
mod present_routing_state_tests {
    use super::{PresentConsoleState, PresentRoutingState};

    #[test]
    fn mapping_roles_change_only_through_their_own_transitions() {
        let mut routing = PresentRoutingState::default();
        routing.note_early_composite(1);
        routing.note_present_candidate(2);
        assert_eq!(routing.early_composite(), 1);
        assert_eq!(routing.presented(), 2);
        assert_eq!(routing.host(), 0);

        routing.begin_present(3);
        assert_eq!((routing.presented(), routing.host()), (3, 3));
        assert!(routing.is_current_present(3));
        assert!(!routing.is_current_present(1));

        routing.cross_content_boundary();
        routing.cross_content_boundary();
        assert!(routing.content_boundary_crossed());
        assert_eq!(routing.early_composite(), 1);
    }

    #[test]
    fn console_geometry_and_successful_paint_are_distinct_witnesses() {
        let mut console = PresentConsoleState::default();
        console.set_geometry_for_test(1440, 900);
        assert_eq!(console.dimensions(), Some((1440, 900)));
        assert_eq!(console.geometry(), None);

        console.establish(1440, 900, 7);
        assert_eq!(console.geometry(), Some((1440, 900)));
        assert!(!console.already_painted(3, 7));

        console.record_paint(3, 1440, 900, 7);
        assert!(console.already_painted(3, 7));
        assert!(!console.already_painted(4, 7));
    }
}

/// Last **command-class** write to a surface mid (not pixel occupancy).
///
/// Used so a DisplaySwap of a mid that only received Clear (no composite Store)
/// does not overwrite a finished +0x188 retain — dual-mid clear flip of empty
/// display buffers while content lives on intermediate mids. This is protocol
/// history (Clear vs Store), not an rgb_nz / content-shape gate (AGENTS).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SurfaceWriteKind {
    #[default]
    Unknown,
    /// Only clear-only streams / software CLEAR Stores since last present.
    ClearOnly,
    /// At least one draw/composite Store (m2v encode, non-clear writeback).
    Composite,
}

/// Byte cap for the guest-CPU-produced content memos (`guest_linear_memo`,
/// `iosurface_plane_view_memo`, `iosurface_texture_memo`). A cap crossing evicts the coldest entries
/// down to a low-water mark — never a bulk clear — so the hot working set (and
/// its avoided re-decode/re-convert cost) survives.
pub const GUEST_LINEAR_MEMO_BYTE_CAP: usize = 128 << 20;

/// Byte cap for the GVA-keyed type-2/3 encode cache
/// ([`HostReplicaState::gva_surfaces`]). Same basis and same value as
/// [`GUEST_LINEAR_MEMO_BYTE_CAP`], which bounds the sibling cache holding the
/// same class of content.
///
/// A byte cap rather than an entry count for the reason that constant already
/// states, measured here directly: one 60-resize boot read `gva_largest =
/// 33 423 360` — a 3840x2176x4 frame, the 4K geometry with its height padded to
/// a multiple of 64 — while the map's 305 entries totalled 291 MB. Entry count
/// cannot tell those apart; the same 305 entries would be ~10 GB if every one
/// had been 4K.
///
/// # Why this cache needs a cap at all
///
/// It is keyed by guest **virtual** address and the store does
/// `.entry(gva).or_default()`, so a new geometry at the same GVA replaces and
/// costs nothing — growth is entirely from *new* GVAs. Every resolution change
/// has the guest allocate its surfaces at fresh addresses, and until this cap
/// nothing anywhere dropped the abandoned ones. Measured over 60 guest-driven
/// resolution changes: 26 entries to 354, **strictly monotonic across all 27
/// census samples**, never once decreasing, while the set of entries a lookup
/// could still be served from stayed at ~13.
///
/// # Why LRU, and not a staleness rule
///
/// The two staleness rules this cache offers both fail, and the measurements
/// that killed them are worth keeping next to the constant:
///
/// - **Dead-task eviction** reclaims nothing. `gva_dead_task` read **0 of 331**
///   accumulated entries — the compositor survives every resize and simply
///   allocates new addresses, so every abandoned entry belongs to a task that
///   is still alive.
/// - **Evicting what no longer translates would black out the wallpaper.** This
///   cache is deliberately retained across Unmap — nothing on the Unmap path
///   touches it — so "the guest unmapped this VA" is the *normal* state of
///   exactly the content the cache exists to hold: at idle, before any resize,
///   14 of 27 entries were already unmapped, and a later driven boot read 105
///   of 138. Only [`crate::runtime::surface_cache::GvaBackingState::Moved`]
///   carries positive evidence that an address belongs to someone else.
///
/// Recency is neither. It is a resource bound, and its safety property is the
/// one those rules lack: [`crate::model::LruBytesMemo`]'s header already names
/// this exact case — an entry read every frame but never rewritten (a wallpaper
/// plane) is touched on every hit, so it is the *hottest* thing in the map and
/// can never be the victim. Eviction reaches only entries nothing has looked at.
pub const GVA_ENCODE_CACHE_BYTE_CAP: usize = 128 << 20;

/// How many evicted keys [`GvaEvictionWitness`] remembers.
///
/// A diagnostic ring, so the bound is a choice about how much history to keep,
/// not a device contract. Sized above the ~305 evictions a 4-minute 60-resize
/// drive produces so that run is covered exactly; a longer boot overflows it,
/// and the overflow is *reported* (`forgotten`) rather than silently dropping
/// the count, because an under-reported harm figure is the failure direction
/// that reads as a pass.
pub const GVA_EVICTION_WITNESS_KEYS: usize = 4096;

/// Did evicting for the byte cap cost a lookup that would otherwise have hit?
///
/// The cap is the first rule that ever removes a live task's content from
/// [`HostReplicaState::gva_surfaces`], so its cost must be countable rather
/// than argued. This remembers the exact `(gva, width, height)` of each evicted
/// entry and counts the later lookups that missed on one — a miss on a key the
/// cap dropped is precisely the harm, and nothing else is.
///
/// Read `wanted` only together with `evicted`: zero harm and zero evictions is
/// a cap that never engaged, not a cap that engaged safely, and the two must
/// not be confused.
///
/// # The reading, x86/Vulkan, 40 boots
///
/// `evicted=186  wanted=0  forgotten=0`, taken as the per-boot maxima of
/// `host_cache_levels gva_cap_*` over a 59 MB always-on log. The cap **has**
/// engaged, so this is the safe-engagement case its own rule above asks for and
/// not the never-engaged one. `forgotten=0` matters as much as `wanted=0`: the
/// ring never overflowed, so `wanted` is an exact count and not a lower bound.
///
/// That is the whole question this struct exists to answer, and it is answered.
/// Keep it anyway — it is the standing alarm on a policy `AGENTS.md` treats as a
/// smell (an eviction rule over storage that may hold the only copy of guest
/// content), it costs one `BTreeSet` insert per eviction and there have been
/// 186, and the reading is a property of this workload rather than of the code.
/// A future session that finds `wanted > 0` is looking at a real regression.
///
/// Corrects a standing claim that this cap "never evicts". It does.
#[derive(Debug, Default)]
pub struct GvaEvictionWitness {
    /// Evicted identities still remembered, for the miss test.
    keys: std::collections::BTreeSet<(u64, u32, u32)>,
    /// Same identities in eviction order, so the ring drops the oldest.
    order: std::collections::VecDeque<(u64, u32, u32)>,
    /// Entries the byte cap has evicted. The denominator.
    pub evicted: u64,
    /// Lookups that missed on an identity the cap had evicted. The harm.
    pub wanted: std::sync::atomic::AtomicU64,
    /// Identities dropped from the ring before they could be tested. Each one
    /// is a lookup `wanted` can no longer notice, so a nonzero value makes
    /// `wanted` a lower bound.
    pub forgotten: u64,
}

impl GvaEvictionWitness {
    /// Record that the cap evicted this identity.
    pub fn note_evicted(&mut self, gva: u64, width: u32, height: u32) {
        self.evicted += 1;
        let key = (gva, width, height);
        if self.keys.insert(key) {
            self.order.push_back(key);
        }
        while self.order.len() > GVA_EVICTION_WITNESS_KEYS {
            if let Some(old) = self.order.pop_front() {
                self.keys.remove(&old);
                self.forgotten += 1;
            }
        }
    }

    /// A lookup missed. Count it if the cap is why. Takes `&self` because every
    /// GVA-cache read path holds a shared borrow of the device state.
    pub fn note_miss(&self, gva: u64, width: u32, height: u32) {
        if self.keys.contains(&(gva, width, height)) {
            self.wanted
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// A store re-populated this identity, so a later miss on it is no longer
    /// attributable to the cap.
    pub fn note_restored(&mut self, gva: u64, width: u32, height: u32) {
        if self.keys.remove(&(gva, width, height)) {
            self.order.retain(|k| *k != (gva, width, height));
        }
    }

    /// `(evicted, wanted, forgotten)` for the census line.
    pub fn counts(&self) -> (u64, u64, u64) {
        (
            self.evicted,
            self.wanted.load(std::sync::atomic::Ordering::Relaxed),
            self.forgotten,
        )
    }
}

/// Host-side content replicas retained independently of guest object and
/// mapping namespaces.
///
/// These maps used to be peer fields of [`DeviceState`], which made their
/// shared lifecycle implicit: a task-local texture reference, a surface
/// mapping, and a recycled GVA could each appear to own host content directly.
/// This aggregate is the single owner of those derived representations and of
/// the bookkeeping that governs their replacement. Guest-visible lifetime
/// state may name a replica, but it does not contain one.
#[derive(Debug)]
pub struct HostReplicaState {
    /// Surface/backing namespace replicas, keyed by mapping identity.
    #[cfg(not(test))]
    surfaces: BTreeMap<u32, HostSurface>,
    #[cfg(test)]
    pub(crate) surfaces: BTreeMap<u32, HostSurface>,
    /// Task-local texture-view replicas.
    #[cfg(not(test))]
    texture_surfaces: BTreeMap<(u32, u32), HostSurface>,
    #[cfg(test)]
    pub(crate) texture_surfaces: BTreeMap<(u32, u32), HostSurface>,
    /// GVA-addressed replicas, qualified internally by their backing witness.
    #[cfg(not(test))]
    gva_surfaces: BTreeMap<u64, HostSurface>,
    #[cfg(test)]
    pub(crate) gva_surfaces: BTreeMap<u64, HostSurface>,
    /// Monotonic recency source for GVA replicas.
    gva_touch_seq: u64,
    /// Running byte total for [`Self::gva_surfaces`].
    #[cfg(not(test))]
    gva_cache_bytes: usize,
    #[cfg(test)]
    pub(crate) gva_cache_bytes: usize,
    /// Test-adjustable policy limit; production construction uses the declared
    /// derived-content memo limit.
    #[cfg(not(test))]
    gva_cache_byte_cap: usize,
    #[cfg(test)]
    pub(crate) gva_cache_byte_cap: usize,
    /// Observation of recomputable GVA replica eviction cost.
    #[cfg(not(test))]
    gva_eviction_witness: GvaEvictionWitness,
    #[cfg(test)]
    pub(crate) gva_eviction_witness: GvaEvictionWitness,
    /// Native-format replicas for task-local linear textures.
    #[cfg(not(test))]
    linear_textures: BTreeMap<(u32, u32), HostLinearTexture>,
    #[cfg(test)]
    pub(crate) linear_textures: BTreeMap<(u32, u32), HostLinearTexture>,
}

impl Default for HostReplicaState {
    fn default() -> Self {
        Self {
            surfaces: BTreeMap::new(),
            texture_surfaces: BTreeMap::new(),
            gva_surfaces: BTreeMap::new(),
            gva_touch_seq: 0,
            gva_cache_bytes: 0,
            gva_cache_byte_cap: GVA_ENCODE_CACHE_BYTE_CAP,
            gva_eviction_witness: GvaEvictionWitness::default(),
            linear_textures: BTreeMap::new(),
        }
    }
}

impl HostReplicaState {
    pub(crate) fn restore_surface(&mut self, surface_id: u32, entry: HostSurface) {
        self.surfaces.insert(surface_id, entry);
    }

    pub(crate) fn surface(&self, surface_id: u32) -> Option<&HostSurface> {
        self.surfaces.get(&surface_id)
    }

    pub(crate) fn forget_surface(&mut self, surface_id: u32) -> bool {
        self.surfaces.remove(&surface_id).is_some()
    }

    pub(crate) fn store_surface_rows(
        &mut self,
        surface_id: u32,
        width: u32,
        height: u32,
        source: &[u8],
        source_stride: u32,
        generation: u64,
    ) {
        let row = (width as usize).saturating_mul(4);
        let need = (height as usize).saturating_mul(row);
        let entry = self.surfaces.entry(surface_id).or_default();
        match Arc::get_mut(&mut entry.bgra) {
            Some(bytes) if bytes.len() == need => {
                Self::fill_surface_rows(bytes, source, source_stride, row, height)
            }
            _ => {
                let mut bytes = vec![0; need];
                Self::fill_surface_rows(&mut bytes, source, source_stride, row, height);
                entry.bgra = Arc::new(bytes);
            }
        }
        entry.host_gen = generation;
        entry.width = width;
        entry.height = height;
        entry.guest_holds_bytes = true;
    }

    fn fill_surface_rows(
        destination: &mut [u8],
        source: &[u8],
        source_stride: u32,
        row: usize,
        height: u32,
    ) {
        if source_stride as usize == row {
            let length = destination.len().min(source.len());
            destination[..length].copy_from_slice(&source[..length]);
            return;
        }
        for y in 0..height as usize {
            let source_offset = y.saturating_mul(source_stride as usize);
            let destination_offset = y.saturating_mul(row);
            if source_offset + row <= source.len() && destination_offset + row <= destination.len()
            {
                destination[destination_offset..destination_offset + row]
                    .copy_from_slice(&source[source_offset..source_offset + row]);
            }
        }
    }

    pub(crate) fn restore_texture(&mut self, task_id: u32, texture_ref: u32, entry: HostSurface) {
        self.texture_surfaces.insert((task_id, texture_ref), entry);
    }

    pub(crate) fn texture(&self, task_id: u32, texture_ref: u32) -> Option<&HostSurface> {
        self.texture_surfaces.get(&(task_id, texture_ref))
    }

    pub(crate) fn forget_texture(&mut self, task_id: u32, texture_ref: u32) -> bool {
        self.texture_surfaces
            .remove(&(task_id, texture_ref))
            .is_some()
    }

    pub(crate) fn forget_task_textures(&mut self, task_id: u32) {
        self.texture_surfaces
            .retain(|&(owner, _), _| owner != task_id);
    }

    pub(crate) fn store_linear(&mut self, window: LinearReplicaWindow, bytes: &[u8]) -> bool {
        let Some(need) = window.storable_bpp().and_then(|bpp| window.tight_len(bpp)) else {
            return false;
        };
        if bytes.len() < need {
            return false;
        }
        let entry = self.linear_textures.entry(window.key()).or_default();
        entry.host_gen = entry.host_gen.wrapping_add(1);
        if entry.host_gen == 0 {
            entry.host_gen = 1;
        }
        window.adopt(entry);
        entry.bytes.extend_from_slice(&bytes[..need]);
        entry.resident_gen = 0;
        true
    }

    pub(crate) fn note_linear_resident(
        &mut self,
        window: LinearReplicaWindow,
        generation: u32,
    ) -> bool {
        if generation == 0 || window.storable_bpp().is_none() {
            return false;
        }
        let entry = self.linear_textures.entry(window.key()).or_default();
        entry.host_gen = generation;
        window.adopt(entry);
        entry.resident_gen = generation;
        true
    }

    pub(crate) fn linear_resident_generation(&self, window: LinearReplicaWindow) -> Option<u32> {
        let entry = self.linear_textures.get(&window.key())?;
        (entry.resident_gen != 0 && window.describes(entry)).then_some(entry.resident_gen)
    }

    pub(crate) fn linear_host_generation(&self, task_id: u32, texture_ref: u32) -> Option<u32> {
        self.linear_textures
            .get(&(task_id, texture_ref))
            .map(|entry| entry.host_gen)
    }

    pub(crate) fn materialize_linear(
        &mut self,
        task_id: u32,
        texture_ref: u32,
        generation: u32,
        bytes: &[u8],
    ) -> Result<(), LinearMaterializeDecline> {
        let Some(entry) = self.linear_textures.get_mut(&(task_id, texture_ref)) else {
            return Err(LinearMaterializeDecline::Superseded { resident_gen: 0 });
        };
        if entry.resident_gen != generation {
            return Err(LinearMaterializeDecline::Superseded {
                resident_gen: entry.resident_gen,
            });
        }
        let Some(bpp) = reims_vgpu_core::pixel_format::bytes_per_pixel(entry.pixel_format) else {
            return Err(LinearMaterializeDecline::FormatUnsized {
                pixel_format: entry.pixel_format,
            });
        };
        let Some(need) = (entry.width as usize)
            .checked_mul(entry.height as usize)
            .and_then(|length| length.checked_mul(bpp as usize))
        else {
            return Err(LinearMaterializeDecline::TightSizeOverflow {
                width: entry.width,
                height: entry.height,
                bpp,
            });
        };
        if bytes.len() < need {
            return Err(LinearMaterializeDecline::ReadbackShort {
                got: bytes.len(),
                need,
            });
        }
        entry.bytes.clear();
        entry.bytes.extend_from_slice(&bytes[..need]);
        entry.resident_gen = 0;
        Ok(())
    }

    pub(crate) fn linear(&self, window: LinearReplicaWindow) -> Option<&HostLinearTexture> {
        self.linear_textures
            .get(&window.key())
            .filter(|entry| window.describes(entry))
    }

    pub(crate) fn take_object_replicas(
        &mut self,
        task_id: u32,
        texture_ref: u32,
    ) -> (bool, Option<HostLinearTexture>) {
        let texture = self.forget_texture(task_id, texture_ref);
        let linear = self.linear_textures.remove(&(task_id, texture_ref));
        (texture, linear)
    }

    pub(crate) fn take_task_linear(&mut self, task_id: u32) -> Vec<(u32, HostLinearTexture)> {
        let keys: Vec<_> = self
            .linear_textures
            .keys()
            .filter(|(owner, _)| *owner == task_id)
            .copied()
            .collect();
        keys.into_iter()
            .filter_map(|key| {
                self.linear_textures
                    .remove(&key)
                    .map(|entry| (key.1, entry))
            })
            .collect()
    }

    pub(crate) fn counts(&self) -> (usize, usize, usize, usize) {
        (
            self.surfaces.len(),
            self.texture_surfaces.len(),
            self.gva_surfaces.len(),
            self.linear_textures.len(),
        )
    }

    pub(crate) fn gva(&self, gva: u64) -> Option<&HostSurface> {
        self.gva_surfaces.get(&gva)
    }

    pub(crate) fn gva_entries(&self) -> impl Iterator<Item = (u64, &HostSurface)> {
        self.gva_surfaces.iter().map(|(&gva, entry)| (gva, entry))
    }

    pub(crate) fn replica_levels(&self) -> [(usize, usize, usize); 3] {
        fn surface_level<K: Ord>(entries: &BTreeMap<K, HostSurface>) -> (usize, usize, usize) {
            let mut bytes = 0usize;
            let mut largest = 0usize;
            for entry in entries.values() {
                bytes = bytes.saturating_add(entry.bgra.len());
                largest = largest.max(entry.bgra.len());
            }
            (entries.len(), bytes, largest)
        }

        fn linear_level<K: Ord>(entries: &BTreeMap<K, HostLinearTexture>) -> (usize, usize, usize) {
            let mut bytes = 0usize;
            let mut largest = 0usize;
            for entry in entries.values() {
                bytes = bytes.saturating_add(entry.bytes.len());
                largest = largest.max(entry.bytes.len());
            }
            (entries.len(), bytes, largest)
        }

        [
            surface_level(&self.surfaces),
            surface_level(&self.gva_surfaces),
            linear_level(&self.linear_textures),
        ]
    }

    /// Issue a strictly increasing recency stamp for a GVA replica.
    fn next_gva_touch(&mut self) -> u64 {
        self.gva_touch_seq = self.gva_touch_seq.saturating_add(1);
        self.gva_touch_seq
    }

    pub(crate) fn restore_gva(&mut self, gva: u64, mut entry: HostSurface) {
        self.gva_eviction_witness
            .note_restored(gva, entry.width, entry.height);
        entry.last_touch = self.next_gva_touch();
        let reclaimed = self
            .gva_surfaces
            .insert(gva, entry)
            .map_or(0, |previous| previous.bgra.len());
        let charged = self
            .gva_surfaces
            .get(&gva)
            .map_or(0, |current| current.bgra.len());
        self.gva_cache_bytes = self
            .gva_cache_bytes
            .saturating_sub(reclaimed)
            .saturating_add(charged);
    }

    pub(crate) fn evict_gva(&mut self, gva: u64) -> Option<HostSurface> {
        let entry = self.gva_surfaces.remove(&gva)?;
        self.gva_cache_bytes = self.gva_cache_bytes.saturating_sub(entry.bgra.len());
        Some(entry)
    }

    pub(crate) fn touch_gva(&mut self, gva: u64) {
        let stamp = self.next_gva_touch();
        if let Some(entry) = self.gva_surfaces.get_mut(&gva) {
            entry.last_touch = stamp;
        }
    }

    pub(crate) fn note_gva_landed(&mut self, gva: u64) {
        if let Some(entry) = self.gva_surfaces.get_mut(&gva) {
            entry.guest_holds_bytes = true;
        }
    }

    pub(crate) fn gva_cache_bytes(&self) -> usize {
        self.gva_cache_bytes
    }

    pub(crate) fn gva_cache_byte_cap(&self) -> usize {
        self.gva_cache_byte_cap
    }

    pub(crate) fn note_gva_evicted(&mut self, gva: u64, width: u32, height: u32) {
        self.gva_eviction_witness.note_evicted(gva, width, height);
    }

    pub(crate) fn note_gva_miss(&self, gva: u64, width: u32, height: u32) {
        self.gva_eviction_witness.note_miss(gva, width, height);
    }

    pub(crate) fn gva_eviction_counts(&self) -> (u64, u64, u64) {
        self.gva_eviction_witness.counts()
    }
}

#[cfg(test)]
mod host_replica_state_tests {
    use super::*;

    fn gva_entry(length: usize) -> HostSurface {
        HostSurface {
            width: 1,
            height: 1,
            bgra: Arc::new(vec![0; length]),
            guest_holds_bytes: true,
            ..HostSurface::default()
        }
    }

    #[test]
    fn gva_replacement_touch_and_eviction_are_one_accounted_lifecycle() {
        let mut replicas = HostReplicaState::default();
        replicas.restore_gva(0x1000, gva_entry(4));
        let first_touch = replicas.gva(0x1000).unwrap().last_touch;
        assert_eq!(replicas.gva_cache_bytes(), 4);

        replicas.touch_gva(0x1000);
        assert!(replicas.gva(0x1000).unwrap().last_touch > first_touch);

        replicas.restore_gva(0x1000, gva_entry(12));
        assert_eq!(replicas.gva_cache_bytes(), 12);
        assert_eq!(replicas.counts().2, 1);

        assert_eq!(replicas.evict_gva(0x1000).unwrap().bgra.len(), 12);
        assert_eq!(replicas.gva_cache_bytes(), 0);
        assert_eq!(replicas.counts().2, 0);
    }

    #[test]
    fn linear_window_replacement_and_materialization_are_atomic() {
        let mut replicas = HostReplicaState::default();
        let window = LinearReplicaWindow {
            task_id: 3,
            texture_ref: 7,
            gva: 0x4000,
            pixel_format: reims_vgpu_core::pixel_format::MTL_FORMAT_RGBA8_UNORM,
            width: 2,
            height: 2,
            row_stride: 8,
        };
        assert!(replicas.store_linear(window, &[1; 16]));
        assert_eq!(replicas.linear(window).unwrap().bytes, [1; 16]);

        assert!(replicas.note_linear_resident(window, 9));
        assert!(replicas.linear(window).unwrap().bytes.is_empty());
        assert_eq!(replicas.linear_resident_generation(window), Some(9));
        assert_eq!(
            replicas.materialize_linear(3, 7, 8, &[2; 16]),
            Err(LinearMaterializeDecline::Superseded { resident_gen: 9 })
        );
        replicas.materialize_linear(3, 7, 9, &[2; 16]).unwrap();
        assert_eq!(replicas.linear(window).unwrap().bytes, [2; 16]);
        assert_eq!(replicas.linear_resident_generation(window), None);
    }
}

/// One byte-exact entry in [`SampledContentState`]'s revalidation memos.
#[derive(Clone, Debug)]
pub struct GuestLinearMemo {
    /// Native guest rows (row-stride bytes as read, pre-conversion) at the last
    /// content change. Padding is included so a write anywhere in the span is
    /// observed by the byte-compare.
    pub native: Vec<u8>,
    /// Tight upload bytes of `native`, in whatever layout [`Self::layout`]
    /// names: converted RGBA8, or the guest's own texels kept exactly.
    pub rgba: std::sync::Arc<Vec<u8>>,
    /// What [`Self::rgba`] holds, so the memo hit re-states the layout the
    /// miss-fill chose.
    ///
    /// This was a `bgra8: bool`, and it could only spell two of the layouts the
    /// loader can now produce — so a half-float image stored on the miss would
    /// have come back out of a hit described as `Rgba8`: eight-byte texels bound
    /// into a four-byte image, which is a length the engine refuses and, if it
    /// had not, garbage. A `bool` standing in for an enum is the one shape
    /// `rustc` cannot tell you has gone short.
    pub layout: reims_vgpu_core::pixel_format::TexelLayout,
    /// Content generation: bumps only when the native bytes change.
    pub generation: u64,
}

/// Device-owned sampled-content identity and byte-revalidation state.
///
/// Every sampled source spends one generation namespace, independent of which
/// memo or host replica retains its bytes. The memos and gather witness live
/// here because they may reuse a generation only while they prove the same
/// source bytes remain current.
#[derive(Debug)]
pub(crate) struct SampledContentState {
    generation: u64,
    pub(crate) guest_linear_memo: LruBytesMemo<(u32, u64, u32, u32, u32, u16), GuestLinearMemo>,
    pub(crate) guest_linear_scratch: Vec<u8>,
    pub(crate) iosurface_plane_view_memo: LruBytesMemo<(u32, u32, u32, u32, u16), GuestLinearMemo>,
    pub(crate) iosurface_texture_memo: LruBytesMemo<(u32, u32, u32), GuestLinearMemo>,
    pub(crate) iosurface_texture_memo_scratch: Vec<u8>,
    pub(crate) gather_witness: reims_vgpu_core::GatherWitness,
}

impl SampledContentState {
    fn new(policies: reims_vgpu_core::GatherPolicies) -> Self {
        Self {
            generation: 0,
            guest_linear_memo: LruBytesMemo::new(GUEST_LINEAR_MEMO_BYTE_CAP),
            guest_linear_scratch: Vec::new(),
            iosurface_plane_view_memo: LruBytesMemo::new(GUEST_LINEAR_MEMO_BYTE_CAP),
            iosurface_texture_memo: LruBytesMemo::new(GUEST_LINEAR_MEMO_BYTE_CAP),
            iosurface_texture_memo_scratch: Vec::new(),
            gather_witness: reims_vgpu_core::GatherWitness::with_policies(policies),
        }
    }

    fn issue_generation(&mut self) -> u64 {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.generation = 1;
        }
        self.generation
    }
}

#[cfg(test)]
mod sampled_content_state_tests {
    use super::SampledContentState;

    #[test]
    fn sampled_identity_never_issues_the_zero_sentinel_across_wrap() {
        let mut sampled = SampledContentState::new(Default::default());
        sampled.generation = u64::MAX;
        assert_eq!(sampled.issue_generation(), 1);
        assert_eq!(sampled.issue_generation(), 2);
    }
}

/// The two distinct contract relations by which an object can name mappings.
///
/// This is a product type rather than a two-slot collection: each field names
/// the relation that can populate it, so there is no capacity, insertion order,
/// or silently ignored third write to reason about.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NamedMappings {
    reference: Option<u32>,
    registered_surface: Option<u32>,
}

impl NamedMappings {
    fn new(reference: Option<u32>, registered_surface: Option<u32>) -> Self {
        Self {
            reference,
            registered_surface: registered_surface.filter(|id| Some(*id) != reference),
        }
    }

    /// The named ids, reference first.
    pub fn iter(self) -> impl Iterator<Item = u32> {
        self.reference.into_iter().chain(self.registered_surface)
    }

    /// Whether this reference named no mapping at all.
    pub fn is_empty(self) -> bool {
        self.reference.is_none() && self.registered_surface.is_none()
    }
}

/// One release effect produced by a model mutation for a host or executor port.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostReleaseEffect {
    RetireGuestImport(reims_vgpu_memory::ImportId),
    /// Revoke the backend import, then release its host alias only after the
    /// backend reports that the final GPU access has retired.
    RetireImportedView {
        import: reims_vgpu_memory::ImportId,
        ptr: usize,
        len: usize,
    },
    ReleaseView {
        ptr: usize,
        len: usize,
    },
    RetireComputeResident(ComputeStorageResidencyKey),
}

/// Semantic state observed at a reset boundary before namespaces are rebuilt.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceResetEffect {
    pub translation_hold: Option<TranslationHoldAtReset>,
}

/// Guest work still parked behind shader translation when reset retired it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TranslationHoldAtReset {
    pub held_mask: u32,
    pub producer_mask: u32,
    pub episodes: u64,
}

/// Namespace entries retired by a task lifetime transition.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TaskNamespaceRetirement {
    pub heaps: usize,
    pub compute_pipelines: usize,
    pub depth_stencil_states: usize,
    pub render_pipelines: usize,
    pub functions: usize,
    pub indirect_command_buffers: usize,
    pub fences: usize,
    pub events: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskDefinitionKind {
    FirstDefinition,
    RedefinedSameRoot,
    RedefinedNewRoot,
}

/// Complete model effect of defining or redefining a task.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TaskDefinitionEffect {
    pub kind: TaskDefinitionKind,
    pub retired: TaskNamespaceRetirement,
}

/// Complete semantic result of withdrawing one mapping's cached page plan.
#[must_use = "mapping invalidation effects include observation state that runtime must publish"]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MappingInvalidationEffect {
    pub had_page_state: bool,
    pub dropped_host_cache: bool,
}

/// Semantic outcome of adopting a mapper-resolved physical page plan.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SurfacePlanAdoption {
    pub(crate) pages_changed: bool,
    pub(crate) previous_page_count: usize,
    pub(crate) lifecycle_generation: u32,
}

/// Semantic outcome of adopting a registered-surface backing plan.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RegisteredSurfacePlanAdoption {
    pub(crate) changed: bool,
    pub(crate) replaced: bool,
    pub(crate) lifecycle_generation: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SurfacePageRefresh {
    pub(crate) page_count: usize,
    pub(crate) page_generation: u32,
}

/// Typed release effects produced by model mutations which require host or
/// executor services to finish.
///
/// The model cannot call `HostOps`, and it must not own backend objects. It can
/// still describe the exact teardown work it caused. Consumers drain imports
/// before host views because a backend allocation may alias the view; linear
/// resident release is a separate executor effect.
#[derive(Debug, Default)]
struct PendingHostReleases {
    effects: Vec<HostReleaseEffect>,
}

impl PendingHostReleases {
    fn retire_view(&mut self, view: (usize, usize)) {
        self.effects.push(HostReleaseEffect::ReleaseView {
            ptr: view.0,
            len: view.1,
        });
    }

    fn retire_guest_import(&mut self, import: reims_vgpu_memory::ImportId) {
        self.effects
            .push(HostReleaseEffect::RetireGuestImport(import));
    }

    fn retire_materialization(
        &mut self,
        view: Option<(usize, usize)>,
        import: Option<reims_vgpu_memory::ImportId>,
    ) {
        match (view, import) {
            (Some((ptr, len)), Some(import)) => self
                .effects
                .push(HostReleaseEffect::RetireImportedView { import, ptr, len }),
            (Some(view), None) => self.retire_view(view),
            (None, Some(import)) => self.retire_guest_import(import),
            (None, None) => {}
        }
    }

    fn retire_compute_resident(&mut self, identity: ComputeStorageResidencyKey) {
        self.effects
            .push(HostReleaseEffect::RetireComputeResident(identity));
    }

    fn take_compute_residents(&mut self) -> Vec<ComputeStorageResidencyKey> {
        self.take_matching(|effect| match effect {
            HostReleaseEffect::RetireComputeResident(identity) => Some(identity),
            HostReleaseEffect::RetireGuestImport(_)
            | HostReleaseEffect::RetireImportedView { .. }
            | HostReleaseEffect::ReleaseView { .. } => None,
        })
    }

    /// Drain alias-bearing host releases in their dependency order.
    /// Imports are revoked before any view they may alias is unmapped.
    fn take_host_view_effects(&mut self) -> Vec<HostReleaseEffect> {
        let effects = self.take_matching(|effect| match effect {
            HostReleaseEffect::RetireGuestImport(_)
            | HostReleaseEffect::RetireImportedView { .. }
            | HostReleaseEffect::ReleaseView { .. } => Some(effect),
            HostReleaseEffect::RetireComputeResident(_) => None,
        });
        let mut imports = Vec::new();
        let mut views = Vec::new();
        for effect in effects {
            match effect {
                HostReleaseEffect::RetireGuestImport(_)
                | HostReleaseEffect::RetireImportedView { .. } => imports.push(effect),
                HostReleaseEffect::ReleaseView { .. } => views.push(effect),
                HostReleaseEffect::RetireComputeResident(_) => unreachable!(),
            }
        }
        imports.extend(views);
        imports
    }

    fn has_compute_residents(&self) -> bool {
        self.effects
            .iter()
            .any(|effect| matches!(effect, HostReleaseEffect::RetireComputeResident(_)))
    }

    #[cfg(test)]
    fn views(&self) -> Vec<(usize, usize)> {
        self.effects
            .iter()
            .filter_map(|effect| match *effect {
                HostReleaseEffect::ReleaseView { ptr, len }
                | HostReleaseEffect::RetireImportedView { ptr, len, .. } => Some((ptr, len)),
                HostReleaseEffect::RetireGuestImport(_)
                | HostReleaseEffect::RetireComputeResident(_) => None,
            })
            .collect()
    }

    #[cfg(test)]
    fn guest_imports(&self) -> Vec<reims_vgpu_memory::ImportId> {
        self.effects
            .iter()
            .filter_map(|effect| match *effect {
                HostReleaseEffect::RetireGuestImport(import) => Some(import),
                HostReleaseEffect::RetireImportedView { import, .. } => Some(import),
                HostReleaseEffect::ReleaseView { .. }
                | HostReleaseEffect::RetireComputeResident(_) => None,
            })
            .collect()
    }

    #[cfg(test)]
    fn compute_residents(&self) -> Vec<ComputeStorageResidencyKey> {
        self.effects
            .iter()
            .filter_map(|effect| match *effect {
                HostReleaseEffect::RetireComputeResident(identity) => Some(identity),
                HostReleaseEffect::RetireGuestImport(_) | HostReleaseEffect::ReleaseView { .. } => {
                    None
                }
                HostReleaseEffect::RetireImportedView { .. } => None,
            })
            .collect()
    }

    fn take_matching<T>(
        &mut self,
        mut select: impl FnMut(HostReleaseEffect) -> Option<T>,
    ) -> Vec<T> {
        let mut selected = Vec::new();
        let mut retained = Vec::new();
        for effect in std::mem::take(&mut self.effects) {
            if let Some(value) = select(effect) {
                selected.push(value);
            } else {
                retained.push(effect);
            }
        }
        self.effects = retained;
        selected
    }
}

/// Host aliases and the ordered release effects produced by their lifetimes.
///
/// A guest-page import may alias a mapped host view, so retirement always
/// queues import revocation before view unmapping. GVA-view removal also queues
/// its view effect in the same transition; callers cannot drop the registry
/// entry while forgetting the host mapping it owns.
#[derive(Debug, Default)]
pub(crate) struct HostMaterializationState {
    gva_views: Vec<GvaHostView>,
    releases: PendingHostReleases,
}

impl HostMaterializationState {
    #[cfg(test)]
    pub(crate) fn retire_view(&mut self, view: (usize, usize)) {
        self.releases.retire_view(view);
    }

    pub(crate) fn retire_guest_import(&mut self, import: reims_vgpu_memory::ImportId) {
        self.releases.retire_guest_import(import);
    }

    pub(crate) fn retire_materialization(
        &mut self,
        view: Option<(usize, usize)>,
        import: Option<reims_vgpu_memory::ImportId>,
    ) {
        self.releases.retire_materialization(view, import);
    }

    pub(crate) fn retire_compute_resident(&mut self, identity: ComputeStorageResidencyKey) {
        self.releases.retire_compute_resident(identity);
    }

    pub(crate) fn take_compute_residents(&mut self) -> Vec<ComputeStorageResidencyKey> {
        self.releases.take_compute_residents()
    }

    pub(crate) fn has_compute_residents(&self) -> bool {
        self.releases.has_compute_residents()
    }

    pub(crate) fn take_host_view_effects(&mut self) -> Vec<HostReleaseEffect> {
        self.releases.take_host_view_effects()
    }

    pub(crate) fn publish_gva_view(&mut self, view: GvaHostView) {
        self.gva_views.push(view);
    }

    pub(crate) fn find_gva_view(
        &self,
        predicate: impl Fn(&GvaHostView) -> bool,
    ) -> Option<&GvaHostView> {
        self.gva_views.iter().find(|view| predicate(view))
    }

    pub(crate) fn find_gva_view_mut(
        &mut self,
        predicate: impl Fn(&GvaHostView) -> bool,
    ) -> Option<&mut GvaHostView> {
        self.gva_views.iter_mut().find(|view| predicate(view))
    }

    pub(crate) fn retire_gva_views_where(
        &mut self,
        predicate: impl Fn(&GvaHostView) -> bool,
    ) -> u32 {
        let mut retired = 0u32;
        let mut index = 0usize;
        while index < self.gva_views.len() {
            if predicate(&self.gva_views[index]) {
                let mut view = self.gva_views.swap_remove(index);
                let host_view = view.take_host_view();
                let import = view.take_owned_import();
                self.releases.retire_materialization(host_view, import);
                retired = retired.saturating_add(1);
            } else {
                index += 1;
            }
        }
        retired
    }

    pub(crate) fn retire_all_gva_views(&mut self) {
        for mut view in self.gva_views.drain(..) {
            let host_view = view.take_host_view();
            let import = view.take_owned_import();
            self.releases.retire_materialization(host_view, import);
        }
    }

    #[cfg(test)]
    pub(crate) fn views(&self) -> &[GvaHostView] {
        &self.gva_views
    }

    #[cfg(test)]
    pub(crate) fn queued_views(&self) -> Vec<(usize, usize)> {
        self.releases.views()
    }

    #[cfg(test)]
    pub(crate) fn queued_guest_imports(&self) -> Vec<reims_vgpu_memory::ImportId> {
        self.releases.guest_imports()
    }

    #[cfg(test)]
    pub(crate) fn queued_compute_residents(&self) -> Vec<ComputeStorageResidencyKey> {
        self.releases.compute_residents()
    }
}

#[cfg(test)]
mod host_materialization_state_tests {
    use super::*;

    #[test]
    fn retiring_a_registered_view_queues_it_and_imports_drain_first() {
        let mut materializations = HostMaterializationState::default();
        materializations.publish_gva_view(GvaHostView::fixture(3, 0x1000, 0x1000, 0xaaaa, 0x1000));
        materializations.publish_gva_view(GvaHostView::fixture(4, 0x2000, 0x1000, 0xbbbb, 0x1000));
        assert_eq!(
            materializations.retire_gva_views_where(|view| view.task_id == 3),
            1
        );
        assert_eq!(materializations.views().len(), 1);

        let import = reims_vgpu_memory::GuestRamImport::new_host_allocation(0x3000, 0x1000, 0x1000)
            .unwrap()
            .id();
        materializations.retire_guest_import(import);
        assert_eq!(
            materializations.take_host_view_effects(),
            vec![
                HostReleaseEffect::RetireGuestImport(import),
                HostReleaseEffect::ReleaseView {
                    ptr: 0xaaaa,
                    len: 0x1000
                }
            ]
        );
    }
}

/// Device-scoped content authority and its derived/revalidation ledgers.
///
/// These owners answer different parts of one question—where the current
/// content exists and whether a guest or host copy may be used. Keeping them in
/// one aggregate makes reset and task retirement cross one semantic boundary;
/// topology and executor placement remain outside it.
#[derive(Debug)]
pub(crate) struct ContentAuthorityState {
    pub(crate) sampled: SampledContentState,
    pub(crate) gva_stores: reims_vgpu_core::GvaStoreWitness,
    pub(crate) preconstruction_writes: reims_vgpu_core::BufferWriteGens,
    pub(crate) host_writes: reims_vgpu_core::HostWrites,
    pub(crate) compute_residency: reims_vgpu_core::ComputeResidencyLedger,
    pub(crate) pending_writebacks: reims_vgpu_core::PendingWritebacks,
}

impl ContentAuthorityState {
    fn new(page_shift: u32, policies: reims_vgpu_core::GatherPolicies) -> Self {
        Self {
            sampled: SampledContentState::new(policies),
            gva_stores: Default::default(),
            preconstruction_writes: Default::default(),
            host_writes: reims_vgpu_core::HostWrites::new(page_shift),
            compute_residency: Default::default(),
            pending_writebacks: Default::default(),
        }
    }

    fn retire_task(&mut self, task_id: u32) {
        self.gva_stores.retire_task(task_id);
        self.sampled.gather_witness.retire_task(task_id);
        self.preconstruction_writes.retire_task(task_id);
    }
}

/// Task-local object reference spaces and their shared task lifetime.
///
/// The serializer assigns independent references to resources, samplers,
/// pipelines, functions, depth-stencil states, ICBs, fences, and events. They
/// must therefore remain separate namespaces. What they share is the task
/// lifetime: redefining or deleting a task retires every namespace as one
/// transition, so adding another task-owned object family cannot leave a
/// parallel cleanup path behind on [`DeviceState`].
#[derive(Debug, Default)]
pub(crate) struct TaskObjectNamespaces {
    pub(crate) resources: TaskResources,
    pub(crate) heaps: TaskHeapStates,
    pub(crate) samplers: TaskSamplerStates,
    pub(crate) compute_pipelines: TaskComputePipelineStates,
    pub(crate) functions: TaskFunctionStates,
    pub(crate) depth_stencil: TaskDepthStencilStates,
    pub(crate) render_pipelines: TaskRenderPipelineStates,
    pub(crate) indirect_command_buffers: reims_vgpu_core::IcbRegistry,
    pub(crate) fences: TaskFenceStates,
    pub(crate) events: TaskEventStates,
}

impl TaskObjectNamespaces {
    fn retire_task(&mut self, task_id: u32) -> TaskNamespaceRetirement {
        self.resources.delete_task(task_id);
        self.samplers.delete_task(task_id);
        TaskNamespaceRetirement {
            heaps: self.heaps.delete_task(task_id),
            compute_pipelines: self.compute_pipelines.delete_task(task_id),
            depth_stencil_states: self.depth_stencil.delete_task(task_id),
            render_pipelines: self.render_pipelines.delete_task(task_id),
            functions: self.functions.delete_task(task_id),
            indirect_command_buffers: self.indirect_command_buffers.delete_task(task_id),
            fences: self.fences.delete_task(task_id),
            events: self.events.delete_task(task_id),
        }
    }
}

/// Deliberately incomplete identities used only by packet-level fixtures.
///
/// These tests exercise deletion and routing records without constructing the
/// descriptors which publish canonical [`TaskResource`] relations. Keeping the
/// shortcuts behind one test-only boundary prevents either map from appearing
/// in product state or becoming an accidental second source of authority.
#[cfg(test)]
#[derive(Debug, Default)]
pub(crate) struct SyntheticFixtureState {
    pub(crate) objects: std::collections::BTreeSet<(u32, u32)>,
    pub(crate) texture_to_mapping: BTreeMap<(u32, u32), u32>,
}

/// Canonical surface namespace with explicitly separate construction rails.
///
/// `mappings` owns shared surface declarations, page plans, content state, and
/// host materialization. `mapper` owns only the arm mapper-service lookup and
/// capture lifecycle. The latter can resolve the former through an explicit
/// typed edge; equal integers never merge their namespaces.
#[derive(Debug, Default)]
pub(crate) struct SurfaceState {
    pub(crate) mappings: SurfaceMappingRegistry,
    mapper: MapperService,
}

/// Display presentation, cursor publication, and guest handshake ownership.
///
/// These states share the display lifecycle but not a representation: present
/// routing owns frame/backing evidence, the cursor owns its atomic host
/// snapshot, and the handshake owns shared-page online progress. Keeping them
/// under one device boundary prevents unrelated resource or scheduling state
/// from becoming a second display owner.
#[derive(Debug)]
pub struct PresentationState {
    pub(crate) present: PresentState,
    pub(crate) cursor: CursorState,
    pub(crate) display: DisplayHandshake,
}

impl Default for PresentationState {
    fn default() -> Self {
        Self {
            present: PresentState::default(),
            cursor: CursorState::initially_visible(),
            display: DisplayHandshake::default(),
        }
    }
}

/// Full device model state (backend-independent).
#[derive(Debug)]
pub struct DeviceState {
    pub id: DeviceId,
    /// Identity, participation, segment cursor, and completion for the active
    /// decoded submission.
    pub(crate) submissions: reims_vgpu_core::SubmissionTracker,
    /// Guest page shift for PFN↔GPA wire math (12 = x86, 14 = arm64e).
    pub page_shift: u32,
    /// Guest-visible register banks. Transport adapters mutate these banks;
    /// semantic scheduling consumes their published state.
    pub registers: DeviceRegisters,
    /// FIFO ingress, rings, translation ordering, and nested drain ownership.
    pub(crate) scheduling: WorkSchedulingState,
    pub tasks: TaskTable,
    /// Observation-only ledgers, separate from lifecycle and content state.
    pub(crate) observations: reims_vgpu_core::DeviceObservations,
    /// Packet-test shortcuts which deliberately omit canonical construction.
    #[cfg(test)]
    pub(crate) fixtures: SyntheticFixtureState,
    /// Independent object reference spaces with one task-retirement boundary.
    pub(crate) task_objects: TaskObjectNamespaces,
    /// Shared surface state with distinct arm mapper and registered-surface rails.
    pub(crate) surfaces: SurfaceState,
    /// Host-derived content replicas and their replacement bookkeeping.
    /// Guest object, mapping, and GVA identities only name entries owned here.
    pub host_replicas: HostReplicaState,
    /// Content currency, revalidation, residency, and writeback obligations.
    pub(crate) content: ContentAuthorityState,
    /// Present routing, cursor publication, and shared-page handshake.
    pub(crate) presentation: PresentationState,
    /// Every `FailEvent` also reached the always-on log through `record_fail`;
    /// this vec is only how an in-crate test reads them back. It is
    /// `#[cfg(test)]` because in a product boot nothing ever read it, so it grew
    /// for the life of the guest holding the one copy of nothing.
    #[cfg(test)]
    pub fails: Vec<FailEvent>,
    /// Host/executor release effects emitted by mapping and object lifetime
    /// transitions. Drained in dependency order by the composition runtime.
    pub(crate) host_materializations: HostMaterializationState,
    /// Completion words from coalescing through guest visibility.
    ///
    /// This owner keeps the owed/queued ledger and its progress witness in one
    /// state machine, so a drain path cannot substitute an unrelated counter
    /// for publication progress.
    pub(crate) completion_publications: reims_vgpu_core::CompletionPublications,
}

impl DeviceState {
    /// Content currency in the resource namespace that owns it.
    ///
    /// Constructed resources always use the canonical graph. The legacy
    /// counter is consulted only when construction has not yet established a
    /// resource identity, preserving validity records which arrive first.
    pub fn resource_write_stamp(
        &self,
        task_id: u32,
        object_id: u32,
    ) -> reims_vgpu_core::ResourceWriteStamp {
        use reims_vgpu_core::ResourceWriteStamp;
        self.task_objects
            .resources
            .content_stamp(task_id, object_id)
            .map(|(resource, version)| ResourceWriteStamp::Resolved { resource, version })
            .unwrap_or_else(|| {
                ResourceWriteStamp::Unresolved(
                    self.content
                        .preconstruction_writes
                        .stamp(task_id, object_id),
                )
            })
    }

    /// Content currency for an already resolved resource lifetime.
    pub fn resource_write_stamp_for(
        &self,
        resource: ResourceId<ResourceObject>,
    ) -> Option<reims_vgpu_core::ResourceWriteStamp> {
        self.task_objects
            .resources
            .content_version_for(resource)
            .map(|version| reims_vgpu_core::ResourceWriteStamp::Resolved { resource, version })
    }

    /// GPA for a guest PFN under this device's page size.
    #[inline]
    pub fn pfn_gpa(&self, pfn: u32) -> u64 {
        (pfn as u64) << self.page_shift
    }

    #[inline]
    pub fn page_size(&self) -> u64 {
        1u64 << self.page_shift
    }

    /// Create device state for a guest with the given page shift.
    ///
    /// `page_shift` must be **12** (x86_64 / Tahoe) or **14** (arm64e). There
    /// is no default — product create and tests must choose explicitly.
    pub fn new(id: DeviceId, page_shift: u32) -> Self {
        Self::new_with_gather_policies(id, page_shift, reims_vgpu_core::GatherPolicies::default())
    }

    pub(crate) fn new_with_gather_policies(
        id: DeviceId,
        page_shift: u32,
        policies: reims_vgpu_core::GatherPolicies,
    ) -> Self {
        Self {
            id,
            submissions: reims_vgpu_core::SubmissionTracker::default(),
            page_shift,
            registers: DeviceRegisters::default(),
            scheduling: WorkSchedulingState::default(),
            tasks: TaskTable::new(),
            observations: reims_vgpu_core::DeviceObservations::default(),
            #[cfg(test)]
            fixtures: SyntheticFixtureState::default(),
            task_objects: TaskObjectNamespaces::default(),
            surfaces: SurfaceState::default(),
            host_replicas: HostReplicaState::default(),
            content: ContentAuthorityState::new(page_shift, policies),
            presentation: PresentationState::default(),
            #[cfg(test)]
            fails: Vec::new(),
            host_materializations: HostMaterializationState::default(),
            completion_publications: Default::default(),
        }
    }

    /// Detach `e`'s contiguous view for later unmap (page table changed).
    /// Returns the retired `(ptr, len)` host-view release effect.
    fn take_mapping_view(
        e: &mut SurfaceMappingEntry,
    ) -> (Option<(usize, usize)>, Option<reims_vgpu_memory::ImportId>) {
        e.materialization.retire()
    }

    /// Retire sampled-window witnesses bound to one mapping lifetime.
    fn retire_mapping_gather_witness(&mut self, mapping_id: u32) {
        self.content
            .sampled
            .gather_witness
            .retire_mapping(mapping_id);
    }

    /// Detach every host mapping owned by the current guest lifetime.
    ///
    /// Device reset is a lifetime boundary even when QEMU itself remains alive.
    /// The returned effects put import retirement before view release, so a
    /// caller cannot accidentally unmap an alias while the executor still
    /// accepts children against its import identity.
    pub fn take_all_host_release_effects(&mut self) -> Vec<HostReleaseEffect> {
        for mapping in self.surfaces.mappings.values_mut() {
            let (view, import) = Self::take_mapping_view(mapping);
            self.host_materializations
                .retire_materialization(view, import);
        }
        self.content.sampled.gather_witness.clear();
        self.content.gva_stores.clear();
        self.host_materializations.retire_all_gva_views();
        self.host_materializations.take_host_view_effects()
    }

    /// Snapshot the generation of one fence object if it has been updated.
    pub fn fence_generation(&self, task_id: u32, fence_ref: u32) -> Option<u64> {
        self.task_objects.fences.generation(task_id, fence_ref)
    }

    /// Store fence generation (monotonic update owned by the planner).
    pub fn set_fence_generation(&mut self, task_id: u32, fence_ref: u32, value: u64) {
        if fence_ref == 0 {
            return;
        }
        self.task_objects.fences.set_update(
            task_id,
            fence_ref,
            value,
            reims_vgpu_core::FenceSignal::Compute,
        );
    }

    pub fn set_fence_update(
        &mut self,
        task_id: u32,
        fence_ref: u32,
        value: u64,
        signal: reims_vgpu_core::FenceSignal,
    ) {
        if fence_ref == 0 {
            return;
        }
        self.task_objects
            .fences
            .set_update(task_id, fence_ref, value, signal);
    }

    pub fn fence_signal(
        &self,
        task_id: u32,
        fence_ref: u32,
    ) -> Option<reims_vgpu_core::FenceSignal> {
        self.task_objects.fences.signal(task_id, fence_ref)
    }

    pub fn event_generation(&self, task_id: u32, event_ref: u32) -> Option<u64> {
        self.task_objects.events.generation(task_id, event_ref)
    }

    pub fn set_event_generation(&mut self, task_id: u32, event_ref: u32, value: u64) {
        if event_ref == 0 {
            return;
        }
        self.task_objects
            .events
            .set_generation(task_id, event_ref, value);
    }

    pub(crate) fn delete_fence(&mut self, task_id: u32, fence_ref: u32) -> bool {
        self.task_objects.fences.delete(task_id, fence_ref)
    }

    #[cfg(test)]
    pub(crate) fn fence_identity(
        &self,
        task_id: u32,
        fence_ref: u32,
    ) -> Option<ResourceId<FenceObject>> {
        self.task_objects.fences.identity(task_id, fence_ref)
    }

    /// Record a clear-only write to `mapping_id` (display_clear / CLEAR Store).
    pub fn note_surface_clear(&mut self, mapping_id: u32) {
        if mapping_id == 0 {
            return;
        }
        // Guest Clear wipes the surface: next present of this mid must not be
        // treated as a finished composite (unless a later Draw Store re-marks
        // Composite).
        self.presentation
            .present
            .write_kind
            .insert(SurfaceId::new(mapping_id), SurfaceWriteKind::ClearOnly);
    }

    /// Record a composite/draw Store to `mapping_id`.
    pub fn note_surface_composite(&mut self, mapping_id: u32) {
        if mapping_id == 0 {
            return;
        }
        self.presentation
            .present
            .write_kind
            .insert(SurfaceId::new(mapping_id), SurfaceWriteKind::Composite);
    }

    /// A draw Store published a **complete** frame for `mapping_id` into guest
    /// pages (full-frame resident writeback, `import_present ok_runs`).
    ///
    /// Protocol-structural dense marker: this mapping now holds a complete full
    /// frame, so advance its publication witness. A surface presented twice with no
    /// advance in between received no full frame of its own, which is the
    /// `present_unbacked` gate in [`Self::note_present_backing`] — the only
    /// reader. The counter is monotonic per full-frame Store across all
    /// mappings, so the value is a witness of "something was published for this
    /// mid", never a staleness measure on its own.
    pub fn note_dense_frame_published(&mut self, mapping_id: u32, width: u32, height: u32) {
        if mapping_id == 0 || width == 0 || height == 0 {
            return;
        }
        self.presentation
            .present
            .backing_evidence
            .publish(mapping_id);
    }

    /// Advance the per-present epoch counter and return the new value. Call
    /// exactly once per present cycle.
    pub fn advance_present_epoch(&mut self) -> u64 {
        self.presentation.present.advance_present_epoch()
    }

    /// Record that `mapping_id` is being presented and report whether the guest
    /// ever sent a full-frame Store **naming it** for what is about to be shown.
    ///
    /// Structural only: decoded Store bookkeeping, never measured content, and
    /// never the resident. Say what that leaves out, because the name reads
    /// broader than the check: a `None` here means the guest sent a frame for
    /// this mid, **not** that the resident this present will read holds it. See
    /// [`PresentBackingEvidence`].
    ///
    /// Records the witness on every call, so a member that stays unbacked
    /// reports once per present rather than once per lifetime — except
    /// [`PresentBacking::NeverStored`], which by construction can only be
    /// reported on a mapping's first present since it was created.
    pub fn note_present_backing(&mut self, mapping_id: u32) -> Option<PresentBacking> {
        if mapping_id == 0 {
            return None;
        }
        self.presentation
            .present
            .backing_evidence
            .present(mapping_id)
    }

    #[cfg(test)]
    pub(crate) fn present_backing_sequence_for_test(&self, mapping_id: u32) -> u64 {
        self.presentation
            .present
            .backing_evidence
            .sequence_for(mapping_id)
    }

    #[cfg(test)]
    pub(crate) fn copy_present_backing_sequence_for_test(&mut self, source: u32, target: u32) {
        self.presentation
            .present
            .backing_evidence
            .copy_sequence(source, target);
    }

    fn forget_compositor_mapping(&mut self, mapping_id: u32) {
        self.presentation
            .present
            .write_kind
            .remove(&SurfaceId::new(mapping_id));
        self.presentation
            .present
            .backing_evidence
            .retire(mapping_id);
    }

    /// Last write class for present keep-prior decisions.
    pub fn surface_write_kind(&self, mapping_id: u32) -> SurfaceWriteKind {
        self.presentation
            .present
            .write_kind
            .get(&SurfaceId::new(mapping_id))
            .copied()
            .unwrap_or(SurfaceWriteKind::Unknown)
    }

    pub(crate) fn reset(&mut self) -> DeviceResetEffect {
        // A translation hold that is still standing here never resolved. The
        // hold itself is control flow — the FIFO is parked until an AIR module
        // finishes loading and the packet is retried, not consumed — so it is
        // census. THIS is the failure: the device went away with guest packets
        // still parked behind a load that never completed, and those packets are
        // lost. Reading it at the lifetime boundary needs no age, depth or
        // timeout; the guest's own teardown is the deadline.
        let effect = DeviceResetEffect {
            translation_hold: self.scheduling.translation.unreleased().map(|hold| {
                TranslationHoldAtReset {
                    held_mask: hold.held_mask,
                    producer_mask: hold.producer_mask,
                    episodes: hold.episodes,
                }
            }),
        };
        let id = self.id;
        let page_shift = self.page_shift;
        let policies = self.content.sampled.gather_witness.policies();
        // Keep the interrupt-status Arcs wired to the registry slot: the
        // lock-free ISR read rail clones them once at device create.
        let intr_disp = Arc::clone(&self.registers.gfx.interrupt_status_disp);
        let intr_gpu = Arc::clone(&self.registers.gfx.interrupt_status_gpu);
        let intr_fault = Arc::clone(&self.registers.gfx.interrupt_fault);
        let fifo_read = Arc::clone(&self.registers.gfx.fifo_read);
        let child_rung = Arc::clone(&self.registers.gfx.child_doorbell_rung);
        intr_disp.store(0, Ordering::Release);
        intr_gpu.store(0, Ordering::Release);
        intr_fault.store(0, Ordering::Release);
        fifo_read.store(0, Ordering::Release);
        // Cleared as well as kept: a reset drops every channel, so a bit rung
        // before it names a channel that no longer exists.
        child_rung.store(0, Ordering::Release);
        *self = Self::new_with_gather_policies(id, page_shift, policies);
        self.registers.gfx.interrupt_status_disp = intr_disp;
        self.registers.gfx.interrupt_status_gpu = intr_gpu;
        self.registers.gfx.interrupt_fault = intr_fault;
        self.registers.gfx.fifo_read = fifo_read;
        self.registers.gfx.child_doorbell_rung = child_rung;
        effect
    }

    /// Queue the engine-unpin for a dying linear cache entry that still owns a
    /// resident image.
    fn retire_linear_resident(&mut self, task_id: u32, texture_ref: u32, e: &HostLinearTexture) {
        if e.resident_gen == 0 || e.row_stride > u32::MAX as u64 {
            return;
        }
        let Some(resource) = self.task_objects.resources.identity(task_id, texture_ref) else {
            return;
        };
        self.host_materializations
            .retire_compute_resident(ComputeStorageResidencyKey::linear(
                resource,
                e.gva,
                e.row_stride as u32,
                e.row_stride.saturating_mul(e.height as u64),
                e.width,
                e.height,
                e.pixel_format,
            ));
    }

    fn retire_task_linear_residents(&mut self, task_id: u32) {
        let doomed = self.host_replicas.take_task_linear(task_id);
        for (r, e) in doomed {
            self.retire_linear_resident(task_id, r, &e);
        }
    }

    fn retire_compute_residency_keys(
        &mut self,
        keys: impl IntoIterator<Item = ComputeStorageResidencyKey>,
    ) {
        for key in keys {
            self.host_materializations.retire_compute_resident(key);
        }
    }

    fn retire_heap_identity(&mut self, heap: ResourceId<HeapObject>) {
        let keys = self.content.compute_residency.retire_heap(heap);
        self.retire_compute_residency_keys(keys);
    }

    fn retire_heap_resource(&mut self, task_id: u32, texture_ref: u32) {
        let Some(retirement) = self
            .task_objects
            .resources
            .heap_resource_retirement(task_id, texture_ref)
        else {
            return;
        };
        let mut keys = self
            .content
            .compute_residency
            .retire_resource(retirement.resource);
        if let Some(origin) = retirement.storage_origin {
            keys.extend(self.content.compute_residency.retire_origin(origin));
        }
        self.retire_compute_residency_keys(keys);
    }

    pub(crate) fn delete_heap(
        &mut self,
        task_id: u32,
        reference: reims_vgpu_protocol::SerializerRef<HeapObject>,
    ) -> bool {
        let Some(heap) = self.task_objects.heaps.identity(task_id, reference) else {
            return false;
        };
        self.retire_heap_identity(heap);
        self.task_objects.heaps.delete(task_id, reference)
    }

    fn retire_task_namespaces(&mut self, task_id: u32) -> TaskNamespaceRetirement {
        let heaps = self.task_objects.heaps.identities_for_task(task_id);
        for heap in heaps {
            self.retire_heap_identity(heap);
        }
        self.task_objects.retire_task(task_id)
    }

    /// Install the guest's task under `task_id`, replacing any previous one.
    ///
    /// The returned effect describes the semantic mutation without choosing
    /// how a runtime observes it. No guest task id can be refused: the table is
    /// keyed by the guest's full `u32` namespace.
    pub fn define_task(
        &mut self,
        task_id: u32,
        length: u64,
        directory_pfn: u32,
    ) -> TaskDefinitionEffect {
        self.observations.observe_task_id(task_id);
        // Redefining a *live* task is the one shape here that can lose published
        // guest state: the objects below are dropped, and if the new directory
        // roots a different physical page at the list's own GVA then everything
        // the guest published into the old one reads back as zero. macOS 13 does
        // not do this and macOS 26 does, which is why it is counted separately
        // from a first definition rather than folded into one route.
        let kind = if self.tasks.is_active(task_id) {
            let same_root = self
                .tasks
                .get(task_id)
                .is_some_and(|t| t.directory_pfn == directory_pfn);
            if same_root {
                TaskDefinitionKind::RedefinedSameRoot
            } else {
                TaskDefinitionKind::RedefinedNewRoot
            }
        } else {
            TaskDefinitionKind::FirstDefinition
        };
        // Drop objects for this task on redefine.
        #[cfg(test)]
        self.fixtures.objects.retain(|&(t, _)| t != task_id);
        self.retire_task_linear_residents(task_id);
        let retired = self.retire_task_namespaces(task_id);
        // A deleted task's whole address space goes with it, so its live
        // mappings are not leaks and a reused id must not inherit them.
        self.observations.map_audit.remove(&task_id);
        // Same lifetime, same reason: the watched pages were nodes of the tree
        // this id is losing, and after a redefine they describe whatever the
        // guest has since done with them.
        self.observations.node_guard.remove(&task_id);
        // New directory ⇒ old GVA HostOps views alias the wrong PT — retire.
        self.retire_task_gva_views(task_id);
        self.content.retire_task(task_id);
        self.tasks
            .define(task_id, TaskEntry::define(length, directory_pfn));
        TaskDefinitionEffect { kind, retired }
    }

    /// Retire every GVA HostOps view registered under `task_id`.
    ///
    /// Both entry points that end a task's page table — `define_task` on a
    /// redefine and `delete_task` on teardown — owe exactly this: the views hold
    /// host pointers into pages the guest is about to recycle, so leaving one
    /// live is a read of memory that no longer belongs to the surface (the
    /// WindowServer SIGSEGV class [`crate::runtime::gva_view::write_span_within`]
    /// documents). The typed view-release effects are drained by
    /// `mapper::flush_retired_views` through `HostOps::unmap_pages`.
    fn retire_task_gva_views(&mut self, task_id: u32) {
        self.host_materializations
            .retire_gva_views_where(|view| view.task_id == task_id);
    }

    /// PVG `CmdDeleteTask` (op `0x20`): drop task directory + object list entries.
    /// Guest reuses task ids; leaving stale active tasks corrupts GVA walks.
    pub fn delete_task(&mut self, task_id: u32) -> Option<TaskNamespaceRetirement> {
        self.observations.observe_task_id(task_id);
        if !self.tasks.is_active(task_id) {
            return None;
        }
        #[cfg(test)]
        self.fixtures.objects.retain(|&(t, _)| t != task_id);
        self.retire_task_linear_residents(task_id);
        let retired = self.retire_task_namespaces(task_id);
        self.host_replicas.forget_task_textures(task_id);
        // Clear texture→mapping latches for this task.
        #[cfg(test)]
        self.fixtures
            .texture_to_mapping
            .retain(|&(t, _), _| t != task_id);
        // GVA encode cache retained until Unmap of that range.
        // Task teardown ≡ all GPU VA maps for this task go away — retire any
        // HostOps views we held (does not touch host_gva_surfaces encode).
        // Runtime drains the typed view effects via HostOps::unmap_pages.
        self.retire_task_gva_views(task_id);
        self.content.retire_task(task_id);
        // The two observation ledgers keyed by task id go with it, exactly as
        // they do on a redefine. Both were reachable only through `define_task`
        // before, which cleaned them up whenever an id came back — so a task the
        // guest deletes and never redefines left its record behind for the life
        // of the process. Neither ledger is read for a task that does not exist,
        // so this costs no behaviour; it stops an id the guest is done with from
        // holding a page set that describes memory it has given back.
        self.observations.map_audit.remove(&task_id);
        self.observations.node_guard.remove(&task_id);
        self.tasks.remove(task_id);
        Some(retired)
    }

    pub fn set_object_list(&mut self, task_id: u32, pfn: u32, count: u32) -> bool {
        self.try_set_object_list(task_id, pfn, count).is_ok()
    }

    pub(crate) fn try_set_object_list(
        &mut self,
        task_id: u32,
        pfn: u32,
        count: u32,
    ) -> Result<(), StateMutationDecline> {
        self.observations.observe_task_id(task_id);
        if !self.tasks.is_active(task_id) {
            return Err(StateMutationDecline::SetObjectListTaskInactive { task_id });
        }
        // A replacement list gives every reference a new construction input.
        // Pre-construction currency belongs to the old naming lifetime just as
        // retained resources and address materializations do.
        self.content.preconstruction_writes.retire_task(task_id);
        let task = self
            .tasks
            .get_mut(task_id)
            .expect("an active task has a task-table entry");
        task.object_list_pfn = pfn;
        task.object_list_count = count;
        Ok(())
    }

    /// Every mapping id one task-local object reference can name.
    ///
    /// This device carries two ways from a reference to a surface, because the
    /// guest has two: on some paths the reference *is* the mapping id, and on
    /// the retained task resource holds the per-task registration an IOSurface
    /// texture create recorded. A statement about the reference — a validity
    /// quad, an owed render frame — is a statement about every mapping it
    /// names, so the candidate set is one rule and lives here.
    ///
    /// It is one rule because it used to be two, spelled differently, and only
    /// one of them was right about what "named nothing" means:
    /// `resource_validity::apply` built both candidates and then asked
    /// [`Self::surfaces`] which of them exists, while
    /// `writeback_debt::pay_for_texture` asked only whether the ledger held a
    /// debt and then reported "this reference named no surface" whenever the
    /// per-task registration was empty. The reference-is-the-mapping-id
    /// spelling never populates that registration, so that report was `100 %`
    /// of its own census on both arms of a driven macos-13 boot — a census
    /// whose whole purpose was to separate "nothing was owed" from "we could
    /// not look".
    ///
    /// Deduplicated, so a reference that is its own mapping id is one target
    /// and not two. Ordered as the guest's own namespaces are asked: the
    /// reference first, the registration second.
    pub fn mappings_named_by(&self, task_id: u32, object_id: u32) -> NamedMappings {
        if object_id == 0 {
            // `writeInvalidates` skips null resources and id 0; `pageBacking`
            // never emits one. A zero id names nothing.
            return NamedMappings::default();
        }
        NamedMappings::new(
            Some(object_id),
            self.registered_texture_mapping(task_id, object_id),
        )
    }

    /// Resolve the IOSurface mapping relation owned by a retained resource.
    ///
    /// Synthetic unit fixtures may supply the old side-map relation without a
    /// descriptor; product builds have no such map and therefore cannot
    /// dual-write or outlive the resource that owns this edge.
    pub fn registered_texture_mapping(&self, task_id: u32, object_id: u32) -> Option<u32> {
        #[cfg(test)]
        let legacy_fixture_mapping = self
            .fixtures
            .texture_to_mapping
            .get(&(task_id, object_id))
            .copied();
        #[cfg(not(test))]
        let legacy_fixture_mapping = None;
        self.task_objects
            .resources
            .get(task_id, object_id)
            .and_then(|resource| resource.registered_iosurface_mapping())
            .map(SurfaceId::get)
            .or(legacy_fixture_mapping)
    }

    /// Whether any mapping this reference names is one this device still holds.
    ///
    /// The question a reader asks before concluding that nothing was owed: a
    /// reference naming no live mapping did not *look*, and a reference naming
    /// one and finding no debt genuinely found nothing. Derived from
    /// [`Self::mappings_named_by`] so the two cannot answer about different
    /// candidate sets.
    pub fn names_live_mapping(&self, task_id: u32, object_id: u32) -> bool {
        self.mappings_named_by(task_id, object_id)
            .iter()
            .any(|id| self.surfaces.mappings.contains_key(&id))
    }

    #[cfg(test)]
    pub fn insert_object(&mut self, task_id: u32, ref_: u32) -> bool {
        self.try_insert_object(task_id, ref_).is_ok()
    }

    #[cfg(test)]
    pub(crate) fn try_insert_object(
        &mut self,
        task_id: u32,
        ref_: u32,
    ) -> Result<(), StateMutationDecline> {
        self.observations.observe_task_id(task_id);
        if !self.tasks.is_active(task_id) {
            return Err(StateMutationDecline::InsertObjectTaskInactive {
                task_id,
                object_ref: ref_,
            });
        }
        self.fixtures.objects.insert((task_id, ref_));
        Ok(())
    }

    pub fn delete_object(&mut self, task_id: u32, ref_: u32) -> bool {
        #[cfg(test)]
        let removed = self.fixtures.objects.remove(&(task_id, ref_));
        #[cfg(not(test))]
        let removed = false;
        let (texture_removed, linear_removed) = self.invalidate_object_host_copies(task_id, ref_);
        self.retire_heap_resource(task_id, ref_);
        let resource_removed = self.task_objects.resources.delete(task_id, ref_);
        self.content
            .preconstruction_writes
            .retire_object(task_id, ref_);
        #[cfg(test)]
        let mapping_removed = self
            .fixtures
            .texture_to_mapping
            .remove(&(task_id, ref_))
            .is_some();
        #[cfg(not(test))]
        let mapping_removed = false;
        removed || resource_removed || texture_removed || linear_removed || mapping_removed
    }

    /// Drop this device's ref-keyed host copies of an object's *contents*, for a
    /// packet saying the guest memory under it has changed. Returns which of the
    /// two held something, `(texture, linear)`.
    ///
    /// The two caches this covers are keyed by object-list ref rather than by
    /// mapping id, and neither carries a page list, so nothing in them can
    /// notice that the pages they were read from are no longer the object's.
    /// `invalidate_mapping_pages` is the same obligation on the mapping rail; a
    /// packet that reaches only one of the two rails still has to discharge it
    /// on that one.
    ///
    /// Contents only. The object stays alive, and so does its retained IOSurface
    /// mapping association — a re-point moves the bytes, it does not unname the resource.
    /// [`Self::delete_object`] takes both halves and calls this for its first.
    ///
    /// A live linear resident goes through [`Self::retire_linear_resident`], so
    /// it is unpinned and its deferred window dropped rather than left to write
    /// pixels read from the old pages into the new ones.
    pub fn invalidate_object_host_copies(&mut self, task_id: u32, ref_: u32) -> (bool, bool) {
        let (had_texture, linear) = self.host_replicas.take_object_replicas(task_id, ref_);
        let had_linear = match linear {
            Some(e) => {
                self.retire_linear_resident(task_id, ref_, &e);
                true
            }
            None => false,
        };
        (had_texture, had_linear)
    }

    /// Bump the mapping lifecycle generation (never 0 after first bump).
    ///
    /// The bump orphans any generation-keyed resident for the mapping.
    pub fn bump_map_generation(e: &mut SurfaceMappingEntry) {
        e.lifecycle.generation = e.lifecycle.generation.wrapping_add(1);
        if e.lifecycle.generation == 0 {
            e.lifecycle.generation = 1;
        }
    }

    /// Bump the physical page-plan generation (never 0 after first bump).
    ///
    /// This is deliberately separate from [`Self::bump_map_generation`]: the
    /// current physical backing of a live resource may change without creating
    /// a new resource incarnation.
    pub fn bump_page_generation(e: &mut SurfaceMappingEntry) {
        e.pages.generation = e.pages.generation.wrapping_add(1);
        if e.pages.generation == 0 {
            e.pages.generation = 1;
        }
    }

    /// Adopt one mapper-resolved page plan as an indivisible lifecycle change.
    ///
    /// A different plan advances both logical and physical generations and
    /// retires every host object tied to the prior pages.
    pub(crate) fn adopt_mapper_surface_plan(
        &mut self,
        surface: SurfaceId,
        entries: Vec<u32>,
        page_table_kva: u64,
        internal_kva: u64,
        device_desc: Option<&[u8]>,
    ) -> Option<SurfacePlanAdoption> {
        let (effect, retired_view, retired_import) = {
            let mapping = self.surfaces.mappings.get_mut(&surface.get())?;
            let previous_page_count = mapping.pages.entries.len();
            let pages_changed = mapping.pages.entries != entries;
            let (retired_view, retired_import) = if pages_changed {
                let retired = mapping.materialization.retire();
                Self::bump_map_generation(mapping);
                Self::bump_page_generation(mapping);
                retired
            } else {
                (None, None)
            };
            mapping.pages.entries = entries;
            mapping.pages.table_kva = page_table_kva;
            mapping.lifecycle.internal_kva = internal_kva;
            mapping.lifecycle.active = true;
            if let Some(device_desc) = device_desc {
                mapping.publish_device_desc(device_desc);
            }
            (
                SurfacePlanAdoption {
                    pages_changed,
                    previous_page_count,
                    lifecycle_generation: mapping.lifecycle.generation,
                },
                retired_view,
                retired_import,
            )
        };
        self.host_materializations
            .retire_materialization(retired_view, retired_import);
        Some(effect)
    }

    /// Adopt one registered-surface page plan and its derivation witness.
    pub(crate) fn adopt_registered_surface_plan(
        &mut self,
        surface: SurfaceId,
        entries: Vec<u32>,
        task: TaskId,
        backing_pfn: u32,
        device_desc: &[u8],
    ) -> Option<RegisteredSurfacePlanAdoption> {
        let (effect, retired_view, retired_import) = {
            let mapping = self.surfaces.mappings.get_mut(&surface.get())?;
            let prior = std::mem::take(&mut mapping.pages.entries);
            let changed = prior != entries;
            let replaced = !prior.is_empty() && changed;
            if changed {
                Self::bump_map_generation(mapping);
            }
            mapping.pages.entries = entries;
            mapping.lifecycle.active = true;
            mapping.pages.table_kva = 0;
            mapping.publish_device_desc(device_desc);
            mapping.pages.surface_walk = Some(SurfaceBackingWalk {
                task_id: task.get(),
                backing_pfn,
                page_generation: mapping.pages.generation,
            });
            let (retired_view, retired_import) = mapping.materialization.retire();
            (
                RegisteredSurfacePlanAdoption {
                    changed,
                    replaced,
                    lifecycle_generation: mapping.lifecycle.generation,
                },
                retired_view,
                retired_import,
            )
        };
        self.host_materializations
            .retire_materialization(retired_view, retired_import);
        Some(effect)
    }

    /// Adopt a freshly walked physical backing for the same logical resource.
    ///
    /// Resource synchronization addresses the resource and resolves its current
    /// backing. Accordingly this replaces only the page-list incarnation: GPU
    /// residents and deferred content remain keyed by `map_generation`, while
    /// every host object bound to the old physical pages is retired here.
    pub fn refresh_mapping_pages(&mut self, mapping_id: u32, entries: Vec<u32>) -> bool {
        let Some(e) = self.surfaces.mappings.get_mut(&mapping_id) else {
            return false;
        };
        if entries.is_empty() || e.pages.entries == entries {
            return false;
        }
        e.pages.entries = entries;
        Self::bump_page_generation(e);
        let (retired, retired_import) = Self::take_mapping_view(e);
        self.host_materializations
            .retire_materialization(retired, retired_import);
        self.retire_mapping_gather_witness(mapping_id);
        true
    }

    /// Refresh pages derived from an existing registered-surface walk and
    /// relatch that derivation at the new page generation.
    pub(crate) fn refresh_surface_walk_pages(
        &mut self,
        surface: SurfaceId,
        entries: Vec<u32>,
        walk: SurfaceBackingWalk,
    ) -> Option<SurfacePageRefresh> {
        if !self.refresh_mapping_pages(surface.get(), entries) {
            return None;
        }
        let mapping = self.surfaces.mappings.get_mut(&surface.get())?;
        mapping.pages.surface_walk = Some(SurfaceBackingWalk {
            page_generation: mapping.pages.generation,
            ..walk
        });
        Some(SurfacePageRefresh {
            page_count: mapping.pages.entries.len(),
            page_generation: mapping.pages.generation,
        })
    }

    pub(crate) fn note_surface_materialization_refused(
        &mut self,
        surface: SurfaceId,
    ) -> Option<u32> {
        let mapping = self.surfaces.mappings.get_mut(&surface.get())?;
        let generation = mapping.pages.generation;
        mapping.materialization.note_refused(generation);
        Some(generation)
    }

    pub(crate) fn install_surface_materialization(
        &mut self,
        surface: SurfaceId,
        view: SurfaceHostView,
    ) -> bool {
        let Some(mapping) = self.surfaces.mappings.get_mut(&surface.get()) else {
            return false;
        };
        mapping.materialization.install(view);
        true
    }

    pub(crate) fn install_surface_import(
        &mut self,
        surface: SurfaceId,
        import: std::sync::Arc<reims_vgpu_memory::GuestRamImport>,
    ) -> bool {
        let Some(mapping) = self.surfaces.mappings.get_mut(&surface.get()) else {
            return false;
        };
        if let Some(retired) = mapping.materialization.replace_import(import) {
            self.host_materializations.retire_guest_import(retired);
        }
        true
    }

    /// Forget an unresolved physical backing without ending the resource.
    ///
    /// A failed current-backing walk proves that the cached pages are unsafe;
    /// it does not prove that the resource object was destroyed. Keep logical
    /// content keyed by `map_generation`, but make every page-bound access
    /// re-resolve before it can proceed.
    pub fn forget_mapping_page_backing(&mut self, mapping_id: u32) -> bool {
        let Some(e) = self.surfaces.mappings.get_mut(&mapping_id) else {
            return false;
        };
        let had = !e.pages.entries.is_empty() || e.materialization.has_view();
        e.pages.entries.clear();
        e.pages.table_kva = 0;
        e.pages.surface_walk = None;
        Self::bump_page_generation(e);
        let (retired, retired_import) = Self::take_mapping_view(e);
        self.host_materializations
            .retire_materialization(retired, retired_import);
        self.retire_mapping_gather_witness(mapping_id);
        had
    }

    /// Drop compute storage-residency mirror entries whose byte window
    /// `[surface_offset, span_end)` intersects a guest write of
    /// `[lo, hi)` on this mapping. The mirror claims "guest pages still hold
    /// exactly the resident's content for this window" — any intersecting
    /// write breaks that claim; disjoint windows (ping-pong canvases) survive.
    pub fn invalidate_storage_residency_window(&mut self, mapping_id: u32, lo: u64, hi: u64) {
        self.content
            .compute_residency
            .invalidate_surface_window(mapping_id, lo, hi);
    }

    /// Drop cached page list + contig view without unmapping the slot.
    ///
    /// Used on ReplacePhysical / rebind: guest may have recycled PFNs into the
    /// zone freelist; the next Store must re-resolve before any host write or
    /// import-present DMA (freelist `0xff000000ff000000` class).
    pub fn invalidate_mapping_pages(&mut self, mapping_id: u32) -> MappingInvalidationEffect {
        // The cached BGRA frame is a host-side copy of the pages this call is
        // invalidating, and it is the only such copy whose key does not carry
        // `map_generation`: the resident's does (`surface_identity`), the
        // contiguous view and the guest-write token are retired below, and every
        // armed window refuses on the generation check it already has. So the
        // bump that disqualifies all of those leaves this entry addressable by
        // `(mapping_id, geometry)` alone, still holding pixels read through the
        // page list that just stopped being this surface's.
        //
        // Retiring the guest-write token is what makes that reachable rather
        // than theoretical. The surface backing sampled ladder's host-cache rung serves
        // its copy unless the witness reports `Wrote`, and a retired token
        // reports `NoStamp` — deliberately not evidence, because "nobody armed
        // this" is a statement about this device and not about the guest. The
        // rung therefore reads the invalidation as *permission* to serve, and
        // keeps serving until some later Store replaces the bytes. A surface
        // composited once and then only sampled — a popup backdrop, a settings
        // pane — never gets that Store, so the stale frame is held for the life
        // of the guest.
        //
        let dropped_host_cache = self.host_replicas.forget_surface(mapping_id);
        let Some(e) = self.surfaces.mappings.get_mut(&mapping_id) else {
            return MappingInvalidationEffect {
                had_page_state: false,
                dropped_host_cache,
            };
        };
        let had = !e.pages.entries.is_empty() || e.materialization.has_view();
        e.pages.entries.clear();
        e.pages.table_kva = 0;
        Self::bump_map_generation(e);
        Self::bump_page_generation(e);
        let (retired, retired_import) = Self::take_mapping_view(e);
        self.host_materializations
            .retire_materialization(retired, retired_import);
        self.retire_mapping_gather_witness(mapping_id);
        MappingInvalidationEffect {
            had_page_state: had,
            dropped_host_cache,
        }
    }

    pub fn map_surface(&mut self, mapping_id: u32) -> bool {
        self.try_map_surface(mapping_id).is_ok()
    }

    /// Ensure a resolver has a registry slot without asserting a new mapping
    /// lifetime. Backing adoption and replacement remain separate transitions.
    pub(crate) fn ensure_surface_slot(
        &mut self,
        mapping_id: u32,
    ) -> Result<(), StateMutationDecline> {
        self.observations.observe_mapping_id(mapping_id);
        if !crate::model::is_surface_mapping_id(mapping_id) {
            return Err(StateMutationDecline::MapSurfaceIdSentinel { mapping_id });
        }
        self.surfaces
            .mappings
            .entry(mapping_id)
            .or_default()
            .lifecycle
            .active = true;
        Ok(())
    }

    pub(crate) fn try_map_surface(&mut self, mapping_id: u32) -> Result<(), StateMutationDecline> {
        self.observations.observe_mapping_id(mapping_id);
        if !crate::model::is_surface_mapping_id(mapping_id) {
            return Err(StateMutationDecline::MapSurfaceIdSentinel { mapping_id });
        }
        let e = self.surfaces.mappings.entry(mapping_id).or_default();
        e.lifecycle.active = true;
        e.pages.entries.clear();
        Self::bump_map_generation(e);
        Self::bump_page_generation(e);
        e.pages.table_kva = 0;
        e.declaration.clear();
        e.content.guest_page_generation = 0;
        e.content.surface_epoch = 0;
        let (retired, retired_import) = Self::take_mapping_view(e);
        self.host_materializations
            .retire_materialization(retired, retired_import);
        self.retire_mapping_gather_witness(mapping_id);
        self.host_replicas.forget_surface(mapping_id);
        self.forget_compositor_mapping(mapping_id);
        Ok(())
    }

    /// Publish the explicit mapper-service lookup edge for one mapped surface.
    pub fn map_mapper_surface(
        &mut self,
        mapper_surface: MapperSurfaceRef,
        surface: MapperResolvedSurfaceId,
    ) -> bool {
        self.try_map_mapper_surface(mapper_surface, surface)
            .unwrap_or(false)
    }

    pub(crate) fn try_map_mapper_surface(
        &mut self,
        mapper_surface: MapperSurfaceRef,
        surface: MapperResolvedSurfaceId,
    ) -> Result<bool, StateMutationDecline> {
        let mapping_id = surface.get();
        if mapper_surface.get() == 0 {
            return Ok(false);
        }
        self.try_map_surface(mapping_id)?;
        Ok(self.surfaces.mapper.map_surface(mapper_surface, surface))
    }

    /// Publish the directed mapper capture taken at an IOSFC producer write.
    pub fn publish_mapper_capture(&mut self, capture: MapperCapture) {
        self.surfaces.mapper.publish_capture(capture);
    }

    /// Consume the capture for exactly this published ring entry.
    ///
    /// A capture for another producer remains pending. A caller that consumes a
    /// matching producer and then finds the request kind differs must restore
    /// it through [`Self::restore_mapper_capture`].
    pub fn take_mapper_capture(&mut self, producer: u32) -> Option<MapperCapture> {
        self.surfaces.mapper.take_capture(producer)
    }

    /// Restore a capture consumed speculatively for a mismatched request kind.
    pub fn restore_mapper_capture(&mut self, capture: MapperCapture) {
        self.surfaces.mapper.restore_capture(capture);
    }

    /// Retain the mapper device identity learned from a directed capture.
    ///
    /// Zero means the capture supplied no device identity and cannot erase the
    /// previously established one.
    pub fn observe_mapper_device(&mut self, device_kva: u64) {
        self.surfaces.mapper.observe_device(device_kva);
    }

    /// The mapper device identity used to resolve mapper-internal fields.
    pub fn mapper_device_kva(&self) -> u64 {
        self.surfaces.mapper.device_kva()
    }

    /// Resolve a mapper view through the edge installed by the mapper service.
    pub fn resolve_mapper_surface(
        &self,
        mapper_surface: MapperSurfaceRef,
    ) -> Option<MapperResolvedSurfaceId> {
        self.surfaces
            .mapper
            .resolve_surface(mapper_surface)
            .filter(|surface| self.surfaces.mappings.contains_key(&surface.get()))
    }

    pub fn unmap_surface(&mut self, mapping_id: u32) -> bool {
        self.try_unmap_surface(mapping_id).unwrap_or(false)
    }

    pub(crate) fn try_unmap_surface(
        &mut self,
        mapping_id: u32,
    ) -> Result<bool, StateMutationDecline> {
        self.observations.observe_mapping_id(mapping_id);
        if !crate::model::is_surface_mapping_id(mapping_id) {
            return Err(StateMutationDecline::UnmapSurfaceIdSentinel { mapping_id });
        }
        self.surfaces
            .mapper
            .retire_surface(MapperResolvedSurfaceId::new(mapping_id));
        self.forget_compositor_mapping(mapping_id);
        if let Some(e) = self.surfaces.mappings.get_mut(&mapping_id) {
            e.lifecycle.active = false;
            e.pages.entries.clear();
            e.pages.table_kva = 0;
            e.lifecycle.internal_kva = 0;
            e.declaration.clear();
            Self::bump_map_generation(e);
            Self::bump_page_generation(e);
            let (retired, retired_import) = Self::take_mapping_view(e);
            self.host_materializations
                .retire_materialization(retired, retired_import);
            self.retire_mapping_gather_witness(mapping_id);
            self.host_replicas.forget_surface(mapping_id);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Attach directed MappingInternal capture to a mapped slot.
    pub fn attach_mapping_internal(&mut self, mapping_id: u32, mapping_internal: u64) -> bool {
        self.try_attach_mapping_internal(mapping_id, mapping_internal)
            .is_ok()
    }

    pub(crate) fn try_attach_mapping_internal(
        &mut self,
        mapping_id: u32,
        mapping_internal: u64,
    ) -> Result<(), StateMutationDecline> {
        self.observations.observe_mapping_id(mapping_id);
        if !crate::model::is_surface_mapping_id(mapping_id) {
            return Err(StateMutationDecline::AttachMappingIdSentinel { mapping_id });
        }
        if mapping_internal == 0 {
            return Err(StateMutationDecline::AttachMappingInternalZero { mapping_id });
        }
        let e = self.surfaces.mappings.entry(mapping_id).or_default();
        // A re-statement of the SAME MappingInternal (notify trailing our
        // eager resolve) is not a new surface: keep bindings, generation,
        // resident, and deferred windows untouched.
        if e.lifecycle.internal_kva == mapping_internal {
            e.lifecycle.active = true;
            return Ok(());
        }
        e.lifecycle.active = true;
        e.lifecycle.internal_kva = mapping_internal;
        e.pages.entries.clear();
        e.pages.table_kva = 0;
        e.declaration.clear();
        e.content.guest_page_generation = 0;
        e.content.surface_epoch = 0;
        Self::bump_map_generation(e);
        Self::bump_page_generation(e);
        // New MappingInternal ⇒ new surface; force device-desc re-resolve.
        let (retired, retired_import) = Self::take_mapping_view(e);
        self.host_materializations
            .retire_materialization(retired, retired_import);
        self.retire_mapping_gather_witness(mapping_id);
        // New MappingInternal ⇒ new surface, and the `bump_map_generation`
        // above is what retires the stale present evidence: it is stamped with
        // the incarnation that recorded it, so the recycled slot cannot inherit
        // a display-plane qualification it did not earn.
        Ok(())
    }

    /// Cache the 0x200-byte guest device descriptor for plane/surface sample windows.
    pub fn set_mapping_device_desc(&mut self, mapping_id: u32, desc: &[u8]) -> bool {
        self.try_set_mapping_device_desc(mapping_id, desc).is_ok()
    }

    pub(crate) fn try_set_mapping_device_desc(
        &mut self,
        mapping_id: u32,
        desc: &[u8],
    ) -> Result<(), StateMutationDecline> {
        self.observations.observe_mapping_id(mapping_id);
        if !crate::model::is_surface_mapping_id(mapping_id) {
            return Err(StateMutationDecline::MappingDeviceDescIdSentinel { mapping_id });
        }
        if desc.is_empty() {
            return Err(StateMutationDecline::MappingDeviceDescEmpty { mapping_id });
        }
        let e = self.surfaces.mappings.entry(mapping_id).or_default();
        e.declaration.publish_device_desc(desc);
        Ok(())
    }

    pub fn set_mapping_geom(
        &mut self,
        mapping_id: u32,
        width: u32,
        height: u32,
        format: u16,
    ) -> bool {
        self.try_set_mapping_geom(mapping_id, width, height, format)
            .is_ok()
    }

    pub(crate) fn try_set_mapping_geom(
        &mut self,
        mapping_id: u32,
        width: u32,
        height: u32,
        format: u16,
    ) -> Result<(), StateMutationDecline> {
        if !crate::model::is_surface_mapping_id(mapping_id) {
            return Err(StateMutationDecline::MappingGeomIdSentinel { mapping_id });
        }
        // The bound itself lives once, in `regs::scanout_extent_fault`; this is
        // the only caller that has to name which half of it broke, so it is the
        // only one that reads the fault rather than the verdict.
        if let Some(fault) = crate::model::scanout_extent_fault(width, height) {
            use crate::model::ScanoutExtentFault as F;
            return Err(match fault {
                F::WidthZero => StateMutationDecline::MappingGeomWidthZero { mapping_id },
                F::HeightZero => StateMutationDecline::MappingGeomHeightZero { mapping_id },
                F::WidthAboveBound => {
                    StateMutationDecline::MappingGeomWidthRange { mapping_id, width }
                }
                F::HeightAboveBound => {
                    StateMutationDecline::MappingGeomHeightRange { mapping_id, height }
                }
            });
        }
        let e = self.surfaces.mappings.entry(mapping_id).or_default();
        // A changed declaration (mode switch / rematerialize) is a new surface
        // identity: reset `content_generation` and `surface_content_epoch`. The
        // guest pages stay authoritative, so the cost of resetting when nothing
        // really changed is one seed copy.
        //
        // **All three fields, not just the extent.** The epoch's claim is that a
        // resident's pixels *are* this mapping's content, and it is what
        // licenses the attachment LOAD elision — so it has to be withdrawn
        // whenever the guest re-declares what those bytes mean. Extent alone
        // read as sufficient because a format change usually moves the
        // `TargetIdentity` too and picks up a different resident by itself. It
        // does not always: `present_identity::surface_format` maps several guest
        // declarations onto one semantic layout and falls back to the scanout order
        // for any it cannot express, so a mapping going from a format with a
        // linear texel to a compressed or planar one keeps its identity, keeps
        // its resident, and keeps an epoch that was stamped against the old
        // interpretation of the same bytes.
        let geometry = SurfaceGeometry {
            width,
            height,
            format,
        };
        if e.geometry() != Some(geometry) {
            e.content.guest_page_generation = 0;
            e.content.surface_epoch = 0;
        }
        e.declaration.publish_geometry(geometry);
        Ok(())
    }

    /// Record that this device is about to write pixel bytes into guest RAM.
    ///
    /// Called from every host-side writer, including the ones that reach guest
    /// pages through a raw task-GVA walk and never name a mapping. The
    /// a retained derived image must distinguish "unchanged" from "another
    /// device path wrote these pages".
    ///
    /// Deliberately called before the write rather than after it succeeds: a
    /// refused write costs a spurious bump, which makes a reader re-read bytes
    /// that did not change. The opposite error hands out a stale copy.
    pub fn note_host_wrote_guest_ram(&mut self) {
        self.content.host_writes.note_unknown();
    }

    /// The same, for a writer that walked the guest page tables and so knows
    /// exactly which pages it landed in even though it names no mapping.
    pub fn note_host_wrote_pages(&mut self, pages: Vec<u64>) {
        self.content.host_writes.note_pages(pages);
    }

    /// Every guest page a mapping covers, or `None` when the set cannot be
    /// named exactly.
    ///
    /// This is the page set the guest-write **reach** test is decided on, from
    /// both ends: [`Self::note_host_wrote_mapping`] names a writeback's
    /// destination with it, and the readers that ask
    /// `render_writeback::settle_guest_writes_unless_disjoint` whether they may
    /// skip the wait name their source with it. Those two answers are compared
    /// against each other, so they must come from one rule — a writer that
    /// named pages by a slightly different rule than the reader would make a
    /// genuine overlap read as disjoint, and skipping *that* wait is a stale
    /// frame. Hence one function rather than the three hand-written copies this
    /// replaced.
    ///
    /// All-or-nothing on purpose: `collect` into an `Option` so a single
    /// unresolvable entry makes the whole set unnamed rather than partially
    /// named. A short list is the one wrong answer that costs a frame, because
    /// it licenses skipping a wait for a page it silently omitted. `None` always
    /// settles.
    ///
    /// Cheap by construction — no revalidation, no host round trip, no
    /// `map_pages`. `page_entries` already *is* the list. That matters because
    /// the settle closure runs on the hot path whenever a writeback is
    /// outstanding. The revalidating cousin is
    /// [`crate::runtime::mapper::mapping_page_gpas`], which needs a `&mut host`
    /// and is for callers about to *map* the pages, not merely name them.
    pub fn mapping_reach_pages(&self, mapping_id: u32) -> Option<Vec<u64>> {
        let m = self.surfaces.mappings.get(&mapping_id)?;
        if m.pages.entries.is_empty() {
            return None;
        }
        let shift = self.page_shift;
        m.pages
            .entries
            .iter()
            .map(|&e| reims_vgpu_paging::geometry::mapper_entry_gpa(e, shift))
            .collect()
    }

    /// The same, for a writer that knows which mapping's pages it is landing in.
    pub fn note_host_wrote_mapping(&mut self, mapping_id: u32) {
        let Some(entries) = self
            .surfaces
            .mappings
            .get(&mapping_id)
            .map(|mapping| mapping.pages.entries.as_slice())
            .filter(|entries| !entries.is_empty())
        else {
            self.content.host_writes.note_unknown();
            return;
        };
        let shift = self.page_shift;
        if entries
            .iter()
            .any(|&entry| reims_vgpu_paging::geometry::mapper_entry_gpa(entry, shift).is_none())
        {
            // A mapping whose pages cannot be named exactly cannot have its
            // write ruled out later. Record one unnamed write rather than the
            // resolvable prefix.
            self.content.host_writes.note_unknown();
            return;
        }
        self.content
            .host_writes
            .note_page_iter(entries.iter().map(|&entry| {
                reims_vgpu_paging::geometry::mapper_entry_gpa(entry, shift)
                    .expect("page entries were validated above")
            }));
    }

    /// Issue a sampled-content generation that has never been issued before.
    ///
    /// Every producer of a sampled-content identity must take its generation
    /// from here and nowhere else. The value is what the engine's sampled
    /// cache binds on without looking at a single byte, so "never issued
    /// before" is the whole of the contract — see
    /// [`SampledContentState`]. Never returns 0, which readers use for
    /// "no host content yet".
    pub fn next_sampled_content_generation(&mut self) -> u64 {
        self.content.sampled.issue_generation()
    }

    /// Issue the next recency stamp for [`HostSurface::last_touch`].
    ///
    /// Strictly increasing, so the smallest stamp in
    /// [`HostReplicaState::gva_surfaces`] is always the coldest entry and the byte cap
    /// needs no other ordering. Saturating rather than wrapping: a wrap would
    /// make one ancient entry look like the newest and pin it forever, and at
    /// one stamp per lookup `u64::MAX` is not reachable by any real session.
    /// Bump content generation after a write into the mapping (0 never skips).
    ///
    /// Also advances the mapping's surface epoch, so every one of
    /// this crate's guest-page writers keeps that epoch closed for free — the
    /// completeness property the IOSurface texture `LoadFromTarget` gate rests on.
    pub fn mark_mapping_written(&mut self, mapping_id: u32) -> u32 {
        self.surfaces
            .mappings
            .mark_written(SurfaceId::new(mapping_id))
    }

    /// Apply one decoded validity statement to a surface mapping.
    ///
    /// Guest-write currency and the validity quad are one transition here, so
    /// runtime callers cannot update one while forgetting the other. Returns
    /// `false` only when the surface identity is not registered.
    pub(crate) fn apply_surface_validity(
        &mut self,
        surface: SurfaceId,
        ops: reims_vgpu_protocol::ResourceValidityOps,
    ) -> bool {
        self.surfaces.mappings.apply_validity(surface, ops)
    }

    /// Remember a task-local owner only as a surface lookup-order hint.
    ///
    /// This does not create a task-resource or storage ownership edge; those
    /// belong to the canonical resource graph.
    pub(crate) fn note_surface_owner_hint(&mut self, surface: SurfaceId, task: TaskId) {
        if let Some(mapping) = self.surfaces.mappings.get_mut(&surface.get()) {
            mapping.owner_task_hint = task.get();
        }
    }

    /// Advance a mapping's content stamps for a publish that changed its pixels
    /// *without* writing its guest pages — the lazy IOSurface texture Store of
    /// [`crate::runtime::writeback_debt`], which leaves the frame in the engine
    /// resident and owes the pages a copy.
    ///
    /// Returns the new mapping surface epoch so the caller can
    /// stamp the resident that holds those pixels in the same breath; the two
    /// must not be separable, or the stamp records a currency that already moved.
    ///
    /// # Why it moves two of [`Self::mark_mapping_written`]'s three stamps
    ///
    /// It is the same statement as that one — "this mapping's pixels are now
    /// different" — differing only in where the pixels are, and the difference is
    /// exactly `content_generation`. That field means *the guest's pages hold
    /// something new*, and its consumers re-read those pages when it moves; a
    /// lazy Store wrote no page, so moving it would send the compute rail to
    /// re-seed bytes that did not change.
    ///
    /// The other two mean *the pixels are new*, wherever they are, and both move:
    /// `surface_content_epoch` licenses the attachment LOAD elision, which is what
    /// keeps a lazy Store from being read straight back off guest pages, and
    /// `host_published_seq` orders this frame against the guest's own later
    /// `clear_host_valid`, which is what
    /// [`crate::runtime::resource_validity::licence_of`] answers at the payment.
    ///
    /// Anything else that has to notice a lazy Store belongs on the epoch and not
    /// on the generation. The host window's publish key is the worked example:
    /// it keyed on the generation, so a driven macos-13 boot with the lazy rail on
    /// published 60 fresh frames a second against 314 `same_key` where the eager
    /// arm published 81 against 131 — real frames discarded as unchanged. It now
    /// carries `PresentState::frame_content_epoch` beside the generation.
    pub fn note_surface_content_published(&mut self, mapping_id: u32) -> u32 {
        self.surfaces
            .mappings
            .note_content_published(SurfaceId::new(mapping_id))
    }
}

#[cfg(test)]
mod device_desc_tests {
    use super::*;
    use reims_vgpu_protocol::{
        device_desc_plane, DEVICE_DESC_LEN, DEVICE_DESC_PLANES, DEVICE_PLANE_DESC_LEN,
    };

    fn entry_with_desc(len: usize) -> SurfaceMappingEntry {
        let mut entry = SurfaceMappingEntry::default();
        entry.publish_device_desc_for_test(&vec![0u8; len]);
        entry
    }

    /// The completeness rule is all-or-nothing, and what it hands back is the
    /// record's own length rather than whatever was cached.
    #[test]
    fn a_partial_device_descriptor_is_no_descriptor() {
        assert!(entry_with_desc(0).device_desc_complete().is_none());
        assert!(
            entry_with_desc(DEVICE_DESC_LEN - 1)
                .device_desc_complete()
                .is_none(),
            "one byte short is not a record"
        );
        assert_eq!(
            entry_with_desc(DEVICE_DESC_LEN)
                .device_desc_complete()
                .map(<[u8]>::len),
            Some(DEVICE_DESC_LEN)
        );
        assert_eq!(
            entry_with_desc(DEVICE_DESC_LEN * 2)
                .device_desc_complete()
                .map(<[u8]>::len),
            Some(DEVICE_DESC_LEN),
            "a longer cached blob is still truncated to the record"
        );
    }

    /// Why the truncation is the answer and not an arbitrary choice between two
    /// spellings that happen to agree.
    ///
    /// `device_desc_plane` bounds each plane read against the slice it is
    /// handed, and the eighth plane's record ends past `DEVICE_DESC_LEN`. So a
    /// caller that passed `device_desc.as_slice()` whole would decode an eighth
    /// plane out of an over-long cached blob while a caller that truncated
    /// refused it — two readers of one mapping disagreeing about how many planes
    /// the surface has. Truncating everywhere removes the disagreement, and this
    /// pins that the boundary the two spellings differ at is real.
    #[test]
    fn the_eighth_plane_lies_past_the_record_the_completeness_rule_hands_back() {
        let eighth = DEVICE_DESC_PLANES + 7 * DEVICE_PLANE_DESC_LEN;
        assert!(
            eighth + DEVICE_PLANE_DESC_LEN > DEVICE_DESC_LEN,
            "the plane table must actually overrun the record, or there is \
             nothing for the two spellings to disagree about"
        );

        // A descriptor declaring eight planes, cached over-long.
        let mut over = vec![0u8; eighth + DEVICE_PLANE_DESC_LEN];
        over[reims_vgpu_protocol::DEVICE_DESC_PLANE_COUNT] = 8;
        assert!(
            device_desc_plane(&over, 7).is_some(),
            "the whole-slice spelling would have found an eighth plane"
        );

        let mut e = SurfaceMappingEntry::default();
        e.publish_device_desc_for_test(&over);
        let truncated = e.device_desc_complete().expect("a full record is cached");
        assert!(
            device_desc_plane(truncated, 7).is_none(),
            "and the rule this crate now uses everywhere refuses it"
        );
        assert!(
            device_desc_plane(truncated, 6).is_some(),
            "without refusing the seventh, which does fit"
        );
    }
}

#[cfg(test)]
mod fail_vocabulary_tests {
    use super::*;
    use crate::observe::Decline;

    /// Every `FailEvent` names a *specific* check. Written as one assertion per
    /// variant rather than a loop so the expected slug is visible next to the
    /// value that produces it — this table is the thing a reader checks against
    /// `/tmp/reims-vgpu-fail.log`.
    #[test]
    fn every_fail_event_variant_names_its_own_check() {
        assert_eq!(
            FailEvent::UnknownRootOpcode {
                opcode: 0x20,
                total_size: 16
            }
            .slug(),
            "unknown_root_opcode"
        );
        assert_eq!(
            FailEvent::UnknownChildOpcode {
                channel: 5,
                opcode: 6,
                total_size: 32,
                stamp_count: 0,
                payload: Vec::new()
            }
            .slug(),
            "unknown_child_opcode"
        );
        assert_eq!(
            FailEvent::BadMmioAccess {
                offset: 0x1000,
                size: 2
            }
            .slug(),
            "bad_mmio_access"
        );
        // The malformed variants forward to the fault, so two different checks
        // on the same variant must not share a slug — that collapse is the
        // defect the vocabulary exists to prevent.
        let desync = FailEvent::MalformedRootPacket {
            fault: PacketFault::DesyncedHeadTail,
            head: 0,
        };
        let header = FailEvent::MalformedRootPacket {
            fault: PacketFault::RootHeaderRead,
            head: 0,
        };
        assert_eq!(desync.slug(), "packet_desynced_head_tail");
        assert_eq!(header.slug(), "packet_root_header_read");
        assert_ne!(desync.slug(), header.slug());
        assert_eq!(
            FailEvent::UnsupportedExec {
                channel: 3,
                fault: ExecFault::Indirect2Short
            }
            .slug(),
            "exec_indirect2_short"
        );
    }

    /// A slug without the value that caused it is half a diagnostic. The fields
    /// carry the load-bearing numbers, and the root/child distinction shows up
    /// as the presence of `ch=`.
    #[test]
    fn fail_event_fields_carry_the_load_bearing_values() {
        let line = crate::observe::Emit::decline(
            "fail_event",
            &FailEvent::UnknownChildOpcode {
                channel: 5,
                opcode: 6,
                total_size: 32,
                stamp_count: 1,
                payload: vec![0x21, 0x43, 0x65, 0x87, 0x01, 0x00, 0x00, 0x00],
            },
        )
        .render();
        assert_eq!(
            line,
            "fail_event reason=unknown_child_opcode ch=5 opcode=0x6 total_size=32 stamps=1 \
             plen=8 payload=0x87654321:0x00000001"
        );

        let root = crate::observe::Emit::decline(
            "fail_event",
            &FailEvent::MalformedRootPacket {
                fault: PacketFault::BadSize,
                head: 4096,
            },
        )
        .render();
        assert_eq!(root, "fail_event reason=packet_bad_size head=4096");

        let child = crate::observe::Emit::decline(
            "fail_event",
            &FailEvent::MalformedChildPacket {
                channel: 2,
                fault: PacketFault::BadSize,
                head: 4096,
            },
        )
        .render();
        assert_eq!(child, "fail_event reason=packet_bad_size ch=2 head=4096");
    }

    /// An unknown child opcode is acknowledged to the guest — its stamps retire
    /// like any other packet's — so this record is the only evidence the command
    /// was ever issued. It therefore has to say enough to identify it.
    ///
    /// `total_size` cannot: it spans the header, the stamps and the payload at
    /// once. A driven arm64 boot reports 968 packets at `opcode=0x3f` and 83 at
    /// `0x3e`, all `total_size=24`, and against a 12-byte header and 8-byte
    /// stamps that is either one stamp and one payload word or no stamps and
    /// three — different commands with the same size. The two readings must not
    /// render alike.
    #[test]
    fn an_unknown_child_opcode_separates_its_stamps_from_its_payload() {
        let render = |stamp_count, payload: Vec<u8>| {
            crate::observe::Emit::decline(
                "fail_event",
                &FailEvent::UnknownChildOpcode {
                    channel: 3,
                    opcode: 0x3f,
                    total_size: 24,
                    stamp_count,
                    payload,
                },
            )
            .render()
        };
        let one_stamp = render(1, vec![0x0c, 0x00, 0x00, 0x00]);
        let no_stamps = render(0, vec![0; 12]);
        assert_ne!(
            one_stamp, no_stamps,
            "two packets of one total_size must not render alike"
        );
        assert!(one_stamp.contains("stamps=1 plen=4 payload=0x0000000c"));
        assert!(no_stamps.contains("stamps=0 plen=12"));

        // A payload longer than the echo is reported by `plen`, so a truncated
        // echo can be told from a complete one rather than read as the whole
        // command.
        let long = render(0, (0..40).collect());
        assert!(long.contains("plen=40"), "{long}");
        assert_eq!(
            long.matches("0x").count(),
            crate::observe::model::UNKNOWN_OPCODE_ECHO_WORDS_MAX + 1,
            "the echo is bounded, and the opcode is the one other hex field: {long}"
        );

        // A sub-word tail is never zero-padded into a word the guest did not
        // write; `plen` is what reports it.
        let ragged = render(0, vec![0xff, 0xff, 0xff, 0xff, 0xaa]);
        assert!(
            ragged.contains("plen=5 payload=0xffffffff") && !ragged.contains("0x000000aa"),
            "{ragged}"
        );

        // Nothing to echo must not emit an empty field.
        assert!(!render(2, Vec::new()).contains("payload="));
    }

    /// The malformed-packet checks used to be hyphenated string literals passed
    /// by hand. They are now variants, and no two may answer with the same slug
    /// — otherwise a child tail read and a child head writeback look identical
    /// in the log.
    #[test]
    fn the_packet_faults_all_differ() {
        const ALL: &[PacketFault] = &[
            PacketFault::DesyncedHeadTail,
            PacketFault::BadSize,
            PacketFault::RootHeaderRead,
            PacketFault::RootSnapRead,
            PacketFault::RootStampWriteback,
            PacketFault::ChildHeaderRead,
            PacketFault::ChildRegsBaseRead,
            PacketFault::ChildRegsHeadRead,
            PacketFault::ChildRegsStampRead,
            PacketFault::ChildSnapRead,
            PacketFault::ChildTailRead,
            PacketFault::ChildHeadWriteback,
            PacketFault::ShortSnapshot,
        ];
        let mut slugs: Vec<&str> = ALL.iter().map(|f| f.slug()).collect();
        slugs.sort_unstable();
        let count = slugs.len();
        slugs.dedup();
        assert_eq!(slugs.len(), count, "two packet faults share a slug");
    }

    #[test]
    fn every_state_mutation_check_has_its_own_registered_reason() {
        let declines = [
            StateMutationDecline::SetObjectListTaskInactive { task_id: 1 },
            StateMutationDecline::InsertObjectTaskInactive {
                task_id: 1,
                object_ref: 3,
            },
            StateMutationDecline::MapSurfaceIdSentinel { mapping_id: 8192 },
            StateMutationDecline::UnmapSurfaceIdSentinel { mapping_id: 8192 },
            StateMutationDecline::AttachMappingIdSentinel { mapping_id: 8192 },
            StateMutationDecline::AttachMappingInternalZero { mapping_id: 1 },
            StateMutationDecline::MappingDeviceDescIdSentinel { mapping_id: 8192 },
            StateMutationDecline::MappingDeviceDescEmpty { mapping_id: 1 },
            StateMutationDecline::MappingGeomIdSentinel { mapping_id: 8192 },
            StateMutationDecline::MappingGeomWidthZero { mapping_id: 1 },
            StateMutationDecline::MappingGeomHeightZero { mapping_id: 1 },
            StateMutationDecline::MappingGeomWidthRange {
                mapping_id: 1,
                width: crate::model::MAX_SCANOUT_DIM + 1,
            },
            StateMutationDecline::MappingGeomHeightRange {
                mapping_id: 1,
                height: crate::model::MAX_SCANOUT_DIM + 1,
            },
        ];
        let mut slugs = std::collections::HashSet::new();
        for decline in declines {
            assert!(slugs.insert(decline.slug()), "duplicate {}", decline.slug());
        }
        assert_eq!(
            slugs.len(),
            13,
            "every state mutation check has its own slug"
        );
        assert_eq!(
            crate::observe::Emit::decline(
                "model_state_mutation",
                &StateMutationDecline::MappingGeomWidthRange {
                    mapping_id: 7,
                    width: 65_535,
                },
            )
            .render(),
            "model_state_mutation reason=model_mapping_geom_width_range \
             mapping=7 width=65535"
        );
    }

    /// A refused geometry must leave no entry behind — and a refusal is only ever
    /// about the sentinel id or the extent, never about how large the id is.
    #[test]
    fn invalid_mapping_geometry_cannot_create_an_entry() {
        let mut state = DeviceState::new(DeviceId(1), crate::model::PAGE_SHIFT_X86);
        assert!(!state.set_mapping_geom(0, 64, 64, 0x50));
        assert!(!state.surfaces.mappings.contains_key(&0));
        assert!(!state.set_mapping_geom(1, 0, 64, 0x50));
        assert!(!state.set_mapping_geom(1, 64, 0, 0x50));
        assert!(!state.surfaces.mappings.contains_key(&1));
    }

    /// The reach set is every page or no pages, never a short list.
    ///
    /// This is the one property the disjoint-settle skip rests on. Both ends of
    /// that comparison — the writeback naming its destination, and a reader
    /// asking whether it may skip the wait — come from
    /// [`DeviceState::mapping_reach_pages`], so a set that silently dropped an
    /// unresolvable entry would let a reader skip a settle for a page the
    /// writeback is about to land in. That is a stale frame with no error
    /// anywhere, which is why the failure direction is asserted and not just the
    /// success one.
    #[test]
    fn a_mapping_reach_set_is_every_page_or_none() {
        use reims_vgpu_paging::geometry::{
            MAPPER_PAGE_ENTRY_PFN_SHIFT as PAGE_ENTRY_PFN_SHIFT,
            MAPPER_PAGE_ENTRY_VALID as PAGE_ENTRY_VALID,
        };
        let shift = crate::model::PAGE_SHIFT_X86;
        let mut state = DeviceState::new(DeviceId(1), shift);
        assert!(state.set_mapping_geom(3, 64, 64, 0x50));

        assert_eq!(
            state.mapping_reach_pages(3),
            None,
            "a mapping with no page list can rule nothing out"
        );
        assert_eq!(
            state.mapping_reach_pages(99),
            None,
            "a mapping that does not exist can rule nothing out"
        );

        let valid = |pfn: u32| (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        state.surfaces.mappings.get_mut(&3).unwrap().pages.entries =
            vec![valid(4), valid(5), valid(6)];
        assert_eq!(
            state.mapping_reach_pages(3),
            Some(vec![4u64 << shift, 5u64 << shift, 6u64 << shift]),
            "every entry resolves, so the whole set is named"
        );

        // The middle entry carries no VALID bit, so it names no backing.
        state.surfaces.mappings.get_mut(&3).unwrap().pages.entries = vec![valid(4), 0, valid(6)];
        assert_eq!(
            state.mapping_reach_pages(3),
            None,
            "one unresolvable entry must unname the set, not shorten it"
        );
    }

    /// Every one of the three entry points must reach the record, whatever it can
    /// say about where it wrote.
    ///
    /// The record's own tests cover what each shape then *answers*; this covers
    /// that a writer announcing itself is heard at all, which is the half that
    /// lives here.
    #[test]
    fn every_host_write_entry_point_reaches_the_page_record() {
        let mut state = DeviceState::new(DeviceId(1), crate::model::PAGE_SHIFT_X86);
        let mut epoch = state.content.host_writes.epoch();
        for announce in [
            &mut DeviceState::note_host_wrote_guest_ram as &mut dyn FnMut(&mut DeviceState),
            &mut |s: &mut DeviceState| s.note_host_wrote_pages(vec![0x1000]),
            &mut |s: &mut DeviceState| s.note_host_wrote_mapping(7),
        ] {
            announce(&mut state);
            let now = state.content.host_writes.epoch();
            assert_ne!(now, epoch, "a host write into guest RAM went unannounced");
            epoch = now;
        }
    }
}

#[cfg(test)]
mod mapping_declaration_tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_X86};

    fn state() -> DeviceState {
        DeviceState::new(DeviceId(1), PAGE_SHIFT_X86)
    }

    #[test]
    fn surface_registry_owns_the_surface_namespace() {
        let mut state = state();
        assert!(state.map_surface(7));

        assert!(state
            .surfaces
            .mappings
            .entries
            .contains_key(&SurfaceId::new(7)));
        assert_eq!(state.surfaces.mappings.len(), 1);
    }

    #[test]
    fn mapper_service_consumes_only_the_capture_for_the_published_entry() {
        let mut state = state();
        let capture = MapperCapture {
            producer: 8,
            mapper_device_kva: 0x1000,
            request_kind: reims_vgpu_protocol::MapperRequestKind::Map,
            mapping_internal: 0x2000,
        };
        state.publish_mapper_capture(capture);

        assert_eq!(state.take_mapper_capture(7), None);
        assert_eq!(state.take_mapper_capture(8), Some(capture));
        assert_eq!(state.take_mapper_capture(8), None);

        state.restore_mapper_capture(capture);
        assert_eq!(state.take_mapper_capture(8), Some(capture));
    }

    #[test]
    fn absent_mapper_device_capture_cannot_erase_the_service_identity() {
        let mut state = state();
        state.observe_mapper_device(0x1234);
        state.observe_mapper_device(0);
        assert_eq!(state.mapper_device_kva(), 0x1234);
    }

    #[test]
    fn a_reused_surface_id_cannot_inherit_present_write_classification() {
        let mut state = state();
        assert!(state.map_surface(7));
        state.note_surface_composite(7);
        assert_eq!(state.surface_write_kind(7), SurfaceWriteKind::Composite);

        assert!(state.unmap_surface(7));
        assert!(state.map_surface(7));
        assert_eq!(state.surface_write_kind(7), SurfaceWriteKind::Unknown);
    }

    #[test]
    fn registered_surface_plan_publishes_pages_and_derivation_together() {
        let mut state = state();
        assert!(state.map_surface(7));
        state.surfaces.mappings.get_mut(&7).unwrap().pages.entries = vec![1, 2];

        let effect = state
            .adopt_registered_surface_plan(
                SurfaceId::new(7),
                vec![3, 4],
                TaskId::new(9),
                0x123,
                &[0; reims_vgpu_protocol::DEVICE_DESC_LEN],
            )
            .unwrap();
        assert!(effect.changed);
        assert!(effect.replaced);
        let mapping = state.surfaces.mappings.get(&7).unwrap();
        assert_eq!(mapping.pages.entries, vec![3, 4]);
        assert_eq!(mapping.pages.surface_walk.unwrap().task_id, 9);
        assert_eq!(mapping.pages.surface_walk.unwrap().backing_pfn, 0x123);
        assert!(mapping.device_desc_complete().is_some());
    }

    fn declared(state: &DeviceState, id: u32) -> (u32, u32) {
        let m = state
            .surfaces
            .mappings
            .get(&id)
            .expect("the mapping exists");
        (m.content.guest_page_generation, m.content.surface_epoch)
    }

    /// Re-declaring a mapping at the same extent but a different pixel format
    /// withdraws the content claim, because the claim is about what the bytes
    /// *mean* and the guest has just changed that.
    ///
    /// The reset used to test the extent alone, on the reasoning that a format
    /// change moves the `TargetIdentity` and so picks up a different resident by
    /// itself. `present_identity::surface_format` collapses several guest
    /// declarations onto one semantic layout and falls back to the scanout order
    /// for any it cannot express, so that reasoning does not hold for every
    /// pair — and the failure is a resident served against an epoch stamped
    /// under the previous interpretation.
    #[test]
    fn re_declaring_a_mapping_at_a_new_format_withdraws_its_content_claim() {
        let mut state = state();
        assert!(state.set_mapping_geom(7, 640, 480, 0x50));
        let m = state
            .surfaces
            .mappings
            .get_mut(&7)
            .expect("the mapping exists");
        m.content.guest_page_generation = 9;
        m.content.surface_epoch = 4;

        // Same declaration in every field: nothing to withdraw.
        assert!(state.set_mapping_geom(7, 640, 480, 0x50));
        assert_eq!(
            declared(&state, 7),
            (9, 4),
            "an unchanged declaration is not a new surface"
        );

        // Format alone, at one extent.
        assert!(state.set_mapping_geom(7, 640, 480, 0x19));
        assert_eq!(
            declared(&state, 7),
            (0, 0),
            "the bytes mean something else now, so nothing may claim they are the content"
        );
    }

    /// The extent half of the same rule, kept beside it so neither can be
    /// dropped without the other being visible.
    #[test]
    fn re_declaring_a_mapping_at_a_new_extent_withdraws_its_content_claim() {
        let mut state = state();
        assert!(state.set_mapping_geom(7, 640, 480, 0x50));
        let m = state
            .surfaces
            .mappings
            .get_mut(&7)
            .expect("the mapping exists");
        m.content.guest_page_generation = 9;
        m.content.surface_epoch = 4;
        assert!(state.set_mapping_geom(7, 800, 480, 0x50));
        assert_eq!(declared(&state, 7), (0, 0));
    }

    #[test]
    fn remap_clears_the_whole_declaration_without_stale_geometry() {
        let mut state = state();
        assert!(state.map_surface(7));
        assert!(state.set_mapping_geom(7, 640, 480, 0x50));
        assert!(state.set_mapping_device_desc(7, &vec![0; reims_vgpu_protocol::DEVICE_DESC_LEN]));

        let before = state.surfaces.mappings.get(&7).expect("mapping");
        assert_eq!(
            before.geometry(),
            Some(SurfaceGeometry {
                width: 640,
                height: 480,
                format: 0x50,
            })
        );
        assert!(before.device_desc_complete().is_some());

        assert!(state.map_surface(7));
        let after = state
            .surfaces
            .mappings
            .get(&7)
            .expect("mapping remains registered");
        assert_eq!(after.geometry(), None);
        assert!(after.device_desc_complete().is_none());
        assert_eq!(
            after.geometry_or_zero(),
            SurfaceGeometry {
                width: 0,
                height: 0,
                format: 0,
            }
        );
    }
}

#[cfg(test)]
mod slot_table_reach_tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_X86};

    /// No task id is out of range, and the mark records how far the guest went.
    ///
    /// This test used to assert the opposite: that `MAX_TASKS + 4096` was
    /// *refused* and still moved the mark. The mark existed to say whether that
    /// bound was close, because a refusal counter cannot — a boot stopping at id
    /// 12 and one stopping at 255 both report zero refusals. The answer it gave
    /// was 25x of headroom, which is not a derivation, and `DeviceState::tasks`
    /// is a `TaskTable` over a map now. `u32::MAX` is the largest id the wire can
    /// carry, so defining a task there is the strongest form of "nothing is out
    /// of range".
    ///
    /// The mark stays, as an occupancy reading on that map rather than a
    /// distance to a refusal.
    #[test]
    fn no_task_id_is_out_of_range() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        assert_eq!(state.observations.max_task_id_seen(), 0);

        state.define_task(12, 0x1000, 2);
        assert!(state.tasks.is_active(12), "an ordinary id is accepted");
        assert_eq!(state.observations.max_task_id_seen(), 12);

        let past = u32::MAX;
        state.define_task(past, 0x1000, 2);
        assert!(
            state.tasks.is_active(past),
            "a task id is a full u32 on the wire and its storage is a map"
        );
        assert_eq!(state.observations.max_task_id_seen(), past);

        // High-water, not last-seen: a later smaller id does not lower it.
        state.define_task(3, 0x1000, 2);
        assert_eq!(state.observations.max_task_id_seen(), past);
        assert_eq!(
            state.tasks.live_count(),
            3,
            "sparse ids do not create the entries between them"
        );
    }

    /// The mapping id space has no ceiling, and this is the test that says so.
    ///
    /// It used to assert the opposite half of the same line — that one past
    /// `MAX_MAPPINGS` was refused and still moved the reach mark. That bound
    /// refused ids its own storage would have held: `surface_mappings` is an
    /// unbounded registry.
    /// `u32::MAX` is the largest id the wire can carry, so accepting it here is
    /// the strongest form of "nothing is out of range", and a reinstated
    /// ceiling fails on the first assertion.
    ///
    /// The mark still moves, because it is an occupancy reading on the map now
    /// rather than a distance to a refusal.
    #[test]
    fn no_mapping_id_is_out_of_range() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        assert_eq!(state.observations.max_mapping_id_seen(), 0);

        assert!(state.map_surface(39), "an ordinary id is accepted");
        assert_eq!(state.observations.max_mapping_id_seen(), 39);

        assert!(
            state.map_surface(u32::MAX),
            "a mapping id is a full u32 on the wire and its storage is a map"
        );
        assert!(state.surfaces.mappings.contains_key(&u32::MAX));
        assert_eq!(state.observations.max_mapping_id_seen(), u32::MAX);

        assert!(
            !state.map_surface(0),
            "0 is the unbound sentinel and is the one id that stays refused"
        );
        assert!(!state.surfaces.mappings.contains_key(&0));
    }

    /// Every task mutator feeds the mark, not just the one that creates the
    /// task — a guest that only ever calls `set_object_list` or `insert_object`
    /// on a high id would otherwise be invisible.
    ///
    /// These three still refuse `past`, but for the reason they always should
    /// have: no task is defined there. That is a liveness answer, not a range
    /// one, and it is the same answer they would give for any undefined id.
    #[test]
    fn every_task_mutator_feeds_the_reach_mark() {
        let past = u32::MAX;
        for (name, mut state) in [
            ("delete_task", DeviceState::new(DeviceId(1), PAGE_SHIFT_X86)),
            (
                "set_object_list",
                DeviceState::new(DeviceId(1), PAGE_SHIFT_X86),
            ),
            (
                "insert_object",
                DeviceState::new(DeviceId(1), PAGE_SHIFT_X86),
            ),
        ] {
            match name {
                "delete_task" => assert!(state.delete_task(past).is_none()),
                "set_object_list" => assert!(!state.set_object_list(past, 1, 1)),
                _ => assert!(!state.insert_object(past, 7)),
            }
            assert_eq!(
                state.observations.max_task_id_seen(),
                past,
                "{name} refused without recording the reach"
            );
            assert!(
                !state.tasks.is_active(past),
                "{name} must not have defined the task it refused"
            );
        }
    }
}

#[cfg(test)]
mod task_lifecycle_effect_tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_X86};
    use reims_vgpu_protocol::{DepthStencilDescriptor, HeapObject, SerializerRef};

    fn state() -> DeviceState {
        DeviceState::new(DeviceId(1), PAGE_SHIFT_X86)
    }

    #[test]
    fn reset_preserves_the_injected_diagnostic_audit_policy() {
        // Both policies are asserted, and both are set to their non-default arm,
        // because a reset that rebuilt the witness from `Default` would still
        // pass this test if either one happened to be the default.
        let injected = reims_vgpu_core::GatherPolicies {
            audit: reims_vgpu_core::AuditDensity::EveryBind,
            vouch: reims_vgpu_core::VouchPolicy::Withheld,
        };
        assert_ne!(injected, reims_vgpu_core::GatherPolicies::default());

        let mut state =
            DeviceState::new_with_gather_policies(DeviceId(1), PAGE_SHIFT_X86, injected);

        let _ = state.reset();

        assert_eq!(state.content.sampled.gather_witness.policies(), injected);
    }

    #[test]
    fn task_mutations_report_kind_and_exact_namespace_retirement() {
        let mut state = state();
        assert_eq!(
            state.define_task(7, 0x4000, 3),
            TaskDefinitionEffect {
                kind: TaskDefinitionKind::FirstDefinition,
                retired: TaskNamespaceRetirement::default(),
            }
        );

        state.task_objects.depth_stencil.register(
            7,
            SerializerRef::new(11),
            Arc::new(DepthStencilDescriptor::default()),
        );
        state.task_objects.depth_stencil.register(
            7,
            SerializerRef::new(12),
            Arc::new(DepthStencilDescriptor::default()),
        );
        state.set_fence_generation(7, 21, 1);
        state.set_event_generation(7, 31, 1);
        state.set_event_generation(7, 32, 1);
        state
            .task_objects
            .heaps
            .register(7, SerializerRef::<HeapObject>::new(41), Arc::new(()));

        assert_eq!(
            state.define_task(7, 0x8000, 3),
            TaskDefinitionEffect {
                kind: TaskDefinitionKind::RedefinedSameRoot,
                retired: TaskNamespaceRetirement {
                    heaps: 1,
                    depth_stencil_states: 2,
                    fences: 1,
                    events: 2,
                    ..TaskNamespaceRetirement::default()
                },
            }
        );

        state.set_fence_generation(7, 22, 2);
        assert_eq!(
            state.define_task(7, 0x8000, 4),
            TaskDefinitionEffect {
                kind: TaskDefinitionKind::RedefinedNewRoot,
                retired: TaskNamespaceRetirement {
                    fences: 1,
                    ..TaskNamespaceRetirement::default()
                },
            }
        );

        state.set_event_generation(7, 33, 3);
        state
            .task_objects
            .heaps
            .register(7, SerializerRef::<HeapObject>::new(42), Arc::new(()));
        assert_eq!(
            state.delete_task(7),
            Some(TaskNamespaceRetirement {
                heaps: 1,
                events: 1,
                ..TaskNamespaceRetirement::default()
            })
        );
        assert_eq!(state.delete_task(7), None);
    }

    #[test]
    fn heap_resource_heap_and_task_deletes_retire_exact_residency_generations() {
        use reims_vgpu_protocol::ObjectKind;

        let mut state = state();
        state.define_task(7, 0x4000, 3);
        let heap_ref = SerializerRef::<HeapObject>::new(41);
        state.task_objects.heaps.register(7, heap_ref, Arc::new(()));
        let heap = state.task_objects.heaps.identity(7, heap_ref).unwrap();
        let texture = state.task_objects.resources.register(
            7,
            9,
            Arc::new(TaskResource::new(
                ListObjectEntry::new(ObjectKind::TextureView, 0, 0),
                Arc::from([]),
            )),
        );
        state
            .task_objects
            .resources
            .link_heap_texture(7, 9, heap, None)
            .unwrap();
        let resource_key = ComputeStorageResidencyKey::heap_allocation(
            heap,
            texture.semantic_id().unwrap(),
            4,
            4,
            0x50,
        );
        state.content.compute_residency.publish(resource_key, 3);

        assert!(state.delete_object(7, 9));
        assert!(!state.content.compute_residency.contains(&resource_key));
        assert_eq!(
            state.host_materializations.take_compute_residents(),
            vec![resource_key]
        );

        let heap_key = ComputeStorageResidencyKey::heap_placement(heap, 0, 64, 4, 4, 0x50);
        state.content.compute_residency.publish(heap_key, 4);
        assert!(state.delete_heap(7, heap_ref));
        assert!(!state.content.compute_residency.contains(&heap_key));
        assert_eq!(
            state.host_materializations.take_compute_residents(),
            vec![heap_key]
        );

        state.task_objects.heaps.register(7, heap_ref, Arc::new(()));
        let replacement = state.task_objects.heaps.identity(7, heap_ref).unwrap();
        assert_ne!(replacement, heap);
        let task_key = ComputeStorageResidencyKey::heap_placement(replacement, 0, 64, 4, 4, 0x50);
        state.content.compute_residency.publish(task_key, 5);
        assert!(state.delete_task(7).is_some());
        assert!(!state.content.compute_residency.contains(&task_key));
        assert_eq!(
            state.host_materializations.take_compute_residents(),
            vec![task_key]
        );
    }
}
