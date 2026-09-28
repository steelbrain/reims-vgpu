//! Device-owned state: registers, rings, tasks, mapper, present, fail log.

use crate::model::{LruBytesMemo, GFX_MMIO_SIZE, MAX_CHANNELS};
use crate::runtime::decode::resource::{
    DecodeStatus as ResourceDecodeStatus, Descriptor, ListObjectEntry, SamplerDescriptor,
};
use reims_vgpu_core::access::BackingId;
use reims_vgpu_core::identity::{ObjectListRef, ResourceId};
use reims_vgpu_core::namespace::Teardown;
use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU32, Ordering};
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
    /// A texture indirect exec packet shorter than its declared descriptor.
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

/// How many leading payload words an unknown child opcode echoes. Four covers
/// every unknown packet a driven boot has produced whole (the largest is 76
/// bytes of which 64 are payload) while bounding the line for a command that
/// carries a large buffer; `plen` always reports the true length, so a reader
/// can tell an echo that was cut from one that was complete.
///
/// The `_MAX` says this is a cut and not a size: the echo stops here whether or
/// not the record has run out, which is why `plen` carries the true length
/// beside it.
const UNKNOWN_OPCODE_ECHO_WORDS_MAX: usize = 4;

/// The wire fields a child packet this device did not execute reports, shared by
/// the unknown-opcode and unimplemented-command records.
///
/// One spelling on purpose: the two records get read side by side and diffed
/// against each other, so a copied field list here would become the next
/// divergence the moment one of them grows a field.
fn packet_echo_fields(
    channel: u32,
    opcode: u16,
    total_size: u32,
    stamp_count: u16,
    payload: &[u8],
) -> Vec<(&'static str, String)> {
    let mut fields = vec![
        ("ch", channel.to_string()),
        ("opcode", format!("{opcode:#x}")),
        ("total_size", total_size.to_string()),
        ("stamps", stamp_count.to_string()),
        ("plen", payload.len().to_string()),
    ];
    // Whole words only, in wire order, so a reader can line the echo up against
    // the packet layout. A trailing sub-word tail is reported by `plen` rather
    // than zero-padded into a word that the guest never wrote.
    let words = payload
        .chunks_exact(4)
        .take(UNKNOWN_OPCODE_ECHO_WORDS_MAX)
        .map(|word| format!("{:#010x}", crate::protocol::endian::ld32(word)))
        .collect::<Vec<_>>()
        .join(":");
    if !words.is_empty() {
        fields.push(("payload", words));
    }
    fields
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

impl crate::observe::Decline for FailEvent {
    fn slug(&self) -> &'static str {
        match self {
            Self::UnknownRootOpcode { .. } => "unknown_root_opcode",
            Self::UnknownChildOpcode { .. } => "unknown_child_opcode",
            // Delegates for the same reason the malformed variants do: the
            // command *is* the reason, so one slug per command beats one coarse
            // slug the reader then has to disambiguate from the fields.
            Self::UnimplementedChildCommand { command, .. } => command.slug(),
            // The malformed variants delegate: the specific check *is* the
            // fault, so forwarding keeps one slug per check instead of two
            // coarse ones that the reader would then have to disambiguate by
            // hand from the fields.
            Self::MalformedRootPacket { fault, .. } | Self::MalformedChildPacket { fault, .. } => {
                fault.slug()
            }
            Self::UnsupportedExec { fault, .. } => fault.slug(),
            Self::BadMmioAccess { .. } => "bad_mmio_access",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::UnknownRootOpcode { opcode, total_size } => vec![
                ("opcode", format!("{opcode:#x}")),
                ("total_size", total_size.to_string()),
            ],
            Self::UnknownChildOpcode {
                channel,
                opcode,
                total_size,
                stamp_count,
                payload,
            } => packet_echo_fields(*channel, *opcode, *total_size, *stamp_count, payload),
            Self::UnimplementedChildCommand {
                channel,
                command,
                opcode,
                total_size,
                stamp_count,
                payload,
            } => {
                let mut fields =
                    packet_echo_fields(*channel, *opcode, *total_size, *stamp_count, payload);
                // Ahead of the wire fields: the command name is what a reader is
                // scanning for, and it is the one thing this record has that the
                // unknown-opcode record does not.
                fields.insert(0, ("cmd", command.command().to_string()));
                fields
            }
            Self::MalformedRootPacket { head, .. } => vec![("head", head.to_string())],
            Self::MalformedChildPacket { channel, head, .. } => {
                vec![("ch", channel.to_string()), ("head", head.to_string())]
            }
            Self::UnsupportedExec { channel, .. } => vec![("ch", channel.to_string())],
            Self::BadMmioAccess { offset, size } => vec![
                ("offset", format!("{offset:#x}")),
                ("size", size.to_string()),
            ],
        }
    }
}

/// Gfx named registers + sparse backing for unnamed offsets.
#[derive(Clone, Debug)]
pub struct GfxRegs {
    pub version: u32,
    pub control_fifo: u32,
    pub fifo_length: u32,
    pub fifo_written: u32,
    /// Main-FIFO consumer byte counter (0x100c), host-advanced. Lock-free
    /// `Arc<AtomicU32>` shared with the registry slot: the guest `writeFifo`
    /// producer spins on this register, so it must observe drain progress
    /// live while the drain worker owns the device lock.
    pub fifo_read: Arc<AtomicU32>,
    pub fifo_start: u32,
    pub root_page: u32,
    pub fifo_base_page: u32,
    /// Read-to-clear interrupt status (0x1014). Lock-free `Arc<AtomicU32>` so
    /// the guest ISR MMIO read observes live bits even while the drain worker
    /// owns the device lock (ack fast: a cached/stale mask loses signals).
    /// The `Arc` is shared with the device registry slot and survives reset.
    pub interrupt_status_disp: Arc<AtomicU32>,
    /// Read-to-clear stamp-signal status (0x1018). Same lock-free contract.
    pub interrupt_status_gpu: Arc<AtomicU32>,
    /// Fault interrupt status (0x102c), host-set, guest-read (not r2c). Same
    /// lock-free read rail (the guest ISR reads it right after 0x1018).
    pub interrupt_fault: Arc<AtomicU32>,
    /// Child channels rung since the drain last folded them in (0x1020/0x1028).
    ///
    /// The lock-free *write* rail, and the only one: every other register the
    /// guest writes finds the device lock free, while this doorbell was
    /// measured queueing about a hundred times a second and applying up to
    /// 45 ms late (`gfx_doorbell_delay off_0x1020`). It queued because
    /// `device_gfx_write` takes the device lock with `try_lock` and the drain
    /// worker holds that lock for its whole tranche, so the delay is the
    /// tranche — `max_age_us` tracks `max_tranche_us` to within 3 %.
    ///
    /// A doorbell is the one register that can be taken this way, because it
    /// carries no state the decode depends on: its whole effect is to say a
    /// child channel has work. So the guest's ring ORs a bit here without any
    /// lock, and [`crate::runtime::drain::fold_rung_child_doorbells`] moves it
    /// into the open-domain set / `pending.child_mask` — including *inside* the
    /// channel loop, so a channel rung mid-tranche is served by that tranche
    /// rather than the next one.
    ///
    /// Bit `n` is channel `n`; bit 0 is unused because channel 0 is the main
    /// FIFO, which has its own register.
    ///
    /// The `Arc` is shared with the device registry slot and survives reset,
    /// like the three above.
    pub child_doorbell_rung: Arc<AtomicU32>,
    pub efi_display: u32,
    pub efi_mode_select: u32,
    pub efi_fb_start: u64,
    pub efi_fb_length: u32,
    pub efi_fb_depth: u32,
    pub efi_fb_mode: u32,
    pub efi_fb_stride: u32,
    /// Backing for offsets without dedicated fields (word index).
    pub sparse: BTreeMap<u32, u32>,
}

impl Default for GfxRegs {
    fn default() -> Self {
        Self {
            version: 0,
            control_fifo: 0,
            fifo_length: 0,
            fifo_written: 0,
            fifo_read: Arc::new(AtomicU32::new(0)),
            fifo_start: 0,
            root_page: 0,
            fifo_base_page: 0,
            interrupt_status_disp: Arc::new(AtomicU32::new(0)),
            interrupt_status_gpu: Arc::new(AtomicU32::new(0)),
            interrupt_fault: Arc::new(AtomicU32::new(0)),
            child_doorbell_rung: Arc::new(AtomicU32::new(0)),
            efi_display: 0,
            efi_mode_select: 0,
            efi_fb_start: 0,
            efi_fb_length: 0,
            efi_fb_depth: 0,
            efi_fb_mode: 0,
            efi_fb_stride: 0,
            sparse: BTreeMap::new(),
        }
    }
}

impl GfxRegs {
    pub fn sparse_get(&self, offset: u64) -> u32 {
        let idx = (offset / 4) as u32;
        self.sparse.get(&idx).copied().unwrap_or(0)
    }

    pub fn sparse_set(&mut self, offset: u64, val: u32) {
        if offset < GFX_MMIO_SIZE {
            self.sparse.insert((offset / 4) as u32, val);
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct IosfcRegs {
    pub ring_base: u64,
    pub capacity: u32,
    pub desc_table: u64,
    pub producer: u32,
    pub consumer: u32,
}

/// Per-channel child ring cache (page list decoded from base_pfn).
#[derive(Clone, Debug, Default)]
pub struct ChannelRing {
    pub valid: bool,
    pub base_pfn: u32,
    pub length: u32,
    pub page_gpas: Vec<u64>,
}

/// Task directory / object-list ownership.
#[derive(Clone, Debug, Default)]
pub struct TaskEntry {
    pub active: bool,
    pub length: u64,
    pub directory_pfn: u32,
    pub object_list_pfn: u32,
    pub object_list_count: u32,
}

/// The device's tasks, keyed by the guest's own task id.
///
/// # It is a map because a task id is a `u32`
///
/// This was `[TaskEntry; MAX_TASKS]` with `MAX_TASKS = 256`, and the number was
/// never derived from anything: a task id is a full `u32` on the wire —
/// `decode_replace_physical` and every resource-list command read it with `ld32`
/// — and past 256 `define_task` returned `false`, the task never existed, and
/// every guest command that needed it was lost. The bound was defended by
/// distance rather than by a derivation: [`DeviceState::max_task_id_seen`]
/// measured a driven boot stopping at id 10, which is 25x of headroom and no
/// answer at all to what a heavier guest does.
///
/// Absence and inactivity are the same state here, which is what makes the
/// translation from the array safe. The array wrote `TaskEntry::default()` on
/// delete — `active: false` — and every reader tested `active` before using an
/// entry; this returns `None` for an id nothing defined, and those readers now
/// get `None` where they used to get an inactive entry. There is no third state
/// for one of them to have branched on.
///
/// The two full-range probes in `runtime::objects` are the visible win.
/// `backing_claimant_tasks` walked all 256 ids and `backing_probe_order` chained
/// `1..256`; both now walk the live ids, because the ids in between were
/// refused by the liveness test at the probe and contributed nothing. Same
/// answer, and the walk is the size of the guest's task set instead of a
/// constant.
#[derive(Clone, Debug, Default)]
pub struct TaskTable(BTreeMap<u32, TaskEntry>);

impl TaskTable {
    pub const fn new() -> Self {
        Self(BTreeMap::new())
    }

    /// The task `id` names, or `None` if the guest never defined it.
    ///
    /// Note the entry may still be present and inactive — `delete_task` removes
    /// it, but a caller that cares about liveness should keep its own `active`
    /// test rather than assume `Some` means live.
    pub fn get(&self, id: u32) -> Option<&TaskEntry> {
        self.0.get(&id)
    }

    pub fn get_mut(&mut self, id: u32) -> Option<&mut TaskEntry> {
        self.0.get_mut(&id)
    }

    /// Whether `id` names a task this device will walk a page table for.
    ///
    /// The single spelling of the liveness test. It had a dozen copies as
    /// `tasks[id].active` against an array that answered for every id in range,
    /// and each of those is now a `get` that can also answer `None`.
    pub fn is_active(&self, id: u32) -> bool {
        self.get(id).is_some_and(|t| t.active)
    }

    /// Install a task under `id`, replacing whatever was there.
    pub fn define(&mut self, id: u32, entry: TaskEntry) {
        self.0.insert(id, entry);
    }

    pub fn remove(&mut self, id: u32) {
        self.0.remove(&id);
    }

    /// Every live task with its id, ascending.
    ///
    /// Ascending because the array it replaced was walked in id order and two
    /// probes in `runtime::objects` depend on that order deciding which of
    /// several claimant tasks a surface resolves against.
    pub fn live(&self) -> impl Iterator<Item = (u32, &TaskEntry)> {
        self.0
            .iter()
            .filter(|(_, t)| t.active)
            .map(|(&id, t)| (id, t))
    }

    /// [`Self::live`] without the entries.
    pub fn live_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.live().map(|(id, _)| id)
    }

    /// How many tasks are live. Not the size of the id space — there is none.
    pub fn live_count(&self) -> usize {
        self.live_ids().count()
    }
}

/// A resource constructed from one task/object-list reference.
///
/// The object-list entry and its descriptor are construction input. Once the
/// resource exists, binds retrieve this retained object rather than consulting
/// guest memory again. The guest ends that lifetime explicitly by deleting the
/// resource or the task that owns it.
#[derive(Debug)]
pub struct TaskResource {
    pub entry: ListObjectEntry,
    pub descriptor: Arc<[u8]>,
    /// Typed form of the construction descriptor, decoded exactly once for
    /// this resource lifetime.
    ///
    /// The serialized object map resolves a reference to an object, not to a
    /// byte string that every bind is expected to parse again. Keep the bytes
    /// because a few partial/legacy consumers deliberately accept shapes the
    /// total decoder refuses, and keep that refusal too so those consumers do
    /// not silently widen the total contract.
    decoded: OnceLock<Result<Descriptor, ResourceDecodeStatus>>,
    /// Mapper-ref-texture construction side effects, completed once for this resource
    /// lifetime. The mapping id is immutable construction state; physical
    /// backing replacement invalidates the mapping's pages without rebuilding
    /// the texture object.
    mapper_ref_texture_mapping: OnceLock<u32>,
    /// Identity whose strong lifetime is exactly this serialized resource.
    /// Direct backend objects keep only a weak reference, so deletion—not an
    /// arbitrary idle timeout—makes them reclaimable.
    lifetime: Arc<TaskResourceLifetime>,
    /// What the running rail retains for exactly this resource's lifetime, in
    /// the rail's own vocabulary. See [`RailResourceState`].
    ///
    /// The model owns the slot and the drop; it does not own — and cannot name
    /// — the contents.
    rail: Mutex<Option<Box<dyn Any + Send + Sync>>>,
}

impl TaskResource {
    pub fn new(entry: ListObjectEntry, descriptor: Arc<[u8]>) -> Self {
        Self {
            entry,
            descriptor,
            decoded: OnceLock::new(),
            mapper_ref_texture_mapping: OnceLock::new(),
            lifetime: Arc::new(TaskResourceLifetime::new()),
            rail: Mutex::new(None),
        }
    }

    /// Resolve this resource's immutable construction descriptor once.
    pub fn decoded(&self) -> &Result<Descriptor, ResourceDecodeStatus> {
        self.decoded.get_or_init(|| {
            crate::runtime::decode::resource::decode_descriptor(
                self.entry.object_type,
                &self.descriptor,
            )
        })
    }

    /// The guest-VA allocation this resource's construction descriptor names,
    /// through the decode this resource already did.
    ///
    /// `None` for every object that does not name storage by an address in its
    /// own task, and for one that names it with a handle or a size the guest
    /// has not written yet.
    ///
    /// See [`Descriptor::backing_window`] for why it is the allocation base and
    /// not a texture's texel base, and
    /// [`crate::runtime::decode::resource::descriptor_is_heap_placement`] for
    /// why the bytes are consulted before the decode: a placement arrives under
    /// the ordinary texture type and the typed form cannot tell.
    ///
    /// Taken off [`Self::decoded`] rather than re-parsing the bytes, because
    /// the callers are hot: a payment asks it per read and a re-point asks it
    /// per cached copy in the task.
    #[must_use]
    pub fn backing_window(&self, page_shift: u32) -> Option<(u64, u64)> {
        if crate::runtime::decode::resource::descriptor_is_heap_placement(&self.descriptor) {
            return None;
        }
        self.decoded().as_ref().ok()?.backing_window(page_shift)
    }

    pub fn lifetime_ref(&self) -> TaskResourceLifetimeRef {
        TaskResourceLifetimeRef {
            id: self.lifetime.id,
            live: Arc::downgrade(&self.lifetime),
        }
    }

    pub(crate) fn registered_mapper_ref_texture_mapping(&self) -> Option<u32> {
        self.mapper_ref_texture_mapping.get().copied()
    }

    pub(crate) fn register_mapper_ref_texture_mapping(&self, mapping_id: u32) -> u32 {
        *self.mapper_ref_texture_mapping.get_or_init(|| mapping_id)
    }

    /// The running rail's own state for this resource, created empty on first
    /// ask, borrowed for the length of `f`.
    ///
    /// `None` means the slot is already held by a *different* rail's type.
    /// [`crate::backend::select`] latches one rail per process, so no live
    /// build can reach it; it is an answer rather than a panic because the one
    /// caller that asks already has a lawful "nothing retained" reply and a
    /// caught panic at a `reims_vgpu_qemu_*` entry point is a dead device.
    pub fn with_rail_state<T, R>(&self, f: impl FnOnce(&mut T) -> R) -> Option<R>
    where
        T: RailResourceState,
    {
        let mut held = self
            .rail
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let slot = held.get_or_insert_with(|| Box::<T>::default() as Box<dyn Any + Send + Sync>);
        Some(f(slot.downcast_mut::<T>()?))
    }
}

/// State one rail retains for exactly one serialized resource's lifetime.
///
/// # Why the model may not name it
///
/// A rail's per-resource retention pins host GPU memory, and the guest ends
/// that lifetime explicitly by deleting the resource. Keeping it *in*
/// [`TaskResource`] is what makes the release deterministic — `AGENTS.md`'s rule
/// that resource state representing guest work follows the contract-owned guest
/// lifetime, rather than being swept on a timer — and it is what lets a warm
/// bind read the retention back without entering the rail at all.
///
/// But *what* is retained is the rail's own vocabulary. The Vulkan rail retains
/// a lease keyed by an engine target identity; Metal retains nothing of the
/// kind and the words mean nothing to it. Spelling those types here made
/// `model` depend on `backend::vulkan`, which closed a cycle — `model` →
/// `backend::vulkan` → `runtime` → `model` — and, on the both-rails build where
/// the `cfg` that used to hide the field is on, put one rail's private table in
/// the other's reach with nothing but convention keeping it out.
///
/// So the model owns the slot and the drop, and the rail owns the contents.
/// A rail reaches its own state through [`TaskResource::with_rail_state`], which
/// it can only call for a type it can name.
pub trait RailResourceState: Any + Default + Send + Sync {}

/// State the running rail keeps for exactly one device's lifetime, in the
/// rail's own vocabulary.
///
/// # Why the model may not name it
///
/// [`RailResourceState`] states the rule for one resource; this is the same
/// rule one scope up. The Vulkan rail retains, per `(task, pipeline_ref)`, a
/// resolved pipeline holding two translated SPIR-V shaders and an engine
/// pipeline-object identity. Metal retains nothing of the kind. Spelling that
/// type in [`DeviceState`] made `model` depend on `backend::vulkan` — the same
/// cycle — and left the struct with two shapes across a feature boundary.
///
/// # What the model still owns
///
/// The slot, the drop, and *when* a lifetime ends. Task teardown is a guest
/// event the model decodes, so the model tells the rail about it through
/// [`Self::delete_task`]; what a task's references mean, and what dropping them
/// costs, is the rail's. The rail reports its own count under its own census
/// name, because a name the model chose would describe a table it cannot see.
pub trait RailDeviceState: Any + Send + Sync + std::fmt::Debug {
    /// This state as `Any`, so the rail that installed it can read it back.
    fn as_any(&self) -> &dyn Any;

    /// Drop everything held under one task's reference namespaces, reporting
    /// what went under this rail's own census names.
    fn delete_task(&self, task_id: u32);

    /// The device's own lifetime has ended. Let go of everything held under it,
    /// reporting what went under this rail's own census names.
    ///
    /// # Why this is a told event and not a `Drop`
    ///
    /// [`Self::delete_task`] already establishes the division: the model
    /// decodes *when* a lifetime ends and the rail knows *what letting go
    /// costs*. A device lifetime ends the same way and had no such telling —
    /// the slot was simply dropped, at `DeviceState::reset`'s wholesale
    /// replacement and again when the device leaves the registry.
    ///
    /// That was survivable only while the slot held nothing a host has to be
    /// told about. It holds identities and `Arc`s today, and every module this
    /// crate still has to move into it — residency, variant families, transfer
    /// and record state — owns native objects whose destruction goes through a
    /// device, which `Drop` cannot reach and cannot report. So the ending is
    /// told, exactly once, at the one place both doors pass through.
    ///
    /// Called with `&self` for the same reason `delete_task` is: the rail owns
    /// whatever synchronization its own tables need, and the model may not name
    /// them.
    fn end_device(&self);
}

/// Both doors out of a device lifetime, joined into one telling.
///
/// A device ends two ways and neither used to reach the rail. [`DeviceState::reset`]
/// replaces the whole struct, and `device::device_destroy` drops it out of the
/// registry. Both of those drop the [`RailDeviceState`] slot, so `Drop` is the
/// one place both pass through — which is why the telling lives here rather
/// than in `reset`, where the destroy door would have missed it.
///
/// Read through [`std::sync::OnceLock::get`] and not through
/// [`DeviceState::rail_state`]: an ending must not *create* a rail state no
/// rail ever installed, which is what the initializing accessor would do and
/// what would then report an empty teardown for a device that had no rail.
impl Drop for DeviceState {
    fn drop(&mut self) {
        if let Some(rail) = self.rail.get() {
            rail.end_device();
        }
    }
}

static NEXT_TASK_RESOURCE_LIFETIME: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1);

#[derive(Debug)]
struct TaskResourceLifetime {
    id: u64,
}

impl TaskResourceLifetime {
    fn new() -> Self {
        let id = NEXT_TASK_RESOURCE_LIFETIME
            .fetch_update(
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
                |id| id.checked_add(1),
            )
            .expect("task resource lifetime identity exhausted");
        Self { id }
    }
}

/// Weak backend-facing proof that one serialized resource still exists.
#[derive(Clone, Debug)]
pub struct TaskResourceLifetimeRef {
    id: u64,
    live: std::sync::Weak<TaskResourceLifetime>,
}

impl TaskResourceLifetimeRef {
    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn is_live(&self) -> bool {
        self.live.strong_count() != 0
    }
}

/// Per-task resource objects, keyed by the **name** the object namespace issued
/// for the slot rather than by the slot number.
///
/// # Why the key carries a generation
///
/// A slot number alone is not an identity: the guest reuses slots, and a memo
/// keyed by one outlives the object it was built for. That is not a theoretical
/// hazard on this interface — the guest replaces an object by writing over its
/// own object-list record, with no packet — and a stale hit binds the bytes of
/// whatever used to live there, which is a wrong texture rather than a missing
/// one.
///
/// So the key is [`reims_vgpu_core::identity::ResourceId`], which
/// [`DeviceState::declare_object`] issues and whose generation advances on every
/// declaration. A caller reaches this table only through a name the namespace
/// has already resolved, and a name the namespace has retired cannot be spelled
/// — which makes "the memo and the namespace disagree" unrepresentable instead
/// of merely unlikely. The namespace is the authority for what a reference
/// names; this is a memo of the bytes behind a name it has already answered.
///
/// Interior synchronization keeps resource lookup available to encode helpers
/// that only borrow [`DeviceState`] immutably. Those helpers run while the
/// device already owns its state, but making the registry itself synchronized
/// also makes the lifetime rule explicit instead of relying on that outer
/// serialization.
#[derive(Debug, Default)]
pub struct TaskResources(Mutex<BTreeMap<(u32, ResourceId), Arc<TaskResource>>>);

impl TaskResources {
    pub fn get(&self, task_id: u32, name: ResourceId) -> Option<Arc<TaskResource>> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&(task_id, name))
            .cloned()
    }

    /// Publish a newly constructed object unless another lookup won the race.
    pub fn register(
        &self,
        task_id: u32,
        name: ResourceId,
        resource: Arc<TaskResource>,
    ) -> Arc<TaskResource> {
        Arc::clone(
            self.0
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .entry((task_id, name))
                .or_insert(resource),
        )
    }

    pub fn delete(&self, task_id: u32, name: ResourceId) -> bool {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&(task_id, name))
            .is_some()
    }

    /// Every constructed resource in one task, as `(name, resource)`.
    ///
    /// Collected rather than iterated under the lock, so a caller may decode
    /// descriptors -- which is what every caller does -- without holding it.
    pub fn in_task(&self, task_id: u32) -> Vec<(ResourceId, Arc<TaskResource>)> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .filter(|&(&(task, _), _)| task == task_id)
            .map(|(&(_, name), resource)| (name, Arc::clone(resource)))
            .collect()
    }

    /// How many constructed resources this device holds, across every task.
    ///
    /// Not a per-task question, and that is the point of it: the one reader is
    /// the device-info reply census, whose destination is a guest page frame in
    /// no task's address space at all, so "is there any storage this could
    /// collide with" cannot be asked of one task.
    pub fn len(&self) -> usize {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }

    /// Whether this device holds no constructed resource at all.
    ///
    /// Spelled out beside [`Self::len`] because clippy asks for it and because
    /// the census's question really is the emptiness rather than the count.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn delete_task(&self, task_id: u32) -> usize {
        let mut resources = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let before = resources.len();
        resources.retain(|&(task, _), _| task != task_id);
        before - resources.len()
    }
}

/// The first object reference seen at each guest-VA window, per task, and the
/// canonical backing identity of every piece of storage this device can name.
///
/// Behind a lock for the reason [`TaskResources`] is: the claim is made on the
/// resolve path, which holds [`DeviceState`] shared.
///
/// # Two tables, one counter
///
/// Storage is reached two ways on this interface and they are not two answers
/// about one thing: an object names pages *by address* in its own task, or it
/// reaches them *through a mapping*, and `crate::runtime::objects::replace_physical`
/// already treats the two as separate counters because a re-point advances
/// exactly one of them. So each route interns on its own key — a window on
/// `(task, base)` with [`DeviceState::storage_incarnation`], a mapping on its id
/// with [`MappingEntry::map_generation`].
///
/// **They share [`Self::next_id`], and that is the load-bearing part.** The
/// identity is a bare `u64` that the dependency compiler compares for equality
/// with no idea which route minted it, so two tables with two counters would
/// hand a window and a mapping the same number and make unrelated storage alias
/// — false equality, which hands storage back under a live reader. One monotone
/// counter makes the two key spaces one identity space by construction.
///
/// Nothing needs the reverse guarantee — that one piece of storage cannot be
/// reached by both routes and so get two numbers — because the routes are
/// disjoint by object type rather than by convention:
/// `crate::runtime::objects::backing_id` answers by address only for buffers and
/// address-named textures, and refuses the two mapping-named texture types to
/// the mapping route; the surface-backing object a mapping's pages are walked
/// from is `OBJECT_TYPE_BACKING`, which the address route names no storage for
/// at all.
pub struct BackingWindowRefs {
    windows: Mutex<BTreeMap<(u32, u64), WindowFacts>>,
    /// The identity of each mapping's surface storage, at the incarnation it
    /// was minted for. See the type doc for why this is a second table and not
    /// a second counter.
    mappings: Mutex<BTreeMap<u32, MappingFacts>>,
    /// The identity of a bare guest page frame.
    ///
    /// The third key space, and the type doc's rule applies to it unchanged: it
    /// interns on [`Self::next_id`] like the other two, so a frame's identity
    /// can never equal a window's or a mapping's.
    ///
    /// One reader — a `CmdGetDeviceInfo` reply, whose destination is a page
    /// frame in no task's address space, so neither of the other two key spaces
    /// can name it. It holds no incarnation because a page frame has no
    /// re-point: the guest names a physical frame and that frame is the storage.
    frames: Mutex<BTreeMap<u32, u64>>,
    /// The next dense backing identity to hand out.
    ///
    /// Monotone and never reused, which is the whole of what makes an id an
    /// identity: two windows that are different storage must not be able to
    /// arrive at one number, and a counter that wrapped or recycled could.
    /// A `u64` at one per re-point does not run out.
    next_id: Mutex<u64>,
}

/// What this device knows about one guest-VA window in one task.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WindowFacts {
    /// The reference that first constructed an object over this window.
    ///
    /// `None` when the entry was created by an identity lookup rather than by
    /// a construction. The two reach this table by different paths and a
    /// placeholder reference number would be reported as a live claimant by
    /// the alias reading, which is a finding about nobody.
    first_ref: Option<u32>,
    /// The incarnation [`Self::id`] was minted for.
    ///
    /// The entry holds one identity, not a history: when the pages behind the
    /// window are replaced, the *current* id becomes a new one and the previous
    /// one stops being mintable — which is right, because nothing needs to mint
    /// an old identity. Work planned against it holds it already, in its own
    /// claim, and that is what keeps the old storage alive. Keeping a history
    /// here would make the table grow with the boot instead of with the live
    /// namespace, for entries no caller could ever ask for.
    incarnation: StorageIncarnation,
    /// The canonical backing identity of this window at that incarnation.
    id: u64,
}

/// What this device knows about the identity of one mapping's surface storage.
///
/// The mapping analogue of [`WindowFacts`], minus the claim: a mapping has no
/// "first reference" because it is not reached by a reference at all. It holds
/// one identity rather than a history for the same reason — nothing mints an
/// identity it already holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MappingFacts {
    /// The [`MappingEntry::map_generation`] [`Self::id`] was minted for.
    map_generation: u32,
    /// The canonical backing identity of this mapping's surface at that
    /// generation.
    id: u64,
}

impl Default for BackingWindowRefs {
    fn default() -> Self {
        Self {
            windows: Mutex::new(BTreeMap::new()),
            mappings: Mutex::new(BTreeMap::new()),
            frames: Mutex::new(BTreeMap::new()),
            // Zero is never handed out, so a zeroed structure cannot read as a
            // valid identity -- the same rule `SlotGeneration` follows.
            next_id: Mutex::new(1),
        }
    }
}

impl std::fmt::Debug for BackingWindowRefs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let claims = self.windows.lock().unwrap_or_else(|e| e.into_inner());
        let mappings = self.mappings.lock().unwrap_or_else(|e| e.into_inner());
        f.debug_struct("BackingWindowRefs")
            .field("windows", &claims.len())
            .field("mappings", &mappings.len())
            .finish()
    }
}

impl BackingWindowRefs {
    /// Reserve an identity for a window this device has not seen before.
    ///
    /// Split from the insert so the counter is never advanced under the
    /// windows lock, and so a lookup that finds a current entry does not touch
    /// it at all.
    fn mint(&self) -> u64 {
        let mut next = self.next_id.lock().unwrap_or_else(|e| e.into_inner());
        let id = *next;
        *next += 1;
        id
    }

    /// How many identities this device has handed out.
    ///
    /// The counter is monotone and shared by both key spaces, so this is the
    /// whole of "what storage could a newly minted identity be equal to" — and
    /// zero is the answer that says *nothing*, which is a claim about the
    /// identity space rather than about any one table's contents.
    ///
    /// It is the counter minus one, because the counter starts at one: zero is
    /// never handed out, so a fresh device holds `next_id == 1` and has minted
    /// nothing. Reading the raw counter here would report every device as
    /// having handed out an identity, which is the answer that keeps a term
    /// open forever.
    fn minted(&self) -> u64 {
        self.next_id
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .saturating_sub(1)
    }

    fn claim(
        &self,
        task_id: u32,
        ref_: u32,
        base: u64,
        incarnation: StorageIncarnation,
    ) -> Option<u32> {
        let fresh = WindowFacts {
            first_ref: Some(ref_),
            incarnation,
            id: 0,
        };
        match self
            .windows
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry((task_id, base))
        {
            std::collections::btree_map::Entry::Vacant(slot) => {
                slot.insert(fresh);
                None
            }
            std::collections::btree_map::Entry::Occupied(mut slot) => {
                match slot.get().first_ref {
                    // Created by an identity lookup, so no construction has
                    // claimed it. This one is the first.
                    None => {
                        slot.get_mut().first_ref = Some(ref_);
                        None
                    }
                    Some(holder) => (holder != ref_).then_some(holder),
                }
            }
        }
    }

    fn take(&self, task_id: u32, ref_: u32, base: u64, incarnation: StorageIncarnation) {
        let mut windows = self.windows.lock().unwrap_or_else(|e| e.into_inner());
        windows
            .entry((task_id, base))
            .and_modify(|facts| facts.first_ref = Some(ref_))
            .or_insert(WindowFacts {
                first_ref: Some(ref_),
                incarnation,
                id: 0,
            });
    }

    /// The identity of this window at this incarnation, minting one if the
    /// window is new or its pages have been replaced since it was last asked.
    fn identity(&self, task_id: u32, base: u64, incarnation: StorageIncarnation) -> u64 {
        // Minted outside the windows lock, and discarded unspent when the entry
        // turns out to be current. An identity is never reused, so an unspent
        // one is a gap in the sequence and not a hazard -- the invariant is
        // that two different pieces of storage never share a number, not that
        // every number is used.
        let candidate = self.mint();
        let mut windows = self.windows.lock().unwrap_or_else(|e| e.into_inner());
        let facts = windows.entry((task_id, base)).or_insert(WindowFacts {
            first_ref: None,
            incarnation,
            id: 0,
        });
        if facts.id == 0 || facts.incarnation != incarnation {
            facts.incarnation = incarnation;
            facts.id = candidate;
        }
        facts.id
    }

    /// The identity of this mapping's surface storage at this generation,
    /// minting one if the mapping is new or its page list has been replaced
    /// since it was last asked.
    ///
    /// The window route's [`Self::identity`] with the other key, including the
    /// mint-outside-the-lock and the discard-unspent: an identity is never
    /// reused, so an unspent one is a gap in the sequence and not a hazard.
    ///
    /// The table is bounded by the mapping namespace because a mapping slot is
    /// never removed from [`DeviceState::mappings`] — it is re-generationed in
    /// place — which is also what makes one entry per id sound. A recycled slot
    /// carries a bumped `map_generation` (every writer of the page list bumps
    /// it), so `(mapping, generation)` never repeats and a later surface at the
    /// same id cannot inherit the previous one's number.
    /// The identity of a bare guest page frame, minted once and stable after.
    ///
    /// No incarnation, unlike the window and mapping routes: those two exist
    /// because storage under a name can be replaced, and a page frame is not a
    /// name — it is the storage. Two asks for one frame are two asks about the
    /// same bytes and get the same number.
    fn frame_identity(&self, pfn: u32) -> u64 {
        if let Some(&id) = self
            .frames
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&pfn)
        {
            return id;
        }
        // Minted outside the table's lock, as the window route mints, so the
        // counter is never advanced under it.
        let id = self.mint();
        *self
            .frames
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(pfn)
            .or_insert(id)
    }

    fn mapping_identity(&self, mapping_id: u32, map_generation: u32) -> u64 {
        let candidate = self.mint();
        let mut mappings = self.mappings.lock().unwrap_or_else(|e| e.into_inner());
        let facts = mappings.entry(mapping_id).or_insert(MappingFacts {
            map_generation,
            id: 0,
        });
        if facts.id == 0 || facts.map_generation != map_generation {
            facts.map_generation = map_generation;
            facts.id = candidate;
        }
        facts.id
    }

    fn delete_task(&self, task_id: u32) {
        self.windows
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|&(t, _), _| t != task_id);
    }
}

/// One immutable sampler object constructed in a task's sampler-reference space.
///
/// Sampler references are not resource-list ownership records. They have their
/// own explicit delete command, so keeping them in [`TaskResources`] would let
/// two distinct reference spaces destroy one another when their integers
/// happen to collide. Construction snapshots the descriptor once; binds retain
/// and retrieve this decoded state until that sampler or its task is deleted.
#[derive(Debug)]
pub struct TaskSamplerState {
    pub descriptor: SamplerDescriptor,
}

/// Immutable objects in an API-specific task/reference namespace.
///
/// The map has no capacity or eviction policy. Its entries are object
/// lifetimes: an explicit delete removes one reference and task teardown removes
/// the namespace. A capacity would invent a third lifetime event that the guest
/// never sent.
pub struct TaskReferenceStates<T>(Mutex<BTreeMap<(u32, u32), Arc<T>>>);

impl<T> Default for TaskReferenceStates<T> {
    fn default() -> Self {
        Self(Mutex::new(BTreeMap::new()))
    }
}

impl<T> std::fmt::Debug for TaskReferenceStates<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let states = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        f.debug_struct("TaskReferenceStates")
            .field("entries", &states.len())
            .finish()
    }
}

impl<T> TaskReferenceStates<T> {
    pub fn contains(&self, task_id: u32, ref_: u32) -> bool {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains_key(&(task_id, ref_))
    }

    pub fn get(&self, task_id: u32, ref_: u32) -> Option<Arc<T>> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&(task_id, ref_))
            .cloned()
    }

    /// Publish a fully constructed object unless another resolver won the race.
    pub fn register(&self, task_id: u32, ref_: u32, state: Arc<T>) -> Arc<T> {
        Arc::clone(
            self.0
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .entry((task_id, ref_))
                .or_insert(state),
        )
    }

    pub fn delete(&self, task_id: u32, ref_: u32) -> bool {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&(task_id, ref_))
            .is_some()
    }

    pub fn delete_task(&self, task_id: u32) -> usize {
        let mut states = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let before = states.len();
        states.retain(|&(task, _), _| task != task_id);
        before - states.len()
    }

    /// Drop every state under every task, returning how many there were.
    ///
    /// The device-lifetime counterpart of [`Self::delete_task`], and not
    /// expressible as a loop over it: the guest's task ids are not enumerable
    /// from here, and a device ending does not name them one at a time.
    pub fn clear(&self) -> usize {
        let mut states = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        std::mem::take(&mut *states).len()
    }
}

/// Per-task sampler objects, keyed by the sampler API's reference space.
pub type TaskSamplerStates = TaskReferenceStates<TaskSamplerState>;

/// Per-task depth-stencil states, keyed by that API's reference space.
///
/// A depth-stencil state is an immutable object with its own explicit delete
/// command (`OPCODE_DELETE_DEPTH_STENCIL_STATE`), exactly like a sampler state
/// and a render pipeline state, so it belongs in this namespace and not in
/// [`TaskResources`] — whose type mask deliberately excludes object serializer-object,
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
/// command is the invalidation, which is why this needs no capacity and no
/// generation.
pub type TaskDepthStencilStates =
    TaskReferenceStates<crate::runtime::decode::resource::DepthStencilDescriptor>;

#[cfg(test)]
mod rail_resource_state_tests {
    use super::*;
    use crate::runtime::decode::resource::ListObjectEntry;
    use std::sync::atomic::AtomicU32;

    static LIVE: AtomicU32 = AtomicU32::new(0);

    #[derive(Default)]
    struct OneRail(u32);
    impl RailResourceState for OneRail {}

    #[derive(Default)]
    struct OtherRail;
    impl RailResourceState for OtherRail {}

    /// Counts its own construction and destruction, so a test can watch the
    /// resource's own drop release it.
    struct Pinned;
    impl Default for Pinned {
        fn default() -> Self {
            LIVE.fetch_add(1, Ordering::Relaxed);
            Self
        }
    }
    impl Drop for Pinned {
        fn drop(&mut self) {
            LIVE.fetch_sub(1, Ordering::Relaxed);
        }
    }
    impl RailResourceState for Pinned {}

    fn resource() -> TaskResource {
        TaskResource::new(ListObjectEntry::default(), Arc::from([]))
    }

    #[test]
    fn a_rails_slot_is_created_empty_once_and_then_borrowed() {
        let resource = resource();
        assert_eq!(resource.with_rail_state(|s: &mut OneRail| s.0), Some(0));
        assert_eq!(
            resource.with_rail_state(|s: &mut OneRail| {
                s.0 += 7;
                s.0
            }),
            Some(7)
        );
        assert_eq!(
            resource.with_rail_state(|s: &mut OneRail| s.0),
            Some(7),
            "a second ask must borrow the state the first one left"
        );
    }

    #[test]
    fn one_resource_holds_one_rails_state_and_no_others() {
        let resource = resource();
        assert_eq!(
            resource.with_rail_state(|s: &mut OneRail| s.0 = 3),
            Some(())
        );
        assert!(
            resource.with_rail_state(|_: &mut OtherRail| ()).is_none(),
            "a second rail must not be handed the first rail's slot"
        );
        assert_eq!(
            resource.with_rail_state(|s: &mut OneRail| s.0),
            Some(3),
            "and must not have disturbed what the first rail left there"
        );
    }

    #[test]
    fn deleting_the_resource_releases_what_the_rail_retained() {
        let live_before = LIVE.load(Ordering::Relaxed);
        let resource = resource();
        assert_eq!(resource.with_rail_state(|_: &mut Pinned| ()), Some(()));
        assert_eq!(
            LIVE.load(Ordering::Relaxed),
            live_before + 1,
            "the slot holds the rail's object"
        );
        drop(resource);
        assert_eq!(
            LIVE.load(Ordering::Relaxed),
            live_before,
            "the guest's delete releases the rail's retention at that instant, \
             not on a later sweep"
        );
    }
}

#[cfg(test)]
mod rail_device_state_tests {
    use super::*;
    use crate::model::PAGE_SHIFT_X86;
    use std::sync::atomic::AtomicU32;

    static ENDINGS: AtomicU32 = AtomicU32::new(0);
    static TASK_DELETIONS: AtomicU32 = AtomicU32::new(0);

    /// One rail type means one counter, and one of the tests below asserts the
    /// counter does **not** move — which a concurrently running sibling would
    /// break. Held for the body of each test rather than relying on
    /// `--test-threads=1`, because a gate that only holds under a flag somebody
    /// has to remember is not a gate.
    static SERIALIZE: Mutex<()> = Mutex::new(());

    fn alone() -> std::sync::MutexGuard<'static, ()> {
        SERIALIZE.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// A rail that records only that it was told, because "was the rail told"
    /// is the whole question at this boundary. What the telling costs is the
    /// rail's own, and is tested where the rail's table is.
    #[derive(Debug, Default)]
    struct CountingRail;

    impl RailDeviceState for CountingRail {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn delete_task(&self, _task_id: u32) {
            TASK_DELETIONS.fetch_add(1, Ordering::Relaxed);
        }
        fn end_device(&self) {
            ENDINGS.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The registry door: a device that leaves the registry is dropped, and the
    /// rail holding native objects under it has to be told before that drop
    /// finishes.
    #[test]
    fn a_device_that_is_dropped_ends_its_rails_device_lifetime_once() {
        let _alone = alone();
        let before = ENDINGS.load(Ordering::Relaxed);
        let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        assert!(
            state.rail_state::<CountingRail>().is_some(),
            "the rail installs its slot on first ask"
        );
        assert_eq!(
            ENDINGS.load(Ordering::Relaxed),
            before,
            "installing a slot is not an ending"
        );

        drop(state);
        assert_eq!(
            ENDINGS.load(Ordering::Relaxed),
            before + 1,
            "the drop tells the rail exactly once"
        );
    }

    /// The reset door. `DeviceState::reset` replaces the struct wholesale, so
    /// the ending has to arrive there too — and it is the same telling, which
    /// is why it is owned by `Drop` and not written at both sites.
    #[test]
    fn a_reset_ends_the_rails_device_lifetime_and_the_next_one_starts_empty() {
        let _alone = alone();
        let before = ENDINGS.load(Ordering::Relaxed);
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        assert!(state.rail_state::<CountingRail>().is_some());

        state.reset();
        assert_eq!(
            ENDINGS.load(Ordering::Relaxed),
            before + 1,
            "the reset ended the lifetime the rail had state under"
        );

        // The replacement is a different device lifetime with its own empty
        // slot: asking again installs a second one rather than resurrecting the
        // first, and ending *that* is a second telling.
        assert!(state.rail_state::<CountingRail>().is_some());
        drop(state);
        assert_eq!(
            ENDINGS.load(Ordering::Relaxed),
            before + 2,
            "the lifetime after the reset ends on its own terms"
        );
    }

    /// The accessor that installs a slot must not be the one an ending uses.
    /// A device no rail ever asked about has nothing to let go of, and telling
    /// it would report an empty teardown for a rail that was never there.
    #[test]
    fn a_device_no_rail_ever_claimed_ends_without_creating_one() {
        let _alone = alone();
        let before = ENDINGS.load(Ordering::Relaxed);
        drop(DeviceState::new(DeviceId(1), PAGE_SHIFT_X86));
        assert_eq!(
            ENDINGS.load(Ordering::Relaxed),
            before,
            "no slot, no ending"
        );
    }
}

#[cfg(test)]
mod task_reference_state_tests {
    use super::TaskReferenceStates;
    use std::sync::Arc;

    #[test]
    fn explicit_reference_and_task_deletion_are_the_only_retirement_events() {
        let states = TaskReferenceStates::default();
        let first = states.register(1, 7, Arc::new(10u32));
        let raced = states.register(1, 7, Arc::new(11u32));
        states.register(1, 8, Arc::new(12u32));
        states.register(2, 7, Arc::new(13u32));

        assert!(Arc::ptr_eq(&first, &raced), "the first construction wins");
        assert_eq!(*states.get(1, 7).unwrap(), 10);
        assert!(states.delete(1, 7));
        assert!(!states.contains(1, 7));
        assert!(states.contains(1, 8));
        assert!(states.contains(2, 7));

        assert_eq!(states.delete_task(1), 1);
        assert!(!states.contains(1, 8));
        assert!(states.contains(2, 7));
        assert_eq!(
            *first, 10,
            "an encoder owner remains valid after registry deletion"
        );
    }

    #[test]
    fn a_live_reference_population_has_no_capacity_eviction() {
        let states = TaskReferenceStates::default();
        for ref_ in 0..2048 {
            states.register(3, ref_, Arc::new(ref_));
        }
        for ref_ in 0..2048 {
            assert_eq!(*states.get(3, ref_).unwrap(), ref_);
        }
    }

    /// The device's ending takes every task, including the ones the guest never
    /// deleted — which is the case `delete_task` cannot cover, since nothing
    /// here can enumerate the guest's task ids.
    #[test]
    fn a_device_ending_clears_every_task_and_says_how_many_it_took() {
        let states = TaskReferenceStates::default();
        states.register(1, 7, Arc::new(10u32));
        states.register(1, 8, Arc::new(11u32));
        states.register(2, 7, Arc::new(12u32));
        let held = states.get(2, 7).unwrap();

        assert_eq!(states.clear(), 3, "every task's states, counted once");
        assert!(!states.contains(1, 7));
        assert!(!states.contains(1, 8));
        assert!(!states.contains(2, 7));
        assert_eq!(
            *held, 12,
            "a state an encoder still owns outlives the table, exactly as it \
             does across a reference delete"
        );
        assert_eq!(
            states.clear(),
            0,
            "a second ending has nothing left to take"
        );
    }
}

/// `tasks[id]` for a task the caller has already defined. **Tests only.**
///
/// 167 test sites index a fixture's task 1, and rewriting each into
/// `get(1).unwrap()` would trade the thing they are asserting for ceremony.
/// Production has no such impl, deliberately: every id there comes off the wire,
/// so it may name no task, and [`TaskTable::get`] is the accessor that says so.
/// A panicking index reachable from a decode path is a guest-triggerable abort,
/// which is why this is `#[cfg(test)]` rather than documented as "do not use".
#[cfg(test)]
impl std::ops::Index<u32> for TaskTable {
    type Output = TaskEntry;

    fn index(&self, id: u32) -> &TaskEntry {
        self.get(id)
            .unwrap_or_else(|| panic!("test indexed task {id}, which nothing defined"))
    }
}

/// See [`TaskTable`]'s `Index`. Tests that mutate a fixture's task in place —
/// clearing `active` or zeroing a directory to build the state a refusal path
/// needs — reach through this.
#[cfg(test)]
impl std::ops::IndexMut<u32> for TaskTable {
    fn index_mut(&mut self, id: u32) -> &mut TaskEntry {
        self.get_mut(id)
            .unwrap_or_else(|| panic!("test indexed task {id}, which nothing defined"))
    }
}

/// Why a `DeviceState` mutator refused a decoded guest record.
///
/// # The `*IdSentinel` five were `*IdRange`
///
/// Five of these named a *range* check, because `is_mapping_id` used to be
/// `id >= 1 && id < MAX_MAPPINGS` and one variant covered both halves. The
/// ceiling is gone — `mappings` is a `BTreeMap` keyed by the full `u32`, so it
/// refused ids its own storage would have held — and the only value these can
/// now refuse is 0, the device-wide "no mapping" sentinel that `runtime::draw`
/// reads as "this attachment is addressed by GVA".
///
/// So the slugs say `_id_sentinel`. A name that still said `_id_range` would
/// tell a reader ranking the fail log that the guest overran a table, and send
/// them looking for a bound that does not exist. Four sibling `*TaskIdRange`
/// variants were deleted outright in the same move, for the same reason: the
/// task table is a map too, and there is no id it refuses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StateMutationDecline {
    SetObjectListTaskInactive { task_id: u32 },
    InsertObjectTaskInactive { task_id: u32, object_ref: u32 },
    MapSurfaceIdSentinel { mapping_id: u32 },
    UnmapSurfaceIdSentinel { mapping_id: u32 },
    AttachMappingIdSentinel { mapping_id: u32 },
    AttachMappingInternalZero { mapping_id: u32 },
    MappingDeviceDescIdSentinel { mapping_id: u32 },
    MappingDeviceDescEmpty { mapping_id: u32 },
    MappingGeomIdSentinel { mapping_id: u32 },
    MappingGeomWidthZero { mapping_id: u32 },
    MappingGeomHeightZero { mapping_id: u32 },
    MappingGeomWidthRange { mapping_id: u32, width: u32 },
    MappingGeomHeightRange { mapping_id: u32, height: u32 },
}

impl crate::observe::Decline for StateMutationDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::SetObjectListTaskInactive { .. } => "model_set_object_list_task_inactive",
            Self::InsertObjectTaskInactive { .. } => "model_insert_object_task_inactive",
            Self::MapSurfaceIdSentinel { .. } => "model_map_surface_id_sentinel",
            Self::UnmapSurfaceIdSentinel { .. } => "model_unmap_surface_id_sentinel",
            Self::AttachMappingIdSentinel { .. } => "model_attach_mapping_id_sentinel",
            Self::AttachMappingInternalZero { .. } => "model_attach_mapping_internal_zero",
            Self::MappingDeviceDescIdSentinel { .. } => "model_mapping_device_desc_id_sentinel",
            Self::MappingDeviceDescEmpty { .. } => "model_mapping_device_desc_empty",
            Self::MappingGeomIdSentinel { .. } => "model_mapping_geom_id_sentinel",
            Self::MappingGeomWidthZero { .. } => "model_mapping_geom_width_zero",
            Self::MappingGeomHeightZero { .. } => "model_mapping_geom_height_zero",
            Self::MappingGeomWidthRange { .. } => "model_mapping_geom_width_range",
            Self::MappingGeomHeightRange { .. } => "model_mapping_geom_height_range",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        let mut fields = match self {
            Self::SetObjectListTaskInactive { task_id } => {
                vec![("task", task_id.to_string())]
            }
            Self::InsertObjectTaskInactive {
                task_id,
                object_ref,
            } => vec![
                ("task", task_id.to_string()),
                ("ref", object_ref.to_string()),
            ],
            Self::MapSurfaceIdSentinel { mapping_id }
            | Self::UnmapSurfaceIdSentinel { mapping_id }
            | Self::AttachMappingIdSentinel { mapping_id }
            | Self::AttachMappingInternalZero { mapping_id }
            | Self::MappingDeviceDescIdSentinel { mapping_id }
            | Self::MappingDeviceDescEmpty { mapping_id }
            | Self::MappingGeomIdSentinel { mapping_id }
            | Self::MappingGeomWidthZero { mapping_id }
            | Self::MappingGeomHeightZero { mapping_id }
            | Self::MappingGeomWidthRange { mapping_id, .. }
            | Self::MappingGeomHeightRange { mapping_id, .. } => {
                vec![("mapping", mapping_id.to_string())]
            }
        };
        match self {
            Self::MappingGeomWidthRange { width, .. } => {
                fields.push(("width", width.to_string()));
            }
            Self::MappingGeomHeightRange { height, .. } => {
                fields.push(("height", height.to_string()));
            }
            _ => {}
        }
        fields
    }
}

impl StateMutationDecline {
    fn emit(self, discriminant: u64) {
        crate::observe::Emit::decline("model_state_mutation", &self).fail_once(discriminant);
    }
}

impl TaskEntry {
    /// A task the guest has defined but not yet given an object list.
    ///
    /// `object_list_pfn` and `object_list_count` are **zero** because
    /// `DefineTask2` does not carry them. `SetObjectList` (`0x33`) does, and
    /// until it arrives the correct answer to "what object does ref N name" is
    /// "the guest has not said".
    ///
    /// This used to invent `pfn = 1, count = 0x100000` — a page frame the guest
    /// never named and a list of a million entries. Measured on the x86/Vulkan
    /// rail: `lookup_list_entry` then computed entry addresses of `0x1000 + off`
    /// for every task with no list, walked them, and failed with `gva_zero_pfn`
    /// because nothing is mapped there — after which the guest-read fallback
    /// walked the *neighbouring task's* page table at the same address and
    /// decoded whatever it found as this task's object-list entry. Seven such
    /// substitutions per boot, every boot, all from that one lookup.
    pub fn define(length: u64, directory_pfn: u32) -> Self {
        Self {
            active: true,
            length,
            directory_pfn,
            object_list_pfn: 0,
            object_list_count: 0,
        }
    }
}

/// Directed mapper capture from guest xregs at iosfc producer write.
#[derive(Clone, Copy, Debug, Default)]
pub struct MapperCapture {
    /// Producer index that published this request (entry = producer - 1).
    pub producer: u32,
    pub mapper_device_kva: u64,
    pub request_type: u32,
    /// Guest kernel VA of MappingInternal.
    pub mapping_internal: u64,
}

/// Which incarnation of the pages behind a task-local name a value describes.
///
/// A pair rather than one number, because the two things that can make a name
/// describe different pages live at two scopes: a re-point or a release names
/// one reference, and a task teardown or redefinition ends every name in the
/// task at once. Summing them would collide, and walking every name at teardown
/// would miss the references this device has never touched — which is most of
/// them, since the guest publishes objects by writing its own object-list page.
///
/// Opaque on purpose. It is an identity component and never an amount: nothing
/// may subtract two of these, order them, or use one half alone. See
/// [`DeviceState::storage_incarnation`] for what advances each.
///
/// Both halves wrap at `u32`, exactly as [`MappingEntry::map_generation`] does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct StorageIncarnation {
    epoch: u32,
    count: u32,
}

impl StorageIncarnation {
    /// The pair as one value, for mixing into a canonical backing identity.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        ((self.epoch as u64) << 32) | self.count as u64
    }
}

/// The guest page table and GPU-VA base a mapping's [`MappingEntry::
/// page_entries`] were walked from, when the list came from a backing record
/// plan.
///
/// Latched at the one site that assigns those entries so the two cannot drift
/// apart. It exists so a later reader can *repeat* the walk without repeating
/// the search: `resolve_backing_ex` finds the surface object by probing up
/// to 256 task object lists, and that cost is why the page list is cached rather
/// than re-derived. The walk itself is cheap — one page-table translation per
/// page — and it is the only thing that can say whether the cached list still
/// names the guest's memory.
/// It carries the [`MappingEntry::map_generation`] it was latched at, and a
/// reader must check that before trusting it. Six sites clear or replace
/// `page_entries` and every one of them bumps the generation, so a carried-over
/// walk is unusable by construction rather than by every future writer
/// remembering to retire a second field — the same rule
/// [`MappingEntry::guest_write_token_gen`] states for the same reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackingWalk {
    /// Task whose page table translated the backing pages.
    pub task_id: u32,
    /// `getGPUVirtualAddress() >> page_shift` of the surface backing — page `i`
    /// of the list is `(backing_pfn + i) << page_shift` in that task.
    pub backing_pfn: u32,
    /// `map_generation` of the list this walk produced.
    pub map_generation: u32,
}

/// Who owns a resource's authoritative bytes, as the guest last stated it and as
/// the device last produced them.
///
/// The bools start `false` because nothing has been said yet, and "nothing has
/// been said" is a third state that neither `true` nor `false` can carry on its
/// own: a resource the guest has never named in a validity quad must not be
/// treated as having been declared stale on either side. `host_stated` and
/// `guest_stated` record whether the corresponding bit is a statement or a
/// default.
///
/// # Why the two sequence numbers, and not just `host_valid`
///
/// `host_valid` alone is a latch, and a latch is wrong here. The guest's
/// `clear_host_valid` says "my CPU write is newer than your last frame **as of
/// this submission**". It is not a standing property of the resource: the moment
/// the device renders into that surface again, the device's frame is the newer
/// one, and a writeback that reads a latched `host_valid == false` would refuse
/// to deliver it — forever, since nothing in the protocol re-affirms a resource
/// the guest is no longer writing.
///
/// One measured boot showed exactly that: 2 415 refused writebacks concentrated
/// on three surfaces (1 800 on one 1240x400 layer, 502 on the 1920x1080 root),
/// which is one `clear_host_valid` each latching every later frame away.
///
/// So the comparison is a happens-before between the guest's last claim and the
/// device's last publish, both stamped from [`DeviceState::next_validity_seq`].
/// Causal, not a heuristic: whoever wrote last owns the bytes.
///
/// # What the four bools are for, now that the seqs decide
///
/// They are the **record** of what the guest said, and nothing reads them to
/// decide anything. That is deliberate, and not the same as dropping them: the
/// guest emits four distinct ops and this is where all four land, so a boot can
/// be asked what it was told and not only what was done about it.
///
/// `set_host_valid` in particular drives nothing, because the device has a
/// strictly better witness for the same fact — its own publish, made when it
/// happens rather than one submission ahead. One boot measured the two agreeing
/// on 19 135 of 19 135 stores. Keeping the guest's version as a second input to
/// the same decision would be two spellings of one value with a way to disagree.
///
/// `guest_valid` / `guest_stated` are the only home for `clear_guest_valid` and
/// `set_guest_valid`, which live traffic barely uses (17 and 0 in a measured
/// boot).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResourceValidity {
    /// The device's copy holds the authoritative bytes.
    pub host_valid: bool,
    /// The guest's own pages hold the authoritative bytes.
    pub guest_valid: bool,
    /// The guest has set or cleared `host_valid` at least once.
    pub host_stated: bool,
    /// The guest has set or cleared `guest_valid` at least once.
    pub guest_stated: bool,
    /// Sequence at the guest's last `clear_host_valid` for this resource.
    /// Zero means the guest has never claimed a CPU write to it.
    pub host_cleared_seq: u64,
    /// Sequence at the device's last publication of newer pixels for this
    /// resource — a deferred Store's content publish, or a write of its guest
    /// pages.
    pub host_published_seq: u64,
}

/// Whether anything has read the copies the last landed render flush made.
///
/// A render flush lands one frame in two places: the mapping's guest pages and
/// the host surface cache. It is armed by a Store and landed by the next fence
/// with no reader having asked for either copy, so "is this flush owed at all"
/// is a question about consumers, and nothing measured it. Each leg is marked
/// unread when a flush lands it, and cleared by the first host-side reader of
/// that leg, so the *next* flush of the same mapping can report whether the
/// previous one was consumed.
///
/// `pages_unread` staying set does not prove nothing read the pages. The guest
/// CPU can load them with no device operation at all and leaves no trace here.
/// It proves only that no reader inside the device took them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RenderFlushWitness {
    /// A render flush has landed this mapping at least once, so the two flags
    /// below describe a real flush rather than a mapping that never had one.
    pub landed: bool,
    /// The flush stored a host surface cache copy, so `cache_unread` below is
    /// a statement about a copy that exists.
    ///
    /// A flush whose frame was borrowed from the engine's readback buffer
    /// stores no cache copy at all — it drops the entry instead, because the
    /// memory holding the frame goes back to the pool
    /// ([`crate::runtime::mapping_write::write_bgra8_uncached`]). Scoring one
    /// of those as an unread cache copy would report a copy that was never
    /// made, and `render_flush_cache_unread` is exactly the number a future
    /// reader would use to decide whether the cache leg is worth keeping. So
    /// the leg is only counted where there is a leg.
    pub cache_stored: bool,
    /// No host-side reader has taken the host surface cache copy since the
    /// flush stored it. Meaningful only where `cache_stored`.
    pub cache_unread: bool,
    /// No host-side reader has gathered the guest pages since the flush wrote
    /// them.
    pub pages_unread: bool,
    /// `observe::elapsed_us` when the flush landed, so the next one can say how
    /// long its predecessor survived.
    ///
    /// An unread flush replaced a whole frame later is the compositor
    /// repainting, and is the rate the rail is designed for. An unread flush
    /// replaced in under a millisecond is a *burst* superseding itself — the
    /// same surface written and rewritten inside one drain tranche — and that
    /// is work no fence boundary separated and nothing could have observed
    /// between. The two have the same `pages_unread` and completely different
    /// consequences, so the age is what tells them apart.
    ///
    /// # Read, and it is the first shape
    ///
    /// Two 25 s driven Safari probes on one x86/PCI/Vulkan boot, 121.0 and
    /// 123.4 fps:
    ///
    /// ```text
    /// render_flush_age_sub_ms         0        0
    /// render_flush_age_sub_frame     94       92
    /// render_flush_age_frame_plus  3079     3090
    /// ```
    ///
    /// **No flush is ever replaced inside a millisecond, and 97% survive a
    /// whole frame.** So the 99% that nothing reads are not redundant writes of
    /// one surface inside a burst — they are one full-screen composite per
    /// displayed frame, written back once each, at exactly the rate the guest
    /// paints. Superseding windows across fence boundaries has nothing to
    /// collapse, and the rail is at its floor for the rate it is asked to run
    /// at.
    ///
    /// That also reframes the 116 ms drain tranche carrying 19 flushes: those
    /// are nineteen *frames* of backlog drained at once, not nineteen writes of
    /// one frame. The worker fell behind and caught up. At `duty` 0.85 it has
    /// almost no headroom to absorb anything, so a hitch is the flush rail's
    /// cost showing up as latency rather than a separate defect — and the only
    /// remaining route to that cost is making the undeclared guest read
    /// observable.
    pub landed_us: u64,
}

/// IOSurface mapper registry entry keyed by mapping_id.
#[derive(Clone, Debug, Default)]
pub struct MappingEntry {
    pub mapped: bool,
    pub has_geom: bool,
    pub width: u32,
    pub height: u32,
    pub format: u16,
    pub content_generation: u32,
    /// What the guest has said about who owns this resource's authoritative
    /// bytes, driven by the two producers of the validity quad: the per-resource
    /// table in every `EXEC_INDIRECT2` payload, and `CmdInvalidateResources`.
    ///
    /// The host framework carries the matching pair as `PGResource._hostValid` /
    /// `._guestValid`, set through `setIsHostValid:` / `setIsGuestValid:`.
    pub validity: ResourceValidity,
    /// Epoch of this mapping's *surface content* in the sense a mapper-ref-texture render
    /// LOAD needs: it advances whenever the pixels that Load would seed from
    /// could have changed, wherever they live.
    ///
    /// Strictly coarser than [`Self::content_generation`], and deliberately so.
    /// `content_generation` counts writes to the mapping's *guest pages*, which
    /// misses the one publisher that writes only the host shadow: the deferred
    /// mapper-ref-texture Store stores into `surface_cache` and arms a window instead of
    /// scattering into guest pages. `surface_cache` holds exactly one entry per
    /// mapping, so a sibling Store at a *different* geometry replaces the entry
    /// an older geometry's resident is being compared against while
    /// `content_generation` never moves — the same one-entry-per-mapping hazard
    /// that cost the `deferred_flush_lost reason=cache_miss` class. Bumping here
    /// on that publish makes the sibling case a mismatch, so the older geometry
    /// falls back to the CPU seed rather than loading from a resident whose
    /// currency nothing established.
    ///
    /// Compared by the running rail against its own resident-content epoch
    /// to decide whether a mapper-ref-texture LOAD may take `LoadOp::LoadFromTarget` and
    /// skip its CPU seed entirely. Never read to decide *what* to present or
    /// draw — only whether a known-equal upload can be elided.
    pub surface_content_epoch: u32,
    /// Who has read what the last landed render flush of this mapping wrote.
    /// See [`RenderFlushWitness`].
    pub render_flush: RenderFlushWitness,
    /// Bumped whenever the guest page list / map lifetime changes (MAP, UNMAP,
    /// ReplacePhysical, MappingInternal reattach, page-table refresh that
    /// changes PFNs). Used as `TargetIdentity` generation for resident
    /// import-present so a recycled mid never reuses a stale GPU target, and
    /// as a fail-closed check before zero-copy DMA into contig views.
    pub map_generation: u32,
    /// Guest page-table entries (valid bit + PFN); empty until resolved.
    pub page_entries: Vec<u32>,
    /// Page entries retired by a trailing `DeleteIOSurfaceBacking2` while the
    /// id may already carry a NEW incarnation (the delete trails the guest
    /// CPU-side release asynchronously; ids recycle within ~20 ms under
    /// scroll). Fingerprint for the next resolve: an identical re-resolved
    /// plan is the SAME incarnation (stale delete — keep generation, resident,
    /// deferred windows); a different plan is a genuine new incarnation
    /// (bump + drop condemned windows). Cleared by every explicit lifecycle
    /// event (fresh MAP, unmap, MappingInternal reattach, ReplacePhysical).
    pub condemned_entries: Option<Vec<u32>>,
    /// Guest KVA of MappingInternal (from capture or recover).
    pub mapping_internal: u64,
    pub page_table_kva: u64,
    /// Cached `sIOSurfaceDeviceDescriptor` (0x200) from MappingInternal+0x38.
    /// Used for biplanar plane selection by texture geometry; empty when unknown.
    pub device_desc: Vec<u8>,
    /// Contiguous host-VA view over `page_entries` (`HostOps::map_pages`,
    /// mach_vm_remap of guest RAM). 0 = not built. This is the surface storage
    /// for the guest mapping. Guest CPU writes and host page reads see this
    /// allocation directly; on a capable unified-memory backend an imported
    /// render attachment retains the same view. Retired (never freed in place)
    /// whenever `page_entries` change; see `DeviceState::retired_views`.
    pub contig_ptr: usize,
    pub contig_len: usize,
    /// Guest-physical pages represented by `contig_ptr`, in allocation order.
    ///
    /// Kept with the view because a resource synchronization names the
    /// resource, not a freshly reconstructed destination. The backend retains
    /// this footprint with an imported attachment and uses it to order host
    /// readers against the GPU write without walking the mapping again at
    /// Store time.
    pub contig_footprint: Option<crate::runtime::guest_ram::GuestPageFootprint>,
    /// Checked backend-import bound over `contig_ptr`, created once for this
    /// mapping incarnation. Keeping it on the mapping makes every plane view a
    /// slice of one resource-owned allocation instead of minting a new import
    /// identity for each bind.
    pub contig_import: Option<std::sync::Arc<crate::runtime::guest_ram::GuestRamImport>>,
    /// `map_generation` whose page list the host refused to expose as one
    /// packed view. `None` = not asked for the current list.
    ///
    /// The host answer is stable for one page list, and `map_generation` names
    /// that list — the same key that makes `contig_ptr` above safe to cache.
    /// Without it every caller repeats a host mapping attempt that cannot
    /// become possible until the guest changes the list.
    pub contig_refused_gen: Option<u32>,
    /// Live [`crate::runtime::host::HostOps::track_guest_writes`] token for the
    /// page list in [`Self::page_entries`], or 0 when the host cannot observe
    /// guest writes (or none has been asked for yet).
    ///
    /// Retired next to [`Self::contig_ptr`] and for the same reason: both name
    /// the page list as it stood, so anything that changes the list invalidates
    /// both. A token that outlived its list would report writes to pages this
    /// surface no longer owns and miss writes to the ones it does.
    pub guest_write_token: u64,
    /// [`Self::map_generation`] the token above was built for.
    ///
    /// The lifecycle mutators retire the token eagerly, but they are not the
    /// only writers of [`Self::page_entries`]: the mapper's plan adoption and
    /// the backing page refresh both replace the list in place, and both retired
    /// the contiguous view while leaving the token behind — a token naming
    /// pages the surface no longer owns, which is the one thing it must never
    /// be. Rather than add a third and a fourth site to remember,
    /// `map_generation` is the key: every writer of the list already bumps it
    /// exactly when the list changes, so a token whose generation does not
    /// match is unusable by construction, and the eager retirement is left as
    /// what it should have been — a way to free host state promptly rather than
    /// the thing correctness rests on.
    pub guest_write_token_gen: u32,
    /// [`crate::runtime::host::HostOps::guest_write_gen`] as it stood when this
    /// mapping's pixels were last published by a device Store.
    ///
    /// The other half of the mapper-ref-texture seed currency test.
    /// [`Self::surface_content_epoch`] can only witness writers inside this
    /// crate — every caller of `mark_mapping_written` is one — and a surface's
    /// pages are plain guest RAM the guest CPU stores into with no device
    /// operation at all. This is what sees that store.
    ///
    /// 0 means no Store has stamped it, or the host could not answer, and
    /// never compares equal to a live generation (the host's first readable
    /// generation is 1).
    pub guest_write_gen_at_store: u64,
    /// A DisplaySwap has presented this surface since it was last unmapped.
    ///
    /// Presentation is what makes a surface one of WindowServer's framebuffers,
    /// and those are the surfaces it copies between with the CPU — see
    /// [`DeviceState::guest_copies_framebuffers`]. "The guest CPU wrote it" is
    /// not the same set: Safari's content tiles are CPU-written too and never
    /// presented, and writing every CPU-written surface back eagerly was
    /// measured to cost what turning the lazy writeback off everywhere does.
    pub scanout_presented: bool,
    /// Task id that last owned this surface as a backing `OBJECT_TYPE_BACKING`
    /// object (0 = no non-trivial hint; task 0 is always probed first anyway).
    /// `resolve_backing_ex` probes this task right after task 0 so a
    /// per-bind present-path scan short-circuits instead of walking all 256
    /// task slots. Purely a search-order hint — a stale/wrong value only costs
    /// one extra probe before the full-table fallback re-finds the owner.
    pub owner_task_hint: u32,
    /// How [`Self::page_entries`] were derived, when they came from a backing
    /// surface plan — see [`BackingWalk`]. `None` for every other source, and for
    /// a mapping whose list has been invalidated.
    ///
    /// Distinct from [`Self::owner_task_hint`], which is a *search* hint and is
    /// allowed to be wrong. This is a statement about the list that is in the
    /// entry right now: repeat this walk and you must get these entries back, or
    /// the guest has moved the surface underneath us without saying so.
    pub backing_walk: Option<BackingWalk>,
}

impl MappingEntry {
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
        self.device_desc
            .get(..crate::protocol::iosurface_pages::DEVICE_DESC_LEN)
    }
}

/// Exact protocol-backed compute storage-image view eligible for residency.
///
/// `map_generation` separates recycled mapping lifetimes. The remaining fields
/// distinguish Metal texture views over one IOSurface; equal mapping ids alone
/// are not enough when formats or plane windows differ.
///
/// Three window kinds share this shape (`texture_ref` appended last so the
/// `(mapping_id, …)` ordering prefix — and every mapping-keyed range scan —
/// is unchanged):
/// - **Surface window** (`mapping_id != 0`): a mapper-ref-texture IOSurface view;
///   `texture_ref == 0`.
/// - **Linear window** (`mapping_id == 0`): a normal-texture raw task-GVA texture,
///   identity-matched to its `host_linear_textures` cache entry —
///   `map_generation` holds the task id, `surface_offset` the level-0 GVA,
///   `surface_bpr` the row stride, `span_end` `row_stride * height`, and
///   `texture_ref` the object-list ref. Mapping-keyed scans never see these
///   (real mapping ids are nonzero).
/// - **Heap texture** (`mapping_id == 0`, `surface_offset == 0`): a host-only
///   opcode-0x15 texture. `map_generation` holds the task id and `texture_ref`
///   the heap-texture object ref. It has no guest GVA to flush or restage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComputeStorageResidencyKey {
    pub mapping_id: u32,
    pub map_generation: u32,
    pub surface_offset: u64,
    pub surface_bpr: u32,
    pub span_end: u64,
    pub width: u32,
    pub height: u32,
    pub pixel_format: u16,
    pub texture_ref: u32,
}

impl ComputeStorageResidencyKey {
    /// Identity of a linear (normal-texture raw task-GVA) texture window.
    #[allow(
        clippy::too_many_arguments,
        reason = "the key constructor names every wire-derived identity component"
    )]
    pub fn linear(
        task_id: u32,
        texture_ref: u32,
        gva: u64,
        row_stride: u32,
        span_end: u64,
        width: u32,
        height: u32,
        pixel_format: u16,
    ) -> Self {
        Self {
            mapping_id: 0,
            map_generation: task_id,
            surface_offset: gva,
            surface_bpr: row_stride,
            span_end,
            width,
            height,
            pixel_format,
            texture_ref,
        }
    }

    /// Identity of a host-only opcode-0x15 heap texture.
    pub fn heap(
        task_id: u32,
        texture_ref: u32,
        width: u32,
        height: u32,
        pixel_format: u16,
    ) -> Self {
        Self {
            mapping_id: 0,
            map_generation: task_id,
            surface_offset: 0,
            surface_bpr: 0,
            span_end: 0,
            width,
            height,
            pixel_format,
            texture_ref,
        }
    }

    /// True for a linear task-GVA window (see the struct doc).
    pub fn is_linear(&self) -> bool {
        self.mapping_id == 0 && self.surface_offset != 0
    }

    /// True for a host-only opcode-0x15 heap texture.
    pub fn is_heap(&self) -> bool {
        self.mapping_id == 0 && self.surface_offset == 0
    }
}

/// Why a present is not backed by guest work, as reported by
/// [`DeviceState::note_present_backing`].
///
/// Two distinct findings, and the callee names which so the caller cannot supply
/// the word. Both are statements about **decoded Store bookkeeping only** —
/// `dense_frame_seq`, advanced when a Store's pixels reached the mapping's guest
/// pages. Neither says what the viewer sees, and that limit is the point: on the
/// resident rail a Store renders into the registry without writing guest pages,
/// so a mapping can be "unbacked" here while a perfectly good resident carries
/// its present. What the viewer sees takes the carrier reading the emission site
/// pairs with this (`resident_presentable`), never this value alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentBacking {
    /// Presented again with no full-frame Store naming this mapping since its
    /// own previous present. Carries the unchanged `dense_frame_seq`.
    Restaled { seq: u64 },
    /// First present since this mapping was created, and no full-frame Store has
    /// ever named it.
    NeverStored,
}

impl crate::observe::Decline for PresentBacking {
    fn slug(&self) -> &'static str {
        match self {
            Self::Restaled { .. } => "present_backing_restaled",
            Self::NeverStored => "present_backing_never_stored",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            // The seq the witness did NOT advance past, which is what makes a
            // restale readable: two presents quoting the same number are the
            // same guest frame shown twice.
            Self::Restaled { seq } => vec![("since_seq", seq.to_string())],
            Self::NeverStored => Vec::new(),
        }
    }
}

/// HostOps view over a **task GVA range** (MapMemory2 / UnmapMemory lifecycle).
///
/// Distinct from [`MappingEntry::contig_ptr`] (iosfc `mapping_id` page list).
/// Created on demand via [`crate::runtime::gva_view::ensure_gva_view`]; torn
/// down on overlapping UnmapMemory / MapMemory2 / delete_task so we never keep
/// a host alias after the guest drops the GPU page-table mapping (Apple
/// `unmapMemory` analogue). Does **not** own discrete encode content
/// (`host_gva_surfaces`) — that cache is retained across Unmap (wallpaper class).
#[derive(Clone, Debug, Default)]
pub struct GvaHostView {
    /// Task slot the walk used when the view was built (resolved active id).
    pub task_id: u32,
    /// Guest VA base of the registered span (not necessarily page-aligned).
    pub gva: u64,
    /// Byte length of the registered GVA span.
    pub length: u64,
    /// Host pointer from [`crate::runtime::host::HostOps::map_pages`].
    pub ptr: usize,
    /// Host view length in bytes (`gpas.len() * page_size`).
    pub ptr_len: usize,
    /// Leaf GPA of the view's first page at build time.
    ///
    /// A registered view is always ONE contiguous run of guest frames —
    /// `ensure_gva_view` refuses a fragmented span before mapping it — so this
    /// plus `ptr_len` is the whole GPA list, and the reuse verify re-walks the
    /// span and compares every page against it. `0` = unverifiable (fixtures),
    /// skip.
    pub first_gpa: u64,
}

/// Which guest pages a GVA-keyed encode was stored against.
///
/// [`DeviceState::host_gva_surfaces`] is keyed by guest **virtual** address, and
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
    /// Decoded object type that produced a GVA-keyed normal-texture encode. Zero for
    /// surface/ref caches and for stores that did not record an owner.
    pub producer_object_type: u8,
    /// Recency stamp for the GVA cache's byte cap
    /// ([`GVA_ENCODE_CACHE_BYTE_CAP`]), from
    /// [`DeviceState::next_gva_touch`]. Bumped on store **and on every
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
    // No guest-CPU-write witness sits here, and that is a known gap rather
    // than an omission. `surface_cache::gva_backing_state` answers whether this
    // GVA still *names* these pages; nothing answers whether the guest CPU
    // *wrote* them.
    // A guest store into pages that never moved produces no notify, no verdict
    // and no device operation, so this entry can keep serving bytes the guest
    // has already replaced.
    //
    // A `track_guest_writes` token used to sit here for exactly that. It could
    // never answer: its baseline was latched immediately after the token was
    // registered, inside the dirty tracker's two-harvest startup window where a
    // generation reads 0, and was re-latched only by a later store to the same
    // address. The entries this cache exists for are stored once and sampled
    // forever, so their baseline stayed 0 for the boot. Over five boots the
    // comparison it existed to make ran zero times. Anything reinstating it has
    // to fix that first: re-read the baseline until it is non-zero, the way
    // `mapper::stamp_guest_write_gen` gets it right on the mapping rail by
    // re-stamping on every write.
}

/// Raw normal texture content retained by the discrete backend.
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

/// Present / scanout model state.
#[derive(Clone, Debug, Default)]
pub struct PresentState {
    pub valid: bool,
    pub width: u32,
    pub height: u32,
    /// Content generation observed at last DisplaySwap enqueue.
    pub generation: u32,
    /// A host-owned presentation window is live (device_drain refreshes this
    /// from the window link each tranche). When false the QEMU console is the
    /// display: every present must enqueue a CPU `ScanoutUpdate` and the
    /// present-completion ack belongs to the console paint
    /// (`device_scanout_copy`), never the drain tail.
    pub window_active: bool,
    /// Mapping id of the last successful console paint (0 = never).
    /// Paired with `painted_generation` so dual-mid DisplaySwap cannot
    /// Unchanged-skip when both mids share the same generation counter.
    pub painted_mapping: u32,
    /// Content generation of the last successful paint (skip if matches).
    pub painted_generation: u32,
    pub present_mapping: u32,
    pub host_mapping: u32,
    pub frame_flush_seen: bool,
    /// Latest mapper-ref-texture **Composite** writeback mid (logo/desktop content).
    /// Pre-boundary: sticky early feed for gfx_update when present_mapping is a
    /// ClearOnly flip buffer (dual-mid buffer-setup thrash class).
    /// Post-boundary: dual-mid *peer* tracker, read only by the failure/census
    /// lines (`front_wb`, `present_order_hold`) — x86 present often names
    /// ClearOnly mid 2/3 while Stores land on Composite mid 1/4/5, and naming
    /// the peer there is what makes that split visible in a boot log.
    pub early_front_mapping: u32,
    /// Present/scanout evidence: mapping → latest geometry it was displayed
    /// at (a `capture_present_frame` action or a retained-frame re-show). The
    /// decoded display transaction naming this surface as plane 0 is the only
    /// thing that writes it, so it separates a scanout buffer from a sampled
    /// sub-surface (a WebKit content tile publishes full frames every paint and
    /// is never presented).
    /// Protocol-structural dense-frame tracking (measure-only, never gates a
    /// present decision): per mapping id, the value of
    /// [`Self::dense_frame_counter`] at the last full-frame (whole-`w`×`h`)
    /// Store **naming that mapping id** — the completeness proof in
    /// [`DeviceState::note_dense_frame_published`], which is the only site that
    /// advances it. Read only by [`DeviceState::note_present_backing`], the
    /// `present_unbacked` gate. Cleared on unmap.
    ///
    /// **What this is keyed on, and what that means it cannot see.** The advance
    /// is a function of the mapping id the Store named and nothing else; it
    /// consults no resident handle. So a full frame the guest sent for a
    /// surface, whose draws were routed to a *different* resident than the one
    /// that surface's present will read, still advances the seq — the gate below
    /// is structurally blind to that. It is also keyed per mapping
    /// id while unified surfaces share ONE resident, so a full frame stored
    /// through one of them does not mark its siblings backed even though they
    /// hold the same pixels.
    pub dense_frame_seq: BTreeMap<u32, u64>,
    /// Per mapping id: the [`Self::dense_frame_seq`] value that mapping held
    /// the last time it was PRESENTED.
    ///
    /// A surface whose seq is unchanged across two of its own presents received
    /// no full-frame Store naming it in between. That is the always-on
    /// `present_unbacked` gate — the loss itself, reported on the mid the guest
    /// named, rather than a rate at which we papered over it. Keyed per mapping
    /// id (not globally) so healthy a/b alternation, where each buffer
    /// legitimately advances on its own turn, stays quiet. Cleared on unmap.
    ///
    /// The "or an inter-buffer seed" half of this condition is gone: `62587b1`
    /// deleted the a/b peer front seed, because unified members share one
    /// resident and a seed between them is a copy onto itself. Nothing else
    /// advances [`Self::dense_frame_counter`].
    pub presented_dense_seq: BTreeMap<u32, u64>,
    /// Monotonic source for [`Self::dense_frame_seq`] (one bump per full-frame
    /// Store). Never reset except on device reset.
    pub dense_frame_counter: u64,
    /// Monotonic present counter, advanced exactly once per present cycle at the
    /// present boundary ([`DeviceState::advance_present_epoch`]). Its only
    /// consumer is the macOS window-publish dedup key, which includes it so that
    /// every present republishes the frame even when the mapping id and resource
    /// generation repeat (an in-place update of the same resident). Never reset
    /// except on device reset.
    pub present_epoch: u64,
    /// Latest presentFrame retain (PGDisplay +0x188) — most recent DisplaySwap.
    /// Tight packed BGRA8, stride = `frame_width * 4`.
    pub frame_bgra: Vec<u8>,
    pub frame_mapping: u32,
    pub frame_width: u32,
    pub frame_height: u32,
    pub frame_generation: u32,
    /// `MappingEntry::surface_content_epoch` of the captured frame — "these are
    /// different pixels", where `frame_generation` is "the guest's pages hold
    /// something different".
    ///
    /// The two came apart when the lazy mapper-ref-texture Store
    /// ([`crate::runtime::writeback_debt`]) started leaving a frame in the engine
    /// resident and owing the pages a copy: the pixels move every frame and the
    /// generation does not. Anything asking "is this a new frame to show" has to
    /// read this one — `device::window_publish::window_frame_key` is the caller
    /// that found out the hard way, discarding 20 % of a driven boot's frames as
    /// unchanged.
    pub frame_content_epoch: u32,
    pub frame_valid: bool,
    /// True only when DisplaySwap capture failed; first host paint retries.
    pub frame_encode_pending: bool,
    /// DisplaySwaps accepted since the last host paint of +0x188.
    ///
    /// apple-gfx `pending_frames` / PGDisplay `waitForPendingFrames` entry gate:
    /// when this is ≥ [`crate::runtime::drain::MAX_UNPAINTED_PRESENTS`], the
    /// drain **declines to run** the released CmdDisplaySwap until paint clears
    /// the count. Accepted presents still stamp at retain.
    pub unpainted_presents: u32,
    /// Suppress repeated fail-log lines while the same present packet remains
    /// held at the pending-frames gate.
    pub backpressure_hold_active: bool,
    pub backpressure_hold_channel: u32,
    /// Which present is being held, as its ordering position.
    ///
    /// A position and not a ring head: a held present no longer sits at a
    /// consumer pointer — the head moved past it at arrival — so the thing that
    /// distinguishes one hold episode from the next is the transaction, which is
    /// what the guest is actually waiting on.
    pub backpressure_hold_position: u64,
    /// Always-on diagnostic counter for distinct pending-frames hold episodes.
    pub backpressure_hold_count: u64,
    /// Recycled scratch for the present-capture frame buffer.
    ///
    /// `capture_present_frame` previously did `vec![0u8; need]` on **every**
    /// present — a fresh 8 MiB allocation that is zeroed and then fully
    /// overwritten, faulting in fresh anon pages each time (a large part of the
    /// per-present `paint_us`). Instead the capture takes this warm buffer,
    /// resizes (no realloc at steady geometry), fills it, and on success swaps
    /// the **old** `frame_bgra` back in here — so exactly two 8 MiB buffers
    /// cycle forever with no per-present malloc/zero/fault. On capture failure
    /// the buffer is returned here unchanged so the prior `frame_bgra` retain is
    /// untouched (keep-prior contract). Serialized with the console paint by the
    /// device lock; never read as content.
    pub capture_scratch: Vec<u8>,
    /// True when the previous present's window publish handed the window a GPU
    /// resident rather than CPU pixels — the macOS engine-swapchain handoff, which
    /// presents the compositor's resident through the engine's own MoltenVK
    /// swapchain and never reads `frame_bgra`. Set by `publish_window_frame` each
    /// present (same drain worker, one present after the capture reads it; the
    /// handoff is stable across steady-state presents). When true,
    /// `capture_present_frame` skips the expensive guest-page readback.
    ///
    /// Always false where the window owns its own swapchain and uploads CPU pixels
    /// — every non-macOS host — so those keep the per-present readback unchanged.
    pub display_from_resident: bool,
    /// Always-on census: full (readback ran) vs light (resident-carried, readback
    /// skipped) captures, so the readback-elision ratio is visible.
    pub full_captures: u64,
    pub light_captures: u64,
}

/// Hardware cursor model.
#[derive(Clone, Debug, Default)]
pub struct CursorState {
    pub show: bool,
    pub x: u16,
    pub y: u16,
    pub width: u16,
    pub height: u16,
    pub hot_x: u16,
    pub hot_y: u16,
    /// QEMUCursor pixels as 0xAARRGGBB (guest BGRA reordered).
    pub pixels: Vec<u32>,
    /// True when `pixels` holds a complete glyph for the host console.
    pub glyph_ready: bool,
}

/// Display shared-state handshake (archive setupSharedState + online poll).
#[derive(Clone, Debug, Default)]
pub struct DisplayHandshake {
    pub shared_gpa: u64,
    pub display_index: u32,
    pub online_acked: bool,
    pub online_tries: u32,
    /// Cadence counter for ONLINE re-drive (archive display_poll_ctr).
    pub poll_ctr: u32,
    /// Samples already logged per observed display-transaction wire shape,
    /// keyed by `(opcode, payload_len, pipe_index, task_field_is_set)`.
    ///
    /// Backs the `display_txn_payload` measurement. A live x86 session showed the
    /// payload is trailer-only and its length never varies, so keying on length
    /// alone spent the whole budget inside the first 400ms of display activity
    /// and stayed silent afterwards. The remaining trailer words are what still
    /// carry news: `pipe_index` changes when a second display pipe appears, and
    /// the task field is zero through early bring-up, so its first non-zero value
    /// re-arms the probe exactly once at the transition into steady-state
    /// compositing.
    ///
    /// Keyed on `(opcode, payload_len)`: the alarm is that a command grew past
    /// the size its own contract declares, and a guest that grew it grew it for
    /// every frame, so one line per distinct shape is the whole signal.
    pub txn_payload_samples: BTreeSet<(u16, usize)>,
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

/// Pending drain flags (MMIO path only sets bits; drain consumes).
#[derive(Clone, Debug, Default)]
pub struct PendingWork {
    pub main_drain: bool,
    pub child_mask: u32,
    pub iosfc: bool,
    /// A present queued a host scanout action. The ordered worker must return
    /// before consuming more guest work so QEMU can apply that action without
    /// blocking on the device lock. Cleared when the action is consumed.
    pub host_action_yield: bool,
}

/// Byte cap for the guest-CPU-produced content memos (`guest_linear_memo`,
/// `ref_texture_view_memo`, `mapper_ref_texture_memo`). A cap crossing evicts the coldest entries
/// down to a low-water mark — never a bulk clear — so the hot working set (and
/// its avoided re-decode/re-convert cost) survives.
pub const GUEST_LINEAR_MEMO_BYTE_CAP: usize = 128 << 20;

/// Byte cap for the GVA-keyed normal-texture encode cache
/// ([`DeviceState::host_gva_surfaces`]). Same basis and same value as
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
/// [`DeviceState::host_gva_surfaces`], so its cost must be countable rather
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

/// See [`DeviceState::guest_linear_memo`].
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
    pub layout: crate::protocol::pixel_format::TexelLayout,
    /// Content generation: bumps only when the native bytes change.
    pub generation: u64,
}

/// The mapping ids one object reference names, as
/// [`DeviceState::mappings_named_by`] resolves them.
///
/// Two at most and that is a property of the contract rather than a capacity
/// this device chose: a reference is its own mapping id or it is not, and the
/// per-task registration holds exactly one entry per `(task, ref)`. Carrying
/// the pair inline rather than in a `Vec` is what makes that statement, and it
/// is why `push` past the second is unreachable rather than a bound to tune.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NamedMappings {
    ids: [u32; 2],
    len: u8,
}

impl NamedMappings {
    /// Add `id` unless it is already named. Silently complete at two, which the
    /// type's doc explains is unreachable rather than a truncation.
    fn push(&mut self, id: u32) {
        if self.iter().any(|held| held == id) {
            return;
        }
        if let Some(slot) = self.ids.get_mut(self.len as usize) {
            *slot = id;
            self.len += 1;
        }
    }

    /// The named ids, reference first.
    pub fn iter(self) -> impl Iterator<Item = u32> {
        (0..usize::from(self.len)).map(move |i| self.ids[i])
    }

    /// Whether this reference named no mapping at all.
    pub fn is_empty(self) -> bool {
        self.len == 0
    }
}

/// A name the lifecycle owner published, and what publishing it obliged.
///
/// Two things because a declaration is two things. The name is what the caller
/// asked for; the effects are the previous occupant's — a guest that writes
/// over a live object-list slot ends the object that was there with no delete
/// packet at all, and the teardown that owes is carried out here or by nobody.
#[must_use = "a declaration's effects are the displaced occupant's teardown"]
pub struct Declaration {
    pub id: ResourceId,
    pub acted: Acted,
}

/// The obligations of a lifecycle operation that this device *acts on* rather
/// than counts.
///
/// [`note_lifecycle_effects`] is the one place `reims_vgpu_core::lifecycle::Effects`
/// is opened, and it splits the six into two kinds. Four of them name work only
/// this device can do — it holds the host textures a teardown frees, the
/// resolution caches a remap invalidates, the transfer stagings a discard
/// releases, and the per-task caches a redefinition ends — so they arrive here
/// as values. The other two are counted there and are not in this type, because
/// there is nothing on this device for them to drive; see that function.
#[must_use = "every field here is work the model has already decided and nothing else will do"]
pub struct Acted {
    pub teardowns: Vec<reims_vgpu_core::namespace::Teardown>,
    pub remapped: Vec<reims_vgpu_core::lifecycle::Remap>,
    pub at_completion: Vec<reims_vgpu_core::lifecycle::DeferredDiscard>,
    pub redefined: Vec<reims_vgpu_core::lifecycle::Redefinition>,
}

/// Open one lifecycle operation's effects, count what this device does not act
/// on, and hand back what it does.
///
/// **The one destructure, so a seventh obligation is a compile error.** Every
/// door and every lifetime arm reaches the owner's `Effects` through here; a
/// caller that named three fields by hand would be correct only for as long as
/// the operation it calls fills exactly those three, which is not a fact a call
/// site can see, and the failure is an obligation that was owed and then
/// dropped on the way out.
///
/// # The two that are counted here and not returned
///
/// * **`storage_freed`** is heap storage whose last allocation went with the
///   operation. This device declares no heap placement at all —
///   `crate::runtime::objects::declared_storage` refuses one by name, because a
///   heap's extent is unrecovered — so the owner's per-task `Heaps` is empty and
///   this list cannot be non-empty. Counted rather than asserted: a reading
///   above zero is the day the heap extent landed and this became work.
/// * **`transfers`** are copies owed before a completion stamp may publish, and
///   they are produced only where a replica is behind. Nothing in this device
///   calls `Lifecycle::record_write`, so `Replica::DeviceOwned` never becomes
///   authoritative over any byte, so no read of the guest's pages is ever
///   behind and no transfer can be owed. The device's own deferred-Store
///   obligation is `crate::runtime::writeback_debt`'s and is a different fact —
///   it is about a render pass that has already run, not about content
///   authority the model tracks. A reading above zero here means the content
///   ledger has started deciding something, and that is a change worth a line
///   on the always-on channel rather than a silent one.
fn note_lifecycle_effects(
    effects: reims_vgpu_core::lifecycle::Effects,
    site: &'static str,
) -> Acted {
    let reims_vgpu_core::lifecycle::Effects {
        transfers,
        teardowns,
        storage_freed,
        at_completion,
        remapped,
        redefined,
    } = effects;
    if !storage_freed.is_empty() {
        crate::runtime::drain::note_store_route_n(
            "lifecycle_storage_freed",
            storage_freed.len() as u64,
        );
        crate::observe::fail(format!(
            "lifecycle_storage_freed site={site} n={} (this device places nothing in a heap, so \
             the owner's heaps hold no storage to free — a heap extent has been recovered \
             somewhere and this is now work)",
            storage_freed.len()
        ));
    }
    if !transfers.is_empty() {
        crate::runtime::drain::note_store_route_n("lifecycle_transfers", transfers.len() as u64);
        crate::observe::fail(format!(
            "lifecycle_transfers site={site} n={} (nothing calls Lifecycle::record_write, so no \
             replica can be behind — the content ledger has started deciding transfers and \
             nothing here executes them)",
            transfers.len()
        ));
    }
    Acted {
        teardowns,
        remapped,
        at_completion,
        redefined,
    }
}

/// Full device model state (backend-independent).
#[derive(Debug)]
pub struct DeviceState {
    pub id: DeviceId,
    /// Guest page shift for PFN↔GPA wire math (12 = x86, 14 = arm64e).
    pub page_shift: u32,
    pub gfx: GfxRegs,
    pub iosfc: IosfcRegs,
    /// The semantic model's ordering plane, and the owner of which child
    /// domains are open.
    ///
    /// **This is the replacement architecture's `SessionModel`, in production,
    /// holding one lifetime.** `active_child_mask` used to be a stored bit per
    /// child domain, written by three events — a channel definition, a locked
    /// doorbell register write, and a lock-free doorbell ring — and read for two
    /// unrelated questions: "is this a publication domain the guest defined" and
    /// "is there work waiting here". One field answering both is why neither
    /// could be moved.
    ///
    /// The two are separated now. Openness is this model's `open_channels`,
    /// reached through [`Self::child_domain_open`] and
    /// [`Self::open_child_mask`]; "there is work here" is `pending.child_mask`,
    /// which the doorbells write. The union the old field held is
    /// [`Self::drainable_child_mask`], derived rather than stored, so the two
    /// cannot drift.
    ///
    /// Only channel lifetime feeds it, and that is deliberate rather than
    /// partial: `SessionModel::apply_control` performs a control operation's
    /// effect, and a channel definition's whole effect is opening the domain its
    /// next packet names. The packet *envelope* — stamps, ordering position,
    /// completion — stays with the drain for every class and moves when the
    /// ordering core does. What has moved is one lifetime, to its one owner.
    ///
    /// **Private, and reached through the doors below.** The same shape the
    /// resource-lifetime group left `lifecycle` in, for the same reason: the
    /// paths that will feed this model next — a pipeline being declared,
    /// translated, built or refused — run on the draw rails, which hold
    /// `&DeviceState` and cannot take a `&mut` field. A `Mutex` and named doors
    /// keep "which state changes, from where" a list a reader can enumerate,
    /// instead of a public field any call site can reach into.
    session: Mutex<reims_vgpu_core::session::SessionModel>,
    /// The bytes every admitted position is executed from, keyed by the
    /// ordinal [`Self::admit_packet`] issued.
    ///
    /// Beside the ordering plane rather than inside it: the model holds
    /// ordering and this device holds bytes, and the ordinal is the join. Not
    /// behind the `Mutex` above, because the drain owns `&mut DeviceState` and
    /// nothing on a draw rail parks or runs work — see
    /// [`crate::runtime::parked::ParkedStore`], which owns the identity of a
    /// parked position and deliberately not its readiness.
    pub parked: crate::runtime::parked::ParkedStore,
    /// Child channels whose head `EXEC_INDIRECT2` packet is held while an
    /// immutable AIR translation is still loading. The packet head and stamp
    /// remain untouched until retry, so this is scheduler state rather than a
    /// submitted async GPU job.
    pub translation_deferred_mask: u32,
    /// Root/child FIFO timelines held behind a cold-translation EXEC. Bit 0 is
    /// the root FIFO; child channel N uses bit N. This is diagnostic scheduler
    /// ownership, not a guest-visible protocol mask.
    pub translation_order_hold_mask: u32,
    /// Distinct cross-FIFO hold episodes (retries of one episode do not grow it).
    pub translation_order_holds: u64,
    /// Display transactions held while another channel remained blocked on
    /// translation after the transaction's rescue drains. This counts hold
    /// episodes, not poll retries of the same packet.
    pub present_translation_holds: u64,
    /// Display channels whose FIFO head is already held for
    /// `translation_deferred_mask`. Suppresses fail-log flooding while the
    /// same head is retried and is cleared with channel lifecycle state.
    pub present_translation_hold_mask: u32,
    pub pending: PendingWork,
    pub child_rings: [ChannelRing; MAX_CHANNELS],
    pub tasks: TaskTable,
    /// Highest task id the guest has ever named.
    ///
    /// # It measured a bound, and the bound is gone
    ///
    /// This and [`Self::max_mapping_id_seen`] were added as reach censuses for
    /// `MAX_TASKS` (256) and `MAX_MAPPINGS` (4096), because a refusal counter
    /// alone cannot say whether a bound is close: a boot stopping at id 12 and
    /// one stopping at 255 both report zero refusals, and only one of them says
    /// there is room. What they measured was 25x and 97x of headroom, and
    /// headroom is not a derivation — so both bounds were removed rather than
    /// defended, and [`TaskTable`] and `mappings` are maps keyed by the guest's
    /// own `u32`.
    ///
    /// So these are **occupancy readings** now, not distances to a refusal.
    /// They are the only thing that says how far the guest spreads either id
    /// space, which is worth publishing for a map with no removal path — but
    /// there is no cap to read them against, and the census prints `none` where
    /// it used to print one. Nothing should turn either back into a bound.
    pub max_task_id_seen: u32,
    /// See [`Self::max_task_id_seen`].
    pub max_mapping_id_seen: u32,
    /// Count of MapMemory2/UnmapMemory packets (measure census).
    pub map_family_events: u64,
    /// Per-task live guest-VA mappings, for the map/unmap pairing audit.
    ///
    /// Observation only — see [`crate::runtime::map_audit`] for what it watches
    /// and why the wire is entitled to answer it. Keyed separately from
    /// [`Self::tasks`] because a map packet may name a task id this device has
    /// no entry for, and that case is itself worth counting rather than
    /// dropping.
    pub map_audit: std::collections::BTreeMap<u32, crate::runtime::map_audit::MapIntervals>,
    /// Per-task page-table node pages, for the host-write guard.
    ///
    /// Observation only — see [`crate::runtime::node_guard`]. Keyed and dropped
    /// exactly as [`Self::map_audit`] is, and for the same reason: these pages
    /// belong to the task's address space, so a reused id inheriting them would
    /// be watching memory that is now somebody else's.
    pub node_guard: std::collections::BTreeMap<u32, crate::runtime::node_guard::NodeWatch>,
    /// Live object refs per task, as `(task_id, ref)`.
    ///
    /// This is membership for host-copy teardown. [`Self::task_resources`]
    /// owns the corresponding resource objects and their immutable descriptor
    /// construction input.
    pub objects: std::collections::BTreeSet<(u32, u32)>,
    /// Retained resource objects, keyed by the name
    /// [`Self::declare_object`] issued. See [`TaskResources`].
    pub task_resources: TaskResources,
    /// The object namespace of each live task: what a guest reference resolves
    /// to, and at which generation.
    ///
    /// **The authority for object naming, and the only issuer of a generation.**
    /// Everything else this device keeps per object is a memo behind a name this
    /// answered — [`TaskResources`] most of all, which is keyed by that name so
    /// a retired one cannot be spelled.
    ///
    /// One namespace per task, because an object-list reference is task-local
    /// and a resolution that reached across tasks would find whatever shared the
    /// integer. Dropped whole when the task's address space ends, which is the
    /// same event that ends every name in it.
    ///
    /// Behind a lock for [`TaskResources`]' reason: the resolve path holds
    /// [`DeviceState`] shared, and `reims_vgpu_core::namespace::Namespace`
    /// counts its own resolutions, so even a lookup takes `&mut`.
    ///
    /// **The semantic model's lifecycle owner, not a map of namespaces.** This
    /// field held `BTreeMap<u32, Namespace>` until the resource-lifecycle group
    /// moved: one namespace per task, and nothing owning the events that begin
    /// or end them. `reims_vgpu_core::lifecycle::Lifecycle` owns the
    /// namespaces, the per-task heaps and the session's content authority
    /// together, because the twelve lifetime commands move all three — a device
    /// that took only the first would be keeping the second record of the other
    /// two, which is the thing the replacement plan forbids.
    ///
    /// **Nothing outside the doors below names this field**, and that is the
    /// group's disjointness claim in its structural form: the legacy path
    /// cannot reach this state except through
    /// [`Self::object_name`], [`Self::declare_object`],
    /// [`Self::retire_object_name`], [`Self::delete_task_namespace`] and the
    /// two task-lifetime entry points, all of which moved in one commit.
    lifecycle: Mutex<reims_vgpu_core::lifecycle::Lifecycle>,
    /// Immutable sampler objects in the sampler API's separate ref space.
    pub task_sampler_states: TaskSamplerStates,
    /// Immutable depth-stencil objects in that API's separate ref space.
    pub task_depth_stencil_states: TaskDepthStencilStates,
    /// What the running rail retains for exactly this device's lifetime, in
    /// the rail's own vocabulary. See [`RailDeviceState`].
    ///
    /// The model owns the slot and the drop; it does not own — and cannot name
    /// — the contents. `OnceLock` rather than a lock, because the slot is
    /// claimed once by the one rail this process latched and never replaced, so
    /// a read on the draw path is an acquire load and not a mutex.
    rail: OnceLock<Box<dyn RailDeviceState>>,
    /// Mapper-ref-texture object ref → mapping_id: (task_id, ref) -> mapping_id.
    pub texture_to_mapping: BTreeMap<(u32, u32), u32>,
    /// Per-window half of [`StorageIncarnation`], keyed `(task, window base)`.
    /// Reset for a task whenever its epoch moves, which is what keeps it
    /// bounded by the live namespace.
    storage_incarnations: BTreeMap<(u32, u64), u32>,
    /// Per-task half of [`StorageIncarnation`].
    task_storage_epochs: BTreeMap<u32, u32>,
    /// The first object reference seen naming each guest-VA window, per task.
    ///
    /// Keyed `(task, window base)`, holding the reference that got there first.
    /// It answers one question and it is the question a canonical backing
    /// identity is blocked on: **can two references in one task name one piece
    /// of storage?**
    ///
    /// `BackingId`'s settled derivation is the window plus a *per-reference*
    /// incarnation, and `replace_physical` advances that count for the
    /// reference the packet names and no other. If two live references ever
    /// share a window, the two would then carry different incarnations for the
    /// same bytes — two ids for one backing, which is a hazard edge the
    /// dependency compiler never draws, which is a data race. If no two ever
    /// do, the per-reference count *is* canonical and the derivation stands.
    ///
    /// Reset with the task, alongside [`Self::storage_incarnations`], and for
    /// the same reason: an entry outliving its namespace would answer for a
    /// window nothing names any more. A stale entry inside a live task is
    /// possible — a deleted reference leaves its window behind — so the reader
    /// re-checks that the claimant is still live before calling it an alias.
    backing_window_refs: BackingWindowRefs,
    /// References found to share their allocation with another live one.
    ///
    /// A lookup table for hot paths that must not pay for the scan that fills
    /// it. See [`Self::note_aliased_reference`] for what its freshness is.
    /// Cleared with the task, alongside every other per-window fact.
    aliased_references: Mutex<BTreeSet<(u32, u32)>>,
    pub mappings: BTreeMap<u32, MappingEntry>,
    /// Host render-cache keyed by surface_id / mapping_id (Linux/Vulkan rail).
    /// See [`crate::runtime::surface_cache`] and kb tahoe-x86-host-reims_vgpu §8.5.
    /// **Surface_id namespace only** — never texture_ref (object list ids collide).
    pub host_surfaces: BTreeMap<u32, HostSurface>,
    /// Discrete encode cache for normal-texture GVA color targets, keyed by
    /// `(task_id, texture_ref)`.
    ///
    /// Object-list refs are local to a task. Separate from
    /// [`Self::host_surfaces`] so list ids cannot clobber backing present mids,
    /// and task-qualified so one address space cannot replace or evict another
    /// task's same-numbered texture.
    pub host_texture_surfaces: BTreeMap<(u32, u32), HostSurface>,
    /// Same normal-texture encode content keyed by target GVA — survives texture_ref
    /// rebinding / small-atlas overwrite of the ref slot.
    ///
    /// Bounded by [`GVA_ENCODE_CACHE_BYTE_CAP`] with least-recently-*used*
    /// eviction; see that constant for why recency and not staleness. Growth is
    /// entirely from new GVAs — a store at an existing key replaces in place.
    pub host_gva_surfaces: BTreeMap<u64, HostSurface>,
    /// Monotonic recency counter behind [`HostSurface::last_touch`].
    pub gva_touch_seq: u64,
    /// Monotonic ordering counter behind [`ResourceValidity::host_cleared_seq`]
    /// and `host_published_seq`. See [`Self::next_validity_seq`].
    pub validity_seq: u64,
    /// Running sum of `host_gva_surfaces[*].bgra.len()`, so the byte cap can be
    /// tested without an O(n) pass over the map on every store.
    ///
    /// The same running total [`crate::model::LruBytesMemo`] keeps, for the same
    /// reason: enforcement runs on the store path, which is the draw path, and
    /// re-summing a map the cap allows to hold thousands of small entries would
    /// put that walk in front of every encode.
    ///
    /// Maintained at exactly the two sites that change a byte count —
    /// `store_gva_owned` and `evict_gva`; the other `get_mut` reachers touch
    /// backing, tokens and recency, never `bgra`. Because a running total is a
    /// second source of truth, the per-second census recomputes the real sum it
    /// was already computing for `gva_bytes` and reports the difference as
    /// `gva_cap_drift`: a nonzero value means a new mutation site was added
    /// without updating this, which is a bug that would otherwise be invisible
    /// until the cap silently stopped bounding anything.
    pub gva_cache_bytes: usize,
    /// The bound [`crate::runtime::surface_cache::enforce_gva_cache_cap`]
    /// holds [`Self::host_gva_surfaces`] to, always
    /// [`GVA_ENCODE_CACHE_BYTE_CAP`] in production.
    ///
    /// A field rather than the constant read directly so the eviction policy is
    /// testable: at 128 MiB a test that wanted to cross the cap would have to
    /// allocate 128 MiB of pixels, so the policy would go untested and only the
    /// arithmetic around it would not. Nothing in the device writes this.
    pub gva_cache_byte_cap: usize,
    /// What [`GVA_ENCODE_CACHE_BYTE_CAP`] cost, measured rather than assumed.
    pub gva_eviction_witness: GvaEvictionWitness,
    /// Raw compute encode for normal textures. Retained across GVA unmap;
    /// evicted on task/object lifetime end or descriptor mismatch.
    pub host_linear_textures: BTreeMap<(u32, u32), HostLinearTexture>,
    /// Perf memo for guest-CPU-produced linear textures (no host cache entry,
    /// so no producer generation exists). Coherence is re-established on
    /// every lookup by re-reading the native guest rows and comparing them
    /// byte-exact against the memoized copy — a guest write is always seen;
    /// only the swizzle+alloc (and the engine's content hash+memcmp, via the
    /// generation identity) are skipped on unchanged content. Keyed by
    /// (task_id, level-0 gva, width, height, depth planes, sample format).
    /// Byte-bounded LRU
    /// ([`GUEST_LINEAR_MEMO_BYTE_CAP`]): a cap crossing evicts the least-recently
    /// -used entries down to a low-water mark, never bulk-clearing the hot set.
    pub guest_linear_memo: LruBytesMemo<(u32, u64, u32, u32, u32, u16), GuestLinearMemo>,
    /// Whether the hypervisor's guest-write generation would be a sound "these
    /// texels did not change" key for the zero-copy sampled gathers, measured
    /// against the bytes themselves. See
    /// [`crate::runtime::gather_witness`] — it selects no behaviour.
    pub gather_witness: crate::runtime::gather_witness::GatherWitness,
    /// The GVA render targets a Store has stamped, and what the two write
    /// witnesses said at the time. The GVA half of the mapper-ref-texture witness that
    /// licenses the attachment LOAD elision — see
    /// [`crate::runtime::gva_store_witness`].
    pub gva_store_witness: crate::runtime::gva_store_witness::GvaStoreWitness,
    /// Draw-time buffer binds resolved once per reference and held. Reached
    /// only through [`DeviceState::retire_bound_buffers_for_task`] and
    /// [`DeviceState::retire_bound_buffers_in_range`] from the packet handlers,
    /// so the retirement rules live in one place rather than at each opcode.
    /// See [`crate::runtime::bound_buffers`].
    ///
    /// Ungated. It holds nothing a rail owns — a bind window is a `GuestRun`
    /// over this device's own import of guest RAM — and only one rail fills it
    /// today, which is a fact about that rail rather than about the build. Gated
    /// it needed a `not(feature)` arm at every retirement returning a fabricated
    /// zero, and on a `--backend both` binary the Metal boot compiled the arm
    /// that retires from a map that boot never filled. Empty on a rail that
    /// resolves binds per encode, which is the same answer arrived at honestly.
    pub bound_buffers: crate::runtime::bound_buffers::BoundBuffers,
    /// When the guest last declared a write to each **buffer** object.
    ///
    /// The half of the validity quad `resource_validity::apply` has nowhere to
    /// put: a buffer has no mapping, so its `content_generation` does not exist
    /// and the statement was being decoded and dropped. Ungated, because the
    /// producer is the decoder rather than a backend. See
    /// [`crate::runtime::buffer_write_gen`].
    pub buffer_write_gen: crate::runtime::buffer_write_gen::BufferWriteGens,
    /// Monotonic source for every sampled-content generation this device
    /// hands the engine. Read only through
    /// [`DeviceState::next_sampled_content_generation`].
    ///
    /// The engine's sampled cache binds a retained image on `(key, generation)`
    /// alone — no hash, no compare — so a generation that ever repeats over
    /// different bytes binds the wrong picture, silently. One counter for all
    /// producers is what makes that impossible: a value is issued once and
    /// never again, so uniqueness does not depend on any producer's entry
    /// lifetime, key space, or eviction policy.
    ///
    /// Each producer used to keep its own counter and the difference was
    /// measured, not theorised. The guest-linear and ref-texture memos shared this
    /// one and were sound; the GVA host cache incremented a *per-entry* field
    /// that restarted at 1 whenever the entry was re-created, and
    /// `evict_gva` re-creates it on every deferred GVA render Store arm. One
    /// boot's audit caught `(0xa4c000, 1)` naming two different 64x64 icons.
    pub sampled_content_gen: u64,
    /// Which guest pages this device has written, and when.
    ///
    /// The hypervisor dirty bitmap witnesses guest CPU stores and nothing else,
    /// so a host-side write into the same pages is invisible to it — a copy
    /// vouched for by "the guest did not write" can still be stale because *we*
    /// wrote. This is the record that separates the two, and it is page-exact
    /// because nothing coarser is sound: guest pages are reachable under more
    /// than one mapping id, so a per-mapping count says nothing about the pages
    /// themselves, and a device-global one invalidates a texture because an
    /// unrelated scanout was composited. Both coarser counts were built, measured
    /// and removed; [`crate::runtime::host_writes`] carries the readings.
    pub host_writes: crate::runtime::host_writes::HostWrites,
    /// Reusable native-row read buffer for the guest-linear memo path.
    pub guest_linear_scratch: Vec<u8>,
    /// Byte-exact revalidated memo for ref-texture serialized texture views
    /// (media IOSurface planes). Same contract as
    /// [`Self::guest_linear_memo`]: every bind re-reads the native plane
    /// window; conversion + upload (via the returned content identity) are
    /// skipped on unchanged bytes. Keyed by
    /// (mapping_id, plane, width, height, view pixel format). Byte-bounded LRU
    /// ([`GUEST_LINEAR_MEMO_BYTE_CAP`]).
    pub ref_texture_view_memo: LruBytesMemo<(u32, u32, u32, u32, u16), GuestLinearMemo>,
    /// Byte-exact revalidated memo for the mapper-ref-texture mapping-backed sampled path
    /// (`load_mapper_ref_texture_mapping_rgba` — small IOSurface textures below the zero-copy
    /// floor, e.g. dock icons under magnification). Same contract as
    /// [`Self::guest_linear_memo`]: every bind re-reads the native BGRA rect;
    /// the BGRA->RGBA convert + the two per-bind allocs + the engine's content
    /// hash+upload (via the returned content identity) are skipped on unchanged
    /// bytes. A dock-magnification burst re-binds the same static icons ~1000x,
    /// so this collapses the `t11_guest` CPU copies that otherwise saturate the
    /// serial drain worker (dock-hover freeze). Keyed by (mapping_id, w, h).
    /// Byte-bounded LRU ([`GUEST_LINEAR_MEMO_BYTE_CAP`]).
    pub mapper_ref_texture_memo: LruBytesMemo<(u32, u32, u32), GuestLinearMemo>,
    /// Reusable native BGRA read buffer for the mapper-ref-texture memo re-read.
    pub mapper_ref_texture_memo_scratch: Vec<u8>,
    /// Last guest-visible generation produced by a compute storage-image
    /// writeback, keyed by the exact window it was produced for.
    ///
    /// **It selects behaviour.** `compute_exec`'s texture staging reads it to
    /// decide whether the engine's resident answers a bind, and for a
    /// [`ComputeStorageResidencyKey::heap`] texture that decision has no
    /// fallback: a heap texture is host-only, so there is no guest window to
    /// re-read. An entry present but unservable is refused by name
    /// (`compute_stage_tex_heap_resident_lost`); an entry *absent* stages a
    /// zero-filled texture, which is why what may remove one matters.
    ///
    /// This doc used to say "measurement-only … does not select engine
    /// behavior", which was true of an earlier rail and invites exactly the two
    /// wrong conclusions: that a bound on it is free, and that it can be cut.
    ///
    /// # What may remove an entry, and why the cap cannot reach the two that
    /// # have no fallback
    ///
    /// Three keyings share this map, and only one of them is subject to the
    /// per-mapping population cap in `compute_exec`:
    ///
    /// - **Mapping-backed** (`mapping_id != 0`) — the only kind the cap's
    ///   sibling walk can select, because it filters on an equal `mapping_id`.
    ///   Dropping one costs the next read its resident and sends it back to the
    ///   mapping's guest pages, which is a cost and not a loss.
    /// - **Linear** ([`ComputeStorageResidencyKey::linear`]) — `mapping_id` is
    ///   0, and `note_storage_residency_writeback` returns before the insert:
    ///   authority for these lives in the `host_linear_textures` entry's
    ///   `resident_gen`, never here.
    /// - **Heap** ([`ComputeStorageResidencyKey::heap`]) — `mapping_id` is also
    ///   0, and that function inserts and returns *before* the cap runs.
    ///
    /// So the cap is genuinely per-mapping, and the two keyings with no guest
    /// fallback are outside it — but only because of an early return two
    /// modules away, not because the filter distinguishes them. Both set
    /// `mapping_id` to 0, so they would share one bucket if the eviction ever
    /// saw them. An audit read the filter alone and concluded heap textures
    /// were being evicted into zero-filled binds; that is wrong today and would
    /// be right the moment a caller reached the cap with a zero-keyed
    /// candidate. Anything that changes when the cap runs must re-check this.
    pub compute_storage_residency: BTreeMap<ComputeStorageResidencyKey, u32>,
    /// Mapping ids the fence-bound writeback has landed a render window on,
    /// for one measurement and nothing else: does the guest declare its CPU
    /// reads on the same surfaces this device writes back eagerly?
    ///
    /// That question gates whether the writeback could become demand-driven,
    /// and the `guest_read_dry` count alone cannot answer it — the fence always
    /// runs first, so every declaration is dry whether or not it names a
    /// surface the fence just wrote. Comparing the declaration's mapping
    /// against this set can. Bounded by the number of mappings that ever carry
    /// a render window, which is single digits on a driven desktop; nothing
    /// reads it to make a flush decision.
    pub fence_flushed_mappings: std::collections::BTreeSet<u32>,
    /// Per-mid last write **command class** (ClearOnly vs Composite) — present path.
    pub surface_write_kind: BTreeMap<u32, SurfaceWriteKind>,
    pub present: PresentState,
    pub cursor: CursorState,
    pub display: DisplayHandshake,
    /// Every `FailEvent` also reached the always-on log through `record_fail`;
    /// this vec is only how an in-crate test reads them back. It is
    /// `#[cfg(test)]` because in a product boot nothing ever read it, so it grew
    /// for the life of the guest holding the one copy of nothing.
    #[cfg(test)]
    pub fails: Vec<FailEvent>,
    /// Last successful directed mapper capture (consumed on matching MAP/UNMAP).
    pub mapper_capture: Option<MapperCapture>,
    /// Cached IOSurfaceParavirtMapperDevice KVA from capture.
    pub mapper_device_kva: u64,
    /// Sync value table for event + encoder fence domains.
    ///
    /// Key: `(task_id, domain_tag, ref)` → value (event: explicit signal value;
    /// fence: monotonic generation). Domain tags match
    /// [`crate::runtime::plan::event_sync::Domain`] as `u8` (`1` = event,
    /// `2` = blitFence, `3` = computeFence, `4` = renderFence). Stored as a
    /// plain map so `model` does not depend on the planner types.
    pub fence_generations: BTreeMap<(u32, u8, u32), u64>,
    /// Child channel currently being drained (0 = none). Convenience for
    /// single-level skip; prefer [`Self::draining_mask`] for nested drains.
    pub draining_channel: u32,
    /// Bitmask of child channels mid-`drain_child_fifo` (stack). Nested
    /// `drain_other_child_fifos` must skip **all** bits set — otherwise it can
    /// re-enter a mid-packet channel and re-process the same head.
    pub draining_mask: u32,
    /// Contiguous mapping views (`MappingEntry::contig_ptr`) whose page tables
    /// changed. `DeviceState` cannot unmap (no HostOps); the runtime flushes
    /// these via `HostOps::unmap_pages` after retiring the backend objects and
    /// parent allocations that alias them (`mapper::flush_retired_views`).
    pub retired_views: Vec<(usize, usize)>,
    /// Backend parent allocations detached with `retired_views`. The runtime
    /// retires the GPU import before releasing the host view it aliases.
    pub retired_guest_imports: Vec<crate::runtime::guest_ram::ImportId>,
    /// Guest-write tokens whose page list is gone, awaiting release through
    /// `HostOps::untrack_guest_writes`. Drained by
    /// `mapper::flush_retired_views` alongside `retired_views`, for the same
    /// reason: both are host-side state this crate cannot free itself.
    pub retired_guest_write_tokens: Vec<u64>,
    /// Task-GVA HostOps views (zero-copy import substrate). Dropped on
    /// overlapping UnmapMemory/MapMemory2; flushed via `retired_views`.
    pub gva_host_views: Vec<GvaHostView>,
    /// Linear-window residency keys whose `host_linear_textures` entry died
    /// (task/object delete). `DeviceState` cannot reach the engine; the
    /// runtime unpins these
    /// ([`crate::runtime::render_writeback::retire_linear_residents`]) so the
    /// pinned images become LRU-evictable instead of leaking.
    pub retired_linear_residents: Vec<ComputeStorageResidencyKey>,
    /// Surface and GVA resources whose latest frame is still only in the engine
    /// resident, because nothing has synchronized or read their guest pages
    /// since the Store that produced it. See
    /// [`crate::runtime::writeback_debt`], which owns every transition.
    ///
    /// Empty unless [`crate::config::LAZY_WRITEBACK`] is on, and empty on the
    /// `backend-metal` arm, which arms nothing.
    pub pending_writebacks: crate::runtime::writeback_debt::PendingWritebacks,
    /// GVA render target → a hash of the guest physical pages its engine
    /// resident was last armed over.
    ///
    /// The census behind `gvares_*`: how hard the guest recycles a render
    /// target's address. The page list behind a GVA is the allocation's identity
    /// — same pages means literally the same memory — so a second arm at the
    /// same address and geometry with a *different* hash is a second allocation
    /// at a name the first one still holds.
    ///
    /// The same hash is the `generation` of the resident's registry key
    /// (`TargetIdentity::Gva`), so those arms now get their own GPU image rather
    /// than inheriting the previous allocation's pixels. This map is what says
    /// how often that separation is doing work, and it is deliberately
    /// independent of the key: a census that reads the thing it is scoring
    /// cannot report the day the two stop agreeing.
    ///
    /// Kept as a hash rather than the page list because this is a census, and
    /// the question is only whether two arms disagree.
    pub gva_resident_backing: std::collections::BTreeMap<u64, (u32, u32, u64)>,
    /// Completion stamps written to the guest this device lifetime.
    ///
    /// A stamp is the guest's fence: [`crate::runtime::drain::write_stamp`] puts
    /// the value in the FIFO page and raises the GPU IRQ, and from that instant
    /// the guest is entitled to treat the work as finished and reclaim anything
    /// it allocated for it. Counting stamps gives every deferred window an
    /// answer to the one question its page-set guard cannot ask: was the guest
    /// told this render was done before we wrote its bytes?
    pub completion_stamp_seq: u64,
    /// The guest CPU has written a presented framebuffer's pixels under a live
    /// resident: its compositor fills each new back buffer by copying from the
    /// front one with the CPU, without naming either in
    /// `CmdSynchronizeResources`.
    ///
    /// On the device this protocol was written for, the GPU renders into guest
    /// memory directly, so there is nothing to synchronize. Under the lazy
    /// writeback the front buffer's frame is still in a host resident, and the
    /// copy lands whole pages of an older frame. From the moment this is set,
    /// every presented framebuffer's owed frame is paid when a DisplaySwap
    /// presents it ([`crate::runtime::writeback_debt::pay_cpu_shared_mapping`]).
    ///
    /// One flag for the device rather than one per mapping, because the buffers
    /// trade roles: latching them one by one left the ones not yet caught stale
    /// as copy sources.
    pub guest_copies_framebuffers: bool,
    /// Census only: what this device has stamped, split by whether the value is
    /// still owed by the coalescing rail or already handed to publication.
    ///
    /// Sizes the one repair available for the held-packet cost, and says which
    /// unmet waits are honest. Feeds no verdict — see
    /// [`crate::runtime::drain::StampLedger`].
    pub stamp_ledger: crate::runtime::drain::StampLedger,
    /// Total stale views the reuse verify caught (fail-logged as
    /// `gva_view_stale`; the view self-heals via retire + rebuild).
    pub view_stale_reads: u64,
}

/// Domain tag for ch-event segment events (matches event_sync::Domain::Event).
pub const FENCE_DOMAIN_EVENT: u8 = 1;
/// Domain tag for blit fences (matches event_sync::Domain::BlitFence).
pub const FENCE_DOMAIN_BLIT: u8 = 2;
/// Domain tag for compute fences.
pub const FENCE_DOMAIN_COMPUTE: u8 = 3;
/// Domain tag for render fences.
pub const FENCE_DOMAIN_RENDER: u8 = 4;

/// What writing a completion word did to the ordering plane's record of that
/// slot's timeline.
///
/// Four facts and not one, because they mean four different things and only
/// one of them is a defect. [`Self::Repeat`] is how a packet that signals
/// nothing is spelled on this wire — the header repeats the slot's current
/// value rather than clearing it — and [`Self::Behind`] is a fence going
/// backwards, which unsatisfies every wait between the two values and is the
/// first thing to rule out before reading an unmet-wait count as an ordering
/// problem.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StampPublication {
    /// Nothing had ever published to this slot.
    First,
    /// The timeline moved forward.
    Advanced,
    /// The same value again.
    Repeat,
    /// The word is behind what the slot already holds, so the model's timeline
    /// did not move and this device's guest-visible write disagrees with it.
    ///
    /// Carries what the plane holds, because the count alone cannot separate
    /// the two things this can be: a rewind of one slot, where the page went
    /// backwards under a timeline that will not follow, and a timeline that has
    /// run ahead of every word the guest can read. The first is a slot to
    /// explain and the second is an ordering defect, and they are one number.
    Behind {
        held: reims_vgpu_core::identity::StampValue,
    },
}

impl DeviceState {
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
        Self {
            id,
            page_shift,
            gfx: GfxRegs::default(),
            iosfc: IosfcRegs::default(),
            gva_store_witness: Default::default(),
            session: Mutex::new(reims_vgpu_core::session::SessionModel::new(
                reims_vgpu_core::identity::SessionId(id.0 as u32),
            )),
            parked: crate::runtime::parked::ParkedStore::new(),
            translation_deferred_mask: 0,
            translation_order_hold_mask: 0,
            translation_order_holds: 0,
            present_translation_holds: 0,
            present_translation_hold_mask: 0,
            pending: PendingWork::default(),
            child_rings: std::array::from_fn(|_| ChannelRing::default()),
            max_task_id_seen: 0,
            max_mapping_id_seen: 0,
            tasks: TaskTable::new(),
            map_family_events: 0,
            map_audit: std::collections::BTreeMap::new(),
            node_guard: std::collections::BTreeMap::new(),
            objects: std::collections::BTreeSet::new(),
            task_resources: TaskResources::default(),
            lifecycle: Mutex::new(reims_vgpu_core::lifecycle::Lifecycle::new()),
            task_sampler_states: TaskSamplerStates::default(),
            task_depth_stencil_states: TaskDepthStencilStates::default(),
            rail: OnceLock::new(),
            texture_to_mapping: BTreeMap::new(),
            storage_incarnations: BTreeMap::new(),
            backing_window_refs: BackingWindowRefs::default(),
            aliased_references: Mutex::new(BTreeSet::new()),
            task_storage_epochs: BTreeMap::new(),
            mappings: BTreeMap::new(),
            host_surfaces: BTreeMap::new(),
            host_texture_surfaces: BTreeMap::new(),
            host_gva_surfaces: BTreeMap::new(),
            gva_touch_seq: 0,
            validity_seq: 0,
            gva_cache_bytes: 0,
            gva_cache_byte_cap: GVA_ENCODE_CACHE_BYTE_CAP,
            gva_eviction_witness: GvaEvictionWitness::default(),
            host_linear_textures: BTreeMap::new(),
            compute_storage_residency: BTreeMap::new(),
            fence_flushed_mappings: std::collections::BTreeSet::new(),
            surface_write_kind: BTreeMap::new(),
            present: PresentState::default(),
            cursor: CursorState {
                show: true,
                ..Default::default()
            },
            mapper_capture: None,
            mapper_device_kva: 0,
            display: DisplayHandshake::default(),
            #[cfg(test)]
            fails: Vec::new(),
            fence_generations: BTreeMap::new(),
            draining_channel: 0,
            draining_mask: 0,
            retired_views: Vec::new(),
            retired_guest_imports: Vec::new(),
            retired_guest_write_tokens: Vec::new(),
            retired_linear_residents: Vec::new(),
            pending_writebacks: crate::runtime::writeback_debt::PendingWritebacks::default(),
            completion_stamp_seq: 0,
            guest_copies_framebuffers: false,
            stamp_ledger: Default::default(),
            gva_resident_backing: std::collections::BTreeMap::new(),
            guest_linear_memo: LruBytesMemo::new(GUEST_LINEAR_MEMO_BYTE_CAP),
            gather_witness: crate::runtime::gather_witness::GatherWitness::default(),
            bound_buffers: crate::runtime::bound_buffers::BoundBuffers::default(),
            buffer_write_gen: crate::runtime::buffer_write_gen::BufferWriteGens::default(),
            sampled_content_gen: 0,
            host_writes: crate::runtime::host_writes::HostWrites::new(page_shift),
            guest_linear_scratch: Vec::new(),
            ref_texture_view_memo: LruBytesMemo::new(GUEST_LINEAR_MEMO_BYTE_CAP),
            mapper_ref_texture_memo: LruBytesMemo::new(GUEST_LINEAR_MEMO_BYTE_CAP),
            mapper_ref_texture_memo_scratch: Vec::new(),
            gva_host_views: Vec::new(),
            view_stale_reads: 0,
        }
    }

    /// Detach `e`'s contiguous view for later unmap (page table changed).
    /// Returns the retired (ptr, len) to push into `retired_views`.
    fn take_mapping_view(
        e: &mut MappingEntry,
    ) -> (
        Option<(usize, usize)>,
        Option<crate::runtime::guest_ram::ImportId>,
    ) {
        let import = e.contig_import.take().map(|import| {
            import.retire();
            import.id()
        });
        let view = (e.contig_ptr != 0).then_some((e.contig_ptr, e.contig_len));
        e.contig_ptr = 0;
        e.contig_len = 0;
        e.contig_footprint = None;
        (view, import)
    }

    /// Detach the guest-write token, returning it for release through
    /// [`crate::runtime::host::HostOps::untrack_guest_writes`].
    ///
    /// Called wherever [`Self::take_mapping_view`] is: the token and the view
    /// both name the page list as it stood, so a change to the list retires
    /// both. Also clears the Store stamp — a generation recorded against a
    /// released token cannot vouch for anything, and leaving it would let a
    /// re-tracked set's first readable generation coincide with it.
    fn take_guest_write_token(e: &mut MappingEntry) -> u64 {
        e.guest_write_gen_at_store = 0;
        e.guest_write_token_gen = 0;
        std::mem::replace(&mut e.guest_write_token, 0)
    }

    /// Detach every HostOps mapping owned by the current guest lifetime.
    ///
    /// Device reset is a lifetime boundary even when QEMU itself remains alive.
    /// Returning the views lets the runtime invalidate backend aliases first,
    /// then release them through the bound HostOps implementation.
    pub fn take_all_host_views(&mut self) -> Vec<(usize, usize)> {
        let mut views = std::mem::take(&mut self.retired_views);
        let mut tokens = std::mem::take(&mut self.retired_guest_write_tokens);
        for mapping in self.mappings.values_mut() {
            let (view, import) = Self::take_mapping_view(mapping);
            if let Some(view) = view {
                views.push(view);
            }
            if let Some(import) = import {
                self.retired_guest_imports.push(import);
            }
            let token = Self::take_guest_write_token(mapping);
            if token != 0 {
                tokens.push(token);
            }
        }
        // The sampled-cache witness arms its own tokens against window page
        // sets, and they are not reachable from any `MappingEntry` — so the
        // loop above cannot see them and a reset that only walked mappings left
        // them armed on the host forever.
        tokens.extend(self.gather_witness.take_tokens());
        tokens.extend(self.gva_store_witness.take_tokens());
        // Back onto the retired list rather than out through the return value:
        // the caller's contract is "invalidate backend aliases, then release
        // views", and a token release is neither. `flush_retired_views` drains
        // both, and `Device::reset_with_host` runs it before `reset` discards
        // the vector.
        self.retired_guest_write_tokens = tokens;
        views.extend(self.gva_host_views.drain(..).filter_map(|view| {
            (view.ptr != 0 && view.ptr_len != 0).then_some((view.ptr, view.ptr_len))
        }));
        views
    }

    /// Snapshot fence generation if present.
    pub fn fence_generation(&self, task_id: u32, domain: u8, fence_ref: u32) -> Option<u64> {
        self.fence_generations
            .get(&(task_id, domain, fence_ref))
            .copied()
    }

    /// Forget every generation this task holds for `fence_ref`, and say which
    /// encoder domains held one.
    ///
    /// **The guest's fence is dead and its number will come back.** A wait is
    /// satisfied when the stored generation is at or past its target, so a
    /// generation that outlives its fence makes the *next* fence to get that
    /// ref start life already signalled — the first wait on it passes with
    /// nothing behind it. Nothing else forgets these: they are not keyed by a
    /// name the namespace retires, and their task's teardown is the only other
    /// event that reaches them.
    ///
    /// The tag is this device's own split of one guest fence across encoder
    /// domains, not a guest-visible term, so one ref can hold up to four
    /// generations and all four are the same object's. All of them go.
    ///
    /// # The two reference spaces are one, and that was measured
    ///
    /// A `CmdDeleteObject` carrying `OPCODE_DELETE_FENCE` names a ref in the
    /// *serializer's* per-kind space, and this table is keyed by the ref a
    /// command stream's fence record carries. Those being the same number for
    /// the same object is not something this device may assume — see
    /// `crate::runtime::drain::apply_delete_object` for the boot that asked the
    /// object table the same question and found its spaces **unrelated**. So
    /// this one was counted before it was acted on: a driven macos-15 boot sent
    /// two fence deletes and **both** named a ref this device held a
    /// render-domain generation under, with `delete_fence_ref_absent=0`. The
    /// spaces coincide, and the retirement below is what that reading buys.
    pub fn retire_fence(&mut self, task_id: u32, fence_ref: u32) -> Vec<u8> {
        let mut cleared = Vec::new();
        for tag in [
            FENCE_DOMAIN_EVENT,
            FENCE_DOMAIN_BLIT,
            FENCE_DOMAIN_COMPUTE,
            FENCE_DOMAIN_RENDER,
        ] {
            if self
                .fence_generations
                .remove(&(task_id, tag, fence_ref))
                .is_some()
            {
                cleared.push(tag);
            }
        }
        cleared
    }

    /// Store fence generation (monotonic update owned by the planner).
    pub fn set_fence_generation(&mut self, task_id: u32, domain: u8, fence_ref: u32, value: u64) {
        if fence_ref == 0 {
            return;
        }
        self.fence_generations
            .insert((task_id, domain, fence_ref), value);
    }

    /// Record a clear-only write to `mapping_id` (display_clear / CLEAR Store).
    pub fn note_surface_clear(&mut self, mapping_id: u32) {
        if mapping_id == 0 {
            return;
        }
        // Guest Clear wipes the surface: next present of this mid must not be
        // treated as a finished composite (unless a later Draw Store re-marks
        // Composite).
        self.surface_write_kind
            .insert(mapping_id, SurfaceWriteKind::ClearOnly);
    }

    /// Record a composite/draw Store to `mapping_id`.
    pub fn note_surface_composite(&mut self, mapping_id: u32) {
        if mapping_id == 0 {
            return;
        }
        self.surface_write_kind
            .insert(mapping_id, SurfaceWriteKind::Composite);
    }

    /// A draw Store published a **complete** frame for `mapping_id` into guest
    /// pages (full-frame resident writeback, `import_present ok_runs`).
    ///
    /// Protocol-structural dense marker: this mapping now holds a complete full
    /// frame, so advance its [`PresentState::dense_frame_seq`] off the global
    /// [`PresentState::dense_frame_counter`]. A surface presented twice with no
    /// advance in between received no full frame of its own, which is the
    /// `present_unbacked` gate in [`Self::note_present_backing`] — the only
    /// reader. The counter is monotonic per full-frame Store across all
    /// mappings, so the value is a witness of "something was published for this
    /// mid", never a staleness measure on its own.
    pub fn note_dense_frame_published(&mut self, mapping_id: u32, width: u32, height: u32) {
        if mapping_id == 0 || width == 0 || height == 0 {
            return;
        }
        self.present.dense_frame_counter = self.present.dense_frame_counter.saturating_add(1);
        let seq = self.present.dense_frame_counter;
        self.present.dense_frame_seq.insert(mapping_id, seq);
    }

    /// Advance the per-present epoch counter and return the new value. Call
    /// EXACTLY ONCE per present cycle (see [`PresentState::present_epoch`]).
    pub fn advance_present_epoch(&mut self) -> u64 {
        self.present.present_epoch = self.present.present_epoch.saturating_add(1);
        self.present.present_epoch
    }

    /// Record that `mapping_id` is being presented and report whether the guest
    /// ever sent a full-frame Store **naming it** for what is about to be shown.
    ///
    /// Structural only: decoded Store bookkeeping, never measured content, and
    /// never the resident. Say what that leaves out, because the name reads
    /// broader than the check: a `None` here means the guest sent a frame for
    /// this mid, **not** that the resident this present will read holds it. See
    /// [`PresentState::dense_frame_seq`].
    ///
    /// Records the witness on every call, so a member that stays unbacked
    /// reports once per present rather than once per lifetime — except
    /// [`PresentBacking::NeverStored`], which by construction can only be
    /// reported on a mapping's first present since it was created.
    pub fn note_present_backing(&mut self, mapping_id: u32) -> Option<PresentBacking> {
        if mapping_id == 0 {
            return None;
        }
        let seq = self
            .present
            .dense_frame_seq
            .get(&mapping_id)
            .copied()
            .unwrap_or(0);
        let previous = self.present.presented_dense_seq.insert(mapping_id, seq);
        match previous {
            Some(prev) if prev == seq => Some(PresentBacking::Restaled { seq }),
            // First present since this mapping was created. `dense_frame_seq` is
            // pruned by `forget_compositor_mapping`, so a *re-created* surface
            // arrives here with no witness and no seq — and this arm is the only
            // thing that can see it.
            //
            // It matters because that is the worst version of this class rather
            // than a corner of it: a surface nothing has ever Stored into is
            // uninitialized, so presenting it shows a fully black screen, not a
            // stale one. Measured on a live boot: the guest re-created its
            // scanout surfaces (`gen` reset 82 → 0) and we presented mid 6 at
            // `gen=0` with `px0=[0,0,0,0]` and `rgb_nz=4254` of 2 073 600 — a
            // black screen — for the three presents that followed.
            // `present_unbacked` fired **zero** times during that whole boot.
            //
            // The guest was awake for all of it. An earlier reading of this
            // boot blamed display sleep and it does not survive the log: the
            // 86 s the guest went quiet is bracketed by seven
            // `sync_exec_lock_hold` events of 935-979 ms each, one guest exec
            // packet apiece, on an otherwise idle device. The surface
            // re-creation is downstream of the stall, not of a power
            // transition. What causes the stall is a separate question and is
            // measured by `draw_phase`.
            //
            // The old shape could not have caught it. It compared this present's
            // seq against the previous present's, which is a check for a
            // *repeat* — a transition — while "this surface has never been
            // written" is a *state*. The state was sitting in `dense_frame_seq`
            // the whole time as an absent key.
            None if seq == 0 => Some(PresentBacking::NeverStored),
            _ => None,
        }
    }

    /// The running rail's own device-lifetime state, created empty on first
    /// ask.
    ///
    /// `None` means the slot is already held by a *different* rail's type.
    /// [`crate::backend::select`] latches one rail per process, so no live
    /// build can reach it; it is an answer rather than a panic because every
    /// caller is on a path whose lawful reply to "nothing retained" is the
    /// ablation it already implements — reconstruct, or report absent.
    pub fn rail_state<T: RailDeviceState + Default>(&self) -> Option<&T> {
        self.rail
            .get_or_init(|| Box::<T>::default() as Box<dyn RailDeviceState>)
            .as_any()
            .downcast_ref::<T>()
    }

    fn forget_compositor_mapping(&mut self, mapping_id: u32) {
        // The plane draw ring is keyed by mapping id and read by two witnesses,
        // so it is dropped with the mapping: bounded by the live compositor
        // surfaces, and a recycled id cannot inherit a predecessor's passes.
        // Through the trait, because the record belongs to whichever rail is
        // running and the model may not name one.
        crate::backend::Backend::forget_mapping(&crate::backend::selected(), mapping_id);
        // Prune the dense-frame seq: a recycled mapping id must not inherit a
        // stale predecessor's dense seq.
        self.present.dense_frame_seq.remove(&mapping_id);
        // Same rule for the presented-seq witness: a recycled id must not
        // compare its first present against a predecessor's seq.
        self.present.presented_dense_seq.remove(&mapping_id);
    }

    /// Last write class for present keep-prior decisions.
    pub fn surface_write_kind(&self, mapping_id: u32) -> SurfaceWriteKind {
        self.surface_write_kind
            .get(&mapping_id)
            .copied()
            .unwrap_or(SurfaceWriteKind::Unknown)
    }

    /// Drop every held bind resolution for one task.
    ///
    /// The answer for every packet after which a reference may name different
    /// bytes: a new page-table root, a new object list, a deleted object, a
    /// deleted task, and a replaced physical page. Each is rare against the
    /// draw rate, so the whole task goes rather than the machinery that would
    /// map an object id back to the references resolved through it.
    ///
    /// Ungated so the packet handlers stay free of `cfg`; on a build with no
    /// Vulkan engine nothing can hold a resolution and this is a no-op.
    /// Returns how many resolutions were dropped, so the caller can name the
    /// cause on the census. A count, not an event: one `SetObjectList` that
    /// retires forty entries and one that retires none read identically as
    /// events, and it is the entries that become the re-walks.
    pub fn retire_bound_buffers_for_task(&mut self, task_id: u32) -> usize {
        // Ungated and unconditional: a task's object ids stop naming its objects
        // whatever backend is compiled in, and a stamp that outlived its task
        // would read as quiet for whatever the next task puts at that id.
        self.buffer_write_gen.retire_task(task_id);
        self.bound_buffers.retire_task(task_id)
    }

    /// Drop the held bind resolutions for `task_id` covering `[gva, gva+len)`.
    ///
    /// The map/unmap answer, which names the exact range the guest moved. See
    /// Drop the held bind resolutions for one reference, at every offset.
    ///
    /// The `CmdDeleteObject` rule. See
    /// [`crate::runtime::bound_buffers::BoundBuffers::retire_ref`] for why this
    /// is scoped to the reference rather than the task.
    pub fn retire_bound_buffers_for_ref(&mut self, task_id: u32, ref_: u32) -> usize {
        self.bound_buffers.retire_ref(task_id, ref_)
    }

    pub fn retire_bound_buffers_in_range(&mut self, task_id: u32, gva: u64, len: u64) -> usize {
        self.bound_buffers.retire_range(task_id, gva, len)
    }

    pub fn reset(&mut self) {
        // Held bind resolutions name guest addresses under a device that is
        // going away; nothing about them survives a reset.
        self.bound_buffers.clear();
        // A translation hold that is still standing here never resolved. The
        // hold itself is control flow — the FIFO is parked until an AIR module
        // finishes loading and the packet is retried, not consumed — so it is
        // census. THIS is the failure: the device went away with guest packets
        // still parked behind a load that never completed, and those packets are
        // lost. Reading it at the lifetime boundary needs no age, depth or
        // timeout; the guest's own teardown is the deadline.
        if self.translation_order_hold_mask != 0 || self.translation_deferred_mask != 0 {
            crate::observe::fail(format!(
                "translation_hold_unreleased held_mask={:#x} producer_mask={:#x} episodes={} \
                 (device reset with guest packets still parked behind an AIR load)",
                self.translation_order_hold_mask,
                self.translation_deferred_mask,
                self.translation_order_holds
            ));
        }
        let id = self.id;
        let page_shift = self.page_shift;
        // Keep the interrupt-status Arcs wired to the registry slot: the
        // lock-free ISR read rail clones them once at device create.
        let intr_disp = Arc::clone(&self.gfx.interrupt_status_disp);
        let intr_gpu = Arc::clone(&self.gfx.interrupt_status_gpu);
        let intr_fault = Arc::clone(&self.gfx.interrupt_fault);
        let fifo_read = Arc::clone(&self.gfx.fifo_read);
        let child_rung = Arc::clone(&self.gfx.child_doorbell_rung);
        intr_disp.store(0, Ordering::Release);
        intr_gpu.store(0, Ordering::Release);
        intr_fault.store(0, Ordering::Release);
        fifo_read.store(0, Ordering::Release);
        // Cleared as well as kept: a reset drops every channel, so a bit rung
        // before it names a channel that no longer exists.
        child_rung.store(0, Ordering::Release);
        *self = Self::new(id, page_shift);
        self.gfx.interrupt_status_disp = intr_disp;
        self.gfx.interrupt_status_gpu = intr_gpu;
        self.gfx.interrupt_fault = intr_fault;
        self.gfx.fifo_read = fifo_read;
        self.gfx.child_doorbell_rung = child_rung;
    }

    /// Queue the engine-unpin for a dying linear cache entry that still owns a
    /// resident image (see `retired_linear_residents`).
    fn retire_linear_resident(&mut self, task_id: u32, texture_ref: u32, e: &HostLinearTexture) {
        if e.resident_gen == 0 || e.row_stride > u32::MAX as u64 {
            return;
        }
        self.retired_linear_residents
            .push(ComputeStorageResidencyKey::linear(
                task_id,
                texture_ref,
                e.gva,
                e.row_stride as u32,
                e.row_stride.saturating_mul(e.height as u64),
                e.width,
                e.height,
                e.pixel_format,
            ));
    }

    fn retire_task_linear_residents(&mut self, task_id: u32) {
        let doomed: Vec<(u32, HostLinearTexture)> = self
            .host_linear_textures
            .iter()
            .filter(|((t, _), e)| *t == task_id && e.resident_gen != 0)
            .map(|((_, r), e)| {
                (
                    *r,
                    HostLinearTexture {
                        bytes: Vec::new(),
                        ..e.clone()
                    },
                )
            })
            .collect();
        for (r, e) in doomed {
            self.retire_linear_resident(task_id, r, &e);
        }
    }

    /// Install the guest's task under `task_id`, replacing any previous one.
    ///
    /// Returns nothing: it used to return `bool`, and the only `false` it could
    /// produce was `task_id >= MAX_TASKS`. With the task table keyed by the
    /// guest's own `u32` there is no id this can refuse, so a `bool` here would
    /// be a value 81 call sites asserted on and none of them could ever see
    /// false — the shape that makes a later real failure easy to add and easy to
    /// ignore.
    pub fn define_task(&mut self, task_id: u32, length: u64, directory_pfn: u32) {
        self.max_task_id_seen = self.max_task_id_seen.max(task_id);
        // Redefining a *live* task is the one shape here that can lose published
        // guest state: the objects below are dropped, and if the new directory
        // roots a different physical page at the list's own GVA then everything
        // the guest published into the old one reads back as zero. macOS 13 does
        // not do this and macOS 26 does, which is why it is counted separately
        // from a first definition rather than folded into one route.
        // The owner's answer, not a second derivation of it beside the owner.
        // `Lifecycle::define_task` ends the previous address space and reports
        // the pair of page-table roots; `Redefinition::root_moved` is the
        // question this route used to ask of `TaskEntry::directory_pfn`, and two
        // records of one fact are what this group moved to stop.
        let redefined = self.define_task_namespace(task_id, directory_pfn);
        if let Some(redefinition) = redefined {
            crate::runtime::drain::note_store_route(if redefinition.root_moved() {
                "define_task_redefined_live_new_root"
            } else {
                "define_task_redefined_live_same_root"
            });
        }
        // Drop objects for this task on redefine.
        //
        // The incarnation epoch moves only when there was a task here to end.
        // A first definition has no prior storage to be told apart from, and
        // bumping then would make the first incarnation of a name depend on
        // whether anything else had happened to its task.
        if self.tasks.is_active(task_id) {
            self.bump_task_storage_incarnations(task_id);
        }
        self.objects.retain(|&(t, _)| t != task_id);
        self.task_resources.delete_task(task_id);
        self.task_sampler_states.delete_task(task_id);
        crate::runtime::drain::note_store_route_n(
            "ds_state_task_deleted",
            self.task_depth_stencil_states.delete_task(task_id) as u64,
        );
        if let Some(rail) = self.rail.get() {
            rail.delete_task(task_id);
        }
        // A deleted task's whole address space goes with it, so its live
        // mappings are not leaks and a reused id must not inherit them.
        self.map_audit.remove(&task_id);
        // Same lifetime, same reason: the watched pages were nodes of the tree
        // this id is losing, and after a redefine they describe whatever the
        // guest has since done with them.
        self.node_guard.remove(&task_id);
        self.retire_task_linear_residents(task_id);
        self.host_linear_textures.retain(|&(t, _), _| t != task_id);
        // New directory ⇒ old GVA HostOps views alias the wrong PT — retire.
        self.retire_task_gva_views(task_id);
        self.tasks
            .define(task_id, TaskEntry::define(length, directory_pfn));
    }

    /// Retire every GVA HostOps view registered under `task_id`.
    ///
    /// Both entry points that end a task's page table — `define_task` on a
    /// redefine and `delete_task` on teardown — owe exactly this: the views hold
    /// host pointers into pages the guest is about to recycle, so leaving one
    /// live is a read of memory that no longer belongs to the surface (the
    /// WindowServer SIGSEGV class [`crate::runtime::gva_view::write_span_within`]
    /// documents). `retired_views` is
    /// drained by `mapper::flush_retired_views` through `HostOps::unmap_pages`.
    fn retire_task_gva_views(&mut self, task_id: u32) {
        let mut i = 0;
        while i < self.gva_host_views.len() {
            if self.gva_host_views[i].task_id == task_id {
                let v = self.gva_host_views.swap_remove(i);
                if v.ptr != 0 && v.ptr_len != 0 {
                    self.retired_views.push((v.ptr, v.ptr_len));
                }
            } else {
                i += 1;
            }
        }
    }

    /// PVG `CmdDeleteTask` (op `0x20`): drop task directory + object list entries.
    /// Guest reuses task ids; leaving stale active tasks corrupts GVA walks.
    pub fn delete_task(&mut self, task_id: u32) -> bool {
        self.max_task_id_seen = self.max_task_id_seen.max(task_id);
        if !self.tasks.is_active(task_id) {
            return false;
        }
        self.objects.retain(|&(t, _)| t != task_id);
        self.task_resources.delete_task(task_id);
        self.delete_task_namespace(task_id);
        self.task_sampler_states.delete_task(task_id);
        crate::runtime::drain::note_store_route_n(
            "ds_state_task_deleted",
            self.task_depth_stencil_states.delete_task(task_id) as u64,
        );
        if let Some(rail) = self.rail.get() {
            rail.delete_task(task_id);
        }
        self.retire_task_linear_residents(task_id);
        self.host_linear_textures.retain(|&(t, _), _| t != task_id);
        self.host_texture_surfaces.retain(|&(t, _), _| t != task_id);
        // Clear texture→mapping latches for this task.
        self.texture_to_mapping.retain(|&(t, _), _| t != task_id);
        self.bump_task_storage_incarnations(task_id);
        // GVA encode cache retained until Unmap of that range.
        // Task teardown ≡ all GPU VA maps for this task go away — retire any
        // HostOps views we held (does not touch host_gva_surfaces encode).
        // Runtime flushes retired_views via HostOps::unmap_pages.
        self.retire_task_gva_views(task_id);
        // The two observation ledgers keyed by task id go with it, exactly as
        // they do on a redefine. Both were reachable only through `define_task`
        // before, which cleaned them up whenever an id came back — so a task the
        // guest deletes and never redefines left its record behind for the life
        // of the process. Neither ledger is read for a task that does not exist,
        // so this costs no behaviour; it stops an id the guest is done with from
        // holding a page set that describes memory it has given back.
        self.map_audit.remove(&task_id);
        self.node_guard.remove(&task_id);
        self.tasks.remove(task_id);
        true
    }

    pub fn set_object_list(&mut self, task_id: u32, pfn: u32, count: u32) -> bool {
        self.max_task_id_seen = self.max_task_id_seen.max(task_id);
        let Some(task) = self.tasks.get_mut(task_id).filter(|t| t.active) else {
            StateMutationDecline::SetObjectListTaskInactive { task_id }.emit(u64::from(task_id));
            return false;
        };
        task.object_list_pfn = pfn;
        task.object_list_count = count;
        true
    }

    /// Every mapping id one task-local object reference can name.
    ///
    /// This device carries two ways from a reference to a surface, because the
    /// guest has two: on some paths the reference *is* the mapping id, and on
    /// the rest [`Self::texture_to_mapping`] holds the per-task registration a
    /// mapper-ref-texture create recorded. A statement about the reference — a validity
    /// quad, an owed render frame — is a statement about every mapping it
    /// names, so the candidate set is one rule and lives here.
    ///
    /// It is one rule because it used to be two, spelled differently, and only
    /// one of them was right about what "named nothing" means:
    /// `resource_validity::apply` built both candidates and then asked
    /// [`Self::mappings`] which of them exists, while
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
        let mut named = NamedMappings::default();
        if object_id == 0 {
            // `writeInvalidates` skips null resources and id 0; `pageBacking`
            // never emits one. A zero id names nothing.
            return named;
        }
        named.push(object_id);
        if let Some(&mid) = self.texture_to_mapping.get(&(task_id, object_id)) {
            named.push(mid);
        }
        named
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
            .any(|id| self.mappings.contains_key(&id))
    }

    pub fn insert_object(&mut self, task_id: u32, ref_: u32) -> bool {
        let discriminant = (u64::from(task_id) << 32) | u64::from(ref_);
        self.max_task_id_seen = self.max_task_id_seen.max(task_id);
        if !self.tasks.is_active(task_id) {
            StateMutationDecline::InsertObjectTaskInactive {
                task_id,
                object_ref: ref_,
            }
            .emit(discriminant);
            return false;
        }
        self.objects.insert((task_id, ref_));
        true
    }

    /// How many backing identities this device has minted.
    ///
    /// The one reader is the device-info reply census, whose question is whether
    /// a freshly minted identity could equal one already in use. It is
    /// deliberately the *counter* and not a table size: the counter is what
    /// every key space draws from, so it is the only number that answers for the
    /// identity space as a whole.
    #[must_use]
    pub fn backing_identities_minted(&self) -> u64 {
        self.backing_window_refs.minted()
    }

    /// The canonical identity of a bare guest page frame.
    ///
    /// The one caller is a `CmdGetDeviceInfo` reply destination — see
    /// [`BackingWindowRefs`]'s `frames` table for why a page frame needs a key
    /// space of its own and why sharing the counter is what makes that safe.
    pub fn frame_backing_identity(&self, pfn: u32) -> reims_vgpu_core::access::BackingId {
        reims_vgpu_core::access::BackingId(self.backing_window_refs.frame_identity(pfn))
    }

    /// Open these child domains, for a test that needs one live without driving
    /// a definition packet through the drain.
    ///
    /// `#[cfg(test)]`, and through the owner's own door rather than by writing a
    /// field: a test that could set openness directly would be testing a second
    /// record of it, which is exactly what moving this lifetime removed. A
    /// domain already open is left open — the model refuses the redefinition and
    /// a fixture asking twice means the same thing either way.
    #[cfg(test)]
    pub(crate) fn open_child_domains_for_test(&mut self, mask: u32) {
        for channel in 1..crate::model::MAX_CHANNELS as u32 {
            if mask & (1u32 << channel) != 0 {
                let _ = self
                    .session
                    .lock()
                    .expect("session")
                    .open_channel(reims_vgpu_core::identity::ChannelId(channel));
            }
        }
    }

    /// Whether a channel definition has opened this child domain.
    ///
    /// The model's answer and no copy of it. A domain becomes drainable three
    /// ways on this interface and only one of them is a definition; see
    /// [`Self::session`] for why the other two answer a different question.
    pub fn child_domain_open(&self, channel: u32) -> bool {
        self.session
            .lock()
            .expect("session")
            .channel_open(reims_vgpu_core::identity::ChannelId(channel))
    }

    /// Perform a resolved control operation's effect on the ordering plane.
    ///
    /// The one door a channel definition or free reaches the model through, and
    /// the reason the field below it is private: the drain used to hold a `&mut`
    /// and call the model directly, which is a shape no `&DeviceState` caller
    /// could copy.
    pub fn apply_channel_control(
        &self,
        op: reims_vgpu_core::control::ControlOp,
    ) -> Result<(), reims_vgpu_core::session::ControlRefusal> {
        self.session.lock().expect("session").apply_control(op)
    }

    /// Tell the ordering plane that the guest has created a pipeline object.
    ///
    /// `reims_vgpu_core::pipeline::PipelineState::Declared` is exactly this:
    /// the object exists and no host work has started on it. The generation is
    /// the model's own and is read inside the lock rather than handed in — a
    /// caller that could state it could declare a pipeline into a lifetime that
    /// has already closed, which is the one thing
    /// `PipelineTable::generation_closed` exists to make impossible.
    ///
    /// Returns whether this call was the declaration. A pipeline already
    /// declared answers `false`, which is the ordinary case: the guest binds
    /// the same pipeline on every draw.
    ///
    /// # Why the model is told at all before anything reads it
    ///
    /// `SessionModel::admit` refuses an exec transaction whose stream binds a
    /// pipeline the table does not hold — `LeaseRefusal::Absent`, which is
    /// every exec packet a real guest sends. The table being empty is the
    /// ordering group's next blocker after the classes, and this is the
    /// rail-neutral half of filling it: *that* a pipeline exists is the guest's
    /// fact, while translating and building it are the running rail's and reach
    /// the model from there.
    pub fn declare_pipeline(&self, pipeline: reims_vgpu_core::identity::ResourceId) -> bool {
        let mut session = self.session.lock().expect("session");
        let generation = session.generation();
        session.pipelines().declare(pipeline, generation)
    }

    /// Tell the ordering plane the guest has ended a pipeline's life.
    ///
    /// The other half of [`Self::declare_pipeline`], and the reason the pair
    /// can land together: a table that only ever grows is a table whose census
    /// says nothing, and the guest's own destroy is the one event that says a
    /// pipeline is over. `CmdDeleteObject` names it — measured, and the name is
    /// the model's own.
    ///
    /// `Ended::stranded` is the transactions parked on a compilation that will
    /// now never finish — empty today, because nothing is admitted into this
    /// model yet, and counted rather than dropped so the day it stops being
    /// empty is a number and not a hang. `Ended::took` is the other half, and a
    /// driven boot needs it: the guest deletes render pipelines this device
    /// never drew with, so 170 retirements a boot were 116 the table took and
    /// 54 that named a slot it has no entry for.
    pub fn retire_pipeline(
        &self,
        pipeline: reims_vgpu_core::identity::ResourceId,
    ) -> reims_vgpu_core::session::Ended {
        self.session
            .lock()
            .expect("session")
            .pipeline_retired(pipeline)
    }

    /// Step a declared pipeline along its build, from the rail that is
    /// building it.
    ///
    /// `Translating` and `Compiling` are the two steps with no consequence
    /// outside the table — nothing is released and nothing is stranded — so
    /// they go through `PipelineTable::advance` directly, which is why
    /// `SessionModel::pipelines` is read-write. The two steps that *do* have a
    /// consequence have their own doors: [`Self::ready_pipeline`] and
    /// [`Self::refuse_pipeline`].
    ///
    /// Returns whether the step was legal and taken. An illegal step is
    /// ordinary rather than a defect — a rail with no memo re-walks the same
    /// pipeline on every draw and finds it already `Ready` — and the caller
    /// counts them rather than ignoring them, because the same `false` is also
    /// what a compile finishing after the guest's delete answers.
    pub fn advance_pipeline(
        &self,
        pipeline: reims_vgpu_core::identity::ResourceId,
        next: reims_vgpu_core::pipeline::PipelineState,
    ) -> bool {
        self.session
            .lock()
            .expect("session")
            .pipelines()
            .advance(pipeline, next)
    }

    /// This rail no longer holds a translation for a pipeline the model called
    /// ready, so work binding it waits again.
    ///
    /// See [`reims_vgpu_core::pipeline::PipelineTable::withdraw`] for why this
    /// is not an [`Self::advance_pipeline`] step.
    pub fn withdraw_pipeline(&self, pipeline: reims_vgpu_core::identity::ResourceId) -> bool {
        self.session
            .lock()
            .expect("session")
            .pipelines()
            .withdraw(pipeline)
    }

    /// What an executor found its shaders do with each bound slot.
    ///
    /// **Published before [`Self::ready_pipeline`], never after.** Ready is the
    /// event that lets a transaction lease the pipeline, and a lease taken
    /// before the reflection landed produces a footprint that is conservative
    /// forever — the walk asks once, at admission, and never asks again. So the
    /// order is publish, then ready; the reverse is not a defect the model can
    /// see, which is exactly why it is written down here.
    ///
    /// Returns whether it landed. `false` for a pipeline this model does not
    /// hold or has already retired or refused; see
    /// [`reims_vgpu_core::pipeline::PipelineTable::publish_usage`].
    pub fn publish_pipeline_usage(
        &self,
        pipeline: reims_vgpu_core::identity::ResourceId,
        usage: reims_vgpu_core::pipeline::PublishedUsage,
    ) -> bool {
        self.session
            .lock()
            .expect("session")
            .pipelines()
            .publish_usage(pipeline, usage)
    }

    /// Whether an executor has already published this pipeline's reflection.
    ///
    /// The question a rail asks before building the answer, so the binding
    /// tables are enumerated once per build rather than once per packet that
    /// dispatches the pipeline. See
    /// [`reims_vgpu_core::pipeline::PipelineTable::usage_published`] for why
    /// the memo is the model's and not the rail's.
    pub fn pipeline_usage_published(
        &self,
        pipeline: reims_vgpu_core::identity::ResourceId,
    ) -> bool {
        self.session
            .lock()
            .expect("session")
            .pipelines()
            .usage_published(pipeline)
    }

    /// The published reflections, for the length of one packet's walk.
    ///
    /// A guard rather than a per-lookup call, because the walk asks once per
    /// bound slot per draw and a lock per slot would put the ordering plane's
    /// mutex on the hottest path this device has. Held across the walk, taken
    /// and dropped by [`crate::runtime::ingress::device_packet`], and safe to
    /// hold there because nothing the walk touches reaches this mutex —
    /// [`Self::task_access`] and the name resolvers are the `lifecycle` owner's,
    /// which is a different lock.
    #[must_use]
    pub fn pipeline_usage(&self) -> PipelineUsage<'_> {
        PipelineUsage(self.session.lock().expect("session"))
    }

    /// A pipeline finished building and is usable.
    ///
    /// The door rather than `advance(.., Ready)`, because becoming ready is
    /// the event that releases the transactions parked on it — and a rail that
    /// recorded the state without releasing the work would leave every exec
    /// that leased this pipeline holding its channel's publication head.
    pub fn ready_pipeline(&self, pipeline: reims_vgpu_core::identity::ResourceId) -> bool {
        self.session
            .lock()
            .expect("session")
            .pipeline_ready(pipeline)
    }

    /// Whether a declared pipeline has already reached `Ready`.
    ///
    /// The read half of [`Self::ready_pipeline`], and it exists so a caller can
    /// decline to redo work whose answer is already in the table. A pipeline the
    /// table does not hold answers `false`: an undeclared lease is not a
    /// promise, and treating it as one would skip the step that makes it one.
    /// What the ordering plane's pipeline table says about one pipeline, for a
    /// caller reporting a failure that should not have been reachable.
    ///
    /// `None` is "the table has no entry", which is a different fact from every
    /// state it could be in and is the one a lease that was never declared
    /// produces. Read-only on purpose: this is asked on a path that has already
    /// lost a draw, and a diagnostic that declared what it was asking about
    /// would answer its own question.
    #[must_use]
    pub fn pipeline_state(
        &self,
        pipeline: reims_vgpu_core::identity::ResourceId,
    ) -> Option<&'static str> {
        self.session
            .lock()
            .expect("session")
            .pipelines()
            .get(pipeline)
            .map(|entry| entry.state.name())
    }

    #[must_use]
    pub fn pipeline_is_ready(&self, pipeline: reims_vgpu_core::identity::ResourceId) -> bool {
        self.session
            .lock()
            .expect("session")
            .pipelines()
            .get(pipeline)
            .is_some_and(|entry| entry.state.is_ready())
    }

    /// A pipeline will never build, with the reason the rail refused it.
    ///
    /// `Ended::stranded` is the transactions that can therefore never be ready.
    /// They come back rather than being dropped for the same reason
    /// [`Self::retire_pipeline`]'s do, and the caller withdraws each and says
    /// why. Empty today, because nothing is admitted into this model yet.
    /// `Ended::took` is whether the refusal was a legal step at all.
    pub fn refuse_pipeline(
        &self,
        pipeline: reims_vgpu_core::identity::ResourceId,
        reason: reims_vgpu_core::pipeline::RefusalReason,
    ) -> reims_vgpu_core::session::Ended {
        self.session
            .lock()
            .expect("session")
            .pipeline_refused(pipeline, reason)
    }

    /// Give one packet an ordering position in the model.
    ///
    /// The door `SessionModel::admit` is reached through, and the last of the
    /// model's four planes to have one — declaration, publication and the
    /// pipeline table already do. What comes back is the model's answer *and*
    /// the incarnation it was admitted into, read under the same lock: a caller
    /// that asked for the epoch separately could be told about a loss between
    /// the two reads and then complete the transaction under the wrong one,
    /// which is the race [`reims_vgpu_core::session::SessionModel::complete`]
    /// takes an epoch argument to answer.
    ///
    /// # Errors
    ///
    /// The model's own refusal, unchanged. Every one of them is the packet
    /// being un-admittable rather than this device being unable to ask — a
    /// closed generation, a domain no definition opened, a payload that is not
    /// the class its opcode declares — so the caller names it on the failure
    /// channel and advances the ring.
    pub fn admit_packet(
        &self,
        packet: &reims_vgpu_core::session::Packet,
    ) -> Result<Admission, reims_vgpu_core::session::Refusal> {
        let mut session = self.session.lock().expect("session");
        let admitted = session.admit(packet)?;
        Ok(Admission {
            epoch: session.epoch(),
            admitted,
        })
    }

    /// The semantic lifetime the model is in right now.
    ///
    /// **A reader's fact, and that is why it is asked here rather than inside
    /// `admit`.** A guest reset races the drain: a packet that left the ring
    /// before the reset and reaches ingress after it names a lifetime that has
    /// closed, and nothing else can tell — the guest's packet carries no
    /// generation. So the reader states the one it was holding when it took the
    /// bytes, and the model compares. See
    /// [`reims_vgpu_core::session::Packet::session`].
    #[must_use]
    pub fn session_generation(&self) -> reims_vgpu_core::identity::SessionGeneration {
        self.session.lock().expect("session").generation()
    }

    /// The positions the model has released to run since the last ask.
    ///
    /// **The one door work leaves the model by.** A transaction taken off this
    /// list and not run is one that never runs, and one taken twice is one that
    /// runs twice; the store that holds its bytes is what makes the second
    /// unrepresentable — see `crate::runtime::parked::ParkedStore::release`.
    #[must_use = "a position taken off the ready list and not run is a packet that never runs"]
    pub fn take_ready(&self) -> Vec<reims_vgpu_core::identity::IngressOrdinal> {
        self.session.lock().expect("session").take_ready()
    }

    /// A transaction finished on the host.
    ///
    /// Releases its dependents, retires its accesses, and hands back what its
    /// channel published — which is not necessarily this transaction's own
    /// word: a channel publishes in its own order, so a completion may release
    /// a queue of earlier words, this one, or nothing at all yet.
    ///
    /// `epoch` is the incarnation the work was *submitted* under, which is why
    /// it is retained with the packet rather than read here. Submission is not
    /// completion: a device loss withdraws every transaction admitted into the
    /// lost epoch, and the host can still report those back.
    ///
    /// # Errors
    ///
    /// If the completion was produced under an incarnation that has ended.
    #[must_use = "what the channel published is what the guest may now read"]
    pub fn complete_transaction(
        &self,
        epoch: reims_vgpu_core::identity::DeviceEpoch,
        ingress: reims_vgpu_core::identity::IngressOrdinal,
    ) -> Result<Vec<reims_vgpu_core::publish::Release>, reims_vgpu_core::session::Refusal> {
        self.session
            .lock()
            .expect("session")
            .complete(epoch, ingress)
    }

    /// Take a transaction that will never publish out of every plane holding
    /// it, and say what its channel released behind it.
    ///
    /// Its own completion word is deliberately not published: the work never
    /// ran, and what the guest is owed instead is the typed reason the caller
    /// names.
    #[must_use = "what the channel published is what the guest may now read"]
    pub fn withdraw_transaction(
        &self,
        ingress: reims_vgpu_core::identity::IngressOrdinal,
    ) -> Vec<reims_vgpu_core::publish::Release> {
        self.session.lock().expect("session").withdraw(ingress)
    }

    /// The guest's completion word for `slot` has moved to `value`.
    ///
    /// The other end of a stamp wait. `SessionModel::admit` parks a packet
    /// whose header names a point another packet must have published, and the
    /// only thing that discharges that wait is this — so a cutover without it
    /// would park every packet the guest orders behind a fence, forever, the
    /// same way an unadvanced pipeline would park every exec.
    ///
    /// Answers with what the write did to the slot's timeline, which is four
    /// different facts and not one. See [`StampPublication`].
    pub fn publish_completion_stamp(
        &self,
        slot: reims_vgpu_core::identity::StampSlot,
        value: reims_vgpu_core::identity::StampValue,
    ) -> StampPublication {
        let mut session = self.session.lock().expect("session");
        let before = session.published_stamp(slot);
        session.stamp_published(reims_vgpu_core::identity::CompletionStamp { slot, value });
        let after = session.published_stamp(slot);
        match before {
            None => StampPublication::First,
            Some(before) if after == Some(value) && before != value => StampPublication::Advanced,
            Some(before) if before == value => StampPublication::Repeat,
            Some(held) => StampPublication::Behind { held },
        }
    }

    /// What point the ordering plane has the timeline for `slot` standing at,
    /// or `None` if nothing ever published to it.
    #[must_use]
    pub fn published_completion_stamp(
        &self,
        slot: reims_vgpu_core::identity::StampSlot,
    ) -> Option<reims_vgpu_core::identity::StampValue> {
        self.session.lock().expect("session").published_stamp(slot)
    }

    /// What the ordering plane holds about the pipelines this session declared.
    #[must_use]
    pub fn pipeline_census(&self) -> reims_vgpu_core::pipeline::Census {
        self.session.lock().expect("session").pipelines().census()
    }

    /// The census line for what the ordering plane's pipeline table is holding
    /// right now, once a second, or `None` between windows.
    ///
    /// # Why occupancy is emitted and not only the event counts
    ///
    /// `store_routes` counts what *happened* — declarations, refusals,
    /// retirements — and a pipeline that was declared and never advanced is
    /// counted once there and never again. A table quietly accumulating
    /// pipelines nothing will ever build therefore reads exactly like a
    /// healthy one, and the difference is a subtraction across counters that
    /// nobody performs while watching a live boot.
    ///
    /// It is the same argument `backing_outstanding_census` and
    /// `slot_recheck::outstanding_census` are emitted beside `store_routes`
    /// for, and it matters more here: `pending` is the pipelines a transaction
    /// can be *waiting* on, so a `pending` that does not fall is work parked on
    /// a build nobody is running, and a rising one is a hang forming — visible
    /// before the guest stops drawing rather than after.
    ///
    /// Quiet while the table is empty, so a boot that never declares one is
    /// not a line a second saying so.
    pub fn pipeline_occupancy_census(&self) -> Option<String> {
        const WINDOW_MS: u64 = 1000;
        static LAST_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let now = crate::observe::elapsed_ms() as u64;
        let last = LAST_MS.load(Ordering::Relaxed);
        if now.saturating_sub(last) < WINDOW_MS {
            return None;
        }
        let resting = self.session.lock().expect("session").pipelines().resting();
        if resting.total() == 0 {
            return None;
        }
        LAST_MS.store(now, Ordering::Relaxed);
        Some(format!(
            "pipeline_table pending={} declared={} translating={} compiling={} ready={} \
             refused={} retired={} total={}",
            resting.pending(),
            resting.declared,
            resting.translating,
            resting.compiling,
            resting.ready,
            resting.refused,
            resting.retired,
            resting.total(),
        ))
    }

    /// The open child domains as the bit mask this device's registers speak in.
    ///
    /// Derived on each ask rather than mirrored into a field: a mirror is the
    /// second record of channel openness that moving this lifetime was for. The
    /// walk is over this device's own channel bound rather than over the model's
    /// set, so a domain outside the FIFO range cannot set a bit that names
    /// another channel.
    pub fn open_child_mask(&self) -> u32 {
        let mut mask = 0u32;
        for channel in 1..crate::model::MAX_CHANNELS as u32 {
            if self.child_domain_open(channel) {
                mask |= 1u32 << channel;
            }
        }
        mask
    }

    /// Every child FIFO worth looking at: open, or holding work.
    ///
    /// What `active_child_mask | pending.child_mask` used to spell, with the
    /// union derived from its two halves instead of one of them being kept up to
    /// date by hand at three write sites.
    pub fn drainable_child_mask(&self) -> u32 {
        self.open_child_mask() | self.pending.child_mask
    }

    /// What a guest reference names in this task right now, or `None` when the
    /// slot holds nothing live.
    ///
    /// The one way into anything this device keeps per object. A caller holding
    /// a bare reference number has a slot; a caller holding the answer to this
    /// has a *name*, and only a name can key [`TaskResources`].
    ///
    /// Takes `&self`, because resolution happens on the shared-borrow path and
    /// this asks the owner's non-consuming door
    /// (`reims_vgpu_core::resolve::TaskNamespaces`) rather than the one that
    /// hands out leases: a lookup that minted a lease nobody acquired would be a
    /// claim this device never pays off.
    ///
    /// Routing through the owner rather than through one task's namespace is
    /// what makes "which task's slots is this reference in" a question with one
    /// answer. A task the owner does not hold answers `None`, which is the same
    /// answer an empty slot gives and is right for both.
    #[must_use]
    pub fn object_name(&self, task_id: u32, obj_ref: u32) -> Option<ResourceId> {
        use reims_vgpu_core::resolve::TaskNamespaces as _;
        self.lifecycle
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .resource(reims_vgpu_core::identity::TaskId(task_id), obj_ref)
    }

    /// This device as the access source for one task's records, in one
    /// submission domain.
    ///
    /// The counterpart of [`Self::object_name`] for the other half of a
    /// command-stream walk: the resolver says what a reference names, and this
    /// says what a participation over that name *is*.
    ///
    /// # Why it is not [`reims_vgpu_core::lifecycle::Lifecycle::task_access`]
    ///
    /// The owner's own door hands out a `TaskAccess<'_>`, which borrows the
    /// `Lifecycle` mutably for as long as it lives. Reaching it through this
    /// device would mean holding the lifecycle guard across the whole walk, and
    /// a walk is exactly where that cannot be held:
    /// `reims_vgpu_core::walk::segment` resolves a record and then records it,
    /// resolution is `crate::runtime::objects::TaskNames` → [`Self::object_name`]
    /// → the same mutex, and `std::sync::Mutex` is not reentrant. Every exec
    /// packet would hang this device on its first record.
    ///
    /// It would also not typecheck: `resolve::TaskNamespaces::resource` takes
    /// `&self` and `access::AccessSource::access` takes `&mut self`, and
    /// ingress wants a resolver and an access source alive at the same time.
    /// One `&mut Lifecycle` cannot back both.
    ///
    /// So this locks **per call**, the way [`Self::object_name`] already does,
    /// and holds nothing between calls. That is not the snapshot
    /// `TaskAccess`'s own doc argues against: every call reaches the same
    /// owner, so the content authority's version reservations accumulate across
    /// a transaction exactly as they do through a held borrow. What is given up
    /// is nothing; what is bought is that the resolver and the access source
    /// can never hold the lock at the same moment.
    #[must_use]
    pub const fn task_access(
        &self,
        task: reims_vgpu_core::identity::TaskId,
        domain: reims_vgpu_core::identity::ChannelId,
    ) -> DeviceAccess<'_> {
        DeviceAccess {
            state: self,
            task,
            domain,
        }
    }

    /// The resource this device constructed for a guest reference, if it has
    /// one that still resolves.
    ///
    /// **The only door to [`Self::task_resources`], and it is two steps for a
    /// reason.** The memo is keyed by a name; a caller holding a bare reference
    /// has a slot; and the step between them is the namespace saying whether
    /// that slot still names anything. A lookup that skipped it would be reading
    /// the memo of whatever used to live there.
    #[must_use]
    pub fn constructed_object(&self, task_id: u32, obj_ref: u32) -> Option<Arc<TaskResource>> {
        self.task_resources
            .get(task_id, self.object_name(task_id, obj_ref)?)
    }

    /// Publish an object into a task's namespace and take its name.
    ///
    /// `storage` is the whole of what this device could establish about the
    /// object's bytes, in the lifecycle owner's own vocabulary.
    /// `Storage::NoBytes` is an object that owns no memory, which is most of a
    /// list — see `crate::runtime::objects::declared_storage`. An object whose
    /// storage this device *cannot describe* is not expressible here at all and
    /// never reaches this door: that distinction is the caller's to make and to
    /// count, because `NoBytes` is a claim and "I could not tell" is not.
    ///
    /// # Errors
    ///
    /// The owner's refusal, unchanged. `NoSuchTask` is the reachable one — a
    /// declaration into a task no `CmdDefineTask2` opened — and a driven boot
    /// of 3902 declarations read zero of them before this door moved.
    pub fn declare_object(
        &self,
        task_id: u32,
        obj_ref: u32,
        storage: reims_vgpu_core::lifecycle::Storage,
    ) -> Result<Declaration, reims_vgpu_core::lifecycle::Refusal> {
        use reims_vgpu_core::identity::TaskId;
        use reims_vgpu_core::lifecycle::LifecycleOp;
        use reims_vgpu_core::resolve::TaskNamespaces as _;
        let task = TaskId(task_id);
        let mut owner = self.lifecycle.lock().unwrap_or_else(|e| e.into_inner());
        // The whole `Storage` reaches the owner, extent included. This door used
        // to flatten it to the backing a namespace slot records, because a
        // namespace was all this device held; the owner holds the content
        // authority too, and the extent is what that needs — a dedicated
        // resource's own pages are authoritative over exactly its bytes, and a
        // flattened declaration would claim the whole backing for a resource
        // that is a window of one.
        let effects = owner.apply(&LifecycleOp::CreateResource {
            task,
            slot: ObjectListRef(obj_ref),
            storage,
        })?;
        // Asked of the owner that just published it, under the same lock, so
        // this is the name this declaration minted rather than one a later
        // redeclaration replaced it with. `apply` returns the operation's
        // effects and not its name — no other operation has one — so the second
        // read is the join, and the lock is what makes the pair one event.
        let id = owner
            .resource(task, obj_ref)
            .expect("the declaration published this slot");
        drop(owner);
        Ok(Declaration {
            id,
            acted: note_lifecycle_effects(effects, "declare_object"),
        })
    }

    /// Retire a name: stop it resolving, and leave accepted work alone.
    ///
    /// `None` when the slot held nothing this device had named — the guest
    /// deleting an object it never made this device construct, which is
    /// ordinary and not a refusal worth a type here. The owner's `NoSuchTask`
    /// collapses into the same `None` for the same reason: a name in a task
    /// that does not exist is a name that resolves to nothing.
    #[must_use]
    pub fn retire_object_name(&self, task_id: u32, name: ResourceId) -> Option<Teardown> {
        use reims_vgpu_core::lifecycle::LifecycleOp;
        let effects = self
            .lifecycle
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .apply(&LifecycleOp::DeleteResource {
                task: reims_vgpu_core::identity::TaskId(task_id),
                object_ref: name.slot.0,
                resource: Some(name),
            })
            .ok()?;
        // Exactly one, because a delete names one resource. Taken as the
        // caller's answer rather than counted away: `delete_object` routes on
        // which teardown it is, and that answer has one source now.
        note_lifecycle_effects(effects, "retire_object_name")
            .teardowns
            .into_iter()
            .next()
    }

    /// Drop a task's whole object namespace.
    ///
    /// Every name in it ends with the address space, which is the same event
    /// that ends the storage those names resolved to — see
    /// [`Self::bump_task_storage_incarnations`]. The owner performs both: the
    /// per-name teardowns arrive as effects, and this device's own per-name
    /// caches are dropped by the two callers around it.
    ///
    /// A task the owner does not hold is not an error here. It is the ordinary
    /// case at a first definition, where there is no previous address space to
    /// end.
    fn delete_task_namespace(&self, task_id: u32) {
        use reims_vgpu_core::lifecycle::LifecycleOp;
        let applied = self
            .lifecycle
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .apply(&LifecycleOp::DeleteTask {
                task: reims_vgpu_core::identity::TaskId(task_id),
            });
        if let Ok(effects) = applied {
            let acted = note_lifecycle_effects(effects, "delete_task_namespace");
            crate::runtime::drain::note_store_route_n(
                "task_namespace_teardowns",
                acted.teardowns.len() as u64,
            );
        }
    }

    /// Apply one lifetime command to the semantic model and take the work it
    /// obliged.
    ///
    /// **The door the nine wire-borne lifetime commands go through**, beside the
    /// three that have doors of their own because this device reaches them from
    /// somewhere other than a packet — a declaration is produced by resolution,
    /// and a task definition and deletion are reached from the device's own
    /// task table as well as from the wire.
    ///
    /// `None` is a refusal, reported on the always-on channel and named. Every
    /// one of the four the owner can make is either measured at zero on a driven
    /// guest or unreachable by construction:
    ///
    /// * `NoSuchTask` — 14 644 lifetime commands on a driven boot, none of them
    ///   into a task no `CmdDefineTask2` opened. See the register's live
    ///   validation table.
    /// * `Namespace` — every command that names a resource resolved that name
    ///   through this same owner a moment earlier, and
    ///   `resolve::TaskNamespaces::resource` answers `Some` only for a slot live
    ///   at its own generation, which is exactly what `Namespace::resolve`
    ///   accepts.
    /// * `Heap` and `PlacedResourceHasNoPhysical` — this device declares no heap
    ///   placement at all, so the owner's per-task heaps are empty and no
    ///   resident is `Placed`.
    ///
    /// A refusal is therefore a statement that one of those three arguments has
    /// stopped holding, which is worth a line rather than a dropped packet.
    #[must_use = "the work a lifetime command obliged is work nothing else does"]
    pub fn apply_lifetime(
        &self,
        op: &reims_vgpu_core::lifecycle::LifecycleOp,
        site: &'static str,
    ) -> Option<Acted> {
        match self
            .lifecycle
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .apply(op)
        {
            Ok(effects) => Some(note_lifecycle_effects(effects, site)),
            Err(refusal) => {
                crate::runtime::drain::note_store_route(refusal.slug());
                if crate::observe::first_sight(
                    "lifetime_command_refused",
                    (u64::from(op.task().0) << 16) | u64::from(op.kind() as u16),
                ) {
                    crate::observe::fail(format!(
                        "lifetime_command_refused kind={} task={} refusal={} (the semantic \
                         model refused a lifetime packet this device would have acted on)",
                        op.kind().name(),
                        op.task().0,
                        refusal.slug()
                    ));
                }
                None
            }
        }
    }

    /// Open a task's address space in the lifecycle owner, and say what the
    /// definition replaced.
    ///
    /// `None` at a first definition. `Some` when a live task was redefined
    /// under the same id, carrying the owner's own answer about whether the
    /// page-table root moved — which is the question that decides whether what
    /// the guest published into the old space is still there, and it is read
    /// from the owner rather than re-derived beside it.
    fn define_task_namespace(
        &self,
        task_id: u32,
        directory_pfn: u32,
    ) -> Option<reims_vgpu_core::lifecycle::Redefinition> {
        use reims_vgpu_core::lifecycle::LifecycleOp;
        let effects = self
            .lifecycle
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .apply(&LifecycleOp::DefineTask {
                task: reims_vgpu_core::identity::TaskId(task_id),
                // The class bit is not this device's to carry: it holds no
                // registry keyed by it, and the owner's own `kernel` field
                // exists so the kernel task and user task zero are two
                // registrations of slot zero rather than one.
                kernel: task_id == 0,
                directory: reims_vgpu_core::identity::DirectoryFrame(directory_pfn),
            })
            .expect("a definition opens the task it names");
        let mut acted = note_lifecycle_effects(effects, "define_task_namespace");
        crate::runtime::drain::note_store_route_n(
            "task_namespace_teardowns",
            acted.teardowns.len() as u64,
        );
        acted.redefined.pop()
    }

    pub fn delete_object(&mut self, task_id: u32, ref_: u32) -> bool {
        let removed = self.objects.remove(&(task_id, ref_));
        // The namespace first, because it is what decides the reference still
        // names anything; the memo behind that name goes with it. A name it has
        // retired cannot be spelled again, so the memo would be unreachable
        // either way — it is dropped here to free the bytes promptly rather than
        // because correctness rests on it, which is the same relationship
        // `MappingEntry::guest_write_token` has to its generation.
        let resource_removed = match self.object_name(task_id, ref_) {
            Some(name) => {
                let teardown = self.retire_object_name(task_id, name);
                crate::runtime::drain::note_store_route(match teardown {
                    Some(Teardown::Now { .. }) => "object_teardown_now",
                    Some(Teardown::WhenUsesRetire { .. }) => "object_teardown_owed",
                    Some(Teardown::HeldByAnotherName { .. }) => "object_teardown_held_by_peer",
                    Some(Teardown::NoStorage) => "object_teardown_no_storage",
                    None => "object_teardown_unnamed",
                });
                self.task_resources.delete(task_id, name)
            }
            None => false,
        };
        if removed || resource_removed {
            self.invalidate_object_host_copies(task_id, ref_);
            self.texture_to_mapping.remove(&(task_id, ref_));
            // The incarnation is deliberately not advanced here. Releasing a
            // *name* is not a statement about storage: other storage at this
            // reference is a different window and is distinct already, and the
            // same storage back under the same window is the same backing.
            // See `storage_incarnation`.
        }
        removed || resource_removed
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
    /// Contents only. The object stays alive, and so does its `texture_to_mapping`
    /// association — a re-point moves the bytes, it does not unname the resource.
    /// [`Self::delete_object`] takes both halves and calls this for its first.
    ///
    /// A live linear resident goes through [`Self::retire_linear_resident`], so
    /// it is unpinned and its deferred window dropped rather than left to write
    /// pixels read from the old pages into the new ones.
    pub fn invalidate_object_host_copies(&mut self, task_id: u32, ref_: u32) -> (bool, bool) {
        let had_texture = self
            .host_texture_surfaces
            .remove(&(task_id, ref_))
            .is_some();
        let had_linear = match self.host_linear_textures.remove(&(task_id, ref_)) {
            Some(e) => {
                self.retire_linear_resident(task_id, ref_, &e);
                true
            }
            None => false,
        };
        (had_texture, had_linear)
    }

    /// Which incarnation of the pages behind a guest-VA window in this task
    /// this is.
    ///
    /// The other half of [`MappingEntry::map_generation`]. Storage this device
    /// reaches through a mapping has that counter; storage named only by an
    /// address in a task — a plain object-list reference, resolved through the
    /// task's page directory — had none, and a canonical backing identity built
    /// from the address window alone would then be *wrong in the dangerous
    /// direction*. A physical replacement re-points the same guest-virtual
    /// window at different host frames, work already accepted was planned
    /// against the old frames and must keep reading them, and two incarnations
    /// sharing an identity would let a claim on the old storage be satisfied by
    /// the new — handing the old frames back under a live reader. So the
    /// identity is a window *and* this value.
    ///
    /// # It is keyed on the window, and it was keyed on the reference
    ///
    /// A re-point packet names a reference and nothing else, so counting per
    /// reference was the shape the packet suggested. It is canonical only if
    /// one window has one live name, and a driven macos-15 boot found that it
    /// does not: two live references in one task named a single 8 294 400-byte
    /// window — 1920×1080×4, the compositor's own scanout allocation. Counting
    /// per reference would give that framebuffer two identities, and a re-point
    /// through one name would leave a claim held under the other still naming
    /// frames that had already been replaced.
    ///
    /// So the re-point resolves its reference to that reference's window and
    /// advances the count there. Both names see it, because both names are the
    /// window.
    ///
    /// # What advances it, and why that list is complete
    ///
    /// One event at the window's own scope, and two at its task's:
    ///
    /// * `CmdReplacePhysical`, which by its own contract says the PFNs under
    ///   this window have already changed. It is the only announcement there
    ///   is — the address, geometry and length are all unchanged.
    /// * [`Self::delete_task`] and [`Self::define_task`] on a redefine, which
    ///   end the task's whole address space: the objects are dropped and a new
    ///   directory root puts different physical pages under the same addresses.
    ///
    /// **Releasing a name is not on the list, and used to be.** It advanced the
    /// count on the reading that whatever the guest puts at that reference next
    /// is other storage — which is true and is already answered by the other
    /// half: other storage is a different window. The same storage back under
    /// the same window is the same backing, and a bump there would have said it
    /// was not. Under per-reference keying the distinction was invisible;
    /// keyed on the window it is the difference between an identity and a
    /// counter that only ever goes up.
    ///
    /// The remaining candidate is the guest overwriting its own object-list
    /// slot in place, which is how objects are replaced on this interface and
    /// which no packet announces. It needs nothing here: an overwrite that
    /// changes the storage changes the descriptor's window with it, and the
    /// window half of the identity separates them. An overwrite that leaves the
    /// window alone names the same pages, which is the same backing — unless
    /// the guest also re-pointed them, and then it emitted the packet above.
    #[must_use]
    pub fn storage_incarnation(&self, task_id: u32, base: u64) -> StorageIncarnation {
        StorageIncarnation {
            epoch: self.task_storage_epochs.get(&task_id).copied().unwrap_or(0),
            count: self
                .storage_incarnations
                .get(&(task_id, base))
                .copied()
                .unwrap_or(0),
        }
    }

    /// Say that the pages behind this guest-VA window may now be different
    /// pages. See [`Self::storage_incarnation`] for the closed list of callers.
    pub fn bump_storage_incarnation(&mut self, task_id: u32, base: u64) {
        let slot = self
            .storage_incarnations
            .entry((task_id, base))
            .or_insert(0);
        *slot = slot.wrapping_add(1);
    }

    /// Say that every name in this task now describes different pages, for the
    /// two events that end a task's address space.
    ///
    /// One counter for the whole task rather than a walk over its names, and
    /// that is not only an optimisation: a name this device never touched has
    /// no per-name entry to bump, so a walk would leave exactly the references
    /// the guest published and never re-pointed — the common case — comparing
    /// equal across a teardown. The epoch covers them without having to have
    /// seen them.
    ///
    /// The per-name counts go with it. They are re-based by the new epoch, so
    /// dropping them cannot make two incarnations collide, and it is what keeps
    /// [`Self::storage_incarnations`] bounded by the live namespace instead of
    /// by how long the device has run.
    fn bump_task_storage_incarnations(&mut self, task_id: u32) {
        let slot = self.task_storage_epochs.entry(task_id).or_insert(0);
        *slot = slot.wrapping_add(1);
        self.storage_incarnations.retain(|&(t, _), _| t != task_id);
        self.backing_window_refs.delete_task(task_id);
        self.aliased_references
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|&(t, _)| t != task_id);
    }

    /// Claim `base` for `ref_` in `task_id`, or say who holds it already.
    ///
    /// Returns the reference that got there first when one did and it is not
    /// this one; `None` when this reference is the first, or is the same
    /// reference re-resolving its own window. See
    /// [`Self::backing_window_refs`] for what the answer decides.
    ///
    /// Takes `&self` because the one caller is a resolve, which holds the
    /// device state shared — the same reason [`TaskResources`] is behind a
    /// lock, and the claim is published under the same race the register there
    /// resolves: first writer wins and every later reader sees that one.
    pub fn claim_backing_window(&self, task_id: u32, ref_: u32, base: u64) -> Option<u32> {
        self.backing_window_refs
            .claim(task_id, ref_, base, self.storage_incarnation(task_id, base))
    }

    /// The canonical identity of the storage behind a guest-VA window in this
    /// task, right now.
    ///
    /// This is what `reims_vgpu_core::access::BackingId` is: two resources over
    /// one piece of storage get one number, and a re-point gives the same
    /// window a different one because it is different storage. It is interned
    /// rather than computed, because the three things that decide it — a
    /// task, a 64-bit address and an incarnation pair — do not fit in the
    /// `u64` the identity is, and a hash of them would trade a guaranteed
    /// distinction for a probable one. False equality here hands storage back
    /// under a live reader.
    ///
    /// The table holds one entry per live window and is cleared with the task,
    /// so it is bounded by the namespace and not by how long the device has
    /// run. A replaced incarnation does not accumulate: the entry moves to the
    /// new identity and the old one stops being mintable, which is correct
    /// because nothing mints an identity it already holds.
    #[must_use]
    pub fn backing_identity(&self, task_id: u32, base: u64) -> BackingId {
        BackingId(self.backing_window_refs.identity(
            task_id,
            base,
            self.storage_incarnation(task_id, base),
        ))
    }

    /// The canonical identity of the storage behind a guest mapping's surface,
    /// right now.
    ///
    /// [`Self::backing_identity`]'s other half, for the storage this device
    /// reaches through a mapping rather than by an address in a task. Same
    /// identity space, same interning, same reason it is interned; the
    /// difference is which incarnation counter it mixes in, and that difference
    /// is not a choice. A `CmdReplacePhysical` naming an object that owns a
    /// mapping does not advance the window counter at all — it drops the page
    /// list and bumps [`MappingEntry::map_generation`], which is the announcement
    /// that these are different pages. An identity derived from the window
    /// would sit still across it, and false equality here hands the old frames
    /// back under a live reader.
    ///
    /// Takes the generation from the caller rather than reading the mapping,
    /// because the caller has already had to find the entry to know there is
    /// one; `crate::runtime::objects::mapping_backing_id` is that caller and is
    /// where a missing mapping is named.
    #[must_use]
    pub fn mapping_backing_identity(&self, mapping_id: u32, map_generation: u32) -> BackingId {
        BackingId(
            self.backing_window_refs
                .mapping_identity(mapping_id, map_generation),
        )
    }

    /// Remember that this reference shares its allocation with another live
    /// one, so a hot path can ask in one lookup instead of a scan.
    ///
    /// Written by [`crate::runtime::objects::note_reference_shares_storage`],
    /// which does the scan once per reference. The set is therefore as fresh as
    /// that sighting: a reference whose *peer* was constructed afterwards is
    /// not in it. That is a floor on what the payment alarm can see and not a
    /// ceiling on the hazard, which is the safe direction for an alarm to be
    /// wrong in — it under-reports rather than crying wolf.
    pub fn note_aliased_reference(&self, task_id: u32, ref_: u32) {
        self.aliased_references
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert((task_id, ref_));
    }

    /// Whether this reference is known to share its allocation with another.
    #[must_use]
    pub fn reference_is_aliased(&self, task_id: u32, ref_: u32) -> bool {
        self.aliased_references
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(&(task_id, ref_))
    }

    /// Move a window's claim to `ref_`, for a holder the guest's own list no
    /// longer says is there.
    ///
    /// The claim table remembers whoever constructed first, and the guest can
    /// free an object without telling this device — so a holder can stop
    /// naming its window with nothing to notice. Handing the window to the
    /// reference that does name it keeps the table's meaning ("who holds this
    /// window") true, and stops the next claimant being compared against a
    /// reference that has been gone for a thousand frames.
    pub fn take_backing_window(&self, task_id: u32, ref_: u32, base: u64) {
        self.backing_window_refs
            .take(task_id, ref_, base, self.storage_incarnation(task_id, base));
    }

    /// Bump [`MappingEntry::map_generation`] (never 0 after first bump).
    ///
    /// The bump orphans any generation-keyed resident for the mapping.
    pub fn bump_map_generation(e: &mut MappingEntry) {
        e.map_generation = e.map_generation.wrapping_add(1);
        if e.map_generation == 0 {
            e.map_generation = 1;
        }
    }

    /// Drop compute storage-residency mirror entries whose byte window
    /// `[surface_offset, span_end)` intersects a guest write of
    /// `[lo, hi)` on this mapping. The mirror claims "guest pages still hold
    /// exactly the resident's content for this window" — any intersecting
    /// write breaks that claim; disjoint windows (ping-pong canvases) survive.
    pub fn invalidate_storage_residency_window(&mut self, mapping_id: u32, lo: u64, hi: u64) {
        self.compute_storage_residency.retain(|key, _| {
            key.mapping_id != mapping_id || key.span_end <= lo || key.surface_offset >= hi
        });
    }

    /// Drop cached page list + contig view without unmapping the slot.
    ///
    /// Used on ReplacePhysical / rebind: guest may have recycled PFNs into the
    /// zone freelist; the next Store must re-resolve before any host write or
    /// import-present DMA (freelist `0xff000000ff000000` class).
    pub fn invalidate_mapping_pages(&mut self, mapping_id: u32) -> bool {
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
        // than theoretical. The backing sampled ladder's host-cache rung serves
        // its copy unless the witness reports `Wrote`, and a retired token
        // reports `NoStamp` — deliberately not evidence, because "nobody armed
        // this" is a statement about this device and not about the guest. The
        // rung therefore reads the invalidation as *permission* to serve, and
        // keeps serving until some later Store replaces the bytes. A surface
        // composited once and then only sampled — a popup backdrop, a settings
        // pane — never gets that Store, so the stale frame is held for the life
        // of the guest.
        //
        // `condemn_surface_backing` already drops it for the same class of
        // event, and the two sit in the same `if`/`else` in the ReplacePhysical
        // teardown, so leaving it here made one arm of one decision correct.
        if self.host_surfaces.remove(&mapping_id).is_some() {
            crate::runtime::drain::note_store_route("invalidate_dropped_host_cache");
        }
        let Some(e) = self.mappings.get_mut(&mapping_id) else {
            return false;
        };
        let had = !e.page_entries.is_empty() || e.contig_ptr != 0;
        e.page_entries.clear();
        e.page_table_kva = 0;
        e.condemned_entries = None;
        Self::bump_map_generation(e);
        let (retired, retired_import) = Self::take_mapping_view(e);
        let retired_token = Self::take_guest_write_token(e);
        if let Some(v) = retired {
            self.retired_views.push(v);
        }
        if let Some(import) = retired_import {
            self.retired_guest_imports.push(import);
        }
        if retired_token != 0 {
            self.retired_guest_write_tokens.push(retired_token);
        }
        had
    }

    /// Trailing `DeleteIOSurfaceBacking2`: retire the page bindings — nothing
    /// may write through possibly-recycled pages (boot-16 PTE-corruption
    /// rule) — but KEEP content state (map_generation, geometry, resident
    /// identity, deferred windows). The deleted backing may belong to a PRIOR
    /// incarnation of a recycled id whose slot already carries a live surface
    /// with an unflushed paint (black-band class): the next page resolve
    /// compares against the stashed fingerprint and either reprieves (same
    /// plan) or bumps + drops (different plan). Returns whether a fingerprint
    /// was stashed; on `false` the caller should fall back to full teardown.
    pub fn condemn_surface_backing(&mut self, mapping_id: u32) -> bool {
        self.forget_compositor_mapping(mapping_id);
        self.host_surfaces.remove(&mapping_id);
        let Some(e) = self.mappings.get_mut(&mapping_id) else {
            return false;
        };
        if e.page_entries.is_empty() {
            return false;
        }
        e.condemned_entries = Some(std::mem::take(&mut e.page_entries));
        e.page_table_kva = 0;
        let (retired, retired_import) = Self::take_mapping_view(e);
        let retired_token = Self::take_guest_write_token(e);
        if let Some(v) = retired {
            self.retired_views.push(v);
        }
        if let Some(import) = retired_import {
            self.retired_guest_imports.push(import);
        }
        if retired_token != 0 {
            self.retired_guest_write_tokens.push(retired_token);
        }
        true
    }

    /// Whether `mapping_id` sits in the condemned state (backing deleted, no
    /// resolve since). A second delete in this state is genuinely dead — the
    /// caller tears down for real.
    pub fn mapping_backing_condemned(&self, mapping_id: u32) -> bool {
        self.mappings
            .get(&mapping_id)
            .is_some_and(|e| e.condemned_entries.is_some())
    }

    pub fn map_surface(&mut self, mapping_id: u32) -> bool {
        self.max_mapping_id_seen = self.max_mapping_id_seen.max(mapping_id);
        if !crate::model::is_mapping_id(mapping_id) {
            StateMutationDecline::MapSurfaceIdSentinel { mapping_id }.emit(u64::from(mapping_id));
            return false;
        }
        let e = self.mappings.entry(mapping_id).or_default();
        e.mapped = true;
        // Fresh MAP invalidates any previous page table / geom for this slot.
        // Stale has_geom after 1920→1440 remap blocks writebacks (size mismatch)
        // and freezes host console at the old mode. The MAP notify often TRAILS
        // our eager resolve of the same surface (a Store discovers the mapping
        // before the guest's notification drains) — so never bump eagerly:
        // stash the page fingerprint and let the next resolve decide (same
        // plan = same incarnation, generation and deferred windows survive;
        // different plan = genuine new surface, bump + drop there). Geometry
        // stays cleared either way — samples fail-closed until re-resolve, so
        // a genuinely new surface can never be served the old resident.
        if !e.page_entries.is_empty() && e.condemned_entries.is_none() {
            e.condemned_entries = Some(std::mem::take(&mut e.page_entries));
        } else {
            e.page_entries.clear();
        }
        e.page_table_kva = 0;
        e.device_desc.clear();
        e.content_generation = 0;
        e.surface_content_epoch = 0;
        e.has_geom = false;
        e.width = 0;
        e.height = 0;
        e.format = 0;
        let (retired, retired_import) = Self::take_mapping_view(e);
        let retired_token = Self::take_guest_write_token(e);
        if let Some(v) = retired {
            self.retired_views.push(v);
        }
        if let Some(import) = retired_import {
            self.retired_guest_imports.push(import);
        }
        if retired_token != 0 {
            self.retired_guest_write_tokens.push(retired_token);
        }
        // Fresh MAP: prior host-cache for this surface_id is stale, and so is
        // any present evidence — the slot may hold a NEW surface.
        self.host_surfaces.remove(&mapping_id);
        // Present evidence is stamped with the incarnation and deliberately NOT
        // dropped here. A fresh MAP does not yet know whether this is a new
        // surface — that is what the fingerprint compare decides, bumping the
        // generation when it is. Dropping it eagerly demoted a proven swapchain
        // buffer to a private resident for every draw until its next present,
        // which is the black-desktop class.
        true
    }

    pub fn unmap_surface(&mut self, mapping_id: u32) -> bool {
        self.max_mapping_id_seen = self.max_mapping_id_seen.max(mapping_id);
        if !crate::model::is_mapping_id(mapping_id) {
            StateMutationDecline::UnmapSurfaceIdSentinel { mapping_id }.emit(u64::from(mapping_id));
            return false;
        }
        self.forget_compositor_mapping(mapping_id);
        if let Some(e) = self.mappings.get_mut(&mapping_id) {
            e.mapped = false;
            // The surface is gone; a recycled id has presented nothing yet.
            e.scanout_presented = false;
            e.page_entries.clear();
            e.page_table_kva = 0;
            e.condemned_entries = None;
            e.mapping_internal = 0;
            e.device_desc.clear();
            Self::bump_map_generation(e);
            e.has_geom = false;
            e.width = 0;
            e.height = 0;
            e.format = 0;
            let (retired, retired_import) = Self::take_mapping_view(e);
            let retired_token = Self::take_guest_write_token(e);
            if let Some(v) = retired {
                self.retired_views.push(v);
            }
            if let Some(import) = retired_import {
                self.retired_guest_imports.push(import);
            }
            if retired_token != 0 {
                self.retired_guest_write_tokens.push(retired_token);
            }
            self.host_surfaces.remove(&mapping_id);
            true
        } else {
            false
        }
    }

    /// Attach directed MappingInternal capture to a mapped slot.
    pub fn attach_mapping_internal(&mut self, mapping_id: u32, mapping_internal: u64) -> bool {
        self.max_mapping_id_seen = self.max_mapping_id_seen.max(mapping_id);
        if !crate::model::is_mapping_id(mapping_id) {
            StateMutationDecline::AttachMappingIdSentinel { mapping_id }
                .emit(u64::from(mapping_id));
            return false;
        }
        if mapping_internal == 0 {
            StateMutationDecline::AttachMappingInternalZero { mapping_id }
                .emit(u64::from(mapping_id));
            return false;
        }
        let e = self.mappings.entry(mapping_id).or_default();
        // A re-statement of the SAME MappingInternal (notify trailing our
        // eager resolve) is not a new surface: keep bindings, generation,
        // resident, and deferred windows untouched.
        if e.mapping_internal == mapping_internal {
            e.mapped = true;
            return true;
        }
        e.mapped = true;
        e.mapping_internal = mapping_internal;
        e.page_entries.clear();
        e.page_table_kva = 0;
        e.condemned_entries = None;
        e.device_desc.clear();
        e.content_generation = 0;
        e.surface_content_epoch = 0;
        Self::bump_map_generation(e);
        // New MappingInternal ⇒ new surface; force device-desc re-resolve.
        e.has_geom = false;
        e.width = 0;
        e.height = 0;
        e.format = 0;
        let (retired, retired_import) = Self::take_mapping_view(e);
        let retired_token = Self::take_guest_write_token(e);
        if let Some(v) = retired {
            self.retired_views.push(v);
        }
        if let Some(import) = retired_import {
            self.retired_guest_imports.push(import);
        }
        if retired_token != 0 {
            self.retired_guest_write_tokens.push(retired_token);
        }
        // New MappingInternal ⇒ new surface, and the `bump_map_generation`
        // above is what retires the stale present evidence: it is stamped with
        // the incarnation that recorded it, so the recycled slot cannot inherit
        // a display-plane qualification it did not earn.
        true
    }

    /// Cache the 0x200-byte guest device descriptor for plane/surface sample windows.
    pub fn set_mapping_device_desc(&mut self, mapping_id: u32, desc: &[u8]) -> bool {
        self.max_mapping_id_seen = self.max_mapping_id_seen.max(mapping_id);
        if !crate::model::is_mapping_id(mapping_id) {
            StateMutationDecline::MappingDeviceDescIdSentinel { mapping_id }
                .emit(u64::from(mapping_id));
            return false;
        }
        if desc.is_empty() {
            StateMutationDecline::MappingDeviceDescEmpty { mapping_id }.emit(u64::from(mapping_id));
            return false;
        }
        let e = self.mappings.entry(mapping_id).or_default();
        e.device_desc = desc.to_vec();
        true
    }

    pub fn set_mapping_geom(
        &mut self,
        mapping_id: u32,
        width: u32,
        height: u32,
        format: u16,
    ) -> bool {
        if !crate::model::is_mapping_id(mapping_id) {
            StateMutationDecline::MappingGeomIdSentinel { mapping_id }.emit(u64::from(mapping_id));
            return false;
        }
        // The bound itself lives once, in `regs::scanout_extent_fault`; this is
        // the only caller that has to name which half of it broke, so it is the
        // only one that reads the fault rather than the verdict.
        if let Some(fault) = crate::model::scanout_extent_fault(width, height) {
            use crate::model::ScanoutExtentFault as F;
            match fault {
                F::WidthZero => StateMutationDecline::MappingGeomWidthZero { mapping_id }
                    .emit(u64::from(mapping_id)),
                F::HeightZero => StateMutationDecline::MappingGeomHeightZero { mapping_id }
                    .emit(u64::from(mapping_id)),
                F::WidthAboveBound => {
                    StateMutationDecline::MappingGeomWidthRange { mapping_id, width }
                        .emit((u64::from(mapping_id) << 32) | u64::from(width))
                }
                F::HeightAboveBound => {
                    StateMutationDecline::MappingGeomHeightRange { mapping_id, height }
                        .emit((u64::from(mapping_id) << 32) | u64::from(height))
                }
            }
            return false;
        }
        let e = self.mappings.entry(mapping_id).or_default();
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
        // declarations onto one `vk::Format` and falls back to the scanout order
        // for any it cannot express, so a mapping going from a format with a
        // linear texel to a compressed or planar one keeps its identity, keeps
        // its resident, and keeps an epoch that was stamped against the old
        // interpretation of the same bytes.
        if e.width != width || e.height != height || e.format != format {
            e.content_generation = 0;
            e.surface_content_epoch = 0;
        }
        e.has_geom = true;
        e.width = width;
        e.height = height;
        e.format = format;
        true
    }

    /// Record that this device is about to write pixel bytes into guest RAM.
    ///
    /// Called from every host-side writer, including the ones that reach guest
    /// pages through a raw task-GVA walk and never name a mapping. The
    /// hypervisor's dirty bitmap cannot see any of them — it witnesses guest CPU
    /// stores only — so without this a reader has no way to tell "nobody wrote
    /// these pages" from "we wrote them ourselves".
    ///
    /// Deliberately called before the write rather than after it succeeds: a
    /// refused write costs a spurious bump, which makes a reader re-read bytes
    /// that did not change. The opposite error hands out a stale copy.
    pub fn note_host_wrote_guest_ram(&mut self) {
        self.host_writes.note_unknown();
    }

    /// The same, for a writer that walked the guest page tables and so knows
    /// exactly which pages it landed in even though it names no mapping.
    pub fn note_host_wrote_pages(&mut self, pages: Vec<u64>) {
        self.host_writes.note_pages(pages);
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
        let m = self.mappings.get(&mapping_id)?;
        if m.page_entries.is_empty() {
            return None;
        }
        let shift = self.page_shift;
        m.page_entries
            .iter()
            .map(|&e| crate::protocol::iosurface_pages::entry_gpa_shift(e, shift))
            .collect()
    }

    /// The same, for a writer that knows which mapping's pages it is landing in.
    pub fn note_host_wrote_mapping(&mut self, mapping_id: u32) {
        let Some(entries) = self
            .mappings
            .get(&mapping_id)
            .map(|mapping| mapping.page_entries.as_slice())
            .filter(|entries| !entries.is_empty())
        else {
            self.host_writes.note_unknown();
            return;
        };
        let shift = self.page_shift;
        if entries
            .iter()
            .any(|&entry| crate::protocol::iosurface_pages::entry_gpa_shift(entry, shift).is_none())
        {
            // A mapping whose pages cannot be named exactly cannot have its
            // write ruled out later. Record one unnamed write rather than the
            // resolvable prefix.
            self.host_writes.note_unknown();
            return;
        }
        self.host_writes
            .note_page_iter(entries.iter().map(|&entry| {
                crate::protocol::iosurface_pages::entry_gpa_shift(entry, shift)
                    .expect("page entries were validated above")
            }));
    }

    /// Issue a sampled-content generation that has never been issued before.
    ///
    /// Every producer of a sampled-content identity must take its generation
    /// from here and nowhere else. The value is what the engine's sampled
    /// cache binds on without looking at a single byte, so "never issued
    /// before" is the whole of the contract — see
    /// [`Self::sampled_content_gen`]. Never returns 0, which readers use for
    /// "no host content yet".
    pub fn next_sampled_content_generation(&mut self) -> u64 {
        self.sampled_content_gen = self.sampled_content_gen.wrapping_add(1);
        if self.sampled_content_gen == 0 {
            self.sampled_content_gen = 1;
        }
        self.sampled_content_gen
    }

    /// Issue the next recency stamp for [`HostSurface::last_touch`].
    ///
    /// Strictly increasing, so the smallest stamp in
    /// [`Self::host_gva_surfaces`] is always the coldest entry and the byte cap
    /// needs no other ordering. Saturating rather than wrapping: a wrap would
    /// make one ancient entry look like the newest and pin it forever, and at
    /// one stamp per lookup `u64::MAX` is not reachable by any real session.
    pub fn next_gva_touch(&mut self) -> u64 {
        self.gva_touch_seq = self.gva_touch_seq.saturating_add(1);
        self.gva_touch_seq
    }

    /// Bump content generation after a write into the mapping (0 never skips).
    ///
    /// Also advances [`MappingEntry::surface_content_epoch`], so every one of
    /// this crate's guest-page writers keeps that epoch closed for free — the
    /// completeness property the mapper-ref-texture `LoadFromTarget` gate rests on.
    pub fn mark_mapping_written(&mut self, mapping_id: u32) -> u32 {
        let seq = self.next_validity_seq();
        let Some(m) = self.mappings.get_mut(&mapping_id) else {
            return 0;
        };
        m.content_generation = m.content_generation.wrapping_add(1);
        if m.content_generation == 0 {
            m.content_generation = 1;
        }
        m.surface_content_epoch = Self::next_epoch(m.surface_content_epoch);
        m.validity.host_published_seq = seq;
        m.content_generation
    }

    /// Next value of the device-wide ordering counter behind
    /// [`ResourceValidity::host_cleared_seq`] / `host_published_seq`.
    ///
    /// One counter for both sides on purpose: the only question either stamp is
    /// ever asked is which of the two happened last, and two counters cannot
    /// answer that. Starts at 1 so a stamp is always distinguishable from the
    /// `0` default that means "this never happened".
    pub fn next_validity_seq(&mut self) -> u64 {
        self.validity_seq = self.validity_seq.saturating_add(1);
        self.validity_seq
    }

    /// Advance a mapping's content stamps for a publish that changed its pixels
    /// *without* writing its guest pages — the lazy mapper-ref-texture Store of
    /// [`crate::runtime::writeback_debt`], which leaves the frame in the engine
    /// resident and owes the pages a copy.
    ///
    /// Returns the new [`MappingEntry::surface_content_epoch`] so the caller can
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
        let seq = self.next_validity_seq();
        let Some(m) = self.mappings.get_mut(&mapping_id) else {
            return 0;
        };
        m.surface_content_epoch = Self::next_epoch(m.surface_content_epoch);
        // The pixels this publishes are newer than anything the guest claimed
        // before now, which is what a deferred writeback later has to know.
        m.validity.host_published_seq = seq;
        m.surface_content_epoch
    }

    /// Wrapping increment that never lands on 0, so 0 keeps meaning "no content
    /// published since attach" and cannot be matched by a resident's own
    /// unstamped default.
    fn next_epoch(epoch: u32) -> u32 {
        match epoch.wrapping_add(1) {
            0 => 1,
            n => n,
        }
    }

    pub fn record_fail(&mut self, ev: FailEvent) {
        // Fail-visible (I2): decode/contract gaps must reach the always-on fail
        // log, not only the in-memory test vec — silently dropped commands
        // (e.g. unknown display-channel opcodes) otherwise leave no trace in a
        // live boot.
        //
        // Through `Emit` rather than `format!("{ev:?}")`: the debug rendering
        // carried the same facts but spelled them `MalformedRootPacket { reason:
        // "bad-packet-size", head: 4096 }`, which is neither `reason=<slug>` nor
        // greppable by the vocabulary every other subsystem uses.
        crate::observe::Emit::decline("fail_event", &ev).fail();
        #[cfg(test)]
        self.fails.push(ev);
    }

    /// [`Self::record_fail`], but the line is emitted only the first time this
    /// `(reason, discriminant)` pair is seen this boot.
    ///
    /// For an event that repeats at the guest's own rate. A refusal the guest
    /// re-triggers every frame does not become more true for being printed
    /// thirty times a second; it becomes unreadable, and takes the rest of the
    /// log with it. The caller pairs this with a route counter, because the
    /// latch is what costs the rate.
    ///
    /// The in-memory vec is still appended on every call. It is the tests'
    /// view, and a test asserting that a second packet was declined would
    /// otherwise be asserting the latch instead.
    pub fn record_fail_once(&mut self, ev: FailEvent, discriminant: u64) {
        if crate::observe::first_sight(crate::observe::Decline::slug(&ev), discriminant) {
            crate::observe::Emit::decline("fail_event", &ev).fail();
        }
        #[cfg(test)]
        self.fails.push(ev);
        #[cfg(not(test))]
        let _ = ev;
    }
}

/// What one admission established.
///
/// The model's answer and the host device incarnation it was admitted into,
/// taken together because they are read under one lock. The epoch travels with
/// the parked packet and comes back at completion — see
/// [`DeviceState::complete_transaction`] for why it is the submission's fact
/// and not a value to look up later.
pub struct Admission {
    pub admitted: reims_vgpu_core::session::Admitted,
    pub epoch: reims_vgpu_core::identity::DeviceEpoch,
}

/// One task's records, in one submission domain, as an
/// [`reims_vgpu_core::access::AccessSource`] backed by this device.
///
/// A `&DeviceState` and two `Copy` fields — the same shape as
/// `crate::runtime::objects::TaskRefResolver`, and for the same reason: the
/// task and the domain are properties of the *packet*, so binding them once at
/// the door is what stops a per-record caller pairing a participation with the
/// wrong one.
///
/// See [`DeviceState::task_access`] for why this locks per call rather than
/// borrowing the owner for the walk.
#[derive(Clone, Copy)]
pub struct DeviceAccess<'a> {
    state: &'a DeviceState,
    task: reims_vgpu_core::identity::TaskId,
    domain: reims_vgpu_core::identity::ChannelId,
}

impl DeviceAccess<'_> {
    /// The task whose names this places participations in.
    #[must_use]
    pub const fn task(self) -> reims_vgpu_core::identity::TaskId {
        self.task
    }

    /// The submission domain the accesses are claimed in.
    #[must_use]
    pub const fn domain(self) -> reims_vgpu_core::identity::ChannelId {
        self.domain
    }
}

impl reims_vgpu_core::access::AccessSource for DeviceAccess<'_> {
    /// The owner's answer, with the owner's own refusal slug.
    ///
    /// The mapping from `lifecycle::Refusal` to `access::AccessRefusal` is the
    /// one `lifecycle::TaskAccess` makes, and it is repeated rather than shared
    /// because sharing it would mean constructing the `TaskAccess` this exists
    /// not to construct. It is three lines and one `slug()` call; a helper
    /// taking `&mut Lifecycle` would be the held borrow again.
    ///
    /// # The lock is dropped before anything is said about a refusal
    ///
    /// **A `map_err` chained onto the locked call runs with the guard still
    /// alive**, because the guard is a temporary of the whole statement and not
    /// of the call that produced it. [`note_access_refused`] reads the object
    /// the slot holds, which reaches `DeviceState::object_name` and takes this
    /// same lock — so writing it as one chain deadlocks the device on its first
    /// refused record. It was measured: a driven boot froze with
    /// `walk_records_render = 42` and a window redrawing stale frames for seven
    /// minutes. This is the deadlock `DeviceState::task_access`'s own doc is
    /// about, re-entered from the other side.
    ///
    /// So the answer is taken in its own scope and the guard is gone before the
    /// refusal is described.
    fn access(
        &mut self,
        participation: &reims_vgpu_core::access::Participation,
    ) -> Result<reims_vgpu_core::access::AccessIntent, reims_vgpu_core::access::AccessRefusal> {
        let answer = {
            self.state
                .lifecycle
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .access(self.task, self.domain, participation)
        };
        match answer {
            Ok(intent) => {
                note_access_widened(self.state, self.task, participation, &intent);
                Ok(intent)
            }
            Err(refusal) => {
                note_access_refused(self.state, self.task, participation, refusal);
                Err(reims_vgpu_core::access::AccessRefusal {
                    resource: participation.resource,
                    reason: refusal.slug(),
                })
            }
        }
    }
}

/// Name a record whose window the owner could not hold, and widened.
///
/// **The event is derived and not reported, because the owner has nowhere to
/// report it from.** `Lifecycle::access` widens a `Range` participation that
/// does not fit its resource's extent to the whole of that resource — a bound
/// built to err long must not be checked as if it were exact, see the arm's own
/// doc — and `reims_vgpu_core` has no failure channel to say so on. It does not
/// need one: a caller that asked with a `Range` and was answered with a key that
/// is not a range has been told, and this is the caller.
///
/// It was a refused packet until the owner widened. Seven a boot on a driven
/// macos-15 run, every one a `0x12c` `copyFromBuffer:…toTexture:…` reading
/// `offset=65536 length=196608` of a 196 608-byte source buffer, and each one
/// cost a whole packet of guest work. Counted now rather than counted then,
/// because the number that matters has changed from "packets dropped" to "edges
/// drawn coarser than the record asked for".
fn note_access_widened(
    state: &DeviceState,
    task: reims_vgpu_core::identity::TaskId,
    participation: &reims_vgpu_core::access::Participation,
    intent: &reims_vgpu_core::access::AccessIntent,
) {
    use reims_vgpu_core::access::{AccessKey, ParticipationExtent};
    let ParticipationExtent::Range(asked) = participation.extent else {
        return;
    };
    if matches!(intent.key, AccessKey::Range(..)) {
        return;
    }
    crate::runtime::drain::note_store_route("access_window_widened");
    if crate::observe::first_sight(
        "access_window_widened",
        (u64::from(task.0) << 32) | u64::from(participation.resource.slot.0),
    ) {
        let resident = state
            .lifecycle
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .resolve(task, participation.resource);
        crate::observe::fail(format!(
            "access_window_widened task={} ref={} mode={:?} asked={asked:?} \
             resident={resident:?} rung={} (the record's window is a bound on what it \
             touches and this resource cannot hold it, so the access orders against the \
             whole resource rather than against nothing)",
            task.0,
            participation.resource.slot.0,
            participation.mode,
            intent.key.rung(),
        ));
    }
}

/// Name a refused participation on the always-on channel, with its numbers.
///
/// **The slug alone cannot be acted on here.**
/// [`reims_vgpu_core::access::AccessRefusal`] carries a `&'static str` by
/// design — every refusal in that crate is greppable by the slug of the check
/// that produced it — and a refused access refuses the whole packet its record
/// belongs to, so this is guest work that did not run. A driven macos-15 boot
/// through the ordering model refused seven exec packets on
/// `lifecycle_heap`, which is `Resident::window` saying a record's window ends
/// past its resource's extent; the slug says that much and says nothing about
/// *which* window and *which* extent, and those three numbers are the whole of
/// what decides whether the record is out of range or this device's extent is
/// short.
///
/// First sight per `(task, slot)`, so a record refused every frame names itself
/// once. The owner's refusal is printed as itself rather than re-encoded: the
/// variant carries its own numbers and a second vocabulary here would drift
/// from it.
fn note_access_refused(
    state: &DeviceState,
    task: reims_vgpu_core::identity::TaskId,
    participation: &reims_vgpu_core::access::Participation,
    refusal: reims_vgpu_core::lifecycle::Refusal,
) {
    crate::runtime::drain::note_store_route("access_refused");
    crate::runtime::drain::note_store_route(refusal.slug());
    if crate::observe::first_sight(
        "access_refused",
        (u64::from(task.0) << 32) | u64::from(participation.resource.slot.0),
    ) {
        // The object's own type and descriptor length, because the two things a
        // window refusal can be are a record naming bytes past the end and this
        // device having recovered too short an extent — and which one it is
        // turns on what kind of object the slot holds. `None` when nothing was
        // ever constructed for the slot, which is itself the answer for a
        // participation over a name with no memo behind it.
        let constructed = state.constructed_object(task.0, participation.resource.slot.0);
        // The resource's own range in its backing's coordinates, which is the
        // half `heap::Refusal` does not carry: it reports the *length* the
        // window was checked against and never the offset the extent starts at.
        // Those two numbers are what tell a record naming bytes past the end
        // from a record whose offset is in the allocation's coordinates while
        // `Resident::window` reads it as the object's — the second is exactly a
        // window whose offset equals its resource's own.
        //
        // Asked after the answer's guard is gone, for the reason this function's
        // caller documents.
        let resident = state
            .lifecycle
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .resolve(task, participation.resource);
        crate::observe::fail(format!(
            "access_refused task={} ref={} reason={} mode={:?} extent={:?} refusal={refusal:?} \
             object_type={:?} desc_len={:?} resident={resident:?} (a record's participation \
             could not become an access, which refuses the whole packet it belongs to)",
            task.0,
            participation.resource.slot.0,
            refusal.slug(),
            participation.mode,
            participation.extent,
            constructed.as_ref().map(|r| r.entry.object_type),
            constructed.as_ref().map(|r| r.descriptor.len()),
        ));
    }
}

#[cfg(test)]
mod device_access_tests {
    use super::*;
    use reims_vgpu_core::access::{
        AccessMode, AccessSource as _, BackingId, ByteRange, Participation, ParticipationExtent,
    };
    use reims_vgpu_core::identity::{ChannelId, TaskId};
    use reims_vgpu_core::lifecycle::Storage;

    const TASK: u32 = 3;
    const DOMAIN: ChannelId = ChannelId(1);
    const BACKING: BackingId = BackingId(0x4000);

    /// A device with one task holding one dedicated resource, and that
    /// resource's name.
    fn state_with_one_resource() -> (DeviceState, ResourceId) {
        let mut state = DeviceState::new(DeviceId(1), 12);
        state.define_task(TASK, 1 << 20, 0x100);
        let declared = state
            .declare_object(
                TASK,
                7,
                Storage::Dedicated {
                    backing: BACKING,
                    extent: ByteRange {
                        offset: 0,
                        length: 4096,
                    },
                },
            )
            .expect("the task is open");
        (state, declared.id)
    }

    fn participation(resource: ResourceId, mode: AccessMode) -> Participation {
        Participation {
            resource,
            extent: ParticipationExtent::Whole,
            mode,
            api_stages: 0,
        }
    }

    /// A widened window returns, and saying so does not re-enter the lock.
    ///
    /// **A hang here is this test failing.** Both of this door's observers —
    /// `note_access_widened` and `note_access_refused` — read state that takes
    /// `DeviceState::lifecycle`, the same lock the answer was produced under.
    /// Written as one chain on the locked call the guard is still alive when
    /// they run, and the device deadlocks on the first record that trips one: a
    /// driven boot froze exactly that way, `walk_records_render = 42` for seven
    /// minutes. There is no assertion that catches a deadlock from inside the
    /// thread it deadlocks, so the assertion is that this returns at all — and
    /// that what comes back is the widened access rather than a refusal.
    #[test]
    fn a_widened_access_returns_and_saying_so_does_not_re_enter_the_lock() {
        use reims_vgpu_core::access::AccessKey;

        let (state, name) = state_with_one_resource();
        let mut access = state.task_access(TaskId(TASK), DOMAIN);
        // A window one byte past the resource's own 4096: a bound the record
        // built long, which this resource cannot hold.
        let intent = access
            .access(&Participation {
                resource: name,
                extent: ParticipationExtent::Range(ByteRange {
                    offset: 0,
                    length: 4097,
                }),
                mode: AccessMode::Read,
                api_stages: 0,
            })
            .expect("a bound past the end is not a refusal");
        assert!(
            !matches!(intent.key, AccessKey::Range(..)),
            "the window that did not fit is not reported as one: {:?}",
            intent.key
        );
        assert_eq!(intent.domain, DOMAIN);
    }

    /// The interleaving a command-stream walk performs, which the owner's own
    /// `TaskAccess` cannot survive.
    ///
    /// `reims_vgpu_core::walk::segment` resolves a record's refs and then
    /// records it, once per record — so a resolver call sits *between* two
    /// access calls of the same packet. A source holding the lifecycle guard
    /// would deadlock on the `object_name` in the middle rather than fail, so
    /// this test hangs where it does not pass; that is the failure mode being
    /// ruled out and there is no assertion that could report it instead.
    #[test]
    fn a_name_resolves_between_two_accesses_of_one_packet() {
        let (state, resource) = state_with_one_resource();
        let mut access = state.task_access(TaskId(TASK), DOMAIN);

        access
            .access(&participation(resource, AccessMode::Read))
            .expect("declared and resident");
        assert_eq!(state.object_name(TASK, 7), Some(resource));
        access
            .access(&participation(resource, AccessMode::Write))
            .expect("declared and resident");
    }

    /// Locking per call is not a snapshot: the content authority's reservations
    /// accumulate across a transaction exactly as they do through a held borrow.
    ///
    /// Two writes over one resource take two reservations. A source that copied
    /// anything out of the owner would hand both records the same one, which is
    /// the failure `lifecycle::TaskAccess`'s own doc names.
    #[test]
    fn two_writes_through_one_door_take_two_reservations() {
        let (state, resource) = state_with_one_resource();
        let mut access = state.task_access(TaskId(TASK), DOMAIN);

        let first = access
            .access(&participation(resource, AccessMode::Write))
            .expect("declared and resident");
        let second = access
            .access(&participation(resource, AccessMode::Write))
            .expect("declared and resident");

        assert!(first.output_content_version.is_some());
        assert_ne!(
            first.output_content_version, second.output_content_version,
            "a second write reserved the version the first one did"
        );
    }

    /// The door carries the packet's task and domain, and the domain reaches
    /// the access it places.
    #[test]
    fn the_domain_is_the_packets_and_not_the_participations() {
        let (state, resource) = state_with_one_resource();
        let mut access = state.task_access(TaskId(TASK), DOMAIN);

        assert_eq!(access.task(), TaskId(TASK));
        assert_eq!(access.domain(), DOMAIN);
        let intent = access
            .access(&participation(resource, AccessMode::Read))
            .expect("declared and resident");
        assert_eq!(intent.domain, DOMAIN);
    }

    /// A name the task never declared refuses with the owner's own slug, and
    /// refuses rather than resolving to whatever the slot used to hold.
    #[test]
    fn an_undeclared_name_refuses_with_the_owners_reason() {
        let (state, resource) = state_with_one_resource();
        let mut access = state.task_access(TaskId(TASK), DOMAIN);

        let stranger = ResourceId {
            slot: reims_vgpu_core::identity::ObjectListRef(9),
            generation: resource.generation,
        };
        let refusal = access
            .access(&participation(stranger, AccessMode::Read))
            .expect_err("slot 9 was never declared");
        assert_eq!(refusal.resource, stranger);
        assert_eq!(refusal.reason, "lifecycle_namespace");
    }
}

#[cfg(test)]
mod device_desc_tests {
    use super::*;
    use crate::protocol::iosurface_pages::{
        device_desc_plane, DEVICE_DESC_LEN, DEVICE_DESC_PLANES, DEVICE_PLANE_DESC_LEN,
    };

    fn entry_with_desc(len: usize) -> MappingEntry {
        MappingEntry {
            device_desc: vec![0u8; len],
            ..Default::default()
        }
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
        over[crate::protocol::iosurface_pages::DEVICE_DESC_PLANE_COUNT] = 8;
        assert!(
            device_desc_plane(&over, 7).is_some(),
            "the whole-slice spelling would have found an eighth plane"
        );

        let e = MappingEntry {
            device_desc: over,
            ..Default::default()
        };
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
            UNKNOWN_OPCODE_ECHO_WORDS_MAX + 1,
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
        assert!(!state.mappings.contains_key(&0));
        assert!(!state.set_mapping_geom(1, 0, 64, 0x50));
        assert!(!state.set_mapping_geom(1, 64, 0, 0x50));
        assert!(!state.mappings.contains_key(&1));
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
        use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
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
        state.mappings.get_mut(&3).unwrap().page_entries = vec![valid(4), valid(5), valid(6)];
        assert_eq!(
            state.mapping_reach_pages(3),
            Some(vec![4u64 << shift, 5u64 << shift, 6u64 << shift]),
            "every entry resolves, so the whole set is named"
        );

        // The middle entry carries no VALID bit, so it names no backing.
        state.mappings.get_mut(&3).unwrap().page_entries = vec![valid(4), 0, valid(6)];
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
        let mut epoch = state.host_writes.epoch();
        for announce in [
            &mut DeviceState::note_host_wrote_guest_ram as &mut dyn FnMut(&mut DeviceState),
            &mut |s: &mut DeviceState| s.note_host_wrote_pages(vec![0x1000]),
            &mut |s: &mut DeviceState| s.note_host_wrote_mapping(7),
        ] {
            announce(&mut state);
            let now = state.host_writes.epoch();
            assert_ne!(now, epoch, "a host write into guest RAM went unannounced");
            epoch = now;
        }
    }
}

#[cfg(test)]
mod pipeline_door_tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_X86};
    use reims_vgpu_core::identity::{ObjectListRef, ResourceId, SlotGeneration};

    fn name(slot: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration(1),
        }
    }

    /// The guest creating a pipeline declares it once, and the guest deleting
    /// it is what ends it.
    ///
    /// The pair is what makes the census readable. Declaration alone is a table
    /// that only grows — the guest re-binds the same pipeline on every draw, so
    /// "how many pipelines does this session have" would be "how many draws has
    /// it seen" — and the destroy is the one event on this interface that says
    /// a pipeline is over.
    #[test]
    fn a_pipeline_is_declared_once_and_the_guests_destroy_is_what_retires_it() {
        let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        assert_eq!(state.pipeline_census().declared, 0);

        assert!(
            state.declare_pipeline(name(9)),
            "the first sight of a pipeline is its declaration"
        );
        assert!(
            !state.declare_pipeline(name(9)),
            "and every re-bind after it is not, which is most of them"
        );
        assert_eq!(state.pipeline_census().declared, 1);

        // A different slot is a different pipeline; a different generation of
        // the same slot is too, which is the whole reason the name carries one.
        assert!(state.declare_pipeline(name(10)));
        assert!(state.declare_pipeline(ResourceId {
            slot: ObjectListRef(9),
            generation: SlotGeneration(2),
        }));
        assert_eq!(state.pipeline_census().declared, 3);

        assert_eq!(
            state.retire_pipeline(name(9)),
            reims_vgpu_core::session::Ended {
                took: true,
                stranded: Vec::new()
            },
            "the table had it, and nothing is admitted into this model yet, so \
             nothing was parked on it"
        );
        let census = state.pipeline_census();
        assert_eq!(
            (census.declared, census.retired),
            (3, 1),
            "the census counts events and not occupancy: three declarations \
             happened and one retirement did, and a retirement does not unsay \
             the declaration it ends"
        );

        // Retiring what the guest never created is not an event — and says so,
        // rather than answering the same empty list a real retirement with
        // nothing parked on it answers. A driven boot needs the difference: the
        // guest deletes render pipelines this device never drew with.
        assert_eq!(
            state.retire_pipeline(name(4000)),
            reims_vgpu_core::session::Ended::default()
        );
        assert_eq!(state.pipeline_census().retired, 1);
    }

    /// A declared pipeline reaches `Ready` only through the rail's three steps,
    /// and a rail with no memo walking them again is declined rather than
    /// reset.
    ///
    /// This is the half that keeps an admitted exec from parking forever. The
    /// table without it holds every pipeline at `Declared`, `Lease` answers
    /// `Pending` to every draw that binds one, and the transaction is admitted
    /// into a wait nothing can ever discharge — strictly worse than the
    /// `Absent` refusal an empty table gave, because a refusal is visible and
    /// a hang is not.
    #[test]
    fn a_rail_walks_a_declared_pipeline_to_ready_and_a_second_walk_is_declined() {
        use reims_vgpu_core::pipeline::PipelineState;

        let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        assert!(state.declare_pipeline(name(9)));

        assert!(
            !state.advance_pipeline(name(9), PipelineState::Compiling),
            "the steps are a lifetime and not a set: nothing compiles that has \
             not been translated"
        );
        assert!(state.advance_pipeline(name(9), PipelineState::Translating));
        assert!(state.advance_pipeline(name(9), PipelineState::Compiling));
        assert_eq!(
            state.pipeline_census().ready,
            0,
            "compiling is not usable, and a draw binding it is not ready"
        );
        assert!(state.ready_pipeline(name(9)));
        assert_eq!(state.pipeline_census().ready, 1);

        // The Metal rail retains no pipeline state, so it walks these same
        // three steps on every draw that binds the pipeline. The second walk
        // must not take it back to `Translating` — a `Ready` pipeline that
        // re-enters the build is a pipeline every parked draw waits on again.
        for step in [
            PipelineState::Translating,
            PipelineState::Compiling,
            PipelineState::Ready,
        ] {
            assert!(
                !state.advance_pipeline(name(9), step),
                "{} is not a step from ready",
                step.name()
            );
        }
        assert_eq!(
            state.pipeline_census().ready,
            1,
            "and the census counts the one time it became usable, not the \
             draws that found it so"
        );
    }

    /// A rail that cannot build a pipeline refuses it with the reason, once.
    #[test]
    fn a_refused_pipeline_is_terminal_and_a_later_step_cannot_revive_it() {
        use reims_vgpu_core::pipeline::{PipelineState, RefusalReason};

        let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        assert!(state.declare_pipeline(name(9)));
        assert!(state.advance_pipeline(name(9), PipelineState::Translating));

        assert!(
            state
                .refuse_pipeline(
                    name(9),
                    RefusalReason::TranslationFailed("vertex_translate")
                )
                .took,
            "the table had it to refuse"
        );
        assert_eq!(state.pipeline_census().refused, 1);

        // The guest re-binds a pipeline this device cannot build on every
        // frame, and each of those draws re-walks the rail. One refusal is the
        // whole point of the state being terminal.
        assert!(!state.advance_pipeline(name(9), PipelineState::Compiling));
        assert!(
            !state
                .refuse_pipeline(name(9), RefusalReason::CompilationFailed("again"))
                .took
        );
        assert_eq!(state.pipeline_census().refused, 1);

        // Refusing what the guest never created is not an event either — which
        // is why the rails skip the decline that *is* the pipeline being
        // absent rather than refusing a name the table has no entry for.
        assert!(
            !state
                .refuse_pipeline(name(4000), RefusalReason::CompilationFailed("absent"))
                .took
        );
        assert_eq!(state.pipeline_census().refused, 1);
    }
}

#[cfg(test)]
mod stamp_publication_tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_X86};
    use reims_vgpu_core::identity::{StampSlot, StampValue};

    fn publish(state: &DeviceState, slot: u32, value: u32) -> StampPublication {
        state.publish_completion_stamp(StampSlot(slot), StampValue(value))
    }

    /// A slot's timeline is first written, then advanced, and a word repeating
    /// the value it already holds is not movement.
    ///
    /// The repeat is not a corner case on this wire: a packet that does not
    /// signal repeats its channel's current completion word rather than
    /// leaving the slot alone, so it is the ordinary shape of most of the
    /// stream and would otherwise read as a fence advancing thousands of times
    /// a second.
    #[test]
    fn a_slot_is_first_written_then_advanced_and_a_repeat_is_neither() {
        let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        assert_eq!(publish(&state, 3, 10), StampPublication::First);
        assert_eq!(publish(&state, 3, 10), StampPublication::Repeat);
        assert_eq!(publish(&state, 3, 11), StampPublication::Advanced);

        // Slots are independent timelines. A device with one counter across
        // them would report every second channel's first word as behind.
        assert_eq!(publish(&state, 4, 1), StampPublication::First);
        assert_eq!(publish(&state, 3, 12), StampPublication::Advanced);
    }

    /// A word behind the slot leaves the model's timeline where it was, and
    /// says so.
    ///
    /// This device writes the packet header's word into the guest's slot
    /// without comparing, so the two records *can* disagree — and the
    /// disagreement is the interesting half: a fence going backwards
    /// unsatisfies every wait between the two values, and a model that
    /// silently followed it would take back readiness the guest has already
    /// been told about.
    #[test]
    fn a_word_behind_the_slot_does_not_move_the_model_and_is_named() {
        let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        assert_eq!(publish(&state, 0, 100), StampPublication::First);
        assert_eq!(
            publish(&state, 0, 99),
            StampPublication::Behind {
                held: reims_vgpu_core::identity::StampValue(100)
            },
            "and it says what the plane holds, so a rewound slot is \
             distinguishable from a timeline that has run ahead"
        );
        assert_eq!(
            publish(&state, 0, 100),
            StampPublication::Repeat,
            "the slot still holds 100, so the model did not follow the rewind"
        );
    }

    /// Later is decided on the wrapping timeline and not numerically.
    ///
    /// A guest whose completion counter wraps past `u32::MAX` writes a smaller
    /// number that is nonetheless the later point. A model comparing with `>`
    /// would call it a rewind and refuse to advance for the rest of the boot,
    /// which is a hang that starts four billion packets in and cannot be
    /// reproduced by driving apps.
    #[test]
    fn a_wrapped_timeline_advances_to_the_smaller_number() {
        let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        assert_eq!(publish(&state, 1, u32::MAX - 1), StampPublication::First);
        assert_eq!(publish(&state, 1, 3), StampPublication::Advanced);
        assert_eq!(
            publish(&state, 1, u32::MAX - 1),
            StampPublication::Behind {
                held: reims_vgpu_core::identity::StampValue(3)
            },
            "and the point before the wrap is behind the one after it"
        );
    }
}

#[cfg(test)]
mod mapping_declaration_tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_X86};

    fn state() -> DeviceState {
        DeviceState::new(DeviceId(1), PAGE_SHIFT_X86)
    }

    fn declared(state: &DeviceState, id: u32) -> (u32, u32) {
        let m = state.mappings.get(&id).expect("the mapping exists");
        (m.content_generation, m.surface_content_epoch)
    }

    /// Re-declaring a mapping at the same extent but a different pixel format
    /// withdraws the content claim, because the claim is about what the bytes
    /// *mean* and the guest has just changed that.
    ///
    /// The reset used to test the extent alone, on the reasoning that a format
    /// change moves the `TargetIdentity` and so picks up a different resident by
    /// itself. `present_identity::surface_format` collapses several guest
    /// declarations onto one `vk::Format` and falls back to the scanout order
    /// for any it cannot express, so that reasoning does not hold for every
    /// pair — and the failure is a resident served against an epoch stamped
    /// under the previous interpretation.
    #[test]
    fn re_declaring_a_mapping_at_a_new_format_withdraws_its_content_claim() {
        let mut state = state();
        assert!(state.set_mapping_geom(7, 640, 480, 0x50));
        let m = state.mappings.get_mut(&7).expect("the mapping exists");
        m.content_generation = 9;
        m.surface_content_epoch = 4;

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
        let m = state.mappings.get_mut(&7).expect("the mapping exists");
        m.content_generation = 9;
        m.surface_content_epoch = 4;
        assert!(state.set_mapping_geom(7, 800, 480, 0x50));
        assert_eq!(declared(&state, 7), (0, 0));
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
        assert_eq!(state.max_task_id_seen, 0);

        state.define_task(12, 0x1000, 2);
        assert!(state.tasks.is_active(12), "an ordinary id is accepted");
        assert_eq!(state.max_task_id_seen, 12);

        let past = u32::MAX;
        state.define_task(past, 0x1000, 2);
        assert!(
            state.tasks.is_active(past),
            "a task id is a full u32 on the wire and its storage is a map"
        );
        assert_eq!(state.max_task_id_seen, past);

        // High-water, not last-seen: a later smaller id does not lower it.
        state.define_task(3, 0x1000, 2);
        assert_eq!(state.max_task_id_seen, past);
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
    /// refused ids its own storage would have held: `mappings` is a `BTreeMap`.
    /// `u32::MAX` is the largest id the wire can carry, so accepting it here is
    /// the strongest form of "nothing is out of range", and a reinstated
    /// ceiling fails on the first assertion.
    ///
    /// The mark still moves, because it is an occupancy reading on the map now
    /// rather than a distance to a refusal.
    #[test]
    fn no_mapping_id_is_out_of_range() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        assert_eq!(state.max_mapping_id_seen, 0);

        assert!(state.map_surface(39), "an ordinary id is accepted");
        assert_eq!(state.max_mapping_id_seen, 39);

        assert!(
            state.map_surface(u32::MAX),
            "a mapping id is a full u32 on the wire and its storage is a map"
        );
        assert!(state.mappings.contains_key(&u32::MAX));
        assert_eq!(state.max_mapping_id_seen, u32::MAX);

        assert!(
            !state.map_surface(0),
            "0 is the unbound sentinel and is the one id that stays refused"
        );
        assert!(!state.mappings.contains_key(&0));
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
                "delete_task" => assert!(!state.delete_task(past)),
                "set_object_list" => assert!(!state.set_object_list(past, 1, 1)),
                _ => assert!(!state.insert_object(past, 7)),
            }
            assert_eq!(
                state.max_task_id_seen, past,
                "{name} refused without recording the reach"
            );
            assert!(
                !state.tasks.is_active(past),
                "{name} must not have defined the task it refused"
            );
        }
    }
}

/// The ordering plane's published reflections, borrowed for one packet's walk.
///
/// See [`DeviceState::pipeline_usage`] for why this is a guard and not a
/// lookup.
pub struct PipelineUsage<'a>(std::sync::MutexGuard<'a, reims_vgpu_core::session::SessionModel>);

impl std::fmt::Debug for PipelineUsage<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PipelineUsage")
    }
}

impl reims_vgpu_core::pipeline::UsageSource for PipelineUsage<'_> {
    fn binding_usage(
        &self,
        pipeline: reims_vgpu_core::identity::ResourceId,
        stage: Option<reims_vgpu_protocol::render::ShaderStage>,
    ) -> Option<&reims_vgpu_core::pipeline::BindingUsage> {
        self.0.pipelines_ref().binding_usage(pipeline, stage)
    }
}
