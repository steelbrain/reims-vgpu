//! Device-owned state: registers, rings, tasks, mapper, present, fail log.

use crate::model::{LruBytesMemo, GFX_MMIO_SIZE, MAX_CHANNELS, MAX_MAPPINGS, MAX_TASKS};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

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

/// How many leading payload words an unknown child opcode echoes. Four covers
/// every unknown packet a driven boot has produced whole (the largest is 76
/// bytes of which 64 are payload) while bounding the line for a command that
/// carries a large buffer; `plen` always reports the true length, so a reader
/// can tell an echo that was cut from one that was complete.
const UNKNOWN_OPCODE_ECHO_WORDS: usize = 4;

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
            } => {
                let mut fields = vec![
                    ("ch", channel.to_string()),
                    ("opcode", format!("{opcode:#x}")),
                    ("total_size", total_size.to_string()),
                    ("stamps", stamp_count.to_string()),
                    ("plen", payload.len().to_string()),
                ];
                // Whole words only, in wire order, so a reader can line the echo
                // up against the packet layout. A trailing sub-word tail is
                // reported by `plen` rather than zero-padded into a word that
                // the guest never wrote.
                if !payload.is_empty() {
                    let words = payload
                        .chunks_exact(4)
                        .take(UNKNOWN_OPCODE_ECHO_WORDS)
                        .map(|word| format!("{:#010x}", crate::contract::endian::ld32(word)))
                        .collect::<Vec<_>>()
                        .join(":");
                    if !words.is_empty() {
                        fields.push(("payload", words));
                    }
                }
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
    /// into `active_child_mask` / `pending.child_mask` — including *inside* the
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StateMutationDecline {
    DefineTaskIdRange { task_id: u32 },
    DeleteTaskIdRange { task_id: u32 },
    SetObjectListTaskIdRange { task_id: u32 },
    SetObjectListTaskInactive { task_id: u32 },
    InsertObjectTaskIdRange { task_id: u32, object_ref: u32 },
    InsertObjectTaskInactive { task_id: u32, object_ref: u32 },
    MapSurfaceIdRange { mapping_id: u32 },
    UnmapSurfaceIdRange { mapping_id: u32 },
    AttachMappingIdRange { mapping_id: u32 },
    AttachMappingInternalZero { mapping_id: u32 },
    MappingDeviceDescIdRange { mapping_id: u32 },
    MappingDeviceDescEmpty { mapping_id: u32 },
    MappingGeomIdRange { mapping_id: u32 },
    MappingGeomWidthZero { mapping_id: u32 },
    MappingGeomHeightZero { mapping_id: u32 },
    MappingGeomWidthRange { mapping_id: u32, width: u32 },
    MappingGeomHeightRange { mapping_id: u32, height: u32 },
}

impl crate::observe::Decline for StateMutationDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::DefineTaskIdRange { .. } => "model_define_task_id_range",
            Self::DeleteTaskIdRange { .. } => "model_delete_task_id_range",
            Self::SetObjectListTaskIdRange { .. } => "model_set_object_list_task_id_range",
            Self::SetObjectListTaskInactive { .. } => "model_set_object_list_task_inactive",
            Self::InsertObjectTaskIdRange { .. } => "model_insert_object_task_id_range",
            Self::InsertObjectTaskInactive { .. } => "model_insert_object_task_inactive",
            Self::MapSurfaceIdRange { .. } => "model_map_surface_id_range",
            Self::UnmapSurfaceIdRange { .. } => "model_unmap_surface_id_range",
            Self::AttachMappingIdRange { .. } => "model_attach_mapping_id_range",
            Self::AttachMappingInternalZero { .. } => "model_attach_mapping_internal_zero",
            Self::MappingDeviceDescIdRange { .. } => "model_mapping_device_desc_id_range",
            Self::MappingDeviceDescEmpty { .. } => "model_mapping_device_desc_empty",
            Self::MappingGeomIdRange { .. } => "model_mapping_geom_id_range",
            Self::MappingGeomWidthZero { .. } => "model_mapping_geom_width_zero",
            Self::MappingGeomHeightZero { .. } => "model_mapping_geom_height_zero",
            Self::MappingGeomWidthRange { .. } => "model_mapping_geom_width_range",
            Self::MappingGeomHeightRange { .. } => "model_mapping_geom_height_range",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        let mut fields = match self {
            Self::DefineTaskIdRange { task_id }
            | Self::DeleteTaskIdRange { task_id }
            | Self::SetObjectListTaskIdRange { task_id }
            | Self::SetObjectListTaskInactive { task_id } => {
                vec![("task", task_id.to_string())]
            }
            Self::InsertObjectTaskIdRange {
                task_id,
                object_ref,
            }
            | Self::InsertObjectTaskInactive {
                task_id,
                object_ref,
            } => vec![
                ("task", task_id.to_string()),
                ("ref", object_ref.to_string()),
            ],
            Self::MapSurfaceIdRange { mapping_id }
            | Self::UnmapSurfaceIdRange { mapping_id }
            | Self::AttachMappingIdRange { mapping_id }
            | Self::AttachMappingInternalZero { mapping_id }
            | Self::MappingDeviceDescIdRange { mapping_id }
            | Self::MappingDeviceDescEmpty { mapping_id }
            | Self::MappingGeomIdRange { mapping_id }
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

/// The guest page table and GPU-VA base a mapping's [`MappingEntry::
/// page_entries`] were walked from, when the list came from a type-4 surface
/// plan.
///
/// Latched at the one site that assigns those entries so the two cannot drift
/// apart. It exists so a later reader can *repeat* the walk without repeating
/// the search: `resolve_type4_surface_ex` finds the surface object by probing up
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
pub struct Type4Walk {
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
    /// remaining route to that cost is the one
    /// [`crate::runtime::storage_flush::flush_mapping_windows_before_fence`]
    /// names: making the undeclared guest read observable.
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
    /// Epoch of this mapping's *surface content* in the sense a type-11 render
    /// LOAD needs: it advances whenever the pixels that Load would seed from
    /// could have changed, wherever they live.
    ///
    /// Strictly coarser than [`Self::content_generation`], and deliberately so.
    /// `content_generation` counts writes to the mapping's *guest pages*, which
    /// misses the one publisher that writes only the host shadow: the deferred
    /// type-11 Store stores into `surface_cache` and arms a window instead of
    /// scattering into guest pages. `surface_cache` holds exactly one entry per
    /// mapping, so a sibling Store at a *different* geometry replaces the entry
    /// an older geometry's resident is being compared against while
    /// `content_generation` never moves — the same one-entry-per-mapping hazard
    /// that cost the `deferred_flush_lost reason=cache_miss` class. Bumping here
    /// on that publish makes the sibling case a mismatch, so the older geometry
    /// falls back to the CPU seed rather than loading from a resident whose
    /// currency nothing established.
    ///
    /// Compared against [`crate::backend::vulkan::engine::resident_content_epoch`]
    /// to decide whether a type-11 LOAD may take `LoadOp::LoadFromTarget` and
    /// skip its CPU seed entirely. Never read to decide *what* to present or
    /// draw — only whether a known-equal upload can be elided.
    pub surface_content_epoch: u32,
    /// Who has read what the last landed render flush of this mapping wrote.
    /// See [`RenderFlushWitness`]; reported by
    /// [`crate::runtime::storage_flush::note_render_flush_landed`].
    pub render_flush: RenderFlushWitness,
    /// Bumped whenever the guest page list / map lifetime changes (MAP, UNMAP,
    /// ReplacePhysical, MappingInternal reattach, page-table refresh that
    /// changes PFNs). Used as [`TargetIdentity`] generation for resident
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
    /// for the guest mapping, and it is read and written by the **CPU only**:
    /// Metal render targets used to be created directly on this view, which
    /// gave the host GPU a handle on guest RAM, and that alias is deleted.
    /// Guest CPU writes and host page reads still see one copy; a GPU Store
    /// reaches it through the writeback. Retired (never freed in place)
    /// whenever `page_entries` change; see `DeviceState::retired_views`.
    pub contig_ptr: usize,
    pub contig_len: usize,
    /// `map_generation` whose page list was measured non-packed, so no
    /// contiguous view can exist over it. `None` = not measured for the
    /// current list.
    ///
    /// "Packed or not" is a pure function of `page_entries`, and
    /// `map_generation` names that list — the same key that makes `contig_ptr`
    /// above safe to cache. Without it every caller on a fragmented mapping
    /// re-collected the whole page-GPA vector and re-scanned it only to reach
    /// the answer it reached last time.
    pub contig_fragmented_gen: Option<u32>,
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
    /// the type-4 page refresh both replace the list in place, and both retired
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
    /// The other half of the type-11 seed currency test.
    /// [`Self::surface_content_epoch`] can only witness writers inside this
    /// crate — every caller of `mark_mapping_written` is one — and a surface's
    /// pages are plain guest RAM the guest CPU stores into with no device
    /// operation at all. This is what sees that store.
    ///
    /// 0 means no Store has stamped it, or the host could not answer, and
    /// never compares equal to a live generation (the host's first readable
    /// generation is 1).
    pub guest_write_gen_at_store: u64,
    /// Task id that last owned this surface as a type-4 `OBJECT_TYPE_SURFACE`
    /// object (0 = no non-trivial hint; task 0 is always probed first anyway).
    /// `resolve_type4_surface_ex` probes this task right after task 0 so a
    /// per-bind present-path scan short-circuits instead of walking all 256
    /// task slots. Purely a search-order hint — a stale/wrong value only costs
    /// one extra probe before the full-table fallback re-finds the owner.
    pub owner_task_hint: u32,
    /// How [`Self::page_entries`] were derived, when they came from a type-4
    /// surface plan — see [`Type4Walk`]. `None` for every other source, and for
    /// a mapping whose list has been invalidated.
    ///
    /// Distinct from [`Self::owner_task_hint`], which is a *search* hint and is
    /// allowed to be wrong. This is a statement about the list that is in the
    /// entry right now: repeat this walk and you must get these entries back, or
    /// the guest has moved the surface underneath us without saying so.
    pub type4_walk: Option<Type4Walk>,
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
/// - **Surface window** (`mapping_id != 0`): a type-11 IOSurface view;
///   `texture_ref == 0`.
/// - **Linear window** (`mapping_id == 0`): a type-2/3 raw task-GVA texture,
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
    /// Identity of a linear (type-2/3 raw task-GVA) texture window.
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

/// Which rail holds the authoritative pixels of a mapping-keyed deferred
/// window, and therefore how a flush must read them.
///
/// Both kinds live in one map — [`DeviceState::compute_deferred_flush`] — and
/// that is the point. The dangerous half of any deferred rail is the set of
/// guest-page readers that must drain it first; a reader that misses one window
/// makes the guest read stale pixels with nothing logged. Sharing the key type
/// means both kinds share the range scan
/// ([`DeviceState::take_deferred_flush_windows`]), the raw-GVA alias index
/// ([`DeviceState::deferred_alias_pages`]), the teardown drop and every
/// existing trigger, so a rail cannot be covered for one kind and missed for
/// the other. A second map keyed the same way would have had to re-derive
/// "does any window still name this mapping" in
/// [`DeviceState::prune_alias_index`], and getting that wrong drops the alias
/// index out from under a live window.
///
/// What genuinely differs is only where the pixels are. Everything else the
/// flush needs — mapping id, geometry, format, guest byte range — is already in
/// the key, which is why neither variant carries geometry of its own.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeferredOwner {
    /// Compute rail: a *storage* resident keyed by this same
    /// `ComputeStorageResidencyKey`, read with
    /// `engine::read_resident_storage(key, generation)`. The generation is the
    /// resident's **content** generation, unrelated to `key.map_generation`.
    Storage {
        generation: u32,
        armed_stamp_seq: u64,
    },
    /// Type-11 render Store rail: the window **owns the frame it deferred**,
    /// tight BGRA8 at `key.width x key.height`, shared with the
    /// [`crate::runtime::surface_cache`] entry that was stored from the same
    /// readback.
    ///
    /// Owning it is what makes the obligation landable. The flush used to source
    /// its pixels from `surface_cache::get(mapping_id, key.width, key.height)`,
    /// and that cache holds exactly **one** entry per mapping: a later Store at a
    /// different geometry replaces it, and every window still armed at the old
    /// geometry then misses and reports `deferred_flush_lost reason=cache_miss`.
    /// One boot lost 15 whole layers that way — a 1920x1080 desktop surface, a
    /// 1920x24 menu bar, several window-sized rects — which is a compositing
    /// layer rendering solid black with the loss reported only after the fact.
    /// An `Arc` clone costs nothing at arm time and cannot be orphaned.
    ///
    /// `source` is the one thing that varies, and it varies for a reason the
    /// paragraph above states in the other direction: owning the bytes is free
    /// *when the Store already read them back*. A Store that skips its readback
    /// has no bytes to own, and the whole point of skipping it is that ~98 % of
    /// these windows are never flushed at all — so that rail names the resident
    /// and pays the readback only if someone asks. Everything else about the
    /// window is identical, which is why this is a field and not a variant:
    /// `matches!(owner, Render { .. })` still selects the rail for the population
    /// cap, the alias index, the teardown drop and `owner_slug`, and only the
    /// pixel read dispatches.
    Render {
        armed_seq: u64,
        armed_stamp_seq: u64,
        source: RenderWindowSource,
    },
}

impl DeferredOwner {
    /// [`DeviceState::completion_stamp_seq`] when this window was armed.
    ///
    /// Both rails carry it for the same reason [`GvaDeferredEntry::
    /// armed_stamp_seq`] does: a window that lands after the guest was fenced
    /// writes memory the guest was already entitled to reclaim, and no check
    /// taken after the fence can tell that memory apart from the target it used
    /// to be. Unlike the GVA rail, these windows are keyed by a
    /// `ComputeStorageResidencyKey` that carries `map_generation`, so the flush
    /// can already refuse a mapping incarnation the guest replaced — which is
    /// why this rail is measured before it is changed rather than assumed to
    /// share the GVA rail's verdict.
    pub fn armed_stamp_seq(&self) -> u64 {
        match self {
            Self::Storage {
                armed_stamp_seq, ..
            }
            | Self::Render {
                armed_stamp_seq, ..
            } => *armed_stamp_seq,
        }
    }
}

/// Where a [`DeferredOwner::Render`] window's frame lives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RenderWindowSource {
    /// The window owns the frame, tight BGRA8 at `key.width x key.height`,
    /// shared with the [`crate::runtime::surface_cache`] entry stored from the
    /// same readback. Cannot be orphaned; see the variant's own note.
    Owned(std::sync::Arc<Vec<u8>>),
    /// The engine's `TargetIdentity::Surface` resident holds the frame and the
    /// window holds a pin on it. `epoch` is the
    /// [`MappingEntry::surface_content_epoch`] this window's pixels were
    /// published at; the flush compares it against the resident's stamp
    /// (`engine::resident_content_epoch`) and declines rather than writing a
    /// frame it cannot vouch for.
    ///
    /// The identity is **reconstructed** at flush time from the key rather than
    /// stored, exactly as `flush_gva_one` does: `key` already carries the mapping
    /// id, the geometry and the `map_generation` that `surface_identity` keys on,
    /// and the flush refuses on generation drift before it reads anything. Storing
    /// it would put a backend type in the model and give the two spellings a way
    /// to disagree.
    Resident { epoch: u32 },
}

/// Everything a later flush needs to land a deferred **GVA render Store**
/// (type-2/3 color0 with `target_gva != 0`): the engine resident
/// `TargetIdentity::Gva { gva, width, height, generation: alloc_gen }` holds the
/// authoritative pixels; guest pages + `host_gva_surfaces` are stale until a
/// flush lands them. One window per `gva` — a newer Store at the same GVA
/// supersedes (same geometry) or flushes (different geometry) the older one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GvaDeferredEntry {
    pub task_id: u32,
    pub texture_ref: u32,
    /// Producer object type captured at defer time (the task/object list may
    /// be gone by flush time) — `host_gva_surfaces` owner-gating input.
    pub producer_object_type: u8,
    pub width: u32,
    pub height: u32,
    /// Guest row stride the sync Store would have written with.
    pub row_stride: u32,
    pub format: u16,
    /// Arm order for oldest-first flush when the window cap is hit.
    pub armed_seq: u64,
    /// [`DeviceState::completion_stamp_seq`] when this window was armed.
    ///
    /// The window's `pages` guard asks whether the GVA still resolves to the
    /// pages it was armed on. That question is blind to the hazard this field
    /// names: a guest that frees the render target and lets its own allocator
    /// hand the same pages to something else keeps the translation identical, so
    /// the guard passes and the flush writes pixels over whatever moved in. The
    /// guest is entitled to do that from the moment it is stamped, so a landing
    /// whose stamp counter has moved is a write the guest never agreed to.
    pub armed_stamp_seq: u64,
    /// Defer-time physical page GPAs of the guest window — raw task-GVA reads
    /// aliasing these flush first (`storage_flush::flush_intersecting_task_gva`).
    pub pages: std::collections::HashSet<u64>,
    /// `generation` of the engine resident this window pinned — the page-set
    /// hash the arming draw resolved (`DrawEncodeRequest::gva_alloc_gen`).
    ///
    /// Stored rather than recomputed. The window exists precisely because the
    /// address may be handed to another allocation before the flush runs, and a
    /// walk taken then would name *that* allocation: the registry lookup would
    /// miss the slot this window is holding pinned, and the frame would be lost
    /// to a `deferred_flush_lost` instead of landing. Every consumer that
    /// rebuilds the identity from a window reads this field.
    pub alloc_gen: u64,
}

/// Everything a later flush needs to land a deferred **linear compute-storage
/// Store** (`ComputeStorageResidencyKey::linear` — a raw task GVA, `mapping_id`
/// 0). The engine resident holds the authoritative pixels;
/// `storage_flush::flush_linear_one` lands them into guest pages and
/// `host_linear_textures`.
///
/// This window names an *address under a task*, exactly like
/// [`GvaDeferredEntry`], and for the same reason: a type-2/3 linear texture has
/// no mapping incarnation to name and the wire format carries no lifecycle
/// notify for one. The mapping-keyed rails
/// (`storage_flush::flush_render_one`/`flush_storage_one`) can refuse on
/// `map_generation` drift because the guest must MAP/UNMAP/ReplacePhysical to
/// reclaim an IOSurface's storage; nothing of the sort exists here, which is why
/// this entry carries the same fence stamp the GVA rail carries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearDeferredEntry {
    /// Engine resident generation the window pinned — `read_resident_storage`'s
    /// second argument, and the only thing that distinguishes two residents at
    /// one key.
    pub generation: u32,
    /// [`DeviceState::completion_stamp_seq`] when this window was armed.
    ///
    /// Same hazard, same reading as [`GvaDeferredEntry::armed_stamp_seq`]: after
    /// the stamp the guest may free this texture's memory and its own allocator
    /// may hand those pages to anything without touching a page table, so
    /// `storage_flush::deferred_pages_still_ours` still passes and the flush
    /// writes a compute-storage image over whatever moved in.
    pub armed_stamp_seq: u64,
    /// Defer-time physical page GPAs of the guest window — raw task-GVA reads
    /// aliasing these flush first
    /// (`storage_flush::flush_intersecting_task_gva`), and the writer's own walk
    /// is bounded to them.
    pub pages: std::collections::HashSet<u64>,
}

impl GvaDeferredEntry {
    /// Guest byte span the flush writes: `row_stride * height`.
    pub fn span(&self) -> u64 {
        (self.row_stride as u64).saturating_mul(self.height as u64)
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
    /// Shared rather than owned so a [`DeferredOwner::Render`] window can hold
    /// the exact frame it deferred without copying it: the window and this entry
    /// point at one allocation, and replacing the entry leaves the window's
    /// pixels intact instead of orphaning them.
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
    /// Latest type-11 **Composite** writeback mid (logo/desktop content).
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
    pub frame_valid: bool,
    /// True only when DisplaySwap capture failed; first host paint retries.
    pub frame_encode_pending: bool,
    /// DisplaySwaps accepted since the last host paint of +0x188.
    ///
    /// apple-gfx `pending_frames` / PGDisplay `waitForPendingFrames` entry gate:
    /// when this is ≥ [`crate::runtime::drain::MAX_UNPAINTED_PRESENTS`], the
    /// child drain **holds** the next CmdDisplaySwap at channel head (no stamp)
    /// until paint clears the count. Accepted presents still stamp at retain.
    pub unpainted_presents: u32,
    /// Suppress repeated fail-log lines while the same present packet remains
    /// held at the pending-frames entry gate.
    pub backpressure_hold_active: bool,
    pub backpressure_hold_channel: u32,
    pub backpressure_hold_head: u32,
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
/// `type5_view_memo`, `type11_memo`). A cap crossing evicts the coldest entries
/// down to a low-water mark — never a bulk clear — so the hot working set (and
/// its avoided re-decode/re-convert cost) survives.
pub const GUEST_LINEAR_MEMO_BYTE_CAP: usize = 128 << 20;

/// Byte cap for the GVA-keyed type-2/3 encode cache
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
    /// Tight upload bytes of `native`: swizzled RGBA8, or — when `bgra8` — the
    /// guest's native BGRA8 order (uploaded into a BGRA8 image, no CPU swap).
    pub rgba: std::sync::Arc<Vec<u8>>,
    /// `rgba` holds native BGRA8 texels (upload as `Bgra8`) rather than RGBA8.
    pub bgra8: bool,
    /// Content generation: bumps only when the native bytes change.
    pub generation: u64,
}

/// A map of deferred writeback windows — pixels this device still owes guest
/// RAM, keyed by the rail that armed them.
///
/// Read-only outside this module: [`Deref`](std::ops::Deref) exposes the whole
/// `BTreeMap` read API, and there is deliberately **no** `DerefMut`, so the
/// inner map can only be mutated through the arm/disarm methods below. Arming
/// stamps a window with the fence generation it was armed under and disarming
/// hands the page set back to the caller that is about to write those pages;
/// a site that inserted or removed a window directly would skip both, so the
/// type refuses to let it compile.
#[derive(Debug)]
pub struct DeferredWindows<K, V>(BTreeMap<K, V>);

impl<K, V> DeferredWindows<K, V> {
    fn new() -> Self {
        Self(BTreeMap::new())
    }
}

impl<K, V> std::ops::Deref for DeferredWindows<K, V> {
    type Target = BTreeMap<K, V>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// `for (k, v) in &windows`, which auto-deref does not reach on its own.
impl<'a, K, V> IntoIterator for &'a DeferredWindows<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = std::collections::btree_map::Iter<'a, K, V>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
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
    pub active_child_mask: u32,
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
    /// When the drain worker last woke (`observe::elapsed_ms`). The stall
    /// reporter compares this against now: a device with outstanding work whose
    /// worker has not woken for seconds is wedged, and every wedge on this
    /// device so far was silent until something external printed a snapshot.
    pub last_drain_wake_ms: u64,
    /// Last stall snapshot emission, so a wedge reports on a bounded cadence
    /// instead of once per poll.
    pub last_stall_report_ms: u64,
    /// Display transactions held while another channel remained blocked on
    /// translation after the transaction's rescue drains. This counts hold
    /// episodes, not poll retries of the same packet.
    pub present_translation_holds: u64,
    /// Display channels whose FIFO head is already held for
    /// `translation_deferred_mask`. Suppresses fail-log flooding while the
    /// same head is retried and is cleared with channel lifecycle state.
    pub present_translation_hold_mask: u32,
    /// When the current display-order hold began. The hold is correct but must
    /// not be unbounded: the guest watchdogs its own ring, and a stall it
    /// attributes to the device costs a `GPU Reset` that discards every frame
    /// in flight, not just the one being ordered.
    pub present_translation_hold_since: Option<std::time::Instant>,
    /// When a render pipeline object was first found unreadable, keyed by
    /// (task, ref). A draw whose pipeline the guest has not finished
    /// publishing is retried rather than lost, and this is what stops that
    /// retry from becoming a wait for something that will never arrive.
    pub pipeline_unreadable_since: std::collections::HashMap<(u32, u32), std::time::Instant>,
    pub pending: PendingWork,
    pub child_rings: [ChannelRing; MAX_CHANNELS],
    pub tasks: [TaskEntry; MAX_TASKS],
    /// Count of MapMemory2/UnmapMemory packets (measure census).
    pub map_family_events: u64,
    /// Live object refs per task, as `(task_id, ref)`.
    ///
    /// Membership only — deliberately carries no descriptor payload. Every
    /// consumer that needs an object's type or descriptor reads the guest's own
    /// list through `objects::lookup_list_entry`, which walks guest memory at
    /// use time; a cached copy here would be a second source of truth for
    /// something the guest can rewrite under us. What the set *is* load-bearing
    /// for is [`Self::delete_object`], which gates the host-side resource
    /// teardown (`host_texture_surfaces`, `host_linear_textures`,
    /// `texture_to_mapping`) on whether the ref was live.
    pub objects: std::collections::BTreeSet<(u32, u32)>,
    /// Type-11 texture object ref → mapping_id: (task_id, ref) -> mapping_id.
    pub texture_to_mapping: BTreeMap<(u32, u32), u32>,
    pub mappings: BTreeMap<u32, MappingEntry>,
    /// Host render-cache keyed by surface_id / mapping_id (Linux/Vulkan rail).
    /// See [`crate::runtime::surface_cache`] and kb tahoe-x86-host-reims_vgpu §8.5.
    /// **Surface_id namespace only** — never texture_ref (object list ids collide).
    pub host_surfaces: BTreeMap<u32, HostSurface>,
    /// Discrete encode cache for type-2/3 GVA color targets, keyed by texture
    /// object ref. Separate from [`Self::host_surfaces`] so list ids cannot
    /// clobber type-4 present mids (live: sky `tex_ref=24` vs mid 24).
    pub host_texture_surfaces: BTreeMap<u32, HostSurface>,
    /// Same type-2/3 encode content keyed by target GVA — survives texture_ref
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
    /// Raw compute encode for type-2/3 textures. Retained across GVA unmap;
    /// evicted on task/object lifetime end or descriptor mismatch.
    pub host_linear_textures: BTreeMap<(u32, u32), HostLinearTexture>,
    /// Perf memo for guest-CPU-produced linear textures (no host cache entry,
    /// so no producer generation exists). Coherence is re-established on
    /// every lookup by re-reading the native guest rows and comparing them
    /// byte-exact against the memoized copy — a guest write is always seen;
    /// only the swizzle+alloc (and the engine's content hash+memcmp, via the
    /// generation identity) are skipped on unchanged content. Keyed by
    /// (task_id, level-0 gva, width, height, sample format). Byte-bounded LRU
    /// ([`GUEST_LINEAR_MEMO_BYTE_CAP`]): a cap crossing evicts the least-recently
    /// -used entries down to a low-water mark, never bulk-clearing the hot set.
    pub guest_linear_memo: LruBytesMemo<(u32, u64, u32, u32, u16), GuestLinearMemo>,
    /// Whether the hypervisor's guest-write generation would be a sound "these
    /// texels did not change" key for the zero-copy sampled gathers, measured
    /// against the bytes themselves. See
    /// [`crate::runtime::gather_witness`] — it selects no behaviour.
    #[cfg(feature = "backend-vulkan")]
    pub gather_witness: crate::runtime::gather_witness::GatherWitness,
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
    /// measured, not theorised. The guest-linear and type-5 memos shared this
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
    /// Byte-exact revalidated memo for type-5 serialized texture views
    /// (media IOSurface planes). Same contract as
    /// [`Self::guest_linear_memo`]: every bind re-reads the native plane
    /// window; conversion + upload (via the returned content identity) are
    /// skipped on unchanged bytes. Keyed by
    /// (mapping_id, plane, width, height, view pixel format). Byte-bounded LRU
    /// ([`GUEST_LINEAR_MEMO_BYTE_CAP`]).
    pub type5_view_memo: LruBytesMemo<(u32, u32, u32, u32, u16), GuestLinearMemo>,
    /// Byte-exact revalidated memo for the type-11 mapping-backed sampled path
    /// (`load_type11_mapping_rgba` — small IOSurface textures below the zero-copy
    /// floor, e.g. dock icons under magnification). Same contract as
    /// [`Self::guest_linear_memo`]: every bind re-reads the native BGRA rect;
    /// the BGRA->RGBA convert + the two per-bind allocs + the engine's content
    /// hash+upload (via the returned content identity) are skipped on unchanged
    /// bytes. A dock-magnification burst re-binds the same static icons ~1000x,
    /// so this collapses the `t11_guest` CPU copies that otherwise saturate the
    /// serial drain worker (dock-hover freeze). Keyed by (mapping_id, w, h).
    /// Byte-bounded LRU ([`GUEST_LINEAR_MEMO_BYTE_CAP`]).
    pub type11_memo: LruBytesMemo<(u32, u32, u32), GuestLinearMemo>,
    /// Reusable native BGRA read buffer for the type-11 memo re-read.
    pub type11_memo_scratch: Vec<u8>,
    /// Measurement-only: last guest-visible generation produced by a compute
    /// storage-image writeback for an exact type-11 view. This does not select
    /// engine behavior; it measures safe residency opportunities.
    pub compute_storage_residency: BTreeMap<ComputeStorageResidencyKey, u32>,
    /// Deferred mapping-keyed writebacks: windows whose guest pages are STALE —
    /// a pinned engine resident is the authoritative content. Every host-side
    /// read or write of intersecting mapping bytes must flush first
    /// (`runtime::storage_flush::flush_intersecting`). The value says which
    /// rail owns the pixels; see [`DeferredOwner`].
    pub compute_deferred_flush: BTreeMap<ComputeStorageResidencyKey, DeferredOwner>,
    /// Arm order for [`DeferredOwner::Render`] windows, so the population cap
    /// can evict oldest-first. Compute windows are bounded by the dispatches
    /// that create them; render windows are armed once per composite Store and
    /// each one pins a display-sized image, so they need their own bound.
    pub surface_deferred_seq: u64,
    /// Physical page bases of each mapping with live deferred windows, for the
    /// raw task-GVA sampling guard (`storage_flush::flush_intersecting_task_gva`).
    /// Built at defer time from the just-resolved `page_entries`
    /// ([`Self::index_deferred_alias_pages`] — per-sample resolution cost the
    /// boot-19 setup_us regression measured at ~1.4 s/boot); entries drop when
    /// the mapping's last deferred window is taken. A stale entry after a PFN
    /// change costs one spurious no-op flush call, never a wrong flush — the
    /// windows map stays the single flush authority.
    pub deferred_alias_pages: DeferredWindows<u32, std::collections::HashSet<u64>>,
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
    /// these via `HostOps::unmap_pages` after dropping the Metal objects that
    /// alias them (`mapper::flush_retired_views`).
    pub retired_views: Vec<(usize, usize)>,
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
    /// runtime unpins these (`storage_flush::retire_linear_residents`) so the
    /// pinned images become LRU-evictable instead of leaking.
    pub retired_linear_residents: Vec<ComputeStorageResidencyKey>,
    /// Deferred linear windows whose guest pages the superseded sync path
    /// WOULD have written (GVA-mapped at defer time): generation + defer-time
    /// page-GPA index. A raw task-GVA read aliasing these pages flushes the
    /// resident into the cache entry and guest pages first
    /// (`storage_flush::flush_intersecting_task_gva`). Cache-only-shaped
    /// windows never enter — their sync path never wrote guest pages either.
    pub linear_deferred_flush: DeferredWindows<ComputeStorageResidencyKey, LinearDeferredEntry>,
    /// Deferred GVA render-Store windows (type-2/3 color0, `target_gva != 0`)
    /// whose guest bytes + `host_gva_surfaces` encode the superseded sync path
    /// WOULD have written. The engine resident `TargetIdentity::Gva` is the
    /// authoritative content until `storage_flush::flush_gva_one` lands it.
    pub gva_deferred_flush: DeferredWindows<u64, GvaDeferredEntry>,
    /// Monotonic arm counter for [`Self::gva_deferred_flush`] oldest-first cap.
    pub gva_deferred_seq: u64,
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
    /// GVA windows whose task died (`delete_task`) — the GVA walk is gone, so
    /// the runtime lands these **cache-only** (no guest write) and unpins
    /// (`storage_flush::retire_gva_windows`).
    pub retired_gva_windows: Vec<(u64, GvaDeferredEntry)>,
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
            active_child_mask: 0,
            translation_deferred_mask: 0,
            translation_order_hold_mask: 0,
            translation_order_holds: 0,
            last_drain_wake_ms: 0,
            last_stall_report_ms: 0,
            present_translation_holds: 0,
            present_translation_hold_mask: 0,
            present_translation_hold_since: None,
            pipeline_unreadable_since: std::collections::HashMap::new(),
            pending: PendingWork::default(),
            child_rings: std::array::from_fn(|_| ChannelRing::default()),
            tasks: std::array::from_fn(|_| TaskEntry::default()),
            map_family_events: 0,
            objects: std::collections::BTreeSet::new(),
            texture_to_mapping: BTreeMap::new(),
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
            compute_deferred_flush: BTreeMap::new(),
            fence_flushed_mappings: std::collections::BTreeSet::new(),
            surface_deferred_seq: 0,
            deferred_alias_pages: DeferredWindows::new(),
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
            retired_guest_write_tokens: Vec::new(),
            retired_linear_residents: Vec::new(),
            linear_deferred_flush: DeferredWindows::new(),
            gva_deferred_flush: DeferredWindows::new(),
            gva_deferred_seq: 0,
            completion_stamp_seq: 0,
            gva_resident_backing: std::collections::BTreeMap::new(),
            retired_gva_windows: Vec::new(),
            guest_linear_memo: LruBytesMemo::new(GUEST_LINEAR_MEMO_BYTE_CAP),
            #[cfg(feature = "backend-vulkan")]
            gather_witness: crate::runtime::gather_witness::GatherWitness::default(),
            sampled_content_gen: 0,
            host_writes: crate::runtime::host_writes::HostWrites::default(),
            guest_linear_scratch: Vec::new(),
            type5_view_memo: LruBytesMemo::new(GUEST_LINEAR_MEMO_BYTE_CAP),
            type11_memo: LruBytesMemo::new(GUEST_LINEAR_MEMO_BYTE_CAP),
            type11_memo_scratch: Vec::new(),
            gva_host_views: Vec::new(),
            view_stale_reads: 0,
        }
    }

    /// Arm (or re-arm) a deferred GVA render-Store window.
    pub fn arm_gva_deferred_window(&mut self, gva: u64, entry: GvaDeferredEntry) {
        self.gva_deferred_flush.0.insert(gva, entry);
    }

    /// Arm (or re-arm) a linear compute-storage deferred window.
    pub fn arm_linear_deferred_window(
        &mut self,
        key: ComputeStorageResidencyKey,
        generation: u32,
        pages: std::collections::HashSet<u64>,
    ) {
        let armed_stamp_seq = self.completion_stamp_seq;
        self.linear_deferred_flush.0.insert(
            key,
            LinearDeferredEntry {
                generation,
                armed_stamp_seq,
                pages,
            },
        );
    }

    /// Disarm a linear compute-storage deferred window.
    ///
    /// Returns the whole window, so a caller about to write those guest pages
    /// can check they still belong to this window (see
    /// `runtime::storage_flush::deferred_pages_still_ours`) and can score the
    /// landing against the fence the window was armed under
    /// ([`LinearDeferredEntry::armed_stamp_seq`]). This used to return a bare
    /// `bool` and drop the pages on the floor, which left the flush with no way
    /// to tell that the guest had re-pointed the span since defer time — the
    /// same hazard the GVA rail already guards. `Some` still means "an entry was
    /// present", so the presence test is unchanged for callers that only want
    /// that.
    pub fn disarm_linear_deferred_window(
        &mut self,
        key: &ComputeStorageResidencyKey,
    ) -> Option<LinearDeferredEntry> {
        let entry = self.linear_deferred_flush.0.remove(key)?;
        Some(entry)
    }

    /// Detach `e`'s contiguous view for later unmap (page table changed).
    /// Returns the retired (ptr, len) to push into `retired_views`.
    fn take_mapping_view(e: &mut MappingEntry) -> Option<(usize, usize)> {
        if e.contig_ptr == 0 {
            return None;
        }
        let v = (e.contig_ptr, e.contig_len);
        e.contig_ptr = 0;
        e.contig_len = 0;
        Some(v)
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
            if let Some(view) = Self::take_mapping_view(mapping) {
                views.push(view);
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
        #[cfg(feature = "backend-vulkan")]
        tokens.extend(self.gather_witness.take_tokens());
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

    fn forget_compositor_mapping(&mut self, mapping_id: u32) {
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

    pub fn reset(&mut self) {
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

    /// Deferred GVA render-Store windows lose their GVA walk with the task —
    /// hand them to the runtime for a cache-only landing
    /// (`storage_flush::retire_gva_windows`); never write guest pages from
    /// teardown.
    ///
    /// Only this task's windows. Both sides of the comparison are slot ids:
    /// `GvaDeferredEntry::task_id` is the word `task_slot::resolve_task_word`
    /// accepted, and `DeleteTask` (`0x20`) carries a slot id too — its words
    /// include `5`, `11` and `13`, odd and greater than one, which the
    /// `DefineTask2` doubled space (`0x1`, then strictly even) does not contain,
    /// and all 968 deletes measured across the boots on disk report `ok=1`
    /// against a live slot.
    ///
    /// So a `task_id >> 1` arm here matched no window this task owns and did
    /// match every window owned by slots `2 * task_id` and `2 * task_id + 1`.
    /// Slots run densely from 0 out of [`MAX_TASKS`] = 256, and boots use ids
    /// well past 14, so those are live tasks: deleting task 5 retired tasks 10
    /// and 11's pending frames. Cache-only landing writes no guest pages, so the
    /// effect was a live task silently losing rendered pixels out of guest RAM
    /// and the guest compositing whatever those pages held before.
    ///
    /// Do not widen this back for symmetry with
    /// [`crate::runtime::gva_view::task_matches`], which deliberately keeps an
    /// aliased arm at its own overlap-retire site. The two are not the same
    /// shape, and copying that pattern here without the asymmetry is how this
    /// arrived. A *view* is a cached translation, so retiring one that did not
    /// need retiring costs a re-walk and nothing else. A *window* is pixels this
    /// device owes guest RAM, and retiring one lands it cache-only — the bytes
    /// never reach the guest and nothing re-derives them. Widening is
    /// conservative for the first and lossy for the second.
    fn retire_task_gva_windows(&mut self, task_id: u32) {
        let doomed: Vec<u64> = self
            .gva_deferred_flush
            .iter()
            .filter(|(_, e)| e.task_id == task_id)
            .map(|(&gva, _)| gva)
            .collect();
        for gva in doomed {
            if let Some(entry) = self.gva_deferred_flush.0.remove(&gva) {
                self.retired_gva_windows.push((gva, entry));
            }
        }
    }

    /// Take the deferred GVA window at exactly `gva`, if any.
    pub fn take_gva_deferred_window(&mut self, gva: u64) -> Option<GvaDeferredEntry> {
        let entry = self.gva_deferred_flush.0.remove(&gva)?;
        Some(entry)
    }

    /// Take the oldest-armed deferred GVA window (window-cap eviction).
    pub fn take_oldest_gva_deferred_window(&mut self) -> Option<(u64, GvaDeferredEntry)> {
        let gva = self
            .gva_deferred_flush
            .iter()
            .min_by_key(|(_, e)| e.armed_seq)
            .map(|(&gva, _)| gva)?;
        let entry = self.gva_deferred_flush.0.remove(&gva)?;
        Some((gva, entry))
    }

    pub fn define_task(&mut self, task_id: u32, length: u64, directory_pfn: u32) -> bool {
        if task_id as usize >= MAX_TASKS {
            StateMutationDecline::DefineTaskIdRange { task_id }.emit(u64::from(task_id));
            return false;
        }
        // Drop objects for this task on redefine.
        self.objects.retain(|&(t, _)| t != task_id);
        self.retire_task_linear_residents(task_id);
        self.retire_task_gva_windows(task_id);
        self.host_linear_textures.retain(|&(t, _), _| t != task_id);
        // New directory ⇒ old GVA HostOps views alias the wrong PT — retire.
        self.retire_task_gva_views(task_id);
        self.tasks[task_id as usize] = TaskEntry::define(length, directory_pfn);
        true
    }

    /// Retire every GVA HostOps view registered under `task_id`.
    ///
    /// Both entry points that end a task's page table — `define_task` on a
    /// redefine and `delete_task` on teardown — owe exactly this: the views hold
    /// host pointers into pages the guest is about to recycle, so leaving one
    /// live is a read of memory that no longer belongs to the surface (the
    /// WindowServer SIGSEGV class `write_span` documents). `retired_views` is
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
        if task_id as usize >= MAX_TASKS {
            StateMutationDecline::DeleteTaskIdRange { task_id }.emit(u64::from(task_id));
            return false;
        }
        if !self.tasks[task_id as usize].active {
            return false;
        }
        self.objects.retain(|&(t, _)| t != task_id);
        self.retire_task_linear_residents(task_id);
        self.retire_task_gva_windows(task_id);
        self.host_linear_textures.retain(|&(t, _), _| t != task_id);
        // Clear texture→mapping latches for this task.
        let doomed_refs: Vec<u32> = self
            .texture_to_mapping
            .keys()
            .filter_map(|&(t, r)| if t == task_id { Some(r) } else { None })
            .collect();
        self.texture_to_mapping.retain(|&(t, _), _| t != task_id);
        // Drop texture-ref encode slots that were latched for this task. Other
        // refs are left (cache is ref-keyed without task); delete_object also
        // evicts. GVA encode cache retained until Unmap of that range.
        for r in doomed_refs {
            self.host_texture_surfaces.remove(&r);
        }
        // Task teardown ≡ all GPU VA maps for this task go away — retire any
        // HostOps views we held (does not touch host_gva_surfaces encode).
        // Runtime flushes retired_views via HostOps::unmap_pages.
        self.retire_task_gva_views(task_id);
        self.tasks[task_id as usize] = TaskEntry::default();
        true
    }

    pub fn set_object_list(&mut self, task_id: u32, pfn: u32, count: u32) -> bool {
        if task_id as usize >= MAX_TASKS {
            StateMutationDecline::SetObjectListTaskIdRange { task_id }.emit(u64::from(task_id));
            return false;
        }
        if !self.tasks[task_id as usize].active {
            StateMutationDecline::SetObjectListTaskInactive { task_id }.emit(u64::from(task_id));
            return false;
        }
        self.tasks[task_id as usize].object_list_pfn = pfn;
        self.tasks[task_id as usize].object_list_count = count;
        true
    }

    pub fn insert_object(&mut self, task_id: u32, ref_: u32) -> bool {
        let discriminant = (u64::from(task_id) << 32) | u64::from(ref_);
        if task_id as usize >= MAX_TASKS {
            StateMutationDecline::InsertObjectTaskIdRange {
                task_id,
                object_ref: ref_,
            }
            .emit(discriminant);
            return false;
        }
        if !self.tasks[task_id as usize].active {
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

    pub fn delete_object(&mut self, task_id: u32, ref_: u32) -> bool {
        let removed = self.objects.remove(&(task_id, ref_));
        if removed {
            self.host_texture_surfaces.remove(&ref_);
            if let Some(e) = self.host_linear_textures.remove(&(task_id, ref_)) {
                self.retire_linear_resident(task_id, ref_, &e);
            }
            self.texture_to_mapping.remove(&(task_id, ref_));
        }
        removed
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

    /// How many deferred windows this mapping currently owes.
    ///
    /// Read before a drop so the drop can be counted: `drop_windows` reports
    /// each window it takes on the fail path but returns nothing, and a rail
    /// that needs to know whether it dropped anything must not re-derive the
    /// answer from the log.
    pub fn deferred_flush_window_count(&self, mapping_id: u32) -> u32 {
        self.compute_deferred_flush
            .keys()
            .filter(|key| key.mapping_id == mapping_id)
            .count() as u32
    }

    /// Remove one deferred window by exact key, pruning the alias index with it.
    ///
    /// For supersede: a writer that fully covers a window's guest range drops
    /// the obligation instead of landing it, and must not disturb the
    /// intersecting siblings [`Self::take_deferred_flush_windows`] would also
    /// take. Going through here rather than `compute_deferred_flush.remove`
    /// keeps the raw-GVA alias index in step — a mapping whose last window
    /// leaves must lose its page refs, or the union index keeps counting pages
    /// nothing defers on.
    pub fn take_deferred_flush_window_exact(
        &mut self,
        key: &ComputeStorageResidencyKey,
    ) -> Option<DeferredOwner> {
        let owner = self.compute_deferred_flush.remove(key)?;
        self.prune_alias_index(key.mapping_id);
        Some(owner)
    }

    /// Remove and return every deferred-writeback window intersecting
    /// `[lo, hi)` on this mapping. The caller owns flushing each returned
    /// entry (or reporting the loss) — once taken, the map no longer names it.
    pub fn take_deferred_flush_windows(
        &mut self,
        mapping_id: u32,
        lo: u64,
        hi: u64,
    ) -> Vec<(ComputeStorageResidencyKey, DeferredOwner)> {
        let keys: Vec<ComputeStorageResidencyKey> = self
            .compute_deferred_flush
            .keys()
            .filter(|key| {
                key.mapping_id == mapping_id && key.span_end > lo && key.surface_offset < hi
            })
            .cloned()
            .collect();
        let taken: Vec<(ComputeStorageResidencyKey, DeferredOwner)> = keys
            .into_iter()
            .filter_map(|key| {
                self.compute_deferred_flush
                    .remove(&key)
                    .map(|owner| (key, owner))
            })
            .collect();
        if !taken.is_empty() {
            self.prune_alias_index(mapping_id);
        }
        taken
    }

    /// Record the physical page bases of `mapping_id` in the raw-GVA alias
    /// index. Called at defer time, when `page_entries` are freshly resolved
    /// (the Store/dispatch just targeted them) — never at sample time.
    pub fn index_deferred_alias_pages(&mut self, mapping_id: u32) {
        let page_shift = self.page_shift;
        let page = self.page_size();
        let Some(m) = self.mappings.get(&mapping_id) else {
            return;
        };
        let set: std::collections::HashSet<u64> = m
            .page_entries
            .iter()
            .filter_map(|&e| crate::contract::iosurface_pages::entry_gpa_shift(e, page_shift))
            .map(|gpa| gpa & !(page - 1))
            .collect();
        if set.is_empty() {
            self.deferred_alias_pages.0.remove(&mapping_id);
        } else {
            self.deferred_alias_pages.0.insert(mapping_id, set);
        }
    }

    /// Drop the alias-index entry once no mapping-keyed deferred window names
    /// this mapping anymore.
    fn prune_alias_index(&mut self, mapping_id: u32) {
        let live = self
            .compute_deferred_flush
            .keys()
            .any(|k| k.mapping_id == mapping_id);
        if !live {
            self.deferred_alias_pages.0.remove(&mapping_id);
        }
    }

    /// Drop cached page list + contig view without unmapping the slot.
    ///
    /// Used on ReplacePhysical / rebind: guest may have recycled PFNs into the
    /// zone freelist; the next Store must re-resolve before any host write or
    /// import-present DMA (freelist `0xff000000ff000000` class).
    pub fn invalidate_mapping_pages(&mut self, mapping_id: u32) -> bool {
        let Some(e) = self.mappings.get_mut(&mapping_id) else {
            return false;
        };
        let had = !e.page_entries.is_empty() || e.contig_ptr != 0;
        e.page_entries.clear();
        e.page_table_kva = 0;
        e.condemned_entries = None;
        Self::bump_map_generation(e);
        let retired = Self::take_mapping_view(e);
        let retired_token = Self::take_guest_write_token(e);
        if let Some(v) = retired {
            self.retired_views.push(v);
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
        self.deferred_alias_pages.0.remove(&mapping_id);
        let Some(e) = self.mappings.get_mut(&mapping_id) else {
            return false;
        };
        if e.page_entries.is_empty() {
            return false;
        }
        e.condemned_entries = Some(std::mem::take(&mut e.page_entries));
        e.page_table_kva = 0;
        let retired = Self::take_mapping_view(e);
        let retired_token = Self::take_guest_write_token(e);
        if let Some(v) = retired {
            self.retired_views.push(v);
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
        if mapping_id as usize >= MAX_MAPPINGS {
            StateMutationDecline::MapSurfaceIdRange { mapping_id }.emit(u64::from(mapping_id));
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
        let retired = Self::take_mapping_view(e);
        let retired_token = Self::take_guest_write_token(e);
        if let Some(v) = retired {
            self.retired_views.push(v);
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

    /// Tell [`crate::observe::footprint`] that a mapping's pages have stopped
    /// being a surface's, so a later write into them is reportable.
    ///
    /// Only the guest's own Unmap calls this. The device's internal
    /// invalidations — a resolve that failed, a condemned list awaiting a
    /// fingerprint compare — are *this device* deciding it no longer trusts a
    /// list, not the guest saying the memory is no longer a surface's, and the
    /// reprieve path can hand the same list straight back. Retiring on those
    /// would flag the reprieve's own legitimate writes, and a detector whose
    /// first finding is its own bookkeeping gets switched off before it ever
    /// reports a real one.
    ///
    /// Frames another live mapping still names are excluded: two mappings can
    /// alias the same guest pages, and the survivor's writes are not a defect.
    ///
    /// # The condemned list is part of the doomed set, and reading only
    /// `page_entries` made this dead code
    ///
    /// Both routes into [`Self::unmap_surface`] from a guest teardown arrive
    /// with `page_entries` already **empty**, because the list was moved into
    /// `condemned_entries` by the step before:
    ///
    /// - `DeleteIOSurfaceBacking2` first calls [`Self::condemn_surface_backing`],
    ///   which does exactly that move. A second delete with no resolve between
    ///   then takes the `mapping_backing_condemned` branch to `unmap_surface` —
    ///   where the only list is the condemned one.
    /// - The same delete falls through to `unmap_surface` directly when
    ///   `condemn_surface_backing` returns `false`, and it returns `false`
    ///   *precisely when* `page_entries` is empty.
    /// - [`Self::map_surface`] moves the list the same way, so a fresh MAP
    ///   followed by an Unmap is the third case.
    ///
    /// So an `is_empty()` bail on `page_entries` alone could never retire a page
    /// on the delete path. Measured: a 600 s driven boot reported
    /// `retire_scans=0`, which made its `write_after_retire=0` UNMEASURED rather
    /// than clean — the failure direction this project's rules call out, a
    /// detector reading zero because it never ran.
    ///
    /// Retiring the condemned list *here* is not the same as retiring at
    /// condemn time, which would be wrong for the reason above: at condemn the
    /// reprieve can still hand the list straight back. By the time this runs the
    /// guest has said the backing is gone and no resolve has re-adopted it —
    /// `resolve_mapping_backing` takes `condemned_entries` and calls
    /// `note_pages_authorized` on whatever it adopts, so a reprieved list is
    /// un-retired through the ordinary adoption path before it can be written.
    ///
    /// Other mappings' condemned lists count as still-held for the same reason,
    /// in the conservative direction: a slot awaiting its fingerprint compare
    /// may be reprieved, and its writes would then be legitimate.
    fn note_mapping_pages_retired(&self, mapping_id: u32) {
        let Some(doomed) = self.mappings.get(&mapping_id) else {
            return;
        };
        let mut going: Vec<u32> = doomed.page_entries.clone();
        going.extend_from_slice(doomed.condemned_entries.as_deref().unwrap_or(&[]));
        self.retire_pages_no_live_mapping_holds(&going, Some(mapping_id));
    }

    /// Retire every page in `going` that no still-live mapping names.
    ///
    /// `skip` is the mapping whose own lists are the doomed ones and must not
    /// therefore count as holding them. Pass `None` when the pages are being
    /// abandoned by a mapping that is *still live under a new backing* — a
    /// superseded incarnation — because there the mapping's current
    /// `page_entries` are the new plan, and a page carried over into it is
    /// genuinely still a surface's and must be kept out of the retired set.
    ///
    /// Other mappings' *condemned* lists count as held, deliberately in the
    /// conservative direction: a slot awaiting its fingerprint compare may be
    /// reprieved, and its writes would then be legitimate.
    pub(crate) fn retire_pages_no_live_mapping_holds(&self, going: &[u32], skip: Option<u32>) {
        if going.is_empty() {
            return;
        }
        let shift = self.page_shift;
        let gpas_of = |entries: &[u32]| -> Vec<u64> {
            entries
                .iter()
                .filter_map(|&e| crate::contract::iosurface_pages::entry_gpa_shift(e, shift))
                .collect()
        };
        let mut still_held: std::collections::HashSet<u64> = Default::default();
        let mut walked = 0u64;
        for (&other, m) in self.mappings.iter() {
            if Some(other) == skip {
                continue;
            }
            let condemned = m.condemned_entries.as_deref().unwrap_or(&[]);
            walked += (m.page_entries.len() + condemned.len()) as u64;
            still_held.extend(gpas_of(&m.page_entries));
            still_held.extend(gpas_of(condemned));
        }
        // Reported rather than assumed small. This runs on the drain worker,
        // which `drain_duty` already shows at 0.93-0.99, and the alias exclusion
        // costs one pass over everything currently mapped per retire.
        crate::observe::footprint::note_retire_scan(walked);
        let retiring: Vec<u64> = gpas_of(going)
            .into_iter()
            .filter(|g| !still_held.contains(g))
            .collect();
        crate::observe::footprint::note_pages_retired(retiring, self.page_size());
    }

    pub fn unmap_surface(&mut self, mapping_id: u32) -> bool {
        if mapping_id as usize >= MAX_MAPPINGS {
            StateMutationDecline::UnmapSurfaceIdRange { mapping_id }.emit(u64::from(mapping_id));
            return false;
        }
        // Before the list is cleared, while the pages are still nameable.
        self.note_mapping_pages_retired(mapping_id);
        self.forget_compositor_mapping(mapping_id);
        if let Some(e) = self.mappings.get_mut(&mapping_id) {
            e.mapped = false;
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
            let retired = Self::take_mapping_view(e);
            let retired_token = Self::take_guest_write_token(e);
            if let Some(v) = retired {
                self.retired_views.push(v);
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
        if mapping_id as usize >= MAX_MAPPINGS {
            StateMutationDecline::AttachMappingIdRange { mapping_id }.emit(u64::from(mapping_id));
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
        let retired = Self::take_mapping_view(e);
        let retired_token = Self::take_guest_write_token(e);
        if let Some(v) = retired {
            self.retired_views.push(v);
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
        if mapping_id as usize >= MAX_MAPPINGS {
            StateMutationDecline::MappingDeviceDescIdRange { mapping_id }
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
        if mapping_id as usize >= MAX_MAPPINGS {
            StateMutationDecline::MappingGeomIdRange { mapping_id }.emit(u64::from(mapping_id));
            return false;
        }
        if width == 0 {
            StateMutationDecline::MappingGeomWidthZero { mapping_id }.emit(u64::from(mapping_id));
            return false;
        }
        if height == 0 {
            StateMutationDecline::MappingGeomHeightZero { mapping_id }.emit(u64::from(mapping_id));
            return false;
        }
        if width > crate::model::MAX_SCANOUT_DIM {
            StateMutationDecline::MappingGeomWidthRange { mapping_id, width }
                .emit((u64::from(mapping_id) << 32) | u64::from(width));
            return false;
        }
        if height > crate::model::MAX_SCANOUT_DIM {
            StateMutationDecline::MappingGeomHeightRange { mapping_id, height }
                .emit((u64::from(mapping_id) << 32) | u64::from(height));
            return false;
        }
        let e = self.mappings.entry(mapping_id).or_default();
        // Geom change (mode switch / rematerialize) is a new surface identity:
        // reset content_generation (the guest pages stay authoritative).
        if e.width != width || e.height != height {
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

    /// The same, for a writer that knows which mapping's pages it is landing in.
    pub fn note_host_wrote_mapping(&mut self, mapping_id: u32) {
        // A mapping with no page list cannot have its write ruled out later, so
        // it is recorded as an unnamed one rather than as an empty page set.
        match self.mappings.get(&mapping_id) {
            Some(m) if !m.page_entries.is_empty() => {
                let generation = m.map_generation;
                self.host_writes.note_mapping(mapping_id, generation);
            }
            _ => self.host_writes.note_unknown(),
        }
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
    /// completeness property the type-11 `LoadFromTarget` gate rests on.
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

    /// Advance [`MappingEntry::surface_content_epoch`] for a publish that
    /// changed the mapping's pixels *without* writing its guest pages — the
    /// deferred type-11 Store, which stores the frame into `surface_cache` and
    /// arms a window. Returns the new epoch so the caller can stamp the
    /// resident that holds those pixels in the same breath; the two must not be
    /// separable, or the stamp records a currency that already moved.
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
            UNKNOWN_OPCODE_ECHO_WORDS + 1,
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
            StateMutationDecline::DefineTaskIdRange { task_id: 64 },
            StateMutationDecline::DeleteTaskIdRange { task_id: 64 },
            StateMutationDecline::SetObjectListTaskIdRange { task_id: 64 },
            StateMutationDecline::SetObjectListTaskInactive { task_id: 1 },
            StateMutationDecline::InsertObjectTaskIdRange {
                task_id: 64,
                object_ref: 3,
            },
            StateMutationDecline::InsertObjectTaskInactive {
                task_id: 1,
                object_ref: 3,
            },
            StateMutationDecline::MapSurfaceIdRange { mapping_id: 8192 },
            StateMutationDecline::UnmapSurfaceIdRange { mapping_id: 8192 },
            StateMutationDecline::AttachMappingIdRange { mapping_id: 8192 },
            StateMutationDecline::AttachMappingInternalZero { mapping_id: 1 },
            StateMutationDecline::MappingDeviceDescIdRange { mapping_id: 8192 },
            StateMutationDecline::MappingDeviceDescEmpty { mapping_id: 1 },
            StateMutationDecline::MappingGeomIdRange { mapping_id: 8192 },
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
            17,
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

    #[test]
    fn invalid_mapping_geometry_cannot_create_an_out_of_range_slot() {
        let mut state = DeviceState::new(DeviceId(1), crate::model::PAGE_SHIFT_X86);
        let bad_mapping = MAX_MAPPINGS as u32;
        assert!(!state.set_mapping_geom(bad_mapping, 64, 64, 0x50));
        assert!(!state.mappings.contains_key(&bad_mapping));
        assert!(!state.set_mapping_geom(1, 0, 64, 0x50));
        assert!(!state.set_mapping_geom(1, 64, 0, 0x50));
        assert!(!state.mappings.contains_key(&1));
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
