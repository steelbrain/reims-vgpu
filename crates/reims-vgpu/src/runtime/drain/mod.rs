//! Root/child FIFO drains, stamp writeback, and fail-visible command dispatch.
//!
//! Prefer structure correctness over full exec.c coverage: known root/child
//! control-plane ops update device state; unknown opcodes are recorded visibly.

use crate::contract::endian::{ld16, ld32, ld64, st16, st32};
use crate::contract::iosurface_pages::{
    MAPPER_REQUEST_ENTRY_LEN, MAPPER_REQUEST_MAP, MAPPER_REQUEST_MAPPING_ID, MAPPER_REQUEST_TYPE,
    MAPPER_REQUEST_UNMAP,
};
use crate::model::*;
use crate::model::{DeviceState, ExecFault, FailEvent, PacketFault, UnimplementedCommand};
use crate::observe::Emit;
use crate::runtime::decode::fifo::{
    display_refresh_hz_1616, display_timing_entry_offset, encode_display_timing_entry,
    DisplayTimingEntry, DISPLAY_DESC_TIMING_STRIDE,
};
use crate::runtime::gpa_map;
use crate::runtime::heap_query::QueryError;
use crate::runtime::host::{HostAction, HostMemory, HostOps, MemError};
use crate::runtime::task_slot::{resolve_task_word, TaskWordSite};

pub(crate) mod census;
pub use census::*;

/// Score a bound-buffer retirement against the cause that ordered it.
///
/// Counted in **entries dropped**, not events: one `SetObjectList` that retires
/// forty resolutions and one that retires none are the same event and very
/// different costs, because every dropped entry is a task-page-table walk the
/// next draw pays again.
///
/// What this is for. The registry answers 86% of draw-time buffer binds on a
/// host that can import guest RAM and 98% on one that cannot — a 12.8x
/// difference in fresh resolutions that the two rails' throughput does not
/// explain. A miss is either a key never seen or a key retired, and these
/// counters are the second half. Sum them against `zc_buffer_imported` +
/// `zc_buffer_gathered`: what the retirements do not account for is churn in the
/// keys themselves, which is the `(task, reference, offset)` question
/// [`crate::runtime::bound_buffers`] states and does not answer.
#[inline]
fn note_bb_retired(cause: &'static str, entries: usize) {
    note_store_route_n(cause, entries as u64);
}

/// apple-gfx `pending_frames >= 2`: hold further guest presents at FIFO head
/// until host paint consumes +0x188. Entry-side waitForPendingFrames — not
/// stamp-after-paint (that inverted PGDisplay completion and stacked tooltips).
pub const MAX_UNPAINTED_PRESENTS: u32 = 2;

/// Bit 0 names the root FIFO in every mask over FIFO timelines —
/// `translation_order_hold_mask` and `stamp_deferred_mask`. Child FIFOs use
/// their channel bit, matching `translation_deferred_mask`.
///
/// Bit 0 is free for the root because `is_child_channel` starts child ids at 1,
/// so no channel bit can collide with it. The constant is not named for either
/// mask: it is the shared convention, and naming it after the first user is how
/// the second one ends up spelling `1` by hand.
const ROOT_FIFO_BIT: u32 = 1;

fn note_translation_order_hold(state: &mut DeviceState, held_mask: u32) {
    let new_mask = held_mask & !state.translation_order_hold_mask;
    if new_mask == 0 {
        return;
    }
    let starts_episode = state.translation_order_hold_mask == 0;
    state.translation_order_hold_mask |= new_mask;
    if starts_episode {
        state.translation_order_holds = state.translation_order_holds.saturating_add(1);
    }
    // Census, not a failure: this is a resolver saying "not ready yet". The FIFO
    // is parked until the AIR module loads and `release_translation_order_holds`
    // takes the mask back down — and its release line was already `off`, so
    // logging the wait half as a failure made one control-flow pair straddle both
    // channels. Boot 87: 34 episodes started, 35 released, i.e. every one. A hold
    // that never releases is caught at `DeviceState::reset` instead, where the
    // guest's own teardown is the deadline and no age or depth has to be invented.
    crate::observe::off(format!(
        "translation_order_hold reason=air_loading held_mask={:#x} new_mask={new_mask:#x} producer_mask={:#x} count={}",
        state.translation_order_hold_mask,
        state.translation_deferred_mask,
        state.translation_order_holds
    ));
}

fn release_translation_order_holds(state: &mut DeviceState) {
    if state.translation_deferred_mask != 0 || state.translation_order_hold_mask == 0 {
        return;
    }
    let held_mask = std::mem::take(&mut state.translation_order_hold_mask);
    crate::observe::off(format!(
        "translation_order_release held_mask={held_mask:#x} producer_mask=0x0"
    ));
}

/// Install a `DefineTask2` payload and record the page-table identity it
/// resolved to.
///
/// The root ring and every child channel carry the same opcode with the same
/// payload layout. `site` names which ring so the two populations stay
/// separable on the census line; nothing else about them differs, and they
/// used to be two copies that could drift — the child arm reading a
/// four-byte length where the root read eight is exactly that having happened.
///
/// A short payload drops the task definition, and every later draw or resolve
/// on that task then fails downstream with no root cause. The child ring named
/// that; the root ring did not, and dropped silently. Both name it now.
/// True when a control packet is too short to carry the fields its opcode
/// needs, having said so.
///
/// Every arm that guards on payload length is acknowledged regardless of
/// whether it did anything: `drain_main_fifo` writes the root completion stamp
/// after the dispatch match, and `drain_child_fifo` calls `write_stamp` the
/// same way. So an arm that just skips on a short payload tells the guest its
/// command completed while nothing happened, and leaves the fail log empty —
/// the worst shape a loss can take, because the symptom surfaces arbitrarily
/// far downstream (a channel that never drains, an object list that never
/// binds) with no record of the cause.
///
/// `site` separates the two rings on the census line the way
/// [`apply_define_task2`] does, since the same opcode arrives on both.
/// `CmdNOP` (`0x1e`) arrived carrying a payload.
///
/// The command allocates no command-space bytes, so a payload is the one thing
/// that can falsify this device's reading of it: bytes here mean the command has
/// grown a form this arm does not decode, and dropping them silently is the
/// failure the unknown-opcode arm was at least loud about.
///
/// Typed rather than a formatted string so the check owns a slug of its own.
/// This is the same rule that keeps two commands from sharing one decline — a
/// reader who greps the slug has to be able to tell which check fired.
pub(crate) struct NopCarriedPayload {
    pub(crate) channel: u32,
    pub(crate) plen: usize,
}

impl crate::observe::Decline for NopCarriedPayload {
    fn slug(&self) -> &'static str {
        "cmd_nop_unexpected_payload"
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("ch", self.channel.to_string()),
            ("plen", self.plen.to_string()),
        ]
    }
}

crate::observe::decline_display!(NopCarriedPayload);

/// `CmdDeleteObject` (`0x28`) carried something that is not a destroy record.
///
/// The command's payload is a task id and one serializer record naming the
/// object to retire. Each variant is a distinct way that record can fail to name
/// one, and each owns its slug: a reader who sees the reason has to be able to
/// tell a truncated record from a well-formed record of the wrong family, since
/// only the second means this device is meeting a command shape it has not
/// decoded.
pub(crate) enum DeleteObjectDecline {
    /// The bytes after the task id are not a well-formed record.
    RecordMalformed { task: u32, plen: usize },
    /// A well-formed record whose opcode is not one of the destroy family.
    ///
    /// Membership is tested against the family, never a range: numbers inside
    /// the destroy span belong to no destroy selector, and retiring an object on
    /// one would destroy something the record never named.
    NotADestroy { task: u32, opcode: u32 },
    /// A destroy record whose payload is not the single object ref it must be.
    RefUnreadable { task: u32, opcode: u32 },
}

impl crate::observe::Decline for DeleteObjectDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::RecordMalformed { .. } => "delete_object_record_malformed",
            Self::NotADestroy { .. } => "delete_object_not_a_destroy_record",
            Self::RefUnreadable { .. } => "delete_object_ref_unreadable",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::RecordMalformed { task, plen } => {
                vec![("task", task.to_string()), ("plen", plen.to_string())]
            }
            Self::NotADestroy { task, opcode } | Self::RefUnreadable { task, opcode } => {
                vec![
                    ("task", task.to_string()),
                    ("record_opcode", format!("{opcode:#x}")),
                ]
            }
        }
    }
}

crate::observe::decline_display!(DeleteObjectDecline);

/// The object kind a destroy record names, as a route-counter name.
///
/// The kind lives in the record's own opcode and nowhere else, so this is the
/// only place it can be read from. Each kind gets its own counter because the
/// distribution is the open question this arm leaves behind: which kinds a guest
/// actually retires this way is what decides whether tracking serializer refs
/// would be worth anything, and one merged counter cannot answer it.
///
/// Spelled against the wire crate's constants rather than against literals, so
/// a renumbering there fails the build here instead of silently re-labelling a
/// counter.
fn delete_object_kind_route(opcode: u32) -> &'static str {
    use reims_vgpu_wire::ops::destroy as d;
    match opcode {
        d::OPCODE_DELETE_BUFFER => "child_delete_object_buffer",
        d::OPCODE_DELETE_TEXTURE => "child_delete_object_texture",
        d::OPCODE_DELETE_DEPTH_STENCIL_STATE => "child_delete_object_depth_stencil_state",
        d::OPCODE_DELETE_SAMPLER_STATE => "child_delete_object_sampler_state",
        d::OPCODE_DELETE_FUNCTION => "child_delete_object_function",
        d::OPCODE_DELETE_COMPUTE_PIPELINE_STATE => "child_delete_object_compute_pipeline_state",
        d::OPCODE_DELETE_RENDER_PIPELINE_STATE => "child_delete_object_render_pipeline_state",
        d::OPCODE_DELETE_FENCE => "child_delete_object_fence",
        d::OPCODE_DELETE_HEAP => "child_delete_object_heap",
        d::OPCODE_DELETE_RASTERIZATION_RATE_MAP => "child_delete_object_rasterization_rate_map",
        d::OPCODE_DELETE_INDIRECT_COMMAND_BUFFER => "child_delete_object_indirect_command_buffer",
        // `is_delete` gates every caller, so this is unreachable rather than a
        // silent bucket. It is named so that if the family ever grows a kind
        // this table has not learned, the counter says which.
        _ => "child_delete_object_unnamed_kind",
    }
}

/// `CmdDeleteObject` (`0x28`): decode the record, name the kind, retire nothing.
///
/// The payload is `{u32 task}` then one serializer destroy record. Every kind in
/// that family writes the identical twelve-byte body — a single object ref — and
/// carries the kind in the record's own opcode.
///
/// # Why the ref is decoded and then not acted on
///
/// The ref is in the **serializer's per-kind ref space**, which this device does
/// not track. Nothing here is a matter of taste: the object table is keyed by
/// the *kernel object-list* ref, established by `0x33 CmdSetObjectList` and
/// populated when a decoded command resolves an entry out of that list, and the
/// caches holding the kinds this command actually names — samplers and pipeline
/// states — are keyed by the object's own **state**, not by a ref, so they have
/// no ref to retire by even in principle.
///
/// Keying the object table with a number from the other namespace was measured
/// and is worse than declining. On a driven boot the guest sent 1 988 of these
/// in 25 s, every sampled one a sampler-state destroy, and not one named a ref
/// the object table held — against 112 466 successful lookups on that same
/// `(task, ref)` key from the resource-table path in the same boot. So the table
/// was healthy and the namespace was wrong. The two do overlap numerically —
/// 22 of those refs existed under a *different* task — so the only effect the
/// call could ever have had is destroying an unrelated object that happened to
/// share the integer.
///
/// This costs the guest nothing. The kinds named are content-keyed and
/// deduplicated, so there is no per-object allocation to leak; what is lost is
/// the chance to drop a cache entry slightly earlier, which is why the decline
/// is a decline and not a loss.
///
/// The per-kind counters are what would justify revisiting it: a kind this
/// device *does* hold by ref appearing here is the finding, and nothing else is.
/// There is exactly one such kind today. `DeviceState::fence_generations` is
/// keyed `(task, domain, fence_ref)` and nothing ever removes an entry, so a
/// `child_delete_object_fence` reading above zero would be both a real leak and
/// the one place this command has something to retire. The driven boot measured
/// it at zero — every sampled record was a sampler state — so wiring it now
/// would be building a handler for traffic no boot has produced. Read that
/// counter before deciding otherwise.
fn apply_delete_object(state: &mut DeviceState, channel_id: u32, payload: &[u8], packet: &Packet) {
    let task_id = ld32(&payload[0..]);
    let record = &payload[4..];
    let Ok(op) = reims_vgpu_wire::op::op(record, 0) else {
        Emit::decline(
            "child_delete_object",
            &DeleteObjectDecline::RecordMalformed {
                task: task_id,
                plen: record.len(),
            },
        )
        .field("ch", channel_id)
        .fail();
        return;
    };
    if !reims_vgpu_wire::ops::destroy::is_delete(op.opcode()) {
        Emit::decline(
            "child_delete_object",
            &DeleteObjectDecline::NotADestroy {
                task: task_id,
                opcode: op.opcode(),
            },
        )
        .field("ch", channel_id)
        .fail();
        return;
    }
    let Ok(rec) = reims_vgpu_wire::ops::destroy::delete(&op) else {
        Emit::decline(
            "child_delete_object",
            &DeleteObjectDecline::RefUnreadable {
                task: task_id,
                opcode: op.opcode(),
            },
        )
        .field("ch", channel_id)
        .fail();
        return;
    };
    // Read so the record is proved decodable and the kind is countable. It is
    // deliberately not used as a key: see this function's doc for the boot that
    // measured what keying the object table with it would have done.
    let _object_ref = rec.object_ref.get();
    note_store_route(delete_object_kind_route(op.opcode()));
    note_unimplemented(
        state,
        channel_id,
        UnimplementedCommand::DeleteObject,
        packet,
    );
}

fn packet_short(op: &'static str, channel: Option<u32>, have: usize, need: usize) -> bool {
    if have >= need {
        return false;
    }
    crate::observe::fail(format!(
        "packet_short reason={op}_short site={} plen={have} need={need}",
        packet_site(channel)
    ));
    true
}

/// Which FIFO a packet arrived on, for a log line.
///
/// One spelling, because several opcodes are carried on **both** the root FIFO
/// and a child channel and are handled by one function for each — so a reader
/// filtering the log for a rail needs the two arms to name it the same way.
/// `apply_define_task2` used to spell the child arm `child ch=4` against
/// `packet_short`'s `ch4`, which is the whole reason this is a function.
fn packet_site(channel: Option<u32>) -> String {
    match channel {
        Some(ch) => format!("ch{ch}"),
        None => "root".to_string(),
    }
}

/// `CmdDeleteTask` (0x20), from either FIFO.
///
/// The guest recycles task ids, so a delete that does not land leaves a later
/// `DefineTask2` writing into a slot that still holds the previous task's page
/// directory. A short payload is not refused: the live shape is a 12-byte header
/// with a 4-byte id, and a packet carrying none names task 0 — which is the
/// kernel task, so the length is reported on the line rather than guessed at.
/// `CmdDisplaySetSharedStatePage`: the display pipe hands the device the guest
/// page its shared state lives on.
///
/// Reachable from both dispatch tables because it is one command in one flat
/// opcode space — see `ROOT_OP_SETUP_SHARED_STATE`'s own declaration, which
/// carries what established that and what the root arm used to do instead.
/// (Not a doc link: `model::regs` is a private module, so no path to the
/// constant resolves from here.)
/// `channel` is `None` for the root channel, matching `apply_delete_task` and
/// the other handlers both tables share.
fn apply_setup_shared_state<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    payload: &[u8],
    channel: Option<u32>,
) {
    // A short SETUP_SHARED_STATE drops display registration:
    // shared_gpa/index never latch, so the display NEVER onlines and the
    // boot wedges on a blank/console frame. The loudest of this class.
    if packet_short(
        "setup_shared_state",
        channel,
        payload.len(),
        CHILD_SHARED_STATE_LEN,
    ) {
        return;
    }
    let index = ld32(&payload[CHILD_SHARED_STATE_INDEX..]);
    let pfn = ld32(&payload[CHILD_SHARED_STATE_PFN..]);
    // reinit=1 means the guest tears down + re-registers the display
    // shared page while it was already ONLINE — the AppleParavirtDisplayPipe
    // setupSharedState/teardownSharedState re-init that makes WindowServer
    // rebuild display attributes (signalDisplay bit2 → process_online).
    // A reinit AFTER present_converge is the smoking gun for the intermittent
    // post-converge boot-progress overlay. Rare
    // event → always-on so a bad boot leaves a display-lifecycle timeline.
    let reinit = state.display.online_acked as u8;
    state.display.display_index = index;
    state.display.shared_gpa = state.pfn_gpa(pfn);
    state.display.online_acked = false;
    state.display.online_tries = 0;
    state.display.poll_ctr = 0;
    crate::observe::fail(format!(
        "display_shared_state_setup index={index} gpa={:#x} reinit={reinit} ch={}",
        state.display.shared_gpa,
        channel.map_or_else(|| "root".to_string(), |c| c.to_string())
    ));
    // Archive apple_pv_gpu_display_setup: fill descriptor + modes
    // before completion so createDisplayAttributes sees TimingElements.
    // Do **not** pulse ONLINE here — enable() has not set +0x104 yet
    // (archive poll waits for mask bit 2, then pending+IRQ).
    fill_display_descriptor(host, state.display.shared_gpa, index, state.page_size());
}

fn apply_delete_task(state: &mut DeviceState, payload: &[u8], channel: Option<u32>) {
    let task_id = if payload.len() >= 4 {
        ld32(&payload[0..])
    } else {
        0
    };
    // The task's GVA space goes with it, so every bind resolution keyed on it
    // names bytes that are no longer this task's. Here rather than at the two
    // call sites: the root and child FIFOs both reach this, and a rule written
    // twice is the one that diverges.
    note_bb_retired("bb_retire_delete_task", state.retire_bound_buffers_for_task(task_id));
    let ok = state.delete_task(task_id);
    crate::observe::off(format!(
        "delete_task site={} task={task_id} ok={} plen={}",
        packet_site(channel),
        ok as u8,
        payload.len()
    ));
}

/// `CmdSetObjectList`, from either FIFO.
///
/// # The one `apply_*` in this file that recorded nothing
///
/// Every sibling emits an `off` line naming what it took from the packet; this
/// one wrote two fields into the task and said nothing, and those two fields are
/// the whole address of every object-list read the device then makes. So a
/// driven boot reports
///
/// ```text
/// object_list_entry_unreadable task=2 ref=32 gva=0x1180 list_pfn=1 list_count=1048576
/// ```
///
/// — the same `list_pfn=1 list_count=1048576` for tasks 2, 3, 4 and 6, which is
/// a one-million-entry object list at guest page 1 — and nothing in the log says
/// whether the guest sent those words or this decode produced them. That is the
/// question the line below exists to answer, and it cannot be answered by
/// reading the source: both readings are consistent with a 12-byte payload of
/// three `u32`s.
///
/// `plen` is on the line for the same reason. `packet_short` only refuses a
/// payload *below* `SET_OBJECT_LIST_LEN`, so a longer one is accepted and its
/// tail ignored — and a longer packet is what a wider field (a 64-bit page
/// address, a byte length beside a count) would look like from here.
///
/// Emitted on the `OFF` channel and deduped per `(task, pfn, count)`: this is a
/// lifecycle packet the guest re-sends, and the reading wanted is which distinct
/// triples a boot produces, not how often.
fn apply_set_object_list(state: &mut DeviceState, payload: &[u8], channel: Option<u32>) {
    if packet_short(
        "set_object_list",
        channel,
        payload.len(),
        SET_OBJECT_LIST_LEN,
    ) {
        return;
    }
    let task_id = ld32(&payload[SET_OBJECT_LIST_TASK_ID..]);
    let pfn = ld32(&payload[SET_OBJECT_LIST_PFN..]);
    let count = ld32(&payload[SET_OBJECT_LIST_COUNT..]);
    // A new object list re-points every reference on this task, so no
    // resolution keyed by reference survives it. Both FIFOs reach this.
    note_bb_retired(
        "bb_retire_set_object_list",
        state.retire_bound_buffers_for_task(task_id),
    );
    let applied = state.set_object_list(task_id, pfn, count);
    if crate::observe::first_sight(
        "set_object_list",
        (u64::from(task_id) << 48) ^ (u64::from(pfn) << 24) ^ u64::from(count),
    ) {
        crate::observe::off(format!(
            "set_object_list site={} task={task_id} pfn={pfn:#x} count={count} \
             applied={} plen={}",
            packet_site(channel),
            applied as u8,
            payload.len()
        ));
    }
}

fn apply_define_task2<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    payload: &[u8],
    channel: Option<u32>,
) {
    if packet_short("define_task2", channel, payload.len(), DEFINE_TASK_LEN) {
        return;
    }
    let raw_id = ld32(&payload[DEFINE_TASK_RAW_ID..]);
    let length = define_task_length(payload);
    let dir = ld32(&payload[DEFINE_TASK_DIRECTORY_PFN..]);
    // `raw_id` is `(task_id << 1) | is_kernel_task`: the guest's kernel-task and
    // user-task registrations differ only in that low bit, and the kernel task's
    // own id is 0, so `0x1` is the kernel task and not user task 1. Both halves
    // are decoded — the id to index the slot, the flag so the log says which
    // class registered rather than leaving the bit unaccounted for.
    let task_id = raw_id >> DEFINE_TASK_ID_SHIFT;
    let kernel_task = raw_id & 1 != 0;
    // The page-table root is replaced here, so every GVA this task resolved
    // through the old one may translate elsewhere now. Keyed on the shifted
    // `task_id`, not `raw_id` — the registry is keyed the way the draw path
    // keys it, and `raw_id` is the wrong number by a factor of two.
    note_bb_retired("bb_retire_define_task2", state.retire_bound_buffers_for_task(task_id));
    state.define_task(task_id, length, dir);
    // Capture directory + root/depth so one boot shows the page-table identity.
    let Some(slot) = state.tasks.get(task_id) else {
        return;
    };
    let walk =
        crate::runtime::gva_mem::diagnose_task_slot(host, slot, task_id, 0, state.page_shift);
    // `task=`/`dir=` are not repeated here: `walk` already carries the task id
    // as `tid=` and the directory as `dir=`, and a key printed twice in one line
    // is a field every log reader resolves arbitrarily.
    crate::observe::off(format!(
        "define_task site={} raw={raw_id:#x} kernel={} len={length:#x} page_shift={} {walk}",
        packet_site(channel),
        kernel_task as u8,
        state.page_shift
    ));
}

/// The task's address-space length from a `DefineTask2` payload.
///
/// The field is 64 bits wide, and the payload layout says so: it sits at
/// `DEFINE_TASK_LENGTH` (0x04) and the next field, `DEFINE_TASK_DIRECTORY_PFN`,
/// is at 0x0c — eight bytes later. Callers must have already checked
/// `payload.len() >= DEFINE_TASK_LEN` (16), which covers the whole field.
///
/// The root and child arms decode the same wire field, and read it the same
/// way here. The child arm used to take only the low 32 bits, which truncated
/// the length of any task spanning 4 GiB or more.
fn define_task_length(payload: &[u8]) -> u64 {
    ld64(&payload[DEFINE_TASK_LENGTH..])
}

/// Trailer size `submitTransaction` appends after serializing the transaction's
/// resource list: `[pipe][task][surface][gamma…]` for the gamma command,
/// `[pipe][surface][task]` otherwise.
fn display_txn_trailer_len(opcode: u16) -> usize {
    if opcode == CHILD_OP_DISPLAY_TRANSACTION3 {
        0x24
    } else {
        0x0c
    }
}

/// Word slots of the surface id and the task field within the trailer, keyed on
/// the command that emitted it.
///
/// These are three different FIFO commands, not one shape with variations, so
/// nothing here may be assumed from one opcode to another:
///
/// - op6 `CmdDisplayTransaction2_DEPRECATED` — `[pipe][surface][task]`.
/// - op7 `CmdDisplayTransaction3` — `[pipe][task][surface][gamma…]`; the first
///   two words are swapped relative to op6, and the gamma table follows. The two
///   are separate commands with separate handlers, not one command with a
///   variant flag, which is why the trailer differs at all.
/// - op8 `CmdDisplaySwapMapping` — `[display][_][mapping]`. This one names a
///   single mapping instead of serializing a transaction, so its surface word
///   is `DISPLAY_SWAP_MAPPING` (0x08), *not* op6's 0x04, and it has no known
///   task field at all. Reading it at op6's slot would return the unidentified
///   word between the display index and the mapping.
///
/// Slots are `regs.rs`'s `PRESENT_*` / `DISPLAY_SWAP_*` byte offsets divided by
/// four; this is the only place either reading of those offsets is spelled out,
/// so the present path and the payload census cannot decode the field
/// differently.
fn display_txn_trailer_slots(opcode: u16) -> (usize, Option<usize>) {
    match opcode {
        CHILD_OP_DISPLAY_TRANSACTION3 => (DISPLAY_TRANSACTION3_SURFACE_ID / 4, Some(1)),
        CHILD_OP_DISPLAY_SWAP => (DISPLAY_SWAP_MAPPING / 4, None),
        _ => (DISPLAY_TRANSACTION2_SURFACE_ID / 4, Some(2)),
    }
}

/// The surface id a display-present payload names, read from offset zero.
///
/// The slot is the one `display_txn_trailer_slots` gives for the emitting
/// command, so this head-relative reading and the tail-relative reading the
/// census takes cannot drift apart.
///
/// `None` is a payload too short to hold the command's own trailer. That loses
/// the present outright, so callers must name it rather than presenting mapping
/// zero and completing the packet in silence.
fn present_surface_id(opcode: u16, payload: &[u8]) -> Option<u32> {
    let trailer = display_txn_trailer_len(opcode);
    (payload.len() >= trailer).then(|| {
        let off = display_txn_trailer_slots(opcode).0 * 4;
        ld32(&payload[off..])
    })
}

/// Alarm for a display-transaction payload longer than its command declares.
///
/// # The wire shape, and why there is nothing left to sample here
///
/// A display present is an `IOAccelDisplayPipeTransaction2` on the guest side —
/// a per-frame list of planes carrying source, destination and dirty rects — and
/// this device decodes only a single surface id from it. That reads like a
/// truncation, and a sampler used to sit here recording whether the rest of the
/// list rode inline in the payload.
///
/// It does not, and it never can: the guest's display pipe serializes the
/// transaction by taking **plane 0's** IOSurface and writing that one id into a
/// fixed-size command. There is no plane list on the wire, so decoding one
/// surface is the whole contract rather than a first approximation of it. The
/// command is 12 bytes for `CmdDisplayTransaction2_DEPRECATED` and 36 for
/// `CmdDisplayTransaction3`, which is what [`display_txn_trailer_len`] returns,
/// and the field order differs between them exactly as
/// [`display_txn_trailer_slots`] says.
///
/// The same reading settles the third word. It is the id of the paravirt task
/// that owns the presented surface, taken from the display pipe's own resource
/// heap rather than from the transaction — so a zero there means the pipe has no
/// task bound yet, not that the field is unused.
///
/// One consequence worth stating because it is easy to go looking for: **the
/// guest's damage rects never reach this device.** They exist in the
/// transaction, and the serializer drops them. Any repair that wants per-frame
/// damage has to get it from somewhere other than the display path.
///
/// What survives is the one thing a static reading cannot promise for a guest
/// this device does not ship: that the payload keeps its declared size. A longer
/// one means this decode has become a truncation, so it is an always-on alarm
/// rather than a sample budget. Under-length is already refused by
/// [`present_surface_id`], which is where it costs a present.
///
/// # The plane-list reading is op6/op7's, and only op6/op7's
///
/// Everything above is about `submitTransaction`, whose two legs emit op6
/// `CmdDisplayTransaction2_DEPRECATED` and op7 `CmdDisplayTransaction3`. **op8
/// `CmdDisplaySwapMapping` does not serialize a transaction at all** — it names
/// a single mapping, as [`display_txn_trailer_slots`] says — so it has no plane
/// list to grow and "a plane list may have appeared" cannot be true of it.
///
/// That distinction is not hypothetical. op8 is the arm64 present path, and on
/// a driven arm64 boot this alarm fires on **every present**, 1 668 times in
/// 212 s, always as `op=0x8 plen=40 trailer=12`. The message it printed was an
/// explanation that structurally could not apply, on a first-class pathway's
/// normal traffic. What is true there is narrower and is what it says now: 28
/// bytes past the words this device knows, contents unnamed.
///
/// So the line carries the undecoded tail's bytes. It is latched per
/// `(opcode, length)` and therefore costs one line per shape per boot, and
/// without them a reader who sees this alarm has to rebuild and reboot before
/// learning anything at all — which is what closing the gap actually needs.
///
/// # What the arm64 guest actually puts there
///
/// Measured, and deliberately not interpreted. On two boots at 1920×1080 the
/// tail is byte-identical and there is exactly **one** `(opcode, length)` shape
/// in the whole boot — `op=0x8 plen=40` — so this is what every present on this
/// pathway carries:
///
/// ```text
/// +0x0c  00 00 00 00
/// +0x10  00 40 10 00
/// +0x14  00 00 00 00
/// +0x18  00 40 00 00
/// +0x1c  00 00 00 00
/// +0x20  01 01 00 00
/// +0x24  00 00 40 48
/// ```
///
/// No field here is named, and none should be until it has been made to move.
/// Read as little-endian `u32` the non-zero words are `0x00104000`, `0x00004000`,
/// `0x00000101` and `0x48400000`, and it is tempting to call the second one the
/// 16 KiB arm64 page size and the first a page-aligned length because
/// `0x104000 / 0x4000` is exactly 65. That is arithmetic agreeing with a guess,
/// which is not a derivation — every one of those words is also consistent with
/// a stride, an extent, a pair of `u16`s, or a `f32` (`0x48400000` is 196608.0).
///
/// **The experiment that would settle it is a display mode change.** Change the
/// guest's resolution, take the alarm's line again — the latch is keyed on
/// length, so a second mode at the same length needs the latch cleared or a
/// second boot — and diff the seven words. Whatever tracks width, height or
/// stride identifies itself immediately. Nothing short of that should turn
/// these into named fields.
/// Report a `CmdInvalidateResources` / `CmdSynchronizeResources` payload that
/// did not decode, naming the check that refused it.
///
/// Both arms used to hand-roll the same line: `… decode_fail w0=… w1=…`, the
/// first two words of the payload and no `reason=`. Two problems with that, and
/// the second is the one that cost. Without a slug the line cannot be ranked —
/// `AGENTS.md`'s own recipe for reading the log is to rank `reason=` on the fail
/// channel, and a line with none is invisible to it. And `w0`/`w1` are the
/// task id and the count, which is the *input* to the check rather than its
/// verdict: a short header and a count the payload cannot carry produced
/// identical-looking lines, so the reader had to re-derive which had happened.
///
/// One helper for both commands, because the decoder's error type is one type.
/// The command name stays a field so the two stay separable in a grep.
///
/// Latched per `(reason, opcode)`: the guest re-sends these on a cadence, and a
/// malformed one is malformed every time. Magnitude belongs to a counter.
fn note_resource_list_decode_fail(
    op: &'static str,
    opcode: u16,
    channel_id: u32,
    error: crate::runtime::decode::fifo::ResourceListDecodeError,
) {
    crate::observe::Emit::decline("map_family", &error)
        .field("op", op)
        .field("opcode", format!("{opcode:#x}"))
        .field("ch", channel_id)
        .fail_once(u64::from(opcode));
}

fn note_display_txn_payload(state: &mut DeviceState, channel_id: u32, packet: &Packet) {
    let plen = packet.payload.len();
    let trailer = display_txn_trailer_len(packet.opcode);
    if plen <= trailer {
        return;
    }
    crate::runtime::drain::note_store_route("display_txn_payload_overlong");
    // Latched per (opcode, length): a guest that grew this command grew it for
    // every frame, and the thousandth line says nothing the first did not.
    if !state
        .display
        .txn_payload_samples
        .insert((packet.opcode, plen))
    {
        return;
    }
    // Bounded, because the length is the guest's. 64 bytes is four times the
    // largest trailer here and is enough to show the shape of whatever follows.
    const TAIL_DUMP_MAX: usize = 64;
    let tail = &packet.payload[trailer..];
    let shown = tail.len().min(TAIL_DUMP_MAX);
    let mut hex = String::with_capacity(shown * 3);
    for b in &tail[..shown] {
        hex.push_str(&format!("{b:02x}"));
    }
    let what = if packet.opcode == CHILD_OP_DISPLAY_SWAP {
        // No transaction is serialized here, so there is no plane list this
        // could be. Naming one would send the next reader looking for a
        // structure that cannot exist on this command.
        "this command names a single mapping and serializes no transaction, so these \
         bytes are not a plane list and nothing in this device names them"
    } else {
        "the command carries more than its declared trailer, so a plane list may have \
         appeared and decoding a single surface id would be dropping planes"
    };
    crate::observe::fail(format!(
        "display_txn_payload_overlong op={:#x} ch={channel_id} plen={plen} trailer={trailer} \
         undecoded={} tail=0x{hex}{} ({what})",
        packet.opcode,
        tail.len(),
        if shown < tail.len() { "..." } else { "" },
    ));
}

/// What the CPU-side capture can say about a present's content.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PresentContentVerdict {
    /// No CPU pixels exist for this present, so nothing can be claimed.
    Unsampled,
    /// Sampled, and every pixel's RGB is zero.
    Black,
    /// Sampled, and something is visible.
    Content,
}

/// Judge a present's captured frame.
///
/// An empty `frame_bgra` is **not** a black frame. When an engine resident
/// carries the present — the window presents it from the engine's own device, so
/// the frame never crosses host memory — `capture_present_frame` deliberately
/// skips the full-frame GPU→CPU readback and drops the buffer, so a plain
/// `max_rgb == 0` test reports black on every such present — 1338
/// `present_black_retain` records against 1312 presents on a live boot. That
/// buries the always-on log under a wolf-cry and hides the genuinely black frame
/// the record exists to catch, which is the opposite of what an always-on
/// failure sink is for. With no pixels there is no evidence either way, so the
/// absence has its own verdict.
pub(crate) fn present_content_verdict(frame_bgra: &[u8], max_rgb: u8) -> PresentContentVerdict {
    if frame_bgra.is_empty() {
        PresentContentVerdict::Unsampled
    } else if max_rgb == 0 {
        PresentContentVerdict::Black
    } else {
        PresentContentVerdict::Content
    }
}

/// One `{stamp_index, stamp_value}` record from a packet's record array.
///
/// The record means **"do not execute this packet until stamp slot `index` has
/// reached `value`"** — a wait, not a signal. See [`note_packet_stamp_records`]
/// for how that reading was established and what the alternative would have been.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StampWait {
    /// Index into the same stamp space [`write_stamp`] writes: one slot per
    /// FIFO, root at 0. Carried raw, exactly as the guest wrote it, so the
    /// masking [`stamp_slot_index`] applies happens at the one place that
    /// resolves a slot rather than twice with two chances to differ.
    pub index: u32,
    pub value: u32,
}

impl StampWait {
    /// Whether stamp slot [`Self::index`] standing at `current` satisfies this
    /// wait.
    ///
    /// The comparison is a **signed wrapping difference**, not `current >=
    /// self.value`. Stamp values are free-running `u32` counters the guest
    /// increments per submission, so a slot that has been live long enough
    /// wraps through zero; a plain `>=` then reads every wait on the far side of
    /// the wrap as unsatisfied and stalls the channel permanently. The signed
    /// difference is correct for any pair less than 2^31 apart, which is the
    /// same window the rest of this protocol's counters assume.
    pub fn satisfied_by(self, current: u32) -> bool {
        current.wrapping_sub(self.value) as i32 >= 0
    }
}

/// Parsed FIFO packet (main + child share framing).
///
/// `stamp_waits` is the record array between header and payload, decoded. The
/// header's own count is not carried beside it: the two would be one fact
/// written twice, and the arm that read the count without the records is the
/// one this field exists to retire. Ask for [`Self::stamp_count`] where the
/// wire count is what a reader wants.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Packet {
    pub opcode: u16,
    pub stamp_waits: Vec<StampWait>,
    pub total_size: u32,
    pub completion_stamp: u32,
    pub payload: Vec<u8>,
    pub next_head: u32,
}

impl Packet {
    /// The header's `stamp_count` as decoded. Never truncates: the count is a
    /// `u16` on the wire and [`decode_packet`] parses exactly that many records.
    pub fn stamp_count(&self) -> u16 {
        self.stamp_waits.len() as u16
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PacketError {
    ShortHeader,
    BadSize,
    Incomplete,
    /// The snapshot was shorter than the packet's own published `total_size`.
    /// Not reachable through [`packet_snapshot_len`]; a reading means this
    /// device under-snapped, so unlike [`Self::BadSize`] it accuses the host.
    ShortSnapshot,
}

impl PacketError {
    /// The registered fault this error reports, or `None` when it is ring
    /// control flow rather than a refusal.
    ///
    /// This is I2's carve-out made mechanical instead of conventional. A partial
    /// packet in the ring is the *normal* state of a producer mid-write: the
    /// drain loop breaks and comes back. Logging it would flood the always-on
    /// sink on every healthy boot, which is why `ShortHeader` and `Incomplete`
    /// answer `None` here — and why a future variant cannot be added without its
    /// author deciding which side of that line it falls on. Both drain loops
    /// go through it, so that decision is made once rather than per ring.
    pub fn fault(self) -> Option<PacketFault> {
        match self {
            Self::ShortHeader | Self::Incomplete => None,
            Self::BadSize => Some(PacketFault::BadSize),
            Self::ShortSnapshot => Some(PacketFault::ShortSnapshot),
        }
    }
}

/// How many bytes of the ring a packet at the consumer pointer has to be
/// snapshotted into before [`decode_packet`] can classify it.
///
/// The whole packet when the header's `total_size` is both sane and already
/// published; otherwise the header alone, which is all `decode_packet` needs in
/// order to say *which* of those two it was — and all that can safely be read,
/// since an unpublished or nonsense `total_size` names bytes the producer has
/// not written.
///
/// Shared by both drain loops rather than spelled at each, because a second copy
/// of this is a second definition of "sane" — and the two copies had already
/// parted: the root ring's carried an extra `else if available >=
/// PACKET_HEADER_LEN` arm that its own caller had made unreachable one line
/// earlier, and the child ring's did not.
fn packet_snapshot_len(header: &[u8], available: u32, ring_capacity: u32) -> u32 {
    let total_size = ld32(&header[PACKET_TOTAL_SIZE..]);
    if total_size >= PACKET_HEADER_LEN && total_size <= ring_capacity && available >= total_size {
        total_size
    } else {
        PACKET_HEADER_LEN
    }
}

/// The per-packet stamp records, reported once per shape.
///
/// A packet header declares `stamp_count`, and that many 8-byte records sit
/// between the header and the payload. This device used to skip straight over
/// them to reach the payload; they are now decoded into [`Packet::stamp_waits`]
/// and evaluated by [`note_packet_stamp_waits`].
///
/// A skip whose consequence nobody had measured is the shape of an unmeasured
/// loss rather than a known-benign skip, and the two are not distinguishable by
/// reading the code — which is the whole argument for an instrument. If the
/// records carry a wait-for-stamp dependency, a packet whose records this device
/// ignores runs before the work it was told to wait for, and the corruption that
/// follows has no line anywhere to explain it. If they are completion stamps to
/// retire, ignoring them strands the guest. If they are neither, the skip is
/// correct and this line is what says so.
///
/// So: **report, do not refuse.** Refusing every packet that carries one would
/// decline live guest traffic on a guess, and this file's own rule is that a
/// refusal needs its zero measured first — the same licence
/// `note_pipeline_tlv_fields` took and `note_color_entry_fields` took before it.
///
/// Latched per `(opcode, stamp_count)`, so a command that always carries the
/// same number of records says so once and a *new* count reports. The first
/// record's two words ride along because they are what would identify the
/// format; a reader with a non-zero count and no words has to go back to the
/// ring for them.
///
/// Off-channel and behind the `stamp_count > 0` guard, so a boot where no
/// packet carries a record pays one comparison per packet and emits nothing.
/// **That silence is not a measurement** — it reads the same as this function
/// never being reached — so the count of packets *offered* to it is what a
/// reader needs beside it, which is `note_store_route`'s `packet_stamps_none`.
///
/// # The first reading, and it is not a healthy zero
///
/// Driven x86/PCI/Vulkan boot, Safari composited on a Ventura desktop:
///
/// ```text
/// packet_stamps_none     31168
/// packet_stamps_present   3426      ~10 % of all decoded packets
///
/// 0x04 CURSOR_GLYPH           count=1  w0=0x2 w1=0x4
/// 0x06 PRESENT_X86            count=1  w0=0x1 w1=0x7
/// 0x06 PRESENT_X86            count=2  w0=0x1 w1=0x5
/// 0x20 DELETE_TASK            count=1  w0=0x2 w1=0x5da
/// 0x22 UNMAP_MEMORY           count=1  w0=0x4 w1=0x1
/// 0x25 DELETE_OBJECT          count=1  w0=0x1 w1=0x69
/// 0x35 SYNCHRONIZE_RESOURCES  count=1  w0=0x1 w1=0xe7
/// 0x37 EXEC_INDIRECT2         count=1  w0=0x2 w1=0xe
/// 0x37 EXEC_INDIRECT2         count=2  w0=0x4 w1=0x4
/// 0x3b GET_COMPUTE_INFO       count=1  w0=0x1 w1=0x19
/// ```
///
/// So this is live traffic, not a dormant field: about one packet in ten
/// carries a record.
///
/// **The ten commands that carry one are not a random ten.** They are work
/// submission (`CHILD_OP_EXEC_INDIRECT2`), explicit synchronisation
/// (`CHILD_OP_SYNCHRONIZE_RESOURCES`), presentation, and then the teardown
/// set — `CHILD_OP_DELETE_TASK`, `CHILD_OP_DELETE_RESOURCE`,
/// `CHILD_OP_UNMAP_MEMORY`. That is the list of commands whose ordering
/// *against GPU work* is the thing that matters, and three of them free or
/// unmap memory the GPU may still be reading. A record on an unmap is either
/// "do not unmap until the GPU has reached this point" or "tell the guest when
/// the unmap is done", and under the first, ignoring it is a use-after-free of
/// guest pages.
///
/// That is the strongest evidence here, and it still does not choose between
/// the two readings — which is the point. It says the records are about
/// ordering without saying which direction they order in.
///
/// # What the records are: a wait, established by RE
///
/// Each record is `{u32 stamp_index, u32 stamp_value}` and means **"do not
/// execute this command until stamp slot `stamp_index` has reached
/// `stamp_value`"**. Apple's host acts on them, before dispatch. The alternative
/// reading — an extra completion stamp to *signal* — is positively excluded
/// rather than merely disfavoured, because the signal has a different home and a
/// different producer: the guest's command allocator has one method that appends
/// these 8 bytes and bumps the header's count, and a separate one that writes a
/// stamp value into the header's own `completion_stamp` field and appends
/// nothing. Wait goes in the record array; signal goes in the header. They
/// cannot be the same thing.
///
/// Details that a first implementation gets wrong without being told:
///
/// * The comparison is a **signed wrapping** difference — satisfied iff
///   `(current - stamp_value) as i32 >= 0`. A plain `>=` is wrong once a slot
///   wraps.
/// * `stamp_index` is an index into the same stamp space [`write_stamp`] writes,
///   one slot per FIFO, root at 0. It is range-checked against the device's FIFO
///   count, so it is an index and not a mask — the observed 1, 2, 4 are just the
///   child FIFOs this workload created.
/// * The wait is **per-FIFO**, not device-wide. The awaited stamp is published
///   by *another* channel's drain, so a single-threaded drain that blocks here
///   deadlocks itself. The shape that works is to defer the channel and move on.
/// * The guest pre-filters dependencies it can already see satisfied, so the
///   ~10 % measured above is genuinely-unsatisfied-at-submit-time rather than
///   noise, and a record naming the submitting channel's own slot should never
///   appear.
/// * The count is bounded at 64 by the host, which refuses above it.
///
/// **So skipping them is an ordering bug, and the opcode list above is exactly
/// its blast radius.** The stamps do eventually get written, so this is not a
/// hang; it is work running ahead of the work it was ordered behind.
/// `CHILD_OP_DELETE_TASK`, `CHILD_OP_DELETE_RESOURCE` and
/// `CHILD_OP_UNMAP_MEMORY` tear down memory the GPU may still be sourcing, which
/// makes ignoring their records a use-after-free of guest pages — the same class
/// as the completion-stamp ordering [`write_stamp`] already carries a flush for,
/// and which its doc records having caused heap corruption once.
///
/// The records are now decoded into [`Packet::stamp_waits`] and evaluated by
/// [`note_packet_stamp_waits`] at both drain sites. This line stays because the
/// *shape* census and the *verdict* census answer different questions: this one
/// says which commands carry records and how many, once per shape, and is what
/// a reader compares a new workload against.
///
/// # How many of those orderings this device actually violates
///
/// Second driven x86/PCI/Vulkan boot, same Safari-drag workload:
///
/// ```text
/// packet_stamps_none      31057
/// packet_stamps_present    2863
/// packet_stamp_wait_met    1658
/// packet_stamp_wait_unmet  1278      44 % of every wait the guest sent
/// ```
///
/// So this is not a rare race that the drain order happens to get right. Nearly
/// half of all waits are unsatisfied at the moment this device runs the packet
/// anyway, on the four shapes the fail log named:
///
/// ```text
/// 0x37 EXEC_INDIRECT2  ch1   index=2  awaited=0x12   current=0xe    behind=4
/// 0x37 EXEC_INDIRECT2  ch1   index=4  awaited=0x3    current=0x2    behind=1
/// 0x22 UNMAP_MEMORY    ch2   index=4  awaited=0x1    current=0x0    behind=1
/// 0x20 DELETE_TASK     root  index=2  awaited=0x1ce  current=0x1cd  behind=1
/// ```
///
/// Two readings matter. First, `behind` is *small* — one submission, four at the
/// worst. The producing work is already queued or about to be, which is what
/// says a deferral would clear in the next tranche rather than parking a channel
/// indefinitely. Second, the root `DELETE_TASK` waiting one submission behind a
/// child FIFO's slot is the use-after-free named above, caught in the act: the
/// task is torn down while the work that sources it has not been drained.
///
/// # And what the hold did to it
///
/// Third driven boot, same workload, with the packet held on an unmet wait:
///
/// ```text
/// packet_stamp_wait_met       2901      was 1652
/// packet_stamp_wait_unmet      251      was 1500
/// packet_stamp_wait_held       239
/// stamp_hold_retry              74
/// stamp_hold_handed_back        32
/// ```
///
/// Not one packet now runs with a wait behind it — the 251 are held rather than
/// executed, and the same packets return as `met` once the slot arrives, which
/// is where the rise on the met side comes from. `stamp_hold_retry` is
/// [`retry_stamp_held_timelines`] clearing a hold that the channel-id walk order
/// created; `stamp_hold_handed_back` is the tail that genuinely had to wait for
/// the guest to submit the producing work.
///
/// The `behind` reading predicted that, and the throughput reading confirms the
/// cost: the drag probe's worst second is 0.9 Hz both before and after. A hold
/// that lasts one tranche does not show up as a frame.
fn note_packet_stamp_records(opcode: u16, waits: &[StampWait]) {
    if waits.is_empty() {
        note_store_route("packet_stamps_none");
        return;
    }
    note_store_route("packet_stamps_present");
    if !crate::observe::first_sight(
        "packet_stamp_records",
        (u64::from(opcode) << 16) | waits.len() as u64,
    ) {
        return;
    }
    let count = waits.len();
    let first = waits[0];
    crate::observe::off(format!(
        "packet_stamp_records opcode={opcode:#x} count={count} w0={:#x} w1={:#x} \
         (wait: do not execute until stamp[w0] has reached w1)",
        first.index, first.value
    ));
}

/// What a packet's stamp waits say the drain should do with it.
///
/// The three answers are not degrees of the same thing. [`Self::Ready`] and
/// [`Self::Hold`] are the wait working; [`Self::Unevaluable`] is the device
/// unable to decide, and collapsing it into `Hold` is how a report becomes a
/// hang — see the variant's own note.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StampVerdict {
    /// Every wait is satisfied, or there were none. Run the packet.
    Ready,
    /// At least one wait is genuinely behind. Hold the packet: a stamp this
    /// device will publish is what clears it.
    Hold,
    /// A wait this device cannot decide, and **no future event changes that**.
    ///
    /// Holding on one would park the timeline forever, which is strictly worse
    /// than the ordering slip it was meant to prevent: an ordering slip loses
    /// one packet's ordering, a parked root FIFO loses the guest. So this runs
    /// the packet, loudly. Every case here is a refusal with a named reason, and
    /// none of them fired on a driven boot.
    Unevaluable,
}

impl StampVerdict {
    /// Fold one wait's answer into the packet's, most restrictive winning.
    ///
    /// `Unevaluable` outranks `Hold`, which outranks `Ready`. That order is the
    /// whole point: a packet with one wait genuinely behind and one that can
    /// never be decided must **run**, because holding for the first would still
    /// park the timeline forever on the second.
    fn and(self, other: Self) -> Self {
        match (self, other) {
            (Self::Unevaluable, _) | (_, Self::Unevaluable) => Self::Unevaluable,
            (Self::Hold, _) | (_, Self::Hold) => Self::Hold,
            _ => Self::Ready,
        }
    }
}

/// Evaluate a packet's stamp waits against the slots this device has published.
///
/// The current value is read back out of the stamp page rather than cached
/// beside [`write_stamp`]. The page is where the guest reads it, so it is the
/// only copy whose staleness cannot be this device's own bug, and a cache keyed
/// by slot would be one more bounded structure to keep honest for four bytes.
///
/// Every wait in the packet is evaluated even once one is known unsatisfied.
/// The verdict would not change, but the census would: a packet carrying two
/// waits where the second is out of range must say so, and stopping at the first
/// hold hides it until the first one clears.
fn note_packet_stamp_waits<H: HostMemory + HostOps>(
    state: &DeviceState,
    host: &H,
    channel: Option<u32>,
    packet: &Packet,
) -> StampVerdict {
    if packet.stamp_waits.is_empty() {
        return StampVerdict::Ready;
    }
    let mut verdict = StampVerdict::Ready;
    for wait in &packet.stamp_waits {
        let index = stamp_slot_index(wait.index);
        let unresolvable = |reason: &'static str, detail: String| {
            note_store_route("packet_stamp_wait_unresolvable");
            if crate::observe::first_sight("packet_stamp_wait_unresolvable", u64::from(index)) {
                crate::observe::fail(format!(
                    "packet_stamp_wait_unresolvable reason={reason} opcode={:#x} \
                     index={index} {detail} (the wait cannot be decided and no \
                     drain can change that, so the packet runs unordered rather \
                     than parking this timeline forever)",
                    packet.opcode,
                ));
            }
        };
        // An index past the stamp page names a FIFO this device does not have,
        // so no drain could ever write the slot the guest is watching.
        let Some(off) = stamp_slot_offset(index, state.page_size()) else {
            unresolvable(
                "stamp_slot_out_of_range",
                format!("slots={}", stamp_slot_count(state.page_size())),
            );
            verdict = verdict.and(StampVerdict::Unevaluable);
            continue;
        };
        // No stamp page means no slot to read and no slot `write_stamp` would
        // write either — it returns early on the same condition.
        if state.gfx.fifo_base_page == 0 {
            unresolvable("no_stamp_page", String::from("fifo_base_page=0"));
            verdict = verdict.and(StampVerdict::Unevaluable);
            continue;
        }
        let gpa = state.pfn_gpa(state.gfx.fifo_base_page) + off;
        let Ok(current) = crate::runtime::host::read_u32(host, gpa) else {
            unresolvable("stamp_slot_unreadable", format!("gpa={gpa:#x}"));
            verdict = verdict.and(StampVerdict::Unevaluable);
            continue;
        };
        if wait.satisfied_by(current) {
            note_store_route("packet_stamp_wait_met");
            continue;
        }
        note_store_route("packet_stamp_wait_unmet");
        verdict = verdict.and(StampVerdict::Hold);
        if crate::observe::first_sight(
            "packet_stamp_wait_unmet",
            (u64::from(packet.opcode) << 32) | u64::from(index),
        ) {
            crate::observe::off(format!(
                "packet_stamp_wait_unmet opcode={:#x} {} index={index} awaited={:#x} \
                 current={current:#x} behind={} (the packet is held until the slot \
                 reaches it)",
                packet.opcode,
                packet_site(channel),
                wait.value,
                wait.value.wrapping_sub(current),
            ));
        }
    }
    verdict
}

/// Decode one packet out of a ring snapshot taken at [`packet_snapshot_len`].
///
/// `ring_capacity` is what bounds a sane `total_size`, and it is a parameter
/// rather than `bytes.len()` for the reason [`PacketError::Incomplete`] exists.
/// The snapshot is *deliberately* short of `total_size` whenever the producer
/// has not published the whole packet, so measuring the declared size against
/// the snapshot conflates "the guest declared a size this ring could never hold"
/// with "we only snapped the header because the packet is still being written".
/// Those are opposite answers — one is a fault, the other is control flow that
/// must stay quiet — and reading the second as the first put a
/// `packet_bad_size` line on the always-on channel for a healthy producer
/// mid-write, while making `Incomplete` unreachable.
fn decode_packet(
    bytes: &[u8],
    head: u32,
    available: u32,
    ring_capacity: u32,
) -> Result<Packet, PacketError> {
    if available < PACKET_HEADER_LEN {
        return Err(PacketError::ShortHeader);
    }
    if bytes.len() < PACKET_HEADER_LEN as usize {
        return Err(PacketError::ShortHeader);
    }
    let opcode = ld16(&bytes[PACKET_OPCODE..]);
    let stamp_count = ld16(&bytes[PACKET_STAMP_COUNT..]);
    let total_size = ld32(&bytes[PACKET_TOTAL_SIZE..]);
    let completion_stamp = ld32(&bytes[PACKET_COMPLETION_STAMP..]);

    // A size the ring itself could never hold is the guest's error.
    if total_size < PACKET_HEADER_LEN || total_size > ring_capacity {
        return Err(PacketError::BadSize);
    }
    // A size the producer has not finished publishing is not.
    if available < total_size {
        return Err(PacketError::Incomplete);
    }
    // Past both, `packet_snapshot_len` took its first arm and the snapshot is
    // the whole packet. A short one here is this device mis-snapping, not
    // anything the guest did, so it is its own fault rather than a `BadSize`
    // charged to the guest — and a healthy boot never reaches it.
    if (bytes.len() as u32) < total_size {
        return Err(PacketError::ShortSnapshot);
    }
    let stamps_bytes = stamp_count as u32 * PACKET_STAMP_LEN;
    let min_payload_off = PACKET_HEADER_LEN + stamps_bytes;
    if total_size < min_payload_off {
        return Err(PacketError::BadSize);
    }
    // Past that check every record is inside `total_size`, and `total_size` is
    // inside the snapshot, so the walk reads only bytes the producer published.
    // Walking the span rather than indexing by record number keeps the stride
    // out of the arithmetic: `chunks_exact` yields exactly `stamp_count`
    // records because the span above is exactly that many strides long.
    let stamp_waits: Vec<StampWait> = bytes[PACKET_HEADER_LEN as usize..min_payload_off as usize]
        .chunks_exact(PACKET_STAMP_LEN as usize)
        .map(|rec| StampWait {
            index: ld32(rec),
            value: ld32(&rec[4..]),
        })
        .collect();
    note_packet_stamp_records(opcode, &stamp_waits);
    let payload = bytes[min_payload_off as usize..total_size as usize].to_vec();
    Ok(Packet {
        opcode,
        stamp_waits,
        total_size,
        completion_stamp,
        payload,
        next_head: head.wrapping_add(total_size),
    })
}

fn read_ring_bytes<M: HostMemory>(
    mem: &M,
    base_gpa: u64,
    ring_size: u32,
    absolute: u32,
    len: u32,
) -> Result<Vec<u8>, MemError> {
    let mut out = vec![0u8; len as usize];
    if ring_size == 0 || len == 0 {
        return Ok(out);
    }
    let mut copied = 0u32;
    while copied < len {
        let off = absolute.wrapping_add(copied) % ring_size;
        let chunk = (ring_size - off).min(len - copied);
        mem.read_gpa(
            base_gpa + off as u64,
            &mut out[copied as usize..(copied + chunk) as usize],
        )?;
        copied += chunk;
    }
    Ok(out)
}

/// Report a completion stamp that moves its slot **backwards**.
///
/// [`write_stamp`] writes the packet header's `completion_stamp` field into the
/// slot without looking at what is already there, so a packet carrying zero — or
/// any value behind the slot — rewinds the guest's fence. That is not a cosmetic
/// slip: [`StampWait`] is decided against this same slot, so a rewind unsatisfies
/// every wait between the old value and the new one, and the guest's own polls
/// see work it had been told was finished become unfinished.
///
/// Which makes it the first thing to rule out before reading an unmet-wait count
/// as an ordering problem: a device rewinding its own fences produces unmet waits
/// that have nothing to do with drain order. Same-value writes are the expected
/// case and stay quiet — a packet that does not signal repeats the slot's value
/// rather than clearing it, and repeating is idempotent.
fn note_stamp_direction<H: HostMemory + HostOps>(host: &H, gpa: u64, index: u32, value: u32) {
    let Ok(current) = crate::runtime::host::read_u32(host, gpa) else {
        return;
    };
    if value == current {
        note_store_route("stamp_write_repeat");
        return;
    }
    // The same signed wrapping difference `StampWait::satisfied_by` uses, so the
    // reading here and the wait's verdict cannot disagree about which way round
    // two values are.
    if (value.wrapping_sub(current) as i32) > 0 {
        note_store_route("stamp_write_forward");
        return;
    }
    note_store_route("stamp_write_backward");
    if crate::observe::first_sight("stamp_write_backward", u64::from(index)) {
        crate::observe::fail(format!(
            "stamp_write_backward index={index} was={current:#x} now={value:#x} \
             back={} (the guest's fence for this slot moves backwards, which \
             unsatisfies every wait between the two values)",
            current.wrapping_sub(value)
        ));
    }
}

/// Record this stamp's word into the GPU queue behind the writebacks it
/// follows, so the drain worker never blocks on them.
///
/// `false` means the caller still owes the stamp and must settle it the blocking
/// way. Four things answer `false`, and only the last is a fault: the operator
/// narrowed the rail with `REIMS_VGPU_GPU_STAMP=off`; nothing was owed, so there
/// is nothing to order behind and the plain store is both correct and cheaper;
/// the stamp page would not resolve to imported guest RAM; or the engine
/// declined, which it reports itself.
///
/// The word is four bytes inside one page, so the contiguity rule
/// `reference_for_pages` enforces is satisfied by construction — but it is asked
/// rather than assumed, because a stamp page outside an imported RAMBlock is
/// exactly the case that must fall back rather than be written blind.
#[cfg(feature = "backend-vulkan")]
fn stamp_word_ordered_on_gpu<H: HostMemory + HostOps>(
    state: &DeviceState,
    host: &mut H,
    index: u32,
    value: u32,
) -> bool {
    if crate::env::switch(crate::env::GPU_STAMP) == crate::env::Switch::Off {
        return false;
    }
    // Nothing owed means nothing to order behind. Taking the GPU rail here would
    // add a submission and a thread hop to buy an ordering that already holds,
    // and most stamps in a boot are this case.
    if !crate::backend::vulkan::engine::guest_writes_outstanding() {
        return false;
    }
    let page_size = state.page_size();
    let Some(off) = stamp_slot_offset(index, page_size) else {
        return false;
    };
    let gpa = state.pfn_gpa(state.gfx.fifo_base_page) + off;
    let page = gpa & !(page_size - 1);
    let in_page = gpa - page;
    let Ok(guest_ref) =
        crate::runtime::guest_ram_map::reference_for_pages(host, &[page], page_size, in_page, 4)
    else {
        return false;
    };
    // The direction check the CPU rail gets from `note_stamp_direction`. Taken
    // before the submit, because after it the word is the GPU's and reading it
    // back says nothing about what this device promised.
    note_stamp_direction(host, gpa, index, value);
    crate::backend::vulkan::engine::write_stamp_after_guest_writes(&guest_ref, index, value).is_ok()
}

/// Write stamp value to FIFO base page slot and set status bit.
pub fn write_stamp<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    stamp_index: u32,
    stamp_value: u32,
) {
    let index = stamp_slot_index(stamp_index);
    if state.gfx.fifo_base_page == 0 {
        return;
    }
    // Before the guest is told anything finished, everything this device still
    // owes guest RAM has to be in guest RAM. After this write the guest may free
    // the render targets and its allocator may hand those pages to anything, and
    // no later check can tell that memory apart from the target it used to be —
    // which is why the page-set guard passed on 810 of 810 landings and the heap
    // corruption continued.
    //
    // **Nothing may settle those writes before the GPU rail below is offered
    // one.** That rail's first question is whether anything is still
    // outstanding, because ordering behind nothing is a submission and a thread
    // hop for an ordering that already holds. A settle here answers that
    // question "no" every time, by blocking — which is the whole cost the rail
    // exists to remove — and leaves `engine_delta` reporting `gpu_stamps=0`
    // beside a `readback_split` `fence` that tracks the flush count exactly.
    // Both quiesces below are reached only when the rail declines, and there
    // they are the settle this comment used to describe.
    // The stamp word ordered behind the copies by the GPU rather than by this
    // thread blocking. Tried before either quiesce because when it takes,
    // neither is owed: its leading barrier names every command submitted before
    // it, which covers the writebacks and the guest reads alike.
    //
    // Nothing about the *interrupt* is deferred by this — the completion thread
    // raises it the moment the submission lands. See
    // `backend::vulkan::engine::stamp_completion` for the measurement that says
    // it cannot be deferred to anything slower.
    #[cfg(feature = "backend-vulkan")]
    if stamp_word_ordered_on_gpu(state, host, index, stamp_value) {
        // Advanced at submit, not at completion. From here the guest may see the
        // word at any moment, so a window still armed has already outlived this
        // fence — which is what `armed_stamp_seq` is compared against, and
        // dating it from the completion would call that window punctual.
        state.completion_stamp_seq = state.completion_stamp_seq.wrapping_add(1);
        return;
    }
    // The flush above submits its copies without waiting for them, so "owed" is
    // not yet "landed" until this returns. One settle for every window the pass
    // issued, taken after all of them are on the queue, rather than one blocking
    // fence per window taken between them.
    crate::runtime::render_writeback::settle_guest_writes(
        crate::runtime::render_writeback::SettleSite::CompletionStamp,
    );
    // And the other half of that sentence: everything this device is still
    // *reading* out of guest RAM has to be done reading. A draw that binds guest
    // pages through the imported RAMBlock reads them when its command buffer
    // executes, and this stamp is what tells the guest those pages are free to
    // repaint. The owed-writes flush above cannot stand in for it — it settles
    // residents into guest memory and knows nothing about what a submission
    // still sources.
    #[cfg(feature = "backend-vulkan")]
    crate::backend::vulkan::engine::quiesce_guest_reads();
    let Some(off) = stamp_slot_offset(index, state.page_size()) else {
        return;
    };
    let gpa = state.pfn_gpa(state.gfx.fifo_base_page) + off;
    let page_size = state.page_size() as usize;
    note_stamp_direction(host, gpa, index, stamp_value);
    if gpa_map::write_u32(host, gpa, stamp_value, page_size).is_ok() {
        // The guest's fence has moved. Everything it allocated for the work this
        // stamp completes may be freed from here on, which is why a Store's
        // guest-page write has to have landed before the stamp rather than
        // after — see `runtime::render_writeback`.
        state.completion_stamp_seq = state.completion_stamp_seq.wrapping_add(1);
        state
            .gfx
            .interrupt_status_gpu
            .fetch_or(1u32 << (index & 0x1f), std::sync::atomic::Ordering::AcqRel);
        host.enqueue(HostAction::irq_gfx());
    }
}

/// What the GPU behind this host can execute, for the device-info keys that
/// describe the GPU rather than the protocol.
///
/// The Metal backend serves an Apple GPU to an Apple guest, so the table's own
/// values already describe the executing device and there is nothing to reduce.
/// The Vulkan backend runs on anything from a discrete part to an iGPU at the
/// Vulkan floor, which is exactly the case a fixed table gets wrong.
fn device_info_limits() -> crate::model::DeviceInfoLimits {
    #[cfg(not(feature = "backend-vulkan"))]
    {
        crate::model::DeviceInfoLimits {
            max_sample_count: u32::MAX,
            d24_stencil8: true,
            max_threads_per_threadgroup: [u32::MAX; 3],
            max_threadgroup_memory_bytes: u32::MAX,
            native_fp16: true,
        }
    }
    #[cfg(feature = "backend-vulkan")]
    {
        crate::backend::vulkan::engine::device_info_limits()
    }
}

/// Answer `CmdGetDeviceInfo` into the guest's reply page.
///
/// Two guest words bound this reply and they bound different things.
/// `key_table_len` is how many arms the guest's key walker has, so the reply may
/// name every key strictly below it — see [`DEVICE_INFO_TAHOE_KEY_TABLE_LEN`]
/// for why that is a table length rather than a highest key. `count` is how many
/// 8-byte pairs its buffer holds. A key at or above the table length costs a
/// pair slot for a value the guest discards on arrival; a pair past `count`
/// would be written past the buffer.
///
/// The guest asks **once**, when its accelerator starts, and never again: it
/// frees the reply buffer right after the single parse, and every later reader
/// is served out of the struct that parse filled. So whatever this reply omits
/// is omitted for the life of the boot, which is why the tail it could not carry
/// is reported rather than dropped quietly.
///
/// Omission is the *safe* direction and not a neutral one. The guest zeroes that
/// struct at allocation and its accessors never consult the per-key "was this
/// defined" flags the walker sets, so an unanswered key reads back as 0 — which
/// for `maxThreadsPerThreadgroup` or `maxThreadgroupMemoryLength` is a device
/// that can run nothing. Answering too much is a lie the guest acts on;
/// answering too little is a zero it also acts on.
fn reply_device_info<H: HostMemory + HostOps>(
    host: &mut H,
    key_table_len: u32,
    count: u32,
    reply_pfn: u32,
    page_shift: u32,
    version: u32,
) -> Result<(), MemError> {
    if reply_pfn == 0 {
        return Ok(());
    }
    let page_size = 1usize << page_shift;
    let gpa = pfn_to_gpa(reply_pfn, page_shift);
    // Contract: guest reply buffer is one page. Cap pairs so we never write past it.
    let max_pairs = (page_size / DEVICE_INFO_REPLY_PAIR_LEN) as u32;
    if max_pairs == 0 {
        crate::observe::fail(format!(
            "device_info fail reason=page_too_small page={page_size:#x}"
        ));
        return Err(MemError::BadArgs);
    }
    let limits = device_info_limits();
    let served = crate::model::device_info_caps(&limits, version);
    // Withhold every key the guest just said it does not parse. This is not a
    // reduction of what the device can do — a key the guest discards on arrival
    // buys nothing, and the slot it occupies is one fewer for a key the guest
    // *does* have an arm for, on a reply that is only ever asked for once.
    let caps: Vec<(u32, u32)> = served
        .iter()
        .copied()
        .filter(|&(key, _)| key < key_table_len)
        .collect();
    let above_ceiling = served.len() - caps.len();
    // Printed on every reply, not only when something changed. A host that
    // already meets the table reduces nothing, and then silence would be
    // indistinguishable from the derivation never having run — which is exactly
    // the failure mode this line exists to rule out on a rig whose GPU exceeds
    // every entry. `derived` names every key the guest was told something other
    // than the table's value, whether the cause was the host GPU or the
    // negotiated version; `version` is printed so key 12's answer can be
    // checked against the rung without a second log line.
    //
    // Zipped against the *unfiltered* answers: the parse ceiling decides which
    // keys are sent, not what any of them says, so folding it in here would make
    // a withheld key read as a derived one.
    let derived: Vec<String> = served
        .iter()
        .zip(DEVICE_INFO_CAPS)
        .filter(|((_, answer), (_, table))| answer != table)
        .map(|((key, answer), (_, table))| format!("key{key}={answer}(was {table})"))
        .collect();
    crate::observe::off(format!(
        "device_info version={} key_table_len={} above_ceiling={} dual_plane={} host_samples={} host_d24s8={} host_threads={}x{}x{} host_tg_mem={} host_fp16={} derived=[{}]",
        version,
        key_table_len,
        above_ceiling,
        u8::from(crate::model::protocol_dual_plane_textures(version)),
        limits.max_sample_count,
        u8::from(limits.d24_stencil8),
        limits.max_threads_per_threadgroup[0],
        limits.max_threads_per_threadgroup[1],
        limits.max_threads_per_threadgroup[2],
        limits.max_threadgroup_memory_bytes,
        u8::from(limits.native_fp16),
        derived.join(" ")
    ));
    let n = (caps.len() as u32).min(count).min(max_pairs);
    // When guest asks for more than one page of pairs, still write at most a
    // page of caps + optional sentinel only if room remains.
    let write_sentinel = n < count && n.saturating_add(1) <= max_pairs;
    if count > max_pairs {
        crate::observe::fail(format!(
            "device_info cap reason=reply_page count={count} max_pairs={max_pairs} page={page_size:#x}"
        ));
    }
    // The tail this reply could not carry, named key by key.
    //
    // Every key still in `caps` is one the guest's own `key_table_len` says it parses,
    // so anything dropped here is a capability that guest spends the rest of the
    // boot without — it asks once and there is no larger re-ask. A healthy boot
    // never emits this: the guest's buffer is a whole page, 512 pairs against a
    // table of a few dozen. A firing *is* the bug, which is the only reason a
    // line that has never been seen is worth carrying.
    if (n as usize) < caps.len() {
        let dropped: Vec<u32> = caps[n as usize..].iter().map(|&(key, _)| key).collect();
        crate::observe::fail(format!(
            "device_info truncated reason=reply_pairs_exhausted key_table_len={key_table_len} count={count} max_pairs={max_pairs} wrote={n} have={} dropped={dropped:?}",
            caps.len()
        ));
    }
    for i in 0..n {
        let (key, value) = caps[i as usize];
        let mut pair = [0u8; DEVICE_INFO_REPLY_PAIR_LEN];
        st32(&mut pair[0..4], key);
        st32(&mut pair[4..8], value);
        gpa_map::write_bytes(
            host,
            gpa + (i as u64) * DEVICE_INFO_REPLY_PAIR_LEN as u64,
            &pair,
            page_size,
        )?;
    }
    if write_sentinel {
        let sentinel = [0u8; DEVICE_INFO_REPLY_PAIR_LEN];
        gpa_map::write_bytes(
            host,
            gpa + (n as u64) * DEVICE_INFO_REPLY_PAIR_LEN as u64,
            &sentinel,
            page_size,
        )?;
    }
    Ok(())
}

/// Wire keys of the `CmdGetComputeInfo` reply the guest reads back.
///
/// The guest's walker has four arms and a terminator, and it sends 5 as its
/// [`DEVICE_INFO_TAHOE_KEY_TABLE_LEN`]-shaped table length. Each arm stores into
/// one field of a five-word struct its `MTLComputePipelineState` answers from —
/// so, as with the device-info table, a value here is an instruction to the
/// guest rather than a description of this device.
///
/// **There are four keys and this device answers three.** The one it does not is
/// key 2, and it is named below rather than left as a hole, because "the guest
/// asked about five keys and we answered three" invites the wrong repair twice:
/// there is no key 5 to add, and key 2 must not be answered the obvious way.
const COMPUTE_INFO_KEY_MAX_TOTAL_THREADS: u32 = 1;

/// Key 2 — `SupportsIndirectCommandBuffers`, and this device deliberately
/// answers nothing.
///
/// The guest's walker stores it into the second word of the struct, and **no
/// accessor in either Metal plugin ever reads that word**. It cannot: the same
/// plugin's `newIndirectCommandBufferWithDescriptor:maxCount:options:` returns
/// nil unconditionally, and its serializer's `executeCommandsInBuffer:` reports
/// "Not supported" — so the guest has no path to an `MTLIndirectCommandBuffer`
/// whatever this key says. (That is not the same thing as this device decoding
/// no ICB records; `runtime::icb` decodes plenty, and they arrive by another
/// route.)
///
/// Omitting it and answering 0 are therefore indistinguishable — the guest
/// zeroes the struct and never consults the walker's per-key "defined" flags —
/// and omitting it costs one fewer pair. Answering **1** is the one wrong move:
/// it would be a promise about a capability the guest cannot exercise and this
/// device has never been asked for through this door.
///
/// Named and unused on purpose. A future guest that grows a reader for the word
/// is what would make this an answer worth sending, and then the value is a
/// question about `runtime::icb`, not about this table.
#[allow(dead_code)] // named so a later session does not read the gap as an omission.
const COMPUTE_INFO_KEY_SUPPORTS_ICB: u32 = 2;

const COMPUTE_INFO_KEY_THREAD_EXECUTION_WIDTH: u32 = 3;
const COMPUTE_INFO_KEY_STATIC_THREADGROUP_MEMORY: u32 = 4;

/// What this host answers for a compute pipeline's threadgroup limits.
///
/// The guest sizes its dispatches from these, so an over-promise is not
/// cosmetic: claim a `maxTotalThreadsPerThreadgroup` the host cannot run and
/// the threadgroup the guest builds from it is one the device rejects. They
/// used to be the fixed triple `(1, 1024), (3, 32), (4, 0)`, whose own comment
/// called itself conservative and deferred the real values to "once
/// metal2vulkan encode lands". That has landed, and both are now read from the
/// host device rather than fixed.
///
/// # Two of the three are pipeline properties, and this answers a device limit
///
/// The request names a `pipeline_ref` and this function ignores it. That is
/// exactly right for `threadExecutionWidth`, which is the SIMD width and is a
/// property of the part; it is **not** right for the other two, and only one of
/// them said so.
///
/// `staticThreadgroupMemoryLength` is the threadgroup memory the kernel
/// declares, so no device limit answers it and it stays 0 until pipeline
/// reflection carries it.
///
/// `maxTotalThreadsPerThreadgroup` is *also* per-pipeline in Metal — a
/// register-heavy kernel reports less than the device maximum, and the canonical
/// app pattern divides it by `threadExecutionWidth` to pick a threadgroup shape.
/// So on the Metal arm this over-promises: the guest sizes a dispatch from the
/// device number and the PSO this device builds from its kernel may refuse it.
/// On the Vulkan arm there is nothing better to say — core Vulkan has no
/// per-pipeline invocation limit, so `maxComputeWorkGroupInvocations` *is* the
/// answer.
///
/// Not fixed, and deliberately: closing it means building the PSO on the query
/// path to ask its own limit, which is Metal-arm-only work whose harm no boot
/// this checkout can take would measure. The proxy that would rank it is a
/// dispatch refused for threadgroup size on a driven arm64 boot.
///
/// A 0 there is not a silent omission but it is not free either. The guest's own
/// default for an unanswered key is also 0, so this reply and no reply are
/// indistinguishable to it, and its plugin does no arithmetic with the value —
/// `setThreadgroupMemoryLength:atIndex:` validates against nothing before
/// serializing. Whether Metal.framework itself checks a declared static length
/// against the device budget this table's key 6 supplies is **not established**,
/// so "answering 0 lets an app over-allocate dynamic threadgroup memory" is an
/// open question rather than a known cost. What is established is that the 0 is
/// wrong for any kernel that declares threadgroup memory, and that only
/// reflection can make it right.
fn compute_info_caps() -> [(u32, u32); 3] {
    // Apple GPUs report 1024 and 32 across every family the arm64 pathway
    // targets, and the Metal backend serves an Apple GPU to an Apple guest.
    #[cfg(not(feature = "backend-vulkan"))]
    let (max_total_threads, thread_execution_width) = (1024, 32);
    #[cfg(feature = "backend-vulkan")]
    let (max_total_threads, thread_execution_width) =
        crate::backend::vulkan::engine::compute_threadgroup_limits();
    [
        (COMPUTE_INFO_KEY_MAX_TOTAL_THREADS, max_total_threads),
        (
            COMPUTE_INFO_KEY_THREAD_EXECUTION_WIDTH,
            thread_execution_width,
        ),
        (COMPUTE_INFO_KEY_STATIC_THREADGROUP_MEMORY, 0),
    ]
}

/// Child `CmdGetComputeInfo` (0x3b): 24B payload
/// `[task_id@0][pipeline_ref@4][key_table_len@8][count@12][reply_gva@16]`.
/// Host writes key/value pairs at reply_gva before stamp (Apple host contract).
fn reply_compute_info<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    payload: &[u8],
) -> bool {
    if payload.len() < 24 {
        return false;
    }
    let raw_task = ld32(&payload[0..]);
    let pipeline_ref = ld32(&payload[4..]);
    let key_table_len = ld32(&payload[8..]);
    let count = ld32(&payload[12..]);
    let reply_gva = u64::from_le_bytes(payload[16..24].try_into().unwrap_or([0; 8]));
    if reply_gva == 0 || count == 0 {
        crate::observe::fail(format!(
            "get_compute_info empty task={raw_task} pipe={pipeline_ref} key_table_len={key_table_len} count={count} gva={reply_gva:#x}"
        ));
        return false;
    }
    // A live slot or nothing; `bad_task` now names the word the guest sent
    // rather than the halved id this used to have resolved to by then.
    let Some(task_id) = resolve_task_word(&state.tasks, TaskWordSite::ComputeInfo, raw_task) else {
        crate::observe::fail(format!(
            "get_compute_info bad_task task={raw_task} pipe={pipeline_ref}"
        ));
        return false;
    };
    let mut wrote = 0u32;
    for (key, value) in compute_info_caps() {
        // Exclusive, like its device-info twin: the guest writes
        // `highest_key_it_parses + 1` here, and its parser's own table stops one
        // below. It sends 5 against a table whose arms are keys 1..=4, so an
        // inclusive read would spend a pair slot on a key 5 that guest discards.
        // See `DEVICE_INFO_TAHOE_KEY_TABLE_LEN`, which carries the argument for both.
        if key >= key_table_len {
            continue;
        }
        if wrote >= count {
            break;
        }
        let mut pair = [0u8; DEVICE_INFO_REPLY_PAIR_LEN];
        st32(&mut pair[0..4], key);
        st32(&mut pair[4..8], value);
        let off = (wrote as u64) * DEVICE_INFO_REPLY_PAIR_LEN as u64;
        if crate::runtime::gva_mem::write_task_gva_product_within(
            state,
            host,
            task_id,
            reply_gva + off,
            &pair,
            // One reply pair at a fixed offset, not a row loop: the span is
            // re-derived from `wrote` rather than from a destination that can
            // move, and the packet being retired is what authorises it.
            None,
        )
        .is_err()
        {
            crate::observe::fail(format!(
                "get_compute_info write_fail task={task_id} gva={reply_gva:#x} wrote={wrote}"
            ));
            return false;
        }
        wrote += 1;
    }
    if wrote < count {
        let sentinel = [0u8; DEVICE_INFO_REPLY_PAIR_LEN];
        let off = (wrote as u64) * DEVICE_INFO_REPLY_PAIR_LEN as u64;
        let _ = crate::runtime::gva_mem::write_task_gva_product_within(
            state,
            host,
            task_id,
            reply_gva + off,
            &sentinel,
            // The terminating pair of the same reply, on the same authority.
            None,
        );
    }
    // Success census — the reply landed. Route to `off()` so it stays always-on
    // in the log but leaves the curated real-error view clean; the genuine
    // failures (`empty`/`bad_task`/`write_fail`/`short`) above stay `fail()`.
    crate::observe::off(format!(
        "get_compute_info ok task={task_id} pipe={pipeline_ref} key_table_len={key_table_len} count={count} wrote={wrote} gva={reply_gva:#x}"
    ));
    true
}

fn reply_heap_texture_size_and_align<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    payload: &[u8],
) -> bool {
    let request = match crate::runtime::heap_query::decode_request(payload) {
        Ok(request) => request,
        Err(error) => {
            Emit::decline("heap_texture_query", &error)
                .field("plen", payload.len())
                .fail();
            return false;
        }
    };
    // A live slot or nothing. `resolved_task` is gone with the arm that made the
    // two differ: the only slot this can act on is the one the guest named.
    let Some(task_id) = resolve_task_word(
        &state.tasks,
        TaskWordSite::HeapTextureQuery,
        request.task_id,
    ) else {
        Emit::decline("heap_texture_query", &QueryError::BadTask)
            .field("task", request.task_id)
            .field("gva", format!("{:#x}", request.reply_gva))
            .fail();
        return false;
    };
    let requirement = match crate::runtime::heap_query::query_size_and_align(&request.descriptor) {
        Ok(requirement) => requirement,
        Err(error) => {
            let desc = request.descriptor;
            Emit::decline("heap_texture_query", &error)
                .field("task", task_id)
                .field("type", desc.texture_type)
                .field("fmt", format!("{:#x}", desc.pixel_format))
                .field(
                    "dims",
                    format!("{}x{}x{}", desc.width, desc.height, desc.depth),
                )
                .field("mips", desc.mipmap_level_count)
                .field("samples", desc.sample_count)
                .field("array", desc.array_length)
                .field("usage", format!("{:#x}", desc.usage))
                .field("options", format!("{:#x}", desc.resource_options))
                .fail();
            return false;
        }
    };
    let reply = requirement.encode();
    if crate::runtime::gva_mem::write_task_gva_product_within(
        state,
        host,
        task_id,
        request.reply_gva,
        &reply,
        // A single completion stamp for the packet being retired, which is
        // what authorises it. No loop, so no window to capture.
        None,
    )
    .is_err()
    {
        crate::observe::fail(format!(
            "heap_texture_query fail reason=write_fail task={task_id} gva={:#x} reply_len={} size={:#x} align={:#x}",
            request.reply_gva,
            request.reply_len,
            requirement.size,
            requirement.align
        ));
        return false;
    }
    let desc = request.descriptor;
    crate::observe::off(format!(
        "heap_texture_query ok task={task_id} gva={:#x} type={} fmt={:#x} {}x{}x{} mips={} samples={} array={} usage={:#x} options={:#x} size={:#x} align={:#x}",
        request.reply_gva,
        desc.texture_type,
        desc.pixel_format,
        desc.width,
        desc.height,
        desc.depth,
        desc.mipmap_level_count,
        desc.sample_count,
        desc.array_length,
        desc.usage,
        desc.resource_options,
        requirement.size,
        requirement.align
    ));
    true
}

fn process_root_packet<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    packet: &Packet,
) {
    match packet.opcode {
        ROOT_OP_DEVICE_INFO_TAHOE => {
            if !packet_short(
                "device_info_tahoe",
                None,
                packet.payload.len(),
                DEVICE_INFO_TAHOE_REPLY_PFN + 4,
            ) {
                let key_table_len = ld32(&packet.payload[DEVICE_INFO_TAHOE_KEY_TABLE_LEN..]);
                let count = ld32(&packet.payload[DEVICE_INFO_TAHOE_COUNT..]);
                let pfn = ld32(&packet.payload[DEVICE_INFO_TAHOE_REPLY_PFN..]);
                let _ = reply_device_info(
                    host,
                    key_table_len,
                    count,
                    pfn,
                    state.page_shift,
                    state.gfx.version,
                );
            }
        }
        ROOT_OP_DEVICE_INFO_MONTEREY => {
            if !packet_short(
                "device_info_monterey",
                None,
                packet.payload.len(),
                DEVICE_INFO_MONTEREY_REPLY_PFN + 4,
            ) {
                let count = ld32(&packet.payload[DEVICE_INFO_MONTEREY_COUNT..]);
                let pfn = ld32(&packet.payload[DEVICE_INFO_MONTEREY_REPLY_PFN..]);
                // This record carries no parse ceiling — see
                // `DEVICE_INFO_MONTEREY_COUNT` — so the reply names every key the
                // table holds and the count alone bounds it, which is what this
                // arm has always done.
                let _ = reply_device_info(
                    host,
                    u32::MAX,
                    count,
                    pfn,
                    state.page_shift,
                    state.gfx.version,
                );
            }
        }
        ROOT_OP_DEFINE_FIFO => {
            if !packet_short("define_fifo", None, packet.payload.len(), 4) {
                let ch = ld32(&packet.payload[0..]);
                if is_child_channel(ch) {
                    let bit = 1u32 << ch;
                    state.active_child_mask |= bit;
                    state.translation_deferred_mask &= !bit;
                    state.translation_order_hold_mask &= !bit;
                    state.present_translation_hold_mask &= !bit;
                    // Invalidate ring cache for this channel.
                    state.child_rings[ch as usize] = Default::default();
                }
            }
        }
        ROOT_OP_FREE_FIFO => {
            if !packet_short("free_fifo", None, packet.payload.len(), 4) {
                let ch = ld32(&packet.payload[0..]);
                if is_child_channel(ch) {
                    let bit = 1u32 << ch;
                    state.active_child_mask &= !bit;
                    state.pending.child_mask &= !bit;
                    state.translation_deferred_mask &= !bit;
                    state.translation_order_hold_mask &= !bit;
                    state.present_translation_hold_mask &= !bit;
                    state.child_rings[ch as usize] = Default::default();
                }
            }
        }
        ROOT_OP_DEFINE_TASK2 => apply_define_task2(state, host, &packet.payload, None),
        ROOT_OP_SET_OBJECT_LIST => apply_set_object_list(state, &packet.payload, None),
        ROOT_OP_DELETE_TASK => apply_delete_task(state, &packet.payload, None),
        // One command in one flat opcode space, not a wrapper. See
        // `ROOT_OP_SETUP_SHARED_STATE` for what this arm used to do with the
        // payload's first word, and why nothing here ever caught it.
        ROOT_OP_SETUP_SHARED_STATE => {
            apply_setup_shared_state(state, host, &packet.payload, None);
        }
        _ => {
            state.record_fail(FailEvent::UnknownRootOpcode {
                opcode: packet.opcode,
                total_size: packet.total_size,
            });
        }
    }
}

/// Drain the main (root) FIFO while producer != consumer.
pub fn drain_main_fifo<H: HostMemory + HostOps>(state: &mut DeviceState, host: &mut H) {
    // The hold bit means "the last drain of this timeline stopped on an unmet
    // stamp wait", so it is cleared on entry and set only by the arm that stops.
    // Left sticky it would outlive the packet that set it and keep
    // `retry_stamp_held_timelines` re-entering a ring with nothing in it.
    state.stamp_deferred_mask &= !ROOT_FIFO_BIT;
    let ring_size = main_ring_data_size(state.gfx.fifo_length, state.gfx.fifo_start);
    if ring_size == 0 || state.gfx.fifo_base_page == 0 {
        state.pending.main_drain = false;
        return;
    }
    let base = state.pfn_gpa(state.gfx.fifo_base_page) + state.gfx.fifo_start as u64;
    let mut completed = false;

    while state
        .gfx
        .fifo_read
        .load(std::sync::atomic::Ordering::Acquire)
        != state.gfx.fifo_written
    {
        let Some(available) = published_byte_count(
            state
                .gfx
                .fifo_read
                .load(std::sync::atomic::Ordering::Acquire),
            state.gfx.fifo_written,
            ring_size,
        ) else {
            state.record_fail(FailEvent::MalformedRootPacket {
                fault: PacketFault::DesyncedHeadTail,
                head: state
                    .gfx
                    .fifo_read
                    .load(std::sync::atomic::Ordering::Acquire),
            });
            break;
        };
        if available < PACKET_HEADER_LEN {
            break;
        }
        // Snapshot up to min(available, ring_size) — header first then full packet.
        let header = match read_ring_bytes(
            host,
            base,
            ring_size,
            state
                .gfx
                .fifo_read
                .load(std::sync::atomic::Ordering::Acquire),
            PACKET_HEADER_LEN,
        ) {
            Ok(h) => h,
            Err(_) => {
                state.record_fail(FailEvent::MalformedRootPacket {
                    fault: PacketFault::RootHeaderRead,
                    head: state
                        .gfx
                        .fifo_read
                        .load(std::sync::atomic::Ordering::Acquire),
                });
                break;
            }
        };
        let snap_len = packet_snapshot_len(&header, available, ring_size);
        let snap = match read_ring_bytes(
            host,
            base,
            ring_size,
            state
                .gfx
                .fifo_read
                .load(std::sync::atomic::Ordering::Acquire),
            snap_len,
        ) {
            Ok(s) => s,
            Err(_) => {
                state.record_fail(FailEvent::MalformedRootPacket {
                    fault: PacketFault::RootSnapRead,
                    head: state
                        .gfx
                        .fifo_read
                        .load(std::sync::atomic::Ordering::Acquire),
                });
                break;
            }
        };
        match decode_packet(
            &snap,
            state
                .gfx
                .fifo_read
                .load(std::sync::atomic::Ordering::Acquire),
            available,
            ring_size,
        ) {
            Ok(packet) => {
                if note_packet_stamp_waits(state, host, None, &packet) == StampVerdict::Hold {
                    // Same hold as the child drain, and this is the timeline
                    // where it matters most: the measured root wait is a
                    // `DELETE_TASK` ordered behind a child FIFO's stamp, so
                    // running it early tears down a task whose work has not
                    // been drained. `fifo_read` stays where it is and no root
                    // completion stamp is written, so the retry is the same
                    // packet with the same effects still owed.
                    note_store_route("packet_stamp_wait_held");
                    state.stamp_deferred_mask |= ROOT_FIFO_BIT;
                    state.pending.main_drain = true;
                    break;
                }
                process_root_packet(state, host, &packet);
                state
                    .gfx
                    .fifo_read
                    .store(packet.next_head, std::sync::atomic::Ordering::Release);
                // Root stamp = slot 0.
                if state.gfx.fifo_base_page != 0 {
                    if let Some(off) = stamp_slot_offset(0, state.page_size()) {
                        // The root stamp is a completion the guest waits on, so
                        // every deferred rail owes guest RAM its bytes here, not
                        // only at `write_stamp`'s child slots. The settle is the
                        // `quiesce_guest_writes` below — one call, not two, since
                        // the first would clear the debt the second then finds
                        // clear.
                        //
                        // Root slot 0 stays on the blocking rail, and the
                        // measurement is the reason rather than caution. Routing
                        // it through the GPU rail as well was booted and scored
                        // worse on the thing the guest sees: presents/s 63 -> 53
                        // against an otherwise identical tree, with draws/s
                        // 3320 -> 3141. The child slots keep the fast rail
                        // because there it pays; this is a submission and a
                        // thread hop for a stamp the guest is already waiting on.
                        //
                        // The first attempt at it was also outright broken, and
                        // that shape is worth keeping: `completed` raises slot
                        // 0's interrupt *after* this loop, so a GPU-ordered root
                        // stamp that still set it announced the completion before
                        // the word moved. The guest treats an interrupt as its
                        // wakeup, re-read the old value, and slept out its
                        // one-second deadline — a total stall, draws/s to 0 and
                        // no presents. Any future attempt here must leave
                        // `completed` false and let the completion thread
                        // announce.
                        crate::runtime::render_writeback::settle_guest_writes(
                            crate::runtime::render_writeback::SettleSite::RootStamp,
                        );
                        let gpa = state.pfn_gpa(state.gfx.fifo_base_page) + off;
                        if gpa_map::write_u32(
                            host,
                            gpa,
                            packet.completion_stamp,
                            state.page_size() as usize,
                        )
                        .is_ok()
                        {
                            // A window armed after this point has outlived a fence
                            // the moment it is still armed at the next one. The
                            // counter is what `armed_stamp_seq` is compared
                            // against, so a rail that does not move it reads as
                            // punctual however long it actually waited.
                            state.completion_stamp_seq = state.completion_stamp_seq.wrapping_add(1);
                            completed = true;
                        } else {
                            // The guest waits on this root completion stamp; a
                            // silent writeback failure hangs it forever with no
                            // trace (drain.rs Rank-2 audit).
                            state.record_fail(FailEvent::MalformedRootPacket {
                                fault: PacketFault::RootStampWriteback,
                                head: state
                                    .gfx
                                    .fifo_read
                                    .load(std::sync::atomic::Ordering::Acquire),
                            });
                        }
                    }
                }
            }
            Err(err) => {
                // `fault()` decides which errors reach the log: a packet the
                // producer is still writing is ring control flow and answers
                // `None`. Either way the drain stops here and comes back.
                if let Some(fault) = err.fault() {
                    state.record_fail(FailEvent::MalformedRootPacket {
                        fault,
                        head: state
                            .gfx
                            .fifo_read
                            .load(std::sync::atomic::Ordering::Acquire),
                    });
                }
                break;
            }
        }
    }

    if completed {
        state
            .gfx
            .interrupt_status_gpu
            .fetch_or(1, std::sync::atomic::Ordering::AcqRel);
        host.enqueue(HostAction::irq_gfx());
    }
    // A root head held on an unmet stamp wait is unfinished work, not a drained
    // ring: clearing the flag unconditionally here would drop the retry and the
    // packet would only run again if some later doorbell happened to set it.
    state.pending.main_drain = state.stamp_deferred_mask & ROOT_FIFO_BIT != 0;
}

fn ensure_child_ring<M: HostMemory>(
    state: &mut DeviceState,
    mem: &M,
    channel_id: u32,
    base_pfn: u32,
) -> u32 {
    // `base_pfn == 0` is a ring the guest has not published yet — expected
    // control flow, and quiet. An out-of-range channel is not: it returns the
    // same `0` and so reads as "not published yet" forever, which is why it is
    // reported through the rule's own spelling rather than folded in here.
    if !crate::model::accept_child_channel(channel_id, "ensure_child_ring") || base_pfn == 0 {
        return 0;
    }
    let page_shift = state.page_shift;
    let page_size = state.page_size();
    let ring = &mut state.child_rings[channel_id as usize];
    if ring.valid && ring.base_pfn == base_pfn {
        return ring.length;
    }
    // Count leading non-zero PFNs in the page list (one page of u32 PFNs).
    let list_gpa = pfn_to_gpa(base_pfn, page_shift);
    let max_entries = (page_size / CHILD_RING_PFN_ENTRY_LEN) as u32;
    let mut page_gpas = Vec::new();
    for i in 0..max_entries {
        let mut b = [0u8; 4];
        if mem
            .read_gpa(list_gpa + i as u64 * CHILD_RING_PFN_ENTRY_LEN, &mut b)
            .is_err()
        {
            break;
        }
        let pfn = u32::from_le_bytes(b);
        if pfn == 0 {
            break;
        }
        page_gpas.push(pfn_to_gpa(pfn, page_shift));
    }
    let length = (page_gpas.len() as u32).saturating_mul(page_size as u32);
    *ring = crate::model::ChannelRing {
        valid: length != 0,
        base_pfn,
        length,
        page_gpas,
    };
    length
}

fn read_child_ring_bytes<M: HostMemory>(
    mem: &M,
    page_gpas: &[u64],
    ring_length: u32,
    absolute: u32,
    len: u32,
    page_shift: u32,
) -> Result<Vec<u8>, MemError> {
    let page_size = 1u64 << page_shift;
    let mut out = vec![0u8; len as usize];
    if ring_length == 0 || page_gpas.is_empty() {
        return Ok(out);
    }
    for i in 0..len {
        let off = absolute.wrapping_add(i) % ring_length;
        let page = (off as u64) >> page_shift;
        let page_off = (off as u64) & (page_size - 1);
        if page as usize >= page_gpas.len() {
            out[i as usize] = 0;
            continue;
        }
        let mut b = [0u8; 1];
        mem.read_gpa(page_gpas[page as usize] + page_off, &mut b)?;
        out[i as usize] = b[0];
    }
    Ok(out)
}

fn shared_w16<H: HostMemory + HostOps>(host: &mut H, gpa: u64, off: u64, v: u16, page_size: usize) {
    let mut b = [0u8; 2];
    st16(&mut b, v);
    let _ = gpa_map::write_bytes(host, gpa + off, &b, page_size);
}

fn shared_w32<H: HostMemory + HostOps>(host: &mut H, gpa: u64, off: u64, v: u32, page_size: usize) {
    let mut b = [0u8; 4];
    st32(&mut b, v);
    let _ = gpa_map::write_bytes(host, gpa + off, &b, page_size);
}

/// Fill the guest display descriptor page (archive `apple_pv_gpu_display_setup`).
///
///: `+0x208` is the timing-element **count**, not a
/// pixel width. Modes are 1920×1080, 1440×1080, 1280×1024 (apple-gfx A/B
/// reference geometry) plus 3840×2160 (4K UHD), each advertised at
/// `DISPLAY_REFRESH_HZ` (120 Hz), so the guest always latches the 120 Hz mode.
/// Element 0 (1920×1080) stays the native/preferred format (+0x210/+0x212 double
/// as NativeFormat*Pixels), so boot resolution is unchanged and 4K is an
/// additional selectable mode; the dynamic scanout/present/host-window geometry
/// follows the surface the guest actually presents at the selected mode.
fn fill_display_descriptor<H: HostMemory + HostOps>(
    host: &mut H,
    gpa: u64,
    index: u32,
    page_size: u64,
) {
    if gpa == 0 {
        return;
    }
    let Some(refresh) = display_refresh_hz_1616(DISPLAY_REFRESH_HZ) else {
        return;
    };
    let psz = page_size as usize;

    shared_w32(host, gpa, DISPLAY_DESC_SERIAL, DISPLAY_SERIAL_NUMBER, psz);
    let _ = gpa_map::write_bytes(
        host,
        gpa + DISPLAY_DESC_PRODUCT_NAME,
        DISPLAY_PRODUCT_NAME,
        psz,
    );
    shared_w16(host, gpa, DISPLAY_DESC_INDEX, index as u16, psz);
    shared_w16(host, gpa, DISPLAY_DESC_WIDTH_MM, DISPLAY_WIDTH_MM, psz);
    shared_w16(host, gpa, DISPLAY_DESC_HEIGHT_MM, DISPLAY_HEIGHT_MM, psz);
    shared_w32(host, gpa, DISPLAY_DESC_FEATURES, 0, psz);

    // HW cursor capability so the guest doorbells glyph/show/move.
    let max_wh = (CURSOR_MAX_DIM & 0xffff) | ((CURSOR_MAX_DIM & 0xffff) << 16);
    shared_w32(host, gpa, DISPLAY_SHARED_CURSOR_MAX_WH, max_wh, psz);
    shared_w32(
        host,
        gpa,
        DISPLAY_SHARED_CURSOR_FEATURES,
        DISPLAY_CURSOR_FEATURE_HW,
        psz,
    );

    const MODES: &[(u16, u16)] = &[
        (DISPLAY_MODE_EFI_W, DISPLAY_MODE_EFI_H),
        (DISPLAY_MODE1_W, DISPLAY_MODE1_H),
        (DISPLAY_MODE2_W, DISPLAY_MODE2_H),
        (DISPLAY_MODE3_W, DISPLAY_MODE3_H),
    ];
    shared_w16(
        host,
        gpa,
        DISPLAY_DESC_TIMING_COUNT,
        MODES.len() as u16,
        psz,
    );

    let mut encoded = [0u8; DISPLAY_DESC_TIMING_STRIDE as usize];
    for (i, &(width, height)) in MODES.iter().enumerate() {
        let Some(off) = display_timing_entry_offset(i as u32, page_size) else {
            return;
        };
        let entry = DisplayTimingEntry {
            width,
            height,
            refresh_1616: refresh,
            tail0: 0,
            tail1: 0,
        };
        encoded.fill(0);
        if !encode_display_timing_entry(&entry, &mut encoded) {
            return;
        }
        let _ = gpa_map::write_bytes(host, gpa + off, &encoded, psz);
    }
}

/// Sample cursor x/y/show from the display shared-state page (GPA +0xe00).
fn sample_cursor_position<M: HostMemory>(state: &mut DeviceState, mem: &M) {
    if state.display.shared_gpa == 0 {
        return;
    }
    let mut pos = [0u8; 4];
    if mem
        .read_gpa(
            state.display.shared_gpa + DISPLAY_SHARED_CURSOR_POS,
            &mut pos,
        )
        .is_err()
    {
        return;
    }
    let packed = ld32(&pos);
    if packed == 0xffff_ffff {
        state.cursor.show = false;
        return;
    }
    state.cursor.x = (packed & 0xffff) as u16;
    state.cursor.y = ((packed >> 16) & 0xffff) as u16;
    let mut show = [0u8; 4];
    if mem
        .read_gpa(
            state.display.shared_gpa + DISPLAY_SHARED_CURSOR_SHOW,
            &mut show,
        )
        .is_ok()
    {
        // Guest may only write a byte; treat non-zero low byte as show.
        state.cursor.show = show[0] != 0 || ld32(&show) != 0;
    }
}

/// Load CmdDisplayCursorGlyph pixels (BGRA guest → ARGB QEMUCursor).
/// Fail-visible, once per reason per boot, for the silent `load_cursor_glyph`
/// drop sites: a malformed cursor-glyph packet leaves the cursor stale/wrong
/// with no log. Cursor glyphs are infrequent (sent when the pointer *image*
/// changes, not per move) but a persistently-bad glyph could repeat, so latch
/// each reason once. Always returns `false` so callers stay `return cg_fail(..)`.
fn cursor_glyph_fail(reason: &'static str, detail: String) -> bool {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<&'static str>>> = Mutex::new(None);
    let mut guard = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if guard.get_or_insert_with(HashSet::new).insert(reason) {
        crate::observe::fail(detail);
    }
    false
}

fn load_cursor_glyph<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &H,
    packet: &Packet,
) -> bool {
    if packet.payload.len() < CURSOR_GLYPH_PAYLOAD_LEN {
        return cursor_glyph_fail(
            "cursor_glyph_short",
            format!(
                "cursor_glyph_fail reason=cursor_glyph_short plen={} need={CURSOR_GLYPH_PAYLOAD_LEN}",
                packet.payload.len()
            ),
        );
    }
    let task_id = ld32(&packet.payload[0x04..]);
    let virtual_offset = u64::from_le_bytes(packet.payload[0x08..0x10].try_into().unwrap());
    let mapped_length = u64::from_le_bytes(packet.payload[0x10..0x18].try_into().unwrap());
    let stride = u64::from_le_bytes(packet.payload[0x18..0x20].try_into().unwrap()) as u32;
    let width = ld16(&packet.payload[0x20..]) as u32;
    let height = ld16(&packet.payload[0x22..]) as u32;
    let hot_x = ld16(&packet.payload[0x24..]) as u32;
    let hot_y = ld16(&packet.payload[0x26..]) as u32;

    if width == 0
        || height == 0
        || width > CURSOR_MAX_DIM
        || height > CURSOR_MAX_DIM
        || stride < width.saturating_mul(CURSOR_GLYPH_BPP)
        || hot_x >= width
        || hot_y >= height
    {
        return cursor_glyph_fail(
            "cursor_glyph_geom",
            format!("cursor_glyph_fail reason=cursor_glyph_geom {width}x{height} stride={stride} hot=({hot_x},{hot_y}) max={CURSOR_MAX_DIM}"),
        );
    }
    let need = (height as u64 - 1)
        .saturating_mul(stride as u64)
        .saturating_add(width as u64 * CURSOR_GLYPH_BPP as u64);
    if mapped_length < need {
        return cursor_glyph_fail(
            "cursor_glyph_mapped_len",
            format!("cursor_glyph_fail reason=cursor_glyph_mapped_len mapped_length={mapped_length} need={need} {width}x{height}"),
        );
    }
    let Some(need_host) = crate::runtime::draw::host_alloc_len(need) else {
        return cursor_glyph_fail(
            "cursor_glyph_alloc",
            format!("cursor_glyph_fail reason=cursor_glyph_alloc need={need}"),
        );
    };

    let mut src = vec![0u8; need_host];
    if crate::runtime::gva_mem::read_task_gva_by_id(
        host,
        &state.tasks,
        task_id,
        virtual_offset,
        &mut src,
        state.page_shift,
    )
    .is_err()
    {
        return cursor_glyph_fail(
            "cursor_glyph_read",
            format!("cursor_glyph_fail reason=cursor_glyph_read task={task_id} voff={virtual_offset:#x} need_host={need_host}"),
        );
    }

    let mut pixels = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        let row = (y as usize).saturating_mul(stride as usize);
        for x in 0..width {
            let px = row + (x as usize) * CURSOR_GLYPH_BPP as usize;
            if px + 4 > src.len() {
                return cursor_glyph_fail(
                    "cursor_glyph_bounds",
                    format!("cursor_glyph_fail reason=cursor_glyph_bounds px={px} src_len={} {width}x{height} stride={stride}", src.len()),
                );
            }
            let b = src[px];
            let g = src[px + 1];
            let r = src[px + 2];
            let a = src[px + 3];
            // QEMUCursor 0xAARRGGBB
            pixels.push(((a as u32) << 24) | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32));
        }
    }

    state.cursor.width = width as u16;
    state.cursor.height = height as u16;
    state.cursor.hot_x = hot_x as u16;
    state.cursor.hot_y = hot_y as u16;
    state.cursor.pixels = pixels;
    state.cursor.glyph_ready = true;
    sample_cursor_position(state, host);
    true
}

/// Account one accepted present and request a worker→host action boundary.
///
/// Yielding here bounds how far the drain runs ahead of the display consumer.
/// Continuing to consume guest work can fill `pending_frames`, then hold
/// Display0 forever while its frame remains unconsumed.
fn enqueue_present_scanout<H: HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    width: u32,
    height: u32,
) {
    // Two presentation paths, selected by the live window link:
    //
    // - Window active (x86 default): no CPU `ScanoutUpdate`. QEMU runs
    //   `-display none`, no DisplayChangeListener ticks `gfx_update`, and the
    //   surface would be painted for nobody. The window is fed by
    //   `publish_window_frame` from this drain, and the present-completion ack
    //   is re-homed onto the drain tail (see `device_drain`).
    //
    // - No window (arm64 MMIO `-display cocoa`, or `REIMS_VGPU_WINDOW=0`): the QEMU
    //   console IS the display, so every present enqueues the CPU
    //   `ScanoutUpdate` (coalesced latest-wins in the action queue) and the
    //   ack comes from the console paint (`device_scanout_copy`), releasing
    //   `unpainted_presents` + `present_action_pending` there. Skipping the
    //   action here freezes the console at the last pre-boundary early-FB
    //   paint while the guest keeps presenting (live class: arm64 boot
    //   serial-20260723-221445, console stuck on the 15% progress bar while
    //   gen 38 presented the login wallpaper).
    if !state.present.window_active {
        host.enqueue(HostAction::scanout_gen(
            state.present.frame_mapping,
            width,
            height,
            state.present.frame_generation,
        ));
    }
    state.present.unpainted_presents = state.present.unpainted_presents.saturating_add(1);
    state.pending.host_action_yield = true;
}

fn present_page_identity_line(state: &DeviceState, mapping: u32, w: u32, h: u32) -> Option<String> {
    use std::collections::HashSet;
    let named = state.mappings.get(&mapping)?;
    let named_pfns: HashSet<u32> = named
        .page_entries
        .iter()
        .filter(|&&e| e & crate::contract::iosurface_pages::PAGE_ENTRY_VALID != 0)
        .map(|&e| e >> crate::contract::iosurface_pages::PAGE_ENTRY_PFN_SHIFT)
        .collect();
    let mut peers = String::new();
    for (&mid, m) in state.mappings.iter() {
        if mid == mapping
            || !m.has_geom
            || m.width != w
            || m.height != h
            || m.page_entries.is_empty()
        {
            continue;
        }
        let identical = m.page_entries == named.page_entries;
        let overlap = if identical {
            named_pfns.len()
        } else {
            m.page_entries
                .iter()
                .filter(|&&e| e & crate::contract::iosurface_pages::PAGE_ENTRY_VALID != 0)
                .filter(|&&e| {
                    named_pfns
                        .contains(&(e >> crate::contract::iosurface_pages::PAGE_ENTRY_PFN_SHIFT))
                })
                .count()
        };
        if !peers.is_empty() {
            peers.push(',');
        }
        peers.push_str(&format!(
            "mid{mid}:pages={}:overlap={overlap}:ident={}:kind={:?}",
            m.page_entries.len(),
            identical as u8,
            state.surface_write_kind(mid)
        ));
    }
    Some(format!(
        "present_page_identity mid={mapping} {w}x{h} pages={} valid={} map_gen={} kind={:?} peers=[{peers}]",
        named.page_entries.len(),
        named_pfns.len(),
        named.map_generation,
        state.surface_write_kind(mapping)
    ))
}

/// Which of the two present routes a present took, once per distinct route per
/// process.
///
/// Every present captures the surface the transaction names. This line splits
/// them on the named surface's write history anyway: `route=clear_only` is a
/// present whose named mid's most recent write was a `display_clear`/CLEAR
/// Store rather than a draw — the guest asking us to show a surface it has only
/// ever cleared. `route=named` is everything else. The split is the standing
/// measurement of whether that case occurs at all on a given rail; two lines per
/// process at most, which is what makes it safe to leave on.
///
/// **Measured: only `route=named write_kind=Composite`, on 104 x86/Vulkan boots
/// — every boot in the failure log since this line landed.** Not one
/// `route=clear_only`, including a 1766 s session driven through the
/// heavy-Safari residue repro. The dedup is per process, so one line per boot is
/// the whole reading for that boot.
///
/// That is an x86 statement only. `note_surface_clear` marks a mid ClearOnly
/// from a decoded `display_clear`/CLEAR Store, which is not rail-specific — what
/// the measurement shows is that on x86 the guest never *presents* a mid whose
/// most recent write was a Clear. An arm64 reading of this same line is what
/// would say whether that holds everywhere.
fn note_present_route(write_kind: crate::model::SurfaceWriteKind, is_clear_only: bool) {
    use std::sync::Mutex;
    static SEEN: Mutex<Option<std::collections::BTreeSet<bool>>> = Mutex::new(None);
    {
        let mut guard = SEEN.lock().unwrap_or_else(|p| p.into_inner());
        if !guard
            .get_or_insert_with(Default::default)
            .insert(is_clear_only)
        {
            return;
        }
    }
    crate::observe::fail(format!(
        "present_route route={} write_kind={write_kind:?}",
        if is_clear_only { "clear_only" } else { "named" },
    ));
}

fn log_present_page_identity(state: &DeviceState, mapping: u32, w: u32, h: u32) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(u32, u32)>>> = Mutex::new(None);
    let Some(named) = state.mappings.get(&mapping) else {
        return;
    };
    let key = (mapping, named.map_generation);
    {
        let mut guard = SEEN.lock().unwrap_or_else(|p| p.into_inner());
        let seen = guard.get_or_insert_with(HashSet::new);
        if seen.len() > 1024 {
            seen.clear();
        }
        if !seen.insert(key) {
            return;
        }
    }
    if let Some(line) = present_page_identity_line(state, mapping, w, h) {
        crate::observe::fail(line);
    }
}

/// Present a named mapping to the host console (op8 DisplaySwapMapping, or the
/// x86 display pipe's op6/op7 transactions).
fn present_named_mapping<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    channel_id: u32,
    mapping: u32,
) -> ChildPacketDisposition {
    if mapping == 0 {
        return ChildPacketDisposition::Complete;
    }
    // Archive apple_pv_gpu_display_swap:
    //   render_wait_surface(s, false, swap->mapping_id);
    //   scanout_present_boundary(...);
    //
    // Plus archive poll_tick Dekker rescue (apple_pv_gpu_poll_tick):
    // guest may publish child work without a doorbell while a drain
    // was in flight. Product has no separate host timer during the
    // DisplaySwap packet; drain **other** child FIFOs (skip mid-
    // packet channel) before and after wait_surface so body-layer
    // draws that land during the wait are frozen into the retain.
    // Never re-enter skip/draining_mask channels (boot wedge).
    // Not gen-stable multi-round; not surface_inflight invent.
    let skip = if state.draining_channel != 0 {
        state.draining_channel
    } else {
        channel_id
    };
    drain_other_child_fifos(state, host, skip);
    drain_other_child_fifos(state, host, skip);
    // Main-ring Dekker only (not full drain_stranded): guest may
    // publish root control work while child drains ran. Full
    // drain_stranded re-enters this child channel and wedged iBoot
    // (6720ce170). Main drain never re-enters a child mid-packet.
    if state.gfx.control_fifo != 0
        && state
            .gfx
            .fifo_read
            .load(std::sync::atomic::Ordering::Acquire)
            != state.gfx.fifo_written
    {
        drain_main_fifo(state, host);
        // Body-layer child work may be doorbell'd from main packets.
        drain_other_child_fifos(state, host, skip);
    }

    // Preflight translation keeps an EXEC packet at its channel head. If one
    // is still held after all rescue drains, accepting this display packet
    // would publish the prior +0x188 retain before the earlier render packet
    // executes. Leave the display head and stamp untouched. Poll-tick re-drives
    // all active channels; once translation is ready, the EXEC runs and this
    // packet is retried in order without blocking a vCPU or the QEMU main loop.
    let current_bit = 1u32.checked_shl(channel_id).unwrap_or(0);
    let deferred_other = state.translation_deferred_mask & !current_bit;
    if deferred_other != 0 {
        if state.present_translation_hold_mask & current_bit == 0 {
            state.present_translation_holds = state.present_translation_holds.saturating_add(1);
            state.present_translation_hold_mask |= current_bit;
            crate::observe::fail(format!(
                "present_order_hold reason=translation_deferred ch={channel_id} mid={mapping} pending_mask={deferred_other:#x} frame_mapping={} early_front={} count={}",
                state.present.frame_mapping,
                state.present.early_front_mapping,
                state.present_translation_holds
            ));
        }
        return ChildPacketDisposition::Deferred;
    }
    if state.present_translation_hold_mask & current_bit != 0 {
        state.present_translation_hold_mask &= !current_bit;
        crate::observe::off(format!(
            "present_order_release ch={channel_id} mid={mapping} pending_mask={:#x}",
            state.translation_deferred_mask
        ));
    }

    state.present.present_mapping = mapping;
    state.present.host_mapping = mapping;
    state.present.valid = true;
    // x86: present surface_id → type-4 object-list slot (heap index =
    // IOSurface getSurfaceID). Arm: MappingInternal page-table resolve.
    // Always attempt type-4 when pages empty; then iosfc/mapper path.
    let _ = crate::runtime::objects::ensure_surface_for_present(state, host, mapping);
    let force = state
        .mappings
        .get(&mapping)
        .map(|m| m.mapping_internal != 0)
        .unwrap_or(false);
    if force {
        let _ = crate::runtime::mapper::resolve_mapping_backing(state, host, mapping);
    }
    // Paint only from the presented surface's own geom — never the
    // previous console size fallback (that freezes mode switches).
    // Re-read gen after wait_surface (writebacks may have landed).
    let paint = state.mappings.get(&mapping).and_then(|m| {
        if m.has_geom && m.width > 0 && m.height > 0 {
            Some((m.width, m.height, m.content_generation))
        } else {
            None
        }
    });
    if let Some((w, h, gen)) = paint {
        state.present.width = w;
        state.present.height = h;
        state.present.generation = gen;
        log_present_page_identity(state, mapping, w, h);
        // Every present takes one route: capture the surface the transaction
        // named. A ClearOnly present — one whose named mid's most recent write
        // was a `display_clear`/CLEAR Store rather than a draw — used to take a
        // six-way resolver instead, choosing some *other* same-geometry surface
        // on the theory that the named one held nothing. `note_present_route`
        // still names which route each present takes, and its reading is why
        // there is only one left.
        let write_kind = state.surface_write_kind(mapping);
        let is_clear_only = matches!(write_kind, crate::model::SurfaceWriteKind::ClearOnly);
        note_present_route(write_kind, is_clear_only);

        // presentFrame names the front surface (leave-BAR1 boundary) once we
        // have a non-init present. Geom/capture may still fail after this.
        state.present.frame_flush_seen = true;
        // PGDisplay presentFrame **retains** the named surface into
        // +0x188 at present time; encodeCurrentFrame later re-shows
        // that retained surface (hostPresentCount). Guest may recycle
        // the mapping as soon as this packet's stamp completes — so
        // freeze guest pages **now**, after wait_surface drains, not
        // at BH after the stamp (that freezes mid-recycle partials:
        // toolbar-only dual-mid under app load). Mid-writeback Stores
        // must not recapture here — present boundary only.
        //
        // Always-on backing gate: a member presented twice with no full-frame
        // Store naming it in between is being displayed with content the guest
        // never sent for it. That is a real loss of guest work and belongs in
        // the log; nothing here papers over it.
        //
        // The line says "naming this mid" rather than "received", because that
        // is the whole of what `note_present_backing` read: decoded Store
        // bookkeeping, never the resident.
        //
        // WHICH IS WHY IT ALSO HAS TO READ THE CARRIER. The gate's witness is
        // `dense_frame_seq`, advanced only by `publish_surface_store` — i.e. when
        // a Store's pixels reached the mapping's GUEST PAGES. The resident rail
        // renders into the registry and skips that write, so "no full frame was
        // published for this mid" no longer implies "nothing can show one". A
        // 524 s boot measured four `reason=…never_stored` lines, each claiming
        // the surface was uninitialized and therefore black, against exactly one
        // `host_window_slate*` line in the whole run — a `covered=1` boot run at
        // t=22 s — with `presents == offered` and `direct_frac=1.00` in every
        // cadence window bracketing all four. A resident carried every one of
        // them. The message asserted a visual consequence the check cannot see,
        // which is "a reason the caller writes is not a reading" applied to an
        // outcome instead of a cause.
        //
        // So ask the presenter's own question, through the rule it shares
        // (`pools::slot_presentable`), and split on the answer the same way
        // `host_window_slate` / `host_window_slate_end` already split: a present
        // nothing can carry is a black frame and belongs on the failure channel;
        // one a resident carries cost no guest work and is a census. Reporting
        // both as black cries wolf every boot and — worse — leaves the real case
        // indistinguishable from the benign one, which is how a genuine
        // black-screen boot once produced zero lines here.
        //
        // Priced where it runs: one registry lookup under the engine lock, inside
        // the arm, so only on a present the structural gate has already refused —
        // four times in that boot, not 60 times a second.
        if let Some(backing) = state.note_present_backing(mapping) {
            let carried = present_resident_carries(state, mapping, w, h);
            let emit = crate::observe::Emit::decline("present_unbacked", &backing)
                .field("mid", mapping)
                .field("geom", format!("{w}x{h}"))
                .field("gen", gen)
                .field("carried", carrier_word(carried));
            if unbacked_present_is_a_loss(carried) {
                emit.fail();
            } else {
                emit.off();
            }
        }
        // The transaction payload carries exactly one field: plane 0's surface
        // id. So the capture source is the surface the guest named, and no
        // comparison between our own full-frame sequences may override it.
        // Presenting a "denser" same-geometry peer instead shows a buffer one
        // rotation step behind the one the guest asked for — residue when a
        // window closed in between, a stale region when one moved, thrash as
        // the choice oscillates.
        let encoded = crate::runtime::scanout::capture_present_frame(state, mapping, w, h, gen);
        if !encoded {
            // Retry encode at first host paint. Do **not** clear
            // frame_valid: PGDisplay keeps the prior presentFrame
            // (+0x188) for hostPresentCount until a new capture
            // succeeds. Invalidating the retain forced a black /
            // empty console when dual-mid page resolve raced.
            state.present.frame_encode_pending = true;
            let (pages, mapped, fmt) = state
                .mappings
                .get(&mapping)
                .map(|m| (m.page_entries.len(), m.mapped as u8, m.format))
                .unwrap_or((0, 0, 0));
            crate::observe::fail(format!(
                "present capture fail mid={mapping} {w}x{h} gen={gen} \
                 keep_prior={} pages={pages} mapped={mapped} fmt={fmt:#x}",
                state.present.frame_valid as u8
            ));
        } else {
            // One pass. `bgra_rgb_stats` already maxes the same
            // `px[0].max(px[1]).max(px[2])` per pixel, so a separate scan for
            // `max_rgb` was a second full 8 MiB walk of the frame, under the
            // device lock, for a value this call already returns.
            let (rgb_nz, max_rgb, px0) = crate::observe::bgra_rgb_stats(&state.present.frame_bgra);
            let verdict = present_content_verdict(&state.present.frame_bgra, max_rgb);
            if verdict == PresentContentVerdict::Unsampled {
                // Not a decline: an engine resident carried the frame to the
                // window, so there are no CPU pixels to judge and no guest work
                // was lost. `present_black` below is the alarm. Wherever the
                // window presents from the engine's own device this is the
                // normal outcome of every present.
                crate::observe::line(format!(
                    "present_content_unsampled mid={mapping} {w}x{h} gen={gen} \
                     (a resident carried the frame; no CPU pixels to judge)"
                ));
            } else if verdict == PresentContentVerdict::Black {
                // Both lines name the mapping the guest asked us to show and say
                // it came out black. They deliberately do not go looking for a
                // different surface that looks better: "which other host surface
                // has real content" is a judgement about observed pixels, and
                // `scanout::present_capture` already removed the same walk —
                // same undefended non-zero-pixel threshold — for that reason. A
                // black present is a decode or a writeback fault, and the mid,
                // geometry and generation here are what locate it.
                crate::observe::off(format!(
                    "present_black mid={mapping} {w}x{h} gen={gen} rgb_nz={rgb_nz} px0=[{},{},{},{}] (QMP will be black)",
                    px0[0], px0[1], px0[2], px0[3]
                ));
                crate::observe::fail(format!(
                    "present_black_retain mid={mapping} {w}x{h} gen={gen} (alpha-only/black +0x188)"
                ));
            } else {
                crate::observe::off(format!(
                    "present_content mid={mapping} {w}x{h} gen={gen} rgb_nz={rgb_nz} max_rgb={max_rgb} px0=[{},{},{},{}] encoded={}",
                    px0[0], px0[1], px0[2], px0[3], encoded as u8
                ));
            }
        }
        // No guest-page comparison here. The presented surface's guest window is
        // stale by construction on the Vulkan rail — `import_present` defers the
        // compositor front buffer's writeback on every present, so the pinned
        // resident is authoritative and those pages hold pre-dispatch bytes
        // until a host path reads them. Measured: ~99.5% of the frame differs at
        // full swing on every present, with a deferred window armed every time.
        // The guest's `screencapture` is an oracle because it makes the guest
        // re-execute the composite; its memory for a surface we render into is
        // not.
        // One line per accepted present, verbose-only. `present_enqueue` carried
        // the same fields through the always-on sink alongside it.
        crate::observe::line(format!(
            "present paint mid={mapping} {w}x{h} gen={gen} encoded={} retain={} unpainted={}",
            encoded as u8,
            state.present.frame_valid as u8,
            state.present.unpainted_presents.saturating_add(1)
        ));
        // Account the accepted present. The retain-vs-DisplaySwap (mapping,
        // generation) choice that used to be computed here addressed
        // `copy_to_bgra8`'s Unchanged/expected_generation checks on the QEMU
        // paint; with no paint action produced, the window resolves the frame
        // from `state.present` directly and the distinction has no consumer.
        enqueue_present_scanout(state, host, w, h);
        // Entry-side waitForPendingFrames / apple-gfx pending_frames:
        // count accepted presents until host paint. Stamp still
        // fires with this packet (below) — PGDisplay completion
        // after +0x188 retain, not after host encode.
    } else {
        // Named present without geom: still a product present attempt — leave
        // BAR1 (not a ClearOnly-init handoff defer, which requires geom).
        // Keep early_front peer tracker for dual-mid ClearOnly presents.
        state.present.frame_flush_seen = true;
    }
    // else: hold last painted console (no HostAction / no resize).

    // PGDisplay completion block runs for every present after the
    // +0x188 retain (also when geometry held the paint): display
    // shared-page present bit + conditional display IRQ.
    signal_display_present_complete(state, host);
    ChildPacketDisposition::Complete
}

/// The six lifecycle commands that share [`apply_map_family`]'s body.
///
/// The body branches on this enum and never on the packet's opcode, which is
/// what keeps every branch in it reachable. A branch has to name a variant; a
/// variant is only ever produced at a dispatch arm in `process_child_packet`;
/// and a second arm for an opcode that already has one is an
/// `unreachable_patterns` warning, which this crate's gates take as an error —
/// clippy runs `-D warnings` on every matrix arm and `feature-matrix` expects a
/// zero warning count. So a handler written here for a command this body is not
/// dispatched for does not land. Verified by probe, not assumed.
///
/// That is worth a type because the failure has no behaviour and no other
/// instrument sees it. `CHILD_OP_REPLACE_PHYSICAL` was once branched on inside
/// this body while also owning its own dispatch arm: the outer `match` had
/// already claimed the opcode, so the inner branch was dead — it read exactly
/// like a handler, it compiled, and a coverage sweep reported it as one untaken
/// branch among thousands. The reachable arm cleared the named mapping's page
/// list and the dead one also took the mapping's deferred windows, so the
/// difference was guest work.
///
/// The residual, stated rather than guarded: a *single*-opcode arm can still
/// host the same dead branch. There the enclosing pattern is one number a few
/// lines up rather than six numbers three hundred lines up, so it is visible on
/// inspection, which is why the structure buys the most here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MapFamily {
    /// `CmdMapMemory2` (`0x39`).
    MapMemory2,
    /// `CmdUnmapMemory` (`0x22`).
    UnmapMemory,
    /// `CmdInvalidateResources` (`0x34`).
    InvalidateResources,
    /// `CmdSynchronizeResources` (`0x35`).
    SynchronizeResources,
    /// `CmdSynchronizeAndDiscardResources` (`0x3e`).
    SynchronizeAndDiscardResources,
    /// `CmdDeleteIOSurfaceBacking2` (`0x36`).
    DeleteIOSurfaceBacking2,
}

impl MapFamily {
    /// The command name the fail log spells, one per variant.
    ///
    /// Exhaustive with no `_`, so a new member of the family cannot reach the
    /// log under a placeholder name.
    const fn name(self) -> &'static str {
        match self {
            Self::MapMemory2 => "MapMemory2",
            Self::UnmapMemory => "UnmapMemory",
            Self::InvalidateResources => "InvalidateResources",
            Self::SynchronizeResources => "SynchronizeResources",
            Self::SynchronizeAndDiscardResources => "SynchronizeAndDiscardResources",
            Self::DeleteIOSurfaceBacking2 => "DeleteIOSurfaceBacking2",
        }
    }
}

/// The shared body of the six lifecycle commands named by [`MapFamily`].
fn apply_map_family<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    channel_id: u32,
    packet: &Packet,
    family: MapFamily,
) {
    // Every branch below stamps the packet complete whatever it did with
    // it: the guest is waiting on the stamp, and a lifecycle event this
    // device chose not to act on is still an event the guest performed.
    //
    // Live MapMemory2 plen=20 layout lead (not yet contract-final):
    //   task_id@0 u32, gva@4 u64, length@12 u64  (matches fifo MapMemoryCommand).
    let plen = packet.payload.len();
    let name = family.name();
    if matches!(family, MapFamily::MapMemory2 | MapFamily::UnmapMemory) && plen >= 20 {
        let task_id = crate::contract::endian::ld32(&packet.payload[0..]);
        let gva = crate::contract::endian::ld64(&packet.payload[4..]);
        let length = crate::contract::endian::ld64(&packet.payload[12..]);
        // Verbose-gated walk probe at map/unmap time. This runs a full
        // guest page-table walk (`diagnose_gva_walk`) purely to build the
        // log string, and fired ~9k times/boot on the drain path — a flood
        // and a real per-map cost. Gate it (and the periodic census) behind
        // `REIMS_VGPU_DRAW_LOG=1` so a normal boot pays neither; the functional
        // view-retire below stays always-on. Wire has no PPNs — the probe
        // asks whether the guest PT is already walkable under wire task_id.
        if crate::observe::draw_log_enabled() {
            let walk = crate::runtime::gva_mem::diagnose_gva_walk(
                host,
                &state.tasks,
                task_id,
                gva,
                state.page_shift,
            );
            crate::observe::line(format!(
                "map_probe op={name} ch={channel_id} task={task_id} gva={gva:#x} len={length:#x} page_shift={} {walk}",
                state.page_shift
            ));
            // Periodic active-task census (every 32 map/unmap) for boot overview.
            state.map_family_events = state.map_family_events.saturating_add(1);
            if state.map_family_events == 1 || state.map_family_events.is_multiple_of(32) {
                let census = crate::runtime::gva_mem::format_active_tasks(&state.tasks);
                crate::observe::line(format!(
                    "map_census n={} last_op={name} task={task_id} {census}",
                    state.map_family_events
                ));
            }
        }
        // RE (AppleParavirtMemoryMap): Unmap/Map only mutate the **task
        // page table** then notify — wire has no PPNs. Guest order is
        // deallocate/allocate **then** FIFO, so:
        // - Unmap notify: PTEs already gone → cannot GVA-write; retain
        //   host_gva_surfaces for sample (wallpaper wipe class).
        // - Map notify: PTEs already live → flush host_gva encode into
        //   **new** PFNs (not invent PTEs; not invent geom). Discrete
        //   type-2/3 content may live only in host_cache until this.
        // Samples still prefer host_cache GVA key on Load.
        //
        // HostOps **views** (gva_host_views) are the opposite of encode
        // cache: they alias the pages that were in the GPU PT. On Unmap
        // those pages are no longer mapped for the GPU — drop any host
        // view covering the range (Apple unmapMemory analogue). On Map
        // the PFNs may have changed under the same GVA — drop stale
        // views so the next ensure_gva_view re-walks. Does not invent
        // PTEs and does not destroy host_gva_surfaces content.
        if gva != 0 && length != 0 {
            // Held bind resolutions over this range name pages the guest
            // has just remapped. Retired by range, which is exactly what
            // this notify carries.
            note_bb_retired(
                "bb_retire_map_range",
                state.retire_bound_buffers_in_range(task_id, gva, length),
            );
            let n = crate::runtime::gva_view::retire_gva_views_overlapping(
                state, task_id, gva, length,
            );
            let op = if family == MapFamily::UnmapMemory {
                "unmap_memory"
            } else {
                "map_memory2"
            };
            crate::runtime::gva_view::log_retire(op, task_id, gva, length, n);
        }
        // Deferred GVA render-Store windows overlapping the notified
        // VA range land **cache-only**: on Unmap the PTEs are already
        // gone; on Map the PFNs are fresh and the map-notify guest
        // flush is forbidden (PTE-corruption class). The encode cache
        // preserves the content for samples (wallpaper-retain).
        // There is deliberately no host_cache→guest GVA flush on
        // MapMemory2. One existed and was disabled after
        // serial-20260714-035023: PTE Corruption (freelist-shaped
        // 0xff100000ff000000) ~135s into boot while it was writing —
        // one Map of len=0x1c3e000 alone drove 13 GVA rewrites. Samples
        // use the `host_gva_surfaces` retain on Unmap instead. Any
        // re-introduction has to be a *narrower* policy than that one
        // (exact-base only, no multi-key heap maps) and RE-justified, so
        // the broad implementation is not kept around to be switched
        // back on. See kb map-memory2 / xnu-pte-corruption-windowserver.
    } else if family == MapFamily::DeleteIOSurfaceBacking2 && plen >= 8 {
        // The live Ventura payload agrees with the resource contract:
        // `{objectID, taskID}`. This is the lifetime
        // boundary for the host IOSurface backing, not stamp-only
        // bookkeeping. Keeping page_entries after it lets later id
        // reuse/clear write pixels into pages the guest has recycled.
        let object_id = crate::contract::endian::ld32(&packet.payload[0..]);
        let task_id = crate::contract::endian::ld32(&packet.payload[4..]);
        // Never write guest pages here — the delete trails the guest's
        // CPU-side release asynchronously and the pages may already be
        // recycled (boot-16 PTE-corruption panic: a 14.7 MB delete-time
        // flush landed pixel bytes in a PTE page). But the id itself
        // may ALSO already be re-used by a live surface whose paint is
        // still deferred (~20 ms recycle under scroll — black-band
        // class), so content state must survive until the next page
        // resolve proves which incarnation this delete was for
        // (fingerprint compare in mapper::resolve). A second delete
        // with no resolve between is genuinely dead: tear down fully.
        let mode = if state.mapping_backing_condemned(object_id) {
            let _ = state.unmap_surface(object_id);
            "dead"
        } else if state.condemn_surface_backing(object_id) {
            "condemn"
        } else {
            // No resolved pages ⇒ nothing a stale delete could hurt.
            let _ = state.unmap_surface(object_id);
            "unmapped"
        };
        crate::runtime::mapper::flush_retired_views(state, host);
        if crate::observe::draw_log_enabled() {
            crate::observe::line(format!(
                "map_family op=DeleteIOSurfaceBacking2 ch={channel_id} object={object_id} task={task_id} plen={plen} mode={mode}"
            ));
        }
    } else if family == MapFamily::InvalidateResources {
        // RE: {task_id, count} + count×{object_id, 4×u8 validity ops}.
        // Ops (PVG host layout): clr_host, set_host, clr_guest, set_guest.
        // Pageon hardcodes LE 01 00 00 01 = clr hostValid + set guestValid.
        //
        // The same four bytes the EXEC_INDIRECT2 resource table carries,
        // through the same consumer: this producer's records are 8 bytes
        // and that one's are 24, but the quad is one contract and must
        // not acquire two meanings.
        use crate::runtime::decode::fifo::{
            decode_invalidate_resources, CHILD_INVALIDATE_PAGEON_FLAGS,
        };
        use crate::runtime::resource_validity::{apply, ValiditySite};
        match decode_invalidate_resources(&packet.payload) {
            Ok(cmd) => {
                let mut bumped = 0u32;
                let mut miss = 0u32;
                for rec in &cmd.records {
                    let outcome = apply(
                        state,
                        cmd.task_id,
                        rec.object_id,
                        rec.ops,
                        ValiditySite::InvalidateResources,
                    );
                    bumped = bumped.saturating_add(outcome.bumped);
                    if outcome.missed {
                        miss = miss.saturating_add(1);
                    }
                }
                // One counter here, two on the exec side: `pageBacking`
                // names mapping ids, so a record this device holds no
                // mapping for is already the surprising case. The exec
                // table names task object refs, most of which have no
                // surface state by construction.
                note_store_route_n("validity_miss_inv", miss as u64);
                let rec0 = cmd.records.first();
                let oid = rec0.map(|r| r.object_id).unwrap_or(0);
                let flags = rec0.map(|r| r.flags).unwrap_or(0);
                let ops = rec0.map(|r| r.ops).unwrap_or_default();
                let pageon = flags == CHILD_INVALIDATE_PAGEON_FLAGS;
                // ~11k/boot of routine guest cache-coherence ops. The
                // always-on rate is the `validity_*` family in the
                // per-second `store_routes` line; gate the per-op decode
                // detail so it does not bury the curated fail view. The
                // `decode_fail` and `inv_multi` paths below stay
                // fail-visible, and the guard also skips the format alloc
                // on a healthy boot.
                if crate::observe::draw_log_enabled() {
                    crate::observe::line(format!(
                    "map_family op=InvalidateResources opcode={:#x} ch={channel_id} plen={plen} task={} count={} oid={oid:#x} flags={flags:#x} clr_h={} set_h={} clr_g={} set_g={} pageon={pageon} bumped={bumped} miss={miss}",
                    packet.opcode,
                    cmd.task_id,
                    cmd.count,
                    ops.clear_host_valid,
                    ops.set_host_valid,
                    ops.clear_guest_valid,
                    ops.set_guest_valid
                ));
                }
                if cmd.count > 1 {
                    let ids: Vec<String> = cmd
                        .records
                        .iter()
                        .map(|r| {
                            format!(
                                "{:#x}:clr_h={}/set_g={}",
                                r.object_id, r.ops.clear_host_valid, r.ops.set_guest_valid
                            )
                        })
                        .collect();
                    crate::observe::fail(format!(
                        "inv_multi ch={channel_id} task={} n={} recs=[{}]",
                        cmd.task_id,
                        cmd.count,
                        ids.join(",")
                    ));
                }
            }
            // A refused Invalidate leaves this device serving
            // host-cached pixels for a resource the guest has just
            // CPU-written, so the line has to say which check refused
            // and not only that one did.
            Err(e) => note_resource_list_decode_fail(
                "InvalidateResources",
                packet.opcode,
                channel_id,
                e,
            ),
        }
    } else if matches!(
        family,
        MapFamily::SynchronizeResources | MapFamily::SynchronizeAndDiscardResources
    ) {
        // Two opcodes, one arm, because the reference host validates
        // both with byte-for-byte the same check — `{u32 task, u32
        // count}` then `count` 4-byte object ids — and the synchronise
        // obligation they name is the same one. `0x3e` adds a discard of
        // the named resources' contents on top, which this device does
        // not act on and reports below.
        //
        // The command carries `{task, count}` and a list of object ids,
        // and nothing else — no region, no direction. It is the guest
        // saying "I am about to touch these resources with the CPU", so
        // what this device owes is that every guest-page write it has
        // already submitted has executed, and nothing more. It is not a
        // request to invent pixels into guest pages.
        //
        // **It is also the contract's only host-to-guest copy trigger,
        // and a driven x86/Vulkan Safari-drag boot issues none at all.**
        // That is what makes this device's per-Store writeback pure
        // surplus on that workload; see `runtime::render_writeback`'s
        // module doc for what the surplus costs and what the reference
        // host does instead. A boot where this arm starts firing is a
        // boot where the deferral described there has a real land point,
        // so the count is worth watching rather than assuming zero.
        use crate::runtime::decode::fifo::decode_synchronize_resources;
        match decode_synchronize_resources(&packet.payload) {
            Ok(cmd) => {
                crate::runtime::render_writeback::settle_guest_writes(
                    crate::runtime::render_writeback::SettleSite::ChildStamp,
                );
                let oid = cmd.object_ids.first().copied().unwrap_or(0);
                if crate::observe::draw_log_enabled() {
                    crate::observe::line(format!(
                        "map_family op={name} opcode={:#x} ch={channel_id} plen={plen} task={} count={} oid={oid:#x}",
                        packet.opcode, cmd.task_id, cmd.count
                    ));
                }
                if cmd.count > 1 {
                    let ids: Vec<String> =
                        cmd.object_ids.iter().map(|id| format!("{id:#x}")).collect();
                    crate::observe::fail(format!(
                        "sync_multi ch={channel_id} task={} n={} oids=[{}]",
                        cmd.task_id,
                        cmd.count,
                        ids.join(",")
                    ));
                }
            }
            // A refused Synchronize lets the guest CPU-read pages whose
            // submitted writeback has not executed, which is a stale or
            // black frame the guest has no way to notice.
            Err(e) => note_resource_list_decode_fail(name, packet.opcode, channel_id, e),
        }
        // The half of `0x3e` that is not the synchronise. Reported after
        // the synchronise has run, so the record cannot be read as the
        // whole command having been dropped.
        if family == MapFamily::SynchronizeAndDiscardResources {
            note_unimplemented(
                state,
                channel_id,
                UnimplementedCommand::SynchronizeAndDiscardResources,
                packet,
            );
        }
    } else {
        let w0 = if plen >= 4 {
            crate::contract::endian::ld32(&packet.payload[0..])
        } else {
            0
        };
        let w1 = if plen >= 8 {
            crate::contract::endian::ld32(&packet.payload[4..])
        } else {
            0
        };
        crate::observe::off(format!(
            "map_family op={name} opcode={:#x} ch={channel_id} plen={plen} w0={w0:#x} w1={w1:#x}",
            packet.opcode
        ));
    }
}


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChildPacketDisposition {
    Complete,
    Deferred,
}

/// One line per `ExecIndirect2` packet, naming everything the packet executed.
///
/// # Do not read these counters as a census
///
/// On the always-on sink this line is **failure-selected**: the caller emits it
/// through `observe::fail` only when `packet_failed`, and sends healthy packets
/// to the verbose-gated `observe::line` instead. So every copy of this line in
/// `/tmp/reims-vgpu-fail.log` is, by construction, a packet that failed a draw
/// or an ICB — which is why the accumulated log shows `draws_ok=0 draws_fail=1`
/// on all 414 of them. That ratio is the filter, not the draw path.
///
/// The same trap hides the ICB rail. `icb_ok=0` across those lines does **not**
/// mean the guest never runs indirect command buffers; a packet whose ICBs all
/// succeeded is exactly a packet that did not fail, so it never reached this
/// sink. Answering "does ICB run at all?" needs `REIMS_VGPU_DRAW_LOG`, or a
/// `note_store_route` counter that is not conditioned on failure.
fn exec_summary(channel_id: u32, result: &crate::runtime::exec::ExecResult, plen: usize) -> String {
    format!(
        "exec_indirect2 ch={channel_id} task={} streams={} saw_draw={} clears={} draws_ok={} draws_fail={} rt_resolves={} guest_stores={} icb_ok={} icb_fail={} compute_ctrl_fail={} compute_icb_fail={} render_unbinds={}/{}/{} total_us={} plen={plen}",
        result.task_id,
        result.streams_loaded,
        result.saw_draw as u8,
        result.clears_applied,
        result.metal_draws_ok,
        result.metal_draws_fail,
        result.render_attachment_resolves,
        result.render_guest_stores,
        result.render_icb_ok,
        result.render_icb_fail,
        result.compute_control_fail,
        result.compute_icb_fail,
        result.buffer_unbinds,
        result.texture_unbinds,
        result.sampler_unbinds,
        result.total_us,
    )
}

/// A synchronous ExecIndirect2 holding `DeviceInner` for this long starves the
/// guest's read-to-clear completion/status registers. This is a diagnostic
/// proxy only; it never changes packet ordering or completion behavior.
const SYNC_EXEC_STALL_US: u64 = 250_000;

#[inline]
fn sync_exec_stalled(total_us: u64) -> bool {
    total_us >= SYNC_EXEC_STALL_US
}

fn process_child_packet<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    channel_id: u32,
    packet: &Packet,
) -> ChildPacketDisposition {
    match packet.opcode {
        CHILD_OP_DEFINE_TASK2 => {
            apply_define_task2(state, host, &packet.payload, Some(channel_id));
        }
        // A short SET_OBJECT_LIST leaves the task's object list unbound — every
        // type-11 texture/object resolve on it then fails
        // (object_list_count==0). Never on a well-formed boot.
        CHILD_OP_SET_OBJECT_LIST => {
            apply_set_object_list(state, &packet.payload, Some(channel_id));
        }
        CHILD_OP_DELETE_RESOURCE => {
            if !packet_short("delete_resource", Some(channel_id), packet.payload.len(), 8) {
                let task_id = ld32(&packet.payload[0..]);
                let id = ld32(&packet.payload[4..]);
                // Which references resolved through this object is not knowable
                // from the packet, so the task's resolutions go together.
                // Scoped to the reference the packet names, like every other
                // response to this opcode — `objects`, the host copies and
                // `texture_to_mapping` are all keyed `(task, ref)`. Retiring
                // the whole task here was the outlier and the device's largest
                // source of re-walks: 54 109 resolutions dropped on one driven
                // boot, 95% of every bind miss.
                note_bb_retired(
                    "bb_retire_delete_resource",
                    state.retire_bound_buffers_for_ref(task_id, id),
                );
                let _ = state.delete_object(task_id, id);
            }
        }
        // PVG CmdDeleteTask (0x20) on child channels too (was SMALL_ID alias only in decode).
        CHILD_OP_DELETE_TASK => {
            apply_delete_task(state, &packet.payload, Some(channel_id));
        }
        CHILD_OP_SETUP_SHARED_STATE => {
            apply_setup_shared_state(state, host, &packet.payload, Some(channel_id));
        }
        CHILD_OP_ONLINE_ACK => {
            state.display.online_acked = true;
            // The connectionChange-ack (process_online opcode 2) is believed to
            // echo the shared-descriptor `+0x200` token back to the host in its
            // payload. We consume the ack (online_acked)
            // but never inspect that token — capture it here (raw first words +
            // len, rare/once-per-online so no flood) so a bad boot records what
            // value the guest round-tripped. Measure-only.
            let w0 = if packet.payload.len() >= 4 {
                ld32(&packet.payload[0..])
            } else {
                0
            };
            let w1 = if packet.payload.len() >= 8 {
                ld32(&packet.payload[4..])
            } else {
                0
            };
            crate::observe::fail(format!(
                "display_online_ack index={} plen={} w0={:#x} w1={:#x}",
                state.display.display_index,
                packet.payload.len(),
                w0,
                w1
            ));
        }
        /*
         * Scanout policy:
         * - Early boot: front type-11 writebacks paint while !frame_flush_seen
         *   and job W×H matches established console (no mid-switch thrash).
         * - After first boundary: display presents paint (op8 DisplaySwap on
         *   arm ch4, **or** the op6/op7 transactions on x86 Ventura/Tahoe
         *   display ch5).
         * - Nothing on ch2 paints: 0x28 is CmdDeleteObject and 0x3b is
         *   CmdGetComputeInfo, and neither is a present however mid-composite
         *   the moment it arrives in.
         */
        // The three display present commands. op8 `CmdDisplaySwapMapping` is
        // the arm/EFI-era path; x86 Ventura/Tahoe drives the display pipe with
        // op6 `CmdDisplayTransaction2_DEPRECATED` and op7
        // `CmdDisplayTransaction3`. They differ only in where the surface word
        // sits, which `display_txn_trailer_slots` owns for all three.
        opcode @ (CHILD_OP_DISPLAY_SWAP | CHILD_OP_DISPLAY_TRANSACTION2 | CHILD_OP_DISPLAY_TRANSACTION3) => {
            note_display_txn_payload(state, channel_id, packet);
            let Some(mapping) = present_surface_id(opcode, &packet.payload) else {
                crate::observe::fail(format!(
                    "packet_short reason=display_present_short ch={channel_id} op={opcode:#x} \
                     plen={} need={}",
                    packet.payload.len(),
                    display_txn_trailer_len(opcode)
                ));
                return ChildPacketDisposition::Complete;
            };
            // Per-present decode census (~30k/session under animation); the
            // present rate lives in the present_proxy summary, so gate the
            // per-packet line behind REIMS_VGPU_DRAW_LOG. `pipe` is the display
            // index for op8 and the pipe index for op6/7 — payload word 0 in
            // both. `task` is op6/7's task field, which is the submitting task's
            // and not a completion stamp; the packet's own stamp lives in the
            // FIFO header. op8 has no such word, and prints `-`.
            let (_, task_slot) = display_txn_trailer_slots(opcode);
            let task = task_slot
                .map(|slot| format!("{:#x}", ld32(&packet.payload[slot * 4..])))
                .unwrap_or_else(|| "-".to_string());
            crate::observe::line(format!(
                "present_txn op={opcode:#x} ch={channel_id} pipe={} sid={mapping} task={task} \
                 plen={} unpainted={} prior_present_mapping={}",
                ld32(&packet.payload[0..]),
                packet.payload.len(),
                state.present.unpainted_presents,
                state.present.present_mapping
            ));
            if present_named_mapping(state, host, channel_id, mapping)
                == ChildPacketDisposition::Deferred
            {
                return ChildPacketDisposition::Deferred;
            }
        }
        // `CmdDeleteObject`, the guest retiring one serializer-allocated object.
        // This device decodes the record and declines it — and this is *not* the
        // arm that retires an entry from the object table, which is
        // `CHILD_OP_DELETE_RESOURCE` two slots down. The two look alike and name
        // different namespaces; `apply_delete_object`'s doc has the boot that
        // measured the difference.
        //
        // This arm used to be a silent no-op named for a present, on a reading
        // that had it painting frames. It never painted and it never spoke, so a
        // real teardown command was arriving and being dropped without a line.
        CHILD_OP_DELETE_OBJECT => {
            // The payload is `{u32 task}` then a self-describing record whose
            // own byte length sits at offset 8. Both bounds are checked before
            // declining, so a corrupt packet is reported as corrupt rather than
            // as a command this device merely has not implemented — those are
            // different problems and only one of them is closed by writing a
            // handler.
            let plen = packet.payload.len();
            if !packet_short("delete_object", Some(channel_id), plen, 12) {
                let record_len = ld32(&packet.payload[8..]) as usize;
                // The record starts one word in, so it is `record_len + 4` that
                // has to fit. Saturating, because the guest's length is an
                // arbitrary `u32` and the sum is what overflows.
                if !packet_short(
                    "delete_object_record",
                    Some(channel_id),
                    plen,
                    record_len.saturating_add(4),
                ) {
                    apply_delete_object(state, channel_id, &packet.payload, packet);
                }
            }
        }
        // `CmdDebug`, a host-side trace marker. Nothing is owed to the guest, but
        // the payload is what the guest wanted recorded, so the record echoes it
        // rather than swallowing the command.
        CHILD_OP_DEBUG => {
            note_unimplemented(state, channel_id, UnimplementedCommand::Debug, packet);
        }
        // The display-state pair. Both carry a payload floor of 8 bytes, and
        // neither moves anything in this device's display model: sleep state is
        // not tracked and no display property is applied. A guest that sleeps a
        // panel and finds it still lit is looking at this arm.
        CHILD_OP_DISPLAY_SLEEP_STATE => {
            note_unimplemented(
                state,
                channel_id,
                UnimplementedCommand::DisplaySleepState,
                packet,
            );
        }
        CHILD_OP_DISPLAY_SET_PROPERTIES => {
            note_unimplemented(
                state,
                channel_id,
                UnimplementedCommand::DisplaySetProperties,
                packet,
            );
        }
        // `CmdDelay`: the guest asking the channel to be held before the next
        // command runs. This device continues immediately. That reorders nothing
        // — the stamps still retire in submission order — but a guest that used
        // the delay to let something settle does not get it.
        CHILD_OP_DELAY => {
            note_unimplemented(state, channel_id, UnimplementedCommand::Delay, packet);
        }
        // PVG / Monterey: 0x3b = CmdGetComputeInfo (query). Must write reply
        // before stamp or createComputePipeline stalls (texture-ref 29-06-26).
        // `present-frame-flush` is the recovered legacy name for the same wire
        // opcode, and it is wrong.
        CHILD_OP_GET_COMPUTE_INFO => {
            if packet.payload.len() >= 24 {
                let _ = reply_compute_info(state, host, &packet.payload);
            } else {
                crate::observe::fail(format!(
                    "get_compute_info short ch={channel_id} len={}",
                    packet.payload.len()
                ));
            }
        }
        CHILD_OP_CURSOR_SHOW => {
            if !packet_short("cursor_show", Some(channel_id), packet.payload.len(), 8) {
                let show = ld32(&packet.payload[4..]) != 0;
                state.cursor.show = show;
                sample_cursor_position(state, host);
                host.enqueue(HostAction::cursor(state.cursor.x, state.cursor.y, show));
            }
        }
        CHILD_OP_CURSOR_GLYPH => {
            if load_cursor_glyph(state, host, packet) {
                host.enqueue(HostAction::cursor_glyph());
                host.enqueue(HostAction::cursor(
                    state.cursor.x,
                    state.cursor.y,
                    state.cursor.show,
                ));
            }
        }
        CHILD_OP_EXEC_INDIRECT2 => {
            if packet.payload.len() < 12 {
                state.record_fail(FailEvent::UnsupportedExec {
                    channel: channel_id,
                    fault: ExecFault::Indirect2Short,
                });
            } else {
                // Process this channel's exec packet. Archive does not drain
                // other child FIFOs here; surface RAW is render_wait_surface on
                // the specific type-11/GVA key at sample/Load/swap sites.
                let result =
                    crate::runtime::exec::process_exec_indirect2(state, host, &packet.payload);
                let channel_bit = 1u32.checked_shl(channel_id).unwrap_or(0);
                if result.deferred {
                    if channel_bit != 0 && state.translation_deferred_mask & channel_bit == 0 {
                        state.translation_deferred_mask |= channel_bit;
                        // Census for the same reason as `translation_order_hold`:
                        // the packet is NOT consumed (`Deferred` leaves it at the
                        // FIFO head to be retried), and the matching
                        // `exec_translation_ready` below is already `off`. Boot 87:
                        // 55 deferrals, 56 readies.
                        crate::observe::off(format!(
                            "exec_translation_deferred reason=air_loading ch={channel_id} task={} pending_mask={:#x}",
                            result.task_id, state.translation_deferred_mask
                        ));
                    }
                    return ChildPacketDisposition::Deferred;
                }
                if channel_bit != 0 && state.translation_deferred_mask & channel_bit != 0 {
                    state.translation_deferred_mask &= !channel_bit;
                    crate::observe::off(format!(
                        "exec_translation_ready ch={channel_id} task={} pending_mask={:#x}",
                        result.task_id, state.translation_deferred_mask
                    ));
                }
                // Failure-carrying packets keep the full per-packet line on the
                // always-on sink (context for the per-site reason=<slug> lines).
                // Healthy packets are expected control flow and stay quiet
                // unless the draw log is on — the per-packet form ran ~1k
                // lines/s under Safari scroll.
                let packet_failed = result.metal_draws_fail > 0
                    || result.render_icb_fail > 0
                    || result.compute_control_fail > 0
                    || result.compute_icb_fail > 0;
                if packet_failed {
                    crate::observe::fail(exec_summary(channel_id, &result, packet.payload.len()));
                } else if crate::observe::draw_log_enabled() {
                    crate::observe::line(exec_summary(channel_id, &result, packet.payload.len()));
                }
                if sync_exec_stalled(result.total_us) {
                    crate::observe::fail(format!(
                        "TRANSPORT reason=sync_exec_lock_hold ch={channel_id} task={} total_us={} draws={} rt_resolves={} guest_stores={} threshold_us={SYNC_EXEC_STALL_US}",
                        result.task_id,
                        result.total_us,
                        result.metal_draws_ok.saturating_add(result.metal_draws_fail),
                        result.render_attachment_resolves,
                        result.render_guest_stores
                    ));
                }
            }
        }
        CHILD_OP_HEAP_TEXTURE_SIZE_AND_ALIGN => {
            let _ = reply_heap_texture_size_and_align(state, host, &packet.payload);
        }
        // The one packet that says a cached page list has gone stale, and the
        // only arm that consumes it. A second handler used to sit inside the
        // stamp-and-forget family below, behind an `else if` on this opcode that
        // the family's own pattern list does not include — structurally
        // unreachable, and the richer of the two: it routed the object id
        // through `texture_to_mapping`, condemned with a page fingerprint rather
        // than clearing, and took the deferred windows. Everything it did lives
        // in `objects::replace_physical` now, so the packet has one meaning
        // again. `mapping_page_drift` reporting "the guest re-pointed this
        // surface and no packet said so" is what a lost half of it looks like
        // from the far end.
        CHILD_OP_REPLACE_PHYSICAL => {
            if !packet_short(
                "replace_physical",
                Some(channel_id),
                packet.payload.len(),
                crate::runtime::decode::fifo::CHILD_REPLACE_PHYSICAL_LEN as usize,
            ) {
                if let Some(cmd) =
                    crate::runtime::decode::fifo::decode_replace_physical(&packet.payload)
                {
                    // The GPA behind this object's GVAs changes here, so any
                    // held bind resolution on the task names the old frames.
                    // The packet names an object, not a range, and which
                    // references resolved through it is not recorded — so the
                    // task's resolutions go together.
                    note_bb_retired(
                        "bb_retire_replace_physical",
                        state.retire_bound_buffers_for_task(cmd.task_id),
                    );
                    crate::runtime::objects::replace_physical(
                        state,
                        host,
                        cmd.task_id,
                        cmd.object_id,
                    );
                }
            }
        }
        // The guest's memory-lifecycle family. All five share this arm because
        // they share one obligation — say what the device owes the guest *before*
        // the guest touches those pages itself — and each discharges it
        // differently in the branches below:
        //
        // - `MapMemory2` / `UnmapMemory` retire the host views that alias the
        //   pages whose PTEs just changed, and land any deferred render-Store
        //   window overlapping the range cache-only.
        // - `InvalidateResources` applies the guest's validity quad through the
        //   same consumer the EXEC_INDIRECT2 resource table uses.
        // - `SynchronizeResources` lands every deferred writeback for the named
        //   objects: the guest is about to CPU-read them, and this is the only
        //   host-visible choke point before it does.
        // - `DeleteIOSurfaceBacking2` flushes the retired views at the backing's
        //   own lifetime boundary.
        //
        // None of them writes a page the guest did not ask for. That restraint
        // is the recurring finding here, stated at each branch: what looks like
        // a missing implementation is usually the device declining to invent.
        CHILD_OP_UNMAP_MEMORY => {
            apply_map_family(state, host, channel_id, packet, MapFamily::UnmapMemory);
        }
        CHILD_OP_MAP_MEMORY2 => {
            apply_map_family(state, host, channel_id, packet, MapFamily::MapMemory2);
        }
        CHILD_OP_INVALIDATE_RESOURCES => {
            apply_map_family(state, host, channel_id, packet, MapFamily::InvalidateResources);
        }
        CHILD_OP_SYNCHRONIZE_RESOURCES => {
            apply_map_family(state, host, channel_id, packet, MapFamily::SynchronizeResources);
        }
        CHILD_OP_SYNCHRONIZE_AND_DISCARD_RESOURCES => {
            apply_map_family(
                state,
                host,
                channel_id,
                packet,
                MapFamily::SynchronizeAndDiscardResources,
            );
        }
        CHILD_OP_DELETE_IOSURFACE_BACKING2 => {
            apply_map_family(
                state,
                host,
                channel_id,
                packet,
                MapFamily::DeleteIOSurfaceBacking2,
            );
        }
        // `CmdDiscardResources`: the discard half of `0x3e` on its own, with the
        // same payload contract, and nothing owed to the guest — a discard this
        // device ignores costs memory and never correctness.
        //
        // The payload is still decoded, with `0x35`'s decoder because the host
        // validates all three with the same check. A malformed one is worth a
        // line even though a well-formed one would have been ignored: it says
        // the guest and this device disagree about the record layout, which the
        // two commands that *do* act on it share.
        CHILD_OP_DISCARD_RESOURCES => {
            use crate::runtime::decode::fifo::decode_synchronize_resources;
            match decode_synchronize_resources(&packet.payload) {
                Ok(_) => note_unimplemented(
                    state,
                    channel_id,
                    UnimplementedCommand::DiscardResources,
                    packet,
                ),
                Err(e) => {
                    note_resource_list_decode_fail("DiscardResources", packet.opcode, channel_id, e)
                }
            }
        }
        // `CmdNOP`, a fence with no payload. The guest emits it from a present's
        // failure and teardown legs to order work it is abandoning, and retiring
        // its stamps — which the drain does for every accepted packet — is the
        // whole contract. Named so it stops being reported as an unknown opcode.
        //
        // The route is keyed on the command, not on that use. Naming it for the
        // channel flush the display pipe happens to want is how the constant
        // itself came to be misnamed, and it leaves a reader grepping the log
        // for the command with nothing under it.
        CHILD_OP_NOP => {
            crate::runtime::drain::note_store_route("child_nop");
            // The command allocates no bytes, so payload is the one thing that
            // can falsify this reading. Bytes here would mean the command grew a
            // form this arm does not decode, and dropping them silently is what
            // the unknown-opcode arm was at least loud about.
            if !packet.payload.is_empty() {
                crate::observe::Emit::decline(
                    "child_nop",
                    &NopCarriedPayload {
                        channel: channel_id,
                        plen: packet.payload.len(),
                    },
                )
                .fail();
            }
        }
        // The reference host's retired slots. Its shared handler accepts the
        // packet, does nothing with the payload and retires the stamps, which is
        // exactly what this arm does — so matching them is fidelity, and the
        // record exists only to say a guest is still emitting one.
        op if is_deprecated_child_opcode(op) => {
            note_unimplemented(state, channel_id, UnimplementedCommand::Deprecated, packet);
        }
        // Everything left is an opcode with no handler at all. Two different
        // things end up here and the record has to keep them apart: an opcode
        // inside `CHILD_OP_MAX` is an unassigned slot — a guest asking for a
        // command this host generation does not have — while one above it is a
        // value the reference host refuses before it ever reaches a table, so a
        // packet carrying it is a corrupt header or a desynced ring and not a
        // missing feature.
        _ => {
            if packet.opcode > CHILD_OP_MAX {
                crate::observe::fail(format!(
                    "child_opcode_out_of_range ch={channel_id} opcode={:#x} max={CHILD_OP_MAX:#x} \
                     total_size={} stamps={} plen={} (above the dispatch ceiling: this is a \
                     malformed header or a desynced ring, not an unimplemented command)",
                    packet.opcode,
                    packet.total_size,
                    packet.stamp_count(),
                    packet.payload.len()
                ));
            }
            state.record_fail(FailEvent::UnknownChildOpcode {
                channel: channel_id,
                opcode: packet.opcode,
                total_size: packet.total_size,
                stamp_count: packet.stamp_count(),
                payload: packet.payload.clone(),
            });
        }
    }
    ChildPacketDisposition::Complete
}

/// Record a command this device names but does not execute.
///
/// One call site per command, and one place the record is built: the fields are
/// the packet's, so a second spelling would be a second chance to report the
/// wrong opcode beside the right command name.
///
/// **A line once per opcode, and a count every window.** These are not rare
/// events — `CmdDeleteObject` arrives from the guest's shared-object allocator
/// mid-composite, so an unlatched line per packet would be a per-frame flood in
/// the always-on log, and a flood is how a real refusal stops being read. The
/// latch costs the rate, which is why the route counter goes up first: the
/// emitter dedupes and the counter does not, and the two answer different
/// questions.
fn note_unimplemented(
    state: &mut DeviceState,
    channel_id: u32,
    command: UnimplementedCommand,
    packet: &Packet,
) {
    // Keyed by the command's own slug, so the counter and the `reason=` a reader
    // greps for are one string rather than two that can drift apart.
    note_store_route(command.slug());
    let ev = FailEvent::UnimplementedChildCommand {
        channel: channel_id,
        command,
        opcode: packet.opcode,
        total_size: packet.total_size,
        stamp_count: packet.stamp_count(),
        payload: packet.payload.clone(),
    };
    // Per opcode and not per command: the deprecated set is fifteen numbers
    // behind one command name, and which of them a guest is still emitting is
    // the whole content of the record.
    state.record_fail_once(ev, u64::from(packet.opcode));
}

/// Drain one child channel.
pub fn drain_child_fifo<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    channel_id: u32,
) {
    if state.gfx.root_page == 0 {
        return;
    }
    // `child_reg_block_offset` is the channel-id bound: it is `None` for exactly
    // the ids `is_child_channel` refuses, so re-testing them here would be the
    // same rule twice one line apart.
    let Some(regs_off) = child_reg_block_offset(channel_id) else {
        return;
    };
    let regs_gpa = state.pfn_gpa(state.gfx.root_page) + regs_off;

    let mut head = match crate::runtime::host::read_u32(host, regs_gpa + CHILD_REG_HEAD) {
        Ok(v) => v,
        Err(_) => {
            state.record_fail(FailEvent::MalformedChildPacket {
                channel: channel_id,
                fault: PacketFault::ChildRegsHeadRead,
                head: 0,
            });
            return;
        }
    };
    let stamp_index = match crate::runtime::host::read_u32(host, regs_gpa + CHILD_REG_STAMP_INDEX) {
        Ok(v) => v,
        Err(_) => {
            state.record_fail(FailEvent::MalformedChildPacket {
                channel: channel_id,
                fault: PacketFault::ChildRegsStampRead,
                head,
            });
            return;
        }
    };
    let base_pfn = match crate::runtime::host::read_u32(host, regs_gpa + CHILD_REG_BASE_PFN) {
        Ok(v) => v,
        Err(_) => {
            state.record_fail(FailEvent::MalformedChildPacket {
                channel: channel_id,
                fault: PacketFault::ChildRegsBaseRead,
                head,
            });
            return;
        }
    };

    let ring_length = ensure_child_ring(state, host, channel_id, base_pfn);
    if ring_length == 0 {
        return;
    }
    let page_gpas = state.child_rings[channel_id as usize].page_gpas.clone();

    // Nested drain_other must skip this channel (no re-enter head).
    // Use a bit mask so nested drains skip the full stack, not only the leaf.
    let prev_channel = state.draining_channel;
    let bit = 1u32 << channel_id;
    state.draining_channel = channel_id;
    state.draining_mask |= bit;
    // Cleared on entry and set only by the arm that stops on an unmet stamp
    // wait, so the bit always describes this drain rather than an older one.
    // See `drain_main_fifo`'s copy for what a sticky bit would cost.
    state.stamp_deferred_mask &= !bit;

    loop {
        let tail = match crate::runtime::host::read_u32(host, regs_gpa + CHILD_REG_TAIL) {
            Ok(v) => v,
            Err(_) => {
                state.record_fail(FailEvent::MalformedChildPacket {
                    channel: channel_id,
                    fault: PacketFault::ChildTailRead,
                    head,
                });
                break;
            }
        };
        if head == tail {
            break;
        }
        let Some(available) = published_byte_count(head, tail, ring_length) else {
            state.record_fail(FailEvent::MalformedChildPacket {
                channel: channel_id,
                fault: PacketFault::DesyncedHeadTail,
                head,
            });
            break;
        };
        if available < PACKET_HEADER_LEN {
            break;
        }
        let header = match read_child_ring_bytes(
            host,
            &page_gpas,
            ring_length,
            head,
            PACKET_HEADER_LEN,
            state.page_shift,
        ) {
            Ok(h) => h,
            Err(_) => {
                state.record_fail(FailEvent::MalformedChildPacket {
                    channel: channel_id,
                    fault: PacketFault::ChildHeaderRead,
                    head,
                });
                break;
            }
        };
        let snap_len = packet_snapshot_len(&header, available, ring_length);
        let snap = match read_child_ring_bytes(
            host,
            &page_gpas,
            ring_length,
            head,
            snap_len,
            state.page_shift,
        ) {
            Ok(s) => s,
            Err(_) => {
                state.record_fail(FailEvent::MalformedChildPacket {
                    channel: channel_id,
                    fault: PacketFault::ChildSnapRead,
                    head,
                });
                break;
            }
        };
        // Entry gate before decode of full payload: hold CmdDisplaySwap when
        // host paint is already two presents behind (apple-gfx pending_frames
        // >= 2). Leave head unmoved so body draws on other channels can still
        // land via drain_other; re-enter after note_present_paint_consumed.
        let peek_opcode = ld16(&header[PACKET_OPCODE..]);
        if matches!(
            peek_opcode,
            CHILD_OP_DISPLAY_SWAP | CHILD_OP_DISPLAY_TRANSACTION2 | CHILD_OP_DISPLAY_TRANSACTION3
        ) && state.present.unpainted_presents >= MAX_UNPAINTED_PRESENTS
        {
            note_present_backpressure_hold(state, channel_id, head, tail);
            // Paint will schedule the next worker slice. Preserve this channel
            // without self-waking the worker ahead of QEMU's action BH.
            state.pending.child_mask |= bit;
            break;
        }
        match decode_packet(&snap, head, available, ring_length) {
            Ok(packet) => {
                if note_packet_stamp_waits(state, host, Some(channel_id), &packet)
                    == StampVerdict::Hold
                {
                    // The guest ordered this packet behind work that has not
                    // reached its stamp. Hold it: head and completion stamp stay
                    // untouched, so the retry re-decodes the same bytes and no
                    // side effect can land twice. Never block — the awaited
                    // stamp is published by *another* timeline's drain, and this
                    // one is single-threaded, so waiting here would deadlock
                    // against the thing being waited for.
                    note_store_route("packet_stamp_wait_held");
                    state.stamp_deferred_mask |= bit;
                    state.pending.child_mask |= bit;
                    break;
                }
                if process_child_packet(state, host, channel_id, &packet)
                    == ChildPacketDisposition::Deferred
                {
                    // Translation owns only immutable AIR bytes. Keep head and
                    // stamp untouched so retry cannot duplicate any packet
                    // side effect; continue with sibling channels in the
                    // outer pending-drain loop.
                    break;
                }
                head = packet.next_head;
                if gpa_map::write_u32(
                    host,
                    regs_gpa + CHILD_REG_HEAD,
                    head,
                    state.page_size() as usize,
                )
                .is_err()
                {
                    // The packet was processed + stamped, but the consumer
                    // pointer never advanced: the next drain re-reads the stale
                    // head and RE-EXECUTES the same packets. Fail-visible so
                    // that silent replay is diagnosable (drain.rs Rank-1 audit).
                    state.record_fail(FailEvent::MalformedChildPacket {
                        channel: channel_id,
                        fault: PacketFault::ChildHeadWriteback,
                        head,
                    });
                }

                // Completion stamp. Execution is sync-per-packet: the packet's
                // work is done by the time control reaches here, so the stamp is
                // owed now. DisplaySwap included — PGDisplay present completion
                // follows the +0x188 retain, not host encode/paint.
                //
                // The archive orders stamps through a per-channel queue because
                // its draw jobs complete asynchronously (`ApplePVGPUDrawJob`,
                // `apple_pv_gpu_render_wait_surface`). If this device ever grows
                // an async execution path, that ordering has to come back with
                // it — and be written against the async model that then exists,
                // not inherited from an empty queue.
                write_stamp(state, host, stamp_index, packet.completion_stamp);
                if state.pending.host_action_yield {
                    if head != tail {
                        state.pending.child_mask |= bit;
                    }
                    break;
                }
            }
            Err(err) => {
                if let Some(fault) = err.fault() {
                    state.record_fail(FailEvent::MalformedChildPacket {
                        channel: channel_id,
                        fault,
                        head,
                    });
                }
                break;
            }
        }
    }

    state.draining_mask &= !bit;
    state.draining_channel = prev_channel;
}

/// Drain iosfc mapper producer→consumer handshake.
///
/// Prefer calling this on the **iosfc producer MMIO path** (publishing vCPU)
/// so `resolve_mapping_backing` KVA walks use `current_cpu`. BH-only resolve
/// with `cpu_memory_rw_debug(first_cpu)` deadlocks against MMIO holding
/// `DEVICES` (see reims-vgpu-mmio.c `read_kva`).
pub fn drain_iosfc<H: HostMemory + HostOps>(state: &mut DeviceState, host: &mut H) {
    let producer = state.iosfc.producer;
    let mut consumer = state.iosfc.consumer;
    if producer == consumer {
        state.pending.iosfc = false;
        return;
    }

    // Process requests between consumer and producer when ring is programmed.
    if state.iosfc.ring_base != 0 && producer > consumer {
        let start = consumer;
        let end = producer;
        for idx in start..end {
            let entry_off = (idx as u64) * MAPPER_REQUEST_ENTRY_LEN as u64;
            let mut e = [0u8; MAPPER_REQUEST_ENTRY_LEN];
            if host
                .read_gpa(state.iosfc.ring_base + entry_off, &mut e)
                .is_err()
            {
                break;
            }
            let rtype = ld32(&e[MAPPER_REQUEST_TYPE..]);
            let mapping_id = ld32(&e[MAPPER_REQUEST_MAPPING_ID..]);
            // Capture was taken at producer write for published entry (idx+1).
            let cap = match state.mapper_capture {
                Some(c) if c.producer == idx + 1 => state.mapper_capture.take(),
                _ => None,
            };
            match rtype {
                MAPPER_REQUEST_MAP => {
                    let _ = state.map_surface(mapping_id);
                    if let Some(c) = cap {
                        if c.request_type == MAPPER_REQUEST_MAP {
                            let _ = crate::runtime::mapper::apply_capture(state, &c, mapping_id);
                            // Eager page-table + device-desc geometry when KVA works.
                            let _ = crate::runtime::mapper::resolve_mapping_backing(
                                state, host, mapping_id,
                            );
                        } else {
                            // Mismatched capture — put back for a later entry.
                            state.mapper_capture = Some(c);
                        }
                    }
                }
                MAPPER_REQUEST_UNMAP => {
                    // Deferred-writeback: DROP, never write — same recycled-page
                    // hazard as DeleteIOSurfaceBacking2 (the unmap request
                    // trails the guest release; writing risks PTE corruption).
                    if let Some(c) = cap {
                        if c.request_type == MAPPER_REQUEST_UNMAP {
                            let _ = crate::runtime::mapper::apply_capture(state, &c, mapping_id);
                        } else {
                            state.mapper_capture = Some(c);
                            let _ = state.unmap_surface(mapping_id);
                        }
                    } else {
                        let _ = state.unmap_surface(mapping_id);
                    }
                }
                _ => {
                    if let Some(c) = cap {
                        state.mapper_capture = Some(c);
                    }
                    // Unknown mapper request: fail-visible, still advance. A
                    // mapper ring entry is not a FIFO packet — it carries no
                    // stamps and no payload span — so the two packet-shaped
                    // fields report the absence rather than borrowing the
                    // entry's bytes and implying a framing it does not have.
                    state.record_fail(FailEvent::UnknownChildOpcode {
                        channel: 0,
                        opcode: rtype as u16,
                        total_size: MAPPER_REQUEST_ENTRY_LEN as u32,
                        stamp_count: 0,
                        payload: Vec::new(),
                    });
                }
            }
            consumer = idx.wrapping_add(1);
        }
    } else {
        // No ring base: still catch consumer up (boot handshake).
        consumer = producer;
    }

    state.iosfc.consumer = consumer;
    if state.iosfc.consumer == state.iosfc.producer {
        host.enqueue(HostAction::irq_iosfc());
    }
    state.pending.iosfc = false;
}

/// Display-side present completion (PGDisplay `_presentMappedSurface`
/// completion block, live PVG binary RE in): after
/// `presentFrame` retains the surface into `+0x188`, the block sets pending
/// bit 1 on the display shared page, reads the enable mask, and pokes the
/// display IRQ when the guest asked for present notifications. This is the
/// guest's frame-done pacing edge — separate from the packet header stamp
/// (the swap fence). Without it the guest keeps swapping (fence releases)
/// but never receives the per-present display event.
pub fn signal_display_present_complete<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
) {
    let gpa = state.display.shared_gpa;
    if gpa == 0 {
        return;
    }
    let mut mask_le = [0u8; 4];
    if host
        .read_gpa(gpa + DISPLAY_SHARED_ENABLE_MASK, &mut mask_le)
        .is_err()
    {
        return;
    }
    let mask = ld32(&mask_le);
    // Pending word is atomic read-and-clear (ldclral) on the guest side; OR
    // the present bit so a not-yet-consumed ONLINE event is preserved.
    let mut pending_le = [0u8; 4];
    let pending = if host
        .read_gpa(gpa + DISPLAY_SHARED_PENDING, &mut pending_le)
        .is_ok()
    {
        ld32(&pending_le)
    } else {
        0
    };
    // A bit2 (ONLINE) still pending *after* online was acked is stale: the guest
    // already consumed that online event (`online_acked`), so re-delivering it
    // makes `signalDisplay` re-run process_online → connectionChange → a
    // boot-progress overlay rebuild (the host-driven strobe).
    // Preserving bit2 via the `pending |` write is only correct *pre-ack*; drop
    // it once acked so we don't hand the guest a redundant online. `stale` is 0
    // on healthy boots (bit2 clears at ack), so this is a no-op there — it only
    // suppresses the intermittent try_display_online/ack race leftover. A fresh
    // legitimate online (after a reinit) clears `online_acked` first, so it is
    // never masked here. Still logged (measure + fix together).
    let stale = state.display.online_acked && pending & DISPLAY_ONLINE_EVENT_MASK != 0;
    let base = if stale {
        pending & !DISPLAY_ONLINE_EVENT_MASK
    } else {
        pending
    };
    shared_w32(
        host,
        gpa,
        DISPLAY_SHARED_PENDING,
        base | DISPLAY_PRESENT_EVENT_MASK,
        state.page_size() as usize,
    );
    if stale {
        crate::runtime::census::present_proxy::note_stale_online_pending("present", pending);
    }
    if mask & DISPLAY_PRESENT_EVENT_MASK != 0 {
        let bit = 1u32 << (state.display.display_index & 0x1f);
        state
            .gfx
            .interrupt_status_disp
            .fetch_or(bit, std::sync::atomic::Ordering::AcqRel);
        host.enqueue(HostAction::irq_gfx());
    }
}

/// Minimum wall-clock interval shared by both display VBL signal paths, in
/// microseconds.
///
/// The x86 QEMU heartbeat oversamples this interval every
/// `REIMS_VGPU_PCI_HEARTBEAT_MS` (4 ms). The shared limiter caps heartbeat and
/// active-console polls at the rate we advertise, without aliasing a
/// heartbeat-only workload down to half rate.
///
/// **Derived from [`DISPLAY_REFRESH_HZ`], not written down.** This was a
/// millisecond grid with a hardcoded `8`, which is 125 Hz — so the device
/// advertised 120 Hz in its timing table and then delivered VBL 4.2% faster.
/// The guest honours what is delivered, not what is advertised: a driven
/// Safari measured its own `requestAnimationFrame` at exactly 125 Hz. 120 Hz is
/// 8333 µs and is simply not expressible on an integer-millisecond grid, so the
/// units are part of the fix rather than incidental to it.
pub(crate) const DISPLAY_VBL_MIN_INTERVAL_US: u64 =
    1_000_000 / crate::model::DISPLAY_REFRESH_HZ as u64;

/// Atomically claim the next display VBL for either the locked or lock-free
/// poll path. A single shared timestamp makes the cadence independent of device
/// lock contention and prevents both paths from signaling the same interval.
///
/// The claimed timestamp advances on a **fixed interval grid** (`last +
/// INTERVAL`), not to `now_ms`. Resetting to `now` lets poll jitter shift the
/// cadence phase permanently: a poll that lands slightly late pushes the *next*
/// deadline out another full interval, so the delivered VBL rate aliases down —
/// when the effective poll spacing sits in the danger zone (just under the
/// interval) it needs two polls per delivery and halves toward ~60 Hz. That is
/// the boot-to-boot 60-vs-120 split the user reports: on a boot where the poll
/// heartbeat jitters into that zone the guest latches 60 Hz. Advancing by exactly
/// one interval keeps delivery phase-locked to the grid and lets a late poll
/// "catch up" (each subsequent poll delivers until the grid is caught, then a
/// poll naturally skips and resyncs) so the *steady* rate converges to the grid
/// (~120 Hz) regardless of poll jitter, erring toward the ceiling the guest caps
/// at rather than latching 60. A long stall (≥2 intervals, e.g. the drain worker
/// held the lock) resyncs the phase to `now_ms` so we never unleash a burst of
/// back-dated VBLs.
pub(crate) fn claim_display_vbl(last_us: &std::sync::atomic::AtomicU64, now_us: u64) -> bool {
    let last = last_us.load(std::sync::atomic::Ordering::Acquire);
    let gap = now_us.saturating_sub(last);
    if gap < DISPLAY_VBL_MIN_INTERVAL_US {
        return false;
    }
    let next = if gap >= 2 * DISPLAY_VBL_MIN_INTERVAL_US {
        now_us
    } else {
        last + DISPLAY_VBL_MIN_INTERVAL_US
    };
    last_us
        .compare_exchange(
            last,
            next,
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
        )
        .is_ok()
}

/// Pulse VBL at the phase-locked ~120 Hz cadence (grid interval
/// [`DISPLAY_VBL_MIN_INTERVAL_US`]; see [`claim_display_vbl`]).
///
/// Writes pending bit 0, sets 0x1014 display bit, and raises MSI after ONLINE
/// has been acked. The limiter is owned outside `DeviceState` so this locked
/// path and `vbl_contended_pulse` use one time base. Without VBL the guest
/// compositor can stick on clear-only DisplaySwap of empty flip buffers.
pub fn signal_display_vbl<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    last_us: &std::sync::atomic::AtomicU64,
) {
    signal_display_vbl_at(state, host, last_us, crate::observe::elapsed_us());
}

/// The engine's own counters, over the window `drain_duty` just reported.
///
/// Two of them were tallied and never reported, and they are the two that price
/// the largest phase of a draw. `draw_phase` puts **70% of all timed draw work
/// in `Acquire`** — 193 µs per draw, 130 ms of a driven second — while
/// `creates=74 allocs=0` rules out the allocation churn that phase's own doc
/// names as its cost. What is left in there and scales with content is
/// [`crate::backend::vulkan::engine::pools`]'s sampled-cache lookup, which
/// fingerprints the **whole** incoming blob with two SipHash passes on every
/// call that does not take the identity fast path.
///
/// `sampled_cache_hit_bytes` is exactly the byte count fed to that fingerprint
/// on the hit path, and `sampled_identity_hits` is the count that skipped it
/// entirely. Together they turn "the cache is working, hits=122 misses=0" —
/// which is what the line said, and which reads as nothing to fix — into a
/// GB/s figure that can be compared against SipHash's throughput. Neither can
/// be derived from the counts alone: 122 hits over 4 KiB blobs and 122 over
/// 8 MiB blobs are three orders of magnitude apart and the line printed the
/// same number for both.
///
/// `drain_duty` established that 96-99% of the saturated drain second is
/// `draw_us`, at 1.5-7 ms per draw — orders of magnitude more than a draw's CPU
/// encode should cost. Which of the engine's per-draw costs that is was already
/// being counted and never reported: `engine::counter_snapshot` had no product
/// caller, so every one of these numbers existed and no boot had read one.
///
/// So this adds no instrumentation, only a window delta of what the engine
/// already tallies, chosen to separate the candidates that could each explain
/// milliseconds per draw:
///
/// - `batch_*` — whether draws coalesce into one submission or each takes its
///   own. Per-draw submission is a full CPU-GPU round trip.
/// - `readbacks` / `readback_bytes` — whether every draw drags its target back
///   to host memory, which is a fence wait plus a copy.
/// - `render_post_wait_skips` / `target_reads` — the two halves of the deferred
///   composite Store. The first counts draws that returned without a fence wait
///   because they kept their pixels on the GPU; the second counts the reads a
///   consumer later asked for. A rail that only *moves* the copy raises the
///   second by as much as it raises the first, and `readbacks` alone — which
///   pooled both until it was split — reported no change at all in that case.
/// - `creates` / `*_misses` — pipeline, shader and descriptor churn, where a
///   miss is a driver compile rather than a lookup.
/// - `sampled_reuploads` — re-staging texture content a cache hit should have
///   kept.
/// - `sampled_gathers` / `sampled_gather_bytes` — sampled binds served by
///   gathering scattered guest pages into staging. The CPU-copy arm of a
///   guest-sourced bind, and the one the import rail exists to empty.
/// - `sampled_guest_imports` / `sampled_guest_import_bytes` — the same binds
///   served by the GPU reading the guest's pages through the imported RAMBlock,
///   with no CPU copy at all. Ranked against `sampled_gathers`, these two divide
///   every guest-sourced bind that had to move bytes into the ones that moved
///   them over the host CPU and the ones that did not; a host that cannot import
///   a host pointer reads zero here and all of it there.
/// - `draw_cover_*` — how much of its target each draw could have written.
///   Nothing acts on it; it is what says whether bounding a flush to a damage
///   rect could pay, and the answer is a rate against `surface_flush` rather
///   than a ratio between the three. See `EngineCounters::note_draw_coverage`.
/// - `buffer_guest_imports` / `buffer_guest_import_bytes` — vertex and storage
///   binds pointed straight at the guest's pages, with no copy in either
///   direction. `buffer_snapshot_binds` is what still had to be gathered, and
///   the `stage_phase` `runs_*` bars are what that gathering costs.
/// - `ring_retire_blocks` / `target_evicts` — the engine waiting on itself.
///
/// One line per second, one atomic load per field. Emitted from the same window
/// as `drain_duty` so the two divide against each other; a delta on its own
/// clock would not.
/// Would a resident carry the present this mapping names, at this geometry?
///
/// `Some(true)` a presentable resident exists, `Some(false)` none does — so a
/// present with no guest-page frame behind it shows black — and `None` on a
/// backend with no target registry to ask, where the honest answer is that this
/// build cannot tell.
///
/// It asks through [`crate::backend::vulkan::engine::resident_presentable`],
/// which shares `pools::slot_presentable` with the window presenter's own
/// selection. Sharing the rule is the point rather than tidiness: a looser
/// predicate here would report a frame as carried that the presenter then
/// refuses, which is a disagreement neither call site can see on its own — the
/// same shape as the publish/present split that once blanked the window.
#[cfg(feature = "backend-vulkan")]
fn present_resident_carries(
    state: &crate::model::DeviceState,
    mapping: u32,
    width: u32,
    height: u32,
) -> Option<bool> {
    let identity =
        crate::runtime::present_identity::surface_identity(state, mapping, width, height);
    Some(crate::backend::vulkan::engine::resident_presentable(
        &identity, width, height,
    ))
}

#[cfg(not(feature = "backend-vulkan"))]
fn present_resident_carries(
    _state: &crate::model::DeviceState,
    _mapping: u32,
    _width: u32,
    _height: u32,
) -> Option<bool> {
    None
}

/// Which channel an unbacked present belongs on: `true` is the failure channel.
///
/// A separate function because the `None` arm is the whole content of the rule
/// and it is one character away from being wrong. `carried != Some(true)` and
/// `carried == Some(false)` differ only when the build cannot answer, and that is
/// exactly the case where a possible black frame would be downgraded to a census
/// with nothing to notice it. Fail-closed: only a resident that positively
/// carries the frame demotes the line.
fn unbacked_present_is_a_loss(carried: Option<bool>) -> bool {
    carried != Some(true)
}

/// The `carried=` field: what answered for this present, or that nothing could.
fn carrier_word(carried: Option<bool>) -> &'static str {
    match carried {
        Some(true) => "resident",
        Some(false) => "nothing",
        None => "unknown",
    }
}

fn signal_display_vbl_at<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    last_us: &std::sync::atomic::AtomicU64,
    now_us: u64,
) {
    // The limiter paces in microseconds because 120 Hz is not expressible in
    // whole milliseconds; the census windows in milliseconds so its `t=` stays
    // on the same scale as every other always-on line.
    let now_ms = now_us / 1_000;
    if state.display.shared_gpa == 0 || !state.display.online_acked {
        note_vbl(VBL_NOT_ONLINE, now_ms);
        return;
    }
    if !claim_display_vbl(last_us, now_us) {
        note_vbl(VBL_NOT_CLAIMED, now_ms);
        return;
    }
    note_vbl(VBL_DELIVERED, now_ms);
    let gpa = state.display.shared_gpa;
    let mut pending_le = [0u8; 4];
    let pending = if host
        .read_gpa(gpa + DISPLAY_SHARED_PENDING, &mut pending_le)
        .is_ok()
    {
        ld32(&pending_le)
    } else {
        0
    };
    // Drop a stale (already-acked) ONLINE bit so we don't re-deliver it and make
    // the guest re-run process_online → connectionChange → overlay rebuild (see
    // signal_display_present_complete). signal_display_vbl only runs post-ack, so
    // online_acked is already true here; `stale` is 0 on healthy boots (no-op).
    let stale = state.display.online_acked && pending & DISPLAY_ONLINE_EVENT_MASK != 0;
    let base = if stale {
        pending & !DISPLAY_ONLINE_EVENT_MASK
    } else {
        pending
    };
    shared_w32(
        host,
        gpa,
        DISPLAY_SHARED_PENDING,
        base | DISPLAY_VBL_EVENT_MASK,
        state.page_size() as usize,
    );
    if stale {
        crate::runtime::census::present_proxy::note_stale_online_pending("vbl", pending);
    }
    let bit = 1u32 << (state.display.display_index & 0x1f);
    state
        .gfx
        .interrupt_status_disp
        .fetch_or(bit, std::sync::atomic::Ordering::AcqRel);
    host.enqueue(HostAction::irq_gfx());
}

/// Assert display ONLINE once the guest has published the enable mask.
///
/// Archive `apple_pv_gpu_display_signal_online` + poll_tick gate:
/// write shared `+0x100` pending bit 2, then pulse display IRQ. Only after
/// `enable()` sets `+0x104` bit 2 — earlier IRQs wedge an unregistered display.
/// createDisplayAttributes then consumes TimingElements (incl. 1440 mode).
pub fn try_display_online<H: HostMemory + HostOps>(state: &mut DeviceState, host: &mut H) {
    if state.display.shared_gpa == 0 || state.display.online_acked {
        return;
    }
    if state.display.online_tries >= DISPLAY_ONLINE_MAX_TRIES {
        // Reaching the cap means this device has stopped asserting ONLINE for
        // this shared-state generation. The desktop the user is waiting for is
        // not coming — with, until this line, nothing in the log to say so. A
        // give-up is the loudest thing a device can do quietly.
        //
        // **What it means is the opposite of what this line used to say.**
        // `online_tries` is incremented at the tail of this function, past the
        // `enable()` mask check, so it only counts ONLINE pulses delivered to a
        // guest that *has* enabled. A guest that never enables returns at that
        // check and never increments it, so it can never reach this cap — the
        // one guest state the old wording named is the one state that cannot
        // produce this line. What it actually reports is a guest that enabled,
        // took 150 ONLINE pulses, and acked none of them.
        //
        // Nor is it "for the life of the boot": `apply_setup_shared_state`
        // zeroes `online_tries` and `poll_ctr`, so a `setupSharedState` reinit
        // hands the handshake a fresh 150. Both corrections matter to whoever
        // is debugging a black screen, because each sent them at the wrong half
        // of the rail.
        //
        // Latched on the display index rather than counted: the cap is crossed
        // on every subsequent poll until a reinit, so a per-crossing line would
        // be a ~5 Hz flood of the same fact. `display_online_signal` fires on
        // the *first* pulse of the same generation, so that line always precedes
        // this one and the pair brackets the whole handshake.
        if crate::observe::first_sight(
            "display_online_abandoned",
            u64::from(state.display.display_index),
        ) {
            crate::observe::fail(format!(
                "display_online_abandoned index={} tries={DISPLAY_ONLINE_MAX_TRIES} \
                 divisor={DISPLAY_ONLINE_POLL_DIVISOR} \
                 (the guest enabled the display and acked none of the ONLINE \
                 pulses; no more are sent until a setupSharedState reinit)",
                state.display.display_index
            ));
        }
        return;
    }
    // Cadence: skip most ticks (archive divisor); still run often enough via
    // gfx_update / drain that enable() is observed within seconds.
    let ctr = state.display.poll_ctr.wrapping_add(1);
    state.display.poll_ctr = ctr;
    if !ctr.is_multiple_of(DISPLAY_ONLINE_POLL_DIVISOR) {
        return;
    }
    let gpa = state.display.shared_gpa;
    let mut mask_le = [0u8; 4];
    if host
        .read_gpa(gpa + DISPLAY_SHARED_ENABLE_MASK, &mut mask_le)
        .is_err()
    {
        // The guest published this page's address itself, so a read of it that
        // the host cannot perform is not expected control flow — it is the
        // handshake's only input becoming unreadable.
        //
        // It does **not** spend a try, which this comment used to claim: the
        // return here is upstream of the increment at the tail, so an
        // unreadable mask spends a *poll* and the cap is never reached. The
        // handshake then polls forever with this one latched line as its only
        // trace, which is why the line has to be here rather than left to the
        // cap to report.
        //
        // Latched on the address, because whatever makes a GPA unreadable makes
        // it unreadable on every poll.
        if crate::observe::first_sight("display_online_mask_unreadable", gpa) {
            crate::observe::fail(format!(
                "display_online_mask_unreadable gpa={:#x} \
                 (the shared enable mask the guest published cannot be read, so \
                 the ONLINE handshake polls forever without ever pulsing; it \
                 spends no tries and will never reach the cap, so this line is \
                 the only trace it leaves)",
                gpa + DISPLAY_SHARED_ENABLE_MASK
            ));
        }
        return;
    }
    let mask = ld32(&mask_le);
    // Not a failure: the guest has published the page but not yet called
    // `enable()`. This is the state the poll exists to wait out.
    if mask & DISPLAY_ONLINE_EVENT_MASK == 0 {
        // Waiting it out has no end, though, and this is the state that had no
        // instrument at all — the one `display_online_abandoned` was worded as
        // though it covered and structurally cannot. A guest that publishes the
        // shared page and never enables polls here at the divisor's cadence for
        // the life of the generation, emitting nothing: no `signal`, no
        // `abandoned`, and a black screen with a clean log.
        //
        // So the wait is bounded for *reporting* only — the poll continues
        // afterwards, because the guest may still enable and there is no reason
        // to stop offering. The bound is the same span the cap already spends,
        // `MAX_TRIES` pulses at one per `DIVISOR` ticks, so no new number is
        // introduced and the two halves of the handshake time out alike.
        let waited = u64::from(DISPLAY_ONLINE_MAX_TRIES) * u64::from(DISPLAY_ONLINE_POLL_DIVISOR);
        if u64::from(state.display.poll_ctr) > waited
            && crate::observe::first_sight(
                "display_online_never_enabled",
                u64::from(state.display.display_index),
            )
        {
            crate::observe::fail(format!(
                "display_online_never_enabled index={} gpa={:#x} mask={mask:#x} polls={} \
                 (the guest published the display shared page and has not set the \
                 enable bit; ONLINE has never been asserted and the poll continues)",
                state.display.display_index,
                gpa + DISPLAY_SHARED_ENABLE_MASK,
                state.display.poll_ctr,
            ));
        }
        return;
    }
    // pending word is atomic read-and-clear on the guest side.
    shared_w32(
        host,
        gpa,
        DISPLAY_SHARED_PENDING,
        DISPLAY_ONLINE_EVENT_MASK,
        state.page_size() as usize,
    );
    let bit = 1u32 << (state.display.display_index & 0x1f);
    state
        .gfx
        .interrupt_status_disp
        .fetch_or(bit, std::sync::atomic::Ordering::AcqRel);
    host.enqueue(HostAction::irq_gfx());
    // Always-on on the first ONLINE pulse per shared-state generation (rare, not a
    // flood): the display-lifecycle timeline entry point. A second pass through here
    // after a reinit setup pairs with display_shared_state_setup reinit=1 to show a
    // post-converge display rebuild.
    if state.display.online_tries == 0 {
        crate::observe::fail(format!(
            "display_online_signal index={}",
            state.display.display_index
        ));
    }
    state.display.online_tries = state.display.online_tries.saturating_add(1);
}

/// Drain active/pending child FIFOs other than channels mid-drain.
///
/// Used by DisplaySwap Dekker rescue and stranded paths — **not** by
/// `render_wait_surface` (archive wait is surface-keyed async completion only).
///
/// Mask matches archive `poll_tick`: `active_child_mask | pending_child_mask`
/// so work doorbell'd while a drain was in flight is not skipped. Skips
/// `skip_channel` and every bit in `state.draining_mask` so nested drains
/// cannot re-enter a mid-packet channel (same head re-process).
pub fn drain_other_child_fifos<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    skip_channel: u32,
) {
    let mask = state.active_child_mask | state.pending.child_mask;
    let nested = state.draining_mask;

    // A cold-translation EXEC is already the oldest accepted item in the host
    // scheduler timeline. Retry its channel(s) first. Sibling FIFO packets may
    // tear down tasks, mappings, objects, or surfaces referenced by that EXEC,
    // so none may pass the artificial translation boundary. Translation owns
    // immutable AIR and completes independently of all FIFO drains.
    let deferred = state.translation_deferred_mask;
    if deferred != 0 {
        for ch in 1..MAX_CHANNELS as u32 {
            let bit = 1u32 << ch;
            if deferred & bit == 0 || ch == skip_channel || nested & bit != 0 {
                continue;
            }
            state.pending.child_mask &= !bit;
            drain_child_fifo(state, host, ch);
        }
        if state.translation_deferred_mask != 0 {
            let held = mask & !deferred & !nested & !(1u32 << skip_channel);
            state.pending.child_mask |= held | state.translation_deferred_mask;
            note_translation_order_hold(state, held);
            return;
        }
        release_translation_order_holds(state);
    }

    let mut remaining = mask;
    for ch in 1..MAX_CHANNELS as u32 {
        if state.pending.host_action_yield {
            break;
        }
        if ch == skip_channel {
            continue;
        }
        if nested & (1u32 << ch) != 0 {
            continue;
        }
        if mask & (1u32 << ch) == 0 {
            continue;
        }
        remaining &= !(1u32 << ch);
        // Clear pending bit for channels we actually drain (archive poll_tick
        // consumes pending when it drains). Leave skip/nested bits alone.
        state.pending.child_mask &= !(1u32 << ch);
        drain_child_fifo(state, host, ch);
        if state.translation_deferred_mask != 0 {
            let held = remaining & !nested & !(1u32 << skip_channel);
            state.pending.child_mask |= held | state.translation_deferred_mask;
            note_translation_order_hold(state, held);
            break;
        }
    }
}

/// Host paint consumed the current +0x188 retain (Painted or Unchanged).
///
/// Clears the entry-side present backpressure counter so a DisplaySwap held
/// at channel head can run on the next drain (schedule_bh from scanout path).
pub fn note_present_paint_consumed(state: &mut DeviceState) {
    state.present.unpainted_presents = 0;
    state.present.backpressure_hold_active = false;
    state.pending.host_action_yield = false;
}

fn note_present_backpressure_hold(state: &mut DeviceState, channel: u32, head: u32, tail: u32) {
    if state.present.backpressure_hold_active
        && state.present.backpressure_hold_channel == channel
        && state.present.backpressure_hold_head == head
    {
        return;
    }
    state.present.backpressure_hold_active = true;
    state.present.backpressure_hold_channel = channel;
    state.present.backpressure_hold_head = head;
    state.present.backpressure_hold_count = state.present.backpressure_hold_count.saturating_add(1);
    crate::observe::fail(format!(
        "THRASH present_action_starvation reason=pending_frames_cap ch={channel} head={head} tail={tail} unpainted={} episode={}",
        state.present.unpainted_presents, state.present.backpressure_hold_count
    ));
}

/// Publish the poll-tick/Dekker rescue to the asynchronous drain owner.
///
/// This performs no guest-memory reads and no command execution. QEMU may call
/// it from its display/main-loop context without accidentally translating or
/// submitting GPU work under the BQL. Active child channels are intentionally
/// coalesced into one mask; the worker's normal ring checks make idle channels
/// cheap no-ops.
pub fn publish_stranded_fifos<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
) -> bool {
    let mut published = false;
    if state.gfx.control_fifo != 0 {
        if state
            .gfx
            .fifo_read
            .load(std::sync::atomic::Ordering::Acquire)
            != state.gfx.fifo_written
        {
            state.pending.main_drain = true;
            published = true;
        }
        if state.active_child_mask != 0 {
            state.pending.child_mask |= state.active_child_mask;
            published = true;
        }
    }
    if state.iosfc.consumer != state.iosfc.producer {
        state.pending.iosfc = true;
        published = true;
    }
    if published {
        host.schedule_bh();
    }
    published
}

/// Run all pending drains (BH body).
///
/// # This runs to completion on purpose, and a wall-clock budget was tried
///
/// This holds the device lock for its whole duration, and on x86 the vCPU does
/// not block on that lock — `device_gfx_write` takes it with `try_lock` and on
/// failure queues the write, so the guest's store retires while the work it rang
/// for does not start until this returns. A driven boot measures tranches of
/// 18-43 ms and ~105 doorbells a second applied at least one whole frame late
/// (`gfx_doorbell_delay`), so capping the tranche is an obvious thing to reach
/// for. It was reached for, measured, and reverted.
///
/// A budget of one frame interval, checked between child channels using the same
/// requeue the translation-hold and `host_action_yield` arms below already use,
/// **made the delay worse**: mean doorbell age 19.0 ms against 10.8 ms before,
/// and `max_tranche_us` 34-37 ms against 18-22 ms. It fired only ~22 times a
/// second against ~200 tranches, because the cost is not spread across channels
/// — a single channel's flush run holds the lock for tens of milliseconds and a
/// between-channel check cannot reach inside it.
///
/// It also introduced a stall. Returning with `child_mask` still set leaves work
/// that nothing re-arms: a doorbell would, but the guest has no reason to ring
/// one for work it already submitted, and the 4 ms poll publishes *producer*
/// state rather than noticing an already-set mask. One boot froze for 29 s with
/// the census silent and the guest reporting a single 30 263 ms frame. The same
/// gap exists in principle for the translation-hold arm below, which has never
/// been observed to hit it — worth knowing if one ever does.
///
/// The cost is inside the render writeback, not in the scheduling around it.
///
/// # What that refutation does *not* cover, and the measurement that separates them
///
/// Read as "the doorbell delay is the flush rail's cost, so shrink the flush",
/// the paragraph above overstates itself, and [`DoorbellCensus`] now says by how
/// much. `max_age_us` tracks `max_tranche_us` to within 3 % across a driven
/// boot — 41711/42563, 40627/42117, 103619/105308 — and one of those windows
/// read `duty=0.147`. The worker was idle for 85 % of that second and still held
/// the guest's next submission for a tenth of it, because a doorbell does not
/// wait for the device to be *busy*, it waits for the device to be *holding the
/// lock*. A tranche whose work halves still costs a doorbell half a tranche.
///
/// The thing that was tried and reverted was **pausing** this function: return
/// early on a budget, requeue the rest. That is what made the delay worse and
/// what introduced the 29 s freeze, and it stays refuted.
///
/// A queued write does not need this to pause. It needs its register applied,
/// which costs microseconds and *adds* available work rather than interrupting
/// any — and the queue is drained today only by `lock_for_drain`, once, before
/// the tranche starts. Nothing has measured which registers are in it; the
/// `off_*` breakdown on `gfx_doorbell_delay` is there to answer that first,
/// because "apply it sooner" is only safe for registers whose effect is to
/// publish more work, and unsafe for any the decode below depends on not
/// changing mid-tranche.
/// How many times one tranche will pick up newly rung child channels.
///
/// Not a time budget and not a work budget — a bound on how many times the
/// drain will go back for doorbells that arrived while it was running. Three
/// covers the measured shape: `gfx_doorbell_delay` reads about a hundred rings
/// a second against tranches of tens of milliseconds, so at most a handful of
/// channels are rung during any one pass and a channel already served is
/// excluded from the refill. The cap exists so a guest that rings continuously
/// cannot hold the device lock indefinitely, not because any run has needed it.
const CHILD_DOORBELL_REFILLS: u32 = 3;

/// Move child channels the guest rang lock-free into the drain's pending mask.
///
/// The guest's doorbell write does not take the device lock — see
/// [`crate::model::GfxRegs::child_doorbell_rung`] for why that register can be
/// taken that way and no other can. This is where the bits become work.
///
/// `active_child_mask` is set as well as `pending.child_mask`, because that is
/// what the locked handler in `runtime::mmio` does for the same register and the
/// two must not disagree: `publish_stranded_fifos` re-publishes from
/// `active_child_mask`, so a channel that only ever rang lock-free would be
/// invisible to the stranded-FIFO rescue.
pub(crate) fn fold_rung_child_doorbells(state: &mut DeviceState) {
    let rung = state
        .gfx
        .child_doorbell_rung
        .swap(0, std::sync::atomic::Ordering::AcqRel);
    if rung == 0 {
        return;
    }
    state.active_child_mask |= rung;
    state.pending.child_mask |= rung;
}

/// Re-offer every timeline held on an unmet stamp wait, for as long as the pass
/// keeps publishing stamps.
///
/// The channel loop above walks channels in id order, so a channel held on a
/// slot that a *higher-numbered* channel publishes is passed over before the
/// thing it waits for has run. Its bit then sits in `pending.child_mask` with
/// nothing guaranteed to ring for it: the doorbell that would re-arm it belongs
/// to the producing channel, whose work this pass already drained. This closes
/// that window inside the pass.
///
/// **The loop is bounded by progress, not by a count.** Each round runs only if
/// the previous one advanced [`DeviceState::completion_stamp_seq`] — a stamp
/// reaching guest RAM is the only event that can turn an unmet wait into a met
/// one, so a round that publishes nothing cannot have unblocked anything and
/// there is no reason to look again. That makes a mutual wait between two
/// channels terminate in one round rather than spin, and it needs no cap to say
/// so: a cap here would be a bound on how far ordering is honoured, which is
/// exactly the shape this device is trying not to have.
///
/// What is left held is handed back with its bit set, which is correct and not a
/// drop: the awaited work has not been submitted yet, the guest will submit it,
/// and that submission rings its own doorbell. Holding is what a GPU does with a
/// wait it cannot yet satisfy.
///
/// # The failure mode honouring the wait creates, named because it is new
///
/// A channel whose head packet cannot be decoded stops there: `drain_child_fifo`
/// records the fault and breaks without advancing the head or writing the
/// stamp, so that channel's slot never moves again. That has always been a
/// stall of one channel. It is now a stall of **every timeline waiting on that
/// channel's slot**, because they are correctly refusing to run ahead of work
/// that will never complete.
///
/// This is faithful — a real GPU's waiters do not proceed past a wedged engine —
/// and it is the direction to prefer, since the alternative was those timelines
/// running ahead and corrupting silently. But it changes the *shape* of the
/// symptom, and that is worth knowing before debugging one: a device that has
/// gone quiet on several channels at once is more likely to have one wedged
/// producer than several independent faults. `MalformedChildPacket` on the
/// producing channel is the line to look for, and it will be the *earliest* of
/// them; `stamp_hold_handed_back` climbing while nothing else moves is the
/// secondary signal.
///
/// No escape hatch is offered on purpose. A hold that gave up after N rounds
/// would be a bound on how long ordering is honoured, and it would fire on the
/// healthy case — a guest that simply has not submitted the producing work yet —
/// long before it ever reached a wedged one.
fn retry_stamp_held_timelines<H: HostMemory + HostOps>(state: &mut DeviceState, host: &mut H) {
    while state.stamp_deferred_mask != 0 {
        let seq_before = state.completion_stamp_seq;
        if state.stamp_deferred_mask & ROOT_FIFO_BIT != 0 {
            drain_main_fifo(state, host);
        }
        for ch in 1..MAX_CHANNELS as u32 {
            let bit = 1u32 << ch;
            if state.stamp_deferred_mask & bit == 0 {
                continue;
            }
            drain_child_fifo(state, host, ch);
            if state.pending.host_action_yield {
                return;
            }
        }
        if state.completion_stamp_seq == seq_before {
            note_store_route("stamp_hold_handed_back");
            return;
        }
        note_store_route("stamp_hold_retry");
    }
}

pub fn drain_pending<H: HostMemory + HostOps>(state: &mut DeviceState, host: &mut H) {
    // A queued present action is part of the ordered device timeline. QEMU
    // cannot paint it while this worker owns the device lock, so later worker
    // wakeups must leave guest work queued until scanout consumes the action.
    if state.pending.host_action_yield {
        return;
    }
    release_translation_order_holds(state);
    // Retry an already translation-held EXEC before allowing either the root
    // FIFO or a sibling child FIFO to overtake it. The guest is free to queue
    // Unmap/Delete immediately after submission; without this boundary the
    // retried EXEC can stage successfully and then write back after its task
    // mapping has been destroyed.
    let deferred = state.translation_deferred_mask;
    if deferred != 0 {
        if state.pending.main_drain {
            note_translation_order_hold(state, ROOT_FIFO_BIT);
        }
        let sibling_pending = state.pending.child_mask & !deferred;
        note_translation_order_hold(state, sibling_pending);
        for ch in 1..MAX_CHANNELS as u32 {
            let bit = 1u32 << ch;
            if deferred & bit == 0 {
                continue;
            }
            state.pending.child_mask &= !bit;
            drain_child_fifo(state, host, ch);
        }
        if state.translation_deferred_mask != 0 {
            state.pending.child_mask |= state.translation_deferred_mask;
            return;
        }
        release_translation_order_holds(state);
    }
    if state.pending.main_drain {
        drain_main_fifo(state, host);
    }
    fold_rung_child_doorbells(state);
    let mut mask = state.pending.child_mask;
    state.pending.child_mask = 0;
    // Channels this pass has already run, so a refill cannot re-run one.
    let mut served = 0u32;
    // Bounded refills. Each pass picks up channels the guest rang *while the
    // previous pass was running, under the device lock it could not take* —
    // which is the whole point, and is why the doorbell was worth making
    // lock-free. Serving them here rather than next tranche is what turns a
    // ring into work that starts now.
    //
    // Bounded because the guest can ring faster than this drains, and an
    // unbounded refill would hold the device lock for as long as it kept
    // ringing. Leaving the remainder is safe here in a way it was NOT for the
    // reverted tranche budget: that one returned with `child_mask` set and
    // nothing to re-arm it, and froze a boot for 29 s. Every bit that arrives
    // here arrives with its own `schedule_bh` already rung by the vCPU, so the
    // worker is guaranteed another wakeup for whatever this pass leaves.
    for _ in 0..CHILD_DOORBELL_REFILLS {
        let mut remaining = mask;
        for ch in 1..MAX_CHANNELS as u32 {
            let bit = 1u32 << ch;
            if mask & bit != 0 {
                remaining &= !bit;
                served |= bit;
                drain_child_fifo(state, host, ch);
                if state.translation_deferred_mask != 0 {
                    state.pending.child_mask |= remaining | state.translation_deferred_mask;
                    note_translation_order_hold(state, remaining);
                    return;
                }
                if state.pending.host_action_yield {
                    state.pending.child_mask |= remaining;
                    return;
                }
            }
        }
        fold_rung_child_doorbells(state);
        // Only channels this pass has not already run: a channel rung again
        // while its own drain was in flight has had that work seen, and
        // re-running it here would spin on one busy channel while the others
        // wait.
        mask = std::mem::take(&mut state.pending.child_mask) & !served;
        if mask == 0 {
            break;
        }
        note_store_route("child_doorbell_refill");
    }
    // Whatever the refill cap left, handed back to the next wakeup.
    state.pending.child_mask |= mask;
    retry_stamp_held_timelines(state, host);
    if state.pending.iosfc {
        drain_iosfc(state, host);
    }
    try_display_online(state, host);
    // Unmap contiguous views retired by MAP/UNMAP/page-table changes (their
    // Metal objects were dropped at retire time; execution is sync-per-packet
    // so nothing aliases them anymore).
    crate::runtime::mapper::flush_retired_views(state, host);
    // Unpin engine residents of linear cache entries dropped by task/object
    // deletes this drain, so they become LRU-evictable instead of leaking.
    crate::runtime::render_writeback::retire_linear_residents(state);
}

#[cfg(test)]
mod tests;
