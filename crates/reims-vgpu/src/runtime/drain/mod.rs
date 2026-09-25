//! Root/child FIFO drains, stamp writeback, and fail-visible command dispatch.
//!
//! Prefer structure correctness over full exec.c coverage: known root/child
//! control-plane ops update device state; unknown opcodes are recorded visibly.

use crate::backend::{Backend as _, RetainedObject};
use crate::model::*;
use crate::model::{DeviceState, ExecFault, FailEvent, PacketFault, UnimplementedCommand};
use crate::observe::Emit;
use crate::protocol::endian::{ld16, ld32, st16, st32};
use crate::protocol::fifo::{
    display_refresh_hz_1616, display_timing_entry_offset, encode_display_timing_entry,
    DisplayTimingEntry, DISPLAY_DESC_TIMING_STRIDE,
};
use crate::protocol::info_reply::{self, ReplyBounds};
use crate::protocol::iosurface_pages::{
    MAPPER_REQUEST_ENTRY_LEN, MAPPER_REQUEST_MAP, MAPPER_REQUEST_MAPPING_ID, MAPPER_REQUEST_TYPE,
    MAPPER_REQUEST_UNMAP,
};
use crate::protocol::packets::Channel as WireChannel;
use crate::protocol::present::{
    self, trailer as present_trailer, PresentForm, Trailer as PresentTrailer,
};
use crate::runtime::gpa_map;
use crate::runtime::heap_query::QueryError;
use crate::runtime::host::{HostAction, HostMemory, HostOps, MemError};
use crate::runtime::task_slot::{resolve_task_word, TaskWordSite};
use reims_vgpu_core::control::{self, ControlKind};
use reims_vgpu_core::query::{self, QueryKind, RequestWords};

pub(crate) mod census;
pub use census::*;

// The Vulkan rail's completion-stamp publication, named rather than
// re-exported flat — `write_stamp` and the root drain reach it only through
// `Backend`, so neither one mentions a rail.
#[cfg(feature = "backend-vulkan")]
pub mod vulkan;

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
/// The completion slot every root packet's word is written to.
///
/// Not read from a register: the root FIFO has no stamp-index register of its
/// own and this device has always written slot 0 for it. Named so the model's
/// packets carry the same number this device writes.
const ROOT_STAMP_SLOT: u32 = 0;

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

/// `CmdDeleteObject` (`0x28`): decode the serializer object and retire its kind.
///
/// The payload is `{u32 task}` then one serializer destroy record. Every kind in
/// that family writes the identical twelve-byte body — a single object ref — and
/// carries the kind in the record's own opcode.
///
/// # Separate reference spaces stay separate
///
/// The ref is in the **serializer's per-kind ref space**. The object table is
/// keyed by the *kernel object-list* ref, established by `0x33 CmdSetObjectList` and
/// populated when a decoded command resolves an entry out of that list, so this
/// command must never call `DeviceState::delete_object`: equal integers across
/// those spaces are unrelated. Samplers and render pipeline states have their
/// own task-local retained registries, and each destroy opcode retires exactly
/// its kind's registry entry.
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
/// The fence kind is no longer among them. It is the one kind this device holds
/// state for by `(task, ref)` rather than by content, and the same question this
/// doc records answering *no* for the object table was asked of it and answered
/// **yes**: a driven boot's two fence deletes both named a ref this device held
/// a render-domain generation under, with `delete_fence_ref_absent=0`. So the
/// spaces coincide there, and the delete retires the generations.
///
/// The kinds left are the ones this device retains nothing by ref for —
/// functions and compute pipeline states are cached by *content*, so a ref-keyed
/// retirement has nothing to find. A driven boot sends 5 and 3 of them, and
/// [`note_delete_object_ref_space`] then asked the only question that could put
/// a retirement back on the table: **all 8 named no object-list entry at all**,
/// `delete_object_ref_type_agrees` and `_differs` both silent. So those refs are
/// in a space this device holds nothing keyed by, and there is nothing to find
/// in either direction.
///
/// They stay fail-visible even so. "There is nothing to retire" is a claim about
/// this device's caches and not about the guest's object, and a counter that
/// went quiet would stop saying which contract gap is still open — and the gap
/// that is still open is now exactly the five kinds a driven guest has never
/// sent: buffer, texture, heap, rasterization rate map, indirect command buffer.
fn apply_delete_object<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    channel_id: u32,
    command: &crate::protocol::fifo::DeleteObjectCommand<'_>,
    packet: &Packet,
) {
    let crate::protocol::fifo::DeleteObjectCommand { task_id, record } = *command;
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
    let object_ref = rec.object_ref.get();
    note_store_route(delete_object_kind_route(op.opcode()));
    // The ordering group's question, asked of the kinds this device *does*
    // retire. The untracked kinds were asked it at the bottom of this function
    // and answered "no list entry at all", which settled that a retirement
    // cannot be keyed there. The retained kinds decide something larger: if a
    // sampler-state destroy's ref is an object-list name, the semantic model
    // can express this command as an operation on a resource it already knows,
    // and if it is not, the serializer's per-kind spaces are a namespace the
    // model does not have and this opcode cannot cross the bridge as one.
    //
    // Asked before the retirement acts, so the reading is about the ref the
    // guest sent and not about what dropping it did.
    let ref_space = is_retained_kind(op.opcode()).then(|| {
        note_delete_object_ref_space(
            state,
            host,
            task_id,
            object_ref,
            op.opcode(),
            RefSpacePopulation::Retained,
        )
    });
    if op.opcode() == reims_vgpu_wire::ops::destroy::OPCODE_DELETE_SAMPLER_STATE {
        let retired = state.task_sampler_states.delete(task_id, object_ref);
        note_store_route(if retired {
            "sampler_state_deleted"
        } else {
            "sampler_state_delete_absent"
        });
        return;
    }
    // The two kinds a rail may retain by `(task, ref)`. Whether one does is the
    // running rail's answer and not the build's: these used to be
    // `cfg(feature = "backend-vulkan")` arms, which on a `--backend both`
    // binary retired from the Vulkan tables during a Metal run — reporting
    // `..._delete_absent` about a table that rail never fills — and on a
    // Metal-only build fell through to `note_unimplemented` instead. One rail,
    // two builds, two different answers about the same guest command.
    // The one kind this device holds state for by `(task, ref)` rather than by
    // content, and the one whose stale entry is a wrong answer rather than a
    // leak: a wait is satisfied when the stored generation is at or past its
    // target, so a generation that outlives its fence makes the next fence to
    // get that ref start life already signalled.
    //
    // Acted on because it was counted first. A driven boot sent two fence
    // deletes and both named a ref this device held a render-domain generation
    // under, `delete_fence_ref_absent=0` — so the serializer's fence ref space
    // and a fence record's are one, which is *not* what the same question
    // answered for the object table above. The census stays as the denominator:
    // `_absent` climbing is the reading that would put this arm back in doubt.
    if op.opcode() == reims_vgpu_wire::ops::destroy::OPCODE_DELETE_FENCE {
        let cleared = state.retire_fence(task_id, object_ref);
        if cleared.is_empty() {
            note_store_route("delete_fence_ref_absent");
        } else {
            note_store_route("delete_fence_ref_live");
            note_store_route_n("delete_fence_ref_live_domains", cleared.len() as u64);
        }
        return;
    }
    let retained = match op.opcode() {
        reims_vgpu_wire::ops::destroy::OPCODE_DELETE_DEPTH_STENCIL_STATE => {
            Some(RetainedObject::DepthStencilState)
        }
        reims_vgpu_wire::ops::destroy::OPCODE_DELETE_RENDER_PIPELINE_STATE => {
            Some(RetainedObject::RenderPipelineState)
        }
        _ => None,
    };
    if let Some(object) = retained {
        // The ordering plane's half, for the kind it holds a lifetime for. A
        // render pipeline is declared into `SessionModel::pipelines` when the
        // guest creates it, and this is the guest saying it is over — without
        // it the table only grows, and a transaction parked on a compilation
        // the guest has just cancelled would wait for a step that never comes.
        if object == RetainedObject::RenderPipelineState {
            // **Only when the slot the ref names still holds a render pipeline.**
            // The census two blocks up asks exactly this and it is not
            // rhetorical: a driven macos-26 boot answered `TypeDiffers` for 34
            // of the 36 destroys that reached this line. A destroy whose ref
            // resolves to an object of another kind is a ref in the
            // serializer's own per-kind space — the census's own conclusion —
            // and `name_resource` then hands back the name of whatever
            // *shares the integer*, which this retires.
            //
            // A pipeline table entry is a tombstone once retired: `declare`
            // refuses an id it already holds and `peek` answers
            // `AbsentBecause::Retired` forever, so one wrong retirement refuses
            // every later packet that binds the real pipeline. That is what it
            // cost — **914 exec packets refused `pipeline_absent_retired` on
            // macos-26 against zero on macos-15**, which is the boot where the
            // guest deletes four.
            //
            // `NoListEntry` retires too, and deliberately: the guest clears its
            // own object-list slot before sending the destroy, so an absent
            // entry is the ordinary case and `name_resource` answers it from
            // the name the slot already had. It is only a *disagreeing* entry
            // that says the integer belongs to something else.
            if ref_space == Some(RefSpaceAnswer::TypeDiffers) {
                note_store_route("pipeline_retire_declined_other_ref_space");
            } else if let Some(name) =
                crate::runtime::objects::name_resource(state, host, task_id, object_ref)
            {
                let ended = state.retire_pipeline(name);
                note_store_route_n("pipeline_retire_released", ended.stranded.len() as u64);
                // Whether the table had an entry to retire, which is not the
                // same question as whether the guest sent a delete. A driven
                // `macos-26` boot sent 170 and the table took 116: the other 54
                // named render pipelines this device never drew with, so they
                // were never declared. That is ordinary, and naming it is what
                // makes the day it stops being ordinary visible — the two
                // numbers were otherwise separated by a subtraction of the
                // `pipeline_table` occupancy line from this counter.
                note_store_route(if ended.took {
                    "pipeline_retired"
                } else {
                    "pipeline_retire_absent"
                });
                // **And the name ends with it.** The pipeline table's entry is a
                // tombstone keyed by this name, so a name that outlived the
                // delete is one the guest's *next* pipeline in this slot
                // inherits: `name_resource` answers from the live name without
                // re-reading the list, the table answers `Retired` for it, and
                // `waits_for` reads `Retired` as nothing to wait on. The packet
                // is released while the new pipeline's shaders are still
                // translating and its first draw is dropped as
                // `m2v_translation_pending_at_sync_boundary` with
                // `model_pipeline=retired`. macOS 26's icon agent builds and
                // deletes pipelines per icon, and lost the one draw each icon
                // is made of — every app icon blank.
                //
                // The name only, under the same gate as the table entry above:
                // the slot stops resolving, accepted work keeps what it
                // resolved, and the next declaration in the slot mints a new
                // generation the table has never seen. Not `delete_object` —
                // the object table and the host copies are keyed by the same
                // integer in the resource space, and
                // `a_delete_object_never_retires_an_object_table_entry_its_ref_collides_with`
                // holds that no destroy record crosses into them.
                note_store_route(match state.retire_object_name(task_id, name) {
                    Some(_) => "pipeline_name_retired",
                    None => "pipeline_name_retire_absent",
                });
            } else {
                note_store_route("pipeline_retire_unnamed");
            }
        }
        let outcome =
            crate::backend::selected().retire_task_object(state, task_id, object, object_ref);
        note_store_route(object.route(outcome));
        return;
    }
    // The kinds this device retains nothing by ref for. Before reporting, ask
    // the one question that decides whether a retirement could ever belong here:
    // **does this ref name a live entry in the guest's own object list, and is
    // that entry's type the one this destroy opcode names?**
    //
    // The integer resolving proves little on its own — the object table's boot
    // above found 22 sampler-delete refs live under a *different* task, which is
    // a collision and not an identity. The *type* agreeing is the evidence: a
    // function destroy whose ref names an `OBJECT_TYPE_FUNCTION` entry in the
    // same task is one ref space, and one naming a serializer object or nothing
    // is not.
    //
    // Only eight packets a boot reach here, so the list read costs nothing that
    // matters, and it is the difference between "we do not know" and a number.
    note_delete_object_ref_space(
        state,
        host,
        task_id,
        object_ref,
        op.opcode(),
        RefSpacePopulation::Untracked,
    );
    note_unimplemented(
        state,
        channel_id,
        UnimplementedCommand::DeleteObject,
        packet,
    );
}

/// The destroy kinds this device holds `(task, ref)`-keyed state for.
///
/// One list, asked by the census above and matched arm by arm below it. Two
/// spellings of "which kinds are retained" would let a kind join the handler
/// without joining the reading, and the reading is the denominator.
const fn is_retained_kind(opcode: u32) -> bool {
    use reims_vgpu_wire::ops::destroy as d;
    matches!(
        opcode,
        d::OPCODE_DELETE_SAMPLER_STATE
            | d::OPCODE_DELETE_FENCE
            | d::OPCODE_DELETE_DEPTH_STENCIL_STATE
            | d::OPCODE_DELETE_RENDER_PIPELINE_STATE
    )
}

/// Which population of destroys a ref-space reading belongs to.
///
/// Two populations and two sets of routes, because they are two different
/// questions with two different consequences and one bucket could answer
/// neither. The **untracked** kinds ask whether a retirement could ever be
/// keyed on the object list, and eight packets a boot reach it. The
/// **retained** kinds ask something the ordering group needs instead: this
/// device already drops a task-local registry entry for them, and whether that
/// ref is an object-list name decides whether the semantic model can express
/// the retirement as an operation on a resource it knows — or whether the
/// serializer's per-kind spaces are a namespace the model does not have.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RefSpacePopulation {
    /// The kinds this device retains nothing by ref for.
    Untracked,
    /// The kinds this device retires a `(task, ref)`-keyed registry entry for.
    Retained,
}

/// The census route for a destroy whose kind the semantic model cannot name.
///
/// A table rather than a format string, for [`RefSpacePopulation::route`]'s
/// reason: a reader greps these, and a `format!` route is one no grep finds
/// before it has already been read.
fn unnamed_kind_route(opcode: u32) -> &'static str {
    use reims_vgpu_protocol::destroy::DestroyKind as K;
    match K::of(opcode) {
        Some(K::Buffer) => "delete_unnamed_buffer",
        Some(K::Texture) => "delete_unnamed_texture",
        Some(K::DepthStencilState) => "delete_unnamed_depth_stencil_state",
        Some(K::SamplerState) => "delete_unnamed_sampler_state",
        Some(K::Function) => "delete_unnamed_function",
        Some(K::ComputePipelineState) => "delete_unnamed_compute_pipeline_state",
        Some(K::RenderPipelineState) => "delete_unnamed_render_pipeline_state",
        Some(K::Fence) => "delete_unnamed_fence",
        Some(K::Heap) => "delete_unnamed_heap",
        Some(K::RasterizationRateMap) => "delete_unnamed_rasterization_rate_map",
        Some(K::IndirectCommandBuffer) => "delete_unnamed_indirect_command_buffer",
        // The opcode is not a destroy at all, which this function's caller
        // cannot produce — it is reached from the destroy arm alone. Counted
        // rather than asserted, because a census that panics is one nobody
        // leaves switched on.
        None => "delete_unnamed_not_a_destroy",
    }
}

/// What the guest's object list held at a destroy record's ref.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RefSpaceAnswer {
    /// Nothing at all — no collision and no identity.
    NoListEntry,
    /// An entry whose type is the one this destroy opcode names. The evidence
    /// that the two spaces are one.
    TypeAgrees,
    /// An entry of some other type. A collision, which is what a resolving
    /// integer is until the type agrees.
    TypeDiffers,
}

impl RefSpacePopulation {
    /// The census route for one population's answer.
    ///
    /// A table rather than a format string: six counters a reader greps, and
    /// the pair that separates "one namespace" from "an integer collision" is
    /// only readable if all six spellings are written in one place.
    const fn route(self, answer: RefSpaceAnswer) -> &'static str {
        match (self, answer) {
            (Self::Untracked, RefSpaceAnswer::NoListEntry) => "delete_object_ref_no_list_entry",
            (Self::Untracked, RefSpaceAnswer::TypeAgrees) => "delete_object_ref_type_agrees",
            (Self::Untracked, RefSpaceAnswer::TypeDiffers) => "delete_object_ref_type_differs",
            (Self::Retained, RefSpaceAnswer::NoListEntry) => {
                "delete_object_retained_ref_no_list_entry"
            }
            (Self::Retained, RefSpaceAnswer::TypeAgrees) => {
                "delete_object_retained_ref_type_agrees"
            }
            (Self::Retained, RefSpaceAnswer::TypeDiffers) => {
                "delete_object_retained_ref_type_differs"
            }
        }
    }

    /// The census route for whether the semantic model can name this reference.
    ///
    /// Separate from [`Self::route`] because the two answer different
    /// questions and a boot needs both: the list entry says whether the *guest*
    /// still holds the object, and this says whether the *model* still holds
    /// its name. A destroy that arrives after the guest has cleared its slot is
    /// the ordinary case, and it is only a problem if the model has no name for
    /// it either — that is the packet the bridge would have to refuse.
    const fn named_route(self, named: bool) -> &'static str {
        match (self, named) {
            (Self::Untracked, true) => "delete_object_ref_model_named",
            (Self::Untracked, false) => "delete_object_ref_model_unnamed",
            (Self::Retained, true) => "delete_object_retained_ref_model_named",
            (Self::Retained, false) => "delete_object_retained_ref_model_unnamed",
        }
    }
}

/// What a destroy record's ref names in the guest's own object list.
///
/// The type a kind expects is the wire's, not a guess: a function is its own
/// object type and every other retained kind — sampler, depth-stencil, both
/// pipeline states, fence, heap, indirect command buffer, rasterization rate
/// map — is a `SERIALIZER_OBJECT`, which is the tag the object list carries for
/// all of them.
fn note_delete_object_ref_space<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    object_ref: u32,
    opcode: u32,
    retained: RefSpacePopulation,
) -> RefSpaceAnswer {
    use crate::runtime::decode::resource::{
        OBJECT_TYPE_BUFFER, OBJECT_TYPE_FUNCTION, OBJECT_TYPE_SERIALIZER_OBJECT,
        OBJECT_TYPE_TEXTURE,
    };
    use reims_vgpu_wire::ops::destroy as d;
    let expected = match opcode {
        d::OPCODE_DELETE_BUFFER => OBJECT_TYPE_BUFFER,
        d::OPCODE_DELETE_TEXTURE => OBJECT_TYPE_TEXTURE,
        d::OPCODE_DELETE_FUNCTION => OBJECT_TYPE_FUNCTION,
        _ => OBJECT_TYPE_SERIALIZER_OBJECT,
    };
    // Whether the *model* can name this reference, which is a different
    // question from whether the guest's list still holds an entry for it and is
    // the one that decides whether this command can cross the bridge.
    //
    // `objects::name_resource` — which is what the production `RefResolver`
    // answers with — consults `DeviceState::object_name` **before** the guest's
    // list, so a reference the model has already named still resolves after the
    // guest has cleared its slot. Asked read-only here: `name_resource` would
    // declare, and a census that declared would be answering a question it had
    // just changed.
    let named = state.object_name(task_id, object_ref).is_some();
    note_store_route(retained.named_route(named));
    if !named {
        // *Which* kinds the model has no name for, which the two counters above
        // do not say. Closing this population means naming each kind where it
        // is *constructed* — a different site per kind, and a fix aimed at the
        // wrong one buys nothing.
        note_store_route(unnamed_kind_route(opcode));
    }
    let entry = crate::runtime::objects::lookup_list_entry(state, host, task_id, object_ref);
    if !named {
        // **Having no name yet is not the same as being unresolvable, and the
        // difference is the whole refusal population.** `objects::name_resource`
        // — which is what the production `RefResolver` answers with — falls
        // through to the guest's list when the model has no name, so a ref whose
        // entry is still live gets a name taken from it *at that moment* and the
        // packet crosses. Only a ref with neither is one the bridge refuses with
        // `ResolveRefusal::UnknownRef`.
        //
        // The census cannot ask `name_resource` itself: naming declares, and a
        // census that declared would be answering a question it had just
        // changed. Asking the list read-only is the same first step and is what
        // separates the two.
        note_store_route(if entry.is_some() {
            "delete_unnamed_but_list_still_has_it"
        } else {
            "delete_unnamed_and_unresolvable"
        });
    }
    let Some(entry) = entry else {
        note_store_route(retained.route(RefSpaceAnswer::NoListEntry));
        return RefSpaceAnswer::NoListEntry;
    };
    let answer = if entry.object_type == expected {
        RefSpaceAnswer::TypeAgrees
    } else {
        RefSpaceAnswer::TypeDiffers
    };
    let route = retained.route(answer);
    note_store_route(route);
    if crate::observe::first_sight(route, (u64::from(opcode) << 32) | u64::from(object_ref)) {
        crate::observe::fail(format!(
            "{route} kind={} task={task_id} ref={object_ref} found_type={} expected_type={} \
             (whether this destroy record's ref is a number in the guest's object-list space \
             is what decides whether a retirement could ever be keyed on it)",
            delete_object_kind_route(opcode),
            entry.object_type,
            expected
        ));
    }
    answer
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

/// Forget everything cached about a child channel whose lifetime just moved.
///
/// One function because both ends of the lifetime owe it. An open must forget
/// as much as a free: the number is reused, and a hold or a cached ring left
/// over from the previous occupant is the new channel inheriting a decision
/// nothing made about it.
fn forget_child_channel(state: &mut DeviceState, channel: u32, bit: u32) {
    state.translation_deferred_mask &= !bit;
    state.translation_order_hold_mask &= !bit;
    state.present_translation_hold_mask &= !bit;
    state.child_rings[channel as usize] = Default::default();
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
    let page = match crate::protocol::fifo::decode_shared_state(payload) {
        Ok(page) => page,
        Err(short) => {
            note_short_payload("setup_shared_state", channel, &short);
            return;
        }
    };
    let index = page.index;
    let pfn = page.pfn;
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
    // A payload too short to hold an id used to default to `0`, which is the
    // **kernel task** — so a truncated delete retired the resolutions of the
    // one slot whose teardown costs the most, and did it silently.
    let task_id = match crate::protocol::fifo::decode_delete_task(payload) {
        Ok(task_id) => task_id,
        Err(short) => {
            note_short_payload("delete_task", channel, &short);
            return;
        }
    };
    // The task's GVA space goes with it, so every bind resolution keyed on it
    // names bytes that are no longer this task's. Here rather than at the two
    // call sites: the root and child FIFOs both reach this, and a rule written
    // twice is the one that diverges.
    crate::runtime::writeback_debt::retire_gva_for_task(state, task_id);
    note_bb_retired(
        "bb_retire_delete_task",
        state.retire_bound_buffers_for_task(task_id),
    );
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
/// Build the model's operation for a lifetime packet, in this device's own
/// namespaces.
///
/// **One join, not one per arm.** `reims_vgpu_core::lifecycle::operation` is the
/// single place a kind picks which of the five record layouts its payload is,
/// and it is exhaustive over the twelve kinds — so an arm that decoded its own
/// record and built its own operation would be the second copy of that choice,
/// and the copy that reads one command's offsets out of another's payload.
///
/// Both resolvers are this device's, and they are different namespaces: an
/// object-list ref and a mapping id arrive as `u32`s that overlap numerically
/// and name unrelated things. `crate::runtime::objects::TaskNames` answers the
/// first, `DeviceState` the second.
///
/// `None` where the packet's bytes are not the command its opcode names, or a
/// ref names nothing live — both reported here, because an arm that acted on a
/// packet the model could not read is an arm the model has no record of.
fn lifetime_operation<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    channel: reims_vgpu_protocol::packets::Channel,
    opcode: u16,
    payload: &[u8],
) -> Option<reims_vgpu_core::lifecycle::LifecycleOp> {
    let kind = reims_vgpu_core::lifecycle::LifecycleKind::of(channel, opcode)?;
    let names = crate::runtime::objects::TaskNames::new(state, host);
    match reims_vgpu_core::lifecycle::operation(kind, payload, &names, state) {
        Ok(op) => Some(op),
        Err(refusal) => {
            note_store_route(refusal.slug());
            if crate::observe::first_sight(
                "lifetime_operation_unresolved",
                (u64::from(opcode) << 32) | u64::from(kind as u16),
            ) {
                crate::observe::fail(format!(
                    "lifetime_operation_unresolved kind={} op={opcode:#x} refusal={} (the \
                     semantic model cannot read this packet as the command its opcode names, \
                     so it has no record of what this device is about to do)",
                    kind.name(),
                    refusal.slug()
                ));
            }
            None
        }
    }
}

/// Report anything a lifetime command obliged that its arm does not act on.
///
/// **Not a discard.** Each caller has already taken the obligations its command
/// can produce; what reaches here is what the command's own contract says it
/// cannot produce, and a non-empty one means the model has started saying
/// something the device is not listening to. That is a line on the always-on
/// channel, not a silence — the failure it prevents is exactly the one
/// `Effects`' `#[must_use]` exists for, one step further out.
fn note_inert_lifetime_effects(site: &'static str, acted: &crate::model::Acted) {
    let counts = [
        ("teardowns", acted.teardowns.len()),
        ("remapped", acted.remapped.len()),
        ("at_completion", acted.at_completion.len()),
        ("redefined", acted.redefined.len()),
    ];
    for (field, n) in counts {
        if n == 0 {
            continue;
        }
        note_store_route_n("lifetime_effect_unacted", n as u64);
        if crate::observe::first_sight("lifetime_effect_unacted", 0) {
            crate::observe::fail(format!(
                "lifetime_effect_unacted site={site} field={field} n={n} (the model obliged \
                 work this command's arm does not perform, so it is owed to nobody)"
            ));
        }
    }
}

fn apply_set_object_list(state: &mut DeviceState, payload: &[u8], channel: Option<u32>) {
    let crate::protocol::fifo::SetObjectListCommand {
        task_id,
        pfn,
        count,
    } = match crate::protocol::fifo::decode_set_object_list(payload) {
        Ok(command) => command,
        Err(short) => {
            note_short_payload("set_object_list", channel, &short);
            return;
        }
    };
    // A new object list changes the construction input for references this
    // device has not constructed yet. Exact-window buffer resolutions are not
    // object states and therefore retire here. Typed objects already
    // constructed in the task's resource, sampler, and pipeline namespaces keep
    // their explicit delete/task lifetime; this packet does not destroy them.
    // Both FIFOs reach this.
    crate::runtime::writeback_debt::retire_gva_for_task(state, task_id);
    note_bb_retired(
        "bb_retire_set_object_list",
        state.retire_bound_buffers_for_task(task_id),
    );
    let applied = state.set_object_list(task_id, pfn, count);
    // The binding, in the model that owns what a reference resolves to. It
    // retires nothing — a new table changes what an *unresolved* reference
    // constructs, and the objects already declared keep their own deletes —
    // which is the same contract the two retirements above implement on this
    // device's caches. See `LifecycleOp::BindObjectList` for why the operation
    // is the binding rather than the table's walk.
    if let Some(acted) = state.apply_lifetime(
        &reims_vgpu_core::lifecycle::LifecycleOp::BindObjectList {
            task: reims_vgpu_core::identity::TaskId(task_id),
            list: reims_vgpu_core::lifecycle::ObjectList {
                page: reims_vgpu_core::identity::DirectoryFrame(pfn),
                capacity: count,
            },
        },
        "set_object_list",
    ) {
        note_inert_lifetime_effects("set_object_list", &acted);
    }
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
    // The floor, the four offsets and the split of the first word are the
    // command's owner's. Both halves of that word are decoded — the id to index
    // the slot, the class bit so the log says which of the two registered
    // rather than leaving it unaccounted for.
    let command = match crate::protocol::fifo::decode_define_task(payload) {
        Ok(command) => command,
        Err(short) => {
            note_short_payload("define_task2", channel, &short);
            return;
        }
    };
    let raw_id = command.id.to_raw();
    let length = command.length;
    let dir = command.directory_pfn;
    let task_id = command.id.task_id;
    let kernel_task = command.id.kernel;
    // The page-table root is replaced here, so every GVA this task resolved
    // through the old one may translate elsewhere now. Keyed on the shifted
    // `task_id`, not `raw_id` — the registry is keyed the way the draw path
    // keys it, and `raw_id` is the wrong number by a factor of two.
    crate::runtime::writeback_debt::retire_gva_for_task(state, task_id);
    note_bb_retired(
        "bb_retire_define_task2",
        state.retire_bound_buffers_for_task(task_id),
    );
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
    error: crate::protocol::fifo::ResourceListDecodeError,
) {
    crate::observe::Emit::decline("map_family", &error)
        .field("op", op)
        .field("opcode", format!("{opcode:#x}"))
        .field("ch", channel_id)
        .fail_once(u64::from(opcode));
}

/// A fixed-size FIFO payload the guest sent short, reported the way
/// [`packet_short`] reports one.
///
/// **The same line, deliberately.** Twelve arms report a short payload and ten
/// of them still do it by calling `packet_short` with their own literal floor;
/// a reader greps `reason=<command>_short`, and two arms emitting a different
/// slug because their floor moved into a decoder would be a log split down a
/// line nobody drew. What changed for those arms is which code owns the bound,
/// not what the guest's operator sees.
///
/// (The `{op}_short` slugs are *computed*, so `reims-vgpu-observe`'s registry
/// cannot see any of the twelve and the crate-wide "no two checks share a slug"
/// claim excludes them. Fixing that is a change to all twelve at once, not to
/// the two that happen to have decoders.)
/// Report a payload too short for the command it claims to be.
///
/// Takes no opcode. It used to, and no arm ever read it: the reason stem `op`
/// already names the command, and an unused parameter is one every call site
/// has to fill with something a compiler cannot check — three of them would
/// have had to invent a zero.
fn note_short_payload(
    op: &'static str,
    channel: Option<u32>,
    short: &crate::protocol::fifo::ShortPayload,
) {
    crate::observe::fail(format!(
        "packet_short reason={op}_short site={} plen={} need={}",
        packet_site(channel),
        short.plen,
        short.need
    ));
}

fn note_display_txn_payload(
    state: &mut DeviceState,
    channel_id: u32,
    packet: &Packet,
    decoded: &PresentTrailer,
) {
    let plen = packet.payload.len();
    if decoded.undecoded_tail == 0 {
        return;
    }
    let trailer = decoded.form.trailer_len();
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
    ///
    /// **Deferred rather than restated.** The rule belongs to
    /// `reims_vgpu_core::identity::StampValue`, which is the type the ordering
    /// plane decides every stamp wait with, and this device held a second copy
    /// of it written in raw `u32`. Two implementations of a wrapping comparison
    /// are two chances to disagree about which of two numbers is later, and the
    /// disagreement is silent in both directions: one parks a timeline forever
    /// on a value already written, the other releases work early. So the answer
    /// comes from the owner and the paragraph above stays as the reason it is
    /// written that way.
    pub fn satisfied_by(self, current: u32) -> bool {
        reims_vgpu_core::identity::StampValue(current)
            .reached(reims_vgpu_core::identity::StampValue(self.value))
    }
}

/// Why a stamp wait was unmet, partitioned by what this device could have done
/// about it. Census only — see [`StampLedger`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnmetSource {
    /// This device has executed the packet that stamps the awaited value but has
    /// **not called [`write_stamp`] yet** — the coalescing latch is holding it
    /// until the drain ends. Publishing early would discharge this wait, and the
    /// settle on the publication path makes that ordering-safe.
    Coalesced,
    /// [`write_stamp`] has been called with a satisfying value, so the settle has
    /// run and the word is with the engine's completion thread, awaiting the GPU
    /// submission that carries it. **Nothing this device can do makes it ready
    /// sooner**; the work is genuinely still in flight, and holding is correct.
    Queued,
    /// No satisfying value exists anywhere in this device. The guest is waiting
    /// on work it has not yet given us, or that we have not yet executed.
    Absent,
}

/// What this device has stamped, split by whether the value is still *owed* or
/// already handed to the publication rail.
///
/// # Why this exists
///
/// A driven Maps boot leaves 44 % of every stamp wait the guest sends unmet, and
/// each unmet wait abandons a drain for a later re-entry — ~520 round trips a
/// second on the frame's critical path. The question that decides whether any
/// repair is available is *which* unmet waits this device was in a position to
/// answer, and the two cases have opposite answers:
///
/// * [`UnmetSource::Coalesced`] is a value we have and have not published. The
///   coalescing latch defers the write to the end of the drain, and
///   `max_tranche_us` runs 42-91 ms, so the guest can wait tens of milliseconds
///   for a word already sitting in a local. Paying it on demand is a real repair
///   and an ordering-safe one, because the payment goes through [`write_stamp`]
///   and takes its settle with it.
/// * [`UnmetSource::Queued`] is a value already through [`write_stamp`], settled,
///   and waiting on the GPU submission that carries it. There is nothing to
///   publish early. Holding is the correct behavior and the wait is honest.
///
/// An earlier attempt (reverted) answered *both* out of a device-side record.
/// That was unsound for the second: it released packets ahead of the settle,
/// against a fence whose meaning is "guest RAM debts have landed, you may
/// reclaim". This type exists to size the first case before repairing it, rather
/// than to answer anything.
///
/// **This is census only.** Nothing here feeds a verdict; [`note_packet_stamp_waits`]
/// decides exactly as it did before, from the stamp page.
///
/// The three counters partition the unmet total, so
/// `stamp_unmet_coalesced + stamp_unmet_queued + stamp_unmet_absent ==
/// packet_stamp_wait_unmet` is the identity that catches a miscount.
#[derive(Clone, Default, Debug)]
pub struct StampLedger {
    /// Slot → greatest value latched by the coalescing rail and not yet passed
    /// to [`write_stamp`].
    owed: std::collections::BTreeMap<u32, u32>,
    /// Slot → greatest value ever passed to [`write_stamp`], whether or not the
    /// completion thread has put it in the page yet.
    written: std::collections::BTreeMap<u32, u32>,
}

impl StampLedger {
    /// Record that the coalescing rail is holding `value` for `slot`.
    ///
    /// Refuses a slot the stamp page of `page_bytes` cannot hold — the same
    /// refusal [`write_stamp`] makes — which is what bounds both maps by
    /// [`stamp_slot_count`] with nothing to scan for.
    pub fn owe(&mut self, slot: u32, value: u32, page_bytes: u64) {
        Self::fold(&mut self.owed, slot, value, page_bytes);
    }

    /// Record that [`write_stamp`] has been called for `slot` with `value`, and
    /// drop anything the coalescing rail was owing at or below it.
    pub fn wrote(&mut self, slot: u32, value: u32, page_bytes: u64) {
        let slot = stamp_slot_index(slot);
        if stamp_slot_offset(slot, page_bytes).is_none() {
            return;
        }
        Self::fold(&mut self.written, slot, value, page_bytes);
        if self
            .owed
            .get(&slot)
            .is_some_and(|held| (held.wrapping_sub(value) as i32) <= 0)
        {
            self.owed.remove(&slot);
        }
    }

    /// Which of the three cases an unmet `wait` falls in.
    pub fn classify(&self, wait: StampWait) -> UnmetSource {
        let slot = stamp_slot_index(wait.index);
        if self.owed.get(&slot).is_some_and(|v| wait.satisfied_by(*v)) {
            return UnmetSource::Coalesced;
        }
        if self
            .written
            .get(&slot)
            .is_some_and(|v| wait.satisfied_by(*v))
        {
            return UnmetSource::Queued;
        }
        UnmetSource::Absent
    }

    fn fold(
        map: &mut std::collections::BTreeMap<u32, u32>,
        slot: u32,
        value: u32,
        page_bytes: u64,
    ) {
        let slot = stamp_slot_index(slot);
        if stamp_slot_offset(slot, page_bytes).is_none() {
            return;
        }
        map.entry(slot)
            .and_modify(|held| {
                if (value.wrapping_sub(*held) as i32) > 0 {
                    *held = value;
                }
            })
            .or_insert(value);
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
) {
    for wait in &packet.stamp_waits {
        let index = stamp_slot_index(wait.index);
        // Census, deciding nothing: would the ordering plane answer this wait
        // the same way this drain is about to?
        //
        // The two read different things and that is the point. This drain reads
        // the word out of guest RAM, which is authoritative and includes
        // anything the *guest* advanced; the model answers from what has been
        // published to it, which today is only what this device wrote. A wait
        // guest RAM satisfies and the model does not is a packet the cutover
        // would park forever — the same hang an unadvanced pipeline was, on the
        // scheduler's other wait kind — so it is measured before it is relied
        // on rather than discovered by a guest that stops drawing.
        let model_satisfies = state
            .published_completion_stamp(reims_vgpu_core::identity::StampSlot(index))
            .is_some_and(|v| wait.satisfied_by(v.0));
        let against_model = |device_satisfies: bool, agree: &'static str| {
            note_store_route(if device_satisfies == model_satisfies {
                agree
            } else if device_satisfies {
                // The hang: this device runs the packet and the model would
                // hold it against a value nothing will publish again.
                "stamp_wait_model_behind"
            } else {
                // The other direction, and not the harmless one: the model
                // would release work this device is still holding back.
                "stamp_wait_model_ahead"
            });
        };
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
            note_store_route(if model_satisfies {
                "stamp_wait_model_decides_the_unevaluable"
            } else {
                "stamp_wait_model_also_cannot_decide"
            });
            continue;
        };
        // No stamp page means no slot to read and no slot `write_stamp` would
        // write either — it returns early on the same condition.
        if state.gfx.fifo_base_page == 0 {
            unresolvable("no_stamp_page", String::from("fifo_base_page=0"));
            note_store_route(if model_satisfies {
                "stamp_wait_model_decides_the_unevaluable"
            } else {
                "stamp_wait_model_also_cannot_decide"
            });
            continue;
        }
        let gpa = state.pfn_gpa(state.gfx.fifo_base_page) + off;
        let Ok(current) = crate::runtime::host::read_u32(host, gpa) else {
            unresolvable("stamp_slot_unreadable", format!("gpa={gpa:#x}"));
            note_store_route(if model_satisfies {
                "stamp_wait_model_decides_the_unevaluable"
            } else {
                "stamp_wait_model_also_cannot_decide"
            });
            continue;
        };
        // After the census comparison and never before it: publishing first
        // would teach the plane the answer and then congratulate it for
        // knowing. This is the arm that covers the GPU-ordered rail — a word
        // queued there lands without this device writing it, and the drain's
        // own read of the slot is the first place this device *observes* that
        // it has.
        note_stamp_visible(state, index, current, "stamp_visible_observed");
        // **The verdict is the plane's.** It has just been told what the page
        // holds, so it decides on the value this drain read — and the one case
        // where the two can part is a rewind, where the plane refuses to move
        // a timeline backwards while a raw read of the page follows it.
        // `stamp_publish_behind` is that population and it is the witness this
        // needs; a driven boot measured 0 across 29 852 waits, alongside a
        // direct comparison of the two answers that read 0 differences and was
        // removed as a second name for the same event.
        let device_satisfies = state
            .published_completion_stamp(reims_vgpu_core::identity::StampSlot(index))
            .is_some_and(|published| {
                reims_vgpu_core::identity::StampWait {
                    slot: reims_vgpu_core::identity::StampSlot(index),
                    value: reims_vgpu_core::identity::StampValue(wait.value),
                }
                .satisfied_by(published)
            });
        if device_satisfies {
            note_store_route("packet_stamp_wait_met");
            against_model(true, "stamp_wait_model_agrees_met");
            continue;
        }
        note_store_route("packet_stamp_wait_unmet");
        against_model(false, "stamp_wait_model_agrees_unmet");
        // Census only, and it does not change the verdict below. Says whether
        // this device was holding the awaited word (publishable early, and
        // ordering-safe because publication carries the settle), had already
        // handed it to the GPU-ordered rail (nothing to do), or never had it.
        let source = state.stamp_ledger.classify(*wait);
        note_store_route(match source {
            UnmetSource::Coalesced => "stamp_unmet_coalesced",
            UnmetSource::Queued => "stamp_unmet_queued",
            UnmetSource::Absent => "stamp_unmet_absent",
        });
        // A `Queued` wait is on a word already through `write_stamp`, so the
        // settle has run and only the GPU is left — unless the submission it
        // was registered against has not been made, because the batch carrying
        // it is still recording. Then the guest is blocked on this device
        // rather than on the GPU, and submitting ends it without changing any
        // ordering. Asked of the running rail, because whether a stamp is
        // parked in an unsubmitted batch is a fact about that rail's queue and
        // not about which rails this binary compiled. Counted both ways so the
        // split stays visible: a flush that fires is a stall that was real.
        if source == UnmetSource::Queued {
            note_store_route(
                if crate::backend::selected().flush_batch_for_waiting_stamp(index) {
                    "stamp_waiter_flushed_batch"
                } else {
                    "stamp_waiter_already_in_flight"
                },
            );
        }
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
/// Timed at the function for the same reason [`read_ring_bytes`] is: the count
/// this keeps is the packet count both FIFOs are normalized by.
fn decode_packet(
    bytes: &[u8],
    head: u32,
    available: u32,
    ring_capacity: u32,
) -> Result<Packet, PacketError> {
    let started = std::time::Instant::now();
    let out = decode_packet_inner(bytes, head, available, ring_capacity);
    census::note_drain_decode(started.elapsed().as_nanos() as u64);
    out
}

fn decode_packet_inner(
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

// ---------------------------------------------------------------------------
// The ingress switch: arrival, admission, parking, and running what the model
// releases.
//
// The shape this replaces is a loop that ran each packet where it found it and
// stopped the whole timeline when it could not. Head advances unconditionally
// here; a packet that cannot run yet holds an ordering position instead of a
// ring pointer, and the positions behind it are free.
// ---------------------------------------------------------------------------

/// Put the root FIFO's completion word in slot 0, and say whether it landed in
/// the page.
///
/// The root reaches slot 0 inline rather than through [`write_stamp`], and the
/// difference is not cosmetic: it is the one slot whose landing raises the
/// device's own interrupt, so the caller has to be told whether it landed. A
/// word the completion rail took is *queued* — the worker publishes it and
/// announces it in that order — and `false` there is the honest answer rather
/// than an early announcement of an unfinished stamp.
fn write_root_stamp<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    value: u32,
) -> bool {
    if state.gfx.fifo_base_page == 0 {
        return false;
    }
    let Some(off) = stamp_slot_offset(0, state.page_size()) else {
        return false;
    };
    // Recorded here or the ledger would never see slot 0 leave the owed state:
    // the ordering plane's half is further down, at the arm that puts the word
    // in the page, and this line is reached by the queued arm too.
    note_stamp_no_longer_owed(state, 0, value);
    // Root and child FIFOs own the same bounded pending-stamp queue contract,
    // so slot 0 takes the same submission-attached completion rail.
    if crate::backend::selected().order_completion_stamp(
        state,
        host,
        0,
        value,
        crate::runtime::render_writeback::SettleSite::RootStamp,
    ) == crate::backend::StampOrdering::Queued
    {
        note_store_route("root_stamp_ordered_gpu");
        state.completion_stamp_seq = state.completion_stamp_seq.wrapping_add(1);
        return false;
    }
    let gpa = state.pfn_gpa(state.gfx.fifo_base_page) + off;
    if gpa_map::write_u32(host, gpa, value, state.page_size() as usize).is_ok() {
        note_stamp_visible(state, 0, value, "stamp_visible_root");
        // A window armed after this point has outlived a fence the moment it is
        // still armed at the next one. The counter is what `armed_stamp_seq` is
        // compared against, so a rail that does not move it reads as punctual
        // however long it actually waited.
        state.completion_stamp_seq = state.completion_stamp_seq.wrapping_add(1);
        return true;
    }
    // The guest waits on this root completion stamp; a silent writeback failure
    // hangs it forever with no trace (drain.rs Rank-2 audit).
    state.record_fail(FailEvent::MalformedRootPacket {
        fault: PacketFault::RootStampWriteback,
        head: state
            .gfx
            .fifo_read
            .load(std::sync::atomic::Ordering::Acquire),
    });
    false
}

/// Put one published completion word where its domain's guest reads it.
///
/// The two rings write their words differently — the root raises this device's
/// interrupt from slot 0, a child goes through [`write_stamp`] — and the model
/// publishes without knowing either. Routing on the domain here is what lets
/// `complete_transaction`'s releases be written by whoever is holding the drain,
/// including one on another channel entirely.
fn publish_word<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    domain: u32,
    slot: u32,
    value: u32,
) {
    if domain == crate::runtime::ingress::Fifo::ROOT.domain().0 {
        if write_root_stamp(state, host, value) {
            state
                .gfx
                .interrupt_status_gpu
                .fetch_or(1, std::sync::atomic::Ordering::AcqRel);
            host.enqueue(HostAction::irq_gfx());
        }
        return;
    }
    write_stamp(state, host, slot, value);
}

/// Everything the device does with one arrived packet *before* the model
/// judges it.
///
/// Two classes need work at arrival, and both for the same kind of reason: the
/// model's packet cannot be built without a fact only this device can read, and
/// reading it later would read the wrong one.
///
/// * **Exec** — the command buffers live in the task's address space, and the
///   ring head is about to move past the packet that names them. Whoever runs
///   the submission later has to have been handed these bytes.
/// * **Re-point** — the operation names the pages the reference holds *now*,
///   and "now" is after this device has moved the storage incarnation. See
///   `crate::runtime::objects::repointed_storage`: an earlier answer is the old
///   one, so the incarnation move is arrival work by construction.
struct Arrived {
    submission: Option<crate::runtime::exec::ExecSubmission>,
    /// The exec result the reading produced, carrying whatever it refused.
    exec: crate::runtime::exec::ExecResult,
    repointed: Result<
        crate::runtime::ingress::ReplacementStorage,
        crate::runtime::objects::RepointStorageRefusal,
    >,
}

/// Do the arrival work one packet's class needs, and say what it produced.
fn arrival_work<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    fifo: crate::runtime::ingress::Fifo,
    packet: &Packet,
) -> Arrived {
    let mut arrived = Arrived {
        submission: None,
        exec: crate::runtime::exec::ExecResult::default(),
        repointed: Err(crate::runtime::objects::RepointStorageRefusal::Unnamed),
    };
    if fifo.channel() != WireChannel::Child {
        return arrived;
    }
    match packet.opcode {
        CHILD_OP_EXEC_INDIRECT2 => {
            let (submission, exec) =
                crate::runtime::exec::read_exec_submission(state, host, &packet.payload);
            arrived.exec = exec;
            arrived.submission = submission;
        }
        CHILD_OP_REPLACE_PHYSICAL => {
            arrived.repointed = repoint_at_arrival(state, host, &packet.payload);
        }
        _ => {}
    }
    arrived
}

/// Move a re-pointed reference's storage incarnation, and say what it names now.
///
/// **The whole of the re-point's device work, performed at arrival.** The
/// guest's packet announces that the pages under this reference have *already*
/// changed, so every host copy of them is stale the moment it arrives and
/// holding the invalidation until an ordering position would keep serving bytes
/// the guest has replaced. The identity of the new pages is this device's own
/// incarnation counter, not guest memory, so reading it after the move is
/// reading this device's bookkeeping rather than racing the guest.
fn repoint_at_arrival<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    payload: &[u8],
) -> Result<
    crate::runtime::ingress::ReplacementStorage,
    crate::runtime::objects::RepointStorageRefusal,
> {
    let cmd = match crate::protocol::fifo::decode_task_object(payload) {
        Ok(cmd) => cmd,
        Err(_) => return Err(crate::runtime::objects::RepointStorageRefusal::Unnamed),
    };
    // This device's own indexes first, then the incarnation, then the model's
    // term — the order the legacy arm used and for its reasons: the caches hold
    // the window accepted work was planned against and must stop comparing
    // equal before anything reads the new one.
    if crate::runtime::writeback_debt::retire_gva_resource(state, cmd.task_id, cmd.object_id) {
        note_store_route("gva_resource_retired");
    }
    note_bb_retired(
        "bb_retire_replace_physical",
        state.retire_bound_buffers_for_ref(cmd.task_id, cmd.object_id),
    );
    crate::runtime::objects::replace_physical(state, host, cmd.task_id, cmd.object_id);
    crate::runtime::objects::repointed_storage(state, host, cmd.task_id, cmd.object_id).map(
        |(_resource, backing, extent)| crate::runtime::ingress::ReplacementStorage {
            backing,
            extent,
        },
    )
}

/// Give one arrived packet an ordering position and keep its bytes until the
/// model releases it.
///
/// **Head advances whatever this returns**, which is the switch. A packet the
/// model refuses is a packet that did not run, and the ring position it
/// occupied is not the place to record that — the failure channel is.
fn admit_and_park<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    fifo: crate::runtime::ingress::Fifo,
    completion_slot: u32,
    packet: Packet,
) {
    use reims_vgpu_core::identity::StampSlot;

    let arrived = arrival_work(state, host, fifo, &packet);
    let session = state.session_generation();
    let built = crate::runtime::ingress::device_packet(
        state,
        host,
        fifo,
        session,
        StampSlot(stamp_slot_index(completion_slot)),
        &packet,
        crate::runtime::ingress::PacketReads {
            submission: arrived.submission.as_ref(),
            repointed: arrived.repointed,
        },
    );
    let mut built = match built {
        Ok(built) => built,
        Err(blocked) => {
            note_unadmitted(
                state,
                host,
                fifo,
                completion_slot,
                &packet,
                blocked.slug(),
                || format!("{blocked:?}"),
            );
            return;
        }
    };
    // A wait naming a slot this device's stamp page cannot hold is one **no
    // drain could ever satisfy**: `write_stamp` returns early on the same
    // `stamp_slot_offset` that refuses here, so the slot is never written and a
    // transaction admitted waiting on it would hold its channel's publication
    // head forever.
    //
    // The asymmetry decides it, and it is the same one the legacy path stated:
    // an ordering slip loses one packet's ordering, while a position parked on
    // a wait nothing can satisfy loses the guest. So the wait is dropped and
    // named, here rather than in the bridge — which slot numbers exist is a
    // property of this device's page and not of the packet.
    let page_size = state.page_size();
    built.stamp_waits.retain(|wait| {
        if stamp_slot_offset(wait.slot.0, page_size).is_some() {
            return true;
        }
        note_store_route("packet_stamp_wait_unresolvable");
        if crate::observe::first_sight("packet_stamp_wait_unresolvable", u64::from(wait.slot.0)) {
            crate::observe::fail(format!(
                "packet_stamp_wait_unresolvable reason=stamp_slot_out_of_range opcode={:#x} \
                 index={} slots={} (no drain writes a slot outside the stamp page, so the \
                 packet is admitted unordered against this wait rather than parking forever)",
                packet.opcode,
                wait.slot.0,
                stamp_slot_count(page_size)
            ));
        }
        false
    });
    note_access_modes(state, &built);
    // The pipelines the records bind, told to the model before it is asked
    // whether the packet may run: `PipelineTable::waits_for` refuses a lease it
    // has no entry for, and the entry is this device's to make. The list is the
    // walk's own answer and not a second scan, so what is declared cannot
    // disagree with what the transaction is then admitted waiting on.
    let leases: &[reims_vgpu_core::identity::ResourceId] = built
        .payload
        .exec()
        .map_or(&[], |work| &work.pipeline_leases);
    // **Asked here and not at arrival, because here is where the leases are.**
    // Admission readies exactly this list on the rail's answer, so the question
    // the rail is asked has to be about exactly this list. It used to be asked
    // before the walk had run, and answered from a second scan of the packet's
    // bytes for `SetPipeline` records — a set that need not be this one. A lease
    // readied on an answer that never examined it is a transaction released
    // against a shader that is still translating, which is the
    // `m2v_translation_pending_at_sync_boundary` loss measured on a driven
    // macos-15 desktop.
    //
    // **Declared before the rail pre-scans, not after.** The pre-scan is where
    // a rail first holds a translated shader, and publishing what that shader
    // reads is refused for a pipeline the table does not yet hold — so
    // declaring afterwards would make the model's own stated order (publish,
    // then ready) unreachable on the packet that first builds the kernel. The
    // step is the guest's fact and the walk's own list; taking it earlier
    // states the same thing at the first moment it is known.
    for &lease in leases {
        note_store_route(if state.declare_pipeline(lease) {
            "pipeline_declared"
        } else {
            "pipeline_declared_already"
        });
    }
    let translating = match (arrived.submission.as_ref(), built.payload.exec()) {
        (Some(submission), Some(resolved)) => {
            let mut measured_ns = 0u64;
            crate::runtime::exec::preflight_submission(
                state,
                host,
                submission,
                resolved,
                &mut measured_ns,
            )
        }
        _ => Vec::new(),
    };
    for &lease in leases {
        if translating.contains(&lease.slot.0) {
            withdraw_lease(state, lease);
        } else if translating.is_empty() {
            ready_lease(state, lease, "pipeline_lease_ready_admission");
        }
    }

    let admission = match state.admit_packet(&built) {
        Ok(admission) => admission,
        Err(refusal) => {
            note_unadmitted(
                state,
                host,
                fifo,
                completion_slot,
                &packet,
                refusal.slug(),
                || format!("{refusal:?}"),
            );
            return;
        }
    };
    let mut transaction = admission.admitted.transaction;
    let ingress = transaction.identity.ingress;
    // The resolved records move out of the transaction the model just handed
    // back. `admit` keeps a transaction's accesses in its dependency graph and
    // returns everything else, and nothing downstream of this line reads the
    // transaction again — so taking the work here is taking the only copy there
    // is, and the position's pipeline leases become a field of it rather than a
    // second list that could disagree with it.
    let resolved = match &mut transaction.payload {
        reims_vgpu_core::transaction::Payload::Exec(work) => Some(std::mem::take(work)),
        _ => None,
    };
    // Both halves or neither: the command buffers are what the packet was
    // admitted *against* and the resolved records are what it was admitted
    // *as*, and a position holding one without the other could plan from bytes
    // its records never named. An exec packet has both; every other class has
    // neither.
    let work = match (arrived.submission, resolved) {
        (Some(submission), Some(resolved)) => crate::runtime::parked::ParkedWork::with_submission(
            fifo.domain().0,
            admission.epoch,
            packet,
            submission,
            resolved,
        ),
        _ => crate::runtime::parked::ParkedWork::new(fifo.domain().0, admission.epoch, packet),
    };
    state.parked.park(ingress, work);
}

/// How much of this packet's ordering is bought by knowing, and how much by not
/// knowing.
///
/// [`reims_vgpu_core::access::AccessMode::Unknown`] is a distinct variant from
/// `ReadWrite` for one reason, and that reason is a number: the two order
/// identically, so the only thing the split can buy is the ability to say how
/// many edges exist because a slot is genuinely read-modify-write and how many
/// exist because no reflection was published for the pipeline that bound it.
/// The variant has carried that argument since it was written and nothing
/// counted it, which made Seam 6's "supported operations reaching an unknown
/// access-mode fallback" unmeasurable rather than met.
///
/// Counted at admission, from the transaction the model is about to be handed,
/// because that is the list the dependency graph will actually order against —
/// a count taken at the encoder would include intents a payload never carried.
///
/// See fn `an_unknown_mode_access_is_counted_apart_from_a_known_one`.
pub(crate) fn note_access_modes(state: &DeviceState, built: &reims_vgpu_core::session::Packet) {
    use reims_vgpu_core::access::AccessMode;
    for access in built.payload.accesses() {
        note_store_route(match access.mode {
            AccessMode::Read => "access_mode_read",
            AccessMode::Write => "access_mode_write",
            AccessMode::ReadWrite => "access_mode_read_write",
            AccessMode::Unknown => "access_mode_unknown",
        });
    }
    // Why the `Unknown`s above are `Unknown`, from the walk that decided it.
    //
    // The total is what the ordering plane is judged on and it says nothing
    // about what to fix. A draw the walk never saw bind a pipeline and a draw
    // whose shader the rail could not reflect both contribute every slot they
    // bind, and the two have nothing in common: the first is a hole in the
    // walk's own encoder state, the second is a contract term the rail has not
    // recovered. Reported apart so the next move is aimed rather than guessed.
    let Some(work) = built.payload.exec() else {
        return;
    };
    let census = &work.usage_census;
    for (route, n) in [
        ("draw_usage_no_pipeline", census.draws_without_a_pipeline),
        ("draw_usage_no_stage", census.draws_with_no_stage_published),
        (
            "draw_usage_one_stage",
            census.draws_with_one_stage_published,
        ),
        (
            "draw_usage_both_stages",
            census.draws_with_both_stages_published,
        ),
        (
            "dispatch_usage_no_pipeline",
            census.dispatches_without_a_pipeline,
        ),
        (
            "dispatch_usage_no_stage",
            census.dispatches_with_no_stage_published,
        ),
        ("dispatch_usage_published", census.dispatches_published),
        (
            "unknown_from_a_table",
            census.unknown_participations_from_a_table,
        ),
        (
            "unknown_with_no_table",
            census.unknown_participations_with_no_table,
        ),
    ] {
        note_store_route_n(route, u64::from(n));
    }
    // Which pipeline, and what the table thinks of it. The counts above say a
    // draw found nothing published; only the pipeline's *state* tells the two
    // causes apart — an executor that has not answered yet, and one whose
    // answer a withdrawal or a retirement threw away. Once per slot.
    if let Some(id) = census.first_unanswered_pipeline {
        if crate::observe::first_sight("draw_usage_no_stage", u64::from(id.slot.0)) {
            crate::observe::fail(format!(
                "draw_usage_no_stage slot={} gen={} model_pipeline={} (a draw bound this \
                 pipeline and the ordering plane had no reflection for it, so every slot \
                 the draw binds ordered against every other access to the same resource)",
                id.slot.0,
                id.generation.0,
                state.pipeline_state(id).unwrap_or("unleased"),
            ));
        }
    }
}

/// Advance what the model is waiting on, then run everything it has released.
///
/// Called after each admission and at the end of each drain, on both rings. It
/// is the whole of the second half of the switch: the loop above puts packets
/// into the model and this takes work out of it, and neither is a function of
/// where a ring's head happens to be.
fn settle_model_work<H: HostMemory + HostOps>(state: &mut DeviceState, host: &mut H) {
    // **Until nothing more runs, not once.** Running a position publishes its
    // channel's completion words, and a published word discharges the stamp
    // waits other positions were admitted with — so one pass over what was
    // ready when the pass began leaves the work it just released for whenever
    // this device is next entered. A guest that ordered a packet behind a fence
    // this very drain published would wait on a doorbell instead.
    loop {
        observe_awaited_stamps(state, host);
        pump_translations(state, host);
        for ingress in state.take_ready() {
            // `false` is a position the model released and this device holds no
            // bytes for. `mark_ready` has already named it; there is nothing
            // here to run.
            let _ = state.parked.mark_ready(ingress);
        }
        let mut ran = false;
        for ingress in state.parked.ready_in_order() {
            if declined_by_the_device(state, ingress) {
                continue;
            }
            let Some(work) = state
                .parked
                .release(ingress, crate::runtime::parked::Release::Ready)
            else {
                continue;
            };
            run_parked(state, host, ingress, &work);
            ran = true;
        }
        if !ran {
            break;
        }
    }
    // A position still parked is work this device owes, and the ring it came
    // from is empty — so nothing but a re-entry will run it. Asking the store
    // rather than a mask is what keeps "which timelines are owed" one fact:
    // the store is what holds the work.
    for ingress in state.parked.waiting_in_order() {
        match state.parked.domain_of(ingress) {
            Some(0) => state.pending.main_drain = true,
            Some(domain) => state.pending.child_mask |= 1u32.checked_shl(domain).unwrap_or(0),
            None => {}
        }
    }
}

/// Whether this device is not able to run a released position yet.
///
/// **The gates the model does not model, and cannot.** Host paint being two
/// presents behind is a property of this device's display rail, not of the
/// packet's ordering, and the model has no term for it. A declined position
/// keeps its place and is offered again — `ready_in_order` is a list for
/// exactly this reason — so declining one costs the positions behind it
/// nothing, which is the head-of-line stall the switch removes.
fn declined_by_the_device(
    state: &mut DeviceState,
    ingress: reims_vgpu_core::identity::IngressOrdinal,
) -> bool {
    let Some(opcode) = state.parked.opcode(ingress) else {
        return false;
    };
    if !matches!(
        opcode,
        CHILD_OP_DISPLAY_SWAP | CHILD_OP_DISPLAY_TRANSACTION2 | CHILD_OP_DISPLAY_TRANSACTION3
    ) {
        return false;
    }
    if state.present.unpainted_presents < MAX_UNPAINTED_PRESENTS {
        return false;
    }
    note_store_route("present_backpressure_parked");
    // Paint will schedule the next worker slice. Preserve the domain without
    // self-waking the worker ahead of QEMU's action BH.
    if let Some(domain) = state.parked.domain_of(ingress) {
        note_present_backpressure_hold(state, domain, ingress);
        state.pending.child_mask |= 1u32.checked_shl(domain).unwrap_or(0);
    }
    true
}

/// Read the completion slots parked positions are waiting on, and tell the
/// ordering plane what the guest's page holds.
///
/// **Without this a queued word is never observed and its waiters never run.**
/// `note_stamp_visible`'s own doc names three places a word is seen to have
/// landed: the two arms that write it inline, and a drain's read of the slot
/// while evaluating a wait. The third covered the GPU-ordered rail, whose word
/// lands when a submission retires and not when this device hands it over — and
/// it used to be enough because it ran for every arriving packet that carried a
/// wait.
///
/// It is not enough now. A parked position's wait is evaluated once, at
/// admission; if the word it waits for is queued and no further packet with
/// waits arrives, nothing looks at the slot again and the position waits
/// forever. So the slots parked work is watching are read every pass, which is
/// the same observation at the moment it has become the only one.
fn observe_awaited_stamps<H: HostMemory + HostOps>(state: &mut DeviceState, host: &mut H) {
    if state.gfx.fifo_base_page == 0 {
        return;
    }
    let waiting = state.parked.waiting_in_order();
    if waiting.is_empty() {
        return;
    }
    let mut seen: Vec<u32> = Vec::new();
    for ingress in waiting {
        for index in state.parked.awaited_slots(ingress) {
            let index = stamp_slot_index(index);
            if seen.contains(&index) {
                continue;
            }
            seen.push(index);
            // A slot past the stamp page names a FIFO this device does not
            // have, so no drain could ever write it. `admit` already holds the
            // position; nothing here can decide it and nothing here pretends to.
            let Some(off) = stamp_slot_offset(index, state.page_size()) else {
                note_store_route("awaited_slot_out_of_range");
                continue;
            };
            let gpa = state.pfn_gpa(state.gfx.fifo_base_page) + off;
            let Ok(current) = crate::runtime::host::read_u32(host, gpa) else {
                note_store_route("awaited_slot_unreadable");
                continue;
            };
            note_stamp_visible(state, index, current, "stamp_visible_observed");
        }
    }
}

/// Try to finish the translations every parked position is still waiting on.
///
/// **The only thing that can release an exec packet parked on its own
/// pipelines.** This device builds a pipeline inside the packet that binds it,
/// so a transaction admitted while its shaders were still translating would
/// wait forever on a build that only its own execution starts. The plan step is
/// the way out: it is pure CPU work over bytes the parked position already
/// holds, it takes a shared borrow, and when it answers `false` the rail has
/// promised the packet can be executed to completion now.
///
/// Nothing here runs anything. It moves pipelines to `Ready`, and readying a
/// pipeline is what the model turns into released work.
fn pump_translations<H: HostMemory + HostOps>(state: &mut DeviceState, host: &mut H) {
    for ingress in state.parked.waiting_in_order() {
        let Some((submission, resolved)) = state.parked.planning(ingress) else {
            continue;
        };
        let leases = resolved.pipeline_leases.as_slice();
        // A position whose every lease is already `Ready` has had the rail's
        // promise made for it, and this pass cannot make it again: `ready_lease`
        // would find every step illegal and `preflight_submission` would re-walk
        // the same records to reach the same memo. It is still parked for
        // something else — a stamp, or a dependency — and neither is this
        // function's to move. Skipping it is what makes the pump proportional to
        // the pipelines still building rather than to the parked population
        // times the passes: a partial boot measured
        // `parked_translations_finished = 282 172` against 137 pipelines
        // declared and readied, which is the same handful of plans re-walked
        // thousands of times.
        if leases.iter().all(|&lease| state.pipeline_is_ready(lease)) {
            note_store_route("parked_translations_already_ready");
            continue;
        }
        let mut measured_ns = 0u64;
        let pending = crate::runtime::exec::preflight_submission(
            state,
            &*host,
            submission,
            resolved,
            &mut measured_ns,
        );
        if !pending.is_empty() {
            // The pump's own withdrawal arm. A position can be parked on one
            // pipeline while another it binds is `Ready` from an earlier
            // packet, and the guest can rewrite that other one's shader while
            // this position waits — so the pass that finds the first still
            // building is also where the second stops being usable.
            for &lease in leases {
                if pending.contains(&lease.slot.0) {
                    withdraw_lease(state, lease);
                }
            }
            continue;
        }
        let leases = leases.to_vec();
        note_store_route("parked_translations_finished");
        for lease in leases {
            ready_lease(state, lease, "pipeline_lease_ready_pump");
        }
    }
}

/// Run one released position from the bytes it was admitted with, then tell the
/// model it finished and write whatever its channel published.
///
/// The arms are unchanged: what moved is that they are reached from a retained
/// packet at an ordering position rather than from the ring at a head, and that
/// the completion word is the model's answer rather than a value this function
/// decides.
fn run_parked<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    ingress: reims_vgpu_core::identity::IngressOrdinal,
    work: &crate::runtime::parked::ParkedWork,
) {
    let domain = work.domain();
    if domain == crate::runtime::ingress::Fifo::ROOT.domain().0 {
        process_root_packet(state, host, work.packet());
    } else {
        // The re-entry guard this packet used to be run under. A child packet's
        // arm can reach `drain_other`, and a channel whose own drain is on the
        // stack must not be re-entered — which used to be true because the
        // packet ran inside that channel's drain, and is now something this has
        // to state, since a released position may belong to a channel nobody is
        // draining.
        let bit = 1u32.checked_shl(domain).unwrap_or(0);
        let was_draining = state.draining_mask & bit;
        let prev_channel = state.draining_channel;
        state.draining_channel = domain;
        state.draining_mask |= bit;
        let started = std::time::Instant::now();
        let _ = process_child_packet(state, host, domain, work.packet(), work.retained());
        census::note_drain_proc(work.packet().opcode, started.elapsed().as_nanos() as u64);
        if was_draining == 0 {
            state.draining_mask &= !bit;
        }
        state.draining_channel = prev_channel;
    }
    match state.complete_transaction(work.epoch(), ingress) {
        Ok(released) => {
            for release in released {
                // **Counted here because the model cannot count it, and a
                // publisher nobody counts reads exactly like one that never
                // runs.** `SessionModel::complete` has already put this stamp
                // into the scheduler's published map — that is what a channel
                // publishing means — so the ordering plane's timeline moves on
                // this line and on no other. `note_stamp_visible`'s routes
                // report the *page*, which the queued rail writes later, and a
                // reading that had only those concluded the plane's timeline
                // had frozen: `stamp_publish_first = 1` and
                // `stamp_publish_advanced = 1` for a whole driven boot, because
                // by the time the page was read the entry was always already
                // there.
                //
                // So the two publishers are counted apart, and the pair is what
                // makes `stamp_publish_behind` readable: this is the numerator
                // the page's observations are behind.
                match release.stamp {
                    Some(stamp) => {
                        note_store_route("stamp_published_by_channel");
                        publish_word(state, host, domain, stamp.slot.0, stamp.value.0);
                    }
                    // A position whose channel released it owing no word. It is
                    // the denominator: without it a channel that published
                    // nothing and a channel that released nothing read alike.
                    None => note_store_route("stamp_released_without_a_word"),
                }
            }
        }
        // The incarnation the work was submitted under has ended. Its
        // withdrawal already released whatever was queued behind it, so there
        // is nothing to publish and the name is the whole of what is owed.
        Err(refusal) => note_store_route(refusal.slug()),
    }
}

/// Take back the promise that a lease is usable, because this rail no longer
/// holds a translation for it.
///
/// The other half of [`ready_lease`], and the half the table had no step for
/// until `77f5bd88`. `Ready` used to mean "was translated once", which is not
/// the question a transaction about to bind it is asking: the guest rewrites
/// the shader behind a live pipeline ref **in place**, so no delete arrives, no
/// generation is minted, the [`reims_vgpu_core::identity::ResourceId`] is
/// unchanged — and the rail's translate cache, being keyed by the shader's
/// content, stops holding one. Four refs a driven macos-15 desktop, named by
/// `m2v_pipe_content_changed`.
///
/// Silent when the lease was already waiting, which is the common case by three
/// orders of magnitude: this runs for every pending ref of every pre-scan, and
/// almost all of those are a cold pipeline that has simply not finished. What
/// is worth a line is the *withdrawal* — a lease that was `Ready` and is not
/// any more — and `PipelineTable::advance` is what distinguishes them, so the
/// counter is read off the table's own census rather than guessed here.
fn withdraw_lease(state: &DeviceState, lease: reims_vgpu_core::identity::ResourceId) {
    if state.withdraw_pipeline(lease) {
        note_store_route("pipeline_lease_withdrawn");
        if crate::observe::first_sight("pipeline_lease_withdrawn", u64::from(lease.slot.0)) {
            crate::observe::fail(format!(
                "pipeline_lease_withdrawn slot={} gen={} (this rail stopped holding a \
                 translation for a pipeline the model called ready; the guest rewrote its \
                 shader without renaming it, and work binding it waits again)",
                lease.slot.0, lease.generation.0
            ));
        }
    }
}

/// Step a lease to `Ready`, through every state between.
///
/// `PipelineState::may_become` is a table and `Declared -> Ready` is not in it,
/// so becoming usable is three steps whoever takes them. They are taken here
/// because this is where the answer arrives: the rail's plan step has promised
/// the packet can be executed to completion now, and that promise is exactly
/// what a lease wait needs to know.
fn ready_lease(
    state: &DeviceState,
    lease: reims_vgpu_core::identity::ResourceId,
    site: &'static str,
) {
    use reims_vgpu_core::pipeline::PipelineState;
    state.advance_pipeline(lease, PipelineState::Translating);
    state.advance_pipeline(lease, PipelineState::Compiling);
    if state.ready_pipeline(lease) {
        note_store_route("pipeline_ready");
        // Which of the three answers put this lease at `Ready`, once per slot
        // per site.
        //
        // A draw has been measured reaching a shader that was still
        // translating while the model called its lease ready
        // (`m2v_translation_pending_at_sync_boundary` with
        // `model_pipeline=ready`), and the three sites that can say ready —
        // admission, the pump, and the rail's own resolve — are indistinguishable
        // in the aggregate counter. The one that answered first is the one whose
        // question was wrong.
        if crate::observe::first_sight(site, u64::from(lease.slot.0)) {
            crate::observe::off(format!(
                "pipeline_lease_ready site={site} slot={} gen={}",
                lease.slot.0, lease.generation.0
            ));
        }
    }
}

/// A packet the model would not take, named on the always-on channel and
/// answered anyway.
///
/// **The compromise, stated.** The guest polls the completion word this packet
/// carries, so a refusal that published nothing would hang whatever is waiting
/// on it — and there is no ordering position to publish through, because the
/// packet never got one. So the word is written out of band and counted, which
/// says "this device dropped your work" rather than saying nothing at all.
///
/// Every route into here measures **zero** on the five rails driven, which is
/// what makes an out-of-band write the lesser of the two: a refusal that never
/// happens costs nothing, and a hang that does is unrecoverable.
fn note_unadmitted<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    fifo: crate::runtime::ingress::Fifo,
    completion_slot: u32,
    packet: &Packet,
    reason: &'static str,
    // The refusal's own value, rendered only when the line is written. The two
    // call sites hold different types and neither is worth a formatted string
    // on a path that is silent after its first sighting.
    detail: impl FnOnce() -> String,
) {
    note_store_route("packet_unadmitted");
    note_store_route(reason);
    // Latched under the *refusal's* own slug rather than under
    // `packet_unadmitted`, because `first_sight` keys on the reason it is given
    // and one opcode refuses for several. A driven macos-26 boot's 918
    // unadmitted packets were 914 `pipeline_absent` and 4
    // `resolve_ref_names_no_object`, every one of them child `0x37` — and under
    // one slug the log printed whichever arrived first and hid the rest.
    if crate::observe::first_sight(
        reason,
        (u64::from(fifo.domain().0) << 32) | u64::from(packet.opcode),
    ) {
        crate::observe::fail(format!(
            "packet_unadmitted ch={} opcode={:#x} reason={reason} refusal={} (the \
             ordering plane would not take this packet, so its work did not run; its \
             completion word is written out of band because the guest polls it and no \
             ordering position exists to publish it through)",
            fifo.domain().0,
            packet.opcode,
            detail()
        ));
    }
    publish_word(
        state,
        host,
        fifo.domain().0,
        completion_slot,
        packet.completion_stamp,
    );
}

/// A FIFO's bytes, and how this device reaches them.
///
/// The root ring is one contiguous guest-physical span; a child ring is a page
/// list walked at the device's page shift. **That is the whole of the
/// difference between the two drains' reads**, and naming it is what lets one
/// arrival step serve both rings instead of each keeping its own copy of
/// header-snapshot-decode.
///
/// The copies had already drifted once — see [`packet_snapshot_len`], whose own
/// doc records the arm one ring grew and the other did not — which is the
/// reading this type exists to stop repeating.
pub enum Ring<'a> {
    /// The device's own FIFO: one span, from a base GPA.
    Root { base_gpa: u64, capacity: u32 },
    /// A guest-defined channel's FIFO: a page list, walked at `page_shift`.
    Child {
        page_gpas: &'a [u64],
        capacity: u32,
        page_shift: u32,
    },
}

impl Ring<'_> {
    /// The ring's data length, which is the modulus every head is taken under.
    #[must_use]
    pub const fn capacity(&self) -> u32 {
        match *self {
            Self::Root { capacity, .. } | Self::Child { capacity, .. } => capacity,
        }
    }

    fn read<M: HostMemory>(&self, mem: &M, absolute: u32, len: u32) -> Result<Vec<u8>, MemError> {
        match *self {
            Self::Root { base_gpa, capacity } => {
                read_ring_bytes(mem, base_gpa, capacity, absolute, len)
            }
            Self::Child {
                page_gpas,
                capacity,
                page_shift,
            } => read_child_ring_bytes(mem, page_gpas, capacity, absolute, len, page_shift),
        }
    }

    /// The faults a failed header read and a failed snapshot read are reported
    /// as, which are the two names each ring already had for them.
    const fn read_faults(&self) -> (PacketFault, PacketFault) {
        match *self {
            Self::Root { .. } => (PacketFault::RootHeaderRead, PacketFault::RootSnapRead),
            Self::Child { .. } => (PacketFault::ChildHeaderRead, PacketFault::ChildSnapRead),
        }
    }
}

/// What one look at a ring's consumer pointer found.
///
/// Three answers and not more, because a drain does three things with them: run
/// the packet, stop quietly, or stop and say why.
pub enum Arrival {
    /// A whole packet, decoded out of a snapshot of the ring.
    Packet(Packet),
    /// Nothing to run, and nothing to report.
    ///
    /// The head has caught the tail, or the producer is mid-write: fewer bytes
    /// published than a header, or a header whose `total_size` names bytes that
    /// are not there yet. **A partial packet is the normal state of a producer**
    /// — see [`PacketError::fault`], which is where that carve-out is made
    /// mechanical — so all three are one answer and none of them is a line in
    /// the log.
    Nothing,
    /// A registered fault. The drain stops and names it on its own ring's
    /// event; which event that is stays the caller's, because the two rings
    /// report through different `FailEvent` variants.
    Fault(PacketFault),
}

/// Take the next packet at `head`, out of whichever ring this is.
///
/// The step both drain loops used to spell for themselves: bound the published
/// bytes, read the header, work out how much of the packet is safe to snapshot,
/// read that, and decode it. Every one of those five is shared, and the two
/// copies differed only in which reader they called and which `FailEvent` they
/// reported through — the first of which is [`Ring`]'s and the second of which
/// stays the caller's.
///
/// The head is passed rather than read here on purpose: the root ring keeps its
/// consumer pointer in an atomic this device also publishes to, and a step that
/// loaded it itself would be a second reader of a value the caller is in the
/// middle of advancing.
pub fn arrival<M: HostMemory>(ring: &Ring<'_>, mem: &M, head: u32, tail: u32) -> Arrival {
    if head == tail {
        return Arrival::Nothing;
    }
    let capacity = ring.capacity();
    let Some(available) = published_byte_count(head, tail, capacity) else {
        return Arrival::Fault(PacketFault::DesyncedHeadTail);
    };
    if available < PACKET_HEADER_LEN {
        return Arrival::Nothing;
    }
    let (header_fault, snap_fault) = ring.read_faults();
    let Ok(header) = ring.read(mem, head, PACKET_HEADER_LEN) else {
        return Arrival::Fault(header_fault);
    };
    let snap_len = packet_snapshot_len(&header, available, capacity);
    let Ok(snap) = ring.read(mem, head, snap_len) else {
        return Arrival::Fault(snap_fault);
    };
    match decode_packet(&snap, head, available, capacity) {
        Ok(packet) => Arrival::Packet(packet),
        Err(err) => err.fault().map_or(Arrival::Nothing, Arrival::Fault),
    }
}

/// Timed at the function rather than at its call sites, so both FIFOs and both
/// reads per packet are counted and a further call site cannot be added without
/// being measured.
fn read_ring_bytes<M: HostMemory>(
    mem: &M,
    base_gpa: u64,
    ring_size: u32,
    absolute: u32,
    len: u32,
) -> Result<Vec<u8>, MemError> {
    let started = std::time::Instant::now();
    let out = read_ring_bytes_inner(mem, base_gpa, ring_size, absolute, len);
    census::note_drain_ring(started.elapsed().as_nanos() as u64);
    out
}

fn read_ring_bytes_inner<M: HostMemory>(
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
pub(super) fn note_stamp_direction<H: HostMemory + HostOps>(
    host: &H,
    gpa: u64,
    index: u32,
    value: u32,
) {
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

/// Record that a completion word is no longer owed by the coalescing rail.
///
/// **One function because there are two writers of slot 0's history and they
/// have already drifted once.** `write_stamp` is the child FIFOs' door; the
/// root FIFO reaches its own slot inline a few lines below its dispatch, and
/// the comment there records why it has to say so separately.
///
/// This is *not* the ordering plane's event. Handing a word to a rail is
/// submission, and submission is not completion — see [`note_stamp_visible`],
/// which is where the plane is told.
fn note_stamp_no_longer_owed(state: &mut DeviceState, index: u32, value: u32) {
    let page_bytes = state.page_size();
    state.stamp_ledger.wrote(index, value, page_bytes);
}

/// Tell the ordering plane a completion word is **readable by the guest**.
///
/// # Why this is not where the word is handed to a rail
///
/// It was, for one commit, and a driven boot measured the cost exactly:
/// `stamp_wait_model_ahead = 12 824` against `stamp_unmet_queued = 12 824`,
/// the same number twice. `write_stamp` hands the word to the GPU-ordered
/// rail and returns; the word lands when that rail's submission retires,
/// which is later, and every wait in between is one the model called
/// satisfied while the guest could still read the old value. A cutover
/// publishing there would release, twelve thousand times a boot, work ordered
/// behind GPU work that had not finished. Submission is not completion.
///
/// So the plane is told at the three places a word is *observed to have
/// landed*: the two arms that write it into the page inline, and the drain's
/// own read of the slot when it evaluates a wait — which is what covers the
/// queued arm, because a queued word nobody waits on is one no admitted
/// transaction is ordered behind.
///
/// `site` names which of the three, so the split stays readable: a boot where
/// the observed arm carries everything is a boot where the inline arms have
/// stopped running. A driven macos-26 desktop is that boot:
/// `stamp_visible_observed = 1 700 804` against `stamp_visible_inline = 25`,
/// because `write_stamp` hands almost every word to the GPU-ordered rail.
///
/// # This is not the plane's only publisher, and the other one is why `Behind`
/// exists
///
/// `reims_vgpu_core::session::SessionModel::complete` publishes the stamp its
/// channel released, straight into the scheduler, at the moment the channel
/// publishes it. The model cannot count a route, so that publisher used to be
/// invisible in a store-route dump — which is why a boot read
/// `stamp_publish_first = 1` while three slots had entries: the entries came
/// from completions, and this function only ever found them already there. Its
/// caller counts it now, as `stamp_published_by_channel`, beside
/// `stamp_released_without_a_word` for a position whose channel released it
/// owing no word.
///
/// So the two are asking different questions of one slot. The plane holds what
/// the *channel* published; this reports what the *page* carries. Between them
/// sits the queued rail's write, and a slot read inside that window answers
/// [`crate::model::StampPublication::Behind`]. It is a latency and not a
/// rewind: the same boot measured `stamp_write_forward = 2203` against
/// `stamp_write_backward = 1`.
fn note_stamp_visible(state: &DeviceState, index: u32, value: u32, site: &'static str) {
    note_store_route(site);
    note_store_route(
        match state.publish_completion_stamp(
            reims_vgpu_core::identity::StampSlot(index),
            reims_vgpu_core::identity::StampValue(value),
        ) {
            crate::model::StampPublication::First => "stamp_publish_first",
            crate::model::StampPublication::Advanced => "stamp_publish_advanced",
            crate::model::StampPublication::Repeat => "stamp_publish_repeat",
            // The plane's timeline refused to move and something has put a
            // smaller value in the page. `note_stamp_direction` sees the same
            // event on the arms that write the word; this sees it on every arm
            // the guest can read from.
            //
            // **Named, once per slot, because the count is two different
            // facts.** A boot measures about 14 000 of these, and the driven
            // boot taken when the publication point was moved to this site read
            // the same counter at zero — so either the meaning drifted or the
            // invariant broke, and a bare count cannot say
            // which. What separates them is *which slot* and *by how much*: one
            // slot whose page rewound under a timeline that will not follow it
            // is a slot to explain, and a timeline running ahead of every word
            // the guest can read is an ordering defect. The pair is on the line.
            //
            // **The meaning drifted, and a driven macos-26 desktop is the
            // reading that says so.** Three slots ever answered this, all of
            // them in the boot's first 24 seconds and none afterwards, by one
            // to seven: `slot=1 page=1 held=3`, `slot=2 page=19 held=26`,
            // `slot=4 page=0 held=1`. The zero was taken when this site was the
            // plane's only publisher; `SessionModel::complete` is the other one
            // now, and it publishes at the channel's publication while the word
            // itself is queued on the GPU-ordered rail. The page is behind
            // because the write has not landed, not because it went back.
            //
            // **And nothing is decided inside that window.** The same boot read
            // `stamp_wait_model_ahead = 0` and `stamp_wait_model_behind = 0`
            // against 767 waits decided — `stamp_wait_model_agrees_met = 405`,
            // `stamp_wait_model_agrees_unmet = 362` — so the model and the
            // legacy evaluation answered every one of them alike. The window
            // exists and no wait falls in it, which is the whole difference
            // between this and the `stamp_wait_model_ahead = 12 824` the
            // publication point was moved to escape.
            crate::model::StampPublication::Behind { held } => {
                if crate::observe::first_sight("stamp_publish_behind", u64::from(index)) {
                    crate::observe::fail(format!(
                        "stamp_publish_behind slot={index} site={site} page={value} held={} \
                         (the ordering plane holds a point later than the word the guest can \
                         read from this slot)",
                        held.0
                    ));
                }
                "stamp_publish_behind"
            }
        },
    );
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
    // Census: from here the value is no longer owed by the coalescing rail, it
    // is with the publication rail below. Recorded before that rail runs so the
    // ledger never reports as still-owed a word already being settled.
    note_stamp_no_longer_owed(state, index, stamp_value);
    // Before the guest is told anything finished, everything this device still
    // owes guest RAM has to be in guest RAM. After this write the guest may free
    // the render targets and its allocator may hand those pages to anything, and
    // no later check can tell that memory apart from the target it used to be —
    // which is why the page-set guard passed on 810 of 810 landings and the heap
    // corruption continued.
    //
    // **Nothing here may settle those writes itself.** Paying the debt is the
    // rail's answer to make, and a rail able to attach the word to the
    // submission the debt is already ordered behind pays it for free. A settle
    // on this line would block first and hand that rail a question already
    // answered "nothing outstanding" — which is the whole cost it exists to
    // remove, and which left `engine_delta` reporting `gpu_stamps=0` beside a
    // `readback_split` `fence` that tracked the flush count exactly.
    //
    // Nothing about the *interrupt* is deferred by a `Queued` answer either: the
    // rail that took the word raises it immediately after publishing.
    if crate::backend::selected().order_completion_stamp(
        state,
        host,
        index,
        stamp_value,
        crate::runtime::render_writeback::SettleSite::CompletionStamp,
    ) == crate::backend::StampOrdering::Queued
    {
        // Advanced at submit, not at completion. From here the guest may see the
        // word at any moment, so a window still armed has already outlived this
        // fence — which is what `armed_stamp_seq` is compared against, and
        // dating it from the completion would call that window punctual.
        state.completion_stamp_seq = state.completion_stamp_seq.wrapping_add(1);
        return;
    }
    let Some(off) = stamp_slot_offset(index, state.page_size()) else {
        return;
    };
    let gpa = state.pfn_gpa(state.gfx.fifo_base_page) + off;
    let page_size = state.page_size() as usize;
    note_stamp_direction(host, gpa, index, stamp_value);
    if gpa_map::write_u32(host, gpa, stamp_value, page_size).is_ok() {
        // The word is in the page: from here the guest can read it, which is
        // the event the ordering plane's stamp waits are about.
        note_stamp_visible(state, index, stamp_value, "stamp_visible_inline");
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
    request: &crate::protocol::fifo::DeviceInfoRequest,
    page_shift: u32,
    version: u32,
) -> Result<(), MemError> {
    // The two bounds are the request's, derived once by the form that knows
    // which of them it carries. The two arms that reach here used to spell the
    // ceiling-less reading twice — `unwrap_or(u32::MAX)` at one, a bare
    // `u32::MAX` at the other — for a field only one of the forms has.
    let bounds = request.reply_bounds();
    let ReplyBounds {
        key_table_len,
        count,
    } = bounds;
    let reply_pfn = request.reply_pfn;
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
    // What the GPU behind this host can execute, for the keys that describe the
    // GPU rather than the protocol. Asked of the rail because the answer is a
    // property of the executing device: one rail serves an Apple GPU to an
    // Apple guest and has nothing to reduce, the other runs on anything from a
    // discrete part to an iGPU at the Vulkan floor, which is exactly the case a
    // fixed table gets wrong.
    let limits = crate::backend::selected().device_info_limits();
    let served = crate::model::device_info_caps(&limits, version);
    // Withhold every key the guest just said it does not parse. This is not a
    // reduction of what the device can do — a key the guest discards on arrival
    // buys nothing, and the slot it occupies is one fewer for a key the guest
    // *does* have an arm for, on a reply that is only ever asked for once.
    let caps: Vec<(u32, u32)> = served
        .iter()
        .copied()
        .filter(|&(key, _)| bounds.parses(key))
        .collect();
    let above_ceiling = served.len() - caps.len();
    // The mirror of `above_ceiling`, and the direction that can cost guest work.
    //
    // `above_ceiling` counts keys this device answers that the guest discards —
    // harmless, and until now it was the only direction reported. The other
    // direction is a key the guest has a parse arm for and this device never
    // sends: the guest's walker stores nothing into that field, so its Metal
    // plugin answers from whatever the capability struct was initialised to.
    // Per [`DEVICE_INFO_CAPS`]'s own doc a value here is an *instruction to the
    // guest about what it may build*, so a key left out is a silent
    // instruction rather than a silent omission, and nothing said how many
    // there were.
    //
    // Split in two, because the halves mean different things and only one of
    // them is a decision this repository made. A **hole** is a key inside the
    // range the table already covers that the table skips — either nobody
    // answered it or the capture it came from never saw it. A **tail** key is
    // past everything this device has ever been asked for, so it is a newer
    // guest asking a newer question. One combined number would make a guest
    // that merely parses further look identical to a table with gaps in it,
    // and those want opposite work.
    //
    // Bounded by construction: the hole list cannot be longer than the table's
    // own highest key, and the tail is a count rather than a list, so a guest
    // declaring an absurd ceiling cannot turn this into a log flood.
    let answered: std::collections::BTreeSet<u32> = caps.iter().map(|&(key, _)| key).collect();
    let table_top = served.iter().map(|&(key, _)| key).max().unwrap_or(0);
    // Key 0 terminates the walk and is not a key, so the parseable set starts
    // at 1.
    // A key the guest's walker has no arm for is not a hole: nothing would ever
    // consume a value sent for it. Reporting one sends the next reader hunting
    // for a value to fill a field that does not exist, which is a boot this
    // investigation actually spent — see
    // [`crate::model::DEVICE_INFO_DEAD_KEYS`].
    let holes: Vec<u32> = (1..key_table_len.min(table_top.saturating_add(1)))
        .filter(|key| !answered.contains(key))
        .filter(|key| !crate::model::DEVICE_INFO_DEAD_KEYS.contains(key))
        .collect();
    let tail = key_table_len.saturating_sub(table_top.saturating_add(1));
    // A hole this device has characterized is a decision; a hole it has not is a
    // guest capability field nobody here has ever looked at. Only the second is
    // a finding, and they are counted apart because the first set is expected to
    // be non-empty forever — a combined count can never read as an alarm.
    let unknown_holes = holes
        .iter()
        .filter(|key| !crate::model::DEVICE_INFO_UNANSWERED_KEYS.contains(key))
        .count();
    note_store_route_n("device_info_key_holes", holes.len() as u64);
    note_store_route_n("device_info_key_holes_unknown", unknown_holes as u64);
    note_store_route_n("device_info_key_tail", u64::from(tail));
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
        "device_info version={} key_table_len={} above_ceiling={} holes=[{}] tail={} dual_plane={} host_samples={} host_d24s8={} host_threads={}x{}x{} host_tg_mem={} host_fp16={} derived=[{}]",
        version,
        key_table_len,
        above_ceiling,
        holes
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(","),
        tail,
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
    // The reply is built once, in the layout's owner, and written once.
    //
    // Three bounds decide it — the guest's parse ceiling, the pair count it
    // sent, and the page the reply lands in — and this arm used to apply them
    // as its own `min` chain while the sibling compute-info reply applied two
    // of the three as its own loop conditions. The staging page is the
    // destination bound made explicit: a buffer exactly as large as the page
    // cannot overrun it, so there is no arithmetic left that could.
    let mut reply = vec![0u8; page_size];
    let written = info_reply::encode(bounds, &served, &mut reply);
    let n = written.pairs;
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
    if written.dropped > 0 {
        let dropped: Vec<u32> = caps[n as usize..].iter().map(|&(key, _)| key).collect();
        crate::observe::fail(format!(
            "device_info truncated reason=reply_pairs_exhausted key_table_len={key_table_len} count={count} max_pairs={max_pairs} wrote={n} have={} dropped={dropped:?}",
            caps.len()
        ));
    }
    // `written.bytes` covers the pairs and the terminator if one was needed, so
    // nothing past the reply is disturbed — the same span the pair-at-a-time
    // loop used to cover, in one write instead of one per key.
    gpa_map::write_bytes(host, gpa, &reply[..written.bytes], page_size)?;
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

/// How many keys [`compute_info_caps`] answers.
///
/// Named because it sizes the reply's staging buffer, and a buffer sized one
/// short of the table would silently drop the last answer rather than fail to
/// compile.
const COMPUTE_INFO_ANSWERS: usize = 3;

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
fn compute_info_caps() -> [(u32, u32); COMPUTE_INFO_ANSWERS] {
    let (max_total_threads, thread_execution_width) =
        crate::backend::selected().compute_threadgroup_limits();
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
    request: &crate::protocol::fifo::ComputeInfoRequest,
) -> bool {
    let crate::protocol::fifo::ComputeInfoRequest {
        raw_task,
        pipeline_ref,
        key_table_len,
        pair_capacity: count,
        reply_gva,
    } = *request;
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
    let served = compute_info_caps();
    // The staging buffer is every answer plus the terminator, and that is the
    // only bound available here.
    //
    // Unlike its device-info twin this request carries **no reply length**: the
    // guest sends a pair count and an address, and nothing on the wire says how
    // large the buffer at that address is. So the destination cannot bound the
    // encode, and `count` — a guest word — must not: sizing the staging buffer
    // from it would let a request asking for 2^32 pairs allocate 32 GiB. What
    // this device could ever write is `served.len()` pairs and one stop word,
    // so that is the buffer, and the encoder's own capacity rule turns the
    // guest's count into the terminator decision it already owns.
    let mut reply = [0u8; (COMPUTE_INFO_ANSWERS + 1) * info_reply::PAIR_LEN];
    let bounds = request.reply_bounds();
    let written = info_reply::encode(bounds, &served, &mut reply);
    let wrote = written.pairs;
    // An answer this device has and the guest's own count could not carry. It
    // cannot fire on a healthy guest — the count a live guest sends is a page
    // of pair slots against three answers — so a firing is the request being
    // wrong, and the three keys are named because each one is a compute limit
    // the guest then sizes dispatches without.
    if written.dropped > 0 {
        let dropped: Vec<u32> = served
            .iter()
            .filter(|&&(key, _)| bounds.parses(key))
            .map(|&(key, _)| key)
            .skip(wrote as usize)
            .collect();
        crate::observe::fail(format!(
            "get_compute_info truncated reason=reply_pairs_exhausted task={task_id} pipe={pipeline_ref} key_table_len={key_table_len} count={count} wrote={wrote} dropped={dropped:?}"
        ));
    }
    // Asked before the write, of the address the request named and the span the
    // encoder produced, so the reading is about the bytes that actually move.
    let _ = crate::runtime::objects::note_query_reply_destination(
        state,
        task_id,
        reply_gva,
        written.bytes as u64,
    );
    if crate::runtime::gva_mem::write_task_gva_product_within(
        state,
        host,
        task_id,
        reply_gva,
        &reply[..written.bytes],
        // The whole reply at the address the request named, not a row loop: the
        // span is the encoder's byte count and the packet being retired is what
        // authorises it.
        None,
    )
    .is_err()
    {
        crate::observe::fail(format!(
            "get_compute_info write_fail task={task_id} gva={reply_gva:#x} wrote={wrote}"
        ));
        return false;
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
    let requirement =
        match crate::backend::heap_placement::heap_texture_size_and_align(&request.descriptor) {
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
    let _ = crate::runtime::objects::note_query_reply_destination(
        state,
        task_id,
        request.reply_gva,
        reply.len() as u64,
    );
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

/// Read a query packet's request at the layout its question uses.
///
/// **The one place this device turns a query opcode into a request.** The four
/// query packets used to be four arms, each naming its own decoder and — for
/// the two device-info opcodes — its own [`DeviceInfoForm`]. The two forms
/// decode to one Rust type and their offsets collide, so reading either at the
/// other's is a well-typed request whose pair count is a page frame and whose
/// reply goes to whatever page that named. Nothing in the arms could compare
/// them, because the choice was made inside each one.
///
/// [`reims_vgpu_core::query::QueryKind`] makes that unrepresentable: the form
/// is an opcode question, the kind answers it, and
/// [`reims_vgpu_core::query::request_words`] reads the layout the kind names.
/// This function is the join, and the arms below name a question rather than a
/// decoder.
///
/// `None` when the request is not one, with the reason already on the failure
/// channel under the question's own name.
fn query_request(
    channel: WireChannel,
    opcode: u16,
    payload: &[u8],
    channel_id: Option<u32>,
) -> Option<RequestWords> {
    let Some(kind) = QueryKind::of(channel, opcode) else {
        // This dispatch table and the closure ledger disagree about which
        // packets are queries. Not reachable while `ledger.rs` passes, and
        // reported rather than dropped because the guest is blocked on the
        // reply either way.
        crate::observe::fail(format!(
            "query_not_a_query_packet site={} op={opcode:#x} (the ledger does not classify this \
             opcode as a query and the drain dispatched it as one)",
            packet_site(channel_id)
        ));
        return None;
    };
    match query::request_words(kind, payload) {
        Ok(words) => Some(words),
        Err(query::WordsRefusal::Short(short)) => {
            note_short_payload(kind.name(), channel_id, &short);
            None
        }
        Err(query::WordsRefusal::HeapTexture(refusal)) => {
            crate::observe::fail(format!(
                "query_malformed_request reason={} site={} question={}",
                refusal.slug(),
                packet_site(channel_id),
                kind.name()
            ));
            None
        }
    }
}

/// The reply this device cannot build, because the question and its request
/// words came apart.
///
/// [`QueryKind::of`] maps an opcode to a question and
/// [`reims_vgpu_core::query::request_words`] returns the variant that
/// question's layout produces, so the two agreeing is
/// `reims_vgpu_core`'s invariant and not this device's. Unreachable, reported
/// rather than ignored: a packet that reached it would otherwise be answered
/// from a request built at the wrong layout, which is the defect the join
/// exists to remove.
fn note_query_layout_mismatch(question: &'static str, channel_id: Option<u32>) {
    crate::observe::fail(format!(
        "query_layout_mismatch site={} question={question} (the request words are not the ones \
         this question's layout produces)",
        packet_site(channel_id)
    ));
}

/// Which payload class the semantic model would read this packet as.
///
/// **The ordering and publication group's denominator, taken before that group
/// moves.** `reims_vgpu_core::session::SessionModel::admit` refuses a packet
/// whose opcode names no command or whose contract is unestablished —
/// `UnknownCommand` and `UnestablishedContract` — and both are the same answer
/// from `reims_vgpu_core::transaction::classify`: `None`. A refusal there is the
/// whole packet, so the question the last group has to answer with a guest and
/// not with an argument is *how many packets a driven boot sends that the model
/// would not classify at all*.
///
/// One route per class rather than one counter, because the classes are not
/// interchangeable: a boot that is nine tenths exec and one that is nine tenths
/// control need different work from the group that moves them, and a single
/// "classified" count reads the same for both.
///
/// This asks `classify` and nothing else. It resolves no reference, reads no
/// guest memory and names nothing — deliberately, because the obvious richer
/// census is to run `crate::runtime::ingress::packet` in shadow, and that would
/// name references on demand out of the guest's object list. Naming is a
/// mutation of the namespace the model owns, so a census that did it would be
/// changing what the device does in order to measure it.
///
/// # What a driven boot answered
///
/// x86 Vulkan, macos-15, three rounds of five applications:
/// **`packet_class_exec=20 802`, `packet_class_lifecycle=13 505`,
/// `packet_class_present=2140`, `packet_class_query=13`,
/// `packet_class_control=11` — 36 471 classified — and
/// `packet_class_unclassified=2166`.**
///
/// The unclassified 2166 were **one opcode**: `CmdDeleteObject`, child `0x28`,
/// whose ledger row was `Closure::Unresolved` because its ref lives in the
/// serializer's per-kind space rather than in the object list. That was five
/// and a half per cent of that guest's packet stream, and it was the whole of
/// what the ordering group would refuse — every other command it sent had a
/// class.
///
/// **That row is settled and the denominator is now zero.** `0x28` reads
/// `Closure::Implemented`; the four rows [`classify`] still answers `None` for
/// are `CmdDebug`, `CmdDisplaySleepState`, `CmdDisplaySetProperties` and
/// `CmdDelay`. A driven macos-26 desktop at x86 Vulkan, three rounds of five
/// applications, sent none of them: **`packet_class_exec=946`,
/// `packet_class_lifecycle=1096`, `packet_class_present=172`,
/// `packet_class_query=3`, `packet_class_control=10` — 2227 classified — and
/// `packet_class_unclassified=0`**, with no line on the failure channel.
///
/// **macos-15 retaken, same workload, same day**: `packet_class_exec=9571`,
/// `packet_class_lifecycle=15033`, `packet_class_present=1066`,
/// `packet_class_query=13`, `packet_class_control=11` — 25 694 classified —
/// and `packet_class_unclassified=0` against the 2166 above, with
/// `packet_unadmitted=0`.
///
/// So on both rails the ordering group refuses nothing for want of a class, and
/// the four remaining rows are a contract question rather than a cutover
/// blocker. The 2166 is kept as the measurement that found the blocker, not as
/// a current reading.
///
/// **And the kind census says the open question is 10 packets wide.** Of the
/// 2166, `child_delete_object_sampler_state=2148`,
/// `child_delete_object_depth_stencil_state=4` and
/// `child_delete_object_render_pipeline_state=4` are the three kinds
/// [`apply_delete_object`] tracks and retires — 2156 of them, and they retire:
/// `sampler_state_deleted=2143`, `ds_state_deleted=4`,
/// `pipeline_state_deleted=4`. The eight untracked kinds are **10 packets**: 5
/// functions, 3 compute pipeline states, 2 fences, which is exactly
/// `cmd_delete_object_unimplemented=10`.
///
/// So the last group's blocker was not a breadth of unknown commands, and it
/// was not 2166 packets of unrecovered contract either. It was one row whose
/// unresolved half a driven guest exercised ten times a boot, holding back the
/// 2156 whose per-kind space this device already resolves — and that row is the
/// one that has since settled.
///
/// The shape of the classified traffic is the other half of the reading. It is
/// over half exec, which is why the class whose payload the bridge could not
/// build at all was the one that had to close first. It has:
/// `crate::runtime::ingress::ExecStreams` carries the three inputs an exec
/// packet costs, and the gap that named them is gone.
fn note_packet_class(channel: reims_vgpu_protocol::packets::Channel, opcode: u16) {
    use reims_vgpu_core::transaction::{classify, PayloadClass};
    match classify(channel, opcode) {
        Some(PayloadClass::Exec) => note_store_route("packet_class_exec"),
        Some(PayloadClass::ResourceLifecycle) => note_store_route("packet_class_lifecycle"),
        Some(PayloadClass::Query) => note_store_route("packet_class_query"),
        Some(PayloadClass::Present) => note_store_route("packet_class_present"),
        Some(PayloadClass::Control) => note_store_route("packet_class_control"),
        None => {
            note_store_route("packet_class_unclassified");
            if crate::observe::first_sight(
                "packet_class_unclassified",
                (u64::from(channel as u8) << 16) | u64::from(opcode),
            ) {
                crate::observe::fail(format!(
                    "packet_class_unclassified channel={channel:?} op={opcode:#x} (the model \
                     has no payload class for this command, so admitting it would promise \
                     ordering and completion for work it cannot describe — the ordering group \
                     refuses the whole packet where this device acts on it)"
                ));
            }
        }
    }
}

/// Whether a lifetime packet names a task the lifecycle owner would hold.
///
/// **The resource-lifecycle group's remaining gate, and the last one it has.**
/// `reims_vgpu_core::lifecycle::Lifecycle` refuses `NoSuchTask` for every one of
/// the twelve commands except the definition itself, and the refusal is of the
/// *whole packet* — so a guest that maps memory, retires a backing or deletes a
/// task it never defined would lose that packet the day the group moves. This
/// device refuses none of them: its task table is created on demand and its map
/// family acts on whatever the record names.
///
/// `declare_object_name` already counts the same question for the declaration
/// door, and this is the other eleven commands' half of it. The two are separate
/// counters because they are separate events: a declaration is produced by
/// resolution and can happen with no packet of its own, and these are packets.
///
/// # Why the refusals a namespace would make are not counted here
///
/// Eight of the twelve resolve a ref or a mapping before they reach
/// `Lifecycle::apply`, and after the group moves that resolution goes through
/// the same `Lifecycle` the apply does — `TaskNamespaces::resource` answers
/// `Some` only for a slot that is live at its own generation, which is exactly
/// what `Namespace::resolve` then accepts. So `Refusal::Namespace` cannot arrive
/// from a packet whose operation was built a moment earlier out of that
/// namespace: it is unreachable by construction rather than merely unobserved,
/// and a counter for it would read zero for a reason a boot does not establish.
/// The task term has no such argument — nothing resolves it — which is why it is
/// the one that gets a census.
///
/// `LifecycleKind::of` is the same table `classify` reads, so a thirteenth
/// command joins this census by existing.
fn note_lifetime_task_definition(
    state: &DeviceState,
    channel: reims_vgpu_protocol::packets::Channel,
    packet: &Packet,
) {
    use reims_vgpu_core::lifecycle::{task_named, LifecycleKind};
    let Some(kind) = LifecycleKind::of(channel, packet.opcode) else {
        return;
    };
    // A definition *is* the event that makes the task hold, so asking whether it
    // already does would count every first definition as a refusal of a packet
    // the model accepts.
    if kind == LifecycleKind::DefineTask {
        return;
    }
    let task = match task_named(kind, &packet.payload) {
        Ok(task) => task,
        // Short of its own record. Already reported by the arm that decodes it,
        // and not this census's question: a packet whose task word is not there
        // is refused before the task is asked about.
        Err(_) => return,
    };
    if state.tasks.is_active(task.0) {
        note_store_route("lifetime_command_in_a_defined_task");
        return;
    }
    note_store_route("lifetime_command_in_an_undefined_task");
    if crate::observe::first_sight(
        "lifetime_command_in_an_undefined_task",
        (u64::from(task.0) << 16) | u64::from(packet.opcode),
    ) {
        crate::observe::fail(format!(
            "lifetime_command_in_an_undefined_task kind={} task={} op={:#x} (no CmdDefineTask2              opened this task, so the lifecycle owner these commands move to would refuse the              whole packet with NoSuchTask — this device acts on it)",
            kind.name(),
            task.0,
            packet.opcode
        ));
    }
}

fn process_root_packet<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    packet: &Packet,
) {
    note_packet_class(reims_vgpu_protocol::packets::Channel::Root, packet.opcode);
    note_lifetime_task_definition(state, reims_vgpu_protocol::packets::Channel::Root, packet);
    match packet.opcode {
        // The two device-info opcodes, one arm. They differ only in which form
        // their payload is — the newer one prepends a parse ceiling — and
        // reading either at the other's offsets would take the count for a page
        // frame and write the reply wherever that landed. Which form an opcode
        // carries is no longer this arm's answer at all: `query_request` asks
        // `reims_vgpu_core::query::QueryKind`, so the two opcodes now differ
        // here in nothing.
        ROOT_OP_DEVICE_INFO_TAHOE | ROOT_OP_DEVICE_INFO_MONTEREY => {
            match query_request(WireChannel::Root, packet.opcode, &packet.payload, None) {
                Some(RequestWords::DeviceInfo(request)) => {
                    // Asked before the reply is written, because the question is
                    // what this device identified at the moment of the write.
                    crate::runtime::objects::note_device_info_reply_destination(
                        state,
                        request.reply_pfn,
                    );
                    let _ = reply_device_info(host, &request, state.page_shift, state.gfx.version);
                }
                Some(_) => note_query_layout_mismatch("device_info", None),
                None => {}
            }
        }
        // The two ends of one channel lifetime. They share a payload because
        // they share a decoder — `crate::protocol::fifo` owns that layout, and
        // reading it here with a length check and an `ld32` at a named offset
        // was a second reading of one record.
        ROOT_OP_DEFINE_FIFO | ROOT_OP_FREE_FIFO => {
            // Which end of the lifetime this is used to be
            // `packet.opcode == ROOT_OP_DEFINE_FIFO` — a direction decided from
            // a constant inside the arm, in the shape the device-info form had.
            // `reims_vgpu_core::control` reads it off the ledger instead, so
            // the two opcodes differ here in nothing and the transition is a
            // value rather than a comparison.
            let op = match control::resolve(WireChannel::Root, packet.opcode, &packet.payload) {
                Ok(op) => op,
                Err(control::ResolveRefusal::Payload(short)) => {
                    // The kind names the slug, so the reason a reader greps for
                    // and the command it names cannot drift apart.
                    let kind = ControlKind::of(WireChannel::Root, packet.opcode);
                    note_short_payload(
                        kind.map_or("channel_lifetime", ControlKind::name),
                        None,
                        &short,
                    );
                    return;
                }
                Err(refusal @ control::ResolveRefusal::NotControl { .. }) => {
                    // This dispatch table and the ledger disagree about what a
                    // control packet is. Unreachable while `ledger.rs` passes.
                    crate::observe::fail(format!(
                        "channel_lifetime_refused reason={} site=root op={:#x}",
                        refusal.slug(),
                        packet.opcode
                    ));
                    return;
                }
            };
            let control::ControlOp::Channel { transition, domain } = op else {
                // `ControlKind::channel_transition` answers `Some` for exactly
                // these two opcodes, so another operation here would mean the
                // ledger and this arm name different commands.
                crate::observe::fail(format!(
                    "channel_lifetime_not_a_channel_op site=root op={:#x} kind={}",
                    packet.opcode,
                    op.kind().name()
                ));
                return;
            };
            let ch = domain.0;
            if !is_child_channel(ch) {
                // Dropped in silence before this. A guest opening a domain this
                // device has no FIFO for gets no ring drained and no completion
                // published on it, which is a hang with nothing in the log.
                crate::observe::fail(format!(
                    "channel_lifetime_out_of_range site=root op={:#x} kind={} channel={ch} \
                     max={MAX_CHANNELS} (this device drains no such FIFO, so every packet the \
                     guest puts on it is a completion word nothing will publish)",
                    packet.opcode,
                    op.kind().name()
                ));
                return;
            }
            let bit = 1u32 << ch;
            // **The transition is the model's now.** This arm used to set and
            // clear a bit of `DeviceState::active_child_mask`; the domain's
            // openness is `SessionModel::open_channels` and this is the one
            // writer of it. What stays here is everything that is *not* the
            // domain's publication lifetime: the pending-work bit, and the
            // per-channel drain state `forget_child_channel` resets.
            let verdict = state.apply_channel_control(op);
            note_channel_transition_verdict(verdict, transition, ch);
            if transition == control::ChannelTransition::Free {
                // Only a free clears it: an open is not a claim that nothing is
                // pending on the channel, and clearing it there would drop a
                // drain the guest is owed.
                state.pending.child_mask &= !bit;
            }
            forget_child_channel(state, ch, bit);
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
    let ring_size = main_ring_data_size(state.gfx.fifo_length, state.gfx.fifo_start);
    if ring_size == 0 || state.gfx.fifo_base_page == 0 {
        state.pending.main_drain = false;
        return;
    }
    let base = state.pfn_gpa(state.gfx.fifo_base_page) + state.gfx.fifo_start as u64;

    let ring = Ring::Root {
        base_gpa: base,
        capacity: ring_size,
    };
    loop {
        // Loaded once per turn and used for the whole of it. Every read below
        // used to load it again, which was the same value each time — nothing
        // advances the consumer pointer until this loop stores to it — read
        // seven ways.
        let head = state
            .gfx
            .fifo_read
            .load(std::sync::atomic::Ordering::Acquire);
        match arrival(&ring, host, head, state.gfx.fifo_written) {
            Arrival::Nothing => break,
            Arrival::Fault(fault) => {
                state.record_fail(FailEvent::MalformedRootPacket { fault, head });
                break;
            }
            Arrival::Packet(packet) => {
                // The census, asked for the reading and not for a verdict. The
                // wait this timeline holds most often is a `DELETE_TASK`
                // ordered behind a child FIFO's stamp, and running it early
                // tears down a task whose work has not been drained — which is
                // exactly the ordering `admit` now holds, on a position rather
                // than on this ring's consumer pointer.
                note_packet_stamp_waits(state, host, None, &packet);

                // The head advances first and unconditionally; see the child
                // drain's copy for why that is the whole of the switch.
                state
                    .gfx
                    .fifo_read
                    .store(packet.next_head, std::sync::atomic::Ordering::Release);

                admit_and_park(
                    state,
                    host,
                    crate::runtime::ingress::Fifo::ROOT,
                    ROOT_STAMP_SLOT,
                    packet,
                );
                settle_model_work(state, host);
            }
        }
    }

    // A last look at what the model released, for the same reason the child
    // drain takes one: a decode fault stops this loop, and a position that
    // became runnable while it was working is still owed.
    settle_model_work(state, host);
    // A parked root position is unfinished work, not a drained ring: clearing
    // the flag here would drop the retry, and the packet would only run again
    // if some later doorbell happened to set it. It is asked of the store
    // rather than of a mask, because the store is what holds the work.
    state.pending.main_drain = !state.parked.is_empty();
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

/// The child FIFO's own ring reader, and **the hot one**.
///
/// There are two readers over one wire form. [`read_ring_bytes`] serves the
/// root FIFO and `ring_reads` for it measured **zero** across a driven
/// `blur=40` boot while the child FIFO decoded ~8 000 packets a second — so
/// timing only that one reports a rail that never runs and reads exactly like a
/// span that is free. Both are counted into the same pair now; a split by FIFO
/// would be the next refinement if the total turns out to matter.
fn read_child_ring_bytes<M: HostMemory>(
    mem: &M,
    page_gpas: &[u64],
    ring_length: u32,
    absolute: u32,
    len: u32,
    page_shift: u32,
) -> Result<Vec<u8>, MemError> {
    let started = std::time::Instant::now();
    let out = read_child_ring_bytes_inner(mem, page_gpas, ring_length, absolute, len, page_shift);
    census::note_drain_ring(started.elapsed().as_nanos() as u64);
    out
}

fn read_child_ring_bytes_inner<M: HostMemory>(
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
///
/// # "Always" is measured, and it holds on the boots that look like it does not
///
/// About three macos-13 boots in ten present ~60 frames a second for their whole
/// life rather than ~100-117, which reads exactly like a guest that selected a
/// 60 Hz mode from this table. It did not. Twelve driven boots, asked over ssh
/// while still up what mode they were running:
///
/// ```text
/// boots      presented      system_profiler SPDisplaysDataType
/// 5 fast    107 - 119 Hz    UI Looks like: 1920 x 1080 @ 120.00Hz
/// 7 slow     59.9 - 61 Hz   UI Looks like: 1920 x 1080 @ 120.00Hz
/// ```
///
/// Every boot, both populations, the same mode and the same 120.00 Hz. So mode
/// negotiation is not where the split comes from, this table is doing its job,
/// and a fix aimed at the timing elements would be aimed at nothing.
///
/// That leaves the cause downstream of mode selection, in what the guest's
/// compositor does with a 120 Hz link it has correctly acquired. `VBL_REPORT_
/// EARLY` beside [`census::VblCensus`] records what else has been ruled out.
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
    // Both encodings of the physical size, from one value each, as the reference
    // host does: it writes the pair and the float pair on every publish with no
    // version test. The integer pair is what the guest's synthesised EDID takes
    // its centimetre fields from and is the one a stock rung-4 guest reads; the
    // float pair is what `displayDimensionFloats` licenses at rung 0x2a and
    // above, and leaving it unwritten is a 0 x 0 mm panel to any guest that gets
    // there.
    let (width_f32, width_mm) = display_dimension_mm(DISPLAY_WIDTH_MM);
    let (height_f32, height_mm) = display_dimension_mm(DISPLAY_HEIGHT_MM);
    shared_w16(host, gpa, DISPLAY_DESC_WIDTH_MM, width_mm, psz);
    shared_w16(host, gpa, DISPLAY_DESC_HEIGHT_MM, height_mm, psz);
    shared_w32(
        host,
        gpa,
        DISPLAY_DESC_WIDTH_MM_F32,
        width_f32.to_bits(),
        psz,
    );
    shared_w32(
        host,
        gpa,
        DISPLAY_DESC_HEIGHT_MM_F32,
        height_f32.to_bits(),
        psz,
    );
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
    // The record's layout is `crate::protocol::fifo`'s. What is *usable* is
    // this function's, because it is the thing that allocates the sprite and
    // reads the guest's rows.
    let record = match crate::protocol::fifo::decode_cursor_glyph(&packet.payload) {
        Ok(record) => record,
        Err(short) => {
            return cursor_glyph_fail(
                "cursor_glyph_short",
                format!(
                    "cursor_glyph_fail reason=cursor_glyph_short plen={} need={}",
                    short.plen, short.need
                ),
            )
        }
    };
    let task_id = record.task_id;
    let virtual_offset = record.virtual_offset;
    let mapped_length = record.mapped_length;
    // Whole, not narrowed. The wire's stride is eight bytes, and taking the
    // low four of it turns a pitch that does not fit into a small one that
    // passes every bound below — the glyph is then read at the wrong pitch and
    // the guest gets a garbled pointer with nothing on the failure channel. It
    // is compared and multiplied in its own width, and the bounds it has to
    // pass — `cursor_glyph_geom`, `cursor_glyph_mapped_len`,
    // `cursor_glyph_alloc` — are what refuse an unusable one.
    let stride = record.stride;
    let width = u32::from(record.width);
    let height = u32::from(record.height);
    let hot_x = u32::from(record.hot_x);
    let hot_y = u32::from(record.hot_y);

    if width == 0
        || height == 0
        || width > CURSOR_MAX_DIM
        || height > CURSOR_MAX_DIM
        || stride < u64::from(width.saturating_mul(CURSOR_GLYPH_BPP))
        || hot_x >= width
        || hot_y >= height
    {
        return cursor_glyph_fail(
            "cursor_glyph_geom",
            format!("cursor_glyph_fail reason=cursor_glyph_geom {width}x{height} stride={stride} hot=({hot_x},{hot_y}) max={CURSOR_MAX_DIM}"),
        );
    }
    let need = u64::from(height - 1)
        .saturating_mul(stride)
        .saturating_add(u64::from(width) * u64::from(CURSOR_GLYPH_BPP));
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
        // `need` bounded the last row's start and `host_alloc_len` accepted it,
        // so the product fits; `try_from` rather than `as` because a cast here
        // would be the narrowing this function just stopped doing.
        let row = usize::try_from(u64::from(y).saturating_mul(stride)).unwrap_or(usize::MAX);
        for x in 0..width {
            let px = row.saturating_add((x as usize) * CURSOR_GLYPH_BPP as usize);
            if px.saturating_add(4) > src.len() {
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
        .filter(|&&e| e & crate::protocol::iosurface_pages::PAGE_ENTRY_VALID != 0)
        .map(|&e| e >> crate::protocol::iosurface_pages::PAGE_ENTRY_PFN_SHIFT)
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
                .filter(|&&e| e & crate::protocol::iosurface_pages::PAGE_ENTRY_VALID != 0)
                .filter(|&&e| {
                    named_pfns
                        .contains(&(e >> crate::protocol::iosurface_pages::PAGE_ENTRY_PFN_SHIFT))
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
    // x86: present surface_id → backing object-list slot (heap index =
    // IOSurface getSurfaceID). Arm: MappingInternal page-table resolve.
    // Always attempt backing when pages empty; then iosfc/mapper path.
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
        // Independent of everything below: the guest's own copy of the plane
        // this present names, sampled where the desktop background belongs.
        crate::runtime::scanout::note_present_field_witness(state, &*host, mapping, w, h);
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
            let carried = crate::backend::selected().present_resident_carries(state, mapping, w, h);
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

    /// The lowercase reason stem this command reports refusals under.
    ///
    /// Separate from [`Self::name`], which is the human column of the census
    /// line: a `reason=` slug is grepped for and every other one in this file
    /// is lowercase, so `reason=MapMemory2_short` would be the one nobody
    /// finds. The map pair's stems were already written as two literals at the
    /// view-retire site; this is that pair, extended to the family.
    const fn slug(self) -> &'static str {
        match self {
            Self::MapMemory2 => "map_memory2",
            Self::UnmapMemory => "unmap_memory",
            Self::InvalidateResources => "invalidate_resources",
            Self::SynchronizeResources => "synchronize_resources",
            Self::SynchronizeAndDiscardResources => "synchronize_and_discard_resources",
            Self::DeleteIOSurfaceBacking2 => "delete_iosurface_backing2",
        }
    }
}

/// Record the page-table nodes `gva` descends through under `task_id`, and say
/// whether this device wrote to any of them since it last saw them as nodes.
///
/// Stands here, on the map/unmap packet, for two reasons: the tree is being
/// edited at exactly this moment, so the nodes read are the live ones; and the
/// device is already holding the task and the address, so the whole cost is one
/// descent of at most [`node_guard::MAX_TREE_NODES`] guest reads.
///
/// A finding is emitted on the fail channel because it is one — a host write
/// into a page of page-table entries is the corruption class that ends a guest,
/// and the zero-word shape of this device's clears is what the guest's own
/// teardown assertion reads as a missing entry. Everything else is counted and
/// silent.
fn observe_page_table_nodes<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &H,
    task_id: u32,
    gva: u64,
) {
    use crate::runtime::node_guard::{self, NodeVerdict};

    if !node_guard::enabled() {
        return;
    }
    let Some(geometry) = reims_vgpu_paging::resolve::geometry_for_page_shift(state.page_shift)
    else {
        return;
    };
    let Some(entry) = state.tasks.get(task_id) else {
        return;
    };
    let task = reims_vgpu_paging::resolve::Task {
        active: entry.active,
        directory_pfn: entry.directory_pfn,
    };
    let mut nodes = [0u64; node_guard::MAX_TREE_NODES];
    let found = reims_vgpu_paging::resolve::task_node_gpas(
        &crate::runtime::gva_mem::HostPhys(host),
        geometry,
        &task,
        gva,
        &mut nodes,
    );
    if found == 0 {
        return;
    }

    let now_us = crate::observe::elapsed_us();
    // The write census is read while the watch is mutated, so it is split off
    // first: both live on `DeviceState` and only one of them is being written.
    let DeviceState {
        host_writes,
        node_guard: watches,
        ..
    } = state;
    let watch = watches.entry(task_id).or_default();
    for &gpa in &nodes[..found] {
        let verdict = watch.observe(host_writes, gpa, now_us);
        note_store_route(verdict.route());
        if let NodeVerdict::Wrote { gap_us } = verdict {
            if crate::observe::first_sight("node_guard_wrote_node_page", gpa) {
                crate::observe::fail(format!(
                    "node_guard reason={} task={task_id} node_gpa={gpa:#x} gva={gva:#x} \
                     gap_us={gap_us} watched={} refused={} (this device wrote into a guest page \
                     holding page-table entries; a zero word landing there is what the guest's \
                     own teardown reads as an entry it has already cleared, and it panics on one)",
                    verdict.route(),
                    watch.watched(),
                    watch.refused(),
                ));
            }
        }
    }
}

/// Record the guest-physical pages of `[gva, gva+length)` as released, on an
/// unmap, or as handed back, on a map.
///
/// Both directions matter and the map one is not an afterthought: a page the
/// guest maps again is a page this device is entitled to write, so leaving it
/// watched would report every recycled page as a defect. See
/// [`crate::runtime::released_pages`].
///
/// The resolve runs against the live page table, which on an unmap means it has
/// to happen before the packet is applied. A range that no longer translates
/// resolves to fewer pages than it spans, and that is not an error here — those
/// pages are already gone and there is nothing left to watch.
/// Hold the guest's hand-back of pages until every writeback this device has
/// queued into them has landed.
///
/// The guest frees a span's pages before it submits the `UnmapMemory` that
/// names it, on the strength of the completion it already read for the work
/// that used them. A GPU-direct writeback is recorded into the command stream
/// and not waited on, so between that completion and the copy landing there is
/// a window in which the kernel can reuse the page — and the copy then lands in
/// whatever it put there. One macOS 15 guest panicked in `kevent_register`
/// with `RGBA16Float` texels inside a kqueue object, with every
/// write-after-release guard at zero, because those guards run when a write is
/// planned and nothing runs when it lands.
///
/// This is the release-side twin of the reader-side settles: cheap when the
/// debt flag is clear (one atomic load, the common case), and a bounded fence
/// wait otherwise. The page walk runs only when something is outstanding, and
/// on this packet it often finds nothing — the guest unwired the span first —
/// which reads as `Unnamed` and settles unconditionally, the fail-safe
/// direction. An `Overlap` is the finding itself: a write still in flight into
/// pages the guest has already released, reported once per task on the
/// always-on channel so a later panic can be attributed rather than guessed.
fn settle_released_span<H: HostMemory + HostOps>(
    state: &DeviceState,
    host: &H,
    task_id: u32,
    gva: u64,
    length: u64,
) {
    use crate::runtime::render_writeback::SettleSite;
    let backend = crate::backend::selected();
    if !backend.guest_writes_outstanding() {
        return;
    }
    let pages: Vec<u64> = crate::runtime::gva_mem::task_gva_page_gpa_set(
        host,
        &state.tasks,
        task_id,
        gva,
        length,
        state.page_shift,
    )
    .into_iter()
    .collect();
    let reach = if pages.is_empty() {
        crate::backend::GuestWriteReach::Unnamed
    } else {
        backend.guest_writes_reaching(&pages)
    };
    settle_release(SettleSite::GuestRelease, reach, || {
        format!(
            "guest_release_reaches_outstanding_write family=unmap_memory task={task_id} \
             gva={gva:#x} len={length:#x} pages={} (a writeback this device queued has not \
             landed and the guest has already released these pages; settled here, which is \
             the last point that can hold it)",
            pages.len()
        )
    });
}

/// [`settle_released_span`] for `DeleteIOSurfaceBacking2`, whose pages are the
/// mapping's own list rather than a span to walk.
fn settle_released_mapping(state: &DeviceState, mapping_id: u32) {
    use crate::runtime::render_writeback::SettleSite;
    let backend = crate::backend::selected();
    if !backend.guest_writes_outstanding() {
        return;
    }
    let reach = match state.mapping_reach_pages(mapping_id) {
        Some(pages) if !pages.is_empty() => backend.guest_writes_reaching(&pages),
        _ => crate::backend::GuestWriteReach::Unnamed,
    };
    settle_release(SettleSite::BackingRelease, reach, || {
        format!(
            "guest_release_reaches_outstanding_write family=delete_iosurface_backing2 \
             mid={mapping_id} (a writeback this device queued has not landed and the guest \
             has already released this mapping's pages; settled here)"
        )
    });
}

/// The shared tail of the two release settles: count the answer under the
/// site, report an overlap once per site, and wait unless the writeback is
/// provably elsewhere.
fn settle_release(
    site: crate::runtime::render_writeback::SettleSite,
    reach: crate::backend::GuestWriteReach,
    finding: impl FnOnce() -> String,
) {
    use crate::backend::GuestWriteReach as Reach;
    note_store_route(match reach {
        Reach::Disjoint => site.route_disjoint(),
        Reach::Overlap => site.route_overlap(),
        Reach::Unnamed => site.route_unnamed(),
    });
    if reach == Reach::Overlap
        && crate::observe::first_sight("guest_release_reaches_outstanding_write", site as u64)
    {
        crate::observe::fail(finding());
    }
    if reach == Reach::Disjoint {
        return;
    }
    crate::runtime::render_writeback::settle_guest_writes(site);
}

fn note_released_or_remapped<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &H,
    task_id: u32,
    gva: u64,
    length: u64,
    family: MapFamily,
) {
    if !crate::runtime::node_guard::enabled() {
        return;
    }
    let pages = crate::runtime::gva_mem::task_gva_page_gpa_set(
        host,
        &state.tasks,
        task_id,
        gva,
        length,
        state.page_shift,
    );
    if pages.is_empty() {
        return;
    }
    let writes = &mut state.host_writes;
    match family {
        MapFamily::UnmapMemory => {
            for gpa in pages {
                writes.release_page(gpa);
            }
        }
        _ => {
            for gpa in pages {
                writes.remap_page(gpa);
            }
        }
    }
}

/// Say whether this range's entries are there, and on a map — the one direction
/// the guest orders — treat their absence as the defect it is.
///
/// The guest finishes wiring a range before it submits the map for it, so a map
/// whose range is not fully covered is a mapping its own tree does not hold. It
/// submits an unmap *before* unwiring, so that direction is a race and is
/// counted rather than judged; keeping it is what proves the walk works, since a
/// broken walk would read absent on both sides. See
/// [`crate::runtime::range_coverage`].
fn observe_range_coverage<H: HostMemory + HostOps>(
    state: &DeviceState,
    host: &H,
    task_id: u32,
    gva: u64,
    length: u64,
    op: crate::runtime::range_coverage::Op,
) {
    use crate::runtime::range_coverage::{self, Coverage};

    if !range_coverage::enabled() {
        return;
    }
    let (spanned, scanned) = range_coverage::pages_of(length, state.page_shift);
    if scanned == 0 {
        return;
    }
    if scanned < spanned {
        note_store_route(op.truncated_route());
    }
    let Some(geometry) = reims_vgpu_paging::resolve::geometry_for_page_shift(state.page_shift)
    else {
        return;
    };
    let Some(entry) = state.tasks.get(task_id) else {
        return;
    };
    let task = reims_vgpu_paging::resolve::Task {
        active: entry.active,
        directory_pfn: entry.directory_pfn,
    };
    let counts = reims_vgpu_paging::resolve::range_coverage(
        &crate::runtime::gva_mem::HostPhys(host),
        geometry,
        &task,
        gva,
        scanned,
    );
    let verdict = counts.as_ref().map_or(Coverage::Unwalkable, Coverage::of);
    let route = verdict.route(op);
    note_store_route(route);
    if verdict.is_finding(op) && crate::observe::first_sight(route, u64::from(task_id)) {
        let leaf = verdict.is_leaf_level().unwrap_or(false);
        crate::observe::fail(format!(
            "range_coverage reason={route} task={task_id} gva={gva:#x} len={length:#x} \
             pages={spanned} scanned={scanned} leaf_level={leaf} detail={verdict:?} \
             (the guest wires a range fully before publishing the map for it, so a page \
             without an entry here is a page its own teardown will later assert on)"
        ));
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
    // The floor and the three offsets are the record's owner's. They were four
    // literals here — a `20` beside a `0`, a `4` and a `12` — and the two
    // eight-byte fields are exactly the shape a reader gets wrong: an address
    // read as a `u32` puts the length word in the middle of the address.
    let notice = match matches!(family, MapFamily::MapMemory2 | MapFamily::UnmapMemory)
        .then(|| crate::protocol::fifo::decode_map_memory(&packet.payload))
        .transpose()
    {
        Ok(notice) => notice,
        // A notice too short to read is one whose interval this device does not
        // know, and every response below is keyed on that interval: the bind
        // resolutions and host views retired are exactly the ones overlapping
        // it. Reported by name rather than falling through to the generic
        // census line, which prints two words of a record it could not read.
        Err(short) => {
            note_short_payload(family.slug(), Some(channel_id), &short);
            return;
        }
    };
    // The same, for the retirement that shares this arm. A payload too short to
    // name the surface is one whose backing this device cannot condemn, and the
    // id is reused within tens of milliseconds under scroll — so a dropped
    // retirement is later pixels written into pages the guest has recycled.
    let retire = match matches!(family, MapFamily::DeleteIOSurfaceBacking2)
        .then(|| crate::protocol::fifo::decode_delete_backing(&packet.payload))
        .transpose()
    {
        Ok(retire) => retire,
        Err(short) => {
            note_short_payload(family.slug(), Some(channel_id), &short);
            return;
        }
    };
    if let Some(crate::protocol::fifo::MapMemoryCommand {
        task_id,
        gva,
        length,
    }) = notice
    {
        // Audit the interval against what this task already has live: a range
        // mapped twice or unmapped without a map is a disagreement the guest's
        // own teardown assertion will eventually find.
        //
        // Both fields are the values the guest's own `allocate`/`deallocate`
        // receive: its length getter forwards to the same call this packet's
        // length field is built from, and the address is one getter used by
        // both. So these intervals are the page-table ranges and not merely
        // consistent with themselves. Observation only; nothing reads the
        // verdict. See `runtime::map_audit`.
        // The notice, in the model that owns the address space it is about.
        // `Lifecycle` holds nothing keyed by a guest address — deliberately, so
        // that no resolution it hands out can go stale behind its back — so it
        // performs nothing here and states the obligation instead: every
        // resolution held over this interval was computed against pages the
        // guest has since moved. The audit and the retirements below are this
        // device discharging exactly that, and the direction they discharge it
        // in is the model's `Remap::established` rather than a second reading of
        // the opcode.
        let remapped = state
            .apply_lifetime(
                &if matches!(family, MapFamily::MapMemory2) {
                    reims_vgpu_core::lifecycle::LifecycleOp::MapMemory {
                        task: reims_vgpu_core::identity::TaskId(task_id),
                        span: reims_vgpu_core::access::GuestSpan { base: gva, length },
                    }
                } else {
                    reims_vgpu_core::lifecycle::LifecycleOp::UnmapMemory {
                        task: reims_vgpu_core::identity::TaskId(task_id),
                        span: reims_vgpu_core::access::GuestSpan { base: gva, length },
                    }
                },
                family.slug(),
            )
            .and_then(|acted| acted.remapped.into_iter().next());
        {
            let page_size = 1u64 << state.page_shift;
            // The model's answer where there is one. A refusal is reported by
            // the door and the audit still runs off the packet's own opcode:
            // this device's caches alias pages whether or not the model kept a
            // task for them, and an invalidation it skipped would leave a host
            // view over memory the guest has taken back.
            let established = remapped.map_or_else(
                || matches!(family, MapFamily::MapMemory2),
                |r| r.established,
            );
            let intervals = state.map_audit.entry(task_id).or_default();
            let verdict = if established {
                intervals.map(gva, length, page_size)
            } else {
                intervals.unmap(gva, length)
            };
            // Counted on every verdict, including `Consistent`. The fail line
            // below is emitted only on a finding and deduped on top of that, so
            // without this the audit's silence would be indistinguishable from
            // the audit never having run — which is what "clean on a dozen
            // panicking boots" actually rested on. The census is the only
            // never-fired signal there is.
            note_store_route(verdict.slug());
            if verdict.is_finding()
                && crate::observe::first_sight(
                    verdict.slug(),
                    u64::from(task_id) << 32 | u64::from(channel_id),
                )
            {
                let live = intervals.live_count();
                crate::observe::fail(format!(
                    "map_audit op={name} reason={} task={task_id} gva={gva:#x} len={length:#x} \
                     live={live} detail={verdict:?} (the guest applies this exact interval to its \
                     own page table; a disagreement here is one its teardown will assert on)",
                    verdict.slug()
                ));
            }
        }
        // The other half of the same question, and the one the interval audit
        // reading clean moves the weight onto: has this device *written* into a
        // page that holds the guest's page-table entries? The descent below is
        // the only work done for it — the write census it asks is already kept
        // for the sampled cache. See `runtime::node_guard`.
        observe_page_table_nodes(state, host, task_id, gva);
        // And the half `node_guard` structurally cannot see: a write landing on
        // a page *before* it becomes a node. The page list has to be resolved
        // here, ahead of the unmap being applied, because this is the last
        // moment those addresses translate. See `runtime::released_pages`.
        note_released_or_remapped(state, host, task_id, gva, length, family);
        // And the question none of the three above asks, because none of them
        // needs a host write to be true: are this range's entries in the state
        // the guest's own next step requires? It asserts per page that an unmap
        // finds one and a map does not. Both directions are read, and the one
        // that cannot end a boot is what makes the other's reading evidence —
        // see `runtime::range_coverage`.
        observe_range_coverage(
            state,
            host,
            task_id,
            gva,
            length,
            if matches!(family, MapFamily::UnmapMemory) {
                crate::runtime::range_coverage::Op::Unmap
            } else {
                crate::runtime::range_coverage::Op::Map
            },
        );
        // Verbose-gated walk probe at map/unmap time. This runs a full
        // guest page-table walk (`diagnose_gva_walk`) purely to build the
        // log string, and fired ~9k times/boot on the drain path — a flood
        // and a real per-map cost. Gate it (and the periodic census) behind
        // `REIMS_VGPU_DRAW_LOG=1` so a normal boot pays neither; the functional
        // view-retire below stays always-on. Wire has no PPNs — the probe
        // asks whether the guest PT is already walkable under wire task_id.
        crate::observe::when_verbose(|| {
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
        });
        // RE (AppleParavirtMemoryMap): Unmap/Map only mutate the **task
        // page table** then notify — wire has no PPNs. Guest order is
        // deallocate/allocate **then** FIFO, so:
        // - Unmap notify: PTEs already gone → cannot GVA-write; retain
        //   host_gva_surfaces for sample (wallpaper wipe class).
        // - Map notify: PTEs already live → flush host_gva encode into
        //   **new** PFNs (not invent PTEs; not invent geom). Discrete
        //   normal-texture content may live only in host_cache until this.
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
            // Before anything else, because everything else here forgets the
            // span: a writeback this device queued into these pages must land
            // before the guest's allocator hands them to someone else.
            if matches!(family, MapFamily::UnmapMemory) {
                settle_released_span(state, host, task_id, gva, length);
            }
            // Held bind resolutions over this range name pages the guest
            // has just remapped. Retired by range, which is exactly what
            // this notify carries.
            note_bb_retired(
                "bb_retire_map_range",
                state.retire_bound_buffers_in_range(task_id, gva, length),
            );
            let n =
                crate::runtime::gva_view::retire_gva_views_overlapping(state, task_id, gva, length);
            crate::runtime::gva_view::log_retire(family.slug(), task_id, gva, length, n);
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
    } else if let Some(retire) = retire {
        // The live Ventura payload agrees with the resource contract:
        // `{objectID, taskID}` — **the reverse** of the `{task, object}` pair
        // `CmdDeleteResource` and `CmdReplacePhysical` carry, which is why the
        // offsets are the decoder's and not two `ld32`s here. This is the
        // lifetime boundary for the host IOSurface backing, not stamp-only
        // bookkeeping. Keeping page_entries after it lets later id reuse/clear
        // write pixels into pages the guest has recycled.
        let crate::protocol::fifo::DeleteBackingCommand { object_id, task_id } = retire;
        // The retirement, in the model that owns the names over these bytes.
        // The contract retires the backing *and* the resources that named it,
        // so the teardowns below are the model's per-name answers rather than a
        // second walk of anything here — and they are counted rather than acted
        // on, because the host-side condemnation this arm performs is keyed by
        // the mapping id and covers the same storage whichever names were on it.
        if let Some(acted) = lifetime_operation(
            state,
            host,
            reims_vgpu_protocol::packets::Channel::Child,
            packet.opcode,
            &packet.payload,
        )
        .and_then(|op| state.apply_lifetime(&op, family.slug()))
        {
            note_store_route_n("backing_retire_teardowns", acted.teardowns.len() as u64);
        }
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
        // Same obligation as the unmap arm, keyed by the mapping whose pages
        // are going back: a queued writeback into them must land first.
        settle_released_mapping(state, object_id);
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
        crate::observe::verbose(|| {
            format!(
                "map_family op=DeleteIOSurfaceBacking2 ch={channel_id} object={object_id} task={task_id} plen={plen} mode={mode}"
            )
        });
    } else if family == MapFamily::InvalidateResources {
        // RE: {task_id, count} + count×{object_id, 4×u8 validity ops}.
        // Ops (PVG host layout): clr_host, set_host, clr_guest, set_guest.
        // Pageon hardcodes LE 01 00 00 01 = clr hostValid + set guestValid.
        //
        // The same four bytes the EXEC_INDIRECT2 resource table carries,
        // through the same consumer: this producer's records are 8 bytes
        // and that one's are 24, but the quad is one contract and must
        // not acquire two meanings.
        use crate::protocol::fifo::{decode_invalidate_resources, CHILD_INVALIDATE_PAGEON_FLAGS};
        use crate::runtime::resource_validity::{apply, ValiditySite};
        match decode_invalidate_resources(&packet.payload) {
            Ok(cmd) => {
                // Read-only, and taken before anything is applied: whether the
                // model's list join could have resolved this packet at all. See
                // `objects::note_lifetime_refs_named`.
                crate::runtime::objects::note_lifetime_refs_named(
                    state,
                    host,
                    cmd.task_id,
                    &cmd.records
                        .iter()
                        .map(|rec| rec.object_id)
                        .collect::<Vec<_>>(),
                );
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
                    // A standalone statement: whatever this device owes these
                    // pages was rendered by earlier work, so it lands now.
                    if outcome.guest_read_requested {
                        crate::runtime::resource_validity::deliver_for_guest_read(
                            state,
                            host,
                            cmd.task_id,
                            rec.object_id,
                        );
                    }
                }
                // One counter here, two on the exec side: `pageBacking`
                // names mapping ids, so a record this device holds no
                // mapping for is already the surprising case. The exec
                // table names task object refs, most of which have no
                // surface state by construction.
                note_store_route_n("validity_miss_inv", miss as u64);
                // The same statement, in the model that owns content authority
                // — after the loop above, because that loop is what names the
                // refs on this device and the model's list join resolves every
                // ref or refuses the whole packet. The guest is declaring it
                // CPU-authored these resources' pages: authority moves to the
                // guest replica, outside the GPU timeline, and the device's own
                // validity quads above are the host-cache half of the same
                // event.
                if let Some(acted) = lifetime_operation(
                    state,
                    host,
                    reims_vgpu_protocol::packets::Channel::Child,
                    packet.opcode,
                    &packet.payload,
                )
                .and_then(|op| state.apply_lifetime(&op, family.slug()))
                {
                    note_inert_lifetime_effects(family.slug(), &acted);
                }
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
                crate::observe::verbose(|| {
                    format!(
                    "map_family op=InvalidateResources opcode={:#x} ch={channel_id} plen={plen} task={} count={} oid={oid:#x} flags={flags:#x} clr_h={} set_h={} clr_g={} set_g={} pageon={pageon} bumped={bumped} miss={miss}",
                    packet.opcode,
                    cmd.task_id,
                    cmd.count,
                    ops.clear_host_valid,
                    ops.set_host_valid,
                    ops.clear_guest_valid,
                    ops.set_guest_valid
                )
                });
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
            Err(e) => {
                note_resource_list_decode_fail("InvalidateResources", packet.opcode, channel_id, e)
            }
        }
    } else if matches!(
        family,
        MapFamily::SynchronizeResources | MapFamily::SynchronizeAndDiscardResources
    ) {
        // Two opcodes, one arm, because the reference host validates
        // both with byte-for-byte the same check — `{u32 task, u32
        // count}` then `count` 4-byte object ids — and the synchronise
        // obligation they name is the same one. `0x3e` then releases the
        // named resources' transfer backings while preserving their host
        // textures and resource identities.
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
        use crate::protocol::fifo::decode_synchronize_resources;
        match decode_synchronize_resources(&packet.payload) {
            Ok(cmd) => {
                crate::runtime::objects::note_lifetime_refs_named(
                    state,
                    host,
                    cmd.task_id,
                    &cmd.object_ids,
                );
                // The command, in the model that owns content authority — after
                // the naming above, because every ref resolves or the whole
                // packet refuses, and this device names a reference the first
                // time anything asks what it names. The model's answer is the
                // deferred discards: a synchronise-and-discard offers each
                // resource's copy for release, and the offer is taken only where
                // nothing else still answers for those bytes.
                //
                // The transfers it can also owe are structurally none here —
                // see `crate::model::state`'s effects door — and the
                // resource-scoped submit below is this device's own deferred
                // Store obligation, which is a different fact from content
                // authority and stays where it is.
                let deferred = lifetime_operation(
                    state,
                    host,
                    reims_vgpu_protocol::packets::Channel::Child,
                    packet.opcode,
                    &packet.payload,
                )
                .and_then(|op| state.apply_lifetime(&op, family.slug()))
                .map(|acted| acted.at_completion.len())
                .unwrap_or(0);
                note_store_route_n("lifetime_discard_offered", deferred as u64);
                // Synchronization is resource-scoped. Apple batches the named
                // resources into transfer encoders; synchronizing one object
                // does not publish every other host-valid texture in the task.
                crate::runtime::writeback_debt::submit_for_resources(
                    state,
                    host,
                    cmd.task_id,
                    &cmd.object_ids,
                );
                if family == MapFamily::SynchronizeAndDiscardResources {
                    let discarded = crate::runtime::writeback_debt::discard_gva_resources(
                        state,
                        cmd.task_id,
                        &cmd.object_ids,
                    );
                    if discarded != 0 {
                        note_store_route_n("gva_transfer_backing_discarded", discarded as u64);
                    }
                }
                let oid = cmd.object_ids.first().copied().unwrap_or(0);
                crate::observe::verbose(|| {
                    format!(
                        "map_family op={name} opcode={:#x} ch={channel_id} plen={plen} task={} count={} oid={oid:#x}",
                        packet.opcode, cmd.task_id, cmd.count
                    )
                });
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
    } else {
        let w0 = if plen >= 4 {
            crate::protocol::endian::ld32(&packet.payload[0..])
        } else {
            0
        };
        let w1 = if plen >= 8 {
            crate::protocol::endian::ld32(&packet.payload[4..])
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

/// Whether a child opcode is one of the reference host's retired slots.
///
/// **The one place this device asks that question.** The fifteen numbers the
/// reference host routes to its shared deprecated handler used to be
/// transcribed into `model::regs` beside this crate's own command constants, so
/// the closure ledger and the drain each carried a list and a slot that went
/// live had to be remembered in both. The ledger is the authority now: a row it
/// judges `ProvenNoOp` is what
/// [`reims_vgpu_core::control::ControlKind::of`] calls a
/// [`RetiredSlot`](reims_vgpu_core::control::ControlKind::RetiredSlot).
///
/// This is deliberately *not* the wider `payload_is_inert` question, which is
/// also true of `CmdNOP`. `CmdNOP` is a live command with a constant of its own
/// and an arm that names it, and `dispatch_agrees_with_the_packet_ledger` reads
/// exactly that: every live command is named by its constant, and the retired
/// slots — which have no constant, by design — are the only rows the dispatch
/// reaches without naming. Selecting `CmdNOP` by kind instead of by constant
/// would move it into that asymmetry and make the scan's claim weaker for
/// nothing.
///
/// Retired is about the payload and never about the packet. The stamp waits,
/// the ordering position and the completion word are owed identically, and the
/// drain discharges them for every accepted packet whatever this returns.
fn is_retired_control_slot(opcode: u16) -> bool {
    ControlKind::of(WireChannel::Child, opcode) == Some(ControlKind::RetiredSlot)
}

/// What `SessionModel::apply_control` answered this channel transition.
///
/// # Two refusals this device did not have
///
/// The model's channel set has two guarded doors, and this device had neither
/// guard while it kept its own bit:
///
/// * a definition of a domain already open is `ChannelAlreadyOpen`. Here an open
///   was `active_child_mask |= bit` — idempotent — and a redefinition is a real
///   event on this interface: `forget_child_channel` resets the ring cursor and
///   the translation masks precisely so a redefined FIFO does not inherit the
///   previous producer's position.
/// * a free of a domain no definition opened is `ChannelNotOpen`, and one that
///   still holds unreleased publication positions is `Owed`. Here a free was
///   `&= !bit` whatever the domain's history.
///
/// A refusal withholds no completion word — a control transaction publishes its
/// stamp whether its transition happened or not, which the model's own doc
/// states, and the drain still owns that envelope. What a refusal costs is the
/// model-side *effect*: a refused redefinition leaves the domain open where this
/// device would have re-opened it, and a refused free leaves it open too. The
/// device-side effects either way are unchanged, because
/// `forget_child_channel` runs on the transition and not on the verdict.
///
/// `channel_transition_model_agrees` is the denominator, without which a boot
/// that sent no channel commands reads like a clean one.
///
/// # What a driven boot answered, and why five is the population
///
/// x86 Vulkan, macos-15, host-driven workload — three rounds of launching and
/// quitting five applications over ssh, 2049 child packets against an idle
/// boot's 1916: **`channel_transition_model_agrees=5`,
/// `channel_open_of_open_domain=0`, `channel_free_of_undefined_domain=0`.**
///
/// The five did not move under the workload, and that is the finding rather than
/// a thin sample. This guest establishes its FIFO set once, at accelerator
/// start, and per-application GPU contexts share those channels — launching and
/// quitting applications creates and destroys no domain. So five is what a boot
/// *has*, the model refuses none of them, and that is what let this lifetime
/// move.
fn note_channel_transition_verdict(
    verdict: Result<(), reims_vgpu_core::session::ControlRefusal>,
    transition: control::ChannelTransition,
    channel: u32,
) {
    let Err(refusal) = verdict else {
        note_store_route("channel_transition_model_agrees");
        return;
    };
    let route = match transition {
        control::ChannelTransition::Open => "channel_open_of_open_domain",
        control::ChannelTransition::Free => "channel_free_of_undefined_domain",
    };
    note_store_route(route);
    if crate::observe::first_sight(route, u64::from(channel)) {
        crate::observe::fail(format!(
            "{route} channel={channel} reason={} (the guest asked for a transition the \
             domain's lifetime owner refuses, so the domain is left as it was)",
            refusal.slug()
        ));
    }
}

/// Whether a packet's own domain was opened by a channel definition, which is
/// the gate the replacement model applies and this device does not.
///
/// # The question this exists to answer, and why a reading cannot
///
/// `reims_vgpu_core::session::SessionModel::admit` refuses any packet whose
/// domain is not in `open_channels`, and `open_channels` has exactly one event:
/// `ControlOp::Channel { transition: Open }`. This device has three events that
/// make a child domain live — that packet, the locked `GFX_REG_CHILD_DOORBELL`
/// write in `crate::runtime::mmio`, and the lock-free ring
/// [`fold_rung_child_doorbells`] folds in — and two of them say nothing about a
/// definition. `DeviceState::active_child_mask` is the union, so it cannot tell
/// the three apart after the fact.
///
/// That difference decides whether channel lifetime can be the first group to
/// cut over. If this guest ever puts work on a domain it never defined, the
/// model's gate turns every packet on that domain into a typed refusal — which
/// is a hang that names itself, and still a hang. If it never does, the gate is
/// a restatement of what this device already requires and the group can move.
///
/// # It selects nothing
///
/// The disposition is unchanged either way. This reads a mask no behaviour
/// consults and writes two routes, which is what
/// `crates/reims-vgpu-observe` is for: it describes the decision the model
/// *would* make, on a device that is not making it.
///
/// `child_packet_domain_defined` is the denominator's other half — without it a
/// boot where every packet's domain was defined and a boot where the guest sent
/// no child packets at all read the same.
///
/// # What a driven boot answered
///
/// x86 Vulkan, macos-15, 391 s to the desktop with a guest frame presented:
/// **`child_packet_domain_defined=1395`, `child_packet_domain_undefined=0`**,
/// and **2049 / 0** on a later boot driven through three rounds of application
/// launches and quits.
/// Every child packet this guest sent was on a domain a `CmdDefineChannel` had
/// opened, so the model's `ChannelNotOpen` gate would have refused none of
/// them. The doorbell-only path is real in this device's code and this guest
/// does not carry packets on a domain it did not define.
///
/// That is one of the three terms the channel-lifetime group needs, and it is
/// the one that would have been a hang. The other two are what
/// [`note_channel_transition_verdict`] counts, and they are not hangs.
fn note_packet_domain_definition(state: &DeviceState, channel_id: u32, opcode: u16) {
    if state.child_domain_open(channel_id) {
        note_store_route("child_packet_domain_defined");
        return;
    }
    note_store_route("child_packet_domain_undefined");
    if crate::observe::first_sight(
        "child_packet_domain_undefined",
        u64::from(channel_id) << 32 | u64::from(opcode),
    ) {
        crate::observe::fail(format!(
            "child_packet_domain_undefined channel={channel_id} op={opcode:#x} (this domain \
             was made live by a doorbell and never by a channel definition, so the \
             replacement model would refuse every packet on it — the definition is the \
             only event its open-channel set has)"
        ));
    }
}

/// Do one child packet's host work.
///
/// `retained` is what a caller running a packet at an ordering position holds:
/// the command buffers read out of the task's address space **before** the ring
/// head moved past this packet — the exec class's only input that is not in the
/// packet — together with the records the model resolved those buffers into.
/// `None` means "read it now", which is what a caller holding the packet at its
/// ring position does. The retained form is the same bytes the transaction was
/// admitted against rather than whatever the guest has since put at those
/// addresses, and the same records it was admitted as.
fn process_child_packet<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    channel_id: u32,
    packet: &Packet,
    retained: Option<crate::runtime::exec::RetainedInputs<'_>>,
) -> ChildPacketDisposition {
    note_packet_domain_definition(state, channel_id, packet.opcode);
    note_packet_class(reims_vgpu_protocol::packets::Channel::Child, packet.opcode);
    note_lifetime_task_definition(state, reims_vgpu_protocol::packets::Channel::Child, packet);
    match packet.opcode {
        CHILD_OP_DEFINE_TASK2 => {
            apply_define_task2(state, host, &packet.payload, Some(channel_id));
        }
        // A short SET_OBJECT_LIST leaves the task's object list unbound — every
        // mapper-ref-texture/object resolve on it then fails
        // (object_list_count==0). Never on a well-formed boot.
        CHILD_OP_SET_OBJECT_LIST => {
            apply_set_object_list(state, &packet.payload, Some(channel_id));
        }
        CHILD_OP_DELETE_RESOURCE => {
            // The same `{task_id, object_id}` record `CmdReplacePhysical`
            // carries, through the same decode. This arm had its own literal
            // `8` and its own two `ld32`s at literal offsets, which is a fourth
            // spelling of one pair — and the third command with these two words
            // carries them the other way round.
            match crate::protocol::fifo::decode_task_object(&packet.payload) {
                Err(short) => note_short_payload("delete_resource", Some(channel_id), &short),
                Ok(crate::protocol::fifo::TaskObjectCommand {
                    task_id,
                    object_id: id,
                }) => {
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
                    if crate::runtime::writeback_debt::retire_gva_resource(state, task_id, id) {
                        note_store_route("gva_resource_retired");
                    }
                    let _ = state.delete_object(task_id, id);
                }
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
         * - Early boot: front mapper-ref-texture writebacks paint while !frame_flush_seen
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
        opcode @ (CHILD_OP_DISPLAY_SWAP
        | CHILD_OP_DISPLAY_TRANSACTION2
        | CHILD_OP_DISPLAY_TRANSACTION3) => {
            // Which of the three, and therefore where every word is, is
            // `reims_vgpu_protocol::present`'s: three commands with three
            // trailers, and one place that says so.
            let form = PresentForm::of(WireChannel::Child, opcode)
                .expect("the three present opcodes are this arm's own pattern");
            let decoded = match present_trailer(form, &packet.payload) {
                Ok(decoded) => decoded,
                Err(refusal) => {
                    let present::Refusal::Short { have, need } = refusal;
                    crate::observe::fail(format!(
                        "packet_short reason={} ch={channel_id} op={opcode:#x} plen={have} \
                         need={need}",
                        refusal.slug()
                    ));
                    return ChildPacketDisposition::Complete;
                }
            };
            let mapping = decoded.target;
            note_display_txn_payload(state, channel_id, packet, &decoded);
            // Per-present decode census (~30k/session under animation); the
            // present rate lives in the present_proxy summary, so gate the
            // per-packet line behind REIMS_VGPU_DRAW_LOG. `pipe` is the display
            // index for op8 and the pipe index for op6/7 — payload word 0 in
            // both. `task` is op6/7's task field, which is the submitting task's
            // and not a completion stamp; the packet's own stamp lives in the
            // FIFO header. op8 has no such word, and prints `-`.
            let task = decoded
                .task
                .map_or_else(|| "-".to_string(), |task| format!("{task:#x}"));
            crate::observe::line(format!(
                "present_txn op={opcode:#x} ch={channel_id} pipe={} sid={mapping} task={task} \
                 plen={} unpainted={} prior_present_mapping={}",
                decoded.pipe,
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
            match crate::protocol::fifo::decode_delete_object(&packet.payload) {
                Ok(command) => apply_delete_object(state, host, channel_id, &command, packet),
                Err(error) => note_short_payload(error.slug(), Some(channel_id), &error.short()),
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
            // The floor is the decoder's, and it is the only one. This arm and
            // `reply_compute_info` each carried their own literal `24`, so
            // neither check could be wrong without the other being wrong too —
            // and no test built a request at all, so neither was exercised.
            match query_request(
                WireChannel::Child,
                packet.opcode,
                &packet.payload,
                Some(channel_id),
            ) {
                Some(RequestWords::ComputeInfo(request)) => {
                    let _ = reply_compute_info(state, host, &request);
                }
                Some(_) => note_query_layout_mismatch("compute_info", Some(channel_id)),
                None => {}
            }
        }
        CHILD_OP_CURSOR_SHOW => match crate::protocol::fifo::decode_cursor_show(&packet.payload) {
            Ok(show) => {
                state.cursor.show = show;
                sample_cursor_position(state, host);
                host.enqueue(HostAction::cursor(state.cursor.x, state.cursor.y, show));
            }
            Err(short) => note_short_payload("cursor_show", Some(channel_id), &short),
        },
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
                // the specific mapper-ref-texture/GVA key at sample/Load/swap sites.
                // A retained submission runs planned: whether its translations
                // were done is what decided the packet could run at all, so
                // asking again would be paying per pipeline for an answer the
                // model already acted on.
                // **There is one execution path and this is it.** An exec
                // packet whose command buffers could not be read at arrival is
                // refused by the ingress bridge (`Gap::ExecStreamsUnread`) and
                // never reaches an ordering position, and a packet that did
                // reach one carries both halves by construction — so `None`
                // here is this device having lost the bytes between admission
                // and release, not a guest fact. Reading the payload again to
                // recover would read whatever the guest has since put at those
                // addresses, which is the second reader the switch exists to
                // remove; the position is named and nothing runs.
                let Some(inputs) = retained else {
                    note_store_route("exec_retained_inputs_missing");
                    crate::observe::fail(format!(
                        "exec_retained_inputs_missing reason=parked_without_submission                          ch={channel_id} (an admitted exec packet reached execution without                          the command buffers it was admitted against, so its records cannot                          be run from the bytes the transaction named)"
                    ));
                    return ChildPacketDisposition::Complete;
                };
                let result = crate::runtime::exec::execute_planned(
                    state,
                    host,
                    inputs,
                    crate::runtime::exec::ExecResult::default(),
                );
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
                } else {
                    crate::observe::verbose(|| {
                        exec_summary(channel_id, &result, packet.payload.len())
                    });
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
                    // What the engine was holding while this tranche waited.
                    // This line is the only join anywhere between a stalled
                    // submission and the pipeline objects in it, and a host GPU
                    // hang is the case it exists for: the counts above say a
                    // tranche took seconds, and nothing else says what it took
                    // them on. Emitted beside the count rather than folded into
                    // it because the two have different lengths and a reader
                    // greps for one or the other.
                    if let Some(trail) = crate::runtime::gpu_hang_trail::trail() {
                        crate::observe::fail(format!(
                            "TRANSPORT reason=sync_exec_lock_hold_trail ch={channel_id} {trail}"
                        ));
                    }
                    // The trail above is what this device *recorded*; this is
                    // what it is still *waiting on*. They are different
                    // questions and the second is the one a stall asks: the
                    // oldest outstanding submission is the one every later one
                    // is queued behind, and a stall whose oldest carries
                    // `draws=0` is a wedge that is not in a draw at all.
                    //
                    // `None` here is a reading rather than a gap — it says this
                    // tranche is blocked on something the submission ring did
                    // not submit.
                    if let Some(outstanding) = crate::runtime::gpu_hang_trail::outstanding() {
                        crate::observe::fail(format!(
                            "TRANSPORT reason=sync_exec_lock_hold_outstanding ch={channel_id} \
                             {outstanding}"
                        ));
                    }
                    // The trail above is the last twelve draws, which at this
                    // rail's rate is the last half millisecond — and a wedged
                    // device goes on drawing about one draw per stall, so by the
                    // second stall the trail is entirely post-wedge. The first
                    // stall's is not, but even that one reaches only the
                    // millisecond before the wait, while the wedge on the rail
                    // this was built for begins in the ~300 ms after an
                    // application's first window.
                    //
                    // Latched to the first stall of the boot, on purpose. It is a
                    // per-process record, so every later stall would print the
                    // same list one draw further on and the log would carry a
                    // dozen near-identical copies of a line whose value is that
                    // there is one of it.
                    if crate::observe::first_sight("sync_exec_lock_hold_pipes", 0) {
                        if let Some(firsts) =
                            crate::runtime::gpu_hang_trail::recent_pipeline_firsts()
                        {
                            crate::observe::fail(format!(
                                "TRANSPORT reason=sync_exec_lock_hold_pipes ch={channel_id} \
                                 {firsts} (the pipelines this device drew for the first time \
                                 most recently, oldest first, each with how many draws ago — \
                                 a wedge in the second an application opens its first window \
                                 has new pipelines in front of it and nothing else says which)"
                            ));
                        }
                    }
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
            // One bounds check, in the decoder that owns the layout. This arm
            // used to make its own with the command's length constant and then
            // call a decoder that checked the same floor again — so the second
            // check was unreachable, and its `None` arm would have dropped the
            // packet in silence if it ever were reached.
            match crate::protocol::fifo::decode_task_object(&packet.payload) {
                Ok(cmd) => {
                    // The GPA behind this resource changes here. The command is
                    // scoped to one task-local resource id, so unrelated
                    // resources on the same task keep both their authoritative
                    // frame and their held address resolution.
                    if crate::runtime::writeback_debt::retire_gva_resource(
                        state,
                        cmd.task_id,
                        cmd.object_id,
                    ) {
                        note_store_route("gva_resource_retired");
                    }
                    note_bb_retired(
                        "bb_retire_replace_physical",
                        state.retire_bound_buffers_for_ref(cmd.task_id, cmd.object_id),
                    );
                    crate::runtime::objects::replace_physical(
                        state,
                        host,
                        cmd.task_id,
                        cmd.object_id,
                    );
                    // And then the model, in that order. The re-point's
                    // operation names pages the packet does not carry, so the
                    // storage is asked for *after* this device has moved the
                    // incarnation — see `objects::repointed_storage` for why an
                    // earlier answer is the old one.
                    match crate::runtime::objects::repointed_storage(
                        state,
                        host,
                        cmd.task_id,
                        cmd.object_id,
                    ) {
                        Ok((resource, backing, extent)) => {
                            if let Some(acted) = state.apply_lifetime(
                                &reims_vgpu_core::lifecycle::LifecycleOp::ReplacePhysical {
                                    task: reims_vgpu_core::identity::TaskId(cmd.task_id),
                                    resource,
                                    backing,
                                    extent,
                                },
                                "replace_physical",
                            ) {
                                note_store_route_n(
                                    "replace_physical_teardowns",
                                    acted.teardowns.len() as u64,
                                );
                            }
                        }
                        // The reference names no describable storage, so there
                        // is no operation to state. Counted by reason rather
                        // than as one bucket: a reference that owns no bytes is
                        // a non-event, while storage this device cannot
                        // describe is a re-point the model never hears about,
                        // which leaves its content authority on the old pages.
                        Err(refusal) => {
                            note_store_route(refusal.route());
                            // The type pair, separately. `route` puts a
                            // no-bytes reference whose integer also names a
                            // live mapping surface under the mapping arm — the
                            // one that would mean a real re-point went unheard
                            // — and that shadows the kind the guest's own list
                            // is holding there. Both are needed: the integer
                            // agreeing is a collision until the type agrees too.
                            if matches!(
                                refusal,
                                crate::runtime::objects::RepointStorageRefusal::NoBytes { .. }
                            ) {
                                note_store_route(refusal.no_bytes_kind_route());
                            }
                            if refusal.leaves_authority_stale()
                                && crate::observe::first_sight(
                                    "replace_physical_storage_undescribable",
                                    u64::from(cmd.task_id) << 32 | u64::from(cmd.object_id),
                                )
                            {
                                crate::observe::fail(format!(
                                    "replace_physical_storage_undescribable \
                                     task={} object={} reason={} (the guest re-pointed \
                                     storage this device cannot describe to the semantic \
                                     model, so the model's content authority for it stays \
                                     on the pages the guest has already rewired)",
                                    cmd.task_id,
                                    cmd.object_id,
                                    refusal.route(),
                                ));
                            }
                        }
                    }
                }
                Err(short) => {
                    note_short_payload("replace_physical", Some(channel_id), &short);
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
            apply_map_family(
                state,
                host,
                channel_id,
                packet,
                MapFamily::InvalidateResources,
            );
        }
        CHILD_OP_SYNCHRONIZE_RESOURCES => {
            apply_map_family(
                state,
                host,
                channel_id,
                packet,
                MapFamily::SynchronizeResources,
            );
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
        // same payload contract. It releases each resource's transfer backing;
        // prepare or synchronize recreates that backing lazily if needed.
        //
        // The payload is still decoded, with `0x35`'s decoder because the host
        // validates all three with the same check. A malformed one is worth a
        // line even though a well-formed one would have been ignored: it says
        // the guest and this device disagree about the record layout, which the
        // two commands that *do* act on it share.
        CHILD_OP_DISCARD_RESOURCES => {
            use crate::protocol::fifo::decode_synchronize_resources;
            match decode_synchronize_resources(&packet.payload) {
                Ok(cmd) => {
                    crate::runtime::objects::note_lifetime_refs_named(
                        state,
                        host,
                        cmd.task_id,
                        &cmd.object_ids,
                    );
                    // The model's half of the same release, after the naming
                    // above for the reason the list join gives: every ref
                    // resolves or the whole packet refuses. It offers each
                    // resource's copy at completion; this arm's own release is
                    // of the transfer staging, which the model has no key for.
                    let deferred = lifetime_operation(
                        state,
                        host,
                        reims_vgpu_protocol::packets::Channel::Child,
                        packet.opcode,
                        &packet.payload,
                    )
                    .and_then(|op| state.apply_lifetime(&op, "discard_resources"))
                    .map(|acted| acted.at_completion.len())
                    .unwrap_or(0);
                    note_store_route_n("lifetime_discard_offered", deferred as u64);
                    let discarded = crate::runtime::writeback_debt::discard_gva_resources(
                        state,
                        cmd.task_id,
                        &cmd.object_ids,
                    );
                    if discarded != 0 {
                        note_store_route_n("gva_transfer_backing_discarded", discarded as u64);
                    }
                }
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
        //
        // Which numbers those are is the closure ledger's judgement, read
        // through [`is_retired_control_slot`]. This device used to carry its own
        // fifteen-entry transcription of them, which is one list too many for a
        // set whose whole content is "the ledger proved these do nothing".
        op if is_retired_control_slot(op) => {
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
    let setup_started = std::time::Instant::now();
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
    // The channel-id bound again, in the vocabulary the model's packets carry.
    // `child_reg_block_offset` above already refused every id `is_child_channel`
    // refuses, so this cannot be `None` — and it is asked rather than
    // constructed because a `Fifo` is what makes "which opcode table" and
    // "which ordering domain" one answer.
    let Some(fifo) = crate::runtime::ingress::Fifo::child(channel_id) else {
        return;
    };
    let page_gpas = state.child_rings[channel_id as usize].page_gpas.clone();
    census::note_drain_setup(setup_started.elapsed().as_nanos() as u64);

    // Nested drain_other must skip this channel (no re-enter head).
    // Use a bit mask so nested drains skip the full stack, not only the leaf.
    let prev_channel = state.draining_channel;
    let bit = 1u32 << channel_id;
    state.draining_channel = channel_id;
    state.draining_mask |= bit;

    loop {
        let regs_started = std::time::Instant::now();
        let tail_read = crate::runtime::host::read_u32(host, regs_gpa + CHILD_REG_TAIL);
        census::note_drain_regs(
            census::RegsOp::TailRead,
            regs_started.elapsed().as_nanos() as u64,
        );
        let tail = match tail_read {
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
        let ring = Ring::Child {
            page_gpas: &page_gpas,
            capacity: ring_length,
            page_shift: state.page_shift,
        };
        match arrival(&ring, host, head, tail) {
            Arrival::Nothing => break,
            Arrival::Fault(fault) => {
                state.record_fail(FailEvent::MalformedChildPacket {
                    channel: channel_id,
                    fault,
                    head,
                });
                break;
            }
            Arrival::Packet(packet) => {
                // The census the model's own stamp plane now acts on. Asked for
                // the reading and not for a verdict: a wait that is not yet met
                // is an ordering position in `admit`, not a ring head this loop
                // refuses to move.
                note_packet_stamp_waits(state, host, Some(channel_id), &packet);

                // **The head advances first, and unconditionally.** Everything
                // this packet needs has been taken out of the ring — the
                // snapshot is decoded and the exec class's command buffers are
                // read at arrival — so the ring position is free whatever the
                // model then decides. That is the switch: a packet that cannot
                // run yet holds an ordering position instead of a consumer
                // pointer, and the packets behind it are not behind anything.
                head = packet.next_head;
                let head_started = std::time::Instant::now();
                let head_write = gpa_map::write_u32(
                    host,
                    regs_gpa + CHILD_REG_HEAD,
                    head,
                    state.page_size() as usize,
                );
                census::note_drain_regs(
                    census::RegsOp::HeadWrite,
                    head_started.elapsed().as_nanos() as u64,
                );
                if head_write.is_err() {
                    // The consumer pointer never advanced: the next drain
                    // re-reads the stale head and admits the same packets again.
                    // Fail-visible so that silent replay is diagnosable
                    // (drain.rs Rank-1 audit).
                    state.record_fail(FailEvent::MalformedChildPacket {
                        channel: channel_id,
                        fault: PacketFault::ChildHeadWriteback,
                        head,
                    });
                }

                admit_and_park(state, host, fifo, stamp_index, packet);
                settle_model_work(state, host);

                if state.pending.host_action_yield {
                    if head != tail {
                        state.pending.child_mask |= bit;
                    }
                    break;
                }
            }
        }
    }

    // Every `break` above lands here, and what it owes is the same thing a
    // drain that ran the ring dry owes: a last look at what the model has
    // released. A drain that stopped on a decode fault or a host-action yield
    // may still be holding positions that became runnable while it was working,
    // and leaving one unrun is a guest waiting on a word that never arrives.
    settle_model_work(state, host);

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

/// Display-side present completion: after `presentFrame` retains the surface,
/// set pending bit 1 on the display shared page, read the enable mask, and poke
/// the display IRQ when the guest asked for present notifications. This is the
/// guest's frame-done pacing edge — separate from the packet header stamp (the
/// swap fence). Without it the guest keeps swapping (fence releases) but never
/// receives the per-present display event.
///
/// **This edge is a wakeup as well as a notification, and one guest wait has no
/// deadline.** The guest's display pipe drains its transaction queue by sleeping
/// on its command gate until the consumed and submitted indices meet. The
/// per-transaction wait is bounded — a second, after which it re-tests and logs
/// — but the *queue-idle* wait taken on the WSAA defer transition is not: it
/// sleeps with no deadline and is woken only by the transaction interrupt this
/// bit raises. A device that finishes the work and never rings this bell
/// therefore hangs that guest permanently rather than slowly, and the guest is
/// blocked in the window server, before any frame it could present to unblock
/// itself. Completion of the underlying event is necessary and not sufficient;
/// the wakeup is the other half.
pub fn signal_display_present_complete<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
) {
    let gpa = state.display.shared_gpa;
    if gpa == 0 {
        note_display_present_signal(DISPLAY_PRESENT_NO_GPA);
        return;
    }
    // The mask gates the pending write and not just the interrupt. It used to
    // gate only the interrupt, which set a bit for a guest that had declined
    // the class and could therefore never clear it — see
    // [`display_event_enabled`] for why that residue is not inert.
    if !display_event_enabled(host, gpa, DISPLAY_PRESENT_EVENT_MASK) {
        note_display_present_signal(DISPLAY_PRESENT_NOT_ENABLED);
        return;
    }
    note_display_present_signal(DISPLAY_PRESENT_DELIVERED);
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
    let bit = 1u32 << (state.display.display_index & 0x1f);
    state
        .gfx
        .interrupt_status_disp
        .fetch_or(bit, std::sync::atomic::Ordering::AcqRel);
    host.enqueue(HostAction::irq_gfx());
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

/// Whether the guest has asked to be told about this class of display event.
///
/// `+0x104` is the guest's own statement of which pending bits it will act on,
/// and its interrupt handler is what makes that binding rather than advisory:
/// it read-clears `pending & enable_mask` and **leaves every bit outside the
/// mask exactly where it found it**. So a bit this device sets for a disabled
/// class is not a notification the guest ignores — it is a word the guest will
/// never clear, which every later read-modify-write of the pending word then
/// carries forward for the life of the boot.
///
/// Both x86 rails measured here disable classes this device was signalling
/// anyway, and both showed the litter: with the display idle and the guest
/// healthy, `+0x100` read `0x1` under a macOS 11 guest whose mask was `0xe`,
/// and `0x3` under a macOS 13 guest whose mask was `0xc`. In each case the
/// residue is exactly the set of bits the device set and the guest had not
/// enabled.
///
/// An unreadable mask answers `false`. The guest published this page's address
/// itself, so a read of it that the host cannot perform is not a reason to
/// start signalling classes nobody asked for; the ONLINE handshake takes the
/// same view of the same read.
///
/// **The mask is dynamic, so it is re-read per signal rather than latched at
/// setup.** `enableVBLInterrupt` and `disableVBLInterrupt` are a `lock or 1`
/// and a `lock and ~1` on this same word, and both x86 rails use them that way:
/// the guest arms VBL while it is compositing and disarms it when it goes idle.
/// A driven macos-13 boot read `delivered=3757` and then flat with
/// `not_enabled` climbing at the grid rate; an undriven macos-11 boot read
/// `delivered=299` and went the same way. Latching the mask would either starve
/// a compositing guest or keep signalling an idle one, depending on when the
/// latch was taken.
///
/// **This is the only reader of the enable word, deliberately.** It used to have
/// a second one: the lock-free VBL pulse in `device::vbl_contended_pulse` wrote
/// the pending bit without consulting the mask at all, so the same wire form had
/// two arms and the one a contended poll took was missing the term. That cost
/// twice over — it manufactured the residue this doc describes, and it counted
/// those writes as `delivered`, inflating the one census number a stalled rail
/// is ranked by. Both arms call this now; keep it that way.
pub(crate) fn display_event_enabled<H: HostMemory>(host: &H, gpa: u64, event_mask: u32) -> bool {
    let mut mask_le = [0u8; 4];
    if host
        .read_gpa(gpa + DISPLAY_SHARED_ENABLE_MASK, &mut mask_le)
        .is_err()
    {
        return false;
    }
    let mask = ld32(&mask_le);
    // Every signal path passes through here, so this is where the guest's own
    // statement of what it wants is observable without stopping the world.
    crate::runtime::drain::census::note_display_enable_mask(mask);
    mask & event_mask != 0
}

/// Signal every display-completion class the guest has armed, for one tick of
/// the refresh grid.
///
/// # Why this is one function and not two
///
/// A display pipe has to be told when the frame it was showing is done with.
/// **Which event class carries that news is the guest's choice, published in the
/// enable word, and the generations do not agree.** Measured on this rig, on
/// consecutive boots of the same device:
///
/// - a macOS 13 guest oscillates its mask `0xc <-> 0xd`, arming and disarming
///   **VBL** around each compositing burst, and never arms the transaction
///   class at all;
/// - a macOS 11 guest sets its mask to `0xe` two seconds after display setup,
///   arming the **transaction** class, and never arms VBL for the life of the
///   boot.
///
/// They are asking for the same thing by different names. On the guest side the
/// transaction class runs `transaction_interrupt_gated`, which completes the
/// *live* transaction — the one on screen — and then consumes the ring. It is a
/// per-refresh retirement edge, not a per-present notification, and a device
/// that raises it only when a present happens rings it exactly once and then
/// stops.
///
/// That is a deadlock rather than a slowdown, because the wait it feeds has no
/// deadline: the guest's window server drains the transaction queue on its
/// accelerated-access transition, and that drain sleeps until the queue is empty
/// *and* no transaction is live. With one frame presented, the queue empties,
/// the frame becomes live, and only a further transaction interrupt would retire
/// it. None arrives, so the window server never returns — before any user
/// session exists, and before it could present anything that might unblock it.
///
/// So the tick signals whatever the guest armed. This is not a per-generation
/// branch: nothing here asks which guest is running, it reads the one word the
/// guest publishes to say what it wants. A guest that arms both gets both; a
/// guest that arms neither gets a counted refusal and no write.
///
/// One mask read, one pending read-modify-write and at most one interrupt per
/// tick, so arming both classes does not double the doorbell rate.
///
/// # The mask is read before the grid slot is claimed, and that ordering is load-bearing
///
/// A macOS 13 guest arms VBL **one shot at a time**: it sets bit 0, takes the
/// delivery, clears it inside its handler, and sets it again when it next wants
/// a frame. So on any given tick the mask is as likely to be mid-turnaround as
/// armed, and a tick that samples it disarmed owes nothing.
///
/// Both callers used to claim the slot *first* and read the mask second, which
/// meant a disarmed sample advanced the grid timestamp and **spent the slot the
/// guest was about to ask for**. The guest then re-armed a millisecond later and
/// waited a further full interval, so its delivery rate aliased to every other
/// grid point — 60 Hz out of an advertised 120.
///
/// # What it is worth, and the thing it does not explain
///
/// Six interleaved driven macos-13 boots, binary A/B (the ordering has no env
/// switch), quiesced host. Delivered rate is taken from the gap between
/// consecutive `arm=delivered` census lines, which are exactly 1024 deliveries
/// apart — the `delivered=` field on the last line is quantised to that cadence
/// and `window_hz` is unusable here, see [`census::VblCensus`].
///
/// ```text
///            presented frames/s   delivered Hz (final window)
/// old  A1           100.0                 92.8
/// old  A3           101.5                107.8
/// new  B1            99.0                110.0
/// new  B2           111.0                112.5
/// old  A2            60.0                 43.8
/// new  B3            59.0                 67.5
/// ```
///
/// Delivery rises on the new ordering in both populations, and in the fast one
/// the arms are nearly disjoint against a 118 Hz grid ceiling. That is the
/// mechanism showing up where it should: a guest re-arming every 8.33 ms races
/// an 8.47 ms grid, so its turnaround lands past the grid point often, and it is
/// precisely those ticks the old ordering spent.
///
/// **It does not by itself explain the boot-to-boot regime split, and the first
/// version of this doc claimed it did.** The split came out 2 fast to 1 slow on
/// *both* arms at n=3. The slow arm is not a steady-state delivery shortfall this
/// ordering repairs: B3 received 54 % more VBL than A2 and presented the same 59
/// frames a second.
///
/// The split is a property of the boot, not of the workload: a boot presents
/// either ~60 or ~95-117 frames a second for its whole life, with nothing in
/// between. What decides it is still open — see `VBL_REPORT_EARLY` beside
/// [`census::VblCensus`], which also records the plausible-looking early-window
/// explanation that twenty boots killed. So the outcome to rank this ordering
/// against is a probability over boots and not a rate.
///
/// **It does not change it.** Forty interleaved driven macos-13 boots, twenty an
/// arm, no guest panic and no boot lost:
///
/// ```text
///          latched fast   latched slow   rate
/// old            13             7        0.65
/// new            15             5        0.75
/// ```
///
/// That is p≈0.7 — nothing. The 9-in-11 against 4-in-9 that earlier runs hinted
/// at was the non-interleaving showing through, which is exactly what the run
/// was set up to falsify. So this ordering is kept on its own merits: it is what
/// the contract says (a tick that delivers nothing has no business spending the
/// interval), it is unit-tested, and it raises the delivered rate within a boot.
/// It is not a frame-rate fix and must not be quoted as one.
///
/// The same forty boots put the base rate at **28 fast to 12 slow**, so about
/// three boots in ten lose half their frame rate on this rail, and they are
/// sharply bimodal — every slow boot presented 59.8-60.5 Hz and every fast one
/// 94.8-117.0, with nothing in between.
///
/// Reading the mask first cannot deliver faster than the advertised rate —
/// [`claim_display_vbl`] still gates every delivery on a full interval having
/// elapsed since the last one. It only stops the device spending intervals on
/// ticks that deliver nothing, which makes the delivered rate converge on the
/// grid the timing table advertises instead of on a fraction of it.
///
/// The cost is that the four-byte enable-mask read now happens once per poll
/// (~240 Hz) rather than once per claimed tick while the guest sits disarmed.
/// That read is of a page already mapped for the VM's lifetime.
///
/// # Both poll arms call this
///
/// The locked poll and the lock-free one taken when the device lock is contended
/// used to carry their own copies of the read-modify-write below, and the copies
/// had already diverged once — the lock-free one omitted the enable-mask check
/// entirely. Sharing the body is what keeps a third divergence from being
/// possible: there is one place that decides what a refresh tick writes.
///
/// The claim moved in here for the same reason. It was two lines in each caller,
/// and both copies had the ordering bug above; as one line inside the body, an
/// arm cannot reintroduce it without deleting it from the arm that works.
pub(crate) fn signal_display_refresh_classes<H: HostMemory + HostOps>(
    host: &mut H,
    gpa: u64,
    display_index: u32,
    intr_disp: &std::sync::atomic::AtomicU32,
    page_size: usize,
    last_us: &std::sync::atomic::AtomicU64,
    now_us: u64,
) {
    // The limiter paces in microseconds because 120 Hz is not expressible in
    // whole milliseconds; the census windows in milliseconds so its `t=` stays
    // on the same scale as every other always-on line.
    let now_ms = now_us / 1_000;
    let mut mask_le = [0u8; 4];
    if host
        .read_gpa(gpa + DISPLAY_SHARED_ENABLE_MASK, &mut mask_le)
        .is_err()
    {
        // The guest published this page's address itself; a read of it the host
        // cannot perform is not a reason to start signalling classes nobody
        // asked for. Counted as not-enabled, which is what it is from here.
        note_vbl(VBL_NOT_ENABLED, now_ms);
        return;
    }
    let mask = ld32(&mask_le);
    crate::runtime::drain::census::note_display_enable_mask(mask);

    let vbl = mask & DISPLAY_VBL_EVENT_MASK != 0;
    let transaction = mask & DISPLAY_PRESENT_EVENT_MASK != 0;

    // A tick that finds nothing armed returns **without consuming a grid slot**.
    // This ordering is the whole of the one-shot fix; see the section on it in
    // this function's doc.
    if !vbl && !transaction {
        note_vbl(VBL_NOT_ENABLED, now_ms);
        return;
    }
    if !claim_display_vbl(last_us, now_us) {
        note_vbl(VBL_NOT_CLAIMED, now_ms);
        return;
    }

    // Counted after the limiter so the arm is on the same grid as `delivered`
    // and the two are directly comparable. A guest that arms only the
    // transaction class still reports `not_enabled` here, because this arm names
    // the VBL class and not "did the tick do anything".
    note_vbl(if vbl { VBL_DELIVERED } else { VBL_NOT_ENABLED }, now_ms);
    if transaction {
        note_display_present_signal(DISPLAY_PRESENT_REFRESH);
    }

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
    // signal_display_present_complete). This tick only runs post-ack, so `stale`
    // is 0 on healthy boots (bit 2 clears at ack) and this is a no-op there.
    let stale = pending & DISPLAY_ONLINE_EVENT_MASK != 0;
    let mut next = if stale {
        pending & !DISPLAY_ONLINE_EVENT_MASK
    } else {
        pending
    };
    if vbl {
        next |= DISPLAY_VBL_EVENT_MASK;
    }
    if transaction {
        next |= DISPLAY_PRESENT_EVENT_MASK;
    }
    shared_w32(host, gpa, DISPLAY_SHARED_PENDING, next, page_size);
    if stale {
        crate::runtime::census::present_proxy::note_stale_online_pending("vbl", pending);
    }
    intr_disp.fetch_or(
        1u32 << (display_index & 0x1f),
        std::sync::atomic::Ordering::AcqRel,
    );
    host.enqueue(HostAction::irq_gfx());
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
    let page_size = state.page_size() as usize;
    signal_display_refresh_classes(
        host,
        state.display.shared_gpa,
        state.display.display_index,
        &state.gfx.interrupt_status_disp,
        page_size,
        last_us,
        now_us,
    );
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
    // The divisor is a cadence for **re-asserting** ONLINE, not for looking at
    // whether the guest has enabled the display. It used to gate both, and that
    // cost the first pulse a whole divisor of latency: `poll_ctr` is zeroed by
    // `apply_setup_shared_state`, so the guest could set the enable bit
    // immediately after registering the shared page and this device would not
    // read it until 50 polls later.
    //
    // Measured, on three macos-11/12 boots, as the interval between
    // `display_shared_state_setup` and `display_online_signal` in each boot's own
    // log: 2365 ms, 2324 ms and 2139 ms. Fifty polls at the ~45 ms this is
    // actually called at during early boot — the poll is driven by `gfx_update`
    // and the drain, and neither runs at the 4 ms heartbeat before the guest is
    // drawing anything. So essentially the whole of that interval was this gate.
    //
    // It is a real cost and not a cosmetic one, because the guest is racing us.
    // macOS's WindowServer asks `AppleParavirtFramebuffer` for a mappable VRAM
    // aperture about 13 s into boot and **aborts the process** if it does not get
    // one — `CGXMappedDisplayStart` asserts, and the desktop never appears for
    // the life of the boot. Onlining at 14.2 s loses that race and onlining at
    // 11.9 s wins it, which is the whole margin.
    //
    // So the counter still advances every poll, the enable mask is read every
    // poll, and the divisor now gates only the *repeat* — first pulse on the
    // first poll that observes the enable bit, retries at the archive's ~200 ms.
    //
    // Measured again the same way on five macos-11 boots afterwards, the same
    // interval reads **1, 1, 1, 3 and 4 ms**. The wait is gone rather than
    // shortened, which is what says the divisor was the whole of it and not one
    // term in it.
    let ctr = state.display.poll_ctr.wrapping_add(1);
    state.display.poll_ctr = ctr;
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
    // The guest has enabled. The first pulse goes now; every later one waits for
    // the archive's cadence, which is what the divisor was always for.
    if state.display.online_tries > 0 && !ctr.is_multiple_of(DISPLAY_ONLINE_POLL_DIVISOR) {
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
/// Mask matches archive `poll_tick`: open domains **or** domains holding work,
/// which is `DeviceState::drainable_child_mask`, so work doorbell'd while a
/// drain was in flight is not skipped. Skips
/// `skip_channel` and every bit in `state.draining_mask` so nested drains
/// cannot re-enter a mid-packet channel (same head re-process).
pub fn drain_other_child_fifos<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    skip_channel: u32,
) {
    let mask = state.drainable_child_mask();
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

/// The pending-frames gate fired on one present, named once per episode.
///
/// Keyed on the ordering position rather than on a ring head, because a
/// declined present has no ring position: the head moved past it when it
/// arrived. The episode is the transaction the display rail is behind on, which
/// is what the guest is waiting for.
fn note_present_backpressure_hold(
    state: &mut DeviceState,
    channel: u32,
    position: reims_vgpu_core::identity::IngressOrdinal,
) {
    if state.present.backpressure_hold_active
        && state.present.backpressure_hold_channel == channel
        && state.present.backpressure_hold_position == position.0
    {
        return;
    }
    state.present.backpressure_hold_active = true;
    state.present.backpressure_hold_channel = channel;
    state.present.backpressure_hold_position = position.0;
    state.present.backpressure_hold_count = state.present.backpressure_hold_count.saturating_add(1);
    crate::observe::fail(format!(
        "THRASH present_action_starvation reason=pending_frames_cap ch={channel} \
         position={} unpainted={} episode={}",
        position.0, state.present.unpainted_presents, state.present.backpressure_hold_count
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
        let open = state.open_child_mask();
        if open != 0 {
            state.pending.child_mask |= open;
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
/// **Only `pending.child_mask`.** This used to set an `active_child_mask` bit as
/// well, so that a channel which had only ever rung lock-free would still be
/// seen by the stranded-FIFO rescue — which read that field. It cannot any more,
/// and must not: a doorbell says there is work on a domain, it does not say the
/// guest defined one, and openness is `DeviceState::session`'s with a channel
/// definition as its one event.
///
/// What the rescue reads instead is `DeviceState::open_child_mask`, so a domain
/// that was rung and never defined is no longer re-published by it. That is a
/// real difference and it is answered rather than assumed: across four driven
/// boots this guest put **zero** packets on a domain no definition opened — see
/// [`note_packet_domain_definition`] — so a channel with a stranded ring is a
/// defined channel, and a defined channel is in the mask the rescue reads.
pub(crate) fn fold_rung_child_doorbells(state: &mut DeviceState) {
    let rung = state
        .gfx
        .child_doorbell_rung
        .swap(0, std::sync::atomic::Ordering::AcqRel);
    if rung == 0 {
        return;
    }
    state.pending.child_mask |= rung;
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
    // The successor of `retry_stamp_held_timelines`, which walked channels in
    // id order and re-offered the ones held on a slot a higher-numbered channel
    // publishes. There is no walk to get out of order any more: a released
    // position is run by whoever is settling, whatever channel it belongs to,
    // and the stamp that released it was observed in the same pass.
    settle_model_work(state, host);
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
