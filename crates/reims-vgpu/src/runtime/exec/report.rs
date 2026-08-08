//! How the exec rail reports what it did, refused and dropped.
//!
//! Every function here **decides nothing**. Each one takes a fact the rail has
//! already established and turns it into a census route, a fail line, a typed
//! refusal, or a first-sight latch — and returns either nothing or the latch's
//! own answer. That is the whole membership rule, and it is why these can sit
//! together despite reporting unrelated things: `mod.rs` is then the rail, and
//! this is what the rail says about itself.
//!
//! Some of it is measurement (`note_pass_extent_coverage` bands what a pass
//! covered) and some is lost guest work (`note_clear_dropped`,
//! `note_draw_encode_fail`). Both belong here for the same reason; neither
//! changes what the rail does next.
//!
//! Not to be confused with `crate::observe`, which is the crate-wide emission
//! machinery these call into, or with `runtime/census/`, whose module doc
//! reserves that directory for something else.

use super::{ChainAbandonDecline, StreamAccum, StreamDrawDrop};
use crate::runtime::compute_exec::ComputeStatus;
use crate::runtime::decode::compute::Kind as ComputeKind;
use crate::runtime::decode::render::{AttachSubresource, ScissorRect};
use crate::runtime::draw::EncodeStatus;
use reims_vgpu_wire::ops::render as wire_render;

/// Name a compute refusal at the rail boundary.
///
/// Until this existed the three dispatch/control/ICB arms below only
/// *counted*: `compute_dispatches_fail` went up and nothing said which of the
/// rail's ~150 checks refused, because nine of `ComputeStatus`'s variants were
/// payload-free. The slug now rides in the status, so one line names the check,
/// the pipeline and the record kind.
///
/// Latched per `(reason, pipeline)`: the guest re-submits the same dispatch
/// every frame, so a persistent refusal would otherwise be a per-frame flood —
/// while a *different* pipeline failing the same check is a distinct event and
/// still gets its line.
pub(super) fn note_compute_refusal(
    status: ComputeStatus,
    task_id: u32,
    pipeline_ref: u32,
    kind: ComputeKind,
) {
    // One event token for the whole rail, with `kind=` separating dispatch
    // from control-flow from ICB: the emission gate reads the *literal* first
    // argument, so a per-arm event passed in as a parameter would leave the
    // registry naming a line the gate cannot find.
    if let Some(e) = crate::observe::Emit::refusal("compute_record", &status) {
        e.field("task", task_id)
            .field("pipe", pipeline_ref)
            .field("kind", format!("{kind:?}"))
            .fail_once(u64::from(pipeline_ref));
    }
}

/// Fail-visible, deduped record of a render opcode the decoder accepts but has
/// no executor for (`RenderKind::OtherAccepted`). Fires exactly ONE line per
/// distinct opcode — the undecoded-opcode set is tiny and boot-stable, so this
/// keeps the "guest render command dropped" signal visible on the always-on
/// sink without the per-draw flood a bare emit would produce (a per-draw op
/// like 0x7c fired ~2620 times across six app launches). The line carries the
/// length, bound targets/pipeline, bind counts, and the first-sighting raw wire
/// (hex) so the exact layout can be decoded offline later. Runs on the drain
/// worker (off the QEMU main/vCPU threads). Diagnostic only — it never gates
/// behavior and never invents semantics for the unknown wire.
// Render opcodes are < 256 by contract (observed max 0x98); a dense lock-free
// table gives a zero-alloc, wait-free fast path after warmup. Module-scope so a
// test can reset it deterministically.
pub(super) const UNIMPL_OPCODE_TABLE: usize = 256;
pub(super) static UNIMPL_OPCODE_SEEN: [std::sync::atomic::AtomicBool; UNIMPL_OPCODE_TABLE] =
    [const { std::sync::atomic::AtomicBool::new(false) }; UNIMPL_OPCODE_TABLE];
pub(super) static UNIMPL_OPCODE_OVERFLOW: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashSet<u32>>,
> = std::sync::OnceLock::new();

/// Returns `true` if this call emitted the line (first sighting of `opcode`),
/// `false` if it was deduped. The caller ignores it; tests use it to assert the
/// anti-flood behavior without depending on the shared always-on log file.
pub(super) fn note_unimplemented_render_opcode(
    opcode: u32,
    cmd_bytes: &[u8],
    task_id: u32,
    acc: &StreamAccum,
) -> bool {
    use std::sync::atomic::Ordering;
    if (opcode as usize) < UNIMPL_OPCODE_TABLE {
        // First sighting only: swap false->true; racers that lose stay quiet.
        if UNIMPL_OPCODE_SEEN[opcode as usize].swap(true, Ordering::Relaxed) {
            return false;
        }
    } else {
        // Out-of-range opcode (decode desync / garbage) — dedup through a
        // small overflow set so a runaway value cannot flood either.
        let set = UNIMPL_OPCODE_OVERFLOW.get_or_init(|| std::sync::Mutex::new(Default::default()));
        if let Ok(mut g) = set.lock() {
            if !g.insert(opcode) {
                return false;
            }
        }
    }
    let hex: String = cmd_bytes
        .iter()
        .take(48)
        .map(|b| format!("{:02x}", b))
        .collect::<Vec<_>>()
        .join("");
    crate::observe::fail(format!(
        "render_unimplemented reason=accepted_without_executor task={task_id} opcode={:#x} len={} target_refs={:?} pipeline={} vbufs={} fbufs={} ftex={} hex={}",
        opcode,
        cmd_bytes.len(),
        acc.color_targets,
        acc.pipeline_ref,
        acc.vertex_buffers.len(),
        acc.fragment_buffers.len(),
        acc.fragment_textures.len(),
        hex
    ));
    true
}

/// Serializes the two tests that share the process-global unimplemented-opcode
/// dedup latch, so one test's reset cannot race the other's emissions.
#[cfg(test)]
pub(super) static UNIMPL_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Clear the unimplemented-opcode dedup latch so a test can deterministically
/// observe the first-sighting line regardless of prior in-process emissions.
#[cfg(test)]
pub(super) fn reset_unimplemented_opcode_dedup_for_test() {
    for slot in UNIMPL_OPCODE_SEEN.iter() {
        slot.store(false, std::sync::atomic::Ordering::Relaxed);
    }
    if let Some(set) = UNIMPL_OPCODE_OVERFLOW.get() {
        if let Ok(mut g) = set.lock() {
            g.clear();
        }
    }
}

/// A `setScissorRect:` naming a rect with no area.
///
/// The five `has_*` guards beside this one in the render dispatch are the
/// decoder saying a record did not carry a field, so falling through them costs
/// nothing. This guard was not that: it tested a *decoded* value, and a record
/// that failed it reached the match's `_ => {}` and left `acc.scissor` holding
/// the **previous** rect. So every draw after it clipped to a region the guest
/// had just replaced, which is wrong pixels rather than missing ones — the
/// class that nothing downstream can detect.
///
/// Behaviour is unchanged: an empty rect still does not become the scissor,
/// because this rail has no way to express "clip everything" and adopting a
/// zero rect would make the *next* draw's clip depend on how the backend
/// handles a degenerate one. What changes is that the substitution is now
/// visible. Deduped on the pair, because a guest emitting one emits it for a
/// reason that does not vary per record.
pub(super) fn note_empty_scissor(task_id: u32, rect: ScissorRect) {
    let (w, h) = (rect.width, rect.height);
    crate::runtime::drain::note_store_route("render_scissor_empty_kept_previous");
    if crate::observe::first_sight("render_scissor_empty", (u64::from(w) << 32) | u64::from(h)) {
        crate::observe::fail(format!(
            "render_scissor_empty reason=render_scissor_empty_kept_previous \
             task={task_id} w={w} h={h} \
             (the guest replaced its scissor with an empty rect; this rail kept \
             the previous one, so later draws clip to a region the guest \
             retired)"
        ));
    }
}

/// An `executeCommandsInBuffer:` naming no buffer.
///
/// Unlike a `setTexture:atIndex:` with `ref == 0` — an unbind, which is a
/// meaning — an ICB execute has no argument to unbind. A zero ref is either a
/// record this device decoded short or a stream that named a buffer it never
/// defined, and either way the entire batch of commands that ICB holds is not
/// executed. That is the largest single loss any one render record can carry,
/// and it used to fall through the arm's guard into `_ => {}`.
///
/// The other fields are on the line because they are what says which of the two
/// it is: a record whose range and args buffer are also zero was probably never
/// populated, and one carrying a plausible range with a zero ICB names a
/// resource the stream lost.
pub(super) fn note_unnamed_icb_execute(
    task_id: u32,
    cmd: &crate::runtime::decode::render::Command,
) {
    crate::runtime::drain::note_store_route("render_icb_execute_unnamed");
    if crate::observe::first_sight("render_icb_execute_unnamed", u64::from(task_id)) {
        crate::observe::fail(format!(
            "render_icb_execute reason=render_icb_execute_unnamed task={task_id} \
             is_range={} range_loc={} range_len={} args={} args_off={:#x} \
             (executeCommandsInBuffer named no buffer; the whole batch is lost)",
            cmd.icb_is_range as u8,
            cmd.icb_range_location,
            cmd.icb_range_length,
            cmd.icb_args_buffer_ref,
            cmd.icb_args_buffer_offset
        ));
    }
}

/// The draw opcodes whose records carry an index buffer.
///
/// `render::decode` collapses every draw form to `Kind::Draw`, so the decoded
/// record cannot say which class it came from and the opcode is the only thing
/// that can.
pub(super) fn is_indexed_draw_opcode(opcode: u32) -> bool {
    // wire opcodes via wire_render import

    matches!(
        opcode,
        wire_render::OPCODE_DRAW_INDEXED
            | wire_render::OPCODE_DRAW_INDEXED_INSTANCED
            | wire_render::OPCODE_DRAW_INDEXED_WIDE
    )
}

/// Name an indexed draw whose record carried no index buffer.
///
/// Deduped on the opcode: the three indexed forms read `index_buffer_ref` from
/// three different payload offsets, so which form fires is the whole diagnostic
/// value — one form firing alone points at that form's offset, all three
/// firing points at the guest.
pub(super) fn note_indexed_draw_without_buffer(task_id: u32, opcode: u32, index_count: u32) {
    crate::observe::fail(format!(
        "stream_draw reason=indexed_without_index_buffer task={task_id} op={opcode:#x} \
         index_count={index_count} drawn_as=non_indexed"
    ));
}

/// Name a depth or stencil attachment dropped for a form this device does not
/// implement.
///
/// Deduped on the pair that decides the arm, not on the task: this fires from a
/// per-`RenderPass` decode, so a guest that uses mip-1 depth throughout would
/// otherwise emit on every pass in every stream. One line per distinct
/// (aspect, level, resolve-present) combination is what answers the question
/// the arm exists to answer — whether any guest asks for this at all.
/// Returns the drop it reported, so the caller can refuse the stream's draws
/// with it rather than rebuilding the same arm from the same fields.
pub(super) fn note_depth_stencil_unsupported(
    task_id: u32,
    aspect: &'static str,
    s: &AttachSubresource,
) -> StreamDrawDrop {
    let drop = StreamDrawDrop::DepthStencilUnsupported {
        aspect,
        level: s.level,
        slice: s.slice,
        depth_plane: s.depth_plane,
        resolve_texture_ref: s.resolve_texture_ref,
    };
    crate::observe::Emit::decline("stream_pass", &drop)
        .field("task", task_id)
        .fail_once(drop.latch());
    drop
}

/// Bands for the stated pass extent as a fraction of its attachment's area.
///
/// Same seven bands as the scissor-union census in `draw::vulkan`, so the
/// two are readable side by side — they answer the same question from two
/// different sources, and the whole point is which of the two carries damage the
/// other does not.
pub(super) const PASS_EXTENT_SLUGS: [&str; 7] = [
    "pass_extent_lt1",
    "pass_extent_le5",
    "pass_extent_le10",
    "pass_extent_le25",
    "pass_extent_le50",
    "pass_extent_le99",
    "pass_extent_full",
];

/// Score slot 0's attachment, whichever resolve arm found it.
///
/// Both arms of the colour-attachment resolve consume the same wire form and
/// must be scored the same way; only the mapping id differs, because a type-11
/// attachment resolves *to* an id and a type-4 attachment *is* one. This existed
/// on the type-11 arm alone, and the consequence was not that the census
/// undercounted — it was that the census read **zero** on the whole x86/Vulkan
/// pathway, where the workload takes the type-4 arm. A pathway-shaped blind spot
/// reads exactly like "the guest never states an extent", which is the opposite
/// of what it does.
///
/// Slot 0 only, and only where the mapping is already resolved: this is a
/// census, not a resolve, and making it resolve would put a guest-memory walk on
/// the hottest record in the device.
pub(super) fn note_pass_extent_for_slot(
    state: &crate::model::DeviceState,
    slot: u32,
    mapping_id: u32,
    cmd: &crate::runtime::decode::render::Command,
) {
    if slot != 0 {
        return;
    }
    if let Some(e) = state.mappings.get(&mapping_id) {
        note_pass_extent_coverage(
            cmd.pass_render_target_width,
            cmd.pass_render_target_height,
            e.width,
            e.height,
        );
    }
}

/// Score the guest's stated pass extent against the attachment it names.
///
/// This is the number the flush rail has been missing. Bounding a writeback by
/// the *draw stream's* scissors was measured to save nothing — 99.92 % of armed
/// windows have a per-pass scissor union of 100 % — and the conclusion drawn
/// then was that a damage-bounded flush needs a different source of damage than
/// the draw stream, and that none is currently decoded. `renderTargetWidth` and
/// `renderTargetHeight` are decoded now, and a driven boot shows the window
/// server naming extents like 170x12 and 32x32 rather than the display's
/// 1920x1080.
///
/// What this cannot say by itself is whether those small extents sit on small
/// attachments. That is exactly what the bands measure: a distribution weighted
/// at `full` means the extent is the surface and there is nothing to bound, and
/// one with mass below `le50` is a writeback that could be halved.
///
/// # The answer is `full` for all but a handful, which is not the same as all
///
/// Driven arm64/Vulkan boot, 60 s Safari drag, once both resolve arms scored:
///
/// ```text
/// pass_extent_full 11826      pass_extent_le5 1      every other band 0
/// ```
///
/// Driven x86/Vulkan boot, 25 s Safari drag:
///
/// ```text
/// pass_extent_full 10537      pass_extent_le5 2      pass_extent_le50 1
/// ```
///
/// **The extent is the attachment, 99.97 % of the time.** The small numbers in
/// the log — 242x5, 1920x24, the many under 110 px — are small *surfaces*, not
/// sub-rects of large ones. The window server states `renderTargetWidth/Height`
/// on nearly every pass and states them equal to the target it is rendering
/// into.
///
/// One conclusion and one open lead:
///
/// - **There is nothing to bound.** The pass extent is not a second source of
///   damage; it is the attachment's own geometry restated on the wire. That
///   earlier conclusion — a damage-bounded flush needs a different source of
///   damage than the draw stream, and none is currently decoded —
///   survives having this one decoded and measured. Three passes in a boot
///   cannot found a damage rail. Do not re-open damage bounding on the strength
///   of the raw extent values in the fail log — they look like damage and are
///   not.
/// - **The handful is a real sub-rect, and this device renders past it.** The
///   x86 reading above is what this doc previously recorded as "every scored
///   pass on x86/Vulkan is full"; it is not, and the bands are what said so.
///   A pass stating a 50 % or 5 % extent with `loadAction = Clear` has its whole
///   attachment cleared here, which destroys content outside the rect the guest
///   named — the class two arms of this same decode now refuse rather than
///   commit. It is **not** fixed by mapping the extent onto Vulkan's
///   `renderArea`: `loadOp` applies inside that area, but the spec leaves the
///   contents outside it undefined rather than preserved, so the mapping that
///   looks equivalent is not one. Establishing what Metal guarantees outside a
///   constrained render target comes before any change here.
///
/// The bands stay, and they are now a live reading rather than a healthy zero.
///
/// A pass that states no extent at all is not scored — there is no fraction to
/// take — and neither is one whose attachment has no geometry yet.
pub(super) fn note_pass_extent_coverage(pass_w: u64, pass_h: u64, surf_w: u32, surf_h: u32) {
    if pass_w == 0 || pass_h == 0 || surf_w == 0 || surf_h == 0 {
        return;
    }
    let full = u64::from(surf_w).saturating_mul(u64::from(surf_h));
    let stated = pass_w.saturating_mul(pass_h);
    // Clamped for the reason the scissor union is: a guest may state an extent
    // larger than the attachment and the rasteriser clips, so an unclamped
    // ratio would read over 100 % and make the census unreadable.
    let pct = stated.min(full).saturating_mul(100) / full.max(1);
    crate::runtime::drain::note_store_route(PASS_EXTENT_SLUGS[pass_extent_band(pct)]);
}

/// The bands, matching `draw::vulkan::coverage_band` exactly.
///
/// Declared here rather than shared because that one is behind
/// `backend-vulkan` and this census runs on every backend; the two are pinned
/// equal by `the_two_coverage_censuses_use_the_same_bands`.
pub(super) fn pass_extent_band(pct: u64) -> usize {
    match pct {
        0 => 0,
        1..=5 => 1,
        6..=10 => 2,
        11..=25 => 3,
        26..=50 => 4,
        51..=99 => 5,
        _ => 6,
    }
}

/// Count a pass that stated an explicit render target extent, which this device
/// does not apply.
///
/// This is the **denominator** for [`note_pass_extent_coverage`]'s bands and
/// nothing more. It used to also put the extent's raw values on the fail
/// channel, so that a reader who had the surface geometry beside them could
/// decide whether ignoring the extent lost anything. That reader is now
/// `note_pass_extent_coverage`, it has the geometry, and it has answered:
/// `pass_extent_full` takes 11 826 of 11 827 scored passes on arm64/Vulkan and
/// 10 537 of 10 540 on x86/Vulkan. The extent is the attachment restated, for
/// all but a handful of passes a boot — see that function for what the handful
/// costs and why it is not fixed by the obvious mapping.
///
/// The line is gone because it was reporting a non-loss on the channel reserved
/// for lost guest work, and doing so at 85 % of that channel's whole volume —
/// while its own text told the reader to disregard the numbers it carried. Its
/// raw values (242x5, 1920x24) read like damage rects and are small surfaces,
/// so the line's net effect on a reader ranking `reason=` was to mislead.
///
/// Keep the gap between this count and the bands' sum in view: a pass counted
/// here and not scored there is one whose attachment had no geometry yet, and
/// the two numbers are only comparable when that gap is understood.
pub(super) fn note_pass_target_extent() {
    crate::runtime::drain::note_store_route("render_pass_target_extent_unapplied");
}

/// A pass declaring more render-target array layers than this device draws.
/// See [`StreamDrawDrop::PassArrayLengthUnsupported`].
///
/// The census route keeps the name it had while this was a bare count, so a
/// boot series spanning the change stays comparable: what moved is that the
/// pass is now refused, and the route's verdict in the counted-loss census
/// moved with it.
///
/// Deduped on the declared layer count, because that is what varies between
/// two guests asking for this and it is the whole of what the arm reports.
pub(super) fn note_pass_array_length_unsupported(task_id: u32, length: u64) -> StreamDrawDrop {
    crate::runtime::drain::note_store_route("render_pass_array_length_dropped");
    let drop = StreamDrawDrop::PassArrayLengthUnsupported { length };
    crate::observe::Emit::decline("stream_pass", &drop)
        .field("task", task_id)
        .fail_once(drop.latch());
    drop
}

/// A pass declaring a default raster sample count this device cannot rasterize
/// at. See [`StreamDrawDrop::PassRasterSampleCountUnsupported`].
///
/// The census route keeps the name it had while this was a bare count, on the
/// same reading [`note_pass_array_length_unsupported`] states: a boot series
/// spanning the change stays comparable, and what moved is the verdict rather
/// than the population.
///
/// Deduped on the requested count, because that is what varies between two
/// guests asking for this.
pub(super) fn note_pass_raster_sample_count_unsupported(
    task_id: u32,
    count: u64,
) -> StreamDrawDrop {
    crate::runtime::drain::note_store_route("render_pass_raster_sample_count_dropped");
    let drop = StreamDrawDrop::PassRasterSampleCountUnsupported { count };
    crate::observe::Emit::decline("stream_pass", &drop)
        .field("task", task_id)
        .fail_once(drop.latch());
    drop
}

/// A colour attachment naming a mip, a slice, a depth plane or a multisample
/// resolve target this device renders past.
/// See [`StreamDrawDrop::ColorSubresourceUnsupported`].
///
/// Deduped on the shape and the slot rather than on the texture, because the
/// question is which *shape* of subresource a guest asks for, not how many
/// textures it asks for it on.
/// Returns the drop it reported, so the caller can refuse the stream's draws
/// with it rather than rebuilding the same arm from the same fields.
pub(super) fn note_color_subresource_unsupported(
    task_id: u32,
    slot: u32,
    att: &crate::runtime::decode::render::ColorAttachment,
) -> StreamDrawDrop {
    crate::runtime::drain::note_store_route("render_color_subresource_unsupported");
    let drop = StreamDrawDrop::ColorSubresourceUnsupported {
        slot,
        level: att.level,
        slice: att.slice,
        depth_plane: att.depth_plane,
        resolve_texture_ref: att.resolve_texture_ref,
    };
    crate::observe::Emit::decline("stream_pass", &drop)
        .field("task", task_id)
        .field("texture", att.texture_ref)
        .fail_once(drop.latch());
    drop
}

/// Report what this stream's draw list cost, and anything it lost building it.
///
/// The distribution stays after the cap is gone, because it is now the only
/// thing that prices the decision to keep every record: it says how long a real
/// render stream is, and therefore what an unbounded list actually costs. The
/// boot that removed the cap read 118 307 streams as 39 913 single-draw, 55 306
/// at 2–4, 14 579 at 9–16 and 8013 at 33–63, with two above 64 — a tail that
/// exists and a body that does not.
///
/// Buckets rather than a mean because the question is about that tail: one
/// 400-draw compositor stream among thousands of 2-draw ones is exactly the case
/// that matters and is exactly what a mean hides. The two buckets above the old
/// ceiling are what say whether removing it changed which streams complete.
pub(super) fn note_stream_draw_drops(task_id: u32, acc: &StreamAccum) {
    let kept = acc.draws.len();
    if kept == 0 && acc.dropped_unbound == 0 {
        return;
    }
    crate::runtime::drain::note_store_route(match kept {
        0 => "stream_draws_0",
        1 => "stream_draws_1",
        2..=4 => "stream_draws_2_4",
        5..=8 => "stream_draws_5_8",
        9..=16 => "stream_draws_9_16",
        17..=32 => "stream_draws_17_32",
        33..=63 => "stream_draws_33_63",
        64..=255 => "stream_draws_64_255",
        _ => "stream_draws_over_255",
    });
    // Latched on the *magnitude* of the loss, not on the task: the same stream
    // shape recurs every frame, so a per-task key would print once and hide a
    // loss that grew, while a bucket key prints again when it gets worse.
    if acc.dropped_unbound > 0 {
        let d = StreamDrawDrop::Unbound {
            dropped: acc.dropped_unbound,
        };
        if crate::observe::first_sight(
            crate::observe::Decline::slug(&d),
            u64::from(acc.dropped_unbound.next_power_of_two()),
        ) {
            crate::observe::Emit::decline("stream_draw", &d)
                .field("task", task_id)
                .field("kept", kept)
                .fail();
        }
    }
}
crate::observe::decline_display!(ChainAbandonDecline);

/// One-shot (per `pipeline_ref` x reason) always-on line for a failed draw
/// encode. `exec_indirect2 draws_fail=N` collapses every cause into one
/// counter with no reason; a persistently failing draw (e.g. an app window
/// layer that never paints) was invisible on a normal boot. The latch keys
/// on the pipeline so a new failing workload logs its own line while a
/// steady repeat (same pipeline failing every packet) stays at one line.
///
/// The `reason=` was the *variant* name until `EncodeStatus` carried its check:
/// six names for the rail's 27 refusals, so `reason=bad_args` could be a
/// zero-size target, a vertexless draw or an unresolvable MRT slot. Now the
/// variant prints as `class=` beside the check that produced it.
pub(super) fn note_draw_encode_fail(
    task_id: u32,
    pipeline_ref: u32,
    status: EncodeStatus,
    di: usize,
    n: usize,
) {
    if let Some(e) = crate::observe::Emit::refusal("draw_encode_fail", &status) {
        e.field("pipe", pipeline_ref)
            .field("task", task_id)
            .field("di", format!("{di}/{n}"))
            .fail_once(pipeline_ref as u64);
    }
}

/// Deduped, fail-visible record of a guest clear directive we did not honor.
/// Keyed by `(reason, texture_ref)` so a persistent condition logs exactly once
/// instead of per stream — no flood. Runs on the drain worker (off the QEMU
/// main core) via the always-on `observe::fail` sink. Returns `true` the first
/// time a given `(reason, tex_ref)` is seen (the call that emitted the line).
pub(super) fn note_clear_dropped(reason: &'static str, tex_ref: u32, detail: &str) -> bool {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(&'static str, u32)>>> = Mutex::new(None);
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let first = seen
        .get_or_insert_with(HashSet::new)
        .insert((reason, tex_ref));
    if first {
        crate::observe::fail(format!(
            "clear_dropped reason={reason} tex_ref={tex_ref} {detail}"
        ));
    }
    first
}

/// Name an indirect draw whose argument buffer this device could not read.
///
/// The buffer *is* the draw: unlike a direct record there is no count in the
/// wire to fall back to, so a failed read is geometry the guest asked for and
/// did not get. `ComputeStatus`'s slug already names which rung of the buffer
/// resolve refused — the ref was unbound, named nothing, named some other
/// object, or the window ran past the buffer's end — so this adds the record's
/// own fields and nothing else.
///
/// Latched per buffer ref, not per task: a guest re-issues the same indirect
/// draw every frame, and one line per distinct buffer is what says whether any
/// guest asks for this at all. The latch guards only the line — the draw is
/// refused every time. A latch that guarded the refusal as well as the line
/// would turn the second draw into a silent accept, which is the failure to
/// watch for whenever a dedup latch sits next to a decision.
pub(super) fn note_indirect_draw_refused(
    task_id: u32,
    cmd: &crate::runtime::decode::render::Command,
    status: crate::runtime::compute_exec::ComputeStatus,
) {
    if let Some(e) = crate::observe::Emit::refusal("render_draw_indirect", &status) {
        e.field("task", task_id)
            .field("op", format!("{:#x}", cmd.opcode))
            .field("args_ref", cmd.indirect_buffer_ref)
            .field("args_off", cmd.indirect_buffer_offset)
            .fail_once(u64::from(cmd.indirect_buffer_ref));
    }
}

/// A depth or stencil store-action override arriving for an attachment the
/// render pass never declared.
///
/// The override has nothing to override. Naming it rather than returning
/// quietly, because the two ways to get here are very different: a guest that
/// sets a depth store action on a colour-only pass is doing something odd, and a
/// pass whose depth attachment this device failed to decode is a bug here — and
/// only the second loses the guest a depth buffer it expected back.
pub(super) fn note_store_action_no_attachment(which: &'static str, action: u16) {
    crate::runtime::drain::note_store_route(match which {
        "depth" => "render_store_action_no_depth_attachment",
        _ => "render_store_action_no_stencil_attachment",
    });
    crate::observe::fail(format!(
        "render_store_action fail reason=render_store_action_no_{which}_attachment \
         action={action}"
    ));
}
