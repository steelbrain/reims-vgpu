//! Scanout paint: guest surface (mapping or EFI FB) → host BGRA8 row buffer.
//!
//! C owns the QEMU DisplaySurface; Rust fills it (apple-gfx's
//! encodeCurrentFrame / getBytes role). Page-table paint uses
//! [`reims_vgpu_protocol::iosurface`] when the mapping has entries;
//! otherwise we fall back to the programmed EFI framebuffer or clear.
//!
//! Early-boot / present policy (archive apple-pv-gpu + live Monterey + PGDisplay):
//! - Front formats: archive prefers IOSurface texture **RGBA16Float** (0x73); live Monterey
//!   boot logo/progress also stores full-screen IOSurface texture **BGRA8** / **RGBA8**
//!   before the first DisplaySwap — paint those formats too pre-boundary.
//! - Geometry barrier (archive same_geom): first early paint establishes console
//!   size from the **guest surface** (mapper geom / job size); later pre-boundary
//!   paints only when the job matches that size. Never invent or clamp dimensions
//!   (Apple `modeChangeHandler` sizeInPixels = presented surface size).
//! - After DisplaySwap (`frame_flush_seen`): writebacks do **not** rename the
//!   presented surface (`present_mapping` stays the last CmdDisplaySwap mid).
//!   Paint is present-boundary only (PGDisplay newFrame / hostPresentCount).
//! - At CmdDisplaySwap the host **retains** the named mapping after wait_surface
//!   drains (`capture_present_frame` = PGDisplay presentFrame → +0x188), then
//!   HostAction paint blits that snapshot. Later scanout / gfx_update re-shows
//!   it (`hostPresentCount`). Freeze is at present (before stamp completion
//!   lets the guest recycle the mid), not deferred to BH after stamp.

use crate::model::{scanout_extent_ok, EFI_BOOT_HEIGHT, EFI_BOOT_WIDTH};
use crate::runtime::host::HostMemory;
use crate::runtime::Device;
use reims_vgpu_core::pixel_format::{
    self, convert_rgba8_to_row, convert_row_to_rgba8, MTL_FORMAT_BGRA8_UNORM,
    MTL_FORMAT_RGBA16_FLOAT, MTL_FORMAT_RGBA8_UNORM, RGBA8_BPP,
};

/// IOSurface texture color formats that may be the compositor front before DisplaySwap.
///
/// Archive `front_buffer` is RGBA16Float only; live Monterey also draws the
/// early boot logo into BGRA8/RGBA8 IOSurface texture full-frame targets. Not a size list.
#[inline]
fn is_front_buffer_format(fmt: u16) -> bool {
    matches!(
        fmt,
        MTL_FORMAT_RGBA16_FLOAT | MTL_FORMAT_BGRA8_UNORM | MTL_FORMAT_RGBA8_UNORM
    )
}

/// Result of a scanout copy attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanoutCopyResult {
    /// Pixels written (or black clear).
    Painted,
    /// Content generation matches last paint — C should skip surface update.
    Unchanged,
    /// Hard failure (bad args).
    Failed,
}

/// Read mapping pages into `dst` without updating present/paint generation.
///
/// Used by draw bind materialization (sampled IOSurface textures). Returns true
/// when geometry and page table produced a full image.
///
/// Backend-agnostic on purpose: it resolves and scatters guest pages and
/// for the same reason the Vulkan arm does.
pub fn read_mapping_bgra8<M: HostMemory + crate::runtime::host::HostOps>(
    state: &mut Device,
    host: &mut M,
    mapping_id: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) -> bool {
    if !scanout_extent_ok(width, height) || dst_stride < width.saturating_mul(RGBA8_BPP) {
        return false;
    }
    let need = (height as u64).saturating_mul(dst_stride as u64) as usize;
    if dst.len() < need {
        return false;
    }
    let _ = crate::runtime::mapper::ensure_resolved_for_scanout(state, host, mapping_id);
    paint_mapping(
        state,
        host,
        mapping_id,
        PaintDst {
            bytes: dst,
            stride: dst_stride,
            width,
            height,
        },
        crate::runtime::render_writeback::SettleSite::SampledMappingRead,
    )
}

/// Always-on census of the capture readback-elision ratio (never silent):
/// `full` = readback + proxy scan ran; `light` = the window is carrying the frame
/// from the engine resident, readback skipped.
///
/// Emitted at power-of-two capture counts rather than at a fixed interval. The
/// interval this used to carry was one line per 1024 captures, sized as "a line
/// every ~8 s at 120 Hz" — but a capture is one accepted DisplaySwap, not one
/// vblank, and a driven boot records 63 of them across 423 s. At that rate the
/// first line landed somewhere past the half-hour mark, so a census documented as
/// always-on emitted nothing on any boot anyone has run, and a regression in the
/// elision it measures would have gone unread.
///
/// Power-of-two spacing fixes both ends at once: the ratio is readable from the
/// first capture onward, and the line count is bounded by log2 of the capture
/// count — about twenty lines per boot even if the rate rises by orders of
/// magnitude, so there is no interval to re-tune when it does.
fn maybe_log_capture_sampling(state: &Device) {
    let (full, light) = state.presentation.present.capture_counts();
    let total = full.wrapping_add(light);
    if total != 0 && total.is_power_of_two() {
        crate::observe::off(format!("capture_sampling full={full} light={light}"));
    }
}

/// Fill `buf` from the mapping's GPU resident, without any guest-page scatter.
///
/// Returns whether the resident supplied the whole frame. On `true` `buf` holds
/// tight BGRA8; on `false` `buf` is untouched and the capture fails
/// (keep-prior) — there is no guest-page path left for the caller to take. A miss
/// is an expected steady-state condition (cold mid / no resident yet), so it is
/// counted in the `capture_source` census rather than logged per present.
fn try_capture_from_resident(
    state: &mut crate::runtime::Device,
    buf: &mut Vec<u8>,
    mapping_id: u32,
    width: u32,
    height: u32,
) -> bool {
    let need = buf.len();
    let identity =
        crate::runtime::present_identity::surface_identity(state, mapping_id, width, height);
    let Some(bgra) = state.executor.read_resident_bgra(&identity, need) else {
        return false;
    };
    debug_assert_eq!(bgra.len(), need);
    // Move (not copy) the readback in. Publishing the completed capture recycles
    // the prior retained frame as the next scratch buffer.
    *buf = bgra;
    true
}

/// Snapshot the named mapping into the stable present frame.
///
/// PGDisplay retains the presented surface at present (`+0x188`); encode /
/// re-show use that retain. Product freezes the finished surface at
/// CmdDisplaySwap after wait_surface drains — before the packet stamp lets the
/// guest recycle the mid (BH-deferred freeze captured mid-recycle partials).
///
/// Two sources fill the frame, and neither is guest memory: the host surface
/// cache when an encode or clear wrote it, otherwise the GPU resident. That is
/// why this takes no `HostOps`. A capture that can use neither fails visibly and
/// keeps the prior retain rather than opening a third vein — see the note at the
/// resident read for why the guest-page fallback was deleted instead of kept.
pub fn capture_present_frame(
    state: &mut Device,
    mapping_id: u32,
    width: u32,
    height: u32,
    generation: u32,
) -> bool {
    // Test isolation: exclude proxy-sequence assertions running in parallel.
    #[cfg(test)]
    let _proxy_shared = crate::runtime::census::present_proxy::test_shared();
    if mapping_id == 0 || !scanout_extent_ok(width, height) {
        return false;
    }
    let stride = width.saturating_mul(RGBA8_BPP);
    let need = (height as u64).saturating_mul(stride as u64) as usize;
    if need == 0 {
        return false;
    }
    // "These are different pixels", which `generation` cannot say for a lazy
    // IOSurface texture Store: it leaves the frame in the engine resident and writes no
    // guest page, so `content_generation` holds still while the pixels move.
    // Read from the entry here rather than threaded in, because the caller
    // resolved `generation` from that same entry and a second parameter is a
    // second chance for the two to name different mappings.
    let content_epoch = state
        .surfaces
        .mappings
        .get(&mapping_id)
        .map(|m| m.content.surface_epoch)
        .unwrap_or(0);
    state.advance_present_epoch();
    // --- Capture readback elision ---
    // When preparation selected an engine resident for this present, the
    // display reads that resident directly and does not consume a CPU capture.
    // The ~8-12 ms guest-page gather + full-frame proxy scan below is then pure
    // present-hot-path overhead that serializes the guest behind the drain lock
    // (the fullscreen-video slowdown class). Skip it on those presents; the cheap
    // protocol-structural a/b guard still runs on every light present.
    //
    // This is the same handoff on every host-window pathway: the engine's own
    // presenter consumes the resident when it can present to the surface. Only
    // a present with no prepared resident carrier falls back to the readback
    // below.
    //
    // The full-frame readback has EXACTLY ONE reason to exist: the DISPLAY needs
    // CPU pixels because no resident is carrying the frame, and the window will
    // blit `frame_bgra`. Nothing else reads it.
    //
    // Consequence: with a resident carrying, `frame_bgra` holds no frame for this
    // present, and the branch below drops it so that stays literally true.
    let display_needs_cpu_frame = !state
        .presentation
        .present
        .current_present_resident_carried();
    if !display_needs_cpu_frame {
        // Publish the new resident and leave `frame_bgra` empty: the window
        // ignores CPU pixels while the resident carries the display. A publish
        // miss costs one dropped frame (the window holds its last good frame and
        // publish logs the drop). The next present prepares its own route afresh.
        //
        // Dropping it makes "no CPU pixels for this present" explicit to both
        // the content verdict and the console blit; neither may consume bytes
        // retained from an earlier CPU-backed present.
        state.presentation.present.publish_light_frame(
            mapping_id,
            width,
            height,
            generation,
            content_epoch,
        );
        state.presentation.present.note_light_capture();
        maybe_log_capture_sampling(state);
        return true;
    }
    state.presentation.present.note_full_capture();
    maybe_log_capture_sampling(state);
    // Attribute this capture's lock hold to the tranche `capture_us` bucket (it
    // runs on the present drain, not a render draw). Every real return below
    // notes the elapsed time so a capture-bound hitch stops hiding in `other_us`.
    // Recycle the warm double-buffered scratch instead of a fresh `vec![0u8;
    // need]` per present (which zeroes 8 MiB and faults fresh anon pages every
    // time, only to overwrite them). `resize` is a no-op at steady geometry;
    // every byte in `[0, need)` is fully written below (host_cache
    // `copy_from_slice`, `paint_mapping` row fill, or the reuse-store copy), so
    // no pre-zero is needed. On failure `buf` returns to `capture_scratch`
    // unchanged, leaving the prior `frame_bgra` retain intact (keep-prior).
    let mut buf = state.presentation.present.take_capture_scratch();
    buf.clear();
    buf.resize(need, 0);
    // Prefer host render-cache when encode/clear wrote it (Linux discrete GPU
    // path — kb tahoe-x86-host-reims_vgpu §8.5); otherwise the resident below.
    // There is no guest-page fallback any more — see the note at the resident
    // capture for why it was deleted rather than kept as a second vein.
    let from_host_cache = if let Some(cached) =
        crate::runtime::surface_cache::get(state, mapping_id, width, height)
    {
        buf.copy_from_slice(cached);
        true
    } else {
        false
    };
    // Resident-direct capture — the ONLY GPU-content capture source.
    //
    // The proxies need the finished frame's BYTES; they do not need those bytes
    // to be in guest pages. This reads the resident and nothing else: no
    // `flush_intersecting`. Nothing is owed — an IOSurface texture render Store lands its
    // own guest-page writeback (`mapping_write::write_rgba8_image_changed`), and
    // the deferred rails that remain (compute storage, linear, GVA) are keyed on
    // resources this capture does not touch and flush on a genuine guest read
    // (LOAD re-seed / SynchronizeResources / guest CPU read). The retained
    // `frame_bgra` filled here is unchanged, so the present-boundary seed (which
    // reads the retained front frame first, guest pages only as fallback) is
    // unaffected.
    //
    // There is deliberately NO guest-page capture fallback. A capture that
    // predated this read the same resident and then scattered it into the
    // fragmented guest pages purely to read it back out — a second, parallel
    // implementation of "get the present frame" that cost a full-frame writeback
    // per sampled present. Keeping it would mean maintaining two veins of the
    // same operation, so a missing resident now fails VISIBLY (keep_prior + the
    // `capture_fail` proxy) instead of silently diverging onto another path.
    // Live evidence for the delete: `capture_source resident=51 guest=0` across a
    // full boot (pre-convergence included), zero `present_capture FAIL`.
    if !from_host_cache && !try_capture_from_resident(state, &mut buf, mapping_id, width, height) {
        crate::observe::off(format!(
            "present_capture FAIL mid={mapping_id} {width}x{height} gen={generation} \
             reason=no_resident_content present_mapping={} frame_mapping={}",
            state.presentation.present.presented_mapping(),
            state.presentation.present.frame().mapping()
        ));
        // Recycle the untouched scratch; the prior retain stays intact.
        state.presentation.present.return_capture_scratch(buf);
        return false;
    }
    // Capture provenance, and there are only two sources to name: the surface backing
    // surface_cache hit, or the resident. Reaching here with `!from_host_cache`
    // means `try_capture_from_resident` returned true above, and it returns true
    // and there is no third source. This used to read a `last_paint_src`
    // provenance field through a five-arm match whose other four arms named
    // `paint_mapping` sub-paths — left over from when this function had a
    // guest-page capture fallback. It no longer calls `paint_mapping` at all, so
    // those arms had become a way for one call path's state to be reported as
    // another's provenance. The field is gone with them.
    let src = if from_host_cache {
        "host_cache"
    } else {
        "resident"
    };
    // The occupancy scan is diagnostic: an O(w*h) walk of the just-captured
    // 8 MiB frame, on the present drain and under the device lock. The
    // always-on alarm for a black console is `present_black`, which does its
    // own scan at the drain boundary where the verdict is acted on.
    //
    // A `peers` field used to ride along here, walking every same-geometry host
    // surface — another O(w*h) each — and admitting one when its non-zero pixel
    // count passed 10 000. That number had no derivation, and "which peer looks
    // like it has real content" is a rule about observed content rather than
    // about the contract. It is gone rather than re-tuned: a peer below the
    // threshold was invisible, so the field could not answer the question it
    // looked like it was answering.
    if crate::observe::draw_log_enabled() {
        let (nz, maxb, rgb_nz, max_rgb, px0) = crate::observe::bgra_present_stats(&buf);
        crate::observe::line(format!(
            "present_capture mid={mapping_id} {width}x{height} gen={generation} src={src} host_cache={} rgb_nz={rgb_nz} max_rgb={max_rgb} byte_nz={nz} byte_max={maxb} px0=[{},{},{},{}] present_mapping={} frame_mapping={} frame_flush={}",
            from_host_cache as u8,
            px0[0],
            px0[1],
            px0[2],
            px0[3],
            state.presentation.present.presented_mapping(),
            state.presentation.present.frame().mapping(),
            state.presentation.present.content_boundary_crossed() as u8,
        ));
    }
    // Publish the new frame and recycle the old retain buffer as the next
    // capture scratch (warm 8 MiB alloc, no per-present malloc/free/zero).
    state.presentation.present.publish_captured_frame(
        buf,
        mapping_id,
        width,
        height,
        generation,
        content_epoch,
    );
    // Force the next host paint to blit +0x188. Early pre-boundary paints may
    // have latched painted_mapping/generation (live IOSurface texture paint_mapping or
    // paint_efi_console) to the same mid+gen; with encode_pending=false that made
    // copy_to_bgra8 return Unchanged and left the QEMU console on frozen EFI
    // while +0x188 held logo+pill (live serial-20260715-054015:
    // present_capture rgb_nz≈6k then present_paint Unchanged only).
    true
}

/// Blit tight BGRA8 `src` into `dst` (tight or strided).
///
/// # A destination row shorter than the frame's is a refusal, not a `min`
///
/// The copy length used to be `src_stride.min(dst_stride)`, which for a
/// destination whose row is shorter than `width * 4` wrote the leading columns
/// of every row and left the rest of each row untouched — a black band down the
/// right of the frame, uniform on every row, with nothing on any channel saying
/// so. It is the only silent column-truncating copy this crate had.
///
/// Every other stride consumer here already refuses a short row by name
/// (`CaptureDecline::BprBelowTight`, and the two `dst_stride < width * 4`
/// guards on the neighbouring entry points), so this was the one place where a
/// destination that cannot hold the frame produced a partial frame instead of a
/// decline. `false` is what the callers already handle: it is the same answer a
/// short `src` gives two lines up.
///
/// This is not the cause of any reported black band — this path fills QEMU's own
/// `DisplaySurface`, which under a host-owned window is not what the screen
/// shows. It is the shape one would have to rule out first, and now the log
/// rules it out instead of a reader having to.
fn blit_bgra_buffer(src: &[u8], dst: &mut [u8], dst_stride: u32, width: u32, height: u32) -> bool {
    let src_stride = width.saturating_mul(RGBA8_BPP) as usize;
    if src.len() < src_stride.saturating_mul(height as usize) {
        return false;
    }
    if (dst_stride as usize) < src_stride {
        crate::observe::fail(format!(
            "present_paint reason=dst_stride_below_tight \
             dst_stride={dst_stride} tight={src_stride} geom={width}x{height}"
        ));
        return false;
    }
    for y in 0..height as usize {
        let so = y * src_stride;
        let doff = y * (dst_stride as usize);
        dst[doff..doff + src_stride].copy_from_slice(&src[so..so + src_stride]);
    }
    true
}

/// Blit the stable present snapshot into `dst` (tight or strided BGRA8).
///
/// presentFrame freezes +0x188 at swap time; post-stamp guest writes must not
/// change the retain (archive encodeCurrentFrame / hostPresentCount re-show).
/// Mid-writeback is_front Stores must **not** recapture +0x188 (archive:
/// post-boundary front writebacks do not paint — tile-through thrash).
fn blit_present_snapshot(
    state: &Device,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) -> bool {
    blit_bgra_buffer(
        state.presentation.present.frame().pixels(),
        dst,
        dst_stride,
        width,
        height,
    )
}

/// Fill `dst` (BGRA8, `dst_stride` bytes/row) for the named mapping.
///
/// `expected_generation` is from the HostAction (0 = always paint).
/// After DisplaySwap, the first copy **encodes** the stable snapshot from live
/// pages (host paint time); later copies re-show that snapshot
/// (`hostPresentCount`) without re-reading guest pages.
///
/// This is now the only console paint. `copy_to_host_ptr_gpu` used to get first
/// refusal on a QEMU-allocated, alignment-negotiated display buffer: the engine
/// imported it and recorded a resident→buffer GPU copy, so no framebuffer bytes
/// crossed the CPU. It went out with the host-pointer import that made it
/// possible — the mechanism is the same one that can address guest RAM, and it
/// is not requested any more, whichever allocation is on the other end.
#[allow(
    clippy::too_many_arguments,
    reason = "the scanout copy API mirrors its destination and present geometry"
)]
pub fn copy_to_bgra8<M: HostMemory + crate::runtime::host::HostOps>(
    state: &mut Device,
    host: &mut M,
    mapping_id: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    expected_generation: u32,
) -> ScanoutCopyResult {
    if !scanout_extent_ok(width, height) || dst_stride < width.saturating_mul(RGBA8_BPP) {
        return ScanoutCopyResult::Failed;
    }
    let need = (height as u64).saturating_mul(dst_stride as u64) as usize;
    if dst.len() < need {
        return ScanoutCopyResult::Failed;
    }
    // PGDisplay encodeCurrentFrame always re-shows +0x188 when the retain
    // matches paint geom — frozen at presentFrame (present boundary only).
    if state.presentation.present.frame().is_valid()
        && state
            .presentation
            .present
            .frame()
            .matches_geometry(width, height)
        && !state.presentation.present.frame().pixels().is_empty()
    {
        if !state.presentation.present.frame().encode_pending()
            && state.presentation.present.console_already_painted(
                state.presentation.present.frame().mapping(),
                state.presentation.present.frame().generation(),
            )
        {
            crate::observe::off(format!(
                "present_paint Unchanged mid={} gen={} (console already holds +0x188)",
                state.presentation.present.frame().mapping(),
                state.presentation.present.frame().generation()
            ));
            return ScanoutCopyResult::Unchanged;
        }
        if blit_present_snapshot(state, dst, dst_stride, width, height) {
            let shown_mid = state.presentation.present.frame().mapping();
            let shown_gen = state.presentation.present.frame().generation();
            // Reuse the fused scan `capture_present_frame` already ran over this
            // frozen frame instead of two more full 8 MiB passes under the lock.
            // `frame_bgra` has a single writer (capture), so a matching
            // mapping+generation means the stashed stats describe these exact
            // bytes; a mismatch (e.g. a test-injected frame) falls back to a scan.
            // Per-paint census on the QEMU display thread — ~30k lines/session
            // each under a continuously-animating app, plus an O(w·h)
            // `bgra_present_stats` full-frame scan built PURELY to
            // populate the log. Both the scan and the two lines are log-only
            // (nothing below consumes the stats), so gate the whole block behind
            // REIMS_VGPU_DRAW_LOG: a normal boot pays neither the scan nor the flood.
            // The always-on present rate/occupancy signal lives in the
            // `present_proxy` summary.
            if crate::observe::draw_log_enabled() {
                let (nz, maxb, rgb_nz, max_rgb, px0) =
                    crate::observe::bgra_present_stats(state.presentation.present.frame().pixels());
                crate::observe::line(format!(
                    "scanout paint_snapshot mid={} (action mid={} gen={}) {}x{} retain_gen={} nz={} max={}",
                    shown_mid, mapping_id, expected_generation, width, height, shown_gen, nz, maxb
                ));
                crate::observe::line(format!(
                    "present_paint Painted mid={shown_mid} (action mid={mapping_id} gen={expected_generation}) {width}x{height} rgb_nz={rgb_nz} max_rgb={max_rgb} px0=[{},{},{},{}] (this is what QMP shows)",
                    px0[0], px0[1], px0[2], px0[3]
                ));
            }
            state
                .presentation
                .present
                .record_console_paint(shown_mid, width, height, shown_gen);
            // First successful +0x188 blit after capture clears encode pending.
            state.presentation.present.mark_frame_encoded();
            return ScanoutCopyResult::Painted;
        }
    }

    // Present-path after first content boundary: never fall through to live
    // paint_mapping of a clear-only dual-mid (would freeze console black).
    let post_boundary = state.presentation.present.content_boundary_crossed()
        && state
            .presentation
            .present
            .console_matches_geometry(width, height);

    if post_boundary {
        let is_current_present = state.presentation.present.is_current_present(mapping_id);

        // Capture failed at DisplaySwap for the still-current present — retry once.
        if is_current_present
            && (state.presentation.present.frame().encode_pending()
                || !state.presentation.present.frame().is_valid())
            && (expected_generation == 0
                || expected_generation == state.presentation.present.console_generation())
        {
            let gen = if expected_generation != 0 {
                expected_generation
            } else {
                state.presentation.present.console_generation()
            };
            let _ = capture_present_frame(state, mapping_id, width, height, gen);
            if state.presentation.present.frame().is_valid()
                && state
                    .presentation
                    .present
                    .frame()
                    .matches_geometry(width, height)
                && blit_present_snapshot(state, dst, dst_stride, width, height)
            {
                let shown_mid = state.presentation.present.frame().mapping();
                let shown_gen = state.presentation.present.frame().generation();
                state
                    .presentation
                    .present
                    .record_painted_identity(shown_mid, shown_gen);
                state.presentation.present.mark_frame_encoded();
                return ScanoutCopyResult::Painted;
            }
        }

        if is_current_present {
            crate::observe::fail(format!(
                "scanout post_boundary no retain mid={mapping_id} {width}x{height} gen={expected_generation}"
            ));
            return ScanoutCopyResult::Failed;
        }
        return ScanoutCopyResult::Unchanged;
    }

    // Pre-boundary only: live mapping paint (early logo/pill before first RGB retain).
    let _ = crate::runtime::mapper::ensure_resolved_for_scanout(state, host, mapping_id);

    // Only latch painted_generation on a real pixel source. A clear-to-black
    // fallback must not stamp generation — that freezes the console on black
    // forever when the first paint races the mapper (Unchanged on next gen).
    if paint_mapping(
        state,
        host,
        mapping_id,
        PaintDst {
            bytes: dst,
            stride: dst_stride,
            width,
            height,
        },
        crate::runtime::render_writeback::SettleSite::ScanoutPaint,
    ) {
        let need = (height as usize)
            .saturating_mul(width as usize)
            .saturating_mul(4);
        let sample = &dst[..need.min(dst.len())];
        let (nz, maxb) = crate::observe::nonzero_stats(sample);
        crate::observe::line(format!(
            "scanout paint_mapping ok mid={} {}x{} gen={} nz={} max={}",
            mapping_id, width, height, expected_generation, nz, maxb
        ));
        state.presentation.present.record_console_paint(
            mapping_id,
            width,
            height,
            expected_generation,
        );
        ScanoutCopyResult::Painted
    } else if paint_efi_console(state, host, dst, dst_stride, width, height) {
        // EFI/BAR1 fallback fills the console for early verbose boot only.
        // Do **not** latch painted_mapping/generation to the product mid —
        // that made post-capture Unchanged skip +0x188 (logo/pill retain)
        // while the console still held EFI text.
        crate::observe::line(format!("scanout paint_efi ok {}x{}", width, height));
        state
            .presentation
            .present
            .establish_console(width, height, expected_generation);
        ScanoutCopyResult::Painted
    } else {
        // Always-on: a total paint failure means a black/stale console. Logging
        // this via the gated `line()` sink made the always-on fail log silently
        // lie about a black screen (scanout audit Rank-3).
        crate::observe::fail(format!(
            "scanout paint FAIL mid={} {}x{} gen={} (console black/stale)",
            mapping_id, width, height, expected_generation
        ));
        ScanoutCopyResult::Failed
    }
}

/// Paint from guest-programmed EFI framebuffer (MMIO 0x1210 + stride 0x1228).
///
/// Used by product scanout fallback and by the pre-boundary host console when
/// the guest relocates the kernel video console off BAR1 into system RAM
/// (live serial: `console relocated to 0xf1000000` while BAR1 freezes).
pub fn paint_efi_console<M: HostMemory + crate::runtime::host::HostOps>(
    state: &Device,
    host: &M,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) -> bool {
    let fb = state.registers.gfx.efi_fb_start;
    if fb == 0 {
        return false;
    }
    // The console is only ever the mode this device advertised: EFI_MODE_COUNT
    // is 1, so a request for any other geometry is not this framebuffer and the
    // caller must fall back. Note these are the ADVERTISED dims, not programmed
    // ones — the stride below is the only part the guest gets to set.
    let efi_w = EFI_BOOT_WIDTH;
    let efi_h = EFI_BOOT_HEIGHT;
    if width != efi_w || height != efi_h {
        return false;
    }
    let stride = if state.registers.gfx.efi_fb_stride != 0 {
        state.registers.gfx.efi_fb_stride
    } else {
        efi_w.saturating_mul(RGBA8_BPP)
    };
    if stride < efi_w.saturating_mul(RGBA8_BPP) {
        return false;
    }
    let row_bytes = (efi_w as usize) * (RGBA8_BPP as usize);
    // Refuse a span that is not guest RAM before reading a byte of it, because
    // this door is *expected* to be shut for most of early boot and the caller
    // has another one for exactly that case.
    //
    // `efi_fb_start` is whatever the guest programmed into 0x1210. Early on
    // that is the BAR1 GOP framebuffer this device exposes, which is device
    // memory rather than RAM — and the QEMU shim reads with `MemTxAttrs.memory`
    // set, so an address space read of it fails closed by design (a guest page
    // entry aimed at our own BAR would otherwise re-enter this device's MMIO
    // handler from inside a Rust call already holding the device lock). Only
    // once the kernel relocates the console into system RAM does this path have
    // anything it can read.
    //
    // Without the pre-flight the loop discovered that the expensive way: it read
    // rows until one crossed out of RAM, then returned false and threw every
    // row it had already copied away — measured at 465 completed reads per
    // attempt, repeated on the ~30 Hz early-console cadence. It also put a
    // `mem_qemu_read_gpa_callback_failed` line on the always-on fail channel,
    // where it read as a device fault rather than as the first of two doors
    // being shut.
    //
    // The pre-flight used to be two `is_ram_gpa` calls, on the span's first and
    // last byte, and that is a two-point sample of eight megabytes rather than a
    // check of it. A driven x86 boot refused a read 375 rows in — `address=
    // 0x802bf200 len=7680`, exactly `fb + 375 * stride` — with both endpoints
    // answering RAM, which is the sampled bound doing precisely what it looks
    // like it cannot. `first_non_ram_page` walks every page and short-circuits,
    // so the usual shut door still costs one call.
    //
    // A driven boot on the same guest image, with the same `efi_fb_start
    // 0x80000000` and `efi_fb_stride 0x1e00`, then read zero of that refusal and
    // zero of the alarm below — so the hole is interior and the two host doors
    // agree. Neither boot produced a `scanout paint FAIL`, so refusing earlier
    // cost the console nothing it was getting.
    let span_len = (efi_h.saturating_sub(1) as u64)
        .saturating_mul(stride as u64)
        .saturating_add(row_bytes as u64);
    if host
        .first_non_ram_page(fb, span_len, 1usize << state.page_shift)
        .is_some()
    {
        return false;
    }
    for y in 0..efi_h {
        let gpa = fb + (y as u64) * (stride as u64);
        let dst_off = (y as usize) * (dst_stride as usize);
        if dst_off + row_bytes > dst.len() {
            return false;
        }
        if let Err(error) = host.read_gpa(gpa, &mut dst[dst_off..dst_off + row_bytes]) {
            // Every page of this row answered RAM before the loop started and
            // the read refused anyway. The pre-flight named two ways that can
            // happen — the walk samples something the read does not ask, or the
            // layout moved between the two — and only the first is a defect.
            // Asking the walk again, about this row alone, is what tells them
            // apart, and it costs one call on a path that is about to return
            // false anyway.
            //
            // The second is a race this device cannot close and should not try
            // to: the span is eight megabytes copied a row at a time while an
            // early-boot guest is relocating its console out of our BAR1 into
            // system RAM, so the guest is entitled to unmap it mid-copy. The
            // caller falls back to the other door. What matters is that it stop
            // being reported as the first, which is a **healthy zero** — this
            // one is not, and a boot has now read it.
            let moved = host
                .first_non_ram_page(gpa, row_bytes as u64, 1usize << state.page_shift)
                .is_some();
            let decline = if moved {
                ConsoleEfiRowRefused::LeftRamMidCopy { row: y }
            } else {
                ConsoleEfiRowRefused::VouchedThenRefused { row: y }
            };
            crate::observe::Emit::decline("console_efi_row", &decline)
                .field("gpa", format!("{gpa:#x}"))
                .field("row_bytes", row_bytes)
                .field("fb", format!("{fb:#x}"))
                .field("error", format!("{error:?}"))
                .fail_once(u64::from(y));
            return false;
        }
    }
    true
}

/// A console row the RAM walk vouched for and whose read refused, and which of
/// the two reasons for that the walk gives when asked a second time.
///
/// Named rather than folded into the `mem_qemu_read_gpa_callback_failed` line
/// the adapter already emits, because those say different things: the adapter's
/// line reports that a host callback said no, and these report that it
/// contradicted the host callback consulted before it.
///
/// Two slugs rather than one because only one of them is a defect, and a single
/// slug would have let the benign one spend the other's `fail_once` latch and
/// its claim to being a healthy zero. That is not hypothetical: this was one
/// slug documented as unreachable, and the first boot that read it read the
/// benign case.
#[derive(Debug, Clone, Copy)]
enum ConsoleEfiRowRefused {
    /// The row still answers RAM. The two host doors disagree about the same
    /// bytes, which is a defect in one of them — the pre-flight walk is the
    /// whole reason this arm should be unreachable.
    VouchedThenRefused { row: u32 },
    /// The row no longer answers RAM. The guest unmapped it between the
    /// pre-flight and this row's turn in the copy, which an early-boot guest
    /// relocating its console off this device's BAR1 is entitled to do at any
    /// moment. Nothing is wrong here; the caller falls back to the other door.
    LeftRamMidCopy { row: u32 },
}

impl crate::observe::Decline for ConsoleEfiRowRefused {
    fn slug(&self) -> &'static str {
        match self {
            Self::VouchedThenRefused { .. } => "console_efi_row_vouched_then_refused",
            Self::LeftRamMidCopy { .. } => "console_efi_row_left_ram_mid_copy",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        let row = match self {
            Self::VouchedThenRefused { row } | Self::LeftRamMidCopy { row } => row,
        };
        vec![("row", row.to_string())]
    }
}

/// Why a console capture paint produced no pixels.
///
/// Every one of these shows as a black or stale console, so the reason is the
/// whole diagnostic for the "why is it black" class.
///
/// # Why these are prefixed
///
/// Bare, three of them were claimed by another rail: `unmapped` and `short_view`
/// were also `import_present`'s words for different checks, and `no_mapping` was
/// also the IOSurface plane view loader's — so `grep reason=unmapped` returned a mix of the
/// capture rail and the import rail and could not be read. The `capture_` prefix
/// is the same fix the slate reasons and the MRT proxies took.
///
/// # Two names became six
///
/// `short_view` stood for one `if` with three `||`-ed conditions — a null host
/// pointer, a view shorter than the sample window, and a base offset past the
/// end of it — which are three different faults with three different fixes.
/// `read_multi_row_oob` and `read_multi_missing` each stood for two sites whose
/// bounds differ (the convert path slices `tight` bytes, the direct path
/// `min(dst_row, tight)`), so they can fire under different conditions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CaptureDecline {
    /// No mapping record for this id.
    NoMapping,
    /// The guest has paged the surface off.
    Unmapped,
    /// Mapped, but the page list is empty — a resolve gap.
    NoPages,
    /// Geometry has never been latched.
    NoGeom,
    /// The mapping's latched geometry is not the console's.
    GeomMismatch { have_w: u32, have_h: u32 },
    /// The pixel format has no known bytes-per-pixel.
    BppUnknown { format: u16 },
    /// The pixel format has no known tight row size.
    TightRowUnknown { format: u16 },
    /// No IOSurface texture sample window could be derived.
    NoSampleWindow,
    /// The descriptor's row stride is narrower than a tight row.
    BprBelowTight { bpr: u64, tight: u32 },
    /// The contig host view resolved to a null pointer.
    ContigViewNull,
    /// The contig host view is shorter than the sample window.
    ContigViewShort { contig_len: u64, span_end: u64 },
    /// The sample window's base is at or past its end, so there is nothing to
    /// read. `contig` says which path found it, because the two reach it
    /// differently and the check is the same.
    BaseBeyondSpan {
        base_off: u64,
        span_end: u64,
        contig: bool,
    },
    /// The fragmented multi-import read of the sample window failed.
    MultiReadFailed { len: usize },
    /// The destination row would run past the end of the console buffer.
    DstOverflow { row: u32 },
    /// Converting path: the requested row lies outside the multi-import buffer.
    ConvertRowOob { row: u32 },
    /// Converting path: neither a contig base nor a multi-import buffer exists.
    ConvertRowMissing { row: u32 },
    /// Converting path: the guest format could not be converted to RGBA8.
    ConvertToRgba { format: u16 },
    /// Converting path: RGBA8 could not be converted back to the console's BGRA8.
    ConvertFromRgba,
    /// Direct-BGRA path: the requested row lies outside the multi-import buffer.
    DirectRowOob { row: u32 },
    /// Direct-BGRA path: neither a contig base nor a multi-import buffer exists.
    DirectRowMissing { row: u32 },
}

impl crate::observe::Decline for CaptureDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::NoMapping => "capture_no_mapping",
            Self::Unmapped => "capture_unmapped",
            Self::NoPages => "capture_no_pages",
            Self::NoGeom => "capture_no_geom",
            Self::GeomMismatch { .. } => "capture_geom_mismatch",
            Self::BppUnknown { .. } => "capture_bpp_unknown",
            Self::TightRowUnknown { .. } => "capture_tight_row_unknown",
            Self::NoSampleWindow => "capture_no_sample_window",
            Self::BprBelowTight { .. } => "capture_bpr_below_tight",
            Self::ContigViewNull => "capture_contig_view_null",
            Self::ContigViewShort { .. } => "capture_contig_view_short",
            Self::BaseBeyondSpan { .. } => "capture_base_beyond_span",
            Self::MultiReadFailed { .. } => "capture_multi_read_failed",
            Self::DstOverflow { .. } => "capture_dst_overflow",
            Self::ConvertRowOob { .. } => "capture_convert_row_oob",
            Self::ConvertRowMissing { .. } => "capture_convert_row_missing",
            Self::ConvertToRgba { .. } => "capture_convert_to_rgba",
            Self::ConvertFromRgba => "capture_convert_from_rgba",
            Self::DirectRowOob { .. } => "capture_direct_row_oob",
            Self::DirectRowMissing { .. } => "capture_direct_row_missing",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::GeomMismatch { have_w, have_h } => vec![("have", format!("{have_w}x{have_h}"))],
            Self::BppUnknown { format }
            | Self::TightRowUnknown { format }
            | Self::ConvertToRgba { format } => vec![("format", format.to_string())],
            Self::BprBelowTight { bpr, tight } => {
                vec![("bpr", bpr.to_string()), ("tight", tight.to_string())]
            }
            Self::ContigViewShort {
                contig_len,
                span_end,
            } => vec![
                ("contig_len", contig_len.to_string()),
                ("span_end", span_end.to_string()),
            ],
            Self::BaseBeyondSpan {
                base_off,
                span_end,
                contig,
            } => vec![
                ("base_off", base_off.to_string()),
                ("span_end", span_end.to_string()),
                ("contig", u8::from(*contig).to_string()),
            ],
            Self::MultiReadFailed { len } => vec![("len", len.to_string())],
            Self::DstOverflow { row }
            | Self::ConvertRowOob { row }
            | Self::ConvertRowMissing { row }
            | Self::DirectRowOob { row }
            | Self::DirectRowMissing { row } => vec![("row", row.to_string())],
            _ => Vec::new(),
        }
    }
}

/// Where a mapping paint lands: the buffer, its row stride, and the extent to
/// fill. One parameter because the four travel together through every caller and
/// a stride that belongs to a different buffer is the mistake worth making
/// unspellable.
struct PaintDst<'a> {
    bytes: &'a mut [u8],
    stride: u32,
    width: u32,
    height: u32,
}

/// `site` is the caller's own, not this leaf's. Both callers read the same guest
/// pages the same way, and they are a once-a-boot console paint and a draw-rate
/// sampled bind — so charging one slug made the console's settle rate unreadable
/// and the sampled arm's invisible. Taken as an argument rather than named here
/// so a third caller has to state which it is.
fn paint_mapping<M: HostMemory + crate::runtime::host::HostOps>(
    state: &mut Device,
    host: &mut M,
    mapping_id: u32,
    dst: PaintDst<'_>,
    site: crate::runtime::render_writeback::SettleSite,
) -> bool {
    let PaintDst {
        bytes: dst,
        stride: dst_stride,
        width,
        height,
    } = dst;
    use crate::runtime::mapping_write::iosurface_texture_sample_window;

    // Every `false` return here shows as a black/stale console; log the specific
    // reason so the "why is it black" class is diagnosable (scanout audit Rank-3).
    // Each site exits the function, so this fires at most once per paint call.
    let fail = |d: CaptureDecline| -> bool {
        crate::observe::Emit::decline("scanout_paint_mapping", &d)
            .field("mid", mapping_id)
            .field("want", format!("{width}x{height}"))
            .fail();
        false
    };

    // The wait narrows to this mapping's own pages, which `settle_for_mapping`
    // does for every caller now. Here it costs no walk at all: `page_entries`
    // already *is* the page list, and writers use the same field to build their
    // destination. A mapping with no page
    // list, or one holding an entry that names no backing, cannot be ruled out
    // and settles.
    crate::runtime::writeback_debt::settle_for_mapping(state, mapping_id, site);

    let Some(m) = state.surfaces.mappings.get(&mapping_id) else {
        return fail(CaptureDecline::NoMapping);
    };
    // Split the two teardown-window causes so a scanout paint miss names which
    // one fired (AGENTS.md: each distinct check owns its slug) — `unmapped` is
    // the guest having paged the surface off, `no_pages` an empty page list from
    // a resolve gap; both are benign-transient but must stay distinguishable.
    if !m.lifecycle.active {
        return fail(CaptureDecline::Unmapped);
    }
    if m.pages.entries.is_empty() {
        return fail(CaptureDecline::NoPages);
    }
    // Geometry must be latched (same rule as write_bgra8 / archive scanout_iosurface_texture).
    if !m.has_geometry() || m.width_or_zero() == 0 || m.height_or_zero() == 0 {
        return fail(CaptureDecline::NoGeom);
    }
    let mw = m.width_or_zero();
    let mh = m.height_or_zero();
    let format = if m.format_or_zero() != 0 {
        m.format_or_zero()
    } else {
        MTL_FORMAT_BGRA8_UNORM
    };
    if mw != width || mh != height {
        return fail(CaptureDecline::GeomMismatch {
            have_w: mw,
            have_h: mh,
        });
    }
    let Some(bpp) = pixel_format::bytes_per_pixel(format) else {
        return fail(CaptureDecline::BppUnknown { format });
    };
    let _ = bpp;
    let Some(tight) = pixel_format::tight_row_bytes(mw, format) else {
        return fail(CaptureDecline::TightRowUnknown { format });
    };
    // Same sample window as writeback (device descriptor base/bpr when present).
    let Some((base_off, bpr_u32, span_end)) = iosurface_texture_sample_window(m, mw, mh, format)
    else {
        return fail(CaptureDecline::NoSampleWindow);
    };
    let bpr = bpr_u32 as usize;
    if (bpr as u64) < tight as u64 {
        return fail(CaptureDecline::BprBelowTight {
            bpr: bpr as u64,
            tight,
        });
    }
    // Contig HostOps view when possible; multi-import read_mapping_bytes otherwise.
    // Never plan_span / read_gpa walk (freelist class).
    let contig = crate::runtime::mapper::ensure_contig_view(state, host, mapping_id);
    if let Some((ptr, contig_len)) = contig {
        // Three separate faults, three separate names: the view resolved to
        // nothing, the view is shorter than the window, or the window itself is
        // degenerate. They shared one `short_view` and one `||`.
        if ptr == 0 {
            return fail(CaptureDecline::ContigViewNull);
        }
        if (contig_len as u64) < span_end {
            return fail(CaptureDecline::ContigViewShort {
                contig_len: contig_len as u64,
                span_end,
            });
        }
        if base_off >= span_end {
            return fail(CaptureDecline::BaseBeyondSpan {
                base_off,
                span_end,
                contig: true,
            });
        }
    } else if base_off >= span_end {
        return fail(CaptureDecline::BaseBeyondSpan {
            base_off,
            span_end,
            contig: false,
        });
    }
    // SAFETY: when Some, contig_len covers span_end; base_off < span_end.
    let base = contig.map(|(ptr, _)| unsafe { (ptr as *const u8).add(base_off as usize) });
    // Fragmented fullscreen IOSurfaces have hundreds of packed GPA runs. Read
    // the sample window once, not once per row: read_mapping_bytes revalidates
    // and rebuilds the run plan, so a row loop made setup O(height × pages)
    // (live 1920×1080 cold draw: setup_us≈7.5s for 2040 pages).
    let multi = if base.is_none() {
        let len = span_end.saturating_sub(base_off) as usize;
        let mut bytes = vec![0u8; len];
        if !crate::runtime::mapper::read_mapping_bytes(
            state, host, mapping_id, base_off, &mut bytes,
        ) {
            return fail(CaptureDecline::MultiReadFailed { len });
        }
        Some(bytes)
    } else {
        None
    };

    let mut src_row = vec![0u8; tight as usize];
    let mut rgba_row = if format == MTL_FORMAT_BGRA8_UNORM
        || format == pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB
    {
        None
    } else {
        Some(vec![0u8; (mw as usize) * (RGBA8_BPP as usize)])
    };

    for y in 0..mh {
        let dst_off = (y as usize) * (dst_stride as usize);
        let dst_row_len = (mw as usize) * (RGBA8_BPP as usize);
        if dst_off + dst_row_len > dst.len() {
            return fail(CaptureDecline::DstOverflow { row: y });
        }
        let src_off = (y as usize).saturating_mul(bpr);

        if let Some(ref mut rgba) = rgba_row {
            // Non-BGRA source: stage the tight guest row, then convert via RGBA8.
            if let Some(base) = base {
                let src = unsafe { base.add(src_off) };
                unsafe {
                    std::ptr::copy_nonoverlapping(src, src_row.as_mut_ptr(), tight as usize);
                }
            } else if let Some(bytes) = multi.as_ref() {
                let end = src_off.saturating_add(tight as usize);
                let Some(row) = bytes.get(src_off..end) else {
                    return fail(CaptureDecline::ConvertRowOob { row: y });
                };
                src_row.copy_from_slice(row);
            } else {
                return fail(CaptureDecline::ConvertRowMissing { row: y });
            }
            let dst_row = &mut dst[dst_off..dst_off + dst_row_len];
            if !convert_row_to_rgba8(format, &src_row[..tight as usize], mw, rgba) {
                return fail(CaptureDecline::ConvertToRgba { format });
            }
            if !convert_rgba8_to_row(MTL_FORMAT_BGRA8_UNORM, rgba, mw, dst_row) {
                return fail(CaptureDecline::ConvertFromRgba);
            }
        } else {
            // Already BGRA8 — copy the guest row straight into dst. Skipping the
            // src_row bounce halves the per-present capture memcpy traffic (the
            // dominant `paint_us` cost on the present drain lock).
            let copy_len = dst_row_len.min(tight as usize);
            let dst_row = &mut dst[dst_off..dst_off + dst_row_len];
            if let Some(base) = base {
                let src = unsafe { base.add(src_off) };
                unsafe {
                    std::ptr::copy_nonoverlapping(src, dst_row.as_mut_ptr(), copy_len);
                }
            } else if let Some(bytes) = multi.as_ref() {
                let end = src_off.saturating_add(copy_len);
                let Some(row) = bytes.get(src_off..end) else {
                    return fail(CaptureDecline::DirectRowOob { row: y });
                };
                dst_row[..copy_len].copy_from_slice(row);
            } else {
                return fail(CaptureDecline::DirectRowMissing { row: y });
            }
            if (tight as usize) < dst_row_len {
                dst_row[tight as usize..].fill(0);
            }
        }
    }
    true
}

/// Resolve host-visible width/height for a scanout action from guest mapping geom.
pub fn present_dims(state: &Device, mapping_id: u32) -> (u32, u32) {
    if let Some(m) = state.surfaces.mappings.get(&mapping_id) {
        if m.has_geometry() && m.width_or_zero() > 0 && m.height_or_zero() > 0 {
            return (m.width_or_zero(), m.height_or_zero());
        }
    }
    if let Some(geometry) = state.presentation.present.console_dimensions() {
        return geometry;
    }
    (0, 0)
}

/// After a successful IOSurface texture color writeback: maybe latch front mapping / paint.
///
/// Contract:
/// - **PGDisplay**: present names one surface; mode size = that surface's geom
///   (`modeChangeHandler` sizeInPixels). We never invent host mode sizes.
/// - **Archive same_geom**: paint pre-boundary only when console unset or job
///   W×H equals established console (strips/other RTs do not resize the window).
/// - **Live Monterey**: early logo also lands in BGRA8/RGBA8 IOSurface texture (not only
///   0x73); accept those formats pre-boundary. Post-boundary paint is DisplaySwap
///   only — writebacks must not rename `present_mapping` after `frame_flush_seen`.
pub fn note_front_buffer_writeback<M: HostMemory + crate::runtime::host::HostOps>(
    state: &mut Device,
    host: &mut M,
    mapping_id: u32,
    width: u32,
    height: u32,
    rt_format: u16,
) {
    use crate::runtime::host::HostAction;

    if mapping_id == 0 || !scanout_extent_ok(width, height) {
        return;
    }
    let (map_fmt, mapped_ok, has_geom, map_w, map_h, gen) =
        match state.surfaces.mappings.get(&mapping_id) {
            Some(m) => (
                m.format_or_zero(),
                m.lifecycle.active && !m.pages.entries.is_empty(),
                m.has_geometry() && m.width_or_zero() > 0 && m.height_or_zero() > 0,
                m.width_or_zero(),
                m.height_or_zero(),
                m.content.guest_page_generation,
            ),
            None => return,
        };
    let fmt = if rt_format != 0 { rt_format } else { map_fmt };
    if !is_front_buffer_format(fmt) {
        return;
    }
    if !mapped_ok {
        return;
    }

    // After the first CmdDisplaySwap, writebacks must not rename
    // `present_mapping` (PGDisplay presents the surface named by DisplaySwap).
    // Still track Composite full-FB writebacks as dual-mid peer for ClearOnly
    // present capture (x86: present mid 2/3 ClearOnly, content mid 1/4/5).
    if state.presentation.present.content_boundary_crossed() {
        if matches!(
            state.surface_write_kind(mapping_id),
            crate::model::SurfaceWriteKind::Composite
        ) && state
            .presentation
            .present
            .console_matches_geometry(width, height)
        {
            // Track the latest Composite full-FB writeback. Pre-boundary this
            // feeds `early_scanout_target`; post-boundary it is the peer named
            // on the `front_wb` / `present_order_hold` lines. Always update
            // here so a later writeback into the same mid refreshes the gen.
            state.presentation.present.note_early_composite(mapping_id);
        }
        return;
    }

    // Archive same_geom vs s->surface: first enqueue establishes provisional
    // console size; later early-boot paints only when job matches. No min/max
    // dimension clamps — different sizes are refused, not rewritten to EFI.
    let console_established = state.presentation.present.console_geometry().is_some();
    if console_established
        && !state
            .presentation
            .present
            .console_matches_geometry(width, height)
    {
        // Still latch which front the compositor is writing for early_scanout,
        // but do not resize/paint (mode change waits for DisplaySwap).
        state
            .presentation
            .present
            .note_present_candidate(mapping_id);
        return;
    }

    // HostAction size = mapper registry geom when known (archive
    // scanout_iosurface_mapping / PG sizeInPixels from the named surface).
    let (paint_w, paint_h) = if has_geom {
        (map_w, map_h)
    } else {
        (width, height)
    };
    if !scanout_extent_ok(paint_w, paint_h) {
        return;
    }

    // Establish console size at enqueue so subsequent different-geom writebacks
    // hit the barrier even before C finishes copy (archive surface after paint
    // is sequential; our async HostAction queue needs the latch here).
    state
        .presentation
        .present
        .note_present_candidate(mapping_id);
    state
        .presentation
        .present
        .establish_console(paint_w, paint_h, gen);
    // Sticky early front: only Composite Stores own the pre-boundary console.
    // ClearOnly buffer-setup presents may overwrite present_mapping but must not
    // clear this — otherwise gfx_update falls back to BAR1 (kdp log thrash).
    if matches!(
        state.surface_write_kind(mapping_id),
        crate::model::SurfaceWriteKind::Composite
    ) {
        state.presentation.present.note_early_composite(mapping_id);
    }

    crate::observe::line(format!(
        "front_wb LATCH mid={mapping_id} {paint_w}x{paint_h} gen={gen} fmt={fmt:#x} early_front={} (pre-boundary early paint enqueue)",
        state.presentation.present.early_composite_mapping()
    ));
    host.enqueue(HostAction::scanout_gen(mapping_id, paint_w, paint_h, gen));
}

/// Target for pre-boundary `gfx_update` re-pull (archive fb_update path).
///
/// Guest mapping id + geometry matching the **established console** only.
/// `None` after DisplaySwap (Apple hostPresentCount re-show only).
///
/// **Mode-switch contract:** writebacks may latch
/// `present_mapping` to a new-resolution FB before the present boundary, but
/// must not resize the host window. Only [`note_front_buffer_writeback`]
/// same-geom paints and **CmdDisplaySwap** (HostAction) may change console
/// size — matching archive same_geom + PG `modeChangeHandler` at present.
///
/// # One source, and it is the guest's own statement
///
/// The pre-boundary console shows `early_front_mapping`: the mapping the guest
/// most recently **composited** into (`SurfaceWriteKind::Composite`). That is a
/// decoded fact with a sentence — the guest composited into M, so the console
/// should show M — and it is the only source here.
///
/// This used to rank a second candidate, `present_mapping`, which pre-boundary
/// is the last writeback of *any* kind including a ClearOnly buffer-setup flip.
/// It had no sentence; it stood in for "no Composite writeback has happened
/// yet", which is a state whose meaning we do not know. Two instrumented x86 /
/// Vulkan boots settled it: across 8 952 pre-boundary calls the composite front
/// served 7 times and the fallback served **zero**, because in every one of the
/// 8 952 calls where the composite front was unset the fallback was unset too.
/// It never once had a value to contribute. The counters also never recorded a
/// composite front rejected for ClearOnly, format, geometry or dimensions — the
/// case the fallback and the first field's stickiness were both built for did
/// not occur.
///
/// Blast radius, for whoever revisits this: it returns `None` once
/// `frame_flush_seen`, so all of it is early boot only.
pub fn early_scanout_target(state: &Device) -> Option<(u32, u32, u32, u32)> {
    if state.presentation.present.content_boundary_crossed() {
        return None;
    }
    let mapping_id = state.presentation.present.early_composite_mapping();
    if mapping_id == 0 {
        return None;
    }
    // ClearOnly init without retain: refuse (would feed solid black).
    if matches!(
        state.surface_write_kind(mapping_id),
        crate::model::SurfaceWriteKind::ClearOnly
    ) && !state.presentation.present.frame().is_valid()
    {
        return None;
    }
    let m = state.surfaces.mappings.get(&mapping_id)?;
    if !m.lifecycle.active {
        return None;
    }
    if m.format_or_zero() != 0 && !is_front_buffer_format(m.format_or_zero()) {
        return None;
    }
    let (w, h) = present_dims(state, mapping_id);
    if w == 0 || h == 0 {
        return None;
    }
    if state.presentation.present.console_geometry().is_some()
        && !state.presentation.present.console_matches_geometry(w, h)
    {
        return None;
    }
    Some((mapping_id, w, h, m.content.guest_page_generation))
}

#[cfg(test)]
mod tests;
