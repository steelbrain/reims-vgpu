//! Vulkan-backend half of the [`super`] draw encode path: sampled-source
//! resolution, zero-copy load paths, metal2vulkan draw submission, and the
//! deferred GVA/surface store windows.
//!
//! The whole module is gated on `backend-vulkan` at its declaration in
//! [`super`], which also re-exports these items flat so callers keep addressing
//! them as `crate::runtime::metal_draw::<name>`. `use super::*` pulls in the
//! parent's imports, which this half shares.

use super::*;

/// Vulkan image shape for a reflected Metal sampled-image dimensionality.
///
/// The engine caps array layers at 1 (a single-layer array is still a distinct
/// descriptor type from a plain 2D image), so array shapes report `layers = 1`
/// and a genuinely multi-layer source declines on its byte length rather than
/// binding a truncated array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SampledImageShape {
    arrayed: bool,
    volume: bool,
    cube: bool,
    one_dim: bool,
    layers: u32,
}

/// Map a translated SPIR-V sampled-image dimensionality onto the Vulkan image
/// shape the sampled-draw path builds. `None` is a shape the path cannot yet
/// express (`Cube` / `CubeArray`); the caller declines it by name so the gap
/// stays visible instead of binding the wrong view type.
fn sampled_image_shape(
    kind: crate::runtime::spirv_bind::SampledImageKind,
) -> Option<SampledImageShape> {
    use crate::runtime::spirv_bind::SampledImageKind;
    let (arrayed, volume, cube, one_dim) = match kind {
        SampledImageKind::D1 => (false, false, false, true),
        SampledImageKind::D1Array => (true, false, false, true),
        SampledImageKind::D2 => (false, false, false, false),
        SampledImageKind::D2Array => (true, false, false, false),
        SampledImageKind::D3 => (false, true, false, false),
        SampledImageKind::Cube | SampledImageKind::CubeArray => return None,
    };
    Some(SampledImageShape {
        arrayed,
        volume,
        cube,
        one_dim,
        layers: 1,
    })
}

pub fn encode_draw_and_writeback<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    req: &mut DrawEncodeRequest,
) -> EncodeStatus {
    encode_draw_chain(state, host, req, true, true).0
}

/// Linux / non-Apple product rail: metal2vulkan + Vulkan offscreen, then Store.
///
/// `writeback_guest` is the archive multi-draw store plan (only the last record
/// of a serialized render-pass chain writes guest memory). Intermediate records **must still
/// encode** and return color0 for chaining — returning `NoMetal` when
/// `!writeback_guest` aborted every multi-draw stream after the first
/// record (live `draw_fail_clear_fallback nometal=1` on clear+draw packets).
pub fn encode_draw_chain<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    req: &mut DrawEncodeRequest,
    writeback_guest: bool,
    // Inert on this arm, and by construction rather than by omission: the Metal
    // arm consults it in `store_seed_policy` to suppress a scissor-local store,
    // and this rail has no scissor-local store to suppress — `req.scissor` only
    // ever reaches the pipeline scissor rect, never the Store extent.
    _force_full_store: bool,
) -> (EncodeStatus, Option<Vec<u8>>) {
    // Charges this chain to one phase at a time all the way down, including the
    // parts of it that live inside `try_metal2vulkan_draw`. Held here rather
    // than there because the Store routing below the engine is on the same
    // clock: `drain_duty`'s `draw_us` brackets exactly this function, and the
    // whole reading is that the phases sum to it.
    let _phase = crate::runtime::chain_phase::ChainTimer::start();
    let colors: Vec<ColorRtRequest> = req.colors.clone();
    let Some((pass_w, pass_h)) = colors.first().map(|c0| (c0.width, c0.height)) else {
        return (EncodeStatus::BadArgs("draw_vk_no_color_target"), None);
    };

    let mut any_store = false;
    let mut color0_rgba: Option<Vec<u8>> = None;
    // Solid CLEAR seed Stores only when this record owns guest writeback
    // (last of a serialized chain, or unified always-writeback).
    if writeback_guest {
        for (i, c) in colors.iter().enumerate() {
            if c.store_action != PASS_STORE_ACTION_STORE {
                continue;
            }
            if c.load_action != PASS_LOAD_ACTION_CLEAR
                && c.load_action != PASS_LOAD_ACTION_DONT_CARE
            {
                // Load/composite needs real encode (metal2vulkan) — skip Store.
                continue;
            }
            if c.width == 0 || c.height == 0 {
                continue;
            }
            let rgba = solid_rgba_local(c.width, c.height, &c.clear_color);
            let ok = if c.target_gva != 0 {
                supersede_gva_window(state, host, c.target_gva, c.width, c.height, "clear_store");
                write_gva_rgba8(
                    state,
                    host,
                    req.task_id,
                    c.target_gva,
                    c.width,
                    c.height,
                    c.row_stride,
                    c.format,
                    &rgba,
                )
                .is_ok()
            } else if c.mapping_id != 0 {
                // Type-11 CLEAR. `write_bgra8` takes guest scanout order and
                // converts to the mapping's native format per row; it handles a
                // fragmented mapping too, staging native rows and landing them
                // through `mapper::write_mapping_bytes`. (A comment here used to
                // call it contig-only, which it has not been.)
                let bgra = swap_rb_channels(&rgba);
                let stride = c.width.saturating_mul(RGBA8_BPP);
                mapping_write::write_bgra8(
                    state,
                    host,
                    c.mapping_id,
                    &bgra,
                    stride,
                    c.width,
                    c.height,
                )
            } else {
                false
            };
            if ok {
                any_store = true;
                if i == 0 {
                    color0_rgba = Some(rgba);
                }
                crate::observe::line(format!(
                    "linux_clear_store mid={} gva={:#x} {}x{} pipe={} load={}",
                    c.mapping_id, c.target_gva, c.width, c.height, req.pipeline_ref, c.load_action
                ));
            }
        }
    }

    // Pages the synchronous GVA Store below is allowed to reach, resolved
    // **before** any GPU work.
    //
    // That Store's write was documented as needing no bound because "the command
    // being executed is what names its destination, and its authorisation is the
    // page table at the moment it runs". That is true of the CLEAR store above,
    // which is a solid colour written on this thread with nothing in between. It
    // is not true here: `try_metal2vulkan_draw` encodes, submits, waits for the
    // GPU and reads the result back, and only then does the Store resolve
    // `target_gva`. The guest runs on its own vCPUs across that round trip, so
    // the walk that finds the destination is not the walk that the command
    // authorised — which is exactly the shape the deferred rail was corrupting
    // guest memory through before it was bounded.
    //
    // Resolved here rather than at the write so the set predates the submit.
    // `None` when there is no GVA target, no writeback, or the walk cannot name
    // the span — an unresolvable span is not an authorisation to write anywhere.
    let sync_store_pages =
        sync_store_allowed_pages(state, host, req.task_id, colors.first(), writeback_guest);
    // metal2vulkan path: load MTLB → AIR → SPIR-V → internal Vulkan engine offscreen.
    let mut draw_rgba: Option<Vec<u8>> = None;
    // Physical order of `draw_rgba`. A type-11 composite Store renders into a
    // BGRA `Surface` resident, so its readback is already in guest scanout
    // order; the pooled and GVA targets stay RGBA. Carried instead of assumed —
    // which of those a record hit depends on whether an identity resolved, and
    // that is not a condition the Store block can re-derive.
    let mut draw_bgra = false;
    // GVA Store landed as a deferred-writeback window (resident authoritative).
    let mut gva_store_armed = false;
    // Type-11 composite Store landed the same way: the pinned engine resident is
    // the only copy of the frame until a guest-side reader flushes the window.
    let mut surface_store_armed = false;
    if req.pipeline_ref != 0 && (req.vertex_count > 0 || req.indexed.is_some()) {
        req.chain_resident_established = false;
        match try_metal2vulkan_draw(state, host, req, writeback_guest) {
            Ok(M2vDrawSpan::Pixels { bytes, bgra }) => {
                draw_rgba = Some(bytes);
                draw_bgra = bgra;
                crate::observe::line(format!(
                    "linux_m2v_draw ok pipe={} {}x{} vtx={}",
                    req.pipeline_ref, pass_w, pass_h, req.vertex_count
                ));
            }
            Ok(M2vDrawSpan::ResidentChain) => {
                req.chain_resident_established = true;
                crate::observe::line(format!(
                    "linux_m2v_draw ok resident_chain pipe={} {}x{} mid={} gva={:#x}",
                    req.pipeline_ref,
                    pass_w,
                    pass_h,
                    req.colors.first().map(|c| c.mapping_id).unwrap_or(0),
                    req.colors.first().map(|c| c.target_gva).unwrap_or(0)
                ));
            }
            Ok(M2vDrawSpan::ResidentGvaStore) => {
                if arm_gva_deferred_store(state, host, req) {
                    note_type11_store_route("gva_deferred");
                    gva_store_armed = true;
                    crate::observe::line(format!(
                        "linux_m2v_draw ok resident_gva_store pipe={} {}x{} gva={:#x}",
                        req.pipeline_ref,
                        pass_w,
                        pass_h,
                        req.colors.first().map(|c| c.target_gva).unwrap_or(0)
                    ));
                } else {
                    // Arm gate failed (unwalkable span / pin refusal): land
                    // synchronously from the resident the draw just produced.
                    // read_resident_chain fail-logs a lost resident.
                    note_type11_store_route("gva_deferred_sync");
                    draw_rgba = read_resident_chain(state, req);
                    crate::observe::line(format!(
                        "linux_m2v_draw ok resident_gva_store_sync_fallback pipe={} {}x{} gva={:#x} rgba={}",
                        req.pipeline_ref,
                        pass_w,
                        pass_h,
                        req.colors.first().map(|c| c.target_gva).unwrap_or(0),
                        draw_rgba.is_some() as u8
                    ));
                }
            }
            Ok(M2vDrawSpan::ResidentSurfaceStore) => {
                // Into the same `t11_store_us` bucket the synchronous and `Owned`
                // routes report, because the whole claim of this rail is that the
                // bucket shrinks. Leaving it unbracketed would move the arm's cost
                // into the residual `draw_phase` cannot attribute — which is
                // exactly the 28 % hole `b872e43` had to instrument, and it would
                // read as a win of the same size as the work it hid.
                let _store_span = StoreCostSpan::new("t11_store_us");
                let c0_store = req
                    .colors
                    .first()
                    .map(|c0| (c0.mapping_id, c0.width, c0.height, c0.format));
                let armed = c0_store.and_then(|(mid, cw, ch, _)| {
                    arm_surface_resident_store(state, host, req, mid, cw, ch)
                });
                match (armed, c0_store) {
                    (Some(epoch), Some((mid, cw, ch, fmt))) => {
                        note_type11_store_route("surface_resident");
                        // The same two publishes the `Owned` rail performs at arm
                        // time, for the same reason: `dense_frame_seq` gates
                        // `present_unbacked`, and a route that skipped it would
                        // make that gate structurally dead.
                        {
                            let _span = StoreCostSpan::new("t11_publish_us");
                            publish_surface_store(state, host, mid, cw, ch, fmt);
                        }
                        surface_store_armed = true;
                        crate::observe::line(format!(
                            "linux_m2v_draw ok resident_surface_store pipe={} {}x{} mid={mid} epoch={epoch}",
                            req.pipeline_ref, pass_w, pass_h
                        ));
                    }
                    _ => {
                        // The arm refused (its typed decline says which gate), so
                        // the frame has to be materialized after all: read the
                        // resident the draw just rendered into and let the
                        // synchronous Store block below run exactly as it does for
                        // a Store that never skipped its readback. This pays the
                        // readback the rail exists to avoid, which is the point —
                        // the fallback is a cost, never a lost frame.
                        note_type11_store_route("surface_resident_sync");
                        draw_rgba = read_resident_chain(state, req);
                        crate::observe::line(format!(
                            "linux_m2v_draw ok resident_surface_store_sync_fallback pipe={} {}x{} mid={} rgba={}",
                            req.pipeline_ref,
                            pass_w,
                            pass_h,
                            req.colors.first().map(|c| c.mapping_id).unwrap_or(0),
                            draw_rgba.is_some() as u8
                        ));
                    }
                }
            }
            Ok(M2vDrawSpan::None) => {
                crate::observe::line(format!(
                    "linux_m2v_draw skip pipe={} (no color0 geom)",
                    req.pipeline_ref
                ));
            }
            Err(e) => {
                // Always-on + latched: a rejected engine draw falls to the
                // clear-store fallback and surfaces as a bare `no_metal`
                // (the Safari padded-stride reject was invisible on a normal
                // boot — the content layer stayed blank with zero fail lines).
                // The decline names the specific check as the primary `reason=`
                // (an engine `vk_*` VkCall slug, a `DrawReason` refusal, or a
                // runtime `DrawPreparationDecline`).
                // The guest re-submits every frame, so latch on
                // (reason, pipeline_ref): a persistent reject cannot flood, but
                // a new reason on the same pipeline still surfaces.
                linux_m2v_draw_failure(&e, req).fail_once(req.pipeline_ref as u64);
            }
        }
    }

    // A resident render-pass chain intermediate: the exec loop reads
    // `chain_resident_established` and arms the next record's LoadFromTarget.
    if req.chain_resident_established {
        return (EncodeStatus::Ok, None);
    }

    // Deferred GVA Store: the window is armed and the resident holds the
    // authoritative pixels — the contract Store lands on first access.
    if gva_store_armed {
        return (EncodeStatus::Ok, None);
    }

    // Deferred type-11 composite Store: the window names the pinned resident and
    // the guest write lands on first access. `None`, not the frame, for the same
    // reason the `Owned` route returns `None` — `writeback_guest` is granted only
    // to the last record of a packet, so there is no record N+1 to seed.
    if surface_store_armed {
        return (EncodeStatus::Ok, None);
    }

    // A type-11 composite Store reaches the guest only through the CPU writeback
    // below. The DMA rail that used to short-circuit it here — a resident BGRA
    // target landed straight in the mapping's guest pages through an imported
    // host pointer — is gone, because a pointer the GPU can read is one it can
    // write and those pages are guest RAM.
    //
    // Taken, not borrowed. Every exit from this block returns the frame, and
    // borrowing forced each of them to `rgba.clone()` a whole framebuffer — 8 MB
    // at 1080p, at the 28-111 Stores/s `store_routes` measures, on the drain
    // worker `drain_duty` shows at duty 0.93-0.99. The deferred type-11 arm is
    // the hot one and it cloned purely to hand back the buffer it already owned.
    if let Some(mut rgba) = draw_rgba.take() {
        // Intermediate multi-draw GVA records: return color0 for chaining without
        // guest Store (archive store plan). Resident type-11 intermediates
        // returned above without materializing CPU pixels.
        if !writeback_guest {
            // A chain value seeds the next record, and `DrawRequest` states a
            // seed's order as `SeedOrder::Rgba8` for this rail.
            reorder_rb_in_place(&mut rgba, draw_bgra, false);
            return (EncodeStatus::Ok, Some(rgba));
        }
        // Store draw result into primary color RT.
        if let Some(c0) = colors.first() {
            // `rgb_nz`/`max_rgb` are diagnostic fields of the Store lines below,
            // and producing them is an O(w*h) pass over a whole framebuffer
            // readback — 2 073 600 pixels per Store at 1080p, at the 28-111
            // Stores/s `store_routes` measures under load. Computing it here
            // paid that on every route, including the type-11 one whose only
            // consumer is a `observe::line` a normal boot discards. Each arm
            // now scans only when it is about to write a line.
            // A free function, not a closure over `rgba`: the deferred arm below
            // takes ownership of that buffer, and a closure capturing it by
            // reference would pin it in place — which is the whole cost this
            // rail is removing.
            fn rgb_stats(rgba: &[u8]) -> (usize, u8) {
                let (nz, max, _) = crate::observe::rgba_rgb_stats(rgba);
                (nz, max)
            }
            let ok = if c0.mapping_id != 0 {
                // Unconditional. This used to be `if
                // type11_cpu_store_fallback_allowed(import_allowed)`, where
                // `import_allowed` asked whether the device could import a host
                // pointer over the mapping's guest pages; when it could, the
                // draw took the import rail and landing here was a fail-closed
                // error (`rgba_not_import`) that preserved the zero-copy
                // invariant. There is no invariant left to preserve, and the
                // else arm was a refusal for a rail that cannot be chosen.
                {
                    // Brackets the whole type-11 arm, into the same per-second
                    // window it divides into. `draw_phase` stops at the engine
                    // boundary, so this arm — the cache publish, the window arm,
                    // the guest scatter — is the bulk of the ~245 ms/s (28 % of
                    // `draw_us`) that no phase claimed.
                    let _span = StoreCostSpan::new("t11_store_us");
                    // Every consumer below wants guest scanout order: the
                    // deferred window's `write_bgra8`, `surface_cache`, and the
                    // synchronous route. A `Surface` resident reads back in that
                    // order already, so this is a no-op on the hot path and the
                    // ~152 ms/s whole-frame swizzle it replaces is gone. It still
                    // has to be written, because a record whose identity did not
                    // resolve rendered into a pooled RGBA target.
                    let mut bgra = rgba;
                    {
                        let _span = StoreCostSpan::new("t11_convert_us");
                        reorder_rb_in_place(&mut bgra, draw_bgra, true);
                    }
                    // Deferred writeback: publish the frame to `surface_cache`
                    // — the source every other consumer already reads, so the
                    // Load seed and the present capture see exactly what they
                    // would have — and arm a window instead of scattering the
                    // frame into the mapping's guest pages now.
                    //
                    // That scatter is the cost. `write_bgra8` converts every row
                    // to the mapping's native format and then copies it out, per
                    // row when the mapping's pages are fragmented: ~8 MB of CPU
                    // work per Store, at the 28-111 Stores/s `store_routes`
                    // measures, on the drain worker `drain_duty` shows at duty
                    // 0.93-0.99. Nothing on the host-window present path reads
                    // those pages, so most of that work is owed to a guest reader
                    // that may never come.
                    //
                    // The capability gate lives in `surface_store_defer_eligible`,
                    // which this reaches through `prepare_surface_deferred_window`;
                    // a denial arrives here as the `Err(bgra)` the sync route below
                    // needs anyway.
                    match arm_surface_deferred_store_with(
                        state,
                        host,
                        req,
                        c0.mapping_id,
                        c0.width,
                        c0.height,
                        bgra,
                    ) {
                        Ok(epoch) => {
                            note_type11_store_route("surface_deferred");
                            {
                                let _span = StoreCostSpan::new("t11_publish_us");
                                publish_surface_store(
                                    state,
                                    host,
                                    c0.mapping_id,
                                    c0.width,
                                    c0.height,
                                    c0.format,
                                );
                            }
                            stamp_type11_resident(state, host, req, writeback_guest, epoch);
                            // `None`, not the frame. This route is reached only
                            // under `writeback_guest`, which `multi_draw_store_plan`
                            // grants solely to the **last** record of a packet — so
                            // the returned pixels have no next record to seed
                            // (`exec.rs` feeds `chain_rgba` into record N+1's
                            // `target_seed_rgba` at line 1609), and every other
                            // reader of `chain_rgba` is an abandon arm inside a loop
                            // that has just ended. Returning the buffer forced this
                            // arm to clone it; returning nothing lets the arm own it.
                            return (EncodeStatus::Ok, None);
                        }
                        // Refused before consuming the frame, and handed it back, so
                        // the synchronous route below still has it. A `Result` rather
                        // than a `bool` because a moved buffer cannot be un-moved:
                        // the type is what makes "refused" and "still have the
                        // pixels" the same statement.
                        Err(returned) => bgra = returned,
                    }
                    note_type11_store_route("cpu_portability");
                    // `write_bgra8`, not `write_rgba8_image_changed`: the frame is
                    // already in guest scanout order, and that entry point would
                    // have to exchange every row back to read it. Both share the
                    // same tail — residency-window invalidation,
                    // `mark_mapping_written`, and the `surface_cache` republish —
                    // so this is a substitution and not a narrowing. Its
                    // changed-span rung is not lost either: this call site has
                    // always passed `None` for the seed.
                    let ok = mapping_write::write_bgra8(
                        state,
                        host,
                        c0.mapping_id,
                        &bgra,
                        c0.width.saturating_mul(RGBA8_BPP),
                        c0.width,
                        c0.height,
                    );
                    // The synchronous route publishes through
                    // `mark_mapping_written`, so the epoch it advanced to is
                    // simply the mapping's current one — read immediately, for
                    // the same reason the deferred route captures its own.
                    let sync_epoch = ok
                        .then(|| state.mappings.get(&c0.mapping_id))
                        .flatten()
                        .map(|m| m.surface_content_epoch);
                    if ok {
                        // Full-frame publish: same completeness proof as the
                        // import-present scatter paths — the write verified
                        // geometry (mw==w, mh==h) and landed the complete
                        // frame into the mapping's guest pages. Without it the
                        // `present_unbacked` gate is structurally dead on the
                        // CPU-portability Store path: no mapping's
                        // `dense_frame_seq` would ever advance.
                        {
                            let _span = StoreCostSpan::new("t11_publish_us");
                            publish_surface_store(
                                state,
                                host,
                                c0.mapping_id,
                                c0.width,
                                c0.height,
                                c0.format,
                            );
                        }
                        if let Some(epoch) = sync_epoch {
                            stamp_type11_resident(state, host, req, writeback_guest, epoch);
                        }
                        if crate::observe::draw_log_enabled() {
                            // Order-independent: both fields reduce over the three
                            // colour channels, so an R/B exchange cannot move them.
                            let (rgb_nz, max_rgb) = rgb_stats(&bgra);
                            crate::observe::line(format!(
                                "linux_m2v_store mid={} {}x{} pipe={} import=0 reason=cpu_portability pages=1 rgb_nz={} max={}",
                                c0.mapping_id,
                                c0.width,
                                c0.height,
                                req.pipeline_ref,
                                rgb_nz,
                                max_rgb
                            ));
                        }
                    } else {
                        let (rgb_nz, max_rgb) = rgb_stats(&bgra);
                        crate::observe::fail(format!(
                            "linux_m2v_store mid={} {}x{} pipe={} reason=cpu_portability_write_fail rgb_nz={} max={} fmt={:#x}",
                            c0.mapping_id,
                            c0.width,
                            c0.height,
                            req.pipeline_ref,
                            rgb_nz,
                            max_rgb,
                            c0.format
                        ));
                    }
                    if ok {
                        // `None`, for the same reason the deferred arm above
                        // returns it: this whole block runs only under
                        // `writeback_guest`, which `multi_draw_store_plan` grants
                        // solely to the **last** record of a packet, so there is no
                        // record N+1 for the chain value to seed. Returning it also
                        // could not be done honestly here — the frame is in guest
                        // scanout order and a chain seed is declared RGBA — so the
                        // alternative is a whole-frame exchange for a buffer with
                        // no reader.
                        return (EncodeStatus::Ok, None);
                    }
                    false
                }
            } else if c0.target_gva != 0 {
                supersede_gva_window(
                    state,
                    host,
                    c0.target_gva,
                    c0.width,
                    c0.height,
                    "sync_store",
                );
                // Bounded to the pages resolved before the GPU round trip
                // above. `None` only when that walk could not name the span,
                // which is the pre-existing behaviour for a target this device
                // cannot resolve at all.
                let gva_ok = write_gva_rgba8_within(
                    state,
                    host,
                    req.task_id,
                    c0.target_gva,
                    c0.width,
                    c0.height,
                    c0.row_stride,
                    c0.format,
                    &rgba,
                    sync_store_pages.as_ref(),
                )
                .is_ok();
                // Discrete-GPU rail: type-2/3 encode into **texture_ref** + **GVA**
                // host caches (not surface_id mid map — list ids collide with
                // present mids;). Sample prefers GVA key then
                // texture_ref with live descriptor geom.
                if gva_ok {
                    let producer_object_type =
                        objects::lookup_list_entry(state, host, req.task_id, c0.texture_ref)
                            .map(|entry| entry.object_type)
                            .unwrap_or(0);
                    host_cache_store_gva_layer(
                        state,
                        host,
                        req.task_id,
                        c0.texture_ref,
                        producer_object_type,
                        c0.target_gva,
                        c0.width,
                        c0.height,
                        &rgba,
                    );
                }
                let (rgb_nz, max_rgb) = rgb_stats(&rgba);
                // A Store that lands is expected control flow and belongs on
                // the census channel, not the failure one — "non-OFF lines are
                // the failures" is the rule the whole always-on log is triaged
                // by. Only the loss gets a failure line, and it carries the
                // census fields so nothing has to be correlated across two.
                crate::observe::off(format!(
                    "m2v_store_gva gva={:#x} {}x{} pipe={} tex_ref={} load={} ok={} rgb_nz={} max_rgb={} bpr={}",
                    c0.target_gva,
                    c0.width,
                    c0.height,
                    req.pipeline_ref,
                    c0.texture_ref,
                    c0.load_action,
                    gva_ok as u8,
                    rgb_nz,
                    max_rgb,
                    c0.row_stride
                ));
                if !gva_ok {
                    crate::observe::fail(format!(
                        "linux_m2v_store lost gva={:#x} {}x{} pipe={} tex_ref={} rgb_nz={} max={} bpr={}",
                        c0.target_gva,
                        c0.width,
                        c0.height,
                        req.pipeline_ref,
                        c0.texture_ref,
                        rgb_nz,
                        max_rgb,
                        c0.row_stride
                    ));
                }
                gva_ok
            } else {
                // No target to write: the frame is lost, so this one is a
                // failure line and there is no census twin to correlate with.
                let (rgb_nz, max_rgb) = rgb_stats(&rgba);
                crate::observe::fail(format!(
                    "linux_m2v_store no_target pipe={} tex_ref={} rgb_nz={} max={}",
                    req.pipeline_ref, c0.texture_ref, rgb_nz, max_rgb
                ));
                false
            };
            if ok {
                // `None`: everything from here up is under `writeback_guest`,
                // which `multi_draw_store_plan` grants only to `di == last_i`, so
                // the chain value has no record N+1 to seed and every other
                // reader of it in `exec.rs` sits inside the record loop that just
                // ended. The intermediate handoff that *is* live returned above,
                // before the Store arms. Returning the frame here handed a whole
                // framebuffer to a binding that is dropped unread.
                return (EncodeStatus::Ok, None);
            }
        }
    }

    if any_store {
        if req.vertex_count > 0 || req.indexed.is_some() {
            crate::observe::fail(format!(
                "linux_clear_store draws_skipped pipe={} vtx={} (m2v pending)",
                req.pipeline_ref, req.vertex_count
            ));
        }
        (EncodeStatus::Ok, color0_rgba)
    } else {
        (EncodeStatus::NoMetal("draw_vk_nothing_stored"), None)
    }
}

/// Sampled texture source + geometry for an engine draw.
pub(super) enum SampledSourceRequest {
    /// Shared texel bytes + optional producer identity (see
    /// [`LinearSampleIdentity`]) + the byte layout of those texels; the Arc lets
    /// memoized repeat binds skip the per-draw copy and the engine skip
    /// re-hashing.
    Bytes(
        std::sync::Arc<Vec<u8>>,
        Option<LinearSampleIdentity>,
        TexelLayout,
    ),
    Target(crate::backend::vulkan::engine::TargetIdentity),
    /// Zero-copy guest gather: the engine copies the texel bytes from
    /// imported guest RAM inside the draw CB — no CPU read, no memo, no
    /// hash. Carries the native texel layout the image is created with.
    /// Guest-RAM runs the engine gathers from, the byte layout of those texels,
    /// and — when both halves of the guest-write witness vouch for them — the
    /// identity that lets the engine bind a retained image instead of gathering
    /// at all (see [`crate::runtime::gather_witness`]).
    GuestRuns(
        crate::backend::vulkan::engine::GuestRunSource,
        TexelLayout,
        Option<LinearSampleIdentity>,
    ),
}

/// Producer identity + generation for CPU-sourced sampled bytes, so that equal
/// identity implies equal bytes under the same coherence model the producing
/// cache already relies on.
///
/// `key` is namespaced by its top two bits, because four producers share one
/// keyspace and a raw id would alias between them:
///
/// | bit 63 | bit 62 | producer | low bits |
/// |---|---|---|---|
/// | 0 | 0 | guest linear | the texture's authoritative GVA (`host_gva_surfaces`) |
/// | 1 | 0 | type-5 view | `plane_index << 32 \| mapping_id` |
/// | 0 | 1 | type-11 host cache | `mapping_id` (`host_surfaces`) |
/// | 1 | 1 | type-11 guest memo | `mapping_id` (`type11_memo`) |
///
/// GVAs are well under 2^62, so the unflagged row cannot collide with a flagged
/// one. `generation` comes from
/// [`crate::model::DeviceState::next_sampled_content_generation`] for every one
/// of them — a device-global counter, never per-entry — so a `(key, generation)`
/// pair names one content for the life of the device and content cannot alias
/// even if two producers did collide on a key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LinearSampleIdentity {
    pub(super) key: u64,
    pub(super) generation: u64,
}

impl From<crate::runtime::gather_witness::GatheredIdentity> for LinearSampleIdentity {
    /// The zero-copy gather rail's key is a hash of the window's name rather
    /// than a bit-namespaced id like the four rows above, so it can collide with
    /// any of them. That is harmless for the same reason the table's last
    /// paragraph gives: the generation comes from the one device-global counter,
    /// which issues a value once and never again, so a `(key, generation)` pair
    /// names one content even when two producers agree on a key.
    fn from(id: crate::runtime::gather_witness::GatheredIdentity) -> Self {
        Self {
            key: id.key,
            generation: id.generation,
        }
    }
}

type LoadedType5View = (
    u32,
    u32,
    std::sync::Arc<Vec<u8>>,
    LinearSampleIdentity,
    TexelLayout,
);
type LoadedLinearSample = (
    u32,
    u32,
    std::sync::Arc<Vec<u8>>,
    Option<LinearSampleIdentity>,
    TexelLayout,
);

/// Authoritative contents when a fragment texture aliases a GVA color target.
///
/// A serialized Metal render stream may read `color(0)` through texture slot 0
/// while several draws remain in one render pass. For GVA targets, records 2+
/// carry the prior draw in `target_seed_rgba`; reloading guest pages here would
/// expose the pre-pass image to the shader even though attachment Load sees the
/// chained image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum AttachmentAliasSample<'a> {
    Clear([f64; 4]),
    Seed(&'a [u8]),
    /// Records 2+ of a resident GVA chain: the prior record's content lives
    /// on the engine-resident target, not in a CPU seed. Bound as a resident
    /// sampled source (the engine snapshots on self-alias).
    ResidentChain,
}

pub(super) fn fragment_attachment_alias_sample<'a>(
    req: &'a DrawEncodeRequest,
    texture_index: u32,
    texture_ref: u32,
) -> Option<(u32, u32, AttachmentAliasSample<'a>)> {
    let color = req.colors.iter().find(|color| {
        color.slot == texture_index
            && color.texture_ref == texture_ref
            && color.mapping_id == 0
            && color.target_gva != 0
    })?;
    let need = (color.width as usize)
        .checked_mul(color.height as usize)?
        .checked_mul(RGBA8_BPP as usize)?;
    match color.load_action {
        PASS_LOAD_ACTION_CLEAR => Some((
            color.width,
            color.height,
            AttachmentAliasSample::Clear(color.clear_color),
        )),
        PASS_LOAD_ACTION_LOAD => {
            if let Some(seed) = color
                .target_seed_rgba
                .as_deref()
                .filter(|seed| seed.len() == need)
            {
                return Some((color.width, color.height, AttachmentAliasSample::Seed(seed)));
            }
            if req.chain_from_resident {
                return Some((
                    color.width,
                    color.height,
                    AttachmentAliasSample::ResidentChain,
                ));
            }
            None
        }
        _ => None,
    }
}

/// A deferred GVA window may serve a sampled bind directly from its resident
/// target only when the sampled view is the exact window content: descriptor
/// geometry equals the window geometry, and the same storage-family gate that
/// would let the post-flush cache layer serve this object type accepts it. Any
/// mismatch must land the window (flush path) instead.
pub(super) fn deferred_gva_sample_eligible(
    win: &crate::model::GvaDeferredEntry,
    desc_width: u32,
    desc_height: u32,
    sampler_object_type: u8,
) -> bool {
    win.width == desc_width
        && win.height == desc_height
        && gva_cache_owner_allows_object_type(win.producer_object_type, sampler_object_type)
}

/// Bind a still-deferred GVA render Store's resident target for a type-2/3
/// sampled bind instead of flushing it to guest memory and re-uploading.
///
/// Value-equal to the flush path: a flush lands the resident RGBA into the
/// `host_gva_surfaces` layer, and the sample would serve that layer back
/// (BGRA→RGBA swap) — so binding the resident directly yields the same texels
/// whenever the descriptor geometry matches the window and the cache owner
/// gate would accept the layer. Mismatches fall through to the flush path.
/// The window stays armed: the contract Store still lands on first guest
/// access, and the resident stays authoritative for further samples.
fn try_sample_deferred_gva<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
) -> Option<(u32, u32, u32, SampledSourceRequest)> {
    use crate::backend::vulkan::engine;
    if state.gva_deferred_flush.is_empty() {
        return None;
    }
    let entry = objects::lookup_list_entry(state, host, task_id, texture_ref)?;
    let desc_bytes = objects::read_descriptor(state, host, task_id, &entry)?;
    let tex = decode_texture_descriptor(&desc_bytes).ok()?;
    let (gva, layout) = tex.level_gva(0, state.page_shift)?;
    // Every rung below this point declines to a bare `None`, and the caller then
    // reads guest pages — so which of four different situations produced the
    // fall-through is invisible at the call site, and the four are not equally
    // safe. A window that is armed but whose resident is not ready is the one
    // that matters: the window's existence is itself the statement that guest
    // pages are stale. Named per (gva, geometry, reason), transition-keyed.
    let decline = |reason: &str, win_geom: (u32, u32)| {
        let mut subject = std::hash::DefaultHasher::new();
        std::hash::Hash::hash(&(gva, layout.width, layout.height), &mut subject);
        let mut st = std::hash::DefaultHasher::new();
        std::hash::Hash::hash(&(reason, win_geom), &mut st);
        if crate::observe::state_changed(
            "gva_sample_rung",
            std::hash::Hasher::finish(&subject),
            std::hash::Hasher::finish(&st),
        ) {
            crate::observe::off(format!(
                "gva_sample_rung reason={reason} task={task_id} ref={texture_ref} gva={gva:#x} desc={}x{} win={}x{}",
                layout.width, layout.height, win_geom.0, win_geom.1,
            ));
        }
        None::<(u32, u32, u32, SampledSourceRequest)>
    };
    let Some(win) = state.gva_deferred_flush.get(&gva) else {
        return decline("no_window", (0, 0));
    };
    let win_geom = (win.width, win.height);
    if !deferred_gva_sample_eligible(win, layout.width, layout.height, entry.object_type) {
        let reason = if win.width != layout.width || win.height != layout.height {
            "window_geometry"
        } else {
            "owner_object_type"
        };
        return decline(reason, win_geom);
    }
    // From the window, not from a walk: this bind must reach the slot the window
    // pinned, and the address may already belong to something else.
    let id = crate::runtime::storage_flush::gva_window_identity(gva, win);
    if !engine::resident_content_ready(&id) {
        // The window says guest pages are stale and the resident says it has
        // nothing — the two together mean this sample has no correct source.
        return decline("resident_not_ready", win_geom);
    }
    Some((win_geom.0, win_geom.1, 0, SampledSourceRequest::Target(id)))
}

/// Resolve a sampled texture ref to `(width, height, mapping_id, source)`.
///
/// Backend-neutral: the returned [`SampledSourceRequest`] is either an engine
/// target to bind directly (zero-copy) or CPU bytes to upload, so this is the
/// resolver the engine draw path uses. Distinct from [`load_sampled_rgba`],
/// which is the Metal-path resolver and always materializes RGBA8 bytes.
///
/// # The type-11 ladder is measured, and every rung carries load
///
/// Four rungs offer the same type-11 surface, and the obvious reading is that
/// three of them are redundant with the first. They are not. A DRIVEN x86/Vulkan
/// session — four Safari page loads, each scrolled six pages and then dragged by
/// its title bar — split as:
///
///   t11rung_resident         31 916   93.0 %   engine image, taken zero-copy
///   t11rung_host_cache        1 694    4.9 %   surface_cache's BGRA mirror
///   t11rung_zero_copy           705    2.1 %   guest pages, gathered
///   t11rung_guest_memo          150    0.4 %   guest pages, CPU convert
///   t11rung_miss                  0             no source at all
///   t11rung_resident_refused        2            guest overwrote the resident
///
/// Measure this on a DRIVEN session or not at all. The same census on an
/// undriven boot to the desktop reported 12 / 5 / 8 / 0, which is far too quiet
/// to tell a rung that never fires from one the boot never reached — and quiet
/// enough to talk someone into deleting it.
///
/// A second driven session with a different drive — Chess, Maps, Safari on the
/// WebGL aquarium, Wikipedia and apple.com, page scrolls and two title-bar
/// drags — reproduced the shape on a smaller population:
///
///   t11rung_resident         15 992   64.5 %
///   t11rung_host_cache        5 777   23.3 %
///   t11rung_zero_copy         3 036   12.2 %
///   t11rung_guest_memo           62    0.25 %
///   t11rung_miss                  0
///   t11rung_resident_refused      0
///
/// The order is the same and no rung is empty, so the two runs agree on which
/// rungs carry load. The share does move with the drive — live 3D and a WebGL
/// canvas push work down off the resident rung — so treat the percentages as a
/// range and not as a constant of the design. The bottom rung is the one to
/// keep watching: 150 binds in one session and 62 in the other is small enough
/// to read as noise, and both are a fallback nothing below would correct.
///
/// Two facts to weigh before touching the order:
///
/// - The host-cache rung is NOT a duplicate of the guest-page rungs below it.
///   A render Store defers its writeback into guest pages, so between the Store
///   and its flush the cache is the only host-side copy that holds the new
///   pixels; the pages still hold the old ones. Its 1 694 binds are that window.
/// - `t11rung_resident_refused` firing twice is not evidence the guest-write
///   witness is dead weight. Those are the binds where the guest CPU painted
///   over a surface the engine still claimed to hold, and the rung sits above
///   both page-reading rungs, so nothing below would have corrected it. Two
///   uncorrected stale binds on a repainted surface is the "renders correctly
///   for a few frames, then stays corrupted" report.
pub(super) fn resolve_sampled_source<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    entry: Option<ListObjectEntry>,
) -> Option<(u32, u32, u32, SampledSourceRequest)> {
    if texture_ref == 0 {
        return None;
    }

    // Opcode-9 buffer-backed texture (type-8): the sampled bytes are an MTLBuffer's
    // guest storage, not a view over another texture. Resolve it directly before
    // the view/surface paths (which would mis-decode the opcode-9 descriptor).
    // `entry` (when supplied by the caller) reuses the object-list read this call
    // and its buffer-texture / view classification below would otherwise each
    // repeat — the guest object list is immutable for the draw.
    if let Some(bt) = buffer_texture_descriptor(state, host, task_id, texture_ref, entry) {
        let (w, h, rgba) = load_buffer_texture_rgba(state, host, task_id, texture_ref, &bt)?;
        return Some((
            w,
            h,
            0,
            SampledSourceRequest::Bytes(std::sync::Arc::new(rgba), None, TexelLayout::Rgba8),
        ));
    }

    // The object list names exactly ONE surface for a sampled ref. Which one is
    // decided by the entry's `object_type`, and the cases below are distinct
    // values of that single u8 field, so they cannot both apply:
    //
    //   type-5 RefTextureHandle — carries the type-4 surface id in its
    //                             descriptor, alongside the Metal texture view
    //   type-4 Surface         — the ref *is* the surface id
    //   type-11 IOSurface      — resolves to the mapping id it was created on
    //
    // `resolve_type11_ref` re-reads this same entry and returns `None` for every
    // type but 11, so it can only ever fill a slot the classification above left
    // empty. Hence an `Option`, not a list of candidates to choose between.
    let mut is_linear_tex = false;
    let mut is_type5 = false;
    let mut type5_view: Option<objects::Type5TextureView> = None;
    let mut surface: Option<u32> = None;
    let resolved_entry =
        entry.or_else(|| objects::lookup_list_entry(state, host, task_id, texture_ref));
    if let Some(entry) = resolved_entry {
        if entry.object_type == objects::OBJECT_TYPE_REF_TEXTURE {
            is_type5 = true;
            if let Some(desc) = objects::read_descriptor(state, host, task_id, &entry) {
                if desc.len() >= objects::TYPE5_MIN_LEN {
                    let sid = crate::contract::endian::ld32(&desc[objects::TYPE5_SURFACE_ID..]);
                    if sid != 0 {
                        type5_view = objects::decode_type5_texture_view(&desc);
                        surface = Some(sid);
                    }
                }
            }
        }
        if entry.object_type == objects::OBJECT_TYPE_SURFACE {
            surface = Some(texture_ref);
        }
        if entry.object_type == OBJECT_TYPE_TEXTURE
            || entry.object_type == OBJECT_TYPE_TEXTURE_VARIANT
        {
            is_linear_tex = true;
        }
    }
    if !is_type5 {
        // Runs for the linear and unclassified types too, not only for the
        // type-11 hit it can return: the resolve records the ref as live in the
        // task's object set and reports a typed failure when the descriptor is
        // unreadable. Both are wanted for any ref a draw sampled.
        surface = surface.or(objects::resolve_type11_ref(
            state,
            host,
            task_id,
            texture_ref,
        ));
    }

    if let Some(mid) = surface {
        // Ensure type-4 pages exist for this surface id.
        let _ = objects::ensure_surface_for_present(state, host, mid);
        // A type-5 serialized record is the exact Metal texture view over the
        // IOSurface bytes. Materialize it only when it differs from (or cannot
        // be inferred from) the base mapping. Exact base views keep the fast
        // resident/cache path below; an unknown 2-B/texel base FourCC exposed
        // as RG8 must instead use the serialized view's native interpretation.
        // `type5_view` is set only on the branch that also set `surface` to that
        // view's own surface id, so reaching here with a view in hand already
        // means `mid` is the surface it describes.
        if let Some(view) = type5_view {
            let needs_materialization = state
                .mappings
                .get(&mid)
                .map(|m| {
                    type5_view_requires_materialization(
                        m.has_geom, m.width, m.height, m.format, view,
                    )
                })
                .unwrap_or(true);
            if needs_materialization {
                // Zero-copy the decoded plane straight from guest pages when
                // it samples byte-identically (video NV12 R8/RG8, BGRA8/
                // RGBA8). This bypasses the ~1.5 MB/plane/frame CPU read +
                // upload the CPU loader below would pay every decoded frame.
                if let Some(src) = try_type5_sample_zero_copy(state, host, mid, view) {
                    // Success path: a healthy video decodes ~2 planes/frame,
                    // so this fires per-bind (~99k lines/boot). The aggregate
                    // lives in `sampled_branch_census` (`t5_zc=count:bytes`),
                    // which is the always-on signal; keep the per-bind detail
                    // for deep debugging behind REIMS_VGPU_DRAW_LOG (observe::line)
                    // rather than flooding the always-on fail sink.
                    crate::observe::line(format!(
                        "type5_view_zc ref={texture_ref} sid={mid} view={}x{} fmt={:#x} plane={}",
                        view.width, view.height, view.pixel_format, view.plane_index
                    ));
                    return Some((view.width, view.height, mid, src));
                }
                let (w, h, rgba, identity, byte_format) =
                    load_type5_view_rgba(state, host, task_id, texture_ref, mid, view)?;
                return Some((
                    w,
                    h,
                    mid,
                    SampledSourceRequest::Bytes(rgba, Some(identity), byte_format),
                ));
            }
        }
        if let Some(m) = state.mappings.get(&mid) {
            if m.has_geom && m.width > 0 && m.height > 0 {
                let (w, h) = (m.width, m.height);
                // Attribute the resident-readiness/bind sub-slice of the resolve so
                // the census can separate engine-lock cost (this block) from the
                // object-list decode prelude. This block acquires the global engine
                // lock (`resident_content_ready`), the suspected dock-hover-freeze
                // contention site.
                // Resident-surface identity: computed once and reused for both the
                // readiness check and the direct bind. `surface_identity` locks a
                // global dedup mutex and does an output-group lookup; this bind
                // resolves the same (mid, w, h), so recomputing it per resident
                // sample (the census shows ~29k/session) is pure waste.
                let resident_id =
                    crate::runtime::present_identity::surface_identity(state, mid, w, h);
                // `content_ready` only. The obvious strengthening — also require
                // the resident's `content_epoch` to match the mapping's, as the
                // attachment LOAD elision does — was tried and reverted, because
                // `content_epoch` is not free for this rung to reinterpret. The
                // deferred flush uses the same field as its own identity check
                // ("is this still the resident my window was armed on"), so
                // withholding the stamp to disqualify a resident *here* made
                // every later window on that resident report
                // `resident_epoch_drift` and drop its frame: `deferred_flush_lost`
                // went from 17 to 3 161 on one boot and the screen stayed black.
                //
                // Separating the two meanings is worth doing and is not this
                // change. What guards this rung today is the guest-write witness
                // below, which is the witness the LOAD elision's epoch pair
                // cannot supply anyway.
                let resident_ready =
                    crate::backend::vulkan::engine::resident_content_ready(&resident_id);

                // What the hypervisor can say about the guest's own stores into
                // this surface since the Store that produced our copies of it.
                // Asked once here and used by every rung below, because every
                // rung below either serves a host-side copy — whose currency
                // this is the only witness for — or reads the pages and does
                // not care. The host-cache rung used to ask it privately; the
                // resident rung above it asked nothing at all.
                //
                // Two stages: the token's generation says whether the guest
                // wrote the *allocation*, and only when it did is the page list
                // enumerated to say whether it wrote the *pixels*. The second
                // stage costs a page-list walk and is paid on the minority of
                // binds the first stage flags.
                let guest_write = mapping_guest_write_verdict(state, host, mid);
                let site = guest_wrote_allocation(guest_write)
                    .then(|| guest_write_site(state, host, mid, w, h));
                if let Some(site) = site.as_ref() {
                    crate::runtime::drain::note_store_route(match site {
                        GuestWriteSite::Pixels(_) => "t11sample_gw_wrote_pixels",
                        GuestWriteSite::Elsewhere => "t11sample_gw_wrote_elsewhere",
                        GuestWriteSite::Unknown => "t11sample_gw_wrote_unknown",
                    });
                }
                let guest_owned = match site.as_ref() {
                    Some(GuestWriteSite::Pixels(ranges)) => Some(ranges.as_slice()),
                    _ => None,
                };
                let guest_replaced = !matches!(site, None | Some(GuestWriteSite::Elsewhere));

                // A ready resident target is authoritative after a product
                // Store — but only while nothing has replaced the bytes it is a
                // copy of. A type-11 surface's pages are plain guest RAM and the
                // guest CPU stores into them with no device operation, so a
                // resident produced for one tenant of the surface keeps claiming
                // to hold its pixels after the guest has painted different ones
                // there. This is the same question `type11_guest_wrote_since_store`
                // asks before the attachment LOAD elision reuses a resident, and
                // the same one the host-cache rung below asks before serving its
                // copy. This rung asked neither, and it is the largest of the
                // three: one 14-round Finder boot measured 92 730 binds here
                // against the cache rung's 14 396.
                //
                // It also sits *above* both rungs that read the guest's own
                // pages, so a stale bind here is never corrected by anything
                // below it — the wrong image is held for as long as the guest
                // keeps re-binding the same surface, which is what "renders
                // correctly for a few frames then stays corrupted" looks like
                // from inside the guest.
                //
                // Refusing is not enough on its own, because neither copy is the
                // surface once both sides have written it. The refusal merges
                // them into the guest's pages first — see
                // [`merge_guest_writes_into_pages`] — and every rung below then
                // reads a surface that holds both halves.
                if resident_ready {
                    if !guest_replaced {
                        note_type11_sample_rung("t11rung_resident", guest_write);
                        return Some((w, h, mid, SampledSourceRequest::Target(resident_id)));
                    }
                    note_type11_sample_rung("t11rung_resident_refused", guest_write);
                    match guest_owned {
                        Some(ranges) => {
                            merge_guest_writes_into_pages(
                                state,
                                host,
                                mid,
                                w,
                                h,
                                &resident_id,
                                ranges,
                            );
                        }
                        // `Unknown`: the host named no pages, so there is no
                        // list to preserve and no merge to make. The rungs below
                        // read whatever the guest's pages hold, which is the only
                        // source not known to be stale.
                        None => crate::runtime::drain::note_store_route(
                            "t11sample_resident_unmergeable",
                        ),
                    }
                    // A resident that stops matching its surface is a real
                    // correction, not routine control flow: without this line a
                    // boot cannot tell "the guest never rewrites its sampled
                    // surfaces" from "the witness is never armed". Latched per
                    // mapping so a surface the guest repaints every frame stays
                    // at one line.
                    if crate::observe::first_sight("sampled_resident_stale", u64::from(mid)) {
                        crate::observe::off(format!(
                            "sampled_resident_stale mid={mid} {w}x{h} \
                             (guest wrote pages inside the sampled window; reading them instead)"
                        ));
                    }
                }

                // 1) Host cache — the other host-side copy of these pages, and
                // so gated on exactly the same witness as the resident above.
                // It sits above both rungs that read the guest's own pages, so a
                // stale hit is never corrected by anything below it; falling
                // through costs a re-read and reaches content that is
                // authoritative by construction.
                //
                // No content scan gates this. What stood here counted non-black
                // pixels (2 073 600 per bind at 1080p) and let the count decide
                // which image got bound — `runtime/census/README.md` forbids
                // exactly that, and an all-black frame is a legal frame, so the
                // test mistook a correct black surface for an empty one.
                if let Some((bgra, host_gen)) = (!guest_replaced)
                    .then(|| crate::runtime::surface_cache::get_shared_with_gen(state, mid, w, h))
                    .flatten()
                {
                    // Uploaded in the order the cache already holds. This rung's
                    // bytes are BGRA8 by construction — it is a type-4 scanout
                    // cache — and `B8G8R8A8_UNORM` is a Vulkan-mandatory sampled
                    // format with linear filtering, so declaring the layout costs
                    // nothing and the hardware reads the channels the guest
                    // stored. What stood here rebuilt the whole frame into RGBA8
                    // first: a 1.7 MB allocation plus a full read+write pass on
                    // every bind, ~116 binds a second live, to reach bytes the
                    // sampler could already address. The linear rail reached the
                    // same conclusion (`linear_native_upload_format(.., true)`).
                    //
                    // The view swizzle applied at bind is a *logical* channel
                    // remap from the guest descriptor and composes with the
                    // physical format rather than substituting for it, so this
                    // does not double-swap.
                    //
                    // The generation is the cache entry's own `host_gen`, which
                    // every writer of `host_surfaces` re-takes from
                    // `next_sampled_content_generation` in the same breath as it
                    // changes the bytes — so an unchanged pair is a statement
                    // that the frame has not moved, and the engine can bind what
                    // it already holds instead of re-digesting 1.7 MB to find
                    // out. See [`LinearSampleIdentity`] for the key namespace.
                    //
                    // A 0 generation is an entry never stored into.
                    // `get_shared_with_gen` already refuses those — it requires
                    // bytes — but a false "unchanged" is the one wrong answer
                    // here that binds a stale frame, so it is not left to that
                    // alone.
                    let identity = (host_gen != 0).then_some(LinearSampleIdentity {
                        key: (1u64 << 62) | mid as u64,
                        generation: host_gen,
                    });
                    note_type11_sample_rung("t11rung_host_cache", guest_write);
                    crate::runtime::storage_flush::note_render_flush_cache_read(state, mid);
                    return Some((
                        w,
                        h,
                        mid,
                        SampledSourceRequest::Bytes(bgra, identity, TexelLayout::Bgra8),
                    ));
                }

                // 2) Guest pages, which are what the surface *is*. Reached only
                // when no host-side copy served the bind — no resident, or one
                // the guest has written over — so the gather always runs and the
                // guest bytes are taken unconditionally. Declining the gather is
                // expected control flow — the CPU byte loader below serves the
                // same pixels — so it stays quiet, like the type-2/3 rail's.
                if let Some(src) = try_type11_sample_zero_copy(state, host, mid, w, h) {
                    note_type11_sample_rung("t11rung_zero_copy", guest_write);
                    crate::runtime::storage_flush::note_render_flush_pages_read(state, mid);
                    return Some((w, h, mid, src));
                }
                // The memo skips the convert/alloc on unchanged content and
                // returns a content identity so the engine skips re-hash+upload;
                // its census (T11Memo hit / T11Guest fill) is emitted internally.
                if let Some((rgba, identity)) = load_type11_rgba_memoized(state, host, mid) {
                    note_type11_sample_rung("t11rung_guest_memo", guest_write);
                    crate::runtime::storage_flush::note_render_flush_pages_read(state, mid);
                    return Some((
                        w,
                        h,
                        mid,
                        SampledSourceRequest::Bytes(rgba, Some(identity), TexelLayout::Rgba8),
                    ));
                }

                {
                    // A sample that resolved to no bytes anywhere is a lost
                    // guest command at any geometry: an app-window layer paints
                    // blank exactly as a full-screen one does. Latched per
                    // (mid, geometry) so a steady repeat stays at one line.
                    use std::collections::HashSet;
                    use std::sync::Mutex;
                    note_type11_sample_rung("t11rung_miss", guest_write);
                    static SEEN: Mutex<Option<HashSet<(u32, u32, u32)>>> = Mutex::new(None);
                    let mut guard = SEEN.lock().unwrap_or_else(|e| e.into_inner());
                    if guard.get_or_insert_with(HashSet::new).insert((mid, w, h)) {
                        crate::observe::fail(format!(
                            "sample_src=miss ref={texture_ref} mid={mid} {w}x{h} resident_ready={} (no guest/cache/resident bytes)",
                            resident_ready as u8
                        ));
                    }
                }
            }
        }
    }

    // Type-2/3: GVA-keyed encode, then texture_ref with **descriptor** geom match.
    if is_linear_tex {
        // A still-deferred GVA render Store is GPU-resident and authoritative;
        // bind the resident target directly instead of flushing + re-uploading
        // (the gvadefer A/B showed 99% of windows were consumed by exactly this
        // sample path — readback relocation, not elimination).
        if let Some(v) = try_sample_deferred_gva(state, host, task_id, texture_ref) {
            return Some(v);
        }
        // Zero-copy gather for large Vulkan-native linear textures: replaces
        // the CPU host-cache/memo byte paths below for eligible formats (the
        // lin_memo full-window re-read + memcmp per bind was the dominant
        // per-draw cost under compositor load).
        // Resolve + decode the texture descriptor ONCE for both linear loaders
        // below. The zero-copy attempt (which returns None on the ~35k/session
        // cache-fallback majority) and the host-cache fallback each read the same
        // descriptor blob and run the identical `decode_texture_descriptor`; the
        // object list is immutable for the draw, so one read+decode serves both.
        // Both readers take only the decoded descriptor.
        if let Some(tex) = resolved_entry.and_then(|e| {
            objects::read_descriptor(state, host, task_id, &e)
                .and_then(|d| decode_texture_descriptor(&d).ok())
        }) {
            if let Some((w, h, src)) =
                try_linear_sample_zero_copy(state, host, task_id, texture_ref, &tex)
            {
                return Some((w, h, 0, src));
            }
            if let Some((w, h, rgba, identity, byte_format)) =
                load_linear_from_host_caches(state, host, task_id, texture_ref, &tex)
            {
                return Some((
                    w,
                    h,
                    0,
                    SampledSourceRequest::Bytes(rgba, identity, byte_format),
                ));
            }
        }
    }

    // Linear / view path returns only RGBA; the geometry comes from the decoded
    // texture descriptor and from nowhere else. A payload shorter than the
    // descriptor's own `width * height * 4` is not a geometry this call may
    // invent one for: the caller turns `None` into a typed
    // `DrawPreparationDecline::TextureResolveMissing`, which names the ref and
    // the stage.
    let mut rgba = load_sampled_rgba_static(state, host, task_id, texture_ref)?;
    let entry = objects::lookup_list_entry(state, host, task_id, texture_ref)?;
    let desc = objects::read_descriptor(state, host, task_id, &entry)?;
    let td = decode_texture_descriptor(&desc).ok()?;
    let w = td.width.max(1);
    let h = td.height.max(1);
    let need = (w as usize).saturating_mul(h as usize).saturating_mul(4);
    if rgba.len() < need {
        return None;
    }
    rgba.truncate(need);
    Some((
        w,
        h,
        0,
        SampledSourceRequest::Bytes(std::sync::Arc::new(rgba), None, TexelLayout::Rgba8),
    ))
}

#[inline]
pub(super) fn type5_view_requires_materialization(
    base_has_geom: bool,
    base_width: u32,
    base_height: u32,
    base_format: u16,
    view: objects::Type5TextureView,
) -> bool {
    !base_has_geom
        || view.depth != 1
        || base_format == 0
        || base_width != view.width
        || base_height != view.height
        || base_format != view.pixel_format
}

/// The decoded device-surface fields a failed sample-window derivation dumps
/// for diagnosis: `(width, height, pixel_format, bytes_per_row, alloc_size)`.
type SampleWindowDesc = (u32, u32, u32, u32, u32);

/// Why the type-5 serialized-view loader refused to materialize a plane.
///
/// # Why these slugs are prefixed `type5_view_`
///
/// The blit rail's `BlitStatus` already owns a `t5_*` vocabulary for the type-5
/// *copy* path (`t5_no_mapping`, `t5_sample_window`, `t5_fmt_bpp`,
/// `t5_unmapped`), and four of this loader's checks are conceptually the same
/// words. A bare `no_mapping` was in fact one of three claimants — console
/// capture, guest-page import and this loader — that the last present-rail
/// migration recorded as still sharing the word. The `type5_view_` prefix keeps
/// `grep reason=type5_view_…` answerable against the copy path that shares the
/// surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Type5ViewDecline {
    /// The serialized view is volumetric; only `depth == 1` planes materialize.
    UnsupportedDepth { depth: u32 },
    /// The mapping's page table is not resident for scanout.
    Unresolved,
    /// The view's MTLPixelFormat has no known bytes-per-pixel.
    FormatBpp,
    /// The mapping id has no live entry.
    NoMapping,
    /// No sample window could be derived from the device descriptor for this
    /// plane geometry. Carries the base geometry and the decoded descriptor (or
    /// its absence) that disagreed.
    SampleWindow {
        base_w: u32,
        base_h: u32,
        base_fmt: u16,
        desc: Option<SampleWindowDesc>,
    },
    /// The mapping's resident pages span fewer bytes than the sample window
    /// ends at.
    Span {
        pages: usize,
        page_bytes: u64,
        span_end: u64,
        bpr: u32,
    },
    /// `width * bpp` overflowed a u32, so a tight row is unrepresentable.
    TightOverflow { bpp: u32 },
    /// The native plane byte length overflowed the host allocation cap.
    NativeLen { tight: u32 },
    /// The native plane window could not be read from guest memory.
    Read {
        base_w: u32,
        base_h: u32,
        base_fmt: u16,
        off: u64,
        bpr: u32,
        span_end: u64,
        pages: usize,
    },
    /// `width * 4` overflowed a u32, so the RGBA row is unrepresentable.
    RgbaStride,
    /// The RGBA buffer length overflowed the host allocation cap.
    RgbaLen { stride: u32 },
    /// A row failed to convert from the native format into RGBA8.
    Convert { row: usize, bpp: u32 },
}

impl crate::observe::Decline for Type5ViewDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::UnsupportedDepth { .. } => "type5_view_unsupported_depth",
            Self::Unresolved => "type5_view_unresolved",
            Self::FormatBpp => "type5_view_format_bpp",
            Self::NoMapping => "type5_view_no_mapping",
            Self::SampleWindow { .. } => "type5_view_sample_window",
            Self::Span { .. } => "type5_view_span",
            Self::TightOverflow { .. } => "type5_view_tight_overflow",
            Self::NativeLen { .. } => "type5_view_native_len",
            Self::Read { .. } => "type5_view_read",
            Self::RgbaStride => "type5_view_rgba_stride",
            Self::RgbaLen { .. } => "type5_view_rgba_len",
            Self::Convert { .. } => "type5_view_convert",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::UnsupportedDepth { depth } => vec![("depth", depth.to_string())],
            Self::SampleWindow {
                base_w,
                base_h,
                base_fmt,
                desc,
            } => {
                let mut v = vec![
                    ("base", format!("{base_w}x{base_h}")),
                    ("base_fmt", format!("{base_fmt:#x}")),
                ];
                match desc {
                    Some((w, h, fmt, bpr, alloc)) => {
                        v.push(("desc", format!("{w}x{h}")));
                        v.push(("desc_fmt", format!("{fmt:#x}")));
                        v.push(("bpr", bpr.to_string()));
                        v.push(("alloc", alloc.to_string()));
                    }
                    None => v.push(("desc", "missing".to_string())),
                }
                v
            }
            Self::Span {
                pages,
                page_bytes,
                span_end,
                bpr,
            } => vec![
                ("pages", pages.to_string()),
                ("page_bytes", page_bytes.to_string()),
                ("span_end", span_end.to_string()),
                ("bpr", bpr.to_string()),
            ],
            Self::TightOverflow { bpp } => vec![("bpp", bpp.to_string())],
            Self::NativeLen { tight } => vec![("tight", tight.to_string())],
            Self::Read {
                base_w,
                base_h,
                base_fmt,
                off,
                bpr,
                span_end,
                pages,
            } => vec![
                ("base", format!("{base_w}x{base_h}")),
                ("base_fmt", format!("{base_fmt:#x}")),
                ("off", off.to_string()),
                ("bpr", bpr.to_string()),
                ("span_end", span_end.to_string()),
                ("pages", pages.to_string()),
            ],
            Self::RgbaLen { stride } => vec![("stride", stride.to_string())],
            Self::Convert { row, bpp } => {
                vec![("row", row.to_string()), ("bpp", bpp.to_string())]
            }
            Self::Unresolved | Self::FormatBpp | Self::NoMapping | Self::RgbaStride => Vec::new(),
        }
    }
}

/// Why a type-11 attachment `LOAD` could not be seeded with the surface's own
/// prior contents.
///
/// This is not a degradation the caller absorbs. `exec.rs` resolves the pass load
/// action as "explicit `load_op` > `target_rgba8` > **Clear**", so a seed of
/// `None` makes `PassKey::single(load = false)` and the render pass begins with
/// `LoadOp::CLEAR` against the hardcoded `[0,0,0,0]` primary clear value. The
/// guest asked for its surface to be preserved and got a transparent-black wipe,
/// and the matching Store then reads that wipe back and publishes it. On a
/// compositor doing a damage-rect redraw that is one whole layer rendering solid
/// black — the reported black-rectangle class, whose screenshots show sharp
/// axis-aligned rectangles at layer boundaries.
///
/// It had no report of any kind. `surface_cache::get_shared` returns `Option` and
/// the arm simply left `target_rgba8` unset, so the loss was invisible on the
/// always-on channel. Measured on one x86/Vulkan boot before the guest-pages rung
/// existed: **121 distinct (mapping, geometry) wipes** in ~170 s, four of them at
/// the full 1920x1080 composite extent, against 0 in the idle phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Type11SeedDecline {
    /// The cache holds no entry for this mapping id and the mapping's own pages
    /// could not be read at the requested extent either.
    ///
    /// This is the whole population the pre-fix boot measured: every one of the
    /// 121 lines carried `hostgen=0`, and every one had `want == mapgeom`, which
    /// is what said the guest pages were readable and made the fallback rung the
    /// fix rather than a guess.
    NoEntry,
    /// An entry exists but at a different geometry, so the exact-geometry hit
    /// rule refuses it. `host_surfaces` keeps exactly one entry per mapping and
    /// every Store replaces it, so a Store at another geometry orphans every
    /// window still living at this one.
    ///
    /// Fired **0** times on that boot. Kept because it is a different check with
    /// a different fix (the entry is stale, not missing), and folding it into
    /// `NoEntry` would hide which one a future boot hit.
    GeomMismatch { have_w: u32, have_h: u32 },
    /// An entry exists at exactly this geometry but its bytes were ceded to the
    /// pinned resident of a deferred type-11 Store that skipped its readback
    /// (`surface_cache::cede_surface_to_resident`).
    ///
    /// Distinct from [`Self::GeomMismatch`] because the geometries *match* — folding
    /// it there would print `have=1920x1080` against `want=1920x1080` and read as
    /// a contradiction — and distinct from [`Self::NoEntry`] because nothing
    /// is missing: the frame is on the GPU, and the guest-pages rung below lands
    /// this very window on its way to reading them, so the seed is served with
    /// the Store's own pixels. Expected, and worth naming so its rate can be read
    /// against the elision that is supposed to make it rare.
    CededToResident,
}

impl crate::observe::Decline for Type11SeedDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::NoEntry => "type11_seed_cache_absent",
            Self::GeomMismatch { .. } => "type11_seed_cache_geom",
            Self::CededToResident => "type11_seed_cache_ceded",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::NoEntry | Self::CededToResident => Vec::new(),
            Self::GeomMismatch { have_w, have_h } => vec![("have", format!("{have_w}x{have_h}"))],
        }
    }
}

/// Which rung of the type-11 `LOAD` seed ladder produced the attachment's prior
/// contents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Type11SeedRung {
    /// The host render cache held this mapping at exactly this geometry.
    Cache,
    /// The cache missed and the surface's own guest IOSurface pages were read.
    GuestPages,
}

impl Type11SeedRung {
    fn name(self) -> &'static str {
        match self {
            Self::Cache => "cache_hit",
            Self::GuestPages => "guest_pages",
        }
    }
}

/// Report which way the type-11 `LOAD` seed branch went, once per
/// `(mapping, requested geometry, outcome)`.
///
/// Every outcome reports, because a zero on the miss arm has to be readable. A
/// probe that only fires on failure cannot separate "the cache always hit" from
/// "this branch never ran", and the branch is reached only for `mapping_id != 0`
/// under `PASS_LOAD_ACTION_LOAD` with no caller-supplied seed. With the served
/// arms beside it, an absent miss line next to present hit lines is evidence
/// rather than silence.
///
/// Naming the *rung* rather than just hit/miss is what prices the fallback: a
/// `guest_pages` line is a cache miss that was recovered, and its rate is the
/// only thing that says whether the recovery is cheap. Fusing it into `cache_hit`
/// would make the fix unmeasurable the moment it worked.
///
/// The mapping's own latched geometry and generation ride along on every arm:
/// `want == mapgeom` is the condition under which the guest-pages rung can serve
/// at all, so the pair says whether a miss was recoverable.
fn note_type11_load_seed(
    state: &DeviceState,
    mapping_id: u32,
    w: u32,
    h: u32,
    served: Option<Type11SeedRung>,
) {
    let (map_w, map_h, map_gen) = state
        .mappings
        .get(&mapping_id)
        .map(|m| (m.width, m.height, m.map_generation))
        .unwrap_or((0, 0, 0));
    let cached = state.host_surfaces.get(&mapping_id);
    let have = cached.map(|e| (e.width, e.height));
    let host_gen = cached.map(|e| e.host_gen).unwrap_or(0);
    // Latch before building the line: `Emit::field` renders eagerly, and this
    // sits on a branch the census measures at 28-111 entries a second.
    let outcome_bits = match served {
        None => 0u64,
        Some(Type11SeedRung::Cache) => 1,
        Some(Type11SeedRung::GuestPages) => 2,
    };
    let disc =
        (u64::from(mapping_id) << 40) | (u64::from(w) << 20) | u64::from(h) | (outcome_bits << 62);
    if let Some(rung) = served {
        if !crate::observe::first_sight("type11_load_seed_served", disc) {
            return;
        }
        crate::observe::off(format!(
            "type11_load_seed outcome={} mid={mapping_id} want={w}x{h} \
             mapgeom={map_w}x{map_h} mapgen={map_gen} hostgen={host_gen}",
            rung.name()
        ));
        return;
    }
    let d = match have {
        Some(_)
            if crate::runtime::surface_cache::surface_ceded_to_resident(
                state, mapping_id, w, h,
            ) =>
        {
            Type11SeedDecline::CededToResident
        }
        Some((have_w, have_h)) => Type11SeedDecline::GeomMismatch { have_w, have_h },
        None => Type11SeedDecline::NoEntry,
    };
    if !crate::observe::first_sight(crate::observe::Decline::slug(&d), disc) {
        return;
    }
    crate::observe::Emit::decline("type11_load_seed", &d)
        .field("mid", mapping_id)
        .field("want", format!("{w}x{h}"))
        .field("mapgeom", format!("{map_w}x{map_h}"))
        .field("mapgen", map_gen)
        .field("hostgen", host_gen)
        .fail();
}

/// The prior contents of a type-11 attachment under `PASS_LOAD_ACTION_LOAD`,
/// with the byte order they are in.
///
/// Two rungs, in freshness order:
///
/// 1. **The host render cache.** The hot one: `store_routes` measures 28-111 of
///    these a second under a browser workload. It holds guest scanout order and
///    the pooled target is RGBA, so the buffer is handed over behind an `Arc` and
///    the R/B exchange rides the engine's single copy into mapped staging rather
///    than materializing a converted frame here.
/// 2. **The surface's own guest IOSurface pages.** The cache is an accelerator,
///    not the surface. What a type-11 attachment *contains* is its pages, so a
///    cache miss is a reason to read them — not a reason to drop the guest's
///    LOAD. Without this rung the pass began with `LoadOp::CLEAR` against the
///    hardcoded `[0,0,0,0]` primary clear and the matching Store published that
///    wipe, which is a whole compositing layer going solid black.
///
/// `load_type11_mapping_rgba` reads at the mapping's own latched geometry and
/// converts to RGBA8, so the length check is what confirms the pass wanted that
/// extent — the engine rejects a seed of any other length, and the decline this
/// falls through to carries both geometries so a mismatch is diagnosable rather
/// than silent. `paint_mapping` underneath it lands every intersecting deferred
/// window first, so the read observes our own not-yet-written-back Stores rather
/// than pre-Store bytes.
///
/// The sibling Metal path already had rung 2: type-11 `seed_color_load` falls
/// through to the same reader via `load_sampled_rgba_static`. Only the Vulkan arm
/// stopped at the cache.
///
/// `None` means the guest's LOAD could not be honoured at all, and
/// [`note_type11_load_seed`] has already said which check refused.
fn resolve_type11_load_seed<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    w: u32,
    h: u32,
) -> Option<(
    std::sync::Arc<Vec<u8>>,
    crate::backend::vulkan::engine::SeedOrder,
)> {
    use crate::backend::vulkan::engine::SeedOrder;
    let served =
        if let Some(bgra) = crate::runtime::surface_cache::get_shared(state, mapping_id, w, h) {
            Some((bgra, SeedOrder::Bgra8, Type11SeedRung::Cache))
        } else {
            load_type11_mapping_rgba(state, host, mapping_id, None)
                .map(|(_, _, r)| r)
                .filter(|rgba| rgba.len() == (w as usize) * (h as usize) * 4)
                .map(|rgba| {
                    (
                        std::sync::Arc::new(rgba),
                        SeedOrder::Rgba8,
                        Type11SeedRung::GuestPages,
                    )
                })
        };
    note_type11_load_seed(state, mapping_id, w, h, served.as_ref().map(|s| s.2));
    match served.as_ref().map(|s| s.2) {
        Some(Type11SeedRung::Cache) => {
            crate::runtime::storage_flush::note_render_flush_cache_read(state, mapping_id)
        }
        Some(Type11SeedRung::GuestPages) => {
            crate::runtime::storage_flush::note_render_flush_pages_read(state, mapping_id)
        }
        _ => {}
    }
    served.map(|(bytes, order, _)| (bytes, order))
}

/// Materialize the exact serialized Metal view carried by a type-5 object.
///
/// The underlying type-4 FourCC is allocation metadata, not necessarily the
/// sampled Metal format. The view's format/geometry define the native row
/// interpretation; the type-4 device descriptor supplies its base/BPR/span.
/// Materialize a type-5 serialized texture view through the byte-exact
/// revalidated memo (same contract as [`load_linear_guest_memoized`]): every
/// bind re-reads the native plane window so a guest write is always observed;
/// conversion, allocation, and — via the returned content identity — the
/// engine upload are skipped when the bytes are unchanged.
pub(super) fn load_type5_view_rgba<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    mapping_id: u32,
    view: objects::Type5TextureView,
) -> Option<LoadedType5View> {
    let fail = |d: Type5ViewDecline| -> Option<LoadedType5View> {
        crate::observe::Emit::decline("type5_draw_view", &d)
            .field("task", task_id)
            .field("ref", texture_ref)
            .field("sid", mapping_id)
            .field("view", format!("{}x{}", view.width, view.height))
            .field("fmt", format!("{:#x}", view.pixel_format))
            .fail();
        None
    };

    if view.depth != 1 {
        return fail(Type5ViewDecline::UnsupportedDepth { depth: view.depth });
    }
    if !mapper::ensure_resolved_for_scanout(state, host, mapping_id) {
        return fail(Type5ViewDecline::Unresolved);
    }
    let Some(bpp) = pixel_format::bytes_per_pixel(view.pixel_format) else {
        return fail(Type5ViewDecline::FormatBpp);
    };
    let (base_off, surface_bpr, span_end, pages_n, base_w, base_h, base_fmt, map_gen, from_device) = {
        let Some(m) = state.mappings.get(&mapping_id) else {
            return fail(Type5ViewDecline::NoMapping);
        };
        let Some((base_off, surface_bpr, span_end, from_device)) =
            mapping_write::type5_sample_window(
                m,
                view.plane_index,
                view.width,
                view.height,
                view.pixel_format,
            )
        else {
            let desc =
                crate::contract::iosurface_pages::decode_device_surface(&m.device_desc).map(|d| {
                    (
                        d.width,
                        d.height,
                        d.pixel_format,
                        d.bytes_per_row,
                        d.alloc_size,
                    )
                });
            return fail(Type5ViewDecline::SampleWindow {
                base_w: m.width,
                base_h: m.height,
                base_fmt: m.format,
                desc,
            });
        };
        (
            base_off,
            surface_bpr,
            span_end,
            m.page_entries.len(),
            m.width,
            m.height,
            m.format,
            m.map_generation,
            from_device,
        )
    };
    // This path binds whatever window came back, invented or not — the per-bind
    // `invent=` echo below is behind `REIMS_VGPU_DRAW_LOG`, so on a normal boot a
    // wrong-plane bind here would be silent.
    if !from_device {
        mapping_write::note_type5_plane_invent(
            mapping_id,
            view.plane_index,
            view.width,
            view.height,
            view.pixel_format,
            (base_off, surface_bpr),
            "type5_draw_view",
        );
    }
    let page_bytes = (pages_n as u64).saturating_mul(1u64 << state.page_shift);
    if page_bytes < span_end {
        return fail(Type5ViewDecline::Span {
            pages: pages_n,
            page_bytes,
            span_end,
            bpr: surface_bpr,
        });
    }
    let Some(tight) = view.width.checked_mul(bpp) else {
        return fail(Type5ViewDecline::TightOverflow { bpp });
    };
    let Some(native_len) = (tight as u64)
        .checked_mul(view.height as u64)
        .and_then(host_alloc_len)
    else {
        return fail(Type5ViewDecline::NativeLen { tight });
    };
    let mut native = vec![0u8; native_len];
    if !mapping_write::read_rect_raw_at(
        state,
        host,
        mapping_id,
        base_off,
        surface_bpr,
        span_end,
        0,
        0,
        view.width,
        view.height,
        bpp,
        &mut native,
        tight,
    ) {
        return fail(Type5ViewDecline::Read {
            base_w,
            base_h,
            base_fmt,
            off: base_off,
            bpr: surface_bpr,
            span_end,
            pages: pages_n,
        });
    }
    // Identity key namespace: bit 63 marks type-5 view content (guest linear
    // identities use the raw sampled GVA as key). Every producer draws its
    // generation from `DeviceState::next_sampled_content_generation`, so a
    // (key, generation) pair cannot alias content even on a key collision.
    let identity_key = (1u64 << 63) | ((view.plane_index as u64) << 32) | mapping_id as u64;
    let memo_key = (
        mapping_id,
        view.plane_index,
        view.width,
        view.height,
        view.pixel_format,
    );
    // A single/dual-channel plane (biplanar video Y = R8, CbCr = RG8) uploads at
    // its native footprint: `texel_to_rgba8` places R8→(r,0,0,255) and
    // RG8→(r,g,0,255), which is exactly what an R8_UNORM / R8G8_UNORM Vulkan
    // image samples to (`.r` / `.rg`, zero-filled tail). Skipping the CPU expand
    // and uploading native cuts 4×/2× the staging bytes with byte-exact texels.
    // Formats with a native sampled rail upload their bytes verbatim; everything
    // else expands per texel into RGBA8 below. A format that has neither an entry
    // here nor an arm in `convert_row_to_rgba8` is refused at every bind, which
    // is what `R16_UNORM` was — 387 refusals of one 3840x2160 view in a single
    // logged-in session.
    let byte_format = match view.pixel_format {
        pixel_format::MTL_FORMAT_R8_UNORM => TexelLayout::R8,
        pixel_format::MTL_FORMAT_RG8_UNORM => TexelLayout::Rg8,
        pixel_format::MTL_FORMAT_R16_UNORM => TexelLayout::R16Unorm,
        pixel_format::MTL_FORMAT_RG16_UNORM => TexelLayout::Rg16Unorm,
        pixel_format::MTL_FORMAT_RG16_UINT => TexelLayout::Rg16Uint,
        pixel_format::MTL_FORMAT_R16_FLOAT => TexelLayout::R16Float,
        _ => TexelLayout::Rgba8,
    };
    let ok_line = |generation_source: &str, rgba: &[u8]| {
        // Per-draw success echo — fires on EVERY type-5 plane bind (thousands/sec
        // under video → ~36k lines/boot, 61% of the fail log), burying real
        // failures. The always-on health signal is the `sampled_branch_census`
        // aggregate (Type5View / T5Memo, noted on both paths below), so this
        // per-bind detail — and its O(w*h) `rgba_rgb_stats` scan — is diagnostic
        // only: gate both behind REIMS_VGPU_DRAW_LOG so a normal boot stays uncluttered.
        if !crate::observe::draw_log_enabled() {
            return;
        }
        let (nz, max, _) = crate::observe::rgba_rgb_stats(rgba);
        crate::observe::line(format!(
            "type5_draw_view ok task={task_id} ref={texture_ref} sid={mapping_id} map_gen={map_gen} view={}x{} fmt={:#x} bpp={bpp} base={base_w}x{base_h} base_fmt={base_fmt:#x} off={base_off} bpr={surface_bpr} span_end={span_end} invent={} src={generation_source} rgb_nz={nz} max_rgb={max}",
            view.width,
            view.height,
            view.pixel_format,
            (!from_device) as u8
        ));
    };
    if let Some(m) = state.type5_view_memo.get_touch(&memo_key) {
        // Vec equality is length + byte memcmp with early exit on change.
        if m.native == native {
            let rgba = m.rgba.clone();
            let generation = m.generation;
            ok_line("memo", &rgba);
            return Some((
                view.width,
                view.height,
                rgba,
                LinearSampleIdentity {
                    key: identity_key,
                    generation,
                },
                byte_format,
            ));
        }
    }
    // RGBA8 formats expand per-pixel into a fresh RGBA8 buffer; native R8/RG8
    // upload the plane bytes verbatim (the memo stores those bytes as both the
    // memcmp key and the upload payload).
    let rgba: std::sync::Arc<Vec<u8>> = if byte_format == TexelLayout::Rgba8 {
        let Some(rgba_stride) = view.width.checked_mul(RGBA8_BPP) else {
            return fail(Type5ViewDecline::RgbaStride);
        };
        let Some(rgba_len) = (rgba_stride as u64)
            .checked_mul(view.height as u64)
            .and_then(host_alloc_len)
        else {
            return fail(Type5ViewDecline::RgbaLen {
                stride: rgba_stride,
            });
        };
        let mut rgba = vec![0u8; rgba_len];
        for y in 0..view.height as usize {
            let src_off = y.saturating_mul(tight as usize);
            let dst_off = y.saturating_mul(rgba_stride as usize);
            if !pixel_format::convert_row_to_rgba8(
                view.pixel_format,
                &native[src_off..src_off + tight as usize],
                view.width,
                &mut rgba[dst_off..dst_off + rgba_stride as usize],
            ) {
                return fail(Type5ViewDecline::Convert { row: y, bpp });
            }
        }
        std::sync::Arc::new(rgba)
    } else {
        std::sync::Arc::new(native.clone())
    };
    let generation = state.next_sampled_content_generation();
    ok_line("fill", &rgba);
    let entry_bytes = native.len() + rgba.len();
    state.type5_view_memo.insert(
        memo_key,
        crate::model::GuestLinearMemo {
            native,
            rgba: rgba.clone(),
            // The type-5 view path carries its own native format (R8/Rg8/…);
            // this reused struct's `bgra8` flag is only read by the guest-linear
            // memo, so it is not load-bearing here.
            bgra8: false,
            generation,
        },
        entry_bytes,
    );
    Some((
        view.width,
        view.height,
        rgba,
        LinearSampleIdentity {
            key: identity_key,
            generation,
        },
        byte_format,
    ))
}

/// Zero-copy floor: below this the CPU byte path (one small read + memo) is
/// cheaper than a cached-window import plus a recorded GPU gather. Performance
/// threshold only — never a correctness gate.
///
/// Set to 64 KiB from a video-playback census: after the type-5 plane rail
/// landed, the whole remaining CPU copy under video was `t11_guest`
/// (~226 MB/session), and 100% of those declines were the floor —
/// per-frame-changing composite surfaces clustered at ~236 KiB, just under the
/// old 256 KiB. No memo can help (content changes every frame), so the CPU path
/// re-read + swizzled + double-SipHashed + re-uploaded ~236 KiB per frame for
/// nothing the GPU gather couldn't do from an already-imported (cached) window.
/// 64 KiB sits ~2× above the largest small-texture band that still legitimately
/// prefers the CPU byte path (small-UI / gva_copy binds measured at ~21–34 KiB,
/// and scroll glyphs at ~3.6 KiB served by the memo) and ~3.7× below the video
/// surfaces, so the band it opens to zero-copy is exactly those per-frame video
/// composites.
///
/// The floor is also the *only* thing that has ever declined the type-11
/// gather: over 1 051 sampled declines it was 100% of them, which is why the
/// rail no longer carries a reason enum to distinguish the rest.
pub(super) const ZERO_COPY_SAMPLED_MIN_BYTES: u64 = 64 * 1024;

/// Zero-copy floor for draw-time vertex/storage buffer binds. Performance
/// threshold only — never a correctness gate; below it the bind takes the CPU
/// staging read instead.
///
/// # The floor earns its keep, measured
///
/// It governs one bind in eight: `zc_buffer_below_floor` 19 218 against
/// `zc_buffer_gathered` 126 693 on a driven x86/Vulkan boot (Chess, Maps, the
/// WebGL aquarium, Wikipedia, apple.com).
///
/// Two boots of that same drive — one with the floor at 16 KiB, one with it at
/// 0 so every bind is gathered — on `draw_phase` normalised per draw (81 759 vs
/// 81 055 draws, 0.9 % apart):
///
/// | field       | floor 16 KiB | floor 0 |   delta |
/// |-------------|-------------:|--------:|--------:|
/// | `prep_us`   |        16.13 |   16.56 |  +2.7 % |
/// | `stage_us`  |        13.33 |   13.49 |  +1.3 % |
/// | `record_us` |         3.87 |    3.82 |  -1.1 % |
/// | `submit_us` |        42.87 |   48.59 | +13.3 % |
/// | total       |        76.19 |   82.46 |  +8.2 % |
///
/// Removing the floor costs 8 % of per-draw device time, so it stays. But the
/// reason it used to give was wrong, and the table is how. The old comment said
/// "below this the CPU staging read is cheaper"; `stage_us` barely moved when
/// 19 218 binds stopped being staged, so those small buffers were never the
/// staging cost. The cost is in **`submit_us`** — every gathered bind is a
/// recorded GPU gather, and 19 218 more of them make the submit 13 % dearer.
///
/// What is still not established is **16 KiB specifically**. This says a floor
/// above zero beats no floor on this workload; it does not say this is the best
/// one. A sweep would settle that, and until one is run the value is a guess
/// whose direction has evidence and whose magnitude does not.
pub(super) const ZERO_COPY_BUFFER_MIN_BYTES: u64 = 16 * 1024;

/// Does this host promise a guest-page alias that stays valid indefinitely?
///
/// Every guest-run producer below needs that promise, and needs it for a reason
/// that survived the removal of the host-pointer import: the engine gathers from
/// these pointers when the submission it armed them for reaches the GPU, which is
/// after this call returns, so a pointer with a bounded lifetime would be read
/// after its view was released.
///
/// A `false` is expected control flow — the caller falls through to the CPU
/// byte loader and the guest gets correct pixels — so it is not a decline. But
/// it is answered by the host once and then forever, and the whole rail
/// disappearing is not something a reader should have to infer from an absence,
/// so the first refusal of the process says so by name.
///
/// This is where the arm64 pathway now diverges: its MMIO shim can return a
/// `mach_vm_remap` view for a fragmented page list, and since that view is
/// released on `unmap_pages` rather than retained until teardown, the shim
/// answers 0. The x86 PCI shim never allocates — it refuses anything that is
/// not a packed host-contiguous run — so it still answers 1.
fn guest_run_alias_available<M: HostOps>(host: &M) -> bool {
    if host.map_pages_stable() {
        return true;
    }
    static NOTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !NOTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        crate::observe::fail(String::from(
            "guest_run_rail off reason=host_page_alias_not_stable \
             (draw binds take the CPU byte loader)",
        ));
    }
    false
}

/// Walk `span` bytes of `task_id`'s GVA space from `gva` and return the guest
/// pages covering it alongside the packed guest-RAM runs over them
/// (GPA-contiguous stretches coalesced and mapped to stable host pointers).
/// `None` when any page is unmapped or the mapping is incomplete. Shared by the
/// sampled and buffer zero-copy rails; callers must land intersecting deferred
/// stores first and verify import coverage per run.
///
/// The page list rides out with the runs because a caller that wants to say
/// anything about the window's *contents* over time needs the pages, not the
/// host pointers: guest-write tracking is registered per page set.
fn task_gva_guest_run_window<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &mut M,
    task_id: u32,
    gva: u64,
    span: u64,
) -> Option<(Vec<u64>, Vec<crate::backend::vulkan::engine::GuestRun>)> {
    if !guest_run_alias_available(host) {
        return None;
    }
    let page = state.page_size();
    let gpas =
        gva_mem::task_gva_page_gpas(host, &state.tasks, task_id, gva, span, state.page_shift);
    if gpas.len() as u64 != gva_mem::pages_spanned(gva, span, page) {
        return None;
    }
    let runs = coalesce_pages_to_runs(host, &gpas, page, gva % page, span)?;
    Some((gpas, runs))
}

/// [`task_gva_guest_run_window`] for callers with nothing to say about the
/// window's page set.
pub(super) fn task_gva_guest_runs<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &mut M,
    task_id: u32,
    gva: u64,
    span: u64,
) -> Option<Vec<crate::backend::vulkan::engine::GuestRun>> {
    task_gva_guest_run_window(state, host, task_id, gva, span).map(|(_, runs)| runs)
}

/// Coalesce GPA-contiguous stretches of `window` into packed host-VA runs
/// covering `span` bytes from `head_off` into the first page.
///
/// The single implementation of the walk every guest-pages rail needs: pick the
/// longest stretch whose GPAs ascend by exactly one page, import it once, and
/// take from it until `span` is met. `map_pages` hands back a direct RAMBlock
/// alias, so the import is a lookup and `unmap` is a no-op.
///
/// `None` if any stretch fails to import, or if the window runs out before
/// `span` — a partial gather would hand the GPU a short buffer, which is a
/// wrong frame rather than a slow one.
fn coalesce_pages_to_runs<M: HostOps>(
    host: &mut M,
    window: &[u64],
    page: u64,
    head_off: u64,
    span: u64,
) -> Option<Vec<crate::backend::vulkan::engine::GuestRun>> {
    use crate::backend::vulkan::engine;
    let mut runs: Vec<engine::GuestRun> = Vec::new();
    let mut consumed = 0u64;
    let mut i = 0usize;
    while i < window.len() && consumed < span {
        let mut j = i + 1;
        while j < window.len() && window[j] == window[i] + ((j - i) as u64) * page {
            j += 1;
        }
        let base = host.map_pages(&window[i..j], page as usize)? as u64;
        let start_in_run = if i == 0 { head_off } else { 0 };
        let avail = ((j - i) as u64) * page - start_in_run;
        let len = avail.min(span - consumed);
        runs.push(engine::GuestRun {
            host_ptr: (base + start_in_run) as usize,
            len,
        });
        consumed += len;
        i = j;
    }
    (consumed == span).then_some(runs)
}

/// The byte extent of a `w × h` image at `bpr` bytes per row and `bpp` bytes per
/// texel, and the `bufferRowLength` in texels the copy needs to stride the
/// padding (0 when rows are tight).
///
/// `None` when the stride cannot describe the image: narrower than one tight
/// row, or not a whole number of texels — `bufferRowLength` is a texel count, so
/// a byte-granular stride has no representation. Padded strides otherwise ride
/// the same rail. The extent stops after the last row's texels because trailing
/// padding may not be mapped.
pub(super) fn strided_window_extent(w: u32, h: u32, bpp: u64, bpr: u64) -> Option<(u64, u32)> {
    let tight = (w as u64).checked_mul(bpp)?;
    if bpr < tight || bpp == 0 || !bpr.is_multiple_of(bpp) {
        return None;
    }
    let span = bpr
        .checked_mul(h.checked_sub(1)? as u64)?
        .checked_add(tight)?;
    let row_length_texels = if bpr == tight {
        0
    } else {
        u32::try_from(bpr / bpp).ok()?
    };
    Some((span, row_length_texels))
}

/// Gather `span` bytes from `base_off` into mapping `mid`'s guest pages as host
/// runs, landing any deferred writeback that aliases them first. Returns the
/// window's own page list beside the runs, for the same reason
/// [`task_gva_guest_run_window`] does.
///
/// Shared by the type-11 and type-5 sampled rails, which reach the same pages
/// through different window math. The flush is the coherence rule the CPU
/// loaders obey: a resident-authoritative window covering this mapping must
/// land before the GPU reads, or the gather sees the pre-Store bytes.
fn mapping_window_guest_runs<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mid: u32,
    base_off: u64,
    span: u64,
) -> Option<(Vec<u64>, Vec<crate::backend::vulkan::engine::GuestRun>)> {
    if !guest_run_alias_available(host) {
        return None;
    }
    let _ = crate::runtime::storage_flush::flush_intersecting(state, host, mid, 0, u64::MAX);
    let gpas = mapper::mapping_page_gpas(state, host, mid)?;
    let page = state.page_size();
    if (gpas.len() as u64).saturating_mul(page) < base_off.checked_add(span)? {
        return None;
    }
    let first_page = (base_off / page) as usize;
    let head_off = base_off % page;
    let need_pages = (head_off + span).div_ceil(page) as usize;
    let window = gpas.get(first_page..first_page + need_pages)?;
    let runs = coalesce_pages_to_runs(host, window, page, head_off, span)?;
    Some((window.to_vec(), runs))
}

/// Zero-copy draw-time buffer bind: resolve a type-1 buffer object's backing
/// span (from `offset`) to guest-RAM runs and hand the engine a
/// [`engine::BufferContent::GuestRuns`] — the GPU gathers the bytes from
/// imported guest RAM inside the draw's own CB. Replaces the per-draw CPU
/// re-read + double memcpy of the same ~50–260 KB vertex/SSBO buffers.
/// Guest CPU writes are still observed: the gather re-executes every draw
/// and reads at execute time (at least as fresh as the CPU path).
///
/// Gates (any miss → `None`, caller stays on the CPU staging read): span ≥
/// the buffer zero-copy floor and every page walkable into mappable runs.
/// Deferred stores intersecting the span are landed first, exactly like the
/// CPU path.
fn try_buffer_zero_copy_resolved<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    backing: &BufferBacking,
    offset: u64,
) -> Option<crate::backend::vulkan::engine::BufferContent> {
    use crate::backend::vulkan::engine;
    let (gva, size) = (backing.gva, backing.size);
    if offset >= size {
        return None;
    }
    let span = host_alloc_len(size - offset).filter(|&n| n > 0)? as u64;
    if span < ZERO_COPY_BUFFER_MIN_BYTES {
        crate::runtime::drain::note_store_route("zc_buffer_below_floor");
        return None;
    }
    if !guest_run_alias_available(host) {
        return None;
    }
    // Same coherence rule as the CPU read: land any resident-authoritative
    // writeback aliasing the span before the GPU reads the pages (the CPU
    // flush completes before this draw's submit).
    crate::runtime::storage_flush::flush_intersecting_task_gva(
        state,
        host,
        task_id,
        gva + offset,
        span,
    );
    // Walk exactly the bound range. Resolving the whole backing and slicing out
    // the bind would translate every page of the allocation to serve one bind,
    // and would refuse a bind whose allocation has an unmapped tail page even
    // though the bind itself resolves.
    let runs = task_gva_guest_runs(state, host, task_id, gva + offset, span)?;
    crate::runtime::drain::note_store_route("zc_buffer_gathered");
    Some(engine::BufferContent::GuestRuns(engine::GuestRunSource {
        runs: std::sync::Arc::new(runs),
        total_len: span,
        row_length_texels: 0,
    }))
}

/// Load one draw-time buffer bind: the zero-copy rail when allowed and
/// eligible, else the CPU staging read. `allow_zero_copy` is false for
/// buffers feeding Constant-step attributes (the engine prepends a CPU
/// base-instance prefix to those).
fn load_buffer_content<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    buffer_ref: u32,
    offset: u64,
    allow_zero_copy: bool,
) -> Option<crate::backend::vulkan::engine::BufferContent> {
    // Resolve the backing (object-list entry + descriptor) ONCE and share it
    // between the zero-copy attempt and the CPU fallback. Sub-floor binds used
    // to walk the task PT twice — once in the failed ZC attempt, once in the
    // CPU read.
    let backing = resolve_buffer_backing(state, host, task_id, buffer_ref)?;
    if allow_zero_copy {
        if let Some(content) = try_buffer_zero_copy_resolved(state, host, task_id, &backing, offset)
        {
            return Some(content);
        }
    }
    let bytes = read_buffer_bytes_resolved(state, host, task_id, &backing, offset)?;
    Some(crate::backend::vulkan::engine::BufferContent::from(bytes))
}

/// Zero-copy linear sampled bind: resolve the texture's tight level-0 GVA
/// window to packed-contiguous guest-RAM runs and hand the engine a
/// [`SampledSource::GuestRuns`] — the GPU gathers the texels from imported
/// guest RAM inside the draw's own CB. Replaces the lin_memo class's
/// full-window CPU re-read + memcmp per bind (guest CPU writes are still
/// observed: the GPU copy re-executes every draw and reads at execute time).
///
/// Gates (any miss → `None`, caller stays on the CPU byte paths): native
/// texel layout Vulkan samples identically (BGRA8/RGBA8 UNORM), tight rows,
/// window inside the allocation, span ≥ the zero-copy floor, every page
/// walkable, and packed-contiguous runs mappable.
fn try_linear_sample_zero_copy<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    _texture_ref: u32,
    tex: &TextureDescriptor,
) -> Option<(u32, u32, SampledSourceRequest)> {
    use crate::backend::vulkan::engine;
    // The object-list entry + descriptor are resolved+decoded once by the caller
    // (`resolve_sampled_source`'s linear branch) and threaded in as `tex`; the
    // cache fallback shares the same decode.
    if !tex.has_pixel_format {
        return None;
    }
    // sRGB variants ride the same rail as their linear siblings: the layout is
    // identical and the CPU loaders never decoded either. The qualifier is
    // still lost, so the census records it rather than letting the fold be
    // silent.
    // Four-byte colour (BGRA8/RGBA8) or a single-channel float LUT: all sample
    // byte-identically through the matching native Vulkan image. Other layouts
    // (R8/Rg8 video planes) keep their existing CPU/type-5 rails. `R32_SFLOAT`
    // additionally needs the optional linear-filter feature — LUTs are sampled
    // with interpolation — so it is gated on the host capability and otherwise
    // declines here, leaving the sample fail-visible (no CPU float loader arm).
    let native = match translate::pixel::sampled_pixels(tex.pixel_format) {
        Ok((layout, decline))
            if layout.is_four_byte_color()
                || layout == TexelLayout::R16Float
                || layout == TexelLayout::R16Unorm
                || layout == TexelLayout::Rg16Unorm
                || layout == TexelLayout::Rg16Uint
                || (layout == TexelLayout::R32Float
                    && engine::supports_sampled_r32f_linear_filter()) =>
        {
            if decline.is_some() {
                srgb_census::note_downgrade(
                    srgb_census::site::LINEAR_SAMPLE_ZERO_COPY,
                    tex.pixel_format,
                );
            }
            layout
        }
        _ => return None,
    };
    let bpp = native.bytes_per_texel();
    let (gva, layout) = tex.level_gva(0, state.page_shift)?;
    let (w, h) = (layout.width, layout.height);
    if w == 0 || h == 0 {
        return None;
    }
    let (span, row_length_texels) = strided_window_extent(w, h, bpp as u64, layout.row_stride)?;
    // The min-byte floor keeps small four-byte textures on the cheaper CPU
    // memo/cache path. Single-channel float LUTs have no CPU loader arm
    // (`texel_to_rgba8` returns `None`), so this native gather is their only
    // correct rail — exempt them from the floor or a small display-profile LUT
    // would fall through to a failed resolve.
    if native.is_four_byte_color() && span < ZERO_COPY_SAMPLED_MIN_BYTES {
        return None;
    }
    if !guest_run_alias_available(host) {
        return None;
    }
    if tex.allocation_size != 0 && layout.offset.saturating_add(span) > tex.allocation_size {
        return None;
    }
    // Same coherence rule as the CPU loaders: land any resident-authoritative
    // writeback aliasing the span before the GPU reads the pages (the CPU
    // flush completes before this draw's submit).
    crate::runtime::storage_flush::flush_intersecting_task_gva(state, host, task_id, gva, span);
    // Fixed per-texture window: the walk covers exactly the bound span.
    let (gpas, runs) = task_gva_guest_run_window(state, host, task_id, gva, span)?;
    let page = state.page_size() as usize;
    let vouched = crate::runtime::gather_witness::note_gather(
        state,
        host,
        crate::runtime::gather_witness::GatherRail::Linear,
        crate::runtime::gather_witness::GatherKey::TaskGva { task_id, gva },
        crate::runtime::gather_witness::GatherWindow {
            gpas: &gpas,
            runs: &runs,
            span,
            page_size: page,
        },
    );
    Some((
        w,
        h,
        SampledSourceRequest::GuestRuns(
            engine::GuestRunSource {
                runs: std::sync::Arc::new(runs),
                total_len: span,
                row_length_texels,
            },
            native,
            vouched.map(LinearSampleIdentity::from),
        ),
    ))
}

/// Zero-copy rail for type-11 mapping-backed sampled binds. Eligible when
/// the mapping's raw bytes sample byte-identically through a native UNORM
/// image (BGRA8/RGBA8 families — the CPU loader's `texel_to_rgba8` is a
/// byte pass-through/swizzle for exactly these) and the caller established
/// the resident is not authoritative. Mirrors `paint_mapping`'s window math
/// (`type11_sample_window`) and its flush-on-access rule; any gate miss
/// falls back to the CPU byte path.
pub(super) fn try_type11_sample_zero_copy<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mid: u32,
    w: u32,
    h: u32,
) -> Option<SampledSourceRequest> {
    use crate::backend::vulkan::engine;
    use crate::runtime::mapping_write::type11_sample_window;
    if w == 0 || h == 0 {
        return None;
    }
    let (native, base_off, bpr) = {
        let m = state.mappings.get(&mid)?;
        if !m.mapped || m.page_entries.is_empty() {
            return None;
        }
        let format = if m.format != 0 {
            m.format
        } else {
            pixel_format::MTL_FORMAT_BGRA8_UNORM
        };
        let native = match translate::pixel::sampled_pixels(format) {
            Ok((layout, decline)) if layout.is_four_byte_color() => {
                if decline.is_some() {
                    srgb_census::note_downgrade(srgb_census::site::TYPE11_SAMPLE_ZERO_COPY, format);
                }
                layout
            }
            _ => return None,
        };
        let (base_off, bpr_u32, _span_end) = type11_sample_window(m, mid, w, h, format)?;
        (native, base_off, bpr_u32 as u64)
    };
    // From the layout the translation chose, as the type-5 rail does, so the
    // texel size cannot disagree with the image the engine creates. The
    // `is_four_byte_color` gate above already fixes it at four.
    let (span, row_length_texels) =
        strided_window_extent(w, h, native.bytes_per_texel() as u64, bpr)?;
    if span < ZERO_COPY_SAMPLED_MIN_BYTES {
        return None;
    }
    let (gpas, runs) = mapping_window_guest_runs(state, host, mid, base_off, span)?;
    let page = state.page_size() as usize;
    let vouched = crate::runtime::gather_witness::note_gather(
        state,
        host,
        crate::runtime::gather_witness::GatherRail::Type11,
        crate::runtime::gather_witness::GatherKey::Mapping { mid, base_off },
        crate::runtime::gather_witness::GatherWindow {
            gpas: &gpas,
            runs: &runs,
            span,
            page_size: page,
        },
    );
    Some(SampledSourceRequest::GuestRuns(
        engine::GuestRunSource {
            runs: std::sync::Arc::new(runs),
            total_len: span,
            row_length_texels,
        },
        native,
        vouched.map(LinearSampleIdentity::from),
    ))
}

/// Zero-copy rail for a type-5 serialized IOSurface plane view — the video
/// hot path. VideoToolbox decodes to NV12 (Y = R8, CbCr = RG8; also
/// BGRA8/RGBA8 surfaces), sampled through the type-5 view path whose CPU
/// loader (`load_type5_view_rgba`) read + uploaded ~1.5 MB per plane per
/// decoded frame (census `t5_view`). This gathers the plane's guest pages
/// directly in the draw CB so the decoded frame never materializes CPU bytes.
/// Mirrors `try_type11_sample_zero_copy`'s page coalescing over the plane
/// window from `type5_sample_window` (which carries the wire plane index +
/// biplanar offset); any gate miss falls back to the CPU byte path.
fn try_type5_sample_zero_copy<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mid: u32,
    view: objects::Type5TextureView,
) -> Option<SampledSourceRequest> {
    use crate::backend::vulkan::engine;
    use crate::runtime::mapping_write::type5_sample_window;
    let (w, h) = (view.width, view.height);
    if w == 0 || h == 0 || view.depth != 1 {
        return None;
    }
    // Match the CPU path's resolution before reading the plane pages.
    if !mapper::ensure_resolved_for_scanout(state, host, mid) {
        return None;
    }
    let (native, bpp, base_off, bpr) = {
        let m = state.mappings.get(&mid)?;
        if !m.mapped || m.page_entries.is_empty() {
            return None;
        }
        // Native formats whose guest bytes sample byte-identically through the
        // matching Vulkan image (the CPU loader's `texel_to_rgba8` is a
        // pass-through/swizzle for exactly these); everything else stays CPU.
        // The texel size comes from the layout the translation chose, so it can
        // never disagree with the image the engine creates.
        let (native, bpp) = match translate::pixel::sampled_pixels(view.pixel_format) {
            Ok((layout, decline)) => {
                if decline.is_some() {
                    srgb_census::note_downgrade(
                        srgb_census::site::TYPE5_PLANE_ZERO_COPY,
                        view.pixel_format,
                    );
                }
                (layout, layout.bytes_per_texel())
            }
            Err(_) => return None,
        };
        // Only a real device-descriptor plane window rides zero copy; the
        // invented packed fallback over a stale multiplanar mapping (menu-strip
        // residual class) stays on the CPU path.
        let (base_off, bpr_u32, _span_end, from_device) =
            type5_sample_window(m, view.plane_index, w, h, view.pixel_format)?;
        if !from_device {
            return None;
        }
        (native, bpp, base_off, bpr_u32 as u64)
    };
    let (span, row_length_texels) = strided_window_extent(w, h, bpp as u64, bpr)?;
    if span < ZERO_COPY_SAMPLED_MIN_BYTES {
        return None;
    }
    let (gpas, runs) = mapping_window_guest_runs(state, host, mid, base_off, span)?;
    let page = state.page_size() as usize;
    let vouched = crate::runtime::gather_witness::note_gather(
        state,
        host,
        crate::runtime::gather_witness::GatherRail::Type5,
        crate::runtime::gather_witness::GatherKey::Mapping { mid, base_off },
        crate::runtime::gather_witness::GatherWindow {
            gpas: &gpas,
            runs: &runs,
            span,
            page_size: page,
        },
    );
    Some(SampledSourceRequest::GuestRuns(
        engine::GuestRunSource {
            runs: std::sync::Arc::new(runs),
            total_len: span,
            row_length_texels,
        },
        native,
        vouched.map(LinearSampleIdentity::from),
    ))
}

/// Serve a guest-CPU-produced linear texture (tight OR padded row stride)
/// through the byte-exact revalidated memo. Every call re-reads the native
/// guest rows (a guest write is always observed); only the swizzle/gather +
/// allocation — and, via the returned generation identity, the engine's
/// content hash + upload — are skipped when the bytes are unchanged. Returns
/// the upload byte format (native BGRA8 when eligible, else RGBA8). Measured
/// on Safari fast-scroll: the padded-stride glyph/tile atlases re-present only
/// ~59 distinct gva keys with ~99% recurrence (`fallback_gva_churn`), so this
/// memo now serves that former `lin_guest_fb` hot path instead of a per-bind
/// re-read+re-upload. Returns `None` (no logging: a fast-path miss, not a
/// failure) only for sub-tight strides or formats `convert_row_to_rgba8`
/// cannot decode, which fall through to the general loader.
/// Convert the raw native rows read for a guest-linear texture (row stride
/// `bpr`, `tight` = the packed row byte count) into the tight upload buffer.
/// A 4-byte straight upload — RGBA8, or BGRA8 kept native — gathers each row
/// with a plain copy (padding skipped, no swizzle) and reports its native
/// format; every other format converts to RGBA8 per row. Shared by the
/// guest-linear memo's miss-fill so its padded and tight branches agree
/// byte-for-byte with the direct loader.
fn native_scratch_to_upload(
    scratch: &[u8],
    w: u32,
    h: u32,
    bpr: u64,
    sample_fmt: u16,
    tight: u64,
) -> Option<(Vec<u8>, TexelLayout)> {
    let out_row = (w as usize).checked_mul(RGBA8_BPP as usize)?;
    let out_len = out_row.checked_mul(h as usize)?;
    let bpr = bpr as usize;
    if let Some(fmt) = linear_native_upload_format(sample_fmt, true)
        .filter(|_| tight == (w as u64).saturating_mul(RGBA8_BPP as u64))
    {
        let row_bytes = tight as usize;
        let mut out = vec![0u8; out_len];
        for y in 0..h as usize {
            let src = y.checked_mul(bpr)?;
            let dst = y * row_bytes;
            out.get_mut(dst..dst + row_bytes)?
                .copy_from_slice(scratch.get(src..src + row_bytes)?);
        }
        return Some((out, fmt));
    }
    let trow = tight as usize;
    let mut out = vec![0u8; out_len];
    for y in 0..h as usize {
        let src = y.checked_mul(bpr)?;
        if !pixel_format::convert_row_to_rgba8(
            sample_fmt,
            scratch.get(src..src + trow)?,
            w,
            &mut out[y * out_row..],
        ) {
            return None;
        }
    }
    Some((out, TexelLayout::Rgba8))
}

/// The sampled linear ladder's hot rung: read the guest's own rows for this
/// texture, reuse the converted copy when the bytes have not changed.
///
/// It carries essentially all of this pathway's sampled traffic — 725 231 of
/// 725 233 loads on a driven boot — so it is where a wrong-content defect would
/// have to live, and it is worth stating plainly that nothing here guesses:
///
/// - The address chain is resolved fresh on every call.
///   `objects::lookup_list_entry` re-reads the object-list entry out of guest
///   memory, `read_descriptor` re-reads the descriptor, and `level_gva` derives
///   the span from that. No step caches, so a recycled `texture_ref` cannot
///   hand this the previous resource's address.
/// - The staleness check is exact, not sampled. The full `bpr * h` native span
///   is re-read every call and compared byte for byte against the memo
///   (`m.native == scratch`), padding included, so a guest write anywhere in
///   the span misses the memo.
/// - The read does not go through a cached host view. `gva_view`'s registered
///   views measured a zero hit rate on this pathway (`view_reuse` = 0 over four
///   boots), so `read_task_gva_by_id` walks the page table here.
///
/// That is measured, not asserted, and it is why the surviving Finder icon
/// class is not a wrong-bytes defect on this rung: the bytes served are the
/// bytes at the address the guest named, checked afresh each time.
#[allow(clippy::too_many_arguments)]
fn load_linear_guest_memoized<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    tex: &TextureDescriptor,
    gva: u64,
    w: u32,
    h: u32,
) -> Option<(
    std::sync::Arc<Vec<u8>>,
    Option<LinearSampleIdentity>,
    TexelLayout,
)> {
    if !tex.has_pixel_format {
        return None;
    }
    let sample_fmt = effective_view_sample_format(tex.pixel_format, None)?;
    let (_, layout) = tex.level_gva(0, state.page_shift)?;
    let bpr = layout.row_stride;
    let tight = pixel_format::tight_row_bytes(w, tex.pixel_format)? as u64;
    // Padded strides ride the same memo now — the native read below covers the
    // full `bpr*h` span (padding included, so a write anywhere is observed) and
    // `native_scratch_to_upload` gathers the tight rows. Only a sub-tight stride
    // (impossible geometry) or a zero dimension declines to the fallback.
    if bpr < tight || w == 0 || h == 0 {
        return None;
    }
    let span = bpr.checked_mul(h as u64)?;
    let native_len = host_alloc_len(span)?;
    if tex.allocation_size != 0 && layout.offset.saturating_add(span) > tex.allocation_size {
        return None;
    }
    // Same coherence rule as the general loader: land any resident-
    // authoritative writeback aliasing the sampled span before reading it.
    crate::runtime::storage_flush::flush_intersecting_task_gva(state, host, task_id, gva, span);
    let mut scratch = std::mem::take(&mut state.guest_linear_scratch);
    scratch.resize(native_len, 0);
    let read = gva_mem::read_task_gva_by_id(
        host,
        &state.tasks,
        task_id,
        gva,
        &mut scratch,
        state.page_shift,
    );
    if read.is_err() {
        state.guest_linear_scratch = scratch;
        return None;
    }
    let key = (task_id, gva, w, h, sample_fmt);
    // Three-way, because "the memo did not answer" has two causes that want
    // opposite fixes and a hit/miss pair cannot tell them apart.
    //
    // This memo cannot skip the guest read — the read is *how* it knows the
    // content is unchanged — so a hit buys exactly one thing: it skips
    // `native_scratch_to_upload`, the pixel-format conversion. Everything else
    // it costs is paid on every bind regardless: the read, the memcmp against
    // `native`, and a cap charged for storing the native bytes *and* their
    // converted copy.
    //
    // So the memo is worth its keep only if `lin_memo_hit` dominates.
    // `lin_memo_changed` is the memcmp running in full and buying nothing — the
    // guest rewrote the plane, which is the case the memo cannot help. And
    // `lin_memo_absent` is the key never repeating, where the cap is holding
    // bytes no bind will ask for again.
    //
    // **Measured, and it earns its keep.** One driven x86 / Vulkan boot (two
    // Safari page loads, scrolls, three title-bar drags): hit 7221, changed 557,
    // absent 179 — a **90.8 %** hit rate, so nine binds in ten skip the format
    // conversion entirely. The three arms sum to 7957, which is exactly
    // `lin_rung_guest_memo` for the same boot; that reconciliation is what says
    // the census is complete rather than merely quiet.
    //
    // This was instrumented to decide whether to delete the memo, on the
    // suspicion it was another cache paying its own miss on every hit. It is
    // not: unlike a walk memo it cannot skip the guest read, but the read was
    // never what it claimed to save. Keep it. The counters stay because the
    // answer is workload-dependent — a guest that rewrites its planes every
    // frame would push `changed` up and invert the conclusion — so this is a
    // ratio worth re-reading, not a settled fact worth deleting.
    let hit = match state.guest_linear_memo.get_touch(&key) {
        None => {
            crate::runtime::drain::note_store_route("lin_memo_absent");
            None
        }
        // Vec equality is length + byte memcmp with early exit on change.
        Some(m) if m.native == scratch => {
            crate::runtime::drain::note_store_route("lin_memo_hit");
            Some((m.rgba.clone(), m.generation, m.bgra8))
        }
        Some(_) => {
            crate::runtime::drain::note_store_route("lin_memo_changed");
            None
        }
    };
    if let Some((rgba, generation, bgra8)) = hit {
        let fmt = if bgra8 {
            TexelLayout::Bgra8
        } else {
            TexelLayout::Rgba8
        };
        state.guest_linear_scratch = scratch;
        return Some((
            rgba,
            Some(LinearSampleIdentity {
                key: gva,
                generation,
            }),
            fmt,
        ));
    }
    // First sight or native bytes changed: convert fresh, new generation.
    let Some((rgba, fmt)) = native_scratch_to_upload(&scratch, w, h, bpr, sample_fmt, tight) else {
        state.guest_linear_scratch = scratch;
        return None;
    };
    let generation = state.next_sampled_content_generation();
    let rgba = std::sync::Arc::new(rgba);
    let entry_bytes = scratch.len() + rgba.len();
    state.guest_linear_memo.insert(
        key,
        crate::model::GuestLinearMemo {
            native: scratch,
            rgba: rgba.clone(),
            bgra8: fmt == TexelLayout::Bgra8,
            generation,
        },
        entry_bytes,
    );
    Some((
        rgba,
        Some(LinearSampleIdentity {
            key: gva,
            generation,
        }),
        fmt,
    ))
}

/// Report a sampled texture served entirely as zeroes out of the guest's pages
/// while the host cache holds an entry for the same address.
///
/// A type-2/3 texture's guest GVA pages are a *pageable alias* of a body this
/// device owns (`surface_cache::store_linear_texture`), so a blank read here
/// could mean the device rendered the span, cached it, and its own writeback
/// never landed in the guest's pages — a silent loss, since the draw then
/// succeeds and paints a blank cell with nothing declining.
///
/// Three questions separate that defect from its lookalikes, and this function
/// asks all three:
///
/// - **Does the cache hold this span at all?** `lin_rung_host_entry` against
///   `lin_rung_guest_memo` is the denominator. Without it a bare count cannot
///   tell "300 of 300" from "300 of 300 000".
/// - **Are the zeroed pages still the pages the entry was produced over?**
///   [`crate::runtime::surface_cache::gva_backing_state`]: `Same` means the
///   cache entry is live over these pages, `Moved`/`Unmapped` means the guest
///   handed the address to another allocation and the *cache* is the stale
///   side — where serving it would be the corruption, not the repair.
/// - **Does the entry hold any pixels?** The question the class was named for
///   and never asked. A span the device CLEARed and cached blank reads blank
///   off blank pages with nothing lost, and `draw_partial_clear` runs in the
///   thousands a boot.
///
/// ## Measured, and the class is not a loss
///
/// One driven x86/Vulkan boot — 30 s Safari window drag plus two web-content
/// probe runs, all declared regions measuring their colour — summed over its
/// `store_routes` windows:
///
/// ```text
/// lin_rung_guest_memo             79898   sampled serves off the guest's pages
/// lin_rung_host_entry             18988   …of which the cache also held the span (23.8 %)
/// lin_rung_guest_blank             1859   …that came back all zeroes (2.3 %)
/// lin_rung_blank_with_host_entry     22   …of those, with a host entry (1.2 % of blanks)
/// lin_rung_blank_host_agrees         22   …where the cache is blank too: nothing lost
/// lin_rung_blank_host_content         0   …where the cache holds pixels: the defect
///
/// 13 distinct spans, backing=Same and fmt=Bgra8 on every one
/// ```
///
/// A second driven boot on the same workload read 28 / 28 / 0 — the same
/// partition, so the zero is not one boot's luck.
///
/// So the two rails agree on every occurrence. The dominant blank class is
/// elsewhere — 98.8 % of blank samples have no cache entry for the span at all,
/// which is "we do not have the pixels", not "we lost them". `fmt=Bgra8`
/// throughout also excludes a conversion artifact: the blank test runs on
/// converted RGBA, and a layout whose conversion zeroed the buffer would show
/// up as a different `fmt`.
///
/// `lin_rung_blank_host_content` is therefore a healthy zero, and a non-zero
/// reading is the alarm: it is the only arm that means guest work was lost, and
/// the place to repair it is the GVA writeback rail upstream of this rung.
///
/// ## What this does not license
///
/// Serving the cache on the whole rung — making the order match
/// [`crate::runtime::metal_draw::seed_color_load`]'s stated rule, "exact target
/// GVA is the strongest identity … Guest memory is last" — would change ~19 000
/// serves to repair nothing. The two rails are not the same case: the seed's
/// entry is for an attachment the pass is about to draw *onto*, while a sampled
/// span may be guest-CPU-produced between the encode and the sample with
/// nothing here able to witness it. Serving the cache only when the sample came
/// back blank is not available either — that is selecting on content.
///
/// The `fail` line is behind `first_sight` on `(gva, w, h)`, so it fires once
/// per distinct span for the life of the boot while the counters beside it are
/// per-occurrence; the two are not comparable. The `gva_backing_state` walk
/// sits under that latch, so it is one page-table walk per distinct span rather
/// than per sample.
///
/// `span` is `(gva, width, height)` — the GVA cache's key, taken as one value
/// because every lookup below needs all three and none of them means anything
/// apart.
fn note_guest_rung_blank<H: HostMemory>(
    state: &DeviceState,
    host: &H,
    task_id: u32,
    texture_ref: u32,
    span: (u64, u32, u32),
    rgba: &[u8],
    byte_format: TexelLayout,
) {
    let (gva, w, h) = span;
    crate::runtime::drain::note_store_route("lin_rung_guest_memo");
    // The denominator for the loss below: every serve off the guest's pages for
    // a span the cache also holds, whatever came back. Taken before the blank
    // test so the blank ones are a subset of a population, not a bare count.
    //
    // The bytes, not just the presence: a blank guest read only means pixels
    // were lost if the cache holds pixels to lose. `has_gva` cannot tell the two
    // apart, so it counted "we cached a blank frame" as loss.
    let host_bytes = crate::runtime::surface_cache::get_gva(state, gva, w, h);
    let host_entry = host_bytes.is_some();
    if host_entry {
        crate::runtime::drain::note_store_route("lin_rung_host_entry");
    }
    if rgba.is_empty() || rgba.iter().any(|&b| b != 0) {
        return;
    }
    crate::runtime::drain::note_store_route("lin_rung_guest_blank");
    // Identity, latched per span, because the count alone cannot say whether a
    // blank sample is a transparent layer doing its job or the icon cell that
    // came out empty. 99.5 % of loads on this rung return content, so the
    // population that matters is small enough to name each member of, and the
    // geometry is what joins one of these to something on screen.
    if crate::observe::first_sight("lin_rung_guest_blank", gva ^ ((w as u64) << 32) ^ h as u64) {
        crate::observe::off(format!(
            "lin_rung_guest_blank task={task_id} ref={texture_ref} gva={gva:#x} {w}x{h}"
        ));
    }
    let Some(host_bytes) = host_bytes else {
        return;
    };
    crate::runtime::drain::note_store_route("lin_rung_blank_with_host_entry");
    // Which of the two cases this span is. A cache entry that is itself all
    // zeroes agrees with the guest's pages, so nothing was lost and there is
    // nothing upstream to repair; only a cache entry holding content while the
    // guest alias reads zero is a coherence loss.
    let host_blank = host_bytes.iter().all(|&b| b == 0);
    crate::runtime::drain::note_store_route(if host_blank {
        "lin_rung_blank_host_agrees"
    } else {
        "lin_rung_blank_host_content"
    });
    if crate::observe::first_sight(
        "lin_rung_blank_with_host_entry",
        gva ^ ((w as u64) << 32) ^ h as u64,
    ) {
        // Under `first_sight`, so this walk is once per distinct blank span for
        // the life of the boot and not once per sample.
        let backing = crate::runtime::surface_cache::gva_backing_state(state, host, gva);
        crate::observe::fail(format!(
            "lin_rung_blank_with_host_entry task={task_id} ref={texture_ref} \
             gva={gva:#x} {w}x{h} bytes={} fmt={byte_format:?} host_blank={} \
             backing={backing:?} (guest alias is zero and the host cache has this span; \
             host_blank=true means the cache agrees and nothing was lost, false means \
             the cache holds content this read did not return; backing=Same means the \
             cache entry is still over these pages, Moved/Unmapped means the address \
             was handed on and the cache entry is the stale one)",
            rgba.len(),
            u8::from(host_blank)
        ));
    }
}

pub(super) fn load_linear_from_host_caches<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    tex: &TextureDescriptor,
) -> Option<LoadedLinearSample> {
    // The descriptor is resolved+decoded once by the caller
    // (`resolve_sampled_source`'s linear branch) and threaded in; the zero-copy
    // attempt above shares the same decode.
    let (gva, layout) = tex.level_gva(0, state.page_shift)?;
    let w = layout.width;
    let h = layout.height;
    if w == 0 || h == 0 {
        return None;
    }
    // A deferred GVA render Store at this base is the authoritative content and
    // the guest's pages do not have it yet — land it before the readers below,
    // which read those pages.
    if state.gva_deferred_flush.contains_key(&gva) {
        crate::runtime::storage_flush::flush_gva_exact(state, host, gva, true, "gva_sample");
    }
    // Two cache rungs that used to sit here are deliberately absent; both were
    // measured dead before removal, so do not restore either.
    //
    // The GVA encode cache rung could not serve. Its freshness gate
    // (`gva_guest_wrote_since_store`) answered `no_entry`/`no_baseline` 286 800
    // times against `clean`/`wrote` zero — exactly the bypass count — because
    // `arm_gva_guest_write_witness` stamps its baseline inside the dirty
    // tracker's two-harvest startup window and re-stamps only on a *later*
    // store. A surface stored once and sampled forever never gets one, so the
    // gate can only answer "stale". The sibling mapping rail escapes this by
    // re-stamping on every write (`mapper::stamp_guest_write_gen`).
    //
    // The other was keyed on `texture_ref` + geometry with no GVA validation,
    // so it could serve one resource's pixels under another's ref after a
    // rebind at the same size. It reached 2 serves against 725 233 sampled
    // loads.
    //
    // Still open, and a contract gap rather than a tuning one: that same cache
    // is read by the colour-LOAD seed paths (`try_metal2vulkan_draw`,
    // `metal_draw::seed_color_load`), neither of which consults the write
    // witness. One boot held 397 entries over 105 MiB with 393/397 of their
    // backing pages unmapped — serving pixels for guest pages that are gone.
    // What the guest's statement of ownership is for a surface whose pages it
    // has unmapped is not known.
    //
    // Guest-CPU-produced linear textures (wallpaper, glyph atlases) have no
    // host producer generation. Re-read the native rows and byte-compare
    // against the memo: unchanged content reuses the retained swizzled Arc
    // and carries a generation identity so the engine skips hash+memcmp too.
    if let Some((rgba, identity, byte_format)) =
        load_linear_guest_memoized(state, host, task_id, tex, gva, w, h)
    {
        note_guest_rung_blank(
            state,
            host,
            task_id,
            texture_ref,
            (gva, w, h),
            &rgba,
            byte_format,
        );
        return Some((w, h, rgba, identity, byte_format));
    }
    // There is deliberately no second guest rung under the memo. One used to
    // re-read through `load_linear_texture_native_host` when the memo declined
    // and never ran: every `None` the memo returns is a decode, geometry,
    // bounds or guest-read failure that the re-read meets on the same
    // descriptor and the same pages.
    //
    // This counter takes its place because `load_linear_guest_memoized` emits
    // on none of its refusal paths, so the deleted rung's decline was the only
    // line that ever named one. A sample refused here falls to
    // `load_sampled_rgba_static` and then to the caller's typed
    // `TextureResolveMissing` — visible, but without the memo's own reason.
    // `lin_rung_memo_declined` against `lin_rung_guest_memo` says whether that
    // gap is worth closing; while it reads zero there is nothing to name.
    crate::runtime::drain::note_store_route("lin_rung_memo_declined");
    None
}

/// Guest pages the Vulkan draw's single synchronous GVA Store may write.
///
/// The Vulkan rail writes back only color attachment 0, so this narrows
/// [`sync_store_target_pages`] to that record and to the case where this record
/// owns guest writeback at all.
pub(super) fn sync_store_allowed_pages<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    color0: Option<&ColorRtRequest>,
    writeback_guest: bool,
) -> Option<std::collections::HashSet<u64>> {
    if !writeback_guest {
        return None;
    }
    sync_store_target_pages(state, host, task_id, color0?)
}

/// How much surface a taken type-11 seed elision covered, in whole texels.
///
/// Measured to price the repair for an unsound witness: the epoch could not see
/// a guest CPU write to the surface's own pages, and the obvious fix was the one
/// the sibling linear rail uses — re-read the guest's bytes and compare before
/// trusting the resident. Whether that was affordable was entirely a question of
/// how much memory the elisions actually cover, and the elision *count* cannot
/// answer it: 8367 elisions a round is either 130 MB of re-reading or 130 KB
/// depending on a distribution nobody had measured.
///
/// So this buckets by extent rather than counting again. A population dominated
/// by icon-sized attachments can be revalidated for almost nothing; one
/// dominated by display-sized composites cannot, and would need the repair to
/// be scoped to the surfaces the guest can actually write.
///
/// Buckets are texel counts at the powers that separate the shapes this device
/// deals in: a 64x64 icon is 4096, a 256x256 thumbnail 65536, a 1920x1080
/// composite 2073600.
///
/// # What it measured, and what it rules out
///
/// One driven boot, x86 / Vulkan, six Finder recomposites:
///
/// ```text
/// type11_seed_elided      41389      t11elide_le_64x64          5
/// type11_seed_uploaded      242      t11elide_le_256x256        4
///                                    t11elide_le_512x512      903
///                                    t11elide_le_1024x1024  15641
///                                    t11elide_display       24836
/// t11elide_texels   59 377 325 642   mean 1 434 616 texels/elision
/// ```
///
/// Two things follow, and they point in opposite directions.
///
/// **An unconditional revalidation is not affordable.** At RGBA8 the elided
/// extent is ~237 GB of guest reads per session. The rail is not a micro-
/// optimisation to be traded away for correctness; it is carrying essentially
/// all of the composite seed traffic, and `type11_seed_uploaded` at 242 against
/// 41 389 says what the un-elided rate would be.
///
/// **The latch is not on the icon target.** Nine elisions in the entire session
/// covered 256x256 or less, and five covered 64x64 or less — so a 64x64 icon
/// attachment is essentially never the surface being elided. The rail that holds
/// the broken cell is the *display-sized composite*: its resident carries a bad
/// region, every later damage draw loads from it and preserves everything it did
/// not cover, and the region stays. That is consistent with every observation in
/// this class, including the small guest damage scissors that made the icon look
/// like a partial draw.
///
/// So the repair could not be "revalidate before trusting", and it could not be
/// "drop the elision". What was missing was a witness for guest CPU writes that
/// does not cost a read of the surface, and the hypervisor already had one.
/// [`type11_guest_wrote_since_store`] is that witness, over
/// `HostOps::guest_write_gen`: O(pages) at the harvest instead of O(bytes) at
/// every LOAD, with the 237 GB left saved. One driven boot after it landed
/// measured `type11_seed_elided` 283 against `type11_seed_uploaded` 23, so the
/// reuse survived the soundness.
fn note_type11_elision_extent(w: u32, h: u32) {
    let texels = (w as u64).saturating_mul(h as u64);
    crate::runtime::drain::note_store_route(match texels {
        0..=4_096 => "t11elide_le_64x64",
        4_097..=65_536 => "t11elide_le_256x256",
        65_537..=262_144 => "t11elide_le_512x512",
        262_145..=1_048_576 => "t11elide_le_1024x1024",
        _ => "t11elide_display",
    });
    // The bytes, so the buckets can be priced without assuming a distribution
    // inside each one. RGBA8 is the seed's own upload format.
    crate::runtime::drain::note_store_route_n("t11elide_texels", texels);
}

/// Census: does this draw's scissor cover the target the pass declared, and if
/// not, what happened to the texels outside it?
///
/// A partial scissor is completely ordinary, so this is a rate with its
/// denominator and not a refusal. What makes it readable is the split by load
/// action, because a partial draw is correct exactly when the attachment
/// already holds the rest of the picture — by a LOAD whose CPU seed supplied
/// it, or by one taking the `chain_load_from_target` arm where the engine
/// resident already holds the prior frame. Only LOAD with neither destroys
/// what it did not draw, and `load_seed_lost` cannot see that case: it counts
/// doors opened and found empty, and such a pass never enters the seed block.
///
/// One driven boot of ten Finder recomposites, x86 / Vulkan:
///
/// ```text
/// draw_scissor_full             1112576
/// draw_scissor_partial           589283
///   draw_partial_load_from_target 517247
///   draw_partial_clear              48303
///   draw_partial_dontcare           23728
///   draw_partial_load_seeded             5
///   draw_partial_load_unseeded           0
/// ```
///
/// **Zero.** No partial draw on this pathway destroys what it did not cover.
/// That is the invariant this census exists to hold, and the reason it stays
/// on: it is one counter, per second, and it is the denominator any future
/// claim about this rail needs.
///
/// # It is not a lead on the broken-cell class, and was once misread as one
///
/// The scissor was the last surviving suspect for the Finder icon defect —
/// a broken cell is a small block of content inside an otherwise empty square,
/// and a scissor smaller than the target is the one mechanism here that can
/// leave part of an attachment untouched by construction. Two readings retire
/// it. The rect is the guest's own, decoded verbatim from `OPCODE_SET_SCISSOR`
/// (`decode::render`, four u64 fields) and latched only when both extents are
/// non-zero, so this device computes, clamps and derives nothing: a 12x40
/// scissor over a 64x64 icon is the compositor's damage rect faithfully
/// carried. And a photograph at pixel scale shows the corrupt cell is not
/// partially drawn at all — over the 11x25 screen rect the clean round renders
/// the folder's light blue (`75D0FB`, `6AC7F4`, `BFDBE8`) and the corrupt one
/// is greyscale end to end (`FFFFFF` 133, `000000` 21, `FCFCFC` 14, `B0B0B0`
/// 5, `505050` 5, `040404` 5), every dominant value R == G == B, no blue
/// anywhere. Nothing was emptied. Something wrote black over the whole cell.
///
/// Scored per round across the break, no counter in this crate separates a
/// corrupt round from a clean one — `lin_rung_guest_blank`,
/// `lin_rung_blank_with_host_entry`, `lin_rung_guest_memo`, `gvac_suspect`,
/// `type11_seed_elided` and `draw_partial_load_from_target` are all simply
/// proportional to round length, and the set of decline names is identical on
/// both sides. A defect that is stable on screen for minutes and leaves no
/// trace in a census this large will not be found by adding another counter to
/// these rails; the next instrument has to observe surface *content* across the
/// transition, not the routes taken to produce it.
#[allow(
    clippy::too_many_arguments,
    reason = "the census joins the scissor rect, the target, and how the attachment was loaded"
)]
fn note_draw_coverage(
    x: u32,
    y: u32,
    sw: u32,
    sh: u32,
    target_w: u32,
    target_h: u32,
    load_action: Option<u16>,
    seeded: bool,
    from_target: bool,
) {
    let covers = x == 0 && y == 0 && sw >= target_w && sh >= target_h;
    crate::runtime::drain::note_store_route(if covers {
        "draw_scissor_full"
    } else {
        "draw_scissor_partial"
    });
    // Into the union before the early return, and clamped to the target: a
    // full-coverage draw is exactly the case that makes a pass's union total,
    // so leaving it out would measure only the passes that were already cheap.
    note_pass_scissor_rect(
        x.min(target_w),
        y.min(target_h),
        sw.min(target_w),
        sh.min(target_h),
    );
    if covers || target_w == 0 || target_h == 0 {
        return;
    }
    // `from_target` is load-bearing and was missing from the first version of
    // this census, which is why its first reading had to be discarded. A LOAD
    // whose prior content lives in the engine resident takes the
    // `chain_load_from_target` arm above: it deliberately resolves no CPU seed,
    // sets `LoadOp::LoadFromTarget`, and preserves the attachment. Scoring
    // `target_rgba8.is_some()` alone put every one of those in the unseeded
    // bucket and produced a 519 715-strong "defect" that was the rail working.
    //
    //   load_seeded   — LOAD and a seed was resolved. The rest is the old frame.
    //   load_unseeded — LOAD and no seed. Becomes a Vulkan CLEAR, so every texel
    //                   outside the scissor is destroyed. The one arm that is a
    //                   defect, and the one `load_seed_lost` cannot count.
    //   clear         — CLEAR was asked for. Destroying them is the contract.
    //   dontcare      — undefined outside the scissor by declaration.
    crate::runtime::drain::note_store_route(match load_action {
        Some(PASS_LOAD_ACTION_LOAD) if from_target => "draw_partial_load_from_target",
        Some(PASS_LOAD_ACTION_LOAD) if seeded => "draw_partial_load_seeded",
        Some(PASS_LOAD_ACTION_LOAD) => "draw_partial_load_unseeded",
        Some(PASS_LOAD_ACTION_CLEAR) => "draw_partial_clear",
        Some(PASS_LOAD_ACTION_DONT_CARE) => "draw_partial_dontcare",
        _ => "draw_partial_load_unknown",
    });
    // How much of the surface this draw's scissor actually covers.
    //
    // The deferred render flush copies the whole attachment on every landing,
    // and that copy is the largest single cost in the device. Whether the
    // guest's own scissors could bound it turns on a number nothing measures:
    // if a partial draw typically covers most of the surface, the union over a
    // pass is near-total and there is nothing to win, and the far more invasive
    // per-pass union accumulator need never be built. These buckets answer that
    // cheaply, from inputs this function already has.
    //
    // Per draw, so it bounds the union from below rather than giving it: a pass
    // is ~3 draws, so a union is at most the sum of its members and at least the
    // largest. Both bounds come from this distribution.
    //
    // Read on a clean driven x86/Vulkan boot (30 s Safari drag, two web-content
    // probe runs), over 65 397 partial draws against 67 729 full-coverage ones:
    //
    // ```text
    // draw_scissor_area_lt1    18 167   27.8 %
    // draw_scissor_area_le5    14 353   21.9 %   (cumulative <=5 %:  49.7 %)
    // draw_scissor_area_le10      513    0.8 %
    // draw_scissor_area_le25    9 029   13.8 %   (cumulative <=25 %: 64.3 %)
    // draw_scissor_area_le50   23 101   35.3 %
    // draw_scissor_area_gt50      234    0.4 %
    // ```
    //
    // The idea is not dead: essentially nothing (0.4 %) covers more than half
    // the surface, and two thirds cover a quarter or less. But it is not
    // confirmed either, and the reason is the *other* counter — **51 % of all
    // draws are full-coverage**, and a pass containing one has a union of 100 %
    // and saves nothing. Whether that matters turns entirely on whether full
    // and partial draws mix within a pass or segregate into whole passes, which
    // a per-draw census cannot see.
    //
    // So the next measurement is the per-pass union itself, and it is worth the
    // plumbing this census was written to avoid paying blind. Take it at the
    // arm point rather than at the flush: the fraction a pass drew is known
    // when the window is armed, and asking there needs no state that has to
    // survive until the deferred landing.
    let area = (sw as u64).saturating_mul(sh as u64);
    let full = (target_w as u64).saturating_mul(target_h as u64);
    let pct = area.saturating_mul(100) / full.max(1);
    crate::runtime::drain::note_store_route(DRAW_AREA_SLUGS[coverage_band(pct)]);
}

/// Which coverage band a percentage falls in, as an index.
///
/// Shared by the per-draw census and the per-pass union so the two are read
/// against the same boundaries. Band 0 is *under* one percent rather than
/// exactly one, because the percentage is integer-truncated.
/// The band function, for `exec`'s pass-extent census to compare against.
///
/// That census declares its own copy because it runs on every backend and this
/// module is behind `backend-vulkan`; the test that calls this is what stops the
/// two from drifting.
#[cfg(test)]
pub(crate) fn coverage_band_for_test(pct: u64) -> usize {
    coverage_band(pct)
}

fn coverage_band(pct: u64) -> usize {
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

/// Per-draw slugs. A draw that reaches this is partial by construction, so the
/// top two bands are one `gt50` bucket rather than the union's `le99`/`full`
/// split — "this draw covered everything" is already `draw_scissor_full`.
const DRAW_AREA_SLUGS: [&str; 7] = [
    "draw_scissor_area_lt1",
    "draw_scissor_area_le5",
    "draw_scissor_area_le10",
    "draw_scissor_area_le25",
    "draw_scissor_area_le50",
    "draw_scissor_area_gt50",
    "draw_scissor_area_gt50",
];

/// Per-pass slugs. `full` is split out from `le99` because a union of 100 % is
/// the answer that decides the question: it is a window where bounding the
/// flush by the guest's scissors would save nothing at all.
const PASS_UNION_SLUGS: [&str; 7] = [
    "pass_scissor_union_lt1",
    "pass_scissor_union_le5",
    "pass_scissor_union_le10",
    "pass_scissor_union_le25",
    "pass_scissor_union_le50",
    "pass_scissor_union_le99",
    "pass_scissor_union_full",
];

/// Union of the scissor rects drawn since the last render window was armed,
/// packed as four 16-bit fields `x0 | y0<<16 | x1<<32 | y1<<48`.
///
/// `u64::MAX` is the empty sentinel, which is unambiguous because `x0 > x1`
/// there and a real union always has `x0 <= x1`. Sixteen bits per field is the
/// contract's own bound: `MAX_SCANOUT_DIM` is 8 192, so a coordinate cannot
/// reach 16 bits and no clamping is needed.
///
/// A static rather than device state, for the same reason `note_store_route` is
/// one: this is a census on the encode path, and threading a field through
/// `exec`'s pass loop into the draw encoder to hold four numbers would put
/// plumbing in the product path for an instrument.
///
/// **The window is "since the last arm", not "this pass".** Resetting at the
/// arm rather than at a pass boundary is what makes it self-synchronising - it
/// needs no hook in `exec` and cannot drift out of step with the thing it is
/// measuring. The cost is that a pass which never arms a window folds its draws
/// into the next union, so this **over**-estimates coverage. That is the safe
/// direction: it under-states how much a bounded flush would save, so a
/// promising reading here is not an artifact of the instrument.
static PASS_SCISSOR_UNION: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(u64::MAX);

/// Fold one draw's rect into the union above.
fn note_pass_scissor_rect(x: u32, y: u32, w: u32, h: u32) {
    use std::sync::atomic::Ordering;
    let (x0, y0) = (x.min(u16::MAX as u32), y.min(u16::MAX as u32));
    let x1 = x.saturating_add(w).min(u16::MAX as u32);
    let y1 = y.saturating_add(h).min(u16::MAX as u32);
    let mut cur = PASS_SCISSOR_UNION.load(Ordering::Relaxed);
    loop {
        let next = if cur == u64::MAX {
            pack_rect(x0, y0, x1, y1)
        } else {
            let (cx0, cy0, cx1, cy1) = unpack_rect(cur);
            pack_rect(cx0.min(x0), cy0.min(y0), cx1.max(x1), cy1.max(y1))
        };
        match PASS_SCISSOR_UNION.compare_exchange_weak(
            cur,
            next,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => return,
            Err(seen) => cur = seen,
        }
    }
}

fn pack_rect(x0: u32, y0: u32, x1: u32, y1: u32) -> u64 {
    (x0 as u64) | ((y0 as u64) << 16) | ((x1 as u64) << 32) | ((y1 as u64) << 48)
}

fn unpack_rect(v: u64) -> (u32, u32, u32, u32) {
    (
        (v & 0xffff) as u32,
        ((v >> 16) & 0xffff) as u32,
        ((v >> 32) & 0xffff) as u32,
        ((v >> 48) & 0xffff) as u32,
    )
}

/// Report the union of the scissors drawn into this window, and reset it.
///
/// Called where a deferred render window is armed, because that is where the
/// question is answerable: the flush that lands this window will copy
/// `width * height` texels, and this says how many of them the pass that
/// produced it actually drew. A union of 100 % means bounding the copy by the
/// guest's own scissors would save nothing for this window.
///
/// # It has been read, and the answer is no
///
/// A clean driven x86/Vulkan boot - 30 s Safari drag, two web-content probe
/// runs, gate clean - over 17 696 armed windows:
///
/// ```text
/// pass_scissor_union_lt1        0
/// pass_scissor_union_le5       11
/// pass_scissor_union_le10       2
/// pass_scissor_union_le25       0
/// pass_scissor_union_le50       0
/// pass_scissor_union_le99       1
/// pass_scissor_union_full  17 682     99.92 %
/// ```
///
/// **Bounding the deferred render flush by the guest's own scissor rects would
/// save nothing.** 99.92 % of windows come from a pass that drew the whole
/// surface, so there is no smaller extent to copy. Do not build a damage-rect
/// flush on the strength of the per-draw distribution above; this is the number
/// that governs, and the two are consistent rather than in conflict.
///
/// The instrument agrees with its own inputs, which is why the reading is
/// trustworthy: the boot runs 7.9 draws per armed window and 50 % of all draws
/// are full-coverage, so a window escapes saturation only if every one of those
/// draws is partial - `0.5^7.9` is 0.4 %, predicting ~99.6 % against the 99.92 %
/// measured. The residual is draws clustering rather than being independent.
///
/// **99.92 % is the per-*window* figure and it overstates the per-*pass* one.**
/// This counter resets at the arm, and a render pass is ~3 draws, so each
/// reading unions roughly 2.6 passes rather than one - the over-estimate the
/// section above declares. Correcting for it does not change the answer: at 3
/// draws a pass and the same 50 % full-coverage rate, `0.5^3` leaves ~87 % of
/// true passes still drawing their whole surface. The honest claim is therefore
/// "~87 % of passes, 99.92 % of the windows a flush actually lands", and a
/// damage-bounded flush would have to beat the second number, because the
/// window is what gets copied.
///
/// **And the bounding box is not what kills it.** A union is a rectangle, so
/// two small disjoint rects produce a large one, and a richer damage
/// representation - a rect list, tiles - would score better here. It would not
/// help: half of all individual draws cover the entire attachment on their own,
/// and no representation makes a full-surface draw smaller. The saving is not
/// hiding behind the approximation.
///
/// What this does not close is the rail's cost, which is real and is still the
/// largest in the device. It closes one candidate repair. `flush_render_one`
/// names the others.
pub(super) fn note_pass_scissor_union(width: u32, height: u32) {
    use std::sync::atomic::Ordering;
    let packed = PASS_SCISSOR_UNION.swap(u64::MAX, Ordering::Relaxed);
    if packed == u64::MAX || width == 0 || height == 0 {
        return;
    }
    let (x0, y0, x1, y1) = unpack_rect(packed);
    let union_area = (x1.saturating_sub(x0) as u64).saturating_mul(y1.saturating_sub(y0) as u64);
    let full = (width as u64).saturating_mul(height as u64);
    // The union is clamped to the surface: a scissor may legitimately exceed the
    // attachment (Metal permits it; the rasteriser clips), and an unclamped
    // ratio would then read over 100 % and make the census unreadable.
    let pct = union_area.min(full).saturating_mul(100) / full.max(1);
    crate::runtime::drain::note_store_route(PASS_UNION_SLUGS[coverage_band(pct)]);
}

/// Score a `MTLLoadActionLoad` colour attachment against whether a seed was
/// actually found for it.
///
/// A LOAD says the guest is drawing **onto the content already in this
/// attachment**. When no seed is produced the attachment starts undefined, so
/// every texel this pass does not itself draw is lost — which is a rectangle
/// of a compositing layer going blank while the freshly drawn geometry
/// survives, held until something redraws the whole layer. That is a real loss
/// of guest work and belongs in the failure log, not in silence.
///
/// `chain_load_from_target` (the resident target already carries the chain) is
/// a different arm and never reaches here: it is a seed that does not need
/// uploading, not a seed that is missing.
fn note_load_seed_outcome(
    door: &'static str,
    seeded: bool,
    c0: &crate::runtime::metal_draw::ColorRtRequest,
    w: u32,
    h: u32,
) {
    crate::runtime::drain::note_store_route(if seeded {
        "load_seed_ok"
    } else {
        "load_seed_lost"
    });
    // Per door, because the doors fail for unrelated reasons and a pooled zero
    // hides a door that never opens.
    //
    // Measured, and the split earns its keep by refuting what it was added to
    // test. One driven boot, ten Finder recomposites, two of them corrupt:
    //
    //   load_seed_ok 395   ok_mapping 346   ok_color 49
    //   ok_gva_or_ref 0    lost_gva_or_ref 0    lost_<any door> 0
    //
    // `gva_or_ref` was the suspect — it drew its seed from the two host caches
    // the sampled ladder found nearly always empty for these spans — and it was
    // never taken. It is now gone; see the LOAD arm for why it could not be.
    // Nor does any door lose a seed, so the earlier pooled `load_seed_lost=0`
    // was a real zero and not a door standing idle. A lost seed turning LOAD
    // into CLEAR is therefore NOT how a broken icon gets its empty square, and
    // the whole rail is small besides: 395 seed resolutions across a boot that
    // composited ten Finder windows.
    //
    // A second driven boot (Safari page load, title-bar drag, page-down) agrees
    // and narrows it further: ok 295 = ok_color 152 + ok_mapping 143. A third
    // door, `req_seed`, read a whole-request copy of the same seed the
    // `color_seed` door reads and measured 0 in every boot; it was 0 because
    // the request-level copy could only be `Some` when color0's already was,
    // and the door sat behind color0's. It is gone with the copy.
    crate::runtime::drain::note_store_route(match (door, seeded) {
        ("color_seed", true) => "load_seed_ok_color",
        ("color_seed", false) => "load_seed_lost_color",
        ("mapping", true) => "load_seed_ok_mapping",
        ("mapping", false) => "load_seed_lost_mapping",
        (_, true) => "load_seed_ok_other",
        (_, false) => "load_seed_lost_other",
    });
    if seeded {
        return;
    }
    // Deduplicated per (door, target) so a compositor re-running the same lost
    // pass hundreds of times a second reports the target once.
    if crate::observe::first_sight(
        door,
        c0.target_gva ^ ((c0.texture_ref as u64) << 40) ^ ((c0.mapping_id as u64) << 20),
    ) {
        crate::observe::fail(format!(
            "load_seed_lost door={door} gva={:#x} ref={} mid={} {w}x{h} bpr={} fmt={:#x}",
            c0.target_gva, c0.texture_ref, c0.mapping_id, c0.row_stride, c0.format
        ));
    }
}

#[inline]
fn gva_cache_linear_texture_type(object_type: u8) -> bool {
    matches!(
        object_type,
        OBJECT_TYPE_TEXTURE | OBJECT_TYPE_TEXTURE_VARIANT
    )
}

/// A GVA cache is keyed by decoded linear texture storage. Type-2 and type-3
/// wrappers may alias the same GVA allocation, so a matching GVA+geometry cache
/// entry can serve either tag. Other nonzero object-type transitions remain
/// separate resource classes and fall through to current ref/guest backing.
#[inline]
pub(super) fn gva_cache_owner_allows_object_type(producer_type: u8, current_type: u8) -> bool {
    producer_type == 0
        || current_type == 0
        || producer_type == current_type
        || (gva_cache_linear_texture_type(producer_type)
            && gva_cache_linear_texture_type(current_type))
}

/// Store type-2/3 encode into texture_ref + GVA host caches (BGRA).
///
/// `task_id` is the task whose page table gives the GVA meaning; the store
/// records the pages it resolves to so a later sample can tell whether the
/// address still names this allocation, and asks the host to watch them so a
/// later sample can also tell whether the guest has since rewritten it.
#[allow(
    clippy::too_many_arguments,
    reason = "the cache identity mirrors the object, GVA, texture geometry, and guest backing"
)]
pub(crate) fn host_cache_store_gva_layer<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    object_type: u8,
    gva: u64,
    width: u32,
    height: u32,
    rgba: &[u8],
) {
    if width == 0 || height == 0 {
        return;
    }
    let need = (width as usize)
        .saturating_mul(height as usize)
        .saturating_mul(4);
    if rgba.len() < need {
        return;
    }
    let bgra = swap_rb_channels(&rgba[..need]);
    if texture_ref != 0 {
        crate::runtime::surface_cache::store_texture(
            state,
            texture_ref,
            width,
            height,
            bgra.clone(),
        );
    }
    if gva != 0 {
        let backing =
            crate::runtime::surface_cache::gva_backing(state, host, task_id, gva, width, height);
        crate::runtime::surface_cache::store_gva_owned(
            state,
            gva,
            width,
            height,
            bgra,
            object_type,
            backing,
        );
    }
}

/// Result of a Linux metal2vulkan draw.
enum M2vDrawSpan {
    /// No drawable color0 geom.
    None,
    /// CPU-side pixels (readback path), in the order the engine reports.
    ///
    /// The order is carried rather than normalized because a type-11 composite
    /// Store's consumers — `surface_cache`, the deferred window, the guest-page
    /// writeback — all want guest scanout order, and a BGRA resident hands them
    /// exactly that. Normalizing to RGBA here would restate a whole framebuffer
    /// per Store purely to have the Store restate it back.
    Pixels { bytes: Vec<u8>, bgra: bool },
    /// Intermediate record of a resident render-pass chain: content stays on
    /// the protocol-keyed engine target (no CPU pixels, no fence wait, no guest
    /// Store this record). The final record reads back and performs the
    /// contract Store on portability devices.
    ResidentChain,
    /// Final/single record of a GVA render Store executed into the registry
    /// resident with `skip_readback`: the caller arms a deferred-writeback
    /// window (`DeviceState::gva_deferred_flush`) instead of the sync
    /// readback + guest write on the stamp path; guest bytes + encode caches
    /// land on first access (`storage_flush::flush_gva_one`).
    ResidentGvaStore,
    /// Type-11 composite Store executed into its registry resident with
    /// `skip_readback`: the caller arms a `RenderWindowSource::Resident` window
    /// naming that image, so the GPU→host readback and the fence wait it implies
    /// are paid only if a guest-side reader ever asks for the pixels
    /// (`storage_flush::flush_render_one`).
    ///
    /// Distinct from [`Self::ResidentGvaStore`] because the two windows live in
    /// different indexes and flush through different readers: this one is keyed by
    /// mapping in `compute_deferred_flush`, that one by GVA in
    /// `gva_deferred_flush`. Distinct from [`Self::Pixels`] because there are no
    /// pixels — a caller that treated an empty frame as one would write a blank
    /// framebuffer into guest memory.
    ResidentSurfaceStore,
}

/// Name the guest-Store route this record actually took, once per distinct
/// route per process.
///
/// Every other record of these branches is `observe::line`, which writes
/// nothing unless `REIMS_VGPU_DRAW_LOG=1` — the only always-on `linux_m2v_store`
/// arm is the CPU write *failure*. So an always-on log could not tell "the CPU
/// Store ran" from "no Store happened at all", which is the branch-vs-arm hole:
/// a probe placed inside one of these arms cannot separate "the condition was
/// false" from "the outcome never occurred". This line names the branch itself,
/// at the point the branch is taken, so a zero for one route is readable
/// against the other routes' presence.
///
/// The dedup key is the route, so this is bounded at one line per outcome per
/// process — after the first record of each kind it costs a `BTreeSet` lookup
/// and a return, which is what makes it safe to leave on permanently.
///
/// Reachability is not uniform and must be read that way. `import` requires the
/// engine to have enabled a host-pointer import, and there is no longer any code
/// that could: the whole `VK_EXT_external_memory_host` subsystem is deleted. So `import` is
/// unreachable, and `rgba_not_import` is its complement's complement — with the
/// import never allowed, `type11_cpu_store_fallback_allowed` is always true and
/// that arm cannot be entered either. Both are kept as call sites so their
/// absence is a *denominator* against the routes that do fire, not an
/// acquittal; if either ever appears, the extension came back.
/// The first-appearance line answers "is this route reachable" and cannot answer
/// "how often". Both questions are live: reachability is what the denominator
/// argument above needs, and the rate is what prices the route — `engine_delta`
/// shows ~20 full-frame readbacks a second and the routes are what attribute
/// them. So the dedup'd line stays and the rate is counted alongside it, into
/// the same one-second window as `drain_duty`.
/// Accumulates its own lifetime into the per-second `store_routes` window.
///
/// A guard rather than a pair of `Instant` reads because the block it brackets
/// `return`s out of its own middle on the deferred route — the measurement that
/// matters most — and a hand-closed bracket there records nothing while looking
/// exactly like one that does. Reporting on `Drop` makes every exit pay.
struct StoreCostSpan {
    name: &'static str,
    started: std::time::Instant,
}

impl StoreCostSpan {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            started: std::time::Instant::now(),
        }
    }
}

impl Drop for StoreCostSpan {
    fn drop(&mut self) {
        crate::runtime::drain::note_store_route_us(
            self.name,
            self.started.elapsed().as_micros() as u64,
        );
    }
}

/// Name which of the six routes a type-11 Store took: counted every time, and
/// fail-logged once per route per process so a boot's route *set* is readable
/// without the draw log.
///
/// # Four of the six read zero, and deleting them would break the support matrix
///
/// Across the whole accumulated fail log — 39 boots — exactly two routes have
/// ever been taken, and both are taken in every single boot:
///
/// ```text
/// surface_resident       39   gva_deferred_sync       0
/// gva_deferred           39   surface_resident_sync   0
///                             surface_deferred        0
///                             cpu_portability         0
/// ```
///
/// A live denominator with four dead arms is normally the strongest deletion
/// signal this crate has. Here it is a trap, for two separate reasons, and both
/// have to be answered before touching any of them.
///
/// **The `_sync` pair is arm-refusal recovery.** Each fires when its arm gate
/// declines, and then reads the resident back and lands the frame synchronously.
/// Zero means the arms have never refused on this host — not that refusal is
/// impossible. They are the reason a refusal costs a readback instead of a lost
/// frame.
///
/// **`surface_deferred` and `cpu_portability` are host-class cells, not
/// workload outcomes.** They sit in the synchronous CPU-readback block that the
/// `_sync` routes fall through to, and which of the two runs is decided by
/// [`crate::backend::vulkan::engine::deferred_gpu_only_content_allowed`] — a **capability** gate,
/// held back by the `guest_pages_stay_authoritative` driver quirk. On a host
/// where that quirk applies, deferral is off, `surface_deferred` cannot be taken
/// and `cpu_portability` carries every Store. That is the "guest pages stay
/// authoritative" cell of the support matrix in `AGENTS.md`, which also requires
/// the discrete-GPU and no-DMA cells this same block serves.
///
/// So these zeros say only: *this* host is in the class where deferral is
/// permitted and the arms succeed. `AGENTS.md` forbids generalising from one
/// host GPU class to another, and this is exactly that boundary. Measuring these
/// four needs a host whose quirk set turns deferral off, or one where the pin
/// refuses — not another boot here.
fn note_type11_store_route(route: &'static str) {
    use std::sync::Mutex;
    static SEEN: Mutex<Option<std::collections::BTreeSet<&'static str>>> = Mutex::new(None);
    crate::runtime::drain::note_store_route(route);
    {
        let mut guard = SEEN.lock().unwrap_or_else(|p| p.into_inner());
        if !guard.get_or_insert_with(Default::default).insert(route) {
            return;
        }
    }
    crate::observe::fail(format!("type11_store_route route={route}"));
}

/// Build the engine's secondary MRT attachments (slot 1..) from a draw's color
/// list. Empty result ⇒ the classic single-RT path (no regression). A fragment
/// shader that writes `location` 1.. has those outputs rendered rather than
/// discarded; each secondary persists as a registry resident keyed by its
/// protocol identity, exactly as slot 0 does.
///
/// Conservative by construction — any ambiguity yields an empty vector rather
/// than a guessed attachment: requires a resident primary, contiguous slots
/// (0,1,2,… matching the shader's `location`s), matching framebuffer geometry,
/// a known color-renderable format, and a resolvable identity.
#[allow(
    clippy::too_many_arguments,
    reason = "every argument is a distinct wire-derived input to the attachment set"
)]
pub(super) fn build_secondary_targets<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &mut M,
    task_id: u32,
    colors: &[ColorRtRequest],
    pipeline: &crate::runtime::decode::resource::RenderPipelineDescriptor,
    primary: &crate::backend::vulkan::engine::TargetIdentity,
    fb_w: u32,
    fb_h: u32,
    blend_constants: [f32; 4],
) -> Vec<crate::backend::vulkan::engine::SecondaryColorTarget> {
    use crate::backend::vulkan::engine::{SecondaryColorTarget, TargetIdentity};
    if colors.len() <= 1 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (i, c) in colors.iter().enumerate().skip(1) {
        // Contiguous slots only — the render pass maps location N → attachment N,
        // so a gap would misalign the shader's outputs.
        if c.slot as usize != i || c.texture_ref == 0 {
            crate::runtime::census::present_proxy::note_secondary_mrt_drop(
                crate::runtime::census::present_proxy::MrtDrop::NonContiguousSlot,
                c.width,
                c.height,
            );
            return Vec::new();
        }
        // MRT requires every attachment to share the framebuffer geometry.
        if c.width != fb_w || c.height != fb_h {
            crate::runtime::census::present_proxy::note_secondary_mrt_drop(
                crate::runtime::census::present_proxy::MrtDrop::GeometryMismatch,
                c.width,
                c.height,
            );
            return Vec::new();
        }
        // Unknown wire format stays unknown — never guess a secondary layout —
        // and a known format whose sRGB qualifier this attachment cannot carry
        // says so instead of folding silently.
        let format = match translate::pixel::color_attachment(c.format) {
            Ok((format, decline)) => {
                if decline.is_some() {
                    srgb_census::note_downgrade(
                        srgb_census::site::SECONDARY_COLOR_TARGET,
                        c.format,
                    );
                }
                format
            }
            Err(_) => {
                crate::runtime::census::present_proxy::note_secondary_mrt_drop(
                    crate::runtime::census::present_proxy::MrtDrop::UnknownFormat,
                    c.width,
                    c.height,
                );
                return Vec::new();
            }
        };
        // Identity mirrors the primary namespaces: type-2/3 linear GVA, else
        // type-11 surface.
        //
        // A secondary GVA is named by its own backing pages, exactly like color0
        // — the primary's generation describes a different address (a secondary
        // equal to the primary is rejected above), so it takes its own walk.
        // Without one this attachment is keyed on `(gva, width, height)` alone
        // and two guest allocations reusing that address at that geometry share
        // one GPU image — the wrong-content class `74748d2` closed for color0.
        let identity = if c.target_gva != 0 {
            TargetIdentity::Gva {
                gva: c.target_gva,
                width: c.width,
                height: c.height,
                generation: gva_span_alloc_generation(
                    state,
                    host,
                    task_id,
                    c.target_gva,
                    c.row_stride,
                    c.height,
                ),
            }
        } else if c.mapping_id != 0 {
            crate::runtime::present_identity::surface_identity(
                state,
                c.mapping_id,
                c.width,
                c.height,
            )
        } else {
            crate::runtime::census::present_proxy::note_secondary_mrt_drop(
                crate::runtime::census::present_proxy::MrtDrop::NoIdentity,
                c.width,
                c.height,
            );
            return Vec::new();
        };
        // A secondary aliasing the primary target is a degenerate feedback loop
        // the engine rejects — bail to the safe single-RT path.
        if identity == *primary {
            crate::runtime::census::present_proxy::note_secondary_mrt_drop(
                crate::runtime::census::present_proxy::MrtDrop::AliasesPrimary,
                c.width,
                c.height,
            );
            return Vec::new();
        }
        let load = c.load_action == PASS_LOAD_ACTION_LOAD;
        let clear = [
            c.clear_color[0] as f32,
            c.clear_color[1] as f32,
            c.clear_color[2] as f32,
            c.clear_color[3] as f32,
        ];
        // This slot's own blend, resolved exactly as the Metal arm resolves it:
        // find the pipeline's attachment entry for this Metal slot. No
        // `or_else(first())` fallback here — the Metal path has one for its
        // compat `color0` alias, but a secondary slot with no entry of its own
        // has no blend state, and borrowing slot 0's would be inventing one.
        // The mask is read from the same entry but *not* through the
        // `blending_enabled` filter below: `MTLColorWriteMask` applies whether
        // or not the slot blends, and an entry with no mask means `all`.
        let color_write_mask = pipeline
            .color_attachments
            .iter()
            .find(|a| a.slot == c.slot)
            .map(|a| a.write_mask)
            .unwrap_or_default();
        let blend = pipeline
            .color_attachments
            .iter()
            .find(|a| a.slot == c.slot)
            .filter(|a| a.blending_enabled)
            .and_then(|a| {
                match translate::blend::state(
                    a.src_rgb,
                    a.dst_rgb,
                    a.op_rgb,
                    a.src_alpha,
                    a.dst_alpha,
                    a.op_alpha,
                    blend_constants,
                ) {
                    Ok(state) => Some(state),
                    // An out-of-contract blend factor or op on a secondary
                    // slot: the attachment still renders, unblended, and the
                    // decline says which value refused rather than the slot
                    // quietly becoming a raw store the way every slot used to.
                    Err(reason) => {
                        crate::observe::fail(format!(
                            "secondary_blend_unmapped {reason} slot={} {}x{}",
                            c.slot, c.width, c.height
                        ));
                        None
                    }
                }
            });
        out.push(SecondaryColorTarget {
            identity,
            width: c.width,
            height: c.height,
            format,
            clear,
            load,
            blend,
            color_write_mask,
        });
    }
    out
}

/// Translate guest MTLB stages via metal2vulkan and raster with the internal Vulkan engine.
///
/// Builds engine [`DrawRequest`] resources from stream binds (stage-in attrs, SSBOs,
/// sampled images) — bare `render_offscreen` without binds yields black alpha-only
/// frames that wipe CLEAR stores. Archive `render_draw_core` is the contract model.
///
/// Type-11 Stores return [`M2vDrawSpan::ResidentBgra`] for zero-copy import
/// (revalidate + strided host ptr) on backends that can keep guest-visible
/// content resident. Portability-subset devices take the synchronous CPU
/// writeback path so guest pages remain authoritative across device recreates.
pub(super) fn prepare_vertex_attribute_format(
    attribute: &crate::runtime::decode::resource::VertexAttribute,
) -> Result<crate::backend::vulkan::engine::VertexAttributeFormat, DrawPreparationDecline> {
    translate::vertex::attribute_format(attribute.format).map_err(|reason| {
        DrawPreparationDecline::VertexAttributeFormat {
            location: attribute.location,
            buffer_index: attribute.buffer_index,
            raw_format: attribute.format,
            reason,
        }
    })
}

pub(super) fn prepare_vertex_step_function(
    attribute: &crate::runtime::decode::resource::VertexAttribute,
) -> Result<crate::backend::vulkan::engine::VertexStepFunction, DrawPreparationDecline> {
    translate::vertex::step_function(attribute.has_step_function, attribute.step_function).map_err(
        |reason| DrawPreparationDecline::VertexStepFunctionUnsupported {
            location: attribute.location,
            buffer_index: attribute.buffer_index,
            reason,
        },
    )
}

fn try_metal2vulkan_draw<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    req: &mut DrawEncodeRequest,
    writeback_guest: bool,
) -> Result<M2vDrawSpan, DrawError> {
    // Only the final record of a portability render-pass chain reads back CPU
    // pixels; used by the resident-chain rail below (harmless on other paths).
    let _ = &writeback_guest;
    crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::Pipeline);
    // Name the color0 GVA target's allocation before anything can render into
    // it, and once: the pinned Store identity, the cross-pass Load identity and
    // the deferred window's stored copy are all keyed on this value, and two
    // walks of one address across a submit are two answers.
    req.gva_alloc_gen = gva_alloc_generation(state, host, req);
    let pd = load_render_pipeline(state, host, req.task_id, req.pipeline_ref).ok_or({
        DrawError::DrawPreparation(
            crate::backend::vulkan::engine::DrawPreparationDecline::PipelineMissing {
                task_id: req.task_id,
                pipeline_ref: req.pipeline_ref,
            },
        )
    })?;
    let v_mtlb = load_mtlb(state, host, req.task_id, pd.vertex_func_ref).ok_or({
        DrawError::DrawPreparation(
            crate::backend::vulkan::engine::DrawPreparationDecline::VertexMtlbMissing {
                task_id: req.task_id,
                function_ref: pd.vertex_func_ref,
            },
        )
    })?;
    let f_mtlb = load_mtlb(state, host, req.task_id, pd.fragment_func_ref).ok_or({
        DrawError::DrawPreparation(
            crate::backend::vulkan::engine::DrawPreparationDecline::FragmentMtlbMissing {
                task_id: req.task_id,
                function_ref: pd.fragment_func_ref,
            },
        )
    })?;
    // Borrowed from the `*_mtlb` locals above, which outlive every use below.
    // These were `.to_vec()`, which allocated and copied both AIR blobs on
    // every chain — `drain_duty` measures ~1142 chains/s, so that is ~2300
    // allocations a second on the drain worker for bytes that are only ever
    // read (`translate_cached_reflected` takes `&[u8]`, and its cache is keyed
    // by hashing them).
    let v_air = crate::runtime::mtlb::extract_air(&v_mtlb).map_err(|reason| {
        DrawError::DrawPreparation(
            crate::backend::vulkan::engine::DrawPreparationDecline::VertexAirExtract {
                function_ref: pd.vertex_func_ref,
                reason,
            },
        )
    })?;
    let f_air = crate::runtime::mtlb::extract_air(&f_mtlb).map_err(|reason| {
        DrawError::DrawPreparation(
            crate::backend::vulkan::engine::DrawPreparationDecline::FragmentAirExtract {
                function_ref: pd.fragment_func_ref,
                reason,
            },
        )
    })?;

    // AIR→SPIR-V is content-cached: live boots re-translated the same pipelines
    // dozens of times on the doorbell vCPU and tripped IPI timeout panics.
    // Reflected translate: the cached shader carries the metal2vulkan reflection
    // facade so per-draw texture provisioning reads dimensionality straight from
    // the AIR-derived metadata (single source of truth) rather than re-walking the
    // emitted SPIR-V. `_shader.reflection` is used at the sampled-image binding
    // loop below; the SPIR-V walk stays as a cold fallback.
    let v_shader = crate::runtime::m2v_cache::translate_cached_reflected(
        v_air,
        metal2vulkan::passes::Stage::Vertex,
        req.pipeline_ref,
    )
    .map_err(|reason| {
        DrawError::DrawPreparation(
            crate::backend::vulkan::engine::DrawPreparationDecline::VertexTranslate {
                pipeline_ref: req.pipeline_ref,
                reason,
            },
        )
    })?;
    let f_shader = crate::runtime::m2v_cache::translate_cached_reflected(
        f_air,
        metal2vulkan::passes::Stage::Fragment,
        req.pipeline_ref,
    )
    .map_err(|reason| {
        DrawError::DrawPreparation(
            crate::backend::vulkan::engine::DrawPreparationDecline::FragmentTranslate {
                pipeline_ref: req.pipeline_ref,
                reason,
            },
        )
    })?;

    // Decline a relooper state machine before the driver is handed it.
    //
    // `vkCreateGraphicsPipelines` runs on the drain worker with the device lock
    // held, so a driver that does not return does not merely lose this draw --
    // it stops the device. The guest's rings stop being consumed and it reports
    // `GPU hang: Name Display0` with the ring cursors frozen. Measured on an
    // NVIDIA host: the WindowServer compositor's fragment module (2 731 blocks,
    // 2 725 switch cases) held that call past 22 minutes at a full core with a
    // flat working set, while every structured module in the same boot compiled
    // in single-digit milliseconds.
    //
    // Checked here rather than in the engine because both stages' modules are in
    // hand and the decline should name which stage carried the shape.
    // An invalid module must not reach the driver, whichever stage carries it.
    // The translator can emit an `OpCompositeInsert` that puts an image or
    // sampler handle into a struct; the Logical addressing model has no
    // representation for that, `spirv-val` rejects the module, and a driver
    // handed an invalid one may do anything. Measured on the compute path, where
    // it stopped the host process being served three boots running — the render
    // path shares the translator, so it shares the exposure.
    for (stage, shader) in [("vertex", &v_shader), ("fragment", &f_shader)] {
        if shader.shape.opaque_in_composite {
            crate::observe::fail(format!(
                "linux_m2v_draw m2v_invalid_module reason=opaque_handle_in_composite                  pipe={} stage={stage}                  (OpCompositeInsert/Extract of an image or sampler handle;                   spirv-val rejects the module)",
                req.pipeline_ref
            ));
            return Err(DrawError::Unsupported(
                crate::backend::vulkan::engine::reason::DrawReason::InvalidTranslatedModule {
                    pipeline_ref: req.pipeline_ref,
                },
            ));
        }
    }
    for (stage, shader) in [("vertex", &v_shader), ("fragment", &f_shader)] {
        if shader.shape.is_relooper_state_machine() {
            let reason = crate::backend::vulkan::engine::reason::DrawReason::
                UnstructuredStateMachineShader {
                    blocks: shader.shape.blocks,
                    switch_cases: shader.shape.max_switch_cases,
                };
            crate::observe::Emit::decline("linux_m2v_draw", &reason)
                .field("pipe", req.pipeline_ref)
                .field("stage", stage)
                .fail_once(u64::from(req.pipeline_ref));
            return Err(DrawError::Unsupported(reason));
        }
    }

    let (w, h) = if let Some(c0) = req.colors.first() {
        (c0.width, c0.height)
    } else {
        return Ok(M2vDrawSpan::None);
    };
    // The bound is the device's own `maxImageDimension2D`, not a fixed number:
    // a guest driving a 5K or 6K display names render targets past the Vulkan
    // 1.2 floor of 4096, and every desktop GPU reports 16384.
    let max_dim = crate::backend::vulkan::engine::max_render_target_dimension();
    if w == 0 || h == 0 || w > max_dim || h > max_dim {
        return Err(DrawError::DrawPreparation(
            DrawPreparationDecline::GeometryUnsupported {
                width: w,
                height: h,
            },
        ));
    }

    crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::Binds);
    crate::runtime::bind_phase::note_bind();

    // SPIR-V words for the engine, shared from the translation cache (Arc — no
    // per-draw materialization; fragment reloc variants are cached per shader).
    let v_words = v_shader.words.clone();
    #[allow(unused_mut)]
    let mut f_words = f_shader.words.clone();

    {
        use crate::runtime::spirv_bind::{
            FRAG_BUFFER_BINDING_OFFSET, FRAG_SAMPLED_RESOURCE_BINDING_OFFSET, SAMPLER_BINDING_BASE,
            TEXTURE_BINDING_BASE,
        };

        // Materialize stream buffer binds (vertex + fragment). Large spans
        // ride the zero-copy rail (the GPU gathers them from imported guest
        // RAM at execute time); the rest stay on the CPU staging read.
        // Constant-step attribute streams stay CPU: the engine prepends a
        // base-instance prefix to those bytes at prepare time.
        let constant_step_bufs: std::collections::BTreeSet<u32> = pd
            .vertex_attributes
            .iter()
            .filter(|a| {
                a.format != 0 && a.stride != 0 && a.has_step_function && a.step_function == 0
            })
            .map(|a| a.buffer_index)
            .collect();
        let mut vtx_storage: Vec<(u32, crate::backend::vulkan::engine::BufferContent)> = Vec::new();
        // The three `bind_phase` spans below divide `chain_phase`'s `binds_us`,
        // which is this draw path's largest column and covered three costs with
        // one number. Each is a lexical scope so an early `return Err` charges
        // the span it left from rather than losing the time.
        let vertex_span =
            crate::runtime::bind_phase::Span::open(crate::runtime::bind_phase::Part::VertexLoad);
        for b in &req.vertex_buffers {
            if b.index >= MAX_BIND_SLOTS || b.buffer_ref == 0 {
                continue;
            }
            let allow_zc = !constant_step_bufs.contains(&b.index);
            let Some(content) =
                load_buffer_content(state, host, req.task_id, b.buffer_ref, b.offset, allow_zc)
            else {
                return Err(DrawError::DrawPreparation(
                    DrawPreparationDecline::VertexBufferMissing {
                        index: b.index,
                        buffer_ref: b.buffer_ref,
                        offset: b.offset,
                    },
                ));
            };
            vtx_storage.push((b.index, content));
        }
        drop(vertex_span);
        let mut frag_storage: Vec<(u32, crate::backend::vulkan::engine::BufferContent)> =
            Vec::new();
        let fragment_span =
            crate::runtime::bind_phase::Span::open(crate::runtime::bind_phase::Part::FragmentLoad);
        for b in &req.fragment_buffers {
            if b.index >= MAX_BIND_SLOTS || b.buffer_ref == 0 {
                continue;
            }
            let Some(content) =
                load_buffer_content(state, host, req.task_id, b.buffer_ref, b.offset, true)
            else {
                return Err(DrawError::DrawPreparation(
                    DrawPreparationDecline::FragmentBufferMissing {
                        index: b.index,
                        buffer_ref: b.buffer_ref,
                        offset: b.offset,
                    },
                ));
            };
            frag_storage.push((b.index, content));
        }
        drop(fragment_span);
        // Stage-in attributes from pipeline vertex block + bound buffer bytes.
        let mut attrs: Vec<crate::backend::vulkan::engine::VertexAttributeResource> = Vec::new();
        let mut stage_in_bufs: std::collections::BTreeSet<u32> = Default::default();
        let attrs_span =
            crate::runtime::bind_phase::Span::open(crate::runtime::bind_phase::Part::Attrs);
        for a in &pd.vertex_attributes {
            if a.format == 0 || a.stride == 0 {
                continue;
            }
            let format = prepare_vertex_attribute_format(a).map_err(DrawError::DrawPreparation)?;
            let content = vtx_storage
                .iter()
                .find(|(idx, _)| *idx == a.buffer_index)
                .map(|(_, d)| d.clone())
                .unwrap_or_else(|| crate::backend::vulkan::engine::BufferContent::from(Vec::new()));
            if !content.is_empty() {
                stage_in_bufs.insert(a.buffer_index);
            } else if a.format != 0 {
                // Pipeline declares stage-in but stream did not bind bytes — fail
                // visibly rather than raster black garbage that wipes CLEAR.
                return Err(DrawError::DrawPreparation(
                    DrawPreparationDecline::StageInBytesMissing {
                        location: a.location,
                        buffer_index: a.buffer_index,
                        raw_format: a.format,
                        stride: a.stride,
                    },
                ));
            }
            let step = prepare_vertex_step_function(a).map_err(DrawError::DrawPreparation)?;
            let step_rate = if a.has_step_rate {
                a.step_rate.max(1)
            } else {
                1
            };
            attrs.push(crate::backend::vulkan::engine::VertexAttributeResource {
                location: a.location,
                // One Vulkan binding per location (archive render_draw_core).
                binding: a.location,
                format,
                offset: a.offset,
                stride: a.stride,
                step_function: step,
                step_rate,
                content,
            });
        }
        drop(attrs_span);

        // Fragment/vertex buffer index collision → relocate fragment SPIR-V buffers.
        let vtx_idx: std::collections::BTreeSet<u32> =
            vtx_storage.iter().map(|(i, _)| *i).collect();
        let buf_collide = frag_storage.iter().any(|(i, _)| vtx_idx.contains(i));
        let has_vtx_tex = req
            .vertex_textures
            .iter()
            .any(|t| t.index < MAX_BIND_SLOTS && t.texture_ref != 0);
        let has_frag_tex = req
            .fragment_textures
            .iter()
            .any(|t| t.index < MAX_BIND_SLOTS && t.texture_ref != 0);
        let reflected_sampled_collision =
            reflected_sampled_binding_collision(&v_shader.reflection, &f_shader.reflection);
        let separate_sampled =
            (has_vtx_tex && has_frag_tex) || buf_collide || reflected_sampled_collision;
        // Sampled relocation first (archive order), then buffer band. The
        // buffer band lands at [104,136), clear of the [96,104) ColorInput /
        // framebuffer-fetch band, which neither relocation touches. The
        // sampled-with-buffer coupling is kept so the engine's image/sampler
        // binding base mirrors one flag pair, not a third variant.
        if separate_sampled || buf_collide {
            f_words = f_shader.fragment_words(separate_sampled, buf_collide);
        }

        // Non-stage-in vertex buffers + fragment buffers as storage buffers.
        //
        // A vertex buffer can be BOTH a stage-in source (the pipeline vertex
        // descriptor declares attributes on it) AND read directly as a
        // StorageBuffer by the vertex function (`[[buffer(N)]]` -> descriptor
        // binding N). WebKit's glyph vertex shader is exactly this: the pipeline
        // declares a stride-48 stage-in on buffer 1, but the translated SPIR-V
        // never reads a stage-in input — it indexes buffer 1 as a per-glyph
        // record array (StorageBuffer binding 1) by `gl_InstanceIndex`. Skipping
        // every stage-in buffer left that binding unbound, so each glyph read a
        // zero position/size and collapsed to a degenerate (zero-area) quad —
        // the "blank Safari body text" class. Bind a stage-in buffer as storage
        // too whenever the vertex SPIR-V structurally declares a StorageBuffer at
        // that binding (decoration-driven, never name-keyed).
        let mut storage: Vec<crate::backend::vulkan::engine::StorageBufferResource> = Vec::new();
        for (idx, content) in &vtx_storage {
            if !vertex_buffer_needs_storage_binding(&v_words, *idx, stage_in_bufs.contains(idx)) {
                continue;
            }
            storage.push(crate::backend::vulkan::engine::StorageBufferResource {
                binding: *idx,
                content: content.clone(),
            });
        }
        for (idx, content) in &frag_storage {
            let binding = if buf_collide {
                *idx + FRAG_BUFFER_BINDING_OFFSET
            } else {
                *idx
            };
            storage.push(crate::backend::vulkan::engine::StorageBufferResource {
                binding,
                content: content.clone(),
            });
        }

        // GUARD (always-on fail-visible, drain worker / off-main-core): does the
        // FRAGMENT shader DECLARE a `[[buffer(n)]]`/`[[texture(n)]]`/`[[sampler(n)]]`
        // the draw never bound? Such a resource reads an undefined descriptor and
        // paints garbage with no other fail-log — the fragment-stage analog of the
        // fixed vertex stage-in "blank body text" class (comment above), which the
        // fragment stage otherwise has no cross-check for. This closes that silent
        // mis-execution hole: the Vulkan engine builds its descriptor layout purely
        // from provided resources (`engine/exec.rs`), so a shader referencing an
        // unbound descriptor executes with no error. Fragment-only: a vertex
        // `[[buffer(n)]]` may legitimately be bound as a stage-in attribute (not
        // storage), so a vertex check would false-fire. Standard directly-bound
        // kinds only; ColorInput / ThreadgroupBuffer / StorageImage reach the shader
        // by other paths and carry their own census (`census_reflection_wellformed`).
        // Verified non-flooding: 0 fires across a full x86 boot (desktop convergence
        // + Safari + CSS gradients + a 23-binding compositor shader), so any fire is
        // a genuine bind gap, not expected control flow.
        {
            // Membership predicates over the (tiny) provided-resource slices — the
            // scan allocates nothing on the all-bound hot path (both result Vecs
            // stay empty). The `frag_embedded_*` reason names a DIFFERENT silent
            // hole: a fragment shader declaring an `EmbeddedArgBufferTexture` (m2v
            // flattened out of an `air.indirect_buffer` arg) that only the compute
            // path can source, so the render path leaves it structurally unbound.
            let (unbound, embedded) = frag_unbound_scan(
                &f_shader.reflection.bindings,
                |i| frag_storage.iter().any(|(x, _)| *x == i),
                |i| {
                    req.fragment_textures
                        .iter()
                        .any(|t| t.index == i && t.index < MAX_BIND_SLOTS && t.texture_ref != 0)
                },
                |i| {
                    req.fragment_samplers
                        .iter()
                        .any(|s| s.index == i && s.index < MAX_BIND_SLOTS && s.sampler_ref != 0)
                },
            );
            if !unbound.is_empty() {
                // Cold path only: build the provided-index sets for the log detail.
                let bufs: std::collections::BTreeSet<u32> =
                    frag_storage.iter().map(|(i, _)| *i).collect();
                let texs: std::collections::BTreeSet<u32> = req
                    .fragment_textures
                    .iter()
                    .filter(|t| t.index < MAX_BIND_SLOTS && t.texture_ref != 0)
                    .map(|t| t.index)
                    .collect();
                let smps: std::collections::BTreeSet<u32> = req
                    .fragment_samplers
                    .iter()
                    .filter(|s| s.index < MAX_BIND_SLOTS && s.sampler_ref != 0)
                    .map(|s| s.index)
                    .collect();
                // The declared side and the raw provided pairs, so a fire can be
                // read without a reproduction: which Metal indices the shader
                // wants (with kinds) and exactly what the guest bound where,
                // refs included. A fire on the WindowServer composite showed
                // `unbound=[tex0] provided_tex={3}` and nothing in the line
                // could say whether the shader also declared tex3, or which
                // texture the guest put there — the difference between "decode
                // read the slot wrong" and "the guest binds this shader's
                // second texture only".
                let declared: Vec<String> = f_shader
                    .reflection
                    .bindings
                    .iter()
                    .map(|rb| format!("{:?}[{}]", rb.kind, rb.metal_index))
                    .collect();
                let tex_pairs: Vec<String> = req
                    .fragment_textures
                    .iter()
                    .filter(|t| t.texture_ref != 0)
                    .map(|t| format!("{}:{:#x}", t.index, t.texture_ref))
                    .collect();
                crate::observe::fail(format!(
                    "shader_resource_declared_unbound reason=frag_declared_descriptor_unbound \
                     pipe={} unbound=[{}] provided_buf={bufs:?} provided_tex={texs:?} \
                     provided_smp={smps:?} {}x{} declared=[{}] tex_pairs=[{}]",
                    req.pipeline_ref,
                    unbound.join(","),
                    w,
                    h,
                    declared.join(","),
                    tex_pairs.join(",")
                ));
            }
            if !embedded.is_empty() {
                crate::observe::fail(format!(
                    "shader_resource_declared_unbound reason=frag_embedded_argbuffer_unsupported \
                     pipe={} embedded_tex={embedded:?} {}x{} \
                     (render path cannot source air.indirect_buffer textures)",
                    req.pipeline_ref, w, h
                ));
            }
        }

        // Framebuffer fetch (`air.render_target` INPUT param `dest_N` →
        // reflection `ColorInput` at binding 96+N): the engine supports the
        // attachment-0 fetch as a Vulkan subpass input. `dest_N>0` (fetching a
        // secondary MRT attachment) has no engine path yet — fail visibly, never
        // execute a shader whose destination read would be unbound.
        let frag_color_input = {
            use metal2vulkan::reflect::ResourceKind;
            let mut fetch0 = false;
            for rb in &f_shader.reflection.bindings {
                if rb.kind == ResourceKind::ColorInput {
                    if rb.metal_index == 0 {
                        fetch0 = true;
                    } else {
                        return Err(DrawError::DrawPreparation(
                            DrawPreparationDecline::ColorInputMrtUnsupported {
                                destination_index: rb.metal_index,
                            },
                        ));
                    }
                }
            }
            fetch0
        };
        crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::Sampled);
        // Sampled textures + samplers (metal2vulkan bands: textures 32+N, samplers 64+M).
        // Texture and sampler **indices are independent** (live logo SPIR-V: image
        // binding 35 = texture(3), sampler binding 64 = sampler(0)). Pairing
        // sampler to texture index left sampler 67 empty → black samples.
        // Fragment sampled resources use +FRAG_SAMPLED when either both stages
        // sample or fragment buffers moved into the sampled/static-sampler band.
        let mut images: Vec<crate::backend::vulkan::engine::SampledImageResource> = Vec::new();
        let mut samplers: Vec<crate::backend::vulkan::engine::SamplerResource> = Vec::new();
        let mut sampler_binds: std::collections::BTreeSet<u32> = Default::default();
        {
            let mut push_tex = |index: u32,
                                texture_ref: u32,
                                frag_stage: bool|
             -> Result<(), DrawError> {
                if index >= MAX_BIND_SLOTS || texture_ref == 0 {
                    return Ok(());
                }
                // Measure-only setup_tex sub-split (off-main-core): time the full
                // per-bind resolution (guest object-list + descriptor reads +
                // resolve + surface ensure) vs the post-resolve stats scan, so a
                // boot log names which half of the ~800us/draw to cut.
                let texture_entry =
                    objects::lookup_list_entry(state, host, req.task_id, texture_ref);
                // A type-8 view's channel remap. Resolved here rather than in
                // the loaders because it describes how the bind READS the
                // texture, not what the texture contains: the engine hands it
                // to the image view as a component mapping and the hardware
                // applies it at sample time, so the texels stay untouched and
                // the bind keeps whatever content rail it was already on.
                let view_swizzle = resolve_texture_view(state, host, req.task_id, texture_ref)
                    .and_then(|view| view.swizzle)
                    .filter(|plan| !pixel_format::swizzle_is_identity(plan));
                let attachment_alias = frag_stage
                    .then(|| fragment_attachment_alias_sample(req, index, texture_ref))
                    .flatten();
                let (tw, th, loaded) = if let Some((aw, ah, alias)) = attachment_alias {
                    match alias {
                        AttachmentAliasSample::Clear(clear) => (
                            aw,
                            ah,
                            SampledSourceRequest::Bytes(
                                std::sync::Arc::new(solid_rgba_local(aw, ah, &clear)),
                                None,
                                TexelLayout::Rgba8,
                            ),
                        ),
                        AttachmentAliasSample::Seed(seed) => (
                            aw,
                            ah,
                            SampledSourceRequest::Bytes(
                                std::sync::Arc::new(seed.to_vec()),
                                None,
                                TexelLayout::Rgba8,
                            ),
                        ),
                        AttachmentAliasSample::ResidentChain => {
                            let identity = render_chain_identity(state, req).ok_or({
                                DrawError::DrawPreparation(
                                    DrawPreparationDecline::AttachmentAliasIdentityMissing {
                                        index,
                                        texture_ref,
                                    },
                                )
                            })?;
                            if !crate::backend::vulkan::engine::resident_content_ready(&identity) {
                                return Err(DrawError::DrawPreparation(
                                    DrawPreparationDecline::AttachmentAliasResidentNotReady {
                                        index,
                                        texture_ref,
                                        width: identity.width(),
                                        height: identity.height(),
                                    },
                                ));
                            }
                            (
                                identity.width(),
                                identity.height(),
                                SampledSourceRequest::Target(identity),
                            )
                        }
                    }
                } else {
                    let Some(loaded) = resolve_sampled_source(
                        state,
                        host,
                        req.task_id,
                        texture_ref,
                        texture_entry,
                    ) else {
                        let detail = sample_miss_detail(state, host, req.task_id, texture_ref);
                        return Err(DrawError::DrawPreparation(
                            DrawPreparationDecline::TextureResolveMissing {
                                stage: if frag_stage { "fragment" } else { "vertex" },
                                index,
                                texture_ref,
                                detail,
                            },
                        ));
                    };
                    let (rw, rh, _mid, src) = loaded;
                    (rw, rh, src)
                };
                let mut bytes_identity = None;
                // Byte layout of a CPU-origin bind. Default RGBA8; a source that
                // already holds its bytes in an uploadable order keeps them —
                // BGRA8 from the type-4 scanout cache, a native single/dual-channel
                // video plane — and the host spelling is applied once, where the
                // engine resource is built (`vk_texel_layout` below).
                let mut sampled_format = TexelLayout::Rgba8;
                let source = match loaded {
                    SampledSourceRequest::Bytes(rgba, identity, byte_format) => {
                        bytes_identity = identity;
                        sampled_format = byte_format;
                        crate::backend::vulkan::engine::SampledSource::Bytes(rgba)
                    }
                    SampledSourceRequest::Target(identity) => {
                        // A resident bound directly reuses the registry's own
                        // image view, which the engine creates once per target
                        // and cannot re-decorate per bind. Refuse rather than
                        // bind it unswizzled: reading the wrong channels is a
                        // rendering bug that looks like content, whereas a
                        // named decline is one grep away.
                        if view_swizzle.is_some() {
                            crate::runtime::census::view_swizzle_census::note_declined(
                                crate::runtime::census::view_swizzle_census::SwizzleDecline::ResidentDirectBind,
                                texture_ref,
                            );
                            return Ok(());
                        }
                        crate::backend::vulkan::engine::SampledSource::Target(identity)
                    }
                    SampledSourceRequest::GuestRuns(src, native, identity) => {
                        sampled_format = native;
                        bytes_identity = identity;
                        crate::backend::vulkan::engine::SampledSource::GuestRuns(src)
                    }
                };
                let base_off = if frag_stage && separate_sampled {
                    FRAG_SAMPLED_RESOURCE_BINDING_OFFSET
                } else {
                    0
                };
                let img_bind = TEXTURE_BINDING_BASE + index + base_off;
                // Texture dimensionality comes solely from the translator's reflection,
                // keyed on the UN-relocated descriptor binding. The always-on
                // `census_reflection_wellformed` guard (m2v_cache) proves the reflection
                // is internally consistent per translate. `Absent` is an unused/unbound
                // sampler slot (Metal permits it) — default 2D silently (expected control
                // flow). `Unsupported` is a texture shape reflection carries but the
                // sampled path can't express — log fail-visibly, then keep the 2D default
                // so the draw still paints rather than dropping content.
                use crate::runtime::spirv_bind::{ReflectedSampledKind, SampledImageKind};
                let reflection = if frag_stage {
                    &f_shader.reflection
                } else {
                    &v_shader.reflection
                };
                let image_kind = match crate::runtime::spirv_bind::reflected_sampled_kind(
                    reflection,
                    TEXTURE_BINDING_BASE + index,
                ) {
                    ReflectedSampledKind::Kind(k) => k,
                    ReflectedSampledKind::Absent => SampledImageKind::D2,
                    ReflectedSampledKind::Unsupported => {
                        crate::observe::fail(format!(
                            "reflection_sampled_shape_unsupported stage={} idx={index} ref={texture_ref} binding={img_bind}",
                            if frag_stage { "frag" } else { "vert" }
                        ));
                        SampledImageKind::D2
                    }
                };
                let Some(shape) = sampled_image_shape(image_kind) else {
                    return Err(DrawError::DrawPreparation(
                        DrawPreparationDecline::TextureDimensionUnsupported {
                            stage: if frag_stage { "fragment" } else { "vertex" },
                            index,
                            texture_ref,
                            binding: img_bind,
                            kind: format!("{image_kind:?}"),
                        },
                    ));
                };
                let SampledImageShape {
                    arrayed,
                    volume,
                    cube,
                    one_dim,
                    layers,
                } = shape;
                // A Vulkan 1D image is defined to have height 1; the descriptor
                // may report the LUT's texel count in either axis, so collapse
                // to a single row and fold the other axis into the width the
                // sampled bytes are validated against.
                let (tw, th) = if one_dim {
                    (tw.saturating_mul(th).max(1), 1)
                } else {
                    (tw, th)
                };
                // What a draw samples decides what it can draw. For a small
                // float target — an icon canvas — an empty source and an empty
                // result are the same picture, and only this separates them.
                if crate::observe::dump_flush_surfaces() && w <= 160 && h <= 160 {
                    let census = match &source {
                        crate::backend::vulkan::engine::SampledSource::Bytes(b) => {
                            format!("bytes={} nonzero={}", b.len(), b.iter().filter(|x| **x != 0).count())
                        }
                        crate::backend::vulkan::engine::SampledSource::Target(id) => {
                            // Bound, ready and geometry-matched is not the same
                            // as having content. Read it back so an empty mask
                            // is distinguishable from a material that computes
                            // nothing from a good one.
                            match crate::backend::vulkan::engine::read_target(id) {
                                Ok(rb) => {
                                    let px = rb.into_bgra8();
                                    format!(
                                        "target_bytes={} target_nonzero={}",
                                        px.len(),
                                        px.iter().filter(|x| **x != 0).count()
                                    )
                                }
                                Err(e) => format!("target_read_failed={e}"),
                            }
                        }
                        other => format!("source={other:?}"),
                    };
                    crate::observe::fail(format!(
                        "draw_sampled_census pipe={} target={}x{} bind={img_bind} ref={texture_ref} {tw}x{th} {census}",
                        req.pipeline_ref, w, h
                    ));
                }
                images.push(crate::backend::vulkan::engine::SampledImageResource {
                    binding: img_bind,
                    width: tw,
                    height: th,
                    layers,
                    arrayed,
                    volume,
                    cube,
                    one_dim,
                    source,
                    format: translate::pixel::vk_texel_layout(sampled_format),
                    identity: bytes_identity.map(|i| {
                        crate::backend::vulkan::engine::SampledContentIdentity {
                            key: i.key,
                            generation: i.generation,
                        }
                    }),
                    swizzle: view_swizzle.unwrap_or_default(),
                });
                Ok(())
            };
            for t in &req.vertex_textures {
                push_tex(t.index, t.texture_ref, false)?;
            }
            for t in &req.fragment_textures {
                push_tex(t.index, t.texture_ref, true)?;
            }
            // Metal's unbound-texture rule, made explicit for Vulkan.
            //
            // A Metal fragment shader may declare a `[[texture(n)]]` the draw
            // never binds; sampling it is defined to return zero. Vulkan has no
            // equivalent: the engine derives its descriptor layout from provided
            // resources alone, so a declared-but-unbound slot is both missing
            // from the pipeline layout and undefined to read. A validation layer
            // reports the pair on the same binding —
            // `VUID-VkGraphicsPipelineCreateInfo-layout-07988` and
            // `VUID-vkCmdDraw-None-08114` at `[Set 0, Binding 160]`, which is
            // `TEXTURE_BINDING_BASE + 0`.
            //
            // Undefined is also what it looked like: the descriptor addressed
            // whatever memory was there, usually the previous frame, so every
            // window dragged a trail behind it.
            //
            // A one-texel zero image restores Metal's rule exactly. It costs
            // four bytes per slot, needs no extension (so it holds on all four
            // support-matrix cells rather than only where `nullDescriptor`
            // exists), and a shader that samples it reads the zero Metal
            // promised. The sampler side of this pair has always defaulted the
            // same way (`SamplerResource::normalized_default` for `ref == 0`);
            // only the texture side was missing.
            for index in declared_fragment_texture_indices(&f_shader.reflection.bindings) {
                if index >= MAX_BIND_SLOTS {
                    continue;
                }
                if req
                    .fragment_textures
                    .iter()
                    .any(|t| t.index == index && t.texture_ref != 0)
                {
                    continue;
                }
                let base_off = if separate_sampled {
                    FRAG_SAMPLED_RESOURCE_BINDING_OFFSET
                } else {
                    0
                };
                let img_bind = TEXTURE_BINDING_BASE + index + base_off;
                if images.iter().any(|i| i.binding == img_bind) {
                    continue;
                }
                // The declared shape decides the placeholder's shape: a shader
                // declaring a cube or an array must not be handed a plain 2D
                // image, or the descriptor is the wrong type and the layout
                // mismatch simply moves. An unsupported shape is left alone —
                // the bound path already declines those by name.
                use crate::runtime::spirv_bind::ReflectedSampledKind;
                let kind = match crate::runtime::spirv_bind::reflected_sampled_kind(
                    &f_shader.reflection,
                    TEXTURE_BINDING_BASE + index,
                ) {
                    ReflectedSampledKind::Kind(k) => k,
                    _ => continue,
                };
                let Some(shape) = sampled_image_shape(kind) else {
                    continue;
                };
                images.push(crate::backend::vulkan::engine::SampledImageResource {
                    binding: img_bind,
                    width: 1,
                    height: 1,
                    layers: shape.layers,
                    arrayed: shape.arrayed,
                    volume: shape.volume,
                    cube: shape.cube,
                    one_dim: shape.one_dim,
                    source: crate::backend::vulkan::engine::SampledSource::Bytes(
                        std::sync::Arc::new(vec![0u8; 4 * shape.layers.max(1) as usize]),
                    ),
                    format: translate::pixel::vk_texel_layout(TexelLayout::Rgba8),
                    identity: None,
                    swizzle: Default::default(),
                });
            }
        }
        {
            let mut push_smp =
                |index: u32, sampler_ref: u32, frag_stage: bool| -> Result<(), DrawError> {
                    if index >= MAX_BIND_SLOTS {
                        return Ok(());
                    }
                    let base_off = if frag_stage && separate_sampled {
                        FRAG_SAMPLED_RESOURCE_BINDING_OFFSET
                    } else {
                        0
                    };
                    let smp_bind = SAMPLER_BINDING_BASE + index + base_off;
                    if sampler_binds.insert(smp_bind) {
                        let sampler = if sampler_ref != 0 {
                            load_vulkan_sampler(state, host, req.task_id, sampler_ref, smp_bind)
                                .map_err(DrawError::DrawPreparation)?
                        } else {
                            crate::backend::vulkan::engine::SamplerResource::normalized_default(
                                smp_bind,
                            )
                        };
                        samplers.push(sampler);
                    }
                    Ok(())
                };
            // Stream sampler slots (often index 0 while texture is 3 for logo).
            for s in &req.vertex_samplers {
                if s.sampler_ref != 0 {
                    push_smp(s.index, s.sampler_ref, false)?;
                }
            }
            for s in &req.fragment_samplers {
                if s.sampler_ref != 0 {
                    push_smp(s.index, s.sampler_ref, true)?;
                }
            }
        }
        // AIR constexpr samplers carry their immutable state in reflection. Bind
        // those exact values before the residual SPIR-V scan provisions defaults
        // for translator-generated sampler-less read helpers.
        for (reflection, frag_stage) in
            [(&v_shader.reflection, false), (&f_shader.reflection, true)]
        {
            for reflected in &reflection.bindings {
                if reflected.kind != metal2vulkan::reflect::ResourceKind::StaticSampler {
                    continue;
                }
                let Some(descriptor) = reflected.descriptor else {
                    return Err(DrawError::DrawPreparation(
                        DrawPreparationDecline::StaticSamplerReflectionDescriptorMissing {
                            stage: if frag_stage { "fragment" } else { "vertex" },
                        },
                    ));
                };
                let Some(state) = reflected.static_sampler else {
                    return Err(DrawError::DrawPreparation(
                        DrawPreparationDecline::StaticSamplerReflectionStateMissing {
                            stage: if frag_stage { "fragment" } else { "vertex" },
                            binding: descriptor.binding,
                        },
                    ));
                };
                let binding = descriptor.binding
                    + if frag_stage && separate_sampled {
                        FRAG_SAMPLED_RESOURCE_BINDING_OFFSET
                    } else {
                        0
                    };
                if sampler_binds.insert(binding) {
                    let sampler = reflected_static_sampler_resource(
                        if frag_stage { "fragment" } else { "vertex" },
                        binding,
                        state,
                    )
                    .map_err(DrawError::DrawPreparation)?;
                    samplers.push(sampler);
                }
            }
        }
        // Reflect the residual shader interface and provision defaults only
        // where explicit guest or constexpr state did not already win.
        for binding in crate::runtime::spirv_bind::sampler_bindings(&v_words)
            .into_iter()
            .chain(crate::runtime::spirv_bind::sampler_bindings(&f_words))
        {
            if sampler_binds.insert(binding) {
                samplers.push(
                    crate::backend::vulkan::engine::SamplerResource::normalized_default(binding),
                );
            }
        }
        crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::Seed);
        // Color load seed: CLEAR → solid; LOAD → guest/host seed when present.
        // `seed_order` names what is in those bytes; the engine folds any needed
        // R/B exchange into its copy into the mapped staging span rather than
        // making this side materialize a converted frame.
        let mut target_rgba8: Option<std::sync::Arc<Vec<u8>>> = None;
        let mut seed_order = crate::backend::vulkan::engine::SeedOrder::Rgba8;
        let gpu_only_content_allowed =
            crate::backend::vulkan::engine::deferred_gpu_only_content_allowed();
        // Records 2+ of a resident render-pass chain load the prior record's
        // content directly from the engine target (no CPU seed, no re-upload).
        let mut chain_load_from_target = false;
        // Resolved once and read by both the Load gate below and the
        // `target_identity` assignment further down, so the record that loads
        // from a resident is by construction the record that renders into it.
        let type11_resident_target = type11_store_identity(state, req, writeback_guest);
        if req.chain_from_resident {
            if let Some(identity) = render_chain_identity(state, req) {
                if crate::backend::vulkan::engine::resident_content_ready(&identity) {
                    chain_load_from_target = true;
                } else {
                    // The armed chain lost its resident (engine reset /
                    // registry eviction). Seeding from stale guest/cache
                    // bytes here would silently wipe the chained records —
                    // fail visibly and let the exec loop abandon the chain.
                    return Err(DrawError::DrawPreparation(
                        DrawPreparationDecline::ChainResidentNotReady {
                            target_gva: req.colors.first().map(|c| c.target_gva).unwrap_or(0),
                            width: w,
                            height: h,
                        },
                    ));
                }
            }
        }
        // A cross-pass GVA resident-Load rung used to sit here, taking colour0
        // from an open deferred GVA Store window instead of a seed. Its
        // denominator read `xpass_c0_gva_load_no_window` 0 and
        // `xpass_c0_gva_load_window_open` 0 against `xpass_c0_not_gva_load`
        // 4859/5616 over two driven boots, with the window rail itself busy
        // (`gva_deferred` 2508/2605): no draw reaching here has a colour0 that
        // is a seedless LOAD'd GVA target, because `req.chain_from_resident`
        // sets `chain_load_from_target` above and skips it. The resident-chain
        // rail carries the case in full, and a LOAD arriving with no seed
        // still says so through `load_seed_lost_other`.
        // Type-11 composite Load: when the resident this record is about to
        // render into was stamped with the mapping's current
        // `surface_content_epoch`, its image already holds exactly the bytes
        // `resolve_type11_load_seed` would upload. Load from it and skip the
        // upload.
        //
        // The epoch is the witness. `mark_mapping_written` advances it and every
        // guest-page writer *in this crate* calls it, so a blit or a compute
        // writeback invalidates the stamp without knowing this rail exists. The
        // deferred type-11 publish — the one writer that changes the pixels
        // without touching guest pages — advances it explicitly. Anything that
        // leaves the answer unknown (no slot, an evicted or recycled image, a
        // draw since the stamp, a `map_generation` rewire that renames the
        // identity) reads back `None` and takes the seed.
        //
        // # The epoch cannot see the guest, so it is only half the test
        //
        // The epoch's closure does *not* include a guest CPU write, and cannot.
        // A type-11 surface's pages are plain guest RAM; the guest writes them
        // with its own CPU and no device operation is involved, so nothing calls
        // `mark_mapping_written` and the epoch does not move. Every caller of it
        // is a device-side writer — `compute_exec`, `mapping_write`, `exec`,
        // `storage_flush` — and there is no entry for the owner of the pages.
        //
        // On the epoch alone the elision answers "current" for a resident that
        // is stale, the pass loads from it, the matching Store publishes it back
        // over the guest's own bytes, and the epoch still has not moved — which
        // arms the same wrong answer for the next frame. That fixpoint is
        // exactly the "renders correctly for a few frames then stays corrupted"
        // report.
        //
        // Measured, three 14-round Finder recomposite boots on the epoch-only
        // rail:
        //
        //   elision on              rounds 3,4,5,6 corrupt — held, none recovered
        //   all reuse rails off     round 4 corrupt, rounds 5-14 clean
        //   this rail off alone     round 1 corrupt, round 2 clean (recovered)
        //
        // Turning the rail off restored recovery and corruption still occurred
        // without it, so the elision was never the cause of a bad frame — it was
        // what made a bad frame permanent. Recovery requires a source of good
        // pixels that is not the resident, and the seed's only other source is
        // the surface's own guest pages, so the guest must be writing them.
        //
        // `type11_guest_wrote_since_store` is the other half: the hypervisor's
        // dirty bitmap, the one witness for a write this device did not make.
        // Both halves must agree, and every unknown on either side reads as
        // "not current".
        //
        // Measured on the same rig with both halves, two 14-round boots:
        //
        //   boot A   14/14 clean          t11_gw_ref_moved 2 973
        //   boot B   1,2 corrupt 3-5 clean; 6 corrupt 7 clean;
        //            8 corrupt 9-14 clean      t11_gw_ref_moved 8 887
        //
        // Boot B is the informative one, and it says exactly what this rail
        // does and does not fix. Corruption still happens — something writes a
        // bad composite and that is still unidentified — but **every corrupt
        // round is followed by a clean one**. On the epoch alone, the same
        // script at the same HEAD gave rounds 3, 4, 5 and 6 corrupt and *held*,
        // with none recovering. The latch is gone; its cause is not.
        //
        // The 8 887 is the mechanism, not the round count. The guest CPU really
        // does write these composites — thousands of times a session — and on
        // the epoch alone every one of those writes was invisible.
        //
        // What boot B's per-round counters rule out, for whoever takes the
        // remaining half: nothing in `store_routes` separates a corrupt round
        // from a clean one. Rounds 6 and 8 (corrupt) against 7 and 9 (clean)
        // ran at 931/854 against 921/839 guest-write refusals, 1338/1305
        // against 1379/1236 elisions, and matching `draw_partial_clear`,
        // `draw_partial_load_seeded` and `load_seed_ok` — the counters agree to
        // within the round-to-round drift. The victim is one whole absent icon
        // (`blobs=6 intact=6`, never shrunk), so whatever produces it is not
        // visible in the seed, reuse, or partial-draw populations this census
        // covers.
        //
        // Why not the sibling rail's shape — `load_linear_guest_memoized`
        // re-reads the guest's native rows on every call and byte-compares
        // before reusing its `Arc`. Priced on one boot: `type11_seed_elided`
        // 41 389 against `type11_seed_uploaded` 242, at a mean 1.43 M texels per
        // elision, so revalidating by re-reading would move ~237 GB of guest
        // memory a session. The bitmap answers the same question in a word.
        if !chain_load_from_target {
            if let Some((identity, mapping_epoch)) = type11_load_currency_query(state, req) {
                // Both arms counted, into the same one-second window as
                // `drain_duty`. An elision count alone cannot tell "the seed was
                // skipped" from "this record was never a candidate", and the
                // ratio of the two is a within-boot number — the only kind that
                // survives the 1.8x `us_per_draw` drift between boots on this rig.
                let resident_epoch =
                    crate::backend::vulkan::engine::resident_content_epoch(&identity);
                let mapping_id = req.colors.first().map(|c| c.mapping_id).unwrap_or(0);
                let guest_wrote = type11_guest_wrote_since_store(state, host, mapping_id);
                if type11_resident_is_current(mapping_epoch, resident_epoch) && !guest_wrote {
                    chain_load_from_target = true;
                    crate::runtime::drain::note_store_route("type11_seed_elided");
                    note_type11_elision_extent(w, h);
                } else {
                    crate::runtime::drain::note_store_route("type11_seed_uploaded");
                    // Separated from the epoch's refusal so a boot can say which
                    // half refused. The two answer different questions and a
                    // single counter would hide a rail that never fires.
                    if guest_wrote {
                        crate::runtime::drain::note_store_route("type11_seed_guest_wrote");
                    }
                }
            }
        }
        if let Some(c0) = req.colors.first() {
            match c0.load_action {
                x if x == PASS_LOAD_ACTION_LOAD && chain_load_from_target => {
                    // Resident target carries the chain; no CPU seed bytes.
                }
                x if x == PASS_LOAD_ACTION_CLEAR => {
                    target_rgba8 =
                        Some(std::sync::Arc::new(solid_rgba_local(w, h, &c0.clear_color)));
                }
                x if x == PASS_LOAD_ACTION_LOAD => {
                    // Which door this pass took, so a pass that ends with no
                    // seed says which source was supposed to have one. A LOAD
                    // means the guest is compositing *onto what is already
                    // there*; arriving with nothing is the previous frame of
                    // that layer being dropped, and everything outside the
                    // geometry this pass draws goes blank.
                    let mut seed_door = "none";
                    if let Some(seed) = c0.target_seed_rgba.as_ref() {
                        seed_door = "color_seed";
                        if seed.len() == (w as usize) * (h as usize) * 4 {
                            // seed_color_load selected this by RT provenance.
                            // Black/transparent bytes are valid attachment data.
                            target_rgba8 = Some(std::sync::Arc::new(seed.clone()));
                        }
                    } else if c0.mapping_id != 0 {
                        seed_door = "mapping";
                        if let Some((bytes, order)) =
                            resolve_type11_load_seed(state, host, c0.mapping_id, w, h)
                        {
                            target_rgba8 = Some(bytes);
                            seed_order = order;
                        }
                    }
                    // Two doors, and there is no third. There is no separate
                    // lookup of the texture_ref encode cache:
                    // `color_target_request` calls `seed_color_load` while
                    // building the request, and that is where the encode cache
                    // is read — it is the `color_seed` door above. A second
                    // lookup of the same map behind a stricter gate can only be
                    // reached when the first has already declined, which for a
                    // cached texture it cannot. Measured across three
                    // independently driven x86/Vulkan boots: 1 558, 395 and 295
                    // colour LOAD seed resolutions, and 0 serves plus 0 misses
                    // at that door in every one.
                    note_load_seed_outcome(seed_door, target_rgba8.is_some(), c0, w, h);
                }
                _ => {}
            }
        }
        crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::Assemble);
        let mut resources = crate::backend::vulkan::engine::DrawRequest {
            // Honor the guest's face-culling state, its winding, and its
            // primitive type. All three come from `translate::raster`, and all
            // three fall back to a Metal default when the guest bound nothing —
            // but an out-of-contract *value* is a different thing from an unbound
            // one, and it says its own name before falling back. Silently
            // coercing here is how a guest that asked for lines got triangles
            // with nothing in the log to say so.
            cull_mode: raster_or_default(
                req.cull_mode,
                translate::raster::cull_mode,
                crate::backend::vulkan::engine::CullMode::None,
                req.pipeline_ref,
                "cull_mode_unmapped",
            ),
            // MTLWinding: CounterClockwise == 1; Metal defaults to Clockwise.
            front_face_ccw: raster_or_default(
                req.front_facing,
                translate::raster::front_face_ccw,
                false,
                req.pipeline_ref,
                "winding_unmapped",
            ),
            first_vertex: req.first_vertex,
            instance_count: Some(req.instance_count.max(1)),
            primitive_topology: raster_or_default(
                Some(req.primitive_type),
                translate::raster::primitive_topology,
                crate::backend::vulkan::engine::PrimitiveTopology::Triangle,
                req.pipeline_ref,
                "primitive_type_unmapped",
            ),
            ..crate::backend::vulkan::engine::DrawRequest::default()
        };
        // Geometry landing outside a small target is invisible in the sink: the
        // draw reports success and the surface comes out empty but for whatever
        // sliver fell inside. Record the two things that decide where it lands
        // against the size of what it lands on.
        if crate::observe::dump_flush_surfaces() && w <= 256 && h <= 256 {
            crate::observe::fail(format!(
                "draw_placement pipe={} {}x{} viewport={:?} scissor={:?} vtx={} inst={}",
                req.pipeline_ref,
                w,
                h,
                req.viewport,
                req.scissor,
                req.vertex_count,
                req.instance_count
            ));
        }
        resources.viewport =
            req.viewport
                .map(|vp| crate::backend::vulkan::engine::ViewportResource {
                    x: vp[0] as f32,
                    y: vp[1] as f32,
                    width: vp[2] as f32,
                    height: vp[3] as f32,
                    min_depth: vp[4] as f32,
                    max_depth: vp[5] as f32,
                });
        if let Some((x, y, sw, sh)) = req.scissor {
            note_draw_coverage(
                x,
                y,
                sw,
                sh,
                w,
                h,
                req.colors.first().map(|c| c.load_action),
                target_rgba8.is_some(),
                chain_load_from_target,
            );
            resources.scissor = Some(crate::backend::vulkan::engine::ScissorResource {
                x,
                y,
                width: sw,
                height: sh,
            });
        }
        if let Some(idx) = req.indexed.as_ref() {
            let index_type = translate::raster::index_type(idx.index_type).ok_or({
                DrawError::DrawPreparation(DrawPreparationDecline::IndexLoad {
                    reason: IndexLoadReason::TypeUnsupported,
                })
            })?;
            let indices =
                load_index_bytes_reason(state, host, req.task_id, idx).map_err(|reason| {
                    DrawError::DrawPreparation(DrawPreparationDecline::IndexLoad { reason })
                })?;
            resources.indexed = Some(crate::backend::vulkan::engine::IndexedDrawResource {
                index_type,
                index_count: idx.index_count,
                // Vulkan's vertexOffset is a signed 32-bit field where Metal's
                // baseVertex is 64-bit, so a value that cannot fit is declined
                // rather than wrapped into an index somewhere else in the
                // buffer. The guest cannot express one: Apple's serializer
                // truncates baseVertex to 16 bits in the compact records and
                // this device's own decode is the only other source.
                vertex_offset: i32::try_from(idx.base_vertex).map_err(|_| {
                    DrawError::DrawPreparation(DrawPreparationDecline::IndexLoad {
                        reason: crate::runtime::metal_draw::IndexLoadReason::BaseVertexOutOfRange,
                    })
                })?,
                indices,
            });
        }
        // Vulkan's `firstInstance` is Metal's `baseInstance`. The field has
        // always been here and always read 0, because nothing upstream decoded
        // the draw forms that carry one; the engine's Constant-step-rate vertex
        // prefix rebuild already reads it.
        resources.base_instance = req.base_instance;
        resources.vertex_attributes = attrs;
        resources.storage_buffers = storage;
        resources.sampled_images = images;
        resources.color_input = frag_color_input;
        resources.samplers = samplers;
        // Load seed always goes to the GPU (workstream D3). Premult One/OMSA is
        // hardware blend over the Load-seeded target — identical math to the
        // retired software `src + seed*(1-src.a)` path. Sampled alpha is
        // protocol data and must not be rewritten from an RGB content census;
        // content-gated keep-seed / alpha0-holes composites are retired.
        let store_is_store = req
            .colors
            .first()
            .map(|c| c.store_action == PASS_STORE_ACTION_STORE)
            .unwrap_or(true);
        resources.target_rgba8 = target_rgba8;
        resources.target_seed_order = seed_order;
        // A Store reads back; anything else skips it.
        //
        // A Store used to have a second option: when the host's page aliases
        // were stable *and* the device could import a host pointer over them,
        // it rendered into a BGRA resident with `skip_readback` and the
        // import-present rail DMA'd that resident into the guest's pages. The
        // import is gone, so the only way a Store's pixels reach the guest is
        // the CPU writeback, and that needs them read back.
        resources.skip_readback = !store_is_store;
        // Ephemeral resident render-pass rail: intermediate Store records render
        // into a protocol-keyed RGBA target on every Vulkan backend. This does
        // not leave guest-visible content GPU-only: portability devices read the
        // final record back and perform the normal synchronous guest Store.
        // Cross-pass deferred ownership remains gated below.
        let mut resident_render_chain = false;
        // Deferred GVA Store rail: the final/single record also stays on the
        // registry resident (skip_readback) — the caller arms a flush-on-
        // access window instead of the sync readback + guest write on the
        // stamp path (`arm_gva_deferred_store`).
        let mut gva_resident_store = false;
        if req.chain_from_resident || (store_is_store && !writeback_guest) {
            if let Some(identity) = render_chain_identity(state, req) {
                resources.target_identity = Some(identity);
                if store_is_store && !writeback_guest {
                    resources.skip_readback = true;
                    resident_render_chain = true;
                }
            }
        }
        if gpu_only_content_allowed && store_is_store && writeback_guest {
            if let Some(identity) = gva_chain_identity(req) {
                // Only the eligibility call can still vary here: the enclosing
                // `&&` already established `store_is_store && writeback_guest`.
                // (The sibling rail above re-tests its pair for real, because
                // its outer condition is an `||`.)
                if gva_store_defer_eligible(req) {
                    resources.target_identity = Some(identity);
                    resources.skip_readback = true;
                    gva_resident_store = true;
                }
            }
        }
        // A type-11 composite Store renders into its registry resident, and skips
        // its readback when the deferred rail can name that resident as the
        // window's frame instead of owning a CPU copy of it.
        //
        // `surface_deferred / readbacks` measured 1.02 and `surface_flush /
        // surface_deferred` measured 0.138: every composite Store read a whole
        // framebuffer back and ~86 % of those frames were never asked for. The
        // gate is the *arm* gate, asked here rather than after the draw, because
        // `skip_readback` has to be decided before submit — so a Store that will
        // not be able to defer keeps its readback and lands synchronously exactly
        // as it always has.
        //
        // The eligibility answer is not carried forward. `arm_surface_resident_store`
        // asks again after the draw and falls back to a materializing read on any
        // refusal, which is what makes a stale answer here cost a readback rather
        // than a lost frame.
        let mut surface_resident_store = false;
        if resources.target_identity.is_none() {
            resources.target_identity = type11_resident_target.clone();
        }
        // Whether the slot this record renders into is the one this rail would
        // pin, asked by comparing the two identities rather than by testing that
        // no other rail set one.
        //
        // The difference is the whole change. A composite Store arrives here with
        // `target_identity` already set when it is the last record of a resident
        // render-pass chain: the block above resolved `render_chain_identity` so
        // the record could take `LoadOp::LoadFromTarget`, and for `mapping_id != 0`
        // that *is* the `surface_identity` `type11_store_identity` returns. An
        // `is_none()` test read that agreement as a conflict and kept the readback
        // for 100 % of the population. The GVA rail cannot collide — its identity
        // requires `mapping_id == 0` and this one requires `mapping_id != 0` — so a
        // genuine mismatch means another namespace owns the attachment and the
        // frame this rail would vouch for is not in the slot it would pin.
        let renders_into_surface_identity =
            type11_resident_target.is_some() && resources.target_identity == type11_resident_target;
        // `!skip_readback` is implied — a set flag means one of the rails above
        // claimed this record, and each returns its own span before
        // `ResidentSurfaceStore` is reached, so a record that armed here as well
        // would skip its readback and never arm anything. Stated rather than
        // derived, because the derivation is a property of two other blocks.
        // `surface_store_defer_eligible` asks the capability gate itself, so it
        // is not repeated here.
        if renders_into_surface_identity
            && !resources.skip_readback
            && surface_store_defer_eligible(state, req).is_some()
        {
            resources.skip_readback = true;
            surface_resident_store = true;
        }
        // A first-failure classifier for a composite Store that still reads
        // back used to sit here. Its outer gate was never once true — none of
        // its four counters, nor its `else` arm, appears anywhere in the
        // always-on log across every boot it holds. Every type-11 Store either
        // skips its readback or is not a writeback Store.
        if chain_load_from_target {
            if resources.target_identity.is_none() {
                // chain_from_resident implies a protocol target identity; a
                // miss here is a rail wiring bug, not a content condition.
                return Err(DrawError::DrawPreparation(
                    DrawPreparationDecline::ChainResidentIdentityMissing {
                        target_gva: req.colors.first().map(|c| c.target_gva).unwrap_or(0),
                        width: w,
                        height: h,
                    },
                ));
            }
            resources.load_from_target = true;
            resources.target_rgba8 = None;
        }
        // Type-11 Load used to have a GPU rail here — ~170 lines of front-frame
        // retention policy resolving which resident image held the frame the
        // guest computes its damage against. It was reachable only under
        // `try_import`. A Store now always reads back and always seeds from
        // guest pages, so there is no resident-only attachment to reseed.
        // Metal path always passes color0 blend into the encoder. Linux/engine
        // previously left `resources.blend = None` → opaque replace for every
        // draw, so Load seeds (gray/wallpaper/logo bases) were wiped by sparse
        // dock/chrome layers that Metal would alpha-blend over the attachment.
        // Contract: type-7 color attachment blend tags (decode/resource.rs).
        // Outside the `blending_enabled` guard below, and deliberately: an
        // unblended attachment with a mask still leaves its unwritten channels
        // alone, so gating the mask on blending would drop it exactly where the
        // guest is replacing rather than compositing.
        resources.color_write_mask = pd.color0.write_mask;
        if pd.color0.blending_enabled {
            let constants = req.blend_color.unwrap_or([0.0; 4]);
            match translate::blend::state(
                pd.color0.src_rgb,
                pd.color0.dst_rgb,
                pd.color0.op_rgb,
                pd.color0.src_alpha,
                pd.color0.dst_alpha,
                pd.color0.op_alpha,
                constants,
            ) {
                Ok(b) => {
                    resources.blend = Some(b);
                }
                Err(e) => {
                    crate::observe::fail(format!(
                        "m2v_blend_map_fail pipe={} {e}",
                        req.pipeline_ref
                    ));
                }
            }
        }

        // The engine ignores this when the draw is indexed (the index count
        // governs), but it still validates it, so it is passed either way.
        let vertex_count = req.vertex_count.max(1);

        // Decide FIRST whether a census line will be emitted at all; the
        // resource metas below (per-attr/ssbo format!, hex prefixes, 16-float
        // matrix dump) cost real per-draw CPU and were previously computed
        // unconditionally on every draw only to be dropped.
        let census_verbose = crate::observe::draw_log_enabled();
        let fixed_state_gap = vulkan_fixed_state_gap(req);
        let fixed_gap_first = !fixed_state_gap.is_empty() && {
            use std::collections::HashSet;
            use std::sync::Mutex;
            type FixedStateGapKey = (u32, u32, u32, String);
            static SEEN: Mutex<Option<HashSet<FixedStateGapKey>>> = Mutex::new(None);
            let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
            seen.get_or_insert_with(HashSet::new).insert((
                req.pipeline_ref,
                w,
                h,
                fixed_state_gap.clone(),
            ))
        };
        // Honor a bound NON-TRIVIAL depth-stencil state: attach a transient depth
        // buffer + enable the depth test. Decoded once per depth draw; the whole
        // 2D UI binds no depth-stencil (`depth_stencil_ref == 0`, 0 decodes), so
        // this is inert there. A trivial state (compare Always, no write, no
        // stencil) stays `None` — no depth attachment, byte-identical 2D path.
        // Still-unrepresented sub-cases (guest depth LOAD, stencil test,
        // out-of-contract compare) are dropped fail-visibly, deduped per
        // (pipe,slug) so 3D content cannot flood the log.
        if req.depth_stencil_ref != 0 {
            let ds = match load_depth_stencil_descriptor(
                state,
                host,
                req.task_id,
                req.depth_stencil_ref,
            ) {
                Ok(ds) => Some(ds),
                Err(reason) => {
                    // The guest bound a depth-stencil state (`ds_ref != 0`) that we
                    // could not resolve/decode: the draw silently renders with the
                    // depth test DISABLED (wrong occlusion for 3D content). Every
                    // other sub-case below is fail-visible, so name this one too —
                    // deduped per (pipe,reason) so 3D content cannot flood, and inert
                    // on the 2D UI path (which binds no depth-stencil).
                    if degrade_log_first(req.pipeline_ref, reason) {
                        crate::observe::fail(format!(
                            "shader_state_degraded reason={reason} \
                             pipe={} ds_ref={} {}x{} \
                             (bound depth-stencil unresolved; depth test disabled)",
                            req.pipeline_ref, req.depth_stencil_ref, w, h
                        ));
                    }
                    None
                }
            };
            if let Some(ds) = ds {
                if !depth_stencil_descriptor_is_trivial(&ds) {
                    match translate::raster::compare_function(ds.depth_compare_function).ok() {
                        Some(compare) => {
                            let (clear_value, load_action) = req
                                .depth_attach
                                .as_ref()
                                .map(|d| (d.clear_depth as f32, d.load_action))
                                .unwrap_or((1.0, PASS_LOAD_ACTION_CLEAR));
                            // The transient depth buffer supports CLEAR only; a
                            // guest depth LOAD needs a persistent depth resident
                            // (deferred). Degrade to CLEAR, fail-visible.
                            if load_action == PASS_LOAD_ACTION_LOAD
                                && degrade_log_first(
                                    req.pipeline_ref,
                                    "depth_load_unsupported_transient",
                                )
                            {
                                crate::observe::fail(format!(
                                    "shader_state_degraded reason=depth_load_unsupported_transient \
                                     pipe={} ds_ref={} {}x{} \
                                     (transient depth clears; multi-pass depth LOAD not yet resident)",
                                    req.pipeline_ref, req.depth_stencil_ref, w, h
                                ));
                            }
                            // Stencil test: engaged when either face is enabled.
                            // A face that is *not* enabled maps to Metal's
                            // documented `MTLStencilDescriptor` default (compare
                            // Always, all ops Keep, full masks) — a no-op face —
                            // NOT its raw decoded bytes, which for a disabled
                            // face need not be initialized. An out-of-contract
                            // compare/op on an enabled face drops stencil
                            // fail-visibly (unknown wire stays unknown); depth is
                            // still honored.
                            let stencil = if ds.front_stencil_enabled || ds.back_stencil_enabled {
                                use crate::backend::vulkan::engine::{
                                    SamplerCompareFunction, StencilFaceOps, StencilOp, StencilState,
                                };
                                const PASS_THROUGH: StencilFaceOps = StencilFaceOps {
                                    compare: SamplerCompareFunction::Always,
                                    fail_op: StencilOp::Keep,
                                    depth_fail_op: StencilOp::Keep,
                                    pass_op: StencilOp::Keep,
                                    read_mask: 0xFFFF_FFFF,
                                    write_mask: 0xFFFF_FFFF,
                                };
                                let front = if ds.front_stencil_enabled {
                                    engine_stencil_face(&ds.front_face)
                                } else {
                                    Ok(PASS_THROUGH)
                                };
                                let back = if ds.back_stencil_enabled {
                                    engine_stencil_face(&ds.back_face)
                                } else {
                                    Ok(PASS_THROUGH)
                                };
                                // Name the field that failed, not just "a
                                // stencil op somewhere did". `TranslateReason`
                                // carries which enum and which value, so a
                                // guest binding an unknown compare on the back
                                // face reads differently from one binding an
                                // unknown pass op on the front.
                                //
                                // The reason is kept **typed** all the way to
                                // the emitter. It used to be rendered into a
                                // nested `field=reason=… value=…` while the
                                // line's own `reason=` carried the coarse
                                // `stencil_op_unmapped` — so a grep for the
                                // specific check found nothing and a grep for
                                // the coarse one could not say which of the
                                // four stencil fields refused.
                                let stencil_reason: Option<translate::TranslateReason> =
                                    front.as_ref().err().or(back.as_ref().err()).copied();
                                let which_face = if front.is_err() { "front" } else { "back" };
                                match (front, back) {
                                    (Ok(front), Ok(back)) => {
                                        let (reference_front, reference_back) =
                                            req.stencil_ref.unwrap_or((0, 0));
                                        let clear_value = req
                                            .stencil_attach
                                            .as_ref()
                                            .map(|s| s.clear_stencil)
                                            .unwrap_or(0);
                                        Some(StencilState {
                                            front,
                                            back,
                                            reference_front,
                                            reference_back,
                                            clear_value,
                                        })
                                    }
                                    _ => {
                                        // Dedup on the *specific* slug, so an
                                        // unknown compare and an unknown pass op
                                        // on the same pipeline both get a line
                                        // rather than the second being silenced
                                        // as a repeat of the first.
                                        if let Some(reason) = stencil_reason {
                                            if degrade_log_first(req.pipeline_ref, reason.slug()) {
                                                crate::observe::Emit::decline(
                                                    "shader_state_degraded",
                                                    &reason,
                                                )
                                                .field("class", "stencil_op_unmapped")
                                                .field("face", which_face)
                                                .field("pipe", req.pipeline_ref)
                                                .field("ds_ref", req.depth_stencil_ref)
                                                .field("stencil_f", ds.front_stencil_enabled as u8)
                                                .field("stencil_b", ds.back_stencil_enabled as u8)
                                                .field("dims", format!("{w}x{h}"))
                                                .fail();
                                            }
                                        }
                                        None
                                    }
                                }
                            } else {
                                None
                            };
                            resources.depth = Some(crate::backend::vulkan::engine::DepthState {
                                test_enable: true,
                                write_enable: ds.depth_write_enabled,
                                compare,
                                clear_value,
                                // The stencil belongs to the pass, not to the
                                // draw: one draw writes a mask and the next
                                // tests it, so only the first draw of the pass
                                // clears and the rest load what it left.
                                // Clearing every draw is what left Tahoe's icon
                                // and glass surfaces an outline with no fill.
                                // Depth alone keeps its per-draw clear — the
                                // transient depth carries nothing between
                                // draws and nothing asks it to.
                                load: stencil.is_some() && !req.stencil_first_in_pass,
                                stencil,
                            });
                        }
                        None => {
                            // Unknown wire stays unknown: no depth rather than a
                            // guessed compare direction.
                            if degrade_log_first(req.pipeline_ref, "depth_compare_unmapped") {
                                crate::observe::fail(format!(
                                    "shader_state_ignored reason=depth_compare_unmapped \
                                     pipe={} ds_ref={} compare={} {}x{}",
                                    req.pipeline_ref,
                                    req.depth_stencil_ref,
                                    ds.depth_compare_function,
                                    w,
                                    h
                                ));
                            }
                        }
                    }
                }
            }
        }
        // The `fixed_gap` anomaly — decoded fixed-function state the Vulkan
        // request cannot represent — is the one thing here the always-on log
        // wants. It is deduped per (pipe, w, h, gap) so recurring depth/stencil
        // shadow draws cannot flood: an active compositor emitted 80k+ of these
        // per interaction before the dedup.
        if fixed_gap_first {
            crate::observe::off(format!(
                "linux_m2v_resources pipe={} {}x{} fixed_gap=[{}] attrs={} ssbo={} img={} smp={} rt_n={}",
                req.pipeline_ref,
                w,
                h,
                fixed_state_gap,
                resources.vertex_attributes.len(),
                resources.storage_buffers.len(),
                resources.sampled_images.len(),
                resources.samplers.len(),
                req.colors.len(),
            ));
        }
        // The per-draw resource census describes the *decoded* request — vertex
        // attribute declarations, storage-buffer bindings, sampler state, colour
        // targets. It is verbose-gated (REIMS_VGPU_DRAW_LOG →
        // /tmp/reims-vgpu-draw.log) because it costs a `format!` per binding.
        if census_verbose {
            let attr_meta: String = resources
                .vertex_attributes
                .iter()
                .map(|a| {
                    format!(
                        "L{}:fmt={:?}:off={}:str={}:sf={:?}:sr={}:n={}",
                        a.location,
                        a.format,
                        a.offset,
                        a.stride,
                        a.step_function,
                        a.step_rate,
                        a.content.len()
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            let ssbo_meta: String = resources
                .storage_buffers
                .iter()
                .map(|b| format!("b{}:n={}", b.binding, b.content.len()))
                .collect::<Vec<_>>()
                .join(";");
            let sampler_meta: String = resources
                .samplers
                .iter()
                .map(|s| {
                    format!(
                        "b{}:un={}:min={:?}:mag={:?}:mip={:?}:uvw={:?}/{:?}/{:?}",
                        s.binding,
                        s.unnormalized_coordinates as u8,
                        s.min_filter,
                        s.mag_filter,
                        s.mip_filter,
                        s.address_mode_u,
                        s.address_mode_v,
                        s.address_mode_w
                    )
                })
                .collect::<Vec<_>>()
                .join(";");
            crate::observe::line(format!(
                "linux_m2v_resources pipe={} {}x{} vtx={} attrs={} ssbo={} img={} smp={} rt_n={} rt=[{}] fixed_gap=[{}] seed={} idx={} idx_n={} meta=[{}] ssbo=[{}] sampler=[{}]",
                req.pipeline_ref,
                w,
                h,
                vertex_count,
                resources.vertex_attributes.len(),
                resources.storage_buffers.len(),
                resources.sampled_images.len(),
                resources.samplers.len(),
                req.colors.len(),
                color_target_diag(&req.colors),
                fixed_state_gap,
                resources.target_rgba8.is_some() as u8,
                resources.indexed.is_some() as u8,
                resources.indexed.as_ref().map(|i| i.index_count).unwrap_or(0),
                attr_meta,
                ssbo_meta,
                sampler_meta
            ));
        }

        resources.vert_spirv = v_words;
        resources.frag_spirv = f_words;
        resources.width = w;
        resources.height = h;
        resources.vertex_count = vertex_count;
        // Attachment-count census, taken before the MRT gate rather than inside
        // it. `build_secondary_targets` returns empty for a single-attachment
        // draw without emitting, and every MRT counter below it therefore reads
        // zero whether the guest issues no MRT draw at all or issues them and
        // the producer drops them. Those are different facts and the log could
        // not tell them apart. `mrt_draw_single` is the denominator that proves
        // this probe runs.
        crate::runtime::drain::note_store_route(if req.colors.len() > 1 {
            "mrt_draw_multi"
        } else {
            "mrt_draw_single"
        });
        // True MRT: render every color attachment (slot 1.. as engine secondary
        // residents) instead of dropping the shader's secondary outputs. Gated
        // on a resident primary + resolvable secondaries (empty ⇒ single-RT,
        // byte-identical).
        if let Some(primary_id) = resources.target_identity.clone() {
            let secs = build_secondary_targets(
                state,
                host,
                req.task_id,
                &req.colors,
                &pd,
                &primary_id,
                w,
                h,
                req.blend_color.unwrap_or([0.0; 4]),
            );
            // Second half of the census: `built` vs `dropped` separates "the
            // guest issued an MRT draw and we render every attachment" from
            // "it issued one and every attachment was refused", which the
            // `mrt_drop_*` reasons alone cannot say because the whole feature
            // is silent when no MRT draw arrives.
            if req.colors.len() > 1 {
                crate::runtime::drain::note_store_route(if secs.is_empty() {
                    "mrt_secondary_dropped"
                } else {
                    "mrt_secondary_built"
                });
            }
            resources.secondary_targets = secs;
        }
        // The engine's own typed `DrawError` (a `vk_*` VkCall slug, a
        // `DrawReason` refusal, an interim `_untyped`) propagates unchanged so
        // the boundary below names the engine's specific check as the primary
        // `reason=` rather than flattening it into a `vk_engine: {e}` blob.
        crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::Engine);
        let out = crate::backend::vulkan::engine::execute_draw_request(&resources)?;
        // Everything from here to the end of the chain is Store routing, and the
        // `?` above deliberately leaves a declined draw charged to `engine`:
        // where it declined is the engine's own typed reason to report, not this
        // census's.
        crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::Store);
        // RGB nonzero (ignore alpha) so black+alpha is not mistaken for content.
        // Resident/import path uses skip_readback → empty `out.pixels` is **expected**
        // and must not be read as "GPU drew black" (use import_content res_rgb_nz).
        // The scan is O(pixels) on the drain worker and the line it feeds is the
        // only consumer, so it runs only when that sink is open.
        if census_verbose {
            if out.pixels.is_empty() {
                crate::observe::line(format!(
                    "linux_m2v_pixels pipe={} {}x{} skip_readback=1 (no CPU pixels; see import_content)",
                    req.pipeline_ref, w, h
                ));
            } else {
                let mut rgb_nz = 0usize;
                let mut max_rgb = 0u8;
                for px in out.pixels.chunks_exact(4) {
                    let m = px[0].max(px[1]).max(px[2]);
                    if m != 0 {
                        rgb_nz += 1;
                    }
                    if m > max_rgb {
                        max_rgb = m;
                    }
                }
                crate::observe::line(format!(
                    "linux_m2v_pixels pipe={} {}x{} rgb_nz={} max_rgb={} px0=[{},{},{},{}]",
                    req.pipeline_ref,
                    w,
                    h,
                    rgb_nz,
                    max_rgb,
                    out.pixels.first().copied().unwrap_or(0),
                    out.pixels.get(1).copied().unwrap_or(0),
                    out.pixels.get(2).copied().unwrap_or(0),
                    out.pixels.get(3).copied().unwrap_or(0),
                ));
            }
        }
        // No content-gated CPU composites: premultiplied One/OneMinusSourceAlpha
        // is hardware Load+blend, and keep-seed / alpha0-hole compositing is not
        // something real Metal does. The blend state below is what makes that
        // true; a draw that lands wrong shows up as a typed decline on this
        // boundary, not as a pixel census.
        // Engine pixels are authoritative (empty when skip_readback; the Store
        // path materializes bytes for surface_cache and the guest writeback).
        //
        // A Store used to be able to return `M2vDrawSpan::ResidentBgra`
        // instead — no pixels at all, the resident staying authoritative until
        // the import-present rail DMA'd it into the mapping's guest pages.
        // That span is unreachable without the import and its variant is gone.
        let pixels_bgra = out.pixels_bgra;
        let pixels = out.pixels;
        if resident_render_chain {
            return Ok(M2vDrawSpan::ResidentChain);
        }
        if gva_resident_store {
            return Ok(M2vDrawSpan::ResidentGvaStore);
        }
        if surface_resident_store {
            return Ok(M2vDrawSpan::ResidentSurfaceStore);
        }
        Ok(M2vDrawSpan::Pixels {
            bytes: pixels,
            bgra: pixels_bgra,
        })
    }
}

/// Land a multi-draw chain image into guest color targets (full-frame store).
/// Used when a later draw in the packet fails after earlier encodes succeeded.
/// Engine-resident identity for a color0 render-pass chain.
///
/// This identity lives only from the first serialized record through its final
/// Store. Type-11 targets use their current protocol mapping identity; linear
/// type-2/3 targets use the GVA identity below. Unlike deferred writeback, this
/// lifetime is safe on portability-subset devices because the final record
/// materializes guest bytes before the packet completes.
pub(super) fn render_chain_identity(
    state: &DeviceState,
    req: &DrawEncodeRequest,
) -> Option<crate::backend::vulkan::engine::TargetIdentity> {
    let c0 = req.colors.first()?;
    let (width, height) = (c0.width, c0.height);
    if width == 0 || height == 0 {
        return None;
    }
    if c0.mapping_id != 0 {
        return Some(crate::runtime::present_identity::surface_identity(
            state,
            c0.mapping_id,
            width,
            height,
        ));
    }
    gva_chain_identity(req)
}

/// The registry resident a type-11 composite Store renders into, if this record
/// is one.
///
/// Unlike the GVA rail this is **not** a `skip_readback` deferral: a composite
/// Store still reads its target back, because that readback is what feeds
/// `surface_cache`, the deferred window's owned frame and the guest writeback.
/// The resident exists so the *next* LOAD on this surface can start from the
/// image that already holds these pixels instead of re-uploading them through
/// the staging span — `draw_phase` prices that upload at ~155 ms per second of
/// wall clock under a browser workload, a copy of bytes the GPU just produced.
///
/// Single source of truth on purpose. The draw's `target_identity`, the LOAD's
/// currency check and the Store's epoch stamp must name the same slot; deriving
/// it three times from three spellings of the predicate is how a stamp ends up
/// vouching for an image the draw never rendered into.
///
/// `chain_from_resident` is **not** a refusal, and used to be. The last record of
/// a resident render-pass chain is both the chain's consumer and the packet's
/// guest-visible Store, and refusing it here cost the whole composite readback
/// population: `t11_keep_chain_from_resident` measured equal to
/// `surface_deferred` in all twelve windows of one boot, 112-132 Stores a second
/// at 366-372 MB/s, with every other keep-reason at zero. Nothing about the chain
/// changes which slot this record renders into — `retarget_render_pass_draw`
/// builds every record of a packet from one attachment template, so records N-1
/// and N carry the same `mapping_id` and geometry and therefore the same
/// [`render_chain_identity`] — and the intermediates already render into that
/// resident under `skip_readback` with `LoadOp::LoadFromTarget`. The last record
/// differs only in what happens *after* the draw.
pub(super) fn type11_store_identity(
    state: &DeviceState,
    req: &DrawEncodeRequest,
    writeback_guest: bool,
) -> Option<crate::backend::vulkan::engine::TargetIdentity> {
    if !writeback_guest {
        return None;
    }
    type11_render_identity(state, req)
}

/// The registry resident this record renders its type-11 color0 *into*, whatever
/// its role in the packet — the strict superset of [`type11_store_identity`],
/// which is this same slot restricted to the record that also stores it for the
/// guest.
///
/// The two are separate because the LOAD and the Store ask different questions of
/// the same slot. "May I start from what is already in this image?" is answerable
/// by any record that renders into it; "may I leave my frame there instead of
/// copying it to guest pages?" is only answerable by the packet's last record.
/// Conflating them cost the whole seed elision on multi-record packets: record 1
/// of a chain has `writeback_guest == false`, so the currency check keyed on the
/// Store identity never ran for it, and its LOAD fell through to
/// `resolve_type11_load_seed` — which, with the host cache ceded to the resident
/// rail, reads the mapping's guest pages and therefore lands the very window the
/// rail armed. One boot measured that loop directly: `surface_flush /
/// surface_resident` = 1369/1373, one flush per arm, with `type11_load_seed`
/// reporting `outcome=guest_pages` 110 times against 17 `cache_hit` and
/// `hostgen=0` on every one.
///
/// A record with `mapping_id != 0` and a real Store action renders into this slot
/// on every route: the chain block claims it for `chain_from_resident` and for a
/// `!writeback_guest` intermediate, and the composite-Store rail claims it for the
/// last record. So the condition here is the same one those blocks share, asked
/// once.
fn type11_render_identity(
    state: &DeviceState,
    req: &DrawEncodeRequest,
) -> Option<crate::backend::vulkan::engine::TargetIdentity> {
    let c0 = req.colors.first()?;
    if c0.mapping_id == 0 || c0.store_action != PASS_STORE_ACTION_STORE {
        return None;
    }
    render_chain_identity(state, req)
}

/// Whether this record's color0 LOAD is one the resident could serve at all —
/// it must be a LOAD, and no explicit seed may already have been selected for
/// it by RT provenance. Separate from the currency question so the two counters
/// on the branch below divide candidates, not all draws.
fn type11_load_is_a_seed_candidate(c0: &ColorRtRequest) -> bool {
    c0.load_action == PASS_LOAD_ACTION_LOAD && c0.target_seed_rgba.is_none()
}

/// The `(resident, mapping epoch)` pair a record's type-11 LOAD has to compare to
/// decide whether the image it is about to render into already holds the seed —
/// or `None` when this record's LOAD is not one a resident could serve at all.
///
/// **Takes no `writeback_guest`, deliberately.** The record's role in the packet
/// is not part of this question, and a signature that cannot see the role cannot
/// be keyed on it. It was: the check read the *Store* identity, so record 1 of a
/// chain — `writeback_guest == false` — never asked, took a CPU seed, found the
/// host cache ceded to the resident rail, read the mapping's guest pages, and by
/// reading them landed the window the packet's Store had just armed. That advanced
/// the epoch and cost the next LOAD its elision in turn, so the rail degraded into
/// a rescheduling with a GPU round trip added: `surface_flush / surface_resident`
/// = 1369/1373 on one boot. Structure rather than a test, because a unit test on
/// the resolver passes whatever the call site then decides to pass it.
pub(super) fn type11_load_currency_query(
    state: &DeviceState,
    req: &DrawEncodeRequest,
) -> Option<(crate::backend::vulkan::engine::TargetIdentity, Option<u32>)> {
    let c0 = req.colors.first()?;
    if !type11_load_is_a_seed_candidate(c0) {
        return None;
    }
    let identity = type11_render_identity(state, req)?;
    let mapping_epoch = state
        .mappings
        .get(&c0.mapping_id)
        .map(|m| m.surface_content_epoch);
    Some((identity, mapping_epoch))
}

/// Has the guest written this surface's pages since the Store that produced
/// the resident stamped them?
///
/// The device-side half of the currency test — the `surface_content_epoch`
/// comparison one function up — can only witness writers inside this crate.
/// A type-11 surface's pages are plain guest RAM, and the guest CPU stores
/// into them with no device operation, so the epoch does not move and the
/// resident silently stops matching what the seed would upload. This is the
/// only witness for that, and it is the hypervisor's.
///
/// Answers `true` — "written, do not reuse" — for every case that is not a
/// live token whose current generation equals the one the Store recorded:
/// a host that cannot observe guest writes, a mapping with no token, a token
/// released with its page list, and a Store that never stamped. A false
/// "unwritten" is the one answer that produces a wrong frame; a false
/// "written" only costs a seed.
///
/// # What the ordering actually promises
///
/// The host observes writes at its own harvest points, and the shims harvest at
/// the register write that hands the device work. So the promise is: every guest
/// store ordered before a submission is visible to the draws that submission
/// carries. It is *not* "immediately" — a store racing the draws that read the
/// same surface is a race the guest already has against the GPU, and the next
/// submission's harvest sees it either way, so the rail cannot latch.
///
/// The generation is read again at the Store, so a write that lands between a
/// record's LOAD and its Store is stamped as though the resident contained it.
/// That window is one packet's execution, it requires the guest to write a
/// surface it has just asked the GPU to render into, and — unlike the epoch-only
/// rail this replaced — the very next guest write moves the generation again and
/// clears it.
pub(super) fn type11_guest_wrote_since_store<M: HostOps>(
    state: &DeviceState,
    host: &M,
    mapping_id: u32,
) -> bool {
    match mapping_guest_write_verdict(state, host, mapping_id) {
        GuestWriteVerdict::Clean => false,
        GuestWriteVerdict::NoMapping => {
            crate::runtime::drain::note_store_route("t11_gw_ref_no_mapping");
            true
        }
        GuestWriteVerdict::NoStamp => {
            crate::runtime::drain::note_store_route("t11_gw_ref_no_stamp");
            true
        }
        GuestWriteVerdict::Wrote => {
            crate::runtime::drain::note_store_route("t11_gw_ref_moved");
            true
        }
        GuestWriteVerdict::Unreadable => {
            crate::runtime::drain::note_store_route("t11_gw_ref_unreadable");
            true
        }
    }
}

/// Whether the hypervisor watched the guest write anywhere in the allocation a
/// type-4 surface's host-side copies were taken from.
///
/// The first of two stages, and the coarse one: the tracking token covers the
/// mapping's whole page list and its generation moves for a write to any page
/// in it, so this answers about the *allocation*, not about the pixels a bind
/// would read. [`guest_write_site`] is what narrows it.
///
/// Only `Wrote` is evidence. `no_stamp` says "nobody asked the host to watch
/// these pages", which is a statement about this device's arming and not about
/// the guest; on the boot that first measured the ladder it was 14 092 of 14 396
/// cache binds, so refusing on it would turn the rung off on the strength of a
/// rail that was never armed.
fn guest_wrote_allocation(verdict: GuestWriteVerdict) -> bool {
    matches!(verdict, GuestWriteVerdict::Wrote)
}

/// Where the guest's writes since the stamping Store landed, relative to the
/// pixel window a sampled bind reads.
#[derive(Clone, Debug, PartialEq, Eq)]
enum GuestWriteSite {
    /// At least one written page overlaps the sampled window, so every
    /// host-side copy of it is out of date. Carries the mapping-offset ranges
    /// the guest owns, which is exactly the `skip` list the merge needs.
    Pixels(Vec<(u64, u64)>),
    /// The guest wrote the allocation but not the bytes this bind samples.
    Elsewhere,
    /// The host could not name the written pages, or the window is unknown.
    /// Indistinguishable from [`Self::Pixels`] to a caller that must be right,
    /// but it cannot be merged either — there is no page list to preserve.
    Unknown,
}

/// Narrow a `Wrote` verdict to the pixel window, so a write that misses it does
/// not discard a resident that is still exactly the surface.
///
/// The tracking token is per *page list*, and a type-4 allocation is more than
/// its sampled plane: `type11_sample_window` reports a `base_off` precisely
/// because the pixels do not start at offset 0, and an allocation can carry a
/// second plane and end padding past `span_end`. A guest store into any of that
/// moves the set-wide generation. Refusing on it discarded whole 1920x1080
/// compositor scanouts whose pixels the GPU had rendered and nothing had
/// touched — measured live as a black desktop at 17 Hz, against 120 Hz and a
/// painted one on the same boot script with the rung ungated.
///
/// Fails closed. Everything the host cannot answer exactly — no token, no
/// enumerable page list, no resolvable sample window, or written GPAs this
/// mapping does not own — is [`GuestWriteSite::Unknown`], which the caller
/// treats as [`GuestWriteSite::Pixels`]. Serving a stale copy is a wrong frame
/// that is then held; re-reading the guest's pages costs a copy.
fn guest_write_site<M: HostOps>(
    state: &DeviceState,
    host: &M,
    mapping_id: u32,
    width: u32,
    height: u32,
) -> GuestWriteSite {
    let Some(m) = state.mappings.get(&mapping_id) else {
        return GuestWriteSite::Unknown;
    };
    let format = if m.format != 0 {
        m.format
    } else {
        pixel_format::MTL_FORMAT_BGRA8_UNORM
    };
    let Some((base_off, _bpr, span_end)) =
        crate::runtime::mapping_write::type11_sample_window(m, mapping_id, width, height, format)
    else {
        return GuestWriteSite::Unknown;
    };
    let Some(pages) = host.guest_written_pages(m.guest_write_token, m.guest_write_gen_at_store)
    else {
        return GuestWriteSite::Unknown;
    };
    let ranges = mapper::mapping_offsets_of_pages(state, mapping_id, &pages);
    if ranges.is_empty() {
        // The set-wide generation moved, so some page of this list was written,
        // yet none of them mapped back to an offset. That is a disagreement
        // between the token and the page list this call resolved against, not a
        // finding about the guest.
        return GuestWriteSite::Unknown;
    }
    if ranges_touch_window(&ranges, base_off, span_end) {
        GuestWriteSite::Pixels(ranges)
    } else {
        GuestWriteSite::Elsewhere
    }
}

/// Put both halves of a surface the guest wrote under a live resident into the
/// guest's own pages, and withdraw the resident's claim to be the surface.
///
/// This is the answer the ladder was missing. Neither copy is the surface on its
/// own: the resident holds what the GPU rendered, the pages hold what the guest
/// CPU painted since, and picking either one loses the other's work. Refusing
/// the resident and reading the pages — which is what the gate did on its own —
/// produced a black desktop at 17 Hz, because the skip-readback Store rail
/// leaves a composite's pixels GPU-side on purpose and the pages it read had
/// never been written.
///
/// So: read the resident out, land it in every page the guest did *not* write
/// (`write_bgra8_skipping`, the same per-page rule the deferred writeback
/// follows), and retire the resident. The pages then hold the merge, and they
/// are what every rung below reads. The next Store into this surface makes a
/// fresh resident and stamps it, so this is paid once per burst of guest writes
/// and not once per bind.
///
/// Returns whether the merge landed. On `false` the caller has a surface whose
/// halves are still split and must say so rather than bind either one.
fn merge_guest_writes_into_pages<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    width: u32,
    height: u32,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    guest_owned: &[(u64, u64)],
) -> bool {
    let readback = match crate::backend::vulkan::engine::read_target(identity) {
        Ok(rb) => rb,
        Err(e) => {
            crate::observe::fail(format!(
                "sampled_resident_merge_fail mid={mapping_id} {width}x{height} stage=readback err={e:?}"
            ));
            return false;
        }
    };
    let bgra = readback.into_bgra8();
    let stride = width.saturating_mul(RGBA8_BPP);
    if !mapping_write::write_bgra8_skipping(
        state,
        host,
        mapping_id,
        &bgra,
        stride,
        width,
        height,
        guest_owned,
    ) {
        crate::observe::fail(format!(
            "sampled_resident_merge_fail mid={mapping_id} {width}x{height} stage=writeback \
             runs={} bytes={}",
            guest_owned.len(),
            bgra.len()
        ));
        return false;
    }
    // `write_bgra8_skipping` has already retired both host-side copies and
    // re-taken the guest-write stamp, because those follow from the skipping
    // write and not from who asked for it. `identity` is the same value it
    // recomputes; it is a parameter here because the readback needs it.
    crate::runtime::drain::note_store_route("t11sample_resident_merged");
    true
}

/// Whether any written mapping-offset range overlaps `[base_off, span_end)`.
///
/// Half-open on both sides: a range that abuts the window without entering it
/// is the padding page after the last row, not the last row.
fn ranges_touch_window(ranges: &[(u64, u64)], base_off: u64, span_end: u64) -> bool {
    ranges
        .iter()
        .any(|&(lo, hi)| lo < span_end && hi > base_off)
}

/// Census only: which rung of the type-4 sampled ladder served this bind, and,
/// for the rungs that serve a host-side *copy* of guest memory, what the
/// hypervisor said about those bytes when the rung was chosen.
///
/// The ladder in [`resolve_sampled_source`] had no counters at all, which is
/// why the GVA cache's identical defect had to be found through a probe-gated
/// byte compare instead of read off a boot.
///
/// `t11rung_resident_refused` counts binds where a ready resident existed and
/// [`guest_replaced_host_copies`] sent the bind to the guest's pages instead.
/// It is the direct measure of how much wrong content that rung used to serve.
///
/// # Baseline, before the resident rung was gated
///
/// One 14-round Finder recomposite boot under load, x86 / Vulkan:
///
/// ```text
/// t11rung_resident    92730     (no currency test at all)
/// t11rung_host_cache  14396     gw_clean 0  gw_no_stamp 14092  gw_wrote 304
/// t11rung_zero_copy    4977
/// t11rung_guest_memo     51
/// t11rung_miss            0
/// ```
///
/// That is the *pre-gating* world and it is kept only as the before-picture: the
/// resident rung ran with no column at all because it asked no question, and
/// every one of the 14 396 cache binds served bytes the hypervisor could not
/// vouch for.
///
/// # What it reads now, which is the reading to reason from
///
/// Both statements above have since been overtaken, so do not carry the bolded
/// `gw_clean == 0` forward — it was true of a build that no longer exists. One
/// driven x86 / Vulkan boot (Safari page loads, scrolls, title-bar drags):
///
/// ```text
/// t11rung_resident    632   gw_clean 242  gw_no_stamp 426
/// t11rung_host_cache  131   gw_clean   8  gw_no_stamp 123
/// ```
///
/// The gate did what it was for: the resident rung now asks the question, and
/// `gw_clean` is non-zero on both copy-serving rungs. `no_stamp` still dominates
/// — most mappings are never armed with a guest-write token — so the currency
/// test mostly answers "cannot vouch", but it no longer *always* does.
///
/// # The whole ladder, and the inversion since
///
/// A later driven boot — Chess, Maps, the WebGL aquarium, page-downs, a
/// title-bar drag, apple.com — summing `store_routes` deltas across the boot:
///
/// ```text
/// t11rung_resident   7166  (76 %)   gw_clean 6552   gw_no_stamp  614
/// t11rung_host_cache 1791  (19 %)   gw_clean  432   gw_no_stamp 1359
/// t11rung_zero_copy   415  (4.4 %)
/// t11rung_guest_memo   47  (0.5 %)
/// t11rung_resident_refused 1
/// ```
///
/// Two things to carry forward. **`gw_clean` and `gw_no_stamp` read the other
/// way round on the resident rung** — 242/426 against 6552/614. Whether that is
/// the build or the workload is not established: the two readings differ in
/// both, and this boot drove two native apps and a WebGL page the earlier one
/// did not. What is safe is that the paragraph above no longer describes any
/// reading we have, so do not reason from it. The host-cache rung still reads
/// the old way *in the same boot*, which says the two rungs are not arming
/// guest-write tokens alike whatever the cause.
///
/// And **all four rungs carry traffic**. The thin ones are thin, not empty:
/// `guest_memo` is 0.5 % of samples and `zero_copy` 4.4 %, so neither is a dead
/// fallback that can be deleted to shorten the ladder. Deleting either loses
/// content the guest asked for. Shortening this ladder is a question about the
/// PVG contract — where a sampled surface's content is *supposed* to live — and
/// not one a traffic census can settle.
///
/// The other six cells of the (rung × verdict) table have never fired on this
/// pathway: `gw_no_mapping` and `gw_unreadable` on either rung, and
/// `gw_wrote_elsewhere` on either rung. They are denominators, not dead code —
/// their being empty is what says a served copy is never a copy the hypervisor
/// actively contradicted.
fn note_type11_sample_rung(rung: &'static str, guest_write: GuestWriteVerdict) {
    crate::runtime::drain::note_store_route(rung);
    if let Some(gw) = sample_rung_gw_route(rung, guest_write) {
        crate::runtime::drain::note_store_route(gw);
    }
}

/// The census column for a rung's guest-write verdict, or `None` for a rung the
/// verdict says nothing about.
fn sample_rung_gw_route(rung: &str, guest_write: GuestWriteVerdict) -> Option<&'static str> {
    Some(match (rung, guest_write) {
        ("t11rung_resident", GuestWriteVerdict::Clean) => "t11rung_resident_gw_clean",
        ("t11rung_resident", GuestWriteVerdict::NoMapping) => "t11rung_resident_gw_no_mapping",
        ("t11rung_resident", GuestWriteVerdict::NoStamp) => "t11rung_resident_gw_no_stamp",
        ("t11rung_resident", GuestWriteVerdict::Unreadable) => "t11rung_resident_gw_unreadable",
        // Served under a `Wrote` verdict, so the write missed the sampled window
        // — `guest_write_site` said `Elsewhere` or the bind would have refused.
        ("t11rung_resident", GuestWriteVerdict::Wrote) => "t11rung_resident_gw_wrote_elsewhere",
        ("t11rung_host_cache", GuestWriteVerdict::Clean) => "t11rung_host_cache_gw_clean",
        ("t11rung_host_cache", GuestWriteVerdict::NoMapping) => "t11rung_host_cache_gw_no_mapping",
        ("t11rung_host_cache", GuestWriteVerdict::NoStamp) => "t11rung_host_cache_gw_no_stamp",
        ("t11rung_host_cache", GuestWriteVerdict::Unreadable) => "t11rung_host_cache_gw_unreadable",
        ("t11rung_host_cache", GuestWriteVerdict::Wrote) => "t11rung_host_cache_gw_wrote_elsewhere",
        // The rungs that read the guest's own pages do not care what the guest
        // wrote, because they read exactly that.
        _ => return None,
    })
}

/// Whether a resident's stamp still vouches for the mapping's current content.
///
/// The `is_some` guard is the whole function. Both values are `Option`, and
/// `None == None` is `true` in Rust — so a bare equality would read "the
/// mapping has no entry" and "this image was never stamped" as agreement and
/// load undefined memory as though it were the guest's prior frame. That is
/// precisely the black-layer class. Absence on either side is a refusal.
fn type11_resident_is_current(mapping_epoch: Option<u32>, resident_epoch: Option<u32>) -> bool {
    mapping_epoch.is_some() && mapping_epoch == resident_epoch
}

/// Record that the resident this Store rendered into holds the mapping's
/// content as of `epoch`, so the surface's next LOAD can skip its CPU seed.
///
/// Keyed through [`type11_store_identity`] — the same call the draw's
/// `target_identity` came from — so the slot stamped is the slot rendered into.
/// A miss is expected and silent: the identity resolves to `None` when this
/// record never took the resident path, and `stamp_resident_content_epoch`
/// refuses a slot that was evicted between the draw and here. Both leave the
/// stamp absent, which costs a seed and never a wrong frame.
///
/// Also records the host's guest-write generation for the surface's pages, and
/// registers them for tracking the first time. That is the half of the currency
/// test `epoch` cannot cover: every writer that advances `epoch` is inside this
/// crate, while the pages are plain guest RAM the guest CPU stores into with no
/// device operation. Registration happens here rather than at mapping resolve
/// because this is the first moment the device has a host-side copy whose reuse
/// would depend on the answer.
fn stamp_type11_resident<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    req: &DrawEncodeRequest,
    writeback_guest: bool,
    epoch: u32,
) {
    if let Some(identity) = type11_store_identity(state, req, writeback_guest) {
        crate::backend::vulkan::engine::stamp_resident_content_epoch(&identity, epoch);
    }
    if let Some(mapping_id) = req.colors.first().map(|c| c.mapping_id).filter(|m| *m != 0) {
        crate::runtime::mapper::stamp_guest_write_gen(state, host, mapping_id);
    }
}

/// Order-independent hash of the guest physical pages behind a GVA span.
///
/// The set is the identity, not the sequence: a walk that reports the same pages
/// in another order must produce the same value, or re-rendering one buffer
/// would read as a new allocation. `0x9E37_79B9_7F4A_7C15` is the odd 64-bit
/// golden-ratio constant (2^64 / phi rounded to odd), used here only to spread
/// page GPAs — which are dense multiples of the page size, so their low bits are
/// all zero — across the whole word before the XOR fold.
fn gva_page_set_hash(pages: &std::collections::HashSet<u64>) -> u64 {
    let mut hash: u64 = 0;
    for p in pages {
        hash ^= p.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    hash
}

/// Allocation identity of this draw's color0 GVA render target: the hash of the
/// guest physical pages its span resolves to right now.
///
/// The engine registry keys a `TargetIdentity::Gva` on all four of its fields,
/// so this is the only field that can separate two guest allocations reusing one
/// address at one geometry. Identical pages means literally the same guest
/// memory and sharing the image is correct; different pages means a different
/// allocation, and sharing would hand the second one the first one's pixels as
/// its prior content — which is exactly what the cross-pass resident Load in
/// [`encode_draw_chain`] reads.
///
/// Resolved once per draw, before any GPU work, and carried on
/// [`DrawEncodeRequest::gva_alloc_gen`] so every identity the draw builds agrees.
/// Returns 0 when color0 is not a GVA target and when the walk does not cover
/// the whole span: an incomplete walk names no allocation, and a hash of the
/// pages that happened to resolve would be an identity the guest never had.
///
/// A generation that disagrees with the one a resident was created under can
/// only *miss* the registry lookup, which costs a CPU seed and never produces
/// wrong pixels. Sharing a slot is the wrong-content direction, so every
/// ambiguity here resolves toward the miss.
fn gva_alloc_generation<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &mut M,
    req: &DrawEncodeRequest,
) -> u64 {
    let Some(c0) = req.colors.first() else {
        return 0;
    };
    if c0.mapping_id != 0 || c0.target_gva == 0 {
        return 0;
    }
    // Same span as the deferred arm walks (`arm_gva_deferred_store`) so the two
    // describe one region: the guest bytes a Store into this target writes.
    gva_span_alloc_generation(
        state,
        host,
        req.task_id,
        c0.target_gva,
        c0.row_stride,
        c0.height,
    )
}

/// The page-set generation of one `row_stride * height` GVA span under one task.
///
/// Every GVA render target is named this way, not only color0: the secondary MRT
/// attachments go through here too ([`build_secondary_targets`]). One spelling,
/// so no two callers can disagree about what names an allocation.
///
/// A short walk yields 0. An incomplete walk names no allocation, and hashing the
/// pages that happened to resolve would be an identity the guest never had.
fn gva_span_alloc_generation<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &mut M,
    task_id: u32,
    gva: u64,
    row_stride: u32,
    height: u32,
) -> u64 {
    if gva == 0 {
        return 0;
    }
    let span = (row_stride as u64).saturating_mul(height as u64);
    if span == 0 {
        return 0;
    }
    let pages =
        gva_mem::task_gva_page_gpa_set(host, &state.tasks, task_id, gva, span, state.page_shift);
    if (pages.len() as u64) < gva_mem::pages_spanned(gva, span, state.page_size()) {
        return 0;
    }
    gva_page_set_hash(&pages)
}

/// Engine-resident identity for a GVA (type-2/3) color0 render target.
///
/// Single source of truth for the resident GVA chain rail: the draw path,
/// the alias-sample bind, and the abandon-path landing must all agree on the
/// exact identity or the registry lookups miss.
///
/// `generation` is the draw's [`DrawEncodeRequest::gva_alloc_gen`] — the hash of
/// the guest pages backing the target — because a GVA is only a name and the
/// guest recycles names. Without it the registry keys this resident on
/// `(gva, width, height)` alone, and the cross-pass resident Load below hands a
/// new allocation the previous one's pixels as its prior content.
pub(crate) fn gva_chain_identity(
    req: &DrawEncodeRequest,
) -> Option<crate::backend::vulkan::engine::TargetIdentity> {
    let c0 = req.colors.first()?;
    if c0.mapping_id != 0 || c0.target_gva == 0 {
        return None;
    }
    let (w, h) = (c0.width, c0.height);
    if w == 0 || h == 0 {
        return None;
    }
    Some(crate::backend::vulkan::engine::TargetIdentity::Gva {
        gva: c0.target_gva,
        width: w,
        height: h,
        generation: req.gva_alloc_gen,
    })
}

/// Read an abandoned resident render-pass chain so the exec loop can land the
/// last good record's pixels (`writeback_chain_rgba`). Every failure is
/// fail-visible; the guest keeps its pre-pass bytes on loss.
pub(crate) fn read_resident_chain(state: &DeviceState, req: &DrawEncodeRequest) -> Option<Vec<u8>> {
    let identity = render_chain_identity(state, req)?;
    match crate::backend::vulkan::engine::read_target(&identity) {
        // Every caller of this function — `writeback_chain_rgba` and the GVA
        // arm-refusal fallback — has an RGBA contract, so the exchange happens
        // here, once, rather than at three call sites that would each have to
        // remember which namespace they were reading. `into_rgba8` uses the order
        // the engine reports for the image it copied, so it is a no-op on the
        // pooled and GVA residents and a single pass on a BGRA surface. Both are
        // abandon paths, so neither is on a hot rail.
        Ok(rb) => Some(rb.into_rgba8()),
        Err(e) => {
            crate::observe::fail(format!(
                "chain_resident_land_fail reason=read_target target={identity:?} \
                 mid={} gva={:#x} {}x{} err={e}",
                req.colors.first().map(|c| c.mapping_id).unwrap_or(0),
                req.colors.first().map(|c| c.target_gva).unwrap_or(0),
                identity.width(),
                identity.height()
            ));
            None
        }
    }
}

/// Deferred GVA windows keep engine registry slots pinned (the LRU sweep
/// skips pinned slots and soft-exceeds `REGISTRY_CAP=64`); arming past this
/// count lands the oldest window first so pinned pressure stays bounded.
///
/// Measured across every boot in a 72 MB accumulated log — Chess, Maps, the
/// WebGL aquarium, page-downs, a title-bar drag, apple.com: the live population
/// never exceeded the one window being armed, and nothing was ever evicted, so
/// **this cap has never bound**. That is what `flush_all_windows_before_fence`
/// forces — every completion stamp lands every window first, so windows cannot
/// accumulate across a fence, only within one drain tranche. 16 is therefore not
/// justified by that reading; it is unfalsified by it. Retiring the cap needs a
/// workload that arms several Stores between two stamps, and the ordering
/// machinery it drives (`gva_deferred_seq`, `take_oldest_gva_deferred_window`)
/// is only reachable at a population above one.
///
/// If it ever does bind, each forced landing says so as
/// `gva_deferred_flush ... trigger=window_cap`.
const GVA_DEFERRED_WINDOW_CAP: usize = 16;

/// Bound on live type-11 render windows. Each one pins a display-sized target
/// resident, so an unbounded population is the "~260 stale residents
/// (~516 MiB) pinned for the guest lifetime" shape. Sized like the GVA cap: a
/// composite touches a handful of layers, so this is headroom, not a working
/// set the guest routinely exceeds.
///
/// Same logs, same verdict: the live population reached only the window being
/// armed, and nothing was ever evicted. "A composite touches a handful of
/// layers" predicted a population of a few; the measured one is one. See the
/// note on `GVA_DEFERRED_WINDOW_CAP` for why, and for what would change the
/// answer. If it ever does bind, `evict_render_windows_to_cap` emits
/// `surface_window_cap_evicted`.
const SURFACE_DEFERRED_WINDOW_CAP: usize = 16;

/// Defer gate for a type-11 (surface) render Store.
///
/// All gates are protocol-shape checks, never content: the later flush has to
/// be able to replay the synchronous Store *exactly*, so anything the sync
/// route would have needed must be resolvable now. In particular the mapping's
/// plane window must resolve, because the deferred window's guest byte range —
/// which is what every reader intersects against to decide whether to flush —
/// comes from it. A window with no range would be armed and then never found
/// by a reader, which is the silent-stale-read failure this rail exists to
/// avoid.
fn surface_store_defer_eligible(
    state: &DeviceState,
    req: &DrawEncodeRequest,
) -> Option<crate::model::ComputeStorageResidencyKey> {
    let c0 = req.colors.first()?;
    if c0.mapping_id == 0 {
        return None;
    }
    // The engine-level gate every deferred-writeback rail asks, so one switch
    // turns them all off together.
    if !crate::backend::vulkan::engine::deferred_gpu_only_content_allowed() {
        return None;
    }
    let (w, h) = (c0.width, c0.height);
    if w == 0 || h == 0 {
        return None;
    }
    let m = state.mappings.get(&c0.mapping_id)?;
    // The sync route calls `write_rgba8_image_changed`, which refuses unless
    // the mapping's latched geometry equals the draw's. Deferring a Store that
    // is going to be refused just moves the refusal somewhere it reads as a
    // lost flush, so gate on the same thing up front.
    let (surface_offset, surface_bpr, span_end) =
        crate::runtime::mapping_write::type11_sample_window(m, c0.mapping_id, w, h, c0.format)?;
    Some(crate::model::ComputeStorageResidencyKey {
        mapping_id: c0.mapping_id,
        map_generation: m.map_generation,
        surface_offset,
        surface_bpr,
        span_end,
        width: w,
        height: h,
        pixel_format: c0.format,
        texture_ref: 0,
    })
}

/// Arm the deferred window for a type-11 render Store, so the CPU writeback
/// into the mapping's guest pages happens on demand instead of every Store.
///
/// The caller has already read the target back and refreshed
/// `surface_cache` with this frame, so the pixels the flush will write are the
/// ones every other consumer already sees. **Only the guest-page copy is
/// deferred** — the readback, the cache, the Load seed and the present capture
/// are untouched, which is what keeps this rail out of the front-buffer
/// resolve problem that a resident-authoritative type-11 Load would reopen (see
/// the note at the type-11 Load arm).
///
/// The index is the mapping-keyed one the compute rail already uses, so every
/// guest-page reader drains this window through the `flush_intersecting` choke
/// point it already calls — no new trigger sites, and no way to cover one rail
/// and miss the other.
#[allow(
    clippy::too_many_arguments,
    reason = "the arm names the frame it is deferring and the geometry it was drawn at"
)]
fn arm_surface_deferred_store_with<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    req: &DrawEncodeRequest,
    mapping_id: u32,
    width: u32,
    height: u32,
    bgra: Vec<u8>,
) -> Result<u32, Vec<u8>> {
    let Some(key) = prepare_surface_deferred_window(state, host, req, mapping_id, width, height)
    else {
        return Err(bgra);
    };
    // Referenced twice, copied never: the Load seed and the present capture read
    // it through `surface_cache`, and the window below owns it so the writeback
    // it defers can always be performed.
    //
    // This used to allocate a second frame and swizzle into it — 776 us per
    // Store, 84 % of everything `draw_phase` could not attribute. The Store's
    // attachment is a BGRA `Surface` resident now, so the readback arrives in
    // this order and the whole pass is gone rather than moved.
    let frame = std::sync::Arc::new(bgra);
    crate::runtime::surface_cache::store_shared(
        state,
        mapping_id,
        width,
        height,
        std::sync::Arc::clone(&frame),
    );
    // Captured here, not re-read at the end: this is the epoch at which these
    // exact pixels became the mapping's content. Anything below — an eviction
    // flush landing a sibling window, which writes guest pages and replaces the
    // one-per-mapping cache entry — advances the epoch past it, and the
    // resident stamped with the captured value then reads as stale. That is the
    // intended direction: a stale stamp costs a CPU seed, a fresh-looking one
    // costs a wrong frame.
    let published_epoch = state.note_surface_content_published(mapping_id);
    evict_render_windows_to_cap(state, host);
    Ok(finish_surface_deferred_window(
        state,
        req,
        key,
        crate::model::RenderWindowSource::Owned(frame),
        published_epoch,
    ))
}

/// Arm the deferred window for a type-11 render Store that **skipped its
/// readback**: the pinned engine resident holds the frame, and no CPU copy of it
/// exists anywhere.
///
/// This is the rail that removes the cost rather than rescheduling it.
/// `surface_flush / surface_deferred` measures 0.138, so ~86 % of these windows
/// are never flushed at all — and [`arm_surface_deferred_store_with`] still pays
/// a whole-framebuffer GPU→host readback per Store to own bytes that nobody
/// reads. `draw_phase` prices that at `wait_us` + `readback_us` = 565 ms per
/// second of wall clock, 68 % of `draw_us`, and `skip_readback` returns from
/// `execute_draw_inner` before `Phase::Wait`, so both fall together.
///
/// Fail-closed at every gate, and the caller must treat `None` as "take the
/// synchronous route" rather than as a loss:
///
/// - **The pin.** [`crate::backend::vulkan::engine::pin_resident_target`] refuses
///   an absent or not-content_ready slot, and without the pin the LRU sweep or the
///   idle drain could reclaim the only copy of the frame.
/// - **The epoch.** `note_surface_content_published` returns 0 for a mapping that
///   is gone, and 0 is the value that means "nothing published since attach" — a
///   window carrying it would compare equal to an unstamped resident, which is the
///   `None == None` hazard [`type11_resident_is_current`] exists to refuse.
/// - **The cession.** The host cache must stop answering for this geometry before
///   the window is armed, or the present capture keeps serving the *previous*
///   frame from bytes this Store superseded.
///
/// The identity is [`type11_store_identity`] — the same call that produced the
/// draw's `target_identity` — so the slot pinned and stamped is the slot the draw
/// rendered into. Deriving it a second way here is how a pin ends up protecting
/// an image the frame is not in.
fn arm_surface_resident_store<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    req: &DrawEncodeRequest,
    mapping_id: u32,
    width: u32,
    height: u32,
) -> Option<u32> {
    // Before any early return below: the union belongs to the draws that just
    // ran whether or not this arm goes on to succeed, and leaving it un-reset on
    // a declined arm would fold this pass into the next window's reading.
    note_pass_scissor_union(width, height);
    let identity = type11_store_identity(state, req, true)?;
    let key = prepare_surface_deferred_window(state, host, req, mapping_id, width, height)?;
    // The slot this arm pins and the slot the flush will look up have to be the
    // same slot. Geometry no longer separates them: both spellings read color0's
    // declared extent, because that is the only extent a draw request carries.
    // It used to carry a second one, and a record whose pass extent differed
    // from its attachment pinned one identity and handed the window another —
    // the flush found no slot, the frame was lost, and the pin leaked for the
    // guest's lifetime because eviction skips pinned slots.
    //
    // `map_generation` is the axis still live here, and it can move *inside* the
    // arm: `prepare_surface_deferred_window` below lands intersecting windows,
    // and a writeback re-resolves the mapping. Taking the identity before that
    // step and the key after it is what this compares.
    //
    // That is the defect shape of `74748d2` and `021e64b` a third time, so it is
    // closed the same way: one spelling, checked, never two reconciled.
    // Declining is free here — the caller treats `None` as "take the synchronous
    // route", which pays a readback and loses nothing.
    if crate::runtime::storage_flush::render_window_identity(&key) != identity {
        crate::observe::Emit::decline(
            "surface_resident_arm",
            &SurfaceResidentArmDecline::IdentitySplit,
        )
        .field("mid", mapping_id)
        .field("key_geom", format!("{}x{}", key.width, key.height))
        .field("pass_geom", format!("{width}x{height}"))
        .fail_once(u64::from(mapping_id));
        return None;
    }
    if !crate::backend::vulkan::engine::pin_resident_target(&identity) {
        crate::observe::Emit::decline(
            "surface_resident_arm",
            &SurfaceResidentArmDecline::PinRefused,
        )
        .field("mid", mapping_id)
        .field("geom", format!("{width}x{height}"))
        .fail_once(u64::from(mapping_id));
        return None;
    }
    // Ordered against `evict_render_windows_to_cap` deliberately. That call can
    // land a sibling window, and a landing writes guest pages through
    // `write_bgra8`, whose tail republishes this mapping's cache entry. Ceding
    // after it means the cession is the last word; ceding before would let a
    // sibling at this same geometry put its own plane's bytes back under the
    // present capture, which then serves them instead of the resident.
    let published_epoch = state.note_surface_content_published(mapping_id);
    evict_render_windows_to_cap(state, host);
    if published_epoch == 0
        || !crate::runtime::surface_cache::cede_surface_to_resident(
            state, mapping_id, width, height,
        )
    {
        crate::backend::vulkan::engine::unpin_resident_target(&identity);
        crate::observe::Emit::decline(
            "surface_resident_arm",
            &SurfaceResidentArmDecline::NoEpoch {
                epoch: published_epoch,
            },
        )
        .field("mid", mapping_id)
        .field("geom", format!("{width}x{height}"))
        .fail_once(u64::from(mapping_id));
        return None;
    }
    // The resident's stamp and the window's copy of the epoch are the two halves
    // of one witness, and both have to be written before any other draw can clear
    // the stamp. `stamp_resident_content_epoch` refuses a slot that is not
    // content_ready — which the pin above already established — so a false here
    // means the slot went away between the two calls under the engine lock.
    if !crate::backend::vulkan::engine::stamp_resident_content_epoch(&identity, published_epoch) {
        crate::backend::vulkan::engine::unpin_resident_target(&identity);
        crate::observe::Emit::decline(
            "surface_resident_arm",
            &SurfaceResidentArmDecline::StampRefused,
        )
        .field("mid", mapping_id)
        .field("geom", format!("{width}x{height}"))
        .fail_once(u64::from(mapping_id));
        return None;
    }
    // The guest half of the same witness, written here rather than by the
    // caller because this rail returns straight out of `encode_draw` — it never
    // reaches `stamp_type11_resident`, and it is where nearly all type-11
    // Stores go.
    crate::runtime::mapper::stamp_guest_write_gen(state, host, mapping_id);
    // Paired with `note_resident_window_flushed` at the readback. Stamped here
    // rather than in `finish_surface_deferred_window`, which also serves the
    // `Owned` route, whose frame is already in host memory and owes no round
    // trip to time.
    crate::runtime::drain::note_resident_window_armed();
    Some(finish_surface_deferred_window(
        state,
        req,
        key,
        crate::model::RenderWindowSource::Resident {
            epoch: published_epoch,
        },
        published_epoch,
    ))
}

/// Why a type-11 Store that skipped its readback could not arm a
/// resident-backed window, and therefore had to materialize its frame and take
/// the synchronous route.
///
/// Typed rather than one slug for three checks: the three have different fixes —
/// a refused pin is a registry-state question, a zero epoch is a mapping that
/// went away mid-draw, a refused stamp is a slot evicted between two engine-lock
/// acquisitions — and a single `reason=arm_failed` would name none of them. Every
/// arm is recoverable and none of them loses guest work, so these are declines and
/// not losses.
enum SurfaceResidentArmDecline {
    PinRefused,
    NoEpoch {
        epoch: u32,
    },
    StampRefused,
    /// The slot this arm would pin is not the slot the flush would look up —
    /// the pass extent and the attachment geometry disagree. Unlike the other
    /// three this is not a state question about the registry; it is the window
    /// and the resident being named differently, and arming through it loses the
    /// frame and leaks the pin.
    IdentitySplit,
}

impl crate::observe::Decline for SurfaceResidentArmDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::PinRefused => "surface_resident_pin_refused",
            Self::NoEpoch { .. } => "surface_resident_no_epoch",
            Self::StampRefused => "surface_resident_stamp_refused",
            Self::IdentitySplit => "surface_resident_identity_split",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::PinRefused | Self::StampRefused | Self::IdentitySplit => Vec::new(),
            Self::NoEpoch { epoch } => vec![("epoch", epoch.to_string())],
        }
    }
}

/// The part of arming a type-11 render window that does not depend on where the
/// frame lives: the protocol-shape gate, the supersede rule, and landing the
/// sibling windows that cover guest bytes this Store does not write.
///
/// Shared by both rails on purpose. These three steps are what make the window
/// *findable and exclusive* — the guest byte range every reader intersects
/// against, with nothing stale left inside it — and a rail that reimplemented one
/// of them slightly differently is how one kind of window ends up covered by a
/// trigger the other kind misses.
fn prepare_surface_deferred_window<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    req: &DrawEncodeRequest,
    mapping_id: u32,
    width: u32,
    height: u32,
) -> Option<crate::model::ComputeStorageResidencyKey> {
    let key = surface_store_defer_eligible(state, req)?;
    if key.mapping_id != mapping_id || key.width != width || key.height != height {
        return None;
    }
    // Supersede — do not flush — the window this Store fully covers. The rule and
    // the reason it is sound live with the release, in
    // `storage_flush::supersede_covered_render_windows`, because taking a window
    // and dropping its hold are one act and this site used to do only the first.
    for (old, unpinned) in
        crate::runtime::storage_flush::supersede_covered_render_windows(state, &key)
    {
        crate::observe::line(format!(
            "surface_deferred_superseded mapping={} {}x{} fmt={:#x} unpinned={}",
            old.mapping_id,
            old.width,
            old.height,
            old.pixel_format,
            unpinned.is_some() as u8
        ));
    }
    // Whatever still intersects covers guest bytes this Store does *not* write —
    // a different plane window on the same mapping — so it has to land.
    //
    // Each of those windows names its own frame, so unlike the first cut of this
    // rail the order against the cache refresh in the caller no longer matters.
    if !crate::runtime::storage_flush::flush_intersecting(
        state,
        host,
        key.mapping_id,
        key.surface_offset,
        key.span_end,
    ) {
        // A window that would not land is a window whose guest bytes are now
        // unknown; arming over it would attribute its loss to this Store.
        return None;
    }
    Some(key)
}

/// Insert the prepared window and index it for the readers that must flush it.
///
/// Returns the epoch, so both rails report the same thing to
/// [`stamp_type11_resident`] on the caller's side.
///
/// Takes no `host` and runs no eviction: the population cap has to be pressed
/// *before* the caller's last write to shared state, because landing a window
/// writes guest pages and republishes this mapping's one cache entry. Each rail
/// therefore calls [`evict_render_windows_to_cap`] where its own last write is,
/// and the two orders differ — which is exactly why this function cannot own it.
fn finish_surface_deferred_window(
    state: &mut DeviceState,
    req: &DrawEncodeRequest,
    key: crate::model::ComputeStorageResidencyKey,
    source: crate::model::RenderWindowSource,
    published_epoch: u32,
) -> u32 {
    state.surface_deferred_seq = state.surface_deferred_seq.wrapping_add(1);
    let armed_seq = state.surface_deferred_seq;
    let resident = matches!(source, crate::model::RenderWindowSource::Resident { .. });
    state.compute_deferred_flush.insert(
        key,
        crate::model::DeferredOwner::Render {
            armed_seq,
            armed_stamp_seq: state.completion_stamp_seq,
            source,
        },
    );
    // Raw task-GVA reads that alias these physical pages flush through
    // `flush_intersecting_task_gva`, which finds the mapping via this index.
    state.index_deferred_alias_pages(key.mapping_id);
    crate::observe::line(format!(
        "surface_writeback_deferred mapping={} {}x{} fmt={:#x} pipe={} windows={} resident={}",
        key.mapping_id,
        key.width,
        key.height,
        key.pixel_format,
        req.pipeline_ref,
        state.compute_deferred_flush.len(),
        resident as u8,
    ));
    published_epoch
}

/// Live [`crate::model::DeferredOwner::Render`] windows, for the population cap.
fn render_window_count(state: &DeviceState) -> usize {
    state
        .compute_deferred_flush
        .values()
        .filter(|o| matches!(o, crate::model::DeferredOwner::Render { .. }))
        .count()
}

/// Land render windows oldest-first until the population is back under
/// [`SURFACE_DEFERRED_WINDOW_CAP`].
///
/// Through the normal choke point rather than taking entries directly:
/// `flush_intersecting` runs the fixpoint that drags in siblings overlapping the
/// same guest bytes, and taking one window out from under that would leave those
/// siblings holding stale ranges.
///
/// A window can legitimately survive its flush — a condemned backing holds its
/// obligation for `mapper::resolve` to settle — so this steps over it and tries
/// the next oldest. Stopping there would wedge the cap behind one stuck mapping
/// for every other mapping, and a window owns the frame it deferred, so the
/// leak would be a full framebuffer per stuck key.
///
/// The order is taken once and walked. Re-deriving "the oldest" after a refusal
/// returns the same stuck window forever, which is the bug this replaced.
fn evict_render_windows_to_cap<M: HostMemory + HostOps>(state: &mut DeviceState, host: &mut M) {
    for (mid, lo, hi) in render_windows_oldest_first(state) {
        let before = render_window_count(state);
        if before < SURFACE_DEFERRED_WINDOW_CAP {
            return;
        }
        crate::runtime::storage_flush::flush_intersecting(state, host, mid, lo, hi);
        // Forcing a window to land early is the cap taking work off the guest's
        // own schedule, so it says so — this is the surface rail's counterpart
        // to `gva_deferred_flush trigger=window_cap` and `compute_mirror_evicted`.
        // The fixpoint drags in siblings overlapping the same guest bytes, so
        // one pass can land more than the window it was aimed at; report what
        // actually left rather than one per pass.
        crate::observe::off(format!(
            "surface_window_cap_evicted mid={mid} off={lo} end={hi} live={before} \
             cap={SURFACE_DEFERRED_WINDOW_CAP} landed={}",
            before.saturating_sub(render_window_count(state)),
        ));
    }
}

/// Guest byte ranges of the live render windows, oldest first, for the cap's
/// eviction order. Compute windows are never chosen — they are bounded by their
/// own dispatches, and evicting one here would land content this cap was not
/// sized for.
///
/// The whole order rather than just the minimum, because a window can
/// legitimately refuse to land: a condemned backing holds its obligation for
/// `mapper::resolve` to settle, and one boot held one for 121 s. Stopping at the
/// oldest would wedge the cap behind it for *every other mapping*, and since a
/// window now owns the frame it deferred that is a full framebuffer per stuck
/// key — the "~260 stale residents pinned for the guest lifetime" shape. Step
/// over it instead.
fn render_windows_oldest_first(state: &DeviceState) -> Vec<(u32, u64, u64)> {
    let mut live: Vec<(u64, u32, u64, u64)> = state
        .compute_deferred_flush
        .iter()
        .filter_map(|(k, o)| match o {
            crate::model::DeferredOwner::Render { armed_seq, .. } => {
                Some((*armed_seq, k.mapping_id, k.surface_offset, k.span_end))
            }
            _ => None,
        })
        .collect();
    live.sort_unstable_by_key(|(seq, ..)| *seq);
    live.into_iter()
        .map(|(_, mid, lo, hi)| (mid, lo, hi))
        .collect()
}

/// Defer gate for the final/single record of a GVA render Store: the record
/// may keep its pixels on the engine registry resident and land guest bytes
/// on access (`storage_flush::flush_gva_one`) instead of a sync readback +
/// fence wait on the stamp path. All gates are protocol-shape checks (never
/// content): the flush must be able to replay the sync `write_gva_rgba8`
/// exactly — identity geometry == c0 geometry, convertible format, sane BPR.
fn gva_store_defer_eligible(req: &DrawEncodeRequest) -> bool {
    let Some(c0) = req.colors.first() else {
        return false;
    };
    if c0.mapping_id != 0 || c0.target_gva == 0 || c0.row_stride == 0 {
        return false;
    }
    let Some(identity) = gva_chain_identity(req) else {
        return false;
    };
    if identity.width() != c0.width || identity.height() != c0.height {
        return false;
    }
    pixel_format::tight_row_bytes(c0.width, c0.format).is_some_and(|t| c0.row_stride >= t)
}

/// Any host-side writer of the guest window at `gva` supersedes the deferred
/// Store window there: a later flush of the old window would clobber the
/// strictly-newer bytes. Same geometry drops the obligation (the new write
/// fully covers the window; its bytes were never observable without a flush,
/// which would have taken it); different geometry lands the old identity
/// first, preserving the sync serialization (old bytes, then new bytes).
pub(crate) fn supersede_gva_window<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    gva: u64,
    width: u32,
    height: u32,
    by: &str,
) {
    let Some(old) = state.gva_deferred_flush.get(&gva) else {
        return;
    };
    if old.width == width && old.height == height {
        // The identity the OLD window pinned, which is the only slot this unpin
        // may release: the caller's geometry says the two windows describe one
        // region, but the resident behind the old one was created under the
        // generation that window stored, not under whatever the address names
        // now.
        let old_identity = crate::runtime::storage_flush::gva_window_identity(gva, old);
        let _ = state.take_gva_deferred_window(gva);
        crate::backend::vulkan::engine::unpin_resident_target(&old_identity);
        crate::observe::line(format!(
            "gva_deferred_superseded gva={gva:#x} {width}x{height} by={by}"
        ));
    } else {
        crate::runtime::storage_flush::flush_gva_exact(state, host, gva, true, by);
    }
}

/// How often does a GVA render target's address change hands between arms?
///
/// The page list behind the span is the allocation's identity and needs no
/// heuristic to say so: identical pages means literally the same guest memory,
/// so sharing an image is correct; different pages means a different allocation
/// and sharing would be a wrong-content bug. `gva_alloc_generation` puts that
/// same hash in the resident's registry key, so the two now get separate images;
/// this counts the arms where that separation is what stands between them.
///
/// Deliberately kept independent of the key it scores. It reads its own
/// `gva_resident_backing` history rather than the identity, because a census
/// that consults the mechanism it is measuring cannot report the day the
/// mechanism stops working.
///
/// Free at this call site by construction — `arm_gva_deferred_store` has just
/// walked the span to build `pages`, so this adds a hash and a map probe rather
/// than a page walk.
///
/// # Measured: it happens, at 5.6 % of arms, and it is not the icon rate
///
/// One driven 14-round x86/Vulkan boot:
///
/// ```text
/// gvares_same_alloc 59 138   gvares_aliased 3 487   gvares_regeom 3 112
/// aliased geometries: 64x64 x227, 1938x42 x32, 675x52 x23, ...
/// ```
///
/// So the address reuse is real and common: about one GVA render-target arm in
/// eighteen lands at an address a *different* guest allocation held, and the
/// geometry that dominates is 64x64 — a folder icon exactly, the same geometry
/// the Finder icon class corrupts at. That boot ran the shared-image key, so
/// each of those 3 487 arms bound the previous allocation's image.
///
/// **It is nevertheless not sufficient for the visible defect.** That boot
/// scored 14 of 14 rounds CLEAN with 3 487 aliased arms in it. Binding another
/// allocation's image is a contract violation on its own terms and worth
/// removing, but no claim that removing it fixes the icon rate is supported by
/// this measurement.
///
/// The same boot also broke the gate it was run under. `b820520` had found that
/// driving the VM for 600 s before the icon harness produced 14 of 14 corrupt,
/// and offered that as a repro; this boot did exactly that and produced 14 of 14
/// clean. Pooled over five 14-round boots on this branch's fixed binary the
/// picture is 0, 0, 0, 14, 0 corrupt — **all-or-nothing per boot**, never
/// mixed. Whatever decides it latches once per boot and then holds for every
/// round, which is why single-boot round counts have been so misleading here and
/// why 8-of-14 and 1-of-14 boots were recorded on one binary earlier.
///
/// Scoring anything on this class therefore needs boots, not rounds. A change
/// that is measured on one boot has measured the latch, not the change.
fn note_gva_resident_aliasing(
    state: &mut DeviceState,
    gva: u64,
    width: u32,
    height: u32,
    pages: &std::collections::HashSet<u64>,
) {
    // The same hash `gva_alloc_generation` puts in the registry key, so the
    // census and the identity cannot disagree about what "same allocation"
    // means — and unconditional, because a census that the bisection knob could
    // silence would stop measuring exactly when the control arm needs it.
    let hash = gva_page_set_hash(pages);
    let prev = state
        .gva_resident_backing
        .insert(gva, (width, height, hash));
    let Some((pw, ph, phash)) = prev else {
        crate::runtime::drain::note_store_route("gvares_first");
        return;
    };
    if (pw, ph) != (width, height) {
        // A different geometry was always a different registry key, so the two
        // never shared an image and this is not the address-reuse case.
        crate::runtime::drain::note_store_route("gvares_regeom");
        return;
    }
    if phash == hash {
        crate::runtime::drain::note_store_route("gvares_same_alloc");
        return;
    }
    crate::runtime::drain::note_store_route("gvares_aliased");
    if crate::observe::first_sight("gva_resident_aliased", gva ^ ((width as u64) << 32)) {
        crate::observe::fail(format!(
            "gva_resident_aliased gva={gva:#x} {width}x{height} pages={} \
             (same address and geometry, different guest allocation)",
            pages.len()
        ));
    }
}

/// Arm the deferred-writeback window for a GVA render Store that just
/// executed into the registry resident (`M2vDrawSpan::ResidentGvaStore`).
///
/// Returns `false` when a gate fails (unwalkable span, pin refusal) — the
/// caller then lands the Store synchronously from a resident readback, and
/// the sync site's supersede handling covers any older window at this GVA.
fn arm_gva_deferred_store<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    req: &DrawEncodeRequest,
) -> bool {
    let Some(identity) = gva_chain_identity(req) else {
        return false;
    };
    let Some(c0) = req.colors.first() else {
        return false;
    };
    if !crate::backend::vulkan::engine::deferred_gpu_only_content_allowed() {
        return false;
    }
    let gva = c0.target_gva;
    let span = (c0.row_stride as u64).saturating_mul(c0.height as u64);
    // Defer-time physical page index: raw task-GVA reads aliasing these pages
    // flush first (`storage_flush::flush_intersecting_task_gva`). A span that
    // does not fully walk cannot be guarded — Store synchronously.
    let pages = gva_mem::task_gva_page_gpa_set(
        host,
        &state.tasks,
        req.task_id,
        gva,
        span,
        state.page_shift,
    );
    if (pages.len() as u64) < gva_mem::pages_spanned(gva, span, state.page_size()) {
        return false;
    }
    note_gva_resident_aliasing(state, gva, c0.width, c0.height, &pages);
    // Supersede any previous window at this GVA before pinning. Same geometry
    // drops the obligation; different geometry is a distinct identity whose
    // resident is intact, so it lands first.
    //
    // The unpin the helper performs releases the identity the OLD window
    // pinned, which is not always the one pinned below: a re-render of the same
    // allocation gives both the same generation and the unpin is undone by the
    // pin, while an address handed to a second allocation gives two identities
    // and the first one's slot correctly stops being held for a buffer that no
    // longer exists there.
    supersede_gva_window(state, host, gva, c0.width, c0.height, "rearm");
    if !crate::backend::vulkan::engine::pin_resident_target(&identity) {
        return false;
    }
    // Each forced landing names itself as `gva_deferred_flush trigger=window_cap`,
    // so the cap is fail-visible without a counter beside it.
    while state.gva_deferred_flush.len() >= GVA_DEFERRED_WINDOW_CAP {
        let Some((old_gva, old_entry)) = state.take_oldest_gva_deferred_window() else {
            break;
        };
        let _ = crate::runtime::storage_flush::flush_gva_one(
            state,
            host,
            old_gva,
            &old_entry,
            true,
            "window_cap",
        );
    }
    let producer_object_type = objects::lookup_list_entry(state, host, req.task_id, c0.texture_ref)
        .map(|entry| entry.object_type)
        .unwrap_or(0);
    // Stale encodes must not serve while the resident is authoritative —
    // host-path consumers flush first; anything else misses (fail-safe).
    crate::runtime::surface_cache::evict_gva(state, gva);
    if c0.texture_ref != 0 {
        crate::runtime::surface_cache::evict_texture(state, c0.texture_ref);
    }
    state.gva_deferred_seq = state.gva_deferred_seq.wrapping_add(1);
    let armed_seq = state.gva_deferred_seq;
    state.arm_gva_deferred_window(
        gva,
        crate::model::GvaDeferredEntry {
            task_id: req.task_id,
            texture_ref: c0.texture_ref,
            producer_object_type,
            width: c0.width,
            height: c0.height,
            row_stride: c0.row_stride,
            format: c0.format,
            armed_seq,
            armed_stamp_seq: state.completion_stamp_seq,
            pages,
            // The generation the pinned identity above was built with, not a
            // hash of the `pages` walked here. The two normally agree; when a
            // remap lands between the draw's walk and this one they do not, and
            // the value that finds the pinned slot is the one the draw used.
            alloc_gen: req.gva_alloc_gen,
        },
    );
    crate::observe::line(format!(
        "gva_writeback_deferred gva={gva:#x} {}x{} fmt={:#x} pipe={} windows={}",
        c0.width,
        c0.height,
        c0.format,
        req.pipeline_ref,
        state.gva_deferred_flush.len()
    ));
    true
}

#[cfg(all(test, feature = "backend-vulkan"))]
mod vulkan_split_tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_X86};
    use crate::runtime::host::FakeHost;

    /// The blank-with-host-entry loss must be reported as a subset of a
    /// population, not as a bare count.
    ///
    /// `lin_rung_blank_with_host_entry` has been read three ways in this file's
    /// history and each reading needed a denominator it did not have: how often
    /// this rung serves the guest's pages for a span the cache also holds. With
    /// only the numerator, "300 a boot" cannot be told apart from "300 of 300"
    /// or "300 of 300 000", and those are different defects. So a content-
    /// bearing serve over a cached span must still be counted, and a blank one
    /// must land in both counters — the subset relation is the property, not
    /// either number.
    #[test]
    fn a_serve_over_a_cached_span_is_counted_whether_or_not_it_came_back_blank() {
        use crate::runtime::drain::store_route_count;
        let mut state = DeviceState::new(DeviceId(0), PAGE_SHIFT_X86);
        let host = FakeHost::new();
        let (w, h) = (4u32, 4u32);
        let gva = 0x40_0000u64;
        crate::runtime::surface_cache::store_gva_owned(
            &mut state,
            gva,
            w,
            h,
            vec![0x7f; (w * h * 4) as usize],
            0,
            None,
        );

        let entries = store_route_count("lin_rung_host_entry");
        let blanks = store_route_count("lin_rung_blank_with_host_entry");

        // Content came back: the cache holds the span, so this is one more
        // fall-through to guest pages over a cached span, and no loss.
        let content = vec![1u8; (w * h * 4) as usize];
        note_guest_rung_blank(
            &state,
            &host,
            1,
            9,
            (gva, w, h),
            &content,
            TexelLayout::Rgba8,
        );
        assert_eq!(
            store_route_count("lin_rung_host_entry"),
            entries + 1,
            "a content-bearing serve over a cached span is part of the population"
        );
        assert_eq!(
            store_route_count("lin_rung_blank_with_host_entry"),
            blanks,
            "content came back, so nothing was lost"
        );

        // All zeroes over the same cached span: the loss, and still a member of
        // the population it is a subset of.
        let blank = vec![0u8; (w * h * 4) as usize];
        note_guest_rung_blank(&state, &host, 1, 9, (gva, w, h), &blank, TexelLayout::Rgba8);
        assert_eq!(store_route_count("lin_rung_host_entry"), entries + 2);
        assert_eq!(
            store_route_count("lin_rung_blank_with_host_entry"),
            blanks + 1
        );

        // No cache entry for the span: a blank serve here is the other class
        // ("we do not have the pixels at all") and must reach neither counter.
        note_guest_rung_blank(
            &state,
            &host,
            1,
            9,
            (gva + 0x10_0000, w, h),
            &blank,
            TexelLayout::Rgba8,
        );
        assert_eq!(store_route_count("lin_rung_host_entry"), entries + 2);
        assert_eq!(
            store_route_count("lin_rung_blank_with_host_entry"),
            blanks + 1
        );
    }

    /// The per-pass union must union, reset, and include full-coverage draws.
    ///
    /// Three properties, and each one is a way the instrument could read
    /// promising while being wrong. If it did not union it would report the last
    /// draw and make every pass look cheap. If it did not reset at the arm it
    /// would accumulate across the whole boot and saturate at 100 %. And if it
    /// skipped full-coverage draws - which take an early return in the per-draw
    /// census - it would measure only the passes that were already cheap, which
    /// is exactly the population whose answer does not matter.
    #[test]
    fn the_pass_scissor_union_unions_resets_and_counts_full_coverage_draws() {
        use crate::runtime::drain::store_route_count;
        // Drain any rect left by another test in this binary.
        note_pass_scissor_union(1000, 1000);

        // Two disjoint quarter-width strips: 20 % each, union 60 % because the
        // bounding box spans them. A "last draw wins" accumulator says 20 %.
        let n = store_route_count("pass_scissor_union_le99");
        note_draw_coverage(0, 0, 200, 1000, 1000, 1000, None, false, false);
        note_draw_coverage(400, 0, 200, 1000, 1000, 1000, None, false, false);
        note_pass_scissor_union(1000, 1000);
        assert_eq!(
            store_route_count("pass_scissor_union_le99"),
            n + 1,
            "the union of two 20 % strips 400px apart is 60 %, not 20 %"
        );

        // The reset landed: a fresh single 4 % draw reads as 4 %, not 64 %.
        let n = store_route_count("pass_scissor_union_le5");
        note_draw_coverage(0, 0, 200, 200, 1000, 1000, None, false, false);
        note_pass_scissor_union(1000, 1000);
        assert_eq!(
            store_route_count("pass_scissor_union_le5"),
            n + 1,
            "the arm must reset the union, or it saturates across passes"
        );

        // A full-coverage draw takes the per-draw early return but must still
        // reach the union - it is the case that makes a pass unbounded.
        let n = store_route_count("pass_scissor_union_full");
        note_draw_coverage(0, 0, 40, 40, 1000, 1000, None, false, false);
        note_draw_coverage(0, 0, 1000, 1000, 1000, 1000, None, false, false);
        note_pass_scissor_union(1000, 1000);
        assert_eq!(
            store_route_count("pass_scissor_union_full"),
            n + 1,
            "a pass containing a full-coverage draw has a union of 100 %"
        );

        // An arm with no draws since the last one reports nothing at all,
        // rather than reporting a stale or empty rect as a real reading.
        let before: u64 = PASS_UNION_SLUGS.iter().map(|s| store_route_count(s)).sum();
        note_pass_scissor_union(1000, 1000);
        let after: u64 = PASS_UNION_SLUGS.iter().map(|s| store_route_count(s)).sum();
        assert_eq!(before, after, "an empty union is not a reading");
    }

    /// The scissor-area buckets must score the fraction, not the extent.
    ///
    /// The whole point of the census is comparing a draw against the surface it
    /// draws into, so the same rect on a small target and a large one belong in
    /// different buckets. A version that bucketed raw pixel area would answer a
    /// question nobody asked and read plausibly while doing it.
    #[test]
    fn scissor_area_buckets_score_the_fraction_of_the_target() {
        use crate::runtime::drain::store_route_count;
        // 1000x1000 target: the rect's area in pixels IS its percentage.
        let cases = [
            (10u32, 10u32, "draw_scissor_area_lt1"), // 0.01 %, rounds to 0
            (100, 100, "draw_scissor_area_le5"),     // exactly 1 %, so not the sub-1 bucket
            (200, 200, "draw_scissor_area_le5"),     // 4 %
            (300, 300, "draw_scissor_area_le10"),    // 9 %
            (500, 500, "draw_scissor_area_le25"),    // 25 %
            (700, 700, "draw_scissor_area_le50"),    // 49 %
            (800, 800, "draw_scissor_area_gt50"),    // 64 %
        ];
        for (sw, sh, slug) in cases {
            let before = store_route_count(slug);
            note_draw_coverage(0, 0, sw, sh, 1000, 1000, None, false, false);
            assert_eq!(
                store_route_count(slug),
                before + 1,
                "{sw}x{sh} of 1000x1000 belongs in {slug}"
            );
        }

        // The same rect against a target it fully covers is not a partial draw
        // at all, so it takes the `covers` early return and reaches no bucket.
        let before = store_route_count("draw_scissor_area_gt50");
        note_draw_coverage(0, 0, 800, 800, 800, 800, None, false, false);
        assert_eq!(
            store_route_count("draw_scissor_area_gt50"),
            before,
            "a full-coverage draw is counted by draw_scissor_full, not bucketed"
        );

        // A degenerate target must not divide by zero.
        note_draw_coverage(0, 0, 4, 4, 0, 0, None, false, false);
    }

    /// A blank guest read is only a loss if the cache holds pixels to lose.
    ///
    /// `lin_rung_blank_with_host_entry` tested that the cache *held* the span
    /// and reported every hit as a coherence loss. It could not distinguish a
    /// span the device cleared and cached blank — where the guest's pages
    /// agreeing with a blank cache is both rails telling the truth — from one
    /// where the cache holds content the guest alias failed to return. Only the
    /// second is a defect, so the two must land in different counters while both
    /// stay inside the population counter above.
    #[test]
    fn a_blank_serve_is_split_by_whether_the_cached_entry_holds_pixels() {
        use crate::runtime::drain::store_route_count;
        let mut state = DeviceState::new(DeviceId(0), PAGE_SHIFT_X86);
        let host = FakeHost::new();
        let (w, h) = (4u32, 4u32);
        let blank = vec![0u8; (w * h * 4) as usize];

        // A span the device cached with content, read back blank: the loss.
        let content_gva = 0x50_0000u64;
        crate::runtime::surface_cache::store_gva_owned(
            &mut state,
            content_gva,
            w,
            h,
            vec![0x7f; (w * h * 4) as usize],
            0,
            None,
        );
        // A span the device cached blank, read back blank: the two rails agree.
        let blank_gva = 0x60_0000u64;
        crate::runtime::surface_cache::store_gva_owned(
            &mut state,
            blank_gva,
            w,
            h,
            vec![0u8; (w * h * 4) as usize],
            0,
            None,
        );

        let agrees = store_route_count("lin_rung_blank_host_agrees");
        let lost = store_route_count("lin_rung_blank_host_content");
        let population = store_route_count("lin_rung_blank_with_host_entry");

        note_guest_rung_blank(
            &state,
            &host,
            1,
            9,
            (content_gva, w, h),
            &blank,
            TexelLayout::Rgba8,
        );
        assert_eq!(
            store_route_count("lin_rung_blank_host_content"),
            lost + 1,
            "the cache holds pixels this read did not return: a real loss"
        );
        assert_eq!(
            store_route_count("lin_rung_blank_host_agrees"),
            agrees,
            "a content-bearing cache entry is not agreement"
        );

        note_guest_rung_blank(
            &state,
            &host,
            1,
            9,
            (blank_gva, w, h),
            &blank,
            TexelLayout::Rgba8,
        );
        assert_eq!(
            store_route_count("lin_rung_blank_host_agrees"),
            agrees + 1,
            "a blank cache entry over blank guest pages loses nothing"
        );
        assert_eq!(
            store_route_count("lin_rung_blank_host_content"),
            lost + 1,
            "agreement must not be counted as loss"
        );

        // Both remain members of the population they subset.
        assert_eq!(
            store_route_count("lin_rung_blank_with_host_entry"),
            population + 2,
            "the split partitions the population rather than replacing it"
        );
    }

    /// A refused arm must hand the frame back intact, because the synchronous
    /// route is the next thing to run and those are the only pixels it has.
    ///
    /// The `Result<u32, Vec<u8>>` exists for this. With a `bool` the buffer had
    /// to be borrowed, which forced the success path to clone a whole frame —
    /// ~4.7 MB at ~200 Stores/s, measured at 152 ms/s of `t11_convert_us`. Moving
    /// it in makes the refusal responsible for giving it back, and an `Err` built
    /// with the wrong buffer (or an empty one) would not fail to compile: it
    /// would write a blank or truncated frame into the guest's pages on every
    /// refusal, which is the black-layer class.
    #[test]
    fn a_refused_deferred_arm_returns_the_frame_it_was_given() {
        let mut state = DeviceState::new(DeviceId(0), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let frame: Vec<u8> = (0..(8 * 4 * 4)).map(|i| (i % 251) as u8).collect();
        let original = frame.clone();

        // A bare request against empty state: no surface window is eligible, so
        // the arm refuses at its first gate, before touching the frame.
        let req = DrawEncodeRequest::default();
        let out = arm_surface_deferred_store_with(&mut state, &mut host, &req, 7, 8, 4, frame);

        match out {
            Ok(epoch) => panic!("expected a refusal from empty state, armed at epoch {epoch}"),
            Err(returned) => assert_eq!(
                returned, original,
                "the refusal must return the same pixels it was handed"
            ),
        }
    }

    /// Both sides absent must not read as agreement.
    ///
    /// `Option<u32> == Option<u32>` makes `None == None` true, so a bare
    /// equality here would let a mapping with no entry match an image that was
    /// never stamped, and the pass would `LoadFromTarget` out of undefined
    /// memory instead of seeding. That is the black-layer class, arrived at
    /// from a new direction: a whole compositing layer renders as a sharp
    /// axis-aligned rectangle of garbage or black.
    /// The registry key for a GVA resident is `(gva, width, height)`, so the
    /// only thing that can tell two allocations at one address apart is the
    /// guest memory behind them. The census has to separate three cases that
    /// look identical from the address alone, and must not call a re-render of
    /// the same buffer an aliased reuse — that would report the common case as
    /// the defect and make the counter useless.
    #[test]
    fn a_second_arm_is_aliased_only_when_the_guest_pages_changed() {
        use crate::runtime::drain::store_route_count;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let a: std::collections::HashSet<u64> = [0x1000, 0x2000, 0x3000].into_iter().collect();
        // Same pages, different traversal order: the set is the identity.
        let a_reordered: std::collections::HashSet<u64> =
            [0x3000, 0x1000, 0x2000].into_iter().collect();
        let b: std::collections::HashSet<u64> = [0x1000, 0x2000, 0x9000].into_iter().collect();

        let (f0, s0, x0) = (
            store_route_count("gvares_first"),
            store_route_count("gvares_same_alloc"),
            store_route_count("gvares_aliased"),
        );

        super::note_gva_resident_aliasing(&mut state, 0x8000, 64, 64, &a);
        assert_eq!(
            store_route_count("gvares_first"),
            f0 + 1,
            "nothing to compare against yet"
        );

        super::note_gva_resident_aliasing(&mut state, 0x8000, 64, 64, &a_reordered);
        assert_eq!(
            (store_route_count("gvares_same_alloc"), store_route_count("gvares_aliased")),
            (s0 + 1, x0),
            "the same buffer re-rendered shares its image correctly, and walk order is not a change"
        );

        super::note_gva_resident_aliasing(&mut state, 0x8000, 64, 64, &b);
        assert_eq!(
            store_route_count("gvares_aliased"),
            x0 + 1,
            "different guest pages at the same address and geometry is a different \
             allocation inheriting the previous one's image"
        );

        // A geometry change makes a different registry key, so the two residents
        // are distinct and nothing is inherited.
        super::note_gva_resident_aliasing(&mut state, 0x8000, 32, 32, &a);
        assert_eq!(
            store_route_count("gvares_aliased"),
            x0 + 1,
            "a different geometry is a different key and must not be scored as aliasing"
        );
    }

    /// One guest page-table entry pointing GVA page 1 at `pfn`, on a task the
    /// GVA walker will accept. Returns the state the walk reads its task from;
    /// the caller re-points the entry by calling this again on the same host.
    fn map_one_gva_page(host: &mut FakeHost, pfn: u32) {
        use crate::contract::endian::st32;
        use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
        use crate::runtime::host::HostMemory;
        let page = 1u64 << PAGE_SHIFT_X86;
        // Directory at pfn 2, its root page table at pfn 3, data pages above.
        for gpa in [2 * page, 3 * page, 4 * page, 5 * page] {
            host.map_range(gpa, page as usize, 0);
        }
        let mut dir = [0u8; 8];
        st32(&mut dir[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut dir[DIRECTORY_DEPTH as usize..], 1);
        host.write_gpa(2 * page, &dir).unwrap();
        let mut pte = [0u8; 4];
        st32(&mut pte, pfn);
        // Entry index 1 (4 bytes wide) is GVA page 1, i.e. GVA 0x1000.
        host.write_gpa(3 * page + 4, &pte).unwrap();
    }

    /// A GVA render target whose color0 span is one guest page at GVA 0x1000.
    fn one_page_gva_request() -> DrawEncodeRequest {
        DrawEncodeRequest {
            task_id: 1,
            colors: vec![ColorRtRequest {
                slot: 0,
                texture_ref: 7,
                mapping_id: 0,
                target_gva: 0x1000,
                row_stride: 32,
                width: 8,
                height: 8,
                ..Default::default()
            }],
            ..DrawEncodeRequest::default()
        }
    }

    /// A GVA render target is named by the guest memory behind it, not by its
    /// address.
    ///
    /// The engine registry keys a resident on every field of
    /// `TargetIdentity::Gva`. With `generation` constant the key was
    /// `(gva, width, height)`, so a second guest allocation handed the same
    /// address at the same geometry got **the same GPU image** — and the
    /// cross-pass resident Load in `encode_draw_chain` then reads the first
    /// allocation's pixels as the second one's prior content. The guest recycles
    /// render-target addresses hard enough for that to be ~5.6 % of arms.
    ///
    /// Only the generation may separate them: the same address at the same
    /// geometry backed by the same page must still be one identity, or every
    /// re-render of a live buffer would mint a slot and lose its content.
    #[test]
    fn a_gva_targets_identity_follows_its_guest_pages_not_its_address() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        map_one_gva_page(&mut host, 4);
        assert!(state.define_task(1, 0x1_0000, 2));

        let mut req = one_page_gva_request();
        let gen_a = super::gva_alloc_generation(&state, &mut host, &req);
        assert_ne!(gen_a, 0, "a fully walked GVA span must name its allocation");
        req.gva_alloc_gen = gen_a;
        let id_a = super::gva_chain_identity(&req).expect("a GVA color0 has a chain identity");

        // The same buffer rendered again: same pages, so the same resident.
        assert_eq!(
            super::gva_alloc_generation(&state, &mut host, &req),
            gen_a,
            "an unchanged mapping must not mint a second identity"
        );

        // The guest frees the target and its allocator hands the address to a
        // different page. Same task, same GVA, same geometry.
        map_one_gva_page(&mut host, 5);
        let gen_b = super::gva_alloc_generation(&state, &mut host, &req);
        assert_ne!(
            gen_b, gen_a,
            "different guest pages are a different allocation"
        );
        req.gva_alloc_gen = gen_b;
        let id_b = super::gva_chain_identity(&req).expect("a GVA color0 has a chain identity");

        assert_ne!(
            id_a, id_b,
            "two allocations at one address must not share one image"
        );
        assert_eq!(
            (id_a.width(), id_a.height()),
            (id_b.width(), id_b.height()),
            "and the generation must be the only thing that separates them"
        );
    }

    /// A deferred window binds, flushes and unpins the resident it armed — even
    /// after the address stops resolving to the pages it was armed on.
    ///
    /// The window exists because the guest may reuse the address before the
    /// flush runs, so the identity has to be rebuilt from the window rather than
    /// from a fresh walk. A walk taken at flush time names whatever lives there
    /// now; the registry lookup then misses the slot the window is holding
    /// pinned, and the deferred frame is lost to a `deferred_flush_lost` instead
    /// of landing in guest memory.
    #[test]
    fn a_deferred_window_rebuilds_the_identity_it_armed_after_the_backing_moves() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        map_one_gva_page(&mut host, 4);
        assert!(state.define_task(1, 0x1_0000, 2));

        let mut req = one_page_gva_request();
        req.gva_alloc_gen = super::gva_alloc_generation(&state, &mut host, &req);
        let armed = super::gva_chain_identity(&req).expect("a GVA color0 has a chain identity");
        let window = crate::model::GvaDeferredEntry {
            task_id: req.task_id,
            texture_ref: 7,
            producer_object_type: 2,
            width: 8,
            height: 8,
            row_stride: 32,
            format: 0x46,
            armed_seq: 1,
            armed_stamp_seq: 0,
            pages: std::collections::HashSet::new(),
            alloc_gen: req.gva_alloc_gen,
        };

        map_one_gva_page(&mut host, 5);
        assert_ne!(
            super::gva_alloc_generation(&state, &mut host, &req),
            window.alloc_gen,
            "the fixture must actually move the backing, or this passes for the wrong reason"
        );
        assert_eq!(
            crate::runtime::storage_flush::gva_window_identity(0x1000, &window),
            armed,
            "the window must still name the resident it pinned"
        );
    }

    /// The set of pages is the identity; the order a walk reports them in is
    /// not, and one page of difference is a different allocation.
    ///
    /// `visit_task_gva_page_gpas` fills a `HashSet`, whose iteration order is
    /// unspecified and varies with insertion history, so an order-sensitive fold
    /// would mint a new registry slot for a buffer nothing had touched — every
    /// re-render would lose its resident content.
    #[test]
    fn the_page_set_names_the_allocation_and_its_traversal_order_does_not() {
        let a: std::collections::HashSet<u64> = [0x4000, 0x5000, 0x6000].into_iter().collect();
        let a_reordered: std::collections::HashSet<u64> =
            [0x6000, 0x4000, 0x5000].into_iter().collect();
        let b: std::collections::HashSet<u64> = [0x4000, 0x5000, 0x7000].into_iter().collect();

        let gen = |pages: &std::collections::HashSet<u64>| super::gva_page_set_hash(pages);
        let identity = |generation: u64| crate::backend::vulkan::engine::TargetIdentity::Gva {
            gva: 0x8000,
            width: 64,
            height: 64,
            generation,
        };

        assert_eq!(
            identity(gen(&a)),
            identity(gen(&a_reordered)),
            "the same pages in another order are the same allocation"
        );
        assert_ne!(
            identity(gen(&a)),
            identity(gen(&b)),
            "one page of difference at one address and geometry is a different allocation"
        );
    }

    #[test]
    fn an_unstamped_resident_never_matches_a_mapping_with_no_epoch() {
        assert!(!type11_resident_is_current(None, None));
        assert!(!type11_resident_is_current(None, Some(0)));
        assert!(!type11_resident_is_current(Some(7), None));
    }

    /// A sampled bind may only be served from a host-side copy of a type-4
    /// surface while the hypervisor has not watched the guest replace the pages
    /// that copy was taken from — and the GPU resident is bound by that rule
    /// exactly as the byte cache is.
    ///
    /// The two rungs were not equal. `t11rung_host_cache` asked
    /// `mapping_guest_write_verdict` before serving; `t11rung_resident`, which
    /// sits above it in the ladder and took 92 730 binds to the cache's 14 396
    /// on the boot that first measured them, asked nothing and returned
    /// `SampledSourceRequest::Target` unconditionally. A type-11 surface's pages
    /// are plain guest RAM: the guest CPU repaints them with no device
    /// operation, so a resident produced for one tenant of a pooled IOSurface
    /// keeps claiming to hold its pixels after a different tenant has been
    /// painted there. Nothing below the rung could correct it, because both
    /// rungs that read the guest's own pages sit underneath — which is why the
    /// wrong image was *held* rather than replaced on the next redraw.
    ///
    /// Both directions are asserted deliberately. Refusing more than `Wrote`
    /// would be just as wrong: `NoStamp` means this device never armed the
    /// witness, and turning the rung off on that answer would send binds to the
    /// guest's pages for surfaces whose content the deferred writeback rail has
    /// not landed there yet.
    #[test]
    fn a_watched_guest_write_refuses_every_host_side_copy_of_a_surface() {
        assert!(
            guest_wrote_allocation(GuestWriteVerdict::Wrote),
            "a resident whose pages the host watched the guest rewrite is not the surface"
        );
        for verdict in [
            GuestWriteVerdict::Clean,
            GuestWriteVerdict::NoMapping,
            GuestWriteVerdict::NoStamp,
            GuestWriteVerdict::Unreadable,
        ] {
            assert!(
                !guest_wrote_allocation(verdict),
                "{verdict:?} is not evidence of a guest write and must not refuse a copy"
            );
        }
    }

    /// The second stage, which is what keeps the first from being ruinous. A
    /// type-4 allocation is bigger than the plane a bind samples — pixels start
    /// at `base_off` and padding follows `span_end` — and the tracking token's
    /// generation moves for a write to any page of it. Refusing on that alone
    /// discarded whole 1920x1080 compositor scanouts the GPU had rendered and
    /// the guest had never touched the pixels of: measured live as a black
    /// desktop at 17 Hz.
    ///
    /// Fails closed in both unknown directions, because the caller cannot
    /// distinguish "no answer" from "written" without being wrong on frames.
    #[test]
    fn a_guest_write_outside_the_sampled_window_keeps_the_resident() {
        // A 1920x1080 BGRA8 plane one page into its allocation.
        const BASE: u64 = 4096;
        const END: u64 = BASE + 1920 * 1080 * 4;
        // The header page before the plane is not the pixels.
        assert!(!ranges_touch_window(&[(0, 4096)], BASE, END));
        // Nor is padding after it.
        assert!(!ranges_touch_window(&[(END + 4096, END + 8192)], BASE, END));
        // Abutting the end exactly is still outside — both bounds half-open.
        assert!(!ranges_touch_window(&[(END, END + 4096)], BASE, END));
        // One page anywhere inside the plane is the whole finding.
        assert!(ranges_touch_window(&[(4_198_400, 4_202_496)], BASE, END));
        // A range straddling the plane's first byte counts.
        assert!(ranges_touch_window(&[(0, 8192)], BASE, END));
        // Outside ranges do not mask an inside one.
        assert!(ranges_touch_window(
            &[(0, 4096), (4_198_400, 4_202_496), (END, END + 4096)],
            BASE,
            END
        ));
        assert!(!ranges_touch_window(&[], BASE, END));
    }

    /// Every rung of the sampled ladder that serves a host-side copy reports the
    /// verdict it was chosen under, so a boot can tell "the guest never rewrites
    /// its sampled surfaces" from "the witness was never armed". A rung with no
    /// column is a rung that asked no question, which is what the resident rung
    /// was.
    #[test]
    fn both_copy_serving_rungs_report_the_verdict_they_were_chosen_under() {
        for rung in ["t11rung_resident", "t11rung_host_cache"] {
            for (verdict, suffix) in [
                (GuestWriteVerdict::Clean, "clean"),
                (GuestWriteVerdict::NoMapping, "no_mapping"),
                (GuestWriteVerdict::NoStamp, "no_stamp"),
                (GuestWriteVerdict::Unreadable, "unreadable"),
            ] {
                assert_eq!(
                    sample_rung_gw_route(rung, verdict),
                    Some(format!("{rung}_gw_{suffix}").as_str()),
                    "sampled ladder lost the {rung} {suffix} column"
                );
            }
            assert_eq!(
                sample_rung_gw_route(rung, GuestWriteVerdict::Wrote),
                Some(format!("{rung}_gw_wrote_elsewhere").as_str()),
                "a bind served under Wrote is one whose write missed the sampled window"
            );
        }
        // A rung that reads the guest's own pages is not a copy and takes no
        // verdict column.
        assert_eq!(
            sample_rung_gw_route("t11rung_zero_copy", GuestWriteVerdict::NoStamp),
            None
        );
    }

    /// Epoch 0 is a legal mapping value — "nothing published since attach" —
    /// and must not be matchable by a slot's unstamped default. It is only
    /// current against a resident explicitly stamped with 0.
    #[test]
    fn epoch_zero_is_current_only_against_an_explicit_stamp() {
        assert!(type11_resident_is_current(Some(0), Some(0)));
        assert!(!type11_resident_is_current(Some(0), None));
    }

    /// The elision is exact equality, not "at least as new". A resident stamped
    /// at an older epoch has been overtaken by some writer — a blit, a compute
    /// writeback, a guest CPU write, a sibling geometry's publish — and must
    /// fall back to the CPU seed.
    #[test]
    fn any_epoch_movement_since_the_stamp_refuses_the_elision() {
        assert!(type11_resident_is_current(Some(4), Some(4)));
        assert!(!type11_resident_is_current(Some(5), Some(4)));
        assert!(!type11_resident_is_current(Some(4), Some(5)));
    }

    /// Every guest-page writer in this crate goes through
    /// `mark_mapping_written`, so making that advance the surface epoch is what
    /// closes the writer set without enumerating it. A blit or a guest CPU
    /// write invalidates a stamp without knowing this rail exists.
    #[test]
    fn a_guest_page_write_advances_the_surface_epoch() {
        let mut state = DeviceState::new(DeviceId(0), PAGE_SHIFT_X86);
        state.set_mapping_geom(7, 8, 4, 0x1e);
        let before = state.mappings.get(&7).unwrap().surface_content_epoch;

        let published = state.mark_mapping_written(7);
        let after = state.mappings.get(&7).unwrap().surface_content_epoch;

        assert!(published > 0, "content_generation still advances");
        assert_ne!(
            before, after,
            "a guest-page write must invalidate any resident stamp"
        );
    }

    /// The deferred type-11 publish writes only the host shadow — no guest
    /// page, so `mark_mapping_written` never runs — and it is the one writer
    /// that would otherwise change the mapping's pixels invisibly to the epoch.
    /// `surface_cache` holds one entry per mapping, so a sibling Store at
    /// another geometry replaces the entry an older geometry is compared
    /// against; without this bump that sibling is silent.
    #[test]
    fn a_deferred_publish_advances_the_epoch_without_a_guest_write() {
        let mut state = DeviceState::new(DeviceId(0), PAGE_SHIFT_X86);
        state.set_mapping_geom(7, 8, 4, 0x1e);
        let gen_before = state.mappings.get(&7).unwrap().content_generation;

        let first = state.note_surface_content_published(7);
        let second = state.note_surface_content_published(7);

        assert_ne!(first, second, "each publish is a distinct epoch");
        assert_eq!(
            gen_before,
            state.mappings.get(&7).unwrap().content_generation,
            "a deferred publish touched no guest page, so content_generation \
             must not move — the compute rail reads it and would re-seed"
        );
    }

    /// Re-attaching a mapping resets the epoch to 0, and 0 is unstampable-by-
    /// default on the resident side, so no resident carried over from the
    /// previous incarnation can vouch for the new one's pixels.
    #[test]
    fn reattaching_a_mapping_resets_the_surface_epoch() {
        let mut state = DeviceState::new(DeviceId(0), PAGE_SHIFT_X86);
        state.set_mapping_geom(7, 8, 4, 0x1e);
        state.mark_mapping_written(7);
        assert_ne!(state.mappings.get(&7).unwrap().surface_content_epoch, 0);

        // A geometry change is a new surface identity; the same reset guards
        // the MAP/UNMAP/reattach paths beside it.
        state.set_mapping_geom(7, 16, 8, 0x1e);
        assert_eq!(state.mappings.get(&7).unwrap().surface_content_epoch, 0);
    }

    /// A record carrying an explicit RT-provenance seed is not a candidate:
    /// that seed was selected for a reason the resident cannot know about, and
    /// the gate must not silently outvote it.
    #[test]
    fn an_explicitly_seeded_load_is_not_an_elision_candidate() {
        let mut c0 = ColorRtRequest {
            load_action: PASS_LOAD_ACTION_LOAD,
            ..Default::default()
        };
        assert!(type11_load_is_a_seed_candidate(&c0));

        c0.target_seed_rgba = Some(vec![0u8; 4]);
        assert!(!type11_load_is_a_seed_candidate(&c0));
    }

    /// A CLEAR is not a LOAD. Eliding a seed there would replace the guest's
    /// explicit clear with whatever the resident happened to hold.
    #[test]
    fn a_clear_is_not_an_elision_candidate() {
        let c0 = ColorRtRequest {
            load_action: PASS_LOAD_ACTION_CLEAR,
            ..Default::default()
        };
        assert!(!type11_load_is_a_seed_candidate(&c0));
    }

    /// A type-11 `LOAD` whose host cache misses seeds from the surface's own
    /// guest pages, and only refuses when those cannot serve the extent.
    ///
    /// Without the guest-pages rung this returns `None`, `target_rgba8` stays
    /// unset, and `exec.rs` resolves the pass load action to `Clear` against the
    /// hardcoded `[0,0,0,0]` — so the guest's request to preserve its surface
    /// became a transparent-black wipe that the matching Store published. One
    /// x86/Vulkan boot measured 121 distinct (mapping, geometry) instances of that
    /// in ~170 s, four at the full 1920x1080 composite extent, with the host
    /// window 62-90 % near-black during a desktop drag against 0.001 % at idle.
    ///
    /// Every one of those 121 lines had `want == mapgeom` and `hostgen=0`: the
    /// cache had never held the surface and its pages were readable. That pair is
    /// what makes reading them the fix rather than a guess.
    #[test]
    fn a_type11_load_seed_falls_back_to_the_surfaces_own_guest_pages() {
        use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
        use crate::runtime::mapping_write::write_bgra8;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let mid = 911u32;
        let pfn = 0x21u32;
        let gpa = (pfn as u64) << PAGE_SHIFT_X86;
        host.map_range(gpa, 0x4000, 0);
        state.map_surface(mid);
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.mapping_internal = 1;
            m.page_entries = vec![(pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        }
        let (w, h) = (4u32, 2u32);
        assert!(state.set_mapping_geom(mid, w, h, MTL_FORMAT_BGRA8_UNORM));

        // Guest-side content the compositor expects a LOAD to preserve. BGRA on
        // the wire; distinct per channel so a swizzle error cannot pass.
        let mut pages = vec![0u8; (w * h * 4) as usize];
        for px in pages.chunks_exact_mut(4) {
            px.copy_from_slice(&[0x10, 0x20, 0x30, 0xFF]);
        }
        assert!(write_bgra8(&mut state, &mut host, mid, &pages, w * 4, w, h));
        // `write_bgra8` mirrors what it wrote into the host cache, so drop that
        // mirror: the case under test is a surface whose pages hold content while
        // the cache holds nothing, which is what every one of the 121 measured
        // lines was (`hostgen=0`) — a first-ever LOAD, or a mapping whose remap
        // made `unmap_surface` evict the entry.
        crate::runtime::surface_cache::forget(&mut state, mid);
        assert!(
            crate::runtime::surface_cache::get(&state, mid, w, h).is_none(),
            "the cache must be cold: this test is about the miss path"
        );

        // Capture the always-on lines so a failure here names the check that
        // refused rather than showing a bare `None`: every rung on this ladder
        // declines by name, and the panic message is where that is worth reading.
        let cap = crate::observe::sink::FailCapture::start();
        let served = resolve_type11_load_seed(&mut state, &mut host, mid, w, h);
        let (bytes, order) = served.unwrap_or_else(|| {
            panic!(
                "a cold cache must not lose the guest's LOAD; sink said {:?}",
                cap.lines()
            )
        });
        drop(cap);
        assert_eq!(
            order,
            crate::backend::vulkan::engine::SeedOrder::Rgba8,
            "the guest-pages reader converts to RGBA8; mislabelling it swaps R and B"
        );
        assert_eq!(bytes.len(), (w * h * 4) as usize);
        assert_eq!(
            &bytes[..4],
            &[0x30, 0x20, 0x10, 0xFF],
            "BGRA guest bytes must arrive as semantic RGBA"
        );

        // A live cache entry still wins: it is the fresher copy (the last Store's
        // output) and the fallback must stay a fallback.
        let mut cached = vec![0u8; (w * h * 4) as usize];
        for px in cached.chunks_exact_mut(4) {
            px.copy_from_slice(&[0xAA, 0xBB, 0xCC, 0xFF]);
        }
        crate::runtime::surface_cache::store(&mut state, mid, w, h, cached);
        let (bytes, order) = resolve_type11_load_seed(&mut state, &mut host, mid, w, h)
            .expect("a warm cache must serve");
        assert_eq!(order, crate::backend::vulkan::engine::SeedOrder::Bgra8);
        assert_eq!(&bytes[..4], &[0xAA, 0xBB, 0xCC, 0xFF]);

        // An extent the surface is not latched at cannot be served by either rung,
        // and refusing is right: a seed of the wrong length is rejected by the
        // engine anyway, and the decline names both geometries.
        assert!(
            resolve_type11_load_seed(&mut state, &mut host, mid, w, h + 1).is_none(),
            "a mismatched extent must refuse by name, not seed something else"
        );
    }

    /// The type-4 host-cache sample rung must not serve a surface the
    /// hypervisor has watched the guest rewrite.
    ///
    /// That rung sits above both rungs that read the guest's own pages
    /// (`t11rung_zero_copy`, `t11rung_guest_memo`), so a stale hit is never
    /// corrected by anything below it — and its first census said `gw_clean` was
    /// **zero** across 14 396 binds. Only the demonstrated write refuses:
    /// `no_stamp` is a statement about this device's arming, not the guest's
    /// behaviour, and refusing on it would re-read a surface per bind.
    #[test]
    fn the_host_cache_sample_rung_refuses_a_surface_the_guest_rewrote() {
        use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        use crate::runtime::host::{FakeHost, HostOps};

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let page = state.page_size();
        // Chosen not to collide with any other test's mapping id: `first_sight`
        // latches per (reason, discriminant) for the life of the process.
        let mid = 911u32;
        assert!(state.map_surface(mid), "mid must be inside MAX_MAPPINGS");
        let gpa = 0x55 * page;
        {
            let m = state.mappings.get_mut(&mid).expect("mapped above");
            m.mapped = true;
            m.page_entries =
                vec![(((gpa >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        }

        // Unarmed: the verdict cannot vouch, but "nobody asked the host to
        // watch" is not "the guest wrote", so the rung still serves.
        assert_eq!(
            mapping_guest_write_verdict(&state, &host, mid),
            GuestWriteVerdict::NoStamp
        );
        assert!(
            !matches!(
                mapping_guest_write_verdict(&state, &host, mid),
                GuestWriteVerdict::Wrote
            ),
            "an unarmed rail must not cost every bind a re-read"
        );

        // Armed and stamped by a Store: the copy and the pages agree.
        let token = crate::runtime::mapper::ensure_guest_write_token(&mut state, &mut host, mid)
            .expect("FakeHost observes guest writes");
        state
            .mappings
            .get_mut(&mid)
            .expect("mapped above")
            .guest_write_gen_at_store = host.guest_write_gen(token).expect("a live token has one");
        assert_eq!(
            mapping_guest_write_verdict(&state, &host, mid),
            GuestWriteVerdict::Clean,
            "a surface nobody wrote since the Store may be served from the copy"
        );

        // The guest CPU rewrites the surface. No device operation, so nothing
        // else in this crate moves — this is the only witness that can see it.
        host.guest_wrote_page(gpa);
        assert_eq!(
            mapping_guest_write_verdict(&state, &host, mid),
            GuestWriteVerdict::Wrote,
            "the host saw the write, so the host-side copy is a stale picture"
        );
    }

    /// The type-11 `LOAD` seed branch reports both ways, and the miss arm names
    /// the geometry the cache actually holds.
    ///
    /// The miss is a whole-layer loss, not a degradation: with no seed the engine
    /// resolves the pass load action to `Clear` against the hardcoded
    /// `[0,0,0,0]`, so the guest's request to preserve its surface becomes a
    /// transparent-black wipe that the matching Store publishes. It reported
    /// nothing at all before this.
    ///
    /// The hit arm is asserted too, because a zero on the miss arm has to be
    /// readable: without a hit line beside it, "the cache always hit" and "this
    /// branch never ran" produce the same empty grep.
    ///
    /// Mapping ids here are chosen not to collide with any other test's, because
    /// `first_sight` latches per `(reason, discriminant)` for the life of the
    /// process and never resets.
    #[test]
    fn the_type11_load_seed_branch_reports_both_ways() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mid = 909u32;
        state.map_surface(mid);
        {
            let m = state.mappings.get_mut(&mid).unwrap();
            m.mapped = true;
            m.has_geom = true;
            m.width = 8;
            m.height = 4;
            m.map_generation = 3;
        }
        crate::runtime::surface_cache::store(&mut state, mid, 8, 4, vec![0u8; 8 * 4 * 4]);

        // The captured lines carry the sink's `OFF `/`FAIL ` severity prefix, so
        // match on the event token rather than on the first word.
        let only = |cap: &crate::observe::sink::FailCapture| -> String {
            let hits: Vec<String> = cap
                .lines()
                .into_iter()
                .filter(|l| l.contains("type11_load_seed"))
                .collect();
            assert_eq!(hits.len(), 1, "expected exactly one line, got {hits:?}");
            hits.into_iter().next().unwrap_or_default()
        };

        let cap = crate::observe::sink::FailCapture::start();
        note_type11_load_seed(&state, mid, 8, 4, Some(Type11SeedRung::Cache));
        let hit = only(&cap);
        assert!(hit.contains("outcome=cache_hit"), "{hit}");
        assert!(hit.contains("mapgeom=8x4"), "{hit}");
        assert!(hit.contains("mapgen=3"), "{hit}");
        drop(cap);

        // The recovered arm is its own outcome, not folded into `cache_hit`: its
        // rate is the only thing that prices the guest-pages fallback, and fusing
        // it would make the fix unmeasurable the moment it worked.
        let cap = crate::observe::sink::FailCapture::start();
        note_type11_load_seed(&state, mid, 4, 4, Some(Type11SeedRung::GuestPages));
        let pages = only(&cap);
        assert!(pages.contains("outcome=guest_pages"), "{pages}");
        drop(cap);

        // Same mapping, a geometry the cache does not hold: the entry's own
        // geometry is the load-bearing field, since it says a Store at another
        // extent orphaned every window still living at this one.
        let cap = crate::observe::sink::FailCapture::start();
        note_type11_load_seed(&state, mid, 8, 1, None);
        let geom = only(&cap);
        assert!(geom.contains("reason=type11_seed_cache_geom"), "{geom}");
        assert!(geom.contains("have=8x4"), "{geom}");
        assert!(geom.contains("want=8x1"), "{geom}");
        drop(cap);

        // A mapping the cache has never held reports absence, not a geometry.
        let cap = crate::observe::sink::FailCapture::start();
        note_type11_load_seed(&state, 910, 8, 4, None);
        let absent = only(&cap);
        assert!(
            absent.contains("reason=type11_seed_cache_absent"),
            "{absent}"
        );
        assert!(!absent.contains("have="), "{absent}");
        drop(cap);

        // Latched per (mapping, geometry, outcome): a repeat of any of the three
        // above emits nothing, so the branch is safe to leave on forever.
        let cap = crate::observe::sink::FailCapture::start();
        note_type11_load_seed(&state, mid, 8, 4, Some(Type11SeedRung::Cache));
        note_type11_load_seed(&state, mid, 4, 4, Some(Type11SeedRung::GuestPages));
        note_type11_load_seed(&state, mid, 8, 1, None);
        note_type11_load_seed(&state, 910, 8, 4, None);
        assert!(
            cap.lines().is_empty(),
            "second sighting must be latched: {:?}",
            cap.lines()
        );
    }

    /// One window that refuses to land must not wedge the cap for every other
    /// mapping.
    ///
    /// A condemned backing holds its obligation for `mapper::resolve` to settle —
    /// one boot held one for 121 s across 13015 flush attempts — and the eviction
    /// loop used to re-derive "the oldest" each pass and stop when it did not
    /// shrink. That returns the same stuck window forever, so the population
    /// grows without bound past the cap. It was survivable while a window was
    /// just a key; now that a window owns the frame it deferred, it leaks a whole
    /// framebuffer per stuck key.
    #[test]
    fn a_stuck_oldest_window_does_not_wedge_the_cap_for_the_others() {
        use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();

        let arm = |state: &mut DeviceState, mid: u32, seq: u64| {
            let gpa = 0xA000_0000u64 + (mid as u64) * 0x10_0000;
            state.map_surface(mid);
            {
                let m = state.mappings.get_mut(&mid).unwrap();
                m.mapped = true;
                m.map_generation = 1;
                m.page_entries = vec![
                    (((gpa >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
                ];
            }
            state.compute_deferred_flush.insert(
                crate::model::ComputeStorageResidencyKey {
                    mapping_id: mid,
                    map_generation: 1,
                    surface_offset: 0,
                    surface_bpr: 64,
                    span_end: 256,
                    width: 4,
                    height: 4,
                    pixel_format: crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM,
                    texture_ref: 0,
                },
                crate::model::DeferredOwner::Render {
                    armed_seq: seq,
                    armed_stamp_seq: 0,
                    source: crate::model::RenderWindowSource::Owned(std::sync::Arc::new(
                        vec![0u8; 4 * 4 * 4],
                    )),
                },
            );
        };

        // The oldest window sits on a condemned backing, so its flush is held.
        arm(&mut state, 1, 1);
        assert!(state.condemn_surface_backing(1), "mapping 1 must condemn");
        assert!(state.mapping_backing_condemned(1));
        // Fill past the cap with windows that can land.
        for i in 0..SURFACE_DEFERRED_WINDOW_CAP {
            arm(&mut state, 2 + i as u32, 2 + i as u64);
        }
        let before = render_window_count(&state);
        assert!(before > SURFACE_DEFERRED_WINDOW_CAP);

        evict_render_windows_to_cap(&mut state, &mut host);

        assert!(
            render_window_count(&state) < before,
            "the stuck oldest must be stepped over, not stopped on"
        );
        assert!(
            state.mapping_backing_condemned(1),
            "and the held window's mapping is left for the resolve to settle"
        );
    }

    #[test]
    fn m2v_draw_runtime_failure_returns_a_typed_decline() {
        use crate::observe::Decline as _;

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let mut req = DrawEncodeRequest {
            pipeline_ref: 41,
            ..DrawEncodeRequest::default()
        };

        let err = match try_metal2vulkan_draw(&mut state, &mut host, &mut req, true) {
            Err(err) => err,
            Ok(_) => panic!("an empty state cannot resolve pipeline 41"),
        };
        assert_eq!(err.slug(), "draw_prepare_pipeline_missing");
        assert_eq!(
            linux_m2v_draw_failure(&err, &req).render(),
            "linux_m2v_draw reason=draw_prepare_pipeline_missing \
             task_id=0 pipeline_ref=41 pipe=41 task=0 geom=0x0 vtx=0 inst=0 \
             prim=0 first=0 idx=0 colors=[] vbuf=[] fbuf=[] vtex=[] ftex=[] \
             viewport=None scissor=None"
        );
    }

    /// The branch line is only worth leaving on forever if it is bounded, and
    /// the bound is the dedup: one line per distinct route per process, however
    /// many Stores take that route.
    ///
    /// The load-bearing assertion is the dedup, not the text — a per-Store line
    /// on this path is a flood (thousands per session under compositing) and
    /// would have to be removed again, which is how the tree ended up with no
    /// always-on record of this branch in the first place.
    #[test]
    fn the_store_route_line_is_one_per_route_per_process() {
        crate::observe::redirect_logs_for_tests();
        let path = crate::observe::fail_log_path();
        let mark = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0) as usize;

        // Routes distinct from every product route so this test cannot be
        // satisfied by a line some other case in this binary emitted.
        note_type11_store_route("test_route_a");
        note_type11_store_route("test_route_a");
        note_type11_store_route("test_route_a");
        note_type11_store_route("test_route_b");

        let whole = std::fs::read_to_string(path).expect("fail log");
        let appended = &whole[mark.min(whole.len())..];
        let count = |route: &str| {
            appended
                .lines()
                .filter(|l| l.contains(&format!("type11_store_route route={route}")))
                .count()
        };
        assert_eq!(count("test_route_a"), 1, "three calls, one line");
        assert_eq!(count("test_route_b"), 1, "a second route still reports");
    }

    #[test]
    fn sampled_image_shape_maps_one_dimensional_luts() {
        use crate::runtime::spirv_bind::SampledImageKind;

        // A color-transfer LUT reflects as `texture1d` / `texture1d_array`.
        // Before this mapping the sampled path declined the whole draw with
        // `draw_prepare_texture_dimension_unsupported`, so the color-managed
        // desktop composite stored nothing and presented unbacked.
        let d1 = sampled_image_shape(SampledImageKind::D1).expect("D1 is expressible");
        assert!(d1.one_dim && !d1.arrayed && !d1.volume && !d1.cube);
        assert_eq!(d1.layers, 1);

        let d1_array =
            sampled_image_shape(SampledImageKind::D1Array).expect("D1Array is expressible");
        assert!(d1_array.one_dim && d1_array.arrayed && !d1_array.volume && !d1_array.cube);
        assert_eq!(d1_array.layers, 1);
    }

    #[test]
    fn sampled_image_shape_keeps_two_dimensional_shapes_flat() {
        use crate::runtime::spirv_bind::SampledImageKind;

        for kind in [
            SampledImageKind::D2,
            SampledImageKind::D2Array,
            SampledImageKind::D3,
        ] {
            let shape = sampled_image_shape(kind).expect("2D/3D shapes stay expressible");
            assert!(!shape.one_dim, "{kind:?} must not be a 1D image");
        }
    }

    #[test]
    fn sampled_image_shape_declines_cube_shapes_by_name() {
        use crate::runtime::spirv_bind::SampledImageKind;

        // Cube sampling is not expressed on the sampled-draw path yet; the
        // shape stays `None` so the caller declines it visibly rather than
        // binding a 2D view under a cube sampler.
        assert!(sampled_image_shape(SampledImageKind::Cube).is_none());
        assert!(sampled_image_shape(SampledImageKind::CubeArray).is_none());
    }
    /// An MRT secondary attachment is named by ITS OWN guest pages.
    ///
    /// `74748d2` gave color0 a page-set generation and deliberately left the
    /// secondaries at 0, because `build_secondary_targets` had no `HostOps` to
    /// walk with. That left every secondary keyed on `(gva, width, height)`
    /// alone — the exact keying that hands a second allocation the first one's
    /// image.
    #[test]
    fn a_secondary_attachment_is_named_by_its_own_guest_pages() {
        use crate::backend::vulkan::engine::TargetIdentity;
        use crate::runtime::decode::resource::{PipelineColorAttachment, RenderPipelineDescriptor};

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        map_one_gva_page(&mut host, 4);
        assert!(state.define_task(1, 0x1_0000, 2));

        let pipeline = RenderPipelineDescriptor {
            color_attachments: vec![
                PipelineColorAttachment {
                    slot: 0,
                    ..PipelineColorAttachment::default()
                },
                PipelineColorAttachment {
                    slot: 1,
                    ..PipelineColorAttachment::default()
                },
            ],
            ..RenderPipelineDescriptor::default()
        };
        // Slot 1 is the mask: one guest page at GVA 0x1000, 8 rows of 32 bytes.
        let colors = vec![
            ColorRtRequest {
                slot: 0,
                texture_ref: 10,
                mapping_id: 9,
                width: 8,
                height: 8,
                format: crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM,
                ..ColorRtRequest::default()
            },
            ColorRtRequest {
                slot: 1,
                texture_ref: 11,
                target_gva: 0x1000,
                row_stride: 32,
                width: 8,
                height: 8,
                format: crate::contract::pixel_format::MTL_FORMAT_RG16_FLOAT,
                ..ColorRtRequest::default()
            },
        ];
        // Any identity that is not the secondary's; the function only compares.
        let primary = TargetIdentity::Gva {
            gva: 0xdead_0000,
            width: 8,
            height: 8,
            generation: 0,
        };

        let gen_of = |host: &mut FakeHost| {
            let secs = super::build_secondary_targets(
                &state, host, 1, &colors, &pipeline, &primary, 8, 8, [0.0; 4],
            );
            assert_eq!(secs.len(), 1, "slot 1 is a resolvable secondary");
            match secs[0].identity {
                TargetIdentity::Gva { generation, .. } => generation,
                ref other => panic!("a target_gva secondary must be a Gva identity, got {other:?}"),
            }
        };

        let gen_a = gen_of(&mut host);
        assert_ne!(
            gen_a, 0,
            "a fully walked secondary span must name its allocation"
        );
        assert_eq!(
            gen_of(&mut host),
            gen_a,
            "an unchanged mapping must not mint a second identity"
        );

        // The guest hands GVA 0x1000 to a different page. Same task, same
        // address, same geometry — only the memory changed.
        map_one_gva_page(&mut host, 5);
        assert_ne!(
            gen_of(&mut host),
            gen_a,
            "two allocations at one secondary address must not share one image"
        );
    }
}
